"""
NeuroSRM_MEDI_demo.py
=====================
Multiple-Exposure Differentiable EDI (MEDI) for contrast-threshold calibration.

This script is Step 0 of NeuralSIM Calibration.  It jointly optimises a shared
contrast threshold  c = (c_pos, c_neg)  over several consecutive APS exposure
windows so that the EDI forward model best explains each blurry frame.

Outputs
-------
* output/MEDI/contrast_params.json  -- calibrated c_pos, c_neg
* output/MEDI/c_convergence.png     -- c and loss curves over iterations
* output/MEDI/latent_sequence.png   -- per-exposure-window latent image panel

Design constraints (matching NeuroSRM_demo.py)
----------------------------------------------
* Linear script structure -- physicist-readable, no hidden state.
* Function encapsulation only for reuse; all heavy logic stays in-line.
* Naming conventions follow NeuroSRM_demo.py.
* Academic English throughout.
"""

import json
import math
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.optim import Adam

from utils.utility import load_events, time_window, np_to_torch, watch_tensor


# ---------------------------------------------------------------------------
# Helper: save a grayscale video [T, H, W] to mp4 (learned from mEDI_debug)
# ---------------------------------------------------------------------------

def save_video_mp4(path, video, fps=12):
    """Write a float [T, H, W] video in [0, 1] to an mp4 file."""
    video = np.asarray(video)
    if video.ndim != 3:
        raise ValueError("video must have shape [T, H, W]")
    h, w = video.shape[1], video.shape[2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             float(fps), (w, h), True)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open mp4 writer: {path}")
    for frame in video:
        frame = np.nan_to_num(frame)
        frame_u8 = (np.clip(frame, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        writer.write(cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR))
    writer.release()


def annotate_video_time(video, times_us):
    """Burn the ABSOLUTE event timestamp (ms) into the top-left of each frame.

    Learned from mEDI_debug.annotate_video_time, but shows the absolute time
    (origin = recording start, not the clip start).  `times_us` is the per-frame
    absolute timestamp in microseconds; label reads "t = <ms> ms".
    """
    video = np.asarray(video, dtype=np.float32)
    times_us = np.asarray(times_us, dtype=np.float64)
    if video.ndim != 3 or len(times_us) != video.shape[0]:
        return video
    annotated = np.empty_like(video, dtype=np.float32)
    for i, frame in enumerate(video):
        frame_u8 = (np.clip(frame, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        label = f"t = {float(times_us[i]) / 1000.0:.1f} ms"
        cv2.putText(frame_u8, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, 0, 3, cv2.LINE_AA)     # black outline
        cv2.putText(frame_u8, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, 255, 1, cv2.LINE_AA)   # white text
        annotated[i] = frame_u8.astype(np.float32) / 255.0
    return annotated


# ---------------------------------------------------------------------------
# Helper: event-to-image binning (self-contained, no external dependency)
# ---------------------------------------------------------------------------

def bin_events_pos_neg(ts_w, xs_w, ys_w, ps_w, sensor_size, num_bins, duration,
                       device=None):
    """Bilinearly splat events into temporal bins -> pos/neg count volumes.

    Differentiable bilinear splat, same architecture as
    utils.utility.interpolate_to_image: each event distributes its unit weight
    to the 4 surrounding pixels by sub-pixel bilinear weights (1-dx)(1-dy) etc.,
    so gradients flow back to the event coordinates.  For the current c-only
    calibration the resulting bins are stored as (detached) EDI buffers, but
    keeping the splat differentiable preserves the architecture for the case
    where events are warped by a learnable flow (positions depending on
    parameters).  For integer DAVIS coordinates the bilinear splat reduces
    EXACTLY to nearest-pixel counting, so the calibrated c is unchanged.

    Parameters
    ----------
    ts_w, xs_w, ys_w, ps_w : 1-D array/tensor, events cropped to the window
    sensor_size             : (H, W)
    num_bins                : int    number of temporal bins
    duration                : float  window length [microseconds]
    device                  : torch device (defaults to cuda if available)

    Returns
    -------
    pos_bins, neg_bins : torch.FloatTensor  (num_bins, H, W)
    """
    H, W = sensor_size
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if len(ts_w) == 0:
        z = torch.zeros(num_bins, H, W, device=device)
        return z, z.clone()

    t = torch.as_tensor(np.asarray(ts_w), dtype=torch.float32, device=device)
    x = torch.as_tensor(np.asarray(xs_w), dtype=torch.float32, device=device)
    y = torch.as_tensor(np.asarray(ys_w), dtype=torch.float32, device=device)
    p = torch.as_tensor(np.asarray(ps_w), dtype=torch.float32, device=device)

    # Hard temporal binning (as in mEDI_debug); soft only in the spatial splat.
    bin_idx = torch.clamp((t / max(float(duration), 1.0) * num_bins).long(),
                          0, num_bins - 1)
    x0 = torch.floor(x)
    y0 = torch.floor(y)
    dx = x - x0
    dy = y - y0
    x0 = x0.long()
    y0 = y0.long()

    pos_w = (p > 0).float()
    neg_w = 1.0 - pos_w

    pos_flat = torch.zeros(num_bins * H * W, device=device)
    neg_flat = torch.zeros(num_bins * H * W, device=device)
    for ox, oy, wgt in ((0, 0, (1 - dx) * (1 - dy)),
                        (1, 0, dx * (1 - dy)),
                        (0, 1, (1 - dx) * dy),
                        (1, 1, dx * dy)):
        xi = x0 + ox
        yi = y0 + oy
        valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        flat = (bin_idx * (H * W) + yi * W + xi)[valid]
        pos_flat.index_add_(0, flat, (wgt * pos_w)[valid])
        neg_flat.index_add_(0, flat, (wgt * neg_w)[valid])

    return pos_flat.view(num_bins, H, W), neg_flat.view(num_bins, H, W)


# ---------------------------------------------------------------------------
# Helper: encapsulated event + frame loading for a given exposure window
# ---------------------------------------------------------------------------

def load_event_frame_window(ts, xs, ys, ps, frames_img, time_img, frame_ind):
    """Extract events and the corresponding blurry frame for one APS exposure.

    This function encapsulates the 'how many events / which frame' logic that
    appears inline in NeuroSRM_demo.py so it can be called uniformly for
    multiple consecutive windows during MEDI calibration.

    Parameters
    ----------
    ts, xs, ys, ps  : np.ndarray  full event stream
    frames_img      : np.ndarray  (N_frames, H, W)  normalised to [0, 1]
    time_img        : np.ndarray  (N_frames, 2)  [t_start, t_end] per frame [us]
    frame_ind       : int

    Returns
    -------
    dict with keys:
        'xs_w', 'ys_w', 'ts_w', 'ps_w'  -- events inside the exposure window
        'duration'                        -- window length [us]
        'blurry_np'                       -- (H, W) float32 in [0, 1]
        't_start', 't_end'               -- window boundaries [us]
        'frame_ind'                       -- frame_ind (int)
    Returns None if the window is invalid or has no events.
    """
    if frame_ind < 0 or frame_ind >= len(time_img):
        return None
    t_start, t_end = int(time_img[frame_ind][0]), int(time_img[frame_ind][1])
    duration = t_end - t_start
    if duration <= 0:
        return None

    xs_w, ys_w, ts_w, ps_w = time_window(xs, ys, ts, ps, s=t_start, win=duration)
    if len(ts_w) == 0:
        return None
    ts_w = ts_w - ts_w.min()

    blurry_np = frames_img[frame_ind].astype(np.float32)
    if blurry_np.max() > 2.0:
        blurry_np = blurry_np / 255.0
    blurry_np = np.clip(blurry_np, 0.0, 1.0)

    return {
        'xs_w': xs_w, 'ys_w': ys_w, 'ts_w': ts_w, 'ps_w': ps_w,
        'duration': float(duration),
        'blurry_np': blurry_np,
        't_start': t_start, 't_end': t_end,
        'frame_ind': int(frame_ind),
    }


# ---------------------------------------------------------------------------
# EDI forward model (differentiable, self-contained)
# ---------------------------------------------------------------------------

# --- bounded-softplus contrast parameterisation (faithful to DiffEDI) --------
# c is kept in a physical range [C_MIN, C_MAX] via a smooth, invertible map so
# the optimiser never drives it to 0 or blows it up.
C_MIN, C_MAX = 1e-3, 0.5


def bounded_softplus(raw, c_min=C_MIN, c_max=C_MAX):
    positive = F.softplus(raw)
    normalized = positive / (1.0 + positive)
    return c_min + (c_max - c_min) * normalized


def raw_from_bounded_softplus(value, c_min=C_MIN, c_max=C_MAX):
    """Inverse of bounded_softplus so we can initialise raw_c from a target c."""
    v = min(max(float(value), c_min + 1e-6), c_max - 1e-6)
    normalized = (v - c_min) / (c_max - c_min)
    positive = normalized / (1.0 - normalized)
    if positive > 20.0:
        return torch.tensor(float(positive))
    return torch.tensor(float(math.log(math.expm1(positive))))


def edi_centered_double_integral(bii):
    """Centered double integral of brightness increments (MEDI core).

    Input  : bii  (2N, H, W)  per-bin brightness increments c_pos*E+ - c_neg*E-.
    Output : E    (2N+1, H, W)  log-intensity change RELATIVE TO THE EXPOSURE
             CENTRE (t = T/2).  Frame N is the reference (all-zero); frames to
             the left integrate backward (negative), frames to the right forward.

    This is the "double integration" that gives MEDI its name and places the
    latent reference at the middle of the exposure (T0 = 1/2 window), which is
    numerically better conditioned than integrating from t=0.
    """
    two_n = bii.shape[0]
    assert two_n % 2 == 0, "num_bins must be even for the centered double integral"
    n = two_n // 2
    left = [-bii[i:n].sum(dim=0) for i in range(n)]          # t < centre
    center = [torch.zeros_like(bii[0])]                      # t = centre
    right = [bii[n:n + 1 + i].sum(dim=0) for i in range(n)]  # t > centre
    return torch.stack(left + center + right, dim=0)         # (2N+1, H, W)


class ExposureEDI(nn.Module):
    """Differentiable MEDI model for one APS exposure window.

    Faithful re-implementation of DiffEDI/differentiable_edi.DifferentiableEDI:

        E(t)  = centered double integral of  c_pos*E+(t) - c_neg*E-(t)
        I(t)  = I_center * exp(E(t))
        I_blur = mean_t I(t)
      => I_center = I_blur / mean_t exp(E(t))       (EDI inversion)

    The reference latent I_center sits at the MIDDLE of the exposure.  With
    shared_contrast=True a single threshold c is used (c_pos == c_neg), matching
    the paper's global-c simplification.

    Parameters
    ----------
    pos_bins, neg_bins  : (2N, H, W)  positive / negative event count tensors
    init_c_pos/neg      : float  initial contrast threshold(s)
    shared_contrast     : bool   if True, one shared c (c_pos == c_neg)
    eps                 : float  numerical floor in the EDI inversion
    """

    def __init__(self, pos_bins, neg_bins, init_c_pos=0.2, init_c_neg=0.2,
                 shared_contrast=True, edge_alpha=1.0, eps=1e-8):
        super().__init__()
        self.shared_contrast = bool(shared_contrast)
        self.edge_alpha = float(edge_alpha)
        self.eps = eps
        if self.shared_contrast:
            init_c = 0.5 * (float(init_c_pos) + float(init_c_neg))
            self.raw_c = nn.Parameter(raw_from_bounded_softplus(init_c))
        else:
            self.raw_c_pos = nn.Parameter(raw_from_bounded_softplus(init_c_pos))
            self.raw_c_neg = nn.Parameter(raw_from_bounded_softplus(init_c_neg))
        # Fixed event bins -- buffers (no gradient).
        self.register_buffer('pos_bins', pos_bins.float())
        self.register_buffer('neg_bins', neg_bins.float())

    @property
    def c_pos(self):
        return bounded_softplus(self.raw_c if self.shared_contrast else self.raw_c_pos)

    @property
    def c_neg(self):
        return bounded_softplus(self.raw_c if self.shared_contrast else self.raw_c_neg)

    def event_integral(self):
        """Centered double integral E(t): (2N+1, H, W)."""
        bii = self.c_pos * self.pos_bins - self.c_neg * self.neg_bins
        return edi_centered_double_integral(bii)

    def event_edge_reference(self):
        """Soft-assignment event-edge reference used by the Sobel edge loss.

        Each of the (2N+1) video frames gets a temporally-weighted sum of the
        (2N) event-count bins (weight decays with |frame_idx - event_idx|), so
        the edge loss aligns image gradients with where events actually fired.
        Faithful to DifferentiableEDI.event_edge_reference.
        """
        event_counts = self.pos_bins + self.neg_bins          # (2N, H, W)
        num_event_bins = event_counts.shape[0]
        num_video_frames = num_event_bins + 1
        frame_idx = torch.arange(num_video_frames, device=event_counts.device,
                                 dtype=event_counts.dtype)
        event_idx = torch.arange(num_event_bins, device=event_counts.device,
                                 dtype=event_counts.dtype) + 0.5
        weights = torch.exp(-self.edge_alpha *
                            torch.abs(frame_idx[:, None] - event_idx[None, :]))
        return torch.einsum("tb,bhw->thw", weights, event_counts)

    def forward(self, blurry):
        """Return (I_center, video, blur_pred).

        I_center   : (H, W)  sharp latent at exposure CENTRE (t = T/2)
        video      : (2N+1, H, W)  latent evolution over the exposure
        blur_pred  : (H, W)  reprojected blurry frame (should approx blurry)
        """
        integral = self.event_integral().clamp(-20.0, 20.0)   # (2N+1, H, W)
        exp_integral = torch.exp(integral)                    # intensity ratio at bin edges
        # OPTIMISATION PATH -- ALWAYS the plain "rect" mean of exp(E). Keeping
        # this simple is deliberate: I_center = blurry / mean(exp) is a clean
        # closed form that transports cleanly through the multi-frame transfer
        # loss (which is what actually fixes c). Any intensity smoothing /
        # interpolation (the "trap" idea) belongs ONLY to the separate render
        # path, applied after c is fixed -- never here, because an interpolated
        # I_center is harder to propagate through transfer.
        I_center = (blurry / (exp_integral.mean(dim=0) + self.eps)).clamp_min(0.0)
        video = I_center.unsqueeze(0) * exp_integral
        blur_pred = video.mean(dim=0)                         # plain mean == blurry
        return I_center, video, blur_pred


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def sqrt_fidelity_loss(blur_pred, blurry):
    """Sqrt-domain L2 fidelity (matches the frame-consistency loss in demo)."""
    return torch.mean(
        (torch.sqrt(blur_pred.clamp(min=0.0) + 1e-8) -
         torch.sqrt(blurry.clamp(min=0.0) + 1e-8)) ** 2
    )


# --- mEDI regularisers (Pan et al., CVPR 2019 "Bringing a Blurry Frame Alive
#     at High Frame Rate with an Event Camera").  Faithful to DiffEDI. ----------

def total_variation(image, eps=1e-6):
    """Isotropic spatial TV over a (H, W) or (T, H, W) tensor."""
    if image.ndim == 2:
        image = image.unsqueeze(0)
    dy = image[..., 1:, :] - image[..., :-1, :]
    dx = image[..., :, 1:] - image[..., :, :-1]
    return torch.sqrt(dy.square() + eps).mean() + torch.sqrt(dx.square() + eps).mean()


def temporal_variation(video, eps=1e-6):
    """Frame-to-frame TV over the reconstructed EDI stack (T, H, W)."""
    dt = video[1:] - video[:-1]
    return torch.sqrt(dt.square() + eps).mean()


def sobel_magnitude(image, eps=1e-6):
    """Sobel edge magnitude for (H, W) or (T, H, W) tensors."""
    squeeze_time = image.ndim == 2
    if squeeze_time:
        image = image.unsqueeze(0)
    kx = image.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
    ky = image.new_tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
    x = image.unsqueeze(1)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    edge = torch.sqrt(gx.square() + gy.square() + eps).squeeze(1)
    return edge[0] if squeeze_time else edge


def edge_correlation_loss(video, event_edge_reference):
    """Negative Sobel cross-correlation between the reconstruction and the
    event-edge reference (Pan et al. CVPR 2019): drives image edges to align
    with where events fired."""
    image_edges = sobel_magnitude(video)
    event_edges = sobel_magnitude(event_edge_reference).detach()
    return -(image_edges * event_edges).mean()


# ---------------------------------------------------------------------------
# Continuous (sliding-window) rendering across exposures AND blind gaps
# ---------------------------------------------------------------------------

def render_blind_bridge(start_latent, end_latent, pos_blind, neg_blind, c_pos, c_neg):
    """Bridge a blind (inter-exposure) gap by overlap-blending two integrations.

    The APS exposure covers only part of each frame period; between expo_end[k]
    and expo_start[k+1] the sensor is blind but events keep flowing.  We fill the
    gap from BOTH sides and cross-fade (sliding-window overlap):

        forward(t)  = start_latent * exp( E(0 -> t) )        [from window k end]
        backward(t) = end_latent   * exp( E(0 -> t) - E_tot) [from window k+1 start]
        bridge(t)   = (1 - w_t) * forward(t) + w_t * backward(t),  w_t: 0 -> 1

    At the gap start w=0 => forward => equals window k's last frame; at the gap
    end w=1 => backward => equals window k+1's first frame.  Both junctions are
    therefore continuous BY CONSTRUCTION, so no residual step remains -- unlike a
    one-sided forward integration whose endpoint need not match the next window's
    independent solution.

    pos_blind, neg_blind : (Bb, H, W) event count bins spanning the blind gap.
    Returns              : (Bb, H, W) blended latent frames across the gap.
    """
    bii = c_pos * pos_blind - c_neg * neg_blind
    E = torch.cumsum(bii, dim=0).clamp(-20.0, 20.0)          # (Bb, H, W)
    E_total = E[-1:]                                          # (1, H, W) full-gap integral
    forward = start_latent.unsqueeze(0) * torch.exp(E)
    backward = end_latent.unsqueeze(0) * torch.exp(E - E_total)
    bb = bii.shape[0]
    # Interior cross-fade weights (avoid exact 0/1 so both ends stay live).
    w = torch.linspace(0.0, 1.0, bb + 2, device=bii.device)[1:-1].view(bb, 1, 1)
    return (1.0 - w) * forward + w * backward


def build_continuous_video(results, blind_pairs, c_pos, c_neg):
    """Concatenate per-exposure latent stacks and the blind bridges between them
    into one gap-free, step-free video.  Each bridge cross-fades from the
    previous window's last frame to the next window's first frame, so every
    boundary is continuous by construction."""
    chunks = []
    for k, (I_center, video, blur_pred) in enumerate(results):
        chunks.append(video)                              # exposure stack
        if k < len(results) - 1 and blind_pairs[k] is not None:
            bridge = render_blind_bridge(
                video[-1], results[k + 1][1][0],          # window k end, k+1 start
                blind_pairs[k]['pos'], blind_pairs[k]['neg'],
                c_pos, c_neg)
            chunks.append(bridge)
    return torch.cat(chunks, dim=0)


def temporal_smooth_render(video, times, factor):
    """RENDER-only: linearly interpolate a (T, H, W) video (and its per-frame
    timestamps) to (T-1)*factor+1 frames.

    This is the "trap" render -- intensity ramps linearly between the optimised
    samples, giving a smoother high-frame-rate video.  It is decoupled from the
    (rect) optimisation: c is already fixed, this only grows the intermediate
    latent frames for playback.  Pure numpy (render runs on CPU arrays).
    """
    T = video.shape[0]
    if factor <= 1 or T < 2:
        return video, times
    out_len = (T - 1) * int(factor) + 1
    src = np.linspace(0.0, 1.0, T)
    dst = np.linspace(0.0, 1.0, out_len)
    pos = np.interp(dst, src, np.arange(T, dtype=np.float64))
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, T - 1)
    w = (pos - lo).astype(np.float32)[:, None, None]
    out = (1.0 - w) * video[lo] + w * video[hi]
    out_times = np.interp(dst, src, times)
    return out.astype(np.float32), out_times


def continuous_timestamps(items, blind_pairs, num_bins):
    """Absolute per-frame timestamps [us] matching build_continuous_video."""
    times = []
    for k, item in enumerate(items):
        times.extend(np.linspace(item['t_start'], item['t_end'],
                                 num_bins + 1).tolist())
        if k < len(items) - 1 and blind_pairs[k] is not None:
            bp = blind_pairs[k]
            times.extend(np.linspace(bp['t_start'], bp['t_end'],
                                     bp['bb'] + 1)[1:].tolist())
    return np.asarray(times, dtype=np.float64)


def temporal_transfer_loss(I_ref_src, I_ref_dst, pos_transfer, neg_transfer, c_pos, c_neg):
    """Symmetric consistency loss: transport I_ref_src to the next window via the
    inter-window event integral and match it against I_ref_dst.

    This term is what actually determines c -- the sqrt-fidelity term is
    degenerate (blur_pred == blurry by construction of the EDI inversion, for
    ANY c), so it carries no information about c.  c is set here, by requiring
    that the SAME c both explains each frame's blur and correctly transports the
    latent between frame centres.

    IMPORTANT: I_ref_dst must NOT be detached.  Both endpoints depend on c, so
    detaching the destination gives a biased (wrong) gradient whose stationary
    point drifts ~1.7% above the true minimum.  Faithful to DiffEDI/mEDI_debug,
    which keeps both endpoints live.
    """
    integral = (c_pos * pos_transfer - c_neg * neg_transfer).clamp(-15.0, 15.0)
    I_transported = I_ref_src * torch.exp(integral)
    return torch.mean((I_transported - I_ref_dst) ** 2)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def make_latent_panel(items, results, sensor_size, max_cols=5):
    """Assemble a panel of (blurry | I_ref | blur_pred) triplets."""
    rows = []
    for item, (I_ref, video, blur_pred) in zip(items, results):
        blurry = item['blurry_t'].detach().cpu().numpy()
        I_r = I_ref.detach().cpu().clamp(0, 1).numpy()
        b_p = blur_pred.detach().cpu().clamp(0, 1).numpy()
        # Normalise IWE-like quantities to [0, 1] for display.
        rows.append(np.concatenate([blurry, I_r, b_p], axis=1))
    panel = np.concatenate(rows[:max_cols], axis=0)
    return panel


# ---------------------------------------------------------------------------
# Main script (linear flow -- easy to step through in a debugger)
# ---------------------------------------------------------------------------

sensor_size = (260, 346)
Ny, Nx = sensor_size
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
eve_dtype = "aedat4"

### Step 1: Data path and time-window configuration
#read_path_e = r"E:\davis\hk\hk.aedat4"
#read_path_e = r"E:\davis\pei\onion.aedat4"
#read_path_e = r"E:\davis\pei\01.aedat4"
read_path_e = r"D:/BaiduNetdiskDownload/Rotation/02_rot.aedat4"
#read_path_e  = r"E:\[X] NeuroSR\event\lunchong.aedat4"
#read_path_e = r"E:\davis\shuo\01_rotation\01.aedat4"
save_dir = "output/MEDI"
os.makedirs(save_dir, exist_ok=True)

# MEDI calibration hyper-parameters (faithful to DiffEDI/mEDI_debug defaults)
# Optimisation ALWAYS uses the simple rect EDI (I_center = mean(exp E)) -- that
# is what fixes c via the multi-frame transfer constraint.  render_mode ONLY
# controls how the final high-frame-rate video grows the intermediate frames,
# and is completely decoupled from the optimisation:
#   "rect" : play the optimised bin-edge frames directly (step transitions).
#   "trap" : linearly interpolate intensity between the optimised frames for a
#            smoother, higher-frame-rate video (the Ding'25 / Flow-EDI idea, but
#            purely at render time).
render_mode    = "rect"   # "rect" or "trap"  (render only, does NOT affect c)
render_upsample = 0       # trap: intensity-interpolation factor for the video
num_bins       = 32    # temporal bins per APS exposure window (MUST be even for double integral)
calib_n_frames = 5     # number of consecutive APS frames to calibrate over (max 30)
calib_start_frame = 0  # first APS frame index (0 = auto-select by exposure-window density)
calib_max_frames = 30  # hard upper limit: do not use more than 30 consecutive frames
shared_contrast = True # single shared threshold c (c_pos == c_neg); the paper's global-c model
transfer_weight = 1.0  # weight for the inter-window temporal-transfer consistency loss
# mEDI regularisers (Pan et al. CVPR 2019), same weights as DiffEDI/mEDI_debug:
tv_weight          = 1e-3  # spatial TV on the latent stack
temporal_tv_weight = 1e-4  # temporal TV between consecutive latent frames
edge_weight        = 1e-4  # Sobel edge cross-correlation with the event edges
steps          = 2000  # optimisation iterations
lr_c           = 4e-3  # learning rate for c (matches the mEDI_debug run that gave c=0.1913 on 02)
lr_min         = 1e-5
plateau_check_every = 300   # iterations between LR plateau checks
plateau_segment_len = 100
plateau_delta       = 1e-5  # relative flatness criterion

assert num_bins % 2 == 0, "num_bins must be even for the centered double integral"

### Step 2: Load the full event stream + frames
print("[MEDI] loading events:", read_path_e)
ts, xs, ys, ps, frames_img3, time_img = load_events(eve_dtype="aedat4", path=read_path_e)
ts -= ts.min()

if len(frames_img3.shape) == 4:
    frames_img = np.zeros(frames_img3.shape[:3], dtype=np.float32)
    for _i in range(frames_img3.shape[0]):
        frames_img[_i] = cv2.cvtColor(frames_img3[_i], cv2.COLOR_BGR2GRAY)
elif len(frames_img3.shape) == 3:
    frames_img = frames_img3.astype(np.float32)
else:
    raise ValueError(f"Unexpected frames shape: {frames_img3.shape}")

print(f"[MEDI] events={len(ts)}, frames={len(frames_img)}, "
      f"t_range=[{int(ts.min())}, {int(ts.max())}] us")

### Step 3: Auto-select calibration start frame by exposure-window event density.
#
# Faithful to DiffEDI/mEDI_debug.choose_calib_start_frame: for every APS frame
# we measure the event density over its OWN exposure window
# time_img[fi] = [expo_s, expo_e].  Fast motion during the exposure produces
# BOTH many events AND motion blur, so exposure-window density is a direct blur
# proxy (empirically corr(density, -sharpness) ~= 0.93 on 02.aedat4).  We pick
# the consecutive block of `calib_n_frames` frames with the HIGHEST mean density
# (argmax over blocks -- NOT the first block passing a threshold, which would
# bias toward early, still-sharp segments where motion has barely started).

n_frames_total = len(time_img)
frame_count   = np.zeros(n_frames_total, dtype=np.int64)
frame_density = np.zeros(n_frames_total, dtype=np.float64)
for _fi in range(n_frames_total):
    _t_s, _t_e = int(time_img[_fi][0]), int(time_img[_fi][1])
    _win = _t_e - _t_s
    if _win <= 0:
        continue
    _cnt = int(np.count_nonzero((ts >= _t_s) & (ts < _t_e)))
    frame_count[_fi]   = _cnt
    frame_density[_fi] = _cnt / float(_win)

# Clamp calib_n_frames to the hard upper limit.
calib_n_frames = min(int(calib_n_frames), int(calib_max_frames))

if calib_start_frame == 0:
    best_start, best_score = 0, -1.0
    for _st in range(0, n_frames_total - calib_n_frames + 1):
        _c = frame_count[_st:_st + calib_n_frames]
        _d = frame_density[_st:_st + calib_n_frames]
        if _c.min() <= 0:          # skip blocks with any empty exposure window
            continue
        _score = float(_d.mean())
        if _score > best_score:
            best_score, best_start = _score, _st
    calib_start_frame = best_start
    print(f"[MEDI] auto calib_start_frame={calib_start_frame}  "
          f"mean_density={best_score:.4f} ev/us  "
          f"frames={list(range(calib_start_frame, calib_start_frame + calib_n_frames))}  "
          f"t={time_img[calib_start_frame][0]/1e6:.3f} s")

# --- diagnostic plot: exposure-window density + sharpness.
#     Sharpness (Var of Laplacian, low = blurred) is shown as an INDEPENDENT
#     confirmation only; it is NOT used to drive the selection.
frame_sharpness = np.zeros(n_frames_total, dtype=np.float64)
for _fi in range(n_frames_total):
    _f = frames_img[_fi].astype(np.float32)
    if _f.max() > 2.0:
        _f = _f / 255.0
    frame_sharpness[_fi] = float(cv2.Laplacian(_f * 255.0, cv2.CV_32F).var())

_fig_diag, (_ax_den, _ax_shp) = plt.subplots(
    2, 1, figsize=(14, 5), sharex=True, num="MEDI_frame_selection")
_ax_den.plot(frame_density, color='darkorange', linewidth=0.8, marker='.')
_ax_den.axvspan(calib_start_frame, calib_start_frame + calib_n_frames - 1,
                alpha=0.25, color='green',
                label=f'selected frames [{calib_start_frame}, '
                      f'{calib_start_frame + calib_n_frames - 1}]  '
                      f'(t={time_img[calib_start_frame][0]/1e6:.3f} s)')
_ax_den.set_ylabel("exposure density\n[ev/us]"); _ax_den.legend(fontsize=8)
_ax_den.set_title("Per-APS-frame selection: exposure-window density (used) "
                  "vs sharpness (confirmation)")
_ax_den.grid(True, alpha=0.25)

_ax_shp.plot(frame_sharpness, color='steelblue', linewidth=0.8, marker='.')
_ax_shp.axvspan(calib_start_frame, calib_start_frame + calib_n_frames - 1,
                alpha=0.25, color='green')
_ax_shp.set_xlabel("APS frame index")
_ax_shp.set_ylabel("sharpness\nVar(Laplacian)  (low = blurred)")
_ax_shp.grid(True, alpha=0.25)
_fig_diag.tight_layout()
_fig_diag.savefig(os.path.join(save_dir, "frame_density_profile.png"), dpi=150)
plt.pause(0.01)

print(f"[MEDI] calibration: start_frame={calib_start_frame}, "
      f"n_frames={calib_n_frames}")

### Step 4: Build per-frame EDI items over each frame's TRUE APS exposure window.
#
# The EDI forward model requires that the event integral covers EXACTLY the
# exposure interval that produced the blur: I_blur = (1/T) integral_0^T I(t) dt
# with T = expo_e - expo_s.  We therefore bin events over time_img[fi] (the
# real per-frame exposure), NOT a fixed-length window -- otherwise the event
# integral and the blur amount are mismatched and I_ref ~= the blurry input.
print("[MEDI] building exposure EDI models ...")

items = []
for _fi in range(calib_start_frame, calib_start_frame + calib_n_frames):
    if _fi >= n_frames_total:
        break
    _t_s, _t_e = int(time_img[_fi][0]), int(time_img[_fi][1])
    _dur = float(_t_e - _t_s)
    if _dur <= 0:
        print(f"[MEDI] skip frame {_fi}: invalid exposure [{_t_s}, {_t_e}]")
        continue

    # Events inside this frame's exposure window.
    _xs_w, _ys_w, _ts_w, _ps_w = time_window(xs, ys, ts, ps, s=_t_s, win=int(_dur))
    if len(_ts_w) == 0:
        print(f"[MEDI] skip frame {_fi}: no events in exposure")
        continue
    _ts_w = _ts_w - _ts_w.min()

    blurry_np = frames_img[_fi].astype(np.float32)
    if blurry_np.max() > 2.0:
        blurry_np = blurry_np / 255.0
    blurry_np = np.clip(blurry_np, 0.0, 1.0)

    pos_bins, neg_bins = bin_events_pos_neg(
        _ts_w, _xs_w, _ys_w, _ps_w, sensor_size, num_bins, _dur)
    model = ExposureEDI(pos_bins, neg_bins, init_c_pos=0.2, init_c_neg=0.2,
                        shared_contrast=shared_contrast).to(device)
    blurry_t = torch.from_numpy(blurry_np).to(device).float()
    items.append({
        'xs_w': _xs_w, 'ys_w': _ys_w, 'ts_w': _ts_w, 'ps_w': _ps_w,
        'duration': _dur,
        'blurry_np': blurry_np,
        'blurry_t': blurry_t,
        't_start': _t_s, 't_end': _t_e,
        'frame_ind': _fi,
        'pos_bins': pos_bins,
        'neg_bins': neg_bins,
        'model': model,
    })
    print(f"[MEDI] frame {_fi}: exposure=[{_t_s}, {_t_e}] us, "
          f"dur={_dur:.0f} us, events={len(_ts_w)}")

if not items:
    raise RuntimeError("[MEDI] No valid calibration windows collected.")

### Step 5: Share the contrast threshold across all windows by linking parameters.
# Every window's EDI model points to the SAME raw-c tensor(s) of the first model,
# so a single c (or c_pos/c_neg pair) is jointly calibrated over the whole block.
if shared_contrast:
    shared_raw = items[0]['model'].raw_c
    for _item in items[1:]:
        _item['model'].raw_c = shared_raw
    calib_params = [shared_raw]
else:
    shared_raw_c_pos = items[0]['model'].raw_c_pos
    shared_raw_c_neg = items[0]['model'].raw_c_neg
    for _item in items[1:]:
        _item['model'].raw_c_pos = shared_raw_c_pos
        _item['model'].raw_c_neg = shared_raw_c_neg
    calib_params = [shared_raw_c_pos, shared_raw_c_neg]

### Step 6: Pre-compute inter-window event bins for the transfer loss.
# For consecutive windows i and i+1 we integrate events between their centre
# timestamps so that I_ref[i] can be transported to match I_ref[i+1].
transfer_pairs = []
for _k in range(len(items) - 1):
    src, dst = items[_k], items[_k + 1]
    t_s_mid = int(0.5 * (src['t_start'] + src['t_end']))
    t_d_mid = int(0.5 * (dst['t_start'] + dst['t_end']))
    _dur = abs(t_d_mid - t_s_mid)
    if _dur <= 0:
        transfer_pairs.append(None)
        continue
    _xs_t, _ys_t, _ts_t, _ps_t = time_window(
        xs, ys, ts, ps, s=min(t_s_mid, t_d_mid), win=_dur)
    if len(_ts_t) == 0:
        transfer_pairs.append(None)
        continue
    _ts_t = _ts_t - _ts_t.min()
    _pb, _nb = bin_events_pos_neg(_ts_t, _xs_t, _ys_t, _ps_t, sensor_size, 1, _dur)
    transfer_pairs.append({
        'pos': _pb[0].to(device),
        'neg': _nb[0].to(device),
    })
    print(f"[MEDI] transfer pair {_k}->{_k+1}: {len(_ts_t)} events, dur={_dur} us")

### Step 6b: Pre-compute blind-region event bins for continuous rendering.
# The blind gap for pair (k, k+1) is [expo_end[k], expo_start[k+1]] -- the dead
# time between exposures.  We bin its events at the same temporal cadence as the
# exposure stack (expo_dt = exposure_len / num_bins) so playback frame rate is
# uniform across exposures and gaps.
blind_pairs = []
for _k in range(len(items) - 1):
    _t_end_k  = items[_k]['t_end']
    _t_start_n = items[_k + 1]['t_start']
    _blind_dur = _t_start_n - _t_end_k
    if _blind_dur <= 0:
        blind_pairs.append(None)
        continue
    _expo_dt = items[_k]['duration'] / num_bins             # us per output frame
    _bb = max(1, int(round(_blind_dur / max(_expo_dt, 1.0))))
    _xs_b, _ys_b, _ts_b, _ps_b = time_window(
        xs, ys, ts, ps, s=int(_t_end_k), win=int(_blind_dur))
    if len(_ts_b) == 0:
        blind_pairs.append(None)
        continue
    _ts_b = _ts_b - _ts_b.min()
    _pbb, _nbb = bin_events_pos_neg(_ts_b, _xs_b, _ys_b, _ps_b,
                                    sensor_size, _bb, _blind_dur)
    blind_pairs.append({
        'pos': _pbb.to(device), 'neg': _nbb.to(device),
        'bb': _bb, 't_start': int(_t_end_k), 't_end': int(_t_start_n),
    })
    print(f"[MEDI] blind gap {_k}->{_k+1}: dur={_blind_dur} us, "
          f"{_bb} frames, events={len(_ts_b)}")

### Step 7: Optimisation
optimizer = Adam(calib_params, lr=lr_c)
loss_history    = []
c_pos_history   = []
c_neg_history   = []

print("[MEDI] starting optimisation ...")
for _it in range(steps):
    optimizer.zero_grad()

    # Forward pass: per-window EDI
    results = [item['model'](item['blurry_t']) for item in items]
    # I_center[k], video[k], blur_pred[k] = results[k]

    c_pos = items[0]['model'].c_pos
    c_neg = items[0]['model'].c_neg

    # Per-window data terms: sqrt-fidelity + edge (event-aligned) on each
    # exposure stack (Pan et al. CVPR 2019).
    per_window_loss = []
    for r, item in zip(results, items):
        I_center, video, blur_pred = r
        term = sqrt_fidelity_loss(blur_pred, item['blurry_t'])
        if edge_weight > 0:
            term = term + edge_weight * edge_correlation_loss(
                video, item['model'].event_edge_reference())
        per_window_loss.append(term)
    loss = torch.stack(per_window_loss).mean()

    # Spatial + temporal TV on the CONTINUOUS video (exposures + blind bridges).
    # Applying TV over the blind gaps keeps their reconstruction as well-
    # conditioned as the exposure frames, and the temporal TV across the
    # window<->gap boundaries directly penalises playback steps.
    cont_video = build_continuous_video(results, blind_pairs, c_pos, c_neg)
    if tv_weight > 0:
        loss = loss + tv_weight * total_variation(cont_video)
    if temporal_tv_weight > 0:
        loss = loss + temporal_tv_weight * temporal_variation(cont_video)

    # Temporal transfer consistency
    if transfer_weight > 0:
        transfer_terms = []
        for _k, pair in enumerate(transfer_pairs):
            if pair is None:
                continue
            transfer_terms.append(
                temporal_transfer_loss(
                    results[_k][0], results[_k + 1][0],
                    pair['pos'], pair['neg'],
                    c_pos, c_neg))
        if transfer_terms:
            loss = loss + transfer_weight * torch.stack(transfer_terms).mean()

    loss.backward()
    optimizer.step()

    c_pos_val = float(items[0]['model'].c_pos.detach().cpu())
    c_neg_val = float(items[0]['model'].c_neg.detach().cpu())
    loss_history.append(float(loss.detach().cpu()))
    c_pos_history.append(c_pos_val)
    c_neg_history.append(c_neg_val)

    # Adaptive LR: reduce when loss plateau is detected.
    if ((_it + 1) % plateau_check_every == 0
            and optimizer.param_groups[0]['lr'] > lr_min):
        seg = plateau_segment_len
        if len(loss_history) >= 3 * seg:
            recent = np.asarray(loss_history[-3 * seg:])
            seg_means = [recent[i * seg:(i + 1) * seg].mean() for i in range(3)]
            if max(abs(seg_means[1] - seg_means[0]),
                   abs(seg_means[2] - seg_means[1])) < plateau_delta:
                old_lr = optimizer.param_groups[0]['lr']
                new_lr = max(old_lr * 0.5, lr_min)
                for _g in optimizer.param_groups:
                    _g['lr'] = new_lr
                print(f"[MEDI] iter={_it} LR plateau: {old_lr:.2e} -> {new_lr:.2e}")

    if _it % 100 == 0 or _it == steps - 1:
        print(f"[MEDI] iter={_it:4d}  loss={loss_history[-1]:.4e}  "
              f"c_pos={c_pos_val:.5f}  c_neg={c_neg_val:.5f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

### Step 8: Extract calibrated contrast thresholds
c_pos_calibrated = float(items[0]['model'].c_pos.detach().cpu())
c_neg_calibrated = float(items[0]['model'].c_neg.detach().cpu())
print(f"\n[MEDI] calibrated c_pos={c_pos_calibrated:.6f}, "
      f"c_neg={c_neg_calibrated:.6f}")

### Step 9: Save calibrated C to JSON
contrast_params = {
    'c_pos': c_pos_calibrated,
    'c_neg': c_neg_calibrated,
    'c_mean': 0.5 * (c_pos_calibrated + c_neg_calibrated),
    'calib_start_frame': calib_start_frame,
    'calib_n_frames': calib_n_frames,
    'calib_start_time': int(time_img[calib_start_frame][0]),
    'loss_final': loss_history[-1] if loss_history else None,
    'data_path': read_path_e,
}
contrast_path = os.path.join(save_dir, "contrast_params.json")
with open(contrast_path, 'w') as _f:
    json.dump(contrast_params, _f, indent=2)
print(f"[MEDI] saved contrast params -> {contrast_path}")

### Step 10: Visualise convergence curves
fig, (ax_c, ax_loss) = plt.subplots(1, 2, figsize=(12, 4), num="MEDI_convergence")
iters = np.arange(len(c_pos_history))
ax_c.plot(iters, c_pos_history, label='c_pos')
ax_c.plot(iters, c_neg_history, '--', label='c_neg')
ax_c.axhline(c_pos_calibrated, color='C0', linestyle=':', alpha=0.5)
ax_c.axhline(c_neg_calibrated, color='C1', linestyle=':', alpha=0.5)
ax_c.set_xlabel("iteration")
ax_c.set_ylabel("contrast threshold c")
ax_c.set_title(f"Calibrated c_pos={c_pos_calibrated:.4f}, c_neg={c_neg_calibrated:.4f}")
ax_c.legend()
ax_c.grid(True, alpha=0.25)

ax_loss.semilogy(iters, loss_history, label='total loss')
ax_loss.set_xlabel("iteration")
ax_loss.set_ylabel("loss (log scale)")
ax_loss.set_title("MEDI calibration loss")
ax_loss.legend()
ax_loss.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(save_dir, "c_convergence.png"), dpi=150)
print(f"[MEDI] saved convergence figure -> {os.path.join(save_dir, 'c_convergence.png')}")

### Step 11: Visualise the reconstructed latent sequence (no-grad forward pass)
with torch.no_grad():
    final_results = [item['model'](item['blurry_t']) for item in items]

n_items = len(items)
fig2, axes = plt.subplots(n_items, 3, figsize=(9, 3 * n_items),
                          num="MEDI_latent_sequence")
col_titles = ["Blurry frame (input)", "I_center (sharp, t=T/2)", "Blur reprojection"]
for _col, title in enumerate(col_titles):
    axes[0, _col].set_title(title, fontsize=10)

for _row, (item, (I_ref, video, blur_pred)) in enumerate(zip(items, final_results)):
    blurry_np  = item['blurry_t'].cpu().numpy()
    I_ref_np   = I_ref.cpu().clamp(0, 1).numpy()
    blur_pr_np = blur_pred.cpu().clamp(0, 1).numpy()
    fi = item['frame_ind']

    axes[_row, 0].imshow(blurry_np, cmap='gray', vmin=0, vmax=1)
    axes[_row, 0].set_ylabel(f"frame {fi}", fontsize=8)
    axes[_row, 1].imshow(I_ref_np, cmap='gray', vmin=0, vmax=1)
    axes[_row, 2].imshow(blur_pr_np, cmap='gray', vmin=0, vmax=1)
    for _ax in axes[_row]:
        _ax.axis('off')

fig2.suptitle(f"MEDI latent sequence  (c_pos={c_pos_calibrated:.4f}, "
              f"c_neg={c_neg_calibrated:.4f})", fontsize=11)
fig2.tight_layout()
fig2.savefig(os.path.join(save_dir, "latent_sequence.png"), dpi=150)
print(f"[MEDI] saved latent sequence -> {os.path.join(save_dir, 'latent_sequence.png')}")

### Step 12: Save the CONTINUOUS (sliding-window) latent video as mp4.
# The video now moves seamlessly through the blind inter-exposure gaps -- no
# more per-window cut every (num_bins+1) frames.  Layout matches mEDI_debug:
# MEDI latent on the LEFT, blurry APS frame on the RIGHT, absolute event
# timestamp burned into the top-left over the latent panel.
with torch.no_grad():
    cont_render = build_continuous_video(
        final_results, blind_pairs,
        items[0]['model'].c_pos, items[0]['model'].c_neg)
long_video = cont_render.clamp(0, 1).cpu().numpy()               # (T, H, W)
render_times_np = continuous_timestamps(items, blind_pairs, num_bins)

# Right-hand comparison frame per continuous frame: hold the APS blurry frame of
# the current exposure window (and keep holding it across the following gap).
frame_seq = []
for _k, item in enumerate(items):
    frame_seq.extend([item['blurry_np']] * (num_bins + 1))       # exposure stack
    if _k < len(items) - 1 and blind_pairs[_k] is not None:
        frame_seq.extend([item['blurry_np']] * blind_pairs[_k]['bb'])  # blind gap
frame_seq = np.stack(frame_seq, axis=0)                          # (T, H, W)

assert frame_seq.shape[0] == long_video.shape[0] == len(render_times_np), \
    "continuous video / frame / timestamp length mismatch"

long_comparison = np.concatenate([long_video, frame_seq], axis=2)   # (T, H, 2W)

# RENDER path (decoupled from optimisation): the optimised c is already fixed;
# render_mode only chooses how the intermediate frames are grown for playback.
#   "rect" -> play the optimised frames directly (step transitions);
#   "trap" -> linearly interpolate intensity to a smoother, higher frame rate.
if render_mode == "trap":
    long_video, render_times_up = temporal_smooth_render(
        long_video, render_times_np, render_upsample)
    long_comparison, _ = temporal_smooth_render(
        long_comparison, render_times_np, render_upsample)
    render_times_np = render_times_up
    print(f"[MEDI] trap render: interpolated to {long_video.shape[0]} frames "
          f"(x{render_upsample})")

long_video      = annotate_video_time(long_video, render_times_np)
long_comparison = annotate_video_time(long_comparison, render_times_np)

video_fps = 12 if render_mode == "rect" else 12 * render_upsample
save_video_mp4(os.path.join(save_dir, "MEDI_video.mp4"), long_video, fps=video_fps)
save_video_mp4(os.path.join(save_dir, "MEDI_video_with_frame.mp4"),
               long_comparison, fps=video_fps)
print(f"[MEDI] saved CONTINUOUS videos ({render_mode} render) -> MEDI_video.mp4 "
      f"and MEDI_video_with_frame.mp4 ({long_video.shape[0]} frames)")

plt.show()
print("[MEDI] done.")
