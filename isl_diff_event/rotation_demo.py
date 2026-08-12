from __future__ import annotations

from pathlib import Path
from torch.optim import Adam, lr_scheduler
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from torch import nn

from solver.dense_iwe_reconstruction import (
    iwe_l2_loss,
    normalized_iwe_l2_loss,
    unit_flow_mag_mask,
    warp_iwe_from_intensity,
    warp_iwe_from_log_image,
    make_iwe_from_log_image_eklt
)
from utils.utility import *

# ===== Debug config =====
event_path = Path("D:/BaiduNetdiskDownload/Rotation/02_rot.aedat4")
flow_init_path = Path("D:/Program Files/PycharmProjects/Diff_Ev_Rot/output/diff_rotation_demo/flow_multi.npy")
save_dir = Path("output/debug_rotation_flow_iwe_latent_summary")
sensor_size = (260, 346)
frame_index = 57
flow_ref_count = 9
min_flow_mag = 0.25
warp_steps = 4
max_metric_events = 20_000
max_train_iwe_events = 200_000
device_name = "cuda" if torch.cuda.is_available() else "cpu"

lin_log_threshold = 1.000001
random_seed = 0
print_every = 100


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame.astype(np.float32)
    max_val = float(frame.max()) if frame.size else 1.0
    if max_val > 2.0:
        frame = frame / 255.0 if max_val <= 255.0 else frame / max_val
    return np.clip(frame, 0.0, 1.0)

def gaussian_blur2d(image: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return image
    radius = max(int(round(3.0 * float(sigma))), 1)
    coords = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel_1d = torch.exp(-0.5 * (coords / float(sigma)).square())
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-12)
    kernel_2d = torch.outer(kernel_1d, kernel_1d).view(1, 1, 2 * radius + 1, 2 * radius + 1)
    return F.conv2d(image[None, None], kernel_2d, padding=radius)[0, 0]


def local_iwe_contrast_score(
    iwe: torch.Tensor,
    *,
    patch_size: tuple[int, int] = (24, 24),
    stride: tuple[int, int] = (12, 12),
    blur_sigma: float = 0.7,
    eps: float = 1e-6,
) -> torch.Tensor:
    signal = gaussian_blur2d(iwe.abs(), blur_sigma)
    x = signal[None, None]
    mean = F.avg_pool2d(x, kernel_size=patch_size, stride=stride)
    mean_sq = F.avg_pool2d(x.square(), kernel_size=patch_size, stride=stride)
    var = (mean_sq - mean.square()).clamp_min(0.0)
    score = var / mean.square().clamp_min(eps)
    weight = mean.detach()
    mask = (weight > weight.mean().detach() * 0.15).to(score.dtype)
    return (score * weight * mask).sum() / (weight * mask).sum().clamp_min(eps)


def gradient_focus_score(iwe: torch.Tensor, blur_sigma: float = 0.7, eps: float = 1e-6) -> torch.Tensor:
    signal = gaussian_blur2d(iwe.abs(), blur_sigma)
    kx = signal.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3) / 8.0
    ky = kx.transpose(-1, -2)
    x = signal[None, None]
    gx = F.conv2d(x, kx, padding=1)[0, 0]
    gy = F.conv2d(x, ky, padding=1)[0, 0]
    grad = torch.sqrt(gx.square() + gy.square() + eps)
    return (grad / signal.mean().detach().clamp_min(eps)).mean()


def signed_event_weights(p: torch.Tensor) -> torch.Tensor:
    return torch.where(p > 0, torch.ones_like(p), -torch.ones_like(p)).float()


def bilinear_splat(x: torch.Tensor, y: torch.Tensor, weights: torch.Tensor, image_shape: tuple[int, int]) -> torch.Tensor:
    h, w = image_shape
    x0_float = torch.floor(x)
    y0_float = torch.floor(y)
    dx = x - x0_float
    dy = y - y0_float
    x0 = x0_float.long()
    y0 = y0_float.long()
    out = torch.zeros(h * w, device=x.device, dtype=x.dtype)
    for ox, oy, interp in (
        (0, 0, (1.0 - dx) * (1.0 - dy)),
        (1, 0, dx * (1.0 - dy)),
        (0, 1, (1.0 - dx) * dy),
        (1, 1, dx * dy),
    ):
        xi = x0 + ox
        yi = y0 + oy
        valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        if valid.any():
            out.index_add_(0, yi[valid] * w + xi[valid], weights[valid] * interp[valid])
    return out.view(h, w)


def sample_dense_flow_at_events(flow: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    _, h, w = flow.shape
    grid_x = x / max(w - 1, 1) * 2.0 - 1.0
    grid_y = y / max(h - 1, 1) * 2.0 - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).view(1, 1, -1, 2)
    sampled = F.grid_sample(flow.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled.view(2, -1).t()


def warp_events_to_ref(
    events_t: torch.Tensor,
    flow: torch.Tensor,
    t_ref: float,
    full_duration: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = events_t[:, 1]
    y = events_t[:, 2]
    dt_norm = (events_t[:, 0] - float(t_ref)) / max(float(full_duration), 1e-6)
    step_dt = dt_norm / float(max(warp_steps, 1))
    for _ in range(max(warp_steps, 1)):
        flow_xy = sample_dense_flow_at_events(flow, x, y)
        x = x - flow_xy[:, 0] * step_dt
        y = y - flow_xy[:, 1] * step_dt
    return x, y


def iwe_at_ref(
    events_t: torch.Tensor,
    flow: torch.Tensor,
    t_ref: float,
    full_duration: float,
    image_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = image_shape
    warped_x, warped_y = warp_events_to_ref(events_t, flow, t_ref, full_duration)
    weights = signed_event_weights(events_t[:, 3])
    iwe = bilinear_splat(warped_x, warped_y, weights, image_shape)
    valid = ((warped_x >= 0) & (warped_x <= w - 1) & (warped_y >= 0) & (warped_y <= h - 1)).float().mean()
    return iwe, valid


def select_event_subset(events_np: np.ndarray, max_events: int, seed_offset: int = 0) -> torch.Tensor:
    if len(events_np) > max_events:
        rng = np.random.default_rng(random_seed + seed_offset)
        idx = np.sort(rng.choice(len(events_np), size=int(max_events), replace=False))
        events_np = events_np[idx]
    return torch.from_numpy(events_np).to(device=device, dtype=torch.float32)


def latent_tv(latent: torch.Tensor) -> torch.Tensor:
    dx = latent[:, :, 1:] - latent[:, :, :-1]
    dy = latent[:, 1:, :] - latent[:, :-1, :]
    return dx.abs().mean() + dy.abs().mean()


def generated_iwe_sequence_np(latent_seq_np: np.ndarray, unit_flow: torch.Tensor) -> np.ndarray:
    pred_iwes = []
    with torch.no_grad():
        latent_t = torch.from_numpy(latent_seq_np).to(device=device, dtype=torch.float32)
        for i in range(latent_t.shape[0]):
            pred_iwe = warp_iwe_from_intensity(latent_t[i], unit_flow, lin_log_threshold=lin_log_threshold)
            pred_iwes.append(pred_iwe.detach().cpu().numpy())
    return np.stack(pred_iwes, axis=0).astype(np.float32)


def flow_convention_candidates(flow: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "xy": flow,
        "neg_xy": -flow,
        "yx": flow[[1, 0]],
        "neg_yx": -flow[[1, 0]],
    }


def choose_iwe_convention(
    events_t: torch.Tensor,
    base_frame: torch.Tensor,
    event_warp_flow: torch.Tensor,
    refs: np.ndarray,
    full_duration: float,
    image_shape: tuple[int, int],
) -> tuple[torch.Tensor, float, list[dict[str, float | str]]]:
    target_iwes = []
    with torch.no_grad():
        for t_ref in refs:
            target, _ = iwe_at_ref(events_t, event_warp_flow, float(t_ref), full_duration, image_shape)
            target_iwes.append(target)
    records: list[dict[str, float | str]] = []
    best_loss = float("inf")
    best_flow = event_warp_flow
    best_sign = 1.0
    for flow_name, flow_candidate in flow_convention_candidates(event_warp_flow).items():
        with torch.no_grad():
            unit_flow, flow_mag, mask = unit_flow_mag_mask(flow_candidate, eps=min_flow_mag, border=1)
            pred = warp_iwe_from_intensity(base_frame, unit_flow, lin_log_threshold=lin_log_threshold)
            for target_sign in (1.0, -1.0):
                losses = []
                for target in target_iwes:
                    target_for_loss = (target * float(target_sign) / flow_mag) * mask
                    losses.append(normalized_iwe_l2_loss(pred, target_for_loss, mask))
                loss = float(torch.stack(losses).mean().detach().cpu())
                rec = {
                    "flow_convention": flow_name,
                    "target_sign": float(target_sign),
                    "calibration_loss": loss,
                }
                records.append(rec)
                if loss < best_loss:
                    best_loss = loss
                    best_flow = flow_candidate.detach().clone()
                    best_sign = float(target_sign)
    records.sort(key=lambda item: float(item["calibration_loss"]))
    return best_flow, best_sign, records


def sequence_iwe_np(
    events_t: torch.Tensor,
    flow: torch.Tensor,
    refs: np.ndarray,
    full_duration: float,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, list[float]]:
    iwes = []
    valid_ratios = []
    with torch.no_grad():
        for t_ref in refs:
            iwe, valid = iwe_at_ref(events_t, flow, float(t_ref), full_duration, image_shape)
            iwes.append(iwe.detach().cpu().numpy())
            valid_ratios.append(float(valid.detach().cpu()))
    return np.stack(iwes, axis=0).astype(np.float32), valid_ratios


def evaluate_flow_metrics(
    events_t: torch.Tensor,
    flow: torch.Tensor,
    refs: torch.Tensor,
    full_duration: float,
    image_shape: tuple[int, int],
) -> dict[str, float]:
    contrasts = []
    gradients = []
    valids = []
    with torch.no_grad():
        for t_ref in refs:
            iwe, valid = iwe_at_ref(events_t, flow, float(t_ref.detach().cpu()), full_duration, image_shape)
            contrasts.append(local_iwe_contrast_score(iwe))
            gradients.append(gradient_focus_score(iwe))
            valids.append(valid)
    return {
        "contrast": float(torch.stack(contrasts).mean().detach().cpu()),
        "gradient_focus": float(torch.stack(gradients).mean().detach().cpu()),
        "valid_ratio": float(torch.stack(valids).mean().detach().cpu()),
    }


save_dir.mkdir(parents=True, exist_ok=True)
torch.manual_seed(random_seed)
np.random.seed(random_seed)
device = torch.device(device_name)

print(f"[debug] device={device}", flush=True)
print(f"[debug] loading aedat4: {event_path}", flush=True)
ts, xs, ys, ps, frames, time_img = load_events(eve_dtype="aedat4", path=str(event_path))
frames = np.asarray(frames)
time_img = np.asarray(time_img)

static_frame_np = normalize_frame(frames[0])
blurry_frame_np = normalize_frame(frames[frame_index])
h, w = blurry_frame_np.shape
image_shape = (h, w)
exp_start, exp_end = [int(v) for v in time_img[frame_index]]
duration = float(exp_end - exp_start)

event_mask = (ts >= exp_start) & (ts <= exp_end)
events_np = np.stack(
    [
        ts[event_mask].astype(np.float32) - float(exp_start),
        xs[event_mask].astype(np.float32),
        ys[event_mask].astype(np.float32),
        ps[event_mask].astype(np.float32),
    ],
    axis=1,
)
events_np[:, 3] = np.where(events_np[:, 3] > 0, 1.0, -1.0)

print(
    f"[debug] frame_index={frame_index}, exposure=[{exp_start}, {exp_end}], "
    f"duration={duration / 1000.0:.3f} ms, events={len(events_np)}, frame={image_shape}",
    flush=True,
)

flow_init_np = np.load(flow_init_path).astype(np.float32)
if flow_init_np.shape != (2, h, w):
    flow_init_np = F.interpolate(
        torch.from_numpy(flow_init_np).unsqueeze(0),
        size=(h, w),
        mode="bilinear",
        align_corners=True,
    )[0].numpy()
flow_init = torch.from_numpy(flow_init_np).to(device=device, dtype=torch.float32)
flow_init_mag = np.sqrt(flow_init_np[0] ** 2 + flow_init_np[1] ** 2)
print(
    f"[debug] flow_init shape={flow_init_np.shape}, "
    f"mag_mean={float(flow_init_mag.mean()):.4f}, mag_max={float(flow_init_mag.max()):.4f}",
    flush=True,
)

events_train_iwe_t = select_event_subset(events_np, max_train_iwe_events, seed_offset=20)
single_ref_np = np.array([duration * 0.5], dtype=np.float32)
events_all_t = torch.from_numpy(events_np).to(device=device, dtype=torch.float32)
blurry_frame = torch.from_numpy(blurry_frame_np).to(device=device, dtype=torch.float32)

events_flow_t = select_event_subset(events_np, max_metric_events, seed_offset=10)
flow_refs_np = np.linspace(0.0, duration, flow_ref_count, dtype=np.float32)
flow_refs = torch.from_numpy(flow_refs_np).to(device=device, dtype=torch.float32)

print("[debug] selecting event-warp flow convention by IWE focus", flush=True)
event_flow_candidates = flow_convention_candidates(flow_init.detach())
event_flow_records = []
for flow_name, flow_candidate in event_flow_candidates.items():
    metrics = evaluate_flow_metrics(events_flow_t, flow_candidate, flow_refs, duration, image_shape)
    score = metrics["contrast"] * metrics["gradient_focus"]
    rec = {
        "flow_convention": flow_name,
        "score": score,
        **metrics,
    }
    event_flow_records.append(rec)
event_flow_records.sort(key=lambda item: float(item["score"]), reverse=True)
for rec in event_flow_records:
    print(
        "[debug] event-flow candidate={flow_convention:>6s} score={score:.6e} "
        "contrast={contrast:.6e} grad={gradient_focus:.6e} valid={valid_ratio:.3f}".format(**rec),
        flush=True,
    )
selected_event_flow = event_flow_records[0]
event_warp_flow = event_flow_candidates[str(selected_event_flow["flow_convention"])].detach()
print(
    "[debug] selected event-warp flow convention={flow_convention}".format(**selected_event_flow),
    flush=True,
)
print("[debug] selecting EKLT image-forward flow convention against fixed event-warped IWE", flush=True)
image_forward_flow, target_sign, convention_records = choose_iwe_convention(
    events_train_iwe_t,
    blurry_frame,
    event_warp_flow,
    single_ref_np,
    duration,
    image_shape,
)
selected_convention = convention_records[0]
print(
    "[debug] selected image-flow convention={flow_convention}, target_sign={target_sign:+.0f}, "
    "calib_l2={calibration_loss:.6e}".format(**selected_convention),
    flush=True,
)

init_flow_metrics = evaluate_flow_metrics(events_flow_t, event_warp_flow, flow_refs, duration, image_shape)
print("[debug] fixed flow: no flow optimization, using selected event-warp dense flow", flush=True)
flow_delta_rms = 0.0
print(
    "[debug] fixed flow metrics contrast={:.6e}, grad={:.6e}, valid={:.3f}".format(
        init_flow_metrics["contrast"],
        init_flow_metrics["gradient_focus"],
        init_flow_metrics["valid_ratio"],
    ),
    flush=True,
)

print(f"[debug] building one fixed-flow IWE target with {events_all_t.shape[0]} exposure events", flush=True)
iwe_single_np, single_valids = sequence_iwe_np(events_all_t, event_warp_flow, single_ref_np, duration, image_shape)
iwe_single_np = (float(target_sign) * iwe_single_np).astype(np.float32)

iwe_single_raw = torch.from_numpy(iwe_single_np[0]).to(device=device, dtype=torch.float32)
border = 1
border_mask = torch.zeros_like(iwe_single_raw)
border_mask[border:-border, border:-border] = True
iwe_single_raw *= border_mask

min_flow_mag = 0.25
# iwe_gt_non_0 = iwe_single_raw.abs() [iwe_single_raw.abs() >0].detach()
# min_flow_mag =  torch.quantile(iwe_gt_non_0, 0.1)

unit_image_forward_flow, flow_mag, mask = unit_flow_mag_mask(image_forward_flow, eps=min_flow_mag, border=1)    #prepare flow
flow_norm = image_forward_flow/flow_mag.max()

iwe_single_raw_norm =iwe_single_raw/ flow_mag.max()
iwe_single = (iwe_single_raw / flow_mag) * mask
iwe_single_unit_np = iwe_single.detach().cpu().numpy().astype(np.float32)

target_abs_p99 = float(np.percentile(np.abs(iwe_single_unit_np), 99.0))
target_abs_max = float(np.max(np.abs(iwe_single_unit_np)))
print(
    f"[debug] unit-flow IWE target abs_p99={target_abs_p99:.6e}, "
    f"abs_max={target_abs_max:.6e}",
    flush=True,
)



latent_lr = 4e-3
latent_event_weight = 10
frame_loss_weight = 0
latent_tv_weight = 1e-1
latent_iters = 4000
from function.optimizer import tv_loss
print("[debug] stage 2: optimizing one log latent image with Ax=b IWE loss", flush=True)
latent_log_img = nn.Parameter(torch.rand_like(blurry_frame).detach().clone(), requires_grad=True)
#latent_img = nn.Parameter(torch.zeros_like(blurry_frame/255).detach(), requires_grad=True)
#latent_optimizer = torch.optim.Adam([latent_log_img], lr=latent_lr)
vars = []
vars += [{'params': latent_log_img, 'lr': latent_lr}]
optimizer = Adam(vars)
scheduler = lr_scheduler.StepLR(optimizer, step_size=400//2 , gamma=0.9)
latent_history = []
best_latent = None
best_latent_score = float("inf")

for it in range(latent_iters):
    optimizer.zero_grad()
    #pred_iwe =  warp_iwe_from_intensity(latent_log_img, unit_image_forward_flow)
    #pred_iwe = warp_iwe_from_log_image(latent_log_img, unit_image_forward_flow)
    pred_iwe = warp_iwe_from_log_image(latent_log_img, flow_norm)
    #pred_iwe = warp_iwe_from_intensity(latent_log_img, flow_norm)
    #pred_iwe = make_iwe_from_log_image_eklt(latent_log_img, image_forward_flow) * border_mask
    # event_loss = iwe_l2_loss(pred_iwe,
    #                          iwe_single)
    event_loss = iwe_l2_loss(pred_iwe,
                             iwe_single_raw_norm)
    # if it ==0:
    #     event_loss = torch.tensor(0.0).to(device=device)
    # else:
    #     event_loss = iwe_l2_loss(pred_iwe_unit,
    #                              iwe_single, mask)
    tv_loss_value = tv_loss(latent_log_img)
    #tv_loss_value = latent_tv(latent_img.unsqueeze(0))
    loss = (
        latent_event_weight * event_loss
        + latent_tv_weight * tv_loss_value
    )
    loss.backward()
    optimizer.step()
    # with torch.no_grad():
    #     latent_log_img.data.clamp_(min = 0)
    rec = {
        "iter": int(it),
        "loss": float(loss.detach().cpu()),
        "event": float(event_loss.detach().cpu()),
        "latent_tv": float(tv_loss_value.detach().cpu()),
    }
    latent_history.append(rec)
    score = rec["event"]
    if score < best_latent_score:
        best_latent_score = score
        # best_latent = latent_log_img.detach().clone()
    if it % print_every == 0 or it == latent_iters - 1:
        print(
            "[debug] latent iter={iter:04d} loss={loss:.6e} event={event:.6e} "
            "tv={latent_tv:.6e}".format(**rec),
            flush=True,
        )
latent_final = best_latent if best_latent is not None else latent_log_img.detach()
#latent_final = best_latent if best_latent is not None else latent_img.detach()
latent_final_np = latent_final.detach().cpu().numpy().astype(np.float32)
latent_seq_np = latent_final_np[None]
frame_residual_np = latent_final_np - blurry_frame_np
frame_loss_mse = float(np.mean(frame_residual_np ** 2))
frame_loss_l1 = float(np.mean(np.abs(frame_residual_np)))

print("[debug] generating IWE from log latent image for target/pred comparison", flush=True)
with torch.no_grad():
    pred_iwe_single_np = pred_iwe.detach().cpu().numpy().astype(np.float32)

summary_path = save_dir / "debug_summary.png"
summary_path.parent.mkdir(parents=True, exist_ok=True)
residual_np = pred_iwe_single_np - iwe_single_unit_np
t_ms = float(single_ref_np[0]) / 1000.0
first_event = float(latent_history[0]["event"]) if latent_history else 0.0
best_event = min(float(item["event"]) for item in latent_history) if latent_history else 0.0
reset_optimizer_and_params(optimizer,vars)
plt.figure(figsize=(10.0, 5.0), dpi=150)
plt.suptitle(f"event loss {first_event:.3e} -> best {best_event:.3e}", fontsize=11)
#signed_iwe_cmap = LinearSegmentedColormap.from_list("signed_iwe", ["blue", "black", "red"])
signed_iwe_cmap = "gray";"seismic"
plt.subplot(231)
plt.imshow(make_np(latent_log_img), cmap="gray")
plt.title(f"log latent {t_ms:.1f} ms", fontsize=9)
plt.axis('off')
ax = plt.subplot(232)
plot_ax_roi(ax, make_np(iwe_single_raw) , cmap = signed_iwe_cmap, axis_ind = "off",  norm_ind = True)
plt.title("IWE", fontsize=9)
ax = plt.subplot(233)
plot_ax_roi(ax, make_np(pred_iwe) , cmap = signed_iwe_cmap, axis_ind = "off",  norm_ind = True)
plt.title("pred IWE", fontsize=9)
plt.subplot(234)
plt.imshow(blurry_frame.detach().cpu(), cmap="gray")
plt.title("frame", fontsize=9)
plt.axis('off')
ax =plt.subplot(235)
plot_ax_roi(ax, iwe_single_np[0] , cmap = signed_iwe_cmap, axis_ind = "off",  norm_ind = True)
plt.title("origin IWE", fontsize=9)
plt.subplot(236)
plt.imshow(torch.linalg.vector_norm(image_forward_flow, dim=0).detach().cpu(), cmap="gray")
plt.title("flow mag", fontsize=9)
plt.axis('off')
plt.tight_layout()




plt.savefig(summary_path, bbox_inches="tight")
# plt.close()


loss_path = save_dir / "debug_loss.png"
loss_iters = [item["iter"] for item in latent_history]
loss_values = [item["loss"] for item in latent_history]
event_values = [item["event"] for item in latent_history]
tv_values = [item["latent_tv"] for item in latent_history]
weighted_tv_values = [latent_tv_weight * value for value in tv_values]

plt.figure(figsize=(6.0, 4.0), dpi=150)
plt.semilogy(loss_iters, loss_values, label="total")
plt.semilogy(loss_iters, event_values, label="event")
plt.semilogy(loss_iters, weighted_tv_values, label="weighted tv")
plt.xlabel("iteration")
plt.ylabel("loss")
plt.title("optimization loss")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(loss_path, bbox_inches="tight")
# plt.close()




print(f"[debug] saved summary figure: {summary_path.resolve()}", flush=True)
print(f"[debug] saved loss figure: {loss_path.resolve()}", flush=True)
print(
    "[debug] final frame_loss_mse={:.6e}, frame_loss_l1={:.6e}, best_event={:.6e}, "
    "single_valid={:.4f}, flow_delta_rms={:.6e}".format(
        frame_loss_mse,
        frame_loss_l1,
        best_latent_score,
        float(np.mean(single_valids)),
        flow_delta_rms,
    ),
    flush=True,
)




# Optical flow
flow_x = image_forward_flow[0, ...].detach().cpu().numpy()
flow_y = image_forward_flow[1, ...].detach().cpu().numpy()

from utils.utils_event_flow import color_optical_flow, estimate_rotation_center

flow_rgb, color_wheel, max_magnitude = color_optical_flow(flow_x, flow_y)

cx, cy = estimate_rotation_center(flow_x, flow_y)
fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.0), dpi=150)

axes[0].imshow(flow_rgb)
axes[0].scatter(
    cx,
    cy,
    s=100,
    marker=".",
    linewidths=1,
    c="red",
    label=f"disc center = ({cx:.1f}, {cy:.1f})",
)
axes[0].scatter(
    sensor_size[1]//2,
    sensor_size[0]//2,
    s=100,
    marker=".",
    linewidths=1,
    c = "white",
    label=f"sensor center = ({sensor_size[0]//2:.1f}, {sensor_size[1]//2:.1f})",
)
axes[0].set_title("Rotational optical flow")
axes[0].legend()
axes[0].axis("off")

axes[1].imshow(color_wheel)
axes[1].set_title("Color wheel")
axes[1].axis("off")

plt.tight_layout()
plt.show()