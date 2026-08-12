import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from typing import Tuple, Union


NUMPY_TORCH = Union[np.ndarray, torch.Tensor]
FLOAT_TORCH = Union[float, torch.Tensor]
def warp_image_torch(im1: torch.Tensor, global_shift: torch.Tensor) -> np.ndarray:
    """Warp image using global shift (translation)

    Args:
        im1 (torch.Tensor): [H, W]
        global_shift (torch.Tensor): [2]

    Returns:
        torch.Tensor: [H, W]
    """
    im1_tensor = im1[None, None]  # b=1, c-=1, h, w

    h, w = im1.shape
    coord_y, coord_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    coord_y = coord_y[None, None] / ((h - 1) / 2.0) - 1
    coord_x = coord_x[None, None] / ((w - 1) / 2.0) - 1
    warp_y = coord_y.to(im1.device) - global_shift[1] / ((h - 1) / 2.0)
    warp_x = coord_x.to(im1.device) - global_shift[0] / ((w - 1) / 2.0)
    grid = torch.cat([warp_x, warp_y], dim=1).double().permute((0, 2, 3, 1)).to(im1.device)
    warped_im1 = F.grid_sample(
        im1_tensor, grid, mode="bilinear", align_corners=True
    )
    return warped_im1.squeeze()

def iwe_masked_flow(iwe, flow):
    event_mask = iwe <= 1e-3
    flow[:,event_mask] = 0.0
    return flow

from utils.utils_img_rec import ImageReconstructor
def  prepare_uiwe_flow_grid(iwe_np, Dxy, device):
    flow_np = np.asarray([np.ones_like(iwe_np) * Dxy[0],np.ones_like(iwe_np) * Dxy[1]])
    mask_e = np.abs(iwe_np) > 1e-3
    flow_torch = torch.from_numpy(flow_np).unsqueeze(0).to(device)
    image_reconstructor = ImageReconstructor(flow_torch)
    uflow_y  = image_reconstructor.uni_flow_np_y * mask_e
    uflow_x = image_reconstructor.uni_flow_np_x * mask_e
    iwe_torch = torch.from_numpy(iwe_np).to(device).unsqueeze(0).unsqueeze(0).to(device)
    image_reconstructor._check_iwe(iwe_torch)
    uiwe =image_reconstructor.iwe
    height, width = uiwe.shape
    map_y, map_x = np.mgrid[0:height, 0:width]
    map_y = map_y.astype(np.float32)
    map_x = map_x.astype(np.float32)
    warped_y = (map_y + uflow_y).reshape(1, height, width)
    warped_x = (map_x + uflow_x).reshape(1, height, width)
    warped_y = 2 * warped_y / (height - 1) - 1
    warped_x = 2 * warped_x / (width - 1) - 1
    grid_y = torch.from_numpy(warped_y).float().unsqueeze(0)
    grid_x = torch.from_numpy(warped_x).float().unsqueeze(0)
    grid_pos = torch.cat([grid_x, grid_y], dim=1).permute(0, 2, 3, 1).to(device)
    uiwe /= np.amax(np.abs(uiwe))
    uiwe_torch = torch.from_numpy(np.ascontiguousarray(uiwe)).float().to(device).squeeze()
    return uiwe_torch, grid_pos, uflow_x, uflow_y
# uiwe_torch, grid_pos = prepare_uiwe_flow_grid(iwe_np, Dxy, device)
def foward_event_flow(I, grid_pos):
    warped_I = F.grid_sample(I.unsqueeze(0).unsqueeze(0), grid_pos, mode="bilinear", padding_mode="zeros", align_corners=True).squeeze()
    return I - warped_I

# uiwe_torch, grid_pos = prepare_uiwe_flow_grid(iwe_np, Dxy, device)
#iwe_pred = foward_event_flow(I, grid_pos)




def warp_image_forward(im1: NUMPY_TORCH, forward_flow: NUMPY_TORCH) -> NUMPY_TORCH:
    """Warp image using forward flow.

    Args:
        im1 (np.ndarray): [H, W]
        forward_flow (np.ndarray): [2, H, W]

    Returns:
        np.ndarray: [H, W]
    """
    if torch.is_tensor(im1) == False:
        im1_tensor = torch.from_numpy(im1.astype(np.float64))[None, None]  # b=1, c-=1, h, w
        flow_tensor = torch.from_numpy(forward_flow.astype(np.float64))[None]  # b=1, c=2, h, w
        _return_numpy = True
    else:
        im1_tensor = im1[None, None]
        flow_tensor = forward_flow[None]
        _return_numpy = False

    h, w = im1.shape
    coord_y, coord_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    coord_x = coord_x[None, None] / ((w - 1) / 2.0) - 1
    coord_y = coord_y[None, None] / ((h - 1) / 2.0) - 1
    warp_x = coord_x.to(flow_tensor.device) - flow_tensor[:, [0]] / ((w - 1) / 2.0)
    warp_y = coord_y.to(flow_tensor.device) - flow_tensor[:, [1]] / ((h - 1) / 2.0)

    grid = torch.cat([warp_x, warp_y], dim=1).permute((0, 2, 3, 1))

    warped_im1 = F.grid_sample(
        im1_tensor, grid, mode="bilinear", align_corners=True
    )
    if _return_numpy:
        return warped_im1.detach().cpu().numpy().squeeze()
    return warped_im1.squeeze()



def warp_image_forward_torch(im1, forward_flow):
    """Warp image using forward flow.

    Args:
        im1 (torch): [H, W]
        forward_flow (torch): [2, H, W]

    Returns:
        torch: [H, W]
    """
    im1_tensor = im1[None, None]
    flow_tensor = forward_flow[None]
    h, w = im1.shape
    coord_y, coord_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    coord_x = coord_x[None, None] / ((w - 1) / 2.0) - 1
    coord_y = coord_y[None, None] / ((h - 1) / 2.0) - 1
    warp_x = coord_x.to(flow_tensor.device) - flow_tensor[:, [0]] / ((w - 1) / 2.0)
    warp_y = coord_y.to(flow_tensor.device) - flow_tensor[:, [1]] / ((h - 1) / 2.0)

    grid = torch.cat([warp_x, warp_y], dim=1).permute((0, 2, 3, 1))

    warped_im1 = F.grid_sample(
        im1_tensor, grid, mode="bilinear", align_corners=True
    )
    return warped_im1.squeeze()

def warp_events_flow_torch(xt, yt, tt, pt, flow_field, t0=None,
                           batched=False, batch_indices=None):
    """
    Given events and a flow field, warp the events by the flow
    Parameters
    ----------
    xs : list of event x coordinates
    ys : list of event y coordinates
    ts : list of event timestamps
    ps : list of event polarities
    flow_field : 2D tensor containing the flow at each x,y position
    t0 : the reference time to warp events to. If empty, will use the
        timestamp of the last event
    Returns
    -------
    warped_xt: x coords of warped events
    warped_yt: y coords of warped events
    """
    if len(xt.shape) > 1:
        xt, yt, tt, pt = xt.squeeze(), yt.squeeze(), tt.squeeze(), pt.squeeze()
    if t0 is None:
        t0 = tt[-1]
    while len(flow_field.size()) < 4:
        flow_field = flow_field.unsqueeze(0)
    if len(xt.size()) == 1:
        event_indices = torch.transpose(torch.stack((xt, yt), dim=0), 0, 1)
    else:
        event_indices = torch.transpose(torch.cat((xt, yt), dim=1), 0, 1)
    # event_indices.requires_grad_ = False
    event_indices = torch.reshape(event_indices, [1, 1, len(xt), 2])

    # Event indices need to be between -1 and 1 for F.gridsample
    event_indices[:, :, :, 0] = event_indices[:, :, :, 0] / (flow_field.shape[-1] - 1) * 2.0 - 1.0
    event_indices[:, :, :, 1] = event_indices[:, :, :, 1] / (flow_field.shape[-2] - 1) * 2.0 - 1.0

    flow_at_event = F.grid_sample(flow_field, event_indices, align_corners=True)

    dt = (tt - t0).squeeze()

    warped_xt = xt - flow_at_event[:, 0, :, :].squeeze() * dt
    warped_yt = yt - flow_at_event[:, 1, :, :].squeeze() * dt

    return warped_xt, warped_yt

# Generation
def generate_dense_optical_flow(image_size: tuple, max_val: int = 30) -> np.ndarray:
    """Generate random optical flow.

    Args:
        image_size (tuple) ... (H, W)

    Returns:
        flow (np.ndarray) ... [2 x H x W] array.
    """
    flow = np.random.uniform(-max_val, max_val, (2,) + image_size)
    return flow


def generate_uniform_optical_flow(image_size: tuple, x: int = 30, y: int = 30) -> np.ndarray:
    """Generate uniform optical flow.

    Args:
        image_size (tuple) ... (H, W)
        x ... H direction component
        y ... W direction component

    Returns:
        flow (np.ndarray) ... [2 x H x W] array.
    """
    flow = np.ones((2,) + image_size) * np.array([x, y])[:, None, None]
    return flow

#
# def warp_event_from_optical_flow(
#         self, event: NUMPY_TORCH, flow: NUMPY_TORCH, reference_time: FLOAT_TORCH
# ) -> Tuple[NUMPY_TORCH, dict]:
#     """Warp events from dense optical flow
#
#     Args:
#         event (np.ndarray) ... [(b,) n x 4]. Each event is (x, y, t, p)
#         flow ... [(b,) 2, H, W]. Velocity (Optical flow) of the image plane at the position (x, y)
#         reference_time (float) ... reference time
#
#     Returns:
#         warped_event (np.ndarray) ... [(b,) n, 4]. Warped event. (x', y', time, p). x' and y' could be float.
#         feature (dict) ... Feature dict. if self.calculate_feature is True.
#     """
#     dt = self.calculate_dt(event, reference_time)
#
#     if len(event.shape) == 2:
#         event = event[None, ...]
#         flow = flow[None, ...]
#         dt = dt[None, ...]
#     assert len(dt.shape) + 1 == len(flow.shape) - 1 == 3
#
#     warped_torch = event.clone()
#     flow_flat = flow.reshape((flow.shape[0], 2, -1))
#     _ind = event[..., 0].long() * self.image_size[1] + event[..., 1].long()
#     warped_torch[..., 0] = event[..., 0] - dt * torch.gather(flow_flat[:, 0], 1, _ind)
#     warped_torch[..., 1] = event[..., 1] - dt * torch.gather(flow_flat[:, 1], 1, _ind)
#     warped_torch[..., 2] = dt
#     return warped_torch.squeeze()



def color_optical_flow(
    flow_x: np.ndarray, flow_y: np.ndarray, max_magnitude=None, ord=1.0
):
    """Color optical flow.
    Args:
        flow_x (numpy.ndarray) ... [H x W], height direction.
        flow_y (numpy.ndarray) ... [H x W], width direction.
        max_magnitude (float, optional) ... Max magnitude used for the colorization. Defaults to None.
        ord (float) ... 1: our usual, 0.5: DSEC colorinzing.

    Returns:
        flow_rgb (np.ndarray) ... [W, H]
        color_wheel (np.ndarray) ... [H, H] color wheel
        max_magnitude (float) ... max magnitude of the flow.
    """
    flows = np.stack((flow_x, flow_y), axis=2)
    flows[np.isinf(flows)] = 0
    flows[np.isnan(flows)] = 0
    mag = np.linalg.norm(flows, axis=2) ** ord
    ang = (np.arctan2(flow_y, flow_x) + np.pi) * 180.0 / np.pi / 2.0
    ang = ang.astype(np.uint8)
    hsv = np.zeros([flow_x.shape[0], flow_x.shape[1], 3], dtype=np.uint8)
    hsv[:, :, 0] = ang
    hsv[:, :, 1] = 255
    if max_magnitude is None:
        max_magnitude = mag.max()
    hsv[:, :, 2] = (255 * mag / max_magnitude).astype(np.uint8)
    # hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Color wheel
    hsv = np.zeros([flow_x.shape[0], flow_x.shape[0], 3], dtype=np.uint8)
    xx, yy = np.meshgrid(
        np.linspace(-1, 1, flow_x.shape[0]), np.linspace(-1, 1, flow_x.shape[0])
    )
    mag = np.linalg.norm(np.stack((xx, yy), axis=2), axis=2)
    # ang = (np.arctan2(yy, xx) + np.pi) * 180 / np.pi / 2.0
    ang = (np.arctan2(yy, xx) + np.pi) * 180 / np.pi / 2.0
    hsv[:, :, 0] = ang.astype(np.uint8)
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = (255 * mag / mag.max()).astype(np.uint8)
    # hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    color_wheel = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return flow_rgb, color_wheel, max_magnitude



def calculate_dt(
    event: NUMPY_TORCH,
    reference_time: FLOAT_TORCH,
    time_period= None,
        normalize_t = True
) -> NUMPY_TORCH:
    """Calculate dt.
    First, it operates `t - reference_time`. And then it operates normalization if
    self.normalize_t is True. `time_period` is effective when normalization.

    Args:
        event (NUMPY_TORCH): [(b,) n, 4]
        reference_time (FLOAT_TORCH): The reference timestamp.
        time_period (Optional[FLOAT_TORCH], optional): If normalize is True, you can specify the
            period for the normalization. Defaults to None (normalize so that the max - min = 1).

    Returns:
        NUMPY_TORCH: dt array. [(b,) n]
    """
    dt = event[..., 2] - reference_time
    if normalize_t:  # to [0, 1]
        if time_period is None:
            pass #time_period = nt_max(dt, -1) - nt_min(dt, -1)
        dt /= time_period[..., None]
    return dt


def warp_event_from_optical_flow(
         event: NUMPY_TORCH, flow: NUMPY_TORCH, reference_time: FLOAT_TORCH ,  image_size) -> Tuple[NUMPY_TORCH, dict]:

    dt = calculate_dt(event, reference_time)

    if len(event.shape) == 2:
        event = event[None, ...]
        flow = flow[None, ...]
        dt = dt[None, ...]
    assert len(dt.shape) + 1 == len(flow.shape) - 1 == 3


    warped_torch = event.clone()
    flow_flat = flow.reshape((flow.shape[0], 2, -1))
    _ind = event[..., 0].long() * image_size[1] + event[..., 1].long()
    warped_torch[..., 0] = event[..., 0] - dt * torch.gather(flow_flat[:, 0], 1, _ind)
    warped_torch[..., 1] = event[..., 1] - dt * torch.gather(flow_flat[:, 1], 1, _ind)
    warped_torch[..., 2] = dt

    return warped_torch.squeeze()



def estimate_rotation_center(
    flow_x: np.ndarray,
    flow_y: np.ndarray,
    magnitude_threshold: float | None = None,
) -> tuple[float, float]:
    """
    Estimate the center of a predominantly rotational optical-flow field.

    Returns:
        cx, cy: rotation-center coordinates in image coordinates.
    """
    flow_x = np.asarray(flow_x, dtype=np.float64)
    flow_y = np.asarray(flow_y, dtype=np.float64)

    if flow_x.shape != flow_y.shape:
        raise ValueError("flow_x and flow_y must have the same shape.")

    height, width = flow_x.shape
    yy, xx = np.mgrid[0:height, 0:width]

    magnitude = np.hypot(flow_x, flow_y)

    if magnitude_threshold is None:
        # 去掉接近零的、不稳定的流。可根据数据进一步调整。
        magnitude_threshold = np.percentile(magnitude, 20)

    valid = (
        np.isfinite(flow_x)
        & np.isfinite(flow_y)
        & (magnitude > magnitude_threshold)
    )

    u = flow_x[valid]
    v = flow_y[valid]
    x = xx[valid]
    y = yy[valid]
    mag = magnitude[valid]

    # u * cx + v * cy = u * x + v * y
    A = np.column_stack((u, v))
    b = u * x + v * y

    # 用光流模长加权，较强的光流通常方向更可靠。
    weights = np.sqrt(mag / (mag.max() + 1e-12))
    A_weighted = A * weights[:, None]
    b_weighted = b * weights

    center, _, rank, _ = np.linalg.lstsq(
        A_weighted,
        b_weighted,
        rcond=None,
    )

    if rank < 2:
        raise RuntimeError(
            "The flow field does not contain enough independent directions "
            "to estimate a unique rotation center."
        )

    cx, cy = center
    return float(cx), float(cy)

