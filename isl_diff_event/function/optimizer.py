# -*- coding: utf-8 -*-
"""
optimizer for inverse diffraction imaging.
@author: Ni Chen (https://ni-chen.github.io/)
"""

import math
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim.optimizer import Optimizer

Params = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]

LossClosure = Callable[[], float]
OptLossClosure = Optional[LossClosure]
Betas2 = Tuple[float, float]
State = Dict[str, Any]
OptFloat = Optional[float]
Nus2 = Tuple[float, float]


########################################################################################################################
class AdamP(Optimizer):
    '''
        https://clovaai.github.io/AdamP/
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, delta=0.1, wd_ratio=0.1, nesterov=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        delta=delta, wd_ratio=wd_ratio, nesterov=nesterov)
        super(AdamP, self).__init__(params, defaults)

    def _channel_view(self, x):
        return x.view(x.size(0), -1)

    def _layer_view(self, x):
        return x.reshape(1, -1)#x.view(1, -1)

    def _cosine_similarity(self, x, y, eps, view_func):
        x = view_func(x)
        y = view_func(y)

        # return F.cosine_similarity(x, y, dim=1, eps=eps).abs_()
        # Modified by Ni Chen (chenni@snu.ac.kr)
        return F.cosine_similarity(x.abs(), y.abs(), dim=1, eps=eps).abs_()

    def _projection(self, p, grad, perturb, delta, wd_ratio, eps):
        wd = 1
        expand_size = [-1] + [1] * (len(p.shape) - 1)
        for view_func in [self._channel_view, self._layer_view]:

            cosine_sim = self._cosine_similarity(grad, p.data, eps, view_func)

            if cosine_sim.max() < delta / math.sqrt(view_func(p.data).size(1)):
                p_n = p.data / view_func(p.data).norm(dim=1).view(expand_size).add_(eps)
                perturb -= p_n * view_func(p_n * perturb).sum(dim=1).view(expand_size)
                wd = wd_ratio

                return perturb, wd

        return perturb, wd

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                beta1, beta2 = group['betas']
                nesterov = group['nesterov']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                # Adam
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1

                if nesterov:
                    perturb = (beta1 * exp_avg + (1 - beta1) * grad) / denom
                else:
                    perturb = exp_avg / denom

                # Projection
                wd_ratio = 1
                if len(p.shape) > 1:
                    perturb, wd_ratio = self._projection(p, grad, perturb, group['delta'], group['wd_ratio'], group['eps'])

                # Weight decay
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'] * wd_ratio)

                # Step
                p.data.add_(perturb, alpha=-step_size)

        return loss


########################################################################################################################


def l1_loss_basic(x):
    y = torch.mean(torch.abs(x))

    return y
def masked_normalized_iwe_l2(iwe_pred, iwe_gt, mask_e, eps=1e-8):
    mask = mask_e.detach().to(device=iwe_pred.device, dtype=iwe_pred.dtype)
    target = iwe_gt.detach().to(device=iwe_pred.device, dtype=iwe_pred.dtype)

    pred_m = iwe_pred * mask
    target_m = target * mask

    pred_n = pred_m / (torch.linalg.vector_norm(pred_m) + eps)
    target_n = target_m / (torch.linalg.vector_norm(target_m) + eps)

    diff = pred_n - target_n
    return (diff.square() * mask).sum() / (mask.sum() + eps)

def l2_loss(x):
    y = torch.mean(torch.square(x))
    return y
def iwe_z_score_loss(x , y):
    x_p = (x - x.mean()) / (x.std() + 1e-3)
    y_p = (y - y.mean()) / (y.std() + 1e-3)
    return  l2_loss(x_p - y_p)


def iwe_cosine_loss(iwe_pred, iwe_gt, eps=1e-8):
    gt = iwe_gt.detach()

    dot = torch.sum(iwe_pred * gt)
    pred_norm = torch.sqrt(torch.sum(iwe_pred ** 2) + eps)
    gt_norm = torch.sqrt(torch.sum(gt ** 2) + eps)

    return 1.0 - dot / (pred_norm * gt_norm + eps)


def l2_loss_mask(x, mask=None):
    if mask is not None:
        return torch.mean( (x* mask) ** 2)
    else:
        return torch.mean(x ** 2)

def l1_loss_simple(x):
    return torch.norm(x,1)


def l1_loss(x, beta = 0.5):
    smooth_loss = torch.nn.SmoothL1Loss(beta = beta, reduction='mean')
    return smooth_loss(x, torch.zeros_like(x))

def l1_loss_mask(x, mask, beta=0.5):
    diff = x* mask
    loss = torch.where(
        torch.abs(diff) < beta,
        0.5 * (diff ** 2) / beta,
        torch.abs(diff) - 0.5 * beta
    )
    return loss.sum() / (mask.sum() + 1e-8)

def masked_l2(x, mask):
    loss = torch.square(x * mask)
    return loss.sum() / (mask.sum() + 1e-8)

def tv_loss(x, tv_order=1, tv_tau=1e-4, iso=True):
    '''The smaller tv_tau, the smoother the image.'''
    arr_size = 1
    for idx in range(len(x.shape)):
        arr_size = arr_size * x.shape[idx]

    if tv_order == 1:
        x_d = x - torch.roll(x, -1, dims=1)
        y_d = x - torch.roll(x, -1, dims=0)

    elif tv_order == 2:
        '''need check'''
        x_d = x - 2 * torch.roll(x, -1, dims=1) + torch.roll(x, 2, dims=1)
        y_d = x - 2 * torch.roll(x, -1, dims=0) + torch.roll(x, 2, dims=0)

    elif tv_order == 3:
        '''need check'''
        x_d = x - 3 * torch.roll(x, -1, dims=1) + 3 * torch.roll(x, 2, dims=1) - torch.roll(x, 3, dims=1)
        y_d = x - 3 * torch.roll(x, -1, dims=0) + 3 * torch.roll(x, 2, dims=0) - torch.roll(x, 3, dims=0)

    else:
        print('order should be smaller than 3')

    x_d[:, -tv_order:] = torch.tensor(0.0, device=x.device)
    y_d[-tv_order:, :] = torch.tensor(0.0, device=x.device)
    # x_d[:, :tv_order-1] = torch.zeros(1, device=x.device)
    # y_d[-tv_order:-1, :] = torch.zeros(1, device=x.device)

    if iso == True:
        TV_amp = torch.sqrt(x_d.abs() ** 2 + y_d.abs() ** 2 + tv_tau).sum()

    elif iso == False:
        TV_amp = (x_d.abs() + y_d.abs()).sum()

    elif iso == 'mix':
        '''https://arxiv.org/abs/2005.04401'''
        TV_amp = (x_d.abs() + y_d.abs()).sum() - 0.5 * torch.sqrt(x_d.abs() ** 2 + y_d.abs() ** 2 + tv_tau).sum()

    else:
        print('wrong parameter')

    return TV_amp / arr_size

def smooth_l0(x: torch.Tensor, alpha: float = 30.0) -> torch.Tensor:
    dx = x[:, 1:] - x[:, :-1]
    dy = x[1:, :] - x[:-1, :]
    g2 = F.pad(dx**2, (0,1)) + F.pad(dy**2, (0,0,0,1))
    loss_map = 1 - torch.exp(-alpha * g2)
    return loss_map.mean()

def smooth_l0_weight(x: torch.Tensor, alpha: float = 30.0, w_xy =[1, 1]) -> torch.Tensor:
    dx = x[:, 1:] - x[:, :-1]
    dy = x[1:, :] - x[:-1, :]
    g2 = F.pad(dx**2, (0,1))* w_xy[0] + F.pad(dy**2, (0,0,0,1))* w_xy[1]
    loss_map = 1 - torch.exp(-alpha * g2)
    return loss_map.mean()


def tv_loss_flow(x, tv_order=1, tv_tau=1e-4,  iso=True, Dxy =[1, 1]):
    '''The smaller tv_tau, the smoother the image.'''
    arr_size = 1
    for idx in range(len(x.shape)):
        arr_size = arr_size * x.shape[idx]

    if tv_order == 1:
        x_d = x - torch.roll(x, -1, dims=1)
        y_d = x - torch.roll(x, -1, dims=0)
    elif tv_order == 2:
        '''need check'''
        x_d = x - 2 * torch.roll(x, -1, dims=1) + torch.roll(x, 2, dims=1)
        y_d = x - 2 * torch.roll(x, -1, dims=0) + torch.roll(x, 2, dims=0)

    x_d[:, -tv_order:] = torch.tensor(0.0, device=x.device)
    y_d[-tv_order:, :] = torch.tensor(0.0, device=x.device)
    d_ver = -x_d * Dxy[1] + y_d * Dxy[0]
    if   iso:
        TV_amp = torch.sqrt((d_ver.abs()) ** 2 + tv_tau).sum()
    else:
        TV_amp = (d_ver.abs()).sum()
    return TV_amp / arr_size


def tv_loss_weight(x, tv_order=1, tv_tau=1e-4, iso=True, w_xy =[1, 1]):
    '''The smaller tv_tau, the smoother the image.'''
    arr_size = 1
    for idx in range(len(x.shape)):
        arr_size = arr_size * x.shape[idx]

    if tv_order == 1:
        x_d = x - torch.roll(x, -1, dims=1)
        y_d = x - torch.roll(x, -1, dims=0)
    elif tv_order == 2:
        '''need check'''
        x_d = x - 2 * torch.roll(x, -1, dims=1) + torch.roll(x, 2, dims=1)
        y_d = x - 2 * torch.roll(x, -1, dims=0) + torch.roll(x, 2, dims=0)

    # x_d[:, -tv_order:-1] = torch.tensor(0.0, device=x.device) * w_xy[0]
    # y_d[:tv_order - 1, :] = torch.tensor(0.0, device=x.device) * w_xy[1]
    # x_d[:, :tv_order-1] = torch.zeros(1, device=x.device)
    # y_d[-tv_order:-1, :] = torch.zeros(1, device=x.device)

    if iso == True:
        TV_amp = torch.sqrt((x_d.abs()* w_xy[0]) ** 2 + (y_d.abs()* w_xy[1]) ** 2 + tv_tau).sum()
    elif iso == False:
        TV_amp = (x_d.abs()* w_xy[0] + y_d.abs()* w_xy[1]).sum()
    return TV_amp / arr_size

def tv_loss_phi(x, tv_order=1, tv_tau=1e-4, iso=True):
    '''The smaller tv_tau, the smoother the image.'''
    arr_size = 1
    for idx in range(len(x.shape)):
        arr_size = arr_size * x.shape[idx]

    if tv_order == 1:
        x_d = x - torch.roll(x, -1, dims=1)
        y_d = x - torch.roll(x, -1, dims=0)

    elif tv_order == 2:
        '''need check'''
        x_d = x - 2 * torch.roll(x, -1, dims=1) + torch.roll(x, 2, dims=1)
        y_d = x - 2 * torch.roll(x, -1, dims=0) + torch.roll(x, 2, dims=0)

    elif tv_order == 3:
        '''need check'''
        x_d = x - 3 * torch.roll(x, -1, dims=1) + 3 * torch.roll(x, 2, dims=1) - torch.roll(x, 3, dims=1)
        y_d = x - 3 * torch.roll(x, -1, dims=0) + 3 * torch.roll(x, 2, dims=0) - torch.roll(x, 3, dims=0)

    else:
        print('order should be smaller than 3')


    x_d[:, -tv_order:] = torch.tensor(0.0, device=x.device)
    y_d[-tv_order:, :] = torch.tensor(0.0, device=x.device)
    # x_d[:, :tv_order-1] = torch.zeros(1, device=x.device)
    # y_d[-tv_order:-1, :] = torch.zeros(1, device=x.device)
    x_sin_d = torch.exp(1j* x_d.angle()).imag
    y_sin_d = torch.exp(1j * y_d.angle()).imag
    TV_amp = torch.sqrt(x_sin_d ** 2 + y_sin_d ** 2 + tv_tau).sum()
    # if iso == True:
    #     TV_amp = torch.sqrt(x_d.abs() ** 2 + y_d.abs() ** 2 + tv_tau).sum()
    #
    # elif iso == False:
    #     TV_amp = (x_d.abs() + y_d.abs()).sum()
    #
    # elif iso == 'mix':
    #     '''https://arxiv.org/abs/2005.04401'''
    #     TV_amp = (x_d.abs() + y_d.abs()).sum() - 0.5 * torch.sqrt(x_d.abs() ** 2 + y_d.abs() ** 2 + tv_tau).sum()
    #
    # else:
    #     print('wrong parameter')

    return TV_amp / arr_size

def tv_loss_phi_comp(t, tv_order=1, tv_tau=1e-4, iso=True):
    """
    Total Variation loss on phase gradient.
    Implements TV(∇φ), where ∇φ = Im(∇t / t)
    t: complex tensor of shape (H, W)
    """

    eps = 1e-8
    t_safe = t + eps  # to avoid division by zero

    if tv_order == 1:
        dx = torch.roll(t, -1, dims=1) - t
        dy = torch.roll(t, -1, dims=0) - t
    elif tv_order == 2:
        dx = torch.roll(t, -2, dims=1) - 2 * torch.roll(t, -1, dims=1) + t
        dy = torch.roll(t, -2, dims=0) - 2 * torch.roll(t, -1, dims=0) + t
    elif tv_order == 3:
        dx = torch.roll(t, -3, dims=1) - 3 * torch.roll(t, -2, dims=1) + 3 * torch.roll(t, -1, dims=1) - t
        dy = torch.roll(t, -3, dims=0) - 3 * torch.roll(t, -2, dims=0) + 3 * torch.roll(t, -1, dims=0) - t
    else:
        raise ValueError("tv_order must be 1, 2, or 3.")

    dx[:, -tv_order:] = torch.tensor(0.0, device=x.device)
    dy[-tv_order:, :] = torch.tensor(0.0, device=x.device)
    # phase gradients: ∇φ = Im(∇t / t)
    dx_phi = (dx / t_safe).imag
    dy_phi = (dy / t_safe).imag

    if iso is True:
        tv = torch.sqrt(dx_phi**2 + dy_phi**2 + tv_tau).sum()
    elif iso is False:
        tv = (dx_phi.abs() + dy_phi.abs()).sum()
    elif iso == 'mix':
        tv = (dx_phi.abs() + dy_phi.abs()).sum() - 0.5 * torch.sqrt(dx_phi**2 + dy_phi**2 + tv_tau).sum()
    else:
        raise ValueError("iso must be True, False, or 'mix'")

    return tv / t.numel()

def tv_loss_ph(x, tv_order=1, tv_tau=1e-4, iso=True):
    """
    Computes TV(sin(φ)) where φ = angle(x), via Im(exp(jφ))
    Input: complex tensor x (dtype=torch.complex64)
    Output: scalar TV loss
    """
    arr_size = x.numel()

    # proxy: sin(φ)
    ph = torch.exp(1j * x.angle()).imag  # shape: same as x, dtype: float32

    # Difference computation
    if tv_order == 1:
        px_d = ph - torch.roll(ph, -1, dims=1)
        py_d = ph - torch.roll(ph, -1, dims=0)
    elif tv_order == 2:
        px_d = ph - 2 * torch.roll(ph, -1, dims=1) + torch.roll(ph, -2, dims=1)
        py_d = ph - 2 * torch.roll(ph, -1, dims=0) + torch.roll(ph, -2, dims=0)
    elif tv_order == 3:
        px_d = ph - 3 * torch.roll(ph, -1, dims=1) + 3 * torch.roll(ph, -2, dims=1) - torch.roll(ph, -3, dims=1)
        py_d = ph - 3 * torch.roll(ph, -1, dims=0) + 3 * torch.roll(ph, -2, dims=0) - torch.roll(ph, -3, dims=0)
    else:
        raise ValueError("tv_order must be 1, 2, or 3")

    # Zero-out border to avoid wrap-around effect

    px_d[:, -tv_order:] = torch.tensor(0.0, device=x.device)
    py_d[-tv_order:, :] = torch.tensor(0.0, device=x.device)

    # TV aggregation
    if iso is True:
        TV_phase = torch.sqrt(px_d ** 2 + py_d ** 2 + tv_tau).sum()
    elif iso is False:
        TV_phase = torch.abs(px_d).sum() + torch.abs(py_d).sum()
    elif iso == 'mix':
        TV_phase = torch.abs(px_d).sum() + torch.abs(py_d).sum() - 0.5 * torch.sqrt(px_d ** 2 + py_d ** 2 + tv_tau).sum()
    else:
        raise ValueError("iso must be True, False, or 'mix'")

    return TV_phase / arr_size



def NPCC(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    return 1-torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))


def  forward_grad(x):
    x_d = torch.roll(x, -1, dims=1) - x
    y_d = torch.roll(x, -1, dims=0) - x
    x_d[:, -1:] = torch.tensor(0.0, device=x.device)
    y_d[-1:, :] = torch.tensor(0.0, device=x.device)
    return y_d, x_d

# def  forward_grad(x):
#     x_d = torch.gradient(x)[1]
#     y_d = torch.gradient(x)[0]
#     return y_d, x_d
def  forward_lap(x):
    return torch.gradient(torch.gradient(x)[0])[0] + torch.gradient(torch.gradient(x)[1])[1]


def forward_lap_x(x):
    return torch.gradient(torch.gradient(x)[0])[0]


def lap_loss(x):
    return (torch.square(forward_lap(x))).sum()

def conv2d_from_kernel(kernel, channels, device, stride=1,padding="valid"):
    kernel_size = kernel.shape
    kernel = kernel / kernel.sum()
    kernel = kernel.repeat(channels, 1, 1, 1)
    filter = nn.Conv2d(
        in_channels=channels, out_channels=channels,
        kernel_size=kernel_size, groups=channels, bias=False, stride=stride,
        padding=padding#"same"#"valid"
    )
    filter.weight.data = kernel
    filter.weight.requires_grad = False
    return filter.to(device)


def conv2d_from_kernel_diff(kernel, x, channels=1, stride=1, padding="valid"):
    """Differentiable counterpart of conv2d_from_kernel.

    Applies a 2D convolution of `x` with `kernel` via F.conv2d directly, so the
    kernel stays attached to the autograd graph. Unlike conv2d_from_kernel (which
    wraps the kernel in an nn.Conv2d whose weight is detached with
    requires_grad=False), gradients here flow back into `kernel` and therefore
    into whatever produced it (e.g. the motion-blur kernel built from Dxy_pc).

    Numerically equivalent to conv2d_from_kernel(kernel, ...)(x): the kernel is
    sum-normalised the same way; only the gradient path to the kernel differs.

    Args:
        kernel: 2D tensor (kH, kW), e.g. already flipped for cross-correlation.
        x: input tensor shaped (N, C, H, W) with C == channels.
        channels: number of image channels (grouped conv, one kernel per channel).
        stride, padding: forwarded to F.conv2d ("valid"/"same" strings allowed).
    """
    kernel = kernel / kernel.sum()
    weight = kernel.unsqueeze(0).unsqueeze(0)          # (1, 1, kH, kW)
    if channels > 1:
        weight = weight.repeat(channels, 1, 1, 1)      # (C, 1, kH, kW) for groups=C
    return F.conv2d(x, weight, stride=stride, padding=padding, groups=channels)


def blur_frame(I_pred, kernel_pred, conv_type="conv", device=None,
               edi_pos=None, edi_neg=None, edi_c=None):
    """Apply a motion-blur kernel to a sharp frame, returning the blurred frame.

    Wraps the several blur back-ends used by the reconstruction loop into one
    switch. The spatial-conv paths replicate-pad the input first, then do a
    "valid" conv, so the output keeps the original (H, W) and the boundary is
    handled identically across them. "conv" and "conv_nondiff" are
    forward-identical (bit-for-bit); they differ ONLY in whether gradients flow
    back into the kernel (and therefore into Dxy_pc).

    conv_type:
        "fft"         FFT convolution with reflect padding
                      (utils.utility.convolve_with_fft_replicate).
        "conv"        differentiable spatial conv (conv2d_from_kernel_diff); the
                      kernel stays on the autograd graph so L_F can refine it and
                      thus Dxy_pc. [default]
        "conv_nondiff" spatial conv with a frozen kernel (conv2d_from_kernel,
                      weight detached); forward-identical to "conv" but the kernel
                      receives no gradient.
        "medi"        EDI/MEDI blur: synthesise the blur DIRECTLY from the latent
                      via the event double integral, using the contrast c and the
                      window's event bins instead of a motion kernel:
                          B = I_pred * mean_t exp( c*(E+ - E-) ).
                      Multiplicative in linear intensity, so it is INVARIANT to the
                      [0,1] vs [0,255] normalisation of I_pred (the scale cancels
                      in ln B - ln I_pred = c*E), needs no lin_log / ln(255), and
                      c stays the PHYSICAL contrast (~0.19, same as MEDI). Per-pixel
                      (correct under non-uniform / rotational flow, unlike a single
                      global kernel) and uses the RAW un-warped exposure events, so
                      c is decoupled from the flow Dxy. Because c enters this
                      forward, the frame-consistency loss L_F JOINTLY optimises c
                      with the latent, at kappa_frame=1 (both sides are [0,1]).
                      Needs edi_pos, edi_neg (2N,H,W event count bins) and edi_c
                      (scalar / Parameter); kernel_pred is ignored.  ("edi" alias.)
        "none"        no blur; returns I_pred unchanged.

    Args:
        I_pred: (H, W) sharp frame (the latent centre for the "medi" path).
        kernel_pred: (kH, kW) motion-blur kernel (unused for "medi"/"none").
        conv_type: one of {"fft", "conv", "conv_nondiff", "medi"/"edi", "none"}.
        device: device for the frozen-conv path; defaults to I_pred.device.
        edi_pos, edi_neg, edi_c: only for conv_type="medi".
    Returns:
        (H, W) blurred frame.
    """
    if conv_type == "none":
        return I_pred

    if conv_type in ("medi", "edi"):
        # EDI/MEDI forward blur from the latent centre I_pred (no motion kernel).
        if edi_pos is None or edi_neg is None or edi_c is None:
            raise ValueError("conv_type='medi' needs edi_pos, edi_neg, edi_c")
        from utils.utils_calib import edi_centered_double_integral  # lazy
        bii = edi_c * (edi_pos - edi_neg)                       # (2N, H, W)
        E = edi_centered_double_integral(bii).clamp(-20.0, 20.0)  # (2N+1, H, W)
        return I_pred * torch.exp(E).mean(dim=0)

    if conv_type == "fft":
        from utils.utility import convolve_with_fft_replicate  # lazy: avoid import cycle
        return convolve_with_fft_replicate(I_pred, kernel_pred, "reflect")

    if device is None:
        device = I_pred.device

    # spatial-conv paths: replicate-pad first so a "valid" conv keeps the size.
    # Asymmetric pad (total = k-1 per axis) keeps (H, W) for BOTH odd and even
    # kernels; for odd kernels this is identical to the symmetric k//2 split.
    nky, nkx = kernel_pred.shape
    p2d = ((nkx - 1) // 2, nkx // 2, (nky - 1) // 2, nky // 2)
    I_pred_pad = F.pad(I_pred.unsqueeze(0).unsqueeze(0), p2d, "replicate").squeeze()
    k = torch.flip(kernel_pred, [0, 1])

    if conv_type == "conv":
        return conv2d_from_kernel_diff(
            k, I_pred_pad.float().unsqueeze(0).unsqueeze(0),
            channels=1, padding="valid",
        ).squeeze()

    if conv_type == "conv_nondiff":
        filter_sr = conv2d_from_kernel(k, 1, device, padding="valid")
        return filter_sr(I_pred_pad.float().unsqueeze(0).unsqueeze(0)).squeeze()

    raise ValueError(f"unknown conv_type: {conv_type!r}")


def apply_blur(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    blurred = F.conv2d(image, kernel, padding='same' #kernel.size(-1) // 2
    )
    return blurred.squeeze()



def down_sampling(x, sigma):
    #     if not isinstance(x, torch.Tensor):
    #         x = torch.tensor(x, dtype=torch.float32)

    # Unfold with a step size equal to the block size to prevent overlap
    patches = x.unfold(0, sigma, sigma).unfold(1, sigma, sigma)
    y = patches.mean(dim=-1).mean(dim=-1)

    return y



def down_sampling_interp(x, M, mode = "bicubic"):
    h, w = x.shape[:]
    size_ds = (int(h/M),int(w/M))
    y = F.interpolate(x.unsqueeze(0).unsqueeze(0),size=size_ds, mode = mode, align_corners= True).squeeze()
    return y


def complex_regularizer(x, kappa_tv = [0.1, 0.001], kappa_l1 =[0.1, 0.001], tv_order = 2,tv_iso=True, tv_tau = 1e-3):
    reg = kappa_tv[0] * tv_loss(x, tv_order=tv_order, iso=tv_iso, tv_tau=tv_tau) \
          + kappa_l1[0] * l1_loss(1 - x.abs()) \
          + kappa_tv[1] * tv_loss_ph(x, tv_order=tv_order, iso=tv_iso, tv_tau=tv_tau) \
          + kappa_l1[1] * l1_loss(torch.exp(1j * x.angle()).imag)
    return reg


import gc

def reset_optimizer_and_params(optimizer, param_groups):
    """
    彻底清除 optimizer 的内部状态和参数梯度，并删除参数引用。
    支持 param_groups 结构形如 [{'params': param, 'lr': ...}, ...]
    """
    # 1️⃣ 遍历所有参数组
    all_params = []
    for group in param_groups:
        params = group.get('params', None)
        if params is None:
            continue
        # 如果 params 是单个 tensor 而不是列表
        if isinstance(params, torch.nn.Parameter):
            params = [params]
        all_params.extend(params)

    # 2️⃣ 清理梯度
    for p in all_params:
        if hasattr(p, "grad") and p.grad is not None:
            p.grad = None

    # 3️⃣ 清空优化器状态
    optimizer.state.clear()

    # 4️⃣ 删除参数对象（从当前命名空间）
    # 注意：无法直接在函数内部删除外部变量名，
    #       这里只能返回需要删除的对象列表。
    del all_params[:]

    # 5️⃣ 回收资源
    torch.cuda.empty_cache()
    gc.collect()

    print("Optimizer state, gradients, and parameter references have been reset.")



