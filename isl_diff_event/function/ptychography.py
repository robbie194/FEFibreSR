import glob
import math
import os
from pprint import pprint
from typing import Tuple

import cv2
import kornia
import matplotlib.pyplot as plt
import natsort
import numpy as np
# import cupy as np
#import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from numpy import fromfile
from skimage.registration import phase_cross_correlation
from torch.fft import fftshift, ifftshift, fft2, ifft2
from torch.nn.functional import conv2d, pad

import torchvision.transforms as transforms
from PIL import Image
#
# sns.color_palette("vlag", as_cmap=True)
# sns.color_palette("Spectral", as_cmap=True)
# sns.color_palette("icefire", as_cmap=True)

axoff_fun = np.vectorize(lambda ax: ax.axis('off'))

def transfer_function(wave_len=[], ps=[], img_size=[], z=[], device='cuda'):
    k0 = 2 * np.pi / wave_len
    kmax = np.pi / ps
    kxm0 = torch.linspace(-kmax, kmax, img_size, device=device, dtype=torch.float64)
    kym0 = torch.linspace(-kmax, kmax, img_size, device=device, dtype=torch.float64)
    kxm, kym = torch.meshgrid(kxm0, kym0, indexing='ij')
    kzm = torch.sqrt((k0 ** 2 - kxm ** 2 - kym ** 2).to(torch.complex64))

    H = torch.exp(1j * z * kzm.real) * torch.exp(-z.abs() * torch.imag(kzm).abs()) * (
            (k0 ** 2 - kxm ** 2 - kym ** 2) >= 0)
    H = H.type(torch.cfloat)

    return H


def RS(x, iH):
    return ifft2(ifftshift(fftshift(fft2(x)) * iH.to(x.device)))


def RS_pad(x, iH, img_size, mag):
    pad_size = int(img_size * (mag - 1) // 2)
    padded_x = F.pad(ifftshift(fft2(x)), pad=(pad_size, pad_size, pad_size, pad_size), mode='constant', value=0.0)
    return ifft2(ifftshift(padded_x * iH.to(x.device)))


def shift_frequency(img_size=[], device='cuda'):
    fy = ifftshift(
        torch.linspace(-np.fix(img_size / 2), np.ceil(img_size / 2) - 1, img_size, dtype=torch.float64, device=device))
    fx = ifftshift(
        torch.linspace(-np.fix(img_size / 2), np.ceil(img_size / 2) - 1, img_size, dtype=torch.float64, device=device))
    fx_shift, fy_shift = torch.meshgrid(fx, fy, indexing='ij')

    return fx_shift, fy_shift


def shift_forward(x, f_shift):
    return ifft2(fft2(x) * f_shift.to(x.device))


def shift_backward(x, f_shift):
    return ifft2(fft2(x) * torch.conj(f_shift.to(x.device)))


def make_sample(img_dir='', img_size=256, a_max = 0.5, p_max = 2 * np.pi, device='cpu'):
    image = Image.open(img_dir).convert('L')
    image = transforms.Resize((img_size, img_size))(image)
    image = transforms.ToTensor()(image).squeeze()
    image = 1 - image / torch.max(image)

    u_obj = torch.exp(-image * a_max) * torch.exp(1j * (p_max * (image - 0.5)))
    u_obj = u_obj.to(device)

    return u_obj


# def down_sampling(x, sigma):
#     # Yunhui's matlab implementation
#     u = torch.zeros_like(x)
#     for r in range(sigma):
#         for c in range(sigma):
#             u[::sigma, ::sigma] += x[r::sigma, c::sigma]

#     y = u[::sigma, ::sigma] / sigma / sigma

#     return y


# def down_sampling(x, sigma):
#     # Convert input to a PyTorch tensor if it's not already one
# #     if not isinstance(I, torch.Tensor):
# #         I = torch.tensor(I)
    
#     # Determine the dimensions of the downsampled image
#     M, N = x.shape
#     M_new, N_new = M // sigma, N // sigma
    
#     # Initialize the downsampled image tensor
#     y = torch.zeros((M_new, N_new), dtype=x.dtype)
    
#     # Downsample by averaging blocks of 'num x num'
#     for r in range(M_new):
#         for c in range(N_new):
#             # Sum up the pixels in each block
#             block = x[r*sigma:(r+1)*sigma, c*sigma:(c+1)*sigma]
#             y[r, c] = torch.sum(block)
    
#     # Normalize by the number of pixels in each block
#     y = y/(sigma * sigma)
    
#     return y

def down_sampling(x, sigma):
#     if not isinstance(x, torch.Tensor):
#         x = torch.tensor(x, dtype=torch.float32)
    
    # Unfold with a step size equal to the block size to prevent overlap
    patches = x.unfold(0, sigma, sigma).unfold(1, sigma, sigma)
    y = patches.mean(dim=-1).mean(dim=-1)
    
    return y


# %% ######################## Forward model ##########################
def light_scanning_capture(u_obj=[], z_os=1000e-6, ps_sensor=2e-6, wave_len=500e-9,
                           pos=[], mag=3, noise_var=0.02, device='cpu'):
    z_os = torch.as_tensor(z_os)
    ps_up = ps_sensor / mag

    img_num = pos.shape[-1]

    img_size_up = u_obj.shape[0]
    img_size = img_size_up // mag

    H_os = transfer_function(wave_len=wave_len, ps=ps_up, img_size=img_size_up, z=z_os, device=device)
    holo_hr = RS(u_obj, H_os).abs() ** 2

    holo_lr = torch.zeros(img_num, img_size, img_size, dtype=torch.float32, device=device)
    holo_lr[0, :, :] = down_sampling(holo_hr, mag)

    fx_shift_up, fy_shift_up = shift_frequency(img_size=img_size_up, device=device)
    
    for idx_img in range(1, img_num):
        f_shift = torch.exp(1j * 2 * np.pi * (fx_shift_up * pos[0, idx_img] / img_size +
                                              fy_shift_up * pos[1, idx_img] / img_size))
        
        holo_hr_shift = shift_forward(holo_hr, f_shift).abs()
        holo_hr_shift = holo_hr_shift + torch.randn_like(holo_hr_shift) * noise_var

        holo_lr[idx_img, :, :] = down_sampling(holo_hr_shift, mag)

    return holo_lr


def holo_AD(u_o=[], z_os=1000e-6, ps=2e-6, wave_len=500e-9, device='cpu'): #参数：u_o: 输入对象（复数张量），表示物体的复振幅分布。z_os: 焦距，默认为 1000e-6 米。ps: 传感器像素大小，默认为 2e-6 米。wave_len: 光波波长，默认为 500e-9 米。device: 计算设备，默认为 CPU。
    img_size = u_o.shape[0]

    H_os = transfer_function(wave_len=wave_len, ps=ps, img_size=img_size, z=z_os, device=device)
    holo_pred = RS(u_o, H_os).abs() ** 2

    # c = (holo_pred.flatten() * holo.flatten()).sum() / (holo_pred.flatten() * holo_pred.flatten()).sum()
    # c = holo.mean()/holo_pred.mean()
    # c = holo_pred.mean() / holo.mean()
    # holo_pred = c * holo_pred

    return holo_pred


# def psr_forward(holo_hr=[], pos=[], mag=3, device='cpu'):
#     img_num = pos.shape[-1]
#     img_size_up = holo_hr.shape[0]
#     img_size = img_size_up // mag

#     fx_shift_hr, fy_shift_hr = shift_frequency(img_size=img_size_up, device=device)

#     holo_lr = torch.zeros(img_size, img_size, img_num, device=device)
#     if mag > 1:
# #         holo_lr[:, :, 0] = down_sampling(holo_hr, mag)
#         I_sensor_down = down_sampling(holo_hr, mag)        
#         I_sensor_down_up = F.interpolate(I_sensor_down[(None,)*2], size=(img_size_up, img_size_up), mode='bilinear').squeeze()
#         I_raw_up = F.interpolate(img_raw[:, :, 0][(None,)*2], size=(img_size_up, img_size_up), mode='bilinear').squeeze()
#         c = I_raw_up.mean()/I_sensor_down_up.mean()
#         holo_lr[:, :, 0] = down_sampling(c*holo_hr, mag)
#     else:
#         holo_lr[:, :, 0] = holo_hr

#     for idx in range(1, img_num):
#         f_shift = torch.exp(1j * 2 * np.pi * (fx_shift_hr * pos[0, idx] / img_size +
#                                               fy_shift_hr * pos[1, idx] / img_size))

#         holo_hr_shift = shift_forward(holo_hr, f_shift).real.abs()
#         holo_lr[:, :, idx] = down_sampling(holo_hr_shift, mag)

#     return holo_lr

def psr_forward(holo_hr=[], pos=[], mag=3, device='cpu'):
    img_num = pos.shape[-1]
    img_size_up = holo_hr.shape[0]
    img_size = img_size_up // mag

    fx_shift_hr, fy_shift_hr = shift_frequency(img_size=img_size_up, device=device)

    holo_lr = torch.zeros(img_num,  img_size, img_size, device=device)
    if mag > 1:
        holo_lr[0, :, :] = down_sampling(holo_hr, mag)
    else:
        holo_lr[0, :, :] = holo_hr

    for idx in range(1, img_num):
        f_shift = torch.exp(1j * 2 * np.pi * (fx_shift_hr * pos[0, idx] / img_size_up +
                                              fy_shift_hr * pos[1, idx] / img_size_up))

        holo_hr_shift = shift_forward(holo_hr, f_shift).real.abs()
        holo_lr[idx, :, :] = down_sampling(holo_hr_shift, mag)
                

    return holo_lr


# def PSR_AD(u_o=[], img_raw=[], z_os=1000e-6, ps_sensor=2e-6, wave_len=500e-9, pos=[], mag=3, device='cpu'):
#     ps_up = ps_sensor / mag
#     img_num = pos.shape[-1]
#     img_size_up = u_o.shape[0]
#     img_size = img_size_up // mag

#     fx_shift_up, fy_shift_up = shift_frequency(img_size=img_size_up, device=device)
#     H_os = transfer_function(wave_len=wave_len, ps=ps_up, img_size=img_size_up, z=z_os, device=device)
#     holo_hr = RS(u_o, H_os).abs() ** 2

#     holo_lr = torch.zeros(img_size, img_size, img_num, dtype=torch.float64, device=device)
#     if mag > 1:
#         holo_lr[:, :, 0] = down_sampling(holo_hr, mag)
#     else:
#         holo_lr[:, :, 0] = holo_hr

#     c = torch.tensor(1.0, device=device)
#     for idx_img in range(1, img_num):
#         f_shift = torch.exp(1j * 2 * np.pi * (fx_shift_up * pos[0, idx_img] / img_size_up +
#                                               fy_shift_up * pos[1, idx_img] / img_size_up))

#         holo_hr_shift = shift_forward(holo_hr, f_shift).real.abs()
#         I_sensor_down = down_sampling(holo_hr_shift, mag)

#         I_recons = I_sensor_down.flatten()
#         I_raw = img_raw[:, :, idx_img].flatten()
#         c = (I_recons * I_raw).sum() / (I_recons * I_recons).sum()
#         holo_lr[:, :, idx_img] = c * I_sensor_down

#     return holo_lr, c




def PSR_AD(u_o=[], img_raw=[], z_os=1000e-6, ps_sensor=2e-6, wave_len=500e-9, pos=[], mag=3, device='cpu'):
    ps_up = ps_sensor / mag
    img_num = pos.shape[-1]
    img_size_up = u_o.shape[0]
    img_size = img_size_up // mag

    fx_shift_up, fy_shift_up = shift_frequency(img_size=img_size_up, device=device) 

    holo_lr = torch.zeros(img_num, img_size, img_size, dtype=torch.float64, device=device)       
        
    H_os = transfer_function(wave_len=wave_len, ps=ps_up, img_size=img_size_up, z=z_os, device=device)
    
    holo_hr = RS(u_o, H_os).abs() ** 2
    
    if mag > 1:
#         holo_lr[:, :, 0] = down_sampling(holo_hr, mag)        
        I_sensor_down = down_sampling(holo_hr, mag)        
        I_sensor_down_up = F.interpolate(I_sensor_down[(None,)*2], size=(img_size_up, img_size_up), mode='bilinear').squeeze()
        I_raw_up = F.interpolate(img_raw[0][(None,)*2], size=(img_size_up, img_size_up), mode='bilinear').squeeze()
        c = I_raw_up.mean()/I_sensor_down_up.mean()
        holo_lr[ 0] = down_sampling(c*holo_hr, mag)
        
    else:
        holo_lr[0, :, :] = holo_hr

    c = torch.tensor(1.0, device=device)
    for idx_img in range(1, img_num):
        f_shift = torch.exp(1j * 2 * np.pi * (fx_shift_up * pos[0, idx_img] / img_size_up +
                                              fy_shift_up * pos[1, idx_img] / img_size_up))
        
        u_o_shift = shift_forward(u_o, f_shift)
        holo_hr = RS(u_o_shift, H_os).abs() ** 2

        I_sensor_down = down_sampling(holo_hr, mag)
        
        I_sensor_down_up = F.interpolate(I_sensor_down[(None,)*2], size=(img_size_up, img_size_up), mode='bilinear').squeeze()
        I_raw_up = F.interpolate(img_raw[idx_img][(None,)*2], size=(img_size_up, img_size_up), mode='bilinear').squeeze()
        c = I_raw_up.mean()/I_sensor_down_up.mean()
        holo_lr[idx_img] = down_sampling(c*holo_hr, mag)
        
#         I_recons = I_sensor_down.flatten()
#         I_raw = img_raw[:, :, idx_img].flatten()
#         c = (I_recons * I_raw).sum() / (I_recons * I_recons).sum()
#         holo_lr[:, :, idx_img] = c * I_sensor_down
        

    return holo_lr, c



def PSR_GS(u_o=[], img_raw=[],
           z_os=1000e-6, ps_sensor=2e-6, wave_len=500e-9,
           pos=[], mag=3, device='cpu'):
    with torch.no_grad():
        ps_up = ps_sensor / mag
        img_num = pos.shape[-1]
        img_size_up = u_o.shape[0]
        img_size = img_size_up // mag

        # subpixel shift parameters
        fx_shift_up, fy_shift_up = shift_frequency(img_size=img_size_up, device=device)

        H_os = transfer_function(wave_len=wave_len, ps=ps_up, img_size=img_size_up, z=z_os, device=device)
        err = 0
        for idx_img in range(img_num):
            f_shift = torch.exp(1j * 2 * np.pi * (fx_shift_up * pos[0, idx_img] / img_size +
                                                  fy_shift_up * pos[1, idx_img] / img_size))
            u_obj_shift = shift_forward(u_o, f_shift)

            u_sensor_up = RS(u_obj_shift, H_os)
            holo_hr = torch.abs(u_sensor_up) ** 2

            I_sensor_down = down_sampling(holo_hr, mag)

            err += torch.mean(torch.square(torch.sqrt(I_sensor_down) - torch.sqrt(img_raw[idx_img])))

            # mag_ratio_map = torch.sqrt(img_raw[:, :, idx_img]) / torch.sqrt(I_sensor_down)
            # mag_ratio_map = F.interpolate(mag_ratio_map[(None,)*2], size=(img_size_up, img_size_up),
            #                               mode='nearest-exact').squeeze()
            # I_calc_down = DownSampling(I_calc, mag);
            # I_calc_up = imresize(I_calc_down, mag, 'nearest');
            # I_measure_up = imresize(I_measure, mag, 'nearest');
            # I = I_calc. * I_measure_up. / I_calc_up;

            holo_hr = F.interpolate(I_sensor_down[(None,) * 2], size=(img_size_up, img_size_up), mode='nearest').squeeze()
            raw_up = F.interpolate(img_raw[idx_img][(None,) * 2], size=(img_size_up, img_size_up),
                                   mode='nearest').squeeze()
            mag_ratio_map = torch.sqrt(raw_up) / torch.sqrt(holo_hr)

            u_sensor_up_update = u_sensor_up * mag_ratio_map

            u_obj_shift_update = RS(u_sensor_up_update, H_os.conj())

            # I_tmp = u_obj_shift_update.abs()**2
            # Is = gaussian_blur(I_tmp, (3, 3), (0.3, 0.3))
            # alpha = 0.1
            # u_obj_shift_update = (1 - alpha) * u_obj_shift_update + alpha * torch.sqrt(Is) * torch.exp(1j * u_obj_shift_update.angle())

            u_o = shift_backward(u_obj_shift_update, f_shift)

        # x_abs = u_o.abs()
        # x_ang = u_o.angle()
        # x_absorb = -torch.log(x_abs)
        # x_ang[x_absorb < 0] = 0.0
        # x_absorb[x_absorb < 0] = 0.0
        # x_abs = torch.exp(-x_absorb)
        # u_o = x_abs * torch.exp(1j * x_ang)

    return u_o, err





def track_position(img_seq=[], is_mask=1):
    if is_mask:
        img_seq = img_seq / torch.mean(img_seq, dim=0, keepdim=True)

    img_ref = img_seq[0]
    img_num = img_seq.size(0)
    pos = np.zeros((2, img_num))
    edge_ignore1 = 10
    standardImg = img_ref[edge_ignore1:-edge_ignore1, edge_ignore1:-edge_ignore1]

    print('calculating position...')
    for idx in range(1, img_num):
        copyImg = img_seq[idx, edge_ignore1:-edge_ignore1, edge_ignore1:-edge_ignore1]
        pos[:, idx], error, diffphase = phase_cross_correlation(standardImg.cpu().numpy(), copyImg.cpu().numpy(),
                                                                upsample_factor=100, space='real', normalization=None)
    
    return pos


def refine_position(img_seq=[], pos=[], is_mask=1):
    device = img_seq.device
    if is_mask:
        img_seq = img_seq / torch.mean(img_seq, dim=0, keepdim=True)

    img_size = img_seq.size(-1)
    img_num = img_seq.size(0)

#     fx_shift, fy_shift = shift_frequency(img_size=img_size, device=device)
#     img_refine = torch.zeros(img_size, img_size, dtype=torch.float64, device=device)
# #     padded_image = F.pad(image, (left_pad, right_pad, top_pad, bottom_pad))
#     for idx in range(img_num):
#         Hs = torch.exp(-1j * 2 * np.pi * (fx_shift * pos[0, idx] / img_size + fy_shift * pos[1, idx] / img_size)).to(
#             device)
#         img_refine += (ifft2(fft2(img_seq[:, :, idx]) * Hs)).real.abs()
#     img_refine = img_refine / img_num

    img_refine = img_seq[0, :, :]

    # Refine the subpixel shifts
    pos = np.zeros((2, img_num))
    edge_ignore1 = 10
    # print('Refine position...')
    img_reference = img_refine[edge_ignore1:-edge_ignore1, edge_ignore1:-edge_ignore1]
    for idx in range(1, img_num):
        copyImg = img_seq[idx, edge_ignore1:-edge_ignore1, edge_ignore1:-edge_ignore1]
        pos[:, idx], error, diffphase = phase_cross_correlation(img_reference.cpu().numpy(), copyImg.cpu().numpy(),
                                                                upsample_factor=100, space='real', normalization=None)


    return pos


def pos_track(holo_raw, iter=3, is_mask=0):
    
    pos_init = track_position(img_seq=holo_raw, is_mask=is_mask)
    pos_refine = pos_init
    for iLoc in range(iter):
        pos_refine = refine_position(img_seq=holo_raw, pos=pos_refine, is_mask=is_mask)

    return pos_refine


def img_reg(holo_raw=[], pos=[], img_size=128, img_num=100, device='cpu', **kwargs):
    fx, fy = shift_frequency(img_size=img_size, device=device)
    holo_raw_sum = torch.zeros(img_size, img_size, device=device)
    for idx_img in range(img_num):
        f_shift = torch.exp(-1j * 2 * np.pi * (fx * pos[0, idx_img] / img_size + fy * pos[1, idx_img] / img_size)).to(
            device=device)
#         holo_raw_sum += shift_forward(holo_raw[:, :, idx_img], f_shift).abs()
        holo_raw_sum += shift_forward(holo_raw[idx_img, :, :], f_shift).real.abs()

    holo_raw_avg = holo_raw_sum / img_num

    return holo_raw_avg


def FT2(field_in, pps=1):
    Ny, Nx = field_in.shape
    x_range = torch.linspace(0, Nx - 1, Nx).to(field_in.device)
    y_range = torch.linspace(0, Ny - 1, Ny).to(field_in.device)

    y, x = torch.meshgrid(y_range, x_range, indexing='ij')

    shift_phase = torch.exp(1j * math.pi * (x + y))

    return fft2(shift_phase * field_in) * shift_phase * pps ** 2


def iFT2(field_in, ppf=1):
    Ny, Nx = field_in.shape
    x_range = torch.linspace(0, Nx - 1, Nx).to(field_in.device)
    y_range = torch.linspace(0, Ny - 1, Ny).to(field_in.device)

    y, x = torch.meshgrid(y_range, x_range, indexing='ij')

    shift_phase = torch.exp(-1j * math.pi * (x + y))

    return ifft2(shift_phase * field_in) * shift_phase / (ppf ** 2)


def gaussian(window_size, sigma):
    def gauss_fcn(x):
        return -(x - window_size // 2) ** 2 / float(2 * sigma ** 2)

    gauss = torch.stack([torch.exp(torch.as_tensor(gauss_fcn(x))) for x in range(window_size)])

    return gauss / gauss.sum()


def get_gaussian_kernel(ksize: int, sigma: float) -> torch.Tensor:
    r"""Function that returns Gaussian filter coefficients.

    Args:
        ksize (int): filter size. It should be odd and positive.
        sigma (float): gaussian standard deviation.

    Returns:
        Tensor: 1D tensor with gaussian filter coefficients.

    Shape:
        - Output: :math:`(ksize,)`

    Examples::
        >>> tgm.image.get_gaussian_kernel(3, 2.5)
        tensor([0.3243, 0.3513, 0.3243])

        >>> tgm.image.get_gaussian_kernel(5, 1.5)
        tensor([0.1201, 0.2339, 0.2921, 0.2339, 0.1201])
    """
    if not isinstance(ksize, int) or ksize % 2 == 0 or ksize <= 0:
        raise TypeError("ksize must be an odd positive integer. Got {}"
                        .format(ksize))
    window_1d: torch.Tensor = gaussian(ksize, sigma)
    return window_1d


def get_gaussian_kernel2d(ksize: Tuple[int, int], sigma: Tuple[float, float]) -> torch.Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.

    Args:
        ksize (Tuple[int, int]): filter sizes in the x and y direction.
        Sizes should be odd and positive.
        sigma (Tuple[int, int]): gaussian standard deviation in the x and y direction.

    Returns:
        Tensor: 2D tensor with gaussian filter matrix coefficients.

    Shape:
        - Output: :math:`(ksize_x, ksize_y)`

    Examples::
        >>> tgm.image.get_gaussian_kernel2d((3, 3), (1.5, 1.5))
        tensor([[0.0947, 0.1183, 0.0947],
                [0.1183, 0.1478, 0.1183],
                [0.0947, 0.1183, 0.0947]])

        >>> tgm.image.get_gaussian_kernel2d((3, 5), (1.5, 1.5))
        tensor([[0.0370, 0.0720, 0.0899, 0.0720, 0.0370],
                [0.0462, 0.0899, 0.1123, 0.0899, 0.0462],
                [0.0370, 0.0720, 0.0899, 0.0720, 0.0370]])
    """
    if not isinstance(ksize, tuple) or len(ksize) != 2:
        raise TypeError("ksize must be a tuple of length two. Got {}".format(ksize))
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError("sigma must be a tuple of length two. Got {}".format(sigma))
    ksize_x, ksize_y = ksize
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = get_gaussian_kernel(ksize_x, sigma_x)
    kernel_y: torch.Tensor = get_gaussian_kernel(ksize_y, sigma_y)
    kernel_2d: torch.Tensor = torch.matmul(kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())

    return kernel_2d


class GaussianBlur(nn.Module):
    r"""Creates an operator that blurs a tensor using a Gaussian filter.

    The operator smooths the given tensor with a gaussian kernel by convolving
    it to each channel. It suports batched operation.

    Arguments:
        kernel_size (Tuple[int, int]): the size of the kernel.
        sigma (Tuple[float, float]): the standard deviation of the kernel.

    Returns:
        Tensor: the blurred tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples::
        >>> input = torch.rand(2, 4, 5, 5)
        >>> gauss = tgm.image.GaussianBlur((3, 3), (1.5, 1.5))
        >>> output = gauss(input)  # 2x4x5x5
    """

    def __init__(self, kernel_size: Tuple[int, int],
                 sigma: Tuple[float, float]) -> None:
        super(GaussianBlur, self).__init__()
        self.kernel_size: Tuple[int, int] = kernel_size
        self.sigma: Tuple[float, float] = sigma
        self._padding: Tuple[int, int] = self.compute_zero_padding(kernel_size)
        self.kernel: torch.Tensor = self.create_gaussian_kernel(kernel_size, sigma)

    @staticmethod
    def create_gaussian_kernel(kernel_size, sigma) -> torch.Tensor:
        """Returns a 2D Gaussian kernel array."""
        kernel: torch.Tensor = get_gaussian_kernel2d(kernel_size, sigma)
        return kernel

    @staticmethod
    def compute_zero_padding(kernel_size: Tuple[int, int]) -> Tuple[int, int]:
        """Computes zero padding tuple."""
        computed = [(k - 1) // 2 for k in kernel_size]
        return computed[0], computed[1]

    def forward(self, x: torch.Tensor):
        if not torch.is_tensor(x):
            raise TypeError("Input x type is not a torch.Tensor. Got {}".format(type(x)))
        if not len(x.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}".format(x.shape))
        # prepare kernel
        b, c, h, w = x.shape
        tmp_kernel: torch.Tensor = self.kernel.to(x.device).to(x.dtype)
        kernel: torch.Tensor = tmp_kernel.repeat(c, 1, 1, 1)

        # convolve tensor with gaussian kernel
        return conv2d(pad(x, (*self._padding, *self._padding), mode='circular'), kernel, stride=1, groups=c)


######################
# functional interface
######################
def gaussian_blur(src: torch.Tensor,
                  kernel_size: Tuple[int, int],
                  sigma: Tuple[float, float]) -> torch.Tensor:
    r"""Function that blurs a tensor using a Gaussian filter.

    The operator smooths the given tensor with a gaussian kernel by convolving
    it to each channel. It suports batched operation.

    Arguments:
        src (Tensor): the input tensor.
        kernel_size (Tuple[int, int]): the size of the kernel.
        sigma (Tuple[float, float]): the standard deviation of the kernel.

    Returns:
        Tensor: the blurred tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples::
        >>> input = torch.rand(2, 4, 5, 5)
        >>> output = tgm.image.gaussian_blur(input, (3, 3), (1.5, 1.5))
    """
    N = len(src.shape)
    if N == 2:
        return torch.squeeze(GaussianBlur(kernel_size, sigma)(src[None, None, ...]))
    elif N == 3:
        return torch.squeeze(GaussianBlur(kernel_size, sigma)(src[None, ...]))
    elif N >= 4:
        return GaussianBlur(kernel_size, sigma)(src)




def torch_mode(x):
    # Flatten the input tensor to 1D
    x = x.view(-1)

    # Calculate unique values and their counts
    unique_vals, counts = torch.unique(x, return_counts=True)

    # Find the maximum count and its index
    max_count_idx = torch.argmax(counts)

    # Get the mode value(s)
    mode_val = unique_vals[max_count_idx]

    # Get the count(s) of the mode value(s)
    mode_count = counts[max_count_idx]

    return mode_val.item(), mode_count.item()


def numpy_mode(x):
    # Flatten the input array to 1D
    x = x.flatten()

    # Calculate unique values and their counts
    unique_vals, counts = np.unique(x, return_counts=True)

    # Find the maximum count and its index
    max_count_idx = np.argmax(counts)

    # Get the mode value(s)
    mode_val = unique_vals[max_count_idx]

    # Get the count(s) of the mode value(s)
    mode_count = counts[max_count_idx]

    return mode_val.item(), mode_count.item()


def plane_wave(theta_x=0.01, theta_y=0.05, img_size=512, pixel_size=1e-6, wave_len=0.405e-6, device='cpu'):
    PI = torch.tensor(np.pi)
    Y = (torch.linspace(-np.fix(img_size / 2), np.ceil(img_size / 2) - 1, img_size, dtype=torch.float32))
    X = (torch.linspace(-np.fix(img_size / 2), np.ceil(img_size / 2) - 1, img_size, dtype=torch.float32))
    x, y = torch.meshgrid(X * pixel_size, Y * pixel_size, indexing='ij')

    theta_x = PI / 180 * theta_x
    theta_y = PI / 180 * theta_y

    wave = torch.exp(1j * 2 * PI / wave_len * (x * torch.sin(theta_x) + y * torch.sin(theta_y)))

    return wave.to(device)


def load_data(data_dir='', file_type='tiff', img_num=None, img_size=None, is_norm=True, shift=[0, 0]):
    files = glob.glob(data_dir + f"*.{file_type}")
    files = natsort.natsorted(files)
    if file_type == 'raw':
        width = 4200
        height = 3120
        img = fromfile(files[0], dtype=np.uint16).reshape((height, width))
    else:
        img = cv2.imread(files[0], -1)
        if len(img.shape)>2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    Ny, Nx = img.shape
    Nyc, Nxc = (Ny // 2), (Nx // 2)

    if img_num == None:
        img_num = len(files)

    img_raw = np.zeros((img_num, img_size, img_size))
    # img_contrast = np.zeros(img_num)
    # img_mean = np.zeros(img_num)
    for idx, file in enumerate(files):
        if idx >= img_num:
            break

        if file_type == 'raw':
            img_full = fromfile(file, dtype=np.uint16).reshape((height, width))
            img_full = img_full / (2 ** 16 - 1)
        else:
            img_full = cv2.imread(file, -1)
            if len(img_full.shape) > 2:
                img_full = cv2.cvtColor(img_full, cv2.COLOR_BGR2GRAY)

            if img_full.dtype == np.uint16:
                img_full = img_full / (2 ** 16 - 1)
            elif img_full.dtype == np.uint8:
                img_full = img_full / (2 ** 8 - 1)

        y_shift = shift[0]
        x_shift = shift[1]
        img = img_full[Nyc - img_size // 2 + y_shift:Nyc + img_size // 2 + y_shift,
              Nxc - img_size // 2 + x_shift:Nxc + img_size // 2 + x_shift]

        # if is_norm:
        #     img = img / np.median(img)
        # img = img / np.mean(img)

        # img_mean[idx] = img.mean()
        # img_contrast[idx] = img.std() / img.mean()
        img_raw[idx, :, :] = img

    if is_norm:
        img_raw = img_raw / np.median(img_raw)

    img_raw = torch.tensor(img_raw)

    return img_raw


def raw_resize(img_raw=[], mag=1):
    img_size = img_raw.shape[0]
    img_size_up = int(img_size * mag)
    img_num = img_raw.shape[2]

    img_raw_up = torch.zeros((img_num, img_size_up, img_size_up), dtype=torch.float64, device=img_raw.device)
    for idx in range(img_num):
        img = img_raw[idx, :, :]
#         img_raw_up[:, :, idx] = F.interpolate(img[(None,) * 2], size=(img_size_up, img_size_up),
#                                               mode='nearest-exact').squeeze()
        img_raw_up[idx, :, :] = F.interpolate(img[(None,) * 2], size=(img_size_up, img_size_up),
                                              mode='nearest').squeeze()

    return img_raw_up


######################################### Scanning pattern of the translation stage #########################################

def line_pattern(Ny, Nx):
    x_range = (np.linspace(0, Nx - 1, Nx) - (Nx + 1) / 2)
    y_range = (np.linspace(0, Ny - 1, Ny) - (Ny + 1) / 2)

    # positon in move steps
    y, x = np.meshgrid(y_range, x_range)

    # line by line scanning position
    pos_line = np.vstack([x.ravel(), y.ravel()])

    return pos_line


def zigzag_pattern(Ny, Nx):
    x_range = (np.linspace(0, Nx - 1, Nx) - (Nx + 1) / 2)
    y_range = (np.linspace(0, Ny - 1, Ny) - (Ny + 1) / 2)

    # positon in move steps
    y, x = np.meshgrid(y_range, x_range)

    # zigzag scanning position
    y[1::2] = y[1::2, ::-1]
    pos_zigzag = np.vstack([x.ravel(), y.ravel()])

    return pos_zigzag


def spiral(width, height):
    N, S, W, E = (0, -1), (0, 1), (-1, 0), (1, 0)  # directions
    turn_right = {N: E, E: S, S: W, W: N}
    turn_left = {N: W, W: S, S: E, E: N}
    if width < 1 or height < 1:
        raise ValueError
    x, y = width // 2, height // 2  # start near the center
    dx, dy = N  # initial direction
    matrix = [[None] * width for _ in range(height)]
    count = 0
    if width % 2 == 0:
        x = x - 1
    if height % 2 == 0:
        y = y - 1
    while True:
        count += 1
        matrix[y][x] = count  # visit
        # try to turn right
        new_dx, new_dy = turn_right[dx, dy]
        new_x, new_y = x + new_dx, y + new_dy
        if (0 <= new_x < width and 0 <= new_y < height and
                matrix[new_y][new_x] is None):  # can turn right
            x, y = new_x, new_y
            dx, dy = new_dx, new_dy
        else:  # try to move straight
            x, y = x + dx, y + dy
            if not (0 <= x < width and 0 <= y < height):
                return np.array(matrix)  # nowhere to go


def spiral_pattern(Ny, Nx):
    # spiral scanning position
    total_num = Nx * Ny
    routescan = np.rot90(spiral(Ny, Nx), 3)
    pos_spiral = np.zeros((2, total_num))

    for i in range(total_num):
        y, x = np.where(routescan == i + 1)
        pos_spiral[0, i] = (y + 1 - (Ny + 1) / 2)
        pos_spiral[1, i] = (x + 1 - (Nx + 1) / 2)

    return pos_spiral

######################################### Configure file #########################################
def save_config_to_file(params, file_path):
    with open(file_path, 'w') as file:
        yaml.dump(params, file, default_flow_style=False, sort_keys=False, indent=4)


def load_config_from_file(file_path):
    with open(file_path, "r") as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def set_config_file(params={}, file_path='', is_debug=True):
    if is_debug:
        if os.path.exists(file_path):
            sys_params = load_config_from_file(file_path)
            params = {**sys_params, **params}
            print("Update the config file.")
        else:
            print("Config file does not exist! Creat it.")

        save_config_to_file(params, file_path)
    else:
        print("Load the config file!")
        params = load_config_from_file(file_path)
        pprint(params)

    return params


######################################### autofocusing #########################################

def autofocus(recons_stack=[], z_range=[], type='laplacian'):
    '''
    Reference:
        - Shaowei Jiang, et.al., ACS Photonics 8(11), 3261-3271, 2021
    '''

    pos_num = len(z_range)
    edge_values = torch.zeros(pos_num)
    for idx in range(pos_num):
        img_plane = recons_stack[idx, :, :]
        if type == 'laplacian':
            edges = kornia.filters.laplacian(img_plane[(None,) * 2], kernel_size=3).squeeze()
        elif type == 'sobel':
            edges = kornia.filters.sobel(img_plane[(None,) * 2]).squeeze()

        # if idx in [10]:
        #     # plt.imshow(edges.real.abs().cpu()), plt.colorbar(), plt.show()
        #     # plt.imshow(edges.angle().cpu()), plt.colorbar(), plt.show()
        #     plt.imshow(img_plane.angle().cpu()), plt.colorbar(), plt.show()
        #     plt.imshow(torch.atan(img_plane.imag/img_plane.real).cpu()), plt.colorbar(), plt.show()

        edge_values[idx] = torch.sum(edges.abs(), dim=(-1, -2))

    edge_values = (edge_values - edge_values.min()) / (edge_values.max() - edge_values.min())
    _, z_index = torch.min(edge_values, 0)
    z_focus = z_range[z_index]

    return z_focus, recons_stack[z_index, :, :], edge_values


def foucs_search(holo, ps=1.1e-6, wave_len=405e-9,
                 z_init=500e-6, z_min=0, z_max=1000e-6, z_scope=100e-6, z_num=101, device='cpu'):
    img_size = holo.shape[0]

    z_recons = torch.linspace(z_min, z_max, z_num, device=device, dtype=torch.float64)
    recons_stack = torch.zeros((z_recons.shape[0]), img_size, img_size, device=device, dtype=torch.complex64)
    for idx in range(len(z_recons)):
        H_os = transfer_function(wave_len=wave_len, ps=ps, img_size=img_size, z=z_recons[idx], device=device)
        recons_stack[idx, :, :] = RS(holo, H_os.conj())
    z_focus, recons_focus, edge_values = autofocus(recons_stack, z_recons, type='laplacian')

    z_min_init = z_min
    z_max_init = z_max
    while z_focus >= z_min_init and z_focus <= z_max_init:
        z_min = z_init - z_scope / 2
        z_max = z_init + z_scope / 2
        z_recons = torch.linspace(z_min, z_max, z_num, device=device, dtype=torch.float64)
        recons_stack = torch.zeros((z_recons.shape[0]), img_size, img_size, device=device, dtype=torch.complex64)
        for idx in range(len(z_recons)):
            H_os = transfer_function(wave_len=wave_len, ps=ps, img_size=img_size, z=z_recons[idx], device=device)
            recons_stack[idx, :, :] = RS(holo, H_os.conj())
        z_focus, recons_focus, edge_values = autofocus(recons_stack, z_recons, type='laplacian')

        if z_focus < z_init - z_scope / 4:
            z_init = z_init - z_scope / 4
        elif z_focus > z_init + z_scope / 4:
            z_init = z_init + z_scope / 4
        else:
            break

        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        ax.plot(z_recons.cpu() * 1e6, edge_values.cpu(), label=f'{z_focus.cpu() * 1e6:.1f} um')
        ax.set_ylabel(r'Normalized critical value')
        ax.set_xlabel(r'z [um]')
        ax.axvline(x=z_focus.cpu() * 1e6, color='g', ls='-')
        ax.text(z_focus.cpu() * 1e6, 0.9, f' z={z_focus * 1e6:.1f}', rotation=0)
        plt.show()
    else:
        print("focus is out of scope!")

    fig, ax1 = plt.subplots(1, 1, figsize=(5, 4))
    ax1.plot(z_recons.cpu() * 1e6, edge_values.cpu(), label=f'{z_focus.cpu() * 1e6:.1f} um')
    ax1.set_ylabel(r'Normalized critical value')
    ax1.set_xlabel(r'z [um]')
    ax1.axvline(x=z_focus.cpu() * 1e6, color='g', ls='-')
    ax1.text(z_focus.cpu() * 1e6, 0.9, f' z={z_focus * 1e6:.1f}', rotation=0)
    plt.show()

    return z_focus.cpu().detach().numpy().item(), recons_focus
