# -*- coding: utf-8 -*-
"""
Phase unwrap.
@author: Ni Chen (https://ni-chen.github.io/)
"""

import math

import numpy as np
import torch


# def unwrap(p, discont=np.pi, axis=-1):
#     nd = len(p.shape)
#     dd = torch.diff(p, dim=axis)
#     slice1 = [slice(None, None)] * nd  # full slices
#     slice1[axis] = slice(1, None)

#     ddmod = torch.remainder(dd + np.pi, 2 * np.pi) - np.pi

#     ddmod[(ddmod == -np.pi) & (dd > 0)] = torch.tensor(np.pi)
#     ph_correct = ddmod - dd
#     ph_correct[abs(dd) < discont] = torch.tensor(0)

#     up = p.clone()
#     up[slice1] = p[slice1] + ph_correct.cumsum(axis)

#     return up


# # wrap function
# def wrap(x):
#     return torch.arctan2(torch.sin(x), torch.cos(x))


# # unwrap function
# def wrap_diff(x):
#     return wrap(torch.diff(x))


# def unwrap_raster(x):
#     """one dimensional unwrap for itoh's raster algrithom"""
#     y = x
#     y[0] = x[0]
#     for i in range(len(x) - 1):
#         i += 1
#         y[i] = y[i - 1] + wrap_diff(x)[i - 1]

#     return y



# def FT2(field_in, pps=1):
#     Ny, Nx = field_in.shape
#     x_range = torch.linspace(0, Nx - 1, Nx).to(field_in.device)
#     y_range = torch.linspace(0, Ny - 1, Ny).to(field_in.device)

#     y, x = torch.meshgrid(y_range, x_range, indexing='ij')

#     shift_phase = torch.exp(1j * math.pi * (x + y))

#     return torch.fft.fft2(shift_phase * field_in)*shift_phase * pps**2

# def iFT2(field_in, ppf=1):
#     Ny, Nx = field_in.shape
#     x_range = torch.linspace(0, Nx - 1, Nx).to(field_in.device)
#     y_range = torch.linspace(0, Ny - 1, Ny).to(field_in.device)

#     y, x = torch.meshgrid(y_range, x_range, indexing='ij')

#     shift_phase = torch.exp(-1j * math.pi * (x + y))

#     return torch.fft.ifft2(shift_phase * field_in) * shift_phase / (ppf**2)


# def myunwrap(in_phase):
#     # x = x.cpu().detach().numpy()

#     Ny, Nx = in_phase.shape

#     x_range = torch.linspace(0, Nx - 1, Nx) - np.floor(Nx / 2)
#     y_range = torch.linspace(0, Ny - 1, Ny) - np.floor(Ny / 2)
#     y, x = torch.meshgrid(y_range, x_range, indexing='ij')

#     f = x**2 + y**2

#     # a = iFT2(FT2(torch.cos(in_phase)*iFT2(FT2(torch.sin(in_phase))*f))/(f+1e-6))
#     # b = iFT2(FT2(torch.sin(in_phase)*iFT2(FT2(torch.cos(in_phase))*f))/(f+1e-6))
#     a = iFT2(FT2(torch.cos(in_phase)*iFT2(FT2(torch.sin(in_phase))*f))/(f + torch.finfo(torch.float32).eps*100))
#     b = iFT2(FT2(torch.sin(in_phase)*iFT2(FT2(torch.cos(in_phase))*f))/(f + torch.finfo(torch.float32).eps*100))

#     y = (a - b).real

#     return y





################################################################################################################
"""
A weighed phase unwrap algorithm implemented in pure Python

author: Tobias A. de Jong
Based on:
Ghiglia, Dennis C., and Louis A. Romero. 
"Robust two-dimensional weighted and unweighted phase unwrapping that uses 
fast transforms and iterative methods." JOSA A 11.1 (1994): 107-117.
URL: https://doi.org/10.1364/JOSAA.11.000107
and an existing MATLAB implementation:
https://nl.mathworks.com/matlabcentral/fileexchange/60345-2d-weighted-phase-unwrapping
Should maybe use a scipy conjugate descent.
"""

import numpy as np
from scipy.fft import dctn, idctn

# def phase_unwrap_ref(psi, weight, kmax=100):
#     # vector b in the paper (eq 15) is dx and dy
#     dx = _wrapToPi(np.diff(psi, axis=1))
#     dy = _wrapToPi(np.diff(psi, axis=0))

#     # multiply the vector b by weight square (W^T * W)
#     WW = weight ** 2

#     # See 3. Implementation issues: eq. 34 from Ghiglia et al.
#     # Improves number of needed iterations. Different from matlab implementation
#     WWx = np.minimum(WW[:, :-1], WW[:, 1:])
#     WWy = np.minimum(WW[:-1, :], WW[1:, :])
#     WWdx = WWx * dx
#     WWdy = WWy * dy

#     # applying A^T to WWdx and WWdy is like obtaining rho in the unweighted case
#     WWdx2 = np.diff(WWdx, axis=1, prepend=0, append=0)
#     WWdy2 = np.diff(WWdy, axis=0, prepend=0, append=0)

#     rk = WWdx2 + WWdy2
#     normR0 = np.linalg.norm(rk)

#     # start the iteration
#     eps = 1e-9
#     k = 0
#     phi = np.zeros_like(psi)
#     while (~np.all(rk == 0.0)):
#         zk = solvePoisson(rk)
#         k += 1

#         # equivalent to (rk*zk).sum()
#         rkzksum = np.tensordot(rk, zk)
#         if (k == 1):
#             pk = zk
#         else:
#             betak = rkzksum / rkzkprevsum
#             pk = zk + betak * pk

#         # save the current value as the previous values
#         rkzkprevsum = rkzksum

#         # perform one scalar and two vectors update
#         Qpk = applyQ(pk, WWx, WWy)
#         alphak = rkzksum / np.tensordot(pk, Qpk)
#         phi += alphak * pk
#         rk -= alphak * Qpk

#         # check the stopping conditions
#         if ((k >= kmax) or (np.linalg.norm(rk) < eps * normR0)):
#             break
#         # print(np.linalg.norm(rk), normR0)
#     print(k, rk.shape)
#     return phi


# def solvePoisson(rho):
#     """Solve the poisson equation "P phi = rho" using DCT
#     """
#     dctRho = dctn(rho);
#     N, M = rho.shape;
#     I, J = np.ogrid[0:N, 0:M]
#     with np.errstate(divide='ignore'):
#         dctPhi = dctRho / 2 / (np.cos(np.pi * I / M) + np.cos(np.pi * J / N) - 2)
#     dctPhi[0, 0] = 0  # handling the inf/nan value
#     # now invert to get the result
#     phi = idctn(dctPhi)
#     return phi


def solvePoisson_precomped(rho, scale):
    """
    Solve the poisson equation "P phi = rho" using DCT
    Uses precomputed scaling factors `scale`
    """
    dctPhi = dctn(rho) / scale

    # now invert to get the result
    phi = idctn(dctPhi, overwrite_x=True)

    return phi


def precomp_Poissonscaling(rho):
    N, M = rho.shape
    I, J = np.ogrid[0:N, 0:M]
    scale = 2 * (np.cos(np.pi * I / M) + np.cos(np.pi * J / N) - 2)
    # Handle the inf/nan value without a divide by zero warning:
    # By Ghiglia et al.:
    # "In practice we set dctPhi[0,0] = dctn(rho)[0, 0] to leave
    #  the bias unchanged"
    scale[0, 0] = 1.
    return scale


def applyQ(p, WWx, WWy):
    """Apply the weighted transformation (A^T)(W^T)(W)(A) to 2D matrix p"""
    # apply (A)
    dx = np.diff(p, axis=1)
    dy = np.diff(p, axis=0)

    # apply (W^T)(W)
    WWdx = WWx * dx
    WWdy = WWy * dy

    # apply (A^T)
    WWdx2 = np.diff(WWdx, axis=1, prepend=0, append=0)
    WWdy2 = np.diff(WWdy, axis=0, prepend=0, append=0)
    Qp = WWdx2 + WWdy2
    return Qp


def _wrapToPi(x):
    r = (x + np.pi) % (2 * np.pi) - np.pi
    return r


def phase_unwrap(psi, weight=None, kmax=100):
    """
    Unwrap the phase of an image psi given weights

    This function uses an algorithm described by Ghiglia and Romero and can either be used with or without weight array.
    It is especially suited to recover a unwrapped phase image from a (noisy) complex type image, where psi would be
    the angle of the complex values and weight the absolute values of the complex image.
    """

    # vector b in the paper (eq 15) is dx and dy
    dx = _wrapToPi(np.diff(psi, axis=1))
    dy = _wrapToPi(np.diff(psi, axis=0))

    # multiply the vector b by weight square (W^T * W)
    if weight is None:
        # Unweighed case. will terminate in 1 round
        WW = np.ones_like(psi)
    else:
        WW = weight ** 2

    # See 3. Implementation issues: eq. 34 from Ghiglia et al.
    # Improves number of needed iterations. Different from matlab implementation
    WWx = np.minimum(WW[:, :-1], WW[:, 1:])
    WWy = np.minimum(WW[:-1, :], WW[1:, :])
    WWdx = WWx * dx
    WWdy = WWy * dy

    # applying A^T to WWdx and WWdy is like obtaining rho in the unweighted case
    WWdx2 = np.diff(WWdx, axis=1, prepend=0, append=0)
    WWdy2 = np.diff(WWdy, axis=0, prepend=0, append=0)

    rk = WWdx2 + WWdy2
    normR0 = np.linalg.norm(rk)

    # start the iteration
    eps = 1e-9
    k = 0
    phi = np.zeros_like(psi)
    scaling = precomp_Poissonscaling(rk)
    while (~np.all(rk == 0.0)):
        zk = solvePoisson_precomped(rk, scaling)
        k += 1

        # equivalent to (rk*zk).sum()
        rkzksum = np.tensordot(rk, zk)
        if (k == 1):
            pk = zk
        else:
            betak = rkzksum / rkzkprevsum
            pk = zk + betak * pk

        # save the current value as the previous values
        rkzkprevsum = rkzksum

        # perform one scalar and two vectors update
        Qpk = applyQ(pk, WWx, WWy)
        alphak = rkzksum / np.tensordot(pk, Qpk)
        phi += alphak * pk
        rk -= alphak * Qpk

        # check the stopping conditions
        if ((k >= kmax) or (np.linalg.norm(rk) < eps * normR0)):
            break

    print(f"Phase unwrap terminated after {k} iterations")

    return phi