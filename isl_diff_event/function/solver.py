# -*- coding: utf-8 -*-
"""
Implementation of automatic differential lensless holographic imaging.

@author: Ni Chen (https://ni-chen.github.io/)
"""

import time

import kornia
import numpy as np
import copy
# import cupy as np
import torch
from piq import psnr, ssim
from torch import nn
from torch.optim import lr_scheduler
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:
        """No-op fallback used when tensorboard is not installed."""
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass
import torch.nn.functional as F

from function.optimizer import AdamP, tv_loss, tv_loss_ph, l1_loss

# from ptychography import psr_forward, holo_AD, PSR_AD, CP_PIE, refine_position, track_position, shift_frequency, \
#     shift_forward, load_config_from_file, CP_AD, transfer_function
from function.ptychography import *

from function.util import mkdir


########################################################################################################################
class Solver:
    def __init__(self, holo_raw=None, is_defocuse=False,
                 img_num=1, img_size=128, mag=4, is_non_neg=False,
                 obj=None,
                 z_os=500e-6, wavelen=0.532e-6, pp_sensor=1.1e-6,
                 method='AD', maxit=[200, 1000], lr=[0.01, 0.01, 1e-6],
                 kappa_l1=0,
                 kappa_tv=0, tv_order=1, tv_iso=True, tv_tau=1e-8,
                 kappa_laplacian=1e-8,
                 is_calibration=0, z_om=0, z_ms=0,  # for CP_PIE
                 writer=None, is_verbose=True, verbose_interval=100, device='cpu',
                 **kwargs):

        self.holo_raw = holo_raw

        self.img_num = img_num
        self.img_size = img_size
        self.is_non_neg = is_non_neg
        self.mag = mag
        self.img_size_up = img_size * mag
        self.pp_sensor = pp_sensor
        self.pp_up = pp_sensor / mag
        self.wavelen = wavelen

        self.is_defocuse = is_defocuse
        self.z_os = z_os

        self.obj = obj
        
        self.pos = None
        self.pos_est = None

        self.method = method
        self.device = device

        self.maxit_obj = maxit[0]
        self.maxit_holo = maxit[1]
        self.lr_obj = lr[0]
        self.lr_holo = lr[1]
        self.lr_z = lr[2]

        self.tv_tau = tv_tau
        self.tv_iso = tv_iso
        self.tv_order = tv_order

        self.kappa_tv = kappa_tv
        self.kappa_l1 = kappa_l1

        self.kappa_laplacian = kappa_laplacian

        self.is_calibration = is_calibration
        self.z_om = torch.as_tensor(z_om)
        self.z_ms = torch.as_tensor(z_ms)

        self.writer = writer
        self.is_verbose = is_verbose
        self.verbose_interval = verbose_interval

        # self.constraint = Constraints(obj=obj)

        # Results
        self.obj_pred = None
        self.holo_hr = None
        self.z_os_pred = None

        self.loss_hist = []
        self.err_hist = []
        self.z_os_hist = []

        self.z_ms_hist = []
        self.z_om_hist = []

        self.holo_loss_hist = []
        self.holo_err_hist = []
        

    def build(self):
        print(f'----------------------------- {self.method} starting -----------------------------')

        if self.method == 'PSR':
            '''PSR lensless hologrpahy'''
            self.psr()

        elif self.method == 'PSR_AD':
            '''Alterative PSR and AD lensless hologrpahy'''
            self.psr_ad()

        elif self.method == 'AD':
            '''End-to-End AD-PSR hologrpahy'''
            self.ad()

        elif self.method == 'CP_PIE':
            '''coded mask ptychography with PIE'''
            self.cp_pie()

        elif self.method == 'CP_AD':
            '''coded mask ptychography with AD'''
            self.cp_ad()

        else:
            raise Exception("Undefined solver type (accepted: TF, RI, depth)")

        print('-----------------------------------------------------------------------------------')

        
    ############################################### end-to-end AD ##############################################
    def ad(self, is_defocuse=True):
        """Estimate the object field, optional propagation distance, and positions by automatic differentiation."""
        self.pos_registration()

        vars = []
        u_obj = torch.ones(
            (self.img_size_up, self.img_size_up),
            dtype=torch.complex64,
            requires_grad=True,
            device=self.device,
        )
        u_obj = nn.Parameter(u_obj, requires_grad=True)
        vars += [{'params': u_obj, 'lr': self.lr_obj}]

        if is_defocuse:
            z_os_pred = nn.Parameter(torch.as_tensor(self.z_os), requires_grad=True)
            vars += [{'params': z_os_pred, 'lr': self.lr_z}]
        else:
            z_os_pred = torch.as_tensor(self.z_os, device=self.device)

        pos_est = torch.as_tensor(copy.deepcopy(self.pos), device=self.device)
        pos_est = nn.Parameter(pos_est, requires_grad=True)
        vars += [{'params': pos_est, 'lr': 1e-2}]

        optim = AdamP(vars)
        scheduler = lr_scheduler.StepLR(optim, step_size=self.maxit_obj // 2, gamma=0.1, verbose=False)
        t_start = time.time()

        for epoch in range(self.maxit_obj):
            optim.zero_grad()
            holo_pred, c = PSR_AD(
                u_o=u_obj,
                img_raw=self.holo_raw,
                z_os=z_os_pred,
                ps_sensor=self.pp_sensor,
                wave_len=self.wavelen,
                pos=pos_est,
                mag=self.mag,
                device=self.device,
            )

            err = torch.mean(torch.square(torch.sqrt(holo_pred) - torch.sqrt(self.holo_raw)), axis=(1, 2)).mean()
            loss = 0.5 * err + self.regularizer(u_obj)

            loss.backward(retain_graph=True)
            optim.step()
            scheduler.step()

            if self.is_non_neg:
                amp = u_obj.data.abs()
                ph = u_obj.data.angle()
                ph = torch.clamp(ph, min=0.0, max=None)
                u_obj.data = amp * torch.exp(1j * ph)

            with torch.no_grad():
                if (epoch + 1) % 50 == 0:
                    print(f'Epoch {epoch}/{self.maxit_obj}: '
                          f'loss is {loss.cpu().data.numpy():.8f}; '
                          f'z is {z_os_pred.cpu().data.numpy() * 1e6:.6f}; '
                          f'c is {c.cpu().data.numpy():.3f}; ')

                self.err_hist.append(err.cpu().data.numpy())
                self.loss_hist.append(loss.cpu().data.numpy())
                self.z_os_hist.append(z_os_pred.cpu().data.numpy() * 1e6)

        self.writer.close()
        H_os = transfer_function(
            wave_len=self.wavelen,
            ps=self.pp_up,
            img_size=self.img_size_up,
            z=z_os_pred.detach(),
            device=self.device,
        )
        self.holo_hr = RS(u_obj.detach(), H_os).abs() ** 2
        self.z_os_pred = z_os_pred.detach()
        self.obj_pred = u_obj.detach()
        self.pos_est = pos_est.detach()

    ######################################## Alterative PSR and AD #####################################
    def psr_ad(self, is_defocuse=True):

        self.pos_registration()
        holo_raw_hr = self._holo_reg_()

        holo_hr = nn.Parameter(holo_raw_hr, requires_grad=True)
        vars2 = [{'params': holo_hr, 'lr': self.lr_holo}]
        
        pos_est = torch.as_tensor(copy.deepcopy(self.pos))        
        pos_est = nn.Parameter(pos_est, requires_grad=True)
        vars2 += [{'params': pos_est, 'lr': 1e-2}]

        vars1 = []
        u_obj = torch.ones((self.img_size_up, self.img_size_up), dtype=torch.complex64, requires_grad=True, device=self.device)
        u_obj = nn.Parameter(u_obj, requires_grad=True)
        vars1 += [{'params': u_obj, 'lr': self.lr_obj}]              

        if is_defocuse:
            z_os_pred = nn.Parameter(torch.as_tensor(copy.deepcopy(self.z_os)), requires_grad=True)
            vars1 += [{'params': z_os_pred, 'lr': self.lr_z}]

        optim_holo = AdamP(vars1)        
        optim_psr = AdamP(vars2)
        scheduler1 = lr_scheduler.StepLR(optim_psr, step_size=self.maxit_holo // 2, gamma=0.1, verbose=False)
        scheduler2 = lr_scheduler.StepLR(optim_holo, step_size=self.maxit_obj // 2, gamma=0.1, verbose=False)
        t_start = time.time()
        for epoch in range(self.maxit_obj + self.maxit_holo):
            optim_psr.zero_grad()

            if epoch < self.maxit_holo:
                img_pred = psr_forward(holo_hr=holo_hr, pos=pos_est, mag=self.mag, device=self.device)

                err_psr = 0.5 * torch.mean(torch.square(img_pred - self.holo_raw))
                holo_hr_filter = kornia.filters.laplacian(holo_hr[(None,) * 2], kernel_size=3).squeeze().flatten().abs()
                loss_psr = err_psr + self.kappa_laplacian * (holo_hr_filter * holo_hr_filter).sum()

                loss_psr.backward(retain_graph=True)
                optim_psr.step()
                scheduler1.step()

                holo_hr.data = torch.clamp(holo_hr.data, min=0.0, max=None)

            optim_holo.zero_grad()
            holo_hr_pred = holo_AD(u_o=u_obj, z_os=z_os_pred, ps=self.pp_up, wave_len=self.wavelen, device=self.device)
            c = (holo_hr_pred * holo_hr).sum() / (holo_hr_pred * holo_hr_pred).sum()
            err_holo = torch.mean(torch.square(torch.sqrt(c * holo_hr_pred) - torch.sqrt(holo_hr)))
            loss_holo = 0.5 * err_holo + self.regularizer(u_obj)

            loss_holo.backward(retain_graph=True)
            optim_holo.step()
            scheduler2.step()
                        
            # non-negative phase
            if self.is_non_neg:
                amp = u_obj.data.abs()
                ph = u_obj.data.angle()
                ph = torch.clamp(ph, min=0.0, max=None)
                u_obj.data = amp*torch.exp(1j*ph)

            with torch.no_grad():
                if (epoch + 1) % 50 == 0:
                    print(f'Epoch {epoch}/{self.maxit_obj + self.maxit_holo}: '
                          f'loss_psr is {loss_psr.cpu().data.numpy():.8f}; '
                          f'loss_holo is {loss_holo.cpu().data.numpy():.6f}; '
                          f'z is {z_os_pred.cpu().data.numpy() * 1e6:.3f}; '
                          f'c is {c.cpu().data.numpy():.3f}; '
                          # f'Grad: '
                          # f'dh is {holo_hr.grad.sum():.8f}; '
                          # f'do is {u_obj.grad.sum():.6f}; '
                          # f'dz is {z_os_pred.grad:.3f}; '
                          )

                self.holo_err_hist.append(err_psr.cpu().data.numpy())
                self.holo_loss_hist.append(loss_psr.cpu().data.numpy())

                self.err_hist.append(err_holo.cpu().data.numpy())
                self.loss_hist.append(loss_holo.cpu().data.numpy())

                self.z_os_hist.append(z_os_pred.cpu().data.numpy() * 1e6)

        self.writer.close()

        self.holo_hr = holo_hr.detach()
        self.z_os_pred = z_os_pred.detach()
        self.obj_pred = u_obj.detach()
        self.pos_est = pos_est.detach()

        
    ################################################ PSR ##########################################
    def psr(self, is_defocuse=True):

        print('PSR hologram reconstruction')
        self.pos_registration()
        holo_raw_hr = self._holo_reg_()

        holo_hr = nn.Parameter(holo_raw_hr, requires_grad=True)
        vars = [{'params': holo_hr, 'lr': self.lr_holo}]
        
        pos_est = torch.as_tensor(copy.deepcopy(self.pos))        
        pos_est = nn.Parameter(pos_est, requires_grad=True)
        vars += [{'params': pos_est, 'lr': 1e-2}]        
        
        optimizer = AdamP(vars)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.maxit_holo // 2, gamma=0.1, verbose=False)
        for epoch in range(self.maxit_holo):
            optimizer.zero_grad()

            holo_pred = psr_forward(holo_hr=holo_hr, pos=pos_est, mag=self.mag, device=self.device)

            err = 0.5 * torch.mean(torch.square(holo_pred - self.holo_raw), axis=(0, 1)).mean()
            holo_hr_filter = kornia.filters.laplacian(holo_hr[(None,) * 2], kernel_size=3).squeeze().flatten()
            loss = err + self.kappa_laplacian * (holo_hr_filter * holo_hr_filter).sum()

            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()

            holo_hr.data = torch.clamp(holo_hr.data, min=0.0, max=None)

            with torch.no_grad():
                if (epoch + 1) % 50 == 0:
                    print(f'PSR {epoch}/{self.maxit_holo}: '
                          f'loss is {loss.cpu().data.numpy():.8f}; '
                          #   f'h_grad is {holo_hr.grad.sum():.8f}; '
                          )

                self.holo_loss_hist.append(loss.cpu().data.numpy())
                self.holo_err_hist.append(err.cpu().data.numpy())

        holo_hr = holo_hr.detach()
        self.holo_hr = holo_hr / holo_hr.median()

        print('Sample reconstruction')
        vars = []

        u_obj = torch.ones((self.img_size_up, self.img_size_up), dtype=torch.complex64, device=self.device)
#         u_obj = torch.randn((self.img_size_up, self.img_size_up), dtype=torch.complex64, device=self.device) + 1.0
        u_obj = nn.Parameter(u_obj, requires_grad=True)
        vars += [{'params': u_obj, 'lr': self.lr_obj}]

        if is_defocuse:
            z_os_pred = nn.Parameter(torch.as_tensor(self.z_os), requires_grad=True)
            vars += [{'params': z_os_pred, 'lr': self.lr_z}]

        optimizer = AdamP(vars)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.maxit_holo//2, gamma=0.1, verbose=False)
        t_start = time.time()
        for epoch in range(self.maxit_obj):
            optimizer.zero_grad()

            holo_hr_pred = holo_AD(u_o=u_obj, z_os=z_os_pred, ps=self.pp_up, wave_len=self.wavelen, device=self.device)
            c = (holo_hr_pred * self.holo_hr).sum() / (holo_hr_pred * holo_hr_pred).sum()
            err = torch.mean(torch.square(torch.sqrt(c * holo_hr_pred) - torch.sqrt(self.holo_hr)))
            loss = 0.5 * err + self.regularizer(u_obj)

            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()
            
            if self.is_non_neg:
                amp = u_obj.data.abs()
                ph = u_obj.data.angle()
                ph = torch.clamp(ph, min=0.0, max=None)
                u_obj.data = amp*torch.exp(1j*ph)         

            with torch.no_grad():
                if (epoch + 1) % 100 == 0:
                    print(f'Sample {epoch}/{self.maxit_obj}: loss is {loss.cpu().data.numpy():.6f}; '
                          f'z is {z_os_pred.cpu().data.numpy() * 1e6:.3f}; '
                          f'c is {c.cpu().data.numpy():.3f}; '
                          # f'do is {u_obj.grad.sum():.6f}; '
                          # f'dz is {z_os_pred.grad:.3f}; '
                          )

                self.loss_hist.append(loss.cpu().data.numpy())
                self.err_hist.append(err.cpu().data.numpy())
                self.z_os_hist.append(z_os_pred.cpu().data.numpy() * 1e6)

                # self.print_log(epoch=epoch, loss=loss, x_pred=holo_hr_pred, x_gt=self.holo_raw, t_start=t_start)

        self.writer.close()

        self.z_os_pred = z_os_pred.detach()
        self.obj_pred = u_obj.detach()
        self.pos_est = pos_est.detach()
        

#     def pos_registration(self, is_mask=0):        
#         pos_track = track_position(img_seq=self.holo_raw, is_mask=is_mask)
#         pos_refine = pos_track

#         for iLoc in range(3):
#             pos_refine = refine_position(img_seq=self.holo_raw, pos=pos_refine, is_mask=is_mask)

#         self.pos = pos_refine
    
    
    def pos_registration(self, is_mask=0):
        holo_raw_hr = torch.zeros(self.img_num, self.img_size_up, self.img_size_up, dtype=torch.float64, device=self.device)
        for idx_img in range(self.img_num):
            tmp = self.holo_raw[idx_img, :,:]
#             holo_raw_hr[:,:,idx_img] = F.interpolate(tmp[(None,) * 2], size=(self.img_size_up, self.img_size_up),
#                                     mode='bicubic').squeeze()
            holo_raw_hr[idx_img, :,:] = F.interpolate(tmp[(None,) * 2], size=(self.img_size_up, self.img_size_up),
                                       mode='bilinear', align_corners=True).squeeze()
        
        pos_track = track_position(img_seq=holo_raw_hr, is_mask=is_mask)
        pos_refine = pos_track

        for iLoc in range(3):
            pos_refine = refine_position(img_seq=holo_raw_hr, pos=pos_refine, is_mask=is_mask)

        self.pos = pos_refine


    def _holo_reg_(self, ):
        fx, fy = shift_frequency(img_size=self.img_size, device=self.device)
#         holo_raw_sum = torch.zeros(self.img_size, self.img_size, device=self.device)
#         for idx_img in range(self.img_num):
#             f_shift = torch.exp(-1j * 2 * np.pi * (fx * self.pos[0, idx_img] / self.img_size + fy * self.pos[1, idx_img] /
#                                    self.img_size)).to(device=self.device)
#             holo_raw_sum += shift_forward(self.holo_raw[:, :, idx_img], f_shift).real.abs()
#         holo_raw_avg = holo_raw_sum / self.img_num
        
        holo_raw_avg = self.holo_raw[0, :, :]
        
#         holo_raw_hr = F.interpolate(holo_raw_avg[(None,) * 2], size=(self.img_size_up, self.img_size_up),
#                                     mode='nearest-exact').squeeze()
        holo_raw_hr = F.interpolate(holo_raw_avg[(None,) * 2], size=(self.img_size_up, self.img_size_up), mode='bilinear').squeeze()

        return holo_raw_hr
    
    

    def regularizer(self, x):
        reg = self.kappa_tv[0] * tv_loss(x, tv_order=self.tv_order, iso=self.tv_iso, tv_tau=self.tv_tau) \
              + self.kappa_l1[0] * l1_loss(1 - x.abs()) \
              + self.kappa_tv[1] * tv_loss_ph(x, tv_order=self.tv_order, iso=self.tv_iso, tv_tau=self.tv_tau) \
              + self.kappa_l1[1] * l1_loss(torch.exp(1j * x.angle()).imag)

        return reg
    
    

    ############################################# Log ###################################################
    def print_log(self, epoch=0, loss=0,
                  x_gt=None, x_pred=None, x_g=None,
                  t_start=None,
                  is_psnr=False, is_ssim=False, is_nmse=True,
                  **kwargs):

        self.loss_hist.append(loss.cpu().data.numpy())
        # self.factor_hist.append(W.mean().cpu().data.numpy())
        # factor_hist.append(W.mean().cpu().data.numpy())

        # if (kwargs.get("z", None) is not None):
        #     z = kwargs.get("z", None).cpu().data.numpy().item() * 1000
        #     self.z_hist.append(z)

        # if (kwargs.get("sigma_ratio", None) is not None):
        #     sigma_ratio = kwargs.get("sigma_ratio", None).cpu().data.numpy()
        #     self.sigma_ratio_hist.append(sigma_ratio)
        #     gaussian_center = kwargs.get("gaussian_center", None).cpu().data.numpy()
        #     self.gaussian_center_hist.append(gaussian_center)

        if (self.is_verbose and (epoch % self.verbose_interval == 0 or epoch == self.maxit[1] - 1)) \
                or (self.is_verbose == False and epoch == self.maxit[1] - 1):

            epoch_time = time.time() - t_start

            # x_pred = torch.view_as_real(x_pred.squeeze())
            # x_pred = (x_pred - x_pred.min()) / (x_pred.max() - x_pred.min())

            print_str = f'epoch {epoch}, loss={self.loss_hist[epoch]}'
            self.writer.add_scalar(f'loss', self.loss_hist[epoch], epoch, walltime=epoch_time)

            if 'z' in locals():
                print_str = f'{print_str}, z={z}'

            # if 'sigma_ratio' in locals():
            #     print_str = f'{print_str}, sigma={sigma_ratio}, gaussian_center={gaussian_center}'

            if (x_g is not None):
                print_str = f'{print_str}, grad={x_g.grad.mean()}'

            if is_psnr:
                psnr_val = psnr(x_gt, x_pred, data_range=1.0)
                print_str = f'{print_str}, PSNR={psnr_val}'
                self.writer.add_scalar(f'psnr', psnr_val, epoch, walltime=epoch_time)

            if is_ssim:
                ssim_val = ssim(x_gt, x_pred, data_range=1.0)
                print_str = f'{print_str}, SSIM={ssim_val}'
                self.writer.add_scalar(f'ssim', ssim_val, epoch, walltime=epoch_time)

            # if is_nmse:
            #     nmse_val = torch.mean(torch.square(x_pred - self.holo)) / torch.mean(torch.square(self.holo))
            #     print_str = f'{print_str}, NMSE={nmse_val}'
            #     self.writer.add_scalar(f'nmse', nmse_val, epoch, walltime=epoch_time)

            print_str = f'{print_str}, time={epoch_time}'
            print(print_str)


############################################# Log ###################################################
def call_solver(methods=['PSR'], **sys_params):
    for idx_m in range(len(methods)):
        # print(f'{opt_params.method} starting...')
        sys_params["method"] = methods[idx_m]

        # Tensorboard writer
        logs_dir = f'{sys_params["logs_dir"]}{methods[idx_m]}/'
        mkdir(logs_dir)
        sys_params["writer"] = SummaryWriter(logs_dir)

        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()

        lensless_solver = Solver(**sys_params)
        lensless_solver.build()

        # end.record()
        # torch.cuda.synchronize()
        # print(f'Total running time is {start.elapsed_time(end)/1000} s')

        ################################ Output results ###########################################
        # plot_loss(lensless_solver.loss_hist, 
        #           save_path=f'{sys_params["prefix"]}_loss_{methods[idx_m].lower()}.png',
        #           title=f'loss {methods[idx_m]}')

        # if len(lensless_solver.z_hist) > 0:
        #     plot_loss((lensless_solver.z_hist),
        #               save_path=f'{sys_params["prefix"]}_z_{methods[idx_m].lower()}.png',
        #               title=f'z/mm')

        # if sys_params["lr_illum"] > 0:
        #     plot_factor(format(lensless_solver.factor_hist, '.2f'),
        #                 save_path=f'{sys_params["prefix"]}_factor_{methods[idx_m].lower()}.png')

    return lensless_solver


def check_parameters(test_name=None,
                     file_type='tiff',
                     is_non_neg = False,
                     mag=4,
                     img_size=300,
                     img_num=11,
                     shifts=[0, 0],
                     z_os=0,
                     z_ms=0,
                     z_om=0,
                     noise_var=0.02,
                     is_calibration=1,
                     is_defocuse=True,
                     wavelen=0,
                     pp_sensor=0,
                     method='AD',
                     maxit=[1000, 200],
                     lr=[0.01, 0.001, 1e-6],
                     kappa_huber=0.0, kappa_l1=[0, 0],
                     kappa_tv=[0, 0], tv_order=2, tv_iso=True, tv_tau=1e-5,
                     kappa_laplacian=1e-8,
                     writer=None, is_verbose=True,
                     **kwargs):
    sys_params = {
        'method': method,         # PSR_AD, PSR, AD, CP_PIE, CP_AD
        # object parameters
        'test_name': test_name,   # folder name of the dataset
        # setup parameters   
        'wavelen': wavelen,       # wavelength of the light source
        'pp_sensor': pp_sensor,   # sensor pixel pitch
        'file_type': file_type,   # file type of the raw data
        'is_non_neg': is_non_neg,
        'shifts': shifts,         # cropped image center
        'noise_var': noise_var,   # simulation
        'img_size': img_size,     # Cropped image size
        'img_num': img_num,       # number of measurements
        'mag': mag,               # upsampling factor
        # lensless parameters
        'z_os': z_os,
        # coded mask parameters
        'z_ms': z_ms,            # mask-sensor distance
        'z_om': z_om,            # object-mask distance
        'is_calibration': is_calibration,
        # optimization parameters
        'is_defocuse': is_defocuse,  # if use autofocus
        'kappa_laplacian': kappa_laplacian,  # [amplitude, phase]
        'kappa_huber': kappa_huber,  # Huber regularization, Not yet implemented
        'kappa_l1': kappa_l1,    # [amplitude, phase]
        'kappa_tv': kappa_tv,    # [amplitude, phase]
        'tv_order': tv_order,    # order of TV regularization, could be 1 or 2
        'tv_tau': tv_tau,        # 
        'tv_iso': tv_iso,        # if use isotropic TV
        'lr': lr,                # [lr_obj, lr_holo, lr_z]
        'maxit': maxit,          # [lr_obj, lr_holo, lr_z]
        # Others
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # 'cpu'
        'data_dir': '/xdisk/nichen/data/lensless/',   # ../../data
        'out_dir': './output/',
        'colormap': 'jet',
        'writer': writer,
        'is_verbose': is_verbose,
    }

    sys_params["data_dir"] = f'{sys_params["data_dir"]}{sys_params["test_name"]}/'

    setup_params = load_config_from_file(f'{sys_params["data_dir"]}config.yaml')
    sys_params = {**sys_params, **setup_params}

    sys_params[
        "prefix"] = f'{sys_params["out_dir"]}{sys_params["test_name"]}_n{sys_params["img_num"]}_m{sys_params["mag"]}_{sys_params["method"].lower()}'
    sys_params["img_size_up"] = sys_params["mag"] * sys_params["img_size"]
    sys_params["pp_up"] = sys_params["pp_sensor"] / sys_params["mag"]

    sys_params["logs_dir"] = f'{sys_params["out_dir"]}log/'
    mkdir(sys_params["out_dir"])
    mkdir(sys_params["logs_dir"])

    ####################################### Lensless ###############################################
    # kappa_laplacian is for optimization of the pixel super resolution of the holograms
    if sys_params["method"] == 'PSR':
        sys_params["kappa_laplacian"] = 5e-7

    if sys_params["method"] == 'PSR_AD':
        sys_params["kappa_laplacian"] = 10e-8
#         sys_params["kappa_laplacian"] = 1e-7

        sys_params["tv_tau"] = 1e-4

    if sys_params["method"] == 'AD':
        sys_params["kappa_laplacian"] = 1e-8

        sys_params["maxit"][1] = 0
        sys_params["lr"][1] = 0

    ################################## Coded Ptychography ##########################################
    if sys_params["method"] == 'CP_PIE':
        # assert sys_params["maxit"][0] > 15, print('PIE iteration should not be larger than 15 iterations')
        if sys_params["z_om"] == 0 or sys_params["z_ms"] == 0:
            print('Please set the distances')

        # sys_params["z_os"] = sys_params["z_om"] + sys_params["z_ms"]

    return sys_params



