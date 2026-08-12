# -*- coding: utf-8 -*-
"""
Created on Tue June 1, 2020
@author: Ni Chen
"""

import torch
import numpy as np
import os
import math
import torch.nn.functional as F

np_dtype = np.float32
pt_dtype = torch.float32


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# masked vectors to ndarray
def masked_to_full(x, idx, shape):
    img_full = torch.zeros(shape, dtype=torch.double).flatten()
    # img_full[np.r_[idx]] = x
    return img_full.reshape(shape)




########################################################################################################################


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


class PSR_Solver:
    def __init__(self, holo_raw=None,
                 img_num=1, img_size=128, mag=3, pp_sensor = 18.5*1e-6):

        self.holo_raw = holo_raw
        self.img_num = img_num
        self.img_size = img_size
        self.mag = mag
        self.img_size_up = img_size * mag
        self.pp_sensor = pp_sensor
        self.pp_up = pp_sensor / mag

    def pos_registration(self, is_mask=0):
        holo_raw_hr = torch.zeros(self.img_num, self.img_size_up, self.img_size_up, dtype=torch.float64,
                                  device=self.device)
        for idx_img in range(self.img_num):
            tmp = self.holo_raw[idx_img, :, :]
            #             holo_raw_hr[:,:,idx_img] = F.interpolate(tmp[(None,) * 2], size=(self.img_size_up, self.img_size_up),
            #                                     mode='bicubic').squeeze()
            holo_raw_hr[idx_img, :, :] = F.interpolate(tmp[(None,) * 2], size=(self.img_size_up, self.img_size_up),
                                                       mode='bilinear', align_corners=True).squeeze()

        pos_track = track_position(img_seq=holo_raw_hr, is_mask=is_mask)
        pos_refine = pos_track

        for iLoc in range(3):
            pos_refine = refine_position(img_seq=holo_raw_hr, pos=pos_refine, is_mask=is_mask)

        self.pos = pos_refine