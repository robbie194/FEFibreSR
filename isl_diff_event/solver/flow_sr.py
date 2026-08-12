

import importlib
import matplotlib
try:
    matplotlib.use("Qt5Agg")  # Keep interactive backend when available.
except Exception:
    matplotlib.use("Agg")
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torchvision.transforms.functional
import sys
from pathlib import Path
from torch import nn
from torch.optim import Adam, lr_scheduler
from trajectory_model.Diff_tratrectory import ContinuousPiecewiseLinear_Dxy_pw
from matplotlib import pyplot as plt
torch.manual_seed(0)
from utils.utility import *
from utils.utils_event_flow import *
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
_FUNCTION_DIR = _REPO_ROOT / "function"

try:
    from function.optimizer import *
except ModuleNotFoundError:
    _optimizer_path = _FUNCTION_DIR / "optimizer.py"
    _spec = importlib.util.spec_from_file_location("_isl_function_optimizer", _optimizer_path)
    _optimizer = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_optimizer)
    for _name in dir(_optimizer):
        if not _name.startswith("_"):
            globals()[_name] = getattr(_optimizer, _name)
# def forward_model(field):
#     diffraction_pattern = forward_wave_prop(field, params).abs()**2
#     return diffraction_pattern
import torchvision.transforms as transforms

try:
    from function.ptychography import track_position, refine_position
except ModuleNotFoundError:
    _ptycho_path = _FUNCTION_DIR / "ptychography.py"
    _spec = importlib.util.spec_from_file_location("_isl_function_ptychography", _ptycho_path)
    _ptycho = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_ptycho)
    track_position = _ptycho.track_position
    refine_position = _ptycho.refine_position



class Simple_sr:
    def __init__(self, holo_raw=None,
                 img_num=9, img_size= np.asarray([260, 346]) , mag=3, pp_sensor=1.1e-6,device = "cpu"
                 ):
        self.holo_raw = holo_raw
        self.device = device
        self.img_num = img_num
        self.img_size = img_size
        self.mag = mag
        self.img_size_up = img_size * mag
        self.pp_sensor = pp_sensor
        self.pp_up = pp_sensor / mag


    def pos_registration(self, is_mask=0):
        holo_raw_hr = torch.zeros(self.img_num, self.img_size_up[0], self.img_size_up[1], dtype=torch.float64,
                                  device=self.device)
        for idx_img in range(self.img_num):
            tmp = self.holo_raw[idx_img, :, :]
            #             holo_raw_hr[:,:,idx_img] = F.interpolate(tmp[(None,) * 2], size=(self.img_size_up, self.img_size_up),
            #                                     mode='bicubic').squeeze()
            holo_raw_hr[idx_img, :, :] = F.interpolate(tmp[(None,) * 2], size=(self.img_size_up[0], self.img_size_up[1]),
                                                       mode='bilinear', align_corners=True).squeeze()

        pos_track = track_position(img_seq=holo_raw_hr, is_mask=is_mask)
        pos_refine = pos_track

        for iLoc in range(3):
            pos_refine = refine_position(img_seq=holo_raw_hr, pos=pos_refine, is_mask=is_mask)

        self.pos = pos_refine

def warp_back_frame(images, flow):
    """
    Warp all frames to align with reference frame (t = 0), keeping shape (9, H, W).

    Args:
        images: Tensor of shape (9, H, W)
        flow: Tensor of shape (2, 9) --(u, v) flow per frame. flow[:, 0] = (0, 0)

    Returns:
        aligned_images: Tensor of shape (9, H, W), where aligned_images[0] = images[0]
    """
    T, H, W = images.shape
    device = images.device

    # Create normalized sampling grid
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    base_grid = torch.stack((x, y), dim=-1).unsqueeze(0)  # (1, H, W, 2)

    aligned_images = [images[0]]  # Start with the reference frame

    for t in range(1, T):
        u = flow[0, t] * 2 / W
        v = flow[1, t] * 2 / H
        flow_grid = base_grid + torch.tensor([u, v], device=device).view(1, 1, 1, 2)

        img = images[t].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        warped = F.grid_sample(img, flow_grid, mode='bilinear', align_corners=True)

        aligned_images.append(warped.squeeze(0).squeeze(0))  # back to (H, W)

    return torch.stack(aligned_images, dim=0)  # shape: (9, H, W)

def sr_model(images, sigma):
    T, H, W = images.shape
    y_list = []
    for t in range(T):
        image_hr = F_upsampling(images[t, ...].detach(), sigma)  # fourier_upsampling(images[t,...], sigma)
        y_list.append(image_hr)
    return torch.stack(y_list, dim=0)  # (T, H//s, W//s)

# watch_tensor(ress)

def generate_normalized_grid(H, W, device):
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    return torch.stack((x, y), dim=-1).unsqueeze(0)  # (1, H, W, 2)

def warp_hr_to_t(x, flow_t, H, W, pad_mode='border'):
    """
    Warp HR image x according to flow_t (relative to t=0).
    """
    # Normalize flow to [-1, 1]
    u = (-flow_t[0] * 2 / W).to(x.device)
    v = (-flow_t[1] * 2 / H).to(x.device)
    base_grid = generate_normalized_grid(H, W, x.device)    # (1, H, W, 2)
    flow_grid = base_grid + torch.stack((u, v)).view(1, 1, 1, 2)
    x_in = x.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
    warped = F.grid_sample(x_in.float(), flow_grid.float(), mode='bilinear', align_corners=True, padding_mode=pad_mode)
    return warped.squeeze(0).squeeze(0)  # (H, W)



def warp_hr_to_t_mask(x, flow_t, H, W, mask = None,pad_mode='border'):
    """
    Warp HR image x according to flow_t (relative to t=0).
    """
    # Normalize flow to [-1, 1]
    u = (-flow_t[0] * 2 / W).to(x.device)
    v = (-flow_t[1] * 2 / H).to(x.device)
    base_grid = generate_normalized_grid(H, W, x.device)
    flow_offset =  torch.stack((u, v)).view(1, 1, 1, 2)

    if mask is not None:
        mask_in = mask.view(1, H, W, 1).float().to(x.device)
        flow_grid = base_grid + mask_in * flow_offset
    else:
        flow_grid = base_grid + flow_offset
    x_in = x.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
    warped = F.grid_sample(x_in.float(), flow_grid.float(), mode='bilinear', align_corners=True, padding_mode=pad_mode)
    return warped.squeeze(0).squeeze(0)  # (H, W)


# def downsample(x, sigma):
#     """
#     Downsample HR image x using average pooling.
#     """
#     x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
#     y = F.avg_pool2d(x, kernel_size=sigma, stride=sigma)
#     return y.squeeze(0).squeeze(0)  # (H//s, W//s)

def forward_mfsr_model(x, flow, M, pad_mode="border"):
    """
    Simulate LR observations y_t from HR image x and provide valid region masks.

    Args:
        x: HR image, shape (H, W)
        flow: optical flow to t=0, shape (2, T)
        M: downsampling factor (e.g., 4)
        edge_ignore: optional int, extra margin to ignore beyond warped region

    Returns:
        y: Tensor of shape (T, H//s, W//s) --simulated LR observations
        mask: Tensor of shape (T, H//s, W//s) --validity mask (1=valid, 0=invalid)
    """
    T = flow.shape[1]
    H, W = x.shape
    y_list = []
    m_list = []

    for t in range(T):
        # Warp HR image
        flow_t = flow[:, t] * M  # scale flow to HR domain
        x_warped = warp_hr_to_t(x, flow_t, H, W, pad_mode)
        y_t = down_sampling(x_warped, M)
        y_list.append(y_t)

        # Compute valid mask: warp all-ones tensor the same way
        ones = torch.zeros_like(x)
        ones[1:-1, 1:-1] = torch.tensor(1.0).detach().to(x.device)  # avoid border artifacts
        mask_warped = warp_hr_to_t(ones, flow_t, H, W, pad_mode='zeros')
        mask_down = down_sampling(mask_warped, M)
        mask_down = (mask_down >= 0.99).float()  # threshold to get binary mask
        m_list.append(mask_down)

    return torch.stack(y_list, dim=0), torch.stack(m_list, dim=0)  # both (T, H//s, W//s)


def warp_frame(x, flow, M, pad_mode="border"):
    """
    Simulate LR observations y_t from HR image x and provide valid region masks.

    Args:
        x: HR image, shape (H, W)
        flow: optical flow to t=0, shape (2, T)
        M: downsampling factor (e.g., 4)
        edge_ignore: optional int, extra margin to ignore beyond warped region

    Returns:
        y: Tensor of shape (T, H//s, W//s) --simulated LR observations
        mask: Tensor of shape (T, H//s, W//s) --validity mask (1=valid, 0=invalid)
    """
    #T = flow.shape[1]
    H, W = x.shape
    #y_list = []
    #m_list = []
    # yd_list = []
    # md_list = []
    flow_t = flow[:, 0] * M  # scale flow to HR domain
    x_warped = warp_hr_to_t(x, flow_t, H, W, pad_mode)
    # Compute valid mask: warp all-ones tensor the same way
    ones = torch.zeros_like(x)
    ones[1:-1, 1:-1] = torch.tensor(1.0).detach().to(x.device)  # avoid border artifacts
    mask_warped = warp_hr_to_t(ones, flow_t, H, W, pad_mode='zeros')
    mask_down = down_sampling(mask_warped, M)
    mask_warped = (mask_warped >= 0.99).float()
    #
    # for t in range(T):
    #         # Warp HR image
    #         # flow_t = flow[:, t] * M  # scale flow to HR domain
    #         # x_warped = warp_hr_to_t(x, flow_t, H, W, pad_mode)
    #         # y_t = down_sampling(x_warped, M)
    #         #y_list.append(x_warped)
    #         #yd_list.append(y_t)
    #
    #         # Compute valid mask: warp all-ones tensor the same way
    #         ones = torch.zeros_like(x)
    #         ones[1:-1, 1:-1] = torch.tensor(1.0).detach().to(x.device)  # avoid border artifacts
    #         mask_warped = warp_hr_to_t(ones, flow_t, H, W, pad_mode='zeros')
    #         # mask_down = down_sampling(mask_warped, M)
    #         mask_warped= (mask_warped >= 0.99).float()  # threshold to get binary mask
    #         m_list.append(mask_warped)
    return x_warped, mask_warped  # both (T, H//s, W//s)


def forward_mfsr_model_v2(x, flow, M, masks_e, pad_mode="border"):
    """
    Simulate LR observations y_t from HR image x and provide valid region masks.

    Args:
        x: HR image, shape (H, W)
        flow: optical flow to t=0, shape (2, T)
        M: downsampling factor (e.g., 4)
        edge_ignore: optional int, extra margin to ignore beyond warped region

    Returns:
        y: Tensor of shape (T, H//s, W//s) --simulated LR observations
        mask: Tensor of shape (T, H//s, W//s) --validity mask (1=valid, 0=invalid)
    """
    T = flow.shape[1]
    H, W = x.shape
    y_list = []
    m_list = []
    # yd_list = []
    # md_list = []

    for t in range(T):
        # Warp HR image
        flow_t = flow[:, t] * M  # scale flow to HR domain
        #x_warped = warp_hr_to_t_mask(x, flow_t, H, W, masks_e[t], pad_mode)
        x_warped = warp_hr_to_t(x, flow_t, H, W, pad_mode)
        # y_t = down_sampling(x_warped, M)
        y_list.append(x_warped)
        #yd_list.append(y_t)

        # Compute valid mask: warp all-ones tensor the same way
        ones = torch.zeros_like(x)
        ones[1:-1, 1:-1] = torch.tensor(1.0).detach().to(x.device)  # avoid border artifacts
        mask_warped = warp_hr_to_t(ones, flow_t, H, W, pad_mode='zeros')
        #mask_warped = mask_warped * masks_e[t]
        mask_down = down_sampling(mask_warped, M)
        mask_down = (mask_down >= 0.99).float()  # threshold to get binary mask
        m_list.append(mask_down)

    return torch.stack(y_list, dim=0), torch.stack(m_list, dim=0)  # both (T, H//s, W//s)




if __name__ == '__main__':
    sensor_size=(260, 346);
    ax_asp =  sensor_size[0]/ sensor_size[1];cy,cx = np.asarray(sensor_size)//2;
    Ny, Nx = sensor_size
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
    eve_dtype = "aedat4"
    M = 2 # SR magnification
    save_path = 'E:/[X] NeuroSR/fig/'
    ############################### Generate Zernike phase ################################
    # a_gt   = norm_torch(torch.from_numpy(cv2.resize(cv2.imread(read_path_f, cv2.IMREAD_GRAYSCALE), (Nx,Ny))).to(device),255**0.5)

    ### Step 1: load data
    #read_path_e = "E:/davis/pei/01.aedat4"
    #read_path_e ="E:/davis/pei/guoyin2.aedat4"
    #read_path_e ="E:/davis/pei/guoyin2.aedat4"
    #read_path_e ="E:/davis/pei/xiao_chang_01.aedat4"
    #read_path_e ="E:/[X] NeuroSR/event/lunchong.aedat4"
    #read_path_e ="E:/davis/pei/res_t.aedat4"
    #read_path_e = "E:/davis/miao/segment_000007.npy";eve_dtype = "npy"
    #read_path_e ="E:/davis/shuo/vib/02_1.aedat4"
    #read_path_e ="E:/davis/pei/4X.aedat4"
    #read_path_e  = "E:/davis/pei/10X.aedat4"
    #read_path_e ="E:/[X] NeuroSR/event/16s.aedat4"   # 50 s is better   # 83
    #read_path_e ="E:/[X] NeuroSR/event/15s.aedat4"
    #read_path_e ="E:/[X] NeuroSR/event/20s.aedat4"
    #read_path_e ="E:/[X] NeuroSR/event/yaping.aedat4"
    #read_path_e ="E:/[X] NeuroSR/event/dvSave-2024_11_20_11_23_53.aedat4"
    #read_path_e = r"E:\davis\yaping\object100\day\A15.aedat4"
    #read_path_e  = r"E:\davis\yuxing\40x_static\dvSave-2024_12_10_16_14_05.aedat4"
    #read_path_e  = r"E:\davis\yuxing\40x_static\dvSave-2024_12_10_16_02_42.aedat4"
    #read_path_e = "E:/davis/pei/02.aedat4"###
    read_path_e ="E:/davis/pei/onion.aedat4"
    #read_path_e ="E:/davis/shuo/01.aedat4"
    #read_path_e ="E:/davis/pei/xiao_chang_01.aedat4"
    #read_path_e  = r"E:\davis\yuxing\40x_static\dvSave-2024_12_10_16_14_05.aedat4";
    #read_path_e ="E:/davis/pei/huafeng.aedat4"
    #read_path_e ="E:/davis/pei/guoyin1.aedat4"
    #save_fig_path = r"E:\[X] NeuroSR\fig\res_t\sam_1"
    #ead_path_e = "E:\\davis\\yuxing\\1-paper_4x-2w\\dvSave-2024_12_10_15_46_37.aedat4"
    #read_path_e  = r"E:\davis\yuxing\40x_1s\dvSave-2024_12_10_15_56_34.aedat4"
    #read_path_e  = r"E:\[X] NeuroSR\event\lunchong.aedat4"
    #read_path_e =r"E:\\davis\miao\9-4-24\rotifer-500um-10x-old-chips\davis-386\3.aedat4"
    #read_path_e ="E:/davis/pei/qiuyin1.aedat4"
    #read_path_e ="E:/davis/pei/xiao_chang_01.aedat4"
    #read_path_e  = r"E:\[X] NeuroSR\event\lunchong.aedat4"
    # img = cv2.imread(r"E:\[X] NeuroSR\fig\sample.png",cv2.IMREAD_GRAYSCALE)/255
    # grad = np.gradient(img)[1]
    # plt.figure("grad")
    # grad = grad- np.median(grad)
    # norm_1 = TwoSlopeNorm(vmin=grad.min(), vcenter=0, vmax=grad.max())
    # plt.imshow(grad,cmap = "seismic", norm= norm_1)

    ### Step 2: select time window
    """   ##need to first watch the 3D scattering map to find a general window
    ts, xs, ys, ps, frames_img3, time_img = load_events(eve_dtype = "aedat4", path= read_path_e); ts = ts-ts.min()
    s=9.2e5; win = int(1e5) 
    xs, ys, ts, ps=time_window(xs, ys,ts, ps ,s =s, win = win);ts = ts - ts.min()
    plt.rcParams['font.size'] = 15
    plot_scatter_bos_rect(xs, ys,ts,ps, polar=True,view = "normal",sensor_size =sensor_size, axis_ind="off",alpha =  0.05,alpha_k=0.01,plot = True, save =None)
    #plot_scatter_bos_rect(xs, ys,ts,ps, polar=True,view = "normal",sensor_size =sensor_size, axis_ind="off",alpha = 0.05,alpha_k=0.01,plot = False,save = save_path +"scatter_guoying.png" )
    """

    save_dir = "E:/[X] NeuroSR/fig/"
    if eve_dtype == "aedat4":
        ts, xs, ys, ps, frames_img3, time_img = load_events(eve_dtype="aedat4", path=read_path_e);
        ts -= ts.min()  # start from 0
        if len(frames_img3.shape) == 4:
            frames_img = np.zeros(frames_img3.shape[:3])
            for i in range(frames_img3.shape[0]//2):
                frames_img[i, :, :] = cv2.cvtColor(frames_img3[i, :, :, :], cv2.COLOR_BGR2GRAY)
        elif len(frames_img3.shape) == 3:
            frames_img = frames_img3
    elif eve_dtype == "npy":
        ts, xs, ys, ps = load_events(eve_dtype="npy", path=read_path_e)
        ts -= ts.min()  # start from 0
    #s = 2e6; win = int(2.5e5);
    #s= 15.3e6; win = int(1e5)
    s=2.45e6;win = int(1e5)# yangcong
    #s = 0.7e6;win = int(5e4)   #guoyin2
    # s = 0.3e6; win = int(1e5) #huafeng
    #s = 2.45e6;win = int(1e5)
    #s=8.2e5; win = int(1e5) # xiao chang
    #s =0.4e6;win = int(1e5) #qiuyin
    #s = 9.9e6; win = int(3e4)  lunchong
    #s=0.82e6; win = int(2e5)
    #s=0.6e6;win = int(1e5);
    #s =0.5e6;win =int(1e5)
    #s =29.75e6;win =int(1e5)
    #s=1.45e6;win = int(1e5);
    #s= 83.8e6; win = int(2e4)
    #s= 51.3e6; win = int(2e4)
    #s = 2e6; win = int(2.5e5);
    #s= 16.7e6; win = int(2e4)
    #s= 20.10e6; win = int(6e4)
    #s = 18/4*1e6;win = int(2e5);
    #s =15e6; win = int(1e5)
    #s = 2.45e6;win = int(2e5);
    # s = 0.82e6; win = int(1e5);
    #s = 5.45e6;win = int(1e5);
    # s =2.3e6; win = int(2e5)
    #s=2.45e6;win = int(2e5);
    #s = 2e4; win = int(2e4)

    if eve_dtype == "aedat4":
        ts, xs, ys, ps, frames_img3, time_img = load_events(eve_dtype="aedat4", path=read_path_e);
        ts -= ts.min()  # start from 0
        if len(frames_img3.shape) == 4:
            frames_img = np.zeros(frames_img3.shape[:3])
            for i in range(frames_img3.shape[0]):
                frames_img[i, :, :] = cv2.cvtColor(frames_img3[i, :, :, :], cv2.COLOR_BGR2GRAY)
        elif len(frames_img3.shape) == 3:
            frames_img = frames_img3
    ts -= ts.min()  # start from 0
    xs, ys, ts, ps = time_window(xs, ys,ts, ps ,s =s, win = win)
    ts = ts -  ts.min()
    delta_t = win

    if eve_dtype == "aedat4":
        frame_ind = np.where(time_img[:, 0] > s)[0][0]- 1 #np.where(time_img[:, 0] > s)[0][0] - 1 #frame_ind = np.where(time_img[:, 0] < s + win)[0][0];
        ##select frame
        frames_sta = frames_img[0,:,:]
        frames_mov = frames_img[frame_ind,:,:]
        #del frames_img,frames_img3
        time_win = time_img[frame_ind,:]
        t_c = time_win.mean()
        win_frame = int(time_img[frame_ind].max()-time_img[frame_ind].min())
        if win < win_frame:
            win = win_frame
        s, e = time_img[frame_ind]
        ts, xs, ys, ps, frames_img3, time_img = load_events(eve_dtype="aedat4", path=read_path_e)
        xs, ys, ts, ps = time_window(xs, ys, ts, ps, s, win)
        ts = ts - ts.min()
        delta_t = win
        frames_sharp = frames_sta
        frames_blur = frames_mov
        ###Optional
        ### update the time win and the events within the exposure time
        """s, e = time_img[frame_ind]
        win = int(e-s)
        ts, xs, ys, ps, frames_img3, time_img = load_events(eve_dtype="aedat4", path=read_path_e)
        xs, ys, ts, ps = time_window(xs, ys, ts, ps, s, win)
        t_c = time_img[frame_ind].mean()
        win = int(time_img[frame_ind][1] - time_img[frame_ind][0])
        ts, xs, ys, ps, frames_img3, time_img = load_events(eve_dtype="aedat4", path=read_path_e)
        xs, ys, ts, ps = time_window(xs, ys, ts, ps, s= int(t_c - win/2) , win=win)
        ts = ts - ts.min()
        delta_t = win
        """
    else:
        frames_sta = np.zeros(sensor_size)
        frames_mov =np.zeros(sensor_size)
    ### transform to torch
    xt, yt, tt, pt = np_to_torch(xs, ys, ts, ps , device)
    tt0 = tt[0]

    n_fig  = 5
    path = save_dir + "frames/"
    imgs  = frames_img[frame_ind:frame_ind+n_fig +1]
    #fig, axs = plt.subplots(nrows=2, ncols=5)
    for i in range(n_fig + 1):
        img = imgs [i]
        # axs[i%2, i//2].imshow(img, cmap='gray')
        # axs[i%2, i//2].set_title("frame %d" % i)
        # axs[i%2, i//2].axis('off')
        cv2.imwrite(path + str(i).zfill(4)+".png", img)


    #compute_dxy_roi(imgs[-1], imgs[0], sensor_size=(260, 346))
    ### should I retain the forward model?



    imgs_t = torch.from_numpy(imgs).to(device)
    sr_d = Simple_sr(holo_raw=imgs_t, mag=1,img_num=n_fig + 1, device = device)
    sr_d.pos_registration()
    flow = torch.from_numpy(sr_d.pos).to(device)     ### negtive y, negtive  x
    flow_xy = -flow[[1, 0], :]
    # flow_xy = (-flow[[1, 0], :]).T
    # Dxy_0_pc = flow_xy[1:, :] - flow_xy[:-1, :]
    #

    #### I need to papagate the image via baais optical flow method.




    M = 2
    imgs_t_sr = sr_model(imgs_t, M)/255
    imgs_lr = imgs_t/255
    img0 = imgs_t_sr[0, ...]
    I_pred_sr = nn.Parameter(torch.ones_like(img0)/2, requires_grad=True)
    kappa_l1 = 1e-5
    kappa_tv = 1e-2
    lr = [2e-2];loss_hist = []
    vars = [];
    vars += [{'params': I_pred_sr, 'lr': lr[0]}];
    optimizer = AdamP(vars);
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100 // 2, gamma=0.7, verbose=False); w_y, w_x= [0.1, 1]
    for iter in range(600):
        optimizer.zero_grad()
        warp_pred, mask = forward_mfsr_model(I_pred_sr, flow_xy, M)
        L_tot =  l1_loss_mask(warp_pred - imgs_lr, mask, beta=1.0)
        #L_tot = l2_loss_mask(warp_pred - imgs_lr, mask)
        L_tot += kappa_tv* tv_loss(x=I_pred_sr, tv_order=1, tv_tau=1e-3)
        #L_tot += tv_loss_weight(x=I_pred_sr, tv_order=1, tv_tau=1e-3, iso=False, w_xy=[kappa_tv * w_x, kappa_tv * w_y])
        #L_tot += kappa_l1 * l1_loss(I_pred_sr)
        loss_hist.append(L_tot.cpu().detach())
        L_tot.backward(retain_graph=False)  # Calculate the derivatives
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            if iter % 100 == 0:
                print("iter = {}: loss = {}, d_phi={}".format(iter, L_tot.data.cpu().numpy(),
                                                              I_pred_sr.grad.mean().cpu().numpy()))
            I_pred_sr.clamp_(min=0, max = 1)
    loss_hist = torch.as_tensor(loss_hist)
    plt.rcParams['font.size'] = 15
    plt.figure("compare", figsize=[10,5])
    plt.subplot(131)
    watch_tensor(img0)
    plt.xticks([]);plt.yticks([])
    plt.subplot(132)
    watch_tensor(I_pred_sr)
    plt.xticks([]);plt.yticks([])
    ax = plt.subplot(133)
    ax.plot(loss_hist)
    ax.set_title('Loss vs. iterations')
    ax.set_xlabel('iterations')
    ax.set_ylabel(r'$-\|y-\hat{y}\|$')
    ax.set_aspect(loss_hist.shape[0]/(loss_hist.max()-loss_hist.min()) * ax_asp)
    plt.xticks([]);plt.yticks([])
    plt.tight_layout()

    from utils.utils_viz import rotate_crop_image
    img_name = "mfsr_220"
    path = r"E:\\[X] NeuroSR\\fig\\result1\\"
    img = I_pred_sr.detach().cpu().numpy()  # norm(np.load(path + img_name + '.npy'));
    M = round(img.shape[0] / 260)
    cmap = "gray"
    center = (243, 123)
    crop = 220;50;
    img = rotate_crop_image(img, -55, center=(center[0] * M, center[1] * M), crop_size=[crop * M, crop * M], flip=True)
    # plt.figure("zoom")
    # plt.imshow(img, cmap)
    # os.mkdir( path + "sub1")
    #cv2.imwrite(path + "compare\\" + img_name + "_" + str(n_fig) + ".png", norm(img))



    ### how about solve a vedio?
    M = 2
    imgs_t_sr = sr_model(imgs_t, M) / 255
    imgs_lr = imgs_t / 255

    # Choose timestamps to solve
    T = imgs_t.shape[0]
    selected_t = [0, T // 2, T - 1]

    # Initialize 3 HR frames
    I_pred_srs = [nn.Parameter(torch.clone(imgs_t_sr[t]).detach(), requires_grad=True) for t in selected_t]
    I_lrs = [imgs_t[t] for t in selected_t]

    # Optimizer setup
    kappa_tv = 1e-2
    lr = [2e-2]
    loss_hist = []

    vars = [{'params': p, 'lr': lr[0]} for p in I_pred_srs]
    optimizer = AdamP(vars)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100 // 2, gamma=0.7, verbose=False)

    # Optimization loop
    for iter in range(900):
        optimizer.zero_grad()
        L_tot = 0

        for i, t in enumerate(selected_t):
            I_hr = I_pred_srs[i]
            # Shift flow so that flow[:, t] is the reference
            flow_shifted =flow_xy- flow_xy[:, t:t+1]
            warp_pred, mask = forward_mfsr_model(I_hr, flow_shifted, M)
            L = masked_l2(warp_pred - imgs_lr, mask)
            L += kappa_tv * tv_loss(x=I_hr, tv_order=1, tv_tau=1e-3)
            # L += tv_loss_weight(x=I_hr, tv_order=1, tv_tau=1e-3, iso=False, w_xy=[kappa_tv * w_x, kappa_tv * w_y])
            # L += kappa_l1 * l1_loss(I_hr)
            L_tot += L

        loss_hist.append(L_tot.cpu().detach())
        L_tot.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            if iter % 100 == 0:
                print("iter = {}: loss = {:.6f}, d_phi = {:.6f}".format(
                    iter, L_tot.item(),
                    torch.stack([x.grad.mean() for x in I_pred_srs]).mean().item()
                ))
                for I in I_pred_srs:
                    I.clamp_(min=0, max=1)

    plt.rcParams['font.size'] = 15
    plt.figure("compare tl", figsize=[10, 5])
    plt.subplot(231)
    watch_tensor(I_lrs[0])
    plt.subplot(232)
    watch_tensor(I_lrs[1])
    ax = plt.subplot(233)
    watch_tensor(I_lrs[2])
    plt.subplot(234)
    watch_tensor(I_pred_srs[0])
    plt.subplot(235)
    watch_tensor(I_pred_srs[1])
    ax = plt.subplot(236)
    watch_tensor(I_pred_srs[2])
    plt.tight_layout()





