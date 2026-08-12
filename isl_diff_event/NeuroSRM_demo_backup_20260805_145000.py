import matplotlib
#matplotlib.use("Qt5Agg")  # Must come before pyplot or anything GUI
import cv2
import numpy as np
import torch
import torchvision.transforms.functional
from torch import nn
from torch.optim import Adam, lr_scheduler
from trajectory_model.Diff_tratrectory import ContinuousPiecewiseLinear_Dxy_pw, integrate_piecewise_displacement
from matplotlib import pyplot as plt
torch.manual_seed(0)
torch.set_num_threads(10)  # default 20; cap CPU intra-op threads so leftover CPU ops don't peg all cores
from utils.utility import *
from utils.utils_event_flow import *
from function.optimizer import *
from gpu_mem_track import MemTracker
gpu_tracker = MemTracker()
# def forward_model(field):
#     diffraction_pattern = forward_wave_prop(field, params).abs()**2
#     return diffraction_pattern
import torchvision.transforms as transforms
def forward_model_EKLT_L(L,Dxy):
    return -forward_grad(L)[1] * Dxy[0] - forward_grad(L)[0] * Dxy[1]
def compute_ver_EKLT(x, Dxy):
    return -forward_grad(x)[1] * Dxy[1] + forward_grad(x)[0] * Dxy[0]
def forward_model_EKLT(I,Dxy,thre = 1.000001):
    L = lin_log(I,thre)
    return -forward_grad(L)[1] * Dxy[0] - forward_grad(L)[0] * Dxy[1]
def forward_model_EKLT_ver(I,Dxy,thre = 1.000001):
    L = lin_log(I,thre)
    return forward_grad(L)[1] * Dxy[1] + forward_grad(L)[0] * Dxy[0]
def forward_f2e(I, flow, thre, norm =False):
    if norm:
        I = I  / (torch.quantile(I , 0.99) + 1e-6)
    L = lin_log(I * 255, thre) / np.log(255)
    iwe = forward_model_EKLT_L(L,  flow)
    return iwe




sensor_size=(260, 346); ax_asp =  sensor_size[0]/ sensor_size[1];cy,cx = np.asarray(sensor_size)//2;
Ny, Nx = sensor_size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

M = 1 # SR magnification
save_path = 'E:/[X] NeuroSR/fig/'
############################### Generate Zernike phase ################################
# a_gt   = norm_torch(torch.from_numpy(cv2.resize(cv2.imread(read_path_f, cv2.IMREAD_GRAYSCALE), (Nx,Ny))).to(device),255**0.5)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
eve_dtype ="aedat4"
### Step 1: load data
#read_path_e = "E:/davis/pei/01.aedat4"
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
#read_path_e = r"F:\davis\dvSave-2026_06_06_22_55_35.aedat4"
#read_path_e = r"E:\davis\hk\hk.aedat4"
#read_path_e = r"E:\davis\yaping\object100\night\A18.aedat4"
#read_path_e = r"E:\davis\yaping\object100\night\A13.aedat4"
#read_path_e  = r"E:\davis\yuxing\40x_static\dvSave-2024_12_10_16_14_05.aedat4"
#read_path_e  = r"E:\davis\yuxing\40x_static\dvSave-2024_12_10_16_02_42.aedat4"
read_path_e = "E:/davis/pei/02.aedat4"###
#read_path_e ="E:/davis/pei/onion.aedat4"
#read_path_e ="E:/davis/shuo/01.aedat4"
#read_path_e ="E:/davis/shuo/line_scan2.aedat4"
#read_path_e =r"E:\davis\rotfer\10ms\shen.aedat4"  ## good data
#read_path_e =r"E:\davis\shuo\vib\02_1.aedat4"
#read_path_e ="E:/davis/pei/xiao_chang_01.aedat4"
#read_path_e  = r"E:\davis\yuxing\40x_static\dvSave-2024_12_10_16_14_05.aedat4";
#read_path_e ="E:/davis/pei/huafeng.aedat4"
#read_path_e ="E:/davis/pei/guoyin2.aedat4"
#save_fig_path = r"E:\[X] NeuroSR\fig\res_t\sam_1"
#ead_path_e = "E:\\davis\\yuxing\\1-paper_4x-2w\\dvSave-2024_12_10_15_46_37.aedat4"
#read_path_e  = r"E:\davis\yuxing\40x_1s\dvSave-2024_12_10_15_56_34.aedat4"
#read_path_e  = r"E:\[X] NeuroSR\event\lunchong.aedat4"
#read_path_e =r"E:\\davis\miao\9-4-24\rotifer-500um-10x-old-chips\davis-386\3.aedat4"
#read_path_e ="E:/davis/pei/qiuyin1.aedat4"
#read_path_e ="E:/davis/pei/xiao_chang_01.aedat4"
#read_path_e  = r"E:\[X] NeuroSR\event\lunchong.aedat4"
Blur_ind = True
sub_path = save_path+"lunchong/"
# img = cv2.imread(r"E:\[X] NeuroSR\fig\sample.png",cv2.IMREAD_GRAYSCALE)/255
# grad = np.gradient(img)[1]
# plt.figure("grad")
# grad = grad- np.median(grad)
# norm_1 = TwoSlopeNorm(vmin=grad.min(), vcenter=0, vmax=grad.max())
# plt.imshow(grad,cmap = "seismic", norm= norm_1)

### Step 2: select time window
"""   ##need to first watch the 3D scattering map to find a general window
ts, xs, ys, ps, frames_img3, time_img = load_events(eve_dtype = "aedat4", path= read_path_e); ts = ts-ts.min()
s = 2.45e6;win = int(8e4) 
xs, ys, ts, ps=time_window(xs, ys,ts, ps ,s =s, win = win);#ts = ts - ts.min();xs, ys, ts, ps=xs[::10], ys[::10], ts[::10], ps[::10]
plt.rcParams['font.size'] = 15
plot_scatter_bos_rect(xs, ys,ts,ps, polar=True,view = "normal45",sensor_size =sensor_size, axis_ind="on",alpha =  0.05,
alpha_k=0.00,plot =  False, save = save_path +"scatter_guoying.png", frame = frames_mov)
#plot_scatter_bos_rect(xs, ys,ts,ps, polar=True,view = "normal",sensor_size =sensor_size, axis_ind="off",alpha = 0.5,alpha_k=0.1,plot = True,save = None )
"""
# plot_scatter_bos_rect_roi(xsw, ysw,ts,ps, roi= roi_zoom, polar=True,view = "verticle",axis_ind="off",alpha =  0.05,
#alpha_k=0.00,plot = True, save = save_path +"scatter_roi_w.png")

if eve_dtype == "aedat4":
    ts, xs, ys, ps, frames_img3, time_img = load_events(eve_dtype="aedat4", path=read_path_e);
    ts -= ts.min()  # start from 0
    if len(frames_img3.shape) == 4:
        frames_img = np.zeros(frames_img3.shape[:3])
        for i in range(frames_img3.shape[0]):
            frames_img[i, :, :] = cv2.cvtColor(frames_img3[i, :, :, :], cv2.COLOR_BGR2GRAY)
    elif len(frames_img3.shape) == 3:
        frames_img = frames_img3
elif eve_dtype == "npy":
    ts, xs, ys, ps = load_events(eve_dtype="npy", path=read_path_e)
    ts -= ts.min()  # start from 0
#s =0.7e6;#win = int(5e4)    # guoyin2
#s = 2e6; win = int(2.5e5);
#s= 15.3e6; win = int(2e5); # shuo  01
#s= 10.1e6; win = int(2e5);  # shuo  line_scan2
#s=3.15e6;win = int(1e5)    # lunchong 40X shen
#s=2.25e6;win = int(1e5)    # lunchong 40X wei
#s=2.45e6;win = int(2e5);   # plant rhizome
#s=2.45e6;win = int(1e5)     # yangcong
# s = 0.3e6; win = int(1e5)  # huafeng
s = 2.45e6;win = int(2e5)  # 02
#s = 21.6e6;win = int(1e5)  # hk
#s=8.2e5; win = int(1e5)    # xiao chang
#s = 3e6;win = int(5e4)  # r"E:\davis\yaping\object100\night\A13.aedat4"
#s = 4.2e6;win = int(5e4)  # r"E:\davis\yaping\object100\night\A13.aedat4"
#s =0.4e6;win = int(1e5)    # qiuyin
#s = 9.9e6; win = int(1e5)  # lunchong
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
#s = 2.45e6;win = int(1e5);
# s = 0.82e6; win = int(1e5)
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
    frame_ind = np.where(time_img[:, 0] > s)[0][0]- 1 #np.where(time_img[:, 0] > s)[0][0] - 1 #frame_ind = np.where(time_img[:, 0] < s + win)[0][0];
    ##select frame
    frames_mov = frames_img[frame_ind,:,:]
    frames_sta =  frames_img[0, :, :]#frames_img[frame_ind+1,:,:] #frames_img[0, :, :]
    watch_tensor(frames_sta)
    #del frames_img,frames_img3
    time_win = time_img[frame_ind,:]
    t_c = int(time_win.mean())
    win_frame = int(time_img[frame_ind].max()-time_img[frame_ind].min())  # single-frame exposure = win lower bound
    s, e = time_img[frame_ind];
    # default mode: compare with 2 frames -> win spans frame_ind start to frame_ind+1 end (win_2)
    compare_2frame = True
    if compare_2frame and frame_ind + 1 < len(time_img):
        win = int(time_img[frame_ind + 1][1] - time_img[frame_ind][0])  # win_2
    # floor: win is at least the single-frame exposure (win_min = win_frame)
    if win < win_frame:
        win = win_frame
    print(f"Using full frame exposure: frame_ind={frame_ind}, s={s}, e={e}, win={win}, compare_2frame={compare_2frame}")
    ts, xs, ys, ps, frames_img3, time_img = load_events(eve_dtype="aedat4", path=read_path_e)
    xs, ys, ts, ps = time_window(xs, ys, ts, ps, s=s, win =win)
    ts = ts - ts.min()
    delta_t = win
    print(f"Selected {len(ts)} events in the exposure window")
    # ts, xs, ys, ps, frames_img3, time_img = load_events(eve_dtype="aedat4", path=read_path_e)
    # xs, ys, ts, ps = time_window(xs, ys, ts, ps, s=time_win[0], win=win)
    # ts = ts - ts.min()
    # delta_t = win
    frames_sharp = frames_sta
    frames_blur = frames_mov
    ###Optional
    ### update the time win and the events within the exposure time
    """s, e = time_img[frame_ind]
    win = int(e-s)
    ts, xs, ys, ps, frames_img3, time_img = load_events(eve_dtype="aedat4", path=read_path_e)
    xs, ys, ts, ps = time_window(xs, ys, ts, ps, s, win)"""
    """
    t_c = time_img[frame_ind].mean()
    win = int(time_img[frame_ind][1] - time_img[frame_ind][0])
    ts, xs, ys, ps, frames_img3, time_img = load_events(eve_dtype="aedat4", path=read_path_e)
    xs, ys, ts, ps = time_window(xs, ys, ts, ps, s= int(t_c - win/2) , win=win)
    ts = ts - ts.min()
    delta_t = win
    """
else:
    frames_sta = np.zeros(sensor_size)
    frames_mov = np.zeros(sensor_size)
del frames_img3

### transform to torch
xt, yt, tt, pt= np_to_torch(xs, ys, ts, ps , device)
tt0 =win_frame//2# tt[0]#0;

### Step 3: determine a good start point
ref_win=int(win/8)
eve_frame_s = window_event_to_img(xs, ys, ts, ps,sensor_size,s=0, win=ref_win)
eve_frame_e = window_event_to_img(xs, ys, ts, ps,sensor_size,s=ts.max()-ref_win, win=ref_win)
plt.rcParams['font.size'] = 15
plt.figure("watch the iwe and frame")
plt.subplot(221)
plt.imshow(eve_frame_s, "inferno")
plt.title("start")
plt.subplot(222)
plt.imshow(eve_frame_e,"inferno")
plt.title("end")
plt.subplot(223)
corr = corr2D(eve_frame_s, eve_frame_e)
scy, scx = np.asarray(sensor_size)/2
corr[int(scy),int(scx)] = 0
plt.imshow(corr)
corr_max  = np.where(corr == corr.max())
plt.title("corr_xy:" + str(corr_max[1][0])+","+str(corr_max[0][0]))
plt.subplot(224)
plt.imshow( frames_mov, cmap="gray")
plt.title("frames_mov")
# plt.subplot(224)
# plt.imshow(iwe_pred_lr)
corr = corr2D(eve_frame_s, eve_frame_e)
cxy= np.asarray([[int(scx)],[int(scy)]])
Dxy_0= -np.asarray([[corr_max[1][0]-scx],[corr_max[0][0] -scy ]])
print(Dxy_0)
Dxy_0_pred = torch.from_numpy(Dxy_0).to(device)



def forward_model_cmax(x, y, t, p, t0, Dxy,M =1, polar = False):
    xw,yw = warp_Disp(x, y, t, t0, Dxy)
    iwe = events_to_image_torch_sr(xw, yw, p, sensor_size, M,
                                   device=None,polar = polar)
    loss =- torch.var(torch.abs(iwe))
    return iwe,loss

def forward_model_vec_cmax(xt, yt, tt, pt, t0, Dxy,M =1, polar = True):       ### actually to = 0

    xw, yw = warp_events_traj_torch(xt, yt, tt, pt, Dxy, t0=t0,
                           batched=False, batch_indices=None)
    # iwe = events_to_image_torch_sr(xw, yw, pt, sensor_size,M,
    #                                device=None,polar = polar)
    # iwe = events_to_image_torch_sr_flow(xw, yw, pt, sensor_size, M, flow_xy=(Dxy.abs().max(1))[0], device=device,
    #                                     polar=polar)
    iwe = events_to_image_torch_sr_flow(xw, yw, pt, sensor_size, M, flow_xy=(Dxy.abs().max(1))[0], device=device, polar=polar)
    #iwe = T.functional.gaussian_blur(iwe.unsqueeze(0).unsqueeze(0), kernel_size=3, sigma=1).squeeze()
    loss = - torch.var(torch.abs(iwe))
    return iwe,loss,xw, yw

# Dxy_torch= torch.ones([2,tt.shape[0]]).to(device)
# Dxy_torch =Dxy_torch* Dxy_0_torch

#### We need to parameterize Dxy using other technology
# def piecewise_

"""
flow = generate_uniform_optical_flow(sensor_size, 30,3)
flow = generate_dense_optical_flow(sensor_size, 3)
cy, cx = sensor_size
cy=cy//2; cx=cx//2;
flow[:,cy-20: cy+20, cx-20:cx+20] = 0
frames_warp=warp_image_forward(frames_mov, flow)
ax = plt.subplot(221)
plot_ax_roi(ax,frames_mov)
ax = plt.subplot(222)
plot_ax_roi(ax,frames_warp)
flow_rgb, color_wheel, max_magnitude  = color_optical_flow(flow[0], flow[1])
plt.subplot(223)
plt.imshow(flow_rgb)
plt.subplot(224)
plt.imshow( color_wheel)
"""

### start diff_N part 1
#Dxy_pred = nn.Parameter(torch.from_numpy(Dxy_0).to(device),requires_grad=True)
#Dxy_pred = nn.Parameter(Dxy_torch,requires_grad=True)
#iwe_0 = forward_model_cmax(xt, yt, tt, pt, t0 = tt0, Dxy = torch.zeros_like(Dxy_pred))[0].detach().cpu().numpy()
#Dxy_para_torch=torch.zeros([2,1,3]).to(device)
#Dxy_para_torch[:,:,0] = Dxy_0_torch


t_win =  win
num_pieces = int(12); num_t_ref = 3
dt = np.arange(t_win); tts = torch.linspace(tt[0], tt[-1],   num_t_ref)
Dxy_vec = dt  * np.zeros_like(Dxy_0)/ (t_win)
Dxy_para_torch = torch.from_numpy(Dxy_vec).T.to(device)

traj_model = ContinuousPiecewiseLinear_Dxy_pw(t_win, num_pieces, device=device)
### Estimate the initial value
"""## Constant initialization
Vxy_0_pc = Dxy_0_pred.repeat(1, num_pieces).T / t_win
Dxy_0_pc = Dxy_0_pred.repeat(1, num_pieces).T / num_pieces
"""
## Autocorrelation initialization
segment_masks = ((tt.unsqueeze(-1) >= traj_model.segment_boundaries[:-1]) & (tt.unsqueeze(-1) < traj_model.segment_boundaries[1:])) ## in shape  of [n_event, num_piece]
tt_m = [(tt[segment_masks[:,i]]).detach().cpu() for i in range(num_pieces)]
iwes = []
for i in range(num_pieces+1):
    boundaries = traj_model.segment_boundaries.detach().cpu().numpy()
    width = traj_model.segment_widths[0].detach().cpu().numpy()
    iwes.append(window_event_to_img(xs, ys, ts, ps, sensor_size, s=boundaries[i]-width//2, win= width))
iwes_t = torch.from_numpy(np.asarray(iwes)).to(device)
from solver.flow_sr import Simple_sr
sr_d = Simple_sr(holo_raw=iwes_t, mag=1,img_num=num_pieces+1, device = device)
sr_d.pos_registration()
flow = torch.from_numpy(sr_d.pos).float().to(device)
flow_xy = (-flow[[1, 0], :]).T
Dxy_0_pc = flow_xy[1:,:] - flow_xy[:-1,:]
### Registration-based initialization
Dxy_pred_0 = traj_model(Dxy_0_pc).squeeze().T
Dxy_pc =  nn.Parameter(Dxy_0_pc.clone().to(device).float(), requires_grad=True)
Dxy_pc_np = Dxy_pc.detach().cpu().numpy()
### Cmax


vars = []
#vars += [{'params': Vxy_pc, 'lr':  1e-6}]
vars += [{'params': Dxy_pc, 'lr': 1e-2}]
loss_hist_v = []
optimizer = Adam(vars)
scheduler = lr_scheduler.StepLR(optimizer, step_size=400//2 , gamma=0.9)
pyramid_level = [1]
for iter in range(2100):
    optimizer.zero_grad()     # Essential for update the derivatives
    Dxy_pred = traj_model(Dxy_pc).squeeze().T
    tmp = forward_model_vec_cmax(xt, yt, tt, pt, t0=tts[0], Dxy=Dxy_pred, M=1, polar=False)[0]
    iwe = T.functional.gaussian_blur(tmp.unsqueeze(0).unsqueeze(0), kernel_size=3, sigma=1).squeeze()
    nvar = - torch.var(torch.abs(iwe))
    loss = nvar + traj_model.regularization_loss(lam=1*2e-1, mu=0, beta=1*1e-4 )#t_win#*t_win#+ kappar_l1 * l1_loss(iwe.abs())#+  kappar_l2* torch.sum(Dxy_pred**2).mean()+ kappar_l1 * l1_loss(Dxy_pred)
    loss.backward(retain_graph=False)     # Calculate the derivatives
    optimizer.step()
    scheduler.step()
    if iter % 100 == 0:
        loss_hist_v.append(loss.detach().cpu().numpy())
        print("iter = {}: loss = {}, d_phi={}".format(iter, loss.data.cpu().numpy(), Dxy_pc.grad.mean().cpu().numpy()))

reset_optimizer_and_params(optimizer, vars)
vx_dense, vy_dense = (traj_model.segment_masks.unsqueeze(-1) * traj_model.slopes).sum(dim=2).squeeze().T
polar_ind = True
M = 1;  M_xy= compute_M_xy(M, flow_xy=(Dxy_pred.abs().max(1))[0]).float(); mode="bilinear";sigma = 0.849  #'bilinear'mode="nearest"
xtw, ytw = forward_model_vec_cmax(xt, yt, tt, pt, t0=tt0, Dxy=Dxy_pred, M=1, polar=polar_ind)[2:]
iwe_pred_lr = events_to_image_torch(xtw, ytw, pt, sensor_size, device=device, polar = True)
iwe_pred_lr = iwe_pred_lr.detach().cpu().numpy()   ### to np
#
# iwe_pred_sr = Gau_events_to_image_torch_sr_flow(xtw, ytw, pt, np.asarray(sensor_size), M, flow_xy=(Dxy_pred.abs().max(1))[0], sigma = 1, device=device, polar=polar_ind)
# iwe_pred_sr = F.interpolate(iwe_pred_sr.unsqueeze(0).unsqueeze(0), (int(Ny * M), int(Nx * M)), mode="bicubic", align_corners=True).squeeze() *2   #mode="bilinear""bicubic"
# iwe_pred_sr = iwe_pred_sr.detach().cpu().numpy()   ### to np
def NeuroSR_Flow(M, type = "numpy",sigma = 1,t0 = tt0, mode="bilinear"):#
    xtw, ytw = forward_model_vec_cmax(xt, yt, tt, pt, t0, Dxy=Dxy_pred, M=1, polar=polar_ind)[2:]
    # iwe_flow = Gau_events_to_image_torch_sr_flow(xtw, ytw, pt, sensor_size, M, flow_xy=(Dxy_pred.abs().max(1))[0],
    #                                              sigma=sigma, device=device, polar=polar_ind)
    iwe_flow = Gau_events_to_image_torch_sr_flow(xtw, ytw, pt, sensor_size, M, flow_xy=(Dxy_pred.abs().max(1))[0],
                                                                                               sigma=sigma, device=device, polar=polar_ind)

    # iwe_flow = Gau_events_to_image_torch_sr(xtw, ytw, pt, sensor_size, M, sigma=sigma,
    #                                  device=device, polar=polar_ind)
    iwe_pred_sr = F.interpolate(iwe_flow.unsqueeze(0).unsqueeze(0), (int(Ny * M), int(Nx * M)), mode=mode).squeeze()
    if type == "numpy":
        iwe_pred_sr = iwe_pred_sr.detach().cpu().numpy()
    return iwe_pred_sr


def NeuroSR(M,type = "numpy", sigma = 1, t0 = tt0):
    xtw, ytw = forward_model_vec_cmax(xt, yt, tt, pt, t0, Dxy=Dxy_pred, M=1, polar=polar_ind)[2:]
    iwe_pred_sr = Gau_events_to_image_torch_sr(xtw, ytw, pt, sensor_size, M, sigma=sigma,
                                 device=device, polar=polar_ind)
    #iwe_pred_sr =  F.interpolate(iwe_flow.unsqueeze(0).unsqueeze(0), (int(Ny * M), int(Nx * M)), mode=mode, align_corners= True).squeeze()
    if type == "numpy":
        iwe_pred_sr = iwe_pred_sr.detach().cpu().numpy()
    return iwe_pred_sr
def NeuroSR_Gau(M, type = "numpy",sigma = 1,t0 = tt0):
    xtw, ytw = forward_model_vec_cmax(xt, yt, tt, pt, t0, Dxy=Dxy_pred, M=1, polar=polar_ind)[2:]
    from torchvision.transforms.functional import gaussian_blur
    iwe_sr = events_to_image_torch_sr(xtw, ytw, pt, sensor_size, M, device, polar=True)
    iwe_sr_Gau = gaussian_blur(iwe_sr.unsqueeze(0).unsqueeze(0), kernel_size=11,
                               sigma=M / 2*sigma ).squeeze()
    if type == "numpy":
        iwe_sr_Gau = iwe_sr_Gau.detach().cpu().numpy()
    return iwe_sr_Gau

# iwe_pred_sr = NeuroSR(M = M, sigma =sigma,t0 = tt0)
# we want to compare with the bi interplate
sub_win = int(win_frame)//8
iwe_0 = window_event_to_img(xs, ys, ts, ps, sensor_size, s=tt0-sub_win//2, win=sub_win)
iwe_all = window_event_to_img(xs, ys, ts, ps, sensor_size, s=tt0-win_frame//2, win=win_frame)



Dxy_ori=compute_dxy_roi(norm(computeLaplace(frames_sharp)), norm(np.abs(iwe_pred_lr)));    roi_0_rat = 2; #Dxy_ori=np.zeros_like(Dxy_ori); Dxy_ori=np.asarray([[0,0],[0,0]])
roi_0 = np.asarray([[sensor_size[1]/2-sensor_size[1]/roi_0_rat, sensor_size[1]/2+ sensor_size[1]/roi_0_rat],[sensor_size[0]/2+sensor_size[0]/roi_0_rat, sensor_size[0]/2- sensor_size[0]/roi_0_rat]])
######################################################################################### plot
#### plot and save parameter
cmap ="gray"#darkblue#"#darkblue#"gray"#"hot" #plt.rcParams['image.cmap']#"hot"#'inferno' #'gray'#'hot'#'inferno'#"gray"##'inferno'
cmap_np ="seismic";"gray";"seismic";"gray";"seismic";"gray";"seismic";#cmap_np ="seismic"#darkblue#"#darkblue;"seismic" #evk  #bi_bwr###"seismic"#"gray"#"bwr"#"seismic"#plt.rcParams['image.cmap']#"hot" #"bwr"  "seismic" "inferno" "hot" "seismic"  "gray"
iwe_cmap ="seismic";"gray"#"hot"
axis_ind = "off"
thre =1.0001

iwe_pred_sr = NeuroSR(M = M, sigma =sigma,t0 = tt0)
# frames_grad=forward_model_EKLT_np(frames_mov,Dxy_0, thre)
plot_full(frames_sharp, frames_blur, iwe_0, iwe_all, iwe_pred_lr, iwe_pred_sr, Dxy_0, cmap,iwe_cmap,cmap_np,axis_ind,
          roi = roi_0,Dxy_ori = Dxy_ori,  thre =thre, save_path = None)

"""  save
plot_SR(frames_mov, iwe_all, iwe_pred_lr, iwe_pred_sr, Dxy_0, cmap, iwe_cmap, cmap_np, axis_ind, roi = roi_0, Dxy_ori =Dxy_ori,thre = thre,save_path= save_fig_path +".png")
plot_motion_blur(frames_mov, frames_blur, iwe_0, iwe_all,iwe_pred_sr, cmap,iwe_cmap,cmap_np,axis_ind, roi = roi_0,Dxy_ori =Dxy_ori,thre =thre, save_path= save_fig_path +"_blur.png")
plot_full(frames_sharp, frames_blur, iwe_0, iwe_all, iwe_pred_lr, iwe_pred_sr, Dxy_0, cmap,iwe_cmap,cmap_np,axis_ind,
          roi = roi_0,Dxy_ori =Dxy_ori,  thre =thre, save_path = save_fig_path +cmap_np+"_full.png")
"""

frames_sr = fourier_upsampling(torch.from_numpy(frames_sharp), M).data
### plot zoom
roi_ind = 2
sz =44;33;int(74/2);33;
roi_sz = [sz,sz/346*260]#roi_sz = [sz,sz/346*260]
roi_cx =234;181;170;195;243;44;40;42;185;136; 229;251;50#int(scx)230;229;
roi_cy =121;200;50;124;115;80;38;80;106; 131; 110;80#int(scy)131;
roi_zoom = np.asarray([[roi_cx-roi_sz[0],roi_cx+roi_sz[0]],[roi_cy+roi_sz[1],roi_cy-roi_sz[1]]])
#roi_zoom = np.asarray([[roi_cx+roi_sz[0],roi_cx-roi_sz[0]],[roi_cy+roi_sz[1],roi_cy-roi_sz[1]]])

plot_full(frames_sharp, frames_blur, iwe_0, iwe_all, iwe_pred_lr, iwe_pred_sr, Dxy_0, cmap,iwe_cmap,cmap_np,axis_ind,
          roi = roi_zoom ,Dxy_ori =Dxy_ori,  thre =1.0001, save_path = None)
# plot_SR(frames_mov, iwe_all, iwe_pred_lr, iwe_pred_sr, Dxy_0, cmap, iwe_cmap, cmap_np, axis_ind, roi = roi_zoom, Dxy_ori =Dxy_ori,thre = thre)
# plot_motion_blur(frames_mov, frames_blur, iwe_0, iwe_all,iwe_pred_lr,iwe_pred_sr, cmap,iwe_cmap,"seismic",axis_ind, roi = roi_zoom,Dxy_ori =Dxy_ori,thre =thre)
M=2
sigma = 0.849
roi_h = roi_0
#from utils.utils_viz import plot_surface_event_roi
# plot_surface_event_roi(np.abs(iwe_pred_sr), roi_zoom,M)
plt.figure("compare interpolation",figsize=(12, 5))
xtw, ytw = forward_model_vec_cmax(xt, yt, tt, pt, tt0, Dxy=Dxy_pred, M=1, polar=polar_ind)[2:]; xsw, ysw = xtw.detach().cpu().numpy(), ytw.detach().cpu().numpy()
# plot_scatter_roi(xs, ys, ts, ps,polar = True, axis_ind  = "off", alpha = 0.02,alpha_k=0.002, plot = True, save = None)
ax = plt.subplot(241)
plot_ax_roi(ax,iwe_pred_lr, cmap_np , axis_ind, M=1, roi = roi_h,norm_ind = True);plt.title("bilinear vote lr")
ax = plt.subplot(242)
iwe_sr  =  NeuroSR(M, sigma = sigma, type = "numpy",t0 = tt0)
plot_ax_roi(ax,iwe_sr , cmap_np , axis_ind, M=M, roi = roi_h,norm_ind = True);plt.title("sr_grid + Gau-splat")
ax = plt.subplot(243)
iwe_sr_Gau  = NeuroSR_Gau(M = M, sigma =sigma, t0 = tt0)
plot_ax_roi(ax,iwe_sr_Gau , cmap_np , axis_ind, M=M, roi = roi_h,norm_ind = True);plt.title("sr_grid + Gau-blur")
ax = plt.subplot(244)
iwe_pred_sr = NeuroSR_Flow(M = M, sigma =0.5,t0 = tt0)
plot_ax_roi(ax, iwe_pred_sr, cmap_np , axis_ind, M=M, roi =roi_h,norm_ind =True);plt.title("bilinear rec grid")
roi_h = roi_zoom
ax = plt.subplot(245)
plot_ax_roi(ax,iwe_pred_lr, cmap , axis_ind, M=1, roi = roi_h,norm_ind = True)
ax = plt.subplot(246)
plot_ax_roi(ax,iwe_sr , cmap , axis_ind, M=M, roi = roi_h,norm_ind = True)
ax = plt.subplot(247)
plot_ax_roi(ax,iwe_sr_Gau  , cmap, axis_ind, M=M, roi = roi_h,norm_ind = True)
ax = plt.subplot(248)
plot_ax_roi(ax, iwe_pred_sr, cmap , axis_ind, M=M, roi =roi_h,norm_ind =True)
plt.tight_layout()



""" save
plot_SR(frames_mov, iwe_all, iwe_pred_lr, iwe_pred_sr, Dxy_0, cmap, iwe_cmap, "seismic", axis_ind, roi = roi_zoom, 
Dxy_ori =Dxy_ori,thre = thre,save_path= save_fig_path +"SR_zoom"+ str(roi_ind) +".png")
plot_motion_blur(frames_mov, frames_blur, iwe_0, iwe_all,iwe_pred_lr,iwe_pred_sr, cmap,iwe_cmap,"seismic",axis_ind, roi = roi_zoom,Dxy_ori =Dxy_ori,thre =thre,
save_path= save_fig_path +"1_blur_zoom"+ str(roi_ind) +".png")
plot_full(frames_sharp, frames_blur, iwe_0, iwe_all, iwe_pred_lr, iwe_pred_sr, Dxy_0, cmap,iwe_cmap,cmap_np,axis_ind,
          roi = roi_zoom,Dxy_ori =Dxy_ori,  thre =thre, save_path = save_fig_path +"_full_zoom"+ str(roi_ind) + str(M) + cmap_np+".png")
"""


def save_png_npy(arr, arr_name= None , output_dir = sub_path, save_npy = False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if arr_name == None:
        arr_name = next((name for name, value in globals().items() if value is arr), None)
    if save_npy:
        np.save(sub_path + arr_name + ".npy", arr);
    cv2.imwrite(sub_path + arr_name + ".png", norm(arr))

#

"""
#### plot3d
polar = True
alpha = 0.01
alpha_k = 0.005
pos_loc = np.where(ps == 1)
neg_loc = np.where(ps == 0)
pos_xs, pos_ys, pos_ts = xs[pos_loc], ys[pos_loc], ts[pos_loc]
neg_xs, neg_ys, neg_ts = xs[neg_loc], ys[neg_loc], ts[neg_loc]

fig = plt.figure("Spatio-temporal stream", figsize=(10, 10))
ax = fig.add_subplot(projection='3d')
ax.set_box_aspect([1.0, 2.0 ,ax_asp])
if polar == True:
    ax.scatter(alpha=alpha, xs=pos_xs, ys=pos_ts/1e6, zs=pos_ys, c='r', marker=".", clip_on=True)
    ax.scatter(alpha=alpha, xs=neg_xs, ys=neg_ts/1e6, zs=neg_ys, c='b', marker=".", clip_on=True)
ax.scatter(xs=xs, alpha=alpha_k, ys=0, zs= ys, marker=".", c='k')  #ys=(ts.max())/1e6/2
# Dxy_para_torch[:,:,0] * dt + Dxy_para_torch[:,:,1]* torch.pow(dt,2) + Dxy_para_torch[:,:,2] * torch.pow(dt,3)
ax.plot(xs=cx+10*Dx, zs=cy+10*Dy, ys=t_cont/1e6, c = "g", label='tenth amplified parametric curve')
plt.legend()
#ax.set_xlabel('X')
# ax.set_zlabel('Y')
#ax.set_ylabel('t/s')
ax.set_xlim3d(0,sensor_size[1]-1 )
ax.set_zlim3d(0, sensor_size[0]-1)
ax.set_ylim3d(ts.min()/1e6, (ts.max())/1e6 )
ax.view_init(azim=90, elev=90)
"""


#ax.set_box_aspect([sensor_size[1]/sensor_size[0], 2, 1.0])
#plt.gca.set_box_aspect((1,1/1000,1))
# if view =="vertical":
#     ax.view_init(azim=0, elev=90)
# elif view =="side":

# elif view == "r-side":

# elif view =="normal":

# elif view =="normal45":

# elif view =="lateral":

# elif view == "reverse":



plt.rcParams.update({'font.size': 15})
#### interp the kernel
seg_width = t_win/num_pieces
Vxy =Dxy_pc.T.detach().cpu().numpy()/seg_width
Dxy=Dxy_pred.detach().cpu().numpy()
Vx,Vy =Vxy
Dx,Dy =Dxy
Pos  = integrate_piecewise_displacement(Dxy_pc).T.detach().cpu().numpy()
Dx_pc, Dy_pc = Pos
Pos_f  = integrate_piecewise_displacement(Dxy_0_pc).T.detach().cpu().numpy()
Dx_pc_f, Dy_pc_f = Pos_f
t_cont = np.linspace(0, t_win, t_win+1)
t_pc = np.linspace(0, t_win, num_pieces)
plt.figure("watch loss, traj and velo")
ax=plt.subplot(221)
plt.plot(loss_hist_v)
plt.title('Loss vs. iterations')
plt.xlabel('iterations')
plt.ylabel(r'$-var(IWE)$')
plt.subplot(222)
plt.plot(t_pc/1e6, Vx*1e3, label='V_x')
plt.plot(t_pc/1e6, Vy*1e3, label='V_y')
plt.legend()
# plt.title("Velocity pixel/ms")
plt.subplot(223)
plt.plot(t_cont/1e6, Dx, label='D_x')
plt.plot (t_cont/1e6, Dy, label='D_y')
# plt.title("Displacement")
plt.legend()
plt.subplot(224)
# plt.title("Traj. xy")
# plt.plot(cx+Dx, cy+Dy, marker='o', color='red', alpha=0.5)
#plt.plot(cx+Dx[:win_frame,...], cy+Dy[:win_frame,...], marker='.', color='blue'  , label='Traj. Frame')
plt.plot(cx+np.insert(Dx_pc, 0,0), cy+np.insert(Dy_pc, 0,0), marker='.', color='red', label='Traj. Optimized')
plt.plot(cx+np.insert(Dx_pc_f, 0,0), cy+np.insert(Dy_pc_f, 0,0), marker='.', color='blue', label='Traj. Before')
flow_max_amp = np.linalg.norm(np.abs(Dxy).max(1),2)
plt.xlim(cx-flow_max_amp,cx+ flow_max_amp)
plt.ylim(cy+ flow_max_amp,cy-flow_max_amp)
plt.legend()
# plt.subplot(224)
# watch_tensor(kernel_pred)
plt.tight_layout()

print("the piece num is ", int(num_pieces))


sigma = 0.849;0.5;1;   tt0 =win_frame//2
#### Step final: link to frame
M=1;    scl = win_frame/win;
kernel_sr = generate_motion_blur_kernel_Dxy_sr(Dxy_pred[:,:win_frame], M, device, kernel_size=21, sigma  =  0.5, t0 = tt0)
thre=1+1e-6; # impractical para
iwe_gt = NeuroSR(M, type = "torch", sigma = sigma, t0 = tt0).detach()  # #mask_static = torch.from_numpy((iwe_0 ==0)).to(device=device, dtype=torch.bool); mask_mov =1- mask_static.float()
#iwe_gt = NeuroSR_Flow(M, type = "torch", sigma = 0.5, t0 = tt0).detach()
border_mask = torch.zeros(iwe_gt.shape).to(device).float();border_mask[1:-1, 1:-1] = 1; #mask_f =  iwe_gt.abs() < 1; #plt.imshow(mask_f.detach().cpu())
border_mask_lr = torch.zeros(iwe_pred_lr.shape).to(device).float();border_mask_lr[1:-1, 1:-1] = 1; #mask_f =  iwe_gt.abs() < 1; #plt.imshow(mask_f.detach().cpu())
iwe_gt *= border_mask
iwe_gt_norm = iwe_gt/np.log(255)#/iwe_gt.abs().max() #norm_torch(iwe_gt,1)
I_lr= torch.from_numpy(frames_mov).to(device);
L_lr = lin_log(I_lr.to(device).float(), thre);
I_lr_norm = I_lr / 255  # norm_torch(I_lr,1)#I_lr/255#norm_torch(I_lr,1)#
I_sr = F.interpolate(I_lr.detach().unsqueeze(0).unsqueeze(0), (int(Ny * M), int(Nx * M))).to(device).float().squeeze()
I_sr_norm =  I_sr/255
L_lr_norm=L_lr/np.log(255)#norm_torch(L_lr,1)#L_lr/np.log(255)#
iwe_gt_lr = torch.from_numpy(iwe_pred_lr).to(device).float(); loc =  iwe_gt.abs() < 1e-2;
mask_e = torch.ones_like(border_mask).detach(); mask_e[ loc] = 0#; plt.imshow(mask_f.detach().cpu())
L_lr_grad= norm_torch(forward_model_EKLT(I_lr, Dxy_pc.sum(0))).detach().cpu().numpy()
L_sr = lin_log(I_sr, thre) ; L_sr_norm= norm_torch(L_sr,1)
iwe_gt_ver_sr = norm_torch(compute_ver_EKLT(L_sr,  Dxy_pc.sum(0)))
#### Start optimization
reset_optimizer_and_params(optimizer, vars)
#I_pred= nn.Parameter(torch.ones_like(I_sr_norm).detach()/2, requires_grad=True)
I_pred= nn.Parameter(torch.clone(I_sr_norm).detach(), requires_grad=True)
kernel_pred = torch.clone(kernel_sr).detach()
filter_sr = conv2d_from_kernel(torch.flip(kernel_pred, [0, 1]), 1, device, padding="valid")
pad_size  = torch.div(torch.tensor(iwe_gt.shape) -  torch.tensor(filter_sr(iwe_gt.detach().unsqueeze(0).unsqueeze(0)).squeeze() .shape),2,rounding_mode='floor')[0]*2
bg_pred = nn.Parameter(torch.clone(torch.zeros_like(I_lr_norm).float().detach()), requires_grad = True);  alter = False
thre_pred = nn.Parameter(torch.tensor(thre).float().to(device), requires_grad=False)
Dxy_pc_pred =  nn.Parameter(Dxy_pc.clone().detach().to(device).float(), requires_grad= True) ;
iwe_gt_non_0 = iwe_gt.abs() [iwe_gt.abs() >0].detach()
mask = torch.zeros_like(border_mask).detach().to(device);mask[torch.where(iwe_gt.abs() >= torch.quantile(iwe_gt_non_0, 0.2))]= 1#;watch_tensor(mask_e)
mask_d  = down_sampling(mask, M).detach()
lr = [4e-3, 1e-4];loss_hist=[]; pyramid_level =[1, M]; conv_type = "conv";
kappa_frame = 1;  #conv_type = "fft" # "conv" # "none" # "fft"  1*1e-2
fed_scl  = 2e3; kappa_event = 1*fed_scl; kappa_tv = 4*1e-2; kappa_l0=0; kappa_l1_bg = 2e-1; kappa_tv_bg = kappa_tv;      beta = 0.1 #kappa_kernel = 1e-1;
###  fed_scl  = 1*1e4; kappa_event = fed_scl; kappa_tv =2e-1; kappa_l1_bg  = 2e-1;kappa_tv_bg = 1e-1   ## this is for the qiuyin data
vars = [];vars += [{'params': I_pred, 'lr': lr[0]}];vars += [{'params': bg_pred, 'lr': lr[0]}];
vars += [{'params': Dxy_pc_pred, 'lr': lr[1]}]
optimizer = Adam(vars);scheduler = lr_scheduler.StepLR(optimizer, step_size=200//2, gamma=0.7);
Dxy_sum = Dxy_pc.sum(0).detach();Dxy_sum_np =Dxy_sum.detach().cpu().numpy(); max_scl = np.max(np.abs(Dxy_sum_np ));  w_y, w_x= np.abs(Dxy_sum_np)/max_scl;
# from solver.psf_pytorch import cho_style_psf_postprocess
for iter in range(2100):
    optimizer.zero_grad()
    #I_pred = torch.exp(L_pred)
    #
    L_pred = lin_log(I_pred * 255, thre_pred)/np.log(255)
    iwe_pred = forward_model_EKLT_L(L_pred, Dxy_sum) * border_mask.detach()
    #iwe_pred =  forward_f2e(I_pred , Dxy_sum, thre_pred)
    L_tot = 0
    if iter == 0:
        L_D = 0;
        nvar = 0
    else:
        Dxy_sum = Dxy_pc_pred.sum(0)
        Dxy_dense_pred = traj_model(Dxy_pc_pred).squeeze().T
        nvar, xtw, ytw = forward_model_vec_cmax(xt, yt, tt, pt, tt0, Dxy=Dxy_dense_pred, M=1, polar=polar_ind)[1:]
        # iwe_flow = Gau_events_to_image_torch_sr_flow(xtw, ytw, pt, sensor_size, M, flow_xy=Dxy_sum,
        #                                          sigma=sigma, device=device, polar=polar_ind)
        # iwe_pred_sr = F.interpolate(iwe_flow.unsqueeze(0).unsqueeze(0), (int(Ny * M), int(Nx * M)), mode=mode).squeeze()* border_mask.detach()
        iwe_pred_sr = Gau_events_to_image_torch_sr(xtw, ytw, pt, sensor_size, M, sigma=sigma,
                                                   device=device, polar=polar_ind) * border_mask.detach()
        iwe_pred = forward_model_EKLT_L(L_pred, Dxy_sum) * border_mask.detach()
        L_D = l2_loss(iwe_pred / torch.norm(iwe_pred, 2) - iwe_pred_sr / torch.norm(iwe_pred_sr, 2))
        # L_D = l1_loss(iwe_pred / torch.norm(iwe_pred, 2) - iwe_pred_sr / torch.norm(iwe_pred_sr, 2), beta)
    # if iter == 0:
    #     L_D = 0
    # else:
    #
    #     L_D = l2_loss(iwe_pred / torch.norm(iwe_pred, 2) - iwe_gt.detach() / torch.norm(iwe_gt.detach(), 2))
    #     #L_D = l1_loss(iwe_pred / torch.norm(iwe_pred, 2) - iwe_gt.detach() / torch.norm(iwe_gt.detach(), 2))  ###for qiuqyin
    #     #L_D = l2_loss(iwe_pred / torch.norm(iwe_pred, 2) - iwe_gt.detach() / torch.norm(iwe_gt.detach(), 2))  ###for qiuqyin
    # #L_D = l2_loss(iwe_pred - iwe_gt_norm)
    L_tot += kappa_event * L_D

    kernel_pred = generate_motion_blur_kernel_Dxy_pc(Dxy_pc_pred, M, device,
                                                     kernel_size=21, sigma=0.5,
                                                     t_s=0, t_e=win_frame,  # 物理时间区间（μs）
                                                     t0=tt0,
                                                     t_win=win,  # 总时间窗口（μs）
                                                     samples_per_seg= int(20))



    # apply the motion-blur kernel to the predicted frame (see optimizer.blur_frame)
    I_pred_b = blur_frame(I_pred, kernel_pred, conv_type=conv_type, device=device)
    # ny, nx = I_pred.shape
    # nky, nkx = kernel_pred.shape
    # p2d = (int(nkx / 2), int(nkx / 2), int(nky / 2), int(nky / 2))  # pad last dim and 2nd to last
    # I_pred_pad = F.pad(I_pred.unsqueeze(0).unsqueeze(0), p2d, "replicate").squeeze()  # pad A
    # filter_sr = conv2d_from_kernel(torch.flip(kernel_pred, [0, 1]), 1, device, padding="valid")
    # I_pred_b = filter_sr(I_pred_pad.unsqueeze(0).unsqueeze(0)).squeeze()[int(nky / 2):(ny + int(nky / 2)), int(nkx / 2):(nx + int(nkx / 2))]
    # I_pred_b = convolve_with_fft_replicate(I_pred, kernel_pred)
    if M == 1:
        I_pred_lr_b = I_pred_b + bg_pred
    else:
        I_pred_lr_b  = down_sampling (I_pred_b, M) + bg_pred
    #pad_size  = torch.div(torch.tensor(iwe_gt_lr.shape) -  torch.tensor(I_pred_lr_b.shape),2,rounding_mode='floor')
    #L_pred_lr_b = lin_log(I_pred_lr_b*255, thre_pred)/np.log(255)

    L_F = 0

    #c_I = compute_c(I_pred_lr_b, I_lr_norm)#.detach();
    #L_F += l2_loss(I_pred_lr_b * c_I - I_lr_norm )
    L_F += l2_loss(torch.sqrt((I_pred_lr_b ).clamp(min=0.0) + 1e-8) - torch.sqrt(I_lr_norm))
    #L_F += l1_loss(I_pred_lr_b * c_I- I_lr_norm)
    L_tot += kappa_frame * L_F
    L_tot += kappa_tv * tv_loss_flow(I_pred, tv_order=1, tv_tau=1e-4, iso=True,  Dxy =Dxy_sum_np/max_scl)
    #L_tot += kappa_tv * tv_loss(I_pred, tv_order=1, tv_tau=1e-4, iso=True)
    #L_tot += kappa_l0 * smooth_l0_weight(I_pred, w_xy=[kappa_l0 * w_x, kappa_l0 * w_y])
    #L_tot += kappa_tv * tv_loss(L_pred, tv_order=2, tv_tau=1e-3)
    #L_tot += kappa_tv_bg * tv_loss(bg_pred, tv_order=1,tv_tau=1e-4, iso=True)
    #L_tot += kappa_tv_bg * tv_loss_flow(bg_pred, tv_order=1, tv_tau=1e-4, Dxy=Dxy_sum_np / max_scl)
    L_tot += kappa_l1_bg * l1_loss(bg_pred, 1e-6)
    #L_tot += kappa_mid * mid_loss_flow(I_pred_b, I_srs[1].detach(), M, Dxy_pred, tc_iwes)
    L_tot += 1 * (nvar + traj_model.regularization_loss(lam=1 * 2e-2, mu=0, beta=1 * 1e-4))

    L_tot.backward()  # Calculate the derivatives
    optimizer.step()
    scheduler.step()
    if alter == True:
        if iter > 200:
            # bg_pred.requires_grad = True
            Dxy_pc_pred.requires_grad = True
    with torch.no_grad():
        #kernel_pred = cho_style_psf_postprocess(kernel_pred)
        kernel_pred.clamp_(min=0, max=None)
        kernel_pred /= kernel_pred.sum()
        I_pred.clamp_(min=0, max=1)
        #thre_pred.clamp_(min=1+1e-9, max=20)
        #bg_pred.data[mask_static] = I_lr_norm[mask_static].float() -1/2
    if iter==1 or ((iter % 100 == 0 ) & (iter >0)):
        #print("iter = {}: loss = {}, d_phi={}".format(iter, L_tot.data.cpu().numpy(),  I_pred.grad.mean().cpu().numpy()))
        print("iter = {}: loss = {}".format(iter, L_tot.data.cpu().numpy()))
        loss_hist.append(L_tot.detach().cpu())
print("thre = ", thre_pred.data.cpu().numpy()); #iwe_pred_ver = compute_ver_EKLT(L_pred,  Dxy_pc.sum(0))
reset_optimizer_and_params(optimizer, vars)
"""
iwe_pred_ver = forward_model_EKLT_ver(I_pred,  Dxy_pc_pred.sum(0))
iwe_pred_f = forward_model_EKLT(I_lr_norm*255,  Dxy_pc_pred.sum(0))
iwe_pred_ver_f = forward_model_EKLT_ver(I_lr_norm*255,  Dxy_pc_pred.sum(0),thre = 1.000001)
iwe_pred_ver_f_gt = forward_model_EKLT_ver(torch.tensor(frames_sta).to(device),  Dxy_pc_pred.sum(0))
iwe_pred_f_gt = forward_model_EKLT(torch.tensor(frames_sta).to(device),  Dxy_pc_pred.sum(0))
Dxy_roi=compute_dxy_roi(norm(computeLaplace(frames_sharp)), norm(np.abs(iwe_pred_lr)));
cmap_d ="inferno" ; "gray";"magma";"hot" ;
cmap_phi = "viridis";"twilight"; "hot" ; cmap = "gray"; plt.rcParams.update({'font.size': 15})
cmap_np = "seismic";"twilight";
fig = plt.figure("compare frame 2", figsize=(16,6))
ax = plt.subplot(241)
# ax.set_aspect(len(loss_hist)/ (np.max(loss_hist) - np.min(loss_hist))* ax_asp)
# ax.plot(loss_hist, c="r")
#plot_ax_roi(ax,bg_pred.detach().cpu(), cmap, axis_ind, roi = roi_0,  M = 1);
plot_ax_roi(ax,frames_sta, cmap, axis_ind, roi = roi_0,  M = 1);
ax = plt.subplot(242)
plot_ax_roi(ax,I_pred.detach().cpu(), cmap, axis_ind, roi = roi_0,  M = M);
# Dx_pc, Dy_pc = Dxy_pc_pred.detach().cpu().T
# ax.set_aspect(Dx_pc.shape[0] / (np.max(Dxy_pc_pred.detach().cpu().numpy()) - np.min(Dxy_pc_pred.detach().cpu().numpy())) * ax_asp)
# ax.plot(Dx_pc, c="b")
# ax.plot(Dy_pc, c="r")
ax1 = plt.subplot(243)
plot_ax_roi(ax1,iwe_pred_ver_f.detach().cpu(), cmap_np, axis_ind, roi = roi_0, M = 1, norm_ind = True );
ax1 = plt.subplot(244)
plot_ax_roi(ax1,iwe_pred_f.detach().cpu(), cmap_np, axis_ind, roi = roi_0, M = 1, norm_ind = True );
# plot_ax_roi(ax1,frames_mov, cmap, axis_ind, roi = roi_0,  M = 1);
ax = plt.subplot(245)
plot_ax_roi(ax,iwe_pred_ver_f_gt.detach().cpu(), cmap_np, axis_ind, roi = roi_0, M = M, norm_ind = True );
ax = plt.subplot(246)
#plot_ax_roi(ax,bg_pred.abs().detach().cpu(), cmap_d, axis_ind, roi = roi_0, M = M);
plot_ax_roi(ax,iwe_pred_ver.detach().cpu(), cmap_np, axis_ind, roi = roi_0, M = 1, norm_ind = True );
ax= plt.subplot(247)
plot_ax_roi(ax,iwe_pred_f_gt.detach().cpu(), cmap_np, axis_ind, roi = roi_0, M = 1, norm_ind = True );
ax1 = plt.subplot(248)
plot_ax_roi(ax1,iwe_pred_sr.detach().cpu(), cmap_np, axis_ind, roi = roi_0, M = M, norm_ind = True );
#plot_ax_roi(ax1,frames_sta, cmap, axis_ind, roi = roi_0,  M = 1);
# plot_ax_roi(ax1,I_lr_norm.detach().cpu(), cmap, axis_ind, roi = roi_0,  M = 1);
plt.tight_layout()
"""


#### Step final: link to frame
M=2;
kernel_sr = generate_motion_blur_kernel_Dxy_sr(Dxy_pred[:,:win_frame], M, device, kernel_size=21, sigma  =  0.5, t0 = tt0)
thre=1+1e-6; # impractical para
iwe_gt = NeuroSR(M, type = "torch", sigma = sigma, t0 = tt0).detach()  # #mask_static = torch.from_numpy((iwe_0 ==0)).to(device=device, dtype=torch.bool); mask_mov =1- mask_static.float()
border_mask = torch.zeros(iwe_gt.shape).to(device).float();border_mask[1:-1, 1:-1] = 1; #mask_f =  iwe_gt.abs() < 1; #plt.imshow(mask_f.detach().cpu())
border_mask_lr = torch.zeros(iwe_pred_lr.shape).to(device).float();border_mask_lr[1:-1, 1:-1] = 1; #mask_f =  iwe_gt.abs() < 1; #plt.imshow(mask_f.detach().cpu())
iwe_gt *= border_mask
iwe_gt_norm = iwe_gt/np.log(255)#/iwe_gt.abs().max() #norm_torch(iwe_gt,1)
I_lr= torch.from_numpy(frames_mov).to(device);
L_lr = lin_log(I_lr.to(device).float(), thre);
I_lr_norm = I_lr / 255  # norm_torch(I_lr,1)#I_lr/255#norm_torch(I_lr,1)#
I_sr = F.interpolate(I_lr.detach().unsqueeze(0).unsqueeze(0), (int(Ny * M), int(Nx * M))).to(device).float().squeeze()
I_sr_norm =  I_sr/255
L_lr_norm=L_lr/np.log(255)#norm_torch(L_lr,1)#L_lr/np.log(255)#
iwe_gt_lr = torch.from_numpy(iwe_pred_lr).to(device).float(); loc =  iwe_gt.abs() < 1e-2; mask_e = torch.ones_like(border_mask).detach(); mask_e[ loc] = 0#; plt.imshow(mask_f.detach().cpu())
L_lr_grad= norm_torch(forward_model_EKLT(I_lr, Dxy_pc.sum(0))).detach().cpu().numpy()
L_sr = lin_log(I_sr, thre) ; L_sr_norm= norm_torch(L_sr,1)
iwe_gt_ver_sr = norm_torch(compute_ver_EKLT(L_sr,  Dxy_pc.sum(0)))
#### Start optimization
reset_optimizer_and_params(optimizer, vars)
#I_pred= nn.Parameter(torch.ones_like(I_sr_norm).detach()/2, requires_grad=True)
I_perd_sr = F.interpolate(I_pred.detach().unsqueeze(0).unsqueeze(0), (int(Ny * M), int(Nx * M))).to(device).float().squeeze()
I_pred= nn.Parameter(torch.clone(I_perd_sr ).detach(), requires_grad=True)
kernel_pred = torch.clone(kernel_sr).detach()
filter_sr = conv2d_from_kernel(torch.flip(kernel_pred, [0, 1]), 1, device, padding="valid")
pad_size  = torch.div(torch.tensor(iwe_gt.shape) -  torch.tensor(filter_sr(iwe_gt.detach().unsqueeze(0).unsqueeze(0)).squeeze() .shape),2,rounding_mode='floor')[0]*2
bg_pred = nn.Parameter(torch.clone(torch.clone(bg_pred).float().detach()), requires_grad =True); alter =False
thre_pred = nn.Parameter(torch.tensor(thre).float().to(device), requires_grad=False)
iwe_gt_non_0 = iwe_gt.abs() [iwe_gt.abs() >0].detach()
mask = torch.zeros_like(border_mask).detach().to(device);mask[torch.where(iwe_gt.abs() >= torch.quantile(iwe_gt_non_0, 0.2))]= 1#;watch_tensor(mask_e)
mask_d  = down_sampling(mask, M).detach()
lr = [4e-3, 1e-4];loss_hist=[]; pyramid_level =[1, M]; conv_type = "conv";
#kappa_frame = 1;  #conv_type = "fft" # "conv" # "none" # "fft"  1*1e-2
#fed_scl  = 1e3; kappa_event = 1*fed_scl; kappa_tv = 4*1e-2; kappa_l0=0; kappa_l1_bg = 1e-1; kappa_tv_bg = 0*1*1e-4;      beta = 0.1 #kappa_kernel = 1e-1;
###  fed_scl  = 1*1e4; kappa_event = fed_scl; kappa_tv =2e-1; kappa_l1_bg  = 2e-1;kappa_tv_bg = 1e-1   ## this is for the qiuyin data
vars = [];vars += [{'params': I_pred, 'lr': lr[0]}];vars += [{'params': bg_pred, 'lr': lr[1]}];
optimizer = AdamP(vars);scheduler = lr_scheduler.StepLR(optimizer, step_size=200//2, gamma=0.7);
Dxy_sum = Dxy_pc.sum(0).detach();Dxy_sum_np =Dxy_sum.detach().cpu().numpy(); max_scl = np.max(np.abs(Dxy_sum_np ));  w_y, w_x= np.abs(Dxy_sum_np)/max_scl;
# from solver.psf_pytorch import cho_style_psf_postprocess
for iter in range(2100):
    optimizer.zero_grad()
    #I_pred = torch.exp(L_pred)
    #
    L_pred = lin_log(I_pred * 255, thre_pred)/np.log(255)
    iwe_pred = forward_model_EKLT_L(L_pred, Dxy_sum) * border_mask.detach()
    #iwe_pred =  forward_f2e(I_pred , Dxy_sum, thre_pred)
    L_tot = 0
    if iter == 0:
        L_D = 0
    else:
        L_D = l2_loss(iwe_pred / torch.norm(iwe_pred, 2) - iwe_gt.detach() / torch.norm(iwe_gt.detach(), 2))
        #L_D = l1_loss(iwe_pred / torch.norm(iwe_pred, 2) - iwe_gt.detach() / torch.norm(iwe_gt.detach(), 2))  ###for qiuqyin
        #L_D = l2_loss(iwe_pred / torch.norm(iwe_pred, 2) - iwe_gt.detach() / torch.norm(iwe_gt.detach(), 2))  ###for qiuqyin
    #L_D = l2_loss(iwe_pred - iwe_gt_norm)
    L_tot += kappa_event * L_D
    # apply the motion-blur kernel to the predicted frame (see optimizer.blur_frame)
    I_pred_b = blur_frame(I_pred, kernel_pred, conv_type=conv_type, device=device)
    # ny, nx = I_pred.shape
    # nky, nkx = kernel_pred.shape
    # p2d = (int(nkx / 2), int(nkx / 2), int(nky / 2), int(nky / 2))  # pad last dim and 2nd to last
    # I_pred_pad = F.pad(I_pred.unsqueeze(0).unsqueeze(0), p2d, "replicate").squeeze()  # pad A
    # filter_sr = conv2d_from_kernel(torch.flip(kernel_pred, [0, 1]), 1, device, padding="valid")
    # I_pred_b = filter_sr(I_pred_pad.unsqueeze(0).unsqueeze(0)).squeeze()[int(nky / 2):(ny + int(nky / 2)), int(nkx / 2):(nx + int(nkx / 2))]
    # I_pred_b = convolve_with_fft_replicate(I_pred, kernel_pred)
    if M == 1:
        I_pred_lr_b = I_pred_b + bg_pred
    else:
        I_pred_lr_b  = down_sampling (I_pred_b, M) + bg_pred
    #pad_size  = torch.div(torch.tensor(iwe_gt_lr.shape) -  torch.tensor(I_pred_lr_b.shape),2,rounding_mode='floor')
    #L_pred_lr_b = lin_log(I_pred_lr_b*255, thre_pred)/np.log(255)

    L_F = 0

    #c_I = compute_c(I_pred_lr_b, I_lr_norm)#.detach();
    L_F += l2_loss(torch.sqrt((I_pred_lr_b ).clamp(min=0.0) + 1e-8) - torch.sqrt(I_lr_norm))
    #L_F += l1_loss(I_pred_lr_b * c_I- I_lr_norm)
    L_tot += kappa_frame * L_F
    L_tot += kappa_tv * tv_loss_flow(I_pred, tv_order=1, tv_tau=1e-4, iso=True,  Dxy =Dxy_sum_np/max_scl)
    #L_tot += kappa_tv * tv_loss(I_pred, tv_order=1, tv_tau=1e-4, iso=True)
    #L_tot += kappa_l0 * smooth_l0_weight(I_pred, w_xy=[kappa_l0 * w_x, kappa_l0 * w_y])
    #L_tot += kappa_tv * tv_loss(L_pred, tv_order=2, tv_tau=1e-3)
    #L_tot += kappa_tv_bg * tv_loss(bg_pred, tv_order=1,tv_tau=1e-4, iso=True)
    #L_tot += kappa_tv_bg * tv_loss_flow(bg_pred, tv_order=1, tv_tau=1e-4, Dxy=Dxy_sum_np / max_scl)
    L_tot += kappa_l1_bg * l1_loss(bg_pred, 1e-6)
    #L_tot += kappa_mid * mid_loss_flow(I_pred_b, I_srs[1].detach(), M, Dxy_pred, tc_iwes)
    L_tot.backward()  # Calculate the derivatives
    optimizer.step()
    scheduler.step()
    if alter == True:
        if iter > 400:
            bg_pred.requires_grad = True
    # if iter % 400 == 0:
    # bg_pred.requires_grad = not bg_pred.requires_grad
    with torch.no_grad():
        #kernel_pred = cho_style_psf_postprocess(kernel_pred)
        kernel_pred.clamp_(min=0, max=None)
        kernel_pred /= kernel_pred.sum()
        I_pred.clamp_(min=0, max=1)
        #thre_pred.clamp_(min=1+1e-9, max=20)
        #bg_pred.data[mask_static] = I_lr_norm[mask_static].float() -1/2
    if iter % 100 == 0:
        loss_hist.append(L_tot.detach().cpu())
        #print("iter = {}: loss = {}, d_phi={}".format(iter, L_tot.data.cpu().numpy(),  I_pred.grad.mean().cpu().numpy()))
        print("iter = {}: loss = {}".format(iter, L_tot.data.cpu().numpy()))
print("thre = ", thre_pred.data.cpu().numpy()); #iwe_pred_ver = compute_ver_EKLT(L_pred,  Dxy_pc.sum(0))
reset_optimizer_and_params(optimizer, vars)

frames_sharp =  frames_sta#frames_img[80, :, :]
path = r"E:\\[X] NeuroSR\\fig\\result1\\"
iwe_pred_ver = compute_ver_EKLT(L_pred,  Dxy_sum)  * border_mask
iwe_pred_ver_f = compute_ver_EKLT(L_lr,   Dxy_sum) * border_mask_lr
iwe_pred_f = forward_model_EKLT_L(L_lr,   Dxy_sum) * border_mask_lr
L_sta = lin_log(torch.from_numpy(frames_sharp).to(device), thre)
iwe_pred_f_sta = forward_model_EKLT_L(L_sta,   Dxy_sum)* border_mask_lr
iwe_pred_ver_f_sta= compute_ver_EKLT(L_sta,  Dxy_sum)* border_mask_lr
Dxy_roi=compute_dxy_roi(norm(computeLaplace(frames_sharp)), norm(np.abs(iwe_pred_lr)));
dirt = bg_pred.abs().detach().cpu().numpy(); dirt /=dirt.max()+ 1e-6;
roi_rec = roi_0
cmap = "gray"
cmap_np = "seismic";"gray";"seismic"; #win_frame = int(3e3)
cmap_d =  "inferno";"hot"; "magma";
#########################################################################################
plt.rcParams.update({'font.size': 15})
fig = plt.figure("compare frame", figsize=(12, 7))
plt.subplots_adjust(wspace=0.4, hspace=0.2)
ax = plt.subplot(341)
plot_ax_roi(ax,frames_sharp, cmap, axis_ind,M=1, Dxy_ori =np.ones_like(Dxy_roi) , roi = roi_rec,norm_ind = False)
add_roi(ax, roi_zoom, 1, Dxy_ori =Dxy_roi, color= 'white')
ax= plt.subplot(342)
plot_ax_roi(ax,frames_mov, cmap, axis_ind,M=1, roi = roi_rec,norm_ind = False)
add_roi(ax, roi_zoom, 1, color= 'white')
ax= plt.subplot(343)
plot_ax_roi(ax,I_pred.detach().cpu(), cmap, axis_ind, M=M, roi = roi_rec)
add_roi(ax, roi_zoom, M, color= 'white')
ax= plt.subplot(344)
plot_ax_roi(ax,iwe_gt.detach().cpu().numpy(), cmap_np, axis_ind, roi = roi_rec, M = M,norm_ind = True)
add_roi(ax, roi_zoom, M, color= 'white')

roi_rec = roi_zoom
ax = plt.subplot(345)
plot_ax_roi(ax,frames_sharp, cmap, axis_ind,M=1, Dxy_ori =Dxy_roi , roi = roi_rec,norm_ind = False)
ax= plt.subplot(346)
plot_ax_roi(ax,frames_mov, cmap, axis_ind,M=1, roi = roi_rec,norm_ind = False)
ax= plt.subplot(347)
plot_ax_roi(ax,I_pred.detach().cpu(), cmap, axis_ind, M=M, roi = roi_rec)
ax= plt.subplot(348)
plot_ax_roi(ax,iwe_gt.detach().cpu().numpy(), cmap_np, axis_ind, roi = roi_rec, M = M,norm_ind = True)
ax = plt.subplot(3,4,9)
plot_ax_roi(ax,iwe_pred_f_sta.detach().cpu().numpy(), cmap_np, axis_ind,M=1 , Dxy_ori =Dxy_roi, roi = roi_rec,norm_ind = True)
#plot_ax_roi(ax, bg_pred.abs().detach().cpu().numpy(), cmap_d, axis_ind, roi=roi_0, M=1, norm_ind = False)iwe_pred_ver_f_sta
ax= plt.subplot(3,4,10)
plot_ax_roi(ax,iwe_pred_f.detach().cpu().numpy(), cmap_np, axis_ind,M=1, roi =  roi_rec,norm_ind = True)
ax = plt.subplot(3,4,11)
plot_ax_roi(ax,iwe_pred.detach().cpu().numpy(), cmap_np, axis_ind,M=M, roi = roi_rec,norm_ind = True)
ax= plt.subplot(3,4,12)
plot_ax_roi(ax,bg_pred.abs().detach().cpu().numpy(), cmap_d, axis_ind,M=1  ,roi  = roi_0,norm_ind = False)
plot_zoom(ax, norm(kernel_pred.detach().cpu().detach()), cmap, False)
ax.axis('off')
plt.tight_layout()
#fig.savefig(r"E:\\[X] NeuroSR\\fig\\gaoda\\" +"_"+cmap_np+"sample.png", transparent=True)
# ax= plt.subplot(243)
# plot_ax_roi(ax, iwe_pred_ver_f.detach().cpu().numpy(), cmap_np, axis_ind, roi=roi_rec, M=1, norm_ind = True)
# ax = plt.subplot(244)
# plot_ax_roi(ax,iwe_pred_f.detach().cpu().numpy(), cmap_np, axis_ind, roi=roi_rec, M=1, norm_ind =True)
# ax = plt.subplot(345)
# plot_ax_roi(ax,frames_sharp, cmap, axis_ind,M=1, Dxy_ori =Dxy_roi , roi = roi_rec,norm_ind = False)
# ax.imshow(dirt, cmap=cmap_d, alpha=dirt)
# ax= plt.subplot(346)
# plot_ax_roi(ax,frames_mov, cmap, axis_ind,M=1, roi = roi_rec,norm_ind = False)
# ax.imshow(dirt, cmap=cmap_d, alpha=dirt)
# ax= plt.subplot(347)
# plot_ax_roi(ax, bg_pred.abs().detach().cpu().numpy(), cmap_d, axis_ind, roi=roi_rec, M=1, norm_ind = False)
# ax = plt.subplot(348)
# plot_ax_roi(ax,iwe_gt.detach().cpu().numpy(), cmap_np, axis_ind, roi = roi_rec, M = M,norm_ind = True)
# plot_zoom(ax, kernel_sr.detach().cpu().detach(), cmap, False)
# ax = plt.subplot(3 ,4 ,9)
# plot_ax_roi(ax,I_pred.detach().cpu(), cmap, axis_ind, M=M, roi = roi_rec)
# ax =plt.subplot(3,4, 10)
# plot_ax_roi(ax, I_pred_lr_b.detach().cpu(), cmap, axis_ind, M=1, roi = roi_rec)
# ax = plt.subplot(3,4, 11)
# plot_ax_roi(ax, iwe_pred_ver_2 .detach().cpu().numpy(), cmap_np, axis_ind, roi=roi_rec, M=M, norm_ind = True)
# ax = plt.subplot(3,4 ,12)
# plot_ax_roi(ax,iwe_pred.detach().cpu().numpy(), cmap_np, axis_ind, roi = roi_rec, M = M,norm_ind = True)

#fig.savefig(r"E:\\[X] NeuroSR\\fig\\semi\\" +"_"+cmap_np+"sample.png", transparent=True)

"""
### To save
save_png_npy(frames_mov, output_dir = sub_path)
save_png_npy(frames_sharp, output_dir = sub_path)
I_pred_norm =  norm(I_pred.detach().cpu().numpy())
save_png_npy(I_pred_norm, output_dir = sub_path)
save_png_npy(iwe_pred_sr, output_dir = sub_path)

save_png_npy(iwe_0, output_dir = sub_path)
save_png_npy(iwe_all, output_dir = sub_path)
cv2.imwrite(sub_path+"f_motion_blur.png", norm(frames_mot))


cv2.imwrite(sub_path+"iwe_0.png", norm(np.abs(iwe_0)))
cv2.imwrite(sub_path+"iwe_all.png", norm(np.abs(iwe_all)))
cv2.imwrite(save_path+"animal/sub/"+"3.png", norm(np.abs(iwe_pred_lr)))
cv2.imwrite(save_path+"animal/sub/"+"4.png", norm(np.abs(iwe_pred_sr)))
"""
"""
cv2.imwrite(sub_path  + "raw.png",crop_img(norm(I_lr.detach().cpu().numpy()), roi_zoom))
cv2.imwrite(sub_path  + "I_pred.png",crop_img(norm(I_pred.detach().cpu().numpy()), roi_zoom*M))
cv2.imwrite(sub_path  + "I_pred_sr.png",crop_img(norm(I_pred.detach().cpu().numpy()), roi_0*M))
cv2.imwrite(sub_path  + "raw_full.png",crop_img(norm(I_lr.detach().cpu().numpy()), roi_0))
cv2.imwrite(sub_path  + "frame_sharp.png",crop_img(norm(frames_sharp) ,roi_0*M))
"""

# path += "tl\\"
# tls = {}
# for i in len(tts):
#     tls[i] = NeuroSR(t0 = tts[i])
# path = r"E:\\[X] NeuroSR\\fig\\time_lapse\\"
# cv2.imwrite(path +"_I_pred"+str(M)+"_.png", norm(I_pred.detach().cpu().detach().cpu().numpy()))
"""
from utils.utils_viz import rotate_crop_image
img_name = "I_pred_2"
path = r"E:\\[X] NeuroSR\\fig\\result1\\"
img =  I_pred.detach().cpu().numpy() #norm(np.load(path + img_name + '.npy'));  
M =round(img.shape[0]/260)
cmap = "gray"
center = (243, 123)
crop = 50; 220; 
img = rotate_crop_image(img, -55,center =(center[0]*M, center[1]*M), crop_size= [crop*M,crop*M], flip=True)
plt.figure("zoom")
plt.imshow(img, cmap)
# os.mkdir( path + "sub1")
cv2.imwrite(path +"comparE 2\\" + img_name +"_"+  str(crop)+"_.png", norm(img))
"""


"""
from utils.utils_img_rec import ImageReconstructor
### apply with drunet
def reconstruct_pnp(iwe_np, Dxy_np, x_best= None, nf = 0.3,blur_ind = 10,iter = 200):
    iwe_torch = torch.from_numpy(iwe_np).to(device).unsqueeze(0).unsqueeze(0).to(device)
    flow_np = np.asarray([np.ones_like(iwe_np) * Dxy_np[0], np.ones_like(iwe_np) * Dxy_np[1]])
    flow_torch = torch.from_numpy(flow_np).unsqueeze(0).to(device)
    image_reconstructor = ImageReconstructor(flow_torch)
    img_rec_cnn = image_reconstructor.image_rec_from_iwe_cnn(iwe_torch, cnn_model_path="E:/pythonProject/NeuroDH/denoiser_model", weight1=nf,
                                                             weight2=blur_ind, x_best=x_best,
                                                             grad_des_iters=iter)
    return img_rec_cnn
nf= 0.6; blur_ind = 2.3;  
img_rec_cnn =  reconstruct_pnp(iwe_gt.detach().cpu().numpy(),Dxy_sum.detach().cpu().numpy(), x_best=L_pred.detach().cpu().numpy(), nf = nf,blur_ind = blur_ind,iter = 100)
#img_rec_cnn =  reconstruct_pnp(iwe_gt.detach().cpu().numpy(),Dxy_sum.detach().cpu().numpy(), x_best=None, nf = nf,blur_ind = blur_ind,iter = 100)
pnp_pred =np.exp(img_rec_cnn*np.log(255))/255
plt.figure()
ax = plt.subplot(121)
plot_ax_roi(ax,I_pred.detach().cpu().numpy(), cmap, axis_ind, roi = roi_0, M = M,norm_ind = False)
ax = plt.subplot(122)
plot_ax_roi(ax,pnp_pred, cmap, axis_ind, roi = roi_0, M = M,norm_ind = False)
save_png_npy(pnp_pred, arr_name ="pnp_pred",output_dir = sub_path)
"""




"""
#### plot more
#start = np.asarray([217,126]);end = np.asarray([239,143])   #group 5
start = np.asarray([237,98]);end = np.asarray([273,122])   #group 4
frames_grad = forward_model_EKLT_L(I_sr, Dxy_pc.sum(0)).detach().cpu().numpy()
fig =  plt.figure("compare frame 2", figsize=(12, 3))
plt.subplots_adjust(wspace=0.05, hspace=0.05)
ax = plt.subplot(141)
plot_ax_roi(ax, I_lr.detach().cpu().numpy(), cmap, axis_ind, roi = roi_zoom, Dxy_ori = Dxy_ori, norm_ind = False)
# cv2.line(frames_sr_grad,start*M,end*M,(255,0,0),3)
ax = plt.subplot(142)
# cv2.line(frames_sr_grad,start*M,end*M,(255,0,0),3)
I_sr_norm = I_sr
plot_ax_roi(ax, L_sr.detach().cpu().numpy(), cmap, axis_ind, roi = roi_zoom, M = M,Dxy_ori = Dxy_ori, norm_ind = False)
ax.plot()
ax = plt.subplot(143)
plot_ax_roi(ax,L_pred.detach().cpu().numpy(), cmap, axis_ind, roi = roi_zoom, M = M, norm_ind = False)
ax = plt.subplot(144)
plt.plot(slice_slanted_column(norm(L_sr.detach().cpu().numpy(),1), start*M, end*M)[0], c = "b")
plt.plot(slice_slanted_column(norm(L_pred.detach().cpu().numpy(),1), start*M, end*M)[0], c = "g")
plt.axis(axis_ind)
#ax.set_aspect(slice_slanted_column(I_pred.detach().cpu().numpy(), start*M, end*M)[0].size*ax_asp)
plt.tight_layout()
# fig.savefig(save_fig_path +"_"+"compare_rest_L_5.png", transparent=True)
"""



# # ax.set_aspect(loss_hist.__len__()/np.asarray(loss_hist).max()*ax_asp)

### To save the original figure is quite important


plt.figure("watch time lapse")
plt.subplot(131)
watch_tensor(NeuroSR(M = M, sigma =0.7,t0 = tts[0]))
plt.subplot(132)
watch_tensor(NeuroSR(M = M, sigma =0.7,t0 = tts[int(tts.shape[0]/2)]))
plt.subplot(133)
watch_tensor(NeuroSR(M = M, sigma =0.7,t0 = tts[-1]))


"""
num_t_ref = 25
dt = np.arange(win_frame);
tts = torch.linspace(tt[0], tt[-1],   num_t_ref)
sub_path = save_path+"tl\\"
iwes =[]; iwes_sr =[]
for i in range(num_t_ref):
    tl_iwe = NeuroSR(M = 1, sigma = sigma, t0 = tts[i], type = "numpy")
    tl_iwe_sr = NeuroSR(M = M, sigma = sigma, t0= tts[i], type="numpy")
    #tl_iwe_crop = crop_img(tl_iwe, roi_zoom *M)
    #cv2.imwrite(sub_path + str(i) + ".png",norm(tl_iwe))
    #iwes_zoom.append(tl_iwe_crop)
    iwes.append(tl_iwe)
    iwes_sr.append(tl_iwe_sr)
iwes = np.stack(iwes, axis=0)
iwes_sr = np.stack(iwes_sr, axis=0)
"""

"""

M = 1
from solver.flow_sr import forward_mfsr_model_v2
I_lrs = frames_img[frame_ind: frame_ind + 2,...] ### two frame supervison is enough for just continous video
I_lr_gts = torch.from_numpy(I_lrs).to(device)/255
t_start = time_img[frame_ind][0]    ## start time of the first frame,  in event stream, I use the first frame's timestamp to use such a video.
### Frame supervision should be evaluated at the center of each frame exposure.
tc_fs = time_img.mean(1)[frame_ind: frame_ind + 2]    # center time of each frame
tc_fs =[int(tc) for tc in tc_fs]
tc_fs -= t_start                                      # minus the frame start time


### Find iwe at the two frame center time
### how we warp the frame?
num_t_ref = 3; alpha = 0.5;
dt = np.arange(win_frame);
tc_iwes = torch.linspace(tc_fs[0], tc_fs[1],   num_t_ref).to(device)  # timestamps to reconstruct
tc_iwes = torch.tensor([int(tc) for tc in tc_iwes]).to(device)
iwe_gts = []; I_srs  = []; mask_e_s = []; t_ref0, t_ref1 = tc_iwes[0], tc_iwes[-1]; weights0=[];weights1=[]; kernels0 =[];kernels1 =[]
for i in range(tc_iwes.shape[0]):
    t_i = tc_iwes[i]
    iwe_tc = NeuroSR(M=M, sigma=sigma, t0=tc_iwes[i], type="tensor").detach()
    border_mask = torch.zeros(iwe_tc.shape).to(device).float();
    border_mask[1:-1, 1:-1] = 1;
    iwe_tc = iwe_tc * border_mask.detach()
    t0, t1 = tc_iwes[0], tc_iwes[-1]
    delta_t = abs(t1 - t0)
    w_to_t0 = 1 - torch.abs(t_i - t0) / delta_t
    w_to_t1 = 1 - torch.abs(t_i - t1) / delta_t
    w_to_t0 = torch.clamp(w_to_t0, 0.0, 1.0)
    w_to_t1 = torch.clamp(w_to_t1, 0.0, 1.0)
    kernels_i0 = generate_motion_blur_kernel_Dxy_pc( Dxy_pc.detach(), M, device,
    kernel_size=21, sigma=0.5,
    t_s=0, t_e=win_frame,      # physical time interval in microseconds
    t0=t_i,
    t_win=win,             # total event window in microseconds
    samples_per_seg=8)
    #generate_motion_blur_kernel_Dxy_sr(Dxy_pred[:, :win_frame+1], M, device, kernel_size=21, sigma=0.5,
    #                                                t0=t_i)
    weights0.append(w_to_t0); weights1.append(w_to_t1)
    kernels0.append(kernels_i0);
    mask_e = torch.zeros_like(border_mask).detach()
    loc = iwe_tc.abs() > torch.quantile(iwe_tc.abs()  , 0.1)
    mask_e[loc] = torch.tensor(1).to(device).float()
    iwe_gts.append(iwe_tc)
    mask_e_s.append(down_sampling(mask_e,M))
iwe_gts = torch.stack(iwe_gts, dim=0);
for i in range(I_lr_gts.shape[0]):
    I_sr = F.interpolate(I_lr_gts[i].detach().unsqueeze(0).unsqueeze(0), (int(Ny * M), int(Nx * M))).to(
    device).float().squeeze()
    I_srs.append(I_sr)

t_sup = [ tc_iwes[0], tc_iwes[-1]]; w_to_ts = []
# Optimizer setup
for i in range(len(t_sup)):
    w_to_ts.append(1 - torch.abs(tc_iwes - t_sup[i]) / (t_sup[1] - t_sup[0]))

#I_pred_srs = [nn.Parameter(torch.clone(I_srs[0]).detach(), requires_grad=True) for _ in tc_iwes]
I_pred_srs = [nn.Parameter(torch.ones_like(iwe_tc).detach()/2, requires_grad=True) for _ in tc_iwes]
I_pred_srs[0].data = I_pred_srs [0]; I_pred_srs[-1].data = I_pred_srs [-1]
#I_pred_srs = [nn.Parameter(torch.clone(I_srs[p]).detach(), requires_grad=True) for p in range(I_srs.shape[0])]
lr = [1e-2, 4e-4]; conv_type = "conv"; alter = False
Dxy_pc =  nn.Parameter(Dxy_pc.clone().to(device).float(), requires_grad=False)

bg_pred = nn.Parameter(torch.clone(torch.zeros_like(I_lr_norm).float().detach()), requires_grad = False);
vars = [{'params': p, 'lr': lr[0]} for p in I_pred_srs];vars += [{'params': bg_pred, 'lr': lr[1]}];
optimizer = AdamP(vars);scheduler = lr_scheduler.StepLR(optimizer, step_size=200//2, gamma=1);
#kappa_frame = 1; kappa_event =2*1e2; kappa_tv = 1e-1;
kappa_mid = 1;  loss_hist=[];
Dxy_pred = Dxy_pred.detach()



# === optimization loop ===
for iter in range(700):
    optimizer.zero_grad()
    L_F, L_E, L_TV = 0.0, 0.0, 0.0
    Flow_at_t = Dxy_pred[:, tc_iwes]  # (2, N)
    Flow_t0, Flow_t1 = Flow_at_t[:, :1], Flow_at_t[:, -1:]
    flow_rel0 = Flow_at_t - Flow_t0  # (2, N)
    flow_rel1 = Flow_at_t - Flow_t1

    for i, I_hr in enumerate(I_pred_srs):
        kernel_i = generate_motion_blur_kernel_Dxy_pc(Dxy_pc.detach(), M, device,
                                           kernel_size=27, sigma=0.5,
                                           t_s=0, t_e=win_frame,  # physical time interval in microseconds
                                           t0=tc_iwes[i],
                                           t_win=win,  # total event window in microseconds
                                           samples_per_seg=18).detach()
        # apply the per-timestamp blur kernel to the sharp frame (see optimizer.blur_frame)
        I_hr_b = blur_frame(I_hr, kernel_i, conv_type=conv_type, device=device)
        I_lr_b = down_sampling(I_hr_b, M) + bg_pred
        L_F += l2_loss(torch.sqrt(I_lr_b.clamp(min=0.0) + 1e-8) - torch.sqrt(I_lr_norm))
        # --- Event supervision ---
        if iter > 0:
            iwe_pred = forward_f2e(I_hr, Dxy_sum, thre_pred, False) * border_mask.detach()
            #iwe_gt = iwe_gts[i] / (torch.norm(iwe_gts[i], 2) + 1e-6)
            L_E += l2_loss(iwe_pred / torch.norm(iwe_pred, 2) - iwe_gt.detach() / torch.norm(iwe_gt.detach(), 2))


        # --- Regularization terms ---
        L_TV += (
            tv_loss_flow(I_hr, tv_order=1, tv_tau=1e-6, Dxy=Dxy_sum_np / max_scl)
            #+ tv_loss(I_hr, tv_order=1, tv_tau=1e-4, iso=True)
        )

    # --- Combine the total loss ---
    L_tot = (
        kappa_frame * L_F +
        kappa_event * L_E +
        kappa_tv * L_TV +
        kappa_tv_bg * tv_loss(bg_pred, tv_order=1, tv_tau=1e-3, iso=True) + kappa_l1_bg * l1_loss(bg_pred, 1e-4)
        # +kappa_l1_bg * (smooth_l0(bg_pred) + l1_loss(bg_pred, 1/255))
    )
    if alter == True:
        if iter % 400 == 0:
            bg_pred.requires_grad = not bg_pred.requires_grad

    # --- Backpropagation ---
    L_tot.backward()
    optimizer.step()
    scheduler.step()

    # --- Post-processing ---
    with torch.no_grad():
        for I in I_pred_srs:
            I.clamp_(0, 1)
        for kernel in kernels0:
            # kernel = cho_style_psf_postprocess(kernel)  #?
            kernel.clamp_(min=0, max=None)
            kernel /= kernel.sum()
    if iter % 100 == 0:
        print(f"[Iter {iter:03d}]  L_tot = {L_tot.item():.6f}")


reset_optimizer_and_params(optimizer, vars)
plt.rcParams['font.size'] = 15
roi_rec = roi_0
plt.figure("compare tl", figsize=[10, 5])
ax = plt.subplot(231)
plot_ax_roi(ax,I_pred_srs[0].detach().cpu().numpy(), cmap, axis_ind, roi = roi_rec, M =  M,norm_ind = False)
plot_zoom(ax, norm(kernels0[0].detach().cpu().detach()), cmap, False)
ax = plt.subplot(232)
plot_ax_roi(ax,I_pred_srs[1].detach().cpu().numpy(), cmap, axis_ind, roi = roi_rec, M =  M,norm_ind = False)
plot_zoom(ax, norm(kernels0[1].detach().cpu().detach()), cmap, False)
ax =  plt.subplot(233)
plot_ax_roi(ax,I_pred_srs[2].detach().cpu().numpy(), cmap, axis_ind, roi = roi_rec, M =  M,norm_ind = False)
plot_zoom(ax, norm(kernels0[2].detach().cpu().detach()), cmap, False)
ax = plt.subplot(234)
plot_ax_roi(ax,I_pred_srs[3].detach().cpu().numpy(), cmap, axis_ind, roi = roi_rec, M =  M,norm_ind = False)
plot_zoom(ax, norm(kernels0[3].detach().cpu().detach()), cmap, False)
ax = plt.subplot(235)
plot_ax_roi(ax,I_pred_srs[4].detach().cpu().numpy(), cmap, axis_ind, roi = roi_rec, M =  M,norm_ind = False)
plot_zoom(ax, norm(kernels0[4].detach().cpu().detach()), cmap, False)
ax = plt.subplot(236)
plot_ax_roi(ax,frames_mov, cmap, axis_ind, roi = roi_rec, M = 1,norm_ind = False)
plt.tight_layout()

print  ("frame_ind in the interval is:" ,  np.where((time_img[:, 0] - t_start > ts[0]) & (time_img[:, 0] - t_start < ts[-1]))[0])

"""
