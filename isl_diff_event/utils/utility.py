from dv import AedatFile
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.stats import multivariate_normal
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import LightPipes as LP
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import scipy
import os
import zipfile
import h5py
import torchvision.transforms as T
from scipy.signal import savgol_filter
from typing import Tuple, Union
NUMPY_TORCH = Union[np.ndarray, torch.Tensor]
FLOAT_TORCH = Union[float, torch.Tensor]
# import pylops
#from utils.utils_img_rec import computeMatrixA




def flow_to_color_wheel(flow_xy: np.ndarray, max_mag: float = None) -> np.ndarray:
    """Convert [2, H, W] flow to HSV color wheel visualization (RGB uint8)."""
    u, v = flow_xy[0], flow_xy[1]
    mag = np.sqrt(u**2 + v**2)
    if max_mag is None:
        max_mag = mag.max() if mag.max() > 1e-6 else 1.0
    ang = np.arctan2(v, u)
    hsv = np.zeros((*u.shape, 3), dtype=np.uint8)
    hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / max_mag * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def norm(x,scl = 255):
    if x.max()-x.min() ==0:
        out = np.ones_like(x)
    else:
        out = (x - x.min()) /( x.max()-x.min()) * scl
    return out
def norm_torch(x,scl = 1):
    if x.max()-x.min() ==0:
        out = torch.ones_like(x)
    else:
        out = (x - x.min()) /( x.max()-x.min()) * scl
    return out

def minmax_norm(x,q = 99):
    den = np.percentile(x, q) - np.percentile(x, 100-q)
    if den != 0:
        x = (x - np.percentile(x, 100-q)) / den
    return np.clip(x, 0, 1)
def minmax_norm_torch(x, q=0.99):
    den = torch.quantile(x, q) - torch.quantile(x, 1 - q)
    if den != 0:
        x = (x - torch.quantile(x, 1-q)) / den
    return torch.clamp(x, 0, 1)
# def minmax_norm_torch(x, q=99.9):
#     # Sort the tensor along each column
#     x_sorted = torch.sort(x, dim=0).values
#
#     # Compute the indices for the percentiles
#     lower_idx = int((100 - q) / 100.0 * (x.size(0) - 1))
#     upper_idx = int(q / 100.0 * (x.size(0) - 1))
#
#     # Extract the percentile values
#     lower = x_sorted[lower_idx]
#     upper = x_sorted[upper_idx]
#
#     # Calculate the denominator
#     den = upper - lower
#     if torch.any(den != 0):
#         # Normalize
#         x = (x - lower) / den
#
#     # Clip the values between 0 and 1
#     return torch.clamp(x, 0, 1)
def circ_mask(L,R=None, val = 0.0):
    x = np.linspace(1, L, L)
    mask = np.ones([L, L])
    Y, X = np.meshgrid(x, x)
    Y = Y - int(L/2) - 1
    X = X - int(L/2) - 1
    dist_sq = X ** 2 + Y ** 2  # squared, no need for sqrt
    if R is None:
        R = int(L/2) - 1
    mask[dist_sq > (R) ** 2] = val
    return mask
def color_event_to_gray(xs, ys, ps, sensor_size):
    """
    Convert color events to grayscale using RGGB CFA and standard luminance weights.
    Assumes input polarity ps ∈ {0,1}, will convert to {-1,1} internally.
    """
    H, W = sensor_size
    # Generate RGGB CFA masks
    mask_R = np.zeros((H, W), dtype=bool)
    mask_G = np.zeros((H, W), dtype=bool)
    mask_B = np.zeros((H, W), dtype=bool)

    for y in range(H):
        for x in range(W):
            if y % 2 == 0:
                if x % 2 == 0:
                    mask_G[y, x] = True
                else:
                    mask_R[y, x] = True
            else:
                if x % 2 == 0:
                    mask_B[y, x] = True
                else:
                    mask_G[y, x] = True

    # Luminance weights
    w_R, w_G, w_B = 0.299, 0.587, 0.114

    # Convert polarity to signed values
    ps_signed = 2 * ps - 1

    # Apply CFA-aware luminance weighting
    gray_ps = np.zeros_like(ps_signed, dtype=np.float32)
    for i in range(len(ps_signed)):
        x, y = xs[i], ys[i]
        if mask_R[y, x]:
            gray_ps[i] = w_R * ps_signed[i]
        elif mask_G[y, x]:
            gray_ps[i] = w_G * ps_signed[i]
        elif mask_B[y, x]:
            gray_ps[i] = w_B * ps_signed[i]
    return gray_ps


def load_events(eve_dtype= "aedat4", path="E:/cosi/mla final/Sphe0_2/s7.aedat4"):
    if eve_dtype == "h5":
        with h5py.File(path, 'r') as hf:   # path_h5 = 'E:/davis/33 MLA/v2e.h5'
            events = hf['events'][:]
        events = events.astype("int64")
        # load events, and determine device and the windows
        ts, xs, ys, ps = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
        t_ref = ts.min()
        ts = ts - t_ref
        return ts, xs, ys, ps
    if eve_dtype == "npy":
        events = np.load(path)
        # load events, and determine device and the windows
        ts, xs, ys, ps = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
        return ts, xs, ys, ps

    elif eve_dtype == "aedat4":
        with AedatFile(path) as f:
            # events will be a named numpy array
            events = np.hstack([packet for packet in f['events'].numpy()])
            # frames_img = np.stack([packet.image for packet in f['frames']], axis=3)
            frames = np.stack([packet for packet in f['frames']])
        ts, xs, ys, ps = events['timestamp'], events['x'], events['y'], events['polarity']
        t_ref = ts.min()
        ts = ts - t_ref
        frames_img = []
        time_img = []
        for i in range(frames.size):
            frames_img.append(frames[i].image)
            time_img.append([frames[i].timestamp_start_of_exposure, frames[i].timestamp_end_of_exposure])
        frames_img = np.asarray(frames_img).squeeze()
        time_img = np.asarray(time_img).squeeze()
        time_img = time_img - t_ref
        return ts, xs, ys, ps, frames_img, time_img


def extract_and_zip_aedat(aedat_file_path, output_dir, sensor_size = (260, 346)):
    """
    Extracts events from an .aedat4 file within a given timestamp range,
    records the maximum width and height, saves them to a .txt file with a header line for dimensions,
    then compresses it into a .zip file in the specified output directory.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_filename = os.path.basename(aedat_file_path).replace('.aedat4', '')
    txt_filename = f"{base_filename}.txt"
    zip_filename = f"{base_filename}.zip"

    # Path setup
    txt_file_path = os.path.join(output_dir, txt_filename)
    zip_file_path = os.path.join(output_dir, zip_filename)

    # Temporary storage for event data
    event_data = []

    # Extract events and determine max width and height
    with AedatFile(aedat_file_path) as f:
        for packet in f['events'].numpy():
            for event in packet:
                    timestamp_seconds = event['timestamp'] / 1_000_000.0  # convert microseconds to seconds
                    event_data.append(f"{timestamp_seconds:.12f} {event['x']} {event['y']} {event['polarity']}")



    # Write to a txt file including width and height at the first line
    with open(txt_file_path, 'w') as txt_file:
        txt_file.write(f"{sensor_size[1]} {sensor_size[0]}\n")
        for line in event_data:
            txt_file.write(line + "\n")

    # Compress the txt file into a zip file
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(txt_file_path, arcname=txt_filename)

    return zip_file_path, txt_file_path

def package_zip_events(events ,aedat_file_path, output_dir, sensor_size = (260, 346)):
    """
    Extracts events from an .aedat4 file within a given timestamp range,
    records the maximum width and height, saves them to a .txt file with a header line for dimensions,
    then compresses it into a .zip file in the specified output directory.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_filename = os.path.basename(aedat_file_path).replace('.aedat4', '')
    txt_filename = f"{base_filename}.txt"
    zip_filename = f"{base_filename}.zip"

    # Path setup
    txt_file_path = os.path.join(output_dir, txt_filename)
    zip_file_path = os.path.join(output_dir, zip_filename)

    # Temporary storage for event data
    event_data = []

    # Extract events and determine max width and height


    for event in events:
        timestamp_seconds = event[0] / 1_000_000.0  # convert microseconds to seconds
        event_data.append(f"{timestamp_seconds:.12f} {event[1]} {event[2]} {event[3]}")


    # Write to a txt file including width and height at the first line
    with open(txt_file_path, 'w') as txt_file:
        txt_file.write(f"{sensor_size[1]} {sensor_size[0]}\n")
        for line in event_data:
            txt_file.write(line + "\n")

    # Compress the txt file into a zip file
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(txt_file_path, arcname=txt_filename)

    return zip_file_path, txt_file_path

def  time_window(xs, ys,ts, ps ,s , win ):
    loc = np.where((ts <= s + win) & (ts >= s))
    ts, xs, ys, ps = ts[loc], xs[loc], ys[loc], ps[loc]
    return xs, ys, ts, ps

def polar_event(xs, ys, ts, ps,ind):
    loc = np.where(ps ==ind )
    xs, ys, ts,ps = xs[loc], ys[loc], ts[loc],ps[loc]
    #pos_xs, pos_ys, pos_ts = xs[loc], ys[loc], ts[loc]
    return xs, ys, ts,ps

def preprocess_img(ref_image, noise_threshold = 0.005):  # threshold
    ref_image = cv2.threshold(ref_image, noise_threshold * np.max(ref_image), 65536, cv2.THRESH_TOZERO)[1]
    # rotate image
    rows, cols = ref_image.shape  # read the rows and cols
    return ref_image

def check_centroid(cxy,img_ref,pitch_px):
    cxy_centroid = np.zeros([2, cxy.size//2])
    img_grid = img_ref.copy()
    for i in range(cxy.size//2):
        cx, cy = cxy[:, i]
        xc, yc = openCV_centroid(img_ref, cx, cy, pitch_px)
        cxy_centroid[0, i] = xc
        cxy_centroid[1, i] = yc
        cv2.rectangle(img_grid, (int(cx - pitch_px / 2), int(cy - pitch_px / 2)),
                      (int(cx + pitch_px / 2), int(cy + pitch_px / 2)), (255, 0, 0), 1)
    return cxy_centroid,img_grid


def centroid_img(img):
    moment = cv2.moments(img)
    Xc =  (moment['m10'] / moment['m00'])
    Yc = (moment['m01'] / moment['m00'])
    return Xc, Yc


def plot_scatter_event(xs, ys, ts,ps, view = "vertical",polar = True, axis_ind  = "on",alpha = 0.2,strech = 0.5):

    fig = plt.figure("Spatio-temporal stream of SHWFS", figsize=(13, 13))
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect([1.0, 1.0, 1.0])
    if polar == True:
        pos_loc = np.where(ps == 1)
        neg_loc = np.where(ps == 0)
        pos_xs, pos_ys, pos_ts = xs[pos_loc], ys[pos_loc], ts[pos_loc]
        neg_xs, neg_ys, neg_ts = xs[neg_loc], ys[neg_loc], ts[neg_loc]
        ax.scatter(alpha=0.5, xs=pos_xs, zs=pos_ts/1e6, ys=pos_ys, c='r', marker=".", clip_on=True)
        ax.scatter(alpha=0.5, xs=neg_xs, zs=neg_ts/1e6, ys=neg_ys, c='b', marker=".", clip_on=True)
    else:
        ax.scatter(xs=xs, alpha=alpha, ys=ys, zs=ts/1e6, marker=".", c='k')

    ax.set_xlabel('X')
    ax.set_zlabel('t /s')
    ax.set_ylabel('Y')
    ax.set_zlim(ts.min()/1e6, (ts.max())/1e6 +  strech )
    if view =="vertical":
        ax.view_init(azim=0, elev=90)
    elif view =="side":
        ax.view_init(azim=135, elev=0)    # 侧视图
    elif view =="normal":
        ax.view_init(azim=45, elev=30)  # 倾斜normal视图
    elif view =="lateral":
        ax.view_init(azim=90, elev=90)  # 倾斜normal视图

    plt.axis(axis_ind)
    fig.savefig("E:/optica/3dscatter.png", transparent=True)

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


def plot_scatter_event_lateral(xs, ys, ts,ps, view = "vertical",polar = True, axis_ind  = "on",strech = 0, alpha = 0.2,alpha_k=0.02, plot = False, save = True):
    pos_loc = np.where(ps == 1)
    neg_loc = np.where(ps == 0)
    pos_xs, pos_ys, pos_ts = xs[pos_loc], ys[pos_loc], ts[pos_loc]
    neg_xs, neg_ys, neg_ts = xs[neg_loc], ys[neg_loc], ts[neg_loc]

    fig = plt.figure("Spatio-temporal stream of SHWFS side", figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')

    if polar == True:
        ax.scatter(alpha=alpha, xs=pos_xs, ys=pos_ts/1e6, zs=pos_ys, c='r', marker=".", clip_on=True)
        ax.scatter(alpha=alpha, xs=neg_xs, ys=neg_ts/1e6, zs=neg_ys, c='b', marker=".", clip_on=True)
    #ax.scatter(xs=xs, alpha=alpha_k, ys=0, zs= ys, marker=".", c='k')
    ax.set_xlabel('X')
    ax.set_zlabel('Y')
    ax.set_ylabel('t/s')
    ax.set_xlim3d(0, 345)
    ax.set_zlim3d(0, 259)
    ax.set_ylim3d(ts.min()/1e6, (ts.max())/1e6 +  strech )
    ax.set_box_aspect([1.0, 2, 1.0])
    #plt.gca.set_box_aspect((1,1/1000,1))
    if view =="vertical":
        ax.view_init(azim=0, elev=90)
    elif view =="side":
        ax.view_init(azim=30, elev=18)    # 侧视图
    elif view =="normal":
        ax.view_init(azim=64, elev=18)  # 倾斜normal视图
    elif view =="normal45":
        ax.view_init(azim=45, elev=0)  # 倾斜normal视图
    elif view =="lateral":
        ax.view_init(azim=90, elev=90)  # 倾斜normal视图
    elif view == "reverse":
        ax.view_init(azim=-60, elev=18)  # reverse 视图

    ax.patch.set_alpha(0.5)
    plt.axis(axis_ind)
    if plot == False:
        plt.close(fig)
    if save == True:
        fig.savefig("E:/optica/3dscatter1.png", transparent=True)





def plot_scatter_dynamic(xs, ys, ts,ps, axis_ind  = "on",alpha = 0.2, eve_win = 1e4, i_win = 1e3, f_range =  np.arange(0, 20),_fps =30):
    fig = plt.figure("Spatio-temporal stream of SHWFS side", figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.set_title("NeuroSH dynamic visualization")
    ax.set_box_aspect([1.0, 2, 1.0])
    ax.view_init(azim=-18, elev=18)  # reverse 视图

    def update(i):
        plt.cla()
        e = (i+1) * i_win
        if e <= eve_win:
            loc = np.where(ts <= e)
            tsi, xsi, ysi, psi = ts[loc], xs[loc], ys[loc], ps[loc]
        else:
            xsi, ysi, tsi, psi = time_window(xs, ys, ts, ps, e - eve_win, win=eve_win)

        xsii, ysii, tsii, psii = time_window(xs, ys, ts, ps, i * i_win , win=i_win)


        pos_loc = np.where(psi == 1)
        neg_loc = np.where(psi == 0)
        pos_xs, pos_ys, pos_ts = xsi[pos_loc], ysi[pos_loc], tsi[pos_loc]
        neg_xs, neg_ys, neg_ts = xsi[neg_loc], ysi[neg_loc], tsi[neg_loc]
        ax.scatter(alpha=alpha, xs=pos_xs, ys=(e-pos_ts) / 1e6, zs=pos_ys, c='r', marker=".", clip_on=True)
        ax.scatter(alpha=alpha, xs=neg_xs, ys=(e-neg_ts) / 1e6, zs=neg_ys, c='b', marker=".", clip_on=True)
        ax.scatter(xs=xsii, alpha=1, ys=0, zs=ysii, marker=".", c='k')
        plt.axis(axis_ind)
        ax.set_xlabel('X')
        ax.set_zlabel('Y')
        ax.set_ylabel('t/s')
        ax.set_xlim3d(0, 259)
        ax.set_zlim3d(0, 259)
        ax.set_ylim3d(0, eve_win / 1e6)

    ani = animation.FuncAnimation(fig, update,f_range, interval=100, blit=False)
    ani.save('E:/optica/scatter_t.gif', fps=_fps)









def plot_scatter_bos_rect(xs, ys, ts, ps, view = "vertical",sensor_size = (260 , 346),polar = True, axis_ind  = "on", alpha = 0.2,alpha_k=0.02, plot = False, save = None, frame = None):
    pos_loc = (ps == 1) #& (ts > ts.max()//2)
    neg_loc = (ps == 0) #& (ts > ts.max()//2)
    pos_xs, pos_ys, pos_ts = xs[pos_loc], ys[pos_loc], ts[pos_loc]
    neg_xs, neg_ys, neg_ts = xs[neg_loc], ys[neg_loc], ts[neg_loc]
    fig = plt.figure("Spatio-temporal stream of SHWFS side", figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    if polar == True:
        ax.scatter(alpha=alpha, xs=pos_xs, ys=pos_ts/1e6, zs=pos_ys, c='r', marker=".", clip_on=True)
        ax.scatter(alpha=alpha, xs=neg_xs, ys=neg_ts/1e6, zs=neg_ys, c='b', marker=".", clip_on=True)
    ax.scatter(xs=xs, alpha=alpha_k, ys=0, zs= ys, marker=".", c='k')  #ys=(ts.max())/1e6/2

    if frame is not None:
        img = frame/frame.max()  # 确保归一化为 [0,1]
        #img[img<1e-1] = np.nan
        H, W = sensor_size
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        T_val = (ts.max() + ts.min()) / 2 / 1e6  # 时间轴中点 (单位: 秒)
        T = np.full_like(img, T_val)

        ax.plot_surface(
            X, T, Y,  # 注意：时间轴是 Y 轴
            facecolors=plt.cm.gray(img),
            rstride=1, cstride=1,
            shade=False,
            alpha=0
        )

    ax.set_xlim3d(sensor_size[1]-1, 0)
    ax.set_zlim3d(sensor_size[0]-1, 0 )
    ax.set_ylim3d(ts.min()/1e6, (ts.max())/1e6 )
    ax.set_box_aspect([sensor_size[1]/sensor_size[0], 2, 1.0])
    #plt.gca.set_box_aspect((1,1/1000,1))
    if view =="vertical":
        ax.view_init(azim=0, elev=90)
    # elif view =="side":
    #     ax.view_init(azim=90, elev=135)    # 侧视图
    elif view == "r-side":
        ax.view_init(azim=-65, elev=23)  # reverse 视图
    elif view == "side":
        ax.view_init(azim=53, elev=37)    # 侧视图
    elif view =="normal":
        ax.view_init(azim=40, elev=18)  # 倾斜normal视图
    elif view =="normal45":
        ax.view_init(azim=66, elev=30)  # 倾斜normal视图
    elif view =="lateral":
        ax.view_init(azim=90, elev=90)  # 倾斜normal视图
    elif view == "reverse":
        ax.view_init(azim=-60, elev=18)  # reverse 视图
    plt.axis(axis_ind)
    #ax.grid(False)
    # 移除3D坐标面背景色和边框
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    if plot == False:
        plt.close(fig)
    if save is not None:
        fig.savefig(save, transparent=True)

def plot_scatter_evk2D(xs, ys, ps, sensor_size, polar= True, alpha =0.5):
    fig= plt.figure("evk5 view")
    ax = fig.add_subplot(111, facecolor='black')
    pos_loc = np.where(ps == 1)
    neg_loc = np.where(ps == 0)
    pos_xs, pos_ys = xs[pos_loc], ys[pos_loc]
    neg_xs, neg_ys = xs[neg_loc], ys[neg_loc]
    if polar == True:
        ax.scatter(alpha=alpha, x=pos_xs, y=pos_ys, c='w',marker=".", clip_on=True)
        ax.scatter(alpha=alpha, x=neg_xs, y=neg_ys, c='royalblue',marker=".", clip_on=True)
    ax.set_xlim(0,sensor_size[1]-1 )
    ax.set_ylim(sensor_size[0]-1, 0)
    ax.set_box_aspect(260/346)
    ax.set_xticks([]);ax.set_yticks([])



def plot_scatter_bos_rect_roi(xs, ys, ts, ps, view = "vertical",roi = [[0,346],[0,260]], polar = True, axis_ind  = "on", alpha = 0.2,alpha_k=0.02, plot = False, save = None):
    sx,ex = np.min(roi[0,:]), np.max(roi[0,:])
    sy,ey = np.min(roi[1,:]), np.max(roi[1,:])
    pos_loc = (ps == 1) & (xs>sx) & (xs< ex) & (ys>sy) & (ys< ey)
    neg_loc = (ps == 0) & (xs>sx) & (xs< ex) & (ys>sy) & (ys< ey)
    pos_xs, pos_ys, pos_ts = xs[pos_loc], ys[pos_loc], ts[pos_loc]
    neg_xs, neg_ys, neg_ts = xs[neg_loc], ys[neg_loc], ts[neg_loc]
    fig = plt.figure("Spatio-temporal stream of SHWFS side", figsize=(5, 5))
    ax = fig.add_subplot(projection='3d')
    if polar == True:
        ax.scatter(alpha=alpha, xs=pos_xs, ys=pos_ts/1e6, zs=pos_ys, c='r', marker=".", clip_on=True)
        ax.scatter(alpha=alpha, xs=neg_xs, ys=neg_ts/1e6, zs=neg_ys, c='b', marker=".", clip_on=True)
    #ax.scatter(xs=xs, alpha=alpha_k, ys=ts.max()/1e6, zs= ys, marker=".", c='k')  #ys=(ts.max())/1e6/2
    ax.set_xlim3d(roi[0,1],roi[0,0])
    ax.set_zlim3d(roi[1,0],roi[1,1])
    ax.set_ylim3d(ts.min()/1e6, (ts.max())/1e6 )
    ax.set_box_aspect([346/260, 2, 1.0])
    ax.view_init(azim=90, elev=0)
    #plt.gca.set_box_aspect((1,1/1000,1))
    if view =="vertical":
        ax.view_init(azim=90, elev=0)
    elif view == "r-side":
        ax.view_init(azim=-26, elev=18)  # reverse 视图
    elif view == "side":
        ax.view_init(azim=53, elev=37)    # 侧视图
    elif view =="normal":
        ax.view_init(azim=40, elev=18)  # 倾斜normal视图
    elif view =="normal45":
        ax.view_init(azim=66, elev=30)  # 倾斜normal视图
    elif view =="lateral":
        ax.view_init(azim=0, elev=90)  # 倾斜normal视图
    elif view == "reverse":
        ax.view_init(azim=-60, elev=18)  # reverse 视图
    plt.axis(axis_ind)
    #ax.grid(False)
    # 移除3D坐标面背景色和边框
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    if plot == False:
        plt.close(fig)
    if save is not None:
        fig.savefig(save, transparent=True)

def corr2D(A,B):
    m = A.shape[0]
    n = A.shape[1]
    corr=abs(np.fft.fftshift(np.fft.ifft2(np.fft.fft2(A)*np.conj(np.fft.fft2(B)))))/(n*m)
    return corr

def corr2D_torch(A,B):
    m = A.shape[0]
    n = A.shape[1]
    corr=torch.abs(torch.fft.fftshift(torch.fft.ifft2(torch.fft.fft2(A)*torch.conj(torch.fft.fft2(B)))))/(n*m)
    return torch.sum(corr/m/n)
def  forward_grad_np(x):
    x_d = np.roll(x, -1, axis=1) - x
    y_d = np.roll(x, -1, axis=0) - x
    return y_d, x_d

def compute_dxy_roi(A, B, sensor_size=(260, 346)):
    corr = corr2D(A, B)
    scy, scx = np.asarray(sensor_size) / 2
    #corr[int(scy), int(scx)] = 0
    corr_max = np.where(corr == corr.max())
    Dxy_ori = np.asarray([[corr_max[1][0]-scx],[corr_max[0][0] -scy ]])
    return  Dxy_ori




# def update(i):
#     plt.cla()
#     start = eve_start + i * eve_win
#     end = int(start + 1 * eve_win)
#     x_win = x[start:end]  # x max is 346
#     y_win = y[start:end]  # y max is 260
#     t_win = timestamps[start:end]
#     p_win = np.where(polarities[start:start + eve_win] == 1)
#     n_win = np.where(polarities[start:start + eve_win] == 0)
# def update(i):
#     ax.scatter(alpha=alpha, xs=pos_xs, ys=pos_ts/1e6, zs=pos_ys, c='r', marker=".", clip_on=True)
#     ax.scatter(alpha=alpha, xs=neg_xs, ys=neg_ts/1e6, zs=neg_ys, c='b', marker=".", clip_on=True)
#     ax.scatter(xs=xs, alpha=0.1, ys=0, zs= ys, marker=".", c='k')



from mpl_toolkits.mplot3d import Axes3D
def plot_surface_event(iwe):
    iwe = norm(iwe,1)
    sr = iwe.shape
    xx, yy = np.meshgrid(np.linspace(0,1,sr[1]), np.linspace(1,0,sr[0]))
    fig = plt.figure()
    # ax1 = fig.add_subplot(121)
    # ax1.imshow(iwe, cmap=plt.cm.BrBG, interpolation='nearest',extent=[0,6.54,0,4.91])
    ax2 = fig.add_subplot(111, projection='3d')
    ax2.plot_surface(xx, yy, iwe, rstride=10, cstride=10, facecolors=plt.cm.BrBG(iwe), shade=False)





def np_to_torch(xs, ys, ts, ps,device):
    xt, yt, tt, pt = torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(ts), torch.from_numpy(ps)
    xt = xt.float().to(device)
    yt = yt.float().to(device)
    tt0 = tt[0]
    tt = (tt[:] - tt0).float().to(device)
    pt = pt.float().to(device)
    return xt, yt, tt, pt


def interpolate_to_image(pxs, pys, dxs, dys, interp_weights, img):
    """
    Accumulate x and y coords to an image using bilinear interpolation
    """
    img_h, img_w = img.shape  # Get image dimensions
    valid_mask = (pxs >= 0) & (pxs < img_w-1) & (pys >= 0) & (pys < img_h-1)
    pxs = pxs[valid_mask]
    pys = pys[valid_mask]
    dxs = dxs[valid_mask]
    dys = dys[valid_mask]
    interp_weights = interp_weights[valid_mask]

    img.index_put_((pys, pxs), interp_weights * (1.0 - dxs) * (1.0 - dys), accumulate=True)
    img.index_put_((pys, pxs + 1), interp_weights * dxs * (1.0 - dys), accumulate=True)
    img.index_put_((pys + 1, pxs), interp_weights * (1.0 - dxs) * dys, accumulate=True)
    img.index_put_((pys + 1, pxs + 1), interp_weights * dxs * dys, accumulate=True)

    return img
#
# def gaussian_interpolate_to_image(pxs, pys, dxs, dys, interp_weights, img, sigma=1):
#     """
#     Perform Gaussian interpolation with a 3x3 kernel.
#     """
#     device = pxs.device
#     offsets = torch.arange(-1, 2, device=device)  # Offsets: [-1, 0, 1]
#
#     # Compute Gaussian weights directly for 3x3
#     weights_x = torch.exp(-0.5 * ((dxs.unsqueeze(-1) - offsets) ** 2) / (sigma ** 2))  # Shape: [N, 3]
#     weights_y = torch.exp(-0.5 * ((dys.unsqueeze(-1) - offsets) ** 2) / (sigma ** 2))  # Shape: [N, 3]
#     weights = weights_x.unsqueeze(2) * weights_y.unsqueeze(1)  # Shape: [N, 3, 3]
#
#     # Compute affected pixel coordinates
#     pxs_b = pxs.unsqueeze(-1) + offsets  # Shape: [N, 3]
#     pys_b = pys.unsqueeze(-1) + offsets  # Shape: [N, 3]
#
#     # Expand and flatten coordinates
#     pxs_b = pxs_b.unsqueeze(2).expand(-1, -1, 3).reshape(-1)  # Flatten to [N*9]
#     pys_b = pys_b.unsqueeze(1).expand(-1, 3, -1).reshape(-1)  # Flatten to [N*9]
#     weights = weights.reshape(-1)  # Flatten to [N*9]
#
#     # Apply contributions to the image
#     img.index_put_(
#         (pys_b, pxs_b),
#         interp_weights.repeat_interleave(9) * weights,
#         accumulate=True
#     )
#     return img


def compute_gaussian_weights(offsets, centers, sigma):
    """
    Compute Gaussian weights for given offsets and centers.
    Args:
        offsets: Tensor of relative offsets (e.g., [-1, 0, 1]).
        centers: Tensor of event positions (e.g., dx or dy).
        sigma: Standard deviation of the Gaussian distribution.
    Returns:
        Gaussian weights for each offset-center pair.
    """
    return torch.exp(-0.5 * ((centers.unsqueeze(-1) - offsets) ** 2) / (sigma ** 2))
#
# def gaussian_interpolate_to_image(pxs, pys, dxs, dys, interp_weights, img, sigma=1.0, kernel_size=3):
#     """
#     Perform Gaussian interpolation with a normalized kernel of arbitrary size.
#     """
#     device = pxs.device
#     img_h, img_w = img.shape  # Get image dimensions
#
#     radius = (kernel_size - 1) // 2
#     offsets = torch.arange(-radius, radius + 1, device=device)  # [-r ... r]
#
#     # Compute 1D base Gaussian
#     gauss_1d = torch.exp(-0.5 * (offsets ** 2) / (sigma ** 2))
#
#     # Compute shifted Gaussian weights (separable)
#     weights_x = torch.exp(-0.5 * ((dxs.unsqueeze(-1) - offsets) ** 2) / (sigma ** 2))  # [N,k]
#     weights_y = torch.exp(-0.5 * ((dys.unsqueeze(-1) - offsets) ** 2) / (sigma ** 2))  # [N,k]
#
#     # Outer product: [N, k, k]
#     weights = weights_x.unsqueeze(2) * weights_y.unsqueeze(1)
#
#     # ★ New: per-sample normalization (small modification 1)
#     norm_x = weights_x.sum(dim=1)  # [N]
#     norm_y = weights_y.sum(dim=1)  # [N]
#     normalization_factor = (norm_x * norm_y).unsqueeze(1).unsqueeze(2)  # [N,1,1]
#
#     # Normalize weights
#     weights = weights / (normalization_factor + 1e-12)
#     # Compute affected pixel coordinates
#     pxs_b = pxs.unsqueeze(-1) + offsets  # Shape: [N, kernel_size]
#     pys_b = pys.unsqueeze(-1) + offsets  # Shape: [N, kernel_size]
#
#     # Expand and flatten coordinates
#     pxs_b = pxs_b.unsqueeze(2).expand(-1, -1, kernel_size).reshape(-1)#.squeeze()  # Flatten to [N * kernel_size^2]
#     pys_b = pys_b.unsqueeze(1).expand(-1, kernel_size, -1).reshape(-1)#.squeeze()  # Flatten to [N * kernel_size^2]
#     weights = weights.reshape(-1)  # Flatten to [N * kernel_size^2]
#     interp_weights_flat = interp_weights.repeat_interleave(kernel_size ** 2)
#
#     # Check bounds only for new interpolated locations
#     valid_mask = (pxs_b >= 0) & (pxs_b <= img_w - 1) & (pys_b >= 0) & (pys_b <= img_h - 1)
#     pxs_b = pxs_b[valid_mask]
#     pys_b = pys_b[valid_mask]
#     weights = weights[valid_mask].float()
#     interp_weights_flat = interp_weights_flat[valid_mask].float()
#     # Apply contributions to the image
#     img.index_put_(
#         (pys_b, pxs_b),
#         interp_weights_flat * weights,
#         accumulate=True
#     )
#     return img


def gaussian_interpolate_to_image(
    pxs, pys,
    dxs, dys,
    interp_weights,
    img,
    sigma=1.0,
    kernel_size=3
):
    """
    Gaussian splatting with per-event normalization and boundary-safe renormalization,
    approximating bilinear interpolation with minimal MAE.

    Args:
        pxs, pys:        [N] integer base pixel coordinates
        dxs, dys:        [N] subpixel offsets in [-0.5, 0.5)
        interp_weights:  [N] per-sample scalar weight
        img:             [H, W] accumulation image (modified in-place)
        sigma:           Gaussian std (in pixels)
        kernel_size:     odd integer (e.g., 3 or 5)

    Returns:
        img: updated image
    """
    device = pxs.device
    img_h, img_w = img.shape
    N = pxs.shape[0]

    # ------------------------------------------------------------------
    # 1. Kernel offsets
    # ------------------------------------------------------------------
    radius = (kernel_size - 1) // 2
    offsets = torch.arange(-radius, radius + 1, device=device)

    # ------------------------------------------------------------------
    # 2. Separable Gaussian weights
    # ------------------------------------------------------------------
    wx = torch.exp(-0.5 * ((dxs[:, None] - offsets[None, :]) ** 2) / (sigma ** 2))  # [N, K]
    wy = torch.exp(-0.5 * ((dys[:, None] - offsets[None, :]) ** 2) / (sigma ** 2))  # [N, K]

    # Per-sample normalization (assumes full kernel support)
    #wx = wx / (wx.sum(dim=1, keepdim=True) + 1e-12) *2
    #wy = wy / (wy.sum(dim=1, keepdim=True) + 1e-12) * M

    # Outer product → [N, K, K]
    w = wx[:, :, None] * wy[:, None, :]
    w = w / (w.sum(dim=(1, 2), keepdim=True) + 1e-12)

    # ------------------------------------------------------------------
    # 3. Pixel coordinates
    # ------------------------------------------------------------------
    pxs_b = pxs[:, None] + offsets[None, :]  # [N, K]
    pys_b = pys[:, None] + offsets[None, :]  # [N, K]

    pxs_b = pxs_b[:, :, None].expand(-1, -1, kernel_size).reshape(-1)
    pys_b = pys_b[:, None, :].expand(-1, kernel_size, -1).reshape(-1)

    w = w.reshape(-1)
    interp_w = interp_weights.repeat_interleave(kernel_size ** 2)

    # Event index for per-event renormalization
    event_ids = torch.arange(N, device=device).repeat_interleave(kernel_size ** 2)


    # ------------------------------------------------------------------
    # 4. Boundary mask
    # ------------------------------------------------------------------
    valid = (
        (pxs_b >= 0) & (pxs_b < img_w) &
        (pys_b >= 0) & (pys_b < img_h)
    )

    pxs_b = pxs_b[valid]
    pys_b = pys_b[valid]
    w = w[valid]
    interp_w = interp_w[valid]
    event_ids = event_ids[valid]

    # ------------------------------------------------------------------
    # 5. Per-event renormalization after boundary clipping
    # ------------------------------------------------------------------
    # w_sum = torch.zeros(N, device=device)
    # w_sum.index_add_(0, event_ids, w)
    #
    # w = w / (w_sum[event_ids].clamp_min(1e-12))

    # ------------------------------------------------------------------
    # 6. Accumulate
    # ------------------------------------------------------------------
    img.index_put_(
        (pys_b, pxs_b),
        interp_w * w,
        accumulate=True
    )

    return img



#
# def gaussian_interpolate_to_image_flow(pxs, pys, dxs, dys, interp_weights, img,sigma_xy, kernel_size=[5,1]):
#     """
#     Perform Gaussian interpolation with a normalized kernel of arbitrary size.
#     """
#     device = pxs.device
#     img_h, img_w = img.shape  # Get image dimensions
#     radius_x = (kernel_size[0]-1)//2+1
#     radius_y =(kernel_size[1]-1)//2+1
#     offsets_x = torch.arange(-radius_x, radius_x + 1,device=device)  # Offsets: [-radius, ..., radius]
#     offsets_y = torch.arange(-radius_y, radius_y + 1,device=device)  # Offsets: [-radius, ..., radius]
#     offsets_shape = [offsets_x.shape[0],offsets_y.shape[0]]
#
#     weights_x = torch.exp(-0.5  * ((dxs.unsqueeze(-1) - offsets_x) ** 2) / ((sigma_xy[0]) ** 2))
#     weights_y = torch.exp(-0.5 * ((dys.unsqueeze(-1) - offsets_y) ** 2) / ((sigma_xy[1]) ** 2))
#     weights = weights_x.unsqueeze(2) * weights_y.unsqueeze(1)  # Shape: [N, kernel_size[0], kernel_size[1]]
#
#     # Normalize weights
#     weights = weights / weights.sum()
#
#     # Compute affected pixel coordinates
#     pxs_b = pxs.unsqueeze(-1) + offsets_x  # Shape: [N, offsets_x_size]
#     pys_b = pys.unsqueeze(-1) + offsets_y  # Shape: [N, offsets_y_size]
#
#     # Expand and flatten coordinates
#     pxs_b = pxs_b.unsqueeze(2).expand(-1, -1, offsets_shape[1]).reshape(-1)#.squeeze()  # Flatten to [N * kernel_size[0]*kernel_size[1]]
#     pys_b = pys_b.unsqueeze(1).expand(-1, offsets_shape[0], -1).reshape(-1)#.squeeze()  # Flatten to [N * * kernel_size[0]*kernel_size[1]]
#     weights = weights.reshape(-1)  # Flatten to [N * kernel_size[0]* kernel_size[1]]
#     interp_weights_flat = interp_weights.repeat_interleave(offsets_shape[0]*offsets_shape[1])
#
#     # Check bounds only for new interpolated locations
#     valid_mask = (pxs_b >= 0) & (pxs_b <= img_w - 1) & (pys_b >= 0) & (pys_b <= img_h - 1)   # in shape N *  K2
#     pxs_b = pxs_b[valid_mask]
#     pys_b = pys_b[valid_mask]
#     weights = weights[valid_mask]   #   weights in shape N  *  K1 *  K2
#     interp_weights_flat = interp_weights_flat[valid_mask]
#     # Apply contributions to the image
#     img.index_put_(
#         (pys_b, pxs_b),
#         interp_weights_flat * weights,
#         accumulate=True
#     )
#     return img


def gaussian_interpolate_to_image_flow(
    pxs, pys,
    dxs, dys,
    interp_weights,
    img,
    sigma_xy,
    kernel_size=(5, 1),
):
    """
    Flow-aware anisotropic Gaussian splatting (per-event normalized).
    Boundary handling: simple clipping (no post-clip renormalization).
    """

    device = pxs.device
    img_h, img_w = img.shape
    N = pxs.shape[0]

    # ------------------------------------------------------------------
    # 1. Kernel offsets
    # ------------------------------------------------------------------
    kx, ky = kernel_size
    rx = (kx - 1) // 2
    ry = (ky - 1) // 2

    offsets_x = torch.arange(-rx, rx + 1, device=device)  # [Kx]
    offsets_y = torch.arange(-ry, ry + 1, device=device)  # [Ky]

    Kx = offsets_x.numel()
    Ky = offsets_y.numel()
    K2 = Kx * Ky

    # ------------------------------------------------------------------
    # 2. Separable Gaussian kernel (per-event, per-kernel)
    # ------------------------------------------------------------------
    wx = torch.exp(
        -0.5 * ((dxs[:, None] - offsets_x[None, :]) ** 2) / (sigma_xy[0] ** 2)
    )  # [N, Kx]

    wy = torch.exp(
        -0.5 * ((dys[:, None] - offsets_y[None, :]) ** 2) / (sigma_xy[1] ** 2)
    )  # [N, Ky]

    # per-event 2D kernel
    w = wx[:, :, None] * wy[:, None, :]          # [N, Kx, Ky]

    # Per-event normalization in the kernel domain.
    w = w / (w.sum(dim=(1, 2), keepdim=True) + 1e-12)

    # ------------------------------------------------------------------
    # 3. Pixel coordinates (expand to event–pixel domain)
    # ------------------------------------------------------------------
    pxs_b = pxs[:, None] + offsets_x[None, :]     # [N, Kx]
    pys_b = pys[:, None] + offsets_y[None, :]     # [N, Ky]

    pxs_b = pxs_b[:, :, None].expand(-1, -1, Ky).reshape(-1)  # [N*K2]
    pys_b = pys_b[:, None, :].expand(-1, Kx, -1).reshape(-1)  # [N*K2]

    w = w.reshape(-1)                              # [N*K2]
    interp_w = interp_weights.repeat_interleave(K2)

    # ------------------------------------------------------------------
    # 4. Boundary mask (simple clipping)
    # ------------------------------------------------------------------
    valid = (
        (pxs_b >= 0) & (pxs_b < img_w) &
        (pys_b >= 0) & (pys_b < img_h)
    )

    pxs_b = pxs_b[valid]
    pys_b = pys_b[valid]
    w = w[valid]
    interp_w = interp_w[valid]

    # ------------------------------------------------------------------
    # 5. Accumulate
    # ------------------------------------------------------------------
    img.index_put_(
        (pys_b, pxs_b),
        interp_w * w,
        accumulate=True
    )

    return img

def gen_rec_mla_grid(cx_start, cy_start,mla_dim,pitch_px):
    mla_centers = np.zeros([mla_dim, mla_dim, 2])
    for q in range(mla_dim):
        cy = cy_start + (q) * pitch_px
        for p in range(mla_dim):
            cx = cx_start + (p) * pitch_px
            mla_centers[q, p, 0] = cx
            mla_centers[q, p, 1] = cy

    cxy = np.concatenate([[mla_centers[:, :, 0].flatten()], [mla_centers[:, :, 1].flatten()]])
    return cxy



def gen_rect_mla_grid(cx_start, cy_start,mla_dim,pitch_px):
    mla_centers = np.zeros([mla_dim[0], mla_dim[1], 2])
    for q in range(mla_dim[0]):
        cx = cx_start + (q) * pitch_px
        for p in range(mla_dim[1]):
            cy = cy_start + (p) * pitch_px
            mla_centers[q, p, 0] = cx
            mla_centers[q, p, 1] = cy

    cxy = np.concatenate([[mla_centers[:, :, 0].flatten()], [mla_centers[:, :, 1].flatten()]])
    return cxy


def window_event_to_img(xs, ys, ts, ps,sensor_size,s=int(1 / 60 * 12 * 1e6), win=int(1 / 60 * 3 * 1e6), polar = True):
    xs_win, ys_win, ts_win, ps_win = time_window(xs, ys, ts, ps, s,
                                                 win)
    img = events_to_image(xs_win, ys_win, ps_win,
                          sensor_size, interpolation='bilinear', polar = polar)
    return img


def denoise_event(xs, ys, ts, ps, sensor_size, thre = 100):
    img = events_to_image(xs, ys, ps, sensor_size, interpolation=False)
    mask = np.abs(img) < thre  # Generate binary mask
    # Ensure mask coordinates are integers
    loc_y, loc_x = np.where(mask == True)
    # Ensure xs and ys are integers
    xs = xs.astype(int)
    ys = ys.astype(int)
    # Create a set of valid locations
    loc_set = set(zip(loc_x, loc_y))
    # Generate boolean mask for filtering xs, ys, ts, ps
    mask = np.array([point in loc_set for point in zip(xs, ys)])
    # Apply mask to filter data
    xs, ys, ts, ps = xs[mask], ys[mask], ts[mask], ps[mask]
    return xs, ys, ts, ps
def get_signed_polarity(pt):
    # only for original {0,1}
    return 2.0 * pt.float() - 1.0

def get_interp_weights(ps_gray, use_polarity=True):
    # for gray-level weighted events
    if not use_polarity:
        return ps_gray.abs()  # or torch.ones_like(ps_gray)
    else:
        return ps_gray
def events_to_image_torch(xt, yt, pt, image_size,polar= False,
                          device=None):
    if device is None:
        device = xt.device
    img_size = (image_size[0] , image_size[1])
    # mask = ((xt <= clipx) & (xt > 0)) & ((yt <= clipy) | (yt > 0))
    img = torch.zeros(img_size).to(device).float()
    pxs = (xt.floor()).float()
    pys = (yt.floor()).float()
    dxs = (xt - pxs).float()#[mask]
    dys = (yt - pys).float()#[mask]
    pxs = pxs.long()#[mask]
    pys = pys.long()#[mask]

    if polar == False:
        interp_weights = torch.ones_like(pt)
    else:
        npt = pt.clone()#[mask]
        npt[torch.where(npt == 0)] = -1
        interp_weights = npt
    img=interpolate_to_image(pxs, pys, dxs, dys, interp_weights , img)
    return img

def plot_3d_surface(ax, zdata, cmap_phi='inferno'):
    zdata = zdata - np.median(zdata)
    z_max = zdata.max() - zdata.min()
    H, W = zdata.shape
    x = np.arange(W)
    y = np.arange(H)
    X, Y = np.meshgrid(x, y)

    ax.set_axis_off()

    surf = ax.plot_surface(
        X, Y, zdata,
        cmap=cmap_phi,
        linewidth=0,
        antialiased=False,
        rstride=2, cstride=2,
        edgecolor='none'   # Disable grid lines.
    )

    ax.set_ylim(H, 0)
    ax.set_xlim(0, W)
    ax.set_zlim(-z_max, z_max)
    ax.view_init(elev=45, azim=-60)
    return surf



def Gau_events_to_image_torch(xt, yt, pt, image_size,polar= False, sigma=0.6,
                          device=None):
    if device is None:
        device = xt.device
    img_size = (image_size[0], image_size[1])
    # mask = ((xt <= clipx) & (xt > 0)) & ((yt <= clipy) | (yt > 0))
    img = torch.zeros(img_size).to(device).float()
    pxs = (xt.floor()).float()
    pys = (yt.floor()).float()
    dxs = (xt - pxs).float()  # [mask]
    dys = (yt - pys).float()  # [mask]
    pxs = pxs.long()  # [mask]
    pys = pys.long()  # [mask]

    if polar == False:
        interp_weights = pt.abs()
    else:
        npt = pt.clone()  # [mask]
        npt[torch.where(npt == 0)] = -1
        interp_weights = npt
    img=gaussian_interpolate_to_image(pxs, pys, dxs, dys, interp_weights , img, sigma)
    return img

def events_to_image(xs, ys, ps, sensor_size=(180, 240), interpolation='bilinear', polar =False,padding=False):
    """
    Place events into an image using numpy
    """
    if interpolation == 'bilinear' and xs.dtype is not torch.long and xs.dtype is not torch.long:
        xt, yt, pt = torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(ps)
        xt, yt, pt = xt.float(), yt.float(), pt.float()
        img = events_to_image_torch(xt, yt, pt,sensor_size, polar)
        img = img.numpy()
        img[np.argmax(img) // sensor_size[1], np.argmax(img) % sensor_size[1]] = 0
    else:
        coords = np.stack((ys, xs))
        try:
            abs_coords = np.ravel_multi_index(coords, sensor_size)
        except ValueError:
            print("Issue with input arrays! coords.shape={}, sum(coords)={}, sensor_size={}. \n minx={}, maxx={}, miny={}, maxy={}".format(coords.shape, np.sum(coords), sensor_size, np.min(xs), np.max(xs), np.min(ys), np.max(ys)))
            raise ValueError
        if polar == True:
            img = np.bincount(abs_coords, weights=ps, minlength=sensor_size[0] * sensor_size[1])
        else:
            img = np.bincount(abs_coords, weights=np.ones(ps.size), minlength=sensor_size[0]*sensor_size[1])
    img = img.reshape(sensor_size)
    return img


def img_interp(img,  M,sensor_size = None):
    """
    Perform bilinear interpolation to magnify an image by a factor of M using PyTorch's interpolate.

    Args:
        img: Input image, can be a NumPy array or PyTorch tensor.
        M: Magnification factor (e.g., 2).

    Returns:
        Magnified image as the same type as the input.
    """
    # Detect if the input is a NumPy array
    is_numpy = isinstance(img, np.ndarray)

    # Convert NumPy to PyTorch tensor if necessary
    if is_numpy:
        img = torch.tensor(img, dtype=torch.float32)

    # Ensure the image has a batch and channel dimension
    while len(img.shape) < 4:
        img = img.unsqueeze(0)
    if sensor_size is None:
        # Perform bilinear interpolation
        H_new = int(sensor_size[0] * M)
        W_new = int(sensor_size[1] * M)
    else:
        H_new = int(img.shape[2] * M)
        W_new = int(img.shape[3] * M)
    img_resized = F.interpolate(img, size=(H_new, W_new), mode='bilinear', align_corners=False)

    # Remove batch and potentially channel dimensions if necessary
    img_resized = img_resized.squeeze(0)  # Remove batch dimension
    if img_resized.shape[0] == 1:  # Grayscale
        img_resized = img_resized.squeeze(0)

    # Convert back to NumPy if the input was NumPy
    if is_numpy:
        img_resized = img_resized.numpy()

    return img_resized




def warp(xs, ys, ts, t0, v_xx, v_yy):
    dt = ts - t0
    warp_x = xs - v_xx * dt
    warp_y = ys - v_yy * dt
    return warp_x, warp_y

def warp_Disp(xs, ys, ts, t0, Dxy):
    dt = ts - t0
    v_xx, v_yy = Dxy/(ts.max()-ts.min())
    warp_x = xs - v_xx * dt
    warp_y = ys - v_yy * dt
    return warp_x, warp_y

def warp_and_interp_flow(xt, yt, tt, pt, tt0, Dxy_max_torch, M, sensor_size = (260, 346),
                         device = None, polar = True, mode="nearest"):
    Ny, Nx = sensor_size
    if device == None:
        device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    xtw,ytw = warp_Disp(xt, yt, tt, tt0, Dxy_max_torch)
    iwe = events_to_image_torch_sr_flow(xtw, ytw, pt, sensor_size, M, flow_xy = Dxy_max_torch, device= device, polar = polar)
    return F.interpolate(iwe.unsqueeze(0).unsqueeze(0),(int(Ny*M),int(Nx*M)), mode=mode,align_corners=True).squeeze()










### e.g. points = np.float32([[30, 30], [10, 40], [40, 10], [5, 15]])
def persp_img_process(img,points1,points2):
    M = cv2.getPerspectiveTransform(points1, points2)
    Perspective_img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    return Perspective_img

def persp_inv_img_process(img,points1,points2):
    M_f = cv2.getPerspectiveTransform(points1, points2)
    M = np.linalg.inv(M_f)
    Perspective_img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    return Perspective_img



def persp_eve_process(xs,ys,points1,points2):
    M = cv2.getPerspectiveTransform(points1, points2)
    x_persp = (xs * M[0, 0] + ys * M[0, 1] + M[0, 2]) / (xs * M[2, 0] + ys * M[2, 1] + M[2, 2])
    y_persp = (xs * M[1, 0] + ys * M[1, 1] + M[1, 2]) / (xs * M[2, 0] + ys * M[2, 1] + M[2, 2])
    return x_persp,y_persp


def persp_inv_eve_process(xs,ys,points1,points2):
    M_f = cv2.getPerspectiveTransform(points1, points2)
    M = np.linalg.inv(M_f)
    x_persp = (xs * M[0, 0] + ys * M[0, 1] + M[0, 2]) / (xs * M[2, 0] + ys * M[2, 1] + M[2, 2])
    y_persp = (xs * M[1, 0] + ys * M[1, 1] + M[1, 2]) / (xs * M[2, 0] + ys * M[2, 1] + M[2, 2])
    return x_persp,y_persp





def numpy_rsoe(img):
    return np.sum(np.exp(img))

def test_cost_demo(xt, yt, pt, tt, delta_t, mla_dim, pitch_px, cxy, sensor_size, crop_sz, L=25, nv = 12, type ="var", cross = True, thre = 1, Dxy_0=None,Dxy_ref=None):
    mla_number = mla_dim**2
    ### recalculate and scan the objective function
    xy = np.linspace(-L, L, 2 * nv + 1)
    vxy = xy / delta_t
    cost = np.zeros([2 * nv, 2 * nv, mla_number])
    for p in range(2 * nv):
        v_xx = vxy[p]
        for q in range(2 * nv):
            v_yy = vxy[q]
            xsw, ysw = warp(xt, yt, tt, 0, v_xx, v_yy)
            iwe_i = events_to_image_torch(xsw, ysw, pt, device=None, image_size=sensor_size)[:-1, :-1]
            if type =="bi-var":
                xsw1, ysw1 = warp(xt, yt, tt, tt.min()+delta_t/2, v_xx, v_yy)
                iwe_i1 = events_to_image_torch(xsw1, ysw1, pt, device=None, image_size=sensor_size)[:-1, :-1]
            for i in range(mla_number):
                cx, cy = cxy[:, i]
                iwe_i_crop = iwe_i[int(cy - (crop_sz / 2 + 1)):int(cy + (crop_sz / 2 - 1)), \
                             int(cx - (crop_sz / 2 + 1)):int(cx + (crop_sz / 2 - 1))].cpu().numpy()
                if type == "bi-var":
                    cx1, cy1 = cxy[:, i] + np.asarray([v_xx,v_yy])*(delta_t/2)
                    iwe_i_crop1 = iwe_i1[int(cy1 - (crop_sz / 2 + 1)):int(cy1 + (crop_sz / 2 - 1)), \
                                 int(cx1 - (crop_sz / 2 + 1)):int(cx1 + (crop_sz / 2 - 1))].cpu().numpy()

                #
                # moment = cv2.moments(iwe_i_crop)
                # if moment['m00'] == 0:
                #     Xc = 0
                #     Yc = 0
                # else:
                #     Xc = - (pitch_px / 2) + (moment['m10'] / moment['m00'])
                #     Yc = - (pitch_px / 2) + (moment['m01'] / moment['m00'])
                if type == "var":
                    cost_val = np.var(iwe_i_crop)#-(Xc**2 + Yc**2)/(pitch_px**2)
                if type == "bi-var":
                    cost_val = np.var(iwe_i_crop) + np.var(iwe_i_crop1)
                elif type =="grad":
                    gradx, grady = torch.gradient(iwe_i_crop)
                    cost_val = (torch.norm(gradx, 1) + torch.norm(grady, 1))/2

                elif type == "var_mse":
                    y_grid, x_grid = np.mgrid[:pitch_px, :pitch_px]
                    m00 = np.sum(iwe_i_crop)
                    # if m00 == 0:
                    #     Yc = 0
                    #     Xc = 0
                    # else:
                    #     Yc = moment['nu02']
                    #     Xc = moment['nu20']
                    #cost_val= -(Xc + Yc) #np.var(iwe_i_crop) - (Xc + Yc)/(pitch_px**2)
                elif type == "rsoe":
                    cost_val = numpy_rsoe(iwe_i_crop) + np.var(iwe_i_crop) * np.sum(iwe_i_crop)

                if cost_val >= thre: # this is the cost treshold
                    cost[q, p, i] = cost_val
                else:
                    cost[q, p, i] = 0

    #cost[nv, nv, :] = 0
    if cross ==True:
        cost[nv, int(nv/2):int(nv*3/2), :] = 0
        cost[int(nv/2):int(nv*3/2), nv, :] = 0

    # Show the scanned vost
    fig, ax = plt.subplots(mla_dim, mla_dim, figsize=(8, 8))
    fig.suptitle("cost function")
    Dxy = np.zeros([2,mla_number])
    dx = (2*L+1) / (2*nv+1)
    for i in range(mla_number):
        plt.subplot(mla_dim, mla_dim, i + 1)
        ax = plt.gca()
        sub_cost = cost[:, :, i]
        ax.imshow(sub_cost)
        if sub_cost.max() >= thre:  # this is the cost treshold
            x_max = np.argmax(sub_cost) % (sub_cost.shape[0])
            y_max = np.argmax(sub_cost) // (sub_cost.shape[0])

        else:
            x_max = nv#Dxy_0[0,i]/dx+ nv
            y_max = nv#Dxy_0[1,i]/dx+ nv
        if Dxy_0.any() != None:
            ax.scatter(x=(Dxy_0[0,i]/dx+ nv),y=(Dxy_0[1,i]/dx+ nv),c="g",marker='x')
        if Dxy_ref.any() != None:
            ax.scatter(x=(Dxy_ref[0, i] / dx + nv), y=(Dxy_ref[1, i] / dx + nv), c="b", marker='x')
        # if i in [0,mla_dim-1,mla_number-mla_dim,mla_number-1] :
        #     x_max, y_max =[nv,nv]
        ax.scatter(x_max,y_max,c="r",marker='x')
        # ax.invert_yaxis()

        Dxy[0,i] =  x_max - nv
        Dxy[1,i] =  y_max - nv
    return Dxy *dx


def test_initial(xt, yt, pt, tt, delta_t, mla_dim, pitch_px, cxy, sensor_size, Dxy_0,var_ind, it_ind = False ,crop_sz = None):
    mla_number = mla_dim**2
    cost = np.zeros(mla_number)
    if crop_sz == None:
        crop_sz = (pitch_px / 2 )
    for i in range(mla_number):
        v_xx,v_yy = Dxy_0[:,i]/ delta_t
        cx, cy = cxy[:, i]
        #xsw, ysw = warp(xt, yt, tt, 0, v_xx, v_yy)
        #iwe_i = events_to_image_torch(xsw, ysw, pt, device=None, image_size=sensor_size)[:-1, :-1]
        iwe_i = events_to_image_torch(xt, yt, pt, device=None, image_size=sensor_size)[:-1, :-1]
        iwe_i_crop = iwe_i[int(cy - crop_sz):int(cy + crop_sz), \
                     int(cx - crop_sz):int(cx + crop_sz)].cpu().numpy()
        moment = cv2.moments(iwe_i_crop)
        if moment['m00'] ==0:
            Xc = - (crop_sz)
            Yc = - (crop_sz)
        else:
            Xc = - (crop_sz) + (moment['m10'] / moment['m00'])
            Yc = - (crop_sz) + (moment['m01'] / moment['m00'])
        if it_ind ==False:
            cost_val = np.var(iwe_i_crop) #- (moment['nu20'] + moment['nu02'])
        else:
            cost_val = - (moment['nu20'] + moment['nu02'])
            #  np.var(iwe_i_crop) - (Xc**2 + Yc**2)/(crop_sz*2)**2  # np.var(iwe_i_crop)
        # else:
        #     gradx, grady = torch.gradient(iwe_i_crop)
        #     cost_val = (torch.norm(gradx, 1) + torch.norm(grady, 1)) / 2
        cost[i] = cost_val
    return cost





def remove_hot_pixels(xs, ys, ts, ps, device,sensor_size=(260, 346), num_hot=500):
    xt, yt, tt, pt = np_to_torch(xs, ys, ts, ps, device)
    img=events_to_image_torch(xt, yt, pt, sensor_size).cpu()
    hot = np.array([])
    for i in range(num_hot):
        maxc = np.unravel_index(np.argmax(img), sensor_size)
        #print("{} = {}".format(maxc, img[maxc]))
        img[maxc] = 0
        h = np.where((xs == maxc[1]) & (ys == maxc[0]))
        hot = np.concatenate((hot, h[0]))
    hot=hot.astype(int)

    xs, ys, ts, ps = np.delete(xs, hot), np.delete(ys, hot), np.delete(ts, hot), np.delete(ps, hot)
    return xs, ys, ts, ps




def crop_event(xs,ys,ps,ts,clipxy):
    ts = ts-ts[0]
    mask = np.ones(xs.size)
    # Do: clip_out_of_range
    # zero_v = 0
    # ones_v = 1
    clipx = clipxy[:,0]
    clipy = clipxy[:, 1]
    # mask = np.where((xs >= clipx[1]) | (xs <= clipx[0]), zero_v, ones_v) * np.where((ys >= clipy[1]) | (ys <= clipy[0]), zero_v, ones_v)
    # xs, ys, ps, ts = [xs, ys, ps, ts] * mask
    loc = np.where((xs <= clipx[1]) & (xs >= clipx[0])&(ys <= clipy[1]) & (ys >= clipy[0]))
    xs, ys, ps, ts = [xs[loc], ys[loc], ps[loc], ts[loc]]
    return xs,ys,ps,ts

# def initialize_vilocity_SAE(xs,ys,ps,ts):
#     velocity =[]
#     for i in range(mla_number):
#
#         xsi, ysi, psi, tsi = crop_event(xs, ys, ps, ts,
#                                     clipxy=np.asarray([cxy[:, i] - pitch_px / 2, cxy[:, i] + pitch_px / 2]))
#         xsi = xsi - (cxy[:, i][0] - pitch_px / 2)
#         ysi = ysi - (cxy[:, i][1] - pitch_px / 2)
#         g_x = np.gradient(xsi)
#         g_y = np.gradient(ysi)
#         g_tk = np.gradient(tsi)
#         GX = np.vstack([g_x, g_y]).T
#         velocity.append( np.dot(np.linalg.pinv(GX), -g_tk))
#     velocity=np.asarray(velocity).T



def crop_img(img, ROI,  invert = False):
    """
    ROI: [[x0, x1],
          [y0, y1]]
    """
    x0, x1 = int(ROI[0, 0]), int(ROI[0, 1])
    y0, y1 = int(ROI[1, 0]), int(ROI[1, 1])
    img_crop = img[min(y0, y1):max(y0, y1), min(x0, x1):max(x0, x1)]

    if invert == True:
        if y1 < y0:
            img_crop = np.flipud(img_crop)   # 上下反转
        if x1 < x0:
            img_crop = np.fliplr(img_crop)   # 左右反转

    return img_crop

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


# def crop_center_img(img, crop_sz = [20,20]):
#     center,  = img.shape
#     return img[]

def Gaussian_Distribution(N=2, M=1000, m=0, sigma=4):
    '''
    Parameters
    ----------
    N 维度
    M 样本数
    m 样本均值
    sigma: 样本方差

    Returns
    -------
    data  shape(M, N), M 个 N 维服从高斯分布的样本
    Gaussian  高斯分布概率密度函数
    '''
    mean = np.zeros(N) + m  # 均值矩阵，每个维度的均值都为 m
    cov = np.eye(N) * sigma  # 协方差矩阵，每个维度的方差都为 sigma

    # 产生 N 维高斯分布数据
    #data = np.random.multivariate_normal(mean, cov, M)
    # N 维数据高斯分布概率密度函数
    Gaussian = multivariate_normal(mean=mean, cov=cov)
    X, Y = np.meshgrid(np.linspace(-1, 1, M), np.linspace(-1, 1, M))
    # 二维坐标数据
    d = np.dstack([X, Y])
    # 计算二维联合高斯概率
    Z = Gaussian.pdf(d).reshape(M, M)
    return Z   #data

def blaze_grating(theta, wavelength, N_sub, pixel_sz):

    """
    :param theta: float: the output angle in the unit of degree
    :param wavelength: float: the monochromatic illumination
    :param N: int: the resolution of sub-area
    :param pixel_sz: float: the pixel size
    :return: wrapped phase : ndarray [N,n] , the phase of the blazed grating in sub-area that will be shown on SLM
             grating_n: int: the number of the step of blazed grating in the 2pi period
    """

    F_g_0 = LP.Begin(1, wavelength, N_sub)  ## Initialize the field of blaze grating
    freq = np.sin(theta/180*np.pi)/wavelength
    d_phi = 2 * np.pi * freq * pixel_sz
    #grating_n = int(1/ (freq*d)) # step of the grating
    phi  = np.zeros([N_sub,N_sub])
    for i in range(N_sub):
        phi[:,i] = d_phi * i
    F_g = LP.SubPhase(F_g_0,phi)
    Phi_g =LP.Phase(F_g)
    return Phi_g#, grating_n


def bilinear_interpolation_ROI(v, cxy, img_size, bat_sz=5):
    # Initialize the velocity image
    h = int(img_size[0])
    w = int(img_size[1])
    velocity_image = np.zeros((h, w))
    # Get the dimensions of the velocity array
    nv = v.shape[0]
    ### dx,dy = bat_sz
    x_range = w * bat_sz
    y_range = h * bat_sz
    # Iterate over each velocity point
    for i in range(nv):
        # Calculate the indices of the surrounding pixels for bilinear interpolation
        x0 = min(int(cxy[0, i] // bat_sz),  w - 1)
        y0 = min(int(cxy[1, i] // bat_sz),  h - 1)
        x1 = min(x0 + 1, w - 1)  # Fix the out-of-bounds issue
        y1 = min(y0 + 1, h - 1)  # Fix the out-of-bounds issue
        # Calculate the interpolation weights
        wx = cxy[0, i]/bat_sz -  x0
        wy = cxy[1, i]/bat_sz -  y0


        # Perform bilinear interpolation
        velocity_image[y0, x0] += (1 - wx) * (1 - wy) * v[i]
        velocity_image[y0, x1] += wx * (1 - wy) * v[i]
        velocity_image[y1, x0] += (1 - wx) * wy * v[i]
        velocity_image[y1, x1] += wx * wy * v[i]

    return velocity_image

def events_to_image_torch_sr(xt, yt, pt, img_size, M = 1,
                             device = None, polar = False):
    if device is None:
        device = xt.device
    img_size_sz = np.asarray([int(img_size[0] * M), int(img_size[1] * M)])
    img = torch.zeros(img_size_sz[0], img_size_sz[1]).to(device).float()
    xt = xt * M
    yt = yt * M
    pxt = (xt.floor()).float()
    pyt = (yt.floor()).float()
    dxt = (xt - pxt).float()
    dyt = (yt - pyt).float()
    pxt = pxt.long()
    pyt = pyt.long()
    if polar == False:
        interp_weights = pt.abs()
    else:
        npt = pt.clone()
        npt[torch.where(npt == 0)] = -1
        interp_weights = npt
    img = interpolate_to_image(pxt, pyt, dxt, dyt, interp_weights, img)
    return img


def Gau_events_to_image_torch_sr(xt, yt, pt, image_size, M,sigma= 1,
                          device=None, polar = False):  ### flow_xy control the grid size of the flow.
    if device is None:
        device = xt.device
    img_size_sz = np.asarray([int(image_size[0] * M), int(image_size[1] * M)])
    img = torch.zeros(img_size_sz[0], img_size_sz[1]).to(device).float()
    xt = xt * M
    yt = yt * M
    pxt = (xt.round()).float()
    pyt = (yt.round()).float()
    dxt = (xt - pxt).float()
    dyt = (yt - pyt).float()
    pxt = pxt.long()
    pyt = pyt.long()
    if polar == False:
        interp_weights = pt.abs()
    else:
        npt = pt.clone()
        npt[torch.where(npt == 0)] = -1
        interp_weights = npt
    kernel_sz = int(3 * sigma * M + 1) | 1
    interp_weights = interp_weights.to(img.dtype)
    #kernel_sz = 2 * (M+1)//2 + M % 2
    img = gaussian_interpolate_to_image(pxt, pyt, dxt, dyt, interp_weights, img, M/2*sigma, kernel_size =  kernel_sz)
    return img


def make_np(x):
   return x.detach().cpu().numpy()
def plot_ax_roi(ax, img, cmap = "gray", axis_ind = "off", roi = None, Dxy_ori =np.asarray([[0,0],[0,0]]), M = 1, norm_ind = False):
    if norm_ind == True:
        img = img- np.median(img)
        pos = img[img > 0]
        neg = img[img < 0]

        # 99% 分位（你也可以换 99.5 或 99.9）
        vmax = np.percentile(pos, 99) if pos.size else 1.0
        vmin = np.percentile(neg, 1) if neg.size else -1.0  # 注意：neg是负数，所以用 1% 更接近“最负端”

        norm1 = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        cax = ax.imshow(img, cmap=cmap, norm=norm1)
        #cax  = ax.imshow(img, cmap=cmap, norm=TwoSlopeNorm(vmin=img.min(), vcenter=0, vmax=img.max()))
    else:
        cax = ax.imshow(img, cmap=cmap)
    if roi is None:
        roi = [[0,img.shape[1]],[img.shape[0], 0]]
    ax.set_xlim(np.asarray(roi[0] * M) + np.asarray(Dxy_ori[0]));
    ax.set_ylim(np.asarray(roi[1] * M) + np.asarray(Dxy_ori[1]));
    ax.axis(axis_ind)
    return cax


def Flow_Gau_events_to_image_torch_sr(xt, yt, pt, image_size, M, flow_xy = torch.tensor([1, 1]),sigma=1,
                          device=None, polar = False):  ### flow_xy control the grid size of the flow.
    if device is None:
        device = xt.device
    flow_xy = flow_xy.to(device)
    max_flow = (torch.abs(flow_xy)).max()
    if max_flow  <= 1e-5:
        scl = torch.zeros_like(flow_xy)
    else:
        scl  = torch.abs(flow_xy)/max_flow
    M_xy =torch.ones_like(flow_xy).float() + (M-1) * scl
    img_size = np.asarray([int(image_size[0]* M), int(image_size[1]* M)])
    img = torch.zeros(img_size[0], img_size[1]).float().to(device)
    xt = xt *  M
    yt = yt *  M
    pxs = (xt.round()).float()
    pys = (yt.round()).float()
    dxs = (xt - pxs).float()
    dys = (yt - pys).float()
    pxs = pxs.long()
    pys = pys.long()
    pt = pt.float()
    if polar == False:
        interp_weights = pt.abs()
    else:
        npt = pt.clone()
        npt[torch.where(npt == 0)] = -1
        interp_weights = npt
    kernel_sz = int(3 * sigma*M + 1) | 1
    #     # ensure normalize##
    #kernel_xy =  torch.ceil(kernel_sz* scl.roll(-1)).to(int).detach().cpu().numpy()
    #sigma_xy = sigma*M_xy.roll(-1)
    sigma_xy = M/2*sigma / M_xy
    img = gaussian_interpolate_to_image_flow(pxs, pys, dxs, dys, interp_weights, img,  sigma_xy, kernel_size =  [kernel_sz, kernel_sz])
    return img


def events_to_image_sr(xs, ys, ps, sensor_size=(180, 240), M =2, interpolation='bilinear', padding=False):
    """
    Place events into an image using numpy
    """
    image_size = np.asarray([sensor_size[0], sensor_size[1]])
    if interpolation == 'bilinear' and xs.dtype is not torch.long and xs.dtype is not torch.long:
        xt, yt, pt = torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(ps)
        xt, yt, pt = xt.float(), yt.float(), pt.float()
        img = events_to_image_torch_sr(xt, yt, pt, image_size, M,
                          device=None)
        img = img.numpy()
    else:
        coords = np.stack((ys, xs))
        try:
            abs_coords = np.ravel_multi_index(coords, sensor_size)
        except ValueError:
            print("Issue with input arrays! coords.shape={}, sum(coords)={}, sensor_size={}. \n minx={}, maxx={}, miny={}, maxy={}".format(coords.shape, np.sum(coords), sensor_size, np.min(xs), np.max(xs), np.min(ys), np.max(ys)))
            raise ValueError
        img = np.bincount(abs_coords, weights=np.ones(xs.size), minlength=sensor_size[0]*sensor_size[1])
    return img

def compute_M_xy(M, flow_xy = torch.tensor([1, 1]),):
    max_flow = (torch.abs(flow_xy)).max()
    if max_flow <= 1e-5:
        M_xy = torch.ones_like(flow_xy)  # torch.tensor([0.0,0.0]).to(device)
    else:
        scl = torch.abs(flow_xy) / max_flow
        M_xy = 1 + scl*(M-1)
    return  M_xy




def events_to_image_torch_sr_flow(xt, yt, pt, image_size, M, flow_xy = torch.tensor([1, 1]),
                          device=None, polar = False):  ### flow_xy control the grid size of the flow.
    if device is None:
        device = xt.device
    flow_xy = flow_xy.to(device)
    max_flow = (torch.abs(flow_xy)).max()
    if max_flow  <= 1e-5:
        scl = torch.zeros_like(flow_xy)#torch.tensor([0.0,0.0]).to(device)
    else:
        scl  = torch.abs(flow_xy)/max_flow
    M_xy =1+ (M-1) * scl
    # Do: interpolation == 'bilinear' and padding, resize it
    img_size = np.asarray([int(image_size[0]* M_xy[1]), int(image_size[1]* M_xy[0])])
    img = torch.zeros(img_size[0], img_size[1]).to(device).float()
    xt = xt *  M_xy[0]
    yt = yt *  M_xy[1]
    pxs = (xt.floor()).float()
    pys = (yt.floor()).float()
    dxs = (xt - pxs).float()
    dys = (yt - pys).float()
    pxs = pxs.long()
    pys = pys.long()

    if polar == False:
        interp_weights =  pt.abs()
    else:
        npt = pt.clone()
        npt[torch.where(npt == 0)] = -1
        interp_weights = npt
    # ensure normalize##
    img = interpolate_to_image(pxs, pys, dxs, dys, interp_weights, img)
    return img
def watch_tensor(img, cmap = "gray"):
    if torch.is_tensor(img) == True:
        img  = img.detach().cpu().numpy()
    return  plt.imshow(img,cmap)

def Gau_events_to_image_torch_sr_flow(xt, yt, pt, image_size, M, flow_xy = torch.tensor([1, 1]),sigma=0.6,
                          device=None, polar = False):  ### flow_xy control the grid size of the flow.
    if device is None:
        device = xt.device
    flow_xy = flow_xy.to(device)
    max_flow = (torch.abs(flow_xy)).max()
    if max_flow  <= 1e-5:
        scl = torch.zeros_like(flow_xy)
    else:
        scl  = torch.abs(flow_xy)/max_flow
    M_xy =1+ (M-1) * scl
    img_size = np.asarray([int(image_size[0]* M_xy[1]), int(image_size[1]* M_xy[0])])
    img = torch.zeros(img_size[0], img_size[1]).to(device).float()
    xt = xt *  M_xy[0]
    yt = yt *  M_xy[1]
    pxs = (xt.floor()).float()
    pys = (yt.floor()).float()
    dxs = (xt - pxs).float()
    dys = (yt - pys).float()
    pxs = pxs.long()
    pys = pys.long()
    if polar == False:
        interp_weights = pt.abs()
    else:
        npt = pt.clone()
        npt[torch.where(npt == 0)] = -1
        interp_weights = npt
    kernel_sz = M + (M+1) % 2+2
    # ensure normalize##
    img = gaussian_interpolate_to_image(pxs, pys, dxs, dys, interp_weights, img,sigma, kernel_size =  kernel_sz)
    return img

def roi_to_mask(roi_zoom, image_shape):
    """
    Create a binary mask from roi_zoom.

    Args:
        roi_zoom (np.ndarray or torch.Tensor): shape (2, 2), e.g. [[x0, x1], [y0, y1]]
        image_shape (tuple): (H, W) of the target image

    Returns:
        torch.Tensor: binary mask of shape (H, W)
    """
    if isinstance(roi_zoom, torch.Tensor):
        roi_zoom = roi_zoom.int()
    else:
        roi_zoom = torch.tensor(roi_zoom, dtype=torch.int)

    H, W = image_shape
    mask = torch.zeros((H, W), dtype=torch.bool)

    x0 = max(0, roi_zoom[0, 0])
    x1 = min(W, roi_zoom[0, 1])
    y0 = max(0, roi_zoom[1, 0])
    y1 = min(H, roi_zoom[1, 1])

    mask[y0:y1, x0:x1] = True
    return mask
def Gau_events_to_image_torch_sr_flow_g(xt, yt, pt, image_size, M, flow_xy = torch.tensor([1, 1]),sigma=0.6,
                          device=None, polar = False):  ### flow_xy control the grid size of the flow.
    if device is None:
        device = xt.device
    flow_xy = flow_xy.to(device)
    max_flow = (torch.abs(flow_xy)).max()
    if max_flow  <= 1e-5:
        scl = torch.zeros_like(flow_xy)
    else:
        scl  = torch.abs(flow_xy)/max_flow
    M_xy =1+ (M-1) * scl
    img_size = np.asarray([int(image_size[0]* M_xy[1]), int(image_size[1]* M_xy[0])])
    img = torch.zeros(img_size[0], img_size[1]).to(device).float()
    xt = xt *  M_xy[0]
    yt = yt *  M_xy[1]
    pxs = (xt.floor()).float()
    pys = (yt.floor()).float()
    dxs = (xt - pxs).float()
    dys = (yt - pys).float()
    pxs = pxs.long()
    pys = pys.long()
    if polar == False:
        interp_weights = pt.abs()
    else:
        interp_weights = pt
    kernel_sz = M + (M+1) % 2+2
    # ensure normalize##
    img = gaussian_interpolate_to_image(pxs, pys, dxs, dys, interp_weights, img,sigma, kernel_size =  kernel_sz)
    return img



def events_to_image_sr_flow(xs, ys, ps, sensor_size, M, flow_xy = np.asarray([1, 1]), interpolation='bilinear', padding=False,polar = False):
    """
    Place events into an image using numpy
    """
    image_size = np.asarray([sensor_size[0], sensor_size[1]])
    if interpolation == 'bilinear' and xs.dtype is not torch.long and xs.dtype is not torch.long:
        xt, yt, pt = torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(ps)
        xt, yt, pt = xt.float(), yt.float(), pt.float()
        img = events_to_image_torch_sr_flow(xt, yt, pt, image_size, M, flow_xy,
                          device=None,polar = polar)
        img = img.numpy()
    else:
        coords = np.stack((ys, xs))
        try:
            abs_coords = np.ravel_multi_index(coords, sensor_size)
        except ValueError:
            print("Issue with input arrays! coords.shape={}, sum(coords)={}, sensor_size={}. \n minx={}, maxx={}, miny={}, maxy={}".format(coords.shape, np.sum(coords), sensor_size, np.min(xs), np.max(xs), np.min(ys), np.max(ys)))
            raise ValueError
        img = np.bincount(abs_coords, weights=np.ones(xs.size), minlength=sensor_size[0]*sensor_size[1])
    return img
def denoise_numpy(events, width=346, height=260, filter_time=500000):
    """Drops events that are 'not sufficiently connected to other events in the recording.' In
    practise that means that an event is dropped if no other event occured within a spatial
    neighbourhood of 1 pixel and a temporal neighbourhood of filter_time time units. Useful to
    filter noisy recordings where events occur  in time.
    Parameters:isolated
        events: ndarray of shape [num_events, num_event_channels]
        filter_time: maximum temporal distance to next event, otherwise dropped.
                    Lower values will mean higher constraints, therefore less events.
    Returns:
        filtered set of events.
    """
    events_copy = np.zeros_like(events)
    copy_index = 0

    timestamp_memory = np.zeros((width, height)) + filter_time

    for event in events:
        x = int(event[0])
        y = int(event[1])
        t = event[3]
        timestamp_memory[x, y] = t + filter_time
        if (
                (x > 0 and timestamp_memory[x - 1, y] > t)
                or (x < width - 1 and timestamp_memory[x + 1, y] > t)
                or (y > 0 and timestamp_memory[x, y - 1] > t)
                or (y < height - 1 and timestamp_memory[x, y + 1] > t)
        ):
            events_copy[copy_index] = event
            copy_index += 1

    return events_copy[:copy_index]

def openCV_centroid(image_array,cx,cy,pitch_px):
    image_grid_crop = image_array[int(cy - (pitch_px / 2 )):int(cy + (pitch_px / 2 )), \
             int(cx - (pitch_px / 2 )):int(cx + (pitch_px / 2 ))]
    moment = cv2.moments(image_grid_crop)
    if moment['m00'] == 0:
        Xc = cx
        Yc = cy
    else:
        Xc = int(cx - (pitch_px / 2) )+ (moment['m10'] / moment['m00'])
        Yc = int(cy - (pitch_px / 2) )+ (moment['m01'] / moment['m00'])
    return Xc, Yc

def img_centroid(img,pitch_px):
    moment = cv2.moments(img)
    cy, cx = img.shape/2
    if moment['m00'] == 0:
        Xc = cx
        Yc = cy
    else:
        Xc = int(cx - (pitch_px / 2) )+ (moment['m10'] / moment['m00'])
        Yc = int(cy - (pitch_px / 2) )+ (moment['m01'] / moment['m00'])
    return Xc, Yc

# def get_centroid(img):
#     moment = cv2.moments(norm(img))
#     y, x = img.shape
#     cy, cx = img.shape//2
#     if moment['m00'] == 0:
#         Xc = cx
#         Yc = cy
#     else:
#         Xc = int(cx)+ (moment['m10'] / moment['m00'])
#         Yc = int(cy - (pitch_px / 2) )+ (moment['m01'] / moment['m00'])
#     return Xc, Yc

import math

def lin_log(x, threshold=20):
    """
    linear mapping + logarithmic mapping.

    :param x: float or ndarray
        the input linear value in range 0-255 TODO assumes 8 bit
    :param threshold: float threshold 0-255
        the threshold for transition from linear to log mapping

    Returns: the log value
    """
    # converting x into np.float64.
    if x.dtype is not torch.float64:  # note float64 to get rounding to work
        x = x.double()
    rounding = 1e-8
    f = (1./threshold) * math.log(threshold)
    #y = torch.where(x <= threshold, x*f, torch.log(x))
    y = torch.where(x <= threshold, x * f, torch.log(x+rounding))
    # important, we do a floating point round to some digits of precision
    # to avoid that adding threshold and subtracting it again results
    # in different number because first addition shoots some bits off
    # to never-never land, thus preventing the OFF events
    # that ideally follow ON events when object moves by
    # rounding = 1e8
    # y = torch.round(y*rounding)/rounding

    #return y.float()
    return y

def inv_lin_log(x, threshold=20):
    """
    linear mapping + logarithmic mapping.

    :param x: float or ndarray
        the input linear value in range 0-255 TODO assumes 8 bit
    :param threshold: float threshold 0-255
        the threshold for transition from linear to log mapping

    Returns: the log value
    """
    # converting x into np.float64.
    if x.dtype is not torch.float64:  # note float64 to get rounding to work
        x = x.double()

    f = (1./threshold) * math.log(threshold)

    y = torch.where(x <= threshold, x/f, torch.exp(x))

    # important, we do a floating point round to some digits of precision
    # to avoid that adding threshold and subtracting it again results
    # in different number because first addition shoots some bits off
    # to never-never land, thus preventing the OFF events
    # that ideally follow ON events when object moves by
    # rounding = 1e8
    # y = torch.round(y*rounding)/rounding

    #return y.float()
    return y.float()
def lin_log_np(x, threshold=20):
    if threshold >= x.max():
        y = x
    else:
        f = (1./threshold) * math.log(threshold)
        y = np.where(x <= threshold, x*f, np.log(x+1e-8))
    return y

def inv_lin_log_np(x, threshold = 20):
    f = (threshold) / math.log(threshold)
    y = np.where(x <= threshold, x * f, np.e**(x))
    return y


def lin_log_grad(L, C = 20):
    threshold = np.log(C)
    f = (1. / threshold) * math.log(threshold)
    y = np.where(L <= threshold, L * f, np.log(L + 1e-9))
    return y



def process_flow_iwe(flow_np, iwe):
    height, width = iwe.shape
    border_mask = np.zeros((height, width))
    border_mask[1:-1, 1:-1] = 1
    def cart2pol(x, y):
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return (rho, phi)

    def pol2cart(rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return (x, y)

    flow_np_x = flow_np[0]
    flow_np_y = flow_np[1]
    flow_np_mag, flow_np_ang = cart2pol(flow_np_x, flow_np_y)
    flow_np_mag = np.where(flow_np_mag > 0, flow_np_mag, 1.0)
    uni_flow = pol2cart(np.ones_like(flow_np_mag), flow_np_ang)
    return uni_flow, iwe/flow_np_mag, border_mask



def remove_hot_pixels(xs, ys, ts, ps, sensor_size=(180, 240), num_hot=50):
    """
    Given a set of events, removes the 'hot' pixel events.
    Accumulates all of the events into an event image and removes
    the 'num_hot' highest value pixels.
    @param xs Event x coords
    @param ys Event y coords
    @param ts Event timestamps
    @param ps Event polarities
    @param sensor_size The size of the event camera sensor
    @param num_hot The number of hot pixels to remove
    """
    img = events_to_image(xs, ys, np.ones_like(ps), sensor_size=sensor_size)
    hot = np.array([]).astype("int")
    for i in range(num_hot):
        maxc = np.unravel_index(np.argmax(np.abs(img)), sensor_size)
        #print("{} = {}".format(maxc, img[maxc]))
        img[maxc] = 0
        h = np.where((xs == maxc[1]) & (ys == maxc[0]))
        hot = np.concatenate((hot, h[0]))
    xs, ys, ts, ps = np.delete(xs.astype(int), hot), np.delete(ys.astype(int), hot), np.delete(ts.astype(int), hot), np.delete(ps.astype(int), hot)
    return xs, ys, ts, ps

def remove_hot_events_v2(E,num_hot):
    for i in range(num_hot):
        maxc = np.unravel_index(np.argmax(np.abs(E)), E.shape)
        # print("{} = {}".format(maxc, img[maxc]))
        E[maxc] = 0
    return E
def remove_hot_events(E,num_hot):
    for i in range(num_hot):
        maxc = np.unravel_index(np.argmax(np.abs(E)), E.shape)
        # print("{} = {}".format(maxc, img[maxc]))
        E[maxc] = 0
    return E

def dct2(mtx):
    return scipy.fftpack.dct(scipy.fftpack.dct(mtx, norm='ortho').T, norm='ortho').T

def idct2(mtx):
    return scipy.fftpack.idct(scipy.fftpack.idct(mtx, norm='ortho').T, norm='ortho').T

def dct_poisson_solver(gx, gy,wid=10):
    # pad size
    gx = np.pad(gx, [(wid, wid), (wid, wid)], 'constant')
    gy = np.pad(gy, [(wid, wid), (wid, wid)], 'constant')
    # define operators in spatial domain
    nabla_x_kern = [[0, 0, 0], [-1, 1, 0], [0, 0, 0]]
    nabla_y_kern = [[0, -1, 0], [0, 1, 0], [0, 0, 0]]
    # specify boundary conditions
    bc = 'mirror'  # use DCT
    # define adjoint operator
    nablaT = lambda gx, gy: scipy.ndimage.convolve(gx, np.rot90(nabla_x_kern, 2), mode=bc) + \
                           scipy.ndimage.convolve(gy, np.rot90(nabla_y_kern, 2), mode=bc)
    # genereate inverse kernel
    H, W = gx.shape
    x_coord, y_coord = np.meshgrid(np.arange(W), np.arange(H))
    mat_x_hat = 2 * np.cos(np.pi * x_coord / W) + 2 * np.cos(np.pi * y_coord / H) - 4
    mat_x_hat[0, 0] = 1
    # do inverse filtering
    irec = idct2(dct2(nablaT(gx, gy)) / -mat_x_hat)
    # get center
    rec = irec[wid+1: -wid, wid+1: -wid]
    #rec = np.delete(rec, (0), 0)
    # rec = np.delete(rec, -1, 0)
    #rec = np.delete(rec, (0), 1)
    # rec = np.delete(rec, -1, 1)
    #rec = np.pad(rec, [(1, 1)], 'edge')
    return rec



def fft_poisson_solver(gx, gy):
    nx, ny = gx.shape
    _yy, _xx = np.meshgrid(np.arange(-ny / 2, ny / 2, 1), np.arange(-nx / 2, nx / 2, 1))
    fx = _xx / nx
    fy = _yy / ny
    dfsx = np.fft.fftshift(np.fft.fft2(-gy))  # FFT sx
    dfsy = np.fft.fftshift(np.fft.fft2(-gx))  # FFT sy
    wf_ft = -1j * (np.sin(fx) * dfsx + np.sin(fy) * dfsy) / (
            2 * np.pi * (np.sin(fx) ** 2 + np.sin(fy) ** 2) + 1e-16)
    wf_ft[int(nx / 2) + 1, int(ny / 2) + 1] = 0.0
    wf_c = np.fft.ifft2(np.fft.fftshift(wf_ft))
    rec = np.real(wf_c)
    return rec  # self.wf=wf

def inverse_lap_solver(lap):
    ny, nx = lap.shape
    _xx, _yy = np.meshgrid( np.arange(-nx / 2, nx / 2, 1),np.arange(-ny / 2, ny / 2, 1))
    fx = _xx / nx
    fy = _yy / ny
    df = np.fft.fftshift(np.fft.fft2(lap))  # FFT sx
    wf_ft = df / (-(2 * np.pi)**2 * (np.sin(fx) ** 2 + np.sin(fy) ** 2) + 1e-16)
    wf_ft[int(nx / 2) + 1, int(ny / 2) + 1] = 0.0
    wf_c = np.fft.ifft2(np.fft.fftshift(wf_ft))
    rec = np.real(wf_c)
    return rec  # self.wf=wf
def rect_aperture(sensor_size, rat = 0.8):
    aperture = np.ones(sensor_size)
    nx, ny =  sensor_size
    _yy, _xx = np.meshgrid( np.arange(-ny / 2, ny / 2, 1)/nx*2,np.arange(-nx / 2, nx / 2, 1)/nx*2)
    aperture[np.sqrt(_yy**2)>rat] = 0
    aperture[np.sqrt(_xx ** 2) >rat] = 0
    return aperture

def tie_laplace_solver(diff_I, I):
    ny, nx = diff_I.shape
    _yy, _xx = np.meshgrid( np.arange(-nx / 2, nx / 2, 1),np.arange(-ny / 2, ny / 2, 1))
    fx = _xx / nx
    fy = _yy / ny
    df = np.fft.fftshift(np.fft.fft2(diff_I))  # FFT sx
    wf_ft = df / (-(2 * np.pi)**2 * (np.sin(fx) ** 2 + np.sin(fy) ** 2) + 1e-16)
    wf_ft[int(nx / 2) + 1, int(ny / 2) + 1] = 0.0
    wf_c = np.fft.ifft2(np.fft.fftshift(wf_ft))
    rec = np.real(wf_c)
    return rec  # self.wf=wf


def calculate_laplace(img):
    return np.gradient(np.gradient(img)[0])[0] + np.gradient(np.gradient(img)[1])[1]

def numpy_shift(img, sx,sy):
    h,w = img.shape
    pad_width = [(np.abs(sy), np.abs(sy)), (np.abs(sx), np.abs(sx))]
    img=np.pad(img, pad_width, 'constant', constant_values=(0, 0))
    img_shift = img[pad_width[0][0]+sy:pad_width[0][0]+sy +h,pad_width[1][0]+sx:pad_width[1][0]+sx+w ]
    return img_shift

def centroid_xy(img):
    moment = cv2.moments(img)
    Xc =  (moment['m10'] / moment['m00'])
    Yc = (moment['m01'] / moment['m00'])
    return Xc, Yc

def center_img(img):
    Xc, Yc = centroid_xy(np.abs(img))
    sx, sy = Xc - img.shape[1] / 2, Yc - img.shape[0] / 2
    img = numpy_shift(img, int(sx), int(sy))
    return img

def compute_dL_torch(uni_flow_np_y, uni_flow_np_x, L,device):
    height, width = L.shape
    fy = cv2.resize(uni_flow_np_y, (width, height))
    fx = cv2.resize(uni_flow_np_x, ( width, height))
    L_torch = torch.from_numpy(L).to(device).unsqueeze(0).unsqueeze(0).to(device)
    gy, gx = torch.gradient(L_torch ,axis = [2,3])
    dL = -(gx.squeeze().cpu().numpy() * fx+gy.squeeze().cpu().numpy() *fy)
    return dL


def save_slice_events(xs, ys, ts, nps, sensor_size, save_path, nz=50):
    Es = np.zeros([sensor_size[0], sensor_size[1], nz])
    win_i = (ts.max()-ts.min())/ nz
    for i in range(nz):
        s = i * win_i
        xsi, ysi, tsi, npsi = time_window(xs, ys, ts, nps, s, win=win_i)
        E_i = events_to_image(xsi, ysi, npsi, sensor_size, polar=False)
        Es[:, :, i] = E_i
        if save_path is not None:
            cv2.imwrite(save_path + "events/" + str(i) + ".png", norm(E_i))
    event_rate = np.sum(Es, (0, 1))
    return event_rate

from scipy.signal import savgol_filter
def auto_focus(ts,nz=400, plot=True):
    nts = ts.copy()
    win_i = (nts.max() - nts.min()) / nz
    event_rate = np.zeros(nz)
    for i in range(nz):
        s = i * win_i + nts.min()
        event_rate[i] = np.where((nts <= s + win_i) & (nts >= s))[0].shape[0]
    event_rate_smooth = savgol_filter(event_rate, window_length=int(15 * 1e6 / (nts.max() - nts.min())), polyorder=2)
    event_rate_grad = np.gradient(event_rate_smooth)
    ### find max:
    loc1 = np.where(event_rate_grad == event_rate_grad.min())[0][0]
    loc2 = np.where(event_rate_grad == event_rate_grad.max())[0][0]
    ### find min:
    temp = np.arange(nz)
    temp2 = np.abs(event_rate_grad[(temp >= loc1) & (temp <= loc2)])
    if len(temp2) == 0:
        loc3 = np.mean([loc1,loc2])
    else:
        loc3 = loc1 + np.mean(np.where(temp2 == temp2.min())[0])
    time_f = (loc3 + 1 / 2) * (nts.max() - nts.min()) / nz + nts.min()
    time_f_ms = time_f / 1e3
    if plot == True:
        plt.figure("event_rate", figsize=(14, 7))
        plt.plot(event_rate_smooth)
        plt.plot(event_rate_grad)
        plt.axvline(x=loc1, color='red', ls='-', lw=2)
        plt.axvline(x=loc2, color='red', ls='-', lw=2)
        plt.axvline(x=loc3, color='green', ls='-', lw=2, label='focal point: ' + str('%.4f' % time_f_ms) + "ms")
        plt.legend()
    return time_f

def total_variation_loss(img, weight=[1, 1]):
    batch_size = img.size(0)
    count_h = img.size(1) * img.size(2) * (img.size(3) - 1)
    count_w = img.size(1) * (img.size(2) - 1) * img.size(3)

    diff_x = img[:, :, 1:, :] - img[:, :, :-1, :]
    diff_y = img[:, :, :, 1:] - img[:, :, :, :-1]

    h_tv = torch.sum(torch.abs(diff_x))
    w_tv = torch.sum(torch.abs(diff_y))

    return (h_tv / count_h * weight[0] + w_tv / count_w * weight[1]) / batch_size

def computeLaplace(img):
    return  np.gradient(np.gradient(img)[0])[0] + np.gradient(np.gradient(img)[1])[1]
def computeLaplace_torch(img):
    return  torch.gradient(torch.gradient(img)[0])[0] + torch.gradient(torch.gradient(img)[1])[1]


##### additional modular: solving a 2D interplation problem through the optimization


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




def warp_events_traj_torch(xt, yt, tt, pt, Dxy, t0=None,
                           batched=False, batch_indices=None):
    """
    Given events and a flow field, warp the events by the flow
    Parameters
    ----------
    xs : list of event x coordinates
    ys : list of event y coordinates
    ts : list of event timestamps
    ps : list of event polarities
    Dxy : 2D   when the flow is just for Dxy:  (1, 2, int(tt.max()) + 1,1)   ## should be (0,1,2,...,t_max)
    t0 : the reference time to warp events to. If empty, will use the
        timestamp of the last event
    Returns
    -------
    warped_xt: x coords of warped events
    warped_yt: y coords of warped events
    """
    if len(xt.shape) > 1:
        xt, yt, tt, pt = xt.squeeze(), yt.squeeze(), tt.squeeze(), pt.squeeze()
    Dxy = Dxy.unsqueeze(-1)
    while len(Dxy.size()) < 4:
        Dxy = Dxy.unsqueeze(0)
    event_indices = torch.transpose(torch.stack((tt, tt), dim=0), 0, 1)
    while len(event_indices.size()) < 4:
        event_indices = event_indices.unsqueeze(0)
    # Event indices need to be between -1 and 1 for F.gridsample, you interpret the event_indices
    event_indices[:, :, :, 0] = event_indices[:, :, :, 0] / (tt.max() - 1) * 2.0 - 1.0
    event_indices[:, :, :, 1] = event_indices[:, :, :, 1] / (tt.max()  - 1) * 2.0 - 1.0

    dxy_at_event = F.grid_sample(Dxy.double(), event_indices.double(), align_corners=True)
    if t0 is None:
        dt_0 = 0
    else:
        dt_0 = int(t0 - tt[0])
    warped_xt = xt - dxy_at_event[:, 0, :, :].squeeze() + Dxy[:, 0, dt_0, :].squeeze()
    warped_yt = yt - dxy_at_event[:, 1, :, :].squeeze() + Dxy[:, 1, dt_0, :].squeeze()
    return warped_xt, warped_yt



def forward_model_EKLT_np(I,Dxy, thre =20):
    L = lin_log_np(I,thre)
    return -forward_grad_np(L)[1] * Dxy[0] - forward_grad_np(L)[0] * Dxy[1]
def EKLT_np(I,Dxy):
    #L = lin_log_np(I,thre)
    L = np.log(1+I)
    return -forward_grad_np(L)[1] * Dxy[0] - forward_grad_np(L)[0] * Dxy[1]
from matplotlib.colors import TwoSlopeNorm
def plot_SR(frames, iwe_all, iwe_pred_lr, iwe_pred_sr,  Dxy, cmap, iwe_cmap, cmap_np, axis_ind, roi, Dxy_ori =np.asarray([[0,0],[0,0]]),  thre =20, save_path = None):
    M = round(iwe_pred_sr.shape[0]/iwe_pred_lr.shape[0])
    frames_grad = forward_model_EKLT_np(frames, Dxy, thre)
    plt.rcParams.update({'font.size': 12})
    fig = plt.figure("Compare SR",figsize=(12, 6))
    plt.subplots_adjust(wspace=0.1, hspace=0.15)
    plt.subplot(231)
    plt.imshow(frames, cmap=cmap)
    #plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlim(roi[0]+Dxy_ori[0]);plt.ylim(roi[1]+Dxy_ori [1]);plt.axis(axis_ind )
    #plt.title('Frame')

    plt.subplot(233)
    plt.imshow(np.abs(frames_grad), cmap=iwe_cmap)
    # plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlim(roi[0] + Dxy_ori[0]);
    plt.ylim(roi[1] + Dxy_ori[1]);
    plt.axis(axis_ind)



    plt.subplot(232)
    plt.imshow(frames_grad, cmap=cmap_np,norm= TwoSlopeNorm(vmin=frames_grad.min(), vcenter=0, vmax=frames_grad.max()))
    #plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlim(roi[0] + Dxy_ori[0]);plt.ylim(roi[1] + Dxy_ori[1]);plt.axis(axis_ind)
    #plt.title('Frame Edge')
    # plt.subplot(233)
    # plt.imshow(np.abs(iwe_pred_lr), cmap=iwe_cmap)
    # plt.colorbar(fraction=0.046, pad=0.04)
    # plt.xlim(roi[0]);plt.ylim(roi[1]);plt.axis(axis_ind )
    # plt.title('Optimal IWE Edge')
    plt.subplot(234)
    plt.imshow(iwe_pred_lr, cmap=cmap_np,norm= TwoSlopeNorm(vmin=iwe_pred_lr.min(), vcenter=0, vmax=iwe_pred_lr.max()))
    #plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlim(roi[0]);plt.ylim(roi[1]);plt.axis(axis_ind )
    #plt.title('IWE')
    # plt.subplot(234)
    # plt.imshow(iwe_0, cmap=cmap_np,norm= TwoSlopeNorm(vmin=iwe_0.min(), vcenter=0, vmax=iwe_0.max()))
    #plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlim(roi[0]);plt.ylim(roi[1]);plt.axis(axis_ind )
    #plt.title('Event Frame')
    plt.subplot(236)
    plt.imshow(np.abs(iwe_pred_sr), cmap=iwe_cmap)
    #plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlim(roi[0]* M);plt.ylim(roi[1]* M);plt.axis(axis_ind )
    #plt.title('SSR Edge')
    plt.subplot(235)
    plt.imshow(iwe_pred_sr, cmap=cmap_np,norm=  TwoSlopeNorm(vmin=iwe_pred_sr.min(), vcenter=0, vmax=iwe_pred_sr.max()))
    #plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlim(roi[0]* M);plt.ylim(roi[1]* M);plt.axis(axis_ind )
    #plt.title('SSR Edge with Polar')
    if save_path is not None:
        fig.savefig(save_path, transparent=True)
        plt.close(fig)

def plot_motion_blur(frames_sta, frames_mov, iwe_0, iwe_all,iwe_pred_lr, iwe_pred_sr, cmap,iwe_cmap,cmap_np,axis_ind, roi,Dxy_ori =np.asarray([[0,0],[0,0]]),  thre =20, save_path = None):
    M = round(iwe_pred_sr.shape[0]/frames_sta.shape[0])
    plt.rcParams.update({'font.size': 12})
    fig = plt.figure("Plot blur",figsize=(12, 5))
    plt.subplots_adjust(wspace=0.1, hspace=0.15)
    plt.subplot(241)
    plt.imshow(frames_sta, cmap=cmap)
    # plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlim(roi[0]+Dxy_ori[0]);plt.ylim(roi[1]+Dxy_ori [1]);plt.axis(axis_ind )
    plt.axis(axis_ind)
    plt.subplot(242)
    plt.imshow(frames_mov, cmap=cmap)
    #plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlim(roi[0]+Dxy_ori[0]);plt.ylim(roi[1]+Dxy_ori [1]);plt.axis(axis_ind )
    #plt.title('Frame')
    plt.subplot(243)
    plt.imshow(np.abs(iwe_pred_lr), cmap=iwe_cmap)
    plt.xlim(roi[0]);
    plt.ylim(roi[1]);
    plt.axis(axis_ind)
    plt.subplot(244)
    plt.imshow(np.abs(iwe_pred_sr), cmap=iwe_cmap)
    plt.xlim(roi[0] * M);
    plt.ylim(roi[1] * M);
    plt.axis(axis_ind)


    plt.subplot(245)
    #plt.imshow(iwe_0, cmap=cmap_np)
    plt.imshow(iwe_0, cmap=cmap_np, norm= TwoSlopeNorm(vmin=iwe_0.min(), vcenter=0, vmax=iwe_0.max()))
    #plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlim(roi[0]);plt.ylim(roi[1]);plt.axis(axis_ind )


    plt.subplot(246)
    plt.imshow(iwe_all, cmap=cmap_np, norm= TwoSlopeNorm(vmin=iwe_all.min(), vcenter=0, vmax=iwe_all.max()))
    #plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlim(roi[0]);plt.ylim(roi[1]);plt.axis(axis_ind )
    plt.subplot(247)
    plt.imshow(iwe_pred_lr, cmap=cmap_np, norm=TwoSlopeNorm(vmin=iwe_pred_lr.min(), vcenter=0, vmax=iwe_pred_lr.max()))
    # plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlim(roi[0]);
    plt.ylim(roi[1]);
    plt.axis(axis_ind)


    plt.subplot(248)
    plt.imshow(iwe_pred_sr, cmap=cmap_np, norm= TwoSlopeNorm(vmin=iwe_pred_sr.min(), vcenter=0, vmax=iwe_pred_sr.max()))
    #plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlim(roi[0] * M); plt.ylim(roi[1] * M);plt.axis(axis_ind)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, transparent=True)
        plt.close(fig)
        plt.close(fig)


def plot_ax_roi(ax, img, cmap = "gray", axis_ind = "off", roi = None, Dxy_ori =np.asarray([[0,0],[0,0]]), M = 1, norm_ind = False):
    if norm_ind == True:
        img = img- np.median(img)
        pos = img[img > 0]
        neg = img[img < 0]

        # 99% 分位（你也可以换 99.5 或 99.9）
        vmax = np.percentile(pos, 99) if pos.size else 1.0
        vmin = np.percentile(neg, 1) if neg.size else -1.0  # 注意：neg是负数，所以用 1% 更接近“最负端”

        norm1 = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        cax = ax.imshow(img, cmap=cmap, norm=norm1)
        #cax  = ax.imshow(img, cmap=cmap, norm=TwoSlopeNorm(vmin=img.min(), vcenter=0, vmax=img.max()))
    else:
        cax = ax.imshow(img, cmap=cmap)
    if roi is None:
        roi = [[0,img.shape[1]],[img.shape[0], 0]]
    ax.set_xlim(np.asarray(roi[0] * M) + np.asarray(Dxy_ori[0]));
    ax.set_ylim(np.asarray(roi[1] * M) + np.asarray(Dxy_ori[1]));
    ax.axis(axis_ind)
    return cax


from matplotlib.colors import TwoSlopeNorm, Normalize
def plot_ax_roi_norm(ax, img, cmap = "gray", axis_ind = "off", roi = None, Dxy_ori =np.asarray([[0,0],[0,0]]), M = 1,_norm= None):
    cax = ax.imshow(img, cmap=cmap, norm=_norm)
    if roi is None:
        roi = [[0,img.shape[1]],[img.shape[0], 0]]
    ax.set_xlim(np.asarray(roi[0] * M) + np.asarray(Dxy_ori[0]));
    ax.set_ylim(np.asarray(roi[1] * M) + np.asarray(Dxy_ori[1]));
    ax.axis(axis_ind)
    return cax
from matplotlib.colors import Normalize, TwoSlopeNorm

def robust_limits(arrays, lo=1, hi=99):
    vals = np.concatenate([np.asarray(a, dtype=np.float32).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(vals, [lo, hi])
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)

def add_roi(ax, roi, M = 1, Dxy_ori = np.asarray([[0,0],[0,0]]), color='red'):
    roi_x = np.asarray(roi[0]) * M + Dxy_ori[0][0]
    roi_y = np.asarray(roi[1]) * M + Dxy_ori[1][0]
    ax.plot([roi_x[0], roi_x[0], roi_x[1], roi_x[1], roi_x[0]],
            [roi_y[0], roi_y[1], roi_y[1], roi_y[0], roi_y[0]],
            color=color, linestyle='--', linewidth=1)
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
def add_colorbars(fig, axes, caxes, num_ticks=5):
    """Add aligned colorbars to multiple axes with modified scientific notation formatting.

    Parameters:
        fig (matplotlib.figure.Figure): The figure object.
        axes (list): List of axes to which the colorbars should be added.
        caxes (list): List of corresponding color axes (images/contours).
        num_ticks (int): Number of ticks on the colorbar.
    """
    for ax, cax in zip(axes, caxes):
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(cax, cax=cbar_ax)

        vmin, vmax = np.min(cax.get_array()), np.max(cax.get_array())
        if vmax == 0:
            vmax = 1
        ticks = np.linspace(vmin, vmax, num_ticks)
        base_exp = int(np.floor(np.log10(max(abs(vmin), abs(vmax)))))  # Determine common exponent
        scale_factor = 10 ** base_exp

        def format_tick(x, _):
            return f'{x / scale_factor:.1f}'  # Show only significant numbers, removing exponent

        cbar.set_ticks(ticks)
        cbar.formatter = ticker.FuncFormatter(format_tick)
        cbar.update_ticks()

        if base_exp != 0:
            # Add exponent label at the top in LaTeX format
            cbar.ax.annotate(f'$\\times 10^{{{base_exp}}}$', xy=(1.0, 1.02), xycoords='axes fraction', ha='right',
                             va='bottom', fontsize=15)

def plot_full(frames_sta, frames_mov, iwe_0, iwe_all, iwe_pred_lr, iwe_pred_sr, Dxy_0, cmap, iwe_cmap, cmap_np, axis_ind, roi, Dxy_ori =np.asarray([[0, 0], [0, 0]]), thre =20, save_path = None):
    M = round(iwe_pred_sr.shape[0]/frames_sta.shape[0])
    # frames_sta_grad = EKLT_np(frames_sta, Dxy_0)
    # frames_mov_grad = EKLT_np(frames_mov, Dxy_0)
    frames_sta_grad = forward_model_EKLT_np(frames_sta, Dxy_0, thre)
    frames_mov_grad = forward_model_EKLT_np(frames_mov, Dxy_0, thre)
    plt.rcParams.update({'font.size': 12})
    fig = plt.figure("Plot deblur and full",figsize=(12, 5))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    ax =  plt.subplot(241)
    plot_ax_roi(ax, frames_sta, cmap, axis_ind,roi, Dxy_ori = Dxy_ori, norm_ind = False)
    ax = plt.subplot(242)
    plot_ax_roi(ax, frames_sta_grad, cmap_np, axis_ind, roi, Dxy_ori = Dxy_ori, M=1,norm_ind = True)
    ax = plt.subplot(243)
    plot_ax_roi(ax, frames_mov,  cmap, axis_ind, roi = roi)
    ax = plt.subplot(244)
    plot_ax_roi(ax, frames_mov_grad, cmap_np, axis_ind, roi, M=1, norm_ind=True)
    ax = plt.subplot(245)
    plot_ax_roi(ax, iwe_0, iwe_cmap, axis_ind, roi, norm_ind=True)
    ax = plt.subplot(246)
    plot_ax_roi(ax, iwe_all, iwe_cmap, axis_ind, roi, norm_ind=True)
    #mask = (np.abs(iwe_all) >0)
    #ax.imshow(mask, cmap='Reds', alpha=0.3)
    ax = plt.subplot(247)
    plot_ax_roi(ax, iwe_pred_lr, iwe_cmap, axis_ind, roi, norm_ind=True)
    ax = plt.subplot(248)
    plot_ax_roi(ax, iwe_pred_sr, iwe_cmap, axis_ind, roi, M=M, norm_ind=True)
    plt.tight_layout()


    if save_path is not None:
        fig.savefig(save_path, transparent=True)
        plt.close(fig)
def compute_c(I_recons,I_raw):
    return(I_recons.flatten() * I_raw.flatten()).sum() / (I_recons.flatten() * I_recons.flatten()).sum()




def plot_events(iwe_0, iwe_all, iwe_pred_lr, iwe_pred_sr,cmap_np,axis_ind, roi, save_path = None):
    M = round(iwe_pred_sr.shape[0]/ iwe_pred_lr.shape[0])
    # frames_sta_grad = EKLT_np(frames_sta, Dxy_0)
    # frames_mov_grad = EKLT_np(frames_mov, Dxy_0)
    plt.rcParams.update({'font.size': 12})
    fig = plt.figure("Plot events",figsize=(12, 5))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    plt.subplot(141)
    # plt.imshow(iwe_0, cmap=cmap_np)
    plt.imshow(iwe_0, cmap=cmap_np, norm=TwoSlopeNorm(vmin=iwe_0.min(), vcenter=0, vmax=iwe_0.max()))
    # plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlim(roi[0]);
    plt.ylim(roi[1]);
    plt.axis(axis_ind)

    plt.subplot(142)
    plt.imshow(iwe_all, cmap=cmap_np, norm=TwoSlopeNorm(vmin=iwe_all.min(), vcenter=0, vmax=iwe_all.max()))
    # plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlim(roi[0]);
    plt.ylim(roi[1]);
    plt.axis(axis_ind)

    plt.subplot(143)
    plt.imshow(iwe_pred_lr, cmap=cmap_np, norm=TwoSlopeNorm(vmin=iwe_pred_lr.min(), vcenter=0, vmax=iwe_pred_lr.max()))
    # plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlim(roi[0]);
    plt.ylim(roi[1]);
    plt.axis(axis_ind)

    plt.subplot(144)
    plt.imshow(iwe_pred_sr, cmap=cmap_np, norm= TwoSlopeNorm(vmin=iwe_pred_sr.min(), vcenter=0, vmax=iwe_pred_sr.max()))
    #plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlim(roi[0] * M); plt.ylim(roi[1] * M);plt.axis(axis_ind)
    plt.tight_layout()


    if save_path is not None:
        fig.savefig(save_path, transparent=True)
        plt.close(fig)


### This is to simulate the blurry image if no motion-blur is seen in the scene

def generate_motion_blur_kernel_Dxy(Dxy, device, kernel_size=21):
    t_win = Dxy.shape[1]
    Dx, Dy = Dxy  ### should be in the same size of dt, and dt is fixed?     1, n
    path_length = torch.norm(Dxy.abs().max(1)[0], 2)
    if kernel_size <= 2* int(path_length)+1:
        kernel_size = 2* int(path_length)+1
    # Center of the kernel
    center = kernel_size // 2
    # Vectorized time steps
    x_cont = center + Dx
    y_cont = center + Dy
    kernel = events_to_image_torch(x_cont, y_cont, torch.ones_like(x_cont),(kernel_size,kernel_size), device = device )
    #path_length = torch.sqrt((Dx) ** 2 + (Dy) ** 2)
    kernel /= t_win #path_length
    return kernel



def generate_motion_blur_kernel_Dxy_sr(Dxy, M, device, kernel_size=21, sigma  =  0.5, t0 = 0):
    t_win = Dxy.shape[1]
    Dx, Dy = Dxy  ### should be in the same size of dt, and dt is fixed?     1, n
    path_length = torch.norm(Dxy.abs().max(1)[0], 2)
    if kernel_size <= 2* int(path_length)*M+1:
        kernel_size = 2* int(path_length)*M+1
    img_size = (kernel_size,kernel_size)
    img_size_sr = np.asarray([int(img_size[0]), int(img_size[1])])
    img_sr = torch.zeros(img_size_sr[0], img_size_sr[1]).to(device).float()
    # Center of the kernel
    center = img_size_sr// 2
    # Vectorized time steps
    #dt = torch.arange(t_win, dtype=torch.float32).to(device) / t_win
    if t0 is None:
        dt_0 = 0
    else:
        dt_0 = int(t0)
    xt = center[1] + (Dx - Dx[dt_0])* M
    yt = center[0] + (Dy-  Dy[dt_0])* M
    pxt = (xt.floor()).float()
    pyt = (yt.floor()).float()
    dxt = (xt - pxt).float()
    dyt = (yt - pyt).float()
    pxt = pxt.long()
    pyt = pyt.long()
    interp_weights = torch.ones_like(xt)
    kernel_sz = M + (M + 1) % 2 + 2
    kernel = gaussian_interpolate_to_image(pxt, pyt, dxt, dyt, interp_weights, img_sr,sigma, kernel_sz)   ###
    #path_length = torch.sqrt((Dx) ** 2 + (Dy) ** 2)
    kernel /= kernel.sum() #path_length
    return kernel.float()

def generate_motion_blur_kernel_Dxy_pc(
    Dxy_pc, M, device,
    kernel_size=21, sigma=0.5,
    t_s=0.0, t_e=None,     # 积分区间
    t0=None,               # reference time
    t_win=None,            # 总时间窗口 (μs)
    samples_per_seg=8
):
    """
    Motion-blur kernel using piecewise displacement (μs time domain).
    Integral over [t_s, t_e], aligned to reference position r(t0).

    Args:
        Dxy_pc: (K, 2) piecewise displacements Δr_k (pixels)
        M: scale factor to HR grid
        t_s, t_e: integration start/end in μs
        t0: reference time in μs (kernel center)
        t_win: total exposure window in μs
    """

    # ---------- Basic ----------
    Dxy_pc = Dxy_pc.to(device).float()
    K = Dxy_pc.shape[0]

    if t_e is None:  t_e = t_win
    if t0 is None:   t0 = (t_s + t_e) * 0.5

    assert 0 <= t_s < t_e <= t_win
    assert 0 <= t0 <= t_win

    # ---------- Segment time boundaries ----------
    t_nodes = torch.linspace(0, t_win, K + 1, device=device)  # (K+1,)
    widths = t_nodes[1:] - t_nodes[:-1]

    # ---------- Cumulative displacements ----------
    r_cum = torch.cat([
        torch.zeros((1, 2), device=device),
        torch.cumsum(Dxy_pc, dim=0)
    ], dim=0)  # (K+1, 2)

    # ---------- r(t) analytic ----------
    def r_at(t):
        idx = torch.searchsorted(t_nodes, t) - 1
        idx = idx.clamp(0, K - 1)
        t0k, t1k = t_nodes[idx], t_nodes[idx + 1]
        alpha = (t - t0k) / (t1k - t0k)
        return r_cum[idx] + alpha * Dxy_pc[idx]

    # Reference position (kernel center)
    r_t0 = r_at(t0)

    # ---------- Dense samples inside [t_s, t_e] ----------
    coords = []
    for k in range(K):
        seg_t0, seg_t1 = t_nodes[k], t_nodes[k + 1]

        if seg_t1 <= t_s or seg_t0 >= t_e:
            continue

        tk0 = max(seg_t0, t_s)
        tk1 = min(seg_t1, t_e)

        seg_t = torch.linspace(tk0, tk1, samples_per_seg, device=device)
        v_k = Dxy_pc[k] / (seg_t1 - seg_t0)

        r_seg = r_cum[k] + (seg_t - seg_t0).unsqueeze(1) * v_k

        # center kernel at r(t0)
        coords.append(r_seg - r_t0)

    coords = torch.cat(coords, 0)

    # ---------- kernel size ----------
    max_abs = coords.abs().max(0)[0]
    path_len = torch.norm(max_abs, 2).item()


    ks = max(kernel_size, int(2 * path_len * M + 1 + path_len))
    img_sr = torch.zeros((ks, ks), device=device)
    center = torch.tensor([ks // 2, ks // 2], device=device)

    # ---------- map to plane ----------
    xt = center[1] + coords[:, 0] * M
    yt = center[0] + coords[:, 1] * M

    pxt = xt.floor().long()
    pyt = yt.floor().long()
    dxt = xt - pxt
    dyt = yt - pyt
    w = torch.ones_like(xt)

    kernel_sz = M + (M + 1) % 2 + 2
    interp_weights = torch.ones_like(pxt)
    kernel =  interpolate_to_image(pxt, pyt, dxt, dyt, interp_weights, img_sr)

    #
    # kernel = gaussian_interpolate_to_image(
    #     pxt, pyt, dxt, dyt, w, img_sr, sigma, kernel_sz
    # )

    kernel /= kernel.sum()
    return kernel.float()
def generate_motion_blur_kernel_Dxy_pc_fixed(
    Dxy_pc,                  # (K, 2)
    M,                       # SR factor
    device,
    kernel_size=21,
    sigma=0.5,

    # 时间区间（单位 μs）
    t_s=0.0,
    t_e=None,
    t0=None,
    t_win=None,

    # 采样密度（你要求 100）
    samples_per_seg=100,
):
    """
    高质量 PC → blur kernel，完全等价于 dense version（采样密度 Ns = 100）。

    Args:
        Dxy_pc: (K, 2) piecewise displacements
        M: SR factor
        t_s, t_e: integration region
        t0: kernel center (μs)
        t_win: full motion window (μs)
    """
    Dxy_pc = Dxy_pc.to(device).float()
    K = Dxy_pc.shape[0]

    if t_e is None:  t_e = t_win
    if t0 is None:   t0 = 0.0

    # --- segment boundaries in physical time ---
    t_nodes = torch.linspace(0, t_win, K + 1, device=device)   # (K+1,)
    widths  = t_nodes[1:] - t_nodes[:-1]                       # per-segment durations

    # --- cumulative positions r_k ---
    r_cum = torch.cat([
        torch.zeros((1, 2), device=device),
        torch.cumsum(Dxy_pc, dim=0)
    ], dim=0)   # (K+1, 2)

    # -------------------------------------------------------------
    # Helper: analytic r(t)
    # -------------------------------------------------------------
    def r_at(t):
        """返回轨迹 r(t) for any t in [0, t_win]."""
        idx = torch.searchsorted(t_nodes, t) - 1
        idx = idx.clamp(0, K - 1)

        t0k = t_nodes[idx]
        t1k = t_nodes[idx + 1]
        alpha = (t - t0k) / (t1k - t0k)
        return r_cum[idx] + alpha * Dxy_pc[idx]

    # reference center
    r_t0 = r_at(torch.tensor(t0, device=device))

    # -------------------------------------------------------------
    # uniformly sample samples_per_seg physical timestamps
    # -------------------------------------------------------------
    t_dense = torch.linspace(t_s, t_e, samples_per_seg, device=device)
    r_dense = torch.stack([r_at(ti) for ti in t_dense], dim=0)  # (N, 2)
    coords = r_dense - r_t0                                    # center alignment

    # -------------------------------------------------------------
    # build kernel
    # -------------------------------------------------------------
    max_abs = coords.abs().max(0)[0]
    path_len = torch.norm(max_abs, 2).item()

    ks = max(kernel_size, int(2 * path_len * M + 1))
    img = torch.zeros((ks, ks), device=device)
    center = torch.tensor([ks // 2, ks // 2], device=device)

    # project coords
    xt = center[1] + coords[:, 0] * M
    yt = center[0] + coords[:, 1] * M

    pxt = xt.floor().long()
    pyt = yt.floor().long()
    dxt = xt - pxt
    dyt = yt - pyt
    w = torch.ones_like(xt)

    kernel_sz = M + (M + 1) % 2 + 2

    kernel = gaussian_interpolate_to_image(
        pxt, pyt, dxt, dyt, w, img, sigma, kernel_sz
    )

    kernel /= kernel.sum()
    return kernel.float()

#
# def slice_slanted_column(img, start, end, M = 1):
#     x_start, y_start = start * M
#     x_end, y_end = end * M
#     # Compute the number of points along the line (based on vertical or horizontal distance)
#     num_points = int(np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)) + 1
#     # Interpolate x and y coordinates along the line
#     x_coords = np.linspace(x_start, x_end, num_points)
#     x_coords = (x_coords).astype(int)
#     y_coords = np.linspace(y_start, y_end, num_points)
#     y_coords = (y_coords).astype(int)
#     r_coords =np.linspace(0, np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2), num_points)
#     # Extract the pixel values along the slanted line
#     slit_values = img[y_coords, x_coords]
#     return slit_values,  r_coords

from scipy.ndimage import map_coordinates
#
# def slice_slanted_column(img, start, end, M=1, order=1):
#     """
#     order=1: bilinear (ImageJ 默认)
#     order=3: bicubic
#     """
#     x0, y0 = np.array(start) * M
#     x1, y1 = np.array(end) * M
#
#     length = np.hypot(x1 - x0, y1 - y0)
#     num_points = int(length) + 1
#
#     x = np.linspace(x0, x1, num_points)
#     y = np.linspace(y0, y1, num_points)
#
#     # 注意：map_coordinates 用的是 (row, col) = (y, x)
#     values = map_coordinates(
#         img,
#         [y, x],
#         order=order,
#         mode='nearest'
#     )
#
#     r = np.linspace(0, length / M, num_points)
#     return values, r


def slice_slanted_column(
    img, start, end, line_width=3, M=1, order=1
):
    """
    ImageJ-like line profile:
    - subpixel sampling
    - bilinear interpolation
    - averaging over line width (normal direction)
    """
    x0, y0 = np.array(start) * M
    x1, y1 = np.array(end) * M

    length = np.hypot(x1 - x0, y1 - y0)
    num = int(length) + 1

    t = np.linspace(0, 1, num)
    x = x0 + t * (x1 - x0)
    y = y0 + t * (y1 - y0)

    # unit normal
    dx, dy = x1 - x0, y1 - y0
    n = np.hypot(dx, dy)
    nx, ny = -dy / n, dx / n

    offsets = np.linspace(
        -(line_width - 1) / 2,
        +(line_width - 1) / 2,
        line_width
    )

    profiles = []
    for o in offsets:
        profiles.append(
            map_coordinates(
                img,
                [y + o * ny, x + o * nx],
                order=order,        # order=1 → bilinear
                mode='nearest'
            )
        )

    values = np.mean(profiles, axis=0)
    r = np.linspace(0, length / M, num)

    return values, r
## auto determine bc
"""
def auto_bc(I_gt = I_sr, E_gt =  iwe_gt_sr):
    #### Optional, why not define b and c at the very first
    thre_pred = nn.Parameter(torch.tensor(1.001).float().to(device), requires_grad=True)
    thre_pred_c = nn.Parameter(torch.tensor(0.3).float().to(device), requires_grad=True)
    lr = [1e-2];
    loss_hist = []
    vars = [];
    vars += [{'params': thre_pred, 'lr': lr[0]}];
    vars += [{'params': thre_pred_c, 'lr': lr[0]}]
    optimizer = AdamP(vars);
    scheduler = lr_scheduler.StepLR(optimizer, step_size=200 // 2, gamma=0.6, verbose=False);
    sub_win = win_frame
    
    mask_e = (E_gt.abs() >= 0.1)
    for iter in range(1100):
        optimizer.zero_grad()  # Essential for update the derivatives
        L_gt = lin_log(I_gt, thre_pred)
        E_pred = forward_model_EKLT_L(L_gt, Dxy_pc.sum(0))
        L_E = l2_loss(E_pred[mask_e] - E_gt[mask_e] * thre_pred_c)
        # L_E = l2_loss(norm_torch(E_pred,1)[mask_e] - norm_torch(E_gt,1)[mask_e])
        loss_hist.append(L_E.cpu().detach())
        L_E.backward()  # Calculate the derivatives
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            # I_pred.clamp_(0, 255)
            thre_pred.clamp_(1.001, 255)
            thre_pred_c.clamp_(0.05, 20)
        if iter % 100 == 0:
            print("iter = {}: loss = {}, d_phi={}".format(iter, L_E.data.cpu().numpy(),
                                                          thre_pred_c.grad.mean().cpu().numpy()))
            # print("iter = {}: loss = {}, d_phi={}".format(iter, L_tot.data.cpu().numpy(), I_pred.grad.mean().cpu().numpy()))

    plt.rcParams.update({'font.size': 15})
    fig = plt.figure("pred bc", figsize=(12, 5))
    ax = plt.subplot(141)
    ax.plot(loss_hist)
    ax.set_title('Loss vs. iterations')
    # ax.set_xlabel('iterations')
    ax.set_ylabel(r'$|I- I^\prime|_2^2$')
    ax.set_aspect(loss_hist.__len__() / (np.asarray(loss_hist).max() - np.asarray(loss_hist).min()) * ax_asp)
    ax.axis('off')
    print("thre_b = ", thre_pred.data.cpu().numpy(), "thre_c= ", thre_pred_c.data.cpu().numpy(), );

    ax = plt.subplot(142)
    plot_ax_roi(ax, E_gt.detach().cpu() * thre_pred_c.detach().cpu(), cmap, axis_ind, roi_0, norm_ind=False)
    ax.set_title("Recorded IWE = \int e_k'")
    ax = plt.subplot(143)
    plot_ax_roi(ax, E_pred.detach().cpu(), cmap, axis_ind, roi_0, norm_ind=False)
    ax.set_title('Pred: grad L = IWE * c')
    ax = plt.subplot(144)
    plot_ax_roi(ax, E_pred.detach().cpu() - E_gt.detach().cpu() * thre_pred_c.detach().cpu(), cmap, axis_ind, roi_0,
                norm_ind=False)
    ax.set_title('Difference')
    return thre_pred.data.cpu().numpy(), thre_pred_c.data.cpu().numpy()
"""


def fourier_upsampling(image: torch.Tensor, upsampling_factor: int) -> torch.Tensor:
    """
    Perform Fourier upsampling on the input image.

    Args:
        image (torch.Tensor): The input image of shape (H, W) or (N, H, W) for batch processing.
        upsampling_factor (int): The factor by which to upsample the image.

    Returns:
        torch.Tensor: The upsampled image after Fourier upsampling.
    """
    if image.dim() == 2:
        image = image.unsqueeze(0)  # Add batch dimension if necessary

    # Get the original size
    original_height, original_width = image.shape[-2], image.shape[-1]

    # Perform the Fourier transform
    fft_image = torch.fft.fft2(image)

    # Shift the zero-frequency component to the center
    fft_image = torch.fft.fftshift(fft_image, dim=(-2, -1))

    # Pad the Fourier transform to upsample
    pad_height = (original_height * upsampling_factor - original_height) // 2
    pad_width = (original_width * upsampling_factor - original_width) // 2
    padded_fft = torch.nn.functional.pad(fft_image, (pad_width, pad_width, pad_height, pad_height))

    # Shift the zero-frequency component back
    padded_fft = torch.fft.ifftshift(padded_fft, dim=(-2, -1))

    # Perform the inverse Fourier transform
    upsampled_image = torch.fft.ifft2(padded_fft).real *upsampling_factor**2

    return upsampled_image.squeeze(0) if image.size(0) == 1 else upsampled_image

def F_upsampling(image: torch.Tensor, upsampling_factor: int, mode = "bicubic", align_corners= True) -> torch.Tensor:
    Ny , Nx = image.shape
    image_sr  = F.interpolate(image.detach().unsqueeze(0).unsqueeze(0), (int(Ny * upsampling_factor), int(Nx * upsampling_factor)), mode = mode, align_corners = align_corners).squeeze()
    return image_sr

#
# def roi_xy_to_mask(roi, sensor_size):
#     mask =  np.zeros(sensor_size, dtype=bool)
#     xs, xe  = roi[0,:]
#     ys, ye  = roi[1,:]
#     mask[ys:ye, xs:xe] = True
#     return mask



def plot_zoom(ax, img_zoom, cmap, invert = False):
    axins_roi = (0, 1 - 1 / 3, 1 / 3, 1 / 3);
    axins_c = 'white';
    axins = ax.inset_axes(axins_roi)
    axins.imshow(img_zoom, cmap)
    axins.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
    axins.spines['bottom'].set_color(axins_c)
    axins.spines['top'].set_color(axins_c)
    axins.spines['right'].set_color(axins_c)
    axins.spines['left'].set_color(axins_c)
    if invert:
        axins.invert_yaxis()
def my_fft(u, flag = 0):
    if flag == 0:
        return torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(u)))
    else:
        return torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(u)))
"""
def convolve_with_fft_replicate(image, kernel,pad_mode='reflect'):
    #prepare padding image
    ny, nx = image.shape
    nky, nkx = kernel.shape
    p2d = (int(nkx / 2), int(nkx / 2), int(nky / 2), int(nky / 2))  # pad last dim and 2nd to last
    if pad_mode == 'lap':
        raise ValueError("Padding mode 'lap' is not supported.")
    else:
        image_pad = F.pad(image.unsqueeze(0).unsqueeze(0), p2d, pad_mode).squeeze()  # pad A
    nyp, nxp = image_pad.shape
    #prepare kernel
    nyk, nxk = kernel.shape
    pad_height = nyp-nyk
    pad_width =  nxp-nxk
    if pad_height < 0 or pad_width < 0:
        raise ValueError("Kernel must be smaller than or equal to the image size.")
    # Calculate padding values: (left, right, top, bottom)
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    padded_kernel = F.pad(kernel.unsqueeze(0).unsqueeze(0), (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0).squeeze()
    f_out = my_fft(my_fft(image_pad) * my_fft(padded_kernel),1).real
    return f_out[int(nky / 2):(ny + int(nky / 2)), int(nkx / 2):(nx + int(nkx / 2))]
"""

def convolve_with_fft_replicate(image, kernel, pad_mode="replicate"):
    """
    image: [H, W]
    kernel: [kH, kW]
    pad_mode: "replicate", "reflect", "constant"
    returns: same-size convolution (identical to conv2d + pad_mode)
    """

    H, W = image.shape
    kH, kW = kernel.shape

    pad_y = kH // 2
    pad_x = kW // 2

    # ---- 1. Spatial padding identical to conv2d behaviour ----
    image_pad = F.pad(image[None, None],
                      (pad_x, pad_x, pad_y, pad_y),
                      mode=pad_mode)[0, 0]

    # ---- 2. fftshift kernel centre ----
    kernel_shift = torch.fft.ifftshift(kernel)

    # ---- 3. pad kernel to same size as padded image ----
    Hp, Wp = image_pad.shape
    kernel_pad = torch.zeros_like(image_pad)
    kernel_pad[:kH, :kW] = kernel_shift

    # ---- 4. FFT convolution ----
    F_img = torch.fft.fft2(image_pad)
    F_ker = torch.fft.fft2(kernel_pad)
    out_full = torch.fft.ifft2(F_img * F_ker).real

    # ---- 5. crop valid region ----
    out = out_full[pad_y:pad_y + H, pad_x:pad_x + W]

    return out
