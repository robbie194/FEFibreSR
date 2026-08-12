import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils.utility import norm
from matplotlib.animation import FuncAnimation, PillowWriter
from function.my_colormap import flow_vectors_to_rgb


def save_image(image, save_path, name):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(save_path, name), image)


def rotate_crop_image(img, ang, center=None, crop_size=None, flip = False):
    '''Rotate the image according to given rotation Angle'''
    rows = img.shape[0]
    cols = img.shape[1]
    if center is None:
        center = (cols  // 2,  rows // 2)
    rotate = cv2.getRotationMatrix2D( center, ang, 1)
    img = cv2.warpAffine(img, rotate,
                                   (cols,rows) ) # get the referenced SH-pattern just after the rotation
    if crop_size is not None:
        img = img[center[1] - crop_size[1] // 2: center[1] + crop_size[1] // 2, center[0] - crop_size[0]//2: center[0] + crop_size[0] // 2]
    if flip == True:
        img = cv2.flip(img, flipCode=1)
    return img


def pre_process(image_array, rotation_angle, x_start, y_start, roi_length, pitch_px, noise_threshold = 0.005, threshold = 0):
    '''Rotate the image according to given rotation Angle'''
    rows = image_array.shape[0]
    cols = image_array.shape[1]
    # read the rows and cols
    rotate = cv2.getRotationMatrix2D((rows//2, cols//2), rotation_angle, 1)  # get the rotation matrix
    image_rotate = cv2.warpAffine(image_array, rotate, (cols, rows))  # get the referenced SH-pattern just after the rotation
    '''crop'''
    x_end = x_start + roi_length * pitch_px
    y_end = y_start + roi_length * pitch_px
    image_crop = image_rotate[int(y_start):int(y_end), int(x_start):int(x_end)]
    return image_crop






def plot_scatter_roi(xs, ys, ts, ps, roi = np.asarray([[0,346],[0,260]]),polar = True, axis_ind  = "on", alpha = 0.2,alpha_k=0.02, plot = True, save = None):
    roi_mask = (xs >= roi[0][0]) & (xs <= roi[0][1]) & (ys >= roi[1][0]) & (ys <= roi[1][1])
    pos_loc = (ps == 1) & roi_mask
    neg_loc = (ps == 0) & roi_mask
    pos_xs, pos_ys, pos_ts = xs[pos_loc], ys[pos_loc], ts[pos_loc]
    neg_xs, neg_ys, neg_ts = xs[neg_loc], ys[neg_loc], ts[neg_loc]
    fig = plt.figure("Spatio-temporal stream of SHWFS side", figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    if polar == True:
        ax.scatter(alpha=alpha, xs=pos_xs, ys=pos_ys, zs=pos_ts/1e6, c='r', marker=".", clip_on=True)
        ax.scatter(alpha=alpha, xs=neg_xs, ys=neg_ys, zs=neg_ts/1e6,  c='b', marker=".", clip_on=True)
    ax.scatter(xs=xs, alpha=alpha_k, zs=0, ys= ys, marker=".", c='k')  #ys=(ts.max())/1e6/2
    ax.set_xlim3d(roi[0][0],roi[0][1])
    ax.set_ylim3d(roi[1][0],roi[1][1])
    ax.set_zlim3d(ts.min()/1e6, (ts.max())/1e6 )
    ax.set_box_aspect([roi[1][:].max()/roi[0][:].max(), 1.0, 1.0])
    ax.view_init(azim=66, elev=18)  # 倾斜normal视图
    plt.axis(axis_ind)
    if plot == False:
        plt.close(fig)
    if save is not None:
        fig.savefig(save, transparent=True)


def plot_surface_event_roi(iwe, roi,M, cmap=plt.cm.gray, save=None):
    """
    Plot 3D surface of IWE within ROI.

    Args:
        iwe (2D array): Image of Warped Events.
        roi (2x2 array): [[x_min, x_max], [y_min, y_max]] in pixel coordinates.
        cmap: matplotlib colormap, e.g. plt.cm.viridis
    """
    # Normalize
    iwe = iwe / np.abs(iwe).max()  # norm(iwe, 1)
    # Crop to ROI
    x0, x1 = int(roi[0][0])*M, int(roi[0][1]*M)
    y0, y1 = int(roi[1][0])*M, int(roi[1][1]*M)
    iwe_roi = iwe[y0:y1, x0:x1]
    sr = iwe_roi.shape
    xx, yy = np.meshgrid(np.linspace(x0, x1, sr[1]), np.linspace(y0, y1, sr[0]))
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(121)
    ax.imshow(iwe_roi, cmap = "gray")
    ax.set_aspect(roi[1][:].max() / roi[0][:].max())
    plt.axis("off")
    ax = fig.add_subplot(122, projection='3d')
    #iwe_roi[iwe_roi == 0] = np.nan  # Set zero values to NaN for better visualization
    facecolors = cmap(iwe_roi)
    ax.plot_surface(xx, yy, iwe_roi, rstride=1, cstride=1, facecolors=facecolors, shade=False)
    # ax.set_xlim3d(roi[0][0], roi[0][1])
    # ax.set_ylim3d(roi[1][0], roi[1][1])
    ax.set_zlim3d(0, 1)
    ax.set_box_aspect([roi[1][:].max() / roi[0][:].max(), 1.0, 1.0])
    ax.view_init(azim=45, elev=60)
    plt.tight_layout()
    plt.show()
    if save is not None:
        fig.savefig(save, transparent=True)

if __name__ == '__main__':
    img_name = "pnp_pred_blur_9_nf_0.3"
    path = r"E:\\[X] NeuroSR\\fig\\result1\\"
    img =  norm(np.load(path + img_name + '.npy'));  M =round(img.shape[0]/260)
    cmap = "gray"
    center = (243, 123)
    crop = [220,  220]
    img = rotate_crop_image(img, -55,center =(center[0]*M, center[1]*M), crop_size= [crop[0]*M,crop[1]*M], flip=True)
    plt.imshow(img, cmap)
    # os.mkdir( path + "sub")
    #cv2.imwrite(path +"sub\\"+img_name +"_"+  str(crop[0])+".png", norm(img))


    ###  Read a lot of pics
    path = r"E:\\[X] NeuroSR\\fig\\compare_DH\\rest_2\\"
    a_es = []; a_fs  = []; phi_es = []; phi_fs = []
    for i in range(25):
        a_es.append(cv2.imread(path +"a_e\\"+ str(i) +".png", 0))
        phi_es.append(cv2.imread(path +"phi_e\\"+ str(i) + ".png", 0))
        a_fs.append(cv2.imread(path +"a_f\\" + str(i) + ".png", 0))
        phi_fs.append(cv2.imread(path + "phi_f\\" + str(i) + ".png", 0))
    a_es = np.asarray(a_es); a_fs = np.asarray(a_fs); phi_es = np.asarray(phi_es); phi_fs = np.asarray(phi_fs)
    N_t,H,W = a_es.shape

    M =a_es.shape[1]//260

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    fig.patch.set_alpha(0)
    f_range = np.arange(0, 25)
    num_t_ref =  25
    i = 0
    roi_h = roi_0
    def update(i):
        plt.rcParams.update({'font.size': 15})
        plt.subplots_adjust(wspace=0.4, hspace=0)
        # left panel
        ax = axes[0]
        ax.clear()
        plot_ax_roi(ax, a_fs[i], cmap, axis_ind, roi=roi_h, Dxy_ori=Dxy_ori, M=1)
        #         # middle panel
        ax = axes[1]
        ax.clear()
        plot_ax_roi(ax, a_es[i], cmap, axis_ind, roi=roi_h, Dxy_ori=Dxy_ori, M=M)
        # right panel
        ax = axes[2]
        ax.clear()
        plot_ax_roi(ax,phi_fs[i],cmap_phi, axis_ind, roi = roi_h, Dxy_ori = Dxy_ori, M = 1, norm_ind =  True)
        ax = axes[3]
        ax.clear()
        plot_ax_roi(ax,phi_es[i],cmap_phi, axis_ind, roi = roi_h, Dxy_ori = Dxy_ori, M = M, norm_ind =  True)
        plt.tight_layout()
        return []
    ani = animation.FuncAnimation(fig, update, f_range, interval=100, blit=False)
    ani.save('E:/[X] NeuroSR/vid/visual_norm_all.gif', fps=15)


    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    fig.patch.set_alpha(0)
    f_range = np.arange(0, 25)
    num_t_ref =  25
    tss = np.linspace(ts[0], ts[-1], num_t_ref)
    def update(i):
        plt.rcParams.update({'font.size': 15})
        plt.subplots_adjust(wspace=0.4, hspace=0)
        # left panel
        ax = axes[0]
        ax.clear()
        plot_ax_roi(ax, frames_sharp, cmap, axis_ind, Dxy_ori=Dxy_roi,
                    M=1, roi=roi_zoom, norm_ind=False)
        # middle panel
        ax = axes[1]
        ax.clear()
        start_i =  tss[i] +s
        loc_i = np.where(time_img[:, 0] > start_i)[0][0] - 1
        frames_i = frames_img[loc_i , :, :]
        plot_ax_roi(ax, frames_i, cmap, axis_ind,
                    roi=roi_zoom, M=1, norm_ind=False)
        # right panel
        ax = axes[2]
        ax.clear()
        plot_ax_roi(ax, iwes[i, ...], cmap_np, axis_ind,
                    roi=roi_zoom, M=1, norm_ind=True)
        ax = axes[3]
        ax.clear()
        plot_ax_roi(ax, iwes_sr[i, ...], cmap_np, axis_ind, roi=roi_zoom, M=M, norm_ind=False)
        plt.tight_layout()
        return []
    ani = animation.FuncAnimation(fig, update, f_range, interval=100, blit=False)
    ani.save('E:/[X] NeuroSR/vid/visual.gif', fps=15)








    iter_num = vid_num
    f_range = np.arange(0, iter_num)
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    fig.patch.set_alpha(0)  # 画布透明
    plt.rcParams.update({'font.size': 15})

    def update(i):
        plt.gca().set_aspect('equal', adjustable='box')
        plt.subplots_adjust(wspace=0.4, hspace=0)
        ax = plt.subplot(221)
        ax.clear()
        plot_ax_roi(axes[0,0], a_preds[i], cmap, axis_ind, roi=roi_0, Dxy_ori=Dxy_ori, M=M)
        plt.gca().set_facecolor('none')
        for spine in plt.gca().spines.values():
            spine.set_color('white')
        plt.tick_params(colors='white')
        plt.axis("off")
        ax = plt.subplot(222)
        ax.clear()
        plot_ax_roi(axes[0,1], phi_preds[i], cmap_phi, axis_ind, roi=roi_0, Dxy_ori=Dxy_ori, M=M)
        plt.gca().set_facecolor('none')
        for spine in plt.gca().spines.values():
            spine.set_color('white')
        plt.tick_params(colors='white')
        plt.axis("off")
        ax = plt.subplot(223)
        ax.clear()
        plot_ax_roi(axes[1,0], a_preds[i], cmap, axis_ind, roi=roi_zoom, Dxy_ori=Dxy_ori, M=M)
        plt.gca().set_facecolor('none')
        for spine in plt.gca().spines.values():
            spine.set_color('white')
        plt.tick_params(colors='white')
        plt.axis("off")
        ax = plt.subplot(224)
        ax.clear()
        plot_ax_roi(axes[1,1], phi_preds[i], cmap_phi, axis_ind, roi=roi_zoom, Dxy_ori=Dxy_ori, M=M)
        plt.gca().set_facecolor('none')
        for spine in plt.gca().spines.values():
            spine.set_color('white')
        plt.tick_params(colors='white')
        plt.axis("off")
        plt.tight_layout()
        return []
    ani = animation.FuncAnimation(fig, update, f_range, interval=100, blit=False)
    ani.save('E:/[X] NeuroSR/vid/physics_twin_jing_refine_e.gif', fps=15)










"""
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    fig.patch.set_alpha(0)
    f_range = np.arange(0, 25)
    num_t_ref =  25
    tss = np.linspace(ts[0], ts[-1], num_t_ref)
    def update(i):
        plt.rcParams.update({'font.size': 15})
        plt.subplots_adjust(wspace=0.4, hspace=0)
        # left panel
        ax = axes[0]
        ax.clear()
        plot_ax_roi(ax, frames_sharp, cmap, axis_ind, Dxy_ori=Dxy_roi,
                    M=1, roi=roi_zoom, norm_ind=False)
        # middle panel
        ax = axes[1]
        ax.clear()
        start_i =  tss[i] +s
        loc_i = np.where(time_img[:, 0] > start_i)[0][0] - 1
        frames_i = frames_img[loc_i , :, :]
        plot_ax_roi(ax, frames_i, cmap, axis_ind,
                    roi=roi_zoom, M=1, norm_ind=False)
        # right panel
        ax = axes[2]
        ax.clear()
        plot_ax_roi(ax, iwes[i, ...], cmap_np, axis_ind,
                    roi=roi_zoom, M=1, norm_ind=True)
        ax = axes[3]
        ax.clear()
        plot_ax_roi(ax, iwes_sr[i, ...], cmap_np, axis_ind, roi=roi_zoom, M=M, norm_ind=True)
        plt.tight_layout()
        return []
    ani = animation.FuncAnimation(fig, update, f_range, interval=100, blit=False)
    ani.save('E:/[X] NeuroSR/vid/visual.gif', fps=15)"""
"""
    f_range = np.arange(0, 25)
    tss = np.linspace(ts[0], ts[-1], num_t_ref)
    fig = plt.figure(figsize=(8, 3))
    fig.patch.set_alpha(0)  # 画布透明
    plt.rcParams.update({'font.size': 15})
    plt.gca().set_aspect('equal', adjustable='box')
    plt.subplots_adjust(wspace=0.4, hspace=0)
    def update(i):
        plt.subplot(121)
        vxd= vx_dense[int(tss[i])].detach().cpu().numpy()
        vyd= vy_dense[int(tss[i])].detach().cpu().numpy()
        plt.plot(int(tss[i]/1e3), vxd*1e3,marker='.',  color=flow_vectors_to_rgb(-1.0, 0.0))
        plt.plot(int(tss[i]/1e3), vyd*1e3, marker='.', color=flow_vectors_to_rgb(0.0, -1.0))
        # 坐标范围
        plt.xlim(ts[0]/1e3, ts[-1]/1e3)
        plt.gca().set_facecolor('none')
        for spine in plt.gca().spines.values():
            spine.set_color('white')
        plt.tick_params(colors='white')
        plt.subplot(122)
        vx = Dx[int(tss[i])]
        vy = Dy[int(tss[i])]
        colors = flow_vectors_to_rgb(vx, -vy)
        plt.plot(cx - vx, cy + vy, marker='.', color=colors)
        # 坐标范围
        plt.xlim(cx - flow_max_amp, cx + flow_max_amp)
        plt.ylim(cy + flow_max_amp, cy - flow_max_amp)
        # 背景透明 + 坐标轴白色
        plt.gca().set_facecolor('none')
        for spine in plt.gca().spines.values():
            spine.set_color('white')
        plt.tick_params(colors='white')
        #plt.axis("off")
        # plt.xlabel('x', color='white')
        # plt.ylabel('y', color='white')
        #plt.title(f"t = {round(tss[i]/1e3, 3)} ms", color='white')
        #plt.tight_layout()
        return []
    ani = animation.FuncAnimation(fig, update, f_range, interval=100, blit=False)
    ani.save('E:/[X] NeuroSR/vid/pos2.gif', fps=15)

"""



    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    fig.patch.set_alpha(0)
    f_range = np.arange(0, 5)
    num_t_ref =  5
    tss = np.linspace(ts[0], ts[-1], num_t_ref)
    roi_h = roi_0
    def update(i):
        plt.rcParams.update({'font.size': 15})
        plt.subplots_adjust(wspace=0.4, hspace=0)
        # left panel
        ax = axes[0]
        ax.clear()
        plot_ax_roi(ax, frames_sharp, cmap, axis_ind,
                    M=1, roi=roi_h, norm_ind=False)
        # middle panel
        ax = axes[1]
        ax.clear()
        frames_i = frames_img[frame_ind , :, :]
        plot_ax_roi(ax, frames_i, cmap, axis_ind,
                    roi=roi_h, M=1, norm_ind=False)
        # right panel
        ax = axes[2]
        ax.clear()
        plot_ax_roi(ax, iwe_gts[i].detach().cpu().numpy(), cmap_np, axis_ind, roi=roi_h, M=M, norm_ind=False)
        plt.tight_layout()
        ax = axes[3]
        ax.clear()
        plot_ax_roi(ax, I_pred_srs[i].detach().cpu().numpy(), cmap_np, axis_ind, roi=roi_h, M=M, norm_ind=False)
        plt.tight_layout()
        return []
    ani = animation.FuncAnimation(fig, update, f_range, interval=100, blit=False)
    ani.save('E:/[X] NeuroSR/vid/visual.gif', fps=5)




iter_num = vid_num
f_range = np.arange(0, iter_num)
fig = plt.figure("PT", figsize=(8, 8))
fig.patch.set_alpha(0)  # 画布透明
plt.rcParams.update({'font.size': 15})
plt.gca().set_aspect('equal', adjustable='box')
plt.subplots_adjust(wspace=0.4, hspace=0)


def update(i):
    ax = plt.subplot(221);
    ax.set_aspect(25 / (np.max(loss_hist_2) - np.min(loss_hist_2)))
    # plt.plot(f_range[i], loss_hist_2[i], marker='.')
    ax.scatter(f_range[i], loss_hist_2[i], marker='.', color="white")
    # 坐标范围
    plt.xlim(0, iter_num)
    plt.ylim(np.min(loss_hist_2), np.max(loss_hist_2))
    # plt.axis("off")
    plt.gca().set_facecolor('none')
    for spine in plt.gca().spines.values():
        spine.set_color('white')
    plt.tick_params(colors='white')
    ax = plt.subplot(222);
    ax.set_aspect(25 / (np.max(zs) - np.min(zs)))
    # plt.plot(f_range[i], zs[i], marker='.')
    ax.scatter(f_range[i], zs[i], marker='.', color="white")
    # 坐标范围
    plt.xlim(0, iter_num)
    plt.ylim(np.min(zs), np.max(zs))
    # 背景透明 + 坐标轴白色
    plt.gca().set_facecolor('none')
    for spine in plt.gca().spines.values():
        spine.set_color('white')
    plt.tick_params(colors='white')
    ax = plt.subplot(223)
    ax.clear()
    plot_ax_roi(ax, a_preds[i], cmap, axis_ind, roi=roi_0, Dxy_ori=Dxy_ori, M=M)
    plt.gca().set_facecolor('none')
    for spine in plt.gca().spines.values():
        spine.set_color('white')
    plt.tick_params(colors='white')
    ax = plt.subplot(224)
    ax.clear()
    plot_ax_roi(ax, phi_preds[i], cmap_phi, axis_ind, roi=roi_0, Dxy_ori=Dxy_ori, M=M)
    plt.gca().set_facecolor('none')
    for spine in plt.gca().spines.values():
        spine.set_color('white')
    plt.tick_params(colors='white')
    return []


ani = animation.FuncAnimation(fig, update, f_range, interval=100, blit=False)
ani.save('E:/[X] NeuroSR/vid/physics_twin_rest_refine.gif', fps=15)



iter_num = len(Dxy_pc_preds)
f_range = np.arange(0, iter_num)
tss = np.linspace(ts[0], ts[-1],iter_num)
fig = plt.figure("Dxy_pc", figsize=(8, 8))
fig.patch.set_alpha(0)  # 画布透明
plt.rcParams.update({'font.size': 15})
plt.gca().set_aspect('equal', adjustable='box')
def update(i):
    ax = plt.subplot(231)
    ax.set_aspect(len(loss_hist) / (np.max(loss_hist) - np.min(loss_hist)) * ax_asp)
    #ax.scatter(f_range[i], loss_hist[i], marker='o', color="white")
    ax.plot(f_range[:i + 1], loss_hist[:i + 1], color="white", linewidth=0.5)
    # 坐标范围
    plt.xlim(0, iter_num)
    plt.ylim(np.min(loss_hist), np.max(loss_hist))
    plt.gca().set_facecolor('none')
    for spine in plt.gca().spines.values():
        spine.set_color('white')
    plt.tick_params(colors='white')
    ax = plt.subplot(232)
    ax.clear()
    ax.set_aspect(Dxy_pc_pred.shape[0] / (
                np.max( Dxy_pc_preds) - np.min( Dxy_pc_preds)) * ax_asp)
    dx, dy = Dxy_pc_preds[i].T
    ax.plot(dx, color=flow_vectors_to_rgb(-1.0, 0.0))
    ax.plot(dy, color=flow_vectors_to_rgb(0.0, -1.0))
    plt.xlim(0, Dxy_pc_pred.shape[0])
    plt.ylim(np.min( Dxy_pc_preds), np.max( Dxy_pc_preds))
    plt.gca().set_facecolor('none')
    for spine in plt.gca().spines.values():
        spine.set_color('white')
    plt.tick_params(colors='white')
    ax = plt.subplot(233)
    ax.clear()
    ax.set_aspect(1)
    dx, dy = Dxy_pc_preds[i].T
    dx = np.insert(dx, 0, 0);
    dy = np.insert(dy, 0, 0)
    colors = flow_vectors_to_rgb(dx, -dy)
    s_x, s_y = cx, cy
    for k in range(dx.shape[0]):
        ax.plot([s_x, s_x + dx[k]], [s_y, s_y + dy[k]], color=colors[k])
        s_x += dx[k]
        s_y += dy[k]
    flow_max_amp = np.linalg.norm(np.abs(Dxy_pred.detach().cpu().numpy()).max(1), 2)
    plt.xlim(cx - flow_max_amp, cx + flow_max_amp)
    plt.ylim(cy + flow_max_amp, cy - flow_max_amp)
    # 背景透明 + 坐标轴白色
    plt.gca().set_facecolor('none')
    for spine in plt.gca().spines.values():
        spine.set_color('white')
    plt.tick_params(colors='white')
    ax1 = plt.subplot(234)
    plot_ax_roi(ax1, I_preds[i], cmap, axis_ind, roi=roi_0, M=M);
    ax2 = plt.subplot(235)
    plot_ax_roi(ax2, iwe_sr_preds[i], cmap_np, axis_ind, roi=roi_0, M=M, norm_ind=True);
    ax = plt.subplot(236)
    plot_ax_roi(ax, np.abs(bg_preds[i]), "inferno", axis_ind, roi=roi_0, M=1);
    return []
ani = animation.FuncAnimation(fig, update, f_range, interval=100, blit=False)
ani.save('E:/[X] NeuroSR/vid/Dxy_pc_no_alter.gif', fps=15)

# plt.plot(cx+Dx, cy+Dy, marker='o', color='red', alpha=0.5)


# ==============================
# 初始化 figure + 3D Axes
# ==============================

fig = plt.figure("3D", figsize=(10, 10))
ax = fig.add_subplot(projection='3d')
ax.set_box_aspect([1.0, 2, ax_asp])
ax.view_init(azim=60, elev=24)

# ---- 透明 Figure ----
fig.patch.set_facecolor((1, 1, 1, 0))

# ---- 透明 Axes 背景 ----
ax.patch.set_facecolor((1, 1, 1, 0))

# ---- 透明 pane （背景面） ----
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# ---- 去掉刻度 ----
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# ---- 去掉坐标轴标签 ----
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_zlabel("")

# ---- 保留轴线 ----
ax.w_xaxis.line.set_color((0.5, 0.5, 0.5, 1))
ax.w_yaxis.line.set_color((0.5, 0.5, 0.5, 1))
ax.w_zaxis.line.set_color((0.5, 0.5, 0.5, 1))

# ---- 坐标范围 ----
ax.set_xlim3d(0, 345)
ax.set_zlim3d(259, 0)
ax.set_ylim3d(0, eve_win / 1e6)



# ==============================
# 在 Y = 0 平面绘制一张图像作为背景
# ==============================

img = frame / frame.max()  # [0,1]
H, W = sensor_size
X, Y = np.meshgrid(np.arange(W), np.arange(H))
T = np.zeros_like(img)     # Y轴位置=0

ax.plot_surface(
    X, T, Y,
    facecolors=plt.cm.gray(img),
    rstride=1, cstride=1,
    shade=False,
    alpha=1.0    # 背景图像 100% 不透明
)



# ==============================
# 初始化 scatter（空的）
# ==============================

pos_scatter = ax.scatter([], [], [], c='r', s=2, alpha=alpha)
neg_scatter = ax.scatter([], [], [], c='b', s=2, alpha=alpha)
cur_scatter = ax.scatter([], [], [], c='k', s=3, alpha=1)



# ==============================
# update 函数（无 cla！）
# ==============================

def update(i):
    e = (i + 1) * i_win
    xsii, ysii, tsii, psii = time_window(xs, ys, ts, ps, i * i_win, win=i_win)

    # 完整累积窗口（如果你需要）
    if e <= eve_win:
        loc = np.where(ts <= e)
        tsi, xsi, ysi, psi = ts[loc], xs[loc], ys[loc], ps[loc]
    else:
        xsi, ysi, tsi, psi = time_window(xs, ys, ts, ps, 0, win=eve_win)

    pos_loc = np.where(psi == 1)
    neg_loc = np.where(psi == 0)

    pos_xs, pos_ys, pos_ts = xsi[pos_loc], ysi[pos_loc], tsi[pos_loc]
    neg_xs, neg_ys, neg_ts = xsi[neg_loc], ysi[neg_loc], tsi[neg_loc]

    # ---- 更新散点，而不是 cla() ----
    pos_scatter._offsets3d = (pos_xs, (e - pos_ts) / 1e6, pos_ys)
    neg_scatter._offsets3d = (neg_xs, (e - neg_ts) / 1e6, neg_ys)
    cur_scatter._offsets3d = (xsii, e/1e6 * np.ones_like(xsii), ysii)

    return pos_scatter, neg_scatter, cur_scatter



# ==============================
# 动画
# ==============================

ani = animation.FuncAnimation(fig, update, f_range, interval=100, blit=False)

ani.save(
    'E:/[X] NeuroSR/vid/3d.gif',
    fps=10,
    savefig_kwargs={"transparent": True}
)

plt.close(fig)