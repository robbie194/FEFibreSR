"""
extract_event_frame.py
=======================
从 .aedat4 / .h5 / .npy 事件文件里读出事件流 (ts, xs, ys, ps) 与 APS 灰度帧
(frames_img, time_img)，按一个时间窗口裁剪后，把事件 / 帧 / 曝光时间 **分别落盘**
到项目内的 ``output/<文件名>/`` 目录：

  - 事件      -> ``events.npy``      (形状 (N, 4)，列顺序 = ts, xs, ys, ps)
  - APS 帧    -> ``frames.zip``      (每帧一张灰度 PNG，打包进 zip)
  - 曝光时间  -> ``exposure_us.npy`` (形状 (帧数, 2)，列 = 曝光起/止) + 可读的 .csv

窗口规则（见 __main__ 的线性 demo）：
  预先给一个 ``s`` 和 ``win``，得到初始窗口 [s, s+win]。脚本会把窗口**自动对齐到帧
  曝光边界**：
    * s   <- 窗口内第一帧的曝光开始 ts
    * e   <- 窗口内最后一帧的曝光结束 ts
    * win <- e - s
  然后取该窗口内的所有帧与所有事件。事件与曝光统一减去 s，落在同一“窗口相对时间轴”
  （第一帧曝光从 0 开始），二者直接对齐。

设计原则：
  * ``load_events`` 与 utils/utility.py 里的同名函数**逐行等价**（直接照搬），存出的
    ``events.npy`` 能被 ``load_events(eve_dtype="npy", path=...)`` 原样读回。
  * ``time_window`` 与 utils/utility.py::time_window 逐行等价。
"""

import os
import zipfile

import numpy as np

# 这几个依赖只在对应分支用到；放顶部更直观，缺了也不影响其它分支。
try:
    from dv import AedatFile          # 读 .aedat4 需要 dv（dv-python）
except Exception:                     # 没装也不影响 npy / h5 分支
    AedatFile = None
try:
    import h5py                       # 读 .h5 需要 h5py
except Exception:
    h5py = None
try:
    import cv2                        # 写 PNG / 彩转灰用 OpenCV
except Exception:
    cv2 = None


# ============================================================================ #
#  事件读取：照搬 utils/utility.py::load_events（保持逐行等价，便于读回）          #
# ============================================================================ #
def load_events(eve_dtype="aedat4", path="E:/cosi/mla final/Sphe0_2/s7.aedat4"):
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


# ============================================================================ #
#  事件时间窗：照搬 utils/utility.py::time_window（保持逐行等价）                  #
# ============================================================================ #
def time_window(xs, ys, ts, ps, s, win):
    loc = np.where((ts <= s + win) & (ts >= s))
    ts, xs, ys, ps = ts[loc], xs[loc], ys[loc], ps[loc]
    return xs, ys, ts, ps


# ============================================================================ #
#  小工具                                                                        #
# ============================================================================ #
def project_output_dir(name):
    """返回（并创建）项目内的 ``output/<name>/`` 目录。"""
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:                 # 在 console 里逐行粘贴时 __file__ 不存在
        root = r"d:\Program Files\PycharmProjects\isl_diff_event"
    out = os.path.join(root, "output", name)
    os.makedirs(out, exist_ok=True)
    return out


def to_gray_stack(frames_img3):
    """把 load_events 返回的帧堆统一成 (N, H, W) 灰度。

    彩色 (N, H, W, 3) -> 逐帧 BGR2GRAY；已是灰度 (N, H, W) 原样返回；
    单帧 (H, W) 补成 (1, H, W)。（前两支照搬源代码流程。）
    """
    frames_img3 = np.asarray(frames_img3)
    if frames_img3.ndim == 4:                       # (N, H, W, 3) 彩色
        out = np.zeros(frames_img3.shape[:3])
        for i in range(frames_img3.shape[0]):
            out[i, :, :] = cv2.cvtColor(frames_img3[i, :, :, :], cv2.COLOR_BGR2GRAY)
        return out
    elif frames_img3.ndim == 3:                     # (N, H, W) 已是灰度
        return frames_img3
    elif frames_img3.ndim == 2:                     # 单帧 (H, W)
        return frames_img3[None, ...]
    raise ValueError(f"未知的帧维度: {frames_img3.shape}")


def select_frame_window(time_img, s, win):
    """把初始窗口 [s, s+win] 对齐到内部帧的曝光边界。

    返回 (frame_inds, s_aligned, e_aligned, win_aligned)：
      - frame_inds  : 起始曝光落在 [s, s+win] 内的帧下标
      - s_aligned   : 第一帧的曝光开始 ts
      - e_aligned   : 最后一帧的曝光结束 ts
      - win_aligned : e_aligned - s_aligned
    """
    e0 = s + win
    inds = np.where((time_img[:, 0] >= s) & (time_img[:, 0] <= e0))[0]
    if inds.size == 0:
        raise ValueError(f"窗口 [{s}, {e0}] 内没有帧，请调大 win 或调整 s。")
    s_aligned = int(time_img[inds[0], 0])
    e_aligned = int(time_img[inds[-1], 1])
    return inds, s_aligned, e_aligned, int(e_aligned - s_aligned)


# ============================================================================ #
#  落盘                                                                          #
# ============================================================================ #
def save_events(ts, xs, ys, ps, out_path):
    """把事件流堆成 (N, 4) 数组并存成 .npy。

    列顺序固定为 [ts, xs, ys, ps]，与 load_events(eve_dtype="npy") 的读取顺序
    完全对应，因此能被原样读回。各分量 dtype 不同（ts 是 int64、x/y 是 uint16、
    p 是 bool/int8），column_stack 会统一上转成能容纳时间戳的 int64。
    """
    events = np.column_stack([
        np.asarray(ts, dtype=np.int64),
        np.asarray(xs, dtype=np.int64),
        np.asarray(ys, dtype=np.int64),
        np.asarray(ps, dtype=np.int64),
    ])
    np.save(out_path, events)
    print(f"[events] {events.shape[0]} 个事件 -> {out_path}")
    return events


def save_frames_zip(frames_img, zip_path, prefix="frame"):
    """每帧编码为 8-bit 灰度 PNG，打包进一个 zip（PNG 已是无损压缩）。"""
    if cv2 is None:
        raise RuntimeError("需要 OpenCV (cv2) 来编码 PNG，请先 `pip install opencv-python`。")

    frames_img = np.asarray(frames_img)
    if frames_img.ndim == 2:                       # 单帧 -> (1, H, W)
        frames_img = frames_img[None, ...]

    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, frame in enumerate(frames_img):
            # APS 帧通常已是 uint8 灰度；保险起见做一次安全转换。
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            frame = np.squeeze(frame)              # 去掉可能残留的单通道维 (H, W, 1)
            ok, buf = cv2.imencode(".png", frame)
            if not ok:
                raise RuntimeError(f"第 {i} 帧 PNG 编码失败。")
            zf.writestr(f"{prefix}_{i:04d}.png", buf.tobytes())
    print(f"[frames] {frames_img.shape[0]} 张灰度 PNG -> {zip_path} (zip)")
    return zip_path


def save_exposure(time_img, out_dir):
    """存曝光时间戳：二进制 .npy（精确）+ 人类可读 .csv。"""
    time_img = np.asarray(time_img)
    if time_img.ndim == 1:                         # 单帧 -> (2,)，补成 (1, 2)
        time_img = time_img[None, :]

    npy_path = os.path.join(out_dir, "exposure_us.npy")
    csv_path = os.path.join(out_dir, "exposure_us.csv")
    np.save(npy_path, time_img)
    np.savetxt(
        csv_path, time_img, fmt="%d", delimiter=",",
        header="exposure_start_us,exposure_end_us", comments="",
    )
    print(f"[exposure] {time_img.shape[0]} 段曝光 -> {npy_path} / {csv_path}")
    return npy_path


# ============================================================================ #
#  可视化                                                                        #
# ============================================================================ #
def render_event_image(xs, ys, ps, sensor_size):
    """把窗口内事件按极性累加成有符号计数图 (H, W)：正极性 +1、负极性 -1。

    与 utils/utility.py::events_to_image(polar=True, 非 bilinear) 的 np.bincount
    思路一致，但把 ps∈{0,1} 显式映射到 {-1,+1}，方便用发散色图同时看正负事件。
    （纯可视化，不参与任何算法。）
    """
    H, W = sensor_size
    pol = np.where(np.asarray(ps) > 0, 1.0, -1.0)
    coords = np.asarray(ys, dtype=np.int64) * W + np.asarray(xs, dtype=np.int64)
    img = np.bincount(coords, weights=pol, minlength=H * W).reshape(H, W)
    return img


def overlay_events_on_frame(frame_gray, ev_img):
    """灰度帧打底，正事件染红、负事件染蓝，叠成一张 RGB 预览。"""
    base = np.asarray(frame_gray, dtype=np.float32)
    base = base / max(base.max(), 1.0)
    rgb = np.stack([base, base, base], axis=-1)
    pos = np.clip(ev_img, 0, None)
    neg = np.clip(-ev_img, 0, None)
    # 用 99 百分位归一，避免个别热像素把整体压暗。
    pscale = np.percentile(pos[pos > 0], 99) if np.any(pos > 0) else 1.0
    nscale = np.percentile(neg[neg > 0], 99) if np.any(neg > 0) else 1.0
    rgb[..., 0] = np.clip(rgb[..., 0] + pos / max(pscale, 1.0), 0, 1)   # R: 正
    rgb[..., 2] = np.clip(rgb[..., 2] + neg / max(nscale, 1.0), 0, 1)   # B: 负
    return rgb


def visualize(frames_sel, xs, ys, ps, sensor_size, time_sel=None,
              out_path=None, max_frames=4, show=True):
    """画一排：选中的 APS 帧 + 事件有符号图 + (帧0 上的)事件叠加图。

    out_path 给定则存 PNG；show=True 则弹窗（console 里逐行跑时方便）。
    """
    import matplotlib.pyplot as plt

    frames_sel = np.asarray(frames_sel)
    if frames_sel.ndim == 2:
        frames_sel = frames_sel[None, ...]
    n_show = min(frames_sel.shape[0], max_frames)

    ev_img = render_event_image(xs, ys, ps, sensor_size)
    overlay = overlay_events_on_frame(frames_sel[0], ev_img)
    absimg = np.abs(ev_img)
    vmax = float(np.percentile(absimg[absimg > 0], 99)) if np.any(absimg > 0) else 1.0
    vmax = vmax or 1.0

    ncols = n_show + 2
    fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 3.6))
    axes = np.atleast_1d(axes)

    for i in range(n_show):
        axes[i].imshow(frames_sel[i], cmap="gray")
        ttl = f"frame {i}"
        if time_sel is not None:
            t0, t1 = np.asarray(time_sel)[i]
            ttl += f"\n[{int(t0)}, {int(t1)}] us"
        axes[i].set_title(ttl, fontsize=9)
        axes[i].axis("off")

    axes[n_show].imshow(ev_img, cmap="bwr", vmin=-vmax, vmax=vmax)
    axes[n_show].set_title(f"events signed\n{np.asarray(xs).size} evs", fontsize=9)
    axes[n_show].axis("off")

    axes[n_show + 1].imshow(overlay)
    axes[n_show + 1].set_title("overlay  R:+ / B:-", fontsize=9)
    axes[n_show + 1].axis("off")

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        print(f"[viz] 预览图 -> {out_path}")
    if show:
        plt.show()
    return fig


# ============================================================================ #
#  一键封装（与下面的线性 demo 调用同一组函数）                                    #
# ============================================================================ #
def extract(path, eve_dtype="aedat4", out_dir=None, s=None, win=None, preview=True):
    """读取 -> (可选)按窗口裁剪 -> 事件 npy / 帧 zip / 曝光 npy+csv (+ 预览图)。

    s, win 给定时（仅 aedat4）：窗口自动对齐到 [s, s+win] 内的首帧 start ~ 末帧 end，
    事件与曝光统一减去对齐后的 s。s, win 为 None 时导出全部帧与事件。
    preview=True 时额外存一张 preview.png（不弹窗）。
    """
    base = os.path.splitext(os.path.basename(path))[0]
    if out_dir is None:
        out_dir = project_output_dir(base)
    os.makedirs(out_dir, exist_ok=True)

    if eve_dtype != "aedat4":
        ts, xs, ys, ps = load_events(eve_dtype, path)
        save_events(ts, xs, ys, ps, os.path.join(out_dir, "events.npy"))
        print("[info] 该格式无 APS 帧/曝光信息，仅导出事件。")
        return out_dir

    ts, xs, ys, ps, frames_img3, time_img = load_events(eve_dtype, path)
    ts = ts - ts.min()
    frames_img = to_gray_stack(frames_img3)

    if s is not None and win is not None:
        frame_inds, s, e, win = select_frame_window(time_img, s, win)
        frames_sel = frames_img[frame_inds]
        time_sel = time_img[frame_inds]
        xs, ys, ts, ps = time_window(xs, ys, ts, ps, s=s, win=win)
        ts = ts - s
        time_sel = time_sel - s
        print(f"对齐窗口: frames {frame_inds[0]}..{frame_inds[-1]} ({frame_inds.size} 帧), "
              f"s={s}, e={e}, win={win}, events={len(ts)}")
    else:
        frames_sel = frames_img
        time_sel = time_img

    save_events(ts, xs, ys, ps, os.path.join(out_dir, "events.npy"))
    save_frames_zip(frames_sel, os.path.join(out_dir, "frames.zip"))
    save_exposure(time_sel, out_dir)
    if preview:
        sensor_size = frames_sel.shape[-2:]
        visualize(frames_sel, xs, ys, ps, sensor_size, time_sel=time_sel,
                  out_path=os.path.join(out_dir, "preview.png"), show=False)
    print(f"完成。输出目录: {out_dir}")
    return out_dir


if __name__ == "__main__":
    # ======================================================================== #
    #  线性 demo —— 习惯在 PyCharm Python Console 里逐行跑、随时看中间变量。       #
    # ======================================================================== #
    #path = r"E:\davis\pei\onion.aedat4"
    #path = r"E:\davis\pei\02.aedat4"
    path = r"D:/BaiduNetdiskDownload/Rotation/02_rot.aedat4"
    eve_dtype = "aedat4"

    # —— 预输入窗口（会被自动对齐到帧曝光边界）——
    s = 2.45e6
    win = int(2e5) #win = int(1e5)

    # 输出到项目内 output/<文件名>/
    base = os.path.splitext(os.path.basename(path))[0]
    out_dir = project_output_dir(base)

    # 1) 读事件 + APS 帧 + 曝光时间
    ts, xs, ys, ps, frames_img3, time_img = load_events(eve_dtype, path)
    ts = ts - ts.min()                          # 从 0 开始（load_events 已减 t_ref，这里再保险）
    frames_img = to_gray_stack(frames_img3)     # 统一成 (N, H, W) 灰度
    print("全部:", "events", ts.shape, "| frames", frames_img.shape, "| exposure", time_img.shape)

    # 2) 把初始窗口 [s, s+win] 对齐到帧曝光边界：s<-首帧start, e<-末帧end, win<-e-s
    frame_inds, s, e, win = select_frame_window(time_img, s, win)
    print(f"对齐窗口: frames {frame_inds[0]}..{frame_inds[-1]} ({frame_inds.size} 帧), s={s}, e={e}, win={win}")

    # 3) 取窗口内的帧与曝光
    frames_sel = frames_img[frame_inds]
    time_sel = time_img[frame_inds]

    # 4) 取窗口内的事件，并与帧共用“窗口相对时间轴”（都减去 s，首帧曝光从 0 开始）
    xs, ys, ts, ps = time_window(xs, ys, ts, ps, s=s, win=win)
    ts = ts - s
    time_sel = time_sel - s
    print(f"窗口内事件: {len(ts)} 个")

    # 5) 分别落盘：事件 npy / 帧 zip / 曝光 npy+csv
    save_events(ts, xs, ys, ps, os.path.join(out_dir, "events.npy"))
    save_frames_zip(frames_sel, os.path.join(out_dir, "frames.zip"))
    save_exposure(time_sel, out_dir)
    print("输出目录:", out_dir)

    # 6) 验证：events.npy 能被 load_events('npy') 读回
    ts2, xs2, ys2, ps2 = load_events("npy", os.path.join(out_dir, "events.npy"))
    print("事件读回一致:", np.array_equal(ts, ts2) and np.array_equal(ps, ps2))

    # 7) 可视化：选中帧 + 事件有符号图 + 叠加图（存 preview.png 并弹窗）
    sensor_size = frames_sel.shape[-2:]          # (H, W)，由帧尺寸自动推断
    visualize(frames_sel, xs, ys, ps, sensor_size, time_sel=time_sel,
              out_path=os.path.join(out_dir, "preview.png"), show=True)
