
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
evk =LinearSegmentedColormap.from_list('evk', (
    # Edit this gradient at https://eltos.github.io/gradient/#4C71FF-000000-FFFFFF
    (0.000, (0.298, 0.443, 1.000)),
    (0.500, (0.000, 0.000, 0.000)),
    (1.000, (1.000, 1.000, 1.000))))
bi_evk = LinearSegmentedColormap.from_list('bi_evk', (
    # Edit this gradient at https://eltos.github.io/gradient/#0:4C71FF-49:4C71FF-50:000000-51:FFFFFF-100:FFFFFF
    (0.000, (0.298, 0.443, 1.000)),
    (0.490, (0.298, 0.443, 1.000)),
    (0.500, (0.000, 0.000, 0.000)),
    (0.510, (1.000, 1.000, 1.000)),
    (1.000, (1.000, 1.000, 1.000))))
bi_bwr =LinearSegmentedColormap.from_list('bi_bwr', (
    # Edit this gradient at https://eltos.github.io/gradient/#0:FF4C4E-49:FF4C4E-50:FFFFFF-51:5E6CFF-100:5E6CFF
    (0.000, (1.000, 0.298, 0.306)),
    (0.490, (1.000, 0.298, 0.306)),
    (0.500, (1.000, 1.000, 1.000)),
    (0.510, (0.369, 0.424, 1.000)),
    (1.000, (0.369, 0.424, 1.000))))


darkblue = LinearSegmentedColormap.from_list('darkblue', (
    # Edit this gradient at https://eltos.github.io/gradient/#0:11113E-25:18405E-50:207181-67.8:5FA38F-83:CFE8B9-100:FFFFFF
    (0.000, (0.067, 0.067, 0.243)),
    (0.250, (0.094, 0.251, 0.369)),
    (0.500, (0.125, 0.443, 0.506)),
    (0.678, (0.373, 0.639, 0.561)),
    (0.830, (0.812, 0.910, 0.725)),
    (1.000, (1.000, 1.000, 1.000))))
import numpy as np
import cv2
import matplotlib.pyplot as plt

def make_flow_color_wheel(size=512):
    """
    生成中心黑、正方形完整的光流色轮 (BGR)
    """
    # 坐标网格
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    A = np.arctan2(Y, X)  # [-pi, pi]

    # HSV 通道
    H = (A + np.pi) / (2 * np.pi) * 179  # Hue in [0,179]
    S = np.ones_like(R) * 255             # 饱和度最大
    V = np.clip(R, 0, 1) * 255            # 半径越大亮度越高 (中心黑)

    hsv = np.stack([H, S, V], axis=-1).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr


def make_flow_color_wheel_rgb(size=512):
    """生成中心黑、正方形完整的相位色轮 (BGR)"""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    A = np.arctan2(Y, X)  # [-pi, pi]

    H = ((A + np.pi) / (2 * np.pi)) * 179
    S = np.ones_like(R) * 255
    V = np.clip(R, 0, 1) * 255

    hsv = np.stack([H, S, V], axis=-1).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

# 生成并显示
wheel = make_flow_color_wheel(1024)
# plt.imshow(cv2.cvtColor(wheel, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# cv2.imwrite("flow_color_wheel.png", wheel)


def flow_vectors_to_rgb(vx, vy, clip_mag=None):
    """
    输入:
        vx, vy : 1D 或 2D ndarray，形状相同
    输出:
        colors : (N, 3) RGB 颜色，可直接用于 plt.scatter(..., c=colors)
    """
    vx = np.asarray(vx)
    vy = np.asarray(vy)

    # 计算幅值与方向
    mag, ang = cv2.cartToPolar(vx, vy, angleInDegrees=False)

    # 幅值归一化
    if clip_mag is not None:
        mag = np.clip(mag, 0, clip_mag)
    else:
        mag = mag / (np.max(mag) + 1e-6)

    # HSV 编码：Hue = 方向，Value = 幅值
    H = ang * 180 / np.pi / 2   # 0–180
    S = np.ones_like(H) * 255
    V = np.clip(mag, 0, 1) * 255
    hsv = np.stack([H, S, V], axis=-1).astype(np.uint8)

    # HSV→BGR→RGB（OpenCV默认是BGR）
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = bgr[..., ::-1] / 255.0  # 转为 [0,1] 以便 matplotlib
    return rgb.reshape(-1, 3)

def visualize_complex_field(a, phi, norm = True):
    """
    a: 振幅 (H×W, 任意范围)
    phi: 相位 (H×W, 单位: rad, [-pi, pi])
    返回: RGB 图像, 可直接 plt.imshow()
    """

    #norm amp to [[0,1])
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    if norm:
        #phi = ((phi - phimin) / (phimax - phimin + 1e-8)) * 2 * np.pi - np.pi
        phi = phi -np.median(phi)
        phimin = phi.min()
        phimax = phi.max()
        phimaxmax  =  np.abs(phi).max()
        phi = TwoSlopeNorm(vmin=-phimaxmax , vcenter=0, vmax=phimaxmax )(phi) * np.pi


    a_norm = a#(a - a.min()) / (a.max() - a.min() + 1e-8)
    # --- 2. map phase to Hue ---
    H = ((phi + np.pi) / (2 * np.pi)) * 179   # hue in [0,179]
    S = np.ones_like(a_norm) * 255
    V = (a_norm * 255).astype(np.uint8)

    hsv = np.stack([H.astype(np.uint8), S.astype(np.uint8), V], axis=-1)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return rgb