# 多芯光纤 Frame–Event 端到端仿真与 NeuroSR 重建方案

- 版本：v0.1
- 日期：2026-07-31（Asia/Hong_Kong）
- 目标光纤：Fujikura FIGH-15-600N
- 当前阶段：先完成可控仿真闭环，再迁移到真实 DAVIS/事件相机采集
- 相关本地资料：[NeuroSR 主文](Paper/Wang%20et%20al.%20-%201%20Neuromorphic%20High-Throughput%20Imaging%20via%202%20Trans.pdf)、[NeuroSR 补充材料](Paper/supp%20-%20Neuromorphic%20High-Throughput%20Imaging.pdf)、[v2e 论文](Paper/v2e_2021_2021%20IEEECVF%20Conference%20on%20Computer%20Vision%20and%20Pattern%20Recognition%20Workshops%20%28CVPRW%29.pdf)

## 1. 先说结论：这条路线是对的，但前向模型必须“光纤感知”

我对目前设想的理解是：

1. 从无蜂窝的高分辨率物体强度图像 `O` 出发。
2. 用多芯成像光纤、成像光学、相对运动以及传感器的统一物理前向模型，得到同源、同步的低速 APS frame 和事件流。
3. 重建时复用同一个前向模型，把 frame 作为绝对强度/低频锚点，把事件作为运动诱导的高频梯度约束，反演出高分辨率、无蜂窝的 `O`。
4. 在已知真值、已知运动的仿真环境中逐级验证，再加入未知运动、噪声和模型失配。
5. 仿真闭环可靠后，标定真实光纤和真实 event-frame 传感器，再迁移到实验数据。

这正符合 NeuroSR 的核心思想：**不是把事件当成一张额外图片，而是把 frame 和 events 都写成同一潜在物体、同一运动和同一成像系统的观测。**论文将强度帧看作低频光度锚点，将运动诱导事件看作高频梯度约束，并联合优化结构、运动及物理参数。[NeuroSR 公开预印本](https://assets-eu.researchsquare.com/files/rs-9045382/v1_covered_00b743f5-7e9b-4452-8a3b-8781b8dfff96.pdf)

但多芯光纤系统有一个必须修正的地方：

> 当物体在光纤远端相对纤芯阵列运动时，近端蜂窝中每个纤芯亮斑的位置通常固定，变化的是每根纤芯输出的强度。因此，不能未经修改地使用论文中“整幅传感器图像平移 `I(x-v(t))`”的简化模型。

我们的模型必须显式包含远端物体相对纤芯格点的运动、每个纤芯的有限孔径采样、近端纤芯亮斑形成和传感器成像。只有这样，优化变量 `O` 才是无蜂窝物体；蜂窝结构只存在于前向算子里，重建结果才不会把蜂窝当成物体纹理保留下来。

## 2. 这项工作的边界与三个重要判断

### 2.1 当前做的是强度成像，不先做相干场恢复

第一版建议假设宽带非相干照明、荧光或其他可用强度线性模型近似的成像方式。潜变量为非负强度图像：

$$
O(\mathbf u) \ge 0.
$$

若以后使用窄带相干照明，纤芯间相位、模态耦合、弯曲引起的相位变化和 speckle 会使模型从强度叠加变成复场传播；那是另一个难度等级，不应与第一版同时解决。

### 2.2 超分辨来自“亚纤芯位置的多次独立采样”，不是来自插值

静止物体通过固定纤芯阵列时，芯间空隙中的信息不可辨识。相对运动让同一物体位置依次落到不同的纤芯孔径中，事件以高时间密度记录每根纤芯强度变化，因此可以把时间采样重新映射为空间采样。

这要求：

- 运动发生在物体/远端像与纤芯格点之间；
- 轨迹必须覆盖一个或多个纤芯晶格单元的不同亚像素相位；
- 最好包含两个不共线方向，否则事件主要约束单一方向的梯度；
- 纤芯孔径、远端 PSF 和相机 PSF 没有传递的频率仍无法恢复。

### 2.3 不应由低帧率视频经插帧后“制造”事件信息

正确的仿真应从高分辨率真值和连续运动直接计算高时间分辨率的传感器辐照度，再分别生成 frame 和 events。不能先生成低帧率、已经模糊的 frame，再把它交给 SuperSloMo/v2e 产生事件；后者只会继承和插值 frame 中已有的信息，不能成为独立的高频物理观测。

现有 `v2e` 可以作为事件传感器后端，但它的输入应是我们的前向模型输出的**高时间分辨率连续强度序列**，并关闭 SuperSloMo，而不是输入最终低速 APS frame。

## 3. FIGH-15-600N 参数及几何一致性

Fujikura 当前产品资料给出的 FIGH-15-600N 规格为：图像单元数 `15000 ± 1500`、有效 image circle 直径 `550 ± 30 µm`、整根 fiber 直径 `600 ± 30 µm`、涂覆直径 `700 ± 35 µm`。不要把 600 µm 总直径直接当成 600 µm 有效成像区。[Fujikura 官方 FIGH N 型数据表](https://www.optic-product.fujikura.com/wp-content/uploads/2024/07/FPE29A.pdf)

本文采用以下工作参数：

| 参数 | 第一版值 | 来源/说明 |
|---|---:|---|
| 有效图像圆直径 `D_active` | 550 µm | 厂商值，容差 ±30 µm |
| 光纤总直径 `D_fiber` | 600 µm | 厂商值，容差 ±30 µm |
| 标称纤芯数 `N_core` | 15000 | 厂商值，容差 ±1500 |
| 芯间距 `p_core` | 4.5 µm | 用户给定工作值；文献也报告约 4.5 µm |
| 芯直径 `d_core` | 2.9 µm | 用户给定工作值，后续用显微标定替换 |
| 数值孔径 `NA` | 暂不固定 | 第三方资料常报告约 0.39，但当前厂商简表未给出，第一版用实测/设定 PSF 更稳妥 |

一项 FIGH-15-600N 的公开系统表征报告了约 550 µm 有效区、15000 cores 和约 4.5 µm inter-core distance，可作为间距的外部交叉参考。[2P-FENDO-II 光纤束参数表](https://pmc.ncbi.nlm.nih.gov/articles/PMC12946751/)

### 3.1 纤芯数、有效直径和间距不能同时随意指定

理想六角晶格中，每个纤芯占据的单元面积为

$$
A_{cell}=\frac{\sqrt{3}}{2}p_{core}^2,
$$

圆形有效区内纤芯数近似为

$$
N \approx \frac{\pi(D_{active}/2)^2}{A_{cell}}.
$$

代入 `D_active=550 µm, p_core=4.5 µm`，得到约 `1.355×10^4` 根，接近厂商 `15000±1500` 的下限。反过来，若要在 550 µm 圆内严格放约 15000 根，理想间距应约为 `4.28 µm`。

因此提供两个仿真模式：

- **推荐的 parameter-first 模式**：固定 `D_active=550 µm`、`p_core=4.5 µm`，接受实际生成约 13.5k 根纤芯。它尊重当前已知的物理尺寸。
- **count-matched 模式**：固定 `D_active=550 µm`、`N_core≈15000`，令间距约 `4.28 µm`。它适合测试标称纤芯数的影响。

后续真实实验中，这两个模式都应被实际提取的 `core_map` 替代。

### 3.2 理想填充率

若芯直径为 2.9 µm、间距为 4.5 µm，则理想六角晶格的几何填充率约为

$$
\rho=\frac{\pi(d_{core}/2)^2}{(\sqrt{3}/2)p_{core}^2}\approx 0.377.
$$

这解释了蜂窝空隙明显的原因，也说明相对运动覆盖芯间空隙对无蜂窝重建非常重要。

## 4. 建议的统一前向模型

### 4.1 坐标和变量

| 符号 | 含义 |
|---|---|
| `O(u)` | 要重建的远端高分辨率、无蜂窝物体强度 |
| `u` | 远端物体/光纤端面的物理坐标，单位 µm |
| `s(t)` | 物体像相对远端纤芯阵列的二维位移，单位 µm |
| `r_i^d` | 第 `i` 根纤芯的远端中心 |
| `r_i^p` | 第 `i` 根纤芯的近端中心 |
| `A_i` | 第 `i` 根纤芯的远端采样孔径/耦合 PSF |
| `g_i` | 第 `i` 根纤芯的透过增益 |
| `P_i` | 第 `i` 根纤芯在近端及继像后的亮斑 PSF |
| `x_p` | event-frame 传感器第 `p` 个像素坐标 |
| `I_p(t)` | 第 `p` 个传感器像素的连续线性辐照度 |
| `Θ` | 纤芯位置、孔径、PSF、增益、耦合、放大率等系统参数 |

符号的平移正负只要在仿真、IWE 和重建中始终一致即可。本文约定正 `s(t)` 表示物体图像相对固定纤芯阵列发生正向位移。

### 4.2 远端物体与成像 PSF

先把真值经过远端光学 PSF，再施加连续运动：

$$
J(\mathbf u,t)=\big[h_d * O\big](\mathbf u-\mathbf s(t)).
$$

第一版可设 `h_d=δ`，只研究纤芯采样；第二版再加入高斯或实测 PSF、畸变和照明不均匀。若远端物镜存在放大率 `M_d`，应把 `s(t)` 和物体像素尺寸统一换算到光纤端面坐标。

### 4.3 每根纤芯的有限孔径采样

第 `i` 根纤芯接收到的强度为

$$
a_i(t)=g_i\int A_i(\mathbf u-\mathbf r_i^d)J(\mathbf u,t)\,d\mathbf u.
$$

第一版使用归一化圆盘孔径：

$$
A_i(\mathbf u)=\frac{1}{\pi(d_i/2)^2}\,\mathbf 1_{\|\mathbf u\|\le d_i/2}.
$$

这一步不能简化为只取纤芯中心处的单像素值，否则会虚构超过 2.9 µm 芯孔径传递能力的高频信息。实现时可用亚像素圆盘积分、预计算稀疏采样矩阵，或在高分辨率网格上做卷积后于 `r_i^d` 采样。

### 4.4 串扰、坏芯和近端蜂窝形成

可用稀疏耦合矩阵 `C` 描述邻芯串扰：

$$
\tilde{\mathbf a}(t)=C\mathbf a(t).
$$

理想情况 `C=I`。近端光强为

$$
F(\mathbf z,t)=b(\mathbf z)+\sum_i \tilde a_i(t)P_i(\mathbf z-\mathbf r_i^p),
$$

其中 `b` 是背景，`P_i` 可以是圆盘、Gaussian 或真实标定的每芯 PSF。坏芯通过 `g_i=0` 表示，芯间透过差异通过不同 `g_i` 表示。

对于相干成像，此处的强度相加不成立，必须改为复振幅叠加；第一阶段明确不做该扩展。

### 4.5 继像、传感器像素积分和共享连续强度

近端光纤图经继像系统和传感器像素孔径后，第 `p` 个像素的连续信号写为

$$
I_p(t)=\int_{\mathcal P_p}\left[h_r*F(\cdot,t)\right](\mathbf x)\,d\mathbf x.
$$

把上述全部步骤合并，记为

$$
\mathbf I(t)=\mathcal H_{\Theta}\big(O,\mathbf s(t)\big).
$$

`H_Θ` 是本项目最核心的共享前向算子。APS frame 和 events 必须从同一个 `I(t)` 分叉产生，时间零点、曝光窗、运动轨迹和坐标系必须完全一致。

### 4.6 APS frame 模型

第 `m` 帧的曝光区间为 `[t_m,t_m+T_exp]`。电子数期望为

$$
\lambda_{m,p}=\eta_p\int_{t_m}^{t_m+T_{exp}}I_p(t)\,dt+d_pT_{exp},
$$

观测可用 Poisson-Gaussian 模型：

$$
n_{m,p}\sim\operatorname{Poisson}(\lambda_{m,p}),
$$

$$
y_{m,p}=Q\left(g_{APS}n_{m,p}+\epsilon_{read}\right),
\qquad \epsilon_{read}\sim\mathcal N(0,\sigma_r^2).
$$

第一版关闭噪声和量化，只保留曝光积分；第二版加入 shot noise、read noise、dark offset、PRNU 和饱和。所有 frame 必须保存为线性值，不能经过 sRGB gamma、自动曝光或视频压缩。

### 4.7 事件模型

传感器像素首先经过有限光感受器带宽：

$$
\bar I_p(t)=h_{lp,p}*I_p(t),
$$

再转到 log domain：

$$
L_p(t)=\log\big(\bar I_p(t)+I_0\big).
$$

若相对上一次事件的 log 强度变化越过阈值，则触发

$$
L_p(t_k)-L_p(t_{last,p})=q_kc_p,
\qquad q_k\in\{-1,+1\}.
$$

输出事件为

```text
e_k = (x_k, y_k, t_k, polarity_k)
```

理想版使用常数 `c_p=c`、无限带宽、无 refractory、无 leak、无噪声，并在两个连续仿真时刻之间线性求阈值交点，从而获得亚时间步的事件时刻。真实版使用当前工程的 v2e 模型加入阈值失配、强度相关低通、leak、shot noise 和 refractory。

### 4.8 为什么本系统的事件梯度算子不同于论文的简单形式

NeuroSR 在局部平移、成像算子与平移可交换时使用

$$
\frac{\partial L}{\partial t}\approx-\nabla L\cdot\mathbf v.
$$

在我们的光纤系统中，固定纤芯孔径采样一般不与物体平移交换。正确的一阶事件算子应写成完整前向模型对运动的 Jacobian：

$$
\frac{d}{dt}\log\mathcal H_{\Theta}(O,\mathbf s(t))
=
\frac{\partial\log\mathcal H_{\Theta}}{\partial\mathbf s}
\dot{\mathbf s}(t).
$$

展开到每根纤芯，它测量的是经过远端 PSF 和芯孔径积分后的物体梯度，沿运动方向的投影。这个模型仍然属于论文的通用 `I=f_Θ(O)` 框架，只是 `f_Θ` 不再是 unity/普通 relay，而是多芯光纤算子。

## 5. 与 NeuroSR 对应的重建模型

### 5.1 论文方法的核心

论文的事件触发模型为 `ΔL=p c`；在短时间、亮度恒定条件下，事件对应运动方向上的 log-intensity 梯度。事件经运动补偿并以可微 Gaussian kernel splat 后形成 IWE：

$$
\operatorname{IWE}(\mathbf x;\mathbf v)
=\sum_k p_k\kappa_{\Sigma}
\left(\mathbf x-\mathbf x'_k(\mathbf v)\right).
$$

缩放后的事件观测近似满足

$$
b_E=\frac{c}{\Delta t}\operatorname{IWE}
\approx-\nabla L\cdot\mathbf v+n_E.
$$

frame 则是同一运动轨迹上的强度曝光积分、再进行传感器采样。论文强调 frame 积分与 event warping 必须共享同一条运动轨迹。论文补充材料还给出了实际的两阶段做法：先用 IWE contrast maximization 估计运动，再固定/初始化运动进行强度反演，最后联合细化。[NeuroSR 补充材料](https://assets-eu.researchsquare.com/files/rs-9045382/v1/2eeb1dbf83797e150aa127be.pdf)

### 5.2 推荐的 fibre-aware IWE

普通 IWE 把事件的传感器坐标按整幅图像运动回参考时刻。对固定近端纤芯蜂窝，这样做并不正确。推荐先把近端事件关联到纤芯 `i(k)`，再把它映射到远端物体采样坐标：

$$
\mathbf u_k=\mathbf r_{i(k)}^d-\mathbf s(t_k),
$$

然后在物体坐标网格上 splat：

$$
\operatorname{IWE}_{fiber}(\mathbf u;\mathbf s)
=\sum_k p_k w_k
\kappa_{\Sigma}
\left(\mathbf u-\mathbf u_k\right).
$$

`w_k` 用于处理同一纤芯亮斑覆盖多个传感器像素所产生的冗余。第一版可以直接设“一根纤芯对应一个虚拟探测器”，此时 `i(k)` 已知；传感器级版本再通过标定的近端 core mask 把事件分配给纤芯。

### 5.3 联合目标函数

推荐的第一版目标为

$$
\begin{aligned}
O^*,\mathbf s^*=
\arg\min_{O\ge0,\mathbf s}\;&
D_F\left(
\mathbf y,
\mathcal F\left[\mathcal H_{\Theta}(O,\mathbf s(t))\right]
\right)\\
&+\alpha D_E\left(
b_E(\mathbf s),
\frac{d}{dt}\log\mathcal H_{\Theta}(O,\mathbf s(t))
\right)\\
&+\beta_{TV}\operatorname{TV}(O)
+\beta_s R_s(\mathbf s).
\end{aligned}
$$

其中：

- `F` 是曝光积分、像素积分和 frame noise/quantization 的模型；
- `b_E` 是 fibre-aware IWE 或按时间窗累积的 signed event count；
- `D_F` 在无噪声/Gaussian 情况先用 L2，低光 photon-limited 情况再用 Poisson NLL；
- `D_E` 第一版用带 event-count 方差权重的 L2；
- `TV(O)` 抑制噪声但不能过强，否则会抹掉正要恢复的细纤维；后续可换 Hessian/Huber-TV、PnP 或学习先验；
- `R_s` 约束轨迹速度和加速度平滑。

为更严格匹配真实事件触发，后续可绕过 IWE 线性近似，直接比较预测 log-intensity 增量与每个像素的事件阈值状态，或构建事件 timestamp likelihood。但第一版宜先复现论文的累积 Gaussian 近似。

### 5.4 建议的求解顺序

1. **Oracle motion**：直接使用仿真真值 `s(t)`，只优化 `O`。先证明光纤前向与伴随/自动微分正确、frame+event 确实能去蜂窝。
2. **Event-only motion initialization**：将轨迹参数化为少量时间节点间的 piecewise-linear 位移/速度，最大化 fibre-aware IWE 方差并加入速度平滑。
3. **固定运动重建 O**：用 frame+event 联合损失重建强度。
4. **联合细化**：小学习率交替优化 `O` 与 `s(t)`。
5. **系统参数细化**：最后才允许少数 `Θ`（全局 gain、PSF 宽度、阈值、微小 core-map affine）参与优化；一开始同时优化所有参数会出现严重的尺度和运动歧义。

### 5.5 为什么这样会得到无蜂窝结果

潜变量 `O` 的网格中没有蜂窝 mask。每次迭代时，预测 frame/events 都必须把 `O` 通过已知的纤芯阵列 `H_Θ` 投影回观测域。只要轨迹覆盖了足够多的亚纤芯相位，优化器会用同一个平滑/稀疏结构 `O` 解释不同时刻、不同纤芯的观测，而不能把固定蜂窝直接复制到 `O` 中。

如果重建代码把近端蜂窝图直接当成论文中的 latent `I`，那么最终只会得到更清晰的蜂窝图，而不是远端无蜂窝物体。这是实现时最重要的验收点。

## 6. 端到端仿真所需 input 数据

### 6.1 第一阶段必须具备的数据

#### A. 无蜂窝高分辨率真值图像

每个样本至少包含：

```text
object.tif / object.npy     线性、非负、最好 float32 或 16 bit
object_meta.yaml            µm/px、强度归一化、样本类型、裁剪位置
```

建议三类对象同时准备：

- 几何/数值单元测试：单点、单线、斜边、正弦条纹、Siemens star、随机点阵；
- 分辨率测试：工程已有的 `分辨率板.png`，但要为它定义明确的 `µm/px` 并缩放出跨越 1–10 µm 的线宽；
- 目标域图像：真实或公开的无蜂窝显微图、细胞/组织图、细纤维和交叉纤维结构。

不要只用 USAF 板；它对验证 MTF 很好，但不能代表细纤维交叉、低对比纹理和非均匀亮度。

#### B. 纤芯几何与传输参数

建议保存为 `fibre_map.npz`：

```text
distal_xy_um      [N,2] 远端芯中心
proximal_xy_um    [N,2] 近端芯中心
diameter_um       [N]   芯直径
gain              [N]   每芯透过增益
active            [N]   是否有效
crosstalk         sparse 邻芯耦合矩阵或邻接表
```

第一版由程序生成理想 hex lattice；以后用真实均匀照明图提取 proximal map，用远端扫描点/标定图估计 distal-to-proximal 对应、gain、PSF 和串扰。

#### C. 连续运动轨迹

建议 `motion.csv`：

```text
t_s, dx_um, dy_um
0.0000, ...
0.0001, ...
...
```

轨迹必须是连续时间定义，仿真时可任意采样。除 `dx,dy` 外，元数据还应保存轨迹类型、随机种子、速度/加速度上限以及物体到纤芯端面的放大率。

#### D. 光学与传感器配置

建议一个版本化 `config.yaml`，至少包含：

```text
distal_psf / NA / wavelength
proximal_psf
relay_magnification
sensor_height, sensor_width, sensor_pixel_pitch_um
frame_fps, exposure_s, bit_depth, full_well
event_pos_threshold, event_neg_threshold, threshold_sigma
event_cutoff_hz, refractory_s, leak_rate_hz, shot_noise_rate_hz
random_seeds
```

### 6.2 第二阶段再加入的数据

- 远端与近端的实际 core center map；
- 每芯 gain、暗芯、异常芯和每芯 PSF；
- 光纤弯曲状态及重复性；
- 真实 illumination flat-field；
- 真实传感器的 APS response curve、black level、gain、read noise、hot pixels；
- event threshold map、bias 设置、refractory 和时钟偏移；
- frame 与 event 的硬件时间戳对应关系；
- 镜头畸变和 fibre-to-sensor homography/非刚性映射。

### 6.3 每次仿真必须保存的 output

建议统一保存到 HDF5：

```text
/truth/object                  高分辨率无蜂窝真值
/truth/motion                  连续轨迹节点
/truth/fibre_map               本次使用的纤芯参数
/frames/data                   [M,H,W] 线性 APS frame
/frames/t_start, /frames/t_end 每帧曝光窗
/events/data                   [N,4]，推荐 [t_us,x,y,p]
/debug/core_signals            [T,Ncore]，第一阶段保留
/debug/dense_sensor_intensity  可选，调试后可删除以节省空间
/meta/config                   完整 YAML/JSON 配置
/meta/seeds                    所有随机种子
```

`core_signals` 很重要：它可以直接判断问题出在物体到纤芯、纤芯到传感器，还是 event generation，而不必从最终事件反推。

## 7. 建议的第一版关键参数

下表中的值是**仿真起点，不是对真实器件的宣称**。凡是当前没有实测依据的参数，都应在后续做 sweep 和标定。

### 7.1 空间、光纤和光学参数

| 参数 | 第一版 baseline | 后续 sweep/说明 |
|---|---:|---|
| 有效图像圆 | 550 µm | 520, 550, 580 µm |
| 纤芯间距 | 4.5 µm | 4.28–4.6 µm |
| 芯直径 | 2.9 µm | 2.7–3.6 µm；先确认“芯直径”与公开资料中“单元直径”的定义差异 |
| 晶格 | 理想 hex | 加位置 jitter 与少量 lattice defect |
| 纤芯数量 | 由圆和间距实际生成，约 13.5k | 另做 count-matched 15k |
| 高分辨率 forward grid | 0.5 µm/px | 0.25、0.5、1.0 µm/px 做收敛性检查 |
| 重建 grid | 1.125 µm/px | 先做 2×：2.25 µm/px，再做 4×：1.125 µm/px |
| 远端图像尺寸 | 至少 650×650 µm | 550 µm FOV 外加运动和 PSF margin |
| 远端 PSF | 第一版 delta | 第二版 Gaussian FWHM 1–3 µm 或实测 |
| core aperture | 直径 2.9 µm 归一化圆盘 | Gaussian 只是消融，不应作为唯一模型 |
| 每芯 gain CV | 0% | 3%、5%、10% |
| 坏芯率 | 0% | 0.05%、0.1% stress test；厂商的 lattice defect 指标不一定等同于坏芯率 |
| 位置 jitter | 0 | σ=0.1、0.2、0.3 µm |
| 最近邻串扰 | 0 | 总能量 0.5%、1%、3%，仅作敏感性扫描 |

推荐先在“每芯一个虚拟探测器”的 core-domain 做闭环。它省去近端 PSF 和相机采样，能最快确认去蜂窝的核心可辨识性。确认成功后才加入 346×260 传感器图像形成。

### 7.2 若按论文 DAVIS346 建立传感器级仿真

论文使用 DAVIS346：`346×260`、pixel pitch `18.5 µm`，APS 最高约 40 fps，frame/event 共焦平面。该参数来自论文实验配置，不代表最终一定要购买或使用同一器件。

若完整 550 µm 有效圆要放进 260 像素短边，继像最大放大率约为

$$
M_r\le\frac{260\times18.5}{550}\approx8.75.
$$

推荐第一版取 `M_r=8.3–8.5`：

- 有效图像圆直径约占 247–253 sensor pixels；
- 光纤端面等效 sensor sampling 约为 `18.5/M_r = 2.18–2.23 µm/px`；
- 4.5 µm 芯间距约对应 2.0–2.1 sensor pixels；
- 2.9 µm 芯直径约对应 1.3 sensor pixels。

这能容纳完整有效区，但每个纤芯只被少量像素采样。若以后希望精确分割每根纤芯亮斑，要么裁小 FOV、提高放大率，要么使用像素更小/分辨率更高的 event-frame sensor。

### 7.3 运动参数

| 参数 | 第一版 baseline | 建议 sweep/理由 |
|---|---:|---|
| 时间窗 `Δt_window` | 100 ms | 50、100、200 ms |
| 轨迹 | 2D 平滑 Lissajous 或圆角 raster | 必须包含两个方向 |
| 总覆盖范围 | x/y 各约 1–2 个 pitch，即 4.5–9 µm | 低于 0.5 pitch 信息覆盖不足；过大则边界损失和模型误差增大 |
| 速度 | 50–200 µm/s（光纤端面坐标） | 结合 event rate 扫描 20–500 µm/s |
| 轨迹节点间隔 | 5–10 ms | 重建使用 piecewise-linear nodes |
| forward 时间步 | 0.1 ms | 0.05、0.1、0.2 ms 收敛检查；最好用阈值交点得到精确事件时刻 |

规则六角晶格的基向量可取

$$
\mathbf a_1=(p,0),\qquad
\mathbf a_2=(p/2,\sqrt3p/2).
$$

若先做离散相位单元测试：

- 2× SR 使用 `m a1/2 + n a2/2, m,n∈{0,1}` 的 4 个相位；
- 4× SR 使用 `m a1/4 + n a2/4, m,n∈{0,1,2,3}` 的 16 个相位；
- 真正的 frame-event 仿真再用穿过这些相位的连续平滑轨迹。

只沿 x 匀速运动会使与 x 方向正交的梯度约束不足，因此不应作为最终轨迹，只能作为程序单元测试。

### 7.4 frame 参数

| 参数 | 第一版 baseline | sweep |
|---|---:|---|
| APS frame rate | 25 fps | 10、20、25、40 fps |
| exposure | 20 ms | 5、10、20、40 ms；不得大于 frame period |
| bit depth | float32/no quantization | 10、12 bit |
| peak electrons / 20 ms | 无噪声 | 100、1k、10k e⁻ 做低/中/高光照 |
| read noise | 0 e⁻ | 1、2、5 e⁻ RMS |
| saturation | 关闭 | 按 full-well 加 clipping |

100 ms 窗口、25 fps 会提供约 2–3 个 frame anchor。第一版不必追求“只有一个 frame”；先证明联合观测有效，再逐步减少到单帧/更低 frame rate。

### 7.5 event 参数

| 参数 | ideal baseline | realistic baseline / sweep |
|---|---:|---:|
| ON/OFF threshold | `c+=c-=0.2` log unit | 0.10、0.15、0.20、0.25、0.30 |
| threshold mismatch σ | 0 | 0.03 log unit，另扫 0.01–0.05 |
| photoreceptor cutoff | 0/infinite | v2e 起点 300 Hz，并验证时间步足够小 |
| refractory | 0 | 0.5 ms，另扫 0–2 ms |
| leak rate | 0 | v2e 起点 0.01 Hz/pixel |
| shot-noise event rate | 0 | v2e 起点 0.001 Hz/pixel；低光应使用更物理的 photoreceptor noise 模型 |
| timestamp unit | 1 µs 存储 | 数值仿真步长可以较大，但阈值交叉时刻应插值 |

`c=0.2` 表示约 20% 量级的 log-contrast 变化才触发事件。若目标低对比且运动很小，事件可能太少；若 `c` 太低，噪声和事件量会急剧上升。参数选择应以事件统计诊断为准：

- 事件是否覆盖大部分有结构的纤芯，而非只在少数极强边缘出现；
- ON/OFF 是否大致平衡，明显偏置是否来自照明漂移；
- 单芯事件是否因同一近端亮斑跨多个 sensor pixels 而严重重复；
- event rate 是否因阈值太低、步长太大或不连续运动产生瞬时爆发；
- 用事件重放得到的累计 log change 是否与真值 core signal 一致。

### 7.6 建议先固定的随机种子

至少分别保存：

```text
seed_object
seed_core_geometry
seed_core_gain_defect
seed_motion
seed_frame_noise
seed_event_threshold
seed_event_noise
```

不要只保存一个总 seed，否则日后改变一个模块会让其余随机量全部变化，难以定位性能变化来源。

## 8. 建议的软件模块划分

建议新建独立于 `v2e/` 的包，避免直接把多芯光纤逻辑塞进已有 v2e：

```text
fibre_neurosr/
  configs/
    figh_15_600n_ideal.yaml
    davis346_baseline.yaml
  data/
    objects/
    fibre_maps/
    trajectories/
  forward/
    object_motion.py
    core_lattice.py
    fibre_operator.py
    relay_sensor.py
    frame_camera.py
    event_camera.py
  reconstruction/
    core_iwe.py
    motion_estimation.py
    joint_loss.py
    optimize.py
  evaluation/
    metrics.py
    mtf_usaf.py
    honeycomb_spectrum.py
  scripts/
    simulate.py
    reconstruct_oracle.py
    reconstruct_joint.py
    run_ablation.py
  tests/
```

推荐用 PyTorch 实现 `H_Θ`、frame integration 和 fibre-aware IWE，使其对 `O`、轨迹节点和少量系统参数可微。理想事件生成含硬阈值，可以作为数据生成器不可微；重建侧使用论文的可微累计事件近似即可。

高层伪代码：

```python
O_gt = load_linear_object(...)
core_map = build_hex_fibre(...)
motion = build_2d_trajectory(...)

# 同一个连续前向强度
for t in dense_times:
    core_signal[t] = distal_core_sampling(O_gt, core_map, motion(t))
    sensor_intensity[t] = proximal_relay(core_signal[t], core_map, sensor_cfg)

frames = aps_integrate_and_noise(sensor_intensity, exposure_windows)
events = event_trigger(sensor_intensity, dense_times, event_cfg)
save_all_truth_and_observations(...)

# 重建
motion0 = oracle_motion_or_core_iwe_contrast(events)
O0 = initialize_from_frames(frames, core_map, motion0)
O_hat = optimize_object(frame_loss + event_loss + regularizer)
O_hat, motion_hat = joint_refine(O_hat, motion0)
```

## 9. 分阶段实验计划

### P0：数值算子与坐标单元测试

目标不是看好看的图，而是排除坐标、积分和时间戳错误。

- 单点/单线经过单根圆盘芯的解析或高精度数值结果；
- hex lattice 的间距、芯数、有效圆和填充率检查；
- 静止物体不产生理想事件；
- 线性亮度 ramp 在恒速运动下产生可预测极性的事件；
- frame exposure 等于 dense intensity 的时间积分；
- 位移的正负号在 forward、event warp 和 reconstruction 中一致；
- 自动微分梯度与 finite difference 对照。

### P1：理想 core-domain、已知运动、无噪声

- 一根纤芯对应一个虚拟探测器；
- 理想 hex、无坏芯、无 gain mismatch、无串扰；
- 使用真实圆盘 core aperture；
- 已知 2D 轨迹；
- 比较 frame-only、event-only、frame+event；
- 先做 2×，再做 4× 重建。

这是最重要的 go/no-go 阶段。如果在 oracle motion、无噪声、同模型条件下都不能优于 frame-only 或不能抑制蜂窝，不应立即添加更真实的噪声，而应检查可辨识性、IWE 映射和损失函数。

### P2：理想 core-domain、未知运动

- 用 fibre-aware IWE contrast maximization 估计 piecewise-linear motion；
- 分别报告 oracle motion 与 estimated motion；
- 先固定运动重建，再做小范围联合细化；
- 检查单方向与二维轨迹的差异。

### P3：加入近端蜂窝和 DAVIS 采样

- 将 core signal 渲染成固定近端亮斑；
- 加入 relay PSF、sensor pixel integration；
- frame 和 event 都在 346×260 同一虚拟焦平面产生；
- 实现 event-to-core assignment 或直接使用全传感器 `H_Θ` 的事件损失；
- 检查同芯多像素事件冗余。

### P4：真实事件效应和光子噪声

- v2e 的 threshold mismatch、finite bandwidth、leak、shot noise、refractory；
- APS Poisson-Gaussian noise、量化、饱和；
- core gain、位置 jitter、坏芯和串扰；
- 光照不均匀和缓慢漂移；
- 对事件阈值和 frame 光子数做二维 sweep。

### P5：模型失配与“反演犯罪”检查

若仿真和重建完全使用同一离散算子，成功只说明代码闭环正确，不代表真实系统会成功。至少做以下 mismatch：

- forward 用圆盘+像素精确积分，inverse 用略有误差的 Gaussian/标称直径；
- forward core positions 有 jitter，inverse 使用理想 hex 或带测量误差的 core map；
- forward 使用每像素 threshold map，inverse 只用全局平均 threshold；
- forward 轨迹含高频扰动，inverse 只用少量 piecewise-linear nodes；
- forward PSF 空间变化，inverse 使用空间不变 PSF；
- forward gain 漂移，inverse 使用固定 flat-field。

### P6：真实实验迁移

真实采集前至少需要：

1. 均匀照明 flat-field 提取近端 core centers、gain、坏芯和亮斑 PSF；
2. USAF/点扫描标定远端尺度、畸变、有效 PSF 和 distal-to-proximal 映射；
3. 标定 event-frame 共焦平面内的 frame/event 对齐和时间偏移；
4. 标定 event thresholds/biases，保存所有相机 bias；
5. 用可控二维压电/位移台生成已知微位移轨迹，先做 oracle-like 实验；
6. 再过渡到由事件自行估计运动。

## 10. 必做对照、指标与验收标准

### 10.1 对照组

每组实验至少包含：

1. 单张原始 fibre frame；
2. core-center interpolation / Voronoi / Gaussian interpolation；
3. frame-only MAP；
4. event-only reconstruction；
5. frame+event，oracle motion；
6. frame+event，estimated motion；
7. 可选：多帧 MFSR，使用相同时间和光子预算。

### 10.2 图像与物理指标

- PSNR、SSIM；LPIPS 仅作感知辅助，不能取代物理分辨率指标；
- USAF 可分辨 group/element、edge spread function、line spread function 和 MTF50/MTF10；
- 细纤维宽度、交叉点结构和连通性；
- honeycomb suppression：六角晶格对应频率峰值能量/邻域背景能量；
- 光度误差：重建均值、动态范围和局部对比度是否由 frame anchor 恢复；
- forward consistency：重建结果重新生成的 frame/event 与观测之间的残差。

### 10.3 运动与事件指标

- trajectory endpoint error / 位移 RMSE，单位 µm；
- IWE variance/锐度随迭代是否改善；
- event count、ON/OFF 比、活跃芯比例、每芯事件分布；
- predicted event count/polarity 与观测的一致性；
- 不同 accumulation window 下的稳定性。

### 10.4 推荐的阶段验收逻辑

- P0：数值误差和 finite-difference 梯度通过；
- P1：frame+event 明显优于 frame-only，并且蜂窝谱峰明显下降；
- P2：estimated-motion 结果接近 oracle-motion，失败时能由轨迹误差解释；
- P3：加入完整传感器采样后仍保持增益，而不是靠访问隐藏的 core truth；
- P4：性能随噪声/阈值平滑退化，没有突然崩溃或异常事件爆发；
- P5：合理模型失配下仍优于 frame-only，才值得进入真实实验。

不要预先把某个 PSNR 数字当成物理成功标准。更重要的是：相同观测预算下，联合方法是否稳定优于所有合理基线，以及恢复频率是否在纤芯孔径/光学 MTF 可支持的范围内。

## 11. 主要风险和避免办法

### 11.1 运动位置不对

若只是让近端蜂窝图在 event sensor 上平移，NeuroSR 可能把蜂窝本身超分，却不能恢复远端芯间信息。第一版应让物体像相对**远端纤芯阵列**运动。

### 11.2 轨迹只提供单方向梯度

`-∇L·v` 只观测沿 `v` 的梯度。采用二维闭合/Lissajous/raster 轨迹，或分段改变运动方向。

### 11.3 芯孔径造成不可逆低通

多次位移能填补采样空隙，但不能恢复 core aperture、NA 和远端 PSF 已经完全抹掉的频率。4× 网格只是数值网格，不等于真实 4× 光学分辨率。先用 2× 作为主要里程碑。

### 11.4 事件太少或只来自蜂窝边缘

在固定近端蜂窝中，几何边缘不随物体运动；理想情况下事件来自每芯输出强度变化。但近端 PSF、背景漂移和相机运动可能使固定蜂窝边缘产生伪事件。应固定光纤到传感器的机械关系，并分别查看 core-domain 和 sensor-domain events。

### 11.5 同一纤芯覆盖多个 event pixels

这些 pixels 不是独立的物体空间样本。若直接全部 splat，会对某些纤芯重复加权。可按 core mask 合并事件、归一化每芯权重，或在完整 sensor likelihood 中显式使用 `P_i`。

### 11.6 log 域与暗区不稳定

必须使用 `log(I+I0)`，保存 `I0`，并在低光下加入 photoreceptor bandwidth/noise。不要对零值直接取 log。

### 11.7 frame 与 event 使用了不同的运动或时间窗

两条观测必须从同一个 `I(t)` 产生并共享轨迹。frame timestamp 最好保存曝光起止，而不只保存一个中心时间。

### 11.8 过早同时优化所有未知量

`O`、motion、core gain、PSF、threshold 和 illumination 之间存在尺度与形状歧义。按“已知 motion/Θ → 未知 motion → 少量 Θ”逐级开放参数。

### 11.9 论文状态与实现缺失

截至本文日期，工程内只有论文和 v2e，没有 NeuroSR 作者代码；公开检索也未发现与该预印本明确对应的官方实现。论文是 2026 年预印本，因此我们需要依据主文和补充材料自行实现、做消融并验证，而不能把论文结果当成已经可直接复现的软件保证。

## 12. 推荐立即开始的最小闭环

第一轮不要直接上 15000 cores、346×260 sensor 和全部 v2e 噪声。最小而物理正确的版本如下：

1. 使用 128×128 µm 局部 FOV，0.5 µm forward grid；
2. 取其中约 800–1000 根理想 hex cores，`p=4.5 µm, d=2.9 µm`；
3. 一根 core 对应一个虚拟强度探测器；
4. 选一张线性无蜂窝 USAF/纤维图；
5. 生成 100 ms 二维轨迹，覆盖约 4.5–9 µm，时间步 0.1 ms；
6. 生成 25 fps、20 ms exposure 的 frame anchors；
7. 生成 `c=0.2`、无噪声理想 events；
8. 先用真值 motion 重建 2× 的 `O`；
9. 比较 frame-only 与 frame+event，并计算 honeycomb 频谱和 MTF；
10. 成功后扩到完整 550 µm，再增加 motion estimation 和传感器级 forward。

这个最小闭环能在较低显存和计算量下回答最关键的问题：**事件记录的连续 core intensity 变化，结合低速绝对强度 frame，是否足以在已知二维运动和有限芯孔径条件下恢复无蜂窝物体。**

## 13. 最终建议

这项工作的正确技术主线应定义为：

```text
无蜂窝高分辨率物体 O
    ↓ 远端光学与连续二维相对运动 s(t)
固定多芯阵列的有限孔径积分、gain、坏芯、串扰
    ↓ 近端芯斑与继像系统
共享的连续传感器强度 I(t)=HΘ(O,s(t))
    ├─ 曝光积分 + APS 噪声 → frames（绝对强度/低频锚）
    └─ log + 阈值触发 + DVS 动态 → events（时间密集的差分约束）

frames + events + fibre-aware HΘ
    ↓ oracle motion → event motion initialization → joint refinement
高分辨率、无蜂窝的 O
```

仿真阶段前向模型与重建模型一致是必要的第一步，但最终可信度来自三层验证：

1. exact-model 无噪声闭环验证实现；
2. noise 与 model-mismatch 验证稳健性；
3. real calibration 与真实采集验证物理适用性。

只要按这个顺序推进，仿真不会变成单纯“生成看起来像事件的数据”，而会真正成为后续真实系统设计、参数选择和重建算法的数字孪生基础。

## 14. 主要参考资料

- Wang et al., *Neuromorphic High-Throughput Imaging via Transport-Induced Differential Sensing*, 2026 preprint：[公开主文 PDF](https://assets-eu.researchsquare.com/files/rs-9045382/v1_covered_00b743f5-7e9b-4452-8a3b-8781b8dfff96.pdf)
- 同文补充材料：[公开 Supplementary Information PDF](https://assets-eu.researchsquare.com/files/rs-9045382/v1/2eeb1dbf83797e150aa127be.pdf)
- Hu, Liu, Delbruck, *v2e: From Video Frames to Realistic DVS Events*, CVPRW 2021：[arXiv](https://arxiv.org/abs/2006.07722)
- Fujikura, *Image Fiber FIGH series N type*：[官方产品数据表](https://www.optic-product.fujikura.com/wp-content/uploads/2024/07/FPE29A.pdf)
- Fujikura FIGH series 产品页：[官方页面](https://www.optic-product.fujikura.com/specialty-fibers/image-fiber-figh-series/)

