# `isl_diff_event` 代码解读与多芯光纤重建适用性报告

生成时间：2026-08-02 16:04:28 HKT  
审查目录：`/home/robbie/tyf_code/EventCode/myFEFibreSR/isl_diff_event`  
代码快照：Git commit `361e3d3`，审查时工作区干净  
默认 Python：`/home/robbie/miniconda3/envs/NeuroFibreSR/bin/python`（Python 3.10.16）

## 1. 先给结论

`isl_diff_event` **可以作为我们后续重建代码的算法参考和局部工具来源，但目前不能直接作为多芯光纤重建程序使用**。

更准确地说：

- 它已经包含论文 NeuroSR 方法中比较有价值的骨架：事件运动补偿、IWE（Image of Warped Events）、基于对比度的运动估计、分段线性轨迹、事件约束、APS 帧约束，以及用自动微分联合优化潜在图像和运动。
- 它的主模型仍是“普通相机观察运动图像”的模型：场景在传感器平面平移，APS 是潜在图像经过运动模糊和降采样得到的。
- 我们的问题中，物体运动时，**多芯光纤输出端的纤芯光斑位置是固定的，变化的是各纤芯承载的强度**。因此，不能把传感器上的事件坐标直接按物体位移平移。
- 代码没有 FIGH-15-600N 的纤芯孔径采样、固定纤芯光斑渲染、relay 成像、像素积分和纤芯标定模型。没有这部分，就无法让重建结果正确消除蜂窝结构。
- 代码还没有可直接运行的配置、测试数据、权重和命令行入口，当前环境中也存在缺失依赖与一个语法错误。

因此建议把它定位为：

> **NeuroSR 优化原型库，而不是可直接执行的多芯光纤重建基线。**

综合判断：

| 项目 | 判断 |
| --- | --- |
| 论文方法复现骨架 | 基本具备 |
| 可复用的事件/IWE/轨迹代码 | 有，价值较高 |
| 直接读取我们当前仿真数据 | 不支持 |
| 直接运行现有 demo | 不可行 |
| 多芯光纤物理前向模型 | 缺失，是最关键缺口 |
| 直接得到无蜂窝、高分辨率光纤图像 | 不能 |
| 经过结构化改造后作为基础 | 可以 |

## 2. 论文方法的核心，以及代码实现到了哪里

论文的联合重建思想不是简单地把事件累积图和 APS 帧送入一个神经网络，而是建立同一个潜在图像和同一条运动轨迹下的两个观测模型：

1. 事件分支：用运动把事件变换到统一参考时刻，形成运动补偿后的 IWE；再约束潜在对数强度梯度与 IWE 一致。
2. 帧分支：沿相同运动轨迹对潜在图像积分，得到曝光期间的运动模糊图像，再经过传感器模型得到 APS 帧。
3. 联合优化：潜在图像、背景和运动参数共同更新，并加入 TV、轨迹平滑等正则项。

普通相机条件下，代码所使用的近似关系可写成：

```text
事件：  b_E ≈ -∇log(I) · v
APS：   B ≈ D[ (1/T) ∫ W_s(t)(I) dt ]
```

其中 `W_s(t)` 是由运动轨迹决定的图像平移，`D` 是相机降采样或像素积分。

`isl_diff_event` 对这套流程的覆盖情况如下：

| 论文环节 | 代码位置 | 完成情况 |
| --- | --- | --- |
| 事件读取与时间窗选择 | `utils/utility.py`、各 demo | 有，但接口不统一 |
| 双线性/高斯事件 splatting | `utils/utility.py` | 有 |
| 事件运动补偿与 IWE | `NeuroSRM_demo.py`、`solver/dense_iwe_reconstruction.py` | 有 |
| 对比度最大化估计运动 | `NeuroSRM_demo.py` | 有 |
| 分段线性轨迹 | `trajectory_model/Diff_tratrectory.py` | 有 |
| APS 运动模糊前向模型 | `NeuroSRM_demo.py` | 有，但只是平移卷积/降采样 |
| 事件和 APS 联合损失 | `NeuroSRM_demo.py` | 有 |
| 二倍空间网格优化 | `NeuroSRM_demo.py` | 有，但其“二倍”是传感器像素网格，不是纤芯采样网格 |
| 多芯光纤前向算子 | 无 | 完全缺失 |
| 可复现训练/推理入口 | 无 | 缺失 |

## 3. 目录和主要文件的实际作用

### 3.1 `NeuroSRM_demo.py`

这是最接近论文 NeuroSR 联合重建的主脚本，也是我们最值得参考的文件。

它的实际流程是：

```text
AEDAT4 事件/APS
    ↓
选择一个 APS 曝光时间窗
    ↓
用事件窗的配准给运动初值
    ↓
分段线性轨迹 + IWE 对比度最大化
    ↓
优化低分辨率潜在图像、背景、运动
    ↓
再在 M=2 的网格上继续优化
```

主要特点：

- 传感器尺寸在脚本内固定为 `260 × 346`。
- 运动使用 12 段分段线性轨迹。
- 先进行约 2100 次运动优化，再进行约 2100 次图像/运动联合优化；后面还有一轮 `M=2` 优化。
- 事件损失使用潜在对数强度图的梯度、位移和运动补偿 IWE。
- APS 损失使用由轨迹生成的全局运动模糊核，再降采样到观察尺寸。
- `frames_sta` 主要用于显示或参考；真正的数据项主要使用运动 APS 帧 `frames_mov`。

它的问题也很明显：

- 整个文件是 1261 行的顶层交互脚本，导入文件就会执行。
- 输入路径、输出路径和时间窗均硬编码为作者本机的 Windows 路径。
- 没有 `argparse`、配置文件或稳定的函数级 API。
- 数据加载有重复代码，实验阶段代码和绘图代码混杂。
- `M=2` 表示 DAVIS 传感器网格上放大两倍；对于我们的光纤问题，这不是正确的超分辨率定义。

### 3.2 `NeuroSRM_MEDI_demo.py`

这个脚本实现的是 EDI/MEDI 风格的事件辅助时间去模糊或视频恢复，并估计事件对比度阈值。它不等价于空间 NeuroSR。

适用范围：

- 多个 APS 帧；
- 需要估计真实事件相机的正、负阈值；
- 需要做时间维度上的中间帧恢复。

对我们当前第一阶段仿真的价值有限：

- 当前只有一个 APS 曝光结果。
- v2e 的理想阈值已知为 `0.2`。
- 它仍然是在固定传感器像素上积累事件，没有多芯光纤坐标映射。

以后进入真实实验、多 APS 帧和未知阈值阶段时，可以参考它的阈值标定思路，但不应该把它当成当前空间超分的主程序。

### 3.3 `rotation_demo.py`

这是一个事件流和给定 dense flow 的调试型 demo，不是完整的帧—事件联合重建程序。

- 输入 AEDAT4 和 `.npy` 光流路径均为硬编码 Windows 路径。
- 当前设置中 `frame_loss_weight = 0`，缺少 APS 光度锚点。
- 从随机潜在图像开始，仅依靠事件和 TV，很难确定绝对强度与低频结构。
- 最优潜在图像保存语句还被注释掉了。

所以这个 demo 只能用于检查事件 warp 和损失方向，不能作为我们的最终基线。

### 3.4 `solver/dense_iwe_reconstruction.py`

这是仓库中最值得直接抽取的模块之一，包含：

- 图像到对数强度的变换；
- dense flow 网格构造；
- flow mask；
- 从潜在图像与运动生成预测 IWE；
- IWE 的 L2/归一化损失。

这部分相对独立，适合经过单元测试后移入我们自己的重建包。

### 3.5 `trajectory_model/Diff_tratrectory.py`

这里实现了可微分分段线性位移轨迹。概念上可复用，但当前实现会用：

```text
torch.linspace(0, t_win, t_win + 1)
```

把微秒时间窗近似成逐微秒的稠密数组。25 ms 的窗口尚可接受，时间更长时会产生不必要的内存和计算开销。更合理的做法是只在事件时间戳、APS 积分采样点或所需查询时刻计算轨迹。

### 3.6 `utils/utility.py`

这是一个 3128 行的综合工具文件，混有：

- AEDAT4/HDF5/NPY 读取；
- 事件图生成和 splatting；
- 图像插值；
- 光学、绘图及其他实验函数。

它能提供原型代码，但不适合原封不动依赖。建议只抽取经过测试的少量函数，避免把无关的光学和硬件依赖带入新工程。

## 4. 与我们当前仿真输出的接口对照

当前仿真配置的重要参数为：

| 参数 | 当前值 |
| --- | --- |
| 输入物面像素 | `0.5 µm/pixel` |
| 物面运动 | `[180, 0] µm/s` |
| 时长 | `25 ms` |
| 位移 | `[4.5, 0] µm`，恰好一个纤芯间距 |
| 时间步长 | `0.1 ms`，共 251 个强度时刻 |
| 光纤区域 | `160 µm × 160 µm` |
| 纤芯间距/直径 | `4.5 µm / 2.9 µm` |
| 传感器尺寸 | `260 × 346` |
| APS 曝光 | `0–25 ms` |
| v2e 阈值 | 正负均为 `0.2` |
| 事件数 | 65,745 |

已有文件结构：

```text
03_fibre/fibre_sequence.h5
    core_centres_xy_um       (1415, 2)
    core_signals             (251, 1415)
    frames                   (251, 320, 320)
    timestamps_s             (251,)

04_sensor/sensor_sequence.h5
    frames                   (251, 260, 346)
    timestamps_s             (251,)

05_aps/aps_frame.npy         (260, 346), float32, [0, 1]

06_events/events.h5
    events_t_s_x_y_p         (65745, 4), p∈{-1,+1}
    events_t_us_x_y_p01      (65745, 4), p∈{0,1}
```

`isl_diff_event` 不能直接使用这些数据，原因包括：

1. 它的 HDF5 读取器只查找名称为 `events` 的数据集。
2. HDF5/NPY 分支没有把 APS、曝光起止时间和事件一起构造成统一样本。
3. 其 HDF5 读取逻辑会减去第一个事件时间戳。我们的第一个事件出现在约 `0.6 ms`，而 APS 曝光从 `0 ms` 开始；直接减去会破坏事件与曝光的共同时间原点。
4. 主脚本有 `I_lr / 255` 的固定归一化；我们的 APS 已经是 `[0,1]`，再除一次会导致强度几乎变为零。
5. 主脚本假定输入来自 AEDAT4，并在非 AEDAT4 分支中缺少完整 APS 数据流。

因此需要先定义一个明确的数据适配层，而不是修改数据去迁就 demo 的隐含假设。

建议的单样本输入结构为：

```text
Sample
├── aps_frame_float          [H, W], float32, [0,1]
├── exposure_start_us        标量
├── exposure_end_us          标量
├── events                   [N,4], [t_us,x,y,p±1]
├── sensor_shape             [H,W]
├── core_centres_sensor_xy   [Nc,2]
├── core_centres_object_um   [Nc,2]
├── core_aperture/diameter
├── relay_and_pixel_params
├── trajectory_truth         仿真阶段可用，真实阶段可空
└── object_truth             仅用于仿真评价，不进入优化
```

## 5. 最关键的物理不匹配

### 5.1 普通传感器运动模型

原代码默认场景图像直接在传感器上平移：

```text
I_sensor(x,t) = I_latent(x - s(t))
```

这样传感器上的事件位置也可以按同一个位移进行 warp。

### 5.2 多芯光纤的真实关系

多芯光纤输出端的纤芯光斑中心固定。物体移动时，各纤芯采到的物面位置变化，导致纤芯信号强度变化；光斑本身不会在传感器上跟着物体平移。

我们的前向模型应是：

```text
O_hr
  → 物体运动 W_s(t)
  → GRIN/输入光学 H_grin
  → 纤芯孔径积分 A_core
  → 固定输出纤芯光斑 R_core
  → relay/PSF H_relay
  → 传感器像素积分 P_sensor
  → I_sensor(t)
```

写成紧凑形式：

```text
I_sensor(t) = P_sensor · H_relay · R_core · A_core · H_grin · W_s(t)[O_hr]
```

然后两种观测都必须由同一个 `I_sensor(t)` 产生：

```text
APS = (1/T) ∫ I_sensor(t) dt
Events = log(I_sensor(t)) 的阈值越界事件
```

如果直接套用原代码，会把传感器上固定的蜂窝光斑错误地当成随场景移动的纹理。结果通常只能是传感器平面的去模糊或放大蜂窝图，而不是物面上的无蜂窝重建。

### 5.3 事件应该在哪个坐标系中使用

对我们的数据，建议分成两个层次：

1. **传感器域**：保留原始事件，用于验证完整前向模型能否解释传感器观测。
2. **纤芯/物面域**：利用已知的纤芯中心和输出光斑 footprint，把事件归属或软分配给纤芯；再把每个纤芯在时刻 `t` 采到的物面坐标 `c_i - s(t)` 放到高分辨率物面网格。

第二种方式计算量较小，更适合作为第一版重建；完整传感器域事件生成可以作为后续高保真版本。

## 6. “超分辨率倍率”必须重新定义

原代码的 `M=2` 是把 DAVIS 的 `346 × 260` 像素网格扩大两倍。这对多芯光纤并没有直接物理意义，因为传感器已经用多个像素观察一个固定纤芯光斑：当前纤芯间距在传感器上约为 `7.3125 pixel`，直径约为 `4.7125 pixel`。

我们真正要突破的是 **4.5 µm 的纤芯空间采样间距**，而不是 18.5 µm 传感器像素经过 relay 后的显示尺寸。

正确做法是直接定义物面重建网格，例如：

- 基线网格：接近纤芯采样极限，用于验证前向模型；
- 目标网格：`1.0 µm/pixel` 或 `0.5 µm/pixel`；
- 输出区域：与当前 `160 µm × 160 µm` 有效光纤视场一致。

这意味着输出尺寸可以是 `160 × 160` 或 `320 × 320`，而不是机械地使用 `692 × 520`。

## 7. 当前运动轨迹对二维超分的影响

当前轨迹在 25 ms 内只沿 x 方向移动 4.5 µm：

```text
s(t) = [180 t, 0] µm
```

优点：

- 总位移恰好覆盖一个纤芯间距；
- 251 个时刻可在 x 方向提供较密的亚纤芯采样；
- 很适合验证一维横向扫描是否能补充纤芯之间的信息。

局限：

- y 方向没有亚纤芯位移覆盖；
- 事件约束 `∇L · v` 在纯水平运动下主要观测 x 方向梯度；
- 因而水平与垂直分辨率提升会不对称；
- 对二维蜂窝采样缺口，单方向运动通常不能提供同等充分的二维条件。

所以第一版可以继续用当前轨迹做代码闭环，但报告结果时只能宣称“水平扫描条件下的重建”。后续要验证二维无蜂窝超分，应增加带 x/y 分量的轨迹，例如斜向、两段正交扫描或小幅二维 Lissajous/dither 轨迹。

## 8. 代码可运行性和工程质量检查

我使用指定的 `NeuroFibreSR` 环境对仓库中的 31 个 Python 文件做了语法解析和关键模块导入检查。

### 8.1 已确认的问题

- `utils/utils_viz.py:343` 存在 `IndentationError: unexpected indent`，该模块当前无法导入。
- `solver.flow_sr` 导入失败，当前环境缺少 `kornia`；继续执行还可能需要 `natsort` 等依赖。
- 当前环境还缺少 `scikit-learn`、`seaborn`、`piq`、`zernike`、`natsort` 等 `environment.yml` 中的依赖。
- `rotation_demo.py` 导入时就尝试打开硬编码 Windows AEDAT4 文件，因此无法作为模块导入。
- 仓库不包含 demo 所需的 AEDAT4、flow 文件、模型权重或可复现实验数据。
- 没有正式的单元测试/集成测试体系。
- 没有发现许可证文件；若后续大量复制代码，需确认上游授权。

### 8.2 不影响核心判断的情况

- `utils.utility`、`trajectory_model.Diff_tratrectory`、`solver.dense_iwe_reconstruction` 和 `models.network_unet` 可在当前环境单独导入。
- 主 NeuroSR 流程是逐样本优化，不依赖预训练 UNet 权重；仓库里的 UNet/PnP 代码不是当前主链路的必要部分。

### 8.3 其他值得修正的实现细节

- 逐微秒生成完整轨迹数组的方式应改成按查询时刻求值。
- `utils/utility.py` 中有事件图生成后强制清零最大像素的处理，应确认这是否只是作者的热像素补丁，不能直接用于我们的理想仿真。
- 图像、事件、绘图、硬件和光学工具混在同一文件，不适合成为稳定依赖。
- 代码没有统一约定 x/y、row/column、µm/pixel 和位移正负号，迁移到光纤坐标前必须用合成点目标做符号测试。

## 9. 哪些代码建议复用，哪些不要直接复用

### 9.1 建议复用思想并抽取实现

1. 双线性或高斯 splatting。
2. IWE 和对比度最大化目标。
3. 分段线性轨迹的参数化方式。
4. 事件梯度一致性损失。
5. APS、事件、TV 和轨迹正则的联合优化框架。
6. `solver/dense_iwe_reconstruction.py` 中相对独立的算子。

### 9.2 不建议原样复用

1. 三个顶层 demo 的数据读取、路径、绘图和循环控制。
2. 普通图像的全局 motion-blur kernel 作为我们的 APS 前向模型。
3. 在传感器像素平面直接按物体位移 warp 事件。
4. 把 `M=2` 当成光纤系统的二倍超分。
5. 当前混杂的 `utils/utility.py` 整体依赖。
6. 与本任务无关的 ptychography、holography、PnP 和 UNet 代码。

建议保持 `isl_diff_event` 上游目录不动，在主工程中另建结构清晰的重建包，通过少量、经过测试的移植函数吸收其思想。

## 10. 推荐的重建实现顺序

### R0：建立数据契约和坐标测试

输入：当前仿真输出、配置、真值轨迹和真值物图。  
输出：统一 `Sample`，以及物面—纤芯—传感器坐标往返测试。

必须通过：

- 时间原点和 APS 曝光严格一致；
- x/y 与 row/column 不交换；
- 位移正负号由单点目标验证；
- 从保存数据重新前向计算能复现 APS 和事件统计。

### R1：已知运动、只用 APS 的光纤逆问题

先固定仿真真值轨迹，不急于估计运动。实现可微分光纤前向算子，优化高分辨率物图，使其生成的 APS 与仿真 APS 一致。

目的不是马上得到最好图像，而是验证：

- 光纤采样和固定光斑模型是否正确；
- honeycomb 是否由前向模型解释，而不是被错误地当作物体纹理；
- 反向梯度是否稳定。

### R2：已知运动、加入事件约束

优先实现纤芯域事件版本：

1. 将传感器事件软分配给纤芯；
2. 根据已知轨迹，把纤芯样本放回物面坐标；
3. 形成纤芯域 IWE 或差分观测；
4. 与潜在物图的预测差分联合优化。

做三组消融：APS-only、Event-only、APS+Event。

### R3：再估计运动

在 R1/R2 正确后，再移植 `isl_diff_event` 的分段线性轨迹和对比度最大化。运动估计应在纤芯/物面坐标中完成，而不是直接平移固定蜂窝光斑。

### R4：真实实验适配

真实数据阶段再加入：

- 暗场、平场和纤芯增益；
- 纤芯中心、直径、缺芯与耦合标定；
- relay PSF 和畸变；
- 正负事件阈值分布、漏电、噪声和 refractory period；
- 多 APS 帧条件下的阈值校准，可参考 MEDI demo。

## 11. 建议的新重建模块边界

```text
fibre_reconstruction/
├── data/
│   ├── sample_schema.py
│   └── simulation_adapter.py
├── geometry/
│   ├── coordinates.py
│   └── core_assignment.py
├── forward/
│   ├── motion_warp.py
│   ├── core_sampler.py
│   ├── core_renderer.py
│   ├── sensor_model.py
│   ├── aps_operator.py
│   └── event_operator.py
├── motion/
│   └── piecewise_trajectory.py
├── losses/
│   ├── aps_loss.py
│   ├── event_loss.py
│   └── regularizers.py
├── solvers/
│   ├── reconstruct_known_motion.py
│   └── reconstruct_joint.py
├── metrics/
│   ├── image_metrics.py
│   └── honeycomb_metrics.py
└── tests/
    ├── test_coordinates.py
    ├── test_forward_adjoint.py
    ├── test_event_polarity.py
    ├── test_time_alignment.py
    └── test_tiny_inverse_problem.py
```

每个模块都应有明确输入输出，避免再次形成千行顶层脚本。

## 12. 第一轮重建应采用的评价指标

有仿真真值时，不应只看图像“好不好看”。至少记录：

1. 物图 PSNR、SSIM；LPIPS 可选。
2. USAF 横向和纵向可分辨组元，分别报告，避免掩盖一维运动造成的各向异性。
3. 蜂窝频率附近的频谱能量或 honeycomb suppression ratio。
4. APS reprojection residual：重建结果重新前向生成 APS 后与输入 APS 的误差。
5. Event/IWE residual：预测差分和事件观测的误差。
6. 已知运动阶段的轨迹固定检查；联合估计阶段再报告轨迹 RMSE。

最重要的验收规则是：

> 重建图既要接近物面真值，也必须重新生成正确的 APS 和事件。只得到一张视觉平滑图不算完成物理重建。

## 13. 最终判断和下一步

### 是否满足“进行重建的基础代码”？

答案分两层：

- **作为论文 NeuroSR 方法的研究原型：满足。** 它证明了事件运动补偿、APS 模糊前向和联合优化应该怎样组织。
- **作为我们的多芯光纤重建基础工程：尚不满足。** 数据契约、光纤前向模型、物面坐标超分定义、测试和可运行入口都需要重新建立。

推荐接下来的第一项实现不是直接运行 `NeuroSRM_demo.py`，也不是先调 2100 次迭代参数，而是：

> **用当前仿真真值轨迹，先实现并验证一个可微分的“物面高分辨率图 → 多芯光纤 → APS”前向/反向闭环。**

这个闭环正确后，再把 `isl_diff_event` 中的 IWE、事件损失和分段轨迹逐项接入。这样能够明确区分“光纤模型错误”“事件模型错误”和“优化器没有收敛”，也是目前风险最低、最可验证的路线。

## 附录 A：本次阅读的关键文件

- `isl_diff_event/NeuroSRM_demo.py`
- `isl_diff_event/NeuroSRM_MEDI_demo.py`
- `isl_diff_event/rotation_demo.py`
- `isl_diff_event/solver/dense_iwe_reconstruction.py`
- `isl_diff_event/solver/flow_sr.py`
- `isl_diff_event/trajectory_model/Diff_tratrectory.py`
- `isl_diff_event/utils/utility.py`
- `isl_diff_event/utils/utils_viz.py`
- `isl_diff_event/demos/extract_event_frame.py`
- `isl_diff_event/environment.yml`
- `Paper/Wang et al. - 1 Neuromorphic High-Throughput Imaging via 2 Trans.pdf`
- `Paper/supp - Neuromorphic High-Throughput Imaging.pdf`
- `fibre_frame_event_sim/configs/phase1_usaf.yaml`
- `fibre_frame_event_sim/outputs/phase1_usaf/03_fibre/fibre_sequence.h5`
- `fibre_frame_event_sim/outputs/phase1_usaf/04_sensor/sensor_sequence.h5`
- `fibre_frame_event_sim/outputs/phase1_usaf/05_aps/aps_frame.npy`
- `fibre_frame_event_sim/outputs/phase1_usaf/06_events/events.h5`

## 附录 B：本次审查边界

本报告是静态代码审查、关键模块导入检查、Python 语法检查，以及与当前仿真数据的接口/物理模型对照。由于仓库不含作者 demo 所需数据和权重，未声称复现作者论文中的最终数值结果；也未修改 `isl_diff_event` 的任何源文件。
