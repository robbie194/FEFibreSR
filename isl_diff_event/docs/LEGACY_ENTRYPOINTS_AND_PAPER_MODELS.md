# 旧目录入口脚本与论文光学模型说明

## 文件角色总览

| 文件 | 定位 | 是否为完整 NeuroSR 入口 |
|---|---|---|
| `NeuroSRM_demo.py` | 运动估计、IWE、运动模糊帧与潜在强度联合优化 | 是，当前主要实验入口 |
| `NeuroSRM_MEDI_demo.py` | 多曝光 EDI、物理事件阈值标定和连续潜在视频 | 独立实验支线，不是主 Demo 的自动前处理 |
| `rotation_demo.py` | 固定稠密旋转光流条件下验证事件 IWE 与图像梯度模型 | 否，是旋转场景诊断脚本 |
| `NeuroSRM_demo_backup_20260805_145000.py` | 旧版主 Demo 的历史快照 | 否，不应作为新入口 |
| `gpu_mem_track.py` | CUDA tensor 和显存占用诊断 | 否，与重建数学模型无关 |

## 1. `NeuroSRM_MEDI_demo.py`

### 什么是 EDI

EDI 通常指 Event-based Double Integral。事件相机在某像素累计对数亮度变化达到对比度阈值时触发事件。理想模型为：

$$
L(t)-L(t_r)=cE(t),
$$

其中 $L=\log I$，$c$ 是物理对比度阈值，$E(t)$ 是相对参考时刻累计的带符号事件数。因此：

$$
I(t)=I(t_r)\exp(cE(t)).
$$

APS 模糊帧是曝光期间瞬时强度的平均：

$$
B=\frac{1}{T}\int_0^T I(t)dt
 = I(t_r)\frac{1}{T}\int_0^T\exp(cE(t))dt.
$$

给定模糊帧、曝光内事件和阈值 $c$，就可以反推出参考时刻的潜在清晰图：

$$
I(t_r)=\frac{B}{\frac{1}{T}\int_0^T\exp(cE(t))dt}.
$$

代码把曝光窗口划分为时间 bin，以曝光中心为参考，分别向前、向后累计正负事件，形成离散的 centered double integral，然后恢复曝光中心潜在图以及整段潜在图像序列。

### 为什么需要标定对比度阈值

物理阈值 $c$ 决定“一个事件代表多少对数亮度变化”。它不是任意可视化缩放参数：

- $c$ 太小，事件对亮度变化的解释不足，EDI 去模糊偏弱；
- $c$ 太大，事件积分被过度放大，容易产生强烈伪影；
- 正负事件电路可能不完全对称，所以可以分别标定 $c_+$ 和 $c_-$。

传感器设置给出的 nominal threshold 不一定等于实际有效阈值，实际值还会受偏置配置、照明、噪声、温度和数据预处理影响。

### 当前脚本怎样标定

脚本执行以下步骤：

1. 读取完整 AEDAT4 事件、APS 帧和每张帧的真实曝光边界。
2. 按“曝光窗口内事件密度”自动选择一组连续 APS 帧。
3. 每张帧只使用自己真实曝光区间内的事件，默认分成 32 个时间 bin。
4. 多个曝光窗口共享同一个 $c$，或共享同一对 $(c_+,c_-)$。
5. 对每个候选 $c$，利用 EDI 反演曝光中心潜在图和曝光内潜在序列。
6. 使用相邻曝光中心之间的事件，把前一张潜在图传递到后一张，并计算 temporal transfer consistency。
7. 同时加入事件边缘一致性、空间 TV、时间 TV 和连续视频约束，优化共享阈值。
8. 保存阈值、收敛曲线、潜在图序列和连续视频。

需要特别注意：单个曝光中的 `blur_pred` 是由输入模糊帧按 EDI 闭式反演后再投影得到的，因此仅靠单窗 frame fidelity 很难唯一确定 $c$。当前实现中，真正让 $c$ 可辨识的主要信息来自跨窗口 transfer consistency、事件边缘项和时空正则项。

### 标定以后能做什么

标定结果写入：

```text
output/MEDI/contrast_params.json
```

它可以用于：

- EDI 去模糊和曝光内高帧率潜在视频重建；
- 把带符号事件计数转换为有物理尺度的对数亮度变化；
- 在支持 EDI forward blur 的联合优化中固定或初始化 $c$；
- 比较不同采集设置下事件通道响应是否发生漂移。

### 什么时候需要使用

适合使用 MEDI 的情况：

- 需要从 APS 模糊帧和曝光内事件直接恢复时间序列；
- 需要物理可解释的事件阈值，而不是归一化 IWE；
- 数据包含多张连续 APS 帧，且相邻窗口场景和运动保持连续；
- 计划使用逐像素 EDI 模糊模型，尤其是非均匀流或旋转运动，单一全局卷积核不合适时。

通常不必使用的情况：

- 只做 CMax 运动估计，目标只依赖 IWE 聚焦度；
- 事件损失和目标 IWE 都做了归一化，绝对事件幅值被消除；
- 没有足够连续 APS 帧，无法可靠建立跨窗口一致性；
- 当前目标只是严格复现已经跑通的 `NeuroSRM_demo.py` 输出。

### 在本项目中具体怎样用

当前脚本没有命令行参数，需要先修改文件顶部配置：

```python
read_path_e = "/path/to/data.aedat4"
save_dir = "output/MEDI"
calib_n_frames = 5
shared_contrast = True
```

然后从旧目录运行：

```bash
cd /home/robbie/tyf_code/EventCode/myFEFibreSR/isl_diff_event
/home/robbie/miniconda3/envs/NeuroFibreSR/bin/python NeuroSRM_MEDI_demo.py
```

运行后首先检查 `c_convergence.png`，确认阈值不是停在边界并且 loss 已稳定；再检查 `latent_sequence.png` 和视频是否存在闪烁、曝光窗口跳变或指数放大伪影。

### 对当前 NeuroSR 重建的实际作用

目前 `NeuroSRM_demo.py` 和 clean 版本都没有读取 `output/MEDI/contrast_params.json`，所以先运行 MEDI **不会自动改变**当前主 Demo 的结果。

`function/optimizer.py` 虽然保留了 `conv_type="medi"` 分支，但当前分支会导入仓库中不存在的 `utils.utils_calib`，而且主 Demo 实际选择的是普通 `conv` 分支。因此它只能看作尚未接通的实验接口，不能作为 MEDI 已经集成进主流程的依据。

此外，主 Demo 中的：

```python
thre = 1 + 1e-6
```

是 `lin_log` 函数的数值转折参数，不是 MEDI 标定的物理事件阈值 $c$，二者不能直接替换。

要让 MEDI 真正服务当前主流程，需要另行设计接入方式，例如：

- 在 EDI 模糊前向模型中读取并固定 $c$；
- 在保留物理幅值的事件损失中使用 $(c_+,c_-)$；
- 用 MEDI 恢复的曝光中心潜在图初始化 NeuroSR 的 `I_pred`。

这些接入都会改变当前算法和结果，不能在“保持 clean 与旧 Demo 输出一致”的任务中静默加入。

## 2. `rotation_demo.py`

### 它解决什么问题

平移运动可以用全局二维轨迹描述，但旋转运动在不同像素处的方向和幅度都不同，需要稠密光流场。`rotation_demo.py` 用来验证：给定一张稠密旋转光流后，事件能否被正确 warp，并且图像梯度模型能否重现相同 IWE。

它的流程是：

1. 选择一张 APS 帧及其真实曝光区间内的事件。
2. 从外部 `.npy` 文件读取 `[2,H,W]` 稠密旋转光流。
3. 在 `xy`、`-xy`、`yx`、`-yx` 等候选 convention 中选择 IWE 最聚焦的事件 warp 方向。
4. 再比较图像前向模型的 flow convention 和事件极性符号。
5. 固定光流，在曝光中心生成事件 IWE 目标。
6. 优化一张 log-intensity 潜在图，使其预测 IWE 匹配事件 IWE，并加入 TV 正则。
7. 输出 IWE 对照、loss 曲线、光流可视化和估计旋转中心。

### 它不做什么

- 不从事件中学习或优化稠密旋转光流；光流来自外部文件。
- 不执行完整的 APS frame-event 联合重建；当前 `frame_loss_weight=0`。
- 不对应论文三项光学实验中的一个独立光学 forward model。
- 不是当前平移数据运行 `NeuroSRM_demo.py` 的必需步骤。

所以它更准确的定位是“稠密旋转流 + IWE 图像梯度算子的研究和调试工具”。

### 什么场景需要使用

- 样品绕某个中心旋转，而不是近似全局平移；
- 已经有 EKLT、事件光流或其他算法输出的稠密光流；
- 不确定 flow 的通道顺序、正负号或事件极性约定；
- 想验证论文中的 $b_E\approx-\nabla L\cdot v$ 在旋转流下是否成立；
- 想估计并检查旋转中心是否符合机械装置。

### 具体使用方法

先准备：

- 一个包含 APS 帧和事件的 `.aedat4` 文件；
- 一个 shape 为 `[2,H,W]` 的稠密 flow `.npy` 文件，其空间尺寸最好与目标 APS 帧相同。

修改文件顶部：

```python
event_path = Path("/path/to/rotation.aedat4")
flow_init_path = Path("/path/to/flow.npy")
save_dir = Path("output/rotation_demo")
frame_index = 57
```

然后运行：

```bash
cd /home/robbie/tyf_code/EventCode/myFEFibreSR/isl_diff_event
/home/robbie/miniconda3/envs/NeuroFibreSR/bin/python rotation_demo.py
```

当前仓库中的两个输入路径仍是 Windows 绝对路径，而且仓库不包含 `flow_multi.npy`，所以必须先提供真实数据和外部光流，不能原样直接运行。

## 3. 仓库是否包含论文不同光学模型的完整代码

### 论文的三种实验实例

论文按顺序展示：

1. **大视场 4f relay 工业检测**：已知、近似匀速平移，曝光近似理想，forward operator 接近恒等映射加传感器降采样。
2. **单帧像素超分辨率 inline holography**：未知复振幅/相位、自由空间传播、传播距离和运动轨迹联合求解。
3. **高通量宽场生物显微**：连续手动扫描、运动模糊、非匀速轨迹、背景杂波和静态遮挡。

### 当前仓库实际具有什么

仓库里确实有一些传统计算光学基础模块：

- `function/diffraction.py`：传播核、正向和反向波场传播；
- `function/ptychography.py`：Rayleigh-Sommerfeld/传递函数、像素超分辨率和位置配准等函数；
- `function/solver.py`：传统 PSR、AD、coded ptychography 求解器；
- `function/zernike.py`、`function/unwrap.py`：像差表示和相位展开；
- `function/optimizer.py`：图像、复场和相位相关正则项。

但检索现有入口和调用关系后，没有发现一个能够把以下部分完整连接起来并直接复现论文 Fig. 3 的入口：

```text
事件运动估计
  + 事件梯度似然
  + 复场 O = a exp(j phi)
  + 自由空间传播 P_z
  + APS 曝光积分和降采样
  + 振幅、相位、传播距离和轨迹联合优化
```

`NeuroSRM_demo.py` 顶部虽然保留了 `forward_wave_prop` 的注释示例，但实际运行路径没有调用衍射传播模型。传统 `function/solver.py` 又没有接入 NeuroSR 的事件似然。因此，“仓库有光学零件”和“仓库有论文完整的事件全息实验”是两回事。

### `NeuroSRM_demo.py` 对应论文哪一种实验

不能简单说它约等于论文第一种实验。

当前脚本的实际 forward model 包括：

- CMax 估计非匀速分段运动；
- 用曝光内运动轨迹生成 motion-blur kernel；
- 对潜在强度图做曝光模糊和传感器降采样；
- 使用可学习的 `bg_pred` 吸收背景/遮挡差异；
- 用事件 IWE 和一张模糊 APS 帧联合约束重建。

这些特点与论文第三种“高通量宽场生物显微”最接近，因为第三种实验明确处理手动非匀速扫描、运动模糊、背景杂波和静态遮挡。

论文第一种工业检测则使用受控、已知的近似匀速运动，并被正文称为 `exposure-ideal`。它主要验证事件高频约束带来的像素超分辨能力，不强调当前 Demo 中的盲运动模糊和背景补偿。因此当前 Demo 可以看成论文通用 NeuroSR 框架在“强度成像 + 运动模糊”场景下的一种实现，但不能准确称为第一种工业检测实验的等价复现。

### 最终判断

- 当前仓库没有三套彼此完整、可直接运行的 NeuroSR 光学实验入口。
- `NeuroSRM_demo.py` 是目前唯一相对完整的 event-frame 强度重建主流程。
- 它在 forward model 上最接近论文第三种宽场显微实验，而不是第二种全息实验，也不严格等于第一种工业检测实验。
- 仓库中的衍射、PSR 和相位模块可以作为未来实现论文全息分支的基础，但还需要把事件似然、曝光模型和联合优化真正接入。
- `NeuroSRM_MEDI_demo.py` 与 `rotation_demo.py` 是围绕事件阈值和复杂运动场的扩展研究脚本，不代表另外两种论文光学实验。
