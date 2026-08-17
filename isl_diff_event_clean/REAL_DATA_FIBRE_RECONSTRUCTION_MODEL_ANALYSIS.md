# 从当前仿真到真实光纤实验：forward 与重建模型合理性分析

## 1. 先给结论

当前 `FibreNeuroSR_demo.py` 的模型可以证明一件重要的事：在**运动、纤芯位置、光学参数和事件响应都准确已知**时，APS 与纤芯事件确实能够共同约束一张连续的远端图像，并实现去蜂窝和沿扫描方向的额外采样。

但是，它目前仍属于 **exact-model proof of concept**，还不能直接代表真实实验难度。仿真器和反演器使用了几乎同一套光学假设，因而存在“生成数据的模型正好就是重建所假设的模型”这一有利条件。真实实验中的主要差异包括：

- 远端物体相对光纤的真实运动通常只有粗略先验，不会有精确的 `motion.npz` 真值；
- 真实芯斑亮度不均匀、形状不规则，不应由理想圆斑生成；
- 各 DAVIS pixel 的背景、增益、正负事件阈值和噪声并不完全一致；
- GRIN、纤芯接收孔径、离焦、串扰和模式变化会合并成未知的有效光学响应；
- APS 和 events 还会有曝光起止、时钟延迟、饱和及低照度噪声问题。

因此，真实数据版不应该继续依赖仿真真值参数，也不应该一开始建立非常庞大的逐模式模型。建议采用下面这条简单主线：

```text
实测 dark/flat/core map
        -> DAVIS pixels 归属到纤芯通道
        -> 低维运动轨迹初始化
        -> 简单、可微的有效光学 forward
        -> APS + core-event 联合损失
        -> 必要时小范围交替修正运动
```

其中：

- **对数梯度 + event warp/IWE**：主要负责未知运动初始化，也可作为可替换的事件约束；
- **有限位移下的逐芯强度变化模型**：作为最终重建的主要数据一致性项；
- **实测芯斑 mask**：处理真实近端芯斑的不均匀和非圆形，不必把每个芯斑硬拟合成圆。

这三个部分并不冲突，组合起来比单独照搬 old 模型或继续使用仿真真值更合适。

## 2. 当前实现到底理想化在哪里

当前代码的主要数据流是：

```text
latent object
  -> GRIN 重采样/模糊
  -> 理想圆形纤芯孔径积分
  -> 每根纤芯的标量信号
  -> APS 时间平均 + lin-log event change
  -> 与仿真 APS/events 比较
```

对应代码为：

- `neurosr/fibre_forward.py::FibreCoreForward.forward()`：运动重采样、GRIN 模糊、圆形芯孔径积分；
- `neurosr/fibre_event_loss.py::predict_cumulative_event_change()`：将逐芯强度经过 lin-log 响应，并减去初始响应；
- `neurosr/fibre_pipeline.py::_optimise()`：计算 APS、event 和 TV loss；
- `neurosr/fibre_data.py::load_fibre_observations()`：直接读取仿真真值轨迹 `motion.npz`。

它包含了合理的第一阶物理因素，但下列条件明显偏理想：

| 当前条件 | 真实实验中的情况 | 影响 |
|---|---|---|
| 轨迹逐时刻精确已知 | 通常只有 stage 命令、encoder 或完全未知的平滑运动 | 轨迹误差会直接变成重建模糊或伪纹理 |
| 芯中心和倍率精确已知 | 需要由 flat-field 图像标定，还可能有旋转和畸变 | core 与 sensor 对应错误会污染 APS/event 通道 |
| 圆形且一致的芯孔径 | 有效接收响应可能逐芯不同 | 高频响应与真实系统不一致 |
| 固定、规则的近端圆斑 | 芯斑可不均匀、非圆、带 speckle | 不能再按理想几何选固定的 4 个像素 |
| gain 和事件阈值已知 | pixel 间阈值、背景和噪声有差异 | event 累计量存在比例误差和偏置 |
| 仿真 forward 与 inverse forward 同源 | 真实系统必然存在模型失配 | 当前指标会高估真实性能 |

所以当前结果应被解释为：**验证信息原则上足够，以及验证代码链路能够优化**；不能解释为：真实硬件接入后无需标定或修改就能得到相同结果。

## 3. 真实实验中的运动轨迹确实可能不知道

### 3.1 “不知道”通常有不同程度

真实系统不一定只有“完全已知”和“完全未知”两种状态：

1. 电控位移台提供命令轨迹；
2. encoder 提供较可靠的实测轨迹；
3. 命令或 encoder 轨迹仍有 scale、rotation、delay 和零点误差；
4. 手持、振动或柔性机构运动可能只有“短时间内平滑”的先验。

第一版物理实验最好保留可控位移台和 encoder。这样能先回答“真实光纤事件是否提升重建”，再逐步增加盲运动估计。即使最终应用没有 encoder，这个受控基线仍然是判断算法问题还是硬件问题所必需的。

### 3.2 完全逐时刻自由的运动不能和图像一起从零估计

若同时让每个时刻的二维位移和所有图像像素自由变化，图像纹理与运动会互相补偿，问题很容易退化。例如，错误的边缘可以被一条错误轨迹解释，低纹理区域则几乎没有运动信息。

因此应把运动写成少量参数控制的平滑轨迹：

$$
\mathbf{s}(t)=\mathbf{s}(t;\boldsymbol{\theta}),
$$

其中参数可以是：

- 已知直线方向下的速度、幅度和时间延迟；
- 少量分段线性节点；
- 少量 B-spline 控制点。

还必须固定坐标自由度，例如令参考时刻位移为零：

$$
\mathbf{s}(t_{ref})=\mathbf{0}.
$$

否则“整张物体平移”和“整条轨迹加同一常量”在数据上不可区分。

### 3.3 推荐的低复杂度估计方式

推荐采用两阶段而不是从随机状态完全联合优化：

```text
阶段 1：用 stage/encoder 作为初值；若没有，则用 core-domain IWE 估计粗运动
阶段 2：固定粗运动先恢复图像
阶段 3：只允许少量轨迹参数小范围更新，再交替修正图像
```

实际优化目标可以保持简单：

$$
\min_{O,\boldsymbol{\theta}}
\mathcal{L}_{APS}
+\lambda_e\mathcal{L}_{event}
+\lambda_O R_O(O)
+\lambda_s R_s(\boldsymbol{\theta}).
$$

这里不要同时在线优化 core 位置、逐 pixel gain、逐 pixel threshold、逐芯 PSF 和运动。它们应先通过标定固定；只有残差明确指出某个参数错误时，再开放极少数全局参数，例如 scale、rotation 或 APS/event delay。

## 4. 为什么当前代码没有直接使用“对数梯度乘运动”

### 4.1 old 模型表达的是什么

普通 NeuroSR 使用的 EKLT 一阶近似可以写成：

$$
\Delta L(\mathbf{x})
\approx
-\nabla L(\mathbf{x})\cdot\Delta\mathbf{s}.
$$

其中：

- $L$ 是线性到对数响应后的图像；
- $\nabla L$ 是空间梯度；
- $\Delta\mathbf{s}$ 是曝光期间的图像位移；
- warp 后的正负 event 累计图 IWE 是观测目标。

代码对应为：

- `neurosr/optimization.py::linear_log_intensity()`；
- `neurosr/optimization.py::forward_gradient()`；
- `neurosr/optimization.py::predicted_iwe()`；
- `neurosr/events.py::warp_events_to_reference()`；
- `neurosr/pipeline.py::estimate_motion()`。

它成立的核心前提是：sensor 图像纹理本身随运动在 sensor 平面平移。

### 4.2 当前 fibre 模型其实是它的有限变化版本

令经过有效光学模糊后的远端图像为 $S$，第 $i$ 根芯在物体坐标中的采样位置为：

$$
\mathbf{q}_i(t)=\frac{\mathbf{r}_i}{M}-\mathbf{s}(t).
$$

先忽略芯孔径积分，纤芯对数信号为：

$$
L_i(t)=\rho\!\left(S(\mathbf{q}_i(t))\right),
$$

其中 $\rho$ 是 DAVIS 的 lin-log 响应。当前模型直接计算有限变化：

$$
\Delta L_i(t)=L_i(t)-L_i(t_0).
$$

若位移很小，对上式做一阶 Taylor 展开，就得到：

$$
\Delta L_i(t)
\approx
-\nabla L(\mathbf{q}_i(t_0))
\cdot
\left[\mathbf{s}(t)-\mathbf{s}(t_0)\right].
$$

因此两套公式不是互相矛盾：

- 当前 fibre forward 计算的是有限位移下的非线性强度差；
- old EKLT 计算的是相同亮度恒常关系在小位移下的一阶近似。

当前实现没有显式写“梯度乘运动”，不是漏掉了事件物理，而是选择直接重新采样图像并计算前后 lin-log 差值。这样不依赖小位移近似，也自然包含芯孔径积分；代价是它必须先有一条可用的运动轨迹。

### 4.3 当前事件 forward 仍然偏理想

`predict_cumulative_event_change()` 目前预测的是连续的 lin-log 变化：

$$
\widehat{E}_{i,p}(t)
=
\rho\!\left(g_{i,p}c_i(t)\right)
-
\rho\!\left(g_{i,p}c_i(t_0)\right).
$$

观测端则将正负事件数分别乘阈值后累计。两者在理想事件相机中相差不超过阈值余量，但真实相机会额外受到：

- 初始阈值 remainder；
- 正负阈值不对称；
- pixel-to-pixel threshold mismatch；
- refractory period、漏事件和背景活动；
- 低照度与固定背景对 log 响应的影响。

第一版真实模型不必完整模拟所有电路细节。使用逐芯或分区标定的正负阈值，并以 Huber loss 比较累计变化，已经是合理的稳健近似；同时应保留低事件数置信度和异常通道 mask。

## 5. 为什么不能直接 warp 原始 DAVIS event

普通相机中，物体边缘运动会让 event 的 sensor 坐标一起移动，所以直接 warp 原始 event 是合理的。

本实验中，远端物体在动，但近端光纤端面和 DAVIS 固定。近端亮斑的地址基本固定，变化的是每根芯斑的亮度。因此若直接对 raw DAVIS event 坐标做普通二维平移 warp，就相当于假设整个近端芯斑阵列在 sensor 上移动，这与当前物理装置不符。

可行的做法是先把 event 映射到纤芯编号，再构造 **core-domain IWE**：

1. event 所在 sensor pixel 通过实测 core map 归属于第 $i$ 根芯；
2. 不再使用该 event 在芯斑内部的 pixel 位置，而把它表示在远端芯中心 $\mathbf{r}_i/M$；
3. 对候选运动 $\mathbf{s}(t;\boldsymbol{\theta})$，把该位置 warp 到同一参考时刻；
4. 累积正负极性并最大化 IWE 的清晰度或对比度。

示意公式为：

$$
\mathbf{x}'_k
=
\frac{\mathbf{r}_{i(k)}}{M}
-
\left[
\mathbf{s}(t_k;\boldsymbol{\theta})
-
\mathbf{s}(t_{ref};\boldsymbol{\theta})
\right].
$$

这里的 $i(k)$ 表示第 $k$ 个 DAVIS event 属于哪根芯。

core-domain IWE 很适合估计运动初值，但它仍是近似模型，因为它把有限芯孔径积分压缩到了芯中心。最终图像重建继续使用逐芯强度 forward 会更稳妥。

## 6. 真实芯斑不均匀、不圆，应该怎样处理

### 6.1 先区分两个不同概念

必须区分：

- **远端有效接收响应**：决定一根芯如何对物体邻域积分；
- **近端输出芯斑形状**：决定同一个纤芯标量如何分布到 DAVIS pixels。

近端芯斑不是圆形，并不自动意味着远端采样位置也要按这个形状解释。尤其在存在模式混合时，芯斑内部的近端位置通常不能映射为远端的同一子位置。

### 6.2 不再生成圆斑，直接测量 core-to-pixel map

真实系统应拍摄 dark 和均匀照明 flat-field，得到每根芯在 DAVIS 上的实际响应。可以用非圆形 soft mask 表示：

$$
I_p(t)=b_p+\sum_i W_{pi}c_i(t)+n_p(t).
$$

其中：

- $b_p$ 是 dark/background；
- $W_{pi}$ 是实测的第 $i$ 根芯对 pixel $p$ 的权重；
- $c_i(t)$ 是该芯的标量通光量；
- $n_p(t)$ 是噪声。

第一版不必在优化中显式生成整张 DAVIS 图。可以由 $W$ 选择每根芯高信噪比、低重叠的 pixels，并稳健聚合为 core channel：

$$
\widehat{c}^{APS}_i
=
\operatorname{RobustAverage}_{p\in\mathcal{P}_i}
\frac{I^{APS}_p-b_p}{W_{pi}}.
$$

对 event 也先按 $\mathcal{P}_i$ 归芯。由于多个 pixels 是同一芯信号的冗余读出，应对各 pixel 的累计对比度做加权平均，而不是简单求和；否则较大的芯斑会仅因覆盖 pixels 更多而获得更大权重。

这会把“芯斑是否圆、内部是否均匀”的问题留在标定矩阵 $W$ 中，而不是写死在重建公式里。

### 6.3 什么时候标量 core 模型不够

需要先做一个实测稳定性实验：固定光纤近端和相机，小范围移动远端目标，检查同一芯斑内部各 pixels 的时序是否近似为同一个标量的固定增益版本。

若归一化后的时序高度相关，说明：

$$
I_{i,p}(t)\approx b_{i,p}+W_{i,p}c_i(t)
$$

是足够的，当前逐芯标量模型可以继续使用。

若芯斑形状随远端入射位置明显变化，说明模式分布也携带变化，标量模型会产生结构化残差。此时才考虑：

- 每芯少量低秩模式系数；
- 实测的小范围 transmission matrix；
- 保留 raw sensor pixels 的 forward。

这些都应是由实验残差触发的升级项，而不是第一版默认模型。

## 7. 建议用于真实数据的最小 forward

### 7.1 有效光学层

不要分别精确拟合 GRIN、离焦、芯孔径和轻微串扰。第一版把它们合并成一个可标定的有效 PSF：

$$
S=h_{eff}*O.
$$

第 $i$ 根芯在时刻 $t$ 的信号为：

$$
c_i(t)
=
a_i
+
g_i S\!\left(
\frac{\mathbf{r}_i}{M}-\mathbf{s}(t)
\right).
$$

这里 $a_i$、$g_i$ 是逐芯背景和增益。若共享 PSF 的重投影残差已经无明显逐芯结构，就不需要逐芯 PSF。

### 7.2 APS 观测层

APS 是曝光期间的时间平均：

$$
\widehat{A}_i(O,\mathbf{s})
=
\frac{1}{T_e}
\int_{t_a}^{t_b}c_i(t)\,dt.
$$

它与实测 core APS 比较，负责绝对强度、低频轮廓和事件无法确定的全局亮度尺度。

### 7.3 event 观测层

主要事件模型使用有限变化：

$$
\widehat{E}_i(t)
=
\rho(c_i(t))-\rho(c_i(t_0)).
$$

实测 events 经 core map、正负阈值和时间累计后得到 $E_i^{obs}(t)$。二者使用 Huber loss：

$$
\mathcal{L}_{event}
=
\operatorname{Huber}
\left(
\widehat{E}_i(t)-E_i^{obs}(t)
\right).
$$

old 的梯度/IWE 模型保留为另一种 `EventModel`，主要用于运动初始化、消融比较，或在有限变化模型失配时提供辅助约束。

### 7.4 完整但仍简单的 loss

$$
\mathcal{L}
=
\mathcal{L}_{APS}
+\lambda_e\mathcal{L}_{event}
+\lambda_{tv}\operatorname{TV}(O)
+\lambda_s\|\mathbf{s}''(t)\|_2^2.
$$

这已经覆盖真实数据第一版最关键的四件事：绝对强度、时间变化、图像正则和运动平滑。暂时不需要 GAN、复杂神经网络或完整模式传播模型。

## 8. 建议的 plug-and-play 代码结构

真实数据路径应新增模块，不修改已经验证 old 行为的普通 `NeuroSRM_demo.py`：

```text
neurosr/
  real_fibre_calibration.py   # dark/flat、core mask、gain、threshold
  real_fibre_data.py          # AEDAT4/HDF5 -> core APS + core events
  fibre_motion.py             # encoder、core-IWE、低维轨迹
  fibre_optics.py             # EffectiveCoreForward
  fibre_event_models.py       # CumulativeContrast / GradientIWE
  real_fibre_pipeline.py      # 初始化、固定运动重建、交替修正

RealFibreNeuroSR_demo.py       # 只负责配置、运行和保存结果
```

每一层只保留清晰接口：

```text
calibration.map_sensor_to_cores(aps, events)
    -> core_aps, core_events, confidence

motion.initialise(core_events, core_centres, encoder=None)
    -> low_dimensional_trajectory

optics.forward(object_image, trajectory)
    -> core_signals_over_time

event_model.predict(core_signals_over_time)
    -> predicted_core_event_change

pipeline.reconstruct(observations, calibration, models)
    -> object_image, trajectory, diagnostics
```

这样可以独立替换：

- `KnownMotion`、`EncoderMotion` 或 `CoreIWEMotion`；
- `SharedEffectivePSF` 或以后需要的 `PerCorePSF`；
- `CumulativeContrast` 或 `GradientIWE`。

入口脚本不需要知道每个公式的内部细节，也不会像 old 脚本一样把数据、warp、forward、loss 和画图混在一个文件中。

## 9. 真实数据的完整最小流程

### 步骤 0：离线标定

至少采集：

1. dark frames：估计背景、坏点和读出噪声；
2. 均匀照明 flat-field：检测真实芯斑、soft mask 和逐芯增益；
3. 已知小位移标定靶：标定远端 um、core lattice、sensor pixel、stage 坐标之间的 scale 和 rotation；
4. 闪烁或已知强度变化：估计正负事件阈值及 APS/event 时间延迟；
5. 点、边缘或细线靶：估计共享的有效 PSF。

### 步骤 1：预处理一次真实曝光

```text
raw APS
  -> dark subtraction / saturation mask
  -> soft core masks 聚合
  -> core APS + confidence

raw events
  -> hot-pixel / background-activity filter
  -> sensor pixel 查 core map
  -> 每芯各 readout 的累计对比度
  -> 稳健平均 + confidence
```

不要根据理想圆形半径分配事件；使用实测 mask。芯斑重叠或低信噪比位置可以直接丢弃。

### 步骤 2：初始化图像和运动

- 在实际 core centres 上插值 core APS，得到低分辨初图；
- 有 encoder 时用其轨迹，并只估计 scale、rotation 和 delay；
- 无 encoder 时用 core-domain IWE 估计少量轨迹参数；
- 固定参考位移为零。

### 步骤 3：重建

1. 固定轨迹，只优化 latent object；
2. 固定图像，只小范围优化轨迹控制点；
3. 重复少量轮次，且始终保留轨迹平滑正则；
4. 不让标定参数随意吸收图像误差。

### 步骤 4：保存能够判断模型是否正确的诊断结果

除了最终图像，至少保存：

- observed/predicted core APS 散点和残差图；
- observed/predicted core event change 的时间曲线；
- 每芯 residual、event count 和 confidence；
- 初始轨迹与最终轨迹；
- core-domain IWE 在修正运动前后的对比；
- 留出时间段或留出 cores 的 forward prediction。

只看最终图是否“清楚”不足以证明模型正确。若模型真实有效，它还应能预测未参与优化的部分观测。

## 10. 如何控制复杂度

建议按下表逐级增加模型，而不是一次全部实现：

| 层级 | 模型 | 何时使用 |
|---|---|---|
| M0 | 实测 core map + shared effective PSF + encoder/受控运动 | 第一套真实实验，必须先跑通 |
| M1 | M0 + core-IWE 初值 + 低维运动小范围修正 | 验证弱先验或未知运动 |
| M2 | M1 + 逐芯 gain/background/threshold/confidence | 实测残差显示明显芯间差异 |
| M3 | M2 + 逐芯或低秩模式响应 | 芯斑内部时序不能由固定增益解释时 |

第一版明确不要做：

- 从零联合估计每个时刻的自由二维位移；
- 把芯斑内每个 pixel 当作远端独立亚像素；
- 直接对 raw DAVIS event 做普通 sensor-plane 平移 warp；
- 在没有标定证据时拟合完整 transmission matrix；
- 同时放开物体、逐芯 PSF、阈值、gain、core centres 和运动；
- 只用与 inverse 完全相同的 simulator 结果证明真实适用性。

## 11. 需要用真实实验回答的关键问题

真正决定当前思路能否落地的不是公式数量，而是下面四项测量：

1. **芯内时序一致性**：同一芯斑不同 pixels 的归一化时序是否高度相关；
2. **core map 稳定性**：光纤轻微弯曲、温漂后，近端芯斑 mask 是否仍稳定；
3. **运动可辨识性**：core-domain IWE 能否在重复实验中恢复一致的方向、幅度和延迟；
4. **forward 泛化**：用一部分时间/cores 拟合后，能否预测留出的 APS/events。

若前两项成立，简单的逐芯标量模型很有希望工作；若第三项较弱，应优先使用受控 stage 或 encoder，而不是增加图像网络复杂度；若第四项失败，再依据 residual 的结构决定补充哪一层物理模型。

## 12. 最终建议

真实数据版的首选方案不是在“当前有限变化 forward”和“old 的梯度/IWE”之间二选一，而是让二者各自做最适合的工作：

```text
实测非圆 core masks
  -> 将 APS/events 可靠地归到纤芯通道
  -> old-style core-domain IWE 给出运动初值
  -> 简单 effective-PSF core forward 预测逐芯强度
  -> finite lin-log event loss + APS loss 恢复图像
  -> 只对低维运动做小范围联合修正
```

这套模型能够清楚说明完整 forward，又没有依赖仿真的理想圆斑、真值轨迹或过度复杂的模式传播。它也保留了 plug-and-play 性：运动模型、光学算子和事件模型都可分别替换，真实数据到来后无需重写整个重建流程。
