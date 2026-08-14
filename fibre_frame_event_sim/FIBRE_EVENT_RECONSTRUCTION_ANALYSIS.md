# 固定近端纤芯事件如何用于重建：可行性与 clean 代码改造方案

## 1. 本文回答的问题

当前实验的物理条件已经明确：

- 远端物体相对多芯光纤端面移动；
- 纤芯阵列、近端输出芯斑、中继系统和 DAVIS 均固定；
- 一根纤芯内部的入射位置与输出芯斑内部位置不能按几何坐标一一对应；
- 当前事件图中同一芯斑内的多个 DAVIS pixels 主要是同一纤芯信号的空间展开和冗余读出。

在此前提下，本文回答：

1. 现有事件是否还有重建价值；
2. 应怎样解释和使用这些事件；
3. 能否基于 `isl_diff_event_clean` 开发；
4. clean 中哪些逻辑可以复用，哪些必须替换；
5. 怎样分阶段实现并判断重建是否真正成功。

## 2. 先给出结论

### 2.1 还能不能重建

**原则上可以。**不需要知道一根芯内部的入射子位置对应哪个输出子位置。

重建所依赖的信息不是“近端圆环内部的位置”，而是：

> 第 $i$ 根固定纤芯在不同远端运动位置上，对物体做有限孔径积分后得到的总通光量 $c_i(t)$ 如何随时间变化。

APS 提供绝对强度和低频锚；事件提供每根通道在高时间分辨率下的对数亮度增减。已知纤芯中心和远端运动后，不同时刻的同一根纤芯等价于用同一个圆孔在物体的不同位置进行扫描测量。

### 2.2 能重建到什么程度

当前仿真使用：

- 1415 根局部纤芯；
- 4.5 um 芯间距；
- 2.9 um 芯径；
- 25 ms 内水平移动 4.5 um；
- 一个方向的匀速轨迹；
- 一张 APS 曝光帧和约 6.57 万个 sensor-pixel events。

在精确模型、已知运动和无噪声条件下，预计可以：

- 去除或显著减弱蜂窝采样外观；
- 恢复一个可识别的 USAF 强度图；
- 提高水平方向的采样相位覆盖和边缘定位；
- 相比 APS-only 获得更好的运动方向细节。

但当前条件不能保证：

- 各向同性二维超分辨；
- 可靠恢复 0.5 um 网格上的全部自由度；
- 恢复被 2.9 um 芯孔径或其他光学 OTF 完全抑制的频率；
- 真实硬件中的模式变化、串扰和标定误差下仍得到同样结果。

当前只有水平运动，垂直方向没有新的扫描相位，所以更可能得到**方向性增强**，而不是严格的二维 2x 或更高倍率超分辨。

### 2.3 能否直接使用 clean 代码

可以把 `isl_diff_event_clean` 当作工程骨架，但不能直接把当前 HDF5 events 填进现有 `NeuroSRM_demo.py`。

可以复用：

- 配置、入口、pipeline 分阶段组织方式；
- PyTorch 优化器和部分正则项；
- 结果保存、loss history 和对比测试结构；
- 后期需要未知运动时的分段轨迹参数化思想。

必须替换：

- AEDAT4 数据加载和曝光选择；
- 普通 sensor-plane event spatial warp；
- 基于 IWE 方差的普通 CMax 运动估计；
- $-\nabla L\cdot\Delta x$ 的普通 EKLT/IWE 图像预测；
- 用一个全局卷积核解释 APS 运动模糊；
- 把 `scale=2` 简单解释为 DAVIS 图像上采样。

最稳妥的做法是新增一条 fibre pipeline，不修改已经验证与旧代码一致的 clean 主流程。

## 3. 为什么没有芯内位置对应仍能重建

### 3.1 每根纤芯是一个扫描积分通道

设待恢复远端物体为 $O(\mathbf u)$，第 $i$ 根纤芯中心为 $\mathbf r_i$，归一化入口孔径为 $A_i$，物体相对光纤位移为 $\mathbf s(t)$。

第 $i$ 根芯在时刻 $t$ 的总强度为：

$$
c_i(t;O)=
\int A_i(\mathbf u-\mathbf r_i)
O(\mathbf u-\mathbf s(t))\,d\mathbf u.
$$

等价地，也可以理解为纤芯孔径在物体坐标中的有效采样中心为：

$$
\mathbf q_i(t)=\mathbf r_i-\mathbf s(t).
$$

所以同一根芯虽然物理位置不动，但它随时间观测到了物体上的不同位置。这就是新增空间采样信息的来源。

### 3.2 事件测量的是该通道的时间变化

如果暂时把一根芯看成一个理想标量通道，则事件近似满足：

$$
\log c_i(t_k)-\log c_i(t_{last})
=p_k C_i,
\qquad p_k\in\{-1,+1\}.
$$

它告诉我们：在已知扫描轨迹上，某个圆孔积分测量增加或减少了一个对比度阈值。

这和扫描显微、线阵扫描或移动孔径测量类似：探测器内部不需要保留物体位置，位置由已知扫描轨迹和探测器中心确定。

### 3.3 近端芯斑内部 pixels 不提供新的远端位置

当前模拟的近端 sensor 强度可以写为：

$$
I_p(t)=\sum_i P_{pi}c_i(t)+b_p,
$$

其中 $P_{pi}$ 表示第 $i$ 根芯的固定输出 spot profile 被 DAVIS pixel $p$ 接收的比例。

在当前标量 core 模型中，同一芯斑内多个 pixels 观测的是同一个 $c_i(t)$ 的不同增益版本。它们可以提高读出稳健性，但不能被当成多个独立远端物体坐标。

因此重建时必须避免：

- 把圆环不同位置当成远端亚纤芯采样点；
- 认为一个直径 4.71 sensor pixels 的芯斑提供了 4.71 pixels 的独立空间分辨率；
- 直接根据 sensor event 坐标做普通图像平移 warp。

## 4. 真实模式传播对模型的影响

### 4.1 为什么不能假设入口和出口逐点对应

更一般地，单芯输出场可以写为：

$$
E_{out,i}(\mathbf z,t)
=\sum_m a_{i,m}(t)\psi_{i,m}(\mathbf z)e^{j\beta_{i,m}L}.
$$

输入位置、角度、偏振、波长、纤芯弯曲和模式耦合会改变 $a_{i,m}$。所以近端模式或 speckle 的某个位置通常不能对应到远端同一位置。

### 4.2 这会不会使重建完全不可能

不会，但决定了应使用哪一级观测：

- 如果模式分布稳定，只是整体幅值随 $c_i(t)$ 变化，可以使用固定逐芯 PSF $P_i$。
- 如果模式图案会随入射位置明显变化，应使用 mode-dependent PSF 或实测 transmission matrix。
- 如果不想依赖模式细节，应尽量对每根芯的整个输出斑做总通光量聚合，在 core-domain 重建。

宽带、空间非相干照明可能平均掉部分模式干涉；高相干照明则更容易产生时变 speckle。最终应由实际光源和实测近端芯斑随远端扫描的稳定性决定模型复杂度。

### 4.3 当前仿真属于哪一种

当前 simulator 采用最简单情况：

```text
远端圆孔面积平均 -> 每芯一个标量 c_i(t)
                -> 固定圆形输出 spot P_i
                -> DAVIS pixels
```

因此当前数据可以用固定 $P_i$ 精确复现。它适合建立第一个重建闭环，但属于 exact-model 验证，不能代表真实模式传播已经解决。

## 5. 现有事件的三种使用方式

### 5.1 方案 A：按纤芯聚合事件，作为主要物理基线

这是最保守、最符合“每芯一个测量通道”理解的方案。

#### 步骤

1. 根据模拟 core centres、中继倍率和固定 spot profile，建立 sensor pixel 到 core 的归属关系。
2. 对重叠或归属不明确的 pixels 降权或丢弃。
3. 把事件按时间分 bin，例如每 `0.1 ms` 或 `0.5 ms`。
4. 对每根芯分别统计 ON/OFF：

$$
N_{i,b}^{+},\qquad N_{i,b}^{-}.
$$

5. 构造累计对数变化观测：

$$
S_i(t_b)
=C_+N_i^+(0,t_b)-C_-N_i^-(0,t_b).
$$

6. 用预测 core intensity 的对数变化拟合它：

$$
\mathcal L_E^{core}
=\sum_{i,b}w_{i,b}\,ho\left(
[\ell(c_i(t_b;O))-\ell(c_i(t_0;O))]-S_i(t_b)
\right).
$$

其中 $\ell$ 应与 v2e 使用的 lin-log 变换一致，$\rho$ 推荐 Huber 或 Charbonnier loss。

#### 聚合时不能直接简单求和

同一芯斑多个 pixels 是冗余读出。如果直接把全部 event counts 相加，会让覆盖更多 sensor pixels 的芯获得更大权重，并人为放大信息量。

建议依次尝试：

1. 只选每根芯中高信噪比、归属明确的中心 pixels；
2. 对每 pixel 的累计 signed count 做 median，再得到每芯观测；
3. 或按每芯有效 pixel 数归一化；
4. 保存 pixel-level dispersion，作为该芯模式不稳定或模型失配的置信度。

#### 优点与局限

优点：

- 不把近端圆环误当远端空间信息；
- 计算量较小；
- 更容易迁移到真实光纤；
- 可以抑制模式形态和 pixel gain 的部分影响。

局限：

- DAVIS 是逐 pixel 触发，聚合后只近似 core log change；
- v2e 的 lin-log、背景和每 pixel reference state 会带来误差；
- 需要可靠的 core-to-sensor 分区和阈值标定。

### 5.2 方案 B：保留原始 sensor events，做完整 sensor forward likelihood

这是当前仿真下最精确的方案，因为 simulator 中的 $P_{pi}$ 已知。

#### 事件观测

把原始 events 构造成每个时间 bin、每个 DAVIS pixel 的累计计数：

$$
Q_{p,b}^{+},\qquad Q_{p,b}^{-}.
$$

预测物体经过 fibre forward 后得到 $I_p(t_b;O)$，再比较：

$$
\mathcal L_E^{sensor}
=\sum_{p,b}\widetilde w_{p,b}\,ho\left(
[\ell(I_p(t_b;O))-\ell(I_p(t_0;O))]
-[C_+Q_{p,b}^{+}-C_-Q_{p,b}^{-}]
\right).
$$

累计形式通常比逐 bin 差分稳定，因为事件 reference 在每次触发后更新，而任意 bin 边界不一定恰好是一次事件时刻。

#### 怎样避免利用虚假的芯内信息

即使保留 pixel events，也应按每根芯归一化总权重。例如，一根芯覆盖 20 个有效 pixels 时，每个 pixel 的权重不能仍然等于一个独立纤芯通道。

可以令：

$$
\sum_{p\in\Omega_i}\widetilde w_p=1,
$$

其中 $\Omega_i$ 是第 $i$ 根芯的 sensor 支持区域。

这样 pixel events 用于拟合真实 sensor readout，但不会因为一个芯斑被多个 pixels 采样而虚构额外空间自由度。

#### 优点与局限

优点：

- 可以原样使用当前 `events.h5`；
- 在当前固定圆斑仿真中最接近真实数据生成过程；
- 能检查 fibre render、relay 和 v2e 是否整体自洽。

局限：

- 计算和显存开销大；
- 对 $P_i$、背景、pixel gain 和模式稳定性敏感；
- exact-model 成功可能包含 inverse crime，不能单独作为真实可行性证据。

### 5.3 方案 C：构造 fibre-aware IWE，仅作初始化和可视化

可以先把 sensor event 映射到所属 core $i(k)$，再根据事件时刻把它放到远端物体坐标：

$$
\mathbf u_k=\mathbf r_{i(k)}-\mathbf s(t_k).
$$

然后在物体网格上累计 polarity，形成 fibre-aware event feature map。

它与普通 IWE 的区别是：

- 普通 IWE warp sensor image feature；
- fibre-aware 映射使用 core identity 和远端采样轨迹；
- 同一芯斑多个 sensor events 必须先归一化或聚合；
- 还应考虑 2.9 um 圆孔，而不只是把事件当成点。

这个 feature map 适合：

- 检查坐标符号和轨迹；
- 初始化物体边缘；
- 做结果可视化；
- 提供运动方向梯度参考。

但它不应替代完整 forward likelihood，因为简单 splat 会忽略孔径积分、事件阈值 remainder 和模式/PSF。

### 5.4 推荐选择

建议同时实现两条基线：

1. **主要科学结论使用方案 A：core-domain 聚合事件。**它最保守，不利用不可信的芯内位置。
2. **代码回归和仿真上限使用方案 B：完整 sensor forward。**它验证当前 simulator 与 inverse 是否闭环。

如果两者都优于 APS-only，并在小幅 PSF、阈值和 core gain 失配下仍成立，才更有理由认为事件确实带来可迁移的信息增益。

## 6. APS frame 应怎样进入重建

预测 APS 不应使用 clean 当前的单一全局 motion kernel。应由同一个 fibre forward 对整个曝光积分：

$$
B_p^{pred}(O)
=\frac{1}{T}\int_{t_0}^{t_1}
I_p(t;O)\,dt.
$$

离散实现与 simulator 一致，使用 251 个端点做梯形平均。

帧损失可先使用：

$$
\mathcal L_F
=\frac{1}{N_p}\sum_p
\left(\sqrt{B_p^{pred}+\epsilon}
-\sqrt{B_p^{obs}+\epsilon}\right)^2.
$$

总损失第一版可写为：

$$
\mathcal L
=\lambda_F\mathcal L_F
+\lambda_E\mathcal L_E
+\lambda_{TV}\operatorname{TV}(O)
+\lambda_+R_+(O).
$$

$R_+$ 表示非负性约束或采用 softplus 参数化。

## 7. clean 代码逐模块复用判断

| clean 模块/逻辑 | 是否复用 | 处理方式 |
|---|---|---|
| `NeuroSRM_demo.py` CLI 风格 | 是 | 新建 fibre 入口，不改变旧入口 |
| `ExperimentConfig` dataclass 风格 | 是 | 新建 `FibreExperimentConfig` |
| `data.py/load_aedat4` | 否 | 新增 HDF5/NPY simulation loader |
| `ExposureSample` 思路 | 部分 | 新建包含 events、APS、motion、core map 的 sample |
| `events.py/bilinear_splat` | 部分 | 用于 fibre-aware feature map，不用于普通 warp |
| `warp_events_to_reference` | 否 | 固定芯斑事件不能按 sensor translation warp |
| `estimate_motion` 普通 CMax | 否 | 第一版固定真值运动；以后对 fibre forward 优化运动 |
| `PiecewiseLinearTrajectory` 思想 | 后期复用 | 改成按真实 timestamp 插值，不必逐微秒展开 |
| `predicted_iwe` | 否 | 替换为 core/sensor lin-log event likelihood |
| `dense_motion_blur_kernel` | 否 | 替换为 fibre forward 的曝光时间积分 |
| `blur_image` | 否 | 当前 APS 不是普通整幅图卷积后下采样 |
| `block_average(scale=2)` | 否 | sensor sampling 已包含在 relay operator 中 |
| `AdamP`、TV、smooth loss | 是 | 用于物体优化和正则化 |
| `output.py` 结构 | 是 | 扩展 core/event/forward diagnostics |
| reference comparison tests | 是 | 增加 forward parity 和消融测试 |

## 8. 建议新增的代码结构

为了不破坏 clean 与旧代码的数值一致性，建议新增文件而不是在现有函数中堆大量条件分支：

```text
isl_diff_event_clean/
  FibreNeuroSR_demo.py
  neurosr/
    fibre_config.py
    fibre_data.py
    fibre_forward.py
    fibre_event_loss.py
    fibre_pipeline.py
    fibre_output.py
  tests/
    test_fibre_forward.py
    test_fibre_event_loss.py
    test_fibre_reconstruction_smoke.py
```

### 8.1 `fibre_data.py`

读取：

```text
outputs/phase1_usaf/00_source/object_intensity.npy  # 只用于评价，不进入优化
outputs/phase1_usaf/01_motion/motion.npz
outputs/phase1_usaf/03_fibre/fibre_sequence.h5     # core centres / oracle diagnostics
outputs/phase1_usaf/04_sensor/sensor_sequence.h5   # forward parity diagnostics
outputs/phase1_usaf/05_aps/aps_frame.npy           # 正式 frame observation
outputs/phase1_usaf/06_events/events.h5             # 正式 event observation
```

训练/重建时不能把 `object_intensity.npy` 或完整 `core_signals` 当输入，否则会泄漏真值。它们只用于单元测试、oracle upper bound 和最终指标。

### 8.2 `fibre_forward.py`

必须使用可微 PyTorch 实现：

1. 用 `grid_sample` 按 $\mathbf s(t)$ 移动物体；
2. 用固定归一化圆盘 kernel 做 fibre aperture average；
3. 在 core centres 处双线性采样得到 $c_i(t)$；
4. 用预计算稀疏矩阵 $P_{pi}$ 渲染到 sensor；
5. 按时间分块计算，避免一次保存完整 autograd sequence；
6. 对 APS 可先积分 core signals，再乘 $P$，减少内存。

不要直接调用 simulator 中的 NumPy/OpenCV forward 参与优化，因为它不可对待恢复物体反向传播。可以用 simulator 结果校验 Torch forward 是否一致。

### 8.3 `fibre_event_loss.py`

负责：

- events 按时间和 pixel/core 分 bin；
- ON/OFF 分开累计；
- v2e-compatible lin-log；
- core-normalized weights；
- cumulative event residual；
- Huber/Charbonnier robust loss；
- 输出每芯 event residual 和置信度图。

### 8.4 `fibre_pipeline.py`

建议五个明确阶段：

1. 数据和几何校验；
2. Torch forward 与 simulator ground truth parity；
3. APS-only 初始化；
4. 固定运动的 APS + event 联合重建；
5. 保存结果、消融和指标。

第一版不要同时优化物体、运动、PSF、threshold 和 core gains，否则变量之间会严重耦合，失败时也无法定位原因。

## 9. 分阶段实施路线

### 阶段 0：验证可微 forward 完全正确

把仿真真值物体送入新 Torch forward，要求：

- core signals 与 simulator 输出一致；
- sensor instantaneous frames 与 HDF5 一致；
- APS 重投影与 `aps_frame.npy` 一致；
- 使用真值物体时 event residual 显著低于随机物体。

如果这一阶段不通过，不应开始优化重建。

### 阶段 1：连续 core signals oracle upper bound

暂时把 simulator 保存的 `core_signals[T,N]` 当观测，测试已知运动下能否从圆孔扫描测量恢复物体。

这不是最终方案，因为真实 DAVIS 不直接提供连续 core signals；它只回答：

> 当前轨迹、芯径、芯间距和视场从信息上是否足以支持所要求的重建网格？

如果连 oracle continuous measurements 都不能恢复目标，就不应期待阈值化 events 能恢复。

### 阶段 2：APS-only baseline

只用一张 APS 和 TV/非负约束重建，记录蜂窝残留和 USAF 对比度。这是判断事件是否真正增益的必要基线。

### 阶段 3：加入 core-domain events

使用方案 A。比较：

```text
APS-only
events-only
APS + events
continuous core oracle
```

events-only 应允许存在全局亮度尺度或低频不确定性；APS + events 应同时改善低频稳定性和扫描方向细节。

### 阶段 4：完整 sensor-event likelihood

使用方案 B 复现当前 v2e events。检查结果是否与 core-domain 结论一致，并测试改变 spot sampling 数量后结果是否被人为改善。

如果 sensor 版本显著优于 core 版本，但优势随一个芯斑覆盖的 pixel 数增加而增长，应警惕把冗余 pixel 当成独立信息。

### 阶段 5：模型失配和二维运动

依次加入：

- 第二个非共线运动方向；
- event threshold 偏差；
- core gain mismatch；
- spot PSF mismatch；
- sensor noise；
- 模式图案轻微时变。

每次只加入一种失配并重新比较 joint 与 APS-only。

## 10. 重建网格应怎样选择

当前 forward truth 使用 0.5 um/px，只表示数值仿真网格足够细，不表示系统能恢复 0.5 um 分辨率。

建议从相对芯间距的 2x 网格开始：

$$
\Delta u_{recon}=\frac{4.5}{2}=2.25\ \mathrm{um/px}.
$$

成功后再尝试：

```text
2.25 um/px -> 1.125 um/px -> 0.5 um/px
```

每一级都需要证明：

- joint 优于 APS-only；
- 不是仅仅输出像素更多；
- USAF 可分辨空间频率确实提高；
- 对模型小失配仍稳定。

直接优化 320 x 320 的 0.5 um 网格可能得到一张看起来锐利但高度依赖 TV 和 exact model 的图，不应作为第一步。

## 11. 运动应该怎样处理

### 第一版

直接使用 `motion.npz` 的真值轨迹，固定不优化。这样首先隔离“事件是否能帮助物体重建”这个问题。

### 后续未知运动

可以复用 clean 的分段线性参数化思想，但优化目标必须经过 fibre forward：

$$
\min_{O,\mathbf s(t)}
\lambda_F\mathcal L_F(O,\mathbf s)
+\lambda_E\mathcal L_E(O,\mathbf s)
+R(O)+R(\mathbf s).
$$

不能继续使用现有 `estimate_motion()` 把固定近端事件坐标做普通平移 CMax。更合适的初始化来源是：

- 已知机械台轨迹；
- 外部位置传感器；
- core intensity sequence 的时延/相关性；
- 在 fibre-aware forward 下对少量轨迹节点做粗到细搜索。

## 12. 如何判断“真的重建出来了”

至少同时报告：

### 数值指标

- valid ROI PSNR、SSIM；
- USAF 条纹对比度和可分辨 element；
- 水平与垂直方向分别统计的边缘/频率响应；
- APS reprojection error；
- event cumulative residual；
- 不同随机初始化的方差。

### 必需消融

- APS-only；
- events-only；
- APS + events；
- 连续 core oracle；
- 错误运动或打乱事件时间的负对照；
- 单方向与二维方向运动。

### 防止虚假成功

- 真值物体只用于评价，不能初始化正式重建；
- forward 和 inverse 至少做一组轻微参数失配；
- spot pixels 必须按 core 归一化，避免重复计权；
- 改变输出网格密度不能被当成分辨率提高；
- 不能只展示经过各自归一化的 PNG，必须比较原始线性数值。

## 13. 最终建议

基于当前数据，推荐的开发决策是：

1. 保留现有 `isl_diff_event_clean/NeuroSRM_demo.py` 不动，避免破坏与旧代码一致性。
2. 在 clean 工程中新建独立的 fibre reconstruction pipeline。
3. 第一版固定真值运动和 fibre 几何，只优化非负物体。
4. 先做 continuous core oracle，确认信息上限。
5. 正式事件基线优先使用按芯聚合、按芯归一化的 cumulative event likelihood。
6. 再实现 raw sensor-event forward，作为当前 simulator 的 exact-model 回归。
7. 重建网格先从 2.25 um/px 开始，不直接宣称恢复 0.5 um 分辨率。
8. 加入第二个运动方向后，再评价二维超分辨能力。
9. 真实数据阶段使用实测 core map、逐芯 spot PSF/gain 和 event threshold 标定。
10. 如果近端模式图案明显随输入变化，升级到 mode-dependent PSF 或 transmission matrix；否则优先聚合总芯通量以降低模型复杂度。

最准确的预期是：

> 当前数据足以尝试并很可能完成一个已知运动、exact-model 条件下的 fibre-aware 联合重建闭环；它有希望改善蜂窝采样和水平方向细节，但在加入二维轨迹和模型失配验证之前，不能宣称已经实现真实、各向同性的多芯光纤 NeuroSR。
