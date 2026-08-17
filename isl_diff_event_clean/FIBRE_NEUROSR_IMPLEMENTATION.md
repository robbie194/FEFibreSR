# FibreNeuroSR：方案 A 的完整实现与结果

## 1. 最终结论

`FibreNeuroSR_demo.py` 已经跑通 `fibre_frame_event_sim` 当前生成的数据，并输出一张连续、无芯斑空隙的远端物体强度图。联合 APS 与事件后，相比 APS-only：

| 方法 | PSNR | SSIM | 相关系数 |
|---|---:|---:|---:|
| APS 初始插值 | 14.76 dB | 0.611 | 0.835 |
| APS-only 优化 | 15.76 dB | 0.728 | 0.869 |
| APS + core events | **17.16 dB** | **0.767** | **0.907** |

因此当前仿真闭环能够实现：

- 把离散蜂窝纤芯读出变成连续 frame；
- 显著压低规则芯格/蜂窝纹理；
- 利用 4.5 um 水平扫描增加采样相位；
- 相比仅使用 APS，恢复更清楚的 USAF 边缘和条纹。

这里应准确称为**沿运动方向获得额外采样的方向性超分辨**。因为轨迹只有水平分量，当前结果不证明各向同性二维超分辨，也不声称恢复了被 2.9 um 芯孔径完全截止的空间频率。

## 2. 正式重建读取什么

优化读取：

- `01_motion/motion.npz`：已知远端运动；
- `05_aps/aps_frame.npy`：绝对强度和低频锚；
- `06_events/events.h5`：高时间分辨率亮度变化；
- YAML 中的纤芯、中继、DAVIS 和事件阈值参数。

`object_intensity.npy` 和 `core_signals` **不进入优化损失，也不用于初始化**。它们只用于两件事：

1. 开始优化前验证可微前向是否忠实复现 simulator；
2. 结束后计算 PSNR、SSIM 等仿真评价指标。

真实实验没有真值时，可以直接去掉这两项评价，重建本身仍只依赖 APS、events、轨迹和标定。

### 【进一步分析 2026-08-16：问题 1】motion.npz 到底是什么，真实实验怎样获得运动

#### 1. 这个文件只有轨迹，不包含图像

`motion.npz` 当前只保存两个数组：

```text
timestamps_s   shape=[T]     每个运动采样时刻，单位 s
shifts_xy_um   shape=[T, 2]  对应时刻的远端物体 (x, y) 位移，单位 um
```

phase 1 文件的实际内容是：

```text
T = 251
t: 0 -> 0.025 s，间隔 0.0001 s
shift: (0, 0) -> (4.5, 0) um
```

它不是 frame、事件图或物体纹理，只是一个带时间戳的二维位移表。仿真中轨迹由配置直接生成，所以重建读取的是无误差真值轨迹。

还要区分三种坐标：

- `shifts_xy_um`：远端物体相对固定光纤端面的位移；
- `core_centres_xy_um`：远端纤芯中心在光纤端面的固定位置；
- DAVIS event `(x,y)`：近端芯斑落在 sensor 上的 pixel 坐标。

当前模型真正采样的远端物体位置近似围绕：

$$
\mathbf q_i(t)
=
\frac{\mathbf r_i}{M}-\mathbf s(t),
$$

其中 $\mathbf r_i$ 是第 $i$ 根纤芯中心，$M$ 是 GRIN magnification，$\mathbf s(t)$ 是 `shifts_xy_um`。严格模型不是在 $\mathbf q_i(t)$ 取一个点，而是在该位置附近按 GRIN PSF 和圆形芯孔径做积分。

#### 2. 真实实验中的运动是否一定不知道

不一定。如果远端 target 安装在压电台、voice-coil 或精密二维位移台上，至少有三层运动信息：

1. 控制器的命令轨迹；
2. stage encoder 的实测轨迹；
3. 光学系统中目标真正相对纤芯端面的轨迹。

三者不会完全相同。命令轨迹可以作为初始化，encoder 通常比命令值可靠，但最终仍要标定 actuator scale、坐标轴旋转、零点、延迟和 APS/event 时间同步。论文中的 `known motion` 应指经过这些标定后可用的轨迹，而不是简单把控制命令当作绝对真值。

对第一版真实实验，推荐主动施加已知微扫描并读取 encoder。这样可以先验证事件是否真的提升空间分辨率，不要同时把“未知图像”和“完全未知运动”两个困难叠加在一起。

#### 3. 能否只根据 APS 和事件自己估计轨迹

可以，但**可行不等于容易**，而且当前 `FibreNeuroSR_demo.py` 尚未实现这一步。当前 fibre pipeline 直接读取 `motion.npz`，优化变量只有 latent image；它不会更新 `shifts_xy_um`。

普通场景版 `NeuroSRM_demo.py` 已有事件运动估计：先把事件切成多个 event frames，通过相位相关得到分段位移初值，再用 warped-event contrast 优化连续轨迹。但是该方法不能原样套到 fibre 数据上，原因是：

> 普通相机中，物体移动时事件的 sensor 坐标随图像移动；当前 fibre 模型中，近端芯斑位置固定，变化的是每根固定芯斑的亮度。

因此不能直接观察 DAVIS 平面上的整幅纹理平移。更适合 fibre 的估计路线是：

1. 先将 DAVIS events 标定并聚合到 core lattice；
2. 把每芯累计事件放在对应的远端 core centre，构造稀疏 core-event frames；
3. 用相位相关或 core-domain contrast maximization 得到粗轨迹；
4. 将轨迹参数化为常速度、分段线性或 B-spline 控制点；
5. 在同一个可微前向中联合优化物体 $O$ 与轨迹 $\mathbf s(t)$；
6. 固定 $\mathbf s(0)=(0,0)$，并加入速度、加速度和路径幅度正则，消除坐标零点与不合理抖动。

联合目标可写成：

$$
\min_{O,\,\mathbf s}
\mathcal L_{APS}(O,\mathbf s)
+\lambda_e\mathcal L_{event}(O,\mathbf s)
+\lambda_O R_O(O)
+\lambda_s R_s(\mathbf s).
$$

这个问题是非凸的，并存在图像与运动互相解释的耦合：错误边缘可能被错误轨迹补偿，低纹理目标也很难提供运动信息。因此需要较好的 stage/encoder 初值、多尺度优化、低维轨迹模型和留出时间点验证。

实际难度可以概括为：

| 条件 | 运动估计难度 |
|---|---|
| 已知恒速方向，只估计速度大小和时间偏移 | 较容易 |
| 已知轨迹形状，只估计 scale、rotation、delay | 可控，建议先做 |
| 分段平滑二维轨迹，有 encoder 初值 | 中等 |
| 完全未知二维轨迹，只靠单次 APS + core events | 较难，容易与图像耦合 |

所以答案是：轨迹可以自己估计，但真实系统的第一阶段最好采用“encoder/命令初始化 + 数据小范围修正”，而不是一开始完全盲估计。

## 3. 数据怎样从 DAVIS pixels 变成纤芯通道

当前 simulator 中，每根纤芯只有一个随时间变化的标量强度。近端一个芯斑覆盖的多个 DAVIS pixels 是冗余读出，不是多个远端亚像素。

代码先用均匀物体做一次几何 flat-field 标定，为每芯选择 4 个高增益且不重叠的 DAVIS pixels。然后：

- APS pixel 除以固定 gain，再对一芯内读数取 median，得到每芯绝对强度；
- ON/OFF 事件按芯和时间累计；
- 一芯内 4 路累计事件先平均，再进入 event loss，避免把它们算成 4 个独立空间观测。

标定验证结果：

- core APS 相对 simulator core temporal average 的 RMSE：`2.64e-7`；
- 相关系数：几乎为 `1.0`。

这说明从 sensor APS 回到 core domain 的过程在当前仿真中是准确的。

### 【进一步分析 2026-08-16：问题 2】为什么不是把芯内事件全部放到芯中心再 warp

#### 1. 先给最简单的结论

你说的“把一根芯内的 event 都当成该芯中心，然后按运动 warp”是一个**可以做的 core-domain event image 方法**。它适合拿来做可视化，或者给未知运动提供粗初值。

但是，当前 `FibreNeuroSR_demo.py` 的正式重建**没有**这样做。它不构造 event image、不逐 event warp；它把每根纤芯当作一个随时间变化的亮度通道，然后让一个物理前向模型去预测这 1415 路亮度如何随远端运动变化。

最重要的区别是：

```text
普通 event warp：移动的是每个 event 的图像坐标
当前 fibre 重建：固定的是近端芯斑坐标；移动的是远端物体，
                  因而同一根固定纤芯看到的总亮度随时间变化
```

所以，`warp` 在普通场景版 `NeuroSRM_demo.py` 里是核心步骤；在当前 fibre 版中，运动通过可微前向的“移动远端取样坐标”进入，而不是通过修改每个 DAVIS event 的 `(x,y)` 进入。

#### 2. 整个过程分成三段：构造数据、归属事件、正式重建

| 阶段 | 发生在什么代码 | 输入和输出 | 是否把 event 放到芯中心 / warp |
|---|---|---|---|
| 生成模拟数据 | `fibre_frame_event_sim/src/fibre_sim/pipeline.py` 的 `generate_fibre_step()`、`generate_sensor_step()`、`generate_events_step()` | 远端物体 -> fibre frames -> DAVIS frames -> 原始 events | **否**。event 保留真实 DAVIS `(x,y)` |
| 把 DAVIS 数据归属到纤芯 | `isl_diff_event_clean/neurosr/fibre_data.py` 的 `build_core_calibration()`、`extract_core_aps()`、`aggregate_cumulative_events()` | APS/events -> 每芯 APS 与每芯事件时间序列 | **不 warp**。只用 `(x,y)` 查“属于哪根芯” |
| 正式反演 | `isl_diff_event_clean/neurosr/fibre_pipeline.py` 的 `_optimise()`，以及 `neurosr/fibre_forward.py` 的 `FibreCoreForward.forward()` | latent image + 已知位移 -> 预测 core APS/core events | **不 warp 原始 event**。模型移动远端物体的取样位置 |

下面按这三段解释。阅读这一节时可以先记住：**只有第二段读取原始 event `(x,y)`；进入第三段后，原始 `(x,y)` 已经不再使用。**

#### 3. 第一段：模拟数据怎样产生，为什么 event 最初不在芯中心

模拟阶段先在远端生成物体，再经过纤芯和 relay：

```text
远端物体 O 与运动 s(t)
  -> 每根纤芯的总通光量 c_i(t)
  -> 近端 fibre spot 图像
  -> DAVIS irradiance frame
  -> v2e 生成原始 events: (t, sensor_x, sensor_y, polarity)
```

对应函数关系是：

```text
generate_fibre_step()
  -> simulate_fibre_sequence()
  -> 每一时刻每根芯的标量 core_signals[t, i] = c_i(t)

generate_sensor_step()
  -> relay_to_sensor_sequence()
  -> 把固定的近端芯斑成像到 DAVIS pixels

generate_events_step()
  -> generate_v2e_events()
  -> 对 DAVIS frame 序列产生原始 event 坐标
```

这里没有任何一步把 event 坐标改成 core centre。原因很简单：v2e 面对的是 sensor image，它自然输出的是 sensor pixel 坐标。

当前 simulator 的关键假设是：一根纤芯只传输一个标量 $c_i(t)$。同一近端芯斑内第 $r$ 个 DAVIS pixel 只是这个标量的不同固定读出：

$$
I_{i,r}(t)
=
g_{i,r}\,c_i(t),
$$

其中 $g_{i,r}$ 是该 pixel 的固定 gain。因此一个芯斑内的多个 DAVIS pixels 不是多个远端空间位置。

#### 4. 第二段：重建读原始 event 时，到底做了什么

这是从 DAVIS pixels 变成 core channels 的地方，全部在：

```text
isl_diff_event_clean/neurosr/fibre_data.py
```

**步骤 4.1：先做 core-to-pixel 标定**

函数 `build_core_calibration()` 用均匀纤芯输入生成一张 flat-field，并为每根纤芯选择 4 个可靠 DAVIS pixels。它保存：

```text
pixel_xy[core, readout]  # 这根芯选中了哪 4 个 sensor pixels
gain[core, readout]      # 每个 selected pixel 的固定 gain
```

例如，假设第 17 根芯选中了四个 sensor pixels：

```text
core 17 -> (100, 80), (101, 80), (100, 81), (101, 81)
```

这只是建立“哪些 DAVIS pixels 是 core 17 的读出”的字典，不是重建，也没有 warp。

**步骤 4.2：APS 的处理**

函数 `extract_core_aps()` 读取 APS frame，在上述四个位置取值、除 gain、取中位数：

```text
APS[100,80], APS[101,80], APS[100,81], APS[101,81]
  -> 除对应 gain
  -> median
  -> core_aps[17]
```

所以 APS 从 `260 x 346` 个 sensor pixels 变成 1415 个 core 值。

**步骤 4.3：events 的处理**

函数 `aggregate_cumulative_events()` 读取原始 event：

```text
(time, sensor_x, sensor_y, polarity)
```

它先建立 lookup table：

```text
(100, 80) -> (core=17, readout=0)
(101, 80) -> (core=17, readout=1)
...
```

每个原始 event 的 `(sensor_x, sensor_y)` 只被用这一次：查找它属于哪个 `(core, readout)`。随后代码把 ON/OFF event 按阈值变成正/负增量，并累积成：

```text
cumulative_event_change[time, core, readout]
```

对于 core 17，四路 selected pixels 最终得到的是四条随时间增长或下降的累计曲线，而不是四个移动坐标。

当前实现只使用每芯选中的 4 个 readouts；同一芯斑中其他 sensor pixels 的 events 不进入 loss。这是为避免 spot overlap 和错误归属的保守选择。真实实验标定足够可靠时，可以改成使用完整 spot mask，并按 gain/threshold 加权。

**到这里为止，没有发生 warp。** 原始 event 坐标已经被转换为 `(core, readout)` 索引，之后不再出现于 fibre 重建的空间计算中。

#### 5. 第三段：正式重建中，运动究竟在哪里进入

正式重建的循环在：

```text
isl_diff_event_clean/neurosr/fibre_pipeline.py::_optimise()
```

其中最关键的一行是：

```python
core_signals = model(image, shifts)
```

这里的 `model` 是：

```text
isl_diff_event_clean/neurosr/fibre_forward.py::FibreCoreForward.forward()
```

它接收：

```text
image   = 当前待恢复的远端物体图像
shifts  = motion.npz 中的远端物体位移表
```

在 `FibreCoreForward.forward()` 中，代码用 `grid_sample()` 计算：对每个时刻、每个固定 fibre grid 位置，应该去远端图像的什么位置取样。核心公式是：

$$
\text{远端取样位置}
=
\frac{\text{固定 fibre 位置}}{M}
-\text{远端物体位移}(t).
$$

接着代码依次执行：

```text
移动后的远端取样
  -> 可选 GRIN blur
  -> 2.9 um 圆形芯孔径积分
  -> 在 1415 个固定 core centre 取值
  -> predicted core_signals[t, core]
```

然后 `_optimise()`：

```text
predicted core_signals
  -> 时间平均 -> predicted core APS
  -> lin-log response -> predicted cumulative core events
  -> 与第二段得到的 core APS/core events 比较
```

也就是说，**远端物体在模型里移动，近端 core centre 和 DAVIS spot 都不移动。** 这就是 fibre 版中相当于“利用运动”的位置。

#### 6. 你说的“放到芯中心再 warp”具体对应哪段现有代码

如果按你的想法实现，伪代码会像这样：

```text
for each raw event (t, sensor_x, sensor_y, polarity):
    core = lookup_core[sensor_y, sensor_x]
    x_core, y_core = core_centres[core]
    x_warp, y_warp = warp(x_core, y_core, motion_at(t))
    iwe[x_warp, y_warp] += polarity
```

这属于 IWE (image of warped events) / contrast-maximization 思路。仓库中真正执行这类操作的是普通场景版：

```text
neurosr/pipeline.py::_warp_and_render()
  -> neurosr/events.py::warp_events_to_reference()
  -> event_image()
```

该路径由 `NeuroSRM_demo.py` 使用，**不被 `FibreNeuroSR_demo.py` 调用**。所以不要把普通 `NeuroSRM_demo.py` 的 sensor-event warp 和 fibre 重建混为同一算法。

在 fibre 中，这种“先归芯、再放到 core centre、再 warp”的 IWE 可以作为后续的运动初始化模块，但使用前要先将每芯事件归一化或平均。若直接把一个芯内所有 raw events 全部叠到同一 centre，覆盖更多 DAVIS pixels 的芯会被重复加权，不能代表更多远端空间采样。

#### 7. 为什么 fibre 的正式损失不采用这个 warp 作为主模型

当前正式方法保留每芯的时间亮度模型：

$$
c_i(t)
=
\int
A_i(\mathbf u-\mathbf r_i)
\,[h_{GRIN}*O](\mathbf u-\mathbf s(t))
\,d\mathbf u.
$$

它表达的是“固定的第 $i$ 根芯在时刻 $t$ 收到多少总光”，而不是“第 $i$ 根芯的一个点坐标移动到哪里”。其优势是能同时利用：

1. 芯孔径不是点而是有限圆形面积；
2. APS 的绝对亮度；
3. 每个 readout 的 gain；
4. ON/OFF event 的 lin-log 阈值变化。

如果只做 centre warp IWE，适合让事件在某个运动假设下变清晰，却不直接保证重建结果能解释 APS 强度，也不显式建模芯孔径积分。

因此当前选择不是“芯中心方法错误”，而是：

```text
芯中心 + warp IWE      -> 很适合可视化和估计运动初值
每芯光通量物理前向     -> 当前正式重建使用，用于恢复强度图
```

#### 8. 真实实验的边界

以上解释依赖当前方案 A 的标量 per-core 假设：同一芯内的 sensor pixels 都是 $c_i(t)$ 的固定 gain 读出。真实光纤若存在随远端入射位置、弯曲、波长或相干 speckle 改变的芯内模式分布，这个假设可能失效。

那时芯斑内部位置可能确实包含额外状态信息，但它不能未经标定就被当作远端亚像素。需要先测量多模式响应、状态相关 spot footprint，或更一般的 transmission operator，再决定是否把芯内事件位置纳入重建。

## 4. 可微前向模型

待恢复物体先在 `112 x 112` 潜在网格上表示，再双线性展开到 simulator 的 `400 x 400`、`0.5 um/px` 网格。潜在网格覆盖 200 um，等效间隔约 1.79 um；它比 4.5 um 芯间距密，但没有把 400 x 400 每个像素都当成可独立恢复的自由度。

对每个已知位移，前向依次执行：

1. 根据物体位移在远端物体上取样；
2. 映射到 `320 x 320` 光纤输入面；
3. 用 2.9 um 圆形芯孔径做面积平均；
4. 在 1415 个固定纤芯中心取值；
5. 得到 `core_signals[time, core]`；
6. 时间积分得到 core APS；
7. 通过 v2e 的 lin-log 响应预测累计事件变化。

Torch 前向与原 OpenCV simulator 的一致性为：

| 指标 | 数值 |
|---|---:|
| core signal MAE | `2.50e-4` |
| core signal RMSE | `5.95e-4` |
| core signal 相关系数 | `0.999998` |

少量误差来自 OpenCV 插值查表与 PyTorch 双线性插值的实现差别，不改变前向物理含义。

### 【进一步分析 2026-08-16：问题 3】112 的依据、1.79 um 的计算与完整可微过程

#### 1. 112 是怎样来的

先明确结论：`112` 不是作者论文给出的数值，也不是由 core pitch、Nyquist 定理或某个光学公式唯一推导出的结果。它是当前方案 A 实现中手动选择的 reconstruction hyperparameter：

```python
latent_shape = (112, 112)
```

它的作用是限制未知图像的自由度，从而在分辨率、稳定性和计算量之间折中。

当前不同尺度为：

| 项目 | 数值 |
|---|---:|
| simulator source | `400 x 400` pixels |
| source pixel size | `0.5 um` |
| 名义物方视场 | `200 x 200 um` |
| latent image | `112 x 112` unknowns |
| latent unknown 数量 | 12,544 |
| 若直接优化 400 x 400 | 160,000 unknowns |
| core 数量 | 1,415 |

`112 x 112` 只保留了 `400 x 400` 参数量的 `7.84%`。这样可以减少稀疏纤芯采样产生的巨大零空间，降低蜂窝伪影和噪声拟合，也减少优化难度。

它还给出以下直观采样密度：

$$
\frac{4.5}{1.786}
\approx2.52
$$

即一个 `4.5 um` core pitch 约含 2.52 个 latent samples；一个 `2.9 um` core diameter 约含：

$$
\frac{2.9}{1.786}
\approx1.62
$$

个 latent samples。它比原始 core lattice 密，允许表达亚芯距结构，但 latent grid 较密本身不等于物理上已经恢复了这些结构；真正可恢复的频率仍由 aperture、轨迹、事件、噪声和正则共同决定。

#### 2. 1.79 um 具体怎样计算

采用名义视场计算：

$$
W
=
400\ \mathrm{pixels}
\times0.5\ \mathrm{um/pixel}
=
200\ \mathrm{um}.
$$

因此每个 latent cell 的名义宽度为：

$$
\Delta_{latent}
=
\frac{200}{112}
\approx1.7857\ \mathrm{um}.
$$

这就是文中的约 `1.79 um`。

若严格按 `align_corners=True` 的像素中心距离计算，400 个 source pixel 的首尾中心距离是：

$$
(400-1)\times0.5
=
199.5\ \mathrm{um},
$$

112 个 latent pixel 有 111 个中心间隔，因此是：

$$
\frac{199.5}{112-1}
\approx1.7973\ \mathrm{um}.
$$

`1.7857` 与 `1.7973` 的区别只来自“按 pixel cell 宽度”还是“按首尾 pixel centre”计数。文中的 `1.79 um` 是尺度说明，不应被解释成经过实验标定的实际分辨率。

#### 3. 112 是否是最佳选择

目前没有证据证明它最佳。它是跑通方案 A 时选择的经验折中，下一步应做 latent-size ablation，例如：

```text
64, 80, 96, 112, 128, 160, 200
```

**【新增数值结果 2026-08-16】** 为了先分离“网格表达能力”和“光纤反演难度”，使用不同 latent size 直接拟合同一个 GT observable crop，不经过 GRIN、纤芯、APS 或事件模型。2500 次 Adam 后的 representational oracle 为：

| latent size | 名义间隔 | unknowns | oracle PSNR | oracle SSIM | oracle correlation |
|---:|---:|---:|---:|---:|---:|
| 64 | 3.125 um | 4,096 | 16.60 dB | 0.694 | 0.894 |
| 80 | 2.500 um | 6,400 | 17.52 dB | 0.747 | 0.915 |
| 96 | 2.083 um | 9,216 | 18.60 dB | 0.790 | 0.935 |
| **112** | **1.786 um** | **12,544** | **19.59 dB** | **0.833** | **0.948** |
| 128 | 1.563 um | 16,384 | 20.52 dB | 0.857 | 0.959 |
| 160 | 1.250 um | 25,600 | 22.02 dB | 0.901 | 0.971 |
| 200 | 1.000 um | 40,000 | 23.96 dB | 0.934 | 0.981 |

这张表证明更大网格确实能表达更多 GT 细节，也证明 `112` 不是表示能力的上限最优点。但它**不能**推出正式重建应该直接改成 200：该 oracle 把 GT 直接交给优化器，没有观测欠定、事件量化、噪声和模型失配。网格越大，在真实逆问题中越可能利用零空间拟合噪声。因此还必须做下面的完整 reconstruction ablation。

需要同时比较：

- object GT、同参数化 oracle 和光学带限指标；
- APS/event 留出时间点重投影；
- x/y MTF 或 USAF line-pair contrast；
- 对噪声和运动误差的稳定性；
- 运行时间和显存。

在仿真中可以用 GT 分析表示能力，但真实实验选择 latent size 时不能只追求训练观测残差最低，否则更大的网格很容易拟合噪声。应通过留出观测、重复实验和稳定性共同选择。

#### 4. 可微前向从 latent image 到 core signals 怎样计算

记待优化的 latent image 为：

$$
Z\in[0,1]^{112\times112}.
$$

**步骤 A：展开到 simulator source grid**

代码用 bilinear interpolation 得到：

$$
O
=
\operatorname{BilinearUpsample}(Z)
\in[0,1]^{400\times400}.
$$

这一步只是可微表示映射，不会凭空增加独立自由度。输出虽然是 `400 x 400`，独立未知量仍只有 12,544 个。

**步骤 B：按远端位移采样到 fibre input grid**

fibre input grid 为 `320 x 320`、`0.5 um/pixel`。对每个时刻 $t$ 和 fibre grid 物理坐标 $(x_f,y_f)$，对应 source pixel 坐标为：

$$
x_s
=
\frac{W_s-1}{2}
+\frac{x_f/M-s_x(t)}{p_s},
$$

$$
y_s
=
\frac{H_s-1}{2}
+\frac{y_f/M-s_y(t)}{p_s},
$$

其中 $p_s=0.5\ \mathrm{um/pixel}$。PyTorch `grid_sample` 在 $O$ 上做双线性采样，得到每个时刻的 fibre input image。

**步骤 C：GRIN PSF**

如果 `grin.sigma_um > 0`，先把物理 sigma 换算成 fibre-grid pixels：

$$
\sigma_{px}
=
\frac{\sigma_{um}}{0.5\ \mathrm{um/pixel}},
$$

然后用归一化 Gaussian kernel 卷积。`sigma=0` 时跳过该卷积。

**步骤 D：圆形芯孔径积分**

代码按照 `2.9 um` 直径和 supersampling 构造 fractional pixel coverage kernel，归一化后卷积。这相当于计算每个可能芯位置附近的 aperture average，而不是读取一个中心 pixel。

**步骤 E：在固定 core centres 取值**

使用第二次 `grid_sample` 在 1415 个 `core_centres_xy_um` 上取 aperture average，并乘 GRIN 与 fibre transmission，得到：

$$
C(O,\mathbf s)
\in\mathbb R^{T\times1415}.
$$

这就是 `core_signals[time, core]`。

**步骤 F：生成 APS 与 event prediction**

APS prediction 对时间做梯形积分：

$$
\hat A_i
=
\frac{1}{T_e}
\int_0^{T_e} C_i(t)\,dt.
$$

Event prediction 则先乘每个 selected DAVIS readout 的 gain 和 `input_white_dn`，经过 lin-log response，再减去初始响应，得到每芯每路累计变化。

**【进一步补充 2026-08-17：输入预处理与联合 loss 怎样对齐】**

“联合”不是把 APS frame 和 event image 拼成一张图，而是：**同一个 latent image 经过同一个前向模型生成两种预测，分别与 APS 观测和 event 观测比较，最后把两个标量 loss 相加。**

观测预处理集中在 `neurosr/fibre_data.py::load_fibre_observations()`：

| 数据 | 预处理函数 | 送入优化的结果 |
|---|---|---|
| `aps_frame.npy` | `extract_core_aps()` | 每芯4个 selected pixels 除 gain 后取 median，得到 `observed_aps[1415]` |
| `events.h5` | `aggregate_cumulative_events()` | event 的 `(x,y)` 只用于查所属 core/readout；ON/OFF 按阈值累计，得到 `observed_events[T,1415,4]` |
| `motion.npz` | `load_fibre_observations()` | 得到 `timestamps[T]` 和 `shifts[T,2]` |

APS 在联合重建中还有一个初始化用途：`fibre_pipeline.py::_initial_image()` 把 1415 个 `core_aps` 放在平均位移对应的物方位置并插值成初始图，再缩小到 `112 x 112` latent。它只是优化初值；真正约束结果的仍是下面的 loss。

在 `neurosr/fibre_pipeline.py::_optimise()` 中，代码先每隔 `event_time_stride=5` 取一个时刻，然后执行：

```python
core_signals = model(image, shifts)  # [T_used, 1415]

predicted_aps = trapezoidal_average(core_signals, times)  # [1415]
aps_loss = mse(predicted_aps, observed_aps)

predicted_events = predict_cumulative_event_change(
    core_signals, gain, input_white_dn
)  # [T_used, 1415, 4]
event_loss = huber(
    predicted_events.mean(dim=-1),
    observed_events.mean(dim=-1),
)
```

两条分支的对应关系是：

```text
同一个 predicted core_signals[T, core]
  ├─ 时间积分 ─> predicted_aps[core]
  │              对 observed_aps[core] 做 MSE
  │
  └─ gain + lin-log + 减初值 ─> predicted_events[T, core, readout]
                                 对 observed_events 先在4路readout取平均，
                                 再做 Huber loss
```

数学上：

$$
\mathcal L_{APS}
=
\frac{1}{N_c}\sum_i
\left(\hat A_i-A_i^{obs}\right)^2,
$$

$$
\mathcal L_{event}
=
\operatorname{Huber}
\left(
\frac{1}{4}\sum_r\hat E_{i,r}(t),
\frac{1}{4}\sum_r E_{i,r}^{obs}(t)
\right).
$$

最后 joint mode 使用：

$$
\mathcal L_{joint}
=
\mathcal L_{APS}
+0.03\mathcal L_{event}
+0.0015\mathcal L_{TV}.
$$

三项 loss 共享同一个 latent image，所以 APS 会把绝对亮度拉回正确尺度，events 会约束曝光期间每根芯的亮度变化，TV 则抑制欠定解中的蜂窝和高频噪声。反向传播更新的是 latent image，不是输入 APS、events 或固定运动轨迹。

**步骤 G：反向传播**

上述 bilinear sampling、Gaussian convolution、aperture convolution、core sampling、时间积分和 lin-log response 都由 PyTorch tensor 运算构成。损失对 $C$ 的梯度可以逐层传播回 $O$，再传播回 $Z$：

```text
APS/event/TV loss
  <- predicted APS and events
  <- core signals
  <- core sampling and aperture integration
  <- GRIN blur and motion sampling
  <- 400 x 400 expanded object
  <- 112 x 112 latent parameters
```

Adam 更新的实际对象是 `112 x 112` 的 $Z$；每次更新后将其限制在 `[0,1]`。运动轨迹在当前 fibre pipeline 中只是前向输入，尚未参与梯度更新。

#### 5. 为什么这仍然可以称为超分辨，而不是普通插值

普通插值只把每芯一个 APS 平均值填进空隙，不要求插值图重新解释时间事件。当前方法要求同一 latent object 在已知亚芯距运动下，同时生成正确的：

- 1415 个芯的曝光平均；
- 每根芯随时间的 lin-log 变化；
- 芯孔径积分和 GRIN blur。

所以新增信息来自运动产生的多个采样相位和事件时间约束，而不是来自把 `112 x 112` 放大到 `400 x 400` 这个操作。`112` 只是允许算法表达这些信息的参数网格，不是超分辨证据本身。

## 5. 损失为什么这样写

总损失为：

$$
\mathcal L = \mathcal L_{APS} + 0.03\mathcal L_{event} + 0.0015\mathcal L_{TV}.
$$

- `APS loss` 拟合曝光期间的每芯时间平均，约束绝对亮度；
- `event loss` 拟合每芯累计 lin-log 变化，并用 Huber 抵抗事件阈值量化余数；
- `TV loss` 抑制欠定逆问题在六角芯格位置产生周期纹理。

事件每 5 个 simulator timestamps 取一次，即每 0.5 ms 形成一个累计约束，同时保留曝光终点。最终验证仍使用完整数据。

真值前向的累计事件残差 MAE 是 `0.0248`，随机物体是 `0.1763`；联合结果是 `0.0246`。这说明重建已经把事件拟合到 v2e 阈值量化允许的误差量级，而不是只得到一张视觉上相似但不解释事件的图片。

## 6. 三组实验怎样理解

### APS-only

能恢复绝对亮度并去掉显式芯斑空隙，但一张长曝光只给每芯一个时间平均，细节和运动方向采样相位有限。

### Events-only

事件只约束亮度变化，不给绝对 DC。代码故意从常数 0.5 初始化，不偷用 APS 结构。它的相关系数仍为 `0.814`，说明事件包含物体结构；但 PSNR 只有 `4.80 dB`，说明绝对亮度漂移严重。这不是实现失败，而是事件相机观测的固有零空间。

### APS + events

APS 锚定绝对强度，事件补充扫描过程中的时间变化。它同时取得最好的 PSNR、SSIM 和相关系数，是应当使用的正式输出。

## 7. 运行与输出

```bash
cd /home/robbie/tyf_code/EventCode/myFEFibreSR/isl_diff_event_clean
MPLBACKEND=Agg /home/robbie/miniconda3/envs/NeuroFibreSR/bin/python \
  FibreNeuroSR_demo.py
```

主输出目录：

```text
results/fibre_neurosr/phase1_usaf/
```

重点文件：

- `reconstruction_comparison.png`：真值、初始插值、三种重建的统一尺度对比；
- `joint/reconstruction.npy`：完整 400 x 400 连续重建；
- `joint/observable_reconstruction.png`：实际有纤芯扫描覆盖的有效区域；
- `joint/diagnostics.png`：loss、APS 重投影和事件残差；
- `run_summary.json`：参数、前向一致性和全部指标。

## 8. 当前边界和下一步

当前结果是 exact-model 仿真闭环，证明方案 A 在“每芯一个标量、固定近端 spot、已知运动”条件下成立。迁移到真实实验前仍需实测：

- 每芯 sensor gain 和归属区域；
- ON/OFF 阈值及 pixel 间离散性；
- 真实运动轨迹；
- 芯斑模式是否随远端入射位置变化；
- 芯间串扰和背景。

若要真正提高二维各向同性分辨率，下一组仿真应加入非共线二维轨迹，例如水平加垂直或小圆轨迹。只增加水平采样时间，不能补足垂直方向缺失的采样相位。

### 【进一步结果 2026-08-16：二维仿真已经完成】

上述非共线运动建议已经在 phase 2 中实现为：

```text
(0, 0) -> (4.5, 0) -> (4.5, 4.5) um
```

`sigma=0` 和 `sigma=0.8 um` 两套仿真、APS-only、events-only 和联合重建均已完成。相对各自 APS-only，联合结果的 PSNR、SSIM、x 梯度相关性和 y 梯度相关性全部提高。完整结果见：

```text
TWO_DIMENSIONAL_RECONSTRUCTION_REPORT.md
PHASE2_RECONSTRUCTION_QUESTIONS_AND_SIMULATION_ROADMAP.md
```

不过 phase 2 仍然读取仿真真值轨迹，没有实现 fibre-domain 自主运动估计。因此本次问题 1 所讨论的“encoder 初始化 + core-domain 轨迹修正”仍然是迁移真实实验前需要补齐的关键模块。
