# NeuroSR 新旧代码迁移对照

本文帮助同时阅读以下两个实现：

- 旧版：`../isl_diff_event/NeuroSRM_demo.py`
- 新版：`NeuroSRM_demo.py` 和 `neurosr/`

新版不是按文件逐句翻译旧版，而是先确定产生
`results/fig/tyf_test` 的有效计算链路，再按职责重新组织。因此，旧版中出现的
函数不一定会在新版中保留同名函数：有些被拆成更小的纯函数，有些被合并进一个
阶段函数，还有一些从未参与正式结果，因而被明确删除。

## 1. 先看整体执行关系

旧版把读取数据、初始化、三轮优化、画图和历史实验都写在模块顶层。新版入口只做
参数解析，实际流程集中在 `neurosr.pipeline.run_experiment()`：

```text
旧版模块顶层                         新版

读取 AEDAT4、选择曝光窗口       ->  load_aedat4()
                                   select_exposure_sample()

构造 13 个事件帧并配准          ->  estimate_motion()
                                   |_ _build_registration_frames()
                                   |_ initialize_segment_increments()

2100 次 contrast maximization    ->  estimate_motion()

2100 次 1x 图像/背景/轨迹优化   ->  refine_at_sensor_scale()

2100 次固定轨迹的 2x 重建       ->  reconstruct_at_two_x()

保存 NPY、PNG、摘要和总览图      ->  save_results()
```

`run_experiment()` 是阅读新版的最佳起点。它只表达阶段和数据流；每个数学细节再沿着
函数调用进入对应模块。

## 2. 主流程阶段对照

| 旧版位置或动作 | 新版位置 | 迁移说明 |
|---|---|---|
| 顶层常量 `sensor_size`、`read_path_e`、`num_pieces`、迭代数、倍率 | `neurosr/config.py` 的 `ExperimentConfig` | 所有可配置项集中管理，不再散落为全局变量。 |
| `load_events(eve_dtype="aedat4")` | `neurosr/data.py::load_aedat4()` | 只保留本实验真正使用的 AEDAT4 分支；返回具名的 `Recording`，不再依赖元组顺序。 |
| 三次重复读取 AEDAT4 | `load_aedat4()` 调用一次 | 重复读取不改变结果，只增加时间和内存开销。 |
| 手工选 `frame_ind`、`win_frame`、双曝光窗口 | `select_exposure_sample()` | 完整保留 frame 11、200000 us 曝光和 418341 us 事件窗口的选择规则。 |
| `time_window()` | `select_time_window()` | 保留闭区间 `[start, start + duration]` 语义。 |
| `np_to_torch()` 和 `tt0` | `_to_event_tensors()`；各阶段显式计算 `reference_time` | 数据类型、设备和参考时间不再隐藏在全局状态中。 |
| 头尾事件图、`corr2D()` 和 `Dxy_0` | 未迁入计算主链路 | 这段只显示粗略位移和图像；后续真正的 12 段初始化不使用 `Dxy_0`。 |
| 每个轨迹边界调用 `window_event_to_img()` | `_build_registration_frames()` + `numpy_event_frame()` | 保留 13 张事件帧、窗口宽度、双线性累加和去除最强像素的规则。 |
| `Simple_sr.pos_registration()` | `register_event_frames()` | 原类在此只用到了相位互相关，新版直接表达该算法，去掉与多帧超分辨无关的类状态。 |
| `flow_xy[1:] - flow_xy[:-1]` | `initialize_segment_increments()` | 保留 `(dy, dx)` 到 `(dx, dy)` 的换轴、符号变换和逐段差分。 |
| `ContinuousPiecewiseLinear_Dxy_pw` | `PiecewiseLinearTrajectory` | 仍将 12 个 `(dx, dy)` 段增量展开为逐微秒累计轨迹。属性统一为 `boundaries`、`segment_widths`、`segment_masks`。 |
| 第一轮 2100 次 `Dxy_pc` 优化 | `estimate_motion()` | Adam、学习率、StepLR、Gaussian blur、负 IWE 方差和轨迹正则项均保留。 |
| 第一轮循环中的 `Dxy_pred` | `MotionEstimate.dense_trajectory_xy` | 特意保留旧版“最终 optimizer step 前计算的 dense trajectory”，用于匹配旧版求值顺序。 |
| 第一轮循环后的 `Dxy_pc` | `MotionEstimate.segment_increments_xy` | 保留最终 optimizer step 后的段增量；因此它和 dense trajectory 存在旧版原有的一步时序差异。 |
| `M=1` 的图像、背景、`Dxy_pc_pred` 联合优化 | `refine_at_sensor_scale()` | 图像、背景和轨迹仍在 1x 阶段共同优化；各损失项改为具名变量。 |
| `M=2` 的固定轨迹重建 | `reconstruct_at_two_x()` | 从 1x 图像上采样；轨迹和 PSF 固定，只优化 2x 图像与 1x 背景。 |
| 顶层 `result_arrays`、`cv2.imwrite()`、`json.dump()` | `neurosr/output.py::save_results()` | 原始数组、显示图、运行摘要和对比总览统一保存。 |

## 3. 函数级对应关系

### 3.1 旧脚本内定义的函数

| 旧函数 | 新版对应 | 状态与原因 |
|---|---|---|
| `forward_model_EKLT_L(L, Dxy)` | `optimization.predicted_iwe(log_image, displacement_xy)` | 直接迁移并改名，明确输入已经是 log intensity。 |
| `forward_model_EKLT(I, Dxy, thre)` | `linear_log_intensity()` + `predicted_iwe()` | 拆开响应曲线和 EKLT 方程，调用方可以看见归一化时机。 |
| `compute_ver_EKLT()` | 无 | 只用于坐标/符号方向的诊断画图，不参与保存的重建结果。 |
| `forward_model_EKLT_ver()` | 无 | 同上，是另一套方向约定的实验函数。 |
| `forward_f2e()` | 无独立封装 | 正式路径没有调用；所需的两步已由 `linear_log_intensity()` 和 `predicted_iwe()` 覆盖。 |
| `forward_model_cmax()` | 无 | 常量位移版本只出现在注释代码中；正式流程使用逐时间轨迹。 |
| `forward_model_vec_cmax()` | `pipeline._warp_and_render()`；loss 留在 `estimate_motion()` / `refine_at_sensor_scale()` | 原函数混合 warp、渲染和 loss，新版拆开以便分别测试。 |
| `NeuroSR()` | `pipeline._render_gaussian_iwe()` | 这是正式 1x/2x 目标 IWE 的直接对应。依赖的全局变量改为显式参数。 |
| `NeuroSR_Flow()` | 无 | 只用于 renderer 对比图；正式联合优化调用的是普通 Gaussian renderer。 |
| `NeuroSR_Gau()` | 无 | 只用于可视化对照，不参与最终优化或正式输出。 |
| `save_png_npy()` | `output.save_results()` | 新函数一次保存完整、命名固定的结果集合，消除通过全局变量猜名称的行为。 |
| `reconstruct_pnp()` | 无 | 整个函数位于三引号历史代码块中，当前不会运行，且依赖外部 DRUNet 权重和 Windows 路径。 |

### 3.2 旧工具模块中主流程实际使用的能力

| 旧模块能力 | 新版对应 | 说明 |
|---|---|---|
| `utils.utility.load_events` | `data.load_aedat4` | AEDAT4 数据读取。 |
| `utils.utility.time_window` | `data.select_time_window` | 事件窗口筛选。 |
| `utils.utility.window_event_to_img` | `events.numpy_event_frame` | 配准事件帧。 |
| `utils.utility.events_to_image_torch[_sr/_flow]` | `events.bilinear_splat` + `event_image` | 双线性事件累加；新版不再用函数名暗示未实际发生的 flow SR。 |
| `utils.utility.Gau_events_to_image_torch_sr` | `events.gaussian_splat` + `gaussian_event_image` | 亚像素 Gaussian IWE；保留 round 中心约定和 sigma/kernel-size 规则。 |
| `utils.utility.warp_events_traj_torch` | `events.warp_events_to_reference` | 用 dense trajectory 采样事件时刻并 warp 到参考时间。 |
| `utils.utility.lin_log` | `optimization.linear_log_intensity` | 保留 float64、阈值线性段以及 `log(x + 1e-8)`。 |
| `function.optimizer.forward_grad` | `optimization.forward_gradient` | 返回 `(dy, dx)`，并将 wrap 边界清零。 |
| `function.optimizer.l2_loss` | `optimization.mean_square` | 名称明确表达实际计算的是均方。 |
| `function.optimizer.l1_loss` | `optimization.smooth_l1_to_zero` | 明确目标是零张量和 Smooth L1，而不是普通绝对值。 |
| `function.optimizer.tv_loss_flow` | `optimization.directional_total_variation` | 只保留当前实验使用的一阶、运动方向约束形式。 |
| `function.optimizer.blur_frame` | `optimization.blur_image` | 保留 replicate padding、kernel flip、归一化和可微卷积。 |
| `function.optimizer.down_sampling` | `optimization.block_average` | 2x 图像按不重叠块平均回传感器分辨率。 |
| `function.optimizer.AdamP` | `optimization.AdamP` | 保留第二级重建实际使用的投影 Adam。 |
| `generate_motion_blur_kernel_Dxy_pc` | `motion.piecewise_motion_blur_kernel` | 1x 阶段从 12 段解析轨迹构造可微 PSF。 |
| `generate_motion_blur_kernel_Dxy_sr` | `motion.dense_motion_blur_kernel` | 2x 阶段从固定 dense trajectory 构造 PSF。 |
| `Simple_sr.pos_registration` | `motion.register_event_frames` | 只提取相位互相关配准本身。 |
| `ContinuousPiecewiseLinear_Dxy_pw.regularization_loss` | `PiecewiseLinearTrajectory.regularization` | 保留相邻段平滑和增量幅值两项。 |
| 多个 `plot_*`、`norm`、`watch_tensor` | `output.normalize_u8` + `save_results` | 只保留解释正式结果所需的图；交互式实验画图不进入计算模块。 |

## 4. 关键变量改名

| 旧变量 | 新变量 | 含义与 shape |
|---|---|---|
| `ts, xs, ys, ps` | `ExposureSample.timestamps_us/x/y/polarity` | 当前窗口的 NumPy 事件数组，长度为 N。 |
| `xt, yt, tt, pt` | `EventTensors.x/y/timestamps_us/polarity` | 同一批事件的设备 tensor。 |
| `frames_sta` / `frames_sharp` | `ExposureSample.sharp_frame` | 第 0 张 APS 参考帧 `[H, W]`。 |
| `frames_mov` / `frames_blur` | `ExposureSample.blurred_frame` | 当前运动模糊 APS 帧 `[H, W]`。 |
| `frame_ind` | `ExposureSample.frame_index` | 本次实验为 11。 |
| `win_frame` | `ExposureSample.frame_exposure_us` | 单帧曝光时间，本次为 200000 us。 |
| `win` / `t_win` | `ExposureSample.event_window_us` | 双曝光事件窗口，本次为 418341 us。 |
| `num_pieces` | `ExperimentConfig.trajectory_segments` | 轨迹段数，本次为 12。 |
| `Dxy_0_pc` | `initial_increments` | 配准得到的初始段位移 `[12, 2]`。 |
| `Dxy_pc` | `motion.segment_increments_xy` / `increments` | contrast maximization 后的段位移 `[12, 2]`。 |
| `Dxy_pred` | `motion.dense_trajectory_xy` / `dense_trajectory` | 逐微秒累计位移 `[2, T]`。 |
| `Dxy_pc_pred` | `trajectory` | 1x 联合优化阶段临时微调的段位移。 |
| `I_pred` | `image` | 当前待优化强度图；先为 1x，后为 2x。 |
| `bg_pred` | `background` | 传感器分辨率背景补偿 `[H, W]`。 |
| `iwe_gt` / `iwe_pred_sr` | `target_iwe` | 按事件和轨迹渲染的目标 IWE。 |
| `iwe_pred` | `predicted` / `image_iwe` / `final_iwe` | 从重建图像梯度和总位移预测的 EKLT IWE。 |
| `kernel_pred` / `kernel_sr` | `kernel` | 当前运动模糊 PSF。 |
| `L_D` | `event_loss` | 归一化预测 IWE 与目标 IWE 的均方差。 |
| `L_F` | `frame_loss` | 模糊重投影与 APS 观测的平方根域均方差。 |
| `L_tot` | `loss` | 当前阶段的加权总损失。 |
| `loss_hist_v` | `MotionEstimate.loss_history` | 运动估计每 100 次的 loss。 |
| `loss_hist` | `ReconstructionState.loss_history` | 2x 重建每 100 次的 loss。 |

## 5. 损失函数逐项对照

第一轮运动估计在两版中都是：

```text
-Var(|GaussianBlur(unsigned IWE)|)
+ 0.2  * 相邻轨迹段平滑项
+ 1e-4 * 轨迹增量幅值项
```

1x 联合优化在两版中都是：

```text
2000 * event_loss
+ frame_loss
+ 0.04 * directional_TV(image)
+ 0.2  * smooth_L1(background)
+ contrast_loss
+ trajectory_regularization(0.02, 1e-4)
```

2x 重建在两版中都是：

```text
2000 * event_loss
+ frame_loss
+ 0.04 * directional_TV(image)
+ 0.2  * smooth_L1(background)
```

这里有一个容易忽略但被新版有意保留的行为：1x 阶段会临时微调轨迹来帮助该阶段
求解，但 2x 阶段重新使用第一轮 contrast maximization 的固定轨迹，而不是继续使用
1x 阶段末尾的临时轨迹。这与旧版一致。

## 6. 为什么旧版很多函数没有迁入

### 6.1 被拆分或合并，并非消失

`forward_model_vec_cmax()` 是最典型的例子。旧函数同时完成轨迹采样、事件 warp、
IWE 渲染和 contrast loss。新版将其拆为：

```text
warp_events_to_reference()  ->  warped_x, warped_y
event_image()               ->  IWE
estimate_motion()           ->  -variance loss
```

这样每一层的坐标、极性和 shape 都能单独测试。

### 6.2 属于通用工具库，但本实验没有调用

旧脚本使用 `from ... import *`，会让 `diffraction`、ptychography、phase unwrap、
Zernike、DnCNN/DRUNet、光流可视化、各种 Poisson solver 等大量能力出现在命名
空间中。它们并不因为“可被访问”就属于 `NeuroSRM_demo.py` 的执行链路。新版采用显式
import，只迁移真实依赖，避免读者误认为这些算法共同参与了重建。

### 6.3 是历史候选方案或诊断代码

以下内容不影响正式保存结果，因此没有进入新版主流程：

- 粗 `Dxy_0` 互相关图和多组 ROI/方向符号对照图；
- `NeuroSR_Flow()`、`NeuroSR_Gau()` renderer 对比；
- PnP/DRUNet 重建，且其旧代码依赖外部模型和 Windows 绝对路径；
- 25 个参考时刻的 time-lapse 数组生成；
- 多时刻、多 APS 帧联合重建实验；
- 被注释掉的 constant-flow、dense-flow、FFT blur、mask loss 等候选分支；
- 内存跟踪器和只用于交互查看的 Matplotlib 图。

旧版在保存正式结果后仍会显示三个参考时刻的 IWE。这是可视化副作用，不会改变已
保存数组；新版用一张固定的结果总览图取代它。

这些内容没有被宣称为“已经迁移”。如果以后要恢复其中某个实验，应当把它作为新的
独立功能实现和验证，而不应重新塞回正式重建主链路。

## 7. 新版新增但旧版没有清晰表达的结构

| 新结构 | 目的 |
|---|---|
| `Recording` | 给完整 AEDAT4 流一个明确的数据契约。 |
| `ExposureSample` | 把本次实验选中的事件、帧和时间范围绑定在一起。 |
| `EventTensors` | 明确哪些数组已经进入 PyTorch 设备。 |
| `MotionEstimate` | 明确区分段增量、dense trajectory 和运动 loss。 |
| `ReconstructionState` | 集中返回最终图像、背景、PSF、IWE 和 loss。 |
| `ExperimentConfig` | 代替散落的全局常量。 |
| 数值有限性检查 | 在 1x 优化首次产生 NaN/Inf 时立即报告具体参数。 |
| `compare_with_reference.py` | 用 shape、最大误差、RMSE、归一 RMSE 和相关系数验证迁移结果。 |
| `tests/test_core.py` | 独立验证事件 splat、轨迹终点、降采样、EKLT 和结果比较。 |

## 8. 推荐阅读顺序

1. `NeuroSRM_demo.py`：了解命令行入口。
2. `neurosr/config.py`：查看实验固定参数。
3. `neurosr/pipeline.py::run_experiment()`：掌握五阶段数据流。
4. `neurosr/data.py`：理解帧 11 和双曝光事件窗口如何选出。
5. `neurosr/pipeline.py::estimate_motion()`：理解 12 段轨迹初始化与优化。
6. `neurosr/pipeline.py::refine_at_sensor_scale()`：理解 1x 联合优化。
7. `neurosr/pipeline.py::reconstruct_at_two_x()`：理解最终 2x 重建。
8. `neurosr/events.py`、`motion.py`、`optimization.py`：按需查看数学算子。
9. `neurosr/output.py` 和 `compare_with_reference.py`：理解保存与一致性验证。

如果阅读旧版时遇到某个名称，先查本文第 3、4 节。没有对应项时，再判断它是否位于
三引号块、注释分支或纯可视化区域；这三类代码都不属于当前正式输出的计算依赖。
