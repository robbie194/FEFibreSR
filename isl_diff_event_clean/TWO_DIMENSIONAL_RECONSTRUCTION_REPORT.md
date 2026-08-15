# 二维远端扫描与 GRIN sigma 对照重建报告

## 1. 结论

此前的理解是正确的：旧仿真只让远端物体沿 x 方向移动 4.5 um，因此事件只为 x 方向增加了新的扫描相位，y 方向没有同等的时间采样增益。

本次新增了真正非共线的二维轨迹：

```text
(0.0, 0.0) um
  -> (4.5, 0.0) um
  -> (4.5, 4.5) um
```

它先水平扫描一个芯间距，再垂直扫描一个芯间距。两段各 25 ms，总时长 50 ms、时间步长 0.1 ms，共 501 个端点样本。

`sigma_um=0` 和 `0.8` 两套数据均已完成：

- source、motion、GRIN、fibre、sensor、APS、events 全链路仿真；
- 仿真输出 validation；
- APS-only、events-only、APS + core events 三组重建；
- 真值前向一致性检查；
- PSNR、SSIM、x/y 梯度指标和重投影残差检查。

两套数据的联合重建都通过了以下质量门：

- PSNR 高于 APS-only；
- SSIM 高于 APS-only；
- x 梯度相关性高于 APS-only；
- y 梯度相关性高于 APS-only。

## 2. 为什么不能只使用对角线直线

如果物体从 `(0,0)` 沿一条直线移动到 `(4.5,4.5)`，虽然 x、y 数值都变化，但轨迹仍只有一个自由方向。两个方向的位移严格耦合，不能提供两组独立采样相位。

当前 L 形轨迹包含水平段和垂直段，两段方向非共线：

$$
\mathbf v_1=(180,0)\ \mathrm{um/s},
\qquad
\mathbf v_2=(0,180)\ \mathrm{um/s}.
$$

因此同一根固定纤芯先沿物体 x 方向扫描，再沿 y 方向扫描。它仍不是覆盖整个二维方形区域的 raster scan，但已经能独立约束两个方向。

## 3. 两套仿真输入

配置文件：

```text
fibre_frame_event_sim/configs/phase2_xy_usaf_sigma0.yaml
fibre_frame_event_sim/configs/phase2_xy_usaf_sigma08.yaml
```

本地完整输入目录：

```text
fibre_frame_event_sim/outputs/phase2_xy_usaf_sigma0/
fibre_frame_event_sim/outputs/phase2_xy_usaf_sigma08/
```

每套目录均包含：

```text
00_source/object_intensity.npy
01_motion/motion.npz
02_grin/grin_sequence.h5
03_fibre/fibre_sequence.h5
04_sensor/sensor_sequence.h5
05_aps/aps_frame.npy
06_events/events.h5
07_validation/validation_report.json
```

两份 validation report 均为 `all_passed: true`。

事件统计：

| GRIN sigma | events | ON | OFF |
|---:|---:|---:|---:|
| 0.0 um | 137,167 | 63,342 | 73,825 |
| 0.8 um | 102,961 | 46,680 | 56,281 |

`sigma=0.8` 的事件更少是合理现象。Gaussian PSF 降低了高频边缘和相邻时间点的亮度变化，因此更少跨过 `0.2` 的事件阈值。

## 4. sigma=0.8 怎样进入重建模型

`sigma=0.8` 不是只在仿真中模糊，然后仍用 `sigma=0` 的错误模型重建。

可微前向读取 YAML 中的：

```yaml
grin:
  sigma_um: 0.8
```

并按照与 simulator 相同的顺序计算：

```text
远端物体和已知位移
  -> 双线性空间采样
  -> sigma=0.8 um GRIN Gaussian PSF
  -> 2.9 um圆形芯孔径面积平均
  -> 1415根纤芯信号
  -> APS时间积分和事件lin-log响应
```

也就是说，优化器每次都通过包含 Gaussian PSF 的模型生成预测观测。恢复清晰物体时同时完成受物理约束的 GRIN 反卷积，而不是对最终图片做无依据锐化。

Torch 前向与原 simulator 的一致性为：

| sigma | core RMSE | core correlation | event correlation |
|---:|---:|---:|---:|
| 0.0 um | 0.000552 | 0.999998 | 0.993563 |
| 0.8 um | 0.000415 | 0.999999 | 0.990657 |

这证明两种 sigma 都由正确的前向模型解释。

## 5. 正式重建结果

结果目录：

```text
isl_diff_event_clean/results/fibre_neurosr/phase2_xy_sigma0/
isl_diff_event_clean/results/fibre_neurosr/phase2_xy_sigma08/
isl_diff_event_clean/results/fibre_neurosr/phase2_xy_sigma_sweep/
```

### 5.1 sigma=0

| 方法 | PSNR | SSIM | x 梯度相关 | y 梯度相关 |
|---|---:|---:|---:|---:|
| APS interpolation | 14.76 | 0.600 | 0.177 | 0.146 |
| APS-only | 16.20 | 0.745 | 0.397 | 0.252 |
| APS + events | **18.41** | **0.800** | **0.460** | **0.433** |

联合重建相对 APS-only：

```text
PSNR:                  +2.22 dB
SSIM:                   +0.055
x gradient correlation: +0.063
y gradient correlation: +0.181
```

### 5.2 sigma=0.8

| 方法 | PSNR | SSIM | x 梯度相关 | y 梯度相关 |
|---|---:|---:|---:|---:|
| APS interpolation | 14.69 | 0.597 | 0.174 | 0.147 |
| APS-only | 16.39 | 0.750 | 0.399 | 0.273 |
| APS + events | **18.24** | **0.788** | **0.453** | **0.406** |

联合重建相对 APS-only：

```text
PSNR:                  +1.85 dB
SSIM:                   +0.038
x gradient correlation: +0.054
y gradient correlation: +0.134
```

`sigma=0.8` 的结果略低于 `sigma=0`，这是合理的，因为 GRIN blur 确实损失了部分高频信息。已知 PSF 可以反演被衰减的频率，但不能可靠恢复已经被完全压制的信息。

## 6. 与旧纯水平轨迹的直接比较

为避免不同可观测范围造成不公平，代码使用三组结果共同覆盖的裁剪：

```text
y: 41..359
x: 35..356
```

比较 joint 结果：

| 轨迹和 sigma | PSNR | SSIM | x 梯度相关 | y 梯度相关 |
|---|---:|---:|---:|---:|
| 旧纯水平，sigma=0 | 17.16 | 0.767 | 0.468 | 0.284 |
| 新二维，sigma=0 | **18.38** | **0.799** | 0.459 | **0.434** |
| 新二维，sigma=0.8 | 18.19 | 0.786 | 0.451 | 0.408 |

新二维 `sigma=0` 相对旧纯水平：

```text
PSNR:                  +1.22 dB
SSIM:                   +0.032
x gradient correlation: -0.009
y gradient correlation: +0.150
```

x 指标基本保持，y 指标显著提高。这正是新轨迹的目标：保留原有水平扫描能力，同时补充原先缺失的纵向采样相位。这里不把 `-0.009` 的微小变化描述成新的 x 提升。

## 7. 怎样查看结果

重点图片：

```text
phase2_xy_sigma_sweep/sigma_comparison.png
phase2_xy_sigma_sweep/trajectory_comparison.png
phase2_xy_sigma0/reconstruction_comparison.png
phase2_xy_sigma08/reconstruction_comparison.png
```

重点数值报告：

```text
phase2_xy_sigma_sweep/run_summary.json
phase2_xy_sigma0/run_summary.json
phase2_xy_sigma08/run_summary.json
```

总报告中的：

```text
all_directional_quality_checks_passed: true
```

表示两种 sigma 的仿真 validation、PSNR/SSIM 和 x/y 方向质量门全部通过。

## 8. 一键复现

从仿真输入开始完整重跑：

```bash
cd /home/robbie/tyf_code/EventCode/myFEFibreSR/isl_diff_event_clean
MPLBACKEND=Agg /home/robbie/miniconda3/envs/NeuroFibreSR/bin/python \
  FibreNeuroSR_xy_sweep.py
```

已有输入时跳过仿真：

```bash
MPLBACKEND=Agg /home/robbie/miniconda3/envs/NeuroFibreSR/bin/python \
  FibreNeuroSR_xy_sweep.py --skip-simulation
```

只重新生成汇总和比较图：

```bash
MPLBACKEND=Agg /home/robbie/miniconda3/envs/NeuroFibreSR/bin/python \
  FibreNeuroSR_xy_sweep.py --skip-simulation --skip-reconstruction
```

## 9. 当前边界

本结果证明了 exact-model 条件下，固定近端纤芯事件可以利用非共线远端运动同时增强两个方向。

仍需保留以下边界：

- L 形轨迹只覆盖两条线，不等同于完整二维 raster 或圆形扫描；
- 2.9 um 芯孔径和 GRIN PSF 完全截止的频率无法从无信息中恢复；
- 当前使用已知运动和已知 sigma；真实实验需要分别标定；
- 真实纤芯模式变化、串扰和阈值离散性仍需加入下一阶段模型。
