# 第一阶段：多芯光纤 Frame–Event 前向仿真

本目录只完成前向数据生成，不包含 NeuroSR 重建。输入是仓库根目录的 `分辨率板.png`，输出是同一曝光窗口内的一帧 APS 图像与一条时间对齐的事件流。

实现完成后的常见疑问、参数来源和当前能力边界见 [IMPLEMENTATION_QA.md](IMPLEMENTATION_QA.md)。

关于 `sigma_um=0`、0.1 ms 时间步长、APS 积分结果和固定芯斑事件极性的专项核验，见 [FOUR_QUESTIONS_SIGMA_TIME_APS_EVENTS.md](FOUR_QUESTIONS_SIGMA_TIME_APS_EVENTS.md)。

关于固定近端芯斑事件是否还能用于重建、如何构造 fibre-aware 事件损失，以及怎样基于 clean 工程实现新重建入口，见 [FIBRE_EVENT_RECONSTRUCTION_ANALYSIS.md](FIBRE_EVENT_RECONSTRUCTION_ANALYSIS.md)。

## 基线链路

```text
USAF 裁剪（物方真值）
  → 匀速平移
  → GRIN 成像和高斯模糊
  → 六角纤芯圆孔面积平均
  → 圆形纤芯出射斑
  → 中继成像及 DAVIS 像素面积积分
  ├→ 25 ms 梯形积分 → APS frame
  └→ 0.1 ms 瞬时序列 → v2e EventEmulator → events
```

所有基线参数集中在 [configs/phase1_usaf.yaml](configs/phase1_usaf.yaml)。局部光纤视场为 160 × 160 µm，仿真网格为 0.5 µm/px；芯间距 4.5 µm、芯径 2.9 µm。DAVIS346 为 346 × 260 像素、像元 18.5 µm。中继倍率按传感器短边完整容纳方形光纤 ROI 自动得到 30.0625×，此时芯间距约为 7.31 sensor px。

25 ms 内含 250 个时间区间和 251 个端点样本。APS 使用全部端点做梯形时间平均；事件生成直接调用已有 `../v2e/v2ecore/emulator.py`，不使用补帧，也不改 v2e 源码。

## 逐步运行

默认使用从 v2e 完整克隆并验证过的 `NeuroFibreSR` Conda 环境：

```bash
conda activate NeuroFibreSR
cd /home/robbie/tyf_code/EventCode/myFEFibreSR/fibre_frame_event_sim
python scripts/00_prepare_source.py
python scripts/01_generate_motion.py
python scripts/02_generate_grin_sequence.py
python scripts/03_generate_fibre_sequence.py
python scripts/04_generate_sensor_sequence.py
python scripts/05_generate_aps_frame.py
python scripts/06_generate_events.py
python scripts/07_validate_outputs.py
```

一键执行：

```bash
python scripts/run_all.py
```

单元测试：

```bash
python -m unittest discover -s tests -v
```

## 各步输入与输出

| 步骤 | 输入 | 主要输出 |
|---|---|---|
| 00 source | `分辨率板.png` + YAML 裁剪参数 | `object_intensity.npy/png`、原始裁剪预览 |
| 01 motion | 速度、曝光和步长 | `motion.npz`、轨迹 CSV/PNG |
| 02 GRIN | object + motion | `grin_sequence.h5`，数据集 `frames[T,320,320]` |
| 03 fibre | GRIN 序列 + 芯参数 | `fibre_sequence.h5`，含 frames、纤芯坐标和每芯信号 |
| 04 sensor | 光纤输出 + 中继/DAVIS 参数 | `sensor_sequence.h5`，`frames[T,260,346]` |
| 05 APS | sensor sequence + 曝光区间 | `aps_frame.npy/png`、瞬时帧对比与差异增强图 |
| 06 events | 同一 sensor sequence + v2e 参数 | `events.h5`、完整累积图、5 ms/1 ms 分段图、事件率图、统计 JSON |
| 07 validation | 上述所有产物 | `validation_report.json` |

HDF5 事件文件同时保存：

- `events_t_s_x_y_p`：float32，列为 `[t_s, x, y, p]`，`p ∈ {-1,+1}`；
- `events_t_us_x_y_p01`：uint32，列为 `[t_us, x, y, p]`，`p ∈ {0,1}`。

默认输出目录为 `outputs/phase1_usaf/`。每一步都能独立重跑，但需保证它依赖的前一步产物已存在。
