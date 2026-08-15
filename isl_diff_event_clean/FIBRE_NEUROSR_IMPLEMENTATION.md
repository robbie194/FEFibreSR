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
