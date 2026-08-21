# 盲运动、前向模型与 Loss 说明

## 1. 当前是不是“全盲”运动估计

严格说，当前是**受约束的低维盲估计**，不是任意轨迹的完全盲估计。

- 盲的部分：终点 $\mathbf d$、法向曲率 $\kappa$ 和非匀速量 $a$ 均未知，只由 `core_mask + APS + events` 估计，不读取仿真真值。
- 受约束的部分：预先假设运动是单次、连续、平滑扫描：

$$
\mathbf u(t)=t\mathbf d
+\kappa\sin^2(\pi t)\mathbf n
+a\sin(2\pi t)\widehat{\mathbf d}.
$$

因此它不能表达突然抖动、往返、停顿或复杂自由曲线。更准确的名称是“**观测驱动的受约束盲运动估计**”。此前直接优化多个自由控制点容易把光纤芯格伪影当成运动，低维限制是为了提高可辨识性和稳定性，而不是声称运动完全无先验。

运动估计分三步：

1. density-normalized CMax 搜索二维终点候选；
2. 用 APS 生成 predicted temporal IWE，与观测 IWE 比较，排除芯格别名；
3. 固定终点后，以同一 APS/event 一致性搜索 $\kappa$ 和 $a$。

第一步最大化密度归一化 IWE 方差减位移惩罚；后两步最小化第 4 节定义的 $L_{\mathrm{event}}$，其中候选图像暂用芯 APS 插值图。CMax 与 APS/event 终点相差不超过 `1 px` 时采用 CMax 的精定位，否则采用 APS/event 结果以避开芯格别名。

对应代码：`src/fibre_iwe/motion.py`。

## 2. 当前仿真参数

| 参数 | 一维基线 | 二维弯曲 |
| --- | ---: | ---: |
| sensor | `192 x 192 px` | `192 x 192 px` |
| core pitch | `5.5 px` | `5.5 px` |
| proximal spot radius | `1.85 px` | `1.85 px` |
| 芯数量 | `1098` | `1098` |
| 曝光时长 | `40 ms` | `40 ms` |
| 仿真时间间隔 | `0.1 ms` | `0.1 ms` |
| 终点位移 | `[5.5, 0] px` | `[5.5, 4.75] px` |
| 法向曲率 | `0` | `1.6 px` |
| effective blur $\sigma$ | `0.9 px` | `0.9 px` |
| APS noise $\sigma$ | `0.003` | `0.003` |
| event contrast threshold | `0.075` log-intensity | 同左 |
| 逐芯 threshold log-$\sigma$ | `0.07` | `0.07` |
| background events | 有效事件数的 `1%` | 同左 |
| 生成的原始 events | `5152` | `7058` |
| 随机种子 | `19` | `29` |

仿真真实轨迹为：

$$
\mathbf u_{\mathrm{sim}}(t)
=\left[t+\frac{0.10}{2\pi}\sin(2\pi t)\right]\mathbf d
+\kappa\sin^2(\pi t)\mathbf n.
$$

运动搜索范围：终点在 $[-8,8]^2$ 内先按 `1 px` 搜索，再以 `0.25 px` 细化；二维曲率搜索 `[-2.5,2.5] px`，$a$ 搜索 `[-0.4,0.4] px`，最后以 `0.05 px` 细化。

完整参数见 `configs/baseline.yaml` 和 `configs/two_dimensional.yaml`。

## 3. APS 与 event 前向模型

### 3.1 仿真如何生成观测

物体先经过共享高斯模糊，得到待恢复的有效图像 $O$。第 $i$ 根芯在时刻 $t$ 的远端强度为：

$$
c_i(t)=O\!\left(\mathbf x_i-\mathbf u(t)\right).
$$

APS 是曝光期芯强度平均，再乘近端不规则亮斑响应并加入噪声。events 则由 $\log c_i(t)$ 跨越逐芯阈值产生；event 的近端像素位置只用于通过 mask 判断属于哪根芯。

### 3.2 重建使用的 APS 前向

原始 APS 先按 mask 做背景扣除，并取每根芯内像素中位数，得到观测 $A_i$。候选图像沿估计轨迹曝光平均，再在芯中心采样：

$$
\widehat A_i(O,\mathbf u)
=\frac{1}{M}\sum_{m=1}^{M}
O\!\left(\mathbf x_i-[\mathbf u(t_m)-\mathbf u(0.5)]\right).
$$

这里没有使用仿真 PSF、pixel gain 或真值运动。

### 3.3 重建使用的 event 前向

芯内 event 先全部映射到所属芯中心，再按估计运动 warp 到曝光中点。每个时间段 $b$ 分别生成观测 IWE：

$$
E_b^{\mathrm{obs}}(\mathbf x)
=\sum_{k\in b}p_k K\!\left(\mathbf x-\mathbf x_k^{\mathrm{warp}}\right).
$$

同一轨迹产生二维累计 flow $\mathbf F_b$，候选图像的 event prediction 为：

$$
\widehat E_b(O,\mathbf u)
=-\nabla\log O(\mathbf x)\cdot\mathbf F_b(\mathbf x).
$$

所以当前 event forward 确实使用了“log-intensity 梯度乘运动量”的模型；只是运动量被写成各时间段的二维 flow，而不是单一总位移。对应代码：`src/fibre_iwe/event_forward.py`。

APS 前向和三种重建 loss 对应 `src/fibre_iwe/reconstruction.py`；仿真观测生成对应 `src/fibre_iwe/simulation.py`。

## 4. Loss 设计

APS 数据项使用 Huber loss：

$$
L_{\mathrm{APS}}=\operatorname{Huber}_{\beta=0.03}(\widehat A,A).
$$

event 阈值在真实实验中未知，因此不比较绝对幅值，而使用各时间段能量加权的 cosine loss：

$$
L_{\mathrm{event}}
=\frac{\sum_b \lVert E_b^{\mathrm{obs}}\rVert_2
\left[1-\cos(\widehat E_b,E_b^{\mathrm{obs}})\right]}
{\sum_b \lVert E_b^{\mathrm{obs}}\rVert_2}.
$$

平滑项 $R(O)$ 是图像水平、垂直二阶差分的稳健均值。三种重建分别为：

$$
L_{\mathrm{APS-only}}=L_{\mathrm{APS}}+\lambda_R R(O),
$$

$$
L_{\mathrm{joint}}=L_{\mathrm{APS}}+\lambda_E L_{\mathrm{event}}+\lambda_R R(O),
$$

$$
L_{\mathrm{event-only}}=L_{\mathrm{event}}+0.003R(O)
+0.20\left[(\mu_O-0.50)^2+(\sigma_O-0.22)^2\right].
$$

| 场景 | $\lambda_E$ | $\lambda_R$ |
| --- | ---: | ---: |
| 一维 | `0.005` | `0.003` |
| 二维 | `0.008` | `0.004` |

event-only 的图像 loss 不使用 APS，但它共享前面由 APS/events 估计的运动。均值/标准差项只负责固定 events 无法确定的亮度与对比度尺度。
