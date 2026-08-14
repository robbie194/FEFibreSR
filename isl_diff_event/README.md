# isl_diff_event

## Documentation

- [事件时间窗与 APS 曝光时间的关系](docs/EVENT_WINDOW_VS_APS_EXPOSURE.md)
- [旧目录入口脚本与论文光学模型说明](docs/LEGACY_ENTRYPOINTS_AND_PAPER_MODELS.md)

## Rotation Flow Reconstruction Demo

`rotation_demo.py` reconstructs a single log-intensity image from a fixed dense rotation flow and the corresponding IWE constraint. The debug summary below shows the recovered log latent image, target IWE, predicted IWE, and residual.

![Rotation flow reconstruction demo](docs/rotation_demo_summary.png)
