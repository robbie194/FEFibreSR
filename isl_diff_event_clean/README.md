# Clean NeuroSR

This directory is an independent, structured implementation of the experiment
in `../isl_diff_event/NeuroSRM_demo.py`. It does not import or modify the legacy
project.

The executable pipeline has five visible stages:

1. Load one DAVIS recording and align events with APS exposure windows.
2. Register short event frames to initialize a 12-segment trajectory.
3. Refine that trajectory by maximizing warped-event contrast.
4. Jointly refine motion and a latent image on the sensor grid.
5. Reconstruct the final 2x image with event and APS fidelity terms.

Run the full reference configuration:

```bash
cd isl_diff_event_clean
MPLBACKEND=Agg /home/robbie/miniconda3/envs/NeuroFibreSR/bin/python NeuroSRM_demo.py
```

Compare its raw arrays with the frozen legacy result:

```bash
/home/robbie/miniconda3/envs/NeuroFibreSR/bin/python compare_with_reference.py
```

The comparison prints both strict `allclose` results and a numerical-equivalence
result. CUDA event splatting uses atomic accumulation, so repeated executions of
the legacy script itself are not bit-for-bit reproducible. Numerical equivalence
requires matching shapes, normalized RMSE at most 0.5%, and correlation at least
0.999. Raw maximum error and RMSE remain visible in the report.

Use `--iterations 1` for a fast wiring check. It is not a meaningful
reconstruction and cannot match the 2100-iteration baseline.

For a stage-by-stage and function-by-function mapping from the legacy script,
including the reason each unused experiment was not migrated, see
[`MIGRATION_GUIDE.md`](MIGRATION_GUIDE.md).

## Code map

- `neurosr/config.py`: explicit experiment parameters.
- `neurosr/data.py`: AEDAT4 and exposure-window data contract.
- `neurosr/events.py`: event warping and differentiable splatting.
- `neurosr/motion.py`: trajectory model, registration, and motion PSFs.
- `neurosr/optimization.py`: image model, losses, blur, and AdamP.
- `neurosr/pipeline.py`: readable end-to-end stage orchestration.
- `neurosr/output.py`: stable artifacts and numerical comparison.
