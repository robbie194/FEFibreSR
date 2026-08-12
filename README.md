# FEFibreSR

Frame-event simulation and neuromorphic reconstruction research for multicore
fibre imaging.

## Repository layout

- `fibre_frame_event_sim/`: reproducible USAF target to APS/event forward
  simulation, including configuration and unit tests.
- `isl_diff_event/`: NeuroSR/IWE reconstruction research code and runnable
  `NeuroSRM_demo.py` experiment.
- `v2e/`: vendored SensorsINI v2e event-camera simulator used by the forward
  pipeline.
- Root Markdown files: design decisions, implementation plans, and code
  analysis notes.

## Python environment

Use the repository environment by default:

```bash
/home/robbie/miniconda3/envs/NeuroFibreSR/bin/python
```

## Forward simulation

```bash
cd fibre_frame_event_sim
/home/robbie/miniconda3/envs/NeuroFibreSR/bin/python scripts/run_all.py
/home/robbie/miniconda3/envs/NeuroFibreSR/bin/python -m unittest discover -s tests -v
```

Generated HDF5/NumPy outputs are intentionally excluded from Git and can be
recreated from `configs/phase1_usaf.yaml`.

## NeuroSR demo

The demo currently expects the DAVIS recording path configured in
`isl_diff_event/NeuroSRM_demo.py`:

```bash
cd isl_diff_event
MPLBACKEND=Agg /home/robbie/miniconda3/envs/NeuroFibreSR/bin/python NeuroSRM_demo.py
```

Display-ready results and a run summary are written under
`isl_diff_event/results/fig/`. Large raw arrays and logs are excluded from Git.

