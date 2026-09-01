# Gantry downsampling study

This folder measures whether the existing 20 kHz hidden-MSD records can be used
at 4, 2, or 1 kHz in the training pipeline.  It deliberately separates two
questions:

1. **Open-loop rate floor.**  The exact eight-state oracle and the six-state FP
   baseline replay the recorded force after the pipeline's block-mean input
   reduction.  The oracle error is the irreducible rate-conversion floor; the
   FP error is the MSD discrepancy available to learn.
2. **Feedback amplification.**  Rates that pass the first gate are wrapped in
   the residual controller.  Two controller conventions are reported, rather
   than silently conflated:
   - `corate`: the current pipeline, with the continuous controller Tustin-
     discretised at the model rate;
   - `controller20k_zoh`: the original 20 kHz controller sees a zero-order-held
     low-rate model output, and its corrected forces are block-averaged before
     the low-rate model step.

The second convention is a diagnostic of the simplest causal multirate
interface.  It is not claimed to be the unique or final multirate design.

## Reproduce

```powershell
conda run --no-capture-output -n GraduationProject `
  python -u scripts/gantry/downsampling/run_rate_sweep.py --stage all --workers 4
```

The existing dataset is read from
`data/gantry/matlab/trajectory/augmentation_ma50_a5`.  Nothing in the dataset
or training code is modified.  Results are written below `results/`:

- `open_loop.json` and `open_loop_summary.csv`
- `controller.json`, `controller_summary.csv`, and `controller_force_summary.csv`
- `summary.png`
- `FINDINGS.md`

Presentation-ready PNG and vector PDF figures can be regenerated with
`make_supervisor_figures.py`; they are stored in
`results/supervisor_figures/`.

Every 20 kHz truth reconstruction is checked against the stored output before
its lower-rate states are used as exact window initial conditions.
