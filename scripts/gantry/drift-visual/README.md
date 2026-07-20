# drift-visual — clean drift-demonstration deck

Purpose: a supervisor-facing figure deck that demonstrates the augmentation drift, its cause,
and why training cannot see it. This folder replaces the scattered `drift-demo/`; the point of
the rewrite is that **every figure states exactly where its data comes from** and nothing is
filtered to a subset of channels.

Answers Jan's mail: it is **not** an energy drift (velocity stays bounded); the learned force is
**not zero-mean** (never enforced); and the **stiffness-free X/Y axes** integrate that constant
into unbounded drift.

---

## The story (each figure proves one link)

1. **Symptom** — the augmented model drifts in free-run (X/Y ramp, yaw bounded).
2. **Fault** — the ANN leaves a small **non-zero-mean** correction on the velocity states.
3. **Cause** — removing that constant collapses the drift (the constant *is* the drift).
4. **Amplifier** — X/Y have no stiffness; the same constant only parks a bounded offset on sprung yaw.
5. **Why unconstrained** — the excitation carries no DC and the 0.1 s window cannot see the free-run cost.
6. **Consequence** — every trained checkpoint is worse in free-run than no ANN at all.

---

## Data provenance (READ FIRST — this is the whole point of the rewrite)

**One trained checkpoint underlies every "trained ANN" figure:**

`simulations/gantry_subnet/diagnostics/checkpoints/gantry_drift_last.pth`

built by `scripts/gantry/diagnostics-drift/make_drift_checkpoint.py`. It is a **deliberately fast,
rough** checkpoint (Optuna job 69399, Trial 3 — the one config whose val sim-RMS moved past epoch 0):

| hyperparameter | value |
|---|---|
| routing (ANN output rows) | all 8 states (0-7), incl. X/Y |
| learning rate | 1.49e-8 |
| training window nf | 1400 samples = 0.35 s |
| epochs | 5 (script default 20; checkpoint history stops at 5) |
| stride | 100 (fast Optuna regime) |
| validation | cropped to 8000 samples |
| seed | 45 (cfg.seed + 3, Trial-3 init) |
| ANN | 2 latent states, 16 nodes x 2 layers, na=nb=17 |
| rate / noise | 4 kHz, noiseless |
| saved as | `_last` (the drifted model, NOT `_best` which reverts to epoch-0 zero-init) |

Because it is one rough checkpoint, the "not a one-run accident" claim rests on the
**9-checkpoint universality figure**, not on this checkpoint alone. State this to Jan.

**Trajectories (all held out from training):**
- **V1** = `V1_standstill_Yp10.mat` — standstill at Y = +10 mm. The free-run drift eval.
- **V3** = `V3_ysweep_Yp10.mat` — moving Y-sweep. The "does the model track the I/O" eval.

**Reference conditions (used across figures):**
- **truth** = the measured ground-truth trajectory (baseline + hidden 150 Hz absorber).
- **baseline / no-ANN** = the augmented model with the ANN output forced to zero.
- **baseline, true start** = baseline physics simulated from the true initial state (no encoder).
- **encoder start** = baseline from the encoder-estimated initial state (adds encoder-IC error).
- **trained ANN (full)** = `gantry_drift_last`.
- **debiased ANN** = trained ANN with its measured per-row time-mean subtracted each step.

---

## Frames and conventions (MANDATORY on every figure)

- **Full-show is the default.** The object is the **full state**: 6 physical states
  (X, Theta, Y, dX, dTheta, dY) in **both frames** (logical X/Theta/Y + stage X1/X2/Y),
  **positions and velocities**. Never collapse to "X/Theta/Y". If a figure legitimately shows
  positions only (e.g. the drift counterfactual — velocities do not drift), say so on the figure.
- **Stage = P^T @ logical** for positions AND velocities (the same clean map). Do **not**
  transform forces (P^-1 + mass matrix): it mixes the sprung yaw into the X-encoders and
  introduces a false near-zero-mean on stage-X1.
- **Show the state correction, not force.** The ANN outputs a per-step state increment; show it
  in the state's own unit (m, rad, m/s, rad/s), un-normalized by `std_x` only. No mass matrix.
- **Scientific notation** on every y-axis (`ticklabel_format(style='sci', scilimits=(0,0))`).
- **Titles are questions**; captions state expect / observe / falsified-if.
- **<= 3 curves per panel** for comparisons; white-boxed notes or line-end labels, never bare
  text on a curve.
- **Provenance footer** on every figure: checkpoint tag + trajectory + frame.
- **Collision pass before delivery**: render at final size, crop-magnify the footer / annotations /
  titles, confirm no overstrike.

---

## Folder layout

```
scripts/gantry/drift-visual/
  README.md          # this file
  config.py          # single source: CKPT path, V1/V3 files, physical constants, style
  generate_data.py   # runs the checkpoint free-runs + captures -> data/*.npz (+ manifest.json)
  make_figures.py    # reads data/*.npz -> figures/*.png (one function per figure)
  data/              # generated npz + manifest (what each file is, which run produced it)
  figures/           # output pngs
  copied/            # g4_split.png, g5_window_ratio.png (dropped in, NOT regenerated)
```

Run:
```
set PYTHONIOENCODING=utf-8
conda run -n GraduationProject python scripts/gantry/drift-visual/generate_data.py
conda run -n GraduationProject python scripts/gantry/drift-visual/make_figures.py
```

### Fast-preview switch (MANDATORY — build this before the real run)

`generate_data.py` takes a data **source** so figure LAYOUT can be checked in seconds before any
slow simulation. Same `data/*.npz` filenames + shapes in every mode, so `make_figures.py` is
source-agnostic.

- `SOURCE=fake` — synthetic arrays with the right SHAPES and plausible SCALES (drift ramps, noisy
  traces with the intended |mean|/rms, bounded yaw). No checkpoint, no simulation; produces the
  full deck in seconds. Use this to check layout / collisions / labels FIRST.
- `SOURCE=reuse` — pull the REAL arrays we already have (`scripts/gantry/drift-demo/figures/*.npz`,
  `simulations/gantry_subnet/diagnostics/d6_ann_mean_force_gantry_drift_last.npz`). Real data, no
  re-simulation. Fast.
- `SOURCE=real` — full regeneration from `gantry_drift_last.pth` (V1 + V3 free-runs, captures). Slow.

```
SOURCE=fake  conda run -n GraduationProject python .../generate_data.py   # layout preview, seconds
SOURCE=reuse conda run -n GraduationProject python .../generate_data.py   # real, from existing npz
SOURCE=real  conda run -n GraduationProject python .../generate_data.py   # full regen (slow)
```

Workflow: build scripts -> `SOURCE=fake` -> show the deck -> user approves layout -> `SOURCE=reuse`
or `SOURCE=real` for the real figures. Every fake array is clearly marked in `manifest.json` so a
fake-data figure is never mistaken for a real result.

---

## The figures (exact list, full member set stated)

| # | file | question (title) | channels shown | data source |
|---|------|------------------|----------------|-------------|
| 1 | `f01_problem.png` | What goes wrong when we augment? | positions X/Theta/Y **and** stage X1/X2/Y; no-ANN vs trained-ANN; 12 s; + provenance table | V1 free-run of CKPT (`e_full`, `e_off`) |
| 2 | `f02_tracks.png` | Does the model track the I/O? | X/Theta/Y **and** X1/X2/Y; measured vs baseline-true-start, trajectory scale | new baseline sim on V3 |
| 3 | `f03_decomp.png` | Where does the free-run error come from? | X/Theta/Y (+ stage); small multiples: truth / baseline-true-start / encoder-start / augmented | untrained baseline + CKPT (V1) |
| 4 | `f04_correction.png` | Is the ANN's correction zero-mean? | **6 states x 2 frames**: X,Theta,Y,dX,dTheta,dY (logical) + X1,X2,Y,dX1,dX2,dY (stage); mean + \|mean\|/rms | CKPT ANN output on V1 (`ann_out`) — **this is the current good `g_correction_channels`** |
| 5 | `f05_counterfactual.png` | Is the constant the drift? Remove it. | positions X/Theta/Y **and** X1/X2/Y; 3 curves each: trained ANN / constant removed / ANN at zero | V1 free-run (`e_full`, `e_deb`, `e_off`) |
| 6 | `f06_cumulative.png` | How does a tiny mean become 29 mm? | 6 states x 2 frames: cumulative integral of the correction over 12 s | CKPT ANN output on V1 |
| 7 | `f07_universality.png` | Is the constant a one-run accident? | all 8 ANN rows, 9 independent checkpoints, per-row sign | 9-checkpoint bank (`bar_means`) |
| 8 | `f08_notenergy.png` | Is it energy? | drifting-axis velocity (logical + stage) stays bounded; + terminal-velocity prediction | V1 free-run of CKPT |
| 9 | `f09_horizon.png` | When can training see the drift? | drift contribution vs evaluation horizon; training window / T* / BPTT wall | cumulative drift of CKPT (V1) |
| 10 | `f10_excitation.png` | Why is the constant unconstrained? | training-input spectrum: zero-mean narrowband 130-180 Hz, no DC | training data |
| 11 | `copied/g4_split.png` | Does improving the objective improve deployment? | train/val window-fit + free-run | COPIED (nf-sweep logs, not regenerated) |
| 12 | `copied/g5_window_ratio.png` | Does a longer window beat no ANN? | end-of-training free-run / no-ANN, bars | COPIED (logs, not regenerated) |

Optional backups (build only if wanted): pole map (X/Y on the unit circle = free integrator).

---

## Status / TODO

- [ ] `config.py` — single source of paths + constants + style helpers
- [ ] `generate_data.py` — produce `data/*.npz` + `manifest.json`
- [ ] `make_figures.py` — figures 1-10
- [ ] port the current good `g_correction_channels.py` -> figure 4 unchanged
- [ ] drop `g4_split.png`, `g5_window_ratio.png` into `copied/`
- [ ] confirm figure 7 reuses existing 9-checkpoint means or regenerates the bank

## Open decisions for the user

1. Figure 7 (universality): reuse existing `bar_means` (fast) or regenerate all 9 checkpoints (slow, cleaner provenance)?
2. Figures 10 (excitation) and the pole backup: in the main deck or backup-only?
3. Keep building here, or start a fresh chat handing off this README?
