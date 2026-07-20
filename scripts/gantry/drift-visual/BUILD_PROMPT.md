# Handoff prompt — build the drift-visual figure deck

Paste this into a fresh session to build the clean deck.

---

You are building a supervisor-facing figure deck in
`scripts/gantry/drift-visual/`. The full spec is in that folder's `README.md` — **read it first**
(data provenance, frames/conventions, exact figure list, the fast-preview switch). Follow it exactly.

## The issue this deck must demonstrate

1. **The augmented model drifts in free-run.**
2. **The ANN leaves a small non-zero-mean (constant) force on the stiffness-free X/Y axes.**
3. **Nothing in the setup constrains that constant:** the excitation does not excite it and the
   0.1 s training window cannot see its free-run consequence. So the training objective is blind
   to the error that dominates deployment, and **the augmentation is unlearnable as set up.**

It is **not** an energy drift (velocity stays bounded); zero-mean was *expected* but never *enforced*;
the stiffness-free X/Y axes integrate the constant into unbounded drift, while the sprung yaw axis
only parks a bounded offset. This answers supervisor Jan's mail.

## What to build

Three scripts in `scripts/gantry/drift-visual/`, per the README:
- `config.py` — single source of the checkpoint path, V1/V3 trajectories, physical constants, style helpers.
- `generate_data.py` — produces `data/*.npz` + `manifest.json`, with the **mandatory `SOURCE` switch**
  (`fake` / `reuse` / `real`; see README "Fast-preview switch").
- `make_figures.py` — reads `data/*.npz` and writes `figures/*.png` (figures 1-10 in the README table).

Reuse the already-good `scripts/gantry/drift-demo/g_correction_channels.py` as **figure 4**
(ANN correction on all 6 states x 2 frames) essentially unchanged. Drop the existing
`g4_split.png` and `g5_window_ratio.png` into `copied/` (do NOT regenerate them).

## Hard rules (from the README, do not deviate)

- **Full-show**: every figure shows its complete member set — 6 physical states
  (X, Theta, Y, dX, dTheta, dY) in **both frames** (logical + stage X1/X2/Y), positions and
  velocities. Never collapse to "X/Theta/Y". If a figure is positions-only (the counterfactual),
  say so on the figure.
- **State correction, not force** (no mass matrix). **Stage = P^T @ logical** for positions and
  velocities. **Scientific notation** on every y-axis. Titles are questions. <= 3 curves/panel for
  comparisons. Provenance footer naming checkpoint + trajectory on every figure.
- **Collision pass before showing**: render at final size, crop-magnify footer / annotations /
  titles, confirm no overstrike or clipped text.

## Order of work (IMPORTANT)

1. Build the three scripts.
2. Run `SOURCE=fake` and **show the full deck first** (layout/collision preview, seconds — no
   simulation). Wait for approval of the layout.
3. Only then run `SOURCE=reuse` (real data from existing npz) or `SOURCE=real` (full regen) for the
   real figures.

Do not run the slow `SOURCE=real` path until the fake-data layout is approved. Propose each figure's
final look from the fake render before committing to real data.

## Provenance caveat to carry into every trained-ANN figure

All "trained ANN" figures ride on ONE rough checkpoint, `gantry_drift_last.pth` (Optuna Trial-3:
routing all 8 states, lr 1.49e-8, nf 1400 = 0.35 s, 5 epochs, cropped 8k val, stride 100). The
"not a one-run accident" claim rests on the 9-checkpoint universality figure (fig 7), not on this
checkpoint alone. Keep this honest in footers/captions.
