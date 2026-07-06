# Gantry Augmentation — Problem Log

This document records the structural problems encountered when applying Jan's parallel
ANN augmentation framework to the gantry system, what was tried, and where the problem
currently stands.

---

## 1. Jan's Parallel Routing (the reference method)

Jan's framework for dynamic parallel augmentation is defined in
`scripts/ecc_2025/msd_ndof_interconnect_dynamic.py`, lines 91–98 (read directly):

```python
    interconnect.connect_block_signals(ANN_state_block, ["x", "u"], ["xp"])

    interconnect.connect_signals("x", physical_state_model_block, "concat", selection_matrix(np.array([0,1,2,3]), nxd))
    interconnect.connect_block_signals(physical_state_model_block, ["u"], [])
    interconnect.connect_signals(physical_state_model_block, "xp", "additive", expansion_matrix(np.array([0,1,2,3]), nxd))

    interconnect.connect_signals("x", physical_output_model_block, "concat", selection_matrix(np.array([0,1,2,3]), nxd))
    interconnect.connect_block_signals(physical_output_model_block, ["u"], ["y"])
```

**What this does:**
- ANN sees the full state `x` and input `u`, outputs additive corrections to ALL `nxd` state rows (`["xp"]`)
- Physical state block (MSD) also updates rows `[0,1,2,3]` additively
- Output `y` comes from the physical output block only (fixed linear map)

**Why it works for MSD:**  
The MSD has spring stiffness K > 0. Any additive position correction from the ANN is
bounded by the spring: if the ANN pushes position up, the spring pushes back. Gradient
magnitude saturates with rollout length T (oscillatory cancellation). The ANN can learn
non-zero corrections at all rollout lengths without causing drift.

Diagnostic: `scripts/gantry/augmentation-error/diag_spring_stiffness.py`  
Result: K=0 → position grows O(T·ε), gradient grows without bound with T.
K>0 → position oscillates and stays bounded, gradient saturates.

---

## 2. K = 0 Problem (gantry axes)

The gantry X, Y, and bridge axes are electric servo drives with **no spring stiffness
and no passive damping** (K = 0, C ≈ 0 for the physical position equations). These axes
are pure double integrators.

**Consequence of applying Jan's routing to the gantry:**

- ANN outputs additive correction ε to a velocity state row each step
- Velocity correction integrates into position: Δq = ε · N_steps · Ts
- Position correction directly: Δq grows O(N · ε)
- Under K = 0 there is no restoring force — corrections accumulate without bound

At training rollout nf = 400 (100 ms): drift is O(400 · Ts · ε) — small for small ε.  
At validation on the full trajectory (sim-RMS, N ≈ 8000 steps): drift is 20× larger.  
This is the root cause of the original training failure (SLURM job 68458):
**best checkpoint = epoch 0 in all 20 training epochs.**

---

## 3. Training / Validation Horizon Mismatch (original failure mode)

**Setup (job 68458):** training nf = 400 (windowed), validation = sim-RMS (full trajectory).

- Training sees 400-step windows → ANN learns corrections that look good at 400 steps
- Validation applies the model to the full trajectory (8 000+ steps)
- K = 0 drift at 8 000 steps is 20× larger than at 400 steps
- Every epoch makes validation worse than epoch 0 → best model = untrained model

**Fix applied:** switch validation to `'400-step-average-RMS'` (windowed, matching
training nf). This removes the horizon mismatch. Windowed validation and training now
see the same rollout length.

---

## 4. Model A — C_aug Dead Zone (first routing attempt)

**Architecture:** ANN → augmented state rows [6, 7] only; a learned `Parameterized_Linear_Output_Block` (C_aug) maps x_aug → y contribution.

**Problem:** C_aug is initialized near zero (Frobenius norm ≈ 1e-2). The gradient of the
loss with respect to ANN parameters flows through C_aug. Near-zero C_aug ≈ 0 means
near-zero gradient → ANN receives no training signal.

Diagnostic: `scripts/gantry/augmentation-error/diag_gradient_routing.py`  
Result: Model A ANN gradient norm = 1.04e-2. Model B (velocity routing) = 2.85e-1.
Ratio B/A = 27×. C_aug confirmed as the gradient bottleneck.

---

## 5. Model B — Velocity Routing (second attempt, superseded)

**Architecture:** ANN → velocity rows [3, 4, 5] + augmented rows [6, 7] via
`expansion_matrix(VEL_AUG_IX, nxd)`. C_aug removed; only fixed `Linear_Output_Block`.

**Result:** Gradient dead zone resolved (27× stronger gradient). However, training still
failed: best checkpoint remained epoch 0. Windowed validation was not yet in place; the
K = 0 horizon mismatch still dominated.

**Note:** routing to velocity-only rows reduces but does not eliminate K = 0 drift.
Velocity corrections still integrate into position errors over N steps (O(N · Ts · ε)).
This approach was superseded before a proper windowed-validation run was attempted.

Decision log: D-066 (superseded by D-067).

---

## 6. Jan's `bestfit` Global Tracking Bug

**Discovery:** Jan's framework (`model_augmentation/fit_systems/interconnect.py`, line 426)
initialises `self.bestfit = float('inf')` in `init_model()` only — not at the start of
each `fit()` call. Each `fit()` compares new validation scores against the global minimum.

**Consequence for curriculum training:**

- Stage 1 sets bestfit = 0.05 (25-step-average-RMS, low because windowed)
- Stage 2 validation (50-step-average-RMS) scores ≈ 0.07 — larger absolute value
- `if self.bestfit >= Loss_val` is never true in stage 2 → checkpoint never saved
- At end of stage 2 `fit()`, best checkpoint is loaded → **model reverts to stage 1 weights**
- Stages 2–6 effectively run training but discard all learned weights at the end of each stage

**Fix applied:** `fit_sys.bestfit = float('inf')` immediately before each stage's
`train_model()` call in `train_model_with_diagnostics()`. Each stage now tracks its own
best checkpoint independently.

---

## 7. FP Model Near-Perfect Accuracy (weak gradient signal)

**Observation:** Initial validation before any training:
`400-step-average-RMS = 0.00015686001` (windowed, FP model + zero ANN).

This is the FP model's prediction error on 400-step windows. The FP model is physics-based
and already extremely accurate. The ANN starts at zero (via `zero_init_feed_forward_nn`)
and needs to find corrections that improve on this near-perfect baseline.

**Consequence:** The gradient signal available to the ANN at nf = 400 is very small.
The dominant unmodeled dynamics (absorber coupling) contribute only a tiny fraction of
the output error at 400 steps. The ANN may fail to learn because the improvement
opportunity is below the effective noise floor of gradient descent.

This is the key structural difference from Jan's MSD benchmark, where the FP model is
intentionally undermodeled (2-DOF approximating 3-DOF system), providing a large
residual for the ANN to learn.

---

## 8. Absorber Dynamics Timescale

The absorber natural period is approximately 100 ms = 400 samples at 4 kHz. This means:

- nf < 400: absorber barely moves within the rollout window → no gradient signal for absorber dynamics
- nf = 400: one full absorber oscillation visible → first rollout length where learning is physically possible
- nf >> 400: absorber dynamics accumulate; stronger signal but K = 0 drift re-emerges for non-zero-mean corrections

Short curriculum stages (nf = 25, 50, 100, 200) do not contain absorber dynamics and
provide no useful training signal for the augmented states. The minimum useful rollout
is nf = 400.

---

## 9. Training Always Windowed (not the full trajectory)

Jan's framework trains on batches of nf-length windows extracted from the training
trajectories. The full training trajectory is never used in a single forward pass.
This is the same for both MSD and gantry.

- `'X-step-average-RMS'` validation: also windowed (rolling X-step windows on val set,
  state re-initialized from encoder at each window start)
- `'sim-RMS'` validation: full trajectory via `apply_experiment` (single initial condition,
  no state resets, errors compound)

The windowed validation does NOT test whether corrections generalize to the full
trajectory. Only sim-RMS does. Corrections learned at nf = 400 generalize to the full
trajectory only if they are zero-mean over the absorber period (so K = 0 drift cancels).

---

## 10. Current Status and Open Questions

**What is implemented (as of 2026-07-01):**

- Full-state routing (Jan's method): `ic.connect_block_signals(ann_block, ["x", "u"], ["xp"])`
- C_aug removed; single physical output block
- Windowed validation matching training nf at each curriculum stage
- `fit_sys.bestfit = float('inf')` reset before each stage
- 6-stage curriculum in code: nf 25→50→100→200→400 (windowed) → 400 (sim-RMS)
- Per-stage checkpointing with SLURM job ID in filename
- Resume via `RESUME_CHECKPOINT` environment variable

**What is decided but not yet implemented:**

- D-068: stiffness-selective routing `STIFF_IX = [1, 4, 6, 7]` via `expansion_matrix`
  (Jan's `state_augment_specific_states` fix — see section 11)

**Open questions:**

1. Does stiffness-selective routing (D-068) eliminate the K=0 blowup under sim-RMS?
   This is the next experiment — single-stage sim-RMS run.
2. Will the ANN learn at all given the near-perfect FP model baseline (0.00015686 windowed)?
3. Do the multisine training trajectories sufficiently excite the absorber natural frequency?
   If not, delta_a ≈ 0 throughout and there is nothing for the ANN to learn regardless of routing.
4. Is the windowed first stage still necessary after D-068 eliminates K=0 drift at X/Y?

**Diagnostic scripts:**

| Script | What it tests | Key result |
|--------|---------------|-----------|
| `scripts/gantry/augmentation-error/diag_gradient_routing.py` | Gradient norm at epoch 0 for Model A vs B | Model B 27× stronger; C_aug confirmed dead zone |
| `scripts/gantry/augmentation-error/diag_spring_stiffness.py` | K=0 drift and gradient growth vs rollout length | K=0: gradient unbounded O(T); K>0: saturates |
| `scripts/gantry/augmentation-error/diag_gradient_series.py` | Gradient through series topology | (see file) |

---

## 11. Jan's Response and Proposed Fix (2026-07-01)

Email sent 2026-06-30 described the K=0 blowup, the K matrix, and the C_aug dead zone.
Jan replied 2026-07-01:

**Key points from Jan's reply:**

1. **Confirms accumulation hypothesis:** "het lijkt me wel voor de XY grid van de
   wirebonder, aangezien dat een integrator is" — the XY grid is an integrator and
   accumulation is expected. A disturbance test can verify this directly.

2. **Validates windowed validation:** suggests `"100-step-RMS"` as an alternative
   validation measure to see what happens before error accumulates. This is exactly the
   approach we independently arrived at (section 3).

3. **Concrete fix:** "je zou kunnen proberen alleen de state te augmenteren die wel een
   spring heeft. Dit kan met de `state_augment_specific_states` flag en de indices array
   eronder." Route ANN corrections ONLY to states with spring stiffness (K > 0).

**Proposed routing for the gantry:**

From the K matrix (transformed coords X, Theta, Y):
- X (index 0, 3): K = 0 — do NOT route
- Theta (index 1, 4): K = kb1+kb2 > 0 — route here
- Y (index 2, 5): K = 0 — do NOT route
- delta_a (index 6): absorber spring — route here
- vdelta_a (index 7): absorber spring — route here

`STIFF_IX = np.array([1, 4, 6, 7])`

**What changes in `build_model()`:**

Replace:
```python
ic.connect_block_signals(ann_block, ["x", "u"], ["xp"])   # full-state routing
```
With:
```python
ic.connect_block_signals(ann_block, ["x", "u"], [])
ic.connect_signals(ann_block, "xp", "additive", expansion_matrix(STIFF_IX, nxd))
```
ANN output width changes from `nxd = NX_PHYS + NX_ANN` to `len(STIFF_IX) = 4`.

**Trade-off:** absorber-to-X/Y coupling is not directly captured (X/Y rows excluded).
The absorber dynamics themselves (delta_a evolution) are still learned. This is a
physically motivated simplification: the dominant error from the absorber appears in its
own state evolution, not in the gantry translation states.

**Implementation note:** Jan refers to a `state_augment_specific_states` flag in his
framework. Inspect `model_augmentation/fit_systems/` to find the exact API before
implementing — do not reconstruct from first principles.

---

## 12. Training Run Log

| Run | Routing | Train nf | Val measure | Epochs | Outcome |
|-----|---------|----------|-------------|--------|---------|
| SLURM 68458 | Model B: ANN → vel[3,4,5]+aug[6,7], no C_aug | 400 | sim-RMS (full traj) | 20 | Best checkpoint = epoch 0. Val loss only increases. x[6]=x[7]=0 (encoder assigns zero to aug states). delta_a R2_linmap = 0.08. Root cause: horizon mismatch (train 400 steps, val ~8000 steps, 20× ratio). |
| — | Model B (vel+aug routing) | 400 | sim-RMS | — | Never tested with windowed validation. Superseded before a windowed run was attempted. Cannot conclude whether routing itself is the failure mode. |
| — | 6-stage curriculum (nf 25→50→100→200→400→sim-RMS) | various | windowed per stage | — | **Never run.** Designed and coded, then abandoned when the `bestfit` global tracking bug was discovered (stages would silently revert weights). Bug fixed before any run. |
| — | Full-state (Jan's method), two-stage curriculum | 400 | Stage 1: 400-step-average-RMS → Stage 2: sim-RMS | 30 + 10 | **Never run.** Superseded by D-068 before first run. |
| **NEXT** | D-068: stiffness-selective routing `STIFF_IX=[1,4,6,7]` (Theta + absorber only; X/Y excluded) | 400 | sim-RMS (single stage) | TBD | **Not yet run.** First test of Jan's `state_augment_specific_states` fix. Goal: confirm K=0 blowup eliminated under single-stage sim-RMS. |
