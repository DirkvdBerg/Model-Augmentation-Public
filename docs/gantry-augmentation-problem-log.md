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

**Convention (D-090):** every run with a new hypothesis or new config gets a row here
BEFORE launch, stating the hypothesis it tests; the outcome is added after the run.
Trivial re-runs (same hypothesis, same config) do not get rows.

| Run | Routing | Train nf | Val measure | Epochs | Outcome |
|-----|---------|----------|-------------|--------|---------|
| ANN-capability isolation, entry file on cluster (user 07-12) | Theta+abs `[1,4,6,7]`, **joint=False (plain block at NOMINAL theta -> target = pure absorber residual)**, orth_beta=0 + orth_observe=True (penalty OFF, orth-frac meter ON), lr=1e-3 (max/pre-D-101 rate; tripwire: train nf-RMS rising from epoch 1 = overshoot -> kill, retry 1e-4), stride=100, up_sample=1, na=17 | 400 | sim-RMS selector + nf-probe + [joint-probe] (combo n/a, orth-frac live) | 5 | **Hypothesis: the ANN CAN learn the absorber given step budget and no joint confound.** Judged on: val nf-RMS ↓ substantially; R2_linmap(vdelta_a) rising well past 70784's 0.16; sim-RMS reported, not the verdict. Secondary yield: free-ANN orth-frac trajectory = the natural negation-tendency baseline for Step 10 (70784 with penalty: 0.26→0.11). Context: 70784 (lr=1e-5, joint, beta_center) improved windowed val 13% monotonically but never beat epoch-0 sim-RMS (0.00440 vs best 0.004475 plateau). **Outcome: STILL PENDING — run 70799 was NOT this run.** 70799's own config header shows `lr: 1e-07` (deployed entry file lagged the local lr=1e-3 edit; joint=False + orth_observe were in). What 70799 (lr=1e-7, nominal theta) DID show: val nf-RMS slowly ↓ (4.386→4.245e-5, ~3%/3ep, consistent with tiny steps — NOT evidence on gradient starvation), val sim-RMS ↑ from epoch 1 (1.661e-4→2.237e-4, +35% by ep3; log truncated mid-ep4) — i.e. even lr=1e-7 degrades the sim selector for Theta-routing/nominal-theta; the 07-11 "clean descent at 1e-7" precedent (X+Θ+Y) does not reproduce here. Free-ANN orth-frac ~0.21 flat (vs penalized 0.26→0.11). The lr=1e-3 capability test still needs to run: sync the entry file first (`grep -n "lr=1e-3" scripts/gantry/gantry_interconnect_dynamic.py` must hit on the cluster). |
| orth-projection smoke (plan Step 9), `gantry_optuna.py` N_TRIALS=1 (user-directed deviation from the plan's entry-point wording; cropped 8000-sample val) | Theta+abs `[1,4,6,7]`, joint_estimation=True, DETUNED start (±10%), orth_beta=4.66e-4 (=beta_center, D7.9: V_MSE 1e-4 / E_drift 2.15e-1), lr=1e-5 (Theta rate, D-101 era), stride=100, na=17 (matches the V_MSE measurement context, jobs 68675/68676) | 400 | cropped sim-RMS + per-epoch nf-RMS probe + per-chunk `[orth-probe]` V_orth/param_loss print | 5 (+5 final full-val) | **Hypothesis (health only, model quality NOT judged):** (1) mse/param_loss/V_orth finite every chunk; (2) V_orth responds to training (0 at zero-init ANN is correct, then changes as the ANN moves); (3) no optimizer collapse (train nf-RMS not monotonically rising; WATCH lr overshoot at 1e-5). First build_model triggers the fresh detuned-theta_bar penalty basis (D-111 data-derived states, ~6 min, cached). Launched before Step 8 parity verdicts landed (user direction); Step 8 gates GATE B, not this observation run. **Outcome (20260712_162055): ALL THREE HEALTH CRITERIA MET.** (1) finite throughout: V_orth=3.79e-7, param_loss=3.04e-10, sqrt-loss 7.5e-3→6.8e-3 monotone ↓; (2) V_orth nonzero after training → penalty responds to the moving ANN; (3) no overshoot (train OBJECTIVE decreases smoothly; the rising train nf-RMS 4.2e-5→9.3e-5 [m] on the cropped probe + val sim-RMS best-at-epoch-2-then-rising 9.9e-5→2.0e-4 is the familiar drift signature = model quality, explicitly not judged here). Detuned basis: rank 11 confirmed (gap 4.7e-6), build 370 s. **Additional finding: theta did NOT move in 5 epochs** — param table learned==init to 4 decimals (param_loss 3e-10). Consistent with step count: ~130 Adam steps × lr 1e-5 ≈ 0.13% log-space movement vs the 10% detune. Step-10 design consequence: recovery needs ~10-100× more steps (stride=10 → 10× batches/epoch, and/or more epochs, and/or a separate optimizer lr for log_params via parameters_optimizer_kwargs) — decide before the pair. ANN: VERDICT 'inactive (aug states ~0)', R2_linmap vdelta_a=0.16 — 5 epochs, expected. |
| SLURM 68458 | Model B: ANN → vel[3,4,5]+aug[6,7], no C_aug | 400 | sim-RMS (full traj) | 20 | Best checkpoint = epoch 0. Val loss only increases. x[6]=x[7]=0 (encoder assigns zero to aug states). delta_a R2_linmap = 0.08. Root cause: horizon mismatch (train 400 steps, val ~8000 steps, 20× ratio). |
| — | Model B (vel+aug routing) | 400 | sim-RMS | — | Never tested with windowed validation. Superseded before a windowed run was attempted. Cannot conclude whether routing itself is the failure mode. |
| — | 6-stage curriculum (nf 25→50→100→200→400→sim-RMS) | various | windowed per stage | — | **Never run.** Designed and coded, then abandoned when the `bestfit` global tracking bug was discovered (stages would silently revert weights). Bug fixed before any run. |
| — | Full-state (Jan's method), two-stage curriculum | 400 | Stage 1: 400-step-average-RMS → Stage 2: sim-RMS | 30 + 10 | **Never run.** Superseded by D-068 before first run. |
| **NEXT** | D-068: stiffness-selective routing `STIFF_IX=[1,4,6,7]` (Theta + absorber only; X/Y excluded) | 400 | sim-RMS (single stage) | TBD | **Not yet run.** First test of Jan's `state_augment_specific_states` fix. Goal: confirm K=0 blowup eliminated under single-stage sim-RMS. |
| diag_xy_routing_blowup (attempt 1) | Theta-only `[1,4,6,7]` **vs** Theta+X+Y `[0..7]`, matched seed/data/lr=1e-5 | 400 | sim-RMS (full) + nf-window probe (D-095) | 8 (1 batch/ep, 1 s data) | **Hypothesis:** block-mean data (D-087/D-099) removed the sampling DC-force, so X/Y routing may no longer blow up (supervisor 07-08 "data issue?"). **INCONCLUSIVE — invalid control.** Theta+X+Y went ×550, but Theta-only ALSO went ×23 above init and never beat epoch 0 (only 8 gradient steps on 90 samples; lr likely still too high). Because the control condition itself misbehaves, the run does NOT isolate routing: cannot attribute the X/Y divergence to K=0 structure. Next: fix setup (lower lr / more steps / clip) until Theta-only is stable, then re-run and compare. No decision logged. |
| encoder-window fix (na=27), `gantry_optuna.py` N_TRIALS=1 | X+Θ+Y+abs `[0..7]`, lr=1e-7 FIXED, **na_nb_override=27** (only change vs the 07-11 de-confound control: lr=1e-7, nf=400, na=17, best 8.06e-5, sim-RMS rising) | 400 | cropped 8000-sample sim-RMS selector + per-epoch nf-RMS probe | 15 | **Hypothesis (d8/d9/d10, status doc §3b):** the drift-DC is the loss-optimal compensation of the encoder's dY init bias (+2.7e-4 m/s, init-scheme property); window = 1 absorber period (na=27) collapses that bias 4.3× to statistically zero (d10-P4, paired +2.0 SE) → the DC's reward disappears → training should stop producing the drift. **Pre-declared success criteria:** (1) val nf-RMS improves; (2) val sim-RMS does NOT degrade monotonically past epoch 0 (control: 8e-5→1.1e-3 rising); (3) post-run trained dY-row \|mean\|/rms ≪ 0.997 on windowed passes. **Falsification:** same rising sim-RMS at na=27 → encoder bias was not the dominant DC-reward source. Pre-flight (pipeline-level, `encoder_init_state` through `build_model` with `na_nb_override=27`, closes the d10 P0 rebuilt-map caveat) PASSED: dY bias +2.675e-4 (na=17) → **+9.4e-5 m/s, 0.41 SE ≈ zero** (na=27), per-window std 2.9e-3 → 2.5e-3. **Outcome (SLURM 70558, blade4, 2026-07-11): criterion 1 ✓, criterion 2 ✗ → HYPOTHESIS FALSIFIED.** val nf-RMS monotone ↓ (1.597e-5 → 1.475e-5 over 5 epochs; floor HALVED vs control's ~3.0e-5 — the encoder fix works as an encoder fix) but val sim-RMS monotone ↑ (7.46e-5 → 8.13e-4 after ONE epoch (11×!) → 1.75e-3 by epoch 5) — same drift signature as the na=17 control. Killing the MEAN encoder dY init bias does not remove the DC's training reward. Refined hypothesis (not yet tested): the ANN compensates the PER-WINDOW encoder init error (std 2.5e-3 m/s, 10× the mean, barely reduced by na=27); its average IS the measured DC; killing the mean leaves the per-window reward intact. Alternative: training-SET mean bias ≠ V1 mean bias (never measured — pre-flight used val windows only). NOTE run mechanics: chunked fit() reloads `_best` (= epoch 0, since sim-RMS only degrades) at each 5-epoch chunk boundary, so chunks 2-3 restart from epoch-0 weights (identical probe values confirm); effective training = 5 epochs/chunk, control had the same behavior → comparison fair. |
| **nf-CURRICULUM (warm-started ladder), `gantry_optuna.py` MODE='curriculum'** | **FULL X+Θ+Y `[0..7]`**, lr=1e-7 FIXED, joint=False, **nominal θ** (param_init_detune=None), free ANN (no orth). Ladder nf=400→800→1600→2000, epochs 8/7/6/5 (~26 total, ~12h budget), warm-started via `checkpoint_load_system('_last')` between rungs (ONE build_model) | per-rung nf (windowed), sim-RMS reported not selected | 26 | **Motivation:** ANN not learning at nf=400 (§7 weak-signal: FP residual near floor); supervisors recommend increasing nf; longer nf accumulates absorber signal (§8) → real gradient. First clean nf climb at the correct lr (69399 was confounded by the lr bug; d8's "longer-nf refuted" was the K=0 DRIFT question, distinct from this SIGNAL/learning question). **Pre-declared readings:** (1) does train nf-RMS actually fall across rungs (ANN learns the absorber)? (2) do train nf-RMS (windows) and full sim-RMS improve TOGETHER or SPLIT — split confirms d8-d12 (drift separate from signal, needs Layer 2 on top); together = nf-conditioning solved it. **WATCH:** train nf-RMS rising from epoch 1 of any rung → lr overshoot at that nf (drop lr reactively, don't schedule). **Mechanics:** windowed selector is stride=1 (too slow at long nf) so selector stays sim-RMS; warm-start preserved by reloading `_last` (trained weights) after each rung since `_best`=epoch0 on the drift route. Config validated (routing/lr/nominal-θ/hp). **Outcome (SLURM 70903, blade4, 2026-07-13, PARTIAL — killed mid-rung-2 @61%, no crash):** rung0 nf=400 initial val sim-RMS **8.015e-5 (epoch 0)**; training DEGRADED it (best stayed epoch 0, end ~1.9e-3). rung1 nf=800: 1.916e-3 → best **8.293e-4**. rung2 nf=1600: 8.565e-4 → **4.601e-4** (cut off). **READING (2) = SPLIT confirmed:** window nf-RMS stays low (3.0e-5→5.4e-5→1.1e-4, grows only with rollout length) while full sim-RMS is 1–2 orders higher; longer nf reduces sim-RMS MONOTONICALLY across rungs (1.9e-3→8.3e-4→4.6e-4) but **NONE beat the epoch-0 encoder-init 8.0e-5** — warm-start confounds improvement with recovery from rung-0 self-degradation. → motivated the independent nf-sweep (next row). Side effects: surfaced the nondeterministic cross-rung `cal_validation_error`→None crash (fixed via `__dict__.pop` after `checkpoint_load_system`, D-113). |
| **nf WINDOW-LENGTH SWEEP (independent grid), `gantry_optuna.py` MODE='nf_sweep'** (D-113) | **FULL X+Θ+Y `[0..7]`**, lr=1e-7 FIXED, joint=False, **nominal θ**, free ANN (no orth), stride=100. GridSampler `nf={800,1600,2400,3200}`, each a FRESH build_model from encoder init (NO warm-start), OPTUNA_EPOCHS=8; best nf retrained FINAL_EPOCHS=8 on full val + state-recovery diag | per-nf nf (windowed) + cropped 8000-sample sim-RMS | 8/nf | **Hypothesis:** isolating window length from the curriculum's recovery confound (70903), does ANY nf let a trained ANN beat the epoch-0 encoder-init sim-RMS (**8.0e-5**)? **Pre-declared readings:** (1) each nf's final sim-RMS vs the 8.0e-5 baseline (beat it = window length alone suffices; all worse = window length is not the missing piece); (2) does sim-RMS improve with nf monotonically as the warm-started ladder suggested, once the recovery confound is removed? **WATCH:** train nf-RMS rising from epoch 1 of any nf = lr overshoot at that window. Config validated (compiles; grid {800,1600,2400,3200}; routing/lr/nominal-θ). **Outcome (SLURM 70905 no-ckpt + 71013 rerun with per-trial `_last` saves, 2026-07-13/14): reading (1) = ALL WORSE than 8.0e-5 (best = epoch 0 in every trial; end/ANN-off = 15.7x @800, 5.4x @1600, 1.3x @2400, 1.3x @3200); reading (2) = monotone improvement with nf CONFIRMED cold, SATURATING at 1.3x (2400 = 3200).** Train nf-RMS flat at every nf (~192 steps @1e-7: step-starved for the signal; NOT rising = no overshoot). Checkpoints -> `trial_ckpts_71013/`; G3 companion measured the dY-DC on all of them: present in 9/9 (incl. warm rungs), ALL NEGATIVE, magnitude ~1/nf (OE-2, demo plan §13). Figures: G4/G5 (drift-demo). |
| **ZERO-MEAN PIN (interventional, G9), `gantry_optuna.py` MODE='zeromean'** (2026-07-14) | **FULL X+Θ+Y `[0..7]`**, lr=1e-7 FIXED, joint=False, nominal θ, FRESH init, stride=100, **+ `ZeroMeanPin` attached via the `orth_penalty` hook** (`gantry_dynamic/zeromean_pin.py`: V_pin = beta·Σ_{K=0 rows}(mean_j w_r(Z_j))², Z_pts = data-manifold points stride 100, beta=7e4 HEURISTIC ~10% of loss at the observed unpinned DC; insensitivity over decades is itself a prediction since the direction is loss-neutral, d12). CONTROL = 70903 rung 0 (same nf=400 warm) / 71013 trial 3 (fresh nf=800), already run | 400 | cropped 8000-sample sim-RMS selector + nf-probe + [orth-probe] prints V_pin | 8 | **Hypothesis (the INTERVENTION closing the causal chain; user 07-14 "show something so that it is zero-mean"):** enforcing during training the zero-mean force Jan expects PREVENTS the drift from ever forming. **Pre-declared predictions:** (1) trained dY \|mean\|/rms ≪ 0.997 (~0); (2) free-run stays at/below the no-ANN level 8.0e-5 (envelope ~1, no monotone sim-RMS degradation, vs control 8.0e-5→1.9e-3); (3) window fit within ~2% of the unpinned control (d12/d14: the mean was loss-neutral). **Falsification:** drift persists despite pinned mean → the DC story is wrong/incomplete (against d6+d12); or window fit degrades ≫2% → the direction was NOT loss-neutral (against d12). Boundary (on the slide + module docstring): this naive pin is the sim demonstrator, NOT the deliverable (real-data friction has legitimate nonzero-mean force → frequency-selective pin, d16). Figure: `demo7_g9_intervention.py` after the checkpoint returns. **Outcome:** _pending launch._ |
| **DRIFT-DECK clean checkpoint, `gantry_interconnect_dynamic.py` (user 07-14)** | **FULL X+Θ+Y `[0..7]`**, lr=1e-7 FIXED, joint=False, **nominal θ** (param_init_detune=None → clean baseline), free ANN (no orth), **stride=10** (denser windows), na=17, up_sample=1, save_flag=True | 400 (nf_seconds=0.100) | sim-RMS selector + per-epoch nf-probe | 20 | **Provenance run, not a new scientific hypothesis:** produce a cleaner, properly-validated X/Y-routed `_last` to underlie the `scripts/gantry/drift-visual/` deck, replacing the rough Trial-3 `gantry_drift_last.pth` (removes the "one rough checkpoint" caveat for f01/f03-aug/f04/f05/f06/f08; unifies baseline/decomp on V1). **Expected (per 71013, 9/9 checkpoints):** trained `_last` leaves a negative dY-DC on the K=0 rows and drifts in the V1 free-run; `_best` reverts toward epoch-0 (the sim-RMS selector is blind to the drift, cf. f09) so **the deck uses `_last`, not `_best`** (extract via the `make_drift_checkpoint.py` pattern). Keep the existing 71013 9-checkpoint bank (via `SOURCE=reuse`) for the f07 universality figure. **WATCH:** train nf-RMS rising from epoch 1 = lr overshoot on the K=0 axes → kill, drop lr to 1e-8. **Outcome:** _pending launch._ |
