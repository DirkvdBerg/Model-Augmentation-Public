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
| **v3 DC-BIRTH MONITOR, `gantry-zero-mean/v3_dc_birth_monitor.py` (2026-07-17)** | **FULL X+Θ+Y `[0..7]`**, lr=1e-7 FIXED, joint=False, **nominal θ**, free ANN (orth_observe=True, beta=0 → Z_pts probe attached), stride=10, na=17, up_sample=1, save_flag=False. **Instrumented per optimizer step:** per-row mean/std of ann(Z_pts) (the DC + dynamic part) and **dLoss/d(bias)** = loss gradient along a constant per-row correction (zero-leaf forward hook read in a patched `Adam.step`). 3 seeds, UNFIXED. Contract-preserving (no edit to deepSI/training path). | 400 (nf_seconds=0.100) | sim-RMS selector + per-epoch nf-probe + per-run multi-horizon free-run gap (0.1/0.5/2.0 s) | 1 × 3 seeds | **Hypothesis (Theme C / G-C; Jan + supervisor 07-17): the ANN DC is born in the FIRST update steps. Does it appear with CONSISTENT SIGN across seeds (systematic gradient the windowed loss rewards) or wander in seed-dependently (unconstrained diffusion)?** Pre-declared readings: (1) mean(ann(Z_pts)) vs step on the K=0 rows (dX/dY) — a DC appearing in the first ~tens of steps? (2) dLoss/d(bias) sign on those rows — consistently <0 = loss REWARDS the DC (systematic); ~0 = indifferent (diffusion); (3) sign agreement across the 3 seeds. Context: **G-A CLOSED (v1f 07-17: physics carries no DC the baseline lacks)** → the DC is estimator/training-side. Lit basis (`literature/stability-training/claude-deep-research-inwindow-accumulation.md`): DAgger O(εT²) horizon amplification, Tallec-Ollivier truncated-BPTT bias, Rubruck 2024 optimal-constant-solution-first; geometric g^T d probe used because variance-based gradient tests (gradient noise scale/GSNR) DEGENERATE in near-deterministic full-batch → multi-seed agreement is the systematic test. **WATCH:** train nf-RMS rising from epoch 1 = lr overshoot on the K=0 axes → kill, drop to 1e-8. **Outcome (2026-07-17): DC IS SYSTEMATIC, not diffusion.** Ran twice: v3 (nn.Module forward hook, B=NaN — the interconnect calls `block.forward` directly, `interconnect.py:92`, bypassing hooks) then v3b (patched `ann.forward` + no-op checkpoint_save; A identical to v3, so v3b is canonical). Two bugs fixed first: local-closure hook unpicklable at deepSI checkpoint_save, and wrong state order (delta_a is idx 6 not 3; map columns via cfg.ann_route_ix). **(A) DC birth, 3 unfixed seeds, reproducible in sign:** dY -4.21/-3.62/-3.55e-6 (dominant, NEGATIVE), Y +4.6/+3.7/+4.1e-7, dX +1.1/+0.39/+0.51e-7; born by ~step 13; loss flat 1.738e-6→1.706-1.711e-6 (NO overshoot). Same-sign across independent inits = systematic (diffusion would scatter). **(B) mean dLoss/d(bias) over the epoch, 3 seeds, reproduces in sign+magnitude AND matches the DC direction on the dominant rows:** dY +2.15/+2.47/+2.70e-5 (pushes dY negative = the observed DC), Y -2.06/-1.48/-1.82e-4 (pushes Y positive = observed); dX is the lone mismatch (grad>0 but DC>0, both tiny). Per-seed |t|<1 (loss nearly flat in this direction, minibatch-noisy) but 3-seed replication is decisive — per-seed variance test is underpowered exactly as the lit predicts (GSNR/grad-noise-scale degenerate at full-batch determinism; cross-seed agreement is the instrument). **CONCLUSION:** the windowed loss is nearly NEUTRAL to the DC (flat loss, weak gradient) but carries a small CONSISTENT bias that the K=0 integrators accumulate into the drift = a systematic loss-geometry cause (G-C), NOT random wander. Ties d12/f09 loss-blindness + Rubruck (mean/constant learned first) + Tallec-Ollivier (truncated-BPTT under-prices the slow direction). Data: `gantry-zero-mean/data/v3b_perstep_seed{0,1,2}.npz`; figures `v3b_perstep_seed*.png` + `v3b_multiseed_dc.png`; lit `literature/stability-training/claude-deep-research-inwindow-accumulation.md`. |
| **v3-JOINT broadband excitation (identifiability test), `v3_dc_birth_monitor.py` V3_MODE=joint** (2026-07-18) | SAME as the v3b augmentation run (full X+Theta+Y `[0..7]`, lr=1e-7, joint=False, nominal theta, 3 UNFIXED seeds, nf=400, per-step DC + dLoss/d(bias) instrumentation) EXCEPT `mode='joint'` -> trains on the **1-200 Hz broadband** multisine data (`data/gantry/matlab/trajectory/joint/`, confirmed complete 2026-07-18) instead of the 130-180 Hz narrowband. | 400 | sim-RMS + nf-probe | 1 x 3 seeds | **Hypothesis (identifiability, tests the "DC direction is unconstrained by DC-free data" cause):** the narrowband 130-180 Hz excitation has no low-frequency content, so the ANN's constant-output direction is UNCONSTRAINED and free to be parked; the broadband [1,200] Hz adds low-freq lines that constrain the near-DC gain, so a DC would cost loss. **Prediction:** the systematically-born DC (v3b: dY -4.21/-3.62/-3.55e-6) SHRINKS or VANISHES under broadband. **Falsification:** same-size DC under broadband -> identifiability is not the (sole) cause. Compare directly to the v3b augmentation result. **WATCH:** train nf-RMS rising from epoch 1 = overshoot -> lower lr. **Outcome (2026-07-18, 3 seeds): IDENTIFIABILITY IS NOT THE CAUSE.** Broadband [1,200] Hz dY DC at step 250 = **-4.18/-3.72/-3.82e-6**, essentially IDENTICAL to v3b narrowband (-4.21/-3.62/-3.55e-6). Low-frequency excitation does NOT reduce the DC. Combined with v3x0 (encoder-init refuted), **BOTH candidate causes are now eliminated by intervention**: the DC forms the same regardless of init AND excitation band -> it is intrinsic to the TRAINING DYNAMICS / loss geometry on this architecture+data (a small systematic loss-nearly-flat gradient parking a constant on the K=0 rows in the first ~13 steps), amplified by the K=0 integrator. Fix implication: neither better init nor input design removes it -> the robust fix is a direct SOFT PIN on the DC direction (zero-at-equilibrium). Remaining SOURCE question: G-B (normalization/init, V2 unrun) or a systematic baseline-discretization bias. |
| **v3-TRUEX0 encoder-bypass (encoder-init test), `v3x0_true_init_probe.py`, BUILT 2026-07-18** | Custom loop (`build_pipeline` model, no deepSI fit): each window x0 = **true physical 6** `(x_logical[p]-x_mean)/std_x` + **aug 2 = 0**, free-run nf steps `yhat,x=hfn(x, u_norm)`, mean per-step MSE, train the ANN (lr=1e-7, nf=400, 3 seeds, augmentation band). Per-step DC + dLoss/d(bias) as v3b. **INIT=encoder mode = CONTROL** (must reproduce v3's DC before trusting the true result). **Blocker RESOLVED:** the 2 augmented states are LATENTS with no fixed physical scale (`W^a` random-init, `pre_encoder.py:396`; related to true delta_a only by a fitted map, hence `R2_linmap`), so "true aug init" is undefined -> use aug=0 (tiny, ~equilibrium); the physical velocity is the actual suspect and IS well-scaled. Interface-validated 2026-07-18: true-init step-0 MSE = **2.2e-18** (model reproduces true output from true init), both init modes run. | 400 | per-step DC | 1 x 3 seeds | **Hypothesis (encoder-init, tests the v4 finding):** the DC is the loss-optimal response to the encoder's velocity init error that DOMINATES the within-window K=0 ramp (v4: ~1.5e-4 on Y, ~7x the absorber, present with ANN off). **Prediction:** with the TRUE init the systematically-born DC SHRINKS or VANISHES. **Falsification:** same DC from true x0 -> encoder-init is NOT the cause (drop it per `causal-claim-needs-intervention-not-observation`); the DC comes from the free-direction/training dynamics. With the broadband row this DISTINGUISHES the two candidate causes. **Outcome (2026-07-18, 1 seed): ENCODER-INIT IS NOT THE CAUSE.** Control (INIT=encoder) validated the custom loop: dY DC = -3.98e-6 at step 250, matching v3b (-3.5..-4.2e-6), same growth shape. TRUE-init run (encoder bypassed, x0=[true physical 6; aug 0]): dY DC STILL grows negative to **-3.36e-6** at step 250 = ~85% of the encoder DC, same sign/shape -> feeding true initial states does NOT remove the DC. **Refutes the v4-based encoder-init hypothesis** (dropped per `causal-claim-needs-intervention-not-observation`). v4's within-window init ramp is REAL but is NOT what the DC compensates (else true init would remove it); the ramp and the DC are decoupled. The DC forms regardless of init = inherent to the training on this data -> points at identifiability (broadband row) / free-direction training dynamics. **Doc debt:** `RESULTS-2026-07-17-dc-drift-diagnosis.md` + README leaned encoder-init; correct after the broadband verdict. 3-seed confirm optional (1-seed verdict clear). |
| **SGD-vs-Adam (THE mechanism test), `v3x0_true_init_probe.py` V3X0_OPT=sgd** (2026-07-18) | Same custom-loop probe as v3x0 (INIT=encoder control, already validated vs v3b), lr=1e-7 MATCHED, augmentation band, only the optimizer changed Adam->SGD. Compares the DC each optimizer builds at matched lr. | 400 | per-step DC + loss | 1 seed (+2-seed confirm running) | **Hypothesis (the mechanism):** the DC is Adam's implicit bias -- its per-step normalization (~sign-descent) amplifies a tiny CONSISTENT gradient in the loss-flat DC direction into a steady ~lr walk, whereas SGD (step ~lr*grad) takes a vanishing step in a near-flat direction. Prediction: at matched lr, SGD builds far less DC than Adam AT THE SAME LOSS. **Outcome (2026-07-18, seed 0): CONFIRMED -- ADAM IS THE AMPLIFIER.** SGD dY DC = **+1.98e-9** at step 250 vs the Adam control **-3.98e-6** (~2000x smaller), at the SAME loss (~1.5-2e-6, so SGD is training, not stalled). SGD reaches the same loss WITHOUT the drift-causing constant -> the DC is an ADAM artifact (implicit bias in a loss-flat, marginally-stable K=0 direction), NOT data/physics/architecture. This CLOSES the mechanism: every source refuted (physics v1f, encoder-init v3x0, excitation broadband, normalization, encoder), the loss is flat in the DC direction, and Adam walks the ANN off the zero-mean center into that flat valley (SGD does not). Fix options (consistent): soft DC pin (any optimizer), a non-adaptive optimizer (SGD avoided it here at no loss cost), or an Adam variant without flat-direction creep. 3-seed confirm running (seeds 1,2). |
| **TRUNCATION-LENGTH SWEEP (stage-1 source test), `v3_dc_birth_monitor.py` V3_NF=800** (2026-07-18) | Same as v3b (full X+Θ+Y `[0..7]`, lr=1e-7, joint=False, nominal θ, seed 0, per-step DC + dLoss/d(bias) instrumentation) EXCEPT **nf=800** (new `V3_NF` knob -> `nf_override`) vs the nf=400 baseline. | 800 | sim-RMS + nf-probe | 1 (seed 0) | **Hypothesis (stage-1 SOURCE, from `claude-research-optimizer-SGD-vs-ADAM-v2.md`): the DC comes from truncated-BPTT bias, which does not decay for the λ→1 K=0 integrator; a longer window captures more of the non-decaying sensitivity tail → less bias → smaller DC.** Prediction: DC shrinks with longer nf. **Outcome (2026-07-18): POSITIVE.** nf=400 tail-20 dY DC **-4.21e-6** → nf=800 **-2.26e-6** (46% reduction; dX +1.17e-7 → +4.27e-8, 64%), loss same order (1.34e-6 → 2.90e-6). **RECONCILE (corroboration, NOT new): this sits on the ~1/nf law already found by the prior independent nf-sweep (SLURM 71013, this table): nf={800,1600,2400,3200} → dY-DC present on ALL 9 checkpoints, magnitude ~1/nf, EVERY free-run worse than the epoch-0 8.0e-5 → drift NOT fixed at any nf.** Our 400→800 point (~half on doubling) fits DC ≈ 1.7e-3/nf across nf∈{400..3200}. **Corrected conclusion: 1/nf scaling CONFIRMS truncated-BPTT bias as a source, but PROVES longer fixed windows are a REFUTED fix** — 1/nf is nonzero at every finite nf and any residual DC integrates to unbounded drift (71013 saw this up to nf=3200). This run is DIAGNOSTIC ONLY, not a candidate fix. Only ARTBP (unbiased gradient at fixed nf, NOT a longer fixed window) is untried; lower priority than the direct DC-direction fix. Caveat: nf=800 loss ~2x higher (not perfectly loss-matched) but DC drop far exceeds it. **You cannot out-window the drift → robust fix = DC-direction intervention (zero-mean constraint in sim / soft DC penalty on real data).** Data: `gantry-zero-mean/data/v3nf800_perstep_seed0.npz`. |
| **POLE-PERTURBATION (stage-1 source test), `v3_dc_birth_monitor.py` GANTRY_KX_ART=GANTRY_KY_ART=1000** (2026-07-18) | Same as v3b EXCEPT env-gated artificial stiffness kx=ky=1000 N/m on the K=0 X/Y axes (`gantry_ss.py`, default 0.0 = exact ground truth; flows consistently into K_mat (fnet) + A_combined (Ax)), moving the position poles z=1 → z=1-δ (λ<1). seed 0. | 400 | sim-RMS + nf-probe | 1 (seed 0) | **Hypothesis (stage-1 SOURCE): if marginal stability (λ=1) is what makes the truncated-BPTT bias non-decaying, restoring λ<1 should shrink/kill the DC.** Prediction: DC vanishes with stiffness. **Outcome (2026-07-18): INCONCLUSIVE (confounded).** kx=ky=1000 WRECKED the fit (val sim-RMS 1.6e-4 → **0.106**, 650x worse) and the DC did NOT shrink (dY **-4.9e-6**, dX +8.5e-6 at step 250, comparable to/larger than baseline). Root cause (pre-run pole analysis, confirmed): the position-mode decay rate is **damping-limited** (-c/(2m) ~ -0.323 rad/s, ~INDEPENDENT of stiffness — modes heavily underdamped), so stiffness cannot move the within-window decay (~3% over 0.1 s; natural timescale ~3 s >> window) but injects a fit-wrecking spurious force K·q ~ 1000·0.3 m ~ 300 N the truth lacks. No stiffness threads the needle (large k wrecks the fit, small k does nothing measurable). **The stiffness pole-perturbation cannot test λ→1 for this system; the truncation sweep (prev row) is the clean source test and settled it.** Lesson: `verify-knob-moves-the-target-before-running`. Log/data: `v3pole1k_run.log`, `v3pole1k_perstep_seed0.npz`. |
| **LIPSCHITZ-CAP SWEEP (stability-preserving prototype, D-118), `v6_lipschitz_sweep.py` ANN_LIPSCHITZ** (2026-07-18) | Full X+Θ+Y `[0..7]`, lr=1e-7, joint=False, nominal θ, stride=10, seed 0, EPOCHS=4, per L in {off, 1.0}. `off` = control (current free ANN); `L=1.0` = by-construction Lipschitz cap on the aug ANN (`SpectralCap` soft per-layer spectral-norm cap, `gantry_dynamic/lipschitz.py`; preserves zero-init; L is the static-ANN analog of the Györök contraction rate). Eval: long-horizon free-run drift (tail-RMS \|pos err\| over 2 s, 3 standstill records) full-ANN vs ANN-off, + windowed nf-RMS fit. | 400 | free-run drift + windowed nf-RMS | 4 | **Hypothesis (D-117/D-118 stability-preserving route): the v5 divergence is the ANN destabilizing the long free-run; a by-construction Lipschitz cap bounds the ANN's Jacobian contribution so the augmented state map cannot inject unbounded gain -> the full-ANN drift collapses toward the ANN-off baseline.** Pre-declared reads: (1) does full/off Y-drift ratio drop from the control's >>1 toward ~1 as L tightens? (2) fit cost -- does windowed nf-RMS stay ~flat (cap free) or rise (contraction-vs-fit trade-off)? **Falsification / branch:** drift only dies when the fit degrades or a leaky-integrator low-freq error appears -> contraction is too blunt for the genuine z=1 mode -> switch to the passivity route (pHNN, D-117). Mechanism verified pre-launch: SpectralCap caps per-layer σ at L^(1/n) (measured Lipschitz <= L), preserves zero-init (ANN output 0 at init), pipeline builds + untrained full≡off (ratio 1.000). **Outcome (2026-07-19, boae5mdee; first attempt hit two bugs — see below): CONTROL VALID, L=1.0 NON-BINDING (inconclusive for the hypothesis).** Two bugs fixed first: (1) eval reverted to `_best` = epoch-0 (zero ANN) so full≡off ratio=1.00 — the DRIFT is in `_last` (D-114); (2) the spectral-norm parametrization cannot be `torch.save`'d (deepSI checkpoint). BOTH fixed by no-op'ing `checkpoint_save_system` during train (keeps `_last`, skips the save). Re-run: **control (off) Y ratio = 114× (X 1.24×)** = valid strong divergence. **L=1.0: Y ratio 156× (NOT reduced), fit nf-RMS 5.216e-4 → 5.217e-4 IDENTICAL** -> the L=1 cap NEVER BOUND (the trained ANN's natural Lipschitz is ~1e-2 ≪ 1, so L=1 is orders too loose). Caveat: the ANN-off baseline shifted across runs (encoder trains jointly), so absolute drift moves; ratio is the meaningful quantity and did not improve. **Signal toward D-117:** the ANN destabilizes Y 114× at a TINY Lipschitz (~1e-2) -> the problem is the SIGN/structure of the feedback on the z=1 axis, not magnitude, which a Lipschitz cap cannot fix (passivity can). NOT decisive until a BINDING sweep (L∈{0.1,0.01,0.001} + Lipschitz print) shows whether drift drops toward 1 only as the fit degrades (magnitude-cap fails -> passivity) or a sweet-spot L exists. Data: `v6_lipschitz_sweep.npz`, fig `v6_lipschitz_sweep.png`. |
| **Telica Coulomb recovery, `run_telica_param_recovery.py` USE_COULOMB=True** (2026-07-16) | LPV-LFR **parameter recovery on real Telica** (22-traj split, D-075), NOT augmentation. Trainable Coulomb cc1/cc2/ccy added on top of the 14 physical params as `u_eff = u - F_c`, `F_c = P(cc*tanh(P'*qdot/v0))` (D-116; format verified vs the direct EOM and the supervisor's MATLAB `gantrySystem`+Coulomb to 2.8e-18 m). init cc = **Telica datasheet static friction (maximal)** 43/43/49 N (X/X/Y, telica-xyz-0750-0800-data.pdf; supersedes Garcia's cross-machine 16.8/18.35/11.6), v0=1e-3; datasheet-anchored physical init (D-112). | SEGMENT_LEN=2600 (130 ms), full BPTT (W=None) | windowed normalized-RMS on held-out val OPs (D-076) | 40, lr=1e-2 | **Hypothesis:** the ~50% open-loop NRMSE plateau of run 70821 is a missing-Coulomb artifact (viscous cg1/cg2/cy inflated 6-7x past datasheet to fake dry friction). Adding trainable Coulomb should (1) drop open-loop NRMSE (init-state free-run) on held-out val/test below 70821, (2) relax cg1/cg2/cy toward datasheet, (3) recover cc positive and order tens of N. **Falsification:** NRMSE not lower AND cg/cy stay inflated means viscous inflation is not primarily faking Coulomb (open-loop drift dominated by the double-integrator, not friction structure). Baseline = 70821 (no Coulomb: OL NRMSE ~50% train, up to 65% test). Sanity: `USE_COULOMB=False` reproduces 70821. **Outcome:** _pending launch._ |
