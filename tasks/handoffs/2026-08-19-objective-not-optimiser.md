# Handoff: the ANN's 37 % ceiling is not the optimiser, and reshaping the objective made it worse. Decide what to change next.
**From**: session of 2026-08-19 | **Branch**: Augmentation | **Effort suggested**: high

## 1. Task

Decide, on evidence, what actually caps the closed-loop augmentation, and take the one next step
that follows. A model with the right weights exists in the same function class and closes 82 % of
the free-run headroom against training's 36.7 %, so the target is reachable in principle. This
session eliminated the optimiser as the cause (three optimisers converge to one basin), eliminated
capacity (the architecture fits the exact correction once its outputs are per-row scaled),
eliminated `W^a` (neutral in training), and then tried to fix the objective with two new terms,
**both of which failed**: burn-in makes the model worse on every comparable metric, and the
state-consistency term was inert at the weight chosen. It then MEASURED why: the better model has a
37 % lower training loss but sits behind a **34.83x barrier**, and the gradient at the trained
weights is exactly orthogonal to the direction toward it (section 4.15). Physically the barrier
separates **a static augmentation from a dynamic one**: training converges to a small correction
that ignores its own latent states, which is precisely the case the literature has never tested
(section 4.16). So no optimiser and no objective reshaping reaches the good solution, and the thing
to change is where training STARTS. Act on section 9.

## 2. Out of scope

- **Re-deriving anything in section 4.** Those numbers are measured, with artefacts on disk.
- **`W^a` tuning.** Closed: arm A' shows `W^a = 0` reproduces the random init to four digits in
  training (`1.503676e-06` vs `1.5038e-06`). It is neutral, keep zero, stop testing it.
- **The controller rate.** Closed: 2 kHz and 1 kHz both fail both criteria (section 4.9). The
  controller stays at 4 kHz. A 1 kHz MODEL inside a 4 kHz loop remains open but is not the task.
- **Multiple shooting as a horizon device.** Refuted by reading the code (section 6). Its defect
  term is an encoder criterion, not a length device.
- **`docs/` rewriting.** D-147 and D-148 already carry this session's findings, eleven numbered
  findings in D-148. Add to them, do not restructure them.
- **Do not modify** `kamtin-fp-model/`, `references/step1_reference.old_impl.{json,npz}`, or the
  thirteen dangling `cl_*` scripts listed in the 2026-08-18 handoff section 2.

## 3. Where things stand

Branch `Augmentation`, last commit `4cdb7c1`, which is also `origin/Augmentation`. **The tree is
dirty and nothing from this session is committed.** Modified: `model_augmentation/fit_systems/`
(`interconnect.py`, `closed_loop.py`, `pre_encoder.py`), `scripts/gantry/gantry_dynamic/`
(`model.py`, `rezero_gate.py`), `scripts/gantry/closed-loop-controller/` (`cl_train.py`,
`cl_validation.py`, `cl_headroom.py`, `p2_rate_compare.py`), plus `docs/decisions.md`,
`docs/gantry-augmentation-problem-log.md`, `docs/references.md`, `tasks/todo.md`. New:
`cl_capability.py`, `cl_band_split.py`, `cl_nf_sweep.py`, `cl_burnin_sweep.py`,
`cl_test_burnin.py`, `cl_reachability.py`, `RESULT-plateau.md`, `runners/run_cl_arms.sh`,
`runners/run_cl_train.sh`.

`scripts/gantry/gantry_dynamic/{config,evaluation,orth_penalty}.py` carry ANOTHER session's
uncommitted P1/P1-e work (a separate `log_params` optimizer group with its own `lr_theta` and
`eps_theta = 1e-16`). Not ours, not reviewed here, and the reason the commit decision is still open.

**Nothing in flight locally.** `cl_reachability.py` COMPLETED; its result is section 4.15 and it is
the reason sections 8 and 9 say what they say. Artefact:
`scripts/gantry/closed-loop-controller/runs/cl_reachability.json`.

**In flight, cluster**: SLURM array `run_cl_arms.sh`, four arms, 3 epochs each, epoch 1 reported in
section 4.10. Logs `~/logs/augmentation/closed-loop-controller/arms_<JOBID>_{0,1,2,3}.out`, results
`runs/cl_train_arm{Ap,B,C,D}_<JOBID>.json`. The cluster tree is deployed **by copy, not git**.

## 4. Established and verified

**4.1 The plateau is real and was already on record.** Run 76573 completed all 12 epochs; best at
validation 2, ten flat validations after, series `2.1866, 1.3966, 1.3934, ... 1.3948` (x1e-06 m)
from its own `[cl-val]` lines. The three-point series in its JSON is an artefact of
`checkpoint_load_system` doing `self.__dict__ = torch.load(file)` (`fit_system.py:501`).

**4.2 The training-window error, measured for the first time (D-147).** 3-epoch local rerun
reproduced 76573 to six significant figures (`1.3933793e-06` vs `1.3933723e-06`). Window RMS
`2.2331e-06` -> `1.5279e-06` against a per-window oracle floor of `7.226e-08` (train) /
`7.017e-08` (val), i.e. **21.1x the floor**. No train/val gap (0.7 %). Window error ~ free-run
error, so horizon and generalisation do not explain the error LEVEL.

**4.3 Adam's `eps` froze 77 % of the ANN (D-148 finding 1).** The closed-loop loss is `2.2e-10`,
its hidden-layer gradients `1e-11`..`1e-14`, and Adam stops normalising below `eps = 1e-8`.
Measured after 3 epochs: **139 of 600 ANN parameters moved, exactly the output layer** (16x8+8);
`Wa_psi_y` moved **0 of 108**. Nobody chose this: torch default, deepSI forwards kwargs unchanged
(`fit_system.py:119-125`), Jan passes only `lr` (`interconnect.py:523`).

**4.4 Those gradients are signal, not float32 noise.** Per-tensor cosine between DISJOINT batches:
hidden layers **0.9988-0.9996**, same as the output layer's 0.9994; `Wa_psi_y` at `6e-15` still
0.98. `cos(full batch, mean of halves)` = **1.0000** everywhere. So `eps = 1e-16` is a fix, not a
licence to random-walk.

**4.5 Fixing it changes nothing about the result.** `+36.19 %` vs `+36.13 %` at one epoch, with
600/600 ANN and 2908/3130 encoder parameters now moving.

**4.6 The basin (D-148 finding 3).** Fixed-batch loss at 260 updates: `1.0246e-10`
(`eps 1e-8`, `lr 1e-7`), `1.0137e-10` (`eps 1e-16`, `lr 1e-7`), `1.0251e-10` (`eps 1e-16`,
`lr 1e-5`). **Within 1.1 % across 100x in learning rate and 4.3x in live parameters.** `lr = 1e-5`
ran a full epoch with no NaN, so the `1e-3` NaN of D-101/D-102 happened under the `eps` imbalance
and **every learning-rate conclusion in the run table predates the fix**.

**4.7 The architecture can represent the correction; its output parameterisation cannot
(D-148 finding 4).** `cl_capability.py` regresses the same 600-parameter net onto the exact target
`phi_true - phi_base`. As parameterised today the two latent rows are unfittable (`1-R^2` 0.98 and
0.91); with per-row output scaling they fit to `9.6e-05` and `1.5e-04`. Cause: **the eight required
outputs span nine decades**, `3.9e-08` on X to `1.03` on the latents, from one shared output layer.
X is unfittable in both arms and that is a float32 artefact, not a finding.

**4.8 The ceiling, and the objective barely ranks it.** Planted weights score free run
**`4.177e-07`** (`4.122e-07` from the true `x0`), i.e. **82 % of headroom against training's
36.7 %**. But at `nf = 400` the loss ranks it only **1.25x** above the trained model; 2.54x at
`nf = 3200`; 3.34x on the 12 s free run. Burn-in at `K = 100` gives **3.40x** by no-train
rescoring. 88 % of a CORRECT model's window loss is startup transient against 15 % of a mediocre
one's, and the planted model has 3.9x more startup energy precisely because it USES its latents.

**4.9 The controller cannot be downsampled.** `p2_rate_compare.py` with
`CL_RATES=20000,4000,2000,1000`: `sigma_max(So)` at 150 Hz `1.7983` -> `2.0738` (4 kHz, +15.3 %)
-> `2.5867` (2 kHz, +43.8 %) -> `4.9401` (1 kHz, +174.7 %); phase-margin shifts 3.59 / 8.08 /
17.12 deg against a 5 deg tolerance. **2 kHz and 1 kHz fail both criteria.** The design's roll-off
pole at `10*w_b = 1000 Hz` is at Nyquist at 2 kHz and above it at 1 kHz.

**4.10 THE ARMS, epoch 1 of 3, and both new terms failed.**

| arm | full-window RMS | 12 s free run | encoder grad |
|-|-|-|-|
| A' (`W^a=0`) | `1.503676e-06` | `1.3952833e-06` | `2.210e-07` |
| B (burn-in `K=100`) | `2.331126e-06` | `1.4063660e-06` | `1.009e-09` |
| C (consistency 10 %) | `1.502929e-06` | `1.3945242e-06` | `2.223e-07` |
| D (both) | `2.306332e-06` | `1.4063718e-06` | `1.006e-09` |
| untrained | `2.2331e-06` | `2.1866e-06` | |

Burn-in is 0.8 % **worse** on the free run and pushes the full-window error **above the untrained
model**. The consistency term is inert: C vs A' differ by 0.05 %, D vs B by 5 digits.

**4.11 Why, mechanically, and this is the session's most transferable finding.** **Under Adam,
step size is independent of how informative a gradient is.** Arm B cut the encoder's gradient 219x
(`2.210e-07` -> `1.009e-09`, matching gate B6's predicted 178x) and the encoder moved **just as
far**: `max abs dW` `2.787e-05` vs A's `2.909e-05`. Burn-in does not freeze the encoder; it makes
it travel at the same rate in a direction that carries almost no information. A drifting `x0`
explains the unscored region degrading below the untrained model.

**4.12 `W^a` is our own assumption with no source, and it is neutral.** `hoekstra2026encoder`
(Hoekstra, Gyorok, Toth, Schoukens, arXiv:2602.13108v1) defines `W^a` in Eq. 8 but initialises only
`W^b` (Eqs. 16-17, 28-29, 31-32, 33-35), and its experiment is a static augmentation with
`nx_aug = 0`. His CODE has the same gap: `scripts/ecc_2025/msd_ndof_interconnect_dynamic.py` has
OUR structure (3-DOF truth, 2-DOF baseline, 2 augmented) and uses the default random encoder with
`nf = 200`, 2 epochs, nothing special; `msd_ndof_pre_encoder.py` DOES pretrain the encoder
supervised on the true state and transplant it, but sets `sys_dof = FP_dof = 2`, so `nx_aug = 0`.
Ours is `kaiming_uniform_(a=0)` = `U(+/-0.333)`, **2.45x wider than the `nn.Linear` default** the
old comment claimed. Corrected in `pre_encoder.py` and in the `references.md` row.

**4.13 Gates B1-B6 all pass** (`cl_test_burnin.py`). B5: the consistency penalty takes
`dL/dWa_psi_y` from **exactly `0.000e+00`** to `1.63` with the random `W^a`, and `0 -> 0` with
`W^a = 0` (correct: that is the penalty's fixed point, both sides zero at init). B3: the penalty
does not perturb the trajectory, `max |dy| = 0.000e+00`.

**4.14 The consistency weight was measured, and 10 % is 70x too small.** At `w = 5.8e-11` the term
contributes `3.6e-04` of the MSE's ANN gradient (so a stop-gradient on the target is unnecessary)
and only `1.4e-02` of its ENCODER gradient. Equal encoder contribution needs `w ~ 1.6e-09`, where
the ANN is still perturbed by ~2.5 %.

**4.15 THE DECISIVE RESULT: the better model is NOT reachable by descent** (`cl_reachability.py`,
`runs/cl_reachability.json`). Three findings, and the third reframes the session:

- **The objective is not wrong.** The planted weights have a **37 % LOWER training loss** than the
  trained ones, `6.3371e-11` against `1.0139e-10` on the same fixed batch. The loss genuinely
  prefers the better model, which vindicates the discrimination measurements of 4.8.
- **The gradient does not point at it.** `cos(-grad, w_planted - w_trained)` = **`+0.0000`** at
  `K = 0` and **`-0.0000`** at `K = 100`. Exactly orthogonal, at both objectives.
- **A 34.83x BARRIER separates them.** Loss along `w(t)`: `1.0139e-10` at `t=0`, rising
  monotonically to `3.5313e-09` at `t=0.85`, then falling to `6.3371e-11` at `t=1`. The free run on
  V1 tracks it: `1.361e-06` -> `7.909e-06` -> `3.887e-07`. Two separated basins, not one valley.

Also measured: `|w_trained| = 3.42` against `|w_planted| = 10.87`, so training has found a
SMALL-WEIGHT solution and the good one is a LARGE-WEIGHT solution three times further out.
**This kills the L-BFGS branch as well**: a second-order method is still a local descent method and
a 35x barrier stops it exactly as Adam does. No optimiser and no objective reshaping reaches this
solution from here, which explains burn-in's failure rather than excusing it.
Caveat that limits the claim: the planted weights are one representative of a gauge equivalence
class, so this shows THAT solution is not downhill from here, not that no good solution is.

**4.16 WHAT THE TWO BASINS ARE, PHYSICALLY, and this is the most useful sentence in the file.**
"Small weights against large weights" understates it. The trained ANN has an output of `2e-05` with
its latent rows effectively inert (4.10, 4.14: `dL/dWa` is exactly zero because the latents do not
influence `y` at all). So what training converges to is **a small STATIC correction that ignores
its own dynamic states**. The planted model uses them and implements the absorber. The barrier
therefore separates **a static augmentation from a dynamic one**, and gradient descent from a
zero-init reliably finds the static one.

That is exactly the case the literature does not cover, and we have it from two independent
artefacts (4.12): `hoekstra2026encoder`'s own experiment is a STATIC parallel augmentation with
`nx_aug = 0` (its Eq. 37), and the one code example that does have augmented states,
`msd_ndof_interconnect_dynamic.py`, trains with a default random encoder and nothing special. So
the honest statement of this project's problem is: **nobody has shown that DYNAMIC augmentation
trains at all in this setting, and our evidence is that from a zero-init it collapses to the static
solution.** Treat that as the thesis-relevant framing, not "the ANN plateaus at 37 %".

**A hypothesis with a name, worth testing and worth searching (NOT verified).** A branch whose
final layer is zero-initialised (`zero_init_feed_forward_nn`), trained at a small step size, is the
textbook **lazy training regime**: the network behaves like a linear model in its parameters and
does not learn new features, only a readout over the ones it started with. That is consistent with
everything measured here, including 4.3 (only the output layer moved for three epochs) and 4.7 (a
random-feature readout fits the large-magnitude rows and fails the latent ones). The standard lever
out of the lazy regime is **initialisation scale**, which is what section 9 tests. If that framing
holds, the zero-output-at-init contract this project relies on for D-072 comparability is also what
pins the model in the static basin, and those two goals are in direct conflict.

## 5. Assumed but not verified

- **That burn-in's degradation is caused by the drifting encoder** (4.11). Consistent with the
  numbers but untested: arm D was built to test it and could not, because the term was inert at
  the chosen weight. Settled by rerunning D with `w` calibrated on the encoder-gradient ratio.
- **That the gradient-calibrated weight would rescue burn-in.** Untested.
- **That `eps = 1e-16` is the right value** rather than merely better than `1e-8`. Two points
  tested. Note the countervailing argument: removing the floor means every parameter now takes a
  full `lr` step even where the loss is flat along it.
- **That `W^a = 0` is optimal** rather than better than `kaiming(a=0)`. Two points tested; a small
  non-zero init is untested and would bootstrap the consistency term earlier.
- **That the planted weights are the nearest good model.** They are one representative of a gauge
  equivalence class, so alignment measures against them are a lower bound.
- **The lazy-regime hypothesis (4.16).** That the zero-init output layer is what pins training in
  the static basin is CONSISTENT with everything measured and is NOT verified. Settled cheaply by
  section 9's larger-init arm: if the basin moves, the hypothesis stands; if it does not, the
  barrier has another cause and the literature search of section 8 becomes the route.
- **Anti-aliasing.** `load_traj` decimates `y` by plain subsampling (`d['y'][::D]`) with no filter,
  while `u` is block-averaged (D-087). At 4 kHz that folds content above 2 kHz. The band split
  suggests the energy there is small (`8.4e-08` of a `2.19e-06` total) but it is an assumption.

## 6. Tried and failed

- **`eps = 1e-16` alone** -> `+36.19 %` vs `+36.13 %` -> the 461 unlocked hidden parameters can
  only move `2.6e-05` per epoch at `lr = 1e-7`, four orders below their `0.3` scale -> section 4.5.
- **`lr = 1e-5`** -> `+35.82 %`, 46x more travel, slightly worse -> the objective has the same
  minimiser regardless of step size; see the basin -> `runs/cl_train_eps16_lr1e5.log`.
- **Per-row ReZero gates + `W^a = 0` + `lr 1e-5`, 2 epochs** -> loss `1.7039e-10` -> `1.5392e-10`,
  free run `+19.56 %` -> the only run still descending at the end of its budget, but 50 % above the
  basin the others reach in one epoch, decay rate halved between epochs -> `runs/cl_train_allthree.log`.
- **Burn-in `K = 100` (arms B, D)** -> full-window RMS `2.33e-06`, ABOVE the untrained `2.2331e-06`,
  free run 0.8 % worse -> nothing penalises `[0, K)` and, under Adam, the encoder keeps moving at
  full rate on an uninformative direction -> section 4.11.
- **Consistency term at 10 % of the MSE loss (arms C, D)** -> indistinguishable from no term ->
  the weight was calibrated on loss values whose units differ by `1.7e9`; it supplies 1.4 % of the
  encoder's gradient -> section 4.14.
- **Multiple shooting as a route to horizon** -> would return roughly today's 1.25x ->
  `multiple_shooting.py` RE-ENCODES at every segment start (`x = x_node`), so `n_seg = 8` x
  `nf_seg = 400` contains eight transients, one per 400 samples, exactly today's density; the 2.54x
  at `nf = 3200` came from ONE transient per 3200 -> D-148 finding 10.
- **The D-095 `_NfProbe` as the training-window diagnostic** -> measures the OPEN loop (it calls
  `n_step_error` -> `measure_act_multi`, no `y_data`) -> factor 20 different from the closed-loop
  number on identical windows -> D-147.

## 7. Achieved

**Implemented and validated.** `closed_loop_window_rms`, `window_starts`, `make_window_tensors`
(`closed_loop.py`); `ClosedLoopNfProbe` (`cl_validation.py`); `per_window_floor` (`cl_train.py`);
diagnostics `cl_capability.py`, `cl_band_split.py`, `cl_nf_sweep.py`, `cl_burnin_sweep.py`,
`cl_reachability.py`; six gates in `cl_test_burnin.py`, all passing. `p2_rate_compare.py` is now
`CL_RATES`-configurable and its default output still reproduces the recorded P2a table exactly.

**Implemented, gated, and NOT validated as useful.** The two objective terms in `interconnect.py`:
`burn_in` and `consistency_points`/`consistency_weight`. Both default to exact no-ops (gates B1 and
B4, bit-identical). **The revert deal made with the user: `interconnect.py` was clean at session
start and today's diff is 70 insertions and 1 deletion, all of it these two terms, so
`git checkout -- model_augmentation/fit_systems/interconnect.py` removes them completely and
touches nothing else. `closed_loop.py` and `pre_encoder.py` must NOT be reverted that way**:
the first carried pre-existing uncommitted work, the second is a comment correction that should
stay regardless.

**Documented.** D-147, D-148 (eleven findings), three run-table rows in the problem log,
`RESULT-plateau.md` as the folder-level record, the `references.md` row for `hoekstra2026encoder`,
and the corrected `W^a` comment in `pre_encoder.py`.

## 8. The open question

Section 4.15 closed the optimiser and objective routes. What it leaves is:

**What legitimate, non-oracle initialisation puts training in the other basin?**

The constraint that makes this hard: the planted weights are built from `x_aug`, so they cannot be
shipped or used as an init for a deliverable. Whatever replaces them must come from data.

Candidates, and what would choose between them:

- **Initialisation SCALE, the cheapest to test.** `|w_trained| = 3.42` against `|w_planted| =
  10.87` says the good basin lies at larger weights, and `zero_init_feed_forward_nn` holds the ANN
  branch near zero by construction. A larger init, or the ReZero gate with the branch allowed to
  grow, may land training in a different basin outright. Chosen if the free run moves off
  `1.3934e-06` at all.
- **A data-driven linear residual model.** Fit a small linear state-space model to the residual
  between the record and the BASELINE simulation, and seed the ANN's latent dynamics from it. Uses
  only `u`, `y` and the baseline, so it is legitimate. Chosen if it reproduces a large part of the
  planted model's latent behaviour without `x_aug`.
- **Continuation in the absorber's influence**, if a data-driven proxy for it exists.
- **A literature search, and it is now well-posed rather than a fishing trip** (user's suggestion,
  2026-08-19). Run it through the `deep-research` skill per D-121, one subagent per seed, NOT
  ad-hoc search. Three seeds, in priority order: (1) **lazy versus rich training regimes and
  initialisation scale**, specifically whether a zero-initialised output layer prevents a network
  from learning latent DYNAMICS rather than a static correction; (2) **initialisation of latent
  states in state-space and recurrent models** where the latents are unobserved, where the S4 /
  HiPPO line is precedent that the state-dynamics initialisation decides whether the behaviour is
  learnable at all; (3) **encoder/observer co-estimation in encoder-based state-space
  identification with AUGMENTED states**, which is the documented gap in the Beintema and Hoekstra
  line (4.12). Worth running in PARALLEL with section 9, since that arm is 45 minutes and may
  answer it outright.

**This is the same conclusion the black-box arm already reached, and it is in the run table**: at
matched budget, 85k updates each, the BLA-initialised arm beat the randomly initialised one 9.1x on
best and 3.8x on median, and the marginal modes were "not reachable by gradient descent at any
budget we can run, while the BLA arm keeps what it was given". Initialisation decided that case
too, which raises the confidence that it decides this one.

## 9. Next action

**Train one arm from a LARGER ANN initialisation on the UNCHANGED objective**, because it is one
config change and it directly targets the `3.42` against `10.87` weight-norm gap that section 4.15
measured. Settings: `burn_in = 0`, no consistency term, `eps = 1e-16`, `lr = 1e-7`, `W^a = 0`,
3 epochs, everything else as the D-147 baseline so the comparison is clean.
`ANN_REZERO_GATE=row` already re-initialises the final layer normally behind a zero scalar gate, so
the machinery exists; what is new is letting the branch reach a larger weight norm instead of being
held near zero by `zero_init_feed_forward_nn`. Report the free run against `1.3934e-06`.

If the basin does not move, the next step is the data-driven linear residual initialisation of
section 8, which is a build rather than a config change.

The four cluster arms finish independently; record their epochs 2 and 3 in the run table when they
land, but they are no longer the decision point.

## 10. Acceptance criterion

Free run below **`1.215e-06`** on the four validation records, which is 45 % of the available
headroom against training's current 36.7 %, with the floor at `2.81e-08` and the untrained
reference at `2.1866e-06`. All three numbers come from the same production scorer
(`closed_loop_free_run_rms`), and the floor is data-derived: it is the error `plant.deriv8`, the FP
model plus the TRUE absorber, makes on the same records in the same numpy harness. Runs are
deterministic given the seed and reproduce to six significant figures across machines, so any
difference above ~0.5 % is real.

Secondary, for the encoder: `Wa_psi_y` moving off `0 of 108`, and the `[0, K)` RMS on held-out
validation windows falling. Both are gauge-invariant and data-only. Do NOT use `R2_raw` from
`aug_state_r2`; the augmented states are the model's own coordinates and only `R2_linmap` is
meaningful, and then only as a simulation-side sanity check.

## 11. Read these first

1. `docs/decisions.md` D-148 — eleven findings, all of this session's evidence with numbers.
2. `scripts/gantry/closed-loop-controller/RESULT-plateau.md` — the same story as a folder-level
   record, with the measurement chain in the order it was taken.
3. `runs/cl_reachability.json` — the probe that decides section 9.
4. `runs/cl_burnin_sweep.json` and `runs/cl_nf_sweep_*.json` — the discrimination evidence that
   motivated burn-in, and which training then contradicted.
5. `docs/gantry-augmentation-problem-log.md` section 12, the three 2026-08-19 rows — hypotheses,
   falsifiers and outcomes in the run-discipline format.

## 12. Do not

- Do not re-test `W^a`, the controller rate, or multiple shooting as a horizon device (section 2).
- Do not raise `nf` for training; refuted, divergent at 900, `O(N^3)` within-segment.
- Do not use `x_aug` as a training signal anywhere. It is oracle information and the latent gauge
  is free, so it is also conceptually wrong for the augmented rows.
- Do not add a stop-gradient to the consistency target; measured unnecessary at `3.6e-04` of the
  MSE's ANN gradient (section 4.14).
- **Do not try L-BFGS, or any other optimiser, to bridge the gap.** A 34.83x barrier stops a local
  descent method regardless of order (section 4.15).
- Do not use the planted weights as an initialisation. They are built from `x_aug`, and a claim
  that the method learns the physics cannot rest on a model that was handed it.
- Do not put unguarded commands in a SLURM runner. `set -eo pipefail` is on; an unguarded
  `git rev-parse` killed array job 77300 before `srun`, and the cluster tree is not a git checkout.
- Do not commit without deciding what to do about the P1/P1-e work in
  `scripts/gantry/gantry_dynamic/{config,evaluation,orth_penalty}.py` (section 3).

## 13. Operational

Env `GraduationProject`. Local, per the live-output convention:

```
cd scripts/gantry/closed-loop-controller
PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output \
  -n GraduationProject python -u cl_reachability.py
```

~18 min, dominated by nine 12 s free-run evaluations; drop to `T_LIST = [0, 0.5, 1]` for the free
runs if only the loss curve is wanted, which cuts it to ~5 min. Do NOT pipe it through `grep` while
it runs: the pipe buffers and nothing appears until it exits.

Cluster: deployed **by copy, not git**. The arms need `interconnect.py`, `closed_loop.py`,
`model.py`, `rezero_gate.py`, `cl_train.py`, `cl_validation.py`, `cl_headroom.py`,
`cl_test_burnin.py`, and the whole `gantry_dynamic/` folder (a stale `training.py` there produced
`TypeError: _NfProbe.__init__() missing 1 required positional argument: 'val_sd'`, since its
`_NfProbe` predates commit `5c9a629`). Run `cl_test_burnin.py` there before any array: six gates,
seconds, and it fails immediately on a partial deployment.

```
sbatch scripts/gantry/closed-loop-controller/runners/run_cl_arms.sh
```

`--array=0-3`, `--mem=24gb` per task so four run concurrently; append `%1` to serialise.

## 14. Delegation

None. The next action is reading one JSON and running one training arm, both targeted. If the
barrier branch is taken and a literature question follows about optimisers for ill-conditioned
recurrent objectives, that triggers the `deep-research` skill per D-121, one subagent per seed
question.
