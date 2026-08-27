# Handoff: give the augmented partition a real parameterised oscillator, and beat the initialisation
**From**: session of 2026-08-11 | **Branch**: Augmentation | **Effort suggested**: xhigh

## 1. Task

Build a new subfolder `scripts/gantry/aug-parameterisation/` and make the augmented state partition
(rows 6-7) a **parameterised marginally-stable-capable oscillator** instead of the dead-beat static
map it is today, then train it through the production interconnect and beat the initialisation on
the deployment metric. The target structure is a `2x2` block with learnable frequency and a
learnable pole magnitude bounded below 1, initialised at the absorber's measured frequency. Three
concrete parameterisations are given in section 4E with their initialisation rules. The reason this
is the right object rather than a guess: the absorber is a lightly damped 158 Hz second-order
oscillator, `2x2` blocks of exactly this form are the standard construction for placing eigenvalues
near `abs(lambda) = 1` and keeping them trainable, and it is measured that our rows 6-7 currently
cannot hold such a mode at all (recurrence gain `0.0` at init, `1.57e-04` after training, against
the `abs(lambda) ~ 0.99` a damped 150 Hz absorber needs). Success is a sustained val sim-RMS below
the epoch-0 reference `1.846056547947228e-04 m`, together with a material drop in the Y per-window
DC, which has never moved by more than 1.40 percent in any run.

## 2. Out of scope

- **Do not modify `scripts/gantry/gantry_interconnect_dynamic.py` or `scripts/gantry/gantry_dynamic/`.**
  Copy what you need into the new subfolder, as `true-init-augmentation/` did by assembling the
  interconnect directly rather than calling `build_model`. The entry file is the production path and
  the comparison reference; changing it destroys the comparison.
- **Do not touch `scripts/gantry/ann-blackbox/`, `full-blackbox/`, or `kamtin-fp-model/`.** The
  black-box arms are a separate, closed thread (section 4A).
- **The black-box pole problem is closed and is not this task.** The literature answer is in
  `docs/gantry-augmentation-problem-log.md` section 11b under "The mechanism". Do not build the
  black-box state-map reparameterisation; the user has seen that option and left it unqueued.
- **Do not re-derive the offset mechanism.** `docs/msd-offset-mechanism-2026-07-29.md` settles it
  closed-form with ratio 1.000. Section 4C summarises what you may rely on.
- **Do not fix the CoG error.** Already done in `true-init-augmentation/plant_cog.py`, gates C1a-C5
  passing.
- **Orthogonal projection / parameter interpretability is a different thread.** This task is about
  whether the augmentation can represent and hold the absorber at all.

## 3. Where things stand

Branch `Augmentation`, last commit `cb9849c` "Create full black box implementation". Tree dirty
across `docs/`, `scripts/gantry/`, `tasks/`, `Matlab-scripts/`, plus untracked
`Matlab-scripts/Augmentation-{coulomb,cubic,kxy,no-controller}/` and `deepSI-master/`.

Nothing is running. Two runs are pending launch and were never started, both in the section 12 run
table: the STAGE 1 cosine-lr rerun and `dc-accumulation/step0b_train_constant.py`. A 12 h BLA run
may or may not have been launched on the user's server; ask before assuming.

`docs/gantry-augmentation-problem-log.md` section 11b was extended this session with the literature
mechanism for the marginal-pole problem; sections 12 and 13 are untouched.

## 4. Established and verified

### A. Marginal poles are the single common cause of all four symptoms

The gantry X and Y have no stiffness (`_K4` diagonal `[0, kb1+kb2, 0, KA]`), so both are pure double
integrators with discrete poles at exactly `z = 1`. On such an axis `x(s) = f(s)/(s(ms+c))`, so
`x(inf) = (integral f dt)/c`: the final offset is set by the residual's **impulse**, not its mean.
That one fact generates every symptom in this project:

| symptom | what gets integrated | evidence |
|-|-|-|
| X offset on open-loop replay | `dM_const * qddot` from the `L0` inertia error | offset doc section 3, `L0 = 0` removes 1145x on V1 |
| Y offset on open-loop replay | absorber momentum with no state to hold it | offset doc section 4, closed form `ma*vdelta_a(t0)/cy`, ratio **1.000** |
| ANN training worse than init | the ANN's own output DC | `dc-accumulation/results/step0_dc_sufficiency.json`, MEAN-ONLY is 99.0 to 99.5 percent of the 12 s error |
| black box cannot learn `z = 1` | not applicable to the augmented model, the baseline supplies the poles | problem log 11b "The mechanism" |

This is worth stating explicitly because the user's framing treats the poles and the offset as two
problems. They are one mechanism with four faces, and it means a fix that controls DC/impulse buys
all of them at once.

### B. Rows 6-7 cannot currently hold an oscillator, and this is measured

- `x_aug(k+1) = h_ann(x_phys(k), x_aug(k), u(k))` **is** a learnable recurrence: per the true-init
  row, `model.py:131` wires the ANN input with no selection matrix so it reads rows 6-7, and `:132`
  writes them. So the object exists. Confirm those line numbers before editing; they are quoted from
  the log, not re-read this session.
- But nothing else writes rows 6-7: per `coulomb-offset/IMPLEMENTATION-LOG.md:650-654`, only
  `phy_block` on `PHY_IX = 0..5` and `ann_block` on `route_ix` connect to `xp`. **There is no `A_aug`
  block.** In Györök's parameterisation that is the degenerate corner `sigma_A -> 0`: a dead-beat
  augmented state.
- Measured recurrence gain (largest singular value of `d x_aug(k+1)/d x_aug(k)`), from
  `diag_aug_state_activity.py`: **`0.0` untrained**, and `5.16e-08 / 1.02e-05 / 1.57e-04` at
  lr `1e-7 / 1e-6 / 1e-5`. It tracks the learning rate linearly rather than the data. Against the
  `abs(lambda) ~ 0.99` a damped 150 Hz absorber needs, that is five orders short.
- `R^2` of the best affine map from the learned rows onto the truth's `[delta_a, vdelta_a]` is
  **0.13**. The rows carry noise, not the absorber.

### C. What the correction needs, and that it becomes representable once the state exists

`diag_static_representability.py`: the ideal per-step correction is `R^2 = 0.879` (V1, narrowband)
and only **`0.033`** (T9, aprbs) explainable by a linear function of the ANN's own inputs
`z = [x_norm(8), u_norm(3)]` with rows 6-7 identically zero. Add `[delta_a, vdelta_a]` and it rises
to **`0.999994` on both records**. A nearest-neighbour test rules out any static `f(z)`
(Delta `2.14`/`1.83` against a control at `0.077`/`0.144`).

So the absorber state is both necessary and sufficient for the correction. This is the strongest
argument that the fix is a *state* with the right dynamics, not more capacity in a static map.

### D. The offset, corrected in one important respect

The user's framing is right on X and needs sharpening on Y.

- **X: correct.** CoG/inertia mismatch, specifically `L0 = 0.10 m` putting `-ma*L0 = -0.101 kg*m` on
  the X-Theta coupling (3.1 percent) and `+0.0707 kg*m^2` on the Theta inertia (1.5 percent). It is
  a settled offset, 86 to 97 percent DC. **Already fixed** by `plant_cog.py`.
- **Y: it is not the missing absorber *dynamics*, it is the missing absorber *state*.** All four
  truth variants give an identical Y offset to four significant figures. Seeding the truth with
  `delta_a = vdelta_a = 0` while keeping all six physical states drops it **4450x** and makes it
  independent of seed time. Closed form `dY(inf) = ma*vdelta_a(t0)/cy`, matching observation at
  ratio 1.000 across six seed phases. The genuine dynamic mismatch contributes **0.02 percent**.
- **Consequence, and it is why Jan's method does not address it:** no parallel *force* block can
  supply a missing initial condition. Only a *state* can. That is a sharper statement than "the
  augmentation is set up for a frequency-domain difference", and it is the mechanical reason this
  task is about `A_aug` rather than about the ANN.
- **But the offset is not the blocker, and this contradicts the premise in the request.** From
  offset doc section 7: baseline with true 6-state `x0` gives Y RMS `7.33e-04`, with encoder init
  `2.11e-04`, oracle with perfect 8-state `x0` `9.02e-05`. **Total headroom from perfect 8-state
  knowledge is 2.3x. One epoch of ANN training costs 127x.** So the reason every run is worse than
  the init is the ANN's own DC, not the offset. Fixing the offset alone cannot get you past the init.

### E. The three parameterisations, with initialisation rules stated to implement

All from the 2026-08-11 literature sweep. Absorber target: linearised eigenvalue **157.89 Hz**,
measured `delta_a` spectrum peak **164.55 Hz**, `ma_frac = 0.10` (fitted, minimum two orders deep).

1. **CTB, continuous-time block (Casoni et al., ICANN 2025, DOI `10.1007/978-3-032-04558-4_43`).**
   `A_hat = Id + tau * diag([[0, omega_n], [-omega_n, 0]])`, eigenvalues `mu_n = 1 +- i*tau*omega_n`,
   one learnable `omega_n` per block. With a dissipation term `alpha_n` on the diagonal (their CTBF),
   `mu_n = (1 - tau*alpha_n) +- i*tau*omega_n`. Projection onto the circle is division of the block
   by `abs(mu_n)`, no Jordan form needed. Their horizontal variant instead solves for the dissipation
   that lands exactly on the circle: `alpha_n = (1 - sqrt(1 - tau^2*omega_n^2))/tau`, valid for
   `tau^2*omega_n^2 <= 1`. Measured: projecting every step enables their task, never projecting
   diverges in every case. Init `omega = 2*pi*157.89`, `alpha` from the absorber's `ca`.
2. **Györök contracting `A_aug`** (`literature/augmentation/Data-driven augmentation of
   first-principles models under constraint-free well-posedness and stability guarantees.pdf`, p9):
   `A = alpha_bar * sigma_A * A_bar` with `sigma_A = sigmoid(alpha)` in `(0,1)` and
   `norm(A_bar)_2 < 1`. **This is the framework's own construction and the one already argued for**
   in `coulomb-offset/IMPLEMENTATION-LOG.md:672`. Its decisive practical property, per the true-init
   row: `sigma_A` starts at `0.5` where our wiring starts at exactly `0`, and Adam moves a weight by
   at most `lr` per step, so the gap is about **where the gain starts**, not whether the object
   exists.
3. **Free-real-part diagonal** (from S4D's own ablation, arXiv:2206.11893 Table 2, where
   unrestricting the real part measured *slightly better*): `A_c = diag(a_j + i*b_j)` with `a_j`
   unconstrained, `A_d = exp(Delta * A_c)`. `a_j = 0` gives `abs(lambda) = 1` exactly at a **finite
   interior parameter value**, which no `exp(-exp(.))`, softplus or sigmoid form can do. This is the
   only one of the three that reaches the unit circle exactly; it is also the only one with no
   stability guarantee.

Prefer 1 or 2. The absorber is genuinely damped (`abs(lambda) < 1`), so per
`coulomb-offset/IMPLEMENTATION-LOG.md:660-670` a hard identity path at `z = 1` is the wrong object
and was already rejected on both physical and literature grounds. Option 3 is listed because it is
the honest answer to "can we reach exactly 1", not because this plant needs it.

### F. Reference numbers any result is scored against

| quantity | value | source |
|-|-|-|
| epoch-0 val sim-RMS, ARTBP protocol, V1-V4 pooled, 12 s | **`1.846056547947228e-04 m`** | verified to 0.064 percent by step 0's control |
| CoG-corrected baseline, 12 s free run from exact IC, V1-V4 | `1.30e-06` to `3.18e-06 m` | true-init row |
| true-init ANN-off exact value, nf=400 free run from exact IC | `6.8486e-05 m` | true-init row, `max abs(w) = 0` measured at init |
| Y per-window DC, ANN-off | `1.03e-04 m`, improved by at most **1.40 percent** ever | true-init row |
| oracle perfect 8-state `x0`, V1 Y RMS | `9.02e-05 m` | offset doc section 7 |
| best DC-removal result to date | `0.868x` init, **but the constant was fitted on the evaluation record** | step 0, C-20 |

## 5. Assumed but not verified

- **That a `2x2` oscillator on rows 6-7 will actually be driven to the absorber's state by the
  windowed loss.** Representability (4C) is not learnability. The gradient still has to find it,
  and the encoder's `W^a` rows have their own dead zone (section 13 of the problem log).
- **That `nx_ann = 2` is enough.** The absorber is one mode, so two rows is the physically minimal
  choice, but nothing has tested `nx_ann = 4`.
- **That the DC problem is solved by giving the state somewhere to live.** Plausible, since the DC
  is provably the absorber IC and a correct absorber state would supply it, but untested and it is
  the central bet of this task.
- **The `model.py` line numbers** in 4B, taken from the logs rather than re-read.
- **Whether step0b's train-computed constant clears the init.** Never launched. This is the cheap
  independent route to the same goal and is the fallback in section 8.

## 6. Tried and failed

- **Full-state routing `(0..7)` at lr `1e-3`** -> 800 to 1634x blowup in one gradient step -> the ANN
  writes a physical row and gets amplified ~400x through the `nf=400` BPTT rollout on a `K=0` axis
  -> D-A / `diag13`, and `docs/decisions.md:3090`.
- **More ANN capacity and longer horizons on the black box** -> best checkpoint early, then
  self-destruction, at nf 400 and 800 and width 16 and 64 -> occurs with **no baseline, no
  interconnect, no routing and no zero-init** -> runs 73940 and 74045. Do not re-run these to
  diagnose the augmented model; they already exclude the augmentation coupling as the cause.
- **Exact initial conditions** -> per-window Y DC scatter `1.0278e-04 m` against a `3.147e-08 m`
  floor, i.e. **1.0x** what the finite-difference rows give -> the residual DC is the absorber IC,
  not an encoder or velocity error -> true-init row, task item (i).
- **Training the existing static ANN with a true initial condition, three learning rates** -> best
  `-0.25 / -0.65 / -0.82 percent`, and **91 to 93 percent of all validation points above the ANN-off
  value** -> no static `f(z)` can produce the DC, as the nearest-neighbour test requires -> true-init
  outcome, P1 null and P2 confirmed.
- **lr at or above `1e-5` with a zero-initialised final layer** -> `+333 percent` at 10 updates,
  `+27833 percent` at `1e-3` -> Adam's update is gradient-normalised so the first step is `+-lr` per
  weight regardless of gradient size -> `coulomb-offset` section 11.4. **A warmup or plain SGD would
  defuse this and neither has been tried.**
- **lr at or below `1e-6`** -> cannot reach the required output magnitude at all: `abs(w)` tops out
  near `2e-05` against a required RMS of `1.13e-03` on the dY row in the same normalised units -> a
  null result there measures the budget, not the model -> true-init row.
- **Removing the ANN's DC using a constant fitted on the evaluation record** -> `0.868x` init, the
  first model to beat its init on the deployment metric -> **not deployable**, the constant saw the
  test data -> step 0, C-20.
- **Assuming the offset is bilinear rectification** -> freezing `M` at `delta_a = 0` changes the Y
  offset by 0.00 percent on four records -> `delta_a` RMS is 22 micrometres against a 0.4 m moment
  arm -> offset doc section 5.

## 7. Achieved

- **CoG-corrected baseline**, `true-init-augmentation/plant_cog.py`, gates C1a-C5 all pass.
  Implemented and validated.
- **Offset mechanism, closed form on both axes**, `docs/msd-offset-mechanism-2026-07-29.md`. X via
  `L0` (1145x ablation), Y via `ma*vdelta_a(t0)/cy` (ratio 1.000, six seed phases). Validated.
  Caveat in its own section 9: the generating scripts printed to stdout and were never moved into
  `scripts/gantry/msd-offset/`, so re-running them is not currently possible.
- **DC sufficiency**, `dc-accumulation/results/step0_dc_sufficiency.json`: on every ARTBP arm the
  per-row mean alone reproduces 99.0 to 99.5 percent of the 12 s error and removing it collapses the
  model to its initialisation floor. Validated, controls passed at 0.064 and 0.133 percent.
- **Absorber parameters fitted from data**, `ma_frac = 0.10` with a minimum two orders deep,
  `L0 = 0.10`. Validated.
- **Literature answer for the marginal-mode problem**, problem log 11b. Written, not implemented.

## 8. The open question

**Is the reason every augmented run loses to its initialisation that the augmented partition cannot
hold the absorber state, or that the windowed loss cannot see the absorber even if it could?**

- *Cannot hold it* -> a parameterised `A_aug` fixes it, and this task is correctly aimed. Evidence
  for: recurrence gain `0.0` at init and `1.57e-04` trained against `0.99` needed; `R^2 = 0.13` onto
  the true absorber state; and `R^2 = 0.999994` once the absorber state is available (4B, 4C).
- *Loss cannot see it* -> `A_aug` will not help, and the fix is the objective. Evidence for: the
  12 s free run degrades **253x to 3575x** on the same checkpoint that improved the windowed metric
  by 0.82 percent, with the initial condition exact by construction, so the 120x train/select
  horizon gap is not an initialisation artefact.
- **What chooses between them:** give rows 6-7 the oscillator, freeze everything else, and teacher-force
  the augmented rows to the truth's `[delta_a, vdelta_a]` for a few hundred updates. If the rows can
  then track the absorber (`R^2` well above 0.13) the partition was the limit; if they cannot, the
  loss is. This is a cheap diagnostic and it should run before the full training arm.

Note for the user, not for the successor to act on: if the answer is "the loss cannot see it", the
next object is the training objective (a DC penalty, or output feedback per the OKID/Simchowitz
route in 11b), not more structure.

## 9. Next action

**Build `scripts/gantry/aug-parameterisation/` with a Györök-form `A_aug` block on rows 6-7
(parameterisation 2 of section 4E, `sigma_A = sigmoid(alpha)` initialised so the block starts as a
157.89 Hz lightly damped oscillator rather than at gain zero), wire it into a locally assembled copy
of the production interconnect, and run the teacher-forcing diagnostic from section 8 before any
training arm.**

Rationale: it is the framework's own construction, it is already the argued-for fix in
`coulomb-offset/IMPLEMENTATION-LOG.md:672`, it composes with the existing Lipschitz cap (D-118), and
unlike parameterisation 3 it cannot reintroduce a marginal mode into a model that already drifts on
marginal modes. The teacher-forcing diagnostic comes first because it costs minutes and it decides
whether the training arm is worth hours.

Write the run-table row in `docs/gantry-augmentation-problem-log.md` section 12 with the hypothesis
before launching the training arm, per the run-discipline rule.

## 10. Acceptance criterion

Two numbers, both against the model's own initialisation rather than an oracle.

1. **Diagnostic gate:** with the augmented rows teacher-forced, `R^2` of the affine map from rows 6-7
   onto the truth's `[delta_a, vdelta_a]` must exceed **0.5**, against the measured **0.13** the
   current static recurrence achieves. Below that, the partition still cannot hold the mode and the
   training arm is not worth launching.
2. **Training gate:** val sim-RMS on the ARTBP protocol (V1-V4 pooled, 12 s) below
   **`1.846056547947228e-04 m`**, on a majority of validation points rather than one lucky
   checkpoint. The 91 to 93 percent above-init figure from the true-init arms is the standard being
   beaten, so report the fraction of points below init, not only the minimum. Secondary and
   diagnostic: the Y per-window DC must fall materially from `1.03e-04 m`, since no run has moved it
   by more than 1.40 percent and the DC is 99 percent of the error.

The floor these are measured against is the CoG-corrected baseline's own `1.30e-06` to `3.18e-06 m`,
so there is roughly two orders of headroom and the criterion is not asking for the impossible.

## 11. Read these first

1. `docs/gantry-augmentation-problem-log.md` section 13 and the true-init row in section 12. The
   measured capability gap in rows 6-7, and the `A_aug` verdict, are both there.
2. `scripts/gantry/coulomb-offset/IMPLEMENTATION-LOG.md` sections 12.1 to 12.4. The Györök
   parameterisation, why the identity path was rejected, and the exact `sigma_A` form.
3. `docs/msd-offset-mechanism-2026-07-29.md` sections 3, 4 and 7. The offset closed form on both
   axes, and the 2.3x-versus-127x arithmetic that says the offset is not the blocker.
4. `scripts/gantry/gantry_dynamic/model.py` lines 96-138. The interconnect you are copying, and the
   place `A_aug` has to go.
5. `scripts/gantry/true-init-augmentation/IMPLEMENTATION-LOG.md`. The most recent full attempt, and
   the falsified premise it ran under.

## 12. Do not

- Do not edit the entry file or `gantry_dynamic/`; copy into the new subfolder (section 2).
- Do not add a hard identity path on rows 6-7. Rejected on physics and literature grounds; the
  absorber is damped, so `abs(lambda) < 1` is correct and an integrator is the wrong object.
- Do not train at lr `1e-5` or above with a zero-initialised final layer without a warmup. The
  `+333 percent` first-step artefact is Adam's normalised update, not a learning signal.
- Do not train at lr `1e-6` or below and read a null as a result about the model; the budget cannot
  reach the required output magnitude.
- Do not cite the first MSD morph `freefloat` numbers `0.618` and `0.999`; noise was scaled to output
  std and buried the dynamics at -55 dB.
- Do not re-run black-box capacity or horizon sweeps to diagnose the augmented model.
- Do not size a long run from a node-speed ratio measured at a different configuration; that cost
  run 74045 its artefacts at 76 percent.
- Do not report a result from `fit()`'s returned `Loss_val`; it is overwritten by the `_best`
  reload and truncates the series exactly when best is not last. Read from `_last`.

## 13. Operational

New subfolder `scripts/gantry/aug-parameterisation/`. Env `GraduationProject`. Data is the existing
`mode='augmentation'` records, 14 train `T1-T14` plus `V1-V4`, via `gantry_dynamic.data.load_datasets`;
no regeneration needed. The `plant_cog.py` CoG-corrected baseline is the one to build on.

Launch per the live-output convention, since the training arm is hours:

```
PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output -n GraduationProject \
  python -u scripts/gantry/aug-parameterisation/<script>.py
```

Write results incrementally. Every artefact in run 74045 was lost because the npz, the gates and
`save_system` all execute after `fit()` returns and the job was killed at 76 percent.

The teacher-forcing diagnostic should run in the foreground; it is minutes, not hours.

## 14. Delegation

**None for the implementation.** The files are named in section 11 and the search space is closed.
One Explore subagent is warranted only if you cannot locate the `Static_ANN_Block` / block-class API
needed to add a linear `A_aug` block, in which case scope it to
`model_augmentation/fit_systems/` and `model_augmentation/` block definitions and cap it at one.
Apply the `@added` / `__project_origin__` / `# CHANGED:` markers if anything lands in
`model_augmentation/`.
