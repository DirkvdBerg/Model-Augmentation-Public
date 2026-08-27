# Handoff: give the baseline an absorber state, and prove the training target goes clean

**From**: session of 2026-08-10 | **Branch**: Augmentation | **Effort suggested**: high

## 1. Task

Build a 4-DOF physical block for the augmentation pipeline: the existing 3-DOF gantry plus the
absorber coordinate `delta_a`, so the model carries 8 physical states instead of 6 and
`PHY_IX = np.arange(8)`. Fold in the centre-of-mass correction that already exists in
`scripts/gantry/true-init-augmentation/plant_cog.py`. Gate the block algebraically the way
`plant_cog.py` gates its own correction. Then measure the per-window training target on the new
block and show it collapses from `1.0278e-04 m` to the integrator floor. Finally, decide which
residual the ANN should be asked to learn, because a baseline that carries the absorber at the
truth's parameters has nothing left to augment. Full reasoning, with every number sourced, is in
`scripts/gantry/ann-learnability/PLAN.md`; this handoff is its stage 1.

## 1b. Containment: everything goes in one new folder

**Create `scripts/gantry/absorber-baseline/` and write every file this session produces inside
it.** Nothing outside that folder is modified, at all. Not one line.

Existing code is **imported, never edited**. `plant_cog.py` is a pattern to copy from, not a file
to change. `model_augmentation/` is subclassed. `plant.py` module constants are overridden at
runtime, not rewritten.

Use subfolders inside it to keep the overview. A structure that matches the sibling threads
(`coulomb-offset/`, `true-init-augmentation/`) and is suggested rather than mandatory:

```
scripts/gantry/absorber-baseline/
    IMPLEMENTATION-LOG.md      written as the work happens, including what did not work
    block4.py                  the 4-DOF LFR block, subclassed
    gates/                     G1-G5, one file or one per gate
    diagnostics/               the per-window target measurement
    results/                   JSON, per the project convention as well
    figures/
```

The one exception is diagnostic JSON, which also goes to
`simulations/gantry_subnet/diagnostics/` per project convention. Write it to both, or write it
there and keep a copy; do not skip the project location.

## 2. Out of scope

- **All ANN training.** No training runs of any kind. The arms are listed in `PLAN.md` section 5
  and belong to a later session. This session ends at a measurement.
- **The black box.** `scripts/gantry/ann-blackbox/` has its own brief, `DIAGNOSIS-PROMPT.md`.
  Do not touch it.
- **`model_augmentation/`.** Jan's framework. Extend by subclassing, as `plant_cog.py` and
  `scripts/gantry/real-data-verification/coulomb_lfr.py` do. Never edit it.
- **`kamtin-fp-model/`.** Read only.
- **`scripts/gantry/true-init-augmentation/`, `msd-offset/`, `coulomb-offset/`.** Finished
  threads. Import from them; change nothing in them.
- **MATLAB regeneration**, and the second absorber in the truth (`PLAN.md` 1d-ii). It needs a
  working `A_aug` first, and that is unverified. A user decision, not a step here.
- **Amending `docs/decisions.md`.** Draft any proposed decision in this folder instead.

## 3. Where things stand

Branch `Augmentation`. Tree is dirty and nothing from this thread is committed. New this session:
`scripts/gantry/ann-learnability/PLAN.md`, `scripts/gantry/openloop-check/`,
`Matlab-scripts/Augmentation-no-controller/`, `data/gantry/matlab/trajectory/openloop/`.

Nothing is in flight. No runs pending.

## 4. Established and verified

**The training target is dominated by the absorber initial condition.** Windowed training seeds
each window at its start, runs `nf = 400` steps (0.100 s at 4 kHz), compares to the record. The
truth has 8 states, the model has 6 that the physics initialises, so every window starts with
the absorber at rest in the model and moving in the truth:

```
per-window mean error on Y = -(ma/mh)*vdelta_a(s)*nf*Ts/2    R^2 = 1.0000, unfitted slope to 3.5 %

per-window Y DC             1.0278e-04 m
absorber signal, Y RMS      2.19e-06 m        ratio 47x
mean/RMS measured           0.869             pure ramp = sqrt(3)/2 = 0.866
```

**And it is not the baseline's fault.** The decisive control, true-init log section 4.2:

```
truth model, re-seeded from the complete 8-state IC       Y scatter 1.735e-08 m
truth model, re-seeded from 6 states, absorber zeroed     Y scatter 1.0278e-04 m
baseline,    re-seeded from 6 states                      Y scatter 1.0278e-04 m
                                                          ratio to baseline 1.000, four digits
```

No baseline, no CoM term, no LFR realisation is involved. **This is the measurement the whole
task rests on: an 8-state model with an 8-state IC has a clean target.**

**No ANN can supply the missing quantity.** `vdelta_a` is recoverable from the encoder's
18-sample window at `R^2 = 1.0000`, out of sample, every record class; from the instantaneous
`[x_phys(k), u(k)]` the ANN reads, `R^2 = 0.10` to `0.49`. A nearest-neighbour test on the 5 %
closest pairs in the ANN's input space gives `2.14` for the required correction against `0.0769`
for a control that is a static function of those inputs by construction, which rules out every
static map, not just this architecture.

**Training with the corrupted target jitters rather than learns.** Best point `0.82 %` below
ANN-off, `91 / 91 / 93 %` of validation points above it at lr `1e-7 / 1e-6 / 1e-5`. The same
checkpoint degrades the 12 s free run by `253x` to `3575x`. Recurrence gain reached
`5.2e-08 / 1.0e-05 / 1.6e-04` against the `~0.99` a damped 150 Hz absorber needs at 4 kHz.

**The CoM correction exists and is gated.** `plant_cog.py`, derived inside the LFR rational form
rather than fitted:

```
C1a  ma = 0 constants vs framework, max rel   2.205e-16   (float64 both sides)
C2   max |M_c N_c/d_c - I| over 71 Y          4.441e-16
C3   max |M_c(Y) - M_truth(Y, da=0)[:3,:3]|   8.882e-16
C5   exact replay vs record, V1 X             5.3692e-10 m
```

Its measured worth: `47x` on X and `55x` on dX in the per-window target, and **`1.0x` on Y to
five digits**. It is confound removal, not an offset fix.

**The pipeline, read from source 2026-08-10.**

```
config.py:107  nx_phys = 6        config.py:58  nx_ann = 2      -> nxd = 8
config.py:149  na = nb = 17       config.py:34  encoder_init = 'linear_map'
config.py:63   ann_route_ix = (1,4,6,7)    D-068 diagnostic baseline, NOT the deliverable
model.py:29    PHY_IX = np.arange(6)
model.py:32    state layout = [X, Theta, Y, dX, dTheta, dY, delta_a, vdelta_a]
model.py:89    phy_block = Gantry_State_Block(...)     3-DOF
model.py:133   phy_block READS  rows PHY_IX
model.py:135   phy_block WRITES rows PHY_IX
```

The encoder **already** produces 8 states and rows 6-7 are **already** labelled
`delta_a, vdelta_a`. What they lack is physics: nothing but the ANN writes them, and its final
layer is zero-initialised, so `x_aug = 0.000000e+00` at every step.

**D-103 is a hard constraint**: routing must include X and Y, `(0..7)`. Theta-only must not be
proposed or defaulted to anywhere. The `K = 0` drift is to be solved with X and Y kept.

**The absorber entries, to be copied not derived** (`gantrySystemExtended.m:29-48`):

```
M(Y,delta_a) = M(delta_a,delta_a) = ma      M(Theta,delta_a) = -ma*d      M(X,delta_a) = 0
C(delta_a,delta_a) = ca                     K(delta_a,delta_a) = ka
ma = 0.10*mh = 1.01 kg   L0 = 0.10 m   fa = 150 Hz   zeta_a = 0.05   ca ~ 95 Ns/m
```

**Alternative truth datasets already exist**, each a nonlinear force rather than an extra mode:
`data/gantry/matlab/trajectory/augmentation_coulomb/`, `augmentation_coulomb_karnopp/`,
`augmentation_cubic_n10/`, `augmentation_kxy/`, with `Matlab-scripts/Augmentation-{coulomb,
cubic,kxy}/` behind them and `check_*_noop.m` / `check_*_reaches_plant.m` gates.

## 5. Assumed but not verified

1. **The detuned baseline's per-window DC has no prediction.** The gate in section 10 is for the
   exact-parameter baseline, where the answer is known. What the DC becomes once the baseline's
   absorber is detuned from the truth's is the number the next session turns on, and nobody has
   estimated it.
2. **Whether the nonlinearity datasets carry enough residual to learn from is unmeasured.** The
   argument that they are the right benchmark is structural (same state count, static function of
   the state, not absorbable by an LPV parameter), not empirical.
3. **The Györök `A_aug` parameterisation is unverified.** It appears in this project only at
   `scripts/gantry/coulomb-offset/IMPLEMENTATION-LOG.md:643` as a second-hand citation, and no
   Györök paper is in `literature/`. `scripts/gantry/baseline-null/diagnostics-literature.md:139`
   records that no verified source provides the marginal-mode carve-out the gantry needs: the
   parameterisation contracts, and the gantry has two poles at `z = 1` that must be preserved.
   Do not build on it in this session.
4. **`ma_frac = 0.10`, `L0 = 0.10 m`, `fa = 150 Hz`, `zeta_a = 0.05` are fitted from the data,
   not measured.** The `.mat` files store no absorber parameters.

## 6. Tried and failed

- **Perfect initialisation of the six physical states** -> per-window Y scatter `1.0278e-04 m`
  from exact analytic velocities against `1.0273e-04 m` from the record's finite differences,
  i.e. `1.0x` -> the corruption is the two states that cannot be initialised, not the six that
  can -> true-init log section 3.1.
- **A shorter training horizon** -> the free-run in-window Y RMS is flat at `2.19e-06 m` from
  6 ms to 400 ms while the IC ramp grows linearly, `4x` the signal at `nf = 25` and `56x` at
  `nf = 400`, with `mean/RMS` pinned at the pure-ramp value throughout -> no horizon both
  contains one absorber oscillation and keeps the IC error subdominant -> log section 3.5.
- **Training anyway, three learning rates** -> jittered, not learned; see section 4 -> log 6.4.
- **The X closed form as a general predictor of the replay offset** -> `R^2 = -0.008` across 44
  record-track replays, and `E4_multisine_off` has a `dTheta` change of `2.9e-16` with a
  `1.5e-03 m` offset -> it is a `V1`-specific, mid-record-seeded result, not a law -> this
  session, `scripts/gantry/openloop-check/`.

## 7. Achieved

**Implemented and gated.** `scripts/gantry/true-init-augmentation/plant_cog.py`, the
CoG-corrected 3-DOF LFR block, gates C1a-C5 in section 4. It lives in a diagnostic folder and is
**not** in the training path: `model.py:89` still builds the stock `Gantry_State_Block`.

**Measured and written up.** The mechanism in section 4, from
`true-init-augmentation/IMPLEMENTATION-LOG.md` and its seven diagnostics.

**This session.** `scripts/gantry/ann-learnability/PLAN.md`, the plan this handoff executes
stage 1 of. `Matlab-scripts/Augmentation-no-controller/` plus
`data/gantry/matlab/trajectory/openloop/OL1_multisine_Yp10.mat`, an open-loop record verified
against the Python truth model to `1.5e-12 m`. `scripts/gantry/openloop-check/`, the
with-versus-without-controller comparison.

## 8. The open question

**How large is the residual once the baseline carries an absorber, and is it something a static
ANN can represent?**

Three candidate answers, in `PLAN.md` 1d:

- **detuned absorber**, baseline `fa'` against the truth's 150 Hz. Same state count so the target
  is clean, but the residual is a parameter error that `joint_estimation` (`config.py:37`) could
  fix with no ANN. Run that as a control and it bounds what the ANN could add.
- **nonlinearity in the truth**, using `augmentation_coulomb`, `augmentation_cubic_n10` or
  `augmentation_kxy`. Same state count, and the correction is a static function of the current
  state, so the nearest-neighbour result in section 4 does not apply to it. An LPV-LFR baseline
  is linear in the states, so no parameter tuning absorbs a cubic or a friction term.
- **second absorber in the truth**, Jan's `een massa in baseline en 2 in het systeem`. Baseline 8
  states against truth 10, so the corruption returns one level up. Needs `A_aug` first.

The evidence points at the nonlinearity route. Measuring the residual fraction on those datasets
is what settles it.

## 9. Next action

**Write the 4-DOF block and run its algebraic gates, before any target measurement.**

Subclass the LFR physical block to four coordinates with the absorber entries from section 4, CoM
correction folded in, following `plant_cog.py` term for term: derive the corrected `N`, `d(Y)`
inside the rational form, restate `deriv()` rather than patch the framework, and keep the
arithmetic associativity (Horner, divide after the matmul) because reassociating costs about
2 ulp and that is what broke the equivalent gate in the Coulomb thread.

Gates, mirroring C1a-C3 and C5:

```
G1  at ma = 0 the constants collapse onto the framework's build_poly_constants   tol 1e-14, float64
G2  max |M_c(Y) N_c(Y)/d_c(Y) - I| over 71 Y in [-0.35, 0.35]                   tol 1e-14
G3  max |M_c(Y) - M_truth(Y)| over the full 4x4                                 tol 1e-14
G4  absorber ON vs OFF, max rel |d xdot|                                        must be NON-zero
G5  free replay from the rest IC vs the record's own positions, V1              expect e-9 to e-10
```

G4 exists because G1-G3 would all pass against a block whose absorber is wired to nothing, which
is the trap that let a gate pass in the Coulomb thread while the thing under test had vanished.

Write to `scripts/gantry/absorber-baseline/`, per section 1b, and nowhere else.

## 10. Acceptance criterion

**The number that means done: the per-window Y DC on the new block, at the truth's absorber
parameters, falls from `1.0278e-04 m` to the integrator floor of `1.735e-08 m`, a factor of
about 6000.**

Both figures are measured on the same window grid in
`true-init-augmentation/diag_window_target.py`: 476 windows of `nf = 400` on a stride-100 start
grid, four validation records, 4 kHz, block-mean input, `up_sample = 1`, float64. `1.735e-08 m`
is the T8 arm, the truth model re-seeded from its own complete 8-state IC, so the threshold is
that model's own integrator floor on this data rather than a chosen tolerance.

Pass means within a factor of a few of `1.735e-08`. Anything still of order `1e-05` means the
section 4 mechanism is wrong and the plan stops there; report that plainly, it is as useful a
result as a pass.

Report the same quantity on all six states, not Y alone, since Theta, dTheta and dY carry the
same corruption at `1.0x` and should collapse together.

## 11. Read these first

1. `scripts/gantry/ann-learnability/PLAN.md` — the plan this executes stage 1 of, with every
   number sourced and the benchmark argument in 1d.
2. `scripts/gantry/true-init-augmentation/IMPLEMENTATION-LOG.md` sections 3, 4 and 5 — the
   measurements in section 4 above, and section 4.2 is the control the acceptance criterion uses.
3. `scripts/gantry/true-init-augmentation/plant_cog.py` — the pattern to follow, including why
   `deriv()` is restated rather than patched, and `check_plant_cog.py` for the gate style.
4. `Matlab-scripts/Augmentation/gantrySystemExtended.m` lines 29-48 — the 4x4 `M`, `C4`, `K4` to
   copy.
5. `scripts/gantry/gantry_dynamic/model.py` lines 29-32 and 89-138 — where the block is built and
   wired, and what `PHY_IX` currently excludes.

## 12. Do not

- **Do not modify any file outside `scripts/gantry/absorber-baseline/`.** See section 1b. Import
  from existing code, subclass it, override its constants at runtime; do not edit it. If a change
  outside the folder looks necessary, write down what and why in `IMPLEMENTATION-LOG.md` and
  leave it undone.
- Do not run any ANN training. This session ends at the section 10 measurement.
- Do not edit `model_augmentation/`, `kamtin-fp-model/`, or the three finished thread folders.
- Do not use `ann_route_ix = (1,4,6,7)` anywhere, including diagnostics. D-103 forbids proposing
  or defaulting to Theta-only. Use `(0..7)`.
- Do not seed any replay from the record's `x_logical` velocity rows. `gtd_save_record.m:22`
  builds them with `gradient()` on a float32 signal; on `V1`, a record starting from rest, they
  read `[9.5e-06, -6.2e-05, -1.0e-04] m/s` where they should be zero. Use the rest IC
  `[0, 0, Y_op, 0, ...]` or an analytic replay.
- Do not add the second absorber to the truth, and do not regenerate MATLAB data.
- Do not treat the `A_aug` parameterisation as established. See section 5 item 3.
- Do not report a gate as passing without G4, which is the one that fails if the absorber is
  wired to nothing.

## 13. Operational

Conda env `GraduationProject`. Anything over a few seconds goes to the background with live
streaming, per the running-scripts rule:

```
PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output -n GraduationProject \
    python -u scripts/gantry/absorber-baseline/<script>.py
```

Records: `data/gantry/matlab/trajectory/augmentation/` (130-180 Hz, the augmentation track),
`joint/` (1-200 Hz), `augmentation_coulomb/`, `augmentation_cubic_n10/`, `augmentation_kxy/`.

Loader and truth model: `scripts/gantry/msd-offset/plant.py`, `load_record` at line 116,
`deriv8`, `deriv6`, `rollout`, `to_stage`, `exact_ic`. Note `plant.py:19` hardcodes the
`augmentation` path; override the module constant rather than editing it.

The exact-truth cache and the per-window harness are in
`scripts/gantry/true-init-augmentation/`: `data_exact.py`, `precompute_exact.py`,
`diag_window_target.py`.

Diagnostic JSON goes to `simulations/gantry_subnet/diagnostics/`, per project convention.

Per the run-discipline rule, any measurement arm gets a run-table row before launch.

## 14. Delegation

None. One derivation, one new file, five gates and one existing harness. An Explore subagent
would cost more than reading `plant_cog.py`, which is the whole pattern.
