# Augmentation-coulomb: Garcia dry friction in the data generator

Answers one question: **is the settled position offset documented in `scripts/gantry/msd-offset/` a
property of the machine, or an artefact of modelling the stages as frictionless?**

The records are generated closed loop, then the recorded input is replayed open loop through the
baseline. Both the 8-state truth and the baseline get the same friction. Both outcomes are thesis
content; nothing here is a fix for the ANN and no conclusion about training follows from it.

**Nothing under `Matlab-scripts/Augmentation/` is modified**, and `scripts/gantry/msd-offset/` is not
touched. This folder holds only new files. Full rationale: **D-138** in `docs/decisions.md`.

## The friction, and the one way to get it wrong

```
u_eff = u - P * ( cc .* sign( P.' * qdot ) )        cc = [cc1; cc2; ccy]  in STAGE coordinates
```

Friction acts on the **physical rails**, not on the logical coordinates. The state is logical
`[X, Theta, Y, delta_a, ...]`, so the term is built in stage coordinates `[X1, X2, Y]` and projected
back with the pipeline's own `P`. Applying `cc` directly to `[dX, dTheta, dY]` is wrong and looks
plausible; gate A4 exists to make that trap visible.

`delta_a` gets no friction. The absorber is an internal payload DOF, not a rail.

## Parameters, and why there is no sweep

| | value | source |
|---|---|---|
| `cc1` | 16.80 N | Garcia 2013, identified |
| `cc2` | 18.35 N | Garcia 2013, identified |
| `ccy` | 11.60 N | Garcia 2013, identified |

These are identified parameters of the machine the FP model is built on, not knobs. They were already
sitting in `gtd_config.m` and being pushed to the base workspace by `gtd_run_simulation`; they were
simply never consumed, because the Simscape Coulomb blocks are orphaned. Sweeping `cc` over decades to
find the value at which the offset dies would be fitting a physical constant to the outcome it is
supposed to explain, and that is not a measurement.

Garcia identifies them by displacing each axis **at constant velocity**. That is a pure sliding
experiment, so the values are slip forces and the paper measures no breakaway. Worth remembering,
because this test is about what happens at rest.

## `sign`, not `tanh`

D-116 uses `tanh(v/v0)` in `coulomb_lfr.py` because that code is differentiated through and BPTT needs
`dF/dv` at `v = 0`. A data generator is not differentiated through, so that justification is absent
here. Beyond that:

- Garcia Fig. 2 writes `sign` and nothing else. `tanh` is our numerical addition.
- For `|v| << v0`, `tanh(v/v0) -> v/v0`, so smoothed Coulomb is **viscous damping of coefficient
  `cc/v0`**, not friction. At `v0 = 1e-3` that is `3.5e4 Ns/m` on the X row against a real `cg1+cg2`
  of `34.8`. With `K11 = K33 = 0` there is no restoring force, so the stage creeps at `F*v0/cc`
  forever: about `1.4e-4 m` over a 5 s hold at `F ~ 1 N`, against the `4.63e-5 m` X offset being
  measured. The regularization error would be several times the signal.
- `sign` is exactly right everywhere the stage slides and wrong only on the measure-zero set `v = 0`.
  `tanh` is wrong on an interval whose width we chose, and the settled offset lives inside it.

## Solver

The original model runs **variable-step `ode45` at `RelTol = 1e-4`** (measured). That is the wrong
integrator for a discontinuous `sign`: the chart declares no zero-crossing signal, so `ode45` either
crawls at the crossings or steps through them with uncontrolled error, and the output grid depends on
where the crossings land. The copy is forced to **fixed-step `ode4` at `h = 5e-5 s`**, which makes the
step a knob we control, keeps the output grid uniform, and matches the Python RK4.

## The artefacts of bare `sign`, and the diagnostic that bounds them

With no stick state the force flips `+cc`/`-cc` at a zero crossing. Two consequences, both scaling
with `h`:

| artefact | size at `h = 5e-5` | matters? |
|---|---|---|
| chatter, `~(cc_row/m)*h^2` in position | X `1.6e-9 m`, Y `2.9e-9 m` | below the `1e-7 m` floor |
| **ratcheting** | accumulates over the hold | **yes, this is the dangerous one** |

Ratcheting: the two half-cycles decelerate at `(cc-F)/m` and `(cc+F)/m`, so each cycle nets a
displacement toward the residual force `F`, and with no stick state nothing stops it. It looks exactly
like "the offset decayed" and would be read as physics.

**A physical settled offset does not depend on `h`. Both artefacts do.** So `check_step_halving.m`
runs the record at `h` and `h/2` and compares the settled value, with a `cc = 0` pair as the control
that separates ordinary RK4 truncation error from the discontinuity specifically.

- PASS: bare Garcia `sign` is adequate at this step and no stick state is needed.
- FAIL: add a Karnopp stick state, hold when `|v| ~ 0` and `|F_applied| <= cc`. That is Coulomb's law
  at `v = 0` rather than the `sign(0) = 0` placeholder, so it is **not** a deviation from Garcia.

## Files

| File | Role |
|---|---|
| `gantrySystemExtendedCoulomb.m` | Copy of `Augmentation/gantrySystemExtended.m`. The A/B assembly is unchanged and the friction enters only as `dxdt = A*x + B*(u - P*F)`, but `F` is NOT a bare `cc.*sign(v)`: it is the set-valued (Karnopp) solution, with a stick band `V_EPS`, a coupled 3x3 solve over the stuck set through the Delassus matrix `G = P'*Minv_B*P`, and breakaway saturation re-solved in an active-set loop. An earlier revision of this row said "the ONLY change is `u_eff = u - Fc_log`", which described the hard-`sign` version this replaced on 2026-08-01. |
| `check_coulomb_noop.m` | Class A gate, functions only. A1 `cc = 0` bit-identical, A2 `cc > 0` changes the derivative, A3 friction never does positive work, A4 the stage/logical frame trap is detectable. |
| `check_coulomb_reaches_plant.m` | Class B gate, through Simulink. `cc` must actually reach the integrator, and `cc = 0` must be bit-identical to the original model run at the same fixed step. |
| `make_coulomb_model.m` | Reproducible builder for the `.slx` copy; appends `cc1/cc2/ccy` as chart PARAMETERS and forces fixed-step `ode4`. |
| `gantry_additional_state_coulomb_2025a.slx` | The model copy. Built, not hand-edited. |
| `generate_trajectory_data_coulomb.m` | Production generator. Writes to `data/gantry/matlab/trajectory/augmentation_coulomb_karnopp/` (the `augmentation_coulomb/` folder holds the superseded hard-`sign` run). |
| `check_step_halving.m` | The `h` vs `h/2` diagnostic above. |
| `check_garcia_constant_velocity.m` | **Gate 1 (D-204).** Replicates Garcia's own identification experiment: for a prescribed constant-velocity state it SOLVES for the force that holds the velocity, then reads the intercept. Recovers `cc1`, `cc2`, `ccy` separately. Inverts the implementation rather than inspecting it. |
| `check_friction_invariants.m` | **Gate 2 (D-204).** Coulomb's law at 60k states the plant actually visited. Recovers the friction force from its EFFECT (the `cc=0` minus `cc=Garcia` acceleration difference), then checks bound, slip, stick-held, breakaway and per-rail dissipation. |
| `check_leine_collocation.m` | **Gate 3 (D-204).** Tests `leine1998`'s published criterion that all four RK4 collocation points stay inside the stick band during stick, and scans the band margin to find the factor that satisfies it. |
| `coulomb_v_eps.m` | The stick band, parsed from the `V_EPS_MARGIN` line in `gantrySystemExtendedCoulomb.m` so the gates cannot drift from the ODE. Added after exactly that drift occurred: when the margin went 1 -> 3 the gates kept their own copy of the old formula and gate 2 reported a `3.67e+01 N` slip violation that was purely the two definitions disagreeing. |

## Gate results

| Gate | Result |
|---|---|
| A1 `cc = 0` reproduces the original EXACTLY | PASS, max abs diff `0.000e+00` |
| A2 `cc = Garcia` changes the derivative | PASS, min abs diff `1.276e+00` |
| A3 friction never does positive work | PASS, min `v'*Fc` `1.578e-02`. **QUALIFIED by gate 2 (2026-09-19): this is the SUM over rails, and the per-rail statement is FALSE inside the stick band.** A held rail's force is whatever zeroes its acceleration, so its sign is unrelated to the residual velocity's sign, and holding a rail that still carries `\|v\| < v_eps` does bounded positive work. Bounded by `v_eps*cc_i`; measured worst `5.993e-04 W` against the bound `5.994e-04 W`. Sliding friction does positive work EXACTLY never (`0.000e+00`). The sum cannot see this; A3 is not wrong, it is answering a weaker question. |
| A4 stage frame differs from logical frame | PASS, min abs diff `1.525e+01` |
| B1 `cc = 0` bit-identical THROUGH the model | PASS, max abs diff `0.000000e+00 m` |
| B2 `cc = Garcia` changes the trajectory | PASS, max abs diff `4.099e-06 m` (X `3.923e-06`, Theta `3.886e-06`, Y `4.099e-06`) |
| **1** Garcia constant-velocity identification | **PASS to machine precision.** `cc1 = 16.800000` (err `3.55e-15`), `cc2 = 18.350000` (err `0.00e+00`), `ccy = 11.600000` (err `0.00e+00`), every affine fit residual at round-off (`1e-16` to `1e-15`), which is what makes the intercept reading valid rather than a curvature artefact. Run at `MA_FRAC = 0.50`. |
| **1b** frame, from gate 1 rather than from A4 | **PASS, and it is the stronger frame evidence.** `cc1` and `cc2` are recovered SEPARATELY and differ by `1.55 N`, which is impossible unless friction lives in stage coordinates. The E2 Theta intercept is `-0.561875 Nm` against the predicted `Lb/2*(cc1-cc2) = -0.561875`; under a logical-frame law that intercept would be exactly `0`, since the Theta row would take `sign(dTheta) = sign(0)`. |
| **2** Coulomb's law at visited states | **PASS** (re-run at the 3x band). 60000 states of `V1`. Census: sliding `130398`, stuck and held `36440`, broke away `13162`, **neither `0`** (the failure the joint stuck-set solve exists to prevent). Bound `1.10e-13 N`; slip `F_i = cc_i*sign(v_i)` `1.10e-13 N`; stick-held `|a_stage_i|` `5.50e-16` relative; breakaway `8.88e-14 N`; slip dissipation EXACTLY `0.00e+00 W`; stick dissipation `1.798e-03 W` against its own bound `v_eps*cc2 = 1.798e-03 W`, i.e. the bound is tight and understood. |
| **3** Leine collocation criterion | **FAIL at the original 1x band, PASS at 3x.** At 1x, 2 of 6919 stuck-and-held rail-steps (`0.03%`) had a collocation point outside the band, worst `1.849 x v_eps`; the margin scan gave `0` violations from `3x` upward and stayed satisfied to `50x`. `V_EPS_MARGIN = 3` was applied to `gantrySystemExtendedCoulomb.m` on 2026-09-19 and the gate re-run: **`0` of `7254` stuck-and-held rail-steps violate, worst ratio `0.993`.** The criterion governs the STICK MODE only, so broken-away rails are excluded; counting them reports `3.87%` and a non-converging required margin of `11.76x`. |

B1 is run with the ORIGINAL model's solver forced to fixed-step `ode4` **in memory only**, closed
without saving, so B1 and B2 differ in nothing but the chart script. Comparing against the stock
variable-step `ode45` would confound the chart edit with a solver change and B1 could never be
bit-identical.

## Step-halving result: bare `sign` is adequate, no stick state needed

`check_step_halving` on `V1_standstill_Yp10`, 12 s, `h = 5e-5` against `h/2 = 2.5e-5`, settled value
= mean over the trailing 0.25 s.

| axis | `cc` | at `h` | at `h/2` | step sensitivity |
|---|---|---|---|---|
| X | 0 | `-2.555501e-09` | `-2.555501e-09` | `2.23e-18` |
| Theta | 0 | `1.457916e-08` | `1.457916e-08` | `2.17e-16` |
| Y | 0 | `1.000000e-01` | `1.000000e-01` | `4.15e-15` |
| X | Garcia | `-3.135837e-09` | `-3.825449e-09` | **`6.90e-10`** |
| Theta | Garcia | `1.245029e-09` | `3.693930e-10` | `8.76e-10` rad |
| Y | Garcia | `1.000000e-01` | `1.000000e-01` | **`1.23e-09`** |

**VERDICT: PASS.** X `6.90e-10 m` and Y `1.23e-09 m` are both about two decades under the `1e-7 m`
floor, so the settled values are not an artefact of integrating a discontinuous `sign()`. Ratcheting,
the dangerous artefact, is absent at this step. No Karnopp stick state is required.

**Chatter is present, and it is larger than predicted.** Full-record `max |q(h) - q(h/2)|`:

| axis | `cc = 0` | `cc = Garcia` |
|---|---|---|
| X | `2.64e-14` | `7.76e-08` |
| Theta | `1.17e-12` | `2.02e-07` |
| Y | `4.85e-12` | `1.23e-07` |

The `cc = 0` control is at machine precision, so the whole `cc > 0` figure is attributable to the
discontinuity and not to ordinary RK4 truncation. That was the point of running the control. The
instantaneous discrepancy reaches `1e-7` to `2e-7 m`, i.e. AT the floor, roughly 50x the
`(cc_row/m)*h^2` per-step estimate quoted above; that estimate was optimistic. It does not change the
conclusion, because chatter is transient and oscillatory while the settled value is what is being
measured, and the settled value is clean by two decades. It does mean **no instantaneous quantity from
this dataset should be quoted below about `2e-7 m`**, only settled ones.

## Stick fraction, against the frictionless control

The absolute number means nothing on its own: V1 is a standstill record, so the stage barely moves
either way. Only the comparison is informative.

| arm | X1 | X2 | Y | rms\|v\| X1 | rms\|v\| Y |
|---|---|---|---|---|---|
| frictionless | 26.8% | 30.4% | 17.6% | `2.875e-03` | `4.439e-03` |
| coulomb | 61.6% | 67.1% | 40.5% | `1.656e-03` | `3.038e-03` |
| delta | +34.8 pp | +36.7 pp | +23.0 pp | | |

Motion amplitude falls to 0.578 / 0.555 / 0.700 of frictionless on X1 / X2 / Y. So Garcia friction
roughly doubles the stuck share and cuts the excitation to 55 to 70 %, but the record is **not** dead:
the stage still slides for about 40 % of samples on X1 and 60 % on Y, because the 130 to 180 Hz
multisine keeps breaking it free. A friction-carrying benchmark stays usable on this evidence.

## The Theta prediction needs narrowing

D-138 pre-registered a `Lb/2*(cc1-cc2) = -0.562 Nm` friction torque deflecting Theta by about
`1.4e-4 rad`. Measured settled Theta on V1 is `~1e-9 rad`, five decades smaller. The prediction is not
wrong, it was stated for **sustained common-mode X travel**; V1 is a standstill record, so
`sign(dX1)` and `sign(dX2)` alternate at multisine rate and the mean torque averages to near zero. The
prediction should be tested on a sweep record (`T6_ysweep_slow`, `T7_ysweep_fast`), where travel holds
one direction for a long stretch.

**UPDATE 2026-09-19: the STATIC form is now closed, exactly.** Gate 1's E2 experiment holds both X
rails at a common constant velocity, which is the sustained common-mode travel the prediction was
stated for, and solves for the force that holds it. The Theta intercept comes out at
`-0.561875 Nm` against the predicted `Lb/2*(cc1-cc2) = -0.561875 Nm`, to machine precision. So the
torque is confirmed and the `V1` non-observation is fully explained: on a standstill record
`sign(dX1)` and `sign(dX2)` alternate at multisine rate and the mean torque averages to near zero.
What remains open is the DYNAMIC form, i.e. the resulting `~1.4e-4 rad` deflection on a real sweep
record (`T6_ysweep_slow`), which needs a generation run and has not been done.

## Two things to expect, neither of which is a bug

**The controller behaves differently.** `Cfb` is designed on a frictionless linear plant and is frozen
(`cfg.K` and `cfg.C_damp` unchanged, since Coulomb is not representable in a linear plant model
anyway). With friction in the loop the closed loop can hunt or sit in a deadband at standstill. That
is real behaviour of a frictional stage under a friction-blind controller. It does mean this dataset
is **not** interchangeable with the frictionless one for anything but the friction question.

**Theta will not stay clean.** `cc1 /= cc2`, so the Theta row of `P*(cc.*sign(.))` carries
`Lb/2*(cc1-cc2) = -0.562 Nm` under common-mode X motion. Theta is the only sprung axis
(`kb1+kb2 = 3975 Nm/rad`), so it deflects about `1.4e-4 rad` with a sign that flips with the direction
of travel. That is a prediction of Garcia's own identified parameters, not an artefact. The
`msd-offset` brief's assumption that Theta shows no offset stops holding once friction is on, though
it should largely cancel in the truth-minus-baseline difference since both sides carry the same
friction.

## Order of work

1. ~~`check_coulomb_noop`~~ PASS.
2. ~~`make_coulomb_model`~~ PASS, 2 input / 1 output / 22 parameter.
3. ~~`check_coulomb_reaches_plant`~~ PASS.
4. ~~`generate_trajectory_data_coulomb` on `V1_standstill_Yp10`~~ DONE (the record every existing
   offset number is quoted on, so the dataset is directly comparable).
5. ~~`check_step_halving`~~ PASS, so offsets from this dataset may be quoted, settled ones only.
6. Python side: give the baseline in the replay the SAME friction law with the same `P` projection, or
   the replay difference measures a friction-law mismatch instead of the model mismatch it is meant to
   isolate. This is a new module beside `scripts/gantry/msd-offset/plant.py`, importing it, **not** an
   edit to it.

```
matlab -batch "addpath('Matlab-scripts/Augmentation-coulomb'); generate_trajectory_data_coulomb"
matlab -batch "addpath(genpath('Matlab-scripts/Augmentation')); addpath('Matlab-scripts/Augmentation-coulomb'); check_step_halving"
```
