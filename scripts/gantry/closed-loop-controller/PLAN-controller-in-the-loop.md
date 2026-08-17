# Plan of approach: testing the augmentation with the controller in the loop

Three experiments, A, B and C, plus the prerequisites all three share. Written after the
controller was verified against MATLAB to `1.9e-16` (`RESULT.md`), which is what makes any of
this trustworthy: the loop can now be built from equations rather than from a MATLAB object.

Everything below assumes the notation of `controller-in-derivation.tex`:

```
So = (I + Gop Cfb)^-1     Si = (I + Cfb Gop)^-1     w = Cfb r + f_ms
u  = Si w                 y  = So Gop w
```

## 0. The three experiments

| | what changes | what it answers | retraining |
|-|-|-|-|
| **A** | evaluation only, model driven by `r` through `Cfb` | does the augmented model work in the loop? | none |
| **B** | training, model driven by `r`, loss is closed-loop error | can the ANN be trained that way? | full |
| **C** | training, still open loop, loss weighted by `So` | does training against the loop-relevant error give a better model? | full |

Order is fixed: **P, then A, then C, then B.** A needs no retraining and builds the machinery the
other two need. C reuses A's `So`. B is the only one that can diverge and the only one whose value
depends on what C shows.

---

## P. Shared prerequisites

### P1. Closed-loop equivalence gate

**Question.** Does a loop assembled in Python from `Cfb` and the truth model reproduce the record
when driven by `r_sim` alone?

This is not what has been verified so far. Every existing check replays a *recorded* `u`. Driving
from `r` exercises the wiring, the sign convention, the sample alignment and the controller state
initialisation, none of which are currently tested.

**Steps.**

1. Create `closed_loop.py` in this folder with a function that, given a state-derivative function
   `f`, an initial state, `r`, and `f_ms`, integrates
   `e_k = r_k - y_k`, `u_k = Cfb(e)_k + f_ms_k`, `x_{k+1} = RK4(f, x_k, u_k)` at `ts = 5e-5` s.
   Run `Cfb` in its exported state-space form `(A, B, C, D)` from `record_reference_*.mat`, not as
   `num/den`: `L4ss` reaches `1.9e-16` and `L4` only `4.7e-10`.
2. Initialise the controller states to zero and the plant to `[0, 0, Y_op, 0, 0, 0, 0, 0]`, which
   is what Simulink used.
3. Drive with `r_sim` and `f_sim` from `V1_standstill_Yp10.mat`, 12 s, 20 kHz.
4. Compare the resulting `y` against the record's `y`, and the resulting `u` against `u_total`.
5. Apply the ramp diagnostic already in `verify_controller.py` to the error.

**Success criteria.**

- **P1a** `max |y_sim - y_record| <= 1e-6 m` over `t >= 0.5 s`. HEURISTIC: ten times the
  established open-loop replay floor of `1e-7 m`, allowing for the loop feeding solver error back
  through a sensitivity that peaks at `1.80`.
- **P1b** the error contains no secular ramp: ramp fraction `< 5 %`. A ramp means an integrator
  interaction, which is a wiring or initialisation fault, not a discretisation difference.
- **P1c** `max |u_sim - u_total| / rms(u_total) <= 1e-3`.

**If it fails.** Do not proceed. A failure here is a bug in the loop, and every number produced by
A, B or C afterwards would be measuring that bug.

### P2. Sample-rate decision

`Cfb` was designed and discretised at `ts = 5e-5` s (20 kHz). The training pipeline decimates to
4 kHz (`plant.py`, `load_record(fs_new=4000)`). At 4 kHz the 100 Hz bandwidth has only 16 samples
per period and the Tustin images of the poles move.

**Steps.**

1. Re-run `ruleOfThumb`'s formula at `ts = 2.5e-4` s to obtain `Cfb_4k`, using the same `kappa_j`
   (the gains come from the continuous plant and do not change).
2. Compute `So`, `Si`, `T` for both loops on the grid of `loop_sensitivity.py`.
3. Tabulate `sigma_max(So)` at 1, 10, 50, 100, 150, 180, 500 Hz for both.
4. Compute the phase margin of each diagonal loop at crossover for both.
5. Repeat P1 at 4 kHz and record how much worse the gate gets.

**Success criteria.**

- **P2a** the comparison table exists and is logged.
- **P2b** decision recorded in `docs/decisions.md` with justification: run the loop experiments at
  20 kHz, or at 4 kHz with a controller that is not the one that made the data.
- **P2c** flag if `sigma_max(So)` at 150 Hz differs by more than 10 % between the two rates, or if
  any phase margin moves by more than 5 degrees. HEURISTIC thresholds, chosen to catch a loop that
  is qualitatively different rather than to certify one that is not.

**Note.** This is a genuine fork. At 20 kHz the controller is exact but training costs five times
more per record. At 4 kHz training is cheap but the loop is a different loop, and that caveat must
then appear in every result derived from it.

### P3. Sensitivity filter `So`

**Steps.**

1. Extend `export_controller.m` to also export `So` and `Si` as state space at the chosen rate, for
   each record's `Y_op`.
2. Build the same objects in Python from `Gop` and `Cfb`.
3. Verify against MATLAB by driving both with the deterministic test signal already used by
   `test_controller_exact.py` and comparing.

**Success criterion.**

- **P3a** relative error `< 1e-10` between the Python and MATLAB `So`, using the same input bits on
  both sides. This is the L4 protocol: any comparison that rounds one side and not the other is
  measuring storage, not agreement (`RESULT.md`).

---

## A. Closed-loop evaluation of an existing model

**Question.** Put an already-trained augmented model inside the loop the data came from. Is the
loop stable, and does the augmentation improve closed-loop behaviour over the baseline?

**Steps.**

1. Pass P1.
2. Run four models through the same loop and the same `r`, on the same records:
   - **truth**, 8-state, the floor
   - **baseline**, 6-state, no absorber, no ANN
   - **augmented**, a trained checkpoint
   - **baseline + oracle absorber**, the FP+MSD model, as the target the ANN is trying to reach
3. Record per run: `y` against the record's `y`, tracking error `r - y`, applied `u`, and the
   controller state trajectory.
4. Metrics: NRMS of `y - y_record`, per channel; peak and rms of `u` against `cfg.lim`
   (`force_peak = [2000, 2000, 1420] N`, `force_rms = [916, 916, 656] N`); ramp fraction of the
   error; and whether the run completed without divergence.
5. Repeat on a held-out record and a held-out `Y_op`.

**Success criteria.**

- **S-A1 stability.** The augmented model completes 12 s with `|y|` bounded and `u` inside
  `cfg.lim`. This is a binary gate.
- **S-A2 improvement.** `NRMS(augmented) < NRMS(baseline)` on the closed-loop output error, by a
  margin of at least 10x the P1 gate floor, so the improvement cannot be solver noise.
- **S-A3 no integrator interaction.** Ramp fraction of the augmented model's error `< 5 %`. A ramp
  means the ANN has a DC component that the pole at `z = 1` is integrating. Recall the measured
  scale: a bias of `1.28e-9 m` produces `0.172 N/s`, so `2 N` over a record.
- **S-A4 ordering.** The models rank truth `<` oracle `<` augmented `<` baseline on NRMS. A
  violation means the closed-loop metric is measuring something other than model quality.

**What each failure means.**

- S-A1 fails: the augmentation is not usable with the existing controller. This is a publishable
  result and it kills B outright.
- S-A2 fails while the open-loop metric shows improvement: the loop is suppressing exactly what the
  ANN learned, which is what `sigma_max(So) = 0.021` at 10 Hz predicts.
- S-A3 fails: fix the ANN's DC component first (a zero-mean output constraint) before anything else.

**Cost.** No training. Six to eight forward simulations per record.

---

## C. Loop-relevant loss on open-loop training

**Question.** If the residual is weighted by `So` before the norm, so that it approximates the
error the loop would actually produce, does training give a better model by A's metric?

```
current   L = || y_model - y_data ||
option C  L = || So(q) [ y_model - y_data ] ||
```

Justification, from `controller-in-derivation.tex` eq. (16) and (17): the recorded `u_total` is
already `Si w`, so the open-loop residual carries that factor, and left-multiplying by `So` turns it
into the closed-loop output error. Exact in SISO, approximate in MIMO.

**Steps.**

1. Pass P3.
2. Implement `So` as a differentiable state-space recursion applied to the residual inside the loss.
   Initialise its states to zero at each window start; `So` has a zero at `z = 1` and damped poles,
   so its transient decays, unlike the plant integrator.
3. Verify the differentiable implementation against the verified `So` on a fixed signal, forward
   pass only, to `1e-10`.
4. Gradient check: finite differences against autograd on a small window, agreement `< 1e-6`
   relative.
5. Retrain with the same config, seeds and horizon as an existing open-loop run, changing only the
   loss. Row in the run table before launch (D-090).
6. Evaluate both models with **A's** closed-loop metric, and with the standard open-loop metric.
7. Run parameter recovery on both to check the physical parameters did not drift.

**Success criteria.**

- **S-C1 correctness.** Steps 3 and 4 pass at the stated tolerances.
- **S-C2 trainability.** Training completes for at least 3 seeds without divergence, with final
  training loss within a factor 2 across seeds.
- **S-C3 the actual test.** The `So`-weighted model beats the unweighted model on A's closed-loop
  metric. This is the only criterion that decides whether C is worth keeping.
- **S-C4 no physical damage.** Recovered physical parameters move by less than their existing
  spread across seeds. `So` attenuates below 45 Hz by up to 50x, so low-frequency physics rests on
  the projection and `param_loss` alone under this loss, and this criterion checks that they hold.

**What a failure means.** S-C3 failing is a clean negative result: the loop weighting does not
change which model is better, and the open-loop loss is adequate. That is worth knowing and cheap
to establish.

---

## B. Closed-loop training

**Question.** Can the ANN be trained with the loop closed around the model during training?

This is last because it is the only configuration that can diverge, it needs the most new machinery,
and its value is conditional on C.

**Steps.**

1. Pass P1, P2, P3, and A.
2. **Oracle learnability test, before any training.** Insert the true absorber correction in place
   of the ANN, run the closed-loop simulation, and confirm the loss sits at the numerical floor.
   Then perturb the oracle by known multiples and confirm the loss rises monotonically and smoothly.
   This separates learnability from optimisation and needs no gradient.
3. **Controller state initialisation.** The nf-windows start mid-record and nothing initialises the
   9 controller states. Choose one and log it:
   - teacher-force the controller with the recorded `u_fb` over a warm-up prefix, then release
   - extend the encoder to output the controller states
   - restrict training windows to start at `t = 0`
   The integrator state is the hazard: its error never decays, since its eigenvalue is exactly 1.
4. **Zero-mean constraint** on the ANN output, or an explicit DC-blocking filter, before the loop is
   closed. Without it the integrator turns the untrained ANN's arbitrary DC component into a ramp.
5. **Divergence guard**: abort and record the epoch if `|y|` leaves the operating range or `u`
   exceeds `cfg.lim`, rather than training through a NaN.
6. **Warm start** from a converged open-loop checkpoint. Do not train from scratch in closed loop.
7. Gradient check through the closed loop, finite differences against autograd, `< 1e-6` relative.
8. Train, 3 seeds, run-table rows before launch.

**Success criteria.**

- **S-B1 learnability.** The oracle test of step 2 shows a loss at the numerical floor for the true
  correction and a monotone response to perturbation. If this fails, stop: the configuration cannot
  learn and no optimiser will fix it.
- **S-B2 gradient correctness.** Step 7 passes.
- **S-B3 trainability.** At least 3 seeds complete without triggering the divergence guard.
- **S-B4 value.** The closed-loop-trained model beats both the open-loop-trained and the
  `So`-weighted models on A's closed-loop metric, and loses no more than 10 % on the open-loop
  metric. HEURISTIC margin; the point is that a closed-loop gain bought with a large open-loop loss
  is a bad trade for a thesis whose claim is physical interpretability.

**Cost.** Highest of the three. Do not start it before A and C have reported.

---

## Decision points

| after | decision | logged in |
|-|-|-|
| P2 | 20 kHz or 4 kHz for all loop work | `docs/decisions.md` |
| A, S-A1 | if unstable, B is abandoned | `docs/decisions.md` |
| A, S-A3 | if a ramp appears, add the zero-mean constraint before C and B | `docs/decisions.md` |
| C, S-C3 | if the weighting does not change the ranking, B is optional | `docs/decisions.md` |

## Artifacts this plan creates

| file | purpose |
|-|-|
| `closed_loop.py` | the loop, shared by A, B and C |
| `test_closed_loop_equivalence.py` | P1 gate |
| `sensitivity_rate_compare.py` | P2 table |
| `so_filter.py` | verified `So`, used by C |
| `figures/closed_loop_equivalence.png` | P1 evidence |
| `figures/closed_loop_models.png` | A evidence |
| `RESULT-loop.md` | numbers from A, B and C as they land |

## Standing constraints that apply

- Every training run gets a run-table row stating its hypothesis before launch (D-090).
- Any comparison against a stored record must feed both sides identical input bits, or it measures
  storage rather than agreement (`RESULT.md`, the L4 protocol).
- Acceptance thresholds must be data-derived. Where a threshold here is an engineering choice it is
  labelled HEURISTIC, and the two that matter most, P1a and S-A2, are both tied to the measured
  `1e-7 m` replay floor rather than to a model-based quantity.
