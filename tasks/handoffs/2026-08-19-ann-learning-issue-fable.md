# Handoff: the augmentation's two extra states have no dynamics. Give them some, and prove it survives noise.
**From**: session of 2026-08-19 | **Branch**: Augmentation | **Model**: Claude Fable 5 | **Effort**: high

## 0. Why you are being asked this

This is a TU/e master thesis with ASMPT, on augmenting a physics-based LPV-LFR model of a dual-gantry
die bonder with a learned dynamic parallel component. The supervisors are M. Schoukens, R. Toth and
J. Hoekstra, and Hoekstra is the author of the augmentation method itself. The deliverable has to
work on real ASMPT Telica hardware, not only on the simulated dual-gantry, so a fix that depends on
noiseless data or on oracle knowledge is worth nothing here even if it scores well.

The augmentation currently closes 36.3 % of the available headroom and stops. Four sessions have
attacked that plateau from the optimiser side and the objective side, and both routes are now closed
with evidence. Yesterday's session measured the actual cause. Your job is to act on it.

Read `scripts/gantry/closed-loop-controller/ANN-learning-issue/` first. Three files: `README.md`
(what happened), `RESULTS.md` (every measured number with its artefact), and
`HYPOTHESES-AND-SOLUTIONS.md` (the diagnosis and the noise-transfer analysis). The predecessor
handoff is `tasks/handoffs/2026-08-19-objective-not-optimiser.md`; its section 9 has been overtaken
by the measurement below and you should not execute it.

## 1. Task

Change the augmentation's parameterisation so the augmented states have live dynamics at
initialisation, initialise those dynamics from a frequency BAND rather than from an identified
resonance, train one arm, and report the 12 s free-run sim-RMS against `1.3933793e-06 m`. The band
form of the initialisation is not a stylistic preference: the real Telica data provably cannot supply
an identified resonance (section 4), so a mode-based initialisation would work in simulation and have
nothing to run on when it reaches the machine.

## 2. How to work on this

When you have enough information to act, act. Do not re-derive what section 4 already establishes,
and do not re-litigate the decisions in section 12. If you are weighing a choice, give a
recommendation rather than a survey.

Before reporting progress, audit each claim against a tool result from this session. Only report work
you can point to evidence for; if something is not yet verified, say so explicitly. If the training
run fails or plateaus, say so with the numbers.

Do not add features, refactor, or introduce abstractions beyond what the task requires.
`model_augmentation/` is Jan's framework and every addition there needs the marker convention in
`CLAUDE.md`. A parameterisation change does not need surrounding cleanup.

Pause for the user only when the work genuinely requires them: a destructive action, a real scope
change, or a decision listed in section 8. Otherwise proceed end to end.

For the final summary: lead with the outcome. The first sentence should answer what happened, in
plain language, for a reader who did not watch any of the tool calls. Supporting detail after.

## 3. Where things stand

Branch `Augmentation`, last commit `4cdb7c1`, which is also `origin/Augmentation`. **Nothing from
2026-08-19 is committed.** The tree is dirty in:

* `model_augmentation/fit_systems/`: `closed_loop.py` and `pre_encoder.py` (both pre-existing
  uncommitted work, leave them). `interconnect.py` was reverted yesterday and is byte-identical to
  `4cdb7c1`.
* `scripts/gantry/gantry_dynamic/`: `model.py` and `rezero_gate.py` (previous sessions' work), plus
  `config.py`, `evaluation.py` and `orth_penalty.py`, which carry **another session's** P1/P1-e work
  and are the reason the commit decision is still open.
* New and untracked: this handoff, the `ANN-learning-issue/` folder, five new scripts (section 13),
  and two `patches/` folders.

**Nothing in flight.** Arm E was killed at 673 of 780 iterations with no result; its run-table row in
`docs/gantry-augmentation-problem-log.md` section 12 still reads `OUTCOME: pending` and should be
corrected to record that it was killed and superseded.

## 4. Established and verified

Measured yesterday, with artefacts on disk. Do not re-derive these.

**4.1 The augmented states have no dynamics.** `cl_aug_spectrum.py`, artefact
`runs/cl_aug_spectrum.json`. The spectral radius of `A_aa = d x_a,k+1 / d x_a,k`, by autodiff on the
real model step at 24 rollout points:

| arm | `rho(A_aa)` median | `rms(x_a)` at depth 400 |
|-|-|-|
| untrained, as built | `0.000000e+00` | `0.000000e+00` |
| trained (`XadbYQ_best`) | `2.887762e-10` | `2.234954e-09` |
| planted, 82 % of headroom | `9.755435e-01` at `159.36 Hz`, `zeta 0.0981` | `1.020372e+00` |

The trained states decay by `3e-10` per step, i.e. they do not survive one sample. The planted ones
have a 40-sample time constant.

**4.2 It is not the encoder.** At depth 0 the encoder supplies `rms(x_a) = 1.091509e+00` in all three
arms. One step later it is `0` or `2.3e-09`.

**4.3 The wiring is as read.** `A_aa` computed from `fs.hfn` and from the ANN alone agree at exactly
`0.000e+00` in all three arms, so rows 6 and 7 are fed by the ANN and nothing else
(`model.py:188-195`). `rho(J_full)` is `1.0000` everywhere, which is the baseline's integrators.

**4.4 The cause is one shared zeroed output layer.** `zero_init_feed_forward_nn` zeroes the ANN's
final Linear. That layer produces both `f_aug` (rows 0-5, the correction into the physical states)
and `g_aug` (rows 6-7, the augmented state update). Zeroing it to satisfy D-072 baseline equality
destroys `g_aug` as a side effect. `dL/dW_out[6:8,:]` is exactly `0.000e+00`.

**4.5 Restoring `A_aa` removes the barrier.** `cl_latent_init_test.py` stage `t3`, artefact
`runs/cl_latent_init_test_t3.json`. From an initialisation with live `A_aa` and physical rows still
zero, the loss along the segment to the planted model falls monotonically `2.323070e-10` to
`5.793970e-11` at `t = 0.85`, peak equal to the `t = 0` endpoint, barrier ratio `1.00x`. At the
trained weights the same probe gave `34.83x`. **The barrier was a property of where training started,
not of the objective.**

**4.6 The readout gradient becomes informative.** Same script, stage `all`. Recomputing the gradient
with the ANN's `x_a` input columns blanked: `cos = +1.000000` (untrained), `+0.999951` (trained),
`+0.464718` (live `A_aa`), `+0.047906` (planted). The gradient norm barely moves; only its content
changes.

**4.7 `A_aa` cannot be set by choosing output weights.** Solving `W_out[6:8,:] = A_target pinv(B)` at
an averaged operating point asked for `rho 0.98 at 159 Hz` and delivered `rho 0.649 at 38.5 Hz`. A
tanh MLP's Jacobian is strongly state dependent. This is the measured argument for a linear bypass.

**4.8 The residual spectrum works, in simulation.** `cl_residual_spectrum.py` on 18 records recovers
`rho = 0.9856 at 158.20 Hz`, `zeta 0.0583`, from `u`, `y` and the baseline alone, against an injected
truth of `plant.FA = 150.0 Hz`, `plant.ZETA_A = 0.050`. All 54 dominant peaks inside a 14 Hz band. No
usable Y dependence (`R2 = 0.024`).

**4.9 The real Telica data cannot supply an identified resonance.** `telica_plant_frf.py`, with the
verified controller divided out and the loop required to measurably act
(`coherence >= 0.90 and abs(L) > 3`): the identifiable band is `<= 83 Hz` on LX1 and LX2 and
`<= 55 Hz` on LY, and **no plant resonance is supported in 10 to 8000 Hz**. `iter0` has no
feedforward, so the reference is the only external signal and it is a smooth point-to-point profile.
This is an excitation limit of the campaign, not an estimator defect.

**4.10 Hoekstra's method keeps the two functions separate and does not zero `g_aug`.**
`arXiv:2602.17297` Table 1 p3 (S-DP has both `f_aug` and `g_aug`), Eq. (29) p9 (pins the physical row
and the output row only), section 5.4.3 p9 ("only considering the linear component of the learning
function and initialising the NL component to be zero"), p10 (everything not pinned is `U(-1,1)`),
Eq. (31) p9 (the augmented encoder block is Xavier, not zero). PDF at
`scratchpad/seed3/2602.17297.pdf`, worth moving into `literature/augmentation/`.

**4.11 Dynamic augmentation demonstrably trains in this framework.** Hoekstra EJC 86:101304 (2025),
already on disk at `literature/augmentation/hoekstra2025_lfr-augmentation-ejc.pdf`, Table 4 at 60 dB:
dynamic parallel `0.00159` beats static parallel `0.00246` and black box `0.00373`, with
`nx_aug = 2`. The project's earlier claim that this has never been shown is false.

## 5. Assumed but not verified

* **That live `A_aa` is sufficient.** The barrier disappears, but `cos(-grad, w_planted - w_C)` is
  still `+0.0007`: the first descent direction does not point at the good solution. Whether descent
  curves round to it is untested and only a training run settles it.
* **That the objective can reward using `x_a`.** At `nf = 400` the loss ranks the planted model only
  `1.25x` above the trained one. `burn_in` was the attempt to fix this and failed.
* **That the augmented states survive noise.** A `zeta ~ 0.1` mode is `Q ~ 5` and amplifies whatever
  sits at its frequency, including measurement noise. Kessels et al. report extended states capturing
  measurement noise at `n_ext = 6`. Untested here.
* **That `nx_aug = 2` is right for the real system.** Kessels needed `n_ext = 14` on an ASMPT stage;
  `n_ext = {0, 2}` "lack sufficient complexity" there.
* **That arm C's no-barrier result transfers.** Arm C carries planted hidden layers as well as a live
  `A_aa`, so part of that result is the planted feature map.

## 6. Tried and failed

* **Optimiser work** to `eps 1e-16`, `lr 1e-5`, three optimisers, one basin within 1.1 % (D-148
  finding 3). A 34.83x barrier stops any local descent method regardless of order.
* **`burn_in = 100`** to `2.33e-06` full-window RMS, above the untrained `2.2331e-06`, free run 0.8 %
  worse, because nothing penalises `[0, K)` and under Adam the encoder keeps moving at full rate on
  an uninformative direction. Reverted yesterday.
* **State-consistency term at 10 % of the MSE** to indistinguishable from no term, because the weight
  was calibrated on loss values whose units differ by `1.7e9`. Reverted yesterday.
* **`ANN_INIT_SCALE = 3.2`** (scaling all ANN weights uniformly) cannot move `A_aa` off zero, because
  the output layer stays zero. The arm was killed at 86 %. Reverted.
* **Solving `W_out[6:8,:]` to realise a target `A_aa`** to `rho 0.649 at 38.5 Hz` against a target of
  `0.98 at 159 Hz`, see 4.7.
* **Telica peaks from tracking error and feedback current** to a near-vacuous criterion, because on
  `iter0` `i_fb = C(z) e` exactly, so a peak in the error reaches the current automatically.
* **Gating the Telica FRF on coherence alone** to roughly 200 fictitious modes per axis, because
  `e = r - q1` is built from `r`, so `S -> 1` and coherence `-> 1` trivially wherever the loop cannot
  follow. Fixed by additionally requiring `abs(L) > 3`.

## 7. Achieved

Implemented and validated: `cl_aug_spectrum.py` (the `rho` measurement, with a Jacobian cross-check
that passes at exactly zero), `cl_latent_init_test.py` (three probes), `cl_residual_spectrum.py`
(validated against a known injected mode), `telica_residual_spectrum.py` and `telica_plant_frf.py`.

Reverted with re-appliable patches, both verified by `git apply --check --reverse`:
`patches/2026-08-19-interconnect-burnin-consistency.patch` and
`gantry_dynamic/patches/2026-08-19-ann-init-scale.patch`.

Documented: the `ANN-learning-issue/` folder. **Not yet documented**: no D-149 entry in
`docs/decisions.md`, and the arm E run-table row still says `OUTCOME: pending`.

## 8. The open question

**Does live `A_aa` actually let training reach a materially better model, and does it survive noise?**

The barrier is gone (4.5) and the gradient carries `x_a` information (4.6), but the gradient still
does not point at the known-better model (`+0.0007`), and nothing has been trained from the new
initialisation. Only a training run answers it.

Two decisions are the user's, not yours. Ask, and end the turn, if you reach either:

1. **Weakening D-072.** With exact baseline equality, `dL/dA_aa` is zero at step 1 and unlocks at
   step 2, which is forced, not a design choice. The alternative is to initialise `W_out[0:6,:]` at a
   small `epsilon` so everything trains from step 1, at the cost of the model no longer being exactly
   the baseline at `t = 0`. Do not make that change on your own.
2. **The commit.** Blocked on what to do about another session's P1/P1-e work in
   `gantry_dynamic/{config,evaluation,orth_penalty}.py`.

## 9. Next action

**Split `f_aug` from `g_aug` in the ANN's parameterisation and train one arm.**

Concretely:

1. Add a linear bypass on the augmented rows only, so that
   `x_a,k+1 = A_aa x_a,k + NL(x, u)` with the nonlinear part zero at initialisation and rows 0-5 of
   the ANN output still exactly zero. `A_aa` is a trainable parameter that is initialised, not a
   frozen constant. `model_augmentation/utils/torch_nets.py:121` has a `zero_init_resnet` whose
   bypass is zeroed at `torch_nets.py:142-143`; it is close to what is needed but not a drop-in.
2. Initialise `A_aa` from a **band**, not from a mode: eigenvalues `r_i exp(+- j theta_i)` with
   `r_i` drawn over an annulus and `theta_i` over the band the augmentation must cover, in the stable
   exponential parameterisation `lambda = exp(-exp(nu) + j exp(theta))` with the input normalisation
   `gamma = sqrt(1 - abs(lambda)^2)` (Orvieto et al., ICML 2023, PDF in `literature/deep-ssm-init/`).
   On the simulated data the band should contain `158 Hz`; verify with `cl_residual_spectrum.py`
   rather than hard-coding, and do not hard-code `0.9856` or `159 Hz` anywhere.
3. Train one arm: `lr = 1e-5` (not `1e-7`; the required readout growth is about `6.4e-04` and
   `lr 1e-7` over 780 updates travels only `7.8e-05`), `eps = 1e-16`, `burn_in` off, no consistency
   term, `W^a` at its default random init, at least 10 epochs. Write the run-table row before launch
   per the run-discipline rule.
4. Report the 12 s free run against `1.3933793e-06 m`, and `rho(A_aa)` after training from
   `cl_aug_spectrum.py`.

Launch pattern and paths are in section 13.

## 10. Acceptance criterion

**Primary.** Free-run sim-RMS on the four validation records below **`1.215e-06 m`**, which is 45 %
of the available headroom against training's current 36.7 %, with the untrained reference at
`2.1866011e-06` and the data-derived floor at `2.81e-08`. All three come from the same production
scorer, `closed_loop_free_run_rms`. Runs reproduce to six significant figures across machines, so any
difference above about 0.5 % is real.

**Secondary, and these are the ones that say whether the mechanism worked rather than whether the
number moved:** `rho(A_aa)` still above `0.5` after training rather than collapsing back toward zero,
and `Wa_psi_y` moving off `0 of 108` entries.

**Noise gate, and it is part of done, not a follow-up.** Repeat the arm with measurement noise added
to the simulated `y` at the Telica SNR. If `rho(A_aa)` collapses or the free run loses its gain, the
result is simulation-only and must be reported as such. `HYPOTHESES-AND-SOLUTIONS.md` section 4e has
the four steps.

## 11. Read these first

1. `scripts/gantry/closed-loop-controller/ANN-learning-issue/RESULTS.md`, every number you need.
2. `scripts/gantry/closed-loop-controller/ANN-learning-issue/HYPOTHESES-AND-SOLUTIONS.md`,
   sections 3 and 4, the fix and why it must be band-based.
3. `scripts/gantry/gantry_dynamic/model.py:143-200`, the ANN construction and routing you are
   changing.
4. `model_augmentation/utils/torch_nets.py:97-146`, `zero_init_feed_forward_nn` and
   `zero_init_resnet`.
5. `docs/decisions.md` D-147 and D-148, the two sessions of evidence behind the plateau.

## 12. Do not

* Do not execute section 9 of `tasks/handoffs/2026-08-19-objective-not-optimiser.md`. It is
  superseded.
* Do not hard-code `159 Hz`, `158.20 Hz` or `rho = 0.9856`. Those are simulation numbers and the real
  machine has no such mode (4.9). Any constant that comes from the planted model is oracle
  information built from `x_aug`.
* Do not use `x_aug` as a training signal anywhere, and do not use the planted weights as an
  initialisation.
* Do not reinstate `burn_in` or the consistency term, and do not try another optimiser.
* Do not revert `closed_loop.py` or `pre_encoder.py`.
* Do not modify `kamtin-fp-model/`, or `gantry_dynamic/{config,evaluation,orth_penalty}.py`.
* Do not read files under `kamtin-data/Data Telica/` directly. Scripts may load them; the raw
  contents must not be printed. `docs/kamtin-telica-schema.md` has the column and unit reference.
* Do not commit without resolving section 8 item 2 with the user.

## 13. Operational

Environment `GraduationProject`. Training runs stream live and must not block:

```
cd scripts/gantry/closed-loop-controller
CL_EPOCHS=10 CL_LR=1e-5 CL_ADAM_EPS=1e-16 CL_STRIDE=10 CL_ITS_PER_VAL=epoch \
CL_PROBE=1 CL_FLOOR=0 CL_BURNIN=0 CL_CONS_FRAC=0 CL_TAG=<tag> \
PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output \
  -n GraduationProject python -u cl_train.py
```

About 4 s per iteration locally, 260 iterations per epoch. Results land in
`runs/cl_train_<tag>.json`. Do not pipe a running job through `grep`: the pipe buffers and nothing
appears until it exits. Three background jobs were killed unexplained yesterday, so prefer a
foreground run or check the output file directly.

Diagnostics, all seconds to a minute: `cl_aug_spectrum.py` (`rho`), `cl_residual_spectrum.py`
(`CL_RS_FILES=all` for 18 records), `cl_latent_init_test.py` (`CL_LIT_STAGE=t3` or `t3rand` to skip
the expensive stages).

**Latent defect to guard before your first training run.** `cl_train.py:257` and `279-280` still set
`fs.burn_in`, `fs.consistency_points` and `fs.consistency_weight`, which the reverted framework no
longer reads. A run launched with `CL_BURNIN=100` will print `objective: burn_in 100 ...` and train
on the full window anyway. Add an assertion that refuses to start in that case.

## 14. Delegation

None for the implementation and the training run: both are targeted and one context should hold them.

One Explore subagent is warranted if you need to find where in the interconnect the augmented rows
could carry a linear term, since that spans `model_augmentation/fit_systems/` and
`scripts/gantry/gantry_dynamic/`. Cap at one.

Do not spawn subagents to verify your own work. If you want a literature question answered, the
`deep-research` skill is the required route per D-121, one subagent per seed question, and three
seeds were already run yesterday (lazy versus rich initialisation, latent-state initialisation, and
encoder co-estimation with augmented states); their results are in `RESULTS.md` section 6, so do not
repeat them.
