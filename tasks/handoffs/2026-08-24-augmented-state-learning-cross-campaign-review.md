# Handoff: reconcile three campaigns into one account of why the augmented states do not learn
**From**: session of 2026-08-24 | **Branch**: Augmentation | **Effort suggested**: xhigh

## 1. Task

Three separate campaigns have produced augmented-state results that have never been compared in one
place: `scripts/gantry/closed-loop-controller/` (arms 1 and 2, `AUG_LRU` via `cl_train.py`),
`scripts/gantry/Augmentation-with-BLA/` (the BLA initialisation campaign, `independent-init-a/b`
and `combined-init`), and `scripts/gantry/augmented-states/` (arms i/ii/iii, SLURM 78465, plus the
J1/GRU/Kessels smoke tests). Produce a single reconciled account of what is actually preventing the
augmented states from learning, backed by measurements from all three folders and by literature,
and end with a written specification for a clean implementation. Deliverables, in order: (a) one
cross-campaign results table in which every arm carries the same columns and every cell is either a
measured number with its artefact path or an explicit "not measured"; (b) a ranked list of
candidate causes, each labelled with which measurement supports or kills it and each marked
verified or assumed; (c) the clean-implementation specification. Write all three into
`scripts/gantry/augmented-states/DISCUSSION-POINTS.md` as new lettered sections; do not create new
documents.

### 1.1 The question to answer

Do **not** begin from the premise that augmented states cannot learn.  The repository contains one
strong positive result (historical arm 2, `3.795974e-07 m`, `5.21x` ablation), several dead or weak
results, and one numerically excellent result whose recurrence is effectively fixed (arm iii).  The
review must distinguish these four questions, which are currently mixed together:

1. **Liveness:** do non-zero augmented trajectories exist during rollout?
2. **Load bearing:** does removing `x_a` materially worsen the trained model?
3. **Writer learning:** did the parameters that generate the augmented recurrence move and improve
   performance beyond the identical frozen recurrence?
4. **Plant accuracy:** does the resulting model improve the agreed free-run plant metric on held-out
   records?

A positive answer to one is not evidence for the next.  In particular, a large `x_a` ablation does
not prove that the recurrence was learned, and a low RMS does not prove that the augmented states
were used.

The final diagnosis should answer, in one paragraph:

> Which measured difference best explains why historical arm 2 learned a load-bearing augmented
> path while the 36 % model, Init-A/Init-B controls, arms i/ii, and the corrected Kessels writer did
> not show the same combination of learned recurrence and accuracy?  If the evidence cannot select
> one cause, which two or three causes remain observationally equivalent, and what is the cheapest
> experiment that separates them?

### 1.2 Evidence protocol

Use this precedence order whenever sources disagree:

1. the run's own `run_summary.json` or numeric JSON artefact;
2. the exact checkpoint measured by a diagnostic, with checkpoint provenance recorded;
3. the run's `.out` launch log and printed configuration;
4. a contemporaneous results document;
5. a later synthesis (`README.md`, `CURRENT-RESULTS.md`, `DISCUSSION-POINTS.md`).

Never fill a cell from a sibling seed or nominally similar arm.  Never infer Adam `eps`, stride,
record set, metric, checkpoint selection, or update count from a shared configuration unless the
actual run artefact proves that it used it.  Use `not measured` or `not recoverable` instead.

Before comparing two RMS values, prove that they use the same:

- plant data and validation records;
- closed-loop versus open-loop rollout;
- one-shot free run versus reset `nf` windows;
- channel pooling and physical units;
- encoder/controller-state initialization;
- checkpoint-selection rule.

If any item differs, report both numbers but label the ranking **not comparable**.  This rule is
load-bearing for historical arm 2 versus arm iii and for both versus the planted-correct model.

### 1.3 Required cross-campaign table

Use one row per arm **and seed** where seed-level artefacts exist.  Required columns:

| category | required fields |
|-|-|
| provenance | campaign, arm, seed, code path, checkpoint path, result path |
| data/score | train records, validation records, rollout type, metric definition, initial RMS, best RMS |
| model | `nx_aug`, number/type of recurrence blocks, pole source, ANN parameter count and routing |
| liveness/memory | `B_u` or other drive, recurrence initialization, encoder `W^a`, measured `x_a` RMS, retention/poles |
| optimization | `nf`, stride, batch size, epochs, actual updates, learning rate, Adam `eps`, dtype |
| attribution | augmented-state ablation, `ANN/drive`, writer/pole movement, learned-versus-frozen control |
| status | positive/negative/inconclusive, exact reason, known confounds |

Include at minimum: untrained reference, Go1qTA, historical arms 1 and 2, Arm F, every recoverable
Init-A/Init-B control family, arms i/ii/iii for all seeds, J0/J1, corrected Kessels learned/frozen,
planted-correct, and oracle reference.  Smoke tests may be placed in a separate block because their
fixed-window metric is not comparable to complete-record validation.

### 1.4 Causal audit

For every candidate cause, provide this compact falsification record:

| cause | mechanism | supporting measurements | contradicting measurements | confounded arms | status | decisive next test |
|-|-|-|-|-|-|-|

Audit at least: zero-state topology, missing drive, insufficient/incorrect memory, basis span and
`nx_aug`, Adam epsilon, per-row gradient/output scaling, ANN routing, stride/update budget, encoder
initialization, startup-transient objective dominance, closed-loop sensitivity/excitation, dense
recurrent credit assignment, reservoir adequacy for an LTI residual, and metric/checkpoint-selection
mismatch.

Use `verified cause`, `verified defect but not outcome-limiting`, `supported hypothesis`,
`contradicted`, or `unresolved`; do not use an unqualified `the cause` unless one controlled
comparison isolates it.

### 1.5 Clean-implementation specification

The specification must be an implementable contract, not a menu of J1/GRU/Kessels ideas.  State:

- the exact state equations and which terms are fixed, learned, or initialized;
- how the augmented states are driven at sample one without `B_u`, if `B_u` is removed;
- how long memory is represented and kept stable;
- the augmented-state coordinate/scaling convention;
- how the encoder initializes `x_a[0]`;
- the only permitted route from `x_a` to the loss (no direct learned `x_a -> h` unless the review
  explicitly overturns that requirement);
- whether the physical correction can bypass `x_a`, and how attribution remains identifiable;
- optimizer settings, including `eps`, per-row scaling, horizon, stride and budget;
- initialization equality and the numerical gate used to verify it;
- the matched controls and ablations required before a long run;
- closed-loop training and open-loop plant-validation metrics.

Where literature does not determine a choice, label it a project hypothesis and attach a
falsification experiment.  The final recommendation may be `reproduce historical arm 2 first` if
the table shows that its mechanism or metric is not yet understood.

## 2. Out of scope

- **Do not implement anything.** No new architecture, no writer replacement, no `q/v` cell, no
  orthogonality code. The deliverable is a specification, not a build. User's decision.
- **Do not launch a training run.** No SLURM array, no 11-hour arm, no retrain. Diagnostics on
  existing checkpoints are allowed and encouraged.
- **Do not run the D1 width sweep or the D5 count sweep** (`DISCUSSION-POINTS.md` section D). Six
  runs at ~11 h; blocked on this review, not the other way round.
- **Do not modify the production training framework**: `model_augmentation/fit_systems/interconnect.py`,
  `scripts/gantry/gantry_dynamic/config.py` (frozen, carries another session's uncommitted work), and
  do not re-apply `closed-loop-controller/patches/2026-08-19-interconnect-burnin-consistency.patch`.
- `kamtin-fp-model/` is read-only, per the hard constraint.
- **Do not re-derive the LPV baseline, the controller, or the pole identification.** Settled.

## 3. Where things stand

Branch `Augmentation`, last commit `a0e3f76`. Tree is dirty across `scripts/gantry/`, `docs/`,
`tasks/`, `Matlab-scripts/`, and several `slprj/` binaries; none of that dirt is from this session
except the handoff you are reading. No run is in flight.

This session wrote nothing into the repo except this file. Two scratch artefacts exist and are
worth moving in if the successor finds them useful (they are not required to start):
`measure_xa0_ablation.py` and `xa0_ablation.json` in the session scratchpad, described in section 7.

All nine SLURM 78465 checkpoints are local under
`scripts/gantry/augmented-states/meeting-23-08/server-results/checkpoints/`. `iii_43` is
`SSE_Interconnect_MultipleShooting_eLepPw_best.pth`; the 36 % model is
`FitSys_ClosedLoop_Go1qTA_best.pth`.

## 4. Established and verified

Each item is measured or read this session, or read out of a document with its pointer.

**The optimizer defect, and which arms it contaminates.**
- Fresh `torch.optim.Adam(...)` with no `eps` argument is constructed in
  `scripts/gantry/augmented-states/kessels_writer_smoke.py` and `j1_gantry_smoke.py`, so both ran at
  the torch default `1e-8`. `run_augmented.py` reaches `train_model` in
  `scripts/gantry/gantry_dynamic/model.py`, which passes only `lr` through `optimizer_kwargs` (grep
  `optimizer_kwargs` there), so **arms i/ii/iii also trained at `eps = 1e-8`**.
  `closed-loop-controller/cl_train.py` exposes `CL_ADAM_EPS` (grep it) and its runners
  (`runners/run_cl_arms.sh`, `runners/run_ablation_wave1.sh`) set `1e-16`.
- With writer gradient norm `1.945e-13` over 220 parameters at `lr = 1e-5`, a 49-step Adam
  simulation predicts writer movement `9.53e-9` at `eps = 1e-8` and `7.21e-3` at `eps = 1e-16`.
  Measured at `eps = 1e-8`: `1.216e-8`. The epsilon floor accounts for the observed non-movement.
- The user's `eps = 1e-16` re-run moved the writer `5.56e-3`, a factor `457,000`. **Confirmed.**
  Learned writer then reduced loss `0.506 %` against the frozen writer's `0.486 %`.

**The `x_a[0]` ablation on `iii_43`** (this session, `xa0_ablation.json`; gate passed at relative
deviation `5.99e-05` against the recorded `5.259e-07`):
- free run V1, full record: intact `5.259315e-07`, `x_a[0] = 0` `5.244482e-07`, ratio `0.9972`.
- `nf = 400` windows on `T3_standstill_Y000`, 119 windows: intact `8.312434e-07`,
  `x_a[0] = 0` `7.303964e-07`, ratio `0.8787`. **Zeroing the encoder's augmented initial condition
  removes `22.79 %` of window residual energy.**
- `99.9 %` of the removed energy sits in `k < 150`; per-`k` energy ratio is `0.69` over `k < 150`
  and `0.999` over `k >= 150`; `k = 0` ratio is exactly `1.000`. The damage tracks `rho^k` and stops
  where the resonator forgets.
- The production configuration has a large startup transient: peak `2.5851e-06` at `k = 10`, floor
  `3.9067e-07` at `k = 108`, peak/floor `6.62`, `73.9 %` of window energy in the first `37.5 %` of
  the window.

**The Kessels smoke model** (user's diagnostics, `runs/kessels_writer_initial_state_diagnostics.json`):
- residual is flat: `1.8763e-06` over `k = 0..99` against `1.9022e-06` over `k = 100..399`;
  `24.49 %` of energy in the first `25 %`. **No startup transient at all in that harness.**
- latent-state retention `85.7 %` at `k = 76`, but output sensitivity `29.3 %` at 76 and `4.39 %`
  at 400, with `97.8 %` of squared output sensitivity in `k < 100`.
- `runs/kessels_writer_smoke.json`: `q_rms = 5.806e-04` against `v_rms = 0.2911`, ratio `501`.
  Memory lives in `q` (pure integrator) and influence lives in `v`. They are different channels.
- `coupling_ablated_loss` equals `initial_loss` to all 17 digits in both arms, so
  `latent_ablation_penalty_fraction_of_improvement = 1.0` is a tautology of that design, not evidence.

**What a `q/v` writer must learn** (computed this session). With `q+ = q + alpha*v`,
`v+ = a*q + b*v + drive`, matching `lambda^2 - (1+b)lambda + (b - alpha*a)` to the plant mode
(`158.1139 Hz`, `zeta = 0.05`, `Ts = 1/4000`, `rho = 0.987643`, `tau = 80.4` samples):
- `b = 2*rho*cos(theta) - 1 = 0.914676`, independent of `alpha`; `alpha*a = b - rho^2 = -0.060763`.
- `rho^2` is the only combination that sets memory. `d tau / d(rho^2) = 3316` samples per unit.
- `tau` within `+20 %` needs `rho^2` within `+0.415 %`; within `-20 %` needs `-0.620 %`;
  `rho^2 = 1` (unstable) is `+2.518 %` away.
- By contrast `A_aa`'s `rho = exp(-exp(nu_log))` gives `d tau / d nu_log = -tau` exactly, so
  relative memory error equals absolute parameter error and `rho >= 1` is unreachable.

**Cross-campaign results already on record** (read, not re-run):
| arm | `nx_aug` | free-run RMS | ablation | source |
|-|-|-|-|-|
| untrained reference | any | `2.1866011e-06` | - | every run |
| 36 % model (`Go1qTA`) | 2 | `1.3933805e-06` | `1.00000x` both surfaces | `meeting-23-08/01-what-training-bought.md` |
| arm 1 | 2 | `1.379891e-06` | `1.0183x` | `augmented-states/README.md`, `CURRENT-RESULTS.md` |
| **arm 2** | 8 | **`3.795974e-07`** | **`5.21x`** | same |
| Arm F (`AUG_LRU`) | 2 | `1.3841e-06` | poles moved `4e-5` | `docs/aug-lru-implementation.md` s7, s9 |
| arm iii (2 of 3 seeds) | 8 | `5.78e-07` / `6.46e-07` | `F > 1/2` | `DISCUSSION-POINTS.md` s0 |
| planted-correct ANN | 2 | `4.177e-07` | - | `RESULT-plateau.md` |
| oracle floor | - | `2.81e-08` | - | `RESULT-plateau.md` |

**Arm iii's augmented dynamics are linear in practice**: `ANN/drive = 0.001524` on a full 47983-step
V1 rollout of `iii_43`, reconstruction error at the float32 floor (`DISCUSSION-POINTS.md` B4 RESULT).

**The objective barely ranks a correct model**: at `nf = 400` the training loss ranks a
planted-correct augmentation only `1.25x` better than what training finds, because `88 %` of a
correct model's window loss is startup transient; burn-in raises the ratio to `3.40x`
(`closed-loop-controller/RESULT-plateau.md`). This is H2.

**The pole gradient points the wrong way**: with the true mode planted, `dL/d(nu_log) < 0` on 7 of 8
batches, monotone over 150 steps (C6); no non-negative reweighting flips that sign (T3). This is a
sign result and is therefore **independent of the epsilon defect**
(`DISCUSSION-POINTS.md`, "Two distinct failures").

**Kessels (`literature/augmentation/kessels2025_ai-control.pdf`, Ch. 5), read this session**:
Eq. (5.4) p151 augments the velocity equation and leaves the position integration and `h` untouched;
Eq. (5.13a) p156 shows the encoder **does** initialise the extended states `x^(2,p)` and `x^(2,v)`
through identity blocks; p159 states the NNs are initialised **close to zero, not exactly zero**;
Remark 5.3 p157 drops the output augmentation when `h` is trusted; Remark 5.6 p174 records that
open-loop identification of the wire bonder was attempted and **failed**, motivating closed-loop
training; Table 5.5 p173 shows `n_ext = 0` (augmentation only) already worth `10x`, and training runs
233 to 498 epochs over 5 to 8 hours.

**Literature located this session** (details and BibTeX-able metadata in section 11 pointers):
- Everett et al., "Scaling Exponents Across Parameterizations and Optimizers", ICML 2024,
  arXiv:2407.05872, Sec. 4.3 "Epsilon Underflow in Adaptive Optimizers": epsilon breaks Adam's scale
  invariance once the gradient scale drops below it; remedies are `eps` `1e-12`/`1e-15`, per-layer
  epsilon, or `Adam-atan2` which removes `eps` entirely. **Read in full.**
- Jere, Zheng, Said, Liu, IEEE J. Sel. Topics Signal Processing 2024, DOI `10.1109/JSTSP.2024.3387274`,
  arXiv:2308.02464: reservoir computing (random untrained recurrent weights) universally approximates
  a general **LTI** system, and the paper analytically characterises the optimal density for
  **configuring** rather than training those weights. **Abstract only.**
- Zucchet & Orvieto, NeurIPS 2024, arXiv:2405.21064, **already on disk** at
  `literature/stability-training/lazy-rich/2405.21064.pdf`: the "curse of memory"; long memory makes
  the loss landscape hypersensitive even without exploding gradients; dense recurrences defeat
  adaptive optimizers, complex-diagonal connectivity with normalisation and exponential
  reparametrisation fixes it. **Grep-read, Secs. 2-4.**
- `literature/stability-training/claude-deep-research-Adam-optimizer-drift.md` already held the
  Cattaneo/Klusowski/Shigida ICML 2024 flow `theta_dot = -grad/sqrt(|grad|^2 + eps)`, which is the
  continuous-time form of the same mechanism, filed under "drift" and never quoted for this.

## 5. Assumed but not verified

- **Whether arms 1, 2 and F ran at `eps = 1e-16`.** They went through `cl_train.py`, whose runners
  set it, and `docs/aug-lru-implementation.md` s4 says `nu_log`/`theta_log` receive the override. Not
  confirmed against a launch line. Settle by grepping `CL_ADAM_EPS` in their `.out` files under
  `closed-loop-controller/server-results/`. This decides whether their nulls are clean or confounded.
- **Whether arm 2's `3.795974e-07` and arm iii's `5.78e-07` are comparable.** Different harness,
  possibly different budget, stride, records and scoring pass. Settle by tabulating both from their
  own run JSONs before ranking them.
- **Whether the `x_a[0]` result generalises.** One record (`T3`, standstill), one seed, one arm, one
  checkpoint, and post-hoc on a model trained *with* that initial condition. Settle by repeating
  `measure_xa0_ablation.py` across seeds 42/44, arm ii, and one APRBS record.
- **Whether T3's swept reweighting class includes per-sample burn-in weights.** If it does, the
  burn-in outcome is already predicted on record. Settle by reading the T3 script.
- **Whether Jere et al.'s LTI class and fading-memory assumptions cover a `zeta = 0.05` resonance.**
  Abstract only. This is load-bearing for the "training the recurrence is unnecessary" reading.
- **Whether the Kessels smoke test's flat residual is caused by the frozen 36 % encoder.** The
  mechanism is plausible (physical `x_b[0]` is good because that encoder was trained) but unmeasured.
- That `Micikevicius` mixed-precision loss scaling unscales before the optimizer step, hence does not
  fix an epsilon floor. Standard description of the method, not verified against the PDF.

## 6. Tried and failed

- **Jan's default zero-init on all ANN output rows** -> augmented route carries nothing:
  `1.00000x` on both ablation surfaces, on 14 trajectories individually -> `x_a` is useless until
  non-zero and only the zero-output ANN can make it non-zero; the bootstrap gradient is `1.34e-10`
  -> `meeting-23-08/01-what-training-bought.md`.
- **Row-selective J1 (physical rows zero, augmented rows Xavier)** -> liveness restored and `x_a`
  load-bearing (10.1 % ablation penalty across 8 windows against 0.85 % for J0) but only 0.7 % raw
  fit advantage, and exact lag-76 state retention `3.31e-4` -> a dense contracting tanh recurrence
  forgets long before the mode does -> `runs/j1_gantry_smoke_8windows_50step.json`,
  `runs/j1_gantry_lag_probe_exact_dense_seed42.json`.
- **`A_aa + B_u` (arms i/ii/iii)** -> reaches `5.78e-07` but `ANN/drive = 0.001524` and poles move
  under `0.5 Hz` in 1044 updates -> the correction network was routed to those rows, was free to
  write them, and declined; the realised model is a fixed linear filter bank with a nonlinear
  readout -> `DISCUSSION-POINTS.md` B4 RESULT. **Caveat: trained at `eps = 1e-8`, so the "declined"
  reading is confounded** (section 4).
- **`AUG_LRU` Arm F** -> `1.3841e-06` against a `1.215e-06` target; `rho(A_aa)` `0.9920 -> 0.9920`,
  `f` `154.52 -> 154.56 Hz` -> the parameterisation change removed the initialisation obstruction
  (encoder `W^a` moved 108/108 entries against 0/108 before) but training neither collapsed nor
  exploited the dynamics; attributed to H2 -> `docs/aug-lru-implementation.md` s1, s7, s9.
- **Kessels-style `q/v` writer, `eps = 1e-8`** -> writer moved `1.216e-8`, learned and frozen arms
  identical -> Adam epsilon floor, quantified in section 4 -> `runs/kessels_writer_smoke.json`.
- **Same, `eps = 1e-16`** -> writer moved `5.56e-3` but learned `0.506 %` against frozen `0.486 %`
  -> the writer is now optimizable and still adds almost nothing over a frozen random reservoir.
  **This is the current unexplained result.**
- **Burn-in `k0 = 100` on the Kessels smoke loss** -> no material change in the learned-versus-frozen
  gap -> that harness has no startup transient to remove (residual flat, section 4), so the test
  measured a condition that was not present. It did **not** refute transient dominance for the
  production runs, where peak/floor is `6.62`.
- **Optimiser sweep in the closed-loop line** -> three optimizers within 1.1 % of the same loss, and
  `eps 1e-16` versus `1e-8` differed by 1 % -> measured on **total** loss, which is dominated by the
  physical head; it does not test whether the epsilon floor blocks the augmented writer, whose share
  is ~0.005 % -> `RESULT-plateau.md`.

## 7. Achieved

- **Adam epsilon identified, quantified and confirmed as the cause of writer non-movement.**
  Prediction `~6e5x`, measured `457,000x`. Closed.
- **A third ablation surface built and run**: `x_a[0]` alone, leaving the recurrence, drive and
  correction network intact. Distinct from `ABLATE_XA_TO_F` and `ABLATE_XA_UPDATE`, which
  `DISCUSSION-POINTS.md` E1 correctly identifies as measuring the same thing. Implemented and
  validated (gate `5.99e-05`). Script and JSON in the session scratchpad, named in section 13.
- **The encoder's augmented initial condition measured to be a net liability** on the training-window
  surface for `iii_43`, worth `22.79 %` of window residual energy, at zero cost on the validation
  metric. Implemented and validated; **not** replicated.
- **The `q/v` writer's target coefficients and their conditioning derived**, giving the first
  quantitative statement of what the Kessels writer must learn and how precisely.
- **Two literature findings that name the phenomena**: epsilon underflow (Everett) and reservoir
  adequacy for LTI targets (Jere). Located, one read in full.

## 8. The open question

**Why does a live, correctly-optimised latent writer add almost nothing over the identical frozen
random one (`0.506 %` against `0.486 %`)?** Candidates, with the evidence that would choose:

1. **Reservoir adequacy.** If the missing dynamics are LTI, a configured random recurrence is
   provably sufficient (Jere et al.), so training it *should* add little and the tie is correct.
   Supported by `ANN/drive = 0.001524`. Chosen by: reading Jere et al. in full and checking its
   assumptions against `zeta = 0.05`; and by whether arm 2's `5.21x` ablation is explicable as a
   basis-span effect rather than learned dynamics.
2. **The `q/v` scaling defect.** `q_rms/v_rms = 501`: the channel with memory carries 0.2 % of the
   energy and is unread, so the smoke test is not a test of Kessels' mechanism at all. Chosen by:
   re-running with `alpha ~ 1/76` instead of `Ts`.
3. **H2 weak discrimination.** The objective ranks a correct augmentation `1.25x`. Applies to the
   production runs (peak/floor `6.62`) and demonstrably **not** to the smoke harness (flat residual).
   Chosen by: measuring discrimination on whichever harness the next arm uses, before training.
4. **Writer parameterisation.** Zucchet & Orvieto: dense recurrences defeat adaptive optimizers
   through the curse of memory. Weakened by Arm F, whose LRU parameterisation is their prescribed
   remedy and whose poles still did not move. Chosen by: confirming Arm F's `eps` (section 5).
5. **Budget.** 50 updates on 8 windows against Kessels' 233 to 498 epochs.

A better task may exist and the successor should say so in one sentence rather than switch to it:
the cross-campaign table may show arm 2 (`3.795974e-07`, `5.21x`) is the project's real result and
that arms i/ii/iii were a detour, in which case the question becomes why arm 2 worked.

## 9. Next action

**Build the cross-campaign results table first, before any further diagnosis.** Every arm from all
three folders in one table with identical columns: harness, `nx_aug`, pole source and count,
ANN parameter count, updates, stride, `nf`, Adam `eps`, records scored, untrained reference,
trained free-run RMS, ablation `F`, `ANN/drive` where measured, artefact path. Fill each cell from
the run's own JSON or `.out`; write "not measured" where it is absent rather than inferring. Include
the `CL_ADAM_EPS` grep from section 5, because it decides which nulls are clean.

Rationale: the project currently cannot rank its own results. Arm 2 at `3.795974e-07` with a `5.21x`
ablation sits in a different folder from arm iii at `5.78e-07`, and every causal claim in
`DISCUSSION-POINTS.md` is written as though arms i/ii/iii are the frontier. Until the table exists,
every candidate in section 8 is being weighed against an incomplete record.

## 10. Acceptance criterion

Two, both required.

**Numeric.** The table names the project's best measured free-run pooled RMS in metres with its
artefact path and its Adam `eps`, and states its distance to the two references already on record:
the planted-correct model at `4.177e-07` and the oracle floor at `2.81e-08`
(`RESULT-plateau.md`). Per the Control Engineering Stance these are references for context, not
acceptance thresholds; the acceptance threshold for any future arm must be derived from data, and if
the successor proposes one it must say from which measurement.

**Completeness.** Every cell is a measured number with a path or the string "not measured", and
every arm carries its Adam `eps`. A cell inferred from a sibling run fails this criterion.

## 11. Read these first

1. `scripts/gantry/augmented-states/DISCUSSION-POINTS.md` - the current account, its own open list,
   and the two-distinct-failures framing. Longest but load-bearing.
2. `scripts/gantry/augmented-states/README.md` - arms 1 and 2, including the `3.795974e-07` and the
   `5.21x`, and section 5's difference list against the BLA campaign. This is the folder the current
   account under-weights.
3. `scripts/gantry/closed-loop-controller/RESULT-plateau.md` - H2, the `1.25x` discrimination, the
   `88 %` transient, the planted and oracle numbers.
4. `scripts/gantry/closed-loop-controller/ANN-learning-issue/RESULTS.md` - the lazy/rich and
   initialisation-theory work, including the already-run arXiv novelty counts. Section 6 there
   records queries that returned zero, so do not repeat them.
5. `docs/aug-lru-implementation.md` - what `AUG_LRU` is, Arm F's numbers, and its section 11 clean
   implementation target, which the specification deliverable should extend rather than replace.

## 12. Do not

- Do not repeat the arXiv queries recorded as zero in `ANN-learning-issue/RESULTS.md` section 6
  (`"lazy training" AND "physics-informed"`, `"lazy" AND "rich" AND "system identification"`,
  `"rich regime" AND "latent dynamics"`, `"linear recurrent unit" AND "system identification"`).
  Search the **reservoir computing** and **washout / echo state property** vocabularies instead;
  they are unsearched and they are where the Jere finding came from.
- Do not treat `ABLATE_XA_TO_F` and `ABLATE_XA_UPDATE` as independent evidence; E1 measured them
  agreeing to three decimals in all seven completed runs.
- Do not treat `latent_ablation_penalty_fraction_of_improvement = 1.0` from the Kessels smoke runs
  as evidence; it is a tautology of that design.
- Do not cite `F = 0.93` for the planted-physics probe. G1 records that it does not reproduce
  (D-157's own formula gives `1.183`), and the three documents carrying it are unedited.
- Do not quote Beintema, Toth and Schoukens (2021) Eq. (3) for the `k0` transient index; the claim
  that SUBNET separates encoder `x_0` from a transient-exclusion index is right, the equation number
  is unverified.
- Do not fix the epsilon defect in `gantry_dynamic/model.py`; that is an implementation change and
  section 2 excludes it. Record it in the specification.

## 13. Operational

Env: `conda run -n GraduationProject python ...`. Per the live-output convention, launch anything
over a few seconds with `PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output -n
GraduationProject python -u <script>` in the background and read the `.output` file.

The `x_a[0]` probe from this session, if you want to replicate across seeds and records:

```
measure_xa0_ablation.py --arm iii --seed 43 --n-pairs 4 --ckpt <checkpoint>
```

It currently lives in the session scratchpad
(`.../3df9f191-af7c-44ec-a588-440e8f5de8e3/scratchpad/measure_xa0_ablation.py`, with
`xa0_ablation.json` beside it). ~630 s per arm pair: two full V1 free runs plus two 119-window
passes. If it is to be kept, its home is `scripts/gantry/augmented-states/` with its JSON in
`runs/`, matching `measure_term_split.py`. Scratchpads do not survive; copy it before relying on it.

`measure_term_split.py --arm iii --seed 43 --ckpt <file>` gives `ANN/drive` in minutes on any local
checkpoint and is the instrument for filling that table column.

Checkpoints for every arm are under
`scripts/gantry/augmented-states/meeting-23-08/server-results/checkpoints/`; the mapping from run to
file is in `run_summary.json` and in `DISCUSSION-POINTS.md` B4's checkpoint-provenance note.

## 14. Delegation

One Explore subagent, for the run-inventory sweep only: locate every run JSON, `.out` and
`run_summary.json` across the three campaign folders and return their paths with the arm each
belongs to. That is a genuinely wide search across unknown naming conventions and is what the
subagent default exists for.

Do the reading, the reconciliation and the causal argument inline. Do not spawn a subagent per
folder, do not spawn one to check the table, and do not use one for the literature step: the
`deep-research` skill is the route there, and per D-121 it must be invoked rather than replaced with
ad-hoc `WebSearch`. One deep-research run, framed on the reservoir/washout vocabularies named in
section 12.
