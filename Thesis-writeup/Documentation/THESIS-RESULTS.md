# Thesis results: what is run and what it answers

Created 2026-10-08 (user request). Scope of the thesis results section after the research question change (D-241).
Replaces the run plan of `RESULTS-DESIGN.md` sections 4 and 8 and `TRAINING-DESIGN.md` section 4; their setup items
(data, noise, metrics, training settings) still hold where this file does not override them. Open points are marked (?).

## 1. Research question and aspects (D-241)

> For the dual-gantry system, how should a physics-based plant model be augmented to capture unmodelled dynamics,
> improving prediction accuracy across operating points and motion profiles while preserving the physical
> interpretability of its parameters?

1. In what form should the baseline be represented for subsequent augmentation, and does a position-dependent (LPV)
   formulation improve prediction compared to position-independent (LTI) alternatives?
2. What augmentation structure can effectively capture dynamics not modeled in the physics-based baseline model?
3. How can parameter interpretability be maintained during data-driven augmentation, compared to unconstrained
   approaches?
4. How well does the augmented model generalize to (i) held-out operating points and (ii) unseen motion profiles, and
   how does its prediction accuracy compare to a black-box state-space model?

Reading: Aspect 2 "structure" = parallel augmentation with or without added states (parallel vs series is argued in
Section III, not run); "effectively" is answered by PS2. Aspect 3 "unconstrained" = the U arm (no OBC, no penalty).
Aspect 4 includes controller transfer as a use-case demonstration only. Aspects 1 and 3 are answered mainly by
derivation (Sections II and IV); their runs illustrate.

## 2. Rules for every result

- Simulation only; truth T_AF (physics baseline + tanh Coulomb rail friction + payload absorber), controller K1
- One dataset and one frozen code version for all runs; git SHA recorded in every run (the previous runs had none)
- Training on the noisy records only (18 training records); no training on the noise-free twins
- Validation (6 records) selects checkpoints and black-box hyperparameters only; it is never a reported score
- **Every reported result is judged on the test set** (I1 to I6 interpolation, E1 to E6 extrapolation, `DATA-DESIGN.md` 7.3, 7.4)
- One seed per grey-box run (compute limit); the same seed in every run and the same 10 % detuning direction in
  every detuned run
- Start (D-243): steps 2 to 4 start at the true combinations, so each step changes one thing; the detuned start is
  used only where it does work: step 5 (OBC vs U recovery) and step 6 (the black box gets no physical knowledge,
  so the final model must not start at the exact parameters)
- Same training settings, budget and checkpoint rule in every grey-box run (`TRAINING-DESIGN.md` sections 2, 3)
- Augmentation network: tanh MLP, 2 x 16, with g_aug initialised to zero (Jan's `zero_init_feed_forward_nn`: final
  weight and bias all zero, no bypass), so f_aug = 0 and g_aug = 0 at the start; `THESIS_ARCH=mlp`,
  `THESIS_INIT=zero`, PS2 selected independently with `THESIS_PS2` (user decision 2026-10-08, D-240; launch path:
  `tasks/handoffs/2026-10-08-zero-init-mlp-runs.md`). Live g_aug (MLP) stays a toggle and the ResNet stays in the
  code; neither is part of the thesis results
- Floor per test record: the noisy record minus its noise-free twin; every error is read against it
- Primary metric (D-256, 2026-10-15): NRMS_e,j = RMS(y_hat_j - y_j) / RMS(y_j - r_j) in % per test record and axis,
  whole record after the encoder window; beside it the absolute RMS(y_hat - y) in m and the noise floor. Records with
  multisine (I1, I2, E1) and without (I3 to I6, E2 to E6) are reported separately, never one mean over I1 to I6. Both
  sides are stated: the gain is in the excited absorber band; on the records without multisine the augmentation is equal
  to or slightly worse than the baseline (2026-10-11 runs). Final runs also select and weight the loss by NRMS_e
  (FINAL-RUN-FIXES items 13 to 18)
- Single-seed rule: a difference counts only if it exceeds 2 times the seed spread measured in the previous campaign
  (HEURISTIC; old runs 121 to 126 and 41 to 46, same settings, older data); smaller differences are reported as "not
  resolved with one seed" (?) spread values to be read from those runs' final logs
- A rollout that diverges counts as a failure of that model on that record

## 3. Results in presentation order

The order builds the final model step by step. The run order on the server is different (section 4): n_a = 2 and PS2
are fixed in advance, so all runs start together.

### Step 1. Baseline form: LPV vs LTI (Aspect 1)
- Shows: explicit Y scheduling lowers the prediction error, increasingly away from the frozen Y of the LTI model
- Runs: none. LPV baseline and frozen-LTI baseline (mid-stroke), both at the true parameters, no training
- Test records: I2 / E1 (standstill, Y -0.225 / -0.39), I3 / E2 (moving, stops up to -0.33 / +0.39)
- Metric: RMS per axis; growth from I to E per model
- Figure: small bar chart, LPV vs LTI, I and E side by side, floor marked
- Stated in the text: the truth contains the LPV structure, so this is a demonstration, not an open test; its value is
  the size of the position-dependent error compared with the absorber error

### Step 2. Plain augmentation, n_a = 2 (Aspect 2)
- Shows: augmentation with added states lowers the error below the baseline, but plain training leaves most of the
  absorber-band error in place
- Runs: R3 (U, plain, n_a 2, correct start)
- Test records: I1 to I6
- Metric: RMS per axis; RMS in the 230 to 297 Hz band on Y; x_a clamp effect (added-state rows zeroed in the rollout)
- Expected (from the previous campaign, detuned runs, so a rough guide): about 90 % of the error below 230 Hz
  removed, about 20 % in the band; clamp changes the model little

### Step 3. PS2 vs plain, n_a = 2 (Aspect 2)
- Shows: with PS2 the added states learn the missing mode; same model, same loss, only the start of the added states
  differs
- Runs: R5 (U, PS2, n_a 2, correct start) against R3
- Test records: I1 to I6
- Metric: as step 2; plus the clamp effect for both
- Expected: about 6x lower error, about 90 % of the band error removed (previous campaign, validation, detuned runs)
- Figure: Y error spectrum for baseline, plain and PS2, absorber band marked, floor PSD
- Counts against: PS2 not better than plain beyond the single-seed rule
- If PS2 fails here: steps 4 to 6 lose their basis; check R5's validation curve early (about update 3000)

### Step 4. Number of added states (Aspect 2)
- Shows: without added states the mode cannot be represented; 2 states suffice; more states add nothing
- Runs: R6 (U, plain, n_a 0; with n_a 0 PS2 has nothing to initialise), R5 (n_a 2), R7 (U, PS2, n_a 8); all from
  the correct start; optional R9 (n_a 1), R10 (n_a 4)
- Test records: I1 to I6
- Metric: RMS per axis and band RMS against n_a; parameter count per n_a in the table
- Figure: error against n_a, plain n_a 2 as a separate marker
- Expected: large drop from 0 to 2, flat from 2 to 8; n_a 1 (optional) gives a real pole only and should miss the mode
- With one seed: only the large drop and the absence of a large further drop are claims; differences between 2, 4
  and 8 below the single-seed rule are not

### Step 5. Final model with OBC (Aspect 3)
- Shows: what OBC does to the ten parameter combinations compared with no OBC, and at what accuracy cost
- Runs: detuned start R1 (OBC, PS2, n_a 2) vs R4 (U, PS2, n_a 2); correct start R2 (OBC) vs R5 (U)
- Data: parameters from the training fit (truth known); accuracy on I1 to I6
- Metric: relative error of each of the ten combinations, their norm, distance travelled from the start; test RMS
- Inner products (D-246), Euclidean in the normalised coordinates of Section IV, Pi = projection onto ran Phi:
  - network overlap ||Pi f_aug|| / ||f_aug||, U and OBC: on held-out tuples at x_a = 0 (where OBC holds by
    construction) and along the test rollouts with active added states (what leaks in away from x_a = 0)
  - truth overlap ||Pi eps|| / ||eps|| (eq:obc_eps, needs the true parameters, simulation only): whether T_AF meets
    the recovery condition; explains why OBC gives a unique but biased split (old name rho*)
- Table: rows = ten combinations, norm, test RMS, network overlap (x_a = 0 and rollout), truth overlap; columns =
  true, start, U, OBC (detuned and correct start)
- Correct start is the cleanest test: starting at the truth, does U drift away while OBC stays?
- Limit stated with the result: OBC orthogonalises the physical correction at x_a = 0 only
  (`AUGMENTATION-INITIALIZATION.md` 12.2); with PS2 the added states are active, so the part of f_aug that runs
  through x_a is not protected; on T_AF the missing physics is not orthogonal, so recovery of the truth is not
  guaranteed (Section IV)
- Counts against: OBC drifts as much as U from the correct start, or OBC clearly less accurate on the test set

### Step 6. Generalisation and black box (Aspect 4)
- Shows: the final model is at least as good as the black box at interpolation and better at extrapolation; it also
  predicts under a changed controller, which the black box cannot by construction (it models the whole loop)
- Runs: R1 (final model), R8 (black box: hyperparameter search, then the selected configuration; same data and
  records)
- Black-box state count nx: a hyperparameter of the search (R8), selected on validation like the widths and the
  learning rate; the black box models the whole closed loop (r, f) -> e, so it needs more states than the plant alone
  (the previous search fixed nx = 16, `BLACKBOX-METHOD.md` 3.2)
- Test records, pairwise (each E differs from its I partner in one axis):
  - Y: I2 / E1, I3 / E2
  - acceleration: I5 / E3
  - velocity: I6 / E4
  - ASMPT ILC move, stress record: I3 / E6
  - cross-coupling: I4, scored on X1 - X2 and on the X-only and Y-only segments
  - controller use case: E5 (K1 gains +20 %) against I3 under K1
- Models compared: LPV baseline, final model, black box
- Metric: NRMS_e per record; the black-box minus final-model gap at E against the same gap at I
- Figure: grouped bars per pair, I and E side by side, three models, floor marked; E5 as a separate panel
- Report absolute test performance at I and E regardless of whether the black box matches the final model at I.
  Interpret the change in the black-box minus final-model gap from I to E as relative extrapolation degradation
  only when their I-level accuracy is comparable within the single-seed rule. Worse interpolation alone does not
  establish that the black box is undertrained; assess that separately from its validation convergence and search effort.
- Counts against: black box as good or better on the E records

### Answer to the research question
Steps 2 to 4 say how to augment (parallel, added states at the order of the missing mode, PS2 start); step 5 says
what OBC preserves and where it stops; step 6 says how far the improvement carries beyond the training conditions.

### Presentation agreed 2026-10-16 (Dirk; Jasper's feedback; cross-check of Hoekstra, Gyorok, Drenth)
Supersedes the figure and table lines of steps 1 to 6 above where they differ.
BINDING RULE (Dirk 2026-10-16, Jasper point 2): the main text shows only what answers the research question, one result
per aspect, each opened by its criterion. Everything that justifies a CHOICE (structure, number of added states, PS,
start, settings, black-box size) goes to the appendix. Metric NRMS_e (D-256) with the RMS in m and the floor; records
with and without multisine reported separately; no oracle row; no "error removed" percentage (factors in the prose).
Main-text models (five): M_LPV and M_LTI (true parameters, untrained), M_U (R4: U, PS, n_a 2, detuned start), the final
model M_OBC (R1: OBC, PS, n_a 2, detuned start), the black box M_BB (R8).
- A (Aspect 1): LPV vs LTI figure, NRMS_e on X, I2/E1 and I3/E2.
- B (Aspect 2): baseline vs final model vs black box, NRMS_e per axis with and without multisine, floor; FRF F_Y -> Y
  (res_frf, mean over Y = -0.39 to 0.39 m) with one sentence on the spurious X-Y cross terms; I1 time trace captioned
  with its NRMS_e. Why this structure (n_a, PS) is in the appendix.
- C (Aspect 3): table in the style of Gyorok 2026 Table 1, detuned start, M_U vs M_OBC: the ten combinations (delta_j) and
  their norm, full-model NRMS_e, physical part alone (Gyorok 2025 Table 1), network overlap rho(F_aug), truth overlap
  rho(Xi), and the PREDICTED first-order bias of eq:obc_recovery beside the observed delta_j of OBC. Verdict per parameter.
- D (Aspect 4): NRMS_e per I/E pair (I2/E1, I3/E2, I5/E3, I6/E4; E6 as stress record), baseline, final model, black box,
  floor marked; plus the multisine twins (evaluation-only extension, 2026-10-16) once generated.
Appendix, choice justification (on the VALIDATION records where a choice is compared, so the test set stays for the main
results; Dirk's confirmation of the validation rule pending): structure comparison n_a 0 / 2 / 8 and plain vs PS (all
true-start runs R3, R5, R6, R7, plus the detuned n_a 0 pair), error spectrum PS vs plain, parameter recovery without added
states (U and OBC, n_a 0, detuned start: two NEW runs, Jasper point 5), OBC vs U from the correct start (R2, R5), full
structure table (all runs, NRMS_e and m, physical part per run), validation curves, black-box search table, full 3 x 3 FRF
(res_frf_full), setup figures (multisine band, coverage), setup checks, hyperparameters.

## 4. Server runs

All runs: noisy training records, one seed, same detuning direction in the detuned runs (D-243), `THESIS_ARCH=mlp`, `THESIS_INIT=zero`, window
0.1 s, settings of `TRAINING-DESIGN.md`. Run-table rows (`docs/gantry-augmentation-problem-log.md` section 12, D-090)
before launch.

| Run | Arm | PS2 | n_a | Start | Steps | Duration |
|-|-|-|-|-|-|-|
| R1 | OBC | yes | 2 | detuned | 5, 6 (final model) | long (48 h or more) |
| R2 | OBC | yes | 2 | correct | 5 | long (48 h or more) |
| R3 | U | no | 2 | correct | 2, 3 | normal |
| R4 | U | yes | 2 | detuned | 5 | normal |
| R5 | U | yes | 2 | correct | 3, 4, 5 | normal |
| R6 | U | no | 0 | correct | 4 | normal |
| R7 | U | yes | 8 | correct | 4 | normal |
| R8 | black box | n/a | nx searched | n/a | 6 | search: many short configurations; final: short |

Total: 7 grey-box runs (2 OBC, 5 U) plus the black box (search + final configuration).

R8, the black-box hyperparameter search (on the new dataset, since all data and runs are regenerated):
- The configuration (including state count, network widths and learning rate) and checkpoint are selected on the
  6 validation records only (free-run servo-error RMS, as for the grey box); test records never loaded during the
  search and test results never guide tuning
- Searched hyperparameters: state count nx, network widths of f, h and the encoder,
  learning rate; the previous search (`BLACKBOX-METHOD.md` section 8, e1 to e8) is the starting point
- State the initial search ranges and total compute budget before launch. The search may expand or refine those
  ranges based on validation results within that budget; record the reason for each expansion.
- Fixed, as in the grey box: the same 18 training records, 4 kHz, window 0.1 s, encoder lag 29, seed 1
- Budget: equal to or larger than the grey box's tuning effort; report configurations tried, training effort
  (GPU-h or CPU-h, with hardware stated), failures and validation convergence. Search effort alone does not
  establish that tuning is sufficient.
- State the validation plateau criterion and maximum training limit before launch. Final: retrain the selected
  configuration to its validation plateau or the stated limit, retaining its best validation checkpoint; this model
  enters step 6. Report any unresolved convergence at the limit; test results do not determine stopping.

Optional, if compute allows:

| Run | Arm | PS2 | n_a | Start | Adds |
|-|-|-|-|-|-|
| R9 | U | yes | 1 | correct | step 4 curve |
| R10 | U | yes | 4 | correct | step 4 curve |
| R11, R12 | black box | n/a | selected nx | n/a | seeds 2 and 3 of the selected configuration: assesses sensitivity to initialisation |

Launch order: R1 and R2 first (longest), then R3 to R7, then R8.

Before launch:
- launch path passes and logs `THESIS_INIT` next to `THESIS_ARCH` and `THESIS_PS2` (`run_thesis.sbatch`, `submit.sh`;
  open in `tasks/handoffs/2026-10-08-zero-init-mlp-runs.md`); each run's log line reads `g_aug=mlp/zero` and its
  config.json has `ANN_ARCH=mlp`, `G_INIT=zero`
- new run ids above 164, never reused
- dataset fixed: lowpass (section 7)
- code frozen; git SHA in every run
- smoke test of each row type (`THESIS_SMOKE=1`)
- the U PS2 and OBC PS2 runs build identical initial parameters for the same seed (D-240 check)

## 5. Evaluation only (no server runs)

Applied to every trained model with the same closed-loop scorer as checkpoint selection (D-227):
- closed-loop table on I1 to I6 and E1 to E6 (E5 under K2), per axis, against the floor
- band RMS (230 to 297 Hz on Y) and the x_a clamp effect
- combination table (ten combinations, norm)
- black box scored with the same scorer on the same records
- LPV and LTI baselines at the true parameters (step 1)
- floors: noisy minus noise-free per test record
- Section III-C evidence: open-loop vs closed-loop rollout drift on one record
- training effort: GPU-h per run and the size of the black-box search, from the logs (Discussion, one sentence or a
  small table)
- R3 plant check (best linear approximation at three Y): kept only if the TF records exist (?); it supports the E5 use
  case by showing the model learned the plant, not only the K1 loop

## 6. Not run, and why

| Item | Reason |
|-|-|
| Several seeds per grey-box run | compute; single-seed rule in section 2 |
| Plain OBC run | Aspect 2 is answered on the U arm; OBC does not change whether the added states learn the mode |
| n_a sweep under OBC | 2 to 4 extra long runs; the sweep on U answers Aspect 2 |
| Plain U detuned start | steps 2 to 4 start correct (D-243); Aspect 3 is about the final model (PS2) |
| Noise-free training runs | training on noisy data is the realistic case; the floor comes from the noise-free records |
| Window sweep (0.1 / 0.2 / 0.4 s) | argued from the closed-loop memory (D-220) |
| Series augmentation | argued in Section III |
| Augmented LTI baseline | Quinten: baseline-only LPV vs LTI is enough |
| R5 orthogonal addition (T_OA) | data do not exist; the guarantee is derived in Section IV |
| Black-box LTI | not needed for any aspect |
| Settling interval | settling is no longer in the research question |
| Live initialisation, ResNet | ongoing investigation; Discussion or future work |
| Real Telica data | out of the main results (user 2026-09-24) |

## 7. Open points

- Dataset for all runs: `Coulomb-tanh-and-MSD-lowpass-noise` (D-238; user 2026-10-08)
- Seed spread for the single-seed rule: read from the final logs of runs 121 to 126 and 41 to 46 (?)
- TF records for the plant check: generated or not (?)
- Black-box search grid (nx, widths, learning rate) and the search budget (?)
