# Results design (thesis Section VI plus the setup items it depends on)

Updated 2026-09-26 to the current decisions; moved from `scripts/gantry/thesis-results-design/`. Companion documents in this folder: `DATA-DESIGN.md` (every record, section 7), `NOISE-INJECTION.md` (noise), `EXCITATION-VALIDATION.md` (band and excitation). Numbering follows the slides: R7 is controller transfer. Literature evidence kinds (A1 to F4) refer to `scripts/gantry/thesis-results-design/LITERATURE.md`.

- Truth for every result: T_AF = physics baseline + Coulomb rail friction + payload absorber (ma 0.50 mh, damping 0.03); nothing is absorber-only (user, 2026-09-26)
- All main results are simulation; real Telica data are a suggestion only (4b)
- Where the history sections (10, 11) conflict with sections 1 to 9, sections 1 to 9 win

## 1. Open questions (proposals, the user decides)

**Q1. Prediction target: settled**
- The closed-loop servo error e = r - y per actuator axis, predicted by the model run inside the known controller from the reference r and the injected force f, over the whole record after the encoder window (`DATA-DESIGN.md` section 2)
- Why: the digital twin evaluates motion and controller changes before the machine; both act on the servo error; under feedback y is within micrometres of r, so position scores are dominated by r

**Q2. Controller transfer: kept, narrowed in the first campaign**
- R7 shows one named case, K1 with all gains +20 % on one ASMPT route; broader controller changes are deferred (`DATA-DESIGN.md` 5.11, 7.5)

**Q3. Which LTI model** (?)
- R1 compares the LPV baseline with a frozen-LTI baseline, both refitted, without augmentation (Quinten)
- Open: the frozen-LTI physics model at mid-stroke refitted (B-LTI-refit), or an identified LTI state-space model (BB-LTI); the proposal is B-LTI-refit in R1 and both as rows in the R6 appendix table

**Q4. Control truths that are not T_AF** (?)
- The user's rule is that all truths are Coulomb + absorber; three proposed controls are not absorber-only but are also not T_AF:
  - T_0, baseline only: the zero point of R5 and the B-refit exact-recovery check
  - T_OA, a designed orthogonal addition (D-213, D-214): needed for R5 as Quinten framed it (only with a truly orthogonal addition)
  - T_F, baseline + friction: a second R2 truth (knee at 0 added states)
- Proposal: keep T_0 and T_OA, conditional on R5 being shown; drop T_F, so R2 runs on T_AF only (?)

## 2. Claims to results
| Claim (source) | Results |
|-|-|
| C1 compact CT LPV-LFR baseline with rational Y dependence | derivation in Section II; position dependence demonstrated in R1 |
| Aspect 1 baseline form | R1 (demonstration) |
| C2 dynamic parallel augmentation in closed loop, added states; Aspect 2 incl. cross-coupling | R2, R3, R6 (cross-coupling on I4), R7 |
| C3 state-level orthogonality and parameter recovery; Aspect 3 | R4, R5 (conditional) |
| Aspect 4 generalisation: held-out operating points (i), unseen motion (ii) | R6 (Y pairs for i; routes, acceleration, velocity and the ILC stress record for ii) |
| ASMPT criterion: training effort and robustness (Quinten) | R8 |
| Not shown, stated in `07` | section 4c |

## 3. Setup items the results depend on

### S1. Data
- `DATA-DESIGN.md` section 7, first campaign, all on T_AF with one fixed controller K1 (designed at Y = 0):
  - training 18 records (3 realisations), validation 6 records (3 realisations), 12 s at 20 kHz downsampled to 4 kHz
  - test: 6 interpolation records (I1 to I6) and 6 extrapolation records (E1 to E6), each E with one I partner differing only in the extrapolated axis (E6 a declared stress record)
  - FRF references for R3 (TF-1 to TF-3); control truths (Q4)
- Everything else is deferred (`DATA-DESIGN.md` 7.5) and listed in 4c

### S2. What the Telica logs contribute (no real-data training)
- The measured ILC move kinematics (D-210): the ILC-shape records in training and validation (at datasheet levels) and the exact move as stress record E6
- The noise level, as the calibration target of the force disturbance (S4)

### S3. Coverage display
- `DATA-DESIGN.md` section 8; optional in the thesis (Quinten: "if it helps")

### S4. Noise
- A force disturbance on the motor force inside the loop, calibrated so the frictionless loop reproduces the measured standstill error 9.8 / 10.4 / 6.3 nm rms (`NOISE-INJECTION.md`)
- Stated as "a force disturbance calibrated to the measured error level": the level matches, the spectrum does not (the measurement is mostly sensor noise above 200 Hz, the force noise acts mostly below it)
- N0 (noise-free twins, d = 0) only for R5, where recovery is derived noise-free
- Floors: noise-only twins of E1, I1, I4 and E5; floor-relative criteria only on those records, absolute errors elsewhere
- Closed-loop noise bias (Forssell and Ljung) is not tested at this level; stated in 4c

### S5. Arms and matching
- B-refit: LPV baseline with the ten combinations refitted, no network (D2); reference for parameter spread
- B-LTI-refit: frozen-LTI baseline, refitted, no network (R1; Q3)
- Aug-U: augmentation with co-estimated combinations, unconstrained
- Aug-Lambda: parameter penalty anchored at the detuned start (D-193), two settings: selected on validation, and matched to Aug-OBC's parameter spread
- Aug-OBC: state-level OBC, tangent basis (the thesis method)
- BB-NL: nonlinear black-box state-space model, same encoder, same closed-loop training; in R6 and R8 only (Quinten: the black box belongs to the final result); network size and hyperparameters chosen by a search on validation, no size limit (Quinten)
- BB-LTI: LTI state-space model trained the same way (R6 appendix row)
- Starts: 5 per arm; a grey-box start is a new network seed plus a new 10 % detuning direction; a black-box start is a new seed
- Data realisations: 3, each a replicate run with its own training and validation realisation, reported separately from the start spread

### S6. Metrics and decision rules
- Primary: servo-error prediction RMS in nm per actuator axis, RMS(e_hat - e) = RMS(y_hat - y), whole record after the encoder window; moving extrapolation records also on their out-of-range segments
- For comparing records: NRMS_e = RMS(e_hat - e) / RMS(e); position NRMS or BFR not used
- Frequency-resolved: PSD of e_hat - e against the floor PSD where a noise twin exists
- Parameters: relative error of each of the ten combinations, all ten always shown, plus the combination-error norm
- Comparisons across records use within-record differences between arms (difference-in-differences): every record has its own phase and noise, so phase is a nuisance factor between records
- "A better than B": mean difference larger than t(0.975, Welch df) times the standard error of the difference of means over starts
- "A as good as B": the 90 % interval of the difference inside ±delta, delta fixed before the runs (HEURISTIC practical margin)
- Power, stated before the runs: with 5 starts per arm and equal start spread s, the Welch rule detects only effects above about 1.46 s, and the 90 % equivalence interval has a half-width of about 1.18 s, so delta must exceed about 1.2 s or equivalence cannot be shown
- Unit of replication: the start (optimisation spread) for every rule above; the 3 data realisations are reported as three replicate outcomes, not pooled into the Welch or F rules
- A rollout that diverges counts as a failure of that arm on that record

### S7. Excitation
- `EXCITATION-VALIDATION.md`: every record class excites all ten combinations above the floor; no parameter band is needed; the absorber band 140 to 390 Hz at the production multisine level (?)
- Open: the floor was computed with encoder noise; it must be rechecked with the force noise before the band and the joint-estimation verdict are final

## 4. Results

### R1. Position dependence: LPV vs LTI baseline (demonstration)
- Claim: the LPV baseline predicts the servo error at held-out and extrapolated Y better than the frozen-LTI baseline
- Why it is a demonstration, not a test: the truth contains the LPV structure, so the outcome is close to predetermined; Quinten: "LPV better than LTI is already enough", without augmentation and without black box (less room for overfitting, fewer things compared)
- Arms: B-refit, B-LTI-refit
- Data: I2 / E1 (standstill at Y -0.225 / -0.39), I3 / E2 (moving, E2 reaching -0.33 and +0.39); training standstill points as reference level
- Physics prediction: the LTI error rises with distance from its frozen Y; the true X-yaw coupling grows 39 % towards -Y and 25 % towards +Y beyond the training edge; the LPV baseline keeps only the absorber-offset residual (constant 0.505 kg m and 1.01 Y + 0.05 kg m²)
- Criterion: the within-record difference LTI minus LPV is larger at E than at its I partner, S6 superiority
- Counts against: LTI as good as LPV at the ends
- Placement: main, small

### R2. Added states
- Claim: the absorber is one dynamic mode, so the error drops at the number of added states that mode needs and not materially after; stated as the outcome for this system, not assumed in the method (Quinten: for a real system the number is not known beforehand)
- Varied: n_a in {0, 1, 2, 4, 8}, on T_AF (T_F as a second truth only if Q4 keeps it)
- Physics prediction: n_a = 0 cannot create the pole pair; n_a = 1 gives a real pole only; knee at 2; Coulomb friction is static in velocity and needs no added state
- Order chosen on validation; curve reported on I1 to I6
- Added-state ablation at the chosen n_a: the added-state contribution zeroed; the model must get worse by more than delta (output-level check, never x_aug against the absorber state)
- Criterion: n_a = 2 better than n_a = 0; n_a = 4 and 8 as good as n_a = 2; ablation worse by more than delta
- Counts against: no drop at 2, a continued drop beyond 2, or an ablation that changes nothing
- Placement: main

### R3. Plant-level check: best linear approximation
- Claim: the closed-loop-trained model reproduces the truth's best linear approximation at the production excitation level, including the anti-resonance and resonance and their Y dependence, at a trained, a between and an extrapolated Y
- Why a BLA: an FRF is linear and the friction truth is not (Quinten); the BLA holds at the stated level only, and friction does not show as a separate feature; Jasper's Bode-plot remark to be followed (?)
- Data: TF-1 to TF-3 at Y 0.15, 0.225 and 0.35, periodic multisine, 4 phase realisations each; TF-4 at -0.35 if negative-Y asymmetry is claimed (?)
- Model side: the trained model linearised at standstill at frozen Y (Jacobian FRF)
- Criterion: resonance and anti-resonance within the frequency resolution; magnitude error inside the BLA's standard deviation from the robust method (noise and nonlinear distortion separated, Pintelon 2020), at all three Y; period count and estimator fixed before generation
- Counts against: good servo-error prediction but a missed resonance or anti-resonance: the model fitted the K1 loop, not the plant
- Placement: main

### R4. Interpretability: OBC vs unconstrained vs penalty
- Claim: without OBC the network absorbs baseline-reachable dynamics and the combinations end at start-dependent values; with OBC their spread over starts is that of the physics-only refit, at no significant loss of accuracy; a penalty reaches that spread only at an accuracy cost
- Table: rows = the ten combinations, combination-error norm, distance travelled from the start, test servo-error RMS, tangent share and output-level overlap of the network contribution on test data; identifiability line under it (rank, condition number, largest column correlation of J on the training data)
- Columns: true, detuned start, B-refit, Aug-U, Aug-Lambda (validation), Aug-Lambda (spread-matched), Aug-OBC; mean ± std over 5 starts
- Appendix figure: Lambda sweep (Gyorok 2026 Fig. 5 form)
- Data: training set (parameters), I1 to I6 (accuracy); T_AF
- Criterion: variance ratio of the combination-error norm, Aug-U over B-refit, above the F(4,4) 5 % critical value; Aug-OBC over B-refit below it; Aug-OBC test RMS as good as Aug-U; spread-matched Aug-Lambda worse than Aug-OBC
- Counts against: Aug-OBC spread above B-refit, a test tangent share comparable to Aug-U's, or Aug-OBC significantly less accurate
- Placement: main (table), appendix (sweep)

### R5. Parameter recovery with a truly orthogonal addition (conditional)
- Shown only if a missing dynamic exists that is truly orthogonal to the parameter directions (Quinten); otherwise omitted and stated in `07`
- Claim: with a truly orthogonal addition, OBC recovers the true combinations from detuned starts (zero bias), where the unconstrained arm drifts (negation)
- Truths: T_OA (gate: its orthogonality check passes before any R5 run); T_0 as the zero-discrepancy reference (Q4)
- Data: training set on each truth, noise-free twins (N0)
- Criterion: after the convergence gate, the Aug-OBC combination errors within 2 start std of zero; Aug-U outside
- Figure: combination-error norm vs epoch per arm, zero line; predicted vs observed per combination
- Not in the first campaign: the predicted nonzero bias on T_AF (the earlier R5 claim); in 4c
- Placement: main if the gate passes

### R6. Generalisation vs black box and LTI
- Claim: interpolating, the augmented model is close to the black box; extrapolating, it holds where the black box degrades; it stays better than the LTI model and predicts the cross-coupling error the baseline misses
- Arms: B-LPV, B-LTI-refit, Aug-U, Aug-OBC, BB-NL; BB-LTI in the appendix table
- Data, pairwise (`DATA-DESIGN.md` 7.3, 7.4): Y (I2 / E1, I3 / E2), acceleration only (I5 / E3), velocity (I6 / E4), the measured ILC move as stress record (E6 against I3); cross-coupling on I4, X-only and Y-only segments scored separately; I1 as the same-setting reference
- Predictions (`DATA-DESIGN.md` 5.1, 5.3, 5.4): Y: black box and LTI rise beyond the edge; velocity: damping is in the physics, the black box saturates; acceleration (E3): force stays inside the multisine's training range, so the black box need not fail, and an equal outcome is reported as such; E6: stress case
- Figure: grouped bars per pair, I and E side by side, NRMS_e per arm on a log axis, floor marker where a noise twin exists; appendix table with per-axis RMS in nm, all arms, 3 data realisations, and BB-NL training error
- Criterion: per pair, the within-record arm differences at E against those at I (S6); at I, Aug-OBC as good as BB-NL or better; at E, Aug-OBC better than BB-NL; Aug-OBC better than LTI throughout; cross-coupling Aug-OBC better than B-LPV
- Interpreted only if BB-NL reaches Aug-OBC's I-level accuracy within delta; otherwise read as "black box undertrained", not "black box generalises worse"
- Counts against: BB-NL as good or better at the E records; LTI as good or better anywhere; no cross-coupling gain over B-LPV
- Placement: main

### R7. Controller transfer (one named case)
- Claim: trained under K1, the augmented model predicts the servo error under K1 with all gains +20 % on the I3 route (use case: test a controller change on the model before the machine)
- Data: E5 against its own floor (noise twin); I3 under K1 as reference
- Physics prediction: +20 % raises the crossover (about 120 Hz on the rails, outside the trained 74 to 108 Hz) and the Y-loop gain around the 266 Hz closed-loop absorber peak, where the baseline misses the absorber: the baseline mispredicts the servo error there, the augmented model does not
- Criterion: absolute prediction error on E5 relative to its floor, Aug-OBC against B-LPV, S6; a model unstable under K2 fails; no claim across controllers or routes
- Deferred: -20 %, 60 and 150 Hz bandwidth, feedforward; with them the predicted-change score e_K2 - e_K1
- Placement: main (a panel of the R6 figure)

### R8. Training effort and robustness: augmentation vs black box (Quinten)
- Claim: the augmented model needs less tuning and trains more reliably than the black box; for ASMPT a new machine means retraining, so a robust, cheap training is a requirement, even at slightly lower accuracy
- Measured on the R4 and R6 runs, no extra data:
  - GPU-h to convergence per run, per arm
  - size of the hyperparameter search needed (configurations tried, GPU-h) to reach the reported accuracy
  - spread of final accuracy over starts and the fraction of diverged or failed runs
- Criterion: spread over starts compared with the F rule; failed-run fractions reported; effort compared at the accuracy each arm reaches; if the search for the black box is hard, that is stated as a result (Quinten)
- Counts against: the black box trains as cheaply and as reliably; then the case rests on extrapolation and interpretability alone
- Placement: main, a small table

## 4b. Suggestion, not a main result: real Telica data (Discussion, future work)
- Why it is the natural next step: real-data demonstration is the field's endpoint (Retzler, Kessels) and the research plan promised it; its absence is a departure to state in `07`
- Claim it would test: at held-out operating points the augmented model predicts the servo error better than the refitted baseline and as well as the black box, with reproducible combinations
- Parameter evidence without truth: physics part alone with co-estimated combinations; spread over starts; identified masses against the datasheet

## 4c. What the thesis does not show (one limitations paragraph in `07`)
- Prediction accuracy on the machine; parameter interpretability on real data
- Other discrepancy types: every result uses Coulomb + absorber (user decision); absorber-only or friction-only behaviour is not shown separately
- The predicted nonzero parameter bias on T_AF (the earlier recovery-condition claim)
- The deferred extrapolation axes (`DATA-DESIGN.md` 7.5): X position, jerk, move distance, 1.75 m/s, multisine level, band outside B_tr, swept sine, payload, the datasheet cycle, controller changes other than +20 %, Y-standstill beyond one pair
- Noise: the spectrum of the measured noise (the force disturbance matches its level only); closed-loop noise bias
- That the machine's missing dynamics resemble the absorber and Coulomb truths, which are chosen, not identified
- Settling
- `01` wording scoped accordingly: "validated" and "industrial" mean simulation with ASMPT motion kinematics

## 5. Method and setup choices (not results), one justification each
- Closed-loop residual training with the known controller: data recorded under feedback and X, Y free integrators
- One fixed controller K1 designed at Y = 0 for every record: one controller, as on the machine; the loop still varies with Y through the plant (`DATA-DESIGN.md` 5.11)
- Evaluation in closed loop, plant checked as a BLA (R3): open-loop simulation of free integrators drifts by construction
- Noise as a calibrated force disturbance (S4)
- Ten identifiable combinations as parameters: structural identifiability; practical identifiability reported in R4
- Encoder and its history length: SUBNET practice (Beintema 2023)
- Training window: the joint-estimation session
- 10 % detuned starts with the prior off: setup; the resulting spread is part of R4
- Network writes X and Y rows (D-103): method constraint
- OBC reference set, per-epoch refresh: method
- Black-box size and hyperparameters: search on validation, budget reported (R8)
- Excitation band: chosen from the mismatch and checked against the delivered spectra (S7), a setup figure, not a result

## 6. Page budget (Section VI, 2.90 pages)
- Main: R1 small figure 0.20, R2 figure 0.25, R3 figure 0.30, R4 table 0.40, R5 figure 0.30 (if shown), R6 / R7 figure 0.40, R8 table 0.15; text 0.90
- Appendix: Lambda sweep, per-axis R6 table, B-refit exact recovery, BB-NL training curves
- If R5 is omitted, its space goes to the R6 pairs and the cross-coupling panel

## 7. Dependencies on other sessions and people
- Session 1: the band and the joint-estimation verdict, rechecked with the force noise (S7)
- Joint-estimation window session: training window for R4 and R5
- Orthogonal-addition sessions (D-213, D-214): T_OA and its orthogonality gate (R5)
- Jasper and Dragan: the ASMPT profile set (replaces I3 to I6 and E2 to E6 values where it differs)
- Quinten: realistic controller changes (R7 beyond +20 %), the LTI model (Q3)

## 8. Feasibility
- Data: 172 committed simulations plus 42 conditional (T_OA), about 4 h of MATLAB (`DATA-DESIGN.md` 7.9)
- Unit costs: one grey-box training run 5 to 11 GPU-h on the cluster; closed-loop black box about 3 h per configuration, times its hyperparameter search; baseline-only refits far cheaper
- Runs: R4 and R6 campaign, 5 network arms (Aug-U, Aug-OBC, Aug-Lambda x 2, BB-NL) x 5 starts on realisation 1 plus one start each on realisations 2 and 3: 35 runs; R2: 5 orders x 5 starts, one shared: 24 runs; R5 (conditional): 2 arms x 5 starts: 10 runs; about 70 runs, roughly 350 to 770 GPU-h plus the black-box search
- Evaluation: 12 test rollouts per trained model; FRF references need no rollouts
- Reduced campaign, pre-stated: 3 starts instead of 5 weakens every rule (Welch threshold about 2.3 s: t(0.975, 4) x sqrt(2/3) = 2.78 x 0.82); then superiority claims only, with intervals, no equivalence claims

## 9. Change log
- 2026-09-26: updated to the current decisions and moved to `Thesis-writeup/Documentation/`:
  - every result on T_AF; nothing absorber-only (user); T_A, T_A at ma 0.10 removed
  - R1 reduced to LPV vs LTI baselines, no augmentation, no black box (Quinten)
  - R3 as a best linear approximation on the friction truth (Quinten)
  - R5 only with a truly orthogonal addition (Quinten); the predicted-bias claim moved to 4c
  - R6 on the 6 + 6 paired test records of `DATA-DESIGN.md`; black box only in R6 and R8 (Quinten)
  - controller transfer renumbered R7 and narrowed to one case
  - R8 added: training effort and robustness (Quinten)
  - S1, S4, S7 now point to `DATA-DESIGN.md`, `NOISE-INJECTION.md`, `EXCITATION-VALIDATION.md`; S6 gained the power statement and the unit of replication (`scripts/gantry/data-set-design/DATA-AUDIT.md`)
- Sections 10 and 11 below are kept from the 2026-09-24 version as history; their truths, rungs and R8 numbering are superseded by sections 1 to 9

## 10. History: evidence kinds of LITERATURE.md, used or omitted (2026-09-24 version)
- A1 structure ablation: used (R2; R6 appendix table)
- A2 contribution decomposition: used as added-state ablation (R2), test-data tangent share and output overlap (R4)
- A3 true vs estimated: used (R4, R5)
- A4 error vs penalty weight: used (R4 appendix, two Lambda rows in main)
- A5 error vs data length: omitted; at the measured noise the finite-data variance is negligible, so a plateau vs N appears whatever the mechanism; replaced by A13 and the T_0 control
- A6 physics part alone: suggestion 4b; in simulation the truth column of R4 is stronger
- A7 extrapolation: used (R6 rungs 2 and 3, R1 ends)
- A8 frequency domain: used, vs truth (R3) and error PSD vs floor (R6b)
- A9 order sweep: used (R2)
- A10 identifiability diagnostics: used (R4 identifiability line)
- A11 initialisation and null controls: used (T_0 in R5); B-refit exact recovery in appendix
- A12 discrepancy severity: used (R5, four truths)
- A13 convergence curves: used (R5a, with predicted floor)
- A14 error vs horizon: omitted; the claim is whole-record closed-loop prediction, horizon belongs to the window session
- B1 SNR sweep: replaced by the measured level N1
- B2 floor reference: used in every error result
- B3 noiseless first: used (R5)
- B4 Monte Carlo over data realisations: used (S5, 3 realisations)
- B5 process noise: omitted; the force-disturbance reading has no predicted effect at these levels (4c)
- B6 closed-loop bias: not testable at N1 (S4); stated in 4c
- C1 record splits: used (S1); operating-point rotation belongs to 4b
- C2 held-out scheduling values: used (S1, R1)
- C3 varying-scheduling validation: used (rung 2 moves cross 80 mm of Y)
- C4 coverage display: used (S3)
- D1 to D5: used (S5)
- D6 linear black box: used as an appendix row (BB-LTI)
- D7 leaderboard: not applicable, no public gantry benchmark
- D8 oracle: replaced by the realisation-difference floor
- E1 frozen responses over Y: used (R3 truth curves at three Y)
- E2 LPV vs LTI on the same data: used (R1)
- E3 scheduling-rate axis: omitted; truth and baseline share the no-Coriolis assumption, stated in 4c
- E4 control-performance validity: covered by R8
- E5 controller transfer: used (R8)
- F1, F2 spread over runs and as threshold: used (S5, S6, with the Welch and equivalence rules)
- F3 best-of-N: checkpoint selection only, spread always reported
- F4 single run: avoided


## 11. History: critic points of the 2026-09-24 version and how they were handled
1. R5 tests the wrong estimator: accepted; Gauss-Newton bias of the actual criterion computed offline beside J+ Delta*, both on the S1 set (R5)
2. R5 passes without convergence: accepted; convergence panel (A13) and the gate "2 std below |bias|" (R5); N-sweep dropped (section 6, A5)
3. Null truth as positive control: accepted; T_0 is the rho* = 0 case of R5; B-refit exact recovery stays an appendix check, the augmented arms on T_0 are the result (section 5)
4. R4 cannot separate negation from non-convergence: accepted; B-refit reference, combination-error norm, distance travelled, F(4,4) criterion (R4)
5. R1 handicaps the LTI arm through OBC: accepted; Aspect 1 decided on Aug-U vs LTI-Aug-U, OBC pair secondary with an offline tangent-share prediction (Q3, R1, S5)
6. ASMPT relevance rests on one inert anchor: partly accepted; wording becomes "ASMPT motion kinematics", N1 stated as realism, not an outcome driver (S2, S4); Telica controller and parameters in the truth offered as (?) because it touches real-data artefacts (S1)
7. What the thesis does not show: accepted; section 4c
8. Equivalence rewards unstable arms: accepted; Welch rule for superiority, interval-inside-margin for equivalence, spread defined per arm (S5, S6)
9. Aspect 4 ii not tested: accepted; Telica-kinematics class held out of training, rung 2 is an unseen class, stressed element stated per rung, conditional retrain (?) (S1, R6)
10. BB-NL undertrained: accepted; equal tuning budget, BB-NL training and rung 1 accuracy reported, rungs 2 and 3 interpreted only if BB-NL reaches rung 1 parity (S5, R6)
11. R3 checks other models than R6: accepted; one main truth T_A for all main prediction results (S1)
12. Cross-coupling has no main result: accepted; cross-coupling group in the R6 figure with its criterion; absorber direction stated (S1, R6)
13. Error PSD missing: accepted; R6 panel b and S6
14. Practical identifiability hidden in setup: accepted; identifiability line under the R4 table
15. Aug-Lambda set up to lose: accepted; anchor stated, two Lambda rows in main (S5, R4)
16. OBC tangent share cannot fail: accepted; tangent share on test data at final J plus output-level overlap (R4)
17. R8 not scale invariant: accepted; NRMS_e ratio K2 / K1, K1 crossover checked first, instability counts as failure (R8)
18. R3 threshold and checkpoints: accepted; nonparametric FRF std as threshold, Y-restricted checkpoints only (R3, section 9)
19. Gap width open: accepted; fixed at 80 mm from the ASMPT grid spacing before any run (S1)
20. Coverage omits the absorber band: accepted; S3 panel c
21. R1 in the Coulomb regime: accepted; R1 on T_A (S1)
22. rho* may not vary: accepted; rho* for all truths offline first, rewording rule stated (R5)
23. Noise settings vary nothing: accepted; N2 dropped, N0 only in R5, N1 elsewhere, untestable closed-loop bias moved to 4c (S4)
24. C1 compactness not shown: accepted as method evidence, not a result; one line in Section II (section 5)
25. Baseline-only LPV vs LTI cannot fail: accepted; refitted LTI, baseline pair demoted to a setup sanity check (Q3, S5)
26. Q1 argument outdated: accepted; re-argued on the controller-evaluation use case with R3 as co-evidence (Q1)
27. Truths are author-chosen: accepted as a limitation (4c); Telica residual spectrum as motivation offered as (?) in 4b
28. R2 ablation prediction wrong: accepted; ablated error must exceed trained error by more than delta (R2)
29. R4 invites judging OBC against truth: accepted; predicted column (R4)
30. Artefact-driven T_A table and affine row: accepted; the second-truth parameter table and the affine arm are removed (R4, section 5)
31. "Self-scheduled" has no evidence: partly accepted; no new result, R6 states that every rollout self-schedules and divergence counts as failure
32. Page budget: accepted; freed space spent on the cross-coupling group and error PSD, appendix `\todo` first (section 7)
33. No pre-stated narrowing: accepted; section 9b
