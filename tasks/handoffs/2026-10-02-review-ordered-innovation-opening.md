# Handoff: critically review the "ordered innovation opening" that made the added states learn an in-band pole pair
**From**: session of 2026-10-02 | **Branch**: Augmentation | **Effort suggested**: xhigh (method critique across code, logs and identification theory; no training)

## 1. Task
Review, as a sceptical control engineer, the training approach developed overnight 2026-10-01/02 in `scripts/gantry/absorber-learning-diagnosis/` (probes E4 to E9). Decide for each claim in section 4 whether the code and logs support it, partly support it, or do not, and say why. Judge the mechanism (does it work for the reason stated?), the evidence (is it enough, is anything biased or confounded?), the constraints (does it stay inside what the user allows, section 2?), and generality (would it carry to an unknown missing dynamic, nonlinear effects, real closed-loop data?). End with the single most informative next experiment. Respond in text to the user; do not edit files or launch runs.

## 2. Out of scope
- Running training, probes or new numerical scripts (user: one job at a time, no exploratory numerics; this session is a review). Proposing experiments is in scope.
- Editing any file, including `DECISIONS.md`, `CAUSE-AND-DIRECTION.md`, the probe scripts, and the prepared E3 server files (`innovation_opening.py`, `model_innovation.py`, `gantry_innovation_opening.py`, `run_gantry_innovation_opening.sh`, `test_innovation_opening.py`, `smoke_innovation_opening.cmd`; already submitted as job 87446).
- Building the server version of the approach (only after the user has seen the review).
- `kamtin-data/Telica.mat`, `kamtin-data/Data Telica/`.

## 3. Where things stand
Branch Augmentation, tree dirty (many unrelated files), nothing committed from this work. No local job running. Server: E3 array (job 87446, arms U and OBC, the unordered E3 form) submitted by the user, results not yet seen.

## 4. The approach and the claims to review
Model (unchanged, Jan's framework): physical rows x(k+1) = physics(x, u) + ANN_phys(z); added rows x_a(k+1) = ANN_a(z); z = [x, x_a, u]; one 2x16 tanh MLP with zero-init output (model equals the baseline at start); encoder x_hat = W^b-map plus random linear W^a on 30 past u, y samples (Hoekstra 2026 Eq. 8); thesis loss = closed-loop 400-step simulation MSE. n_a = 2 added states (thesis setting). Data: the thesis simulated noisy dataset, run 42 settings (`e_thesis_checkpoints.cfg_for('u', 2, 1)`), 18 train / 6 val records.

The training change (all temporary, `f2_open.py`):
1. Innovation term: e = next 40 output errors of the current model's rollout (detached, divided by batch RMS); a training-only zero-init LINEAR head H predicts e from x_hat_a(k); loss ||H x_hat_a(k) - e||^2 trains H and the encoder W^a. Intended effect: x_hat_a becomes the 2-d summary of the past that predicts the model's coming errors, i.e. (a linear transform of) the missing mode's state.
2. Equation-error (EE) term: ||x_hat_a(k+1) - ANN_a(x_hat(k), u(k))||^2 with both encoder states detached (the encoder acts as teacher, only the ANN learns; detaching prevents collapse to x_a = 0). Intended effect: A_aa = dANN_a/dx_a becomes the rotation that propagates the oscillating x_hat_a, i.e. the complex pole pair; also the drive via u.
3. Thesis loss throughout and alone at the end: learns the readout (how x_a enters ANN_phys) and corrects closed-loop vs open-loop pole.
Schedule (best form, E9): updates 1 to 500 term 1 only; 501 to 800 term 1 + term 2 with the shared hidden layers at 1e-3 and the added-row output layer at 1e-2 (thesis lr 1e-5 elsewhere; W^a and head 1e-3 throughout); linear decay of both terms to 0 at 950; then pure thesis loss (added-row output and W^a 1e-3, rest 1e-5). Weights of the terms matched to the thesis loss at their first evaluation. Batch 64, CPU.

Claims (evidence in `DECISIONS.md` section 4, ids in brackets):
- C1 Cause: (a) zero-init output gives no first-order pole gradient (Steiglitz-McBride), (b) nothing supplies a target with the mode, (c) E3's state map could not fit its target within the step budget, (d) order matters. [AA1, AB1, E4]
- C2 The innovation target carries the mode without system information: E3's x_a 90 % power in 150 to 300 Hz; its optimal 2-d state is a 232 Hz pair, zeta 0.05 (truth 212 Hz, 0.043; difference read as closed-loop shift); IV = LS, so no noise bias. [AA1, `outputs/aa_inn.log`]
- C3 Fit budget: EE-only fit on E3's encoder reaches its linear optimum (residual/var 0.002) and a 217 to 224 Hz pair at 96 to 100 % of points within 500 updates only with hidden 1e-3 + output 1e-2; output layer alone at 1e-3 reaches about half the points after 3000. [AB1, `outputs/ab_inn.log`]
- C4 Result: E5 (seed 1, 300-update phase) and E9 (seed 2, 500) learn a pair at 208 to 209 Hz, zeta 0.00 to 0.09, at 96 to 100 % of encoder points; eval loss 1.16e-9 / 1.18e-9 vs control 6.91e-9; validation MSE on/off 0.077 / 0.078 vs control 0.441; whole-record free-run sim-RMS 3.8e-6 m vs control 1.15e-5 m; physical poles shift <= 1.2e-3, combinations within 1.25 %. [E5, E9, AC1, AD1, g2 logs]
- C5 Robustness: with 300 updates seeds 2 and 3 were partial (E6 pair at 71 %, self-oscillates on the quiet val records; E8 79 %, zeta 0.33); 500 made seed 2 complete (E9). [E6, E8, E9, AD1]
- C6 Stability: 600 more pure-thesis updates (E7) keep the pair at 100 % of points but drive it marginal (local |z| 1.000); still stable over 12 s free runs. [E7, AD1]

## 5. Assumed but not verified (the review should weigh these)
- That the gain is the target plus fit mechanism, not just the large opening learning rates (U1/U4 raised rates 100x on the plain loss without effect, but no rate ablation of E9 exists).
- That 500 is a rule and not a post-hoc fit to seed 2 (chosen after E6 failed; seeds 1 and 3 untested at 500).
- That it transfers to batch 512, 26k updates at 1e-5, best-checkpoint selection, and to the OBC arm (projection of the ANN's physical-row output; untested locally).
- That a LINEAR head is adequate when the missing dynamics or their effect on the error are nonlinear (the head is discarded; the model stays a tanh MLP; a nonlinear head is untested).
- That the residual (innovation) target is acceptable as "no system information" and generalises to real closed-loop data (it has closed-loop poles and, on measured data, noise correlated with the encoder inputs; literature `literature/added-states-training/deep-research-LC-closed-loop-innovation.md`).
- That AC1's "Y error removed in 230 to 297 Hz" (0.1 s windows, checkpoint's own baseline) ranks like run 42's metric (full multisine records, true-parameter baseline).
- Gauge: x_a is any invertible linear transform of the mode's state; only eigenvalues of A_aa at encoder points and output effect are meaningful (pair share counts Jacobian eigenvalues at 24 encoder states of training windows).

## 6. Tried and failed (do not re-propose as new)
- Optimiser only: per-group lr 100x (U1, U4), early LM (L1) -> poles stay real/tiny -> direction, not speed, is the limit.
- EE without the innovation target (E1, E2) -> real lags (0.27 to 0.30) -> the random encoder's content is slow and baseline-like.
- E3 (innovation + EE, E3 rates, unordered) -> in-band pairs but radius 0.25, zeta 0.99 -> state map fit budget (AB1).
- E4 (fast rates from update 1) -> slow real pole 1.0 to 1.3, loss spike, final pair zeta 0.72 -> map fitted the encoder's early slow content.
- Init / parameterisation changes (identity skip, random linear part, polar pair) -> excluded by the user as remedies.

## 7. Achieved
Probe scripts `f2_open.py` (E4 to E9), diagnostics `aa_target_ceiling.py`, `ab_ee_fit.py`, `ac_band.py`, `ad_valfree.py`, steal check `g2_steal.py`; checkpoints `outputs/f2_e4.pt` to `f2_e9.pt`; logs `outputs/f2_e*.log`, `g2_e*.log`, `ac_band*.log`, `ad_valfree.log`. Implemented and run locally; not validated at server scale.

## 8. The open question
Is this a sound, general training method that makes added states learn missing dynamics, or a system-tuned recipe whose success on this case owes most to large learning rates and hand-picked switch points? Candidate answers: sound and general / sound mechanism but fragile settings / confounded. Evidence that decides: section 5 items, a rate ablation, seeds at the 500 schedule, the E3 server array as control.

## 9. Next action
Read the files in section 11, then write the review in text: per claim C1 to C6 a verdict (supported / partly / not) with the decisive evidence or the missing evidence; the strongest alternative explanation of the results; whether section 2 of the user constraints (see `tasks/handoffs/2026-10-01-overnight-added-states-remedies.md` section 2) is respected, item by item (innovation target, rate factors, schedule); generality to an unknown or nonlinear missing dynamic and to real closed-loop data; and exactly one next experiment with prediction, pass/fail and cost.

## 10. Acceptance criterion
Done when every claim C1 to C6 and every item in section 5 has a verdict with a pointer to a log line, a code location, or a cited source, and one next experiment is named with a falsifiable pass/fail.

## 11. Read these first
1. `scripts/gantry/absorber-learning-diagnosis/DECISIONS.md` section 4 from "2026-10-01 (overnight session" to the end, including the morning report: every prediction and result.
2. `scripts/gantry/absorber-learning-diagnosis/f2_open.py`: the exact objective, detaching, rate switching and schedule.
3. `scripts/gantry/absorber-learning-diagnosis/CAUSE-AND-DIRECTION.md`: the summary; its tone overclaims ("the remedy that works"), read it critically.
4. `scripts/gantry/absorber-learning-diagnosis/aa_target_ceiling.py` and `outputs/aa_inn.log`: the evidence that the targets carry the mode.
5. `tasks/handoffs/2026-10-01-overnight-added-states-remedies.md` sections 2 and 10: the user's constraints and acceptance criterion.

## 12. Do not
- Launch any run or script; modify any file (section 2).
- Re-propose section 6 items or any init / parameterisation / band / pole prior as a remedy.
- Judge the added states by individual x_a coordinates (gauge); use poles at encoder points and output effect.

## 13. Operational
None needed for the review. If you cite numbers, take them from the logs listed in section 7 (conda env GraduationProject only if the user later asks for a run).

## 14. Delegation
None. Literature already gathered in `literature/added-states-training/` (three reports); read them only if a verdict depends on a source.
