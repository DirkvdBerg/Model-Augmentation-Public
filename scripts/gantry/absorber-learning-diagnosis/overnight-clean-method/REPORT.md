# Overnight clean-method search: report (2026-10-02/03)

Outcome: Candidate A (PS2, a two-stage predictive-state opening that leaves Hoekstra's S-DP model unchanged) passed
the gradient, collapse, added-state clamp and paired gantry gates: on held-out validation free runs it beat its exact
paired control in 3 of 3 seeds (median error ratio 0.28), the deployed model depends on its added-state memory, and no
quiet record degrades. Gate D was implemented and run, and it did NOT support generality: on a hidden-Duffing
benchmark where ordinary S-DP already learns the missing subsystem, PS2 was worse than S-DP on the test record in all
seeds. It is a gantry-supported thesis prototype, not a framework contribution. The nonlinear head beat the linear head
on amplitude extrapolation in the two runs where both completed, a hint only.

## 1. Gate A: frame, literature, candidates

**Decision to be made.** How a zero-initialised added state x_a in Hoekstra's S-DP augmentation gets (i) a target that
carries predictive information about the missing dynamics before the deployed transition can produce it, and (ii) a
deployed nonlinear transition g_aug that propagates it, under the known-controller closed-loop objective, with no
oracle quantity.

**Literature used (full text read this session unless marked).**
- Hefny, Downey, Gordon, *Supervised learning for dynamical system learning*, NeurIPS 2015, arXiv:1505.05310. Predictive
  state q_t = E[psi_t | h_t]; learning = S1A/S1B regressions from history features to future and extended-future
  features, "possibly non-linear" (p. 4), then S2 regression of the extended future on the predictive state (p. 4,
  Fig. 2 p. 2). Error bound of S2 in terms of S1 (Theorem 2, p. 5 to 6). Uncontrolled; controlled processes are named as
  future work (p. 8).
- Downey, Hefny, Li, Boots, Gordon, *Predictive State Recurrent Neural Networks*, NeurIPS 2017, arXiv:1705.09353.
  Learning = 2SR initialisation followed by BPTT refinement (Sec. 4.2, p. 5); "if we initialize PSRNNs ... using random
  weights, BPTT fails and we cannot learn a usable model" (p. 8); the behaviour of the initialisation is preserved by
  refinement (Fig. 4b discussion, p. 9). Consistency of the init only in the discrete realisable case (App. A); lost in
  the continuous setting (p. 5). Horizon 10 in its continuous experiments (p. 8), no rule given.
- Venkatraman, Rhinehart, Sun, Pinto, Hebert, Boots, Kitani, Bagnell, *Predictive-State Decoders*, NeurIPS 2017,
  arXiv:1709.08520. Training-only decoder F on the recurrent internal state, R = sum_t ||F(h_t) - phi(future obs)||^2
  (Eq. 4, p. 5), added to the task loss as L + lambda R; F affine, k in {2..10}, lambda tuned (p. 6). Data collected
  with controllers / policies (p. 7): closed-loop data, but no consistency claim.
- Already in `literature/added-states-training/`: Ni et al. ICLR 2024 (stop-gradient target prevents collapse, linear
  case, Thm 3 p. 6); Ouala et al. 2020, Forgione and Piga 2020, Bemporad 2023 (free latent trajectories / multiple
  shooting); LC report (closed-loop residual carries closed-loop poles; reference-based instruments remove noise bias
  only).
- Not read (metadata only): Hefny et al. 2018, *Recurrent Predictive State Policy Networks*, arXiv:1803.01489 (the
  controlled extension of 2SR + BPTT); Subramanian et al. 2022, JMLR, approximate information states.

**Candidate matrix (ranked before any new training).**

| | A: two-stage predictive-state opening (PS2) | B: predictive-state decoder on the rolled-out x_a | C: latent multiple shooting + encoder amortisation |
|-|-|-|-|
| Deployed model change | none | none | none (free latents are training-only) |
| Objective | S1: thesis + ||h(psi_a(past_k)) - e_k||^2 (h training-only); S2: ||g_aug(x_hat_k, u_k)_a - sg(psi_a(past_{k+1}))||^2 on a frozen snapshot; S3: thesis only | thesis + lambda sum_j ||F(x_a^roll(k+j)) - e_{k+j}||^2 throughout | thesis-type output fit over free x_a sequences + gamma ||x_a(k+1) - g_aug(.)||^2, then fit psi_a to the optimised latents |
| Mode-carrying signal | future closed-loop residual e (data, current model) | same e, through the rollout | output error only |
| Collapse defence | S1: e is a fixed (detached) target, x_a must predict it; S2: target detached and frozen (Ni Thm 3 analogue) | e anchors F(x_a); no stop-gradient issue | none at zero readout (x_a = g = 0 is a stationary solution) |
| Closed-loop assumptions | known controller in the residual rollout and in S3; e carries closed-loop poles (LC Sec. 0); S3 maps back | same | same |
| Nonlinear support | S1 regression "possibly non-linear" (Hefny p. 4); nonlinear head and encoder rows; g_aug nonlinear | yes (RNNs in PSD) | yes |
| Zero-step gradient path | S1: head gets -e x_a^T, psi_a through head (first order once head moves); S2: g_aug supervised, first order | F learns from x_a(0) only; g_aug first order through F once F moves | free latents: zero (readout dy/dx_a = 0 exactly); g_aug: only toward the latents |
| System-specific choices | none intended: M = encoder past window; switch on calibration plateaus (heuristic tolerances) | k, lambda (tuned in PSD) | gamma, segment length |
| Literature guarantee | linear realisable case: 2SR consistent (Hefny Thm 2 bound; linear); nonlinear: none, PSRNN reports empirical benefit only | none (empirical) | none for this setting |
| Cheapest falsification | one seed, 600 updates, paired with the existing exact control | one seed vs control | analytical (below) |
| Rank | 1 | 2 | rejected at Gate B |

**Hyperparameter rules declared before the first run.** M = the encoder's past output window (30 samples; equal past and
future horizons as in subspace identification; framework quantity, not the 40 of E5/E9). Auxiliary loss scale:
irrelevant by construction (the auxiliary terms have their own Adam optimiser, which is invariant to the loss scale at
eps 1e-16), so no loss-matching weight. Learning rates: auxiliary and S2 fits at Adam's default 1e-3; thesis rates
those of the existing U1 control (added-row output 1e-3, rest 1e-5), identical in both arms. Switches: plateau of a
data-only quantity on two held-out training records (HEURISTIC tolerances 2 % / 1 %, caps 350 updates / 8000 steps);
poles never inspected. Limitation: the tolerances and caps have no literature value.

## 2. Gate B: mechanism analysis at the zero initialisation

Notation: z = [x_b, x_a, u]; ANN output layer zero, so f_aug = 0, g_aug = 0 at update 0; psi_a(past) = W^a past + 0.

**A (PS2).**
1. psi_a before the model reads x_a: S1 gradient d/dW^a ||h(psi_a) - e||^2 = J_h^T (h - e) past^T; zero at update 0
   (head output layer zero) and first order after one head step (the head's output layer gets -e h1(x_a)^T, nonzero
   because x_a = W^a past is nonzero and random). So psi_a is trained by S1 from update 2 on, independent of g_aug.
2. Against a constant x_a: e is a fixed detached target with zero conditional mean only if unpredictable; a constant x_a
   leaves the head unable to beat the mean, so the S1 loss strictly prefers any x_a correlated with E[e | past]. The
   constant solution is not stationary while E[e | past] varies.
3. g_aug learns to propagate: S2 is a supervised regression with a frozen target, first-order in the output layer and
   hidden layers; no rollout, no pole gradient needed (the dynamics-learning problem is converted into regression, the
   point of 2SR).
4. Detached / fixed: e (detached, recomputed from the current model each update); S2 inputs and targets (frozen
   snapshot); the thesis optimiser never sees the auxiliary gradients.
5. Bypass of x_a by f_aug(x_b, u): possible in S3, as in any S-DP training. It is measured by the clamp test, which is
   the acceptance criterion; S1/S2 do not prevent it.
6. The known-controller closed-loop rollout is the optimised object throughout (S1 and S3); e itself comes from the
   closed-loop rollout.
Not rejected.

**B (decoder on the rollout).** Through the rollout x_a(j) = g_aug(z(j-1)) = 0 for j >= 1 at update 0, so F learns only
from x_a(0) (encoder) and g_aug first-order receives d/dg ||F(x_a(j)) - e_j||^2 once F is nonzero. No exact dead path.
But the transition is then learned through BPTT from a far-away start, the regime in which plain output error failed
here (L1, W1, X1) and in which PSRNN reports that random-init BPTT fails (p. 8); its only difference from the failed
output-error path is a denser target. Ranked second; not run (budget).

**C (multiple shooting).** With f_aug = 0 the output does not depend on x_a, so the free latents receive no output
gradient (exact dead path), and the consistency term alone is minimised by x_a = g_aug = 0 (unopposed constant
solution). Making it live needs a temporary readout (which turns it into A with free latents instead of an encoder) or
a live initialisation (excluded). Rejected at Gate B.

## 3. Gate C: runs (all predictions written before launch in `RUNS.md`)

Six training starts, as budgeted: P1, P1b, P2, C2, P3, C3; the seed-0 control is the existing exact pair `u_u1ctl600`
(same init, batch sequence and rates; updates 1 to 2 reproduce its losses bit for bit). Common: run 42 settings, batch
64, 600 updates, thesis rates of the U1 control (added-row output 1e-3, rest 1e-5) in both arms, final checkpoint at
the fixed budget (no selection), CPU.

**P1 failed its screen, and why.** Calibration on training records TR-T3/T4 (chosen blind as "the last two") stayed at
0.97 to 0.99 unexplained while training batches reached 0.71. Two data-only diagnostics, readings written first: D1
(full-rank linear) showed the residual IS predictable across records including the T class (0.26 / 0.34); D2 (rank-2
linear, the 2-d predictive state) showed that no 2-d state explains anything on the T class (unexplained 1.00, even on
T1/T2 inside the fit set): the T class is the no-multisine dwell class with residual 0.3x the others. The screen
therefore measured an unrepresentative calibration class. P1b reran the same seed with one held-out record per class
(TR-S5, Y3, P4, T4). This is a post-failure design correction, reported as such.

**Gantry results (held-out validation, closed-loop free run over whole records, metres).**

| Seed | S1 cal unexplained (at cap 350) | S2 cal EE residual | Eval loss cand / ctrl | Val sim-RMS cand / ctrl | Clamp x_a: cand error | Paired clamp diff CI | Post-run pair (diagnosis) |
|-|-|-|-|-|-|-|-|
| 0 (P1b / C0) | 0.605 | 0.0087 | 1.30e-9 / 6.91e-9 | 3.12e-6 / 1.145e-5 | 1.49e-5 | excludes 0 | 211 Hz, zeta 0.048 |
| 2 (P2 / C2) | 0.652 | 0.0081 | 5.14e-9 / 6.87e-9 | 9.39e-6 / 1.148e-5 | 1.29e-5 | includes 0 | 203 Hz, zeta 0.33 |
| 3 (P3 / C3) | 0.657 | 0.0067 | 1.22e-9 / 6.84e-9 | 3.20e-6 / 1.140e-5 | 1.49e-5 | excludes 0 | 208 Hz, zeta 0.07 |

Truth absorber (quoted only after all runs): 211.9 Hz, zeta 0.043. Block-off sim-RMS 1.67e-5 in all runs.
- Criterion 5 (beats paired control on held-out free run in >= 2 of 3 seeds, median improved): met, 3 of 3, median
  ratio 0.28 (per-run 0.27 / 0.82 / 0.28).
- Criterion 6 (clamp CI over records excludes 0): met in all three. Stricter reading added before P1's evaluation (the
  control itself has a 1.5 % clamp effect): the candidate-minus-control clamp difference excludes 0 in 2 of 3.
- Criterion 7 (quiet records): no candidate record above block-off in any seed (C0 has two); P2's quiet record 5 is
  1.4 % worse than its control's (on/off 0.911 vs 0.898). No failed or unstable seed; all runs reported.
- Mechanism as designed in all three seeds: after S2 the added pair is at the residual's closed-loop frequency
  (233 / 206 Hz, zeta 0.16 to 0.17), and the thesis loss moves it to the plant (LC Sec. 0 prediction). Seed 2's pair
  stays overdamped in the short S3 (the E6/E8 pattern of the earlier work).
- Every S1 ended at its 350-update cap with the calibration fraction still falling about 1 % per 25 updates. The cap,
  a heuristic, decided the switch in all runs; the plateau rule never fired.

## 5. Gate D: nonlinear missing-dynamics benchmark (specification as written before the run; results at the end)

**System.** Jan's 3-DOF chain `Msd_ndof` (`model_augmentation/systems/mass_spring_damper.py`), unchanged code, with
m = [0.5, 0.4, 0.1], k = [100, 100, 100], c = [0.5, 0.5, 0.5], dt = 0.02, input on mass 1, output position of mass 2,
and a = [0, 0, a3] with a3 > 0: the omitted third mass is a hidden Duffing oscillator,
m3 q3'' = -k3 (q3 - q2) - c3 (q3' - q2') - a3 (q3 - q2)^3 (class lines 149 to 153). Its INTERNAL transition is
nonlinear in the hidden state (q3 - q2, q3'); the hidden natural frequency depends on amplitude (hardening backbone).
This differs from Jan's benchmark a = [0, 100, 0], whose omitted mass is linear and whose nonlinearity sits in the
retained spring. The middle cubic is set to 0 so that the ONLY mismatch is the hidden nonlinear subsystem.
**Nonlinearity level (truth side, before any training).** a3 is chosen on a noise-free pre-simulation of the training
input so that the RMS cubic force a3 RMS((q3 - q2)^3) is 0.5 of the RMS linear force k3 RMS(q3 - q2). This uses truth
only in data generation, never in training or selection.
**Baseline.** The ideal linear 2-DOF model `data/mass_spring_damper/msd_2dof.mat` (masses 1 and 2, k, c as above),
parameters frozen (as in the gantry negative control) and, in a second arm, jointly estimated.
**Data.** Multisine as `scripts/ecc_2025/msd_ndof_data_generation_dynamic.py` (pmax 4999, crest-factor optimised), new
random phases per set: train 2 periods of 10000, validation 1 period, test 1 period, plus an amplitude-extrapolation
test at 1.5x amplitude (the hidden frequency shifts there). Output noise at Jan's SNR 30 (sigma 5.2e-3). Training-
internal calibration = the last 2000 samples of the second training period (non-overlapping with the windows used to
fit S1/S2); validation and test only for reporting.
**Arms (identical model: S-DP, zero-init 2x16 tanh MLP, n_a = 2, Jan's encoder).** (1) ordinary S-DP; (2) PS2 with the
MLP head; (3) PS2 with the linear head; 3 seeds each, same init draws and batches across arms.
**Rules carried over unchanged.** M = encoder past output window; S1/S2 plateau rules and caps of `ps2.py`; auxiliary
and S2 fits at Adam 1e-3; thesis-phase rates identical across arms. No truth state, a3, or hidden frequency enters
training, switching or selection; final checkpoint = fixed budget.
**Predictions.** (2) and (3) beat (1) on test free-run NRMS in at least 2 of 3 seeds; (2) beats (3) on the
extrapolation test (a linear-head predictive state is the best LINEAR 2-d summary, which cannot represent an
amplitude-dependent hidden frequency; the nonlinear g_aug can, if the state it is taught is a nonlinear coordinate).
If (3) equals (2) everywhere, the head nonlinearity is not needed and the claim narrows to "linear predictive target,
nonlinear transition".
**Pass criteria.** As the gantry: median and 2 of 3 seeds better than ordinary S-DP; clamp-x_a test with a block
bootstrap over contiguous test segments of 500 samples (well above the encoder window), median-increase CI excluding
0. Post-selection diagnosis only: learned x_a amplitude-frequency relation vs the truth backbone.
**Cost.** Jan's MSD runs in minutes per arm on CPU (nf 200, small data); 9 runs plus evaluation is about 1 to 2 h.
**Commands (to build).** A data script `gate_d_data.py` (Msd_ndof as above, saves npz to `outputs/gate_d/`), and a copy
of `ps2.py`'s S1/S2/S3 loop around `msd_ndof_interconnect_dynamic.py`'s model construction (`FP_type="ideal"`,
`dynamic_aug=True`, parallel, nonlinear MLP). Built as `gate_d.py` (single script, Jan's classes unchanged), run
03:00 to 04:45 after the gantry chain. Deviation from this text: one fit record A plus a separate calibration record C
(new phase), instead of a calibration segment inside A (A's two periods are identical, so an in-record segment would
not be held out in phase).

**Results (`outputs/gate_d.log`, `RUNS.md`).**

| Free-run RMS (sigma 5.2e-3) | seed 0 | seed 1 | seed 2 |
|-|-|-|-|
| test, sdp | 6.61e-3 | 6.39e-3 | 6.68e-3 |
| test, ps2_mlp | 8.65e-3 (S2 0.275) | 1.11e-2 (FAIL-SCREEN 125) | 7.54e-3 (S2 0.082) |
| test, ps2_lin | 8.10e-3 (S2 0.232) | 1.11e-2 (FAIL-SCREEN 125) | 9.22e-3 (FAIL-SCREEN 150) |
| 1.5x, sdp | 1.91e-2 | 2.47e-2 | 1.56e-2 |
| 1.5x, ps2_mlp | 1.78e-2 | 3.92e-2 | 1.12e-2 |
| 1.5x, ps2_lin | 2.50e-2 | 3.93e-2 | 3.71e-2 |

Prediction 1 refuted (0 of 3 for both heads). Prediction 2 met in the two seeds where the MLP head completed. Reading:
this benchmark does not have the gantry's failure: ordinary S-DP learns the hidden subsystem (its clamp increase CI
excludes 0 in every seed). Here the residual of the current model shrinks toward noise during S1 (screens fail at
0.92 to 0.98), and S2 fits the encoder's states poorly (0.23 to 0.28 in two of three), so PS2 costs training time
without a target to supply. Two design consequences: a failed screen should fall back to plain training (PS2 then
reduces to S-DP) instead of stopping, and the MECHANISM gate should also gate S2's implant. Neither was tested.

## 6. Critique of the two earlier proposals (not edited)

- `IMPLEMENTATION-PLAN.md`: right structure (target, fixed-target state-map fit, output-error refinement), but its tested
  form B1 keeps the 100x / 1000x rate factors and fixed counts, M = 40 is a gantry-informed choice, and its loss weights
  are matched by hand. PS2 replaces B1 by its own option B2 in a stronger form (a supervised S2 solve on a frozen
  snapshot, not inner steps on a moving target), removes the weights (separate optimiser), takes M from the encoder,
  and switches on held-out training records. Its switch thresholds were never tested; ours were, and the S1 cap, not
  the plateau, fired in every run, so that part of its idea is still unsettled.
- `NONLINEAR-PREDICTIVE-STATE-IMPLEMENTATION-PLAN.md`: rejected as in the handoff. Output simulation plus an
  encoder-consistency term has no mode-selecting signal beyond the failed output-error path, admits the collapsed
  solution it itself names (Sec. 5.3), replays recorded inputs instead of the closed loop, and changes the deployed
  network. Useful parts kept: the record-level calibration split (its Sec. 4.2; our P1 showed the split must also be
  class-representative), the "auxiliary terms leave no trace in the deployed model" requirement, and its Sec. 9.4
  generality list (Gate D covers item 2 only).

## 7. Production integration plan (no monkey patching)

Target: Jan's framework objects only, one new class, the thesis entry script calling one extra method.
1. `model_augmentation/fit_systems/ps2_opening.py` (`__project_origin__ = "added"`): class
   `PS2Opening` holding the head, its optimiser, the calibration record indices and the switch rules; methods
   `s1_step(fit_sys, batch)` (returns the auxiliary gradients for the encoder's added-state outputs and the head),
   `s2_fit(fit_sys, fit_records, cal_records)` (snapshot fit of the ANN on the added rows), `done`.
2. No layer split. The added-row output is addressed by row index of the existing final Linear (rows
   `nx_b:nx_b + n_a` of `Static_ANN_Block.net.net[-1]`), and the encoder's added-state outputs by row index of
   `W^a` and of the NL net's final layer; masking is applied to the auxiliary gradients only (as now), so the thesis
   optimiser and the model code are untouched. The experiment's `RowSplitNet` is a probe convenience, not needed.
3. `scripts/gantry/gantry_interconnect_dynamic.py` gets one marked call: during the first epochs the training loop calls
   `opening.s1_step` after each thesis step until `opening.done`, then `opening.s2_fit` once, then continues as today
   (Adam, L-BFGS polish, free-run checkpoint selection unchanged). Implemented through deepSI's existing per-batch
   callback seam if it exists, otherwise via `SSE_Interconnect_Composed.loss`'s training-only branch (as the E3
   `InnovationOpening` class already does), never by replacing methods on instances.
4. OBC arm: the auxiliary terms and S2 touch only the added rows (the OBC basis has zero rows there,
   `obc_projection.py`), so it composes; untested.
5. Tests: `test_ps2.py` (update-0 equality, identical thesis gradient, physical / x_b / physical-row parameters
   untouched by the opening, S2 changes g_aug, split selection), ported to the class.
6. Budget at server scale: S1 at most 350 updates + S2 (minutes) of about 26k updates; the cap must be re-examined
   (Section 9).

## 8. Outcome level against handoff Section 10

Gantry-supported prototype, items: 1 (objective from Hefny 2015 S1/S2 regression and PSRNN 2SR-then-BPTT; new heuristic
parts: the nonlinear head on an encoder state, the closed-loop residual as the future feature, the snapshot S2 on the
deployed shared network, the switch tolerances and caps); 2 (no oracle quantity in training, switching or
checkpointing; poles read only after all runs); 3 (production known-controller closed-loop simulator); 4 (Section 2);
5 (3 of 3, median 0.28); 6 (clamp CIs exclude 0; stricter paired version 2 of 3); 7 (no candidate record above
block-off; one quiet record 1.4 % worse than its control); 8 (Gate D specified, and run); 9 (Section 9). All met.
Framework candidate: not met (Gate D Prediction 1 refuted).

## 9. Limitations

- Closed loop: the S1 target is the closed-loop residual; it carries closed-loop poles (post-S2 pair 233 / 206 Hz) and,
  on measured data, feedback-correlated noise (LC report). Only the S3 thesis loss maps the pair back to the plant; this
  is observed (211 / 203 / 208 Hz), not proven. No reference-based instrument is used.
- Identifiability and gauge: x_a is any invertible transform of the mode state; only the learned eigenvalues at encoder
  points and the clamp effect are meaningful. With a nonlinear head the state is identified at best up to a nonlinear
  map; the cited guarantees are linear (Hefny Theorem 2; PSRNN consistency only in its discrete case).
- Excitation: the predictive state is class-dependent (D2: no 2-d state transfers to the no-multisine T class), so the
  calibration split must be class-representative; P1 shows a blind split can stop the method.
- Order: n_a = 2 is the thesis setting, not selected by the method.
- Heuristics that decided outcomes: the S1 cap (350) ended S1 in every gantry run; the stratified split was introduced
  after a failure; S3 used the U1 rates (added-row output 1e-3), a stand-in for the 26k-update server schedule.
- Seeds and scale: 3 local seeds, 600 updates, batch 64; seed 2 kept an overdamped pair (zeta 0.33). Nothing at server
  scale, nothing on the OBC arm.
- Generality: not shown. The only nonlinear test (Gate D) is a case where plain S-DP already works; PS2 did not help.
- Novelty: the opening transfers predictive-state two-stage regression (Hefny 2015; Downey 2017) into Hoekstra's S-DP
  training; Predictive-State Decoders (Venkatraman 2017) is the closest training-only-head precedent. No source found
  applies it to grey-box augmentation, closed-loop data, or a zero-initialised added state (not a systematic search).

**Single most informative next experiment.** A benchmark where ordinary S-DP demonstrably fails for the gantry's reason
(a weak missing mode under feedback) and the missing transition is genuinely nonlinear: e.g. the gantry simulator with
its absorber spring made cubic (a new dataset folder), PS2 with the fall-back rule vs ordinary S-DP, 3 seeds. It
separates "PS2 rescues a dead start" from "PS2 helps only linear modes".

## 10. Supervisor-ready method statement

**Gap.** In Hoekstra et al.'s dynamic-parallel augmentation (arXiv:2602.17297, Table 1, S-DP) with a zero-initialised
learning function, the added states start dead: the simulation loss gives them no first-order gradient toward a missing
mode, and on the gantry benchmark plain training never learns the missing absorber.

**Proposal (training only).** Two supervised stages from predictive-state learning (Hefny, Downey, Gordon 2015; Downey
et al. 2017), then ordinary training:
1. S1: alongside the unchanged loss, train a temporary head h and the encoder's added-state outputs on
   min ||h(psi_a(past_k)) - e_k||^2, where e_k is the current model's next M closed-loop output residuals (detached)
   and M is the encoder's past window.
2. S2: fit the deployed added-state transition to the frozen encoder states,
   min ||g_aug(psi(past_k), u_k) - psi_a(past_{k+1})||^2, a one-step regression on a snapshot.
3. S3: discard h; continue with the unchanged closed-loop simulation loss.
Auxiliary terms have their own optimiser, so the training optimiser sees only the plain training-loss gradient (checked: `test_ps2.py`, all pass); switches
use held-out training records only.

**Unchanged.** Model class, network, zero initialisation, encoder, routing, closed-loop objective, parameter
regularisation, checkpoint selection; the final model has no extra parameters.

**Evidence.** Gantry, 3 seeds, exact paired controls, held-out validation free runs: 3.1e-6 / 9.4e-6 / 3.2e-6 m vs
1.15e-5 m for plain training; removing the added-state memory returns the error to about the no-augmentation level;
the learned pair (diagnosed afterwards) is at 211 / 203 / 208 Hz vs the true 212 Hz. On a hidden-Duffing MSD benchmark,
where plain S-DP already works, no gain.

**Not yet supported.** Generality to nonlinear missing dynamics; any benefit where plain S-DP works; consistency on real
closed-loop noisy data; server-scale and OBC behaviour; a principled S1 switch (the cap decided every gantry run); order
selection.

## 11. Artefacts

`ps2.py` (prototype), `evalc.py` (held-out free run, clamp, bootstrap), `d1_predictability.py`, `d2_rank2.py`,
`d3_poles.py` (diagnostics), `gate_d.py` (nonlinear benchmark), `test_ps2.py` (focused checks), `chain_*.cmd`, `RUNS.md`
(predictions and outcomes), `outputs/` (logs, checkpoints, histories). Decision D-234; run row in the problem log
Section 12.

## 12. Follow-up 2026-10-03: cubic absorber (plain S-DP fails under feedback, nonlinear missing transition; D-235)

Roles kept separate (user discussion 2026-10-03): training uses no absorber knowledge; success = held-out input-output
generalisation; local poles are diagnosis only, read after all decisions.

**Set-up.** The gantry truth with a hardening absorber spring, ka*delta + ka3*delta^3, ka3 = 2.41e14 N/m^3 (truth side:
RMS cubic force 0.5 of the linear on the linear truth's stroke; on the generated data 0.33 at RMS and 1.0 to 1.25 at peak
stroke; output differs from the linear set by about 1e-5 m RMS). 12 records (TR-S1..3, Y1..3, P1..3, VA-S1, Y1, P1), noisy,
plus test records E1, I1, I3, E3, E4, all generated with the thesis generator's verified replica and a copied ODE
(`cubic-absorber/`), in `Thesis-writeup/Data/Cubic-absorber-test/`. MATLAB `-nojvm`, about 1 min per record, no crash.
Same ps2.py rules as Section 3 plus the fall-back rule (a failed screen continues with plain training; never triggered).

**Gate 1 (does plain S-DP fail?).** By its written rule it did NOT fail: plain S-DP's clamp effect is 42 % of its error
(rule: < 5 %). Diagnosis afterwards: its added states have only tiny, heavily damped poles (|z| 0.19), i.e. a fast
nonlinear correction routed through x_a, not the absorber. Plausible reading (unverified): the quasi-static absorber force
gains an a^3 term under the cubic spring, which a fast static map can partly represent. Proceeding with PS2 anyway was a
deviation decided after seeing the result, stated in `RUNS.md`.

**Validation (3 records, paired, fixed budget).**

| Seed | PS2 | Plain S-DP | Ratio | Paired clamp difference |
|-|-|-|-|-|
| 0 | 7.17e-6 | 1.29e-5 | 0.55 | excludes 0 |
| 2 | 7.47e-6 | 1.43e-5 | 0.50 | excludes 0 |
| 3 | 7.16e-6 | 1.32e-5 | 0.54 | excludes 0 |

No record above block-off in any run.

**Generalisation (test records, never seen in training or evaluation before).** Whole-record ratios 0.66 / 0.51 / 0.56.
Per record (I1 interpolation, I3 unseen route class, E1 standstill, E3 100 % acceleration = larger stroke, E4 long
strokes at higher velocity): PS2 better than plain in 3 of 3 seeds on I1, I3, E1, E3 and in 2 of 3 on E4 (seed 0 1.07x).
Amplitude: on E3 the advantage is smaller than on I1 in 2 of 3 seeds (0.87 vs 0.60, 0.63 vs 0.54; seed 2 0.51 vs 0.56),
so part of what PS2 learned behaves like an averaged mode; no evidence that the hardening itself was captured. Plain S-DP
is worse than block-off on several unseen route records; PS2 is not, except seed 0 on E3 (1.04x). The added-state memory
carries PS2's gain on multisine-excited records; on route records clamping it changes little.

**Diagnosis (after all decisions).** Post-run local pairs: Q0 229 Hz (zeta 0.06), above the linear set's 208 to 211 Hz,
consistent with a stiffer effective spring; a local linearisation, not evidence of amplitude dependence. D4 rollout
consistency (`d4_rollout_consistency.py`): plain S-DP's map and encoder share no coordinate (relative error about 1.0
from one step); PS2's agree at one step (0.12 to 0.20) and drift by 10 to 30 steps (0.5 to 1.1). Without a noise-floor
reference this is not a Markov-state failure test; the 12 s free runs are the stronger evidence. D4 does not explain the
linear set's partial seed 2.

**Updated outcome.** On the gantry, PS2 beats plain S-DP on held-out data in 3 of 3 seeds for the linear absorber (median
ratio 0.28) and in 3 of 3 for the cubic absorber (validation median 0.54, test median 0.56), including unseen route
classes and a larger-stroke record. It remains a gantry-only result (one plant, one controller, n_a = 2 equal to the
absorber's order, local scale). The cubic case shows the method works when the missing transition is nonlinear; it does
not show that the learned transition represents the nonlinearity. Next informative test: an order sweep (n_a 1, 2, 4)
on the cubic set, and a server-scale run.
