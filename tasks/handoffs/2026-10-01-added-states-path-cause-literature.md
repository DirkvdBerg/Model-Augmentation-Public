# Handoff: determine what stops the added states learning the absorber, and find literature-backed remedies within Jan's framework as is
**From**: session of 2026-10-01 | **Branch**: Augmentation | **Effort suggested**: xhigh (causal analysis across ~20 local measurements plus a literature sweep; the method is yours)

## 0. How to read this prompt
- **The user's words (2026-10-01, spelling cleaned):** "look into the current issue, and look for solutions online"; "describe the exact issue we are facing"; "look at your results and try to determine what is creating the issue"; earlier the same day: "critically analyze what is causing our model not to learn"; "it is learning but not the right dynamics"; "the goal of this approach is to be able to extend it to real data, and we want the capabilities of our model to be sufficient to capture actual dynamics without me having to push it that way. But a generalized approach."
- **Task type:** diagnosis (from the existing measurements, plus at most the small local measurements you need) AND a literature search for remedies. The deliverable is a defended cause and a ranked list of remedies that satisfy section 2, each with its source (page / equation) and the local measurement that would test it.
- **Confirmed user decisions** are marked (user). Everything else in sections 4 and 5 is the previous session's reading: treat it critically and re-measure anything that looks wrong; that session made errors (section 6 lists them).

## 1. Task
Using the results in `scripts/gantry/absorber-learning-diagnosis/` (`DECISIONS.md` section 4, `CAUSE-AND-DIRECTION.md`, `outputs/`), determine what makes gradient training of the thesis augmentation model learn the WRONG dynamics in its added states: they learn a delayed static map or fast real lags, never the payload absorber's lightly damped resonance (212 Hz, zeta 0.043), although a member of the same model class scores 7x lower on the training loss. Then search the literature (control: grey-box / output-error / adaptive IIR / SUBNET / LFR augmentation; ML: recurrent and state-space model training, curricula, optimisation) for remedies that stay within the user's constraints (section 2), and rank them by how directly they attack the cause you defend. Finish by correcting `CAUSE-AND-DIRECTION.md` (section 3 of it is mislabelled, see section 6 here) and appending to `DECISIONS.md`, and propose exactly one next local measurement and one server experiment (prediction, pass/fail, cost) for the user to decide on.

## 2. Out of scope (user, repeated several times on 2026-10-01; absolute)
- **Any initialisation or parameterisation change** of the added states: identity / residual skip, random linear part (Hoekstra Sec. 5.4.3), residual- or BLA-based init (tried by the user before in `scripts/gantry/` BLA folders), band or pole init, complex / polar / oscillator / resonator parameterisations, banks of modes. They may appear only as labelled diagnostics of the mechanism, never as a remedy. The user calls them "monkey patching" and "too biased / known dynamics".
- **Penalties that say what the static path may learn** (no subspace exists without system information; see section 4).
- **Burn-in in the loss: never** (user).
- Changing the data or the multisine band; editing the main pipeline (`scripts/gantry/gantry_interconnect_dynamic.py`, `scripts/gantry/gantry_dynamic/`, `model_augmentation/`): copy into the diagnosis folder; `kamtin-fp-model/`, `Thesis-writeup/`: read only.
- Server jobs: propose, never launch. Do not touch the thesis campaign (array 86892) or run 87195.
- Allowed remedy families (user): optimiser, objective (horizons, weighting, multi-RATE as the supervisor suggests, see section 8), training schedule / order, encoder. A remedy must keep Jan's model, Static_ANN_Block, routing and zero-init output unchanged, equal the baseline at start, use no system-specific information, work with joint estimation of the physical combinations and with the OBC projection (or state and argue the incompatibility), and be implemented as a copied script with proper classes in the diagnosis folder.

## 3. Where things stand
- Branch Augmentation, tree dirty (many unrelated files; do not commit unless asked).
- No local job running. Checkpoint ready for the planned measurement Z2: `scripts/gantry/absorber-learning-diagnosis/outputs/u_early150.pt` (zero-init thesis model after 150 updates at U1 settings; eval loss 7.1188e-9, reproduces U1). Script ready, not run: `y2_valley_early.py early150 100 64` (prediction NOT yet written to `DECISIONS.md`: the write was rejected when the user paused the session).
- Not yet logged in `DECISIONS.md` (write them first): (a) user correction: the supervisor's suggestion is MULTI-RATE (same window length nf, data at several sampling rates, so a window covers more seconds), not the multi-horizon variant M1 tested; (b) OBC is a projection in the forward pass (`model_augmentation/fit_systems/interconnect.py` forward, `xp = xp - _obc(x, u, obc_pass)` near l.170), not a penalty; (c) option "penalise the static correction" dropped.

## 4. Established and verified (ids in `DECISIONS.md`; run 42 = thesis U arm, n_a 2, seed 1, unless stated)
- **Setup.** Truth = FP gantry + tanh Coulomb rail friction + payload absorber (coupled pole 211.9 Hz, zeta 0.043, |z| 0.986 at 4 kHz). Model: reduced 10-combination LPV physics block (joint estimation, 10 % detuned) + Static_ANN_Block (2x16 tanh MLP, zero-init output layer) taking z = [x (6), x_a (2), u (3)], writing the routed physical rows additively and ALL added rows. The interconnect starts xp from zeros and sums the blocks (`interconnect.py` forward, `gantry_dynamic/model.py` l.138 to 174), so the added states' A_aa = dF_a/dx_a, B_a = dF_a/du, readout C_a = dF_phys/dx_a are the MLP's Jacobians (zero at init). Closed-loop residual rollout with controller K1, nf 400 (0.1 s) at 4 kHz, float64, Adam lr 1e-5 eps 1e-16, about 33k updates; no burn-in.
- **No learned resonance:** 0 of 672 encoder points with a pair in 95 to 380 Hz in runs 42, 44, 48 (H1, H1b). Run 42's dominant added pole is +0.60 (a real lag), CORRECTED from H1's "-0.60" (Z1; `h_learned_poles.py` l.142 to 144 sorted moduli and angles separately).
- **Objective and class reward the absorber:** truth with absorber state zero 1.35e-9 vs model 5.09e-9 (G1); class member with the true resonance 7.0e-10 (K1).
- **Early vs late value of a resonance (the central contrast):**
  - X1 (diagnostic, on the U3 identity-skip model after 300 updates): fixing the added self-map at a band pole and re-fitting the rest for 100 updates gives 2.12e-9 (212 Hz) vs 5.21e-9 for the learned real-lag solution; 150 / 260 Hz about 3.9e-9 to 4.0e-9.
  - Z1 (run 42 end point, unchanged model): pushing the added self-map toward 0.95 / 212 Hz in steps s and re-fitting 100 updates gives the same loss to +-1 % up to s 0.75 and +7 % at s 1: a flat plateau, no downhill path.
  - W1, O1, H2: with drive and readout FIXED, moving the pole toward the band is uphill (W1: 1.8x to 4.6x).
- **Horizon (H2, run 42, drive/readout fixed):** loss(s 0.5)/loss(0) = 1.126 / 1.334 / 1.443 / 1.559 / 1.624 at nf 25 / 50 / 100 / 200 / 400; full move (s 1) 3.7x to 4.5x uphill at every nf.
- **Optimiser and objective probes on the unchanged model, all negative:** U1 (added-row output at lr 1e-3): added radius 0.021 at 150 updates, real; U4 (hidden layers too at 1e-3): no faster (0.008 at 75 vs U1 0.010); M1 (multi-HORIZON 25/100/400 at 4 kHz, short first, 300 of 600 updates): radius 0.015, real, loss 1.6 % below U1.
- **Diagnostics that did move the poles** (all excluded as methods, section 2): U3 identity skip -> slow REAL lags 0.78 / 0.96, loss -30 % vs U1, learned 2x2 self-map nearly symmetric; Y1 polar pair from 100 Hz -> frequency rose to 128 Hz (139 Hz realised) in 75 updates; U2 random linear part -> pole did not move.
- **Literature already in hand:** Ribeiro et al. 2020 (Automatica 121:109158) Thm 1 p.4 Eqs. 8 and 9; Hoekstra 2026 (arXiv:2602.17297) Sec. 5.4.3 p.9 to 10; Q2 sweep (this session): AntisymmetricRNN (arXiv:1902.09689), LRU (arXiv:2303.06349), Hayes et al. 2023 (arXiv:2210.14476), LinOSS (arXiv:2410.03943), Zucchet and Orvieto 2024 (arXiv:2405.21064), Soderstrom and Stoica 1982 and Nayeri/Jenkins, Shynk 1989 (second-hand via Netto 1996 PhD, UVic DSpace 1828/6337). Most of these are parameterisation remedies, excluded by section 2; their diagnostic content (why OE / dense recurrences learn real lags, saddles in non-direct realisations) is still usable. Multi-rate literature: `literature/multi-rate-horizon/deep-research-multi-rate-loss-ladder.md` (section 0 bottom line).

## 5. Assumed but not verified (the previous session's reading; test, do not adopt)
- Reading: "from zero init the gradient spends the added states on drive (delayed static map) because the static / mass-like part of the absorber correction gives the largest first-order loss reduction; a resonance pays only as a coordinated pole + drive + readout move, and its value disappears as the static correction grows (X1 early vs Z1 late)." Weak points: X1's early point is on a CHANGED model (U3), not the thesis model (Z2 tests the unchanged one); Z1's re-fit can partly cancel the probe through the MLP's own added rows at small s; every training probe is short (<= 300 updates, batch 64) against 33k thesis updates.
- "No in-framework optimiser or objective opens the route" rests on short probes; M1 did not reach its long-horizon phase; multi-RATE is untested.
- Whether an early curvature optimiser (Gauss-Newton / Levenberg-Marquardt) could make the coordinated move while it still pays: open (Z2 decides whether the early path is a valley).
- Whether encoder learning speed matters (x_a informative from the start -> gradient on A_aa = adjoint x x_a^T would favour memory): untested.

## 6. Tried and failed
- Zero-init added rows at 100x lr (U1) -> radius 0.02 at 150 updates, real -> gradient asks for drive, not memory -> `outputs/u_lrA1e-3.log`.
- Hidden layers also at 1e-3 (U4) -> no faster -> not conditioning but direction -> `outputs/u_lrH1e-3.log`.
- Multi-horizon at 4 kHz (M1) -> no change in poles -> NOT the supervisor's suggestion (multi-RATE); `CAUSE-AND-DIRECTION.md` section 3 wrongly proposes it as such: correct it -> `outputs/u_mh.log`.
- End-stage curvature idea -> Z1 plateau at run 42 -> nothing to descend late in training -> `outputs/y_valley42.log`.
- Earlier (previous sessions): two-phase / static net gated (N1, O1: added states imitate static with a delay); supervised one-step fit (I1, fitting failure); partial graft (M1/M1b, construction error); lr 1e-3 on all nets in the closed loop (R1, diverged at update 2); band-pole install and B_a injection (D-150, D-151, problem log section 15: the gradient damped the injected mode); multiple-shooting defect (collapsed the added states, problem log 15.2 item 3).
- Previous session's own errors to avoid repeating: proposed init / parameterisation remedies after the user excluded them; overstated "Adam cannot see the coupling" (wrong: by the envelope theorem the gradient follows the profile slope locally when drive and readout are near-optimal); misread the supervisor's multi-rate idea as multi-horizon; called OBC a penalty.

## 7. Achieved
Validated instruments in `scripts/gantry/absorber-learning-diagnosis/`: `u_probe.py` (short training with continuous pole tracking; env U_LR, U_LR_A, U_LR_H, U_INIT, U_MH), `v_inspect.py`, `w_rot_scan.py`, `x_profile.py`, `y_valley.py` (Z1), `y2_valley_early.py` (Z2, ready), `z_horizon_cut.py` (H2), `run.cmd` (detached launcher), plus the earlier `h_learned_poles.py` (note its sorting bug, section 4), `g_window_loss.py`, `k_linear_member.py`, `l_pole_scan.py`, `m_graft.py`.

## 8. The open question
What, mechanically, makes the gradient choose drive over memory from zero init, and which allowed training change makes the coordinated move happen? Candidates with the evidence that separates them:
- (a) Early curvature optimiser: Z2 shows a valley early (re-fitted loss falls with s on the unchanged model) -> candidate; flat or barrier -> rejected.
- (b) Multi-rate (supervisor): at 2 kHz / 800 Hz the 212 Hz mode sits at |z| 0.972 / 0.93 (angle 0.67 / 1.66 rad), shorter memory in steps; test whether the pole-move cut (H2 protocol) and short training probes behave differently at coarse rates. First check how the August multi-rate work handled Ts in the ANN (rate transfer, note section B) and the 800 Hz bandwidth caveat (note section 0.2).
- (c) Encoder speed / co-estimation (Hoekstra Eq. 31): per-group lr on the encoder's W^a; tracked poles.
- (d) Something the literature names that the previous session did not consider.

## 9. Next action
Write the three missing `DECISIONS.md` lines (section 3) and a Z2 prediction, then run Z2: `y2_valley_early.py early150 100 64` (detached via `run.cmd`, about 45 min), because it is the one measurement that decides candidate (a) on the unchanged model and tests the weakest link of the reading (X1 was on a changed model). Launch the literature sub-agents (section 14) while it runs.

## 10. Acceptance criterion
Done when (1) `CAUSE-AND-DIRECTION.md` states the cause in one paragraph with, for it and each alternative, the deciding local number (file, `DECISIONS.md` id, prediction written first) or source (page / equation), and its section 3 is corrected; (2) a ranked remedy list where each entry satisfies section 2, names its source and the local measurement that would test it, and at least the top entry has a local result with continuous pole tracking (added-state pair at 150 to 300 Hz with zeta <= 0.15 forming or its radius growing at the right angle, or a measured downhill route, against the same measure on U1); (3) one server experiment with prediction, pass/fail (pole pair at 150 to 300 Hz, zeta <= 0.15, at >= 90 % of encoder points AND > 22 % of the Y error removed in 230 to 297 Hz) and cost. A cause supported only by literature is not done.

## 11. Read these first
1. `scripts/gantry/absorber-learning-diagnosis/DECISIONS.md` section 4: every measurement of 2026-10-01 with its prediction.
2. `scripts/gantry/absorber-learning-diagnosis/CAUSE-AND-DIRECTION.md`: the previous conclusion (correct section 3).
3. `scripts/gantry/absorber-learning-diagnosis/MECHANISM.md`: the 2026-09-30 conclusion (contains the -0.60 error).
4. `literature/multi-rate-horizon/deep-research-multi-rate-loss-ladder.md` sections 0, B, C: the multi-rate literature already done.
5. `docs/gantry-augmentation-problem-log.md` section 15: earlier added-state attempts.

## 12. Do not
- Anything in section 2. Re-run U1 to U4, M1, I1, R1 or the M1/M1b partial graft.
- Treat U3, U2, Y1 or X1's model as a remedy.
- Judge the added states by their internal values; judge by realised poles and output.
- Launch server jobs; touch `kamtin-data/Telica.mat` or `kamtin-data/Data Telica/`.

## 13. Operational
- Detached launch (survives session exit): from the diagnosis folder, PowerShell `Start-Process cmd.exe -ArgumentList '/c ""<folder>\run.cmd" <script.py> <args> > "<folder>\outputs\<name>.log" 2>&1"' -WindowStyle Hidden -PassThru`; env switches (U_*) are inherited.
- CPU about 4 s per update at batch 64, nf 400 (local GPU fails with a device mismatch in the closed-loop simulator; not worth fixing); model load 2 to 5 min; free RAM often 1 to 3 GB: one numerical job at a time.

## 14. Delegation
Code and numerics inline, at most one subagent. Literature via the deep-research skill, up to 3 agents, one sub-question each; the user asked for this sweep on 2026-10-01, so these three may be launched:
- L1: In output-error / adaptive IIR / grey-box identification, why does gradient descent from a zero (or static) start fit a lightly damped mode with a static map or real lags, and which TRAINING methods (not parameterisations) are known to reach the resonant solution: e.g. Steiglitz-McBride / iterative prefiltering, instrumental variables, equation-error to output-error homotopy, Gauss-Newton or Levenberg-Marquardt, variable projection?
- L2: Multi-rate / multi-resolution training of one discrete-time (or rate-normalised) model on the same data at several sampling rates: does it help learn slow or lightly damped dynamics, and what are the conditions (rate transfer of a learned correction, aliasing, weighting or hard-switch schedules)? Start from the local note, do not redo it.
- L3: Training order and encoder in SUBNET / LFR augmentation and neural state-space identification (Beintema, Schoukens, Toth, Hoekstra, Bolderman, Forgione): how are latent states made to carry dynamics rather than static corrections, without changing the model's parameterisation or init?
