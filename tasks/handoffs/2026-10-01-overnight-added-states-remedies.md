# Handoff: overnight, unattended: pin down why the added states do not learn the absorber, search the literature, and test in-framework remedies locally
**From**: session of 2026-10-01 | **Branch**: Augmentation | **Effort suggested**: xhigh (causal diagnosis plus literature plus a chain of local probes, unattended)

## 0. How to read this prompt
- **The user's words (2026-10-01, spelling cleaned):** "do a proper literature search and have good understanding of our problem"; "find out exactly what our problem is and try during the night possible solutions which we can do locally, without needing feedback from me"; "respect the constraints we have set up in this session"; "use what we have learned here". Earlier the same day: "the goal of this approach is to be able to extend it to real data ... capture actual dynamics without me having to push it that way. But a generalized approach."
- **Task type: computation plus literature.** The local runs ARE the deliverable, not a plan. The user is asleep: never ask a question, never wait for approval, never end the session early. Decide with stated defaults, log every decision in `DECISIONS.md`, and keep working until the morning or until the candidates are exhausted. Asking "shall I" made the user angry today (memory `feedback_continue_means_decide.md`).
- Sections 4 to 6 hold measured results. Section 5 marks the previous session's readings; test them, do not adopt them.
- **You decide the direction.** This prompt gives evidence and leads, not a plan. The previous session concentrated on one line (E3, the innovation encoder) and may have narrowed too early. Form your own view of what the problem is from the measurements and the literature, and choose your own probes, including directions the previous session did not consider. If you conclude the problem is something else than stated here, follow your conclusion and say why in `DECISIONS.md`.

## 1. Task
Using `scripts/gantry/absorber-learning-diagnosis/DECISIONS.md` section 4 and the outputs listed below, state precisely why gradient training of the thesis augmentation model never makes the 2 added states realise the payload absorber's lightly damped mode (212 Hz, zeta 0.043, |z| 0.986 at 4 kHz). Run a proper literature search (section 14) for remedies that satisfy section 2. Then design, predict and run local probes overnight, one at a time, each to completion, with continuous added-pole tracking. Which probes is your decision, based on your own diagnosis and the literature: E3 (the only probe so far that produced in-band complex added poles) is one lead, not the assignment. In the morning, leave (a) `DECISIONS.md` entries for every probe (prediction first, then result and verdict), (b) an updated `CAUSE-AND-DIRECTION.md`, and (c) a short morning report at the end of `DECISIONS.md` section 4: what was run, what worked, the one recommended next server experiment (prediction, pass/fail, cost).

## 2. Out of scope and constraints (user, absolute, repeated many times on 2026-10-01)
- **No initialisation or parameterisation change of the model**: no identity or residual skip, no random linear part, no complex / polar / oscillator / resonator / modal-bank structure, no residual- or BLA-based initialisation (the user tried that before in `scripts/gantry/` BLA folders and calls it a workaround), no band, pole, frequency or damping prior. Such constructions are allowed ONLY as labelled diagnostics, never as a remedy.
- **Never burn-in** in the loss.
- **No system-specific information**: the remedy must work unchanged on a system whose missing dynamics are unknown (real data later).
- **Jan's framework unchanged**: same model, Static_ANN_Block with its zero-init MLP (keep `ann.net.net` as Jan's Sequential: post-training diagnostics read it, `closed_loop_report.py` l.58, `diagnostics.py` l.254), same routing, the model equals the baseline at start, compatible with joint estimation and the OBC projection (or the incompatibility argued).
- **Allowed families**: optimiser, objective (temporary training terms, horizons, weighting), schedule / order, encoder (how x_a is estimated and trained). User-approved today: the temporary equation-error term, and a residual-driven (innovation) encoder target "for now" as a TRAINING signal, not as an initialisation.
- **Implementation**: copied scripts with proper classes in the diagnosis folder; no importing a pipeline module and overriding its functions at runtime. Do not edit `scripts/gantry/gantry_interconnect_dynamic.py`, `scripts/gantry/gantry_dynamic/`, `model_augmentation/`; `kamtin-fp-model/`, `Thesis-writeup/` read only.
- **No server jobs** (propose only). **Do not modify** the prepared server files (the user is copying them to the cluster): `innovation_opening.py`, `model_innovation.py`, `gantry_innovation_opening.py`, `run_gantry_innovation_opening.sh`, `test_innovation_opening.py`, `smoke_innovation_opening.cmd`. Build new variants as new files.
- Not: Jan's MSD example as the main line; the multi-RATE idea of the supervisor as the first probe (see section 6); data or band changes; `kamtin-data/Telica.mat` or `kamtin-data/Data Telica/`.

## 3. Where things stand
- Branch Augmentation, tree dirty (many unrelated files); do not commit.
- **Server runs prepared, not launched** (the user submits): `run_gantry_innovation_opening.sh`, array 1 = arm U (run 42 settings), 2 = arm OBC (run 41 settings), partition oahu, 24 h; E3's opening phase for 2000 updates (decaying to 0), then the thesis run. Pre-flight: `test_innovation_opening.py` ALL CHECKS PASSED (both arms); smoke runs of the real script (`outputs/smoke_innov_u.log` EXIT 0; `outputs/smoke_innov_obc.log` EXIT 0; 0 errors in both). The user has since submitted the array (job 87446; task 1 = 87447 on blade1, RTX 2080 Ti). Logs on the cluster: `/home/dirk_van_den_berg/logs/augmentation/absorber-learning-diagnosis/gantry_innovation_opening_<job>_<task>.out`. Do not duplicate this run locally.
- No local job running at handoff (check `Get-CimInstance Win32_Process` for python before the first launch).

## 4. Established and verified (DECISIONS.md ids; all on the unchanged thesis model unless stated)
- **Setup.** Truth = FP gantry + tanh Coulomb friction + payload absorber (pole 211.9 Hz, zeta 0.043). Model: 10-combination LPV physics block (joint, 10 % detuned) + Static_ANN_Block (2x16 tanh MLP, zero-init output) writing physical correction rows additively and the 2 added rows fully from z = [x, x_a, u]; the added states' A_aa, B_a, readout are the MLP's Jacobians (zero at init). Closed-loop rollout, nf 400 at 4 kHz, float64, Adam lr 1e-5 eps 1e-16, no burn-in. Encoder: W^b from the reconstructability map, W^a random (kaiming), trained jointly by the rollout loss = Hoekstra 2026 Sec. 5.4.2 Eqs. 30, 31 (pipeline follows Jan's method).
- **Class and objective reward the absorber**: G1 (truth 1.35e-9 vs model 5.09e-9), K1 (class member 7.0e-10).
- **Early vs late**: Z2 (150 Adam updates, unchanged model): moving the added self-map to a 212 Hz pair and re-fitting 100 updates gives 0.427x the re-fitted control, nothing uphill, but the first half of the path is flat (1.1 %), the gain arrives once |z| > 0.7. Z1 (run 42 end point): the same move is flat (+-1 %), +7 % at the end. W1, H2: with drive / readout fixed the move is uphill 1.8x to 4.6x at every horizon (barrier at s 0.5 grows with nf: +13 % at nf 25, +62 % at nf 400).
- **Run 42's dominant added pole is +0.60 (real lag)**, not -0.60 (`h_learned_poles.py` l.142 to 144 sorted moduli and angles separately).
- **Optimiser family, all NO on the poles**: U1 (added-row output at lr 1e-3): radius 0.021 at 150, real; U4 (hidden layers too): no faster; L1 (`g3_lm.py`, 30 matrix-free Levenberg-Marquardt steps from early150): poles unchanged to three digits, long-horizon eval best at step 5 then overfits the fixed batch.
- **Objective / encoder probes (`f_eeoe.py`, 600 updates, U1 settings, control `u_u1ctl600` 6.91e-9 at 600)**:
  - E1, equation error (EE) on all 8 rows: poles grow 10x faster than U1 but as a REAL lag (0.27); loss equal to control; W^a barely moves.
  - E2, EE on the added rows + W^a at lr 1e-3: W^a changes 35 %, one x_a coordinate less baseline-like (R^2 0.45 -> 0.19), poles still real (0.30), discriminant positive at 100 %.
  - **E3, E2 + innovation head** (`F_INN=40`: a training-only zero-init linear head predicts the current model's next 40 output errors, normalised by their batch RMS, from x_a(k); weights matched to the rollout loss, linear decay to 0 at update 400): complex share rose 0 -> 62 %, pair in 95 to 380 Hz at 42 to 46 % (about 120 Hz), the median pole frequency rose steadily (0 -> 80 Hz), A_aa gained an antisymmetric part (off-diagonals of opposite sign, antisymmetric fraction 0.19), x_a no longer baseline-like (R^2 0.15 / 0.16), head explained 26 % of the innovation at 400; loss equal to control (7.05e-9). **Gap: radius 0.25 (absorber 0.986), zeta 0.99; the radius stopped growing when the terms reached 0 at 400; pure output error kept but did not extend the pairs.**
- **No pole stealing, no parameter drift** in any run (`g2_steal.py`: physical poles shift about 5e-4; combinations within 0.5 %).
- **Literature already read** (cite, do not redo): Ribeiro et al. 2020 Automatica 121:109158 Thm 1 (p.4, Eqs. 8, 9); Hoekstra 2026 arXiv:2602.17297 Sec. 5.4; Shynk 1989 (EE quadratic vs OE); Masti and Bemporad 2021 Automatica 129:109666 Eq. 5 (k / k+1 encoder consistency, grounded by a decoder); Tang et al. ICML 2023 arXiv:2212.03319 Thm 1 (stop-gradient plus fast predictor conserves, does not create, representation); Tian, Chen, Ganguli ICML 2021 (self-predictive linear predictor drifts to symmetry, i.e. real eigenvalues); Diniz, Cousseau, Antoniou 1994 (reduced-order saddle manifolds in adaptive IIR); Emami et al. ICML 2021 (implicit bias to short memory); Zucchet and Orvieto 2024. Two web reports agree: zero-output init = saddle, coordinated pole-numerator move needed; no paper reports a learned resonance in added states.

## 5. Assumed but not verified (test these)
- That E3's radius gap is only a duration effect (terms switched off too early). Alternative readings: the EE target (one-step) rewards a small-radius map that predicts x_a(k+1) from x_a(k) well enough; the innovation head predicts mostly the static / mass-like part of the error; the symmetry drift (Tian 2021) limits rotation. Each predicts a different probe outcome.
- That multi-step consistency (x_hat(k+d) vs d model steps from x_hat(k)) would reward the correct radius and angle (a wrong radius compounds over d steps). Untested.
- That LM helps AFTER the opening (poles complex, drive / readout non-zero). Untested (L1 was from early150 only).
- Collapse risk of a non-detached consistency term (problem log 15.2 item 3).

## 6. Tried and failed (do not retry)
- Init / parameterisation probes U2 (random linear part), U3 (identity skip -> real lags), Y1 (polar pair): diagnostics only, excluded as remedies.
- U1, U4 (learning rates), L1 (LM from early150), M1 (multi-HORIZON 25/100/400 at 4 kHz, stopped at 300 of 600, INCONCLUSIVE, and NOT the supervisor's idea: the supervisor meant multi-RATE, same window length at several sampling rates; `literature/multi-rate-horizon/deep-research-multi-rate-loss-ladder.md` section 0 argues against it as proposed, both web reports too: the 4 kHz one-step map is not the map at a coarser rate).
- E3's first two launches: a randomly initialised head and an un-normalised target broke the term (collapse pressure, overshoot); fixed by a zero-init head and an RMS-normalised target. Any new head must use the same scaling.
- Process errors to avoid: stopping runs on extrapolation (user rule: every run goes to its planned length unless it diverges or has a bug, which must be stated); a background bash chain started from the wrong cwd (use the `.cmd` chain files that `cd /d "%~dp0"`); two sessions working in the same folder (check for other python jobs and recent file changes first).

## 7. Achieved (instruments, validated)
- `u_probe.py` (training probe, pole tracking every 25 updates; env U_LR, U_LR_A, U_LR_H, U_INIT, U_MH), `f_eeoe.py` (EE / encoder / innovation probe; env F_EE_ROWS=added, F_LR_ENC_A, F_INN, plus U_LR_A), `g2_steal.py` (pole stealing, x_a R^2, combinations via `recover_combinations()`, A_aa discriminant and antisymmetric fraction), `g3_lm.py` (matrix-free LM), `y_valley.py` / `y2_valley_early.py` (profile scans), `z_horizon_cut.py`, `v_inspect.py`.
- Checkpoints: `outputs/u_early150.pt` (150 Adam updates, zero init), `outputs/f_inn.pt` (E3 end, 600 updates), `outputs/f_enc.pt` (E2), `outputs/f_eeoe.pt` (E1), `outputs/u_u1ctl600.pt` (control).
- Launchers: `run.cmd <script> <args>` (detached via PowerShell `Start-Process cmd.exe -ArgumentList '/c ""<folder>\run.cmd" <script> <args> > "<folder>\outputs\<log>" 2>&1"' -WindowStyle Hidden`), chain pattern `chain_enc.cmd` (waits for a `.done` file, runs jobs sequentially, survives session exit).

## 8. The open question
Why does training, from the zero start and through the points it actually visits, settle the added states on static maps or real lags instead of the missing lightly damped mode, and which allowed training change (section 2) lets it learn the mode by itself? The previous session's reading and leads follow; they are inputs to your own judgement, not a ranked plan, and the list is not exhaustive.
Leads around E3 (its gap: in-band pairs formed, but radius 0.25 and zeta 0.99, growth stopped with the terms):
- (a) Longer opening (terms held to 1500 or more updates): radius keeps growing linearly -> duration; plateaus near 0.25 -> the terms themselves cap it. The server run tests 2000 updates; locally test the trend more cheaply, e.g. 1000 updates with the terms held constant (no decay) for the first 800.
- (b) Multi-step equation error (d = 5 to 40 steps from x_hat(k) against x_hat(k+d)): a small radius cannot predict d steps ahead, so this should push the radius up if the innovation content oscillates.
- (c) LM after the opening, from `outputs/f_inn.pt`: Gauss-Newton on short windows once drive and readout exist.
- (d) Innovation horizon (F_INN 40 -> 160): more of the oscillation inside the target.
- (e) Something the literature sweep names (bidirectional self-prediction, Tang et al. Sec. 5, to counter the symmetry drift, if it fits section 2).
Other directions are open, for example: a different reading of the cause (section 5), the closed-loop objective itself (weighting, what the controller makes cheap to fit), the encoder's role beyond W^a, training order, or anything the literature or your own analysis of the existing checkpoints suggests.

## 9. Next action
Read `DECISIONS.md` section 4 and `CAUSE-AND-DIRECTION.md` critically, launch the literature sub-agents (section 14), and write your own statement of the problem in `DECISIONS.md` (what you agree with, what you doubt, what you would measure). While the agents run, use the CPU for the first probe of your choice; then continue with whatever the evidence and the literature point to.

## 10. Acceptance criterion
Done for the night when: (1) every probe has a prediction written before its launch and a result with the same criteria; (2) at least three probes of your own choosing (from section 8, the literature, or your own diagnosis; each satisfying section 2) were run to completion with pole tracking and a steal check; (3) `CAUSE-AND-DIRECTION.md` states the cause with the deciding numbers and the ranked remedies; (4) the morning report names the single best next server experiment with prediction, pass / fail (added pole pair at 150 to 300 Hz, zeta <= 0.15, at >= 90 % of encoder points AND > 22 % of the Y error removed in 230 to 297 Hz) and cost. A probe "works" locally if its added pair moves toward the band with a GROWING radius (zeta falling) at a loss not worse than the matched control by more than the late fluctuation (about 5 %).

## 11. Read these first
1. `scripts/gantry/absorber-learning-diagnosis/DECISIONS.md` section 4 (from "## 4. Cause-and-direction phase").
2. `scripts/gantry/absorber-learning-diagnosis/CAUSE-AND-DIRECTION.md`.
3. `scripts/gantry/absorber-learning-diagnosis/f_eeoe.py` and `innovation_opening.py` (the E3 mechanism, probe and server form).
4. `docs/gantry-augmentation-problem-log.md` section 15 (earlier added-state attempts, the multiple-shooting collapse).
5. `literature/closed-loop-id/hoekstra2026_lfr-augmentation-fp-models.pdf` Sec. 5.4 (pp. 9 to 10).

## 12. Do not
- Anything in section 2; re-run U1 to U4, M1, L1-from-early150, I1, R1.
- Stop a run before its planned length unless it diverges (non-finite loss) or has a bug (state it in `DECISIONS.md`).
- Judge the added states by internal values; judge by realised poles (eigenvalues of A_aa at fixed encoder states), output and the steal check.
- Ask the user anything; launch server jobs; modify the prepared server files.

## 13. Operational
- Conda env GraduationProject (`C:\Users\20203253\AppData\Local\anaconda3\envs\GraduationProject`); run probes detached (section 7); one numerical job at a time; free RAM is often 1 to 3 GB (check before launch).
- Cost: about 4.5 to 5.5 s per update at batch 64, nf 400 with the EE / innovation terms (600 updates about 50 to 55 min; model load 2 to 5 min); `g2_steal.py` about 5 min; LM step about 40 s.
- Monitors on logs expire after 30 min: re-arm; grep for `[eval`, `Traceback`, `EXIT`.
- The laptop may sleep: if a run stalls, note it and relaunch.

## 14. Delegation
Literature: deep-research skill, up to 3 agents in parallel, one sub-question each (the user asked for this sweep; launch without asking):
These are starting sub-questions; replace or add one (within the 3-agent cap) if your own diagnosis points elsewhere.
- L-A: Training methods that make latent / added states of grey-box or neural state-space models learn lightly damped modes without priors: multi-step self-prediction, bidirectional self-prediction (Tang et al. 2023 Sec. 5), observer / innovation-based state estimation during training, equation-error to output-error continuation with multi-step terms, SUBNET / LFR augmentation encoder training. Explicitly report what is compatible with a fixed zero-init MLP state map.
- L-B: Why self-predictive / equation-error training yields symmetric (real-eigenvalue) linear predictors, and how rotation (complex eigenvalues) is obtained without parameterisation constraints (Tian, Chen, Ganguli 2021 and follow-ups; adaptive IIR realisation effects).
- L-C: Closed-loop identification with a known controller: how innovation / prediction-error signals are used to estimate unmodelled dynamics in augmentation settings, and the bias risk under noise (Forssell and Ljung, Van den Hof, Gonzalez et al. 2024 SRIVC closed loop).
Code and numerics: inline, at most one subagent.
