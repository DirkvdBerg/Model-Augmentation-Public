# Overnight clean-method runs (2026-10-02/03): predictions before launch, outcomes after

Candidate A = PS2 (`ps2.py`, D-234). Common to every run: run 42 settings (U arm, noisy, detuned, n_a 2), batch 64,
600 outer updates, thesis rates of the U1 control (added-row output 1e-3, rest 1e-5), CPU, one job at a time, launched
detached through `chain_ps2.cmd`. Calibration = training records 17 and 18 (record split fixed in code before any run;
the S1/S2 fits therefore use 16 of 18 training records). Final checkpoint = last update (no selection). Held-out
evaluation `evalc.py` on the 6 validation records (reporting only). Poles are saved, not inspected, until all runs end.

Pass rules (fixed now, Gate C of the handoff):
- SCREEN: S1 ends (plateau or cap 350) with calibration unexplained fraction <= 0.90, and by update 150 it is <= 0.90
  unless S1 already ended. Else FAIL-SCREEN, the run stops.
- MECHANISM: S2 reaches calibration EE relative residual <= 0.10 (the deployed map fits the representation).
- HELD-OUT (`evalc.py`): clamp increase median CI over records excludes 0 (above 0); whole-record free-run sim-RMS
  below the paired control's; no validation record with block-on above block-off where the control has none.
- AMENDMENT (21:37, after the evalc test on C0, before any P1 result): the control itself has a clamp increase whose CI
  excludes 0 (median 1.76e-7 m over records, CI [1.41e-7, 2.74e-7], whole-record on 1.1450e-5 vs clamp 1.1647e-5;
  `outputs/evalc_test.log`): its slow real-lag x_a carries a little memory. "CI excludes 0" therefore does not
  discriminate. Added discriminating reading (reported beside the handoff's criterion, not replacing it): the paired
  per-record clamp increase of the candidate minus that of its control, median CI over records excluding 0.
- ABANDON on: collapse (encoder x_a RMS falls below 1 % of its start), only the head improves (MECHANISM fails),
  clamp CI includes 0 (map unused), a self-oscillating quiet record, non-finite loss.

| Run | Config | Prediction (written before launch) | Outcome |
|-|-|-|-|
| P1 | cand, seed 0, MLP head | S1 plateau at 150 to 300 updates, cal unexplained 0.75 to 0.88 (E3/E5 linear head, M 40: 0.77 at 300 on training batches). S2 cal EE residual 0.01 to 0.05 (AB1: 0.002 to 0.03 on E3's encoder). Eval loss jumps after S2 (smoke test: 1.5e-8 to 4.0e-8 with an unshaped encoder), then recovers below the control (6.91e-9 at 600). Held-out: sim-RMS 4e-6 to 8e-6 m vs control 1.145e-5; clamp CI excludes 0. Main risk (about even odds): the readout must be learned in S3 with the physical-row output and hidden layers at 1e-5; E5/E9 learned it with the hidden layers at 1e-3, so on 300 to 450 S3 updates the gain may be small, giving clamp CI including 0 = FAIL (map unused at this budget). | **FAIL-SCREEN at update 150** (`outputs/ps2_p1.log`, 3.8 s/upd). Training-batch unexplained 0.99 / 0.95 / 0.89 / 0.84 / 0.85 / 0.71 at 25 to 150 (faster than E5's linear head, 0.885 at 150); calibration (TR-T3, TR-T4) 0.988 / 0.968 / 0.967 / 0.975 / 0.994: the head did not transfer to the held-out records. Thesis loss tracked the control (7.2133e-9 vs 7.2127e-9 at 100). Not anticipated: TR-T is the no-multisine dwell class (D-decisions near the TR-T1 friction gate), so the blind "last two records" split put a different excitation class in calibration. D1 decides between "unrepresentative calibration" and "head learned record-specific content". |
| C0 | control seed 0 | existing `u_u1ctl600.pt` (exact pair: same init, batch sequence and rates; updates 1 to 2 reproduce its losses bit for bit). Val sim-RMS 1.145e-5 (AD1). Clamp increase near 0 (x_a unused). | sim-RMS 1.145e-5 (reproduces AD1); clamp increase median 1.76e-7 (1.5 %), CI excludes 0 (see the amendment); block-on above block-off on quiet records 5 and 6. |
| P2, P3 | cand, seeds 2, 3 (f2_open redraw), stratified split | run only if P1 passes MECHANISM and HELD-OUT. Same predictions as P1; at least one of two passes. | Both pass SCREEN and MECHANISM (cal unexplained at the S1 cap 0.652 / 0.657; S2 residual 0.0081 / 0.0067). Final eval loss 5.14e-9 / 1.22e-9. HELD-OUT (`outputs/evalc_seeds.log`): P2 9.39e-6 vs C2 1.148e-5 (per-record ratio 0.80 to 0.83 excited, 1.01 / 0.93 quiet; median 0.814), clamp CI excludes 0, but the paired clamp difference CI includes 0 (quiet records slightly negative); no record above block-off; quiet record 5 on/off 0.911 vs the control's 0.898. P3 3.20e-6 vs C3 1.140e-5 (median ratio 0.264, CI [0.241, 0.727]), clamp and paired clamp difference CIs exclude 0, quiet records better than the control. Post-run poles (`d3_poles.py`): P2 pair 203 Hz, zeta 0.33 (overdamped, E8-like); P3 208 Hz, zeta 0.07. |
| C2, C3 | control, seeds 2, 3 | paired controls; val sim-RMS near 1.1e-5, clamp near 0. | Eval loss 6.87e-9 / 6.84e-9; val sim-RMS 1.148e-5 / 1.140e-5; clamp increase medians 3.7e-8 / 1.2e-7. |
| A1 | cand, seed 0, LINEAR head | ablation of the head nonlinearity (the simplest linear predictive-head control for Gate D). Prediction: similar to P1 on the gantry (the missing dynamics are linear; AA1: linear rank-2 predictive state is the mode). | **Not run on the gantry**: the six-start budget went to P1, P1b, P2, C2, P3, C3 (P1b replaced P1 after D2). The linear-head comparison is made on Gate D (arm ps2_lin), where the head nonlinearity can matter. |

**D1 readings, written before its result (21:55).** `d1_predictability.py`: leave-one-record-out ridge prediction of the
update-0 model's 30-step residual from the encoder window. CALIBRATION-UNREPRESENTATIVE if TR-T3/T4 held-out unexplained
>= 0.9 while the multisine-class records (S, Y, P, L) are <= 0.7: the screen measured a class with no transferable
predictable residual; rerun the candidate (P1b, start 2) with a class-stratified record split (hold out the last record
of every class with >= 3 records: TR-S5, TR-Y3, TR-P4, TR-T4; 4 of 18 records), everything else unchanged. HEAD-OVERFIT
if T3/T4 are also <= 0.7 (a linear map transfers but the nonlinear head did not): the next start is A1 (linear head,
original split), and the nonlinear-head form is rejected unless A1 also fails for the same reason.

**D1 result (`outputs/d1_predictability.log`).** Held-out unexplained: S, Y, P, L records 0.15 to 0.26; TR-T1 to T4 0.34 /
0.38 / 0.26 / 0.34; T-class residual RMS 0.26 to 0.41x the median record. By the written rule: HEAD-OVERFIT (T3/T4 <= 0.7).
Flaw in that rule, noticed after the result: D1 is FULL-RANK (180 regressors); the head sees a 2-d state. Whether a 2-d
predictive state can transfer to the T class at all is not what D1 measured.
**D2 readings, written before its result (22:05).** Same data, rank-2 reduced-rank regression (the linear 2-d predictive
state; CVA-type, whitened-target SVD of the ridge fit) fitted on the 16 fit records, scored on each record. BOTTLENECK-
CLASS-SHIFT if T3/T4 unexplained >= 0.9 while held-out-style multisine records (scored in-sample here, so optimistic) are
<= 0.85: no 2-d linear state transfers to the T class, the screen on T3/T4 cannot be passed by any 2-d head, and P1b with
the stratified split is justified. TRANSFERABLE if T3/T4 rank-2 unexplained <= 0.85: a 2-d state exists that transfers;
the MLP head's 0.99 is a real failure, and the next start is A1 (linear head) on the original split as D1's rule says.

**D2 result (`outputs/d2_rank2.log`).** Rank-2 RRR fitted on the 16 fit records: S, Y, P, L records 0.40 to 0.62; TR-T1 to
T4 1.002 / 1.000 / 1.001 / 0.999 (T1, T2 are IN the fit set; their residual is 0.3x the others, so they do not shape the
pooled directions); rank 4: T class 0.99. Reading: BOTTLENECK-CLASS-SHIFT. No 2-d linear predictive state transfers to the
T class; the P1 screen could not be passed by any 2-d head and measured the calibration choice, not the method.
Consequence (as written): P1b with the stratified split. Counted as a new start (2 of 6) and reported as a post-failure
design correction justified by D2, not as a pre-planned arm.

| P1b | cand, seed 0, MLP head, P_CAL=stratified (calibration TR-S5, Y3, P4, T4; S1/S2 fit on the other 14) | Written 22:10 before launch. Screen passes: calibration unexplained <= 0.90 by update 150 (the rank-2 linear ceiling on held-out-type multisine records is 0.45 to 0.55; T4 barely weighs in the pooled batch normalisation). S1 plateau 150 to 300 at 0.6 to 0.85. Then as P1's prediction (S2 residual 0.01 to 0.05; post-S2 loss jump; risk that the readout is not learned in S3 at the thesis rates). | **PASS all gates** (`outputs/ps2_p1b.log`, `outputs/evalc_p1b.log`). SCREEN: cal unexplained 0.986 / 0.949 / 0.906 / 0.865 / 0.809 / 0.757 at 25 to 150, 0.605 at 350 (S1 ended at the CAP, not a plateau: still falling about 1 % per 25 updates). MECHANISM: S2 cal EE residual 1.039 -> 0.0087 in 3800 steps (2.5 min). Eval loss 6.83e-9 (350) -> 1.58e-8 after S2 (the live x_a meets an untrained readout) -> 1.15e-8 / 5.95e-9 / 4.53e-9 / 1.69e-9 / 1.30e-9 at 375 / 400 / 425 / 500 / 600 (control 6.91e-9 at 600: 5.3x lower). HELD-OUT: whole-record sim-RMS 3.121e-6 vs control 1.145e-5 (per record ratio 0.23 to 0.27 on the four excited records, 0.61 to 0.63 on the quiet ones; median 0.252, CI [0.235, 0.619]); clamp: 1.490e-5 (worse than the control: the model relies on x_a), increase median 1.72e-5, CI [4.6e-7, 1.77e-5] excludes 0; candidate minus control clamp increase median 1.70e-5, CI [1.8e-7, 1.76e-5] excludes 0; no record with block on above block off (control: records 5 and 6, the quiet ones, 1.04x and 1.10x above). |

## Gate D (`gate_d.py`, written 23:20, before any full run; smoke runs of 75 updates only checked the code path)

Hidden Duffing third mass in Jan's 3-DOF chain, a3 = 2.354e4 (RMS cubic / linear spring-3 force 0.5 on the linear
pre-simulation; truth side only). Ideal 2-DOF baseline, S-DP, Jan's ANN (2 x 8 tanh, zero-init) and SSE encoder, open
loop, normalisation fixed to Jan's block constants. 800 updates, batch 256, nf 200, Adam 1e-3 in every arm (Jan's
deepSI default), seeds 0, 1, 2; arms sdp / ps2_mlp / ps2_lin with ps2.py's rules (M = 13 = encoder window). Launched
by `chain_gate_d.cmd` only after `chain_seeds.done`.
- Predictions. ps2_mlp and ps2_lin beat sdp on test free-run RMS in >= 2 of 3 seeds, with the median improved, and with a
  clamp increase whose segment-bootstrap CI excludes 0; sdp's clamp increase includes 0 or is negative (smoke: x_a unused).
  ps2_mlp beats ps2_lin on the 1.5x extrapolation record in >= 2 of 3 seeds (amplitude-dependent hidden frequency).
  Risk: 450 S3 updates may not be enough for the readout after S2's loss jump (smoke: test RMS 8.7e-2 right after S2
  vs sdp 1.6e-2 at 75 updates).
- Pass (framework-candidate level): the first prediction. The head-nonlinearity claim needs the second as well.

**Gantry summary (seeds 0, 2, 3; paired, held-out validation free run).** Candidate / control sim-RMS: 3.12e-6 / 1.145e-5, 9.39e-6 / 1.148e-5, 3.20e-6 / 1.140e-5: 3 of 3 better, median ratio 0.28. Clamp CI excludes 0 in every candidate; the paired candidate-minus-control clamp difference excludes 0 in 2 of 3 (seed 2 includes 0). No candidate record above block-off.

**Gate D result (`outputs/gate_d.log`, 03:00 to 04:45).** Test free-run RMS (noise sigma 5.2e-3), seeds 0 / 1 / 2:
sdp 6.61e-3 / 6.39e-3 / 6.68e-3; ps2_mlp 8.65e-3 / 1.11e-2 (FAIL-SCREEN 125, 0.955) / 7.54e-3; ps2_lin 8.10e-3 /
1.11e-2 (FAIL-SCREEN 125, 0.977) / 9.22e-3 (FAIL-SCREEN 150, 0.922). S2 residual where S2 ran: mlp 0.275 (s0) and 0.082
(s2), lin 0.232 (s0): MECHANISM (<= 0.10) failed in two of three. Extrapolation 1.5x: sdp 1.91e-2 / 2.47e-2 / 1.56e-2;
ps2_mlp 1.78e-2 / 3.92e-2 / 1.12e-2; ps2_lin 2.50e-2 / 3.93e-2 / 3.71e-2. Clamp: sdp's x_a is used in every seed (test
clamp increase CI excludes 0, +2.9e-3 to +3.2e-2).
- Prediction 1 (PS2 beats sdp on test in >= 2 of 3): REFUTED, 0 of 3 for both heads. Framework-candidate level not reached.
- Prediction 2 (MLP head beats linear head on extrapolation in >= 2 of 3): met (s0, s2; s1 both stopped), and ps2_mlp beats
  sdp on extrapolation in s0 and s2. Two completed runs only: a hint, not evidence.
- Not anticipated: ordinary S-DP already learns the hidden subsystem here (unlike the gantry), so the benchmark tests "does
  PS2 hurt when S-DP works" rather than "does PS2 rescue a dead start". It hurts at this budget: the stop-on-failed-screen
  protocol truncates training (s1), and the S2 implant of a poorly fitting map costs S3 time (s0).

## Cubic-absorber rescue test (D-235, written 2026-10-03 10:30, before any run)

Data: `Thesis-writeup/Data/Cubic-absorber-test/` (12 noisy records; stratified calibration = TR-S3, Y3, P3; S1/S2 fit on
6; validation VA-S1, Y1, P1). Same ps2.py rules and rates as the gantry runs, plus P_FALLBACK=1. 600 updates, batch 64.
- K0 (plain S-DP, seed 0) first. Gate 1 PASSES (plain fails) if its held-out clamp increase is below 5 % of its sim-RMS
  (linear-set control: 1.5 %) and its eval loss stays within 20 % of the update-300 value from 300 to 600 (no late
  learning). Prediction: PASS (the cubic absorber is at least as weak under feedback as the linear one). If it FAILS
  (plain S-DP learns it), stop: no rescue can be shown.
- Then Q0, Q2, Q3 (PS2, MLP head, seeds 0 / 2 / 3) and K2, K3 (plain, seeds 2 / 3). Prediction: PS2 beats plain on
  held-out free run in >= 2 of 3 seeds with the median improved and clamp CIs excluding 0; the gain smaller than on the
  linear set (the cubic transition must be represented by the 2 x 16 tanh map after S2; post-run poles at the encoder
  points will spread with the stroke).
- Not tested (budget): the linear head on this set.

**K0 result (`outputs/ps2_k0.log`, `outputs/evalc_k0.log`).** Eval loss 1.51e-8 (0), 8.39e-9 (100), 7.01e-9 (300), 6.03e-9
(600): 14 % drop from 300 to 600, inside the 20 % rule. Held-out (3 val records): on 1.287e-5, clamp 1.795e-5, off 2.069e-5;
clamp increase 42 % of sim-RMS, CI excludes 0. **Gate 1 FAILS by its written rule** (clamp >= 5 %).
Post-run diagnosis (`d3_poles.py k0`, read after the gate decision): added poles tiny and heavily damped, median dominant
|z| 0.19, a complex pair at 12 % of points only, 633 Hz, zeta 0.95: no slow or lightly damped mode. Plain S-DP routes a fast
(few-sample) nonlinear correction through x_a, it does not learn the absorber. The clamp proxy therefore did not measure
what Gate 1 meant ("plain S-DP learns the missing dynamics"). **Deviation, decided after seeing the result and stated as
such:** PS2 is run anyway, one seed first (Q0, paired with K0, same init and batches); seeds 2 and 3 only if Q0 beats K0 on
held-out free run. Prediction for Q0: SCREEN and MECHANISM pass (as on the linear set); held-out sim-RMS below K0's
1.287e-5 with a candidate-minus-control clamp difference above 0; post-run pair near the cubic absorber's amplitude-
dependent frequency (hardening: above the linear 211 Hz on strokes near peak).

**Q0 result (`outputs/ps2_q0.log`, `outputs/evalc_q0.log`).** SCREEN pass (cal 0.838 at 150; 0.742 at the S1 cap 350),
MECHANISM pass (S2 cal residual 0.919 -> 0.031, 6200 steps; looser than the linear set's 0.007 to 0.009). Eval loss 2.20e-8
after S2, 4.68e-9 (400), 2.89e-9 (500), 2.47e-9 (600) vs K0 6.03e-9. HELD-OUT: 7.17e-6 vs K0 1.287e-5 (per record 0.52 /
0.55 / 0.60), clamp increase 1.10e-5 (CI excludes 0), candidate minus control 5.4e-6 (CI excludes 0), no record above
block-off. Post-run poles: pair at 100 % of points, 229 Hz, zeta 0.059, |z| 0.979 (linear set 208 to 211 Hz; above it, as
a hardening spring predicts). Q0 beats K0: seeds 2 and 3 launched (`chain_cubic_seeds.cmd`), same predictions.

**Generalisation test, written 2026-10-03 13:00 before the records exist (user discussion: success = held-out IO
generalisation to unseen amplitudes / trajectories; poles only as diagnosis).** Four cubic TEST records, same generator:
I1 (as TR-P2, new phases: interpolation), I3 (ASMPT sine route: unseen trajectory class), E3 (S-curve route at 100 % max
acceleration, 30 / 50 m/s^2 vs at most 22.5 / 37.5 in training: larger absorber stroke, amplitude extrapolation of the
cubic spring), E4 (long strokes, v cap 1.95 m/s), plus E1 (standstill). No training; the six checkpoints Q0/K0, Q2/K2,
Q3/K3 evaluated with `evalc.py` under E_SPLIT=test.
- Prediction: PS2 below plain S-DP on every record class in >= 2 of 3 seeds, including E3. If PS2 only learned an averaged
  linear resonance, its ratio to plain on E3 is closer to 1 than on I1 (the hardening shifts the mode at large stroke);
  if it captured the amplitude dependence, the E3 ratio is no worse than I1's.

**Cubic seeds 2 and 3 (`outputs/ps2_q2|k2|q3|k3.log`, `outputs/evalc_cubic_seeds.log`).** Q2 / Q3: SCREEN and MECHANISM
pass (cal at cap 0.727 / 0.759; S2 residual 0.019 / 0.015); final eval loss 3.40e-9 / 2.53e-9 vs K2 / K3 7.36e-9 / 6.51e-9.
HELD-OUT: Q2 7.47e-6 vs K2 1.434e-5 (per record 0.49 / 0.50 / 0.57), Q3 7.16e-6 vs K3 1.323e-5 (0.55 / 0.54 / 0.54);
paired clamp differences exclude 0; no record above block-off in any run.
**Cubic summary (validation, 3 records): PS2 vs plain S-DP 3 of 3 seeds better, ratios 0.55 / 0.50 / 0.54 (median 0.54).**

**Generalisation test result (`outputs/evalc_cubic_test.log`; records I1, I3, E1, E3, E4; no training, no selection).**
Whole-record sim-RMS PS2 / plain: 4.67e-6 / 7.06e-6 (seed 0), 3.97e-6 / 7.84e-6 (2), 3.94e-6 / 7.04e-6 (3): ratios 0.66 /
0.51 / 0.56. Per-record ratios (I1, I3, E1, E3, E4): seed 0 0.60 0.86 0.55 0.87 1.07; seed 2 0.56 0.44 0.47 0.51 0.51;
seed 3 0.54 0.62 0.51 0.63 0.87.
- Prediction 1 (PS2 below plain on every record class in >= 2 of 3 seeds): MET (E4 2 of 3, all others 3 of 3).
- Prediction 2 (amplitude): E3 ratio vs I1 ratio 0.87 vs 0.60, 0.51 vs 0.56, 0.63 vs 0.54: the advantage shrinks at the
  larger stroke in 2 of 3 seeds. Not evidence that the amplitude dependence was captured; partly an averaged mode.
- Block-off: plain S-DP above block-off on unseen routes (K2: I3, E3, E4; K0, K3: E3); PS2 below block-off everywhere
  except Q0 on E3 (1.04x).
- Clamp: PS2's x_a memory carries its gain on the multisine records (I1, E1: +9 to +11e-6 when clamped); on the route
  records clamping changes little or helps slightly (-1e-6 to +5e-7); there PS2's advantage comes from the rest of the
  trained model. The paired clamp differences include 0 over the five records for this reason.
