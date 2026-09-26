# Noise in the simulated data: level and injection point

Status 2026-09-26: approach fixed; force level and correlation computed (section 3); dataset not yet generated.

## 1. Goal and instruction
- Goal: a realistic noise level from the Telica data, added in the simulation that generates the training data
- ASMPT supervisor: "Stop het direct in de simulatie, maak het realistischer en doe het op de input."
- So: inside the simulated closed loop, on the input (the motor force), not on the encoder
- Difference with Hoekstra et al. (open loop): there, noise is added to the output after simulation; in a closed loop the controller reacts to the noise, so it must enter during the simulation

## 2. Step 1: the measured noise level (done)
- Data: the standstill part of every Telica log, before each move (reference constant, no feedforward), 145 logs, 20 kHz
- Signal: servo error per axis, e = r - y_measured
- Level (rms): X1 9.8 nm, X2 10.4 nm, Y 6.3 nm
- Details: `scripts/gantry/closed-loop-noise/REPORT.md` G1; spectrum `outputs/g1_spectra/g1.npz`

## 3. Step 2: from nm to newtons, with the cross-axis correlation
- The level is a position error (nm); the input is a force (N), so it needs one conversion
- Noise force d: white per 20 kHz sample (held at the controller rate), stage axes [F_X1, F_X2, F_Y], correlated across the axes through one 3 x 3 covariance Sd
- Target: the measured zero-lag error covariance (from `g1.npz`): rms 9.85 / 10.40 / 6.34 nm, correlation X1-X2 -0.08, X1-Y -0.04, X2-Y 0.00
  - the overall correlation is weak; the strong X1-X2 coherence (up to 0.82) is local to 400-700 Hz
- Loop used: the simulation's own truth, linearised at rest, friction off (frictionless 8-state plant with the production absorber, `scripts/gantry/transient/code/truth.py`), ZOH at 20 kHz, controller Cfb at 20 kHz (`controller.py`)
- Method: the error covariance is linear in Sd (discrete Lyapunov equation of the closed loop), so the 3 variances and 3 cross-terms of Sd follow from one 6 x 6 linear solve per Y_op
- Friction off in this step only: with Karnopp friction a rail at rest is locked below 11-18 N, and a sub-newton force produces no motion to calibrate against

Result at Y_op = 0 (used for all records):

| | X1 | X2 | Y |
|-|-|-|-|
| sigma per 20 kHz sample [N] | 0.312 | 0.342 | 0.099 |
| of which below 300 Hz [N rms] | 0.054 | 0.059 | 0.017 |

- Force covariance Sd [N^2]: [[0.0975, -0.0132, -0.0035], [-0.0132, 0.1167, 0.0010], [-0.0035, 0.0010, 0.0098]]; force correlation X1-X2 -0.12, X1-Y -0.11, X2-Y 0.03; a valid covariance at every Y_op
- Dependence on Y_op (controller and M(Y) change): sigma 0.29-0.43 / 0.32-0.46 / 0.10 N over Y_op -0.3 to 0.3 m; with the Y_op = 0 forces, records at other Y_op get a correspondingly different error level (up to about 30 % on X) (?)
- Correlation matters little here: independent axes give the same sigmas within 1 % and error correlations +0.03 / +0.04 / -0.02 instead of -0.08 / -0.04 / 0.00; the full Sd keeps the measured values at no extra cost
- Checks: the loop built here equals the noise session's simulated loop (`closed-loop-noise/outputs/recipe/recipe.mat`) within 4e-8; a 20 s time-domain run of the linear loop with Sd gives 9.94 / 10.55 / 6.38 nm (+0.9 / +1.4 / +0.6 %) and correlations -0.06 / -0.03 / 0.00
- Run: `scripts/gantry/noise-training/tests/t6_force_level.py`, result `outputs/t6_force_level/t6.json` (NT-008)
- Generating d per record: draw w ~ N(0, I) per 20 kHz sample, d = L w with L L^T = Sd (Cholesky), own seed per record

## 4. Step 3: where it enters the simulation
- Node: the motor force entering the plant, per stage axis: plant force = u_ff + u_fb + d
  - u_ff: feedforward (multisine or Telica profile, `f_sim`)
  - u_fb: controller output (`LTI System1`)
  - d: the noise force
- One point only: nothing on the encoder, nothing separately on the multisine (the amplifier adds noise to the summed force)
- Simulink, "Extended ODE" loop (the recorded loop): a Sum block between the existing sum of `LTI System1` and `Feedforward` and the input `u` of `Extended ODE`, fed by a From Workspace block `d_in` = [t, d] (interpolation off, sample time ts); same edit pattern as `closed-loop-noise/matlab/gen/make_noise_model.m`
- Friction on, as in the truth: d passes through the Karnopp friction like any motor force
- Every record its own noise realisation (seed per record)

## 5. What is recorded
- `y`: true position (no measurement noise)
- `u_total` = u_ff + u_fb: the commanded force, WITHOUT d (on the machine the logged force is the command)
- `d_in`: the injected force, for diagnostics only, never a model input
- `noise_seed`, and all signals in double precision (float32 rounds Y by up to 8.6 nm)

## 6. Checks
1. d = 0 reproduces the existing records bitwise
2. Friction off, one standstill record per axis: simulated servo error rms equals 9.8 / 10.4 / 6.3 nm within 2 % (checks the conversion)
3. Friction on, every record: report the servo-error rms the noise adds; stuck rails will show almost none, sliding records close to the target

## 7. Known limits (stated, not corrected)
- The level matches, the spectrum does not: a force moves the mass mostly at low frequency, so the simulated noise sits mainly below ~200 Hz, while Telica's is over 90 % above 200 Hz (mostly sensor noise, not modelled here)
- At true standstill the simulated Karnopp friction holds the rails, so standstill records carry almost no noise; the real machine shows 10 nm there
  - why the real machine differs: most of its 10 nm is sensor noise, which needs no motion and is not modelled here; and real friction does not lock hard at nm scale: a held rail behaves as a stiff spring with damping (pre-sliding), so small forces still move it slightly
  - evidence on Telica (`telica-real` TR-015): Karnopp's hard stick drove a stick-slip limit cycle the machine does not have; a softer (tanh) friction law matched the machine at standstill, 12 / 12 / 8 nm against the measured 11 / 11 / 7 nm
  - so the limit lies in the simulation's friction model at rest, not in the conversion to force (section 3); sliding records carry the intended level
  - kept as a stated limit; changing it means changing the friction model of the truth, a separate decision
- SNR for comparison with Hoekstra et al.: compute afterwards from the data, relative to the servo error; relative to position (mm to cm motion, nm noise) it is ~140 dB and meaningless

## 8. For training
- See `scripts/gantry/noise-training/NOISE-TRAINING.md` (written for encoder injection); with input injection:
  - lapses: y carries no measurement noise, so no aliasing of the target and no noise in the controller branch
  - holds: the noise acts as an unmeasured force, so each record has an error floor the model cannot remove; float64; report each error against its record's floor

## 9. To update
- Still placing the noise at the encoder: `docs/decisions.md` D-212, `scripts/gantry/closed-loop-noise/DECISIONS.md` CN-012, `Thesis-writeup/Documentation/DATA-DESIGN.md` (line 415)
