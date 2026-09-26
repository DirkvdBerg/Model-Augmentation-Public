# Noise in the simulated data: level and injection point

Status 2026-09-26: approach fixed; force level not yet computed; dataset not yet generated.

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

## 3. Step 2: from nm to newtons (one number per axis)
- The level is a position error (nm); the input is a force (N), so it needs one conversion
- Noise force: white, independent per axis [F_X1, F_X2, F_Y], rms sigma_X1, sigma_X2, sigma_Y, held at the 20 kHz controller rate
- Choose the three sigmas so the simulation's standstill servo error equals the measured level:
  - linear closed loop of the simulation: Garcia plant (`controller.py` parameters) with the controller Cfb at 20 kHz, Y_op = 0
  - e = S G d, so the error variance per axis is linear in the three force variances: var(e_i) = sum_j q_ij sigma_j^2, with q_ij the squared impulse-response sum (the H2 gain) from force j to error i
  - solve this 3 x 3 linear system for sigma_j^2 with var(e) = (9.8, 10.4, 6.3 nm)^2
- Computed on the linear loop only (friction does not enter the conversion): with Karnopp friction a rail at rest is locked below 11-18 N, and a sub-newton force produces no motion to calibrate against
- One set of sigmas for all records
- Result: sigma = (?) / (?) / (?) N, not yet computed

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
- SNR for comparison with Hoekstra et al.: compute afterwards from the data, relative to the servo error; relative to position (mm to cm motion, nm noise) it is ~140 dB and meaningless

## 8. For training
- See `scripts/gantry/noise-training/NOISE-TRAINING.md` (written for encoder injection); with input injection:
  - lapses: y carries no measurement noise, so no aliasing of the target and no noise in the controller branch
  - holds: the noise acts as an unmeasured force, so each record has an error floor the model cannot remove; float64; report each error against its record's floor

## 9. To update
- Still placing the noise at the encoder: `docs/decisions.md` D-212, `scripts/gantry/closed-loop-noise/DECISIONS.md` CN-012, `Thesis-writeup/Documentation/DATA-DESIGN.md` (line 415)
