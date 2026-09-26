# Noise in the simulated data: level and injection point

Status 2026-09-26: approach fixed; calibration changed to the measured error spectrum below 300 Hz, level and shape (section 3, agreed 2026-09-26); force spectrum not yet recomputed; dataset not yet generated.

## 1. Goal and instruction
- Goal: a realistic noise level from the Telica data, added in the simulation that generates the training data
- ASMPT supervisor: "Stop het direct in de simulatie, maak het realistischer en doe het op de input."
- So: inside the simulated closed loop, on the input (the motor force), not on the encoder
- Supervisor feedback (J., 2026-09-25, given on the encoder proposal): also check in the frequency domain; "het zal niet exact hetzelfde zijn maar de vorm zal wel moeten kloppen"
  - with a force at the input the shape can only match below 300 Hz (section 3); agreed 2026-09-26
- Difference with Hoekstra et al. (open loop): there, noise is added to the output after simulation; in a closed loop the controller reacts to the noise, so it must enter during the simulation

## 2. Step 1: the measured noise level (done)
- Data: the standstill part of every Telica log, before each move (reference constant, no feedforward), 145 logs, 20 kHz
- Signal: servo error per axis, e = r - y_measured
- Level (rms): X1 9.8 nm, X2 10.4 nm, Y 6.3 nm
- Details: `scripts/gantry/closed-loop-noise/REPORT.md` G1; spectrum `outputs/g1_spectra/g1.npz`

## 3. Step 2: from nm to newtons, below 300 Hz (level and shape)
- The level is a position error (nm); the input is a force (N), so it needs one conversion
- Band: below 300 Hz only
  - a force moves the mass mainly at low frequency; above about 1 kHz a force would need 30 to 2000 N to explain the measured error (`closed-loop-noise/REPORT.md` G3), so that part is sensor noise and is not modelled (section 7)
  - below 300 Hz a force of tenths of a newton explains the measured error as well as sensor noise does
- Target: the measured error spectrum from `g1.npz`, 19.5 to 300 Hz, level and shape: 3.4 / 3.7 / 1.1 nm rms (X1 / X2 / Y), about 12 % of the measured variance on X
- Force spectrum, per frequency up to 300 Hz: Phi_d = (S_sim G_sim)^-1 Phi_e,meas (S_sim G_sim)^-H
  - S_sim G_sim: the simulation's own force-to-error transfer (e = -S_sim G_sim d), known exactly from the model; no Telica plant needed
  - 3 x 3 matrix form: keeps the measured cross-axis correlation below 300 Hz (X1-X2 coherence up to 0.44 at 100 Hz) at no extra cost
- Loop used: the simulation's own truth, linearised at rest, friction off (frictionless 8-state plant with the production absorber, `scripts/gantry/transient/code/truth.py`), ZOH at 20 kHz, controller Cfb at 20 kHz (`controller.py`); equals the noise session's simulated loop (`closed-loop-noise/outputs/recipe/recipe.mat`) within 4e-8
- Per record Y_op: S_sim G_sim at each record's own Y_op (controller and M(Y) change with Y_op; one Y_op for all records was up to about 30 % off on X); moving records use the Y_op they are built at, a stated approximation since Y changes during the move
- Band edges
  - below 19.5 Hz (not resolved by the 9.77 Hz Welch bins): hold the lowest band value, stated as an extrapolation
  - at 300 Hz: a smooth roll-off, not a hard cut (a hard cut rings in time); roll-off shape fixed before generation (?)
- Generating d per record: a shaping filter fitted to Phi_d (as `closed-loop-noise/model/noise_model.py`), unit white input, own seed per record
- Friction off in this step only: with Karnopp friction a rail at rest is locked below 11-18 N, and a sub-newton force produces no motion to calibrate against
- Superseded (kept for the record, not used): white force calibrated to the TOTAL rms (sigma 0.312 / 0.342 / 0.099 N per sample at Y_op = 0; `scripts/gantry/noise-training/tests/t6_force_level.py`, NT-008). It matched 9.8 / 10.4 / 6.3 nm by putting about 3 times the measured error below 300 Hz and nothing above, so its spectrum could not pass the shape check

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
2. Friction off, standstill records at the Y_op values used: the simulated servo-error spectrum below 300 Hz matches the measured one in level and shape (rms 3.4 / 3.7 / 1.1 nm within 2 %; band means plotted over the measured spectrum; supervisor's frequency-domain check)
3. Friction on, every record: report the servo-error rms the noise adds; stuck rails will show almost none, sliding records close to the target

## 7. Known limits (stated, not corrected)
- Only the part below 300 Hz is modelled: about 12 % of the measured variance on X (3.4 of 9.8 nm); the rest, over 90 % above 200 Hz, is sensor noise, which a force cannot produce, so the records carry about 3.4 / 3.7 / 1.1 nm, not 9.8 / 10.4 / 6.3 nm
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

## 9. Updated 2026-09-26
- `docs/decisions.md` D-218 (this approach; supersedes D-212's encoder injection), D-212 (marked superseded)
- `scripts/gantry/closed-loop-noise/DECISIONS.md` CN-012 (marked superseded)
- `Thesis-writeup/Documentation/DATA-DESIGN.md` (noise lines in sections 2, 7 and the change list)
- Not updated, written for encoder injection by the training session: `scripts/gantry/noise-training/NOISE-TRAINING.md`, `tasks/handoffs/2026-09-25-noise-training.md`
