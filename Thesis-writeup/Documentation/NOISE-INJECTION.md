# Noise in the simulated data: level and injection point

Status 2026-09-28: noise on the encoder, one independent generator per axis, the measured spectrum's shape mapped above 50 Hz (D-225 and its amendment; supersedes the input force of D-218); implemented in the generator; shape check, Simulink check (A) and a three-record pilot passed; the dataset is not yet regenerated.

Status 2026-10-06: after Jasper's feedback a low-order noise model replaces the copied spectrum (section 10). Sections 1 to 9 describe the previous method and are kept for reference.

Status 2026-10-07: implemented in the generator (D-238); all 48 noisy records generated without failures in `Thesis-writeup/Data/Coulomb-tanh-and-MSD-lowpass-noise` (replica, about 30 min); the noise-free set is unchanged and shared.

## 1. Goal and instruction
- Goal: a realistic noise level from the Telica data, added in the simulation that generates the training data
- ASMPT supervisors (2026-09-28, via the user): the noise goes on the encoder position, as originally proposed (D-212); the measured noise is a position error, so no conversion to a force
- Jasper: map the SHAPE of the measured spectrum ("het zal niet exact hetzelfde zijn maar de vorm zal wel moeten kloppen", 2026-09-25); the level only about the same
- Supervisors: one noise generator per axis; do not correlate the axes
- In a closed loop the controller reacts to the noise, so it enters during the simulation (Hoekstra et al., open loop, add it to the output afterwards)
- History: from 2026-09-26 to 2026-09-28 the noise was a force on the motor input ("doe het op de input", D-218), calibrated below 300 Hz only, about 12 % of the measured variance on X; superseded

## 2. Step 1: the measured noise level (done)
- Data: the standstill part of every Telica log, before each move (reference constant, no feedforward), 145 logs, 20 kHz
- Signal: servo error per axis, e = r - y_measured
- Level (rms): X1 9.8 nm, X2 10.4 nm, Y 6.3 nm
- Details: `scripts/gantry/closed-loop-noise/REPORT.md` G1; spectrum `outputs/g1_spectra/g1.npz`, exported to `Thesis-writeup/Code/Data/noise/noise_target.mat`

## 3. Step 2: the noise signal
- One independent generator per stage axis (X1, X2, Y), own seed per axis, each shaped by its own axis's measured auto-spectrum; the measured cross-spectra (X1-X2 coherence up to 0.82) are dropped: separate encoders, the correlation is taken as mechanical
- Shape mapping: the simulation's loop filters the injected noise (e = -S v), so each axis's measured spectrum is divided by its own loop sensitivity: phi_j(f) = Phi_e,jj(f) / |S_jj(f)|^2
  - S: K1 with the truth linearised at rest at the record's Y_op, including the tanh friction's slope at rest (cc g per rail; the noise was measured at standstill, where the law is a linear damper)
  - per record Y_op; K2 records (E5) get the K1-calibrated noise, since the sensor does not change with the controller
  - per axis, not the full 3 x 3 loop: a 3 x 3 correction would put the loop's cross-coupling back into v; the per-axis form is analytically exact to 1 % here (the cross-coupling is negligible for the auto-spectra)
- Corner 50 Hz: below it S goes to 0 (integral action) and mapping the shape would need a large slow encoder noise that the stage physically follows, so there the measured spectrum is injected uncorrected (section 7); below the first Welch bin (9.77 Hz) held at that bin; DC zero
- Full band up to 10 kHz: an encoder noise, unlike a force, carries the sensor noise above 200 Hz that is over 90 % of the measured variance
- Generation: a frequency-domain shaping filter on unit complex Gaussian noise, periodic over the 12 s record (`tdg_noise_draw.m`); seed per axis from the record's noise seed (`tdg_seed('<seed>|axis<j>')`)
- The 50 Hz corner and the 10 % shape tolerance are agreed with the user and to be confirmed with Jasper

## 4. Step 3: where it enters the simulation
- Node: the encoder; the controller sees y = q + v
- Simulink, "Extended ODE" loop (the recorded loop): a Sum `EncoderSum` on the branch `Gain4 -> Sum3` (P^T q into the error sum), fed by a From Workspace block `v_in` = [t, v] (interpolation off, sample time ts); `q_aug` still logs the true q (`tdg_make_model.m` edit 2)
- Friction on, as in the truth (tanh, D-224, D-226)
- Every record its own realisation

## 5. What is recorded
- `y`: the MEASURED position q + v, the signal the controller saw, as on the machine; the training reads this
- `y_true`: the true q; `x_logical`, `delta_a`, `Y_trajectory` from the true q (ground truth)
- `u_fb` = lsim(K1, r - y), `u_total` = u_ff + u_fb: the plant force (no noise force)
- `v_enc`: the encoder noise, diagnostics only, never a model input; zeros in the noise-free twin
- `noise_seed`, `meta.noise.axis_seeds`, and all signals in double precision (float32 rounds Y by up to 8.6 nm)

## 6. Checks
- (A) the generator model with encoder noise equals the replica of its loop bitwise (TR-T1 inputs, both versions), and u_plant = u_total per record (the saved y is the signal the controller saw): PASS 2026-09-28, max difference 0, 1.3e-8 N
- (B) shape, standstill at Y 0, friction on at g = 1000, one draw: per axis the simulated / measured rms ratio in the bands above 50 Hz (50-100, 100-300, 300-1000, 1000-10000 Hz) lies within 10 % of one common level. Result 2026-09-28: PASS, band ratios 0.97 to 1.03, common level 1.00, total 9.82 / 10.34 / 6.33 nm against the measured 9.85 / 10.40 / 6.34 nm; 10 to 50 Hz 0.68 / 0.71 / 0.79 (the stated limit). Figure `scripts/gantry/coulomb-tanh-gain/outputs/noise_shape_Y0.png`
- Every record: the rms the noise adds to the measured servo error is logged (noisy minus noise-free). Pilot 2026-09-28 (friction on, g = 1000): 9.96 / 10.50 / 6.41 nm on TR-T1 (mostly dwells), 10.4 / 10.8 / 6.6 nm on TR-S3, 10.4 / 11.0 / 6.6 nm on VA-P1, against the measured 9.8 / 10.4 / 6.3 nm: the friction at rest changes the noise by 1 to 2 %

## 7. Known limits (stated, not corrected)
- Below 50 Hz the shape is not mapped: the simulated error there is 0.7 to 0.8 of the measured (with friction on); under 1 % of the variance
- Below the loop bandwidth the controller follows the sensor noise, so the true position moves with the slow part of v; the recorded y shows only S v of it; this is what an encoder noise does on a real machine too
- The measured "noise" also holds small disturbances (e.g. vibration) that are not sensor noise (CN-012 item 4); all of it is injected as sensor noise
- The correction uses the loop at rest; during moves the tanh law slides (no damper) and the noise sees a slightly different loop
- SNR for comparison with Hoekstra et al.: compute afterwards from the data, relative to the servo error; relative to position (mm to cm motion, nm noise) it is ~140 dB and meaningless

## 8. For training
- `y` is now noisy up to 10 kHz: decimating it to 4 kHz by point sampling folds the 2 to 10 kHz noise into the training band (+28 to 40 % rms). Implemented (D-232, 2026-09-29): in the noisy mode the loader (`gantry_dynamic/data.py`) takes y = r + a zero-phase order-8 Butterworth at 1.6 kHz on y - r, then every 5th sample; both thesis modes drop 5 ms at both record ends (the filter's edge transient; the twins keep identical samples); u keeps its block mean. config.json records Y_DECIMATION and RECORD_TRIM. Evidence and checks: `scripts/gantry/anti-alias/`
- The closed-loop residual rollout u_total + K1 (y_data - y_model) equals f + K1 (r - y_model) exactly only when y_data is the signal K1 saw, which the recorded y is; the rate change (K1 at 20 kHz in the data, 4 kHz in training) breaks it by a leak of the noise into the model input: measured (D-232 check iv) 1.7 to 4.6 nm with point sampling (40 to 60 % of the in-band noise), 0.2 to 0.4 nm with the filter (4 to 7 %)
- `scripts/gantry/noise-training/NOISE-TRAINING.md` was written for encoder injection and applies again; each record's floor is its noisy minus noise-free difference; float64

## 9. Updated 2026-09-28
- `docs/decisions.md` D-225 and its amendment (this approach), D-226 (friction gain 1000), D-218 (superseded)
- Generator: `tdg_config.m`, `tdg_noise_psd.m`, `tdg_noise_draw.m`, `tdg_truth_lin.m`, `tdg_make_model.m`, `gtd_run_simulation.m`, `gtd_save_record.m`, `tdg_post_check.m`, `generate_thesis_data.m`
- Checks: `scripts/gantry/coulomb-tanh-gain/check_encoder_noise.m`, `plot_noise_shape.py`

## 10. Low-order noise model (2026-10-06, after Jasper's feedback)

### 10.1 Why the previous method is replaced
- Jasper (2026-10-05): copying the measured spectrum bin by bin puts dynamics of the real system into the simulation ("nu neem ik dynamica van het echte systeem op"): the measured standstill error was shaped by the real loop and holds real resonances and disturbances that the truth model does not have
- His advice: a very low-order noise model (order 2 or 3), a slope at low frequency and the constant noise floor, or fully white; match the SHAPE of the measured standstill noise, not its exact frequency content
- The sensitivity comes from the controller and the state-space model, not from an FRF
- His follow-up (2026-10-06): keep the low-frequency weighting so the 10 to 100 Hz slope is captured; for Y leave out the region around 1 kHz; the Y behaviour around 100 to 200 Hz is not wanted (see 10.5)

### 10.2 The model
- Node, generators and seeds as before (sections 3 and 4): encoder noise, one independent generator per axis, the controller sees y = q + v
- Same structure on X1, X2 and Y: v = g * filter(b, a, w1) + floor * w2, with [b, a] = butter(2, fc / (fs/2), 'low') and w1, w2 independent white noise
- Three numbers per axis: cutoff fc, low-pass level g, floor level
- A Butterworth low-pass of order 2 has no resonance, so the model cannot reproduce the peaks of the real machine

### 10.3 Where the numbers come from (all simulation based)
- Reference: the measured Telica standstill servo error (section 2, `noise/noise_target.mat`)
- Floor: copied from the measured spectrum, the median above 5 kHz
- Simulation: v through the closed loop at standstill (truth linearised at rest at Y = 0 with the tanh slope, K1, `feedback(eye(3), G*K1)`), the error analysed with the same estimator as the measurement (`pwelch`, Hann 2048, 50 % overlap, 20 kHz); 20 s per run, first 2 s dropped
- fc and g: least-squares fit (`lsqnonlin`) of the log simulated spectrum to the log measured spectrum, all bins from 9.77 Hz, equal weight per decade (the bins are linear in frequency, so without it 1 to 10 kHz would dominate); the noise draws are fixed during the fit, so the cost is deterministic
- Y: 0.7 to 2 kHz (the 1.3 kHz resonance of the real machine) left out of the fit and of the rms
- Level: g then set (`fzero`) so the simulated error rms over the fitted range equals the measured rms there

### 10.4 Result (`temp/noise-lowfit/noise_sim_figure.m`, figure `noise_sim_jasper.png`)
| axis | fc | g [m/sqrt(Hz)] | floor [m/sqrt(Hz)] | rms measured / simulated [m] |
|-|-|-|-|-|
| X1 | 988 Hz | 2.54e-10 | 4.83e-11 | 9.85e-09 / 9.85e-09 |
| X2 | 973 Hz | 2.73e-10 | 4.91e-11 | 1.04e-08 / 1.04e-08 |
| Y (without 0.7 to 2 kHz) | 2596 Hz | 5.64e-11 | 4.57e-11 | 4.81e-09 / 4.82e-09 |
- Y over the full band: 5.48e-09 m simulated against 6.34e-09 m measured; the difference is the 1.3 kHz peak, left out on purpose
- The simulated error follows the measured 10 to 100 Hz slope, the level and roll-off around 1 kHz and the floor on all three axes; the real-machine peaks (100 Hz, 2.5 and 4 kHz on X, 1.3 kHz on Y) are not reproduced
- Final check through the full 3 x 3 loop (all axes together): rms equal to the per-axis fit
- An analytic version (|S_jj|^2 times the noise spectrum, smeared with the Welch window) gave the same cutoffs (980, 967, 2606 Hz); the simulation-based version is kept because it needs no extra smearing step

### 10.5 Notes and limits
- The bump and dip on Y at about 130 and 180 Hz are not in the noise (flat there) but in the simulated loop: |S33| rises to 1.05 at 130 Hz and falls to 0.78 at 180 to 190 Hz, from the hidden absorber at 150 Hz (sqrt(ka/ma)/2pi); this is the simulation's own dynamics; to be explained to Jasper (his point 3)
- Calibrated at standstill and Y = 0; during moves and at other Y the noise is the same, the loop responds slightly differently
- The measured standstill error may hold small disturbances besides sensor noise; all of it is modelled as sensor noise
- Training anti-alias filter (section 8, D-232: Butterworth order 8 at 1.6 kHz in `gantry_dynamic/data.py`) stays as is; it depends on the sample rates, not on the noise shape; the new noise is milder above 2 kHz (floor only); the D-232 leak check is to be rerun on the regenerated data

### 10.6 Open
- Confirm with Jasper (Y at 130 to 180 Hz, see 10.5; figures `temp/noise-lowfit/noise_model_jasper.png`, `absorber_S33_fig.png`)
- Done 2026-10-07: D-238 logged; generator changed (`tdg_config.m`, `tdg_noise_psd.m`, `generate_thesis_data.m`, `tdg_write_manifest.m`); checks on TR-T1 (`temp/noise-lowfit/test_generator.m`): noise-free bitwise equal except the `pair` path, noisy post-check pass, standstill rms ratio 1.007 / 1.000 / 1.005; 48 noisy records generated
- Note: the stored noise-free files' `pair` field still names the old noisy folder; the new noisy files point to the noise-free set correctly
- Next: SNR over all records (`temp/noise-lowfit/snr_pilot.py` generalised), the D-232 anti-alias leak check on the new data (done 2026-10-09, `scripts/gantry/anti-alias/measure_aliasing_lowpass.log`: Q5 leak with the filter 0.22 to 0.50 nm, 4 to 10 % of the in-band noise, against 0.2 to 0.4 nm and 4 to 7 % before; the 1.6 kHz filter stays), and the training config pointed to the new folder
