# Frozen-Y MIMO FRF Pretest

## Goal

Use simple frozen-Y FRFs to choose the multisine frequency range for later trajectory-plus-multisine experiments.

Questions to answer:

```text
Which plant frequencies are relevant?
How do modes change with Y?
Which modal couplings matter?
What f_low/f_high should the final multisine use?
```

## Core Decision

Do FRFs at fixed Y positions, not during Y sweeps.

Reason:

```text
Y fixed -> local LTI system -> FRF is meaningful
Y sweep -> time-varying LPV data -> standard FRF is smeared
```

## Script-Derived Constants

Use the current MATLAB experiment scripts as the source for practical constants:

```text
source script: Matlab-scripts/generate_identification_experiment_adjusted.m
fs = 20e3 Hz
df = 1 Hz
T_period = 1 / df = 1 s
N_period = round(fs / df) = 20000 samples
Y_ctrl = 0.25 m
```

Interpretation:

```text
fs = 20 kHz supports the 1-300 Hz exploratory FRF band.
N_period = fs / df gives integer-period records and DFT-bin spacing df.
Y_ctrl = 0.25 m keeps the controller design point fixed across experiments.
```

Use the script limits as hard validity checks:

```text
X position limit:      lim.pos_X = +/-0.375 m
Y position limit:      lim.pos_Y = +/-0.400 m
differential/yaw limit: lim.diff = sin(0.1)*Lb
velocity limit:        lim.vel = 2.0 m/s
X acceleration limit:  lim.acc_X = 30 m/s^2
Y acceleration limit:  lim.acc_Y = 50 m/s^2
force peak limits:     lim.force_peak = [2000, 2000, 1420] N
force RMS limits:      lim.force_rms  = [916, 916, 656] N
```

For FRF estimation, always log the actual plant input:

```text
u_total = u_controller + f_multisine
```

The FRF input is the force/current entering the plant.

## Y Grid

Use the trajectory-relevant frozen-Y grid:

```text
Y = [-0.30, -0.25, -0.15, 0.00, 0.15, 0.25, 0.30] m
```

This covers the current trajectory range while staying inside the hardware limit `lim.pos_Y = +/-0.400 m`.

Keep the controller setup fixed across all Y positions. Use the same controller design point as the adjusted MATLAB script:

```text
Y_ctrl = 0.25 m
```

Including `Y = 0.25 m` gives one frozen-Y FRF at the controller design point. Other Y positions intentionally test the same fixed controller away from its design point.

## Modal Coordinates

Use the gantry modes:

```text
common X       -> both X actuators same sign
differential X -> X actuators opposite sign, yaw mode
Y              -> Y actuator
```

Physical modal input directions:

```text
common: [ 1,  1, 0]
diff:   [ 1, -1, 0]
Y:      [ 0,  0, 1]
```

These are the same modal directions used by `mode_def` in the MATLAB experiment scripts.

For quantitative FRF magnitudes, use output coordinates consistent with the model `P` transform. For frequency-range selection, common/diff/Y modal plots are sufficient.

Normalization convention:

```text
common input amplitude a -> FX1 = a, FX2 = a
diff input amplitude a   -> FX1 = a, FX2 = -a
Y input amplitude a      -> FY  = a

y_common = (X1 + X2) / 2
y_diff   = (X1 - X2) / 2
y_Y      = Y
```

Therefore `G_common,common` is average X displacement per per-actuator common force. This is useful for actuator interpretation, but its magnitude includes the chosen input/output normalization.

## Simple MIMO FRF Method

Use one modal input at a time.

At each fixed Y:

```text
1. common-input test
2. diff-input test
3. Y-input test
```

Measure all outputs every time.

Important:

```text
Because the gantry is coupled and controlled in closed loop,
one injected mode is not assumed to equal one plant FRF column.
```

The three modal tests are independent experiments. At each frequency, combine them into matrices:

```text
F_all = [F_common_run, F_diff_run, F_Y_run]
U_all = [U_common_run, U_diff_run, U_Y_run]
Y_all = [Y_common_run, Y_diff_run, Y_Y_run]
```

where `U_all` is the measured total modal plant input and `Y_all` is the measured modal output.

For the first noise-free simulation pretest:

```text
G(jw, Y) = Y_all(jw, Y) * U_all(jw, Y)^(-1)
```

No simultaneous MIMO excitation is needed for the first pretest. One-mode-at-a-time excitation is used only to create independent experiments. Check that `U_all` is full rank and well-conditioned at each frequency.

## Excitation Signal

Use periodic random-phase multisines.

For one period:

```text
u(t) = sum_k A_k cos(2*pi*f_k*t + phi_k)
```

with:

```text
f_k   = integer DFT-bin frequencies
phi_k = random in [0, 2*pi]
A_k   = simple flat amplitude initially
```

Repeat the period.

Periodicity gives clean FRF lines and low leakage. Random phases give genuine realizations; avoid using only seeded Schroeder phase shifts.

Use one scalar multisine per mode:

```text
s_common(t)
s_diff(t)
s_Y(t)
```

Reuse each mode signal at every Y position.

## Sampling And Resolution

Use the current simulation/acquisition rate:

```text
fs = 20 kHz
```

This matches the current MATLAB identification scripts.

For the first exploratory FRF:

```text
df = 1 Hz
T_period = 1 / df = 1 s
N_period = round(fs / df) = 20000 samples
f_lines = 1:300 Hz
```

Requirements:

```text
fs / df must be integer, or choose df so N_period is integer
all excited frequencies must be FFT-bin frequencies
no partial periods enter the FFT
```

If sharp resonances/antiresonances are found, do a later refined scan around those frequencies:

```text
df = 0.25-0.5 Hz
```

## First Excitation Step: Crest Factor

Before choosing the final excitation amplitude or running the FRF tests, design a low-crest-factor signal.

Reason:

```text
same RMS excitation + lower crest factor -> lower actuator peaks
lower actuator peaks -> less saturation and less nonlinear distortion
```

Crest factor is:

```text
CF = max(abs(u)) / rms(u)
```

For the gantry, compute crest factor in physical actuator coordinates after modal mapping, not only in abstract modal coordinates.

For one-mode-at-a-time FRFs:

```text
common -> FX1 = u, FX2 = u
diff   -> FX1 = u, FX2 = -u
Y      -> FY  = u
```

So the one-mode signals have the same crest factor in the active physical actuators as in the scalar modal signal. For later combined-mode tests, compute:

```text
CF_FX1 = max(abs(FX1)) / rms(FX1)
CF_FX2 = max(abs(FX2)) / rms(FX2)
CF_FY  = max(abs(FY )) / rms(FY )

CF_total = max(CF_FX1, CF_FX2, CF_FY)
```

Use simple random candidate selection:

```text
1. choose frequency lines and relative line amplitudes
2. generate many random phase candidates
3. map each candidate to physical actuator forces
4. compute actuator crest factors
5. choose the candidate with the lowest CF_total
6. only then scale to the desired RMS force level
```

Target:

```text
single sine:             CF = sqrt(2) ~= 1.41
selected low-CF candidate: use the lowest found
random-phase multisine:  often around 3-4
poor phase choice:       much higher
```

No fixed CF threshold is required for the pretest. If two candidates have similar crest factor, prefer the one with cleaner actuator margins and no suspicious time-domain bursts.

## Same Signals Across Y

Use the same multisine for the same mode at every Y.

Recommended:

```text
s_common(t) reused at all Y positions
s_diff(t)   reused at all Y positions
s_Y(t)      reused at all Y positions
```

This makes Y-comparison clean:

```text
same input, different Y -> observed change is system/Y dependence
```

Different modes may use different random-phase signals. That is fine.

## Amplitude Choice

For this frozen-Y FRF pretest, amplitude is not trajectory-relative because there is no moving trajectory.

Choose amplitude from:

```text
1. actuator force limits
2. output motion remaining within the local operating region
3. no saturation/clipping
4. response large enough to produce finite, clean FRFs
```

Current simulation has no measurement noise, so SNR/coherence are not the first limiting factors. When noise is added or hardware data is used, amplitude must also satisfy output detectability and coherence checks.

Use the same amplitude for the same mode across Y if possible. If one Y position is too sensitive, reduce the global mode amplitude rather than changing it per Y.

Never choose amplitude as a fixed percentage of hardware limits. Use the hardware limits only as rejection/safety checks after the selected multisine is mapped to physical actuator forces.

## Initial Frequency Range

The FRF pretest itself uses a broad exploratory band:

```text
f_low_pre  = 1 Hz
f_high_pre = 300 Hz
```

Do not choose `f_high` from controller bandwidth.

Choose:

```text
f_low_pre  = low enough for slow plant/coupling dynamics
f_high_pre = high enough to reveal relevant plant resonances/antiresonances
```

Controller bandwidth is a check, not the design limit.

Post-controller force injection can still reveal plant dynamics above controller bandwidth.

Use prior plant clues:

```text
known resonances
model eigenfrequencies
Simulink linearization
sensor/actuator bandwidth
model-validity frequency range
```

If unknown, use a broad low-amplitude scan.

## Final Frequency Range Selection

After FRF plots:

```text
f_low_final  = lowest relevant dynamic/coupling needed
f_high_final = highest relevant resonance/antiresonance/coupling across all Y
```

Include dynamics that:

```text
1. affect common/diff/Y motion
2. shift with Y
3. matter for model/control validation
4. have usable response/coherence
5. are within intended model scope
```

Do not include high-frequency noise or irrelevant unmodeled artifacts just because they appear.

## Periods And Averaging

Use integer periods.

Recommended:

```text
settle at each Y before starting the recorded multisine experiment
1-2 periods settling/transient
10-15 transient-free periods for FRF averaging
```

Frequency resolution:

```text
df = 1 / T_period
```

Choose `T_period` from desired `df`.

Transient removal reason:

```text
FRF estimation assumes periodic steady-state response.
The first periods can contain initial-condition/controller/plant transients.
```

Procedure:

```text
1. move/initialize plant to Y_fixed
2. hold the frozen reference until plant and controller settle
3. start the periodic multisine record
4. split the record into periods of N_period samples
5. discard the first N_drop periods, e.g. N_drop = 1-2
6. FFT each remaining complete period separately
7. average only complete clean periods
8. do not window periodic steady-state periods
```

Do not analyze the DC bin:

```text
do not excite 0 Hz
ignore the 0 Hz FFT bin
remove per-period mean if drift/offset contaminates low frequencies
```

## Logged Signals

Minimum signals to save for every run:

```text
Y_fixed
mode name
fs
df
N_period
period count
excited frequency lines
total modal plant input u_total_modal
physical plant input forces/currents
X1, X2, Y outputs
modal outputs y_modal
controller setup/gains
saturation or clipping flags
```

For closed-loop force injection, the FRF input must be the actual force/current entering the plant, not only a reference signal.

Scope:

```text
This pretest is for the current noise-free simulation only.
No closed-loop noisy estimator is implemented here.
```

Use the same convention as the MATLAB trajectory experiments:

```text
u_total = u_controller + f_multisine
```

Save `u_total` because it is the FRF input.

## Modal Output Transform

For initial FRF plots use:

```text
y_common = (X1 + X2) / 2
y_diff   = (X1 - X2) / 2
y_Y      = Y
```

For quantitative comparison with the model, convert `y_diff` to the `P`-consistent yaw/logical coordinate if needed. Be explicit about the scaling.

## FRF Estimator

The first implementation assumes noise-free simulation data.

At each excited frequency line `f_k` and frozen position `Y_fixed`, form matrices from the three modal experiments:

```text
Y_all(f_k) = [Y_common_run(f_k), Y_diff_run(f_k), Y_Y_run(f_k)]
U_all(f_k) = [U_common_run(f_k), U_diff_run(f_k), U_Y_run(f_k)]
```

Then estimate the plant FRF:

```text
G(f_k, Y_fixed) = Y_all(f_k) * U_all(f_k)^(-1)
```

Use measured total plant input in `U_all`, not only the injected multisine.

```text
u_total = u_controller + f_multisine
```

With multiple clean periods, first compute the FFT for each clean period, then average the complex spectra for each run before forming `Y_all` and `U_all`.

Conditioning check:

```text
rank(U_all(f_k)) = 3
cond(U_all(f_k)) should not be large
```

If `U_all(f_k)` is singular or ill-conditioned, the FRF at that frequency is unreliable.

## Combining Experiments

Combine in three different ways:

```text
periods at same Y/input -> average for lower variance
modal input tests       -> build Y_all and U_all, then invert U_all
Y positions             -> keep separate as G(jw, Y_i)
```

Do not average FRFs across Y positions.

Across Y:

```text
compare the FRF family G(jw, Y_i)
```

## Comparing Across Y

Do not average across Y. Overlay the same FRF element for all Y positions.

Minimum comparison plots:

```text
|G_common,common(f, Y)| for all Y
|G_diff,diff(f, Y)| for all Y
|G_Y,Y(f, Y)| for all Y
```

Then inspect important cross terms:

```text
|G_common,diff|, |G_diff,common|
|G_common,Y|,    |G_Y,common|
|G_diff,Y|,      |G_Y,diff|
```

Look for:

```text
resonance shifts with Y
antiresonance shifts with Y
coupling terms that grow/shrink with Y
frequency where response becomes irrelevant or dominated by artifacts
```

Use these plots to choose the final band:

```text
f_low_final  = below the lowest relevant feature
f_high_final = above the highest relevant feature
```

Add margin around selected band edges. If a sharp mode is near a boundary, extend the boundary or run a refined scan.

## Basic Validity Checks

Required even in noise-free simulation:

```text
input lines are exactly FFT-bin frequencies
input spectrum has energy at all intended lines
no partial periods are used
no actuator saturation/clipping
position/yaw limits remain valid
FRF values are finite at excited lines
same-mode signal is reused across Y
U_all is full rank at each excited line
cond(U_all) is acceptable at each excited line
```

Useful later with noise/hardware:

```text
coherence at excited lines
period-to-period FRF variance
repeat one Y/mode test for repeatability
linearity check at two amplitudes
```

## Optional Later

Not required for first pretest:

```text
multiple random-phase realizations for BLA/nonlinear distortion
simultaneous orthogonal/zippered MIMO multisines
combined-mode validation tests
LPV model fitting directly to FRF family
closed-loop noisy FRF estimator
```

Orthogonal/zippered excitation can improve conditioning, but it is not part of this first noise-free pretest.

## Required Plots

For each Y:

```text
3x3 modal FRF magnitude matrix
input/output spectra
time-domain response and actuator forces
```

3x3 magnitude matrix layout:

```text
rows    = output modes: common, diff, Y
columns = input modes:  common, diff, Y
entry   = 20*log10(abs(G_ij(f,Y)))
```

Matrix entries:

```text
             input common   input diff   input Y
output common     Gcc          Gcd        GcY
output diff       Gdc          Gdd        GdY
output Y          GYc          GYd        GYY
```

Optional per Y:

```text
3x3 modal FRF phase matrix
entry = unwrap(angle(G_ij(f,Y))) in degrees
```

Across Y:

```text
overlay diagonal Bode plots across Y:
  Gcc, Gdd, GYY

overlay important cross-coupling Bode plots across Y if large:
  Gcd, Gdc, GcY, GYc, GdY, GYd

mark resonances/antiresonances
show selected f_low_final/f_high_final
```

Optional:

```text
singular values of G(jw, Y)
coherence or period-to-period variance
```

## Pitfalls

```text
Do not use Y-sweep data as a standard FRF.
Do not average FRFs across Y.
Do not choose f_high because of controller bandwidth.
Do not change the same-mode input signal across Y unless necessary.
Do not use hardware-limit percentage as excitation design.
Do not scale the multisine amplitude before checking crest factor.
Do not interpret moving trajectory-plus-multisine data as stationary FRF data.
Do not forget actuator-coordinate force checks after modal mapping.
Do not treat one injected mode as one plant FRF column in closed-loop coupled MIMO.
```

## Minimal Workflow

```text
1. Choose broad plant-based exploratory frequency band.
2. Set df, T_period = 1/df, N_period = round(fs/df), with fs/df integer.
3. Generate low-crest-factor periodic random-phase multisines for common/diff/Y.
4. Map candidates to physical actuator forces and choose the lowest CF_total.
5. Scale the selected signals to safe RMS force levels.
6. Choose frozen Y grid.
7. Reuse each mode signal across all Y positions.
8. At each Y, settle at the frozen operating point.
9. Run common/diff/Y input tests one at a time.
10. Split into periods, discard transients, FFT clean periods.
11. For each frequency, build Y_all and U_all from the three runs.
12. Estimate G = Y_all * U_all^(-1).
13. Check rank and condition number of U_all.
14. Compare FRF family across Y.
15. Select final f_low/f_high for trajectory-plus-multisine design.
```
