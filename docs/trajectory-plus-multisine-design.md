# Trajectory Plus Multisine Design

## Purpose

Design identification experiments where the nominal trajectory is the main excitation and the multisine is a small perturbation added on top.

The multisine amplitude must not be chosen as a percentage of hardware limits. Hardware limits are safety constraints only.

Primary design question:

```text
How much multisine can be added relative to the trajectory-only experiment while it remains a small perturbation?
```

## System Context

- System: closed-loop dual-drive gantry / planar positioning system.
- Inputs: actuator forces approximately `[FX1, FX2, FY]`.
- Outputs/states: `[X1, X2, Y]`.
- Useful modal coordinates:
  - common X: `X1 = X2`
  - differential/yaw: `X1 = -X2`
  - Y motion
- Dynamics depend on Y through the mass/inertia matrix `M(Y)`.
- Y is the scheduling variable for the quasi-LPV model.
- Controller should remain fixed across experiments unless explicitly testing controller effects.

## Experiment Structure

Use simple trajectory-only experiments first.

Recommended trajectory groups:

```text
1. Frozen-Y common X
   Y fixed at low/mid/high positions.
   X1 = X2 motion.
   Identifies common-X dynamics versus Y.

2. Frozen-Y differential/yaw
   Y fixed at low/mid/high positions.
   X1 = -X2 or equivalent differential motion.
   Identifies yaw/differential dynamics versus Y.

3. Y-axis motion
   X approximately fixed.
   Y step/sweep trajectory.
   Identifies Y dynamics.

4. Combined validation trajectories
   X common + yaw + Y motion.
   Used mainly for validation after simpler tests work.
```

For early LPV/Y-dependence experiments, prioritize fixed-Y tests at multiple Y positions over complex combined sweeps.

## Frequency Band Design

Choose the multisine band from plant/system dynamics, not from which frequencies escape the controller.

Use the controller analysis only as a feasibility/SNR check.

Choose:

```text
f_low  = lowest relevant plant dynamic or frequency needed for the model
f_high = highest relevant plant mode/resonance/antiresonance to identify
```

This means:

```text
1. identify which plant dynamics the model must capture;
2. choose frequency lines that excite those dynamics;
3. check afterward whether the closed-loop controller allows enough response/SNR.
```

Do not set `f_low` and `f_high` primarily from a closed-loop "which injected frequencies survive" pre-analysis. That pre-analysis is useful only after the plant-relevant band has been chosen.

### Anchoring the Band to the FP Model

The frequency band must be informed by what the existing FP model already captures and where it is expected to be inaccurate.

```text
Before choosing f_low and f_high:
1. Compute the FP model Bode plots for all input-output pairs at several Y positions.
2. Identify frequency regions where:
   - Y-dependent shifts in resonance or antiresonance occur
   - M(Y) approximation introduces model error
   - Modal shapes or coupling change significantly with Y
3. Set the multisine band to cover those regions.
4. Verify the band includes all modes targeted by the augmentation.
```

The residual between FP model and measured data — once available — is the definitive guide. For now use the FP model's known structure to predict where augmentation will be needed.

Use step/clutch responses to estimate slow dynamics and segment length:

```text
T_segment ~ 5-6 dominant time constants
```

or enough cycles of the lowest relevant frequency:

```text
T_segment ~ 3-5 / f_low
```

Use the more conservative value when uncertain.

## Baseline Trajectory-Only Simulation

For every trajectory `i`, first run without multisine:

```text
f_ms_i = 0
```

Save:

```text
q_nom_i         = response without multisine
u_traj_only_i   = controller/actuator force for trajectory-only
r_i             = commanded reference trajectory
```

These trajectory-only signals are the references for multisine scaling and comparison.

## Multisine Amplitude Reference

Define multisine amplitude relative to trajectory-only actuator effort.

For trajectory `i`, actuator or mode `j`:

```text
rms(f_ms_i,j) = rho_design * rms(u_traj_only_i,j)
```

where `rho_design` is one global candidate percentage for the whole dataset.

Examples:

```text
rho_design = 0.01, 0.02, 0.05, 0.10
```

These mean 1%, 2%, 5%, and 10% of the trajectory-only actuator RMS, not hardware capacity.

## Scaling Over Multiple Trajectories

Use one global `rho_design` across the dataset.

For each trajectory, compute the Newton-level multisine amplitude from that trajectory's own trajectory-only RMS:

```text
A_ms_i,j = rho_design * rms(u_traj_only_i,j)
```

This keeps the perturbation meaning consistent:

```text
gentle trajectory     -> small multisine in Newtons
aggressive trajectory -> larger multisine in Newtons
```

Do not use one fixed Newton amplitude for all trajectories unless all trajectory-only force levels are comparable.

## MIMO And Mode Design

Use modal channels first. They are easier to interpret than independent random multisines on `FX1` and `FX2`.

Modal-to-actuator mapping:

```text
common X       -> [ 1,  1, 0]
differential X -> [ 1, -1, 0]
Y              -> [ 0,  0, 1]
```

All RMS, peak, and scaling calculations are done in physical actuator coordinates `[FX1, FX2, FY]` (units: Newtons). Modal multisines are generated in modal space and then mapped to physical forces before any metric is computed.

### Active Channel Rule

Only inject a multisine in the modal channel that is active in the current trajectory. Set all other modal channels to zero.

```text
common-X trajectory:     common-mode multisine only    -> [F, F, 0]
differential/yaw test:   differential-mode only        -> [F, -F, 0]
Y trajectory:            Y-channel only                -> [0, 0, F]
combined validation:     MIMO multisine only if needed
```

This avoids the problem of scaling a multisine from a near-zero trajectory RMS in an inactive channel (e.g., the Y-hold force during a common-X test). Cross-channel multisine injection is only considered in combined experiments at a later stage.

Random multisine design:

```text
1. For each active modal channel, generate a random-phase multisine.
2. Map the modal signals to physical actuator forces [FX1, FX2, FY].
3. Compute crest factor on the physical actuator forces.
4. Repeat for many random phase candidates.
5. Choose the candidate with the lowest combined actuator crest factor.
```

Crest factor per actuator:

```text
CF_j = max(abs(F_j)) / rms(F_j)
```

Candidate score:

```text
CF_candidate = max_j CF_j
```

Choose the candidate with the smallest `CF_candidate`.

This keeps the excitation robust in the actual actuator coordinates while preserving modal interpretation.

For later MIMO multisines:

- use random phases per channel/mode;
- generate multiple phase realizations;
- choose the realization with the lowest combined crest factor;
- check combined actuator peaks after summing all active modes.

Odd-only frequency grids are only needed if nonlinear distortion detection is an explicit goal.

## Train / Validation / Test Multisines

Training, validation, and test sets must not reuse the exact same multisine time signals.

Use the same plant-relevant frequency band/grid initially, but use different random phase realizations:

```text
train: random phase candidate pool with train seeds
val:   random phase candidate pool with validation seeds
test:  random phase candidate pool with test seeds
```

For each split:

```text
1. generate random-phase modal multisines;
2. map to [FX1, FX2, FY];
3. select the lowest-crest-factor candidate for that split;
4. scale relative to the trajectory-only RMS.
```

Different phase realizations check FRF variance: a well-identified frequency response should be consistent across realizations. This is a variance/repeatability check, not a model generalization test.

For model generalization — testing whether the augmented model works at unseen operating conditions — the splits must differ in Y-position or trajectory type, not just phase:

```text
FRF variance check (phase split):
    train/val/test = different random phase seeds, same Y, same trajectory type

Model generalization check (condition split):
    train: Y in {low, mid}, standard trajectory shapes
    val:   Y in {mid}, different trajectory shape
    test:  Y in {high}, unseen Y-position
```

Decide which goal applies before generating the splits. Both serve different purposes and should not be conflated.

Disjoint frequency-line grids for train/validation/test are optional. Add them only if frequency-generalization must be tested explicitly.

## Metrics Per Trajectory

For each candidate `rho_design`, generate all trajectories with multisine and compare against trajectory-only baselines.

For trajectory `i`, actuator/mode `j`:

### RMS Force Ratio

```text
rho_rms(i,j) = rms(f_ms_i,j) / rms(u_traj_only_i,j)
```

Meaning: extra broadband force relative to nominal trajectory effort.

### Peak Force Ratio

```text
rho_peak(i,j) = max(abs(f_ms_i,j)) / max(abs(u_traj_only_i,j))
```

Meaning: peak perturbation relative to nominal trajectory peak. Catches crest-factor problems.

### Total Force Increase

```text
rho_total(i,j) = rms(u_total_i,j) / rms(u_traj_only_i,j)
```

where:

```text
u_total_i,j = u_traj_only_i,j + f_ms_i,j
```

Meaning: how much more demanding the experiment becomes after injection.

Use excess form when ranking:

```text
rho_total_excess(i,j) = rho_total(i,j) - 1
```

### Trajectory Distortion Ratio

```text
rho_track(i,j) =
    rms(q_ms_i,j - q_nom_i,j) / max(range(r_i,j), delta_min)
```

where `delta_min` is a small floor to prevent division by near-zero on frozen-axis tests (e.g., 0.1 mm for position channels, 0.5 mrad for yaw). Choose `delta_min` from the positioning repeatability of the hardware.

Use `range(r_i,j)` (the span of the commanded reference) as the normalizer for all trajectory types. This keeps cross-trajectory comparisons consistent: frozen-Y tests and moving-Y tests use the same formula.

Do not switch between two different denominators depending on the trajectory type. That makes dataset-level aggregation meaningless.

Meaning: how much the multisine changes the actual trajectory relative to the commanded motion range.

Note on `rho_total`: `rms(u_total)^2 = rms(u_traj)^2 + rms(f_ms)^2` holds approximately when the trajectory and multisine occupy different frequency regions. State the multisine frequency band explicitly so this assumption can be verified.

## Dataset-Level Aggregation

For each global candidate `rho_design`, aggregate over all trajectories and channels.

Benefit metric:

```text
B(rho) = min_i,j rms(q_ms_i,j - q_nom_i,j)    [physical units: mm or mrad]
```

This is the smallest output perturbation produced across all active trajectory/channel combinations. It answers: "does the multisine actually move the system?"

`rho_rms` by itself is not a useful benefit metric — it is circular (it equals `rho_design` by construction) and says nothing about whether the perturbation is physically detectable.

A perturbation that is large enough should produce output deviations clearly above positioning repeatability. Report `B(rho)` in absolute units and compare it to the known positioning repeatability of the hardware.

Disturbance metrics:

```text
C_peak(rho)  = max_i,j rho_peak(i,j)
C_total(rho) = max_i,j rho_total_excess(i,j)
C_track(rho) = max_i,j rho_track(i,j)
```

These are the worst disturbances across all trajectories.

Hardware margins are still checked separately:

```text
force_margin(rho)
velocity_margin(rho)
acceleration_margin(rho)
yaw_margin(rho)
position_margin(rho)
```

Any candidate that violates a hard physical limit is rejected.

## Ranking Candidate Amplitudes

Do not use hardcoded arbitrary thresholds unless they come from hardware or explicit experiment requirements.

Rank candidates by added excitation versus disturbance.

After rejecting hardware-limit violations, rank the remaining candidates by plotting:

```text
x-axis: rho_design
y-axis (benefit):     B(rho)      [mm or mrad — absolute output perturbation]
y-axis (disturbance): C_peak(rho), C_total(rho), C_track(rho)
```

Decision rule:

```text
Choose the lowest rho_design where B(rho) is clearly above the hardware
positioning repeatability, while C_peak, C_total, and C_track remain
at acceptable levels.
```

Do not normalize and then look for a "knee." Both B(rho) and the disturbance metrics scale roughly linearly with rho for small perturbations, so normalization does not reveal structure. Use absolute units and hardware knowledge instead.

Interpretation:

```text
rho too low  -> B(rho) below positioning repeatability, perturbation not reliably detectable
rho useful   -> B(rho) clearly above repeatability, disturbance metrics still small
rho too high -> trajectory distortion or force peaks become unacceptable
```

The final selected `rho` is the smallest global multisine percentage that produces a physically detectable output perturbation while keeping trajectories and actuator loads nearly unchanged.

## Required Plots

Per trajectory and candidate `rho_design`:

```text
1. q_nom vs q_ms
2. q_ms - q_nom
3. r, q_nom, q_ms overlay
4. u_traj_only, f_ms, u_total
5. actuator peak/RMS bars
6. velocity, acceleration, yaw checks
7. PSD/input spectrum: trajectory-only vs multisine vs total
8. state-space plots
```

Dataset-level:

```text
1. rho_design vs B(rho)           [absolute mm or mrad, with repeatability floor marked]
2. rho_design vs C_peak(rho)
3. rho_design vs C_total(rho)
4. rho_design vs C_track(rho)
5. per-trajectory breakdown of rho_track and rho_peak  [to see which trajectory is the outlier]
6. hardware margin versus rho_design
```

## Implementation Notes For `generate_identification_experiment.m`

Current behavior to change:

```text
amp_vec(m) = A_max(md.A_idx)
```

This scales from hardware-derived limits. Replace with trajectory-relative scaling:

```text
1. Simulate trajectory-only.
2. Compute u_traj_only RMS per actuator/mode.
3. Generate normalized multisine with RMS = 1.
4. Scale multisine:

   f_ms_i,j = rho_design * rms(u_traj_only_i,j) * f_norm_i,j

5. Simulate trajectory + multisine.
6. Compute metrics and plots.
7. Repeat for candidate rho_design values.
```

Save both cases:

```text
trajectory-only
trajectory-plus-multisine
```

so that all comparisons are reproducible.

## Final Workflow

```text
1.  Define simple trajectory-only experiments (one per modal DOF, multiple Y positions).
2.  Run short step/clutch responses to estimate dominant time constants.
3.  Choose T_segment from dominant time constants: T_segment ~ 5-6 * tau_dominant.
4.  Inspect FP model Bode plots to identify where augmentation is needed.
5.  Choose f_low, f_high to cover those regions plus all Y-dependent modes.
6.  Run full trajectory-only baselines at the chosen segment length.
7.  Generate normalized modal multisines (active channel only per trajectory).
8.  Map to physical actuator forces [FX1, FX2, FY].
9.  Minimize crest factor over phase candidates in physical actuator coordinates.
10. Sweep one global rho_design; scale each trajectory by its active-channel trajectory RMS.
11. Reject any candidate that violates hard hardware/state limits.
12. Rank remaining candidates: plot B(rho) vs rho_design and disturbance metrics vs rho_design.
13. Choose the lowest rho_design where B(rho) exceeds positioning repeatability.
14. Report with plots, not only text.
```
