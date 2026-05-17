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

Start simple:

```text
common-X trajectory      -> common-mode multisine
differential/yaw test    -> differential-mode multisine
Y trajectory             -> Y-force multisine
combined validation      -> small MIMO multisine only if needed
```

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

This tests generalization to unseen phase realizations while keeping the experiment design simple.

Disjoint frequency-line grids for train/validation/test are optional later. They are more complex and should only be added if frequency-generalization must be tested explicitly.

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

Preferred when the nominal trajectory has clear motion:

```text
rho_track(i,j) =
    rms(q_ms_i,j - q_nom_i,j) / rms(q_nom_i,j - mean(q_nom_i,j))
```

Alternative when the denominator is near zero:

```text
rho_track(i,j) =
    rms(q_ms_i,j - q_nom_i,j) / range(r_i,j)
```

Meaning: how much the multisine changes the actual trajectory.

## Dataset-Level Aggregation

For each global candidate `rho_design`, aggregate over all trajectories and channels.

Benefit metric:

```text
B(rho) = min_i,j rho_rms(i,j)
```

This is the weakest relative multisine level across the required dataset.

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

For candidate set `R = {rho_1, ..., rho_N}`, normalize:

```text
b(rho) = B(rho) / max_{r in R} B(r)
```

```text
c_peak(rho)  = C_peak(rho)  / max_{r in R} C_peak(r)
c_total(rho) = C_total(rho) / max_{r in R} C_total(r)
c_track(rho) = C_track(rho) / max_{r in R} C_track(r)
```

Conservative combined disturbance:

```text
c(rho) = max(c_peak(rho), c_total(rho), c_track(rho))
```

Decision rule:

```text
Choose the lowest rho at the knee of b(rho) versus c(rho),
after rejecting candidates that violate hard physical limits.
```

Interpretation:

```text
rho too low  -> little added excitation
rho useful   -> added excitation visible, trajectory remains nearly unchanged
rho too high -> disturbance rises faster than useful excitation
```

The final selected `rho` is the smallest global multisine percentage that gives useful added spectral content while keeping the nominal trajectories nearly unchanged.

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
1. rho_design vs B(rho)
2. rho_design vs C_peak(rho)
3. rho_design vs C_total(rho)
4. rho_design vs C_track(rho)
5. b(rho) versus c(rho) benefit-disturbance curve
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
1. Define simple trajectory-only experiments around system dynamics.
2. Run trajectory-only baselines.
3. Estimate slow dynamics from step/clutch responses.
4. Choose f_low, f_high from plant dynamics.
5. Generate normalized multisines.
6. Sweep one global rho_design over all trajectories.
7. Scale each trajectory's multisine by its own trajectory-only RMS.
8. Reject any candidate that violates hard hardware/state limits.
9. Rank remaining candidates using dataset-level benefit-disturbance curves.
10. Choose the lowest useful rho_design.
11. Report with plots, not only text.
```
