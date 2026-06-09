# Session Handoff — Multisine Excitation Design for Gantry Augmentation

**Last written**: 2026-06-09

---

## Background and motivation

The gantry augmentation pipeline uses a hybrid encoder (analytical physical states + learned augmented states via ANN). The hybrid encoder achieves **val sim-RMS = 0.000130** at initialization (ANN=0), which is better than the physics-only baseline (0.000262). This proves the encoder bottleneck is solved.

However, **training makes things worse** (val sim-RMS jumps to 0.05-0.29). The ANN augmented states have nothing useful to learn because the hidden MSD dynamics (150 Hz resonance) are not excited in the current trajectory data. The MSD IS present in the model (delta_a RMS = 3.37 um) but the current trajectories only excite it at 0-50 Hz, not near its resonance.

**Next step**: Design and generate multisine excitation data that excites the MSD resonance so the ANN can learn the augmented dynamics.

---

## What needs to be done

### 1. Determine multisine frequency range from analytical plant dynamics

The frequency band must be driven by the plant dynamics (NOT controller bandwidth). Key references:
- `model_augmentation/systems/gantry_ss.py` has all physical parameters, masses, stiffnesses, damping
- The MSD resonance is at 150 Hz (fa=150, from `Matlab-scripts/Augmentation/data/generate_trajectory_data_without_multisine.m` lines 39-46)
- The system's dominant dynamics are below ~50 Hz, but the MSD adds a 150 Hz mode
- Supervisor guidance: base flow and fhigh on relevant dynamics range, not on controller escape frequencies

Use the state-space model to analytically determine:
- Where the MSD resonance sits (150 Hz, confirm from eigenvalues)
- What frequency range captures the MSD effect on outputs
- Upper bound should include 150 Hz; lower bound covers the baseline dynamics
- Sampling at 1 kHz (Nyquist = 500 Hz) is sufficient for 150 Hz

### 2. Generate random-phase multisines and select low crest factor

Supervisor instruction: "generate multiple, choose the lowest crest factor. Need to work robust, not optimal."

Implementation approach:
- Generate N random-phase MIMO multisines (different random phases per channel)
- Compute crest factor for each
- Select the ones with lowest crest factor
- deepSI has `deepSI.exp_design.multisine(..., n_crest_factor_optim=N)` which does exactly this

Reference implementations:
- `scripts/ecc_2025/msd_ndof_data_generation_dynamic.py` uses deepSI multisine with crest factor optimization
- `scripts/encoder_initialisation/system_simulation.py` same pattern
- `Matlab-scripts/multisine_muli_traject.m` Schroeder-phase multisine in MATLAB

### 3. Determine safe amplitude (critical, previous attempt used too much)

Previous issue: 40% excitation saturated actuators. Key constraint from meeting notes:
- Must visualize the total signal (trajectory + multisine) before running
- Plot the response with and without multisine to see the difference
- Check against hardware limits (actuator saturation, angle limits at 72 mm)
- The amplitude must be determined relative to what the system already experiences from the trajectory alone

The amplitude translation: F_equiv = M_eff * (2*pi*f)^2 * A. See `Matlab-output/parameter-recovery-multisine/multisine-analysis.md` for previous amplitude scan results.

### 4. Verify the data is informative

Before committing to long training runs, verify the multisine data actually excites the MSD:
- Run the existing `scripts/gantry/verification/verify_msd_visibility.py` on the new data
- Check that the PSD shows energy at 150 Hz
- Check that residuals (measured - physics-only) are larger with multisine than without
- Run `diagnose_encoder.py --encoder hybrid` on the new data and check if ANN states become non-zero during training (val sim-RMS should decrease below 0.000130)

### 5. MATLAB integration

The multisine needs to be injected in the MATLAB Simulink model that generates trajectory data. The current data generation script is `Matlab-scripts/Augmentation/data/generate_trajectory_data_without_multisine.m`. A version WITH multisine needs to:
- Add the multisine as a reference injection or force injection signal
- Save the same output format (u, y, delta_a per trajectory)
- Generate T1-T8 + V1 + E1 with multisine applied

---

## Key files to read

| What | Where |
|------|-------|
| **Gantry system matrices** | `model_augmentation/systems/gantry_ss.py` |
| **Current data generation (no multisine)** | `Matlab-scripts/Augmentation/data/generate_trajectory_data_without_multisine.m` |
| **Previous multisine MATLAB script** | `Matlab-scripts/multisine_muli_traject.m` |
| **Previous multisine export** | `Matlab-scripts/export_param_recovery_multisine.m` |
| **deepSI multisine reference** | `scripts/ecc_2025/msd_ndof_data_generation_dynamic.py` |
| **Multisine design theory** | `docs/multisine-diagnostics-interface.md` |
| **Trajectory + multisine design** | `docs/trajectory-plus-multisine-design.md` |
| **Meeting notes on multisine problem** | `meeting-notes/Meetings Tue/13-05-2026-joint-meeting-multisine-problem.txt` |
| **Previous amplitude analysis** | `Matlab-output/parameter-recovery-multisine/multisine-analysis.md` |
| **MSD visibility diagnostic** | `scripts/gantry/verification/verify_msd_visibility.py` |
| **Encoder comparison diagnostic** | `scripts/gantry/verification/diagnose_encoder.py` |
| **Training script** | `scripts/gantry/gantry_interconnect_dynamic.py` |
| **Hybrid encoder** | `model_augmentation/utils/torch_nets.py` (`HybridGantryEncoder`) |

---

## Supervisor constraints (from meeting 2026-05-13)

1. Frequency range from plant dynamics, not controller escape
2. Random-phase MIMO multisine, select lowest crest factor
3. Conservative amplitude, must not saturate actuators (40% was too much)
4. Plot total signal (trajectory + multisine) before running experiments
5. Keep controller constant across all experiments
6. Document everything, even failures, with plots

---

## What has been completed this session

- Hybrid encoder (`HybridGantryEncoder`) implemented and verified in `model_augmentation/utils/torch_nets.py`
- `USE_HYBRID_ENCODER` toggle in training script
- `diagnose_encoder.py` updated for parallel default vs hybrid comparison with separate log files
- Confirmed hybrid encoder works: val sim-RMS = 0.000130 at init (better than physics-only 0.000262)
- Confirmed ANN has nothing to learn from current data (training makes things worse)
- MSD visibility fully characterized via `verify_msd_visibility.py` (excited at 0-50 Hz, not at 150 Hz resonance)
