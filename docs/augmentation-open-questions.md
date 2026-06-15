# Augmentation: Open Questions and Evidence Map

Status: living document, updated 2026-06-10.

This document tracks the open questions that need answering before the augmentation results can be presented to supervisors. Each question lists what evidence is needed, what already exists, and what is missing.

---

## Dependency Chain

Design choices depend on each other in this order. A wrong choice upstream invalidates everything downstream.

```
System properties (MSD resonance, mass ratio, output equation)
  1. Is the MSD visible in the output?
  2. Downsampling: what sample rate preserves the MSD?
  3. RK4 accuracy: is dt small enough?
  4. nf: does the rollout horizon capture the MSD dynamics?
  5. na_nb: does the encoder history window cover enough cycles?
  6. Encoder type: default vs hybrid, and why?
  7. Data type: multisine vs trajectory, and why?
```

---

## Q1: Is the MSD visible in the output?

**Why it matters:** If the MSD contribution to the output is negligible compared to the main gantry dynamics, no encoder or excitation design will help. The optimizer will spend all capacity fitting the dominant dynamics.

**What we know:**
- MSD resonance at ~150 Hz (10% payload mass) or lower frequency at 50% mass
- Baseline vs augmented FRFs show a resonance/anti-resonance pair where MSD adds information

**Existing evidence:**
- `Matlab-scripts/Augmentation/diagnostics/frf_augmented_vs_baseline.m`: analytical FRF comparison
- `Matlab-scripts/Augmentation/diagnostics/msd_residual_spectrum.m`: paired augmented vs baseline comparison
- `Matlab-scripts/Augmentation/diagnostics/PBH_observability_test_MSD.m`: PBH observability sweep over Y and delta_a
- `scripts/gantry/verification/verify_msd_visibility.py`: checks MSD resonance visibility in training data

**What is missing:**
- Quantitative number: what fraction of output variance comes from the MSD? (e.g. "MSD adds 0.3% to output power")
- How this fraction changes with mass ratio (10% vs 50%)
- Whether the MSD visibility depends on the excitation type (trajectory vs multisine)

---

## Q2: Downsampling

**Why it matters:** The original data is at 20 kHz. Training at 20 kHz is too slow. But downsampling too aggressively destroys the MSD resonance.

**What we know:**
- MSD at 150 Hz: Nyquist requires fs > 300 Hz
- RK4 accuracy needs ~20 steps per oscillation period: fs > 3000 Hz
- Current choice: fs = 4000 Hz (D=5). This satisfies both constraints.
- Previous experiments also tried fs = 1000 Hz (D=20)

**Existing evidence:**
- `scripts/gantry/verification/verify_msd_visibility.py`: tested whether D=20 destroys 150 Hz mode
- `scripts/gantry/verification/verify_decimation_and_device.py`: decimation accuracy test at D=20

**Clean justification method:**
1. Nyquist: fs > 2 * f_MSD (hard floor)
2. RK4: simulate at dt and dt/2 from the same x0, compare trajectories. If they diverge, dt is too large. This gives a numerical answer, not a rule of thumb.
3. Choose the coarsest fs that satisfies both. Document the RK4 convergence test result.

---

## Q3: RK4 Integration Accuracy

**Why it matters:** If the RK4 step (dt = 1/fs) is too large for the MSD dynamics, the physics block produces wrong state derivatives. The model then compensates through the ANN or encoder, learning numerical artifacts instead of physics.

**What we know:**
- At fs=4000 Hz, dt=0.25 ms, the 150 Hz mode gets ~27 steps per period. Should be adequate for RK4.
- Not formally verified.

**Existing evidence:**
- `scripts/gantry/verification/verify_one_step.py`: one RK4 step comparison, but only for the baseline (no MSD)
- `scripts/gantry/verification/verify_data_model_match.py`: rollout comparison for baseline and MSD data

**What is missing:**
- RK4 convergence test for the augmented system: simulate the same trajectory at dt and dt/2, report max state error. One script, one number.

---

## Q4: Rollout Horizon (nf)

**Why it matters:** nf determines how many future steps the training loss sees. If nf is too short, the MSD dynamics don't have time to affect the output. If too long, training is slow and memory-intensive.

**What we know:**
- MSD at 150 Hz: period = 6.7 ms. At fs=4000 Hz, one period = 27 samples.
- Current nf = 400 (100 ms) covers ~15 MSD periods. Should be enough.
- nf = 1200 (300 ms) caused OOM.
- The question is not just "can the MSD oscillate within nf" but "does the loss gradient through nf steps carry enough signal about the MSD?"

**Existing evidence:**
- `encoder_state_recovery.py` tested nf=400 for baseline (no MSD)
- `gantry_interconnect_dynamic.py` uses nf from NF_SECONDS config

**What is missing:**
- Sensitivity analysis: train with different nf values, compare how well the MSD states are recovered. If nf=400 and nf=100 give the same result for MSD states, then nf is not the bottleneck.

---

## Q5: Encoder History Window (na_nb)

**Why it matters:** The encoder maps the last na outputs and nb inputs to the initial state. If the history is too short, the encoder cannot distinguish states that require observing multiple oscillation cycles.

**What we know:**
- Current na_nb = 120 (30 ms at 4 kHz), covers ~4.5 MSD periods
- For the baseline system (no MSD), the encoder already struggles with velocities at this setting

**Existing evidence:**
- `encoder_state_recovery.py`: tests encoder at na_nb=120 on baseline data
- `diagnose_default_encoder.py`: informativity analysis (Fisher discriminant ratio per state)

**What is missing:**
- Same analysis for the augmented system: is the MSD state (delta_a, d_delta_a) informative in the encoder input at na_nb=120?

---

## Q6: Encoder Type (Default vs Hybrid)

**Why it matters:** This is a key design decision that needs justification.

**The argument:**
1. The default encoder learns all states from output loss alone
2. For the gantry, some states (q2/theta, velocities) are weakly observable from the output over short rollouts
3. The hybrid encoder computes the 6 physical states analytically and only learns the augmented states (MSD)
4. This is better because the physical states are known exactly, and the encoder capacity is focused on the unknown augmented states

**Existing evidence:**
- `encoder_state_recovery.py` (running): shows default encoder fails on q2 and velocities for baseline
- `diagnose_default_encoder.py`: per-state error, correlation, gradient analysis
- `diagnose_encoder.py`: three-way comparison (encoder vs analytical vs physics-only)

**What is needed for supervisors:**
- **Baseline (no MSD):** default encoder state recovery results (correlation matrix, scatter plots) showing which states it can/cannot learn. The script is running now.
- **Augmented (with MSD):** same comparison on augmented data, showing the hybrid encoder recovers physical states exactly and only needs to learn delta_a.
- **Output quality comparison:** sim-NRMS for default encoder vs hybrid encoder on the same augmented data. If hybrid gives better NRMS, the case is made.
- **Analytical baseline:** physics-only rollout from analytical x0 (no encoder). This is the ceiling. Shows how much performance the encoder costs.

---

## Q7: Data Type (Multisine vs Trajectory)

**Why it matters:** The user switched from trajectory data to multisine because the encoder was not learning. But it is unclear whether the problem was the data or the encoder itself.

**The argument (if multisine is better):**
- Trajectories excite specific frequency bands (dominated by motion profile shape)
- Multisine provides broadband persistent excitation covering the MSD resonance
- The encoder needs persistent excitation to learn the MSD state from I/O history

**The counter-argument:**
- If the encoder fundamentally cannot learn certain states (as shown in Q6), switching data type does not fix the problem
- The hybrid encoder computes states analytically, so it does not depend on excitation richness for state estimation

**Existing evidence:**
- `encoder_state_recovery.py`: runs both multisine and trajectory experiments (results pending)
- `Matlab-scripts/Augmentation/diagnostics/multisine_frequency_range.m`: eigendecomposition and observability analysis for multisine design
- `Matlab-scripts/Augmentation/data/generate_multisine_data.m`: trajectory + multisine injection
- `Matlab-scripts/Augmentation/data/generate_trajectory_data_without_multisine.m`: trajectory-only data

**What is needed:**
- Compare encoder state recovery on multisine vs trajectory data (the running script does this)
- If both are equally bad: the problem is the encoder, not the data. Multisine may still help for training the augmentation (ANN), but not for the encoder.
- Frequency content comparison: PSD of multisine inputs vs trajectory inputs, overlaid with the MSD resonance frequency. Shows whether the trajectory excites the MSD band at all.

---

## Q8: MSD Excitation Level

**Why it matters:** The MSD mass ratio (10% or 50% of payload) determines how strongly the MSD affects the output. A larger mass ratio makes the MSD more visible but is less realistic.

**What we know:**
- 10% mass ratio: `ma = 0.1 * mh` (realistic)
- 50% mass ratio: `ma = 0.5 * mh` (exaggerated, for debugging)
- Data exists for both: `data/gantry/matlab/multisine/baseline/` and `data/gantry/matlab/multisine/m50/`

**What is missing:**
- Direct comparison: train the same model on 10% vs 50% data. If 50% works and 10% doesn't, the MSD signal is too weak at 10%. This tells you the problem is signal-to-noise, not methodology.

---

## File Map

### MATLAB data generation
| File | Purpose |
|------|---------|
| `Matlab-scripts/Augmentation/data/generate_multisine_data.m` | Trajectory + multisine for baseline or augmented system |
| `Matlab-scripts/Augmentation/data/generate_trajectory_data_without_multisine.m` | Trajectory-only data (T1-T8, V1, E1) |
| `Matlab-scripts/Augmentation/additional_state_lagrangian.m` | Extended Lagrangian with hidden MSD |
| `Matlab-scripts/Augmentation/gantrySystemExtended.m` | 8-state ODE (6 gantry + 2 MSD) |

### MATLAB diagnostics
| File | Purpose |
|------|---------|
| `Matlab-scripts/Augmentation/diagnostics/frf_augmented_vs_baseline.m` | Analytical FRF comparison showing MSD resonance |
| `Matlab-scripts/Augmentation/diagnostics/msd_residual_spectrum.m` | MSD measurability from paired data |
| `Matlab-scripts/Augmentation/diagnostics/multisine_frequency_range.m` | Frequency range design via eigendecomposition and observability |
| `Matlab-scripts/Augmentation/diagnostics/PBH_observability_test_MSD.m` | PBH observability test for hidden MSD state |

### Python training
| File | Purpose |
|------|---------|
| `scripts/gantry/gantry_interconnect_dynamic.py` | Main training script: dynamic parallel augmentation |
| `scripts/gantry/gantry_baseline_validation.py` | Physics block validation against MATLAB ground truth |

### Python encoder diagnostics
| File | Purpose |
|------|---------|
| `scripts/gantry/verification/encoder_state_recovery.py` | Default encoder state recovery test (baseline, no ANN) |
| `scripts/gantry/verification/diagnose_encoder.py` | Pre-training encoder diagnostic (default vs hybrid, 3-way rollout comparison) |
| `scripts/gantry/verification/diagnose_default_encoder.py` | Why default encoder is 100x off: per-state error, gradients, informativity |
| `scripts/gantry/verification/diagnose_convergence.py` | Why encoder fails to converge: 6 targeted tests |

### Python physics verification
| File | Purpose |
|------|---------|
| `scripts/gantry/verification/verify_data_model_match.py` | RK4 rollout matches MATLAB data (baseline and MSD) |
| `scripts/gantry/verification/verify_msd_visibility.py` | MSD resonance visible in training data after decimation |
| `scripts/gantry/verification/verify_baseline_rms.py` | Physics-only baseline RMS (analytical x0, no encoder) |
| `scripts/gantry/verification/verify_one_step.py` | Single RK4 step accuracy |

### Related docs
| File | Purpose |
|------|---------|
| `docs/gantry-augmentation-plan.md` | Implementation plan and phase status |
| `docs/multisine-diagnostics-interface.md` | Theory for multisine design |
| `docs/frozen-y-mimo-frf-pretest.md` | FRF-based frequency range selection |
| `docs/experiment-design-closed-loop.md` | Closed-loop experiment design |
