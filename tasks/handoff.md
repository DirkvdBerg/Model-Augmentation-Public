# Session Handoff

_Previous sessions archived to `archive/sessions/`._

**Last written**: 2026-05-10 by Claude (Sonnet 4.6)

---

## Current Goal

Design and implement a clean, theoretically justified experiment design pipeline for the
gantry identification experiment. The pipeline replaces `export_param_recovery_multisine.m`,
which is not being modified further (choices not grounded in theory, scan-based diagnostics
are not safe for hardware).

---

## What Was Decided This Session

### diagnostics_system.m — partial implementation, broken f_high

Script exists at `Matlab-scripts/diagnostics_system.m`. Current state:
- Probes Y-channel only (channel 3)
- Computes S_hat and G_hat from closed-loop simulation
- f_low works: reads lowest freq where |S_hat|^2 > 0.1
- **f_high is broken**: Y-channel is a free mass (K(3,3)=0, no spring), so |G_hat_y| is a
  pure -40 dB/dec double integrator with no resonance peaks. findpeaks finds nothing.

Fix agreed: probe all 3 modes using a clean f_vec projection structure (see below).
**Do not use diagnostics_system.m until this is fixed.**

### 3-mode probe design (agreed, not yet implemented)

Probe all 3 diagonal modes in closed-loop simulation. For each mode c:
- Inject: `f = sig * ch(c).f_vec` (N x 3 force matrix, one line)
- Extract modal signals by projecting q1 and u_tot onto the same f_vec:
  `f_modal = f_s * fv'`, `u_modal = u_tot * fv'`, `q_modal = q1 * fv'`
- Compute S_hat(:,i,c) = |FFT(u_modal) / FFT(f_modal)|
- Compute G_hat(:,i,c) = |FFT(q_modal) / FFT(u_modal)|

Mode definitions (motor coordinate forcing):

| Index | Name   | f_vec      | What it excites         | Expected G_hat |
|-------|--------|------------|-------------------------|----------------|
| 1     | common | [1, 1, 0]  | X symmetric (rigid body)| double integrator, no peaks |
| 2     | diff   | [1,-1, 0]  | theta (tilt mode)       | resonance from kb1+kb2 — gives f_high |
| 3     | y      | [0, 0, 1]  | Y translation           | double integrator, no peaks |

Outputs (worst-case across all modes and Y operating points):
- f_low = max over (mode, Y) of lowest freq where |S_hat|^2 > 0.1
- f_high = max over (mode, Y) of last peak in |G_hat| — will come from diff mode

Probe parameters:
- Y_vals = [-0.4, -0.2, 0.0, 0.2, 0.4] (5 operating points, hardware limits +/-0.4 m)
- amp_rms = 50 N for all modes (well within limits: 916/916/656 N RMS for FX1/FX2/FY)
- N_period = 20000 (1 s period, f0 = 1 Hz), N_periods = 2, use last period only
- Probe: odd harmonics 1 Hz to Nyquist, zero-phase cosines (phase cancels in ratios)
- Total simulations: 3 modes x 5 Y points = 15

### Amplitude for multisine injection (agreed, grounded in theory)

From diagnostics outputs, for each mode c, the maximum safe force amplitude is:

```
A_max(c) = min(
    F_limit_rms(c),                              % hardware actuator limit
    pos_limit(c) / max(G_hat(:,:,c) .* S_hat(:,:,c))  % position response limit
)
```

Where:
- F_limit_rms: hardware RMS limits [916, 916, 656] N per motor, from TELICA spec
- pos_limit: hardware position limits [0.375, 0.375, 0.400] m per channel
- G_hat * S_hat: the plant response per unit injected force at each frequency

This replaces the empirical scan in export_param_recovery_multisine.m.
The hardware limits are NOT declared in diagnostics_system.m — they come from the
trajectory/multisine script (which owns the hardware specs).

### New file structure (agreed)

Do NOT modify `export_param_recovery_multisine.m`. Create two new files:

1. `Matlab-scripts/diagnostics_system.m` (fix existing) — outputs: f_low, f_high,
   S_hat, G_hat per mode and Y. Saves to `Matlab-output/step0_outputs.mat`.

2. `Matlab-scripts/generate_identification_experiment.m` (new) — loads step0_outputs.mat,
   designs multisine from f_low/f_high/S_hat/G_hat + hardware limits, generates trajectories,
   validates everything, exports .mat files for BPTT training.

---

## Next Session Goal

**Design** `generate_identification_experiment.m` — no code yet. For each design choice:
- Justify from theory (P&S, Lecture notes, Schroeder 1970) or hardware limits
- Document what is a THEORY, what is a HEURISTIC, what is a hardware constraint

Key decisions to make in next session:
1. Which trajectories to generate (T1-T8 or redesign from scratch)?
2. Frequency band: f_low to f_high per mode (from diagnostics output)
3. Amplitude: A_max(c) formula above — is the pos_limit or force_limit binding?
4. Phase: Schroeder 1970 (justified for crest factor minimization, P&S Ch.2)
5. Odd harmonics only (justified: PE condition F >= 7, P&S Ch.4 §4.3.2)
6. How to handle multi-mode injection (common + diff + y simultaneously vs sequential)
7. What to validate before saving (force limits, position limits, velocity limits)

---

## Open Blockers (carried forward)

- **Sample rate**: D-012 — 16 kHz (main.m) vs 20 kHz (ETEL spec), unresolved
- **Float32 acceptability**: Run training in both dtypes, compare param_table()
- **MIMO decorrelation**: Phase offset insufficient per Pintelon et al. (2011) — declared limitation

---

## Key Files

| What | Where |
|------|-------|
| Theory validation | `docs/theory-validation.md` |
| Design decisions | `docs/decisions.md` |
| Diagnostics script (needs fix) | `Matlab-scripts/diagnostics_system.m` |
| Old multisine script (reference only, do not modify) | `Matlab-scripts/export_param_recovery_multisine.m` |
| Lessons (read before anything) | `tasks/lessons.md` |
