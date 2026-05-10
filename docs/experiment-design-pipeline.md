# Experiment Design Pipeline

**Status:** Design document — no code written yet.  
**Supersedes:** The ad-hoc approach in `experiment_diagnostics.py` (Python) and
`export_param_recovery_inject_ref.m` / current multisine scripts (MATLAB).

---

## Design Intent and Extensibility

This pipeline is written for the current setup: **no measured noise, complete model,
simulation-based parameter recovery**. It is explicitly designed so that future
additions are natural extensions at well-defined hook points, not retrofits.

The four planned extension axes are:

| Extension | Current assumption | How to extend |
|-----------|-------------------|---------------|
| Noise model | White noise (Phi_v = const, drops out of FIM) | Replace constant with measured Phi_v(w) from encoder/ADC spec in `diagnostics_system.m`. The FIM formula already contains the /Phi_v slot. |
| Additional unmodeled states | Model state-space order matches the true system | The FIM in `diagnostics_system.m` is computed from the current model's A_c, B_c, C_c. When the augmented model is available (neural network augmentation adds the missing states), rerun `diagnostics_system.m` with the augmented state-space. The augmented model may have different poles, a different S(jw), and different dG/dtheta -- so f_low, f_high, and the FIM value per frequency may all shift. No changes to the pipeline structure are required; only the model passed in changes. |
| Additional operating points / scheduling | Single Y per design | `diagnostics_system.m` already computes G, S, v(w) per Y in a loop. Extend aggregation from single-Y to worst-case min_Y(v(w)) before frequency selection. |
| Hardware closed-loop identification | Open-loop BPTT training | The FIM formula is already the closed-loop form. For hardware, switch training to a closed-loop identification framework; the diagnostics output (G, S, f_low, f_high) is reused unchanged. |

**Rule:** When implementing any extension, add a `% EXTENSION: <axis>` label at the
hook point so it is clear which assumption is being relaxed.

---

## Purpose

This document defines:
1. What each file in the experiment design pipeline does.
2. What theory drives each decision (with citations to existing theory docs).
3. What is still an open question before code is written.

Rule: no formula or threshold in any of these files may be written without a
`% THEORY: <source>` or `% HEURISTIC: <reason>` label.

---

## Pipeline Overview

```
[1] diagnostics_system.m
      Characterize the closed-loop system at each Y operating point.
      Saves: diagnostics_output.mat

        |
        v

[2] design_multisine.m
      Design multisine signals using FIM criterion from diagnostics output.
      Saves: multisine_signals.mat  +  trajectory specs

        |
        v

[3] MATLAB simulation (existing Simulink / gantrySystem)
      Run closed-loop simulation with designed multisine.
      Saves: trajectory .mat files (t, fs, u_total, q1, ...)

        |
        v

[4] determine_segment_length.m  (or Python)
      Inspect simulated trajectories to set segment length for BPTT.
      Segment length = N_periods x T_period  (from multisine design).
      Saves: segment_length recommendation to diagnostics_output.mat
```

---

## Guiding Principles

Full theory is in `docs/experiment-design-closed-loop.md`. Summary here.

### Two design dimensions for the multisine

| Dimension | Question | Design variable |
|-----------|----------|----------------|
| Informativeness | Which frequencies must be included? | Frequency line set {f_k} |
| Survival | What amplitude at each frequency? | Amplitude A_k at each f_k |

These are orthogonal. Frequency selection and amplitude shaping are separate steps.  
**Source:** `docs/experiment-design-closed-loop.md` Section 4.

### The unified criterion: Fisher Information Matrix

The FIM combines both dimensions into one quantity. Its inverse is the
Cramer-Rao lower bound on parameter estimation variance.

For **reference injection** (multisine added to reference r):

```
FIM(theta) proportional to
  integral  |dG/dtheta_i|^2  x  |C(jw) S(jw)|^2  x  Phi_r(w) / Phi_v(w)  dw
              informativeness      survival (ref inj)   your design   noise
```

For **force injection** (multisine added directly at plant input, current setup):

```
FIM(theta) proportional to
  integral  |dG/dtheta_i|^2  x  |S(jw)|^2  x  Phi_f(w) / Phi_v(w)  dw
              informativeness      survival (force inj)
```

The difference: for reference injection, survival inside the control bandwidth
is governed by |C S| = |T| which is approximately 1 inside bandwidth — good.
For force injection, survival is governed by |S| which is small inside bandwidth — poor.

**Source:** `docs/experiment-design-closed-loop.md` Section 5 (FIM formula with citations
to Ljung 1999 and Pintelon & Schoukens 2001/2012).

### Injection point: force injection (decided 2026-05-08)

The multisine is injected at the plant input (after the controller output).
Survival is governed by `|S(jω)|²`. Amplitude shaping compensates: `A_k ∝ 1/|S(f_k)|`.

Reference injection was tried and ruled out. The training objective is open-loop:
`||simulate(x0, u_recorded) - q1_recorded||²`. With reference injection, the
controller attenuates `u_ms ≈ 0` inside the bandwidth while `q1_ms ≈ r_ms` is
full amplitude. The residual becomes nearly independent of plant parameters, masking
gradients and causing divergence. Supervisor confirmed: inject after the controller.

**Source**: `docs/decisions.md` D-048. Full analysis: `docs/ref-injection-openloop-incompatibility.md`.

---

## File 1: diagnostics_system.m

### Purpose
Characterize the linearized closed-loop system at each Y operating point.
Output drives the multisine design. This is a system property computation,
not an experiment analysis.

### Inputs
- Physics parameters (from existing MATLAB setup: mb, mh, m1, m2, ...)
- Controller design spec (fbw, ruleOfThumb)
- Y operating points (e.g., 0.0, 0.2, 0.3 m)
- Noise floor estimate (encoder resolution, or white noise assumption)

### Computations
All at each Y operating point:

1. **Open-loop plant G(jw)** from state-space (A_c, B_c, C_c, D=0).
   `% THEORY: G(jw) = C(jwI - A_c)^-1 B_c  -- standard state-space to FRF`

2. **Controller Cfb** from ruleOfThumb at each Y.

3. **Sensitivity S(jw) = 1 / (1 + G(jw) Cfb(jw))**
   `% THEORY: Standard closed-loop sensitivity function`

4. **Transfer from injection to output**
   - Force injection: T_f(jw) = G(jw) S(jw)
   - Reference injection: T_r(jw) = G(jw) Cfb(jw) S(jw)
   Select based on injection point decision (see open questions).

5. **Natural frequencies** from eigenvalues of A_c.
   These are the frequencies where |dG/dtheta_i| is largest (modes).
   `% THEORY: poles of A_c give resonant modes; identification needs those frequencies`

6. **Parameter sensitivity dG/dtheta_i(jw)**
   Finite difference: perturb each parameter by eps, recompute G, take difference.
   `% THEORY: dG/dtheta_i = lim_{eps->0} [G(theta+eps*e_i) - G(theta)] / eps`

7. **FIM value per frequency v(w)**
   ```
   v(w) = sum over i of  |dG(jw)/dtheta_i|^2 * |survival(jw)|^2 / Phi_v(w)
   ```
   where `survival = S` for force injection, `survival = C*S` for reference injection.

8. **Recommended f_low and f_high**
   f_low  = lowest frequency where v(w) is above threshold AND covers at least one mode.
   f_high = highest frequency where v(w) is above threshold.
   These define the frequency band for the multisine.

9. **Recommended fs_new**
   Smallest integer-D candidate satisfying fs_new >= 10 * f_high.
   `% THEORY: Lecture 9 slides 10-12 (5SMB0) -- "10 omega_b <= omega_s"`
   Here omega_b = 2 pi f_high (upper model band, not lower).
   D must be integer: D = round(fs_orig / fs_new).

### Outputs saved to diagnostics_output.mat
- G_f (cell: G evaluated at frequency grid, per Y)
- S_f (cell: sensitivity, per Y)
- v_f (vector: FIM value per frequency, aggregate over Y)
- f_low, f_high (recommended frequency band)
- fs_new, D (recommended sampling rate and decimation factor)
- nat_freqs (natural frequencies per Y)
- dGdtheta_f (cell: parameter sensitivities per frequency, per Y, per parameter)

### What NOT to compute here
- Segment length (depends on trajectories, determined in step 4)
- Multisine amplitudes (determined in step 2)
- tau_max for FRF settling (wrong paradigm, see lessons.md)

---

## File 2: design_multisine.m

### Purpose
Design the multisine excitation using the FIM criterion from diagnostics_output.mat.
One file handles all trajectory types and both injection points (parameterized).

### Inputs
- diagnostics_output.mat (f_low, f_high, v_f, fs_new, S_f, G_f)
- Trajectory specification (nominal motion + ETEL limits)
- Injection point (reference or force)
- Period length T_p (determines frequency resolution Df = 1/T_p)
- Number of channels (3 for MIMO gantry)

### Computations

**Step 1: Frequency line selection**
Integer multiples of Df = 1/T_p in [f_low, f_high].
Select lines where v(f_k) > threshold (from diagnostics).
Minimum line count: F >= 2 * n_params for PE condition.
`% THEORY: Lecture 6 (5SMB0) -- PE order = 2F >= 2 n_theta`
`% THEORY: Lecture 3 (5SMB0) -- leakage-free: f_k = k * Df, T_p = 1/Df`

**Step 2: Amplitude shaping**
A_k proportional to 1 / |survival(j 2pi f_k)|  (force injection)
A_k = constant                                  (reference injection)
`% THEORY: docs/experiment-design-closed-loop.md Section 4 -- amplitude shaping rule`
Then normalize to desired RMS, clip at ETEL limits in simulation.

**Step 3: Phase assignment (Schroeder)**
phi_k = -k(k-1) pi / F  (minimizes crest factor to approximately 1.58)
`% THEORY: Lecture 9 (5SMB0); Schroeder (1970) -- minimum crest factor phases`
`% HEURISTIC: Different random seed per channel -- ensures MIMO PE condition`
`% THEORY: Lecture 9 (5SMB0) -- Phi_u(w) > 0 requires independent channels`

**Step 4: Tiling and simulation check**
Tile to trajectory length. Run check_ref_total on position / velocity / acceleration.
Amplitude sweep: increase RMS until ETEL limit is reached.

### What to AVOID from current multisine code
- Hard-coded frequency ranges (f_high = 100 Hz) not driven by diagnostics
- MIMO crest factor issue: if two modes share a channel, use interleaved odd harmonics
  `% THEORY: docs/decisions.md D-046`
- No `% THEORY` labels
- Magic constants not traceable to any source

### Outputs saved
- One multisine signal per trajectory spec
- Metadata: f_low, f_high, T_p, F (line count), A_k per channel, phi_k per channel

---

## File 4: determine_segment_length

### Purpose
After simulation, look at actual q1 trajectories to determine BPTT segment length.
This is NOT driven by tau_max or FRF estimation rules.

### Principle
The multisine is periodic with period T_p = 1/Df. Each period is identical excitation.
The segment length for BPTT should be an integer multiple of T_p:

```
segment_len = N_periods * T_p * fs_new   (in samples)
```

N_periods is chosen by inspecting the trajectory:
- Does the response look periodic after the first period (transient settled)?
- Is each period long enough to excite all modes at least once?
- Is the total segment short enough to give enough segments for training?

`% HEURISTIC: N_periods = small integer (1-3); not from any specific rule`
`% THEORY: Lecture 3 (5SMB0) -- integer periods give exact periodicity, no leakage`

The key property: with force injection and a periodic multisine, segment boundaries
that align with period boundaries mean each segment sees the same excitation pattern.
This is better for BPTT than arbitrary segment cuts.

### What NOT to use for segment length
- tau_max rule from Lecture 9: that rule is for FRF estimation (transient must settle
  within a segment for spectral accuracy). For BPTT, the transient IS the signal.
  See lessons.md for the context-mismatch rule.
- f_osc_min period rule: 3 / f_osc_min gives the oscillation period, not a segment rule.

---

## Open Questions Before Any Code Is Written

| Question | Why it matters | Where to resolve |
|----------|---------------|-----------------|
| ~~Injection point: reference or force?~~ | **Resolved 2026-05-08**: force injection. See D-048. | — |
| ~~Noise floor estimate~~ | **Resolved 2026-05-08**: white noise assumed, Phi_v = const drops out of FIM. | — |
| ~~Single Y or worst-case over Y?~~ | **Resolved 2026-05-08**: worst-case. Aggregate as min_Y(v(w)) before frequency selection. | — |
| ~~n_params for PE check?~~ | **Resolved 2026-05-08**: n_params = 14. | — |

---

## Cross-References

| Topic | Document |
|-------|----------|
| FIM formula, citations, injection point theory | `docs/experiment-design-closed-loop.md` Sections 4-5 |
| Why current diagnostics are wrong | `docs/diagnostics-theory-basis.md` (old Python approach; superseded) |
| Injection interface in Simulink | `tasks/handoff.md` (archived 2026-05-08) |
| D-046: crest factor for multi-mode channels | `docs/decisions.md` D-046 |
| González et al. aliasing consistency | `docs/experiment-design-closed-loop.md` Section 2 |
| Sampling rate rule (10 omega_b) | Lecture 9 slides 10-12 (5SMB0) |
| Leakage-free design | Lecture 3 (5SMB0); Lecture 13 slides 28-40 |
| Schroeder phases | Lecture 9 (5SMB0) |
