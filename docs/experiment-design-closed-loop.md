# Closed-Loop Experiment Design for Parameter Recovery

**Session date:** 2026-05-08  
**Context:** Session analysis of `experiment_diagnostics.py` output for multisine vs. base dataset.

---

## 1. The Problem Discovered

Running `experiment_diagnostics.py` with `--grad-check` on the multisine dataset revealed:

| Dataset   | fs_new  | segment_len | Gradient correct |
|-----------|---------|-------------|------------------|
| Base      | 1000 Hz | 608 samples | 11/14            |
| Multisine | 2000 Hz | 1215 samples| 9/14 (worse)     |

The **multisine dataset performed worse** despite being designed to be more informative. Root cause: trajectory T4 has Y-channel content at **195 Hz**, which forces `fs_new = 2000 Hz` via the `f_99` rule. This doubles segment length, and the gradient check showed three additional parameters failing (cb1, cb2, Jh).

The 195 Hz content is **not noise** — it is real broadband multisine excitation above the system's physical bandwidth. Welch's method cannot remove it because it is genuine signal energy.

---

## 2. Physical Bandwidth Cap — Implemented Fix

### Rationale

The system's slowest oscillatory mode is `f_osc_min ≈ 4.94 Hz` (from frozen LTI pole analysis). The system's physical dynamics are fully captured below approximately `10 × f_osc_min ≈ 49 Hz`. Excitation content above this frequency carries **no information** about the parameters we want to identify — it only inflates `f_99` and forces higher sampling rates.

### Rule

```
f_99_capped = min(f_99, k × f_osc_min)    with k = 10
```

- k = 10 is the **canonical engineering rule** ("fs = 10 × system bandwidth"), confirmed by Ljung (1999), Pintelon & Schoukens (2001/2012), and IEEE/NI engineering guidelines.
- Applied to system bandwidth (natural frequency), not excitation bandwidth — this is the key distinction.

### Online Validation

The approach was validated by literature search (2026-05-08 session):

- **Ljung (1999), *System Identification: Theory for the User***: Setting fs too high causes discrete-time poles to cluster near unity, degrading numerical conditioning of the parameter estimator. Argument for not over-sampling relative to system bandwidth.
- **Pintelon & Schoukens (2001/2012), *System Identification: A Frequency Domain Approach***: Explicitly distinguishes between *excitation band* (frequencies where multisine has power) and *model band* (frequencies where you trust the model). fs should be set from the model band.
- **González, van Haren, Oomen, Rojas — arXiv:2410.19629 / IEEE TAC 2024**: "Sampling in Parametric and Nonparametric System Identification: Aliasing, Input Conditions, and Consistency." Confirms: parametric estimator consistency survives aliasing of out-of-band input content, **provided in-band (model band) frequencies are correctly resolved.** Directly supports ignoring the 195 Hz content if the model band ends around 50 Hz.
- **IEEE 4790921 (2009)**: "Sample Rate Effects on Disturbance Rejection for Digital Control Systems." For digital control (real-time), fs ≥ 40× f_BW gives near-continuous performance. For offline identification, k=10 is standard. k=5 is minimum in practice.

### Mandatory Caveat

The physical cap is only safe if an **anti-aliasing filter is applied before downsampling**. High-frequency content (>50 Hz) that folds back into the model band after decimation will appear as in-band noise and bias estimates. `scipy.signal.decimate` applies a Chebyshev Type I filter automatically — verify this is in the pipeline.

### Implementation

In `experiment_diagnostics.py`:
```python
_F99_PHYSICAL_CAP_FACTOR = 10  # k: cap f_99 at k × f_osc_min
```

Applied in `_diag_fft` after computing raw `f_99`:
```python
f_osc_min = _get_f_osc_min()
f_99_capped = min(f_99_raw, _F99_PHYSICAL_CAP_FACTOR * f_osc_min)
```

---

## 3. The Root Problem — Multisine Design was Backwards

The current workflow is:
```
MATLAB generates broad multisine (survive controller)
    → closed-loop simulation
    → diagnostics discover 195 Hz problem
    → apply physical cap as workaround
```

This is backwards. The multisine was designed without regard for the system's model band, then the cap was applied post-hoc.

### The Correct Workflow

```
1. Physics → model band (f_osc_min, poles, bandwidth)
2. Controller → sensitivity function S(jω)
3. FIM optimization → optimal reference spectrum Φ_r*(ω)
4. Design multisine within model band with amplitude shaped by 1/|S(jω)|
5. MATLAB generates this multisine as reference signal
6. Closed-loop simulation → data
7. Diagnostics confirm f_99 ≤ model band (no cap needed)
```

This eliminates the need for the physical cap entirely.

---

## 4. Separation of Survival and Informativeness

These are two orthogonal design dimensions for a multisine, often incorrectly conflated:

| Dimension | Design variable | Addresses |
|-----------|----------------|-----------|
| **Informativeness** | Which frequencies to include | Cover the modes you want to identify |
| **Survival** | Amplitude at each frequency | Compensate for controller suppression S(jω) |

### Survival in closed-loop

The sensitivity function `S(jω) = 1/(1 + C(jω)G(jω))`:
- Inside controller bandwidth: `|S| ≈ 0` — controller suppresses disturbances strongly
- Outside controller bandwidth: `|S| ≈ 1` — controller has little effect

For **reference injection** (adding multisine to reference signal r):
- Signal path: r → y through complementary sensitivity `T = GC/(1+GC)`
- Inside bandwidth: `|T| ≈ 1` — reference injection is **amplified**, works well
- Outside bandwidth: `|T| ≈ 0` — reference injection does not reach the plant

**Key insight:** Reference injection naturally provides good excitation inside the controller bandwidth, which is also where the interesting dynamics are. This makes it the preferred injection point for closed-loop identification (Pintelon & Schoukens).

For **plant input injection** (adding signal after controller output):
- Signal path through `S` — suppressed inside bandwidth
- Need more amplitude inside bandwidth to compensate

### Amplitude shaping rule

```
A(fk) ∝ 1 / |S(j·2π·fk)|    (for plant input injection)
A(fk) = const                  (for reference injection, already amplified by T)
```

---

## 5. Fisher Information Matrix (FIM) for Closed-Loop ID

### What it is

The FIM quantifies how much information the experiment provides about each parameter. Its inverse is the Cramér-Rao lower bound on parameter estimation variance:

```
Var(θ̂) ≥ FIM⁻¹(θ)
```

Optimal experiment design maximizes `det(FIM)` or minimizes `tr(FIM⁻¹)` (D-optimal or A-optimal criterion).

### FIM in closed-loop (indirect method)

With plant `G(q, θ)`, controller `C(q)`, reference `r(t)` with spectrum `Φ_r(ω)`, noise `v(t)` with spectrum `Φ_v(ω)`:

```
FIM(θ) ∝ ∫ |∂G/∂θ|²_{θ₀} × |C(jω)·S(jω)|² × Φ_r(ω) / Φ_v(ω) dω
```

To maximize FIM: concentrate `Φ_r(ω)` where:
1. `|∂G/∂θ|` is large — parameter sensitivity is high (near poles/zeros)
2. `|C·S|` is large — the reference actually influences the plant input
3. `Φ_v(ω)` is small — signal-to-noise is favorable

### Can FIM be done without pre-recorded trajectories?

**Yes.** You need:
1. A nominal model `G(θ₀)` — available from physics (LPV model at true parameters)
2. The controller `C(q)` — known
3. A noise model `Φ_v(ω)` — can be assumed white for a first design

Then the FIM optimization is purely analytical. **No trajectories are required** — you compute S(jω), T(jω), and ∂G/∂θ from the model, then solve:

```
max_{Φ_r(ω₁),...,Φ_r(ωₖ)}  det(FIM)
subject to:  Σ_k Φ_r(ωk) ≤ P_max          (power constraint)
             Φ_r(ωk) ≥ 0  for all k
```

This is a convex optimization problem over the reference amplitudes at chosen frequencies.

### Why this is cleaner than the current approach

| Current approach | FIM-based approach |
|---|---|
| Generate broad multisine → discover bandwidth problem | Solve FIM → multisine frequencies are correct by construction |
| Physical cap as post-hoc workaround | No cap needed |
| Python diagnostics → export → MATLAB redesign | Physics model → analytical optimization → MATLAB gets optimal amplitudes |
| Bandwidth driven by excitation content | Bandwidth driven by model band |

### Parameter sensitivity for the gantry model

`∂G/∂θ` is largest near the system's poles and zeros. For the dual-gantry:
- Oscillatory mode at ~5–35 Hz (varies with Y) — all mass/stiffness parameters are sensitive here
- Low-frequency rigid-body dynamics — coupling parameters (d, cb1, cb2) are sensitive
- The damping parameters (cg1, cg2, cy) require energy at their resonance frequencies to be identifiable

The FIM will naturally concentrate power around 5–35 Hz, within the model band.

---

## 6. Recommended Clean Workflow (for next experiment generation)

```
Step 1: Python (physics)
    from ParameterizedLFRBlock compute S(jω), T(jω) at nominal θ₀
    compute ∂G/∂θᵢ(jω) numerically (finite differences on frequency response)
    
Step 2: Python (optimization)
    solve FIM optimization for Φ_r*(ωk) at k=20–30 log-spaced frequencies in [1, 50 Hz]
    extract optimal amplitudes and phases (random or Schroeder)
    
Step 3: MATLAB
    reads optimal {fk, Ak} from JSON
    constructs multisine r(t) = Σ Ak sin(2π fk t + φk)
    runs closed-loop simulation
    saves trajectory tensors
    
Step 4: Python diagnostics
    verify f_99 ≤ 50 Hz (no cap needed)
    confirm gradient convergence check passes for all 14 parameters
```

**The key change:** the physics model drives the experiment design, not the other way around. The diagnostics become a **verification step**, not a discovery step.

---

## 7. Degenerate Parameter Pairs

Three pairs of parameters only appear as sums in the physics, not individually:
- `kb1 + kb2` — bridge stiffness sum
- `cb1 + cb2` — bridge damping sum  
- `Jb + Jh` — combined inertia

These cannot be individually identified from any experiment — no amount of FIM optimization can fix this. The gradient check consistently shows kb1/kb2 and cb1/cb2 as "wrong" because any update that maintains the sum is equally valid.

**Implication:** reparametrize as `kb_tot = kb1+kb2`, `cb_tot = cb1+cb2`, `J_tot = Jb+Jh` for identification. Only 11 parameters are actually identifiable.

---

## 8. cy Consistently Fails

`cy` (Y-channel damping) fails in both base and multisine datasets. This is not a sampling rate or segment length problem — it is an **excitability problem**:

- Y motion is small in practice (gantry moves in X primarily)
- The Y-damping only affects the Y output channel
- 0.607 s segments may not provide enough Y-channel excitation for cy to be observable

Possible fixes:
1. Include trajectories with deliberate Y-axis motion (larger Y range)
2. Increase segment length specifically for Y-sensitive segments
3. Accept cy as poorly identifiable and regularize it

---

## 9. Open Questions / Pending Implementation

| Item | Status | Notes |
|------|--------|-------|
| Physical cap (`_F99_PHYSICAL_CAP_FACTOR = 10`) | **Planned** | Implement in `experiment_diagnostics.py` |
| Anti-aliasing filter before decimation | **Pending** | `scipy.signal.decimate` has it, verify it's called |
| FIM-based multisine redesign | **Design phase** | Next experiment generation |
| Reparametrize to 11 identifiable params | **Open** | Remove kb1/kb2 degeneracy |
| cy fix (longer Y trajectories) | **Open** | Needs new MATLAB trajectory generation |
| Drop YES/NO columns in gradient report | **Pending** | Show cosine similarity / raw numbers |

---

## 10. Key References

| Source | Claim supported |
|--------|----------------|
| Ljung (1999), *System Identification: Theory for the User* | fs relative to system bandwidth, not excitation bandwidth; clustering of poles near unity when oversampled |
| Pintelon & Schoukens (2001/2012), *System Identification: A Frequency Domain Approach* | Model band vs. excitation band distinction; multisine design for closed-loop ID; reference injection preferred |
| González, van Haren, Oomen, Rojas — arXiv:2410.19629 / IEEE TAC 2024 | Parametric estimator consistency with out-of-band aliasing; in-band resolution is what matters |
| IEEE 4790921 (2009) | Sample rate rules: k=10 for offline ID, k=40 for real-time control |
| 4CM00 lecture: `lecture_digital-filters.pdf`, slides 30–35 | 20× rule for anti-aliasing filter feasibility |
| 4CM00 lecture: `lecture_FRF-measurements.pdf`, slides 41–55 | Leakage, multisine period alignment, reference injection in closed-loop |
| 5CTA0 lecture: `VL13_non-parametric_spectral_estimation.pdf` | Welch WOSA, biased ACF estimator, K≥8 frames |
