# Diagnostics Theory Basis

**Purpose:** Theory foundation for `experiment_diagnostics.py`. Every numerical formula or threshold must be traceable to a source here before being written in code. Use `# THEORY: <source>` or `# HEURISTIC: <reason>` inline.

**Literature files:** `literature/experiment-design/System-identification/` (Lectures 0–13, 5SMB0)  
**Synthesis:** `literature/experiment-design/System-identification/sysid-experiment-design-notes.md`  
**Session doc:** `docs/experiment-design-closed-loop.md`

---

## Decision 1 — Sampling Rate (fs_new)

### What the code currently does
Uses `_FS_RULE_FACTOR = 8`: selects smallest `fs_new` in `{1000, 2000, 4000, 8000}` satisfying `fs_new ≥ 8 × f_99`. `f_99` is derived from Welch PSD of the measured trajectories.

### The theory (what it should do)

**Source: Lecture 9, slides 10–12 (5SMB0)**
> "10ωb ≤ ωs ≤ 30ωb"

Where ωb is the **system bandwidth** (rad/s), derived from physics — not from signal content.

**Source: Ljung (1999), *System Identification: Theory for the User***
> Setting fs too high causes all discrete-time poles to cluster near unity, degrading numerical conditioning of the parameter estimator.

**Source: Pintelon & Schoukens (2001/2012)**
> The recommended practice is to set fs from the model band (frequencies where the model is trusted), not from the excitation band (frequencies where the signal has power).

**Source: González, van Haren, Oomen, Rojas — arXiv:2410.19629 / IEEE TAC 2024**
> Parametric estimator consistency survives aliasing of out-of-band input content, provided in-band (model band) frequencies are correctly resolved.

### Correct derivation chain

```
Physics (poles) → f_osc_min → system bandwidth → fs_new
```

Not:

```
Signal content (f_99) → fs_new   ← WRONG variable
```

### Formula

```python
# THEORY: Lecture 9 slides 10-12 (5SMB0) — "10ωb ≤ ωs ≤ 30ωb"
fs_new = first candidate in _FS_CANDIDATES >= 10 * f_osc_min
```

`f_osc_min` is the slowest oscillatory natural frequency of the system, computed from the eigenvalues of A_c in `_diag_step_response`. This is already computed in Diagnostic 2.

### Current implementation gap

- `_FS_RULE_FACTOR = 8` is below the lecture's stated lower bound of 10. **Change to 10.**
- `f_99` is the wrong variable. `f_99` should be reported as a **signal content check only** — a warning if f_99 > model band, not the driver of fs_new.
- The correct driver of fs_new is `f_osc_min` from pole analysis (Diagnostic 2), not Welch PSD (Diagnostic 1).

### Role of f_99 after restructuring

`f_99` becomes a **diagnostic warning only**:

```
if f_99 > 10 * f_osc_min:
    WARN: "Excitation contains energy above model band (f_99={f_99:.0f} Hz > {cap:.0f} Hz).
           Content above model band will alias after decimation but should not bias
           in-band estimates (González et al. 2024). fs_new is set from physics, not signal."
```

`f_99` is a **HEURISTIC** metric — a practical signal content summary not defined in the cited literature.

---

## Decision 2 — Segment Length

### What the code currently does
`segment_len = ceil(N_PERIODS / f_osc_min * fs_new)` with `N_PERIODS = 3`.

Only one rule is applied: 3 periods of the slowest oscillatory mode.

### The theory (three independent rules, all must be satisfied)

**Rule A — 10× slowest time constant**  
**Source: Lecture 9, slide 9 (5SMB0)**
> N ≥ 10 × τ_set,95

Where τ_set,95 is the 95% settling time of the dominant (slowest) mode. Equivalent to ≈ 3τ_max (since 95% settling ≈ 3τ for first-order). Giving segment_len ≥ 10 × τ_max × fs_new.

**Rule B — 10× number of parameters**  
**Source: Lecture 9, slide 9 (5SMB0)**
> Minimum N ≥ 10 × n_θ

For 14 parameters: minimum 140 samples. This is a necessary condition for the parameter covariance to be well-conditioned, not a sufficient one.

**Rule C — Multiple complete periods**  
**Source: Lecture 3, periodic measurement material (5SMB0)**  
**Source: Lecture 12, slide examples (5SMB0)**
> "Only then x(t) is exactly periodic: spectrum is exact"
> Lecture 12 examples use 10 periods for FRF estimates.

Supports segment_len = N × (1/f_osc_min). N=3 is the current choice. The lecture examples use 10 periods for FRF quality, but for BPTT parameter recovery the minimum observable length (covering the dominant dynamics) drives this. **N=3 is a HEURISTIC** — justified as "covers one full response of the slowest mode with margin" but not directly specified in the cited sources.

### Correct formula

```python
# THEORY: Lecture 9 slide 9 (5SMB0) — "N ≥ 10 × τ_set,95" and "N ≥ 10 × n_θ"
# THEORY: Lecture 3 periodic measurement (5SMB0) — integer periods required
# HEURISTIC: N_PERIODS = 3 — covers slowest mode with margin; lecture uses 10 for FRF quality
segment_len = max(
    math.ceil(N_PERIODS / f_osc_min * fs_new),   # period rule
    math.ceil(10 * tau_max * fs_new),              # 10× time constant rule
    10 * n_params,                                 # 10× parameter count rule
)
```

### Current implementation gap

`_diag_step_response` only applies the period rule. The time-constant rule and parameter-count rule are not computed. Add both with the `max()` shown above.

### Comparison at current system parameters

| Rule | Formula | Value at fs_new=1000 Hz |
|------|---------|------------------------|
| Period rule (N=3) | 3 / 4.94 × 1000 | 608 samples |
| 10× τ_max | 10 × 1.572 × 1000 | 15720 samples |
| 10× n_params | 10 × 14 | 140 samples |
| **max (correct)** | | **15720 samples** |

> The 10× τ_max rule dominates strongly. The current 608-sample segments are **25× shorter than the Lecture 9 rule requires**. This is likely the primary cause of poor gradient convergence for slow parameters (cy, kb1/kb2).

**Note:** the 10× τ_max rule may be overly conservative for BPTT training (it is derived for stationary FRF estimation, not gradient-based identification). Discuss with supervisor before changing. But the discrepancy should be reported in the diagnostics output.

---

## Decision 3 — Anti-Aliasing Before Decimation

### What the code currently does
`_diag_gradient_convergence` decimates with `q1_dec = traj['q1'][::D]` — naive stride, no filter.

### The theory

**Source: Lecture 9, slide (5SMB0) — pre-processing steps**
> "Apply anti-aliasing filter before any downsampling"

**Source: `docs/trajectory-diagnostic.md` (project notes)**
> "Apply a digital anti-aliasing (decimation) filter with cutoff at f_eff/2 and attenuation ≥40 dB at the new Nyquist frequency"

**Source: `lecture_digital-filters.pdf` (4CM00), slides 30–35**
> The anti-aliasing filter must provide ≥40 dB, preferably ≥60 dB attenuation at fs_new/2.

### Fix

```python
# THEORY: Lecture 9 (5SMB0) pre-processing; lecture_digital-filters.pdf slides 30-35
# scipy.signal.decimate applies Chebyshev Type I filter before striding — use this
from scipy.signal import decimate as _decimate
q1_dec = torch.tensor(_decimate(traj['q1'].numpy(), D, axis=0))
```

`scipy.signal.decimate` applies the anti-aliasing filter automatically. Replace all `[::D]` stride operations with `decimate(..., D)`.

---

## Decision 4 — Welch Window for f_99

### What the code currently does
```python
delta_f = _get_f_osc_min() / 3.0
nperseg = max(256, int(fs_orig / delta_f))
```

### Theory status

> **HEURISTIC.** Welch's method is standard (Lecture 12, 5CTA0 VL13), but the specific choice `delta_f = f_osc_min / 3` is not from any cited source. It ensures at least 3 frequency bins per oscillatory period, which is a practical resolution choice.

The f_99 metric computed from the Welch PSD is itself a **HEURISTIC** — a practical signal content summary. It is not defined in the cited literature and should not drive fs_new (see Decision 1).

Welch parameters:
```python
# THEORY: 5CTA0 VL13 non-parametric spectral estimation — Welch WOSA, Hann window
# HEURISTIC: delta_f = f_osc_min / 3 — ensures ≥3 bins per oscillatory period
# THEORY: 5CTA0 VL13 slide 19 — biased ACF estimator, non-negative PSD guaranteed
window='hann'   # THEORY: VL12 slides 61-93 — Hann window, good sidelobe suppression
```

---

## Decision 5 — FS_CANDIDATES Set

### Current
`_FS_CANDIDATES = (1000, 2000, 4000, 8000)`

### Theory status

> **HEURISTIC.** These are round numbers compatible with integer decimation from 20000 Hz (D=20, 10, 5, 2.5). The set is a practical engineering choice. There is no literature source for this specific set.

The constraint that D must be an integer is a **THEORY** requirement for clean decimation.

---

## Restructuring Plan for experiment_diagnostics.py

Priority order based on theory gaps identified above:

| Priority | Change | Source |
|----------|--------|--------|
| 1 | fs_new derived from `f_osc_min` (Diagnostic 2), not `f_99` (Diagnostic 1) | Lecture 9, Ljung, Pintelon & Schoukens |
| 2 | `_FS_RULE_FACTOR` 8 → 10 | Lecture 9 slides 10-12: "10ωb ≤ ωs" |
| 3 | `segment_len = max(period, 10×τ_max, 10×n_params)` | Lecture 9 slide 9 |
| 4 | Replace `[::D]` with `scipy.signal.decimate` | Lecture 9; lecture_digital-filters.pdf |
| 5 | `f_99` becomes warning-only in report | González et al. 2024 (arXiv:2410.19629) |

**Before implementing any of these:** cite the specific source, confirm variable and context match, label `# THEORY:` or `# HEURISTIC:` inline.
