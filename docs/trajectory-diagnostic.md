# Trajectory Diagnostic

## Purpose

Before training or precomputing anything, two decisions must be made from the raw trajectory data:

1. **Effective sampling rate** — what rate is actually needed to capture the system dynamics?
2. **BPTT segment length** — how many samples must a segment contain for the loss to discriminate true parameters from detuned alternatives?

These decisions are made once, cached to disk, and consumed by `precompute.py`. Neither can be made before the trajectories exist, because both depend on the actual frequency content and time constants observed in the data.

---

## Why this must happen before precompute

`precompute.py` builds the training dataset: it loads trajectories, computes sigma, and chooses a segment length. All of these depend on the sampling rate and segment length chosen here. Decimation must happen before the precompute cache is built — training should never touch raw oversampled data.

Pipeline order:

```
MATLAB trajectory generation  (raw .mat at hardware rate, e.g. 20 kHz)
        ↓
trajectory_diagnostic.py      (FFT → decimation factor → step response → segment length)
        ↓
precompute.py                  (works on decimated data, uses chosen segment_len)
        ↓
train_param_recovery.py
```

---

## Decision 1 — Effective sampling rate

### Rationale

**From supervisor meeting notes (2026-04-24):**
> *"take the FFT of the trajectories and then we'll see what the frequencies are"*
> *"20 kHz might be overkill — expect from similar systems 4 kHz might be enough already"*
> *"ASMPT: over what range of frequency do you want to have a good model"*

**From course lecture:**
BPTT cost scales as O(n²) in segment length. If the system has no meaningful energy above 2 kHz but is sampled at 20 kHz, every segment is 10× longer than necessary.

---

### The 20× rule — do not use bare Nyquist

**From course lecture** (`lecture_digital-filters.pdf`, 4CM00, slide 35):

> **Rule of thumb: fs ≥ 20 × bandwidth**

The Nyquist minimum (fs > 2 × f_max) is **not sufficient** in practice. A practical anti-aliasing filter cannot be a brick wall — it has a transition band. To allow the filter to attenuate content above fs/2 by ≥40–60 dB, the usable signal band must stay well below fs/2. The 20× rule ensures the filter only needs to attenuate content approximately one decade above the signal band, making a practical Butterworth or Chebyshev filter feasible.

> Source: `lecture_digital-filters.pdf` (4CM00), slides 30–35

The required anti-aliasing filter attenuation before any A/D conversion or decimation step is **≥40 dB, preferably ≥60 dB, at fs/2**.

> Source: `lecture_digital-filters.pdf` (4CM00), slides 30–35; `lecture_FRF-measurements.pdf` (4CM00), slides 54–59

---

### Zero-order hold phase lag

**From course lecture** (`lecture_digital-filters.pdf`, 4CM00, slides 6–10):

Each discrete-time step introduces a zero-order hold (ZOH), which adds an effective delay of T/2 (half a sample period). The resulting phase lag at bandwidth frequency ω_bw is:

> φ_hold ≈ ω_bw × T/2 radians

This is an additional argument for keeping T small (fs large relative to bandwidth). At fs = 20× BW, the ZOH phase lag at the bandwidth is ≈ π/20 ≈ 9°, which is acceptable. At fs = 2× BW (Nyquist minimum) it would be ≈ 90°, which is destructive.

---

### Method: FFT-based bandwidth identification

**From supervisor meeting notes (2026-04-24):**
> *"take the FFT of the trajectories and then we'll see what the frequencies are"*

**Procedure (informed by course lectures):**

1. Load all raw trajectories. Compute the **power spectral density (PSD)** of each output channel (X1, X2, Y) using **Welch's method** (MATLAB `pwelch` / `scipy.signal.welch`).
2. Use the **biased ACF estimator** when computing the PSD via the correlogram route. The biased estimator guarantees a non-negative PSD; the unbiased estimator can yield physically inadmissible negative values.
   > Source: `5CTA0_VideoLectures_VL13_non-parametric_spectral_estimation.pdf`, slides 20–35
3. Find `f_useful`: the frequency above which the cumulative power across all channels and trajectories exceeds a threshold (e.g. 99% of total power).
4. Apply the 20× rule: target effective sampling rate `f_eff = 20 × f_useful`.
5. Compute decimation factor `d = floor(f_raw / f_eff)`, round to a power of 2 for efficiency.
6. Apply a digital anti-aliasing (decimation) filter with cutoff at `f_eff / 2` and attenuation ≥40 dB at the new Nyquist frequency, then downsample by `d`.
   > Decimation procedure: `lecture_digital-filters.pdf` (4CM00), slides 20–29

**Important — zero-padding does not improve resolution (from course lecture):**
> Zero-padding only interpolates the existing spectrum at more frequency points. It does NOT increase frequency resolution. Resolution is determined solely by data length N and window choice.
>
> Source: `5CTA0_VideoLectures_VL12_intro_spectral_estimation.pdf`, slide 86

---

### Spectral quality check — coherence function

**From course lecture** (`lecture_FRF-measurements.pdf`, 4CM00, slides 26–40):

After decimation, compute the coherence function between each input–output pair:

> γ²(ω) = |S_yu(ω)|² / (S_uu(ω) · S_yy(ω))

Values near 1 confirm that the spectral estimate is reliable. Low coherence indicates noise, nonlinearity, or leakage. Coherence should be checked across the full frequency band of interest before accepting the decimated data as suitable for training.

---

### Note on leakage — implication for MATLAB trajectory design

**From course lecture** (`lecture_FRF-measurements.pdf`, 4CM00, slides 41–55):

Leakage occurs when the measurement window does not contain an integer number of periods of an excitation component. The DFT smears energy from one frequency bin into neighbouring bins.

**Implication for MATLAB:** if trajectories use periodic excitation (e.g. multisine), ensuring the trajectory length is an integer multiple of the excitation period eliminates leakage and produces cleaner spectral estimates. The preferred excitation for FRF quality is a **multisine** (sum of sinusoids at chosen frequencies with a period that fits exactly in the measurement window).

> Source: `lecture_FRF-measurements.pdf` (4CM00), slides 26–35

---

### Output

- `decimation_factor`: int
- `f_raw`: float (Hz) — original rate from `.mat` file
- `f_useful`: float (Hz) — identified useful bandwidth
- `f_effective`: float (Hz) — chosen effective rate after decimation
- Decimated trajectory tensors (passed to `precompute`, not reloaded from `.mat`)

### Standalone plot

- PSD of each channel for each trajectory (overlaid), log frequency axis
- Cumulative power curve with 99% threshold marked
- Vertical line at `f_useful` and `f_eff / 2`
- Annotated decimation factor and `f_effective`
- Coherence function per input–output pair

---

## Decision 2 — BPTT segment length

### Rationale

**From supervisor meeting notes (2026-04-24):**
> *"step response: give step response to the system, see when it has settled down"*
> *"time constant of the largest — largest time constant scales with the largest sample needed"*
> *"4000 is long — backward runs n²"*
> *"differences with nonlinearities — with MSD normally fine"*
> *"preferable the [shorter segment]"*

The concrete implication: simulate a unit step input on the system model at `f_eff`, measure how many samples it takes to settle — that number is the required segment length. Use the slowest mode (largest time constant) to set the minimum.

**From course lecture:**
A segment shorter than the settling time cannot contain enough dynamic information for the loss to discriminate between parameter values — gradients become uninformative.

---

### Settling time criterion

**From course lecture** (`lecture_FRF-measurements.pdf`, 4CM00, slides 17–30):

> **Settling time ≈ 5τ for <1% of step response amplitude**

Where τ is the dominant (largest) time constant of the system. For a system with multiple modes, use the slowest pole.

**In FRF measurement practice:** the first period(s) of any new excitation must be discarded to allow transients to settle before recording steady-state data. The number of periods to discard depends on τ relative to the excitation period T_exc.

> Source: `lecture_FRF-measurements.pdf` (4CM00), slides 17–30

**Implication for training data:** the start of each trajectory (first ~5τ samples) is potentially contaminated by initial condition transients. Consider trimming this from the valid segment start indices in `_attach_valid_start_idx`.

---

### Method

1. Using the **true physical model** (from `lfr_param_block.py` at true parameters), simulate a unit step input on each input channel independently at `f_eff`.
2. Measure the settling time per channel: first sample where the output remains within ±1% of its steady-state value without leaving again.
3. Required segment length = `ceil(max settling time × f_eff)` samples.
4. Round up to the nearest multiple of the BPTT window size `W` for alignment.

---

### Why the linear step response is sufficient

**From supervisor meeting notes (2026-04-24):**
> *"differences with nonlinearities — with MSD normally fine"*

For this system, the nonlinear coupling terms involve `sin(θ)` where θ is small in practice. The nonlinear and linear step response settling times are therefore close. Verify this by simulating both; if they agree within a small tolerance (e.g. <10% on settling time), use the linear result, which is cheaper and analytically interpretable.

---

### Frequency resolution cross-check

**From course lecture** (`5CTA0_VideoLectures_VL12_intro_spectral_estimation.pdf`, slides 61–93; `5CTA0_VideoLectures_VL13_non-parametric_spectral_estimation.pdf`, slides 40–75):

The step response result can be cross-checked using the frequency resolution constraint:

> To resolve dynamics at frequency f_min (Hz), the segment must satisfy:
> N_seg ≥ f_eff / f_min   (i.e. T_seg ≥ 1 / f_min seconds)

If the system's slowest mode is at f_min Hz, the segment must be at least 1/f_min seconds long. This cross-checks the step response result — they should agree.

The segment length should also satisfy:

> ACF lag length rule: for reliable spectral estimates from a segment of N samples, compute ACF up to lag L ≈ N/4.
>
> Source: `5CTA0_VideoLectures_VL13_non-parametric_spectral_estimation.pdf`, slide 19

This means a segment of length N gives reliable spectral information down to frequencies where at least N/4 autocorrelation lags are available.

---

### Output

- `segment_len`: int (samples at `f_eff`)
- `settling_time_s`: float (seconds)
- Per-channel and per-input settling times
- Linear vs nonlinear comparison result and agreement flag

### Standalone plot

- Step response curves (linear and nonlinear overlaid) per input channel
- Settling threshold bands (±1%)
- Annotated settling time in seconds and implied segment length in samples
- One panel per input channel

---

## Discriminability check

After the segment length is set by the step response, a discriminability check confirms the choice works empirically on the actual trajectories:

- Sample 24 random segments of length `segment_len` from the decimated trajectories.
- Evaluate 4 fixed parameter sets (true, all +10%, all −10%, coupling mix).
- Compute `true_best_rate`: fraction of segments where true parameters give the lowest MSE.
- Compute `median_margin`: median of (min detuned loss − true loss).
- If `true_best_rate < 0.8`, increase segment length and repeat.
- When multiple lengths give equal `true_best_rate` and `median_margin`, choose the shorter one (O(n²) BPTT cost, confirmed by supervisor: *"preferable the [shorter segment]"*).

This check is a safety net that confirms the analytically-derived segment length works on real data. It is not the primary decision method.

---

## Welch PSD estimation — practical notes

**From course lecture** (`5CTA0_VideoLectures_VL13_non-parametric_spectral_estimation.pdf`):

**Segment length for Welch (not to be confused with BPTT segment length):**
> For Welch averaging, collect K ≥ 8–16 non-overlapping segments for reliable variance reduction.
> Variance of averaged periodogram scales as approximately 1/K.
>
> Source: slides 40–75

**Overlap in Welch (WOSA):**
> 50% overlap is standard. Overlap improves the bias (resolution) of the estimate, but variance does NOT decrease as fast as 1/K when segments are overlapping (they are correlated).
>
> Source: slides 64–66

**Preferred window** (`lecture_digital-filters.pdf`, `5CTA0_VideoLectures_VL12_intro_spectral_estimation.pdf`): Hanning window (peak sidelobe −32 dB, good balance between resolution and sidelobe suppression).

> Source: `5CTA0_VideoLectures_VL12_intro_spectral_estimation.pdf`, slides 61–93

**Estimator preference:**
> Use the **biased ACF estimator**: r̂_b[τ] = (1/N) Σ x[n]x*[n−τ]
> It is asymptotically unbiased and guarantees a non-negative PSD. The unbiased estimator can yield negative PSD values, which are physically inadmissible.
>
> Source: `5CTA0_VideoLectures_VL13_non-parametric_spectral_estimation.pdf`, slides 20–35

---

## Caching

The cache stores:

```python
{
    'version':            int,
    'fingerprint':        tuple,          # (version, traj_ids, f_raw)
    # Sampling rate decision
    'f_raw':              float,          # original sampling rate from .mat
    'f_useful':           float,          # identified useful bandwidth (Hz)
    'f_effective':        float,          # chosen effective rate after decimation (Hz)
    'decimation_factor':  int,
    # Segment length decision
    'settling_time_s':    float,          # dominant settling time (seconds)
    'segment_len':        int,            # samples at f_effective
    'linear_nonlinear_agreement': bool,   # True if <10% difference on settling time
    # Discriminability check
    'true_best_rate':     float,
    'median_margin':      float,
    # Decimated trajectory data (passed directly to precompute)
    'trajs_decimated':    list[dict],
}
```

Cache is invalidated if any trajectory file changes, `f_raw` changes, or the trajectory set changes.

---

## Standalone mode

When run directly (`python -m lpv_lfr_baseline.scripts.trajectory_diagnostic`):

- Runs the full diagnostic (or loads cache and re-plots from cached results)
- Prints a summary table: `f_raw`, `f_useful`, `f_effective`, `decimation_factor`, `settling_time_s`, `segment_len`, `true_best_rate`, `median_margin`
- Produces and saves all figures to `simulations/param_recovery/`
- Does **not** modify any training config — output is informational; update `train_param_recovery.py` manually

---

## Integration with train_param_recovery.py

`train_param_recovery.py` calls this diagnostic at startup before precompute:

```python
from lpv_lfr_baseline.scripts.trajectory_diagnostic import run_trajectory_diagnostic

diag = run_trajectory_diagnostic(TRAJ_SPECS, TRAJ_DIR, SAVE_DIR)
# diag['trajs_decimated'] and diag['segment_len'] passed into precompute()
```

If a valid cache exists, this is a fast no-op. If not, the full diagnostic runs once and caches.
