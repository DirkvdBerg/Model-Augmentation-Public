# Parameter Recovery Pipeline Redesign

**Date**: 2026-04-30  
**Session**: Supervisor meeting follow-up — diagnostics and training data pipeline  
**Scope**: `experiment_diagnostics.py` (replaces `segment_diag.py`), trajectory subdivision, mini-batch structure

---

## 1. Context

Two things are being redesigned together because they share a dependency: the segment
length used to subdivide trajectories is determined by the diagnostics, and the
training mini-batch structure is defined by how those subdivisions are laid out.

---

## 2. `experiment_diagnostics.py` — replaces `segment_diag.py`

### 2.1 What changes

`segment_diag.py` is deleted. `experiment_diagnostics.py` takes its place and is
broadened to cover all experiment-level diagnostics, not just segment-length selection.

The old segment-length diagnostic (detuned parameter discrimination test) is removed.
It was fragile: it scored candidate window lengths by whether true parameters beat
detuned alternatives on short segments, which is not a reliable physical criterion.
The new approach derives the segment length from the system's dominant time constant
(see §2.2).

### 2.2 Diagnostics included

**FFT / frequency content**  
Compute and plot the FFT of each trajectory's output signals (X1, X2, Y). Show where
signal energy is concentrated relative to the current sampling frequency `fs`. This
answers whether `fs` (currently 20 kHz) is overkill — the supervisor expectation is
that 4 kHz may already be sufficient. Output: per-trajectory and combined spectrum
plots, printed bandwidth estimate.

**Step response / dominant time constant**  
Simulate a unit step input through the LFR model (using true/initial parameters).
Read off settling time and estimate the dominant time constant `τ_max`. This gives a
physically motivated minimum segment length: a segment must be long enough to contain
at least one full transient. Shorter segments that still capture the full dynamics are
preferred over longer ones (longer windows increase backward pass cost quadratically).

**Segment length recommendation**  
Derived from `τ_max` and `fs`: minimum samples = `ceil(k * τ_max * fs)` for some
coverage factor `k` (e.g. 3–5 time constants). Printed as a recommendation; stored
by `precompute.py` as `segment_len`. This replaces the `run_segment_diag()` call in
`precompute._compute()`.

1. Observability horizon, not just τ_max
  The segment must be long enough for the gradient to carry parameter information,
  not just long enough to see one transient. For a state-space/LFR system, the
  relevant quantity is the observability index — how many output steps are needed
  to reconstruct the initial state. For a system with n_state states, this is at
  most n_state steps, but in practice depends on the specific (C, A) pair of the
  LFR. For some parameters (like cy which is only visible in velocity-dominated
  low-frequency dynamics), the effective gradient horizon may be much longer than
  the step-response settling time.
  2. Truncated BPTT as the primary lever, not just segment shortening
  The standard solution to BPTT scaling in sequence models is truncated BPTT
  (TBPTT): backpropagate gradients only k steps instead of through the full
  segment. This reduces backward pass cost from O(T²) to O(T·k) and is the main
  technique in practice. The question is: for physical parameter recovery in LPV
  mechanical systems, what truncation length k is sufficient? This is separate from
   segment length — you can have a long segment but truncate the gradient horizon.
  3. Decimation as the first lever before segment shortening
  Reducing from 20 kHz to 4 kHz gives a 5× reduction in samples for the same time
  window. This directly reduces the O(T) or O(T²) cost without changing what
  dynamics are covered. The FFT analysis should output a concrete decimation factor
   recommendation, not just a "bandwidth estimate."
  4. Parameter-specific gradient timescales
  Different parameters have different effective gradient horizons: mh is visible on
   fast timescales (inertia dominates at high frequency), cy is visible on slow
  timescales (damping dominates at low frequency relative to inertia). The minimum
  segment length should be driven by the slowest parameter, not by τ_max of the
  dominant mode.
  5. The O(n²) claim needs clarifying
  Standard BPTT through a length-T segment is O(T) in compute per parameter and
  O(T) in memory (store activations). O(T²) arises if you are computing full
  covariance-like quantities, or if T appears in both the number of segments and
  segment length simultaneously. Worth being precise about where the scaling
  actually comes from before the research targets a fix.
So I need to take into consideration also the windowing that is being done with the current gradient maybe and adjust that, or maybe different problem??

Suggested additions to the diagnostic plan

  - Compute and print the LFR observability matrix rank and an estimate of the
  observability horizon (minimum steps to full rank)
  - Plot the parameter sensitivity dY/dθ over time for each parameter to see which
  has the longest gradient tail — this gives an empirical minimum window
  - Add a decimation recommendation as the primary output of the FFT analysis, not
  just a plot

### 2.3 Interface

```
recommend_segment_len(trajs, fs) -> int
    Called by precompute._compute() to determine segment_len.
    Uses step response simulation internally.

run_diagnostics(pre, save_dir) -> None
    Called with the precompute output dict. Runs all diagnostics and saves plots.

__main__
    Calls precompute() to get trajectory data from cache, then run_diagnostics().
    Produces all plots and printed summaries for standalone inspection.
```

### 2.4 Integration with precompute

`precompute.py` calls `recommend_segment_len()` from `experiment_diagnostics.py`
instead of `run_segment_diag()`. The result is stored as `segment_len` in the
precompute cache, exactly as before. No change to the precompute cache format.

`experiment_diagnostics.py` imports `_DATASETS` from `train_param_recovery.py`
for standalone use (the module-level assignments there are benign at import time).

### 2.5 Helpers migrated out of `segment_diag.py`

`segment_diag.py` exported three helpers that `train_param_recovery.py` uses:

| Helper | Destination |
|--------|-------------|
| `_attach_valid_start_idx` | `train_param_recovery.py` (training-only utility) |
| `_sample_balanced_segments` | `train_param_recovery.py` (training-only utility) — see §3 for redesign |
| `_traj_set_tag` | `precompute.py` (used for cache key naming) |

---

## 3. Training data pipeline — trajectory subdivision and mini-batch structure

### 3.1 What is wrong with the current approach

The current training loop samples 8 random segments per epoch from any trajectory.
This has two problems:

1. **No balance guarantee**: random sampling can draw multiple segments from the same
   trajectory and none from another, especially with unequal trajectory lengths.
2. **Redundant work at runtime**: `_sample_balanced_segments` re-samples start indices
   every epoch. The valid start index range is fixed — there is no reason to redo this
   inside the training loop.

The supervisor recommendation: **pre-divide trajectories into fixed segments before
training begins**, based on the segment length from diagnostics. Do this once in
precompute or at training setup, not per epoch.

### 3.2 Proposed structure

**Pre-division (at setup time, not per epoch)**  
Each trajectory is divided into non-overlapping (or optionally overlapping) segments
of length `segment_len`. This produces a fixed pool of segments per trajectory.
Overlapping is allowed to increase pool size, but the stride and overlap factor are
fixed parameters, not random.

**Mini-batch = one segment per trajectory**  
Batch size equals the number of active trajectories (currently 6 for base dataset,
8 with multisine). Each trajectory contributes exactly one segment per batch. This
guarantees every trajectory is represented in every gradient update and removes the
need for post-hoc balancing logic.

**Epoch = one pass through the segment pool**  
Within an epoch, iterate through the pre-divided segment pool. Each step draws one
segment per trajectory (randomly or in order). When the shortest pool is exhausted,
the epoch ends. This is analogous to standard mini-batch SGD over a fixed dataset.

### 3.3 Open questions (to be resolved before implementation)

**Q1: Consistent segment length vs. minimum coherent length**  
Should all trajectories be divided into segments of the same length (simplest, enables
pre-allocated batch tensors), or should each trajectory use its own minimum coherent
length based on its local dynamics? The supervisor noted trajectories differ in
excitation, so a fixed global `segment_len` may be reasonable.

**Q2: Full trajectory vs. mini-batch per trajectory**  
Two options per gradient step:
- **Full trajectory**: roll out the complete trajectory, accumulate loss over all time
  steps. Maximum gradient signal, but cost scales with trajectory length.
- **Mini-batch**: one fixed segment per trajectory per step (as described in §3.2).
  Lower variance requires more steps but each step is cheap.

The BPTT window `W` is orthogonal to this choice: it controls how many steps are
unrolled before detaching state, not how many segments are used per step.

**Q3: Overlap factor**  
If segments overlap, what stride should be used? A stride of `segment_len // 2`
(50% overlap) doubles the pool size. More overlap increases data reuse but also
correlation between consecutive mini-batches.

### 3.4 What this does not change

- The BPTT window `W` (inner loop, detaches state between windows) is unchanged.
- The `simulate()` call signature and the loss computation are unchanged.
- `precompute.py` continues to own trajectory loading and `segment_len`.

---

## 4. Summary of file changes

| File | Change |
|------|--------|
| `segment_diag.py` | **Deleted** |
| `experiment_diagnostics.py` | **New** — FFT, step response, segment length recommendation |
| `precompute.py` | Replace `run_segment_diag` import/call with `recommend_segment_len` from experiment_diagnostics; absorb `_traj_set_tag` |
| `train_param_recovery.py` | Absorb `_attach_valid_start_idx`, `_sample_balanced_segments`; redesign mini-batch loop (§3.2, pending Q1–Q3) |

---

## 5. Open items before implementation

- [ ] Resolve Q1–Q3 in §3.3 (segment length consistency, full vs. mini-batch, overlap)
- [ ] Confirm step response approach for time constant estimation (simulate with true params, or estimate from data?)
- [ ] Confirm FFT plot format (per trajectory separate, or overlaid?)
- [ ] Decide whether `experiment_diagnostics.__main__` saves plots to `save_dir` or shows them interactively
