# Data Generation for ANN Augmentation — Overview & Advice Questions

**Dirk van den Berg, 2026-07-06. One page for discussion; full design in
`docs/data-generation-design-draft.md`, sources in `docs/excitation-recipe-extraction.md`.**

## Setting

Grey-box augmentation of the gantry LPV model with an ANN (SUBNET encoder, truncated
prediction loss). Hidden dynamic to learn: payload MSD, resonance ~157 Hz, zeta ~0.05.
Data: noiseless Simulink simulation at 4 kHz, excitation through the closed-loop tracking
controller. Design grounded in the published augmentation examples (Hoekstra EJC 2025 +
released code; F1Tenth real-data case; Bouc-Wen benchmark; Bolderman CCTA 2021 for
closed-loop injection) instead of invented from scratch.

## Proposed design (summary)

Two excitation layers per record: **reference trajectories** for operating-range and
scheduling coverage (following the F1Tenth protocol: trajectory classes x amplitude/rate
ladder) and a **random-phase force multisine** to activate the resonance (following the
MSD simulation recipe), injected as a force at the plant input in parallel with feedback
(Bolderman's choice, with stated rationale).

| Set | Records (10 s each) | Content | Purpose |
|---|---|---|---|
| Train | 14 | 5 standstill multisine at Y = {-0.30,-0.15,0,+0.15,+0.30}; 3 Y-sweeps (2 rates); 4 randomized jerk-limited setpoint sequences; 2 multi-axis lissajous; amplitude ladder across records; fresh multisine realization per record | Frozen-Y local information; Y-rate (dynamic scheduling) terms; off-orbit coverage; MIMO coupling |
| Val | 3 | Same classes, fresh realizations + new seeds, at **unseen interior Y points** (e.g. +0.10, -0.22) | Checkpoint selection, incl. Y-interpolation (Jan's protocol: separate generation runs, never data slices) |
| Test | 6 | Unseen Y points (distinct from val); unseen Y-rate; force sinesweep through the resonance band (ISO 7626-2 rate limit); independent multisine; multisine-OFF trajectory record; amplitude above training ladder | Untouched generalization measures: scheduling, rate, signal class, amplitude extrapolation, no-hallucination regression |

Train keeps the Y-range edges so val/test measure interpolation, not extrapolation.

## Question 1 — Multisine band: broadband [1, 200] Hz or narrowband [130, 180] Hz?

Current generator default: narrowband around the resonance. Published precedent is
uniformly the opposite: full-grid multisine over the system's entire dynamic range
(Hoekstra full grid to Nyquist; Bouc-Wen 5-150 Hz full grid). No literature found (4
search rounds) supporting narrowband concentration around a known target mode. Narrowband
also presumes the FP model is exact outside the band.
Trade-off: at fixed force budget, broadband has ~2x lower per-line amplitude near 157 Hz.
**Our lean: broadband default, decided by the existing delta_a activation diagnostic
(with/without multisine, both bands, matched budget); narrowband as fallback.**
Note: in closed loop the delivered force spectrum is shaped by the input sensitivity
(content below the ~100 Hz bandwidth is partly cancelled); low-band coverage therefore
comes from the references, and the multisine's effective job is the band above ~100 Hz.

## Question 2 — Multisine channels in logical coordinates?

Currently: three independent per-stage-channel draws (F_X1, F_X2, F_Y). Consequence: the
anti-symmetric (yaw) component (F_X1 - F_X2)/2 is an uncontrolled byproduct at ~0.71x the
per-channel RMS, aimed at the one mode with the hard 6 mm |X1 - X2| limit, and the
sym/anti energy split is random per realization.
Proposal: design three independent multisines in logical coordinates [F_sym, F_anti,
F_Y], each with its own amplitude, F_anti sized deliberately from the 6 mm yaw budget
(through the closed-loop yaw compliance), then transform to stage forces before
simulation. The transform is invertible, so MIMO informativity is unaffected; the model
also operates on logical coordinates internally.
**No excitation-design precedent exists either way (nearest analogy: normal-mode force
appropriation in GVT). This is an engineering choice we would like your view on.**

## Question 3 — Record allocation and amplitudes

Is the 5/3/4/2 class allocation sensible, or would you weight motion classes higher?
Multisine amplitude policy: hard TELICA limits as constraints, level chosen as the lowest
with robust delta_a activation (diagnostic-driven). Reasonable?

## Open flags (for transparency)

The yaw-budget coordinate design and the 200 Hz band cap are our own (no citation);
Y-rate coverage and the sweep-rate limit have theory anchors being finalized (Toth 2010
LPV book; ISO 7626-2). Everything else is copied function-first from the named sources.
