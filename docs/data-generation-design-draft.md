# Gantry Data-Generation Design — DRAFT (copy-plus-delta)

**Date:** 2026-07-06. **Status: DRAFT for discussion, nothing implemented.**
Grounded in `docs/excitation-recipe-extraction.md` (donors A-E). Every element carries a
provenance flag:
- **COPY(x)** — taken from donor x, function-first (never literal numbers where systems differ)
- **THEORY** — model-structure or classical-theory derived, source named
- **HEURISTIC** — ours, no citation; per CLAUDE.md labeling rule
- **GATE** — open parameter, to be settled by a named empirical diagnostic, not by opinion

Decisions marked **OPEN** need user/supervisor input before this draft goes to
`docs/decisions.md` and implementation.

---

## 1. Architecture: two excitation layers

**COPY(C) + COPY(A), delta 1.** Following the F1Tenth protocol, coverage of the operating
domain comes from reference-trajectory diversity, not from the injected signal. Following
the MSD simulation recipe, the hidden 157 Hz dynamic is activated by a random-phase force
multisine, injected at the plant input in parallel with feedback (**COPY(D)**, matches the
existing generator's injection point; Bolderman's stated rationale now citable).
The one-sentence justification for needing both layers where F1Tenth needed one: their
missing dynamics live inside the maneuver bandwidth, ours is two decades above it.

## 2. Experiment grid (reference layer)

| Axis | Draft value | Flag |
|---|---|---|
| Reference classes | 2: (i) Y-sweeping oscillatory/lissajous records, chosen so Y traverses the full [-0.30, +0.30] m range within a record (the "lemniscate" function: scheduling variable traverses the whole operational domain); (ii) randomized jerk-limited setpoint sequences (position-level APRBS): new random setpoint list per record, no two records alike | COPY(C) for class logic; APRBS adaptation HEURISTIC (Nelles PARTIAL, see brief §4) |
| Frozen-Y operating points | 5 points: Y in {-0.30, -0.15, 0, +0.15, +0.30}; minimum 3 required by the quadratic Y-dependence of M(Y) | THEORY (Ghosh 2018 local approach + model structure) for >= 3; 5 instead of 3 is HEURISTIC margin |
| Y-rate coverage | Y-sweep records at >= 2 distinct sweep rates (slow ~0.2 Hz, fast ~0.7 Hz equivalent) | THEORY-track (LPV lit: local/frozen experiments cannot capture dynamic scheduling dependence -> varying-Y records required; exact section to be quoted from Toth 2010 book, locally available — see lit log §10); rate VALUES still HEURISTIC |
| Amplitude/velocity ladder | >= 3 levels per class, spanning small to near-limit motion (F1Tenth used 12 velocity levels; our per-record duration is longer so fewer records per level) | COPY(C) ladder function; level count OPEN |
| Records | ~24 records x 10 s (order matches F1Tenth's 24; sizes Section 6) | COPY(C) scale; count OPEN |
| Fixes "all oscillations look the same" | Randomized setpoint class + ladder + per-record multisine realization: no two records share (trajectory, amplitude, phase draw) | consequence of the above |

## 3. Multisine layer

| Element | Draft value | Flag |
|---|---|---|
| Phases | Uniform random [0, 2pi), fresh draw per record; NO crest-factor optimization | COPY(A) (script: `n_crest_factor_optim = 1`) |
| Period | = record length (10 s -> 0.1 Hz grid; ~160 lines across the 16 Hz 3-dB bandwidth of the mode); no tiling | COPY(A) function (period-long realization); no-tiling delta forced by noiseless setting (lessons rule: repetition adds zero) |
| Realizations | Independent per record (not just per split) | COPY(A) delta 6, noiseless justification |
| Band | Default: broadband [1, 200] Hz (donor function "full grid over the system's dynamic range", translated: structural band + mid + resonance + margin, NOT to 10 kHz Nyquist). Narrowband [130, 180] retained only as fallback if GATE fails | COPY(A) function; cap at 200 Hz HEURISTIC; final choice GATE 1 |
| Amplitude | Per-channel RMS from (i) hard TELICA limits (peak/RMS force, 6 mm yaw, velocity) as constraints, (ii) delta_a activation as target | limits THEORY (spec); level GATE 2 |
| Channel coordinates | Design in logical coordinates [F_sym, F_anti, F_Y]; F_anti sized from the 6 mm yaw budget through the closed-loop yaw compliance; transform to stage forces before simulation | HEURISTIC (no excitation-design precedent; analogy: GVT normal-mode force appropriation, Wright & Cooper 1999, different paradigm, flagged) |
| Injection | Additive force at plant input, parallel to feedback (unchanged) | COPY(D) |

## 4. Splits and validation

| Element | Draft value | Flag |
|---|---|---|
| Split unit | Record-level; both classes, multiple Y-points and ladder levels present in every split | COPY(C) |
| Validation set | Contiguous random segments, 20% of each training record, cut before concatenation; validation metric = training loss on these segments (user's earlier spec: same measure as training) | COPY(C); metric per user spec |
| Test records | Fully held-out records (never same trajectory parameters as train) | COPY(C) |
| Realization independence | Every record has own multisine draw, so split independence is automatic | COPY(A)+delta 6 |

## 5. Test suite (beyond held-out records)

| Test | Draft value | Flag |
|---|---|---|
| Cross-class sweep | Force sinesweep through [130, 180] Hz at reduced amplitude (~80% of training RMS), sweep rate slow enough for the Q~10 mode; placeholder <= 10 Hz/s. Definitive bound from ISO 7626-2 max-sweep-rate formula (S_max ~ (3 dB bandwidth)^2 rule; BW ~16 Hz here), Gloth & Sinapius 2004 as the analysis paper | COPY(E) function; rate bound THEORY-track via ISO 7626-2 (constant to be pulled from standard/paper; lit log §10) |
| Independent multisine | Fresh realization, same band, reduced amplitude | COPY(E) |
| Cross-class trajectory | p2p-class test record for a multisine-trained... i.e., held-out class combinations (e.g., a Y-sweep rate not in training) | COPY(C) spirit; OURS |

## 6. Sizes and rates

| Element | Draft value | Flag |
|---|---|---|
| Training sample rate | OPEN. Requirements: (i) discretization validation passes, (ii) encoder-init quality near native-rate reference (lessons rule: both, separately). 157 Hz mode needs comfortable margin above 314 Hz Nyquist; earlier project results: encoder init good at 400 Hz, bad at 200 Hz. Candidate: 1 kHz | GATE 3 (two named diagnostics) |
| Samples | ~24 records x 10 s: at 1 kHz -> 2.4e5 total, ~1.4e5 train. Donor scale is 2e4 train at 50x dominant timescale; ours is proportionally consistent | COPY(A/C) scale |
| Optional learning curve | Train on 25/50/100% of records, check saturation | OURS, cheap in sim |

## 7. Empirical gates (run before finalizing, all on existing or one new dataset)

1. **GATE 1 (band):** with/without-multisine delta_a RMS and residual PSD at the resonance
   for broadband vs narrowband at matched total RMS. Generator already computes this.
2. **GATE 2 (amplitude):** delta_a activation vs amplitude at fixed band; pick lowest level
   with robust activation, under hard limits. Also confirms yaw budget consumption.
3. **GATE 3 (rate):** downsampling validation + encoder-init sweep vs native rate.
4. **Delivered-spectrum check:** PSD of u_total vs designed f per band; quantifies
   closed-loop reshaping (input sensitivity) and decides whether mid-band lines need
   pre-emphasis. THEORY (classical input sensitivity), consequence design OURS.
5. **Spike root-cause** (brief §2) before trusting any p2p records: 1 Hz-periodicity test
   discriminates tiled-multisine artifact from move-edge transient.

## 8. What this draft deliberately does NOT decide

- Exact record counts per grid cell (Section 2) — OPEN, discuss.
- Band cap value (200 Hz) beyond "resonance + margin" — HEURISTIC, challengeable.
- Whether mid-band (10-130 Hz) needs dedicated probe lines — deferred to gate 4 evidence.
- Sweep-rate number — blocked on Gloth & Sinapius full text.
- fa = 150 vs observed ~157 Hz bookkeeping (which number the diagnostics should window
  on) — pending user confirmation of where 157 comes from.
