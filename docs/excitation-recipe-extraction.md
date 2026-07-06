# Excitation Recipe Extraction (copy-target pass)

**Date:** 2026-07-06.
**Purpose:** full extraction of the data-generation recipes from the closest in-framework
sources, so the gantry design can be written as copy-plus-delta instead of assembled from
principles. Companion to `docs/excitation-design-literature.md` (Sections 1-9) and
`docs/data-generation-brief.md`.

**What was actually read for this document (honesty ledger):**
- `literature/augmentation/hoekstra2025_lfr-augmentation-ejc.pdf` (EJC 86 (2025) 101304):
  full text extracted via pdftotext; Sections 1-2, 3.5-3.6, 4 (MSD study), 5 (Bouc-Wen),
  6 read. VERIFIED.
- `literature/augmentation/Encoder initialisation methods in the model augmentation
  setting.pdf` (arXiv 2602.13108): experiment sections (4.1-4.4) read. VERIFIED.
- `scripts/journal_model_augmentation/msd_ndof_data_generation.py` from
  github.com/JanHHoekstra/Model-Augmentation-Public: read in full. VERIFIED.
- arXiv 2602.17297 (journal version, submitted to Automatica): Sections 6.1-6.3 (MSD
  data generation) and 7.1-7.4 (F1Tenth) read from fetched full text. Abstract and
  experiment sections read; remainder of paper (theory sections) not read this pass.
  VERIFIED for the sections cited below.
- Bolderman, Lazar, Butler, CCTA 2021 (arXiv 2103.06092): full text read 2026-07-06
  (see literature log Section 9.1). VERIFIED.
- Bouc-Wen official benchmark: nonlinearbenchmark.org page read (pointer only); the
  detailed signal-generation guide sits inside the 4TU.ResearchData zip
  (doi 10.4121/12967592, MATLAB-based) and was NOT retrieved. Spec numbers below rest
  on the benchmark description as previously search-verified plus the EJC paper's usage.
  PLAUSIBLE on fine detail (exact fs, line count), VERIFIED on signal classes and bands.

---

## 1. Donor recipes

### Donor A — Hoekstra MSD simulation study (EJC 2025 §4; arXiv 2602.17297 §6; public script)

The in-framework template. System: 3-DOF MSD with cubic hardening spring; baseline =
linear 2-DOF model (third mass + nonlinearity missing) → exactly our topology class
(baseline missing one resonant subsystem = 2 states).

| Element | Value | Source |
|---|---|---|
| Loop | Open loop, force directly on m1, position p2 measured | paper |
| Integration | RK4, Ts = 0.02 s (fs = 50 Hz), ZOH input | paper + script |
| Signal | Random-phase multisine, phases uniform [0, 2pi) | paper + script |
| Band / grid | Paper: 1666 components in [0, 25] Hz. Script: period N = 10000 samples (200 s, 0.005 Hz grid), pmax = 4999 lines = full grid up to Nyquist (25 Hz). DISCREPANCY in line count, agreement on full-grid-to-Nyquist. | paper vs script |
| Crest factor | None: `n_crest_factor_optim = 1` (single random draw, no optimization) | script |
| Amplitude | Single scalar, `amp_scale = 10` (unit-RMS multisine x 10). No stated rationale, single level. | script |
| Periods / transient | Simulate P+1 periods, discard first full period (transient), keep steady state. Train = 2 (identical) periods = 2e4 samples; val = 1 period; test = 1 period. | script |
| Realizations | Independent realization per split (separate random draw for train/val/test) | paper + script |
| Noise | Additive white output noise, SNR 20/30/60 dB (EJC), 30 dB (journal). Noise stage optional/commented in script. | paper + script |
| States | True states saved (`save_state=True`) for encoder pre-training | script |
| SUBNET hyperparameters | na = nb = 7, T = 200 (4 s), aug nets 2x8 tanh, encoder 2x16, 3000 epochs, batch 2000 | paper Tables 5/6 |
| Variants | Same recipe reused for input saturation (30·tanh(u/30), RMS 10 → 9.11 N) and output LPF configurations | arXiv §6.1 |

### Donor B — Hoekstra encoder-init study (arXiv 2602.13108 §4)

2-DOF MSD, cubic spring + cubic damper; baseline = same structure with wrong damping
parameter. RK4 with Ti = 0.01 s inner step, sampled at Ts = 0.1 s (integration 10x finer
than sampling). Multisine 1666 components, full grid [0, 5] Hz = up to Nyquist at fs = 10
Hz. SNR 20 dB. Splits 2e4/1e4/1e4. na = nb = 9, T = 200, nets 2x16, 2000 epochs, batch
3000. Confirms the Donor A pattern: full grid to Nyquist, single amplitude, independent
realizations, same split sizes.

### Donor C — F1Tenth real closed-loop study (arXiv 2602.17297 §7.3) — the closed-loop protocol

The only in-framework case with real, closed-loop, trajectory-tracking data. No injected
broadband signal at all; coverage comes entirely from reference diversity:

| Element | Value |
|---|---|
| Excitation | Reference trajectories only, tracked in closed loop. NO multisine, NO dither. |
| Reference classes | 2: lemniscate ("heading angle traverses the whole operational domain" + quick velocity changes) and circle ("typical maneuver"). |
| Rate/amplitude ladder | Velocity reference varied 0.45 to 1.0 m/s in 0.05 steps (12 levels) per class → 24 records total. |
| Split design | Record-level: half of the 24 records → train(+val), half → test. Both classes and alternating velocities present in every split. |
| Validation set | Contiguous segments, 20% of the length of each of the 12 training records, randomly located, cut out BEFORE concatenation. |
| Sizes | fs = 40 Hz; N_est = 6467, N_val = 1669, N_test = 8041 (test larger than train). |
| Measurement | Full state (OptiTrack + IMU); 9 baseline parameters jointly estimated. |
| Hyperparameters | na = nb = 12, T = 40 (1 s), nets 2x128, 3000 epochs, batch 256. |

Key contrast with our case: their missing dynamics (tire forces) are excited by the
maneuvers themselves. Our missing dynamic (157 Hz payload MSD) is far above the reference
bandwidth, so reference diversity alone cannot activate it. That single fact is what
justifies adding Donor A's multisine on top of Donor C's protocol.

### Donor D — Bolderman CCTA 2021 (closed-loop injection point)

Industrial coreless linear motor, closed loop (PID always active). Two simultaneous
excitation mechanisms: (1) zero-mean white-noise force dither, variance (80 N)^2, held at
100 Hz update, injected directly at the plant input in parallel with the feedback
controller — stated rationale: "Dithering the CLM input directly prevents the signal from
being filtered by the feedback controller..."; (2) third-order (jerk-limited)
point-to-point reference (0 to 0.05 m, v 0.05 m/s, a 4 m/s^2, jerk 1000 m/s^3, 50%
constant-velocity time). 4 back-and-forth motions, Ts = 1e-4 s, split 70/15/15 by
fraction. (Full extraction in literature log Section 9.1.)

### Donor E — Bouc-Wen benchmark (test-suite template)

Train: random-phase multisine, full grid 5-150 Hz, 8192 samples per period, 50 N RMS.
Tests: (i) sinesweep, 40 N, 20-50 Hz at 10 Hz/min (through the system's own resonant
band, at reduced amplitude); (ii) independent multisine realization, same band, also
reduced amplitude. Hoekstra EJC usage on this data: na = nb = 13, T = 500.
RESIDUAL: exact fs and line spacing not re-verified from the official 4TU signal
generation guide (MATLAB zip, not retrieved); do not quote those beyond the above.

---

## 2. Cross-donor patterns (what "copy" means)

1. **Simulation studies:** full-grid random-phase multisine up to a Nyquist-like cap; no
   crest-factor optimization; single unexplained scalar amplitude; no spectral shaping;
   one extra period simulated and discarded for transient; independent realization per
   split; split sizes 2e4/1e4/1e4 at ~50x the dominant dynamics' timescale.
2. **Repetition:** Donor A's training set is 2 identical periods WITH output noise
   (repetition = averaging information). In our noiseless phase repetition adds zero
   (lessons rule), so the copy translates to: one period per realization, more independent
   realizations instead of repeats. This delta is forced by the noise setting, not taste.
3. **Real closed-loop hardware (in-framework):** no injected signal; coverage =
   reference-class diversity x rate ladder spanning the operational/scheduling domain;
   record-level train/test split; validation as contiguous random segments cut from
   training records.
4. **Closed-loop injection, when injecting (nearest-field):** force dither at plant input
   in parallel with feedback, plus jerk-limited references for range coverage
   (Donor D). Matches our generator's existing injection point.
5. **Test suite:** cross-class sweep through the system's own resonant band + independent
   same-class realization, both at reduced amplitude relative to training (Donor E).

## 3. Delta list for the gantry (everything not covered by a donor)

| # | Delta | Donor gap | Resolution path |
|---|---|---|---|
| 1 | 157 Hz mode unreachable by references | Donor C's protocol alone cannot activate it | Add Donor A multisine component through Donor D injection point (already the generator's structure) |
| 2 | Band choice under closed loop + force budget | Donors are open-loop; full-grid-to-Nyquist at 20 kHz is absurd here | Empirical: delta_a activation + delivered-spectrum (input sensitivity) diagnostics on existing data; band cap is OURS to justify |
| 3 | LPV scheduling (Y, Ydot) coverage | No donor is LPV | Synthesis: Donor C's ladder logic (levels spanning the domain) + Ghosh frozen-Y local experiments; >= 3 frozen-Y points (quadratic M(Y) dependence) + Y-sweep records |
| 4 | Yaw constraint (|X1-X2| <= 6 mm) budgeting | No donor has a constrained mode | Logical-coordinate design, GVT force-appropriation analogy; # HEURISTIC |
| 5 | Amplitude sizing | Donors: single unexplained scalar | Force limits (hard) + delta_a activation diagnostic (empirical); document rationale donors omit |
| 6 | Realization diversity | Donor A reuses one draw per split (2 identical periods) but has noise | Noiseless: fresh independent realization per EXPERIMENT, one period each — fixes the current generator's shared-draw defect |
| 7 | Validation-set construction | Current generator: whole experiments per split | Copy Donor C: contiguous random segments (20%) cut from training records |
| 8 | Cross-class test | Current data has none | Copy Donor E's functions: sweep through 130-180 Hz at sweep rate safe for Q~10 (Gloth & Sinapius criterion still to be pulled) + independent multisine, both at reduced amplitude |

## 4. Open residuals from this pass

- Donor A line-count discrepancy (paper 1666 vs script 4999 of 5000 lines): could be
  settled by inspecting the released .npz spectra; not done (no code runs this pass).
- Bouc-Wen official fs / grid: inside the 4TU MATLAB zip; quote only band/class/amplitude
  facts until retrieved.
- Gloth & Sinapius (2004) quantitative sweep-rate criterion: still unread (paywalled);
  needed before fixing the sweep-rate number in Delta 8.
