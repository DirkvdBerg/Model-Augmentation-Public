# X+Θ+Y Augmentation Drift — Diagnosis Status, Findings, Hypothesis, Solutions

**Date**: 2026-07-09. **Scope**: why the X+Θ+Y (full-routing) augmentation cannot learn
without the free-run drifting. Companion diagnostics live in
`scripts/gantry/diagnostics-drift/`; figures/data in `simulations/gantry_subnet/diagnostics/`.

> **STANDING CONSTRAINT (supervisor + user directive).** The velocity/acceleration-domain loss
> ("fix C" — fitting velocity/acceleration instead of position; DIDIM/force-domain; Ljung
> prefilter AS THE TRAINING LOSS) is an explicit **LAST RESORT**. It must NOT be presented,
> ranked, or planned as the primary/first solution, even though the literature (Ljung, DIDIM,
> Tustin-Net) keeps converging on the velocity/force domain. Try the **position-based** fixes
> first (DC guardrail / projection / dead-zone; incremental-passivity/NI relaxation; structural
> integrator factoring that keeps a position loss). Elevate fix C only on explicit user go-ahead
> or after the position-based fixes demonstrably fail. NOTE: Tustin-Net structural integrator
> factoring can keep a POSITION-domain loss, so it is NOT the velocity-domain loss and is not
> gated by this constraint — do not conflate the two.

> **STANDING CONSTRAINT (closed-loop evaluation).** IF we ever evaluate or deploy the augmentation
> CLOSED-LOOP (with the Telica servo, or any controller, closing the loop), no-drift is NOT sufficient
> evidence that the model is correct. In closed-loop the servo bounds position for ANY model, so it MASKS a
> spurious model DC / bad fit: a wrong augmentation and a correct one both stay at setpoint, differing only
> in control effort and residuals. Bounded position in closed-loop certifies the LOOP, not the MODEL.
> Therefore closed-loop no-drift MUST be paired with a FIDELITY metric that feedback CANNOT hide:
> (a) prediction-error / one-step or short-horizon residual analysis on the measured signals, and/or
> (b) a control-effort / bias audit (a spurious model DC shows up as a steady-state control-effort offset).
> Rationale: closed-loop provides requirement 4 (bounded position) via the servo, not via the model
> (see `augmentation-literature-verdict.md`, "the one setting where all four hold", and dissipativity-limits
> A5). Never report closed-loop no-drift as model validation on its own.

---

## 0. Companion documents (index) — read these alongside this doc
This is the main working doc for the X/Y augmentation-drift problem. The analysis is split across companion
docs created in the 2026-07-10/11 literature thread; this is the index.
- **`docs/all-five-construction-spec.md`** — HOW TO HIT ALL FIVE: the buildable assembly (Route B empirical-R4
  / Route A structural-R4), requirement→mechanism→validation, build sequence. **The build target now that the
  search concluded no off-the-shelf method exists.** First action = D-107 clean re-run (Layer 1).
- **`docs/literature-search-conclusion.md`** — SEARCH CONCLUSION (D-108): no published method meets the 5
  requirements; gap CONFIRMED by a 2025 authoritative survey (Sivaranjani et al. arXiv:2512.06315,
  "remains an open challenge" / time-varying-LPV-ID = future work). Thesis-positive negative result; the
  contribution sits in an explicitly-open area. **Read for the search's bottom line.**
- **`docs/open-loop-solution-decision.md`** — DIRECTION DECISION: stay open-loop (closed-loop hides a bad
  model); the drift must be SOLVED not hidden; the estimation route is the sole open-loop path; first step =
  clean position-domain re-run at correct lr. **Start here for the current direction.**
- **`docs/augmentation-literature-verdict.md`** — exhaustive requirement table (every method vs the 4
  requirements) + verdict: no single method has all four; Tustin-Net is the best real partial; closed-loop is
  the one setting all four hold (but it hides). Primary-read grounded.
- **`docs/dissipativity-limits.md`** — consolidated list of EVERY dissipativity/passivity/NI restriction
  (A: boundedness limits, B: expressivity limits), with verified quotes.
- **`docs/data-silent-regularization-concept.md`** (+ `-limits.md`) — the estimation-route method
  (Gyorok orthogonal projection re-aimed at the unexcited subspace) and its honest limits.
- **`docs/augmentation-validation-design.md`** — how to validate: injected-dynamics library (each falsifies a
  failure mode), identifiability precondition, subspace-correctness metric, excitation ablation.
- **`docs/rollout-stability-literature.md`** — the drift under OTHER names (exposure bias / distribution
  shift / rollout stability): pushforward trick + noise injection (primary-read, proven, expressivity-
  preserving, open-loop, solve-not-hide, UNTRIED here), transient-amplification, discrepancy modeling,
  precision-mechatronics (closed-loop domain). Includes "do these extend to real nonlinear data" (yes, with
  caveats, keeps friction). **The most on-target new direction; feeds the D-107 first step.**
- **`docs/gns-encoder-diagnostic-plan.md`** — PLAN (unbuilt): a 3-arm sim diagnostic (control / GNS-A velocity
  noise / GNS-B X-position exposure) testing H1 no-drift, H2 absorber, **H3 marginal-pole eigen-check
  (load-bearing)**, H4 trainability, adapted to our encoder re-init; Y-position never perturbed (scheduling).
  Awaiting go-ahead to implement.
- **`docs/ml-for-control-search-sweep.md`** — ongoing 6-direction literature sweep (ML-for-control), one
  direction per prompt, quotes transcribed from primary PDFs. **Direction 9 (LPV cost function): our nf-window
  loss IS the framework-native truncated multi-step LPV-SUBNET cost (Verhoek 2204.04060 Eq 7a), NOT naive
  single-step MSE. Literature ADDS (all supervisor-group-grounded): state-consistency regularization (Layer
  1), exogenous/measured-y scheduling (Layer 3/R5), the predictor-vs-simulation drift lever. No scheduling-
  aware DRIFT cost term exists = novel R5 piece. Velocity-form LPV = LAST-RESORT-gated, flagged not adopted.**
  **Direction 8 (full-req + survey): gap CONFIRMED authoritative.** **Direction 7 (R5-driven): R5 = the recognized
  "corrupted-scheduling / errors-in-variables LPV" problem (TU/e Tóth-group) — BUT their problem is stochastic
  NOISE at identification; ours is deterministic DRIFT at inference self-scheduling → gives framing + real-data
  tools, NOT a fix for the sim drift; flagship paywalled.** Direction 4 DONE: bias-corrected/IV ID for
  closed-loop→open-loop LPV (Piga-Bemporad, exact-setting match; for the REAL-DATA baseline fit, not the
  ANN). Direction 5 DONE: LPV+ML — our framework has CONSISTENCY not no-drift (Verhoek/Toth 2204.04060); the
  state-consistency regularizer (Sertbas-Kumbasar 2510.24757 Eq 13) is a keepable in-LPV drift-reducer, but
  REJECT its bundled Schur stabilization (damps the marginal pole). Direction 6 DONE: diagnosis-side — one
  transferable check (Jacobian non-normality/commutator, 2605.08856) to confirm/refine the d6 DC-only
  finding; semigroup consistency weak, PDE-aliasing mechanism does NOT transfer. Direction 1 DONE:
  rollout-stability frontier — unrolled-training study (List-Thuerey 2402.12971) is directly about OUR
  "correction setup" (baseline+ANN), disentangles distribution-shift vs gradients, and shows TRUE unrolling
  is more faithful than noise injection (a proxy) → reorder D-107 to unrolling-first. Direction 2 DONE:
  hybrid/UDE identifiability (contribution intersection) — adopt PARAMETRIC vs FUNCTIONAL identifiability
  vocabulary (Loman-Baker 2510.14140); interpretability preserved via param regularization even when
  non-identifiable (Hotvedt 2010.13416); systems-biology independently uses "decorrelate NN from mechanistic"
  = orthogonal projection (npj 2024, UNVERIFIED/paywalled) → validates Györök, localizes our novelty to the
  LPV/LFR extension. Direction 3 DONE: symmetry/equivariance = CLOSED DOOR (our system is forced+dissipative
  not conservative; translation-equivariance would forbid the position-dependence we need — cogging,
  Y-scheduling). **SWEEP COMPLETE** — see the consolidated takeaways at the end of the sweep doc; strongest
  actionable outputs are Directions 1+5 (unrolling-first + state-consistency regularizer) for D-107.
- **`docs/passivity-augmentation-literature.md`** — primary-read literature catalog (§G adversarial
  verification, §H marginal-native dissipativity theory).
- **Sweep Directions 10–11 (in `ml-for-control-search-sweep.md`) + D-110** — the post-D-108 extended R5
  rounds (2026-07-11/12): corrupted-scheduling flagship + Cox thesis boundary statement (§11.3.1, gap
  confirmed in-lineage); the three converging detune-bound legs (Cox Ch.3 / NASA parasitic-term / Hanema
  tubes) — all control-premised, so the R4→R5 argument is FINITE-HORIZON only; Layer-3 vocabulary
  (self- vs external scheduling, LPV-C/A/O). PDFs: `literature/corrupted-scheduling/`,
  `literature/theses-lpv-lineage/`, `literature/aerospace-qlpv/`.
- **`docs/decisions.md`** — D-104 (lit verified), D-105 (identifiability reframe), D-106 (marginal
  dissipativity exists), D-107 (open-loop-only decision).

---

## 1. The problem in one paragraph

A physics-based LPV-LFR baseline (Garcia-Herreros dual-gantry, Y-scheduled inertia) is
augmented with a learned parallel ANN (Hoekstra LFR framework, SUBNET encoder). Training is
BPTT over short free-run windows (nf = 0.1 s at 4 kHz); the deliverable metric is
full-trajectory free-run simulation. Truth = baseline + a hidden mass-spring-damper absorber
(fa = 150 Hz, `delta_a` RMS ≈ 2.2e-5 m). Excitation is a narrowband multisine [130,180] Hz.
Logical state `[X, Θ, Y, Ẋ, Θ̇, Ẏ]` + absorber `[δ_a, δ̇_a]`. **X and Y are zero-stiffness
mass-damper axes**: velocity is damped (finite τ_X ≈ 1.55 s, τ_Y ≈ 1.01 s) but **position is a
free integrator**. Routing the ANN to X/Y is a hard requirement (D-103). Symptom: `val sim-RMS`
rises during training, best checkpoint = epoch 0, while the windowed loss stays flat.

---

## 2. Diagnosis — ruled OUT (with evidence)

Each was tested on the actual data/model, not asserted.

| Hypothesis | Test | Result | Verdict |
|---|---|---|---|
| **A1** input / storage / frame / alignment bug | `d1`: full-truth from true x0 + stored u, ±1-sample & P-frame perturbations | correct variant ~5e-8 m; shifts/no-P 100–1000× worse | **rejected** |
| baseline physics broken | `d2`: full-truth from true x0, full 12 s | reproduces data to **1.2e-7 m** | **physics correct** |
| **A2** encoder init x0 seeds the drift | `d3`: encoder x0 vs true x0; free-run from each | encoder x0 clean (dX≈7e-6, dY≈5e-4); encoder-seeded run drifts *less* | **rejected** |
| lr is the lever | Optuna 69399 (lr∈[1e-8,1e-5]) | 4/5 trials revert to epoch 0 (one lr improved 18%) | **not the lever** |
| nf is the lever | same | best = epoch 0 up to nf=1600 | **not the lever** |

**Key correction to prior framing**: X/Y are **mass-dampers (finite τ), not pure double
integrators**. The "no damping" wording in earlier notes is falsified by `d2` Part 2: an initial
velocity error settles to a bounded `τ·dv` (measured Y −9.97e-5 vs predicted τ_Y·dv = −1.06e-4),
which only happens with damping. Position is still a free integrator; velocity is damped.

---

## 3. Diagnosis — CONFIRMED cause (measured on the trained ANN)

**`d6` (direct test on the trained X+Θ+Y checkpoint `gantry_drift_last.pth`)**: the ANN output
on the K=0 velocity rows is a **near-pure DC (constant) offset**.

- `dY` row: `|mean|/rms = 1.00` (pure DC), mean = −1.42e-6 (normalized).
- Counterfactual — subtract the per-row mean during the free-run:
  - Y drift 2.59e-2 m → 1.95e-4 m (**~133× smaller**),
  - X1/X2 2.19e-3 → 1.84e-4 (~12×).

**Mechanism**: the ANN adds its output to the next state on the routed rows. A constant on a
**velocity** row is a persistent force; a **free position integrator** turns it into a linear
ramp `≈ c·t` with nothing to arrest it. A zero-mean oscillatory output (what the absorber
correction should be) would integrate to a bounded wobble. So only the DC part ramps.

**Why training can't see it (`d7`, S1)** — **CORRECTED by §3b (d8, 2026-07-11): the loss is not
merely blind to the DC, it actively PREFERS it; read §3b for the measured loss-side mechanism**:
- In-horizon RMS (0.1 s) is absorber-level (Y ≈ 2.65× absorber over all windows) — **not**
  "bad in-horizon"; the model is roughly fine on the training horizon.
- RMS-vs-horizon: flat (≈absorber) until ~0.25 s, drift **enters ~0.5 s**, 733× absorber by 12 s.
- So **nf = 0.1 s sits below where the drift appears**; the ramp is RMS-invisible over 0.1 s, so
  the loss never penalises it and it compounds over the 12 s free-run. (SUBNET/Ribeiro theory:
  the simulation-error loss is exponentially ill-conditioned on non-contractive/integrator modes,
  so a short window is blind to the mode that dominates the long rollout.)

**Open-loop isolation (`s3`)**: driving both models open-loop with only the stored multisine
(feedback killed), the augmented model drifts ~40× more than the true system → the drift is
intrinsic to the ANN, not the loop or the data. (Multisine confirmed zero-mean over the full
record.)

---

## 3b. LOSS-SIDE MECHANISM CLOSED (d8 + d9, 2026-07-11): the windowed loss REWARDS the drift-driving DC, because the DC compensates the encoder-re-init ramp of the training geometry itself

**Scripts**: `scripts/gantry/diagnostics-drift/d8_dc_visibility_horizon.py` (D-109) and
`scripts/gantry/diagnostics-drift/d9_dc_compensation_shape.py`. Both are FORWARD-ONLY (no BPTT;
the nf=4000 566 MB training wall does not apply) and reuse the `d6` ANN-shadow + `d7` S1a
windowed-encoder-re-init machinery. **Object**: `gantry_drift_last.pth` (post-D-101 checkpoint,
lr=1.49e-8/nf=1400 config; NOT the 07-11 de-confound config lr=1e-7/nf=400, whose checkpoint was
never saved — regenerate via `make_drift_checkpoint.py LR=1e-7 NF=400` if config-sensitivity is
suspected). **Artifacts**: `simulations/gantry_subnet/diagnostics/d8_dc_visibility_gantry_drift_last.{png,npz}`,
`d9_dc_compensation_gantry_drift_last.{png,npz}`, `d8_run.log`, `d9_run.log`.

### d8 — the visibility question, answered by SIGN (refutes the "just increase nf" route)
Windowed RMS of the trained model vs its mean-debiased twin (fixed per-row ANN mean, measured on
the near-truth windowed passes, subtracted at evaluation), paired per-window statistics:

| nf | window | Δ pooled (full−debias) | paired Δ/SE | Y Δ/SE | Y windows full-better |
|---|---|---|---|---|---|
| 400 | 0.10 s | −2.7e-7 | −2.0 | −2.2 | 73/119 |
| 1000 | 0.25 s | −3.8e-6 | −1.9 | −2.1 | 32/47 |
| 2000 | 0.50 s | −2.0e-5 | −1.7 | −1.8 | 16/23 |
| 4000 | 1.00 s | −2.7e-5 | −0.4 | −0.4 | 6/11 (underpowered) |

Δ NEGATIVE = the DC-carrying model fits the window BETTER. The d7 story ("the ramp is
RMS-invisible at 0.1 s") is corrected: **the loss can distinguish the model from its DC-free twin
at nf=400 — and prefers the one that drifts.** X1/X2 neutral (|Δ/SE| ≤ 1.2 everywhere); the
effect is Y. The ANN's dY-row output is pure DC on the training distribution (|mean|/rms = 0.997).
**Consequence**: the in-window benefit and the t² free-run cost cross over ABOVE 1 s — beyond any
feasible BPTT horizon — so moderate-nf training is refuted by SIGN, not just by memory cost.

### d9 — what the DC compensates (H1 confirmed: the encoder-re-init ramp; H2 force-like refuted)
Ensemble-mean SIGNED Y-error over 119 encoder-re-init nf=400 windows, full vs debiased, plus the
encoder's init error measured directly against `val_x_logical` at every window start. Fit
`e(t) = a + b·t + c·t²` on the debiased (= exposed-systematic) curve:

| check | fitted (debiased trend) | independent measurement | verdict |
|---|---|---|---|
| offset a | +1.62e-6 m | trained-encoder Y init bias +1.39e-6 m (6.1 SE) | match |
| slope b | +2.09e-4 m/s | encoder dY init bias +2.68e-4 m/s | match (22%) |
| H2 curvature c | −1.13e-4 m/s² | force-like prediction −0.5·a_DC = **+1.35e-3** m/s² | **wrong sign, 12× → H2 REFUTED** |

Trend is linear-dominated (|b|·T = 2.1e-5 m vs |c|·T² = 1.1e-6 m at T = 0.1 s → 18:1). Machinery
check passed: the exposed (deb−full) trend mirrors the DC's own predicted effect (b −4.62e-5 vs
−4.81e-5; c +1.288e-3 vs +1.349e-3). **Untrained-encoder discriminator**: the dY init bias is
IDENTICAL untrained vs trained (+2.675e-4 vs +2.681e-4 m/s) → a property of the encoder INIT
SCHEME (the linear reconstructability map differentiating the position window,
[[trace-state-reconstruction]]), not created by training. The Y-POSITION init bias grew 12×
through training (1.1e-7 → 1.39e-6, 6 SE) — a secondary co-training artifact.

### The closed causal chain (each link measured)
1. The encoder over-estimates dY by ~+2.7e-4 m/s (mean over V1 window starts; init-scheme
   property, d9 Measurement 2).
2. Windowed training RE-CREATES this error at every window — the re-init geometry manufactures a
   recurring positive in-window Y-ramp (d9 debiased curve).
3. The ANN's loss-optimal response is a persistent negative dY-DC (a_DC ≈ −2.7e-3 m/s²) whose
   quadratic partially cancels the ramp (~40% at 0.1 s) — measured as the d8 preference.
4. In free-run the encoder error occurs ONCE (bounded τ·dv offset, d2/d3), but the learned DC
   applies EVERY step and integrates without bound (d6: removing it cuts drift 133×) = the drift.

**One sentence**: *windowed encoder-re-init training manufactures a systematic (the encoder's dY
init ramp) that the augmentation learns to compensate with a persistent DC, and that compensation
IS the free-run drift.* This also explains the start-sample dependence of drift magnitude, and it
is the §5m real-data collision made concrete on the frictionless sim: the DC slot is occupied by
pipeline artifact, not physics.

### Honest caveats
- The debiased-trend endpoint is 1.9 SE (marginal alone); the verdict rests on the two
  coefficient matches + the H2 sign refutation.
- The dY init bias is 1.0 SE across V1's windows (large per-window variance): it is the realized
  mean on the evaluation set — confirm on the TRAINING-set windows before building the fix.
- Subtracting a fixed mean from a co-trained net is an off-manifold counterfactual; a
  state-dependent slow component is only partially removed.
- Single checkpoint/config; regenerate the de-confound-config checkpoint if sensitivity is suspected.

### Consequences for §5 (direction, not yet a decision entry)
- **Conditioning-by-longer-nf is dead on this problem** (refuted by sign, d8) — this also
  retro-explains the Optuna 69399 / D-067-curriculum failures.
- **The primary fix moves UPSTREAM of Layer 2: reduce the encoder's dY init error at the source**
  (fix-where-the-assumption-lives; the encoder owns the error). If the recurring ramp disappears,
  the loss's reward for the DC disappears with it.
- **Layer 2 (data-silent projection) stays as insurance**, now with a precisely-stated target:
  the direction the projection must pin is the encoder-bias-compensation direction, and the
  encoder fix is what makes that direction genuinely data-silent (else the projection fights a
  loss-rewarded component — exactly what d8 showed it would lose to).
- Note for the joint-estimation picture: encoder dY-bias ↔ ANN dY-DC is a coupled, nearly
  loss-neutral degenerate direction — a projection acting on the ANN alone leaves the encoder
  half in place.

---

## 3c. BASELINE-LEVEL DECOMPOSITION (d17, 2026-07-13): the UNTRAINED free-run error = encoder-IC bounded offset (dominant) + absorber residual + baseline replay offset; NOTHING drifts unbounded without the ANN

**Script**: `scripts/gantry/diagnostics-drift/d17_msd_vs_encoder_decomposition.py`. **Object**:
`T1_standstill_Ym30`, with-MSD (`data/gantry/matlab/trajectory/augmentation/`) vs no-MSD
(`.../augmentation/baseline/`), UNTRAINED baseline (no ANN). **Artifacts** (in
`simulations/gantry_subnet/diagnostics/`): `d17_excitation.png`, `d17_decomp_stage.png`,
`d17_decomp_logical.png`, `d17_decomp_stage.npz`.

**Method (closed-loop-consistent, with-MSD frame).** The absorber changes the recorded total
force (excitation finding below), so `u(with MSD) != u(no MSD)` and the no-MSD OUTPUT is NOT a valid
floor; the no-MSD data is used ONLY for the input/excitation figure. The decomposition uses the exact
additive identity, everything driven by the recorded `u_w`, referenced to `y_w`:
- `E = sim_baseline(x0_enc, u_w) - y_w` = actual free-run error (encoder x0)
- `R = sim_baseline(x0_true, u_w) - y_w` = residual at TRUE x0 (absorber + baseline replay)
- `enc_IC = sim_baseline(x0_enc, u_w) - sim_baseline(x0_true, u_w)` = pure encoder-IC effect; and
  `E = R + enc_IC` (verified visually and numerically).

**Excitation finding (measured).** No-MSD `u_total` ~ 0 (RMS 0.44-0.72 N/axis); with-MSD `u_total`
= 36-45 N RMS / 134-170 N peak, concentrated in the 130-180 Hz band on ALL three force channels;
`delta_a` peaks at 150 Hz. MECHANISM: the multisine is a closed-loop force disturbance; the nominal
loop rejects it (sensitivity `S ~ 0` in-band). The absorber is a tuned-mass-damper whose
ANTI-RESONANCE (notch) at 150 Hz collapses the loop gain there, so the servo can no longer reject the
multisine and it leaks into `u_total`. Self-consistent: the leaked force is exactly what excites the
absorber. So the with-MSD data IS informative (real 45 N of 150 Hz force, absorber excited), and
`u_w != u_n` by construction. This is also why the earlier "u_total ~ 0, uninformative" worry applies
only to the no-MSD baseline (which has no absorber to learn anyway).

**Decomposition findings (measured, T1).**
1. **NOTHING drifts unbounded.** Every channel settles to a BOUNDED plateau (X by ~8 s, Y by ~4 s,
   Theta rings down by ~2 s). The untrained baseline + encoder IC gives bounded offsets (`tau*dv`),
   not runaway. Unbounded drift requires the TRAINED ANN's persistent DC (§3/§3b); it is absent here.
2. **Dominant term = encoder IC on X: 1.45e-3 m (bounded).** On X1/X2 (and logical X),
   `E ~ enc_IC ~ 1.45e-3`, `R ~ 0`; matches the single-axis prediction `tau_X*dvX = +1.28e-3` from
   the encoder's dX velocity error (+8.3e-4 m/s). **The offsetting axis is TRAJECTORY-DEPENDENT**:
   T1 -> X (dX worst-reconstructed); V1 (d9) -> Y (dY). Same mechanism (encoder velocity error ->
   K=0 offset), different axis by force distribution / operating point.
3. **Theta rings down (sprung yaw mode).** The encoder dTheta error (+2.07e-3 rad/s) kicks the sprung
   + damped yaw mode (~5-6 Hz), which decays to 0 by ~2 s (`R` flat on Theta). Benign, no steady
   contribution. This IS the d6 "X1/X2 oscillation toward 0" (stage X1/X2 = X +/- Theta). STIFFNESS
   CONTRAST, on one plot: Theta returns to ZERO (has stiffness `kb1+kb2`); X/Y settle to a PERMANENT
   offset (K=0, no stiffness). Same IC velocity error, two fates, decided by the spring.
4. **Absorber residual `R` is on Y.** Settles ~+3.8e-4 with a 150 Hz ripple. `R ~ 0` on X because the
   absorber couples only to Y and Theta in the mass matrix (`da` column is zero in the X row), never
   X. The OFFSET beneath the ripple is a baseline OPEN-LOOP-REPLAY effect (the free Y integrator
   accumulating `u_w`'s low-frequency content, plus rigid-vs-sprung absorber-mass handling), the §4
   baseline-multisine side-finding made concrete; bounded. [INFERRED, not isolated; check = scale /
   notch the low-freq band of `u_w` and see whether the offset tracks it.]
5. **`enc_IC` on Y is opposite sign and 9x its direct prediction.** `enc_IC(Y) = -2.3e-4` vs
   `tau_Y*dvY = -2.5e-5`. It is dominated by CROSS-COUPLING of the encoder's large dX (+8.3e-4) and
   dTheta (+2.07e-3) errors into Y via the mass-matrix off-diagonals, not the small direct dY error.
   Contrast: on X, `enc_IC` (1.45e-3) matches its direct prediction (1.28e-3) because X is weakly
   coupled. So Y's `E = +1.46e-4` is a PARTIAL CANCELLATION of two independent bounded effects
   (`R` and `enc_IC` opposite signs).

**Implication.**
- The encoder x0 is the DOMINANT error source in the untrained free-run (1.45e-3 on X vs ~3.8e-4 Y
  residual vs 2.2e-5 absorber RMS). Replacing encoder x0 with MEASURED initial conditions removes the
  largest term. Direct quantified support for the supervisor's "gebruik geen encoder, kijken initial
  conditions" (2026-07-13).
- But the encoder IC is BOUNDED, so it is NOT itself the unbounded-drift culprit; it is the SEED
  (§3b/d9): windowed training re-creates this IC velocity error every window, the ANN learns a
  persistent DC to compensate the recurring ramp, and THAT DC integrates into the unbounded drift.
  d17 shows the seed; the ANN turns it into the drift.
- Damping bounds a one-time IC velocity error (a cart pushed once coasts to a stop); only a persistent
  force (the ANN DC) drifts unbounded. d17 is the on-plot demonstration.

**Honest caveats.** T1 only (single trajectory / operating point Y=-30); rerun on a second trajectory
to confirm the encoder-IC axis tracks the force distribution. The R-Y offset mechanism (low-freq
integration vs absorber-mass handling) is not isolated. Untrained baseline (no ANN): this is the
drift SEED, not the drift.

---

## 4. Unifying rule

```
drift = (force/model error with nonzero DC)  ×  (free integrator, K=0)  ×  (full free-run, no re-seed)
```

Remove any one factor and drift vanishes:

| Case | force error? | free integrator? | re-seeded? | drift |
|---|---|---|---|---|
| sim param recovery, perfect params | no (model = data) | yes | no | **none** |
| real param recovery, open-loop | yes (friction, model wrong) | yes | no | yes |
| real param recovery, training | yes | yes | yes (masks) | bounded |
| Θ axis (has a spring) | yes | no | no | bounded |
| **X/Y augmentation** | **yes (ANN DC)** | **yes** | no | **yes** |

The augmentation can never reach "perfect" (an ANN always leaves a residual force, and a generic
NN residual has nonzero DC on the K=0 rows), so the fix must either forbid the DC or stop it
accumulating — while keeping X/Y in the routing (D-103).

**Side finding (flagged, not concluded)**: even the true baseline, driven purely open-loop by a
zero-mean multisine, has a small constant drift on the K=0 axes (~1e-4 m/s), likely a nonlinear
`M(Y)` rectification — mechanism not isolated. It is ~40× below the ANN drift and the real
validation replays `u_total` (reproduces to 1e-7), so it does not affect the augmentation
problem; raise it as a meeting nuance only.

---

## 5. Candidate solutions — ranked by the REAL-DATA selection criterion

**Selection criterion (the deliverable is real nonlinear data with UNKNOWN dynamics).** A solution
must give a guarantee that is:
1. **Knowledge-free** — holds without knowing the true residual/dynamics (a real-data method cannot
   assume it knows what the ANN "should" output);
2. **Friction-permitting** — still represents genuine dissipative, state-dependent residuals on X/Y
   (friction `~sign(v)`, cogging, standstill preload — R3);
3. **Marginal-preserving** — keeps the free-integrator (zero-stiffness) physics; must NOT damp it
   into a strictly-stable axis;
4. **Non-drifting** — the augmented free-run stays bounded on X/Y (R2), and keeps X/Y routing (D-103).
5. **Scheduling-integrity (the Y conflict)** — **Y is SIMULTANEOUSLY a K=0 free-integrator (it DRIFTS)
   AND the LPV scheduling variable (`M(Y)` depends on Y-position).** A valid method must NOT corrupt the
   Y-scheduling dependence: (a) it must not train out / suppress the legitimate `M(Y)` position-dependence
   when making the model robust to Y-drift; (b) it must not achieve no-drift on Y by a restoring action
   that both damps the Y pole (violates R3) and forces the model to ignore Y-position. **Consequence: Y is
   the HARDEST axis.** On X (translation-invariant, not a scheduling variable) drift-fixes are clean; on Y,
   drift-robustness and scheduling-dependence directly CONFLICT — exposing/perturbing Y-position to fight
   drift risks destroying `M(Y)`, and "correcting Y back" risks damping the Y pole.

   **CONFIRMED (2026-07-11, code-read): the pipeline uses LPV SELF-SCHEDULING — `M(Y)` reads the PROPAGATED,
   DRIFTING Y-state, not an exogenous/measured Y.** `scripts/gantry/gantry_dynamic/model.py` builds the
   physics block with `Y_op=None` at both sites (L80, L88); `model_augmentation/fit_systems/blocks.py:659`:
   `Y_op=None` -> "LPV self-scheduling: Y = x[2] per step". So the conflict is REAL and ACTIVE now (not just
   a Phase-3 hypothetical), and it is confirmed by code, not assumed.
   - **Extra coupling X does NOT have:** because `M(Y=x[2])` reads the drifting Y, **Y-drift DETUNES the mass
     matrix** — as Y ramps, `M(Y)` is evaluated at the wrong Y -> wrong dynamics -> drift feeds back through
     the scheduling. X has no scheduling feedback. So Y-drift is structurally worse than X-drift (it corrupts
     its own scheduling as it drifts), making X the clean/interpretable axis and Y the hard sub-problem.
   - **Design lever revealed (not adopted):** if `M` were scheduled off a de-drifted / reference Y instead of
     the raw propagated `x[2]`, the conflict would soften — a real option to evaluate, flagged, not chosen.
   - **Still to MEASURE per method:** Y-scheduling integrity after training (held-out Y positions / `M(Y)`
     dependence) AND the Y-pole eigen-check — do not assume any given method's Y outcome.
   [Added 2026-07-11; surfaced by the GNS diagnostic design — `gns-encoder-diagnostic-plan.md`. Self-scheduling
   confirmed by reading `model.py` (Y_op=None) + `blocks.py:659`.]

**Why the "penalise the average force" family is DEMOTED for real data.** The DC/mean guardrail works
in SIM only because we KNOW the true residual is zero-mean (the absorber), so any average ANN force is
provably error. On REAL data the true residual DOES carry near-constant content (friction, preload);
the ANN's average force is then a **mix of real friction + spurious drift that cannot be separated
without knowing the dynamics** — which is the unknown we are trying to learn. So a mean penalty would
suppress real friction. **Fails criterion 1.** It is retained only as a SIM-phase diagnostic (it proved
the mechanism, `d6`: −133× drift), NOT as the real-data deliverable.

### PRIMARY — Dissipativity / passivity-constrained augmentation (knowledge-free)
Constrain the learned block STRUCTURALLY so it can only remove energy from the mechanical system, never
inject it. This is knowledge-free (a property of the block's parametrization, independent of the true
dynamics) and separates good from bad by the SIGN OF POWER alone:
- **drift mode** = sustained force along motion → power `F·v > 0` → injects energy → a dissipative block
  cannot produce it → **forbidden automatically**;
- **friction** = force opposes motion → power `< 0` → removes energy → **permitted**.
So dissipativity forbids exactly the runaway and permits exactly the friction WITHOUT knowing the
friction curve. Meets criteria 1–4 in principle.
**The catch = the open problem = the contribution.** Existing dissipative-NN methods overshoot:
- **contraction-based** (RENs arXiv:2104.05942; our supervisors' **Györök 2026 arXiv:2604.11421**,
  VERIFIED `‖𝒜‖<1` strict) → **damp the free axis, destroy the zero-stiffness physics** (fails crit. 3);
- **port-Hamiltonian / zero-at-equilibrium** (Roth 2025; pseudo-Hamiltonian Eidnes 2023) → force **zero
  force at rest** → **kill Coulomb/static friction** (fails crit. 2).
The needed object — **dissipative, MARGINAL-STABILITY-PRESERVING, permitting nonzero-at-rest velocity-
dependent force** — is exactly what no found method provides. Relaxation of Györök's `<1` to `≤1`
(incremental passivity / Negative-Imaginary). **NI theory** (free-body: Mabrok et al. 2014
arXiv:1305.1079; original NI: Lanzon & Petersen 2008) is the
only classical framework native to free-body/integrator motion systems; constraining a neural block to
be NI is essentially open. **This is the real-data deliverable and the scientific contribution.**

### SECONDARY — Structural integrator factoring (Tustin-Net principle, knowledge-free)
Build the KNOWN integrator (`position = ∫velocity`) as a FIXED layer so the ANN only learns the
bounded, asymptotically-stable RESIDUAL and can never corrupt the integrator itself — the classical
Ljung prescription ("integrator as a known factor, learn only the residual") realised architecturally.
Knowledge-free and keeps a position loss. **Papers**: Tustin-Net (arXiv:2408.12266); Ljung PEM.
**Honest limit**: this bounds WHAT the ANN can corrupt; it does NOT by itself forbid the residual from
having a low-frequency component that still drifts — so it likely needs pairing with the dissipativity
constraint on the residual. Weaker guarantee than I earlier stated ("impossible by construction" was an
overstatement).

### SUPPORTING (conditioning, not the guarantee) — Multiple shooting / continuity
Re-seed state per segment + cross-window continuity penalty `ρ·‖x_end^(m)−x_start^(m+1)‖²`. Makes a
residual bias VISIBLE as a continuity violation (training signal) and bounds accumulation — the reason
param recovery never drifted (`d4`). **Papers**: Turan & Jäschke (arXiv:2109.06786); Ribeiro et al.
(arXiv:1905.00820 — note: ill-conditioning is POLYNOMIAL `N²/N³` at the `L_h=1` integrator borderline,
exponential only for truly unstable). This CONDITIONS the training but does not itself guarantee no
drift on unknown data — a support tool, not the deliverable.

### LAST RESORT — Velocity / acceleration-domain loss (do NOT adopt without explicit go-ahead)
Fit velocity/acceleration instead of position (DIDIM force-domain; Ljung prefilter as the loss). Robust
and knowledge-free, and the literature keeps converging on it — but **supervisor + user directive: LAST
RESORT** (see the standing constraint at the top). `d5`; DIDIM (Gautier/Janot); Ljung. Hold until the
position-based routes fail. (Tustin-Net is structural factoring with a POSITION loss and is NOT this —
do not conflate.)

**Cross-cutting literature point (four independent search passes, key cites verified)**: every
SUBNET/LFR/deepSI paper — including our own framework (Hoekstra LFR augmentation arXiv:2404.01901;
encoder-init arXiv:2602.13108, "we assume a stable baseline model") — and every stable-by-design ID
method enforces **strict stability (`λ<1`)** and gives NO treatment of free integrators. The encoder
itself has no reconstructability map for an integrator (infinite memory). So a **marginal-stability-
preserving dissipative augmentation is genuinely outside the published theory** = the contribution.

---

## 5b. CURRENT CHOSEN SOLUTION and its end goal

Source: deep-research survey `literature/stability-training/claude-research-stability.md` (2026-07-09),
which confirmed the §5 gap is genuine and turned the "dissipativity/passivity direction" into a
concrete, buildable construction. This section is the resolved direction.

### End goal (what we are building toward)
A learned parallel augmentation `g_w` on the free-integrator (X/Y) axes that is **non-drifting on real
nonlinear data with unknown dynamics** while **still representing genuine dissipative state-dependent
residuals (Coulomb/static friction, cogging)** — i.e. it meets all four criteria of §5 (knowledge-free,
friction-permitting, marginal-preserving, non-drifting) and keeps X/Y routing (D-103). The theoretical
target is a **positive-SEMIdefinite-storage passive / nonlinear Negative-Imaginary** learned block
(storage flat along the rigid-body coordinate → pole stays exactly at the origin). This is the marginal
case that every existing **NEURAL/LEARNED** stability construction forbids (they force strictly PD storage).
**[CORRECTED 2026-07-10 -- §5m marginal-dissipativity primary-read.] The CLASSICAL (analytical) theory for
this marginal case EXISTS and is mature** -- indefinite storage (cyclo-passivity, van der Schaft
arXiv:2003.10143), continuum-of-equilibria passivity (EIP/EID, Hines-Arcak-Packard 2011 / arXiv:1709.06986),
flat-storage Casimirs (PH). What is missing is its LEARNED/forward/LPV realization AND the criterion-4
position-bound layer (net-impulse/NI) -- the classical marginal-storage relaxation does NOT itself bound
position (cyclo gives "only instability results"; EID bounds shifted I/O, not the free coordinate; = §5j).
So the scientific contribution is the LEARNED assembly + position-bound coupling, NOT the storage relaxation
(reused) -- a narrower, better-founded claim than "no theory exists".

### The construction (staged; the pragmatic bridge to that goal)
1. **Structural integrator factoring (Tustin/Forgione-Piga).** Route the ANN to the **acceleration
   (velocity-derivative) row** of each K=0 axis and fix the position row to the exact integral of
   velocity. Removes the encoder-reconstructability pathology (d3) and localises drift to the DC of the
   learned acceleration. Keeps a POSITION loss and keeps X/Y routing.
2. **Two structurally-typed channels on the free axis:**
   - **Dissipative friction channel** `F_fric = −Φ(x,q̇)` via a maximal-monotone / set-valued `sign(q̇)`
     structure → `F_fric·q̇ ≤ 0` by construction AND `F_fric(0)≠0` allowed (Coulomb/static). Gives
     criteria 1+2.
   - **Zero-DC coupling channel** as an **exact time-derivative of a bounded function**
     `g = d/dt[ψ(x)]`, `ψ = tanh(net)` bounded (discrete: `g_k = ψ(x_k) − ψ(x_{k-1})`). Its running sum
     telescopes → bounded → the derivative's `s` cancels the integrator's `1/s`, so **position is a
     stable-filtered bounded signal (bounded, not just "no ramp")**. Instantaneous force unconstrained
     (nonzero-at-rest allowed). Gives criterion 4 WITHOUT a data-dependent mean penalty (so it does not
     suppress real friction). No stiffness/damping added to the position row → pole stays at origin →
     criterion 3.
3. **(optional, for a single certificate) Semidefinite-storage passive/NI REN.** Relax a passive
   REN/NodeREN metric `P≻0` to `P⪰0` with the zero eigenvalue on the rigid-body coordinate, using the NI
   supply rate `uᵀẏ`. The clean single-theorem route; hardest and unbuilt.

### Critical caveat (buildable plan vs clean theorem — do not conflate)
The bounded-integral trick gives **bounded impulse**, which is NOT the same as **passivity**: the
two-channel split therefore has TWO separate boundedness reasons (dissipative friction + bounded-impulse
coupling), not one energy certificate. So the staged construction is the **best buildable plan now**;
the **single semidefinite-NI certificate (step 3)** is the cleaner thesis theorem but remains the hard,
unbuilt prize. Also: the clean two-channel separability is a physically-reasonable but a-priori
unverifiable assumption on real data; if the true system has a genuine SUSTAINED non-dissipative DC force
(gravity/preload), it belongs in `f_base`, not the residual.

### How to de-risk before building (diagnostic plan)
- **D-A (DONE 2026-07-09, `dA_residual_dc.py`).** Along the true trajectory computed the residual the
  ANN must supply on X/Y = `(truth q̈) − (baseline q̈ at same state)` (baseline 3-DOF EOM = truth EOM
  with absorber removed; VERIFIED vs `simulate_baseline` to 3e-12 m). **Result (nuanced, validates the
  two-channel split):**
  - The DOMINANT residual (absorber coupling, rms ≈ 2.2 m/s² on Y) is **zero-DC oscillatory**
    (`|mean|/rms` ≈ 1e-4…1e-5) → the **bounded-integral channel represents the bulk perfectly** ✓.
  - A small absolute DC exists → integrated drift ≈ 3.5e-4 m (V1) … 3.8e-3 m (T10), i.e. 16–160×
    absorber. This is the nonlinear `M(Y)` **rectification** (quantifies the S3 side-finding; larger when
    X/Y move).
  - **v-projection (fit `a_res = α·v + β`; β = velocity-INDEPENDENT DC = the part that drifts), DONE.**
    Result is AXIS-DEPENDENT and at the edge of measurability (DC ≈ 1e-4 of RMS, fit R²≈0; V1 standstill
    is unreliable — no velocity to test against, gives a cancellation artifact). On T10 (X/Y moving, the
    informative record): **Y** DC is ~97% velocity-explained → **dissipative, self-limiting → safe**
    (dissipative channel); **X** DC is ~100% **external** (β≈mean, α≈0) → NOT self-limiting → would drift
    open-loop (~3.8e-3 m) → belongs in `f_base`. Magnitudes ~1e-3 m and **controller-counteracted** in the
    real closed-loop data (true X/Y don't actually drift).
  - **CORRECTED conclusion (earlier "D-A validates the two-channel / DC is dissipative" was too
    optimistic):** ROBUST result — the dominant coupling is zero-DC → **bounded-integral channel viable**.
    NUANCE — the small residual DC is **partly dissipative (Y, safe)**, **partly external (X, would drift
    open-loop)**, noisy, and controller-counteracted → a **second-order refinement, not a showstopper**;
    the two-channel split does NOT cleanly absorb ALL of it, a genuine external DC (X) would need `f_base`.
- **D-B (DONE 2026-07-09, `dB_boundedintegral_projection.py`).** On the drifted checkpoint, applied
  raw / mean-removed / bounded-integral (online causal high-pass, fc=30 Hz) to the ANN routed-row output
  during the 12 s free-run. **Results (strong, clean):**
  - **Absorber capture UNTOUCHED:** 130–180 Hz band RMS on Y = 2.11e-6 m for ALL three treatments →
    the high-pass leaves the 150 Hz absorber correction completely intact. **Structural projection is
    compatible with the learned signal** (the key D-B question) ✓.
  - **High-pass STRICTLY BEATS mean-removal on X:** X drift raw 2.19e-3 → mean 1.84e-4 → **high-pass
    2.05e-6** (~90× beyond mean-removal). So the bounded-integral projection catches drift-causing
    LOW-FREQUENCY content that pure DC-removal (d6) misses — consistent with D-A's finding that X carries
    an external (non-pure-DC) component. This makes bounded-integral **stronger than the DC guardrail**.
  - **On Y, high-pass ≈ mean-removal** (both ~2e-4 m, ~10× absorber). Slope check: raw slope −2.7e-3 m/s
    (ongoing drift), mean −6.5e-6, **high-pass −1.5e-7 m/s (flat → bounded offset, NOT ongoing drift)**.
    So the ~2e-4 Y residual is a **bounded settle**, most likely the ENCODER initial-state offset
    (d3: Y velocity error ~5e-4 → `τ_Y·dv` ~ 2e-4 m) — a fixed IC effect, not ANN drift, that the output
    constraint cannot and should not fix.
  - **Conclusion:** the bounded-integral (high-pass) projection removes the ANN-induced drift (Y
    2.6e-2→flat, X →2e-6) while preserving absorber capture, and is strictly better than mean-removal on
    X. Residual Y offset is a separate bounded encoder-IC effect. D-B validates the bounded-integral
    channel as the drift-removal mechanism (post-hoc; trainability is D-C).
- **D-C (the fix itself; short training run).** Train a small ANN WITH the bounded-integral
  parametrization; check it learns the absorber without drift. Only after D-A/D-B pass.

### Validation once built (from the deep-research Stage 4)
Long-horizon free-run ≥50× the training window; constant-force probe (bounded position from the learned
block); at-rest breakaway-force test; energy audit `∫F·q̇ ≤ 0` on the free axes; linearised eigen-check
that the X/Y pole stays at the origin (not pushed inside).

### Honest provenance
Classical anchors verified (NI free-body Mabrok et al. 2014 arXiv:1305.1079; RENs arXiv:2104.05942;
Tustin-nets Forgione-Piga). The four-requirement method is a **synthesis of building blocks, not a
citable result** (the deep-research author self-flags this). Several 2025/2026 friction-net and
dissipative-NN cites are UNVERIFIED — leads, verify before thesis use. The bounded-integral position-
boundedness (pole cancellation) was re-derived and holds; trainability and separability are unproven.

---

## 5c. Coulomb friction vs integrator-bounding — two DIFFERENT problems

These are often conflated; they are distinct and complementary. One is what we are solving now; the
other is a real-data-only addition.

### Problem 1 — DRIFT / integrator bounding (what we are solving NOW, on sim)
On the K=0 (X/Y) axes position is a free integrator, so ANY nonzero-mean (DC) force the ANN produces
ramps into unbounded free-run drift (measured: d6, −133x on Y). This exists even with the TRUE
parameters and NO friction, because it is a **training pathology**: the 0.1 s windowed loss cannot see
the DC mode (drift only appears past ~0.5 s, S1), so the optimizer is free to put a spurious DC there.
- **Fix = integrator bounding (the bounded-integral channel).** Parametrize the ANN's free-axis
  contribution as `g_k = ψ(x_k) − ψ(x_{k-1})`, `ψ = tanh(net)` bounded → its running sum telescopes →
  bounded → the derivative's `s` cancels the integrator's `1/s` → position is a stable-filtered bounded
  signal → NO drift, by construction. Validated post-hoc (D-B: high-pass = same mechanism, removes
  drift, keeps the absorber, beats mean-removal ~90x on X). This is knowledge-free (a property of the
  parametrization, not of the data). **This forbids the ANN from PRODUCING a drifting force.**

### Problem 2 — FRICTION representation (real-data only; the sim has no friction)
The real machine has friction (Coulomb `−Fc·sign(v)`, or LuGre): a genuine, physical, DC-carrying
force on the K=0 axes that the model MUST reproduce. The current frictionless baseline lacks it, so on
real data the ANN would be forced to learn it.

### The crux — why they COLLIDE, and how Coulomb-in-baseline resolves it
On real data the two problems conflict directly:
- integrator bounding **forbids** the ANN's DC force (to stop drift), but
- representing friction **requires** a DC-carrying force (friction has a nonzero mean over asymmetric
  motion).

You cannot both forbid and require the ANN's DC. **Resolution = put friction in the physics baseline
`f_base` as a fitted Coulomb/LuGre term (grey-box).** Then:
- friction is explained by PHYSICS → the ANN no longer NEEDS to produce a DC to represent it →
- the ANN's remaining residual (the coupling; D-A showed it is ~zero-DC) is compatible with the
  bounded-integral channel.
- Coulomb friction is itself dissipative (`F·v ≤ 0`) so it also adds a physical dead-zone that resists
  small drift (a sub-threshold force produces no motion).

### The clean division of labour (both are needed; they are NOT substitutes)
| | Coulomb in `f_base` | Bounded-integral channel (integrator bounding) |
|---|---|---|
| Solves | FRICTION representation (real data) | DRIFT (integrator) |
| Mechanism | model the known DC-carrying physics | forbid the ANN from producing a DC force |
| Removes | the ANN's **legitimate** need for DC | the ANN's **spurious** ability to make DC |
| Needed on sim? | NO (no friction in sim → mismatch) | YES (drift is a training pathology) |
| Needed on real data? | YES (real friction exists) | YES (spurious DC still possible) |

**Coulomb does NOT by itself fix the drift:** it removes the ANN's *legitimate* reason to produce DC,
but the *training pathology* can still make the ANN produce a *spurious* DC — only the bounded-integral
channel forbids that. Conversely the bounded-integral channel alone, applied on real data WITHOUT
friction in `f_base`, would suppress real friction. **So the real-data solution is BOTH: Coulomb in
`f_base` (friction) + bounded-integral ANN (no drift).** Caveat: this assumes a parametric friction
model (Coulomb/LuGre) is adequate; if it is not, friction falls back to a learned dissipative channel
(§5b, the harder research route). Adding Coulomb to `f_base` is a real-data step; do NOT add it to the
sim (no friction there → model–data mismatch).

## 5d. POSSIBLE BETTER ALTERNATIVE — grey-box: bounded-integral ANN + Coulomb in `f_base`

A simpler, lower-risk alternative to the full two-channel research construction (§5b). Not yet decided;
recorded as the leading candidate for the real-data deliverable.

### The architecture
| Piece | Role | Status |
|---|---|---|
| **ANN = bounded-integral (integrator-bound) channel ONLY** | learns the zero-DC coupling; structurally cannot drift | validated (D-A/D-B) |
| **Coulomb/LuGre friction in `f_base`** | captures the DC-carrying friction physically (grey-box, fitted `Fc`) | established system ID |
| ~~learned dissipative friction channel~~ | ~~learn friction with a constrained maximal-monotone net~~ | **DROPPED** |

**Coulomb-in-`f_base` REPLACES the learned dissipative friction channel** (the hard, unbuilt,
research-grade part of §5b). The ANN then needs only the integrator-bound channel.

### Why it may be better than the full two-channel construction
- **Lower risk / fewer unknowns:** uses established friction physics (fitted `Fc`/LuGre) + the
  bounded-integral channel already validated — instead of a novel constrained dissipative net that has
  no off-the-shelf recipe and is nonsmooth to train.
- **Clean separation, no conflict:** friction's DC lives in physics; the ANN's residual is the zero-DC
  coupling (D-A) → the bounded-integral channel is exactly the right (and only) tool for it.
- **Integrates with the thesis interpretability contribution:** `Fc` is a fitted `f_base` parameter, so
  joint estimation + orthogonal projection (Györök) applies to keep it identifiable.
- **Derivation impact is small** (§ below / m-matrix-invertibility.md): friction is a RHS force, not in
  `M(Y)`, so M-invertibility and well-posedness are unchanged; Coulomb is dissipative and adds no
  stiffness, so it PRESERVES the marginal (integrator) mode and cannot destabilize.

### Conditions (when it is sufficient)
1. **Coulomb/LuGre must adequately model the real friction.** If it does, the ANN residual is zero-DC
   coupling and bounded-integral suffices.
2. **Any residual dissipative DC that Coulomb misses must be small.** Whatever Coulomb does not capture
   (nonlinear friction beyond the model, the `M(Y)` rectification from D-A, etc.) carries a DC that the
   bounded-integral channel FORBIDS rather than represents → a small BOUNDED OFFSET (not drift). That is
   the price of dropping the dissipative channel: un-modeled dissipative DC is refused, not learned.

### Risks (from the composition analysis)
- **Nonsmoothness of `sign(q̇)` can break the BPTT training** of the bounded-integral ANN (the whole
  rollout becomes nonsmooth) → use **LuGre or a smoothed `tanh(q̇/ε)`** friction to keep the rollout
  differentiable. Main risk to watch.
- **Bounded-integral forbids the ANN from correcting a friction-model DC error** → the combo relies on
  the friction model being adequate; if not, a bounded fidelity offset remains.

### Fallback
If, on real data, Coulomb/LuGre is inadequate and the leftover dissipative DC offset is unacceptable,
escalate to the learned dissipative friction channel (§5b, the harder route). So:
**Default = bounded-integral ANN + Coulomb `f_base` (simple, low-risk). Escalate to the learned
dissipative channel only if the friction model proves inadequate.**

### Test plan (stage the composition, don't jump to real data)
1. **D-C on sim (no friction):** bounded-integral ANN trains + no drift + keeps the absorber.
2. **Sim + INJECTED known Coulomb friction:** inject a known `Fc` into the sim truth; confirm (a) the ANN
   drifts if friction is unmodeled, then (b) add the matching Coulomb term to `f_base` and confirm
   bounded-integral + Coulomb trains and stays bounded. The controlled proxy that actually tests the
   composition before real data.
3. **Real Telica data.**

**Status: unbuilt/untested — reasoned expectation, not demonstrated.** Neither D-C nor the Coulomb
composition has been run. Do NOT add Coulomb to the current sim (no friction there → model-data
mismatch); it is a real-data / injected-friction-proxy step only.

## 5e. ACCEPTANCE CHECKLIST — what must be shown for the passivity-constrained augmentation

For the passivity/NI-constrained ANN (the single-constraint generalization of the bounded-integral
channel: forbid energy-injecting drift, allow oscillatory coupling AND dissipative friction) to be an
acceptable method in the LPV-LFR + SUBNET framework, the following must be shown. Three tiers.
Terminology note: use "passive (dissipativity-constrained) ANN", a SINGLE constraint on the whole X/Y
output, not "dissipative channel" -- the passivity (integral) notion allows both the coupling (net-
dissipative, transiently injecting) and friction, so no two-channel split is needed (pointwise
`F·v<=0` would be too tight for the oscillatory coupling; passivity `int F·v` bounded is the right one).

### A. Theoretical proofs (the contribution; hardest)
1. **Passive/NI by construction, for ALL parameter values.** The parametrization guarantees the supply
   rate (`int F·v <=` storage, or the NI condition) for every weight setting and input -- not just at the
   trained optimum. This is the knowledge-free "by construction" claim.
2. **Boundedness of the augmented interconnection on the marginal modes -- THE key theorem.** `f_base`
   (passive mechanical system incl. the free integrator) + passive ANN => bounded free-run on X/Y.
   SUBTLETY: passivity force->velocity bounds VELOCITY (kinetic energy), NOT POSITION on a free
   integrator; bounding the position output needs the NEGATIVE-IMAGINARY (free-body) argument, not plain
   passivity. This is why Mabrok et al. 2014 (free-body NI) is load-bearing.
3. **Marginal-mode preservation.** The constraint adds NO stiffness/damping that moves the integrator
   pole off the origin (keeps `||A|| = 1` admissible; storage positive-SEMIdefinite, flat along the
   rigid-body coordinate). Contrast: contraction (RENs/Gyorok) forces `||A|| < 1` and damps it away.
4. **Expressivity (friction-permitting).** The constrained class can still represent the target:
   dissipative friction (`F·v <= 0`, nonzero at rest) AND the oscillatory coupling (R3).

**The genuine gap = the contribution:** proofs 2-3 for the NONLINEAR LPV case (our `M(Y)`, Y-scheduled,
not LTI). **[CORRECTED 2026-07-10 -- see §5L verification addendum.]** ~~NI theory is mostly LTI; the
nonlinear-NI SEMIdefinite-storage version (Ghallab-Petersen direction) is unworked-out.~~ The nonlinear-NI
semidefinite-storage version IS worked out analytically (Shi-Petersen-Vladimirov 2011.14610 Def 1;
Ghallab-Petersen 2201.00144) -- what is unworked is the **LEARNED, forward-augmentation, LPV-scheduled**
realization of it. Hardest, not off-the-shelf, but built ON an existing analytical foundation rather than
from scratch.

### B. Framework well-posedness (fits LPV-LFR + SUBNET)
5. **LFR well-posedness with the added block:** no algebraic loop (acyclic signal graph), and `M(Y)`
   invertibility unaffected (the passive ANN is a FORCE, not in `M` -- same argument as Coulomb,
   m-matrix-invertibility.md; verify it holds).
6. **Encoder compatibility:** any added storage/filter states must be initialized well-definedly by the
   SUBNET encoder (or init to a known rest value). The wrinkle the D-C pre-check surfaced.
7. **Coexistence with joint estimation + orthogonal projection:** the passivity constraint must not
   conflict with the interpretability (orthogonal-projection) constraint and must not break physical-
   parameter identifiability. (Argued to act on orthogonal subspaces -- must be SHOWN, not asserted.)

### C. Empirical validation (diagnostics)
8. **Trainability** -- constrained ANN trains (nf-RMS AND sim-RMS decrease; BPTT works). = D-D1 (parallel
   to D-C).
9. **No drift in free-run** -- trained augmented model's free-run bounded (confirms proof 2).
10. **Energy audit** -- `int F·v <= 0` measured on the trained free-run (confirms proof 1 is enforced).
11. **Friction capture** -- on injected-friction sim (then real data): passive ANN captures friction and
    BEATS bounded-integral (which structurally cannot). = D-D2.
12. **No regression** -- keeps the absorber/coupling, doesn't degrade parameter recovery, marginal pole
    stays at origin (eigen-check).

### Load-bearing (where the risk + contribution live)
- **Proofs 2 + 3** (bounded, marginal-preserving augmented interconnection, nonlinear LPV) = the thesis
  theorem, hardest, no off-the-shelf result; needs NI free-body adapted to our block.
- **Proof 1** (passive-by-construction parametrization) = the buildable part, from the neural-passivity
  literature (RENs/dissipative NODEs/KYP) but adapted to SEMIdefinite storage.
- **Empirical 9-11** (no drift + energy audit + friction capture) = the diagnostics backing the proofs.

**Status:** nothing in A/B/C is done for the passive constraint yet. Basis: classical passivity
(Willems/van der Schaft/Khalil, bedrock) + NI free-body (Mabrok 2014, verified) + neural-passivity
machinery (RENs etc., verified to force STRICT stability -> need the semidefinite adaptation). The
marginal-preserving passive block is our synthesis, not a citable result (§9).

### 5e.1 EMPIRICAL VALIDATION PLAN for the passive constraint (D-D1, D-D2) -- scope now

We focus on tier C (empirical) for now; proofs (A) and framework (B) deferred. Constraint SCOPED TO X
AND Y ONLY (the K=0 free integrators). Theta (spring kb1+kb2) and the absorber (spring ka) have
restoring forces -> bounded, no drift -> left UNCONSTRAINED (raw ANN output), to preserve expressivity.
(NB: D-C applied bounded-integral to ALL routed rows for simplicity; harmless there, but the correct
design and D-D1/D-D2 scope the constraint to X/Y.)

**Passive parametrization (the one open design choice, no off-the-shelf answer):** start simple and let
the data decide.
- pointwise viscous-dissipative `F = -softplus(net)*v` -> guarantees `F·v<=0` but likely TOO TIGHT (only
  a damper; may not keep the oscillatory absorber coupling). Informative even if it fails.
- integral-passivity / storage-function (RENs-style, semidefinite) = the right general one, research-grade.
Test the simple one first; if it cannot keep the absorber, that empirically motivates the harder version.

**D-D1 -- current sim (NO friction).** Passive-constrained ANN on X/Y: does it (8) train (nf-RMS AND
sim-RMS decrease), (9) NOT drift (flat slope), (12) keep the absorber (130-180 Hz band), (10) satisfy the
energy audit `int F·v <= 0`? Parallel to D-C. **LIMITATION (important): the current sim has NO friction,
so D-D1 CANNOT test the passive constraint's key advantage (friction capture). It only de-risks
trainability + no-drift + (possibly) recovery of the small dissipative M(Y) rectification that
bounded-integral forbids. D-D1 passing is NECESSARY but NOT SUFFICIENT.**

**D-D2 -- injected-friction sim (REQUIRED to know for sure).** The current sim cannot validate friction;
D-D2 is the decisive test and REQUIRES first building an injected-friction sim (inject a KNOWN nonlinear
friction, e.g. Coulomb + Stribeck, into the sim truth). Then compare on data that actually contains
friction:
- **bounded-integral** (current, validated) -> EXPECTED TO FAIL to capture the friction DC (forbids all
  DC) -> residual error. Showing this quantifies the gap and validates the critique with data.
- **passive-constrained ANN** -> should CAPTURE the friction (allows dissipative DC) and BEAT
  bounded-integral, without drift.
- optionally **Coulomb-in-`f_base`** -> captures the simple part only (grey-box baseline, §5d).

**Bottom line to remember: nothing about friction is validated on the current sim (it has none). To know
FOR SURE that the passive constraint captures real nonlinear friction without drift, we MUST build the
injected-friction sim and run D-D2. D-D1 alone (current sim) proves trainability + no-drift only.**

Enabling step (highest leverage): build the injected-friction sim first -- it is needed for D-D2
regardless of parametrization, and it lets us SHOW bounded-integral's friction gap and MEASURE the
passive/Coulomb gain in a controlled setting with a known `Fc`.

### 5e.2 CROSS-COUPLING -- HARD CONSTRAINTS on the passive-constraint diagnostics

The truth model (`Matlab-scripts/Augmentation/gantrySystemExtended.m`, verified port in `drift_common`)
has STRONG NONLINEAR CROSS-COUPLING: `M(Y, delta_a)` couples the absorber `delta_a` directly into Theta
and Y (mass entries `-ma*d`, `ma`) and into X indirectly (via the X-Theta term `(m1-m2)Lb/2 -
(mh+ma)Y - ma*delta_a`, then `M^{-1}` mixes all axes); `M` is NONLINEAR in Y (`mh*Y^2`, `(mh+ma)Y`) and
in `delta_a` (`ma*(Y+L0+delta_a)^2`). X and Y are K=0 (no stiffness). The absorber is a tuned mass
DAMPER -- net dissipative on the WHOLE system, but it TRANSFERS energy between axes (D-A energy proxy:
Y strongly dissipative ~-6.4e-2, but X can be energy-INJECTING ~+6e-4 on T10 -- energy transferred into
X from Theta/Y).

**HARD CONSTRAINTS the passive-constraint diagnostics (D-D1/D-D2) MUST satisfy:**
1. **The passivity constraint MUST be MIMO (whole X/Y output jointly), NOT per-axis.** A per-axis
   `F_X*v_X <= 0` forbids the legitimate cross-axis energy transfer INTO X -> the ANN cannot learn the
   cross-coupling. MIMO `int (F_X*v_X + F_Y*v_Y) <= storage` allows the transfer (net dissipative, since
   Y's dissipation dominates X's injection) while still forbidding the net drift.
2. **The passivity constraint MUST be INTEGRAL (`int F*v` bounded), NOT pointwise.** Pointwise
   `F*v <= 0` at every instant forbids the OSCILLATORY absorber coupling (which transiently injects
   energy). The integral notion allows the transient injection, net-dissipative over time.
3. **D-D1 MUST explicitly verify the absorber coupling is still LEARNED under the constraint** (130-180
   Hz band kept, cross-coupling captured). If the constraint kills the coupling, the formulation
   (per-axis vs MIMO, pointwise vs integral) or the whole approach needs rework. **This is the KEY RISK
   of the dissipative approach.**

**Caveat (honest):** this is reasoned from the D-A energy proxies (proxies -- they ignore the
mass-matrix cross terms) + the model structure, NOT proven. The true X+Y coupling APPEARS net-
dissipative (Y dominates), so MIMO integral passivity SHOULD be satisfiable by the true dynamics -- but
D-D1 is the test that decides it.

**For the bounded-integral BOUND (already validated, D-C):** the bound is a ZERO-DC constraint (bounded
running sum, applied per routed row), NOT an energy-sign constraint, so it does NOT have the per-axis
cross-coupling problem for the OSCILLATORY coupling: oscillatory cross-coupling is zero-DC -> allowed,
and **D-C empirically confirmed the bound keeps the absorber (130-180 Hz band unchanged)**. The bound
only forbids the small DC part of the cross-coupling (the `M(Y)` rectification) -> a bounded offset. So
the bound DOES learn the oscillatory cross-coupling; its only limitation is DC content (friction +
rectification) -- the known caveat, not a cross-coupling-learning failure.

## 5f. THE CONSTRAINT IS PASSIVITY (store + return), NOT PURE DISSIPATIVITY (only remove) -- CRITICAL

This is the single most important conceptual point for the passive-augmentation approach; getting it
wrong makes the method unable to represent the true system. Two DIFFERENT constraints, often conflated:

### Pure dissipativity (`int F·v <= 0`) -- TOO STRONG, do NOT use
"The learned force may only EXTRACT energy, never add it." This is a pure DAMPER. It **forbids energy
STORAGE** -- springs (`½ ka delta_a^2`) and masses (`½ ma v^2`). The hidden absorber is a mass-spring-
damper: it STORES energy in its spring and mass and returns it. So a pure-dissipativity constraint
**cannot represent the absorber (or any resonant / energy-storing dynamics)** -> it would FAIL to
capture the nonlinear system. Also forbids adding energy-storing augmented states.

### Passivity (`V(x) >= 0`, `dV/dt <= F·v`) -- THE CORRECT CONSTRAINT
There exists a stored-energy function `V(x) >= 0` (storage) with `dV/dt <= F·v`, equivalently
`int_0^T F·v dt >= V(T) - V(0) >= -V(0)`. Plain language: **the block may STORE energy and GIVE IT BACK
(springs, masses, augmented states), but may not CREATE net energy from nothing.** This ALLOWS the
absorber's energy-storing dynamics (its spring, mass, and the augmented states delta_a / v_delta_a),
because the real absorber is passive (stores + dissipates, net non-creating). It still FORBIDS the
drift: a sustained drift force PRODUCES net energy, which violates passivity.

### Why this is exactly what we need
| Behaviour | Pure dissipative (`int F·v<=0`) | **Passive (`dV/dt<=F·v`, V>=0)** |
|---|---|---|
| remove energy (damping, friction) | allowed | allowed |
| **STORE + return energy (spring, mass, aug states)** | **FORBIDDEN** -> can't do the absorber | **ALLOWED** -> captures the absorber |
| transfer energy between axes (cross-coupling) | (MIMO) allowed | (MIMO) allowed |
| **CREATE net energy (the drift)** | forbidden | **forbidden** |

So passivity keeps EVERYTHING a real passive system does (dissipation, storage, return, cross-axis
transfer) and blocks ONLY net energy creation = the drift. The target (nonlinear MSD + friction +
coupling) is a passive physical system -> it lives INSIDE the passivity set -> the constraint does NOT
"hit a wall" on the target, only on the drift. Pure dissipativity WOULD hit the wall (it walls off
storage). This is also why Negative-Imaginary / passivity-with-storage is the right frame (it natively
handles resonant / free-body / energy-storing dynamics); pure dissipativity is not.

### Correct statement (MIMO, use THIS, not "only remove energy")
> The power the learned block injects into the X/Y axes is `F_X·v_X + F_Y·v_Y`. The PASSIVITY constraint
> requires no NET energy CREATION: there is a stored-energy function `V(x) >= 0` with `dV/dt <= F·v`, i.e.
> `int (F_X·v_X + F_Y·v_Y) dt >= -V(0)`. The ANN may STORE and RETURN energy (springs, masses, augmented
> states -- e.g. the absorber), but not add it from nothing. (Pure dissipativity `int F·v <= 0` would be
> too strong -- it forbids energy storage and hence the absorber's spring/mass.)
>
> Nederlands: het geleerde blok mag energie OPSLAAN en TERUGGEVEN (veren, massa's, augmented states --
> zoals de absorber), maar geen NETTO energie uit het niets CREEREN (`dV/dt <= F·v`, `V >= 0`). Zuivere
> dissipatie (`int F·v <= 0`) zou te streng zijn: die verbiedt energieopslag en dus de absorber.

### Consequence for the design (carry into D-D1/D-D2)
- Implement **passivity with an explicit storage function** (allows augmented states / energy storage),
  NOT a pure `F·v <= 0` penalty.
- Combined with 5e.2: the constraint must be **MIMO (X/Y jointly)** and **INTEGRAL (storage-based)**.
- D-D1 must verify the absorber (energy-storing) IS still learned under the passivity constraint -- this
  is the direct check that we chose passivity, not pure dissipativity, and drew the wall correctly.

## 5g. PLAN OF APPROACH -- passivity-constrained augmentation (build + verify, staged)

Each phase states what to BUILD, how to VERIFY, and what GATES the next. Principle: verify cheaply and
in isolation before the pipeline; each gate must pass before the next phase.

**Phase 0 -- DONE (de-risking).** Diagnosis complete (drift = ANN DC on X/Y; input/IC/encoder/lr/nf
cleared). Bounded-integral block validated (D-C): the constrained-output TRAINING MECHANISM works
(trains, no drift, keeps absorber) -> de-risks "constrain the ANN and train it end-to-end".

**Phase 1 -- Build the passive block + unit-verify in ISOLATION** (before any pipeline).
- Build a block that is PASSIVE WITH AN EXPLICIT STORAGE FUNCTION `V(x) >= 0`, `dV/dt <= F·v`, MIMO on
  X/Y (per 5f + 5e.2). Start simplest: a neural port-Hamiltonian-style form (learn `H >= 0`, `R >= 0`,
  `J` skew -> passive by construction; storage = `H`; ALLOWS energy storage, not just damping).
- Verify (isolation, as for the bounded-integral block): (a) energy audit `int F·v >= -V(0)` for random
  inputs; (b) it can produce a STORED-ENERGY (spring-like) response, not only damping; (c) gradients flow.
- Gate: passive-by-construction + trainable, confirmed standalone.

**Phase 2 -- D-D1: trainability + no-drift + KEEPS ABSORBER** (current sim, no friction).
- Self-contained diagnostic (like `dC_...`): build pipeline, swap in the passive block, train short.
- Verify: (a) trains (nf-RMS AND full sim-RMS improve); (b) no drift (flat slope); (c) **KEEPS THE
  ABSORBER (130-180 Hz band)** -- the critical check that we chose PASSIVITY (not pure dissipativity) and
  drew the wall correctly; (d) energy audit holds on the free-run.
- Gate: passive block learns the absorber WITHOUT drift. **If it kills the absorber -> wrong formulation
  (per-axis / pointwise / pure-dissipative) -> rework Phase 1. LOAD-BEARING: this is where "will we hit
  the wall" is decided.**

**Phase 3 -- Build the injected-friction sim** (enabling infrastructure).
- Inject a KNOWN nonlinear friction (Coulomb + Stribeck) into the sim truth.
- Verify: friction present, `Fc` known, data reproducible.
- Gate: a controlled sim with known friction to test against.

**Phase 4 -- D-D2: friction capture** (injected-friction sim).
- Compare on the injected-friction data: bounded-integral (expected to FAIL on friction DC), passive
  (should capture it), Coulomb-in-`f_base` (captures the simple part).
- Verify: passive captures friction (lower held-out free-run error, no drift) and BEATS bounded-integral.
- Gate: passivity captures real nonlinear friction better than the bound, without drift. **The real-data
  value, proven in a controlled setting.**

**Phase 5 -- Framework integration** (deferred 5e-B items).
- Promote the passive block into `build_model` behind a flag; resolve: no algebraic loop + `M(Y)`
  invertible; encoder init of the passive block's STORAGE STATES; coexistence with joint estimation +
  orthogonal projection.
- Verify each (well-posedness; encoder reproduces; parameter recovery unharmed).
- Gate: integrated, well-posed, encoder-compatible.

**Phase 6 -- Real Telica data** (the deliverable).
- Fit the passivity-constrained augmented model to real closed-loop data.
- Verify: held-out free-run fidelity (BFR), no drift, energy audit; compare to bounded-integral / Coulomb
  on the SAME real data.
- Gate: best held-out free-run wins = the deliverable.

**Phase T -- Theory (parallel, for the thesis; 5e-A).** Proofs: passive-by-construction; bounded
augmented interconnection (NI free-body -> position boundedness); marginal-mode preservation;
friction-permitting -- for the nonlinear LPV case = the contribution.

### Order in one line
block passive in isolation (P1) -> **learns absorber, no drift (D-D1, P2)** -> injected-friction sim (P3)
-> **captures friction, beats the bound (D-D2, P4)** -> framework integration (P5) -> real data (P6);
theory in parallel (PT).

### The two make-or-break checks
1. **D-D1 (P2): does the passive block still learn the absorber?** If no, the constraint is mis-drawn
   (pure-dissipative / per-axis / pointwise) -> rework. This decides "will we hit the wall".
2. **D-D2 (P4): does passive beat bounded-integral on friction?** If no, passivity gives no real-data
   advantage -> fall back to grey-box (Coulomb + bound, 5d).

## 5h. INJECTED-FRICTION SIM -- which nonlinearity to inject (Phase 3 design + literature)

For D-D2 (Phase 3/4) we inject a KNOWN nonlinear friction into the sim truth, then test whether the
passive ANN captures it and beats bounded-integral. This section records how to inject it, which model,
the success metric, and the references (2026-07-10 targeted web search; abstracts/snippets, verify
before thesis citation).

### How to inject (mechanism, from `additional_state_lagrangian.m` + `gantrySystemExtended.m`)
The model is Lagrangian: `M q̈ + C q̇ + K q = f`, where `C` comes from the QUADRATIC Rayleigh
dissipation `D` (linear viscous damping only). **Nonsmooth/nonlinear friction does NOT fit the Rayleigh
form** -> add it as an ADDITIVE FORCE on the RHS:
`q̈ = M^{-1}( f - C q̇ - K q - F_fric(q̇, z) )`.
- **Static friction (Coulomb, Stribeck):** one force term `F_fric(q̇)` in `gantrySystemExtended.m`. Simple.
- **Dynamic friction (LuGre, GMS):** needs an INTERNAL bristle state `z` per axis -> extend the state
  vector, EXACTLY the pattern used for the absorber extra state in `additional_state_lagrangian.m`.
All are DISSIPATIVE -> inside the passivity set -> a fair test of the passive constraint.

### Friction-model realism hierarchy (literature)
Static -> dynamic, increasing nonlinearity/realism:
**Coulomb -> Coulomb+viscous -> Stribeck -> Dahl -> LuGre -> GMS / Leuven.**
- Coulomb: `Fc*sign(v)`. Too simple (grey-box captures it exactly -> doesn't test the ANN advantage).
- Stribeck: static nonlinear, velocity-dip. No extra state.
- **LuGre:** bristle-based DYNAMIC model, 6 params (sigma0 bristle stiffness, sigma1 bristle damping,
  sigma2 viscous, Fc, Fs static, vs Stribeck vel), internal state z (avg bristle deflection, nonlinear
  `|v|*z` dynamics). Captures presliding, Stribeck, stick-slip, varying breakaway force. Limitation: does
  NOT capture NON-LOCAL memory hysteresis.
- **GMS (Generalized Maxwell-Slip):** successor to the Leuven model; models presliding hysteresis WITH
  non-local memory -> the MOST nonlinear/realistic; more complex, passivity less cleanly established.

### DECISIVE finding for us: LuGre is a PROVEN PASSIVE operator (velocity -> force)
Literature: friction models should reflect friction's dissipative nature, which "translates into the
requirement of defining a PASSIVE operator from velocity to friction force -- necessary and sufficient
conditions for this property have been established for the LuGre model" (boundedness + internal-state +
input/output dissipativity). **So injecting LuGre gives a FAIR test of the passive ANN: the friction is
nonlinear, dynamic, realistic AND provably INSIDE the passivity set.** If the passive ANN cannot capture
LuGre friction, that is a real failure of OUR formulation (per-axis/pointwise/pure-dissipative), not the
target being outside the wall -- directly supports the approach and sharpens D-D1/D-D2.

### RECOMMENDATION
- **Inject LuGre** as primary: nonlinear + dynamic + realistic + PROVABLY PASSIVE + standard/defensible +
  implementable (bristle state per X/Y axis, same pattern as the absorber).
- **GMS** = max-realism follow-up (non-local hysteresis), harder, passivity less clean.
- **NOT Coulomb-only** (too simple).

### Success metric (literature-grounded)
Friction-ID validates via **free-run / trajectory fidelity + friction-force reconstruction**; LuGre
parameter ID is known to be HARD (the 2 dynamic params) -- which is why learned approaches exist. For
D-D2: **free-run fidelity of the passive-augmented model vs the LuGre-friction data**, benchmarked
against (a) bounded-integral (expected worse -- can't do friction DC), (b) Coulomb-in-`f_base` (simple
part only), (c) an ORACLE floor (true injected LuGre known). Success = passive reaches near the oracle
floor AND beats bounded-integral, with the energy audit holding and no drift.

### Relevant precedent: friction IS learned with NNs
"Learning Transferable Friction Models and LuGre Identification via Physics-Informed Neural Networks"
(arXiv:2504.12441) -- learned LuGre friction; direct precedent for the ANN capturing friction in D-D2.

### References (2026-07-10 search; abstracts/snippets -- VERIFY before thesis citation)
- GMS model (generalized Maxwell-slip; presliding hysteresis, non-local memory): Al-Bender, Lampaert,
  Swevers -- "The generalized Maxwell-slip model: a novel model for friction simulation and
  compensation", IEEE TAC 2005. https://www.academia.edu/21910650/
- LuGre passivity/dissipativity: "Stability and Dissipativity of the Distributed LuGre Friction Model"
  (ResearchGate 391957955) -- passive velocity->force operator, necessary+sufficient conditions.
- Presliding hysteresis, LuGre vs Maxwell-slip: ScienceDirect S0957415815001221.
- Learned friction (PINN LuGre): arXiv:2504.12441 (recent, verify).
- Bristle-dynamics friction models (lumped/distributed): arXiv:2602.09429 (recent, verify).
- Friction models survey + compensation: V. van Geffen, "A study of friction models and friction
  compensation", DCT 2009.118 (TU/e report). (COCC-hosted PDF.)
- Classical friction-model references (standard, for the thesis): Armstrong-Helouvry, Dupont, Canudas
  de Wit "A survey of models, analysis tools and compensation methods for the control of machines with
  friction", Automatica 1994; Canudas de Wit et al. "A new model for control of systems with friction",
  IEEE TAC 1995 (the original LuGre paper).

---

## 5i. CONCRETE PHASE-1 CONSTRUCTION — neural port-Hamiltonian passive MIMO port on X/Y (candidate route)

**Status: candidate route, discussed and specified 2026-07-10, NOT yet built.** This section pins down
one concrete, buildable realization of the passivity-with-storage block of §5f/§5g Phase 1, at the level
of detail needed to implement it. It is the leading candidate; alternatives (pointwise viscous damper
§5e.1; semidefinite-storage REN §5b step 3) remain on the table. Nothing here changes the diagnosis or
the plan of approach; it makes §5g Phase 1 explicit.

### The block — a neural port-Hamiltonian (PH) one-port, velocity-in / force-out, MIMO on X/Y
Internal storage state `ξ ∈ R^m` (the block's own absorber-like/bristle-like states; `m` small, `m ≥ 2`
since the truth absorber is 2 states). Continuous-time dynamics:

```
ξ̇ = ( J(ξ) − R(ξ) ) ∇H(ξ) + G · v      v = [v_X, v_Y]   (collocated X/Y velocity, INPUT)
F  = − Gᵀ ∇H(ξ)                          F = [F_X, F_Y]   (force applied to the X/Y rows, OUTPUT)
```

with the three structural constraints that make it passive **for every weight setting**:
- `H(ξ) ≥ 0` — storage function (learnable; quadratic `½ ξᵀ Q ξ`, `Q = L_Q L_Qᵀ ⪰ 0` for the linear
  default, or a neural nonnegative form — ICNN / `½‖N(ξ)‖²` — for the nonlinear version). `∇H` by autograd.
- `R(ξ) = L_R(ξ) L_R(ξ)ᵀ ⪰ 0` — dissipation (learnable Cholesky factor; state-dependent allowed).
- `J(ξ) = S(ξ) − S(ξ)ᵀ` — skew interconnection (learnable `S`; state-dependent allowed).
- `G` — input matrix `(m × 2)`, **full (not diagonal)** so both `v_X` and `v_Y` drive the shared `ξ`.

### Passivity-with-storage proof (explicit; holds for nonlinear H, R, J)
Using `Jᵀ = −J` (so `∇Hᵀ J ∇H = 0`) and `R ⪰ 0`:
```
dH/dt = ∇Hᵀ ξ̇ = ∇Hᵀ(J − R)∇H + ∇Hᵀ G v = − ∇Hᵀ R ∇H + ∇Hᵀ G v
F · v = (−Gᵀ∇H)ᵀ v = − ∇Hᵀ G v            ⇒   ∇Hᵀ G v = − F·v
⇒  dH/dt = − ∇Hᵀ R ∇H − F·v               ⇒   F·v = − dH/dt − ∇Hᵀ R ∇H ≤ − dH/dt
⇒  ∫₀ᵀ F·v dt ≤ H(0) − H(T) ≤ H(0)        (H ≥ 0)
```
So the block can inject **at most its initially stored energy** `H(0)`, and with the block reset to rest
each rollout (`ξ(0) = 0`, `H(0) = 0`) this is `∫ F·v dt ≤ 0`. It CANNOT create net energy (forbids the
drift force), but it CAN store energy in `ξ` and return it (spring/absorber), and `R` dissipates
(friction). This is passivity-with-storage (§5f), NOT pointwise `F·v ≤ 0` (which would forbid the
absorber's storage). Only three pointwise facts are used (`J` skew, `R ⪰ 0`, `H ≥ 0`), so the guarantee
survives arbitrary nonlinear `H(ξ), R(ξ), J(ξ)` — the guarantee is **knowledge-free** (independent of the
true dynamics), which is exactly §5 selection-criterion 1, the property needed on the unknown real system.

### The off-diagonal coupling is captured in the velocity domain (no position input needed)
Both `F_X` and `F_Y` are read out of the **shared** internal state `ξ`, and `ξ` is driven by **both**
`v_X` and `v_Y` through the full `G`. Hence `F_X` depends on `v_Y` and `F_Y` depends on `v_X`: the
**off-diagonal (cross-axis) coupling is present by construction**, entirely from velocities and `ξ`, with
no machine-position input. This is why the constraint must be **MIMO** (combined X+Y port), not per-axis
(§5e.2): a per-axis block would break this coupling. The combined-port passivity `∫(F_X v_X + F_Y v_Y) dt
≤ H(0)` allows the transient cross-axis energy transfer (Y-dissipative dominates, X can transiently
inject — D-A energy proxy) while forbidding the net drift.

### Input / output selection — the THREE distinct "Y" roles (do not conflate)
Verified pipeline wiring (`gantry_dynamic/model.py`): the ANN block currently reads the FULL propagated
state `x` plus `u` (`connect_block_signals(ann_block, ["x","u"], [])`, `nz = nxd + nu`) and its output is
added to the routed rows (`connect_signals(ann_block, "xp", "additive", expansion_matrix(route_ix,
nxd))`). The velocity components of that state are ENCODER-RECONSTRUCTED from a past-position window (a
generalized-differentiation map), not directly measured — we only measure positions.

| # | "Y" role | Index | Decision | Reason |
|---|---|---|---|---|
| 1 | ANN force OUTPUT on the Y row | idx 5 (`dY` row) | **KEPT — non-negotiable** | D-103: X/Y must be routed |
| 2 | ANN reads Y VELOCITY as input; off-diagonal coupling | idx 5 (`dY` state) | **KEPT** | captures the cross-axis coupling via shared `ξ` + full `G` |
| 3 | ANN reads Y/X POSITION as input to the force path | idx 0, 2 (`X`,`Y` states) | **EXCLUDED (Phase 1/2)** | position→force = stiffness on the free integrator → pole leaves origin |

So the passive port's INPUT is `[v_X, v_Y]` (idx 3, 5) plus internal `ξ`; the position states (idx 0, 2)
are NOT fed into the X/Y force path. This is a selection on the block input; the current block happens to
receive everything. Θ and the absorber rows (if handled by a parallel raw `Static_ANN`) are unconstrained
(§5e.1) and MAY read position (Θ has a spring, no marginal mode to protect).

### Two DIFFERENT guarantees — do not conflate (acceptance A2 vs A3)
- **Passivity / no-drift (A1, A2).** From `∫F·v ≤ H(0)`. Forbids the energy-injecting sustained-DC drift
  force. On our X/Y (mass-DAMPERS, finite τ_X ≈ 1.55 s, τ_Y ≈ 1.01 s, per d2), block-passivity + the
  axis's own velocity damping gives `∫ c v² dt` bounded → `v ∈ L²` → `v → 0` → **position bounded**. (For
  the idealized c=0 pure double integrator, position-boundedness instead needs the Negative-Imaginary /
  free-body argument, Mabrok 2014 — the theory-phase route.) Passivity would TOLERATE a position input.
- **Marginal-mode preservation (A3).** The X/Y position pole stays exactly at the origin iff the force has
  NO dependence on the position state (`∂F/∂position = 0`, no added stiffness). This is a property of the
  propagated dynamics Jacobian, independent of how `x0` was reconstructed. It is what specifically
  requires excluding position from the force-path input (role #3 above). Distinct from passivity.

### Parametrization ladder (start simple, build nonlinear-capable)
- **Default (first D-D1):** linear/quadratic — `H = ½ξᵀQξ` (`Q ⪰ 0`), constant `R`, `J`, `G`. A passive
  LTI one-port; represents the linear absorber resonance (130–180 Hz), the make-or-break for D-D1. If a
  linear passive port cannot keep the absorber, that points to the FORMULATION (per-axis/pointwise/pure-
  dissipative), not expressivity, since the absorber is a linear resonance.
- **Nonlinear (Phase 4 friction, real data):** neural `H(ξ) ≥ 0`, state-dependent `R(ξ)`, `J(ξ)`. The
  passivity proof is unchanged. This is structurally the LuGre class — LuGre is a proven passive
  velocity→force operator with an internal bristle state that maps onto `ξ` (§5h) — so the nonlinear PH
  block is the right object to capture the injected friction in D-D2 and the unknown coupled real system.
- **Build the block nonlinear-capable from the start** (so `H, R, J` CAN be nonlinear), default the config
  to the simple form; escalate the config, not the code, when D-D1/friction demand it. No rewrite.

### Deferred refinement — Y-position as an LPV scheduling input (optional, stiffness-free form)
The coupling STRENGTH varies with operating Y (`M(Y)` scheduling; T9–T11 move Y). A Y-independent port has
fixed/`ξ`-dependent coupling gains. If D-D1 shows the absorber/coupling is under-captured because its
strength varies with Y across the trajectory, add Y-position as a scheduling input **into the cross-terms
only**, enforcing `∂F_Y/∂Y = 0` (and `∂F_X/∂X = 0`) structurally so it modulates the coupling WITHOUT
adding a self-axis stiffness that would move the free pole. This is a testable, staged decision surfaced
by the "keeps the absorber" gate, not a removal of anything. Do NOT feed Y-position into the diagonal
(self-axis) force path.

### Discrete-time realization + the one caveat to verify
`ξ` is carried inside the block across timesteps within a rollout and stepped by the block itself (drop-in
for `Static_ANN_Block`, mirroring how `bounded_integral_block.py` carries `psi_prev`); `reset()` zeros `ξ`
at the start of every window/sim; `ξ` is a live tensor (BPTT-connected), detached only at reset; auto-
reset on batch-size change. Framework integration of `ξ` as genuine LFR states (encoder-initialized) is
Phase 5, not Phase 1. **Caveat:** continuous PH is exactly passive; explicit-Euler stepping of `ξ` incurs
an `O(Δt²)` passivity defect. Phase 1 uses explicit Euler and the energy audit QUANTIFIES the defect at
the pipeline Δt (4 kHz / up_sample); the exact-discrete route (discrete-gradient / implicit-midpoint) is
the theory-phase (PT) upgrade for an exact discrete-time theorem.

### Phase-1 isolation unit tests (the gate; mirror the bounded-integral block verification)
1. **Energy audit** — random `v(t)` + random init: assert `Σ F·v Δt ≤ H(0) − H(T) + tol`; with `ξ=0`
   reset assert `≤ 0`; report the discrete defect magnitude.
2. **Storage / spring** — velocity burst then `v=0`: block still exerts (ringing) force as `ξ` oscillates,
   `H(ξ)` rises then falls → proves STORAGE (not a memoryless damper). The direct check that we chose
   passivity, not pure dissipativity.
3. **Gradients flow** — backprop a dummy loss to `L_Q, L_R, S, G`; finite, nonzero.
4. **MIMO cross-coupling** — nonzero off-diagonal `G/J` produces an `F_X` response to `v_Y`.
5. **Passive-for-all-weights** — many random weight sets (linear AND nonlinear configs); audit holds for
   every one (the "by construction for all parameters" claim, acceptance A1).
6. **Reset / BPTT hygiene** — `ξ` cleared on reset, batch-size change handled, `ξ` live (BPTT-connected).

### What Phase 1 does and does NOT prove (honest scope)
- Proves: passive-by-construction (A1), storage-expressivity + MIMO coupling (part of A4), trainable
  (gradients). Does NOT prove: position-boundedness / NI (A2 theorem — theory phase), marginal-mode
  preservation and "keeps the absorber" (the empirical D-D1 gate, Phase 2), friction capture (D-D2,
  Phase 4). Maps to acceptance checklist §5e: Phase 1 = A1 + the buildable part; A2/A3/B/C-9..12 follow.

### Relationship to the (validated) bounded-integral block
Bounded-integral (D-C, validated) enforces a ZERO-DC (bounded running sum) constraint per routed row —
forbids ALL DC → cannot represent friction (the sim/drift-half solution). The PH passive port GENERALIZES
it to a single energy constraint that additionally PERMITS dissipative DC (friction) via `R` and the
internal state — the real-data deliverable. Both keep the oscillatory absorber coupling; the PH block adds
the ability to represent the dissipative, nonzero-at-rest residual that bounded-integral refuses.

---

## 5j. PASSIVITY BOUNDS VELOCITY, NOT POSITION -- the stored-energy finding + the Negative-Imaginary decision

**Status: measured + decided 2026-07-10.** Phase-1 Step-2/3.5 tested the PH block of 5i in isolation.
The energy audit PASSED (block is passive by construction; implicit-midpoint integrator makes it EXACTLY
discrete-passive, `max_r ~ 4e-17`). But the long-horizon drift probe surfaced a REAL limit that decides
the theory route. This section records the finding, the online literature check, and the decision.

### The empirical finding (Step-3.5 drift probe, `p1_drift_probe.py`)
Closed the PH block onto a toy free-integrator X/Y axis (exact `m,c` from `drift_common`: tau_X=1.55 s,
tau_Y=1.01 s), drove it 12 s, measured position-envelope growth (RMS|q| 4th quarter / 3rd quarter).
- **passive block from REST (xi(0)=0, the operating case -- pipeline resets xi each rollout):** bounded,
  env_ratio ~ 1.00, max|q|=5e-5 m, cumulative int F.v <= 0. NO drift. This is the case that matters.
- **energy-injecting DC control (falsifiable control):** env_ratio ~ 1.48, max|q|=4.3e-3 growing -> drifts,
  as it must (control valid).
- **passive block seeded WITH stored energy (xi(0)!=0, H(0)=2e-3, a STRESS test):** bounded max|q|=2.2e-4
  and passivity held (int F.v = 2.3e-7 <= H(0)), BUT the position ENVELOPE GREW over 12 s (env_ratio 1.62).

The stored-energy growth is NOT a metric artifact and NOT dismissable. It is the core theoretical limit.

### The theory: passivity bounds VELOCITY (L2), position can still grow O(sqrt(T))
Passivity gives `int F.v <= H(0)`. With axis damping `c>0`, the energy balance gives
`c * int_0^inf v^2 dt <= const + H(0) < inf`, so `v in L2`. But by Cauchy-Schwarz,
```
|q(T) - q(0)| = |int_0^T v dt| <= sqrt(T) * sqrt(int_0^T v^2 dt) <= sqrt(T) * sqrt(H(0)+const)
```
so passivity ALONE permits position to grow like **O(sqrt(T))** -- sub-linear, but unbounded as T->inf.
It is NOT a bounded-position guarantee. Equivalently (momentum balance `m v(T) + c q(T) = const +
int F_ext + int F`): position is bounded IFF the block's **net impulse `int F dt`** is bounded, which
passivity does not force (a bounded-energy near-DC force is invisible to `int F.v` at low velocity yet
walks the integrator). Truly bounded position needs `v in L1` (summable decay), which needs a STRICTER
structural property than passivity. **This is the A2 gap (5e) made empirical: passivity bounds velocity/
kinetic energy, not position on a free integrator.**

### Heuristic REJECTED
Proposed fix `R = L_R L_R^T + eps*I` (internal damping floor) was REJECTED (user): it is an arbitrary
constant (HEURISTIC) that masks the offending internal mode for tuned magnitudes but gives NO bounded-
position guarantee and does not transfer. A structural guarantee gap is fixed structurally, not with a
numerical regularizer. [[lessons: structural-guarantee-not-heuristic]]

### The correct structural property = bounded net impulse == Negative-Imaginary (free body)
The drift-causing quantity is the net impulse `int F dt`. Bounding it structurally bounds the position.
Two routes:

**Route A -- Negative-Imaginary (NI) constrained block (the formal certificate; the contribution).**
NI is the **force->position** class of passive mechanical systems (PR is force->velocity; integrating once
rotates PR into NI). NI bounds the position of a system with a **pole at the origin** via a DC-gain/
residue condition, NOT a positive-definite-in-position Lyapunov function (which would be stiffness) -> it
**preserves the marginal pole** (unlike contraction/RENs/Gyorok which force ||A||<1 and damp it away).
- **References (online-verified 2026-07-10):**
  - **Mabrok et al. 2014, arXiv:1305.1079** ("Generalizing Negative Imaginary Systems Theory to Include
    Free Body Dynamics", IEEE TAC 59(10):2692-2707) -- THE free-body result: NI plant with poles at
    origin in positive feedback with a strictly-NI controller; nec+suf stability via Laurent-expansion
    residues (`G2 = lim s^2 G(s)` rigid-body residue Hermitian PSD; `G1`; `G0`), allowing the map norm
    `=1` at DC on the free-body subspace = the marginal-preserving property. **LINEAR/LTI only.**
    CORRECTION: our earlier notes mis-attributed 1305.1079 to "Lanzon-Petersen"; it is **Mabrok et al.**
    Lanzon & Petersen 2008 is the separate ORIGINAL NI paper.
  - **Xiong, Petersen, Lanzon 2010** -- the NI Lemma (state-space LMI characterization; the parametrization
    handle).
  - **Ghallab-Petersen; Shi-Vladimirov-Petersen** -- nonlinear NI storage-function definition
    (`Vdot <= u^T ydot`, V positive SEMIdefinite) -- the only definition allowing a flat storage direction
    (pole at origin).
  - **NINODE, arXiv:2504.19497 (Apr 2025)** -- "Negative Imaginary Neural ODEs", the closest neural NI
    construction (Hamiltonian framework). BUT it is a CONTROLLER for an NI plant (not a forward-model
    augmentation) and its Assumption 3 re-imposes a strict DC-gain condition `!=1`, so as published it
    **abandons the marginal case we need.**
- **The gap = the contribution [SHARPENED 2026-07-10 after primary read -- see §5L verification addendum]:**
  ~~NI free-body theory is LINEAR-only~~. **CORRECTION: the nonlinear + free-body + semidefinite-storage NI
  theory is NOT unworked -- it exists analytically** (Shi-Petersen-Vladimirov arXiv:2011.14610 Def 1: nonlinear
  NI via positive-SEMIdefinite storage `V̇ ≤ uᵀẏ̃`, poles at origin included; Ghallab-Petersen arXiv:2201.00144).
  What IS unworked, and remains the contribution, is the **LEARNED / neural + FORWARD-augmentation (+ LPV
  `M(Y)`) combination**: every *neural* NI/passive construction (NINODE, PLNet, passive RENs/NodeRENs,
  Learning-Stable-and-Passive-NODEs 2404.12554) forces strict PD storage / strict DC gain and excludes the
  marginal mode. So the thesis theory contribution is **bringing the existing analytical nonlinear-NI
  free-body semidefinite-storage theory into a learned parallel forward augmentation in the LPV-LFR/SUBNET
  framework** -- NOT inventing the NI theory from scratch. This is a NARROWER, better-founded claim.
- **HEAVY.** Making NI rigorous for our case is a months-scale derivation (Phase T), not a Phase-1 test.
  No NI primary PDF is on disk; grounding it means fetching + verifying Mabrok 2014 and the NI Lemma.

**Route B -- bounded net-impulse / exact-derivative output (the constructive guarantee; buildable now).**
Constrain the block's FORCE output so its running sum telescopes (`int F dt` bounded by construction, all
weights) -- the exact-derivative / bounded-integral form -- while keeping the PH storage `H(xi)` for the
absorber. Knowledge-free; its core mechanism is ALREADY validated (D-C, bounded-integral block: no drift,
keeps the 130-180 Hz absorber). For our damped axes (`c>0`), bounded impulse GENUINELY bounds position --
a real guarantee, not a hack. The deep-research doc (`literature/stability-training/`) independently calls
this "the highest-leverage novel building block". Known limit: it forbids a SUSTAINED DC force, so real
friction DC needs a separate dissipative channel or `f_base` (Phase 4+; sim has no friction).

### DECISION (2026-07-10)
**Do NOT add NI to the block now.** NI is HEAVY (linear-only theory needing a nonlinear/semidefinite/
neural extension; NINODE is a controller with strict DC gain, not our case). It is the **parallel theory
contribution (Phase T)**, now correctly grounded in Mabrok 2014. For the block we BUILD and TEST now, use
the **Route-B bounded net-impulse structural guarantee** (bounds position on our damped axes, keeps the
absorber, knowledge-free, validated mechanism). The two are complementary: **Route B = constructive
guarantee (now); NI = elegant single certificate (contribution, parallel).** Next: fold a bounded-impulse
constraint into the PH block output and re-run the stored-energy probe to confirm it now bounds position.

### Honest provenance
Mabrok 2014 (= 1305.1079) and NINODE (2504.19497) online-verified to exist and to match the descriptions
above (abstract/summary level; the exact LMI/residue conditions are NOT yet read at the primary source).
The nonlinear-free-body-semidefinite NI construction does not exist in the literature (genuine gap). The
Cauchy-Schwarz `O(sqrt(T))` bound and the bounded-impulse->bounded-position argument (for `c>0`) are
re-derived here and hold; the `c=0` idealization still needs NI.

---

## 5k0. HARD CONSTRAINT -- DO NOT restrict the ANN INPUT to achieve a structural property

**Status: HARD CONSTRAINT, user directive 2026-07-10. Non-negotiable, same tier as D-103.**

> **The augmentation ANN MUST keep the FULL-state input `["x","u"]`. It is FORBIDDEN to buy a structural
> guarantee (marginal-mode preservation, no-drift, no-added-stiffness) by RESTRICTING WHAT THE BLOCK
> READS -- e.g. velocity-only input, dropping X/Y position, or input differencing. Any required property
> must be enforced on the block's OUTPUT, with the input left full.**

**Why it is forbidden (two reasons):**
1. **It is the velocity-domain LAST RESORT in disguise.** Amputating position from the input throws away
   information exactly as fitting in the velocity/acceleration domain does (the supervisor-forbidden
   fix C, see top-of-doc standing constraint). The user flagged this directly: "This is exactly the last
   resort we don't want to do."
2. **It is unnecessary AND it destroys expressivity.** Dropping X/Y position kills LPV **Y-scheduling**
   (Y-position IS the `M(Y)` scheduling variable) and position-dependent effects (cogging). And it is not
   needed: bounded-integral READS THE FULL STATE INCLUDING POSITION and still bounds drift via its output.

**ALLOWED alternative solutions (enforce the property on the OUTPUT, keep full input) -- for reference:**
| # | Mechanism | Keeps full input? | Gives | Status |
|---|---|---|---|---|
| S1 | **Bounded-integral / exact-derivative output** `g_k = psi(z_k)-psi(z_{k-1})`, `psi` reads full `z` incl. position -> telescoping bounds accumulated force | yes | no-drift | VALIDATED (D-C) |
| S2 | **Passivity/dissipativity constraint on the output power** `int F.v <= storage` (holds for ANY input the block reads) | yes | no energy-injection | 5i/5j (bounds velocity, not position alone) |
| S3 | **Structural constraint on `dF/dq_free`** (net-impulse / q-Jacobian on the free-axis coordinate driven to zero) -- block still READS `q`, but the OUTPUT carries no net stiffness on the free pole | yes | marginal-mode preservation | design option |
| S4 | **NI constraint on the output map** (Route A) | yes | bounded position + marginal | theory (5j) |
**Rule of thumb:** state every marginal/no-drift requirement as a property of `F(.)` (its net impulse, its
`q`-Jacobian, its power), never as an input selection. Verify empirically (12 s position probe, B1/B3).

### Clarification (user, 2026-07-10): the constraint STEERS the learning, it is not a blunt output bound
"Enforce on the output" does NOT mean "cap the output magnitude". The right frame is **STEERING the
learning in a targeted, knowable subspace/region** -- exactly how **orthogonal projection** works: it does
not bound magnitude, it penalizes the ANN's component IN THE KNOWN FP-model subspace so the ANN learns
only the orthogonal residual (keeps physics interpretable, R4). Two complementary mechanism types, BOTH
used:
- **(a) HARD architectural** (by construction, all weights): passive-PH, bounded-integral -> makes the bad
  output impossible.
- **(b) SOFT targeted steering** (a regularizer/projection on training): orthogonality; and the drift
  constraint framed as **dissipativity steering** -> push the output into the energy-removing half-space
  (target known from the SIGN of `F.v`), away from the drift direction, leaving friction/coupling FREE.

**Knowledge-free line (A1):** the steering TARGET must be knowable WITHOUT the true dynamics. Physics
subspace (orthogonality) and power sign (dissipativity) qualify; the residual MEAN does NOT -> that is why
the DC/mean-force penalty is sim-only (it would suppress real friction, 5). **Same layer:** the drift
steering and the orthogonal-projection steering live in ONE regularization/projection layer (the
LPV/MIMO/LFR extension of orthogonal projection = the thesis interpretability contribution) and MUST
COEXIST (C5). Frame the drift constraint as a SIBLING steering there, not a separate cap.
[[lessons: steer-learning-targeted-subspace]]

---

## 5k. REQUIREMENTS CHECKLIST -- every candidate augmentation block MUST satisfy ALL of these

**Status: living checklist, established 2026-07-10 (user-directed).** This is the acceptance contract for
the drift-fix / augmentation block, independent of WHICH construction (bounded-integral, dissipative PH,
NI, grey-box) is chosen. A candidate is only acceptable if it meets EVERY item, or the failure is
explicitly justified and accepted. "Shown" = demonstrated by test/derivation, NOT asserted (we over-
claimed position-boundedness once from passivity and it failed the 12 s probe -- see 5j). Where an item is
an EMPIRICAL gate, the long-horizon probe (>=12 s, >>50x the nf window), not the energy audit, is the
judge.

### A. Expressivity -- must NOT wall off real dynamics (the "don't disallow" requirements)
- [ ] **A1. Works for NONLINEAR real data with UNKNOWN dynamics.** The guarantee must be KNOWLEDGE-FREE
      (a property of the parametrization, holding for all weights, not requiring knowledge of the true
      residual). A guarantee that assumes the residual is zero-mean / known does NOT qualify (that is why
      the mean/DC penalty is sim-only, 5). Must hold with nonlinear H,R,J / nonlinear net, not just LTI.
- [ ] **A2. Does NOT disallow energy-STORING dynamics (the absorber).** The block MUST be able to augment a
      mass-spring-damper (store energy in a spring/mass and return it): passivity-WITH-STORAGE, NOT pure
      dissipativity `int F.v<=0` (which forbids the spring). Direct check: keeps the 130-180 Hz absorber
      band. This is the make-or-break (5f).
- [ ] **A3. Captures FRICTION: nonzero-at-rest, velocity-dependent DISSIPATIVE force** (Coulomb/Stribeck/
      LuGre; `F(0)!=0` allowed, `F.v<=0`). The block must NOT be forced to zero force at equilibrium
      (kills Coulomb) NOR forbid all DC (bounded-integral's gap -> forbids friction). This is exactly the
      "middle condition" (9): permit velocity-dependent force, forbid constant-at-rest force.
- [ ] **A4. Captures MIMO cross-axis COUPLING (X<->Y).** Constraint acts on the WHOLE X/Y port jointly, NOT
      per-axis (per-axis forbids legitimate cross-axis energy transfer -> can't learn the coupling, 5e.2).

### B. Stability / no-drift -- must bound the free-run (the "must forbid" requirements)
- [ ] **B1. No drift: BOUNDED POSITION over the full free-run** (>=12 s, >>50x nf), on X and Y. EMPIRICAL
      GATE = long-horizon position-envelope flat (env_ratio ~ 1), NOT the energy audit (passivity bounds
      velocity, not position -- 5j). Must hold from rest AND (stress) with stored energy.
- [ ] **B2. MARGINAL-MODE PRESERVATION: X/Y position pole stays exactly at the origin.** The block adds NO
      net stiffness/damping to the axis position row. This is a property of the OUTPUT `F(.)` (its
      dependence on / net impulse over the free-axis coordinate), enforced by the output PARAMETRIZATION --
      NOT by restricting the input. **KEEP THE FULL-STATE INPUT `["x","u"]`**; do NOT drop X/Y position or
      go velocity-only (that cripples LPV Y-scheduling and echoes the velocity-domain last resort;
      bounded-integral reads the full state INCLUDING position and still bounds drift via its telescoping
      output). [[lessons: constrain-output-not-input]] Contrast: contraction (RENs/Gyorok, ||A||<1) FAILS
      this -- it damps the free pole away. Check: linearized augmented model keeps the X/Y eigenvalue at 0.
- [ ] **B3. Guarantee holds in DISCRETE time at the pipeline dt**, not only continuous. (Explicit Euler
      injected O(dt) energy and FAILED the audit at 4 kHz; implicit-midpoint/discrete-gradient fixed it.
      Any block with internal dynamics must show its discrete step preserves the property -- 5j / Step 2.)

### C. Framework compatibility (LPV-LFR + SUBNET) -- must not break the pipeline
- [ ] **C1. Keeps X/Y ROUTING (D-103).** The ANN force output on X and Y stays; the fix must NOT be to drop
      X/Y from the routing (Theta-only is not acceptable). [[project_xy_routing_constraint]]
- [ ] **C2. LFR well-posed: no ALGEBRAIC LOOP** (block output depends only on states/inputs, not
      instantaneously on its own output -> acyclic signal graph).
- [ ] **C3. `M(Y)` INVERTIBILITY unaffected** (the block is an RHS force / added state, NOT a term in the
      mass matrix -- same argument as Coulomb, m-matrix-invertibility; verify it holds).
- [ ] **C4. ENCODER-compatible: any added storage/internal states are well-defined-ly INITIALIZED** by the
      SUBNET encoder (or init to a known rest value). New states change nx -> encoder-init must cover them.
- [ ] **C5. COEXISTS with joint estimation + ORTHOGONAL PROJECTION** (interpretability): the constraint
      must not conflict with the parameter-orthogonality regularizer nor break physical-parameter
      identifiability. Must be SHOWN (argued to act on orthogonal subspaces), not assumed.

### D. Trainability / practicality
- [ ] **D1. TRAINABLE end-to-end via BPTT**: nf-RMS AND full-12 s sim-RMS both decrease; gradients flow.
- [ ] **D2. DIFFERENTIABLE rollout**: any nonsmoothness (friction `sign(v)`, set-valued maps, implicit
      solves) handled so BPTT is stable (smoothed `tanh(v/eps)` or a differentiable implicit step).
- [ ] **D3. VERIFIED, not asserted**: every guarantee above is demonstrated by a test or derivation on the
      actual object (the long-horizon probe / energy audit / eigen-check / friction-capture D-D2), with
      the falsifiable control well-behaved. No property is claimed from a proxy or from the continuous-time
      idealization alone.

### How the candidates score against this checklist (current understanding, 2026-07-10)
| Item | Bounded-integral (B, validated) | Dissipative PH + friction channel | NI (Route A, unbuilt) | grey-box (Coulomb in f_base + bounded-int, 5d) |
|---|---|---|---|---|
| A1 nonlinear/knowledge-free | yes | yes | yes | yes |
| A2 absorber (storage) | yes (zero-DC osc.) | yes (H storage) | yes | yes (bounded-int part) |
| **A3 friction (nonzero-at-rest DC)** | **NO (forbids all DC)** | yes (monotone D(v)) | yes | yes (Coulomb in f_base) |
| A4 MIMO coupling | yes | yes | yes | yes |
| B1 bounded position 12 s | yes (validated D-C) | **must be SHOWN (5j)** | yes (by NI theorem) | yes |
| B2 marginal pole | yes | yes (velocity-only in) | yes (native) | yes |
| B3 discrete-time | yes (telescoping exact) | **needs midpoint/disc-grad** | to derive | yes |
| C1-C5 framework | drop-in; C4 wrinkle (psi_prev) | new states -> C4 heavier | heaviest | moderate |
| D trainable | validated | risk: nonsmooth friction | hardest | established |
**Reading:** bounded-integral fails A3 (the unacceptable friction gap). Dissipative PH + friction channel
meets A1-A4 and B2 but B1/B3 are NOT yet shown (the open work). NI meets everything on paper but is
unbuilt/linear-only (heavy). Grey-box meets all but splits friction into f_base (pragmatic fallback, 5d).

---

## 5L. TARGETED LITERATURE SYNTHESIS (2026-07-10) -- we reuse 4 of 5 pieces; only marginal-stability is new

> **Detailed literature catalog with direct quotes + full references:
> `docs/passivity-augmentation-literature.md`** (provenance-tagged: on-disk reads vs online extractions to
> verify). This section is the design-level synthesis; that file is the reference catalog.

**Status: depth-first read of 4 on-disk artifacts end-to-end (user-directed "don't reinvent the wheel").**
Bottom line: the augmentation we need is mostly ASSEMBLY of existing, in-framework pieces. Storage,
friction, interpretability-steering, and well-posedness all EXIST and are reusable; only the
**marginal-mode-preserving stability** (relax contraction to `||A||<=1` on the rigid-body subspace) is
genuinely missing = the thesis contribution. This is the good news: we are not inventing 5 things, we are
inventing 1 and reusing 4.

### Artifacts read (provenance)
- **[STORAGE]** `Extra dynamische toestand(en) ... H-type dual-drive gantry.pdf` -- AI lit study (verify
  primaries before citing). Candidate hidden states for OUR gantry.
- **[FRICTION]** `Literatuuronderzoek voor Dahl-frictiestaten ... H-type dual-drive gantry stage.pdf` --
  AI lit study (verify primaries). Friction-state templates + gantry friction PARAMETERS.
- **[STEERING]** Gyorok, Hoekstra, Kon, Peni, Schoukens, Toth, **"Orthogonal projection-based
  regularization for efficient model augmentation", L4DC 2025** (arXiv:2501.05842) -- PRIMARY, read full.
- **[GUARANTEE]** Gyorok, Drenth, Verhoek, Peni, Schoukens, Toth, **"Data-driven augmentation ... under
  constraint-free well-posedness and stability guarantees", 2026** (arXiv:2604.11421) -- PRIMARY, read full.

### The consolidated recipe (copy-plus-delta): each piece = donor + our delta
| Piece | Donor recipe (copy) | Our delta (for the gantry augmentation) |
|---|---|---|
| **STORAGE = absorber** | 2nd-order modal state `xi_dd + 2 zeta_xi omega_xi xi_d + omega_xi^2 xi = b(Y) F` (support mode ~37.7 Hz), OR 1st-order viscoelastic/Zener `z_d = -(1/tau(Y)) z + k(Y) l_d(q)`, `F = k0(Y) l(q) + z` [STORAGE study] | our hidden mode is a 2nd-order MSD at **150 Hz** (`fa`, `zeta_a=0.05`); realize as the PH storage `H(xi)` with skew `J` (5i). Y-dependent gains = LPV scheduling (into cross-terms only, no self-stiffness, 5i). |
| **FRICTION = nonzero-at-rest dissipative** | Dahl/LuGre bristle state `z_d = v - sigma |v|/Fc * z`, `F = sigma z (+ sigma2 v)`; nonzero-at-rest via `z`; EP (elasto-plastic, Hayward 2009) if it drifts vs static friction; discrete-time OK at 20 kHz [FRICTION study] | this is the **monotone friction channel `D(v)`** of the dissipative block; `Fc(Y)` Y-dependent (off-centre payload -> rail normal-force scaling). LuGre is a PROVEN passive v->F operator -> fits the passivity frame. Smooth `tanh(v/eps)` for BPTT (D2). |
| **FRICTION params (Phase-3 injection!)** | **Gantry stage (Lan Jia 2023 MSc Delft, Prodrive):** `Fs ~ 1.9-3.2 N`, `Fc ~ 1.4-2.75 N` (position-dependent), viscous `sigma2 ~ 30-33`, Stribeck `v0 ~ 1e-4 m/s`, pre-sliding ~tens of um; `sigma0,sigma1` position-dependent. THK ball guide: `Fc=10, Fs=12 N`. High-prec actuator: `Fc~400 mN, zss~100 nm` [FRICTION study] | **directly the D-D2 injected-LuGre parameter set** (5h Phase 3). Verify the Lan-Jia numbers at the primary MSc before thesis use. |
| **STEERING = interpretability** | Orthogonal projection [STEERING L4DC 2025]: FP-Jacobian regressor `Phi=partial f_theta/partial theta`, reduced SVD `Phi=Q Sigma V^T`, projection `Pi=Q Q^T`, add `beta ||Q^T f_ANN(X,U)||^2` to the cost -> penalize ONLY ANN directions inside the FP output subspace. Nonlinear-in-params: Taylor-expand about `theta_bar`, extended param `[theta;1]` (eqs 15-19). SOFT, KNOWN-subspace, small `beta` suffices (Remark 1); `Pi` independent of `theta` -> precompute SVD | THIS IS our C5 interpretability layer, and the TEMPLATE for the user's "steer the learning" point. The dissipativity/no-drift constraint is a **sibling steering** in the SAME cost (5k0 clarification). Its target is the power-sign `F.v` (state-dependent), not a fixed linear subspace -- so it is a nonlinear steering, but the same "penalize the undesired component" pattern. |
| **WELL-POSEDNESS (C2)** | [GUARANTEE 2026] Cayley-transform parametrization of `D_zw` so `||D_zw||<1/L` -> the LFR feedback `g(.)` is a contraction (Banach) -> algebraic loop has a UNIQUE fixed point, found by iteration. Constraint-free (unconstrained optimizer) | **REUSE directly** for our C2 (no algebraic loop when we add storage/friction states). Independent of the stability question. |
| **STABILITY (B1/B2)** | [GUARANTEE 2026] stability-by-construction = **CONTRACTION** (Def 11: `||xhat_i(k)-xhat_j(k)|| <= K alpha^k ...`, `alpha<1`), enforced by Cayley/matrix-exp on `A`; ANN Lipschitz bounded by soft `r_L=rho max{prod||Wi||-L,0}^2` or hard 1-Lipschitz layers | **CANNOT reuse as-is: contraction `alpha<1` DAMPS the free-integrator pole -> FAILS B2 (marginal-mode preservation).** Our delta = **RELAX contraction to MARGINAL (`||A||<=1` / passivity / NI) on the rigid-body (X/Y-position) subspace, keep contraction elsewhere.** THIS IS THE GAP = THE CONTRIBUTION (confirms 9; the Lipschitz-reg machinery is a reusable steering template). |

### What is reused vs invented (the "don't reinvent" answer)
- **REUSE (exists, in-framework):** storage-state template, friction-state template (+ gantry params),
  orthogonal-projection interpretability steering, Cayley well-posedness, Lipschitz-regularization
  machinery, group-lasso structure selection, JAX pipeline. Four of five pieces.
- **INVENT (the one genuine gap = the contribution):** marginal-mode-preserving stability -- relax the
  framework's contraction (`||A||<1`) to `||A||<=1` (passivity/NI) on the free-integrator subspace so the
  X/Y pole stays at the origin while the augmentation stays bounded (B1) and captures friction (A3). This
  is exactly the Gyorok-2026 delta identified in 9 and 5j; the literature confirms nothing off-the-shelf
  provides it.

### Immediate reusable wins for the buildable block (independent of the contribution)
1. **Cayley `D_zw`** -> our C2 (no algebraic loop) when storage/friction states are added.
2. **Orthogonal-projection term** -> our C5 interpretability, and the pattern for framing the
   dissipativity constraint as a sibling steering.
3. **Lan-Jia gantry LuGre params** -> the Phase-3 injected-friction sim (D-D2), no need to invent numbers.
4. **Friction-state = Dahl/LuGre bristle** and **storage-state = 2nd-order 150 Hz mode** -> the two
   physical augmentation states, both standard, both passive.

### Honest caveats
- The two `[STORAGE]`/`[FRICTION]` docs are AI-generated lit studies: their PRIMARY citations (esp. the
  Lan-Jia gantry params, THK numbers) must be verified at source before thesis use -- strong leads, not
  yet verified quotes.
- The `[GUARANTEE 2026]` contraction result was already primary-verified (9); this read confirms the
  Cayley well-posedness and the Lipschitz-reg forms in addition.
- Still unread at primary source: the neural friction/PH leads (arXiv:2504.12441 passive LuGre;
  arXiv:2401.09520 PH-NN dissipation) and the NI conditions (Mabrok 2014) -- verify only where the
  above leaves a gap.

### ADDENDUM (2026-07-10): the two FRAMEWORK papers (Hoekstra EJC 2025 + Drenth 2025), read full
These are the two papers whose method we implement; both read end-to-end. They LOCALIZE our contribution
precisely and hand us the reusable well-posedness trick.

**[FRAMEWORK] Hoekstra, Verhoek, Toth, Schoukens, "Learning-based model augmentation with LFRs",
EJC 2025 (doi 10.1016/j.ejcon.2025.101304).** The exact structure we run.
- Baseline `phi_base` + learning `phi_aug` via a FIXED interconnection matrix `S` (LFR, Fig 1); dynamic
  parallel augmentation adds extra states `x_tilde` (`x_hat = [x_bar; x_tilde]`); learning components read
  the model state and write to `x_tilde` AND `x_hat` (Table 1). SUBNET truncated multi-shooting loss
  (eq 5), encoder `psi_theta` sets the per-subsection initial state.
- Well-posedness Condition 1 = unique solution, enforced by **acyclic interconnection graph** (Thm 1,
  topological ordering). Parameter regularization `V_reg = ||Lambda(theta_base - theta_base^0)||^2`
  (Bolderman) = OUR param_loss/Lambda -- bounds PARAMETER deviation, nothing else.
- **THE KEY OBSERVATION (localizes our whole problem):** the headline example is a **3-DOF hardening MSD**
  (Fig 3): 2-DOF known baseline + a HIDDEN third mass-spring-damper + hardening spring, learned by dynamic
  parallel augmentation with **2 extra states**, and Fig 6/Sec 4.3 show it captures the hidden mass
  WITHOUT replacing the baseline. **That is EXACTLY our absorber-augmentation case -- EXCEPT every mass in
  his MSD has a SPRING (k1,k2,k3), so there is NO free integrator and NO drift is possible.** So the
  framework is PROVEN to learn a hidden MSD (our absorber) cleanly -- the ONLY thing his demonstrated case
  lacks and ours has is the **K=0 free-integrator axis (X/Y)**. This pinpoints the contribution: not "can
  augmentation learn a hidden MSD" (Hoekstra: yes), but "can it on a FREE-INTEGRATOR axis without drift"
  (open). And his two guarantees (acyclic well-posedness + parameter reg) do NOT touch drift -- drift is
  orthogonal to what the framework provides. Confirms param_loss != drift fix (D-076).

**[FRAMEWORK] Drenth, Hoekstra, Schoukens, Toth, "Efficient Learning of Affine and Rational Dependency
LPV Models with LFR", 2025.** How our BASELINE `M(Y)` is represented/kept well-posed.
- LPV-LFR `{M, Delta(p)}`: interconnection `M` with diagonal `Delta(p)` block, RATIONAL scheduling
  dependency (eq 6-9). Well-posedness Def 1 = `I - D_zw Delta(p)` non-singular for all `p`, enforced by
  **`rho(D_zw) < 1` (Condition 5)** via a **matrix-exponential direct parametrization `D_zw = e^{-N}`,
  `N > 0`** (eq 16-17, `N = Psi(D_A^T D_A + D_B - D_B^T + eps I)`) -> constraint-free (unconstrained
  optimizer). Scheduling set `||p||_inf <= 1` via tanh output on the NN scheduling map.
- **Reusable:** this is the SAME well-posedness mechanism as Gyorok-2026's Cayley `D_zw` and Hoekstra's
  acyclic graph -- three papers, one trick: parametrize `D_zw` with `rho(D_zw)<1` -> `I - D_zw Delta`
  invertible -> no algebraic loop. This DIRECTLY gives our C2 when we add storage/friction states, and it
  is how the rational `M(Y)` baseline stays well-posed. (It says nothing about drift either -- again
  orthogonal.)

**Consolidated takeaway from the framework papers:** the method we implement already learns a hidden MSD
(Hoekstra's 3-DOF example = our absorber) and already has constraint-free well-posedness for both the
augmentation and the rational `M(Y)` baseline (Hoekstra/Drenth/Gyorok, one `D_zw` trick). The SINGLE
thing none of them addresses is the **free-integrator (K=0) drift** -- it is orthogonal to well-posedness
and to parameter regularization. So our contribution sits in exactly one spot: **marginal-mode-preserving
stability on the free axis** (5j/5L), bolted into an otherwise-complete, reusable framework.

### ONLINE RESEARCH (2026-07-10): the KEY find = DiLaR-PINN validates our block and confirms our exact gap
Targeted online search (locate/triage; primaries flagged). Headline: our §5i skew-dissipative PH block is
essentially a PUBLISHED 2026 method (DiLaR-PINN), which validates the construction AND leaves exactly our
free-integrator gap open.

**[KEY PRECEDENT] DiLaR-PINN -- "Dissipative Latent Residual Physics-Informed Neural Networks for Modeling
and Identification of Electromechanical Systems", arXiv:2604.18277 (2026).** Abstract read verbatim.
- Problem it states = OUR diagnosis, verbatim: "residual-learning PINNs augment imperfect first-principles
  models, but unconstrained MLP residuals may inadvertently **INJECT ARTIFICIAL ENERGY into the system**."
- Method = essentially OUR §5i block: "the residual network operates only on **unmeasurable (latent)
  state** components and is parameterized in a **SKEW-DISSIPATIVE form that GUARANTEES NON-INCREASING
  ENERGY FOR ANY CHOICE OF NETWORK PARAMETERS**." Skew-dissipative (`J` skew minus `R` dissipative) latent
  state + energy-non-increasing-for-all-weights = the port-Hamiltonian passive-by-construction block we
  built. **We are NOT reinventing -- this is a citable precedent for the construction.**
- It beats a **SOFT** dissipativity-constraint variant and an unstructured MLP and a black-box LSTM on
  **long-horizon extrapolation** (real helicopter). -> independent confirmation of our "structural, not a
  soft penalty / not a heuristic" stance (5j lesson).
- Uses a **recurrent rollout with curriculum sequence-length extension** = our multi-shooting / nf-
  curriculum.
- **BUT: the abstract does NOT mention integrators, marginal stability, or free-body modes.** DiLaR-PINN
  guarantees NON-INCREASING ENERGY -- which (5j) bounds velocity/kinetic energy, NOT position on a free
  integrator. So **DiLaR-PINN would STILL DRIFT on our K=0 X/Y axes.** This is the sharpest possible
  confirmation of the contribution: the state-of-the-art skew-dissipative augmentation is exactly our
  block and exactly stops where we must go on (marginal-mode preservation / bounded POSITION on the free
  integrator). Thesis framing: "DiLaR-PINN-style skew-dissipative latent residual + marginal-mode-
  preserving (NI/semidefinite) extension for free-integrator axes."
- **METHOD READ (primary, arxiv HTML, 2026-07-10) -- confirms the construction IS ours AND pinpoints the
  gap to ONE assumption:**
  - Residual `r_phi = (S_phi - K_phi) grad_{xlat} V(x)`, `S_phi` skew, `K_phi = L_phi L_phi^T >= 0`
    (Cholesky) -- **identical to our §5i `(J - R) gradH`.** Dissipativity `grad V^T r_phi <= 0` for ALL
    params (their Prop 1) via the SAME proof we derived (skew term = 0, PSD term >= 0). Confirms the
    construction is citable, not homemade.
  - Residual acts **only on latent (unmeasured) states**; observed-state eqs get zero residual (Eq 3).
    (For us: X/Y velocity/force rows are the natural latent target; positions are measured -- routing the
    FORCE to the velocity row is compatible with "residual on latent state".)
  - **THE DECISIVE GAP = their Proposition 3:** the augmented system stays stable (ISS) **only IF the
    baseline `f_phys` is ISS** (`grad V^T f_phys <= -alpha3(||x||) + sigma(||u||)`). **Our baseline is NOT
    ISS on X/Y: a free integrator is not asymptotically stable / not 0-GAS.** So DiLaR-PINN's stability
    theorem's PREMISE FAILS for us -> NO guarantee on our K=0 axes -> (5j) it would drift. **This is the
    exact, single-assumption statement of our contribution: extend the dissipative-residual stability
    guarantee from an ISS baseline to a MARGINALLY-STABLE (free-integrator) baseline.**
  - In their case study `S_phi -> 0` (pure dissipation, no storage/skew) -- OUR absorber REQUIRES the
    skew/storage term, so our case exercises the full PH structure more than their example.
  - Training: RK4 recurrent rollout + curriculum sequence-length extension (= our multi-shooting /
    nf-curriculum); loss on measured states only, variance-weighted.

**Other online results (triaged):**
- **RENs (2104.05942):** CONFIRMED contraction is w.r.t. a **strictly positive-definite metric `P>0`**;
  incremental passivity is enforced JOINTLY with contraction. No `P>=0` (semidefinite / marginal)
  relaxation exists as published -> confirms 9. (lead: relax `P>0` to `P>=0` = the REN route to our gap.)
- **Passive learned LuGre PINN (2504.12441):** captures **nonzero-at-rest** friction via a LuGre bristle
  latent state with learned params; usable as a grey-box block; **but NOT provably passive by
  construction** (fit, not guaranteed). -> good friction MODEL, does not itself give the passivity
  guarantee; the guarantee must come from the block structure (DiLaR-PINN/PH). Also: "Newtonian neural ODE
  ... serial manipulators with LuGre friction" (IEEE TIE 2025) -- another learned-LuGre lead.
- **Projection operator (Lavretsky-Gibson, arXiv:1112.4232):** CONFIRMED -- the **hard-projection** anti-
  drift steering (constrain weights to a feasible set -> zero steady-state bias INSIDE the set -> keeps
  friction). The hard version of the DC-guardrail steering; a real option for the steering layer.
- **Bounded-impulse / zero-DC-gain / gradient-of-bounded-potential output:** search returned NO clean
  named control literature -> genuinely under-explored (consistent with the deep-research doc calling it a
  "novel building block"). The bounded-integral block (D-C) remains our own construction here.
- **Nonlinear NI with free-body motion (2011.14610):** LEAD only -- it is a networked-consensus/controller
  paper; the small-model fetch could not extract a nonlinear free-body SEMIdefinite-storage theorem.
  Needs a real read IF we pursue the NI route; not confirmed to close the gap.

**Net update to the plan:** the skew-dissipative latent-residual augmentation (our §5i block) is now a
CITABLE, state-of-the-art construction (DiLaR-PINN), not a homemade guess -- and it provably STOPS at our
gap (it bounds energy, not free-integrator position). Our contribution is unchanged and now even more
sharply bounded: **add marginal-mode preservation (bounded POSITION on the K=0 axis) to the DiLaR-PINN/PH
skew-dissipative residual**, via NI/semidefinite storage or the bounded-impulse structural trick.

### SEARCH SATURATION (multiple passes, 2026-07-10) -- the gap is confirmed from every angle
After several independent query phrasings, the results CONVERGE on one small cluster (RENs, NINODE,
DiLaR-PINN, the dissipative-boundedness family). Every candidate falls into exactly one of three buckets,
and NONE covers a marginally-stable / free-integrator forward-model baseline:
1. **Assumes ISS / attractor** (converges to a bounded invariant set): DiLaR-PINN (Prop 3, ISS baseline);
   "Dissipative Deep Neural Dynamical Systems" (2011.13492, bounded positively-invariant level set);
   "Learning Dissipative Chaotic Dynamics" (2410.00976); ECO energy-constrained operator learning
   (2512.01984). All need a RETURN to equilibrium -> exclude the free integrator.
2. **Contraction / strict stability by design** (`||A||<1`, `P>0`): RENs/NodeRENs (metric strictly PD,
   confirmed again), stable-by-design ID. Excludes the pole at the origin.
3. **Controller that stabilizes (damps) the plant**: NINODE (2504.19497), closed-loop dissipativity
   synthesis (2404.07373), L2-bounded SSM controllers (2606.11049). These DAMP the free mode, not preserve
   it; and they are controllers, not forward-model augmentations.
~~The only classical framework native to the free-body/marginal case (NI, Mabrok 2014) is LTI-only.~~
**[CORRECTED 2026-07-10 -- verification addendum below. Mabrok 2014 is LTI-only, but the NONLINEAR NI
free-body semidefinite-storage theory exists analytically (Shi-Petersen-Vladimirov 2011.14610 Def 1;
Ghallab-Petersen 2201.00144); it is just not LEARNED and not a forward augmentation.]** So the
**learned dissipative FORWARD augmentation that PRESERVES a marginally-stable (free-integrator) baseline
mode** is genuinely absent = confirmed contribution. Reusable pieces are settled (DiLaR-PINN/PH skew-
dissipative construction; Cayley/matrix-exp `D_zw` well-posedness; orthogonal-projection steering; LuGre
friction model; Lavretsky-Gibson projection-operator anti-drift steering). Remaining OPTIONAL deep-reads
(not expected to close the gap): nonlinear NI free-body (2011.14610); "Dissipative Deep Neural Dynamical
Systems" (2011.13492). Diminishing returns reached -- further breadth search not warranted; next value is
in BUILDING the marginal-preservation layer or deriving the NI/semidefinite relaxation (theory).

### VERIFICATION ADDENDUM (2026-07-10) -- independent adversarial check of the §5L / §5j / §5e claims

**Provenance.** A fresh, adversarial verification pass (brief: `docs/fable-review-brief.md`) opened every
on-disk PDF, extracted full text (PyMuPDF), read the two flagship papers (DiLaR-PINN, RENs) end-to-end
BEFORE comparing to the catalog, and red-teamed the central gap on-disk and by web search. Full per-item
verdicts with exact page/prop locations are in **`docs/passivity-augmentation-literature.md` §G**. Summary:

**CONFIRMED (load-bearing claims verified at the primary source):**
- **DiLaR-PINN (2604.18277):** residual `(S−K)∇V`, S skew, K PSD (Eq 5 = our §5i PH form; Prop 3 Remark
  confirms `[J−R]∇H` identity), dissipative for ALL params (Prop 1, p.3), residual on latent states only
  (Eq 3). **Its stability theorem (Proposition 3, p.4) REQUIRES an ISS baseline** (`∇Vᵀfphys ≤ −α₃+σ`,
  αᵢ,σ∈K∞) -- and a mass-damper/free integrator is not 0-GAS, so the premise fails on our K=0 X/Y axes.
  The "would-drift" conclusion rests on OUR §5j argument, not on DiLaR (Prop 3 is only sufficient). Authors
  now recorded: **Long, Solak, Ajoudani (IIT), IFAC 2026.** Bonus: Prop 2 proves `(S−K)∇V` covers the whole
  dissipative cone (expressivity is not lost).
- **RENs (2104.05942):** contraction is w.r.t. a STRICTLY positive-definite metric `P≻0` (Thm 1, Def 2,
  α∈(0,1)); incremental passivity is enforced JOINTLY with contraction (Thm 1(2), Thm 3), never as a
  marginal alternative; NO `P⪰0`/marginal variant exists. (Watch-out: Thm 1's `ᾱ∈(0,1]` still yields
  `α<ᾱ`, i.e. strictly contracting -- not a marginal case.)
- **Mabrok 2014 (1305.1079):** free body / poles at origin, residue-PSD DC condition, LTI/transfer-function
  ONLY; attribution = Mabrok et al. (paper itself credits the original NI notion to Lanzon & Petersen).
- **NINODE (2504.19497):** a CONTROLLER (stabilizes an NI plant); Assumption 3 rules out DC gain = 1 →
  abandons the marginal case.
- **§5j math (this doc):** `∫F·v ≤ H(0)` + damping `c>0` ⟹ v∈L² ⟹ Cauchy-Schwarz `|q(T)| ≤ C√T`
  (sub-linear, unbounded as T→∞); position bounded IFF net impulse `∫F dt` bounded. Mass-damper
  (`F→q = 1/(s(ms+c))`, pole at origin + at −c/m) is CONSISTENT with -- and `c>0` ENABLES -- the
  bounded-impulse→bounded-position (Route B) argument. **All independently re-derived and correct.**

**CORRECTED (one substantive over-claim, appears in §5e, §5j, and §5L above):**
- The statement **"NI theory is LTI-only" / "the nonlinear-NI semidefinite-storage version is
  unworked-out"** is **REFUTED at the primary source.** `2011.14610` (Shi-Petersen-Vladimirov 2021),
  **Definition 1**, defines a **nonlinear** NI system by a **positive-SEMIdefinite** storage function
  `V∈C¹` with `V̇ ≤ uᵀẏ̃`, **explicitly including poles at the origin (free body)**; Def 2 gives nonlinear
  OSNI; a nonlinear NI⊗OSNI stability theorem is proved. Corroborated by Ghallab-Petersen `2201.00144`
  ("NI Theory for Nonlinear Systems: A Dissipativity Approach") and the nonlinear-NI quadrotor line
  (2101.04916, 2603.27560). **The analytical nonlinear-NI free-body semidefinite-storage theory EXISTS.**
  Mabrok 2014 specifically is LTI-only, but it is not "the only" free-body NI framework.
  Corrected inline at all three locations (struck through, dated).
- **Provenance fix (catalog D1):** the LuGre-PINN "not provably passive by construction" line is an
  INFERENCE, not a paper quote ("passive"/"dissipative" appear 0× in 2504.12441). Don't cite as a quote.

**THE CENTRAL GAP CLAIM STILL HOLDS -- with a sharper, better-founded rationale.** No published **learned
dissipative FORWARD augmentation preserving a pole at the origin with bounded position** was found on disk
or by "search beyond" (web): every learned candidate is ISS/attractor (DiLaR-PINN, 2011.13492, 2410.00976,
2404.12554 "Learning Stable and Passive NODEs" [PD storage bounded below by a quadratic → attractor],
2309.16032), contraction (RENs), or a controller that damps the plant (NINODE, 2011.14610). The gap is NOT
in the underlying NI theory (which exists analytically, per the correction above) -- it is in the
**LEARNED + forward-model + LPV-scheduled** realization. So the contribution is best framed as **"adapt the
existing nonlinear-NI free-body semidefinite-storage theory (Shi-Petersen-Vladimirov) to a learned parallel
forward augmentation in the LPV-LFR/SUBNET framework,"** not "invent nonlinear-NI free-body theory." This is
narrower, stronger, and gives a citable classical foundation to build on. (Verdict: **holds-with-caveats.**)

---

## 5m. PROBLEM RE-FRAMED (2026-07-10) -- it is an IDENTIFIABILITY problem, not (only) a stability problem; the search was too narrow

**Trigger (user, 2026-07-10):** "we are not limited to dissipative forward augmentation... define the actual
problem we are facing, maybe our search has just been too limited." Correct on both counts. The entire §5-§5L
search was anchored on ONE solution family (dissipativity/passivity/NI). Re-derived from the diagnosis, the
problem has a broader and more mature home. This section states the problem SOLUTION-NEUTRALLY, decomposes
the solution space into FOUR families, and records a reframed literature pass (deliberately NOT using the
"dissipative/passive" keyword that biased earlier results).

### The problem, stated with zero reference to any solution
> **The rigid-body (DC / net-impulse) direction of the learned residual is a NULL / UNEXCITED direction of
> the training objective -- the data carries almost no information about it -- while the plant (free
> integrator on X/Y) makes the DELIVERABLE metric (long free-run position) unboundedly sensitive to exactly
> that direction.**

Two facts collide, neither of which is "dissipativity":
1. **Estimation side -- the direction is UNIDENTIFIABLE.** Excitation is a narrowband zero-mean multisine
   [130,180] Hz: no energy in the low-frequency/DC band, and the 0.1 s window cannot see slow modes. So the
   DC/net-impulse component of the residual force is a **flat direction of the loss** -- a gradient/
   simulation-error estimator does not "fail to remove" a DC, it **wanders** into one because nothing pulls
   it back. (The one true DC -- M(Y) rectification -- is tiny; the dominant true residual is zero-DC, D-A.)
2. **Structure side -- the plant AMPLIFIES that exact direction without bound.** Free integrator: a constant
   force → a ramp with nothing to arrest it. The deliverable is maximally sensitive precisely where the
   estimator is blind.

This is a classic **ill-posed inverse / null-space / unexcited-direction** problem. In adaptive-control
language it is literally **"parameter drift along an unexcited direction"** -- the founding problem of ROBUST
adaptive control (Ioannou, Narendra, 1980s), which we never searched because we were anchored on passivity.

### The decomposition that resolves the confusion (TWO different problems fused into one wish)
- **Problem 1 -- estimation/identifiability (what sim-drift actually is):** the good model is NOT the unique
  minimizer of the short-window loss along the marginal direction. Fixed by a BETTER ESTIMATOR (families A/B/C
  below). Knowledge-REQUIRED is acceptable here for SIM (we know the true DC is 0).
- **Problem 2 -- knowledge-free guarantee (real-data insurance):** on UNKNOWN real data we want no drift for
  any residual, without knowing the dynamics. Fixed by a STRUCTURAL property (family D). Only HERE does
  passivity/NI genuinely earn its place.

**Consequence = an inversion of the §5 ranking:** sim-drift is a Problem-1 (estimation) pathology and should
be attacked with ESTIMATION tools FIRST; passivity/NI is Problem-2 insurance for real data, NOT the primary
sim fix. The prior §5 PRIMARY(passivity)/SUPPORTING ranking had this backwards for the sim phase.

### FOUR solution families (only D was deeply searched before)
| # | Attack | What it does | Literature (families A-C were UNDER-searched) |
|---|---|---|---|
| **A. Pin the null direction with a prior** | regularize the unexcited DC/impulse toward a known value | ridge/Tikhonov-in-null-space; **adaptive-control drift mods (σ-mod, e-mod, PROJECTION)**; Bayesian/stable-kernel priors; **our own orthogonal projection (Györök) IS a null-space regularizer**; Lavretsky-Gibson projection (ON DISK, §E1) |
| **A-phys. Physically-structured residual prior** | grey-box: known linear dynamics + structured prior on the unknown FORCE | **latent restoring-force / GP latent-force models** (Rogers-Friis); switching-GP for friction |
| **B. Remove the direction from the hypothesis space** | reparametrize so the residual CANNOT carry net impulse/DC | integrator factoring / Tustin-net (Forgione-Piga); **bounded-impulse (OURS, validated)**; integrating-mode-as-constrained-parameter (Kuntz-Rawlings) |
| **C. Recondition / re-excite** | make the direction observable | multiple-shooting + continuity (Turan-Jäschke); rollout curriculum (Farina-Piroddi, Ribeiro); prefiltered-LS for marginal systems; change the SIM input to excite low freq |
| **D. Knowledge-free structural guarantee** | boundedness for ANY weights | dissipativity / passivity / NI / contraction -- the ONLY branch deeply searched in §5-§5L |

### SCORECARD -- the four families vs the §5 four requirements (reconciles the reframe with the real-data criterion)
The four families widen the SIM toolset; the §5 four requirements are the REAL-DATA selection criterion and
must still gate any real-data solution. Requirements: **(1) knowledge-free** (holds without knowing the true
residual), **(2) friction-permitting** (represents genuine dissipative state-dependent DC force), **(3)
marginal-preserving** (keeps the free-integrator pole at the origin, no added stiffness/damping), **(4)
non-drifting** (bounded free-run + keeps X/Y routing, D-103). Honest mapping:

| Family | 1. Knowledge-free | 2. Friction-permitting | 3. Marginal-preserving | 4. Non-drifting |
|---|---|---|---|---|
| **A. Pin null dir** (σ-mod, projection, Györök) | ⚠️ pins toward a prior | ❌ suppresses real friction DC unless friction in `f_base` | ✅ | ✅ (bounds the estimate) |
| **A-phys. Latent-force GP** (Rogers-Friis) | ✅ | ✅ (models the force incl. friction) | ✅ | ⚠️ conditions, not guarantees |
| **B. Bounded-impulse / remove dir** (OURS, validated) | ✅ | ❌ forbids sustained DC → forbids friction (needs `f_base`) | ✅ | ✅ (proven) |
| **C. Re-excite** (broadband low-freq) | ✅ | ✅ | ✅ | ⚠️ conditions the estimator; **NOT available for fixed real hardware** |
| **D. Passivity / NI** (semidefinite storage) | ✅ | ✅ (permits dissipative DC) | ✅ | ✅ |
| **D2. Momentum-conservation** (Newton 3rd law, Dynami-CAL) | ✅ | ⚠️ friction is EXTERNAL → unconstrained channel | ⚠️ only INTERNAL mode; external channel unprotected | ❌ external-force channel (= our drift) is UNCONSTRAINED |
| **D3. Marginal-native dissipativity** (EID, cyclo-passivity, Casimir) | ✅ | ✅ | ✅ (characterizes the marginal mode) | ❌ bounds velocity / gives "only instability results", NOT position (§5j) |

**What the scorecard says (a genuine corrective to over-selling A/B/C):** **only Family D natively satisfies
ALL FOUR for the real-data deliverable.** A and B each fail criterion 2 (they kill real friction DC unless
friction is moved into `f_base`, §5d); C fails criterion 4's AVAILABILITY on fixed real hardware (you cannot
re-excite a machine whose logs you are handed). So the four requirements **RE-VINDICATE why passivity/NI was
the PRIMARY** -- for the REAL-DATA case. This does NOT contradict the reframe; it sharpens the split:
- **SIM / training-pathology phase (Problem 1):** the four requirements do not all bind -- no friction in sim
  → crit 2 is moot; the sim CAN be re-excited → C is available. Here A/B/C are the cheap, PROVEN fixes.
- **REAL-DATA deliverable (Problem 2):** all four bind, and **D is the only single mechanism meeting them
  all** -- exactly the original §5 argument. A/B become viable for real data ONLY combined with grey-box
  friction in `f_base` (§5d), which is itself legitimate.
So the identifiability reframe widens the SIM toolset; the four requirements keep the REAL-DATA bar where D
earns its place. (Both views are now consistent: A/B/C solve the sim estimation pathology; D + optional
`f_base`-friction is the real-data deliverable.)

### EXCITATION NOTE (Family C) -- the reframe partly hinges on the narrowband sim excitation
The "unexcited null direction" of Problem 1 is a property of the CURRENT sim excitation: a **narrowband
[130,180] Hz zero-mean multisine** (§1) puts NO energy in the low-frequency/near-DC band. **Physics:** a free
integrator has position sensitivity `|q/F| ≈ 1/(mω²)`, so exciting at 1 Hz vs 150 Hz makes a low-frequency
force error **~(150/1)² ≈ 2.2e4× more visible in position** → broadband LOW-FREQUENCY excitation would
LARGELY DISSOLVE Problem 1 (the estimator finally sees the near-DC force it currently cannot). That is Family
C, the most classical fix ("excite what you must identify"). **Caveats:** (i) a zero-mean broadband signal
still has no content at EXACTLY 0 Hz, so the pure-DC/net-mean sliver stays weakly identifiable -- but it is
practically negligible once 1 Hz is strongly excited; (ii) **re-excitation is only available where WE choose
the input (sim), not for fixed real Telica logs.** **DATA STATUS (unresolved 2026-07-10):** the real Telica
data in `docs/kamtin-telica-schema.md` is **closed-loop motion-profile / ILC tracking (`M0` reference,
`MF230/MF30` currents, iter0-8 over an X/Y grid), NOT a broadband multisine.** A separate "joint 1-200 Hz"
identification dataset is **NOT documented in the schema**; if it exists (user asked 2026-07-10) it must be
located and its band confirmed -- it would materially change the SIM-phase analysis (Family C becomes the
lead) and possibly the real-data identifiability. **ACTION: confirm whether a broadband low-frequency joint
dataset exists before finalizing the family choice.**

### NEW CANDIDATE EVALUATED (2026-07-10, PRIMARY-READ) -- momentum-conservation (Dynami-CAL GraphNet): REFUTED as a full solution, but yields one insight
A reframed search (avoiding the "dissipative" keyword) surfaced a genuinely new angle: attack the marginal
mode via a DIFFERENT conserved quantity -- **linear/angular MOMENTUM (Noether / Newton's third law)** --
instead of energy/dissipativity. Flagship: **Dynami-CAL GraphNet, "A physics-informed graph neural network
conserving linear and angular momentum for dynamical systems", Nature Communications, Jan 2026
(arXiv:2501.07373).** Read at primary (arXiv HTML). Verdict: **conceptually valuable, but NOT a solution to
our drift.** Details:
- **What it guarantees (CONFIRMED by construction, all weights):** internal pairwise edge forces are
  equal-and-opposite `F⃗ᵢⱼ = −F⃗ⱼᵢ` (via antisymmetric local-basis geometry + node-interchange-invariant edge
  embeddings), so **internal interactions conserve linear momentum for ANY network weights** (analogous to
  DiLaR-PINN's "for all parameters"). Angular momentum conserved locally per edge (proof in Supplementary
  §5.1; NO numbered theorem in the main text).
- **THE DECISIVE FINDING -- external forces are a SEPARATE, UNCONSTRAINED channel:** verbatim, "If external
  forces are present, the changes in velocity and angular velocity are **decoded directly from the node
  scalar embeddings** hᵢ for each node." So momentum-conservation protects only the INTERNAL (edge) channel;
  the external-force channel has NO conservation constraint. **Our drift is a NET external force on X/Y (the
  ANN's spurious DC) → it would live in exactly this UNCONSTRAINED channel → the architecture does NOT forbid
  it.** This is the same structural truth as §5j from a new angle: nothing on internal interactions bounds
  position against an external DC.
- **No stability claim:** the paper claims only EMPIRICAL "stable error accumulation over extended rollouts"
  on collision/granular systems -- NO bounded-trajectory, marginal-stability, or pole-at-origin theorem. So
  criterion 4 is not structurally met.
- **Framework mismatch:** it is a GRAPH neural network requiring an explicit multi-body graph
  (`G=(V,E)`, nodes=bodies, edges=interactions); the force output is edge-aggregated node updates, NOT a
  generic residual. Dropping it into LPV-LFR/SUBNET is a major integration mismatch, not a swap.
- **Scorecard (row D2 above):** knowledge-free ✅; friction-permitting ⚠️ (friction is external →
  unconstrained channel, permitted but unprotected); marginal-preserving ⚠️ (only the internal mode, via
  Noether; the external channel can still inject net force on the free pole); non-drifting ❌ (the external
  channel = our drift is unconstrained). **Fails exactly where it matters (crit 4).**
- **The ONE salvageable insight:** the *principle* "constrain the augmentation's NET momentum injection to
  zero" is precisely our **bounded-impulse block (family B)** -- momentum-conservation gives a clean PHYSICAL
  justification for it (spurious drift = spurious momentum injection from nothing), but Dynami-CAL implements
  it only on the internal channel, which we do not need (the absorber coupling is already zero-DC/oscillatory
  and family B already handles it). So the physical framing is a nice thesis narration for the bounded-impulse
  constraint; the GNN itself adds nothing buildable for us. **Do NOT pursue the GNN; DO reuse "zero net
  momentum injection" as the physical name for the bounded-impulse constraint.**
- **Net:** the reframed search has now been pushed to a second saturation. No published method meets all four
  requirements; momentum-conservation (the most promising new angle) fails crit 4 because the external-force
  channel is unconstrained. Family D (passivity/NI, semidefinite storage) remains the only single mechanism
  that meets all four -- confirming D-104's gap from yet another direction.

### MARGINAL-NATIVE DISSIPATIVITY THEORY (PRIMARY-READ 2026-07-10) -- CORRECTS a second over-claim; the theory EXISTS but still does not bound POSITION
User challenge (2026-07-10): "I don't believe there is not more theory on the dissipative method." **Correct.**
Prior searches kept the "learned/neural" keyword, which returned only strict-PD constructions and hid the
CLASSICAL dissipativity theory for marginal / continuum-equilibrium / free systems. That theory is mature.
Four distinct notions, all directly on our case:
- **Equilibrium-Independent Passivity/Dissipativity (EIP/EID)** -- Hines, Arcak, Packard, *Automatica*
  47:1949-1956 (2011); detailed treatment Simpson-Porco, IEEE TAC (arXiv:1709.06986). **PRIMARY-READ.**
- **Cyclo-dissipativity / cyclo-passivity** -- Willems → Hill-Moylan → van der Schaft, IEEE TAC 2021
  (arXiv:2003.10143 "Cyclo-dissipativity revisited"). **PRIMARY-READ.**
- **Port-Hamiltonian Casimir functions** -- flat storage direction = the rigid-body/free coordinate; learned
  versions appearing (Neural Energy-Casimir Control arXiv:2112.03339; PHAST arXiv:2602.17998, 2026). Lead.
- **Shifted / Krasovskii / differential passivity** -- Kawano, Kosaraju, Scherpen, IEEE TAC 2021
  (arXiv:1907.07420). Continuum/nonzero-equilibrium toolkit. Lead.

**PRIMARY-READ verdicts (verbatim quotes from the on-disk-extracted PDFs):**
- **Cyclo-dissipativity (arXiv:2003.10143) -- CONFIRMED the indefinite-storage relaxation, but it does NOT
  bound trajectories.** Storage need not be `≥0` or bounded below: (p.6 footnote) *"Note that we do not yet
  require S to be nonnegative or bounded from below."* Definition 3.1: cyclo-dissipative if `∮ s(u,y)dt ≥ 0`
  for all `x(T)=x(0)`. **THE DECISIVE caveat -- Remark 3.4, p.8:** *"the Lyapunov function obtained for the
  interconnected system by summing the storage functions ... is no longer nonnegative. Hence in principle
  only **instability** results can be inferred."* So cyclo-passivity is the correct marginal/indefinite-
  storage relaxation but gives LESS than passivity on boundedness -- explicitly NOT a position bound.
- **EID (arXiv:1709.06986; Hines-Arcak-Packard 2011) -- CONFIRMED the continuum-of-equilibria
  characterization, still no position bound.** Definition 3.2 (p.3-4): EID requires *"for every equilibrium
  x̄ ∈ EΣ, ... a storage function V_x̄ : X → R≥0 with V_x̄(x̄)=0"* and the shifted supply rate; the assignable-
  equilibria set `EΣ = X` when m=n (EVERY state is an equilibrium = the free-integrator/continuum case).
  But its stability results rest on an INCREMENTAL stability condition and characterize the
  passivity-around-any-equilibrium property. A mass-damper IS EID (velocity-passive around any position) yet
  **position still integrates** → EID nails criterion 3, NOT criterion 4.

**WHAT THIS CHANGES (a second over-claim corrected, after the nonlinear-NI one in §5L):**
1. The docs' repeated claim that the **"semidefinite / marginal-preserving dissipativity notion is
   unworked-out / has no off-the-shelf result"** (§5b, §5e, §5j) is **PARTLY WRONG.** The
   indefinite-storage relaxation (cyclo-passivity) and the continuum-of-equilibria characterization (EIP/EID)
   and the flat-storage-direction object (PH Casimir) ARE that theory -- mature, classical, citable. Marginal
   dissipativity is NOT unworked. Corrected inline at those sections (struck-through, dated).
2. **BUT the four-requirements verdict is UNCHANGED and, if anything, sharper:** every one of these notions
   PERMITS/CHARACTERIZES the marginal mode (criterion 3) WITHOUT bounding position (criterion 4). Cyclo
   explicitly gives "only instability results"; EID bounds the shifted I/O behaviour, not the free
   coordinate. This is the SAME §5j fact from the classical side: **passivity of ANY flavour bounds
   velocity/kinetic-energy, not position on a free integrator.** Position-boundedness needs the EXTRA
   structural ingredient -- net-impulse (Route B, ours) or Negative-Imaginary free-body (the force→position
   class) -- layered on top of the (now well-founded) marginal-storage relaxation.
3. **Net effect on the contribution:** it shrinks AGAIN and is better-founded. The marginal-preserving
   storage relaxation (crit 3) = REUSE (EID/cyclo/Casimir, classical). The LEARNED realization + FORWARD
   augmentation + LPV + the POSITION-boundedness layer (crit 4, via net-impulse or NI) = the genuine
   contribution. We are now reusing the storage-relaxation theory AND the NI free-body theory (§5L
   correction), and inventing only their LEARNED/forward/LPV assembly plus the position-bound coupling.

**Honest scope note:** EID/cyclo/Casimir are ANALYTICAL system-theory (not learned) and are
CHARACTERIZATIONS/interconnection tools, not forward-augmentation recipes. They give the rigorous LANGUAGE
for stating criterion 3 (marginal-preserving) in the thesis proofs (§5e proofs 2-3) -- which is exactly where
the docs had wrongly declared a theory gap. They do not remove the need for the criterion-4 layer.

### KEY realization -- we already OWN two family-A tools
- **Györök orthogonal projection** (our C5 interpretability layer) is structurally a **null-space
  regularizer** -- exactly the family-A tool for the identifiability problem. We built it for
  interpretability; it is ALSO the principled steering for the unexcited DC direction. (Reframed use, same code.)
- **Lavretsky-Gibson projection operator (arXiv:1112.4232, ON DISK, catalog §E1)** is the HARD-projection
  member of the adaptive-control drift-modification family -- filed earlier as a minor "steering option"
  without recognizing it is a direct, Lyapunov-proven answer to Problem 1.

### Reframed literature pass (2026-07-10) -- NEW leads, all `[online-primary/search-level -- verify at source]`
Searched the axes we under-searched, explicitly avoiding "dissipative/passive":
- **[A -- adaptive drift]** σ-modification (Ioannou & Kokotović), e-modification (Narendra & Annaswamy),
  parameter projection -- introduced SPECIFICALLY to prevent parameter drift in unexcited directions, with
  **Lyapunov-proven uniform ultimate boundedness and NO knowledge of the true parameter**. Recent:
  "Constrained Parameter Update Law for Adaptive Control" (arXiv:2504.19412). This is Problem 1 solved, with
  proofs = NOT heuristics.
- **[A -- Bayesian/kernel]** **Pillonetto & Ljung, "Full Bayesian identification of linear dynamic systems
  using stable kernels", PNAS 2023**; stable-spline/stable-kernel priors -- pin the unexcited impulse-response
  direction to sensible decay.
- **[A-phys.]** **Rogers & Friis, "A Latent Restoring Force Approach to Nonlinear System Identification",
  MSSP 2022 (arXiv:2109.10681)** + sliding-window follow-up (arXiv:2602.21918); **switching GP latent-force
  model for a discontinuous nonlinearity/FRICTION (arXiv:2303.03858)**. Grey-box: known linear dynamics +
  physically-structured prior on the unknown force, EXPLICITLY separating physical model from learned
  residual force (incl. friction). = our exact structural setup, from the structural-dynamics community.
- **[B -- integrating mode]** **Kuntz & Rawlings, "Maximum Likelihood Identification of Linear Models with
  Integrating Disturbances for Offset-Free Control", IEEE TAC 70(9):5675-5689, 2025 (arXiv:2406.03760)** --
  estimates a FREE-INTEGRATOR ("integrating disturbance") mode DIRECTLY from data, kept well-behaved via
  eigenvalue-LMI constraints; code released. A principled parametrization of exactly our marginal mode.
- **[C -- marginal-system ID]** Farina & Piroddi multi-step SEM (already cited); "No-Regret Prediction in
  Marginally Stable Systems" (Ghai et al., PMLR 2020) -- prefiltered-LS guarantees for marginal systems.

### The genuinely-novel residual (much smaller than "invent neural NI")
Family A/B pinning is legitimate and proven for SIM (true DC = 0). The one case that still needs family D is
**real data where the unexcited direction ALSO carries real friction DC** -- there family-A pinning would
suppress SIGNAL (the §5 anti-mean-penalty argument). Resolution is either grey-box friction in `f_base`
(families A-phys/B handle it, §5d) or the family-D "permit dissipative DC, forbid injecting DC" guarantee.
So the honest novel core is **that real-data collision**, not "marginal-preserving neural NI from scratch" --
a real but far smaller and better-supported contribution.

### Status / next step
Reframe recorded; NOT yet acted on. Highest-value next actions (position-based, non-heuristic, per the
standing constraint): (1) recognize/deploy the Györök projection AND Lavretsky-Gibson projection as the
family-A null-direction pin (we own both); (2) evaluate the Kuntz-Rawlings integrating-mode-LMI and the
latent-restoring-force GP as A-phys/B candidates for the residual; (3) keep the validated bounded-impulse
block (B) and the passivity/NI story (D) as the real-data insurance, now correctly scoped. All new cites are
`[verify-at-source]` leads, not yet primary-read.

---

## 6. Status of the plan

- **Diagnosis: complete and measured.** Cause = ANN DC force on the K=0 rows, invisible to the
  0.1 s loss, ramped by the free integrator.
- **Supervisor-first checks** (before building fixes): S1 (validation horizon) ✅, S2 (hypothesis
  figure `s2_hypothesis_dc_force.png`) ✅, S3 (open-loop isolation) ✅, S4 (train/val audit) ✅.
- **Next**: the controlled fix comparison (§7 below).

### S4 — train/val audit (read-audit; no leakage, but a selector mismatch)

**Split is clean (no leakage):** TRAIN=T1–T14, VAL=V1–V4, TEST=E1–E4 are disjoint files
(`data.py`); normalization constants are built from `train_list` only
(`compute_normalization`); checkpoint selection uses `val_ckpt_data` = V1–V4, disjoint from
train; windows are per-trajectory (non-overlapping probe, `stride=nf`). So **`best=epoch 0` is
NOT a split/leakage artifact.**

**Selector is mismatched to the training horizon (the mechanistic reason for `best=epoch 0`):**
training optimises the 0.1 s windowed loss, but checkpoint selection uses
`validation_measure='sim-RMS'` (`training.py`) = **full free-run simulation RMSE over all of
V1–V4** (≈192k samples, 12 s each). From S1 the drift enters ~0.5 s and dominates by 12 s, so the
selector is **drift-dominated** and rewards the *least-drifting* model — the untrained one. The
moment the ANN adds any DC, the 12 s sim-RMS rises and epoch 0 wins, even if the 0.1 s windowed
loss improves. The matched-horizon metric (windowed val `nf-RMS`) is computed as a monitoring
probe but is **not** the selector.

**Tension (not a prescription):** do not simply switch the selector to 0.1 s — the deliverable IS
full free-run fidelity, so a 0.1 s selector would pick a model that is fine in-horizon yet still
drifts on the real metric. The selector horizon should be at least where drift becomes visible
(~0.5 s), not 0.1 s and not necessarily the full 12 s. This is the same nf sweet-spot tension as
S1 and **gates the Tier-2 scoring** (§7 precondition): fix the scoring horizon/metric before
comparing fixes.

---

## 7. Tier 2 — controlled fix comparison (test protocol)

**Goal**: build toward the real-data deliverable (§5 selection criterion: knowledge-free, friction-
permitting, marginal-preserving, non-drifting). The comparison is done on the SIM data first because it
is the controlled setting where we KNOW the true residual — but every variant is judged on whether its
guarantee would survive on real unknown data, not just whether it removes the sim drift.

**Precondition (gates the scoring)**: S1/S4 settled the horizon question — judge at a horizon that can
see the drift (~0.5 s), not the 0.1 s window, and note the selector mismatch (§6). Fix the scoring
horizon/metric before running so all variants are scored the same way.

**Common setup** (identical across variants — a fair comparison): routing `ann_route_ix=(0..7)`,
`stride=100`, cropped validation (fast), same seed, ~5–10 epochs, `nf` fixed. Reuse
`make_drift_checkpoint.py`'s fast setup.

**Variants** (ordered by the real-data criterion — primary = knowledge-free structural guarantee):

| # | Variant | Change | Real-data status |
|---|---|---|---|
| 0 | **Control** | current loss (must reproduce best=epoch 0) | baseline |
| 1 | **Dissipativity / passivity constraint (PRIMARY)** | constrain the ANN block dissipative (`‖𝒜‖≤1` / incremental-passivity / NI), biases free for friction | **the deliverable** — knowledge-free, friction-permitting; needs the marginal-preserving relaxation (contribution) |
| 2 | **Structural integrator factoring (SECONDARY)** | ANN learns residual only; integrator as fixed layer (Tustin-Net), position loss kept | knowledge-free; pair with (1) on the residual |
| S | **DC guardrail (SIM-ONLY DIAGNOSTIC)** | `+ λ·‖mean_t(ANN_out on 0,2,3,5)‖²` | **NOT a real-data solution** — needs to know residual is zero-mean; use only to reconfirm the mechanism |
| C | **Velocity/accel loss (LAST RESORT)** | differenced-output loss | do NOT adopt without explicit go-ahead (top-of-doc constraint) |
| — | **Multiple shooting** | cross-window continuity `ρ·‖x_end−x_start‖²` | SUPPORT (conditioning), not the guarantee |

**Metrics per variant** (a fix must satisfy the first two; the rest explain why / test real-data
viability):

1. **Drift** — full free-run sim-RMS at the drift-visible horizon. Lower = better.
2. **Absorber still learned / friction-permitting** — ANN keeps zero-mean oscillatory content and X/Y
   authority, NOT zeroed. Measure: routed-row ANN output RMS retained; absorber-band (130–180 Hz) error
   → 2.2e-5 m floor. **A fix that kills drift by zeroing the ANN FAILS (D-103).** For real-data
   readiness: check the constraint permits a *velocity-dependent* (dissipative) force, not just AC.
3. **Marginal-preserving** — the constraint must NOT add damping to the X/Y axes (check the identified
   X/Y pole/`‖𝒜‖` stays at the integrator, not pushed inside). This is where contraction-based variants
   (RENs/Györök) FAIL the criterion.
4. **ANN X/Y DC mean** (`d6` metric) — diagnostic of whether the bias was removed.

**Pass/fail logic**:
- Pass = drift ↓ **AND** absorber/friction still representable **AND** marginal mode preserved.
- The **control must be healthy** (reproduce epoch-0 revert / stay bounded) or the comparison is invalid
  (a comparison proves nothing until its control is well-behaved).
- Sweep the one knob per variant to find drift-removal with least capture loss.

**Scope honesty**: the PRIMARY variant (dissipativity with a marginal-preserving relaxation) is a
research/derivation task, not a config change — no existing method provides it off-the-shelf (§5, §9).
The DC guardrail (S) is the quick sim-diagnostic that reconfirms the mechanism while the real constraint
is derived. Show any harness skeleton before writing it in full.

---

## 8. Formal problem statement (literature-search brief)

**Setup.** Plant `ẋ = f_true(x,u)`, physics baseline `f_base(x,u;θ)`, learned parallel
augmentation `g_w(x,u)` added on routed rows `R`: `ẋ = f_base(x,u;θ) + S_R·g_w(x,u)`, `y=h(x,u)`.
Training: short windows length `T` (nf), encoder `ψ_η` sets `x̂(t0)` per window, minimize
simulation (free-run) error. **Deliverable metric: full-trajectory free-run error, `T_full ≫ T`.**

**Structural feature.** A subset of axes `F` are **free-integrator (marginally stable) modes**:
zero stiffness, `m_i q̈_i + c_i q̇_i = F_i` — velocity stable, **position a pure integrator
(pole at 0)**. Routing must include `F` (coupling): `R ⊇ F`.

**True residual.** `Δ = f_true − f_base`. Sim: zero-mean oscillatory absorber. **Real system:
also friction (`~sign(q̇)Fc`, dissipative), cogging (position-periodic), preload — state-dependent,
dissipative, possibly nonzero trajectory-mean.** The augmentation must represent these.

**Failure mode.** `g_w` can contain a component whose projection on the `F` force rows has nonzero
low-frequency/net content that the integrator maps to **unbounded `q_i` growth over `[0,T_full]`**,
while its position contribution over `[0,T]` is `O(½(p/m)T²)` — negligible — so the **short-window
simulation-error loss cannot penalise it** (Ribeiro: loss ill-conditioned on non-contractive modes).

**Requirements.**
- **R1 routing**: `g_w` acts on `F` (coupling) — cannot avoid them.
- **R2 no drift**: augmented free-run keeps `F` position modes **bounded** — `g_w` **non-destabilising /
  no net accumulating force** on `F`.
- **R3 real-data expressivity (crux)**: `g_w` must still represent **dissipative, state-dependent**
  residuals on `F` (friction, cogging). The constraint must **NOT** be `g_w=0` or `⟨g_w⟩_t=0` on `F`.
  Distinguish drift-causing (non-dissipative net) from legitimate (dissipative/state-dependent).
- **R4 interpretability**: under joint estimation, `g_w` must not negate baseline parameter
  directions (orthogonal projection) — coexists.
- **R5 realizability**: enforce by **architecture or added term**, since the short-window loss cannot
  see the drift.

**One-sentence problem.** Train/constrain a learned parallel augmentation on a plant with
free-integrator modes, under encoder-based simulation-error training, so the augmented free-run stays
**bounded on the marginal modes (R2)** while retaining expressivity for **dissipative state-dependent
residuals (R3)** — the learned term must be **non-destabilising on the free integrators without being
forced to zero on them** — keeping routing to those axes (R1) and compatible with parameter-
orthogonality (R4).

**Acceptance criteria (data-derived on real data).** (1) free-run drift below the measured noise
floor / a data-derived threshold; (2) residual captured (sim: 130–180 Hz absorber-band ↓; real:
friction/cogging residual ↓, not suppressed); (3) parameter recovery unharmed under joint estimation.

**Key reframing for the search**: the target is **non-destabilising (dissipative) on the marginal
modes**, NOT "zero output there" (a crude DC/mean penalty). Search on
stability/dissipativity-constrained learning for marginally-stable systems, plus the broader
control toolbox for adding uncertain/learned components to integrating plants without drift.

## 9. Literature consolidation (two targeted searches, 2026-07-09)

Two background searches: (A) exact/narrow — dissipative learned augmentation on integrator modes;
(B) broad control toolbox. arXiv IDs dated 2026 are agent-reported — **verify before citing**;
some formula forms came from abstracts (UNVERIFIED).

### Core theoretical finding (Search A)
Mainstream "stable learned dynamics" methods guarantee **contraction** (poles strictly inside the
disk), which **excludes a free integrator (pole at origin)**. The correct weaker condition is
**incremental passivity**: an incrementally passive `g_w` cannot inject net energy into perturbations
of the integrating state (no drift) **without forcing the integrator to converge**. This is the
precise R2-vs-R3 target.
- **Györök, Drenth, Verhoek, Péni, Schoukens, Tóth (2026), arXiv:2604.11421** — OUR supervisors, OUR
  LFR augmentation setup. Submitted 13 Apr 2026. **VERIFIED at primary source (2026-07-09).**
  Contraction condition (Corollary 13 / Eq. 40): `‖A‖₂ + L‖Bw‖₂‖Cz‖₂/(1−L‖Dzw‖₂) < ᾱ ≤ 1`, i.e. the
  map norm `‖𝒜‖₂ < 1` **strictly**. A free integrator has `‖𝒜‖ = 1` → violates it. The paper does NOT
  discuss integrators / poles at origin / marginally stable modes.
  **Sharper (and worse for naive reuse):** it does NOT assume the baseline is stable — the
  parametrization *forces* the whole augmented system to be contracting regardless of baseline. So
  applying it to our K=0 axes would **inject damping and destroy the zero-stiffness physics**, not just
  fail to cover them. → The needed relaxation is **`‖𝒜‖ ≤ 1` (marginal / incremental passivity), NOT
  `< 1` (contraction)** — a precise one-symbol delta on the supervisors' own condition whose feasible
  set *includes* the integrator. This is the clean contribution framing (R2).
- **Negative-Imaginary (NI) systems theory** (free-body: Mabrok et al. 2014, arXiv:1305.1079; original:
  Lanzon & Petersen 2008) — the ONLY classical
  framework native to **free-body/integrator poles** in colocated force→position systems (a precision
  stage). Constraining a neural `g_w` to be NI is essentially open (one linear Koopman attempt,
  arXiv:2305.04191, unverified).
- **Recurrent Equilibrium Networks** (arXiv:2104.05942) — **VERIFIED (2026-07-09), agent claim
  corrected.** ALL RENs are contracting (both C-REN and R-REN; Thm 1.2: R-REN "is contracting with
  rate α<ᾱ") — the incremental-passivity supply rate (Q=0,R=−2νI,S=I) is enforced JOINTLY with
  contraction, NOT as an alternative. So a REN **cannot natively be an integrator** (same `‖𝒜‖<1`
  exclusion as Györök), and the **free bias** that lets it represent friction (`b̃` "freely
  parameterized", so `f(0)≠0`) is exactly a nonzero-at-rest DC force → **reintroduces the drift**. RENs
  give contraction (can't host the integrator) + free DC bias (drifts it) — the two needs in tension.
  **Not a clean solution.** The earlier "passive-but-not-contracting REN" reading does not survive the
  full text.
- **Xu & Sivaranjani, Learning Dissipative Neural Dynamical Systems** (arXiv:2309.16032) — two-phase:
  train unconstrained → SDP weight projection → **retrain biases free (keeps Coulomb offset)**.
- **Port-Hamiltonian NNs** — force **zero output at equilibrium** → cannot represent static (Coulomb)
  friction; good for viscous only. This is the R3 failure mode of the "zero at equilibrium" family.

### The decisive parallel (Search B): adaptive-control parameter drift
Our problem is structurally identical to **parameter drift / bursting** in robust adaptive control (a
solved 1980s problem): the parameter update `θ̇=−Γεφ` is an integrator on error; a persistent small
disturbance drifts `θ` unboundedly, exactly as the ANN's DC output drifts the position integrator.
The canonical fixes map onto our DC-guardrail, more surgically than a blunt mean penalty:
- **σ-modification / leakage** (`θ̇=−Γεφ−σθ`, Ioannou & Sun 1996) = **L2 weight-decay** on the ANN.
  Trivial; small systematic bias (slightly under-estimates friction). Immediate first try.
- **Projection operator** (Lavretsky & Gibson, arXiv:1112.4232) = hard-constrain the ANN's DC to a
  data-derived feasible set, **zero steady-state bias inside the set** → keeps genuine friction.
  Agent's top pick; better than a soft penalty.
- **Dead-zone** (Ioannou & Tsakalis 1986) = freeze updates when residual < noise floor →
  **data-derived, defensible threshold** (matches our thresholds-from-data rule).
- **e-modification** (Narendra & Annaswamy 1989) = error-gated decay.

### Theory backing zero-DC-necessity + velocity-domain loss (Search B)
- **Zero-DC necessity is ELEMENTARY** (verified/corrected 2026-07-09): a pure integrator driven by a
  constant `u` gives `x(t)=x₀+u·t → ∞` (`∫const=∞`). No heavy theorem needed. **iISS** (Angeli, Sontag,
  Wang, IEEE TAC 2000, confirmed authoritative) is the general-nonlinear framing but does NOT directly
  apply here — a free-integrator position is not even 0-GAS (zero force → stays put, doesn't return to
  origin), so no input-energy robustness protects it; only zero net force keeps it bounded. Use the
  elementary integral as the justification, iISS as context — do NOT cite iISS as the load-bearing
  theorem.
- **Internal Model Principle** (Francis & Wonham, Automatica 1976): dual view — DC rejection needs an
  internal model (integrator) in the loop, absent in open-loop training → constrain DC to zero instead.
- **Ljung (1999)**: identifying integrating plants requires differencing/prefiltering =
  **velocity-domain loss**; explicitly, a long-horizon position loss *incentivizes* the model to add a
  DC force (the exact mechanism measured in d6). Backs fix (c).
- **Jacobian/spectral regularization** (arXiv:2602.04608, 2026, verify) — penalize combined-system
  Jacobian to bound long-horizon rollout error; complementary to the DC constraints.

### Verification synthesis (primary-source checks, 2026-07-09)

Three cites were checked at the primary source; two agent claims were corrected.

| Cite | Agent claim | Verdict | What changed |
|---|---|---|---|
| Györök 2026 (2604.11421) | strict contraction, excludes/damps integrator | **confirmed** | none — solid |
| RENs (2104.05942) | passive-not-contracting REN can host integrator, biases free → friction | **corrected** | ALL RENs contract; free bias IS the drift → not a solution |
| iISS (Angeli-Sontag-Wang) | formally proves zero-DC necessary | **downgraded** | necessity is elementary (∫const=∞); iISS doesn't apply (integrator not 0-GAS) |

**Stronger conclusion after verification:**
1. **Contraction-based augmentation is structurally UNFIT for this problem — as a family, not just Györök.**
   Any method guaranteeing `‖𝒜‖<1` either excludes the free integrator, or (if a free bias is added to
   represent friction) reintroduces the constant-at-rest DC force that drifts it. Györök 2026 and RENs
   both fail for this same reason.
2. **The precise requirement is a MIDDLE condition no single found paper provides:** permit
   **velocity-dependent forces that are zero at rest** (friction `~sign(v)`, cogging) while forbidding a
   **constant-at-rest** force. This is neither "contraction" (`<1`), nor "free bias" (`f(0)≠0`), nor
   "`f(0)=0`" (kills Coulomb). Its absence in the literature strengthens the contribution case.
3. **Zero-DC-on-the-free-axis-at-rest is the load-bearing necessity**, justified elementarily. The
   structural guarantee we want is `‖𝒜‖ ≤ 1` (marginal / incremental passivity / NI), which permits the
   integrator — the relaxation of Györök's `<1`.

**Verification status:** VERIFIED — Györök 2026 (full), RENs (full), iISS (existence/authority).
NOT yet verified — adaptive-control fixes (σ-mod/projection/dead-zone; canonical textbook, low risk),
**Negative-Imaginary / free-body theory (Mabrok et al. 2014 = arXiv:1305.1079, free-body NI; existence +
attribution online-verified 2026-07-10 — the only classical framework that natively allows `‖𝒜‖=1`;
exact LMI/residue conditions still to be read at the primary source)**, and all 2026 ML preprints.

### SUBNET/encoder-identification lineage search (Thread 1, 2026-07-09)

**Confirmed gap — airtight across the whole lineage.** Every encoder-based ID paper requires
**incremental exponential stability (λ<1)** and NONE relaxes it for integrators:
- SUBNET 2023 (arXiv:2210.14816), Condition 1 (quoted): `𝔼‖y_k−ỹ_k‖² < C(δ)λ^{k−k₀}, 0≤λ<1` — excludes
  poles on/outside the unit circle. MIMO: yes (explicit design goal).
- CT-SUBNET (arXiv:2204.09405) — same IEOS excludes the `s=0` pole; its `‖ẋ‖²` penalty is ad-hoc
  anti-blowup, not a stability guarantee.
- **Hoekstra/Györök 2026 encoder-init (arXiv:2602.13108)** — verbatim *"We assume a stable baseline
  model."* Reason is structural: the analytic encoder needs an invertible observability map, which a
  marginal baseline lacks — **a pure integrator has infinite memory → no finite encoder window / no
  reconstructability map exists.**
- SIMBa (arXiv:2311.03197) and stable-LPV NN (arXiv:2510.24757) prove "stable-by-design" ID exists but
  only by **excluding** the marginal region (Schur `|λ|<γ<1`) — same contraction trap, not a solution.

**The encoder is exactly what breaks on an integrator** (Forgione 2022, arXiv:2206.12928 + SUBNET
Condition 1): the encoder approximates the reconstructability map, which requires exponential forgetting
of the initial condition; a free integrator never forgets, so the map does not exist over a finite
window. (Ties to our d3: the encoder was clean on X/Y only because velocity is damped; the position
integrator is the un-reconstructable part.)

**NEW directly-applicable mechanism — Tustin-Net structural integrator factoring (arXiv:2408.12266).**
Hard-codes `position = ∫velocity` as a FIXED integration layer, so the NN learns only velocity
increments / accelerations — the free integrator is removed from the learned component, leaving only
asymptotically-stable residual dynamics for the ANN. This is the STRUCTURAL version of the
velocity-domain idea (vs penalizing DC): the ANN influences X/Y through **acceleration, not a direct
force offset on the position state**, so drift is impossible BY CONSTRUCTION while the integrator stays
exact (unlike contraction, which would damp it). Compatible with D-103 (ANN still acts on X/Y).
Caveat: Tustin-Net is a standalone architecture, NOT in the SUBNET lineage nor an augmentation paper →
transferable principle, not a drop-in.

**Three structural options now on the table:**
1. **Constrain the DC** (adaptive-control projection / dead-zone; Search B) — soft, keeps friction.
2. **Relax contraction → incremental passivity / NI** (`‖𝒜‖≤1`; the Györök-delta) — rigorous, harder.
3. **Factor the integrator out structurally** (Tustin-Net principle) — ANN writes accelerations only →
   drift structurally impossible, integrator physics kept exact. **Cleanest; new this search.**

### Control-theoretic ID of integrating systems (Thread 2, 2026-07-09)

**Decisive answer to "does the field allow a marginal eigenvalue BY CONSTRUCTION?" — NO for every
stable-by-design method, and the field's actual answer validates our architecture.** All stable-by-
design ID (Umenberger-Manchester interior-point, RENs, Miller-De Callafon / Lacy-Bernstein eigenvalue-
constrained subspace, SUBNET, stable-LPV) enforces STRICT stability and excludes the unit circle. The
field's real prescription for a pole at the origin is **structural, not a learning constraint**:
- **Ljung (PEM):** do NOT identify the integrator from data — build it into the model structure as a
  KNOWN fixed factor (`A` with a `z=1` root), fit only the residual, and use a prefilter `L(1)=0`
  (difference operator `Δ=1−q⁻¹`) so the criterion doesn't accumulate offset. → This is EXACTLY our
  setup: integrator in the fixed baseline, ANN learns the residual. The field says our separation is
  correct; the only failure is the ANN residual carrying a **DC gain** (nonzero force at ω=0).
- **Consensus triad** (synthesis of all four areas): (a) integrator in the baseline + strictly-stable
  learned block (our structure), (b) **closed-loop or force-domain training**, (c) **zero-DC-gain**
  constraint on the learned output. Contraction methods (RENs/SUBNET) correctly forbid the integrator
  IN THE LEARNED BLOCK — a feature, not a bug — provided the baseline carries it and the block has no DC.

**CORRECTION to the Ribeiro ill-conditioning claim (arXiv:1905.00820, Thm 1).** Earlier notes said
"exponential." Precisely: loss Lipschitz `O(L_h^{2N})`, gradient `O(L_h^{3N})` — **exponential only for
`L_h>1` (truly non-contractive)**. A free integrator is the borderline `L_h=1` case, where growth is
**POLYNOMIAL (N² loss, N³ gradient), not exponential.** So our short-window loss is ill-conditioned but
less catastrophically than a truly unstable system — short windows + continuity monitoring help, and a
DC bias still shows up as cross-segment continuity violation.

**DIDIM — closed-loop + FORCE-domain loss (Gautier/Janot 2013), and the reconciliation with the
supervisor's "controller in training" idea.** Robot ID solves the exact double-integrator problem by
minimizing `‖τ_measured − τ_simulated(θ)‖²` (TORQUE/force residual, not position), with both real and
simulated systems under the SAME closed-loop controller. Two mechanisms combine: (i) matching in the
**force domain** makes a DC bias visible DIRECTLY (not hidden by the integrator), (ii) closed-loop
stabilizes both sides so the free-run ill-posedness never arises. **This reconciles two things I earlier
treated as opposed:** the supervisor's "add a controller in training" instinct and my "velocity/force-
domain loss (fix C)". My earlier objection (a controller hides the DC) applies only to closed-loop
**position** loss; DIDIM uses closed-loop **force** loss, where the DC is exposed. So the robust
literature fix = **force/acceleration-domain loss** (≡ our fix C / Ljung prefilter), optionally with
closed-loop stabilization — a convergence of the supervisor's idea, Ljung, DIDIM, and Tustin-Net.

**Diagnostic (Rawlings 2025, arXiv:2406.03760):** fit an integrating-disturbance state `d_{k+1}=d_k+K_d
e_k` in innovation form; the identified `K_d` quantifies how much integral action the data needs — run
before/after augmentation to MEASURE the residual DC bias the ANN introduced. A quantitative check, not
a fix.

### Oomen / precision-motion PGNN lineage (targeted author search, 2026-07-09, abstracts verified)

TU/e precision-motion group (Oomen; and the Lazar/Bolderman PGNN line) works on physics + neural for
exactly this system class. **Key nuance: most of it is FEEDFORWARD control (learn the inverse dynamics /
a controller signal), NOT forward-model augmentation (learn the forward residual we do).** Related and
highly transferable, but not the same object — do not conflate.
- **Kon, Bruijnen, van de Wijdeven, Heertjes, Oomen (2022), "Physics-Guided Neural Networks for
  Feedforward Control: An Orthogonal Projection-Based Approach" (arXiv:2201.03308)** — VERIFIED abstract.
  Penalises the NN output **in the subspace of the physics model** via orthogonal projection →
  uniquely identifiable physical coefficients while the NN captures out-of-subspace residual (friction).
  **This is the SAME orthogonal-projection idea as Gyorok** (our R4/interpretability contribution),
  independently in the Oomen line, on friction-limited motion systems. Confirms our projection direction
  is mainstream in precision motion. Feedforward, not augmentation.
- **Bolderman, Butler, Koekebakker, van Horssen, Kamidi, Spaan-Burke, Strijbosch, Lazar (2023/2024,
  Control Eng. Practice), "Physics-guided neural networks for feedforward control with INPUT-TO-STATE
  STABILITY guarantees" (arXiv:2301.08568)** — VERIFIED abstract. Merges a physics layer + black-box NN
  in ONE model; a **regularization cost prevents competition between layers and preserves physics-
  parameter consistency** (again the projection/negation idea); and — most relevant to us — gives
  **sufficient conditions to IMPOSE ISS during training via less-conservative Lipschitz bounds on the
  NN.** Validated on a lithography linear motor with **mass-friction**. This is the closest thing found
  to "constrain the learned block's stability during training while keeping friction" — but ISS here
  bounds the FEEDFORWARD controller, and (like the others) does not treat a free-integrator plant mode.
- Adjacent Oomen work: LPV/position-dependent feedforward and ILC (de Rozario & Oomen, ACC 2017; GP
  position-dependent feedforward arXiv:2201.07511, 2202.00257); closed-loop NN training hazards +
  instrumental-variable fix (arXiv:2202.05337 — relevant since our data is closed-loop); add-on PGNN for
  interventional X-ray (arXiv:2303.07994, friction/cable forces).

**What this lineage gives us:** (1) strong independent confirmation that **orthogonal projection**
(physics ⟂ NN) is the accepted way to keep parameters identifiable while an NN learns friction (backs
R4 and Gyorok); (2) a concrete **ISS-during-training via Lipschitz bounds** recipe (Bolderman/Lazar) as
a candidate stability mechanism — Lipschitz-bounding is weaker/more general than contraction and worth
checking against our marginal-preserving requirement. **Gap remains:** all of it is feedforward-control
learning on stable inverse maps; **none imposes stability on a FORWARD augmentation of a
free-integrator plant** — our exact object is still open, now confirmed across the augmentation lineage
AND the precision-motion feedforward lineage.

### How this maps onto the §7 fixes
- **DC guardrail** → upgrade from a soft mean penalty to a **projection operator or dead-zone**
  (adaptive control), which keep friction (R3); σ-mod (=weight decay) is the cheap first try. iISS/IMP
  are the formal justification.
- **Multiple shooting / velocity loss** → Ljung prefiltering is the classical grounding for the
  velocity-domain loss.
- **Structural / thesis angle** → **incremental passivity / NI-constrained `g_w`** is the rigorous
  guarantee that covers the integrator (where contraction, incl. Györök 2026, fails). Likely the
  novel contribution: a passivity/NI-constrained augmentation for a marginally-stable baseline.

### DEEP-RESEARCH PASS (2026-07-13): confirms the framing + the d16 build target; but it is ENTIRELY about the ANN-DC prong and SILENT on the encoder prong (d17)

**Artifact**: `literature/stability-training/claude-deep-research-drift.md` (Claude deep-research review,
"Preventing Drift Along an Unexcited Direction in a Learned Residual"). Read in full 2026-07-13.

**What it CONFIRMS (external agreement with our position).**
- Our diagnosis is the right lens: the drift is the neural-residual instance of the classical
  robust-adaptive-control **parameter-drift-along-an-unexcited-direction** failure, cured ESTIMATOR-side
  (sigma-mod / e-mod / parameter-projection / dead-zone / null-space reg), never by restricting the model
  class. Exact formal object = the range/null-space split of the accumulated regressor info matrix
  (composite-learning result, arXiv:2408.01731). This is our §5m / D-105 reframe, now citable.
- The **gap is confirmed**: no single published method does all four (expressive + no marginal-mode drift
  + open-loop sim-error + interpretable). The CLOSEST is the SUPERVISORS' OWN LINEAGE, the constraint-free
  contracting-LFR augmentation (Györök/Drenth/Verhoek/Péni/Schoukens/Tóth, arXiv:2604.11421, 2026), and it
  FAILS our requirement precisely because its guarantee is **contraction**, which kills the marginal mode
  (adds the fake restoring force we keep rejecting). External confirmation of our expressivity-XOR-
  structural-guarantee impossibility.
- It **validates the d16 build target**: its Stage-2 recommendation is "project the residual onto the
  constant/low-frequency subspace of that coordinate and penalize its norm" = a direction-selective
  sigma-modification on the DC weight = exactly our direct near-DC frequency-selective joint pin (concept
  §7, after d16 refuted the Fisher-SVD constructor). Its own caveat that generic informativity/density
  weighting is "over state-input data density, not the specific DC direction of a free mode" is the same
  lesson d16 taught: the penalty must be direction-selective on the near-DC band, not a generic density prior.
- Its "do NOTs" match ours: no contraction/REN/stable-pHNN (adds restoring force, kills the marginal mode);
  no closed-loop to hide the bias (D-107, Jan); Jacobian/spectral rollout reg does not touch low-freq drift.

**THE KEY GAP (new organizing insight): PRONG 1 vs PRONG 2.** The review is entirely about a LEARNED
RESIDUAL drifting (manifestation 2 = the trained ANN's loss-neutral DC). It is SILENT on manifestation 1,
the **encoder velocity-reconstruction bias** that d17 (§3c) just measured to be the DOMINANT term in the
untrained free-run (1.45e-3 m on X, ~60x the absorber signal). The two prongs are orthogonal:
- **Prong 1 (encoder IC)**: differentiating the position window amplifies the 150 Hz ripple into a
  velocity bias on the K=0 axes -> bounded `tau*dv` offset (d17). Fix is IN-FRAMEWORK encoder
  conditioning (window length d10, better/nonlinear velocity map, or logged velocities), NOT covered by
  any method in the review. Keep the encoder (it is integral to SUBNET); condition its velocity step.
- **Prong 2 (ANN DC)**: the loss-neutral DC on the unexcited direction (d12) -> unbounded drift (d6).
  This is what the whole review addresses; the fix is the estimator-side direction-selective pin.
A perfect encoder alone does NOT fix prong 2 (d12); the pin alone does NOT fix prong 1 (d17). Both needed.

**Where we are already PAST its recommendations.** Its Stage 1 ("extend the horizon; if a feasible horizon
eliminates the drift, no penalty needed") is ALREADY answered NO by d8 (the DC is preferred through every
feasible nf). So we are at Stage 2 (the penalty), which it and we agree on. It does not advance past d16.

**New leads to primary-read (status: NOT yet primary-read).**
- **W-PGNN** (Liu, Tóth, Schoukens 2024, arXiv:2405.10429): in-framework data-informativity-weighted
  regularizer, "follows the baseline physics where data has low information content." Most on-point
  published mechanism for requirement (4); Eindhoven-line, citable. Caveat: its weight is a STATE-DENSITY
  KDE, not the DC/frequency direction, so it needs re-pointing to our near-DC target (the d16 lesson).
- **Constraint-free contracting-LFR augmentation** (arXiv:2604.11421): the closest integrated method;
  fails (b) via contraction. The supervisors' lineage, so cite as the nearest prior art the contribution
  extends past (marginal-mode-preserving where they contract).
- **Translation-equivariance / momentum conservation (Noether)**: the one STRUCTURAL cure that preserves
  the marginal mode (translation invariance -> no net DC force), valid ONLY IF the free-coordinate
  residual is velocity-only (friction), not position-dependent. A genuine class restriction (gated by the
  unknown-system rule), but a clean fallback if the X/Y residual can be argued velocity-only.

**Honest status.** The review's own caveats (W-PGNN weight formula via secondary summary; 2604.11421
partly unretrieved and its marginal-mode incompatibility inferred not stated; Fisher/GP equivalence
linear-Gaussian only) are noted; treat all three leads as `[verify-at-source]`, not primary-read.

## 10. Diagnostic file index (`scripts/gantry/diagnostics-drift/`)

| File | What it shows |
|---|---|
| `drift_common.py` | shared loader, 8-state truth EOM, baseline block, P-helpers, τ_X/τ_Y |
| `d1_input_selfconsistency.py` | input/frame correct (A1 rejected) |
| `d2_truex0_drift.py` | physics/true-x0 clean; velocity error settles to τ·dv |
| `d3_encoder_vs_truex0.py` | encoder x0 clean (A2 rejected) |
| `d4_freerun_vs_windowed.py` | re-seeding bounds accumulation |
| `d5_velaccel_fit.py` | differentiating removes the ramp (integrator is the mechanism) |
| `d6_ann_mean_force.py` | **trained ANN DC on K=0 rows; removing it cuts drift 133×** |
| `d7_validation_horizon.py` | in-horizon fine; drift enters ~0.5 s, invisible at nf=0.1 s |
| `d8_dc_visibility_horizon.py` | **D-109 RESULT: the windowed loss PREFERS the drift-driving Y-DC at every feasible horizon** — paired Δ(full−debias) NEGATIVE on Y at nf=400 (−2.2 SE, 73/119 windows) through nf=2000 (−1.8 SE); X neutral. Removing the DC that d6 showed collapses the 12 s drift makes the ≤1 s windowed fit WORSE. Moderate-nf training is refuted by SIGN, not just cost; the in-window benefit vs t² drift cost cross over above 1 s. dY row |mean|/rms=0.997 (pure DC) on the training distribution. Checkpoint: gantry_drift_last (post-D-101, lr=1.49e-8/nf=1400) |
| `d9_dc_compensation_shape.py` | **the Y-DC compensates the ENCODER-RE-INIT ramp of the training geometry itself (H1)**: debiased ensemble Y-error is LINEAR-dominated (\|b\|T=2.1e-5 m vs \|c\|T²=1.1e-6 m), offset a=+1.62e-6 matches the trained encoder's Y init bias +1.39e-6 (6 SE), slope b=+2.09e-4 matches the dY init bias +2.68e-4 m/s (within 22%); H2 curvature REFUTED (fitted −1.1e-4 vs −0.5·a_DC=+1.35e-3, wrong sign, 12×). The dY init bias is UNCHANGED from the untrained encoder (+2.675e-4 vs +2.681e-4) = init-scheme property; the Y-position bias grew 12× through training (co-training artifact). Mechanism: windowed training re-creates the encoder dY ramp EVERY window → the ANN's persistent negative dY-DC is the loss-optimal partial cancellation (~40% at 0.1 s) → in free-run the ramp happens ONCE (bounded, τ·dv) but the DC integrates FOREVER = the drift. Caveats: debiased-trend endpoint significance 1.9 SE (marginal); dY bias mean is 1.0 SE across V1 windows (realized mean on this set, not a firm population mean) |
| `d10_encoder_absorber_bias.py` | **step-1a closed-form encoder analysis + na fix designer.** P4 (the actionable result): mean dY init error drops 4.3× (+2.67e-4 → +6.3e-5 m/s) at na=27 (window ≥ 1 absorber period), plateau at na=40/53; paired improvement +2.0 SE; residual mean 0.3 SE ≈ zero. P1: the na=17 map under-responds at 150 Hz (0.62× ideal), longer windows attenuate the band (0.26/0.04×). P2 (reframes everything): V1's excitation is the NARROWBAND 130–180 Hz multisine — u band/ref ~1e12 — so essentially ALL data content sits in the absorber band; the "absorber in y, not in u" discriminator was ill-posed. P3: the clean "150 Hz line mis-read" attribution FAILS quantitatively (R²=0.14, predicted mean wrong sign) — the mean-bias mechanism is the map's broadband handling of the fully-narrowband excitation, not simple line aliasing. P0 caveat: rebuilt maps match pipeline to 0.36% (flagged); P3/P4-na17 used the pipeline encoder itself |
| `d11_trainset_encoder_bias.py` | **post-70558: the TRAINING set never had a mean dY encoder bias** — pooled +5.1e-5 m/s (0.69 SE) at na=17, +9.2e-5 (1.4 SE) at na=27, no paired improvement (−0.34 SE, n=1666); per-trajectory means mixed-sign, ≤2.8 SE. Retroactively explains 70558 (the na=27 "fix" changed nothing training could see) AND kills the mean-bias reward story on the training distribution. The V1 +2.7e-4 was a realized val-set mean. Since the ANN's dY output is ~CONSTANT (\|mean\|/rms 0.997), it also cannot be compensating per-window (zero-mean) errors — surviving hypothesis: the DC is a ~loss-NEUTRAL direction on the training distribution, injected as a shared-net byproduct of legitimate learning on the other rows (the §5m identifiability framing, reward story cleared). Discriminator: d8 paired test on TRAINING windows (preferred vs neutral) — if neutral, Layer-2 projection targets a genuinely data-silent direction |
| `d12_dc_trainwindow_preference.py` | **the d11 discriminator, VERDICT: NEUTRAL.** Paired full-vs-debiased on TRAINING windows (T3/T7/T10/T13 × 30, n=120, drifted checkpoint): pooled Δ/SE = +0.71, Y-channel +1.04, per-trajectory mixed signs (T13 even +2.0, DC slightly hurts there). d8's V1 preference (−2.0/−2.2 SE) was a val-set artifact. **The dY-DC is loss-NEUTRAL on the training distribution** → training wanders into it unopposed (shared-net byproduct), free-run integrates it. Diagnosis chain CLOSED: the fix is to pin/steer this proven-loss-neutral, proven-drift-causing (d6: 133×) direction — the Layer-2 projection premise now holds with direct evidence |
| `d13_scheduling_detune_tolerance.py` | **Layer-3 NECESSITY: REFUTED — the M(Y) detune channel is negligible on this system.** Truth-sim with M scheduled at Y+δY (12 s, V1 standstill + V3 ysweep): at the WORST measured drift (2.6e-2 m) the output deviation is 0.03× absorber RMS (V1) / 0.01× (V3); even δY=0.5 m (beyond the machine's whole ±0.3 m Y-range) stays BELOW absorber level (0.75×). Closed form: theta-mode \|df/f\|=0.64% at 2.6e-2; absorber mode exactly 0 (k_a/m_a Y-independent, sanity ✓). ε_tol lies beyond the physical Y range ⇒ **self-scheduling can stay; the drift→detune feedback exists (code-confirmed) but its gain is second-order; R5 reduces to R4 on this system; the biased Layer 3 is unnecessary.** CORRECTS the §5-R5 "Y is the HARDEST axis" framing: the conflict was asserted from the mechanism's existence, never quantified. Caveats: constant-offset probe (upper-bounds a ramp of same final value); property of THIS M(Y) (mild Y-dependence, off-diag ∝ m_h·Y vs ~54 kg diagonal), not a general theorem |
| `d14_datasilence_estimator_alignment.py` | **Layer-2 pre-build, v2 (operational-magnitude metric; v1's cross-frequency raw-S ranking was gain-confounded, caught at smoke).** Full run (T3/T10, n=40 paired): (1) **DC removal is NOT strictly loss-free**: full-mean_w removal costs +2.0 SE (+0.45% rel), dY-row-only removal +2.6 SE (+1.9%) — vs d12's neutral (+0.71 SE, n=120, 4 trajs). Reconciliation: the DC sits in a SHALLOW, window-set-dependent valley (\|effect\| ≤ ~2% rel, sign varies by trajectory set: d12 per-traj T3/T13 positive, T7/T10 negative); B1 risk = real but small and bounded, a soft β trades ≤2% window fit for the 133× free-run gain. (2) **The DC is a COUPLED multi-row object**: single-row (dY) removal hurts 4× more than full-vector removal — the row means partially cancel; build guidance: pin the JOINT measured direction, not per-row DCs. (3) The informed-side certification is VOID on this checkpoint (its learned 150 Hz content is ~0; the barely-trained ANN is DC-dominated) — moved to post-build validation (concept §9) where the model actually has band content. C-probes show coherent phase-dependent band interaction (±2.3 SE, tiny magnitude) — the machinery would detect band defense when present |
| `d15_pin_target_stationarity.py` | **pre-implementation Layer-2 check (distribution-shift failure mode): the pin target is PRACTICALLY STATIONARY along the 12 s drift.** Per-1s-block means of the d6 free-run ANN capture: the dominant dY-row DC changes only **−0.7%** first→last block while the state drifts to 2.6e-2 m; minor K=0 rows change more relatively (X −79%, Y +33%, dX −16%) but are 10–100× smaller in magnitude → joint-direction rotation ~1–2%. A FIXED joint-direction pin holds to first order off-distribution; iterative re-aiming (C2) stays a contingency, not a plan item. NOTE: the script's pre-declared verdict rule (20% AND 2-SE) flagged all rows "DRIFTING" via the significance branch — with 4000-sample blocks the SEs are minuscule and ANY real trend is "significant"; the materiality branch (0.7%) is the operative one. Rule conflated detectability with materiality — same trap as d14-v1, caught in reporting |
| `d16_pilow_spectrum.py` | **the last pre-implementation Layer-2 diagnostic: C1 MISALIGNMENT CONFIRMED for the Fisher/Gram Pi_low constructor — by structure, not tuning.** Loss-gradient Gram over a 12-probe basis (DC × 8 routed rows + sin/cos 150 Hz on dY/dθ), central FD at operational eps: the spectrum is DOMINATED by K=0 DC probes (DC-Y ev 290, DC-X 3.1, DC-dY 0.44 — integrator gain makes constant offsets the MOST loss-informed directions per unit amplitude) while band probes sit at 2.6e-3 … 1e-9. The drift direction (mean_w) ranks above 83% of the spectrum; ALL band directions rank BELOW it (band/drift ratio ~1e-9). ⇒ **a pure information-spectrum cutoff can never pin the drift while sparing the band on this system — the concept doc §3 Fisher-SVD construction is refuted as the primary target constructor (measured).** What separates drift from band is FREQUENCY (near-DC) + free-run consequence (integrator), not low information: the build target is the DIRECT measured joint-DC pin with near-DC frequency selectivity (concept §7's "optional frequency weighting" is now the required core). R1 preserved: the direction is data-measured; the DC-band selection is justified by the KNOWN baseline's K=0 structure, not the unknown residual. Smoke n=6 verdict (9 orders); full n=40 run for the record |
| `d17_msd_vs_encoder_decomposition.py` | **BASELINE-LEVEL decomposition (§3c): the UNTRAINED free-run error = encoder-IC bounded offset (dominant) + absorber residual + baseline replay offset; NOTHING drifts unbounded without the ANN.** Closed-loop-consistent identity `E = R + enc_IC` (all driven by `u_w`, vs `y_w`), on T1 with-MSD vs no-MSD. Excitation: no-MSD `u~0`, with-MSD 45 N RMS in 130-180 Hz (absorber anti-resonance breaks the loop cancellation, `u_w != u_n`), `delta_a` peaks 150 Hz. Findings: everything SETTLES (bounded, not drift); dominant = encoder IC on X = 1.45e-3 m (matches `tau_X*dvX`; axis trajectory-dependent, V1 was Y); Theta rings down (sprung yaw = the d6 "X1/X2 oscillation"); `R` (absorber) on Y ~3.8e-4 (ripple + baseline replay offset), `R~0` on X (absorber not coupled to X); `enc_IC(Y)` opposite sign, cross-coupling-dominated (9x its direct pred). Encoder x0 is the DOMINANT untrained error -> supports measured-ICs (supervisor); but bounded = the SEED d9 turns into drift, not the drift itself |
| `s2_hypothesis_figure.py` | the four-panel falsifiable hypothesis figure |
| `s3_openloop_multisine.py` | open-loop isolation: ANN drifts ~40× more than the true system |
| `make_drift_checkpoint.py` | trains a fast X+Θ+Y `_last` checkpoint for d6/d7/s3 |
