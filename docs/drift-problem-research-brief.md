# Drift Problem — Research Brief (for an analysis + literature session)

**Purpose.** Hand this to a fresh session. Task: (1) understand the problem below, cross-checking it against
the iteration history in the gantry subfolders (§3, pointers included); (2) then do a DETAILED extended
online literature search on the questions in §4; (3) synthesize a cited report. A LATER session will build an
iterative-learning testing harness from that report (run overnight). This brief is the problem + evidence map;
it does NOT prescribe the solution.

---

## 1. The problem in one paragraph
We augment a physics-based model of a dual-gantry motion system with an additive neural-network (ANN)
correction (the "augmentation"). The X and Y **position** states are **marginal integrators** (discrete pole
≈ 1, "K=0" — no restoring stiffness). The deliverable metric is long **free-run (open-loop) simulation**. In
free-run the augmented model **DRIFTS** away from truth on X/Y: the ANN acquires a spurious, near-**constant
(DC)** output on the velocity rows, and the marginal integrator double-integrates that DC into an unbounded
position ramp. This is an **estimator/optimizer artifact, not learning** — it happens even in a perfect-model
NULL (nothing to learn, correct ANN output ≡ 0). The augmentation must SIMULTANEOUSLY remain able to learn
real nonlinear dynamics (e.g. Coulomb friction) — that expressivity is non-negotiable — and friction carries a
net impulse that is ALSO a velocity-DC. So the spurious drift-DC and the wanted friction-DC live in the SAME
subspace, and this session found they are **not separable by output direction**.

## 2. Why it happens (the CONVERGED diagnosis — measured chain, do not re-derive)
There are TWO components; on REAL data the first dominates. Both put a near-constant force on the velocity rows.
- **(A) Encoder-init-compensation DC (dominant on real data; the closed causal chain, `drift-diagnosis-status.md`
  §3b + `gantry-zero-mean` V3/V4 + `diagnostics-drift` d6/d9/d12).** The SUBNET encoder over-estimates dY by
  ~**+2.7e-4 m/s** at every window start (an init-scheme property, measured on train AND val — NOT
  training-created). Windowed training re-creates that init error at every window start, seeding a systematic
  constant-velocity **ramp** inside the 0.1 s / 400-step window (V4: linear, R²=0.995 on K=0). The ANN's
  loss-optimal response is a **persistent DC** (~−2.7e-3 m/s² on dY) that partially cancels the in-window ramp
  — and the windowed loss slightly **PREFERS** it (d8/d9/d12). In free-run the encoder error is **one-time**
  (bounded τ·dv offset) but the learned DC applies **every step** → unbounded drift. Birth-of-the-DC probe (V3):
  the dY DC appears ~step 13 and **reproduces in sign across seeds** = a systematic gradient push, not diffusion.
- **(B) Adam displacement DC (present even in a perfect NULL; `baseline-null`).** With a perfect model / nothing
  to learn, Adam still displaces the ANN by **exactly 3.48·lr** per step and free-run drift ∝ lr (→ floor at
  lr=0); SGD builds ~2000× less. Curvature finding (`curvature_sensitivity.py`): the windowed loss is actually
  **STIFFEST** on the integrator DC (~1e5× Theta), but the operating point is **not** the minimizer — curvature-
  blind Adam parks ~lr away. This null DC (~1e-7) is ~40× below the real-data DC, i.e. a secondary contributor.
- **CRITICAL (d16, refutes a tempting framing):** the drift-DC is **NOT low-information / data-silent** — on a
  K=0 axis a DC is **HIGH-gain and the MOST loss-informed** direction per unit amplitude (Gram 0.44–290 for DC
  probes vs 2.6e-3–1e-9 for band probes). A Fisher/low-information cutoff therefore pins the absorber band
  BEFORE the drift — the opposite of the intent. **What separates the drift from learnable dynamics is
  FREQUENCY (near-DC) + FREE-RUN CONSEQUENCE (the marginal integrator) + SOURCE (it compensates a one-time init
  error), NOT information content.**
- **Two sub-problems (may unify).** *Problem 1* = the velocity-DC above → ~linear X drift (ARTBP reduces it).
  *Problem 2* = a slow GROWING Y oscillation (~0.085 Hz, seconds-timescale), shown NOT to be the LPV `M(Y)`
  self-scheduling feedback loop (teacher-forcing true Y changed nothing), refined toward a marginal/anti-damping
  displacement on the double-integrator (pole ~1, possibly >1) — likely needs a stability constraint (D-117).

## 3. What is OPEN, and the finding that sharpens it (this session, 2026-07-24)
The planned Layer-2 cure is a re-aimed **orthogonal projection** (Gyorok) pinning the ANN's output along the
**measured near-DC / joint-DC drift direction** (d14) with near-DC frequency selectivity (d16), leaving other
directions (friction) free. Two prior strong results frame it: **(dB) a structural bounded-integral / high-pass
factoring** (ANN output = derivative of a bounded function → position stays bounded, pole at origin) cut X drift
**1100×** (2.19e-3→2.05e-6) while preserving the 150 Hz absorber — but it removes the DC, so it shares the
friction tension below. **ARTBP** (unbiased long-horizon gradient) reduced the Problem-1 DC variance ~4.6× (5-seed).

This session tested the REAL projection operator (`orth_projection.py`, `V=β‖Qᵀf‖²`) on the perfect-model NULL and
found (robust, mechanistic): **a rank-1 direction pin is dodged** — pin the measured drift direction
`(dX,dY)=(−0.70,+0.71)`, and the optimizer displaces into the ORTHOGONAL DC direction `(−0.71,−0.71)` and drifts
anyway; a **soft penalty under Adam saturates in β** (bit-identical 1e3→1e12, since the Adam step →
`lr·sign(penalty grad)`); and **pinning ALL velocity-DC = the mean penalty = kills friction** (+41%→+4%). (Caveat:
that NULL testbed was noise/transient-dominated — DC flipped sign each epoch — so it is not a clean pass/fail; use
the deterministic `baseline-null` one-step protocol.) **THE CRUX, restated correctly:** the spurious drift-DC and
the friction's net-impulse DC both live in the velocity-row DC subspace and are BOTH loss-informed; they are NOT
separable by output direction, and NOT by raw information (d16). The candidate separators are (i) FREQUENCY /
near-DC vs the friction's spectral signature, (ii) FREE-RUN CONSEQUENCE (does it accumulate unboundedly), (iii)
SOURCE (the drift-DC compensates a one-time encoder-init error — so **fixing the encoder-init bias may remove the
reward for the DC at the root**, a class of fix that constrains the ENCODER, not the ANN output). Which of these
actually separates them, and by what mechanism, is the open question.

## 4. Literature questions for the extended online search (the core deliverable)
Search deeply and return cited, verified findings on:
1. **Implicit bias of Adam vs SGD on flat / degenerate / weakly-identified directions** of a loss (why Adam
   wanders near-zero-gradient directions; whether it can be curbed SELECTIVELY on a known subspace). Any
   results specific to marginal / integrator / near-unit-root modes.
2. **Separating a spurious near-DC bias from a genuine DC-carrying residual on a marginal/integrator mode when
   BOTH are loss-informed** (NOTE: on this system a Fisher/low-information cutoff is REFUTED — the drift-DC is
   HIGH-information/high-gain, d16). So the separator is NOT information. Search: frequency-domain / near-DC
   regularization vs a residual's spectral signature; free-run-consistency (multi-step) penalties that price the
   accumulation; and whether ANY estimator-side prior can down-weight a near-DC bias while keeping a physical
   friction impulse. Also survey how the non-identifiable / unexcited subspace is even ESTIMATED for a NONLINEAR
   (ANN) residual (Fisher-SVD vs ensemble/bootstrap disagreement vs GP predictive variance) — for completeness,
   noting the info-cutoff is refuted here.
3. **Training neural-ODE / recurrent / SUBNET simulation models that contain a MARGINAL (pole-1, integrator)
   mode without free-run drift**: exposure bias, multiple shooting, state-consistency / free-run-consistency
   regularizers — AND their known failure as the pole → 1 (TBPTT bias ~ `ρ^K/(1−ρ)` diverges).
4. **Separating a data-silent DC / bias from a genuine data-informed restoring or dissipative force on a free
   integrator**: system-identification of near-integrator / marginally stable systems; the net-impulse / DC
   identifiability problem; how friction (Coulomb, carries net impulse) is identified WITHOUT being confounded
   with an integrator bias.
5. **Physics-augmented / grey-box ML that PRESERVES a marginal pole while learning residual dynamics** and
   keeps full expressivity: port-Hamiltonian with R=0, "do-no-harm" / W-PGNN (Liu–Toth–Schoukens),
   orthogonal-projection regularization (Gyorok et al. L4DC 2025), cyclo-dissipativity / indefinite storage.
   Which of these keep pole=1 AND do not forbid a DC-carrying dissipative residual?
6. **Optimizer-side selective fixes**: gradient / natural-gradient / K-FAC projection restricted to a subspace;
   removing the update component along a known null direction; whether a curvature-aware step (SGD-like) can be
   applied ONLY to the marginal/null direction while Adam runs elsewhere.
7. **ROOT-CAUSE angle — observer/encoder initialization bias.** The dominant DC here is the ANN COMPENSATING a
   systematic **encoder/state-estimator initial-condition bias** (SUBNET/deep-SS encoders; the encoder
   over-estimates a velocity at every window start, and windowed training rewards a compensating DC). Search:
   bias/consistency of learned state-estimators & encoders in SUBNET / deep state-space / neural-ODE-with-encoder
   models; init-error propagation on marginal modes; multiple-shooting or learned-initial-state schemes that
   remove the per-window init bias; teacher-forcing / free-run mismatch (exposure bias) as the root, not the ANN.
   Is the correct fix to remove the init bias (constrain the ENCODER) rather than the ANN output?
8. **Bounded-integral / Tustin-Net / high-pass output factoring** for a free integrator: parametrizing the ANN
   output as the derivative/difference of a bounded function so the integrated position stays bounded and the
   pole stays at the origin (this cut X drift 1100× here, `dB`). KEY question the search must resolve: can such a
   factoring be made to still admit a genuine DC-carrying dissipative friction (net impulse), or does it — like a
   mean/zero-mean penalty — necessarily forbid it? Tustin-Net (integrator-factoring architectures), bounded-real
   / high-pass residual parametrizations, and any method that bounds position WITHOUT forbidding a net impulse.

## 5. Constraints the solution space must respect (from the project)
- **Full expressivity is non-negotiable** (the true residual is unknown; any forbidden class might BE it). A
  for-all-weights no-drift GUARANTEE is proven incompatible with full expressivity → no-drift must be
  **training-conditional** (estimator-side), not a hard model-class restriction.
- **Must preserve the marginal pole** |λ|=1 on X/Y (no artificial damping; Lipschitz/contraction caps are
  ruled out — they pull the pole inside).
- **X and Y stay in the ANN routing** (Theta-only routing is never the deliverable).
- Judge everything **per-axis**, against the **measured noise floor**, never an oracle.

## 6. Iteration map — what has already been TRIED across the gantry subfolders
> Filled from a repo survey; each entry = approach → outcome → where the evidence lives.

### 6a. Fixes TRIED and their verdicts (do NOT re-propose the ruled-out ones without a new angle)
| Approach | Where | Verdict |
|---|---|---|
| **Longer fixed BPTT horizon (nf)** | `diagnostics-drift/d7,d8`; Optuna 69399; curriculum 70903 | **FAILS by sign** — the windowed loss PREFERS the DC at EVERY horizon (nf up to ~4 s); best-epoch never beats epoch 0; memory wall at nf≥4000. |
| **Lower lr / lr tuning** | `baseline-null/lr_sweep`; Optuna 69399 | **Not the lever** — drift ∝ lr by construction; can't train at lr→0. |
| **SGD instead of Adam** | `baseline-null`; `gantry-zero-mean` | **Not a standalone fix** — SGD builds ~2000× less DC but underfits; single-knob R2↔R4 tradeoff. |
| **Naive long-horizon / multiple shooting** | this session step-3a; Optuna 69399 (MS best=ep0) | **Made it worse** — BPTT gradient explodes through the marginal mode (bias ~ρ^K/(1−ρ), ρ→1). Conditioning support, not a guarantor. |
| **Lipschitz / contraction cap / strict passivity / RENs** | `passive-augmentation`; `baseline-null/diagnostics-literature.md` | **Ruled out** — pulls the marginal pole strictly INSIDE, destroys the genuine pole-1 integrator (criterion-3 fail). |
| **DC / zero-mean mean penalty** | `diagnostics-drift/d6`; `gantry-zero-mean` | **Works in sim (zero-mean absorber) but DEMOTED** — on real data friction carries a net-DC, so the penalty suppresses real friction. Knowledge-free requirement violated. |
| **Bounded-integral / high-pass output factoring (Tustin-Net principle)** | `diagnostics-drift/dB` (proven), `dC` (unbuilt) | **STRONG on drift** — X drift 2.19e-3→2.05e-6 (**1100×**), Y ~1e-4, 150 Hz absorber preserved. BUT removes the DC → shares the friction-net-impulse tension (untested on friction). `dC` (train an ANN with this parametrization) NOT yet built. |
| **ARTBP (unbiased truncated-BPTT long-horizon gradient)** | `ARTBP/` Phases A–E (5-seed grid DONE) | **Reduces Problem-1 DC** (variance ~4.6× smaller, poly6). Gate-2 (converged 20-epoch fit) pending. Does NOT fix Problem 2 (Y). |
| **Orthogonal projection (Gyorok re-aim to measured joint-DC)** | `orth-projection/` step0–8b (operator built + formula-validated); d14/d16 (target measured); this session step-4 | **Built, integration open.** This-session NULL test: rank-1 direction pin is DODGED (drift moves orthogonal); soft penalty saturates in β under Adam. Direction-based separation of drift vs friction is in doubt (§3). |
| **Negative-Imaginary (marginal, pole-1-preserving dissipativity)** | `baseline-null/diagnostics-literature.md` survey | **Viable in principle, UNBUILT** — classical theory keeps pole=1; a learned/LPV NI realization is open (a possible thesis contribution). |
| **Stiffness-selective / Theta-only routing** | D-068 / §11 problem-log | **Rejected (D-103)** — X/Y must stay in the ANN routing; Theta-only is not the deliverable. |
| **Velocity/accel-domain training loss** | standing constraint | **LAST RESORT, supervisor-gated** — literature keeps converging on it; positioned last until position-based fixes are exhausted. |

### 6b. What is CONFIRMED (measured, closed — do not re-diagnose)
- Model is correct: baseline reproduces its own data to ~1.2e-7 m with the true x0; the ~1e-5 free-run floor is
  the SUBNET encoder-init error (`diagnostics-drift/d2,d17`; `baseline-null/floor_horizon`).
- The DC IS the drift driver: subtract the measured DC → Y drift 2.59e-2 → 1.95e-4 (**133×**) (`d6`, `drift-visual/f05`).
- The DC is universal in magnitude, arbitrary in sign across 9 checkpoints (`drift-visual/f07`).
- Physics carries NO DC the baseline lacks (all mechanisms ≤1e-7, 5+ orders below the ANN DC) (`gantry-zero-mean` V1f).
- Problem 2 is NOT the M(Y) self-scheduling loop (`ARTBP/test_self_scheduling`: all conditions ~1.00×).

### 6c. Open threads
- Build + integrate Layer 2 (measured joint-DC + near-DC frequency projection) and test it CLEANLY (deterministic
  testbed), incl. the friction-preservation test.
- The bounded-integral `dC` (train an ANN with the factoring) is unbuilt — and its friction-compatibility untested.
- Confirm Problem 2 marginal-vs-exponential on a 2nd checkpoint; decide the stability-constraint route if needed.
- The ROOT-CAUSE angle (fix the encoder-init bias so the loss stops rewarding the DC) is under-explored.

## 7. Key file pointers (start here)
- Converged diagnosis + all failure modes + §12 run table: `docs/gantry-augmentation-problem-log.md`,
  `docs/drift-diagnosis-status.md`.
- The clean optimizer-displacement evidence: `scripts/gantry/baseline-null/README.md`, `lr_sweep.py`,
  `gain_vs_dc.py`, `floor_horizon.*`, `diagnostics-literature.md` (a prior cited cause/cure landscape).
- Adam-vs-SGD DC / zero-mean investigation: `scripts/gantry/gantry-zero-mean/README.md` +
  `RESULTS-2026-07-17-dc-drift-diagnosis.md`.
- ARTBP (unbiased long-horizon) + Problem-2 (Y) instruments: `scripts/gantry/ARTBP/README.md`,
  `feedback_instrument.py`, `test_efolding.py`, `test_self_scheduling.py`.
- Layer-2 concept (re-aimed projection) + limits: `docs/data-silent-regularization-concept.md`,
  `docs/data-silent-regularization-concept-limits.md`.
- The real orthogonal-projection operator: `model_augmentation/fit_systems/orth_projection.py`,
  `scripts/gantry/gantry_dynamic/orth_penalty.py`; formula-validation ladder `scripts/gantry/orth-projection/`.
- This session's projection-vs-null test: `scripts/gantry/datasilent-friction-sim/step4_orth_projection_null.py`,
  `PROGRESS.md`; and the run-table STEP-4 row.
