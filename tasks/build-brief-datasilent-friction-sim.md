# BUILD BRIEF: injected-friction sim + data-silent projection (open a CLEAN session with this)

**Purpose.** This is the opening instruction for a fresh session that BUILDS the documented plan (Route B:
fully-expressive augmentation that does not drift AND still learns friction). It carries the decisions and
the corrected understanding from the 2026-07-24 session so the new session does not repeat its detours.
Read this, then read the three plan docs below, then start at "FIRST ACTION".

## The plan (read these first, in order)
1. `docs/all-five-construction-spec.md` — Route B = fully-expressive ANN + **Layer 1** (long-horizon
   conditioning) + **Layer 2** (re-aimed orthogonal projection) + **Layer 3** (Y-scheduling). Build order §3/§10.
2. `docs/augmentation-validation-design.md` — the decisive **3-model comparison** on injected dynamics and
   the §6 per-case protocol, §7 excitation ablation, §8 metrics (all data-derived, judged vs the noise floor).
3. `docs/data-silent-regularization-concept.md` — Layer 2 = Gyorok's projection RE-AIMED at the drift
   direction. **Read §7 carefully:** the d16 measured update REFUTED the naive "pin the low-information
   direction" (on the K=0 axes the DC/drift direction is HIGH-information because the integrator amplifies it).
   **Projection target = the measured joint-DC direction (d14) with near-DC frequency selectivity.**

## Corrected understanding from the 2026-07-24 session (do NOT relitigate)
- **The cure is MULTI-LAYER, not one route.** No-drift comes from the ESTIMATOR via (Layer 1) CONDITIONING
  [long-horizon / multiple-shooting / ARTBP = unbiased truncated BPTT, removes the truncation-bias DC, mostly
  X] + (Layer 2) PROJECTION [pins the residual unexcited near-DC direction; needed because the DC persists
  ~1/nf at every horizon, SLURM 71013, so conditioning alone is insufficient] + (Layer 3) Y-SCHEDULING [the Y
  anti-damping gain, which ARTBP does not fix]. Never say "orthogonal projection is the only route".
- **The optimizer-only shortcut is ELIMINATED.** Swapping SGD for Adam is a single-knob R2<->R4 tradeoff
  (SGD-slow = no-learn/low-drift; Adam-fast = learn/diverge); it is NOT a fix. The 2026-07-24 SGD-vs-Adam
  probes (`scripts/gantry/baseline-null/r2_fit_probe.py`, gain_vs_dc, pole_check) were an OFF-PLAN detour;
  their only lasting value is the reusable truth-injection + windowed/free-run/eigen machinery.
- **R3 baseline poles confirmed:** baseline X/Y one-step |lambda|=1 (marginal), none >1, across the Y range
  (`pole_check.py`). Use the same autograd-Jacobian eigen-check for R3 on the trained model.

## Decisions already made (do not re-ask)
- **Light harness, sim-first** (user, 2026-07-24). Build the mechanism test on the cheap controllable harness
  first; the production pipeline + real Telica is the later DELIVERABLE (ASK-gate). "Sim validates the
  MECHANISM; held-out real-data free-run BFR is the deliverable" (validation-design §9).
- **Injection = DISSIPATIVE Coulomb** `F = -c*tanh(v/eps)` on the X/Y velocity rows (stable, opposes velocity,
  still nonlinear, carries net impulse under net motion). NOT a scaled random ANN (that is anti-damping and
  ill-conditions the BPTT; see lesson `inject-a-well-conditioned-residual`).

## Build order (each step maps to validation-design §10 / all-five §3)
1. **Injected-dissipative-friction sim harness** (selectable dynamics type + excitation). VERIFY the §6
   preconditions BEFORE any method claim: (a) truth free-run BOUNDED; (b) friction is EXCITED (loss
   sensitivity / Fisher wrt the routed X/Y force is non-negligible on the driven records); (c) an
   UNCONSTRAINED ANN (Adam) LEARNS it (windowed nf-RMS -> floor) but DRIFTS in free-run (reproduces the
   "Unconstrained ANN" row: C3 pass, C1 fail). Metric = WINDOWED nf-RMS for learning (R2), free-run ENVELOPE
   ratio for drift (R4); keep the eval-window magnitude FIXED across compared runs.
2. **Excited/null subspace decomposition** for the nonlinear residual; unit-verify on a known LINEAR case
   first (validation-design §10.2). Target from d14/d16 = the measured near-DC/joint-DC direction.
3. **The 3-model x library grid + excitation ablation** (validation-design §4-8): Unconstrained ANN (drifts) /
   Net-impulse block (no-drift but FORBIDS friction) / **Data-silent-projected ANN** (no-drift AND learns
   friction = the claim). Report §8 metrics vs the measured noise floor. The decisive cell is
   data-silent-projected vs net-impulse on "learns dissipative Coulomb".
4. **Only then real Telica** (held-out free-run BFR) = the deliverable (ASK-gate).

## Reusable machinery / existing scaffolding (do not reinvent)
- `scripts/gantry/baseline-null/r2_fit_probe.py` — truth-injection + windowed/free-run loops (swap the
  injection to dissipative Coulomb; drop the SGD framing).
- `scripts/gantry/baseline-null/pole_check.py` — autograd-Jacobian R3 eigen-check (|lambda|=1).
- `scripts/gantry/diagnostics-drift/` — `drift_common.py`, `d14_datasilence_estimator_alignment.py` (the
  joint-DC projection-target diagnostic, production pipeline), `d16_pilow_spectrum.py` (why low-information is
  refuted), `d6_ann_mean_force.py`. `scripts/gantry/gantry_dynamic/zeromean_pin.py` (a DC-pin reference).
- ARTBP (Layer-1 conditioning): `scripts/gantry/ARTBP/` (train_artbp, feedback_instrument, test_efolding).

## Guardrails / lessons (active)
- `inject-a-well-conditioned-residual`, `background-job-unreliable-run-foreground` (run must-finish jobs
  FOREGROUND, sized under ~500s), `no class restriction as the DELIVERABLE`, `X/Y stay in ANN routing`,
  per-axis + noise-floor thresholds, `velocity LOSS is last resort`. Full set: `tasks/lessons.md` (read first).
- Every training run with a new hypothesis gets a §12 run-table row BEFORE launch (D-090).

## Open supervisor decisions (STOP/ASK, do not decide autonomously)
- **Y-scheduling: exogenous/measured vs self-scheduled** (all-five §5, the keystone of Layer 3).
- **R4-empirical acceptance** (is demonstrated no-drift acceptable as the deliverable?).
- **Real Telica / production training** venue + budget.

## Hard placement constraints (user, 2026-07-24)
- **All NEW code lives in `scripts/gantry/datasilent-friction-sim/`.** You may READ / import / adapt the
  reusable machinery (`baseline-null/r2_fit_probe.py`, `pole_check.py`, `diagnostics-drift/`) but write new
  files in the new folder; do not edit those in place.
- **`model_augmentation/` is READ-ONLY** (project hard constraint). Conform to it, never modify it.

## FIRST ACTION for the clean session
Build step 1 in `scripts/gantry/datasilent-friction-sim/`: the injected-dissipative-Coulomb harness + the
three §6 precondition checks (reuse `r2_fit_probe.py` machinery, adapted into the new folder). Log a §12
run-table row first. Confirm (a) truth bounded, (b) friction excited, (c) unconstrained Adam-ANN
learns-but-drifts. That establishes the correct baseline the projection must beat, and fixes the
anti-damping-injection confound from 2026-07-24.
