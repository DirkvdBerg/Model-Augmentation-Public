# Session Handoff — diagnosis CLOSED, Layer-2 pre-build COMPLETE, nf-curriculum IMPLEMENTED & ready to launch, NEW supervisor feedback incoming

_Archived 2026-07-15 from `tasks/handoff.md` (written 2026-07-13) per the CLAUDE.md archival rule._

**Written**: 2026-07-13. _Prior handoff archived to `archive/sessions/2026-07-12-handoff-litsearch.md`._
**Why a new session**: the user received NEW supervisor feedback (content not yet in this handoff — the user
will state it). Everything below is the state that feedback lands on. Do NOT re-derive it.

## State in one paragraph
The X/Y drift diagnosis is CLOSED end-to-end (d6–d16, status doc §3b). Mechanism, all measured: the trained
ANN emits a near-CONSTANT (DC) force on the K=0 X/Y velocity rows; those axes have no stiffness (free
integrators), so a constant force integrates to unbounded position drift (12 s free-run: Y → −2.9 cm vs the
~22 µm absorber signal). Removing the DC collapses drift 133× (d6). The DC is loss-NEUTRAL on the training
distribution (d12, n=120), so training wanders into it and the windowed 0.1 s loss is structurally blind to
it (d7: drift only enters the RMS past ~0.5 s). Longer-nf training is refuted BY SIGN as a DRIFT fix (d8:
the loss prefers the DC through nf=2000) — but longer nf DOES help the separate weak-SIGNAL/learning problem
(§7). Literature gap confirmed at primary-read depth incl. from the supervisors' own lineage (D-108/D-110).

## THE decision that now gates the build (was two, d13 collapsed one)
1. **Is empirical R4 acceptable as the deliverable?** For-all-weights R4 is PROVEN incompatible with full R2
   expressivity; Route B = R4 demonstrated, not guaranteed (`all-five-construction-spec.md` §4). THE open call.
2. ~~Y-scheduling exogenous vs self-scheduled~~ **ANSWERED by d13**: the M(Y) detune channel is measured
   30–100× below materiality even at worst-case drift → self-scheduling STAYS, no Layer 3, R5 reduces to R4
   on this system. Decision 2 is now a one-line measured statement, not a supervisor call.

## What's IMPLEMENTED and READY (not yet launched)
- **nf-CURRICULUM in `scripts/gantry/gantry_optuna.py`, `MODE='curriculum'`** (2026-07-13). Full X+Θ+Y
  routing (D-103 deliverable), lr=1e-7 FIXED, joint=False, nominal θ, free ANN (no orth). Warm-started ladder
  `NF_LADDER=[(400,8),(800,7),(1600,6),(2000,5)]` (~26 epochs, ~12 h budget the user gave). ONE build_model;
  between rungs `checkpoint_load_system('_last')` recovers the trained weights (fit reloads `_best`=epoch0 on
  the drift route). Selector stays sim-RMS (windowed measure is stride=1 → too slow at long nf). lr fixed
  (per-rung lr ignored by fit, D-101); overshoot watched REACTIVELY via the train nf-RMS print, not scheduled.
  Config validated (compiles + CFG/hp check). D-090 run-table row written (problem log §12, outcome pending).
  Orth-smoke config preserved behind `MODE='orth_smoke'`. **Pre-declared readings:** does train nf-RMS fall
  across rungs (ANN learns absorber)? do windows and full sim-RMS improve TOGETHER or SPLIT (split → drift
  separate from signal, Layer 2 still needed; together → nf-conditioning solved it)? Launch check (70799
  lesson): confirm the log's Configuration block prints `routing=(0..7)`, `lr=1.0e-07`, `RUNG 0: nf=400`.

## Layer-2 build target — FULLY SPECIFIED by the pre-build diagnostics (no design guesswork left)
- **d13 (Layer-3 necessity): REFUTED** — see decision 2 above. Not a build item.
- **d12 + d14v2 (pinning safety): qualified pass** — removing the DC costs ≤ ~2% window RMS (2.0–2.6 SE),
  a SHALLOW set-dependent valley; soft-β trade (≤2% cost vs 133× free-run gain) strongly favorable. The DC is
  a COUPLED multi-row object → pin the JOINT measured direction, not per-row (single-row hurts 4×).
- **d15 (pin stationarity): PASS** — the dY-DC pin target changes only −0.7% across the full 12 s drift; a
  FIXED joint-direction pin holds off-distribution (iterative re-aiming is a contingency, not planned).
- **d16 (estimator choice): Fisher-SVD REFUTED as the target constructor** — on K=0 axes the integrator gain
  makes DC directions the MOST loss-informed, so a low-information cutoff pins the absorber band before the
  drift (C1 misalignment, by structure). **Build the DIRECT measured joint-DC pin with near-DC frequency
  selectivity** (concept §7 "optional frequency weighting" is now REQUIRED). R1 preserved: direction is
  data-measured; DC-band selection follows the KNOWN baseline's K=0 structure, not the unknown residual.
  Ensemble disagreement stays a candidate for RESIDUAL silent directions only (not refuted by d16).

## What is DOCUMENTED where (do NOT re-derive)
- **Diagnosis + all diagnostics**: `docs/drift-diagnosis-status.md` §3b (causal chain) + §10 index (every
  d1–d16 row, one-line result each). D-109 outcome addendum.
- **Supervisor drift-demonstration figures** (all in `simulations/gantry_subnet/diagnostics/`, explained to
  the user 2026-07-13): `d6_ann_mean_force_gantry_drift_last.png` (the drift + the DC counterfactual, red vs
  blue, Y→−2.9 cm), `d7_validation_horizon_V1.png` (why 0.1 s training is blind: drift enters past 0.5 s),
  `s3_openloop_multisine_V1_standstill_Yp10.png` (drift is intrinsic to the ANN: truth clean, ANN drifts
  10–40×). Untrained control: `d6_ann_mean_force_untrained.png`. The drift checkpoint they dissect:
  `simulations/gantry_subnet/diagnostics/checkpoints/gantry_drift_last.pth` (from `make_drift_checkpoint.py`,
  Optuna Trial-3 config lr=1.49e-8/nf=1400, X+Θ+Y, `_last` weights).
- **Literature**: `docs/ml-for-control-search-sweep.md` Directions 10–12; D-110 (R5 evidence file). PDFs in
  `literature/corrupted-scheduling/`, `literature/theses-lpv-lineage/`, `literature/aerospace-qlpv/`.
- **Layer-2 concept + limits**: `docs/data-silent-regularization-concept.md` (§7 updated with the d16
  measured correction: direct frequency-selective joint-DC pin) + `-limits.md`.
- **Construction / 5 requirements**: `docs/all-five-construction-spec.md`. **Decisions**: `docs/decisions.md`
  D-104…D-110. **Run table**: `docs/gantry-augmentation-problem-log.md` §12 (70558 falsified; curriculum row).

## Next actions (in order — but the NEW supervisor feedback may reorder these)
1. **Launch the nf-curriculum** (implemented, ready) OR act on the new supervisor feedback first.
2. **Supervisor-meeting brief** (approved direction, not written): now MAINLY one decision (empirical R4);
   R5/Y-scheduling answered by d13. Quotes ready (Cox §11.3.1; Verhoek external-scheduling; Hanema motion-
   systems; Schuet parasitic-term). House style: `LPV/LFR-derivation-supervisor.tex`; one object per section.
3. **Build Layer 2** (target fully specified above) → validate per concept §9 (12 s envelope ~1, absorber
   band untouched, injected-friction discriminator — the last needs a NEW friction-injected sim dataset,
   MATLAB side, D-090 forbids adding friction to the current data).

## Standing constraints (enforced)
- Velocity/accel-domain loss = LAST RESORT (supervisor-gated). Theta-only routing = never the deliverable
  (D-103); X/Y stay in the routing; fixes act on loss/estimator, not the routing set.
- No compute-cost adjectives without a measured basis; the user runs the jobs — ask what is runnable.
- MANDATORY when reading ANY run log: read its printed Configuration/hyperparameter block FIRST and check it
  against intent (70799 was mis-analysed as lr=1e-3; the log said lr=1e-7 — deployed copies lag local edits).
- Every gantry training script MUST print per-epoch `[nf-probe] train/val nf-RMS` (D-102).
- One recommendation at a time; documents are not progress; primary-read before quoting; separate measured
  from inferred; a hypothesis is not a conclusion.

## Environment notes
- `conda run -n GraduationProject python ...`; set `PYTHONIOENCODING=utf-8` EVERY PowerShell call (state does
  not persist; missing it cp1252-crashes conda's output relay). Bash tool lacks coreutils (no head/grep/cp) —
  use PowerShell or the dedicated Read/Grep/Glob tools.
- PDF extract script + all lit text in the session scratchpad; research.tue.nl blocks curl (Cloudflare),
  pure.tue.nl / rolandtoth.eu / NTRS work.
- Chunked `fit()` reloads `_best` at chunk boundaries (epoch 0 if sim-RMS only degrades); the curriculum
  handles this by reloading `_last` between rungs.
