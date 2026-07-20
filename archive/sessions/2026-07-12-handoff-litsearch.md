# Session Handoff — find literature to satisfy the FIVE requirements

**Written**: 2026-07-11. _Prior handoff archived to `archive/sessions/2026-07-11-handoff-passivity-implement.md`._

## Mission for the new session
Help the user **find relevant papers (ML-for-control / LPV / system-ID)** toward an augmentation method that
satisfies **all five requirements** below. The prior session concluded no single published method meets all
five (survey-confirmed); the goal now is to keep searching intelligently AND/OR assemble a construction. The
user explicitly wants a session that is (a) genuinely capable on the requirements and (b) good at finding
papers. Be rigorous, quote from PRIMARY PDFs, and DO NOT over-claim.

## The FIVE requirements (the target; canonical in `docs/drift-diagnosis-status.md` §5)
1. **Knowledge-free** — guarantee holds without knowing the true residual (real system unknown).
2. **Friction-permitting / full expressivity** — represents ANY dissipative state-dependent residual
   (Coulomb/cogging); NO restriction on the learnable dynamics class.
3. **Marginal-preserving** — keeps the zero-stiffness X/Y free-integrator pole at the origin; must NOT damp.
4. **Non-drifting** — bounded free-run position on X/Y (OPEN-LOOP metric; closed-loop HIDES a bad model).
5. **Scheduling-integrity (R5)** — Y is SIMULTANEOUSLY a K=0 free-integrator (drifts) AND the LPV
   self-scheduling variable (`M(Y)` reads the drifting `x[2]`, CONFIRMED code-read: `model.py` Y_op=None,
   `blocks.py:659`). Y-drift DETUNES `M(Y)` (feedback X lacks). Method must not corrupt Y-scheduling nor
   damp the Y pole. **Y is the HARDEST axis.**

## READ FIRST (this session's output — do NOT re-derive)
- **`docs/drift-diagnosis-status.md`** — master doc; §0 is a full INDEX of every companion doc. §5 = the five
  requirements; §5m = the identifiability reframe; the STANDING CONSTRAINTS at top (velocity-domain = LAST
  RESORT; closed-loop HIDES).
- **`docs/literature-search-conclusion.md`** — the search verdict (D-108): no published method meets all 5;
  gap CONFIRMED by a 2025 survey (Sivaranjani et al. arXiv:2512.06315, "remains an open challenge").
- **`docs/ml-for-control-search-sweep.md`** — the 9-direction search, ~25 papers PRIMARY-READ with quotes.
  Directions: 1 rollout-stability, 2 hybrid-identifiability, 3 symmetry(closed), 4 bias/IV, 5 LPV+ML,
  6 drift-diagnosis, 7 corrupted-scheduling(R5), 8 survey, 9 LPV cost function.
- **`docs/all-five-construction-spec.md`** — the buildable assembly (Route B empirical-R4 / Route A
  structural-R4), requirement→mechanism→validation.
- **`docs/dissipativity-limits.md`**, **`docs/rollout-stability-literature.md`**,
  **`docs/data-silent-regularization-concept.md`** (+ `-limits.md`), **`docs/augmentation-literature-verdict.md`**
  (the requirement TABLE), **`docs/decisions.md`** D-104…D-108.
- `CLAUDE.md`, `tasks/lessons.md` — active constraints. **Read the lessons; this session violated several.**

## STATE OF PLAY (honest)
- **Literature: exhaustively searched, gap confirmed by a 2025 survey.** ~25 primary-reads. Still-open
  TARGETED reads worth doing: the PAYWALLED corrupted-scheduling flagship (Piga/Cox/Toth Automatica 2015,
  for R5); Verhoek LPV-SUBNET consistency (arXiv:2204.04060, already on disk in scratchpad); the UNVERIFIED
  npj-2024 HNODE decorrelation quote (do NOT cite until verified).
- **The impossibility:** a for-all-weights STRUCTURAL R4 guarantee is incompatible with full R2 expressivity.
  So R4 on the unknown-system deliverable is EMPIRICAL, not structural. All 5 are still jointly achievable
  (see construction spec); "hitting all 5" is DEMONSTRATED on the injected-friction sim, not proven on paper.
- **Empirical (this session):** the lr bug (D-101) that invalidated Optuna 69399 IS fixed and live
  (`model.py:179`). A clean de-confound run (lr=1e-7, nf=400, full X+Θ+Y routing) showed: **at the correct lr
  the model learns IN-WINDOW (nf-RMS ↓: train 4.3e-5→4.2e-5, val 3.2e-5→3.0e-5) but STILL DRIFTS
  (val sim-RMS 8e-5 → 1.1e-3 and rising).** So the drift is REAL and lr-independent — NOT an lr-overshoot
  artifact. This is the key new empirical fact.
- **Long-nf BPTT is INFEASIBLE:** nf=4000 (1 s) = 566 MB, would not run; nf=2000 (0.5 s) is also heavy (the
  user was clear these are NOT cheap). So "unroll past the 0.5 s drift onset via brute-force nf" is off the
  table on the current hardware. Any horizon-conditioning must avoid long BPTT (truncated/pushforward unroll,
  or a per-step state-consistency term) — but DO NOT label any of these "cheap"; ask the user what is runnable.

## HOW TO SEARCH (what worked; the user values this)
- **Translate the problem into each community's native vocabulary, don't rephrase "drift":** exposure bias /
  rollout stability (ML-for-dynamics), zero-stability (numerics), simulation-error consistency / corrupted-
  scheduling / EIV-LPV (control-ID), identifiability / null-space regularization (statistics).
- **PRIMARY-READ, quote from the PDF text layer.** Download with `curl`, extract with PyMuPDF
  (`conda run -n GraduationProject python /tmp/extract_pdf.py <pdf> <out>` — a script exists in scratchpad).
  NEVER quote from a WebSearch snippet or a paywalled page. Mark search-level vs primary-read.
- Respect saturation: the dissipativity/passivity/NI family and the survey are DONE — do not re-search them.
  Highest-value untried threads: (a) the corrupted-scheduling EIV-LPV flagship + follow-ups (R5, TU/e group);
  (b) truncated/pushforward unrolled training for CORRECTION setups (List-Thuerey 2402.12971 primary-read);
  (c) velocity-form LPV embedding — BUT that is velocity-domain-adjacent = supervisor LAST-RESORT-gated.

## PROCESS — mistakes THIS session made; do NOT repeat (see tasks/lessons.md)
- **Do NOT assert compute cost** ("cheap/fast/affordable/~10x"). Increasing nf is expensive at every scale;
  the user has the hardware context — ASK what is runnable, or cite a measured sec/batch. (Newest lesson.)
- **Use the script the user NAMES** (they said "the optuna script" — `gantry_optuna.py` — 3 times; it is the
  fast one: cropped 8000-sample val + pruning). Do not push a "more correct" sibling.
- **Do NOT menu-dump.** Give ONE concrete recommendation, not a list of options, when the user wants to act.
- **Do NOT over-claim results.** Separate measured from inferred; a hypothesis is not a conclusion.

## The one OPEN DESIGN DECISION (supervisor call, blocks R5/Layer 3)
**Y-scheduling: exogenous/measured Y vs self-scheduled (`M(Y=x[2])`).** Self-scheduling makes Y the hard axis
(drift detunes M). Exogenous/measured Y (natural on real data; Verhoek's own `φ(x,u,y)`) dissolves R5. This
is a legitimate standard LPV formulation either way — surface it to the supervisor.

## ENVIRONMENT
- Env python: `conda run -n GraduationProject python ...` (works this session). Set `PYTHONIOENCODING=utf-8`.
- Under SLURM, output BUFFERS (esp. via `conda run`): use `python -u` + `PYTHONUNBUFFERED=1`, avoid bare
  `conda run` or use `--live-stream`. The de-confound ran via `gantry_optuna.py` (N_TRIALS=1, fixed lr/nf).
- Every gantry training/search script MUST print `[nf-probe] train nf-RMS / val nf-RMS` per epoch AND report
  full free-run sim-RMS (the deliverable metric).

## Out of scope for now
Noise/SNR; the formal theory proofs (parallel); do NOT add Coulomb to the current (frictionless) sim.
