# Session prompt — NEW SESSION #1: critically analyze ALL our results, then do the extended online search

Paste this into a fresh session. TWO phases: (1) independently and CRITICALLY analyze the ENTIRE project's
evidence base, then (2) do a DETAILED extended online literature search. You are NOT testing or coding fixes —
a later session does that.

## Phase 1 — CRITICALLY ANALYZE THE WHOLE PROJECT (this is the point of this session)
Analyze the results WE HAVE — the entire project history, across ALL the gantry subfolders and docs and prior
sessions — **independently and critically**. Do NOT center any single session. In particular, `docs/drift-
problem-research-brief.md` and the `datasilent-friction-sim/` step1/step2/step4 work are just ONE recent chat's
output and may be wrong or over-claimed — treat them (and every other session's claims) as things to SCRUTINIZE,
not as the frame. Build your OWN assessment of what the problem actually is, what has been genuinely established
vs asserted/over-claimed, and where the evidence conflicts.

Read broadly and critically (not exhaustive — follow the evidence):
- `scripts/gantry/diagnostics-drift/` (d1–d17, dA–dC — the causal-chain claims; esp. d6, d8, d9, d14, d16, dB).
- `scripts/gantry/baseline-null/` (lr_sweep, curvature_sensitivity, gain_vs_dc, diagnostics-literature.md).
- `scripts/gantry/gantry-zero-mean/` (README + RESULTS-2026-07-17-dc-drift-diagnosis.md; V1f, V3, V4).
- `scripts/gantry/ARTBP/` (Problem-1 DC result; test_self_scheduling / test_efolding for Problem 2).
- `scripts/gantry/orth-projection/`, `scripts/gantry/passive-augmentation/`, `scripts/gantry/drift-visual/`,
  `scripts/gantry/drift-demo/`, `scripts/gantry/encoder_initialisation/`, `scripts/gantry/datasilent-friction-sim/`.
- Docs: `docs/gantry-augmentation-problem-log.md` §12 (the full run table), `docs/drift-diagnosis-status.md`,
  `docs/data-silent-regularization-concept.md`, `docs/decisions.md`, `tasks/handoff.md`.
- `docs/drift-problem-research-brief.md` is a STARTING POINTER only — verify or refute it against the primary
  evidence; correct it (and note what you changed) if it is wrong.

Deliver a written critical assessment: (a) what the problem IS, defended from the primary evidence; (b) what is
solidly established vs over-claimed vs contradictory across sessions; (c) every fix TRIED and the real verdict;
(d) the sharpest open question; (e) the hard constraints (preserve pole-1 |λ|=1; full expressivity; must not
forbid a DC-carrying friction; training/estimator-side). Save it to `docs/drift-critical-analysis.md`.

## Phase 2 — Extended online search (only after Phase 1)
Run the deep-research harness on the §4 questions of the brief (refine them with anything Phase 1 taught you).
Return a CITED, adversarially-verified, synthesized report that names concrete method candidates and marks which
respect ALL the constraints, with trade-offs. Cover: Adam-vs-SGD implicit bias on flat/near-unit-root directions;
separating a loss-informed near-DC bias from a DC-carrying friction (info-cutoff is refuted); training pole-1
integrator sim models without drift; net-impulse/DC identifiability of near-integrator systems; pole-preserving
grey-box ML (port-Hamiltonian R=0, do-no-harm/W-PGNN, Gyorok orthogonal projection, Negative-Imaginary,
cyclo-dissipativity); optimizer-side subspace fixes; the encoder-init root-cause angle; bounded-integral/Tustin
factoring and whether it can still admit a friction impulse.

**SAVE the report to `docs/drift-research-report.md`** (durable artifact for the next session).

## Output of this session
- A verified/corrected understanding (brief updated if needed).
- `docs/drift-research-report.md` — the cited research report.
Hand these to NEW SESSION #2, which will build the iterative-learning testing harness (score candidate fixes vs
R1–R5, reschedule the plan from outcomes, run overnight).
