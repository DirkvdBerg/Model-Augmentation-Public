# The augmentation training issue: reading list for an INDEPENDENT diagnosis

**Written 2026-07-25.** **Purpose: hand a fresh session everything it needs to decide FOR ITSELF what
causes the failure.** This is an index and a briefing, not a conclusion. It deliberately does not tell
you what the answer is.

## 0. Your instruction

Re-assess the **whole project's evidence**, not just the most recent session's. Treat every claim in
every document below, including the ones written on 2026-07-25, as **scrutinizable**. Several
long-standing claims in this repo have already been falsified by later measurement (see §5), so
inheriting a framing is the main failure mode here.

Where a document and the raw data disagree, the raw data wins. Every unit JSON, log and checkpoint
needed to recompute the recent results is on disk (§4).

## 1. The complaint, in the user's own words

Two failures, stated by the user, and they are **not** the same thing:

1. **"the ANN not learning, and it only becomes worse than the encoder initialization"**, i.e. the best
   checkpoint is epoch 0 and validation only degrades. This is the R2 / fit failure.
2. **"the drift is still a problem when we are using the data with MSD"**, i.e. free-run position drift
   on the with-absorber data. This is the R4 failure.

Most of the 2026-07-25 measurement campaign scored (2) on data where there is nothing to learn. That was
a scoping error by the assistant, recorded in `tasks/lessons.md`
(`score-the-metric-the-user-is-actually-failing-on`). An independent reviewer should decide whether the
two failures share a cause or not.

## 2. Requirements and framing: what a solution must satisfy

| File | What it gives you |
|---|---|
| `docs/all-five-construction-spec.md` | **THE FIVE REQUIREMENTS** (R1 knowledge-free, R2 full expressivity, R3 marginal-preserving, R4 non-drifting, R5 scheduling integrity), each mapped to a mechanism and to how it would be validated. Also Route A versus Route B and why Route B is primary |
| `docs/literature-search-conclusion.md` | why no published method meets all five (D-108), with the 2025 survey that confirms the area is open |
| `docs/augmentation-literature-verdict.md` | the earlier four-requirement version of the same question |
| `docs/drift-diagnosis-status.md` §5 | the original statement of the five requirements |
| `CLAUDE.md` "Project Identity" and "Control Engineering Stance" | the thesis, the three pipelines, and the reasoning rules the project holds itself to |
| `docs/control-reasoning.md` | the control-engineering checklist (loop, coordinates, identifiability, excitation, noise, negation, well-posedness, thresholds) |
| `docs/prioirity-list-meeting-07-07.md` | the supervisor's priorities, including multiple shooting as the preferred position-based route and the velocity-loss last-resort constraint |

## 3. Problem statements and prior diagnoses, oldest to newest

Read in this order; each supersedes parts of the previous one and says so.

| File | Date | What it is |
|---|---|---|
| `docs/gantry-augmentation-problem-log.md` | running | **The single most useful file.** Every failure mode with its symptom and root cause, plus §12, the run table: every training run, its hypothesis stated BEFORE launch and its outcome after |
| `docs/drift-diagnosis-status.md` | 07-09 | the X+Theta+Y drift diagnosis: findings, hypothesis, solution candidates |
| `docs/drift-critical-analysis.md` | 07-24 | an INDEPENDENT critical analysis of the whole project's evidence, written by an earlier adversarial session. Closest precedent for what you are being asked to do |
| `docs/drift-problem-statement.md` | 07-25 | the evidence-graded statement: issues I1 to I8, what is genuinely unknown, the six hard constraints, and §7 "closed, do not re-open". Carries a STATUS banner naming the parts now superseded |
| `docs/drift-conclusions-2026-07-25.md` | 07-25 | conclusions C1 to C7 from the D1 to D6 campaign, each with its evidence grade and the artifact that holds the number |
| `docs/ann-worse-than-init-diagnosis.md` | 07-25 | the R2 failure: the measurement, the proposed mechanism, and a PRE-REGISTERED decision table (§5b) for the MSD test that was still running when this index was written |
| `docs/drift-problem-statement-post-diagnostics.md` | 07-25 | the same material framed as input for a literature session, with an explicit anti-scope |

## 4. Measurements and raw data you can recompute from

| Where | What |
|---|---|
| `scripts/gantry/drift-diagnostics/results/D1..D6*.md` | six diagnostics, each with the question, the exact command, the rig hash, numbers with units, verdict against a pre-decided table, thresholds with their data source, and what remains unresolved |
| `scripts/gantry/drift-diagnostics/results/DECISIONS.md` | every judgement call made while measuring, with the alternative that was rejected |
| `scripts/gantry/drift-diagnostics/data/` | unit JSONs, 3 ANN checkpoints at step 84, 3 Adam optimizer states, the D4 real-data payload |
| `scripts/gantry/drift-diagnostics/logs/` | raw stdout of every run |
| `scripts/gantry/drift-fix-trials/` | the FROZEN rig (`rig.py`, hash `e1b0511a4c`) and the T0/T1 campaign that preceded it, with its own `results/` and `research/` threads |
| `scripts/gantry/diagnostics-drift/` | 20 earlier drift diagnostics (d1 to d17) plus `drift_common.py`, which holds the verified 8-state truth simulator including the MSD absorber |
| `scripts/gantry/baseline-null/` | the earlier null harness and the OTHER-RIG curvature numbers that the 07-25 campaign re-measured |
| `scripts/gantry/ARTBP/`, `gantry-zero-mean/`, `orth-projection/`, `passive-augmentation/`, `datasilent-friction-sim/` | prior candidate families, each with its own README and results |

**Rig caution.** `rig.py` uses `mode='augmentation/baseline'`, the **no-MSD** records, and its hash refers
to that data. `mode='augmentation'` is the **with-MSD** data. Numbers from the two are not comparable.
The production entry point is `scripts/gantry/gantry_interconnect_dynamic.py`.

## 5. Claims that have already been falsified, and by what

Do not inherit these. They appear in documents that are otherwise still valid.

| Claim | Status |
|---|---|
| "the windowed loss cannot constrain the drift direction" | refuted: positive definite Hessian, Frye index `4e-16` (`drift-conclusions` C1) |
| "the drift direction is practically non-identifiable from this excitation" | refuted: 3030x more Y modulation does not move the parked constant (C2) |
| "the H^3.7 curvature was a synthetic-direction artefact" | refuted: `p = 3.75` along the trained direction (C3) |
| "the rank-1 pin gets dodged by the optimizer" | refuted earlier (I4); the real barrier is the cancellation in C6 |
| "Adam's step is of order lr, so it cannot be mid-relaxation" | refuted: measured step is `0.005` to `0.013 x lr` (C5). This voided a load-bearing argument in the literature sweep |
| "Bock and Weiss derive without bias correction at eps = 0" | false: bias correction is that paper's contribution; their bifurcation inequality also puts this rig on the stable side (C5, and `drift-problem-statement-post-diagnostics` §4 Q1) |
| "multiple shooting was tried and failed (Optuna 69399)" in `tasks/lessons.md` | wrong twice: lr-bug confounded, and that run was an `nf` sweep under single shooting with no continuity term (`docs/multiple-shooting-sweep-2026-07-25.md`) |
| `arXiv:2006.06650` supports "projection induces a compensating stochastic bias" (`thread-AB` A8) | refuted: the word "bias" does not occur in that paper's body |
| the over-damped-baseline argument in `drift-problem-statement` §6 constraint 4 | not supported: real-data `dF/dv` is negative on 22 of 22 X logs (C7) |

## 6. Known limits of the recent evidence

State these back to the user rather than working around them:

* D1 to D8 are on the **perfect-match null**, where the correct ANN output is exactly zero. The project's
  own I7 says that testbed cannot discriminate candidates, because its converged drift sits below the
  absorber signal.
* D8 has 2 complete seeds plus a partial third, below the project's 3-seed floor.
* D7 (the 400-step run testing whether the parked constant converges) was stopped at ~step 350 of seed 0.
  Its partial result suggests the constant does converge, which would make part of "the drift" a
  convergence artefact. **Unfinished, one seed, do not treat as settled.**
* D9 (the MSD version of the R2 test) was still running when this was written. Its first data point:
  training loss falls 2.5x over 2 epochs while validation sim-NRMS goes `16.3` to `513` to `443`.
* The literature sweeps' negative claims are graded provisional where OpenAlex rate-limited.

## 7. Suggested first questions for you to answer independently

Not a plan, just the questions the project cannot currently answer:

1. Are the R2 failure and the R4 failure one mechanism or two?
2. Is the windowed-train / free-run-validate gap sufficient on its own to explain "best checkpoint =
   epoch 0", without any optimizer story?
3. Does anything measured on the null transfer to the MSD data, and what is the evidence either way?
4. Given the five requirements in §2, which of the previously-killed candidate families were killed on
   evidence that still stands after §5?
