# Prompt for a fresh session: build a verified diagnostic overview

Copy everything below the line into a new chat.

---

## Mission

Read this repository's diagnostic record and produce **one document**,
`docs/diagnostic-overview.md`, that states (a) what problem this project is actually
facing, (b) which claims about it are **verified against artifacts** rather than asserted
in prose, and (c) **how the understanding evolved** — what was believed, what replaced it,
and why.

You are not being asked to solve the problem, propose a fix, or run training. You are being
asked to produce the document a new person would need in order to trust anything here.

## The one hard rule

**Artifacts beat documents.** This repo contains many confident, well-written claims that
later measurement falsified — including several written on 2026-07-25 and 2026-07-26. Every
document is scrutinizable, including the most recent ones. Where a document and a stored
number disagree, the number wins.

Do **not** adopt any existing framing as your starting point. Inheriting a framing is the
documented failure mode here (`docs/ISSUE-OVERVIEW-FOR-INDEPENDENT-REVIEW.md` §0). Read the
measurements first, form your own view, and only then compare it to what the documents
claim.

## Verification standard

For each claim you carry into the overview, record:

| field | meaning |
|---|---|
| the number | with units and horizon (see the horizon trap below) |
| the artifact | the `.json` / `.npz` / `.pth` / log file that holds it, by path |
| how it was produced | which script, which rig, which data |
| grade | ROBUST (3 seeds, 2 protocols) / SOLID (3 seeds, 1 protocol) / SINGLE (1 seed or 1 record) / ORACLE (uses information unavailable on real data) / VOID (the run's own control failed) / ASSERTED (prose only, no artifact found) |

**ASSERTED is a real and important category.** If a claim is repeated across several
documents but you cannot find an artifact behind it, say so. At least one campaign-level
"do not revisit" decision in this repo rests on a script with no retained output.

Where you can cheaply open a stored `.npz` / `.json` / `.pth` and confirm a quoted number,
do it, and mark that you did. Prefer verifying a few load-bearing claims properly over
listing many unverified ones.

## Where to look — enumerate, do not sample

A previous sweep failed by not enumerating prior work. Read at minimum:

**Primary record**
- `docs/gantry-augmentation-problem-log.md` — especially §12, the run table. Every run with
  its pre-declared hypothesis and its outcome. This is the single most useful file.
- `docs/results-log-2026-07-26.md` — measurements only, deliberately free of interpretation.
- `docs/decisions.md` — design decisions, newest first.

**Framings, oldest to newest (each supersedes parts of the previous, and says so)**
- `docs/drift-problem-statement.md`
- `docs/drift-critical-analysis.md` — an earlier independent adversarial review; the closest
  precedent for your task
- `docs/drift-conclusions-2026-07-25.md` — conclusions C1-C7 with evidence grades
- `docs/ann-worse-than-init-diagnosis.md`
- `docs/ISSUE-OVERVIEW-FOR-INDEPENDENT-REVIEW.md`
- `docs/narrowband-objective-problem-2026-07-26.md` — note its §5 was superseded the same day
- `docs/flat-direction-problem-2026-07-26.md` — the newest framing; treat as scrutinizable
- `docs/ms3-decision-table.md`

**Literature sweeps (none superseded)**
- `docs/narrowband-literature-sweep-2026-07-26.md`
- `docs/multiple-shooting-sweep-2026-07-25.md`
- `docs/drift-literature-sweep-2026-07-25.md`
- `docs/rollout-stability-literature.md`

**Requirements and reasoning rules**
- `docs/all-five-construction-spec.md` — the five requirements R1-R5
- `docs/control-reasoning.md`, `CLAUDE.md`
- `tasks/lessons.md` — the project's own ruleset, including its record of past errors
- `tasks/session-handoff-2026-07-25.md`

**Script folders — check every one for a README, `results/`, `PROGRESS.md` and stored units**
`scripts/gantry/`: `pysynth-data/`, `drift-diagnostics/`, `drift-fix-trials/`,
`diagnostics-drift/`, `baseline-null/`, `datasilent-friction-sim/`, `gantry-zero-mean/`,
`orth-projection/`, `passive-augmentation/`, `ARTBP/`, `drift-demo/`, `drift-visual/`,
`encoder-augmentation/`, `encoder-baseline/`, `encoder_initialisation/`,
`augmentation-error/`, `real-data-verification/`, `parameter-diagnostics/`, `verification/`

**Stored model state**
`simulations/gantry_subnet/augmentation_linear_map/71167/` and
`simulations/gantry_subnet/diagnostics/checkpoints/`

Use parallel Explore subagents for the folder sweep; do the load-bearing reads yourself.

## Capture the progression — this is half the deliverable

The overview must show how the diagnosis moved, not just where it landed. For each major
claim, trace its lifecycle:

- when it was first asserted, and on what evidence
- what challenged it
- what it was replaced by, and whether the replacement is better evidenced or merely newer
- whether the original is still cited anywhere as though it stood

Include a **falsified-claims table**: claims this project believed and then disproved, with
what disproved them. Several exist. At least one propagated into a decision before being
caught, and at least one was still being cited as settled a day after being refuted.

Also record **void runs** — attempted measurements whose own controls failed. Knowing what
was tried and did not work is what stops it being re-run. Several are marked in the run
table; do not silently drop them, and do not quote their numbers either.

## Known traps — you will hit these

- **Always state the horizon with any error number.** The same ANN-off model measures
  `7.86e-05` at 2 s and `1.66e-04` at 12 s. Numbers quoted without a horizon are unusable,
  and at least one wrong conclusion in this repo came from comparing across horizons.
- **`gantry_ckpt_*.pt` is the BEST checkpoint.** Since the failure is that best = epoch 0,
  that file IS the initialisation, not a trained model. Use `*_last.pth` for a trained one.
- **`.pth` files pickle `gantry_dynamic` as a top-level module** — put `scripts/gantry` on
  `sys.path` before `torch.load`. They carry their own `norm`; take weights and `norm`
  together, never mixed with locally computed constants.
- **They also carry full per-epoch histories** (`Loss_val`, `Loss_train`, `Loss_val_nf`,
  `Loss_train_nf`). Several questions can be answered by reading these rather than running
  anything.
- `RunConfig` defaults `up_sample = 2`; the entry file and every checkpoint use `1`.
- Trimming the training file list changes the normalisation, which changes the encoder built
  from it, which changes every downstream number.
- Filtering non-finite values out of a metric series makes divergence look like a flat pass.

## Output

Write `docs/diagnostic-overview.md` with:

1. **The problem**, in one paragraph, in your own words, from the measurements.
2. **The verified core** — the claims that survive your verification standard, each with its
   number, artifact, and grade. Order by how load-bearing they are.
3. **The progression** — how the understanding evolved, with dates and what forced each
   change.
4. **Falsified claims**, with what refuted them and whether they are still cited anywhere.
5. **Void runs**, with why each is unusable.
6. **What is genuinely open**, separated from what is merely unfinished.
7. **What is ruled out**, with the evidence and its grade — so nothing here is re-proposed.
8. **Your own assessment of the evidence base**: how much of it is single-seed, how much is
   oracle-dependent, and which load-bearing claims you could not verify.

Be critical. If the newest framing is weaker than it presents itself, say so. If a claim is
repeated everywhere but rests on one unreplicated run, say that. The value of this document
is that it can be trusted, which means it must be willing to say where the record is thin.

Do not run training. Read-only work plus cheap artifact inspection only.
