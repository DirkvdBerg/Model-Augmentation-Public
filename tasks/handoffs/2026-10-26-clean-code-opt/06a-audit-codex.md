# Prompt for Codex: independent keep audit (part A) and diff review (part B), step 4b
**From**: Claude session of 2026-10-30 (D-284 amendment 8), revised after two reviews, for Dirk to give to Codex |
**Branch**: Augmentation | **Effort**: high

## Part A: independent keep audit (starts together with the Claude step-4b session)
Read `tasks/handoffs/2026-10-26-clean-code-opt/00-overview.md` sections 4 and 6 (fixed rules), then
`tasks/handoffs/2026-10-26-clean-code-opt/06c-audit-criteria.md`: it is the complete yardstick (scope, row format,
keep reasons, evidence, speed sites), the same one Claude uses. Your audit is independent of Claude's: enumerate the
sites of `simulations/clean-golden/src-snapshots/final/Clean/` yourself and run your own static tools on it (the
criteria file lists them; read-only tool runs on that snapshot are allowed). Do not read anything the Claude
session writes (`tasks/clean-code/keep-audit-claude.md`, `simulations/clean-golden/perf/audit-before/` except
`coverage/combined.json`, which is raw coverage data and appears about an hour after the start).

Write `tasks/clean-code/keep-audit-codex.md`: your site list and one row per site and per speed site, in the row
format of `06c-audit-criteria.md`, with the summary at the top. Judge as an expert who knows the thesis and PyTorch
and would write this code from scratch.

Done when every site of your list and every speed site has a row. Report every finding, small ones too. Final message: the path,
counts per verdict, number of proposals, number of compute findings, number of disagreements with the step-2
reasons.

## Part B: diff review (Dirk gives this after Claude has implemented the ruled items)
Read `tasks/clean-code/keep-audit-merged.md` and D-284 amendment 9 in `docs/decisions.md`. Compare:
- `Clean/` (grey-box files) with `simulations/clean-golden/src-snapshots/final/Clean/` (the tree before this step;
  check first that `simulations/clean-golden/src-snapshots/final/snapshot.json` still names manifest
  `98d23ae2...1f99`, otherwise compare with `final-step4/`);
- `Thesis-writeup/Code/Verification/clean-code/` and `Thesis-writeup/Code/Evaluation/` with
  `simulations/clean-golden/src-snapshots/audit-harness-before/`.
In `Clean/gantry_dynamic/data.py`, the MM16 line (D-284 amendment 6) must be in its final form, as in the snapshot.
For every accepted `K` item: implemented as ruled and nothing more; the label holds (bitwise: same operations in the
same order on the same values, dtype, RNG draws; overview 6.1); interfaces of overview 6.2 intact; Jan's changed
blocks carry `# CHANGED:`; changed functions keep comments at the D-284 amendment 4 and 7 standard. For every harness
edit: only clean-side paths changed; each key added to `DROPPED_IN_CLEAN` names its `K` ID; `equation_tests.py` still
has 72 tests or each removed one names its `K` ID. List every change without a `K` ID.
Write `tasks/clean-code/keep-audit-diff-review-codex.md`: one finding per line, `file:line`, the `K` ID, the fix.
Final message: the path and the finding count.

## Rules for both parts
Read-only on everything except your output file (static tools write nothing outside a temporary folder). No
training or harness runs. No git write operations. Never read
`kamtin-data/Data Telica/` or `kamtin-data/Telica.mat`. Never compile the thesis. No em dashes or double hyphens as
dashes.
