# Keep audit: the shared criteria (step 4b, Claude and Codex use this block unchanged)
D-284 amendment 8. Both audits judge by this file only, so their rows can be merged one to one. Where it differs
from `00-overview.md` section 2, this file holds (amendment 8 overrides criterion 5(d) "harness").

## The goal, in Dirk's words
"clean minimal optimal code", "every file has a reason to be there, all functions make sense to have". Minimal
means "writing the code in a smart efficient way", not "less lines is automatically better". Optimal: "do we
approach computation in a smart way, can we combine processes, are we just not repeating steps unnecessarily".
"No pass-through layers." The yardstick per callable: would an expert who knows the thesis and PyTorch, writing it
from scratch, end up with this code? If they would write it more directly, that is a finding, even when their
version is not shorter.

## Scope and rows
- The 24 grey-box files of snapshot `final` (`simulations/clean-golden/src-snapshots/final/`, manifest
  `98d23ae2...1f99`), which equals `Clean/` at the start of this step. Line numbers refer to that snapshot.
- **The two audits are independent.** Each enumerates its own sites from `final` (its own AST pass: every file,
  function, method, property, constructor, nested function and lambda) and runs its own static tools on `final`
  (ruff `F,B,SIM,PLR,PERF,C90`, vulture, radon `cc`, a pass-through and called-once scan, its own caller search
  outside `Clean/`). Neither reads the other's audit, site list or tool output. One row per site of its own list.
  At the merge, a site that only one audit lists is itself a finding.

## Row format
`site | name | verdict | reason | evidence | compute | change | label | layer`
- **verdict**: `keep`, `remove`, `inline`, `merge into <x>`, `simplify`. A file row says whether the file should
  exist or merge into another.
- **reason** for a keep, one of:
  - (a) thesis concept: the entry from `tasks/clean-code/thesis-code-map.md`. For a pass-through or called-once
    function, (a) holds only if its body itself computes the concept; a body that forwards fails (a). Answer the
    inline test: does the call site get clearer or murkier with the body inlined?
  - (b) compile boundary: the `torch.compile` call that needs it.
  - (c) interface: name it (deepSI hook, torch `forward`, checkpoint keys, CLI, `config.json`, rerun rules of overview
    section 6.2). For Jan's code: the definition in `Model-Augmentation-Public-main/` by `file:line`.
  - (d) caller outside `Clean/` by `file:line`. The harness (`Thesis-writeup/Code/Verification/clean-code/`) is not
    a reason: it is ours and follows the code. The evaluation scripts import mainline, not `Clean/`
    (`Thesis-writeup/Code/Evaluation/eval_common.py:19-21`), so they are no reason today, and a future clean
    evaluation is no reason either (Dirk 2026-10-30); the evaluation cleanup adds back what it needs.
  A row without one of these is a finding, never a keep.
- **Uncovered lines and branches** (coverage over the 16 thesis cases): keep only as a guard with an actionable
  message for invalid input, data or solver state, or as a path a named thesis row, CLI option or rerun mode
  reaches; otherwise remove (overview criterion 4).
- **Radon grade C or worse**: keep the length only if each branch maps to a named thesis step; otherwise propose
  the split.
- **compute** (every function with a loop or a per-call computation): a hoist, combine or repeat finding (what is
  recomputed, what could be batched or merged, what result already exists), or `none`.
- **change** (every non-keep, every compute finding): the concrete new code, or a one-line sketch of it.
- **label**: `bitwise` (same operations in the same order on the same values, dtype, RNG draws; overview section
  6.1) or `rounding` (which operations change rounding, expected size). Plus `touches Jan's code` where it does,
  and the compiled effect; a change inside `hfn` or `output_only` states the `torch._dynamo.explain` op count before
  and after (measured, or "estimate").
- **layer**: the harness layer that executes the changed lines (L1-2 forward and gradient; L3 PS2, polish; L3b
  forced reject; L3c reruns; CLI smoke).

## Speed sites
Every problem 1 to 28 of `tasks/clean-code/perf-baseline.md` section 4 gets a row: improved in the round (M ID and
measured gain), a new proposal with evidence, or a measured reason why not.

## Fixed rules
Jan's classes, names and files stay 1:1 (design choice 3); his definitions are judged inside. Checkpoint keys,
`config.json`, the CLI and rerun rules stay (overview 6.2). Rejected items of D-284 amendment 2 stay rejected unless
the row gives new evidence. Only `bitwise` items are implemented in this step; `rounding` proposals are listed with
their gain for a later ruling (Dirk 2026-10-30).

## Evidence
- Your own enumeration and static tool runs on `final` (above).
- The only shared input: the raw coverage data `simulations/clean-golden/perf/audit-before/coverage/combined.json`
  (branch coverage of the 24 files over the 16 thesis cases, run on `final` by the Claude session's item 0; machine
  output, no judgment). It appears about an hour after the start; fill the coverage-dependent rows when it exists.
  Do not use `audit-before/findings.tsv` or `scoreboard.json` (they hold the Claude session's scans).
- `tasks/clean-code/optimise2-claude.md` section 3 and `optimise2-codex.md` (step-2 reasons: re-judge each, say where
  you disagree); `perf-baseline.md` section 4.

## Summary at the top of each audit
Counts per verdict; proposals ranked by gain; compute findings; disagreements with the step-2 reasons.

Row example (invented content):
`Clean/gantry_dynamic/example.py:42 | helper_x | inline | none: called once, body forwards to build_y | scoreboard
pass-through #7, only caller training.py:88 | none | inline the call at training.py:88 | bitwise, compiled none | L1-2`
