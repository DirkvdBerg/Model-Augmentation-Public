# Step 4b: keep audit, ruling, implementation and the proof (Claude)
**From**: session of 2026-10-30 (D-284 amendment 8), revised after two reviews | **Branch**: Augmentation |
**Effort suggested**: xhigh | Read `00-overview.md` first, then `06c-audit-criteria.md` (the criteria; it holds
over the overview where they differ).

## 1. Task
Make the 24 grey-box files minimal and optimal by `06c-audit-criteria.md`, with a verdict for every site, then
prove the output once. Codex audits the same code independently and at the same time (`06a-audit-codex.md` part A),
with its own site list and its own static tools; the only shared input is the raw coverage data of item 0. In this
order:
0. **Coverage and baselines** (start of the step; Codex starts at the same time):
   - `Thesis-writeup/Code/Verification/clean-code/scoreboard.py` (edit authorised here): restrict `py_files()` to the
     24 grey-box files (the `final` manifest) and drop the harness folder from `OUTSIDE`. Run `static`, `coverage`
     (16 cases, about 64 min, one detached job, log path to Dirk) and `report` with `--label audit-before`. The `hfn`
     op counts: take the step-3 values (implementation report, "Scoreboard") unless a row changes `hfn`.
   - Copy `Thesis-writeup/Code/Verification/clean-code/` and `Thesis-writeup/Code/Evaluation/` (`.py`, `.sh`) to
     `simulations/clean-golden/src-snapshots/audit-harness-before/`, the baseline of Codex's diff review.
1. **Your audit** `tasks/clean-code/keep-audit-claude.md`: your own site enumeration of `final` (as
   `06c-audit-criteria.md` says), one row per site and per speed site, in its row format, summary at the top. Your
   scoreboard output is your own evidence. Do not read anything Codex writes before yours is written.
2. **Merge.** When Dirk says Codex is done, write `tasks/clean-code/keep-audit-merged.md`: one row per site of the
   union of both site lists with both verdicts, IDs `K<n>`; a site only one audit lists is a finding, decided from
   the code. Where both audits say keep, the keep stands. Where both propose the same change, take the
   more concrete text. Where they disagree, list the item for Dirk with both arguments in one line each; never
   settle a disagreement yourself.
3. **Ruling.** One short reply to Dirk: totals per verdict, the agreed non-keep items by gain, the disagreements,
   the rounding proposals (listed only, with gain and expected deviation), items touching Jan's code. Dirk rules on
   non-keep items and disagreements; record D-284 amendment 9 (accepted, rejected with his reason in his words,
   deferred). Record only what Dirk said.
4. **Implement** the accepted bitwise items, one change group at a time.
   - **Gate per group**: the harness layer that executes the changed lines (the row's `layer`): L1-2
     `bash run_l12_all.sh clean 221-lowpass 227-lowpass` then `python -u compare.py l12 221-lowpass 227-lowpass`; L3,
     L3b, L3c with `run_l3.py`, `run_l3bc.py` and their comparators on the cases of the implementation report
     ("Gates"). Site timing only for a change meant to change runtime.
   - **MM16**: `final` carries the ruled rounding fix MM16 (D-284 amendment 6), so every gate and the suite run with
     that line reverted (the `sed` of "Server commands (D-284 step 5)" step 2 in the implementation report; reverted
     `data.py` hash `360155fc...a6bb` before any edit of `data.py`), and the line is restored after each run.
   - **Harness edits**, where a verdict needs one: only clean-side code paths; never run `--side mainline` (it writes
     the goldens); every key added to `DROPPED_IN_CLEAN` in `compare.py` names its `K` ID, and per case the compared
     count in `compare-l12.json` equals the step-4 count minus those keys; the patch targets in `capture_l12.py`
     (`least_squares_start`) and `perf_diag.py` (`closed_loop_rollout`) and the names it reads (`x_phys_next_rec`,
     `B.tol`, `ref.n`) move with the code; `equation_tests.py` keeps its 72 tests, retargeted to the host function, or
     each removed test names its `K` ID. List every harness edit in the report.
   - Markers (overview 6.3) on every changed block of Jan's code; comments of changed functions at the D-284
     amendment 4 and 7 standard; `thesis-code-map.md` rows that move are updated. If a file is removed or merged,
     update `check_imports.py` `MODULES` and every "24 files" statement in the handoff files and the report.
5. Dirk gives `06a-audit-codex.md` part B (diff review) to Codex and pastes the result back; apply what you agree
   with, answer the rest in one line each in the merged audit.
6. **Prove, once.**
   - Scoreboard after: `scoreboard.py static`, `coverage`, `report` with `--label audit-after`.
   - **Goal check**: every remaining pass-through, called-once function, vulture hit, uncovered line or branch and
     radon grade C or worse in `audit-after/findings.tsv` maps to a `K` row with a keep reason of
     `06c-audit-criteria.md`. Write the mapping into the merged audit.
   - `equation_tests.py`, `ruff --select F`, `check_imports.py`.
   - The full suite once with `run_clean_suite.sh` (archive the step-4 `_clean` outputs to `_clean-step4` first),
     MM16 reverted, detached, log path to Dirk; then restore MM16.
   - Retake the `final` snapshot (there is no `snapshot.py`): copy the grey-box `.py` files to
     `simulations/clean-golden/src-snapshots/final/Clean/`, `MANIFEST.sha256` = sorted `sha256sum *` lines with paths
     relative to `Clean/`, `snapshot.json` with label, file count and the manifest's SHA256; copy
     `Clean/model_augmentation/LICENSE` beside it, not hashed. Archive the old one as `final-step4`.
   - Update the new hashes wherever they are cited: `timing_driver.py` (expected manifest), "Server commands (D-284
     step 5)" in the implementation report (manifest, MM16 line number, reverted `data.py` hash, the arm-A and local
     test lines), and the step-5 row in `docs/gantry-augmentation-problem-log.md` section 12.
   - Add section "Keep audit (D-284)" to the implementation report: before and after scoreboard, accepted items,
     harness edits, suite result.

## 2. Out of scope
The server run (step 5, after this step). Rounding items (listed, not implemented). Comment rewording beyond
changed functions. Results changing beyond rounding. Renaming, merging or moving Jan's classes, functions or files.
`Clean/blackbox/`, `Clean/data_generation/`, `Thesis-writeup/Code/Evaluation/` and `Figures/` (other sessions, todo
item 8).

## 3. Where things stand
`Clean/` equals snapshot `final` (manifest `98d23ae2...1f99`), 24 grey-box files, MM16 at
`Clean/gantry_dynamic/data.py:218`. Step-4 suite: bitwise with MM16 reverted (implementation report, "Comments and
map (D-284)"). The step-5 server plan is prepared, not submitted, and waits for this step.

## 4. Established and verified
- Step-3 after-scoreboard: 22 pass-throughs, 70 called-once, 15 vulture hits (grey box), branch coverage 61.7 % over
  16 cases, `lbfgs_polish` E 31, `train` D 21, 11 grade C (implementation report, "Scoreboard").
- Step-2 reasons: `optimise2-claude.md` section 3 has 35 `keep (d)` rows and 9 bare `keep` rows in 3.2 (plus 6 in
  3.1); many (d) name the harness or the evaluation, which no longer count.
- The evaluation scripts import mainline (`Thesis-writeup/Code/Evaluation/eval_common.py:19-21`); the outside caller
  of `Clean/` names is the harness (`l4_loader.py:171-235`).
- Speed: validation 166 to 47 s on 221 (M1); layer-3 runtime on 221 548 to 413 s with M19; the training update is
  unchanged within noise.

## 5. Assumed but not verified
- That the uncovered branches are guards or reachable paths: step 2 grouped them; item 1 gives the per-branch
  verdict.

## 6. Tried and failed
- M5 (rounding): the L-BFGS polish amplifies last-bit changes on 221 to 4.7e-3 in the final fit; reverted.
- M8 'closed' as default: outside tolerance through the encoder start (condition 1.4e16); kept as a toggle.
- Running gates on the tree with MM16 in place: every bitwise comparison fails (step 4 needed suite A).

## 7. Achieved
Steps 1 to 4 of the round (overview section 3); `thesis-code-map.md` merged; comments at amendments 4 and 7.

## 8. The open question
Nothing blocked.

## 9. Next action
Item 0: edit `scoreboard.py` to the 24 files without the harness as outside caller, launch the coverage job, copy
the harness baseline, and start your own enumeration while it runs.

## 10. Acceptance criterion
Every site of the union of both site lists and every speed site has a row in the merged audit;
amendment 9 records each non-keep `K` ID and each disagreement; the goal check of item 6 maps every remaining
scoreboard entry to a keep reason of `06c-audit-criteria.md` (an unmapped entry fails the step); per-group gates and
the full suite PASS bitwise with MM16 reverted; the new `final` snapshot and all cited hashes agree.

## 11. Read these first
1. `06c-audit-criteria.md`.
2. `00-overview.md` sections 4 and 6; `docs/decisions.md` D-284 amendments 2, 6 and 8.
3. `tasks/clean-code/optimise2-claude.md` section 3 and `optimise2-merged.md`.
4. `tasks/clean-code/perf-baseline.md` section 4.
5. Implementation report sections "Gates" and "Server commands (D-284 step 5)".

## 12. Do not
Submit server jobs. Run `--side mainline` of any harness script. Edit the thesis, mainline,
`Model-Augmentation-Public-main/` or the mainline goldens. Record a ruling Dirk did not give.

## 13. Operational
One numerical job at a time; coverage and the full suite as detached jobs with a log. Final message: the merged
audit path, counts per verdict, accepted items implemented, before and after scoreboard, goal-check result, suite
result, new `final` manifest.

## 14. Delegation
At most one Explore subagent for caller searches outside `Clean/`; none otherwise.
