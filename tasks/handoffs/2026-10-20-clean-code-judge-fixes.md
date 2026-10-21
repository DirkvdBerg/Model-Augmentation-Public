# Handoff: apply the Codex judge findings to `Clean/` and the harness, then re-prove equivalence (step 4c)
**From**: session of 2026-10-20 | **Branch**: Augmentation | **Effort suggested**: xhigh (coordinated edits across
`Clean/` and the harness, then the full local harness again)

## 1. Task
Apply the findings of the Codex judge (`tasks/clean-code/judge-codex.md`) that Dirk accepted, then rerun the local
equivalence harness so the evidence describes the fixed code. In this order:
1. **Mainline oracle first:** two mainline edits were made on 2026-10-20 (D-278, section 4). The layer-3 and 3c
   goldens that cover polish-only or a spent Adam budget must be recaptured from mainline with these edits. Decide
   from the code which golden cases they touch (at least the 3c polish-only cases); recapture those, keep the rest.
2. **`Clean/` edits:** D-278 (the same two changes), and findings F1, F4, F5, F6, F7, F8 to F16 as section 4 states.
3. **Harness edits:** H1 and H2.
4. **Re-prove:** `pyflakes` and imports, then layers 1, 2, 3, 3b, 3c and the `R6 lowpass --cpu --smoke` acceptance
   run, bitwise as before. Extend layer 3c with the cases the new behaviour needs (section 10).
5. **Update `tasks/clean-code/implementation-report.md`:** a section "Judge fixes (step 4c)" with one line per finding
   (what changed, where, how it was tested), the new layer results and runtimes, and the updated server command
   list.

## 2. Out of scope
- The evaluation scripts in `Thesis-writeup/Code/Evaluation/` (D-275 amendment 4: untouched).
- Comment style (a later pass); keep the required markers and `# THEORY:` / `# HEURISTIC:` labels only.
- Thesis text, including the two phrase fixes T1 and T2 of the judge (Dirk decides those separately).
- The six D-276 polish-only reruns on the server: they run the campaign code, not mainline or `Clean/`.
- Server jobs (Dirk submits) and any git operation.

## 3. Where things stand
`Clean/` (24 files, 4,447 lines) passed every local layer bitwise before these fixes
(`tasks/clean-code/implementation-report.md`). Mainline has D-274, D-275 amendment 3 and now D-278 (compile check
only, not run). The server layers 4/4b and GPU smokes have not run; Dirk waits for this step. No run in flight.

## 4. What to change (Dirk accepted all, 2026-10-20)
**D-278 (also in `Clean/`):**
- F3: a polish-only start compares with the checkpoint's selected score `bestfit` whenever it is finite (Dirk: "I
  prefer best checkpoint still"); a U polish-only start first runs one eager validation, because the skipped
  remeasure completed lazy initialisation before the compiled probe. Mainline: `training.py:790-797`.
- F2: a resume with no epochs left skips Adam and goes to the polish (no `epochs or hp['epochs']` truthiness).
  Mainline: `training.py:898-901`.
**Robustness (`Clean/` only):**
- F1: a fresh run refuses an existing run folder before writing anything and points to `--resume`/`--polish-only`.
- F6: a rerun checks data, device and checkpoint readability and compatibility before it moves files to
  `previous/<rerun_id>/`, or restores the archive if initialisation fails. D6's one-folder and archive rules stay.
- F16: `_last.pth` (and the other dumps the periodic save replaces) is written to a temporary file and renamed into
  place only after a complete write; the save point and the explicit `--from last` rule stay.
- F5: the whole mutating polish, including FixedBatch construction and the final scoring, sits under one
  restore-and-resync boundary; an allocation failure counts as an aborted polish and keeps the pre-polish result.
- F4: carry the cumulative epoch count across chained continuations (a second continuation must not recompute its
  remaining budget from its own local counter); refuse a continuation whose source checkpoint lies inside an
  unfinished PS2 opening, with a clear message, instead of silently restarting or skipping the opening. Do not claim
  an uninterrupted-trajectory continuation where the batch stream or PS state is not restored.
**Fidelity and minimality (`Clean/` only):**
- F7: restore the `# THEORY:` label beside the kept first difference in `Clean/gantry_dynamic/data.py:189`
  (mainline `data.py:350`).
- F8 to F15: the trims as the judge lists them, each with the judge's "keep" conditions (fixed loader RNG,
  overwritten draws, column scaling and its Gram diagonal, the basis builder's `space` argument, externally used
  signatures, the S2 variance convention).
**Harness (`Thesis-writeup/Code/Verification/clean-code/`):**
- H1: `compare_l4.py` first asserts the expected set (9 runs, 3 splits, every 4b interface, nonempty eval check,
  finite positive reference scores) on both sides, and labels anything less PARTIAL, never PASS.
- H2: every comparison recomputes the hashes of the full current `Clean/` tree (all files, including the package
  `__init__.py` files and the import-origin guard) and refuses clean outputs made from a different tree.

## 5. Established and verified
- Judge verdict, findings and the per-file table: `tasks/clean-code/judge-codex.md`.
- Decisions: D-274, D-275 with amendments 1 to 4, D-278 (end of `docs/decisions.md`).
- Harness method, budgets, fingerprints and layer definitions: `tasks/clean-code/design.md` section 6 and
  `implementation-report.md`; the lstsq solver pin of deviation 7(e) stays.

## 6. Assumed but not verified
- That the trims F8 to F15 change no number: each must still pass layers 1 to 3c bitwise; a trim that does not is
  reverted and recorded, never toleranced.
- That the D-278 mainline edits behave as intended: they have only been compiled. The recaptured goldens and the new
  3c cases are their first run.
- Runtimes: the full local harness took roughly 2 to 3 h before; time the first case of each layer again.

## 7. Tried and failed
- A launch command that converted Windows paths through cygpath created a stray folder under
  `C:\Users\20203253\AppData\Local\anaconda3\Library\c\`; use literal Windows paths for `LOCALAPPDATA` and roots.
- Deleting files outside the allowed folders is refused by the permission guard; list leftovers for Dirk instead.

## 8. The open question
Nothing blocked. If a finding cannot be fixed as stated without changing a number on a tested path, keep the
current code for it, record why, and leave the decision to Dirk.

## 9. Next action
Read `judge-codex.md`, then decide which goldens D-278 touches and recapture them from mainline.

## 10. Acceptance criterion
- `pyflakes` reports no undefined name in `Clean/`; every module imports with only `Clean/` on the path.
- Layers 1, 2, 3, 3b, 3c and the CLI smoke pass bitwise, with H2's freshness check active in every comparison.
- Layer 3c gains, both codes where mainline supports it: a polish-only start on an OBC row whose decision uses the
  selected score (D-278 F3), and a resume with zero epochs left (D-278 F2). Clean only: a fresh launch into an
  existing folder is refused (F1); a rerun whose checkpoint is unreadable leaves the live folder unchanged (F6); a
  chained second continuation computes its remaining budget from the cumulative count (F4).
- `implementation-report.md` has the step-4c section and the updated server command list.

## 11. Read these first
1. `tasks/clean-code/judge-codex.md`.
2. Section 4 of this handoff.
3. `tasks/clean-code/implementation-report.md`.
4. `docs/decisions.md` D-278 (end of file).
5. `tasks/clean-code/design.md` section 6.

## 12. Do not
- Do not edit mainline (D-278 is already in it), the evaluation scripts, `design.md` or the thesis.
- Do not run more than one numerical job at a time; launch runs in the background with streamed output; no
  multi-hour single runs.
- Do not write into `Thesis-writeup/Results/`, the mainline run folders, `~/.deepSI/checkpoints` or data folders.
- Do not read `kamtin-data/Data Telica/`.

## 13. Operational
Same environment and conventions as the implementation step (`implementation-report.md`): env `GraduationProject`,
single-threaded CPU float64 harness processes, goldens under `simulations/clean-golden/`. Final message: the
step-4c results, the leftovers for Dirk, and the server command list.

## 14. Delegation
At most one general-purpose subagent, for the harness edits H1 and H2 while you edit `Clean/`. None for the
`Clean/` edits. No Workflow.
