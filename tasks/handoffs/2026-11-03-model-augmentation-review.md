# Handoff: Dirk reviews `Clean/model_augmentation/`, each file against Jan's original
**From**: walkthrough session of 2026-11-03 (`training.py`) | **Branch**: Augmentation | **Effort suggested**: high
(function-level comparison of six files against Jan's code, plus four files of our own)

## 1. Task
`Clean/` is production code for ASMPT. After `Clean/gantry_dynamic/evaluation.py` (own handoff
`tasks/handoffs/2026-11-03-code-walkthrough-7.md`), Dirk reads `Clean/model_augmentation/` in this order:
`fit_systems/blocks.py`, `utils/torch_nets.py`, `fit_systems/pre_encoder.py`, `fit_systems/interconnect.py`,
`fit_systems/closed_loop.py`, `fit_systems/obc.py`, `fit_systems/ps2_opening.py`, `fit_systems/lbfgs_polish.py`,
`utils/utils.py`, `utils/deepSI_corrections.py`. For each file, prepare before he asks: the deviation list against
Jan's code (section 13) and the line test (section 12), then give him one complete list in the format of section 12.
Apply only after "apply" or "yes", exactly what he approved.

## 2. Out of scope
- Finished files, do not modify: `Clean/gantry_interconnect_dynamic.py`, `Clean/gantry_dynamic/*` (`evaluation.py`
  once its own walkthrough ends), `Clean/model_augmentation/systems/gantry_ss.py` (ours, done 2026-11-02).
- `Clean/blackbox/`: next, after this handoff.
- Harness folder `Thesis-writeup/Code/Verification/clean-code/`: never edit (its session `harness-f64` is gone);
  report harness consequences to Dirk in one line.
- The `__init__.py` files and `LICENSE`: review only with the file they belong to if a change touches them. Jan's
  `white_box_models.py`, `systems/cascaded_tanks.py`, `systems/mass_spring_damper.py` were removed from `Clean/`
  entirely (thesis path only); confirm once that nothing in `Clean/` imports them, then leave them.

## 3. Where things stand
Last commit `a82e573`; the tree is dirty (`Clean/gantry_dynamic/training.py` and many files outside `Clean/`; Dirk
commits). Nothing running.

## 4. Established and verified
- Jan's code: `Model-Augmentation-Public-main/model_augmentation/`, the last public version of Jan's
  Model-Augmentation (upstream `8200923`, git resolves it to `820092373c7f...`). Compare against this folder only,
  never against the repo-root `model_augmentation/`, which is our mainline and already changes 13 of his 136
  functions. Jan counterparts exist for `blocks.py`, `interconnect.py`, `pre_encoder.py`, `torch_nets.py`,
  `utils.py`, `deepSI_corrections.py`. Ours (`__project_origin__ = "added"`): `closed_loop.py`, `obc.py`,
  `ps2_opening.py`, `lbfgs_polish.py` (plus `gantry_ss.py` and the package `__init__.py`).
- Line diff Jan -> Clean (`git diff --no-index --ignore-cr-at-eol --numstat`, lines added/removed): blocks +574/-586,
  interconnect +474/-505, pre_encoder +72/-431, torch_nets +73/-183, utils +29/-146, deepSI_corrections +51/-87.
  Comments were rewritten and unused functions removed, so a line diff is no deviation list.
- `@added` decorators (`^\s*@added`): blocks 3, interconnect 10, torch_nets 2, pre_encoder 1, utils 0,
  deepSI_corrections 0. `# CHANGED:` grep hits include the module header lines.
- Renames of Jan names: D-282 (`docs/decisions.md:8517`, list at about :8524-8533), e.g. `fixed_System_data` ->
  `WindowedSystemData`, `fixed_System_data_norm` -> `SystemDataNormalizer`, `mlp_stack` -> `build_mlp`,
  `Dtype_DataLoader` -> `DtypeDataLoader`, `_replace` -> `_replace_with_retry`.
- Dirk's rulings on checks and guards: D-281 O1 to O6 (`docs/decisions.md:8493-8502`): keep the `optimizer` property
  (O1), the `least_squares_start` checks (O2), `_json_safe` (O3), the PS `.grad` save and restore (O4), the
  KeyboardInterrupt handler (O5), `fixed_System_data` as it is (O6).
- Labels: D-293 point 3 (`docs/decisions.md:9152`): "no references or labels in `Clean/`"; a reason is one plain
  cause-and-effect clause (amendment, :9157). In `Clean/model_augmentation/` there are 10 `# HEURISTIC:` and 8
  `# THEORY:` labels: blocks 433; interconnect 350, 589, 657; lbfgs_polish 18, 263; obc 35, 119; ps2_opening 60, 154;
  pre_encoder 56, 64, 65; closed_loop 41, 62, 85, 116. D-293 is later than the memory note that allowed labels "on
  values that change results" (2026-11-01) and than the CLAUDE.md labelling rule, which governs research code.
- Every reason clause needs a fact-check row (location, clause, source file:line, quote) in the second table of
  `tasks/clean-code/walkthrough-flags.md` (D-293 point 4).
- Comment-only edits are proved with `clean-up-verification/comment_ast_check.py` (code unchanged apart from text).

## 5. Assumed but not verified
- That every real change to Jan's code carries its marker: settled by the comparison of section 13.
- That `Clean/` still reproduces the thesis runs after the 2026-11-02/03 changes: only the server harness settles it.

## 6. Tried and failed
- A line diff against Jan as the deviation list -> the whole file differs (counts in section 4) -> rewritten comments
  and removed functions dominate -> compare per function with comments stripped (section 13).
- Calling a file clear without testing every line -> Dirk found problems in single functions of `training.py` right
  after I said it was fine -> I had checked only my own recent edits -> test every line (section 12).
- Rewording a rejected comment two or three times -> each rejected -> the fix was a code change (e.g.
  `hist_arrays` as one loop, `_adam_phase` reduced to Adam only) -> after one rejection, look for a code change.
- Starting reads and lint runs when Dirk asked for a handoff -> he stopped them ("what did i say") -> do only what he
  asked.
- A comment claimed the PS phase is 'S1' or 'S2' -> the code stores only 'S1' or 'S3' (`ps2_opening.py:85`, `:129`,
  `:135`) -> I wrote it without checking -> verify every claim in the code first.

## 7. Achieved
`gantry_dynamic/training.py` reviewed and changed on 2026-11-03 (ruff E501,F "All checks passed!", import OK; not run
on the server). Rows for it in `walkthrough-flags.md`: not written yet.

## 8. The open question
Jan's unchanged code and comments: keep them as he wrote them, or bring them to the Clean comment style? Clean already
rewrote many of his comments. Keeping his text keeps the comparison readable; rewriting gives one style. Ask Dirk this
one question first. Code deviations can be prepared before his answer; comment items only after it.

## 9. Next action
Ask the question of section 8; then build the deviation list of `fit_systems/blocks.py` (section 13) and wait for
Dirk to start reading it.

## 10. Acceptance criterion
Per file: every deviation from Jan's code is listed with a verdict (needed or not, reason true or not, marker present
or not); every agreed change is in the file exactly as Dirk approved it; comment-only edits pass
`comment_ast_check.py`; the end-of-file check of section 13 passes. The thesis numbers are proved only by the server
harness, later.

## 11. Read these first
1. `docs/decisions.md` D-282 (:8517): the renames, markers and comment standard of `Clean/`.
2. `docs/decisions.md` D-293 (:9141) and D-281 (:8493): reason clauses, no labels, the kept checks.
3. `Clean/gantry_dynamic/training.py` and `obc_gantry.py`: the latest accepted style.
4. `tasks/clean-code/walkthrough-flags.md`: past rulings and the fact-check table (skim; do not re-raise).

## 12. Do not, and how Dirk wants answers
- Answers of 1 to 3 lines; one complete list only when he asks about a whole file.
- Proposals: "Now:" then a python block with the lines exactly as in the file, "Proposed:" then its own python block;
  one change per pair; no "..." elisions. Example of an applied change:
  Now:
  ```python
          polish_only: True restores the weights only; False also the Adam moments, for a resume of Adam.
  ```
  Proposed:
  ```python
          polish_only: True loads the weights only, for a polish rerun or an evaluation; False also loads the Adam
              moments, so a resume continues Adam where it stopped.
  ```
- Line test before anything reaches him: every term defined where it is used, in plain words (Dirk rejected symbol
  tables for code that explains no formulas); no torch or Python words (state dict, native, object, attributes,
  lambda, .item()); no abbreviations (sd, sim, out, h); one name per thing; every claim checked in the code.
- Legends of names in a module docstring (`blocks.py:3-6`, `torch_nets.py:3`, `deepSI_corrections.py:3`,
  `closed_loop.py:4`): raise each as an item; the accepted style keeps a symbol list only where the file computes the
  formulas it lists (`obc_gantry.py`).
- Remove self-checks and checks that cannot fire on the thesis path (as for `training.py` on 2026-11-03); keep the
  D-281 ones.
- No reads, runs or files he did not ask for, beyond the preparation of the file he is on. "better" and "sounds good"
  are not approval.

## 13. Operational
Deviation list per Jan-derived file (a read-only script in the session scratchpad, never in the repository):
1. Parse Jan's and Clean's file with Python `ast`; strip docstrings and comments. Map Jan names to Clean names with
   the D-282 rename list first. Compare module-level statements (imports, constants) and every class and method by
   qualified name (`Class.method`). Classify each as identical, renamed only (difference only in renamed locals),
   changed, removed (Jan only) or added (Clean only).
2. Changed: show the code difference; docstring and comment changes separately.
3. Per deviation: needed by the thesis path (its caller in `Clean/`)? stated reason true and concise (in the code or
   `docs/decisions.md`; never invent one)? marker present (`@added` on a new function in a Jan file,
   `# CHANGED: <reason>` in a changed Jan function)?
4. Removed: confirm no caller in `Clean/`. For a class that subclasses deepSI (`SSE_Interconnect`,
   `WindowedSystemData`, `SystemDataNormalizer`), check whether the deepSI base class defines the removed method
   (e.g. Jan's `init_nets` and `measure_act_multi` in `SSE_Interconnect`, Jan `interconnect.py:396`, `:451`; Jan's
   `loss` there is replaced by `AugmentedModel.loss`); if so, deepSI's version now runs, so list it as a behaviour
   change, not dead code.
5. The four files of our own: line by line as in section 12.

A change that reaches other files: one grep over `Clean/`, `clean-up-verification/` and the harness folder; edit
callers in `Clean/` and `clean-up-verification/`; report harness callers to Dirk. Removing the `coordinate_version`
buffer (`blocks.py:436`, `:463`; its only reader in `training.py` was removed on 2026-11-03) changes the saved keys:
it would need `coordinate_version` in `DROPPED_STATE_KEYS` (`clean-up-verification/convert_run.py:30`), and the
harness mapping `is_dropped_state_key` (`compare_l3.py:56`) is Dirk's to report.

Pending harness item: `compare_l3.py:66-67` flags differing key sets; the clean pre-polish pair no longer has
`optimizer`.

End-of-file check when Dirk says a file is done, from `Clean/`:
`conda run -n GraduationProject python -m ruff check --select E501,F --line-length 120 <touched files>` and
`conda run -n GraduationProject python -c "import <touched modules>"`; report the output in one line. Re-run ruff
before calling any `interconnect.py` long line old (13 existed before 2026-11-03; line numbers may have moved).

Raise when the file is read, not before: `interconnect.py` `cal_validation_error` passes (model, validation records,
score) to each probe and only the score is used (Dirk: "no need to change interconnect" for now); `blocks.py` names
`free_params` (a parameter attribute), `combinations_from_free`, `transition_from_free`, `mats_and_tangent_from_free`,
`rk4_with_tangent`: whether each name says what it gives.

## 14. Delegation
None by default. Dirk allows at most two sub-agents for code work, only when he asks.
