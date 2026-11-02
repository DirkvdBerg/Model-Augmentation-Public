# Handoff: be Dirk's discussion partner while he reads the `Clean/` code himself
**From**: session of 2026-11-02 | **Branch**: Augmentation | **Effort suggested**: high (answers must be exact about
the code; edits only on Dirk's yes)

## 1. Task
Dirk reads the files of `Clean/` himself, in the order of section 9, and brings you what he wants to discuss:
questions ("what does this do", "why is this here"), doubts and flags. Dirk: "I dont like this type of walkthrough i
prefer walking through the file myself" and "i want a session to discuss it with". So: never lead, never present
blocks or files unasked; answer what he brings, short, from the code (cite `file:line`). For the files of
`Clean/model_augmentation/` that Jan also has, answer comparison questions against `Model-Augmentation-Public-main/`:
is this Jan's 1:1, changed by us (what, why, marked or not), or ours. Record each flag Dirk raises in
`tasks/clean-code/walkthrough-flags.md` (`Wnnn | file:line | flag | proposal | ruling`; create it at the first flag).
Apply a change only when Dirk says yes, then prove it (section 10).

## 2. Out of scope
- Leading the reading, summarising files he has not asked about, or bulk rewrites.
- The `final` snapshot, its pinned hashes and the step-5 server job: the parent session remakes them once after
  Dirk has finished reading.
- The black-box evaluation golden (D-285 amendment 6): the session "Clean black box to the clean-code rules" owns it.
- `Model-Augmentation-Public-main/`, `kamtin-fp-model/`, `Thesis-writeup/`: read only.

## 3. Where things stand
Tree dirty, Dirk commits. Grey-box comments in the style approved on 2026-11-01 (sample
`Clean/model_augmentation/fit_systems/ps2_opening.py`); `clean-up-verification/comment_ast_check.py` prints
`AST_IDENTICAL 24/24` against `simulations/clean-golden/src-snapshots/final-step5a/`. `Clean/blackbox/` cleaned under
D-285 amendment 6 (`tasks/clean-code/blackbox-audit.md`), golden `clean3` bitwise; launcher `run_black_box.sh`.

## 4. Established and verified
- Jan's reference: `Model-Augmentation-Public-main/` = `upstream/main` commit `8200923`. Jan's files also in `Clean/`:
  `fit_systems/blocks.py`, `interconnect.py`, `pre_encoder.py`, `utils/torch_nets.py`, `utils/utils.py`,
  `utils/deepSI_corrections.py`, `__init__.py`. Ours entirely: `closed_loop.py`, `lbfgs_polish.py`, `obc.py`,
  `ps2_opening.py`, `systems/gantry_ss.py`, all of `gantry_dynamic/` and `blackbox/`.
- Our changes to Jan's code, per function, with classes Extension / Technical / Narrowing / Functional:
  `tasks/clean-code/jan-changes-audit.md` (2026-10-27; predates the later D-284 steps, so check a function against the
  current file before quoting the audit).
- Markers: `@added`, `__project_origin__ = "added"`, `# CHANGED: <reason>`. A changed block without a marker is a flag.
- Clean-code rules: D-284 amendments 9 and 10 in `docs/decisions.md` (an option stays only if it selects a working
  alternative; no verification code or invented thresholds in `Clean/`). Comment style: short docstrings, plain-word
  numbered step comments with what and why, every parameter defined inline, no trivia.

## 5. Assumed but not verified
- That `jan-changes-audit.md` still holds per function; verify a function before relying on it.

## 6. Tried and failed
- A guided walk that presented one block at a time and waited for "clear" -> Dirk: "I dont like this type of
  walkthrough" -> he wants to read himself and discuss -> this handoff.

## 7. Achieved
- `Clean/` complete and proven by the D-284 harness and the D-285 goldens.

## 8. The open question
Nothing blocked.

## 9. Next action
Reply in one or two lines that you are ready, then wait for Dirk. His reading order (run order):
`gantry_interconnect_dynamic.py`, `gantry_dynamic/config.py`, `data.py`, `controller.py`,
`model_augmentation/systems/gantry_ss.py`, `fit_systems/blocks.py`, `utils/torch_nets.py`, `gantry_dynamic/model.py`,
`fit_systems/pre_encoder.py`, `interconnect.py`, `closed_loop.py`, `obc.py` + `gantry_dynamic/obc_gantry.py`,
`gantry_dynamic/training.py`, `fit_systems/ps2_opening.py`, `lbfgs_polish.py`, `gantry_dynamic/evaluation.py`,
`utils/utils.py`, `utils/deepSI_corrections.py`; then `Clean/blackbox/` (`bb.py`, `jan_model.py`, `train_bb.py`,
`eval_bb.py`, `select_bb.py`, `run_black_box.sh`).

## 10. Acceptance criterion
- Every flag Dirk raises has a row with his ruling.
- After a yes: comment-only edits -> `comment_ast_check.py` ends with `AST_IDENTICAL 24/24`; a code edit -> tell Dirk
  which golden proves it (grey box: the D-284 harness on the server; black box: `compare_bb.py` on a new training
  golden) and run nothing without his go.

## 11. Read these first
1. `Clean/README.md`: file roles and thesis sections.
2. `tasks/clean-code/jan-changes-audit.md`: our changes to Jan's code.
3. `docs/decisions.md` D-284 amendments 9 and 10.

## 12. Do not
- Lead the reading or present unasked blocks.
- Edit before Dirk's yes, or beyond it; never touch `Model-Augmentation-Public-main/`.

## 13. Operational
No runs except the AST check (seconds) and, when Dirk asks for a comparison with Jan, a scratch per-function AST diff
in the session scratchpad.

## 14. Delegation
None.
