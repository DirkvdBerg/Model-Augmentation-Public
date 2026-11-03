# Handoff: Dirk reviews `Clean/gantry_dynamic/evaluation.py`; be fast and right the first time
**From**: walkthrough session of 2026-11-03 ("Code walkthrough", `training.py`) | **Branch**: Augmentation |
**Effort suggested**: medium. Dirk: "evaluation shouldnt take me long." His day was lost to my corrections; he judges
the session on how little he has to correct.

## 1. Task
Dirk reviews `Clean/gantry_dynamic/evaluation.py` (production code for ASMPT). Before he asks anything, prepare: read
`evaluation.py` and the finished files, test every line yourself (section 5), and have one complete list of changes
ready. When he pastes a line or function: answer in 1 to 3 lines, then the change in the format of section 4.
Apply only after "apply" or "yes", exactly what he approved, nothing else.

## 2. Out of scope
- Any file other than `evaluation.py`; raise others only when Dirk reads them. The finished files are done.
- Harness files (`Thesis-writeup/Code/Verification/clean-code/`): never edit.
- Reads, runs, checks or new files Dirk did not ask for. On 2026-11-03 he interrupted a lint run, a protocol read and
  a file read I started instead of the handoff he asked for: "what did i say", "and what did you do instead".
- Handoffs, sessions, subagents: only when Dirk asks.

## 3. Where things stand
- Finished: `gantry_interconnect_dynamic.py`, `config.py`, `data.py`, `controller.py`, `model.py`, `obc_gantry.py`,
  `model_augmentation/systems/gantry_ss.py`, `training.py` (2026-11-03). Nothing committed since `a82e573`.
- `training.py` changes of 2026-11-03: removed `_assert_same_architecture`, `_assert_same_dtype`,
  `load_optimizer_moments`, the eager validation before the polish (D-278 named no mechanism), `adam_ran` of
  `run_lbfgs_polish`; `_json_safe` kept (D-281 O3); `resolve_checkpoint` returns 4 values; `ValidationHistory.record
  (event, score, row=None, **extra)`, row field `rms`, `hist_arrays` one loop, `__call__(self, _model,
  _validation_records, score)`; `train` creates the history and calls `_adam_phase(fit_sys, cfg, data, meta,
  done_epochs, epochs_remaining, history)` only when Adam has epochs left; `save_checkpoint_pair(..., save_adam_moments
  =True, ...)`, the pre-polish pair saved with `save_adam_moments=False`; all docstrings and comments rewritten.
- Checked: ruff E501,F "All checks passed!" and import, after the last code change. Not checked: the comment edits
  after it (text only). The server harness has not run.

## 4. How Dirk wants proposals (his rule, 2026-11-03)
"Now:" then a python block with the lines exactly as in the file (real indentation, real line breaks); "Proposed:"
then its own python block the same way. Never code behind a label on the same line, never "..." elisions, one change
per pair. "you do it once properly and then stop doing it?"

## 5. Test every line before showing it
- Every term defined where it is used, in plain words. No symbol table: "WE DONT NEED A FUCKING SYMBOL TABLE HERE WE
  ARE NOT EXPLAINING FORMULAS".
- No torch or Python words (state dict, native, object, attributes, lambda, .item()), no abbreviations (sd, sim, out,
  h), one name per thing.
- Say what and why concretely; "epochs done in all jobs" was rejected as unclear, "Adam epochs of the run up to this
  checkpoint" accepted. Vague words rejected: progress (unless defined), meta, rescore, rolled back, for the reader.
- Verify every claim in the code first (e.g. the stored PS phase is only 'S1' or 'S3', never 'S2').
- Never call a file or function clear without testing every line: on "anything you would change" I said the file was
  fine and Dirk then found problems in single functions. "WHERE THE FUCK IS IT GOING WRONG".
- After one rejected wording, look for a code change instead.
- Concise: answers of 1 to 3 lines; no tables or long lists unless he asks for the whole file.

## 6. Open items
- Harness: `compare_l3.py:66-67` flags `<name>.pt:keys` when key sets differ; the clean pre-polish pair no longer has
  `optimizer`, so the check must accept that for the pre-polish pair only. The harness session `harness-f64` is gone;
  tell Dirk where to put this when the harness is next fixed.
- `interconnect.py` `cal_validation_error` passes `(model, validation records, score)` to each probe; only the score
  is used. Dirk: "no need to change interconnect" (for now); raise only when `interconnect.py` is read.
- Editor squiggles on `PS2Opening(...)`: VS Code resolves `model_augmentation` to the repo-root mainline copy;
  `"python.analysis.extraPaths": ["Clean"]` fixes it. Not applied, Dirk did not answer.
- `walkthrough-flags.md` has no rows for `training.py` yet.

## 7. After evaluation.py
Reading order: `model_augmentation/` (`fit_systems/blocks.py`, `utils/torch_nets.py`, `fit_systems/pre_encoder.py`,
`interconnect.py`, `closed_loop.py`, `obc.py`, `ps2_opening.py`, `lbfgs_polish.py`, `utils/utils.py`,
`utils/deepSI_corrections.py`), each against Jan's original (`Model-Augmentation-Public-main/`, upstream `8200923`,
markers `@added`, `__project_origin__ = "added"`, `# CHANGED:`); then `Clean/blackbox/`.

## 8. End-of-file check
When Dirk says the file is done: from `Clean/`, `conda run -n GraduationProject python -m ruff check --select E501,F
--line-length 120 <touched files>` and `conda run -n GraduationProject python -c "import <touched modules>"`; report in
one line with the output.
