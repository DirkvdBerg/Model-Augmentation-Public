# Handoff: continue as Dirk's discussion partner while he reads `Clean/`, from `gantry_dynamic/config.py` on
**From**: session of 2026-11-02 (second walkthrough session) | **Branch**: Augmentation | **Effort suggested**: medium
(short exact answers about single lines; speed matters to Dirk)

## 1. Task
Dirk reads the files of `Clean/` himself in the order of section 9 and asks about single lines: what it does, why it
is there, whether the wording is good. Answer each question in a few lines from the code with `file:line`, then give
one concrete proposal when something should change (current text, proposed text). Apply only on his yes, as line-level
`Edit`s after a fresh read of those lines, then run the quick checks of section 10. Record every flag and ruling in
`tasks/clean-code/walkthrough-flags.md` (next id W011). Dirk wants this FAST: he said "you are way to slow" and "i ask
for small changes and somehow this takes 10 minutes". One question, one short answer, one proposal; no surveys, no
long tool chains for a wording question.

## 2. Out of scope
- Leading the reading or presenting files and blocks he has not asked about.
- Rewriting whole blocks or files (see section 6); only the lines he agreed.
- The `final` snapshot, pinned hashes and the D-284 server harness run: the parent session redoes them after the
  walkthrough. Run nothing long (no smoke run, no harness) without his explicit go.
- `Model-Augmentation-Public-main/`, `kamtin-fp-model/`: read only. `Thesis-writeup/` read only EXCEPT the harness
  folder `Thesis-writeup/Code/Verification/clean-code/`, which Dirk allowed ("Yes, update it") when a `Clean/` change
  breaks a caller there.

## 3. Where things stand
Tree dirty, Dirk commits. Done in the first walkthrough session (all in `walkthrough-flags.md` W001-W010, decisions
D-292 and D-293 in `docs/decisions.md`):
- Settings at the top of `Clean/gantry_interconnect_dynamic.py` (MODE, RUN_DIR, RESUME_FROM, EVALUATE, SETTINGS);
  `RunConfig` fields without defaults; no PRESETS, no `hp`; PS and polish settings via cfg; black box uses
  `bb.GREY_BOX`. L-BFGS optimizer constants moved to module constants in `fit_systems/lbfgs_polish.py`.
- Entry script comments, help texts and function names reworked line by line with Dirk (`_deepsi_checkpoint_paths`,
  `_rerun_checkpoint`, `_archive_previous_outcome`). `n_a` accepts any integer >= 0. `start_phase` removed:
  `polish_only` goes from `main` into `load_checkpoint` and `train`. `_NOT_RECORDED` renamed `_PATH_FIELDS`.
- `config.py` comments fixed (module docstring, docstrings, per-field comments removed, group headers kept).

Dirk is reading `Clean/gantry_dynamic/config.py`. Open on the table (section 8): `y_decimation`.

## 4. Established and verified
- After the last edits: all touched files compile, ruff E501/F clean; `RunConfig(**SETTINGS)` round-trips through
  `config_json_dict` / `from_config_json` equal; `load_checkpoint(fit_sys, base_path, polish_only=False)`,
  `train(..., resume=None, polish_only=False)`.
- The smoke run (M_OBC axes) passed VAL_CHECK before the L-BFGS-constant move.
- `y_decimation` (`config.py:26`) is only text: written as `Y_DECIMATION` (`config.py:207`), printed at
  `data.py:182`, written by `clean-up-verification/convert_run.py:62`. Every part comes from recorded settings.
  `record_trim` is real logic (`data.py:143`, `eval_blackbox.py:46`).

## 5. Assumed but not verified
- Bitwise equivalence of all the code edits with the thesis runs: only the D-284 server harness proves it; not run.
- The smoke run has not run since the L-BFGS constants, renames, `start_phase` removal. `comment_ast_check.py` no
  longer gives AST_IDENTICAL against `final-step5a` (code changed), so it is not the proof any more.

## 6. Tried and failed
- Whole-block swaps of SETTINGS from a scratch copy -> overwrote Dirk's own edits and made a comment false ("PS
  opening ... before Adam"; S1 runs during the first Adam updates) -> Dirk angry. Always re-read and edit per line.
- Comments with reasons nobody documented or nobody can parse (examples below) -> rejected each time.
- Slow turns: subagent sweeps and long multi-file checks for a one-line wording question -> Dirk: too slow.

## 7. Achieved
See section 3; checks of section 4 passed. Not validated on the server.

## 8. The open question
`y_decimation`: my proposal (not yet ruled) is to remove it: config.json loses `Y_DECIMATION` (old folders still load,
the check in `from_config_json` only compares entries this code writes), `data.py:182` prints the settings directly,
`convert_run.py:62` drops the entry. Wait for Dirk's ruling.

## 9. Next action
Reply in one line that you are ready and on `config.py` (open: `y_decimation`), then wait for Dirk. Remaining reading
order: `gantry_dynamic/config.py` (in progress), `data.py`, `controller.py`, `model_augmentation/systems/gantry_ss.py`,
`fit_systems/blocks.py`, `utils/torch_nets.py`, `gantry_dynamic/model.py`, `fit_systems/pre_encoder.py`,
`interconnect.py`, `closed_loop.py`, `obc.py` + `gantry_dynamic/obc_gantry.py`, `gantry_dynamic/training.py`,
`fit_systems/ps2_opening.py`, `lbfgs_polish.py`, `gantry_dynamic/evaluation.py`, `utils/utils.py`,
`utils/deepSI_corrections.py`; then `Clean/blackbox/` (`bb.py`, `jan_model.py`, `train_bb.py`, `eval_bb.py`,
`select_bb.py`, `run_black_box.sh`).

## 10. Acceptance criterion
- Every flag Dirk raises has a row in `walkthrough-flags.md` with his ruling, and every reason clause added to a
  `Clean/` comment has a row in its "Design-reason fact check" table (location | clause | source file:line | quote).
- After an applied edit: the touched files compile and pass `ruff check --select E501,F --line-length 120`; a code
  edit also gets the quick import/round-trip check in the scratchpad (seconds) and a line telling Dirk that the
  server harness is the proof.

## 11. Read these first
1. `tasks/clean-code/walkthrough-flags.md`: every ruling so far and the fact-check table.
2. `docs/decisions.md` D-293 (design-reason comment rule) and D-292 (settings at the top).
3. `Clean/gantry_dynamic/config.py`: the file Dirk is on.

## 12. How to write comments (Dirk's rules, learned the hard way)
A comment says WHAT happens and, where it matters, WHY, in plain cause and effect. Concrete, short, true to the code.
- Name the mechanism, not an abstraction. Accepted:
  `# paths on the machine that runs the code; config.json leaves them out, so a run folder copied to another machine`
  `# (e.g. the cluster) uses that machine's paths`
  Rejected before it: "the folders belong to the machine, not to the run" (abstract, Dirk: "doesnt make sense").
- Cause and effect. Accepted:
  `aa_trim_s=0.005, # [s] cut from both record ends: the filter has no samples beyond a record's ends, so its`
  `# output is wrong in the first and last 4.6 ms`
  `# load first: if the checkpoint does not fit, load_checkpoint raises an error and the outputs are never moved`
  Rejected: "archive the run folder only after the checkpoint has loaded" (says what, not why), "stops the run"
  without saying how.
- A design reason only when documented (decisions.md, problem log, measurement log): the clause says what the
  alternative would cost or break. Accepted: `adam_eps=1e-16, # Adam epsilon; with the 1e-8 default the hidden layers
  do not train`; `nf_seconds=0.100, # [s] training window; twice the 50 ms settling time of the closed loop`. No
  documented reason: say nothing, or "other X not compared" once in a header. Never invent one (a false memory-use
  claim for `lbfgs_windows` was caught).
- Banned in `Clean/`: `D-` numbers, labels (THEORY:, HEURISTIC:, "Design:"), history ("earlier", "that run"), values
  repeated in the comment, formulas to decode ("sample i maps to fs_orig index (i + trim) D"), vague verbs without the
  mechanism ("check", "fixes", "forget its start", "damps out a force error"), vague nouns ("run family", "budgets",
  "phase", "code version", "dump", "pair" unexplained), pointers instead of reasons ("as in scipy's decimate"),
  properties presented as reasons ("-33 dB at Nyquist").
- Names follow the same rule: a name says what the thing is (`_PATH_FIELDS`, `polish_only`, `_rerun_checkpoint`),
  not a vague role (`_NOT_RECORDED`, `start_phase`, `_dumps`).
- Help texts and error messages: say what the flag does in words (`'run on the CPU, without torch.compile'`, not
  `'device cpu, eager'`).
- Before writing a comment that claims something, check it against the code that backs it. Dirk's own wording stays:
  change his comments only where he asks, at most the words he agreed.

## 13. Do not
- Swap whole blocks; edit lines Dirk did not agree to; overwrite his own edits (re-read the lines first, the file
  changes under you).
- Launch subagents or long runs for wording questions.
- Answer with option menus: one proposal.

## 14. Operational
Quick checks only, foreground, PowerShell:
`conda run -n GraduationProject python -m ruff check --select E501,F --line-length 120 <files>`;
`conda run -n GraduationProject python -m py_compile <files>`; scratch checks in the session scratchpad.
Bash has crashed intermittently here; PowerShell works.

## 15. Delegation
None.
