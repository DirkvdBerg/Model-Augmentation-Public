# Handoff: bring `Clean/blackbox/` to the clean-code rules and comment style of the grey box (D-285 amendment 6)
**From**: session of 2026-11-01 | **Branch**: Augmentation | **Effort suggested**: high (audit and judgment over
about 500 lines; two short local runs, no derivations)

## 1. Task
Dirk: "do you remember the rules we have setup for minimum optimal code? i would also like to do that for the black
box code. and make the comments as we have just defined with the clear function docstring and naming." Do it in two
phases. **Phase 1, audit (no edits):** write `tasks/clean-code/blackbox-audit.md` with one row per constant, setting
(argparse flag), key (`result.json`, `val_hist.json`), function, class and comment block of `Clean/blackbox/`
(`bb.py`, `train_bb.py`, `eval_bb.py`, `select_bb.py`, `jan_model.py`, `run_r8.sh`): verdict `keep` (with the
working alternative it selects, or the reason it is needed), `remove`, `rename old -> new` (thesis name, with the
grey-box name it should match), or `move`. Include the proposed settings block for the top of each entry script and
a fix for the grid file outside `Clean/` (section 4). Stop and give Dirk a short summary (one line per proposed
change). **Phase 2, after his ruling:** apply exactly what he ruled, rewrite the comments in the style of
`Clean/model_augmentation/fit_systems/ps2_opening.py`, and prove the output unchanged (section 10).

## 2. Out of scope
- `Clean/gantry_dynamic/`, `Clean/model_augmentation/`, `Clean/gantry_interconnect_dynamic.py`: another session is
  rewriting comments there now; the black box imports from them, never edits them. A wish to change them goes in
  your report.
- `Clean/data_generation/` (D-286), the evaluation and figures cleanup (todo item 8).
- `scripts/gantry/blackbox-cl-trainable/`: the mainline reference that produced R8, unchanged.
- `Thesis-writeup/Results/`: never written. Outputs go under `simulations/clean-golden/blackbox/`, new folders only.
- The `final` snapshot, its hashes and the step-5 server job (`tasks/clean-code/implementation-report.md`).
- Moving the black-box compare scripts out of `Thesis-writeup/Code/Verification/clean-code/`; use them where they are.

## 3. Where things stand
Tree dirty, Dirk commits. `Clean/blackbox/` was built and proven on 2026-10-27/28 (`tasks/todo.md`, section "ACTIVE
2026-10-27 - Clean copy of the thesis black box R8 (D-285)"). Open there: 4b (re-run `clean1` after the grey-box
changes), 2/6 (server evaluation golden). This task closes 4b on the way (section 9).

## 4. Established and verified
- The rules, D-284 amendments 1, 9, 10 (`docs/decisions.md`): only the final path; an option stays only if it selects a
  working alternative a user of the method could choose (amendment 10: "use_gain, up_sample, baseline_form, float32
  branches ... stay"); write-only values, fixed values with no code behind them and branches that cannot run go;
  interfaces and "mainline has it" are no keep reason; no verification code, test modes or harness names in `Clean/`;
  no invented thresholds (an existing exact guard such as the weight-row check in `make_training_data` is a judgment
  call for the audit, not an automatic removal); settings as parameters at the top of the entry script, a CLI flag may
  override, never CLI-only (see `Clean/gantry_dynamic/config.py` and `Clean/gantry_interconnect_dynamic.py` for the
  grey-box form).
- Gaps seen on a read (2026-11-01), to be confirmed by the audit:
  - `train_bb.py:16-24` settings are argparse-only and `--out` is required.
  - `run_r8.sh` reads `Thesis-writeup/Code/Server/blackbox/thesis_r8_grid.conf`, a file outside `Clean/` that ASMPT will
    not have.
  - D-285 point 2 kept `--truncate-s` and `--n-its` "for the golden runs" (a verification reason); the grey box keeps
    the same two as working options (`truncate_s`, `n_its` in `RunConfig`), so they likely stay on that ground.
  - Comments in the pre-2026-11-01 style: "Conventions:" legends, "(Table hyper)", the `__dict__` reload note in
    `ServoWeightedEncoder`.
  - Short names: `pa`, `sds`, `tr_f`, `yhat`; `NA` is `na_nb` in the grey box.
- Fixed by the pickled thesis checkpoints, never renamed: the module name `jan_model` and the classes `JanEncoder`,
  `JanStateNet` (D-285 amendment 2). Check whether `ServoWeightedEncoder` is pickled in the `_best.pth` before
  proposing to rename it.
- The comment style (approved 2026-11-01, sample `ps2_opening.py`): module docstring two lines at most, no symbol
  legend; numbered step comments that say what the step does and why, word first, symbol after; formula comments stay
  under the step line; every setting and parameter defined inline, one line, unit in brackets; no trivia (seed
  offsets, memory-order notes, internals, history wording, D-numbers). Example from the sample:
  ```python
  # 4. Adam step on decoder and encoder parts only; the main training's gradients are put back afterwards.
  ```
- Goldens (`tasks/todo.md` D-285 section): mainline `simulations/clean-golden/blackbox/run1`, clean `clean1`; flags
  `--truncate-s 3 --n-its 200 --its-per-val 100` at grid index 9 defaults, local CPU 4 threads, about 200 to 300 s; a
  repeat is bitwise except deepSI's unseeded checkpoint code, the output path and `batch_feq`. Evaluation golden
  `simulations/clean-golden/blackbox/eval_clean` (4 `preds_BB*.npz`, R8 checkpoint in
  `Thesis-writeup/Results/24-10-2026/Black-box/9_nx16_w16_lr1e-3_n80000_89756`), 1421 s locally.
- Compare scripts: `Thesis-writeup/Code/Verification/clean-code/compare_bb.py <golden> <other>` (training) and
  `compare_bb_preds.py <golden> <other>` (predictions), both print a bitwise verdict.

## 5. Assumed but not verified
- That `clean1` still reproduces with today's `Clean/gantry_dynamic` (the grey box changed after it, todo 4b). Section
  9 step 1 settles it before any edit.
- That `select_bb.py` needs no new golden: it was proven on `simulations/clean-golden/blackbox/grid_89756` against
  `Results/24-10-2026/CSV/res_bb_selection.csv`; if phase 2 touches its code, rerun that comparison (seconds).

## 6. Tried and failed
- Comment passes by answering cold-read questions (D-284 amendment 11) -> dense, hard-to-read comments, Dirk: "i think
  the current comments in the files are bad" -> answering every question adds a line per question -> the style is now
  set by example (section 4).
- A first trimming pass to that style -> too thin, Dirk: "for the context and understanding i dont find it enough" ->
  the explanation belongs in the step comments, not in long docstrings ("i dont want such a long docstring").

## 7. Achieved
- `Clean/blackbox/` exists and was proven bitwise against mainline (training golden, evaluation golden, selection
  table), D-285 amendments 1 to 5.

## 8. The open question
Nothing blocked. Phase 2 waits for Dirk's ruling on the audit.

## 9. Next action
1. Re-run the current, unedited `Clean/blackbox/train_bb.py` with the golden flags into
   `simulations/clean-golden/blackbox/clean2` and compare it with `clean1` (closes todo 4b). Bitwise: `clean2` is the
   reference for phase 2. Not bitwise: stop and report the differing fields to Dirk, no edits.
2. Write the audit (section 1) and stop for Dirk.
3. After his ruling: edit, then rerun the training golden into `clean3` and compare it with `clean2`. Ask Dirk
   whether the evaluation golden (about 24 min locally) runs locally or with his next server batch; then compare its
   predictions with `eval_clean`.

## 10. Acceptance criterion
- Every constant, setting, key, function, class and comment block of the six files has an audit row.
- `compare_bb.py clean2 clean3` and `compare_bb_preds.py eval_clean <new eval>` print bitwise identical (deepSI's
  checkpoint code, output path and timing fields excepted, as in `clean1` vs `run1`).
- No file in `Clean/blackbox/` reads a path outside `Clean/` except the data root and the run's output folder.
- Results recorded in the D-285 section of `tasks/todo.md`.

## 11. Read these first
1. `docs/decisions.md`: D-284 amendments 9 and 10 (the rules), D-285 and its amendments 1 to 6.
2. `Clean/model_augmentation/fit_systems/ps2_opening.py`: the comment style.
3. `Clean/gantry_dynamic/config.py` and `Clean/gantry_interconnect_dynamic.py`: the settings form and names to match.
4. `tasks/todo.md`, D-285 section: golden folders and flags.
5. `Clean/blackbox/` itself (463 Python lines).

## 12. Do not
- Edit before Dirk's ruling, or beyond it.
- Rename `jan_model`, `JanEncoder`, `JanStateNet`, or anything the thesis checkpoints pickle.
- Write over an existing golden folder or into `Thesis-writeup/Results/`.
- Add checks, thresholds, options or test modes to `Clean/`.

## 13. Operational
Training golden from the repo root (check the exact form in `tasks/todo.md`; the run imports `bb` from its own folder):
`PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output -n GraduationProject python -u
Clean/blackbox/train_bb.py --truncate-s 3 --n-its 200 --its-per-val 100 --out simulations/clean-golden/blackbox/clean2`,
in the background, about 200 to 300 s. One local numerical job at a time: another session may run only the static AST
check now.

## 14. Delegation
None; the code is small and one consistent voice decides the names and comments.
