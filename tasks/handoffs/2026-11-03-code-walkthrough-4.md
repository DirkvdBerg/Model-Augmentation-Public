# Handoff: be Dirk's fast partner while he reviews `Clean/` file by file, from `model_augmentation/systems/gantry_ss.py`
**From**: review of the walkthrough session of 2026-11-03 ("Code walkthrough 3") | **Branch**: Augmentation |
**Effort suggested**: medium (short exact answers about single lines; Dirk judges the session on speed and on how
little he has to correct)

## 1. Task
Dirk reads the files of `Clean/` (production code for ASMPT) in the order of section 9 and asks about single lines or
functions: what it does, why it is there, whether the name, docstring or comment is good. Answer what he asked, in a
few lines from the code with `file:line` links. When something should change, give ONE proposal as now / proposed,
showing only the lines that change, and apply it after his yes. When he asks for a whole function or file ("what
would you change", "what do you think I want changed"), list every point at once, including the same kind of issue
elsewhere (duplicates in other files, the same term used elsewhere), judged by section 12.

## 2. Out of scope
- Leading the reading: never present files or blocks Dirk has not asked about.
- Any edit before Dirk's yes to that exact proposal; whole-block swaps of his text.
- The harness folder `Thesis-writeup/Code/Verification/clean-code/`: owned by the session "Fix the clean-code harness
  for float64 (D-294)" (`harness-f64`, id `local_fc31fca2-30a2-460b-aed3-7b58e7ada929`).
- Long runs (training, smoke run, harness layers, MATLAB): the server harness is the proof, later.
- Handoffs, new sessions, subagents: only when Dirk asks.

## 3. Where things stand
- Dirk committed both finished files: `e460319` (data.py) and `633b33f` (controller.py and everything touched for it:
  `gantry_ss.py`, `blocks.py`, `obc_gantry.py`, `model.py`, `evaluation.py`, `eval_bb.py`, `ps2_opening.py`, plus the
  harness side). Uncommitted in `Clean/`: only the `import os` removal in `fit_systems/ps2_opening.py`.
- `tasks/clean-code/` has many untracked files from other sessions (Codex works on `Clean/` in parallel); leave them.
- Harness: a grep of the harness folder and `clean-up-verification/` for the old names (`train_data`, `_load_record`,
  `P_INV_T64`, `controller_design`, `normalized_controller_coefficients`, `mass_matrix_at(`, `_build_M1M2`,
  `except KeyError`) finds none.

## 4. Established and verified
- `gantry_dynamic/data.py` and `gantry_dynamic/controller.py` are done (Dirk: "this file is done.", twice).
- Interface changes made on 2026-11-03, all callers in `Clean/` updated, harness sites sent to `harness-f64`:
  - `data.py`: `_load_record` -> `load_record`; `Datasets` lost `train_data`/`val_data` (callers wrap
    `deepSI.System_data_list(data.train_list)` at the deepSI call); new `record_name(path)`, used by `evaluation.py`,
    `blackbox/eval_bb.py`, `ps2_opening.py`.
  - `P_INV_T64` -> `P_INV_T`, now defined in `gantry_ss.py:42` under `P`; all users import it from `gantry_ss`.
  - `gantry_ss.py` is the one home of the matrix formulas: `damping_and_stiffness` (:51), `inertia_constants` (:78),
    `inertia_parts` (:95), `inertia_matrix` (:117). `blocks.py` and `controller.py` call them; the controller's own
    copy of theta^0 is gone.
  - `obc_gantry.build_reference_set(..., records=data.train_list)` no longer reloads the `.mat` files.
  - `controller.py` names: `BANDWIDTH_HZ`, `BANDWIDTH_RAD_S`, `mass_matrix_at_fixed_y`, `actuator_response_at_fixed_y`,
    `controller_shape`, `axis_controllers`, `design_y_and_gain` (raises `ValueError`, was `KeyError`),
    `build_controller_bank(..., dtype, ...)` (no float32 default), `build_closed_loop(model, ...)`. `COMPILE_BACKEND`
    and `_stem` removed. `controller_ss` keeps its name (Dirk: "keep ss").
  - "head" -> "payload", "beam" -> "cross-arm" in all comments of `controller.py` and `gantry_ss.py`.
- Values: none moved. `controller_ss` old file vs new: bitwise equal at 5 Y x 2 ts x 2 gains. A snapshot of 77 arrays
  (model K, C, M parts and combos at theta^0 and a perturbed theta, controller matrices, OBC Z_step and Z_field on the
  18 training records) stayed bitwise equal after the `gantry_ss` move, the OBC reuse, `record_name` and the `D_fb`
  line. No `docs/decisions.md` entry needed.
- New design-reason clauses have fact-check rows: `walkthrough-flags.md:79-82` (OBC velocities, normalisation first
  difference, Tustin in `axis_controllers`, eager validation in `build_closed_loop`).

## 5. Assumed but not verified
- `Clean/` reproduces the thesis runs: only the server harness proves it; not run.
- `gantry_ss.py`'s four new functions and `P_INV_T` were written by that session and checked bitwise, but Dirk has not
  read them line by line; that happens now, as the next file.
- `controller.py:52` reads `A =np.block(...)` (space missing); flagged, not changed, may be Dirk's own edit.
- Dirk asked where the thesis's residual closed loop (x^c = 0 at the window start, so the encoder estimates only the
  model states, `03_augmentation.tex:838-898`) is done. Answer given: `closed_loop.py:50-93` and `:116-118`;
  `build_closed_loop` step 2 now only points there ("residual form, closed_loop.py").

## 6. Tried and failed
Each with Dirk's own words; the full transcript is `C:\Users\20203253\Downloads\session-export-1793706039114.zip`.
- Started a multi-file edit on a "good" that answered the earlier, smaller proposal -> "wtf are you doing without
  discussing wtf", "that was for the previous message". A yes applies only to the proposal it answers; a change that
  reaches other files or fields gets its own yes.
- Applied only part of a proposal and asked a side question at its end ("Should I change all of them?"); Dirk said
  "apply", the session changed one docstring -> "how is it possible that this still uses head ... youre being very
  inconsistent". Put every affected place in the proposal; "apply" covers all of it.
- Asked separately about a third copy of the same duplicate -> "why do i need to keep spelling out these things".
  Same-kind findings go into the same proposal.
- Rewrites that grew or dropped content -> "no this is worse", "youre adding too much", "i dont think this sounds
  clean", "this looks really messy", "why are you removing useful information", "your doc is too long", "again too
  fucking long", "your suggestions are terrible". Start from his line and change only the weak part.
- `build_closed_loop` docstring rewritten three times, each longer -> rejected; what was accepted was four one-line
  edits (section 12 examples).
- When a wording failed, escalating to a code change (make `s` an `f_hz` argument) -> "no wtf is solution". Go to the
  standard term instead.
- Renames that dropped his words -> "no i like mass_matrix", "we state actuator for a reason", "thats too long",
  "keep the name".
- A claim made without reading the source (the data generator tunes on a ZOH plant) -> corrected only after "what is
  your reasoning"; `ruleOfThumb.m` showed the continuous plant. Read the source before stating it.
- ruff and import checks after single edits -> "the ruff thing and checks were fucking once per file wasnt i".
- Repeating an explanation in longer or more analogical form -> "i still dont get it", "i wont be able to explain
  this", "i dont understand it well enough". What worked: answering the literal question (when would a bias term be
  needed: the four cases), naming the property (linearity of the mean), and a line-by-line match with the MATLAB
  (`ruleOfThumb.m`, `gtd_build_plant.m:15`).

## 7. Achieved
See section 4: implemented, compiled, ruff-clean, bitwise-checked locally; not validated on the server.

## 8. The open question
Nothing blocked.

## 9. Next action
Reply in one line that you are ready and ask which file or line Dirk is on; then wait. Reading order:
`model_augmentation/systems/gantry_ss.py`, `fit_systems/blocks.py`, `utils/torch_nets.py`, `gantry_dynamic/model.py`,
`fit_systems/pre_encoder.py`, `interconnect.py`, `closed_loop.py`, `obc.py` + `gantry_dynamic/obc_gantry.py`,
`gantry_dynamic/training.py`, `fit_systems/ps2_opening.py`, `lbfgs_polish.py`, `gantry_dynamic/evaluation.py`,
`utils/utils.py`, `utils/deepSI_corrections.py`; then `Clean/blackbox/` (`bb.py`, `jan_model.py`, `train_bb.py`,
`eval_bb.py`, `select_bb.py`, `run_black_box.sh`).

## 10. Acceptance criterion
Each agreed change is in the code right after Dirk's yes, and matches what he agreed. Text edits: one Edit, then a
one-line reply, no other tool call. A change that reaches other files: one Grep for every use in `Clean/` and
`clean-up-verification/`, edit all callers, harness sites to `harness-f64`; say in one line whether values change; a
value check only when numbers could move (in the background), and a `docs/decisions.md` entry if they do. When Dirk
says a file is done: one compile + ruff + import run over the files touched for it
(`conda run -n GraduationProject python -m ruff check --select E501,F --line-length 120 <files>`). No W rows in
`walkthrough-flags.md`; only a fact-check row for a new design-reason clause.

## 11. Read these first
1. `Clean/model_augmentation/systems/gantry_ss.py`, the next file, as it is now.
2. `Clean/gantry_dynamic/controller.py`, the latest file Dirk accepted: the current style reference.
3. `tasks/clean-code/walkthrough-flags.md` (past rulings and accepted wordings; skim, do not re-derive or re-raise).

## 12. How Dirk wants code written
Wordings he accepted on 2026-11-03, each after shorter or plainer rounds:
```python
        s: Laplace variable [rad/s].
        norm: means and standard deviations of the training records (Norm).
        three (b, a) pairs: the numerator and denominator coefficients of each axis's K_1(z).
    # compiled for the training rollouts only; validation runs eager: deepSI moves the model to the CPU for every
    # validation, where a compiled step would recompile each time
    # row divided by its position's standard deviation. No bias term is needed: y = P^T q holds for every sample, so
    # the mean of y is P^T times the mean of q.
        raise ValueError('sigma_e divides the loss and NRMS_e, so it must be finite and positive, got %s' % sigma_e)
```
- His existing text stays; the proposal adds the missing clause or swaps the weak words, never longer than needed.
- Docstrings short; shared facts once in the docstring, not one `#` per field ("without the comments for Datasets").
- Say what a thing holds, not its class name ("this feels terrible: deepSI System_data").
- A design reason only when documented (then a fact-check row); else nothing.
- Names keep his words; short; a leading `_` only when nothing outside the file uses it.
- One source for each formula or constant ("this is way too messy and bad coding habit"), unless a documented reason
  needs the copy ("unless there is a reason for it of course").
- Not in `Clean/`: `D-` numbers, `THEORY:`/`HEURISTIC:` labels, history, formulas to decode, wildcards,
  abbreviations, vague words.
- When he does not know the right choice ("i dont know the correct choice to be honest"): one recommendation with the
  deciding reason.

## 13. Do not
- Edit before his yes, or beyond what the yes covered.
- Follow a text edit with any other tool call.
- End a proposal with a side question; offer option menus.
- Edit harness files: send changed call sites to `harness-f64` with SendMessage, one message per change.

## 14. Operational
PowerShell works. Quick checks foreground in the GraduationProject env; scratch scripts in the session scratchpad.
Data for value checks: `Thesis-writeup/Data/` (read only).

## 15. Delegation
None.
