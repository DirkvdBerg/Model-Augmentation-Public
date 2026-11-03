# Handoff: be Dirk's fast partner while he reviews `Clean/` file by file, from `gantry_dynamic/obc_gantry.py`
**From**: walkthrough session of 2026-11-03 ("Code walkthrough 4") | **Branch**: Augmentation |
**Effort suggested**: medium (short exact answers about single lines; Dirk judges the session on speed and on how
little he has to correct)

## 1. Task
Dirk reads the files of `Clean/` (production code for ASMPT) in the order of section 9 and pastes single lines,
comments or functions, often with only "?", "is this clear?" or "what does this do". Answer the literal question in a
few lines from the code with `file:line` links. When something should change, give ONE proposal as now / proposed,
showing only the changed lines, and apply it after his "apply" or "yes". On "what would you change" for a function or
file, list every point at once, including the same kind of issue elsewhere in `Clean/` (a grep, every place in the
proposal). For `model_augmentation/` files, compare each file with Jan's original in the same read (section 9).

## 2. Out of scope
- Leading the reading: never present files or blocks Dirk has not asked about.
- The harness folder `Thesis-writeup/Code/Verification/clean-code/`: owned by the session "Fix the clean-code harness
  for float64 (D-294)" (`harness-f64`, id `local_fc31fca2-30a2-460b-aed3-7b58e7ada929`).
- Long runs (training, smoke run, harness layers, MATLAB): the server harness is the proof, later.
- Handoffs, new sessions, subagents: only when Dirk asks.

## 3. Where things stand
- Last commit `7d948e8` ("Update code for comments model.py"); `Clean/` has no uncommitted changes.
- Done (Dirk: "this file is done"): `gantry_dynamic/data.py`, `controller.py`, `model_augmentation/systems/gantry_ss.py`.
- `gantry_dynamic/model.py`: reviewed to the end and committed by Dirk; its end-of-file check (section 10) has NOT been
  run. Run it once Dirk says the file is done (or treat his commit plus "what file is next" as that, and run it first).
- Harness call sites sent to `harness-f64` (delivered, not confirmed done): `_sync_obc_coefficient` ->
  `sync_obc_coefficient` (`perf_diag.py:227`, `equation_tests.py:293`); `prefit_physical_encoder` -> `encoder_start`
  (`perf_diag.py:106`, `equation_tests.py:306`, `capture_l12.py:130`).

## 4. Established and verified
- `gantry_ss.py` now: `stiffness_and_damping` (was damping_and_stiffness), `inertia_adjugate(alpha, beta, gamma, mh, d)`
  (was build_poly_constants), `inertia_determinant(Y, alpha, beta, gamma, mh)` (new; also used at `blocks.py` in
  `_deriv_with`), `lfr_state_matrix(M0_inv, M1, M2, K, C)` returning only A_combined (was build_G_matrix_entries);
  `P` written out; `Cd`, `Dd` numpy (only data.py reads them; `Linear_Output_Block` turns them into buffers). Old vs
  new matrices and determinant bitwise equal at theta^0 and 4 perturbed theta (local check, this session).
- `model.py` now: `PHYSICAL_STATE_IX` -> local `physical_rows = np.arange(cfg.nx_phys)`; R^2 stop
  (`_MAX_ENCODER_UNEXPLAINED_FRACTION`), Adam-eps check, reproduce check and W^a check removed (self-checks of the code
  just above or invented thresholds); non-finite checks kept; CUDA check moved to the start of `main` in
  `gantry_interconnect_dynamic.py`; `encoder_start` (was prefit_physical_encoder); `least_squares_start` sets the
  physical bias to b directly (the old subtraction of the encoder output at zero subtracted exactly 0), splits W with
  `W.split`, takes its dtype from `xt`.
- "raw" is gone from `Clean/` (`parameter_names`, `nominal_parameters`, `_mats_from_parameters`, `combos_of(theta, Lb)`,
  `theta_init`; `obc.py` `gram`, `column_norms`); "start model" is "untrained model" everywhere.
- Fact-check row added: `walkthrough-flags.md`, `inertia_adjugate` cofactor clause (thesis `99_appendix.tex:225`,
  `:234-236`).
- The encoder start matches the thesis: start state `03_augmentation.tex:1387-1390`, fit eq:aug_enc_fit `:1395-1403`,
  bias b and zero weights `:1433-1436`. The normalisation is eq:aug_norm `:532-546`.

## 5. Assumed but not verified
- `Clean/` reproduces the thesis runs: only the server harness proves it; not run.
- `model.py` after this session's edits: compiled locally after the last structural edit, not ruff-checked, not run.
- `harness-f64` has applied the two renames: delivered, not confirmed.
- `obc.py` `J.T @ J` diagonal vs `J.norm(dim=0)`: kept as is because the rounding would differ and break the
  bitwise comparison on the reverted tree (`d294_revert.py`); not measured.

## 6. Tried and failed
Dirk's words in quotes; the full transcript is this session.
- Ran verify_quote and several greps before answering "anything you would change" -> six minutes for a style answer ->
  "This shouldnt take 6 minutes". Answer from the file as read; tools only when a change reaches other files.
- Ran ruff after the first edits of a file -> "its not fucking clear to you to do the ruff at the end????".
- Took "ow we apply these" (a question: "oh, these are functions being applied?") as approval and edited -> "NO i meant
  as in these arre functions BEING applied". Only "apply" or "yes" approves.
- Python-idiom or invented words in comments -> rejected each time: "with block", "stands in for the model",
  "mapped back", "original", "passes on the model's", "the second is the model's own", "as_". The accepted versions
  name the concrete thing and the reason literally (section 12).
- Withdrawing after "i dont find your suggestions improvements" -> "no suggest better when i say this". The better
  answer was a code change that removed the need for a comment (`W.split`, `zeros_like`, dropping `param`).
- Asserting a reason without reading the code (the Adam-eps check "guards a documented failure") -> reversed after
  reading `init_model` (`interconnect.py:403-406` passes the kwargs straight on). Read the callee before giving a reason.
- Adding text to fix vagueness -> "you are adding more text instead of being clear". Shorter and literal wins.

## 7. Achieved
Section 4; implemented, compiled, committed by Dirk; not validated on the server.

## 8. The open question
Nothing blocked.

## 9. Next action
Reply in one line that you are ready and that the next file is `gantry_dynamic/obc_gantry.py`; then wait. Reading order:
`gantry_dynamic/obc_gantry.py`, `gantry_dynamic/training.py`, `gantry_dynamic/evaluation.py`; then
`model_augmentation/` (`fit_systems/blocks.py`, `utils/torch_nets.py`, `fit_systems/pre_encoder.py`, `interconnect.py`,
`closed_loop.py`, `obc.py`, `ps2_opening.py`, `lbfgs_polish.py`, `utils/utils.py`, `utils/deepSI_corrections.py`), each
read together with its deviations from Jan's original (`Model-Augmentation-Public-main/`, upstream `8200923`; markers
`@added`, `__project_origin__ = "added"`, `# CHANGED:`): per deviation, is it needed and is its stated reason true and
concise. Whole-new files (`__project_origin__`) get a normal read. Then `Clean/blackbox/`.

## 10. Acceptance criterion
Each agreed change is in the code right after Dirk's "apply"/"yes" and matches what he agreed, at every place the
proposal named. Text edits: one Edit, then a one-line reply. A change that reaches other files: one Grep over `Clean/`
and `clean-up-verification/`, all callers edited, harness sites sent to `harness-f64` (one message per change); one
line on whether values change; a value check only when numbers could move. When Dirk says a file is done: one run over
the files touched for it, `conda run -n GraduationProject python -m ruff check --select E501,F --line-length 120
<files>` plus compile and import. No W rows in `walkthrough-flags.md`; a fact-check row only for a new design-reason
clause.

## 11. Read these first
1. `Clean/gantry_dynamic/obc_gantry.py`, the next file, as it is now.
2. `Clean/gantry_dynamic/model.py`, the latest accepted file: the current style reference.
3. `tasks/clean-code/walkthrough-flags.md` (past rulings and the fact-check table; skim, do not re-raise).

## 12. How Dirk wants code written
Wordings he accepted this session, each after plainer or shorter rounds:
```python
    # put both blocks in the model; step 3 connects them
    # cast the interconnect to the run's dtype: torch builds the network in its default float32
    # record each state and force of the simulation: closed_loop_rollout returns only the outputs, so recorder
    # wraps the model and saves its inputs x and u at every step
    # closed_loop_step calls recorder.output_only(x) for y_hat = C x, the output the controller needs at
    # each step; recorder uses the model's own function for it
    # a column that never changes has s = 0; give it scale 1, so it is not divided by zero
    # rcond: singular values below rcond times the largest count as zero; eps max(M, N), the numpy.linalg.lstsq default
    # the encoder starts as the fit, psi_b = W^b chi + b: psi_tilde's physical rows get output weights 0 and
    # bias b, so they add b for every window; the nonlinear correction psi_tilde is learned during training
```
- Keep his math (formulas with W^b, psi, mu, sigma); make the words around it literal: name the function, value or
  array, and the reason. No Python idioms, no metaphors.
- When the reason is in the thesis or Jan's papers, use it and point to it (equation label, `file:line`).
- When a comment needs a third round, look for a code change that makes it unnecessary.
- Checks: remove self-checks of the code just above and invented thresholds; keep checks for real failures
  (non-finite data, missing GPU). Raise each one with that reason; he decides (he kept the non-finite solve check).
- One dtype source: `cfg.dtype_np` / `cfg.dtype_pt`, or the data's dtype where no cfg is passed; no hard-coded float64.
- "not anywhere" means all of `Clean/`.
- Names keep his words; no abbreviations (`enc`, `sim`, `sdl`, `ixs`, `ic`, `raw` were replaced); a leading `_` only
  when nothing outside the file uses it.
- Earlier rules still hold: docstrings short, shared facts once, no `D-` numbers or THEORY/HEURISTIC labels in
  `Clean/`, design reasons only when documented (then a fact-check row).

## 13. Do not
- Edit before "apply"/"yes", or beyond what it covered.
- Run any tool before answering a style question, except the one Grep a cross-file change needs.
- Run ruff, compile or value checks before Dirk says the file is done.
- Follow a text edit with another tool call; end a proposal with a side question; offer option menus.
- Edit harness files.

## 14. Operational
PowerShell and Bash both work. Quick checks foreground in the GraduationProject env; scratch scripts in the session
scratchpad. Data for value checks: `Thesis-writeup/Data/` (read only).

## 15. Delegation
None.
