# Handoff: be Dirk's fast partner while he reviews `Clean/` file by file, from `gantry_dynamic/training.py`
**From**: walkthrough session of 2026-11-03 ("Code walkthrough 5", `obc_gantry.py`) | **Branch**: Augmentation |
**Effort suggested**: medium (short exact answers about single lines; Dirk judges the session on speed and on how
little he has to correct; his time is short)

## 1. Task
Dirk reads the files of `Clean/` (production code for ASMPT) in the order of section 9 and pastes single lines,
comments or functions, often with only "?", "is this clear?" or "what does this do". Answer the literal question in
1 to 3 lines from the code. When something should change, give ONE proposal as now / proposed code, only the changed
lines, and apply it after his "apply" or "yes". On "what would you change" for a function or file, give every point at
once after testing every line (section 12), in short form. For `model_augmentation/` files, compare each file with
Jan's original in the same read (section 9).

## 2. Out of scope
- Leading the reading: never present files or blocks Dirk has not asked about.
- Other files while reviewing one: e.g. the `blocks.py` names `free_params`, `combinations_from_free`,
  `transition_from_free`, `mats_and_tangent_from_free`, `rk4_with_tangent` are raised only when `blocks.py` is read.
- The harness folder `Thesis-writeup/Code/Verification/clean-code/`: owned by the session "Fix the clean-code harness
  for float64 (D-294)" (`harness-f64`, id `local_fc31fca2-30a2-460b-aed3-7b58e7ada929`).
- Long runs (training, smoke run, harness layers, MATLAB): the server harness is the proof, later.
- Handoffs, new sessions, subagents: only when Dirk asks.

## 3. Where things stand
- Last commit `a82e573` ("Update code for comments obc_gantry.py"); `Clean/` has no uncommitted changes.
- Done: `gantry_dynamic/data.py`, `controller.py`, `model.py`, `obc_gantry.py`; `model_augmentation/systems/gantry_ss.py`.
  The end-of-file check of `obc_gantry.py` ran in the previous session after its last edit (section 4).
- Harness changes 1 to 10 sent to `harness-f64` (delivered or queued, none confirmed applied): buffer and method
  renames, `correction_matrices`, `make_float64_block` removed, builder returns J only, `build_reference_set` without
  `n_a` and `verbose`, reference fields renamed, the correction holds the tuples, `make_basis_builder` /
  `make_reference_field` removed (methods `basis`, `network_field`).

## 4. Established and verified
- `obc_gantry.py` now: module docstring = purpose sentence + one line per symbol (tuples, beta_lin, varphi, J,
  F^n_aug, theta_aux, f_obc^n) + the [Phi Gamma] vs J relation; `GantryOBCCorrection(baseline_block, nx, nx_phys,
  dtype, reference, chunk)` stores two baseline functions (`baseline_matrices_and_derivative`,
  `baseline_step_and_derivative`) and `baseline_step`, buffers `beta_lin`, `theta_aux` (saved) and `baseline_input`,
  `network_input` (persistent=False); methods `set_theta_aux`, `correction_matrices`, `forward` (zero pad instead of
  E), `basis`, `network_field`; `attach_obc(fit_sys, baseline_block, cfg, data, norm)`, called from `model.py`.
- `interconnect.py:607`, `:647` read beta from `self.hfn.connected_blocks[self._mats_ix].free_params`;
  `obc.py` `maybe_refresh(epoch, beta)`; `blocks.py` `rk4_with_tangent` (leading `_` dropped).
- J from the live float64 block is bitwise equal to J from the former float64 copy (500 tuples, beta = 0 and a
  perturbed beta; scratch check, previous session).
- End-of-file check of `obc_gantry.py`, run after the last edit (the `attach_obc` signature): the touched modules
  import; ruff E501,F at 120 finds only 13 E501 in `interconnect.py` (lines 143, 178, 272, 276, 290, 302, 303, 306,
  488, 521, 538, 552, 562), long lines that existed before 2026-11-03.
- Fact-check rows in `walkthrough-flags.md`: fourth-order difference (212 Hz: 0.04 % vs 1.8 %), OBC velocity clause
  without Janot (Janot is not cited in the thesis).

## 5. Assumed but not verified
- `Clean/` reproduces the thesis runs: only the server harness proves it; not run.
- F^n_aug unchanged after moving the tuples into the correction: argued (same network, same tensor, same rows), not
  measured.
- Old Clean run folders no longer load (state-dict keys `vbar`, `theta_frozen`, `E` changed); thesis folders go
  through `convert_run.py` and are not affected. Not tested.
- `harness-f64` applied changes 1 to 10: not confirmed.

## 6. Tried and failed
Dirk's words in quotes.
- Ran ruff silently in reply to "which file is next", then wrote a handoff that moved the check to the next session
  -> "how ... can i trust your handoff if youre too ... stupid to even understand to do the ruff here". The check
  belongs to the session that finishes the file: answer the question first, then say in one line that the check
  runs, and run it.
- Called comments clear, then "those are the only problems left" -> Dirk found more each time ("YOU REALLY ... THINK I
  FIND THESE COMMENTS ACCEPTABLE"). Cause: judged correctness, not whether a reader understands.
- Own comments with jargon or undefined symbols -> rejected: "registered", "state dict", "kept in a list", "tangent",
  "objective", "protected", "stencil", "jacfwd", "z_i", "E", "the two baseline functions that f_obc^n uses".
- Rewording a comment two or three times -> "ARE YOU STUPID". The accepted fix each time was a code change: store two
  functions instead of the block in a list; pad instead of E; methods instead of nested factory functions; pass
  `baseline_block` instead of `connected_blocks[0]`.
- Long answers (tables, 18-point lists) -> "am i supposed to read all of this". Formulas without meaning -> "how am i
  supposed to read the fucking math".
- Proposing the `blocks.py` rename family during `obc_gantry.py` -> "stop waisting my fucking time".

## 7. Achieved
Section 4; implemented, committed by Dirk (`a82e573`), checked locally (ruff, import), not validated on the server.

## 8. The open question
Nothing blocked.

## 9. Next action
Say in one line that you are ready and that the next file is `gantry_dynamic/training.py`; then wait. Reading order
after it: `gantry_dynamic/evaluation.py`; then `model_augmentation/` (`fit_systems/blocks.py`, `utils/torch_nets.py`,
`fit_systems/pre_encoder.py`, `interconnect.py`, `closed_loop.py`, `obc.py`, `ps2_opening.py`, `lbfgs_polish.py`,
`utils/utils.py`, `utils/deepSI_corrections.py`), each read with its deviations from Jan's original
(`Model-Augmentation-Public-main/`, upstream `8200923`; markers `@added`, `__project_origin__ = "added"`,
`# CHANGED:`): per deviation, is it needed and is its stated reason true and concise. Then `Clean/blackbox/`.

## 10. Acceptance criterion
Each agreed change is in the code right after Dirk's "apply"/"yes" and matches what he agreed, at every place the
proposal named. A text edit: one Edit, then a one-line reply. A change that reaches other files: one Grep over
`Clean/` and `clean-up-verification/`, all callers edited, harness sites sent to `harness-f64` (one message per
change); one line on whether values change; a value check only when numbers could move. When Dirk says a file is done
or asks for the next file: answer first, then run the end-of-file check (section 14) over the files touched for it,
in this session, and report it in one line.

## 11. Read these first
1. `Clean/gantry_dynamic/training.py`, the next file.
2. `Clean/gantry_dynamic/obc_gantry.py`: the latest accepted file and the style reference (module symbol table,
   step comments with formulas, docstrings in words).
3. `tasks/clean-code/walkthrough-flags.md` (past rulings and the fact-check table; skim, do not re-raise).

## 12. How Dirk wants answers and code
Reply shape, accepted on 2026-11-03:
```
`connected_blocks[0]` is the baseline block only because build_model adds it first. Cleaner: pass it in.

Now:      baseline_block = fit_sys.hfn.connected_blocks[0]
Proposed: attach_obc(fit_sys, baseline_block, cfg, data, norm)
```
- 1 to 3 lines of answer, then now / proposed code. No tables or long lists unless he asks "what would you change"
  for a whole file. "better", "sounds good" are not approval; only "apply" or "yes".
- Before saying anything is clear, and before showing your own comment, test every line: does it name the concrete
  thing and its reason literally; is every symbol defined; no torch or Python words (registered, state dict, list
  trick, tangent, objective, closure, cache key).
- After one rejected wording, look for a code change that removes the need for the comment.
- Math, the accepted pattern (`obc_gantry.py`):
  - module docstring: one purpose sentence, then one line per symbol with its meaning and formula;
  - each formula once, as a step comment at the line that computes it, e.g.
    `# J = [Phi  col(f_base^n(beta_lin, x_tilde^n_i, u^n_i))], (6 N, 11)`;
  - docstrings say in words what the function gives, e.g. `"""J at beta_lin: at every tuple, how the baseline step
    changes with each of the 10 parameters, and the step itself."""`;
  - Args give the symbol and its meaning: `x: (B, nx, 1) x_hat^n_k, the normalised model state (physical and added)
    at step k; B the number of rollouts simulated in parallel.`
- Design reasons only when documented (fact-check row); derived reasons are fine when argued (the 212 Hz error
  numbers); never cite a source the thesis does not cite.
- Earlier rules still hold: names keep his words, no abbreviations, one dtype source (`cfg.dtype_np`/`dtype_pt`),
  no `D-` numbers or THEORY/HEURISTIC labels in `Clean/`, remove self-checks and invented thresholds.

## 13. Do not
- Run any tool in answer to a question, except the one Grep a cross-file change needs.
- Run ruff, compile or value checks while a file is still under review; they run once the file is done (section 10).
- Edit before "apply"/"yes", or beyond what it covered.
- Propose changes in files other than the one being read.
- Edit harness files.

## 14. Operational
End-of-file check, from `Clean/`, over the files touched for the finished file:
`conda run -n GraduationProject python -m ruff check --select E501,F --line-length 120 <files>`, then
`conda run -n GraduationProject python -c "import <touched modules>"`. `interconnect.py` has 13 old E501 lines (143,
178, 272, 276, 290, 302, 303, 306, 488, 521, 538, 552, 562); report them as old, not as findings. Under 30 s.

## 15. Delegation
None.
