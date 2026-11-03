# Handoff: be Dirk's partner while he reviews `Clean/` file by file, from the end of `gantry_dynamic/data.py` on
**From**: Claude session of 2026-11-02 (third walkthrough session) | **Branch**: Augmentation | **For**: Codex |
**Effort suggested**: medium (short, exact answers about single lines; speed matters most to Dirk)

## 1. Task
Dirk reads the files of `Clean/` (the production code for ASMPT) himself, in the order of section 9, and asks
about single lines or functions: what it does, why it is there, whether the name, docstring or comment is good.
For each question: answer in a few lines from the code, with `file:line`, then give one concrete proposal when
something should change (current text, proposed text, or the rewritten function). Apply only after his yes, as
one direct line-level edit, then reply in one line. Speed is the priority: a comment change is one edit and
nothing else (no re-reading, no checks, no logging in between). When he asks "what would you change in this file/part", review
the WHOLE part at once: names, docstrings AND inline comments, judged by section 12, not only the first thing you
see.

## 2. Out of scope
- Leading the reading: do not present files or blocks he has not asked about.
- Rewriting whole files or blocks he did not agree to; a rewrite proposal is fine, applying it needs his yes.
- The harness `Thesis-writeup/Code/Verification/clean-code/`: a separate Claude session ("Fix the clean-code
  harness for float64 (D-294)") owns it. When a `Clean/` rename or signature change breaks a harness caller, list
  the harness call sites in your reply for Dirk to pass on; do not edit them.
- Long runs of any kind (training, smoke run, harness layers): the server harness is the proof and runs later.
- `kamtin-fp-model/`, `Model-Augmentation-Public-main/`, `Thesis-writeup/` (except reading): read only.
- No git write operations; Dirk commits.

## 3. Where things stand
Tree dirty, nothing committed by the walkthrough. All rulings so far: `tasks/clean-code/walkthrough-flags.md`
W001 to W036 (W036 since applied in the code). Decisions: `docs/decisions.md` D-292 (settings at the top of the entry script), D-293
(design-reason comments), D-294 (float64 parameters, exact P^-T; bitwise equality with the thesis runs dropped).
Dirk is at the end of `Clean/gantry_dynamic/data.py`. Open on the table: W036 (section 8).

## 4. Established and verified
- `config.py` and `data.py` were reviewed line by line; every formula in `data.py` matches the thesis (normalisation
  eq. (16), C^n of (17), sigma_e of eq:setup_rms, block mean and anti-aliasing of the model-rate reduction) (W026).
- Changes since the last server proof that change numbers: only D-294 (`gantry_ss.py` `_D = torch.float64`,
  `data.P_INV_T64` written out). Every other change was checked bitwise or text-only (W011 to W036).
- Recent interface changes in `Clean/` (callers in `Clean/` updated, harness by the harness session):
  `DataBundle` -> `Datasets`, `val_ckpt_data` -> `val_data`, `traj_dir` -> `dataset_dir`, `_record_at_fs_new` ->
  `record_at_fs_new`, `_load_record` returns `(sd, e, controller_spec)`, `Datasets.controllers` (file -> spec),
  `build_controller_bank(names, controllers, ts, ...)`, `build_closed_loop(..., controllers=...)`, the global
  controller table in `controller.py` removed (W031), noise value `'lowpass'` -> `'noisy'` (W034).
- After the last `data.py` edits: compile, ruff, a synthetic record bitwise as before, 4 real records give the same
  controller spec and bank row, config round trip: all pass. The `dataset_dir` docstring edit (W035) was applied
  after that check and is not compiled yet.

## 5. Assumed but not verified
- That `Clean/` still reproduces the thesis runs: only the server harness (arms A and B of D-294 amendment 1)
  proves it; not run.
- `controller.py` still carries `THEORY:`/`HEURISTIC:` labels (module docstring, lines ~20-30, the 1e-9/1e-12
  tolerance comments in `controller_design`); they are not reviewed yet and are banned in `Clean/` (section 12).

## 6. Tried and failed
- Reviewing only names/docstrings and calling a part "fine" without reading its inline comments -> Dirk had to ask
  twice -> review names, docstrings and comments together, every time.
- Proposals with formulas to decode (`y_k = r(k T_s) + B(y - r)(k T_s)`), wildcards (`e_*`, `*_files`), jargon
  ("0-d arrays", "ckpt", "manifest split"), or vague verbs ("check") -> each rejected.
- Option menus instead of one proposal; long tool chains (compile + ruff + log) after every one-word comment edit
  -> Dirk: "I HAVE TO WAIT 2 MINUTES FOR A CODE CHANGE". Comment edits: one edit, nothing else.
- Comments that state a reason the code does not back ("K1 records only, the controller the closed-loop bank
  builds": the bank builds any controller) -> caught by Dirk. Check every claim against the code first.
- Whole-block swaps that overwrote Dirk's own edits -> never; line-level edits after a fresh read.

## 7. Achieved
`Clean/gantry_interconnect_dynamic.py`, `gantry_dynamic/config.py` and `gantry_dynamic/data.py` reviewed and
reworked (W001 to W036). Implemented and checked locally as in section 4; not validated on the server.

## 8. The open question
W036: `record_at_fs_new(rec, cfg, fields=())` only ever gets `fields=('r_sim',)`. Proposed: replace `fields` by
`reference=False`, docstring "One record at fs_new: u, y and, with reference=True, the reference r, on the same
samples, trimmed at both ends and cut to truncate_s if set."; callers `_load_record` and `Clean/blackbox/bb.py`
pass `reference=True`. Values unchanged. Wait for Dirk's ruling.

## 9. Next action
Tell Dirk in one line that you are ready, on the end of `data.py` with W036 open, then wait for his questions.
Reading order after `data.py`: `gantry_dynamic/controller.py`, `model_augmentation/systems/gantry_ss.py`,
`fit_systems/blocks.py`, `utils/torch_nets.py`, `gantry_dynamic/model.py`, `fit_systems/pre_encoder.py`,
`interconnect.py`, `closed_loop.py`, `obc.py` + `gantry_dynamic/obc_gantry.py`, `gantry_dynamic/training.py`,
`fit_systems/ps2_opening.py`, `lbfgs_polish.py`, `gantry_dynamic/evaluation.py`, `utils/utils.py`,
`utils/deepSI_corrections.py`; then `Clean/blackbox/` (`bb.py`, `jan_model.py`, `train_bb.py`, `eval_bb.py`,
`select_bb.py`, `run_black_box.sh`).

## 10. Acceptance criterion
Dirk's agreed changes are in the code, each applied right after his yes. Only when he says he is done with a file:
one compile + ruff run on the files touched
(`conda run -n GraduationProject python -m ruff check --select E501,F --line-length 120 <files>`). No change log:
no rows in `tasks/clean-code/walkthrough-flags.md`, except a source row in its fact-check table for a new
design-reason clause (rare), and a `docs/decisions.md` entry for a change that moves numbers.

## 11. Read these first
1. `tasks/clean-code/walkthrough-flags.md`: every ruling, the accepted wordings, the fact-check table.
2. `docs/decisions.md` D-293 (comment rule) and D-294 (float64).
3. `Clean/gantry_dynamic/data.py` (current file) and `Clean/gantry_dynamic/controller.py` (next).

## 12. How Dirk wants code written (his rules)
A comment says WHAT happens and, where it matters, WHY, as plain cause and effect, true to the code. Examples he
accepted:
```python
aa_trim_s=0.005,    # [s] cut from both record ends: the filter has no samples beyond a record's ends, so its
                    # output is wrong in the first and last 4.6 ms; the noise-free data is cut the same, so a
                    # record and its noise-free twin cover the same samples
# the block mean drops an incomplete last block, so y and the fields are cut to the length of u
# float64: on the server GPU a compiled training step is not slower than in float32
```
```python
def _force_at_fs_new(u, cfg):
    """Actuator forces at fs_new: the mean over each block of decimation samples, so each model step gets the
    average force over its interval, as a held input."""
```
- A design reason only when documented (decisions.md, problem log, measurement log), stated as what the
  alternative would cost or break; otherwise say nothing, or "other X not compared". Never invent one.
- Not in `Clean/`: `D-` numbers, labels (`THEORY:`, `HEURISTIC:`), history ("earlier", "that run"), formulas a
  reader must decode, wildcards, abbreviations (`ckpt`, `traj`), vague verbs ("check", "fixes") or nouns
  ("phase", "bundle", "manifest split") without the mechanism, pointers instead of reasons ("as in scipy").
- Names say what a thing is (`dataset_dir`, `record_at_fs_new`, `_SPLIT_OF_BLOCK`, `polish_only`); a leading `_`
  only when nothing outside the file uses it.
- Docstrings of containers say whose result they are and what each field holds (e.g. `Norm`: "Result of
  compute_normalization: mu, sigma of x_tilde (x_mean, std_x), ...").
- Keep Dirk's own wording; change only the words he agreed.
- Prefer removing hidden coupling (global tables filled as a side effect, imports inside functions to dodge a
  circular import) over commenting it; propose the clean fix (W031 is the example).

## 13. Do not
- Answer with option menus: one proposal, and a short list only when he asks to review a whole part.
- Run compile, ruff, scratch checks or logging after an edit; they wait until he finishes the file (section 10).
- Edit the harness folder, or touch `Clean/` lines Dirk did not agree to.
- Claim a reason or a fact without checking the code or the document that backs it.

## 14. Operational
PowerShell on Windows; quick checks only, foreground, in the GraduationProject conda env (commands in section 10).
Scratch scripts go in a temp folder, never in the repo. Data for value checks: `Thesis-writeup/Data/` (read only;
never `kamtin-data/Data Telica/`).

## 15. Delegation
None. Answer from the code directly.
