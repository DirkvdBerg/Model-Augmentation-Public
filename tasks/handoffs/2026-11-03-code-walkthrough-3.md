# Handoff: be Dirk's fast partner while he reviews `Clean/` file by file, from the end of `gantry_dynamic/data.py`
**From**: session of 2026-11-02/03 (third walkthrough session) | **Branch**: Augmentation | **Effort suggested**: medium
(short exact answers about single lines; speed is what Dirk judges this session on)

## 1. Task
Dirk reads the files of `Clean/` (production code for ASMPT) in the order of section 9 and asks about single lines or
functions: what it does, why it is there, whether the name, docstring or comment is good. Answer in a few lines from
the code with `file:line` links and give ONE concrete proposal when something should change (now / proposed). Apply
after his yes. When he asks "what would you change in this part", review the whole part at once: names, docstrings
AND inline comments, judged by section 12.

Two kinds of change, two speeds:
- **Text only** (comment, docstring, a local variable name): one Edit, then a one-line reply. Nothing else in that
  turn: no re-read, no compile, no ruff, no scratch check, no flag-row logging, no task-list ceremony beyond what the
  app requires.
- **Functionality** (logic, a signature, a name used in other files, a moved constant): do it properly in one turn:
  one Grep for every use in `Clean/` and `clean-up-verification/`, edit all callers, then ONE quick check of a few
  seconds (compile + import of the touched modules; a value check in a scratch file only when numbers could change).
  If harness files call the changed code, send the call sites to the harness session (section 13) instead of editing
  them. Say in one line whether values change; a change that moves numbers also needs a `docs/decisions.md` entry
  (as D-294).

## 2. Out of scope
- Leading the reading: never present files or blocks Dirk has not asked about.
- Edits Dirk did not agree to; whole-block swaps (they overwrote his own edits before).
- The harness folder `Thesis-writeup/Code/Verification/clean-code/`: the session "Fix the clean-code harness for
  float64 (D-294)" (name `harness-f64`, id `local_fc31fca2-30a2-460b-aed3-7b58e7ada929`) owns it.
- Long runs (training, smoke run, harness layers, MATLAB): the server harness is the proof, later.
- Handoffs, new sessions, subagents: only when Dirk asks.

## 3. Where things stand
Tree dirty, Dirk commits. Rulings: `tasks/clean-code/walkthrough-flags.md`, rows W001 to W038 at the time of
writing; another session (Codex) works on `Clean/` in parallel and adds rows, so its last row is the latest state.
Decisions D-292, D-293, D-294 (+ amendment 1 by the harness session) in `docs/decisions.md`. `data.py` was reworked
whole after this handoff's author last read it (W036 to W038, Dirk's own edits); the files on disk are the current
state, never revert them, and do not re-raise what a row already settled.

## 4. Established and verified
- `gantry_interconnect_dynamic.py`, `gantry_dynamic/config.py`, `gantry_dynamic/data.py` reviewed and reworked; every
  formula in `data.py` matches the thesis (W026).
- Only D-294 changes numbers (float64 `gantry_ss.py`, exact `data.P_INV_T64`); all other changes were checked
  bitwise or are text only.
- Interface changes, `Clean/` callers done, harness clean side done by `harness-f64`: `Datasets` (was `DataBundle`),
  `val_data`, `dataset_dir`, `record_at_fs_new`, `_load_record` -> `(sd, e, controller_spec)`,
  `Datasets.controllers`, `build_controller_bank(names, controllers, ts, ...)`, `build_closed_loop(...,
  controllers=...)`, noise value `'noisy'` (was `'lowpass'`; data folder names unchanged).
- For the checks run after the later `data.py` rework, see rows W036 to W038.

## 5. Assumed but not verified
- `Clean/` reproduces the thesis runs: only the server harness (D-294 arms A and B) proves it; not run.
- `controller.py` still has `THEORY:`/`HEURISTIC:` labels (module docstring, design constants ~lines 20-30, the
  tolerance comments in `controller_design`); not reviewed yet.

## 6. Tried and failed
- compile + ruff + logging after every comment edit -> each edit took minutes -> Dirk: "I HAVE TO WAIT 2 MINUTES FOR
  A CODE CHANGE". Text edits get no follow-up tool calls (section 1).
- Calling a part "fine" after reading only names and docstrings -> Dirk had to ask twice -> read the comments too.
- Proposals with formulas to decode, wildcards (`e_*`), jargon ("0-d arrays", "ckpt", "traj"), vague verbs ("check")
  -> rejected each time.
- A comment reason the code does not back ("K1 records only, the controller the closed-loop bank builds") -> caught
  by Dirk. Check each claim against the code before writing it.

## 7. Achieved
See section 4. Implemented and checked locally; not validated on the server.

## 8. The open question
Nothing blocked. Where Dirk is: at or past the end of `data.py`; he says where he continues.

## 9. Next action
Reply in one line that you are ready and ask which file or line Dirk is on; then wait. Reading order after
`data.py`: `gantry_dynamic/controller.py`, `model_augmentation/systems/gantry_ss.py`, `fit_systems/blocks.py`,
`utils/torch_nets.py`, `gantry_dynamic/model.py`, `fit_systems/pre_encoder.py`, `interconnect.py`, `closed_loop.py`,
`obc.py` + `gantry_dynamic/obc_gantry.py`, `gantry_dynamic/training.py`, `fit_systems/ps2_opening.py`,
`lbfgs_polish.py`, `gantry_dynamic/evaluation.py`, `utils/utils.py`, `utils/deepSI_corrections.py`; then
`Clean/blackbox/` (`bb.py`, `jan_model.py`, `train_bb.py`, `eval_bb.py`, `select_bb.py`, `run_black_box.sh`).

## 10. Acceptance criterion
Each agreed change is in the code right after Dirk's yes. When Dirk says he is done with a file: one compile + ruff
run over the files touched for it (`conda run -n GraduationProject python -m ruff check --select E501,F
--line-length 120 <files>`). No change log: do not write rows to `walkthrough-flags.md` (it is a reference of past
rulings only). Two exceptions, both rare: a new design-reason clause gets its source row in that file's fact-check
table (D-293), and a change that moves numbers gets a `docs/decisions.md` entry.

## 11. Read these first
1. `tasks/clean-code/walkthrough-flags.md` (rulings and accepted wordings; skim, do not re-derive them).
2. `Clean/gantry_dynamic/data.py`, then `Clean/gantry_dynamic/controller.py`.

## 12. How Dirk wants code written
A comment says what happens and, where it matters, why, as plain cause and effect, true to the code. Accepted:
```python
# the block mean drops an incomplete last block, so y and the fields are cut to the length of u
# float64: on the server GPU a compiled training step is not slower than in float32
```
```python
def dataset_dir(cfg: RunConfig) -> str:
    """Folder of the run's data set: data_root plus DATASETS[cfg.noise], the noisy or the noise-free records. A noisy
    run also needs the noise-free records for its noise floor; evaluation finds each record's noise-free twin through
    the pair_file column of MANIFEST.csv."""
```
- A design reason only when documented, as what the alternative would cost or break; else nothing.
- Not in `Clean/`: `D-` numbers, `THEORY:`/`HEURISTIC:` labels, history, formulas to decode, wildcards,
  abbreviations, vague verbs or nouns without the mechanism.
- Names say what a thing is; a leading `_` only when nothing outside the file uses it.
- Prefer removing hidden coupling (global tables filled as a side effect, imports inside functions) over commenting
  it (W031).
- Keep Dirk's own wording; change only what he agreed.

## 13. Do not
- Follow a text edit with any other tool call.
- Offer option menus; one proposal (a short list only when he asks to review a whole part).
- Edit harness files: send changed call sites to `harness-f64` with SendMessage, in one message per change.

## 14. Operational
PowerShell works; Bash has crashed here. Quick checks foreground in the GraduationProject env; scratch scripts in the
session scratchpad. Data for value checks: `Thesis-writeup/Data/` (read only).

## 15. Delegation
None.
