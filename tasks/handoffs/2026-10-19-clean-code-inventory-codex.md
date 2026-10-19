# Prompt for Codex: independent inventory of the code the final grey-box method needs (static reading)
**From**: Claude session of 2026-10-19, for Dirk to give to Codex | **Branch**: Augmentation |
**Effort suggested**: high

## 1. Task
Dirk's words: "make a new self contained folder only needing the required python files and to be written with
minimal code", for the `scripts/gantry/gantry_interconnect_dynamic.py` pipeline. You do step 1 of 4 only: an
independent inventory of what that clean code must contain, by reading the code and the thesis. Write it to
`tasks/clean-code/inventory-codex.md` (the only file you create) in three parts:
- **Part A, required-code inventory.** Every file, class, function and branch in
  `scripts/gantry/gantry_interconnect_dynamic.py`, `scripts/gantry/gantry_dynamic/`, `scripts/gantry/common/` and
  `model_augmentation/` that executes for the spec of section 4, traced by hand from the config values each spec
  row produces. List what is NOT needed as well, per file. For `RunConfig` (`gantry_dynamic/config.py`): every
  field, whether the spec varies it, keeps it constant (value), or never reads it.
- **Part B, thesis-to-code table.** Every method statement and equation of
  `Thesis-writeup/Writing/sections/02_system_baseline.tex`, `03_augmentation.tex`, `04_obc.tex`, `05_setup.tex`
  (and the appendix parts they point to), mapped to the function that implements it, with a verdict: match,
  mismatch (what differs), in thesis but not in code, in code but not in thesis.
- **Part C, equivalence hazards.** Everything that would make a faithful rewrite give different numbers: random
  number draws or RNG state changes outside the required path (for example in diagnostics), state mutated as a
  side effect, operation order that changes float results, GPU or `torch.compile` only paths, dtype switches.
  Also the interface the evaluation scripts in `Thesis-writeup/Code/Evaluation/` use from this pipeline (imports,
  checkpoint and `state_dict` names, report and config fields), since the clean code must keep it loadable.

A Claude session does the same task independently, using a coverage trace. Do not look for or read its output
(`tasks/clean-code/inventory-claude.md`, `tasks/clean-code/coverage/`). Dirk compares the two afterwards; a
different method reaching the same list is the point.

## 2. Out of scope
- Writing clean code, creating the new folder, or proposing its design (step 2, after Dirk compares).
- Editing any existing file.
- Diagnostics code as a deliverable (`diagnostics.py`, `obc_diagnostics.py`, `closed_loop_report.py`, the
  diagnostic hooks in `training.py`): Dirk excluded it. List it under "not needed"; report it in Part C if removing
  it would change numbers.
- Table and figure scripts (`Thesis-writeup/Code/Evaluation/make_results.py`, `Thesis-writeup/Code/Figures/`): only
  their interface to the pipeline goes into Part C.
- The black box R8 (`scripts/gantry/blackbox-cl-trainable/`): a different pipeline.

## 3. Rules for this repository
- Read-only: no edits, no training or simulation runs. Quick import or syntax checks are fine.
- No git write operations of any kind (commit, push, reset, checkout, merge, pull, rebase). Read-only git is fine.
- Never read `kamtin-data/Data Telica/` or `kamtin-data/Telica.mat`. `kamtin-fp-model/` is read only.
- In `model_augmentation/`, project additions are marked: `@added` (new function or class),
  `__project_origin__ = "added"` (new file), `# CHANGED: <reason>` (changed block). Record the marker of each
  required item in Part A; the clean code keeps these markers.
- No em dashes in your output (not the Unicode character, not a double hyphen).

## 4. The spec (verified 2026-10-19)
- The final campaign is rows 221 to 230 of `Thesis-writeup/Code/Server/tables/runs.tsv` (9 rows, there is no
  228), launched by `Thesis-writeup/Code/Server/submit.sh` through
  `Thesis-writeup/Code/Server/greybox/run_thesis.sbatch`. It exports a row as `THESIS_ARM`, `THESIS_NOISE`,
  `THESIS_START`, `THESIS_NA`, `THESIS_SEED`, `THESIS_NF`, `THESIS_ARCH`, `THESIS_PS2`, `THESIS_INIT` and runs the
  entry script, whose THESIS block (about lines 291 to 380) turns them into the config.
- Across the 9 rows: arm obc or u, start detuned or correct, na 0, 2 or 8, ps2 0 or 1; fixed: noise lowpass,
  seed 1, nf 0.1, arch mlp, init live.
- Dirk adds one axis: the clean code must also run on the noise-free data. So the spec is the 9 rows with
  `THESIS_NOISE` lowpass or noisefree. The `noisy` option and all other data modes are out.
- `gantry_dynamic/data.py:64-66` maps the modes to `Thesis-writeup/Data/Coulomb-tanh-and-MSD-lowpass-noise` and
  `.../Coulomb-tanh-and-MSD-noise-free`; `config.py:33` applies anti-aliasing to the noisy modes only, so
  noise-free takes a different branch.
- `THESIS_SMOKE` and `THESIS_DEVICE` are test switches, not part of the method. Code reached only on the GPU
  (compilation) is part of the method run and belongs in Part A.

## 5. Context (agreed with Dirk 2026-10-19; for orientation, not to act on)
- Target folder: `gantry_interconnect_dynamic.py`, `gantry_dynamic/` and a trimmed `model_augmentation/`; what is
  needed from `common/` moves into `gantry_dynamic/`. Data stay outside, as two named entries.
- Equivalence is to be shown later by a layered harness against golden outputs of the current code (forward
  pass, loss and gradients, short seeded training, final checkpoint metrics per split, CPU float64). Part C feeds it.
- Later steps: Claude writes a design document, you and Claude critique it, Claude implements, you judge the code.

## 6. Read these first
1. `scripts/gantry/gantry_interconnect_dynamic.py`, the THESIS block.
2. `Thesis-writeup/Code/Server/greybox/run_thesis.sbatch` and `tables/runs.tsv` rows 221 to 230.
3. `tasks/final-run-code-changes-2026-10-16.md` and `Thesis-writeup/Documentation/FINAL-RUN-FIXES.md`: what the
   final method changed; decisions D-248 to D-271 in `docs/decisions.md`.
4. `Thesis-writeup/Writing/sections/03_augmentation.tex` and `04_obc.tex`.
5. `Thesis-writeup/Code/Evaluation/eval_common.py`: how the evaluation finds and loads a grey-box model.

## 7. Done when
- Part A accounts for every function and class in the four code locations: required (with the spec rows and the
  call path that reach it) or not needed. No function is unclassified.
- Part B covers every numbered equation and every method paragraph of Sections II to V, each row with a
  `file:line` you read yourself. Where thesis and code disagree, report both; do not decide which is right.
- Part C lists every RNG use in the four code locations with its position relative to the required path.
Report everything you find; Dirk filters.

Format example for Part A rows (illustrative names and lines):
`gantry_dynamic/training.py:410 | _orth_meter() | not needed | never called when orth=False (all spec rows) | call path: none`

Final message: the path of the inventory and any question for Dirk.
