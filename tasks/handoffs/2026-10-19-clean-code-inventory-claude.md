# Handoff: independent inventory of the code the final grey-box method needs (Claude, dynamic trace)
**From**: session of 2026-10-19 | **Branch**: Augmentation | **Effort suggested**: xhigh (wide read across
two packages plus a thesis-to-code mapping)

## 1. Task
Dirk's words: "make a new self contained folder only needing the required python files and to be written with
minimal code", for the `gantry_interconnect_dynamic.py` pipeline. This session does step 1 of 4 only: an
independent inventory of what that clean code must contain. Produce three things, from the code and the thesis
alone, and write them to `tasks/clean-code/inventory-claude.md` (the one file you create, plus the coverage data
of section 13):
- **Part A, required-code inventory.** Every file, class, function and branch in
  `scripts/gantry/gantry_interconnect_dynamic.py`, `scripts/gantry/gantry_dynamic/`, `scripts/gantry/common/` and
  `model_augmentation/` that executes for the spec of section 4, with its evidence (the coverage trace of section 9,
  or your reading for code the trace cannot reach, marked as such). List what is NOT needed as well, per file.
  For `RunConfig` (`gantry_dynamic/config.py`): every field, whether the spec varies it, keeps it constant (value),
  or never reads it.
- **Part B, thesis-to-code table.** Every method statement and equation of
  `Thesis-writeup/Writing/sections/02_system_baseline.tex`, `03_augmentation.tex`, `04_obc.tex`, `05_setup.tex`
  (and the appendix parts they point to), mapped to the function that implements it, with a verdict: match,
  mismatch (what differs), in thesis but not in code, in code but not in thesis.
- **Part C, equivalence hazards.** Everything that would make a faithful rewrite give different numbers: random
  number draws or RNG state changes outside the required path (for example in diagnostics), state mutated as a side
  effect, operation order that changes float results, GPU or `torch.compile` only paths, dtype switches.
  Also the interface the evaluation scripts in `Thesis-writeup/Code/Evaluation/` use from this pipeline (imports,
  checkpoint and `state_dict` names, report and config fields), since the clean code must keep it loadable.

A second, independent session (Codex) does the same task by static reading. Do not look for or read its output.

## 2. Out of scope
- Writing any clean code, creating the new folder, or proposing its design: that is step 2, after Dirk compares
  the two inventories.
- Editing any existing file. The inventory is read-only apart from the files named in section 1 and 13.
- Diagnostics code as a deliverable (`diagnostics.py`, `obc_diagnostics.py`, `closed_loop_report.py` and the
  diagnostic hooks inside `training.py`): Dirk excluded it. List it under "not needed", but report it in Part C
  if removing it would change numbers.
- The table and figure scripts (`Thesis-writeup/Code/Evaluation/make_results.py`, `Thesis-writeup/Code/Figures/`):
  out for now. Only their interface to the pipeline goes into Part C.
- The black box R8 (`scripts/gantry/blackbox-cl-trainable/`): a different pipeline.
- Thesis text edits, server jobs, git writes.

## 3. Where things stand
Branch `Augmentation`, last commit 4271a31, large dirty tree (thesis, docs and script edits, not committed; leave
them). No run in flight. The final grey-box campaign ran 2026-10-16 to 10-18 (logs
`Thesis-writeup/Results/18-10-2026/thesis-results-4/*.out`, one per run id).

## 4. Established and verified
- **The spec.** The final campaign is rows 221 to 230 of `Thesis-writeup/Code/Server/tables/runs.tsv` (9 rows,
  there is no 228), launched by `Thesis-writeup/Code/Server/submit.sh` through
  `Thesis-writeup/Code/Server/greybox/run_thesis.sbatch`, which exports the row as `THESIS_*` variables and runs
  `scripts/gantry/gantry_interconnect_dynamic.py`. Its THESIS block (about lines 291 to 380) turns them into the
  config. Across the 9 rows: arm obc or u, start detuned or correct, na 0, 2 or 8, ps2 0 or 1; fixed: noise
  lowpass, seed 1, nf 0.1, arch mlp, init live.
- **Dirk adds one axis:** the clean code must also run on the noise-free data. So the spec is the 9 rows with
  `THESIS_NOISE` lowpass or noisefree. The `noisy` option and every other data mode are out.
- Data modes: `gantry_dynamic/data.py:64-66` maps `thesis_taf_lowpass_noise` to
  `Thesis-writeup/Data/Coulomb-tanh-and-MSD-lowpass-noise` and `thesis_taf_noisefree` to
  `.../Coulomb-tanh-and-MSD-noise-free`. `config.py:33` applies anti-aliasing to the noisy modes only, so the
  noise-free mode takes a different branch.
- Sizes: entry 475 lines; `gantry_dynamic/` 11 files, about 4770 lines (`config.py` 811, `training.py` 1080);
  `common/` 4 files, about 880 lines; `model_augmentation/` about 9600 lines. The pipeline imports
  `common.orth_penalty`, `common.rezero_gate` and about ten `model_augmentation` modules.
- The final method's code changes: `tasks/final-run-code-changes-2026-10-16.md`,
  `Thesis-writeup/Documentation/FINAL-RUN-FIXES.md`, decisions D-248 to D-271 in `docs/decisions.md`.

## 5. Assumed but not verified
- That a smoke run (`THESIS_SMOKE=1`, `THESIS_DEVICE=cpu`) takes the same code path as the real run apart from
  data length, update count, polish iterations and compilation. Read the smoke branches and list every
  difference; code reached only in a real or GPU run goes into Part A as "by reading".
- That the coverage tool is installed in the `GraduationProject` env. If it is not, ask Dirk before installing.
- That smoke runs write nowhere the evaluation reads. `eval_common.py:90` reads grey-box checkpoints from
  `~/.deepSI/checkpoints`; check where the entry writes outputs and checkpoints before the first run and send them
  to the scratchpad if the config allows. If it does not, ask before running.

## 6. Tried and failed
None for this task. One related lesson: monkey patching or run-time wrapping of the framework was rejected by Dirk
earlier; this matters for step 2, not for the inventory.

## 7. Achieved
Agreed with Dirk on 2026-10-19 (this is the plan the inventory serves):
- Target folder: a new self-contained folder holding `gantry_interconnect_dynamic.py`, `gantry_dynamic/` and a
  trimmed `model_augmentation/`. Whatever is needed from `common/` moves into `gantry_dynamic/`. The `@added`,
  `__project_origin__` and `# CHANGED:` markers stay. Data stay outside the folder, as two named entries
  (lowpass, noisefree) below one data root.
- Equivalence of the clean code is to be shown later by a layered harness against golden outputs of the current
  code: forward pass, loss and gradients, a short seeded training, and the final checkpoints' metrics per split
  (train, val, test), on CPU in float64. Part C is the input for that harness.
- Steps after this one: Dirk compares this inventory with Codex's; Claude writes a clean-implementation design
  document; Codex and Claude critique the same document; Claude implements; Codex judges the code.

## 8. The open question
Nothing blocked. Where the thesis and the code disagree, do not decide which is right: report both in Part B.

## 9. Next action
Produce the coverage trace first, because it settles most of Part A objectively: one smoke run per spec row
under branch coverage, one at a time, then read the code it did not reach. Rows: all 9 of 221 to 230 with
lowpass, plus 221 and 224 with noisefree (the noise-free branch for each arm). Time the first run and report the
runtime before the other ten. Then write Parts A, B and C.

## 10. Acceptance criterion
- Part A accounts for every function and class in the four code locations of section 1: each is required (trace
  evidence or "by reading" with the reason), or not needed. No function is unclassified.
- Part B covers every numbered equation and every method paragraph of Sections II to V; each row cites the
  `file:line` you read yourself.
- Part C lists every RNG use in the four code locations with its position relative to the required path.
Report everything you find; Dirk filters.

Format example for Part A rows (illustrative names and lines):
`gantry_dynamic/training.py:410 | _orth_meter() | not needed | never called when orth=False (all spec rows) | trace`

## 11. Read these first
1. `scripts/gantry/gantry_interconnect_dynamic.py`, the THESIS block: how a row becomes a config.
2. `Thesis-writeup/Code/Server/greybox/run_thesis.sbatch` and `tables/runs.tsv` rows 221 to 230: the exact runs.
3. `tasks/final-run-code-changes-2026-10-16.md`: what the final method changed in the code.
4. `Thesis-writeup/Writing/sections/03_augmentation.tex` and `04_obc.tex`: the method Part B checks against.
5. `Thesis-writeup/Code/Evaluation/eval_common.py`: how the evaluation finds and loads a grey-box model.

## 12. Do not
- Do not edit any existing file, create the new folder, or write clean code.
- Do not run more than one numerical job at a time, and no run longer than a smoke run.
- Do not write into `Thesis-writeup/Results/`, `~/.deepSI/checkpoints` or any data folder.
- Do not read `kamtin-data/Data Telica/`.

## 13. Operational
- Runs follow the live-output convention of CLAUDE.md, with the row's `THESIS_*` variables plus `THESIS_SMOKE=1`
  and `THESIS_DEVICE=cpu`, under branch coverage with a separate data file per row.
- Keep the coverage data in `tasks/clean-code/coverage/` (one file per row plus a combined report), so the step 2
  critique can use it.
- Final message: the path of the inventory, the coverage runtime per row, and any question for Dirk.

## 14. Delegation
At most one Explore subagent, for Part B's thesis-to-code mapping if it is wide. No subagent for the coverage runs
or Part A.
