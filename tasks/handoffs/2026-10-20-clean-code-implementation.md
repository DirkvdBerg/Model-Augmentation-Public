# Handoff: implement the clean minimal code in `<repo>/Clean/` and prove it equivalent to mainline (step 4)
**From**: session of 2026-10-20 | **Branch**: Augmentation | **Effort suggested**: xhigh (about 4,000 lines of
copy-and-trim across 24 files plus a four-layer equivalence harness)

## 1. Task
Dirk's words: "make a new self contained folder only needing the required python files and to be written with
minimal code". Implement `<repo>/Clean/` exactly as `tasks/clean-code/design.md` (revised, changelog at its top)
specifies, as amended by section 4 below, and prove with the harness of design section 6 that it reproduces mainline.
In this order:
1. **Golden outputs first**, from mainline, before any clean code exists: layers 1, 2, 3, 3b and 3c of design 6.3,
   with the environment, budget, seeds and fingerprints of 6.2, into `<repo>/simulations/clean-golden/`.
2. **Write `Clean/`**: the 24 files of design section 1, row by row from design section 2.
3. **Static check**: `pyflakes` over `Clean/` with no undefined name, and an import of every module.
4. **Compare**: layers 1, 2, 3, 3b, 3c on CPU, bitwise as 6.3 states.
5. **Prepare the server parts** (Dirk submits; he does all git and server work): the layer-4 and 4b scripts against
   the staging folder of 6.5, and the compiled GPU smokes of 6.4 (a) and (b), each as an sbatch file plus the exact
   command list.
The deliverable is a `Clean/` folder that passes layers 1 to 3c locally, plus server scripts ready to run. Write a
short report `tasks/clean-code/implementation-report.md` (per layer: cases, result, runtime; every accepted deviation
with its cause).

## 2. Out of scope
- **The evaluation scripts in `Thesis-writeup/Code/Evaluation/`: not touched** (D-275 amendment 4). Design section 5
  edits (a) and (b) and every use of `THESIS_CODE_ROOT` are dropped; see section 4.
- A comment standard: comments get their own pass later (Dirk). Keep only the required markers (`@added`,
  `__project_origin__ = "added"`, `# CHANGED:`, including D4's `# CHANGED (clean): trimmed to the spec path`) and the
  `# THEORY:` / `# HEURISTIC:` labels; carry the mainline comments of kept lines or drop them, but write no new style.
- Mainline code (it is the oracle; the D-274 and D-275 amendment 3 fixes are in it), the thesis text, the black box.
- Submitting server jobs, any git operation (Dirk does both).
- Re-deciding anything in `design.md`, D-274 or D-275. If the code makes a design row impossible, take the nearest
  exact alternative and record it in the report.

## 3. Where things stand
Branch `Augmentation`, dirty tree; leave it. `Clean/` does not exist. Mainline carries D-274 (polish compared with
`bestfit`; PS2 S2 start kept as a candidate; K1 records only) and D-275 amendment 3 (Adam moments restored on a
resume from a deepSI dump, `scripts/gantry/gantry_dynamic/training.py:115`); smoke runs of rows 229 and 224 passed
before amendment 3, which has not run yet. No run in flight. Two leftover smoke folders under
`simulations/gantry_subnet/thesis_taf_lowpass_noise_data_lsq/` (`20261019_151607`, `20261019_153554`) are Dirk's to
delete; ignore them.

## 4. Amendments to `design.md` that override it
1. **No evaluation edits (D-275 amendment 4a).** Layer 4 and 4b get a harness-local loader in
   `Thesis-writeup/Code/Verification/clean-code/`: it builds the model from a run's `config.json`
   (`RunConfig.from_config_json` on the clean side, mainline's own construction on the mainline side), loads the
   component pair, and computes what design 6.3 rows 4 and 4b compare (per-record predictions from k0, RMS and NRMS_e
   per split, record and axis, `params_learned`; the overlap, negation, added-pole and frozen-Y calls). Run it once
   under each code root, in separate processes; on the mainline side also check it against the mainline evaluation
   (`eval_greybox`, `eval_overlap`, `eval_negation` called unmodified), so the loader itself is proven before it
   judges the clean code. The root assert of 6.2 (every loaded `gantry_dynamic*` and `model_augmentation*` module
   lies under the selected root) stays, without `THESIS_CODE_ROOT`.
2. **Adam moments on resume (D-275 amendment 3).** Design text saying a resume from a deepSI dump starts with fresh
   moments (section 3 "Reruns", layer 3c) is outdated: both codes now restore them
   (`d.get('optimizer') or d.get('_optimizer')`). Layer 3c compares the restored optimizer `state` entries too.
   Capture the 3c goldens only with this fix in place (it is in mainline now).
3. **No design-time name checker (D-275 amendment 4b).** Step 3 of section 1 uses `pyflakes` plus imports instead.
If `pyflakes` is not in the `GraduationProject` env, ask Dirk before installing it.

## 5. Established and verified
- Spec: rows 221 to 230 of `Thesis-writeup/Code/Server/tables/runs.tsv` (no 228), presets R1 to R7, R13, R14, each
  on `lowpass` or `noisefree`.
- Decisions: D-274 and D-275 with amendments 1 to 4 at the end of `docs/decisions.md`; Dirk's 2026-10-20 decisions
  (folder `<repo>/Clean/`, `data_root`/`out_root` arguments with defaults, CLI default R1 lowpass on cuda, resume and
  polish-only kept, reruns inside their own run folder, periodic `_last.pth`).
- The revised `design.md` passed a name check (no kept code reads a dropped name; 265 cited lines match mainline);
  its changelog lists where each merged item landed.
- Step-1 trace runtimes (`tasks/clean-code/inventory-claude.md` A.0): 218 to 447 s per u row, 826 to 1249 s per obc
  row at 0.3 to 0.6 s truncation and 4 updates.
- The trace-runner method (design 6.2): import mainline EP unmodified, set the row's `THESIS_*`, replace `EP.CFG`,
  call `EP.main()`, redirect deepSI checkpoints (LOCALAPPDATA) to a scratch folder.

## 6. Assumed but not verified
- Bitwise equality at layers 1 to 3c (design 6.3). Untested; a mismatch is localised as 6.3 prescribes, never
  toleranced, except the documented 1e-12 fallback of layers 1 and 2 for a named, accepted cause.
- Harness runtimes: layer 3 has 18 updates per run, more than the trace. Time the first case of every layer, report
  it, and only then launch the rest.
- The nine final checkpoints and the `[lbfgs]` lines of 221 and 222 are on the server (design 6.5); layer 4 waits for
  them, nothing else does.

## 7. Tried and failed
- Monkey patching or run-time wrapping of the framework: rejected by Dirk; `Clean/` is copied and trimmed code.
- Comparing full training runs bit for bit: not a criterion; the layered harness replaces it.

## 8. The open question
Nothing blocked locally. Server inputs (section 6) block layer 4 only.

## 9. Next action
Read `design.md` (changelog first, then sections 0, 6, 1, 3, 2), set up the harness runner, and capture the first
layer-1 golden (221-lowpass) with its runtime.

## 10. Acceptance criterion
- `pyflakes` reports no undefined name in `Clean/`; every module imports with only `Clean/` on the path.
- Layers 1, 2, 3, 3b and 3c pass bitwise on every case of design 6.3, each comparison run in processes whose loaded
  modules all lie under the selected root.
- `python Clean/gantry_interconnect_dynamic.py R6 lowpass --cpu --smoke` runs to the end on the PC.
- The layer-4/4b scripts and the 6.4 (a)/(b) sbatch files exist with their command list for Dirk.
- `implementation-report.md` lists, per layer, the cases, the result and the measured runtime, and every deviation
  from `design.md` with its cause.

## 11. Read these first
1. `tasks/clean-code/design.md` (revised; changelog at the top).
2. Section 4 of this handoff.
3. `docs/decisions.md` D-274 and D-275 with its amendments (end of file).
4. `tasks/clean-code/critique-merged.md`: the reasons behind the revised rows.
5. `scripts/gantry/gantry_interconnect_dynamic.py` THESIS block and `scripts/gantry/gantry_dynamic/training.py`.

## 12. Do not
- Do not edit mainline, the evaluation scripts, `design.md`, or files outside `Clean/`,
  `Thesis-writeup/Code/Verification/clean-code/`, `simulations/clean-golden/` and the report.
- Do not run more than one numerical job at a time; launch every run in the background with streamed output
  (CLAUDE.md convention); no multi-hour local runs.
- Do not write into `Thesis-writeup/Results/`, the mainline run folders, `~/.deepSI/checkpoints` or any data folder.
- Do not read `kamtin-data/Data Telica/`.

## 13. Operational
- Env `GraduationProject`; harness processes single-threaded (`torch.set_num_threads(1)`, `OMP_NUM_THREADS =
  MKL_NUM_THREADS = 1`), CPU, float64.
- Goldens: `<repo>/simulations/clean-golden/L<n>/<row>-<noise>/` with `meta.json` fingerprints (design 6.2).
- Final message: the report path, the layer results, and the server command list for Dirk.

## 14. Delegation
At most one general-purpose subagent, for writing the harness comparison scripts while you write `Clean/`; none for
the copy-and-trim itself, which needs one context. No Workflow.
