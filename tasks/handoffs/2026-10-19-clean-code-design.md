# Handoff: design document for the clean minimal implementation of the gantry augmentation pipeline (step 2 of 4)
**From**: session of 2026-10-19 | **Branch**: Augmentation | **Effort suggested**: xhigh (design across two packages,
an equivalence plan, and a mapping of every required function)

## 1. Task
Dirk's words: "make a new self contained folder only needing the required python files and to be written with
minimal code", for the `scripts/gantry/gantry_interconnect_dynamic.py` pipeline. Write the design document
`tasks/clean-code/design.md` that the implementation will follow. It is critiqued next by a Codex session and a
Claude session (step 3), then implemented by Claude and judged by Codex (step 4), so it must be complete enough that
an implementer needs no other source to know what to build, and concrete enough that a critic can disagree with a
specific line. The document contains:
1. **Folder layout**: the new self-contained folder, its name and location, and its tree. Inside it:
   `gantry_interconnect_dynamic.py`, `gantry_dynamic/` and a trimmed `model_augmentation/` (Dirk's choice).
2. **File-by-file mapping**: for every function, class and module constant classified as required in the step-1
   inventories, its destination (file and name in the new folder), and whether it is copied, merged or simplified;
   for every not-needed item, "dropped". Branches inside required functions that the spec never takes are removed,
   and the document names them.
3. **Config design**: one small config with only the axes the spec varies (arm, start, n_a, ps2, noise) as fields
   and the rest as constants, the 9 final rows as named presets, and how a run is launched (CLI or env) on the
   server and locally.
4. **Data**: two named entries, `lowpass` and `noisefree`, below one data root outside the folder; the split from
   MANIFEST.csv (18/6/11, K1 records only).
5. **Evaluation interface**: what the clean code keeps so that `Thesis-writeup/Code/Evaluation/` can load its
   checkpoints later (imports, `state_dict` names, `.npz` and `config.json` fields), per step-1 Part C.
6. **Equivalence plan**: the harness of section 7 made concrete: which golden outputs to capture from mainline
   (inputs, files, seeds, rows), the comparison per layer, and the tolerance per layer with its reason.
7. **Open choices**: every place where you chose between options, with the alternatives, for the critics.

## 2. Out of scope
- Writing or copying any code, creating the new folder, capturing golden outputs: step 4.
- Changing mainline. It already carries the agreed D-274 changes and is the oracle.
- Diagnostics (`diagnostics.py`, `obc_diagnostics.py`, `closed_loop_report.py` except the evaluation interface, the
  probe prints and meters): Dirk excluded them.
- The table and figure scripts (`Thesis-writeup/Code/Evaluation/make_results.py`, `Thesis-writeup/Code/Figures/`):
  left out for now (Dirk). Only their interface to the pipeline is designed (item 5).
- The black box R8 (`scripts/gantry/blackbox-cl-trainable/`), thesis text, server jobs, git writes.

## 3. Where things stand
Branch `Augmentation`, last commit 4271a31, dirty tree; leave it. Mainline edits of 2026-10-19 (D-274), not
committed: `scripts/gantry/gantry_dynamic/training.py` (polish comparison), `model_augmentation/fit_systems/
ps2_opening.py` (S2 start kept as a candidate), `scripts/gantry/gantry_dynamic/data.py` (`record_files` loads K1
records only). Two smoke run folders from that day can be ignored (Dirk deletes them):
`simulations/gantry_subnet/thesis_taf_lowpass_noise_data_lsq/20261019_151607` and `.../20261019_153554`.
No run in flight.

## 4. Established and verified
- **Spec**: rows 221 to 230 of `Thesis-writeup/Code/Server/tables/runs.tsv` (no 228), run by
  `Thesis-writeup/Code/Server/greybox/run_thesis.sbatch` through the entry's THESIS block. Varied: arm obc/u, start
  detuned/correct, n_a 0/2/8, ps2 0/1; fixed: seed 1, n_f 0.1 s, mlp, live init, encoder `data_lsq`. Plus Dirk's
  axis: every row on `lowpass` or `noisefree` data.
- **Step 1 (done)**: `tasks/clean-code/inventory-claude.md` (coverage trace of 11 runs, data in
  `tasks/clean-code/coverage/`) and `tasks/clean-code/inventory-codex.md` (static reading), compared in
  `tasks/clean-code/comparison.md`. They agree on 502 of 538 matched definitions; the three real disagreements are
  settled there (`load_traj` not needed; `OBCBasis.apply/jt/r_perp/frobenius/relative_change` diagnostic only;
  `_combos_from_raw` evaluation interface only). Use Claude's A.2 for the per-file decision and Codex's A2 for the
  branch-level call paths.
- **Decisions D-274 and its amendment** (`docs/decisions.md`, end of file), all in mainline and smoke-tested
  (rows 229 and 224, CPU): PS2 S2 keeps the start when no check beats it; only K1 records (E5 dropped); the OBC
  polish must beat the selected checkpoint's score (`bestfit`), otherwise the rebuilt model is kept. The S2 cost
  keeps `T.var(0)` (B-1); a code comment says it is the thesis quantity up to (B-1)/B.
- **Equivalence hazards agreed by both inventories** (step-1 Part C): the global torch stream draws in a fixed order
  after the seed at entry:409-410 (live g_aug MLP, then the W^b draw whose values least squares overwrites, then W^a
  Xavier, then the encoder net, then the PS2 decoder at the first update); with n_a = 0 the g_aug constructor still
  draws; `Dtype_DataLoader` shuffles with `default_rng(0)` independent of the run seed; float32 materialisation of
  nominal tensors (`gantry_ss.py` `_D=torch.float32`), `to_tensor` and `nn.Linear` defaults cast to float64 must be
  reproduced, not fixed; no diagnostic draws from a global stream before the final checkpoint is written.
- **External dependency**: the installed `deepSI` package (`System_data`, `SS_encoder_general`, `System_fittable`,
  `My_Simple_DataLoader`, `loop_dataset`). Keeping it or vendoring it is an open choice for the document; vendoring
  must reproduce the in-place shuffle exactly.
- **Markers**: the clean `model_augmentation/` keeps the `@added`, `__project_origin__` and `# CHANGED:` markers of
  CLAUDE.md (Dirk).

## 5. Assumed but not verified
- That `Thesis-writeup/Code/Evaluation/` imports the pipeline only through the interface listed in step-1 Part C
  (C6 in Claude's, C4 in Codex's). Settle by reading `eval_common.py` and the four `eval_*.py` imports.
- The polish outcome of runs 221 and 222 (their `[lbfgs]` lines are not in the local logs
  `Thesis-writeup/Results/18-10-2026/thesis-results-4`). It decides only which checkpoint is the layer-4 reference
  for those two rows; the document marks it as an input, not a blocker.
- The final checkpoints of 221 to 230 are on the server, not local. Layer 4 needs them; the document says where they
  come from.

## 6. Tried and failed
- Monkey patching or wrapping the framework at run time: rejected by Dirk; the clean code is copied and trimmed code.
- Comparing whole training runs bit for bit: not a criterion (float order and long training amplify rounding); the
  layered harness replaces it.

## 7. Achieved
Agreed equivalence harness (Dirk, 2026-10-19), golden outputs captured from mainline, CPU, float64:
1. Forward pass: the same `state_dict` in old and new model, the same batch: outputs match.
2. Loss and gradients: same batch and parameters: every loss term and every gradient match.
3. Short seeded training: about 20 Adam updates (plus a PS2 opening and an OBC refresh where the row has them):
   parameters match.
4. Final checkpoints: the metrics per split (train, val, test) of the final campaign checkpoints match.
The document sets the tolerances and the row set for each layer, covering both arms, n_a 0 and > 0, PS2 on and off,
and both noise modes.

## 8. The open question
Nothing blocked. Choices the document must make and expose in item 7: folder name and location; deepSI kept or
vendored; how far "minimal" goes inside Jan's classes (trimming unused methods of a required class versus keeping
the class intact for marker clarity); launch interface (presets via CLI or env).

## 9. Next action
Read the inputs of section 11, then write `tasks/clean-code/design.md` with items 1 to 7 of section 1.

## 10. Acceptance criterion
- Every definition classified required or interface in the step-1 inventories has exactly one destination in the
  mapping; every other definition is listed as dropped. No definition is missing.
- Every D-274 change appears in the mapping at its destination.
- The equivalence plan names, per layer: rows, captured files, comparison, tolerance with its reason.
- Each open choice lists its alternatives and the recommended one with the reason.
Report everything; Dirk and the critics filter.

## 11. Read these first
1. `tasks/clean-code/comparison.md`: agreed results of step 1 and Dirk's rulings.
2. `tasks/clean-code/inventory-claude.md` sections A.1, A.2 and Part C: per-file decisions and hazards.
3. `tasks/clean-code/inventory-codex.md` sections A2, A5 and Part C: call paths, dependencies, evaluation contract.
4. `docs/decisions.md` D-274 (end of file).
5. `scripts/gantry/gantry_interconnect_dynamic.py` THESIS block: how a row becomes a config today.

## 12. Do not
- Do not write code, create the folder, or run training; quick imports and reads only.
- Do not edit mainline, the inventories or `comparison.md`; the only file you create is `design.md`.
- Do not read `kamtin-data/Data Telica/`.

## 13. Operational
No runs needed. If a fact needs a check (an import, a field name), read the code; quick Python checks follow the
CLAUDE.md environment rule. Final message: the path of `design.md` and the list of open choices.

## 14. Delegation
None. The inputs are two documents and targeted code reads; one Explore subagent only if the evaluation-interface
check (section 5) turns out wide.
