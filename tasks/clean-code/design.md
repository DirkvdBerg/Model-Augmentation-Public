# Design: clean minimal implementation of the gantry augmentation pipeline (step 2 of 4)

2026-10-19. Handoff `tasks/handoffs/2026-10-19-clean-code-design.md`. Dirk's words: "make a new self contained
folder only needing the required python files and to be written with minimal code", for the
`scripts/gantry/gantry_interconnect_dynamic.py` pipeline. This document is what step 4 implements and what the step-3
critics (one Codex session, one Claude session) check. Sections 1 to 7 follow the handoff's items 1 to 7.

**Revision 2026-10-20 (end of step 3).** Handoff `tasks/handoffs/2026-10-20-clean-code-design-revision.md`: every
item of `critique-merged.md` (rulings D1 to D6 of D-275 with its two amendments, fixes B1 to B23) is carried into
the sections below. Changelog, one line per item (C = `critique-claude.md`, X = `critique-codex.md`):
- D1 `optimizer` property kept: interconnect rows `SSE_Interconnect_Composed` (`_optimizer`) and `optimizer` getter
  and setter; consequence in section 3 "Reruns". The code reads differently from D1's rationale: a mainline resume
  restores Adam moments only from a pair; from a deepSI dump `resolve_checkpoint` gets none (`d.get('optimizer')`,
  `GD/training.py:115`, while the property stores `_optimizer`). The clean code reproduces that, so every
  `--resume` (D6: dump sources) starts Adam with fresh moments; layer 3c is worded for it. Restoring them would be
  a one-term deviation in `resolve_checkpoint`, not made (Dirk's call).
- D2 adapter: `RunConfig.from_config_json` (section 3), section 5 item 2 and edit (b), open choice 8, layer 4.
- D3 own `_best.pth` only: section 5 item 3, `resolve_checkpoint` row, layer 3c sources, layer 4 (pairs only).
- D4 marker: rule 4.
- D5 rule G consistent: rules 2 and 3, open choices 13 and 14; each removed guard named in its row (start
  admissibility, block value guards, controller-bank shapes, `_check_val_records`, wiring and shape checks). A
  script sweep over every `raise` and `assert` in the kept definitions found 47 more that no row named: 20 re-check
  the clean code's own wiring and are now removed in their rows (block input-shape asserts, `Interconnect` wiring
  asserts, `_mats`, `window_controller_index`, `OBCBasis.coefficient`, the two OBC basis-None checks, the PS2
  alignment check, `live_g_feed_forward_nn`); 27 guard data, files, solver results or arguments from the
  evaluation and stay as copies.
- D6 run folder is one model: section 1, section 3 "Reruns" (`previous/<rerun_id>/`, `RESUMES`), section 5 item 3,
  layer 3c, 6.5 staging folder.
- D6 amendment 2 periodic `_last.pth`: `SSE_Interconnect.fit.validation` row (placement, why no training state
  changes), `LAST_SAVE` in config.json, `--from last`, layer 3 (with and without), layer 3c (iii).
- B1 `debugging`: removed with all eleven reads and `GD/model.py:79` (the smaller exact option): `Interconnect`
  rows, `build_model` row.
- B2 `Static_ANN_Block` default: `net` made required (`Static_ANN_Block.__init__` row, blocks Imports).
- B3 `e_net`: `SSE_Interconnect.__init__` row and `build_model` row (197-200).
- B4 `checkpoint_chunk`: rows `build_closed_loop` (355), `closed_loop_rollout`, `ClosedLoopSimulator.__init__`,
  `ClosedLoopSimulator.__call__` (598).
- B5 encoder call: `build_model` row (terms of 251-261, 136, headers 81, 96, 138, 246, 300) and both
  `linear_encoder_init_aug` rows (365, 443 and the rest); no Kaiming path remains to select.
- B6 constants: `use_f64`, `lbfgs` (and `joint_estimation`, read by the resume block) in section 3.
- B7 `RankReport`: trimmed with `from_matrix` (the smaller exact option), Gram diagonal and column scale kept exact.
- B8 `OBCLifecycle`: counters and timing kept as merged; the code adds one consequence: 496 writes the refresh time
  into the `entry` dict, so 458-459 stay too; `verbose`, `name_columns` removed at `obc_gantry.py:397-398`.
- B9 polish trim: `lbfgs_polish` and `run_lbfgs_polish` rows. The code adds one statement: 621
  (`frozen = set(spec.freeze)`) also reads a dropped field, so it goes and 626 loses its `n not in frozen and` term
  (exact: nothing is frozen); 626 and 630 otherwise kept.
- B10 `COMPILE_BACKEND`: `build_closed_loop` row (332), model constants.
- B11 dead imports: rule 1 ("imports follow the kept names") and an "Imports" bullet in every file block that loses
  an import (EP, data, model, obc_gantry, training, blocks, closed_loop, interconnect, lbfgs_polish, obc).
- B12 statement fixes: `hp['na_nb']` (`get_encoder_dims` merge), one `na_nb` (ClassVar; property dropped), the
  `int` form of `nf`, `y_decimation` and `_decimate_y` on `cfg.noise`, `float()` in the detuning (section 3), the
  progress line as a rewrite (rule 7, `fit` row), the config.json constants (section 3; job 89792's file is on
  the server only, so the list comes from the code that wrote it and `from_config_json` checks it against 89792 to
  89800 at layer 4), the line ranges of C section 3 (rows `_Trainable_Gantry_State_Block.__init__`,
  `GantryOBCCorrection.forward`, `build_reference_set`, `build_closed_loop`, `fit`, `linear_encoder_init_aug`,
  `_NfProbe.record` now a trim, `ClosedLoopSimulator.extra_array_names` listed).
- B13 `space`: rows `GantryOBCCorrection.__init__` (attribute kept), `make_basis_builder` (default `'affine'`,
  offset branch kept), section 5 items 1 and 5.
- B14 package root: section 1 "Imports" (guard in `Clean/gantry_dynamic/__init__.py`), section 5 edit (a), 6.2.
- B15 interface checks: layer 4b.
- B16 validation-score assertion: layer 4, tolerance column.
- B17 `reason`: `decide` kept as a trimmed function with mainline's text; layer 3 compares `reason`.
- B18 fingerprints: 6.2 "Golden outputs" (manifests, record files, config).
- B19 compiled GPU smoke: 6.4 (a), plus a compiled polish-only smoke 6.4 (b) for the restored compiled try.
- B20 folder `<repo>/Clean/`: section 1, `ROOT` and `REPO_ROOT = ROOT/..` (config constants), server command
  (section 3), evaluation edit (a), open choice 1.
- B21 roots as arguments: fields `data_root`, `out_root` (section 3), `traj_dir` and `save_dir` rows, section 4,
  recorded as `DATA_ROOT`, `OUT_ROOT`, never read back by `from_config_json`.
- B22 CLI: section 3 "Launch", open choice 4.
- B23 resume and polish-only: EP `main` row, rows `resolve_resume_hp`, `load_checkpoint`, `load_optimizer_moments`,
  `train` (now a trim of `train_model_with_diagnostics`), `run_lbfgs_polish` (788), `lbfgs_polish` (compiled try,
  eager fallback, second probe), `_eager`, `ClosedLoopSimulator.eager`; open choices 12 and 16 deleted; layer 3c.
- Acceptance check (handoff section 10): the name check of section 2 now finds no kept statement that reads a
  dropped or undefined name. It also found construction errors outside the merged list, fixed in their rows:
  `load_datasets` (the old removed range swallowed the kept split; `TRIM_MODES`, `cfg.snr`, the `DataBundle`
  terms), `_load_record` and `traj_dir` statement edits, the truth-field terms of `_load_record_float64` and
  `build_reference_set`, the `_TRAJ_*` terms of `checkpoint_save_system` and `checkpoint_load_system`, the class
  alias `rho = r_overlap` (a NameError at import), the diagnostics part of `_prediction_solve`, `pass_mats`, the
  fit timer lines with `rang = range(n_its)`, "Imports" bullets for data, model and interconnect, "Module-level
  statements" bullets (`config.py:811`, `data.py:72`, `gantry_ss.py:108-113`, the pre_encoder `__main__` block),
  and five ranges that ended inside a statement (`servo_sigma`, `GantryOBCCorrection.__init__`, `tangent_pair`,
  `window_controller_index`, `output_only`).

Inputs: `comparison.md` (agreed step-1 results and Dirk's rulings), `inventory-claude.md` (A.1, A.2, A.3, Part C),
`inventory-codex.md` (A2, A4, A5, Part C), D-274 and its amendment (`docs/decisions.md`), the entry's THESIS block,
`Thesis-writeup/Code/Evaluation/*.py`, and targeted reads of the mainline code.

**Oracle.** Mainline = the working tree of 2026-10-19 on branch `Augmentation` with the D-274 edits (since
committed in e588e63; the copied files are unchanged on 2026-10-20, the three D-274 hashes below re-checked). Line
numbers in this document are mainline lines of that tree. Source fingerprint of the
copied files: the SHA256 values of `inventory-codex.md` C7 hold for every copied file except the three D-274 files,
which are now `GD/data.py` 866e0152...7566, `GD/training.py` a50b18fb...2a30, `MA/fit_systems/ps2_opening.py`
a1e291cb...d85 (full hashes in the golden `meta.json`, section 6.2).

**Spec.** runs.tsv rows 221 to 230 (no 228), each on `lowpass` and on `noisefree` data. Varied: arm (obc, u), start
(detuned, correct), n_a (0, 2, 8), ps2 (0, 1), noise (lowpass, noisefree). Fixed: seed 1, n_f 0.1 s, tanh MLP 16x2,
live g_aug initialisation, encoder `data_lsq`, everything in `_THESIS_FIXED` (EP:342-352).

Abbreviations: EP = `gantry_interconnect_dynamic.py`, GD = `gantry_dynamic/`, MA = `model_augmentation/`,
CM = `scripts/gantry/common/`. H1 to H11 are the equivalence hazards of section 6.1.

## 0. Implementation rules

These rules decide every case the mapping (section 2) does not spell out.

1. **Copy first.** A kept definition is copied from mainline statement by statement. "copy" = no statement changes
   (comments may be shortened); "trim" = only the listed branches and statements are removed; "merge" = the
   statements move into the named function unchanged; "rewrite" = new code with the contract stated in the row.
   Nothing else changes: no renamed locals, no reordered arithmetic, no "cleaner" equivalent formula (H4 to H6).
   A trim may also remove part of a kept statement, each listed in its row with the line: **collapse** = an `if`
   header whose condition is constant on the spec path goes and its body stays, dedented (its `else` goes with it);
   **term** = a constant keyword argument, condition term or default is removed from a kept statement.
   **Imports follow the kept names:** an import line, or a name in it, goes when what it imports is dropped; the
   lines are listed in the file block's "Imports" bullet.
2. **Removed branches.** A branch is removed when the spec selects the other side through a constant (an option
   value, a fixed dimension, a feature flag that is always off). Branches whose side is decided by data or by the
   device (GPU moves, the compiled rollout, OBC rank loss, polish failure, NaN break, PS2 fallback) are kept even
   when the CPU trace never took them. One exception (D-275 D5): the per-call device and dtype self-heals
   (`Interconnect.forward` 137-141, `Interconnect.output_only` 330-333, the `out_gate` re-home in
   `Static_ANN_Block.forward` 76-77) are removed although the device decides them, because `Interconnect._apply`
   (IC:77-91) keeps the connection matrices on the module's device and dtype and `out_gate` is a registered buffer
   (`blocks.py:67`) that follows `.to()`; on a model built by `build_model` they never fire. The compiled GPU
   smoke (6.4) runs the device-moving path.
3. **Guards (rule G, applied consistently, D-275 D5).** Kept: guards on data and files (manifest, records, servo
   sigma, a duplicate record across splits), on solver and numerics results (rank, NaN, least-squares readiness,
   polish determinism), on the environment (CUDA availability, the eps deepSI's `init_optimizer` produced, the
   package origin of section 1), and on arguments that arrive from outside the clean code (the evaluation's
   `space`, the CLI combinations). Removed: re-checks of values the clean code builds itself (the start
   admissibility from constants and a fixed RNG, the validation-record identity, controller-bank shapes from
   `controller_ss`, block value guards, internal wiring and shape asserts, validation of removed options). Every
   removed guard is named in its row.
4. **Markers (Dirk, D-275 D4).** The clean `model_augmentation/` keeps the CLAUDE.md markers exactly where mainline
   has them: `__project_origin__ = "added"` at the top of every file that has it, `@added` on every kept class or
   function that has it, and every `# CHANGED: <reason>` comment that sits on a kept statement (a comment whose
   block is removed goes with it). Every trimmed Jan definition (a `trim` row in `model_augmentation/` whose class
   or function carries neither `@added` nor a `__project_origin__` file marker) gets one line
   `# CHANGED (clean): trimmed to the spec path` directly under its `def` or `class` line; the removed statements
   carry no marker of their own, this mapping is their record. A new statement in a Jan definition (the only one
   is the periodic `_last.pth` save in `SSE_Interconnect.fit.validation`) carries `# CHANGED (clean): <reason>`.
   `gantry_dynamic/` files keep their `__project_origin__` line.
5. **Comments.** Kept: every `# THEORY:` and `# HEURISTIC:` label on a numerical formula, constant or threshold
   (CLAUDE.md), and comments that state a non-obvious reason on a kept statement. Dropped: history narratives
   (job numbers, measured incidents of removed options), long docstrings of removed alternatives.
6. **No run-time patching.** Every behaviour is in the copied code; nothing wraps or patches a class at run time
   (rejected by Dirk). The harness (section 6) imports mainline and the clean code side by side in separate
   processes.
7. **Prints.** Kept: the run line, the data lines, the bank line ("one bank over N records, k distinct
   controllers", a sbatch sanity grep), "training rollout COMPILED", the encoder-start line, one progress line per
   validation (a rewrite: update count, square root of the mean training loss since the last print, latest
   validation score; the timing fields go with Tictoctimer, see the `SSE_Interconnect.fit` row), the resume lines
   of `load_checkpoint` and `resolve_resume_hp`, the PS2 summary, the polish outcome lines. Everything printed only
   by a dropped diagnostic goes.

## 1. Folder layout

**Location and name: `<repo>/Clean/`** (Dirk, 2026-10-20; open choice 1). A top-level folder of the repository,
beside `scripts/`, `model_augmentation/` and `Thesis-writeup/`. The data and output roots are arguments with
defaults (section 3), so no path depends on the folder's depth: `ROOT` = `Clean/` (the parent of the
`gantry_dynamic` package) and `REPO_ROOT = ROOT/..`, which feeds `git_provenance` and the two root defaults.

```
Clean/
  gantry_interconnect_dynamic.py   entry: cli() (section 3) and main(cfg, run_id=None, rerun=None)
  gantry_dynamic/
    __init__.py                    __project_origin__ and the import-origin guard (below); no sys.path insertion
    config.py                      RunConfig (5 axes + run environment + roots), PRESETS, from_config_json,
                                   constants, config.json, save_dir
    data.py                        DATASETS (lowpass, noisefree), MANIFEST split, record loading, normalisation
    controller.py                  K1 controller bank, closed-loop simulator construction, torch.compile
    model.py                       build_model, least-squares encoder start
    obc_gantry.py                  OBC reference set, correction module, basis builder, attach_obc
    training.py                    train (resume, Adam, PS2, selection, polish, checkpoints), ValHistory,
                                   checkpoint loading, resolve_resume_hp
    closed_loop_report.py          _without_learned, _true_free only (evaluation interface, open choice 10)
  model_augmentation/
    __init__.py                    empty (mainline eagerly imports benchmark modules; nothing needs that)
    fit_systems/
      __init__.py                  empty
      interconnect.py              Interconnect, SSE_Interconnect (Jan, trimmed), SSE_Interconnect_Composed, Dtype_DataLoader
      blocks.py                    Block, Static_ANN_Block, Linear_Output_Block, Gantry/Trainable/Reduced state blocks
      closed_loop.py               ControllerBank, closed-loop rollout, ClosedLoopSimulator, batch free-run scorer
      lbfgs_polish.py              PolishSpec, FixedBatch, lbfgs_polish
      obc.py                       OBCBasis, OBCLifecycle, build_stacked_sensitivity
      pre_encoder.py               linear_encoder_init_aug
      ps2_opening.py               PS2Spec, PS2Opening
    systems/
      __init__.py                  empty
      gantry_ss.py                 nominal float32 constants, P, Cd, Dd, LFR builders
    utils/
      __init__.py                  empty
      utils.py                     added, to_tensor, detect_algebraic_loop, expansion_matrix, selection_matrix
      torch_nets.py                live_g_feed_forward_nn
      deepSI_corrections.py        fixed_System_data, fixed_System_data_norm
```

Twenty-four files, every one of them a copy-and-trim of the mainline file with the same package path (1:1, open
choice 3), so a reviewer diffs each against its mainline original. Size: the four mainline locations hold 15,750 lines in 42
files; the kept definitions span 5,945 of them, about 4,900 of which are code (blank, comment and docstring lines
excluded, rough tokenizer count). The branch removals of section 2 come on top, so the folder should land between
3,500 and 4,500 lines (estimate from these counts; the implementer reports the measured number).

**Outside the folder, by design:**
- Data: field `data_root`, default `<repo>/Thesis-writeup/Data` (`--data-root`, section 4). Read only.
- Run output: field `out_root`, default `<repo>/simulations/gantry_clean/` (`--out-root`), run folder
  `out_root/{mode}_data_lsq/<run_id>/`, with
  `mode` = `thesis_taf_lowpass_noise` or `thesis_taf_noisefree` (the mainline mode strings, so a family folder has the
  name the evaluation already expects) and `run_id` = `SLURM_JOB_ID`, else a timestamp. A separate root keeps clean
  runs out of the mainline family folders, where `discover_runs` would raise on two runs of one design (open
  choice 15). A run folder holds one model: a continuation or polish-only rerun writes into it under the same
  run id (D-275 D6, section 3 "Reruns").
- deepSI's own checkpoints `<Class>_<run_id>_best.pth` and `_last.pth` stay where deepSI writes them
  (`~/.deepSI/checkpoints`, or `%LOCALAPPDATA%/deepSI/checkpoints` on Windows); fit() needs them (H8), and the
  evaluation's `checkpoint_status` looks for an unfinished run's `_best.pth` there (`eval_common.py:90, 120`).
  The clean code also writes `_last.pth` at every validation (D-275 amendment 2, `SSE_Interconnect.fit.validation`
  row). `--resume <run_dir>` finds them by the run folder's `run_id`.
- The installed `deepSI` package, pinned at 0.3.29, with torch 2.5.1, numpy 2.0.1, scipy 1.17.0 (open choice 2).
- The harness (section 6) in `Thesis-writeup/Code/Verification/clean-code/`, with `Clean/` as the import root of
  its clean-side processes, and its golden outputs in `<repo>/simulations/clean-golden/`.

**Imports.** Running `python Clean/gantry_interconnect_dynamic.py` puts `Clean/` on `sys.path[0]`, so
`gantry_dynamic` and `model_augmentation` resolve inside the folder with the same package and module names as
mainline. One process must never import both. Mainline `gantry_dynamic` lives in `scripts/gantry/`, so a launch
that puts the repository root first on the path (`python -m`, a notebook at the repo root, `PYTHONPATH=Clean`)
would import the clean `gantry_dynamic` with the MAINLINE `model_augmentation` and run without an error (C23).
`Clean/gantry_dynamic/__init__.py` therefore holds one environment guard (rule 3): it imports `model_augmentation`
and raises `ImportError` unless `os.path.dirname(os.path.dirname(model_augmentation.__file__))` equals the parent
folder of `gantry_dynamic`. The evaluation selects its package root once (section 5, edit (a)).

## 2. File-by-file mapping

Every definition of the four code locations (647, the full syntax ledger of `inventory-codex.md` A4, nested
functions included) and every module or class constant (154) appears below exactly once. 262 definitions are kept,
merged or rewritten; 385 are dropped. The table was generated from a decision list and checked by script: every
definition labelled required or interface in either inventory (Codex "required", "required interface"; Claude
"run (pipeline)") has an explicit row or an explicit drop reason, and no row names a definition that does not exist.
Unlisted dropped definitions carry the reason of their inventory label ("diagnostic (Dirk excluded)" or "not on the
spec path").

**Name check (2026-10-20, revision).** An AST script over this section (session scratchpad, not in the repository)
reads every kept statement (copy, trim and merge rows, minus the lines and terms each row removes, plus the module
level) and resolves every name it reads: module-level names and imports, `self.<attr>`, `cfg.<name>`, `hp[...]`,
keyword arguments of calls to kept callables, dataclass constructor fields, defaults and other-object attributes.
Result on this document: 0 names that are dropped or never defined, 0 for manual triage, 0 removals whose reading
line is unlisted, all 265 row lines equal to the mainline definition lines, and 0 missing names in the evaluation's
imports. Its limits: rewrite rows and new code are not checked; two rows sharing a name (the `optimizer` getter and
setter) are matched to the first definition; no flow analysis. A removed module-level statement is written in a
"Module-level statements" bullet of its file block.

Columns: source definition (mainline line) | destination in the new folder (bare name = same file as the heading
arrow) | action (section 0, rule 1) | removed branches with mainline lines, kept items that a trim could be expected
to remove, interface notes.

**D-274 at its destinations:** change 1 (S2 keeps the start as a candidate) `MA/fit_systems/ps2_opening.py
PS2Opening._fit_map` lines 233-235, plus the agreed comment on `T.var(0)` in `_fit_map.ee`; change 2 (K1 records
only) `GD/data.py record_files` lines 89-90; change 3 as amended (the polish must beat `bestfit`; the rebuilt model is
kept on rejection) `GD/training.py run_lbfgs_polish` lines 784-796 and the final-event rule in `GD/training.py train`.

**New definitions (no mainline source):** `GD/config.py` `PRESETS`, `ROOT`, `DATA_ROOT`, `OUT_ROOT`, the
RunConfig fields `start_phase`, `data_root`, `out_root`, the axis properties (`obc`, `nx_ann`, `ann_route_ix`,
`mode`, `combo_init_detune`), the ClassVar constants and `RunConfig.from_config_json` (section 3); `GD/data.py`
`DATASETS`; `GD/training.py` `ValHistory`; EP `cli()`; the import-origin guard in `GD/__init__.py` (section 1).
**Renamed:** `train_model_with_diagnostics` -> `train` (its one call site is EP `main`).

#### `EP`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `main` (382) | `EP main(cfg, run_id=None, rerun=None)` | rewrite | the call order of EP:383-453, resume included: run folder (fresh: `os.path.join(save_dir(cfg), run_id)`, `run_id` = argument, else `SLURM_JOB_ID`, else a timestamp, EP:384; rerun: steps 1 to 4 of section 3 "Reruns", which also give `run_id` and `resume_ckpt`; fresh: `resume_ckpt = None`); `cfg = replace(cfg, start_phase='lbfgs')` for a polish-only rerun; seed (np, torch, EP:390-391); load_datasets; compute_normalization; `hp = resolve_resume_hp(cfg, cfg.hp, resume_ckpt)` (EP:400); config.json (fresh run, EP:403-406, keys of section 3); seed again (EP:409-410); build_model; `fit_sys.simulator = build_closed_loop(...)` (EP:437-440); `if resume_ckpt is None: prefit_physical_encoder(fit_sys, cfg, data, norm)` (EP:444-445, term `cfg.encoder_init == 'data_lsq' and` removed, constant); `train(fit_sys, hp, cfg, data, norm, resume_ckpt=resume_ckpt, checkpoint_dir=sdir, run_id=run_id)`; the "Training complete" print (EP:454). Removed: print_run_banner (397), the `RESUME_CHECKPOINT` read (399, replaced by `rerun`), save_flag tests (403, 450: always save), the `if cfg.closed_loop:` header (432, collapsed) and the open-loop branch (446-447), post-run evaluate/diagnostics (456-471) |
| (new) | `EP cli()` | new | section 3 "Launch": argparse, `PRESETS`, the `--smoke` budget, `from_config_json` for a run folder; calls `main` |

- Dropped (THESIS_* env parsing replaced by the CLI (section 3)): `_thesis_env`.
- Module and class constants: `CFG`: replaced -> GD/config.py PRESETS + RunConfig; `_dataset_ab_mode`, `_dataset_ab_seed`, `_lbfgs_ab_freeze`, `_obc_arm`, `_thesis_arm`: drop; `_LBFGS_TOL_GRAD`, `_LBFGS_TOL_CHANGE`: move -> GD/config.py RunConfig.lbfgs_tol_grad, lbfgs_tol_change (ClassVar).
- Imports (EP:11-29): rewritten with `main` and `cli`; the `gantry_dynamic.diagnostics` and `gantry_dynamic.evaluation` imports (EP:25-28) go, `train_model_with_diagnostics` becomes `train`.

#### `GD/closed_loop_report.py` -> `GD/closed_loop_report.py`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `_without_learned` (38) | `_without_learned` | copy | evaluation interface; module imports reduced to Static_ANN_Block, copy, torch |
| `_true_free` (67) | `_true_free` | copy | evaluation interface |

- Dropped (post-run report): `_score`, `closed_loop_table`.
- Module and class constants: `__project_origin__`: copy (marker); `ROWS`: drop.

#### `GD/config.py` -> `GD/config.py`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `y_decimation` (46) | `y_decimation` | trim | term: the test `cfg.mode in ANTI_ALIAS_MODES` (48) becomes `cfg.noise == 'lowpass'` (C29; `ANTI_ALIAS_MODES` is dropped); both return strings copied |
| `record_trim` (53) | `record_trim` | trim | term: ` if cfg.mode in TRIM_MODES else 0` (56) removed, `return int(round(AA_TRIM_S * cfg.fs_new_hz))` = 20 (both noise entries trim); evaluation imports it |
| `RunConfig` (60) | `RunConfig` | rewrite | five axis fields + run-environment fields (with `start_phase`) + the two roots; fixed settings as ClassVar constants under the mainline names (section 3) |
| `RunConfig.__post_init__` (370) | `RunConfig.__post_init__` | rewrite | validates the axis and environment values only (allowed sets, `start_phase` as config.py:501-505, the compile/device rule of 480-481); the checks of removed options go |
| (new) | `RunConfig.from_config_json` | new | D-275 D2, section 3: clean and mainline `config.json`, rejection of any other setting |
| `RunConfig.fs_new_hz` (542) | `RunConfig.fs_new_hz` | copy |  |
| `RunConfig.d` (546) | `RunConfig.d` | copy |  |
| `RunConfig.ts_new` (550) | `RunConfig.ts_new` | copy |  |
| `RunConfig.dtype_np` (554) | `RunConfig.dtype_np` | copy |  |
| `RunConfig.dtype_pt` (558) | `RunConfig.dtype_pt` | copy |  |
| `RunConfig.nf` (562) | `RunConfig.nf` | trim | nf_override branch (571-572) removed; `return max(1, int(self.nf_seconds / self.ts_new))` (573) copied = 400 (C28) |
| `RunConfig.hp` (583) | `RunConfig.hp` | copy | hp keeps key names and order (NX_ANN, n_nodes_per_layer, n_hidden_layers, up_sample, nf, na_nb, batch_size, lr, epochs): the .npz hp contract; `self.na_nb` reads the ClassVar |
| `save_dir` (603) | `save_dir` | trim | term: `REPO_ROOT, 'simulations', 'gantry_subnet'` (605) becomes `cfg.out_root`: `os.path.join(cfg.out_root, f'{cfg.mode}_{cfg.encoder_init}')` (section 1) |
| `git_provenance` (610) | `git_provenance` | copy | provenance only; cwd = `REPO_ROOT` (= `ROOT/..`, the repository holding `Clean/`) |
| `git_provenance._git` (628) | `git_provenance._git` | copy |  |
| `config_json_dict` (668) | `config_json_dict` | rewrite | mainline keys in mainline order with the spec values, then ARM, START, N_A, NOISE, PS2_SMOKE, then GIT (section 5) |

- Dropped (never called): `default_hp`.
- Dropped (one definition of `na_nb`, the ClassVar `na_nb = 30` of section 3; the dimension rule and the override go, C27): `RunConfig.na_nb`.
- Dropped (config_json_dict writes every field explicitly; the import-time guard served evolving fields): `_assert_json_coverage`.
- Module-level statements: the import-time call `_assert_json_coverage()` (811) removed with the function.
- Module and class constants: `__project_origin__`: copy (marker); `_PKG_DIR`, `REPO_ROOT`: replaced -> GD/config.py `ROOT` (= `Clean/`, `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`), `REPO_ROOT = os.path.dirname(ROOT)` (git cwd), `DATA_ROOT = os.path.join(REPO_ROOT, 'Thesis-writeup', 'Data')`, `OUT_ROOT = os.path.join(REPO_ROOT, 'simulations', 'gantry_clean')` (field defaults); `ANTI_ALIAS_MODES`: drop (`y_decimation` and `_decimate_y` test `cfg.noise`); `AA_ORDER`, `AA_FC_HZ`, `AA_TRIM_S`: copy (HEURISTIC/THEORY labels kept) -> GD/config.py; `TRIM_MODES`: drop (both noise entries trim); `_JSON_EXEMPT`: drop.

#### `GD/controller.py` -> `GD/controller.py`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `M_op` (72) | `M_op` | copy |  |
| `sys_stage_frf` (81) | `sys_stage_frf` | copy |  |
| `cnorm_coeffs` (92) | `cnorm_coeffs` | copy |  |
| `cnorm_at` (99) | `cnorm_at` | copy |  |
| `build_cfb_at` (104) | `build_cfb_at` | copy |  |
| `_tf_to_ss_batch` (124) | `_tf_to_ss_batch` | copy |  |
| `controller_ss` (143) | `controller_ss` | copy |  |
| `_stem` (181) | `_stem` | copy |  |
| `register_stored_controller` (198) | `register_stored_controller` | copy | the unsupported-controller branch (205-207) stays: the evaluation skips a non-K1 record through y_op_for raising KeyError |
| `y_op_for` (210) | `y_op_for` | trim | legacy RECORD_Y_OP branch (218-219) removed; stored and unsupported branches stay (evaluation interface) |
| `gain_for` (232) | `gain_for` | copy |  |
| `build_controller_bank` (246) | `build_controller_bank` | copy | evaluation imports it |
| `build_closed_loop` (269) | `build_closed_loop` | trim | removed: val-count guard (304-308, D5: `val_data` and `val_files` come from the same loader), the `if cfg.loss_weighting == 'servo':` header (350, collapsed) and the e_train-is-None guard (351-352, always servo weights), physical_D print (316-317), `from .model import COMPILE_BACKEND` (332: the constant now lives in this module, C14), term `checkpoint_chunk=cfg.checkpoint_chunk` in the `ClosedLoopSimulator(...)` call (355, C8); kept: duplicate-name guard (292-296, a record listed twice in the manifests is a data error, rule 3), compile block (329-341, GPU), keyword-only signature incl. verbose, e_train, e_val (evaluation calls it) |

- Module and class constants: `__project_origin__`: copy (marker); `mb`, `mh`, `m1`, `m2`, `Jb`, `Jh`, `cg1`, `cg2`, `cy`, `cb1`, `cb2`, `kb1`, `kb2`, `Lb`, `d`, `FBW`, `W`, `TS`, `P`, `C_DAMP`, `K_STIFF`: copy (Python doubles, independent of gantry_ss float32) -> GD/controller.py; `RECORD_Y_OP`: drop (legacy modes); `_STORED_DESIGN_Y`, `_STORED_GAIN`, `_UNSUPPORTED`: copy (registry filled by _register_controller) -> GD/controller.py.

#### `GD/data.py` -> `GD/data.py`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `record_files` (79) | `record_files` | trim | legacy branch (81-82) removed; D-274 K1 filter (89-90) copied; manifest order; 18/6/11 |
| `_register_controller` (94) | `_register_controller` | copy |  |
| `DataBundle` (104) | `DataBundle` | trim | fields kept: train_list, val_list, test_list, train_data, val_ckpt_data, train_files, val_files, test_files, e_train, e_val, e_test; dropped: val_data, test_data, val_x_logical, val_x_aug, test_x_logical, test_x_aug, sigma_n |
| `Norm` (129) | `Norm` | trim | fields kept: x_mean, std_x, std_u, u_mean, ystd, y0, Cd_norm, Dd_np; dropped: u_all, y_all, x_all (linear_map only) |
| `traj_dir` (143) | `traj_dir` | trim | the `if cfg.mode in THESIS_DATASETS:` header (144, collapsed); (145) becomes `return os.path.join(cfg.data_root, DATASETS[cfg.noise])`; legacy branch (146) removed; evaluation imports it |
| `_load_u` (149) | `_load_u` | copy | evaluation imports it |
| `_resample_u` (154) | `_resample_u` | trim | D == 1 branch (168-169) removed; float64 block mean kept |
| `_decimate_y` (174) | `_decimate_y` | trim | term: `cfg.mode not in ANTI_ALIAS_MODES` (177) becomes `cfg.noise != 'lowpass'` (C29); 20 -> 4 kHz guard (179-181) removed (constants); both rules kept |
| `_record_at_fs_new` (187) | `_record_at_fs_new` | copy | truncate_s branch kept for the smoke and the harness |
| `_load_record` (205) | `_load_record` | trim | the `if is_thesis(cfg):` header (211, collapsed) and `has_r` (213) removed; (214) becomes `u, y, *r = _record_at_fs_new(d, cfg, ('r_sim',))` and (215) becomes `e = np.asarray(y, np.float64) - np.asarray(r[0], np.float64)` (every thesis record holds r_sim); folder= kept (evaluation twins) |
| `servo_sigma` (224) | `servo_sigma` | trim | None guard (228-230) removed; finite-positive guard kept (data) |
| `load_datasets` (260) | `load_datasets` | trim | removed: the `if cfg.mode in TRIM_MODES:` header (268, collapsed: both noise entries trim); `SNR = cfg.snr` and the SNR guard (271-275); the noise branch and `sigma_n` (286-295); val_data and test_data (299-300); the primary val/test truth loads (303-313); in the `DataBundle(...)` call the terms `val_data=val_data, test_data=test_data,` (322) and 323-325 (`val_x_logical`, `val_x_aug`, `test_x_logical`, `test_x_aug`, `sigma_n`). Kept: the data-dir and decimation prints (267, 269-270), `record_files` and `_split` (276-283), the two `System_data_list` (297-298), the split-count prints (315-317), the `DataBundle` call with its kept fields |
| `load_datasets._split` (278) | `load_datasets._split` | copy |  |
| `compute_normalization` (331) | `compute_normalization` | trim | u_all/y_all/x_all no longer returned; arithmetic unchanged incl. inv(P.numpy().T) in float32 then cast (hazard H6) |

- Dropped (every mode is a thesis mode): `is_thesis`.
- Dropped (not used (comparison.md)): `load_traj`.
- Dropped (diagnostic ground truth (comparison.md)): `load_mat_aug`.
- Module and class constants: `__project_origin__`: copy (marker); `TRAIN_FILES`, `VAL_FILES`, `TEST_FILES`: drop (legacy lists); `THESIS_DATASETS`: replaced -> GD/data.py DATASETS = {lowpass, noisefree} below `cfg.data_root`; `_THESIS_SPLIT`: copy -> GD/data.py _THESIS_SPLIT.
- Module-level statements: the import-time assert `set(TRIM_MODES) == set(THESIS_DATASETS)` (72) removed (both names are gone).
- Imports: `REPO_ROOT`, `ANTI_ALIAS_MODES`, `TRIM_MODES` from the `.config` import (21-22).

#### `GD/diagnostics.py`

- Dropped (diagnostic (Dirk excluded)): `r2_per_channel`, `best_affine_r2`, `encoder_state_estimates`, `state_recovery_diagnostic`, `compute_gradient_norms`, `encoder_init_state`, `zeroed_ann_validation`.
- Dropped (printing only (both inventories)): `print_run_banner`.
- Module and class constants: `__project_origin__`: drop (file dropped).

#### `GD/model.py` -> `GD/model.py`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `get_encoder_dims` (39) | (merged) | merge | into GD/model.py build_model, replacing the call (77): `na = nb = hp['na_nb']` (50, copied: under a resume `hp` comes from `resolve_resume_hp`, which takes `na_nb` from the checkpoint, `GD/training.py:1055`, C26) and `na_right = nb_right = 0` (51), the data_lsq branch (47-51) |
| `build_model` (60) | `build_model` | trim | removed: term `debugging=False` in `Interconnect(...)` (79, C5); the `if cfg.joint_estimation:` header (81, collapsed) with its else branch (124-129); the `if cfg.physics_parameterization == 'reduced':` header (96, collapsed) with its else branch (104-109); static_pass_buffers (113-115); the start admissibility check (116-123, D5: the start is built from constants and a fixed RNG); `_act = torch.nn.Tanh` replaces the `cfg.ann_activation` test (136, C9); the `if cfg.g_init == 'live':` header (138, collapsed), `_ann_net = live_g_feed_forward_nn` replaces the resnet choice (139), its else branch (141-142); ANN_REZERO_GATE (157-164); term `e_net_kwargs={...}` in `SSE_Interconnect_Composed(...)` (197-200, C7); the linear_map encoder branch (210-245) and the `elif cfg.encoder_init == 'data_lsq':` header (246, collapsed); in the encoder call (251-261) the terms `A=None, B=None, C=None, D=None` (252), `flag_linear_only=False` (257), `wa_encoder_init=cfg.wa_encoder_init` (259) and `wb_init='random'` (260): the trimmed encoder keeps only the `wb_init='random'` and `'xavier_uniform_gain1'` statements, so dropping the keywords cannot select Kaiming (C9, X5; H1 draws unchanged); the `if cfg.encoder_init in (...)` header (300, collapsed) with its else (302-304); orth (312-314). Term: `_names = Reduced_Gantry_State_Block.PARAM_NAMES` (85, the same class attribute of `_Trainable_Gantry_State_Block`; `Parameterized_Gantry_State_Block` is dropped). Kept in order (hazard H1): reduced block (detuned combos), Linear_Output_Block, Static_ANN_Block with live_g_feed_forward_nn, connections, SSE_Interconnect_Composed, norm fields, data_lsq encoder, cuda-availability guard (286-292, environment), init_model, eps guard (295-299, deepSI's optimizer), hfn.to(float64), attach_obc for arm=obc |
| `prefit_physical_encoder` (328) | `prefit_physical_encoder` | copy | cuda moves kept |
| `_prefit` (352) | `_prefit` | trim | simulator-is-None guard (355-356) removed |
| `_prefit.recorder` (389) | `_prefit.recorder` | copy |  |
| `least_squares_start` (415) | `least_squares_start` | copy | every guard kept (data-dependent readiness, D-271) |
| `train_model` (467) | (merged) | merge | into GD/training.py train, replacing the call (training.py:892-895): eps guard (471-476, deepSI's optimizer, rule 3), cuda move (483-484), `fit_sys.fit(...)` (485-510) with the same arguments (`epochs=epochs or hp['epochs']`, 487; `its_per_val=(cfg.n_its if cfg.n_its else (cfg.its_per_val or 'epoch'))`, 501), where the call's `epochs`, `nf`, `validation_measure` are `epochs_remaining`, `hp['nf']`, `cfg.validation_measure` as at training.py:892-895; cpu move (518-519); `bestfit = fit_sys.bestfit` (520) |

- Module and class constants: `__project_origin__`: copy (marker); `COMPILE_BACKEND`: move -> GD/controller.py COMPILE_BACKEND (where torch.compile is called; the import at controller.py:332 goes, C14); `PHY_IX`, `_START_MAX_UNEXPLAINED`: copy -> GD/model.py.
- Imports: `import os` (4, only ANN_REZERO_GATE read it) and `import deepSI` (8, only the linear_map branch); in 11 `paper_resnet` and `zero_init_feed_forward_nn` go, `live_g_feed_forward_nn` stays; 17 (`gantry_linearize_and_discretize`) and 18 (`normalize_linear_ss_matrices`) go (C, X2).

#### `GD/obc_gantry.py` -> `GD/obc_gantry.py`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `GantryReferenceSet` (49) | `GantryReferenceSet` | trim | fields kept: Z_step, Z_field, record_ix, sample_ix, record_names, x_phys_next_rec; dropped: x_phys, u_stage, x_phys_next, x_abs, n_per_record, ts (diagnostic or unused) |
| `GantryReferenceSet.n` (68) | `GantryReferenceSet.n` | copy |  |
| `fourth_order_central_difference` (79) | `fourth_order_central_difference` | copy |  |
| `_load_record_float64` (91) | `_load_record_float64` | trim | returns (u, y) only, the truth fields are not read: terms, 101-102 become `u, y = (np.asarray(a, dtype=np.float64) for a in _record_at_fs_new(d, cfg))` and 103 `return u, y` |
| `build_reference_set` (106) | `build_reference_set` | trim | removed: terms, the default `= TRAIN_FILES` of `files` (106, required) and the `stride: int = 1` parameter (107); snr guard (114-118); stride option (136-137); the truth arrays: 130 becomes `u, y = _load_record_float64(f, cfg)`, terms `xn, xa,` and their two `[]` in 127, 142 and 145 removed; in the `GantryReferenceSet(...)` call the terms `x_phys=torch.from_numpy(x_phys), u_stage=torch.from_numpy(u_stage),` (158), `x_phys_next=torch.from_numpy(x_next), x_abs=torch.from_numpy(x_abs),` (159, the whole line), `n_per_record=counts,` (160) and `+ ([] if stride == 1 else ['(stride %d: probe only)' % stride]), ts=ts,` (161); in the reference print the terms `stride %d, ` and `stride,` (165-166); the Y-coverage print (164, 168-170, it reads the dropped `memory_bytes`). Signature (cfg, norm, nx_ann, files, verbose) kept for the evaluation |
| `GantryOBCCorrection` (175) | `GantryOBCCorrection` | trim | affine space only |
| `GantryOBCCorrection.__init__` (189) | `GantryOBCCorrection.__init__` | trim | removed: the `use_jvp` parameter and attribute (190, 200), the space guard (193-194, D5: its one caller passes `cfg.obc_space`), `_static` (202), the empty-route guard (209-211, D5); term: the `'tangent'` default of `space` (191; `attach_obc` always passes it). Kept: `self.space = space` (195): the evaluation reads `corr.space` (`eval_overlap.py:51`, C2, X11); buffers E, vbar, theta_frozen unchanged (state_dict names; 218-219 still size `theta_frozen` by `self.space`) |
| `GantryOBCCorrection.expansion_point` (225) | `GantryOBCCorrection.expansion_point` | copy |  |
| `GantryOBCCorrection.set_frozen` (230) | `GantryOBCCorrection.set_frozen` | copy |  |
| `GantryOBCCorrection.tangent_pair` (250) | `GantryOBCCorrection.tangent_pair` | trim | _static branches (261-263, 267-268) and tangent-space branch (264-265) removed; 259 copied |
| `GantryOBCCorrection.forward` (286) | `GantryOBCCorrection.forward` | trim | use_jvp branch (297-299) removed; the `if self.space == 'tangent':` branch (301-302) removed and its else (303-304) collapsed to `mats, dmats = obc_pass[:2]`; 307 copied |
| `make_float64_block` (312) | `make_float64_block` | copy | evaluation imports it |
| `make_basis_builder` (319) | `make_basis_builder` | trim | term: the default of `space` (321) becomes `'affine'` (C2, X11); signature `(blk64, ref, row_ix, chunk, device=None, space='affine')` kept for the evaluation, which passes `space=corr.space` |
| `make_basis_builder.build` (323) | `make_basis_builder.build` | copy | the `if space == 'affine':` offset column (332-338) kept, so a `'tangent'` caller still gets the mainline tangent basis `[J]` (exact, no guard needed); device move (327-328) kept |
| `make_reference_field` (344) | `make_reference_field` | copy |  |
| `make_reference_field.field` (350) | `make_reference_field.field` | copy |  |
| `attach_obc` (362) | `attach_obc` | trim | removed: the `ref` and `ref_stride` parameters (362-363) and the `if ref is None:` header (376, collapsed); term: `files = data.train_files` replaces the `TRAIN_FILES` fallback (378); term `stride=ref_stride` (380); the block-type guard (372-374, D5: `build_model` always adds both blocks); static_pass_buffers (385-389); terms `verbose=verbose` (397, C12: `OBCLifecycle` has no `verbose`) and `name_columns=...` (398) in the `OBCLifecycle(...)` call. Kept: `space=cfg.obc_space` (383, 394), expected_rank 11, float64 block, lifecycle, `fit_sys.obc_reference`, the attach print (403-407) |

- Dropped (print only): `GantryReferenceSet.memory_bytes`.
- Dropped (jvp reference path): `GantryOBCCorrection.step`, `GantryOBCCorrection.correction_from_theta`.
- Module and class constants: `__project_origin__`: copy (marker); `COMBO_NAMES`: drop (name_columns print only).
- Imports: `StaticPassBuffers` and `directional_correction` from the `obc` import (38-39); `TRAIN_FILES` from the `.data` import (42).

#### `GD/training.py` -> `GD/training.py`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `_json_safe` (26) | `_json_safe` | copy |  |
| `save_checkpoint_weights` (39) | `save_checkpoint_weights` | copy | .pt keys hfn, encoder, optimizer (omitted after an accepted polish) |
| `resolve_checkpoint` (67) | `resolve_checkpoint` | copy | both formats; the pair is read by both codes, a deepSI dump only by the code that wrote it (D3, section 5). `_opt = d.get('optimizer')` (115) copied: with the kept property a dump yields no Adam state (section 3 "Reruns", changelog D1) |
| `load_optimizer_moments` (124) | `load_optimizer_moments` | copy | D-173 moment restore, restored for decision 4 (B23); reached from a pair on an Adam start |
| `_assert_same_architecture` (158) | `_assert_same_architecture` | copy | evaluation interface |
| `_assert_same_dtype` (248) | `_assert_same_dtype` | copy | evaluation interface |
| `load_checkpoint` (277) | `load_checkpoint` | copy | Adam-moment restore (292-297) and the fresh-optimizer NOTE (299-315) kept (decision 4, C18); signature (fit_sys, base_path, joint_estimation, start_phase) used by `train` and the evaluation |
| `_NfProbe` (319) | `ValHistory` | rewrite | the D-267 history only (open choice 9): val_hist, hist_path, record, hist_arrays, __call__ |
| `_NfProbe.__init__` (362) | `ValHistory.__init__` | rewrite | `__init__(self, fit_sys, hist_path)`: stores `fit_sys`, `hist_path`, `val_hist = []`; no window batch, no `WindowErrorStats` (mainline sets `fit_sys.loss_stats` here, 375; the clean system has no `loss_stats`), no combo meters (no global RNG either way, hazard H2) |
| `_NfProbe.record` (593) | `ValHistory.record` | trim | open-loop early return (600-601) removed (the simulator is always attached) |
| `_NfProbe.hist_arrays` (610) | `ValHistory.hist_arrays` | copy | val_hist.npz keys unchanged |
| `_NfProbe.__call__` (627) | `ValHistory.__call__` | trim | only `self.record('adam', fit_sys, value)` (629); train/val nf statistics and prints (630-668) removed |
| `_install_nf_val_probe` (671) | (merged) | merge | into GD/training.py train, replacing the call (881-883): `prev = fit_sys.validation_probes` (675), `probe = ValHistory(fit_sys, os.path.join(checkpoint_dir, 'val_hist.npz'))` (rewrite of 676-677), `fit_sys.validation_probes = tuple(prev) + (probe,)` (678), `_prev_probes = prev` (the return, 700); the header prints (679-699) removed |
| `run_lbfgs_polish` (753) | `run_lbfgs_polish` | trim | removed: lbfgs-off return (764-765); terms `freeze=tuple(cfg.lbfgs_freeze), meter_rel_tol=cfg.lbfgs_meter_rel_tol,` (773), `time_budget_s=cfg.lbfgs_time_budget_s, chunk=cfg.lbfgs_chunk,` (774) and `meters_gate=cfg.lbfgs_meters_gate` (775) in the `PolishSpec(...)` call (C13; `seed=cfg.seed` on 774 stays); term `meters_fn=_gantry_meters,` (814). Kept: PolishSpec from the config (769-772), cuda moves and empty_cache (776-782, 820-822), the D-274 rule (784-796) with `val_before = (bestfit if (np.isfinite(bestfit) and cfg.start_phase != 'lbfgs') else None)` (788, copied: a polish-only start re-measures, C18), OBC rebuilt score measured and recorded as "rebuilt", used only if `val_before` is None; polish_candidate record (815-818) |
| `train_model_with_diagnostics` (825) | `train` | trim | renamed (call site: EP `main`), signature `(fit_sys, hp, cfg, data, norm, resume_ckpt=None, checkpoint_dir=None, run_id=None)` unchanged. Kept as copies, in order (decision 4, C18, X8): 835-837; `unique_code = run_id` (847-848); `bestfit = inf` (849); the resume block (852-858) with `load_checkpoint(fit_sys, resume_ckpt, cfg.joint_estimation, start_phase=cfg.start_phase)`; `epochs_remaining` and its prints (860-864); `fit_sys.bestfit = inf` (866); `t0`; the PS2 gate `cfg.ps2 and done_epochs == 0 and cfg.start_phase == 'adam'` (871) with the attach (872-879); the history install (the merged `_install_nf_val_probe`, replacing 881-883) and `history` (884); the try (885) with the `start_phase == 'lbfgs'` Adam skip (886-890) and, else, the merged `train_model` (replacing 892-895); finally restore probes and `training_phase` (896-898); PS2 summary print (899-900); `elapsed` (903); the epoch rule (907-908); the pre-polish pair (932-938, `cfg.lbfgs` is a ClassVar); selection (941-944); polish (945); final event (948-952); bestfit (956-962); the `_polished`, `_no_adam_ran`, `_drop_optimizer` rule (975-977); weights (965, 978-984); eval and print (986-987); the `.npz` with the real `start_phase`, `resumed_from`, `resumed_at_epoch` (989-1011). Removed: the nf statistics (901-902, 910-926); term: `if os.environ.get('THESIS_SMOKE') == '1':` (873) becomes `if cfg.ps2_smoke:`; the return (1013-1017) becomes `return bestfit` (`main` prints it; the dict fed the dropped post-run report) |
| `resolve_resume_hp` (1020) | `resolve_resume_hp` | trim | restored for decision 4 (C18, X8): checkpoint owns the five structural keys (`NX_ANN, n_nodes_per_layer, n_hidden_layers, na_nb, up_sample`), this run the rest, with its diff print; removed: the `start_phase == 'lbfgs'` without checkpoint guard (1035-1038, D5: `main` sets `'lbfgs'` only for `--polish-only`, which always has a checkpoint) |

- Dropped (never called, also not by the resume path): `infer_arch_from_hfn`, `config_for_checkpoint`.
- Dropped (nf/joint/OBC probe printing (Dirk excluded)): `_NfProbe._val_stats`, `_NfProbe._fmt`, `_NfProbe._joint_probe`, `_NfProbe._obc_probe`.
- Dropped (polish meters, no gate (meters_gate=False)): `_gantry_meters`.
- Module and class constants: `__project_origin__`: copy (marker); `_NfProbe.N_VAL_WINDOWS`: drop.
- Imports: `from .model import train_model` (20, merged) and `from model_augmentation.fit_systems.interconnect import WindowErrorStats` (21, dropped) (C, X2).

#### `MA/fit_systems/blocks.py` -> `MA/fit_systems/blocks.py`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `Block` (12) | `Block` | trim | Jan |
| `Block.__init__` (15) | `Block.__init__` | copy |  |
| `Block.init_block` (24) | `Block.init_block` | copy |  |
| `Static_ANN_Block` (31) | `Static_ANN_Block` | copy | Jan; evaluation imports it (type identity, out_gate) |
| `Static_ANN_Block.__init__` (32) | `Static_ANN_Block.__init__` | trim | term: the default `= zero_init_feed_forward_nn` of `net` (36) removed, so `net` is required (`build_model` passes `net=live_g_feed_forward_nn`; the default would need the dropped class at import, C6, X2); net constructed here: first global draws (hazard H1) |
| `Static_ANN_Block.forward` (69) | `Static_ANN_Block.forward` | trim | out_gate re-home (76-77) removed (non-persistent buffer follows .to(), rule 2 exception); the input-shape assert (71) removed (D5: `z` is built by the interconnect or the reference field) |
| `Linear_Output_Block` (120) | `Linear_Output_Block` | copy | @added |
| `Linear_Output_Block.__init__` (121) | `Linear_Output_Block.__init__` | trim | ny == 0 branch (130-131) removed; to_tensor float32 materialisation kept (hazard H6) |
| `Linear_Output_Block.forward` (138) | `Linear_Output_Block.forward` | trim | the input-shape assert (139) removed (D5) |
| `Discrete_Nonlinear_Function_Block` (251) | `Discrete_Nonlinear_Function_Block` | trim | Jan |
| `Discrete_Nonlinear_Function_Block.__init__` (252) | `Discrete_Nonlinear_Function_Block.__init__` | copy |  |
| `Discrete_Nonlinear_Function_Block.forward` (260) | `Discrete_Nonlinear_Function_Block.forward` | trim | the input-shape assert (264) removed (D5) |
| `Gantry_State_Block` (664) | `Gantry_State_Block` | trim | @added |
| `Gantry_State_Block.__init__` (699) | `Gantry_State_Block.__init__` | trim | Y_op-at-construction branch (757-763) removed (Y_op = None; the evaluation sets Y_op, N_op, d_op afterwards); nominal float32 buffers unchanged (hazard H6) |
| `Gantry_State_Block.nonlinear_function` (779) | `Gantry_State_Block.nonlinear_function` | trim | the input-shape assert (781) removed (D5) |
| `Gantry_State_Block._rk4` (791) | `Gantry_State_Block._rk4` | copy |  |
| `Gantry_State_Block._deriv_with` (814) | `Gantry_State_Block._deriv_with` | trim | terms capture (871-874) removed; frozen-Y branch (843-845) kept: evaluation frozen_y |
| `Gantry_State_Block._deriv_both_with` (902) | `Gantry_State_Block._deriv_both_with` | trim | terms/sterms capture (968-972) removed; frozen-Y guard (909-910, raises NotImplementedError) kept |
| `Gantry_State_Block._rk4_with_tangent` (979) | `Gantry_State_Block._rk4_with_tangent` | copy |  |
| `_Trainable_Gantry_State_Block` (996) | `_Trainable_Gantry_State_Block` | trim | @added; static_pass class attribute removed; PARAM_NAMES kept |
| `_Trainable_Gantry_State_Block.__init__` (1008) | `_Trainable_Gantry_State_Block.__init__` | trim | params_init None branch (1024-1026) removed; positivity guard (1030-1031) removed (D5: `params_init` is the nominal `gantry_ss` vector, C33); buffers unchanged (state_dict names) |
| `_Trainable_Gantry_State_Block._build_KC` (1042) | `_Trainable_Gantry_State_Block._build_KC` | copy |  |
| `_Trainable_Gantry_State_Block._build_M1M2` (1064) | `_Trainable_Gantry_State_Block._build_M1M2` | copy |  |
| `_Trainable_Gantry_State_Block._mats_from_raw` (1079) | `_Trainable_Gantry_State_Block._mats_from_raw` | copy |  |
| `_Trainable_Gantry_State_Block.nonlinear_function` (1102) | `_Trainable_Gantry_State_Block.nonlinear_function` | copy |  |
| `_Trainable_Gantry_State_Block.pass_mats` (1112) | `_Trainable_Gantry_State_Block.pass_mats` | trim | term: the static_pass branch ` if self.static_pass is None else self.static_pass(mats)` (1120) removed, so `return mats` |
| `_Trainable_Gantry_State_Block._mats` (1122) | `_Trainable_Gantry_State_Block._mats` | trim | the `_cur is None` check (1123-1126) removed (D5: a call-order check inside the class; `return self._cur`, 1127, kept) |
| `_Trainable_Gantry_State_Block.__getstate__` (1147) | `_Trainable_Gantry_State_Block.__getstate__` | copy |  |
| `_Trainable_Gantry_State_Block.__setstate__` (1152) | `_Trainable_Gantry_State_Block.__setstate__` | copy |  |
| `Reduced_Gantry_State_Block` (1250) | `Reduced_Gantry_State_Block` | trim | @added; COMBO_NAMES and index constants kept |
| `Reduced_Gantry_State_Block.__init__` (1304) | `Reduced_Gantry_State_Block.__init__` | trim | combo_init None branch (1308-1316) removed; value guards (1318-1323) and the m_diff domain guard (1338-1340) removed (D5: the start combinations come from the nominal vector and the fixed detuning draw, C33); Lambda_combo and RMSE buffers kept (state_dict names) |
| `Reduced_Gantry_State_Block.combos_of` (1350) | `Reduced_Gantry_State_Block.combos_of` | copy | evaluation interface |
| `Reduced_Gantry_State_Block.gauge_section` (1365) | `Reduced_Gantry_State_Block.gauge_section` | copy |  |
| `Reduced_Gantry_State_Block._m_diff_bound` (1385) | `Reduced_Gantry_State_Block._m_diff_bound` | copy |  |
| `Reduced_Gantry_State_Block.combinations_from_free` (1397) | `Reduced_Gantry_State_Block.combinations_from_free` | copy | evaluation interface |
| `Reduced_Gantry_State_Block.free_from_combinations` (1409) | `Reduced_Gantry_State_Block.free_from_combinations` | copy | evaluation interface (_true_free) |
| `Reduced_Gantry_State_Block.recover_combinations` (1424) | `Reduced_Gantry_State_Block.recover_combinations` | copy |  |
| `Reduced_Gantry_State_Block.transition_from_free` (1429) | `Reduced_Gantry_State_Block.transition_from_free` | copy | evaluation interface |
| `Reduced_Gantry_State_Block.mats_and_tangent_from_free` (1461) | `Reduced_Gantry_State_Block.mats_and_tangent_from_free` | copy |  |
| `Reduced_Gantry_State_Block.mats_and_tangent_from_free._mats` (1463) | `Reduced_Gantry_State_Block.mats_and_tangent_from_free._mats` | copy |  |
| `Reduced_Gantry_State_Block._recover_params` (1482) | `Reduced_Gantry_State_Block._recover_params` | copy |  |
| `Reduced_Gantry_State_Block.estimation_parameters` (1551) | `Reduced_Gantry_State_Block.estimation_parameters` | copy | evaluation interface |

- Dropped (not on the spec path): `Block.forward`, `Cubic_Block`, `Cubic_Block.__init__`, `Cubic_Block.forward`, `Linear_State_Block`, `Linear_State_Block.__init__`, `Linear_State_Block.forward`, `Parameterized_Linear_State_Block`, `Parameterized_Linear_State_Block.__init__`, `Parameterized_Linear_State_Block.forward`, `Parameterized_Linear_State_Block.param_loss`, `Parameterized_Linear_Output_Block`, `Parameterized_Linear_Output_Block.__init__`, `Parameterized_Linear_Output_Block.forward`, `Parameterized_Linear_Output_Block.param_loss`, `Cascaded_Tanks_State_Block`, `Cascaded_Tanks_State_Block.__init__`, `Cascaded_Tanks_State_Block.nonlinear_function`, `Parameterized_Cascaded_Tanks_State_Block`, `Parameterized_Cascaded_Tanks_State_Block.__init__`, `Parameterized_Cascaded_Tanks_State_Block.nonlinear_function`, `Parameterized_MSD_State_Block`, `Parameterized_MSD_State_Block.__init__`, `Parameterized_MSD_State_Block.nonlinear_function`, `Nonlinear_MSD_State_Block`, `Nonlinear_MSD_State_Block.__init__`, `Nonlinear_MSD_State_Block.nonlinear_function`, `Nonlinear_MSD_State_Block.deriv`.
- Dropped (not used): `Discrete_Nonlinear_Function_Block.init_norm`, `Gantry_State_Block.deriv`, `Gantry_State_Block._deriv_tangent_with`, `Reduced_Gantry_State_Block.transition_and_tangent_from_free`.
- Dropped (abstract; overridden): `Discrete_Nonlinear_Function_Block.nonlinear_function`.
- Dropped (overridden by _Trainable_Gantry_State_Block._mats): `Gantry_State_Block._mats`.
- Dropped (reporting and meters; the evaluation uses estimation_parameters only (comparison.md: _combos_from_raw)): `_Trainable_Gantry_State_Block.physical_params`, `_Trainable_Gantry_State_Block._combos_from_raw`, `_Trainable_Gantry_State_Block.identifiable_combinations`, `_Trainable_Gantry_State_Block._initial_identifiable_combinations`, `_Trainable_Gantry_State_Block._combination_training_label`, `_Trainable_Gantry_State_Block.param_table`.
- Dropped (raw coordinates; never instantiated (PARAM_NAMES read from the reduced block)): `Parameterized_Gantry_State_Block`.
- Dropped (raw coordinates): `Parameterized_Gantry_State_Block.__init__`, `Parameterized_Gantry_State_Block._recover_params`, `Parameterized_Gantry_State_Block.param_loss`, `Parameterized_Gantry_State_Block.estimation_parameters`.
- Dropped (returns the float 0.0 (param_prior=False); the loss sweep that called it is removed): `Reduced_Gantry_State_Block.param_loss`.
- Dropped (meters and the excluded run report): `Reduced_Gantry_State_Block.identifiable_combinations`, `Reduced_Gantry_State_Block._initial_identifiable_combinations`, `Reduced_Gantry_State_Block._combination_training_label`, `Reduced_Gantry_State_Block.physical_params`.
- Dropped (its callers are the start check in `build_model`, removed by D5, and the excluded report `GD/evaluation.py:67-74`; no evaluation script reads it): `Reduced_Gantry_State_Block.admissibility`.
- Imports: `zero_init_feed_forward_nn` (8, the default it served is removed); `System_data` (5, annotation of dropped blocks only, 255, 275, 362).
- Module and class constants: `_Trainable_Gantry_State_Block.PARAM_NAMES`, `Reduced_Gantry_State_Block.COMBO_NAMES`, `Reduced_Gantry_State_Block.M_TOTAL_IX`, `Reduced_Gantry_State_Block.M_DIFF_IX`, `Reduced_Gantry_State_Block.J_EFF_IX`, `Reduced_Gantry_State_Block.M_DIFF_RHO`, `Reduced_Gantry_State_Block.COORDINATE_VERSION`: copy -> MA/fit_systems/blocks.py; `_Trainable_Gantry_State_Block._IDENT_ORDER`: drop (param_table).

#### `MA/fit_systems/closed_loop.py` -> `MA/fit_systems/closed_loop.py`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `ControllerBank` (76) | `ControllerBank` | trim |  |
| `ControllerBank.__init__` (122) | `ControllerBank.__init__` | trim | shape guards (132-133, 135-137) removed (D5: the matrices come from `controller_ss`, and `ystd`, `std_u` from the clean normalisation, C33); 134 kept |
| `ControllerBank.n_controllers` (150) | `ControllerBank.n_controllers` | copy | the bank print in build_closed_loop (sbatch sanity line) |
| `ControllerBank.gather` (158) | `ControllerBank.gather` | copy |  |
| `ControllerBank.zero_state` (162) | `ControllerBank.zero_state` | copy |  |
| `ControllerBank.step` (167) | `ControllerBank.step` | copy |  |
| `_rollout_segment` (216) | `_rollout_segment` | copy | hazard H4 (output, controller, corrected step) |
| `closed_loop_rollout` (245) | `closed_loop_rollout` | trim | the `chunk=0` parameter (245) and the checkpointed branch (295-311, checkpoint_chunk=0) removed; the `if not chunk or chunk >= nf or not torch.is_grad_enabled():` header (291, collapsed: its return, 292-293, is the only path); bank re-home (269-270) kept (GPU) |
| `window_controller_index` (314) | `window_controller_index` | trim | the record-count guard (340-345) removed (D5: the rows and the records come from the same loader through `build_closed_loop`) |
| `WindowControllerIndex` (354) | `WindowControllerIndex` | copy |  |
| `WindowControllerIndex.__init__` (370) | `WindowControllerIndex.__init__` | copy |  |
| `WindowControllerIndex.build` (377) | `WindowControllerIndex.build` | trim | removed: the `record_weights is None` branch (417-418, the servo weights are always given, C34) and the weight-count guard (420-422, D5: both lists come from the same split); kept: the window-count guard (406-409) and `verify_by_content` (415), which check deepSI's windowing (rule 3) |
| `WindowControllerIndex.verify_by_content` (427) | `WindowControllerIndex.verify_by_content` | copy |  |
| `ClosedLoopSimulator` (445) | `ClosedLoopSimulator` | trim |  |
| `ClosedLoopSimulator.__init__` (482) | `ClosedLoopSimulator.__init__` | trim | removed: the `checkpoint_chunk=0` parameter (482) and `self.checkpoint_chunk` (516, C8), the timing history (510-511), `self._val_checked` (522, with `_check_val_records`), `self.last_window_counts` (523); the `if train_weights is not None:` header (488, collapsed, C34: `build_closed_loop` always passes the servo weights) |
| `ClosedLoopSimulator.__getstate__` (535) | `ClosedLoopSimulator.__getstate__` | copy | drops the compiled pair: the polish after fit is eager (hazard H8) |
| `ClosedLoopSimulator.eager` (541) | `ClosedLoopSimulator.eager` | copy | restored for decision 4 (C15, X8): `lbfgs_polish._eager` switches to the eager pair on the polish-only start, where `fit()` never ran and the compiled pair is still attached (`GD/controller.py:329-341`) |
| `ClosedLoopSimulator.__call__` (559) | `ClosedLoopSimulator.__call__` | trim | ctrl_ix-None guard (563-567) and the update timer (595, 605) removed; term `chunk=self.checkpoint_chunk,` (598, C8); compiled branch with cudagraph_mark_step_begin and clones (584-604) kept (GPU) |
| `ClosedLoopSimulator.augment_training_data` (640) | `ClosedLoopSimulator.augment_training_data` | trim | `self.last_window_counts = ...` (643) removed |
| `ClosedLoopSimulator.validation_error` (667) | `ClosedLoopSimulator.validation_error` | trim | sim-NRMSe only: measure guards (684-692), the record-identity check (693-694, D5) and the sim-RMS return (716) removed, the `if validation_measure == 'sim-NRMSe':` header (714) collapsed; last_per_channel and last_sim_rms kept (ValHistory) |
| `closed_loop_free_run_rms_batch` (719) | `closed_loop_free_run_rms_batch` | copy | evaluation interface incl. return_pred; grouping by length kept (hazard H5) |

- Dropped (print and check_units only): `ControllerBank.physical_D`.
- Dropped (not used): `ControllerBank.check_units`.
- Dropped (print only): `ClosedLoopSimulator._note_update_time`.
- Dropped (D5: re-checks that the validation list passed to `validation_error` is the one the clean code registered from the same loader, C33): `ClosedLoopSimulator._check_val_records`.
- Dropped (one-record wrapper, not used): `closed_loop_free_run_rms`.
- Module and class constants: `__project_origin__`: copy (marker); `ClosedLoopSimulator.extra_array_names` (480): drop (the instance attribute of `__init__`, 489, is always set after the collapse of 488; `fit` and `FixedBatch` read the instance).
- Imports: `import torch.utils.checkpoint` (67, the checkpointed branch).

#### `MA/fit_systems/interconnect.py` -> `MA/fit_systems/interconnect.py`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `Interconnect` (21) | `Interconnect` | trim | Jan |
| `Interconnect.__init__` (22) | `Interconnect.__init__` | trim | removed: the `debugging` parameter (22) and `self.debugging = debugging` (39), together with all eleven reads listed in the rows below and the `debugging=False` argument at `GD/model.py:79` (C5, X4; the smaller exact option: every read is a print); save_signals, first_step_eval |
| `Interconnect._apply` (77) | `Interconnect._apply` | copy | keeps connection matrices on the module device and dtype (makes the per-call heals redundant) |
| `Interconnect._module_device_dtype` (88) | `Interconnect._module_device_dtype` | copy |  |
| `Interconnect.init_model` (93) | `Interconnect.init_model` | copy |  |
| `Interconnect.forward` (100) | `Interconnect.forward` | trim | removed: 3-D input branches (112-113, 119-120), per-call device heal (137-141, rule 2 exception), save_signals (172-182), the output-shape assert (189, D5); kept: the batch assert (122: `eval_frf` calls `hfn(x, u)` with its own inputs, rule 3), lazy init_forward (125-127, the evaluation calls hfn before any rollout), summation order, OBC subtraction (hazard H4) |
| `Interconnect.init_forward` (199) | `Interconnect.init_forward` | trim | debugging prints (224, 248, 255, 265) removed |
| `Interconnect._signal_cone` (280) | `Interconnect._signal_cone` | copy |  |
| `Interconnect.output_only` (303) | `Interconnect.output_only` | trim | per-call device heal (330-334) removed (rule 2 exception) |
| `Interconnect.init_connection_matrices` (354) | `Interconnect.init_connection_matrices` | trim | removed: debugging prints (355, 375-378, 383, 432, 439-440); add_to connections (362, 372-373, 428-429) and the add_to-only guard (385-386) that reads them; the wiring checks (408; 434-437) and, in the additive branch, the empty-matrix fill with its warning and the size asserts (414-419: every additive connection of `build_model` carries its matrix), D5: the connections come from `build_model` |
| `Interconnect.add_block` (444) | `Interconnect.add_block` | trim | the duplicate-block assert (445, D5) and the debugging print (455) removed |
| `Interconnect.connect_signals` (457) | `Interconnect.connect_signals` | trim | the method assert (465, D5) and the debugging print (470-472) removed |
| `Interconnect.connect_block_signals` (474) | `Interconnect.connect_block_signals` | copy | small helper, copied whole |
| `Interconnect.determine_signal_ix` (483) | `Interconnect.determine_signal_ix` | trim | the fall-through raise (491) removed (D5: `build_model` passes only blocks and names) |
| `Interconnect.convert_signal_name_to_ix` (493) | `Interconnect.convert_signal_name_to_ix` | trim | the assert (500) and the fall-through raise (503) removed (D5) |
| `Interconnect.parse_connection_function_method` (525) | `Interconnect.parse_connection_function_method` | copy | small helper, copied whole |
| `Signal_Connection` (535) | `Signal_Connection` | trim |  |
| `Signal_Connection.__init__` (537) | `Signal_Connection.__init__` | copy |  |
| `Dtype_DataLoader` (568) | `Dtype_DataLoader` | copy | default_rng(0) and the in-place shuffle of the deepSI parent (hazard H3) |
| `Dtype_DataLoader.__init__` (594) | `Dtype_DataLoader.__init__` | copy |  |
| `Dtype_DataLoader.__init__._conv` (595) | `Dtype_DataLoader.__init__._conv` | copy |  |
| `SSE_Interconnect` (604) | `SSE_Interconnect` | trim | Jan; class attributes simulator, validation_probes, training_phase kept |
| `SSE_Interconnect.__init__` (628) | `SSE_Interconnect.__init__` | trim | removed: the `e_net=modified_encoder_net, e_net_kwargs={}` parameters (629) and their assignments (633-634): the encoder is always injected, `modified_encoder_net` is dropped, and `build_model` stops passing `e_net_kwargs` (`GD/model.py:197-200`, C7, X2); `self.multi_loss_val` (642) |
| `SSE_Interconnect.init_model` (651) | `SSE_Interconnect.init_model` | trim | sys_data-None branch (653-655), auto_fit_norm branch (658-659), init_nets call removed; to_device, optimizer, scheduler, history arrays, hfn.init_model kept in order |
| `SSE_Interconnect.simulate` (678) | `SSE_Interconnect.simulate` | trim | open-loop rollout (693-705) removed: always the simulator |
| `SSE_Interconnect.make_training_data` (746) | `SSE_Interconnect.make_training_data` | trim | no-simulator branch removed: the `if self.simulator is not None and hasattr(...)` header (749, collapsed) and the fallback return (751) |
| `SSE_Interconnect.cal_validation_error` (753) | `SSE_Interconnect.cal_validation_error` | trim | deepSI fallback (764-765) removed; probes loop kept (ValHistory) |
| `SSE_Interconnect.fit` (808) | `SSE_Interconnect.fit` | trim | removed: `_flip_for_validation` (879); the init_model_done=False branch (915-923) and its else header (924, collapsed); scheduler warning (929-930); Dataset/DataLoader branch (960-963); concurrent_val (973-976, 1062-1066, 1109-1118); timeout (985, 989-990, 1098-1101); list_val_measures; print_full_time_profile; the extra-array name guards (1010-1020, D5: `extra_array_names` and `make_training_data` are the clean simulator's); the scheduler step (1049-1052, `init_scheduler({})` returns None); Tictoctimer and tqdm (983, 986-987, 994-995, 1000, 1038, 1048, 1060, 1071, 1096); term: 985 becomes `rang = range(n_its)` (the timeout alternative goes). Kept in order: _check_and_refresh_optimizer_if_needed, dt, cuda move, train(), counters, make_training_data(norm.transform(...)), N_batch_updates_per_epoch = N // batch_size, Dtype_DataLoader, initial validation, loop over loop_dataset with batch .cuda(), closure, optimizer.step(closure), NaN break (1041-1043), training_phase.after_step, counters (epoch_counter += 1/N), validation every its_per_val, end: train(); cpu(); _last save (1122); history keep; _best load; refresh_basis sync and _best re-save (1136-1149). The progress line is a rewrite (C31, rule 7): 1075-1079 (new-lowest line) and 1085-1088 kept; the concurrent `valfeqstr` (1080-1084), the timer and batch-rate strings with `self.batch_feq` (1089-1092) and the time profile (1094-1095) removed; 1093 becomes `print(Loss_str)` (update count, square root of the mean training loss since the last print, latest validation score) |
| `SSE_Interconnect.fit.validation` (881) | `SSE_Interconnect.fit.validation` | trim | cpu flip (883-884, 909-910) and list_val_measures with `multi_loss_val` (891-896) removed; tie rule bestfit >= Loss_val kept. New statement (D-275 amendment 2) right after the improving-checkpoint block (902-908), before `self.train()` (911): `self.checkpoint_save_system(name='_last')` with `# CHANGED (clean): D-275, _last.pth at every validation so a stopped run can continue`. It changes no training state: the composed `checkpoint_save_system` (IC:1505-1512) pops `obc`, `obc_reference`, `training_phase` and puts the same objects back in `finally`; deepSI's save is `torch.save(self.__dict__, file)` (`fit_system.py:493-496`), which only reads; every `__getstate__` it meets returns a copy (`ClosedLoopSimulator` CL:535-538, the composed class IC:1570-1597, the physics block `blocks.py:1147-1150`); it draws no random number, syncs no basis (no `_sync_prediction_state_for_validation` call) and does not touch the optimizer. Mainline already calls the same function at every improving validation (908). The end-of-fit `_last` save (1122) still writes the final state. Layer 3 checks it (6.3) |
| `SSE_Interconnect.fit.closure` (1023) | `SSE_Interconnect.fit.closure` | trim | timer calls (1024-1025, 1027, 1029, 1031-1032, 1034-1035) removed; loss, zero_grad, backward order kept |
| `SSE_Interconnect_Composed` (1257) | `SSE_Interconnect_Composed` | trim | ours (@added); class attributes kept: obc, `_optimizer = None` (1287, the backing field of the kept `optimizer` property, D1); dropped: orth_penalty, loss_stats, traj_penalty, _traj_*, obc_validation_diagnostics |
| `SSE_Interconnect_Composed.__init__` (1305) | `SSE_Interconnect_Composed.__init__` | trim | Parameterized_MSD guard (1309-1317) and _param_ix removed; _ann_ix, _mats_ix kept |
| `SSE_Interconnect_Composed._sync_prediction_state_for_validation` (1332) | `SSE_Interconnect_Composed._sync_prediction_state_for_validation` | trim | diagnostic dict (1367-1380) removed; (1364-1365) becomes `theta = self.obc.prediction_coefficient(self.hfn.connected_blocks[self._ann_ix])` (the renamed solve, `OBCLifecycle` rows), then `corr.set_frozen(theta, self.obc.basis.vbar)` (1366); the wiring guard (1343-1346) removed like the identical one in `loss` (D5, C34; `corr = ...` at 1342 kept); refresh rules (1350-1359) and the epoch floor(+1e-6) kept |
| `SSE_Interconnect_Composed.loss` (1383) | `SSE_Interconnect_Composed.loss` | trim | removed: wiring guards (1402-1405), w-is-None branch (1421-1422), loss_stats (1430-1431), param_loss sweep (1435-1436: adds the float 0.0, exact), orth penalty (1440-1445); kept in order: encoder, pass_mats hoist, OBC maybe_refresh/coefficient/set_frozen/tangent_pair, weighted MSE (hazard H5) |
| `SSE_Interconnect_Composed.checkpoint_save_system` (1505) | `SSE_Interconnect_Composed.checkpoint_save_system` | trim | stash = obc, obc_reference, training_phase: term `self._TRAJ_UNPICKLABLE + ` (1506) removed |
| `SSE_Interconnect_Composed.checkpoint_load_system` (1514) | `SSE_Interconnect_Composed.checkpoint_load_system` | trim | stash and reinstall obc, obc_reference, training_phase: terms `self._TRAJ_UNPICKLABLE + ` and ` + self._TRAJ_COUNTERS` (1516) removed; optimizer migration (1521-1524: a clean dump never holds `optimizer`, the kept property writes `_optimizer`), adapter rebind and re-wrap (1525-1567) removed |
| `SSE_Interconnect_Composed.__getstate__` (1570) | `SSE_Interconnect_Composed.__getstate__` | trim | drops obc, obc_reference, training_phase (deepcopy in the evaluation) |
| `SSE_Interconnect_Composed.optimizer` (1622) | `SSE_Interconnect_Composed.optimizer` (getter) | copy | kept (D-275 D1): `return self._optimizer` |
| `SSE_Interconnect_Composed.optimizer` (1626) | `SSE_Interconnect_Composed.optimizer` (setter) | trim | kept (D-275 D1); the trajectory-penalty re-wrap (1628-1630) removed, `self._optimizer = opt` (1627) kept. With the getter four lines: a clean deepSI dump stores the Adam object as `_optimizer`, exactly as mainline, so a resume restores moments exactly where mainline does (from a pair) and nowhere else (section 3 "Reruns") |

- Dropped (save_signals path): `Interconnect.reset_saved_signals`.
- Dropped (its only readers are the debugging prints 355, 439, 470, removed with `debugging`): `Interconnect.convert_signal_ix_to_name`.
- Dropped (printing): `Signal_Connection.__str__`, `Signal_Connection.__repr__`.
- Dropped (not on the spec path): `modified_encoder_net`, `modified_encoder_net.__init__`, `modified_encoder_net.forward`, `SSE_Interconnect_ParamLoss.loss`.
- Dropped (only builds an encoder when none is injected (647-649); init_model no longer calls it): `SSE_Interconnect.init_nets`.
- Dropped (overridden by SSE_Interconnect_Composed.loss): `SSE_Interconnect.loss`.
- Dropped (base no-op; the instance is always SSE_Interconnect_Composed): `SSE_Interconnect._sync_prediction_state_for_validation`.
- Dropped (open-loop deepSI paths): `SSE_Interconnect._model_dtype`, `SSE_Interconnect.init_state_multi`, `SSE_Interconnect.measure_act_multi`.
- Dropped (replaced by SSE_Interconnect_Composed): `SSE_Interconnect_ParamLoss`.
- Dropped (nf probe statistics (no numeric effect)): `WindowErrorStats`, `WindowErrorStats.__init__`, `WindowErrorStats.reset`, `WindowErrorStats.update`, `WindowErrorStats.summary`.
- Dropped (legacy optimizer migration only): `SSE_Interconnect_Composed.__setstate__`.
- Dropped (trajectory penalty (fit delegates to the parent when it is None)): `SSE_Interconnect_Composed._assert_optimizer_supported`, `SSE_Interconnect_Composed._wrap_optimizer_step`, `SSE_Interconnect_Composed._wrap_optimizer_step.step_with_penalty`, `SSE_Interconnect_Composed._wrap_optimizer_step.step_with_penalty.wrapped`, `SSE_Interconnect_Composed._unwrap_optimizer_step`, `SSE_Interconnect_Composed.fit`.
- Module and class constants: `SSE_Interconnect_Composed._TRAJ_UNPICKLABLE`, `SSE_Interconnect_Composed._TRAJ_COUNTERS`, `SSE_Interconnect_Composed._TRAJ_ALLOWED_OPTIMIZERS`: drop (trajectory penalty); `SSE_Interconnect_Composed._OBC_UNPICKLABLE`, `SSE_Interconnect_Composed._PHASE_UNPICKLABLE`: copy -> MA/fit_systems/interconnect.py.
- Imports: `import warnings` (6, only the removed empty-matrix warning), `from tqdm.auto import tqdm` (10), `DataLoader` from 11 (the Dataset branch; the other name of 11 stays, 943 and 950 read it), `from copy import deepcopy` (12, the removed init branch), `Tictoctimer` from 14; in 18 the names `Parameterized_Linear_Output_Block`, `Parameterized_MSD_State_Block`, `Parameterized_Linear_State_Block` (dropped; their one reader, the `Parameterized_MSD` guard of `SSE_Interconnect_Composed.__init__`, is removed) (C, X2).

#### `MA/fit_systems/lbfgs_polish.py` -> `MA/fit_systems/lbfgs_polish.py`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `PolishSpec` (50) | `PolishSpec` | trim | fields kept: n_windows, max_iter, inner_iter, history_size, tol_grad, tol_change, seed; dropped: freeze, meter_rel_tol, time_budget_s, chunk, meters_gate |
| `decide` (97) | `decide` | trim | kept as a function so the `reason` text is mainline's, which layer 3 compares (C22, the "copy the strings exactly" option): signature `decide(val_before, val_after, *, loss_finite, validation_measure='sim-RMS')` (terms `meters_before, meters_after`, `meter_rel_tol`, `accept_without_validation=False` removed from 97-99); kept 110-116 with the `if not accept_without_validation:` header (114) collapsed; removed the meter path of the unchecked case (117-120) and the meter regression (124-126), since `meters_gate=False` passes empty meters; kept 121-123 and 127 |
| `_named_params` (134) | `_named_params` | copy | parameter order = deepSI parameters_with_names order (hazard H7) |
| `_snapshot` (155) | `_snapshot` | copy |  |
| `_restore` (165) | `_restore` | copy |  |
| `_resync` (171) | `_resync` | copy |  |
| `window_count` (206) | `window_count` | trim | stride == 1 branch (226-227) removed |
| `FixedBatch` (234) | `FixedBatch` | trim | one chunk; yields (uh, yh, uf, yf, kwargs) once |
| `_eager` (347) | `_eager` | copy | restored for decision 4 (C15): the compiled try and eager fallback of a polish-only start |
| `FixedBatch.__init__` (258) | `FixedBatch.__init__` | trim | removed: the `chunk=None` parameter (258), the extra-array guards (309-318, D5: names and arrays come from the clean simulator), `self.chunk` and `self.n_chunks` (323-324); term: `'%d chunk(s), '` and its `self.n_chunks` argument in the print (329-332). Kept: stride scaling, window_count guard (277-283, deepSI's windowing), default_rng(seed).choice + sort (288-289) |
| `FixedBatch.__init__._to_tensor` (296) | `FixedBatch.__init__._to_tensor` | copy |  |
| `FixedBatch.__iter__` (334) | `FixedBatch.__iter__` | rewrite | one chunk: `uh, yh, uf, yf = self._cols[:4]`; `kw = dict(self._loss_kwargs)`; `kw.update(zip(self._names, self._cols[4:]))`; `yield uh, yh, uf, yf, kw`. Mainline's one chunk is `lo, hi = 0, n_used`, so its slices `c[0:n_used]` hold the same values, and its weight is 1.0 |
| `_make_closure` (464) | `_make_closure` | copy |  |
| `_make_closure.closure` (466) | `_make_closure.closure` | trim | terms: the loop unpacks five values, `for uh, yh, uf, yf, kw in batch:` (470), and `* w` goes from 471 (`x * 1.0` is exact); grad-None fill (476-478) removed (torch's LBFGS reads a None gradient as zeros); `zero_grad(set_to_none=False)`, `total`, `return torch.as_tensor(total)` (468-469, 473, 479) copied (a default-dtype float32 scalar, hazard H7) |
| `_loss_value` (483) | `_loss_value` | trim | terms: `* w` and the six-value unpacking in 496-497 become `fit_sys.loss(uh, yh, uf, yf, **kw) for uh, yh, uf, yf, kw in batch` |
| `_lbfgs_counts` (500) | `_lbfgs_counts` | copy |  |
| `_is_oom` (515) | `_is_oom` | copy | GPU failure path (reason text, empty_cache) |
| `_probe_closure` (522) | `_probe_closure` | copy | two closures before stepping; a failure or non-reproducible loss aborts the polish (weights unchanged) |
| `lbfgs_polish` (549) | `lbfgs_polish` | trim | removed: the `meters_fn` and `accept_without_validation` parameters (551; `force_eager=False` stays, 679 assigns it); terms `meters_before={}, meters_after={}` (581), `n_chunks=0` (582), `peak_mem_bytes=0` and `frozen=tuple(spec.freeze)` (583) in the outcome dict; `_all_meters` (588-595) and its calls (614, 774); `frozen = set(spec.freeze)` (621), the unknown-name check (622-625) and the empty-parameter return (627-629), with term `n not in frozen and` removed from 626, so `params = [p for n, p in named if p.requires_grad]` (626 kept with its `requires_grad` filter, C13); 630 kept (`params_polished`, a section 5 key); the `with _quiet_probes(fit_sys), _frozen(named, frozen):` header (632, collapsed: `train` restored the probe tuple before the polish, and nothing is frozen); term `chunk=spec.chunk,` in the `FixedBatch(...)` call (634); term `n_chunks=batch.n_chunks` (636); `warm_up_lazy_init(fit_sys, batch)` (641; on a polish-only start `val_before` is None, so `_validate()` at 613, or the OBC rebuilt score at `GD/training.py:790`, runs an eager no-grad forward first and completes the lazy init, C15); peak-memory stats (647-648, 769-770); the time budget (757-759); the meters gate (778) and, in the `decide` call (779-784), the terms `out['meters_before'] if gate else {}, out['meters_after'] if gate else {},`, `meter_rel_tol=spec.meter_rel_tol` and `accept_without_validation=accept_without_validation`; term: the peak-memory suffix of the print (795-798) becomes `''`. Kept as copies, in order (decision 4, C15, C16, X8): snapshot, `val_before` (passed, else measured), FixedBatch, `_new_opt` (650-654), `n_closure`, `sim` and `compiled_available` (656-658), the compiled try (666-671), the eager fallback (673-679) and the second eager probe with its abort, restore and `_resync` (681-698), n_outer blocks of opt.step(closure) with exception restore (721-741), convergence break done < inner_iter (749-756), iteration cap (760-761), loss_last (765-768), ran, val_after, `decide`, reason (785), restore + _resync on rejection (787-789), the outcome prints |
| `lbfgs_polish._validate` (597) | `lbfgs_polish._validate` | trim | the `val_sys_data is None` branch (598-599, always given) removed and the `with _quiet_probes(fit_sys):` header (604) collapsed; eval(), _resync, cal_validation_error, train() kept |
| `lbfgs_polish._new_opt` (650) | `lbfgs_polish._new_opt` | copy | LBFGS lr 1, strong_wolfe |

- Dropped (meters do not gate (meters_gate=False)): `meter_regression`.
- Dropped (meters): `generic_meters`, `lbfgs_polish._all_meters`.
- Dropped (freeze = () (no-op)): `_frozen`.
- Dropped (no probes or loss_stats are attached during the polish (train restores the probe tuple before it)): `_quiet_probes`.
- Dropped (after `fit()` the lazy init is long done and the call only re-solves the same OBC coefficient; on a polish-only start `_validate()` or the rebuilt score runs an eager forward before any probe, C15): `warm_up_lazy_init`.
- Module and class constants: `__project_origin__`: copy (marker).
- Imports: `Optional` and `Tuple` from the typing import (39), read only by the dropped `PolishSpec` fields.

#### `MA/fit_systems/obc.py` -> `MA/fit_systems/obc.py`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `OBCRankLossAbort` (118) | `OBCRankLossAbort` | copy |  |
| `RankReport` (124) | `RankReport` | trim | dataclass fields kept: n_rows, n_cols, singular_values, rank, tol, cond (126-130, 132; read by `OBCBasis.__init__` 178-179 and `OBCLifecycle.refresh` 456-464); removed: rtol, col_norms, col_corr, build_seconds, svd_seconds, extra (131, 133-137). The smaller exact option of C10, X3: `from_matrix` is trimmed to match |
| `OBCBasis` (160) | `OBCBasis` | trim | the class-body alias `rho = r_overlap` (287) removed (both names are dropped methods; as a copy it would raise NameError at import) |
| `OBCBasis.__init__` (169) | `OBCBasis.__init__` | copy |  |
| `OBCBasis.rank_tolerance` (183) | `OBCBasis.rank_tolerance` | copy |  |
| `OBCBasis.from_matrix` (191) | `OBCBasis.from_matrix` | trim | removed: the `build_seconds` parameter (192), the dtype and shape guards (194-198, D5: `J` is the clean basis builder's float64 stack), the SVD timing (212, 214), `declared_rtol` (215-216) and `corr` (221), which only fed the report; in the `RankReport(...)` call (222-233) the terms `rtol=float(declared_rtol),` (228) and 229-233 (`col_norms`, `col_corr`, `build_seconds`, `svd_seconds`, `extra`). Kept exactly: the Gram diagonal and column norms (205-206), which give the column scale, the column scaling (207-211), the float64 SVD (213), tol, rank and cond (217-219), the rank truncation (234-238) (hazard H9; C10, X3) |
| `OBCBasis.coefficient` (242) | `OBCBasis.coefficient` | trim | the field-shape guard (244-245) removed (D5: `F` is the clean reference field) |
| `OBCBasis.to` (322) | `OBCBasis.to` | copy | GPU path (basis and field on different devices) |
| `OBCBasis.device` (330) | `OBCBasis.device` | copy |  |
| `build_stacked_sensitivity` (341) | `build_stacked_sensitivity` | copy | evaluation interface; dtype guard kept |
| `evaluate_step_rows` (367) | `evaluate_step_rows` | copy |  |
| `OBCLifecycle` (392) | `OBCLifecycle` | trim |  |
| `OBCLifecycle.__init__` (410) | `OBCLifecycle.__init__` | trim | removed: the refresh-policy guard (415-416, D5), the `verbose` and `name_columns` parameters (413) with `self.verbose`, `self.names` (424-425; `attach_obc` stops passing them, `GD/obc_gantry.py:397-398`, C12), `basis0` (429), `n_objective_evals`, `last_theta`, `last_field_norm` (434-436), `last_validation_diagnostics` (441), `history` (442). Kept: the rank policy state (basis, last_refresh_epoch, consecutive_rank_loss, max 2) and, because `refresh` still writes them (C11, merged B8), the counters `n_refresh`, `n_rank_loss` (431-432) and `timing` (443) |
| `OBCLifecycle.refresh` (446) | `OBCLifecycle.refresh` | trim | removed: term `build_seconds=build_s,` (451; `from_matrix` has no such parameter, `J, build_s = ...` at 450 copied); the verbose prints (473-477, 487-493); `entry['retained_previous']` (478); `prev`, `basis0` and the r_J entries (481, 483-486; `relative_change` is dropped); the history append (498). Kept (B8): `t0` (448), the entry dict (458-459, because 496 writes the refresh time into it), the counters (461, 494), `last_refresh_epoch` (495, read by `maybe_refresh`), the timing append (496-497); the rank policy whole: retain the previous basis on one loss, abort on two consecutive or with no basis (460-472, 479-480, 482); device move (454-455) |
| `OBCLifecycle.maybe_refresh` (501) | `OBCLifecycle.maybe_refresh` | copy |  |
| `OBCLifecycle.coefficient` (512) | `OBCLifecycle.coefficient` | trim | bookkeeping (522, 527-530: t0, n_objective_evals, last_theta, last_field_norm, timing) removed; the basis-None guard (519-521) removed (D5: `loss` calls `maybe_refresh` first, which builds a basis or raises); device move kept |
| `OBCLifecycle.validation_coefficient` (543) | `OBCLifecycle.prediction_coefficient` | rewrite | returns theta only: the no-grad solve of _prediction_solve(diagnostics=False) (558-568); theta is computed before the diagnostics in mainline, so it is the same tensor |
| `OBCLifecycle._prediction_solve` (558) | (merged) | merge | into OBCLifecycle.prediction_coefficient (lines 561-568, with the `if not diagnostics: return theta, None` of 567-568 becoming `return theta`); the basis-None guard (559-560) removed (D5: `_sync_prediction_state_for_validation` builds a basis when none exists, IC:1351-1358); the diagnostics part (569-591, reads the dropped `r_perp`, `apply` and `last_validation_diagnostics`) dropped |

- Dropped (static_pass_buffers=False): `_StaticCopy`, `_StaticCopy.forward`, `_StaticCopy.backward`, `StaticPassBuffers`, `StaticPassBuffers.__init__`, `StaticPassBuffers._alloc`, `StaticPassBuffers.__call__`, `StaticPassBuffers.reset`.
- Dropped (refresh print (comparison.md: diagnostic)): `RankReport.lines`.
- Dropped (diagnostic (comparison.md)): `OBCBasis.apply`, `OBCBasis.jt`, `OBCBasis.project_onto`, `OBCBasis.project_out`, `OBCBasis.r_perp`, `OBCBasis.r_overlap`, `OBCBasis.frobenius`, `OBCBasis.relative_change`, `OBCBasis.memory_bytes`.
- Dropped (jvp path): `directional_correction`.
- Dropped (its name is reused by the merged validation path above): `OBCLifecycle.prediction_coefficient`.
- Dropped (diagnostic): `OBCLifecycle.stacked_diagnostics`, `OBCLifecycle.memory_bytes`, `OBCLifecycle.state_summary`.
- Module and class constants: `__project_origin__`: copy (marker).
- Imports: `field` from the dataclasses import (37), read only by the dropped `RankReport.extra`.

#### `MA/fit_systems/pre_encoder.py` -> `MA/fit_systems/pre_encoder.py`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `linear_encoder_init_aug` (306) | `linear_encoder_init_aug` | trim | @added; state_dict names Wb_psi_y, Wb_psi_u, Wa_psi_y, Wa_psi_u, net.{0,2,4} |
| `linear_encoder_init_aug.__init__` (321) | `linear_encoder_init_aug.__init__` | trim | only wb_init="random", xavier_uniform_gain1, no offsets. Parameters removed: `A, B, C, D` (323-326), `flag_linear_only` (336), the offset inputs `u_mean, std_u, y0, ystd, x_mean, std_x` (340-342), `wa_encoder_init` (351), `wb_init` (355); `GD/model.py:251-261` stops passing them (C9, X5). Statements removed: `self.flag_linear_only` (365), the `na == nb` assert (367, D5), the wb_init guard (369-370), the `if wb_init == 'random':` header (371, collapsed) with the reconstructability map (378-417); the legacy Kaiming branch (422-427), the `elif wa_encoder_init == 'xavier_uniform_gain1':` header (428, collapsed) and its else raise (436-437); the `if not self.flag_linear_only:` header (443, collapsed); the `n_hidden_layers > 0` assert (447, D5); the `if wb_init == 'random':` header (455, collapsed) with its else (460-461); the D-055 offsets (467-486). Draw order kept: Wb uniform_ (376-377), Wa xavier (432-433), net Linear x3 (446-451), then zeroing (454, 458-459) (hazard H1) |
| `linear_encoder_init_aug.forward` (488) | `linear_encoder_init_aug.forward` | trim | fix_enabled offsets (498-500, 508-509) removed; the `if self.flag_linear_only: return x` branch (518-519) removed and its else (520) collapsed; input order [u, y] for the net, W_u uh + W_y yh for the linear part kept |

- Dropped (not on the spec path): `convert_to_same_type`, `System_data_with_x`, `System_data_with_x.__init__`, `System_data_with_x.to_hist_future_data`, `System_data_norm_with_x`, `System_data_norm_with_x.__init__`, `System_data_norm_with_x.transform`, `System_data_norm_with_x.inverse_transform`, `linear_encoder_init`, `linear_encoder_init.__init__`, `linear_encoder_init.forward`, `SS_pre_encoder`, `SS_pre_encoder.__init__`, `SS_pre_encoder.init_nets`, `SS_pre_encoder.loss`, `SS_pre_encoder.n_step_error`.
- Module-level statements: the `__main__` demo block (651-686) removed (it runs the dropped `linear_encoder_init`; not run on import).

#### `MA/fit_systems/ps2_opening.py` -> `MA/fit_systems/ps2_opening.py`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `PS2Spec` (37) | `PS2Spec` | copy |  |
| `calibration_split` (59) | `calibration_split` | copy |  |
| `PS2Opening` (69) | `PS2Opening` | copy |  |
| `PS2Opening.__init__` (70) | `PS2Opening.__init__` | trim | the alignment guard (74-75) removed (D5: `for_pipeline` builds the three lists together); the empty-calibration guard (79-80) kept (it depends on the record classes of the manifest, data) |
| `PS2Opening.for_pipeline` (87) | `PS2Opening.for_pipeline` | copy |  |
| `PS2Opening.done` (98) | `PS2Opening.done` | copy |  |
| `PS2Opening.after_step` (101) | `PS2Opening.after_step` | copy |  |
| `PS2Opening.summary` (130) | `PS2Opening.summary` | copy |  |
| `PS2Opening.summary_json` (135) | `PS2Opening.summary_json` | copy | summary feeds the .npz ps2_summary |
| `PS2Opening._device` (139) | `PS2Opening._device` | copy |  |
| `PS2Opening._start` (143) | `PS2Opening._start` | copy |  |
| `PS2Opening._sample` (161) | `PS2Opening._sample` | copy |  |
| `PS2Opening._residual` (183) | `PS2Opening._residual` | copy |  |
| `PS2Opening._xa` (191) | `PS2Opening._xa` | copy |  |
| `PS2Opening._s1_step` (194) | `PS2Opening._s1_step` | copy |  |
| `PS2Opening._cal_unexplained` (207) | `PS2Opening._cal_unexplained` | copy |  |
| `PS2Opening._fit_map` (213) | `PS2Opening._fit_map` | copy | D-274 change 1 (233-235): the start parameters are the first candidate |
| `PS2Opening._fit_map.snap` (220) | `PS2Opening._fit_map.snap` | copy |  |
| `PS2Opening._fit_map.ee` (226) | `PS2Opening._fit_map.ee` | copy | D-274: add the comment that T.var(0) divides by B-1, i.e. (B-1)/B times eq:ps2_s2, same minimiser |

- Module and class constants: `__project_origin__`: copy (marker).

#### `MA/systems/gantry_ss.py` -> `MA/systems/gantry_ss.py`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `build_poly_constants` (150) | `build_poly_constants` | copy |  |
| `build_G_matrix_entries` (184) | `build_G_matrix_entries` | copy |  |

- Module and class constants: `__project_origin__`: copy (marker); `_D`, `mb`, `mh`, `m1`, `m2`, `Jb`, `Jh`, `cg1`, `cg2`, `cy`, `cb1`, `cb2`, `kb1`, `kb2`, `Lb`, `d`, `_z`, `M1`, `M2`, `C`, `K`, `P`, `Cd`, `Dd`: copy (float32, hazard H6) -> MA/systems/gantry_ss.py; `fs`, `ts`, `M0`: drop (not read); `_kx_art`, `_ky_art`: drop (GANTRY_KX_ART/KY_ART env overrides).
- Module-level statements: the two overrides and their K[0,0], K[2,2] branch (108-113) removed.

#### `MA/utils/deepSI_corrections.py` -> `MA/utils/deepSI_corrections.py`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `fixed_System_data` (6) | `fixed_System_data` | copy |  |
| `fixed_System_data.__init__` (7) | `fixed_System_data.__init__` | copy |  |
| `fixed_System_data.to_hist_future_data` (10) | `fixed_System_data.to_hist_future_data` | copy | window arithmetic (hazard H5) |
| `fixed_System_data_norm` (30) | `fixed_System_data_norm` | trim |  |
| `fixed_System_data_norm.__init__` (44) | `fixed_System_data_norm.__init__` | copy |  |
| `fixed_System_data_norm.is_id` (51) | `fixed_System_data_norm.is_id` | copy |  |
| `fixed_System_data_norm.transform` (69) | `fixed_System_data_norm.transform` | copy |  |

- Dropped (not on the spec path): `fixed_System_data_norm.make_training_data`, `fixed_System_data_norm.fit`, `fixed_System_data_norm.inverse_transform`, `fixed_System_data_norm.u_transform`, `fixed_System_data_norm.y_transform`, `fixed_System_data_norm.u_inv_transform`, `fixed_System_data_norm.y_inv_transform`, `fixed_System_data_norm.__repr__`, `convert_to_same_type`.
- Module and class constants: `__project_origin__`: copy (marker).

#### `MA/utils/torch_nets.py` -> `MA/utils/torch_nets.py`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `live_g_feed_forward_nn` (170) | `live_g_feed_forward_nn` | copy | three nn.Linear draws at float32 (hazard H1) |
| `live_g_feed_forward_nn.__init__` (178) | `live_g_feed_forward_nn.__init__` | trim | the routing-length guard (182-183) and the `n_hidden_layers > 0` assert (187) removed (D5: both from the config) |
| `live_g_feed_forward_nn.forward` (203) | `live_g_feed_forward_nn.forward` | copy |  |

- Dropped (not on the spec path): `linear_encoder_net`, `linear_encoder_net.__init__`, `linear_encoder_net.forward`, `feed_forward_nn`, `feed_forward_nn.__init__`, `feed_forward_nn.forward`, `identity_init_simple_res_net`, `identity_init_simple_res_net.__init__`, `identity_init_simple_res_net.forward`, `linear_mapping`, `linear_mapping.__init__`, `linear_mapping.forward`, `zero_init_linear_mapping`, `zero_init_linear_mapping.__init__`, `zero_init_linear_mapping.forward`, `zero_init_feed_forward_nn`, `zero_init_feed_forward_nn.__init__`, `zero_init_feed_forward_nn.forward`, `paper_resnet`, `paper_resnet.__init__`, `paper_resnet.forward`, `zero_init_resnet`, `zero_init_resnet.__init__`, `zero_init_resnet.forward`, `unit_variance_feed_forward_nn`, `unit_variance_feed_forward_nn.__init__`, `unit_variance_feed_forward_nn.forward`, `HybridGantryEncoder`, `HybridGantryEncoder.__init__`, `HybridGantryEncoder.forward`, `LinearInitEncoderWrapper`, `LinearInitEncoderWrapper.__init__`, `LinearInitEncoderWrapper.forward`, `positive_default_encoder_net`, `positive_default_encoder_net.__init__`, `positive_default_encoder_net.forward`.

#### `MA/utils/utils.py` -> `MA/utils/utils.py`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `added` (7) | `added` | copy | the @added marker |
| `to_tensor` (15) | `to_tensor` | copy | torch.Tensor(ndarray) = float32 materialisation (hazard H6) |
| `detect_algebraic_loop` (31) | `detect_algebraic_loop` | copy |  |
| `expansion_matrix` (157) | `expansion_matrix` | copy | real matmuls in the summation (hazard H4) |
| `selection_matrix` (162) | `selection_matrix` | copy |  |

- Dropped (not on the spec path): `to_ndarray`, `plot_fft_freq`, `plot_sys_data`, `get_sys_data_dimensions`, `determine_sys_data_signal_dimension`, `determine_std_T`, `determine_std_T_sys_data`, `normalize_linear_ss_matrices`, `determine_yes_no_query_binary_output`, `compute_noisy_y_from_SNR`.

#### Files dropped whole

- `GD/baselines.py` (diagnostic (Dirk excluded)): `stepwise_rollout`, `compute_baseline_fp_nrms`, `compute_baseline_fp_nrms._fp_step`, `compute_all_baselines`; constants `__project_origin__`.
- `GD/evaluation.py` (diagnostic (Dirk excluded)): `capture_loss_history`, `report_joint_estimation`, `_record_mean`, `evaluate_and_save`, `_make_plots`, `_build_save_dict`, `per_record_nrms`; constants `__project_origin__`.
- `GD/obc_diagnostics.py` (diagnostic (Dirk excluded)): `qualify_basis`, `print_qualification`, `principal_angles`, `absorber_truth_one_step`, `absorber_truth_one_step.M`, `absorber_truth_one_step.deriv`, `check_truth_parameters`, `discrepancy_fields`, `rho_table`; constants `__project_origin__`.
- `CM/feedback_curriculum.py` (not on the spec path): `FixedWindowBatch`, `build_fixed_window_batch`, `save_training_snapshot`, `load_training_snapshot`, `FeedbackTargetSelector`, `FeedbackTargetSelector.__init__`, `FeedbackTargetSelector.initial_strength`, `FeedbackTargetSelector._evaluate`, `FeedbackTargetSelector._checkpoint_path`, `FeedbackTargetSelector._write_state`, `FeedbackTargetSelector.__call__`, `local_dynamics_diagnostic`, `local_dynamics_diagnostic.state_from_x`, `local_dynamics_diagnostic.state_from_u`, `local_dynamics_diagnostic.output_from_x`; constants `FeedbackTargetSelector.SCHEDULE`.
- `CM/oracle.py` (diagnostic (Dirk excluded)): `_M`, `_deriv`, `simulate`, `oracle_open_loop`, `truth_from_meta`, `_deriv_truth`, `oracle_closed_loop`; constants `__project_origin__`, `m1`, `m2`, `mb`, `mh`, `Jb`, `Jh`, `cg1`, `cg2`, `cy`, `cb1`, `cb2`, `kb1`, `kb2`, `Lb`, `d`, `P_np`, `MA_FRAC`, `MA`, `MH_RIGID`, `L0`, `FA`, `KA`, `ZETA_A`, `CA`, `_C4`, `_K4`, `_E43`.
- `CM/orth_penalty.py` (not on the spec path): `theta_bar_for`, `_cache_path`, `_x_logical_from_data`, `_make_block`, `_compute`, `build_orth_penalty`; constants `__project_origin__`, `_F64`.
- `CM/rezero_gate.py` (not on the spec path): `_Gate`, `_Gate.__init__`, `_Gate.gate`, `_Gate.forward`, `apply_rezero_gate`; constants `__project_origin__`.
- `MA/fit_systems/augmented_dynamics.py` (not on the spec path): `AugmentedDynamics`, `AugmentedDynamics.__init__`, `AugmentedDynamics.net`, `AugmentedDynamics.rho`, `AugmentedDynamics.gamma`, `AugmentedDynamics.effective_B`, `AugmentedDynamics.trainable_parameters`, `AugmentedDynamics.pole_table`, `AugmentedDynamics.forward`, `find_ann_block`, `find_wrapper`, `final_linear`, `reset_final_layer`, `empirical_input_scale`, `draw_input_map`; constants `__project_origin__`, `GATE_ZERO`, `OUTPUT_ZERO`.
- `MA/fit_systems/continuous_augmentation.py` (not on the spec path): `rk4_step`, `CombinedContinuousGantryBlock`, `CombinedContinuousGantryBlock.__init__`, `CombinedContinuousGantryBlock.correction_field`, `CombinedContinuousGantryBlock.derivative`, `CombinedContinuousGantryBlock.forward`, `CombinedContinuousGantryBlock.param_loss`, `CombinedContinuousGantryBlock.__getattr__`.
- `MA/fit_systems/gyorok_orth_projection.py` (not on the spec path): `_block_step`, `gantry_orth_matrix`, `gantry_orth_matrix.f_single`, `finite_difference_check`, `orth_basis`, `spectrum_report`, `GyorokOrthPenalty`, `GyorokOrthPenalty.__init__`, `GyorokOrthPenalty.Q`, `GyorokOrthPenalty.penalty_of_field`, `GyorokOrthPenalty.forward`; constants `__project_origin__`, `_F64`.
- `MA/fit_systems/multiple_shooting.py` (not on the spec path): `SSE_Interconnect_MultipleShooting`, `SSE_Interconnect_MultipleShooting._norm`, `SSE_Interconnect_MultipleShooting._defect_penalty`, `SSE_Interconnect_MultipleShooting.loss`; constants `__project_origin__`.
- `MA/fit_systems/obc_projection.py` (not on the spec path): `reduced_step`, `stacked_basis`, `stacked_basis.f`, `directional`, `rank_report`, `principal_angles`, `RankLossAbort`, `OBCBasis`, `OBCBasis.__init__`, `OBCBasis._check_rank`, `OBCBasis.range_basis`, `OBCBasis.coefficient`, `OBCBasis.rho`, `OBCBasis.residual_ratio`, `OBCBasis.spectrum`, `OBC_Projected_ANN_Block`, `OBC_Projected_ANN_Block.__init__`, `OBC_Projected_ANN_Block.phys`, `OBC_Projected_ANN_Block.init_block`, `OBC_Projected_ANN_Block.phys_input`, `OBC_Projected_ANN_Block.rebuild_basis`, `OBC_Projected_ANN_Block.clear_basis`, `OBC_Projected_ANN_Block._stack_physical_write`, `OBC_Projected_ANN_Block.coefficient`, `OBC_Projected_ANN_Block.forward`; constants `__project_origin__`, `DEFAULT_RCOND`.
- `MA/fit_systems/orth_projection.py` (not on the spec path): `OrthProjectionPenalty`, `OrthProjectionPenalty.__init__`, `OrthProjectionPenalty.penalty_of_field`, `OrthProjectionPenalty.forward`, `SSE_Interconnect_OrthLoss`, `SSE_Interconnect_OrthLoss.loss`; constants `__project_origin__`.
- `MA/fit_systems/white_box_models.py` (not on the spec path): `Discrete_White_Box_Model`, `Discrete_White_Box_Model.__init__`, `Discrete_White_Box_Model.forward`, `Discrete_White_Box_Model.f`, `Discrete_White_Box_Model.h`, `Discrete_Cascaded_Tanks`, `Discrete_Cascaded_Tanks.__init__`, `Discrete_Cascaded_Tanks.init_model`, `Discrete_Cascaded_Tanks.f`, `Discrete_Cascaded_Tanks.h`.
- `MA/systems/cascaded_tanks.py` (not on the spec path): `Cascaded_Tanks`, `Cascaded_Tanks.__init__`, `Cascaded_Tanks.deriv`, `Cascaded_Tanks.h`, `Extended_Cascaded_Tanks`, `Extended_Cascaded_Tanks.__init__`, `Extended_Cascaded_Tanks.deriv`, `Extended_Cascaded_Tanks.h`.
- `MA/systems/gantry_linearization.py` (not on the spec path): `gantry_linearize_and_discretize`, `verify_linearization`, `verify_linearization.deriv`; constants `__project_origin__`.
- `MA/systems/mass_spring_damper.py` (not on the spec path): `Mass_spring_damper`, `Mass_spring_damper.__init__`, `Mass_spring_damper.deriv`, `Mass_spring_damper.h`, `Msd_2dof`, `Msd_2dof.__init__`, `Msd_2dof.deriv`, `Msd_2dof.h`, `Msd_ndof`, `Msd_ndof.__init__`, `Msd_ndof.cast_parameter_to_array`, `Msd_ndof.determine_mask`, `Msd_ndof.deriv`, `Msd_ndof.h`, `Msd_ndof.expand_with_ix`, `Msd_ndof_lpf_output`, `Msd_ndof_lpf_output.__init__`, `Msd_ndof_lpf_output.cast_parameter_to_array`, `Msd_ndof_lpf_output.determine_mask`, `Msd_ndof_lpf_output.deriv`, `Msd_ndof_lpf_output.h`, `Msd_ndof_lpf_output.expand_with_ix`, `Msd_ndof_input_saturation`, `Msd_ndof_input_saturation.__init__`, `Msd_ndof_input_saturation.cast_parameter_to_array`, `Msd_ndof_input_saturation.determine_mask`, `Msd_ndof_input_saturation.deriv`, `Msd_ndof_input_saturation.h`, `Msd_ndof_input_saturation.expand_with_ix`.
- `GD/__init__.py` (no definitions): `__project_origin__` copy (marker).

## 3. Config design

One frozen dataclass. The five axes are its fields; the run environment (device, compilation, budgets that only the
smoke test and the harness change) are fields with production defaults; every fixed setting the kept code reads is
a `ClassVar` under its mainline attribute name, so copied code (`cfg.stride`, `cfg.adam_eps`, `hp['lr']`) needs no
edit, and a setting that only selected a removed branch is not defined at all (an `AttributeError` then exposes a
missed trim).

```python
@dataclass(frozen=True)
class RunConfig:
    # the five axes
    arm: str                                   # 'obc' | 'u'
    start: str                                 # 'detuned' | 'correct'
    n_a: int                                   # added states (spec: 0, 2, 8)
    ps2: bool                                  # PS2 opening of the added-state part
    noise: str                                 # 'lowpass' | 'noisefree'
    # run environment: production defaults; only the CLI flags, main (start_phase) and the harness change them
    device: str = 'cuda'                       # 'cuda' | 'cpu'
    compile_mode: Optional[str] = 'reduce-overhead'   # None = eager; needs device 'cuda'
    truncate_s: Optional[float] = None         # seconds kept per record (tests only)
    epochs: int = 200
    n_its: Optional[int] = None                # Adam update cap; then one validation at the cap (mainline rule, model.py:501)
    its_per_val: int = 1300
    lbfgs_max_iter: int = 100
    lbfgs_inner_iter: int = 20
    lbfgs_windows: int = 16384
    ps2_smoke: bool = False                    # the short PS2Spec of the mainline smoke (training.py:873-876)
    start_phase: str = 'adam'                  # 'adam' | 'lbfgs'; main sets 'lbfgs' for --polish-only only
    # roots (Dirk 2026-10-20: arguments with defaults); recorded in config.json, never read back from it
    data_root: str = DATA_ROOT                 # <repo>/Thesis-writeup/Data
    out_root: str = OUT_ROOT                   # <repo>/simulations/gantry_clean
```

`ClassVar` constants (values of the spec; labels as in mainline): `seed = 1`, `nx_phys, nu, ny = 6, 3, 3`,
`fs_orig = 20000`, `fs_new = 4000`, `stride = 10`, `nf_seconds = 0.1`, `na_nb = 30` (the one definition of
`na_nb`: the mainline property and its override go, C27), `n_nodes_per_layer = 16`,
`n_hidden_layers = 2`, `up_sample = 1`, `batch_size = 512`, `lr = 1e-5`, `adam_eps = 1e-16`, `param_prior = False`,
`param_rmse_baseline = 0.01` (HEURISTIC, D-076: still sets the `Lambda_combo` buffer, a state_dict key),
`use_f64 = True` (read by the copied `dtype_np`, `dtype_pt`, config.py:555, 559), `joint_estimation = True` (read
by the copied resume block, training.py:853), `lbfgs = True` (read by the copied pre-polish test,
training.py:932), `encoder_init = 'data_lsq'`, `validation_measure = 'sim-NRMSe'`,
`controller_gain = False`, `lbfgs_history = 50`, `lbfgs_tol_grad = 2.6e-5`,
`lbfgs_tol_change = 2.6e-7` (HEURISTIC, D-263), `obc_space = 'affine'`, `obc_refresh = 'epoch'`,
`obc_rank_rtol = None`, `obc_ref_successor = True`, `obc_ref_chunk = 16384`. The list covers every `cfg.<name>`
read of a kept statement (the name check of section 2); the implementer confirms it with a grep of `cfg\.\w+` over
the clean folder.

Properties: `obc` (arm == 'obc'), `nx_ann` (n_a), `ann_route_ix` (`tuple(range(6 + n_a))`), `mode`
(`thesis_taf_lowpass_noise` | `thesis_taf_noisefree`), `combo_init_detune` (None for 'correct'; for 'detuned' the
EP:336-337 statements copied, `float()` included (C30): `_signs = np.random.default_rng(self.seed).choice([-1.0,
1.0], size=10)` then `[float(1.0 + 0.10 * s) for s in _signs]`, with the EP:333-334 HEURISTIC (D-222) label; an
independent generator, H1), and the mainline derived properties `fs_new_hz`, `d`, `ts_new`, `dtype_np` (float64),
`dtype_pt` (float64), `nf` (the `int` form copied, `max(1, int(self.nf_seconds / self.ts_new))` = 400, C28), `hp`
(same keys, same order). `__post_init__` checks the axis and environment values against their allowed sets
(`start_phase` in {'adam', 'lbfgs'}, config.py:501-505) and `compile_mode is None or device == 'cuda'` (mainline
config.py:480-481); these values arrive from the CLI, a `config.json` or the harness (rule 3).

**`RunConfig.from_config_json(meta, **overrides)`** (classmethod, D-275 D2). The one way a run's `config.json` (a
dict) becomes a clean config: the evaluation (section 5, edit (b)), the reruns below and layer 4 use it.
- Axes: a clean file (`meta['CODE'] == 'clean'`) gives `ARM, START, N_A, PS2, NOISE`. A mainline file gives
  `arm = 'obc' if meta['OBC'] else 'u'`, `start = 'correct' if meta['COMBO_INIT_DETUNE'] is None else
  'detuned'`, `n_a = int(meta['NX_ANN'])`, `ps2 = bool(meta.get('PS2', False))`,
  `noise = {'thesis_taf_lowpass_noise': 'lowpass', 'thesis_taf_noisefree': 'noisefree'}[meta['MODE']]` (a
  `KeyError` rejects every other mode).
- Run environment from the file where present: `DEVICE, COMPILE_MODE, TRUNCATE_S, EPOCHS, N_ITS, ITS_PER_VAL,
  LBFGS_MAX_ITER, LBFGS_INNER_ITER, LBFGS_WINDOWS, PS2_SMOKE` (mainline files have all but `PS2_SMOKE`).
  `start_phase`, `data_root` and `out_root` are never read from a file: they come from `overrides` or the defaults,
  so a run folder copied from the server evaluates against the local data root (C17).
- Rejection ("it rejects any other setting"): with `cfg` built from the above, `want =
  json.loads(json.dumps(config_json_dict(cfg)))`; every key of `want` except `GIT`, `NF_PROBE_PRINT` (open choice
  20), the run-environment keys above, `START_PHASE`, and the clean-only keys (`ARM, START, N_A, NOISE, PS2_SMOKE,
  DATA_ROOT, OUT_ROOT, LAST_SAVE, CODE`) must equal `meta[key]`, else `ValueError` naming every differing key with
  both values. This checks the constant list below against each mainline file it reads (the nine final runs at
  layer 4), and a detuned mainline start must equal the clean draw.
- Returns `dataclasses.replace(cfg, **overrides)` (the evaluation passes `device='cpu', compile_mode=None,
  truncate_s=None`, as `eval_greybox.py:47` does now).

**Presets** (`GD/config.py PRESETS`, named by the thesis and evaluation labels, open choice 5):

| Preset | runs.tsv id | arm | start | n_a | ps2 |
|-|-|-|-|-|-|
| R1 | 221 | obc | detuned | 2 | 1 |
| R2 | 222 | obc | correct | 2 | 1 |
| R3 | 223 | u | correct | 2 | 0 |
| R4 | 224 | u | detuned | 2 | 1 |
| R5 | 225 | u | correct | 2 | 1 |
| R6 | 226 | u | correct | 0 | 0 |
| R7 | 227 | u | correct | 8 | 1 |
| R13 | 229 | obc | detuned | 0 | 0 |
| R14 | 230 | u | detuned | 0 | 0 |

Checked against runs.tsv rows 221 to 230 and `eval_common.FINAL_DESIGN` (identical). A preset fixes four axes; noise,
the fifth, is given separately, so 18 runs come from 9 presets.

**Launch (CLI, open choice 4; Dirk 2026-10-20: CLI with a default).** `cli()` in EP, `argparse`, usage line:
`python -u Clean/gantry_interconnect_dynamic.py [preset] [noise] [--cpu] [--smoke] [--resume RUN_DIR [--from
{best,last}] | --polish-only RUN_DIR] [--data-root P] [--out-root P]`, with the text "no arguments = R1 lowpass on
cuda with reduce-overhead compilation (the production run); without CUDA use --cpu".
- `preset` (choices `PRESETS`) and `noise` (choices `lowpass`, `noisefree`) are `nargs='?'` with `default=None`;
  `None` becomes `R1` and `lowpass` when no run folder is given. Flags, not words, so `entry.py --cpu` cannot be
  read as a preset (C21).
- No arguments: R1 lowpass with the production defaults (`device='cuda'`, `compile_mode='reduce-overhead'`); on a
  machine without CUDA `build_model` stops at its CUDA check (`GD/model.py:286-292`, kept), which the usage line
  states (C36).
- `--cpu`: `device='cpu'`, `compile_mode=None`. `--smoke`: the mainline THESIS_SMOKE values (EP:368-372 and
  training.py:873-876), eager: `truncate_s=2.0, compile_mode=None, n_its=4, its_per_val=4, lbfgs_inner_iter=1,
  lbfgs_max_iter=2, lbfgs_windows=256, ps2_smoke=True`. The compiled production path is tested by the harness's
  compiled GPU smoke (6.4), not by this flag.
- `--resume RUN_DIR`, `--polish-only RUN_DIR` (mutually exclusive), `--from {best,last}` (only with `--resume`,
  default `best`): see "Reruns" below. Giving a run folder together with `preset` or `noise` is an argparse error:
  the folder fixes the configuration. The config is `RunConfig.from_config_json(<RUN_DIR>/config.json,
  data_root=..., out_root=...)`, then `--cpu` / `--smoke` as above; `main` sets `start_phase='lbfgs'` for
  `--polish-only`.
- `--data-root P`, `--out-root P`: the two root fields (defaults `DATA_ROOT`, `OUT_ROOT`, `GD/config.py`).
- Server: from the repository root, `srun python -u Clean/gantry_interconnect_dynamic.py R1 lowpass` (a new sbatch is
  step-4 or later work; the mainline `run_thesis.sbatch` environment block applies unchanged: threads, inductor and
  triton caches, CUDA_VISIBLE_DEVICES). Local: `python -u Clean/gantry_interconnect_dynamic.py R1 lowpass --cpu`.
- The harness builds `RunConfig(...)` (or `from_config_json`) directly and calls `main(cfg, run_id, rerun)`.
- Environment variables the clean code reads: `SLURM_JOB_ID` (run id, rerun id) and `GIT_SHA` / `GIT_COMMIT`
  (provenance fallback in `git_provenance`). No `THESIS_*`, `RESUME_CHECKPOINT`, `DATASET_AB_*`, `OBC_ARM`,
  `ANN_REZERO_GATE`, `GANTRY_KX_ART` or `GANTRY_KY_ART`.

**Reruns (D-275 D6 and its two amendments).** A run folder is one model; a continuation or a polish-only rerun
continues inside it. `main(cfg, rerun=(kind, run_dir, source))`, `kind` in {'resume', 'polish-only'}:
1. `run_id` = the folder's `config.json` `run_id` (kept: every file name, deepSI's `unique_code`, `orig_run_id`);
   `rerun_id` = `SLURM_JOB_ID`, else a timestamp.
2. Source checkpoint: polish-only takes the pair `<run_dir>/gantry_ckpt_<run_id>_pre_lbfgs`; resume takes deepSI's
   `<checkpoints>/SSE_Interconnect_Composed_<run_id>_best.pth`, or `_last.pth` with `--from last`, never one as a
   fallback for the other (mainline rule, `GD/training.py:97-98`). A missing source raises `FileNotFoundError`
   with the path (a file guard, rule 3).
3. Before anything is written, every file of `<run_dir>` except `config.json` (final pair, `_pre_lbfgs` pair,
   `val_hist.npz`, and in a mainline folder also `gantry_results_<id>.npz` and the other report files, which
   describe the old final model), and for a resume both deepSI dumps of `run_id` that exist, move into
   `<run_dir>/previous/<rerun_id>/` (`shutil.move`). Nothing is overwritten; `resume_ckpt` is the moved source (a
   `.pth` file, or the pair base path), which `resolve_checkpoint` reads as mainline does.
4. `config.json` gets one entry appended to a `RESUMES` list (created when absent): `time` (ISO), `type`
   ('resume' | 'polish-only'), `source` (the moved source, relative to the run folder), `code` ('clean'), `git`
   (`git_provenance()`), `rerun_id`; nothing else in the file changes, so the evaluation still sees one run per
   design and noise and needs no change.
5. Then the sequence of the EP `main` row with `resume_ckpt`: the outcome is written under the same run id; the
   four optional D-274 polish reruns (223, 225, 226, 230) write into their mainline folders this way (the
   `_pre_lbfgs` pair is a component pair both codes read, D3), and `from_config_json` reads their mainline
   `config.json`.
What a continuation restores is mainline's resume, copied (`resolve_resume_hp`, `load_checkpoint`, the `train`
resume block): the weights; `done_epochs` from the pair, or `round(epoch_counter)` of a deepSI dump
(`GD/training.py:106-109`), so `epochs_remaining = hp['epochs'] - done_epochs`; no PS2 replay (`done_epochs > 0`,
training.py:871); no encoder prefit (EP:444); Adam moments only from a pair on an Adam start (training.py:292-297).
A deepSI dump gives no moments in either code: `resolve_checkpoint` reads `d.get('optimizer')` (training.py:115),
while the kept `optimizer` property (D1) stores the Adam object in `__dict__` as `_optimizer` (IC:1287, 1626-1627),
and deepSI pickles `self.__dict__` (`deepSI/fit_systems/fit_system.py:493-496`). So every `--resume` (D6 sources
are dumps) starts Adam with fresh moments and prints the `load_checkpoint` NOTE (training.py:305-315), exactly as a
mainline resume from a dump does; the batch order restarts at deepSI's `default_rng(0)` and `epoch_counter` at 0
(a fresh `fit()`). Restoring moments from a dump would need `d.get('optimizer') or d.get('_optimizer')` in
`resolve_checkpoint`: a deviation from mainline, not made (changelog D1).

**config.json** (`main`, as EP:403-406): `config_json_dict(cfg, git)` writes every mainline key in mainline order
(`GD/config.py:679-790`) with the value of the run, then the new keys `ARM, START, N_A, NOISE, PS2_SMOKE,
DATA_ROOT, OUT_ROOT, LAST_SAVE` (`'every validation'`, D-275 amendment 2: the addition beyond mainline, recorded),
`CODE: "clean"`, then `GIT`; `main` adds `hp`, `run_id`, `data_dir` (= `traj_dir(cfg)`, as mainline). Keys of the
run: `MODE, SEED, COMBO_INIT_DETUNE, ANN_ROUTE_IX, NX_ANN, OBC, PS2, Y_DECIMATION, NF, NA_NB, FS_NEW, D, FS_ORIG,
STRIDE, NF_SECONDS, USE_F64, N_NODES_PER_LAYER, N_HIDDEN_LAYERS, UP_SAMPLE, BATCH_SIZE, LR, ADAM_EPS, EPOCHS, N_ITS,
ITS_PER_VAL, DEVICE, COMPILE_MODE, TRUNCATE_S, LBFGS_MAX_ITER, LBFGS_INNER_ITER, LBFGS_HISTORY, LBFGS_WINDOWS,
LBFGS_TOL_GRAD, LBFGS_TOL_CHANGE, START_PHASE, ENCODER_INIT, VALIDATION_MEASURE, CONTROLLER_GAIN, OBC_SPACE,
OBC_REFRESH, OBC_REF_CHUNK, OBC_RANK_RTOL, OBC_REF_SUCCESSOR, PARAM_PRIOR, PARAM_RMSE_BASELINE, JOINT_ESTIMATION,
LBFGS, RECORD_TRIM` from the fields, ClassVars and properties. Keys of removed options, written as literals (C25):
job 89792's `config.json` is on the server only, so the values are those the mainline code wrote it from (the
`RunConfig` defaults, EP `CFG` at EP:38-212 and `_THESIS_FIXED` at EP:342-352 with `THESIS_INIT=live`), and
`from_config_json` checks every one of them against 89792 to 89800 at layer 4: `ENCODER_INIT 'data_lsq'`,
`ANN_ACTIVATION 'tanh'`, `ANN_ARCH 'mlp'`, `G_INIT 'live'`, `LIVE_G_WEIGHT_INIT 'kaiming_uniform_a_sqrt5'`,
`WA_ENCODER_INIT 'xavier_uniform_gain1'`, `SNR None`, `PHYSICS_PARAMETERIZATION 'reduced'`, `PARAM_INIT_DETUNE None`,
`ORTH_BETA 4.66e-4`, `ORTH_POINT_STRIDE 100`, `ORTH_RANK_TOL 1e-12`, `CLOSED_LOOP True`, `ORTH False`,
`SAVE_FLAG True`, `CHECKPOINT_CHUNK 0`, `LBFGS_CHUNK None`, `LBFGS_FREEZE []`, `LBFGS_TIME_BUDGET_S None`,
`LBFGS_METER_REL_TOL 0.001`, `STATIC_PASS_BUFFERS False`, `LOSS_WEIGHTING 'servo'`, `LBFGS_METERS_GATE False`. One
key changes meaning: `NF_PROBE_PRINT` is written `False` (the probe no longer exists, open choice 20). All keys the
evaluation reads are present (section 5).

## 4. Data

- Data root `cfg.data_root` (default `DATA_ROOT = REPO_ROOT/Thesis-writeup/Data`, `--data-root`), recorded in
  `config.json` as `DATA_ROOT` and, resolved, `data_dir`. Two named entries,
  `DATASETS = {'lowpass': 'Coulomb-tanh-and-MSD-lowpass-noise', 'noisefree': 'Coulomb-tanh-and-MSD-noise-free'}`;
  `traj_dir(cfg) = os.path.join(cfg.data_root, DATASETS[cfg.noise])`, the one path every loader uses
  (`_load_record`, `load_datasets`, `_load_record_float64`). Nothing else in the data root is read (the twins are
  evaluation-only, section 5).
- **Split** from each entry's `MANIFEST.csv`, in manifest order: `block` training to train, validation to val,
  test-I and test-E to test, `frf` ignored; only rows with `controller == 'K1'` (D-274). Checked on both manifests on
  2026-10-19: training 18 K1, validation 6 K1, test-I 6 K1, test-E 5 K1 plus E5 (K2-g+, dropped), frf 12. Result
  18/6/11. File path `<folder>/<file>.mat` from the manifest columns.
- **Per record** (`_load_record`, unchanged arithmetic): `loadmat`; register the record's stored controller
  (`meta.controller`, into the controller registry); `u` = block mean of `u_total` over D = 5 samples in float64
  (D-087); `y` by the noise entry: `lowpass` = `r[::5] + sosfiltfilt(butter(8, 1600 Hz, fs 20 kHz, sos), y - r)[::5]`
  (D-232), `noisefree` = `y[::5]` (point sampling); `r` point-sampled; every array trimmed by `record_trim` = 20
  samples at both ends; then `truncate_s` (tests only); `e = y - r` in float64; `System_data(u, y, dt=2.5e-4)` in
  float64.
- **Derived**: `servo_sigma(e, k0 = 30)` (population std from k0, D-262) per record and axis; training loss weights
  `ystd / sigma_e`; validation scales for `sim-NRMSe`. Normalisation over the concatenated training records
  (`compute_normalization`, H6).
- The noise entry changes exactly two things: the y rule above and `mode` (folder family name, `MODE` and
  `Y_DECIMATION` in config.json). The step-1 trace found no other line that depends on it (`inventory-claude.md`
  A.0).
- Truth fields in the records (`x_logical`, `delta_a`, `vdelta_a`, `meta.truth`) are not read (open choice 11).

## 5. Evaluation interface

**Assumption of the handoff, settled.** `eval_common.py`, `eval_greybox.py`, `eval_overlap.py`, `eval_negation.py`,
`eval_frf.py` and `eval_blackbox.py` were read on 2026-10-19 (imports and every attribute access). They reach the
pipeline only through the interface of `inventory-claude.md` C6 and `inventory-codex.md` C4, with three additions
found since: `eval_overlap.deployed_basis` reads `corr.space` (default 'tangent' when absent) and passes `space=` to
`make_basis_builder` (`eval_overlap.py:47-54`; called only for an OBC run, where `corr` is not None,
`eval_overlap.py:105-106`, `eval_negation.py:74-76`); `eval_blackbox` builds its config with the black-box
`bbcl.setup.cfg_for`, a mainline `RunConfig` with about fifteen fields the clean config does not have; and
`scripts/gantry/thesis-results/inspect_checkpoint.py:30-31` inserts the mainline repository root and
`scripts/gantry` at the front of `sys.path` when it is imported (`eval_greybox.py:105`, `eval_overlap.py:141`), so a
path change in `eval_common` alone does not select the clean code (X1, edit (a) below).

What the clean code keeps for the evaluation:
1. **Package and module names**: `gantry_dynamic.{config, data, model, controller, training, closed_loop_report,
   obc_gantry}` and `model_augmentation.fit_systems.{closed_loop, obc, blocks}` with the imported names
   `RunConfig, record_trim, load_datasets, compute_normalization, _load_record, servo_sigma, traj_dir,
   _record_at_fs_new, _load_u, _resample_u, record_files, build_model, build_closed_loop (keywords train_files,
   val_files, val_data, verbose, e_train, e_val), build_controller_bank, y_op_for, load_checkpoint,
   _without_learned, _true_free, build_reference_set (cfg, norm, nx_ann=, files=, verbose=), make_float64_block,
   make_basis_builder (blk64, ref, row_ix, chunk, device=None, space='affine'; the evaluation passes
   `space=corr.space`), closed_loop_free_run_rms_batch (return_pred=), build_stacked_sensitivity, Static_ANN_Block`.
2. **Config round trip** (D-275 D2): `eval_greybox.run_config` calls `RunConfig.from_config_json(meta,
   device='cpu', compile_mode=None, truncate_s=None)` (edit (b)), which reads a clean `config.json` and, through
   its adapter, a mainline one (section 3). `discover_runs` reads `TRUNCATE_S, N_ITS, run_id, OBC, NX_ANN,
   COMBO_INIT_DETUNE, PS2`; `checkpoint_status` reads `EPOCHS`; `make_results` reads `OBC, COMBO_INIT_DETUNE` from
   the cached config. All written, and a `RESUMES` list (reruns) is a key none of them reads. Config attributes read
   by the evaluation: `seed, nx_ann, nx_phys, obc, obc_ref_chunk, ann_route_ix, controller_gain, dtype_pt, ts_new,
   hp`: all provided (section 3).
3. **Files**: run folder `out_root/{mode}_data_lsq/<run_id>/` with `config.json`, `gantry_ckpt_<rid>.pt` (keys
   `hfn`, `encoder`, `optimizer` unless dropped after an accepted polish or on a polish-only start),
   `gantry_ckpt_<rid>.npz` with the twelve mainline keys (`done_epochs, bestfit, bestfit_selected, elapsed, hp,
   orig_run_id, lbfgs_outcome, optimizer_state_dropped, start_phase, resumed_from, resumed_at_epoch, ps2_summary`;
   `start_phase`, `resumed_from`, `resumed_at_epoch` hold the real values, mainline `training.py:1005-1007`: `'adam'`,
   `''`, `-1` for a fresh run), the `_pre_lbfgs` pair (`done_epochs, bestfit, hp, orig_run_id`), `val_hist.npz`
   (mainline keys), and after a rerun `previous/<rerun_id>/` (D6). `bestfit` is the final event's score, read by the
   evaluation as `ref_score`. Checkpoint compatibility (D-275 D3): the component pair (`.pt` state_dicts, `.npz`)
   is read by both codes; a deepSI whole-system `_best.pth` is read only by the code that wrote it, because it
   pickles that code's classes (a mainline dump holds `_NfProbe` and `WindowErrorStats` instances,
   `GD/training.py:375, 678`, IC:902-908, which the clean namespace lacks; a clean dump holds `ValHistory`). So the
   clean `resolve_checkpoint` loads the `_best.pth` of an unfinished clean run, a preliminary evaluation of a
   mainline run stays on mainline, and layer 4 uses component pairs only. No `gantry_results_<rid>.npz`: the report
   is excluded, and `eval_greybox.py:178-179` already skips its check when the file is absent. `lbfgs_outcome` keeps
   the keys `ran, accepted, reason, n_iter, max_iter, n_closure, func_evals, wall_s, loss_first, loss_last,
   val_before, val_after, n_windows_used, n_windows_total, effective_stride, deterministic, used_compiled,
   params_polished` (no reader outside the run); `reason` is built by the same text as mainline (`decide` kept).
4. **state_dict names and shapes**: unchanged, because modules, attribute names, buffer registrations and block
   order (physics 0, output 1, ANN 2) are unchanged. For n_a = 2 (221 smoke checkpoint, C6):
   `connected_blocks.0.{free_params, mh, alpha, beta, gamma_, N0, N1, N2, Ax, Bw, Bu, A_combined, K_mat, C_mat,
   P_mat, std_x, std_x_1d, std_u, x_mean, u_mean, params_init, Lb, combo_init, Lambda_combo, coordinate_version,
   m_diff_eta0, m_diff_scale}`, `connected_blocks.1.{C, D}`, `connected_blocks.2.net.net.{0,2,4}.{weight, bias}`,
   `obc_correction.{E, vbar, theta_frozen}` (obc arm), `encoder.{Wb_psi_y, Wb_psi_u, Wa_psi_y, Wa_psi_u,
   net.{0,2,4}.{weight, bias}}`. Every buffer stays even where the trained block does not read it (the nominal LFR
   buffers): they are keys that `load_checkpoint` checks exactly. `out_gate` stays a non-persistent buffer.
   Layer 1 checks key sets and shapes against mainline for all 16 model cases.
5. **Objects and behaviour**: `fs.hfn.connected_blocks`, `fs.encoder`, `fs.na`, `fs.nb`, `na_right`, `nb_right`,
   `fs.norm` (`transform`, `u0`, `ustd`, `y0`, `ystd`), `fs.simulator`; `hfn.obc_correction.{theta_frozen, vbar,
   space}` (`GantryOBCCorrection.space = 'affine'` stays an attribute although it is constant: without it
   `getattr(corr, 'space', 'tangent')` would silently score overlap and negation on the tangent basis `[J]`
   instead of the deployed `[J f_bar]`, C2); physics block `free_params,
   params_init, combo_init, Lb, PARAM_NAMES, COMBO_NAMES, combos_of, pass_mats()` (the ten-tuple), the frozen-Y
   attributes `Y_op, N_op, d_op` that `frozen_y` sets, `estimation_parameters, combinations_from_free,
   free_from_combinations, transition_from_free`; reference set `Z_step, Z_field, n, record_ix, sample_ix,
   record_names, x_phys_next_rec`; `Norm` and `DataBundle` fields of section 2. `load_checkpoint(fs, path,
   joint_estimation=True, start_phase='lbfgs')` restores the saved OBC buffers and nothing re-syncs them;
   `y_op_for` raises `KeyError` for a non-K1 record (the evaluation's skip); `Interconnect.forward` initialises
   itself lazily (`eval_frf` calls `hfn(x, u)` before any rollout).

**Evaluation edits** (made in step 4 together with the harness, because layer 4 runs the evaluation on the clean
code; with `THESIS_CODE_ROOT` unset every existing evaluation command behaves exactly as now):
(a) One package-root selection (X1, B14). `eval_common.py:19-21` reads `THESIS_CODE_ROOT`: unset, it inserts the
repository root and `scripts/gantry` as now; set (e.g. `<repo>/Clean`), it inserts that folder only. In both cases
`scripts/gantry/thesis-results` follows (`inspect_checkpoint.clamped`, `added_poles`), and then `eval_common`
imports `gantry_dynamic` and `model_augmentation` at once and asserts that both `__file__` lie under the selected
root. That import pins both packages in `sys.modules`: when `inspect_checkpoint.py:30-31` later puts the mainline
root at the front of `sys.path`, every further `gantry_dynamic.*` or `model_augmentation.*` import still resolves
through the pinned package's `__path__`. `THESIS_RUNS_DIR` points at the clean family folder (or, for layer 4, the
staging folder of 6.5).
(b) `eval_greybox.run_config` (`eval_greybox.py:35-49`): after its `RunConfig` import, `if hasattr(RunConfig,
'from_config_json'): return RunConfig.from_config_json(meta, device='cpu', compile_mode=None, truncate_s=None)`;
the mainline lines below it stay for the mainline `RunConfig` (open choice 8).
(c) `eval_blackbox` stays on the mainline root (`THESIS_CODE_ROOT` unset): the black box R8 is out of scope and its
`cfg_for` needs the mainline `RunConfig`. Likewise `inspect_checkpoint.checkpoint_config` and `thesis_model`
(`inspect_checkpoint.py:35-82`, its own command line) build the mainline `RunConfig` with `na_nb_override` and
`nf_override` and stay mainline-only; the evaluation imports only `clamped` and `added_poles` from that file,
which read no config.

## 6. Equivalence plan

### 6.1 Hazards (what a faithful rewrite must reproduce)

Agreed by both inventories (Part C), made specific to the mapping:
- **H1 global draw order.** After the second seed (EP:409-410) the global torch stream is consumed by, in order:
  the live g_aug MLP (three `nn.Linear` at float32, input to output, in `Static_ANN_Block.__init__`, also for
  n_a = 0); `Wb_psi_y` then `Wb_psi_u` `uniform_` at float64 (values later overwritten by the least-squares start,
  the draws are not); one joint Xavier draw for W^a at float64; the encoder net's three `nn.Linear` at float32 then
  cast; on PS2 rows the decoder's two `nn.Linear` at the first `after_step`. Nothing may draw from a global stream
  between the second seed and the first update, and the first seed before the data (EP:390-391) stays. The detuning
  signs come from an independent `default_rng(1)` (section 3).
- **H2 dropped code draws nothing global.** The nf probe used `default_rng(cfg.seed)` (training.py:402) and the
  post-run diagnostics run after the final checkpoint; dropping them moves no stream (C1 of both inventories).
- **H3 Adam batch order.** `Dtype_DataLoader` keeps deepSI's `default_rng(0)` and the in-place `shuffle(ids)` of
  `My_Simple_DataLoader.__iter__` at every pass, consumed through `loop_dataset`, full batches only,
  `N_batch_updates_per_epoch = N // 512`, independent of the run seed. PS2 uses its own seeded generators (spec.seed,
  +321, +1000, a CPU `torch.Generator`); the polish subset is `default_rng(seed).choice` then sorted.
- **H4 step order.** Interconnect sums through the connection matrices (`expansion_matrix`, `selection_matrix` are
  real matmuls, `+=` in `order_output_signal_computation`), then `xp - obc_correction(x, u, obc_pass)`; the closed
  loop computes `output_only(x)`, then `baddbmm(bmm(M_error, e), M_state, xc)`, then `hfn(x, u + u_fb)`, with the
  normalisation folded into the bank matrices.
- **H5 reductions and windows.** Loss `((y_pred - yfuture) * w[:, None, :]).square().mean()`; weights and sigmas in
  float64 with `np.std` (ddof 0) from k0; windows from `fixed_System_data.to_hist_future_data` per record in manifest
  order with stride 10, `ctrl_ix` and `loss_w` aligned per record; validation grouped by post-k0 length, score
  `mean(per_channel / val_sigma_e)`.
- **H6 float32 materialisation, reproduced, not fixed.** `gantry_ss` tensors are float32 (`_D`); `to_tensor` is
  `torch.Tensor(ndarray)` (float32) for `std_x, std_u, x_mean, u_mean, C, D`; nominal LFR buffers are built in
  float32 before the cast; `compute_normalization` inverts `P.numpy().T` in float32 and casts; the OBC reference set
  and the pre-fit cast `P` to float64 before inverting; `nn.Linear` defaults are drawn at float32 and cast.
- **H7 polish numerics.** Parameter order = deepSI `parameters_with_names` (it walks `dir(self)`, so `encoder` then
  `hfn`); one parameter group; the closure returns `torch.as_tensor(float(loss))`, a default-dtype (float32) scalar,
  so the strong-Wolfe search compares float32-rounded values; `zero_grad(set_to_none=False)`; LBFGS lr 1,
  `max_iter = inner_iter`, history 50, the two tolerances; blocks until the cap or `done < inner_iter`.
- **H8 state across the end of fit().** `checkpoint_load_system('_best')` replaces `__dict__` with an unpickled copy:
  the simulator comes back without its compiled pair (so the polish is eager), the optimizer is the pickled one, the
  OBC lifecycle and PS2 phase are carried across by the stash; the loss history is restored after; then the basis is
  rebuilt at the restored parameters and `_best` re-saved. The clean code keeps this mechanism (no in-memory
  best-state shortcut).
- **H9 OBC numerics.** Float64 deep copy of the physics block; `jacfwd` stack in chunks of 16384, sample-major then
  physical row; column scaling; SVD rank cutoff `max(rows, cols) * eps * s_max`; coefficient `V S^-1 U^T F` in float64;
  refresh at `floor(epoch_counter + 1e-6)`; the pre-fit builds the first basis; PS2 re-syncs the frozen coefficient
  between updates; rank-loss policy (retain once, abort on two).
- **H10 least-squares encoder start.** Centred, column-scaled regressors, `gelsd`, `rcond = eps * max(N, m)`,
  records grouped by length in dict insertion order, stride-10 windows. The solution is rank-truncated (the 2 s
  smoke reported rank 141 of 180), so any change in the window set or order can move the rank: the highest-risk
  single item.
- **H11 libraries.** deepSI 0.3.29, torch 2.5.1, numpy 2.0.1, scipy 1.17.0 (local env and
  `environment.server-train.yml`). deepSI's `checkpoint_load_system` calls `torch.load(file)` without
  `weights_only`, which torch 2.6 and later default to True: the pin is part of the method.

### 6.2 Harness environment

- One conda env (H11), CPU, `device='cpu'`, `compile_mode=None`, float64 (the method). `torch.set_num_threads(1)`
  and `OMP_NUM_THREADS = MKL_NUM_THREADS = 1` in every harness process, because multi-threaded BLAS and OpenMP
  reductions may block differently from run to run, and bitwise comparison needs the same operation order.
- Mainline and the clean code run in separate processes (same package names). Mainline is imported unmodified:
  the runner sets the row's `THESIS_*` variables plus `THESIS_SMOKE=1` and `THESIS_DEVICE=cpu`, imports EP, replaces
  `EP.CFG` with the harness budget and calls `EP.main()` (the method of the step-1 trace runner; the smoke variable
  is needed for the PS2 smoke spec, training.py:873). deepSI checkpoints go to a scratch folder per run (LOCALAPPDATA
  or HOME redirect) and are kept with the golden (layer 3c resumes from them). Mainline writes its run folder into
  its own family folder (`save_dir`, `GD/config.py:603-606`; `TRUNCATE_S` is set, so `discover_runs` ignores it);
  the runner moves it into the golden folder when the run ends. The clean side gets `out_root` = the golden folder
  and the default `data_root`.
- Package roots: every clean-side process starts with `Clean/` alone on its path (no mainline root) and, for the
  evaluation, `THESIS_CODE_ROOT=<repo>/Clean` (section 5, edit (a)). At start, after its imports, and again at exit,
  every harness process asserts that each loaded `gantry_dynamic*` and `model_augmentation*` module's `__file__`
  lies under its selected root (`Clean/`, or the repository root and `scripts/gantry` for mainline), so layer 4 can
  never compare mainline with itself (X1).
- Harness budget, identical on both sides: `truncate_s = 0.6`, `ps2_smoke = True`, and for layer 3 `epochs = 3`,
  `n_its = None`, `its_per_val = 5`, `lbfgs_max_iter = 2`, `lbfgs_inner_iter = 1`, `lbfgs_windows = 256`.
  Why 0.6 s: the shortest truncation at which both noise entries have a positive servo sigma (noise-free TR-T and
  VA-T stand still until 0.495 s, A.0) and every window type is valid. It gives 198 windows per record (first window
  end 430, stride 10, 2,400 samples), 3,564 in total, so 6 Adam updates per epoch and 18 in three epochs, with OBC
  refreshes at the two epoch boundaries inside the run, validations at update 0 (the initial one), 5, 10 and 15, and three
  unvalidated updates at the end (as in production).
- Seeds: every run uses the spec seed 1 (both seeding points, EP:390-391 and 409-410, the PS2 generators and the
  polish subset follow from it); the layer 1-2 batch indices come from `default_rng(0)`; no other seed is introduced.
- Golden outputs: `<repo>/simulations/clean-golden/L<n>/<row>-<noise>/`, each with `meta.json`: SHA256 of the 19
  copied mainline files, of both `MANIFEST.csv` files, of every record file the row read (its train, val and test
  lists; the OBC reference set reads the train records again, `GD/obc_gantry.py:100`) and of the resolved
  `config.json`, plus the data root, library versions, thread settings, `git_provenance()` and the budget (X14).
  `compare.py` recomputes them and refuses a golden on any difference: identical source hashes beside a changed
  manifest or record would otherwise pass a stale golden (window order, normalisation, LS rank and OBC basis all
  follow from the data).

### 6.3 Layers

| Layer | Cases | Golden content (mainline) | Comparison | Tolerance and reason |
|-|-|-|-|-|
| 1 construction and forward pass | the 8 distinct models (221, 222, 223 = 225 for this layer, 224, 226, 227, 229, 230) x lowpass and noisefree = 16 | data per split (u, y, e, file lists), `Norm` fields, servo sigmas, loss weights; training window arrays (uh, yh, uf, yf, ctrl_ix, loss_w); controller bank matrices; OBC reference set fields; state_dict right after `build_model` (H1) with key and shape lists; after the pre-fit: state_dict, LS rank and singular values, first OBC basis (U, S, column scale, vbar), theta_frozen; then, with the mainline post-pre-fit state_dict loaded into the clean model: encoder x0 on a fixed 64-window batch (`default_rng(0).choice(N, 64, replace=False)`, sorted), `pass_mats()`, OBC coefficient and tangent pair, the 400-step closed-loop prediction, `cal_validation_error` and `last_per_channel` | `np.array_equal` / `torch.equal` per array; equal key sets and shapes | Bitwise. Same statements in the same order on the same inputs, single-threaded float64: any difference marks a changed statement. Only for a difference traced to a named cause and accepted by the judge: max abs difference over max abs reference at most 1e-12 per tensor (eps 1.1e-16 times the about 1e4 terms of the longest reduction: 400-step rollout sums, about 4e4-row OBC stack) |
| 2 loss and gradients | the same 16 cases, same batch and state | `fit_sys.loss(uh, yh, uf, yf, nf=400, stride=10, ctrl_ix=..., loss_w=...)` and, after `backward()`, the gradient of every named parameter in `parameters_with_names` order (for obc through theta, H9) | per tensor | Bitwise, same reason and the same documented 1e-12 fallback |
| 3 short seeded training | 221-lowpass (obc, detuned, n_a 2, PS2), 229-noisefree (obc, n_a 0, plain), 224-noisefree (u, detuned, 2, PS2), 226-lowpass (u, correct, 0, plain), 227-lowpass (u, correct, 8, PS2) | the run folder of `main()` with the layer-3 budget: config.json, `_pre_lbfgs` pair, final pair, val_hist.npz, the PS2 summary, `lbfgs_outcome`, the deepSI dumps | state_dicts of both pairs (hfn, encoder, and the optimizer `state` entries `step`, `exp_avg`, `exp_avg_sq`, always present in the `_pre_lbfgs` pair, in the final pair when not dropped); `.npz` scalars and strings except `elapsed` and timings; the common `lbfgs_outcome` fields except `wall_s`, `reason` included (`decide` is kept with its text, C22); val_hist arrays; config.json on the shared keys except GIT, run_id, data_dir and NF_PROBE_PRINT. Mainline has no periodic `_last.pth` save, so this comparison is also the with/without check of D-275 amendment 2: equal parameters and Adam moments show the save changed no training state | Bitwise, no fallback. Deterministic CPU with the same draws (H1 to H3); the PS2 S2 Adam (lr 1e-3, up to 100 steps) and the LS rank decision (H10) amplify any earlier difference, so a mismatch is localised (bisect with `n_its` 1, 2, 4, 8 on both sides, then layers 1-2; the first bisect step for the periodic save is the parameters after update 5, before and after the first validation), never toleranced |
| 3b polish rejection | 221-lowpass and 226-lowpass, from the golden layer-3 `_pre_lbfgs` state loaded into a freshly built model on each side | `lbfgs_polish(..., val_before=0.0)`, which forces the reject path (restore and `_resync`) | outcome fields and parameters (must equal the start state) | Bitwise; restore is a copy |
| 3c reruns (D-275, decision 4) | 226-lowpass (u) and 221-lowpass (obc), each started from its own code's layer-3 golden (D3): (i) Adam continuation from the deepSI `_best.pth`, budget of layer 3 with `epochs = 4` (with `epochs = 3` a `done_epochs` of 3 leaves 0, which `epochs=epochs or hp['epochs']`, `GD/model.py:487`, turns into a full 3-epoch rerun, C19); (ii) polish-only from the `_pre_lbfgs` pair; (iii) for 226 only, (i) from `_last.pth` (`--from last`) | mainline: `RESUME_CHECKPOINT` = a copy of the golden source, `EP.CFG` = the budget, `start_phase='lbfgs'` for (ii), new run folder | clean: `main(cfg, rerun=...)` on a copy of the clean golden run folder (section 3 "Reruns"). Compared: both pairs' state_dicts and optimizer entries, `.npz` except `elapsed`, `orig_run_id` (clean keeps the folder's id) and `resumed_from` (compared by file name), `lbfgs_outcome` as layer 3, val_hist arrays. Clean only, asserted: the folder's previous files now lie byte-identical in `previous/<rerun_id>/`, `config.json` differs only by the appended `RESUMES` entry. Both sides start (i) and (iii) with fresh Adam moments, a dump carries none (section 3); (ii) never restores them (training.py:292) | Bitwise, no fallback, as layer 3 |
| 4 final checkpoints | the 9 final rows on lowpass (no noise-free final run exists): jobs 89792 (221), 89796 (222), 89797 (223), 89793 (224), 89798 (225), 89799 (226), 89800 (227), 89794 (229), 89795 (230), from the thesis-results-4 logs, read from the staging folder of 6.5 (component pairs only, D3) | mainline evaluation with `THESIS_EVAL=final`, `THESIS_RUNS_DIR` = staging, CPU: `eval_greybox.evaluate(run, only=..., split=...)` called in-process with one thread (the script's `__main__` sets 4 threads, `eval_greybox.py:212`) for R1 R2 R3 R4 R5 R6 R7 R13 R14 on `train, val, test`, rows `model` for all and `physics, base_start, base_true` for R1, R4, R6, R13 (covers `_without_learned`, `_true_free`); the cache folder `Results/thesis-eval/final-<staging name>/` is moved into the golden after the run | the same calls on the clean code: `THESIS_CODE_ROOT=<repo>/Clean`, section 5 edits (a) and (b), `run_config` through `from_config_json` and its mainline adapter (D2), which also checks the constant list of section 3 against each of the nine `config.json`; compared per record from k0: `y`, `sigma_e` and every system's prediction; derived RMS and NRMS_e per split, record and axis; `params_learned` | Bitwise against mainline (same weights, same eager scoring code). The harness asserts, from each side's `preds_<R>_val.npz`, that `ref_score` is finite and positive, `val_score` is finite and `abs(val_score - ref_score) / ref_score < 1e-3`: the evaluation's own check only prints OK or FAILED (`eval_greybox.py:196-199`, X12); the 1e-3 bar is the existing acceptance for CPU eager against GPU compiled, reused, not introduced |
| 4b evaluation interfaces | the staged checkpoints of R1 and R13 (obc), R4 and R6 (u), plus R7 for the poles (n_a 8) | in-process on mainline: `eval_overlap.evaluate(run)` (returns the overlap and bias rows, no CSV is written) for R1, R4, R13, which calls `deployed_basis` with `space=corr.space` on the OBC runs; `eval_negation.evaluate(run)` (returns the rows) for R1, R4; `inspect_checkpoint.added_poles(fs, data, cfg.nx_phys, cfg.nx_ann)` (as `eval_overlap.py:143`) for R4 and R7; the frozen-Y baseline `frozen_y(_without_learned(fs, _true_free(phy)))` rolled out with `closed_loop_free_run_rms_batch(..., return_pred=True)` on the val records of R4 (the `--lti` path, `eval_greybox.py:140-142`); the `deployed_basis` matrix of R1 on the training stack | the same calls on the clean code (X11) | Bitwise; same weights, same float64 code. These are verification calls on the evaluation's functions, not diagnostics inside `Clean/` |

Rows chosen so that every axis value occurs in layer 3: arm obc (221, 229) and u (224, 226, 227); n_a 0 (229, 226)
and > 0 (221, 224, 227); PS2 on (221, 224, 227) and off (229, 226); lowpass (221, 226, 227) and noisefree (229,
224); detuned (221, 229, 224) and correct (226, 227).

**Cost.** No harness timing exists yet. The only measured reference is the step-1 trace (`inventory-claude.md` A.0):
218 to 447 s per u row and 826 to 1249 s per obc row at 0.3 to 0.6 s truncation and 4 updates, post-run
diagnostics included. Layer 3 has 18 updates, so obc rows will take longer; the first case of each layer is timed
(and its memory measured) before the rest are launched, and all runs go to the background with streamed output.
Layer 4 is best run where the checkpoints are (server CPU), at about 4 min per evaluation run
(`Thesis-writeup/Code/Figures/README.md`). Layer 3c is five reruns per side: (i) and (iii) train `4 - done_epochs` epochs of 6 updates (`done_epochs` =
`round(epoch_counter)` of the dump, 0 to 3) plus the polish, (ii) the polish only; layer 4b adds the overlap, negation and pole calls on
four staged runs, of the order of one evaluation run each (no measurement yet: the first case is timed first).

### 6.4 Not covered by the layers

GPU and compiled paths (compiled rollout, `cudagraph_mark_step_begin`, device moves, the compiled try and eager
fallback of a polish-only start, the CUDA OOM path of the polish), the OBC rank-loss branches, the polish exception
path, the NaN break, and the PS2 plateau and fallback with the real thresholds. Covered instead by:
(a) A compiled GPU smoke on the server, R1 and R6 lowpass, both codes on the same node (C4, X10). The `--smoke`
flag and mainline's `THESIS_SMOKE` set `compile_mode=None` (EP:371), so they never reach the production path; the
harness instead builds the smoke budget with compilation on: `truncate_s=2.0, n_its=4, its_per_val=4,
lbfgs_inner_iter=1, lbfgs_max_iter=2, lbfgs_windows=256, ps2_smoke=True, device='cuda',
compile_mode='reduce-overhead'` (mainline: the same `EP.CFG` replacement as 6.2, not `THESIS_SMOKE`'s `replace`;
`THESIS_SMOKE=1` stays set only for the PS2 smoke spec, training.py:873). Each run must finish, log "training
rollout COMPILED", log no "skipping cudagraphs", and write both pairs; the harness asserts in the run's own process
that the compiled branch RAN, not only that the wrappers were built: `torch._dynamo.utils.counters['stats']
['unique_graphs'] > 0` after `fit()`. Next to each other, the initial validation score and the four training
losses must agree within 1e-9 relative (identical graphs on identical inputs; CUDA kernels with atomic
accumulation may reorder sums). Budget: the one-off CUDA-graph recording at nf = 400 took 55 to 61 min in
production (EP:106-112), so about one hour per run; the four runs are one GPU job each.
(b) A compiled polish-only smoke from (a)'s `_pre_lbfgs` pair, R6 on both codes (the path of the four optional
D-274 polish reruns): `fit()` never runs, so the simulator keeps its compiled pair and the polish takes the
compiled try (LB:657-671) with its eager fallback and second probe (LB:673-698). Both must report the same
`deterministic`, `used_compiled` and, within 1e-9 relative, `loss_first`.
(c) Review: every one of these branches is copied verbatim (section 2).

### 6.5 Inputs from the server

- The nine final run folders under `simulations/gantry_subnet/thesis_taf_lowpass_noise_data_lsq/<job>/` (config.json,
  both checkpoint pairs, `gantry_results_<job>.npz` for the evaluation's report check). They are not local.
- The `[lbfgs]` lines of 221 and 222 (the local logs end before the polish). They decide only which saved file is
  the layer-4 reference for those two rows, under D-274 change 3 as amended: polish ACCEPTED with val_after below
  the selected checkpoint's `bestfit`: `gantry_ckpt_<job>` (same under both rules); ACCEPTED with val_after at or
  above `bestfit` but below the rebuilt score (accepted only under the D-253 rule): the D-274 final model is the
  rebuilt one, saved as `gantry_ckpt_<job>_pre_lbfgs`, and that row's thesis numbers move by about 2e-6 relative;
  ROLLED BACK or ABORTED: `gantry_ckpt_<job>`. Known from the local logs: 229 accepted (2.925875e-01 below the
  checkpoint's 2.926736e-01, D-274), 224 and 227 rolled back, 223, 225, 226 and 230 aborted by CUDA OOM in the eager
  determinism probe with weights unchanged.

**Layer-4 staging folder** (D-275 D6 and its amendment; X13). Layer 4 never reads the server folders themselves:
`THESIS_RUNS_DIR = <repo>/simulations/clean-golden/L4-staging/` with one `<job>/` folder per row (`discover_runs`
needs only the `config.json` files). The folder name must not be the server family name: the evaluation's cache is
`Thesis-writeup/Results/thesis-eval/final-<RUNS_DIR name>/` (`eval_common.py:87`), so the staging runs write
`final-L4-staging/` and can never overwrite the thesis caches in `final-thesis_taf_lowpass_noise_data_lsq/`; each
side's cache folder is moved into the golden after its run (6.3). It is built once, before any rerun touches a
server folder (a polish rerun moves the originals into `previous/`, section 3), as copies:
- Rows whose final model D-274 does not change (every row except a 221 or 222 in the middle case above): the
  folder copied as is (`config.json`, both pairs, `val_hist.npz`, `gantry_results_<job>.npz`).
- A 221 or 222 in the middle case: `gantry_ckpt_<job>.pt` = a copy of `gantry_ckpt_<job>_pre_lbfgs.pt` (the rebuilt
  model); `gantry_ckpt_<job>.npz` = the original `.npz` with `bestfit` replaced by that model's score, measured by
  the MAINLINE evaluation of the staged weights (`eval_greybox` on `val`, its `val_score`), because the `_pre_lbfgs`
  `.npz` holds the selected score, not the rebuilt one (`GD/training.py:935-937`), and with `lbfgs_outcome` marked
  rejected by D-274; `gantry_results_<job>.npz` is left out, since it scores the polished model and
  `eval_greybox.py:178-179` then skips the report check. The row is an intended deviation (D-274), recorded in the
  staging folder's `meta.json`; clean against mainline equivalence is still bitwise on these staged weights.
- The original server files stay untouched.

## 7. Open choices

Each lists the choice taken, the alternatives, and the reason.

1. **Folder.** Decided by Dirk (2026-10-20): `<repo>/Clean/`, a top-level folder; `REPO_ROOT = ROOT/..`; data and
   output roots are arguments with defaults (section 3), so no path depends on where the folder sits. Earlier
   alternatives: `Thesis-writeup/Code/Training/`, `scripts/gantry/clean/`.
2. **deepSI.** Taken: installed dependency, pinned 0.3.29. Alternative: vendor the used subset (System_data and
   System_data_list, the windowing, `SS_encoder_general` init and `make_training_data`, `System_torch`
   `parameters_with_names`, `init_optimizer`, `_check_and_refresh_optimizer_if_needed`, checkpoint save and load,
   `My_Simple_DataLoader` with its iterator, `loop_dataset`; a few hundred lines). Reason: H3 and H8 live in deepSI,
   the evaluation and the black box import deepSI anyway; vendoring can follow later with this harness as its test.
3. **How far "minimal" goes inside Jan's classes.** Taken: keep Jan's classes, names and files; delete unused
   methods and option branches; copy small helpers whole; keep `SSE_Interconnect_Composed` as our subclass.
   Alternatives: merge `SSE_Interconnect` and `SSE_Interconnect_Composed` into one class and trim the helpers too
   (fewer lines, blurs the Jan/ours markers, no 1:1 diff); keep Jan's classes intact (most lines, least risk).
   Reason: markers keep meaning and every file diffs against its mainline original.
4. **Launch.** Decided by Dirk (2026-10-20): CLI with a default, `[preset] [noise] [--cpu] [--smoke] [--resume
   RUN_DIR [--from {best,last}] | --polish-only RUN_DIR] [--data-root P] [--out-root P]`, no arguments = R1 lowpass
   on cuda (section 3). Optional positionals plus flags, because optional words beside defaulted positionals would
   be parsed as a preset (C21). Alternative: `THESIS_*` environment variables as now (the run is then not visible in
   the command line, and the launcher's export-all setting can leak variables, Codex C3 point 16).
5. **Preset names.** Taken: R labels (R1 to R7, R13, R14). Alternatives: runs.tsv ids 221 to 230; both as aliases.
   Reason: the thesis and the evaluation use the R labels; the id is in the preset table.
6. **RunConfig fields.** Taken: five axes plus eleven run-environment fields (with `start_phase`) and the two
   roots, all with production defaults. Alternatives:
   axes plus one `smoke` flag (the harness could not set 18 updates and 4 validations without patching); a separate
   budget object (the evaluation's flat field-name round trip and the flat config.json prefer flat fields).
7. **config.json.** Taken: every mainline key with the run's value, plus ARM, START, N_A, NOISE, PS2_SMOKE,
   DATA_ROOT, OUT_ROOT, LAST_SAVE, CODE (and RESUMES after a rerun).
   Alternative: only the keys the evaluation reads (about fifteen). Reason: complete provenance; the layer-3 check is
   a plain equality on shared keys.
8. **Evaluation round trip.** Decided (D-275 D2): `RunConfig.from_config_json(meta, **overrides)` in the clean
   config, called by `eval_greybox.run_config` (edit (b)) and by the reruns; it reads clean files and, through one
   adapter, mainline files, and rejects any setting outside the spec (section 3). Needed because layer 4 evaluates
   the mainline final runs, whose `config.json` has no `ARM, START, N_A, NOISE` (C1, X6). Earlier alternatives:
   drop two keywords from `run_config` (fails on every mainline file); inert `nf_override`, `na_nb_override` fields.
9. **Validation history.** Taken: `ValHistory` (record, hist_arrays, the 'adam' probe) and `val_hist.npz`.
   Alternative: drop it and compute the final `bestfit` from three scalars (selected score, rebuilt score, polish
   val_after). Reason: about 30 lines; the final event follows the same rows as mainline; it is the only record of
   the validation curve now that the run report is excluded.
10. **Interface helpers.** Taken: `gantry_dynamic/closed_loop_report.py` with `_without_learned` and `_true_free`
    only. Alternative: move them into `model.py` and change three evaluation imports. Reason: zero import changes.
11. **Truth fields.** Taken: not read (data loader and OBC reference set). Alternative: keep reading them for later
    diagnostics. Reason: no method use (D-255), less I/O, less code.
12. **Resume and polish-only.** Deleted: Dirk decided on 2026-10-20 to keep both (section 3 "Reruns", D-275 D6
    and its amendments; the restored rows of section 2).
13. **Guards.** Taken: rule G (section 0), applied consistently (D-275 D5). Alternative: keep every mainline guard
    (more lines, no numeric difference). Reason: a guard that re-checks the code's own constants cannot fire; the
    data, numerics, environment and external-argument guards stay.
14. **Per-call device self-heals** (`Interconnect.forward` 137-141, `output_only` 330-333,
    `Static_ANN_Block.forward` 76-77). Taken: removed, the stated exception to rule 2 (D-275 D5);
    `Interconnect._apply` and the buffer mechanism keep devices and dtypes in sync. Alternative: keep them (about 12
    defensive lines). Covered by the compiled GPU smoke (6.4).
15. **Output root.** Taken: default `out_root = <repo>/simulations/gantry_clean/`, overridable with `--out-root`
    (Dirk 2026-10-20). Alternatives: the mainline family folders (the evaluation's `discover_runs` raises on two runs
    of one design); inside the folder (mixes outputs into code). A rerun writes into its own run folder (D6).
16. **Polish simplification.** Deleted: with polish-only kept (decision 4), the compiled try, its eager fallback
    and the second eager probe are copied (LB:657-698, the `lbfgs_polish` row); the warm-up, chunk loop, freeze and
    meters still go, each with its reason in section 2.
17. **param_loss.** Taken: removed with the loss sweep (it adds the Python float 0.0, which leaves the loss and its
    gradient bit for bit unchanged; `Lambda_combo` stays a buffer, a state_dict key). Alternative: keep both (about 8
    lines).
18. **Harness tolerances.** Taken: bitwise for every layer, with a documented 1e-12 fallback for layers 1-2 only.
    Alternative: relative tolerances everywhere. Reason: a tolerance would let a changed statement pass unnoticed.
19. **Harness location.** Taken: `Thesis-writeup/Code/Verification/clean-code/`, golden outputs in
    `simulations/clean-golden/`. Alternative: `tasks/clean-code/harness/`. Reason: Verification exists (empty) for
    exactly this; `tasks/` holds documents.
20. **NF_PROBE_PRINT.** Taken: written `False`. Alternative: the mainline value `True` (would claim prints that
    do not happen).
21. **GantryReferenceSet fields.** Taken: six fields kept (training and evaluation readers). Alternative: keep all
    twelve so the class matches mainline field for field. Reason: the other six hold truth or bookkeeping that no
    kept code and no evaluation script reads.
