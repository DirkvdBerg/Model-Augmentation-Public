# Merged critique of `design.md` (step 3), 2026-10-20

Inputs: `critique-claude.md` (C1 to C36) and `critique-codex.md` (X1 to X14). Both critics agree: no trim checked on
the CPU float64 path changes a training number, and the RNG order of design 6.1 holds. The problems are construction
errors (copied code that still uses dropped names), an evaluation and harness plan that cannot run as written, and
Dirk's four decisions not yet carried through. Two claims re-checked in the code by the main session: X1
(`inspect_checkpoint.py:30-31` puts the mainline root on `sys.path` at import; `eval_greybox.py:105` and
`eval_overlap.py:141` import it) and the optimizer property of C3 (`interconnect.py:1287, 1521-1524, 1605-1627`; the
pickling detail is not fully re-checked, see D1).

## A. Decisions for Dirk

| # | Question | Source | Recommendation |
|-|-|-|-|
| D1 | `SSE_Interconnect_Composed.optimizer` property: keep it, or make it a plain attribute? | C3 | Keep the 4-line property: resume then restores Adam moments exactly as mainline does, and the resume test compares like with like |
| D2 | How does the clean config read a mainline `config.json` (no ARM, START, N_A, NOISE keys)? | C1, X6 | One adapter in the clean config: arm from OBC, start from COMBO_INIT_DETUNE, n_a from NX_ANN, noise from MODE; it rejects any other setting. Needed for layer 4 |
| D3 | Can one code load the other's whole `_best.pth`? | C3, X7 | No, by design: each code resumes only its own `_best.pth`; layer 4 and the evaluation use the component pair (`.pt`/`.npz`), which needs no pickled classes; preliminary evaluation of mainline runs stays on mainline |
| D4 | Markers on Jan functions that are only trimmed | C32 | One `# CHANGED (clean): trimmed to the spec path` line per trimmed Jan definition (CLAUDE.md asks for a marker on every changed block) |
| D5 | Guard rule G applied inconsistently (some self-built re-checks kept) | C33, C34, C35 | Apply rule G consistently: drop re-checks of values the clean code builds itself; keep data, solver and environment guards; state the out_gate/self-heal exception in rule 2 |
| D6 | Where a continuation or polish-only rerun writes | C20, X13 | A run folder is one model: `--resume <run_dir>` / `--polish-only <run_dir>` write into that folder under the same run id; the previous outcome moves to `<run_dir>/previous/<rerun_id>/`; `config.json` gets a `RESUMES` list (time, type, source checkpoint, code); the evaluation needs no change. Layer-4 references for re-decided rows (D-274) are built in a separate golden staging folder, original server files untouched (D-275 amendment) |

## B. Fixes to apply (no decision needed)

**Construction errors (literal copy would fail at import or first build):**
1. `debugging` attribute: keep it, or list all eleven reads and the `model.py:79` argument as removed (C5, X4).
2. `zero_init_feed_forward_nn` is the default `net` of `Static_ANN_Block` (`blocks.py:8, 36`): make the live net the default or required (C6, X2).
3. `e_net`, `e_net_kwargs`, `modified_encoder_net` default: remove the arguments at `model.py:197-200` too (C7, X2).
4. `checkpoint_chunk`: list both call-site edits (`controller.py:355`, `closed_loop.py:598`) (C8).
5. Encoder call: `wa_encoder_init` must stay `'xavier_uniform_gain1'` (dropping the keyword silently selects Kaiming and changes every result); list the call edit, `ann_activation`, the collapsed headers at `model.py:81, 96, 138` and `pre_encoder.py:365, 443` (C9, X5).
6. `use_f64` (and `lbfgs`) missing from the constants (C24, X5).
7. `RankReport`: keep it whole, or trim the `from_matrix` call and keep the Gram diagonal and column scales exact (C10, X3).
8. `OBCLifecycle`: keep the counters and timing that `refresh` still writes, keep `last_refresh_epoch`; list the `verbose` argument at `obc_gantry.py:397` (C11, C12).
9. Polish trim: keep `LB:626` (`params` with the `requires_grad` filter) and `LB:630`; remove the other listed uses; trim `training.py:773-775` (C13).
10. `COMPILE_BACKEND` import in `controller.py:332` (C14).
11. Dead imports: rule 1 gets "imports follow the kept names"; list `interconnect.py:18`, `model.py:11, 17, 18`, `obc_gantry.py:38, 42`, `training.py:20-21` (C spot-check, X2).
12. Small statement-level fixes: keep `hp['na_nb']` (C26), one `na_nb` definition (C27), the `int` form of `nf` (C28), `y_decimation` on `cfg.noise` (C29), `float()` in the detuning (C30), progress line marked as a rewrite (C31), `config.json` constants taken from job 89792 (C25), the line-range corrections of C section 3.

**Evaluation and harness plan:**
13. Keep the `space` keyword of `make_basis_builder` (default `'affine'`) and `GantryOBCCorrection.space`; without them overlap and negation fail or silently use the tangent basis (C2, X11).
14. `inspect_checkpoint.py` puts the mainline root on `sys.path` at import: layer 4 could compare mainline with itself. One package-root selection for the evaluation, plus an assert at harness start that `gantry_dynamic` and `model_augmentation` both resolve under `Clean/` (X1, with C23's guard in `Clean/gantry_dynamic/__init__.py`).
15. Add small interface checks on loaded checkpoints: `deployed_basis`, overlap and bias, negation, added-state poles, frozen-Y prediction (X11).
16. The harness asserts the validation-score comparison itself; `eval_greybox.py:196-199` only prints OK or FAILED (X12).
17. Layer-3 comparison excludes the `reason` string or copies it exactly (C22).
18. Golden fingerprints include both manifests and the record files, not only source hashes (X14).
19. A compiled GPU smoke: the `smoke` word turns compilation off, so the compiled production path is never tested; add a compiled short run on both codes and assert the compiled branch ran (C4, X10).

**Dirk's four decisions, carried through:**
20. Folder `<repo>/Clean/`: tree, `REPO_ROOT = ROOT/..`, server command, evaluation path, open choice 1 (C section 2, X9).
21. Roots as arguments: `data_root`, `out_root` as config fields with the two defaults, used by `traj_dir` and `save_dir`, recorded in `config.json` so a local evaluation can override a server path (C17, X9).
22. CLI with default: `[preset] [noise] [--cpu] [--smoke] [--resume BASE] [--polish-only] [--data-root P] [--out-root P]`, default R1 lowpass on cuda, stated in the usage line (C21, C36, X decision 3).
23. Resume and polish-only kept: restore the full list of C18 (resume input, `resolve_resume_hp`, prefit gate, `start_phase` as a field, moment restore, PS2 gate, Adam skip, epoch rule, provenance fields, `val_before` rule) and the compiled try with eager fallback and second probe (C15, C16, X8); delete open choices 12 and 16; add layer 3c (Adam continuation and polish-only start from a `_pre_lbfgs` pair, u and obc rows, both codes, bitwise) (C19, X8).

## C. Next

Dirk accepted the recommendations for D1 to D6 (2026-10-20, D-275). The main session then writes the revision brief for the step-2 session (or a fresh one) to
amend `design.md` with A and B; after that, step 4 (implementation).
