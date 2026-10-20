# Critique of `tasks/clean-code/design.md` (step 3 of 4, Claude)

2026-10-20. Handoff `tasks/handoffs/2026-10-20-clean-code-critique-claude.md`. Every claim was checked against the
mainline working tree of 2026-10-20 (branch `Augmentation`, D-274 edits in place), not against the design's own
reasoning. The Codex critique was not read. Code quotes were checked with `scripts/verify_quote.py` (MATCH OK). The
mapping spot-check of design section 2 ran in one delegated agent (handoff section 14); its high findings were
re-read in the code before they went in here, its line-range findings (section 3) were not.

Format: `design.md:<line> | what the design says (mainline file:line) | consequence | fix | severity`.
Abbreviations as in the design (EP, GD, MA), plus IC = `MA/fit_systems/interconnect.py`, CL =
`MA/fit_systems/closed_loop.py`, LB = `MA/fit_systems/lbfgs_polish.py`, OBC = `MA/fit_systems/obc.py`,
EV = `Thesis-writeup/Code/Evaluation/`.

No finding changes a training number on the spec path in silence: every trim I checked on the CPU float64 path is
exact (section 4). The serious problems are an equivalence plan that cannot run as written, a broken evaluation call,
a resume that does not match mainline, and a list of loud construction errors.

## 1. Findings, ranked by consequence

### High: silent, or the plan cannot run

1. `design.md:872` (layer 4), `design.md:631-635`, `design.md:743-745` | layer 4 runs the evaluation on the clean code
   against the nine MAINLINE final checkpoints; the clean `RunConfig` has four required fields `arm, start, n_a, noise`
   without defaults, and `EV/eval_greybox.py:43` builds it as
   `opts = {f.name: meta[f.name.upper()] for f in fields(RunConfig) if f.name.upper() in meta}` | a mainline
   `config.json` (jobs 89792 to 89800) has no `ARM`, `START`, `N_A`, `NOISE` (`GD/config.py:668-773`), so
   `RunConfig(**opts)` raises `TypeError` on every row: layer 4, the only test of the evaluation interface and the final
   numbers, cannot run. It also needs evaluation edits (a) and (b) (`design.md:783-788`), which the same section says are
   "not made now" | give the clean config one constructor from a mainline `config.json` (`OBC` to arm,
   `COMBO_INIT_DETUNE is not None` to start, `NX_ANN` to n_a, `MODE` to noise, `PS2`), i.e. alternative 3 of open choice
   8, and say who makes edits (a) and (b) for the harness (a harness-local copy of `eval_common` and `run_config` is
   enough) | high

2. `design.md:267`, `design.md:260-261` against `design.md:741`, `design.md:772` | section 2 fixes `make_basis_builder`
   to affine with signature `(blk64, ref, row_ix, chunk, device=)`; section 5 keeps `space=` (`GD/obc_gantry.py:319-321`,
   default `space: str = 'tangent'`) | `EV/eval_overlap.py:51-53` does `space = getattr(corr, 'space', 'tangent')` and
   passes `space=space`; with the section 2 signature `eval_overlap` and `eval_negation` (which imports
   `deployed_basis`) raise `TypeError` on R1, R2, R13. If `GantryOBCCorrection.space` is also dropped under "affine space
   only", the `getattr` returns `'tangent'` and the overlap and negation numbers use the tangent basis `[J]` instead of the
   deployed `[J f_bar]`, with no error (spot-check agrees) | keep the `space` keyword (default `'affine'`, raise for
   anything else) and keep `self.space`; make sections 2 and 5 agree | high

3. `design.md:448`, `design.md:756` | the `SSE_Interconnect_Composed.optimizer` property is dropped and `optimizer`
   becomes a plain attribute (IC:1626-1627 store `self._optimizer = opt`; `GD/training.py:115` reads
   `_opt = d.get('optimizer')` from a deepSI `_best.pth`) | deepSI pickles `self.__dict__` (`deepSI/fit_systems/fit_system.py:496`):
   a mainline `_best.pth` holds `_optimizer`, so a mainline resume from it starts Adam with fresh moments, while a clean
   `_best.pth` holds `optimizer` and a clean resume restores them. Decision 4's continuation of a stopped run can only
   start from `_best.pth` (the pair is written after the polish), so every resumed Adam update differs from mainline and
   the required resume test fails. The two codes also cannot read each other's `_best.pth`: a mainline dump pickles
   `_NfProbe` and `WindowErrorStats` (instance attributes `validation_probes`, `loss_stats`, `GD/training.py:375, 678`),
   both dropped; a clean dump pickles `ValHistory`, which mainline lacks; so `design.md:756` holds only within one code
   base (spot-check agrees) | either keep the 4-line property (equivalence first, mainline is the oracle) or keep the plain
   attribute and record the moment restore as an intended deviation (the D-173 intent) that the resume test excludes;
   Dirk decides. State that each code resumes and evaluates only its own `_best.pth` | high

4. `design.md:889-893` (6.4 a) against `design.md:689-691` | the GPU smoke uses the word `smoke` and must log "training
   rollout COMPILED" and no "skipping cudagraphs"; `smoke` applies EP:371
   (`CFG = replace(CFG, truncate_s=2.0, compile_mode=None, n_its=4, its_per_val=4,`) | the smoke never compiles: the only
   check of the production path (compiled rollout, `cudagraph_mark_step_begin`, output clones) passes without running
   it, the mainline reference smoke is eager too, and finding 14 (a compile-only `ImportError`) would surface only at the
   first production launch | add a compiled smoke (e.g. `--smoke` keeps `compile_mode` on cuda, or `--smoke-compiled`);
   build the mainline reference through the harness `EP.CFG` replacement, not `THESIS_SMOKE`; budget the one-off
   CUDA-graph recording at nf = 400 (about 55 min, EP:106-112) | high

### High: loud, every run or evaluation build fails at construction (layer 1 would catch each)

5. `design.md:400`, `design.md:405`, `design.md:408-410`, `design.md:438` | `Interconnect.__init__` drops `debugging`;
   `init_forward`, `add_block`, `connect_signals` are "copy"; `convert_signal_ix_to_name` is dropped | eleven kept
   statements read `self.debugging` (IC:224, 248, 255, 265, 355, 375, 383, 432, 439, 455, 470), so the first
   `ic.add_block` raises `AttributeError`; `GD/model.py:79` still passes `debugging=False`, a `TypeError` once the
   parameter goes | keep `self.debugging = debugging` and `convert_signal_ix_to_name`, or list all eleven statements and
   model.py:79 as removed | high

6. `design.md:311`, `design.md:584` | `Static_ANN_Block.__init__` "copy" while `zero_init_feed_forward_nn` is dropped |
   `blocks.py:8` imports it and `blocks.py:36` uses it as the default `net`: `ImportError` on importing `blocks`, so
   training and every evaluation script fail | keep the function, or list the import and default edit | high

7. `design.md:421`, `design.md:242` | `SSE_Interconnect.__init__` loses `e_net`, `e_net_kwargs`; the `build_model` call is
   kept | `GD/model.py:197-200` passes `e_net_kwargs={...}` and the composed `__init__` forwards `**kwargs` (IC:1305-1306):
   `TypeError` in every build | add model.py:197-200 to the `build_model` removals | high

8. `design.md:380`, `design.md:203`, `design.md:373` | `ClosedLoopSimulator.__init__` drops `checkpoint_chunk` and
   `closed_loop_rollout` its `chunk` argument; section 3 defines no `checkpoint_chunk` | `GD/controller.py:355` passes
   `checkpoint_chunk=cfg.checkpoint_chunk` and CL:598 passes `chunk=self.checkpoint_chunk`: `AttributeError` in
   `build_closed_loop` (every run, the evaluation build) | list both call-site edits | high

9. `design.md:521`, `design.md:242`, `design.md:649-656` | `linear_encoder_init_aug.__init__` drops `A, B, C, D,
   flag_linear_only`; section 3 has no `ann_activation`, `wa_encoder_init` | the kept encoder call `GD/model.py:251-261`
   passes `A=None ... flag_linear_only=False, wa_encoder_init=cfg.wa_encoder_init`; `GD/model.py:136` reads
   `cfg.ann_activation`; the condition headers model.py:81, 96, 138 (`joint_estimation`, `physics_parameterization`,
   `g_init`) are not listed; `pre_encoder.py:365, 443` still read `flag_linear_only` | list the call edit, `_act =
   torch.nn.Tanh`, the collapsed headers in the `build_model` row, and pre_encoder 365/443 in the encoder row | high

10. `design.md:490` against `design.md:492`, `design.md:494` | `RankReport` keeps rank, tol, n_cols, cond, singular
    values | `OBCBasis.__init__` (copy) reads `report.n_rows` (OBC:178) and `from_matrix` (copy) passes `n_rows, rtol,
    col_norms, col_corr, build_seconds, svd_seconds, extra` (OBC:222-233): `TypeError` at the first basis build (the
    pre-fit), so R1, R2, R13 cannot start | keep `RankReport` whole, or trim the `from_matrix` call too | high

11. `design.md:501-502` | `OBCLifecycle.__init__` drops counters and timing; the `refresh` trim names basis0, r_J,
    the history entry and prints | `refresh` keeps `self.n_rank_loss += 1`, `self.n_refresh += 1` and the timing
    append (OBC:461, 494, 496-497): `AttributeError` at the end of the first refresh | add those lines to the `refresh`
    removals; keep `last_refresh_epoch` (OBC:495, read by `maybe_refresh`) | high

12. `design.md:271`, `design.md:501` | `OBCLifecycle` loses `verbose`; `attach_obc` keeps passing `verbose=verbose`
    (`GD/obc_gantry.py:397`) | `TypeError` in every OBC build, training and evaluation | list obc_gantry.py:397 | high

13. `design.md:473`, `design.md:456`, `design.md:478-482`, `design.md:293` | "freeze checks (621-630)" removed;
    `PolishSpec` loses freeze, meter_rel_tol, time_budget_s, chunk, meters_gate | the range holds `params = [...]` (LB:626,
    with the `p.requires_grad` filter) and `out['params_polished']` (LB:630, a section 5 key); kept lines still use dropped
    names: LB:583 `spec.freeze`, LB:614 and 774 `_all_meters()`, LB:632 `_quiet_probes`/`_frozen`, LB:634 `spec.chunk`,
    LB:641 `warm_up_lazy_init`; `run_lbfgs_polish` still passes the five dropped fields (`GD/training.py:773-775`). The
    polish then fails in every run after the whole Adam phase, and the final pair is never written | remove exactly
    LB:622-625, 627-629 and the listed uses, keep 626 and 630, trim training.py:773-775 | high

### Medium

14. `design.md:249`, `design.md:203` | `COMPILE_BACKEND` moves to `GD/controller.py` | the kept compile block still does
    `from .model import COMPILE_BACKEND` (`GD/controller.py:332`): `ImportError` in `build_closed_loop` whenever
    `compile_mode` is set, i.e. at the start of every production GPU run; neither the CPU harness nor the eager smoke
    reaches the line (finding 4) | replace the import by the local constant in the row | medium

15. `design.md:473`, `design.md:479`, `design.md:390`, `design.md:959-961` (open choice 16, decision 4) | no `_eager`,
    no `ClosedLoopSimulator.eager`, no compiled try, "exact or unreachable after fit" (LB:347-369, LB:657-698, CL:540-555)
    | decision 4 brings back the polish-only start, where `fit()` never runs and the simulator keeps its compiled pair
    (`GD/controller.py:329-341`). The clean probe then runs the COMPILED rollout (the simulator routes on
    `_compiled is not None and is_cuda and grad enabled`, CL:584), with no eager fallback: a compile error or a
    non-reproducible compiled loss aborts the clean polish where mainline retries eagerly (LB:673-686). The compiled
    branch is also reachable after `fit()` when no `_best` exists (IC:1140-1142). Device-decided branches, which rule 2
    (`design.md:33-36`) keeps. (The spot-check reads this as "clean polishes eager"; it does not, the routing is in the
    kept `__call__`.) The warm-up itself is not needed: on a polish-only start `val_before` is None, so `_validate()`
    (LB:613) runs an eager forward that completes the lazy init before any probe | restore LB:657-698 with `_eager`,
    `ClosedLoopSimulator.eager` and `used_compiled`; delete open choice 16 | medium

16. `design.md:473` | "reduced to one eager probe" | after `fit()` (no compiled pair) mainline probes eagerly twice when
    the first probe fails: `if not out['deterministic']:` re-creates the optimizer and probes again (LB:681-685) before
    aborting. One probe changes the GPU failure path that 223, 225, 226 and 230 took (spot-check agrees) | copy
    LB:681-698 as is | medium

17. `design.md:704`, `design.md:61`, `design.md:103-104` (decision 2) | data root `ROOT/../../Data` and `OUT_ROOT` as
    module constants (`GD/data.py:143-146` `traj_dir(cfg)`, `GD/obc_gantry.py:100`, `GD/config.py:603-606`) | as
    arguments the roots must travel on `cfg`, because `traj_dir`, `load_datasets`, `_load_record`,
    `_load_record_float64` and `save_dir` take only `cfg` and the evaluation calls them with a cfg rebuilt from
    `config.json`. A `RunConfig` field is then restored by `run_config` from `config.json`: the server's absolute data path
    on the PC, and the local evaluation fails at the first `loadmat` | `data_root`, `out_root` fields with defaults,
    recorded in `config.json` under keys that are not the upper-cased field names (e.g. `DATA_DIR` as now), or overridden
    in evaluation edit (b); drop the "fixed relative path" reasoning of sections 1, 4 and open choice 1 | medium

18. `design.md:147`, `design.md:294`, `design.md:296-300`, `design.md:286`, `design.md:654`, `design.md:693-695`,
    `design.md:947-950` (decision 4) | resume dropped | the resume path needs, from mainline: a resume input (add
    `--resume BASE`, `--polish-only`; the design reads no `RESUME_CHECKPOINT`), `resolve_resume_hp`
    (`GD/training.py:1020-1084`), the pre-fit gate EP:444-445
    (`if cfg.encoder_init == 'data_lsq' and resume_ckpt is None:`), `start_phase` as an instance field (now a ClassVar),
    a `joint_estimation = True` ClassVar (read at `GD/training.py:853`), `load_optimizer_moments` and the moment restore
    of `load_checkpoint` (`GD/training.py:292-315`), and in `train` the resume block (852-864), the PS2 gate
    `done_epochs == 0 and start_phase == 'adam'` (871), the Adam skip (886-890), the epoch rule (907-908),
    `_no_adam_ran` (976-977), the real `.npz` values of `start_phase`, `resumed_from`, `resumed_at_epoch` (1005-1007), and
    `val_before = (bestfit if (np.isfinite(bestfit) and cfg.start_phase != 'lbfgs') else None)` (`GD/training.py:788`)
    | restore these as copies | medium

19. `design.md:864-876` (decision 4) | no resume test | Dirk's decision requires one against mainline resume | add a
    layer 3c, CPU, bitwise: from the golden layer-3 `_pre_lbfgs` pairs of 226-lowpass (u) and 221-lowpass (obc), (i) an
    Adam continuation and (ii) a `start_phase='lbfgs'` polish, both sides, comparing both pairs, the `.npz` and
    `val_hist.npz`. Set `epochs = 4` for (i): the pair records `done_epochs = 3`, and with `epochs = 3` the remaining 0
    becomes a full 3-epoch rerun through `epochs=epochs or hp['epochs']` (`GD/model.py:487`). A continuation from each
    side's own `_best.pth` only after finding 3 is decided | medium

20. `design.md:104-108` (open choice 15, decision 4) | one run folder per run id | a continuation or polish-only rerun
    gets a new run id and a `config.json` that `discover_runs` counts as a full run (`EV/eval_common.py:50-59`: no
    `TRUNCATE_S`, no `N_ITS`, digit id) beside the source run of the same design, so `discover_runs` raises "two full runs"
    at `eval_common` import and every evaluation script stops. The four D-274 reruns start from mainline `_pre_lbfgs`
    pairs in the mainline family folder while their results land in `simulations/gantry_clean/`, and the evaluation
    reads one `RUNS_DIR` | state where a continuation and a polish-only rerun write, and how the four reruns enter the
    evaluated set | medium

21. `design.md:683-692` (decision 3) | `<preset> <noise> [cpu] [smoke]` with argparse choices | with defaults the two
    positionals become `nargs='?'`; then `python entry.py cpu` assigns `cpu` to `preset` and argparse rejects it, so the
    optional words and the defaults cannot coexist | `[preset] [noise] [--cpu] [--smoke] [--resume BASE] [--polish-only]
    [--data-root P] [--out-root P]`, defaults R1 lowpass | medium

22. `design.md:870` (layer 3) | compare "the common `lbfgs_outcome` fields except `wall_s`" | `reason` is common and is
    built from the loop text plus the acceptance text of `decide` (LB:97-131, LB:785); the `decide` merge
    (`design.md:457`) states only the boolean, so the strings differ and layer 3 fails on text | copy the strings, or
    exclude `reason` | medium

23. `design.md:115-118` (decision 1) | "one process must never have both roots on its path" | at `<repo>/Clean/` the
    mainline `model_augmentation/` sits in the natural working directory; a launch that puts the cwd first (`python -m`,
    `python -c`, a notebook at the repo root, `PYTHONPATH=Clean`) imports the clean `gantry_dynamic` with the MAINLINE
    `model_augmentation` (mainline `gantry_dynamic` lives in `scripts/gantry/`, so only the framework is shadowed) and
    runs without an error | one guard in `Clean/gantry_dynamic/__init__.py` that `model_augmentation.__file__` lies in
    `Clean/` (it checks the environment, which rule G keeps) | medium

### Low

24. `design.md:173-174`, `design.md:649-657` | `dtype_np`, `dtype_pt` "copy" (`GD/config.py:554-560` read `self.use_f64`)
    | `use_f64` is not in the ClassVar list; nor is `lbfgs` if the `cfg.lbfgs and checkpoint_dir` test
    (`GD/training.py:932`) stays | add `use_f64 = True` (or make the rows "rewrite") | low

25. `design.md:181`, `design.md:697-700` | `config_json_dict` writes every mainline key "with the spec values" | no
    source for the constants of removed options (`SNR`, `ORTH_BETA`, `ORTH_POINT_STRIDE`, `ORTH_RANK_TOL`,
    `LIVE_G_WEIGHT_INIT`, `CHECKPOINT_CHUNK`, `STATIC_PASS_BUFFERS`, `LBFGS_CHUNK`, `LBFGS_FREEZE`,
    `LBFGS_TIME_BUDGET_S`, `LBFGS_METER_REL_TOL`, `LBFGS_METERS_GATE`, `LOSS_WEIGHTING`, ...); layer 3 compares every
    shared key | take each value from a final-run `config.json` (job 89792) and list it | low

26. `design.md:241` | `get_encoder_dims` merged as `na = nb = cfg.na_nb` (`GD/model.py:50`: `na = nb = hp['na_nb']`) |
    equal for a fresh run; under decision 4 `hp` comes from `resolve_resume_hp`, which takes `na_nb` from the checkpoint
    (`GD/training.py:1055`), and rule 1 forbids the substitution | keep `hp['na_nb']` | low

27. `design.md:176`, `design.md:650` | `na_nb` is a trimmed property and a ClassVar `na_nb = 30` | one name, two
    definitions, the later silently wins | keep one | low

28. `design.md:175` | `nf` as `round(NF_SECONDS / ts_new)` (`GD/config.py:573`:
    `return max(1, int(self.nf_seconds / self.ts_new))`) | same value (0.1 / 0.00025 evaluates to 400.0), different
    statement | copy the `int` form | low

29. `design.md:166`, `design.md:185`, `design.md:218` | `y_decimation`, `_decimate_y` copied while `ANTI_ALIAS_MODES`
    is "replaced" (`GD/config.py:48`, `GD/data.py:177`) | the copies reference a removed name | rewrite both on
    `cfg.noise == 'lowpass'`, or keep the constant | low

30. `design.md:661` | `combo_init_detune` without `float()` (EP:337) | values and JSON identical (`np.float64` subclasses
    `float`) | copy the statement | low

31. `design.md:426`, rule 7 | "one progress line per validation kept" while Tictoctimer goes | the line prints timer
    percentages (IC:1089-1093), so it is rewritten | call it a rewrite, list its fields | low

32. `design.md:41-44` (rule 4) | "Deletions carry no marker" inside Jan's classes | CLAUDE.md asks for
    `# CHANGED: <reason>` on a changed block inside an existing function of `model_augmentation/`; a trimmed Jan
    function is one | confirm with Dirk, or one `# CHANGED` line per trimmed Jan definition | low

33. `design.md:242`, `design.md:348`, `design.md:384`, `design.md:367`, `design.md:327`, `design.md:337` (case e, rule G)
    | kept: the `admissibility` check of the start (`GD/model.py:116-123`), `ClosedLoopSimulator._check_val_records`, the
    `ControllerBank` shape guards, the block positivity and value guards (`blocks.py:1030-1031, 1318-1323, 1338-1340`,
    spot-check) | each re-checks values the clean code builds itself (combos from constants and a fixed RNG; the list it
    registered; matrices from `controller_ss`), which rule G removes, about 60 lines | remove, or say why rule G does not
    apply | low

34. `design.md:494`, `design.md:377`, `design.md:380`, `design.md:431` (case e, spot-check) | `from_matrix` computes
    `col_corr`, `col_norms` only for the report (OBC:205-206, 221); `WindowControllerIndex`'s `record_weights is None`
    branch (CL:417-418) and `if train_weights is not None` (CL:488) are fixed by the servo weights; the `_sync` wiring
    guard (IC:1343-1346) stays while the identical `loss` guard (IC:1402-1405) goes | trim consistently | low

35. `design.md:312`, `design.md:404`, `design.md:407` against rule 2 | the out_gate re-home and per-call heals go
    although rule 2 keeps device-decided branches | no number changes (`init_forward` places the matrices at the module
    device and dtype, IC:270-276, and `_apply` keeps them), but the rule says otherwise | state the exception in rule 2 |
    low

36. `design.md:683` (decision 3) | no arguments runs R1 lowpass with `device='cuda'` | on the PC `build_model` stops at
    the CUDA check (`GD/model.py:286-292`); correct, but say so in the usage line | low

## 2. What the document must change for Dirk's four decisions

**Decision 1, `<repo>/Clean/`.** The section 1 tree and its "beside the thesis code" and "fixed relative path"
reasoning; `REPO_ROOT = ROOT/..` instead of `ROOT/../../..` (`design.md:185`; it feeds `git_provenance` and the default
`OUT_ROOT`); the server command (`srun python -u Clean/gantry_interconnect_dynamic.py ...`, `design.md:685`);
evaluation edit (a) puts `Clean/` on the path (`design.md:783`); open choice 1; the harness may stay in
`Thesis-writeup/Code/Verification/clean-code/` with `Clean/` as its import root; the import-origin guard (finding 23).

**Decision 2, roots as arguments.** `DATA_ROOT`, `OUT_ROOT` become `RunConfig` fields `data_root`, `out_root` with the
two defaults, set by CLI flags and by the harness; `traj_dir` and `save_dir` read them; their `config.json` keys and the
evaluation round trip as in finding 17; the fixed-path sentences of sections 1 and 4 go.

**Decision 3, CLI with a default.** Optional positionals plus flags (finding 21); the default stated with its GPU
settings (finding 36); the same CLI carries the root and resume arguments of decisions 2 and 4.

**Decision 4, resume and polish-only kept.** Restore the items of finding 18 as copies; restore the compiled try,
`_eager`, `ClosedLoopSimulator.eager` and the second eager probe (findings 15, 16); delete open choices 12 and 16;
decide the Adam moments of a `_best.pth` resume and state the cross-code `_best.pth` limit (finding 3); add layer 3c
(finding 19); define where continuations and reruns write and how they reach the evaluation (finding 20); keep
`hp['na_nb']` (finding 26). Section 5 item 3 then records the real `START_PHASE`, `start_phase`, `resumed_from` and
`resumed_at_epoch`.

## 3. Mapping spot-check (design section 2)

Checked in full by the delegated agent: every row of `MA/fit_systems/` (blocks, closed_loop, interconnect,
lbfgs_polish, obc, pre_encoder, ps2_opening, and the files dropped whole), `GD/training.py`, `GD/model.py`, plus
`MA/utils/*`, `MA/systems/gantry_ss.py` and `GD/obc_gantry.py`. Wrong destinations and drops are findings 5 to 14
above. Line numbers are correct apart from these ranges, each one to four lines short (low, agent-read, not re-checked
here): `_deriv_both_with` 968-973 and guard 909-914 (`design.md:324`); `params_init` branch 1024-1026 (`:327`);
`ControllerBank` guards 132-137 (`:367`); `output_only` heal 330-334 (`:407`); `fit` init branch 915-923 (`:426`);
pre_encoder map 378-417 and offsets 467-486 (`:521`); obc_gantry snr guard 114-118, `_static` 261-263, `use_jvp`
297-299, block-type guard 372-374 (`:259`, `:264`, `:265`, `:271`); `build_closed_loop` val-count guard 304-308
(`:203`, checked here). Also: `_NfProbe.record` is labelled "copy" but loses 600-601, so "trim" (`:289`);
`ClosedLoopSimulator.extra_array_names` (CL:480) is an unlisted class constant (`:393`); the module-level imports of
dropped names are in no row (IC:18; `GD/model.py:11, 17, 18`; `GD/obc_gantry.py:38, 42`; `GD/training.py:20-21`), so
add "imports follow the kept names" to rule 1.

## 4. Checked, no issue

- H1: `init_nets` draws nothing when the encoder is injected (IC:641-645); the dropped reconstructability branch,
  offsets and Kaiming branch draw nothing on the data_lsq path (`pre_encoder.py:369-433`); draw order Wb, Wa, net.
- H2: the nf probe's validation forward is `simulate` under `no_grad`, its indexer swap is restored
  (`GD/training.py:391-412, 469-477`), its generator is local; no model state changes.
- H7: no `torch.set_default_dtype` in the four locations, so the closure scalar is float32 as stated.
- The `decide` merge reproduces the `meters_gate=False` boolean (LB:110-131); dropping `* w` with one chunk is exact
  (`w = 1.0`); `param_loss` returns the Python float 0.0 (`blocks.py:1530-1531`), so the dropped sweep is exact.
- Dropping `warm_up_lazy_init` after `fit()` is exact (the epoch after the `_best` reload equals the rebuilt basis'
  epoch, so its `maybe_refresh` does nothing); the per-call heals are no-ops on the spec paths; deepSI
  `init_scheduler({})` returns None (`fit_system.py:127-137`), so the scheduler step is exact to drop.
- `prediction_coefficient` returns the same theta as `validation_coefficient`; dropping the truth reads keeps
  `x_phys_next_rec`, which is reconstructed from y (`GD/obc_gantry.py:140-142`).
- D-274 changes 1 to 3 sit where the design says (`ps2_opening.py:233-235`, `GD/data.py:89-90`,
  `GD/training.py:784-796, 948-962`); the preset table equals runs.tsv rows 221 to 230.
- Not verified: GPU and compiled behaviour beyond reading the code.
