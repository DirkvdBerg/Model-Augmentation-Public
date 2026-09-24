# blackbox-closed-loop: decisions

Every entry is written BEFORE the code or run it governs. Gate thresholds are pre-registered here
and are not moved after a number is seen; a changed threshold gets a new entry that says why.
Source handoff: `tasks/handoffs/2026-09-23-blackbox-closed-loop.md` (section 0 authorises
unattended work for this session; files only inside this folder, no commit).

### BB-001 Scope, folder, resource rules
**Date**: 2026-09-24
- All files live in `scripts/gantry/blackbox-closed-loop/`. Nothing outside it is written.
  Every Python run sets `PYTHONDONTWRITEBYTECODE=1` so importing repo modules (the grey-box entry
  file, and the original packages in the G0 reproduction check) cannot write `__pycache__` outside
  this folder.
- Every run goes through `tools/run.sh`, which refuses to launch below 4.0 GB available RAM
  (handoff section 13) and wraps the command in `tools/watchdog.ps1` (copied unchanged from
  `telica-real/tools/watchdog.ps1`: alert < 4.0 GB, kill < 2.5 GB RAM or < 2.0 GB C: free).
  One process at a time. `torch.set_num_threads(4)`.
- Compute limit: no training epochs; at most 2 optimizer updates in the whole session (G4). Forward
  and backward passes without an optimizer step (G1, G3) are not updates.
- Local runs use `device='cpu'` and `compile_mode=None` (eager), per handoff section 13. These are
  the only two fields overridden on the grey box's `CFG` for local work, via `dataclasses.replace`,
  and each script prints them.

### BB-002 Vendoring and how the grey box's configuration is read
**Date**: 2026-09-24
**What**: `vendor/` holds unmodified copies of `model_augmentation/` (whole package: the grey box's
`model.py` imports interconnect, blocks, pre_encoder, systems and utils, and `closed_loop.py` is
inside it), `scripts/gantry/gantry_dynamic/` (whole package), the black-box sources
`ann_blackbox.py`, `bla_init.py`, `frf_init.py`, `data.py`, `vector_fit.py`, `lpm_frf.py`, and
`msd-offset/plant.py` (the loader the four existing checkpoints were trained through). Also `scripts/gantry/common/{oracle,orth_penalty,rezero_gate}.py` as
`vendor/common/` (`gantry_dynamic/evaluation.py` imports `common.oracle` at module level; the other
two are conditional imports of `model.py`). This folder's own bootstrap is therefore named
`adapter/bbcl.py`, not `common.py`, to avoid shadowing that package (found by run g0_infra,
attempt 1). Provenance
(commit, sha256, dirty flag) in `vendor/VENDORED.md`. The ONLY edits are path lines marked
`VENDOR-PATCH (BB-002)`: repo root five levels up instead of three, and `sys.path` pointing at
`vendor/` instead of the repo root, so every `model_augmentation` import resolves to the copy.
**CFG is read, not retyped**: `bbcl.load_cfg()` imports the repo's own
`scripts/gantry/gantry_interconnect_dynamic.py` by path AFTER the vendored `gantry_dynamic` and
`model_augmentation` are in `sys.modules`, so the entry file's module code runs unchanged and its
`from gantry_dynamic... import` lines bind to the vendored package. It asserts that the entry's
experiment env overrides (`DATASET_AB_MODE`, `DATASET_AB_SEED`, `BURN_IN_AB`, `LBFGS_AB_FREEZE`,
`OBC_ARM`) are unset. The data path is `traj_dir(cfg)` of the vendored `data.py`, which resolves
to the grey box's folder in `data/gantry/matlab/trajectory/<cfg.mode>`: read, never copied.
**G0 PASS criteria** (all three):
1. After importing every module this folder uses, every loaded module whose name starts with
   `model_augmentation`, `gantry_dynamic`, `common`, `bla_init`, `plant`, `frf_init`,
   `vector_fit`, `lpm_frf` has `__file__` inside `vendor/`.
2. `CFG` comes from the repo entry file (its module `__file__` is
   `scripts/gantry/gantry_interconnect_dynamic.py`) and `traj_dir(CFG)` exists and holds all 29
   names of `TRAIN_FILES + VAL_FILES + TEST_FILES`.
3. Each vendored file equals its source byte for byte except for lines containing
   `VENDOR-PATCH`, checked by a line diff at run time (so a later edit of a source is visible).

### BB-003 The adapter: the grey box's own fit system with a black-box `hfn`
**Date**: 2026-09-24
**What**: the black box is an instance of the SAME class the grey box trains,
`SSE_Interconnect_Composed`, with two objects swapped:
- `hfn` = `BlackBoxHF`, a subclass of deepSI 0.3.29's `hf_net_default` (Jan's
  `SS_encoder_general_hf` step, `y = h(x)`, `x+ = f(x, u)`, `f = default_state_net`,
  `h = default_output_net`, no feedthrough), plus `output_only(x) = h(x)` (legal because `D = 0`),
  `connected_blocks = ()` and a no-op `init_model`. `.view` is replaced by `.reshape` (the
  documented non-contiguity gotcha of `full-blackbox/README.md` step 1); identical values.
- `encoder` = deepSI's `default_encoder_net` with the same `.reshape` fix.
**Why**: then `loss` (MSE, `burn_in` slice), the `simulate` seam, `make_training_data` with
`ctrl_ix`, `fit()`, `cal_validation_error`, checkpoint selection and the controller bank are the
grey box's code by construction, not a reimplementation that could drift. With
`connected_blocks = ()` the composed loss has no parameter prior, no orthogonality term and no OBC,
which is correct for a black box. The simulator is attached with the grey box's own
`build_closed_loop(fit_sys, norm, cfg, train_files=TRAIN_FILES, val_files=VAL_FILES,
val_data=data.val_ckpt_data)` (keyword-only, never swapped).
**Ruled out**: subclassing deepSI's `SS_encoder_general_hf` and patching its `loss`: its `fit()`
is deepSI's stock one, which has no simulator seam, no `ctrl_ix` passing and flips the model to
the CPU for validation. That would be a second closed-loop implementation.

### BB-004 Black-box settings and their sources
**Date**: 2026-09-24
| Setting | Value | Source |
|-|-|-|
| `nx` | 8 | Jan `msd_ndof_deepSI_encoder.py` L30 `nx = 2*dof`; the gantry truth has 4 dof (X, Theta, Y, absorber); same as the four existing arms |
| `f`, `h` | `simple_res_net`, 2 x 8, tanh | Jan L27 |
| encoder | `simple_res_net`, 2 x 16, tanh | Jan L28 |
| `na = nb` | `cfg.na_nb` (29) | grey box, so the windows are identical (see below) |
| `na_right = nb_right` | 1 | grey box `get_encoder_dims` for `linear_map` |
| optimizer | Adam, `lr = 1e-3`, `eps = 1e-8` | Jan L33 (deepSI default Adam); NOT the grey box's 1e-5 / 1e-16, which were tuned for a zero-initialised ANN on top of physics (D-148, D-189). **Superseded for the server job by BB-013: lr = 1e-7** (measured margin); 1e-3 was used only in G4 smoke |
| norm | grey box `u_mean, std_u, y0, ystd`, `auto_fit_norm=False` | `full-blackbox/README.md` section 5 match list; also required because the controller bank folds exactly these scalings |
| dtype | float32 | `cfg.use_f64 = False` |
| init | random (seed `cfg.seed`), N4SID BLA, FRF BLA | handoff section 8; BB-007 |
**Deviation from Jan, stated**: Jan's `na = 2 nx + 1 = 17`, `na_right = 0`. Taking the grey box's
29 and `na_right = 1` makes `k0 = max(na, nb) = 29` and `k0_right = nf` identical, so the TRAINING
WINDOWS (start sample, count per record, `ctrl_ix`) and the SCORED SAMPLES of the free run are
identical between the two models. The black box sees 30 past samples including `y[k]` instead of
17; that is more information, never less, so it cannot bias the comparison against it.

### BB-005 Data, split, rate, horizon, burn-in, selection and budget: all from CFG
**Date**: 2026-09-24
Read from the grey box's `CFG` at run time: `mode`, `fs_new` (so `ts_new`), `nf` (from
`nf_seconds`), `stride`, `burn_in`, `batch_size`, `epochs`, `its_per_val`, `n_its`, `seed`,
`lbfgs*` (the polish phase), `closed_loop`. Records: `TRAIN_FILES` (18), `VAL_FILES` (6),
`TEST_FILES` (5) of the vendored `data.py` (equality with the repo's lists is G0 criterion 3).
Selection: closed-loop free-run `sim-RMS` in metres over `val_ckpt_data`, N-sample weighted mean
(`ClosedLoopSimulator.validation_error`). The server job calls the vendored grey-box
`model.train_model` (Adam phase, identical `fit()` arguments) and `training.run_lbfgs_polish`
(identical `PolishSpec`); the grey box's `_NfProbe` diagnostic is not installed (reporting only,
and it assumes a physical block). Both functions take the black-box config
`bb_cfg = replace(CFG, lr=1e-3, adam_eps=1e-8)`: exactly the two optimizer fields of BB-004 change
(`train_model` asserts the optimizer's eps equals `cfg.adam_eps`, so they must travel together);
every other field is CFG's.

### BB-006 G1 pre-registered tolerances
**Date**: 2026-09-24
All on the black box at random init (seed `cfg.seed`), float32, one batch of 64 training windows
drawn with `numpy.random.default_rng(cfg.seed)` from the production training windows, `nf` from
CFG.
- **(a) zero controller = open loop.** Replace the bank by one with all four matrices zero, run
  `closed_loop_rollout`, compare with the open-loop `simulate` of the same fit system (simulator
  detached). PASS: `max|y_cl - y_ol| <= 1e-6 * max|y_ol|`.
  `# HEURISTIC:` float32 unit roundoff is 6e-8; u_fb is exactly zero here so the expected value is
  0; 1e-6 relative leaves room only for reduction-order effects.
- **(b) residual identity.** A probe `hfn` whose `output_only` returns `y_data[k]` exactly (so
  `e = 0` at every step) records the input it is stepped with. PASS:
  `max|u_seen - u_data| == 0` in normalised units (tolerance `<= 1e-7 * max|u_data|`), AND the bank's
  own units self-check `check_units` has relative error `<= 1e-5` on all three channels at
  every controller row. `# HEURISTIC:` float32 folding of `ystd/std_u` into `D`.
- **(c) grey-box number reproduced.** The grey box built by the vendored `build_model(hp, cfg,
  data, norm)` at `CFG` (untrained: FP baseline, zero-output ANN, `linear_map` encoder; this is the
  model whose closed-loop score is the grey box's `Loss_val[0]` in every run), with its own
  simulator from `build_closed_loop`. Three numbers on the 6 validation records:
  1. `fit_sys.cal_validation_error(val_ckpt_data, 'sim-RMS')` (the grey box's own selection path);
  2. this folder's scorer `score/cl_score.py::closed_loop_score` (the function G2 and G3 use),
     which builds its own bank from the record names and calls `closed_loop_free_run_rms_batch`;
  3. number 1 recomputed in a separate process from the ORIGINAL repo packages (not the vendored
     copies), to prove the vendoring changed nothing.
  PASS: `|2 - 1| / 1 <= 1e-6` and `|3 - 1| / 1 <= 1e-6`. `# HEURISTIC:` same arithmetic in
  float32; the handoff's example shows agreement to all printed digits.
  No trained grey box exists on this dataset (searched `simulations/`, `scripts/**` for the mode
  string on 2026-09-24); the untrained model's number is the grey box's own closed-loop number
  that a run records at epoch 0, so it is the one available to reproduce.

### BB-007 G2 transfer measurement on the four existing checkpoints
**Date**: 2026-09-24
Checkpoints (open-loop, trained on `T10_aprbs_60`, validated on `V2_aprbs_Ylow`, 800 Hz, old
`augmentation` folder, `nf = 400`):
random `ann-blackbox/server-results/paired_random/ann_blackbox_fs800_nf400_s0`,
N4SID BLA `ann-blackbox/server-results/paired_bla/ann_blackbox_fs800_nf400_s0_bladynz`,
FRF `ann-blackbox/results/phase1_frf/ann_blackbox_fs800_nf400_s0_blafrfz`,
oracle `ann-blackbox/results/phase1_oracle/ann_blackbox_fs800_nf400_s0_blafrfz_oracle`.
Read in place (read-only). Data through the vendored `ann_blackbox/data.py::load` at 800 Hz (their
own loader, FIR anti-alias on y). Controller: vendored `controller_ss(Y_op, 1/800)`, the Tustin
controller at THEIR rate; `Y_op` from `RECORD_Y_OP` and cross-checked against `y[0, 2]` of the
record (must agree to 1e-6 m). Bank folded with each checkpoint's own `norm.ystd`, `norm.ustd`.
Scored records: V2 (their validation record) and T10 (their training record). Encoder window
`k0 = max(na, nb) = 17` (theirs). Open-loop counterpart recomputed with `apply_experiment` on V2
and checked against the checkpoint's `best_sim_rms` in its metrics JSON (relative difference
`<= 1e-3`, else the load is wrong and the closed-loop number is not reported).
**Stability rule** (also used by G3), `# HEURISTIC:` a closed-loop free run is STABLE if every
sample is finite and `max|y_model - y_data| <= 10 * max|y_data|` over the scored samples; the
factor 10 separates bounded misfit from divergence, which on a free integrator grows without bound.
PASS: four closed-loop numbers with their open-loop counterparts and stability flags reported,
one process per checkpoint.

### BB-008 G3 initialisation on the production data
**Date**: 2026-09-24
Arms, all with the BB-004 structure on the production data at `cfg.fs_new`:
- **random**: deepSI/PyTorch default init, `torch.manual_seed(cfg.seed)` (Jan-faithful control).
- **N4SID BLA** (`dyn`, nonlinear branches zeroed; the existing best open-loop arm `bladynz`):
  vendored `bla_init.apply_bla_init(mode='dyn', zero_nonlinear=True)`, N4SID fitted on
  `T10_aprbs_60` of the production folder at `cfg.fs_new`, normalised by the grey box's norm.
  One record, as the existing arm: 18 records at 4 kHz are 864k samples and the N4SID Hankel
  matrices would need about 1.7 GB each at the grid's `SS_f = 40`.
- **FRF BLA** (nonlinear zeroed, `blafrfz`): the continuous-time model `(Ac, Bc, C)` of
  `ann-blackbox/results/frf_init_joint_lowf_ma50_a5/frf_init_ss.npz` (LPM FRF of a standstill
  multisine on the ma50 plant, unpinned, i.e. data only) discretised ZOH at `cfg.ts_new` and
  written with `bla_init.apply_frf_bla_init`. `# THEORY:` ZOH discretisation via the augmented
  matrix exponential, frf_init.py `c2d`. Caveat stated with every number: it was identified on a
  separate experiment (ma50, no friction), not on the production records.
Encoders stay random in all three arms (the existing arms did the same, `dyn`/`frf` write only f
and h).
Measured per arm: (1) closed-loop free run on the 6 validation records through the production
simulator (`fit_sys.cal_validation_error`), per-record RMS, stability flag (BB-007 rule);
(2) the initial closed-loop validation sim-RMS; (3) three batches of `cfg.batch_size` training
windows (rng seeded `cfg.seed`), forward and backward through `fit_sys.loss(..., ctrl_ix=...)`,
no optimizer step: loss, total gradient 2-norm, finiteness of every gradient entry.
**PASS**: at least one arm stable on all 6 records with all three losses finite and gradient norm
finite and `> 1e-12` (`# HEURISTIC:` distinguishes a live gradient from an exactly dead one).
**Server arm selection rule**: among passing arms, the lowest initial closed-loop validation
sim-RMS; random always runs as the Jan-faithful control arm regardless.

### BB-009 G4 smoke and cost
**Date**: 2026-09-24
The chosen arm, through the grey box's `fit()` exactly as the server job calls it but with
`n_its = 2` (so `its_per_val = 2`: the initial validation and one after update 2), CPU, eager,
under the watchdog. That is the session's full update budget. Measured: wall time per update
(from fit's timer lines), validation wall time, peak tree RAM (watchdog).
**Projection**: updates `= cfg.epochs * (N_windows // cfg.batch_size)` exactly as `fit()` computes
it, validations `= floor(updates / its_per_val) + 1`. Server time per update is not measured here;
it is estimated as `t_gb_server * (t_bb_cpu / t_gb_cpu)`, where `t_gb_server = 1.236 s/update`
(grey box, nf 400, batch 512, compiled, job 83978, entry-file comment) and the ratio is measured
locally on the same machine, same batch, forward plus backward without an optimizer step for both
models. `# HEURISTIC:` assumes both rollouts are dispatch-bound on the GPU (the entry file's own
statement for the grey box), so their per-update cost scales like their per-step op cost. Also
reported: the local CPU cost as an upper bound. PASS: runs end to end; cost written.

### BB-010 G5 server job and the pre-registered comparison rule
**Date**: 2026-09-24
Job: `runners/train_bb.py` + `runners/run_bb_arm.sh` (SLURM, same partition/time/GPU header as
the grey box's own runner), arm by env `BB_ARM`, NOT submitted. Everything in BB-005 read from CFG
at run time; `device` and `compile_mode` left at CFG's values on the server.
**Comparison rule**, fixed now: on the same 6 validation records (selection) and 5 test records
(reporting), closed-loop free-run RMS in metres, same harness. Ratio `R = BB / GB` (trained grey
box from its own run on this dataset). `# HEURISTIC:` verdict bands, since no seed-spread of the
grey box exists on this data to derive them from: `R <= 1.25` similar; `1.25 < R <= 2` same order,
worse; `R > 2` clearly worse; `R < 0.8` black box better. If both arms are run with two or more
seeds, the bands are REPLACED by the data-derived rule: similar iff the black box's seed range
overlaps the grey box's. Both are also reported against the FP baseline (grey box epoch 0, G1(c));
a model that does not beat the baseline is reported as such regardless of R.
The oracle floor is not computable in this harness on this dataset: the truth contains Karnopp
friction with a stick state and there is no torch model of it; the noiseless truth reproduces
`y_data` up to the 20 kHz to 4 kHz rate change, so the floor is that resampling error, not a
modelling number. The row is kept and marked so.

### BB-011 Memory: the full training-window set is not built locally
**Date**: 2026-09-24
**Measured basis**: at CFG (18 records of 48000 samples, `k0 = 29`, `nf = 400`, `stride = 10`)
deepSI builds `len(range(429, 48001, 10)) = 4758` windows per record, 85,644 in total; `ufuture` and
`yfuture` are 85,644 x 400 x 3 float32 = 411 MB each, and `System_data_list.to_hist_future_data`
concatenates per-record arrays, so the peak is about twice that, ~1.7 GB, on a machine that idles at
4.7 to 5.2 GB available with the watchdog killing at 2.5 GB.
**G3**: gradient batches are drawn WITHOUT building the full set: global window indices are sampled
with `numpy.random.default_rng(cfg.seed)` over the per-record counts from the framework's own
`window_controller_index`; each record's windows are then built with deepSI's own
`to_hist_future_data` on that record alone (~46 MB), the sampled rows kept, the rest freed. The
per-record count from deepSI is asserted equal to the framework's count, and `ctrl_ix` is the
simulator's `train_ctrl_rows[record]`, so the batch is exactly what `make_training_data` would have
delivered for those indices.
**G4**: the smoke run goes through the server job's own script (`runners/train_bb.py`) with a
`--smoke-records K` option that passes the first K training records (K = 4) to `fit()` and to
`build_closed_loop(train_files=...)`, `n_its = 2`, and the L-BFGS polish OFF (every L-BFGS
iteration is an optimizer update, so it cannot run inside the 2-update budget; it is stated as not
exercised locally). Per-update cost does not depend on K (fixed batch 512, fixed nf); the
projection uses the full 18-record window count computed with `window_controller_index`, which is
the count `fit()` itself uses.

### BB-012 G4 smoke: deepSI checkpoints stay inside this folder; G3 outcome fixes the server arm
**Date**: 2026-09-24
- deepSI's `fit()` writes `_best`/`_last` checkpoints to `%LOCALAPPDATA%/deepSI/checkpoints` on
  Windows, which is outside this folder. The smoke run sets the process's `LOCALAPPDATA` to
  `outputs/g4/<run>/localappdata` (no other effect on the run). The server keeps `~/.deepSI`.
- G3 applied the BB-008 selection rule: the only passing arm is **frf** (random unstable, n4sid
  not constructible at 4 kHz), so the server arm is `frf` and `random` runs as the control. The
  control is expected to stop at its first update on a NaN loss (fit's own NaN break); it is kept
  because the rule says it always runs and it documents that failure on the server at no cost.
- G4 smoke therefore uses `BB_ARM=frf`.

### BB-013 The server learning rate, from the loop's measured perturbation margin
**Date**: 2026-09-24
**Why**: G4 smoke (run g4_smoke) applied update 1 at the BB-004 lr = 1e-3 (Jan's) from the FRF start
and the loss of update 2 was NaN: one Adam step destabilised the closed loop. Adam's first step
moves every parameter by about `lr` (`# THEORY:` Kingma and Ba 2015, Sect. 2.1, the effective step
is bounded by the step size, |Delta theta| ~ alpha at initialisation). The update budget is spent,
so the admissible step size is measured WITHOUT an optimizer step.
**Method** (`init/g4_lr_margin.py`, one process): FRF-init black box as in G3. For
`delta in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6)` and 3 sign draws
(`numpy.random.default_rng(cfg.seed + draw)`), a copy of the model with every parameter
`theta + delta * s`, `s` uniform in {-1, +1} (random, NOT the gradient sign, so no training
direction is ever computed). Per copy: the 3 G3 gradient batches' loss (forward only) finite, and
the closed-loop validation free run stable on 6/6 records (BB-007 rule).
**Rule**: `delta*` = the largest grid value for which all 3 draws pass both checks, and every
smaller value passes too. Server `lr = delta* / 10`. `# HEURISTIC:` the factor 10 covers that a
random direction is not the worst case and that Adam's per-step move can reach a few times `lr`
when the gradient is consistent; it is not derived. If `delta*` does not exist in the grid, the
frf arm is reported as not trainable at any tested step size (G5 records it; no job arm is changed
on a guess). The new lr replaces BB-004's for the server job (BB-004's 1e-3 stays the recorded
Jan value); `train_bb.py` reads it from `bb_model.BB_LR`.
