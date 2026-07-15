# Evaluation Restructure — Step-by-Step Implementation Plan

Scope: improve the evaluation/reporting of the gantry augmentation pipeline
(`scripts/gantry/gantry_dynamic/`) without changing the augmentation implementation
(model, training dynamics, checkpoint format, RNG order). Executed one step per prompt.
Each step logs its own decision entry in `docs/decisions.md` before implementing.

## Why (three problems this fixes)
1. **Misleading comparison.** The `augmented vs baseline FP` table compares the augmented
   model (encoder-init) against the true-x0 baseline (different init), so the reported
   "+77%" is an initialization artifact, not the ANN. With the ANN at zero it is provably
   100% artifact (augmented == encoder-init baseline exactly).
2. **The real problem is invisible.** The ANN is not learning (best checkpoint = epoch 0,
   aug states = 0). The current output does not surface this clearly, and there is no
   metric on the training horizon to tell "wrong selector" from "not learning".
3. **Messy, ungrouped output** with mixed init conventions and no per-run folder.

## Grounding — current package (`scripts/gantry/gantry_dynamic/`)
- `config.py`   : `RunConfig` (frozen dataclass), `default_hp(cfg)`, `save_dir(cfg)`, `config_json_dict(cfg)`.
- `data.py`     : `DataBundle`, `Norm`, `load_datasets`, `compute_normalization`, `_resample_u` (D-087).
- `model.py`    : model build.
- `training.py` : `train_model_with_diagnostics` (the only file we touch for training).
- `baselines.py`: `compute_baseline_fp_nrms`, `stepwise_rollout`.
- `evaluation.py`: `evaluate_and_save`, `capture_loss_history`, `report_joint_estimation`, `_make_plots`, `_build_save_dict`.
- `diagnostics.py`: `state_recovery_diagnostic` (D-053), `aug_state_r2`, `compute_gradient_norms`, `encoder_init_state`, `r2_per_channel`.
- Entry `gantry_interconnect_dynamic.py` : `main()` orchestrates; `sdir = save_dir(cfg)` threads into
  training (checkpoint_dir), `evaluate_and_save`, `state_recovery_diagnostic`, grad-norm save.

## Metric definitions (label every number)
- **RMS [m]** — physical error, primary quantity; compares directly to the noise floor sigma_n [m].
- **NRMS [-]** — RMS / ystd; cross-channel comparison and comparison to a normalized floor.
- Report BOTH, labeled; a single improvement % (ystd cancels, RMS%==NRMS%).
- **full-traj sim-RMS** — open-loop over the whole record from one encoder init. THE SELECTOR
  (checkpoint + LR scheduler). Unchanged.
- **nf-window RMS** — windowed, encoder re-init at each window start, same nf as training.
  Measures what training optimizes. DIAGNOSTIC ONLY, never changes selection.

## Guardrails (hold across all steps)
- Do not change what drives training/checkpoint selection (stays full-traj sim-RMS).
- Do not change the checkpoint `.npz`/`.pt` format or the `hp` dict keys (resume must keep working).
- Oracle (FP+true MSD) is a diagnostic REFERENCE, never the acceptance threshold (that is the
  data-derived noise floor, D-078; supervisor rejected oracle floors).
- Cache the sim TRAJECTORY (noise-independent), never the score; recompute NRMS vs current
  `y_data` each run so the comparison is always fair.
- `save_dir(cfg)` stays the run FAMILY dir (home of the shared reference cache); per-run outputs
  go in `save_dir(cfg)/<run_id>/`.

---

## Step 1 — Per-run output folder + config snapshot
**Goal**: all artifacts of a run in one folder; each run self-documenting.
**Files**: entry `gantry_interconnect_dynamic.py` (`main()`).
**Change**: `sdir = os.path.join(save_dir(cfg), run_id)`; `os.makedirs(sdir, exist_ok=True)`.
Write `config.json` at `sdir` root = `{**config_json_dict(cfg), 'hp': hp, 'run_id': run_id}`.
Everything downstream already uses `sdir` (model, npz, plots, checkpoint), so no other change.
**Behavior**: preserving except output location. Checkpoint moves into the run folder;
`RESUME_CHECKPOINT` is a full path, so resume is unaffected.
**Verify**: run produces `simulations/gantry_subnet/augmentation_linear_map/<run_id>/`
containing the model, `gantry_results_*.npz`, plots, checkpoint, `config.json`.
**Decision**: D-090.

## Step 2 — Same-init table + RMS/NRMS + verdict + grouped output
**Goal**: honest, readable evaluation; the ANN's true contribution visible.
**Files**: `evaluation.py` (`evaluate_and_save` + its print helpers).
**Change**:
- Compare augmented (encoder-init) vs the **encoder-init** baseline (`baseline_encinit_nrms`),
  not the true-x0 baseline. Keep true-x0 baseline as a clearly-labeled reference row.
- Print every metric as **RMS [m] and NRMS [-]**, labeled; one improvement % column.
- **Verdict line** at top: ANN active? (aug-state RMS), best epoch, same-init aug/base %.
- Regroup prints into sections **A. Model quality / B. Encoder quality (D-053) / C. Augmentation
  (aug-state RMS, rollout R2) / D. Training health (loss, best epoch, grad norms)**. Keep every
  existing metric; only relocate/label.
**Behavior**: reporting only; no sims added (both numbers already computed in `main()`).
**Verify**: same-init aug/base = +0.0% on the current run; verdict reads "ANN inactive".
**Decision**: D-091.

## Step 3 — Dual validation metric (nf-window RMS alongside sim-RMS)
**Goal**: see training-horizon progress; distinguish "wrong selector" from "not learning".
**Files**: `training.py` (`train_model_with_diagnostics`).
**Change**: add a per-epoch hook that computes **nf-window RMS** on the val set (encoder
re-init per window, `nf=hp['nf']`), logs it to `history`, and prints per-epoch:
`train nf-loss | val nf-RMS (NRMS) | val sim-RMS (NRMS) *selector`. Selection/`bestfit`
still use full-traj sim-RMS — unchanged.
**Behavior**: adds a diagnostic; selection identical to before.
**Verify**: per-epoch line shows all three; `bestfit`/best-epoch identical to a run without the hook.
**Decision**: D-092.

## Step 4 — Diagnostic plots (into the run folder `plots/`)
**Goal**: make the failure/behavior visible; the overlay hides it.
**Files**: `evaluation.py` (`_make_plots` + helpers).
**Change**: `plots/` with `val/` and `test/` subdirs. Add:
- Enhance the existing **loss convergence plot** (`semilogy`, log y-axis, evaluation.py:244):
  add **val nf-window RMS (dotted)** next to val sim-RMS (solid, the selector), plus train loss
  (dashed) as reference. Solid-vs-dotted = full-traj vs training-horizon on the same data.
  (This replaces a separate metric-over-epochs figure.)
  UNITS (resolve at Step 3, do not guess): the plotted `Loss_val`/`Loss_train` come straight from
  deepSI and their relative normalization is unverified. Before adding the dotted line, read
  deepSI's sim-RMS definition and either (a) compute the nf-RMS to match `Loss_val`'s
  normalization (keeps the authentic selector line), or (b) compute BOTH val curves ourselves in
  one explicit unit (RMS[m]) and keep framework train loss as the reference line. Pick so that
  solid and dotted are guaranteed comparable.
- `*_error_trace.png` — `(y_model - y_data)` per axis vs time, baseline/augmented, warmup shaded
  (oracle line added in Step 6). Primary + all val/test records.
- `*_error_spectrum.png` — FFT of the Y error (does the absorber peak at ~157 Hz get removed?).
  Primary val/test.
- `03_summary_nrms_bars.png` — augmented vs baseline across val+test, per axis.
Keep existing loss/overlay/ann-states plots (moved into `plots/`).
**Behavior**: additive.
**Verify**: files present; error-trace reveals offsets; spectrum shows/does-not-show 157 Hz.
**Decision**: D-093.

## Step 5 — Oracle model (FP + true MSD) + pipeline verification
**Goal**: a best-case-structure reference and a Python==Simulink check.
**Files**: new `gantry_dynamic/oracle.py`.
**Change**: build the 8-state FP+MSD plant from `gtd_config.m` params (ma=0.5*mh, ka, ca,
fa=150, zeta_a=0.05); simulate open-loop from true-x0 (6 physical + delta_a/vdelta_a from the mat).
**Behavior**: new, standalone; not yet wired into tables (Step 6).
**Verify**: oracle open-loop reproduces the mat `y` to ~discretization error on a few records
(confirms the Python truth model matches Simulink and that open-loop eval is trustworthy).
**Decision**: D-094.

## Step 6 — Reference cache + wire oracle into tables/plots + coverage/granularity
**Goal**: cheap references, oracle visible, per-record coverage; keep runtime sane.
**Files**: new `gantry_dynamic/reference_cache.py`; `evaluation.py`; `baselines.py`.
**Change**:
- Cache the **trajectory** for the training-independent references (true-x0 baseline, oracle).
  Fingerprint key: record id + mtime, MODE, fs_new, up_sample, K0, dtype, init type, (oracle:
  absorber params), compute-hash of the sim function. One **shared** file in `save_dir(cfg)`
  (family dir), flat keys `"<fp>/<record>/<init>"`, **append-only** (never overwrite). Recompute
  NRMS vs current `y_data` each run. Do NOT cache the encoder-init baseline (fragile key;
  recompute for primary records).
- Add the **oracle column** to the model table and the **oracle line** to error-trace plots.
- **Per-record coverage summary** over val+test (augmented-only, one sim each); train records opt-in.
**Behavior**: additive + speed; numbers identical to non-cached on a cold cache.
**Verify**: second run reuses cache (no recompute of true-x0/oracle); oracle column ~machine
precision; coverage summary present; cold-cache numbers match non-cached.
**Decision**: D-095.

---

## Order and dependencies
1 -> 2 -> 3 -> 4 (needs 3's curves) ; 5 -> 6 (needs 5's oracle). 1 and 2 are independent of 3-6.
Recommended sequence: 1, 2, 3, 4, 5, 6.

---

# Implementation sketches (clean / minimal)

Grounding facts verified in the code (not assumed):
- Entry `main()` threads a single `sdir` into training (`checkpoint_dir`), `evaluate_and_save`,
  `state_recovery_diagnostic`, and the grad-norm save. So the per-run folder is one line.
- `train_model` (model.py:183) hands the whole loop to `fit_sys.fit(...)`; there is no per-epoch
  callback. deepSI validates once per epoch via `self.cal_validation_error(val, validation_measure)`
  (fit_system.py). `'sim-RMS'` -> `System_data.RMS` (meters); `'{n}-step-average-RMS'` ->
  `n_step_error(..., mode='RMS')` (meters). **Both metrics are physical-meter RMS -> dotted vs
  solid is directly comparable; the earlier units worry is resolved.** (train loss stays the
  framework's normalized sqrt-loss, a reference line in different units.)
- `compute_baseline_fp_nrms(...)` already returns `(nrms, y_hat)` -> we cache `y_hat` (the
  trajectory) and recompute `nrms` against current `y_data`.

## Step 1 sketch — entry `gantry_interconnect_dynamic.py:main()`
```python
run_id = os.environ.get('SLURM_JOB_ID') or datetime.now().strftime('%Y%m%d_%H%M%S')
family_dir = save_dir(cfg)                       # stays the cache home (Step 6)
sdir = os.path.join(family_dir, run_id)          # per-run folder
os.makedirs(sdir, exist_ok=True)
# ... after hp is resolved (post-resume block):
if cfg.save_flag:
    with open(os.path.join(sdir, 'config.json'), 'w') as f:
        json.dump({**config_json_dict(cfg), 'hp': hp, 'run_id': run_id}, f, indent=2)
```
Everything downstream already uses `sdir`. No other change. Verify: artifacts land in
`.../augmentation_linear_map/<run_id>/`.

## Step 2 sketch — `evaluation.py:evaluate_and_save` (reporting only)
- The headline `augmented vs baseline` table currently pairs `y_hat_enc` (encoder-init) against
  `baseline_nrms` (true-x0). Change the pairing to `baseline_encinit_nrms` (same init). Keep
  `baseline_nrms` and the oracle as clearly-labeled reference rows.
- Add per-channel `RMS[m]` and `NRMS[-]` columns (NRMS = RMS/ystd; one improvement %).
- Add a verdict line up top: `ann_active = ann_rms_enc.max() > tol`; print same-init %.
- Wrap the existing prints in `A. MODEL / B. ENCODER / C. AUGMENTATION / D. TRAINING` headers;
  no metric removed, only relocated. No new sims (both baselines already computed in `main()`).

## Step 3 sketch — `training.py` (per-epoch nf-window RMS, selection unchanged)
```python
def _install_nf_val_probe(fit_sys, hp, cfg):
    """Piggyback nf-window RMS on each epoch's validation. concurrent_val=False only."""
    nf, probe_stride = hp['nf'], max(1, cfg.stride)   # bigger stride = cheaper probe
    fit_sys.Loss_val_nf = []
    orig = fit_sys.cal_validation_error
    def wrapped(val_sys_data, validation_measure='sim-NRMS'):
        sel = orig(val_sys_data, validation_measure=validation_measure)   # selector, untouched
        try:
            e = fit_sys.n_step_error(val_sys_data, nf=nf, stride=probe_stride,
                                     mode='RMS', mean_channels=True)
            fit_sys.Loss_val_nf.append(float(np.mean(e)))
        except Exception:
            fit_sys.Loss_val_nf.append(float('nan'))
        return sel
    fit_sys.cal_validation_error = wrapped
    return orig
```
In `train_model_with_diagnostics`, wrap around the `train_model(...)` call:
```python
_orig = _install_nf_val_probe(fit_sys, hp, cfg)
bestfit = train_model(fit_sys, hp, cfg, data, epochs=..., nf=hp['nf'], validation_measure='sim-RMS')
fit_sys.cal_validation_error = _orig                      # restore
loss_val_nf = np.array(getattr(fit_sys, 'Loss_val_nf', []))   # aligns with fit_sys.Loss_val
```
Return `loss_val_nf` (via the diag dict) for the plot/print. Selection and `bestfit` untouched
(the wrapper returns the original selector value). Print per-epoch line stays the framework's;
add a post-training summary of the aligned nf curve.

## Step 4 sketch — `evaluation.py:_make_plots` + helpers
- `capture_loss_history` also returns `loss_val_nf`; thread it into `_make_plots`.
- Plot 1 (loss convergence, already `semilogy`): add one line
  `ax1.semilogy(epoch_id_full, loss_val_nf, 'C0', ls=':', label='Val nf-RMS')`
  next to the solid `Val loss` (sim-RMS) and dashed `Train loss`. Both val curves are meters.
- New helpers (small, own functions): `plot_error_trace(record)` `(y_model-y_data)` per axis vs
  time (baseline/augmented; oracle line added Step 6); `plot_error_spectrum(record)` FFT of Y
  error; `plot_nrms_summary(val+test)` bars. Save under `sdir/plots/{val,test}/`.

## Step 5 sketch — new `gantry_dynamic/oracle.py`
- Build the 8-state FP+MSD plant from `gtd_config.m` params (ma=0.5*mh, ka=ma*(2*pi*150)^2,
  ca=2*0.05*sqrt(ka*ma)); reuse `gantry_ss` matrices for the 6 physical states, couple the
  absorber on Y. Simulate open-loop from true-x0 (`x_logical[K0]` + `delta_a/vdelta_a[K0]`).
- Verification sub-step (must pass before use): oracle open-loop reproduces the mat `y` to
  ~discretization error on a few records -> confirms Python == Simulink.
- Detailed matrix construction to be confirmed against the MATLAB augmented plant when we reach
  this step (read `Matlab-scripts/Augmentation` generator + `gantry_ss` augmented block).

## Step 6 sketch — `gantry_dynamic/reference_cache.py` + wire-in
```python
def ref_fingerprint(cfg, hp, record_file, init_type):
    return hashlib.md5(repr((
        os.path.basename(record_file), os.path.getmtime(record_file),
        cfg.mode, cfg.fs_new_hz, hp['up_sample'], K0, str(cfg.dtype_np), init_type,
        ORACLE_PARAMS if init_type=='oracle' else None, _SIM_COMPUTE_HASH,
    )).encode()).hexdigest()[:12]
```
- One shared file `os.path.join(family_dir, '_reference_cache.pt')`; dict keyed
  `f'{fp}/{record}/{init}' -> y_hat` (trajectory). Append-only; never overwrite.
- `compute_baseline_fp_nrms` wrapper: on hit, load `y_hat` and recompute nrms vs current
  `y_data`; on miss, compute + append. Encoder-init baseline NOT cached (fragile key).
- Wire the oracle column into the Step-2 table and the oracle line into the Step-4 error traces;
  add the per-record coverage summary over val+test (augmented-only, train opt-in).
