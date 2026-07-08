# Design Decisions

Decisions are logged here before implementation. Each entry states what was decided, why, what was ruled out, and what it constrains going forward.

---

## Decision Template

```
### [D-NNN] Title
**Date**: YYYY-MM-DD
**What**: What was decided.
**Why**: The reason — constraint, evidence, or trade-off that drove the choice.
**Ruled out**: Alternatives considered and why they were rejected.
**Constrains**: What future decisions or implementations this locks in.
```

---

## Decisions

### [D-101] Pass hp['lr'] into init_model in build_model — the configured learning rate was silently ignored (every gantry run trained at Adam default 1e-3)
**Date**: 2026-07-08
**What**: One-line change in `scripts/gantry/gantry_dynamic/model.py:build_model`:
`fit_sys.init_model(sys_data=..., auto_fit_norm=False)` becomes
`fit_sys.init_model(sys_data=..., auto_fit_norm=False, optimizer_kwargs={'lr': hp['lr']})`.
**Why**: The learning rate knob was disconnected. `build_model` calls `init_model`, which creates
the optimizer (`interconnect.py:425` -> `init_optimizer` -> `Adam(parameters)` with no `lr` ->
**Adam default 1e-3**) and sets `init_model_done=True`. Later `train_model` (model.py:185) calls
`fit(..., optimizer_kwargs={'lr': hp['lr']})`, but `fit` only consumes `optimizer_kwargs` inside the
`if init_model_done==False` branch (interconnect.py:548); the `else` branch just runs
`_check_and_refresh_optimizer_if_needed()` (a CUDA-graph health check, fit_system.py:520) which never
touches lr. So `hp['lr']=1e-4` (config `cfg.lr`) was silently dropped and **every gantry augmentation
run — real pipeline and all historical run-table entries — trained at 1e-3, 10x the intended rate.**
Discovered when a Theta-only lr sweep (`diag_theta_lr_sweep.py`, lr in {1e-5,1e-6,1e-7} then
{1e-10,1e-12,1e-13}) produced **bit-identical** loss curves across all lrs (val sim-RMS
0.006071/0.001105/0.002977 at It 1/2/3 regardless of lr) — proof the lr never reached the optimizer.
Strong candidate cause for the "even Theta blows up after init" instability: the effective step was
10x too large, matching the supervisor's "learning step too high can blow up NN, last 0".
**Ruled out**: (a) `param_groups` lr override after `build_model` in each caller — works but must be
repeated in every entry point and diagnostic, and hides the real fix. (b) Editing Jan's `fit()` to
honor `optimizer_kwargs` when `init_model_done` — touches shared framework code affecting MSD/Bouc-Wen/
all systems, higher blast radius; the wrong assumption actually lives in OUR `build_model` (it creates
the optimizer early, then relies on `fit` to set lr). (c) Not calling `init_model` in `build_model` and
deferring to `fit` — `build_model` must init the nets so the post-build encoder-init x0 capture and
baseline sims work; deferring breaks those. Chosen fix passes lr where the optimizer is actually
created, in our own code, minimal diff.
**Constrains**: (1) All callers (`gantry_interconnect_dynamic.py`, `diag_theta_lr_sweep.py`,
`diag_xy_routing_blowup.py`) set `hp['lr']`/`cfg.lr` before `build_model`, so this single fix repairs
all of them. (2) `hp['lr']` MUST be set before `build_model`; setting it only via `fit`'s
`optimizer_kwargs` remains dead — do not rely on it. (3) **All prior gantry run-table results were at
lr=1e-3, not their stated lr; re-interpret accordingly and re-run any lr-sensitive conclusion.**
(4) The sibling pipelines (`lpv_lfr_baseline/`, `scripts/gantry/real-data-verification/`) likely share
the `init_model`-before-`fit` pattern and the same stranded-lr bug — audit separately before trusting
their lr settings.

### [D-100] Unified config: all parameters in one RunConfig; hp is a derived view (supersedes the D-092 split)
**Date**: 2026-07-08
**What**: `RunConfig` now holds EVERY user-tunable parameter, including the model/training
hyperparameters that D-092 had left in a separate `default_hp(cfg)` dict (`nx_ann`,
`n_nodes_per_layer`, `n_hidden_layers`, `up_sample`, `batch_size`, `lr`, `epochs`,
`nf_seconds`). The entry file `gantry_interconnect_dynamic.py` constructs one object with all
fields visible. `nf` and `na_nb` are derived properties (from `nf_seconds`/`ts_new` and Jan's
`(nx_phys+nx_ann)*2+1` rule) with optional direct overrides `nf_override` / `na_nb_override`
(None = derive). `cfg.hp` is a read-only property returning the legacy dict (exact keys/order),
so the ~67 `hp['...']` call sites, the checkpoint `.npz` meta, the results-npz `config`/`hp`
JSON, and resume are all unchanged. `default_hp(cfg)` remains as a one-line backward-compat
accessor returning `cfg.hp` (used by `diag_xy_routing_blowup.py`); the entry file and
`gantry_optuna.py` now use `cfg.hp` directly.
**Why**: The D-092 split left the setting surface in two places; the user could not set all
parameters from the entry file and found the separate dict messy ("why is this still a separate
dict compared to all the parameters?"). One object, one place, one source of truth.
**Verification**: Stage A re-run bit-exact vs the unchanged legacy copy after the change
(`cfg.hp` byte-identical to old `default_hp` incl. JSON; all model tensors, both RNG streams,
66626 training windows, and first-batch loss hex `0x1.5ddeac0p-21` identical). Behavior-preserving.
**Ruled out**: (1) Plain dict of all params in the entry file (no dataclass) -- loses the frozen
guarantee and derived properties. (2) Keeping two objects both edited in the entry file -- still
two things to reconcile. (3) Making `nf`/`na_nb` plain settable numbers -- loses the physical
`5*tau` default and Jan's-rule default; the override fields cover the "set directly" need.
**Constrains**: `cfg.hp` key set/order remains the frozen checkpoint/npz contract. New tunables
go on `RunConfig` as fields; if they belong in the persisted hp dict, add them to the `hp`
property in the same key position.

### [D-099] Anti-aliasing is a non-issue for the simulated dataset; keep asymmetric resampling (block-mean u, point-sample y/states)
**Date**: 2026-07-08
**What**: Empirically scoped the supervisor's anti-aliasing concern (07-07) with
`scripts/gantry/augmentation-error/diag_downsample_spectra.py` — 20 kHz Welch PSDs of
`y`, `x_logical` (6 ch), `delta_a`, and `u_total` over the worst-case records
(E1_resonance_sweep, E3_aprbs_above, E4_multisine_off, T11_aprbs_100, V1) in BOTH modes.
Metric: fraction of power above the new Nyquist (2 kHz). Result: **every signal is band-limited
far below 2 kHz.** Worst-case `frac_above`: `y`/states = 2.5e-8, and — critically —
`u_total` = ~4e-14 (machine floor). All PSDs roll off steeply and hit a flat numerical/solver
noise floor by ~500 Hz, ~15 decades of headroom below Nyquist. Decision: **do NOT add
`resample_poly`/`decimate` to the simulation pipeline; keep point-sampling `y`/states (exact
here) and keep the block-mean for `u` unchanged.**
**Why**: Point sampling folds only the energy above 2 kHz — which is absent — so "point sampling
is exact" (data.py:101) is now verified to ≤2.5e-8, not assumed. The block-mean `u` fix (D-087)
is retained but its justification is corrected: `u_total` has NO HF content either, so its benefit
is NOT anti-aliasing — it is a DC/area-consistency (impulse-equivalent ZOH reduction) effect that
matters only because the K=0 axes are open-loop integrators that accumulate any systematic
force-mean offset. This also **falsifies the handoff premise** that "the 20 kHz ZOH controller
puts step-harmonic energy far above 2 kHz in u": the controller force is smooth (band-limited
excitation), not a sample-rate square wave.
**Ruled out**: (a) Switching everything to one `resample_poly` — adds filter transients, edge
effects and u/y group-delay bookkeeping for zero measured benefit, and risks reintroducing a u/y
phase mismatch. (b) Replacing block-mean-u with plain `[::D]` — would reintroduce the D-087 open-loop
drift (Y −3.5e-4 m). (c) FIR-decimating the FD-derived velocity states — perturbs the fragile
boundary velocities used for interior-K0 seeding for no benefit.
**Constrains**: The anti-alias machinery belongs to the **real-data (Telica) pipeline**, not the
simulation pipeline. Real logs carry measurement noise, quantization, and true HF resonances that
WILL alias under `[::D]` and MUST be anti-alias filtered (`scipy.signal.resample_poly`, same
zero/linear-phase filter on u and y) before decimation. This conclusion is data-specific to the
noiseless simulation; re-scope with the same diagnostic if the sim excitation band or FS_ORIG changes.

**Measurement noise does NOT change this conclusion (supervisor 2026-07-08).** The noise model is
**measurement noise only, added post-hoc to the output, NOT injected through the closed loop** —
identical to Jan's ECC SNR convention (`msd_ndof_interconnect_dynamic.py:46-47`,
`train_data.y += np.random.normal(0, sigma_n, ...)`). Implementation:
`data.py:150-151` adds `sd.y = sd.y + N(0, sigma_n)` **after decimation, at the 4 kHz working rate,
on the measured output `y` only** (`sigma_n = rms(y)·10^(-SNR/20)`, the acceptance floor, D-078).
Direct supervisor instruction: *"only measurement noise. DONT ADD IN THE CLOSED-LOOP. SHOULD NOT GO
THROUGH THE CLOSED-LOOP. Same as how jan does it with his SNR."*

Consequence for the anti-alias filter (the reason this is recorded here): because the noise is
generated at the 4 kHz working rate and never passes through the loop or the 20 kHz plant, it is
**white only up to the 4 kHz Nyquist by construction — it has no energy above 2 kHz to fold.** So
turning SNR on does NOT reintroduce an aliasing problem and does NOT require `resample_poly` in the
simulation pipeline. **D-099 holds unchanged with measurement noise on.** The ONLY scenario that
would reactivate the anti-alias requirement is noise injected *before* decimation (at 20 kHz, or
in-loop) — which the supervisor has explicitly ruled out for this simulation and which remains a
real-data (Telica) concern only. Future sessions: do not add an anti-alias filter to the sim
pipeline "because we added noise"; the noise is post-decimation output noise and is band-limited by
construction.

### [D-098] Wire the oracle into evaluation tables/error-trace + per-record coverage; cache deferred
**Date**: 2026-07-07
**What**: `evaluation.py` now runs the FP+MSD oracle (D-097) on the val and test records (true-x0,
interior-K0 seed, `hp['up_sample']`, pipeline rate) and shows it as a labeled reference column in
the A-tables and a dotted line on the error-trace plot; oracle NRMS/RMS/trajectory added to the
results npz (conditional keys). `_print_same_init_comparison` generalized to take a list of
reference columns (true-x0 baseline + oracle). Entry `main()` prints per-record augmented NRMS
over BOTH val and test (was test only). Step 6 of `docs/eval-restructure-plan.md`.
**Why**: The oracle bounds the achievable error (best-case augmentation) and is shown on the same
rate/up_sample footing as baseline/augmented (fairness), but as a true-x0 REFERENCE, not a same-init
"+%" target (the encoder cannot observe the absorber). Per-record coverage surfaces where
augmentation helps/hurts across the operating range, not just on V1/E1.
**Ruled out / DEFERRED -- the reference cache**: The plan's shared trajectory cache (fingerprint-keyed,
append-only) for the training-independent references is DEFERRED. Rationale: D-089 moved all baseline
sims to AFTER training, so there is no longer a pre-training wait; the true-x0 baseline + oracle sims
total ~10-30 s and run once post-training. Caching would save that only on repeat runs of an identical
config, at the cost of fingerprint-correctness / stale-cache risk (the exact class flagged as
dangerous for a fair comparison). Low value now, non-trivial risk -> not implemented; revisit only if
per-run eval time becomes a real bottleneck.
**Constrains**: Oracle failure is caught and reported (never breaks eval). npz gains optional
`nrms_oracle`, `rms_oracle`, `y_hat_oracle`, `nrms_oracle_test`. The error-trace baseline/oracle
lines are now present; the NRMS-summary bar figure remains optional/unimplemented (per-record numbers
print to the log). This completes the eval-restructure plan (Steps 1-6 = D-093..D-098).

### [D-097] Python 8-state FP+MSD oracle model (gantry_dynamic/oracle.py)
**Date**: 2026-07-07
**What**: New `gantry_dynamic/oracle.py`: the FP baseline plus the true hidden absorber, an RK4
port of `Matlab-scripts/Augmentation/gantrySystemExtended.m` (state `[X,Th,Y,da, dX,dTh,dY,vda]`,
nonlinear M(Y,da), logical-coordinate force). Simulates a record open-loop from the true interior
state and returns the stage-coordinate output + delta_a. MSD params from ma_frac=0.10
(project_gantry_msd_params; the mat does not store them): ma=1.01, mh_rigid=9.09, fa=150,
ka=ma*(2pi*fa)^2, ca=2*0.05*sqrt(ka*ma), L0=0.10. Step 5 of `docs/eval-restructure-plan.md`.
**Why**: A best-case "augmentation target" reference: how well the FP + true absorber reproduces
the data. Makes "augmented sitting on baseline" read as "ANN did nothing" and bounds the achievable
error. Verified before wiring in (`scripts/gantry/augmentation-error/diag_oracle_vs_data.py`):
native 20 kHz isolates model correctness (delta_a ratio 3.4e-5, X 0.02, Y 0.19 -> model is exact);
pipeline-matched 4 kHz/up_sample=2/block-mean-u confirms fairness at run conditions (delta_a 0.5%,
Y RMSE 2e-6 m vs baseline ~2e-4 m, ~100x below baseline). Two D-087-consistent facts baked in:
seed from an interior sample (sample-0 qdot is a one-sided gradient() artifact); up_sample=1 is
already converged at 20 kHz (up_sample=4 identical), residual is the ZOH-force replay limit.
**Ruled out**: Reading the MATLAB plant at run time (no MATLAB dependency in the Python pipeline);
adding ma_frac to RunConfig (kept as an oracle-module constant, documented, single source);
finer up_sample/native rate in the pipeline oracle (fairness: it MUST match cfg.up_sample and
cfg.fs_new like baseline/augmented; only the standalone diagnostic goes finer -- lessons.md).
**Constrains**: Oracle uses `hp['up_sample']` and `cfg` rate, block-mean u, interior-K0 seeding
-- identical footing to the same-init comparison (D-094). Wiring into the tables/error-trace and
the reference cache is D-098 (Step 6). As a true-x0 reference it is a labeled row, not a same-init
"+%" target (the encoder cannot observe the absorber states).

### [D-096] Diagnostic plots: dotted nf-RMS on the loss plot, error-trace, error-spectrum, plots/ subtree
**Date**: 2026-07-07
**What**: `evaluation.py:_make_plots` now (1) routes all figures into a per-run `plots/` subtree
(`plots/val/` for record-specific ones); (2) adds the **val nf-window RMS as a dotted line** on the
loss convergence plot next to the solid sim-RMS selector and dashed train loss (y-axis relabeled
RMS [m]; the two val curves are the same deepSI physical-meter unit, D-095); (3) adds an
**error-vs-time** plot (residual `y_model - y_data` per axis, augmented encoder-init and true-x0
init) that reveals sub-mm drift/absorber structure the overlay hides; (4) adds a **Y error
spectrum** marking the 130-180 Hz absorber band and ~157 Hz resonance. Step 4 of
`docs/eval-restructure-plan.md`.
**Why**: The existing overlay hides a 4e-4 m residual on a 0.24 m axis; the error trace makes it
visible (ramp=drift, oscillation=absorber). The Y spectrum is direct absorber evidence: if
augmentation removes the ~157 Hz peak, the ANN learned it (with the ANN at zero it is fully
present). The dotted nf-RMS answers "good on the training horizon while full-traj rises?"
**Ruled out**: Separate metric-over-epochs figure (folded into the loss plot); baseline/oracle
lines on the error trace and the val+test NRMS-summary bars (deferred to Step 6 - they need the
cached baseline trajectory and per-record coverage sims not yet plumbed into `_make_plots`).
**Constrains**: nf-RMS plotting aligns to the tail of `epoch_id_full` (resume-safe). Existing PNG
filenames are unchanged, only relocated to `plots/`. Step 6 adds the baseline/oracle error-trace
lines and the coverage summary.

### [D-095] Per-epoch nf-window RMS diagnostic alongside the sim-RMS selector
**Date**: 2026-07-07
**What**: `training.py` records a second validation curve during training: the nf-window RMS (same
nf as training, encoder re-init per window), alongside the framework's full-traj sim-RMS. deepSI
validates once per epoch via `self.cal_validation_error` (concurrent_val=False), so a temporary
instance wrapper (`_install_nf_val_probe`) piggybacks the extra metric into `fit_sys.Loss_val_nf`
and returns the selector value untouched. Restored after training. Returned via the diag dict for
plotting (Step 4). Step 3 of `docs/eval-restructure-plan.md`.
**Why**: The sim-RMS selector currently picks epoch 0 (training makes full-traj worse). The
nf-window curve measures what training actually optimizes (its 0.1 s horizon), distinguishing
"wrong selector / horizon" (nf-RMS improves while sim-RMS rises) from "not learning" (both rise).
Both metrics are deepSI physical-meter RMS (`'sim-RMS'` -> `System_data.RMS`; nf via
`n_step_error(mode='RMS')`), so they are directly comparable. **Selection and `bestfit` are
untouched** (the wrapper returns the original selector value); this is diagnostic only.
**Ruled out**: Changing the selector to windowed now (deferred until the curves are seen);
epoch-by-epoch fit loop (invasive, risks framework state); probe `stride=cfg.stride`
(~40x sim cost). Chose non-overlapping windows `stride=nf` (~1 sim-pass; the average windowed RMS
is near-invariant to stride, more windows only reduce estimator variance).
**Constrains**: Adds ~one sim-pass to per-epoch validation time (acceptable; diagnostic). Valid
for `concurrent_val=False` (our config); the wrapper would not propagate to concurrent-val remote
workers. On resume, `Loss_val_nf` covers only this call's epochs (tail of `Loss_val`); Step-4
plotting aligns to the tail. `n_step_error` runs under `torch.no_grad()`; failures record NaN and
never break training.

### [D-094] Same-init augmented-vs-baseline reporting + RMS/NRMS + verdict + grouped output
**Date**: 2026-07-07
**What**: `evaluation.py:evaluate_and_save` now compares the augmented model (encoder-init) against
the **encoder-init** baseline (`baseline_encinit_nrms`), not the true-x0 baseline; the true-x0
baseline is kept as a labeled reference column. Every metric prints both **RMS [m]** and
**NRMS [-]**. A **verdict** line is printed first (ANN active? via aug-state RMS; same-init
improvement %). Output is grouped under section headers (A. Model / B. Encoder / C. Augmentation /
D. Training health; B and D headers added in `main()` before `state_recovery_diagnostic` and the
grad-norm block). Step 2 of `docs/eval-restructure-plan.md`.
**Why**: The prior table paired augmented (encoder-init) against the true-x0 baseline (different
init), so its "+77%" was an initialization artifact, not the ANN — provably 100% artifact when the
ANN is at zero (augmented == encoder-init baseline exactly). Same-init pairing isolates the ANN's
actual contribution (currently +0.0%, honest). RMS[m] is the physical/defensible quantity
(compares to the noise floor sigma_n); NRMS enables cross-channel comparison. Reporting-only: no
sims added (both baselines already computed in `main()`), plots and the results npz unchanged
(still receive the true-x0 `baseline_nrms`).
**Ruled out**: Dropping the true-x0 baseline (keeps value as an oracle-init reference); computing a
single "+%" across mixed inits (the artifact being fixed).
**Constrains**: When the ANN starts learning, the headline % reflects the ANN alone; the oracle
column (D-097) and per-record coverage (D-098) extend this same table. Falls back to the true-x0
baseline for the comparison when `baseline_encinit_nrms` is absent (non-linear_map encoder).

### [D-093] Per-run output subfolder + config.json snapshot
**Date**: 2026-07-07
**What**: Entry `main()` writes all run artifacts to `save_dir(cfg)/<run_id>/` instead of
`save_dir(cfg)/`. `save_dir(cfg)` stays the run FAMILY dir (reserved as the shared reference-cache
home, D-098). A `config.json` (`config_json_dict(cfg)` + `hp` + `run_id`) is written at the run
folder root. Step 1 of the eval-restructure plan (`docs/eval-restructure-plan.md`).
**Why**: Runs currently drop model/npz/plots/checkpoint into one shared folder with `run_id` baked
into every filename — hard to browse, archive, or delete a single run. `sdir` already threads into
training (checkpoint_dir), `evaluate_and_save`, `state_recovery_diagnostic`, and the grad-norm save,
so the subfolder is a one-line change; nothing else moves. `RESUME_CHECKPOINT` is a full path, so
resume is unaffected. config.json makes each run self-documenting at a glance.
**Ruled out**: Per-run folder inside filenames only (status quo — cluttered); writing config.json to
the family dir (would be overwritten per run).
**Constrains**: Downstream steps write into the run folder; the shared reference cache (D-098) lives
in the family dir `save_dir(cfg)`, not the run folder. A run that crashes still creates its folder.

### [D-092] Behavior-preserving restructure of gantry_interconnect_dynamic.py into a package
**Date**: 2026-07-07
**What**: `scripts/gantry/gantry_interconnect_dynamic.py` (1231 lines) is restructured into a
package `scripts/gantry/gantry_dynamic/` (config, data, model, baselines, diagnostics,
evaluation, training) plus a thin entry file at the unchanged path holding the run knobs and
`main()` under a `__main__` guard. Config boundary: a frozen `RunConfig` dataclass carries
experiment identity (MODE, SNR, STRIDE, FS_NEW, ENCODER_INIT, ...; serialized to the npz
`config` JSON); `hp` stays a plain dict with exactly the current keys (incl. `up_sample`)
because it is JSON-round-tripped in checkpoints and results npz, and resume of existing
checkpoints must keep working. Module-level globals (~20) become two explicit objects
(`DataBundle`, `Norm`) passed as parameters. Duplications factored: shared encoder-window
builder, shared stepwise open-loop rollout, shared affine-map R2. `evaluate_and_save` splits
into metrics / plots / npz-save internals with identical orchestration order. Checkpoint I/O
extracted from `train_model_with_diagnostics`; formats frozen. Importers
`gantry_optuna.py` and `diag_nf100_fullrouting.py` updated to the new API. The restructure is
strictly behavior-preserving: numerics, RNG consumption order, D-087 data conditioning,
the training call, prints, plot files, and all npz/checkpoint keys are unchanged.
**Why**: The monolith made nothing importable or testable (importing it triggered a full
training run at import time; D-091's preflight had to duplicate `build_model` for exactly
this reason), config was split over ~15 module constants plus DEFAULT_HP with an unclear
boundary, and a ~330-line `evaluate_and_save` mixed four concerns. The user explicitly chose
the restructure and accepts losing diff-comparability with Jan's ECC reference script; this
supersedes the lessons.md "preserve the reference-script skeleton" rule for this file only.
**Verification**: Stage A (mandatory): harness monkeypatches the deepSI `fit` entry to
capture, at the training call, the fit kwargs, normalization constants, full hfn+encoder
state_dicts, np/torch RNG states, and a deterministic first-batch loss; old vs new must match
bit-exactly (`np.array_equal`, no tolerance). Stage B (recommended): end-to-end 1-epoch CPU
run of both versions, comparing all output npz files key-by-key. Harness lives in the session
scratchpad, not the repo.
**Outcome (verified 2026-07-08)**: Stage A passed bit-exactly (all fit kwargs, 4 norm
constants, 27 hfn tensors, 13 encoder tensors, numpy+torch RNG states, 66626 training windows,
first-batch loss to identical float hex). Full-config confirmation on the cluster: job 69124
(old code) vs 69125 (refactored), both 10 epochs / nf=400 on the same node. Every printed
training loss and Val sim-RMS (It 260 -> 2600), bestfit=0.00017, R2_linmap
(delta_a=+0.0060, vdelta_a=+0.1640), and all downstream NRMS/RMS/baseline/state-recovery/
gradient-norm tables were identical. Only differences: job-id in filenames, wall-clock seconds
(20232 vs 20369 s), time-profile percentages (measurement noise), one tqdm/print interleaving
artifact, and a cosmetic path string (old `scripts/gantry/../../data`, new abspath-collapsed
`data` -- same resolved location).
**Ruled out**: (1) Single-file restructure with `main()`: fixes side effects but keeps a
1200-line file (user chose the package). (2) Converting `hp` to a dataclass: breaks the
JSON/npz/checkpoint contract and resume of existing checkpoints. (3) Moving `up_sample` out
of `hp`: same contract reason. (4) Keeping the old file as a legacy sibling in the repo:
git history + scratchpad snapshot suffice.
**Constrains**: The pipeline is now multi-file; cluster syncs must include the whole
`scripts/gantry/gantry_dynamic/` directory. Diagnostic scripts that previously copied
config/normalization blocks "verbatim from gantry_interconnect_dynamic.py" should import
from `gantry_dynamic` instead going forward. npz keys, checkpoint `.pt`/`.npz` layout, and
the `hp` dict keys remain a frozen contract for any future edit. The D-088 pipeline-table
rule applies: entry-point path is unchanged, so CLAUDE.md needs no edit.

### [D-091] WITHDRAWN — Pre-flight gate script for augmentation training runs
**Date**: 2026-07-07
**Status**: Withdrawn same day. The script was written but never run; the user rejected it
on review ("I'm not sure about this preflight script" -> remove). `scripts/gantry/preflight.py`
deleted; the CLAUDE.md run-discipline rule now references D-090 only. The entry below is kept
as the design record in case the idea returns (e.g. before the Aspect 3 beta sweeps).
**What**: New standalone diagnostic `scripts/gantry/preflight.py`, run before committing a
cluster training job. Four checks with PASS/WARN/FAIL verdicts, results printed and saved as
JSON to `simulations/gantry_subnet/diagnostics/`: (1) measurability: baseline FP residual on
V1 (true-x0 open-loop sim) vs the D-078 noise floor sigma_n; (2) gradient routing: one
forward+backward at epoch 0 on a small batch, per-group gradient norms (encoder / hfn),
calibrated against the documented dead-zone incident (ANN grad 1.04e-2 dead vs 2.85e-1
healthy, diag_gradient_routing); (3) encoder-init quality at FS_NEW vs the 20 kHz native
reference (per-channel state NRMS of the untrained reconstructability map); (4) absorber
excitation: delta_a std per training record (delta_a ~ 0 means nothing to learn). Checks 1
and 2 survive the hardware transition (data-derived); checks 3 and 4 are marked
simulation-only (they need ground-truth states / delta_a).
**Why**: Three documented incidents wasted cluster runs on conditions checkable before
launch (C_aug dead zone, 200 Hz encoder-init trap, job 68458). Consolidating the fragments
into one pre-launch gate converts prose checklist items (CLAUDE.md stance,
control-reasoning Section 7) into an executable that a session cannot forget to apply.
**Ruled out**: (1) Importing `build_model` from `gantry_interconnect_dynamic.py`: the
training script executes at module level (data loading + training), so importing it runs it;
a __main__ guard refactor would restructure the experiment file (rejected per the
no-scaffolding lesson). The preflight duplicates the minimal build per the diagnostic
independence lesson (construct the component from scratch). (2) Hard thresholds from
invented numbers: verdicts are calibrated on documented incident values and labeled
HEURISTIC, or expressed relative to the native-rate reference (encoder check).
**Constrains**: Config constants (MODE, FS_NEW, SNR, hp) are duplicated from the training
script header and must be kept in sync manually; the script prints the values it used so a
mismatch is visible. Preflight is advisory: a FAIL does not block anything mechanically.

### [D-090] Hypothesis-per-run discipline for training runs
**Date**: 2026-07-07
**What**: Every training run with a new hypothesis or new config gets a row in the run table
(`docs/gantry-augmentation-problem-log.md`, Section 12) BEFORE launch, stating the hypothesis
the run tests; the outcome is added to the same row after the run. Trivial re-runs (same
hypothesis, same config) do not get rows. Enforced via a one-line Workflow rule in CLAUDE.md
and a convention note at the top of the run table.
**Why**: The run table is the registry of dead hypotheses; when it is stale, sessions
re-derive and re-test failure hypotheses that are already answered. Writing the hypothesis
before launch forces every run to be a falsifiable experiment, and the maintained table
becomes the experimental narrative for the thesis (writing phase W21-23). Near-zero cost.
**Ruled out**: A separate run-log file: the problem log Section 12 table already exists and
is referenced; a second location would split the history.
**Constrains**: Launching a run without a hypothesis row is a process violation; sessions
asked to launch runs must add the row first.

### [D-089] Baseline FP sims moved post-training; untrained-encoder x0 captured pre-training
**Date**: 2026-07-07
**What**: In `gantry_interconnect_dynamic.py`, the four full-record baseline simulations
(`compute_baseline_fp_nrms`: val/test x true-x0/encoder-init) move from before
`train_model_with_diagnostics` to directly after it. Pre-training, only the untrained-encoder
initial-state estimates are captured (`_encoder_init_state`, one no-grad forward per record);
the encoder-init baseline sims consume those captured vectors post-training.
**Why**: The four sims are ~2 min each (~8-10 min before the first epoch), delaying visible
training start on the cluster; nothing in training consumes their results (they feed only
`evaluate_and_save` and the convergence plot). Correctness: `compute_baseline_fp_nrms` builds
its own fresh `Gantry_State_Block` and never touches `fit_sys`; the sims draw no randomness, and
the encoder capture stays at the same pre-training point in the RNG stream — training and all
reported numbers are bit-identical to the previous ordering.
**Ruled out**: Skip-flag / env hook (operational scaffolding in an experiment script,
lessons.md); disk cache with config fingerprint (deferred — only pays off on repeat configs and
the encoder-init cache key is fragile); batching the four sims (optimization, separate concern).
**Constrains**: Log order becomes training -> baselines -> test NRMS -> evaluation. A run that
crashes during training leaves no baseline numbers in its log.

### [D-088] Context system: control-reasoning reference doc + CLAUDE.md identity/stance sections
**Date**: 2026-07-07
**What**: (1) New reference doc `docs/control-reasoning.md`: project identity, three-pipeline
map with signal chains, plan-vs-code status table, expanded 8-item control reasoning checklist,
Lambda-vs-Pi interpretability section (standalone-baseline negation test, the three thesis
extensions to the Gyorok method), identifiable-combination table, diagnosis-order pointer.
(2) CLAUDE.md gains two always-on sections placed after Hard Constraints: "Project Identity"
(thesis one-liner + three-pipeline table) and "Control Engineering Stance" (8-item checklist as
one-liners, closing pointer to the doc). Compressions elsewhere (quote-verification 10 -> 3
lines, ownership table -> 2 lines, workflow subagent-trigger block dropped) keep net size
roughly flat. (3) Key File Map extended with the two key training scripts, the new doc, the
research plan PDF, and the literature folders.
**Why**: CLAUDE.md contained process rules but no domain identity and no control-engineering
stance; every session re-derived the project framing from scattered docs and tended to answer
from generic ML knowledge instead of control/system-identification reasoning. Checklists
transfer to future sessions (and to smaller models) better than prose. Keeping depth in one
on-demand doc (~2.5k tokens when read) instead of always-on context avoids instruction
saturation.
**Ruled out**: (1) Expanded reasoning content directly in CLAUDE.md: saturates always-on
context and degrades compliance with all other rules. (2) Duplicating the problem log's
failure detail in the new doc: drift liability; the log stays the single owner, the doc only
points. (3) Hook-based stance enforcement: stronger mechanism, deferred until file-based
guidance proves insufficient.
**Constrains**: Rule ownership split: behavioral rules and incident history live in
`tasks/lessons.md`; the domain checklist lives in CLAUDE.md as one-liners; expansions live
only in `docs/control-reasoning.md`. Project Identity holds slow-changing facts only; phase
state stays in `tasks/todo.md`. Restructuring a pipeline now requires updating CLAUDE.md's
pipeline table and Section 2 of the doc.

### [D-087] ZOH-consistent input resampling (block mean) + interior-sample true-x0 init
**Date**: 2026-07-07
**What**: Two data-conditioning fixes in `gantry_interconnect_dynamic.py`. (1) Downsampling of
the plant force 20 kHz -> FS_NEW uses the per-interval block mean (`u[:n*D].reshape(n, D, nu).mean(axis=1)`)
instead of point sampling `u[::D]`; outputs and states stay point-sampled (`y[::D]`, exact for
states). (2) All "true x0" open-loop simulations (the x_logical-init model sim in
`evaluate_and_save` and the true-x0 baselines in the main block) start from the interior sample
K0 = cheat_n with state `x_logical[K0]`, instead of sample 0.
**Why**: The slide-21 "open-loop problem" (meeting 07-07) decomposes exactly into these two
artifacts, amplified by the K=0 axes (any low-frequency input/init error integrates into a
permanent offset tau*dv, tau = m/c: X 1.55 s, Y 1.01 s; verified to 3 digits by dv injection).
(a) ~75%: `u_total` is ZOH at 20 kHz (discrete controller), so the exact FS_NEW input is the
mean force per hold interval (impulse equivalence); `u[::D]` leaves a nonzero-mean force error.
V1 baseline-only open-loop offset: Y -3.47e-4 m / X +6.1e-5 m with `[::D]`, -2.8e-9 / -3.0e-8 m
with block mean (and -2.7e-9 / -2.7e-7 m at native D=1). (b) ~25%: `gtd_save_record.m` computes
`qdot_logical` with `gradient()`, one-sided at sample 0; V1 starts at rest yet stored v0 is
[9.5e-6, -6.2e-5, -1.05e-4], contributing tau*dv = -1.06e-4 m on Y. Interior samples carry
central differences at 20 kHz (accurate on noiseless positions). Evidence:
`scripts/gantry/augmentation-error/diag_openloop_x0.py`, `diag_onestep_residual.py`; artifacts
in `simulations/gantry_subnet/diagnostics/` (openloop_x0_V1, onestep_residual_V1,
openloop_x0_V1_20kHz, openloop_x0_V1_4kHz_uavg). Sum of the two contributions matches the
observed offsets within 1% (Y: -1.01e-4 + -3.47e-4 = -4.48e-4 vs -4.46e-4 observed).
**Comparison to Jan's ECC MSD method** (`scripts/ecc_2025/msd_ndof_data_generation_dynamic.py`):
Jan has no resampling step at all — the discrete truth system is simulated at the model rate
(dt=0.02 both), so data and model share one ZOH convention by construction; and he discards the
first multisine period (`train[Ntrain:]`), so no simulation ever starts on a cold-start sample.
This decision restores those two invariants for the gantry pipeline, where the truth is a
continuous Simulink plant logged at 20 kHz.
**Ruled out**: (1) `scipy.signal.decimate` on u — an IIR anti-alias filter distorts an
already-ZOH signal; block mean is exact, not an approximation. (2) Filtering y — states are
point-sampled exactly; filtering would inject phase error. (3) Zeroing v0 at sample 0 —
assumes at-rest records, fails for sweep records; interior-sample init needs no assumption.
(4) Regenerating data with logged Simulink states — valid long-term fix for
`gtd_save_record.m`, but not needed once sims start at K0; defer to next data regeneration.
**Constrains**: All baseline and sim-RMS numbers change (improve); results from runs before
this decision are not comparable. The x_logical-init sim now starts at cheat_n (same instant
as encoder-init — cleaner comparison; its saved `y_hat_xlog` is NaN before cheat_n). The
encoder-init velocity error remains a separate open item (needs the cluster run npz:
`x_enc_phys[cheat_n]` vs mat `x_logical[cheat_n]`).

### [D-086] E1 sinesweep tapered with a fade envelope; delta_a panel added to the plot
**Date**: 2026-07-06
**What**: (1) `make_sinesweep` (E1) now applies a 0.5 s half-cosine fade-in/out to the chirp
amplitude over the active window, instead of switching the 34 N / 130 Hz force on and off
abruptly. (2) `gtd_plot_record` gains a 5th full-width row showing the hidden MSD displacement
delta_a (in micrometres), present whenever the MSD is simulated.
**Why**: The un-tapered chirp slammed the resting system on and off, kicking all modes and
producing large onset/offset transients ("peaks at the start and end of the envelopes") that
buried the actual swept response. The steady-state Y-position response to a ~150 Hz force is only
~micrometres (F/(m*omega^2)), correct physics but uninterpretable in the raw position plot, and
the resonance signature lives in delta_a, which was not plotted. With the taper the response
shows a clean resonance bulge where the sweep crosses ~150-157 Hz; the delta_a panel makes that
bulge directly visible. Sweep rate (5 Hz/s) is slow enough for the Q~10 mode (settling ~0.02 s
vs ~3 s in the resonance band).
**Ruled out**: Leaving the chirp un-tapered (transients dominate, not presentable). Hardcoding a
resonance-crossing marker at 150/157 Hz in the plot (the coupled peak is uncertain; left to the
viewer's eye on the delta_a bulge).
**Constrains**: The fade slightly lowers the active-window RMS below the nominal amp_frac*A_Y
(faded ends carry less energy); pack() still reports the nominal RMS. E1's multisine-row RMS on
the plot reads low anyway because RMS is over the full 12 s record while the sweep fills only the
10 s active window.

### [D-085] Save full 8-state augmented ground truth; vdelta_a by differentiation; 4x3 force plot
**Date**: 2026-07-06
**What**: (1) `gtd_save_record` now saves the full augmented state for encoder pre-training:
`x_logical` (6 baseline states, logical coords) plus `x_aug = [delta_a, vdelta_a]` (the 2 MSD
states), where `vdelta_a = gradient(delta_a, ts)`. Previously only `delta_a` (7 of 8 states).
(2) `gtd_plot_record` splits the forces into three separately-scaled rows (total / feedback /
multisine) in a 4x3 grid (positions + 3 force types), so the ~30 N multisine is visible instead
of buried under the ~300 N feedback.
**Why**: The hidden MSD is second-order, so the true augmented state has 8 components; encoder
pre-training (Donor A/B save true states) needs both delta_a and its velocity. Velocities are
obtained by differentiation to match the reference generators, which pull only positions
(q_aug, delta_a) from the model To Workspace and differentiate all velocities; noiseless 20 kHz
data makes gradient exact. Overlaying total/feedback/multisine on one axis hid the small
multisine on moving records; per-type rows fix it.
**Ruled out**: Routing vdelta_a to a new To Workspace block (the model is ours, in
Matlab-scripts/Augmentation, not read-only, so it is possible, but it would be the only
velocity in the schema coming from the model rather than differentiation, an inconsistency for
no accuracy gain in noiseless sim). Adding a force PSD panel (deferred by user: revisit once the
multisine is not visually noisy).
**Constrains**: Baseline states saved in LOGICAL coordinates (project convention: reference
generators save x_logical + stage-coord y); if the training pipeline expects stage-coord states,
both old and new data would need transforming. amp_rms has mixed units [N, N*m, N] (A_anti is a
torque, D-080), documented in the save header.

### [D-084] A_anti sized as a modest fixed torque capped by the yaw budget, not sized to fill it
**Date**: 2026-07-06
**What**: `gtd_size_anti_amp` now returns `A_anti = min(cfg.A_anti, budget_cap)`, where
`cfg.A_anti = 0.5*A_sym*Lb` (a fixed torque chosen so the anti channel contributes the same
per-rail force RMS as the symmetric channel) and `budget_cap = yaw_budget / yaw_peak_per_unit`.
The 2 mm yaw budget is a CEILING that can only scale A_anti down, never a target to fill.
Reverses the original D-081 implementation, which set `A_anti = yaw_budget / yaw_peak` (fill).
**Why**: Filling the 2 mm budget in the augmentation band (130-180 Hz) demanded kilonewtons of
anti force, because moving real mass 2 mm at ~150 Hz costs force ~ omega^2 (order of magnitude:
2 mm yaw at 150 Hz => ~9000 N*m => ~thousands of N per rail). Observed on T9: FX1/FX2 > 1000 N
while FY ~ 150 N (the mirror-image X-only signature of a dominant anti channel). Worse, the
anti/yaw channel excites the theta mode, not the Y-axis hidden MSD, so the force was both huge
and irrelevant to the augmentation target. The forces still passed `gtd_enforce_limits` because
1-2 kN is within the TELICA hardware ceiling [2000,2000,1420] N; the limit check is "won't break
the machine", not "sensible excitation", which is why activation-based sizing (GATE-2) is needed.
**Ruled out**: Filling the budget (original). Driving anti to zero (kept a modest level for MIMO
identifiability, but it is now cheap and could be zeroed for the augmentation track later).
**Constrains**: A_sym=40, A_Y=30 N remain GATE-2 defaults, still unvalidated by a delta_a
activation diagnostic. The cap essentially never binds at 130-180 Hz (modest torque produces
microns of yaw), so anti force is now comparable to sym force rather than kilonewtons.

### [D-083] Phase 4 Simulink integration: base-workspace contact contained in `gtd_run_simulation`
**Date**: 2026-07-06
**What**: The generator's Simulink call runs through `gtd_run_simulation`, which pushes every
model input to the BASE workspace via `assignin` and launches the run with
`evalin('base','sim(...)')`, then fetches `q_aug`/`delta_a` (MSD) or `q1` (baseline). It runs the
model twice for the MSD case (with and without multisine, informativeness baseline), swapping `mh`
to `mh_rigid` during each run. `gtd_enforce_limits` does the hard limit check + proportional
scale-down on the LINEAR closed loop (lsim), before Simulink; the scaled force is what gets
simulated. `gtd_save_record` writes the spec-1.12 schema. The driver `generate_trajectory_data.m`
is a thin loop; validation records run in the same loop (distinct seeds already give independent
realizations).
**Why**: Simulink resolves block variables from the base workspace, not a calling function's
locals, so a pure function with local inputs would be invisible to `sim()`. Containing all base
contact in one function keeps the rest pure and the driver thin, instead of making the whole
driver a base-workspace monolith.
**Ruled out**: `Simulink.SimulationInput`/`setVariable` (needs every model variable enumerated and
the output-return config confirmed; not verifiable here). Inlining the sim in the driver script
(reproduces the monolith). A pure `gtd_run_simulation` with local variables (invisible to `sim`).
**Constrains**: The base variables the `gantry_additional_state_2025a` model references are
inferred from the working `generate_oscillatory_multisine_data.m`; `push_params` sends a superset.
If `sim` errors "Undefined variable X", add X to `push_params`. Assumed model outputs: `q_aug`,
`delta_a` (MSD); `q1` (baseline). `gtd_check_sim` smoke-tests one record to surface a missing name
before the full run.

### [D-082] Section 3 of Jan writeup: ANN presented as `phi_aug`, name-only (no `W_1/W_2`)
**Date**: 2026-07-06
**What**: Rebuilding Section 3 of `docs/jan-augmentation-writeup.tex` around the finished
`jan-blockscheme-v2.pdf` figure (components table 3a / figure interconnection 3b /
dynamic-parallel model 3c). Two notation choices: (1) the learning component is written
`phi_aug` to match the figure, not `N_theta` from the outline; (2) its internals are named
only, as a `tanh` feedforward network with output `w in R^4`, zero-initialised so
`phi_aug ~ 0` at start, with NO explicit `W_1/W_2` layer matrices.
**Why**: (1) The figure is locked/done and is the section centrepiece; text must agree with
it, so `phi_aug` wins over `N_theta`. (2) `Static_ANN_Block` builds a `zero_init_feed_forward_nn`
with 2 hidden layers x 64 nodes; the outline's single-hidden-layer sketch
`w = W_2 tanh(W_1[.]+b_1)+b_2` would misstate the depth, which Jan could catch. Lowercase `w`
is kept because it is Jan's genuine LFR interconnection-channel signal (Eq. 4), used verbatim
in the framework code (`blocks.py` `forward(z)->w`, `nz/nw`); capital `W_1,W_2` are not paper
notation and were dropped.
**Ruled out**: `N_theta` symbol (would force a `phi_aug == N_theta` aside or a figure edit);
the `W_1/W_2` single-layer formula (wrong depth, introduces non-paper symbols).
**Constrains**: The top Notation table now declares `phi_aug`, `w`, `psi`. Any future change to
the ANN architecture must keep the "name-only, paper-altitude" presentation unless Jan asks for
internals. Writeup compiles to 4 pages.

### [D-081] Multisine layer: purpose-built `gtd_make_multisine` with IFFT synthesis and yaw-budget A_anti sizing
**Date**: 2026-07-06
**What**: `gtd_make_multisine` generates the injected stage force per record. Design points:
(1) It is self-contained, NOT a refactor of the shared `generate_cached_multisine` (reversing
the Phase-3 outline), because the per-channel constrained crest-factor scoring does not fit that
helper's joint-selection contract, and the old script still depends on it unchanged.
(2) Synthesis is by IFFT (`ifft(X,'symmetric')` with unit-magnitude random-phase in-band bins),
not the explicit cosine sum: at period = record the grid is df = fs/N = 1/12 Hz with ~2388 lines,
so cosine-sum is O(N*F) ~ 5e8/signal (minutes); IFFT is O(N log N).
(3) Crest-factor selection keeps the best of `cfg.n_ms_candidates` (=30) random draws per logical
channel: f_sym/f_Y scored on their own signal (stage force is a uniform scaling), f_anti scored
on the closed-loop yaw response via the SISO transfer `H_yaw = [1 -1 0]*sys_cl*([1;-1;0]/Lb)`
(the same P^{-1} anti column verified in D-080).
(4) `gtd_size_anti_amp` sizes A_anti (a torque, N*m) so the anti-driven peak |X1-X2| equals the
2 mm yaw budget exactly (linear loop). Sym/Y coupling into yaw (M_op off-diagonals ~5%) is left
to the 2 mm margin of the 6 mm budget (spec 1.8) and the hard 6 mm enforced downstream in Phase 4.
(5) Realizations cached per record keyed by seed/band/period/Y_op/n_cand.
**Why**: Period = record and fine df make synthesis the cost bottleneck; IFFT removes it. Scoring
f_anti on the yaw response (not raw CF) is what the spec's "CF on the constrained coordinate"
requires, since the anti channel is yaw-budget-limited. A torque-unit A_anti follows directly
from D-080.
**Ruled out**: Refactoring `generate_cached_multisine` (contract mismatch, shared-helper risk).
Cosine-sum synthesis (too slow at period=record). Sizing A_anti on total yaw including sym/Y
coupling (unnecessary; the budget margin covers it, and the hard limit is enforced in Phase 4).
**Constrains**: A_sym=40, A_Y=30 N remain GATE-2 defaults (unvalidated). Phase 4 `gtd_enforce_limits`
must still check the full stage force and total 6 mm yaw and scale down if needed. The multisine
spans the full 12 s record (including holds); only the E1 sinesweep is confined to the active window.

### [D-080] Logical->stage force transform is P^{-1} (f_anti is a yaw torque), not the naive f_sym +/- f_anti
**Date**: 2026-07-06
**What**: The multisine is designed in logical (generalized) force channels [f_sym, f_anti, f_Y]
and injected into the plant as stage rail forces via `gtd_logical_to_stage`, which applies
**F_stage = P^{-1} f_logical**: F_X1 = 0.5*f_sym + f_anti/Lb, F_X2 = 0.5*f_sym - f_anti/Lb,
F_Y = f_Y. Derived from the plant convention (sys = P'*G*P, q_stage = P'*q_logical) by
virtual-work invariance (force map is the dual of the position map). Verified 5 independent
ways in `gtd_check_transform`: P^{-1} vs analytic inverse; f_sym -> equal rails; f_anti ->
opposite rails scaled 1/Lb; F_stage.q_stage = F_logical.q_logical; and DC consistency with the
actual built plant (injecting through the transform into the stage plant equals injecting into
the logical plant).
**Why**: The spec placeholder (F_X1 = f_sym + f_anti) is wrong two ways: it over-scales the
symmetric force by 2x, and it adds a torque to a force. Logical coordinate 2 is the tilt angle
theta ~ (X1-X2)/Lb, so its conjugate force f_anti is a yaw TORQUE [N*m]; dividing by Lb is what
makes it a rail force [N]. Getting this wrong silently corrupts the entire yaw budget and the
anti-symmetric channel, and a shape-only check would not catch it (lessons rule).
**Ruled out**: The naive shape-based map f_sym +/- f_anti (mis-normalized, dimensionally
invalid). Assuming P' or P instead of P^{-1} for forces (P' is the position map, not the force
map).
**Constrains**: `gtd_size_anti_amp` sizes A_anti in torque units [N*m] and must apply the same
P^{-1} before checking the 2 mm |X1-X2| budget. Any amplitude specified "per logical channel"
(spec 1.7: A_sym, A_Y in N; A_anti in N*m) is pre-transform; force-limit and yaw checks are
post-transform on stage forces.

### [D-079] Trajectory-data generator rewritten as modular `gtd_*` functions with three reference shapes
**Date**: 2026-07-06
**What**: The new gantry trajectory-data generator (spec `docs/trajectory-generation-spec-draft.md`)
is built as a set of single-responsibility functions in `Matlab-scripts/Augmentation/data/`
(`gtd_config`, `gtd_build_records`, `gtd_build_plant`, `gtd_make_reference`, `gtd_validate_ref`,
plus later `gtd_make_multisine`, `gtd_run_simulation`, `gtd_enforce_limits`, `gtd_save_record`,
and a thin `generate_trajectory_data.m` driver), replacing the 830-line monolith
`generate_oscillatory_multisine_data.m`. The 22 records (T1-14, V1-4, E1-4) collapse to THREE
reference shapes: `standstill`, `oscillatory`, `aprbs`. Ladder limits are derived from `cfg.lim`
(training top T11 = 75% of the enforced limits, test E3 = 90%). Each mode writes to its own
top-level folder: `data/gantry/matlab/trajectory/<joint|augmentation>/<m50|baseline>/`.
**Why**: The spec is a redesign (fixed-absolute amplitudes, period=record multisine,
logical-coordinate transform, 22-record table), not a delta on the old script. Separating
concerns makes each piece independently verifiable in MATLAB (the P-transform gate and the
Simulink integration become isolated checkpoints), which matters because the assistant cannot
run MATLAB. Y-sweep and lissajous are the same sinusoidal-sum builder with different parameters,
and E1's sinesweep is a standstill motion with a swept excitation, so six spec "classes" reduce
to three motion shapes with no loss.
**Ruled out**: (1) Minimal edit of the existing monolith, rejected because the amplitude
strategy inverts and the trajectory table/timing are rewritten, so a diff-only adaptation would
be more error-prone than a clean decomposition. (2) One reference builder per spec class (six),
rejected as redundant. (3) Hardcoding the ladder numbers, rejected in favour of deriving them
from `cfg.lim` so the enforced limits are the single source of truth.
**Constrains**: Downstream modules read `cfg` and the `records` struct array; the P force
transform stays a derive-and-verify step (D-... / spec 7.5) before `gtd_make_multisine`.
Interpretations pending user confirmation: APRBS X_anti is active only for T12/T14 (off for
T9-T11); APRBS `Y_op` = midpoint of the record's Y range; V2 uses the T10 (60%) jerkTime.

### [D-077] Residual-force diagnostic: dominant model mismatch is inertial-scale (~2x), not friction
**Date**: 2026-07-05
**What**: New real-data diagnostic `scripts/gantry/real-data-verification/diag_residual_force.py`
computes the generalized force the FP model cannot account for from measured motion and
measured applied force: `f_missing = u_applied - [M(Y) qdd + C qd + K q]`, using the model's
own physics matrices (physics.py) and Savitzky-Golay smoothing differentiation for qd, qdd.
Least-squares decomposition of f_missing over moving samples onto [M qdd, qd, sign(qd)] gives,
consistently across 3 operating points (R^2 = 0.997-0.999):
  inertial-scale s = 1 + a:  X ~ 0.48-0.51,  Y ~ 0.65   (dominant term)
  viscous b:  ~60-100 N/(m/s) on X1 and Y, ~10 on X2
  Coulomb c:  ~27-66 N  (present but secondary; below the 136/98 N static-friction spec)
So the FP model's inertial force M(Y) qdd is ~2x (X) / ~1.5x (Y) larger than the applied
MF30*Kt force needed to produce the observed acceleration. Coordinate frame is validated
(corr between model force and applied force 0.93-0.99).
**Why (interpretation, HYPOTHESIS not yet committed)**: M qdd = force is degenerate between
"applied force under-scaled" and "model inertia too large". A pure global current-unit error
(RMS/peak sqrt(2), or x2) would give the SAME scale on all axes; the axis-dependent s (X~0.5,
Y~0.65) instead points at the MF30->N conversion (Kt = MotorForceConst = 109 N/Arms X, 77.6 Y)
possibly being wrong per axis / per motor topology (X rails have multiple sub-motors). This
would corrupt all open-loop training (model driven by mis-scaled force) and explains why
parameter recovery kept trying to halve the masses (compensating the scale error) and why the
optimum is horizon-dependent. Overturns the prior "dominant residual is friction -> go to
augmentation" framing (superseded pending verification).
**Ruled out**: Friction as the primary gap (it is real but secondary, ~30-65 N vs the ~100%
inertial-scale residual). Accepting the e-1 m open-loop error as purely structural before
checking the force-input units.
**Constrains**: Before any further parameter-recovery training, verify the MF30 -> N force
conversion (Kt units: Arms vs peak vs per-motor; number of sub-motors per axis; any amplifier
factor). If the force scale is wrong, fixing it may remove most of the open-loop error without
friction augmentation. Open question for Kamtin: exact definition/units of MF30 and Kt.
**Update (2026-07-05) -- CONFIRMED by linear identification** (`diag_linear_identification.py`,
globally-optimal linear-in-parameters inverse-dynamics fit on all 11 training ops, no training):
the data determines the identifiable mass lumps at ~HALF nominal, rock-solid consistent across
every operating point independently: m_total = 26.2 kg (nom 53.8, ratio 0.49; per-op 26.1-26.4),
mh = 5.9 kg (nom 10.1, ratio 0.59). Adding Coulomb columns changes the residual 24.1% -> 24.0%
and assigns only 2-4 N (vs 98/136 N spec), so FRICTION IS RULED OUT as the primary gap (this
refines/overturns the residual-diagnostic's earlier friction reading, which came from allowing
only a single scalar on the nominal inertia per axis). Since M*qdd = force, half-mass <=> half-
force: the applied force u = MF30*Kt is ~2x too small. The clean factor of ~2, consistent across
all ops, plus Telica.mat showing each X rail split into L+R sub-axes, points to force being
under-counted by the sub-motor count (true force ~ N_submotors * MF30 * Kt). Verdict: the data
is clean and useful but NOT directly fittable by the physical baseline until the force scale is
fixed. Cheap confirmation available: scale applied force by ~2 (or per-axis sub-motor count) and
re-run the linear ID -- m_total should jump to ~53.8 and the residual drop. Other params
(damping/stiffness) are unidentifiable from this data (cond ~2.5e15; rotation mode barely
excited, X1=X2 commanded together) -- only the mass lumps are determined, and they show the 2x.
**Correction (2026-07-05, user)**: the earlier "force is ~2x too small" framing OVER-COMMITTED.
F = m*a is degenerate: the data determining m_total ~ 26 kg is equally consistent with (a) the
real Telica moving mass genuinely being ~26 kg -- the kamtin-fp-model masses come from main.m,
a simulation of possibly a DIFFERENT/earlier gantry, so they need not match the real hardware
(user: "different system") -- or (b) a ~2x force-input scale/units error. The fit cannot
distinguish them. If (a), the data IS directly fittable and parameter recovery is WORKING (it
found the real mass); the "half" is the correct value, not a bug. Resolve with an EXTERNAL
reference: the real Telica moving-mass spec / mechanical drawing vs the main.m assumed masses,
and the MF30/Kt units definition. Do not treat "recovered != model nominal" as automatically a
data error.

### [D-076] Telica validation selector rewired to windowed loss (same measure as training)
**Date**: 2026-07-05
**What**: `run_telica_param_recovery.py` monkeypatches `tr._full_traj_eval` with
`_windowed_val_eval`: sigma-normalized MSE on teacher-forced windows (length SEGMENT_LEN) of
the held-out VAL_SPECS trajectories, returning (normalized-RMS, entries) with the same call
signature so the scheduler, best-checkpoint selection, and both sync/async eval paths are
untouched. Global sigma computed once from the validation q1.
**Why**: The user specified validation should use the same method as training; the framework
default (full-trajectory OPEN-LOOP RMSE) was kept instead (deviation noted only in D-075). On
real data that OL metric is dominated by friction/force-scale drift on the K=0 double-integrator
axes, which no parameter can reduce, so in run 68775 it rose monotonically from epoch 0,
collapsed the ReduceLROnPlateau LR to 1e-5 by epoch 360, and selected epoch 0 (nominal) as
"best". The windowed metric measures the same short-horizon fit training optimizes, on held-out
operating points, so scheduler and selection now track a quantity that can actually improve.
**Ruled out**: Full-trajectory OL as selector (structural drift makes it monotone-degrading);
metres RMS without sigma normalization (Y channel dominates, ignores X fit, differs from
training's per-channel weighting). Closed-loop metric as selector (deployment-relevant but adds
the controller to every eval; kept for FINAL assessment only).
**Constrains**: Validation numbers are now a dimensionless normalized RMS, not metres (printed
`eval_rmse` column relabeled in intent). Final assessment still reports full-trajectory OL and
closed-loop separately. A windowed-val improvement does NOT imply small full-trajectory or
closed-loop error while the D-077 force-scale/friction gap remains.

### [D-001] Target system is the ASMPT dual-gantry (García-Herreros et al.)
**Date**: 2026-03-16
**What**: The sole target system for this project is the ASMPT dual-gantry stage modeled by García-Herreros et al.
**Why**: This is the industrial use case for the graduation project. All other benchmarks (MSD, Bouc-Wen, Cascaded Tanks) are reference implementations of the augmentation framework only.
**Ruled out**: Using MSD or other benchmarks as the target system.
**Constrains**: All new code, data pipelines, and model structures must be built around the gantry system.

---

### [D-002] MATLAB files in `kamtin-fp-model/` are immutable
**Date**: 2026-03-16
**What**: The MATLAB model files defining the FP model structure are the ground truth and must never be modified.
**Why**: They represent the validated physical model from García-Herreros et al. and are the hard constraint that the Python implementation must conform to.
**Ruled out**: Adapting the MATLAB model to fit the Python code. The direction of adaptation is always MATLAB → Python, never the reverse.
**Constrains**: Any Python state-space implementation must reproduce the structure defined in the MATLAB files exactly.

---

### [D-003] Augmentation structure is parallel dynamic LFR
**Date**: 2026-03-16
**What**: The augmentation architecture is a parallel dynamic structure within the LFR framework.
**Why**: Parallel structure is required for orthogonal projection-based regularization (Gyorok et al.), which prevents the learned component from capturing dynamics already described by the baseline. Dynamic (not static) augmentation is needed because cross-coupling and position-dependent flexible dynamics require additional learned states beyond the baseline.
**Ruled out**: Series interconnection (incompatible with orthogonal projection regularization); static augmentation (cannot capture dynamics requiring additional states).
**Constrains**: The LFR interconnection must be realized as a parallel structure. Regularization implementation follows Gyorok et al.

---

### [D-004] Scheduling variable is payload position Y
**Date**: 2026-03-16
**What**: The LPV scheduling variable is the payload position Y.
**Why**: Y enters the inertia matrix algebraically in the García-Herreros model, making it the natural scheduling variable. Since Y is a system state (not an exogenous signal), the formulation is quasi-LPV. Y is directly available from the physical model and does not need to be identified from data.
**Ruled out**: Data-driven scheduling variable identification (not needed here since Y follows from the physics).
**Constrains**: The LPV discretization must handle Y as a state-dependent scheduling variable. Invertibility of the position-dependent inertia matrix must be verified across the full operational range.

---

### [D-006] Python implementation uses stage coordinates
**Date**: 2026-03-16
**What**: The Python discrete-time state-space model is implemented in stage coordinates: states q = [X1, X2, Y, dX1, dX2, dY], inputs u = [F_X1, F_X2, F_Y], outputs y = [X1, X2, Y].
**Why**: Real experimental gantry data is measured in stage coordinates (X1, X2, Y from encoders; F_X1, F_X2, F_Y from amplifiers). The model must match the data — the model is coordinate-independent, so the data determines the choice. The MATLAB model also discretizes in stage coordinates (`c2d(StageCoordinatesSystem, ts, 'zoh')`), providing a direct reference.
**Ruled out**: Logical coordinates [X, Θ, Y] — the augmentation framework trains on measured data, which is in stage coordinates. Working in logical coordinates would require transforming every data sample and adds no benefit.
**Constrains**: The A, B, C, D matrices passed to the augmentation blocks must be in stage coordinates. Normalization statistics (T_x, T_u, T_y) must also be computed from stage-coordinate data.

---

### [D-009] One file per responsibility — scripts import from gantry_ss.py, not duplicate it
**Date**: 2026-03-17
**What**: Each script in `scripts/gantry/` has a single responsibility. `gantry_ss.py` is the sole definition of the model (physics → discrete A, B, C, D). All other scripts (simulation, validation, augmentation wiring) import `gantry_discrete_ss()` from it rather than redefining the matrices.
**Why**: Avoids parameter duplication — if a physical parameter changes, it changes in one place only. Makes the boundary between "model definition" and "model use" explicit.
**Ruled out**: Copying A, B, C, D into each script — creates silent inconsistencies if parameters are updated.
**Constrains**: Any script that needs the discrete model must import from `gantry_ss.py`. Extensions (LPV variant, different Y) are added as new functions in `gantry_ss.py`, not in the calling scripts.

---

### [D-008] Fixed SISO-only bug in modified_encoder_net; kept local copy over deepSI default
**Date**: 2026-03-16
**What**: Uncommented line 361 in `model_augmentation/fit_systems/interconnect.py` so `self.ny` is set from the `ny` argument instead of hardcoded to `tuple()`.
**Why**: The original code forced `np.prod(self.ny) = 1` regardless of actual ny, making the encoder input `nb·nu + na·1` even for MIMO systems. For the gantry (ny=3) this would silently drop output channels 2 and 3 from encoder history, giving input size 40 instead of the correct 60.
**Verified**: Unit test confirmed SISO (ny=1) input unchanged at 20; MIMO (ny=3) input now 60 (was 40).
**Ruled out**: Replacing `modified_encoder_net` with deepSI's `default_encoder_net` — kept local copy to allow gantry-specific encoder extensions later. The two are now functionally identical.
**Constrains**: Nothing locked in — local copy can still be extended independently of deepSI upstream.

---

### [D-007] Implement fixed baseline first, add trainability in a second step
**Date**: 2026-03-16
**What**: The Python FP model is first implemented as a fixed (non-trainable) baseline using `Linear_State_Block` and `Linear_Output_Block`. Trainability (`Parameterized_Linear_State_Block` / `Parameterized_Linear_Output_Block`) is added only after the fixed baseline is validated end-to-end in the augmentation interconnect.
**Why**: Stepwise approach reduces the number of failure modes at each stage. A fixed baseline is easier to verify (output is deterministic and can be compared directly against the MATLAB `G` matrices). Trainability introduces regularization and gradient flow, which should only be debugged once the structural wiring is confirmed correct.
**Ruled out**: Going straight to parameterized blocks — adds trainable parameters and param_loss complexity before the block shapes, wiring, and normalization are validated.
**Constrains**: Validation milestone required before promoting to parameterized blocks: simulated output from the Python baseline must match the MATLAB `c2d` matrices to numerical tolerance.

---

### [D-005] LFR structure confirmed for the LPV augmentation
**Date**: 2026-03-16 (updated 2026-03-20)
**What**: The augmentation framework will use an LFR structure for the LPV scheduling. This was initially deferred but was confirmed as the right approach by the supervisor in the meeting of 2026-03-20.
**Why**: The supervisor stated: "LFR gives more flexibility. Can always compute a state-space representation if we want to remap. Suggestion: start with LFR structure for scheduling/LPV." The LFR parameterization allows the learned correction to vary with Y in a principled way through the delta-p block (see D-017). Rank of the M matrix across different trajectories should be computed to confirm no rank drop occurs (expected to be fine, but must be verified).
**Ruled out**: Pure state-space augmentation without LFR structure. Deferring LFR indefinitely (supervisor explicitly suggested it as the starting point for the LPV scheduling).
**Constrains**: Step 3 implementation targets the LFR structure for LPV scheduling. A paper on discretizing LFRs must be found and reviewed before implementation (supervisor action item from 2026-03-20 meeting). The CT conversion must be written up first before the LFR structure is implemented (see D-018).

---

### [D-010] LPV baseline and LPV augmentation are separate concerns
**Date**: 2026-03-17
**What**: The LPV extension has two distinct parts that must not be conflated:
  1. **LPV baseline** — the FP model with A(Y[k]), B(Y[k]) recomputed each step from physics. This is what Step 2 builds and validates.
  2. **LPV augmentation** — a data-driven network on top of the baseline that also varies with Y. This is a Step 3+ concern.
**Why**: Jan's original augmentation framework has no LPV support. The `Parameterized_LPV_Affine_Linear_State_Block` found in the codebase is a user-added augmentation component, not a baseline block. Treating it as the LPV baseline would conflate two separate responsibilities.
**Ruled out**: Using `Parameterized_LPV_Affine_Linear_State_Block` as the LPV baseline block — it is trainable, augmentation-side, and uses an affine-in-Y² approximation that does not represent the full physics.
**Constrains**: Step 2 validates the LPV baseline purely in Python (no framework). Step 3 requires a new `LPV_Linear_State_Block` (see D-011).

---

### [D-011] Framework integration of LPV baseline requires a new block type
**Date**: 2026-03-17 (updated 2026-03-22)
**What**: Wiring the LPV baseline into the augmentation interconnect requires a new block, `CT_RK4_State_Block`, that reads Y from the current state at each forward call and integrates the CT ODE using one RK4 step.
**Why**: The existing `Linear_State_Block` stores A and B as fixed attributes set at init, so it cannot update them per step. The LPV baseline needs physics that change every timestep as Y evolves. No existing block in the framework supports this.
**Ruled out**: Reusing `Linear_State_Block` with a single frozen operating point (that is the frozen LTI). Reusing `Parameterized_LPV_Affine_Linear_State_Block` (wrong structure: affine-in-Y², trainable, augmentation-side).
**Constrains**: The block computes A_c(Y), B_c(Y) from physics at each step and applies RK4 with dt=ts (see D-018). The baseline should also be expressed in LFR form for compatibility with Drenth's augmentation procedure (see D-005, updated 2026-03-22). Y is read from state index 2 in stage coordinates (self-scheduled).

**Update 2026-03-22**: Changed from `LPV_Linear_State_Block` calling `gantry_discrete_ss(Y)` (pre-discretized DT) to `CT_RK4_State_Block` integrating the CT ODE with RK4 (per D-018). Additionally, the baseline should be expressed in LFR form per supervisor confirmation (D-005).

---

### [D-012] LPV discretization: frozen ZOH for validation, exact ZOH via matrix_exp for training
**Date**: 2026-03-17 (updated 2026-03-18)
**What**: Two discretization approaches are used, one per use case:
  1. **Validation (Step 2)**: frozen-at-sampling-instant — call `cont2discrete(A_c(Y[k]), ts)` at each step.
  2. **Training loop (Step 3)**: exact ZOH via `torch.linalg.matrix_exp(A_c(Y) * ts)`. Fully torch-differentiable (confirmed by test).

**Theoretical status — quasi-LPV caveat (important)**:
  Tóth (2010) states the ZOH setting is *"only reasonable for the discretization of LPV-SS
  representation with static dependence as dynamic dependence requires a higher-order hold
  approach"* (Section I, page 2).
  Our system is **quasi-LPV with dynamic dependence**: Y = x(3) is a system state, not an
  exogenous signal. Within each sampling interval, Y evolves continuously as the state
  integrates — it is not truly held constant by ZOH. Consequently:
  - The "errorless" property (Tóth Section IV-A: *"The complete method theoretically provides
    errorless discretization in terms of the ZOH setting"*) applies strictly to static
    dependence only.
  - For our system there is a **small but nonzero residual intra-sample error** from the
    within-interval variation of Y.

**Formal requirements from Tóth (Assumptions 1 and 2, page 5–6)**:
  - Assumption 1 (ZOH setting): *"We are given a CT-LPV system S, with CT input signal uc,
    scheduling signal pc, and output signal yc, where uc and pc are generated by an ideal ZOH
    device and yc is sampled in a perfectly synchronized manner with Td > 0 as the sampling
    period or discretization time-step."*
    Satisfied: our 20 kHz discrete control loop holds u_c and p_c (=Y) constant within each
    50 µs sample interval ✓
  - Assumption 2 (Switching effects): *"The switching behavior of the ZOH actuation has no
    effect on the CT plant, i.e. the switching of the signals is assumed to take place smoothly."*
    Tóth notes: *"this assumption is automatically satisfied in most numerical simulations of
    LPV systems, like in the implemented numerical approaches of SIMULINK in MATLAB."*
    Satisfied: Y changes continuously — no discontinuous jumps; our Python numerical simulation
    mirrors the SIMULINK approach Tóth explicitly endorses ✓
  Note: Tóth provides no quantitative bound on dp/dt. The qualitative remark on page 20
  (*"p_c changes smoothly and relatively slowly with respect to the actual dynamics of the
  plant"*) is motivating prose, not a formal condition.
  Closed-loop applicability: *"The presented ZOH setting is also applicable for closed-loop
  controllers in the structure given in Figure 2"* — our closed-loop Python simulation is
  within the scope Tóth explicitly covers.

**Self-scheduling vs external scheduling**:
  Tóth's Assumption 1 requires p[k] to be held by an ideal ZOH device -- it must be
  *measurable* (externally available) at each step k, not predicted from internal state.
  This implies external scheduling: Y[k] is read from the encoder at step k and held for
  that interval.

  Using Y[k] = x_predicted[k][2] from the model's own state (self-scheduling) introduces
  a further approximation on top of the dynamic dependence caveat already accepted above:
  - Dynamic dependence caveat: Y is a state, not an exogenous signal -- ZOH is approximate.
  - Self-scheduling: Y[k] itself is approximate (from predicted state, not measured). If the
    open-loop state drifts, the scheduling variable is wrong, compounding the error.

  External scheduling (Y[k] from measurement) is more consistent with Tóth and is used
  wherever measurements are available:
  - Training loop: Y[k] = x_measured[k][2] from real data (external, consistent with Tóth).
  - Validation against q1: Y[k] = Y_trajectory[k] from the MATLAB reference (external).
  Self-scheduling is reserved for autonomous simulation with no external measurements and
  carries the additional compounding approximation noted above.

**A_c invertibility (Tóth footnote 2)**:
  Tóth writes the complete discretization formula assuming A_c invertible *"for convenience"*
  but footnote 2 states: *"To compute the resulting matrix functions of this discretization
  approach, Ac(p) is not required to be invertible, but if it is, we can write the resulting
  DT description of the state-evolution conveniently as (9a)."*
  Our A_c is singular (rigid body modes → top-left 3×3 block is zero). The naive formula
  B_d = A_c⁻¹(A_d − I)B_c is therefore undefined. The augmented matrix exponential (D-015)
  is the correct general form — directly supported by Tóth's own footnote.

**Practical justification for small residual error**:
  The intra-sample Y variation is bounded by ΔY ≤ 0.100 mm/sample
  (= v_max × ts = 2 m/s × 50 µs; v_max from ETEL datasheet and main.m vmax=2).
  Physical timescale argument: Y traverses its full 700 mm operational range (ETEL datasheet,
  5% margin from 800 mm stroke) at maximum speed in ≥ 350 ms = 5600 samples, while the
  plant's fastest relevant dynamics act on the closed-loop bandwidth timescale
  ~1/(2π×100 Hz) ≈ 1.6 ms (fbw=100, main.m) — a ~220:1 timescale separation. This makes
  the intra-sample Y variation negligible in practice. Rigorous numerical confirmation:
  ‖A(Y+ΔY) − A(Y)‖/‖A(Y)‖ at ΔY = 0.125 mm is verified in Task 2.5.

**RESOLVED: sample rate set to 20 kHz (matching PLTI spec)**:
  AccurET-Oper&Soft-VerV.pdf confirms PLTI = 50 µs (20 kHz), matching the position control
  loop rate. main.m, export_lpv_sim.m, export_lpv_matrices.m, and physics.py all updated
  to fs = 20e3 (T_d = 50 µs). ΔY_max = 2 × 50e-6 = 0.1 mm — strengthens the slowly-varying
  argument relative to the old 0.125 mm/sample at 16 kHz.

**Why**:
  - Validation: `cont2discrete` is exact for the frozen ODE and fast enough for a one-off
    simulation. The residual quasi-LPV error is accepted as small (see above).
  - Training: `torch.linalg.matrix_exp` is a native PyTorch op — autograd traces through it,
    gradients flow back to Y[k]. The rectangular approximation (Option D, O(ts) error) is a
    valid fallback but is strictly inferior — there is no reason to accept approximation error
    when the matrix exponential is differentiable.
**Ruled out**:
  - Polynomial expansion (Option A): A_c(Y) is rational (from M(Y)⁻¹), so no exact polynomial A_d(Y) exists.
  - Linear-affine approximation (Option B): drops dominant Y² term in M[1,1].
  - Grid interpolation (Option C): not natively torch-differentiable.
  - Rectangular approximation (Option D): O(ts) error — valid fallback only. Superseded by Option E.
  - scipy `cont2discrete` in training loop: not inside autograd graph.
**Constrains**: `LPV_Linear_State_Block.forward()` must compute A_c(Y) analytically from M(Y)⁻¹ using tensor ops, then apply `torch.linalg.matrix_exp(A_c(Y) * ts)`. See `docs/lpv-discretization.md` for full rationale and option comparison table.

**Update 2026-03-20 (supervisor meeting)**: For the augmentation training loop (Step 3+), the discretization approach shifts from pre-discretized ZOH to CT model with RK4 integration. The ZOH approach remains valid for Step 2 validation (completed). See D-018, which supersedes the "training loop" part of this decision. Read D-012 as: Steps 1-2 validation used ZOH (done); Step 3+ training loop uses RK4 on the CT model (see D-018).

---

### [D-013] LPV baseline uses LFR form with CT+RK4 integration
**Date**: 2026-03-17 (updated 2026-03-22)
**What**: The LPV baseline must be *available* in LFR form {M^b, Δ^b(Y)} and integrated using RK4 inside a custom `CT_RK4_State_Block`. The `SSE_Interconnect` wiring machinery is used unchanged. Internally, the forward simulation may collapse to evaluating an equivalent CT vector field A_c(Y)x + B_c(Y)u (as Drenth Ch. 2 eq. 2.29 confirms), but the baseline must remain representable in LPV-LFR form for compatibility with Drenth's augmentation framework (Ch. 5 eq. 5.1-5.2).
**Why**: Supervisor confirmed (2026-03-22) that the baseline itself should use the LFR structure. Drenth Ch. 5 eq. 5.1 assumes the baseline is available in LPV-LFR form. Self-scheduled quasi-LPV (Y from state) is supported. The LFR representation of the baseline requires converting A_c(Y) with its rational M(Y)^{-1} entries into LFR form using standard LFT realization methods (Zhou, Doyle & Glover, 1996).
**Ruled out**: Computing A_c(Y) directly without LFR form (originally chosen, but revised per supervisor guidance). New `SSE_Interconnect` subclass (existing class is sufficient).
**Constrains**: The baseline LFR must be realized from the known physics. Normalization is handled by Drenth eq. 5.5: T_x, T_u, T_y scaling applies to all LFR submatrices. The conversion requires choosing η (repetition count in Δ) and verifying LFR well-posedness. One implementation detail remains open: whether runtime code evaluates the explicit LFR loop or the equivalent collapsed CT vector field. See `docs/lpv-lfr-interconnect.md` for the original assessment (partially superseded by this update).

**Update 2026-03-22**: Major revision. Original decision said LFR is NOT required for the baseline. Supervisor confirmed the opposite: use LFR structure for the baseline. Also changed from pre-discretized A_d(Y), B_d(Y) to CT+RK4 (per D-018). Normalization question is answered by Drenth eq. 5.5.

---

### [D-014] gantry_discrete_ss stays numpy; torch version lives in a separate file
**Date**: 2026-03-17
**What**: `gantry_ss.py` / `gantry_discrete_ss()` is not modified to support PyTorch. A separate file `scripts/gantry/gantry_lpv_torch.py` holds a torch-native implementation that mirrors `gantry_discrete_ss` in structure but uses tensor ops and `torch.linalg.matrix_exp` throughout.
**Why**: Two entirely different use cases with different dependencies and contracts:
  - `gantry_discrete_ss`: numpy in, numpy out, scipy `cont2discrete`, validation and MATLAB comparison only. Pure, simple, zero framework dependency.
  - torch version: torch tensor in, torch tensor out, differentiable, lives inside the training loop. Must stay inside the autograd graph.
  Adding a `use_torch=True` flag to `gantry_discrete_ss` would mix two concerns, add a conditional dependency on torch in a validation-only file, and violate D-009 (one file per responsibility).
**Ruled out**: Modifying `gantry_discrete_ss` to support a torch mode via flag — mixes validation and training concerns in one function.
**Constrains**: `gantry_lpv_torch.py` is a full torch reimplementation — NOT a wrapper around `gantry_discrete_ss`. Every value (physical parameters, M(Y), A_c, B_c, P transform, A_d, B_d) is defined as a `torch.tensor` from the start. No numpy intermediates, no conversion. This ensures gradients flow through the entire computation and physical parameters can optionally be made trainable later without refactoring. The only structural change from `gantry_ss.py` is replacing `cont2discrete` with `torch.linalg.matrix_exp` on the 9×9 augmented matrix (see D-015).

---

### [D-015] B_d(Y) must use augmented matrix exponential — naive formula fails
**Date**: 2026-03-17
**What**: Computing B_d(Y) via the naive formula `B_d = A_c⁻¹ · (A_d − I) · B_c` is forbidden. The correct formula uses the augmented matrix exponential:
```
M_aug = [[A_c(Y),  B_c(Y)],    # (n+m) × (n+m) = 9×9 for gantry
         [  0,        0   ]]

[A_d, B_d] = expm(M_aug · ts)[:n, :], split at column n
```

**Mathematical background**:
  The general ZOH formula for B_d (Tóth complete method, always valid) is:

    B_d = [∫₀^{T_d} exp(A_c · τ) dτ] · B_c

  This integral has no simple closed form when A_c is singular.
  When A_c is invertible, the integral simplifies algebraically to:

    ∫₀^{T_d} exp(A_c · τ) dτ  =  A_c⁻¹ · (exp(A_c · T_d) − I)  =  A_c⁻¹ · (A_d − I)

  giving the convenient form:  B_d = A_c⁻¹ · (A_d − I) · B_c   [Tóth eq. 9a]

  Tóth footnote 2: *"To compute the resulting matrix functions of this discretization
  approach, Ac(p) is not required to be invertible, but if it is, we can write the
  resulting DT description of the state-evolution conveniently as (9a)."*

  The augmented matrix exponential (Van Loan 1978) computes the integral numerically
  without any inversion:

    exp([[A_c, B_c], [0, 0]] · T_d)  =  [[A_d, B_d], [0, I]]

  B_d drops out of the top-right block directly. No A_c⁻¹ anywhere.
  This is what scipy cont2discrete(method='zoh') uses internally.

**Why A_c is singular for our system**:
  The gantry A_c has block structure:

    A_c = [[  0,    I  ],
           [-M⁻¹K, -M⁻¹C]]

  The top-left 3×3 block is identically zero. The K matrix has zero rows for X and Y
  (rigid body modes — no spring restoring force in those directions), so det(K) = 0,
  which propagates to det(A_c) = 0. A_c⁻¹ does not exist.
  Note: B_c itself is not the problem — it is well-defined as [0; M⁻¹].
  The singularity is entirely in A_c, and only affects the shortcut for B_d.

**Complexity increase vs invertible case**:
  - Invertible A_c: compute A_d = expm(A_c · ts) [6×6], then B_d algebraically — two steps.
  - Singular A_c: must form 9×9 augmented matrix and compute one expm — A_d and B_d
    obtained together. Cannot be separated. Computationally more expensive but exact.

**Why**: The gantry A_c(Y) is singular — the top-left 3×3 block is all zeros (position states
  have no velocity-independent dynamics; rigid body modes give zero eigenvalues). `A_c⁻¹`
  does not exist, so the naive formula is undefined. The augmented exponential sidesteps the
  singularity and is mathematically identical to what scipy `cont2discrete(method='zoh')`
  does internally. Both scipy and the torch version must use this formula — any discrepancy
  between them is a numerical precision issue only.
**Ruled out**: `B_d = A_c⁻¹ · (A_d − I) · B_c` — undefined for singular A_c. `B_d = ts · B_c` (rectangular fallback) — O(ts) error, only valid as Option D fallback.
**Constrains**: Both `gantry_lpv_torch.py` and any future `LPV_Linear_State_Block` must form the 9×9 augmented matrix before calling `torch.linalg.matrix_exp`. See `docs/lpv-discretization.md` for the full derivation.

---

### [D-016] Step 2 validation is matrix comparison, not trajectory simulation
**Date**: 2026-03-17
**What**: Step 2 validation compares discrete A(Y), B(Y) matrices directly against MATLAB output at 5 operating points (Y = 0.1, 0.2, 0.3, 0.4, 0.5 m). It does not require simulating a full trajectory.
**Why**: A(Y), B(Y) already match MATLAB to 1e-19 at Y=0.3 (Task 1.2). The LPV question is whether the same holds at other Y values. If the matrices match at every Y, the physics is correct — no trajectory needed to confirm that. Trajectory simulation would add complexity (need input data, initial conditions, etc.) without providing additional information about the correctness of the physics parameterization.
**Ruled out**: Running a full closed-loop trajectory simulation at each Y — unnecessary for validating the LPV matrix computation. The trajectory simulation in Step 1 already validated the dynamics at Y=0.3.
**Constrains**: Requires a new MATLAB script `Matlab-scripts/export_lpv_matrices.m` (does not modify immutable files — calls existing functions) that evaluates G at each Y and saves A, B, C, D per operating point to `Matlab-output/lpv_matrices.mat`. Python comparison script `gantry_lpv_validate.py` checks max absolute error < 1e-10 per matrix per Y. Validation sweep: Y = linspace(0.05, 0.75, 50) — confirmed from ETEL Telica datasheet (total Y stroke = 800 mm, 5% margin from hard limits). 5 points is insufficient: M(Y)⁻¹ is rational in Y and could have non-monotone error behaviour between sparse samples. Dense 50-point sweep allows plotting error vs Y to confirm uniformity across the full operational range.

**Important distinction — what matrix comparison proves vs simulation comparison**:
Matrix comparison (Task 2.4) proves implementation correctness only: Python A(Y), B(Y) match
the same physics as MATLAB G(Y). It does NOT prove that the LPV simulation is a better baseline
than the frozen LTI. That requires Export 2 (Task 2.2) on a varying-Y trajectory.

**Correct simulation comparison target: q1, not q (Simscape).**
q1 (gantrySystem.m in Simulink) is a continuous-time quasi-LPV simulation — M(Y) is
re-evaluated each integration step as Y evolves. It uses identical physics to the LPV model
(same M(Y), C, K; no Coriolis, no Coulomb). Comparing LPV vs frozen LTI both against q1
isolates the Y-varying inertia effect cleanly, without Coriolis/Coulomb interference.
q (Simscape) is the secondary target: q1 vs q quantifies the augmentation gap (Coriolis +
Coulomb). The model is quasi-LPV: captures Y-dependent inertia only — Coriolis, centripetal,
and velocity-dependent friction are dropped and must be learned by the augmentation.

---

### [D-017] Both baseline and augmentation use LFR Δ(Y) structure
**Date**: 2026-03-19 (updated 2026-03-22)
**What**: Both the FP LPV baseline and the learned augmentation use the LFR Δ(Y) structure, as required by Drenth Ch. 5 eq. 5.1-5.2. The baseline has its own Δ^b(Y) block derived from the known physics (M(Y)^{-1}). The augmentation has a separate Δ^a(Y) block with trainable parameters. The two Δ blocks are block-diagonal (no cross-coupling in Δ), but the interconnection between baseline and augmentation happens through the combined M matrix (Drenth eq. 5.2, the `ab` and `ba` submatrices).
**Why**: Supervisor confirmed (2026-03-22) that the baseline should use LFR structure. Drenth Ch. 5 eq. 5.1 explicitly assumes the baseline is in LPV-LFR form. The baseline's Δ^b(Y) is fixed (derived from physics, not trained). The augmentation's Δ^a(Y) has trainable parameters. Well-posedness of the combined LFR is guaranteed by Drenth's direct parameterization (D_zw = exp(-N), Theorem 2.5).
**Open questions**:
- Whether parameter refinement of the FP baseline (making mb, mh, etc. trainable) changes the baseline's Δ^b structure during training. To be confirmed with supervisor at April 9 meeting.
- Whether the baseline implementation should live internally in logical coordinates or be similarity-transformed to stage coordinates before coding, given D-006.
- Whether the current latent-variable realization is accepted as the project baseline or treated as an intermediate realization pending a canonical/minimal LFT realization.
**Ruled out**: Original decision that the baseline does not need LFR (revised per supervisor guidance 2026-03-22).
**Constrains**: The baseline LFR realization must be derived from M(Y)^{-1}. A latent-variable realization now exists and is acceptable as a valid candidate baseline unless a stronger canonical/minimal realization requirement is imposed. This determines the baseline's Δ^b structure and the practical η (repetition count). The combined well-posedness (baseline + augmentation) must be ensured.

**Update 2026-03-22**: Major revision. Original decision said baseline does NOT need Δ(Y). Supervisor confirmed the opposite. Both baseline and augmentation now use LFR structure, per Drenth Ch. 5.

---

### [D-018] CT model kept in continuous time; RK4 used for integration at fixed step
**Date**: 2026-03-20
**What**: The gantry FP model is implemented and maintained as a continuous-time (CT) ODE. Simulation and augmentation training both integrate the CT equations using RK4 with a fixed time step equal to the sampling period (ts = 1/fs). The model is not pre-discretized before the integration step in the training loop.
**Why**: Supervisor confirmed in meeting (2026-03-20), quoting directly from notes: "write up the ct conversion. dont do discretization first will get messy." and "use rk4 not euler discretization. better to not precompute." Key reasoning:
  - RK4 with fixed step always takes the same dt, so it responds correctly to the sampling period and is compatible with the discrete control loop.
  - RK4 is a sum of 4 terms (4 evaluations with weighting), strictly more accurate than Euler (1st order) at the same step size.
  - ODE45 uses variable step sizes (cannot enforce a consistent sampling period by default). The ode4 variant forces a fixed step, but that is equivalent to RK4 directly.
  - ZOH pre-discretization is kept only for Steps 1-2 validation (already completed) where exact MATLAB matrix comparison was the goal. It is not used in the augmentation training loop.
  - When using system identification with a CT baseline, the same RK4 approach applies: keep the model in CT, apply RK4 alongside it.
  - ZOH (zero-order hold) holds the input constant within each interval but says nothing about how the ODE is integrated inside the interval. RK4 is the integration method used inside that interval.
**Ruled out**:
  - Euler discretization: O(h) truncation error, inferior accuracy for the same step size. Supervisor confirmed: "use rk4 not euler."
  - ODE45 with variable step: incompatible with a fixed sampling period in a discrete control loop. Acceptable only as the ode4 variant (fixed step), but RK4 achieves the same result directly.
  - Pre-discretizing with ZOH for the training loop: supervisor explicitly said not to pre-compute. Write up CT first, apply RK4 at runtime.
**Constrains**:
  - The CT model equations must be written up in full before integration is applied. This means: coordinate transforms, all physical quantities with dimensions and units, the full state-space ODE in logical and stage coordinates. This write-up is a prerequisite for Step 3.
  - A paper on discretizing LFRs must be found and reviewed (supervisor action item from 2026-03-20). The LFR structure also operates on the CT equations; understanding how LFRs are discretized informs the Step 3 implementation.
  - The torch training loop integrates the CT ODE using RK4 with dt=ts. The `LPV_Linear_State_Block` planned in D-011 is revised: instead of computing and storing A_d(Y), B_d(Y), it computes A_c(Y), B_c(Y) and applies one RK4 step.
  - The LFR structure for LPV augmentation (D-005, confirmed 2026-03-20) also builds on the CT formulation.
  - Rank of the M matrix should be computed across different trajectories to confirm no rank drop occurs across the operational range.

---

### [D-020] Two methods for rational LPV dependency; Method 2 (state-space form) chosen
**Date**: 2026-03-29 (resolved 2026-03-31, Roland Tóth meeting)
**What**: Two methods exist for handling the rational LPV dependency introduced by M(Y)⁻¹. Method 2 is chosen.

**Method 1 — Online resolve (what Roel implemented):**
Keep the full LFR structure live at runtime. G and Δ(Y) remain as separate blocks. During training, the backward pass propagates through the matrix inverse, implemented either by differentiating through the explicit inverse or via fixed-point iteration. Benefits: stays in true LFR form; LTI and parameter-varying blocks remain separated (useful for control design); potentially faster inference. Disadvantage: must deal with the rational symbolic form of M(Y)⁻¹ explicitly; more complex to implement.

**Method 2 — State-space form (chosen):**
Take M(Y)⁻¹ analytically and absorb it into Ac(Y), Bc(Y). Runtime evaluates `ẋ = Ac(Y)x + Bc(Y)u` directly via RK4. Rational dependency on Y is retained (do NOT rewrite to affine). LFR is used for derivation and structural analysis only, not as a live runtime loop. The augmentation block operates on the same collapsed signals; its black box component can remain affine.

**Why Method 2**: Roland confirmed in 2026-03-31 meeting that this is acceptable. The "algebraic loop" concern was a misapplication of the definition: M(Y) being invertible means the system is well-posed and no true algebraic loop exists. Need to stick to the original parameter structure of M(Y) (augmentation can be added on top without changing the baseline structure). Simpler to implement.
**Ruled out**: Method 1 for now. Not blocked, but not needed: the simpler SS form suffices and Method 1 can be revisited if control design or faster inference become priorities.
**Note — third option not pursued (delay)**: ASMPT mentioned a third approach: introduce a unit delay into the scheduling loop to break the algebraic dependency, rather than collapsing it analytically (Method 2) or resolving it online during training (Method 1). Not chosen because Method 2 is simpler and sufficient, but recorded here for completeness.
**Constrains**: Implement `CT_RK4_State_Block` using Ac(Y), Bc(Y) with rational-in-Y entries (from M(Y)⁻¹). Do not rewrite to affine. Verify M(Y) invertibility numerically: compute singular values of M(Y) across the full Y operational range and confirm they remain bounded away from zero. Check that maximum signal values in M(Y) are below 1 (or 1/0.75) to bound remaining concern.

---

### [D-021] Verify M(Y) invertibility numerically across the Y operational range
**Date**: 2026-03-31
**What**: Before relying on M(Y)⁻¹ in the runtime implementation, numerically verify that M(Y) remains invertible across the full operational Y range. Compute singular values of M(Y) for Y swept across [0, 0.7] m. Confirm all singular values stay bounded away from zero. Also check that maximum signal values in M(Y) are below 1 (or 1/0.75) to bound any remaining well-posedness concern.
**Why**: Roland noted this as a concrete verification step. Y range is also relevant for centering the scheduling variable: centering Y (e.g., Y_c = Y - Y_mean) improves numerical conditioning and avoids potential singularities near the boundary of the operational range.
**Ruled out**: Assuming invertibility without verification.
**Constrains**: This is a prerequisite check before implementing `CT_RK4_State_Block`. Script can be a short standalone MATLAB or Python check. Results should confirm M(Y) is positive definite (physical mass matrix) throughout the range.

---

### [D-022] Non-baseline physics go in augmentation, not in baseline
**Date**: 2026-03-31
**What**: Physical effects not present in the García-Herreros first-principles equations must not be added to the baseline model. They belong in the augmentation component and can be parametrized there.
**Why**: Confirmed by Roland in the 2026-03-31 meeting, specifically in response to the ASMPT-raised question about hysteresis. The concrete example: using sign(dY/dt) as an additional scheduling variable to capture hysteresis direction is a good idea, but it goes in the augmentation, not the baseline. Hysteresis is the motivating example that established this rule. The baseline must remain the exact FP model as derived. Adding extra physics to the baseline would conflate the known physics with the learned correction, making it harder to isolate what the augmentation is doing.
**Ruled out**: Extending the baseline state-space equations with additional physical terms (hysteresis, Coriolis, resonance, etc.).
**Constrains**: The baseline is frozen at the García-Herreros equations. Additional dynamics, forces, and scheduling variables (including sign(dY/dt) for hysteresis) are added in the augmentation block only.

---

### [D-023] Training roadmap: validate parameter estimation on synthetic MATLAB data before adding augmentation
**Date**: 2026-03-31
**What**: The training proceeds in two phases before full augmentation:
  1. Generate synthetic data from MATLAB for various Y values and parameter volumes.
  2. Train the baseline model with free (trainable) physical parameters only — no augmentation black box (Jan's parameter update method). Initialize parameters close to the true values. Show that the parameter estimation recovers the correct parameters from MATLAB-generated data.
  Only after this is demonstrated does augmentation (extra states, Coriolis, etc.) get added.
**Why**: Roland specified this phasing in the 2026-03-31 meeting. Validating the parameter update step in isolation (no black box) proves the baseline training pipeline works before adding augmentation complexity. This mirrors Jan's original method.
**Ruled out**: Jumping straight to augmentation training without first showing the baseline parameter estimation works on synthetic data.
**Constrains**: Synthetic data must cover a representative range of Y and other parameter volumes. The parameter initialization must be close enough to the true values for convergence. The "show it works" milestone (baseline parameters converge to MATLAB ground truth) is required before Step 4 (augmentation) begins.

---

### [D-024] Augmentation ordering: resonance first, Coriolis second
**Date**: 2026-03-31 (ASMPT meeting)
**What**: The augmentation is built up in two steps: first catch resonance dynamics, then add Coriolis as a second step. Coriolis is the more complex effect and should not be targeted before resonance is demonstrated to work.
**Why**: ASMPT guidance from the 2026-03-31 meeting. Resonance is the simpler and more immediate correction; Coriolis requires additional states and is a larger modelling step.
**Ruled out**: Adding Coriolis in the first augmentation step.
**Constrains**: The augmentation milestones in D-023 (training roadmap) follow this ordering.

---

### [D-025] Hysteresis: significant effect, sign(dY/dt) scheduling variable in augmentation
**Date**: 2026-03-31 (ASMPT meeting)
**What**: Hysteresis is a significant unmodelled effect in the gantry. The current scheduling structure (Y-only) cannot capture hysteresis direction because that requires the sign of velocity. Proposed approach: add sign(dY/dt) as an additional scheduling variable, or add a simple explicit hysteresis sub-model. Both approaches belong in the augmentation, not the baseline (see D-022). If hysteresis is not addressed at all, the network will absorb it through black-box fitting, which may reduce interpretability.
**Why**: Raised by ASMPT in the 2026-03-31 meeting. Confirmed by Roland as a good idea for the augmentation side.
**Open**: Whether to apply cost function weighting for hysteresis-dominated regions. Whether a dedicated simple hysteresis sub-model is better than the scheduling variable approach.
**Ruled out**: Adding hysteresis handling to the baseline model.
**Constrains**: When designing the augmentation scheduling structure, include sign(dY/dt) as a candidate scheduling variable. Revisit after resonance augmentation is validated (D-024 ordering).

---

### [D-026] Remove G from lfr_forward — replace G-matrix steps with direct physics expressions
**Date**: 2026-04-02

#### What was decided

The `G` argument is removed from `lfr_forward`. Steps 6 and 7 of the forward pass are replaced with direct physics expressions:

**Before (removed):**
```python
def lfr_forward(x, u, Y, G, M0, M1, M2, K, C):
    ...
    # Step 6: state derivative via G matrix  →  (batch, 6)
    xdot = x @ G.Ax.T + w @ G.Bw.T + u @ G.Bu.T

    # Step 7: output  →  (batch, 3)
    y = x @ G.Cy.T
```

**After (implemented):**
```python
def lfr_forward(x, u, Y, M0, M1, M2, K, C):
    ...
    # Step 6: state derivative — direct from physics (no G needed)
    xdot = torch.cat([x[:, 3:], v], dim=-1)   # (batch, 6)

    # Step 7: output — positions in logical coordinates
    y = x[:, :3]   # (batch, 3)
```

The `G` argument is also removed from `rk4_step` in `lfr_simulate.py`, from the `simulate` function signature, and from `LFRBaselineBlock` in `lfr_block.py` (the `self._G` attribute is removed; `rk4_step` no longer needs it).

---

#### Why this is a valid change — mathematical justification

**The physical state equations.** The gantry equation of motion in logical coordinates is:

```
M(Y) q̈ = -K q - C q̇ + u
```

The state is `x = [q; q̇] ∈ R⁶`, with `x[0:3] = q` (positions) and `x[3:6] = q̇` (velocities). The continuous-time state derivative is therefore:

```
ẋ = [q̇; q̈] = [x[3:6];  M(Y)⁻¹ fnet]       (equation 1)
```

where `fnet = -K x[0:3] - C x[3:6] + u` is the net generalized force. After **step 3** of `lfr_forward`, the quantity `v = M(Y)⁻¹ fnet` is already computed via `torch.linalg.solve(M_Y, fnet)`. Equation (1) then gives directly:

```
xdot = cat([x[:, 3:], v], dim=-1)            (equation 2)
```

This is always exactly correct, for any value of Y, because it is derived directly from the physical equations of motion.

**What G.Ax/Bw/Bu encode.** The LFR G-matrix representation expresses the same state equation as:

```
xdot = G.Ax @ x + G.Bw @ w + G.Bu @ u
```

where `w = [v₁; v₂] = [Y·v; Y²·v]` are the LFR latent signals (already computed in steps 4–5). The entries G.Ax, G.Bw, G.Bu are **constant matrices**, constructed by `build_G_matrix()` using `M₀⁻¹` (the mass matrix at a nominal point). The Y-dependence is captured through the latent signals `w`, not through G directly.

This G-matrix expression is algebraically identical to equation (2) — the LFR G matrices were derived precisely to encode the physical state equations in the LFR framework. The identity holds because the LFR structure is exact: the LFR is not a linearization or approximation; it is an exact rewriting of the rational-in-Y equations using the Δ(Y) = Y·I₆ block (verified in `lfr_forward.py` Check 2 against the collapsed form A_c(Y)@x + B_c(Y)@u).

**Why the G-matrix expression is inferior to the direct expression.** Even though the two forms are algebraically equivalent, the G-matrix form has a hidden dependency: it is only correct when G.Ax/Bw/Bu are consistent with the current values of M0/M1/M2/K/C. G is precomputed at import time in `lfr_matrices.py` by calling `build_G_matrix(M0, M1, M2, K, C)`. If M0/M1/M2/K/C are updated during parameter estimation, but G is not rebuilt, the G-matrix expression silently produces incorrect gradients and incorrect dynamics. The direct expression (equation 2) has no such dependency: it is always correct for whatever M0/M1/M2/K/C are passed to `lfr_forward` at that call.

**Why G.Cy = [I₃ | 0₃] is also removed.** The output `y = x @ G.Cy.T` selects the first 3 state components (logical positions). G.Cy is always `[I₃ | 0_{3×3}]` by the gantry output definition (output = position in logical coordinates). This is directly `x[:, :3]`. Unlike G.Ax/Bw/Bu, G.Cy would not become stale during parameter estimation (it does not depend on M0). However, replacing it with `x[:, :3]` is simpler, removes the G dependency entirely, and is more readable.

**Autograd implications.** The gradient path for physical parameters (M0, M1, M2) flows through `torch.linalg.solve(M_Y, fnet)` → `v` → `xdot`. This path exists in both the old and new implementation. The G-matrix form additionally has gradient paths through G.Ax/Bw/Bu entries when G is built dynamically from M0 inside the forward context. These extra paths disappear with the G removal. However, the physically correct gradient path (through the solve) is the one that was always present and is the one required for parameter estimation. The extra G-entry gradient paths in the old implementation were an artifact of redundant parameterization, not a feature.

---

#### What was ruled out

**Option A: Keep G in the signature but always rebuild it inside forward.**
`G = build_G_matrix(M0, M1, M2, K, C)` at each forward call, then use `G.Ax/Bw/Bu`. This adds unnecessary matrix computation at every forward step (linalg.solve inside build_G_matrix) and computes the same result as equation (2) through a much more expensive path. Rejected: unnecessary overhead, no benefit over the direct expression.

**Option B: Keep G and require the caller to always pass a freshly built G.**
Documented as a constraint ("caller must keep G consistent"). This is error-prone: the interface has two representations of the same physics, and nothing prevents them from diverging silently. Rejected: fragile by design, no benefit over the direct expression.

**Option C: Keep G only for documentation/clarity.**
G was never purely documentary — it participates directly in computation and autograd. Keeping a live computational dependency on G for readability reasons is not justified. Rejected.

---

#### What this constrains

- **lfr_forward signature** is now `(x, u, Y, M0, M1, M2, K, C)`. Any call site must be updated.
- **rk4_step and simulate** no longer accept or pass G. All call sites updated accordingly.
- **LFRBaselineBlock** does not store `self._G`. `build_G_matrix` is not called inside `forward()`.
- **G and build_G_matrix** remain in `lfr_matrices.py` — they are still useful for numerical analysis, LFR structure inspection, and offline verification. They are not deleted.
- **SVD-reduced forward pass** (`svd/lfr_svd_forward.py`) must NOT apply this shortcut. In the reduced realization the state and latent vectors are rotated by the SVD transformation matrices; the physical structure (positions first, velocities last) no longer holds, so `cat([x[:,3:], v])` is incorrect for the reduced system. The SVD-reduced forward must retain its G_reduced.Ax/Bw/Bu parameterization.
- **Check F in test_jan_compat.py** (trainable physical parameter gradient test) is simplified: only the solve-path gradient path exists. The distinction between "static G" and "dynamic G" is removed. The updated check verifies that M0.grad is non-None after backward — which is guaranteed by the linalg.solve gradient — and reports the gradient norm.

---

### [D-027] Fix y-output coordinate mismatch in the Interconnect connection matrix
**Date**: 2026-04-02
**What**: The `S_y` selection matrix in `build_baseline_interconnect` and `build_augmented_interconnect` (in `test_jan_compat.py`) was:
```python
S_y = selection_matrix(np.arange(3), 18)    # (3, 18) — selects logical positions
```
This routes `x_next[0:3]` (logical positions [X, Θ, Y]) directly as the Interconnect output `y`. The reference and training data use stage coordinates [X1, X2, Y]. The fix embeds the logical→stage transform into the connection matrix:
```python
S_y = P.numpy() @ selection_matrix(np.arange(3), 18)    # (3, 18) — logical → stage
```
In row-vector convention used throughout the Python code, `y_stage = y_logical @ P` (see `simulate()`: `Y_list.append(y_k @ P)`). For the Interconnect where the connection matrix acts as `y = S_y @ w_block` (column-vector convention), the correct transform is `S_y = P.numpy().T @ selection_matrix(np.arange(3), 18)`.

Wait — the Interconnect uses column-vector convention (w_block is (batch, nw, 1)), so `y = S_y @ w_block` computes (3, 18) @ (18, 1) = (3, 1). To obtain `y_stage = P.T @ y_logical` (column-vector form), `S_y = P.numpy().T @ selection_matrix(np.arange(3), 18)`.

**Why**: The MATLAB reference data (`q3`, simulation outputs) are in stage coordinates [X1, X2, Y]. The `lfr_forward` output `y = x[:, :3]` is in logical coordinates [X, Θ, Y]. The two coordinate systems differ in the X1/X2 vs X/Θ representation — they are related by `y_stage = P.T @ y_logical` (column-vector form). Without the P-transform in S_y, the Interconnect would output logical positions as training targets, causing incorrect loss computation when compared against stage-coordinate reference data.
**Ruled out**: Embedding the P-transform in `lfr_block.py` (adding y-routing logic to the block output, changing nw). The connection matrix is the correct place for coordinate transforms in Jan's framework — the block output format is fixed by the nw=18 contract.
**Constrains**: `build_baseline_interconnect` and `build_augmented_interconnect` in `test_jan_compat.py` apply this fix. Any future Interconnect wiring for the gantry baseline must use `P.numpy().T @ selection_matrix(np.arange(3), 18)` for the y connection matrix, not a plain selection matrix.

---

### [D-028] Add BPTT mode toggle to simulate()
**Date**: 2026-04-03
**What**: `simulate()` in `lfr_simulate.py` gains a `bptt_mode` parameter with three options: `"full"` (default, unchanged behaviour — retains entire graph), `"truncated"` (detach state every `segment_len` steps), and `"checkpoint"` (use `torch.utils.checkpoint` for exact gradients at O(sqrt(N)) memory). `simulate_frozen()` moved from `validate_lfr.py` to `lfr_simulate.py`.
**Why**: The full computation graph across N RK4 steps is O(N) in memory. For realistic training horizons (N > 1000), this becomes impractical. Jan's framework handles this implicitly via `nf`-bounded windows (typical nf=200), but our standalone `simulate()` had no such bound. The three modes give callers explicit control: `"truncated"` matches Jan's nf pattern (cheap, biased gradients); `"checkpoint"` gives exact gradients at ~1.3x compute; `"full"` remains the default for backward compatibility and short horizons.
**Ruled out**: Adjoint method (torchdiffeq) — exact O(1) memory but numerically unstable for stiff systems and adds an external dependency. Hardcoding a single BPTT strategy — different training scenarios benefit from different trade-offs.
**Constrains**: Training scripts should choose `bptt_mode` explicitly based on horizon length and gradient quality requirements. `segment_len` for truncated mode should cover the system's settling time (~200-1000 steps at 20 kHz).

### [D-029] LPV-LFR baseline code cleanup: performance and CUDA readiness
**Date**: 2026-04-05
**What**: Cleaned up the lpv_lfr_baseline package based on a line-by-line code review. Changes: (1) Pre-transform u_seq from stage to logical coords once before the simulate() loop instead of N times inside it. (2) Pre-allocate output tensors in simulate() and simulate_frozen() instead of list+stack. (3) Removed `_rk4_step_for_checkpoint` wrapper (identical to `rk4_step`; checkpoint calls `rk4_step` directly now). (4) Added `Y_override` parameter to `rk4_step` so `simulate_frozen` reuses the same RK4 logic instead of duplicating it. (5) Made lfr_block.py dtype cast conditional (skip when already float64). (6) Fixed CUDA device bug in simulate_frozen (`torch.full` was missing `device=x0.device`). (7) Pre-allocated tensors use `x0.new_empty()` to inherit device and dtype. (8) Trimmed module docstrings in lfr_forward.py and lfr_simulate.py. (9) Fixed test_jan_compat.py S_y construction to avoid unnecessary numpy round-trip.
**Why**: Preparing for GPU training. The original code had N redundant P.T matmuls per trajectory, N+1 tensor object allocations in Python lists, and a device bug that would crash on CUDA.
**Ruled out**: Deleting lfr_matrices.py (still used by svd/). Switching from torch.linalg.solve to Cholesky (negligible difference for 3x3 matrices).
**Constrains**: `rk4_step` now has an optional `Y_override` keyword argument. Callers using positional args are unaffected. `simulate_frozen` is now a thin wrapper around `simulate`-style logic with `Y_override`.

### [D-030] Trainable physical parameter set for ParameterizedLFRBlock
**Date**: 2026-04-06
**What**: 10 trainable scalars, 2 fixed scalars, in `ParameterizedLFRBlock`. Trainable: `kb_sum` (=kb1+kb2), `cg1`, `cg2`, `cy`, `cb_sum` (=cb1+cb2), `mh`, `m1`, `m2`, `mb`, `J_sum` (=Jb+Jh). Fixed buffers: `Lb`, `d`.
**Why**: Identifiability analysis on the matrix structure of M(Y), C, K:
- `kb1`, `kb2` appear only as their sum in K[1,1] → not individually identifiable; train sum.
- `cg1`, `cg2` appear as both sum and difference in C → individually identifiable.
- `cy` appears isolated in C[2,2] → directly identifiable.
- `cb1`, `cb2` appear only as sum in C[1,1] → train sum.
- `mh` is the sole LPV parameter (enters M0, M1, M2) → strongest signal, must train.
- `m1`, `m2` appear individually via M0[0,1]=(m1-m2)*Lb/2 → identifiable.
- `mb` appears only in M0[0,0] sum with m1+m2+mh → weakest signal; train with tight Lambda.
- `Jb`, `Jh` appear only as sum in M0[1,1] → train sum.
- `Lb` appears in M0, C, and the P coordinate transform; changing P corrupts stage↔logical mapping during training → fixed.
- `d` appears only in products mh*d and mh*d² alongside trainable mh → not separately identifiable; fixed.
All 10 trainable scalars are simultaneously trained from the start (same pattern as `Parameterized_MSD_State_Block`). Lambda regularization weights handle the varying identifiability — tighter for `mb` (2% detuning), standard for others (5–10% detuning).
**Ruled out**: Training `Lb` (corrupts P transform), training `d` (unidentifiable alongside mh), training `Jb`/`Jh` individually (only sum is identifiable), phased training (Jan trains all params at once; regularization handles weak identifiability).
**Constrains**: `_build_matrices()` in `lfr_param_block.py` must reconstruct M0, M1, M2, K, C from these 10 scalars plus fixed `Lb`, `d`. Detuning amounts: kb_sum −5%, cg1/cg2/cy/cb_sum −10%, mh/m1/m2/J_sum −5%, mb −2%.

---

### [D-031] Implement ParameterizedLFRBlock in a separate file lfr_param_block.py
**Date**: 2026-04-06
**What**: The trainable-parameter LFR block lives in `lpv_lfr_baseline/lfr_param_block.py`, not in `lfr_block.py`.
**Why**: `lfr_block.py` has a single well-tested responsibility (stateless frozen-parameter wrapper). The parameterized variant adds substantial new logic: scalar parameter management, `_build_matrices()` differentiable reconstruction, and `param_loss()` regularization. Mixing these two concerns would make both files harder to read and test independently. The existing module follows a one-concern-per-file pattern.
**Ruled out**: Extending `lfr_block.py` with a subclass (same file becomes bloated); creating a generic `parameterized_block.py` (too abstract for one use case).
**Constrains**: `lfr_block.py` stays untouched as the frozen baseline reference. `lfr_param_block.py` imports `rk4_step` from `lfr_simulate.py` and scalar constants from `physics.py` as initial values only.

---

### [D-032] Subclass SSE_Interconnect to handle ParameterizedLFRBlock.param_loss()
**Date**: 2026-04-06
**What**: A thin subclass of `SSE_Interconnect` (living in `lpv_lfr_baseline/`) overrides `loss()` to add a generic `hasattr(m, 'param_loss')` sweep over connected blocks. Jan's `model_augmentation/` code is not modified.
**Why**: `SSE_Interconnect.loss()` calls `param_loss()` only on hard-coded `isinstance` checks for its own block types. `model_augmentation/` is read-only (CLAUDE.md). A subclass override is the minimal, non-invasive extension.
**Ruled out**: Editing Jan's `interconnect.py` (violates read-only constraint); monkey-patching at runtime (fragile).
**Constrains**: The subclass must call `super().loss()` minus the block-type sweep, then add its own generic sweep — or replicate the loss structure with the generic check. It lives in `lpv_lfr_baseline/` and is the entry point for all training scripts in this project.

---

### [D-033] Data strategy: Option A (MATLAB) for first experiment, Option B (Python simulate) future
**Date**: 2026-04-06
**What**: The first parameter-recovery experiment uses the existing `Matlab-output/lpv_sim_varying_y.mat` as training data (Option A). Option B — generating fresh synthetic data via Python `simulate()` with a multisine input, controlled noise (SNR), and explicit train/val/test splits — is deferred to a future experiment.
**Why**: The MATLAB trajectory was generated with the true physical parameters and provides the ground-truth output we need to train against. It exercises varying Y (0.3→0.1 m), which is exactly the range where M(Y) variation is observable. Option B is more rigorous and mirrors Jan's experimental design exactly, but requires additional scripting (input design, noise model, data splits) that is not needed to prove the concept.
**Ruled out**: Using frozen-Y data (LPV parameter mh not identifiable without Y variation); skipping Option B entirely (it is the right long-term approach for a rigorous benchmark).
**Constrains**: The training script must load and convert `lpv_sim_varying_y.mat` to deepSI format. When Option B is implemented, the training script should be parameterizable to switch data sources without changing the model structure.

---

### [D-019] Use Drenth thesis for CT LPV-LFR citations; treat IFAC paper as DT companion
**Date**: 2026-03-24
**What**: For any continuous-time LPV-LFR definition, notation, or generic interconnection equations used in the gantry write-up, the primary source is Drenth's thesis (`literature/books/drenth2025_lpv-lfr-thesis.pdf`). The IFAC paper (`literature/lpv-lfr/drenth2025_lpv-lfr-rational.pdf`) is treated as the discrete-time companion paper and cited as such.
**Why**: The two local Drenth sources are not interchangeable. The thesis explicitly gives the LPV-LFR pair `(G, Delta(p))` in continuous time with `x_dot(t)`, `z(t)`, `w(t)`, `y(t)` and the equivalent rational LPV-SS form. The IFAC paper defines the LPV-LFR pair `{M, Delta(p)}` in discrete time. Citing the IFAC paper as if it were the primary CT definition overstates the DT-to-CT adaptation and obscures the notation difference between the two sources.
**Ruled out**: Treating the thesis and IFAC paper as equivalent sources for Section 2-style CT LPV-LFR definitions. Citing IFAC eq. 6-9 as if it were the primary CT source.
**Constrains**: `docs/references.md`, `docs/lfr-structure.md`, and future LaTeX source notes should cite the thesis for CT LPV-LFR definitions. The IFAC paper remains useful for DT LPV-LFR context, rational-dependency motivation, and well-posedness discussion, but should be labeled as the DT companion when referenced.

---

### [D-034] RMSE_baseline for Lambda regularization computed from detuned baseline on MATLAB data
**Date**: 2026-04-06 (updated 2026-04-20)
**What**: Before training begins, compute the per-trajectory RMSE of `ParameterizedLFRBlock` with `params = params_init` (detuned values) on the active MATLAB trajectories. Two quantities are derived from this:

1. `rmse_baseline` — group-balanced RMSE **in metres** (physical units). Used only for reporting and to instantiate the block when the loss is in physical units (not the current training setup).
2. `rmse_baseline_normalized` — the same RMSE expressed **in sigma-normalized units** (dimensionless), computed via `_aggregate_normalized_rmse_baseline()`. This is what is actually passed to `ParameterizedLFRBlock.__init__()` as `RMSE_baseline`.

The distinction matters because the training loss is normalized by sigma (see D-042):
```
mse_loss = mean(((Y_pred - q1) / sigma)²)    # dimensionless, O(1)
```
Lambda must be calibrated in the same unit system as `mse_loss`. Passing the metre-space value would make Lambda ~450× too small, effectively disabling regularization.

Inside the block, Lambda is computed as:
```python
Lambda[i] = RMSE_baseline_normalized / params_init[i]
```
This ensures the regularization cost is comparable to the simulation MSE when parameters have moved enough to reduce the (normalized) prediction error by one `RMSE_baseline_normalized` unit.

**Why**: RMSE_baseline_normalized scales the regularization relative to the simulation loss in the same unit system. Computing it from the actual detuned baseline on actual data gives principled, automatic calibration. Jan's fixed constant (0.2) is only valid because his data is already normalized to O(1) — our raw data is in metres and sigma-normalization must be applied first.
**Ruled out**: Passing `rmse_baseline` (metres) to the block — Lambda would be ~450× too small and regularization would be ineffective. Manual constant without sigma normalization — arbitrary and unit-dependent.
**Constrains**: `train_param_recovery.py` must compute both `rmse_baseline` (for logging) and `rmse_baseline_normalized` (for the block). The block always receives the sigma-normalized value. Both values should be logged in the saved `.pt` file for reproducibility. See D-042 for the sigma normalization itself.

---

### [D-036] OPEN — Augmentation training: state initialisation and mini-batch strategy
**Date**: 2026-04-08
**Status**: Deferred — decide when implementing augmentation training.
**What**: Two coupled design choices must be made when extending from parameter recovery to augmentation training:

**Choice A — State initialisation for segment start states:**

Option 1 (data-derived, current): positions from measured q1, velocities from central finite differences. Cached as `state_traj_n{N}.pt`. Works for parameter recovery because all states are observable (q, q̇ from positions). **Will not generalise to augmentation**: the augmentation block introduces latent states (e.g. hidden flexible modes) that cannot be read from measured positions or computed by finite differences.

Option 2 (encoder, Jan's approach — `model_augmentation/fit_systems/interconnect.py` line 417): `x = self.encoder(uhist, yhist)`. A learned neural network maps a window of past inputs and outputs to the full augmented state. The encoder is trained jointly with the physics parameters. This is the only correct approach when latent states exist.

**Recommendation**: Keep data-derived states for parameter recovery (current code). Switch to an encoder when augmentation is added. The encoder architecture Jan used is `modified_encoder_net` in `interconnect.py` — a `simple_res_net` mapping `[uhist, yhist]` → `x0`.

**Choice B — Segmentation strategy (overlapping vs non-overlapping):**

Current (parameter recovery): non-overlapping segments, stride = segment_len. Batch = n_seg = N // segment_len (e.g. 70). One gradient update per epoch = full-batch GD.

Jan's approach (augmentation): overlapping sliding windows, stride controlled by deepSI data loader (typically stride=1 or small). Many more gradient updates per epoch — effectively mini-batch SGD. More diverse gradient signal; helps generalisation and can escape local minima.

Trade-off: overlapping windows require the encoder to re-estimate state at every window start (batch × encoder forward pass per epoch). Non-overlapping is cheaper but less diverse. For noisy real data with a learned augmentation, mini-batch SGD over overlapping windows is the standard choice (confirmed by Jan's code).

**Recommendation**: For augmentation training, adopt Jan's overlapping strategy with encoder-based state init. The precomputed `state_traj` cache is still useful for the physical (observable) state components as a warm-start or validation reference.

**Ruled out at this stage**: None — decision deferred until augmentation implementation begins.
**Constrains**: Augmentation training script design. Encoder architecture and hyperparameters (nb, na window lengths) must be chosen at that time.

---

### [D-035] Physical parameter positivity enforced via log/exp reparameterization
**Date**: 2026-04-06
**What**: Physical scalars in `ParameterizedLFRBlock` are stored as `self.log_params = nn.Parameter(torch.log(params_init))`. Physical values are recovered as `params = torch.exp(self.log_params).clamp(min=1e-6)` inside `forward()` and `param_loss()`. The clamp is a numerical crash guard only, not an optimization mechanism.
**Why**: If any physical parameter goes zero or negative during training, `M(Y) = M0 + M1*Y + M2*Y²` becomes singular and `torch.linalg.solve` crashes or produces garbage. L2 regularization alone provides no hard guarantee. Log/exp reparameterization maps the unconstrained real line to `(0, ∞)` — the optimizer trains `log_params` freely in ℝ and positivity is guaranteed by construction. Literature survey (GPyTorch, Stan, neural ODE grey-box models, PINN parameter ID papers) confirms log/exp is the dominant choice for positive scalar physical parameters. Initialisation is trivial: `log(params_init)` exactly inverts the exp transform, so training starts at the correct physical values.
**Ruled out**:
- *Softplus*: `params = log(1 + exp(raw))`. Functionally equivalent to log/exp at our parameter magnitudes (all ≥ 1.05 kg) — softplus saturates to identity for large inputs so the two are numerically indistinguishable. Softplus is GPyTorch's default because it prevents overflow during large hyperparameter searches; this concern does not apply here since L2 regularization keeps params near init. Rejected in favour of log/exp for simplicity (no `softplus_inverse` needed at init) and because it is the more standard choice in the system identification literature.
- *Projected gradient / clamping as training strategy*: `params.clamp_(min=1e-6)` after each optimizer step. Creates a discontinuous gradient at the boundary — the optimizer sees a flat landscape and cannot recover. Parameters cluster at the clip value. Widely considered an antipattern (cf. WGAN weight clipping critique). Retained only as a numerical safety net after exp, not as a constraint mechanism.
- *Log-barrier term*: Add `-λ · Σ log(params)` to the loss. Requires scheduling λ toward 0 (interior point method) to be principled; in stochastic gradient training with Adam this scheduling is difficult to get right. Adds a hyperparameter with no clear benefit when L2 regularization already anchors parameters near positive initial values.
- *Unconstrained training relying on regularization alone*: L2 regularization provides a soft pull toward positive init values but no hard guarantee. For a small detuning (5-10%) and well-calibrated Lambda this would likely work in practice, but provides no protection against edge cases (aggressive learning rates, long training, poor RMSE_baseline calibration).

---

### [D-036] OPEN: LFR structure vs. state-space-only for LPV baseline and augmentation
**Date**: 2026-04-09 (raised in supervisor meeting, not yet decided)
**What**: Decide whether to express the LPV baseline as a true LFR (with M(Y) invertibility as a
rational/symbolic expression) or remain in state-space form (current: `torch.linalg.solve` at
every step).
**Why this matters**:
- Current `linalg.solve` approach is numerically correct but gives zero LFR structural benefit.
- LFR structure is almost essential for control design (H-inf, mu-synthesis) — a primary interest
  of ASMPT even when a black-box augmentation is added on top.
- Expressing M(Y)^{-1} symbolically as a rational function (MATLAB can do this) means no per-step
  matrix inversion; the forward pass becomes matrix-vector products only — computationally cheaper
  and structurally a proper LFR.
- Jan's interconnect framework supports state-space directly (no LFR required), but this trades
  away the control-design benefit.
**Open sub-questions**:
1. Does the parallel augmentation (D-003) still provide the orthogonality regularization benefit
   if the baseline is in state-space form rather than LFR? (I.e., what exactly is traded away?)
2. SVD on the LFR channels: reduces latent signals (good for control), but how does it affect
   interpretability of the learned augmentation states?
3. Identifiability / uniqueness of parameter updating: which parameter combinations only appear
   as sums in M(Y)? Can trajectory excitation separate them, or is norm regularization needed?
**Decision path**:
- If project scope includes control design deliverable → invest in symbolic M(Y)^{-1} (MATLAB)
  to recover LFR structure before augmentation.
- If scope is simulation/prediction only → state-space form is acceptable; note the limitation
  explicitly in the thesis.
**Ruled out**: Nothing ruled out yet — decision deferred pending scope clarification with supervisors.
**Constrains**: LPV model implementation (`lpv_lfr_baseline/`), augmentation interconnect structure,
and any control design work downstream.

---

### [D-037] IMPLEMENTED: Split regularization on degenerate parameter pairs
**Date**: 2026-04-09 (raised in supervisor meeting); **Implemented**: 2026-04-22
**What**: kb1/kb2, cb1/cb2, and Jb/Jh each appear only as sums in the physics equations (K[1,1]=kb1+kb2, C[1,1]=cb1+cb2, M[1,1] contains Jb+Jh). This creates a flat ridge in loss: any split summing to the correct value gives identical RMSE. A scale-invariant "split loss" breaks the degeneracy.
**Why**: The standard RMSE loss has zero gradient in the split direction for these pairs. Without a tiebreaker the optimizer stagnates on a line rather than converging to the true split.
**Implementation** (`SPLIT_REG_WEIGHT = 1e-2`):
```python
# lfr_param_block.py -- ParameterizedLFRBlock.split_loss()
def split_loss(self) -> Tensor:
    p = self._recover_params()
    kb1, kb2 = p[0], p[1]
    cb1, cb2 = p[5], p[6]
    return (
        ((kb1 - kb2) / (kb1 + kb2)).pow(2)   # symmetric pairs -- prefers equal split
        + ((cb1 - cb2) / (cb1 + cb2)).pow(2)
        + (self.log_params[11] - self.log_params[12]).pow(2)  # Jb/Jh -- log-space (true values differ)
    )
```
- kb/cb pairs: normalised squared difference `((a-b)/(a+b))^2` — dimensionless, scale-invariant, zero at a=b. Correct because true values are equal by design (kb1=kb2=1987.5, cb1=cb2=9.0).
- Jb/Jh: log-space squared difference — prefers proportional fractional detuning rather than equal split. Correct because true values differ (Jb=1.0, Jh=0.05); forcing equal split would be physically wrong.
- Weight `1e-2` is small enough that it does not meaningfully distort the RMSE landscape when the sum is already near its correct value; it only resolves the flat direction.
**Compute cost**: Three tensor ops per backward pass — negligible.
**Constrains**: `train_param_recovery.py` (`SPLIT_REG_WEIGHT`, `train()` signature, loss assembly, hist_entry, save dict); `lfr_param_block.py` (`split_loss()` method).
**Old notes (pre-implementation)**:
- Roland's suggestion: centre and normalize log-parameters around ~1 before gradient step. Not implemented — log/exp reparameterization (D-035) already handles scale.
- Alternative to log: `p^2` reparameterization. Not needed; log/exp stable in practice.

---

### [D-038] Simulation study extra state: Y-position-dependent Dahl friction states [z₁, z₂]
**Date**: 2026-04-10
**What**: The 8-state data-generating model for the augmentation simulation study adds two Dahl friction states [z₁, z₂] — bristle deflections on the X₁ and X₂ guides — to the 6-state LPV baseline. The baseline remains unmodified (6 states, constant C and K). The augmentation must discover the extra states and their coupling.

Data-generating model dynamics (extra states):
```
ż₁ = Ẋ₁ − (|Ẋ₁|/g) · z₁     where Ẋ₁ = Ẋ + (Lb/2)·Θ̇
ż₂ = Ẋ₂ − (|Ẋ₂|/g) · z₂     where Ẋ₂ = Ẋ − (Lb/2)·Θ̇

Y-dependent Coulomb amplitudes:
  Fc₁(Y) = Fc · (Lb/2 − Y) / Lb
  Fc₂(Y) = Fc · (Lb/2 + Y) / Lb

Modified force equations in data generator:
  F_X_friction = Fc₁(Y)·z₁ + cg1·Ẋ₁ + Fc₂(Y)·z₂ + cg2·Ẋ₂
  τ_Θ_friction = (Fc₁(Y)·z₁ − Fc₂(Y)·z₂) · Lb/2 + (cg1·Ẋ₁ − cg2·Ẋ₂) · Lb/2
```

**Why**: Five candidates were evaluated; the friction states were the only choice satisfying all criteria simultaneously:
1. Genuine dynamic states (own ODE, memory — not computable from current [X,Θ,Y,Ẋ,Θ̇,Ẏ])
2. Creates coupling: asymmetric Fc₁(Y) ≠ Fc₂(Y) when Y ≠ 0 generates Y-dependent torque on Θ from X motion
3. Position-dependent: coupling amplitude varies with Y, enriching the LPV structure (C(Y) alongside M(Y))
4. Direction-sensitive: z₁, z₂ carry history through direction reversals (pre-sliding transient)
5. Physically motivated: load distribution N₁(Y), N₂(Y) on X-guides changes with payload Y — documented in gantry literature
6. Directly connects to D-025 (supervisor's hysteresis observation) as the proper dynamic formulation of sign(Ẏ) scheduling
7. Exact Jan-analogy: extra states in data generator (absent from baseline), augmentation must rediscover them

**Ruled out**:
- *Support structure resonance [x_b, ẋ_b]*: Garcia's 37.7 Hz die-cast base resonance is specific to his rig; Telica uses granite/polymer-concrete frame with first resonance >100 Hz, above control bandwidth. No Y-dependence — does not enrich LPV structure.
- *Cross-arm bending mode [δ, δ̇]*: Garcia explicitly calls cross-arm vibration "negligible in comparison to the coupling between actuators." Building the simulation study on a phenomenon the original paper dismisses is a weak foundation.
- *Coriolis coupling (Ẏ·Θ̇ terms)*: Not a state — a static nonlinear function of existing states. A non-dynamic augmentation could capture it without extra states. Reserved for second augmentation step (D-024).
- *sign(Ẏ)*: Not a state — a static (memoryless) nonlinearity. Already approximately modelled as Coulomb friction in the baseline. The friction states [z₁, z₂] are the correct dynamic version that captures the hysteresis memory sign(Ẏ) approximates.

**Constrains**:
- Data generator implementation extends `rk4_step` / `lfr_simulate.py` to an 8-state variant; the 6-state baseline code is NOT modified.
- Augmentation interconnect uses `nxd=2` extra states (analogous to Jan's `nxd=2` for m₃ in MSD).
- Verification: true z₁(t), z₂(t) from the data generator are saved and compared against the augmentation's learned states.
- Key metric: Θ prediction error as a function of Y-position and motion direction.
- Parameter g (Dahl stiffness) and Fc (nominal Coulomb amplitude) must be chosen to produce a physically plausible but clearly observable effect — suggested range: g ≈ 1–5 μm (pre-sliding displacement), Fc ≈ 10–30 N.
- Cross-references: D-022 (extra states in augmentation, not baseline), D-023 (validate parameter recovery before augmentation), D-024 (friction study is the first augmentation demonstration), D-025 (friction states are the dynamic formulation of hysteresis scheduling).

---

### [D-039] Feedback controller operating point per trajectory: Y_initial
**Date**: 2026-04-17
**What**: In `export_lpv_multi_traj.m`, the feedback controller `Cfb` and frozen LTI `G`
are designed at `Y_op = sp.Y_initial` for each trajectory — the Y position at the start
of the main motion. This replaces the previous single frozen choice of `Y_op = 0.3` for
all trajectories.

| Trajectory | Y_initial | Cfb designed at |
|---|---|---|
| T1 | 0.3 | Y = 0.3 |
| T2 | 0.3 | Y = 0.3 |
| T3 | 0.0 | Y = 0.0 |
| T4 | 0.2 | Y = 0.2 |
| T5 | 0.2 | Y = 0.2 |
| T6 | 0.3 | Y = 0.3 |

**Why**: Designing at `Y_op = 0.3` for all trajectories is unnecessarily wrong for T3
(Y=0.0), T4 (Y=0.2), T5 (starts at Y=0.2). Using `Y_initial` gives each trajectory a
controller optimally matched to its operating condition without requiring any Simulink
changes — `Cfb` and `G` are still plain workspace variables.

**Ruled out**:
- *Single Y=0.3 for all*: unnecessarily off-design for T3/T4/T5.
- *Gain-scheduled LPV controller Cfb(Y)*: the correct solution for trajectories where
  Y varies during motion (T1, T5, T6). Requires replacing the fixed LTI `Cfb` block in
  Simulink with an online-scheduled controller (S-function or MATLAB function block).
  Not implemented because it requires modifying the Simulink model, which is out of
  scope for the current parameter recovery phase.

**Constrains**:
- For T1, T5, T6 where Y actively sweeps during the main motion, `Cfb` at `Y_initial`
  is still an approximation — the controller is off-design-point as Y moves. This is
  accepted for now; the recorded `(u_q1, q1)` pair remains a valid input-output dataset
  for parameter recovery regardless of controller quality, since both signals are saved
  exactly as simulated.
- If gain-scheduled control is added later, `Cfb` computation must move inside the
  trajectory loop and be evaluated online using the current Y state.

---

### [D-040] torch.compile on rk4_step deferred — hardware constraint
**Date**: 2026-04-18
**What**: `@torch.compile(fullgraph=True, dynamic=False)` was added to `rk4_step` as
Phase 2 of the Step 3c training speed optimization. It has been removed and deferred.
**Why**:
- Training GPU is a Quadro P2000 (CUDA Capability 6.1). Triton requires CC ≥ 7.0 (Volta+).
  `backend='inductor'` fails with: *"Found Quadro P2000 which is too old to be supported
  by the triton GPU compiler"*.
- CPU path also blocked: MSVC `cl.exe` is not installed on this Windows machine; TorchInductor
  cannot compile C++ kernels for the CPU fallback.
- `backend='aot_eager'` works on both but provides no kernel fusion — only Python dispatch
  overhead reduction, which is negligible on a GPU-bound workload.
**What WAS completed (kept)**: Phase 1 (GMatrix → (15,15) tensor refactor) is complete
and stays. It reduces the buffer count from 7 to 1 in `lfr_block.py`, simplifies the API,
and is the necessary prerequisite for Triton kernel fusion once hardware is upgraded.
**Ruled out**: `aot_eager` as a permanent solution — it provides ~0% speedup on CUDA.
**Re-enable when**: Training moves to a Volta/Turing/Ampere GPU (CC ≥ 7.0). The code
comment in `lfr_simulate.py` contains the exact decorator to uncomment.
**Known issue to fix on re-enable**: `rk4_step` is called in both gradient (training loop)
and no-grad (eval pass) contexts. With the default `cache_size_limit=8`, this triggers
`GLOBAL_STATE changed: grad_mode` recompilations that eventually raise `CacheLimitExceeded`.
Fix: use `options={"cache_size_limit": 4}` in the decorator — allows grad/no-grad × dtype
specializations without restructuring the call sites. No logic change needed.
**Constrains**: `lfr_simulate.py` — the commented-out decorator block must not be removed;
it documents the intended optimization for future hardware.

---

### [D-041] Physics computation kept in float64 — float32 not precise enough
**Date**: 2026-04-18
**What**: All physics in `rk4_step` and `lfr_forward` (the polynomial loop solve, RK4
integration, matrix products) is computed in float64. The Jan framework uses float32
throughout; explicit casts are applied at the block boundary in `lfr_block.py` and
`lfr_param_block.py` (float32 → float64 on entry, float64 → float32 on exit).
**Why**:
- The polynomial loop solve `N(Y)/d(Y)` uses Horner evaluation of the adjugate matrix
  (N0, N1, N2) and the scalar determinant polynomial d(Y). These involve subtraction
  of near-equal terms and division by a scalar that can be small near the limits of the
  Y operational range. float32 provides only ~7 decimal digits of precision — insufficient
  to guarantee numerical accuracy of the solve across the full Y range and over long
  trajectories (4000 RK4 steps per segment).
- RK4 integration accumulates truncation error per step; float32 rounding adds a second
  error source on top. Over 4000 steps at ts = 1/16 kHz the accumulated float32 error
  has not been validated against the required parameter recovery accuracy.
- Physical parameters (masses ~10–25 kg, stiffnesses ~2000 N/m) span two orders of
  magnitude. float32 relative error (~1e-7) translates to absolute errors that may not
  be negligible for gradient-based parameter recovery where small parameter deltas matter.
**Ruled out**: float32 physics — not validated, risk of gradient degradation during
parameter recovery training. The Quadro P2000 has 1/32 fp64-to-fp32 throughput ratio
(Pascal), so float32 would be significantly faster, but correctness must come first.
**Future investigation**: If training speed becomes a bottleneck after moving to better
hardware (or if float64 remains slow), run a controlled experiment:
1. Train with float64 (reference), record `param_table()` and val RMSE per epoch.
2. Remove the two cast lines in `lfr_block.py` to run entirely in float32.
3. Compare `param_table()` — if parameters agree to within ~0.1% and RMSE curves match,
   float32 is acceptable and the cast lines can be removed permanently.
The comment in `lfr_block.py` marks the exact two lines to change.
**Constrains**: `lfr_block.py` and `lfr_param_block.py` — the float32↔float64 cast lines
must not be removed without the above validation. `lfr_forward.py` and `lfr_simulate.py`
need no changes; they operate on whatever dtype the caller passes.

---

### [D-042] Training loss normalized by per-channel output standard deviation (sigma)
**Date**: 2026-04-20
**What**: The MSE training loss in `train_param_recovery.py` is computed in sigma-normalized space:
```python
sigma = std of q1 across all 6 TRAJ_SPECS trajectories, per channel  # (3,) float64 tensor
err   = (Y_pred - q1_seg) / sigma
mse_loss = err.pow(2).mean()                                           # dimensionless
```
`sigma` is computed over the **full trajectory set** (all TRAJ_SPECS, not just ACTIVE_TRAJ_IDS) and cached to disk. It does not change when the active trajectory subset is changed.

**Why**: The three output channels [X1, X2, Y] are in metres but have different signal amplitudes. Without normalization the Y channel (largest excursion) dominates the loss, pulling parameter gradients toward Y-related parameters (mh, cy) at the expense of X-related ones (m1, m2, cg1, cg2). Dividing by sigma gives each channel unit variance, so MSE contribution is proportional to relative prediction error, not absolute channel amplitude.

Using the full TRAJ_SPECS for sigma (not the active subset) means:
- Sigma is stable regardless of which trajectories are active — no cache invalidation when ACTIVE_TRAJ_IDS changes.
- Sigma represents the full operating envelope of the system, not just the subset being trained on.

**Connection to D-034 (RMSE_baseline_normalized)**: Because the loss is dimensionless, the RMSE_baseline passed to `ParameterizedLFRBlock` must also be in sigma-normalized units. `rmse_baseline_normalized` is computed by `_aggregate_normalized_rmse_baseline()`, which applies the same per-channel sigma division to the per-trajectory RMSE before aggregating. This is the value passed to the block — not the metre-space `rmse_baseline`. See D-034 for the full Lambda calibration rationale.

**Ruled out**:
- *No normalization*: Y channel dominates; X1/X2 parameter gradients are suppressed.
- *Global scalar normalization*: a single scalar (e.g. overall std) does not correct the per-channel imbalance.
- *Normalizing by active-subset sigma*: sigma would shift when ACTIVE_TRAJ_IDS changes, making Lambda (which is fixed at block construction) inconsistent across runs.

**Constrains**: `train_param_recovery.py` — `sigma` must always be computed from the full TRAJ_SPECS, not the active subset. The `SIGMA_CACHE_VERSION` constant must be incremented if TRAJ_SPECS itself changes. Any future training script for this system must apply the same sigma normalization and pass `rmse_baseline_normalized` (not metres) to the block.

---

### [D-043] Checkpoint/epoch selection strategy for parameter recovery training
**Date**: 2026-04-20
**What**: Three decisions about which parameter vector to save and how to track convergence:

1. **Current phase (clean MATLAB data):** Use **Polyak-Ruppert tail averaging** over the plateau phase. Start averaging on the first LR reduction event from `ReduceLROnPlateau` — this trigger is automatic and requires no additional hyperparameter. The averaged `log_params` are saved alongside the final-epoch `log_params` in the `.pt` file. This is not yet implemented.

2. **Convergence tracking:** Run a full-trajectory eval (same as step 5) every `PARAM_LOG_INTERVAL` epochs. Save the result in `history`. This gives a clean convergence curve comparable to the final step 5 result, and provides the signal for best-epoch tracking if needed. This is not yet implemented.

3. **Future phase (measurement noise):** Polyak averaging over the late plateau becomes harmful — the late iterates are corrupted by semi-convergence (the optimizer fits noise after exhausting the clean signal). Switch to early stopping:
   - Known noise variance → **Morozov Discrepancy Principle**: halt when the smoothed training residual hits the noise floor `τ·δ²`.
   - Unknown noise variance → **L-curve method**: log `(residual norm, solution norm)` at each epoch; find the corner post-training. This requires logging `‖log_params‖` (or deviation from init) alongside the loss in `history`. The `log_params_snapshot` already saved at `PARAM_LOG_INTERVAL` supports this.

**Why:**
- **Saving last epoch is not principled.** The last epoch may not be optimal: the stochastic 8-segment train loss has high variance, and `ReduceLROnPlateau` does not guarantee the last iterate is the best. Last ≈ best only if LR has fully decayed to `min_lr` — which may not happen within 2000 epochs.
- **Best-epoch on stochastic train loss is actively wrong.** It rewards lucky random batches, not genuine parameter improvement. Confirmed by both the subagent research and Gemini Deep Research.
- **Polyak tail averaging is theoretically optimal for clean data.** For a 13-parameter, physics-constrained, locally convex problem, iterate averaging achieves the Cramér-Rao lower bound. It cancels the zero-mean batch noise algebraically without any additional computation beyond a running sum of 13 scalars.
- **Full-trajectory eval every PARAM_LOG_INTERVAL solves two problems at once:** the convergence plot becomes directly comparable to the step 5 final result, and it provides a stable signal for best-epoch tracking that is immune to batch sampling noise.
- **Semi-convergence is a real risk when noise is added.** With only 13 parameters, structural overfitting cannot occur. But the optimizer will eventually start fitting measurement noise rather than physics — "clean priority learning" means accuracy peaks mid-training, not at the end. Polyak averaging the corrupted plateau would amplify this effect.

**Ruled out:**
- *Best-epoch on stochastic train loss:* rewards sampling variance; statistically invalid for epoch selection.
- *Best-epoch on fixed held-out segment set:* computationally wasteful per epoch; vulnerable to trajectory divergence and the same noise issue as the train set (just with a fixed random seed instead of a varying one). Less principled than full-trajectory eval.
- *Schedule-Free optimizer (Defazio 2024):* eliminates epoch selection entirely by unifying momentum and iterate averaging — promising but not implemented. Would remove `ReduceLROnPlateau` and its associated patience/factor hyperparameters. Deferred as a future experiment.
- *Stochastic Weight Averaging (SWA) with cyclical LR:* correct in principle but requires replacing `ReduceLROnPlateau` with a cyclical schedule. More disruptive to the current setup than Polyak tail averaging which re-uses the existing scheduler trigger.

**Constrains:**
- `train_param_recovery.py`: add `averaging_active` flag, `AveragedModel` from `torch.optim.swa_utils`, triggered by first LR reduction. Save `averaged_log_params` in the `.pt` file.
- `train_param_recovery.py`: add full-trajectory eval loop inside the `PARAM_LOG_INTERVAL` block. Save per-trajectory RMSE snapshots in `history`.
- When noise is added: `history` must log solution norm `‖log_params − log(params_init)‖` per epoch to support L-curve analysis post-training. The `log_params_snapshot` at `PARAM_LOG_INTERVAL` already provides this at coarser resolution.
- Both `params_learned` (last epoch) and `params_learned_avg` (Polyak average) must appear in the final `.pt` save so results can be compared.

---

### [D-044] Multi-trajectory loss function: binary masking + per-trajectory per-channel sigma
**Date**: 2026-04-21
**What**: Replace the current global-sigma unweighted MSE loss with a loss that applies
binary channel masks per trajectory group, normalizes by per-trajectory per-channel signal
std, and averages per segment before averaging over the batch.

**The six problems with the current implementation (global sigma, no masking):**

1. **Dormant channels included in the loss.** On T1/T6 (Y-only), X1 and X2 are actively
   suppressed by the feedback controller but contribute equally to the MSE. The optimizer
   receives gradient signal from controller suppression dynamics rather than plant physics,
   pulling physical parameters away from their true values.

2. **Global sigma dilutes Y, inflates X.** sigma[Y] is computed from all 6 trajectories
   including T2/T3/T4 where Y is constant → sigma[Y] is artificially small → Y is
   over-weighted. sigma[X1] is computed across all 6 trajectories including T1/T6 where
   X1 ≈ 0 → sigma[X1] is artificially large → X1 is under-weighted on trajectories where
   it is actually active. Both biases compound simultaneously.

3. **Within-trajectory amplitude imbalance.** On T5 (X + Y both active), if Y sweeps much
   more than X1/X2, Y dominates the loss. Parameters primarily identified by X motion
   (m1, m2, cg1, cg2) are undertrained relative to Y-related parameters (mh, cy).

4. **Cross-trajectory amplitude imbalance.** Trajectories with the same active channels
   can have very different amplitudes (T1 conservative vs T6 aggressive Y sweep). A single
   global sigma[Y] does not capture this: T6 segments always dominate T1 segments in the
   loss, even though both are Y-only trajectories contributing equal information about Y.

5. **Denominator is inconsistent across segments.** Different segments have different numbers
   of active channels (T1: 1 active, T2/T3/T4: 2 active, T5: 3 active). A fixed global
   denominator gives unequal weight per active channel-step across trajectory groups. No
   single global denominator is correct for all segments simultaneously.

6. **Adam sees inconsistent loss scale across batches.** With 8 segments sampled from
   different trajectory groups per batch, the loss magnitude depends on which groups appear.
   Without per-segment normalization, Adam's second moment estimate v_t cannot stabilize,
   making its adaptive learning rate unreliable.

**Why**: Problems 1–6 compound. Problems 1 and 2 corrupt the gradient direction. Problems
3 and 4 create systematic undertraining of specific parameter subsets. Problems 5 and 6
make Adam's adaptation unreliable across epochs. The combination means the optimizer is
simultaneously given wrong gradient directions AND wrong step sizes.

**Chosen solution:**
```
For each segment in the batch:
  1. Binary mask:  zero out dormant channels for this trajectory group
  2. Normalize:    divide residual by sigma[traj_id][channel]
                   (sigma computed from that trajectory individually, active channel only)
  3. Per-segment loss = masked_normalized_err².mean() over (active_channels × T)
Average segment losses over the batch.
```

Formally:
```
loss = (1/B) Σ_i [ (1 / (n_active_i · T)) Σ_c Σ_t  m_{g,c} · ((ŷ_c - y_c) / σ_{traj,c})² ]
```

where m_{g,c} ∈ {0,1} is the binary mask for channel c in trajectory group g,
and σ_{traj,c} is the std of channel c computed from that trajectory only.

**Why per-trajectory sigma solves problems 3 and 4:** Each trajectory's sigma reflects
its own excitation amplitude. T6's sigma[Y] ≈ 300 mm; T1's sigma[Y] ≈ 50 mm. After
normalization, a 30 mm residual on T6 contributes (30/300)² = 0.01 — equal to a 5 mm
residual on T1 contributing (5/50)² = 0.01. Equal relative contribution regardless of
absolute excitation amplitude.

**Why per-segment averaging solves problems 5 and 6:** Each segment contributes O(1) to
the loss regardless of how many active channels it has. Adam sees a consistent loss
magnitude across all batches regardless of trajectory group composition. The second
moment estimate v_t stabilizes correctly.

**Forward compatibility (future hardware data):** When moving to real measurements with
additive noise, per-trajectory sigma transitions directly to the principled Λ⁻¹ weighting
(Ljung 1999 §7.4, Gautier, Janot & Vandanjon 2013). At high SNR (gantry encoders:
signal mm–cm, noise µm), signal std ≈ noise-floor-independent scale → per-trajectory
sigma is the high-SNR approximation of Λ⁻¹ weighting. No architectural change required
at the transition to hardware data; only the interpretation of sigma changes.

**Literature support:**

*Problem 1 — Dormant channel masking in gradient-based SysID (verified by direct quote):*
- **Werling et al., "Trajectory-based actuator identification via differentiable
  simulation"** (PDF p. 5, Eq. 2 and p. 12, Appendix B): loss `L = (1/MN) Σ ‖W(s'−s)‖²`
  with `W = diag(w_q, w_qdot)`; set to `diag(1, 0)` so velocity remains in the rollout
  but *"velocity residuals are not penalized because the measured velocity signal is
  noticeably noisier than position."* Directly confirms: mask in the loss, keep in the
  dynamics. Optimizer: Adam (Appendix B).
- **Gautier & Khalil (1990)** — dormant joints produce structural zeros in the regressor
  (classical least-squares analog). Forssell & Ljung (1999) additionally applies when
  measurement noise is present (closed-loop bias-pull mechanism).

*Problems 2 & 3 — Amplitude normalization across channels in gradient-based SysID (verified):*
- **Lutter et al., "Dynamic Modeling of Robotic Manipulator via an Augmented Deep
  Lagrangian Network"** (PDF p. 4, Eq. 8): Mahalanobis norm with diagonal covariance
  matrix W_τ; explicit justification: *"It is necessary to normalize the loss function
  using covariance matrix since the torque magnitude may vary greatly from joint to joint."*
- **Lutter et al., "Combining Physics and Deep Learning to learn Continuous-Time Dynamics
  Models" (Deep Lagrangian Networks, IJRR)** (PDF p. 7, Eq. 12): same Mahalanobis norm
  with diagonal W_τ; *"It is beneficial to normalize the loss using the covariance matrix
  because magnitude of the residual might vary between different joints."*
- **"Constrained Gray-Box Identification of Electromechanical Systems Under Unfiltered
  Step-Response Data"** (PDF pp. 6–7, Eq. 3): normalized composite residual dividing
  trajectory errors by `RMS(signal)` per channel; *"naturally balances the relative
  contribution of current and velocity; thus α_ω = α_i = 1 is sufficient and avoids
  additional manual scaling."*

*Problems 5 & 6 — Segmented minibatch objective for Adam consistency (verified):*
- **Werling et al. (above)**, Eq. 2: loss averaged over M segments and N timesteps as
  `(1/MN) Σ_j Σ_i ‖W(s'_{i,j} − s_{i,j})‖²` — each segment normalized independently
  before batch average. Adam confirmed as optimizer (Appendix B).

*Problem 4 — Cross-trajectory amplitude imbalance:*
- **No exact citable method found** that matches all of: multiple trajectories + same
  active channels + different amplitudes + joint gradient-based physical parameter ID +
  trajectory-specific normalization in the training loss.
- **Citable principle — experiment-balanced weighting:** adjacent inverse-identification
  literature explicitly supports the broader principle that multiple experiments should
  contribute in a balanced or uncertainty-weighted way to the cost function, rather than
  in proportion to raw residual magnitude:
  - **Zhang et al., Int. J. Solids Struct. (2023), doi:10.1016/j.ijsolstr.2023.112534**:
    explicitly states that good inverse-identification results depend on *"maintaining
    equal contribution of the strain states from each experiment to the cost function"*
    — the clearest paper-level support for equal cross-experiment contribution.
  - **Neggers et al., Mech. Mater. (2019), doi:10.1016/j.mechmat.2019.03.001**:
    when combining multiple experiments and data sources, weighting should follow
    measurement uncertainty derived from a Bayesian formulation — citable basis for
    experiment-wise balancing rather than raw aggregation.
- **Framing for thesis:** per-trajectory sigma normalization is an engineering
  realization of experiment-balanced weighting — supported in adjacent inverse-
  identification literature as a principle, but not a canonical standard method in
  robot gradient-based SysID. It is not "uncited" but it is also not "established."

*Supporting context — gradient-based physical SysID as established paradigm (verified):*
- **Muratore et al., "Differentiable Simulation for Physical System Identification"
  (RA-L 2021)** (PDF p. 6, Sec. IV-B): friction and mass estimated by backpropagating
  MSE loss through differentiable simulator via PyTorch AD; Adam optimizer.
- **Saveriano et al., "Physics-informed online learning of gray-box models by moving
  horizon estimation" (EJC 2023, 100861)** (PDF pp. 3–4): physical submodel + neural
  network trained via BPTT; arrival cost covariance *"can be seen as an adaptive
  learning-rate."*
- **Ljung (1999) §7.4 eq. (7.27)** — Λ⁻¹ weighting of multi-output prediction errors
  (classical PEM; per-trajectory sigma is the high-SNR approximation of this).
- **Gautier, Janot & Vandanjon (2013), IEEE TCST** — per-joint inverse-std normalization
  *"normalises the errors"* in closed-loop robot ID (regressor analog).

**Ruled out:**
- *Global sigma (D-042):* contaminated by inactive-channel samples for every channel
  (Problems 1–4). Documented as the identified flaw in D-042.
- *Per-channel-global sigma (no per-trajectory split):* solves Problems 1–2 partially
  but not Problems 3–4. T6 still dominates T1 after normalization.
- *Per-segment sigma (normalize each segment by its own std):* independently normalizes
  each segment but breaks Adam — momentum estimates are built from segments with
  incompatible normalization bases, corrupting gradient direction across batches.
- *GradNorm (Chen et al. 2018):* correct in principle but requires computing ‖∂L_i/∂θ‖
  through the RK4 graph at every step — expensive and unverified on physical grey-box
  sensitivity Jacobians.

**Constrains:**
- `train_param_recovery.py`: precompute `sigma[traj_id][channel]` from each trajectory's
  active samples before training. Pass trajectory ID with each segment in the batch.
- Loss function must use per-segment averaging (Option B), not global averaging (Option A).
- When hardware data is available: replace sigma computation with noise std estimated from
  static measurements; loss architecture unchanged.

**Implemented**: 2026-04-21 in `lpv_lfr_baseline/scripts/train_param_recovery.py`.
Changes: `CHANNEL_MASKS` dict (6 changes), `_get_or_compute_sigma` rewritten to return
`{traj_id: (3,) tensor}` (SIGMA_CACHE_VERSION bumped to 2), sigma display table updated,
`sample_plan` captured in training loop, per-segment loss loop replacing 2-line MSE,
`_aggregate_normalized_rmse_baseline` updated to mask + per-trajectory sigma.
Verified: sigma table output correct (dormant channels = 1.0 m, active channels physically
meaningful); exit code 0; loss value is O(1) per segment.

---

### [D-046] Multi-mode crest factor not fixed for simulation; fix specified for hardware
**Date**: 2026-04-30
**File**: `Matlab-scripts/export_param_recovery_inject_ref.m`, function `generate_ref_multisine`

**What**: The multisine generator designs each spatial mode (common, diff, y) as an
independent Schroeder-phase odd-harmonic signal. When two modes are combined on the same
actuator channel, the combined signal is no longer guaranteed to be Schroeder-optimal.
The only affected trajectory is T8 (`ms_modes = {'common', 'diff', 'y'}`), where:

```
X1_ms = common_sig + diff_sig   (two modes, overlapping frequency bands at 1-20 Hz)
X2_ms = common_sig - diff_sig
Y_ms  = y_sig                   (single mode, no issue)
```

T1-T7 each assign at most one mode per actuator channel, so T8 is the only case.

**Why the gap exists**: Schroeder phases minimize crest factor for a single multisine
signal. When two Schroeder signals with different seeds are summed, the combined CF
is not guaranteed to be ~1.58. The seed-based phase offset in the script provides partial
decorrelation between modes:

```matlab
phi = phi + 2*pi*freqs*(seed - 1)/(7*f_high);
```

This is a linear phase ramp (time shift) that decorrelates modes but does not produce
a Schroeder-optimal combined signal.

**Why we are not fixing it for simulation**: The kinematic pre-check (`check_ref_total`)
evaluates the actual position, velocity, and acceleration of `r_total = r_traj + r_ms`
before each simulation. Any elevated peak caused by non-optimal CF is caught there and
stops the amplitude sweep. For noise-free simulation data, crest factor is a hardware-safety
metric, not a parameter identifiability metric. The sweep already enforces the binding
constraint (kinematics), so the CF gap has no practical consequence in the current pipeline.

**Ruled out for simulation**: Interleaved frequency grids and per-channel numerical
phase optimization. Both add complexity with no measurable benefit when `check_ref_total`
already catches kinematic violations.

**Fix for hardware experiments**: Use interleaved odd harmonics to eliminate frequency
overlap between modes on the same channel. For T8 with two X-modes:

```
common mode: odd harmonics 1, 5,  9, 13, 17 Hz  (every other odd)
diff   mode: odd harmonics 3, 7, 11, 15, 19 Hz  (interleaved)
```

Combined on X1: harmonics at 1, 3, 5, 7, 9, 11, 13, 15, 17, 19 Hz with no overlap.
Each mode's Schroeder phases apply to non-overlapping lines, so the combined CF is
still bounded by the per-mode Schroeder construction.

Cost: each mode gets half the lines in the shared band (1-20 Hz). For diff mode this
gives 5 lines instead of 10 in 1-20 Hz, which remains above the F >= 7 guard only
if the full common mode band (1-100 Hz) is counted. If the F >= 7 guard is applied
per-mode, the diff band would need to be widened or the grid adapted. Verify the
guard on the actual interleaved line count before implementing.

**Constrains**: For all hardware experiments on T8 involving simultaneous common and diff
modes, switch to interleaved odd harmonics in `generate_ref_multisine`. For simulation,
no change required.

---

### [D-045] param_loss disabled (PARAM_LOSS_WEIGHT = 0.0) for parameter recovery training
**Date**: 2026-04-22
**What**: `PARAM_LOSS_WEIGHT = 0.0` in `train_param_recovery.py` — `param_loss()` is not
added to the training loss. The method exists on `ParameterizedLFRBlock` but is bypassed.

**Why**: `param_loss` is a Lambda-weighted L2 pull toward `params_init` (the detuned
initial values). It was designed for the noisy-data regime, where the MSE landscape is
rough and the optimizer needs an anchor to stay in a physically plausible region.

In the parameter recovery setting:
- Training data is noise-free MATLAB simulation output.
- The Python model reproduces MATLAB exactly at the true parameter values (verified:
  full-trajectory RMSE on T1 = 0.000 mm at `_TRUE_PARAMS`).
- The MSE landscape therefore has an unambiguous global minimum at the true parameters.

Under these conditions, `param_loss` provides no benefit and actively harms convergence:
it adds a competing gradient pull toward `params_init` (the detuned values, ±10% from
true), which is the wrong target. The stronger the regularization weight, the further the
optimizer is biased away from the true parameter values.

**Ruled out**: Enabling `param_loss` at any non-zero weight for noise-free parameter
recovery — it anchors toward detuned init, not toward truth, and slows or prevents
convergence to the true parameters.

**When to revisit**: If training data gains additive measurement noise (encoder noise,
etc.) and the MSE landscape becomes rough or ill-conditioned, a small `param_loss` weight
anchored toward physically plausible values may help stability. At that point, `params_init`
should ideally be updated to the best currently known parameter estimate rather than the
detuned starting values, to avoid the wrong-anchor problem documented here.

---

### [D-047] Parameter sensitivity diagnostic removed from experiment_diagnostics.py
**Date**: 2026-05-03
**What**: `_diag_param_sensitivity` was implemented and then removed. The final
`experiment_diagnostics.py` contains three diagnostics only: FFT, step response,
and observability. Segment length is determined from the step response oscillatory
frequency alone.

**Why it was built**: An attempt to determine the minimum segment length rigorously —
by computing `∂y/∂log(θᵢ)` for each of the 14 parameters over time (via finite
differences through `simulate_frozen`), finding the time `t_95` at which 95% of
cumulative sensitivity energy is captured, and setting `segment_len = t_95_max`.

**Why it was removed**:
1. **Not supervisor-suggested.** Supervisors explicitly recommended FFT + step response.
   Parameter sensitivity was an independent addition from research reasoning, not
   requested or validated by supervisors. Their guidance: keep it simple, don't solve
   problems you are not facing.
2. **Slow.** 14 parameters × 8 trajectories × 2 forward passes = 224 `simulate_frozen`
   calls per diagnostic run. On CPU eager mode this takes several minutes.
3. **Result was unusable.** `t_95` for all parameters hit the T_test cap of 2.0 s
   (the full decimated trajectory length), meaning sensitivity never converged within
   the available data. The diagnostic returned `segment_len ≈ 39420 samples at 20000 Hz`
   — essentially the full trajectory — giving only 1 segment per trajectory and no
   meaningful segment pool.
4. **Wrong reference timescale.** An earlier version used `segment_len = max(10×tau_max,
   t_95_max)`, which produced 314436 samples (15.7 s) — longer than the trajectories
   entirely. Even after removing the 10× multiplier, the result was still impractical.

**What replaced it**: Segment length is derived from the oscillatory poles in the step
response. The slowest oscillatory frequency `f_osc_min` is extracted from the complex
eigenvalues of `A_c` at each frozen Y operating point. Segment length is then:

    segment_len_s = N_PERIODS / f_osc_min

with `N_PERIODS = 3` (configurable). At `f_osc_min ≈ 4.94 Hz` (Y=0.30 m):
`segment_len_s ≈ 0.61 s → 610 samples at 1000 Hz`. This gives multiple segments per
2 s trajectory and is consistent with the supervisor-recommended approach.

**Ruled out**: Re-enabling sensitivity in any form unless supervisors specifically request
it and longer trajectories are available (so t_95 can actually converge).

**Constrains**: `recommend_segment_len` now only calls `_diag_step_response` — it no
longer requires trajectory data as input (only `fs` and `dtype`). The function signature
changes accordingly.

**Constrains**: `PARAM_LOSS_WEIGHT = 0.0` must be kept for all clean-data parameter
recovery runs. If re-enabled, the anchor target (`params_init`) and weight must be
revisited together.

---

### [D-049] experiment_diagnostics.py: fs_new derived from system physics, not signal content
**Date**: 2026-05-08
**What**: Restructured `experiment_diagnostics.py` in five concrete ways:

1. `fs_new` is now determined from `f_osc_min` (pole analysis, Diagnostic 2) using
   `_FS_RULE_FACTOR = 10`: first candidate in `_FS_CANDIDATES` satisfying
   `fs_new >= 10 * f_osc_min`. Previously, `fs_new` was set from `f_99` (Welch PSD,
   Diagnostic 1) with `_FS_RULE_FACTOR = 8`.

2. `_FS_RULE_FACTOR` changed from 8 to 10 to match the lecture lower bound.

3. `segment_len` is now the maximum of three rules:
   ```python
   segment_len = max(
       ceil(N_PERIODS / f_osc_min * fs_new),   # period rule
       ceil(10 * tau_max * fs_new),              # 10x time constant rule
       10 * n_params,                            # 10x parameter count rule
   )
   ```
   Previously only the period rule was applied (yielding ~608 samples at 1000 Hz).
   With the 10x tau_max rule the correct lower bound is ~15720 samples at 1000 Hz.

4. `f_99` demoted to a warning-only check: if `f_99 > 10 * f_osc_min`, a warning is
   printed that excitation energy is above the model band. `f_99` no longer drives
   any design variable.

5. `[::D]` stride in `_diag_gradient_convergence` replaced with
   `scipy.signal.decimate`, which applies a Chebyshev Type I anti-aliasing filter
   before striding.

**Why**:

*For change 1 and 2:*
- Source: Lecture 9, slides 10-12 (5SMB0): "10 * omega_b <= omega_s <= 30 * omega_b"
  where omega_b is the system bandwidth — a physics quantity, not a signal quantity.
- `f_99` is the 99% energy frequency of the excitation. It measures where the
  injected signal has power, not where the system has dynamics. Setting `fs_new` from
  `f_99` ties the sampling rate to the excitation design rather than the model band.
  This is the wrong causal direction: the sampling rate should be set first (from
  physics), and then the multisine frequency range should be designed to stay within
  the model band.
- Factor 8 is below the lecture-stated lower bound of 10. Factor 10 is used.
- Source: Ljung (1999) — setting fs too high causes all discrete-time poles to cluster
  near unity, degrading numerical conditioning.
- Source: Pintelon & Schoukens (2001/2012) — set fs from the model band, not the
  excitation band.

*For change 3:*
- Source: Lecture 9, slide 9 (5SMB0): "N >= 10 * tau_set,95" and "N >= 10 * n_theta".
- Source: Lecture 3, periodic measurement material (5SMB0) — integer periods required.
- N_PERIODS = 3 is a HEURISTIC (covers the slowest mode with margin; lecture uses 10
  for FRF quality, which is more conservative than needed for BPTT training).
- The 10x tau_max rule dominates at the current parameters: tau_max = 1.572 s,
  giving 15720 samples at 1000 Hz — 25x larger than the period rule alone.
  This may be overly conservative for BPTT (the rule is derived for stationary FRF
  estimation). The discrepancy is now reported in the diagnostics output and should
  be discussed with the supervisor before shortening trajectories.

*For change 4:*
- Source: Gonzalez, van Haren, Oomen, Rojas (arXiv:2410.19629 / IEEE TAC 2024):
  parametric estimator consistency survives aliasing of out-of-band input content,
  provided in-band frequencies are correctly resolved. Therefore `f_99` above the
  model band is not a problem for parameter recovery, only a warning.

*For change 5:*
- Source: Lecture 9 (5SMB0) pre-processing steps: "Apply anti-aliasing filter before
  any downsampling."
- Source: lecture_digital-filters.pdf (4CM00), slides 30-35: filter must provide
  >= 40 dB attenuation at the new Nyquist frequency.
- `scipy.signal.decimate` applies Chebyshev Type I filter automatically.

**Ruled out**:
- `_F99_PHYSICAL_CAP_FACTOR`: applying the 10x rule to `f_99` to cap it at
  `10 * f_osc_min`. Documented in `docs/multisine-diagnostics-interface.md` —
  this applies the 10x rule to the wrong variable and conflates two separate
  design choices.

**Constrains**:
- `experiment_diagnostics.py`: `run_all_diagnostics` now computes `f_osc_min`,
  `fs_new`, and `D` before calling `_diag_fft`. `_diag_fft` accepts `fs_new` and
  `f_osc_min` as keyword parameters.
- `recommend_segment_len`: now returns the max-of-three segment_len, which is larger
  than before. Any caller that relies on the old (period-only) segment length will get
  longer segments and fewer segments per trajectory. This is the correct direction.
- Note: the 10x tau_max rule may produce segments longer than available trajectory
  data (tau_max = 1.572 s => 15720 samples at 1000 Hz; trajectories are approximately
  40000 samples at 20000 Hz = 2000 samples at 1000 Hz). This is a trajectory design
  issue, not a code issue — the diagnostic now correctly reports it.

---

### [D-048] `ref_injection` dataset is incompatible with open-loop parameter recovery training
**Date**: 2026-05-04
**What**: The `ref_injection` dataset (multisine injected into the reference `r`) is
fundamentally incompatible with the open-loop simulation objective used in
`train_param_recovery.py`. The `multisine` dataset (force injection via `f_sim`) is
the correct choice for parameter recovery.

**Why**: The training minimises `||simulate(x0, u_recorded, params) - q1_recorded||²`
open-loop. In `ref_injection`, within the controller bandwidth (≤ 100 Hz):

    u_ms = C * S * r_ms ≈ 0          (sensitivity S ≈ 0 kills the force)
    q1_ms = T * r_ms ≈ r_ms          (position closely tracks reference)

The open-loop model receives a near-zero multisine force but must predict a full-amplitude
multisine position. The residual `q1_ms - simulate(u_ms) ≈ r_ms` is large and almost
independent of plant parameters. This uninformative residual dominates the MSE, masks the
parameter-sensitive gradient from trajectory dynamics, and drives the optimizer into bad
local minima. Observed: `ref_injection` stalls at loss `2.8e-3` vs `base` converging to
`3.2e-7`; recovered parameters off by up to +1083% for `cy`.

With force injection (`multisine`), `f_sim` is generated independently of the plant and
added as a direct input. The open-loop model receives the full multisine force and must
produce the matching oscillations at the correct frequency/amplitude — a parameter-sensitive
residual that gives informative gradients.

**Ruled out**: Continuing to use `ref_injection` for open-loop training. The
S-attenuation argument ("ref injection reaches plant via T≈1") is correct for
closed-loop identification on real hardware; it is irrelevant for the open-loop
simulator in `train_param_recovery.py`.

**Constrains**:
- Use `DATASET = 'multisine'` for parameter recovery training runs.
- `ref_injection` data can still be used for: (a) closed-loop identification frameworks,
  (b) training with the `r_ms` component subtracted from `q1` targets (see D-048 options
  in `docs/ref-injection-openloop-incompatibility.md`).
- T7 and T8 provide genuine observability benefit (all 13 parameters excited simultaneously)
  but only when the multisine injection method is compatible with the training objective.
  They should be included in the `multisine` dataset runs.

---

### [D-050] Resonance/bandwidth-weighted broadband multisine as active experiment design strategy
**Date**: 2026-05-10 (updated same day)
**What**: Active multisine design strategy: all odd harmonics from f_low to f_high, with
amplitude biased toward resonances and system bandwidth. Replaces FIM-driven scan-score
band selection. Declared HEURISTIC — variance motivation is PEM/noise-based, not
BPTT-specific; declared as such to supervisors.

**Design**:
- All odd harmonics from f_low to f_high (full band coverage)
- Amplitude concentrated toward resonances and system bandwidth (Lecture 9 slide 13, 27)
- Schroeder phases: φ_k = -k(k-1)π/F (Schroeder 1970, IEEE Trans. IT)
- Odd harmonics only: enables nonlinearity detection via even output lines (P&S Ch.4 §4.3.2)
- Force injection after controller (D-048): keeps excitation in u_recorded for BPTT replay
- PE condition: F ≥ 7 positive sinusoids (2F ≥ 14 = n_params; Lecture 6 slides 17–20,
  Lecture 9 slide 22: "PE(u) = 2 × harmonics")
- f_low, f_high from system physics (f_osc_min ≈ 4.9 Hz from eigenvalues; f_high ≈ 100 Hz)

**Why resonance-weighted over flat uniform**:
5SMB0 Lecture 9 slide 13 explicitly supports concentrating input power at resonances and
bandwidth. This is the lecture-backed middle ground: stronger motivation than flat uniform
(Ljung §13 §number unconfirmed for our claim), weaker than FIM-optimal but without
FIM's source gaps. Qualitatively compensates for |S| attenuation of force injection
inside the controller bandwidth without requiring the unjustifiable A_k ∝ 1/|S| formula.

**Why broadband over FIM-driven**:
FIM-driven requires ∂G/∂θ at each operating point and has unresolved source gaps for
deterministic BPTT (Gap G1). When NN augmentation is added, FIM-optimal for the 14
known params under-excites model-error frequencies. Broadband with resonance weighting
covers both needs without redesign. FIM-optimal deferred to G12.

**Constrains**:
- Drop scan-score band selection from `export_param_recovery_multisine.m`.
- Replace with all odd harmonics from f_low to f_high, resonance-weighted amplitudes.
- F ≥ 7 bins is the PE lower bound; more is better up to available trajectory length.
- Amplitude weighting shape must be declared as HEURISTIC in thesis and to supervisors.

---

### [D-051] Step 0 preanalysis uses simulation-based empirical Ŝ(jω), not analytical S(jω)
**Date**: 2026-05-10
**What**: The Step 0 survival profile is estimated empirically by injecting a flat broadband
probe into the closed-loop simulation and computing `Ŝ(jω) = FFT(u_total) / FFT(f_sim)`,
rather than computing S(jω) analytically from A_c, B_c, C_c, and the controller.

**Why**: In the current parametric model both methods give identical results. However,
when the model becomes incomplete (NN augmentation added) or moves to hardware, the
analytical S(jω) from the nominal model diverges from the true survival profile. The
simulation-based approach uses the actual closed-loop response at every stage, so the
same code path applies to:
- Current parametric simulation: Ŝ = S (equivalent)
- Augmented simulation: Ŝ reflects changed dynamics automatically
- Hardware: replace simulation run with real measurements — same formula

**Ruled out**: Purely analytical S(jω) from state-space matrices. Correct now but
requires explicit code change at every model update; simulation-based is forward-compatible
at no additional cost.

**Constrains**:
- Step 0 requires a short simulation run before Step 1 can proceed.
- Probe signal: flat broadband multisine (all harmonics, equal amplitude, force injection).
- f_low threshold from `|Ŝ|²` has no universal source — must be declared as engineering choice.

---

### [D-052] FRF pretest uses stage coordinates directly -- no input/output transform
**Date**: 2026-05-19
**What**: The frozen-Y MIMO FRF pretest uses raw stage coordinates throughout:
- Inputs: `[F1, F2, FY]` (physical actuator forces)
- Outputs: `[X1, X2, Y]` (physical position sensors)
No `output_to_modal` or `input_to_modal` transform is applied.

**Why**: Orthogonality of the input matrix comes entirely from the excitation design
(`f_vec = [1,1,0]`, `[1,-1,0]`, `[0,0,1]`), not from transforming the measured signals.
At each frequency line k the U_all columns are orthogonal in stage coordinates by
construction (the [1,1;1,-1] X-block is the Hadamard structure from Lecture 9; Y is
independent). The pretest purpose is frequency range selection -- resonance peak
locations are invariant to coordinate transforms. Stage coordinates are the simplest
valid choice.

**Ruled out**: Kamtin logical coordinates (P matrix transform) -- would enable a direct
oracle-test overlay against the analytical model, but adds scaling decisions with no
benefit for frequency range selection. Ad-hoc symmetric transform `(X1+/-X2)/2`,
`(F1+/-F2)/2` -- neither stage nor logical, has no clear benefit and mismatches kamtin
by constant factors anyway.

**Constrains**:
- FRF is 3x3 in stage coordinates. Plot axis labels are X1/X2/Y for both inputs and outputs.
- All 3 excitation modes (common X, diff X, Y) are retained -- Y is a physical DOF, not
  only a scheduling variable.
- A post-hoc coordinate transform would be needed to directly compare this FRF against
  kamtin's `StageCoordinatesSystem` (which is in logical coordinates).

---

### [D-053] State recovery diagnostic appended to gantry_interconnect_dynamic.py (not standalone)
**Date**: 2026-06-10
**What**: A `state_recovery_diagnostic()` function is added at the end of
`scripts/gantry/gantry_interconnect_dynamic.py` and called after `evaluate_and_save` in
both main paths. It compares encoder state estimates x_hat(k) on the validation set against
physical states reconstructed from measurements (q = inv(P^T) y, velocities via backward FD),
reporting per channel: R2_raw (x_hat[:, :6] read directly as normalized physical states),
R2_linmap (best OLS linear map x_true ~ x_hat @ W + b), and R2_raw_lag1 (against x_true(k-1)).

**Why**: The 2026-06-10 code review verified physics, normalization, wiring, and data loading
as correct, leaving two candidate explanations for poor theta/velocity recovery:
(a) basis rotation -- the dynamic-parallel ANN corrects all 8 derivative channels, so the
output-only loss does not pin states 3:6 to physical velocities; (b) information genuinely
absent (observability / training config). R2_linmap ~ 1 with low R2_raw proves (a);
low R2_linmap proves (b). R2_raw_lag1 > R2_raw exposes the separately-found hybrid encoder
one-sample misalignment (deepSI na_right=0: ypast ends at y[k-1] while the encoder
initializes x(k)).

**Ruled out**: Standalone script in `scripts/gantry/verification/` (preferred per the
self-contained-diagnostic rule) -- rejected by user because no trained checkpoint exists;
the diagnostic must piggyback on the next training run. Window construction follows the
deepSI hist convention exactly (ypast = y[k-na:k], na_right=0) so the diagnostic sees what
training saw.

**Constrains**: Diagnostic runs on the validation trajectory only; windows are subsampled
(~2000) to bound memory. "True" velocities are backward-FD reconstructions at fs=4000 Hz,
exact only for noise-free data (currently the case).

---

## D-054: Encoder initialization via reconstructability map (Hoekstra 2026)

**Date**: 2026-06-11

**Decision**: Replace detached `HybridGantryEncoder` with `linear_encoder_init`-based encoder
from Hoekstra 2026 ("Encoder initialisation methods in the model augmentation setting").

**Why**: The `HybridGantryEncoder` computes physical states analytically with `.detach()`,
freezing them. The FP model's positions/velocities don't exactly match the real system, and
the optimizer cannot correct this mismatch. The `linear_encoder_init` approach initializes
encoder weights from the baseline model's reconstructability map (Eq. 16-17) while keeping
all weights as trainable `nn.Parameter`. This gives a good starting point that the optimizer
can then refine.

**Implementation**:
- Linearize CT gantry model at Y_op=0 and discretize (ZOH at TS_NEW=1/4000)
  → `model_augmentation/systems/gantry_linearization.py`
- Normalize (Ad, Bd, Cd, Dd) with `normalize_linear_ss_matrices()` using training data stats
- Create `linear_encoder_init(A_bar, B_bar, C_bar, D_bar, nx=6, na=25, nb=25)`
- Wrap with `LinearInitEncoderWrapper` (physical encoder + zero-init ANN for augmented states)
- Inject with `na_right=1, nb_right=1` (encoder window includes y(k), required by
  reconstructability map)
- `na = nb = 4*NX_PHYS + 1 = 25` (Jan's rule of thumb)
- Observability rank verified = 6 (full), ZOH vs RK4 error < 1e-11

**Ruled out**:
- Data-based encoder init (SS_pre_encoder, Eq. 35): deferred, not ruled out. Will use if
  model-based struggles with LPV nonlinearity.
- Keeping HybridGantryEncoder: `.detach()` prevents learning of physical state corrections.

**Constrains**: Requires `na_right=1, nb_right=1` in SSE_Interconnect. Baseline simulation
states must exist at `data/gantry/baseline_simulations/multisine_LPV/baseline_states.npz` for
the normalization of the DT matrices.

---

### [D-055] D-017 convention fix migrated into linear_encoder_init_aug
**Date**: 2026-06-23
**What**: The normalization convention fix (D-017) is moved from `LinearInitEncoderWrapper`
(torch_nets.py) into `linear_encoder_init_aug` itself (pre_encoder.py). Six optional
keyword arguments are added: `u_mean, std_u, y0, ystd, x_mean, std_x`. The fix is
implicit: it is enabled if and only if all six are provided; omitting any one disables it
(backward-compatible, collapse property diag1 unaffected).

**Why**: `LinearInitEncoderWrapper` had a dead-code ANN bug (augmented states were not
wired into the optimizer). That bug was the reason `linear_encoder_init_aug` was created.
Putting the convention fix back in a wrapper would recreate the same structural problem.
Embedding it in the class directly keeps the encoder self-contained and eliminates the need
for the wrapper entirely for the augmented case.

**Implementation** (`model_augmentation/fit_systems/pre_encoder.py`):
- `__init__`: if fix_enabled, register three non-learnable buffers:
  - `u_off` (nu*(nb+1), 1): tile(u_mean/std_u, nb+1)
  - `y_off` (ny*(na+1), 1): tile(y0/ystd, na+1)
  - `x_off` (nx, 1): x_mean/std_x
- `forward`: if fix_enabled, add u_off/y_off to uhist_mod/yhist_mod before W^b/W^a;
  subtract x_off from x_b (physical states) after. x_a (augmented) untouched.
  ANN receives original pipeline-convention inputs.

**Verified by**: diag6 (5/6 checks pass; S1/S2/S3 confirm 28-285x NRMS improvement
at init; T1 failure is expected for exact linear system due to self-cancellation).

**Constrains**: Call sites of `linear_encoder_init_aug` that want the fix must pass
all 6 constants. `gantry_interconnect_dynamic.py` must be updated to use
`linear_encoder_init_aug` directly (replacing `linear_encoder_init` + `LinearInitEncoderWrapper`).

---

### [D-056] Narrowband multisine amplitude uses 5% of trajectory RMS, not 40%
**Date**: 2026-06-23
**What**: When `MULTISINE_BAND == 'narrowband'` (130–180 Hz), `force_cap_frac` is 0.05 instead of 0.40.
**Why**: The 40% heuristic was calibrated for broadband excitation where the multisine overlaps with trajectory frequency content (1–7 Hz or 1–200 Hz). For narrowband at 130–180 Hz, the trajectory has zero spectral content, so 40% of trajectory RMS forces is applied entirely in a band where it has no competition — causing the multisine to dominate the total force. The MSD resonance provides Q=10 amplification, so 5% (~10–30 N RMS) still yields 2–10 µm of delta_a, which is measurable. Using 40% produced 100–200 N RMS of narrowband force, far exceeding what is needed.
**Ruled out**: Absolute cap (Option B) — depends on knowing force levels per experiment in advance. Target delta_a SNR (Option C) — requires noise floor characterisation not yet done.
**Constrains**: If `force_cap_frac` is ever made a parameter, narrowband must remain at 5% unless delta_a SNR is verified to be sufficient at lower amplitudes.

---

### [D-057] Narrowband MIMO floor = force_cap_frac × max(traj_rms)
**Date**: 2026-06-23
**What**: After the per-channel 5% rule and inactive_frac, apply `amp_ch = max(amp_ch, force_cap_frac * max(traj_rms))` in narrowband mode only.
**Why**: The per-channel 5% rule undersizes weak channels when one channel dominates (e.g. T1 Y-only: X channels get 0.1 N; T3 X-only: Y channel gets 1.1 N). The floor referenced to the dominant channel's RMS keeps all channels proportionate to the experiment's overall intensity without arbitrary constants. Acknowledged: T1/T5 still give small amplitudes (~0.6–0.95 N) and negligible MSD excitation (~20–30 nm delta_a), accepted because those experiments cover scheduling range, not MSD identification.
**Ruled out**: Absolute floor (5 N) — arbitrary constant with no physical grounding. Per-channel skip based on symmetric-mode activity — excluded anti-symmetric excitation incorrectly.
**Constrains**: T3, T4, T7, T10 carry the MSD identification burden. T1, T5 contribute scheduling and coupling data only.

---

---

### [D-058] Telica real-data verification reads .log directly, no .mat conversion
**Date**: 2026-06-23
**What**: `telica_loader.py` reads Telica `iter*.log` files directly and returns
`(u, q1, fs)` matching `precompute._load_trajectory`'s contract. `run_telica_param_recovery.py`
monkey-patches `precompute._load_trajectory` and `compute_rmse_baseline_metrics` before
calling `train_param_recovery.train()` without modifying either original file.
**Why**: Converting to intermediate `.mat` files adds a redundant step with no benefit — Python
can read the `.log` files directly. Monkey-patching keeps both original scripts untouched.
**Ruled out**: (1) Intermediate `.mat` conversion — unnecessary overhead. (2) Adding a Telica
entry to `_DATASETS` in `train_param_recovery.py` — modifies a shared file for a single use case.
**Constrains**: The loader must always return `(u, q1, fs)` with shapes `(1, T, 3)`, `(T, 3)`,
`float`. If `precompute._load_trajectory` signature changes, `telica_loader.py` must match it.

### [D-059] Telica force input is MF30 kept in raw ci units; I_max unknown without Telica.mat
**Date**: 2026-06-23 (corrected 2026-06-23)
**What**: `u = MF30 × 1.0` (raw ci) is used as the plant input. `_CI_TO_AMP = 1.0` — no
conversion to Amperes. MF30 is the total current command (feedback + feedforward + cogging,
after KF60 saturation).
**Why**: `I_max = M82/100` (AccurET §23.2) is stored in `Telica.mat`, which is not in the repo.
Without I_max the formula `I[A] = MF30 × I_max/32768` cannot be evaluated. The factor
`1/481.882` (earlier logged) comes from the **old 5-column** Telica log format (commented out
at MATLAB line 438); the active MATLAB code does **not** convert MF30 at all. The ci scale
folds uniformly into all recovered mass/stiffness/damping parameters; NRMSE is position-based
and is unaffected by the force-unit scale.
**Ruled out**: (1) `1/481.882` (old 5-column format, not applicable to current logs).
(2) Estimating I_max from drive specs (AccurET 400 15/40A → I_max≈40A gives ~49 A from raw
MF30 values, which is physically impossible, confirming the estimate is wrong without Telica.mat).
(3) Using `MF30 - MF230` as feedforward-only — valid only when cogging is off and saturation
confirmed absent, which cannot be verified from the log files alone.
**Constrains**: Recovered parameter values are in `[unit] × I_max/32768` — not in SI.
Physical values recoverable once `Telica.mat` provides I_max.

### [D-060] Structural validation criterion: NRMSE with 15%/30% thresholds from SEM literature
**Date**: 2026-06-23
**What**: Post-training evaluation computes NRMSE = RMSE / std(q1_measured) × 100% per channel.
Decision rule: < 15% = structure compatible; > 30% = structural mismatch or force-signal problem;
15-30% = ambiguous, inspect trajectory plot.
**Why**: NRMSE is the scale-independent metric recommended for simulation error method (SEM)
structural validation. Thresholds from Schoukens & Ljung (2011) and Paduart et al. (2018).
Absolute RMSE [m] alone is not interpretable without knowing trajectory amplitude.
**Ruled out**: Absolute RMSE threshold — depends on motion amplitude which varies per operating point.
**Constrains**: NRMSE is computed against the full-trajectory simulation, not per-segment training loss.

### [D-061] Telica native sampling rate = 10 kHz from AccurET PLTI; timestamps discarded
**SUPERSEDED by D-073 (2026-07-03)**: the iter logs are 20 kHz native; the controller-notch
fingerprint contradicts the 1/(2*PLTI) formula for these files. Kept for the still-valid
part: raw .log timestamps are host-side reception artefacts and must never be used.
**Date**: 2026-06-23
**What**: `_NATIVE_FS = 10_000.0` Hz (fixed constant). Raw `.log` timestamps are discarded.
Synthetic time axis is built from sample index: `t = arange(N) / _NATIVE_FS`, exactly matching
MATLAB `runFDILCAllHostSwLog.m` line 92. Upsampling to `_FS_TARGET = 20_000.0` Hz is done by
linear interpolation, matching MATLAB `interp1` default.
**Why**: AccurET manual §1 (page 18): position-loop PLTI = 50 µs → FsHz = 1/(2×PLTI) = 10 kHz.
Raw `.log` timestamps are host-side reception times (non-uniform artefact — burst at ~66 kHz
during the first 125 ms, then ~200 Hz). Inferring fs from these timestamps gives ~411 Hz,
which is wrong by a factor of 24. MATLAB explicitly discards them (line 92).
**Ruled out**: Inferring fs from median timestamp difference — produces wildly wrong rate due
to non-uniform host logging. Reading `_NATIVE_FS` from a header field — no such field exists.
**Constrains**: Any code that processes Telica `.log` files must use `_NATIVE_FS = 10_000` Hz
and build synthetic timestamps from sample index. Raw timestamps must never be used for resampling.

### [D-062] Motion detection threshold 1e-9 µm; ILC data has no meaningful standstill to trim
**Date**: 2026-06-23
**What**: `_find_motion_start` uses threshold `> 1e-9` (post µm→m conversion, so effectively
1e-15 m). In the ILC experiment, M0 has quantization noise ±0.012 µm from sample 1, triggering
detection at sample index 1. Pre-motion samples to keep: `max(0, 1 - 500) = 0` → `trim_start = 0`
→ all data is kept (T = 32 856 samples at 20 kHz = 1.64 s).
**Why**: MATLAB `runFDILCAllHostSwLog.m` line 100 uses threshold `> 0` (any deviation from M0[0]).
Python `> 1e-9` is equivalent: both trigger at sample 1 due to quantization noise. MATLAB then
computes `startIdx = 1 - 500 - 1 = -499` (1-indexed); `(1:-499)=[]` is an empty range → nothing
deleted. Python replicates: `trim_start = max(0, 1 - 500) = 0`. The ILC experiment parks the
gantry at a fixed absolute setpoint; the relative M0 signal is near-zero throughout with only
quantization noise — there is no true standstill period to discard.
**Ruled out**: Threshold 0.5 µm (old version) — triggers at sample ~8218 (mid-ramp), discarding
valid ILC data. Threshold 0 (exact MATLAB match) — identical outcome in practice because float
quantization noise is always > 0.
**Constrains**: If a different Telica log has a genuine standstill (non-ILC trajectory), the
threshold still works: quantization noise will again trigger at sample ~1, and the 500-sample
pre-motion window will be preserved. The logic is therefore general.

### [D-017] Convention fix in LinearInitEncoderWrapper

**Date**: 2026-06-12

**Decision**: Add normalization convention conversion inside `LinearInitEncoderWrapper` to
bridge the mismatch between `normalize_linear_ss_matrices` (pure scaling) and the pipeline's
mean-subtracted data.

**Why**: `normalize_linear_ss_matrices` produces (Ad_bar, Bd_bar, Cd_bar, Dd_bar) in the
pure-scaled convention: x_scaled = x/std_x, u_scaled = u/std_u, y_scaled = y/std_y. The
Wb_psi_y and Wb_psi_u matrices are derived from these normalized matrices. But the training
pipeline normalizes with mean subtraction: u_norm = (u - u_mean)/std_u, y_norm = (y - y0)/ystd.
Diagnostic results showed this mismatch caused up to 97% velocity NRMS (dq3), while pure-scaled
reconstruction achieved ~10% (limited by LTI model accuracy and O_n conditioning at 818).

**Implementation**: `LinearInitEncoderWrapper` now accepts optional normalization constants
(u_mean, std_u, y0, ystd, x_mean, std_x). In `forward()`:
1. Add u_mean/std_u and y0/ystd to input (undo mean subtraction → pure-scaled)
2. Wb_psi_y @ y_scaled + Wb_psi_u @ u_scaled (reconstruction in pure-scaled space)
3. Subtract x_mean/std_x from output (pure-scaled → pipeline convention)
Constants are stored as registered buffers (no gradients, move with `.to(device)`).
ANN branch still receives original pipeline-convention data.

**Ruled out**: Adding bias correction to encoder output (CHECK 7 in diagnostic showed this
doesn't work because it assumes perfect pure-scaled reconstruction, which doesn't hold due
to LTI model error). Modifying `normalize_linear_ss_matrices` itself (would break other users).

**Constrains**: All call sites of `LinearInitEncoderWrapper` must pass the 6 normalization
constants. Old call sites (without constants) still work -- convention fix is skipped when
constants are None (backward compatible).

---

### [D-063] Epoch-0 diagnostic thresholds for augmented encoder (diag8)

**Date**: 2026-06-23

**Decision**: `diag8_aug_encoder_init.py` uses absolute NRMS thresholds for physical channels,
and reports augmented channels (delta_a, vdelta_a) without a pass/fail check.

Revised checks:
- C1: all physical NRMS < 1.0 (encoder in-signal range)
- C2: all velocity NRMS < 0.5 (W^b gives reasonable velocity estimates)
- C3: output is finite (no NaN/Inf)
- C4: all position NRMS < 0.2 (position tracking with ANN perturbation)

**Why**: The original checks compared against `1.1 * max(analytical)` and `1.5 * analytical_pos`.
The analytical P_inv baseline is kinematically exact -- positions are computed directly from y
via P_inv, giving NRMS near machine zero. Any absolute threshold relative to that value is
trivially violated by the ANN random-weight perturbation on W^b. Specifically, q3 analytical
NRMS = 0.0 exactly, making `1.5 * 0.0 = 0` an impossible pass criterion.

For the augmented channels (delta_a, vdelta_a): W^a is randomly initialized. delta_a signal
std = 84 µm. Any random output will give NRMS >> 1. Checking NRMS < 1.0 at epoch 0 is
testing a property that only emerges after training, not a property of initialization.

**Ruled out**: Relative thresholds vs analytical; augmented-channel NRMS checks at epoch 0.

**Constrains**: When comparing across training iterations, delta_a NRMS should decrease
below 1.0. If it does not after training, it indicates W^a failed to learn the MSD state.
The diag8 results (.npz file) can serve as the epoch-0 baseline for this comparison.

---

### [D-064] Encoder history na_nb = nxd*2+1 (Jan's standard formula)
**Date**: 2026-06-23
**What**: `na_nb` in `DEFAULT_HP` set to `(NX_PHYS + NX_ANN) * 2 + 1 = 17` samples (4.25 ms at 4 kHz), replacing the previous time-based `NANB_SECONDS = 0.025` (100 samples, 25 ms).
**Why**: Jan's reference implementations (`msd_ndof_interconnect_dynamic.py`, `msd_ndof_interconnect_fit.py`) both use `na = nb = nxd * 2 + 1` as the principled minimum. The factor of 2 provides a margin over the observability lower bound (nxd outputs needed to reconstruct nxd states). Using a physically-motivated time window was longer than necessary and inconsistent with Jan's pipeline.
**Ruled out**: `NANB_SECONDS = 0.025 s` (100 samples) — not principled; more than 6× Jan's formula without justification. Longer windows are not needed because W^b initialization already gives a good physical state estimate from short history.
**Constrains**: If `NX_ANN` changes in `DEFAULT_HP`, `na_nb` updates automatically via the formula. The Optuna search range for `na_nb` should also be anchored around this formula, not a fixed sample count.

---

### [D-065] Output augmentation: y = Cd@x_phys + C_aug@x_aug with trainable C_aug
**Date**: 2026-06-25
**What**: Changed output equation from `y = Cd@x_phys` to `y = Cd@x_phys + C_aug@x_aug`. Replaced `Linear_Output_Block(C=Cd_norm)` with `Parameterized_Linear_Output_Block(C=[Cd_norm|C_aug_init], flag_loss_reg=False)` and changed the state selection from `PHY_IX` to `np.arange(nxd)` (all states).
**Why**: Two constraints blocked training. Constraint 1: gantry has 2 DT poles exactly at |z|=1 (K[q1]=K[q3]=0, rigid-body integrators). ANN routed to any physical state row amplifies ~400x over nf=400 BPTT rollout, producing 800-1634x blowup in 1 gradient step (diag13). Constraint 2: with ANN routed to x_aug only and y=Cd@x_phys, the ANN output is unobservable -- (A_aug, C_aug=0) forms an unobservable pair, ANN gradient is identically zero (diag11 T1). Output augmentation resolves both: the gradient path loss->y->C_aug->x_aug->ANN never passes through A_phys, so Constraint 1 is bypassed. C_aug nonzero gives ANN a gradient path, resolving Constraint 2. Verified by diag15: T1 ANN grad = 3.5e-4 (vs 0 before), T2 val ratio = 1.03x at nf=400 (vs 800-1634x before).
**Why Jan's approach (ANN->all states) does not apply**: Jan's MSD has min(1-|z|) = 4.4e-3 (all springs nonzero), amplification = 4.4x at nf=400. Gantry min(1-|z|) = 0 exactly, amplification = 400x. Jan's default is architecturally safe for MSD and architecturally unsafe for the gantry (diag14).
**C_aug initialization**: `C_aug_init[2,0] = 1e-2` (Y channel receives delta_a weakly). Absorber is coupled to Y axis. Scale 1e-2 keeps the init ANN contribution sub-percent of normalized output. C_aug is trainable (`nn.Parameter` via `Parameterized_Linear_Output_Block`) so it grows during training.
**Ruled out**: (1) ANN->velocity rows [3,4,5]+x_aug: velocities also near-unit-circle; T_vel test showed 836x blowup (diag13). (2) Fixed C_aug (register_buffer): ANN signal stays at 3.5e-4 permanently; C_aug must be trainable. (3) Gradient clipping: clip was inactive at nf=400 (grad_norm 0.26 < max_norm 1.0), so clipping cannot prevent the eigenvalue-amplification blowup (diag13 T_clip: 1634x with clip).
**Constrains**: The 5-step stability test (diag15 T3) showed +14% val degradation when training on 1 trajectory. This is encoder overfitting to a single trajectory, not architectural instability -- the ANN gradient (3.5e-4) is 10000x smaller than the encoder gradient (3.67). Full training on all 8 trajectories is required to assess real convergence. Monitor C_aug magnitude during training: if it stays near 1e-2 after many epochs, the ANN/encoder may not be learning the absorber dynamics.

---

### [D-078] Noise-floor acceptance criterion for the augmentation benchmark (Jan's SNR method, pinned to the baseline residual)
**Date**: 2026-07-05
**What**: Define "good enough" for the augmentation on the multisine pipeline via Jan's output-noise convention (`msd_ndof_interconnect_dynamic.py`: `sigma_n = rms(y) * 10^(-SNR/20)`, noise added to `y`, floor = `sigma_n` plotted as a horizontal line the val RMS descends to). Added measurement noise per output channel; success = augmented val sim-RMS reaches the noise floor `sigma_n`. Chosen level: **SNR = 50 dB** primary, sweep **55 and 60 dB** to locate the plateau. Signal levels backed out of run 68676's baseline table (`std(y) = RMS_error / NRMS`): X1 0.060 m, X2 0.065 m, Y 0.230 m. Resulting per-channel floor at 50 dB: `sigma_n` = X1 1.9e-4, X2 2.0e-4, Y 7.3e-4 m; aggregate val sim-RMS floor = **4.5e-4 m** (aggregate confirmed to be `sqrt(mean of per-channel MS)`, matches the printed 5.175e-4). Runs are **JOINT_ESTIMATION off** and use the **linear** augmentation (D-071), so parameter fitting cannot soak up the residual and a memoryless shortcut cannot fake the absorber's dynamic contribution. Noise is injected in **Python at the output** (reproducible, sweepable, exactly measurement noise), NOT in the MATLAB generator.
**Why**: In noiseless simulation there is no acceptance line: error can crawl toward 0 indefinitely (Jan: numerically sensitive, slow), and any value hit has a smaller one below it, so there is no pass/fail. Random noise cannot be predicted by any model and does not average out of the error, and neither does uncaptured deterministic content, so the error bottoms at `sqrt(unmodeled_deterministic^2 + noise^2)`: reaching `sigma_n` certifies all learnable signal (incl. the absorber) was captured to below the noise; plateauing above it quantifies the uncaptured content (the D-068 closed-Y-row routing ceiling) in absolute NRMS. The SNR is pinned to the measured baseline residual (NRMS ~0.0031-0.0037 val -> crossover 49.4 dB) so the floor sits BELOW the unmodeled content: at SNR 50 the baseline lands at 1.53x the floor (cannot reach without improving), the current augmentation at 1.19x (nearly there), giving a clean, falsifiable, achievable separation. The noise does not make learning easier; it converts an open-ended optimization into a bounded one with a known optimum, which is the entire value.
**Ruled out**: (1) Oracle-model floor (baseline-vs-oracle NRMS as the target): numerically valid but model-dependent, simulation-only, and not defensible (no oracle exists on hardware) - user rejected it explicitly; a threshold must be model-free and data-derived (lessons.md). (2) Jan's low SNRs 20/30 (floors 10%/3.2%): far ABOVE our baseline residual (~0.3-0.4%), so the untrained baseline already sits on the floor - reaching it is trivial and proves nothing. His grid is calibrated to his deliberately under-modeled 2-DOF-approx-3-DOF baseline; our near-perfect FP baseline needs SNR ~50. (3) 40 dB (Jan's loose "40 dB ofzoiets"): floor NRMS 0.01, still above the val residual, uninformative for our system; it is at best a top anchor. (4) A dedicated SNR helper function: the level is a two-line inline op (per-channel `sigma_n`), wrapping it is scaffolding (lessons.md: no operational scaffolding in the experiment script). (5) Estimating `sigma_n` from the data via the standard nonparametric methods (period variance, non-excited multisine lines; Pintelon & Schoukens): correct for REAL measured data where noise exists and is unknown, but the current phase is a NOISELESS simulation where there is nothing to estimate - those estimators belong to the future real-gantry phase, and the simulator's role there is to validate the estimator (inject known `sigma_n`, repeated periods, confirm recovery). (6) Injecting noise in the MATLAB generator: process/closed-loop noise is not what the floor criterion needs (open-loop pipeline) and baking one realization into the dataset loses reproducibility/sweepability.
**Constrains**: Output-floor is necessary but NOT sufficient for the state-interpretability claim (the augmented states could carry the absorber in a rotated basis, or a nonlinear correction could fake part of it), so the full acceptance test stays TWO-AXIS: output val sim-RMS reaching `sigma_n` AND augmented-state R2_linmap vs the oracle absorber (>~0.9), the latter measured NOISELESS (noise only lowers the achievable R2 ceiling). The linear augmentation couples the two axes on Y (a memoryless linear map cannot add the absorber's pole pair), so a run that reaches the Y floor should also raise R2_linmap; if it reaches the floor with R2 still ~0, the routing (Option A/B) is binding. Per-channel `sigma_n` is mandatory (X1/X2/Y amplitudes differ ~4x, MEET-05); a single global SNR would give three different effective SNRs. Numbers derive from run 68676, which is a JE-ON pilot: the baseline table is JE-independent (nominal params) so the floor is solid, but the augmented clean error (2.94e-4, the 1.19x) must be re-confirmed on a JE-OFF linear run before claiming the separation. Margin note: at SNR 50 the floor is only ~1.15x below the baseline content (tight); SNR 55 (aggregate floor ~2.5e-4 m) gives cleaner headroom and is the safer primary if the 50 dB separation proves borderline in practice.

### [D-077] Joint estimation v2: all 14 raw physical parameters trainable (train raw, trust combinations)
**Date**: 2026-07-05
**What**: `Parameterized_Gantry_State_Block` extended from the 5 damping/stiffness sums (D-076 v1) to ALL 14 raw physical scalars, mirroring `lpv_lfr_baseline/train_param_recovery.py` exactly. `PARAM_NAMES = [kb1, kb2, cg1, cg2, cy, cb1, cb2, mh, m1, m2, mb, Jb, Jh, d]`, each log-reparameterized (D-035) and detuned individually; only `Lb` stays frozen (it defines the coordinate frame via P, not the M(Y) rational structure). Because masses are now trainable, `nonlinear_function()` rebuilds the ENTIRE M(Y) structure per timestep from the parameters: `gantry_ss.build_poly_constants` -> alpha/beta/gamma/N0/N1/N2, `d0 = mh(alpha*gamma - beta^2)`, M1/M2 from mh, K/C from the stiffness/damping (using only the identifiable sums kb1+kb2, cb1+cb2), then `build_G_matrix_entries` -> A_combined. The parent `_mats()` hook is widened from `(K, C, A_combined)` to also carry `(mh, alpha, beta, gamma_, N0, N1, N2)`, and `deriv()`'s LPV branch reads all of them from the hook instead of as buffers (behavior-neutral for the fixed block, A0-guarded). Reporting (`identifiable_combinations()`, `param_table()`) exposes only the 10 data-identifiable quantities [kb_sum, cg1, cg2, cy, cb_sum, mh, m_total, m_diff, J_eff, d]; `m_diff = m1 - m2` is a SIGNED derived readout of the individually-logged (positive) m1, m2 — never itself a parameter.
**Why**: v1 froze masses to keep M(Y) constant and dodge the invertibility question. The Task 3.1 proof (D-077 companion) shows M(Y) is positive-definite for ALL Y provided every mass/inertia/geometry scalar > 0, which the log-parameterization guarantees by construction — so all 14 can be trained safely. Training the raw params does not "mess up" training even though only combinations are trusted: the non-identifiable splits (kb1 vs kb2, cb1 vs cb2, mass flat direction) are FLAT (zero data-gradient) and rest at their `param_loss` anchor; only the identifiable combinations receive gradient signal. "Train raw, trust combinations" is exactly the train_param_recovery design; the combination view is a reporting choice applied after training, not a parameterization constraint.
**Ruled out**: parameterizing the combinations directly (m_diff is signed — cannot be logged; unnecessary because it is derived from logged m1, m2); keeping a 5-vs-14 selector flag (speculative flexibility against minimal — freezing a subset can be a future flag if ever needed); a 5-param and 14-param class coexisting (v1 superseded, its results preserved in D-076); mutating parent buffers in the child to inject trainable matrices (breaks autograd/state_dict — the `_mats()` hook is the clean seam); rebuilding once per forward instead of per timestep (needs a forward-boundary hook Jan's step-by-step block calling does not cleanly provide; the redundancy is only the ~10% overhead and A1/gate measure it).
**Constrains**: Runtime ~+10% over v1 (est. ~+20% over the fixed block; the M-gate/D-timing measure the exact figure before any long run). Gate updated: A1 now also validates the full nominal M(Y) rebuild; B covers all 14 log_params; NEW check M samples the positive parameter orthant and asserts `M(Y) @ N(Y)/d(Y) = I` (inverse-consistency, off-nominal transcription guard) AND `eig(M(Y)) > 0` (PD, proof realization). Check D judges recovery on the 10 identifiable combinations: the well-conditioned subset {kb_sum, cg1, cg2, cy, cb_sum} is gated <= 0.5 (regression vs v1), mass combos are reported and only guarded against divergence (their identifiability from short data is the open run-design question, not a correctness gate). Same scientific-scope caveat as D-076: on this benchmark FP params are true by construction, so v2 JE is machinery validation + bias-demonstration, not a fix for the absorber output-reachability issue.
**Gate results (2026-07-05)**: Gate 1 PASS — A0 and A1 bitwise 0.0 (the full nominal M(Y) rebuild reproduces the fixed block exactly); B PASS on all 14 params under the gradcheck-style tolerance `|auto-fd| <= atol(1e-9) + rtol(1e-5)|fd|` (pure-relative would fail only on Jh, gradient ~2.7e-7, where autograd and central-difference agree to 4 sig figs — a roundoff floor, not an error; A1 + the M-check independently verify the Jb/Jh path); NEW check M PASS — `max|M·N/d - I| = 6.66e-16`, `min eig(M) = 2.97 > 0` over 200 positive-orthant samples x 7 Y (inverse-consistency + PD verified off-nominal). Gate 2 PASS — C loss integration rel 1.16e-9; D recovery on the state-readout config (C=I, flag_loss_reg=False) recovered ALL 10 identifiable combinations, not just the gated v1 subset: kb_sum 0.008, cg1 0.141, cg2 0.052, cy 0.019, cb_sum 0.033 (gate <= 0.5), and the reported masses mh 0.001, m_total 0.000, m_diff 0.002 (detuned +1.59 -> learned -0.4961, truth -0.5, signed), J_eff 0.028, d 0.000 gap-ratio. **Measured overhead**: parameterized-block forward +15.9% vs the fixed block (v1 was +11.7%, so ~+4% forward vs v1); fwd+bwd 1.49 s/batch vs v1's 1.09 s (+37% on the training step in isolation, because backprop through the per-timestep mass rebuild is costlier than forward alone — smaller net effect in the real pipeline where the ANN/encoder/longer windows dilute the physics-block share). Script edits (14-nominal `build_model` derivation from `gantry_ss`, `param_table()` report) validated via a no-train `build_model` rehearsal: params_init exact, combination table correct. Flag-off anchor (JE=False initial validation sim-RMS 6.4948e-4) is guaranteed unchanged by A0 (fixed-block deriv bitwise identical) and by the JE=False path not touching any new code; not re-run. Artifacts: `simulations/gantry_subnet/diagnostics/joint_estimation/` (gate1/gate2 JSON, gate_v2*.log).

### [D-076] Joint estimation in the multisine pipeline: Parameterized_Gantry_State_Block + generic param_loss trainer
**Date**: 2026-07-04
**What**: Three additions enabling joint estimation of physical parameters in `gantry_interconnect_dynamic.py`:
(1) `Parameterized_Gantry_State_Block` in `model_augmentation/fit_systems/blocks.py`, placed directly below `Gantry_State_Block` (mirroring Jan's fixed/parameterized adjacency, marked `@added`). Subclass with trainable vector `[kb_sum, cg1, cg2, cy, cb_sum]` stored as `log_params = nn.Parameter(zeros)` meaning log(theta/params_init) (D-035 positivity, MEET-02 centering: all params start at 1 in normalized space); regularization `param_loss()` with `Lambda = RMSE_baseline / params_init` toward `params_init` in physical space (D-034); `params_init` constructor override exists for detuned recovery tests. Per timestep the block rebuilds K and C (`torch.stack` construction, pattern copied from `lpv_lfr_baseline/blocks/lfr_param_block.py`, NOT imported) and `A_combined` via `gantry_ss.build_G_matrix_entries` — the functions kept autograd-safe for exactly this call. The parent `Gantry_State_Block` gains a ~3-line `_mats()` hook returning `(K_mat, C_mat, A_combined)` read by `deriv()`, so the child overrides matrices without duplicating the deriv kernel. New child buffers: `d0`, `M1`, `M2`.
(2) `SSE_Interconnect_ParamLoss` in `model_augmentation/fit_systems/interconnect.py` (`@added`): delegating `loss()` = `super().loss(...) + sum(m.param_loss())` over blocks exposing `param_loss` (hasattr sweep). Reimplementation of the D-032 idea from `lpv_lfr_baseline/blocks/lfr_fit_system.py`; deliberately NOT imported from there (no cross-pipeline dependency — user decision 2026-07-04). Used unconditionally in the script: exact no-op when no block exposes `param_loss`. Documented caveat: would double-count Jan's `Parameterized_Linear_*` blocks if ever combined (never used in this pipeline).
(3) `JOINT_ESTIMATION` flag in `gantry_interconnect_dynamic.py` gating ONLY the block class; `PARAM_RMSE_BASELINE = 0.01` constant (HEURISTIC: measured initial sqrt-loss of jobs 68675/68676); flag-guarded learned-vs-nominal parameter printout plus `params_init`/`params_learned` fields in the results npz; clear error when `RESUME_CHECKPOINT` points at a pre-JE checkpoint while the flag is on.
**Why**: Joint estimation machinery is the prerequisite for the gray-box absorber path (Option A applies the same log_params/param_loss pattern to a 3-scalar absorber block) and for real hardware where nominal parameters are uncertain. v1 trains damping+stiffness only: none of these enter M(Y), so every M(Y)-derived parent buffer (N0/N1/N2, Horner d(Y) constants, M0inv, Bw, Bu) stays constant and valid, and the Task 3.1 M(Y)-invertibility proof is untouched. Sums (kb_sum, cb_sum) are parameterized directly because only the sums are identifiable (flat-ridge analysis in lfr_param_block).
**Ruled out**: importing `LFRFitSystem` from `lpv_lfr_baseline` (wrong dependency direction); a new module or experiment script inside `model_augmentation/` (experiments live in `scripts/`, blocks belong beside their fixed siblings); duplicating `deriv` in the child (~35-line maintenance hazard, replaced by the `_mats()` hook); hand-assembled Ax/Bw/Bu in the child (second copy of the G-matrix math, replaced by reusing `build_G_matrix_entries`); in-script Lambda auto-calibration (`set_RMSE_baseline` machinery is operational scaffolding — a measured constant with provenance suffices; log-space centering makes Lambda scale non-critical); flag-gating the trainer class (delegation makes the subclass a provable no-op when unused).
**Constrains**: Verification gates precede any full run; diagnostic results go to `simulations/gantry_subnet/diagnostics/joint_estimation/`. Gate 1 (after block): A0 = refactored fixed block reproduces a reference batch captured BEFORE the refactor; A1 = parameterized block at `log_params=0` matches the fixed block (~1e-6 float32); B = finite-difference vs autograd gradients for all 5 params (float64). Gate 2 (after trainer): C = one `loss()` call equals MSE + param_loss on a minimal no-ANN interconnect; D = mini recovery on self-generated absorber-free data (fixed block, nominal params, real multisine u), `params_init` detuned ±10%, pass = gap to nominal shrinks ≥50% for all 5 params; it/s from D doubles as the runtime-overhead measurement. Then the manual 1-epoch rehearsal (D-071 procedure) with the flag on, and the flag-off anchor check (initial validation sim-RMS must remain exactly 6.4948e-4). Scientific scope note: in THIS benchmark the FP parameters are true by construction, so JE of FP params serves machinery validation and bias-demonstration ablation only — it cannot address the absorber state-learning issue (output reachability unchanged) and is expected to bias parameters if trained on absorber-containing data. JE runs start from fresh checkpoints (old .pt files lack `log_params`).
**Gate results (2026-07-05)**: Gate 1 PASS (A0 and A1 bitwise 0.0; B max FD-vs-autograd rel err 2.5e-7). Gate 2 PASS with a redesigned check D: two recorded failed attempts showed that the ORIGINAL check-D configuration could not isolate the machinery — (attempt 1, random default encoder + sim-RMS validation) the K=0 horizon-mismatch checkpoint trap restored epoch 0 and all 15 epochs went into encoder learning; (attempt 2, Hoekstra-style encoder + windowed validation) the co-trained encoder absorbed the loss (down 1000x with parameters frozen at init; nominal and detuned params gave IDENTICAL initial loss 2.0921, proving the loss was encoder-driven). Final check D therefore isolates parameter learning: synthetic output = full state (C = I), exact parameter-free readout encoder, flag_loss_reg=False (check C separately proves the regularization path, exact to rel 4e-10). Result: gap ratios kb_sum 0.000, cg1 0.211, cg2 0.189, cy 0.001, cb_sum 0.055 (pass <= 0.5); parameterized-block forward overhead +11.7% (fwd+bwd 1.09 s/batch, nf=100, batch 128). **Run-design findings for real JE (input to Phase 3/4 and the Jan discussion)**: (1) with position-only outputs, short-window BPTT and a co-trained encoder, physical parameters are practically unidentifiable — the encoder compensates; (2) param_loss anchored to the init values actively pins parameters there once the MSE landscape flattens; (3) windowed validation is mandatory on this plant (sim-RMS checkpoint selection reproduces the documented horizon-mismatch trap even without an ANN). Artifacts: `simulations/gantry_subnet/diagnostics/joint_estimation/` (gate1/gate2 JSON, gate_run*.log including the failed attempts).

### [D-075] Telica train/validation/test wiring: supervisor's split, iter0+iter8, SEGMENT_LEN 650 confirmed
**Date**: 2026-07-04
**What**: `run_telica_param_recovery.py` switched from single-trajectory to the supervisor's
split (folders under `06 40 mm XL 80 mm YL/`, split by operating point): TRAIN = 11 OPs x
{iter0, iter8} = 22 trajectories; VALIDATION = 2 OPs x {iter0, iter8} = 4; TEST = 2 OPs x
{iter0, iter8, iterTEST} = 6, final evaluation only. IDs T1a..T11b / V1a..V2b / E1a..E2T
(a = iter0, b = iter8, T = iterTEST); `tr._traj_set_tag` is monkeypatched to '22traj' to
keep the checkpoint filename inside the Windows 260-char path limit. EPOCHS = 40,
VALIDATION_INTERVAL = 5, SEGMENT_LEN = 650 (re-picked consciously per D-073: 32.5 ms at
the true 20 kHz spans 6+ periods of the ~200 Hz servo band and 27 periods of the 845 Hz
notch resonance), NORM_MODE 'global', FULL_COVERAGE, no overlap. Final evaluation loops
open-loop AND closed-loop (`_post_eval` + `_post_eval_cl`) over a train sample (T1a/T1b),
all validation and all test trajectories.
**Why**: iterations within an OP share the same reference and differ only in feedforward;
iter0 (feedback-dominated, transient-rich) and iter8 (converged ILC, smooth) are the two
extreme input spectra; the 7 in between are near-duplicates (5x cost, little information).
Operating-point diversity, not iteration count, drives LPV identifiability. iter6_1.log
(redo artifact) excluded.
**Deviation from the stated plan**: the framework's built-in validation
(`_full_traj_eval`) is a FULL-TRAJECTORY OPEN-LOOP RMSE on the validation trajectories,
not the windowed training measure. It is controller-free and it is the metric we
ultimately care about for OL quality, so checkpoint selection and LR scheduling use it
as-is rather than adding a windowed-validation code path.
**Ruled out**: all ~110 iterations (redundant, ~5x runtime); iter0-only (single input
character); windowed validation implementation (extra code path in the training script
for marginal benefit).
**Constrains**: Runtime estimate revised: FULL_COVERAGE gives ~13 gradient steps/epoch at
batch 22 (not one batch of 294), so an epoch is ~30-40 s CPU; 40 epochs + 9 validation
passes = roughly 30-45 min training, plus ~30-60 min for the 12 OL + 12 CL final
evaluations. TEST trajectories are also evaluated open-loop by tr.train()'s own Step 4
at the very end (after best-checkpoint restore); they never influence training.

### [D-074] Closed-loop validation added to run_telica_param_recovery.py; training stays open loop
**Date**: 2026-07-03
**What**: Three additions, no change to the training path: (1) `telica_loader.load_telica_log_cl`
returns r [m], q1 [m], u_ff [N] ((MF30-MF230)*Kt) and logged i_fb [A] (MF230) on the same
grid/trim as the training loader. (2) New `telica_controller.py`: `TelicaFeedbackController`,
per-sample direct-form-II-transposed stepper for the LX1/LX2/LY controllers from
`dFeedbackControllersTelica_ba.mat` (exported from the supervisor's zpk file); self-test
verifies bit-exactness vs scipy lfilter and replays iter0 (corr 0.96-0.97 against logged
MF230; known amplitude offsets: time-domain LS 1.16/1.33/3.55, coherent-band 0.74/1.08/1.23
per D-073, attributed to an unmodeled rail decoupling transform). (3) `_post_eval_cl` +
`_run_closed_loop` appended to the runner: initial condition only, then full trajectory with
u = u_ff + Kt * K(r - y_model) stepped through rk4_step under no_grad; reports the same
RMS/NRMSE table as the open-loop eval plus a feedback-current plot with a built-in wiring
check (controller fed with measured error vs logged MF230). Called for baseline and best in
`__main__`, next to the open-loop evals.
**Why**: Supervisor decision: train open loop (windowed BPTT parallelizes; a controller in
the training loop forces one long sequential rollout), validate closed loop (the
control-relevant metric; a model is good if it behaves like the plant inside the same loop).
Timing convention: controller acts on the same-sample error, no computational delay; the
output has no feedthrough (y = Cy x), so the loop is well-posed.
**Ruled out**: CLOE training (controller inside the gradient path): runtime and complexity
not justified while open-loop windowed training suffices; everything learned here (loader,
controller module, conventions) is reusable if it becomes necessary. Closed-loop-only
validation: CL suppresses model error inside the bandwidth, so it must always be read next
to the open-loop numbers.
**Constrains**: CL numbers are only comparable between models evaluated with the same
controller file. Smoke test (untrained Kamtin-parameter block on iter0): the loop diverges
(~3 m RMS), which is the expected verdict for a plant-mismatched model under the real
controller, not a wiring bug (the replay check inside the same run is clean).
**Verified (diag_cl_correctness.py, 2026-07-03)**: (1) controller bit-exact vs MATLAB
filter(tfdata) on the original zpk (0.0 deviation; the 2.15e-5 deviation vs lsim is
MATLAB-internal tf-vs-ss conditioning of the unit-circle integrator poles); (2) perfect-model
test: same-x0 repeat exact, rebuilt-x0 NRMSE 0.006-0.03% with a gain-scaled (stable)
controller; (3) OL replay of the CL force reproduces CL positions exactly; (4) timing: the
logged MF230 lags K(logged M2) by a BROAD ~2.5 ms correlation maximum; a 2.5 ms in-loop delay
is physically impossible (the ~300 Hz-crossover loop would be unstable), so it is a
logging-path artifact and the zero-delay loop is correct; (5) all six controllers marginally
stable by design (integrator poles at |z|=1 within zpk rounding 6e-8), logged currents max
~6.1 A vs 27.9 A peak limit, so no saturation modeling needed; controller tuning consistent
across iterations and operating points (replay corr/scale identical on train and validation
positions).

### [D-073] Telica iter logs are 20 kHz native; loader upsampling removed (supersedes D-061)
**Date**: 2026-07-03
**What**: `telica_loader.py` now uses `fs_native = 1/SamplingTime = 20000 Hz` and performs
no resampling (a guard raises if native rate and pipeline rate ever diverge). The previous
chain (assume 10 kHz native, linearly upsample 2x to 20 kHz, D-061) stretched the time axis
by a factor 2: every Telica training and evaluation before this date fitted a plant with
2x slowed dynamics and is not comparable to later runs.
**Why**: Controller-notch fingerprint (`diag_controller_fingerprint.py`). The real Telica
controllers (dFeedbackControllersTelica.mat from the supervisor, 6x6 diagonal zpk at
Ts = 5e-5 s; axis order confirmed by supervisor: LX1, LX2, LY, RX1, RX2, RY) have notches
at fixed normalized frequencies of the 20 kHz DSP. In the iter0 empirical FRF
(M2 -> MF230, exact controller I/O pair since feedforward = 0), the X1 notch appears at
normalized frequency 0.10 of the LOG, exactly where LX1 has it under the 20 kHz
interpretation; under the 10 kHz interpretation it would appear at 0.20, where the data
shows nothing. Shape residuals: X1 3.2 dB (20 kHz) vs 5.5 dB (10 kHz); gain scales at
20 kHz: X1 0.74, X2 1.08, Y 1.23. Telica.mat SamplingFrequency description ("The number
of samples logged per second" = 20000) agrees. Figures:
`simulations/gantry_subnet/diagnostics/controller_fingerprint/`.
**Ruled out**: FsHz = 1/(2*TsSec) = 10 kHz from runFDILCAllHostSwLog.m line 30 (basis of
D-061): that formula belongs to a different logging configuration; the data itself
contradicts it for the iter*.log files.
**Constrains**: All pre-2026-07-03 Telica training results are invalidated for comparison.
`SEGMENT_LEN = 650` samples now means 32.5 ms instead of 65 ms: re-pick consciously before
the next training run. Diagnostic scripts hardcoding `_FS_NATIVE = 10_000`
(diag_cloe_signals.py) predate this finding. Remaining gain offsets (0.74/1.08/1.23) and
corr ~0.97 are consistent with a decoupling transform around the SISO controllers
(rail cross-coupling), acceptable for closed-loop validation. Related earlier decisions
D-069 (diagnostics) and the AeroPro finding (Telica.mat MachineType = "AeroProCoC").

### [D-072] Baseline comparison matrix: oracle-x0 and encoder-init baselines, revived oracle model sim, aligned averaging windows
**Date**: 2026-07-03
**What**: Four changes to `gantry_interconnect_dynamic.py` making every baseline/model comparison a well-posed cell of a matrix {baseline FP, augmented model} x {true x0, encoder init}: (1) `compute_baseline_fp_nrms` gains `x0_norm`, `start_ix`, `avg_from` parameters. (2) NEW encoder-init baseline: the baseline FP is seeded with the state estimated by the UNTRAINED `linear_encoder_init_aug` (Hoekstra reconstructability map W^b, built from the baseline's own linearization) from the first measured I/O window, simulating from sample k0=max(na,nb); computed for val and E1, printed with explicit labels distinguishing 'true x0' from 'encoder-init', stored in the results npz. Using the untrained map is deliberate: before training it is purely baseline-derived (no co-training with the augmented dynamics), so 'baseline + linear init' is well-defined; the trained encoder would not be. (3) The x_logical-initialised model simulation (augmented model from true x0) is revived: it was dead code because it checked `val_data.x` which `load_traj` never sets; it now seeds from `val_x_logical[0]` (always loaded). Its NRMS is now averaged over `[cheat_n:]` like the encoder-init model metric. (4) All baseline averaging windows aligned to k0 (`avg_from=k0` for oracle baselines, `start_ix=k0` for encoder-init), removing the ~0.2% asymmetry where the baseline was averaged over all N samples but the model over N-cheat_n.
**Why**: The pre-existing comparison was baseline(true x0, full window) vs model(encoder init, cheat_n window) — biased in the baseline's favor. Conservative for improvement claims, but on the K=0 axes initial-state errors do not decay (they integrate into position drift over the full horizon), so the bias understates the model-vs-baseline gap by an encoder-quality-dependent amount rather than a negligible one. The completed matrix separates model quality (true-x0 vs true-x0) from encoder contribution (model true-x0 vs model encoder-init) from realistic end-to-end performance (encoder-init vs encoder-init).
**Constrains**: Oracle baseline NRMS values shift slightly vs earlier logs (averaging now starts at k0). k0=max(na,nb) may differ from deepSI's cheat_n by one sample — negligible and documented here rather than plumbed through.

### [D-071] Linear parallel augmentation experiment (Jan's ECC config) + E1 generalization evaluation + smoke-test hook
**Date**: 2026-07-02
**What**: (1) `ANN_ACTIVATION` default switched from 'tanh' to 'linear' (Identity activation, Jan's `linear_parallel` ECC configuration) for the next training run. (2) `evaluate_and_save` extended with a test-set (E1) simulation: per-channel NRMS plus baseline-FP comparison on the unseen excitation; `compute_baseline_fp_nrms` generalized with `(data, x0_phys, label)` arguments so the baseline can be computed on E1 as well. (3) ~~`SMOKE_TEST=1` environment hook~~ — implemented, then REMOVED at user request (no operational scaffolding in the experiment script). The rehearsal is now a manual procedure: before a long submission, temporarily set epochs=1/nf=10, run the script end-to-end once (fresh + resume), revert, submit. Both rehearsals were executed on 2026-07-03 and passed (exit 0), validating the D-070 checkpoint save, the resume load, and the E1 evaluation.
**Why**: Run 68597 (tanh ANN, D-068 routing) improved aggregate val sim-RMS by ~43% vs the baseline FP but R2_linmap(delta_a) stayed at 0: the ANN compensates memorylessly from the instantaneous state instead of learning the absorber. A tanh static correction can imitate much of the absorber effect; a LINEAR static correction cannot add the missing pole pair, so any error reduction at the absorber resonance must flow through the two augmented states. The hidden absorber is itself LTI (Y-scheduling enters only via the fixed LPV FP block that propagates the corrections downstream), so a linear augmentation is the correct residual class, not a restriction. The E1 evaluation separates compensation from captured dynamics: a memoryless compensator tuned to training-excitation correlations degrades on unseen excitation, a learned oscillator transfers.
**Ruled out (for now, staged behind this run)**: (1) Gray-box absorber via `Parameterized_Linear_State_Block`: guarantees state meaning by construction but changes the model class and touches the D-068 routing question (reopening the Y row); escalation if the linear run keeps R2 near 0, after consulting Jan. (2) LPV-linear augmentation (correction linear in [x,u] with coefficients affine in Y): restores scheduling without adding dynamics; refinement if states learn but an error gap vs the tanh run remains. (3) Supervised x_aug loss on saved delta_a ground truth: simulation-only scaffold, does not transfer to real data.
**Constrains**: The outcome is diagnostic in both directions: R2 rises means the memoryless shortcut was the blocker; R2 stays near 0 means the closed Y injection row (D-068) is binding and the gray-box path becomes necessary rather than optional. Full capture of the MSD effect is structurally impossible while the Y row is closed (the absorber force reaches Y only via the Theta inertial coupling M0[1,2]=-mh*d, with collateral X1/X2 cost). The augmented loop has exactly zero gradient at init (zero-init final layer), so use more epochs than 30 for this run.

### [D-070] Weights-only training checkpoints via component state_dicts, saved before diagnostics
**Date**: 2026-07-02
**What**: Four changes to `train_model`/`train_model_with_diagnostics` in `gantry_interconnect_dynamic.py`: (1) Checkpoint save changed from `torch.save(fit_sys.state_dict(), ...)` to a dict of component state_dicts `{'hfn': ..., 'encoder': ..., 'optimizer': ...}`. `SSE_Interconnect` inherits from deepSI `System` (not `nn.Module`) and has no `state_dict`; only `fit_sys.hfn` (Interconnect) and `fit_sys.encoder` are torch modules, and together they hold all trainable parameters. (2) Resume path loads these component dicts into the model built by `build_model(hp)`; optimizer state included so Adam moments continue instead of restarting. (3) The `.pt` checkpoint is written immediately after `fit()` returns, BEFORE `aug_state_r2`, and the diagnostic is wrapped in try/except (NaN placeholders on failure), so no post-training step can lose the weights. (4) ~~`fit()` called with `verbose=1` under SLURM~~ — implemented, then REVERTED on 2026-07-03 at user request: the tqdm progress bar is how long cluster runs are monitored (ETA, it/s); log length is not a defect.
**Why**: SLURM job 68597 (30 epochs, 11 h, first successful D-068 run) crashed at the old save line with `AttributeError: 'SSE_Interconnect' object has no attribute 'state_dict'` after training completed. Because the save ran before `evaluate_and_save`, all artifacts (model save, results npz, plots, state recovery diagnostic) were lost; the best weights survived only in deepSI's internal `~/.deepSI/checkpoints/SSE_Interconnect_<code>_best.pth`. The resume path (`fit_sys.load_state_dict`) had the same bug and would have failed on first use.
**Ruled out**: `fit_sys.save_system()` whole-object pickle (`torch.save(self, file)`): works and is already used in `evaluate_and_save` for the final model, but deepSI's own docstring warns it is "quite unstable for long term storage or switching between versions", and it would replace the build-then-restore resume design rather than fit into it.
**Constrains**: Checkpoint `.pt` format is now the component dict; resume requires `build_model` with the same hp (already guaranteed: hp is read from the checkpoint `.npz` meta). No backward compatibility needed: the old save line never executed successfully, so no old-format checkpoints exist.

### [D-069] Controller reconstruction gain mismatch: three diagnostics before CLOE
**Date**: 2026-07-02
**What**: Three diagnostic scripts in `scripts/gantry/real-data-verification/` to resolve the
11-22x amplitude mismatch between the documented controller chain (M2[um] x 1024 cnt/um ->
Filter1 -> Filter2 -> x AmplifierGain 0.002075 A/DAC) and the logged feedback current MF230:
(1) `diag_log_rate.py`: resolves the log-rate ambiguity (10 kHz per D-061 vs 20 kHz per
Telica.mat SamplingFrequency description "The number of samples logged per second") by
locating known filter features (notches, integrator slope) in the empirical M2 -> MF230 FRF
on the normalized frequency axis. A filter feature at normalized frequency nu (designed at
20 kHz) appears at nu in the log FRF if the log is at 20 kHz, at 2*nu if decimated to 10 kHz.
No timestamps used (D-061 forbids them).
(2) `diag_frf_controller.py`: extracts the controller actually active during the FRF campaign
via K_eff = G^-1 (S^-1 - I) from frfPlant [cnt/dac] and frfSensitivity in Telica.mat
(THEORY: S = (I+GK)^-1, Skogestad & Postlethwaite 2005 Ch. 2). Compares K_eff against
Filter1*Filter2 per frequency. This is excitation-based and free of the closed-loop
correlation concern; a flat ratio identifies the missing per-channel gain and its value.
(3) `diag_iteretel_decode.py`: dumps the full column schema of iterETEL.log and iter0.log
(iter0 has 25 columns, only 13 identified so far; DatalogListVarMapping lists 25 ETEL
channels including X_HIGS_INPUT/X_HIGS_OUTPUT, X_FB_OUTPUT, X_ENC_POS, X_DAC) and runs
conditional analyses: HIGS input/output scatter (gain-mode slope) and raw-unit chain checks.
**Why**: Telica.mat is now fully read (MATLAB batch confirmed a single top-level variable);
no additional scale parameter exists in it. The mismatch is real (reproduced independently in
MATLAB with native filter()). The DatalogListVarMapping names HIGS blocks in the servo loop:
a HIGS (hybrid integrator-gain system) between error and filters acts approximately as a
per-channel constant gain, which fits every observation (corr near 1, constant ratio per
axis, different ratio per axis). A static-gain workaround was rejected (lessons.md): the
missing element must be identified, not approximated away.
**Ruled out**: LS scale from iter0 time-domain fit as the final answer: it conflates the
missing gain with rate misapplication effects (11x at 10 kHz native vs 22x at 20 kHz
upsampled shows the estimate is method-dependent). Asking Kamtin first: these diagnostics
use data already on disk and can fully resolve the question; ask only if they fail.
**Constrains**: CLOE implementation is gated on the missing gain being explained and
reproduced (reconstruction matching MF230 in iter0 within a few percent). Results go to
`simulations/gantry_subnet/diagnostics/`.

### [D-068] Route ANN only to states with spring stiffness (Jan's state_augment_specific_states)
**Date**: 2026-07-01
**What**: Change `build_model()` in `gantry_interconnect_dynamic.py` to route ANN corrections only to state rows with K > 0, instead of all `nxd` rows. For the gantry: `STIFF_IX = [1, 4, 6, 7]` (Theta position, Theta velocity, delta_a, vdelta_a). ANN output width changes from `nxd=8` to `len(STIFF_IX)=4`. Implementation uses `expansion_matrix(STIFF_IX, nxd)` per Jan's `state_augment_specific_states` API.
**Why**: Jan confirmed K=0 gantry axes (X, Y) cause ANN correction accumulation and suggested this fix. X (index 0,3) and Y (index 2,5) have K=0 — additive corrections accumulate without restoring force (O(N) drift). Theta (index 1,4) has kb1+kb2 stiffness; absorber states (index 6,7) have absorber spring. Routing only to K>0 states eliminates drift at the source. This is the physically motivated fix within Jan's framework.
**Ruled out**: (1) Full-state routing (D-067): K=0 axes accumulate drift — existing failure mode. (2) Velocity-only routing (D-066): velocity corrections still integrate to position drift under K=0. (3) Aug-only routing (D-065): C_aug gradient dead zone.
**Constrains**: ANN output dim becomes 4. Absorber-to-X/Y coupling is not directly captured (X/Y rows excluded). First test should use single-stage sim-RMS to confirm K=0 blowup is eliminated before revisiting curriculum design.

### [D-067] Revert to Jan's full-state routing + curriculum nf training
**Date**: 2026-07-01
**What**: Reverted `gantry_interconnect_dynamic.py` from Model B (velocity+aug rows) back to Jan's full-state routing (`connect_block_signals(ann_block, ["x","u"], ["xp"])`), matching `msd_ndof_interconnect_dynamic.py:91`. ANN `nw` reverted from 5 back to `nxd=8`. `VEL_AUG_IX` removed. `DIAG_INTERVAL` replaced by `NF_CURRICULUM`: a 6-stage curriculum schedule `(nf, epochs, validation_measure)` progressing 25→50→100→200→400 (windowed) → 400 (sim-RMS). `train_model` signature extended with optional `nf` and `validation_measure` overrides. `train_model_with_diagnostics` iterates `NF_CURRICULUM`, logging R2_linmap after each stage.
**Why**: Model B training failed: best checkpoint = epoch 0 (untrained) on all 20 epochs. Root cause is the K=0 + training/validation horizon mismatch: ANN learns velocity corrections that reduce nf=400 training loss but cause O(N_val/nf)=20× larger position drift on full 8000-sample validation. Excluding position rows from routing does not fix this — velocity corrections still integrate to unbounded position drift under K=0. Full-state routing is correct (same as Jan's working MSD implementation); the K=0 instability is addressed via curriculum nf. Literature precedent: CHyLL (arXiv:2512.10117) shows direct training at long nf diverges while curriculum converges; Farina & Piroddi (2011) establishes sub-sequence length as critical hyperparameter; Uy et al. (arXiv:2212.01418) demonstrates rollout training suppresses drift on marginally stable systems.
**Why curriculum fixes K=0**: At small nf, position drift per window O(nf·ε·Ts) is small — ANN learns absorber oscillation. Absorber displacement is physically zero-mean, so correctly learned corrections are also zero-mean and don't cause net long-horizon drift. Increasing nf progressively forces the ANN to maintain zero-mean corrections. Final sim-RMS stage continues training using dynamics already learned at nf=400, not just evaluates — the model adapts to the full-trajectory regime.
**Ruled out**: (1) nf=1000+ windowed validation — more expensive than sim-RMS (7M vs 8k steps). (2) Stay with Model B — doesn't fix K=0 drift, only fixes C_aug dead zone. (3) Increase nf to 8000 directly — computationally infeasible, equivalent to CHyLL failure mode. (4) Series augmentation — excluded per project scope.
**Constrains**: NF_CURRICULUM controls all training. Optuna objective unaffected (uses `train_model` directly). Model B (D-066) is superseded.

### [D-066] Model B routing: ANN → velocity rows [3,4,5] + aug rows [6,7]; C_aug removed
**Date**: 2026-06-30
**What**: Replaced D-065 C_aug routing with Model B routing in `gantry_interconnect_dynamic.py`. Changes: (1) `Parameterized_Linear_Output_Block` and C_aug removed; output is `Linear_Output_Block(Cd_norm)` only. (2) `VEL_AUG_IX = [3,4,5,6,7]` defined. (3) ANN `nw` changed from `NX_ANN=2` to `len(VEL_AUG_IX)=5`. (4) `expansion_matrix(AUG_IX, nxd)` → `expansion_matrix(VEL_AUG_IX, nxd)`. ANN input unchanged (sees full state + u).
**Why**: D-065 C_aug routing has a gradient dead zone by construction. C_aug is initialized near-zero (Frobenius norm = 1e-2). The gradient chain Loss→y→C_aug@x_aug→x_aug→ANN scales with ‖C_aug‖_F ≈ 0, so the ANN receives no learning signal. Confirmed by `diag_gradient_routing.py` on real gantry data: Model A (C_aug) ANN grad = 1.04e-2, Model B (vel routing) ANN grad = 2.85e-1, ratio 27x. This is the root cause of R²≈0 from diag12.
**Why velocity rows are safe (contra D-065 ruling)**: D-065 ruled out velocity routing citing "836x blowup (diag13)". That test used a non-zero-initialized ANN at long nf=400 rollout. Model B is safe at epoch 0 because ANN is zero-initialized: correction starts at 0, so initial position drift is zero. Velocity states have stable eigenvalue z=1-C*Ts/m < 1 (C>0 for gantry). Gradient of position loss w.r.t. velocity correction converges to Ts·m/C as T→∞ — a finite bound, not O(T²) as for position-row routing. `diag_spring_stiffness.py` confirms K=0 position routing gives O(T²) gradient growth; velocity routing is structurally bounded.
**Why position rows remain excluded**: Gantry X/Y/bridge axes have K=0 (no spring stiffness). DT position eigenvalue z=1 exactly. Additive correction to position accumulates without restoring force: gradient O(T) unbounded (confirmed `diag_spring_stiffness.py`). No spring stiffness can be added — this is the physical system.
**Literature**: Tustin-Net (Pozzoli et al. 2019/2020), van Esch et al. 2024 — multiple independent groups converged on ANN injection at force/velocity level, never at position level, for systems with integrating modes.
**Ruled out**: (1) C_aug routing (D-065): gradient dead zone, ANN learns nothing. (2) Full-state parallel (Jan's default): position rows give unbounded gradient at K=0. (3) CT force injection via LFR deriv(): gradient 7.89e-4, ~500x weaker than Model B (two CT integrations vs one DT). (4) Series-in (identity init): 177x stronger gradient but comes from position modification path — same drift risk.
**Constrains**: Output is now solely through fixed Cd_norm (no trainable C_aug). ANN must cause sufficient position change via the velocity→position integration for the loss signal to drive learning. The aug states [6,7] still exist and receive ANN correction — they are free latent states for any unmodeled dynamics the ANN discovers. First training run needed to confirm R² improvement.
