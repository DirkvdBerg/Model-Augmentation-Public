# L-BFGS polish phase: implementation plan

Design and rationale: **D-171** (`docs/decisions.md:22`). Literature: `docs/references.md`, section
"Adam to L-BFGS handover". This file is the code plan only: exactly which files change, what the
new code is, and how it is proved to work. Nothing here re-argues a decision.

**Scope.** A terminal quasi-Newton phase, staged after Adam in the same run, plus a resume entry
point that can start directly in that phase.

**Invariant that governs every choice below.** With `lbfgs=False` the Adam path must be
bit-identical to today's. Every change is therefore additive and gated, and no existing call
signature loses a default.

---

## 1. New files

### 1.1 `model_augmentation/fit_systems/lbfgs_polish.py` (NEW, ~220 lines)

Marker: `__project_origin__ = "added"` at module top (CLAUDE.md tracking rule).

Framework-level and gantry-agnostic. It touches a system only through
`loss`, `make_training_data`, `norm`, `cal_validation_error`, `parameters_with_names` and
`simulator`, so `scripts/gantry`, the MSD pipeline and `encoder_initialisation` can all call it.
No `RunConfig` import; the caller passes plain numbers.

```python
def lbfgs_polish(fit_sys, train_sys_data, *, nf, stride,
                 n_windows=4096, seed=42,
                 max_iter=500, inner_iter=20, history_size=50,
                 tol_grad=1e-12, tol_change=1e-14,
                 freeze=(), val_sys_data=None, validation_measure='sim-RMS',
                 time_budget_s=None, verbose=True) -> dict
```

Internal helpers, all private:

| Helper | Does |
|-|-|
| `_snapshot(fit_sys)` / `_restore(fit_sys, snap)` | deep-copied CPU state dicts of `hfn`, `encoder` and every bare `nn.Parameter`, for rollback |
| `_build_fixed_batch(...)` | the one deterministic batch, see below |
| `_collect_params(fit_sys, freeze)` | one flattened list out of `parameters_with_names` |
| `_meters(fit_sys)` | reads the last entry of `Probe_combo_err` / `Probe_orth_frac` / `Probe_V_orth` / `Probe_param_loss` when present, else `{}` |
| `_eager(fit_sys)` | context manager: `simulator._compiled = None` for the duration, restored after |

Execution order inside the function:

1. `snap = _snapshot(fit_sys)`.
2. `before = fit_sys.cal_validation_error(val_sys_data, validation_measure)` when `val_sys_data`
   is given. This fires `validation_probes`, so `_meters` has a "before" row.
3. `_build_fixed_batch`: `fit_sys.make_training_data(fit_sys.norm.transform(train_sys_data), nf=nf, stride=stride)`,
   seeded subsample of `n_windows` rows (`0` = all), cast to the model dtype, moved **once** to
   `next(fit_sys.hfn.parameters()).device`, split into `(uh, yh, uf, yf)` plus kwargs zipped from
   `fit_sys.simulator.extra_array_names`.
   The zip mirrors `interconnect.py:921-935` and carries a comment saying so; `fit()` is not
   refactored to share it, because that would touch the Adam path.
   No `indexer.record_rows` swap is needed here, unlike `_NfProbe.__init__` (`training.py:115-133`),
   because these are the training records the indexer already holds.
4. Inside `_eager(...)`: build `torch.optim.LBFGS(params, lr=1, max_iter=inner_iter,
   history_size=history_size, tolerance_grad=tol_grad, tolerance_change=tol_change,
   line_search_fn='strong_wolfe')` as a LOCAL variable, and set `fit_sys.loss_stats = None`
   for the duration so line-search evaluations do not pollute the nf probe's running mean.
5. Outer loop `ceil(max_iter / inner_iter)` calls to `opt.step(closure)`, with a non-finite check,
   the wall-clock check and one log line between calls. Break when the loss stops moving
   (L-BFGS's own tolerances make `step` a no-op) or on either guard.
6. `after = fit_sys.cal_validation_error(...)`, `_meters` again.
7. Accept if `after < before` **and** no meter degraded (D-171); else `_restore(fit_sys, snap)`.
8. Return the outcome dict (below), whichever branch ran.

Returned dict, which is also exactly what goes into the checkpoint meta:

```
ran, accepted, reason, n_iter, n_closure, wall_s,
loss_first, loss_last, val_before, val_after,
meters_before, meters_after, n_windows_used, params_polished, frozen
```

Two guards worth naming because they are silent failures otherwise:
- every collected parameter must be a leaf with `requires_grad=True`; a parameter that receives no
  gradient gets a zero grad installed, so torch's flat-gradient gather cannot raise mid-phase;
- `_build_fixed_batch` asserts the extra-array count matches `extra_array_names`, the same check
  `fit()` makes at `interconnect.py:924-929`.

### 1.2 `scripts/gantry/gantry_dynamic/tests/test_lbfgs_polish.py` (NEW, ~150 lines)

Alongside the two existing tests in that directory. Contents in section 4.

---

## 2. Modified files

### 2.1 `scripts/gantry/gantry_dynamic/config.py`

**(a) Fields**, added to `RunConfig` next to the training hyperparameters (after `epochs`, near
line 134), so they sit with the other optimiser knobs:

```python
lbfgs: bool = False                  # the switch; False = today's run, exactly
lbfgs_max_iter: int = 500            # a MAXIMUM; tolerances fire first (D-171)
lbfgs_inner_iter: int = 20           # iterations per opt.step() call, i.e. logging granularity
lbfgs_history: int = 50              # m, matching jax_sysid / Drenth
lbfgs_windows: int = 4096            # size of the ONE fixed batch; 0 = full batch
lbfgs_tol_grad: float = 1e-12
lbfgs_tol_change: float = 1e-14
lbfgs_freeze: tuple = ()             # e.g. ('encoder',); the D-171 fallback, off by default
lbfgs_time_budget_s: Optional[int] = None
start_phase: str = 'adam'            # 'adam' | 'lbfgs'; 'lbfgs' requires a resume checkpoint
```

**(b) Validation** in `__post_init__` (line 195), three checks, each with a message in the house
style saying what to do:
- `start_phase not in ('adam', 'lbfgs')` raises;
- `start_phase == 'lbfgs'` with `lbfgs=False` raises (nothing would run);
- `lbfgs=True` with `lbfgs_windows < 0`, or `lbfgs_inner_iter <= 0`, or
  `lbfgs_max_iter < lbfgs_inner_iter` raises.

The `start_phase == 'lbfgs'` without `RESUME_CHECKPOINT` case cannot be checked here, because the
checkpoint is an env var read in the entry file; it is checked at the call site (2.4).

**(c) `config_json_dict`** (line 368): every field above added, so a polished run is
distinguishable in `config.json` from an Adam-only one.

**NOT changed:** `hp` (line 293). The L-BFGS knobs are runtime, not model hyperparameters, so they
stay out of the checkpoint `hp` schema exactly as `n_its` does (`config.py:135-141`).

### 2.2 `scripts/gantry/gantry_interconnect_dynamic.py`

**(a)** The matching block inside `CFG = RunConfig(...)`, in the file's commented style, placed
after the `epochs` entry. Comments carry the D-171 rationale in one or two lines each, not the
whole argument.

**(b)** `use_f64` comment at lines 127-134: **one sentence appended, nothing deleted.** The
existing text is correct and measured; its stated reason is "Adam's per-parameter normalisation
absorbs the rollout noise", which is exactly the property the polish phase does not have. The
appended sentence says so and points at D-171.

**(c)** Around line 336, where `RESUME_CHECKPOINT` is read: raise if `cfg.start_phase == 'lbfgs'`
and no checkpoint was given, and print the phase alongside the existing resume line.

### 2.3 `scripts/gantry/gantry_dynamic/training.py`

**(a) `resolve_checkpoint(path)` (NEW function, ~40 lines).** Normalises both formats to
`(hfn_sd, encoder_sd, optim_sd_or_None, meta_dict)`:

| Input | Route |
|-|-|
| `<path>.pt` and `<path>.npz` exist | today's pair format; `ckpt['hfn']`, `ckpt['encoder']`, `ckpt.get('optimizer')`, meta from the `.npz` |
| `<path>` ends in `.pth`, or `<path>_best.pth` exists | deepSI whole-system dump; `d['hfn'].state_dict()`, `d['encoder'].state_dict()`, `None`, meta from `bestfit`/`epoch_counter`/`batch_counter`/`Loss_val` |

Fixture proving the second route exists locally:
`scripts/gantry/meeting/meeting-07-09-2026/server/checkpoints/SSE_Interconnect_Composed_p2LPDA_best.pth`
(job 81262, verified to load on CPU and to contain `hfn`, `encoder`, `norm`, `simulator`,
`optimizer`, `bestfit`).

**(b) `load_checkpoint` (line 30) rewritten to sit on top of `resolve_checkpoint`.** Behaviour on a
pair-format checkpoint is unchanged, including the D-076 pre-JE guard (which now reads the
normalised `hfn_sd`, so it protects both formats identically). Two additions:
- an **architecture-mismatch guard**: compare keys and shapes before `load_state_dict` and raise
  naming the offending entries, because resuming 81265 (`nx_ann=2`, 16x2) into an 81262 config
  (`nx_ann=8`, 24x3) is the mistake this workflow invites;
- the optimizer state is restored only when present **and** `start_phase == 'adam'`.

**(c) `train_model_with_diagnostics` (line 305)**, three edits:
- new keyword-only parameters `lbfgs_kwargs=None`, `start_phase='adam'`, both defaulting to today's
  behaviour;
- `start_phase == 'lbfgs'` sets `epochs_remaining = 0` and skips the `train_model` call at line 336
  (the nf probe install at line 334 stays, so the before/after meters still exist);
- after training and **before** the existing `save_checkpoint_weights` at line 360: write
  `gantry_ckpt_{run_id}_pre_lbfgs.{pt,npz}`, then call `lbfgs_polish`, then continue into the
  existing save. A crash inside the phase therefore cannot lose the Adam result.

**(d) `.npz` meta (line 386)** gains `lbfgs_outcome` (the returned dict, `json.dumps`ed like `hp`),
`resumed_from`, `resumed_at_epoch` and `start_phase`. Existing keys keep their names and order, so
old checkpoints still load.

### 2.4 `scripts/gantry/gantry_dynamic/model.py`

One edit. `train_model` (line 266) gains a `polish=None` keyword; when given, it is called after
`fit_sys.fit(...)` returns at line 284 and **before** the `fit_sys.cpu()` at line 317, so the phase
runs on the training device. Two lines plus a comment. The Adam eps guards at lines 242 and 270 are
untouched and still run.

Alternative considered and rejected: calling the phase from `training.py` after `train_model`
returns. That would run it on the CPU, since line 317 has already moved the model back.

---

## 3. What is deliberately NOT touched

- **`model_augmentation/fit_systems/interconnect.py`.** No edit. `fit()` keeps its Adam loop, its
  closure and its counters. The cost is six duplicated lines of the `extra_array_names` zip in the
  new module, commented as mirroring `interconnect.py:921-935`. Refactoring `fit()` to share them
  would put the Adam path at risk for no functional gain.
- **`fit_sys.optimizer`.** Stays the Adam instance. Replacing it breaks the eps guards
  (`model.py:242`, `model.py:270`) and the checkpoint contract (`training.py:26`).
- **The L-BFGS optimizer state is not persisted** (D-171). Resuming *into* the phase does not need
  it; resuming a half-finished phase would lose only the curvature history, which rebuilds in a few
  iterations.
- **`save_checkpoint_weights` schema** (`training.py:21-27`). Unchanged keys.
- **`docs/` siblings.** `docs/gpu-nf12000-feasibility-2026-08-31.md` contradicts itself on the cost
  of float64 (line 17 versus line 203) and `scripts/gantry/drift-fix-trials/research/thread-AB-optimizer-mechanics.md:505`
  mis-describes arXiv:2510.27018. Both are logged as open items in D-171 and are NOT fixed by this
  work without a separate go-ahead.

---

## 4. Tests (`test_lbfgs_polish.py`)

Fixture: the job 81262 checkpoint above plus
`data/gantry/matlab/trajectory/augmentation_ma50_b140-230_a6_z03` (present locally, 22 records).
Run config from the job log: nf=400, stride=10, batch 512, float32, closed loop, `nx_ann=8`, 24x3,
`na_nb=29`, `orth=False`, `joint=False`.

| # | Test | Asserts | Cost |
|-|-|-|-|
| 1 | smoke | `n_windows=64`, `max_iter=2` runs end to end; batch builds through `augment_training_data`; every collected parameter has a gradient after one closure | seconds |
| 2 | determinism | two closure calls without a step return **bit-identical** losses. This is the precondition strong Wolfe needs and it tests the fixed-batch and eager decisions directly | seconds |
| 3 | descent | loss on the fixed batch is non-increasing over the phase | seconds |
| 4 | rollback | with acceptance forced to fail, every parameter matches the snapshot **exactly**, tensor by tensor | seconds |
| 5 | freeze | `freeze=('encoder',)` leaves encoder parameters bit-identical | seconds |
| 6 | loader | both checkpoint formats resolve to equal `hfn`/`encoder` state dicts; the mismatch guard raises on 81265-into-81262 with a message naming a key | seconds |
| 7 | no-op | `lbfgs=False` leaves the Adam path unchanged: `train_model_with_diagnostics` makes the identical `fit()` call | seconds |

**Verified by construction only, and it will be reported as such:** the `orth=True` and
`joint_estimation=True` branches of the acceptance rule. The local fixture has both off. They get a
stub unit test for the meter comparison logic, and a real confirmation needs one cluster run.

---

## 5. Order of work

1. `lbfgs_polish.py` with tests 1 to 5 against the fixture.
2. `resolve_checkpoint` and the `load_checkpoint` rewrite, with test 6.
3. Config fields, validation and `config_json_dict`.
4. The `training.py` and `model.py` call sites, with test 7.
5. The `CFG` block and the `use_f64` comment sentence.
6. Measure the two tolerances on the fixture and replace the placeholder defaults in section 2.1(a).

Gate between 1 and 2: tests 1 to 5 pass. Gate before reporting done: all seven pass, and the two
construction-only items are stated plainly rather than claimed as tested.

---

## 6. Numbers still to measure, not guessed

- `lbfgs_tol_grad` and `lbfgs_tol_change`: one measurement of the loss and gradient-norm magnitude
  at the fixture checkpoint. The torch defaults (`1e-7`, `1e-9`) are meaningless against a loss of
  order 1e-9 and would end the phase at iteration one.
- `lbfgs_windows`: whether 4096 windows fit alongside float64 activations at the target `nf`. If
  not, the closure gradient-accumulates in chunks, which changes step 5 of section 1.1 and nothing
  else.
- The switch epoch, per D-171 open item 1. Out of scope for this plan: it is a run parameter, not
  code.

---

## 7. STATUS: implemented 2026-09-04

All seven planned gates plus twenty more are in
`scripts/gantry/gantry_dynamic/tests/test_lbfgs_polish.py`: **27 tests, all passing**
(`python -m unittest discover -s scripts/gantry/gantry_dynamic/tests -p test_lbfgs_polish.py`,
~82 s, the fixture tests dominating).

Three departures from this plan, each recorded with its reason in D-171's implementation note:
meters are computed rather than read from the probes (`fit()` replaces `__dict__` at the end, so the
surviving probes are stale copies); `model.py` needed no edit at all, because the device round-trip
moved into `run_lbfgs_polish` and now serves both entry paths; and `resolve_checkpoint` gained the
`*_best.pth` route, which is the only format available for a run that was killed mid-training.

Section 6's tolerance question is answered: the loss on the fixture is 1.29e-9, so torch's defaults
would have ended the phase at iteration one, and `1e-12` / `1e-14` are in use. `lbfgs_windows`
replaces it as the open number: at 256 of 66,612 windows the phase improved the training loss by
0.55% and did NOT improve sim-RMS, so it rolled back. Size it on the GPU before judging the phase.
