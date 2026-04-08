# Session Handoff

_Full session archived to `archive/sessions/2026-04-07-handoff.md`._

**Last written**: 2026-04-07 by Claude (Sonnet 4.6)

---

## What Was Found Out This Session

### `lpv_lfr_baseline` package architecture
The package implements the LPV-LFR baseline for a 3-DOF gantry (X1, X2, Y axes).
Key files:
- `physics.py` — all physical constants as `torch.float64` CPU tensors (M0, M1, M2, K, C, P, ts)
- `lfr_forward.py` — single-step LPV update: M(Y)=M0+M1·Y+M2·Y², `torch.linalg.solve`, dtype-neutral
- `lfr_simulate.py` — RK4 integration over full trajectory, 3 BPTT modes (full/truncated/checkpoint)
- `lfr_param_block.py` — `ParameterizedLFRBlock`: 10 trainable params via log/exp reparameterization,
  `register_buffer` for Lb, d, P, ts (auto-move with `.to(device)`)
- `train_param_recovery.py` — Step 3b training script: recovers true physical params from MATLAB data
- `data_utils.py` — `compute_rmse_baseline()` and `load_gantry_data()` (deepSI format)
- `compare_dtype.py` — standalone script: float64 vs float32 simulation accuracy vs MATLAB q1

### GPU support added to `train_param_recovery.py`
**Problem**: module-level physics tensors from `physics.py` are always CPU. Using them inside
the training loop means GPU tensors (block params) and CPU tensors (M0, M1, M2) mix → crash.

**Fix**: use `block._Lb`, `block._d`, `block._P`, `block._ts` (registered buffers) instead of
importing from `physics.py`. These move automatically with `block.to(device)`. Key changes:
- `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` — detect GPU
- `block = ParameterizedLFRBlock(...).to(device)`, `x0 = X0_LOGICAL.to(device)`
- `u_train.to(device)`, `q1_train.to(device)`
- Training loop: `_build_matrices(params, block._Lb, block._d)`, `simulate(..., block._P, block._ts)`
- `_simulate_no_grad`: same buffer pattern

### Train/val split removed from `train_param_recovery.py`
**Rationale**: only 10 params (M0, M1, M2 scalars, K[3], C[3]) cannot overfit a 35001-step
trajectory. Jan's validation / early stopping is designed for ANN models. The real go/no-go
criterion is the parameter recovery table (`block.param_table()`), not val MSE.
`load_gantry_data()` in `data_utils.py` still exists for potential future use (deepSI format).

### N_STEPS=4000 issue — MSE=0, no gradient
With N_STEPS=4000 (first 0.2s), Y barely moves from 0.3m → M(Y) detuning error is negligible
→ MSE≈0 → no gradient → params unchanged after 2 epochs. Fix: `N_STEPS=None` (full 35001 steps,
Y sweeps 0.3→-0.3m, ΔY=0.6m covers full operational range).

### Float64 vs float32 simulation accuracy (`compare_dtype.py`)
Results vs MATLAB ground truth (q1, float64):

| Channel | float64 vs MATLAB | float32 vs MATLAB |
|---------|------------------|--------------------|
| X1      | ~0 (1e-14 range) | ~5e-06 m          |
| X2      | ~0 (1e-14 range) | ~2e-06 m          |
| Y       | 4.0e-12 m        | 2.2e-05 m         |

float32 is 7 orders of magnitude worse on Y (22 µm floor). The comparison is not perfectly fair
(q1 is float64, MATLAB uses float64 internally) — there's no ideal fair comparison.

**Supervisor concern**: floating-point discretization/integration errors stack over 35001 steps.
float32 gives 22 µm on Y — this is the numerical floor, independent of parameter detuning.
The detuning signal is ~29 mm on Y, so float32 has 3 orders of margin there. However, the
concern is whether float32 errors are uniform or accumulate in ways that corrupt gradient signals.

**Recommendation**: stick with float64. The codebase was designed for it. float64 on NVIDIA RTX
2080 Ti is ~1/32 of float32 TFLOPS (~420 vs ~13,500 GFLOPS) but the simulation is memory-bound
not FLOP-bound, so the slowdown is milder in practice.

### Float32 acceptability: definitive test
There is no analytically fair comparison (MATLAB is float64, Python-vs-Python is not vs ground truth).
The only definitive test is: run `train_param_recovery.py` in float64 AND float32, compare the final
`param_table()` columns. If recovered parameters match true values equally well, float32 is acceptable.
This test has NOT been run yet.

### Lambda regularization and parameter scale invariance
`ParameterizedLFRBlock` uses the same Lambda formula as Jan's `Parameterized_Linear_State_Block`:
```python
Lambda[i] = RMSE_baseline / params_init[i]
```
This penalizes relative deviations equally regardless of scale — critical because parameters span
3 orders of magnitude (Lb~4e-3 m vs M~10 kg). The log/exp reparameterization makes Adam steps
multiplicative (scale-invariant). Design is directly aligned with Jan's framework.

---

## Open Blockers (carried forward)

- **LFR discretization paper**: Still not found. Less critical since RK4 is chosen.
- **M0 choice**: M0 = M(0) vs M(Y_nom=0.3). State explicitly in write-up.
- **Sample rate**: D-012 — 16 kHz (main.m) vs 20 kHz (ETEL spec), unresolved.
- **April 9 meeting**: Confirm with supervisor whether trainable inertia parameters affect
  Delta^b structure during training (D-017). Meeting was scheduled — outcome unknown.
- **Float32 acceptability**: Run training in both dtypes, compare param_table() (see above).
  Recommendation: use float64 for correctness-first approach.

---

## Exact Next Steps

### Step 1 — Run parameter recovery training (float64, full trajectory)
```
conda run -n GraduationProject python -m lpv_lfr_baseline.train_param_recovery
```
Expected config: `N_STEPS=None`, `EPOCHS=500`, `LR=1e-3`, `SEGMENT_LEN=500`.
Hardware: 7× NVIDIA RTX 2080 Ti (11 GB each). float64 is ~1/32 TFLOPS of float32 on RTX 2080 Ti.
Estimated time: unknown — first run will reveal per-epoch timing.

**Success criterion**: `param_table()` shows recovered params converging toward true values.
The detuning is 2–10% on each parameter; recovery should bring error below 1% if training works.

### Step 2 — Interpret and log results
After training completes:
- Read the printed `param_table()` (detuned → trained → true)
- Log result in `docs/decisions.md` under D-033 / D-034
- If params converge: mark Step 3b complete in `tasks/todo.md`
- If params diverge or loss plateaus: diagnose (check gradient magnitudes, LR sensitivity)

### Step 3 — Continue Step 2 pipeline (LPV vs frozen LTI comparison)
The tasks/todo.md Step 2 items (frozen LTI, nat-freq plot, trajectory comparison figure) were
NOT addressed this session. Resume after Step 3b is validated.

See `tasks/todo.md` Step 2 section for the five exact implementation steps with layout specs.

---

## Files Modified This Session

| File | Change |
|------|--------|
| `lpv_lfr_baseline/train_param_recovery.py` | GPU support, removed val split, N_STEPS cap, progress print |
| `lpv_lfr_baseline/compare_dtype.py` | Created — float64 vs float32 vs MATLAB comparison script |
| `archive/sessions/2026-04-07-handoff.md` | Archived 2026-04-03 handoff content |

---

## Proposed Improvements for Claude / Codex

None at this time.
