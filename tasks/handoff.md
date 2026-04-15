# Session Handoff

_Previous sessions archived to `archive/sessions/`._

**Last written**: 2026-04-13 by Claude (Sonnet 4.6)

---

## COMPLETED — LPV-SS → True LPV-LFR Conversion

The conversion from collapsed LPV-SS to genuine LFR-first structure is **complete and fully verified**.

### What was done

All files rewritten/updated to implement the genuine LFR interconnection:

| File | Change |
|------|--------|
| `lpv_lfr_baseline/core/physics.py` | Added `build_poly_constants()` — differentiable builder for N0, N1, N2, alpha, beta, gamma |
| `lpv_lfr_baseline/core/lfr_forward.py` | **Full rewrite** — genuine LFR-first signal flow, new signature |
| `lpv_lfr_baseline/core/lfr_simulate.py` | Updated rk4_step/simulate/simulate_frozen signatures |
| `lpv_lfr_baseline/blocks/lfr_block.py` | G and poly constants precomputed in `__init__`, stored as buffers |
| `lpv_lfr_baseline/blocks/lfr_param_block.py` | G and poly constants rebuilt in `forward()` from current nn.Parameter values |
| `lpv_lfr_baseline/scripts/train_param_recovery.py` | Updated `_SimWrapper` and `_run_no_grad` |
| `lpv_lfr_baseline/scripts/compare_dtype.py` | Updated simulate calls |
| `lpv_lfr_baseline/scripts/data_utils.py` | Updated simulate calls |
| `lpv_lfr_baseline/scripts/validate_lfr.py` | Updated simulate calls |
| `lpv_lfr_baseline/scripts/plot_lpv_vs_frozen.py` | Updated simulate calls |
| `lpv_lfr_baseline/tests/test_jan_compat.py` | Updated all call sites, revised Check F |
| `lpv_lfr_baseline/tests/test_augmentation_compat.py` | Updated lfr_forward calls |

### All verification checks pass

- `lfr_forward.py`: 6/6 checks PASS including **structural audit (Check 4)**
- `lfr_simulate.py`: All checks PASS (BPTT modes, autograd)
- `lfr_block.py`: 5/5 checks PASS
- `lfr_param_block.py`: 7/7 checks PASS (gradient to log_params confirmed)
- `test_jan_compat.py`: 12/12 checks ALL PASS (including Check F: gradient via G.Bw@w)
- `test_augmentation_compat.py`: 2/2 checks PASS

### The decisive structural audit (Check 4)

```
Check 4: STRUCTURAL AUDIT — w upstream of xdot via G.Bw  (batch=1)
  w.grad is not None           : True
  w.grad.abs().max() > 0       : True
  w.grad.abs().max()           : 2.935899e+00
Check 4: PASS
```

`w` is now causally upstream of `xdot` through `G.Bw`. This is what distinguishes
true LFR-first from collapsed LPV-SS.

### New function signatures

```python
# physics.py
build_poly_constants(m1, m2, mb, mh, Jb, Jh, Lb, d) -> (alpha, beta, gamma, N0, N1, N2)

# lfr_forward.py
lfr_forward(x, u, Y, G, K, C, mh, alpha, beta, gamma, N0, N1, N2) -> (xdot, z, w, y)

# lfr_simulate.py
rk4_step(x, u_logical, G, K, C, mh, alpha, beta, gamma, N0, N1, N2, ts, Y_override=None)
simulate(x0, u_seq_stage, G, K, C, mh, alpha, beta, gamma, N0, N1, N2, P, ts, ...)
simulate_frozen(x0, u_seq_stage, G, K, C, mh, alpha, beta, gamma, N0, N1, N2, P, ts, Y_freeze)
```

### Pattern for callers

**Non-trainable (fixed physics):**
```python
from lpv_lfr_baseline.core.physics import M0, M1, M2, K, C, P, ts, build_poly_constants
from lpv_lfr_baseline.core.physics import mh as _mh, m1 as _m1, ...
from lpv_lfr_baseline.core.lfr_matrices import build_G_matrix

G = build_G_matrix(M0, M1, M2, K, C)
alpha, beta, gamma, N0, N1, N2 = build_poly_constants(_m1, _m2, _mb, _mh, _Jb, _Jh, _Lb, _d)
simulate(x0, u, G, K, C, _mh, alpha, beta, gamma, N0, N1, N2, P, ts)
```

**Trainable (inside forward() of ParameterizedLFRBlock):**
```python
params = self._recover_params()
kb1, kb2, cg1, cg2, cy, cb1, cb2, mh, m1, m2, mb, Jb, Jh = params
M0, M1, M2, K, C = _build_matrices(...)
G = build_G_matrix(M0, M1, M2, K, C)              # rebuilt each call
alpha, beta, gamma, N0, N1, N2 = build_poly_constants(m1, m2, mb, mh, Jb, Jh, ...)
rk4_step(x, u, G, K, C, mh, alpha, beta, gamma, N0, N1, N2, ts)
```

---

## Open Blockers (carried forward)

- **LFR discretization paper**: Still not found. Less critical since RK4 is chosen.
- **M0 choice**: M0 = M(0) vs M(Y_nom=0.3). State explicitly in write-up.
- **Sample rate**: D-012 — 16 kHz (main.m) vs 20 kHz (ETEL spec), unresolved.
- **Float32 acceptability**: Run training in both dtypes, compare param_table().

---

## Exact Next Steps

### Step 1 — Run parameter recovery training (float64, full trajectory)
```
conda run -n GraduationProject python -m lpv_lfr_baseline.scripts.train_param_recovery
```
Expected config: `N_STEPS=None`, `EPOCHS=1000`, `LR=1e-3`, `SEGMENT_LEN=4000`.

**Success criterion**: `param_table()` shows recovered params converging toward true values.

### Step 2 — Interpret and log results
After training:
- Read the printed `param_table()` (detuned → trained → true)
- Log result in `docs/decisions.md` under D-033 / D-034
- If params converge: mark Step 3b complete in `tasks/todo.md`
- If params diverge or loss plateaus: diagnose (gradient magnitudes, LR sensitivity)

### Step 3 — Continue Step 2 pipeline (LPV vs frozen LTI comparison)
See `tasks/todo.md` Step 2 section for the five exact implementation steps.
