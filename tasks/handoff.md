# Session Handoff

_Previous sessions archived to `archive/sessions/`._

**Last written**: 2026-04-16 by Claude (Sonnet 4.6)

---

## Context

The LPV-LFR baseline implementation is complete and all verification checks pass (see
`archive/sessions/2026-04-16-handoff.md` for the full record). The current focus is
**parameter recovery**: recover true physical parameters from MATLAB-simulated data
using the `ParameterizedLFRBlock` and `train_param_recovery.py`.

A training run was attempted. Supervisor feedback identified 7 concrete problems with
the current setup. These are the active work items, ordered by priority.

---

## Open Blockers (carried forward)

- **LFR discretization paper**: Still not found. Less critical since RK4 is chosen.
- **M0 choice**: M0 = M(0) vs M(Y_nom=0.3). State explicitly in write-up.
- **Sample rate**: D-012 — 16 kHz (main.m) vs 20 kHz (ETEL spec), unresolved.
- **Float32 acceptability**: Run training in both dtypes, compare param_table().

---

## Parameter Recovery — 7 Open Issues

### Issue 1 — Channel normalization (Y dominates loss)

**Problem:** `F.mse_loss(Y_pred, q1_seg)` is unweighted in physical units. Y sweeps
600 mm; X1 and X2 hold near 0 m. Y's absolute error is orders of magnitude larger,
so gradients are almost entirely driven by Y. Parameters governing X1/X2 dynamics
(cg1, cg2, m1, m2) receive almost no gradient signal.

**Fix:** Normalize each channel by its standard deviation (measured once from the
training data) before computing the loss:
```python
sigma = q1_train.std(dim=0)   # (3,)  per-channel std
loss  = F.mse_loss(Y_pred / sigma, q1_seg / sigma)
```

**Status:** Not yet implemented.

---

### Issue 2 — Single trajectory (current priority)

**Problem:** All training data comes from one MATLAB trajectory: Y sweeps 0.3 to
-0.3 m while X1=X2=0 throughout. Parameters governing X dynamics are barely
identifiable because those channels carry almost no output variation. Multiple
shooting over segments of the same trajectory does not help — it is still the
same monotone Y sweep.

**Fix:** Generate multiple MATLAB trajectories from `export_lpv_sim.m` with varied
references (X1/X2 steps, different Y amplitudes, combined X+Y motion) and train
jointly on all of them. This excites all parameter sensitivities.

**Concrete next step:**
1. Extend `export_lpv_sim.m` to export at least 2-3 additional trajectories
   (e.g. X1/X2 step while Y holds; combined X+Y sweep).
2. Update `train_param_recovery.py` to load and concatenate multiple `.mat` files.
3. Re-run training and compare `param_table()` convergence.

**How to use multiple trajectories well:**

- Do **not** sample proportional to trajectory length only. That will overrepresent
  long or easy trajectories.
- Balance by **information content** instead.
- Start with one global `segment_len` for all trajectories; optimize the sampling
  strategy before introducing per-trajectory segment lengths.
- Group the trajectories by what they excite:
  - `T1`, `T6`: Y-only excitation
  - `T2`, `T3`: X-symmetric / `mh`-coupling contrast
  - `T4`, `T5`: rotational + coupled excitation
- Then allocate roughly equal batch budget to each group, so `T4` and `T5` do not
  get drowned out just because they are fewer or shorter.
- Parallel training should mean:
  - sample segments from different trajectories
  - stack them into one batch
  - simulate them together
  - update one shared parameter vector

**Status:** Not yet started. **Start here.**

---

### Issue 3 — MSE vs RMSE inconsistency in logging

**Problem:** The training loop logs `train_mse` and `val_mse` (units: m²). Step 5
of the evaluation reports per-channel RMSE (units: m). These are not directly
comparable. Comparing the logged `val_mse` against the Step 5 RMSE numbers gives
wrong magnitude intuition (RMSE = sqrt(MSE); for small errors RMSE >> MSE).

**Fix:** Either log RMSE everywhere (`loss.item() ** 0.5`) or add clear unit labels
to the printout so the two quantities are never compared directly.

**Status:** Not yet fixed.

---

### Issue 4 — Fixed initialization (no multi-start)

**Problem:** `_DETUNING_SIGNS = [+1, -1, +1, -1, ...]` is hardcoded. Every run
starts from the exact same ±10% detuned point. If the optimizer converges to a
local minimum, restarting simply hits the same minimum again.

**Fix:** Multi-start with random log-space initialization. Draw `log_params` from
e.g. `Uniform(-0.2, 0.2)` at the start of each run. Run several independent trials
and compare `param_table()` across runs to assess whether the landscape has one basin
or many.

**Status:** Not yet implemented.

---

### Issue 5 — Local minimum (Adam, gradient descent)

**Problem:** Adam converges to a local minimum, not the global one. Contributing
factors: single trajectory with limited excitation (Issue 2), fixed initialization
(Issue 4), and structural non-identifiability — the model observes only sums
kb1+kb2, cb1+cb2, Jb+Jh, so individual components of each pair are not identifiable
from output data alone without the `param_loss` regularization.

**Relationship to other issues:** Issues 2 and 4 are the primary root causes. Fix
those first before tuning the optimizer itself.

**Status:** Diagnosis only; blocked on Issues 2 and 4.

---

### Issue 6 — Log parameterization constraint

**Problem:** Positivity is enforced via `params_init * exp(log_params).clamp(min=1e-6)`.
The hard clamp kills gradients at the boundary and is a symptom of the optimizer
stepping outside a safe region.

**Two proposed alternatives:**

- **Jasper's suggestion:** Use a constrained optimizer (e.g. L-BFGS-B with box
  constraints in log space) where positivity is intrinsic to the optimizer, never
  requiring a clamp.

- **Quinten's suggestion:** Add a barrier/penalty term to the cost function (e.g.
  `-lambda * sum(log_params)`) that penalizes approaching zero smoothly, keeping
  gradients well-defined everywhere.

**Current status:** The `exp` reparameterization already guarantees positivity if
`log_params` stays finite; the clamp is only hit if the learning rate is too large
or the loss landscape is pathological near zero. Resolve Issues 2 and 4 first —
this may become a non-issue.

**Status:** Design decision pending; not yet implemented.

---

### Issue 7 — Identifiability limit

**Problem:** With a single trajectory where X1=X2=0, certain parameter combinations
are not identifiable from the output. This creates flat directions in the loss
landscape that appear as false local minima to a gradient optimizer. Channel
normalization (Issue 1) does not resolve this: even after normalizing, X1/X2 errors
carry little information because the channels barely move.

**Root cause:** The trajectory does not sufficiently excite all parameter
sensitivities. This is the same root cause as Issue 2.

**Fix:** Multiple diverse trajectories (Issue 2) is the correct solution. Identifiability
analysis (computing the Fisher information matrix or output sensitivity w.r.t. each
parameter along the trajectory) could confirm which parameters are unidentifiable
from the current data.

**Status:** Diagnosis only; fix is subsumed by Issue 2.

---

## Priority Order

| # | Issue | Status | Dependency |
|---|-------|--------|------------|
| 2 | Multiple trajectories | **Start here** | none |
| 1 | Channel normalization | Not started | none (can do in parallel) |
| 3 | MSE vs RMSE logging | Not started | none (trivial fix) |
| 4 | Multi-start initialization | Not started | needs Issue 2 first |
| 5 | Local minimum diagnosis | Blocked | needs Issues 2 + 4 |
| 6 | Log constraint | Design pending | resolve Issues 2+4 first |
| 7 | Identifiability | Subsumed by 2 | fix is Issue 2 |
