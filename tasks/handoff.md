# Session Handoff

_Previous sessions archived to `archive/sessions/`._

**Last written**: 2026-04-16 by Claude (Sonnet 4.6) — updated normalization design

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

**Problem:**
`F.mse_loss(Y_pred, q1_seg)` is unweighted in physical units. The output vector
has three channels: `[X1, X2, Y]` in metres (stage coordinates). Y sweeps ±300 mm;
X1 and X2 remain near 0 m in Y-only trajectories. Because MSE scales as amplitude²,
Y's contribution to the loss is 36×–900× larger than X1/X2 depending on how much
X moves. Gradients flowing back to parameters governing X dynamics (cg1, cg2, m1,
m2) are suppressed by this same factor. This is not a numerical instability — it is
correct but uninformative gradient signal: Y dominates because it dominates the
data, not because it dominates the physics.

---

**What the literature says:**

The standard in classical system identification (Ljung 1999, §7.2–7.3) is the
weighted prediction error method (PEM):

```
V(θ) = (1/N) Σ_k ε(k,θ)ᵀ Λ⁻¹ ε(k,θ)
```

where Λ is the output noise covariance. The unweighted MSE we use corresponds to
`Λ = I`, which is only theoretically valid when all output channels have equal noise
power and equal signal amplitude. Neither holds for our gantry.

The MATLAB System Identification Toolbox normalizes outputs to unit variance before
any gradient computation (Ljung, 2014). The robot identification benchmark
(Weigand et al., 2023) uses NRMSE per channel — each channel divided by its
standard deviation — as both training criterion and evaluation metric. The physics-
informed neural network literature (Karniadakis et al., Nature Reviews Physics,
2021) lists output normalization to O(1) scale as a prerequisite for training
stability in systems with multi-scale outputs.

**What Jan Hoekstra (EJC 2025) does — and why it does not directly apply:**

Hoekstra bakes normalization into the model architecture itself via transformation
matrices (Section 3.5, eq. 9a–9b):

```
f̄_base = T_x · f_base(T_x⁻¹ x̄, T_u⁻¹ u)
h̄_base = T_y · h_base(T_x⁻¹ x̄, T_u⁻¹ u)
```

where `T_y = diag(σ_y⁻¹)`. The loss (eq. 5a) is then MSE on `ȳ` — the entire
coordinate system is rescaled, not just the loss weights. He computes σ_y by
simulating the baseline model under nominal input and initial conditions, then
taking the std of that simulation output. Crucially, σ_y comes from the baseline
model simulation, not from the raw training data.

**Why this does not apply to us:**
1. Our model is physics-based with physical parameters — baking T_y into the LFR
   matrices would transform all internal signals into normalized units, destroying
   the direct physical interpretation of the states and outputs.
2. We are doing parameter recovery, not ANN augmentation. The learning components
   in Hoekstra's framework are ANNs that genuinely need normalized inputs for
   stable gradient flow. Our only optimizable variable is `log_params` (a 13-vector
   in log space) — no ANN weight matrices that require normalization.
3. Hoekstra uses a single broadband multisine dataset that excites all channels by
   design. We have multiple distinct trajectories with very different per-channel
   excitation levels. His σ_y is stable because the multisine covers the full
   operating range. Ours must be computed carefully.

**The correct equivalent for our case:**
Apply sigma as a loss weight only — compute it once from the training data, divide
the error before squaring. Mathematically equivalent to Hoekstra's loss in
normalized coordinates, but without touching the model.

---

**Options considered for computing sigma:**

| Option | Source | Problem |
|--------|---------|---------|
| Per-batch sigma | Each `q1_seg` | Noisy, changes every epoch — bad |
| Per-trajectory sigma | Each traj separately | Sigma_X ≈ 0 for Y-only trajs → amplification |
| Active-traj sigma | Concatenated active `q1` | Changes with `ACTIVE_TRAJ_IDS` |
| Fixed physical sigma | Known design ranges (e.g. σ_Y=0.3) | Requires knowing X design amplitude; not data-driven |
| All-TRAJ_SPECS sigma | All 6 trajectories concatenated | Stable regardless of active set |

**Resolved decision:**
Compute sigma from the **concatenated active training trajectories** after they are
loaded in Step 2, before the training loop. This is what Hoekstra and the benchmark
both do — sigma comes from the actual training data. The key requirement is that
the active trajectory set must include X-motion trajectories (T2–T5); if only
Y-only trajectories are active, sigma_X collapses to near-zero and the 1e-4 clamp
does all the normalization work — which is a diagnostic signal that Issue 2
(trajectory diversity) is not yet solved.

**Is using training data sigma "cheating"?**
No. In a real experiment you would compute sigma from measured training data. Our
MATLAB trajectories are our "measured data" — the observations we would have from
the real system. The sigma does not encode knowledge of the true parameters; it is
a statistic of the observed outputs. This is exactly what Hoekstra and Ljung do.

**The amplification concern — resolved:**
If sigma_X is small (X barely moves), then (error_X / sigma_X)² is large for even
a small X error. This is the *correct* behavior: it says "X errors are large
relative to how much X moves in this dataset." If sigma_X is small because the
active trajectory set poorly excites X, the amplified normalized X error is a
correct reflection of that poor coverage — not a normalization pathology. The fix
is Issue 2 (better trajectories), not a different sigma formula.

**Relationship to identifiability (Issue 7):**
Normalization removes the artificial dominance of Y in the loss. After
normalization, gradients correctly reflect the physical identifiability structure.
If gradients for X-governing parameters are still near zero after normalization,
that means those parameters have near-zero output sensitivity in the current data —
which is the identifiability problem (Issue 7), not a scale problem (Issue 1).
Normalization is necessary but not sufficient. Issue 2 (diverse trajectories) is
the root fix.

---

**Implementation plan:**

```python
# After Step 2 (trajs loaded), before training loop
all_q1 = torch.cat([traj['q1'] for traj in trajs], dim=0)  # (N_total, 3)
sigma = all_q1.std(dim=0).clamp(min=1e-4).to(device)       # (3,) metres
# Log at startup so the user can see what normalization is applied:
# sigma_X1 = X mm,  sigma_X2 = Y mm,  sigma_Y = Z mm

# In training loop — replace F.mse_loss(Y_pred, q1_seg):
err      = (Y_pred - q1_seg) / sigma      # normalized error, (batch, T, 3)
mse_loss = err.pow(2).mean()              # dimensionless scalar
```

The same normalization applies to the validation loss:
```python
val_err  = (wrapper(val_x0, val_u) - val_q1) / sigma
val_mse  = val_err.pow(2).mean().item()
```

Sigma must be saved in the checkpoint for reproducibility:
```python
'sigma': sigma.cpu(),
```

**Diagnostic to add alongside normalization:**
Print per-parameter gradient norms at `LOG_INTERVAL` after `loss.backward()`:
```python
g = block.log_params.grad  # (13,)
# Print as table: param_name → |grad|
```
Before normalization: X-governing params (cg1, cg2, m1, m2) should show near-zero
gradient. After normalization with diverse trajectories: all params should show
gradients of comparable magnitude. If X-governing params are still near zero, the
root cause is Issue 2, not Issue 1.

**Future-proofing for param_loss (currently PARAM_LOSS_WEIGHT = 0.0):**
`param_loss` is calibrated via `RMSE_baseline` (D-034). When the training loss was
in physical units (metres²), RMSE_baseline was also in physical units (metres).
After normalization the training loss is dimensionless. If param_loss is re-enabled
in the future, RMSE_baseline must be normalized consistently before being passed to
`ParameterizedLFRBlock`. The normalized baseline is:

```python
rmse_baseline_normalized = rmse_baseline / sigma.norm()  # or /sigma_Y only
```

This is a one-line change, but it must not be forgotten. The checkpoint stores
`sigma` specifically so this conversion is always possible.

**Status:** Design complete. Implementation ready — not yet applied to code.

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

**Status:** Fixed (2026-04-16). Training loop now logs `train_rmse[m]` and
`val_rmse[m]` (sqrt of MSE). Column headers and printed values both updated.
Scheduler still steps on `val_mse` internally (monotone — equivalent).

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
| 1 | Channel normalization | Design complete — implement next | none (can do in parallel with 2) |
| 3 | MSE vs RMSE logging | **Done** (2026-04-16) | — |
| 4 | Multi-start initialization | Not started | needs Issue 2 first |
| 5 | Local minimum diagnosis | Blocked | needs Issues 2 + 4 |
| 6 | Log constraint | Design pending | resolve Issues 2+4 first |
| 7 | Identifiability | Subsumed by 2 | fix is Issue 2 |
