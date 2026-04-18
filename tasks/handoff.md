# Session Handoff

_Previous sessions archived to `archive/sessions/`._

**Last written**: 2026-04-17 by Claude (Sonnet 4.6)

---

## Trajectory generation & supervisor feedback (2026-04-17)

### What `export_lpv_multi_traj.m` actually does

The 6-trajectory script generates **reference positions** `r = [X1_ref, X2_ref, Y_ref]`
in stage coordinates. These are NOT inputs to the actuators. The actual actuator inputs
are **force commands** produced by the closed-loop controller:

```
u = Cfb*(r - q1)      ← force commands [F_X1, F_X2, F_Y] — what the actuators receive
```

`[X1, X2, Y]` are the **output measurements**, not the inputs. The forces `u_q1` are
reconstructed post-hoc via `lsim(Cfb, r - q1, t)` and saved alongside `q1`.

**Correct framing (for supervisor):** "We design reference position trajectories in
stage coordinates. A feedback controller converts tracking errors to force commands
`[F_X1, F_X2, F_Y]` — the actual inputs to the three actuators. `[X1, X2, Y]` are
the measured position outputs."

### Feedback controller — single operating point for all 6 trajectories

`Cfb` is designed at `Y_op = 0.3` (frozen M(Y=0.3)) and **is the same for all 6
trajectories**. This is a known imperfection:

| Trajectory | Y during main motion | Mismatch vs Y_op=0.3 |
|---|---|---|
| T1, T2, T6 | starts at 0.3 | none at start |
| T3 | held at 0.0 m | moderate — M[0,1] changes sign |
| T4 | held at 0.2 m | small |
| T5 | sweeps 0.2 → −0.2 m | continuously varying |

The quasi-LPV `q1` path self-schedules M(Y) correctly (gantrySystem.m reads Y=x(3)
at every ODE step), so the **plant dynamics are correct**. The controller is off-
design-point for T3/T4/T5 but should remain stable at fbw=100 Hz across the full
Y range. Performance degrades but identifiability is not broken.

The post-hoc force reconstruction `u_q1 = lsim(Cfb, r-q1, t)` is mathematically
exact for the current setup (same Cfb, f=0).

### Multisine injection — not yet implemented

**Current state:** `f = zeros(size(r))` for all 6 trajectories — no excitation signal.

**No Simulink changes needed.** The feedforward slot is already wired in the Simulink
model — `f` is read from the workspace via a FromWorkspace block and summed with the
controller output (from main.m: `S2 = sumblk("u = ufb + f", n)`):
```matlab
% Currently in export_lpv_multi_traj.m:
f = zeros(size(r));   % ← replace this with multisine signal
sim(mdl, t(end));     % Simulink picks up f automatically
```

**Injection point (supervisor):** Multisine goes **at the feedforward slot**, after
the controller, directly at the actuator force level:
```
r ──► [Cfb] ──► (+) ──► [Plant G(Y)] ──► q1 = [X1, X2, Y]
               ▲   ▲
               │   └── f_multisine   ← injected here
               └─────────────────────── feedback (r - q1)

u_total = Cfb*(r - q1) + f_multisine     [F_X1, F_X2, F_Y]
```
The reference `r` sets the nominal operating point; `f_multisine` provides broadband
identification excitation directly at the plant input, independent of the closed-loop
sensitivity function.

---

### Optimal multisine design — decided

The sysid lecture notes (`sysid-experiment-design-notes.md`) directly support the
supervisor's approach and resolve all open design choices.

**Theoretical justification:**
The parameter covariance scales inversely with input power:
```
V(θ) ∝ ∫ |G_0(ω) - G(ω,θ)|² · Φ_u(ω) dω
```
Maximising `Φ_u(ω)` minimises parameter uncertainty. In simulation (zero cost), the
optimal strategy is: push amplitude as high as possible, then filter on hardware limits.
Schroeder phases maximise RMS for a given peak force (crest factor 1.58 vs 22.3 for
linear phases), so more amplitude passes the ETEL filter for the same peak constraint.

---

#### Step 1 — Multisine structure (identical for all 6 trajectories)

| Parameter | Value | Reason |
|---|---|---|
| Frequency range | **1–200 Hz** | Covers dynamics; 2× controller bandwidth (100 Hz); well below Nyquist (10 kHz) |
| Period length | **T_p = 1 s = 20,000 samples** | Δf = 1 Hz; multisine frequencies are integer multiples of Δf → zero leakage |
| Frequency lines | 200 (one per integer Hz from 1–200) | PE order = 400 ≫ 13 parameters |
| Phases | **Schroeder:** φ_n = −n(n−1)π/F | Crest factor 1.58 — minimum peak force for given RMS |
| Channels | **3 independent signals** (different seeds per channel) | Ensures Φ_u(ω) ≻ 0 (MIMO PE condition); inputs not linearly correlated |
| Harmonics | All (not odd-only) | Parameter recovery goal; odd-only needed only for nonlinearity detection |

**Schroeder phase MATLAB implementation:**
```matlab
function sig = multisine_schroeder(N, fs, f_low_hz, f_high_hz, amp_rms)
    freq_lines = f_low_hz : fs/N : f_high_hz;   % integer Hz lines, Δf = fs/N
    F          = length(freq_lines);
    phi        = -((1:F) .* (0:F-1)) * pi / F;  % Schroeder phases
    t          = (0:N-1)' / fs;
    sig        = zeros(N, 1);
    for k = 1:F
        sig = sig + cos(2*pi*freq_lines(k)*t + phi(k));
    end
    sig = amp_rms * sig / rms(sig);             % normalise to desired RMS [N]
end
```
Call 3× with different random offsets added to `phi` to get independent channels.

#### Step 2 — Tile over full trajectory duration

The trajectory is longer than one period. Tile to fill:
```matlab
n_tile   = ceil(size(r,1) / N_period);
f_tiled  = repmat(f_one_period, n_tile, 1);
f        = f_tiled(1:size(r,1), :);   % (N_traj × 3)
```
Parameter recovery uses the full `(u_total, q1)` time series — no per-period averaging
needed for time-domain gradient-based fitting.

#### Step 3 — Amplitude sweep and ETEL filter

Run simulation at increasing RMS amplitudes. Filter on the **actual simulated `q1`**
(not on the reference `r`) — the closed loop shapes the response and may amplify
certain frequencies.

```matlab
amp_rms_grid = [1, 2, 5, 10, 20, 50, 100, 200];  % [N] RMS per channel

amp_max = 0;
for amp = amp_rms_grid
    for ch = 1:3
        f(:,ch) = multisine_schroeder(size(r,1), fs, 1, 200, amp);
    end

    sim(mdl, t(end));          % produces q1 in workspace

    vel = diff(q1) * fs;       % (N-1 × 3) velocity  [m/s]
    acc = diff(vel) * fs;      % (N-2 × 3) acceleration [m/s²]

    ok =    max(abs(q1(:,1)))            <= 0.375  ...  % X1 position [m]
         && max(abs(q1(:,2)))            <= 0.375  ...  % X2 position [m]
         && max(abs(q1(:,3)))            <= 0.400  ...  % Y  position [m]
         && max(abs(q1(:,1)-q1(:,2)))   <= 0.100  ...  % |X1-X2| differential [m]
         && max(abs(vel(:)))             <= 2.0    ...  % all axes velocity [m/s]
         && max(abs(acc(:)))             <= 50.0;       % all axes acceleration [m/s²]

    if ok
        amp_max = amp;
    else
        break;   % first failure: stop, use amp_max
    end
end
fprintf('  Trajectory %s: amp_max = %.1f N RMS\n', sp.id, amp_max);
```

#### Step 4 — Per-trajectory amplitude differs (expected and correct)

The nominal motion already uses part of the actuator headroom. The filter naturally
gives different `amp_max` per trajectory:

| Trajectory | Nominal motion headroom | Expected amp_max |
|---|---|---|
| T4 (anti-sym, small X) | High — small nominal forces | Highest multisine amplitude |
| T3 (X-sym at Y=0) | Medium | Medium |
| T2 (X-sym at Y=0.3) | Medium | Medium |
| T1 (Y sweep, conservative) | Medium-high | Medium |
| T5 (X+Y combined) | Lower | Lower |
| T6 (Y sweep, hardware max) | Low — already near accel limit | Lowest multisine amplitude |

This is correct: the filter is doing exactly what optimal experiment design prescribes —
maximum excitation per trajectory subject to physical realizability.

#### Step 5 — What to save

The full actuator input is `u_total = u_q1 + f_multisine`. Both must be saved:
```matlab
save(out_path, 't_sim', 'fs', 'r_sim', ...
     'u_q1',        ...   % Cfb*(r-q1) — feedback forces
     'f_multisine', ...   % identification excitation
     'q1', 'q_simscape', 'Y_trajectory', 'amp_max');
```
Parameter recovery uses `u_total = u_q1 + f_multisine` as the model input and `q1`
as the target output.

---

### Sysid theory — key pitfalls for our setup

Full reference: `literature/experiment-design/System-identification/sysid-experiment-design-notes.md`

The most critical for our multisine design:

| Pitfall | Our mitigation |
|---|---|
| **Leakage** — non-integer periods corrupt all FRF estimates | Δf = fs/N = 1 Hz; frequencies are exact integer Hz; T_p tiles exactly |
| **Actuator saturation** — high crest factor → nonlinear response | Schroeder phases: CF = 1.58; amplitude filtered on actual q1 |
| **MIMO inputs correlated** — singular Φ_u(ω) | 3 independent channels (different Schroeder seeds) |
| **Loss of excitation** — Cfb attenuates multisine at some frequencies | Filter checks plant response q1, not reference; amplitudes are at plant input |
| **Transient not removed** — biases FRF at resonances | Not critical for time-domain parameter recovery; but discard first period if doing FRF analysis |
| **Closed-loop residual misread** — R̂_eu(τ≠0) seen as model error | Expected for τ<0 due to feedback; not a model failure |

**Theoretical backing for max-excitation + filter approach (Lecture 9):**
> "Minimize identification cost while maximising information content."
In simulation (cost=0): maximise Φ_u(ω) subject to ETEL constraints. Schroeder phases
maximise RMS for given peak → more amplitude passes the filter.

---

### Hoekstra comparison

| | Hoekstra (MSD) | Our gantry |
|---|---|---|
| Multisine role | Only input (open loop) | Feedforward on top of closed-loop Cfb |
| Language | Python (`deepSI.exp_design.multisine()`) | MATLAB (`multisine_schroeder`) |
| Channels | SISO | MIMO 3×3 — independent per channel |
| Amplitude | Fixed `amp_scale=10` | Sweep per trajectory, filter on ETEL limits |
| Phases | Random + crest factor optim | Schroeder (deterministic optimal) |
| Noise | Commented out | Not yet added to q1 |

### Current `validate_ref()` is incomplete

The current filter only checks **reference positions**. It does not check velocity
or acceleration of the actual simulated response. Missing checks:

| Quantity | ETEL limit | Currently checked? |
|---|---|---|
| X1, X2 position | ±375 mm | Yes (reference only) |
| Y position | ±400 mm | Yes (reference only) |
| \|X1−X2\| differential | ≤ 100 mm | Yes (reference only) |
| X1, X2 velocity | ≤ 2 m/s | **No** |
| Y velocity | ≤ 2 m/s | **No** |
| X1, X2 acceleration | ≤ 50 m/s² | **No** |
| Y acceleration | ≤ 50 m/s² | **No** |

Velocity and acceleration must be computed from `q1` via finite differences on the
simulated output (not from the reference profile).

### No measurement noise yet

All 6 trajectories are currently noise-free. This gives overly optimistic parameter
recovery results. Realistic position measurement noise needs to be added to `q1`
before evaluating identifiability seriously.

### Open items from supervisor meeting

- [ ] Implement multisine injection at `f` slot in `export_lpv_multi_traj.m`
- [ ] Add velocity and acceleration checks to the trajectory filter (on `q1`, not `r`)
- [ ] Add measurement noise to `q1` outputs
- [ ] Check MUSSV / LFR well-posedness: verify `σ_max(Dzw) < 1/max|δ(Y)|` over
      operating range — ensures `I - Dzw·δ(Y)` is non-singular for all Y in [-0.3, 0.3]
- [ ] Kinematic transformation `[X, Theta, Y] ↔ [X1, X2, Y]` (P matrix) is a
      linearized small-angle approximation — accuracy cannot be verified from
      simulation data alone; requires additional sensor or measurement

---

## LPV-LFR structural review (2026-04-17)

Reviewed `lfr_forward.py`, `lfr_matrices.py`, `lfr_block.py`, `lfr_param_block.py`.

**The implementation is genuinely LFR.** Key findings:

**Loop solve is analytically pre-solved, not bypassed:**
The formal loop `(I - Dzw·Δ(Y))·z = Cz·x + Dzu·u` is solved analytically via Cramer's rule.
`N0, N1, N2` (adjugate polynomial coefficients) and `dY` (determinant polynomial) are exactly
`L(Y)⁻¹·rhs` pre-derived in closed form — not a collapse. `Cz`, `Dzw`, `Dzu` are baked into
this derivation; they're absent as runtime variables, not absent from the computation.

**Output equation is LFR-structured:**
`xdot = Ax@x + Bw@w + Bu@u` — w is causally upstream of xdot through G.Bw.
The decisive structural audit (Check 4 in `lfr_forward.py`) passes: `d(xdot)/d(w) = G.Bw ≠ 0`.

**G is properly rebuilt for the trainable block:**
`lfr_param_block.py` calls `build_G_matrix()` and `build_poly_constants()` inside every
`forward()` — gradient flows correctly back to `log_params` (nn.Parameter). ✅
`lfr_block.py` (non-trainable) precomputes G at `__init__` and stores as buffers — correct. ✅

**Open question for supervisor (Roland Tóth):**
The loop solve uses the analytical pre-solved form of `L(Y)⁻¹·rhs` rather than materializing
`L(Y)` at runtime. When augmentation extends G (new rows/columns in Cz, Dzw, Dzu), the
analytical loop solve must also be re-derived or replaced by a numerical solve of the augmented
`L(Y)·z = rhs`. Ask: is this the expected path, or is there a clean way to extend the
analytical solution to the augmented system?

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
