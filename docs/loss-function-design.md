# Loss Function Design — Multi-Trajectory Parameter Recovery

**Date**: 2026-04-21  
**Decision entry**: D-044 in `docs/decisions.md`  
**Implementation target**: `lpv_lfr_baseline/scripts/train_param_recovery.py`

This document is the full design rationale for the loss function used in multi-trajectory
physical parameter identification. It explains every step, why it is needed, and what
literature supports it. A future session should be able to justify every choice from this
document alone.

---

## 1. Setup

We are recovering 13 physical parameters (masses, stiffnesses, dampings) of a dual-gantry
system from 6 MATLAB-simulated trajectories. The system has 3 output channels: X1, X2, Y
(all in metres). Training uses batched multiple shooting: at each epoch, 8 short segments
are sampled from the trajectories, the model is rolled out with RK4 from a cached initial
state, and the loss between predicted and true output is minimised with Adam.

**Trajectory groups and active channels:**

| ID | Group | Active channels | Dormant channels |
|----|-------|----------------|-----------------|
| T1 | y_only | Y | X1, X2 (controller suppresses to ≈ 0) |
| T6 | y_only | Y | X1, X2 (controller suppresses to ≈ 0) |
| T2 | x_sym_mh | X1, X2 | Y (fixed at 0.30 m) |
| T3 | x_sym_mh | X1, X2 | Y (fixed at 0.00 m) |
| T4 | rot_coupled | X1, X2 | Y (fixed at 0.20 m) |
| T5 | rot_coupled | X1, X2, Y | — (all active) |

---

## 2. What is a residual?

The residual is the difference between what the model predicts and what the true output is:

```
residual_c(t) = ŷ_c(t) - y_c(t)
```

For example at one timestep:
- Model predicts Y = 0.32 m, true Y = 0.30 m → residual = +0.02 m (20 mm)
- Model predicts X1 = 0.011 m, true X1 = 0.010 m → residual = +0.001 m (1 mm)

The current (flawed) loss squares and averages these directly:

```
loss = mean over all timesteps, channels, segments of residual²
```

This is naive MSE. The rest of this document explains why it is wrong and how to fix it.

---

## 3. The six problems with naive MSE

### Problem 1 — Dormant channels included in the loss

On T1/T6, X1 and X2 are actively suppressed to near-zero by the feedback controller. The
residual on these channels is tiny but it is controller tracking error, not plant dynamics.
Including it in the loss causes the optimizer to receive gradient signal from controller
suppression dynamics rather than from physical parameters — pulling parameter estimates
away from their true values.

**Why normalization cannot fix this:** the issue is not amplitude — it is that the signal
carries zero information about the physical parameters. Dividing by a small sigma makes
the noise larger in relative terms; Adam then amplifies it further through its adaptive
learning rate. Masking is the only correct solution.

### Problem 2 — Global sigma dilutes Y, inflates X

The current implementation computes sigma from all 6 trajectories concatenated:

- sigma[Y]: computed across T1, T2, T3, T4, T5, T6 — but Y is constant on T2/T3/T4,
  so those samples drag sigma[Y] toward zero → sigma[Y] is artificially small → Y is
  over-weighted in the loss.
- sigma[X1]: computed across all 6 trajectories including T1/T6 where X1 ≈ 0 → sigma[X1]
  is artificially large → X1 is under-weighted on trajectories where it is actually active.

Both biases go in opposite directions and compound.

### Problem 3 — Within-trajectory amplitude imbalance

On T5, X1/X2 and Y are both active but may have very different excursion amplitudes (e.g.
Y sweeps 300 mm while X1 moves ±10 mm). The raw MSE contribution of Y is then ~900× larger
than X1. Parameters primarily identified by X motion (m1, m2, cg1, cg2) are undertrained
relative to Y-related parameters (mh, cy).

### Problem 4 — Cross-trajectory amplitude imbalance

Trajectories with the same active channels can have very different excursion amplitudes:
T1 (conservative Y sweep) vs T6 (aggressive Y sweep). A single global sigma[Y] cannot
capture this — T6 segments always dominate T1 segments in the loss even though both are
Y-only trajectories contributing the same type of information about Y.

### Problem 5 — Inconsistent denominator across segments

Different trajectory groups have different numbers of active channels:
- T1/T6: 1 active channel
- T2/T3/T4: 2 active channels
- T5: 3 active channels

A single global denominator (e.g. 3 channels × T × B segments) gives unequal weight per
active channel-step across trajectory groups. A T5 segment contributes 3× as many
channel-steps as a T1 segment, so T5 dominates the denominator and therefore the gradient.

### Problem 6 — Adam sees inconsistent loss scale across batches

With 8 segments sampled from different trajectory groups per batch, the loss magnitude
varies depending on which groups appear. Without per-segment normalization, Adam's second
moment estimate v_t cannot stabilize, making its adaptive learning rate unreliable.

---

## 4. Two design principles

The six problems reduce to two conceptually distinct issues:

**Principle 1 — Information masking** (Problem 1)  
Channels carrying no parameter information must be excluded from the loss entirely. This is
not a weighting problem — no normalization constant can fix a signal with zero information
content. The solution is a binary mask.

**Principle 2 — Equal contribution** (Problems 2–6)  
Every contributing unit (channel, trajectory, segment) should contribute in proportion to
its typical scale, not its raw residual magnitude. Applied at three levels:
- Channel level (Problems 2 & 3): normalize by per-trajectory per-channel sigma
- Trajectory level (Problem 4): use per-trajectory sigma (not global)
- Segment level (Problems 5 & 6): average per segment before averaging over the batch

---

## 5. The solution: three steps

### Step 1 — Binary mask (fixes Problem 1)

Before anything else, look up which trajectory the segment came from. If a channel is
dormant for that trajectory, set its residual to zero. It contributes nothing to the loss.

```
mask per trajectory:
  T1, T6:          [X1=0, X2=0, Y=1]
  T2, T3, T4:      [X1=1, X2=1, Y=0]
  T5:              [X1=1, X2=1, Y=1]
```

Note: masks must be per-trajectory, not per-group. T4 and T5 are both in 'rot_coupled'
but T4 has Y fixed while T5 has Y sweeping — they need different masks.

The dormant channels remain in the RK4 rollout (mechanical coupling still acts through
them), but their residuals are excluded from the loss. This is the same principle as
Werling et al. (2025), where velocity remains in the rollout state but `W = diag(1, 0)`
sets its loss weight to zero.

**Literature:**
- Werling et al., "Trajectory-based actuator identification via differentiable simulation"
  (PDF p. 5, Eq. 2; p. 12, Appendix B): diagonal weight matrix W on output residuals;
  explicitly sets W = diag(1, 0) so velocity is in the rollout but not penalized in the
  loss. Adam optimizer confirmed.
- Gautier & Khalil (1990), IEEE T-RAS: dormant joints produce structural zeros in the
  regressor — the masking falls out of the algebraic structure in classical least-squares.
  Binary masking is the explicit gradient-based equivalent.

### Step 2 — Divide residual by per-trajectory per-channel sigma (fixes Problems 2, 3, 4)

For each trajectory individually, compute the std of each **active** channel from that
trajectory's data only. Dormant channel entries are set to 1.0 (never used — masked out):

```
sigma[T1][Y]  = std(Y on T1)         # conservative Y sweep
sigma[T6][Y]  = std(Y on T6)         # aggressive Y sweep
sigma[T2][X1] = std(X1 on T2)        # X at Y=0.30
sigma[T2][X2] = std(X2 on T2)
sigma[T3][X1] = std(X1 on T3)        # X at Y=0.00
sigma[T3][X2] = std(X2 on T3)
sigma[T4][X1] = std(X1 on T4)        # X at Y=0.20
sigma[T4][X2] = std(X2 on T4)
sigma[T5][X1] = std(X1 on T5)        # X+Y both active
sigma[T5][X2] = std(X2 on T5)
sigma[T5][Y]  = std(Y on T5)
```

Then divide the residual:
```
normalized_residual_c(t) = (ŷ_c(t) - y_c(t)) / sigma[traj_id][c]
```

This puts residuals in units of "standard deviations of that channel's typical excursion
on that specific trajectory." A 30 mm Y residual on T6 (sigma=300 mm) and a 5 mm Y
residual on T1 (sigma=50 mm) both become 0.10 — equal relative contribution regardless
of absolute excitation amplitude.

**Why per-trajectory sigma (not global, not per-channel-global):**
- Global sigma is contaminated by inactive-channel samples (Problems 2 & 3).
- Per-channel-global sigma (one sigma per channel across all active trajectories) is
  better but still fails for Problem 4: if T6 always moves more than T1, sigma[Y] is
  pulled toward T6's scale and T1 gets systematically underweighted.
- Per-trajectory sigma: each trajectory's sigma reflects its own excitation. T1 and T6
  each get normalized by their own scale → equal contribution.

**Literature:**
- Lutter et al., "Dynamic Modeling of Robotic Manipulator via an Augmented Deep Lagrangian
  Network" (PDF p. 4, Eq. 8): Mahalanobis norm with diagonal covariance matrix W_τ; "It
  is necessary to normalize the loss function using covariance matrix since the torque
  magnitude may vary greatly from joint to joint."
- Lutter et al., "Combining Physics and Deep Learning to learn Continuous-Time Dynamics
  Models" (IJRR; PDF p. 7, Eq. 12): same Mahalanobis norm; "It is beneficial to normalize
  the loss using the covariance matrix because magnitude of the residual might vary between
  different joints."
- Fuentes-Silva et al., "Constrained Gray-Box Identification of Electromechanical Systems
  Under Unfiltered Step-Response Data" (PDF pp. 6–7, Eq. 3): trajectory errors normalized
  by RMS of corresponding measured signal; "naturally balances the relative contribution
  of current and velocity."
- Problem 4 (cross-trajectory): no robotics paper prescribes per-trajectory sigma exactly.
  Supported by the principle in Zhang et al. (Int. J. Solids Struct., 2023,
  doi:10.1016/j.ijsolstr.2023.112534): "maintaining equal contribution of the strain states
  from each experiment to the cost function." Per-trajectory sigma is an engineering
  realization of this principle. Also supported by Neggers et al. (Mech. Mater., 2019,
  doi:10.1016/j.mechmat.2019.03.001): experiment-wise weighting from Bayesian uncertainty.

**Forward compatibility:** when moving to noisy hardware data, per-trajectory sigma
transitions directly to the principled Λ⁻¹ weighting (Ljung 1999, §7.4, Eq. 7.27). At
high SNR (gantry encoders: signal mm–cm, noise µm), signal std ≈ noise-floor-independent
scale. No architectural change needed at the transition; only the interpretation of sigma
changes (signal std → noise std).

### Step 3 — Average per segment, then average over batch (fixes Problems 5, 6)

After masking and normalizing, compute the mean squared error over **active channel-steps
for that segment only**, then average segment losses over the batch.

The key variable here is **n_active** — the number of active channels — which differs
between trajectory groups (1, 2, or 3). T is fixed (equal segment length).

**Without per-segment averaging (global denominator):**
```
loss = total_masked_normalized_sq / (sum of n_active_i × T over all segments)
```
A T5 segment contributes 3×T to the denominator; a T1 segment contributes 1×T. T5 is
3× heavier — it dominates the gradient purely because it has more active channels, not
because it is more informative.

**With per-segment averaging:**
```
seg_loss_i = masked_normalized_err_i² .sum() / (n_active_i × T)   → O(1) per segment
loss = mean(seg_loss_1, ..., seg_loss_8)
```
Each segment contributes equally regardless of how many active channels it has.

**Why this matters for Adam:**
Adam maintains a second moment estimate v_t = E[g_t²] per parameter. For this estimate to
stabilize, the loss must be approximately O(1) across all batches regardless of which
trajectory groups appear. Per-segment averaging guarantees this — every batch produces
loss ≈ O(1), and the moment estimates converge correctly.

**Literature:**
- Werling et al. (above), Eq. 2: `L_batch = (1/MN) Σ_j Σ_i ‖W(s'_{i,j} − s_{i,j})‖²` —
  explicitly summed over M segments and N timesteps, giving equal weight per segment.
  "All identified models minimize the segmented trajectory-matching objective ... using
  Adam." This is the gradient-based SysID precedent for the per-segment structure.

---

## 6. The full loss function

Combining all three steps:

$$L = \frac{1}{B} \sum_{i=1}^{B} \frac{1}{n_{\text{active},i} \cdot T}
      \sum_{c=1}^{3} \sum_{t=1}^{T}
      m_{\text{traj}_i, c}
      \left( \frac{\hat{y}_c(t) - y_c(t)}{\sigma_{\text{traj}_i, c}} \right)^2$$

where:
- B = 8 (segments per batch)
- T = segment_len (fixed, equal for all segments)
- m_{traj,c} ∈ {0, 1}: binary mask for channel c on trajectory traj
- σ_{traj,c}: std of channel c computed from trajectory traj, active samples only
- n_active,i = Σ_c m_{traj_i, c}: number of active channels for segment i

This is still MSE — applied to residuals that have been masked and normalized. Everything
else (Adam, log-space reparameterization, RK4, param_loss regularization) is unchanged.

---

## 7. What Adam sees after this change

Adam's gradient for each parameter is:

$$\frac{\partial L}{\partial \theta} = \frac{1}{B} \sum_i \frac{2}{n_{\text{active},i} \cdot T}
  \sum_c \sum_t m_{\text{traj}_i, c} \cdot
  \frac{\hat{y}_c - y_c}{\sigma_{\text{traj}_i, c}^2} \cdot
  \frac{\partial \hat{y}_c}{\partial \theta}$$

σ²_{traj,c} appears in the denominator — each channel's gradient contribution is scaled
by 1/σ²_{traj,c}. This is equivalent to Ljung's Λ⁻¹ weighting with Λ_c = σ²_{traj,c}.

After normalization, the remaining gradient variation between segments comes from the
sensitivity term ∂ŷ_c/∂θ, which reflects how much each trajectory actually excites each
parameter. This is informative variation — more aggressively excited trajectories
legitimately contribute more gradient for the parameters they identify. The sigma
normalization removes only the spurious amplitude bias.

| Source of gradient variation | Before fix | After fix |
|------------------------------|-----------|-----------|
| Raw signal amplitude (T1 vs T6) | Corrupts | Removed by sigma |
| Number of active channels (T1 vs T5) | Corrupts | Removed by per-segment avg |
| Parameter sensitivity ∂ŷ/∂θ | Informative | Preserved |

---

## 8. Implementation outline

Six targeted changes to `train_param_recovery.py`:

**Change 1**: Add `CHANNEL_MASKS` dict in the config section. Per-trajectory (not
per-group) because T4 and T5 are both 'rot_coupled' but have different active channels.

**Change 2**: Replace `_get_or_compute_sigma` to return a dict {traj_id: (3,) tensor}
instead of a single (3,) tensor. Each entry holds the per-active-channel std for that
trajectory; dormant channels are set to 1.0 (masked out anyway). Bump SIGMA_CACHE_VERSION.

**Change 3**: Update sigma display in Step 1 to print a per-trajectory table.

**Change 4**: Capture `sample_plan` from `_sample_balanced_segments` (currently discarded
as `_`). No change to `_sample_balanced_segments` itself — it already returns traj_id
per segment.

**Change 5**: Replace the loss computation (currently 2 lines) with a per-segment loop:
```python
seg_losses = []
for i, plan in enumerate(sample_plan):
    mask_i = channel_masks[plan['traj_id']]    # (3,) on device
    sigma_i = sigma[plan['traj_id']]           # (3,) on device
    err_i = (Y_pred[i] - q1_seg[i]) / sigma_i # (T, 3)
    err_i = err_i * mask_i                    # (T, 3) — dormant = 0
    n_active = mask_i.sum()
    seg_losses.append(err_i.pow(2).sum() / (n_active * segment_len))
mse_loss = torch.stack(seg_losses).mean()
```

**Change 6**: Update `_aggregate_normalized_rmse_baseline` to use `sigma[entry['id']]`
instead of global sigma, keeping Lambda regularization calibrated in the same unit system
as the new loss.

**What does NOT change**: `_sample_balanced_segments`, `simulate`, `_SimWrapper`,
`wrapper`, `loss.backward()`, `optimizer.step()`, `scheduler.step()`, checkpointing,
saving, `_print_param_detail`, Step 5 eval (which stays in metres for human readability).

---

## 9. Verification after implementation

Before running full training, verify:

1. **Print sigma table**: confirm no near-zero sigmas on active channels and no inflated
   sigmas where channels are dormant. Expected order of magnitude: X1/X2 ≈ 1–50 mm,
   Y ≈ 10–300 mm depending on trajectory.

2. **Print seg_loss values for one batch**: all should be O(1) (roughly 0.1–10) regardless
   of which trajectory group the segment came from. If T5 segments give 10× larger
   seg_loss than T1 segments, the normalization is not working.

3. **Check n_active per segment**: confirm T1/T6 → 1, T2/T3/T4 → 2, T5 → 3.

4. **Compare training curves**: with the fix, convergence should be smoother and
   per-parameter convergence should be more uniform across parameter types.

---

## 10. Key sources

| Source | What it supports |
|--------|-----------------|
| Werling et al., "Trajectory-based actuator identification via differentiable simulation" (arXiv:2604.10351) | Binary masking (W=diag(1,0)); per-segment objective; Adam. PDF p.5 Eq.2, p.12 App.B |
| Lutter et al., "Dynamic Modeling ... Augmented Deep Lagrangian Network" | Mahalanobis norm with diagonal covariance; "torque magnitude may vary greatly from joint to joint." PDF p.4 Eq.8 |
| Lutter et al., "Combining Physics and Deep Learning ... Continuous-Time Dynamics" (IJRR) | Same Mahalanobis norm; "magnitude of the residual might vary between different joints." PDF p.7 Eq.12 |
| Fuentes-Silva et al., "Constrained Gray-Box Identification of Electromechanical Systems" | RMS normalization per signal type; "naturally balances the relative contribution." PDF pp.6–7 Eq.3 |
| Zhang et al., Int. J. Solids Struct. 2023, doi:10.1016/j.ijsolstr.2023.112534 | "Equal contribution of strain states from each experiment to the cost function" — principle for cross-experiment balancing |
| Neggers et al., Mech. Mater. 2019, doi:10.1016/j.mechmat.2019.03.001 | Bayesian experiment-wise weighting — citable basis for balancing across experiments |
| Ljung, "System Identification: Theory for the User," 2nd ed. (1999), §7.4 Eq. 7.27 | Λ⁻¹ weighting of multi-output prediction errors — classical PEM justification; per-trajectory sigma is the high-SNR approximation |
| Gautier & Khalil (1990), IEEE T-RAS | Dormant joints produce structural zeros in regressor — classical analog of binary masking |
| Houska, Logist, Diehl & Van Impe (2011), tutorial | Multiple shooting with covariance-scaled residuals; extra scaling beyond noise model is "heuristic" — honest framing for noiseless case |
