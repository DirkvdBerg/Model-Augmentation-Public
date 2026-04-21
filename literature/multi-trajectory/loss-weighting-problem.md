# Loss Weighting Problem — Multi-Trajectory Parameter Recovery

**Date**: 2026-04-21  
**Status**: Open — no complete literature solution found  
**Related decisions**: D-034, D-042

---

## 1. System description

The training script (`lpv_lfr_baseline/scripts/train_param_recovery.py`) recovers 13 physical
parameters (masses, stiffnesses, dampings) of a dual-gantry quasi-LPV mechanical system from
multiple MATLAB-simulated trajectories. The system has 3 output channels — X1, X2, Y — all in
metres. Data is **noiseless** (clean MATLAB simulation). A **feedback controller is active**
during all trajectories.

Training uses batched multiple shooting: short segments are sampled from the trajectories, the
model is rolled out with RK4 from a cached initial state, and the MSE between predicted and
true output is minimised using Adam with log-space parameter reparameterisation.

---

## 2. Trajectory structure

Six trajectories are used, organised in three groups:

| ID | Group | Excitation | Y behaviour |
|----|-------|-----------|-------------|
| T1 | y_only | Y sweeps conservatively | Y active |
| T6 | y_only | Y sweeps aggressively | Y active |
| T2 | x_sym_mh | X symmetric, Y = 0.30 m | Y fixed |
| T3 | x_sym_mh | X symmetric, Y = 0.00 m | Y fixed |
| T4 | rot_coupled | X antisymmetric, Y = 0.20 m | Y fixed |
| T5 | rot_coupled | X symmetric, Y sweeps | X and Y both active |

The trajectories are **designed experiments**: each group targets a different subset of
physical parameters and deliberately suppresses the non-target DOFs via the feedback
controller.

---

## 3. The core problem

When segments from different trajectory groups appear in the same mini-batch, the MSE loss
must handle two distinct but related issues:

### Problem 1 — Dormant channels

On a Y-only trajectory (T1, T6), the feedback controller actively suppresses X1 and X2 to
near-zero. The residual X1/X2 motion is controller tracking error, not plant physics. It
carries no meaningful parameter information.

If dormant channels are included in the loss with equal weight:
- The optimizer receives gradient signal from controller dynamics, not plant physics
- Adam's adaptive learning rate amplifies tiny, noisy controller-residual errors because the
  second moment estimate shrinks with the small gradient magnitude
- Physical parameters are pulled toward values that explain controller suppression rather than
  plant dynamics — systematic bias
- The log-space reparameterisation exponentially scales gradients, making the optimizer
  hypersensitive to microscopic dormant-channel errors

**Why this is hard to normalise away**: no normalization constant can fix this. The problem
is not amplitude — it is that the signal carries no parameter information. Dividing by sigma
makes the noise smaller in relative terms but does not remove it; Adam still amplifies it.

### Problem 2 — Active channel amplitude differences

On T5 (X and Y both active), X1/X2 and Y may have very different excursion amplitudes. If Y
sweeps 300 mm while X moves ±10 mm, the Y MSE contribution is ~900× larger than X1/X2. The
optimizer focuses almost entirely on Y-related parameters (mh, cy) and underweights X-related
parameters (m1, m2, cg1, cg2).

This is a gradient conditioning problem, not an information problem. Both channels carry
genuine parameter information — the issue is that one channel dominates purely because of
signal amplitude, not because it is more informative.

---

## 4. Approaches considered

### 4.1 Global signal sigma (original implementation — rejected)

Compute per-channel std over all 6 trajectories concatenated. Divide loss by sigma².

**Problem**: sigma[X1] is large because it is pulled up by T2/T3/T4/T5. On T1 (Y-only),
dividing the tiny X1 controller residual by the large sigma[X1] makes it appear negligible —
but the controller residual was already near-zero. The normalization does not fix the
information problem; it just rescales the noise. Furthermore, sigma[Y] computed from all
trajectories including T2/T3/T4 (where Y is fixed) is diluted by constant-Y samples, making
sigma[Y] smaller than the natural Y excursion scale.

**Verdict**: conflates signal amplitude with data reliability. Does not solve Problem 1.
Documented in D-042.

### 4.2 Per-trajectory sigma (rejected)

Compute sigma per channel from each trajectory individually. Use trajectory-specific sigma
for segments from that trajectory.

**Problem 1**: on T1, sigma[X1] ≈ std(controller residual) ≈ micrometres. Dividing by this
amplifies controller noise by ~10⁶. Makes Problem 1 dramatically worse.

**Problem 2**: different segments in the same batch have different normalization scales.
Adam's momentum estimates are built from heterogeneous loss scales → gradient direction is
corrupted across batches. Not Adam-consistent.

**Verdict**: rejected on both theoretical and practical grounds.

### 4.3 FIM-based weighting (proposed — partially applicable)

At initialization, compute the per-channel output sensitivity norm:

$$s_c = \sum_t \left\|\frac{\partial \hat{y}_c(t,\theta)}{\partial \theta}\right\|^2$$

Use s_c as a fixed, global weight per channel per trajectory.

**Problem 1 — dormant channels**: FIM handles this cleanly. On a Y-only trajectory, the
sensitivity of X1 to any parameter is near zero because the feedback controller cancels the
response before it manifests. Therefore s_c[X1] ≈ 0 → weight → 0 automatically. No manual
mask required; the physics provides the masking.

**Problem 2 — active channel amplitude differences**: FIM does NOT directly solve this. FIM
weights by parameter sensitivity ∂ŷ/∂θ, not by signal amplitude. If Y has 30× larger
excursion than X1 on T5, the sensitivity might also be larger for Y — or it might not be,
depending on the physics. FIM does not guarantee equal amplitude contribution across active
channels.

**Practical limitation**: the specific use of s_c as a fixed loss weight is not an
established method in the system identification or optimal control literature. Gemini Deep
Research explicitly flagged it as "unverified inference — an innovative heuristic bridge".
FIM is well-established for experiment design (D-optimal, A-optimal), not for loss weighting
from already-collected data.

**Computational limitation**: computing the full Jacobian ∂ŷ/∂θ through an RK4 rollout in
PyTorch reverse-mode requires storing the full computation graph — memory-intensive for long
segments. Forward-mode AD (JAX jacfwd) would be more efficient but requires switching
framework.

### 4.4 Binary masks + global sigma (current best candidate)

Assign a binary mask per trajectory based on experimental design:

| Trajectory | X1 | X2 | Y |
|------------|----|----|---|
| T1, T6 (y_only) | 0 | 0 | 1 |
| T2, T3, T4 (X-only) | 1 | 1 | 0 |
| T5 (X + Y) | 1 | 1 | 1 |

Apply global sigma normalization only to active (unmasked) channels. Average the loss over
active channel-steps only:

```python
err        = (Y_pred - q1_seg) / sigma   # global sigma, per channel
err_masked = err * mask_seg              # zero dormant channels
mse_loss   = err_masked.pow(2).sum() / (mask_seg.sum() * segment_len)
```

**Solves Problem 1**: dormant channels are zeroed regardless of their amplitude or sigma.
The earlier sigma-contamination problem (sigma[X1] large but X1 dormant on T1) disappears
because the mask zeroes X1 before sigma matters.

**Partially solves Problem 2**: global sigma normalises amplitude differences between active
channels. However, sigma[Y] is still computed from all trajectories including T2/T3/T4 where
Y is fixed — those constant-Y samples dilute sigma[Y]. A cleaner variant computes sigma[c]
only from trajectories where channel c is active.

**Limitation**: binary mask is a manual decision based on trajectory design. It does not
adapt automatically if trajectory structure changes (e.g., multisine added that partially
excites dormant channels). It also requires knowing the excitation structure in advance.

---

## 5. Literature gap

This specific problem — multi-trajectory physical parameter recovery with designed
heterogeneous excitation, closed-loop feedback, noiseless simulation, gradient-based
optimization — does not appear as a unified published problem in any single field.

Individual components are covered:

| Aspect | Field | Key references |
|--------|-------|---------------|
| Dormant channel bias in closed-loop ID | Control / System ID | Forssell & Ljung (1999) |
| Multi-experiment stacking with regressor | Robotics ID | Gautier & Khalil (1990), Swevers et al. (1997) |
| OED / FIM for experiment design | System ID | Goodwin & Payne (1977) |
| Loss scaling in NLP / multiple shooting | Optimal control | Bock & Plitt (1984) |
| Per-task loss weighting | ML / multi-task | Kendall et al. (2018) |
| Closed-loop identification | System ID | Forssell & Ljung (1999), Van den Hof & Schrama (1993) |

The robotics inertial parameter identification literature (regressor matrix approach) is the
closest analog: designed exciting trajectories, multiple experiments, physical parameter
recovery, identifiability analysis. In that literature, dormant joints produce zero rows in
the regressor — the masking falls out of the mathematical structure rather than being
explicitly applied. The normalisation question (amplitude differences between joint torques)
is handled by weighting the regressor by the inverse measurement noise covariance — which
applies to noisy physical measurements, not noiseless simulation.

**The specific gap**: noiseless simulation removes the statistical anchor (noise variance)
that the robotics and system identification literature uses to justify weighting choices. No
published method addresses this combination directly.

---

## 6. Open questions

1. What is the actual amplitude ratio between X and Y channels on T5? If it is within
   a factor of 2–3, no amplitude normalisation is needed and binary masks alone suffice.

2. Is binary masking with per-active-channel sigma (computed only from trajectories where
   each channel is active) the right practical choice? This is an engineering judgment, not
   a literature citation.

3. If a multisine is added to the trajectories in future experiments, binary masks become
   stale (dormant channels receive partial excitation). At that point, FIM-based continuous
   weights (or the regressor rank approach from robotics ID) would be more appropriate.

4. For augmentation training, the latent states (e.g. friction states z₁, z₂) are not
   directly observable. The sensitivity structure changes. The masking approach would need
   to be re-evaluated for the augmented system.

---

## 7. Current implementation status

The training script currently uses global sigma (Option A) — the approach identified as
conceptually flawed in D-042. The decision to replace it with binary masks + per-active-channel
sigma (or binary masks alone if amplitude ratios are acceptable) is pending verification of
the T5 amplitude ratio.

D-042 documents the sigma normalization design. A new decision entry is needed once the
replacement approach is finalised.
