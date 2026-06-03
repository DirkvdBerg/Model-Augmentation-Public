# Gantry SubNet Augmentation — Implementation Plan

---

## Current Status (2026-06-02)

| Item | State |
|------|-------|
| Phase 1 — frozen Y, no augmentation | DONE. NRMS verified near zero. |
| `diagnose_dynamic_parallel.py` | DONE. 15/15 tests pass. |
| `gantry_interconnect_dynamic.py` | Written. Frozen Y=0.3, NX_ANN=2. Evaluation section cleaned (no zero-state, no train rollout). Not yet run on MATLAB augmented data. |
| `generate_gantry_lti_baseline.m` | Written. One trajectory, Y frozen at 0.3 m. |
| `generate_gantry_lti_augmented.m` | Written. Known bug: addpath missing for `gantry_additional_state_2025a.slx`. Y reference constant at 0.3 — MSD may not be excited. |
| Phase 2 (LPV, no augmentation) | **Deferred.** Will do LPV later, after validating augmentation at frozen Y. |
| Phase 3 (LPV + dynamic parallel ANN) | Blocked on data. |

---

## Immediate Next Steps — Data Generation and MSD Visibility

**Goal:** generate augmented gantry data at Y=0.3 with small Y perturbations, and verify the hidden MSD shows up in the output signal before training.

### Step 1 — Fix `generate_gantry_lti_augmented.m` addpath

The script calls `sim('gantry_additional_state_2025a')` but never adds `Matlab-scripts/Augmentation` to the path. Add:
```matlab
addpath(fullfile(pwd, 'Matlab-scripts', 'Augmentation'))
```
after the existing `addpath(genpath(...))` line.

### Step 2 — Add small Y perturbations to both scripts

With Y reference constant at 0.3 m, Y acceleration is near zero and the 400 Hz MSD is not reliably excited. Add a slow sinusoidal Y perturbation to `make_ref()` in both `generate_gantry_lti_baseline.m` and `generate_gantry_lti_augmented.m`:

- Amplitude: 5-10 mm (keeps Y variation small, LTI frozen-at-0.3 approximation still holds)
- Frequency: 0.5-2 Hz (well within controller bandwidth; generates Y acceleration without hitting limits)
- Reference: `r_Y = Y_initial + A_Y * sin(2*pi*f_Y * t)` added to the constant hold

Both scripts must use identical Y perturbation parameters so the baseline vs augmented comparison is fair (same input, different plant).

### Step 3 — Run both scripts and verify MSD appears

Run `generate_gantry_lti_baseline.m` and `generate_gantry_lti_augmented.m`. Check:
- `delta_a` amplitude in augmented data (`max(abs(delta_a))` printed by the script)
- Y-channel difference: `max(abs(y_aug(:,3) - y_baseline(:,3)))` — if this is above numerical noise, MSD is visible in output
- Plot: `y_aug(:,3) - y_baseline(:,3)` vs time; expect oscillation near 400 Hz

If `delta_a` is negligible (sub-micrometre): increase Y perturbation amplitude or frequency.

### Step 4 — Load augmented data into Python and run a smoke test

Load `gantry_aug_train.mat`, verify shapes, check `std_x[2]` is non-negligible with Y moving, and run one forward pass of the interconnect. This confirms the Python pipeline accepts the new data before starting training.

### Step 5 — Train Phase 3 model

Run `gantry_interconnect_dynamic.py` with augmented data. Check loss curve and val NRMS.

---

## End Goal

The full research target — not yet implemented, but every design choice must keep
the path to it open:

1. **Joint estimation** — baseline physical parameters (`mb, mh, cg1, ...`) and ANN
   augmentation weights trained simultaneously, with `param_loss()` regularization
   preventing physics drift (following `Parameterized_MSD_State_Block` pattern)
2. **Orthogonality constraint** — ANN augmentation is penalised for learning dynamics
   already captured by the baseline, so it only corrects genuine model error
3. **LPV self-scheduling** — `Gantry_State_Block` with `Y_op=None`, M(Y) updated every
   step from the current state `x[2]`
4. **LFR signal routing** — LFR latent variables z and w routed explicitly through the
   Interconnect so the ANN can target specific physical channels

**Current implementation covers:** LFR-structured RK4 block at one frozen Y (z and w
computed explicitly but not yet routed through the Interconnect), basic parallel ANN
augmentation, no orthogonality, no joint estimation. This is intentional —
validate the pipeline before adding routing complexity.

---

## Design Principle: Expansion-Friendly

Every choice made now must not block the end goal:

| Choice now | Why it keeps the path open |
|------------|---------------------------|
| **LFR rational form in `deriv()`** — `a = N(Y)/d(Y) @ fnet`, then `xdot = Ax@x + Bw@w + Bu@u` through G | Preserves causal chain z → w → xdot. Future routing = expose z/w as Interconnect signals. `torch.linalg.solve` collapses this and loses the structure permanently. |
| **G matrix from M0_inv** (not M(Y_op)_inv) | G is always constant; Y-variation only enters via z and w. Correct per LFR derivation. |
| **z and w computed explicitly in `deriv()`** | Not yet routed through Interconnect, but structurally present. Adding routing = split into separate blocks + connect_signals calls. |
| `Gantry_State_Block` from Phase 1 (not `Linear_State_Block`) | Adding trainable params later = change constants to `nn.Parameter`; no structural rewrite |
| Physics constants as plain attributes in block `__init__` | Mirrors `Parameterized_MSD_State_Block` — swap to `nn.Parameter` + `param_loss()` for joint estimation |
| `Y_op` parameter on block (`float` → `None`) | Phase 1 frozen Y, Phase 3 LPV — one-line change in `deriv()` |
| Parallel ANN wiring from Phase 2 | Orthogonality constraint adds a loss term on top — wiring unchanged |
| Jan's block/interconnect structure throughout | LFR routing = split `Gantry_State_Block` into G-block + Δ-block, add connect_signals for z/w — no training loop changes |

---

## Code Location Constraint

```
model_augmentation/fit_systems/blocks.py          ← Gantry_State_Block added here
                                                     follows Nonlinear_MSD_State_Block pattern
                                                     Jan's file — extend it, don't treat as read-only

model_augmentation/systems/gantry_ss.py           — gantry physical constants + LFR poly constants
                                                     + build_poly_constants() + build_G_matrix_entries()
                                                     importable from blocks.py via model_augmentation.systems.gantry_ss

scripts/gantry/
  gantry_subnet.py              — training script
  gantry_evaluate.py            — evaluation
  gantry_state_comparison.py    — internal signal inspection

scripts/gantry/verification/
  verify_block_shapes.py        — shape + no-NaN check for Gantry_State_Block forward pass
  verify_lfr_residual.py        — check M(Y)@a - fnet < tol (mirrors lfr_forward.py Check 1)
  verify_one_step.py            — compare one RK4 step against numpy reference
```

All verification scripts live in `scripts/gantry/verification/`. They are standalone (no training),
run quickly, and must pass before any Phase training begins.

`gantry_subnet.py` imports `Gantry_State_Block` from `model_augmentation.fit_systems.blocks`
— same import as any other block. `gantry_ss.py` imports constants from `gantry_ss.py`.

**No imports from `lpv_lfr_baseline/`** in any gantry scripts or blocks. Physics constants
are hardcoded in `gantry_ss.py` and imported from there. The `lpv_lfr_baseline/` module
is a separate research implementation and must not be coupled to the SubNet pipeline.

---

## Mismatch Strategy

| Phase | Data source | Baseline model | Mismatch |
|-------|-------------|----------------|----------|
| 1 | Python: single trajectory from same frozen-Y RK4 block | Same frozen-Y RK4 block | None — NRMS → 0 is the sanity check |
| 2 | MATLAB: motion-profile trajectories (no multisine) from gantry + extra MSD | Frozen-Y RK4, no extra MSD | Known controlled mismatch |
| 3 | Same as Phase 2 | LPV RK4, no extra MSD | LPV variation + extra MSD residual |

---

## System Summary

| Property | Value |
|----------|-------|
| Inputs `u` | 3 (stage forces: F_X1, F_X2, F_Y) |
| Outputs `y` | 3 (stage positions: X1, X2, Y) |
| States `x` | 6 (logical: q_logical, qdot_logical) |
| Block | `Gantry_State_Block` — continuous ODE integrated with RK4 |
| Scheduling variable | Y = x[2]; frozen in Phase 1/2, self-scheduled in Phase 3 |
| Sampling rate | 20 kHz |

**Why no SISO phase:** X1 and X2 are mechanically coupled — common mode drives X
translation, differential mode drives rotation (mechanically limited). No clean SISO
on X. Start at full MIMO directly.

---

## Starting Point

```
scripts/ecc_2025/msd_ndof_interconnect_dynamic.py
    → scripts/gantry/gantry_subnet.py
```

Replace MSD-specific sections in order: data → block → wiring → save path.

---

## Phase Ordering Rationale

1. **MIMO, frozen Y, no augmentation** — pipeline sanity check; confirms wiring, normalisation, training loop. DONE.
2. **MIMO, frozen Y, dynamic parallel ANN** — core augmentation pipeline at fixed operating point; ANN learns MSD dynamics with small Y perturbations. **Current focus.**
3. **MIMO, LPV + dynamic parallel ANN** — full research contribution; self-scheduled M(Y) on top of step 2. Deferred until step 2 is validated.

**Why LPV is deferred:**
Augmentation at frozen Y is simpler to debug and sufficient to verify whether the ANN can learn the hidden MSD at a single operating point. LPV adds gradient complexity (Horner Jacobian, h-detach, shorter NF) and data requirements (Y-varying trajectories with adequate std_x[2]). Doing augmentation first at frozen Y avoids conflating LPV training issues with augmentation issues.

**Why small Y perturbations at frozen Y (not fully frozen):**
The hidden MSD resonates at 400 Hz. With Y reference exactly constant, Y acceleration is near zero and the MSD is not excited. Adding small sinusoidal Y perturbations (5-10 mm, 0.5-2 Hz) provides Y acceleration that excites the MSD while keeping Y variation small enough that the frozen-Y LTI approximation still holds for the baseline block.

**Why dynamic parallel (not static):**
The hidden MSD resonates at 400 Hz, sampled at 20 kHz — 50 samples per oscillation cycle. A
static ANN sees only the current `(x, u)` at each RK4 step and has no memory between steps. It
cannot track `delta_a` from instantaneous state alone. Dynamic parallel with 2 extra states
(`n_hidden=2`) implicitly tracks `delta_a` and `delta_a_dot`, giving the ANN the memory it
physically requires. See the static vs dynamic table in the Phase 3 section.

---

## LFR Structure and LPV Implementation

### LFR signal flow (all phases)

The gantry mass matrix is Y-dependent: `M(Y) = M0 + M1*Y + M2*Y²`. The LFR
factorisation avoids recomputing `M(Y)^{-1}` directly by instead computing the
rational adjugate `N(Y)/d(Y)` — polynomial in Y — via Horner evaluation:

```
fnet = -K@q - C@qdot + P@u_stage       # net logical force  (3-vec)
a    = N(Y)/d(Y) @ fnet                # M(Y)^{-1} @ fnet  (3-vec, rational in Y)
z    = [a; Y*a]                        # LFR latent z       (6-vec)
w    = Y * z                           # LFR latent w = Δ(Y)·z = [Y*a; Y²*a]  (6-vec)
xdot = Ax@x + Bw@w + Bu@u_log         # through G — NOT directly from a
```

`G = [Ax, Bw, Bu]` is constant, built from `M0_inv = N0/d0` (Y=0). Y-variation
enters `xdot` **only through w** (which carries Y and Y² scaled accelerations).

Polynomial constants:
- `N(Y) = N0 + N1*Y + N2*Y²`   (3×3 adjugate, quadratic in Y)
- `d(Y) = d0 + d1*Y + d2*Y²`   (scalar determinant, quadratic in Y)
- At Y=0: `N(0)=N0`, `d(0)=d0=det(M0)` — the constant reference point

Implemented in: `model_augmentation/fit_systems/blocks.py:Gantry_State_Block.deriv()`
(lines 754–805) and `model_augmentation/systems/gantry_ss.py:build_poly_constants()`.

### LPV self-scheduling — already implemented

`Gantry_State_Block(Y_op=None)` activates LPV. Inside `deriv()` (line 781):

```python
# Frozen (Y_op is float):  N_op, d_op precomputed at __init__; deriv() is pure matmul
# LPV   (Y_op is None):    Y = x2[:, 2] per step; Horner form recomputes N(Y), d(Y)
Y   = x2[:, 2]                     # scheduling variable extracted from current state
dY  = mh*(alpha*gamma - beta² + 2*beta*mh*Y + mh*(alpha-mh)*Y²)
a   = (N0 + Y*(N1 + Y*N2)) @ fnet / dY    # Horner evaluation
```

**The encoder is structurally unchanged for LPV.** It still maps `(u_past, y_past) → x̂₀`
(6D). LPV scheduling happens inside `deriv()` during rollout — the encoder estimates
initial conditions from I/O history and has no awareness of whether the downstream block
is frozen-Y or LPV. With Y-varying data, `y_past` contains richer information (genuine Y
variation), which can only help encoder quality.

**The interconnect wiring is unchanged for LPV.** Self-scheduling is internal to
`Gantry_State_Block` — no new signals, no wiring changes. Phase 2 is literally:

```python
# Phase 1:  Gantry_State_Block(Y_op=0.3)
# Phase 2:  Gantry_State_Block(Y_op=None)   ← only change
```

### LFR routing — future end goal, not needed for Phase 2/3

The `lpv_lfr_baseline/blocks/lfr_param_block.py` (Jan-compatible block wrapper) outputs
`nw=18`: `(x_next=6, z_lfr=6, w_lfr=6)`. This routes z and w as explicit interconnect
signals so an augmentation ANN can connect to specific physical excitation channels.

Our `Gantry_State_Block` computes z and w internally (blocks.py lines 795–796) but
returns only `xdot` (nw=6). The LFR structure is structurally preserved but not yet
exposed as interconnect signals.

**Phase 2 and Phase 3 do not need LFR routing.** The dynamic parallel ANN connects to
the full state x and input u — not to z or w. LFR routing is only required when the ANN
must target specific physical channels (the end-goal architecture: split `Gantry_State_Block`
into G-block + Δ-block, connect_signals for z and w). This is a future step.

### Block class structure — factory + subclasses

The frozen-Y and LPV paths have different `__init__` requirements (frozen precomputes
`N_op`, `d_op`; LPV does not) and different hot-path logic in `deriv()`. Encoding
both in one class with `if self.Y_op is not None:` adds a branch inside the RK4 inner
loop (10 substeps × NF steps) and makes the two paths harder to test independently.

**Design: factory function + two private subclasses.**

```python
# Public API — call signature unchanged from current code
def Gantry_State_Block(Y_op=0.3, std_x=..., std_u=..., Ts=1/20000, **kwargs):
    """Factory. Y_op=float → frozen; Y_op=None → LPV self-scheduled."""
    if Y_op is None:
        return _Gantry_State_Block_LPV(std_x=std_x, std_u=std_u, Ts=Ts, **kwargs)
    return _Gantry_State_Block_Frozen(Y_op=Y_op, std_x=std_x, std_u=std_u, Ts=Ts, **kwargs)

class _Gantry_State_Block_Base(Discrete_Nonlinear_Function_Block):
    """Shared buffers (Ax, Bw, Bu, K, C, P, std_x, std_u, N0, N1, N2, mh, ...)
    and shared nonlinear_function() RK4 loop. deriv() left abstract."""

class _Gantry_State_Block_Frozen(_Gantry_State_Block_Base):
    """Frozen-Y: N_op and d_op precomputed at __init__. deriv() is pure matmul.
    No branch. No dynamic computation of N(Y)/d(Y) at runtime."""

class _Gantry_State_Block_LPV(_Gantry_State_Block_Base):
    """LPV self-scheduled: Y = x_phys[:,2] at each RK4 substep.
    Horner form for N(Y) and d(Y). Physical validity guards TBD (open question)."""
```

**What each class owns:**

| | `_Gantry_State_Block_Frozen` | `_Gantry_State_Block_LPV` |
|---|---|---|
| `__init__` extras | Precomputes `N_op`, `d_op` at `Y_op` | None |
| `deriv()` | Pure matmul — `N_op @ fnet / d_op` | Horner: `(N0 + Y*(N1+Y*N2)) / d(Y)` |
| Branch in hot path | None | None |
| Physical validity guards | Not needed (Y fixed) | Open — to be decided |
| Gradient behaviour | Linear Jacobian (Ax term only) | Nonlinear Y-dependent Jacobian |

**Existing call sites are unchanged:**
```python
Gantry_State_Block(Y_op=0.3, std_x=std_x, std_u=std_u)   # → _Frozen, as before
Gantry_State_Block(Y_op=None, std_x=std_x, std_u=std_u)   # → _LPV, Phase 2+
```

**Open questions before implementing `_Gantry_State_Block_LPV`:**
- Physical validity guards on Y and d(Y) — approach not yet decided
- Gradient behaviour through LPV Jacobian — NF choice and clipping strategy not yet decided
- These must be resolved and logged in `docs/decisions.md` before implementation begins

### LPV Implementation: Five Problems and Mitigations

These problems become active when switching from `_Gantry_State_Block_Frozen` to
`_Gantry_State_Block_LPV`. Each must be resolved (or explicitly accepted) before
Phase 2 training begins. Decisions must be logged in `docs/decisions.md`.

**Problem 1 — `std_x[2] ≈ 0` with frozen-Y training data**

Y is frozen at 0.3 m → `std(x[:,2]) ≈ 0` → normalisation guard `1e-8` fires but
`std_x[2]` carries no information. The LPV block's Horner evaluation uses
`Y = x_norm[:,2] * std_x[2]` to recover physical Y, so a near-zero `std_x[2]`
collapses all Y variation at runtime.

Mitigation: Y-varying training data (Phase 2 prerequisite). This is a data
requirement, not a code change. Verify `std_x[2]` before LPV training.

**Problem 2 — d(Y) singularity / negative denominator**

`d(Y) = d0 + d1·Y + d2·Y²` is the determinant of M(Y) — positive by physics for all
physical Y. But the encoder can output x̂₀[2] outside the physical operating range.
If Y strays far from [Y_min, Y_max], `d(Y)` can become small or negative → NaN in
`a = N(Y)/d(Y) @ fnet`.

Mitigations (not yet decided — see open questions below):
- Clamp x̂₀[2] to [0.0, 0.5] m (physical range) before entering rollout
- State-consistency regularization (Sertbas & Kumbasar 2025, arXiv 2510.24757):
  `L_consistency = ||x̂_enc(t) - f_physics(x̂_enc(t-1), u(t-1), p(t-1))||²`
  anchors encoder output to the physics manifold and indirectly keeps Y physical
- Log-reparametrisation (from `lfr_param_block.py` lines 274–276):
  `Y_phys = Y_min + (Y_max - Y_min) * sigmoid(Y_raw)` — maps encoder Y to physical range

**Problem 3 — Jacobian explosion over NF BPTT steps**

The LPV Jacobian `∂ẋ/∂x` includes `∂(N(Y)/d(Y))/∂x[2]` — a Y-dependent term absent
in the frozen case. Over NF BPTT steps these extra terms can compound and cause
gradient explosion, especially with long NF.

Mitigations (not yet decided):
- Start with NF=50 (shorter than frozen Phase 1 NF=200) — Verhoek et al. CDC 2023
  (arXiv 2204.04060) use short windows for LPV-SUBNET for exactly this reason
- h-detach (Arpit et al. ICLR 2019, arXiv 1810.03023): treat scheduling variable Y
  as non-differentiable in the backward pass — `Y_sched = Y.detach()` — stops the
  Jacobian chain through Y while preserving gradients through x and u
- Gradient clipping (`torch.nn.utils.clip_grad_norm_`) as fallback

**Problem 4 — Y re-extraction at each RK4 substep**

Inside `_Gantry_State_Block_LPV.deriv()`, `Y = x_phys[:,2]` is extracted at each
of 4 RK4 substeps. This propagates gradients through 4 nonlinear Y-dependent paths
per BPTT timestep, compounding Problem 3.

Mitigations:
- h-detach (Problem 3 mitigation) also stops substep Y gradients as a side effect
- Alternative: extract Y only at the start of each full timestep (pass Y as argument
  to `deriv()` rather than extracting from x mid-RK4). Less physically accurate but
  gradient-safe. Note: this changes the ODE integration slightly.

**Problem 5 — Encoder Y estimation (least severe)**

LPV scheduling uses `Y = x̂₀[2]` from the encoder output. A wrong initial Y means the
wrong `M(Y₀)` is used for the first NF steps.

Mitigation: Y is always `y[:,2]` in the output — directly measurable. A structured
encoder reads `x̂₀[2] = y_past[-1, 2]` without learning, guaranteeing a physically
correct Y₀. See "Encoder Architecture and Limitations" section below.

**Open decisions before implementing `_Gantry_State_Block_LPV`:**
- Y clamping strategy (sigmoid reparametrisation vs explicit clamp vs none)
- Gradient strategy (h-detach vs short NF vs clipping vs combination)
- Whether state-consistency regularization adds enough value to justify implementation
- NF starting value for Phase 2 (propose: NF=50, tune upward if stable)

---

### Comparison: two LPV implementations in this repo

| | `Gantry_State_Block` (Jan's framework) | `lpv_lfr_baseline` |
|---|---|---|
| Location | `model_augmentation/fit_systems/blocks.py` | `lpv_lfr_baseline/` |
| LPV scheduling | `_LPV` subclass: `Y=x[:,2]` in `deriv()` | `Y_override=None` in `rk4_step()` |
| z/w routing | Internal only (not exposed) | Output as `nw=18` (x_next + z + w) |
| Trainable params | Fixed buffers (Phase 1–3) | `nn.Parameter` via log-reparametrisation |
| Use in Jan's interconnect | Yes — this is the training block | Reference only — do not import |
| Purpose | SUBNET training (Phases 1–3) | Parameter recovery (separate research arm) |

---

## Encoder Architecture and Limitations

### Current encoder — `modified_encoder_net`

Location: `model_augmentation/fit_systems/interconnect.py` (lines ~360–380).

```python
class modified_encoder_net(nn.Module):
    def forward(self, upast, ypast):
        # Flatten (nb, nu) + (na, ny) → single vector
        net_in = torch.cat([upast.view(upast.shape[0], -1),
                            ypast.view(ypast.shape[0], -1)], dim=1)
        return self.net(net_in)   # simple_res_net: linear residual stack
```

Properties of Jan's encoder:
- **Static feedforward** — ResNet with Tanh activations, no LSTM, no attention
- **Flattens history** — `nb×nu + na×ny` = `100×3 + 100×3 = 600` inputs for Phase 1
- **No temporal structure** — all timesteps in history treated as unordered features
- **No physics awareness** — maps I/O history to x̂₀ without any dynamical constraint
- **Output:** 6D for Phase 1/2; `(6 + n_hidden)D` for Phase 3

The encoder's role is **initial condition estimation** for the BPTT rollout — it maps
`(u_past[0:NB], y_past[0:NA])` → x̂₀ which seeds the interconnect simulation.
It does NOT estimate state at every timestep; after x̂₀, the interconnect propagates
state forward via the physics block.

### Measurement structure insight

The output equation is `y = Cd @ x` where `Cd = [P^T | 0]` (positions only):

```
y[:,0] = X1  =  q1  =  q_logical[0] + q_logical[1]
y[:,1] = X2  =  q2  =  q_logical[0] - q_logical[1]   (via P^T)
y[:,2] = Y   =  q3  =  q_logical[2]
```

All **three positions** are linearly recoverable from output via `q = P^{-T} @ y`.
**Velocities are NOT in the output.** The encoder's primary job is therefore
**velocity estimation** — positions are redundant given y_past.

### Structured encoder improvement (Phase 2+)

Under the structured encoder design:
- `x̂₀[2]` (initial Y) is read directly from `y_past[-1, 2]` — Y is always `y[:,2]`,
  no learning needed. This guarantees a physically correct Y₀ for M(Y₀) in LPV.
- Remaining 5 components (q0, q1 and all three velocities) are learned from history.
- This reduces the encoder's effective output from 6D to 5D learned, with 1D hard-coded.

**Justification:** Y determines M(Y₀) for LPV initialisation. The measurement gives
physical Y directly, regardless of hidden MSD mismatch — no ambiguity possible.

**Mismatch caveat (Phase 3):** Under augmented-system mismatch, the 8-state truth has
the same output `y = Cd_6 @ x_6` (positions only). Y₀ = y_past[-1,2] remains
unambiguously correct (physical Y is measured). For positions X1, X2: reading directly
from y_past via P^{-T} gives positions at `t = t_0 - 1 sample`, not `t = t_0`, so the
encoder still needs to learn when to apply corrections. Velocities remain fully hidden.

### Velocity validation gap

BPTT trains on MSE loss over output y (positions only). Velocity components in x̂₀
are **never directly supervised**. They are trained implicitly: wrong velocities →
rollout diverges from reference → MSE increases → gradient corrects them. But this
supervision is indirect and may leave velocities poorly constrained.

**The only direct velocity check** is the Python matched case (Phase 1 gate):
simulate from x̂₀ and verify NRMS → near zero. If the block is correct and data is
matched, BPTT must have learned velocities correctly to achieve low NRMS.

For Phase 3 (mismatched data), there is no direct velocity check. Plausibility
bounds (physical velocity range from MATLAB data) and smooth state trajectories
are the available sanity checks.

### Literature — encoder for MIMO LPV SUBNET

| Reference | Key finding | Relevance |
|-----------|-------------|-----------|
| Verhoek et al. CDC 2023, arXiv 2204.04060 | LPV-SUBNET with p-net for scheduling variable; short NF for stability | Direct precedent; our case simpler (Y measured, no p-net) |
| Beintema et al. Automatica 2023, arXiv 2210.14816 | Foundational MIMO SUBNET theory and stability | Underpins Jan's framework |
| Ramkannan et al. 2023, arXiv 2304.02119 | BLA encoder initialisation — init encoder from Best Linear Approx reconstructability map | Practical warm-start before BPTT |
| Sertbas & Kumbasar 2025, arXiv 2510.24757 | State-consistency regularization `L_cons = \|\|x̂_enc(t) - f(x̂_enc(t-1),u(t-1))\|\|²` | Anchors encoder; shortens effective BPTT horizon |
| Arpit et al. ICLR 2019, arXiv 1810.03023 | h-detach: freeze scheduling variable gradient in backward pass | Justifies detaching Y in LPV Jacobian |

---

## General Failure Modes (all phases)

| Symptom | Most likely cause |
|---------|-------------------|
| Flat loss from epoch 0 | Data format wrong, normalisation broken, or block wiring disconnected |
| NaN loss | float32/float64 mismatch at physics boundary, or LR too high |
| Loss decreases then immediately plateaus | Encoder `na`/`nb` too short |
| Simulation worse than zero-state init | Encoder making things worse — wiring or normalisation error |
| NRMS > 1.0 | Model worse than predicting the mean — something fundamental is broken |

---

## Phases

### Phase 1 — MIMO, frozen Y, no augmentation

**Goal:** full 3×3 pipeline running end-to-end. No augmentation, no LPV.

**Block:** `Gantry_State_Block(Y_op=0.3)` — LFR-structured, physics frozen at one
operating point. RK4 integration with LFR signal flow inside `deriv()`:

```
u_log = P @ u_stage                        # stage → logical  (applied inside deriv, not by caller)
fnet  = -K@q - C@qdot + u_log             # net logical force
a     = N(Y_op)/d(Y_op) @ fnet            # rational M(Y)^{-1} — precomputed at init for frozen Y
z     = [a;  Y_op*a]                       # LFR latent z  (6-vector, not yet routed externally)
w     = Y_op * z                           # LFR latent w = Δ(Y)·z  = [Y_op*a; Y_op²*a]
xdot  = Ax@x + Bw@w + Bu@u_log            # through G — NOT directly from a
```

G is built from **M0_inv = N0/d0** (Y=0 constant, purely polynomial — no solve).
Y never appears in G. Y-variation enters xdot **only** through w (which carries Y*a and Y²*a).
Frozen path (`_Gantry_State_Block_Frozen`): N(Y_op), d(Y_op) precomputed at `__init__`;
`deriv()` is pure matmul — no branch, no Horner at runtime.
LPV path (`_Gantry_State_Block_LPV`): Y = x_phys[:,2] per substep; Horner form for N(Y)/d(Y).
Both accessed via factory: `Gantry_State_Block(Y_op=0.3)` or `Gantry_State_Block(Y_op=None)`.

**P-transform note:** `lfr_forward.py` expects u already in logical coordinates — the caller
applies P. Jan's Interconnect passes u_stage directly to the block, so P is applied inside
`deriv()` before the LFR flow. Mathematically identical; structurally a one-step shift inward.

**What to build:**
1. `model_augmentation/systems/gantry_ss.py` — physical constants + `build_poly_constants()` + G matrix entries
2. `Gantry_State_Block` in `model_augmentation/fit_systems/blocks.py` — LFR `deriv()`, Jan's RK4 `nonlinear_function`
3. **Data generation** — simulate with same block (frozen Y) to get matched train/val data
4. **Interconnect wiring** — `Interconnect(nx=6, nu=3, ny=3)` + `Linear_Output_Block`
5. **SSE_Interconnect + fit** — adapt ECC 2025 training call

**Not yet implemented (end goal):**
- Physical parameters are fixed (not `nn.Parameter`) — joint estimation deferred
- No orthogonality constraint
- z/w not yet routed as explicit Interconnect signals — future split into G-block + Δ-block

**Success criterion:** NRMS → near zero (data from same model).

**Verification checklist:**
- [ ] Loss decreases over epochs
- [ ] Loss does not go NaN
- [ ] `fit_sys.simulate(val_data)` returns ŷ of shape `(T, 3)`
- [ ] Per-channel NRMS reported (X1, X2, Y)
- [ ] NRMS → near zero after sufficient epochs
- [ ] Encoder-initialised simulation beats zero-state initialisation
- [ ] **One channel flat** → data shape error; check `System_data` construction
- [ ] **All channels flat** → MIMO encoder fix (`self.ny` line 369 in `interconnect.py`)
- [ ] **NRMS stays high despite low loss** → normalisation mismatch

---

### Phase 2 — MIMO, LPV (self-scheduled Y), no augmentation

**Goal:** validate LPV scheduling on Y-varying data. Prerequisite for augmentation.

**Why this comes before augmentation:** with Y frozen, the hidden MSD is not excited
(`delta_a ≈ L0`). For the ANN to learn MSD dynamics, Y must move. But when Y moves
with a frozen-Y baseline, the ANN absorbs both LPV mismatch and MSD error simultaneously
— the two cannot be separated. LPV must be correct first.

**Block swap:** `Gantry_State_Block(Y_op=0.3)` → `Gantry_State_Block(Y_op=None)`

Inside `deriv()`: `Y = x[:, 2]` instead of the fixed scalar. `N(Y)` and `d(Y)` computed
via Horner form each step. Everything else — wiring, encoder, training loop — unchanged.

**Data requirement:** Y-varying MATLAB trajectories. The existing data files
(`gantry_lti_train.mat`, `gantry_aug_train.mat`) both have Y frozen at 0.3 m — they
cannot be used for Phase 2. New data generation scripts are needed with Y motion included
in the reference trajectory. Both the nominal gantry model (for Phase 2) and augmented
model (for Phase 3) must be re-simulated with Y-varying profiles.

**Not yet implemented (end goal):**
- Joint estimation still deferred
- Orthogonality still deferred
- LFR routing still deferred

**Speed note:** RK4 backprop is 4× more compute than a linear block. Use HPC for longer `nf`.

**Success criterion:** LPV NRMS < Phase 1 NRMS on Y-varying validation data.

**Verification checklist:**
- [ ] Training stable (no divergence from self-scheduling)
- [ ] NRMS lower than Phase 1 baseline on Y-varying data
- [ ] Largest improvement on Y channel (scheduling variable carries the LPV correction)
- [ ] Y trajectory coverage verified (Y must actually move in training data)
- [ ] **Loss diverges** → self-scheduling unstable; try frozen Y at mean as warm-start
- [ ] **No improvement over Phase 1** → check Y-variation in data; check Horner form in `deriv()`

---

### Phase 3 — MIMO, LPV + dynamic parallel ANN

**Goal:** add ANN augmentation on top of correct LPV baseline. Core research contribution.

**Mismatch:** training data from gantry + extra MSD on payload mass with Y-varying motion.
`Gantry_State_Block` (LPV) handles `M(Y)` correctly; the ANN learns the remaining MSD residual.

**Why dynamic parallel (not static):**
The hidden MSD resonates at 400 Hz (50 samples per cycle at 20 kHz). A static ANN maps
instantaneous `(x, u)` → correction with no memory between timesteps — it cannot track
`delta_a`. Dynamic parallel adds `n_hidden=2` extra states to the interconnect that the
ANN owns and integrates via RK4, implicitly learning to track `delta_a` and `delta_a_dot`.

**Static vs dynamic parallel — architecture reference:**
Both use `Static_ANN_Block`. "Dynamic" refers to the interconnect state dimension, not the
ANN itself. Reference: `scripts/ecc_2025/msd_ndof_interconnect_dynamic.py` lines 70–75, 83–98.

| | Static parallel | Dynamic parallel (Phase 3) |
|---|---|---|
| `nx` | 6 (physics only) | 6 + n_hidden (= 8 for MSD) |
| ANN memory | none — stateless each step | yes — extra states integrated by RK4 |
| Encoder output | 6D | (6 + n_hidden)D |
| ANN drives | correction to xdot[0:6] | correction to xdot[0:nx]; owns xdot[6:nx] entirely |
| Can learn | steady-state residual | transient dynamics with their own timescale |

For static parallel: `nxd = 2*FP_dof` (line 73 in reference script).
For dynamic parallel: `nxd = 2*dof` (line 71), where `dof` is the true system DOF.
The physics block receives only states 0:6 via `selection_matrix` (line 93);
its xdot is placed back at those indices via `expansion_matrix` (line 95).
The ANN block receives all `nxd` states and outputs corrections to all of them (line 91).
Extra states (6:nxd) have no physics block — only the ANN drives them.

**Wiring (dynamic parallel, nx=8):**
```python
nxd = 8  # 6 physics + 2 ANN-owned (implicit delta_a, delta_a_dot)
interconnect = Interconnect(nxd, nu=3, ny=3)

# Physics block — operates on states 0:6 only
interconnect.connect_signals("x", state_block, "concat", selection_matrix(list(range(6)), nxd))
interconnect.connect_signals("u", state_block, "concat")
interconnect.connect_signals(state_block, "xp", "additive", expansion_matrix(list(range(6)), nxd))

# ANN block — sees all 8 states and all 3 inputs, corrects all 8 xdot components
aug_block = Static_ANN_Block(nz=nxd+3, nw=nxd, n_nodes_per_layer=64, n_hidden_layers=2,
                              net=zero_init_feed_forward_nn, activation=nn.Tanh)
interconnect.add_block(aug_block)
interconnect.connect_block_signals(aug_block, ["x", "u"], ["xp"])

# Output block — reads only physics states 0:6
interconnect.connect_signals("x", output_block, "concat", selection_matrix(list(range(6)), nxd))
interconnect.connect_block_signals(output_block, ["u"], ["y"])
```

**Reference wiring (original frozen-Y static ANN, preserved for pipeline check):**
```python
# Static parallel, nx=6 — validates wiring and training loop with frozen Y + MSD mismatch.
# Not a research phase (Y frozen → MSD not excited), but useful for sanity checking
# the augmentation path before running Phase 3.
interconnect.connect_signals("x",  aug_block, "concat", selection_matrix(list(range(nx)), nx))
interconnect.connect_signals("u",  aug_block, "concat")
interconnect.connect_signals(aug_block, "xp", "additive", expansion_matrix(list(range(nx)), nx))
```

**`gantry_interconnect_dynamic.py` — conversion from `msd_ndof_interconnect_dynamic.py`**

Reference: `scripts/ecc_2025/msd_ndof_interconnect_dynamic.py` (= `scripts/gantry/gantry_subnet.py` verbatim copy).
Target: `scripts/gantry/gantry_interconnect_dynamic.py`.
Rule: minimal changes only — copy Jan's structure as-is wherever possible.

| Section | Jan (ECC 2025) | Gantry | Status |
|---|---|---|---|
| Imports | `from model_augmentation.systems.mass_spring_damper import *` | `from model_augmentation.systems.gantry_ss import Cd, Dd, P` | Trivial |
| Flags | `FP_type`, `SNR`, `dynamic_aug`, `type_aug`, `linear_parallel`, `wait_minutes` | Drop all — not applicable | Trivial |
| Hyperparams | `nf`, `epochs`, `batch_size` | Keep; add `NX_ann=2`, `NX_total=8`, `Y_OP`, `SEED`, `N_HOLD` | Verified in Phase 1 |
| Data loading | `.npz` + noise injection | MATLAB `.mat` load, no noise | Verified in Phase 1 |
| FP model + `normalize_linear_ss_matrices` | Loads `A,B,C,D`; pre-normalizes matrices | Drop entirely — replaced by `std_x`, `std_u`, `ystd`, `Cd_norm`, `Y_OP` override | Verified in Phase 1 + `verify_normalization.py` |
| `Interconnect(nxd, nu, ny)` | `Interconnect(6, 3, 3)` | `Interconnect(8, 3, 3)` | Trivial dimension change |
| Physical state block | `Parameterized_MSD_State_Block(nz=5, nw=4)` | `Gantry_State_Block(Y_op=Y_OP, std_x=std_x, std_u=std_u)` | Verified in Phase 1 |
| Output block | `Linear_Output_Block(C=C_bar_bla)` — shape `(2,4)` | `Linear_Output_Block(C=Cd_norm, D=Dd.numpy())` — shape `(3,6)`; `selection_matrix` handles state slicing | Verified in Phase 1 + `verify_normalization.py` |
| ANN block `nz`, `nw` | `nz=6+3=9`, `nw=6` | `nz=8+3=11`, `nw=8` | Trivial from dimension change |
| ANN block `net` | `zero_init_feed_forward_nn` | 1:1 copy — critical for stability at init | Verified in Jan's code |
| ANN block `activation` | `torch.nn.Tanh` | 1:1 copy | Verified in Jan's code |
| ANN block `n_nodes_per_layer` | 8 | TBD — hyperparameter, start with 64 | Not a correctness issue |
| ANN wiring | `connect_block_signals(ANN, ["x","u"], ["xp"])` | 1:1 copy | 1:1 |
| Physics wiring | `selection_matrix([0,1,2,3], 6)` / `expansion_matrix([0,1,2,3], 6)` | `selection_matrix(np.arange(6), 8)` / `expansion_matrix(np.arange(6), 8)` | **UNVERIFIED — pretest required** |
| Output wiring | `selection_matrix([0,1,2,3], 6)` | `selection_matrix(np.arange(6), 8)` | **UNVERIFIED — same pretest** |
| `SSE_Interconnect` `na`, `nb` | `nxd*2+1 = 13` | `NX_total*2+1 = 17` | Trivial from dimension change |
| `SSE_Interconnect` `e_net_kwargs` | `{"n_nodes_per_layer": 16}` | `{"n_nodes_per_layer": 64, "n_hidden_layers": 2}` | Carried from Phase 1 |
| `fit()` `auto_fit_norm` | `True` | `False` + manual norm setup before `fit()` | Verified in Phase 1 |
| `fit()` `validation_measure` | `"sim-RMS"` | 1:1 copy | Verified in Phase 1 |
| Manual norm setup | — (handled by `auto_fit_norm=True`) | `fit_sys.norm.u0=0`, `fit_sys.norm.ustd=std_u`, `fit_sys.norm.y0=0`, `fit_sys.norm.ystd=ystd` | Verified in Phase 1 |
| Save | MSD-specific path logic | Gantry `simulations/gantry_subnet/` pattern | Verified in Phase 1 |

**Why `auto_fit_norm=False` (the only non-trivial structural difference from Jan):**
Jan uses `auto_fit_norm=True` because his physical block is linear — the mean offset in `u` can
be absorbed into the B matrix via `normalize_linear_ss_matrices`. `Gantry_State_Block` is
nonlinear (LFR/RK4); there is no B matrix to absorb the mean. Setting `u0=0` ensures the
block sees full physical forces. Setting `y0=0` keeps `Cd_norm` consistent: the output
equation `y_norm = Cd_norm @ x_norm` holds only when no additive offset is present.

**Pretest required — `selection_matrix` / `expansion_matrix` with gantry dimensions:**
In Phase 1 the interconnect wiring used no selection/expansion matrices (NX_total=NX=6,
so no partitioning was needed). In Phase 3 the state is partitioned: physical states [0:6]
and ANN states [6:8]. This wiring pattern is taken directly from Jan but has not been
exercised with our block shapes. Before training:

1. Build the augmented interconnect with dummy data (no training)
2. Run one forward pass: `x = torch.zeros(1, 8)`, `u = torch.zeros(1, 3)`
3. Assert `y.shape == (1, 3)` and `xp.shape == (1, 8)`
4. Assert that `Gantry_State_Block.deriv()` receives a `(1, 6)` tensor (not `(1, 8)`)
5. Assert that `ANN_state_block` receives a `(1, 11)` tensor (`x` + `u`)

This pretest can be added to `scripts/gantry/verification/` before writing the full training script.

**Not yet implemented (end goal):**
- Joint estimation: physics parameters still fixed
- Orthogonality: ANN can still learn baseline-captured dynamics
- LFR routing: ANN connects to state, not to LFR latent variables

**Success criterion:** Phase 3 NRMS < Phase 2 NRMS on Y-varying validation data with MSD.

**Verification checklist:**
- [ ] Loss lower than Phase 2
- [ ] Per-channel NRMS improvement over Phase 2 on validation data
- [ ] ANN output magnitude reasonable (not dominating the physics)
- [ ] Extra ANN states (6:8) bounded and smooth — not diverging
- [ ] Encoder output 8D — verify shape before training
- [ ] **No improvement** → check additive connection to `xp`; check ANN state initialisation
- [ ] **ANN states diverge** → reduce LR or ANN size; add state norm penalty
- [ ] **Overfitting** → reduce ANN size or add regularisation

---

### Future — Joint Estimation + Orthogonality + LFR Routing

**Not implemented yet. Expansion path from Phase 3:**

**Joint estimation:**
- Change physics constants in `Gantry_State_Block` from plain attributes to `nn.Parameter`
- Add `param_loss()` method (following `Parameterized_MSD_State_Block` pattern)
- Interconnect picks up `param_loss()` automatically — no training loop changes

**Orthogonality:**
- Add orthogonality penalty on ANN augmentation output
- Penalises ANN for learning directions already spanned by the baseline Jacobian
- Adds a loss term — wiring and block structure unchanged

**LFR signal routing:**
- Decompose `Gantry_State_Block` into G-matrix block + Δ(Y) block
- Route z and w as explicit signals through the Interconnect
- ANN connects to specific LFR channels instead of full state

---

## Key Reference Files

| File | Role |
|------|------|
| `scripts/ecc_2025/msd_ndof_interconnect_dynamic.py` | Template |
| `model_augmentation/fit_systems/blocks.py` | `Nonlinear_MSD_State_Block` — RK4 pattern to follow; `Parameterized_MSD_State_Block` — joint estimation pattern; **add `Gantry_State_Block` here** |
| `model_augmentation/fit_systems/interconnect.py` | `SSE_Interconnect`, `Interconnect` |
| `model_augmentation/systems/gantry_ss.py` | Gantry physical constants — single source of truth (importable by blocks.py) |
| `scripts/gantry/gantry_subnet.py` | Training script |
| `lpv_lfr_baseline/scripts/train_param_recovery.py` | **Reference only — do not import.** Shows how to inspect internal model state, plot trajectories, and evaluate a trained gantry model. Pattern to follow in `gantry_evaluate.py` / `gantry_state_comparison.py`. |
| `docs/lfr-baseline-implementation-method.md` | Justifies why z/w must be computed explicitly (resolve-and-retain argument). Supervisor requirement D-005/D-013/D-017. Validates `Gantry_State_Block` design over collapsed `A_c(Y)x + B_c(Y)u`. |
| arXiv 2204.04060 — Verhoek et al. CDC 2023 | LPV-SUBNET: SUBNET extended to LPV systems; p-net estimates scheduling; short NF for stability. Direct architectural precedent for Phase 2. PDF in `docs/references.md`. |
| arXiv 2210.14816 — Beintema et al. Automatica 2023 | Foundational MIMO SUBNET — theoretical basis for Jan's framework and our encoder. |
| arXiv 2304.02119 — Ramkannan et al. 2023 | BLA encoder initialisation — warm-start encoder from Best Linear Approximation before BPTT. |
| arXiv 2510.24757 — Sertbas & Kumbasar 2025 | State-consistency regularization — anchors encoder to physics manifold; helps d(Y) remain physical. |
| arXiv 1810.03023 — Arpit et al. ICLR 2019 | h-detach — freeze scheduling variable gradient in backward pass; justifies LPV Jacobian treatment. |

---

## Data Pipeline

### Phase 1 — Python simulation (matched case)

Single trajectory, simulated from `Gantry_State_Block(Y_op=0.3)` directly in Python.
Y is frozen at the operating point (residual Y variation comes from coupling only, but
M(Y) is still evaluated there — the frozen-Y model captures this).

```python
# Generate: step Gantry_State_Block with a motion-profile input, save u and y
# Also save the full 6D state x = [q_logical; qdot_logical] at every step
# → enables SS_pre_encoder (state supervision) and encoder quality verification

train_data = System_data_with_x(u=u.astype(np.float32),
                                 y=y.astype(np.float32),
                                 x=x.astype(np.float32),
                                 dt=1/20000)
```

Input signal: simple motion profile (ramp/hold), not multisine.
Split: one trajectory for train, a shorter separate trajectory (different initial
conditions or input) for validation.

**Why save x:** the encoder maps past (u, y) → x̂0. Saving the true x allows:
1. Direct comparison x̂ vs x to verify encoder quality per channel
2. `SS_pre_encoder` training (state supervision before BPTT)
3. Plotting state trajectories to verify physics is correct

**Encoder coordinate convention:** the encoder output x̂₀ is always in **logical
coordinates** — it feeds directly into `Gantry_State_Block` as the initial state,
and that block operates internally in logical coordinates. `x_logical` saved from
MATLAB is also in logical coordinates (derived via `q_logical = P^{-T} @ q_stage`).
Stage coordinates appear only at the output (`Cd = [P^T | 0]`) and never inside the
state evolution.

**Encoder verification — matched case (nominal data):**
- True system IS the nominal model → `x̂₀ ≈ x_logical[0]` should hold channel-by-channel
- Verify by: simulating forward from `x̂₀` and from `x₀=0`; encoder-initialised NRMS
  should be significantly lower

**Encoder verification — mismatched case (augmented data, Phase 2):**
- True system has 8 states (hidden MSD); nominal model has 6 → irreducible mismatch
- `x̂₀ ≠ x_logical[0]` (x_logical is the 6D projection of the 8-state trajectory;
  encoder finds the *best nominal initial condition*, which is different)
- Verify by: plausibility checks (bounded, smooth states), and NRMS improvement over
  zero-state init — NOT by comparing x̂₀ to x_logical directly
- `x_logical` saved from augmented MATLAB data still useful as a ceiling: it shows the
  best possible nominal-state projection, bounding how close the encoder can get

**Note — velocities from MATLAB data (Phase 2+):** the existing MATLAB script
(`Matlab-scripts/generate_identification_experiment_without_multisine.m`) saves
stage positions `q1` [X1, X2, Y] and forces `u_total`, but NOT velocities.
To get the full state for pre-encoder training from MATLAB data, either:
- Add velocity ToWorkspace blocks to the Simulink model (`gantry_2025a.slx`), or
- Derive velocities in Python via `np.gradient(q_logical, 1/fs, axis=0)` after
  applying the inverse P-transform: `q_logical = np.linalg.solve(P_np, q_stage.T).T`
The finite-difference approach introduces noise at 20 kHz — prefer Simulink output
if velocities are needed for pre-encoder training on real data.

### Future data expansion (not yet implemented)

- **Multiple trajectories:** `System_data_list([traj1, traj2, ...])` is supported
  natively by `SSE_Interconnect.fit()`. When moving to MATLAB data or real
  experiments, use the 8 train + 1 val + 1 test trajectories defined in
  `generate_identification_experiment_without_multisine.m`.

- **Multisine excitation:** the existing MATLAB script has multisine infrastructure
  (Schroeder-phase, odd-harmonic, band-limited). For real-data identification or
  when motion-profile data gives insufficient frequency coverage, revisit. For Phase 1
  (Python-simulated, matched case), a motion profile suffices.

- **Phase 2 (LPV, no ANN):** Y-varying trajectories from nominal gantry model.
  Existing data files (`gantry_lti_train.mat`, `gantry_aug_train.mat`) both freeze
  Y at 0.3 m — new MATLAB data generation scripts are required with Y motion profiles.
  Save as `gantry_lpv_train.mat` / `gantry_lpv_val.mat`.

- **Phase 3 (LPV + ANN):** Y-varying trajectories from augmented gantry model (extra MSD).
  Same Y-varying motion profile as Phase 2 but simulated with `gantry_additional_state_2025a`.
  Save as `gantry_aug_lpv_train.mat` / `gantry_aug_lpv_val.mat`.
  MATLAB data saved as `single()` in MATLAB → float32 on Python side.

### Format (all phases)
```
System_data / System_data_with_x:
  u : (T, 3)  stage forces [F_X1, F_X2, F_Y]  [N]       float32
  y : (T, 3)  stage positions [X1, X2, Y]       [m]       float32
  x : (T, 6)  logical states [q; qdot]           [m, m/s]  float32  (Phase 1 only)
  dt: 1/20000
```

---

## Interconnect Wiring (Phase 1 reference)

```python
nx, nu, ny = 6, 3, 3
interconnect = Interconnect(nx=nx, nu=nu, ny=ny)

state_block  = Gantry_State_Block(Y_op=0.3)   # Phase 1 only: frozen Y
# state_block = Gantry_State_Block(Y_op=None)  # Phase 2+:  self-scheduled LPV
interconnect.add_block(state_block)
interconnect.connect_signals("x", state_block, "concat",  selection_matrix(list(range(nx)), nx))
interconnect.connect_signals("u", state_block, "concat")
interconnect.connect_signals(state_block, "xp", "additive", expansion_matrix(list(range(nx)), nx))

output_block = Linear_Output_Block(Cd, Dd)
interconnect.add_block(output_block)
interconnect.connect_signals("x", output_block, "concat", selection_matrix(list(range(nx)), nx))
interconnect.connect_signals("u", output_block, "concat")
interconnect.connect_signals(output_block, "y", "additive")

fit_sys = SSE_Interconnect(na=13, nb=13, interconnect=interconnect)
fit_sys.fit(train_data, val_data, epochs=30, batch_size=256,
            auto_fit_norm=True, loss_kwargs={'nf': 50}, validation_measure='sim-NRMS')
fit_sys.save_system('simulations/gantry_subnet/phase1')
```

---

## Potential Improvements

### State-supervised encoder loss
The MATLAB data includes ground-truth velocities (`x_logical[:, 3:6]`) which are never
used in the training loss. The current loss is output-only: stage position error via
`Cd_norm @ x_norm` vs `y / ystd`. Adding a state loss term would directly supervise
velocity initialization, giving the encoder a gradient signal toward the true velocities
rather than having to infer them indirectly through NF-step position error.

Expected benefit: faster encoder convergence, better initial velocity estimates,
more physically meaningful internal state trajectories.

Trade-off: depends on having state measurements (not available in real deployment).
Feasible here because MATLAB provides full `x_logical`. Requires extending
`SSE_Interconnect` loss or adding an auxiliary loss term alongside the output loss.

Revisit after Phase 1 baseline is validated.

---

## Open Questions

- **Normalisation:** Follow Jan's `Nonlinear_MSD_State_Block` pattern — store `std_x`
  and `std_u` precomputed from a short reference simulation; denormalise inside
  `deriv()`, renormalise output.

- **BPTT length `nf`:** start with `nf=50`. Tune after Phase 1.

- **Encoder history `na`, `nb`:** start with `na=nb=13` (nx*2+1).

- **HPC:** needed for Phase 3 with longer `nf`.
