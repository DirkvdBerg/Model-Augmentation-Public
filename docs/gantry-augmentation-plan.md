# Gantry SubNet Augmentation — Implementation Plan

## Goal

Integrate the dual-gantry LPV-LFR baseline into Jan Hoekstra's `SSE_Interconnect` framework
to enable data-driven model augmentation. Start from Jan's ECC 2025 example, adapt it
step-by-step to the gantry, and validate each phase before adding complexity.

---

## System Summary

| Property | Value |
|----------|-------|
| Inputs `u` | 3 (stage forces: F_X1, F_X2, F_Y) |
| Outputs `y` | 3 (stage positions: X1, X2, Y) |
| States `x` | 6 (logical: q_logical, qdot_logical) |
| Physics | Continuous-time LPV-LFR, RK4 integrated |
| Scheduling variable | Y = x[2] (Y position) |
| Sampling rate | 20 kHz (decimated during training) |

---

## Starting Point: Copy from ECC 2025

Copy Jan's end-to-end script and adapt it — do not build from scratch.

```
scripts/ecc_2025/msd_ndof_interconnect_dynamic.py
    → scripts/gantry/gantry_subnet.py
```

This script already has: data loading → block wiring → `SSE_Interconnect.fit()` → save.
Replace the MSD-specific parts section by section.

---

## Phase Ordering Rationale

1. **SISO LTI** — minimal moving parts, validate pipeline end-to-end
2. **MIMO LTI** — extend dimensions, test MIMO data format and encoder
3. **MIMO LTI + augmentation** — add ANN block, validate the core contribution
4. **MIMO LPV** — swap `Linear_State_Block` for `CT_RK4_State_Block`, keep everything else

Augmentation comes before LPV because: (a) augmentation is the research goal and should be
validated on the simplest baseline first; (b) the LTI model error (Y-dependent dynamics) is
real model error — the ANN compensating for it is a meaningful experiment; (c) if augmentation
training fails on LTI, the cause is isolated to the augmentation, not the LPV dynamics.

---

## General Failure Modes (all phases)

Before phase-specific checks, these apply everywhere:

| Symptom | Most likely cause |
|---------|-------------------|
| Flat loss from epoch 0 | Data format wrong, normalization broken, or block wiring disconnected — stop and fix before continuing |
| NaN loss | float32/float64 mismatch at physics boundary, or LR too high |
| Loss decreases then immediately plateaus | Encoder `na`/`nb` too short, or physics so wrong x̂0 can't help |
| Simulation worse than zero-state init | Encoder making things worse — wiring or normalization error |
| NRMS > 1.0 | Model worse than predicting the mean — something fundamental is broken |

---

## Phases

### Phase 1 — SISO LTI (validate the pipeline)

**Goal:** get the full SubNet pipeline running and saving. No augmentation, no LPV.

**System:** 1 input (F_Y), 1 output (Y position), nx=6.

**Approach:**
- Pre-discretize the gantry at a fixed Y operating point (Y=0) using ZOH or `matrix_exp`
  to get frozen `Ad`, `Bd` matrices → `Linear_State_Block(Ad, Bd)`
- Output equation: `C` selects Y from the state → `Linear_Output_Block(C, zeros)`
- No augmentation block

**What to build:**
1. **Data loader** — load one .mat trajectory, extract `u[:, 2]` (F_Y) and `q1[:, 2]` (Y),
   pack into `deepSI.System_data(u=..., y=..., dt=1/fs)`
2. **Frozen linear block** — compute `Ad`, `Bd` at Y=0 from existing physics constants,
   instantiate `Linear_State_Block(Ad, Bd)` and `Linear_Output_Block(C, D)`
3. **Interconnect wiring** — `Interconnect(nx=6, nu=1, ny=1)` following the standard pattern
   from `fp-augmentation-interface.md`
4. **SSE_Interconnect + fit** — adapt the ECC 2025 training call

**Success criterion:** script runs, trains for 2 epochs, saves a checkpoint without errors.

**Verification checklist:**
- [ ] Loss decreases over epochs (encoder is learning)
- [ ] Loss does not go NaN
- [ ] `fit_sys.simulate(val_data)` returns ŷ of correct shape `(T,)`
- [ ] Plot ŷ vs measured Y — trajectory shape should be followed
- [ ] NRMS < 1.0 (better than predicting the mean)
- [ ] NRMS comparable to `simulate_frozen()` reference at same Y operating point
- [ ] Encoder-initialized simulation beats zero-state (x̂0=0) initialization

---

### Phase 2 — MIMO LTI (nu=3, ny=3)

**Goal:** extend to the full 3×3 gantry system, still frozen LTI.

**Changes from Phase 1:**
- Data loader: use full `u` (T, 3) and `q1` (T, 3)
- `Interconnect(nx=6, nu=3, ny=3)`
- State block: `Linear_State_Block(Ad, Bd)` with `Bd` now (6×3)
- Output block: `Linear_Output_Block(C, D)` where C maps x(6) → q1(3) in stage coordinates

**Data format note (meeting):** if something fails here, check `System_data` shape first.
DeepSI expects `u.shape = (T, nu)` and `y.shape = (T, ny)` for MIMO.

**Success criterion:** MIMO pipeline trains and saves; per-channel loss tracked.

**Verification checklist:**
- [ ] Loss decreases (MIMO encoder learning all 3 channels)
- [ ] `fit_sys.simulate(val_data)` returns ŷ of shape `(T, 3)`
- [ ] Per-channel NRMS reported separately (X1, X2, Y)
- [ ] Per-channel NRMS comparable to `simulate_frozen()` per-channel reference
- [ ] **One channel flat, others learn** → data shape error on that channel; check `System_data` construction
- [ ] **All channels flat** → MIMO encoder fix (`self.ny` line in `interconnect.py`) not active — verify line 369
- [ ] X1/X2 NRMS worse than Y — expected (frozen Y=0 is a poor operating point for X dynamics), not a bug

---

### Phase 3 — MIMO LTI + augmentation (dynamic parallel)

**Goal:** add the ANN augmentation block. This is the core research contribution.

**State block:** unchanged `Linear_State_Block` (still frozen LTI).

**Augmentation block:** `Static_ANN_Block` in parallel to the state update:
```python
interconnect.connect_signals("x",  aug_block, "concat", selection_matrix(list(range(nx)), nx))
interconnect.connect_signals("u",  aug_block, "concat")
interconnect.connect_signals(aug_block, "xp", "additive", expansion_matrix(list(range(nx)), nx))
```

Later upgrade path: `Dynamic_ANN_Block` (has its own internal state) for dynamic augmentation.

**Success criterion:** augmented model NRMS < LTI-only NRMS from Phase 2 on validation data.

**Verification checklist:**
- [ ] Loss lower than Phase 2 (ANN is contributing)
- [ ] Per-channel NRMS improvement over LTI-only baseline on validation data
- [ ] ANN output magnitude is reasonable (not exploding)
- [ ] Regularization loss (`param_loss`) tracked alongside simulation loss
- [ ] **Loss lower on train but not val** → ANN overfitting; reduce network size or add regularization
- [ ] **No improvement over Phase 2** → ANN not contributing; check additive connection to `xp`

---

### Phase 4 — MIMO LPV

**Goal:** replace frozen LTI block with RK4-integrated self-scheduled LPV physics.

**Block swap only:** `Linear_State_Block(Ad, Bd)` → `CT_RK4_State_Block`

`CT_RK4_State_Block` subclasses `Discrete_Nonlinear_Function_Block` following the
`Nonlinear_MSD_State_Block` pattern in `blocks.py`:
- `nonlinear_function(z)`: receives `z = [x(6), u(3)]`, calls one `rk4_step` with
  `Y_override=None` (self-scheduled), returns `x_next (6)`
- All other wiring, augmentation block, encoder, and training loop unchanged

**Note on speed:** RK4 backprop is 4× more compute per step than a linear block.
This is the discretization cost flagged in the meeting. Use HPC for longer training runs.

**Success criterion:** LPV model NRMS < LTI model NRMS (more accurate physics).

**Verification checklist:**
- [ ] Training stable (no divergence from LPV self-scheduling)
- [ ] NRMS improves over Phase 3 LTI+augmentation baseline
- [ ] Per-channel improvement largest on Y channel (Y is the scheduling variable)
- [ ] **Loss diverges** → Y self-scheduling unstable; try `Y_override` at mean Y first to diagnose
- [ ] **No improvement over LTI** → LPV variation small at this operating regime; expected for small Y range

---

## Key Reference Files

| File | Role |
|------|------|
| `scripts/ecc_2025/msd_ndof_interconnect_dynamic.py` | Template to copy |
| `docs/fp-augmentation-interface.md` | Block interface contract + wiring patterns |
| `model_augmentation/fit_systems/blocks.py` | Block base classes + `Nonlinear_MSD_State_Block` as RK4 pattern |
| `model_augmentation/fit_systems/interconnect.py` | `SSE_Interconnect`, `Interconnect`, `modified_encoder_net` |
| `lpv_lfr_baseline/core/lfr_simulate.py` | `rk4_step`, `simulate`, `lfr_forward` |
| `lpv_lfr_baseline/core/physics.py` | Physical constants, `build_poly_constants` |
| `lpv_lfr_baseline/scripts/precompute.py` | Reference for data loading from .mat files |

---

## Data Pipeline (per phase)

```
.mat file
  └─ scipy.io.loadmat(...)
       ├─ u_stage  (T, 3)   stage forces
       └─ q1       (T, 3)   stage positions

Phase 1 (SISO):
  u_siso = u_stage[:, 2]    # F_Y only
  y_siso = q1[:, 2]         # Y position only
  train_data = System_data(u=u_siso, y=y_siso, dt=1/fs)

Phase 2+ (MIMO):
  train_data = System_data(u=u_stage, y=q1, dt=1/fs)
```

Multiple trajectories → `System_data_list([traj1, traj2, ...])`

---

## Interconnect Wiring (Phase 1 reference)

```python
nx, nu, ny = 6, 1, 1
interconnect = Interconnect(nx=nx, nu=nu, ny=ny)

# Physics state block
state_block = Linear_State_Block(Ad, Bd)     # Phase 1/2
# state_block = CT_RK4_State_Block(ts, ...)  # Phase 4
interconnect.add_block(state_block)
interconnect.connect_signals("x",  state_block, "concat",  selection_matrix(list(range(nx)), nx))
interconnect.connect_signals("u",  state_block, "concat")
interconnect.connect_signals(state_block, "xp", "additive", expansion_matrix(list(range(nx)), nx))

# Output block
output_block = Linear_Output_Block(C, D)
interconnect.add_block(output_block)
interconnect.connect_signals("x",  output_block, "concat", selection_matrix(list(range(nx)), nx))
interconnect.connect_signals("u",  output_block, "concat")
interconnect.connect_signals(output_block, "y", "additive")

# Wrap in SSE_Interconnect
fit_sys = SSE_Interconnect(na=10, nb=10, interconnect=interconnect)
fit_sys.fit(train_data, val_data, epochs=2, batch_size=256)
fit_sys.save_system('simulations/gantry_subnet/phase1.pt')
```

---

## Open Questions

- **Normalization:** For Phase 1/2 (linear block), use `normalize_linear_ss_matrices()` from
  `fp-augmentation-interface.md`. For Phase 4 (RK4 block), normalization is handled at the
  encoder level (DeepSI normalizes u/y to unit variance); physics operates in physical units.

- **BPTT length `nf`:** start with `nf=50`. The MSD example uses `nf=200`.
  At 20 kHz with RK4 sub-steps, longer horizons are expensive — tune after Phase 1.

- **Encoder history `na`, `nb`:** start with `na=nb=10` (same as MSD example).

- **HPC:** for Phase 4 with long `nf`, request CPU time on `hpc.tue.nl`
  (action item from meeting).
