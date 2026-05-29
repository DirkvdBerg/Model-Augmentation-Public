# Gantry SubNet Augmentation — Implementation Plan

## Goal

Integrate the dual-gantry LPV baseline into Jan Hoekstra's `SSE_Interconnect` framework
to enable data-driven model augmentation. Start from Jan's ECC 2025 example, adapt it
step-by-step to the gantry, and validate each phase before adding complexity.

All data is **simulated** throughout. Real measured gantry data is a later step.

### Code location constraint

All gantry SubNet code lives in exactly two places:

| What | Where |
|------|-------|
| Physics constants + discrete LTI matrices | `scripts/gantry/gantry_ss.py` |
| Gantry-specific block classes | `model_augmentation/fit_systems/gantry_blocks.py` |
| Training scripts | `scripts/gantry/gantry_subnet.py` (and phase variants) |

**No imports from `lpv_lfr_baseline/`** in any of the above. Physics constants are
hardcoded in `gantry_ss.py`. The `lpv_lfr_baseline/` module is a separate research
implementation that must not be coupled to the SubNet pipeline.

---

### Mismatch strategy

| Phase | Data source | Baseline model | Mismatch |
|-------|-------------|----------------|----------|
| 1 (MIMO LTI) | Simulated from same LTI model | LTI gantry (frozen Y) | None — NRMS → 0 is the sanity check |
| 2 (augmentation) | Simulated from gantry + extra MSD on payload mass | LTI gantry without extra MSD | Known, controlled mismatch — ANN learns to correct it |
| 3 (LPV) | Same as Phase 2 | LPV gantry without extra MSD | LPV variation + extra MSD residual |

The extra MSD on the payload mass is a clean, controlled model mismatch — same strategy
as the ECC 2025 MSD example (ideal vs approximate parameters). The baseline doesn't
know about the extra MSD; the augmentation block must learn to compensate for it.

---

## System Summary

| Property | Value |
|----------|-------|
| Inputs `u` | 3 (stage forces: F_X1, F_X2, F_Y) |
| Outputs `y` | 3 (stage positions: X1, X2, Y) |
| States `x` | 6 (logical: q_logical, qdot_logical) |
| Physics | Continuous-time LPV, RK4 integrated (Phase 3) |
| Scheduling variable | Y = x[2] (Y position) |
| Sampling rate | 20 kHz |

**Why no SISO phase:** X1 and X2 are mechanically coupled — common mode drives X
translation, differential mode drives rotation (mechanically limited). A SISO test
on a single X channel has no physical meaning. F_Y→Y is decoupled but provides
little additional value over the full MIMO test. Start at MIMO directly.

---

## Starting Point: Copy from ECC 2025

```
scripts/ecc_2025/msd_ndof_interconnect_dynamic.py
    → scripts/gantry/gantry_subnet.py
```

Replace the MSD-specific sections in order: data → model matrices → blocks → wiring.

---

## Phase Ordering Rationale

1. **MIMO LTI** — validate full 3×3 pipeline end-to-end; NRMS → 0 is achievable
2. **MIMO LTI + augmentation** — add ANN block with known mismatch; validate core contribution
3. **MIMO LPV** — swap `Linear_State_Block` for `CT_RK4_State_Block`; physics now self-schedules on Y

Augmentation before LPV: (a) augmentation is the research goal — validate it on the
simplest baseline first; (b) if augmentation fails on LTI, the cause is isolated from
LPV dynamics; (c) the frozen-Y LTI error is itself real model error the ANN can learn.

---

## General Failure Modes (all phases)

| Symptom | Most likely cause |
|---------|-------------------|
| Flat loss from epoch 0 | Data format wrong, normalization broken, or block wiring disconnected — stop and fix |
| NaN loss | float32/float64 mismatch at physics boundary, or LR too high |
| Loss decreases then immediately plateaus | Encoder `na`/`nb` too short |
| Simulation worse than zero-state init | Encoder making things worse — wiring or normalization error |
| NRMS > 1.0 | Model worse than predicting the mean — something fundamental is broken |

---

## Phases

### Phase 1 — MIMO LTI (validate the pipeline)

**Goal:** get the full 3×3 SubNet pipeline running and saving. No augmentation, no LPV.

**System:** nu=3, ny=3, nx=6.

**Approach:**
- Compute frozen `Ad` (6×6), `Bd` (6×3) at Y=0 via augmented `matrix_exp` (ZOH,
  handles singular `Ac` because Y has no spring) → `Linear_State_Block(Ad, Bd)`
- Output block: `Linear_Output_Block(Cd, Dd)` where `Cd` maps x(6) → stage positions(3)
- All matrices computed in `gantry_ss.py` from hardcoded physics constants; no lpv_lfr_baseline import

**What to build:**
1. `scripts/gantry/gantry_ss.py` — standalone: physics constants + `gantry_discrete_ss(Y_op)` returning `(Ad, Bd, Cd, Dd)`
2. **Data generation** — simulate Phase 1 MIMO trajectories using `CT_RK4_State_Block`
   forward pass (frozen Y=0) or standalone numpy RK4 in `gantry_ss.py`; pack into
   `System_data(u=u_stage, y=q1_stage, dt=1/fs)`
3. **Interconnect wiring** — `Interconnect(nx=6, nu=3, ny=3)` with `Linear_State_Block` + `Linear_Output_Block`
4. **SSE_Interconnect + fit** — adapt ECC 2025 training call

**Success criterion:** script runs, trains, saves; NRMS approaches near-zero.

**Verification checklist:**
- [ ] Loss decreases over epochs (encoder learning)
- [ ] Loss does not go NaN
- [ ] `fit_sys.simulate(val_data)` returns ŷ of shape `(T, 3)`
- [ ] Per-channel NRMS reported separately (X1, X2, Y)
- [ ] NRMS → near zero after sufficient epochs (data generated from same model)
- [ ] Encoder-initialized simulation beats zero-state (x̂0=0) initialization
- [ ] **One channel flat, others learn** → data shape error; check `System_data` construction
- [ ] **All channels flat** → MIMO encoder fix (`self.ny` line 369 in `interconnect.py`) not active
- [ ] X1/X2 NRMS worse than Y — expected (frozen Y=0 is a poor X operating point), not a bug
- [ ] **NRMS stays high despite low loss** → normalization mismatch between encoder and physics block

---

### Phase 2 — MIMO LTI + augmentation

**Goal:** add the ANN augmentation block with a controlled mismatch. Core research contribution.

**Introduce mismatch:** generate training data from gantry + extra MSD on payload mass.
`Linear_State_Block` does not know about the MSD. ANN must learn to compensate.

**State block:** unchanged `Linear_State_Block` from Phase 1.

**Augmentation block:** `Static_ANN_Block` in parallel to the state update:
```python
interconnect.connect_signals("x",  aug_block, "concat", selection_matrix(list(range(nx)), nx))
interconnect.connect_signals("u",  aug_block, "concat")
interconnect.connect_signals(aug_block, "xp", "additive", expansion_matrix(list(range(nx)), nx))
```

**Success criterion:** augmented NRMS < LTI-only NRMS from Phase 1 on validation data.

**Verification checklist:**
- [ ] Loss lower than Phase 1 (ANN contributing)
- [ ] Per-channel NRMS improvement over Phase 1 on validation data
- [ ] ANN output magnitude reasonable (not exploding)
- [ ] Regularization loss (`param_loss`) tracked alongside simulation loss
- [ ] **Loss lower on train but not val** → ANN overfitting; reduce size or add regularization
- [ ] **No improvement over Phase 1** → ANN not contributing; check additive connection to `xp`

---

### Phase 3 — MIMO LPV

**Goal:** replace frozen LTI block with RK4-integrated self-scheduled LPV physics.

**Block swap:** `Linear_State_Block(Ad, Bd)` → `CT_RK4_State_Block`

`CT_RK4_State_Block` lives in `model_augmentation/fit_systems/gantry_blocks.py` and
subclasses `Discrete_Nonlinear_Function_Block` following the `Nonlinear_MSD_State_Block`
pattern in Jan's `blocks.py`:

```
nonlinear_function(z):           z = [x(6), u_stage(3)], normalised
    deriv(x_phys, u_phys):       continuous-time xdot from gantry ODE
                                 M(Y)*q_ddot + C*q_dot + K*q = P @ u_stage
    RK4 integration (4 substeps) → x_next (normalised)
```

- `deriv()` computes `xdot` directly from the second-order ODE — no lpv_lfr_baseline import
- Physics constants (M0, M1, M2, C, K, P, ts) hardcoded in the block or passed at init from `gantry_ss.py`
- All other wiring, augmentation block, encoder, training loop unchanged from Phase 2

**Note on speed:** RK4 backprop is 4× more compute per step than a linear block.
Use HPC for longer training runs.

**Success criterion:** LPV NRMS < LTI+augmentation NRMS from Phase 2.

**Verification checklist:**
- [ ] Training stable (no divergence from Y self-scheduling)
- [ ] NRMS improves over Phase 2 LTI+augmentation baseline
- [ ] Per-channel improvement largest on Y channel (Y is the scheduling variable)
- [ ] **Loss diverges** → Y self-scheduling unstable; freeze Y at mean first to diagnose
- [ ] **No improvement over LTI** → LPV variation small at this operating regime; expected for small Y range

---

## Key Reference Files

| File | Role |
|------|------|
| `scripts/ecc_2025/msd_ndof_interconnect_dynamic.py` | Template to copy and adapt |
| `model_augmentation/fit_systems/blocks.py` | Block base classes; `Nonlinear_MSD_State_Block` as RK4 pattern for Phase 3 |
| `model_augmentation/fit_systems/interconnect.py` | `SSE_Interconnect`, `Interconnect`, `modified_encoder_net` |
| `model_augmentation/fit_systems/gantry_blocks.py` | `CT_RK4_State_Block` (to be created) |
| `scripts/gantry/gantry_ss.py` | Physics constants + `gantry_discrete_ss(Y_op)` (to be created) |
| `scripts/gantry/gantry_subnet.py` | Main training script (to be created from ECC 2025 template) |

---

## Data Pipeline

```
Phase 1/2 (LTI baseline):
  Simulate with frozen Y using CT_RK4_State_Block or numpy RK4 in gantry_ss.py
  → u_stage (T, 3), q1_stage (T, 3)
  → System_data(u=u_stage, y=q1_stage, dt=1/fs)

Phase 2 (mismatch):
  Same simulation + extra MSD on payload mass
  → data looks different, baseline block unchanged

Phase 3 (LPV):
  Same data as Phase 2; only the block changes
```

Multiple trajectories → `System_data_list([traj1, traj2, ...])`

---

## Interconnect Wiring (Phase 1 MIMO reference)

```python
nx, nu, ny = 6, 3, 3
interconnect = Interconnect(nx=nx, nu=nu, ny=ny)

# Physics state block
state_block = Linear_State_Block(Ad, Bd)        # Phase 1/2
# state_block = CT_RK4_State_Block(...)         # Phase 3
interconnect.add_block(state_block)
interconnect.connect_signals("x",  state_block, "concat",  selection_matrix(list(range(nx)), nx))
interconnect.connect_signals("u",  state_block, "concat")
interconnect.connect_signals(state_block, "xp", "additive", expansion_matrix(list(range(nx)), nx))

# Output block
output_block = Linear_Output_Block(Cd, Dd)
interconnect.add_block(output_block)
interconnect.connect_signals("x",  output_block, "concat", selection_matrix(list(range(nx)), nx))
interconnect.connect_signals("u",  output_block, "concat")
interconnect.connect_signals(output_block, "y", "additive")

fit_sys = SSE_Interconnect(na=13, nb=13, interconnect=interconnect)
fit_sys.fit(train_data, val_data, epochs=30, batch_size=256,
            auto_fit_norm=True, loss_kwargs={'nf': 50}, validation_measure='sim-NRMS')
fit_sys.save_system('simulations/gantry_subnet/phase1_mimo_lti')
```

---

## Open Questions

- **Normalization:** State in physical units for Phase 1/2 (no state data to compute std from).
  Scale only `Bd` and `Cd` by `sigma_u` and `sigma_y` from training data. For Phase 3
  (RK4 block), follow Jan's pattern: store `std_x`, `std_u` precomputed from a reference
  simulation run; denormalize inside `deriv()`, renormalize output.

- **BPTT length `nf`:** start with `nf=50`. MSD example uses `nf=200`.
  At 20 kHz, longer horizons are expensive — tune after Phase 1.

- **Encoder history `na`, `nb`:** start with `na=nb=13` (nx*2+1).

- **HPC:** for Phase 3 with long `nf`, request CPU time on `hpc.tue.nl`.
