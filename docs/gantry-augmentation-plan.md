# Gantry SubNet Augmentation — Implementation Plan

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

**Current implementation covers:** fixed-physics RK4 block at one frozen Y, basic
parallel ANN augmentation, no orthogonality, no joint estimation. This is intentional —
validate the pipeline before adding complexity.

---

## Design Principle: Expansion-Friendly

Every choice made now must not block the end goal:

| Choice now | Why it keeps the path open |
|------------|---------------------------|
| `Gantry_State_Block` from Phase 1 (not `Linear_State_Block`) | Adding trainable params later = change constants to `nn.Parameter`; no structural rewrite |
| Physics constants as plain attributes in block `__init__` | Mirrors `Parameterized_MSD_State_Block` — swap to `nn.Parameter` + `param_loss()` for joint estimation |
| `Y_op` parameter on block (`float` → `None`) | Phase 1 frozen Y, Phase 3 LPV — one-line change |
| Parallel ANN wiring from Phase 2 | Orthogonality constraint adds a loss term on top — wiring unchanged |
| Jan's block/interconnect structure throughout | LFR routing = add more blocks and connect_signals calls — no redesign |

---

## Code Location Constraint

```
model_augmentation/fit_systems/blocks.py   ← add Gantry_State_Block here
                                             follows Nonlinear_MSD_State_Block pattern
                                             Jan's file — extend it, don't treat as read-only

scripts/gantry/
  gantry_ss.py               — gantry physical constants (single source of truth)
  gantry_subnet.py           — training script
  gantry_evaluate.py         — evaluation
  gantry_state_comparison.py — internal signal inspection
```

`gantry_subnet.py` imports `Gantry_State_Block` from `model_augmentation.fit_systems.blocks`
— same import as any other block. `gantry_ss.py` imports constants from `gantry_ss.py`.

**No imports from `lpv_lfr_baseline/`** in any gantry scripts or blocks. Physics constants
are hardcoded in `gantry_ss.py` and imported from there. The `lpv_lfr_baseline/` module
is a separate research implementation and must not be coupled to the SubNet pipeline.

---

## Mismatch Strategy

| Phase | Data source | Baseline model | Mismatch |
|-------|-------------|----------------|----------|
| 1 | Simulated from same frozen-Y RK4 model | Same frozen-Y RK4 block | None — NRMS → 0 is the sanity check |
| 2 | Simulated from gantry + extra MSD on payload | Frozen-Y RK4, no extra MSD | Known controlled mismatch |
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

1. **MIMO, frozen Y, no augmentation** — pipeline sanity check; NRMS → 0 is achievable
2. **MIMO, frozen Y, + ANN augmentation** — validate core contribution with known mismatch
3. **MIMO, LPV (self-scheduled Y)** — one-line block change; validate LPV improvement

Augmentation before LPV: (a) augmentation is the research goal — validate on simplest
baseline first; (b) frozen-Y error is real model error the ANN can learn; (c) if
augmentation fails on frozen Y, cause is isolated from LPV scheduling.

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

**Block:** `Gantry_State_Block(Y_op=0.3)` — physics frozen at one operating point.
RK4 integration used from the start so the block needs no structural change later.

**What to build:**
1. `scripts/gantry/gantry_ss.py` — gantry physical constants
2. `Gantry_State_Block` added to `model_augmentation/fit_systems/blocks.py`,
   following `Nonlinear_MSD_State_Block` pattern exactly, importing constants from `gantry_ss.py`
2. **Data generation** — simulate with same block (frozen Y) to get matched train/val data
3. **Interconnect wiring** — `Interconnect(nx=6, nu=3, ny=3)` + `Linear_Output_Block`
4. **SSE_Interconnect + fit** — adapt ECC 2025 training call

**Not yet implemented (end goal):**
- Physical parameters are fixed (not `nn.Parameter`) — joint estimation deferred
- No orthogonality constraint
- No LFR signal routing

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

### Phase 2 — MIMO, frozen Y, + ANN augmentation

**Goal:** add ANN augmentation with controlled mismatch. Core research contribution.

**Mismatch:** training data from gantry + extra MSD on payload mass.
`Gantry_State_Block` unchanged — it does not know about the extra MSD.

**Augmentation block:** `Static_ANN_Block` in parallel (same wiring as ECC 2025):
```python
interconnect.connect_signals("x",  aug_block, "concat", selection_matrix(list(range(nx)), nx))
interconnect.connect_signals("u",  aug_block, "concat")
interconnect.connect_signals(aug_block, "xp", "additive", expansion_matrix(list(range(nx)), nx))
```

**Not yet implemented (end goal):**
- Joint estimation: physics parameters still fixed
- Orthogonality: ANN can still learn baseline-captured dynamics
- LFR routing: ANN connects to state, not to LFR latent variables

**Success criterion:** augmented NRMS < Phase 1 NRMS on validation data.

**Verification checklist:**
- [ ] Loss lower than Phase 1
- [ ] Per-channel NRMS improvement on validation data
- [ ] ANN output magnitude reasonable
- [ ] **No improvement** → check additive connection to `xp`
- [ ] **Overfitting** → reduce ANN size or add regularisation

---

### Phase 3 — MIMO LPV (self-scheduled Y)

**Goal:** unlock LPV. One-line block change.

**Block swap:** `Gantry_State_Block(Y_op=0.3)` → `Gantry_State_Block(Y_op=None)`

Inside `deriv()`: `Y = x_phys[:, 2]` instead of the fixed value. Everything else
— wiring, augmentation, encoder, training loop — unchanged.

**Not yet implemented (end goal):**
- Joint estimation still deferred
- Orthogonality still deferred
- LFR routing still deferred

**Speed note:** RK4 backprop is 4× more compute than a linear block. Use HPC for
longer `nf`.

**Success criterion:** LPV NRMS < Phase 2 NRMS.

**Verification checklist:**
- [ ] Training stable (no divergence from self-scheduling)
- [ ] NRMS improves over Phase 2
- [ ] Largest improvement on Y channel (scheduling variable)
- [ ] **Loss diverges** → self-scheduling unstable; try frozen Y at mean first

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
| `scripts/gantry/gantry_ss.py` | Gantry physical constants — single source of truth |
| `scripts/gantry/gantry_subnet.py` | Training script |
| `lpv_lfr_baseline/scripts/train_param_recovery.py` | **Reference only — do not import.** Shows how to inspect internal model state, plot trajectories, and evaluate a trained gantry model. Pattern to follow in `gantry_evaluate.py` / `gantry_state_comparison.py`. |

---

## Data Pipeline

```
All phases: simulate with Gantry_State_Block forward pass
  Phase 1: frozen Y, no extra MSD → matched data, NRMS → 0
  Phase 2: frozen Y, + extra MSD on payload mass → controlled mismatch
  Phase 3: same data as Phase 2, block changes only

Format: System_data(u=u_stage (T,3), y=q1_stage (T,3), dt=1/20000)
Multiple trajectories: System_data_list([traj1, traj2, ...])
```

---

## Interconnect Wiring (Phase 1 reference)

```python
nx, nu, ny = 6, 3, 3
interconnect = Interconnect(nx=nx, nu=nu, ny=ny)

state_block  = Gantry_State_Block(Y_op=0.3)   # Phase 1/2: frozen Y
# state_block = Gantry_State_Block(Y_op=None)  # Phase 3:   self-scheduled LPV
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

## Open Questions

- **Normalisation:** Follow Jan's `Nonlinear_MSD_State_Block` pattern — store `std_x`
  and `std_u` precomputed from a short reference simulation; denormalise inside
  `deriv()`, renormalise output.

- **BPTT length `nf`:** start with `nf=50`. Tune after Phase 1.

- **Encoder history `na`, `nb`:** start with `na=nb=13` (nx*2+1).

- **HPC:** needed for Phase 3 with longer `nf`.
