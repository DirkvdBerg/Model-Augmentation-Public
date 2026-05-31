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
At frozen Y_op: N(Y_op), d(Y_op) precomputed at `__init__`; `deriv()` is pure matmul.
At LPV (Phase 3): Y = x[2]; N(Y), d(Y) computed via Horner form each step.

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
| `model_augmentation/systems/gantry_ss.py` | Gantry physical constants — single source of truth (importable by blocks.py) |
| `scripts/gantry/gantry_subnet.py` | Training script |
| `lpv_lfr_baseline/scripts/train_param_recovery.py` | **Reference only — do not import.** Shows how to inspect internal model state, plot trajectories, and evaluate a trained gantry model. Pattern to follow in `gantry_evaluate.py` / `gantry_state_comparison.py`. |
| `docs/lfr-baseline-implementation-method.md` | Justifies why z/w must be computed explicitly (resolve-and-retain argument). Supervisor requirement D-005/D-013/D-017. Validates `Gantry_State_Block` design over collapsed `A_c(Y)x + B_c(Y)u`. |

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

- **Phase 2+:** same MATLAB trajectories, different block (extra MSD mismatch).
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
