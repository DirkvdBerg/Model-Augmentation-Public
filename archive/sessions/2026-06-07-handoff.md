# Session Handoff — Dynamic Parallel Augmentation for Gantry LPV-LFR

**Last written**: 2026-05-26
_Previous sessions archived to `archive/sessions/`._

---

## Goal (one month)

Dynamic parallel augmentation trained and validated in simulation. No joint estimation. No orthogonal regularization. Encoder + ANN trained jointly, baseline parameters frozen.

---

## What exists

### Jan's augmentation framework (`model_augmentation/`)
Read `scripts/ecc_2025/msd_ndof_interconnect_dynamic.py` to understand how it is used end-to-end.

- **`Interconnect`** — manages the full state vector x, routes signals between blocks via selection matrices, detects algebraic loops. Blocks are stateless: they map `(batch, nz, 1) → (batch, nw, 1)`. The state lives in the Interconnect, not inside blocks.
- **`SSE_Interconnect`** — SUBNET training wrapper. Owns an encoder + an Interconnect. Training loop: `encoder(u_past, y_past) → x̂₀` then rolls forward `nf` steps computing `MSE(ŷ, y)`. The encoder and ANN are trained jointly.
- **`Static_ANN_Block`** — feedforward ANN, nz in, nw out. No internal state.
- **`modified_encoder_net`** — ResNet mapping flattened `[u_past (nb×nu), y_past (na×ny)] → x̂₀ (nx)`. Scales to any nu, ny, nx automatically.

### Gantry baseline (`lpv_lfr_baseline/`)
- **`LFRBaselineBlock`** (`blocks/lfr_block.py`) — the gantry LPV-LFR wrapped as a Jan-compatible Block. nz=9 (x[:6] + u[3]), nw=18 ([x_next(6), z_lfr(6), w_lfr(6)]). Uses buffers — permanently frozen, never enters optimizer. LPV scheduling Y=x[2] extracted internally — never route Y as an external signal (algebraic loop). Operates float64 internally.
- **`test_jan_compat.py`** (`tests/`) — all checks pass: forward shapes, gradient flow (BPTT 200 steps), algebraic loop detection, static ANN augmentation routing (checks C, D, E). The static parallel wiring is already tested here.
- **`train_param_recovery.py`** (`scripts/`) — parameter recovery producing a checkpoint with physical baseline parameters. This checkpoint provides the frozen baseline for augmentation training.

### Design documents
- `docs/augmentation-extra-state-design.md` — full analysis of the chosen extra state. **Primary recommendation: yaw-flex mode [φ, φ̇], n_aug=2.** Alternative: first-order lag [z], n_aug=1. Read Section 5.7 and Section 0.
- `docs/fp-augmentation-interface.md` — interface between baseline and augmentation.
- `docs/loss-function-design.md` — loss function design notes.

---

## What Hoekstra (2025) and Drenth (2025) tell us about the target structure

Read:
- `literature/augmentation/hoekstra2025_lfr-augmentation-ejc.pdf`
- `literature/lpv-lfr/drenth2025_lpv-lfr-rational.pdf`

**From Hoekstra (2025) — the augmentation structure (Eq. 4):**

The full model is one interconnected system with interconnection matrix S:

```
φ_base  ←→  S  ←→  φ_aug
```

For **dynamic parallel**, the model equations are (Hoekstra Table 1, bottom-left):

```
x̂_{k+1} = f_base(x̂_k, u_k)  +  f_aug(x̂_k, x̃_k, u_k)   ← additive, parallel
x̃_{k+1} = g_aug(x̂_k, x̃_k, u_k)                          ← extra state dynamics
ŷ_k      = h_base(x̂_k)
```

Where:
- `x̂` = baseline states (6 in our case)
- `x̃` = augmentation states (n_aug, for unmodeled flexible/coupling dynamics)
- `f_aug` = ANN correction to state update
- `g_aug` = ANN governing extra state dynamics
- The ANN receives: current baseline states, extra states, inputs — and the LFR latent z_lfr

In Jan's framework, `f_aug` and `g_aug` are implemented as a single **Dynamic_ANN_Block** that takes `[x̃, z_lfr]` as input and outputs `[x̃_next, correction to xp]`. The block is stateless; x̃ lives in the Interconnect state vector at indices `x[6:6+n_aug]`.

**From Drenth (2025) — the LPV-LFR structure:**

The baseline is already an LPV-LFR (implemented in `LFRBaselineBlock`). Drenth's JAX code and D_zw well-posedness parameterization are for identifying a full LPV-LFR from scratch. For our case, the baseline is **fixed** (frozen parameters), so Drenth's identification machinery is not needed. The LFR well-posedness is already guaranteed by M(Y) > 0 (proved in `docs/m-matrix-invertibility.md`).

---

## The 5 problems that must be solved

### Problem 1 — Augmented simulation data does not exist yet

The data-generating model (baseline LPV + extra state) must be built in Simulink and simulation data generated. Without this, nothing else can be trained or validated.

**What is needed:**
- Choose extra state: yaw-flex [φ, φ̇] (n_aug=2) or first-order lag [z] (n_aug=1). See `docs/augmentation-extra-state-design.md`.
- Implement the extra state ODE in the existing augmented Simulink model
- Generate simulation trajectories (u → y, with extra state z(t) saved as ground truth)
- Save as `.mat` files with u (N×3), y (N×3), z_true (N×n_aug)

**Blocker**: everything downstream depends on this.

---

### Problem 2 — deepSI normalizes data before it reaches the LFR block

Jan's `SSE_Interconnect` inherits from deepSI's `SS_encoder_general`. DeepSI calls `self.norm.fit(sys_data)` and `self.norm.transform(train_sys_data)` before training begins. This normalizes u and y to zero mean, unit variance. The `LFRBaselineBlock` expects physical-scale forces (N) and states (m, rad/s). Normalized forces fed into the RK4 physics produce garbage.

**Three options:**
- **Option A**: Disable deepSI normalization (`auto_fit_norm=False`) and keep everything in physical units
- **Option B**: Embed denormalization/renormalization inside `LFRBaselineBlock` (block handles the transform)
- **Option C**: Override the normalization step in a custom subclass of `SSE_Interconnect`

**This decision must be made before any training script can be written.** Ask the supervisor (Jan Hoekstra) which approach he recommends — he designed both the LFR block and SSE_Interconnect.

---

### Problem 3 — No end-to-end training script for the gantry baseline exists

`test_jan_compat.py` verifies the forward pass and gradients in isolation. There is no script that:
1. Loads gantry `.mat` data into `deepSI.System_data`
2. Instantiates `SSE_Interconnect` with `LFRBaselineBlock`
3. Calls `.fit(train_data)` and produces a training curve
4. Evaluates BFR on validation data

This baseline-only training script must exist and run cleanly before augmentation is added. It is the integration test that catches all MIMO/LPV/dtype/normalization issues.

**Specific things that can break:**
- MIMO encoder: input size = `nb*3 + na*3` (vs Jan's MSD: `nb*1 + na*1`) — check shape
- Float32/64: encoder outputs float32, LFR block casts to float64 and back — check for NaN over long rollouts
- Coordinate transform: `y` from Interconnect is in logical coordinates, training data is in stage coordinates — `S_y` must embed `P.T` (verified in Check A2 but not in a training context)
- Multi-trajectory: Jan's examples use a single `.npz` file; gantry has T1–T8 `.mat` files at different Y positions — verify `System_data_list` works with `SSE_Interconnect.fit()`

---

### Problem 4 — Dynamic_ANN_Block does not exist

Jan has `Static_ANN_Block` (feedforward, no states). The dynamic version needs:

```python
class Dynamic_ANN_Block(Block):
    def __init__(self, n_aug, n_lfr=6, n_hidden=..., n_layers=...):
        nz = n_aug + n_lfr    # input:  [x̃ (n_aug),   z_lfr (6)]
        nw = n_aug + 6        # output: [x̃_next (n_aug), correction (6)]
        super().__init__(nz, nw)
        self.net = zero_init_feed_forward_nn(nz, nw, n_hidden, n_layers)
```

The zero initialization guarantees that at training start the ANN outputs zeros — so the augmented model starts identical to the baseline. Gradient checks must confirm: (a) zero-init ANN produces same xp as baseline-only, (b) non-zero ANN weights change xp, (c) gradients flow back to ANN weights through the Interconnect.

---

### Problem 5 — Dynamic parallel wiring does not exist

The Interconnect must be configured with `nx = 6 + n_aug` and the following signal routing:

```
Interconnect(nx = 6 + n_aug, nu=3, ny=3)

Selection matrices needed:
  S_x6   : select x[:6]          from x (6+n_aug → 6)
  S_xaug : select x[6:6+n_aug]   from x (6+n_aug → n_aug)
  S_zlfr : select w_out[6:12]    from LFRBlock output (18 → 6)
  S_xp6  : expand correction     to xp[:6]  (6 → 6+n_aug, zeros elsewhere)
  S_xpaug: expand x̃_next        to xp[6:]  (n_aug → 6+n_aug, zeros elsewhere)
  S_y    : select w_out[:3] via P.T          to y (18 → 3, with coordinate transform)

Wiring:
  [S_x6 @ x,  u]          → LFRBaselineBlock (nz=9)
  LFRBlock.w_out[:6]       → xp (via S_xp_baseline)
  [S_xaug @ x, S_zlfr @ LFRBlock.w_out]  → Dynamic_ANN_Block (nz = n_aug+6)
  Dynamic_ANN_Block.out[:n_aug]  → xp (via S_xpaug)
  Dynamic_ANN_Block.out[n_aug:]  → xp (via S_xp6, additive)
  LFRBlock.w_out via S_y   → y
```

Write `build_dynamic_augmented_interconnect(n_aug)` function returning the configured Interconnect.

---

## Clear bullet-point goals for one month

**Week 1 — Prerequisites**
- [ ] Decide: yaw-flex [φ, φ̇] or first-order [z] as extra state. Set n_aug.
- [ ] Implement extra state in Simulink, generate simulation data (u, y, z_true as .mat)
- [ ] Decide normalization strategy (Options A/B/C above) — ask Jan Hoekstra

**Week 2 — Baseline pipeline working**
- [ ] Write data loader: `.mat` → `deepSI.System_data(u, y)` MIMO, multi-trajectory
- [ ] Write baseline-only training script: `SSE_Interconnect` + `LFRBaselineBlock`, `.fit()`, BFR evaluation
- [ ] Verify: training loss decreases, baseline BFR on simulation data matches known simulation output

**Week 3 — Dynamic parallel structure**
- [ ] Write `Dynamic_ANN_Block` (nz=n_aug+6, nw=n_aug+6, zero-initialized)
- [ ] Write `build_dynamic_augmented_interconnect(n_aug)` with correct selection matrices
- [ ] Verify: zero-init ANN matches baseline-only output; gradients flow to ANN weights; BFR evaluates

**Week 4 — Augmented training and validation**
- [ ] Train dynamic parallel on augmented simulation data
- [ ] BFR improves over baseline-only
- [ ] Compare learned x̃ trajectory to true z_true from simulation
- [ ] Confirm baseline parameters unchanged (still frozen — check buffers not accidentally modified)

---

## Open blockers (carried forward)
- Sample rate: D-012 — 16 kHz (main.m) vs 20 kHz (ETEL spec), unresolved
- Float32 acceptability: unresolved
- MIMO decorrelation: declared limitation
