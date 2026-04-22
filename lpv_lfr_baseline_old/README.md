# LPV-LFR Baseline — Standalone Implementation

## Purpose

This folder contains a self-contained Python implementation of the dual-gantry
LPV-LFR baseline derived in `LPV/LFR-derivation-supervisor.tex`.

The physics (`lfr_forward.py`, `lfr_simulate.py`) is self-contained and independent of
Jan's framework. The integration layer (`lfr_block.py`, `test_jan_compat.py`) wraps the
physics as a `Block` subclass compatible with Jan's `Interconnect`. All phases are complete
and validated.

---

## Derivation reference

The LPV-LFR realization is derived in `LPV/LFR-derivation-supervisor.tex`. Key results:

**Scheduling variable**: `Y` — payload Y-position. Quasi-LPV: Y = x[2] is a system state.

**Scheduling block**: `Δ(Y) = Y·I₆` — one scalar Y repeated across 6 latent channels.

**Latent variables** (standard z/w notation per Roland Tóth):
```
z = [z₁; z₂]    z₁ := q̈  (first latent variable, defined directly)
w = [w₁; w₂]    w₁  = Y·z₁ = z₂
                 w₂  = Y·z₂ = Y²·z₁
```
Note: z₂ = w₁ (the second z-channel equals the first w-channel). The chain is:
```
z₁ →^Y w₁ = z₂ →^Y w₂
```

**Mass matrix decomposition**: `M(Y) = M₀ + M₁·Y + M₂·Y²` where M₀, M₁, M₂ are
constant matrices derived from the physical parameters.

**Constant G matrix** (all entries built from M₀⁻¹, precomputed once):

```
         x              w              u
ẋ  [ Ax          ]  [ Bw          ]  [ Bu  ]
z  [ Cz          ]  [ Dzw         ]  [ Dzu ]
y  [ Cy          ]  [ 0           ]  [ 0   ]

Ax  = [0,       I₃      ]
      [-M₀⁻¹K, -M₀⁻¹C  ]

Bw  = [0,        0       ]
      [-M₀⁻¹M₁, -M₀⁻¹M₂]

Bu  = [0   ]
      [M₀⁻¹]

Cz  = [-M₀⁻¹K, -M₀⁻¹C]
      [0,       0      ]

Dzw = [-M₀⁻¹M₁, -M₀⁻¹M₂]
      [I₃,       0       ]

Dzu = [M₀⁻¹]
      [0    ]

Cy  = [I₃, 0]
```

Collapsing the internal loop recovers `M(Y)⁻¹` exactly — verified algebraically in the
derivation.

---

## Implementation method: resolve-and-retain

The G matrix has `Dzw ≠ 0`, which creates an algebraic loop if G and Δ(Y) are wired
as separate runtime blocks. Jan's `SSE_Interconnect` rejects algebraic loops via assertion.

The method used here resolves the loop analytically in a **forward sequence**, while
retaining z and w as explicit tensors (not discarding them):

```
Step 1:  fnet = [-K  -C]·x + u
Step 2:  z₁   = M(Y)⁻¹ · fnet              ← loop resolved analytically (z₁ := q̈)
Step 3:  z₂   = Y · z₁  (= w₁)
         w₂   = Y · z₂
Step 4:  z    = [z₁; z₂]
         w    = [z₂; w₂]                   ← note: w₁ = z₂
Step 5:  ẋ    = Ax·x + Bw·w + Bu·u         ← G matrix applied explicitly
Step 6:  integrate with RK4, step = ts
Step 7:  y    = Cy·x
```

This is a **partial resolution** — the algebraic relation is resolved, but z and w survive
as runtime quantities. This is structurally stronger than a full collapse to `A_c(Y)·x + B_c(Y)·u`
because z and w are available for the augmentation to connect to.

See `docs/lfr-baseline-implementation-method.md` for full justification and open questions.

---

## Discretization

RK4 with fixed step `ts = 1/fs`. No ZOH, no pre-discretization. Consistent with D-018.

At 16kHz (ts = 62.5µs), RK4 and ZOH produce very similar trajectories. Validation against
MATLAB (ZOH) will show small but bounded differences — this is expected and acceptable.

---

## Validation strategy

Three checks, in order of strength:

1. **G matrix algebraic check** (`validate_lfr.py`): verify Ax, Bw, Bu, Cz, Dzw, Dzu, Cy
   match expected values computed directly from M₀⁻¹. Exact — discretization-independent.

2. **Loop resolution check** (`validate_lfr.py`): for sampled (x, u, Y) points, verify
   `M(Y)·v = fnet` holds to numerical precision. Confirms the algebraic loop is correctly
   resolved.

3. **Trajectory check** (`validate_lfr.py`): simulate a trajectory with the LFR block and
   compare against `scripts/gantry/gantry_lpv_torch.py` (ZOH reference, validated against
   MATLAB). Expect small RK4 vs ZOH differences — tolerance ~1e-4 at 16kHz.

---

## Jan's framework compatibility

`lfr_block.py` implements `LFRBaselineBlock(Block)` — a stateless Jan-compatible wrapper:

- **Stateless**: state `x` is managed externally by Jan's `Interconnect`, not stored in the block
- `forward(z_in: (batch,9,1) float32) → w_out: (batch,18,1) float32`
- `z_in = cat([x(6), u_stage(3)])` — Interconnect routes x and u into the block each step
- `w_out = cat([x_next(6), z_lfr(6), w_lfr(6)])` — fixed slot contract, do not change
- dtype boundary: float32→float64 at entry, float64→float32 at exit (marked in code)

**Augmentation wiring** (parallel, additive — D-003):
- `w_out[:, 6:12, :]` (`z_lfr`) → augmentation block input via `selection_matrix(arange(6,12), 18)`
- `w_out[:, 12:,  :]` (`w_lfr`) → optional second augmentation input channel
- Augmentation output → `xp` additively via `connect_signals(aug_block, 'xp')`

See `test_jan_compat.py` for the full working wiring (checks C and D).

---

## Implementation workflow

Build and validate in four phases. Each phase can be verified independently before
proceeding. Do not start a phase until the previous one passes its checks.

### Phase 1 — `physics.py`: constants and M(Y) decomposition

1. Define all scalars as `torch.float64` module-level tensors
2. Extract M₀, M₁, M₂ explicitly by reading off constant/linear/quadratic terms from M(Y):
   - M₁ is sparse: only M₁[0,1] = M₁[1,0] = -mh, rest zero
   - M₂ is sparse: only M₂[1,1] = mh, rest zero
   - Build from `torch.zeros(3,3, dtype=torch.float64)`, set individual entries — do not transcribe a full matrix
3. Build C, K, P, fs, ts
4. **Verify**: `M0 + M1*Y + M2*Y²` matches `gantry_lpv_torch.py`'s M(Y) entry-wise at several Y values,
   and `det(M(Y))` matches `det_M` from `lpv_matrices.mat` across all 50 Y values

**Pitfalls:**
- Every constant must be `torch.tensor(..., dtype=torch.float64)`. Python float literals
  in mixed expressions will silently produce float32.
- M₁ and M₂ are sparse — build with `torch.zeros` and assign entries, not by writing a full matrix

---

### Phase 2 — `lfr_matrices.py`: G matrix precomputation

1. Compute M₀⁻¹ via `torch.linalg.solve(M0, eye3)` — **never** `torch.linalg.inv`
2. Compute all products (M₀⁻¹K, M₀⁻¹C, M₀⁻¹M₁, M₀⁻¹M₂) via `torch.linalg.solve`
3. Assemble all G matrix entries and store in a frozen `GMatrix` dataclass
4. **Verify**: each entry matches the formula in the derivation by computing it independently

**Critical design decision — trainable parameters:**

If physical parameters (M₀ entries, K, C) will later become trainable `nn.Parameter`,
then M₀⁻¹ and all G entries **cannot be precomputed once at init** — they must be
recomputed every forward pass from current parameter values.

Design rule: implement `build_G_matrix(M0, M1, M2, K, C)` as a **callable function**,
not a one-time constructor. This makes it trivial to call inside `forward()` later
without changing the signature.

---

### Phase 3 — `lfr_forward.py`: CT forward pass

1. Signature: `lfr_forward(x, u, Y, M0, M1, M2, G) → (xdot, z, w, y)`
   - M0, M1, M2 are passed as explicit arguments — not captured from module state.
     This ensures gradients flow correctly and enables trainable parameters later.
   - `u` enters in stage coordinates; P transform is applied **before** calling this function
2. Build `M_Y = M0 + M1*Y + M2*Y²` at runtime — always, every call
3. Compute `fnet = -(K @ x[:3] + C @ x[3:]) + u_logical` — lower-3 component only
4. Solve `v = torch.linalg.solve(M_Y, fnet)` — shape (3,)
5. Build `v1 = Y * v`, `v2 = Y * v1`, then `z = cat([v, v1])`, `w = cat([v1, v2])`
6. Apply G: `xdot = Ax @ x + Bw @ w + Bu @ u_logical`
7. Return `(xdot, z, w, y)` where `y = Cy @ x`

**Validate at CT level before any integration:**
Compare `xdot` from `lfr_forward` against the collapsed formula
`A_c(Y)·x + B_c(Y)·u` (computable directly from M(Y)⁻¹ via `gantry_G_matrices.mat`).
They must match to machine precision at all tested (x, u, Y) points.
This confirms the forward pass is algebraically correct before introducing RK4.

**Pitfalls:**
- `Y = x[:, 2]` is shape `(batch,)`. `M_Y` is built as `(batch,3,3)` via `Y[:, None, None]`;
  `fnet` is `(batch,3)`. `torch.linalg.solve(M_Y, fnet.unsqueeze(-1)).squeeze(-1)` gives `(batch,3)`.
- All operations are batch-native — no squeeze/unsqueeze hacks in the physics internals.
- **No in-place operations**: `x[0] = ...` or `.fill_()` break autograd.
  Use `torch.cat`, `torch.stack`, always create new tensors
- Confirm autograd: set `Y.requires_grad=True`, run `lfr_forward`, call `.backward()`
  on a scalar reduction of `xdot`. `Y.grad` must be non-None.

---

### Phase 4 — `lfr_simulate.py`: RK4 integration

1. Extract `Y = x[2]` **at each RK4 sub-step** from the intermediate state —
   Y is self-scheduled and changes across sub-steps; using only x[k] would be less accurate
2. Hold `u` constant across the step (ZOH input assumption)
3. Record `z` and `w` at the **start** of each step from `x[k]`, not from sub-steps
4. For training: the entire loop must stay in-graph — never call `.detach()` on
   intermediate states during the RK4 loop

**Pitfalls:**
- A naive Python for-loop over N timesteps creates a computation graph of depth N.
  Memory grows linearly with sequence length — this will OOM for long sequences during training.
  Check how Jan's framework handles its simulation loop (likely TBPTT) before committing
  to a loop structure that conflicts with it.
- RK4 intermediate states (`x + h/2 * k1`, etc.) must remain float64 — verify no
  implicit dtype downcast during addition with `ts`

### Phase 5 — `lfr_block.py` + `test_jan_compat.py`: Jan's framework integration ✅

**Status: complete. All 11 checks pass.**

Run as: `conda run -n GraduationProject python -m lpv_lfr_baseline.tests.test_jan_compat`

#### `lfr_block.py` — stateless Block wrapper

Key design decisions implemented:
- **Stateless**: `x` is NOT stored in the block. Jan's `Interconnect` routes `x_next` back as
  the next step's `x` input — the block receives it via `z_in[:, :6, :]` each call.
- `z_in = cat([x(6), u_stage(3)])` — nz=9; `w_out = cat([x_next(6), z_lfr(6), w_lfr(6)])` — nw=18
- Y = x[:, 2] extracted inside `forward()` — never a named Interconnect signal (no algebraic loop)
- dtype boundary: `z_in.squeeze(-1).double()` at entry; `w_f64.float().unsqueeze(-1)` at exit
- Physical parameters stored as plain attributes (not `nn.Parameter`). For GPU: replace with
  `self.register_buffer()`. For trainable params: call `build_G_matrix()` inside `forward()`.

#### `test_jan_compat.py` — integration tests

**Checks 1–5** (structural compatibility):
1. `isinstance(block, Block)` and `nz==9`, `nw==18`
2. `detect_algebraic_loop` does not trigger (Y not an external signal)
3. Forward shapes: `y (batch,3)`, `xp (batch,6)` ✅
4. Single-step BPTT: `x.grad is not None` after `xp.sum().backward()` ✅
5. `z_lfr` slot has `grad_fn` after `selection_matrix @ w_out` ✅

**Checks A–F** (value correctness + augmentation readiness):

| Check | What it verifies | Bug it catches |
|-------|-----------------|----------------|
| A | `xp` values match `rk4_step` directly | Wrong slot selection in S_xp |
| B | `z_lfr`/`w_lfr` slot values match `lfr_forward` | z/w slot swap (both 6-dim) |
| C | Aug weights get gradients end-to-end | Broken z_lfr routing to aug block |
| D | Non-zero aug output changes `xp` | Silent aug→xp disconnection |
| E | 3-step BPTT: gradient reaches x0 | `.detach()` at step boundary |
| F | Dynamic G gives larger M0 grad than static G | Incomplete gradient with stale G |

---

## Interconnect and algebraic loop — structural pitfalls for Jan's integration

The G matrix has `Dzw ≠ 0`. If G and Δ(Y) were wired as separate blocks in Jan's
`SSE_Interconnect`, the graph would contain an algebraic loop and his assertion at
`interconnect.py:135` (`assert not detect_algebraic_loop(...)`) would reject it.

The loop is resolved **inside** `lfr_forward` — but Jan's Interconnect does not know
this. The danger is accidentally re-introducing the loop at the Interconnect level by:
- Wiring a separate Δ(Y) block
- Routing Y as an external signal through `connect_signals`
- Exposing z and w as separate block outputs that feed back into the same block

**Safe rule:** Y = x[2] is extracted **inside** `forward()`, never routed through
`connect_signals`. The block sees only `(x, u)` as inputs. This mirrors `blocks.py`
line 145: *"p is computed from state to avoid algebraic loops in the Interconnect graph"*.

---

## Jan's augmentation coupling — design decisions for future integration

**Stacked block output:**
Jan's `Block.forward` returns a single tensor. To expose z and w for the augmentation:
```
output = cat([x_next, z, w])   # (18,) — state + latent signals
y      = Cy @ x_next           # (3,)  — computed via output matrix block
```
`connect_signals` then routes:
- `output[:6]`   → x_next, fed back as the block's state input
- `output[6:12]` → z_baseline, available as augmentation input channel (M_ab coupling)
- `output[12:18]`→ w_baseline, available as augmentation input channel (M_ab coupling)

This stacking contract must be fixed before implementing `lfr_forward` — changing it
later requires rewiring all Interconnect connections.

**M_ba coupling (augmentation → baseline):**
For parallel augmentation (D-003), the augmentation adds to xdot additively and the
baseline does not receive augmentation signals. M_ba = 0 is the safe default.
If M_ba ≠ 0 is ever required, a new algebraic loop analysis is needed before implementing.

**Trainable physical parameters:**
If M₀, K, or C entries later become `nn.Parameter` (learning physical corrections from data):
- `build_G_matrix(M0, M1, M2, K, C)` must be called inside `forward()`, not at init
- `lfr_forward`'s signature already accepts M0/M1/M2 as arguments — no signature change needed
- All operations remain differentiable through `torch.linalg.solve`

---

## Torch pitfall summary

| Pitfall | Rule |
|---------|------|
| float32 contamination | Every constant: `torch.tensor(..., dtype=torch.float64)`. No bare Python floats in mixed expressions |
| Cached inverse invalid if params trainable | Use `torch.linalg.solve`; implement `build_G_matrix` as a callable function, not a stored object |
| In-place ops break autograd | Never use `x[i] = ...` or `.fill_()` in the forward path |
| Shape mismatch in solve | M_Y must be exactly (3,3), fnet exactly (3,) — check before calling |
| Y dimensionality | `Y = x[:, 2]` — batch-native (batch,) tensor. Inside `lfr_block`, extracted as `z_f64[:, 2]` after squeeze |
| Long-sequence memory | Do not write a naive N-step loop assuming it fits in memory during training — align loop structure with Jan's framework first |
| numpy in training path | No numpy from `lfr_forward` onward — numpy ops are not in the autograd graph |

---

## Related documents

| Document | Path | Description |
|----------|------|-------------|
| LFR derivation (LaTeX) | `LPV/LFR-derivation-supervisor.tex` | Full step-by-step CT LPV-LFR derivation; supervisor-reviewed version |
| Supervisor feedback | `LPV/Feedback Supervisor/Roland_Toth_Feedback.md` | All of Roland Tóth's annotations on the derivation PDF, structured by page and priority |
| Dimension reduction notes | `LPV/Feedback Supervisor/Roland_Toth_Reduction_Notes.md` | Roland's SVD-based method to reduce n_z from 6 to 4; includes numerical verification |
| Annotated PDF | `LPV/Feedback Supervisor/LPV_LFR_Stepwise_Derivation_Feedback.pdf` | Original annotated document |
| Reduction notes PDF | `LPV/Feedback Supervisor/LPV_LFR_Roland_notes.pdf` | Roland's handwritten notes on the SVD reduction method |
| Implementation method | `docs/lfr-baseline-implementation-method.md` | Resolve-and-retain architecture, algebraic loop analysis, and Architecture 1 vs 2 comparison |
| Dzw=0 alternative | `lpv_lfr_baseline/docs/lfr-dzw-zero-alternative.md` | Alternative LFR using ρ=[σ,Yσ,Y²σ] scheduling — achieves Dzw=0, enables proper LFR augmentation |

---

## Package structure

```
lpv_lfr_baseline/
├── core/                          # Pure LFR model — no Jan dependency
│   ├── physics.py                 # Physical constants as torch tensors: M₀, M₁, M₂, K, C, P, fs, ts
│   ├── lfr_matrices.py            # Precompute constant G matrix entries from M₀⁻¹
│   ├── lfr_forward.py             # Resolve-and-retain forward pass: steps 1–7
│   └── lfr_simulate.py            # RK4 integration loop; exposes rk4_step and simulate
│
├── blocks/                        # Jan-compatible wrappers
│   ├── lfr_block.py               # Stateless Block subclass; nz=9, nw=18; float32↔float64 boundary
│   ├── lfr_param_block.py         # Block with trainable physical parameters (nn.Parameter)
│   └── lfr_fit_system.py          # SSE_Interconnect subclass with param_loss support
│
├── svd/                           # SVD-reduced LFR (n_z: 6 → 4)
│   ├── lfr_svd_reduction.py       # G matrix reduction via SVD
│   ├── lfr_svd_forward.py         # Forward pass using reduced G
│   ├── lfr_svd_simulate.py        # RK4 with reduced G
│   └── lfr_svd_block.py           # Jan-compatible Block for reduced model
│
├── scripts/                       # Runnable entry points (not imported by core)
│   ├── train_param_recovery.py    # Step 3b: recover physical params from MATLAB data
│   ├── validate_lfr.py            # G matrix check, loop resolution, trajectory comparison
│   ├── compare_dtype.py           # float64 vs float32 accuracy diagnostic
│   └── data_utils.py              # Data loading and RMSE baseline computation
│
├── tests/                         # Integration and compatibility tests
│   ├── test_jan_compat.py         # 11 checks: structural (1–5) + value/aug readiness (A–F)
│   └── test_augmentation_compat.py # Gradient/graph tests — no Jan imports
│
└── docs/                          # Design notes and LaTeX documentation
    ├── lfr-dzw-zero-alternative.md # Alternative LFR with Dzw=0 via ρ=[σ,Yσ,Y²σ]
    └── Documentation/             # LaTeX source and compiled PDF
```

---

## Lessons from `scripts/gantry/`

The scripts `gantry_ss.py` and `gantry_lpv_torch.py` are working, validated implementations
of the same physics. They are not to be followed as a template, but the following patterns
are worth carrying over:

**Physical parameters (reuse directly)**
- All constants (masses, inertia, friction, stiffness, geometry) are already verified against
  `main.m`. Copy exact values from `gantry_lpv_torch.py` into `physics.py` — do not re-transcribe
  from main.m manually.
- Use `torch.float64` for everything. Jan's framework differentiates through the simulation;
  float32 will lose precision silently.

**M(Y) decomposition (do it explicitly here)**
- `gantry_lpv_torch.py` builds M(Y) directly as a function of Y. For the LFR derivation we need
  the explicit decomposition `M(Y) = M0 + M1·Y + M2·Y²`. Extract the constant/linear/quadratic
  terms symbolically and store M0, M1, M2 as separate tensors in `physics.py`.

**torch.linalg.solve over torch.linalg.inv**
- `gantry_lpv_torch.py` uses `torch.linalg.solve(M, X)` instead of `M.inverse() @ X`.
  This is numerically more stable and is the correct pattern to follow for computing M₀⁻¹·K,
  M₀⁻¹·C, etc. in `lfr_matrices.py`. Only compute M₀⁻¹ explicitly when it is needed as a
  block in its own right (e.g. Bu top block = 0, bottom block = M₀⁻¹).

**Tensor stacking idiom**
- `gantry_lpv_torch.py` uses `torch.stack([...])` with scalar tensors as elements.
  `torch.cat([...], dim=0/1)` with 2D tensors as elements. Both patterns appear in `lfr_matrices.py`
  — use the same convention.

**Autograd must flow through M(Y)⁻¹**
- The reason everything must be in torch (not numpy) is that during training, Y is extracted from
  the state, which has `requires_grad=True`. The solve for `v = M(Y)⁻¹·fnet` in `lfr_forward` is
  on the training path — gradients must pass through it. `torch.linalg.solve` is differentiable;
  `numpy.linalg.solve` is not.

**Gradient and shape tests (reuse pattern from gantry_lpv_torch.py)**
- `validate_lfr.py` Check 2 should include a backward pass test identical in structure to the
  gradient test at the bottom of `gantry_lpv_torch.py`: set `Y.requires_grad=True`, run
  `lfr_forward`, call `.backward()` on a scalar loss, verify `Y.grad is not None`.

**ZOH reference for trajectory comparison**
- `gantry_lpv_torch.py` + `gantry_lpv_sim_torch.py` serve as the ZOH reference for Check 3
  in `validate_lfr.py`. Do not replicate their logic — call them directly from `validate_lfr.py`.

---

## MATLAB reference files (`Matlab-output/`)

These files are used by `validate_lfr.py`. Their provenance matters for knowing
what they actually prove.

**Generated by original `kamtin-fp-model/` (immutable ground truth):**

| File | Contents | Generated by |
|------|----------|--------------|
| `gantry_G_matrices.mat` | A, B, C, D at Y=0.3 (ZOH) | `main.m` |
| `gantry_input.mat` | `u` (force inputs), `r` (reference), `t` | `main.m` |
| `gantry_q3_lsim.mat` | `q3` — lsim trajectory at fixed Y=0.3 | `main.m` |
| `gantry_q_simscape.mat` | `q` — Simscape (nonlinear) trajectory | `main.m` via Simulink |

**Generated by scripts we added (`Matlab-scripts/`):**

| File | Contents | Generated by |
|------|----------|--------------|
| `lpv_matrices.mat` | A_all, B_all over 50 Y values (−0.35 to 0.35), `det_M` | `export_lpv_matrices.m` |
| `lpv_sim_varying_y.mat` | `q1`, `q_simscape`, `u_q1`, `Y_trajectory` — Y steps 0.3→0.1 m | `export_lpv_sim.m` |

**Which file to use per validation step:**

| Step | Reference | File | Origin |
|------|-----------|------|--------|
| 1 — M(Y) decomposition | `det_M` at 50 Y values | `lpv_matrices.mat` | our script |
| 2 — Loop resolution | internal check only | — | — |
| 3 — CT vector field | A_c(Y)·x + B_c(Y)·u at Y=0.3 | `gantry_G_matrices.mat` | original |
| 4 — RK4 trajectory (fixed Y) | `q3` lsim with input `u` | `gantry_q3_lsim.mat` + `gantry_input.mat` | original |
| 4b — RK4 trajectory (varying Y) | `q1` with `Y_trajectory` | `lpv_sim_varying_y.mat` | our script |

Note: `lpv_matrices.mat` and `lpv_sim_varying_y.mat` are derived from the same physics
as `main.m` (constants copied verbatim, `getss.m` called directly) — they are not
independent from the ground truth, but they were not part of the original codebase.

### Expected error magnitudes (what the numbers mean)

| Comparison | Expected error | Why |
|------------|---------------|-----|
| **Check 4** — Python RK4 vs `q1` (lpv_sim_varying_y.mat) | ~2e-14 m | Both integrate the **same CT quasi-LPV ODE** (`gantrySystem.m` uses `Y=x(3)` self-scheduled). Error is RK4 fixed-step vs MATLAB ode45 adaptive — pure integration method mismatch, not a physics difference. This is **not machine precision** (float64 ε ≈ 2.2e-16). |
| **Check 3** — Python RK4 vs `q3` (gantry_q3_lsim.mat) | ~11 mm | Different physics: our sim self-schedules M(Y) while MATLAB lsim **freezes Y=0.3** (LTI). Error grows as Y moves away from 0.3 m. Informational only — not a correctness check. |
| Old ZOH simulator vs `q1` | ~2.7e-11 m | Same quasi-LPV ODE, but ZOH discretization error on top of the integration mismatch. |

**Key insight:** A ~2e-14 agreement on Check 4 means the physics and LPV scheduling are correct — both methods compute the same trajectory to within floating-point noise of the ODE solver. If you ever see Check 4 errors above ~1e-10, that indicates a physics bug (wrong M(Y), wrong initial condition, or wrong input coordinates).

---

## Ground truth

Physical parameters come from `kamtin-fp-model/03 Simulink gantry/main.m` (immutable).
All values in this folder must match that file exactly.
Sampling rate: `fs = 16e3`, `ts = 1/fs = 62.5e-6 s` (from `main.m` line 164).
