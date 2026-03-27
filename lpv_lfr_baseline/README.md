# LPV-LFR Baseline — Standalone Implementation

## Purpose

This folder contains a self-contained Python implementation of the dual-gantry
LPV-LFR baseline derived in `LPV/LFR-derivation-supervisor.tex`.

It is intentionally **independent of Jan's `model_augmentation/` framework** — no imports
from `model_augmentation/fit_systems/` or `model_augmentation/utils/`. The goal is to
implement and verify the derivation cleanly before integrating it into the augmentation
framework as a future step.

---

## Derivation reference

The LPV-LFR realization is derived in `LPV/LFR-derivation-supervisor.tex`. Key results:

**Scheduling variable**: `Y` — payload Y-position. Quasi-LPV: Y = x[2] is a system state.

**Scheduling block**: `Δ(Y) = Y·I₆` — one scalar Y repeated across 6 latent channels.

**Latent variables**:
```
z = [v;  v₁]    v  = q̈ = M(Y)⁻¹·fnet
w = [v₁; v₂]    v₁ = Y·v
                 v₂ = Y·v₁ = Y²·v
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
Step 1:  fnet = [-K, -C]·x + u
Step 2:  v    = M(Y)⁻¹ · fnet              ← loop resolved analytically
Step 3:  v₁   = Y · v
         v₂   = Y · v₁
Step 4:  z    = [v;  v₁]
         w    = [v₁; v₂]
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

This folder does **not** import from `model_augmentation/`. However, it is designed so
the baseline block can be wrapped as a `Block` subclass with minimal changes:

- `LFRBaselineBlock.forward(x, u)` returns a stacked tensor `cat([xp, z, w]) ∈ R¹⁸`
- Physical parameters stored as instance attributes (torch buffers, not global state)
- Stacked output means `connect_signals` + selection matrices route slices to `xp`,
  augmentation z-input, and augmentation w-input respectively

**Open questions before Jan's integration** (see `docs/lfr-baseline-implementation-method.md`):
- Whether M_ba coupling (augmentation signals into baseline) is needed for parallel augmentation
- Coordinate system of z/w (logical vs stage — see D-006)
- Exact connection matrix wiring for the stacked output

---

## File map

| File | Contents |
|------|----------|
| `physics.py` | Physical constants as torch tensors: M₀, M₁, M₂, K, C, P, fs, ts |
| `lfr_matrices.py` | Precompute constant G matrix entries from M₀⁻¹ |
| `lfr_forward.py` | Resolve-and-retain forward pass: steps 1–7 above |
| `lfr_simulate.py` | Standalone RK4 simulation loop using `lfr_forward` |
| `validate_lfr.py` | Validation script: G matrix check, loop resolution, trajectory comparison |

---

## Ground truth

Physical parameters come from `kamtin-fp-model/03 Simulink gantry/main.m` (immutable).
All values in this folder must match that file exactly.
Sampling rate: `fs = 16e3`, `ts = 1/fs = 62.5e-6 s` (from `main.m` line 164).
