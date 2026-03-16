# FP Model — Augmentation Interface Reference

Describes the interface between a First Principles (FP) baseline model and the augmentation framework (`model_augmentation/fit_systems/`). Derived from `1.pdf` (EJC 2025) and the augmentation codebase. Source files are authoritative.

---

## Paper: Required Form of the Baseline (1.pdf, Section 2)

The baseline must be expressed in **discrete-time** state-space form:

```
x̃_{k+1} = f_base(θ_base, x̃_k, u_k)    state transition
ỹ_k      = h_base(θ_base, x̃_k, u_k)    output readout
```

| Symbol | Meaning | Dimension |
|--------|---------|-----------|
| x̃_k | Baseline model state | n_x̃ |
| u_k | System input | n_u |
| ỹ_k | Baseline model output | n_y |
| θ_base | Baseline parameters | — |

The baseline connects to the augmentation through a fixed **interconnection matrix S** (LFR structure). Signals flowing through S:

```
w_{1,k} = φ_base(z_{1,k})    baseline output (state + readout stacked)
w_{2,k} = φ_aug(z_{2,k})     augmentation output
```

**Well-posedness constraint**: the combined signal graph must be acyclic — no algebraic loops. Checked by `detect_algebraic_loop()` in `interconnect.py`.

**Normalization constraint**: baseline matrices must be expressed in normalized coordinates matching the training data statistics:

```
f̄_base = T_x · f_base(T_x⁻¹ x̃, T_u⁻¹ u)
h̄_base = T_y · h_base(T_x⁻¹ x̃, T_u⁻¹ u)
```

Where T_x, T_u, T_y are diagonal matrices of inverse standard deviations from training data.

---

## Code: Block Interface (`blocks.py`)

Every component in the interconnect — including the FP baseline — must be a `Block` subclass.

### Base class contract

```python
class Block(nn.Module):
    nz: int    # number of inputs  (rows of input tensor z)
    nw: int    # number of outputs (rows of output tensor w)

    def forward(self, z: Tensor) -> Tensor:
        # z: (batch, nz, 1)  →  w: (batch, nw, 1)
        ...

    def init_block(self, z: Tensor):
        # optional: called once at interconnect initialization
        ...

    def param_loss(self) -> Tensor:
        # optional: regularization loss, called by SSE_Interconnect.loss()
        # required if parameters are trainable
        ...
```

### A baseline always requires two blocks

| Block | nz | nw | Computes |
|-------|----|----|---------|
| State block | n_x̃ + n_u | n_x̃ | x̃_{k+1} = A x̃_k + B u_k |
| Output block | n_x̃ + n_u | n_y | ỹ_k = C x̃_k + D u_k |

Input tensor `z` layout for both blocks:
```
z = [ x̃_k  ]   ← first n_x̃ rows  (selected baseline states)
    [ u_k   ]   ← last  n_u  rows  (system input)
```

### Existing block classes for FP baselines

| Class | Parameters | Trainable | Regularized | Use case |
|-------|-----------|-----------|-------------|----------|
| `Linear_State_Block(A, B)` | A, B | No | No | Fixed linear baseline |
| `Linear_Output_Block(C, D)` | C, D | No | No | Fixed linear output |
| `Parameterized_Linear_State_Block(A, B, RMSE_baseline)` | A, B | Yes | Yes | Trainable linear baseline |
| `Parameterized_Linear_Output_Block(C, D, RMSE_baseline)` | C, D | Yes | Yes | Trainable linear output |
| `Parameterized_LPV_Affine_Linear_State_Block(A0, B0, A1_init, sched_state_ix, ...)` | A0, A1, B0 | Yes | Yes | LPV baseline: A(p) = A0 + p·A1, p = x[i]² |

**Regularization** in `Parameterized_*` blocks:

```
Λ = (1 / RMSE_baseline) / |θ_init|    (per-element, zero where θ_init = 0)
param_loss = MSE(Λ · θ, Λ · θ_init)
```

---

## Code: Wiring into the Interconnect (`interconnect.py`)

### Adding blocks

```python
interconnect.add_block(state_block)     # assigns block_ix automatically
interconnect.add_block(output_block)
```

### Standard wiring pattern for a linear baseline

```python
from model_augmentation.utils.utils import selection_matrix, expansion_matrix

# State block: receives selected states + u, contributes to x_{k+1}
interconnect.connect_signals("x",  state_block,  "concat",    selection_matrix(FP_state_ix, nx))
interconnect.connect_signals("u",  state_block,  "concat")
interconnect.connect_signals(state_block, "xp",  "additive",  expansion_matrix(FP_state_ix, nx))

# Output block: receives selected states + u, drives y
interconnect.connect_signals("x",  output_block, "concat",    selection_matrix(FP_state_ix, nx))
interconnect.connect_signals("u",  output_block, "concat")
interconnect.connect_signals(output_block, "y",  "additive")
```

`selection_matrix(ix, n)` — selects rows `ix` from a vector of length `n`
`expansion_matrix(ix, n)` — maps a short vector into position `ix` of a vector of length `n`

### Signal flow through the interconnect (forward pass)

```
x ──(selection_matrix)──→ state_block ──(expansion_matrix)──→ x_{k+1}
u ──────────────────────→ state_block

x ──(selection_matrix)──→ output_block ──────────────────────→ y
u ──────────────────────→ output_block
```

Augmentation blocks wire additively into the same `x_{k+1}` and `y` signals.

### Connection methods

| Method | Effect |
|--------|--------|
| `"concat"` | Concatenates signal into block input z |
| `"additive"` | Adds block output w into target signal |

### Algebraic loop check

`init_forward()` (called on first `forward()`) builds a topological ordering of block computations. Any cycle raises an error.

---

## Code: Normalization (`utils/utils.py`)

```python
A_bar, B_bar, C_bar, D_bar = normalize_linear_ss_matrices(A, B, C, D, train_data, state_ix)
```

- `state_ix`: indices of which states correspond to baseline states
- Applies T_x, T_u, T_y scaling derived from training data statistics
- Normalized matrices are passed directly to block constructors

---

## Code: Loss Function (`interconnect.py` — `SSE_Interconnect.loss()`)

```
total_loss = simulation_MSE + Σ block.param_loss()
```

`param_loss()` is called automatically for:
- `Parameterized_Linear_State_Block`
- `Parameterized_Linear_Output_Block`
- `Parameterized_LPV_Affine_Linear_State_Block`
- `Parameterized_MSD_State_Block`

Blocks of type `Linear_State_Block` / `Linear_Output_Block` contribute no regularization loss.

---

## Working Example Reference

`scripts/bouc_wen/bouc_wen_pre_encoder.py` — most complete example of FP baseline + augmentation:
- Loads A, B, C, D from `.mat` file
- Normalizes with `normalize_linear_ss_matrices()`
- Instantiates `Parameterized_Linear_State_Block` and `Parameterized_Linear_Output_Block`
- Wires both into the interconnect with selection/expansion matrices
- Adds `Static_ANN_Block` as parallel augmentation
