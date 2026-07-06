# Outline: system / baseline / augmentation write-up for Jan

Deliverable: a short (about 2 page) note, concept + equations, that states the
data-generating system, the baseline we augment, and the augmentation, so Jan can
check the setup. Clean-sheet, no code listings, no em-dashes.

---

## 0. What to focus on (read this first)

The whole point is that Jan can verify the setup at a glance. That only works if we
speak his language. Four rules, in priority order:

1. **Use Jan's paper notation exactly** (Hoekstra et al. 2025). Every symbol and every
   equation number should match the paper. See the notation block below. This is the
   most important rule; everything else is secondary to it.
2. **System and baseline at the same explicit level.** Both are given as the actual
   `M`, `C`, `K` matrices taken directly from `Matlab-scripts/Augmentation`. No stubs,
   no Lagrangian energy sketch, no "paste from MATLAB later." The matrices exist; use them.
3. **Show the augmentation as components + interconnection, not an abstract Table 1
   echo.** Jan's framework is blocks wired by an interconnection, so the section must
   show (a) OUR blocks (each = one of his roles + the function it computes), (b) the
   routing between them (a small block scheme in his Fig. 1 style, plus words), and
   (c) the dynamic-parallel model that results. Restating Table 1 in adjectives teaches
   Jan nothing; showing our actual `f_base` (the gantry block), our actual network
   `N_θ`, and how the interconnect splits `N_θ` into his `f_aug`/`g_aug` is the point.
   Still no expansion-matrix code, no `STIFF_IX`, no RK4 substeps in the main text.
4. **Block scheme rules.** One minimal TikZ diagram (NOT matplotlib), Fig. 1 style,
   labels only, no math inside the boxes. It carries the topology/routing; the formulas
   carry the content. If it starts encoding dimensions or equations, it has failed.
5. **Surface the problem.** One boxed "open questions" block so Jan sees exactly what to
   look at (normalization source, informativeness of the added states, the `f_aug`
   routing restriction).

What went wrong before, so we do not repeat it: the previous draft transcribed the
interconnect code (expansion matrices, row indices) instead of the paper's model, and
stubbed the truth mass matrix instead of reading `gantrySystemExtended.m`. Start from
the paper's abstraction, fill in the gantry specifics from the MATLAB matrices.

---

## Notation (Jan's paper, Hoekstra et al. 2025)

| Symbol | Meaning | Paper |
|--------|---------|-------|
| `x̃ ∈ R^6` | baseline (physical) state `[q1,q2,q3,q̇1,q̇2,q̇3]`, logical coords | Eq. 2 |
| `x̄ ∈ R^2` | added states of the learning component (target `[δ_a, δ̇_a]`) | Sec. 2 |
| `x̂ = [x̃; x̄] ∈ R^8` | full model state | Eq. 3 |
| `u ∈ R^3`, `y ∈ R^3` | input (stage forces), output (stage positions) | Eq. 1 |
| `Y = q3` | scheduling variable | (gantry) |
| `f_base`, `h_base` | baseline state-transition and output functions | Eq. 2 |
| `f_aug` | learned correction added to the baseline states `x̃` | Table 1 |
| `g_aug` | learned dynamics of the added states `x̄` | Table 1 |
| `h_aug` | learned output correction (we set this to 0) | Eq. 3 |
| `ψ` | encoder, sets the initial state from past I/O | Eq. 5c |
| `V_trunc` | truncated multi-step simulation loss | Eq. 5a |
| `S` | LFR interconnection matrix (general form) | Eq. 4 |

---

## Section 1: System (data-generating truth), paper Eq. 1

Source: `Matlab-scripts/Augmentation/gantrySystemExtended.m` (explicit matrices).

8-state truth `x = [X, Θ, Y, δ_a, Ẋ, Θ̇, Ẏ, δ̇_a]`, EOM `M(Y,δ_a) q̈ = f - C₄ q̇ - K₄ q`:

- `M(Y,δ_a)` = the exact 4×4 from `gantrySystemExtended.m` (absorber sits in the
  off-diagonal `Y`, `δ_a` terms; here `mh = mh_rigid`, so `mh+ma = 10.1`).
- `C₄ = blkdiag(C_baseline, c_a)`, `K₄ = blkdiag(K_baseline, k_a)`.
- Written as `x_{k+1} = f(x_k, u_k)`, `y_k = h(x_k)` (paper Eq. 1), RK4-discretized.
- Absorber params (from the generators): `m_a = 0.10·mh = 1.01`, `f_a = 150 Hz`,
  `ζ_a = 0.05`, `L0 = 0.10`; `k_a = m_a(2π f_a)²`, `c_a = 2ζ_a√(k_a m_a)`.
- Key point: the absorber couples through the mass matrix to `X, Θ, Y` (via the
  `Y`, `δ_a` entries), not a single-axis spring.

## Section 2: Baseline (what we augment), paper Eq. 2

Source: `Matlab-scripts/Augmentation/main_augmentation.m` (explicit matrices).

Same system with `m_a → 0`, which recovers the 3-DOF gantry exactly (verified in the
MATLAB). Give the exact `M(Y)`, `C`, `K` (3×3). Then:

- `x̃_{k+1} = f_base(x̃_k, u_k)` (Eq. 2a): the LPV gantry, `M(Y)^{-1} = N(Y)/d(Y)`,
  RK4. (Name the LFR/RK4; do not expand it.)
- `ỹ_k = h_base(x̃_k, u_k) = C̄_d x̃_k + D_d u_k` (Eq. 2b).
- Mass conserved: baseline uses full `mh = 10.1`; truth splits `mh = mh_rigid + m_a`.
  So truth − baseline is only the absorber.

## Section 3: Augmentation, how we use the framework (paper Eq. 3-4, Table 1)

Jan's framework = blocks + interconnection. So the section has three parts:
components, interconnection, resulting model. This replaces the abstract Table 1 echo.

### 3a. Components (our blocks)

A table: our block, the paper role it plays, the function it computes.

| Our block | Role (paper) | What it computes |
|-----------|--------------|------------------|
| Gantry state block (`Gantry_State_Block`) | `f_base` | `x̃_{k+1} = RK4[ M(Y)^{-1}(P^⊤u - Cq̇ - Kq) ]` (matrices from Sec. 2) |
| ANN block (`Static_ANN_Block`) | learning component | `w = N_θ(x̂,u) = W₂ tanh(W₁[x̂;u] + b₁) + b₂ ∈ R⁴` |
| Output block (`Linear_Output_Block`) | `h_base` | `ŷ = C̄_d x̃ + D_d u` |

This is where we show we IMPLEMENTED the gantry: `f_base` is our own gantry block, shown
as its function, not a generic `f_base`.

### 3b. Interconnection (the routing = how we use his framework)

A minimal block scheme (TikZ, Fig. 1 style, labels only) plus a few words:
- encoder `ψ` sets the initial state `x̂_{k|k}` (Eq. 5c);
- the ANN reads the full state and input, `z = [x̂; u]`;
- its 4 outputs are added into the state: `w₁,w₂` onto the `Θ` baseline rows `(q₂, q̇₂)`,
  `w₃,w₄` onto the two added states `x̄`;
- the gantry block updates the 6 baseline states from `[x̃; u]`;
- the output block reads `ŷ` from `[x̃; u]`.

This is a fixed-interconnection special case of the general LFR form (Eq. 4); one line
saying so, do not expand `S`.

### 3c. Resulting model (dynamic parallel, Table 1)

Derived from 3a + 3b, so the equations now explain rather than quote:

```
x̃_{k+1} = f_base(x̃,u) + f_aug ,   f_aug = [w₁,w₂] on the Θ rows, 0 elsewhere
x̄_{k+1} = g_aug = [w₃,w₄]
ŷ       = C̄_d x̃ + D_d u           (h_aug = 0)
```
So `f_aug` and `g_aug` are the two slices of our single network `N_θ`. The model is
simulated: `ψ` sets `x̂_{k|k}`, the recursion is rolled out, the loss compares `ŷ` to
`y` (Eq. 5a).

Design choices (state as properties, each one line):
- `Θ`-only routing of `f_aug`. Reason: `X`, `Y` are pure integrators (`K = 0`), additive
  correction there diverges. Flag honestly: true coupling also reaches `X`, `Y` (open Q).
- 2 added states = the absorber DOF `[δ_a, δ̇_a]`.
- `tanh`: the `Y`-scheduled coupling needs bilinear `Y·δ_a` terms a linear map cannot form.
- `h_aug = 0`: output is 100% baseline; all learning is in the state.

## Section 4: Normalization, paper Eq. 9

I/O normalized to zero mean, unit variance (`T_u`, `T_y`). For the state scaling `T_x`,
Eq. 9 prescribes `σ_x` from the **baseline model's** simulated states. We instead take
`σ_x` from the **data** (finite-diff logical states of the measured truth output), and
the encoder uses baseline-simulated states, so two sources coexist. Flag this as a
deviation from Eq. 9 and a candidate baseline-vs-data mismatch (likely on velocity scale).

## Section 5: Training objective + acceptance target, paper Eq. 5

`V_trunc` truncated multi-step simulation loss (Eq. 5a), encoder-initialized (Eq. 5c).
Acceptance: reach the output-noise floor `σ_n = rms(y)·10^(-SNR/20)` on validation sim-RMS.

## Open questions for Jan (boxed)

1. Normalization: `σ_x` from data vs. baseline states (Eq. 9). Problem? Share one source?
2. Are the added states `x̄` informative (do they track `δ_a`, i.e. augment not replace)?
3. Is `f_aug` restricted to `Θ` adequate, given the true coupling also hits `X`, `Y`?

## Appendix: parameter values

The 15 gantry parameters + absorber `(m_a, f_a, ζ_a, L0)` with units. Reference Garcia
for the gantry parameters; do not re-derive.

---

## Immediate next actions

1. Build the block scheme first, as its own standalone TikZ file, and iterate on the
   figure alone until it is clean (Fig. 1 style, labels only). Do not touch the document
   until the figure is right.
2. Rewrite Section 3 of the `.tex` to the 3a/3b/3c structure (components table,
   interconnection + the figure, resulting model), replacing the current abstract
   Table 1 echo. System/baseline/normalization/training sections stay as they are.
3. Status of decisions: routing presented as `Θ`-only with the honest open-question flag
   (confirmed). `tanh` nonlinear (confirmed). `h_aug = 0` (confirmed).
