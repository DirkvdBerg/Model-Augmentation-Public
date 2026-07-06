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
3. **Augmentation in the paper's model form, not the code's plumbing.** State it as the
   Table 1 *dynamic parallel* structure: `f_base`, `f_aug`, `g_aug`, `h_base`. Then say
   in words what each is for the gantry. Do NOT write expansion/selection matrices,
   `STIFF_IX` index sets, RK4 substeps, or the encoder derivation. Those are the
   implementation, not the model, and they are what made the last draft unreadable.
4. **Surface the problem.** One boxed "open questions" block so Jan sees exactly what to
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

## Section 3: Augmentation, paper Eq. 3 + Table 1 (dynamic parallel)

State it as the paper's *dynamic parallel* row of Table 1, verbatim structure:

```
x̃_{k+1} = f_base(x̃_k, u_k) + f_aug(x̃_k, x̄_k, u_k)     (correct the baseline states)
x̄_{k+1} = g_aug(x̃_k, x̄_k, u_k)                          (drive the added states)
ŷ_k     = h_base(x̃_k, u_k) = C̄_d x̃_k + D_d u_k          (h_aug = 0)
```

The model is simulated: `ψ` sets `x̂_{k|k}` from past I/O (Eq. 5c), then the recursion
runs for the horizon and the loss compares `ŷ` to `y` (Eq. 5a). Show that loop once.

What each function is for the gantry (in words, plus the design choice + reason):

- `f_base` = the baseline gantry map (Section 2).
- `f_aug` = correction to the baseline states, **nonzero only on the `Θ` components**
  (rotational position/velocity). Reason: `X`, `Y` are pure integrators (`K = 0`); an
  additive correction there diverges. (Flag honestly: the true coupling also reaches
  `X`, `Y`, so whether this restriction is adequate is an open question.)
- `g_aug` = the dynamics of the two added states `x̄` (targeting `δ_a, δ̇_a`).
- `h_aug = 0`: no learned output term; the output is 100% baseline.
- `f_aug`, `g_aug` are realized by one `tanh` network (implementation note, one line).
  `tanh` needed: the `Y`-scheduled coupling requires bilinear `Y·δ_a` terms a linear
  map cannot form.

Optional: one line noting this is a special case of the general LFR form (Eq. 4) with a
fixed interconnection `S`. Do not expand `S`; the three lines above are clearer.

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

1. Rewrite the `.tex` (`docs/jan-augmentation-writeup.tex`) to this structure: paste the
   exact `M(Y,δ_a)`, `C₄`, `K₄` (truth) and `M(Y)`, `C`, `K` (baseline) from the MATLAB;
   augmentation as the Table 1 three lines; drop all expansion matrices / index sets.
2. Confirm the `f_aug` = "Θ only" description is how we want to present the routing.
