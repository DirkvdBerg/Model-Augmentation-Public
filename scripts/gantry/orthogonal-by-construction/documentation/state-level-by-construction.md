# Orthogonal-by-construction augmentation at the state-equation level, with additional states

Status: derivation only. No code exists for this construction and none is to be written against
this document until the two decisions in Sect. 8.2 are logged.

Source: `gyorok2026obc`, local PDF
`literature/Orthogonality/gyorok2026_orthogonal-by-construction-augmentation_IFAC-JSC_arXiv2511.01321.pdf`
(arXiv v3), read against the released code `orthogonal-IO-augm-main/orthogonalx_augm/src.py`
(classes `IO_augmentation` and `IO_orthogonal_augmentation`).

The paper's model class is discrete-time input-output, a NARX regressor on lagged measured
signals. That class, and not the choice of what to call the orthogonality condition's arguments,
is what the supervisors mean by "input output is way more complex": recasting the LPV-LFR gantry
as an input-output model is the hard path, so the condition is imposed on the right-hand side of
the state equation instead. The paper's own conclusion names exactly this as open: it proposes
extension to baseline models in state-space form and states that the absence of full-state
measurement complicates the orthogonal projection of the learning component and requires careful
investigation. The companion preprint `gyorok2026wellposed` defines the augmentation states
`x_a` and then sets `n_xa = 0` in both of its case studies. So what follows is not a
transcription of a published result; it is the extension, and every place it departs is marked.

Related records: the decision titled "The trajectory protected basis keeps parameter-variation
directions only" in `docs/decisions.md` (cited by title, since two entries currently carry the
number D-186), and `D-185` for the reference-rollout point set. The existing trajectory penalty
in `scripts/gantry/orthogonality/` is a verified method that answers a different question and is
not replaced by anything here.

---

## 1. What the paper actually requires

Four objects, in the paper's own notation.

```
baseline          y_hat_k = phi(x_k) theta_b                       (Eq. 2, linear in theta_b)
additive augm.    y_hat_k = phi(x_k) theta_b + f_ANN(x_k; theta_a) (Eq. 3)
stacked           Y_hat   = Phi theta_b + F_ANN                    (Eq. 5)
projection        theta_aux = (Phi' Phi)^-1 Phi' F_ANN             (Eq. 8)
                  F_tilde   = [I - Phi (Phi' Phi)^-1 Phi'] F_ANN   (Eq. 10)
orthogonality     Phi' F_tilde = 0                                 (Eq. 11, Lemma 3)
prediction        y_hat_k = phi(x_k) theta_b_hat
                            + f_ANN(x_k) - phi(x_k) theta_aux_hat  (Eq. 13)
```

Notice five things, because each one decides something below.

1. **The inner product is the Euclidean one on the stacked output over the estimation set.**
   `Phi' F_tilde = 0` is `n_theta` scalar conditions, each a sum over samples and output
   channels. There is no per-sample condition anywhere in the paper.
2. **`theta_aux` is a least-squares coefficient, so the subtracted correction lies in
   `range(Phi)`.** The construction removes from the learned component exactly the part that
   some parameter move could have produced instead.
3. **The projector is independent of `theta_a`.** `Phi` is built from the data before
   optimisation starts. This is the single property Theorem 18 needs, see point 5.
4. **The paper explicitly permits an auxiliary evaluation set.** Immediately after Lemma 3 it
   states that `theta_aux` may be constructed from any auxiliary evaluation of the regressor
   `phi`, either on a synthetically generated data set or on a subset of the estimation data, and
   that using the whole training set is an assumption of convenience rather than a restriction of
   the method. This licenses the freeze of Sect. 5 from inside the paper rather than as our
   invention.
5. **Theorem 18, zero covariance, is the payoff, and its proof needs only point 3.** The proof
   forms `psi_k = [phi(x_k), J_f(x_k)]` with `J_f = d f_tilde / d theta_a`, and argues that all
   changes in the projected ANN output remain orthogonal to the columns of `Phi`, hence
   `phi' J_f = 0` and `psi_k' psi_k` is block diagonal. That argument is
   `J_f = (I - P_Phi) dF_ANN/dtheta_a` together with `Phi' (I - P_Phi) = 0`, and it is valid
   precisely because `P_Phi` does not depend on `theta_a`. It does **not** require exactness
   against a regressor recomputed at the current iterate. This answers the question the handoff
   left open: a frozen projector is admissible for the block structure.

The released code confirms the reading. In `IO_orthogonal_augmentation.fit`, `Phi_train` and
`K_phi = pinv(Phi'Phi) Phi'` are formed once outside the jitted cost; inside the cost the raw
network output is fitted onto that regressor and the fit subtracted; after training the residual
`Phi' (F_ANN - Phi theta_aux)` is stored as the orthogonality check. In `predict`, the
`theta_aux` from training is reused unchanged and the correction applied to test data is
`Phi_test theta_aux`, so the projector is never recomputed. A pseudo-inverse is used throughout.

### 1.1 Lemma 3 does not need Assumption 1

Worth isolating, because our regressor is rank deficient by construction. Write `A = Phi` and
`P = A (A'A)^+ A'`. With the pseudo-inverse, `(A'A)^+ A' = A^+`, so `P = A A^+`, the orthogonal
projector onto `range(A)`. It is symmetric and satisfies `P A = A`, hence

```
A' P = (P' A)' = (P A)' = A'     =>     A' (I - P) = 0
```

exactly, at any rank. Orthogonality therefore survives rank deficiency verbatim. What does not
survive is Assumption 1 itself, which Theorem 7 and Eq. 21 need for uniqueness of `theta_b_hat`.
The separation matters: rank deficiency costs the parameter-uniqueness claim, not the
orthogonality claim.

---

## 2. Our model in these objects

Read from `scripts/gantry/gantry_dynamic/model.py` (`build_model`),
`model_augmentation/fit_systems/blocks.py` (`Gantry_State_Block`,
`Parameterized_Gantry_State_Block`, `Static_ANN_Block`, `Linear_Output_Block`) and
`scripts/gantry/gantry_dynamic/config.py`. Defaults: `nx_phys = 6`, `nx_ann = 2`, `nu = 3`,
`ny = 3`, `ann_route_ix = (0,...,7)`, 14 trainable physical scalars.

| Paper object | Ours | Established from |
|-|-|-|
| `x_k`, a measured lagged input-output vector | state `x(k) = col(x_p(k), x_a(k))`, `x_p` in R^6, `x_a` in R^2, with `u(k)` in R^3 | `PHY_IX` and `nxd = NX_PHYS + NX_ANN` in `build_model` |
| `phi(x_k) theta_b`, linear in the parameters | `f_theta(x_p, u)`, RK4 of the CT LPV-LFR ODE, `Y = x_p[2]` scheduling, `theta` the 14 log-reparameterised physical scalars entering through `M(Y)^-1` | `Gantry_State_Block.deriv`, `Parameterized_Gantry_State_Block.nonlinear_function` |
| `f_ANN(x_k; theta_a)` | `g_eta(x_p, x_a, u)`, static feed-forward, reading the **full** state including `x_a`, with `nz = nxd + nu = 11` inputs and `nw = len(route_ix) = 8` outputs | `Static_ANN_Block`, `connect_block_signals(ann_block, ["x","u"])` |
| output map | `y(k) = C x_p(k) + D u(k)`, `C` and `D` fixed, reading `x_p` only | `Linear_Output_Block(C=Cd_norm, D=Dd_np)`, wired through `selection_matrix(PHY_IX, nxd)` |
| absent in the paper | initial state `x(0) = e_rho(past y, past u)`, the SUBNET encoder | encoder section of `build_model` |

Three structural facts follow from the wiring, and each has consequences later.

- The physical block is connected to `xp` through `expansion_matrix(PHY_IX, nxd)`, so **the
  baseline writes nothing into the `x_a` rows**.
- The ANN block is connected through `expansion_matrix(route_ix, nxd)`, and the default route
  covers all eight rows, so **the learned block writes both physical and additional rows**.
- The output block reads `selection_matrix(PHY_IX, nxd)`, so **`y` does not read `x_a`**. The
  additional states reach the output only by re-entering the physical rows through the ANN one
  step later.

---

## 3. The state equation with the baseline separated

Let `R_p` in R^{6 x n_w} and `R_a` in R^{n_a x n_w} be the two row blocks of the routing
expansion, `R_p` collecting the routed physical rows and `R_a` the additional rows. With the
default route, `R_p = [I_6  0]` and `R_a = [0  I_2]`.

```
x_p(k+1) = f_theta(x_p(k), u(k))  +  R_p g_eta(x_p(k), x_a(k), u(k))
x_a(k+1) =                           R_a g_eta(x_p(k), x_a(k), u(k))
y(k)     = C x_p(k) + D u(k)
x(0)     = e_rho(y(-1), ..., u(-1), ...)
```

What to notice. The second row has **no baseline term at all**. The additional states are a
purely learned object: the baseline defines no equation for them, so in that row there is nothing
for a learned contribution to overlap with. The third row reads `x_p` only, so the entire
influence of `x_a` on anything observable is routed through `R_p g_eta` at the next step. And the
fourth row contains parameters `rho` that appear in no other row, so any condition imposed on
rows one and two is silent about the encoder.

### 3.1 The baseline regressor at the state level

Our baseline is rational in `theta` through `M(Y)^-1` and then RK4-integrated, so `f_theta` is
not linear in `theta` and no exact `phi` exists. Take the device the predecessor paper
`gyorok2025l4dc` uses for nonlinear parameter dependence, expanding at a fixed `theta_bar` with
the evaluation pair held fixed:

```
f_theta(x_p, u)  ~=  Phi_p(x_p, u) theta  +  c(x_p, u)

Phi_p(x_p, u) = d f_theta(x_p, u) / d theta  at theta_bar        (6 x 14)
c(x_p, u)     = f_theta_bar(x_p, u) - Phi_p(x_p, u) theta_bar    (6 x 1)
Phi_tilde_p   = [ Phi_p   c ]                                    (6 x 15)
```

What to notice. The appended column `c` is not cosmetic. It makes the protected set the
**affine** set the baseline can produce, its nominal response included, rather than only the
parameter-variation directions. A learned contribution that cancels the baseline's nominal
response therefore *is* charged under `Phi_tilde_p`, and is *not* charged under `Phi_p` alone.
This is a deliberate departure from the decision titled "The trajectory protected basis keeps
parameter-variation directions only": that decision omits the offset column for the trajectory
*penalty*, whose declared target is the parameter-displacement functional. Here the declared
target is non-overlap of the one-step maps, the offset column is what makes the negation channel
chargeable at all, and the paper's `theta_aux` is a least-squares fit onto whatever columns the
regressor has. Sect. 7 item 1 states the price: the protected set then depends on the state
origin, while parameter derivatives do not.

---

## 4. The orthogonality condition for this structure

### 4.1 A per-step condition is empty in one block and fatal in the other

Stack the full `n_x = 8` rows at a single step. On the `x_a` block the baseline's rows are
identically zero, so the basis has no component there and the condition constrains no part of
`R_a g_eta`. Orthogonality against a zero basis is not a weak constraint, it is no constraint. On
the physical block the opposite happens: with 6 rows and 15 regressor columns the per-step
`Phi_tilde_p` generically spans all of R^6, its orthogonal complement is the zero subspace, and a
per-step projection would force `R_p g_eta = 0` identically. This is why the paper stacks, and it
is why our condition can only be a condition over a set of evaluation points. Do not retry the
per-step form.

### 4.2 The condition

Fix an evaluation set `Z = {(x_p(i), x_a(i), u(i))}` of `M` points, deferring to Sect. 5 the
question of where those points come from. Stack the physical rows only, since those are the only
rows the baseline writes:

```
Phi   = col_i Phi_tilde_p(x_p(i), u(i))                     (6M x 15)
G_eta = col_i R_p g_eta(x_p(i), x_a(i), u(i))               (6M x 1)
K     = (Phi' Phi)^+ Phi'                                   (15 x 6M)
```

The condition is

```
eta_aux      = K G_eta                                       (15 x 1)
G_tilde      = G_eta - Phi eta_aux
Phi' G_tilde = 0
```

What to notice. The condition is on the learned **physical-row write** only, summed over the
evaluation set, and it says that no parameter move at `theta_bar` and no multiple of the nominal
transition reproduces that write. The additional-state write `R_a g_eta` does not appear. That is
not an oversight to be patched: it is the correct verdict for this structure, and Sect. 6 says
what it means.

---

## 5. The by-construction parameterisation

### 5.1 Why the projection cannot be computed along the trajectory

If `Z` is the training rollout then `x(i)` depends on `eta`, so `Phi` and `K` depend on `eta`,
and worse, `eta_aux` fitted over a window depends on the learned writes at later steps in that
window, which depend on `eta_aux`. The correction at step `k` would be a function of the future.
There is no exact fix inside one forward pass. The paper never meets this, because its model is
static.

### 5.2 The construction

Take the paper's own licence, Sect. 1 point 4, and evaluate the regressor on a fixed auxiliary
set. Let `Z_ref` be the closed-loop rollout of the baseline alone at `theta_bar`, ANN gated off,
augmented rows zero, encoder initial state, controller in the loop, and the input the physical
block actually sees. That is the point set already specified in `D-185`. Build

```
Phi_ref = col_i Phi_tilde_p(x_p_ref(i), u_ref(i))
K_ref   = (Phi_ref' Phi_ref)^+ Phi_ref'
```

both **constant** in `eta` and in `theta`. Then parameterise the augmentation as

```
eta_aux(eta) = K_ref col_i R_p g_eta(x_ref(i), u_ref(i))            once per pass

x_p(k+1) = f_theta(x_p(k), u(k))  +  R_p g_eta(x_p(k), x_a(k), u(k))
                                  -  Phi_tilde_p(x_p(k), u(k)) eta_aux(eta)
x_a(k+1) =                           R_a g_eta(x_p(k), x_a(k), u(k))
y(k)     = C x_p(k) + D u(k)
```

What to notice. `eta_aux` is a function of `eta` alone, computed from the frozen reference set
before the rollout starts, so the recursion stays causal and the whole map stays differentiable
in `eta`. The correction subtracted at time `k` is evaluated at the **current** point and scaled
by the **frozen** coefficient, which is exactly the form of the paper's Eq. 13, the form its own
`predict` uses on test data. There is no trade-off parameter anywhere: this is a
parameterisation, not a penalty.

### 5.3 What is exact, and where

On the reference set, substituting `eta_aux = K_ref G_eta_ref`:

```
Phi_ref' ( G_eta_ref - Phi_ref K_ref G_eta_ref )
    = Phi_ref' ( I - Phi_ref Phi_ref^+ ) G_eta_ref
    = 0
```

exactly, at any rank, by Sect. 1.1. This is Lemma 3 term for term. The numerical confirmation is
the check the released code performs after training, and it must be reported next to
`norm(G_eta_ref)` so a reader can see a cancellation rather than a small input.

The block structure survives on the same set. Differentiating the projected write, and using that
`K_ref` is constant,

```
d/deta [ G_eta_ref - Phi_ref eta_aux(eta) ] = ( I - Phi_ref Phi_ref^+ ) dG_eta_ref/deta
```

so `Phi_ref' J = 0` exactly, which is Theorem 18's requirement in our notation. Differentiating
through `eta_aux` is therefore not optional: it is what makes this hold. Treating `eta_aux` as a
constant during the backward pass would break the property the construction exists to provide.

Along the actual training trajectory nothing of the kind holds. The residual
`Phi_traj' (G_eta_traj - Phi_traj eta_aux)` is not zero, and its size is an empirical quantity of
the run, not a bound. The honest statement of the guarantee is therefore: **orthogonal by
construction with respect to the frozen reference set, and approximately orthogonal elsewhere by
an amount that has to be measured.**

### 5.4 What the state-level condition implies about the output

The condition of Sect. 4.2 is between the two state-update contributions, which is the level the
supervisors specified. This subsection records what it does not reach, namely the measured
output accumulated over a horizon. Output sensitivities along a rollout obey

```
S(k+1) = A(k) S(k) + [ Phi_p(k) ; 0 ]                  d y(k) / d theta = C S_p(k)
T(k+1) = A(k) T(k) + [ R_p ; R_a ] dg_tilde/deta       d y(k) / d eta   = C T_p(k)
```

with `A(k)` the state Jacobian of the closed augmented map. Step-wise orthogonality does not
transfer through this recursion, for three separate reasons. The accumulation mixes time steps
through `A(k)`, and pairwise orthogonality at equal time says nothing about the cross-time terms.
`C` keeps 3 of the 6 physical rows and discards the rest, and orthogonality in R^6 is not
preserved by a projection. And the additional-state write is unconstrained yet enters `T` through
`R_a` and returns to the physical rows through `A(k)`.

So the state-level condition is **necessary and not sufficient** for output-level non-overlap
over a horizon. What it does buy is the removal of the one-step overlap: the Example-2 flat
direction of the paper, in which a learned linear layer and the baseline parameters trade off
along `W + theta_b = theta_b*`, is eliminated at the level of the one-step augmented map. That is
a real result, and it is narrower than a horizon-level claim. Any claim wording must say
one-step non-overlap and must not say the augmentation cannot reproduce baseline dynamics.
Verified symbolically, see Sect. 10 check 4.

---

## 6. What the additional states do under this construction

Four statements, in order of how much they matter.

1. **They are the unconstrained part of the augmentation.** The projector acts as the identity on
   the `x_a` block, because the baseline's rows there are zero. No component of `R_a g_eta` is
   ever subtracted, at any evaluation point, under any freeze.
2. **That is their intended role.** The `x_a` rows are where the learned component may carry
   memory longer than one step, which is the thing the baseline provably cannot represent and
   therefore the thing that cannot be an overlap. Leaving them unconstrained is the correct
   design, not a hole in the condition.
3. **The construction nonetheless does not bound what they do to the output.** The path
   `x_a(k) -> g_eta -> R_p g_eta -> x_p(k+1) -> y` is charged only through the pointwise
   condition on the physical write at the reference points. The learned block can still produce,
   over a horizon, an output contribution lying in the baseline's reachable output set, provided
   no single-step physical write does. Any claim that this construction prevents the augmentation
   from reproducing or negating baseline dynamics over a horizon is false and must not be made.
4. **The condition is invariant under the additional-state gauge, which is a correctness check.**
   `x_a` has no physical meaning and its coordinates are arbitrary. Under `x_a -> V x_a` with `V`
   invertible, absorbed into the ANN input and output weights, `Phi_ref` is unchanged and the
   condition is unchanged. A condition that did constrain the `x_a` block would have to depend on
   a gauge choice, which is a reason to distrust it rather than a reason to want it.

---

## 7. The paper's assumptions our setting breaks

Each item ends in a verdict: "weakens to X" or "invalidates".

1. **Baseline linear in the parameters, Eq. 2.** Ours is rational in `theta` through `M(Y)^-1`
   and then RK4-integrated. *Weakens to* orthogonality against the affine tangent set at
   `theta_bar` rather than against the baseline's true reachable set. Theorem 7 does not
   transfer, its proof substituting `Phi theta_b*` exactly. The offset column `c` restores the
   nominal response to the protected set, at the price that the protected set now depends on the
   state origin while parameter derivatives do not.
2. **`x_k` is a measured lagged input-output vector.** Ours are unmeasured states supplied by an
   encoder. This is the gap the paper's own conclusion names as open. *Weakens to* a condition
   relative to a chosen reference rollout, and introduces a dependence of the regressor on
   `theta_bar` that the paper does not have.
3. **The encoder lies outside the condition.** `rho` appears in no row the condition touches.
   *Invalidates* any claim of coverage for a contribution injected purely through `x(0)`: such a
   contribution is never charged. Declared here, not fixed.
4. **The learned component is static.** Ours carries `n_a` states for which the baseline defines
   no equation. On that block the guarantee is not weakened, it is **absent**; and by Sect. 6
   item 3 this *invalidates* any horizon-level non-overlap claim.
5. **No recursion.** *Weakens* the by-construction property from "with respect to the current
   trajectory" to "with respect to the frozen reference set", Sect. 5.3. The block-Jacobian
   argument survives that weakening, by Sect. 1 point 5, which is what makes the freeze the right
   answer rather than a concession.
6. **Assumption 1, `rank(Phi) = n_theta`.** Ours fails by construction: only 10 of the 14 raw
   scalars are identifiable, with `kb1 + kb2`, `cb1 + cb2` and `Jb + Jh` appearing only as sums.
   *Weakens to* uniqueness on `range(Phi')` only. The pseudo-inverse does not repair this, it
   selects the minimum-norm coefficient; the flat directions stay held by the `param_loss`
   anchor. Orthogonality itself is unaffected, by Sect. 1.1.
7. **Condition 4, `Phi' Delta = 0` on the data.** `Delta` is the true missing dynamics and is
   unknown outside simulation, so Condition 4 cannot be verified on Telica data and Theorem 7
   cannot be invoked. *Weakens to* the error expression of Eq. 21, namely that the physical
   estimate is unique on the identifiable directions and its error is the projection of the
   missing dynamics, plus Sect. 4.2's comparative statement under Assumption 8. This is the claim
   wording the existing decision record already fixed, and this construction does not change it.
8. **Theorem 18's noise setting, `Sigma_e` diagonal.** Noiseless simulation has no `e` at all and
   the covariance result is vacuous there; the closed-loop Telica channels are not known to be
   uncorrelated. *Weakens to* block diagonality of `Psi' Psi`, which follows from orthogonality
   alone, without the covariance conclusion.
9. **Condition 11, stable predictor, and Condition 14, persistency of excitation.** Our augmented
   rollout is marginally stable on the X and Y axes, the documented `K = 0` integrator problem.
   *Invalidates* the consistency results, Theorems 12 and 16. The paper's Remark 13 points at
   stable-by-design parameterisations as the route, which is a separate project decision.
10. **A basis frozen on one rollout is valid elsewhere.** Our `Phi_tilde_p` depends on the
    scheduling variable `Y = x_p[2]`, and held-out `Y` positions are a validation axis of the
    project. *Weakens to* exactness only over the `Y` range the reference rollout visits.
    Mitigation is to build `Z_ref` as a union of rollouts spanning the operating range, which is
    a design choice and not a theorem.
11. **One-step prediction cost.** The paper's `V` is a one-step error on stacked data, ours is
    free-run with multiple shooting. *Weakens* nothing in the condition itself, which never
    referred to the cost, but it removes direct comparability with every numerical claim in the
    paper's Sect. 5.

---

## 8. Reduction to the published result, and the two decisions

### 8.1 Reduction

Three reductions are needed, not two. Set `n_a = 0`; route to all physical rows so `R_p = I_6`;
make the baseline linear in `theta`, so `c = 0` and `Phi_tilde_p = Phi_p` exactly; take `C = I`
and `D = 0` with a one-step predictor and measured states in place of the encoder; and take
`Z_ref` to be the estimation set. Then

```
x(k+1) = Phi_p(x(k), u(k)) theta + g_eta(x(k), u(k))   <->  Eq. 5,  Y_hat = Phi theta_b + F_ANN
eta_aux = (Phi'Phi)^+ Phi' G_eta                        <->  Eq. 8
G_tilde = [I - Phi (Phi'Phi)^+ Phi'] G_eta              <->  Eq. 10
Phi' G_tilde = 0                                        <->  Eq. 11, Lemma 3
write with frozen eta_aux at the current point          <->  Eq. 13
```

term for term, under the correspondence `phi -> Phi_p`, `F_ANN -> G_eta`,
`theta_aux -> eta_aux`, with the inverse replaced by the pseudo-inverse as the released code
already does. The third reduction, one-step prediction with `C = I`, is the level gap of
Sect. 5.4 and is named rather than hidden: without it the paper's stacked object is the output
and ours is the state update, and those are not the same object.

### 8.2 The two decisions to log before any code

**The level.** The condition is imposed on the state equation, per the supervisors' direction.
Its implication for the accumulated output is the necessary-not-sufficient statement of
Sect. 5.4, now verified. Log the choice with that cost attached, so no later claim silently
upgrades it.

**The freeze.** Basis and coefficients frozen on the reference rollout, refreshed between epochs
at most. It is admissible for the block structure by Sect. 1 point 5, it is licensed by the
paper's own auxiliary-evaluation sentence, it is causal, and it keeps Lemma 3 exact on the set it
is defined on. Differentiating through a projection recomputed every pass is non-causal inside
the recursion and buys nothing the block structure needs. A penalty at the state level is already
implemented as a different method and is not needed here.

---

## 9. Acceptance, restated as checks

1. Reduction: Sect. 8.1, algebraic, done.
2. Exactness on its own terms: Sect. 5.3, algebraic and valid at any rank. The numerical
   confirmation is `norm(Phi_ref' G_tilde_ref)` reported next to `norm(G_eta_ref)`, which is the
   released code's own post-training check. Not yet run, since no code exists.
3. Broken assumptions: Sect. 7, eleven items, each with a verdict.

The one thing this derivation cannot settle is the size of the trajectory residual of Sect. 5.3,
which decides whether the frozen construction behaves like a construction or like a weak penalty
in practice. That is a measurement, and it is the first thing any implementation should report.
