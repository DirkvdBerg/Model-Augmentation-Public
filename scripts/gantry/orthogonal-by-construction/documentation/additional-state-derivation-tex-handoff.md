# Audited handoff: Györök IO model, delay-state realization, and additional learned states

Prepared 2026-09-15. This file is a writing specification for a new session, not the final thesis derivation. It supersedes inconsistent explanations in the preceding conversation within the scope stated below. Its algebra is explained explicitly so the next session can check it independently.

## 1. Assignment and boundaries

Write a clean, self-contained, stepwise LaTeX derivation with citations and short explanatory proofs. Deliver a standalone `.tex` source and a compiled PDF if the local TeX toolchain is available. Use automatic equation numbering and `\label`/`\eqref`, not unexplained tags such as `2-SS-output`.

The subject is **Györök's discrete-time IO predictor and its artificial delay-state realization, extended with additional learned states**. It is not yet a derivation for the gantry's physical coordinates, continuous-time model, or nonlinear physical-parameter basis.

The user and supervisors want orthogonality at the state-equation level, using the baseline model-function contribution and the learned correction. They do not require a new two-rollout comparison, trajectory regularizer, cross-Fisher penalty, or observability-weighted projector. Do not introduce those as necessary conditions. Explain the scope of the property without changing its definition.

The current session prepares this handoff only. The next session writes the final TeX. Do not overwrite existing documents or modify production code merely to prepare the manuscript.

### Documents to inspect, critically

- `scripts/gantry/orthogonal-by-construction/documentation/io-additional-state-realisation.md`: existing derivation, useful but not an unquestionable source.
- `scripts/gantry/orthogonal-by-construction/code/io_additional_state_equivalence.m`: exact symbolic surrogate checks.
- `scripts/gantry/orthogonal-by-construction/code/logs/io_additional_state_equivalence.log`: existing successful run, not new verification of the physical gantry.
- `scripts/gantry/orthogonal-by-construction/documentation/external-briefing.md`: physical-gantry context; keep its separate physical-coordinate questions outside the main proof here.
- Read project instructions before writing. Older project decisions may contain superseded assertions about recurrence, covariance, reference sets, or causality; inspect them, but do not use them as mathematical authority.

## 2. Sources, equation anchors, and attribution

Fix the main paper version for reproducible equation numbering:

**G26:** B. M. Györök, M. Schoukens, T. Péni, R. Tóth, *Orthogonal-by-construction augmentation of physics-based input-output models*, arXiv:2511.01321v2, 6 January 2026. The journal version is IFAC Journal of Systems and Control 35, 100376 (2026). Verify journal/version metadata before preparing the bibliography.

- HTML: https://arxiv.org/html/2511.01321v2
- PDF: https://arxiv.org/pdf/2511.01321v2
- DOI: https://doi.org/10.1016/j.ifacsc.2026.100376

Source anchors: Eq. (1), IO history; Eq. (2), baseline; Eq. (3), additive correction; Eqs. (4)–(5), loss and stacking; Eqs. (6)–(7), informativity/rank; Eqs. (8)–(11), auxiliary coefficient/projection/orthogonality; Eq. (13), post-training prediction. The feedforward architecture is specified immediately after Eq. (3). The conclusion leaves state-space extension open.

**B22:** F. Bonassi, J. Xie, M. Farina, R. Scattolini, *An Offset-Free Nonlinear MPC scheme for systems learned by Neural NARX models*.

- Local PDF: `literature/Orthogonality/IO-StateSpace/An Offset-Free Nonlinear MPC scheme for systems learned by Neural NARX models.pdf`
- Open record: https://arxiv.org/abs/2203.16290
- Section II, Eqs. (1)–(3), PDF page 2: NARX regressor, history state, and `A, B_u, B_x` shift/insertion structure.
- Local PDF is an extended manuscript, arXiv v4 (13 January 2023). Its first-page footer gives the 2022 CDC article as pp. **2123–2128**, DOI https://doi.org/10.1109/CDC51059.2022.9992362 . Do not repeat the different page range found in an earlier project research report.

**S21:** K. Seel, E. I. Grøtli, S. Moe, J. T. Gravdahl, K. Y. Pettersen, *Neural Network-Based Model Predictive Control with Input-to-State Stability*.

- Local PDF: `literature/Orthogonality/IO-StateSpace/Neural Network-Based Model Predictive Control with Input-to-State Stability.pdf`
- DOI: https://doi.org/10.23919/ACC50511.2021.9483190
- Eq. (2), PDF page 2: history state.
- Eqs. (6)–(7): explicit noisy and nominal history updates.
- Eqs. (8)–(9): measured-history/series-parallel versus predicted-history/parallel NN operation.

Use B22 and S21 for realization and feedback conventions, not their MPC stability theorems. Verify any quoted formula against the actual source. Source equation numbers and the manuscript's equation numbers are different: write “corresponding to G26, Eq. (3)” alongside a locally numbered adapted equation.

**Attribution rule:** the additional-state extension and the propositions below are our derivation. Do not present them as Györök's equations or theorems. Distinguish source statements, exact rewriting, and new model-class choices.

## 3. Essential corrections to earlier conversation and documents

1. **Recurrence was already present.** Recursive IO simulation feeds predicted outputs into subsequent histories. Adding learned states supplies additional memory; it does not introduce recurrence for the first time. A feedforward network can be embedded in a recurrent dynamical model.
2. **“Static ANN” needs qualification.** G26 specifies a feedforward function of an IO regressor. Say “no separate learned-state recursion,” rather than implying no dynamic behavior of the whole model. Projection algebra itself does not require a feedforward architecture.
3. **Keep G26's output index.** The predictor is `hat y_k`, using outputs through `k-1` and current input `u_k`. Do not silently switch to `hat y_(k+1)`.
4. **An output index shift alone does not resolve input timing.** B22/S21 predict `y_(k+1)` using `u_k`, whereas G26 predicts `y_k` using `u_k`. Relabeling the former gives a last input `u_(k-1)`, not automatically `u_k`. Cite structural precedent; construct our matrices independently for G26's chosen timing. Do not claim exact model equivalence between the cited papers by only shifting the output index.
5. **The output map is not automatically `C s_k`.** Our state stores past outputs. Current `hat y_k` is the predictor function; the newest output slot of `s_(k+1)` stores it after updating.
6. **Orthogonality is stacked, not per sample.** Individual products may be nonzero. Never replace a summed condition with one equation at every time point.
7. **Different notions of basis.** Protecting the entire baseline parameter span is stronger than being orthogonal to one baseline vector at one parameter value. G26 uses the span of `Phi`.
8. **The zero latent block is a defined baseline extension.** It means the baseline supplies no next-state contribution for learned coordinates. It does not say learned coordinates are always zero, unobservable, or irrelevant to future outputs.
9. **Do not transfer a covariance theorem from a projected derivative identity.** Fixed-reference write Jacobians can be orthogonal while total rollout/output Jacobians are not.
10. **Rank-deficient extrapolation is not necessarily unique.** Different least-squares coefficients subtract the same vector on the reference set, but can subtract different functions at new points. The Moore–Penrose convention specifies one coefficient unambiguously.
11. **Fixed latent reference values are a sufficient choice, not a universal necessity.** If measured IO histories keep `Phi` fixed, latent trajectories generated independently of the output subtraction can depend on learned parameters without spoiling the projector identity. Do not claim every recurrent augmentation requires frozen latent tuples.
12. **No unconditional no-hypotheses claim.** Exact realization requires the specified finite-history map, matching initialization, feedback and timing, and well-defined evaluations. It needs no extra stability, observability, or rank assumption for that identity.
13. **No simple IO-to-physical-state identification.** Delay coordinates are not gantry positions/velocities. No equivalence to the physical gantry is established by the delay realization or surrogate MATLAB checks.

## 4. Notation and dimensions

Use one notation consistently throughout the TeX.

| Symbol | Dimension | Meaning |
|---|---|---|
| `p` | scalar | output dimension |
| `m` | scalar | input dimension |
| `L` | scalar | number of past output samples |
| `J` | scalar | number of past input samples, excluding current input |
| `r` | scalar | number of additional learned-state coordinates |
| `q` | scalar | number of baseline parameters |
| `d_s=Lp+Jm` | scalar | delay-state dimension |
| `s_k` | `d_s` | stored IO history |
| `u_k` | `m` | current external input |
| `x_a,k` | `r` | additional learned state |
| `xi_k=col(s_k,x_a,k)` | `d_s+r` | complete artificial state |
| `theta_b` | `q` | baseline parameters |
| `eta` | arbitrary finite dimension | learned parameters; do not also use it for a predictor function |
| `phi(s,u)` | `p × q` | source basis after explicitly defined argument permutation |
| `g_y,eta` | `p` | learned output correction |
| `g_a,eta` | `r` | next additional-state value |
| `A` | `d_s × d_s` | fixed history shift |
| `B_x` | `d_s × p` | insert newly calculated output |
| `B_u` | `d_s × m` | insert current input |
| `Phi` | `Np × q` | stacked baseline basis on reference points |
| `G_y,eta` | `Np` | stacked output corrections |
| `a_eta` | `q` | auxiliary coefficient; not a separately optimized free parameter |
| `E=col(B_x,0_(r×p))` | `(d_s+r) × p` | output insertion into complete state |
| `M_N=I_N ⊗ E` | `N(d_s+r) × Np` | repeated insertion per reference point |
| `Psi=M_N Phi` | `N(d_s+r) × q` | augmented stacked baseline parameter basis |
| `calG_eta` | `N(d_s+r)` | stacked full learned transition writes |

For the main matrix derivation choose `L≥1, J≥1`. Relate `L,J` to G26's output/input lag orders, and do not use `n_a` for both a lag count and a latent-state dimension. A zero past-input count can be handled by empty input-history blocks (`B_u=0`). If there is no output history, either discuss it separately or add a redundant newest-output storage slot; the unmodified `B_x^T B_x=I_p` proof presumes an output slot exists.

## 5. Stepwise manuscript outline with derivation

### Step 1 — Begin from G26's finite-memory IO model

Display the source process and explain measured versus predicted quantities:

\[
y_k=f(x_k^{\mathrm{src}})+e_k.
\]

Define

\[
s_k=\operatorname{col}(y_{k-1},\ldots,y_{k-L},
u_{k-1},\ldots,u_{k-J}).
\]

There is a fixed permutation `Pi` with

\[
x_k^{\mathrm{src}}=\Pi\operatorname{col}(s_k,u_k).
\]

If the source's order already matches, the appropriate permutation may be trivial. Define the adapted function explicitly:

\[
\phi(s,u):=\phi_{\mathrm{src}}
\bigl(\Pi\operatorname{col}(s,u)\bigr).
\]

This separates state from external input without discarding or adding prediction information. Do not retain an unexplained second regressor `z_k` alongside `s_k` unless needed. State that `u_k` remains included in the predictor.

### Step 2 — Rewrite the baseline, G26 Eq. (2)

\[
\hat y_k=\phi(s_k,u_k)\theta_b.
\]

Explain linearity in `theta_b`, allowing nonlinear dependence of `phi` on signals. This is not state/input linearization and is not yet the gantry's nonlinear parameter model.

### Step 3 — Define the shift and insertion matrices

Let `S_L` be the `L×L` shift with ones at `(i+1,i)` and zeros elsewhere; similarly define `S_J`. With `e_1` the first coordinate vector of the relevant dimension,

\[
A=\operatorname{diag}(S_L\otimes I_p,S_J\otimes I_m),
\quad
B_x=\begin{bmatrix}e_1^{(L)}\otimes I_p\\0_{Jm\times p}\end{bmatrix},
\quad
B_u=\begin{bmatrix}0_{Lp\times m}\\e_1^{(J)}\otimes I_m\end{bmatrix}.
\]

Then

\[
s_{k+1}=As_k+B_u u_k+B_x v_k.
\]

Explain each term in words. Verify

\[
B_x^TB_x=I_p,\qquad B_x^TA=0,\qquad B_x^TB_u=0.
\]

Include the scalar example `L=2,J=1`:

\[
s_k=\begin{bmatrix}y_{k-1}\\y_{k-2}\\u_{k-1}\end{bmatrix},\quad
A=\begin{bmatrix}0&0&0\\1&0&0\\0&0&0\end{bmatrix},\quad
B_x=\begin{bmatrix}1\\0\\0\end{bmatrix},\quad
B_u=\begin{bmatrix}0\\0\\1\end{bmatrix}.
\]

`B_x` inserts an output only; `B_u` inserts the input independently. Neither converts an output into both output and input. Cite B22 and S21 as precedent with the timing qualifications in Section 3 above.

### Step 4 — Choose feedback and output timing explicitly

For recursive deterministic simulation, use `v_k=hat y_k` and store predicted output histories. Initial histories can be supplied from measurements; subsequent entries are simulated. For measured-history prediction use `v_k=y_k` and store actual observations. If helpful, put hats on simulated histories and then announce a lighter convention for the main deterministic formulas.

The baseline recursive realization is

\[
s_{k+1}=As_k+B_u u_k+B_x\phi(s_k,u_k)\theta_b,
\qquad
\hat y_k=\phi(s_k,u_k)\theta_b.
\]

In this convention,

\[
B_x^T s_k=v_{k-1},\qquad B_x^Ts_{k+1}=v_k.
\]

Do not use `hat y_k=C s_k` to select the current prediction. That would select a past output. A different state/timing convention is possible but would require a separate alignment of current-input dependence.

“Recursive simulation” does not mean “without a controller.” The input sequence may be prescribed or generated causally by a controller. Closing the loop does not alter this bookkeeping identity. It also does not establish closed-loop stability.

### Step 5 — Add the existing output correction, G26 Eq. (3)

\[
\hat y_k=\phi(s_k,u_k)\theta_b+g_{y,\eta}^{(0)}(s_k,u_k),
\]

\[
s_{k+1}=As_k+B_u u_k
+B_x\phi(s_k,u_k)\theta_b
+B_xg_{y,\eta}^{(0)}(s_k,u_k).
\]

The superscript `(0)` can indicate “without learned additional states” and may be omitted after explaining the stage. The predicted output already feeds the next history: recurrence is present. Distinguish this from G26's measured-regressor training evaluations.

### Step 6 — Stack only the quantities needed for projection, G26 Eq. (5)

At reference pairs `(s_i^ref,u_i^ref)`, define

\[
\Phi=\operatorname{col}_i\phi(s_i^{\mathrm{ref}},u_i^{\mathrm{ref}}),
\qquad
G_\eta^{(0)}=\operatorname{col}_i g_{y,\eta}^{(0)}(s_i^{\mathrm{ref}},u_i^{\mathrm{ref}}).
\]

Then the stacked predictor evaluations are `hat Y_ref=Phi theta_b+G_eta^(0)`. Reference points can be measured histories or a declared auxiliary evaluation set. These are evaluations of a function at selected arguments, not automatically the model's evolving free-run trajectory.

The full one-step equation remains in the manuscript. Full stacks `S,S^+`, `I_N⊗A`, and `I_N⊗B_u` are unnecessary for the projection proof. Fixed shift/input terms have no newest-output component, so they are orthogonal to an inserted correction in this particular Euclidean realization.

### Step 7 — Explain rank without confusing realization with identification

G26 Eqs. (6)–(7) combine model distinguishability and data informativity to require full column rank for the displayed inverse and unique baseline regression estimate. State both ingredients, not “data condition alone guarantees rank.”

Evaluating the predictor or constructing its history shift uses no inverse and remains valid for rank-deficient `Phi`. Do not infer recoverability of raw parameters from representation equivalence.

### Step 8 — Reproduce the original subtraction, G26 Eqs. (8)–(11)

First show the full-rank source expression,

\[
a_\eta=(\Phi^T\Phi)^{-1}\Phi^T G_\eta^{(0)},
\]

then introduce the declared generalization:

\[
a_\eta=\Phi^\dagger G_\eta^{(0)},\quad
D_\eta^{(0)}=G_\eta^{(0)}-\Phi a_\eta
=(I-\Phi\Phi^\dagger)G_\eta^{(0)}.
\]

\[
\Phi^T D_\eta^{(0)}=0.
\]

`Phi Phi^dagger` is the Euclidean orthogonal projector onto the baseline column space. Write the pointwise corrected function

\[
d_\eta^{(0)}(s,u)=g_{y,\eta}^{(0)}(s,u)-\phi(s,u)a_\eta.
\]

Use it in BOTH the output predictor and the history update. Otherwise the allegedly equivalent IO and SS models use different feedback signals.

Interpretation: for any baseline coefficient vector `b`, `(Phi b)^T D=0`. This is orthogonality to all sampled baseline directions, not only to `Phi theta_b` at the current estimate. It is equivalent to `sum_i phi_i^T d_i=0`, not generally `phi_i^T d_i=0`.

### Step 9 — Show insertion preserves the original projection

Before adding latent rows, let `M_0=I_N⊗B_x`, `Phi_SS=M_0 Phi`, and `G_SS=M_0 G_eta^(0)`. Then `M_0^T M_0=I_(Np)` and

\[
\Phi_{\mathrm{SS}}^T G_{\mathrm{SS}}=\Phi^T G_\eta^{(0)},
\quad
\operatorname{rank}(\Phi_{\mathrm{SS}})=\operatorname{rank}(\Phi).
\]

The auxiliary coefficient calculated in state coordinates is the same Moore–Penrose coefficient. Provide the isometry/pseudoinverse argument, not an inverse applied to a rank-deficient normal matrix.

### Step 10 — Explain training versus deployment, G26 Eq. (13)

For a fixed reference set, recompute `a_eta` when learned parameters change. In a differentiable training evaluation it depends on `eta`; do not treat it as a free optimized parameter or detach it while claiming the projected derivative identity.

After fitting, use fixed `hat theta_b`, `hat eta`, and `hat a`:

\[
\hat y_k=\phi(s_k,u_k)\hat\theta_b
+g_{y,\hat\eta}^{(0)}(s_k,u_k)-\phi(s_k,u_k)\hat a.
\]

The coefficient is shared across reference samples. It is not separately fitted at each sample. New-point correction is causal as a function evaluation, but exact aggregate orthogonality is guaranteed on the construction set only.

### Step 11 — Introduce additional learned memory in IO form

This is the first model-class extension. Define `x_a,k∈R^r`, choose its initial value, and write

\[
\hat y_k=\phi(s_k,u_k)\theta_b
+g_{y,\eta}(s_k,u_k,x_{a,k}),
\qquad
x_{a,k+1}=g_{a,\eta}(s_k,u_k,x_{a,k}).
\]

The baseline IO output function remains unchanged. The ANN correction now reads extra learned memory. The latent equation supplies its next value.

Both functions are evaluated from OLD `(s_k,x_a,k,u_k)`; the written architecture does not require an algebraic loop or a direct same-step input `hat y_k` to `g_a`. If another architecture uses such an input, specify it separately.

Optional routing:

\[
g_{y,\eta}=R_y g_\eta,\qquad g_{a,\eta}=R_a g_\eta.
\]

State the dimensions of `R_y,R_a` and their common network output. Independent blocks are also possible. The additional state is distinct from explicit measured/predicted history, and is not automatically a physical mode.

### Step 12 — Realize the extended IO model exactly

Set `xi_k=col(s_k,x_a,k)` and substitute the complete output prediction into the history update:

\[
\xi_{k+1}
=\underbrace{\begin{bmatrix}
As_k+B_u u_k+B_x\phi(s_k,u_k)\theta_b\\0_r
\end{bmatrix}}_{F_b(\xi_k,u_k;\theta_b)}
+\underbrace{\begin{bmatrix}
B_xg_{y,\eta}(s_k,u_k,x_{a,k})\\
g_{a,\eta}(s_k,u_k,x_{a,k})
\end{bmatrix}}_{F_a(\xi_k,u_k;\eta)}.
\]

Retain the explicit current-output equation. The total predicted output, baseline plus the entire correction including its latent dependence, is what enters the newest history slot. There is no third independently additive “latent output” unless a particular network architecture defines one; `g_y(s,u,x_a)` already includes the latent readout.

The zero block expresses our baseline assignment. It does not prove additional states are zero during the augmented recursion. A latent update of the form `g_a=x_a+r_a(...)` can be encompassed by this definition; assigning a separate known latent baseline instead would change the zero-block argument.

### Step 13 — Define the extended reference evaluations

For the main proof choose fixed tuples

\[
\mathcal Z_{\mathrm{ref}}
=\{(s_i^{\mathrm{ref}},u_i^{\mathrm{ref}},x_{a,i}^{\mathrm{ref}})\}_{i=1}^N.
\]

Let `Phi` use the same history/input pairs and

\[
G_{y,\eta}=\operatorname{col}_i
g_{y,\eta}(s_i^{\mathrm{ref}},u_i^{\mathrm{ref}},x_{a,i}^{\mathrm{ref}}),
\qquad a_\eta=\Phi^\dagger G_{y,\eta}.
\]

This is a sufficient, simple causal construction: calculate the coefficient from reference evaluations before a recursive simulation and keep it shared during that simulation. Re-evaluate it when optimizing `eta`; freeze it after training.

Representative nonzero latent values improve coverage of latent-to-output dependence. Zero latent references still satisfy the algebraic identity and are not universally forbidden. They may miss features that vanish on that slice. Do not turn this coverage concern into a claim that additional latent rows need another orthogonality objective.

Do not automatically impose previous-epoch reference refresh as a theorem requirement. If discussing it, label it a design option with an exact within-stage identity and no transferred convergence claim.

### Step 14 — Prove the zero-block reduction and coefficient equivalence

At each reference tuple define

\[
K_i=\begin{bmatrix}B_x\phi_i\\0_{r\times q}\end{bmatrix},
\qquad
w_{\eta,i}=\begin{bmatrix}B_x g_{y,\eta,i}\\g_{a,\eta,i}\end{bmatrix}.
\]

For fixed arguments, `K_i` is the partial baseline transition derivative with respect to `theta_b`. It is NOT the total derivative of a recursive rollout.

Let

\[
E=\begin{bmatrix}B_x\\0_{r\times p}\end{bmatrix},\quad
M_N=I_N\otimes E,\quad
\Psi=\operatorname{col}_i K_i=M_N\Phi,\quad
\mathcal G_\eta=\operatorname{col}_i w_{\eta,i}.
\]

Then

\[
M_N^TM_N=I_{Np},\qquad M_N^T\mathcal G_\eta=G_{y,\eta},
\]

\[
\Psi^T\mathcal G_\eta=\Phi^T G_{y,\eta}.
\]

The `g_a` rows disappear because the baseline basis has zero rows there. State-space coefficients obey

\[
\Psi^\dagger\mathcal G_\eta=\Phi^\dagger G_{y,\eta}=a_\eta.
\]

Proof at any rank: a compact SVD `Phi=U_r Sigma_r V_r^T` gives orthonormal columns `M_N U_r` and hence `(M_N Phi)^dagger=Phi^dagger M_N^T`. The zero-rank case follows directly with zero pseudoinverses.

### Step 15 — Apply and prove the additional-state projection

Define

\[
d_\eta(s,u,x_a)=g_{y,\eta}(s,u,x_a)-\phi(s,u)a_\eta.
\]

The final corrected IO model is

\[
\hat y_k=\phi(s_k,u_k)\theta_b+d_\eta(s_k,u_k,x_{a,k}),
\qquad
x_{a,k+1}=g_{a,\eta}(s_k,u_k,x_{a,k}).
\]

The final corrected SS model is

\[
\xi_{k+1}=F_b(\xi_k,u_k;\theta_b)
+\begin{bmatrix}B_x d_\eta(s_k,u_k,x_{a,k})\\g_{a,\eta}(s_k,u_k,x_{a,k})\end{bmatrix}.
\]

On the reference set,

\[
\widetilde{\mathcal G}_\eta=\mathcal G_\eta-\Psi a_\eta,
\]

\[
\boxed{\Psi^T\widetilde{\mathcal G}_\eta
=\Phi^T(G_{y,\eta}-\Phi\Phi^\dagger G_{y,\eta})=0.}
\]

The subtraction has zero additional-state rows. Thus it leaves the numerical latent-update function unchanged at fixed parameters/arguments. The entire output/history correction `g_y(s,u,x_a)`, including its current dependence on learned memory, is projected. No extra projection of `g_a` against zero is needed.

Shared learned parameters can still change `g_a` during optimization. “Not directly restricted by this inner-product equation” does not mean “unaffected by training.”

### Step 16 — Optional stronger statement for this particular delay realization

At fixed tuples let

\[
b_i=\begin{bmatrix}As_i+B_u u_i\\0_r\end{bmatrix}.
\]

The fixed bookkeeping has zero newest-output slot, so

\[
b_i^T\begin{bmatrix}B_xd_i\\g_{a,i}\end{bmatrix}=0.
\]

Consequently,

\[
\sum_i F_b(\xi_i,u_i;\theta_b)^T\widetilde F_a(\xi_i,u_i;\eta)
=\theta_b^T\Phi^T D_{y,\eta}=0.
\]

This proves aggregate orthogonality of the full sampled baseline transition to the learned transition in this Euclidean delay realization. It is not a general result for arbitrary physical state-space baselines or arbitrary metrics. The primary result remains orthogonality to the parameter basis.

### Step 17 — Establish representation equivalence by induction

Use identical initial IO histories, latent initial values, parameters, input rule, and auxiliary coefficient in the two forms.

Base: the histories and latent states coincide. Step: equal current arguments yield equal output correction, predicted output, and next latent value; the shift stores precisely that output and current input. Therefore both representations generate the same subsequent histories and outputs at every finite step for which they are defined.

The projection does not change this argument when the same coefficient, or same specified causal coefficient schedule, is used in both representations. Equal input rules also cover a common causal controller with matching controller initialization.

Do not attribute the new augmented model to an exact rewriting of unextended G26. Adding `x_a` changes the class first; IO versus its own extended delay realization is the exact rewriting.

### Step 18 — Feedback/noise appendix, if included

Measured-history prediction stores `y_k`, not `hat y_k`. Replacing one by the other changes the estimator.

For the original stochastic process `y_k=H(s_k,u_k)+e_k`, direct storage gives

\[
s_{k+1}=As_k+B_u u_k+B_xH(s_k,u_k)+B_xe_k.
\]

This is our algebraic consequence, not a copied stochastic theorem. With additional states a generative stochastic extension can be defined analogously, with noise in the history output slot and zero direct noise in the latent rows. Defining that extension is not the same as assuming an imperfect fitted model exactly generates the measurements.

Avoid copying S21's output-noise equation indiscriminately: its chosen state already contains measured output and its noise conventions require separate interpretation. We need only its shift and measured/predicted-history distinction.

### Step 19 — Check reduction to the source construction

Set `r=0`, remove `g_a`, choose `g_y(s,u)` to be G26's original regressor function after permutation, and use the measured estimation histories as the reference set. The output predictor, stacked coefficient, subtraction and deployment expression then reproduce G26 Eqs. (3), (5), (8)–(11), and (13), using the pseudoinverse generalization where needed. The remaining history update is an exact realization of that predictor under the chosen feedback convention. This check recovers the original model-function construction, not a new statistical result.

## 6. Projection derivatives: precise scope

With fixed reference coordinates and `Phi` independent of `eta`,

\[
D_{y,\eta}=(I-P_\Phi)G_{y,\eta},\qquad
\partial_\eta D_{y,\eta}=(I-P_\Phi)\partial_\eta G_{y,\eta},\qquad
\Phi^T\partial_\eta D_{y,\eta}=0.
\]

Differentiating through `a_eta` produces this derivative. Stopping its gradient can preserve the forward orthogonal value while giving the optimizer a different derivative.

This is a derivative of reference evaluations. It does not prove block orthogonality of output sensitivities obtained by recursively propagating states, or zero covariance in joint free-run estimation.

Fixed `Phi` is the algebraic requirement for this simple differentiated identity. If only latent reference trajectories depend on `eta` while measured histories keep `Phi` fixed, the identity still holds for the total derivative of that `G_y(eta)` if its computation is differentiated consistently.

For the written architecture, measured-history latent updates do not read the projected predicted output directly. Their trajectory can therefore be generated before fitting the coefficient. In recursive simulation predicted outputs enter later `s_k`, so fitting a shared coefficient on that same corrected trajectory is an implicit problem unless a separate construction is specified. Do not claim all dynamic reference constructions are universally noncausal or impossible.

## 7. Rank, uniqueness, and numerical caveats

- The projector identity holds at any rank when `Phi^dagger` is the true Moore–Penrose inverse.
- Full-rank source formulas are reproduced first so readers can recognize G26's derivation.
- All least-squares minimizers `a+v`, `v∈ker(Phi)`, give the same `Phi a` on the reference set.
- At a new tuple, `phi(s,u)v` need not vanish. Thus coefficient choices can produce different deployment corrections. Use and state the minimum-norm convention.
- Reference-set rank deficiency need not be structural rank deficiency of the function class. Conversely, a pseudoinverse does not resolve structural physical-parameter ambiguity.
- A truncated SVD makes the residual exactly orthogonal to the **retained** column space, not necessarily to every nonzero direction of the original `Phi`. If discarded singular values are merely small rather than exactly zero, `Phi^T D=0` is only approximate.
- Ridge inverses generally do not produce an exact orthogonal projector.
- Orthogonality does not identify ANN parameters or latent coordinates uniquely.

## 8. Propagation, metrics, and permitted claims

The learned memory reaches current IO prediction through

\[
x_{a,k}\to g_{y,\eta,k}\to\hat y_k,
\]

and its next update reaches subsequent predictions through both `x_a,k+1` and the shifted history. **This is the IO timing convention here.** Do not import the physical gantry's one-step output delay into this equation: its measured output map is a separate structure.

Such propagation does not contradict the stated stacked state-function orthogonality. Future output-response orthogonality is a different property and is not required to define or prove the current construction. One concise limitation paragraph is sufficient. Do not turn it into a new trajectory penalty.

The zero-block reduction uses the chosen direct-sum Euclidean metric. Compatible block-diagonal metrics preserve the fact that zero latent baseline rows contribute nothing, but the simple isometry/projector formulas may need weighting. Full metrics with physical/latent cross blocks can couple latent writes into the condition and define another property.

Coordinate transformations preserving physical/history versus latent blocks require corresponding metric transformations if preserving the original geometry. Arbitrary coordinate mixing need not preserve the meaning of “zero latent baseline rows.” Do not claim Euclidean orthogonality is invariant under arbitrary rescaling.

| Statement | Verdict in this manuscript |
|---|---|
| Same finite IO and delay-state behavior for the same extended model, initialization and feedback | Exact realization identity |
| Stacked output-correction orthogonality on the declared reference set | Exact projector identity |
| Stacked augmented-write orthogonality to the baseline parameter basis | Exact isometry/zero-block derivation |
| Aggregate full-baseline-transition orthogonality in the displayed Euclidean delay realization | Exact, with the bookkeeping proof |
| Latent update orthogonal to zero baseline latent rows | Automatic under the specified block geometry |
| Per-sample output/history correction orthogonality | Not established; generally false |
| Latent states are output-irrelevant | False |
| No future effect can resemble a physical parameter change | Not guaranteed |
| Auxiliary coefficient rank deficiency has no effect on new-point predictions | False in general |
| True physical parameter recovery, estimator consistency, zero cross-covariance | Not transferred by this derivation |
| Stability or closed-loop ISS | Not established |
| Delay states equal gantry physical states | Not established |

If summarizing G26 theory, refer readers to its assumptions instead of reproducing a broad theorem audit. Exact coordinate rewriting preserves a genuinely identical estimator; adding learned states, changing feedback, using another loss, or changing reference selection may change that estimator.

## 9. Verification evidence and what was checked for this handoff

The existing MATLAB log reports 18 passed checks in R2025a Update 1 / Symbolic Math Toolbox 25.1, dated 2026-09-14. The script uses scalar output, two output delays, one input delay, two additional states, a nonlinear surrogate, and shared learned parameters. It checks projection, timing, four recursive steps, rank deficiency, measured history, and noise storage.

For this handoff the source and existing log were inspected; the MATLAB script was **not rerun**. No model code has changed. Finite symbolic unrolling supports the induction; it does not replace it.

An additional exact in-memory SymPy 1.14.0 audit was run in the `GraduationProject` environment on 2026-09-15 with `p=m=r=2`, `L=2`, `J=1`, `N=2`, and

\[
\Phi=\begin{bmatrix}1&0&1\\0&1&1\\2&1&3\\-1&2&1\end{bmatrix},
\qquad \operatorname{rank}\Phi=2.
\]

It passed the multi-output shift/insertion identities, repeated-insertion isometry, output-write extraction, rank preservation, Moore–Penrose coefficient equivalence, stacked orthogonality, unchanged latent rows, and aggregate full-baseline-transition overlap. Arbitrary symbolic output writes, latent writes, histories, inputs and baseline parameters were used. It also confirmed the extrapolation counterexample: `v=(-1,-1,1)^T` is in `ker(Phi)`, while the new regressor `[1,0,0]` does not annihilate it. These checks support the handoff's proofs; the next session should still review the general argument independently.

The tests do not establish equivalence to the rational physical gantry, correctness of a real encoder, closed-loop stability, statistical recovery, or a final training/reference schedule.

## 10. Recommended document organization

1. Scope and attribution.
2. Source IO model, notation, and current-input separation.
3. Baseline and additive delay-state realization.
4. Original stacked orthogonal construction and its preservation by output insertion.
5. Additional learned memory: IO model and combined state equation.
6. Augmented baseline basis, coefficient equivalence, and zero-block projection proof.
7. IO/SS representation equivalence.
8. Interpretation, reference-set scope, and restricted guarantees.
9. Optional feedback/noise and derivative appendices.
10. Symbolic verification note and bibliography.

Each numbered step should answer: What is being defined? Which source equation motivates it? What changes? Why is it valid? Avoid unexplained jumps and prohibit ambiguous reuse of symbols.

## 11. Acceptance checklist for the next session

- [ ] Independently verify G26 version and equation anchors.
- [ ] Use current `hat y_k` indexing and show current input explicitly.
- [ ] State input/output timing differences of the realization references accurately.
- [ ] Explain `A,B_u,B_x` with an explicit example and dimensions.
- [ ] Distinguish measured versus predicted histories; recurrence is already present.
- [ ] Keep output map explicit; no physical-only `C` substitution.
- [ ] Define additional states and initialize them before writing their recursion.
- [ ] Separate baseline and augmentation without an unexplained third output contribution.
- [ ] Stack only necessary writes; retain full one-step equation.
- [ ] Show `E^TE=I`, `M_N^T calG=G_y`, and pseudoinverse coefficient equivalence.
- [ ] Prove aggregate orthogonality and avoid per-sample claims.
- [ ] Explain why additional-state rows receive no subtraction.
- [ ] State projection coefficient training/deployment and derivative policies separately.
- [ ] State rank-deficient reference uniqueness versus possible extrapolation differences.
- [ ] No compulsory two-rollout penalty, metric redesign, or physical gantry adaptation.
- [ ] No unsupported parameter recovery, covariance, stability or ISS claims.
- [ ] Show the zero-additional-state reduction to the original output construction.
- [ ] Cite source/adapted equations clearly; use automatic local labels.
- [ ] Verify bibliography against PDFs; distinguish extended manuscript metadata.
- [ ] Compile and inspect the PDF if possible; report compiler limitations honestly.
- [ ] Provide a brief mathematical review identifying any correction to this handoff.

## 12. Copy-paste prompt for the next session

```text
Read CODEX.md and the project session rules, then read:
scripts/gantry/orthogonal-by-construction/documentation/additional-state-derivation-tex-handoff.md

Use this audited handoff to create a clean standalone LaTeX derivation and, if available, a compiled PDF for extending Györök's IO augmentation to additional learned states and realizing it in delay-state coordinates.

Independently check the handoff against the cited papers and existing symbolic script. Do not copy assertions just because they appear in project documents. In particular verify timing, dimensions, stacking, the isometric insertion proof, zero latent rows, and rank-deficient extrapolation.

Keep the manuscript inside scripts/gantry/orthogonal-by-construction/documentation/ using a new basename, e.g. gyorok-io-additional-state-derivation.tex. Preserve existing files. Use stepwise prose, automatic equation labels, citations with source equation anchors, explicit propositions and short proofs. Distinguish Györök's equations from our exact rewriting and additional-state extension.

The scope is discrete-time IO/delay-state augmentation first. Do not substitute physical gantry states or add two-rollout regularization, horizon projectors, or observability metrics as requirements. Retain Györök's hat y_k indexing and the stacked state-level definition approved by the supervisors. Additional states supply extra learned memory; recurrence through output history already existed.

Finish the TeX document, verify compilation if possible, and report the guarantee and limitations plus any corrections you made to the handoff. Do not implement training or production-model changes.
```
