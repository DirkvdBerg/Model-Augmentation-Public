# Additional-state input-output augmentation and its state-space realisation

Date: 2026-09-14. Scope: derive first in IO coordinates, then realise exactly
in delay-state coordinates. The gantry physical-state adaptation is separate.

Sources and their roles:

- [Gyorok 2026](https://arxiv.org/html/2511.01321v2): Eq. (1) defines the
  measured-history IO model; Eqs. (2)–(3) define the baseline and additive
  predictor; Eq. (5) stacks them; Eqs. (8)–(11) give the orthogonal construction;
  Eq. (13) evaluates the correction after training.
- [Bonassi, Xie, Farina and Scattolini, *An Offset-Free Nonlinear MPC scheme for
  systems learned by Neural NARX models*](../../../../literature/Orthogonality/IO-StateSpace/An%20Offset-Free%20Nonlinear%20MPC%20scheme%20for%20systems%20learned%20by%20Neural%20NARX%20models.pdf):
  Section II, Eqs. (1)–(3), PDF page 2, supplies the history-state construction
  and the matrix notation A, B_u, B_x.
- [Seel, Grøtli, Moe, Gravdahl and Pettersen, *Neural Network-Based Model
  Predictive Control with Input-to-State Stability*](../../../../literature/Orthogonality/IO-StateSpace/Neural%20Network-Based%20Model%20Predictive%20Control%20with%20Input-to-State%20Stability.pdf):
  Sections II–IV, Eq. (2) and Eqs. (6)–(7), PDF page 2, give the history state
  and explicit shift update; Eqs. (8)–(9) distinguish measured and predicted
  output histories.

The MPC papers are used only for the realisation and feedback conventions,
not their closed-loop stability theorems. Equations below are our adaptation
with explicit timing, not verbatim quotations. The additional learned-state
extension and its projection are our derivation. The nonlinear physical-baseline
adaptation in Gyorok 2025 is a later step, outside this equivalence check.

**Indexing convention.** The main derivation keeps Gyorok's indexing exactly:
his regressor at time k contains outputs only through y_(k-1), and his Eqs. (2)
and (3) predict hat y_k. After that prediction is formed, s_(k+1) stores hat y_k
in recursive simulation or measured y_k in measured-history prediction. Bonassi
and Seel instead put y_k in the state at time k and predict y_(k+1); their formulas
support the same shift-register construction after a one-sample index shift.

## 1. IO model with additional states

Let the output dimension be p, input dimension m, output history length L >= 1,
and input history length J >= 1. Empty histories can be omitted. Use x_a for the
additional states; do not confuse their dimension with the paper's lag orders.

Define the history and current regressor by

\[
s_k=\operatorname{col}(y_{k-1},\ldots,y_{k-L},
                      u_{k-1},\ldots,u_{k-J}),\qquad z_k=(s_k,u_k).
\]

This separates the current input from Gyorok's Eq. (1) regressor, up to an
explicit reordering of its entries. Here s_k denotes only stored history;
x_a,k denotes learned memory, and xi_k = col(s_k,x_a,k) denotes their combined
state. Retain H for an output predictor, rather than Bonassi's eta, because
eta here denotes learned parameters.

Gyorok's Eq. (2) is hat y_k = phi(z_k) theta_b. His Eq. (3) adds the
learning component at that same output index. Retain this hat y_k convention
and extend the learned contribution with additional states:

\[
\boxed{\begin{aligned}
\hat y_k &= \phi(z_k)\theta_b+g_{y,\eta}(z_k,x_{a,k}),\\
x_{a,k+1} &= g_{a,\eta}(z_k,x_{a,k}).
\end{aligned}} \tag{IO}
\]

Here phi is p by n_theta. The two g blocks may share parameters; in a routed
network they are R_y g_eta and R_a g_eta. Specify x_a,0 (fixed or encoder supplied)
and the initial input-output history. The baseline defines no latent update.
This adds recurrent memory to the paper's static learning function; it is an
extension of the model class even before we rewrite its coordinates.

## 2. Exact state-space realisation

Following Bonassi's Section II, Eqs. (2)–(3), let A shift the stored output and
input histories by one position, leaving zero slots for their newest samples.
Let B_x insert a p-vector into the newest-output slot; let B_u insert the
current m-vector input into the newest-input slot. These are fixed routing
matrices, not a linearisation of the nonlinear predictor. Thus

\[
s_{k+1}=A s_k+B_u u_k+B_x v_k,\qquad B_x^\top B_x=I_p.
\]

The meaning of v_k matters: v_k = hat y_k in deterministic free simulation,
and v_k = measured y_k when updating a one-step predictor from observations.
Seel's Eqs. (6)–(9) support this distinction. Both sources use a predictor for
y_(k+1) and a state that includes y_k. We retain Gyorok's current-output
prediction convention: s_k contains outputs only through y_(k-1), and u_k
is external. Their output-selection equation y_k = C x_k must therefore not
be copied without aligning both output and input timing.

For free simulation, set xi_k = col(s_k,x_a,k). Then

\[
\boxed{\begin{aligned}
\xi_{k+1}&=
\underbrace{\begin{bmatrix}
A s_k+B_u u_k+B_x\phi(z_k)\theta_b\\0
\end{bmatrix}}_{F_b(\xi_k,u_k;\theta_b)}+
\underbrace{\begin{bmatrix}
B_x g_{y,\eta}(z_k,x_{a,k})\\g_{a,\eta}(z_k,x_{a,k})
\end{bmatrix}}_{F_a(\xi_k,u_k;\eta)},\\
\hat y_k&=\phi(z_k)\theta_b+g_{y,\eta}(z_k,x_{a,k}).
\end{aligned}} \tag{SS}
\]

The output map remains parameter dependent and contains the learned term.
It must not silently be replaced by a physical-state-only output map. In these
coordinates B_x^T s_k is y_(k-1), whereas B_x^T s_(k+1) is hat y_k in free simulation.

For scalar outputs with two output delays and one input delay,

\[
A=\begin{bmatrix}0&0&0\\1&0&0\\0&0&0\end{bmatrix},
\quad B_u=\begin{bmatrix}0\\0\\1\end{bmatrix},
\quad B_x=\begin{bmatrix}1\\0\\0\end{bmatrix}.
\]

The delay states are stored signals, not additional learned states or physical
positions and velocities. This realisation need not be minimal.

## 3. Why IO and SS are equivalent

Base case: initialise s_0 with exactly the IO recursion's initial history and
use the same x_a,0. Inductive step: if histories and latent states coincide at
k, both evaluate the same phi, g_y and g_a on the same arguments. Their outputs
and next latent states coincide. The shift equation stores this output and
current input in exactly the history required at k+1. Equivalence follows for
every finite step on which the maps are defined. No stability, observability,
parameter linearisation, or regressor rank assumption is needed for this identity.

For measured-history prediction, both implementations must shift measured y_k,
not hat y_k. Starting from Gyorok's Eq. (1), before adding learned states,
write y_k = H(s_k,u_k) + e_k. Substitution in the history update gives

\[
\boxed{\begin{aligned}
s_{k+1}&=A s_k+B_u u_k+B_xH(s_k,u_k)+B_xe_k,\\
y_k&=H(s_k,u_k)+e_k.
\end{aligned}}
\]

This noise placement is our direct algebraic consequence of storing the actual
output, not a stochastic formula attributed to the MPC papers. With additional
states, the predictor is H(s_k,x_a,k,u_k), and the combined state update includes
col(B_x e_k,0). An output-noise-only model with a noise-free history update is
a different stochastic model. These feedback conventions are checked separately
in the companion script.

## 4. Add the orthogonal-by-construction correction

The complete one-step state equation remains the realisation established above.
For the stacked projection, however, only the physics and learned writes are
stacked. At fixed evaluation tuples, the history-shift and input-insertion terms
`A s_k + B_u u_k` contain neither parameterised model contribution and occupy
state rows different from the newly predicted-output insertion. They therefore
cannot create overlap between the baseline parameters and learning parameters,
so no stacked `S`, `S^+`, `I_N kron A`, or `I_N kron B_u` notation is needed.

Following the stacked model in Gyorok's Eq. (5) and the coefficient/subtraction
in Eqs. (8)–(10), choose the SAME fixed reference tuples (z_i,x_a,i^ref) in both representations.
Stack Phi = col_i phi(z_i), G_y = col_i g_y,eta(z_i,x_a,i^ref). Define

\[
a_\eta=\Phi^\dagger G_y,\qquad
d_\eta(z,x_a)=g_{y,\eta}(z,x_a)-\phi(z)a_\eta.
\]

Replace g_y by d_eta in both (IO) and (SS), including the output equation.
Neither changes g_a. The same induction proves equivalence of the projected
models for a shared coefficient (or shared causal coefficient-update schedule).
Here the coefficient is held fixed over each compared rollout, consistent with
the prediction-time correction in Gyorok's Eq. (13). Our pseudoinverse notation
also covers rank deficiency; his displayed inverse requires full column rank.

Let B_aug = col(B_x,0_(n_a by p)) and Q = I_N kron B_aug. Then

\[
\Psi=Q\Phi,\qquad
W=\operatorname{col}_i\begin{bmatrix}B_x g_{y,i}\\g_{a,i}\end{bmatrix},
\qquad Q^\top Q=I,
\]
\[
\Psi^\dagger W=\Phi^\dagger G_y=a_\eta,\qquad
\boxed{\Psi^\top(W-\Psi a_\eta)
=\Phi^\top(G_y-\Phi a_\eta)=0.}
\]

The last identity is our extended-state counterpart of Gyorok's Eq. (11).
This holds at any rank with the Moore-Penrose inverse. The subtraction has zero
latent rows. In these unscaled Euclidean coordinates, the history-shift term
A s+B_u u has zero newest-output slot and hence is pointwise orthogonal to B_x d.
Consequently the stacked FULL baseline transition F_b is also orthogonal to
the corrected learned transition when the displayed identity holds. Arbitrary
coordinate scaling or metrics require transforming the inner product explicitly.

For fixed Phi and fixed reference coordinates, differentiating through a_eta gives
Phi^T d(G_y-Phi a_eta)/deta = 0. Detaching a_eta generally destroys this identity.
These are reference-set identities; representation equivalence does not make
them exact on arbitrary free-running trajectories.

An important IO-specific distinction: with measured histories, x_a,k+1 = g_a(z_k,x_a,k)
does not feed back the current predicted output. Its trajectory can be generated
causally before fitting a_eta, although it depends on eta. That situation is
different from free simulation, where predicted outputs enter later regressors.
The fixed-tuple choice here isolates the first equivalence check; it is not a
claim that every dynamic IO projection requires that choice.

## 5. What this establishes, and what comes next

An exact rewriting leaves predictions, loss and derivatives unchanged if the
same feedback, initialisation, parameterisation and projection procedure are used.
It cannot invalidate or strengthen a theorem about that same estimator. Adding
latent dynamics changes the model class; statistical conclusions need reassessment.

The physical gantry formulation has a different output map and injection location:

\[
x_{p,k+1}=f_\theta(x_{p,k},u_k)+R_p g_\eta(x_{p,k},x_{a,k},u_k),
\quad x_{a,k+1}=R_a g_\eta(\cdot),\quad y_k=C_p x_{p,k}+D u_k.
\]

No equivalence to that physical realisation is proved here. The present comparison
is IO versus its exact delay-state SS realisation. Physical-coordinate projection,
nonlinear parameter dependence, the linearisation point and stability are later steps.

## 6. Symbolic verification

Companion: `../code/io_additional_state_equivalence.m`.
The existing script retains its original variable names: its h is this document's
s, its combined state is xi, and its T, E, B, Eaug correspond respectively to
A, B_x, B_u, B_aug. This document-only notation change does not alter the checks.
Uses exact rational reference points and a polynomial surrogate with two latent
states, nonlinear input/history interaction, and a parameter shared across the
output and latent blocks. Checks arbitrary symbolic one-step identities and
four-step recursion with rational coefficients and symbolic initial conditions.
The general all-horizon argument is the induction in Section 3; finite unrolling
is a supporting implementation check, not its replacement.

Run from the repository root:

```matlab
run('scripts/gantry/orthogonal-by-construction/code/io_additional_state_equivalence.m')
```

Results: all 18 checks passed on 2026-09-14 in MATLAB R2025a Update 1 with
Symbolic Math Toolbox 25.1, exit code 0. Checks 12.1–12.4 are counted separately.
The measured-versus-predicted feedback distinction uses an explicit nonzero
counterexample, not failure to prove an identity. Log:
`../code/logs/io_additional_state_equivalence.log`.
A zero symbolic residual verifies the stated identity, not physical-coordinate
equivalence, parameter recovery, stability or covariance.
