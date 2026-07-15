# Literature Reference: Passive / Dissipative Augmentation of a Free-Integrator Baseline

**Purpose.** Catalog of the sources behind the passivity-constrained augmentation direction, with direct
quotes and full references. This is a REFERENCE document (what each source says); the design reasoning and
plan live in `docs/drift-diagnosis-status.md` (§5b, §5f-§5L). Compiled 2026-07-10.

## Provenance legend (READ THIS BEFORE QUOTING ONWARD)
Quote reliability differs by how the source was accessed. Every quoted item is tagged:
- **[disk]** = read from the on-disk PDF rendered to page images; the text was seen directly. Quotes are
  transcribed from the rendered page and cited by section / equation / figure. Reliable; re-verify a
  character-exact quote against the PDF before it goes into the thesis.
- **[online-primary]** = read from the publisher/arXiv page. Existence + attribution verified online.
- **[online-extract]** = obtained via automated page extraction (a small model summarizing the page).
  **NOT guaranteed verbatim.** Treated as a paraphrase/lead; MUST be verified at the primary source before
  any thesis citation. Marked inline as `[extract - verify]`.
- **[internal-AI]** = an AI-generated in-project literature study (not a peer-reviewed paper); its own
  primary citations are listed and must be verified at source.

**UPDATE 2026-07-10: the `[online-*]` PDFs are now ON DISK** in `literature/passivity-augmentation/`
(downloaded from arXiv by ID; filenames prefixed with the arXiv ID). So every `[extract - verify]` /
`[online-*]` claim below can now be checked against the local PDF. Files:
`2604.18277_DiLaR-PINN...pdf`, `2104.05942_RENs...pdf`, `1305.1079_Mabrok-2014-free-body...pdf`,
`2504.19497_NINODE...pdf`, `2504.12441_passive-LuGre-PINN...pdf`,
`2011.13492_dissipative-deep-neural-dynamical-systems.pdf`,
`2410.00976_learning-dissipative-chaotic-boundedness.pdf`, `1112.4232_Lavretsky-Gibson...pdf`,
`2011.14610_nonlinear-NI-free-body-consensus.pdf`. NOT localizable: the Lan Jia (2023) TU Delft MSc gantry
friction numbers (§D2) -- not on arXiv; still verify-by-hand.

---

## A. The augmentation framework we implement

### A1. Hoekstra, Verhoek, Toth, Schoukens (2025) -- the augmentation method  [disk]
**Reference.** J.H. Hoekstra, C. Verhoek, R. Toth, M. Schoukens, "Learning-based model augmentation with
LFRs", *European Journal of Control* 86 (2025) 101304. doi:10.1016/j.ejcon.2025.101304.
`literature/augmentation/hoekstra2025_lfr-augmentation-ejc.pdf`.

**What it establishes.** The LFR-based augmentation structure we run: baseline `phi_base` + learning
`phi_aug` connected by an interconnection matrix `S` (Fig 1); dynamic parallel augmentation adds extra
states; SUBNET truncated multiple-shooting loss with an encoder for the per-subsection initial state;
parameter regularization keeping baseline parameters near nominal.

**Quotes (transcribed from the rendered PDF, cited by location).**
- Abstract: "We introduce a novel *linear-fractional-representation* (LFR) model structure that allows for
  the unified representation of various augmentation structures ... and an identification algorithm for
  estimating the proposed structure together with appropriate initialization methods." [disk, Abstract]
- Well-posedness: "Condition 1 (*Well-posedness*). For every state `x_k` and input `u_k`, model (4) admits
  a unique solution `z_k`." Enforced via "Theorem 1 (*Acyclic Directed Graph*) ... `S` is acyclic if and
  only if it has a topological ordering." [disk, Sec 3.2]
- Parameter regularization: `V_reg(theta) = || Lambda (theta_base - theta_base^*) ||_2^2` with
  `Lambda = ( (1/V_MSE) eps )^{1/2} diag(theta_base^*)^{-1}`. [disk, Sec 3.4, Eqs 6-7]
- The demonstration system: a 3-DOF hardening mass-spring-damper, Fig 3 caption: "The linear dynamics
  `m_1` and `m_2` are assumed known, while the dynamics of `m_3` and the contribution of `a_1` are aimed
  to be learnt from data." [disk, Fig 3]
- Result: "we have obtained an accurate model whose dominant behavior is given by an interpretable
  baseline model" and (Sec 4.3) the learning components augment the baseline "without replacing the
  baseline model with their own dynamics." [disk, Sec 4.3]

**Relevance / scope note.** The 3-DOF MSD example is the direct structural analog of our absorber
augmentation, EXCEPT every mass in that example has a spring (`k_1,k_2,k_3`), so there is no free
integrator and no drift is possible. The paper's guarantees (acyclic well-posedness, parameter
regularization) do not concern free-integrator / marginal modes.

### A2. Drenth, Hoekstra, Schoukens, Toth (2025) -- the rational LPV-LFR baseline  [disk]
**Reference.** R. Drenth, J.H. Hoekstra, M. Schoukens, R. Toth, "Efficient Learning of Affine and Rational
Dependency LPV Models With Linear Fractional Representation", 2025.
`literature/lpv-lfr/drenth2025_lpv-lfr-rational.pdf`.

**What it establishes.** How an LPV system with rational scheduling dependence (our `M(Y)`) is written as
an LFR `{M, Delta(p)}` and kept well-posed by a direct, constraint-free parametrization.

**Quotes (transcribed).**
- Well-posedness: "Definition 1. The LPV-LFR model defined by the pair `{M, Delta(p)}` ... is *well-posed*
  if `I - D_zw Delta(p)` is non-singular ... for all possible realizations of the scheduling variable."
  [disk, Def 1]
- The enforced condition: "Condition 5. The spectral radius `rho` of `D_zw` is strictly smaller than 1,
  i.e., `rho(D_zw) < 1`." [disk, Condition 5]
- The parametrization: "we define `D_zw = e^{-N}`, `N > 0` ... As the matrix exponential maps negative
  definite matrices to matrices `D_zw < I`." [disk, Sec 4.2, Eq 16]

**Relevance.** Same well-posedness mechanism as A1 (acyclic) and A4 (Cayley): parametrize `D_zw` with
`rho(D_zw) < 1` so `I - D_zw Delta` is invertible. Reusable for the well-posedness of our baseline and of
any added augmentation states. Silent on drift / marginal modes.

### A3. Gyorok, Hoekstra, Kon, Peni, Schoukens, Toth (2025) -- orthogonal-projection steering  [disk]
**Reference.** B.M. Gyorok, J.H. Hoekstra, J. Kon, T. Peni, M. Schoukens, R. Toth, "Orthogonal
projection-based regularization for efficient model augmentation", *Proc. Machine Learning Research* vol.
283, 7th Annual Conf. on Learning for Dynamics and Control (L4DC), 2025. arXiv:2501.05842.
`literature/Orthogonality/Hoekstra - Orthogonal projection-based regularization for efficient model.pdf`.

**What it establishes.** The interpretability layer (our R4/C5): a soft regularizer that penalizes the
component of the ANN output lying in the subspace the FP model can already generate, so the ANN learns
only the orthogonal residual and the physical parameters stay identifiable.

**Quotes (transcribed).**
- The mechanism: "the aim is to penalize such `eta` parameters that result in significant contributions in
  this subspace, by adding `|| Pi_{X,U} f_eta^ANN(X,U) ||_2^2` to (4a), as an orthogonality-promoting
  term. Thus, it can be interpreted as a targeted `l_2` regularization of the ANN that only penalizes the
  directions of `eta` which generate output in the subspace of the model `f_theta`." [disk, Sec 3, around
  Eq 12-13]
- Projection: `Pi_{X,U} = Q_{X,U} Q_{X,U}^T` from the reduced SVD `Phi(X,U) = Q Sigma V^T`. [disk, Eqs
  10-12]
- Cost: `V^{(orth)}(eta,theta) = V^{(sec)}(eta,theta) + beta || Pi_{X,U} f_eta^ANN(X,U) ||_2^2`. [disk,
  Eq 13]
- Remark 1: "In practice, often only a small amount of orthogonal regularization is needed to achieve the
  desired complementarity." [disk, Remark 1]
- Nonlinear-in-parameters extension: Taylor-expand `f_theta` about `theta_bar`, use the Jacobian
  `partial f_theta / partial theta` as the regressor, extended parameter `[theta;1]`. [disk, Sec 4, Eqs
  15-19]

**Relevance.** This is the canonical "steer the learning in a KNOWN subspace" template (not a magnitude
cap). The no-drift / dissipativity constraint can be framed as a sibling steering in the same cost.

### A4. Gyorok, Drenth, Verhoek, Peni, Schoukens, Toth (2026) -- constraint-free stability  [disk]
**Reference.** B.M. Gyorok, R. Drenth, C. Verhoek, T. Peni, M. Schoukens, R. Toth, "Data-driven
augmentation of first-principles models under constraint-free well-posedness and stability guarantees",
2026. arXiv:2604.11421 (submitted to Automatica).
`literature/augmentation/Data-driven augmentation of first-principles models under constraint-free
well-posedness and stability guarantees.pdf`.

**What it establishes.** Direct, constraint-free parametrizations giving (i) LFR well-posedness (Cayley
transform on `D_zw`) and (ii) stability of the augmented model via CONTRACTION.

**Quotes (transcribed).**
- Contraction definition: "Definition 11. The model represented by (10) is said to be *contracting* with
  rate `alpha in (0,1)` if for any two initial conditions ... `|| x_i(k) - x_j(k) ||_2 <= K alpha^k
  || x_i(0) - x_j(0) ||_2`, for all `k > 0`." [disk, Def 11]
- Well-posedness via Cayley: "Lemma 8 (Generalized Cayley transform) ... `M^T M < I` if and only if there
  exist ..." used to parametrize `D_zw` so `|| D_zw ||_2 < 1/L`. [disk, Sec 3.4, Lemma 8]
- ANN Lipschitz regularization: `r_L(theta_a) = rho_L max{ prod_i ||W_i||_2 - L, 0 }^2`. [disk, Eq 21]

**Relevance / scope note.** Contraction `alpha < 1` (strict) excludes the pole at the origin: applied to
our K=0 X/Y axes it damps the free integrator and destroys the zero-stiffness physics. This is the result
our contribution must RELAX to marginal (`<= 1`).

---

## B. Dissipative / passive learned augmentation

### B1. DiLaR-PINN (2026) -- KEY PRECEDENT: skew-dissipative latent residual  [disk-verified 2026-07-10 -- see §G1]
**Reference.** **Youyuan Long, Gokhan Solak, Arash Ajoudani** (Istituto Italiano di Tecnologia),
"Dissipative Latent Residual Physics-Informed Neural Networks for Modeling and Identification of
Electromechanical Systems", arXiv:2604.18277v1 (20 Apr 2026), **accepted to IFAC**. Method claims below
(Eq 5 = Prop 1 = Eq 3 = Prop 3) are all CONFIRMED at the primary source -- see §G1.

**Abstract (via online extraction -- `[extract - verify]` against the arXiv abstract).**
> "Accurate dynamical modeling is essential for simulation and control of embodied systems, yet
> first-principles models of electromechanical systems often fail to capture complex dissipative effects
> such as joint friction, stray losses, and structural damping. While residual-learning physics-informed
> neural networks (PINNs) can effectively augment imperfect first-principles models with data-driven
> components, the residual terms are typically implemented as unconstrained multilayer perceptrons (MLPs),
> which may inadvertently inject artificial energy into the system. To more faithfully model the
> dissipative dynamics, we propose DiLaR-PINN, a dissipative latent residual PINN designed to learn
> unmodeled dissipative effects in a physically consistent manner. Structurally, the residual network
> operates only on unmeasurable (latent) state components and is parameterized in a skew-dissipative form
> that guarantees non-increasing energy for any choice of network parameters. To enable stable and
> data-efficient training under partial measurability of the state, we further develop a recurrent rollout
> scheme with a curriculum-based sequence length extension strategy. We validate DiLaR-PINN on a
> real-world helicopter system and compare it against four baselines: a pure physical model (without a
> residual network), an unstructured residual MLP, a DiLaR variant with a soft dissipativity constraint,
> and a black-box LSTM. The results demonstrate that DiLaR-PINN more accurately captures dissipative
> effects and achieves superior long-horizon extrapolation performance." `[extract - verify]`

**Method (via online extraction of the HTML -- `[extract - verify]`; equation labels approximate).**
- Residual parametrization: `r_phi(x,u) = (S_phi(x,u) - K_phi(x,u)) grad_{x^lat} V(x)`, with `S_phi`
  skew-symmetric (`S_phi^T = -S_phi`) and `K_phi = L_phi L_phi^T >= 0` (Cholesky-like). [extract - verify,
  their Eq ~5]
- Dissipativity condition: `grad_{x^lat} V^T r_phi <= 0` for all `(x,u)`, proved (their Proposition ~1)
  by `v^T S v = 0` (skew) and `v^T K v >= 0` (PSD), holding "for all parameters, no additional constraints
  needed during training." [extract - verify, their Prop 1]
- Interconnection: residual affects ONLY latent states, `f = f_phys + [0; r_phi]` (observed-state equation
  gets zero residual). [extract - verify, their Eq ~3]
- Stability: (their Proposition ~3) the augmented system remains ISS IF the nominal `f_phys` is ISS, i.e.
  there is an ISS-Lyapunov function with `grad V^T f_phys <= -alpha_3(||x||) + sigma(||u||)`.
  [extract - verify, their Prop 3]
- Training: RK4 recurrent rollout, variance-weighted MSE on measured states, curriculum sequence-length
  extension. [extract - verify]

**Relevance / scope note.** The skew-minus-PSD residual (`(S-K) grad V`) is structurally identical to the
port-Hamiltonian block in `drift-diagnosis-status.md` §5i (`(J-R) gradH`). Its stability guarantee (their
Prop 3) REQUIRES an ISS baseline; a free integrator is not ISS (not 0-GAS), so the guarantee's premise
does not hold for our K=0 X/Y axes. Verify the exact propositions at the primary source before citing.

### B2. Recurrent Equilibrium Networks (RENs)  [online-primary / prior verification]
**Reference.** M. Revay, R. Wang, I.R. Manchester, "Recurrent Equilibrium Networks: Flexible Dynamic
Models with Guaranteed Stability and Robustness", arXiv:2104.05942; IEEE Trans. Automatic Control.

**Quotes / findings.**
- "all models in the proposed class are contracting -- a strong form of nonlinear stability -- and models
  can satisfy prescribed incremental integral quadratic constraints (IQC), including Lipschitz bounds and
  incremental passivity." `[extract - verify]`
- Prior primary-source check (recorded in `drift-diagnosis-status.md` §9): contraction is enforced with
  respect to a strictly positive-definite metric `P > 0`; the incremental-passivity supply rate is
  enforced JOINTLY with contraction, not as an alternative. No published `P >= 0` (semidefinite / marginal)
  relaxation.

**Relevance.** Contraction (`P > 0`, `||A|| < 1`) excludes the pole at the origin; the "free bias" that
would let a REN represent friction reintroduces a constant-at-rest force. Not a marginal-preserving option
as published; relaxing `P > 0` to `P >= 0` is a candidate route to our gap.

### B3. Related dissipative / boundedness families (triaged; none cover the free integrator)
- "Dissipative Deep Neural Dynamical Systems", arXiv:2011.13492 -- learned dynamics guaranteed dissipative
  with a "bounded and positively invariant level set" (trajectories converge to an attractor).
  `[online-extract]`. Requires return to a bounded set -> excludes the free integrator.
- "Learning Dissipative Chaotic Dynamics with Boundedness Guarantees", arXiv:2410.00976 -- boundedness via
  a dissipative projection layer; a primary read (2026-07-10) confirmed it "assumes the underlying system
  is already dissipative with trajectories converging to a bounded attractor" and "cannot bound
  free-integrator position states." `[online-primary]`.
- "ECO: Energy-Constrained Operator Learning for Chaotic Dynamics with Boundedness Guarantees",
  arXiv:2512.01984 `[online-extract]`; "Stable Port-Hamiltonian Neural Networks", arXiv:2502.02480 --
  "global Lyapunov stability" `[online-extract]`. Both attractor / asymptotic-stability based.

---

## C. Negative-Imaginary / free-body theory

### C1. Mabrok et al. (2014) -- free-body NI (the classical marginal-case theory)  [online-primary]
**Reference.** M.A. Mabrok, A.G. Kallapur, I.R. Petersen, A. Lanzon, "Generalizing Negative Imaginary
Systems Theory to Include Free Body Dynamics: Control of Highly Resonant Structures With Free Body
Motion", *IEEE Trans. Automatic Control* 59(10):2692-2707, Oct 2014. arXiv:1305.1079.

**Findings (existence + attribution verified online; exact conditions NOT yet read at primary).**
- The paper "generalizes negative imaginary systems theory to include free body dynamics for control of
  highly resonant structures with free body motion" and gives "necessary and sufficient conditions for the
  stability of positive feedback control systems where the plant is NI according to the new definition and
  the controller is strictly negative imaginary", including "a related stability result on the feedback
  interconnection of negative imaginary systems with poles at the origin." `[extract - verify]`
- **Attribution correction:** arXiv:1305.1079 is Mabrok et al. (2014), NOT Lanzon-Petersen (an earlier
  separate NI paper). Earlier project notes had this wrong.

**Relevance.** The only classical framework native to poles at the origin (free body) in colocated
force->position systems. LTI / transfer-function only; the nonlinear + semidefinite-storage subcase is not
worked out (the theory gap).

### C2. NINODE (2025) -- neural NI, but a controller  [online-extract]
**Reference.** "Negative Imaginary Neural ODEs: Learning to Control Mechanical Systems with Stability
Guarantees", arXiv:2504.19497 (2025).
- "a novel negative imaginary neural ordinary differential equation (NINODE) controller ... can serve as a
  controller that asymptotically stabilizes a negative imaginary plant under certain conditions. For
  mechanical plants with colocated force actuators and position sensors, the stability conditions can be
  translated into regularity constraints on the neural networks used in the controller." `[extract -
  verify]`

**Relevance.** Neural + NI + colocated force/position = adjacent to us, but it is a CONTROLLER that
stabilizes (damps) the plant, and (per project §5j notes) it re-imposes a strict DC-gain condition, so it
does not preserve a marginal forward-model mode. Not a forward-augmentation solution.

### C3. Nonlinear NI with free-body motion  [disk-verified 2026-07-10 -- see §G8]
**Reference.** K. Shi, I.R. Petersen, I.G. Vladimirov, "Output Feedback Consensus for Networked
Heterogeneous Nonlinear Negative-Imaginary Systems with Free Body Motion", arXiv:2011.14610 (2021).
- ~~A targeted extraction did NOT surface a nonlinear free-body semidefinite-storage theorem usable for a
  learned forward model; it is a networked-consensus / controller paper. Lead only.~~ **[CORRECTED after
  full on-disk read -- §G8.] This paper DOES contain the nonlinear free-body semidefinite-storage NI
  theorem** the drift-doc elsewhere calls "unworked-out": **Definition 1** = nonlinear NI system via a
  **positive-SEMIdefinite** storage function `V∈C¹`, `V̇ ≤ uᵀẏ̃`, explicitly **including poles at the origin
  (free body motion)**; **Definition 2** = nonlinear OSNI; plus a nonlinear NI⊗OSNI stability theorem.
- **It does NOT close our gap** because it is (i) analytical (not learned/neural) and (ii) a
  controller/consensus construction, not a forward-model augmentation. But it REFUTES the "NI is LTI-only /
  nonlinear-NI-semidefinite is unworked-out" framing: the analytical theory exists (also Ghallab-Petersen
  arXiv:2201.00144). Our gap is the LEARNED + forward-augmentation combination on top of this theory.
- **Direct quotes (verbatim from on-disk PDF; full set + lineage quote in §G8):**
  - *Def 1, nonlinear NI via positive-SEMIdefinite storage (p.2):* "The system (1), (2) is said to be a
    nonlinear negative-imaginary (NI) system if there exists a **positive semideﬁnite storage function**
    V : Rⁿ → R of class C¹ such that V̇(x(t)) ≤ u(t)ᵀ ẏ̃(t), (3) for all t ≥ 0, where ỹ(t) = h(x(t)). (4)"
  - *Remark, marginal-mode point (p.2):* "In contrast to Deﬁnition 3 in [19], which excludes linear NI
    systems with poles at the origin, Deﬁnition 1 now includes all linear NI systems ... **by allowing the
    storage function of the system to be positive semideﬁnite instead of positive deﬁnite.**"
  - Cite for the DEFINITION / supply rate only (analytical, not learned; controller, not augmentation).

---

## D. Friction (nonzero-at-rest, dissipative)

### D1. Passive learned LuGre via PINN (2025)  [disk-verified 2026-07-10 -- see §G5]
**Reference.** **Asutay Ozmen, João P. Hespanha, Katie Byl**, "Learning Transferable Friction Models and
LuGre Identification via Physics-Informed Neural Networks", arXiv:2504.12441 (2025).
- Findings (CONFIRMED on disk, §G5): captures a NONZERO force at rest (Coulomb/pre-sliding) via a LuGre
  bristle latent state with learned parameters; the block maps velocity (and internal state) -> friction
  force and can be cascaded into a larger model. **PROVENANCE FIX:** the string *"not provably passive by
  construction ... only empirical validation"* is a correct INFERENCE, **not a quote** -- the words
  "passive"/"dissipative" appear ZERO times in the paper (it never discusses passivity). Do not cite it as
  a quotation. The passivity guarantee, if wanted, must come from the block structure (DiLaR-PINN/PH).

**Relevance.** A ready friction MODEL (nonzero-at-rest, LuGre-structured). The passivity GUARANTEE, if
wanted, must come from the block structure (a skew-dissipative / PH form), not from this fit.

### D2. Classical + gantry friction sources (via internal AI studies)  [internal-AI, secondary]
From `literature/augmentation/Literatuuronderzoek voor Dahl-frictiestaten in een H-type dual-drive gantry
stage.pdf` (an in-project AI literature study; primaries below to be verified at source):
- Dahl / LuGre bristle state: `z_dot = v - (sigma |v| / F_c) z`, `F = sigma z`, nonzero at rest via `z`.
- **Gantry friction parameters (Lan Jia, 2023, MSc thesis, TU Delft, "Friction Identification on the
  Gantry Stage", sponsor Prodrive):** position-dependent over the X-workspace, `F_s ~ 1.9-3.2 N`,
  `F_c ~ 1.4-2.75 N`, viscous `sigma_2 ~ 30-33`, Stribeck `v_0 ~ 1e-4 m/s`, `sigma_0, sigma_1`
  position-dependent. `[internal-AI - verify Lan Jia primary]`
- THK SNR25 ball guideway (Tanaka et al., 2009): `F_c = 10 N`, `F_s = 12 N`, pre-sliding ~tens of um.
  `[internal-AI - verify]`
- Elasto-plastic friction (Hayward et al., 2009, IEEE TCST) fixes Dahl/LuGre static-friction "drift";
  Swevers et al. (2000, IEEE TAC) integrated model with pre-sliding + position/normal-force scaling.
  `[internal-AI - verify]`
- Confirmed passive-LuGre statement to be verified: "LuGre is a proven passive velocity->force operator"
  (necessary+sufficient conditions in the LuGre-dissipativity literature). `[internal-AI - verify]`

---

## E. Anti-drift steering (adaptive control)

### E1. Projection operator (2011)  [online-primary (search-level)]
**Reference.** E. Lavretsky, T.E. Gibson, "Projection Operator in Adaptive Systems", arXiv:1112.4232, 2011.
- Findings (search-level): the projection operator constrains adaptive parameters to a feasible set to
  "prevent parameter drift", giving "zero steady-state bias inside the set" while keeping genuine effects.
  Related: sigma-modification / leakage (Ioannou & Sun); dead-zone (Ioannou & Tsakalis). `[extract -
  verify]`

**Relevance.** The hard-projection version of a drift guardrail steering (as opposed to a soft mean
penalty): keeps friction while forbidding runaway. A candidate mechanism for the steering layer.

---

## F. Physical hidden-state candidates for the gantry (internal AI study)

From `literature/augmentation/Extra dynamische toestand(en) voor grey-box modelaugmentatie van een H-type
dual-drive gantry.pdf` (in-project AI study; primaries to verify):  [internal-AI, secondary]
- Candidate A -- support-structure / foundation mode as an extra 2nd-order state: `xi_dd + 2 zeta_xi
  omega_xi xi_d + omega_xi^2 xi = b(Y) F`, `omega_xi ~ 2 pi * 37.7` rad/s (reported first X resonance).
- Candidate B -- cable/dresspack viscoelastic (Zener) internal state: `z_dot = -(1/tau(Y)) z + k(Y)
  l_dot(q)`, `F_cable = k_0(Y) l(q) + z`.
- Candidate C -- bearing/guideway compliance as a yaw-flex mode (`phi, phi_dot`).
- Candidate D -- actuator/force-path first-order lag.
- "Maximally identifiable" designed state: `z_dot = -(1/tau(Y)) z + k(Y) F_Delta`, `tau_Theta^aug =
  g(Y) z`, positioned on the yaw (`X_1 - X_2`) channel; Y-dependent, memory. `[internal-AI - verify]`

**Relevance.** Physically-motivated templates for what the augmentation state could REPRESENT (our absorber
is the 150 Hz 2nd-order mode). Complementary to the passivity/stability question.

---

## Summary of the gap (what the catalog shows)
Every learned dissipative/passive/stable construction found falls into one of three buckets:
(1) assumes ISS / attractor (returns to equilibrium) -- DiLaR-PINN Prop 3, 2011.13492, 2410.00976,
2512.01984, 2502.02480; (2) contraction / strict stability (`P > 0`, `||A|| < 1`) -- RENs, A4, stable-by-
design; (3) a controller that damps the plant -- NINODE, closed-loop dissipativity synthesis. ~~The only
classical framework native to the free-body / pole-at-origin case (Mabrok 2014) is LTI-only.~~ **[CORRECTED
2026-07-10 -- see §G below: the classical nonlinear NI free-body theory WITH positive-SEMIdefinite storage
EXISTS (Shi-Petersen-Vladimirov 2011.14610, Def 1; Ghallab-Petersen 2201.00144). Mabrok 2014 is LTI-only,
but nonlinear NI free-body is NOT unworked-out at the analytical level.]** A **learned** dissipative
FORWARD augmentation that PRESERVES a marginally-stable (free-integrator) baseline mode is absent from the
surveyed literature. (Negative result -- provisional per repeated-search convergence, not a single pass.)

**[SECOND CORRECTION 2026-07-10 -- see §H below.]** The claim that the marginal / semidefinite-/indefinite-
storage dissipativity notion is "unworked-out" is ALSO wrong: the classical theory exists (cyclo-passivity /
indefinite storage, van der Schaft arXiv:2003.10143; equilibrium-independent passivity, Hines-Arcak-Packard
2011 / arXiv:1709.06986; PH Casimirs; shifted/Krasovskii passivity). **BUT** (held to the §5 four
requirements) none of these bounds POSITION on a free integrator -- cyclo gives "only instability results",
EID bounds shifted I/O not the free coordinate -- so criterion 4 still needs the net-impulse (Route B) or
Negative-Imaginary free-body layer. Net: the marginal-storage relaxation is REUSE (classical); the LEARNED +
forward + LPV realization AND the position-bound coupling is the genuine contribution.

---

## G. INDEPENDENT ADVERSARIAL VERIFICATION (2026-07-10) -- every `[extract-verify]` claim checked against the on-disk PDF

**Provenance of this section.** A fresh, deliberately adversarial verification pass (brief:
`docs/fable-review-brief.md`) opened each on-disk PDF, extracted its full text (PyMuPDF), read the two
flagship papers (DiLaR-PINN, RENs) end-to-end BEFORE comparing to the catalog above, and red-teamed the
central gap claim on-disk and by web search. Verdicts below are **CONFIRMED / CORRECTED / REFUTED** with the
exact primary-source location. **Where this section and the `[extract-verify]` / `[online-*]` tags in §A-§F
above disagree, THIS SECTION WINS** (it is the on-disk primary read; the tags above were automated
extractions). The extracted text lives in the session scratchpad; page numbers are the arXiv-PDF pages.

### G1. DiLaR-PINN (`2604.18277`) -- **CONFIRMED (all sub-claims)**; author list now recorded
- **Authors (NEW -- catalog B1 said "not yet recorded"):** **Youyuan Long, Gokhan Solak, Arash Ajoudani**,
  Human-Robot Interfaces and Interaction Lab, Istituto Italiano di Tecnologia (IIT), Genoa. Accepted to
  IFAC for publication; arXiv:2604.18277v1, 20 Apr 2026. **Use this citation, not a placeholder.**
- **(a) Residual `(S-K)∇V`, S skew, K PSD, `∇Vᵀr ≤ 0` for ALL parameters -- CONFIRMED.**
  **Eq (5), p.3**: `rϕ(x,u) = (Sϕ(x,u) − Kϕ(x,u)) ∇_{xlat}V(x)`, `Sϕᵀ=−Sϕ`, `Kϕ = LϕLϕᵀ ⪰ 0` (Cholesky,
  lower-triangular). **Proposition 1 (Dissipation Guarantee), p.3** proves `∇_{xlat}Vᵀ rϕ ≤ 0` for any
  parameters, by the exact skew-term-is-zero + PSD-term-is-nonneg argument the catalog reproduced.
  **BONUS not in catalog -- Proposition 2 (Coverage of the Dissipative Cone), p.3**: the `(S-K)∇V`
  parameterization is *pointwise expressively complete* -- it spans the ENTIRE dissipative cone
  `C(x) = { r : ∇_{xlat}Vᵀ r ≤ 0 }`. This STRENGTHENS the catalog (the constraint costs no expressivity
  within the dissipative set). **Remark 3, p.4** explicitly identifies `(S-K)∇V` with the port-Hamiltonian
  `[J(x)−R(x)]∇H(x)` form -- i.e. DiLaR-PINN's block IS our §5i PH block, confirmed at the primary source.
- **(b) THE load-bearing claim: stability result requires an ISS baseline, excluding a free integrator --
  CONFIRMED. Proposition 3 (ISS Preservation), p.4.** Verbatim premise: `fphys` admits an ISS-Lyapunov
  function `V∈C¹` with `α₁(‖x‖) ≤ V(x) ≤ α₂(‖x‖)` (6a) and
  `∇V(x)ᵀ fphys(x,u|θ) ≤ −α₃(‖x‖) + σ(‖u‖)` (6b), `αᵢ, σ ∈ class-K∞`. Conclusion: the augmented system is
  ISS under the SAME `V`. **Why this excludes our K=0 X/Y axes:** an ISS-Lyapunov function requires the
  unforced system to be 0-GAS (`u=0 ⟹ x→0`). Our X/Y axes are mass-dampers -- with `u=0`, velocity decays
  but **position settles to an arbitrary constant, not to 0** -- so the system is NOT 0-GAS, there is no
  radially-unbounded `V` with a class-K∞ `α₃` covering the position coordinate, and **premise (6b) cannot
  be satisfied**. Prop 3's guarantee therefore does not apply. **Correct nuance (already right in the
  drift-doc):** Prop 3 is only SUFFICIENT (ISS baseline ⟹ ISS augmented); its failure gives NO guarantee,
  it does NOT prove the augmented system drifts. The "would drift" conclusion is backed by our own §5j
  argument/empirics, not by DiLaR-PINN. Do not overstate this as "DiLaR-PINN proves drift."
- **(c) Residual acts only on latent/unmeasured states -- CONFIRMED, with one nuance. Eq (3), p.3:**
  `f = fphys + [0 ; rϕ]` (observed rows get zero residual). **NUANCE (catalog omits): Remark 1, p.4** states
  this latent-only restriction is "**not mandatory and can be modified**" when the measured-state dynamics
  are themselves imperfectly modeled. So latent-only is a DESIGN CHOICE, not an intrinsic property of the
  method. For us this is favorable: routing the FORCE to the X/Y velocity (latent) rows is compatible with
  "residual on latent state", and the paper itself sanctions extending it if needed.
- **Case study note (CONFIRMED, catalog was right):** their latent state is 1-D, so `S→0` (pure
  dissipation, no skew/storage); **p.5**. Our absorber REQUIRES the skew/storage term, so our case exercises
  the full PH structure that their example does not.

### G2. RENs (`2104.05942`) -- **CONFIRMED (strict `P≻0`, joint passivity, NO marginal variant)**
- **Contraction is w.r.t. a STRICTLY positive-definite metric -- CONFIRMED. Definition 2, p.3:** contracting
  with rate `α ∈ (0,1)` (strict). **Theorem 1(1), p.4:** the contracting-REN LMI requires `P = Pᵀ ≻ 0` and
  `Λ ∈ D+`; incremental Lyapunov `V(Δx)=|Δx|²_P`.
- **Incremental passivity enforced JOINTLY with contraction, not as a marginal alternative -- CONFIRMED.**
  Incremental passivity is the incremental-IQC special case **(Definition 3, p.3-4: `Q=0, R=−2νI, S=I`,
  `ν≥0`).** **Theorem 1(2), p.4** and **Theorem 3, p.6** state the Robust REN is SIMULTANEOUSLY (i)
  well-posed, (ii) contracting with rate `α < ᾱ`, AND (iii) IQC-satisfying -- all under the SAME `P ≻ 0`.
  Passivity is never offered as a stand-alone marginal relaxation.
- **Any `P⪰0` / marginal / integrator-allowing variant? NONE -- CONFIRMED.** The metric is `P ≻ 0`
  everywhere. **Watch-out that could mislead a future reader:** Theorem 1 admits `ᾱ ∈ (0,1]` (the rate
  bound may be 1) -- but the CONCLUSION is `α < ᾱ`, i.e. the model is ALWAYS strictly contracting even when
  `ᾱ=1`. So the `ᾱ=1` allowance is NOT a marginal (`α=1`) case; there is no `α=1` REN. This *reinforces*
  the catalog claim rather than weakening it. Relaxing `P≻0` to `P⪰0` remains an UNpublished route (our
  candidate contribution), not something RENs provides.

### G3. Mabrok 2014 (`1305.1079`) -- **CONFIRMED (free body / poles at origin; LTI-only; attribution)**
- **Authors: M.A. Mabrok, A.G. Kallapur, I.R. Petersen, A. Lanzon.** Attribution CONFIRMED: it is
  **Mabrok et al.**, and the paper itself (p.1) credits the ORIGINAL NI notion to "Lanzon and Petersen
  [1],[2]" -- so the catalog's correction (1305.1079 ≠ Lanzon-Petersen) is right. (Lanzon IS a co-author,
  but the paper is Mabrok-first.)
- **Treats poles at the origin / free body -- CONFIRMED (p.2:** "free body motion which results in poles at
  the origin"). The generalized NI definition (p.3) allows a simple pole on the `jω`-axis / at the origin
  with **residue matrix `K₀ = lim (s−jω₀) s G(s)` positive-semidefinite Hermitian** -- the marginal-
  preserving DC condition our §5j references.
- **LTI / transfer-function only -- CONFIRMED.** All conditions are on `G(jω)` transfer-function matrices
  (`j(G(jω)−G(jω)*) ≥ 0`); "the proofs rely on state space techniques" but the RESULT is purely LTI. **No
  nonlinear and no semidefinite-storage version in THIS paper.** (But see G8 -- the nonlinear version exists
  ELSEWHERE.)

### G4. NINODE (`2504.19497`) -- **CONFIRMED (a controller; re-imposes strict DC-gain)**
- **It is a CONTROLLER, not a forward-model augmentation -- CONFIRMED (abstract):** "NINODE controller ...
  can serve as a controller that asymptotically stabilizes an [NI] plant".
- **Re-imposes a strict DC-gain condition / excludes the marginal case -- CONFIRMED. Assumption 3, p.4:**
  "rules out the case that the cascade of H1 and H2 has the DC gain 1", stated as "the nonlinear counterpart
  of the DC gain condition required in ... the controller H2". So as published it abandons the marginal
  (map-norm-`=1` at DC) case we need to PRESERVE.

### G5. Passive LuGre PINN (`2504.12441`) -- **CONFIRMED substance; provenance of the "quote" CORRECTED**
- **Authors: Asutay Ozmen, João P. Hespanha, Katie Byl.**
- **Nonzero-at-rest friction via a LuGre bristle latent state -- CONFIRMED** (bristle deflection `z`, p.1;
  pre-sliding displacement, p.2; "the bristles ... generate a reactive force, even when the [system is at
  rest]", p.5).
- **Passivity BY CONSTRUCTION vs only-fit -- CONFIRMED it is NOT by construction, but CORRECT the
  provenance:** the words "**passiv...**" and "**dissipat...**" appear **ZERO times** in the paper -- it
  never discusses passivity at all. So the catalog D1 `[extract-verify]` string *"not provably passive by
  construction ... only empirical validation"* is **substantively correct but is NOT a quotation from the
  paper** -- it is an inference. **Do not cite it as a quote.** The paper "enforces physical consistency"
  (abstract) only in the sense of a LuGre-structured friction fit; the passivity guarantee, if wanted, must
  come from the block structure (DiLaR-PINN/PH), exactly as the catalog concludes.

### G6. Dissipative Deep Neural Dynamical Systems (`2011.13492`) -- **CONFIRMED (attractor bucket)**
- Authors: J. Drgoňa, A. Tuor, S. Vasisht, D. Vrabie (PNNL). Provides sufficient conditions for
  dissipativity + **LOCAL ASYMPTOTIC STABILITY to the origin / to attractors confined in a compact set**
  (contraction condition `‖A⋆(x)‖ < 1`; equilibria shifted by bias terms, bounded by Corollary 3). Requires
  return to a bounded equilibrium set → **excludes the free integrator.** CONFIRMED.

### G7. Learning Dissipative Chaotic Dynamics (`2410.00976`) -- **CONFIRMED (attractor bucket)**
- Authors: S. Tang, T. Sapsis, N. Azizan (MIT). Here "dissipative" MEANS **trajectories converge to a
  bounded, positively invariant set** (strange attractor): "every trajectory ... is bounded and converges
  to M(c) asymptotically" (Corollary 1); "assume the trajectories in the training set are already inside
  the attractor". A dissipative projection layer forces `V̇ ≤ 0` toward a learned level set. **Cannot bound a
  free-integrator position** (there is no bounded invariant set for a ramping integrator). CONFIRMED.

### G8. Nonlinear NI free-body consensus (`2011.14610`) -- **CORRECTED: this REFUTES the "nonlinear NI is unworked-out / NI is LTI-only" over-claim**
- Authors: **Kanghong Shi, Ian R. Petersen, Igor G. Vladimirov** (2021). The catalog §C3 dismissed this as
  "a networked-consensus / controller paper; the small-model fetch could not extract a nonlinear free-body
  SEMIdefinite-storage theorem." **That dismissal is WRONG at the primary source.**
- **What it actually contains -- Definition 1, p.2:** a system `ẋ=f(x,u)`, `y=h(x)+Du` is a **nonlinear
  NI system** if there exists a **positive-SEMIdefinite** storage function `V:Rⁿ→R`, `V∈C¹`, with
  `V̇(x) ≤ u(t)ᵀ ẏ̃(t)` (the NI supply rate on `ẏ`), `ỹ=h(x)`. **Definition 2** gives nonlinear **OSNI**
  (output-strictly-NI): `V̇ ≤ uᵀẏ̃ − ε‖ẏ̃‖²`. The paper explicitly notes Def 1 "now includes all linear NI
  systems ... by allowing the storage function ... to be positive SEMIdefinite instead of positive
  definite" and **includes systems with poles at the origin (free body motion).** A single-interconnection
  stability theorem (nonlinear NI ⊗ nonlinear OSNI) is proved.
- **DIRECT QUOTES (transcribed verbatim from the on-disk PDF text layer; page numbers are arXiv-PDF pages;
  math symbols `V̇`, `ẏ̃`, `≤` are as rendered by the text extraction -- re-verify character-exact before
  thesis use).** Full reference: **K. Shi, I.R. Petersen, I.G. Vladimirov, "Output Feedback Consensus for
  Networked Heterogeneous Nonlinear Negative-Imaginary Systems with Free Body Motion", arXiv:2011.14610v2,
  1 Jul 2021.**
  - *Abstract (p.1):* "We extend the deﬁnition of nonlinear NI systems to allow for systems with free body
    motion. A new stability result is developed for the interconnection of a nonlinear NI system and a
    nonlinear output strictly negative-imaginary (OSNI) system."
  - *Introduction / lineage (p.1)* -- establishes that PRIOR nonlinear-NI work excluded free body, and THIS
    paper removes that exclusion: "The deﬁnition has been extended again in [18] to include systems with
    poles at the origin. Systems with free body motion such as single integrators and double integrators
    were included in this new deﬁnition ... The original deﬁnition of NI systems has also been recently
    extended to include nonlinear systems [19] and some stability results were established for nonlinear NI
    systems in [19] and [20]. **However, systems with free body motion are excluded in the nonlinear NI
    deﬁnition in [19].**"
  - *Definition 1 -- nonlinear NI via positive-SEMIdefinite storage (p.2):* "The system (1), (2) is said to
    be a nonlinear negative-imaginary (NI) system if there exists a **positive semideﬁnite storage function**
    V : Rⁿ → R of class C¹ such that V̇(x(t)) ≤ u(t)ᵀ ẏ̃(t), (3) for all t ≥ 0, where ỹ(t) = h(x(t)). (4)"
  - *Remark after Def 1, the marginal-mode point (p.2):* "In contrast to Deﬁnition 3 in [19], which excludes
    linear NI systems with poles at the origin, Deﬁnition 1 now includes all linear NI systems satisfying
    the deﬁnition given in [18] **by allowing the storage function of the system to be positive semideﬁnite
    instead of positive deﬁnite.**"
  - *Definition 2 -- nonlinear OSNI (p.2):* "The system (1), (2) is said to be a nonlinear output strictly
    negative-imaginary (OSNI) system if there exists a positive semideﬁnite storage function V : Rⁿ → R of
    class C¹ and a scalar ε > 0 such that V̇(x(t)) ≤ u(t)ᵀ ẏ̃(t) − ε‖ẏ̃(t)‖², (5) for all t ≥ 0 ..."
  - **How to cite it:** for the nonlinear-NI free-body **semidefinite-storage definition / supply rate**
    (Def 1, Eq 3), NOT as a solution to our problem -- it is analytical (not learned) and a
    controller/consensus construction (not a forward-model augmentation).
- **So the object the drift-doc repeatedly calls "unworked-out" (nonlinear + free-body + semidefinite-
  storage NI) EXISTS analytically** -- here, and in Ghallab-Petersen "NI Theory for Nonlinear Systems: A
  Dissipativity Approach" (arXiv:2201.00144), and the nonlinear-NI quadrotor line (2101.04916, 2603.27560).
- **BUT it does NOT falsify the central gap claim:** `2011.14610` is (i) **analytical, not learned/neural**;
  (ii) a **controller/consensus** construction (OSNI *controllers* driving NI *plants* to consensus), not a
  forward-model augmentation. It sits in bucket (3) "controller" AND fails the "learned" requirement. The
  gap survives; what changes is the *rationale* -- the missing piece is the **learned + forward-augmentation
  (+ LPV)** combination, NOT the underlying nonlinear-NI free-body theory. This is GOOD for the thesis: it
  gives a citable classical foundation (Shi-Petersen-Vladimirov Def 1) on which to build the learned
  version, instead of having to invent the theory from scratch.

### G9. "Search beyond" (web, 2026-07-10) -- no learned falsifier found
- **`2404.12554` "Learning Stable and Passive Neural Differential Equations"** (learned, passive) --
  parameterizes the field as descent directions of a PLNet Lyapunov function whose Hamiltonian is
  **lower- AND upper-bounded by quadratics and positive-definite about a (learnable) equilibrium** → strict
  PD storage → asymptotic/attractor behavior → **damps the free mode.** Not a falsifier (attractor bucket).
- **`2309.16032` "Learning Dissipative Neural Dynamical Systems"** -- baseline neural ODE + minimal
  perturbation to a dissipative attractor. Attractor bucket. Not a falsifier.
- **`2201.00144` nonlinear NI dissipativity approach**, **`2101.04916` / `2603.27560` nonlinear-NI
  quadrotor** -- further confirm nonlinear NI is a developed ANALYTICAL theory (reinforces G8's correction),
  all controllers, none a learned forward augmentation.
- **Net:** no published **learned dissipative FORWARD augmentation preserving a pole at the origin with
  bounded position** was found on disk or by search. **Central gap claim HOLDS** (with the G8 rationale
  correction).

### G10. `§5j` math logic ("passivity bounds velocity not position, O(√T)") -- **CONFIRMED (independent check)**
- Passivity ⟹ `∫F·v ≤ H(0)`; axis damping `c>0` ⟹ energy balance `c∫v² ≤ const + H(0) < ∞` ⟹ **v ∈ L²**.
  Cauchy-Schwarz: `|q(T)−q(0)| = |∫₀ᵀ v| ≤ √T·√(∫₀ᵀ v²) ≤ C√T`. Valid UPPER bound; for a fixed L² signal
  position can still grow unboundedly but sub-linearly (e.g. `v∼t^{−0.6} ∈ L²` gives `q∼T^{0.4}`), so
  "sub-linear but unbounded as T→∞" is correct. **Sound.**
- Momentum balance `m·v(T)+c·q(T) = const + ∫F` ⟹ position bounded **iff net impulse `∫F dt`** bounded,
  which passivity does not force. **Correct.**
- **Mass-damper consistency (brief's second §5j question) -- CONFIRMED consistent.** `F→q = 1/(s(ms+c))` has
  a pole at the origin AND at `−c/m`; `F→v = 1/(ms+c)` is BIBO-stable. Damping bounds velocity while
  position keeps the integrator -- exactly §5j's claim. Crucially `c>0` is what makes "bounded impulse ⟹
  bounded position" valid; a pure double integrator (`c=0`) would not, and §5j correctly flags that the
  `c=0` idealization still needs NI. The mass-damper structure ENABLES the Route-B bounded-impulse argument.

### G11. Items NOT independently re-verified in this pass
- The `[disk]` quotes in §A (Hoekstra EJC 2025, Drenth 2025, Gyorok L4DC 2025, Gyorok-2026 constraint-free)
  were transcribed from earlier on-disk reads and were DE-PRIORITIZED by the brief; not re-checked here.
  They remain `[disk]`-reliability, re-verify character-exact before thesis use.
- `2512.01984` (ECO), `2502.02480` (Stable PH-NN), `2606.11049` (L2-bounded SSM), `2404.07373` (closed-loop
  dissipativity) are NOT on disk; bucket assignments (attractor / controller) are from abstracts, unchanged.
- Lan Jia (2023) gantry friction numbers (§D2): still NOT localizable / unverifiable here.

---

## H. MARGINAL-NATIVE CLASSICAL DISSIPATIVITY THEORY (primary-read 2026-07-10) -- the "semidefinite-storage is unworked" claim was wrong

**Why this section exists.** Prior passes (§B, §5-§5L) kept the "learned/neural" keyword and concluded the
marginal/semidefinite-storage dissipativity notion was "unworked-out". A reframed search (user-prompted:
"there is more theory on the dissipative method") surfaced the CLASSICAL (analytical) theory for marginal /
continuum-equilibrium / free systems, which is mature. Two flagship notions were PRIMARY-READ (PDFs
downloaded + PyMuPDF-extracted); two more are strong leads. **These CORRECT the over-claim (criterion 3 has a
worked theory) but do NOT change the four-requirements verdict (none bounds POSITION -- criterion 4 -- on its
own; = §5j).**

### H1. Cyclo-dissipativity / cyclo-passivity -- INDEFINITE storage  [primary-read]
**Reference.** A.J. van der Schaft, "Cyclo-dissipativity revisited", *IEEE Trans. Automatic Control* 66(6),
2021. arXiv:2003.10143. (Notion coined by Willems; developed by Hill & Moylan, "Cyclo-dissipativeness,
dissipativeness and losslessness for nonlinear dynamical systems".)
- **The relaxation (verbatim):** (p.6 footnote) *"Note that we do not yet require S to be nonnegative or
  bounded from below."* **Definition 3.1:** Σ is cyclo-dissipative if `∮ s(u(t),y(t)) dt ≥ 0` for all `T ≥ 0`
  and all `u` with `x(T)=x(0)`; cyclo-lossless if equality. So the storage function is allowed to be
  **INDEFINITE** -- exactly the "relax `P≻0`" object the drift-doc called missing.
- **The decisive caveat (verbatim, Remark 3.4, p.8):** *"the Lyapunov function obtained for the interconnected
  system by summing the storage functions ... is no longer nonnegative. Hence in principle only **instability**
  results can be inferred; this is the motivation for cyclo-dissipativity."* → cyclo-passivity gives LESS than
  passivity on boundedness; it does NOT bound position (or even give a Lyapunov stability conclusion).
- **Relevance.** Confirms the indefinite-/semidefinite-storage marginal theory EXISTS (corrects the
  over-claim). Does NOT satisfy criterion 4. It is the correct THEORY LANGUAGE for the marginal storage
  relaxation (§5e proofs), not a boundedness mechanism.

### H2. Equilibrium-Independent Passivity / Dissipativity (EIP/EID) -- continuum of equilibria  [primary-read]
**Reference.** G.H. Hines, M. Arcak, A.K. Packard, "Equilibrium-independent passivity: A new definition and
numerical certification", *Automatica* 47:1949-1956 (2011). Detailed quadratic-supply treatment:
J.W. Simpson-Porco, "Equilibrium-Independent Dissipativity with Quadratic Supply Rates", IEEE TAC 2019,
arXiv:1709.06986 (primary-read).
- **Definition 3.2 (verbatim, arXiv:1709.06986 p.3-4):** the system is EID with supply rate `w` if *"for
  every equilibrium x̄ ∈ EΣ, there exists a continuously-differentiable storage function V_x̄ : X → R≥0 such
  that V_x̄(x̄)=0"* and `d/dt V_x̄ ≤ w(u−ū, y−ȳ)`. The assignable-equilibria set `EΣ = X` when `m=n` -- i.e.
  **EVERY state is an equilibrium** = the free-integrator / continuum-of-equilibria case.
- **What it gives / does not:** EID characterizes passivity w.r.t. ANY equilibrium (criterion 3), with
  stability results resting on an INCREMENTAL stability condition. A mass-damper IS EID (velocity-passive
  around any position) yet **position still integrates** → EID does NOT bound the free coordinate (criterion 4).
- **Relevance.** The natural characterization of our K=0 axes (continuum of position-equilibria). Reuse for
  the criterion-3 framing; not a position-bound.

### H3. Port-Hamiltonian Casimir functions -- flat storage direction = the free coordinate  [primary-read]
**Reference.** van der Schaft PH theory (Casimir = quantity conserved by the Dirac structure, independent of
H); Energy-Casimir method historically for rigid-body non-zero equilibria. Learned version PRIMARY-READ:
**L. Xu, M. Zakwan, G. Ferrari-Trecate, "Neural Energy Casimir Control for Port-Hamiltonian Systems",
arXiv:2112.03339.** Also PHAST arXiv:2602.17998 (2026, lead).
- **PRIMARY-READ verdict:** the Casimir CONCEPT (a quantity conserved by the Dirac structure, independent of
  H → a flat storage direction) is criterion-3-relevant. BUT its DEPLOYMENT in energy-Casimir CONTROL damps
  the mode: verbatim (abstract/intro) the method "asymptotically stabilize[s] port-Hamiltonian systems at a
  desired equilibrium" and "an additional damping term is employed to asymptotically stabilize the
  closed-loop system." So the learned Casimir work (i) is a CONTROLLER (not a forward augmentation) and
  (ii) ADDS DAMPING to drive a desired equilibrium → **fails criterion 3 (marginal-preserving) in
  deployment.** The concept is reusable for stating "flat storage direction"; the control method is not our
  case. Does NOT bound the free coordinate as a forward model (criterion 4).

### H4. Shifted / Krasovskii / differential passivity  [primary-read]
**Reference.** Y. Kawano, K.C. Kosaraju, J.M.A. Scherpen, "Krasovskii and Shifted Passivity Based Control",
IEEE TAC 66(2):666-672 (2021), arXiv:1907.07420.
- **PRIMARY-READ verdict:** a passivity-based CONTROL paper. Establishes four passivity concepts
  (differential, Krasovskii, incremental, shifted) and their relations ("Krasovskii passivity implies
  shifted passivity"; "differential passivity with respect to a constant metric implies incremental
  passivity"). Shifted passivity handles "a system whose equilibrium point is not necessarily the origin"
  (continuum/nonzero-equilibrium toolkit, same crit-3 family as EID). It develops dynamic CONTROLLERS
  (illustrated on a DC-Zeta converter) -- NOT a forward-model position bound. Same criterion-3-yes /
  criterion-4-no status; controller-oriented. Confirms the lead; verdict unchanged.

### H-summary
The marginal-native dissipativity theory (indefinite storage, continuum equilibria, flat Casimir) is MATURE
and CITABLE -- the drift-doc's "semidefinite-storage marginal dissipativity is unworked" was an artifact of
the neural keyword. **But held to the four requirements, ALL of H1-H4 characterize/permit the marginal mode
(criterion 3) WITHOUT bounding position (criterion 4).** Position-boundedness still needs the net-impulse
(Route B) or Negative-Imaginary free-body (force→position) layer on top. Net: the storage relaxation is
REUSE; the learned/forward/LPV realization + the criterion-4 coupling is the contribution.
