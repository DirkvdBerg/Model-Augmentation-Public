# Source audit and proposed mathematical structure

> **Scope, set during the 2026-09-06 consolidation of `gaps.md`. Nothing here was deleted.**
>
> This file is a **supporting appendix**, not a second live theory document.
>
> - **Authoritative here:** Sections 1, 2 and 7 (what the local sources actually supply, the
>   targeted external checks, and the novelty boundary).
> - **Superseded:** Sections 3, 4 and 5 restate theory that [`gaps.md` Section 9](gaps.md) now
>   owns. They are kept for their derivation of how the target was arrived at. Where the two
>   differ, `gaps.md` Section 9 wins.
> - Every claim in this file needs a row in `gaps.md` Section 10 before it may be quoted.

**Supporting appendix.** The live derivation and subsequent decisions are maintained in
[gaps.md, Section 9](gaps.md#9-finite-horizon-separation-with-learned-states-and-an-encoder).
The proposed formulations below record the audit-stage recommendation; they are not a second
authoritative status record.

Date: 2026-09-06. Purpose: choose a defensible theorem target before implementing a new penalty.

Status: literature audit and proposed formulation, not a completed gantry theorem, novelty certification, or experimental demonstration. The local PDFs were inspected directly, including the relevant statements and proofs; the project reference summaries were treated as leads. The checkpoint measurements and existing symbolic scripts were not rerun. No training code or existing research notes were changed.

Under [RULES.md](RULES.md), the proposed identities are **structural**, their evaluation on fitted models and measured records is **data-computable**, and comparisons with true physical parameters are **truth-requiring validation**. The ten identifiable combinations are inherited project context, not independently reproved here. No new structural proof about the FP model is claimed; a later such proof must follow the dual-engine rule.

The recommendation is to start with a **finite-horizon, weighted output-sensitivity formulation**, explicitly including the freedom in the learned component and initial-state estimator. Use local least-squares geometry to define separation; use variational dynamics to compute the relevant matrices. Make physical recovery a separate, conditional statement. An H2 specialization can follow if its assumptions are met.

## 1. What the six local papers actually supply

Page references below are PDF page numbers. Several local copies are arXiv versions; theorem numbering must be checked again against the final publication before submission.

### Györök et al., L4DC 2025

[Local PDF](../../../../literature/Orthogonality/Hoekstra%20-%20Orthogonal%20projection-based%20regularization%20for%20efficient%20model.pdf), Sections 2–4, Eqs. (3), (4), (9)–(19).

- The learned correction is a feedforward map of the physical state and input. Training uses a truncated multistep prediction loss, while regularization acts on stacked evaluations of the one-step correction.
- The linear-in-parameters construction protects the image of the stacked physical regressor. The nonlinear extension protects the image of `[Phi, Gamma]`, where `Gamma = F(theta_bar) - Phi theta_bar`, not just the parameter tangent `Phi`.
- Updating estimated states or the parameter linearization point is discussed. A frozen projector is an implementation option, not an assumption required by the definition of orthogonality.
- The paper derives the regularizer and reports identification results. It does not establish a general exact parameter-recovery theorem or rollout-orthogonality guarantee.

**Transfer:** the stacked geometry and the penalty factorization. **Limit:** neither the learned internal-state route nor a proof that the regularizer decouples the actual multistep fitting problem is supplied.

The zero-augmented-state evaluation slice is a choice in this project's adaptation. Györök's original architecture has no such augmented states.

### Kon et al., ACC 2022

[Local arXiv v1](../../../../literature/Orthogonality/kon2022_pgnn-feedforward-orthogonal-projection_ACC_arXiv2201.03308.pdf), Section IV, PDF pp. 4–6.

- Assumption 12 requires full column rank of the lifted model regressor for each reference. The fitting problem is inverse/feedforward regression on a supplied reference and required input.
- Theorem 17 decomposes the finite-penalty objective into model-space fit, complementary-space fit, and overlap penalty. Its model-space term still depends on both physical and neural parameters.
- Eqs. (32)–(35) introduce a hard constraint and projected parametrization. Theorem 20 gives the resulting disjoint optimization. Remark 21 identifies the physical fit with a best linear approximation in its stated setting.
- The discussion following Theorem 20 explicitly allows true unknown dynamics to overlap the physical-model space. The hard projection can then restrict additional performance.
- The trajectory projection requires the reference in advance; this matters when transferring it to a causal state-space predictor.

**Transfer:** the cleanest algebraic precedent for distinguishing a soft penalty from exact separation. **Limit:** Theorem 20 cannot simply be cited for a finite-beta stateful training run.

### Kon et al., IFAC World Congress 2023

[Local arXiv v2](../../../../literature/Orthogonality/kon2023_pgnn-feedforward-trajectory-orthogonal_IFAC-WC_arXiv2303.07994.pdf), Sections 3.3 and 4, PDF pp. 3–4, Eqs. (7)–(23).

- The physical parameters are divided into linear and nonlinear subsets. For fixed nonlinear parameters, the finite-time basis matrix is linear in the remaining parameters.
- Eqs. (16)–(22) already stack a trajectory, construct an SVD basis, and penalize the neural contribution in that space. The SVD explicitly uses `r = rank(X)`, so the projection itself does not require full column rank. Unique physical coordinates remain a separate issue.
- The projector is frozen at an initial physical-model fit. The claim that recursive updating makes no practical difference is an empirical observation in this example.
- The FNN acts on prescribed features, but the preprocessing includes a relay retaining its previous value when velocity is zero, Eq. (10). Consequently, describing the whole construction as having no memory is too broad. It has no *jointly learned internal-state feedback of the present kind*.

**Transfer:** stacked separation geometry and an example with a moving physical basis approximated by a frozen one. **Limit:** this is inverse regression on fixed measured/reference features, not forward rollout sensitivity through a learned latent state.

### Van Haren et al., ECC 2024

[Local arXiv v1](../../../../literature/Orthogonality/vanharen2024_finite-time-ILC-basis-functions_ECC_arXiv2403.02039.pdf), Eqs. (8), (18)–(20), Remark 2, PDF pp. 3–4.

- The feedforward trajectory is deliberately overparameterized as `f = psi theta + f_free`.
- Remark 2 proposes penalizing the free component in the image of `psi` using an SVD, alongside a weighted finite-time ILC objective.
- The free component is a learned signal updated between trials, not a static neural map. Calling every construction in this group a memoryless learned map misdescribes this example.

**Transfer:** separation of a low-dimensional physical trajectory component from a flexible trajectory component is established practice. **Limit:** an unrestricted finite vector does not provide a causal recurrent realization for unseen inputs, or a physical-parameter recovery theorem for the gantry.

### Györök et al., orthogonal-by-construction 2026

[Local arXiv v3, dated 9 January 2026](../../../../literature/Orthogonality/gyorok2026_orthogonal-by-construction-augmentation_IFAC-JSC_arXiv2511.01321.pdf), Sections 2–4, PDF pp. 3–6; conclusions pp. 9–10.

- Baseline: linear-in-parameters NARX regression, with lagged measured IO values. Assumption 1 is full regressor rank.
- Lemma 3 establishes exact **training-set** orthogonality of the projected neural component. At deployment, Eq. (13) uses the learned fixed auxiliary coefficient; orthogonality is not thereby guaranteed under every new data distribution.
- Theorem 7 requires full rank, exact noise-free recovery of the training relations (Assumption 6), and **orthogonality of the actual unmodelled dynamics** to the regressor (Condition 4).
- Without Condition 4, Eq. (21) gives a unique but generally nonzero physical-parameter error. This is the decisive distinction between a preferred decomposition and the physical truth.
- Theorem 16 adds statistical assumptions including stability, excitation, realizability, and asymptotic minimization. It is not a guarantee that Adam finds the true parameters.
- Theorem 18 concerns asymptotic cross-covariance. Its proof differentiates the projected correction with respect to neural parameters. This is the relevant conceptual bridge from correction values to permissible variations.

**Care when transferring Theorem 18:** stacked orthogonality yields a zero *sum* of cross-products; it does not generally yield zero cross-products at every sample, as the passage leading to Eq. (40) suggests. Unweighted orthogonality also does not automatically survive unequal output-channel noise weighting. For our MIMO problem, derive the aggregate weighted statement explicitly and distinguish Gauss–Newton curvature from the exact nonlinear Hessian. Do not use the theorem as a plug-in covariance guarantee.

**Transfer:** conditional recovery and differentiation of an orthogonality identity. **Limit:** measured-regressor geometry, linear parameter dependence, and the statistical assumptions must be replaced or justified. The conclusion explicitly identifies partially observed state-space augmentation as future work.

### Hoekstra et al., LFR augmentation 2026

[Local preprint, 20 February 2026](../../../../literature/closed-loop-id/hoekstra2026_lfr-augmentation-fp-models.pdf), Table 1; Section 5, PDF pp. 8–10; Section 6.3, p. 11.

- Table 1 includes dynamic parallel augmentation with separate learned states.
- Eq. (22) uses multistep output prediction with an encoder estimating initial states. The joint parameter vector includes baseline, augmentation, interconnection, and encoder parameters.
- Section 5.2 identifies nonunique decompositions and uses a penalty on departure from nominal physical parameters, Eqs. (26)–(27).
- Section 5.5 discusses consistency as convergence to an equivalent system representation. That is not uniqueness or truth of the physical part of that representation.
- Section 6.3 reports physical estimates staying near initialization even for approximate initializations, while dynamic augmentations fit accurately.

**Transfer:** the architecture, objective, and directly relevant motivation. **Limit:** the observation does not identify the zero-slice mechanism as its cause or validate our proposed regularizer.

## 2. Targeted external checks after the local audit

These were checks of specific theoretical connections, not an exhaustive novelty search.

**Plumlee (2017), Bayesian Calibration of Inexact Computer Models.** Sections 2.1 and 3.3, Theorems 1–2, define a calibration target through a loss and derive discrepancy–model-gradient orthogonality at an interior optimum, including a Hilbert-space generalization. The target is explicitly dependent on model and loss, not automatically a physical constant. This is direct prior art for the general sensitivity–residual orthogonality principle. Only these relevant sections were audited. [Primary paper](https://www.asc.ohio-state.edu/statistics/comp_exp/jour.club/Bayesian_calibration_of_inexact_computer_models_Plumlee_2017.pdf).

**Méndez-Blanco et al. (2021), Local parameter identifiability of large-scale nonlinear models.** Section 3, Eqs. (12)–(14), treats trajectory linearization and forms an identifiability matrix through the Schur complement of an initial-state/parameter observability matrix. This supplies a relevant precedent for accounting for initial-state freedom. The weighted neural-parameter extension below is our proposed application, not a theorem quoted from that paper. [Primary paper](https://pure.tue.nl/ws/portalfiles/portal/190347509/1_s2.0_S2405896321010508_main.pdf).

**Donati et al. (2025), Combining off-white and sparse black models in multi-step physics-based systems identification.** Problem 1 jointly fits physical parameters, an initial state, and a black component. Section 3, Assumption 1 and Theorem 1, derives a local error bound under positive-definite physical fitting curvature and recovery assumptions; its analysis simplifies to known initial state. It does not establish that our unconstrained joint network has the needed curvature. Its useful role is a perturbation-bound precedent, not a proof that regularization creates identifiability. [Published primary paper](https://iris.polito.it/retrieve/b71ae976-e5d5-43d6-bb53-7716a07a0c03/1-s2.0-S0005109825003036-main.pdf).

## 3. Proposed theorem target: first-order physical distinguishability

This is a proposed elementary finite-dimensional formulation. It is not claimed as novel or as a completed nonlinear theorem.

Fix input records, window definitions, an interior reference model, and a positive-definite fitting weight `W`. Let `p(kappa, eta, xi)` stack **all output predictions entering the loss**, where:

- `kappa` denotes independent physical coordinates (the project's ten combinations, subject to an output-excitation check);
- `eta` denotes learned augmentation parameters and any other learned interconnection parameters;
- `xi` denotes the actual initial-state degrees of freedom, or encoder parameters when the encoder is trained.

At the reference model write

`delta p = S delta kappa + R delta eta + E delta xi + higher-order terms`.

Here `S`, `R`, and `E` are derivatives of the **full predictor**, including its recurrence. They are not merely instantaneous network outputs. No truth model is needed to evaluate them.

Whiten by `L` with `L^T L = W`. Remove the initial-state/encoder tangent by defining

`M_E = I - (L E)(L E)^dagger`,

`S_bar = M_E L S`,  `R_bar = M_E L R`.

Proposed first-order criterion:

> No nonzero physical perturbation can be cancelled by learned and initial-state perturbations if and only if
>
> `rank([L E, L R, L S]) - rank([L E, L R]) = dim(kappa)`.

Equivalently, `S_bar` has full column rank and its image intersects the image of `R_bar` only at zero. The profiled least-squares curvature is

`H_eff = S_bar^T (I - R_bar R_bar^dagger) S_bar`.

Positive definiteness of this matrix is the same finite-dimensional criterion. Full rank of the neural-parameter Jacobian is unnecessary; pseudoinverses accommodate neural redundancies. With restricted permissible parameter directions, replace the free tangent spaces by the corresponding admissible tangent description.

The stronger condition

`S_bar^T R_bar = 0`

is sufficient to prevent additional *first-order* information loss from the learned tangent, relative to the physical model with initial states already profiled out. It is not necessary: nonorthogonal subspaces can still have a trivial intersection. This distinction rules out an unrestricted claim that physical identifiability holds **if and only if** the subspaces are orthogonal.

This concerns a linearized prediction map on a fixed experiment. Passing to nonlinear local identifiability needs regularity and neighborhood assumptions. A singular derivative at an isolated point alone does not prove nonlinear non-identifiability. With a nonzero residual, the exact fitting Hessian also contains residual-weighted second derivatives; the displayed matrix is Gauss–Newton curvature.

## 4. A different proposition: displacement from a fixed contribution

For the frozen affine fitting problem, with initial-state effects profiled out, compare the physical least-squares fit with and without an additive fixed output contribution `d`. If `S_bar` is full column rank and `d_bar = M_E L d`, the change is

`delta kappa_with - delta kappa_without = -S_bar^dagger d_bar`.

It is zero exactly when `S_bar^T d_bar = 0`. Its norm is bounded by

`||P_Sbar d_bar|| / sigma_min(S_bar)`.

This is the appropriate setting for an exact zero-displacement/orthogonality equivalence. The reference estimate need not equal the physical truth. The formula is a **structural** least-squares relation; its matrices and fitted contribution can be **data-computable**. The discrepancy at true physical parameters is **truth-requiring** unless additional information identifies it.

A value penalty at a single trained network does not establish tangent separation. For example, the family `d(eta) = eta s` has zero contribution at `eta = 0`, but derivative `s`, which can be a physical sensitivity direction. Conversely, an identity `S_bar^T d(eta) = 0` for every admissible nearby `eta`, with `S_bar` fixed, differentiates to `S_bar^T D_eta d = 0`. Finite soft regularization does not automatically provide such an identity.

## 5. Where the dynamic architecture enters

For the illustrative dynamic parallel model

`x_next = f_kappa(x,u) + a_eta(x,z,u)`,

`z_next = g_eta(x,z,u)`,  `y = h_kappa(x,u)`,

the full trajectory state Jacobian is

`J_k = [[f_x + a_x, a_z], [g_x, g_z]]`.

The physical and learned sensitivities propagate through this full block matrix, with their own parameter forcing terms and initial-state derivatives. The affine hidden-state coupling `K` is the special case of `a_z`. This connects the theorem target to the existing blind-spot calculation.

There are two possible protected objects that must not be conflated:

1. A standalone baseline sensitivity: preserves a chosen baseline decomposition.
2. The physical-parameter sensitivity of the full augmented predictor: describes substitution in actual joint training.

These differ because changing physical parameters changes states subsequently read by the augmentation. Their agreement must be derived under explicit assumptions if we want to substitute the first for the second.

Since the scheduling variable is a state, its effect belongs in the **first-order chain rule** when computing `J_k` and the parameter sensitivities. Changes of the sensitivity matrix under finite model changes then enter higher-order analysis. Freezing a scheduler while differentiating omits a first-order path; it cannot be dismissed as only a second-order coupling.

The measured `288 x 14` matrix stacks window endpoints in physical-state space. It is not yet the full stack of output samples in the training loss. Its rank does not establish the rank of `S_bar` after accounting for actual observations and encoder freedom.

## 6. What remains to decide and demonstrate

1. **Choose the claim:** fixed-contribution displacement, first-order physical distinguishability, or both as separate propositions. Neither alone proves true physical recovery.
2. **Choose the reference:** full augmented predictor versus standalone baseline, which parameters are held fixed, and how the encoder is treated. Profile actual encoder directions rather than assuming independently free initial states unless that stronger model is intentional.
3. **Match the metric and samples:** weight and stack the outputs used in training. Overlapping windows and colored residuals matter for later statistical covariance claims, even though the deterministic least-squares geometry remains well defined.
4. **Connect an implementable penalty to the chosen condition:** determine whether the proposed rollout-contribution penalty controls only values or also admissible tangent overlap. Do not assume the latter.
5. **Treat hard and soft enforcement as open alternatives:** a window projection is algebraically possible; a consistent causal recurrent realization across overlapping and unseen windows requires a construction. Failure of a pointwise `K` projection does not prove all hard approaches impossible. Calling beta a dual variable does not remove tuning: a squared equality penalty is not itself a Lagrange multiplier, and a tolerance-constrained interpretation still requires a tolerance and a dual algorithm.
6. **Provide a nonlinear remainder statement:** smoothness, bounded derivatives, a fixed finite horizon and a valid reference neighborhood are needed. A finite ablation difference is not automatically the first-order contribution in the proposition.
7. **Demonstrate the actual method:** joint-estimation on/off comparisons remain necessary. Geometric alignment in a frozen-physical-parameter checkpoint cannot demonstrate improved physical estimates.

## 7. Role of H2 and current novelty boundary

H2 is a possible stable-LTI corollary after the signal-space result. Specify a common exogenous input, output weighting, stable transfer realizations, initialization, and excitation statistics before equating a time-domain fitting geometry with an H2 inner product. A realized finite trajectory is not itself a transfer matrix. General colored excitation changes the weighting. A Stein equation can evaluate a cross-Gramian, but the existing finite mixed propagation kernel must be related to that equation with dimensions and contractions made explicit; it is not automatically its solution.

Removing the learned states recovers a static augmentation architecture, but a static correction still propagates through the physical dynamics. Therefore it does not by itself make a rollout or H2 condition equal to Györök's one-step-map penalty. Any claimed memoryless special-case equivalence must also specify the prediction operator and inner product.

There is already direct prior art for sensitivity–discrepancy orthogonality, stacked physical/free separation, and elimination of initial-state nuisance directions. This audit therefore does **not** support novelty of those ideas in isolation. What remains a candidate contribution is an architecture-aware, data-computable enforcement and analysis of physical/learned separation for the partially observed dynamic LPV-LFR predictor.

The defensible next proof task is the finite-horizon formulation in Sections 3–5, with clearly stated first-order scope, followed by a check that the proposed algorithm actually enforces or bounds its relevant overlap. An exhaustive H2 novelty search, the unavailable parity/adaptive-control papers, and any global recovery claim remain outside this audit.
