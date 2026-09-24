# Agent A: E1, E2, E4 (experiment design for Y-scheduled LPV grey-box augmentation)

Frame: `FRAME.md` (not re-derived). Skill: `.claude/skills/deep-research/SKILL.md`. Date 2026-09-24.
TU/e browser route: UNAVAILABLE (layer 1, no browser bridge in this session). Items that need it are
marked `needs-browser-route`. dblp was UNREACHABLE this run (bot wall, see Research Log), so CDC/ECC/ACC
2023-2026 are unreached and every negative claim below is provisional against them.

Notation used in the "compute" column: `J` = baseline output-sensitivity matrix over the training
tuples (the one with `cond(J) = 766`), `Delta*` = the true unmodelled contribution, `P` = projector onto
`span(J)`.

## A. Tables

### E1. Informative data / PE for the physical parameters, closed loop, known controller

| condition or method | formula as read | paper | where | applies to closed-loop LPV grey-box augmentation? | quantity we could compute |
|-|-|-|-|-|-|
| Closed-loop positive definiteness of the information matrix | `I(theta) > 0` iff `N_Vecl = {0}` or, for each nonzero `alpha` in the left kernel of `V_ecl`, `E[alpha^T grad_theta W_y(q,theta) r(t)]^2 = E[alpha^T K(q) grad_theta W_u(q,theta) r(t)]^2 != 0` | Gevers, Bazanella, Bombois, Miskovic, IEEE TAC 2009 (held) | Theorem 4.3, Eq. (29), p6 | per frozen Y only: LTI, PEM; the gradient may be w.r.t. any (physical) parametrisation, so the rank test transfers to our 10 combinations per record | rank and `lambda_min` of the Gauss-Newton Gramian `J^T J` built only from the `r`- and `f_ms`-driven parts, per record (noiseless data: the noise branch `V_ecl` contributes nothing) |
| Transfer of excitation | "`Ker{psi(t)} = Ker{V(q)}` for almost all `u(t)` in `U_n` if and only if `n >= rho_V`" | same | Theorem 5.1, p7 | per frozen Y only | number of excited lines per channel vs rank of the sensitivity filter; note: "almost all" is GENERIC rank, it says nothing about conditioning (our `cond(J) = 766` is a conditioning problem, not a rank problem) |
| Controller complexity vs reference richness | (i) `r = 0`: `I(theta) > 0` at all identifiable `theta` iff `max(nx - na, ny - nb) >= 0` (47); (ii) otherwise, for almost all `r` in `U_k` iff `k >= min(na - nx, nb - ny)` (48) | same | Theorem 6.3, p9 | no for (i): it relies on noise excitation, absent in noiseless simulation; (ii) per frozen Y, SISO ARMAX only | none directly; states that without noise all information must come from `r` and `f_ms` |
| Not all references need excitation (MIMO) | Conclusion: "it is not necessary to excite both reference signals to attain a given accuracy level" (2x2 example) | Bazanella, Gevers, Miskovic, EJC 2010 (held) | Sect. 5 Conclusions, p18 | per frozen Y only | rank of `J` with one logical channel's multisine removed (tests whether the three realisations are redundant) |
| MIMO closed-loop informativity check | Condition "depends on the controller complexity, the external excitation parametrization and the complexity" of the model structure (condition itself NOT extracted) | Colin, Bombois, Bako, Morelli, Automatica 121:109171, 2020 (held, HAL) | Conclusion p12 only | per frozen Y only; classical structures, multisine or filtered white noise | not stated here; read Sect. 5 to 8 before use |
| Closed-loop data informativity (lecture form) | `Phi_z(omega) > 0` holds when: a PE `r`; or `v` through a controller of sufficiently high order; or "A time-varying or nonlinear controller"; direct method consistent "If S in M, r is PE of sufficiently high order and there are no algebraic loops" | 5SMB0 Lecture 11 (TU/e, local) | slides 19, 23 | partly: `S in M` is violated by the hidden absorber, so consistency fails regardless of informativity; that gap is exactly what Condition 4 must cover. Our per-record controller is LTI, so the "time-varying controller" route is unused within a record | `Phi_z` over the excited band per record |
| PE definition | "`u` is persistently exciting of order `n` if `R_u^n` ... is non-singular"; nonzero spectrum at `n` points gives PE of order `n` | 5SMB0 Lecture 6 | slides 17, 19 | per frozen Y only, open loop | count of excited bins per channel (trivially large for our multisine) |
| Open-loop sufficient richness (Ljung) | "an excitation that is SR of order nb + na is sufficient for an experiment to be informative both for ARMAX and Box-Jenkins" | Ljung 1999 Thm 13.1, second-hand via Gevers et al. 2009 p8 (book not held) | Thm 13.1 as cited, not read | no (open loop) | none |
| Exciting trajectories for base parameters (robotics grey-box) | "Using the base parameters and tracking 'exciting' reference trajectories [25], we get a full rank and well conditioned matrix W"; LS `chi_hat = W^+ Y` (15) | Gautier, Janot, Vandanjon, IEEE TCST 2013 (held), citing Gautier and Khalil, IJRR 1992 | p6, Eqs. (13) to (15) | yes, closest in kind: known controller, closed loop, physical parameters linear in an inverse-dynamics regressor; our 10 identifiable combinations play the role of base parameters | `cond(W)` of the inverse-dynamics regressor built from measured `y, ydot, yddot` per logical channel and per Y; compare with `cond(J) = 766` |
| Optimal robot excitation (periodic, condition-number criterion) | metadata only | Swevers et al., IEEE TRA 13(5):730-740, 1997; Gautier and Khalil, IJRR 11(4):362-375, 1992 | not read | yes in kind | as above |
| Collinearity index (practical identifiability) | formula text OCR-garbled; interpretation read: a parameter shift "can be compensated ... up to a fraction of 1 divided by the collinearity index"; "Critical values for gamma lie in the range of 5-20"; "approximatively 20 is already critical" in their case; defined on column-normalised `S` (Eq. 12) via the smallest eigenvalue of `S_K^T S_K` (Eq. 13, reconstructed as `gamma_K = 1/sqrt(lambda_K)`, re-check before quoting) | Brun, Reichert, Kuensch, Water Resour. Res. 2001 (held) | Sect. 4.2, Eqs. (12) to (14), pp5-6, 13 | yes: model-agnostic, needs only `J`; the 5 to 20 band is their field's heuristic, not a noise-floor threshold | `gamma_K` for all subsets of the 10 combinations; for `{cg1, cg2}` with 97.7 % correlation, `gamma ~ 1/sqrt(1 - 0.977) ~ 6.6` (our arithmetic under the reconstructed formula) |
| Profile likelihood, practical non-identifiability | `chi^2_PL(theta_i) = min_{theta_j != i} chi^2(theta)` (10); Definition 1: practically non-identifiable if the likelihood-based confidence region is infinitely extended although a unique minimum exists | Raue et al., Bioinformatics 2009 (held) | Eq. (10), Def. 1, pp3-4 | partly: the threshold `Delta_alpha` is noise-based; in noiseless simulation it must be replaced by a data-derived floor (CLAUDE.md item 8) | profile of the training loss along `m_diff` (the 98.8 % bias direction) and along `cg1 - cg2` |

**Synthesis E1.** All closed-loop informativity theorems read are LTI and PEM, so they apply per frozen Y
only, and they certify generic RANK, never conditioning. With noiseless data the controller-complexity
route (Thm 6.3 (i)) is closed, so all information must come from `r` and `f_ms`. The rank conditions are
almost surely met by a many-line multisine; our measured problem (`cond(J) = 766`, `cg1/cg2` 97.7 %
correlated) is practical identifiability, which the robotics "exciting trajectory" line (condition number
of the base-parameter regressor) and the Brun collinearity index address directly. Informativity does not
rescue consistency when `S not in M` (hidden absorber); that is where E4 takes over.

### E2. Excitation for learned nonlinear state-space models (SUBNET, neural SS)

| condition or method | formula as read | paper | where | applies? | quantity we could compute |
|-|-|-|-|-|-|
| Weak PE for the augmented model | for all `theta_1, theta_2` with `V(.)(theta_1) != V(.)(theta_2)`: `V_DN(theta_1) != V_DN(theta_2)` w.p.1 (30); "PE requirements must be assessed at the level of the nonlinear regressor"; "this condition depends not only on the input signal, but also on the resulting output trajectories" | Gyorok, Schoukens, Peni, Toth, IFAC J. Syst. Control 35:100376, 2026, arXiv:2511.01321 (held) | Condition 14, Eq. (30), p5 | partly: stated for IO linear-in-parameters baseline plus ANN; a distinguishability definition, not a checkable test | none directly; operationalise as rank of `[J, dF_ANN/dtheta_a]` on the reference set |
| Weak PE for SUBNET | same form: `V^enc_DN(theta_1, eta_1) != V^enc_DN(theta_2, eta_2)` w.p.1 (23) | Beintema, Schoukens, Toth, Automatica 156:111210, 2023 (held) | Condition 6, Eq. (23), p7 | partly: grep for "feedback", "closed-loop", "open loop" returns no treatment of feedback in the paper (grep-negative, medium) | as above |
| Incremental exponential output stability of the data-generating system | `E_e[||y_k - y~_k||_2^4] < C(delta) lambda^(k - k_o)` (15) | same | Condition 1, p7 | yes: our closed loop is stable per record; a required property, not an excitation design | empirical decay of output difference under perturbed initial state in the simulator |
| Random-feature output-layer PE | "the data are persistently exciting for the output layer of sub-model L, i.e., `M_L` is non-singular", `M_L := (1/N) Phi_L^T Phi_L`; strict decrease "iff `(1/N) sum_i phi(x_i) r_i != 0`" | Hassaballa, Lazar, arXiv:2606.04290, 2026 (held) | Assumption 1, Lemma 1, Eqs. (10) to (12), p3 | partly: static / NARX one-step regression, no encoder, no closed loop | `lambda_min((1/N) Phi^T Phi)` of the last hidden layer of our ANN on training tuples |
| Space-filling of the joint input-state region | filling distance `rho(D_N) = max_{varsigma in Z~} d(varsigma, D_N)` (8); cost `W = (1/M) sum_i Var(f_hat | z~_i, D_N)` (13); Theorem 1: `epsilon`-space filling with `N >= M (T_d + 1)` under controllability (Condition 1) and a freely specifiable input (Condition 4) | Liu, Kiss, Toth, M. Schoukens, IEEE L-CSS 2025, arXiv:2502.17042 (held) | Def. 1, Def. 2, Eq. (13), Theorem 1, pp2-4 | partly: open loop, known model; region of interest is user-defined; closed-loop input cannot be "freely specified", so Condition 4 of that paper fails for `u` but may hold for `(r, f_ms)` | `rho(D_N)` over `(Y, Ydot, X_anti, force)` or over encoder states, per training set, with anchors spanning `Y in [-0.30, 0.30]` |
| Least-costly space-filling | `min_theta C(theta)` s.t. `V(theta; D_N(theta)) <= gamma` (11), `C = P_u = (1/N) sum u^2` (10) | Kiss, M. Schoukens, Toth, arXiv:2605.02517v2, 2026 (downloaded). NOTE: `docs/excitation-design-literature.md` line 58 attributes this to "Bombois et al."; the authors are Kiss, Schoukens, Toth | Eqs. (7), (10), (11), p3 | partly: SISO NARX, open loop, NOE models | same `V`, with multisine RMS as `C` |
| Active learning with feature-space reachability | Assumption 4: for any `x_0` and unit `v`, inputs exist with `|<phi(x_t,u_t), v>| >= alpha > 0` for some `t <= H`; goal: grow `lambda_min(Phi^T Phi)` (19) | Mania, Jordan, Recht, arXiv:2006.10277 (downloaded; journal version not checked) | Assumptions 1, 2, 4; Theorem 1; Eq. (19), pp4-12 | partly: known features, linear in parameters, full state measured | `lambda_min` of the ANN feature Gramian on training data, and which directions `v` are never reached |
| Non-asymptotic PE (linear) and its nonlinear analogue | "persistency of excitation holds if and only if the empirical covariance matrix is strictly positive definite"; Theorem 5.2 (ARX PE with burn-in); nonlinear: the "lower uniform law" (7.5) plays the role of PE (Remark 7.1), under A2 (iid blocks, i.e. mixing) and A3 (finite fourth moments) | Ziemann, Tsiamis, Lee, Jedra, Matni, Pappas, arXiv:2309.03873, 2023 (downloaded) | pp22-23, Sect. 7, pp32-34, App. F | no for our data: stochastic, mixing inputs; noiseless deterministic multisine not covered | none; cite only as the ML-side statement that PE generalises to a lower-tail bound on `sum ||f(X_t)||^2` |
| Closed-loop SUBNET training feasibility | (from repo notes, not re-read) black-box SUBNET could not be identified in closed loop from random init; FP-initialised model made it feasible | Kessels et al., Nonlinear Dyn. 2025 (held, `docs/references.md`) | thesis pp199-201 as recorded | yes (same class of machine, ASMPT) | none |
| APRBS / data-distribution excitation | metadata only | Heinz, Nelles, at-Automatisierungstechnik 2018 (repo doc) | not read | unknown | none |

**Synthesis E2.** No paper read gives a checkable excitation condition for an encoder-based neural
state-space model trained in closed loop; both SUBNET and Gyorok state PE as distinguishability
(Condition 6 / Condition 14) and Gyorok explicitly says it must be checked at the nonlinear-regressor
level, depending on outputs too. The operational substitutes are (a) coverage: filling distance `rho(D_N)`
of the joint input-state region (Liu 2025, Kiss 2026), and (b) feature-Gramian rank / `lambda_min`
(Hassaballa-Lazar, Mania). For the 8-state learned component the region of interest must include the Y
range and the absorber band; coverage is the only one of these that a closed-loop multisine plus
reference can be designed against, since the controller removes the freedom Liu's Condition 4 assumes.

### E4. Condition 4 as input design, and relatives

| condition or method | formula as read | paper | where | applies? | quantity we could compute |
|-|-|-|-|-|-|
| Condition 4 (empirical orthogonality) and its domain form | `sum_{i=0}^{N-1} phi^T(x_i) delta(x_i) = 0` (17); `int_X phi^T(x) delta(x) dx = 0` (16); "(17) is a specific excitation condition"; Remark 5: if the structure of `delta` is known, "it is possible to specifically design an input sequence that satisfies Condition 4"; example design: second half of input = first half times -1 | Gyorok et al. 2026 (held) | Eqs. (16), (17), Condition 4, Remark 5, p4; Sect. 5.1, p7 | partly: IO linear-in-parameters; our baseline is nonlinear-in-parameters state-space, so `J` replaces `phi` locally | `J^T Delta*` and `rho* = ||P Delta*|| / ||Delta*||` (measured 0.205; Condition 4 requires 0) |
| Alias matrix: bias of estimated coefficients from omitted terms | `E(beta_hat) = beta + A beta~` (5), `A = (H^T H)^(-1) H^T H~` (6); "The alias matrix A determines the pattern of bias in beta_hat ... and can be controlled through the choice of design" | Woods, Lewis, "Design of experiments for screening", Handbook of UQ, Springer 2017, arXiv:1510.05248 (downloaded) | Eqs. (5), (6), p3 | yes as structure: this IS Gyorok Eq. (21) with `Phi = H`, `Delta = H~ beta~`; the design literature targets `H^T H~ = 0` (robust to unknown `beta~`), which is stronger than Condition 4 (`H^T H~ beta~ = 0`) | ALIAS MATRIX OF THE ABSORBER: `A = (J^T J)^(-1) J^T H~`, with `H~` = output sensitivities to the hidden absorber parameters (`ma`, `ka`, `ca`, `L0`); its rows say which of the 10 combinations each absorber term biases (the 98.8 % `m_diff` bias should appear as one dominant row) |
| Foldover design | "The second run in the pair is formed by multiplying all the elements in the first run by -1 ... the 2d runs form a foldover design. This foldover property ensures that main effects and two-variable interactions are orthogonal"; "This foldover structure ensures that any two-variable interactions will not bias estimators of grouped main effects" | same | Sect. 2.2 p7; Sect. 3.2 p12 | static only: Gyorok Sect. 5.1 is a foldover design; Sect. 5.2 concedes it breaks once the regressor holds past outputs (prior sweep) | sign-symmetry check of `J^T Delta*` under `f_ms -> -f_ms` records (odd/even split of `Delta*`) |
| Alias matrix origin; model-robust (minimum-bias) response-surface design | Scholar snippet: "for any orthogonal design of this type the expected value of the ith linear effect is ..."; Box-Draper 1959: metadata only (cited by Krishna 2021 as the start of designs robust to model assumptions) | Box, Hunter, Ann. Math. Stat. 28(1), 1957; Box, Draper, JASA 54(287), 1959 | not read (snippet / metadata) | static | none |
| Identification-experiment design with unmodelled dynamics, bias-partitioned `X^T X` | Scholar snippet only: "magnitude and smoothness of the unmodelled dynamics, and uses ... X^T X, is partitioned according to the parameters of interest (beta_1) ... to bias comparisons" | Koung, MacGregor, "Identification for robust multivariable control: the design of experiments", Automatica 30(10):1541-1554, 1994 | snippet only, `needs-browser-route` | unknown until read; the only dynamic-identification paper found that appears to import the Box-Draper bias partition | would give a bias criterion for choosing the multisine spectrum |
| Input design making bias of a property insensitive to undermodelling | abstract: input spectrum chosen so that "the (asymptotic) variance error of a scalar function of the model parameters becomes independent of the order of the true system"; "there are circumstances when using this type of input allows some model properties to be estimated consistently even when the model order is lower than the order of the true system"; open loop, LTI | Martensson, Hjalmarsson, IEEE TAC 56(1):100-112, 2011 | abstract only (KTH DiVA record), `needs-browser-route` | no as stated (open loop, LTI, scalar property); closest control-side statement of "design the input so undermodelling does not bias the quantity of interest" | none until read |
| Bias distribution via design variables | `theta* = argmin int |G_0 - G(theta)|^2 Q(omega, theta) d omega`, `Q = |L|^2 Phi_u |W|^2` (4.22); `D* = argmin J(D)` iff `Q(omega, theta_F, D*) = alpha C(omega)` (4.11) | Wahlberg, Ljung, IEEE TAC 31(2), 1986 (held) | Theorem 4.1, Eqs. (4.4), (4.5), (4.11), (4.22) | per frozen Y only; shapes WHERE in frequency the misfit lands, not WHICH parameters absorb it | predicted frequency weighting of our OE loss per record (`Phi_u` after closed-loop shaping) |
| Robust design for calibration with model discrepancy | `D = D_eta union D_delta`; `D_eta = argmax |J_0^T J_0|` (3) (locally D-optimal, `J_0` sensitivity matrix); `D_delta` space-filling (MaxPro); `r` replicates at `D_eta`, default `r = 2` | Krishna, Joseph, Ba, Brenneman, Myers, J. Quality Technology 54(4):441-452, 2021, arXiv:2008.00547 (downloaded) | Sect. 2.1, Eq. (3), pp4-6 | partly: static computer model; designs for estimability of `eta` and `delta`, not for their orthogonality | `log det(J^T J)` plus coverage `rho(D_N)`, i.e. E1 and E2 quantities combined |
| Preposterior identifiability design | abstract: preposterior covariance of calibration parameters "can be used as a criterion for designing physical experiments to help achieve better identifiability" | Arendt, Apley, Chen, IIE Transactions 48(1):75-88, 2016 | abstract only (figshare holds only supplementary material) | partly: KOH GP discrepancy, static | none |
| BED with neural discrepancy | abstract: hybrid BED decoupling "low-dimensional physical parameters" from "high-dimensional model discrepancy" (AD-EKI) | Yang, Dong, Wu, arXiv:2504.20319, 2025 | abstract only | partly: PDE, Bayesian | none |
| L2 calibration defines the estimand by orthogonality | (prior sweep) `theta* := argmin ||zeta - y_s(., theta)||_L2`, orthogonality holds automatically at `theta*` | Tuo, Wu, AoS 2015 (held) | Eq. (2.2) as recorded in `literature-orthogonality-by-design.md` | framing only | none |
| Orthogonal-projection regulariser on trajectory data | `R(phi) = ||U^T_{1,zeta^0_n} g_phi(T(theta_d))||_2^2` (22) | Kon et al., IFAC WC 2023, arXiv:2303.07994 (held); Kon et al. ACC 2022 (held) | Eq. (22), p4 | yes as estimator, no as design (no excitation condition stated) | none new |
| Excitation allocation, fixed modules | generic identifiability of networks with fixed modules via external-signal allocation (graph condition) | Dreef et al., IEEE L-CSS 6:2587-2592, 2022 (held) | abstract p2 | no: generic rank, known LTI fixed module, no inner product with a discrepancy (agrees with prior sweep) | none |
| Undermodelling-aware least-costly / optimal input design | (prior GPT sweep, abstract level) Bombois, Gilson IFAC 2006; Suzuki, Sugie CDC 2007; Hildebrand, Gevers IFAC 2003 (variance effect of undermodelling via correlation of prediction errors with their gradients) | as listed in `deep-research-orthogonality.md` | not re-read this run | leads only | none |

**Synthesis E4.** Condition 4 is, verbatim, the zero-alias-matrix condition of classical design of
experiments: Gyorok Eq. (21) is `E(beta_hat) = beta + A beta~` with `A = (H^T H)^(-1) H^T H~`, and the
Sect. 5.1 "input times -1" construction is a foldover design, which the DoE literature proves makes
main effects orthogonal to two-factor interactions. So Remark 5 (design the input when the structure of
`delta` is known) is a solved problem for STATIC linear-in-parameters regression with a known omitted-term
basis. What is not solved in anything read: the dynamic case where the regressor contains states and
past outputs the experimenter cannot set, closed loop, and a nonlinear-in-parameters baseline. The
control-side near misses (Koung-MacGregor 1994, Martensson-Hjalmarsson 2011) are unread in full.

## B. What the literature says about Condition 4 as a design problem

Closest precedents, ranked:
1. **Alias matrix / foldover / resolution designs (statistics DoE)**, read in Woods and Lewis 2017 Eqs.
   (5), (6) and pp7, 12; origin Box and Hunter 1957, Box and Draper 1959 (not read). Exactly the
   structure of Condition 4 plus Remark 5, solved by design for static regression. The design goal there
   is `H^T H~ = 0` (robust to the unknown size of the omitted terms), strictly stronger than Condition 4.
   This vocabulary appears nowhere in the repo's held PDFs (multi-PDF grep of `literature/Orthogonality/`
   and `literature/experiment-design/Papers/` for "alias matrix", "foldover": zero hits before this run).
2. **Koung and MacGregor, Automatica 1994** (snippet only): appears to bring the Box-Draper bias
   partition into identification-experiment design with unmodelled dynamics. Highest-value unread item,
   `needs-browser-route`.
3. **Martensson and Hjalmarsson, IEEE TAC 2011** (abstract only): input spectra for which some model
   properties are estimated consistently despite undermodelling; open-loop LTI.
4. Wahlberg and Ljung 1986: bias distribution is design-controlled, but in frequency, not in parameter
   directions. Krishna 2021, Arendt 2016, Yang 2025: calibration design for estimability, not
   orthogonality.

**Novelty grade.** The static form of "design the input so the unmodelled term does not bias the
parameters of interest" is NOT novel (sixty-year-old DoE; Gyorok's own Sect. 5.1 is a foldover). A
design for a closed-loop, Y-scheduled, nonlinear-in-parameters state-space baseline, where the regressor
is a dynamic sensitivity that the input shapes only indirectly, was not found: grade MEDIUM-PROVISIONAL,
because (a) dblp was bot-walled so CDC/ECC/ACC/SYSID 2023-2026 are unreached, (b) Koung-MacGregor and
Martensson-Hjalmarsson were not read in full, (c) OpenAlex search was briefly rate-limited.
Vocabularies searched: control / sysid ("input design", "undermodeling", "unmodeled dynamics",
"bias distribution", "informative experiment"); statistics / DoE ("alias matrix", "foldover",
"model discrepancy", "experimental design", "calibration", "preposterior"); machine learning / UQ
("Bayesian experimental design", "model discrepancy", "persistency of excitation" with "neural").

Concrete consequence for us (our reasoning, not a literature claim): with the absorber structure known,
the DoE route suggests computing the alias matrix of the absorber sensitivities onto `J` and choosing
`(r, f_ms)` spectra, per Y, to shrink its dominant row (the `m_diff` direction), rather than targeting
the scalar `rho*`.

## C. Findings, Access, Evidence, Research Log

## Findings

- **Gevers, Bazanella, Bombois, Miskovic**, "Identification and the information matrix: how to get just
  sufficiently rich?", IEEE TAC 54(12), 2009. Local `literature/Orthogonality/gevers2009_...pdf`. Thms 4.3,
  5.1, 6.3 as in E1 table. Already in `docs/references.md`.
- **Bazanella, Gevers, Miskovic**, EJC 2010, local `literature/closed-loop-id/`. Not all references need
  excitation in MIMO closed loop. Held.
- **Colin, Bombois, Bako, Morelli**, Automatica 121:109171, 2020, HAL hal-02351669, local. Condition not
  extracted this run. Held.
- **Gautier, Janot, Vandanjon**, IEEE TCST 21(2), 2013, local. p6 exciting-trajectory sentence is new to
  the repo's notes (previous notes quote its CLOE/DIDIM mechanism, not this).
- **Brun, Reichert, Kuensch**, WRR 2001, local. Collinearity index, critical 5 to 20. Held.
- **Raue et al.**, Bioinformatics 2009, local. Profile likelihood Eq. (10), Def. 1. Held.
- **Gyorok et al. 2026**, local. Condition 4, Remark 5, Condition 14. Held; no new clause beyond prior sweep
  except the Condition 14 sentence "PE requirements must be assessed at the level of the nonlinear
  regressor".
- **Beintema, Schoukens, Toth**, Automatica 2023, local. Conditions 1 and 6. Held.
- **Hassaballa, Lazar**, arXiv:2606.04290, local. Assumption 1, Lemma 1. Held.
- **Liu, Kiss, Toth, Schoukens**, L-CSS 2025, local. Held.
- **Kiss, Schoukens, Toth**, "Least costly space-filling experiment design for the identification of a
  nonlinear system", arXiv:2605.02517v2, 2026. NEW download:
  `literature/experiment-design/Papers/kiss2026_least-costly-space-filling-nonlinear_arXiv2605.02517.pdf`.
  Finding: `min C(theta)` s.t. `V <= gamma` (11); 40 % lower input power for <1 % change in `V` in their MSD
  study. Corrects the author attribution in `docs/excitation-design-literature.md` (stale sibling, not
  edited).
- **Mania, Jordan, Recht**, "Active learning for nonlinear system identification with guarantees",
  arXiv:2006.10277. NEW download `.../mania2022_active-learning-nonlinear-sysid-guarantees_JMLR_arXiv2006.10277.pdf`
  (filename assumes the JMLR venue; not verified this run).
- **Ziemann, Tsiamis, Lee, Jedra, Matni, Pappas**, "A tutorial on the non-asymptotic theory of system
  identification", arXiv:2309.03873. NEW download `.../ziemann2023_tutorial-non-asymptotic-sysid_CDC-tutorial_arXiv2309.03873.pdf`
  (venue in filename unverified).
- **Woods, Lewis**, "Design of experiments for screening", Handbook of UQ, Springer 2017, pp1143-1185,
  DOI 10.1007/978-3-319-12385-1_33, arXiv:1510.05248. NEW download `.../woods2017_design-of-experiments-for-screening_HandbookUQ_arXiv1510.05248.pdf`.
  Finding: alias matrix (5), (6); foldover orthogonality (pp7, 12). Headline find for E4.
- **Krishna, Joseph, Ba, Brenneman, Myers**, JQT 54(4):441-452, 2021, DOI 10.1080/00224065.2021.1930618,
  arXiv:2008.00547. NEW download `.../krishna2021_robust-experimental-designs-model-calibration_JQT_arXiv2008.00547.pdf`.
- **Koung, MacGregor**, Automatica 30(10):1541-1554, 1994, DOI 10.1016/0005-1098(94)90094-9, closed,
  `needs-browser-route`. Snippet only.
- **Martensson, Hjalmarsson**, IEEE TAC 56(1):100-112, 2011, DOI 10.1109/tac.2010.2052294, closed,
  `needs-browser-route`; abstract via KTH DiVA urn:nbn:se:kth:diva-7544.
- **Arendt, Apley, Chen**, IIE Trans. 48(1):75-88, 2016, DOI 10.1080/0740817x.2015.1064554, abstract only.
- **Yang, Dong, Wu**, arXiv:2504.20319, 2025, abstract only.
- **Wahlberg, Ljung**, IEEE TAC 31(2), 1986, local `literature/Orthogonality/Design variables for bias distribution in transfer function estimation.pdf`. Held.
- **Box, Hunter** 1957 (DOI 10.1214/aoms/1177707047) and **Box, Draper** 1959 (DOI 10.1080/01621459.1959.10501525): metadata / snippet.
- **Swevers et al.** 1997 (DOI 10.1109/70.631234), **Gautier, Khalil** 1992 (DOI 10.1177/027836499201100408): metadata.

BibTeX (Crossref content negotiation, keys renamed to repo style):
```bibtex
@inbook{woods2017screening, title={Design of Experiments for Screening}, booktitle={Handbook of Uncertainty Quantification}, publisher={Springer International Publishing}, author={Woods, David C. and Lewis, Susan M.}, year={2017}, pages={1143-1185}, doi={10.1007/978-3-319-12385-1_33}, note={arXiv:1510.05248}}
@article{koung1994identification, title={Identification for robust multivariable control: the design of experiments}, journal={Automatica}, volume={30}, number={10}, pages={1541-1554}, year={1994}, author={Koung, Ching-Wei and MacGregor, John F.}, doi={10.1016/0005-1098(94)90094-9}}
@article{martensson2011insensitive, title={How to Make Bias and Variance Errors Insensitive to System and Model Complexity in Identification}, journal={IEEE Transactions on Automatic Control}, volume={56}, number={1}, pages={100-112}, year={2011}, author={Martensson, Jonas and Hjalmarsson, H{\aa}kan}, doi={10.1109/TAC.2010.2052294}}
@article{krishna2021robustcalib, title={Robust experimental designs for model calibration}, journal={Journal of Quality Technology}, volume={54}, number={4}, pages={441-452}, year={2021}, author={Krishna, Arvind and Joseph, V. Roshan and Ba, Shan and Brenneman, William A. and Myers, William R.}, doi={10.1080/00224065.2021.1930618}}
@article{arendt2016preposterior, title={A preposterior analysis to predict identifiability in the experimental calibration of computer models}, journal={IIE Transactions}, volume={48}, number={1}, pages={75-88}, year={2016}, author={Arendt, Paul D. and Apley, Daniel W. and Chen, Wei}, doi={10.1080/0740817X.2015.1064554}}
@article{box1959basis, title={A Basis for the Selection of a Response Surface Design}, journal={Journal of the American Statistical Association}, volume={54}, number={287}, pages={622-654}, year={1959}, author={Box, G. E. P. and Draper, Norman R.}, doi={10.1080/01621459.1959.10501525}}
@article{box1957multifactor, title={Multi-Factor Experimental Designs for Exploring Response Surfaces}, journal={The Annals of Mathematical Statistics}, volume={28}, number={1}, pages={195-241}, year={1957}, author={Box, G. E. P. and Hunter, J. S.}, doi={10.1214/aoms/1177707047}}
@article{swevers1997optimal, title={Optimal robot excitation and identification}, journal={IEEE Transactions on Robotics and Automation}, volume={13}, number={5}, pages={730-740}, year={1997}, author={Swevers, J. and Ganseman, C. and Tukel, D. B. and De Schutter, J. and Van Brussel, H.}, doi={10.1109/70.631234}}
@article{gautier1992exciting, title={Exciting Trajectories for the Identification of Base Inertial Parameters of Robots}, journal={The International Journal of Robotics Research}, volume={11}, number={4}, pages={362-375}, year={1992}, author={Gautier, M. and Khalil, W.}, doi={10.1177/027836499201100408}}
@misc{kiss2026leastcostly, title={Least Costly Space-Filling Experiment Design for the Identification of a Nonlinear System}, author={Kiss, M{\'a}t{\'e} and Schoukens, Maarten and T{\'o}th, Roland}, year={2026}, eprint={2605.02517}, archivePrefix={arXiv}}
@misc{mania2020active, title={Active Learning for Nonlinear System Identification with Guarantees}, author={Mania, Horia and Jordan, Michael I. and Recht, Benjamin}, year={2020}, eprint={2006.10277}, archivePrefix={arXiv}}
@misc{ziemann2023tutorial, title={A Tutorial on the Non-Asymptotic Theory of System Identification}, author={Ziemann, Ingvar and Tsiamis, Anastasios and Lee, Bruce and Jedra, Yassir and Matni, Nikolai and Pappas, George J.}, year={2023}, eprint={2309.03873}, archivePrefix={arXiv}}
```
(Ziemann et al. author list verified on the PDF title page, arXiv v2, 16 Jun 2024.)

## Access status
TU/e browser access: UNAVAILABLE (layer 1: no browser bridge in this session). Items marked
`needs-browser-route`: Koung and MacGregor 1994 (rank 1), Martensson and Hjalmarsson 2011 (rank 2),
Box and Draper 1959, Box and Hunter 1957, Swevers et al. 1997, Gautier and Khalil 1992, Arendt et al.
2016 full text. No publisher fetch attempted. MCP DOI retrieval (route 6) not used.

## Evidence quality
- Grep-read (equations and theorem statements read in the PDF text): Gevers 2009; Bazanella 2010
  (conclusion); Gautier 2013 (p6); Brun 2001 (formula OCR-garbled, interpretation and thresholds read);
  Raue 2009; Gyorok 2026; Beintema 2023; Hassaballa-Lazar 2026; Liu 2025; Kiss 2026; Mania 2020;
  Ziemann 2023 (Sects. 5.1, 7, App. F partially); Woods-Lewis 2017 (pp3-13); Krishna 2021 (pp1-6);
  Wahlberg-Ljung 1986; Kon 2023 (Eq. 22); 5SMB0 Lectures 6 and 11.
- Conclusion / first page only: Colin 2020; Dreef 2022.
- Abstract only: Martensson-Hjalmarsson 2011; Arendt 2016; Yang 2025.
- Snippet only: Koung-MacGregor 1994; Box-Hunter 1957.
- Metadata only: Box-Draper 1959; Swevers 1997; Gautier-Khalil 1992; Heinz-Nelles 2018.
- Second-hand: Ljung 1999 Thm 13.1 (via Gevers 2009); Kessels 2025, Tuo-Wu 2015, Bombois-Gilson 2006,
  Suzuki-Sugie 2007, Hildebrand-Gevers 2003 (via repo notes).

## Research Log
- Queries run:
  - Local: grep of 6 local docs, repo-wide name grep (13 names), 5SMB0 lectures 6/10/11/12, multi-PDF grep of
    two literature folders for "alias matrix"/"foldover"/"Box and Draper" (0 relevant hits: vocabulary new to repo).
  - OpenAlex (7 of 15): title.search Martensson (HTTP 429 search-cluster transient, 34 s), title.search
    Krishna (1, on-target), title.search Arendt preposterior (1, on-target), works/doi x3 (Martensson closed,
    Arendt green via figshare supplement only, Koung closed), title_and_abstract "alias matrix" bias design
    is_oa (3, 0 on-target).
  - arXiv API (9 of 10, all non-zero bytes): PE+neural+state-space (0, genuine), Ziemann tutorial title (1,
    on-target), active learning nonlinear sysid title (3, 2 on-target), nonlinear+persistence of excitation (0),
    Krishna title (1), "alias matrix"+"bias" (0), model discrepancy+experimental design+calibration (3, 2
    on-target), input design+undermodeling/unmodeled dynamics (0), experiment design+closed-loop+physical
    parameters (0).
  - dblp (3 of 4): informativity+closed-loop, input+design+undermodeling, persistency+excitation+neural: all
    returned HTTP 200 with a 7.4 kB Anubis bot-check page ("Making sure you're not a bot!"). Not zeros:
    unreachable.
  - Google Scholar (4 of 6): E4 sentence with "parameters of interest" (12, 1 on-target: Koung-MacGregor);
    E4 "alias matrix" sentence (10, 3 on-target: Box-Hunter, Woods-Lewis, Jones-Nachtsheim); E1 closed-loop
    grey-box "informative experiment" (6, 0 on-target); E2 neural SS closed-loop "state space coverage" (10,
    0 on-target).
  - Crossref: ~10 metadata lookups; DOI BibTeX x9. KTH DiVA export API x1 (abstract). figshare API x1.
- What worked: translating Condition 4 into DoE vocabulary ("alias matrix", "foldover") and writing the
  Scholar query as the sentence the paper would contain; that single query surfaced the open Woods-Lewis
  chapter that states the formula. KTH DiVA MODS export recovered a closed TAC abstract.
- What failed: dblp (bot wall, new failure mode); OpenAlex search briefly 429; Scholar on E1/E2 (keywords
  shared with Koopman/RL/mHealth noise).
- Dead ends: figshare record for Arendt holds only supplementary material; OpenAlex "alias matrix" returned
  orthogonal-array papers only.
- Coverage gaps: CDC/ECC/ACC/SYSID 2023-2026 unreached (dblp); forward citations of Koung-MacGregor and
  Martensson-Hjalmarsson not traversed; Colin 2020 condition not extracted; Soderstrom-Stoica and Ljung 1999
  not held; closed-loop nonlinear informativity (IVNN arXiv:2202.05337) not read.
- Grade of negative claims: "no checkable PE condition for closed-loop encoder-based neural SS" MEDIUM
  (arXiv abstract count 0 is genuine; dblp unreached). "No design for Condition 4 in dynamic closed-loop
  grey-box" MEDIUM-PROVISIONAL (two strong unread candidates). "SUBNET paper does not treat feedback"
  MEDIUM (grep of full text).
- Suggested skill fix: add to the dblp section: "dblp may serve an Anubis bot-check HTML page (title 'Making
  sure you're not a bot!', ~7.4 kB, HTTP 200) to curl; detect with `grep -q 'not a bot'` before JSON parsing
  and record the source as unreachable, never as zero hits". Also add "alias matrix / foldover / minimum-bias
  design" to the statistics vocabulary list for any "unmodelled term biases the parameters" question.

## D. REFERENCES-NEW candidates
| key | metadata | local path |
|-|-|-|
| `woods2017screening` | Woods, Lewis, "Design of experiments for screening", Handbook of UQ, Springer 2017, pp1143-1185, DOI 10.1007/978-3-319-12385-1_33, arXiv:1510.05248. Alias matrix Eqs. (5), (6); foldover p7 | `literature/experiment-design/Papers/woods2017_design-of-experiments-for-screening_HandbookUQ_arXiv1510.05248.pdf` |
| `kiss2026leastcostly` | Kiss, M. Schoukens, Toth, arXiv:2605.02517v2, 2026. Eq. (11) | `literature/experiment-design/Papers/kiss2026_least-costly-space-filling-nonlinear_arXiv2605.02517.pdf` |
| `krishna2021robustcalib` | Krishna, Joseph, Ba, Brenneman, Myers, JQT 54(4):441-452, 2021, DOI 10.1080/00224065.2021.1930618, arXiv:2008.00547 | `literature/experiment-design/Papers/krishna2021_robust-experimental-designs-model-calibration_JQT_arXiv2008.00547.pdf` |
| `mania2020active` | Mania, Jordan, Recht, arXiv:2006.10277 (journal version unverified) | `literature/experiment-design/Papers/mania2022_active-learning-nonlinear-sysid-guarantees_JMLR_arXiv2006.10277.pdf` |
| `ziemann2023tutorial` | Ziemann et al., arXiv:2309.03873, 2023 | `literature/experiment-design/Papers/ziemann2023_tutorial-non-asymptotic-sysid_CDC-tutorial_arXiv2309.03873.pdf` |
| `koung1994identification` | Koung, MacGregor, Automatica 30(10):1541-1554, 1994, DOI 10.1016/0005-1098(94)90094-9 (snippet only, needs-browser-route) | none |
| `martensson2011insensitive` | Martensson, Hjalmarsson, IEEE TAC 56(1):100-112, 2011, DOI 10.1109/TAC.2010.2052294 (abstract only, needs-browser-route) | none |
| `arendt2016preposterior` | Arendt, Apley, Chen, IIE Trans. 48(1):75-88, 2016, DOI 10.1080/0740817X.2015.1064554 (abstract only) | none |
| `yang2025bedmd` | Yang, Dong, Wu, arXiv:2504.20319, 2025 (abstract only) | none |
| `box1957multifactor` | Box, Hunter, Ann. Math. Stat. 28(1):195-241, 1957, DOI 10.1214/aoms/1177707047 (snippet) | none |
| `box1959basis` | Box, Draper, JASA 54(287):622-654, 1959, DOI 10.1080/01621459.1959.10501525 (metadata) | none |
| `swevers1997optimal` | Swevers et al., IEEE TRA 13(5):730-740, 1997, DOI 10.1109/70.631234 (metadata; check for duplicate `swevers` entries) | none |
| `gautier1992exciting` | Gautier, Khalil, IJRR 11(4):362-375, 1992, DOI 10.1177/027836499201100408 (metadata) | none |

Stale sibling to flag (not edited): `docs/excitation-design-literature.md` line 58 attributes
arXiv:2605.02517 to "Bombois et al."; the authors are Kiss, M. Schoukens, Toth.
