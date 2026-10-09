# Citation log

Passages read in the PDF under `literature/` and what each supports. One row per passage. Append; do not rewrite.

| Key | Location | Supports | Date |
|-|-|-|-|
| hoekstra2025lfr | Sec. 2, p. 2 | static augmentations add no states beyond the baseline ("flexible modes, actuator dynamics"); dynamic structures add x_a; learning outputs affect the entire model state | 2026-10-09 |
| hoekstra2025lfr | Sec. 3.2, Thm. 1, p. 3 | fixed interconnection matrices in this work; well-posedness by avoiding algebraic loops, acyclic graph iff topological ordering | 2026-10-09 |
| hoekstra2025lfr | Sec. 4.3, p. 4 | parallel learning components are feedforward networks; ResNets only needed for series structures, "not necessary for the parallel model, as the baseline model is additively augmented"; approximate initialisations not all corrected | 2026-10-09 |
| hoekstra2026lfr | Table 1, p. 3 | S-DP: x_b+ = f_base + f_aug(x_b, x_a, u), x_a+ = g_aug(x_b, x_a, u) in discrete time | 2026-10-09 |
| hoekstra2026lfr | Eq. (22a) to (22e), Sec. 5.1, p. 8 | truncated simulation criterion over windows of length T, every sample scored, encoder initial state from past y and u | 2026-10-09 |
| hoekstra2026lfr | Sec. 5.2, Eq. (26), p. 8 to 9 | parameters minimising (22) are not unique; learning components can represent or cancel baseline dynamics | 2026-10-09 |
| hoekstra2026lfr | Sec. 5.3, Eq. (28), p. 9 | normalisation by diagonal scaling only; sigma_x from a simulation of the baseline; zero-mean assumption | 2026-10-09 |
| hoekstra2026lfr | Sec. 5.4.1, Eq. (29), p. 9 | initialise so the augmented model behaves as the baseline | 2026-10-09 |
| hoekstra2026lfr | Sec. 5.4.2, Eqs. (30), (31), p. 9 | baseline encoder fitted to simulated baseline states (30); augmented-state encoder initialised by "the Xavier approach" (31) | 2026-10-09 |
| hoekstra2026lfr | Sec. 5.4.3, p. 10 | matrices not required for baseline behaviour initialised U(-1,1) (permits a live start) | 2026-10-09 |
| hoekstra2026lfr | Sec. 5.5, p. 10 | consistency inherited from SUBNET Conditions 2.1 to 2.4; 2.1 requires an incrementally exponentially output stable data-generating system | 2026-10-09 |
| hoekstra2026lfr | Sec. 6.3, p. 11 | parallel learning components are feedforward tanh networks; estimated physical parameters remain very close to initial values, learning components learn dynamics the baseline could represent | 2026-10-09 |
| hoekstra2026encoder | Eq. (8), Sec. 2, p. 2 | encoder = ANN correction plus linear parts W_b (baseline states) and W_a (augmented states) | 2026-10-09 |
| hoekstra2026encoder | Sec. 2, footnote 1 | co-estimating baseline parameters is outside the scope of the paper | 2026-10-09 |
| hoekstra2026encoder | Sec. 2, p. 2 | assumes a stable baseline model | 2026-10-09 |
| hoekstra2026encoder | Sec. 3.1, Eqs. (10) to (17), p. 3 | W_b,u = -A^n O_n^+ T_n + r_n, W_b,y = A^n O_n^+; left inverse exists if observable | 2026-10-09 |
| hoekstra2026encoder | Sec. 3.1, Eqs. (30) to (32), p. 4 | nonlinear baseline: linearise at an equilibrium, then apply the LTI map | 2026-10-09 |
| hoekstra2026encoder | Sec. 4.2, p. 5 | linearisation "around the 0 point", no reason given | 2026-10-09 |
| hoekstra2026encoder | Sec. 4.4, p. 5 to 6 | model-based init: better starting values, faster convergence, comparable final RMSE; about 1.5 ms computation vs 4.8 s data-based | 2026-10-09 |
| beintema2023subnet | Sec. 2 to 3, p. 2 to 3 | simulation-error minimisation scales with N and is unstable; truncated encoder windows reduce cost and improve stability | 2026-10-09 |
| beintema2023subnet | Condition 1, p. 5 | incremental exponential output stability of the data-generating system | 2026-10-09 |
| kessels2025ai | Eqs. (5.12), (5.13a) to (5.13d), thesis pp. 156 to 157 | closed-loop truncated-window criterion with encoder, EA model, FF plus FB controller; error formed against the model output; minimisation over NN weights only (FP parameters fixed) | 2026-10-09 |
| kessels2025ai | Remark 5.3, p. 157 | if h is accurate, positions can be initialised from measured outputs | 2026-10-09 |
| kessels2025ai | Remark 5.4, p. 157 | true FB controller states taken from the machine or reconstructed from y, r and the known controller, to initialise each window | 2026-10-09 |
| kessels2025ai | Remark 5.6, p. 174 | open-loop identification from closed-loop data attempted for the wire bonder, unsuccessful; noise-induced bias explanation | 2026-10-09 |
| kessels2025ai | Ch. 5 summary, p. 144 | embedding stabilising controllers lets open-loop unstable systems be updated | 2026-10-09 |
| forssell1999revisited | Sec. 5.2.1 (LiU report LiTH-ISY-R-2021 numbering), Eqs. (44), (45), (49) | indirect approach: identify r to y with known regulator; estimation is an open-loop problem since r and noise are uncorrelated; output-error models with fixed noise model remain consistent; parameterise the closed loop by the open-loop model | 2026-10-09 |
| gyorok2026obc | Abstract, p. 1 | OBC is defined for augmenting physics-based IO models with a learning component in an additive (parallel) sense | 2026-10-09 |
| gyorok2026obc | Sec. 3, p. 3 | with non-unique theta_b the baseline could lose its physical meaning | 2026-10-09 |
| drenth2025lpvlfr | Sec. 5 (optimisation), p. 4 | Adam iterations followed by L-BFGS-B, Adam giving a better starting point | 2026-10-09 |
| drenth2025thesis | Eq. (5.2), Sec. 5.1, p. 29 | augmented LPV-LFR: added states, latent variables and its own scheduling block Delta^a; LTI interconnection matrix G^a | 2026-10-09 |
| hefny2015supervised | Sec. 3, p. 4 | two-stage regression: S1A (possibly non-linear) regression of future features on history; S2 regression of a linear operator between S1 outputs | 2026-10-09 |
| downey2017psrnn | Sec. 4.2, p. 5 to 6 | initialise by two-stage regression, then refine by BPTT; 2SR gives an initialisation that converges to a good local optimum | 2026-10-09 |
| gyorok2026obc | Sec. 3.1, Example 2, p. 3 (arXiv 2511.01321v3 numbering) | linear baseline plus single linear ANN layer: any pair with W + theta_b = theta_b* minimises the loss (non-uniqueness of the additive split) | 2026-10-09 |
| gyorok2026obc | Sec. 2, Assumption 1, Eq. (7), p. 3 | rank(Phi) = n_theta on the training data | 2026-10-09 |
| gyorok2026obc | Sec. 3.2, Eqs. (8) to (11), Lemma 3, p. 3 | theta_aux = (Phi'Phi)^-1 Phi' F_ANN; F_tilde = (I - Phi(Phi'Phi)^-1 Phi') F_ANN; Phi' F_tilde = 0 on the training set; no trade-off parameter | 2026-10-09 |
| gyorok2026obc | Sec. 3.2, text after Eq. (12), p. 3 | theta_aux may be built from any auxiliary evaluation of Phi (synthetic set or subset of the estimation data) | 2026-10-09 |
| gyorok2026obc | Sec. 3.2, Eq. (13), p. 3 | prediction on new data: phi(x_k) theta_b + f_ANN(x_k) - phi(x_k) theta_aux with the fixed theta_aux | 2026-10-09 |
| gyorok2026obc | Sec. 4.1, Eqs. (14) to (17), Condition 4, Remark 5, Assumption 6, Theorem 7, Eq. (21), p. 4 | true system phi theta_b* + delta; Condition 4: sum phi' delta = 0; under Assumptions 1, 6 and Condition 4 the baseline error is zero; otherwise error norm (Phi'Phi)^-1 Phi' Delta; Remark 5: input can be designed to meet Condition 4 when the structure of delta is known; Assumption 6 realistic only for noise-free data | 2026-10-09 |
| gyorok2026obc | Sec. 4.2, Eq. (25), pp. 4 to 5 | when Condition 4 fails the estimate is unique with error (21); the standard additive structure can give larger errors in a worst-case sense | 2026-10-09 |
| gyorok2026obc | Sec. 4.3 to 4.4, Theorems 16, 18, p. 6 | consistency and zero covariance between baseline and ANN parameters, for the DT IO linear-in-parameters setting with a stable data-generating system (Condition 10) | 2026-10-09 |
| gyorok2026obc | Sec. 6 (Conclusion), pp. 9 to 10 | extension to state-space baselines without full-state measurement named as future research | 2026-10-09 |
| gyorok2025l4dc | Sec. 3, Example 1, p. 5 (arXiv 2501.05842v2) | state transition f = theta x with ANN ~ W x: any pair with W + theta = theta* minimises the cost | 2026-10-09 |
| gyorok2025l4dc | Sec. 3, Eq. (13), Remark 1, p. 6 | orthogonality promoted by a penalty with coefficient beta, a trade-off between orthogonality and performance | 2026-10-09 |
| gyorok2025l4dc | Sec. 3, text before Remark 2, p. 7 | without state measurements an approximate state set is obtained by forward simulating the FP model on the training data | 2026-10-09 |
| gyorok2025l4dc | Sec. 4, Eqs. (15) to (18), p. 7 | nonlinear FP model: Taylor expansion in the parameters at a linearisation point; extended regressor [Phi Gamma] protects the offset too; the point is updated at every cost evaluation (SVD recomputed each time) or fixed at the nominal parameters | 2026-10-09 |
| kessels2025ai | Sec. 5.3.1.3, thesis p. 166 | extension and augmentation terms might (partially) negate the FP terms, called negation; undesirable; mitigation left to future work | 2026-10-09 |
| garcia2013model | Sec. 2.2, pp. 8 to 9 (accepted-manuscript PDF in literature/gantry) | fourteen-parameter lumped model; coordinates (X, Theta, Y) avoid closed kinematic chains and expose the non-uniform load-distribution coupling; Y measured from the centre of the cross-arm; flexible joints between cross-arm and X actuators | 2026-10-09 |
| garcia2013model | Sec. 2.2, Eq. (6), p. 9 | generalised forces include Coulomb terms cc1, cc2, ccy times sign of the velocities | 2026-10-09 |
| garcia2013model | Sec. 2.3, Eqs. (9) to (11), pp. 10 to 11 | Theta limited to tens of microrad in operation; cos Theta ~ 1, sin Theta ~ 0 give M12 = -(mhY - (m1-m2)Lb/2); Coriolis and centripetal matrix H ignored; simplified model (11) keeps X and Theta only, M13, M31, M32 neglected, M23 moved to the force vector | 2026-10-09 |
| garcia2013model | Table I, p. 13 | nominal values mb, mh, m1, m2, Jb, Jh, cg1, cg2, cy, cb1 = cb2, kb1 = kb2, Lb, d (match gantry_ss.py) | 2026-10-09 |
| drenth2025thesis | Sec. 2.1, Eqs. (2.1), (2.2), (2.4), p. 5 | continuous-time LPV-LFR pair (G, Delta(p)) with constant G, repeated diagonal Delta block; elimination needs I - Dzw Delta nonsingular for all p | 2026-10-09 |
| drenth2025thesis | Sec. 2.1.1, p. 6 | treating scheduling variables that depend on the same state independently admits combinations that cannot occur; rational dependence keeps the coupling | 2026-10-09 |
| drenth2025lpvlfr | Sec. 3.1, Eqs. (6), (7), Def. 1, p. 3 | DT LPV-LFR {M, Delta(p)}; well-posed iff det(I - Dzw Delta(p)) != 0 for all p in P | 2026-10-09 |
| drenth2025lpvlfr | Sec. 3.2, p. 3 | self-scheduled LPV-LFR models require joint estimation of the LPV model and the scheduling map | 2026-10-09 |
| drenth2025lpvlfr | Sec. 4.1, Assumptions 2, 3, Condition 5, Thm. 6, pp. 3 to 4 | sufficient well-posedness: diagonal Delta, scheduling in the unit infinity-ball, rho(Dzw) < 1 | 2026-10-09 |
| toth2010modeling | Sec. 7.3, Def. 7.2, p. 174 | quasi-LPV: the scheduling variable is not a free variable (depends on the system signals) | 2026-10-09 |
| ovchinnikov2021computing | Sec. 2, Def. 7 (arXiv 2004.07774v3) | identifiable function of the parameters; stated as equivalent to Hong et al. Def. 2.5 | 2026-10-09 |
| hoekstra2026lfr | Eq. (22e), Sec. 5.1, p. 8 | the encoder of the truncated criterion reads y and u from k-n to k-1 (past samples only) | 2026-10-09 |
| hoekstra2026encoder | Sec. 3.1, Eqs. (11) to (17), p. 3 | the reconstructability map gives x_{b,k} from y and u over k-n to k (current sample included) | 2026-10-09 |
| hoekstra2026encoder | Sec. 3.2, Eqs. (33), (34), p. 4 | data-based alternative: fit W_b to states obtained by forward simulating the baseline on the data | 2026-10-09 |
| kessels2025ai | Remark 5.1, PDF p. 179 (thesis p. 153) | the output augmentation does not depend on the input, since otherwise y would depend on the FB controller output that depends on y (non-causal relation) | 2026-10-09 |
| kessels2025ai | Remark 5.3, PDF p. 184 (thesis p. 158 by its footer) | if h is accurate there is no need to augment the output equation; positions can then be initialised from the measured outputs | 2026-10-09 |
| hoekstra2026lfr | Sec. 7.2, 7.4, p. 13 | F1Tenth: full state measured (y = x_b); integrators detached, only the input-to-velocity relationship identified, positions recovered afterwards by putting the integrators back | 2026-10-09 |
| gyorok2026obc | Sec. 2, Eq. (2), p. 2 (arXiv 2511.01321v3) | the baseline is a DT IO model in linear-in-the-parameters form, y_hat = phi(x) theta_b | 2026-10-09 |
| drenth2025thesis | Sec. 2.1.1, Eqs. (2.8), (2.9), p. 6 | rational embedding of the NL-MSD with one scheduling variable p1 repeated in Delta_rat = diag(p1, p1, p1), realising the quadratic dependence by chained latent signals instead of a second scheduling variable p2 = x^2 | 2026-10-09 |
| drenth2025thesis | Sec. 2.1, text after Eq. (2.4), p. 5 | CT LPV-LFR is well posed when I - Dzw Delta(p) is nonsingular for all p (elimination of z, w gives the rational LPV-SS form) | 2026-10-09 |
| gyorok2026obc | Sec. 4.4, Eq. (39), p. 6 (arXiv 2511.01321v3) | output and output-Jacobian relations of the covariance analysis; does NOT support a parameter-uniqueness claim (use Assumption 1, Eq. (7) for that) | 2026-10-09 |
| gyorok2025l4dc | Sec. 3, Remark 2, p. 7 (arXiv 2501.05842v2) | the projection matrix can be updated with current state estimates, recomputing the SVD at the start of each epoch | 2026-10-09 |
| drenth2025thesis | Sec. 5.2, Eq. (5.1), p. 29 | LPV-LFR model augmentation assumes the prior (baseline) model in (LPV-)LFR form (G^b, Delta^b), self-scheduled with a user-defined scheduling map; discrete time | 2026-10-09 |
| gyorok2026obc | Sec. 4.1, Assumption 6, Eq. (18), Thm. 7 proof Eqs. (19) to (21), p. 4 (arXiv 2511.01321v3) | Assumption 6: the identified model reproduces Phi theta_b* + Delta exactly on the training data (exact fit); Thm. 7 uses Assumptions 1 and 6 with Condition 4; the thesis Proposition 1 does not need the exact fit | 2026-10-09 |
| gyorok2025l4dc | p. 1 (front matter) | PMLR vol. 283, 7th L4DC 2025; authors Gyorok, Hoekstra, Kon, Peni, Schoukens, Toth; per-paper pages 1 to 13 (bib metadata) | 2026-10-09 |
| gyorok2026obc | p. 1 (arXiv v3 front matter) | title and authors Gyorok, Schoukens, Peni, Toth (bib metadata; journal data not on the arXiv PDF) | 2026-10-09 |
| hoekstra2026encoder | Sec. 4.4, p. 6 | model- and data-based initialisations give better starting values than random; data-based and, to a lesser extent, model-based converge faster; final RMSE comparable; model-based needs only a Moore-Penrose inverse (about 1.5 ms vs 4.8 s); data-based "might be preferred for nonlinear baseline models" | 2026-10-09 |
| gyorok2025l4dc | Sec. 4, text after Eq. (19), p. 8 (arXiv 2501.05842v2) | updating the linearisation point recomputes the SVD at every cost evaluation; fixing it at the nominal values is argued valid because nominal values typically align with the system (the paper does NOT say a fixed point loses accuracy) | 2026-10-09 |
| gyorok2026obc | Sec. 3.2, Eq. (9) and text after Eq. (12), p. 3 (arXiv 2511.01321v3) | during training the coefficient theta_aux is a function of the ANN parameters; after training a fixed theta_aux results | 2026-10-09 |
| gyorok2026obc | Sec. 4.3, Condition 10, Eq. (26), p. 5 (arXiv 2511.01321v3) | stable data-generating system: the effect of an initial-state difference decays exponentially | 2026-10-09 |
| garcia2013model | Sec. 1 (Introduction), p. 2 (accepted-manuscript PDF, literature/gantry/garcia.txt) | "dual-drive gantry stage": cross-arm mounted on two parallel linear actuators X1, X2; a third linear actuator (Y) on the cross-arm carries the payload; flexible joints connect cross-arm and X actuators | 2026-10-09 |
| hoekstra2025lfr | Sec. 3.1, Eq. (4), p. 3 | LFR-based augmentation structure: an interconnection matrix connects the baseline block and the learning block | 2026-10-09 |
| hefny2015supervised | Sec. 2, p. 4 (arXiv 1505.05310v2) | CORRECTION of the "Sec. 3" row: the S1A (possibly non-linear) and S2 (linear operator) regressions are stated in Sec. 2 "A framework for spectral algorithms"; Sec. 3 is Related Work | 2026-10-09 |
| hoekstra2026lfr | Sec. 5.1, Eq. (22), p. 8 | the joint parameter vector of the truncated criterion is col(theta_base, theta_aug, theta_LFR, theta_encoder): theta_base is co-estimated in Hoekstra's criterion too | 2026-10-09 |
| downey2017psrnn | p. 1 (arXiv 1705.09353v2 front matter) | title; first authors Downey, Hefny, Li (bib metadata) | 2026-10-09 |
| drenth2025lpvlfr | Sec. 3.1, text after Def. 1, p. 3 | the LPV-LFR admits rational LPV-SS models (affine is the case Dzw = 0); for nonlinear systems p is defined by a scheduling map psi of state and input (LPV embedding) | 2026-10-09 |
| drenth2025lpvlfr | Sec. 3.2, p. 3 | "self-scheduled LPV-LFR models, requiring the joint-estimation of LPV model and scheduling map"; the map is an FNN of x, u, d (estimated, unlike a known selection) | 2026-10-09 |
| hoekstra2026lfr | Sec. 5 (well-posedness), Condition 8, Theorem 9 | Condition 8: phi_base and phi_aug are C2; Thm. 9: the augmentation structure is well posed if Conditions 6 (acyclic graph) and 8 hold (a sign term in the baseline would violate Cond. 8; own reading, not stated in the paper) | 2026-10-09 |
