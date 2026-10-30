# Committee read (fresh eyes)

One read of the live thesis sections (2026-10-29) by a subagent without this project's context, as a
committee member reading it for the first time. Items are reader impressions, not checked facts: any
fact used in an answer is checked first (skill rule 1). Strike (`~~ ~~` or delete) what nobody would ask;
the rest feeds `practice`. Items marked (revnote) come from my own `\revnote`s.

## 1. Obvious choices
- [ ] The intro motivates the model for evaluating a changed controller, yet the same $K_1$ generates, trains and evaluates every model. Why no $K_2$ check, even one record? `01_introduction.tex:107`, `06_results.tex:332`
- [ ] Motion profiles barely excite the added mass, so a multisine is added to 14 of 18 training records; the gain shows only on multisine records and on operational profiles the model is slightly worse than the baseline. Is the data designed around the method rather than machine use? `05_setup.tex:266`, `06_results.tex:290`
- [ ] Absorber frequencies placed above crossover to reduce suppression by feedback, and $m_a=0.5\,m_h$ is large. Was the missing dynamics made easy to find? `05_setup.tex:255`, `99_appendix.tex:50`
- [ ] One seed per model; detuned start is 10 % with one fixed sign pattern, so start dependence rests on two starts per model. `05_setup.tex:497`, `05_setup.tex:502`
- [ ] Added states screened over 0, 2 and 8 only, and 2 matches the true single mode. Were 1, 3 or 4 tried, and how would you choose without the truth? `06_results.tex:133`, `99_appendix.tex:711`
- [ ] An exact six-channel LPV-LFR is derived, but the implementation inverts $M(Y)$ directly and the augmentation never uses the LFR. Why derive it? `02_system_baseline.tex:645`
- [ ] Noise-free outputs $y^\circ$ are generated but used only for the noise floor; no noise-free training run separates noise effects from method effects. `05_setup.tex:583`
- [ ] Three velocity reconstructions: encoder estimates positions too, OBC tuples use five-point finite differences, normalisation uses one-step differences. Why? `03_augmentation.tex:1182`, `04_obc.tex:250`, `03_augmentation.tex:523`
- [ ] Overlap $\rho$ reported (0.00 train, 0.24 test, truth 0.59) without a data-derived threshold for "low enough". `06_results.tex:181`
- [ ] In the structure table $\mathcal M_{\mathrm{OBC}}$ starts detuned while the other grey-box models start from the true parameters, so the comparison mixes two factors. `06_results.tex:80`
- [ ] Output map not augmented, and only $f_{\mathrm{aug}}$ on the physical states is projected; the added-state path into $\tilde x$ stays unprojected. `03_augmentation.tex:408`, `06_results.tex:307`

## 2. Out-of-the-box topics
- [ ] Classical closed-loop identification (indirect, joint input-output, two-stage or projection methods, instrumental variables) for the noise-input correlation that blocks a consistency claim. `06_results.tex:337`
- [ ] Lur'e or Wiener-Hammerstein structure with static friction in feedback, or a physical friction model (LuGre, Stribeck, Coulomb as extra base parameter) instead of a network. `05_setup.tex:90`
- [ ] Best linear approximation or nonparametric FRF (local polynomial method) of the baseline residual; its LTI realisation could initialise the added states instead of PS. `03_augmentation.tex:1569`
- [ ] Subspace identification (N4SID, CVA, LPV subspace) on the closed-loop residual, as initialisation or as a stronger black-box comparison. `06_results.tex:324`
- [ ] Practical identifiability: singular values or condition number of $\Phi_\Gamma$, Fisher information, Cramér-Rao bounds, profile likelihood, especially for the damping combinations. `04_obc.tex:263`
- [ ] Optimal input design (D-optimal multisine) aimed at parameter identifiability rather than the absorber band. `05_setup.tex:281`
- [ ] Sensitivity-based model updating with parameter subset selection, or Bayesian/MAP estimation with priors on $\theta_{\mathrm{base}}$, as alternatives to OBC. `04_obc.tex:96`
- [ ] Soft orthogonal regularisation (Györök 2025) or a drift penalty as an explicit comparison arm. `04_obc.tex:184`
- [ ] Continuous-time augmentation (neural ODE, CT SUBNET with CT encoder), since the baseline is CT and data and model rates differ. `03_augmentation.tex:450`
- [ ] Classical validation: residual whiteness, residual-input cross-correlation, FRF error bands, ensemble uncertainty. `05_setup.tex:575`
- [ ] Other structures from Hoekstra's framework (series, LFR feedback) or a $Y$-scheduled augmentation. `03_augmentation.tex:403`

### Added by me, per member (saved as Q-010 to Q-023, grounding outside, answers not drafted yet)
- [ ] Schoukens: PNLSS or Volterra model of the residual instead of a neural augmentation (Q-010)
- [ ] Schoukens: closed-loop validation and the criterion stated beforehand (Q-011)
- [ ] Schoukens: Gaussian-process or Koopman-based models (Q-023)
- [ ] Toth: Y-scheduled LPV augmentation (Q-012)
- [ ] Toth: well-posedness over the Y range and stability of the learned part (Q-013)
- [ ] Toth: uncertainty bounds on the parameter estimates (Q-014)
- [ ] Hoekstra: parallel versus series or feedback LFR augmentation (Q-015)
- [ ] Hoekstra: what you would change in the encoder in hindsight (Q-016)
- [ ] Hoekstra: physics-informed neural networks as an alternative (Q-017)
- [ ] Gerritsen: online or adaptive re-estimation for health monitoring (Q-018)
- [ ] Gerritsen: what it takes to use the real controller code and machine data (Q-019)
- [ ] Gerritsen: transfer to another gantry of the same type (Q-020)
- [ ] Shakib: Lur'e structure with an explicit friction block (Q-021)
- [ ] Shakib: classical model updating instead of augmentation (Q-022)

## 3. How would you continue
- [ ] Changed-controller transfer test; train with several controllers so feedback cannot hide plant-model errors. `06_results.tex:341`
- [ ] Measured ASMPT gantry data: unknown controller implementation, delays, real noise spectrum, rate mismatch. `06_results.tex:341`
- [ ] Extend OBC to the simulated output with a window Jacobian, and project the added-state contribution too. `06_results.tex:342`
- [ ] Several seeds and random detuning directions and magnitudes; report distributions of NRMS$_e$ and parameter errors. `05_setup.tex:502`
- [ ] Friction-only and absorber-only datasets to test the claim that the dampers absorb the friction. `06_results.tex:313`
- [ ] Production-compatible excitation (short multisine bursts in standstill or settling) so operational data alone suffice. `05_setup.tex:270`
- [ ] Root-cause demo: change one physical parameter in the simulated plant and check the estimates track it. `01_introduction.tex:149`
- [ ] Match model and controller rates (controller rate, or ZOH controller) and quantify the residual force term. `03_augmentation.tex:930`
- [ ] A settling or tracking metric ASMPT uses, beyond NRMS$_e$ on the servo error. `05_setup.tex:575`

## 4. Hard to follow
- [ ] Normalised equations: why $f_{\mathrm{base}}^{\mathrm n}$ subtracts $\mu_x$ while $f_{\mathrm{aug}}$ is scaled only by $T_x$, and how position means tie to $P^{-\top}\mu_y$; what does the network output? `03_augmentation.tex:599`, `03_augmentation.tex:631`
- [ ] Controller-state difference $x^c$: "simulated with respect to $r$, but $r$ and $u^{\mathrm{ff}}$ are not used"; the rate-mismatch force term is explained only later. `03_augmentation.tex:843`, `03_augmentation.tex:930`
- [ ] PS: which optimiser updates which parameters when; $\theta^a_{\mathrm{enc}}$ updated by both $V$ and S1; left-out records still in $V$; S1/S2 stop rules only in a table. `03_augmentation.tex:1836`, `03_augmentation.tex:1814`
- [ ] OBC mechanics: $\theta_{\mathrm{aux}}$ solved over all tuples per evaluation of $V$ but applied at the simulated state; $\beta_{\mathrm{lin}}$ per epoch; the $\gamma$ column; size of $N$ and cost per step. `04_obc.tex:296`, `04_obc.tex:344`
- [ ] Overlap $\rho$ uses $\Phi$, projection uses $\Phi_\Gamma$; which $Z$ gives network overlap and which truth overlap is only implied. `04_obc.tex:451`, `06_results.tex:182`
- [ ] Recovery result $\hat\beta$ reads as the trained estimate but is never computed and holds only for the one-step error (revnote). `04_obc.tex:395`
- [ ] Interpretability is defined as identifiable combinations estimated uniquely, but results are judged against true values and start dependence. Which criterion decides? `01_introduction.tex:145`, `06_results.tex:315`
- [ ] Per-record weighting $W_r$ by servo-error std: why per record, and how it interacts with the metric. `03_augmentation.tex:1149`
- [ ] Detuned start "10 % larger or smaller ... same signs" does not state the pattern. `05_setup.tex:497`
- [ ] "Far above the noise floor" without the floor value next to the model error. `06_results.tex:294`
- [ ] Data pipeline (20 kHz sim, 4 kHz model, zero-phase Butterworth, force averaging) and its effect on the hold model are scattered over Section V and the appendix. `05_setup.tex:325`

## 5. Backup-slide candidates
- [ ] Changed-controller motivation versus evidence: what a $K_2$ test would show, why it is missing.
- [ ] Excitation spectra: motion profiles versus multisine against absorber band and crossover.
- [ ] Added states 0, 2, 8 with and without PS: validation curves, choosing $n_{\bar x}$ without the truth.
- [ ] Velocity reconstructions side by side (encoder, one-step, five-point) and where each is used.
- [ ] Closed loop with the $x^c$ difference, step by step, with the rate-mismatch force term.
- [ ] PS timeline: parameters and optimisers active in S1, S2 and hand-over.
- [ ] OBC in one picture: tuples at $\bar x=0$, $\Phi_\Gamma$, projection, per-step subtraction, $\beta_{\mathrm{lin}}$ updates.
- [ ] Network overlap versus truth overlap definitions with 0.00, 0.24, 0.59.
- [ ] Per-parameter estimate errors from both starts, damping versus friction.
- [ ] Simpler alternatives versus what was built: physical absorber, soft orthogonal penalty, open loop with recorded input.
- [ ] Singular values of $\Phi_\Gamma$ as a practical identifiability check.

## 6. Effort vs a simpler alternative
- [ ] Network plus added states for the absorber. Simpler: extend the physics baseline with one mass, spring and damper and estimate them. `05_setup.tex:95`
- [ ] PS two-stage initialisation with decoder, left-out records, heuristic stops. Simpler: LTI subspace or BLA fit of the residual, or more seeds and longer windows. `03_augmentation.tex:1569`
- [ ] OBC extension (Jacobian, offset column, projection per evaluation of $V$). Simpler: soft orthogonal penalty, drift penalty, or estimate $\theta_{\mathrm{base}}$ first and freeze it (Kessels); OBC does not make estimates start-independent anyway. `04_obc.tex:109`, `06_results.tex:315`
- [ ] Controller-difference $x^c$ with $x^c_\tau=0$. Simpler: full closed loop with $r$, $u^{\mathrm{ff}}$ and controller state from record start, or open loop over short windows. `03_augmentation.tex:838`
- [ ] Six-channel LFR derivation and well-posedness proof while the code uses $M(Y)^{-1}$; Sylvester's criterion alone would suffice. `99_appendix.tex:75`
- [ ] Log plus tanh-bounded parametrisation. Simpler: box bounds or projected gradient, given 10 % detuning. `02_system_baseline.tex:865`
- [ ] L-BFGS polish that barely changed validation NRMS$_e$ (revnote). Simpler: stop after Adam. `04_obc.tex:365`
- [ ] Encoder estimating measured positions with its own least-squares fit. Simpler: $q_\tau$ from $y_\tau$, finite-difference velocities, encoder only for $\bar x$. `03_augmentation.tex:1182`
- [ ] 16-state SUBNET black box from random init. Simpler and stronger reference: LTI closed-loop ARX or subspace model, or BLA-initialised black box. `06_results.tex:324`

## 7. Choices I might not make again
- [ ] Tustin instead of ZOH for the controller (revnote). `03_augmentation.tex:883`
- [ ] Model at 4 kHz, data at 20 kHz with Butterworth; validating at 20 kHz then illogical; discrete controller shown in CT form (revnote). `03_augmentation.tex:886`, `03_augmentation.tex:890`
- [ ] LPV versus LTI aspect dropped to an appendix (revnote). `01_introduction.tex:445`
- [ ] NRMS$_e$ on servo error instead of the research plan's BFR (revnote). `05_setup.tex:577`
- [ ] Keeping the never-computed first-order estimate $\hat\beta$, off by an order of magnitude as a number (revnote). `04_obc.tex:425`
- [ ] PS section detail; own note asks to simplify and move to appendix (revnote). `03_augmentation.tex:1533`
- [ ] Negation removed from the OBC motivation because never measured, leaving the original argument untested (revnote). `04_obc.tex:230`
- [ ] Running an L-BFGS polish that did not change the models (revnote). `04_obc.tex:365`
- [ ] Garcia's parameters rather than ASMPT-specific values for the simulated plant (revnote). `02_system_baseline.tex:145`
- [ ] Notation clashes: $d$ as parameter and PS-figure symbol, $\theta$ for baseline and network (revnote). `03_augmentation.tex:1642`, `03_augmentation.tex:394`
- [ ] Noise-free twins generated but never trained on (revnote). `99_appendix.tex:563`
- [ ] Only 4 of 18 training records without multisine; one seed per model. `06_results.tex:292`, `99_appendix.tex:725`
