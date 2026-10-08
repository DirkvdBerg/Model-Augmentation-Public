# L-A: training methods that make added latent states learn lightly damped modes

Date: 2026-10-01. Sub-question L-A only. Seeds already read by the project (Ribeiro 2020, Hoekstra 2026, Beintema 2023, Masti and Bemporad 2021, Tang 2023, Tian 2021, Emami 2021, Zucchet and Orvieto 2024) are cited, not redone.

## Frame (step 0)

- Sub-question: which objective terms / training schemes (no parameterisation, no pole prior, no special init) make unconstrained latent states learn near-unit-circle complex modes.
- Entry points: Koopman multi-step and bidirectional losses (Otto/Rowley, Azencot), self-predictive RL (Ni 2024), free-latent-trajectory / multiple shooting (Ouala, Forgione/Piga, Bemporad), smoother/DA in the loop (Demirkaya, Brajard), predictive-state and CCA targets (Hefny, Kostic).
- Disqualify: remedies that are a parameterisation or initialisation change.
- Vocabularies: system identification, ML representation learning (RL/JEPA), Koopman/fluids, data assimilation.
- Held locally before this run: Ouala 2020 (`literature/augmented-states/`), Hafner 2019, Hess 2023 GTF, FranSys (`literature/multiple-shooting/`).

## Findings, ranked by fit to the constraints

Constraint reading: "fixed zero-init MLP state map" = the model class and architecture stay unchanged; added objective terms, auxiliary training-only networks or per-sample decision variables are allowed.

### 1. Free latent trajectory + weak-constraint consistency (closest positive evidence)
**Ouala, Nguyen, Drumetz, Chapron, Pascual, Collard, Gaultier, Fablet**, "Learning latent dynamics for partially observed chaotic systems", *Chaos* 30(10):103121, 2020, DOI 10.1063/5.0019309 (held, arXiv version; eq. numbers from it).
- Objective Eq. (6), p4: min over theta and the unobserved latent sequence {y_t}: sum ||x_t - G(Phi_theta(X_{t-1}))||^2 + lambda ||X_t - Phi_theta(X_{t-1})||^2, X_t = [x_t, y_t]. Test-time latent inference Eq. (7), p4.
- Oscillatory result: p5, only the real part of z' = alpha z, alpha = -0.1 - 0.5j (damping ratio about 0.20) is observed; the augmented latent recovers the hidden quadrature and forecast MSE < 1e-5.
- Compatibility: yes. Objective-only, the latent trajectory is a decision variable, not produced by the zero-init map, so the map is fitted to a latent that already oscillates. A SUBNET encoder can replace Eq. (7) after training.
- Failure modes: noise-free data only; the per-sample latents can absorb measurement noise when lambda is small; p5 warns about local minima and limit-cycle basins.

**Forgione and Piga**, "Model structures and fitting criteria for system identification with neural networks", IEEE AICT 2020, pp. 1-6, DOI 10.1109/AICT50176.2020.9368834 (arXiv 1911.13034, downloaded). Eq. (16), p11: alpha * (sim vs data) + (1-alpha) * (sim vs free hidden sequence). p10 (Sec. 4.2): for state-space models a free hidden state X is optimised. Caveat: when the state is not fully observed, X "cannot be initialized directly with measurements". The text argues measurement noise then never enters the simulation (p9). No oscillatory latent reported.

**Bemporad**, "Recurrent neural network training with convex loss and regularization functions by extended Kalman filtering", IEEE TAC 68(9):5661-5668, 2023, DOI 10.1109/TAC.2022.3222750 (arXiv 2111.02673, downloaded). Non-condensed (multiple shooting) Eq. (6), p6: free x_1..x_{N-1} with penalty gamma/2N sum ||x_{k+1} - f_x(x_k,u_k)||^2. EKF joint state/weight estimation Eq. (10) is the observer-in-the-loop variant. Results: fluid damper only, no learned mode reported.

### 2. Multi-step latent self-prediction against encoded future data
**Otto and Rowley**, "Linearly recurrent autoencoder networks for learning dynamics", SIAM J. Appl. Dyn. Syst. 18(1):558-593, 2019, DOI 10.1137/18M1177846 (arXiv 1712.01378, downloaded).
- Objective Eq. (25), p12: decay-weighted (delta^tau) relative errors of reconstruction AND latent evolution ||z_hat_{t+tau} - z_{t+tau}||^2 / (||z_{t+tau}||^2 + eps), tau = 1..T-1, with z_hat_{t+tau} = K^tau z_t and target z_{t+tau} = encoder(x_{t+tau}).
- Oscillatory result: p27, cylinder wake. The unconstrained K learns lambda_1 of about 0.002 + 0.845i (continuous time, essentially undamped), matching KDMD.
- Compatibility: yes as an added term. Use rollout F^tau(x_hat(k)) versus encoder(window at k+tau) for the x_a block, over many tau rather than tau = 1. Normalising by the target norm makes the collapsed solution non-trivial to reach.
- Failure modes: autonomous system, noise-free, 400-dim delay embedding. The Eq. (29) "known eigenvalue block" variant is a parameterisation and is excluded.

**Ni, Eysenbach, SeyedSalehi, Ma, Gehring, Mahajan, Bacon**, "Bridging state and history representations: understanding self-predictive RL", ICLR 2024, no DOI (arXiv 2401.08898, downloaded).
- ZP loss Eq. (2), p5: ||g_theta(f_phi(h), a) - f_phi~(h')||^2. Theorem 3, p6: a stop-gradient (detached or EMA) TARGET keeps phi^T phi constant, so there is no collapse. Online targets lack this guarantee (Prop. 3).
- Check against the project's variant: in Ni, gradient flows through the encoder on the input side (time k), and only the target side (k+1) is detached. Detaching the encoder states on the input side, as described in the problem, is a different scheme with no guarantee.
- No oscillatory result (RL benchmarks).

**Ruiz-Morales, Vanoost, Pissoort, Verbeke**, "Koopman invariants as drivers of emergent time-series clustering in JEPAs", AAAI 2026 (arXiv 2511.09783, downloaded).
- Failure-mode evidence. Theorem 3.4, p4: the idealised JEPA self-prediction loss is minimised by eigenvalue-1 Koopman invariants when the predictor is near-identity.
- This is a published mechanism for self-prediction preferring static or slow latents. That matches the observed static or real-lag x_a by analogy only. The predictor class here differs (zero-init MLP, not near-identity), so do not claim it as the established cause.

**Ulmen, Sundaram, Goerges**, "Learning state-space models of dynamic systems from arbitrary data using JEPAs", IFAC-PapersOnLine 59(18):19-24, 2025, DOI 10.1016/j.ifacol.2025.10.190 (arXiv 2508.10489, downloaded).
- Total loss Eq. (7), p3: the VICReg variance term Eq. (9) and covariance term Eq. (11) are anti-collapse terms on latent states.
- Data: closed-loop (PID) pendulum. Results are qualitative only, with no pole or damping report.
- Compatible as added terms on x_a statistics.

### 3. Bidirectional (forward + backward) prediction with consistency
**Azencot, Erichson, Lin, Mahoney**, "Forecasting sequential data using consistent Koopman autoencoders", ICML 2020, PMLR 119, no DOI (arXiv 2003.02236, downloaded).
- Eq. (9) forward and Eq. (10) backward multi-step losses (lambda_s = 8), consistency Eq. (12), total Eq. (13), all p4-5. p5 states that Eq. (12) "promot[es] the eigenvalues to get closer to the unit circle" (Fig. 4).
- Compatibility: partial. It needs a training-only backward map. For a nonlinear F the analogue is a backward net psi with ||psi(F(x)) - x||^2, which is not validated in the paper.
- Failure mode: the push toward the unit circle acts on ALL modes, including well-damped physical ones. It is a soft reversibility bias, close to a pole prior; flag it before use.

### 4. Smoother / DA in the loop, then multi-horizon fit to the estimated latents
**Demirkaya, Stratis, Imbiriba, Danziger, Erdogmus**, "RTS smoother-guided learning of physics-based neural differential models", arXiv 2607.15180, 2026 (downloaded).
- One-step surrogate Eq. (10), p7. Multi-horizon state + output loss weighted by the smoother covariance Eq. (14)-(16), p8, with alternating smoother/parameter steps (Sec. 3.4). Fig. 2, p8: horizon 1 alone drifts.
- Oscillatory result: the unmeasured velocity of an undamped oscillator is recovered (Table 1, p15).
- Incompatible at zero init: the latent is coupled to the output by KNOWN kinematics, q' = v (Eq. 28, p23). With a zero-init MLP, x_a has no output coupling, so a smoother returns the prior and gives no target.

**Brajard, Carrassi, Bocquet, Bertino**, "Combining data assimilation and machine learning to emulate a dynamical model from sparse and noisy observations", J. Comput. Sci. 44:101171, 2020, DOI 10.1016/j.jocs.2020.101171 (arXiv 2001.01520, downloaded).
- Multi-step loss on analysed states Eq. (6), p4.
- p6: initialisation is critical and convergence is not guaranteed. Abstract (p2): skill "drops abruptly if less than half of the model domain is observed". This is the same observability limitation.

### 5. Latent-target construction from past/future statistics (noise-bias side)
**Hefny, Downey, Gordon**, "Supervised learning for dynamical system learning", NeurIPS 2015, no DOI (arXiv 1505.05310, downloaded).
- Predictive state q_t = E[future features | history]. W is fitted through the instrument moment condition Eq. (2), p3, using history as the instrument.
- p3: naive regression between overlapping noisy future windows "will give a biased estimate".
- Relevance: the innovation head (x_a predicts the next 40 output errors) is a predictive-state target without this instrumental step. Under measurement noise its target is biased toward noise-driven (fast, heavily damped) structure. This is consistent with radius 0.25, but only as an argument, not tested.
- Uncontrolled system; no damping result read.

**Kostic, Novelli, Grazzi, Lounici, Pontil**, "Learning invariant representations of time-homogeneous stochastic dynamical systems" (DPNets), ICLR 2024 (arXiv 2307.09912, downloaded). Score Eq. (7) and relaxed score Eq. (9), p5. Theorem 1 / Eq. (10), p5: the maximisers are the leading r singular functions of the transfer operator. A radius-0.986 pair has large lag-1 singular values, but the gantry's free X/Y integrators and constants would rank first. The setting is autonomous and has no exogenous input. Weak fit.

### Excluded (parameterisation or initialisation remedy, or off-question)
- Lusch, Kutz, Brunton, Nat. Commun. 9:4950, 2018, DOI 10.1038/s41467-018-07210-0: oscillation obtained through an auxiliary network parameterising the eigenvalue frequency (p1, p4); parameterisation.
- Otto and Rowley Eq. (29): fixed known-eigenvalue block; parameterisation (Eq. 25 kept above).
- Noel/Schoukens grey-box PNLSS (held): BLA initialisation.
- Gedon, Wahlstroem, Schoen, Ljung, IFAC-PapersOnLine 54(7):481-486, 2021, DOI 10.1016/j.ifacol.2021.08.406: VAE-RNN ELBO (Eq. 5-6, p3), no latent-mode result; not excluded but no evidence for this question.

### Synthesis (my reading, not a paper claim)
In every positive case, the latent target exists independently of the current (zero) state map:
- free latent trajectories (Ouala, with Forgione and Bemporad as the control-side form), or
- encoder-of-future targets over many steps with collapse protection (Otto/Rowley normalisation, Ni stop-gradient on the target).

The two tried terms each miss one of these ingredients:
- The one-step detached equation error has horizon 1 (Demirkaya Fig. 2: insufficient) and detaches the input side rather than the target side.
- The innovation head is a predictive-state target without the instrumental-variable debiasing.

Observer/smoother targets need the latent to be output-coupled first, which a zero-init map does not provide.

No paper found reports a lightly damped latent pair (damping below 0.05) learned under input-driven, closed-loop, noisy data with an unconstrained map. The positives are autonomous and noise-free (Otto/Rowley, Ouala), or have the latent coupled to the output by known physics (Demirkaya). This negative is provisional: arXiv abstract counts were 0 for `"lightly damped" AND "neural network" AND "identification"` and 0 for `"model error" AND "augmented" AND "hidden states" AND oscillat`. Google Scholar (2 queries) found Demirkaya only.

## Access status
TU/e browser access: NOT TESTED (subagent rule: no browser preflight). Every paper above was read through arXiv or local files, so none needed the route. Nothing was marked unreachable.

## Evidence quality
- Full-text targeted reads (grep plus page reads of the cited pages): Otto/Rowley, Azencot, Ni, Hefny, Forgione/Piga, Bemporad, Ouala, Demirkaya, Brajard, Ulmen, Ruiz-Morales, Kostic.
- Partial reads: Lusch (abstract and p4 only); Gedon (pp. 1-3 only).
- Equation numbers come from the arXiv versions, not the journal Versions of Record.
- Metadata checked against Crossref for every DOI listed.

## Research Log
- Queries run: arXiv API 14 (hits: consistent Koopman 2/2 on target; self-predictive collapse 2, 1 on target; JEPA dynamical 88, 2 on target; titles resolved 7/7; zeros: hybrid-DA-latent, lightly-damped-NN-ID, model-error-hidden-oscillat, spectral-loss-sysid, frequency-domain neural SS, VAMP non-reversible). Google Scholar 2 (1 sole-source on-target find: Demirkaya 2026). Crossref 7 DOI lookups plus 2 title queries. dblp 0, OpenAlex 0.
- What worked: arXiv `ti:` for known titles; `abs:` with 3 quoted concepts. Scholar as an unquoted-plus-one-phrase sentence found the hybrid smoother paper that no enumeration reached.
- What failed: the first Koopman multi-step query was too generic (speech, quantum). Scholar on "lightly damped" returned power-system and SHM noise.
- Dead ends: Gedon 2021 (no latent-mode result); Lusch (parameterisation).
- Coverage gaps: no citation-graph traversal (OpenAlex unused); controlled predictive-state follow-ups (Hefny 2018), deep Kalman filters (Krishnan) and structured inference networks not read; CDC/ECC venue sweep not done.
- Suggested skill fix: add to step 0 that "added-state / augmentation" questions must check whether a candidate's latent is OUTPUT-COUPLED by known physics in its experiments. Demirkaya looked like a direct hit until its appendix (Eq. 28) showed the latent was observable by construction. The compatibility verdict hinged on that appendix, not on the method section.
