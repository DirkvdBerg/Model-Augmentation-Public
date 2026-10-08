# L-B: equation-error bias with noisy / estimated regressors, remedies, symmetric predictors

Sub-question L-B of the added-states sweep (2026-10-01). PDFs saved in this folder are named below.
Seeds cited, not redone: Shynk 1989; Steiglitz and McBride 1965; Tian, Chen, Ganguli ICML 2021;
Tang et al. ICML 2023 (arXiv:2212.03319); Diniz, Cousseau, Antoniou 1994.

## Frame (step 0)

- Sub-questions: (1) direction and size of EE/LS bias of a transition matrix when regressors are
  noisy or estimated, and its sampling-rate scaling; (2) remedies that keep the parameterisation
  free (IV, d-step EE, TLS, bias compensation, prefiltering) and their learned-model analogues;
  (3) follow-ups to Tian 2021 that give a mechanism for non-symmetric (rotational) predictors.
- Seeds: as above, plus Dawson et al. 2016 and Hemati et al. 2017 named in the brief.
- Entry points: Söderström author corpus (OpenAlex), DMD noise literature (arXiv), self-predictive
  RL (arXiv abs search), closed-loop IV (Gilson, Van den Hof on HAL), RIV (Young).
- Disqualifies: methods that fix the pole location or parameterise A_aa (rotation blocks, pole
  implanting), per project memory `feedback_no_pole_init_workaround`.
- Out of scope: output-error gating at zero init (already covered by Steiglitz-McBride / Shynk in
  `scripts/gantry/absorber-learning-diagnosis/CAUSE-AND-DIRECTION.md`).
- Vocabularies: control (errors-in-variables, IV, bias compensation), signal processing (AR plus
  noise, high-order Yule-Walker), fluid dynamics / applied maths (DMD de-biasing, TLS), ML
  (self-predictive learning, stop-gradient, two-stage instrumental regression), molecular
  dynamics (reversible vs non-reversible Koopman estimation, VAMP).
- Already held locally: nothing on EIV/DMD bias. The project already plans an IV check (AA1 in
  `scripts/gantry/absorber-learning-diagnosis/DECISIONS.md`), which this sweep supports.

## Findings

### 1. Bias direction and size

**F1. Dawson, Hemati, Williams, Rowley (2016).** "Characterizing and correcting for the effect of
sensor noise in the dynamic mode decomposition", *Experiments in Fluids* 57(3):42.
DOI 10.1007/s00348-016-2127-7. Free: arXiv:1507.02264 (`dawson2016_sensor-noise-dmd.pdf`). Read in full.
- LS one-step fit A = Y X^+ with noisy X: E(A_m) = A (I - E(N_X N_X^*)(X X^*)^{-1}), Eq. (7), p.5
  (POD-space version Eq. (6)). Noise in Y does not bias; noise in the regressor X does.
- Direction: "sensor noise has the effect of shifting the computed DMD eigenvalues to appear to be
  more stable than they actually are (i.e., moving them further inside the unit circle)", p.7
  (argued in their Appendix 1). Bias is proportional to sigma_N^2 and does not shrink with the
  number of samples m (p.11).
- Low-amplitude modes are hit hardest: for a hidden decaying mode "DMD ... estimates a decay rate
  that is almost double the true value" (p.15); frequencies are much less affected than growth
  rates (p.20).
- Applies to the zero-init MLP: yes, to its local linearisation A_aa (the analysis is for a linear
  regression on noisy regressors; our x_a has small amplitude relative to encoder noise, exactly
  the worst case named on p.15 and p.20).

**F2. Kay (1979)**, "The effects of noise on the autoregressive spectral estimator", *IEEE TASSP*
27(5):478-485, DOI 10.1109/TASSP.1979.1163275. Not read (closed); cited second-hand by F3, p.1:
"the estimated AR poles are biased toward the center of the unit circle, leading thus to a
smoothed spectrum". Needs browser route for the equation.

**F3. Soverini, Söderström**, "Frequency domain identification of autoregressive models in the
presence of additive noise", Uppsala DiVA preprint diva2:1861717 (2015/2016; journal venue not
resolved). `soderstrom2016_freq-domain-AR-additive-noise.pdf`. Pages 1-2 and remarks read.
- p.1: AR poles biased toward the centre (citing Kay 1979); remedies listed: ARMA/PEM, high-order
  Yule-Walker (HOYW), bias-compensated LS (BCLS: Sakai and Arase 1979; Zheng 2005), Frisch scheme.
- p.5, Remark 5: their HOYW-type equations are "analogue to an instrumental variable (IV) method in
  the time domain, where delayed outputs are used as instruments", solvable by TLS.
- Applies: yes for the linear part; it is the scalar/AR version of our case.

**F4. Söderström (2007)**, "Errors-in-variables methods in system identification", *Automatica*
43(6):939-958, DOI 10.1016/j.automatica.2006.11.025. Not read: closed, no OA copy found
(`needs-browser-route`). Its content is restated with attribution in F5.

**F5. Söderström, Soverini (2023)**, "Aspects on errors-in-variables identification: Some ways to
mitigate a large bias", *IFAC-PapersOnLine* 56(2):4019-4024, DOI 10.1016/j.ifacol.2023.10.1383.
Free: DiVA (`soderstrom2023_eiv-mitigate-large-bias_IFAC.pdf`). Read in full.
- PEM/LS bias approximation theta_tilde ~ -[V''(theta*)]^{-1} V'(theta*), Eq. (10), p.2: bias is
  large when the Hessian is near singular (near pole-zero cancellation or weak excitation).
- IV estimate theta_hat = (R_zphi^T R_zphi)^{-1} R_zphi^T r_zy, Eqs. (18)-(19), p.2; bias
  theta_tilde = -R_zphi^dagger r_{z phi_tilde} theta_0, Eq. (20), p.3: zero when instruments are
  uncorrelated with regressor noise.
- Stated conditions (p.2-3): persistence of excitation, coprimeness, noise signals mutually
  uncorrelated, z uncorrelated with output noise ("delayed and/or filtered values of the input"),
  N to infinity, and **open-loop operation** (p.1).
- Applies: linear part only; the open-loop condition does not hold for the gantry data (see F9).

**Sampling-rate scaling (no direct full-text source reached).** The CT-AR fast-sampling results
(Söderström, Fan, Carlsson, Bigi, IEEE TAC 42(5), 1997, DOI 10.1109/9.580871; Larsson, Mossberg,
Söderström, CSSP 2006, DOI 10.1007/s00034-004-0423-6; Larsson and Söderström, Automatica 2002) are
all closed (`needs-browser-route`); Scholar snippets only say LS "can lead to a very large bias" as
h tends to zero. What can be stated from F1 without them (derivation, not literature): the
discrete radius factor is set by the noise-to-state covariance ratio, independent of T, but the
implied CT damping is -ln(r)/T, so a fixed discrete shrink costs 1/T in CT damping.
Check on the reported numbers (my arithmetic): true pole 0.986 at 2*pi*212/4000 = 0.333 rad. An
estimate with r = 0.25 to 0.3 and damping ratio 0.99 has angle 0.17 to 0.20 rad. So the learned
pole lost about half its angle as well as most of its radius. A pure radial shrink at fixed angle
0.333 rad would give a damping ratio of about 0.96 to 0.97, close to but below what you observed.
Note on hypothesis (c): 0.333 rad per step is about 19 samples per period. That is moderate, not
extreme, oversampling, so the near-identity argument alone is weak here.

### 2. Remedies that keep the parameterisation free

**F6. Instrumental variables with delayed regressors (HOYW / basic IV).** Estimator: F5 Eq. (18).
For our setup (derivation): with instruments z(k) = x_hat(k-d) (plus u lags), the probability
limit is A_IV = C_{d+1} C_d^{-1}, where C_j = E[x_hat(k+j) z(k)^T]. This is consistent if the
equation residual is uncorrelated with z.
**Key condition specific to our encoder:** x_hat(k) is a 17-tap FIR of noisy y, so the errors in
x_hat(k) and x_hat(k+1) share 16 samples of noise. Hence d must be at least 17 (one encoder window
plus the residual's memory), not 1. This is exactly the failure Hefny et al. name (F10, p.3).
Applies to the MLP: as an extra moment term on the local linear map. For a nonlinear F_a, plain
IV is not consistent in general; use the two-stage form of F10.

**F7. Bias compensation / noise-corrected regression.** Dawson et al. ncDMD (F1):
A ~ A_m (I - m sigma_N^2 (X_m X_m^*)^{-1})^{-1}, Eq. (12), p.7. This needs the regressor noise
covariance. Zheng (2002), "A bias correction method for identification of linear dynamic
errors-in-variables models", *IEEE TAC* 47(7):1142-1147, DOI 10.1109/tac.2002.800661, is closed
(`needs-browser-route`).
Relevance (derivation): our regressor noise covariance S_0 = sigma^2 sum_i h_i h_i^T is computable
from the encoder FIR weights h and a data-derived noise floor, so it needs no oracle.
Applies: to the linear part.

**F8. Forward-backward and TLS.**
- fbDMD (F1): A ~ (A_m A_m^back)^{1/2}, Eqs. (13)-(15), p.7 (backward fit inverted, geometric
  mean); CT form A_c ~ (A_c,m + A_c,m^back)/2, p.8.
- Hemati, Rowley, Deem, Cattafesta (2017), "De-biasing the dynamic mode decomposition for applied
  Koopman spectral analysis of noisy datasets", *Theor. Comput. Fluid Dyn.* 31(4):349-368,
  DOI 10.1007/s00162-017-0432-2. Free: arXiv:1502.03854 (`hemati2017_debiasing-dmd-tls.pdf`).
  TLS problem min ||[dX dY]||_F s.t. Y + dY = A(X + dX), Eq. (6), p.3. Algorithm: project X and Y
  onto the leading right singular vectors of Z = [X; Y], then fit DMD (p.4). Consistent as the
  number of snapshots goes to infinity "under certain assumptions" (p.3). The paper warns that TLS
  is less numerically stable (p.5).
- Both assume noise of equal size, independent across snapshots. Our encoder errors are correlated
  across lags (F6), so this assumption fails unless the samples are thinned by at least 17.
- Applies: linear only; no MLP analogue found.

**F9. Closed-loop IV and refined IV (prefiltering, Steiglitz-McBride family).**
- Gilson, Van den Hof (2005), "Instrumental variable methods for closed-loop system
  identification", *Automatica* 41(2):241-249, DOI 10.1016/j.automatica.2004.09.016. Free: HAL
  hal-00089684 (`gilson-vandenhof2005_iv-closed-loop_hal.pdf`), pp.3-4 read.
  - Basic IV: Eq. (11); consistent iff E[z phi^T] is non-singular and E[z v_0] = 0 (p.4).
  - Extended prefiltered IV: Eq. (13).
  - Closed-loop choice: "instruments vector is composed of delayed samples of the reference
    signal" (p.4). This is the safe instrument for our closed-loop logs.
  - Noise model need not be correct.
- Young (2015), "Refined instrumental variable estimation: Maximum likelihood optimization of a
  unified Box-Jenkins model", *Automatica* 52:35-46, DOI 10.1016/j.automatica.2014.10.126.
  Free: ANU repository (`young2015_refined-iv-unified-bj.pdf`; pdf pages cited).
  - RIV: rho_hat = [sum phi_hat_f phi_f^T]^{-1} [sum phi_hat_f y_f], Eq. (28), p.5.
  - The instrument is the noise-free auxiliary-model output x_hat = (B/A) u, Eq. (35), p.6.
  - Data and instruments are prefiltered by the current C/(D A), p.7. This is iterated, which is
    the IV relative of Steiglitz-McBride.
- Applies: the auxiliary-model idea maps onto using the model's own simulated x_a as the
  instrument. That is legitimate only for the linear part, and only after the path has opened.

**F10. Learned/latent-state analogue: two-stage instrumental regression.** Hefny, Downey, Gordon
(2015), "Supervised Learning for Dynamical System Learning", NeurIPS 28. No DOI; arXiv:1505.05310
(`hefny2015_supervised-learning-dynamical-system-learning.pdf`), pp.1-5 and 8 read.
- p.3: because future and shifted-future observation windows overlap, "the noise terms ... are
  correlated. So, naive linear regression will give a biased estimate of W". The fix is past
  observations, which do not overlap, as instruments.
- Stage 1 (any nonlinear regressor) denoises E[x|h], E[x'|h]; stage 2 fits the dynamics between the
  denoised quantities (Fig. 2, p.2). Consistency bounds are in Sec. 4, p.5.
- Applies: closest analogue found to our encoder plus one-step EE. Stage 1 may be nonlinear, but
  the transition being fitted in their theory is a linear map W. A zero-init MLP is covered through
  its linearisation only.

**F11. Whole-trajectory (output-error) fitting as the de-biasing remedy.** Askham, Kutz (2018),
"Variable projection methods for an optimized dynamic mode decomposition", *SIAM J. Appl. Dyn.
Syst.* 17(1):380-416, DOI 10.1137/M1124176. Free: arXiv:1704.02343
(`askham-kutz2018_optimized-dmd-varpro.pdf`). The bias of pairwise DMD comes from treating data
"snapshot to snapshot rather than as a whole, and favor[ing] one direction (forward in time)". The
optimized DMD, which fits exponentials to all snapshots at once, "avoids much of the bias" (p.2).
Applies: this is the project's simulation loss. The EE term is the biased step, not the remedy.

**d-step EE (derivation, no source found).** Least squares of x_hat(k+d) on x_hat(k) with d of 17
or more removes the shared-window term. It does not remove the attenuation from regressor noise:
A^d_hat = C_d (C_0 + S_0)^{-1} still has S_0. The d-th root shrinks the per-step radius error, but
does not remove it. IV (F6) removes both.

### 3. Self-predictive learning: symmetric vs rotational predictors

- **Khetarpal et al. (2024)**, "A Unifying Framework for Action-Conditional Self-Predictive
  Reinforcement Learning", arXiv:2406.02035 (`khetarpal2024_action-conditional-self-predictive.pdf`).
  Restates Tang et al.'s Assumption 3 "Symmetric Dynamics" (p.3) and keeps it for its theorems.
  Fig. 6 shows eigenvectors are learned empirically for non-symmetric MDPs too (p.4). No mechanism
  for rotation. Not applicable as a remedy.
- **Ni et al. (2024)**, "Bridging State and History Representations: Understanding Self-Predictive
  RL", ICLR 2024, arXiv:2401.08898 (`ni2024_bridging-self-predictive-rl.pdf`).
  - Prop. 3 (p.6): ℓ2 with stop-gradient targets has stationary points satisfying the expected ZP
    condition.
  - Thm. 3 (p.6) assumes a linear encoder on a k-truncated history (our FIR encoder is exactly
    this), a linear latent transition, and theta at its LS stationary point. Then phi^T phi
    "will retain its initial value".
  - Implication (inference): the detached EE alone neither collapses nor grows the encoder.
    theta is then the plain LS (attenuated) estimate of F1. No rotation mechanism.
- **Wu, Noé (2020)**, "Variational Approach for Learning Markov Processes from Time Series Data",
  *J. Nonlinear Sci.* 30(1):23-66, DOI 10.1007/s00332-019-09567-y. Free: arXiv:1707.04659
  (`wu-noe2020_vamp-nonreversible.pdf`).
  - "If the dynamics obey detailed balance, those eigenvalues are real-valued", and the reversible
    variational approach applies there (p.4).
  - For non-reversible dynamics, VAMP uses the SVD of the Koopman operator with separate left and
    right feature maps f, g. This is CCA between x_t and x_{t+tau} (pp.5-6) and is "valid for both
    reversible and nonreversible processes" (p.2).
  - Mechanism without constraining A: rotation is carried by the antisymmetric part of the
    time-lagged cross-covariance C_01. A symmetric objective (shared encoder, Rayleigh-type score)
    implicitly assumes reversibility. This is the cross-field counterpart of Tang et al.'s
    bidirectional fix (seed).
  - Applies: as a diagnosis lens. Check whether the EE estimate is dominated by C_00-like
    (symmetric) terms. VAMP itself is a linear-in-features estimator.
- **Hypothesis for the observed near-symmetry (derivation, testable without training):**
  - With encoder error covariances S_0 = E[n(k) n(k)^T] and S_1 = E[n(k+1) n(k)^T] (both
    computable from FIR weights and the noise floor), the one-step LS gives
    A_hat = (C_1 + S_1)(C_0 + S_0)^{-1}.
  - When x_a is weak relative to encoder noise (F1 p.15), A_hat tends to S_1 S_0^{-1}. That is the
    one-step predictor of the filtered noise, which has no 212 Hz rotation. This would explain a
    small, nearly real, nearly symmetric A_aa without invoking Tian's weight-decay mechanism.
  - Check: compute S_1 S_0^{-1} from the trained encoder weights and compare its eigenvalues with
    the learned A_aa.

## Access status

TU/e browser access: UNAVAILABLE (not attempted; per skill the browser preflight belongs to the
parent, so layer 1 and layer 2 are untested). Items marked "unreachable - browser route
unavailable":
- Söderström 2007 Automatica
- Zheng 2002 TAC
- Kay 1979 TASSP
- Söderström and Stoica 2002 CSSP (IV methods)
- Söderström et al. 1997 TAC
- Larsson, Mossberg, Söderström 2006 CSSP
- Garnier 2015 EJC (HAL metadata only)

Sci-Hub was deliberately not used.

## Evidence quality

- Read in full or in the load-bearing sections: Dawson 2016, Hemati 2017 (pp.1-7), Söderström and
  Soverini 2023 (all 6 pp.), Soverini and Söderström AR preprint (pp.1-2 and 5), Gilson and Van den
  Hof (pp.2-4), Young 2015 (pp.5-10), Hefny 2015 (pp.1-5 and 8), Ni 2024 (pp.1-6), Khetarpal 2024
  (pp.2-6), Wu and Noé (pp.1-6), Askham and Kutz (pp.1-3 and 12).
- Second-hand only: Kay 1979 and the BCLS/HOYW references, via the Soverini and Söderström p.1.
- Metadata only: Söderström 2007, Zheng 2002, the CT-AR sampling papers.
- All IV / d-step / S_1 S_0^{-1} statements about our encoder are my derivations from the cited
  linear formulas, not literature claims.
- Page numbers are PDF pages of the free versions read. Journal pagination differs except for the
  IFAC paper.

## Research Log

- Queries run:
  - arXiv API, 15 calls. Sensor-noise DMD (1, hit); TLS DMD (1, off); self-predictive
    eigenvalue (0, genuine); self-predictive non-collapse (0); de-biasing DMD (1, hit);
    self-predictive RL abs (22, three on-target); IV plus Koopman (0); IV plus neural sysid (0);
    EIV plus neural dynamical (2, off); IV plus latent dynamics (7, off); titles for Hefny, online
    IV, PSRNN, VAMP (all hits).
  - OpenAlex, about 12 calls. title.search on 5 EIV classics (all resolved, all closed);
    works/doi for locations (12 DOIs); Söderström author corpus filtered to OA and EIV words
    (27 hits, decisive).
  - Google Scholar: 4 queries. 2-quoted-phrase query returned [] (known failure); unquoted EIV
    query gave 10 on-target Söderström items; CT-AR sampling query on-target metadata; fast-sampling
    delta-operator query 0 on-target.
  - CORE: 1 query (EIV, metadata only).
  - HAL API: 4 calls (2 files found, 2 metadata-only).
  - Crossref: 11 DOIs plus 1 bibliographic query.
- What worked: OpenAlex author-corpus enumeration with is_oa:true and a title regex. It surfaced
  the open DiVA copies of Söderström 2016 and 2023 that title searches marked closed. Also arXiv
  ti: exact titles, and the HAL API check before download.
- What failed: download_with_fallback without Sci-Hub on Elsevier DOIs (no OA URL). The UU EIV
  web page (no PDF links). Garnier HAL records (no files). Scholar on "fast sampling" (dominated by
  unrelated fields).
- Dead ends: the held `literature/multi-rate-horizon` oversampling papers (variance / impulse
  response, not EE pole bias). `literature/identification/ljung1999_mem.pdf` (model validation,
  not ARX bias).
- Coverage gaps:
  - No full-text source for the EE-bias versus sampling-interval law (the CT-AR papers are closed).
  - No paper found on IV inside gradient-trained neural state-space models: arXiv abs counts of 0
    for IV plus Koopman and IV plus neural sysid. Provisional, because only arXiv was counted.
  - Tian 2021 follow-ups beyond Khetarpal, Ni and Tang were not enumerated (PMLR, no DOI, so no
    citation graph).
  - A sibling agent writes into the same folder (`ljung2000_mem-control-design.pdf` is not mine).
- Suggested skill fix: add to step 2 "for a classic EIV/IV topic, enumerate the theorem author
  with `filter=author.id:<AID>,is_oa:true` and regex the titles locally; Uppsala DiVA serves
  FULLTEXT01 links that title.search rows show as closed". Also add to ACCESS: "route 6 means OA
  fallbacks only; do not call Sci-Hub".
