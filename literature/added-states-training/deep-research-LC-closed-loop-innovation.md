# L-C: closed-loop residual / innovation targets with a known controller

Sub-question L-C of the added-states training sweep (2026-10-01). PDFs in this folder.

## Frame (step 0)

- Sub-questions: (1) how residuals of a nominal model are used to identify unmodelled dynamics in closed loop, which signals are noise-uncorrelated, and what bias arises when the residual is regressed on past outputs; (2) recent IV / closed-loop work on CT and neural / physics-guided models (Gonzalez, Classens, Oomen, Kon, Bolderman); (3) the exact instrument or projection for an innovation target on encoder states.
- Seeds: Hoekstra 2026 (held, `literature/closed-loop-id/hoekstra2026_lfr-augmentation-fp-models.pdf`), Forssell and Ljung 1999 (held as Linkoping updated report), Van den Hof and Schrama 1993 and 1995 (held), Ljung 2000, Hansen-Franklin-Kosut 1989, Van den Hof and de Callafon 1996.
- Disqualifies: methods needing open-loop data; methods needing an unknown controller to be estimated (ours is known).
- Out of scope: the encoder architecture itself; orthogonal projection against parameter drift.
- Vocabularies: control (closed-loop identification, dual-Youla, model error modelling), ML (IV neural networks, physics-guided NN), econometrics (endogeneity, two-stage least squares; arXiv `endogeneity AND system identification` = 1 off-target hit).
- Held locally already: Van den Hof and Schrama 1993 and 1995, Forssell and Ljung, Landau-Karimi CLOE, Boroujeni 2025 (neural dual-Youla), `docs/research/closed-loop-plant-identification-review.md`, `docs/closed-loop-objective-literature-2026-08-28.md`. Hoekstra 2026 grep for `closed-loop`, `instrument`, `innovation`: 0 hits, so the seed gives no closed-loop treatment.

## Findings

Applicability tag: **[A]** usable directly as a training-time term on a neural latent-state model with known C and r; **[B]** usable after adaptation; **[C]** background / diagnosis.

### 0. What the innovation target actually contains (derived here, not quoted)

From the loop equations used in Van den Hof and Schrama (1995) Sec. 4.3 and Gonzalez et al. (2024) Setting 2, with u = C(r - y), y = P0 u + v, nominal Px, coprime factors P0 = N0/D0, Px = Nx/Dx, C = Nc/Dc, and Phi0 = D0 Dc + N0 Nc, Phix = Dx Dc + Nx Nc:

y - y_sim = S0 (P0 - Px) Sx C r + S0 v = Dc Nc (N0 Dx - Nx D0) / (Phi0 Phix) r + S0 v.

Consequences [C]:
1. The open-loop mode polynomial D0 cancels. The residual's poles are the roots of Phi0 (true closed-loop poles) and Phix (nominal closed-loop poles). A latent state trained to predict this residual is pulled toward the closed-loop pole of the 212 Hz mode, which feedback can damp. This is a candidate explanation for "complex poles in band, but too damped". It is a hypothesis: test it by computing the roots of Phi0 near 212 Hz from the truth model plus the known controller and comparing them with the learned poles.
2. The noise part S0 v carries the same Phi0 poles, coloured by the loop. Any encoder that reads past measured y (or u, which contains -C S0 v) is correlated with future S0 v through the S0 impulse-response tail. So a 40-step residual predictor driven by measured signals partly learns a noise predictor (a noise model). This is the direct-method regime. It is unbiased only if the noise model is correct (item 3 below).

### 1. Model-error modelling and closed-loop residual identification (part 1)

**Ljung, L. (2000). Model error modeling and control design.** IFAC Proceedings Volumes 33(15), 31-36. DOI 10.1016/S1474-6670(17)39722-7. Free copy: Linkoping report LiTH-ISY-R-2220 (DiVA), saved as `ljung2000_mem-control-design.pdf`. Read in full (font-decoded).
- Residual definition: eps(t) = y(t) - y_hat_N(t), with y_hat_N = G_hat_N(q) u(t), simulated from the MEASURED u (eqs. 6-7, p. 32). The model error model is eps = G_tilde(q) u + H_tilde(q) w (eq. 24, p. 34), or a nonlinear NNFIR in u (eq. 26, p. 35).
- To separate model error from noise you must "correlate out w". Two routes are given: periodic input with period averaging (eqs. 20-23, p. 34), or a parametric model of g_mem "assuming u and w to be independent" (Sec. 3.2, p. 34). In closed loop, u and w are not independent, so the second route is invalid as stated. Ljung does not treat the closed-loop case. [B]: periodic averaging of y - y_sim over reference periods is a valid noise-free target IF the reference is periodic.
- Also p. 34: "an arbitrarily thin resonance peak will require an arbitrarily long sequence of data". This applies to gain bounds; at damping 0.043 the -3 dB width is about 18 Hz, so it is not binding.
- Page mapping from report to IFAC pages is by page count (report pp. 2-7 = pp. 31-36). Not verified against the publisher PDF.

**Van den Hof, P.M.J. and Schrama, R.J.P. (1995). Identification and control: closed-loop issues.** Automatica 31(12), 1751-1770. DOI 10.1016/0005-1098(95)00094-X (DOI from the held file's journal record, not re-fetched). Held: `literature/closed-loop-id/vandenhof-schrama-1995-automatica-survey.pdf`. Read Sec. 4.2-4.3.
- Dual-Youla (Hansen-Franklin-Kosut) scheme, eqs. (25)-(29), p. 1760, known controller:
  z = (Dc + Px Nc)^-1 (y - Px u)   (27)
  x = (Dc + C Nc)^-1 (u + C y)     (28)
  z = R x + S e0                   (29), with R = Dc^-1 [I + P0 C]^-1 (P0 - Px) Dx (25).
  Quote (p. 1760): "signal x is uncorrelated with e0, as u + Cy = r1 + C r2", so "the identification of R forms an open-loop identification problem".
  Conditions: C known; Px stabilised by C; r uncorrelated with e0. [A/B]: z is a filtered baseline residual (the one-step residual y - Px u), and x is a pure reference signal, because u + C y = C r for u = C(r - y). Regressing z on x-driven latent states is noise-unbiased by construction. The recovered R is a closed-loop object; the plant follows from the standard dual-Youla map P = (Nx + Dc R)(Dx - Nc R)^-1. That map is not in the grepped text, so it is unverified here.
- Two-stage spectrum (eq. 24, p. 1760): the bias integrand is |[P0 - P(theta)] S0 + P(theta) [S0 - S(beta*)]|^2 Phi_r |L|^2.
- Hansen, Franklin and Kosut (1989), ACC, pp. 1422-1427, DOI 10.23919/ACC.1989.4790411: metadata only. Its content above comes second-hand via this survey.

**Van den Hof, P.M.J. and Schrama, R.J.P. (1993). An indirect method for transfer function estimation from closed loop data.** Automatica 29(6), 1523-1527. Held: `vandenhof-schrama-1993-automatica-twostage.pdf`. Read in full.
- Step 1 (eqs. 8-9, p. 1524): u(t) = T(q, beta) r(t) + R(q, gamma) e_u(t), least squares. Step 2 (eqs. 12-14): u_hat_r(t) = T(q, beta_N) r(t), then y(t) = G(q, theta) u_hat_r(t) + H(q, eta) e_y(t).
- Theorem 3.1 (p. 1524): G(q, theta_N) -> G0 w.p. 1. Conditions: T0 asymptotically stable, e and r uncorrelated and quasi-stationary, r persistently exciting, T0 and G0 in their model sets. Consistency holds "irrespective of the noise model" (eq. 18). A separate data set for step 1 makes the decorrelation exact at finite N (p. 1524).
- [A]: in a neural model, the analogue is driving the encoder / latent head with noise-free simulated closed-loop signals (u_sim = S_hat_uo r, y_sim), never the measured ones.

**Forssell, U. and Ljung, L. (1999). Closed-loop identification revisited.** Automatica 35(7), 1215-1241. DOI 10.1016/S0005-1098(99)00022-9. Read from the held Linkoping updated report; page numbers below are REPORT pages, not Automatica pages.
- Direct-method limit, Corollary 4 eq. (96), report p. 24: theta_N -> argmin of the integral of |G0 + B - G|^2 Phi_u + |H0 - H|^2 (lambda0 - |Phi_ue|^2 / Phi_u), weighted by |H|^-2. Here B is the bias pull, nonzero whenever H differs from H0 and Phi_ue is nonzero (feedback). This is the formal answer to "regress the residual on past outputs / measured inputs in closed loop": unbiased only if the noise model is exact.
- Projection method (Sec. 5.4.3, eqs. 76-79, report pp. 20-21): u_hat(t) = sum over k = -M..M of s_k r(t - k), with M -> infinity and M = o(N). This is a non-causal FIR projection onto r. Then fit an output-error model from u_hat. It is consistent "regardless of the feedback" (also nonlinear feedback) with a FIXED prefilter (report p. 27), because the residual disturbance is uncorrelated with r by construction. The report adds (p. 21) that "it would also be possible to project both the input u and the output y onto r". [A]: project the innovation target y - y_sim onto r the same way before using it as the auxiliary target. Its noise part S0 v is then removed asymptotically.
- Forssell and Ljung (2000), "A projection method for closed-loop identification", IEEE TAC 45(11), 2101-2106, DOI 10.1109/9.887634: metadata only. Same method, cited.

**Not reached in full:** Van den Hof and de Callafon (1996) CDC, DOI 10.1109/CDC.1996.572706 (closed, no OA location); Douma and Van den Hof (2005) Automatica 41, 439-457, DOI 10.1016/j.automatica.2004.11.005 (uncertainty structures; metadata only, judged peripheral); Ninness and Goodwin (1995) Automatica 31(12), 1771-1797 (cited inside Ljung 2000; not fetched); Reinelt, Garulli and Ljung (2002) Automatica 38, 787-803, DOI 10.1016/S0005-1098(01)00269-2 (DiVA copy exists; not read).

### 2. Recent work (part 2)

**Gonzalez, R.A., Pan, S., Rojas, C.R. and Welsh, J.S. (2024). Consistency analysis of refined instrumental variable methods for continuous-time system identification in closed-loop.** Automatica 166, 111697. DOI 10.1016/j.automatica.2024.111697. arXiv 2404.08955, saved as `gonzalez2024_cl-rivc-consistency.pdf`. Read Secs. 2-5.
- CLSRIVC instrument, eq. (7), p. 4 (arXiv): phi_hat_f = [-p Bj Suo,j / Aj^2, ..., Suo,j / Aj, ..., p^m Suo,j / Aj]^T r(tk), with Suo,j = C / (1 + Gj C) built from the current model iterate. This is reference-only and noise-free. IV step: eq. (3).
- Setting 2 (discrete controller, ZOH, sampled output; this matches the 4 kHz loop): CLSRIVC is generically consistent, also with coloured noise and biproper loops. Plain SRIVC, whose instrument uses measured u, is consistent only if v is white AND the loop has no direct feedthrough (Theorem 11 and Corollary 13, pp. 8-10). With a continuous-time controller, generic consistency is lost (abstract). [B]

**Gonzalez, R.A., Classens, K., Rojas, C.R. and Welsh, J.S. (2025). Identification of additive continuous-time systems in open and closed loop.** Automatica 173, 112013. DOI 10.1016/j.automatica.2024.112013. arXiv 2401.01263, saved as `gonzalez2024_additive-ct-open-closed-loop.pdf`. Read Secs. 2-4.
- This is the closest classical analogue of "baseline plus one missing mode". The model is a sum of modal submodels G = sum_i Gi. Each submodel is fitted on its residual output y_i = y - sum over l != i of G_l u (eq. 23, p. 6), filtered by 1/Ai (eq. 22).
- Closed-loop estimator: solve (1/N) sum zeta(tk)(y - sum_i Gi u) = 0 (eq. 11, p. 4). Instrument per submodel: zeta_i = [p Bi/Ai^2, ..., 1/Ai, ...] Suo(q) r(tk) (eq. 20, p. 6). Generic consistency (Lemma 1, p. 4) needs zeta uncorrelated with v, r and v mutually independent and quasi-stationary, and E{zeta phi_r^T} generically nonsingular. Theorem 2 (p. 9) also covers marginally stable submodels (Remark 6). Plain output-error in closed loop "can result in an asymptotically biased estimator" (Sec. 3.2, p. 4). [B]: with the baseline frozen, K = 1 extra submodel fitted on y - G_base u with an Suo r instrument is exactly the classical form of the user's auxiliary task.

**Classens, K., Gonzalez, R.A. and Oomen, T. (2025). Recursive identification of structured systems: an instrumental-variable approach applied to mechanical systems.** arXiv 2504.17927, saved as `classens2025_recursive-iv-mechanical.pdf`. Abstract and Sec. 2 only. A recursive closed-loop version of the above; tracks a flexible-beam resonance. [C]

**Kon, J., Heertjes, M. and Oomen, T. (2022). Neural network training using closed-loop data: hazards and an instrumental variable (IVNN) solution.** IFAC-PapersOnLine 55(12), 182-187. DOI 10.1016/j.ifacol.2022.07.308. arXiv 2202.05337, saved as `kon2022_ivnn-closed-loop-hazards.pdf`. Read in full.
- LS on closed-loop data, J_LS = ||u - F_theta(y)||^2 (eq. 9), is inconsistent even when initialised at theta0 (Theorem 11). The bias grows with the noise level.
- IV criterion (eqs. 16-17, p. 3): J_IV = ||Z^T (u - F_theta(y))||_2^2. Locally (Lemma 13, Theorem 15) theta_IV - theta0 = (Z^T F'(y))^-1 Z^T d.
- Conditions: Assumption 14, Z^T F'(y) nonsingular (instrument correlated with the parameter Jacobian); Assumption 16, z uncorrelated with d. Instruments used: the reference and its lags, z(k) = [r(k) ... r(k - N + 1)] (eq. 25). Alternatives named on p. 4: a second noise realisation of the output, or a model-predicted output. Consistency is local only (Theorem 17: initialised close to theta0).
- [A]: this is the only found paper with an IV objective on a neural network trained by gradient descent on closed-loop data. It transfers directly to an equation-error / innovation term on encoder states: replace u - F_theta(y) by (eps_target - head(x_enc)) and project onto lagged or filtered r.

**Bolderman, M., Lazar, M. and Butler, H. (2022). Physics-guided neural networks for feedforward control: from consistent identification to feedforward controller design.** CDC 2022, pp. 1497-1498 (Crossref). DOI 10.1109/CDC51059.2022.9992852. arXiv 2204.00431, saved as `bolderman2022_pgnn-consistent-identification.pdf`. Read Sec. IV.
- Remark 4.2 (p. 4): with output (NOE) noise, consistency needs the regressor built from SIMULATED outputs y_hat(theta, t - i), not measured y.
- Remark 4.4 (p. 4): in closed loop, a NARX regressor stays consistent only because v is white equation error and u(t - nk - 1) is the most recent input. "The regressor must be chosen to appropriately account for where the noise enters the system"; otherwise use IV. [B]: the gantry's noise is measurement noise (NOE-type), so an encoder fed measured y is the inconsistent case.
- Bolderman PhD thesis (TU/e 2024, `research.tue.nl/files/337535717/20240911_Bolderman_hf.pdf`): Scholar hit only, not read.

**Boroujeni et al. (2025), Neural identification of feedback-stabilized nonlinear systems**, arXiv 2503.22601. Held. A neural dual-Youla parameterisation; motivates the dual-Youla route for neural models (p. 1). [B]

### 3. Recommended instrument / projection for our setup (part 3)

All of these exploit that r is known, C is known, and r is independent of the measurement noise.

| # | Term | Estimator | Source, eq., page | Conditions | Tag |
|-|-|-|-|-|-|
| P1 | Reference-projected target | eps_r(t) = sum over k=-M..M of s_k r(t-k), fitted by LS to eps = y - y_sim; train the 40-step head on eps_r | Forssell-Ljung (76)-(79), report pp. 20-21 | r independent of v; M -> inf, M = o(N); noise part removed asymptotically | A |
| P2 | Noise-free encoder input | Encoder reads (r, u_sim, y_sim) of the closed-loop simulation, not measured (u, y); then x_enc depends on r only | Van den Hof-Schrama 1993 (12)-(14), Thm 3.1, p. 1524; Bolderman Rem. 4.2 | T0 stable, r PE, r and e uncorrelated | A |
| P3 | IV on the encoder equation error | J_IV = norm of Z^T (eps_target - head(x_enc)), squared; Z = lagged r or Suo-filtered r | Kon 2022 (16)-(17), (25), p. 3-4 | Z uncorrelated with noise (Asm. 16); Z^T Jacobian nonsingular (Asm. 14); local, start near optimum | A |
| P4 | Dual-Youla signals | z = (Dc + Px Nc)^-1 (y - Px u), x = (Dc + C Nc)^-1 C r; fit z = R x | Van den Hof-Schrama 1995 (25)-(29), p. 1760 | C known, Px stabilised by C | B |
| P5 | CLSRIVC instrument | zeta = filtered derivatives of Suo,j r with Suo,j = C/(1 + Gj C) | Gonzalez 2024 (7) p. 4; additive form Gonzalez 2025 (20), (23) p. 6 | Discrete controller (Setting 2); generic consistency | B |
| P6 | Period averaging | Average y - y_sim over reference periods | Ljung 2000 (20)-(23), p. 34 | Periodic r only | B |

Caveat common to P1-P6: they remove NOISE bias only. They do not change the fact (Section 0) that the residual's dynamics are closed-loop dynamics. If the target is to carry the open-loop mode, the closed-loop object must be mapped back to the plant. In P4 this is the dual-Youla map; in the user's full closed-loop simulation objective the known controller already does it. A latent head fitted to eps alone does not.

### BibTeX

```
@article{vandenhof1993twostage, author={Van den Hof, P. M. J. and Schrama, R. J. P.}, title={An indirect method for transfer function estimation from closed loop data}, journal={Automatica}, volume={29}, number={6}, pages={1523-1527}, year={1993}}
@article{vandenhof1995closedloop, author={Van den Hof, P. M. J. and Schrama, R. J. P.}, title={Identification and control: closed-loop issues}, journal={Automatica}, volume={31}, number={12}, pages={1751-1770}, year={1995}}
@inproceedings{ljung2000mem, author={Ljung, L.}, title={Model Error Modeling and Control Design}, booktitle={IFAC Proceedings Volumes}, volume={33}, number={15}, pages={31-36}, year={2000}, doi={10.1016/S1474-6670(17)39722-7}}
@article{forssell1999revisited, author={Forssell, U. and Ljung, L.}, title={Closed-loop identification revisited}, journal={Automatica}, volume={35}, number={7}, pages={1215-1241}, year={1999}, doi={10.1016/S0005-1098(99)00022-9}}
@article{forssell2000projection, author={Forssell, U. and Ljung, L.}, title={A projection method for closed-loop identification}, journal={IEEE Transactions on Automatic Control}, volume={45}, number={11}, pages={2101-2106}, year={2000}, doi={10.1109/9.887634}}
@article{gonzalez2024clrivc, author={Gonz{\'a}lez, R. A. and Pan, S. and Rojas, C. R. and Welsh, J. S.}, title={Consistency analysis of refined instrumental variable methods for continuous-time system identification in closed-loop}, journal={Automatica}, volume={166}, pages={111697}, year={2024}, doi={10.1016/j.automatica.2024.111697}}
@article{gonzalez2025additive, author={Gonz{\'a}lez, R. A. and Classens, K. and Rojas, C. R. and Welsh, J. S.}, title={Identification of additive continuous-time systems in open and closed loop}, journal={Automatica}, volume={173}, pages={112013}, year={2025}, doi={10.1016/j.automatica.2024.112013}}
@article{kon2022ivnn, author={Kon, J. and Heertjes, M. and Oomen, T.}, title={Neural Network Training Using Closed-Loop Data: Hazards and an Instrumental Variable ({IVNN}) Solution}, journal={IFAC-PapersOnLine}, volume={55}, number={12}, pages={182-187}, year={2022}, doi={10.1016/j.ifacol.2022.07.308}}
@inproceedings{bolderman2022consistent, author={Bolderman, M. and Lazar, M. and Butler, H.}, title={Physics-Guided Neural Networks for Feedforward Control: From Consistent Identification to Feedforward Controller Design}, booktitle={IEEE Conference on Decision and Control}, pages={1497-1498}, year={2022}, doi={10.1109/CDC51059.2022.9992852}}
```

## Access status

TU/e browser access: NOT ATTEMPTED (sub-agent; preflight belongs to the parent). Items marked `needs-browser-route`: Van den Hof and de Callafon 1996 (CDC), Hansen-Franklin-Kosut 1989 (ACC), Forssell and Ljung 2000 TAC (DiVA copy likely, bot-walled by Anubis on the record page), Douma and Van den Hof 2005.

## Evidence quality

- Read in full or in the relevant sections: Van den Hof-Schrama 1993; Van den Hof-Schrama 1995 Sec. 4.2-4.3; Forssell-Ljung (report version) Sec. 5.4, 6; Ljung 2000 (report version); Gonzalez 2024 Secs. 2-5; Gonzalez 2025 Secs. 2-4; Kon 2022; Bolderman 2022 Sec. IV.
- Abstract / section-2 only: Classens 2025, Boroujeni 2025.
- Metadata only: Hansen et al. 1989, Van den Hof-de Callafon 1996, Douma-Van den Hof 2005, Reinelt et al. 2002, Forssell-Ljung 2000 TAC, Ninness-Goodwin 1995.
- Section 0 is my own derivation from the loop equations, not a quoted result. Its pole-cancellation claim is checkable by algebra; the "too damped = closed-loop pole" link is an untested hypothesis.

## Research Log

- Queries run:
  - arXiv `abs:"closed-loop" AND abs:"instrumental variable" AND abs:"continuous-time"`: 4, all on target.
  - arXiv `au:Gonzalez_Rodrigo AND abs:"closed-loop"`: 8, 5 on target.
  - arXiv `abs:"instrumental variable" AND abs:"neural network" AND abs:"closed-loop"`: 1, bullseye (Kon IVNN).
  - arXiv `au:Bolderman AND abs:"closed-loop"`: 0 (genuine; 746 bytes valid body).
  - arXiv `abs:"physics-guided neural network" AND abs:"closed-loop"`: 0.
  - arXiv `au:Bolderman`: 8. arXiv `au:Kon_Johan`: 11.
  - arXiv novelty counts: `abs:"instrumental variable" AND abs:"state-space" AND abs:"neural"` = 0; `abs:"dual-Youla" AND abs:"neural"` = 1 (Boroujeni, held); `abs:"endogeneity" AND abs:"system identification"` = 1, off-target.
  - Crossref `query.bibliographic`: 10 lookups, all resolved.
  - OpenAlex `works/doi:`: 4 calls, no 429.
  - Google Scholar: 1 sentence query; only new lead was the Bolderman thesis.
- What worked: arXiv field-prefixed AND queries (Kon IVNN was found in one call). Mining held PDFs: the dual-Youla equations and the projection method were already on disk. The DiVA `smash/get/diva2:<id>/FULLTEXT01.pdf` direct path worked although the record page is bot-walled.
- What failed: the DiVA record / urn resolver returned an Anubis JS challenge. The Ljung 2000 report and the held Forssell report extract as font-coded or space-broken text with pypdf. The Ljung font code was decoded as two base-36 characters minus 360 (for example `/CP` = 'a'). Forssell needed a whitespace-insensitive grep.
- Dead ends: Scholar results were dominated by robotics sim2real and fault-diagnosis theses.
- Coverage gaps: no dblp queries were spent. CDC/ACC closed-loop IV for neural SS models after 2023 may be missed. Forward citations of Kon 2022 were not traversed (OpenAlex `cites:` not run). So the claim that "no neural latent-state model trained on a closed-loop residual with an instrument exists" is PROVISIONAL, backed only by the arXiv zero count.
- Suggested skill fix: add to Step 4/5: "DiVA (liu.diva-portal.org) record pages are Anubis-walled; take the `diva2:<id>` from the redirect URL and fetch `https://liu.diva-portal.org/smash/get/diva2:<id>/FULLTEXT01.pdf` directly (worked first try 2026-10-01). Old Linkoping TeX reports extract as `/XX` font codes: decode with chr(36*idx(X1)+idx(X2)-360) over 0-9A-Z."
