# Literature synthesis of 2026-09-22 (session 8bf42294), saved verbatim (G0, CN-002)

Output of session `8bf42294-1c7b-495d-a10a-e84592f5160f` (the four-agent deep-research
sweep of 2026-09-22), extracted from the transcript by `tools/extract_literature.py`.
This is that session's text, not new work of this session. Selection rule: assistant
messages starting "SQ1", "SQ4", "SQ2", "All four agents are in", plus the answers of
2026-09-22 16:49 and 17:11 UTC (18:49 and 19:11 local). Headings show local time. The only edit:
em and en dashes replaced (project dash rule).

## Message 1 (2026-09-22 09:55)

SQ1 (closed-loop ID) is back, and it lands the core deliverable: the de-shaping identity is stated by the literature itself, not inferred. Ljung & Forssell 1999 Eq. (13)/(15) give `H = H*·(I + GK) = H*/S`, i.e. a noise model identified in closed loop is converted to a source model by multiplying by `S^-1` ,  and the paper's framing is that indirect methods are just direct methods whose noise model contains the regulator.

Two results that will shape the phase-2 choice, flagged now because they are negative:
- **Encoder noise `n` and output disturbance `v` are not separable at all** from closed-loop data ,  both enter `q1` at the same node, so every method returns their sum. Separating them needs a second sensor or a shape prior (formal reference: Anderson & Gevers 1982, metadata only, browser route down).
- **The `i_fb` hypothesis is split.** In `iter0`, `i_fb = K·e` identically, so the joint noise spectrum is rank 1 and `i_fb` carries zero *statistical* information `e` does not. It becomes informative only when `u_ff` is on, and then for separating input-side `d` from `v+n` ,  not for improving the estimate of `v+n`.

Three agents still running; I'll hold the full synthesis until they land.

## Message 2 (2026-09-22 09:57)

SQ4 (ILC) is back, and it's the strongest of the two so far. Headline: the split is clean.

- **Estimator, for the repeatable part.** Dijkstra 2004 (TU Delft wafer-stage ILC thesis, nm scale) Proposition 4 / Eq. 5.23: the converged feedforward is `u_∞ = P⁻¹d^{ti}` ,  an *input-referred*, model-independent, `L`-independent exact inverse of the repeatable disturbance. `S` cancels. That is stronger than the `i_fb` hypothesis I carried in: `u_ff` isn't merely un-suppressed by the sensitivity, it is `S`-free by construction.
- **Bound only, for the non-repeatable part.** `e_∞ = (1+Q)/(1+T)·S·d^{tv}`, with the factor ≥ 1 for any stable ILC and = 2 in the `Q=1` limit. So your 137/158/113 nm `iter8` residual is an *upper* bound on the trial-varying content, inflated by a factor between 1 and 2. That makes the standstill window's blind spot quantitative rather than asserted.
- **Butcher, Karimi & Longchamp 2008** gives the closed form you asked for: `σ_res² = σ_d² + (h/2π)∫|Q|²(Φ_d+Φ_n)dω`, reducing to `2σ_d² + σ_n²` at `Q=1, L=G⁻¹`. Invertible *given `Q`*. Incidental but notable: their rig is an **ETEL linear motor**.
- **Nobody inverts ILC convergence to estimate the covariance** ,  it is a design input in every stochastic-ILC paper found (provisional-strong negative, five vocabularies). Norrlöf 2002 comes closest, and does it from `u_ff` increments.
- Caveat to carry: Butcher's `σ_res²` is output-referred, so **ILC does not solve the double-`S` problem**. It solves the repeatable/non-repeatable split, which nothing else in the campaign does.

One coverage gap worth knowing: the Bristow/Tharayil/Alleyne 2006 survey was unreachable (closed IEEE, browser route down), so its noise section is unverified.

SQ2 and SQ3 still running.

## Message 3 (2026-09-22 09:58)

SQ2 (nonparametric FD) is back. The decisive clause was **already on disk, unread**: `literature/BLA/System_Identification_-_A_Frequency_Domain_Approach.pdf` Chapter 7 *is* the MSSP 2010 Part I/II pair in book form, so the two paywalled papers the handoff named as targets were never actually out of reach.

- **§7.2.1 p.228 settles (a) and (b):** the LPM works on non-periodic deterministic input; open loop is required by the *direct* LPM and is explicitly lifted "if a known reference signal is available" (§7.2.7). Your `r` is bit-identical and noise-free, so you are inside the stated validity set. Eq. (7-13) gives `Ĉ_V` from **one record, no periods, no repeats**, and §7.2.2.5 shows it is robust to *lack of excitation* ,  which is what makes it applicable to the 444 ms standstill window.
- **The real obstacle is not periodicity.** §7.2.2.1 requires the input to vary "wild" over frequency so `G·U` separates from the transient `T`. A point-to-point profile has a smooth sinc-enveloped spectrum ,  exactly the excluded case. That, not the lack of multisines, is what disqualifies naive LPM here.
- **Independent confirmation of SQ1's hard negative.** SQ2 derived the same rank-1 result by different algebra: `V_U = K·V_Y` identically, so `i_fb` adds no equation for source separation. Two agents, two routes, same conclusion ,  I'm treating that as settled.
- **Hazard, sharpened:** the 2.5 ms logging lag rotates phase by only ~0.1-0.2 rad over a local window, so the LPM's smoothness assumption survives it. What does *not* survive is any construction treating `i_fb = K·e` ,  the lag turns an exactly rank-1 pair into an apparently rank-2 one, manufacturing information that is pure artefact. De-rotate before any joint use.
- **New corpus, sole-sourced by one Google Scholar sentence:** Maeda & Iwasaki (Nagoya IT) estimate plant FRFs from *point-to-point positioning motion in closed loop with no added excitation*, via differential filtering ,  the direct answer to the excitation obstacle. None of their papers estimates a noise covariance, so the combination (differential filtering to satisfy the LPM condition, then Eq. 7-13 on the residual) appears unpublished.

SQ3 still running; I'll write the merged synthesis when it lands.

## Message 4 (2026-09-22 10:00)

All four agents are in. Synthesis below, against the section 10 acceptance criterion.

# What the measured floor represents, and which inversion recovers the source

## 1. The core answer, and it is stated by the literature rather than inferred

With `S = (I+GK)^-1`, `T = I-S`, sensor noise `n` at the encoder, input/process disturbance `d`, and output-referred disturbance `v`:

| signal | reference part | noise part |
|-|-|-|
| `q1` | `T r` | `S(n + G d)` |
| `e = r - q1` | `S r` | `-S(n + G d)` |
| `i_fb = K e` | `K S r` | `-K S(n + G d)` |

so `Phi_e^noise = |S|^2 Phi_v`, and the de-shaping operator is `Phi_v = |I + GK|^2 Phi_e = |S|^-2 Phi_e`.

The supervisor's objection is correct and it is the literature's own statement. **Ljung and Forssell 1999** (IEEE TAC 44(11):2206-2209, DOI `10.1109/9.802946`, free at LiU DiVA, read in full) Eq. (13)/(15): `H(q,theta) = H*(q)(I + G(q,theta)K(q))`, i.e. `H = H*/S`. Their framing is that "the so-called indirect methods are indeed 'direct' methods employing noise models that contain the regulator". Read backwards, that is exactly the conversion you need: a noise model identified in closed loop becomes a source model by multiplying by `S^-1`. The project has cited Forssell and Ljung only for the bias catalogue and Eq. (117); this clause has never been quoted.

**Eq. (117) is a dead end for this question** and should be dropped from the framing. It is the variance of the *plant* estimate, and Forssell and Ljung's own Remark (2) p.30 says the feedback-noise part of the input has no information value for estimating `G`. The equations that bear on `Phi_v` are (11), (18), (19) and Ljung and Forssell (21).

## 2. Three hard negatives, and two of them were derived independently by separate agents

**(a) `n` and `v` are not separable at all.** Both enter the measured `q1` additively at the same node. Every method in the register returns their sum. Formal reference: Anderson and Gevers, *Automatica* 18(2):195-213, 1982, DOI `10.1016/0005-1098(82)90108-X` (closed, unreachable). This is a structural fact about the topology, not a gap in the literature.

**(b) `i_fb` adds no information for attribution.** SQ1, SQ2 and SQ3 each derived this by different routes and agreed: in `iter0`, `i_fb = K e` identically with `K` known, so the joint noise spectrum of `(e, i_fb)` is **rank 1**. Both sources enter both channels in the ratio `1 : |G|`. SQ3 anchors it in a held paper, Gevers, Bazanella and Miskovic, CDC 2008: closed-loop identification without external excitation works "if the controller is of sufficient degree", and a diagonal PID-class `K` is not.

This splits your carried hypothesis cleanly, and the thesis should keep the halves apart:
- **For detecting unexplained dynamics, the hypothesis stands.** `e` is `S`-suppressed below crossover, `i_fb` is not. That is `telica_residual_spectrum.py`'s claim and it survives.
- **For source attribution, it fails.** No paper found in five vocabularies supports it, and the algebra refutes it.
- `i_fb` recovers value only if it carries a source `e` does not: current-loop/DAC noise, current-sensor noise, or the known `u_ff`. Each is a *third* independent source, i.e. a third corner of the hat.

**(c) Your case is a two-cornered hat.** Three-cornered hat (Gray and Allan 1974; Premoli and Tavella, IEEE TIM 1993, DOI `10.1109/19.206671`) needs `N >= 3` channels to give `N(N-1)/2` pairwise variances. You have 2 sources and 1 measured spectrum, one channel short of even the ill-conditioned case. Confirmed no control-loop analogue exists (374 OpenAlex + 15 arXiv hits, all metrology, geodesy and hydrology). The metrology remedy for the degenerate case, imposing a positivity or minimum-norm criterion to pick a point from an admissible set, is exactly the remedy available to you, and it is **a choice, not an estimate**.

## 3. The framing correction that matters most

**Section 5 of the handoff has the shape argument backwards, and this changes the phase-2 design.**

Your reasoning was: the standstill PSD sits 31 to 36 dB above the quantisation floor, so a purely-sensor explanation is doubtful. Two corrections:

1. **Encoder noise is not quantisation noise.** Interpolation error, sub-divisional error and electronic noise in the read channel all sit in the sensor path and are not bounded by `q/sqrt(12)`. The 31 to 36 dB gap excludes *quantisation* as the dominant term, which you already established. It does not argue against a sensor explanation at all.
2. **The rising shape favours the sensor path, not against it.** White `n` gives `Phi_e = sigma_n^2|S|^2`, rising toward 1. White `d` gives `Phi_e = sigma_d^2|SG|^2`, and for a mass line `|G| ~ 1/(m w^2)` this *falls* as `w^-4`. A spectrum with most variance above 200 Hz and rising into the kHz is to first order the signature of the sensor path, unless `Phi_d` itself rises like `w^4`.

That asymmetry is your real lever for separation, and it is strong. But note precisely what it rests on: the two basis shapes differ *exactly* by `|G(jw)|^2`, so **any shape-based separation claim is a claim that you know `|G|`**. Separability is a statement about the plant and nothing else.

## 4. Method register, assumptions checked one by one against section 4

Legend for the dataset properties: **P1** non-periodic point-to-point; **P2** ILC iterations are not repeated realisations; **P3** exactly one genuine repeat pair; **P4** `K` known and verified; **P5** `r` known and noise-free; **P6** no open-loop data; **P7** `G` under recovery, not reliably known; **P8** records ~0.5-1 s plus 444 ms standstill; **P9** the 2.5 ms `MF230` logging lag.

### Survives

| Method | Citation | Estimates, in `S,T,G,K` | Assumption check |
|-|-|-|-|
| **Indirect LPM (noisy-input / feedback)** | Pintelon and Schoukens, *SI: A Frequency Domain Approach* 2nd ed., **Ch. 7, §7.2.7, Eqs. (7-47) to (7-50)**. **Held**, `literature/BLA/`. | `G_ry = T`, `G_ru = KS`, `Ghat = T(KS)^-1 = G`; `Chat_Vz` = 2x2 output-referred noise covariance. `Chat_Vy / |S|^2 = Phi_d` at the position node, which is the injectable quantity. | Needs a known reference: **P5 satisfied**. Non-periodic single record: **P1, P2, P3, P8 satisfied** via Eq. (7-13), `q = 2n+1-(R+1)(n_u+1)` dof. **P9**: survives, a 2.5 ms lag rotates only ~0.1 to 0.2 rad over a local window. **Fails H1**, see below. |
| **LPM record concatenation** | Same book **§7.2.9, Eqs. (7-56) to (7-62)**; paper form is the held `relan2017lpm_bla` Eqs. (21)-(22). | Full concatenated-resolution `Ghat`, `Chat_V`, with `M_c - 1` appended unit steps on the known input. Bias reduction up to 33.9 dB. | Directly answers **P8**. Handles **P2** properly: `u_ff` is carried as a *second known exogenous input*, not as a repeat. |
| **Bound on `sigma_n^2` needing `S` only** | Derived; the set-valued framing follows Kong et al. below. | `sigma_n^2 in [0, min_w Phi_e(w)/|S(jw)|^2]`, `Phi_d = (Phi_e - sigma_n^2|S|^2)/|SG|^2 >= 0` | **Needs no `G`, so immune to P7.** The strongest statement the data supports without the plant. |
| **Trial-domain separation (repeating vs iteration-varying)** | **Aarnoudse, Pavlov, Oomen**, *Automatica* 171:111902, 2025, DOI `10.1016/j.automatica.2024.111902`. Free VoR on TU/e Pure, downloaded. | Splits by "the difference in amplitude characteristics of repeating and varying disturbances". Not sensor-vs-input, but a genuinely **second independent equation**, and it does **not need `G`**. | **P2 is the obstacle**, but in the **standstill window with `u_ff` known and subtractable** the iterations are far closer to repeats than the move segments. |
| **Converged ILC feedforward as a disturbance estimate** | **Dijkstra**, PhD thesis, TU Delft 2004 (wafer stage, Philips Research, nm scale), **Prop. 4 / Eq. 5.23**. Free, 245 pp, read. | `u_inf = P^-1 d^ti`: the converged feedforward is an **input-referred, model-independent, `L`-independent** exact inverse of the repeatable disturbance. **`S` cancels.** | Needs `Q ~ 1` in band only. This is a *stronger* form of your hypothesis: `u_ff` is not merely un-suppressed by `S`, it is `S`-free by construction. |
| **ILC residual as a bound** | Dijkstra Eqs. (5.24)-(5.27); **Butcher, Karimi and Longchamp**, *Int. J. Control* 81(1):156-166, 2008, DOI `10.1080/00207170701484851`, free via CORE, read in full, **Eq. (53)**. | `e_inf = (1+Q)/(1+T)·S·d^tv`, factor `>= 1` for any stable ILC, `= 2` at `Q=1,kappa=1`. Butcher: `sigma_res^2 = sigma_d^2 + (h/2pi)∫|Q|^2(Phi_d+Phi_n)dw`, reducing to `2 sigma_d^2 + sigma_n^2` at `Q=1, L=G^-1`. | **Quantifies the standstill window's blind spot.** Your 137/158/113 nm `iter8` residual is an upper bound on the trial-varying content, inflated by a factor between 1 and 2. **Output-referred, so ILC does not solve the double-`S` problem.** Incidental: Butcher's rig is an **ETEL linear motor**. |
| **Identifiable-element count, run before fitting anything** | **Kost, Dunik, Straka**, *IFAC-PapersOnLine* 54(7):316-321, 2021, DOI `10.1016/j.ifacol.2021.08.378`. Diamond OA, downloaded, 6 pp. | How many unique elements of `Q`/`R` are identifiable by correlation methods, given the model. | Answers "how many free noise parameters may I fit?" before you fit one. |

### Fails, with the reason

| Method | Citation | Why it fails |
|-|-|-|
| Direct LPM | Book §7.2.1, Guideline 2 §7.5 | Requires open loop; the direct closed-loop estimate is **biased**, collapsing to `-1/K` when disturbance dominates. **P6.** |
| Robust BLA, fast BLA, robust/fast local-polynomial BLA | Book §7.3.3, 7.3.4, 7.3.6-7.3.7 | Periodic multisine + `P >= 2` periods + `M >= 2` realisations. **P1, P2.** |
| `pintelon2020bla_feedback` (held) as a noise estimator | IEEE TIM 69(10), DOI `10.1109/TIM.2020.2987476` | Needs `P = 2` periods of a random-phase multisine, `M = 100` realisations. **P1, P2.** Its structural framing (BLA as `T/(KS)`) does qualify and is what §7.2.7 implements. |
| Oomen and Bosgra 2008 (held) **as published** | CDC 2008, DOI `10.1109/CDC.2008.4738592` | Eq. (16) needs `w` held identical over `n_exp` experiments; Assumption 10 needs periodic excitation. **P1, P2, P3.** See the rescue below. |
| Two-stage, projection, CLOE family, X-CLOE, DIDIM | held in `literature/closed-loop-id/` | Fixed noise model by construction. They estimate no `H`, and their residual is `S v` again. Van den Hof and Schrama 1995 §4.1 says it outright. |
| Joint I/O as formulated by Forssell and Ljung | Eqs. (101)-(102) | `H1`, `H2` are *fixed* weights; the criterion contains `Phi_r` only. |
| ALS | **Odelson, Rajamani, Rawlings**, *Automatica* 42(2):303-308, 2006, DOI `10.1016/j.automatica.2005.09.006` | Needs `(A,C,G)` **exactly**. It is the state-space spelling of dividing by `|S|^2` and `|SG|^2`, and the second division needs the plant. **P7.** |
| Measurement-difference autocovariance | **Dunik, Kost, Straka**, *Automatica* 90:16-24, 2018 | Needs `C` full column rank, i.e. more independent measurements than states. One encoder per axis on a flexible-mode state vector fails outright. |
| Desroziers diagnostics | **Desroziers et al.**, QJRMS 131:3385-3396, 2005, DOI `10.1256/qj.05.108` | Structurally your two-channel idea, and mature. Works because its two residuals are taken *before and after* the analysis update. Yours obey `u = Ke`, so they are not two independent residuals. |
| Three-cornered hat / triple collocation | Premoli and Tavella 1993 | Needs a third independent sensor. `N=2`. |
| PES Pareto | **Abramovitch**, IEEE CCTA 2020, DOI `10.1109/CCTA41146.2020.9206364`; Abramovitch, Hurst, Henze, IEEE TMAG 34(1):17-23, 1998 | The **correct framework**, and the settled mechatronics name nobody in this repo had searched: it estimates `Phi_i` at each injection point and recomposes `Phi_e = sum_i |H_i|^2 Phi_i` with `H_n = S`, `H_d = SG`. Source-referred, injectable. **Partially disqualified**: it gets uniqueness from ingredient (2), per-element isolation measurements you do not have. Honest reading: the disk-drive community did not separate sources from PES alone either. |
| Closed-loop subspace noise covariance | Danesh Pour, Huang, Shah, *J. Process Control* 19(10), 2009 | Consistent, but returns the innovation covariance of the closed-loop predictor: output-referred, no entry-point split. |
| RPCA over trials | Lin, Sun, Tomizuka, ACC 2015 | Presumes a common low-rank repeatable component across stacked trials. Your feedforward changes every iteration. **P2.** |
| Stochastic/Kalman ILC (Norrlof 2002; Wallen Axehill 2014; Gunnarsson and Norrlof 2006; Saab 2001; Schoellig 2012) | see agent logs | `sigma_d^2`, `sigma_n^2` are **design inputs assumed known** in every one. Norrlof 2002 Algorithm 1 step 2: "Choose `r_Delta_d` and `r_n` **from physical insight**". |
| Set-membership / deterministic white-noise modelling | Paganini 1996 via Oomen ref [15] | Computationally infeasible at 20 kHz x 444 ms x 6 axes, per Oomen's own §I. |

## 5. Two limit results you should cite rather than work around

**Kong, Sukkarieh, Arnold, Chen, Zheng**, *Systems & Control Letters* 162:105172, 2022, DOI `10.1016/j.sysconle.2022.105172`, free at arXiv:2202.04963, read in full. For ALS under an arbitrary unknown input: `Q` and `R` cannot be uniquely jointly identified, and **neither is uniquely identifiable even when the other is known**. Their own closing clause, never quoted anywhere in this project: they say regularisation is the open route out.

**Kong et al.**, IEEE TAC 2022, DOI `10.1109/TAC.2022.3208338`, arXiv:2209.07104, read in full. The positive companion: necessary and sufficient rank conditions for `Q` given `R`. Gating assumption is `C` full column rank, which you fail. Their verdict: "only under restrictive conditions".

Note the scope boundary that decides which one applies to you. Their "unknown input" is an arbitrary *unmodelled deterministic* signal. If you insist `Phi_d(w)` is an arbitrary non-negative function, you are in their case and the answer is a flat no. **If you commit to a parametric `Phi_d`, you leave their case**, and the problem becomes variance-component estimation on known spectral shapes, identifiable iff `|S|^2` and `|SG|^2 Phi_d^shape` are non-collinear over the band, i.e. iff `|G(jw)|` varies appreciably across it.

And the held paper that states your limit in TU/e's own words, a clause never quoted in `docs/references.md`. Oomen and Bosgra 2008 p.2: model residual can be attributed either to model uncertainty or to disturbances, so "presence of a trade-off in the model validation problem implies that the problem is **ill-posed**", and in the conclusions, "the notion of a disturbance is **poorly defined** in a deterministic framework". Their resolution is periodic input design, which you do not have. The trade-off therefore remains open in your data **by construction**, which is a defensible thesis sentence rather than a failure.

## 6. The real obstacle is not periodicity

This is the single most actionable finding, and it reframes phase 2.

Book **§7.2.1 p.228**, verbatim: the local polynomial approach "can be applied to both random and deterministic (periodic and non-periodic) input signals", and the open-loop assumption is lifted because "if a known reference signal is available, then these two assumptions can be relaxed, and the noisy input case and systems operating in feedback can also be handled (see Section 7.2.7)". Your `r` is bit-identical and noise-free, so you are inside the stated validity set. §7.2.2.5 additionally shows `Chat_V` is computable **even when `U(k+r) = 0` across the window**, i.e. robust to lack of excitation, which is exactly the standstill case.

**§7.2.2.1 is what actually bites.** The LPM needs `U(k)` to vary "sufficiently wild over the frequency" so `G·U` separates from the transient `T`. A point-to-point profile has a smooth sinc-enveloped spectrum, precisely the excluded case. Locally `U` is as smooth as `G` and `T`, so the LS problem cannot separate them and the transient leaks into both `Ghat` and `Chat_V`.

The literature's answer to exactly this, and it is a corpus new to the project, sole-sourced by one Google Scholar sentence: **Maeda and Iwasaki**, "Empirical Transfer Function Estimation With Differential Filtering and Its Application to Fine Positioning Control of Galvano Scanner", IEEE TIE, DOI `10.1109/TIE.2022.3217616`, free at the Nagoya Institute of Technology repository, read. They estimate the plant FRF from **point-to-point positioning motion in closed loop with no additional excitation**, by differential filtering both channels until the record satisfies a leakage-free condition. Companion: Maeda et al., AIM 2024, DOI `10.1109/AIM55361.2024.10637189` (the theoretical order condition), and **Maeda 2026, "Local Rational Modeling with Differential Filtering for FRF Estimation from Point-to-Point Motion Data"**, IEEE Xplore doc `11625566`, which is the LRM fusion and is the single most on-question unread item in the sweep.

**None of the Maeda papers estimates a noise covariance.** The combination (differential filtering to satisfy the excitation condition, then Eq. (7-13) on the residual to get `Chat_V`) is, as far as this sweep reaches, unpublished. That is plausibly a contribution rather than a gap.

## 7. One more hazard, and one rescue

**Hazard, sharpened.** The 2.5 ms lag is harmless for magnitude-based quantities and survives LPM local smoothing. What does not survive is any construction treating `i_fb = K e`: the lag turns an exactly rank-1 pair into an apparently rank-2 one, **manufacturing information that is pure logging artefact**. If any joint `e`/`i_fb` frequency-domain method is ever used, de-rotate `I_fb(k)` by `exp(+j w_k tau_hat)` first. Given §2(b), this is moot for attribution.

**Rescue.** Oomen and Bosgra's §III-C says the approach is "to keep `w` constant during repeated experiments"; periodicity enters only in §IV as an *efficiency* device. Your 444 ms standstill windows are `n_exp` segments with `w` held constant at zero. **Eq. (17) therefore applies to the standstill windows directly**, with no periodic excitation and no repeated realisations of the move. That converts a held paper the register lists as disqualified into a usable estimator, feeding the de-shaping step.

## 8. Access status and coverage gaps

**TU/e browser access: UNAVAILABLE.** Layer 1 down, Chrome extension not connected (`list_connected_browsers` returned `[]`). Preflight run once in the parent. Everything below is `unreachable - browser route unavailable`, not `unreachable`:

- **Bristow, Tharayil and Alleyne 2006** (IEEE CSM 26(3):96-114, DOI `10.1109/MCS.2006.1636313`), the standard ILC survey named in the handoff. Its noise section is **unverified**; do not cite it for a noise claim from this report. Butcher and Dijkstra cover the same ground and were read.
- **Abramovitch CCTA 2020 PES Pareto tutorial** bodies. Its §II Bode-integral section may already discuss the rising-`S` confound, so retrieve `10.1109/CCTA41146.2020.9206364` before any novelty claim there.
- **Maeda 2026 LRM-Diff**, IEEE doc `11625566`. No resolvable DOI, not in Crossref or OpenAlex.
- Van den Hof, *Annual Reviews in Control* 22:173-186, 1998 (268 citations, full-text snippet mentions noise on the *controller output*, could invert the §2(b) verdict); Egardt ECC 1997; Anderson and Gevers 1982; Odelson 2006; Gunnarsson and Norrlof 2006; MSSP 2010 Parts I and II.

On the MSSP pair specifically: they were named as sweep targets and are closed, **but Chapter 7 of the held textbook is the book form of those two papers**, so the content was on disk all along. That is the run's best single lesson.

**Gaps worth naming.** No dblp coverage (it returned HTML errors and empty bodies to two agents, so recent CDC/ECC/ACC 2010-2026 is unswept). Ljung's textbook Ch. 13 still not held, now the third sweep to record it. Statistics/BSS and acoustics/OMA vocabularies unsearched by SQ2. Dijkstra's experimental chapters 4.7-4.11 and 5.3 unread, and he ran nm-scale ILC noise-amplification experiments on a real wafer stage whose numbers are directly comparable to your 137/158/113 nm. Tom Oomen's 494-work corpus was enumerated by SQ3 but not by SQ4.

Gap claims graded: §2(a) and the Kong limit results are **firm** (read in full). §2(b) is **firm on the algebra, provisional on the "nobody has published this"**. The "no control-loop three-cornered-hat analogue" claim is **provisional**.

# Recommendation

Per the one-recommendation rule, and stopping short of phase 2 as instructed:

**Estimate `Phi_v = Phi_e/|S|^2` on the 444 ms standstill windows via the indirect LPM (Ch. 7 §7.2.7 + §7.2.9 concatenation), report `sigma_n^2` as the interval `[0, min_w Phi_e/|S|^2]` rather than a point, and inject `Phi_v` at the position node.** Rationale: it is the only route that satisfies every one of P1 through P8, the bound needs `S` only and so is immune to P7, and it sidesteps the `n`-versus-`d` split which §2(a) shows is not identifiable anyway. Injecting a combined position-referred source at the encoder node is exactly what your in-loop injection decision requires; the decomposition you framed as the goal is neither necessary for it nor achievable.

Two things to settle before phase 2, both of which change the estimator:

1. **The excitation obstacle (§6).** Standstill windows may dodge it, since §7.2.2.5 covers lack of excitation, but that needs checking against your actual standstill spectrum before committing.
2. **Whether you want the trial-domain axis (Aarnoudse) as a second equation.** It is the only extra equation available without new hardware, it does not need `G`, and it answers the one thing standstill structurally cannot see: the non-repeatable content during motion, which §4 bounds at between 1x and 2x your `iter8` residual.

Nothing was written to disk and no estimate was produced. If you want this logged, the right home is an append to `docs/gantry-augmentation-problem-log.md` plus reference-key entries in `docs/references.md` for the roughly 15 new papers; say the word and I'll write it.

## Message 5 (2026-09-22 18:49)

## The shaping, in one table

With `S = (I+GK)^-1`, sensor noise `n` at the encoder, input disturbance `d`, and `v = n + G d` the combined output-referred source:

| what | expression |
|-|-|
| what you measured on the logged error | `Phi_e = \|S\|^2 · Phi_v` |
| the source | `Phi_v` |
| de-shaping | `Phi_v = \|I + GK\|^2 · Phi_e = \|S\|^-2 · Phi_e` |

That inversion is not something I derived: Ljung and Forssell 1999 (IEEE TAC 44(11), DOI `10.1109/9.802946`) Eq. (13)/(15) state `H = H*·(I + GK)`, i.e. a noise model identified in closed loop becomes a source model by multiplying by `S^-1`. Their phrasing is that indirect methods are just direct methods whose noise model contains the regulator.

## Why it matters for injection

Your simulator applies `S` itself. So:

- Inject `Phi_v` at the encoder node, and the simulated error comes out at `|S|^2 Phi_v`, matching the machine. Correct.
- Inject the **measured** `Phi_e` there, and you get `|S|^4 Phi_v`. `S` applied twice. That is the supervisor's objection, and it is exactly right.

Practically, `|S| << 1` below crossover and `|S| -> 1` above it, so the double-`S` error is negligible at high frequency and large at low frequency. Since most of your standstill variance sits above 200 Hz and rises into the kHz, the two quantities are actually close in the band that dominates the rms, which is worth knowing but is not a reason to skip the inversion.

## The free lunch, and the part that costs

De-shaping by `|S|^-2` needs only `K` (verified, D-074) and `G`. But note where `S ~ 1`, the measurement already **is** the source with no model at all. Your spectrum rises into exactly that band, so the high-frequency portion needs no inversion; only the sub-crossover portion does, and that is the part where `G` enters and where the recovered number is only as good as your plant.

The bound that needs `S` alone and no `G`:

```
sigma_n^2  in  [ 0 ,  min_w  Phi_e(w) / |S(jw)|^2 ]
```

## The thing you cannot get, and do not need

`n` and `G·d` both enter the measured position at the same node, so `Phi_e` gives you only their sum. No method separates them from this data (three of the four agents derived this independently; formal reference Anderson and Gevers 1982). The `i_fb` channel does not help, because `i_fb = K·e` makes the pair rank 1.

But for your purpose that is fine: you decided to inject at the encoder node, and `Phi_v` is precisely the combined source referred to that node. The decomposition you framed as the goal is neither achievable nor required by the injection you actually want.

## Message 6 (2026-09-22 19:11)

Short answer: "in the loop" is not one choice, it is at least two, and they are not equivalent. Naming the node is the first thing to settle with ASMPT.

## The two candidate nodes

Write the loop with feedforward `u_ff`, sensor noise `n`, input disturbance `d`:

| injected at | effect on the **true** position |
|-|-|
| encoder, `y_meas = y + n` | `y = T·r + G·S·u_ff - T·n` |
| plant input (force), `u -> u + d` | `y = T·r + G·S·u_ff + G·S·d` |

Sensor noise reaches the true position through `T`, disturbance through `GS`. Above crossover `T -> 0` and below it `T -> 1`, so encoder noise mostly corrupts the machine at **low** frequency; `GS` behaves the opposite way. Same injected variance, opposite consequence. ASMPT almost certainly means the encoder node, since that is the physical noise they care about, but it is worth confirming.

## What actually changes versus today

Today `data.py` does `sd.y = sd.y + N(0, sigma_n)` post hoc. That noise corrupts the **training target only**. It never reaches the controller, so the trajectory is unchanged and the noise is a pure measurement artefact.

In the loop, the noise becomes causal. Three consequences follow immediately:

1. **The trajectory itself becomes stochastic.** There is no longer a noise-free `y` to regress against. Your loss has a nonzero floor set by the noise, and the achievable BFR is bounded by it. That is a feature, not a bug: it is the right way to set a defensible acceptance threshold, and it is exactly the "data-derived, never oracle-based" threshold the project stance asks for.
2. **The controller amplifies it and injects it as real force.** `i_fb` gains a `K·S·n` term. Check headroom: logged currents peak around 6.1 A against a 27.9 A limit (D-074), and `|K|` is large at high frequency where `n` lives. This can eat more of that margin than you expect.
3. **Your regressor becomes correlated with the noise.** `u = u_ff + K(r - y - n)` now contains `n`. Training a model on that `u -> y` pair is a *direct closed-loop method with a fixed noise model* `H = 1`, which is output-error, which Forssell and Ljung show is **biased** in closed loop unless the noise model is correct. This is the implication I would raise with ASMPT first, because it silently changes what your parameter recovery converges to. It is also why D-074's "train open loop, validate closed loop" split exists, and injecting in-loop noise partially dissolves that split.

## Two practical traps

**The integrators.** All six controllers are marginally stable by design, integrator poles at `|z| = 1` within zpk rounding (D-074). White noise into an integrator is a random walk: the controller state drifts without bound over a long rollout. Given this project's history with drift, I would expect this to show up as a failure mode before anything else does, and it will look like a model problem when it is a noise-injection problem.

**The variance is rate-dependent.** A per-sample `sigma` is not a physical noise level; the PSD is `sigma^2 · Ts`. The current hard-coded table (`SNR 50 -> 4.5e-4`, Jan's ECC convention, D-078) is a per-sample number that only means something at one sample rate and in the open-loop post-hoc setting. Once the noise passes through `K` at 20 kHz and the pipeline runs at 4 kHz, that table is not transferable. Specify the injected quantity as a **PSD in m²/Hz**, not as a per-sample standard deviation.

## What this implies for the number to inject

It is exactly the load-bearing point from the synthesis: because the simulator now applies `S` itself, the injected quantity must be `Phi_v`, the **source**, not the measured `Phi_e`. Injecting the measured closed-loop variance gives `|S|^4 Phi_v`.

And it resolves the separation question in your favour. Injecting at the encoder node needs only the combined position-referred source `Phi_v = Phi_e/|S|^2`. You do **not** need to split `n` from `G·d`, which is fortunate, since that split is not identifiable from this data anyway.

One methodological note: when you compare baseline against augmented under injected noise, use common random numbers (same seed, same noise realisation) across the two runs, or the comparison inherits the noise variance and you will need many more runs to see the same effect.

## What I would settle before implementing

Confirm with ASMPT that they mean the encoder node rather than a force disturbance, because the two shape the true position through `T` and `GS` respectively and the choice changes which frequencies your augmentation is being asked to survive. Everything else follows from that.
