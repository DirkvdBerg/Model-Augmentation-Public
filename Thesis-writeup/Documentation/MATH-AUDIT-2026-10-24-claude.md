# Math audit of the thesis (Claude), 2026-10-24

**Scope.** Every displayed equation, inline formula and mathematical claim in the printed reading of `Thesis-writeup/Writing/sections/00` to `08` and `99_appendix.tex`, plus `tables/*.tex`. Three checks where they apply: math (hand derivation), citation (against the PDF in `Thesis-writeup/References/`), code (against the main pipeline of `scripts/gantry/gantry_interconnect_dynamic.py`). Read-only: nothing else was edited, built or run.

**Snapshot.** All agents read one copy of the sources, taken at the start of the audit (2026-10-24, branch `Augmentation`, HEAD `e9d3e21`, dirty tree). Modification times of the audited files:

| File | Modified | Lines |
|-|-|-|
| `sections/00_abstract.tex` | 2026-10-11 21:05 | 34 |
| `sections/01_introduction.tex` | 2026-10-20 19:38 | 495 |
| `sections/02_system_baseline.tex` | 2026-10-22 20:26 | 895 |
| `sections/03_augmentation.tex` | 2026-10-23 18:37 | 1955 |
| `sections/04_obc.tex` | 2026-10-24 10:10 | 501 |
| `sections/05_setup.tex` | 2026-10-21 16:42 | 401 |
| `sections/06_results.tex` | 2026-10-22 12:51 | 283 |
| `sections/07_discussion.tex` | 2026-10-18 22:11 | 53 |
| `sections/08_conclusion.tex` | 2026-10-21 10:02 | 148 |
| `sections/99_appendix.tex` | 2026-10-23 19:01 | 871 |
| `util/format.tex` | 2026-10-21 14:36 | 116 |
| `tables/res_*.tex` (22 files) | 2026-10-18 17:31 to 18:07 | |
| `tables/setup_numbers.tex` | 2026-10-21 11:55 | 65 |

Line numbers in this report refer to that snapshot (file:line, with `02` = `02_system_baseline.tex` etc.). Change marks are read in their final reading (`util/format.tex`).

**Agents run.** Six independent sub-agents, one per slice of the handoff (`tasks/handoffs/2026-10-24-claude-math-audit.md`, Section 4.2):

| Agent | Slice | Wrong | Conditional | Notation | Correct | Not checkable |
|-|-|-|-|-|-|-|
| A | 02 lines 1 to 533 (sec:problem, sec:physics, sec:baseline); app:params, app:frf | 0 | 5 | 4 | 26 | 5 |
| B | 02 line 534 to end (sec:lfr, sec:params); app:lfr-details, app:lfr-wellposed, app:combos | 0 | 5 | 7 | 54 | 5 |
| C | 03 lines 197 to 1005 (sec:augmentation, sec:aug_structure, sec:aug_loop); residual paragraph of app:setup_checks | 2 | 5 | 3 | 53 | 4 |
| D | 03 line 1006 to end (sec:aug_closed_loop, sec:aug_encoder, sec:aug_ps2); app:ps_stop | 0 | 7 | 9 | 70 | 6 |
| E | 04 (all); appendix "OBC and experiment details" | 0 | 3 | 9 | 50 | 5 |
| F | 05, 06, quantitative claims of 00, 01, 07, 08; app:splits to app:hyper; tables | 3 | 11 | 8 | 75 | 9 |

The lead then did the cross-slice notation pass (Section "Notation pass") and checked three points across slices: the Gyorok 2026 conclusion (C1), and the OBC space and basis rank in the thesis run logs (two of E's "not checkable" items, now resolved, see below).

**Merged totals.** Wrong 5, conditional 36, notation 35 (40 agent findings; B7 with E4, C8 with D16, E5 with D11, and D15 with B11 and the letter part of F21 merged), correct 330 (328 from the agents plus the two resolved E items), not checkable 31 (the agents listed 34; two resolved by the lead, one concerned unprinted text and was dropped).

**Pipeline and provenance.**
- The thesis run configurations select: `physics_parameterization='reduced'`, `UP_SAMPLE=1`, `ANN_ROUTE_IX` all rows, `ENCODER_INIT=data_lsq`, `NA_NB=30`, `G_INIT=live`, `LOSS_WEIGHTING=servo`, `PARAM_PRIOR=false`, `VALIDATION_MEASURE=sim-NRMSe`, `LBFGS_METERS_GATE=false`, and for the OBC arm `obc_space='affine'`, `obc_ref_successor=True`, `obc_refresh='epoch'`. The text matches these values. Several `RunConfig` defaults in `scripts/gantry/gantry_dynamic/config.py` do not (for example `up_sample=2` at config.py:191, `encoder_init='linear_map'`, `param_prior=True`, `loss_weighting='none'`, `meters_gate=True`). A rerun with defaults would not reproduce the text.
- OBC (lead check): all six run logs in `Thesis-writeup/Results/18-10-2026/thesis-results-4/` print `space=affine`; all 52 logged basis refreshes report rank 11/11, and no rank loss or basis retention occurs.
- Every grey-box run log prints that git was unavailable, so it has no code provenance. That HEAD is the code that produced the 18-10-2026 numbers cannot be confirmed from the logs. The printed run settings all match app:hyper.
- The tables in the snapshot match the CSVs of `Results/18-10-2026`. A newer `Results/24-10-2026` folder exists and was not compared.

## Findings, most severe first

Slice tag in brackets ([A3] = agent A, finding 3). Where text and code disagree, both are reported and neither is declared right.

| # | Severity | Where | Check | Finding | Evidence | Smallest fix |
|-|-|-|-|-|-|-|
| 1 | wrong | [C1] sec:aug_structure, "The dynamic-parallel augmentation structure is chosen, because the orthogonal-by-construction augmentation is defined for this additive structure", 03:404-406 | citation | Gyorok 2026 defines OBC for discrete-time input-output baselines, linear in the parameters, with an additive learning component at the output. It names the extension to state-space baselines as future research. The state-level dynamic-parallel structure of eq:aug_transition is not the structure the paper defines it for. | Gyorok 2026 PDF p2, Eqs. (2), (3); PDF p9, Conclusion ("Future research may be directed at extending the approach for augmenting baseline models in state-space form"), confirmed by the lead in the extracted text. | `is defined for the additive augmentation of input-output models~\cite{gyorok2026obc}, and Section~\ref{sec:obc} extends it to this state-level structure` |
| 2 | wrong | [C2] sec:aug_structure, "...models containing neural networks~\cite{bishop1995neural}", 03:509 | citation | The key `bishop1995neural` is not in `Writing/refs.bib`, so the citation prints as [?]. The `\todo` at 03:510 that asks for the entry is not printed. | Lead check: no `bishop` in any `.bib` under `Writing/`. | Add the bib entry (C. M. Bishop, Neural Networks for Pattern Recognition, Oxford University Press, 1995), or cite only `\cite[Sec.~5.3]{hoekstra2026lfr}`. |
| 3 | wrong | [F1] fig:res_generalisation caption, "Simulation error RMS per test record, mean over X1, X2, Y", 06:176 | code | The figure plots NRMS_e in % per axis (one panel row per axis), not an RMS averaged over the axes. The text at 06:155-161 reads it per axis. | `Thesis-writeup/Code/Figures/res_generalisation.py:36` (`nrms_pct(r["nrms_e_%s" % a])`), :55 (ylabel per axis). | `\caption{$\mathrm{NRMS}_e$ per test record and stage axis of the final model, the baseline and the black box.}` |
| 4 | wrong | [F2] revnote "on $X$ the LTI error is 1.5 to 3.7 times the LPV error", 01:209 (printed in the review build) | math (data) | The LTI/LPV NRMS_e ratio is 1.50 to 3.69 on X1 but 1.02 to 2.55 on X2. "1.5 to 3.7 on X" holds for X1 only. | `Results/18-10-2026/CSV/res_lpv_lti.csv` (I2 X2 0.2789/0.2723; E1 X1 0.9071/0.2461). | `on $X_1$ the LTI error is 1.5 to 3.7 times the LPV error, on $X_2$ 1.0 to 2.5 times` |
| 5 | wrong | [F3] sec:res_obc, "$\mathcal M_{\mathrm U}$ deviates close to double that of $\mathcal M_{\mathrm{OBC}}$", 06:122 | math | The norms are 83.1 and 47.7: a ratio of 1.74, not close to 2. | tables/res_obc_combinations.tex, Norm row. | `deviates about 1.7 times as far as $\mathcal M_{\mathrm{OBC}}$ (norm 83.1 against 47.7\,\%)` |
| 6 | conditional | [E1] sec:obc_ident, "To first order around $\beta^\star$, the one-step error on the tuples is ...", eq:obc_recovery, 04:381-394 | math, code | The expansion uses the Jacobian at $\beta^\star$, but eq:obc_coef makes $\tilde F$ orthogonal to $\Phi$ at $\beta_{\mathrm{lin}}$. "$\Phi^\top\tilde F=0$" holds for the $\Phi$ of eq:obc_recovery only if $\beta_{\mathrm{lin}}=\beta^\star$, or if $(\Phi(\beta_{\mathrm{lin}})-\Phi(\beta^\star))^\top\tilde F$ is neglected as higher order. The minimisation over $\beta$ also holds $\theta_{\mathrm{aug}}$ and $\beta_{\mathrm{lin}}$ fixed, which is not stated. Under these conditions the algebra is right. | Normal equations: $\hat\beta-\beta^\star=(\Phi^\top\Phi)^{-1}\Phi^\top(\Xi-\tilde F)$; eq:obc_coef gives $\Phi(\beta_{\mathrm{lin}})^\top\tilde F=0$. Code builds the basis at the current $\beta$: interconnect.py:1407, obc_gantry.py:330. The revnote at 04:418-425 reports a mismatch between the predicted and observed bias. | Replace "To first order around $\beta^\star$, the one-step error on the tuples is" with `To first order around $\beta^\star$, with $\Phi_\Gamma$ evaluated at $\beta_{\mathrm{lin}}=\beta^\star$ and $\theta_{\mathrm{aug}}$ held fixed, the one-step error on the tuples is` |
| 7 | conditional | [D1] sec:aug_ps2, "PS only changes the starting point of minimising $V$", 03:1966-1967 | math, code | Holds only after $i_1$. During S1, eq:ps2_alt inserts an $\mathcal A_{\mathrm{S1}}$ step on $\theta^a_{\mathrm{enc}}$ between $\mathcal A_V$ steps, so the $V$ iterates for $i<i_1$ change. The selection also scores the initial and S1-era models, so the selected model need not descend from $\vartheta_{\mathrm{PS}}$. | eq:ps2_alt; interconnect.py:1045-1046 (after_step after every optimiser step); interconnect.py:978 (validation before training) and :902 (best of all validations). | `PS changes the iterates of the minimisation of $V$ during S1 and its starting point after S1, not what is minimised.` |
| 8 | conditional | [D2] sec:aug_ps2, "The expected squared error is minimised by the conditional mean", 03:1761-1762 | math | The conditional mean minimises the expected squared error over all functions of $\chi_\tau$. Eq. (28) optimises over the restricted class $p(\theta_{\mathrm p},\psi_a(\theta_{\mathrm{enc}},\chi_\tau))$, which has an $n_{\bar x}$-dimensional bottleneck; its minimiser equals $\mathbb E[e_\tau\mid\chi_\tau]$ only if that function is in the class. The target is also scaled by the batch-dependent $\sigma_e$ and moves with $\vartheta$. | $L_2$ projection result; Hefny 2015 Sec. 2 (PDF p. 3) states the S1A target for an unrestricted regression. | `Over all functions of $\chi_\tau$, the expected squared error is minimised by the conditional mean $\mathbb E[e_\tau\mid\chi_\tau]$; the decoder approximates it through $\bar x_{\tau\mid\tau}$.` |
| 9 | conditional | [B2] sec:params, "Since the matrices depend on ten independent combinations ... only these can be identified", and "we estimate $\theta_{\mathrm{base}}$", 02:793-826 (goal 02:694-698) | math | The argument at 02:735-772 is structural: it uses arbitrary forces and initial states, so all $(\tilde x,\ddot q)$ and at least three distinct $Y$. The upper bound ("only these") holds regardless. Uniqueness from the finite closed-loop records $\mathcal D_N$ additionally needs informative data, which is not stated. | The identity $(M-M')\ddot q+(C-C')\dot q+(K-K')q=0$ forces equality only on the span of the visited $(q,\dot q,\ddot q)$ and $Y$. | After "only these can be identified \cite[Def.~7]{ovchinnikov2021computing}" add `They are determined uniquely by $\mathcal D_N$ if the records excite all three coordinates at at least three distinct $Y$.` (Settling quantity: rank 10 of $\partial\hat y/\partial\theta_{\mathrm{base}}$ stacked over $\mathcal D_N$.) |
| 10 | conditional | [E2] sec:obc_map, "Like the regressor, the stacked Jacobian must have full column rank.", 04:186, used by eq:obc_coef (04:267-279) | math, code | eq:obc_coef inverts $\Phi_\Gamma^\top\Phi_\Gamma$ ($11\times11$), so it needs rank 11: full column rank of $\Phi$ and $\Gamma\notin\ran\Phi$. Only the rank of $\Phi$ is stated. The code demands rank 11 and aborts after two consecutive rank losses. In the thesis runs rank 11 held at all 52 refreshes (lead check of the logs). | obc_gantry.py:396 (expected rank 10 plus 1 for affine); obc.py:460-472; `Results/18-10-2026/thesis-results-4/*.out` ("basis rank 11/11"). | After "must have full column rank." add `With the offset column, $\Phi_\Gamma$ must have rank $11$, so $\Gamma\notin\ran\Phi$.` |
| 11 | conditional | [C3] sec:aug_loop, "The linearised RK4 step has a pole at $z=1$ on each of these axes", 03:755-756 | math | Always true for $X$ (no entry of $F$ depends on $X$). For $Y$ only where $\ddot X=\ddot\Theta=0$: $\partial(M(Y)^{-1}f)/\partial Y=-M^{-1}M'(Y)\ddot q$ with $M'\ddot q=[-m_h\ddot\Theta,\;-m_h\ddot X+2m_hY\ddot\Theta,\;0]^\top$, which moves that eigenvalue when the operating point accelerates. The drift argument is unaffected. | Derivation from eq:base_ss, eq:Msplit, eq:mass_matrix ($M_{1,12}=-m_h$, $M_{2,22}=m_h$). | `The RK4 step, linearised at rest, has a pole at $z=1$ on each of these axes,` |
| 12 | conditional | [D3] sec:aug_closed_loop, "In closed-loop the feedback controller is able to correct this offset", 03:1251-1252 | math | Correct only if the closed loop (system with $K_{\mathrm{fb}}$, and the model loop eq:aug_residual_loop) is incrementally exponentially stable over the whole $Y$ range. Neither stated nor shown. | Beintema 2023 Cond. 1 (PDF p. 7) is the property being restored. | Append `if the closed loop is stable over the range of $Y$`. (Settling quantity: spectral radius below 1 of the linearised closed loop over the $Y$ range at $\theta^0$ and the final $\theta$.) |
| 13 | conditional | [A1] sec:baseline, "$\Theta=(X_1-X_2)/L_b$", 02:370-373, with eq:Ptransform 02:396-409 and "assumes $\cos(\Theta)\approx1$ and $\sin(\Theta)\approx0$", 02:419 | math, citation | The linear map $q=P^{-\top}\qact$ ($X_1=X+\tfrac{L_b}{2}\Theta$) is Garcia's kinematics only under $\sin\Theta\approx\Theta$; Garcia defines $\Theta=\sin^{-1}((X_1-X_2)/L_b)$. Only $\sin\Theta\approx0$ is stated, which applied to Garcia Eq. (2) would give $X_1=X_2$. Two small-angle orders are used (zeroth in $M$, first in the kinematics) and only one is stated; the adaptation of Garcia Eq. (1) is not stated. | Garcia 2013 PDF p3, Eqs. (1), (2), Sec. 2.3.1 (Eq. 9). Code uses the linear map: gantry_ss.py:120-136. | `assumes $\cos(\Theta)\approx1$ and $\sin(\Theta)\approx\Theta$, and drops the terms of order $\Theta$ in the inertia matrix` |
| 14 | conditional | [A2] sec:problem, "We use the SUBNET method of Beintema et al. ... where each window is simulated with the controller $K_{\mathrm{fb}}$", 02:292-295 | citation | SUBNET is an open-loop method; its consistency result assumes the input is independent of the noise, which 02:211 says does not hold here. Simulating each window in closed loop is this work's adaptation (code attributes it to Kessels 2025), but the sentence reads as part of the cited method. | Beintema 2023 PDF p7, Theorem 4; closed_loop.py:1-29. | `We adapt the SUBNET method of Beintema et al.~\cite{beintema2023subnet} to closed loop: the model is trained on short windows of $\mathcal D_N$, and each window is simulated with the controller $K_{\mathrm{fb}}$.` |
| 15 | conditional | [C4] app:setup_checks, "This force error is the last term of \eqref{eq:aug_residual_split}.", 99:760 (with 99:742-747) | math, code | $u_{\mathrm{act},k}$ in eq:aug_residual_split is the recorded force at $T_s$, which by eq:setup_decim is the block mean over the hold, and $\check u$ uses the filtered $y_k$. The last term is the controller-rate error plus the within-hold force change (reported separately as \SetupKholdMed) plus any filter effect. The two reported medians are each a part of the term. | eq:setup_decim (99:687); data.py:154-171 (block mean), data.py:174-184 (filtered $y$); closed_loop.py:235-239. Settling quantity: RMS of $u_{\mathrm{act},k}-\check u_{\mathrm{act},k}$ on the training records. | `The last term of \eqref{eq:aug_residual_split} contains this force error and the force change within one hold.` |
| 16 | conditional | [B1] app:lfr-wellposed, "$r_0=m_{\Delta,0}/b_0$, $\lvert r_0\rvert<1$", 99:280-285; used by eq:mdelta_map 02:871-888 | math, code | The construction needs $m_{\Delta,0}\neq0$. If $m_{\Delta,0}=0$ then $s=0$ and $m_\Delta\equiv0$ for every $\beta$, although $\lvert r_0\rvert<1$ holds. The code refuses a zero start. The nominal $m_{\Delta,0}=-0.5$ kg, so the thesis runs are unaffected. | blocks.py:1318-1319 (raises on a zero combination); docstring blocks.py:1273-1274. | `r_0=\frac{m_{\Delta,0}}{b_0},\quad 0<\lvert r_0\rvert<1,` |
| 17 | conditional | [F4] sec:res_obc, "Both models have deviated further from the true combinations, than their initial values", 06:121; same at 06:203 | math (data) | True for the norm of $\delta$ only (30.0 to 83.1 and 47.7). Per combination OBC ends closer than its start for 6 of 10 ($k_{b,\Sigma}$, $c_{b,\Sigma}$, $m_h$, $m_\Sigma$, $J_{\mathrm{eff}}$, $d$); U for 3 of 10. | tables/res_obc_combinations.tex. | `In norm, both models end further from the true combinations than their start; the increase comes from the actuator dampers $c_{g1}$, $c_{g2}$, $c_y$.` |
| 18 | conditional | [F5] sec:discussion, "The truth overlap of 0.59 means $\Phi^\top\Xi\neq0$, so the recovery condition of Section IV-C is not met", 06:205-209 | math (data) | The recovery condition is on the training tuples of eq:obc_tuples; 0.59 is the test-tuple value. The training value is 0.51. The conclusion is unchanged. | res_obc_overlap.csv (train 0.508, test 0.594); tables/res_obc_combinations_train.tex; 04:406-409. | `The missing dynamics have a truth overlap of 0.51 on the training tuples (0.59 on the test tuples), so $\Phi^\top\Xi\neq0$ ...` |
| 19 | conditional | [F6] tab:res_obc OBC column: network overlap 0.13, learned pole 212, damping 0.058 (also tab:app_obc_full R1, R2), footnote "update 19500 of 33400" | code (data) | The overlap, pole and damping of R1 and R2 are from update 18200; the combination errors and NRMS_e are from update 19500. Two checkpoints in one column under one footnote. | res_obc_overlap.csv ("preliminary, update 18200 of 33400"); res_obc_combinations.csv (19500). | Footnote: `R1, R2 preliminary: combinations and NRMS$_e$ at update 19500, overlap and learned mode at update 18200`, or recompute at one checkpoint. |
| 20 | conditional | [F8] sec:conclusion, "OBC makes the estimates less sensitive to the start than unconstrained estimation, without loss of accuracy", 08:87-88 | math (data) | Test NRMS_e OBC 6.4/6.4/4.6 against U 5.7/5.6/4.8 (X1/X2/Y); on the multisine records 9.7/9.5 against 8.8/8.4 on X. OBC is 0.7 to 1.1 points worse on X, better on Y. | tables/res_obc_combinations.tex; tables/res_all_splits.tex (test block). | `... with the same accuracy within one percentage point, ...` |
| 21 | conditional | [E3] sec:obc_map, "The offset does not fully prevent negation ..., where the neural network cancels the baseline response", 04:225-227 | math, citation | With "part of" struck, the sentence reads as full cancellation. On the tuple stack at $\bar x=0$ and $\beta=\beta_{\mathrm{lin}}$ the $\Gamma$ column excludes full cancellation ($\Phi\beta_{\mathrm{lin}}+\Gamma\in\ran\Phi_\Gamma$). The sentence holds for partial cancellation or off the stack. | Kessels 2025 Sec. 5.3.1.3 (PDF p. 193): networks "(partially) negate" the FP terms. | `where the neural network partly cancels the baseline response` |
| 22 | conditional | [C5] sec:aug_structure, "The inputs to the neural network are normalised to zero mean and unit variance", 03:507-508 | math | The network also takes $\bar x$, which is not normalised (03:563-566, $\hat x^{\mathrm n}=[\tilde x^{\mathrm n\top}\;\bar x^\top]^\top$). True only for the $\tilde x$ and $u_{\mathrm{act}}$ inputs. | 03:563-571; model.py:179. | `The baseline states and actuator forces are normalised to zero mean and unit variance before they enter the network, which ...` |
| 23 | conditional | [B4] sec:params, "ten independent combinations of $\theta$ (Appendix~\ref{app:combos})", 02:793-795 | math | app:combos shows only that $\theta_{\mathrm{base}}$ is recoverable from the matrices, not that the ten are independent functions of $\theta$. The claim holds: the Jacobian with respect to $(k_{b1},c_{g1},c_{g2},c_y,c_{b1},m_h,m_b,m_2,J_b,d)$ is triangular with determinant $-1$. | Hand derivation from 02:805-807. | Append to app:combos: `Each combination contains a parameter that no earlier one contains ($k_{b1},c_{g1},c_{g2},c_y,c_{b1},m_h,m_b,m_2,J_b,d$), so the Jacobian of $\theta_{\mathrm{base}}$ with respect to these is triangular with determinant $-1$ and the ten combinations are independent.` |
| 24 | conditional | [B3] sec:params, "also have the same $\dot q$ and $\ddot q$, since both are derivatives of $q$~\cite[Def.~7]{ovchinnikov2021computing}", 02:735-739 | citation | Def. 7 defines identifiability; it makes output derivatives admissible information but does not state that equal outputs give equal derivatives. The claim itself is trivially true. | Ovchinnikov 2021 PDF p. 3, Definition 7. | `also have the same $\dot q$ and $\ddot q$, since both are derivatives of $q$; identifiability may use the outputs, the inputs and their derivatives~\cite[Def.~7]{ovchinnikov2021computing}.` |
| 25 | conditional | [B5] sec:lfr, "making the baseline a self-scheduled LPV model~\cite{drenth2025lpvlfr}", 02:618-619 | citation | In Drenth the scheduling map is estimated jointly with the model. Here it is the fixed selection $Y=q_3$. The difference is stated only in the unprinted wip block of app:combos (99:324). | Drenth 2026 PDF p. 2 and p. 3, Sec. 2.3. | `making the baseline a self-scheduled LPV model with a known scheduling map, as in~\cite{drenth2025lpvlfr} but with $Y$ selected from the state instead of learned.` |
| 26 | conditional | [D4] sec:aug_encoder, "We use Xavier initialisation for $W^a$, following~\cite[Eq.~(31)]{hoekstra2026lfr}", 03:1521-1525 | citation | Hoekstra Eq. (31) Xavier-initialises all weights and biases of a separate added-state encoder. Here only the linear part $W^a$ is Xavier; the added-state rows of $\tilde\psi$ start at zero (eq:aug_init). The adaptation is not stated. Code matches the thesis. | Hoekstra 2026 LFR PDF p. 9, Eq. (31); pre_encoder.py:433, :454, :459. | `adapted from~\cite[Eq.~(31)]{hoekstra2026lfr}, where all weights and biases of the added-state encoder are Xavier initialised` |
| 27 | conditional | [D5] sec:aug_encoder, "fitted to the simulated states by least squares~\cite[Eq.~(33)]", eq:aug_enc_fit, 03:1408-1416 | citation | Encoder paper Eq. (33) fits $W^b$ only, without offset, for an LTI baseline. Eq. (26) adds the offset $b$. The offset is an unstated adaptation (the LTI contrast is stated later). | Hoekstra 2026 encoder PDF p. 5, Eqs. (33), (34); model.py:435-437 (intercept). | `by least squares with an offset, adapted from~\cite[Eq.~(33)]{...}` |
| 28 | conditional | [D6] sec:aug_closed_loop, "Following Beintema et al., we use an encoder ... from the past inputs and outputs $\chi_\tau$~\cite[Sec.~3.2]", 03:1110-1111 | citation | Beintema's encoder input includes the current output $y_t$; $\chi_\tau$ stops at $\tau-1$ as in Hoekstra (22e). The $\tau-1$ convention is stated with the correct source at 03:1179-1180, so only the attribution of $\chi_\tau$ is loose. | Beintema 2023 PDF p. 5, Eqs. (12), (13b); Hoekstra 2026 LFR PDF p. 8, Eq. (22e). | Replace "the past inputs and outputs $\chi_\tau$" with `past inputs and outputs`. |
| 29 | conditional | [C6] sec:aug_structure, "The positions are also corrected, because within one sample period an unmodelled force would change both the velocity and the position~\cite[Table~1]{hoekstra2026lfr}", 03:500-502 | citation | The physics reason is correct, but Table 1 only shows the S-DP structure; it supports "positions are corrected", not "because". | Hoekstra 2026 LFR PDF p3, Table 1, row S-DP. | `The positions are also corrected~\cite[Table~1]{hoekstra2026lfr}, because ...` |
| 30 | conditional | [C7] sec:aug_structure, "as in~\cite[Sec.~5.3]{hoekstra2026lfr}" and "We include the means in $f^{\mathrm n}_{\mathrm{base}}$ ...", 03:509, 03:656-657 | citation | The zero-mean adaptation is stated. A second adaptation is not: Hoekstra takes $\sigma_x$ from a baseline simulation, the thesis takes $\mu_x,\sigma_x$ from $P^{-\top}y$ and finite differences of the records. | Hoekstra 2026 LFR PDF p9, Sec. 5.3, Eq. (28); data.py:341-373. | Optional: `with $\sigma_x$ from the records instead of a baseline simulation` after the Sec. 5.3 cite. |
| 31 | conditional | [F11] sec:setup_metrics, "The primary metric is the NRMS of~\cite[Eq.~(43)]{beintema2023subnet}, adapted by using the servo error as output", 05:342-343 | citation | Beintema Eq. (43) is one scalar over the output vector with one $\sigma_y$. The thesis normalises each axis by its own $\sigma_{e,j}$; that second adaptation is not stated. | Beintema 2023 PDF p. 8, Sec. 5.2, Eq. (43). | `adapted by using the servo error as output and normalising each stage axis separately` |
| 32 | conditional | [F14] revnote "Hoekstra et al.\ show the validation error during training for every model~\cite[Fig.~8]{hoekstra2026lfr}", 06:76-77 | citation | Fig. 8 shows the validation loss of selected models only (S-DP, S-DSO, ANN-SS), configuration (a). | Hoekstra 2026 LFR PDF p. 12, Sec. 6.3, Fig. 8 caption. | `for selected models~\cite[Fig.~8]{hoekstra2026lfr}` |
| 33 | conditional | [F7] app:setup_checks, "The block mean is the exact input of a zero-order-hold model at $T_s$", 99:700 | math, code | The block mean is the ZOH value with the same force impulse over each hold. The end-of-hold state equals the 20 kHz response only for the velocity of a pure mass. The code itself calls it impulse equivalence. | data.py:154-171 (docstring "impulse equivalence"; block mean at :171). | `The block mean is the zero-order-hold input with the same force impulse over each hold of $T_s$.` |
| 34 | conditional | [F12] sec:setup_metrics, "At standstill $y^\circ=r$, so the floor equals the standstill servo error that the noise model was fitted to", 05:366-367 | math | Holds for standstill records without multisine. I2 and E1 are standstill records with multisine, where $y^\circ\neq r$. | tab:setup_records (I2, E1: standstill, multisine yes). | `On standstill records without multisine, $y^\circ=r$, so ...` |
| 35 | conditional | [F13] revnote "On the servo error the BFR equals $100-\mathrm{NRMS}_e$", 05:347 | math | Exact without clipping. With the usual $\max(\cdot,0)$, the baseline Y (116.6 %) gives BFR 0, not $-16.6$. | $\hat y-y=\hat e-e$ since $r$ cancels. | `equals $100-\mathrm{NRMS}_e$ (without clipping at zero)` |
| 36 | conditional | [F9] sec:setup, E6 "131\,\% acc." with "relative to the machine limits", 05:53-54; fig:setup_coverage "in \% of the machine limits", 99:649, 667 | math | 131 % of a limit is outside the limit. The reference is the datasheet maximum, which the E6 move exceeds, so "limit" has no single meaning. | Results/setup_coverage.csv (E6 aX 131.05 %); tdg_config.m:86 ("datasheet maximum"). | `relative to the datasheet maxima of the machine` (05:53 and the figure caption) |
| 37 | conditional | [F10] tab:hyper, "SVD cutoff ($N$ windows, $m$ regressors) $\max(N,m)\,\epsilon$", 99:820 | code | The cutoff is relative: singular values below rcond times the largest are treated as zero (standardised regressors). | model.py:434-435 (`rcond = torch.finfo(torch.float64).eps * max(Phi.shape)` passed to `torch.linalg.lstsq`). | `$\max(N,m)\,\epsilon\,\sigma_{\max}$` |
| 38 | conditional | [D7] sec:aug_ps2, "to keep the cost independent of the scale of $\bar x$", 03:1903-1904 | math | The denominator of eq:ps2_s2 sums the variance over all $j$, so the cost is invariant to a common scaling of $\bar x$, not to per-state scaling. | Substitute $c\bar x$ (invariant) and $\operatorname{diag}(c_j)\bar x$ (not invariant) in eq:ps2_s2. | `a common scale of $\bar x$` |
| 39 | conditional | [A3] app:frf, caption of fig:res_frf_full, "Geometric mean over $Y=-0.39$ to $0.39$~m", 99:448 | code | The mean is over seven discrete positions, not the interval. | Thesis-writeup/Code/Evaluation/eval_frf.py:76 (seven Y); res_frf_full.py:27-30. | `Geometric mean over the seven positions $Y\in\{0,\pm0.15,\pm0.30,\pm0.39\}$~m; shaded: range of the final model over these positions.` |
| 40 | conditional | [A4] app:frf caption, "linearised at rest", 99:448 | code | The system curve is the linearisation with the Coulomb friction removed; the final model is linearised at rest with its learned terms, and rest need not be its equilibrium (the code prints a rest drift). The caption reads as one common operation. | _setup_common.py:136; eval_frf.py:45-53 (drift at :47). | `..., linearised at rest; the system without its Coulomb friction.` |
| 41 | conditional | [A5] app:frf caption, "Bode magnitude from the stage forces ... to the stage positions", 99:448 | code | The final model is evaluated as a discrete-time response at $z=e^{j2\pi fT_s}$, the system and baseline as continuous-time $G(j2\pi f)$. For a rigid mode the hold effect is about $-0.35$ dB at 600 Hz (hand calculation, $\cos x\,x^2/\sin^2x$, $x=\pi fT_s$). | eval_frf.py:62; _setup_common.py:150. | Add: `The final model is evaluated at $z=e^{j2\pi fT_s}$, the system and baseline at $s=j2\pi f$.` |
| 42 | notation | [B6] sec:lfr / app:lfr-details, LFR output $q$, eq:lfr_G with $C_y=[I_3\;0]$, 99:126-133, 99:182 | code | Output frame differs. Text: the output of $G$ is $q$. Main path: the output block gives $\qact=P^\top q$, the measured output of eq:system (and eq:aug_transition row 3 uses $P^\top q$). Equivalent through the invertible $P^\top$. Reported, not decided. | gantry_ss.py:135-136 (`Cd[:, :3] = P.T`); model.py:132 (Linear_Output_Block). | Either `C_y=\begin{bmatrix}P^\top&0\end{bmatrix}` with output $\qact$, or keep $q$ and add `the measured output is $\qact=P^\top q$ \eqref{eq:Ptransform}`. |
| 43 | notation | [E10] sec:obc_map, eq:obc_coef and eq:obc_corr, $\Phi_\Gamma=[\Phi\ \Gamma]$, $\theta_{\mathrm{aux}}\in\R^{11}$ | code | Eleventh column differs, outputs agree. Text: $\Gamma$ = stacked $\gamma=f^{\mathrm n}_{\mathrm{base}}(\beta_{\mathrm{lin}})-\varphi\beta_{\mathrm{lin}}$. Code: the stacked $f^{\mathrm n}_{\mathrm{base}}(\beta_{\mathrm{lin}})$ itself, with column equilibration. Same range, so $\tilde F$ and $f^{\mathrm n}_{\mathrm{obc}}$ are identical; only the coefficients differ ($\theta^{\mathrm{code}}_{1:10}=\theta_{\mathrm{aux},1:10}-\beta_{\mathrm{lin}}\theta_{\mathrm{aux},11}$). Matters only if $\theta_{\mathrm{aux}}$ is reported. | obc_gantry.py:335-338, :307; obc.py:209-211, :248. | Code note only. If coefficients are reported, state the column used. |
| 44 | notation | [D13] eq:ps2_s2 denominator vs code | code | Thesis divides by the sum of squared deviations over $\mathcal B_{\mathcal P}$; code divides by the mean unbiased variance, a constant factor $(\lvert\mathcal B_{\mathcal P}\rvert-1)/\lvert\mathcal B_{\mathcal P}\rvert$ per batch. Changes neither the minimiser nor the selection. | ps2_opening.py:227 (`T.var(0).mean()`). | Code note: `T.var(0, unbiased=False)` matches eq:ps2_s2 exactly; or leave both. |
| 45 | notation | [B7, E4] $\gamma$: safety margin $\gamma\in(0,1)$ in eq:mdelta_map (02:872-877, 02:890; 99:277) and offset function $\gamma:\R^{10}\times\R^6\times\R^3\to\R^6$ in eq:obc_offset (04:207-215) | notation | One symbol, two meanings; $\Gamma$ in eq:obc_coef stacks the second. | Cross-slice (both agents). Code calls the margin `M_DIFF_RHO` (blocks.py:1301). | Rename the margin to `\kappa\in(0,1)` in eq:mdelta_map, 02:877, 02:890 and 99:277 (keeps Gyorok's $\gamma,\Gamma$ in Section IV). Not $\rho$: $\rho$ is the overlap of eq:obc_overlap. |
| 46 | notation | [E5, D11, F21] $N$ and $\mathcal D_N$: record length in $\mathcal D_N=\{u_{\mathrm{act},k},y_k\}_{k=1}^N$ (02:180) and eq:setup_rms; tuple count $N=\lvert\mathcal I\rvert$ (04:275, 276, 278, 377, 463); number of windows (tab:hyper SVD row, 99:820). $\mathcal D_N$ is one sequence in Section II but "multiple records" at 03:1096, and in eq:aug_loss a start $\tau$ does not identify its record $r$ in $W_r$. | notation | Three meanings of $N$; $\mathcal D_N$ defined as one sequence but used as a set of records. | 02:180; 03:1096, eq:aug_loss (03:1144-1161); 04:275; obc_gantry.py:135 (tuples drop 2 samples at the start and 3 at the end of each record). | Define $\mathcal D_N$ as a set of records in Section II; in eq:aug_loss sum over $(r,\tau)\in\mathcal W$ and write $y^{\mathrm n}_{r,k}$; rename the tuple count `N_{\mathcal I}` (and `6N` to `6N_{\mathcal I}` at 04:276, 278, 377, 463) and the window count `N_w`. |
| 47 | notation | [F17] added-state count: $n_{\bar x}$ in 03, 05:305, tab:hyper (99:821); $n_a$ in the tab:app_obc_na0 caption (99:501) and every res_structure* / res_obc_na0 row label | notation | Two symbols for one quantity; $n_a$ also collides with the absorber subscript $a$ ($m_a$, $k_a$, $c_a$, $q_a$, $f_a$). | snapshot tables; 03:381. | Replace $n_a$ by $n_{\bar x}$ in the caption and the table generators. |
| 48 | notation | [F16] "Norm" row of tab:res_obc, tab:app_obc_full, tab:app_obc_na0; 06:201-202 | math, code | The Euclidean norm of the ten $\delta_j$ is used but not defined; eq:setup_delta defines only $\delta_j$. | res_obc_combinations.py:62; check $\sqrt{9\cdot10^2+0.5^2}=30.0$. | After eq:setup_delta: `The combination error of a model is $\lVert\delta\rVert_2$ over the ten combinations.` |
| 49 | notation | [D14] $\theta_{\mathrm p}$ in eq:ps2_s1 (03:1716), eq:ps2_nu, app:345-354 | notation | The definition $\theta_{\mathrm p}=(W_{\mathrm p,1},b_{\mathrm p,1},W_{\mathrm p,2},b_{\mathrm p,2})$ is struck in both places, so $\theta_{\mathrm p}$ is used but never defined; the decoder width is no longer stated. | app:349-352 (struck). | After the decoder display: `with parameters $\theta_{\mathrm p}=(W_{\mathrm p,1},b_{\mathrm p,1},W_{\mathrm p,2},b_{\mathrm p,2})$ and $n_h$ hidden neurons`. |
| 50 | notation | [F15] eq:setup_rms, 05:353-361 | math | The sum $k=n+1,\dots,N-1$ with $1/(N-n-1)$ and "$n+1$ the encoder window length" is consistent only with 0-based samples $0,\dots,N-1$; Section II indexes $\mathcal D_N$ from $k=1$. Code is 0-based and matches the 0-based reading. | closed_loop.py:756 (`k0 = max(na, nb)`), :801. | `where samples are indexed $k=0,\dots,N-1$`, or change the sums to $\sum_{k=n+2}^{N}$. |
| 51 | notation | [D9] $\sigma_e$ and $e_\tau$: batch RMS of the S1 residual and the stacked residual (eq:ps2_s1 03:1719, 03:1746, app:339-340) versus servo-error std $\sigma_{e,r,j}$ (03:1171-1173), $\sigma_{e,j}$ (eq:setup_rms) and servo error $e=y-r$ (Section V) | notation | Same stem, two meanings. | ps2_opening.py:189. | Rename the S1 symbols, e.g. `e^{\mathrm{S1}}_\tau` and `\sigma_{\mathrm{S1}}`, in eq:ps2_s1, eq:ps2_nu, the Fig. ps2_windows caption and app:339 (not $\rho$, which is the overlap). |
| 52 | notation | [E6] sec:obc_map, "uses the recorded actuator force $u_i$", 04:251-252 | notation | $u$ is the generalised force $P\uact$; the recorded actuator force is $u_{\mathrm{act}}$, which eq:aug_norm normalises and the code uses. | 02:410; obc_gantry.py:141. | `tuple $(\tilde x_i,\bar x_i,u_{\mathrm{act},i})$ uses the recorded actuator force $u_{\mathrm{act},i}$` |
| 53 | notation | [C8] sec:aug_structure network paragraph vs display after eq:aug_norm_maps, 03:489-492 and 03:672-673; also [D16] Fig. ps2_windows caption, 03:1644-1646 | math | The paragraph calls the outputs $g_{\mathrm{aug}}$, the display and eq:aug_norm_maps $g^{\mathrm n}_{\mathrm{aug}}$; $f^{\mathrm n}_{\mathrm{aug}}=T_xf_{\mathrm{aug}}$ and $g^{\mathrm n}_{\mathrm{aug}}=g_{\mathrm{aug}}$ are never written, and $f^{\mathrm n}_{\mathrm{aug}}$ is used in eq:aug_norm_maps before "We scale $f_{\mathrm{aug}}$ only by $T_x$". The Fig. ps2_windows caption mixes $g_{\mathrm{aug}}$, $\hat x^{\mathrm n}$ and the unnormalised $u_\tau$. | $T_x(\tilde x_{k+1}-\mu_x)=T_x(f_{\mathrm{base}}-\mu_x)+T_xf_{\mathrm{aug}}$; eq:ps2_s2. | After eq:aug_norm_maps: `with $f^{\mathrm n}_{\mathrm{aug}}=T_xf_{\mathrm{aug}}$ and $g^{\mathrm n}_{\mathrm{aug}}=g_{\mathrm{aug}}$`; in the Fig. ps2_windows caption write $g^{\mathrm n}_{\mathrm{aug}}$ and $u^{\mathrm n}_\tau$. |
| 54 | notation | [B10] initial values: $\theta^0_{\mathrm{base}}$ (02:862) vs $m_{\Delta,0}$, $b_0$, $r_0$ (02:883, 99:277-285) | notation | "Initial" is a superscript 0 on $\theta_{\mathrm{base}}$ but a subscript 0 on $m_\Delta$, $b$, $r$; subscript 0 also means the $Y$-free part $M_0$ and the offset $\eta_0$, and $m_{\Delta,0}$ has the shape of the entry notation $M_{0,jk}$. | 02:862-866, 02:883, 99:277-315. | One mark: `m_\Delta^0`, `b^0`, `r^0`. |
| 55 | notation | [E7] sec:obc_construction, "$\hat y_k=\varphi(x_k)\theta_{\mathrm b}$" and "where $x_k$ is the vector of past measured outputs ...", 04:130, 146 | notation | $x$ is the system state of eq:system; here $x_k$ is Gyorok's lagged input-output vector. | 02:166; Gyorok 2026 PDF p. 2, Eq. (1). | `x^{\mathrm{IO}}_k` at 04:130 and 146 (not $z_k$: $z$ is the LFR latent signal of eq:delta). |
| 56 | notation | [D15, B11, F21] letter reuse: $m$ (layer input size 03:1519; batch mean $m_j$ in eq:ps2_s2; regressor count, 99:820) next to masses $m_h,m_\Sigma,m_\Delta$; $a$ (Xavier bound eq:aug_init; superscript of $W^a$, $\theta^a_{\mathrm{enc}}$; $a=L_bm_\Delta/2$ at 99:244) next to the latent signals $a_1,a_2$ (99:96-114); $j$ (stage axis in eq:setup_rms, eq:setup_K1, eq:setup_noise; combination index in eq:setup_delta); $s$ (tanh scale, eq:mdelta_map) vs $s_j$ (eq:setup_delta); $H$ (S1 horizon) vs $H_j$ (Butterworth, eq:setup_noise) | notation | Locally defined each time, but several letters carry two or three meanings across sections. | As cited. | E.g. `\mu_{\bar x,j}` for the mean in eq:ps2_s2, `a_{\mathrm X}` for the Xavier bound, `\mu=L_bm_\Delta/2` at 99:244, index combinations by $i$ in eq:setup_delta, `n_\phi` for the regressor count. |
| 57 | notation | [C9] $\hat x_k$ (augmented state) vs $\hat x^{\mathrm{fb}}_k$ (model-loop controller state), 03:382, 831, 845; $x^c_k$ in eq:aug_residual_loop and eq:aug_cl_step without $\vert\tau$ | notation | The hat means "augmented state" on $\hat x_k$ but "model loop" on $\hat x^{\mathrm{fb}}$, $\hat y$, $\hat u$; $x^c$ depends on $\tau$ but carries no $\vert\tau$. | 03:382, 845-846, 914, 986. | Optional: `x^{c}_{k|\tau}` in eq:aug_cl_step; state the hat convention once. |
| 58 | notation | [C10] sec:aug_structure, "$\mu_x\in\R^6$ ... sample means of $\tilde x$", 03:546 | math | $\tilde x$ is the model state and is not measured; $\mu_x$ is the mean of $[P^{-\top}y;\ \text{finite differences}/T_s]$. | data.py:344-372. | `sample means of $\tilde x$ estimated from $\mathcal D_N$` |
| 59 | notation | [D10] app:ps_stop, $\mathcal W_i$ (last $w$ checks), app:394-405 | notation | $\mathcal W$ is the set of training-window starts of eq:aug_loss; $\mathcal W_i$ reuses it for a set of update indices. | eq:aug_loss vs the $i_1$ display. | Rename to `\mathcal L_i` (last $w$ elements of $\mathcal J_i$). |
| 60 | notation | [D12] sec:aug_encoder, "We only fit the linear map $W^b$.", 03:1435 | math | eq:aug_enc_fit fits the affine pair $(W^b,b)$; so does the code. | model.py:435-437, :446-449. | `We only fit the affine map $W^b\chi_\tau+b$.` |
| 61 | notation | [D8] sec:aug_closed_loop, sentence beginning "keeps the computation per gradient step independent of the record length", 03:1115-1116 | math (statement) | The printed sentence has no subject, so it is unclear whether the windows, the encoder or both keep the cost independent of $N$. Beintema attributes it to the truncated windows together with the encoder. | Beintema 2023 PDF pp. 4-5, Eq. (9). | `Together with the short windows, the encoder keeps the computation ...` |
| 62 | notation | [B9] sec:lfr, "the matrices in \eqref{eq:base_ss} depend rationally on $Y$", 02:645-646 | notation | eq:base_ss is written as $F(\tilde x,u)$; no matrices appear in it. The math is correct for $A(Y)$, $B(Y)$. | eq:base_ss, eq:Msplit. | `the state and input matrices of \eqref{eq:base_ss}, which contain $M(Y)^{-1}K$, $M(Y)^{-1}C$ and $M(Y)^{-1}$, depend rationally on $Y$.` |
| 63 | notation | [B8] eq:mdelta_map, value of the margin, 02:877 | code | The value is not printed; code uses 0.99 (tab:hyper lists 0.99 per agent F). Consistent with $\gamma\in(0,1)$. | blocks.py:1301, :1394-1395. | Optional: state the value at 02:877 or point to tab:hyper. |
| 64 | notation | [E8] sec:obc_construction, "An input with this property is persistently exciting [Eq. (6), Sec. 4.3]", 04:162-163 | citation | Gyorok's term is "weakly persistently exciting" in both cited places. | Gyorok 2026 PDF p. 3 (before Eq. (6)); PDF p. 5, Condition 14. | `weakly persistently exciting` |
| 65 | notation | [E9] sec:obc_map, "Jacobian ... taken with respect to $\beta$, as in ... [Sec. 4, Eq. (15)]", 04:182-185 | citation | Gyorok 2025 Eq. (15) differentiates with respect to the physical parameters; with respect to $\beta$ is an adaptation. Harmless: $\partial\theta_{\mathrm{base}}/\partial\beta$ is invertible, so the range is the same. | Gyorok 2025 PDF p. 7, Eq. (15). | `as in` to `adapted from` |
| 66 | notation | [B12] sec:lfr, "in linear parameter-varying (LPV) form ... \cite{toth2010modeling}", 02:614-617 | citation | Toth calls a system scheduled by its own state quasi-LPV. The next sentence ("self-scheduled") carries the qualification. | Toth 2010, Def. 7.2, PDF p. 194. | Optional: `(quasi-LPV in~\cite[Def.~7.2]{toth2010modeling}, since $Y$ is a state)` |
| 67 | notation | [E11] sec:obc_map, "$f^{\mathrm n}_{\mathrm{obc}}:\R^{n_{\mathrm{aug}}}\times\R^6\times\R^3\to\R^6$", 04:314 | notation | $f^{\mathrm n}_{\mathrm{obc}}$ and $\theta_{\mathrm{aux}}(\theta_{\mathrm{aug}})$ also depend on $\beta_{\mathrm{lin}}$, which the signature hides (fixed per epoch). | 04:304-306. | Optional: `for fixed $\beta_{\mathrm{lin}}$, $f^{\mathrm n}_{\mathrm{obc}}:\dots$` |
| 68 | notation | [F18] sec:setup_system, "The added mass changes the inertia through $M_a$, which also includes its coupling to $\Theta$ and $Y$", 05:149-150 | math | $M_a$ has a zero third row and column; it couples $X$ and $\Theta$ and changes the $\Theta$ inertia. The coupling of $q_a$ to $\Theta$ and $Y$ is in the off-diagonal block $m_a[0\;-d\;1]$. | Kinetic energy of $m_a$ at $Y+\ell$; gantrySystemExtendedTanh.m:34-37. | `The added mass changes the $X$-$\Theta$ inertia through $M_a$ and couples $q_a$ to $\Theta$ and $Y$ through $m_a[0\;-d\;1]$.` |
| 69 | notation | [F20] band-limited SNR: "at least 59~dB above the noise", 05:202-203; tab:setup_checks "SNR $0.24f_b$ to $1.05f_b$" | notation | app:setup_system defines only the broadband $\mathrm{SNR}_j$; the band-limited SNR used here is undefined. | 99:562-564; setup_numbers.tex (SetupGapSnrMinMotion "min SNR 24-105 Hz"). | `with $\mathrm{SNR}_j$ evaluated on the band $0.24\,f_b$ to $1.05\,f_b$` |
| 70 | notation | [F19] tab:setup_records Motion column: I3 "operational", I4 to I6 "route", 99:640-643; text 99:621 | notation | "route" is not one of the five motion types (99:617-620), while the text calls all four operational. | 99:617-623. | Motion column `operational` for I3 to I6, or define "route". |
| 71 | notation | [F22] tab:setup_checks "$K_1$ at 20~kHz, force error $1.5\cdot10^{-9}$", 99:735 | notation | Relative RMS without unit, between rows given in %. | setup_numbers.tex (SetupKcheck "largest relative RMS"). | `$1.5\cdot10^{-7}$\,\%` or `(relative)` |
| 72 | notation | [A6] sec:baseline, "Following Garc\'ia-Herreros et al. ..." and eq:mass_matrix, 02:419-454 | citation | eq:mass_matrix is Garcia's full inertia matrix (Eq. 8) under the stated assumptions (correct). Garcia's own simplified model moves $M_{23}=-m_hd$ to the force vector; the thesis keeps it, and the sentence saying so is commented out (02:424). | Garcia 2013 PDF p3, Sec. 2.3.1, Eqs. (12), (15), (17). | Restore 02:424: `Unlike~\cite[Sec.~2.3.1]{garcia2013model}, $M(Y)$ keeps the coupling $-m_hd$ between $\Theta$ and $Y$.` |
| 73 | notation | [A7] app:frf caption, "stage forces (columns) ... stage positions (rows)", 99:448 (also fig:res_frf, 06:99) | notation | Section II calls these the actuator forces $\uact$ and actuator positions $\qact$; "stage" means the whole machine elsewhere. | _setup_common.py:27 (AXES X1, X2, Y). | `from the actuator forces $\uact$ (columns) to the actuator positions $\qact$ (rows)` |
| 74 | notation | [A9] sec:problem, "the state $x$ contains the actuator positions $\qact$", 02:173-174 | notation | The baseline and simulated states are in generalised coordinates $[q;\dot q]$, which determine $\qact=P^\top q$ but do not contain it. | gantry_ss.py:128-136. | `the state $x$ determines the actuator positions $\qact$` |
| 75 | notation | [A8] $K_{\mathrm{fb}}$, $C_{\mathrm{fb}}$ (02:194-195) vs stiffness and damping $K$, $C$ (eq:matrices) | notation | Shared letters, disambiguated by the subscript. | 02:195, 456, 466. | None needed if the subscript is always kept. |
| 76 | notation | [E12] 99:865, wip bullet "Derive $J^+\Delta^*$" | notation | Stale symbols; Section IV now writes $(\Phi^\top\Phi)^{-1}\Phi^\top\Xi$. Inside `wip`: not in the review build, printed in the editor build. | 04:388. | `$(\Phi^\top\Phi)^{-1}\Phi^\top\Xi$` |

## Notation pass (lead)

Checked across slices: the four macros of `util/format.tex` (`\qact`, `\uact`, `\fnet`, `\Dsch`) and every symbol listed in the six agents' symbol tables.

- **Macros.** `\qact` and `\uact` keep one meaning (actuator positions and forces, $\R^3$, m and N) in 02, 03, 05 and 99. `\Dsch` ($\Delta(Y)=YI_6$) is used only in 02 and 99 with one meaning. `\fnet` is defined but not used in the printed sections audited here. Two places name $\qact$, $\uact$ differently: "stage forces/positions" in the FRF captions (row 73) and $u_i$ for $u_{\mathrm{act},i}$ in Section IV (row 52).
- **Symbols with more than one meaning across sections:** $\gamma$ (row 45), $N$ and $\mathcal D_N$ (row 46), $n_a$ vs $n_{\bar x}$ (row 47), $x$ (rows 55, 74), the hat accent (row 57), $e$, $\sigma_e$ (row 51), $\mathcal W$ (row 59), $m$, $a$, $j$, $s$, $H$ (row 56), subscript vs superscript 0 (row 54), $K$, $C$ (row 75).
- **Dimension and index conventions.** $\tilde x\in\R^6$ (m, rad, m/s, rad/s), $u,\uact,y,\hat y\in\R^3$, $\theta_{\mathrm{base}},\beta\in\R^{10}$, $\Phi\in\R^{6N_{\mathcal I}\times10}$, $\chi_\tau\in\R^{6(n+1)}$ and $n+1=30$ are consistent in every slice. Indexing is not: Section II starts $\mathcal D_N$ at $k=1$, eq:setup_rms and the code are 0-based (row 50).
- **Proposed fixes checked against each other.** Two agent fixes would have created new clashes and are changed above: the margin is renamed $\kappa$, not $\rho$ ($\rho$ is the overlap of eq:obc_overlap); the S1 residual is renamed $e^{\mathrm{S1}}$, not $\rho_\tau$; Gyorok's lagged vector is renamed $x^{\mathrm{IO}}_k$, not $z_k$ ($z$ is the LFR latent signal).
- **Resolved across slices.** Section III uses $\hat y=P^\top q$ (eq:aug_transition) while the LFR of app:lfr-details outputs $q$ (row 42); both are exact through eq:Ptransform. The controller is $K_{\mathrm{fb}}$ in 02 to 04 and $K_1$ (continuous design) in Section V and the appendix, with $K_{\mathrm{fb}}=K_1$ discretised at $T_s$ stated at 05:165-168: consistent.

## Correct

One line per formula or claim, with its evidence pointer, per slice as returned by the agents (line numbers refer to the snapshot).

### Slice A


- 02:113 to 114, gantry description (two parallel linear actuators, cross-arm, third linear actuator): matches Garcia PDF p2 Sec. 2.1.
- 02:150 to 151, $\uact=[F_{X1}\;F_{X2}\;F_Y]^\top$, $\qact=[X_1\;X_2\;Y]^\top$: matches code input/output order (gantry_ss.py:13 to 14; _setup_common.py:27).
- 02:155, each position from a linear encoder: Garcia PDF p2 Sec. 2.1.
- eq:system (02:165 to 168): continuous-time unknown $f$, sampled output with additive noise; dimensions consistent ($y_k,v_k\in\R^3$). Hold assumption 02:197 to 199 makes $\uact(t)$ piecewise constant, consistent with RK4 with input fixed over all substeps (blocks.py:791 to 798).
- 02:180, $\mathcal D_N=\{u_{\mathrm{act},k},y_k\}$: sufficient for the training path, which uses only $u_{\mathrm{act}}$ and $y$ in residual form (closed_loop.py:21 to 29, 41 to 46).
- eq:system_loop (02:187 to 193): discrete LTI controller on $e_k=r_k-y_k$ plus feedforward; dimensions consistent; no algebraic loop since $y_k=\qact(kT_s)+v_k$ depends on the input only up to $kT_s$; the residual form in closed_loop.py:41 to 46 is equivalent for this linear controller (difference of two loops with the same $A_{\mathrm{fb}},\dots,D_{\mathrm{fb}}$ and equal controller initial state).
- Fig. system_loop: signs ($+r_k$, $-y_k$), feedforward summation and noise injection agree with eq:system_loop and eq:system.
- 02:210 to 212, $v$ independent of $r,u^{\mathrm{ff}}$ while $u_{\mathrm{act}}$ correlated with $v$: follows from eq:system_loop ($u_{\mathrm{act},k}$ contains $-D_{\mathrm{fb}}v_k$ and, through $x^{\mathrm{fb}}$, past $v$).
- 02:267 to 272, augmentation with learned components and additional states: Hoekstra 2026 PDF p3 Eq. (3b) and text ("add new augmentation states $x_a$ to model missing dynamics").
- 02:296, encoder estimates initial states from past inputs and outputs: Beintema 2023 PDF p1 abstract and Sec. 3.
- 02:366 to 368, EOM derived by Euler-Lagrange, Garcia Eqs. (3) to (8): Garcia PDF p3, (3) T, (4) V, (5) D, (6) forces, (7) Lagrange, (8) $M\ddot q+H\dot q+C\dot q+Kq=f$.
- 02:369, $L_b$ the cross-arm length between the flexible joints: Garcia Fig. 2 (p3) dimension $L_b$ spans the joints $k_{b1},c_{b1}$ to $k_{b2},c_{b2}$; consistent with Eq. (2).
- 02:371 to 372, $X=(X_1+X_2)/2$: Garcia Eq. (1). $\Theta$: see finding 1.
- 02:381 to 385, closed kinematic chains and non-uniform load distribution: Garcia PDF p3 below Eq. (2).
- eq:Ptransform (02:396 to 409): $P^\top q=[X+\tfrac{L_b}{2}\Theta,\;X-\tfrac{L_b}{2}\Theta,\;Y]^\top=\qact$, so $q=P^{-\top}\qact$ gives $X=(X_1+X_2)/2$, $\Theta=(X_1-X_2)/L_b$; $u=P\uact=[F_{X1}+F_{X2},\;\tfrac{L_b}{2}(F_{X1}-F_{X2}),\;F_Y]^\top$ equals Garcia Eq. (6) without Coulomb terms and with $\cos\Theta\approx1$; power consistent ($\uact^\top\dot q_{\mathrm{act}}=(P\uact)^\top\dot q$); $P$ invertible for $L_b\neq0$. Code: gantry_ss.py:120 to 136, blocks.py:832, _setup_common.py:100 to 102, 144, 147.
- 02:410, $u$ generalised force with components [N, N m, N]: from the above.
- 02:419 to 421, small-angle and Coriolis/centripetal assumptions: Garcia PDF p3 Sec. 2.3.1 ("Assuming that cos(Θ)≈1 and sin(Θ)≈0") and Eq. (10) $M\ddot q+C\dot q+Kq\gg H\dot q$; operating yaw "several tens of µrad" (p3).
- 02:425 to 427, Coulomb friction of Garcia Eq. (6) excluded from the baseline: Garcia Eq. (6) contains $c_{c1},c_{c2},c_{cy}$ sign terms; baseline code has no friction (blocks.py:836 to 838, verified: `fnet = (-(x2[:, :3] @ K_mat.T)` / `- (x2[:, 3:] @ C_mat.T)` / `+ u_log)`).
- eq:eom (02:434 to 437): form of Garcia (8) with $H$ dropped; code fnet then $M(Y)^{-1}$fnet (blocks.py:836 to 855).
- eq:mass_matrix (02:446 to 454): hand derivation from Garcia (3) with $\cos\Theta=1$, $\sin\Theta=0$: $M_{11}=m_1+m_2+m_b+m_h$; $M_{12}=(m_1-m_2)\tfrac{L_b}{2}-m_hY$ (from $(m_1-m_2)\dot X\dot\Theta\cos\Theta\tfrac{L_b}{2}$ and $-m_h\dot X\cos\Theta\,Y\dot\Theta$; the $+m_hd\sin\Theta$ part vanishes); $M_{13}=-m_h\sin\Theta=0$; $M_{22}=J_b+J_h+\tfrac{L_b^2}{4}(m_1+m_2)+m_h(d^2+Y^2)$; $M_{23}=-m_hd$ (from $-m_hd\dot Y\dot\Theta$); $M_{33}=m_h$; symmetric. Agrees with Garcia Eq. (9) for $M_{12}$ and Eq. (12) for the $2\times2$ block. Units consistent with $q$ in [m, rad, m]. Code: gantry_ss.py:70 to 84 ($M_0+YM_1+Y^2M_2$) and _setup_common.py:115 to 120.
- eq:damping_matrix (02:456 to 463): from Garcia (5) with $\cos\Theta=1$: $C_{11}=c_{g1}+c_{g2}$, $C_{12}=\tfrac{L_b}{2}(c_{g1}-c_{g2})$, $C_{22}=c_{b1}+c_{b2}+\tfrac{L_b^2}{4}(c_{g1}+c_{g2})$, $C_{33}=c_y$; matches Garcia Eq. (13); code gantry_ss.py:89 to 94.
- eq:stiffness_matrix (02:466 to 473): from Garcia (4) $V=\tfrac12(k_{b1}+k_{b2})\Theta^2$; matches Eq. (14); code gantry_ss.py:99 to 100 (with GANTRY_KX_ART/KY_ART unset, the default 0.0, gantry_ss.py:108 to 113).
- 02:480 to 489, symbol meanings ($m_1,m_2,m_b,m_h,J_b,J_h,c_{g1},c_{g2},c_y,k_{b1},k_{b2},c_{b1},c_{b2},d$): Garcia PDF p2 to p3 parameter list and p4 (d: distance between centres of mass of payload and cross-arm); $J_b,J_h$ about own centres of mass (p3).
- 02:490 to 492 and app:params upper block (99:39 to 46): all 15 values equal Garcia Table I (PDF p4) and gantry_ss.py:37 to 56 and tdg_config.m:23 to 25; units match (Garcia Nm/(rad/s) = N m s/rad).
- app:params lower block (99:48 to 51): $c_{c,X_1}=16.80$, $c_{c,X_2}=18.35$, $c_{c,Y}=11.60$ N equal Garcia Table I and tdg_config.m:28; $v_{\mathrm{fr}}=1$ mm/s $=1/g$ with $g=1000$ s/m (tdg_config.m:37, setup_numbers.tex:43); $m_a=0.5m_h$, $L_0=0.10$ m, $f_a=150$ Hz, $\zeta_a=0.03$ equal tdg_config.m:41 to 47.
- app:frf text (99:443 to 444), "all nine entries of the response in Fig. res_frf": res_frf shows the $F_Y\to Y$ entry, res_frf_full the $3\times3$ (eval_frf.py:106 to 120); rows outputs, columns inputs as stated (res_frf_full.py:23 to 39); baseline and system FRFs use $\qact=P^\top q$, $u=P\uact$ (_setup_common.py:144, 147), consistent with eq:Ptransform.


### Slice B


Section III-B (sec:lfr):
- 02:551 "Moving $Y$ changes ... $M(Y)$": $Y$ enters eq:mass_matrix at (1,2), (2,2).
- eq:Msplit $M(Y)=M_0+YM_1+Y^2M_2$ (02:555): exact; $M_0$, $M_1$, $M_2$ as in 99:62-66 reproduce eq:mass_matrix entry by entry. Code gantry_ss.py:70-84 identical.
- 02:560-561 "With positive masses and inertias, $M(Y)\succ0$ for every $Y\in\R$": follows from app:lfr-wellposed final display (all terms positive) plus $m_h>0$ and $J_{\mathrm{eff}}+m_hY^2>0$.
- 02:562 "we invert $M(Y)$ to calculate $\ddot q$": code computes $M(Y)^{-1}f_{\mathrm{net}}$ as adjugate over determinant, blocks.py:847-855.
- 02:578-579 "augmentation assumes a baseline model in state-space form [hoekstra2026lfr]": Hoekstra Eq. (2a,b), PDF p. 3 (discrete time; discretisation is Section IV's, no claim of continuous time attributed).
- eq:base_ss (02:584-585): state derivative of eq:eom with $\tilde x=[q^\top\;\dot q^\top]^\top$; code path blocks.py:832-867 gives the same $\dot{\tilde x}$ (fnet, rational inverse, then $G$ with $w$ built from the solved $\ddot q$, which reproduces $\ddot q$ since $M_0\ddot q=f_{\mathrm{net}}-YM_1\ddot q-Y^2M_2\ddot q$). Code $u_{\log}=P u_{\mathrm{stage}}$ (blocks.py:832) matches $u=P\uact$.
- 02:617 "For a given $Y$, \eqref{eq:base_ss} is linear in $\tilde x$ and $u$": true with $Y$ frozen.
- 02:630-633 "same equations of motion ... for any time-varying $Y$, without an assumption on $\dot Y$": $M$ depends on $Y$ only, no $\dot Y$ terms (Coriolis neglected, 02:419-421).
- 02:645-646 rational dependence on $Y$: through $M(Y)^{-1}=\operatorname{adj}M(Y)/\det M(Y)$, both polynomial in $Y$ (see finding 9 for wording only).
- 02:647-651 exact LFR, well posed if $M(Y)$ nonsingular: shown in app:lfr-details (iff).

Appendix A.2 (app:lfr-details):
- 99:62 "$M_0$ is the part without $Y$": correct; 99:64-65 $M_1$, $M_2$: correct (code gantry_ss.py:79-84, blocks.py:1064-1077).
- eq:lfr_mass_factorization (99:93): substitution of eq:Msplit into eq:eom.
- eq:lfr_chain, $a_1=Y\ddot q$, $a_2=Y^2\ddot q=Ya_1$ (99:100-103): correct.
- eq:lfr_constant (99:107): correct with $a_1,a_2$ free.
- eq:delta (99:116-120): $w=[Y\ddot q;Ya_1]=\Dsch(Y)z$, $\Dsch(Y)=\operatorname{blkdiag}(YI_3,YI_3)=YI_6$; matches Drenth Eq. (3) with $n_p=1$, $\eta=6$; code blocks.py:861-862 `z = torch.cat([a, Y_val * a], dim=-1)` / `w = Y_val * z`.
- Fig. fig:lfr caption (99:84-86): $w=\Dsch(Y)z$, $\Dsch(Y)=YI_6$, $Y$ a state: consistent with eq:delta (figure file lfr_loop not in snapshot, caption only).
- eq:lfr_G (99:127-129) and 99:123-124 "adapting the discrete-time block structure of [Drenth Eq. (2)]": Drenth Eq. (2), PDF p. 2, is discrete time with rows $(x_{k+1},z_k,y_k)$, columns $(x_k,w_k,u_k)$, blocks $A_x,B_w,B_u;C_z,D_{zw},D_{zu};C_y,D_{yw},D_{yu}$; adaptation stated.
- 99:131-133 "$\ddot q$ obtained from eq:lfr_constant by multiplying with $M_0^{-1}$": correct.
- 99:137-139 loop closing gives $a_1=Y\ddot q$, $a_2=Y^2\ddot q$ and back eq:eom: correct (second block row of the loop equation gives $a_1=Y\ddot q$).
- 99:140-141 "well posed if the algebraic loop has a unique solution for every $Y$ [Drenth Def. 1]": Def. 1 (PDF p. 2) is $\det(I-D_{zw}\Delta(p_k))\neq0$ for all $p_k\in\mathcal P$, equivalent to a unique loop solution.
- 99:157-159 $R=M_0^{-1}$, $M_0=M(0)\succ0$, $\ddot q=R(u-Kq-C\dot q-M_1a_1-M_2a_2)$: correct.
- 99:160-161 partition "as in [Drenth Eq. (2)]", rows/columns: matches Drenth Eq. (2).
- eq:app_lfr_blocks (99:176-184): all nine blocks derived by hand and correct; code gantry_ss.py:208-232 builds $A_x$, $B_w$, $B_u$ identically ($C_z$, $D_{zw}$, $D_{zu}$, $C_y$ have no code counterpart, see finding 6).
- 99:189 $\tilde x,z,w\in\R^6$, $u,q\in\R^3$, $G\in\R^{15\times15}$: $6+6+3=15$ rows and columns.
- 99:195 loop equation $(I_6-D_{zw}\Dsch(Y))z=C_z\tilde x+D_{zu}u$: correct.
- 99:200 $I_6-D_{zw}\Dsch(Y)=[[I_3+YRM_1,\;YRM_2],[-YI_3,\;I_3]]$: correct.
- 99:211-213 Schur complement: $\det=\det(I_3+YRM_1+Y^2RM_2)=\det(R\,M(Y))=\det M(Y)/\det M_0$: correct.
- 99:216-217 Def. 1 condition iff $M(Y)$ nonsingular: correct ($\det M_0>0$).

Appendix A.3 (app:lfr-wellposed):
- 99:244-250 $M(Y)$ in $\theta_{\mathrm{base}}$ terms with $a=L_bm_\Delta/2$: matches eq:mass_matrix and eq:phi definitions.
- 99:252-256 Sylvester on trailing principal minors: $m_h$; $\det[[J_{\mathrm{eff}}+m_h(d^2+Y^2),-m_hd],[-m_hd,m_h]]=m_h(J_{\mathrm{eff}}+m_hY^2)$; $\det M(Y)$. Valid (permutation similarity).
- 99:260-263 $\det M(Y)/m_h=(m_\Sigma+m_h)(J_{\mathrm{eff}}+m_hY^2)-(a-m_hY)^2$ and completed square: both expanded by hand to $m_\Sigma m_hY^2+2am_hY+(m_\Sigma+m_h)J_{\mathrm{eff}}-a^2$. Code determinant blocks.py:848-850 `dY  = mh * (alpha * gamma_ - beta ** 2` / `+ 2 * beta * mh * Y` / `+ mh * (alpha - mh) * Y ** 2)` with alpha $=m_\Sigma+m_h$, beta $=a$, gamma_ $=J_{\mathrm{eff}}$ (gantry_ss.py:160-162): identical.
- 99:266-267 positive for every $Y$ iff eq:lfr_admissibility (given $m_h,m_\Sigma,J_{\mathrm{eff}}>0$): minimum over $Y$ is $(m_\Sigma+m_h)(J_{\mathrm{eff}}-a^2/m_\Sigma)$; dimensions kg$^2$m$^2$ throughout.
- 99:271-273 $m_\Sigma J_{\mathrm{eff}}-\tfrac{L_b^2}{4}m_\Delta^2=m_\Sigma(J_b+J_h)+\tfrac{L_b^2}{4}[4m_1m_2+m_b(m_1+m_2)]$: correct ($(m_1+m_2)^2-(m_1-m_2)^2=4m_1m_2$).
- 99:276-278 $b=\gamma(2/L_b)\sqrt{m_\Sigma J_{\mathrm{eff}}}$, $b_0$ at $\theta^0_{\mathrm{base}}$: code `_m_diff_bound`, blocks.py:1394-1395.
- 99:280-285 $\eta_0=\operatorname{atanh}(r_0)$, $s$, $m_\Delta(\beta=0)=m_{\Delta,0}$, $\partial m_\Delta/\partial\beta_{m_\Delta}=b_0s(1-r_0^2)=m_{\Delta,0}$ ($b$ does not depend on $\beta_{m_\Delta}$): correct (see finding 1); code blocks.py:1341-1342 `eta0 = torch.atanh(r0)` / `scale = combo_init[self.M_DIFF_IX] / (self._m_diff_bound(combo_init) * (1 - r0 ** 2))`.

Section III-C (sec:params):
- 02:710-712 $L_b$ fixed: code `self.Lb` buffer, blocks.py:1036 (frozen).
- 02:716-720 "fourteen remaining parameters": Table upper block has 15 entries including $L_b$.
- equation* 02:751 $(M(Y)-M'(Y))\ddot q+(C-C')\dot q+(K-K')q=0$: subtraction of eq:eom for $\theta,\theta'$ with equal $u$, $q$, $\dot q$, $\ddot q$.
- 02:760-762 any $\ddot q$ reachable at any state with $P$ invertible: $\ddot q=M(Y)^{-1}(P\uact-C\dot q-Kq)$, $M(Y)$ and $P$ invertible.
- 02:768-772 identity for all states and accelerations gives $K=K'$, $C=C'$, $M(Y)=M'(Y)$, then $M_i=M_i'$ (quadratic in $Y$, all $Y$): correct structurally (see finding 2).
- 02:772 equal matrices give equal input-output behaviour: same eq:eom, same $L_b$, same initial state.
- 02:793-796 "only these can be identified [Def. 7]": upper bound correct (functions of $\theta$ not determined by $\theta_{\mathrm{base}}$ leave the output unchanged); see findings 2 to 4.
- eq:phi (02:801-807) and definitions $k_{b,\Sigma},c_{b,\Sigma},m_\Sigma,m_\Delta,J_{\mathrm{eff}}$: match code order and formulas, blocks.py:1296-1297, 1358-1362.
- 02:811-812 $\theta_{\mathrm{base}}$ uniquely determined from the matrices: app:combos inverse map (needs $m_h>0$, stated there).
- 02:841-849 eq:lfr_admissibility, iff for $m_h,m_\Sigma,J_{\mathrm{eff}}>0$: proved in app:lfr-wellposed (above).
- 02:850-853 log parametrisation for nine, tanh for $m_\Delta$ ensures eq:lfr_admissibility for every estimate: $\lvert m_\Delta\rvert<b$ gives $\tfrac{L_b^2}{4}m_\Delta^2<\gamma^2m_\Sigma J_{\mathrm{eff}}<m_\Sigma J_{\mathrm{eff}}$, and logs keep $m_h,m_\Sigma,J_{\mathrm{eff}}>0$.
- 02:861-866 $\beta\in\R^{10}$, zero at $\theta^0_{\mathrm{base}}$, $\theta_{\mathrm{base},j}=\theta^0_{\mathrm{base},j}\exp(\beta_j)$: code blocks.py:1404-1406 `c = self.combo_init * torch.exp(free)` / `m_diff = self._m_diff_bound(c) * torch.tanh(` / `self.m_diff_eta0 + self.m_diff_scale * free[..., i])`, with `physics_parameterization='reduced'` (gantry_interconnect_dynamic.py:51) routing model.py:96-103 to Reduced_Gantry_State_Block.
- eq:mdelta_map (02:872-873): code above, $\gamma$ = `M_DIFF_RHO`; dimensions kg.
- 02:867-868 "$m_\Delta$ can be positive or negative and is bounded by eq:lfr_admissibility": correct.
- 02:882-888 $\eta_0$, $s$ chosen so $\beta=0$ gives $m_{\Delta,0}$: correct (finding 1 for the unstated $m_{\Delta,0}\neq0$).

Appendix A.4 (app:combos):
- 99:298-304 all twelve entry formulas ($M_{0,11},M_{0,12},M_{0,22},M_{0,23},C_{11},C_{33},C_{12},K_{22},M_{0,33}=-M_{1,12}=M_{2,22}=m_h,C_{22}$): checked against eq:matrices and eq:phi; code gantry_ss.py:70-100 and blocks.py:1051-1060.
- 99:307 entry notation $M_{i,jk}$: defined (see finding 10 for the clash with $m_{\Delta,0}$).
- 99:308-316 inverse map, ten formulas: each verified by substitution ($c_{g1}=\tfrac12C_{11}+C_{12}/L_b$ etc.); $m_h>0$ needed only for $d$.
- 99:318 "two different values of $\theta_{\mathrm{base}}$ cannot give the same matrices": follows from the inverse map.


### Slice C


Section intro and IV-A
- 03:228-241 roadmap (augment, closed loop, joint with $\theta_{\mathrm{base}}$, baseline start): no math; consistent with Sec. II.
- 03:300-302 "dynamic augmentation adds states $\bar x$": Hoekstra 2026 PDF p3 ("Dynamic augmentation structures ... add new augmentation states $x_a$").
- 03:305-306 "updates these states using the baseline states, the added states and the actuator forces [Sec. 2, Table 1]": Table 1 S-DP, $f_{\mathrm{aug}}(x_b,x_a,u)$, $g_{\mathrm{aug}}(x_b,x_a,u)$.
- 03:311-312 "dynamic parallel structure [Table 1]": Table 1 S-DP matches rows 1-2 of (14).
- (14) eq:aug_transition, rows 1-2: S-DP of Table 1 with $f_{\mathrm{base}}$ on $\tilde x$ only; code: interconnect.py:129-160 (`xp` starts at zero and sums only block contributions), model.py:179-183 (verified quote: `ic.connect_signals(ann_block, "xp", "additive", expansion_matrix(route_ix, nxd))` ... `ic.connect_signals(phy_block, "xp", "additive", expansion_matrix(PHY_IX, nxd))`), so $\bar x_{k+1}=g$ and $\tilde x_{k+1}=f_{\mathrm{base}}+f_{\mathrm{aug}}$; holds with ANN_ROUTE_IX = all rows (all thesis configs; RunConfig default config.py:188 also all rows for nx_ann=2).
- (14) row 3 $\hat y_k=P^\top q_k$ and $h_{\mathrm{base}}(\tilde x)=P^\top q$: from eq:Ptransform $q=P^{-\top}q_{\mathrm{act}}\Rightarrow q_{\mathrm{act}}=P^\top q$; code gantry_ss.py:135-138 ($C_d=[P^\top\;0]$, $D_d=0$).
- 03:380 $\tilde x_k=[q_k^\top\;\dot q_k^\top]^\top\in\R^6$, state of eq:base_ss: matches 02:581-587.
- 03:381-382 $\bar x_k\in\R^{n_{\bar x}}$, $\hat x_k=[\tilde x^\top\;\bar x^\top]^\top$; 03:388-389 $u_{\mathrm{act},k}\in\R^3$, $\hat y_k\in\R^3$: dimension-consistent.
- 03:382 "baseline state-transition and output functions": Hoekstra 2026 PDF p3 Eq. (2) wording.
- 03:394 $\theta_{\mathrm{base}}\in\R^{10}$ the combinations eq:phi: eq:phi lists ten; code PHYSICS_PARAMETERIZATION = reduced in all thesis configs.
- 03:395 $\theta_{\mathrm{aug}}\in\R^{n_{\mathrm{aug}}}$: definition only.
- 03:400-402 added states without physical meaning vs positions and velocities: consistent with (14).
- 03:410-415 output map not augmented, $h_{\mathrm{aug}}=0$; exact relation via eq:Ptransform: $q$ is defined by eq:Ptransform, so $q_{\mathrm{act}}=P^\top q$ holds exactly; code has no learned output block (model.py:132-134).
- 03:453-456 $f_{\mathrm{base}}$ = one RK4 step of $F$ (eq:base_ss) over $T_s$ with $u=Pu_{\mathrm{act}}$ held: blocks.py:791-798 RK4 with the same `u` in all four stages, blocks.py:832 `u_log = u2 @ self.P_mat.T`; "one step" holds for UP_SAMPLE = 1 (all thesis configs; the RunConfig default is up_sample = 2, config.py:191).
- 03:459-460 $M(Y)$ per RK4 sub-step from its intermediate state: blocks.py:847 (verified quote: `Y   = x2[:, 2]`) inside `_deriv_with`, called at each stage with the stage state.
- 03:478-481 tanh MLP, linear output layer: torch_nets.py:96-113 (Linear+Tanh hidden, final nn.Linear), ANN_ACTIVATION tanh, ANN_ARCH mlp.
- 03:484-485 network inputs $\tilde x_k$, $\bar x_k$, $u_{\mathrm{act},k}$ with $n_u=3$: model.py:144 `nz=nxd + nu`, model.py:179.
- 03:489-492 first six outputs correct the physical states, remaining $n_{\bar x}$ give $\bar x_{k+1}$: see (14) row; routing all rows.
- 03:503-505 $Y$ in $\tilde x_k$ reaches the network: $Y=q_3$, network takes full $\hat x$.
- 03:526 $\mu$, $\sigma$ from $\mathcal D_N$ (training set): data.py:335-377 uses `train_list` only.
- 03:527-529 $q=P^{-\top}y$, velocities by finite difference divided by $T_s$: data.py:344 and data.py:367-368 (verified quote: `vel_logical = np.diff(pos_logical, axis=0) * fs`, `vel_logical = np.vstack([vel_logical[:1], vel_logical])`), backward difference, first sample copied.
- (16) eq:aug_norm, $T_i=\operatorname{diag}(\sigma_i)^{-1}$, $\mu_x\in\R^6$, $\mu_u,\mu_y\in\R^3$: data.py:372-377 (code adds 1e-8 to each $\sigma$, negligible); y normalisation via fit_sys.norm y0, ystd (model.py:204-207).
- 03:555-557 RK4 evaluates $F$ in SI units obtained by inverting (16): blocks.py:824-825 (verified quote: `x_phys = x * self.std_x + self.x_mean`, `u_phys = u * self.std_u + self.u_mean`), derivative divided by $\sigma_x$ at blocks.py:870.
- 03:563-571 only baseline states normalised; $\hat x^{\mathrm n}=[\tilde x^{\mathrm n\top}\;\bar x^\top]^\top$: encoder/ANN carry $\bar x$ raw.
- (17) eq:aug_norm_maps rows 1-2: from (14) and (16), $T_x(\tilde x_{k+1}-\mu_x)=T_x(f_{\mathrm{base}}-\mu_x)+T_xf_{\mathrm{aug}}$ (notation in finding 8).
- (17) row 3 $\hat y^{\mathrm n}=C^{\mathrm n}\tilde x^{\mathrm n}$, $C^{\mathrm n}=T_yP^\top[I_3\;0]T_x^{-1}$ (3x3, 3x3, 3x6, 6x6 = 3x6): $T_y(P^\top q-\mu_y)=T_yP^\top[I_3\;0]T_x^{-1}\tilde x^{\mathrm n}+T_y(P^\top\mu_q-\mu_y)$, offset zero with $\mu_q=P^{-\top}\mu_y$; code data.py:380 (verified quote: `Cd_norm = Cd.numpy() * std_x.flatten()[None, :] / ystd[:, None]`) $=\operatorname{diag}(\sigma_y)^{-1}C_d\operatorname{diag}(\sigma_x)$.
- (17) rows 4-5 $f^{\mathrm n}_{\mathrm{base}}=T_x(f_{\mathrm{base}}(T_x^{-1}\tilde x^{\mathrm n}+\mu_x,T_u^{-1}u^{\mathrm n}+\mu_u)-\mu_x)$: code integrates RK4 in normalised coordinates with $\dot{\tilde x}^{\mathrm n}=T_xF(\cdot)$ (blocks.py:824-870); RK4 commutes with affine state maps, so this equals row 4 exactly.
- 03:637-638 "$f_{\mathrm{aug}}$ scaled only by $T_x$, since $f^{\mathrm n}_{\mathrm{base}}$ already subtracts $\mu_x$": same derivation as rows 1-2.
- 03:645-647 position means $P^{-\top}\mu_y$, $\mu_y$ cancels, no output offset: code computes $\mu_q$ as the mean of $P^{-\top}y$ over the same samples as $\mu_y$, which equals $P^{-\top}\mu_y$ by linearity (data.py:344, 372, 377).
- 03:656-657 Hoekstra's (28) assumes zero mean: Hoekstra 2026 PDF p9 "For a baseline model operating around zero mean $\hat x,u,\hat y$".
- 03:670-673 $[f^{\mathrm n\top}_{\mathrm{aug}}\;g^{\mathrm n\top}_{\mathrm{aug}}]^\top=\phi_{\mathrm{aug}}(\theta_{\mathrm{aug}},\hat x^{\mathrm n}_k,u^{\mathrm n}_k)$: model.py:143-151, 179-180 (network on normalised `x`, `u`, output added to `xp`).

IV-B
- 03:729-732 closed-loop simulation as Kessels Eqs. (5.12), (5.13): Kessels PDF p183-184, (5.13c)-(5.13d) controller error against model output; contrast with Hoekstra (22): Hoekstra PDF p8, (22b) driven by recorded $u_{k_i+\ell}$.
- 03:752-754 $X$, $Y$ without stiffness in eq:damping_stiffness_matrices: $K$ has only $(2,2)=k_{b1}+k_{b2}$.
- 03:756 "not asymptotically stable" follows from a pole at $z=1$ (with finding 3 for $Y$).
- 03:761-764 constant correction error accumulates in position; velocity error also accumulates: $q_{k+1}=q_k+T_s\dot q_k+\ldots$ with no restoring term on $X$, $Y$.
- 03:765-767 recorded input leaves drift uncorrected; feedback corrects it: qualitative, consistent with the loop.
- (18) eq:aug_two_loops: same $(A_{\mathrm{fb}},B_{\mathrm{fb}},C_{\mathrm{fb}},D_{\mathrm{fb}})$ as eq:system_loop with $y\to\hat y$; Kessels (5.13d).
- 03:854-856 $\hat u_{\mathrm{act},k}$ replaces $u_{\mathrm{act},k}$ in (14): closed_loop.py:239.
- 03:858-862 subtracting eq:system_loop from (18) cancels $r$, $u^{\mathrm{ff}}$; $x^c=\hat x^{\mathrm{fb}}-x^{\mathrm{fb}}$ driven by $y-\hat y$: $x^c_{k+1}=A_{\mathrm{fb}}x^c_k+B_{\mathrm{fb}}((r-\hat y)-(r-y))$.
- 03:883-884 $x^c_\tau=0$ sets the model controller state to the system's: definition of $x^c$; code closed_loop.py:284-285 zero state per window.
- 03:899-902 Kessels Remark 5.4: PDF p184, controller states measured or reconstructed from $\bar y$, $\bar r$ and the controller with $\bar x^{\mathrm{FB}}_{k=1}=0$.
- (19) eq:aug_residual_loop: $\hat u-u=C_{\mathrm{fb}}x^c+D_{\mathrm{fb}}(y-\hat y)$ from the subtraction; code closed_loop.py:235-239 (verified quote: `u_fb, xc = step(xc, y_t - y_model, ctrl)`, `_, x = hfn(x, u_t + u_fb)`), controller K_1 re-discretised (Tustin) at $T_s$ (controller.py:116, 299).
- 03:919 needs only recorded $u_{\mathrm{act}}$, $y$: training path uses only `u_total`, `y` (closed_loop.py:21-26, data.py:214).
- (20) eq:aug_residual_split: with $\check x^{\mathrm{fb}}_{k+1}=A\check x^{\mathrm{fb}}_k+B(r_k-y_k)$, $\check u=C\check x^{\mathrm{fb}}+D(r-y)+u^{\mathrm{ff}}$, $\hat x^{\mathrm{fb}}=\check x^{\mathrm{fb}}+x^c$ obeys $\hat x^{\mathrm{fb}}_{k+1}=A\hat x^{\mathrm{fb}}_k+B(r_k-\hat y_k)$, and $u+Cx^c+D(y-\hat y)=C\hat x^{\mathrm{fb}}+D(r-\hat y)+u^{\mathrm{ff}}+(u-\check u)$. Exact.
- 03:950-952 first three terms are the closed-loop model, last term the force difference: from the line above.
- 03:953 noise independent of $r$, $u^{\mathrm{ff}}$: restates 02:210.
- 03:954-956 differing controllers: (19) no longer equals (18) and $v$ enters via $u-\check u$, since both depend on $y=q_{\mathrm{act}}+v$ through different operators.
- 03:957-959 noise enters through the initial state (encoder on recorded $u$, $y$; inherited controller state from earlier $y$): consistent with $x^c_\tau=0$.
- 03:961-962 $\hat y_k$ depends on $\tilde x_k$ only, not on $\hat u_{\mathrm{act},k}$ at the same sample: $D_d=0$ (gantry_ss.py:138), closed_loop.py:235 `output_only(x)`.
- 03:966-969 order output, error, input, then states; no algebraic loop through $D_{\mathrm{fb}}$; Kessels Remark 5.1 (PDF p179: $l^{(3)}$ independent of the input to avoid the non-causal loop through the FB controller): same reason; code order closed_loop.py:235-239.
- (21) eq:aug_cl_step: $\varepsilon=T_y^{-1}(y^{\mathrm n}-\hat y^{\mathrm n})=y-\hat y$ (means cancel); $\hat u^{\mathrm n}=T_u(u+C x^c+D\varepsilon-\mu_u)=u^{\mathrm n}+T_u(C x^c+D\varepsilon)$; $x^c_{k+1}=Ax^c_k+B\varepsilon$. Code folds $B'=B\operatorname{diag}(\sigma_y)$, $C'=T_uC$, $D'=T_uD\operatorname{diag}(\sigma_y)$ (closed_loop.py:139-141) and steps on the normalised error (closed_loop.py:167-184): identical.
- 03:990-992 next state from (17) with input $\hat u^{\mathrm n}$: closed_loop.py:239.

Appendix residual paragraph
- 99:742 "training loop re-discretises $K_1$ at $T_s$": controller.py:116 (Tustin at `ts`), controller.py:299 (`cfg.ts_new`).
- 99:742-747 the two medians: numerical, see Not checkable.


### Slice D


IV-C (sec:aug_closed_loop)
- 03:1096-1099: tau is the window start and only positions are measured. Consistent with eq:system.
- 03:1107-1109: estimating x̂_tau per window makes the number of parameters grow with the data. Matches Beintema Sec. 3.1, p. 5 ("number of parameters would scale O(N)").
- 03:1117-1118: short windows increase loss smoothness. Matches Beintema Sec. 3.1 p. 4 and Sec. 4.2, Theorem 9, p. 8.
- 03:1127-1129: joint estimation of theta_aug, theta_enc and theta_base. Hoekstra LFR Eq. (22) has theta = col(theta_base, theta_aug, theta_LFR, theta_encoder), p. 8. The encoder paper footnote 1, p. 2, puts co-estimation of theta_base "outside the scope".
- 03:1130-1136: vartheta = (beta, theta_aug, theta_enc), adapted from Hoekstra Eq. (22a) with closed loop and W_r. The adaptation is stated.
- eq:aug_loss, V and its normalisation 1/(|W| n_f n_y): code mean over windows x nf x ny (interconnect.py:1426 `L = ((y_pred - yfuture) * w[:, None, :]).square().mean()`).
- eq:aug_loss constraints. Row 1 (x̃ update) and row 2 (x̄ update) match eq:aug_norm_maps. Row 3 (encoder) matches eq:aug_encoder, with ŷ and û from eq:aug_cl_step. Code: closed_loop.py:235-241 computes the output first, then the controller, then the step; the window controller state is zero at closed_loop.py:284.
- 03:1159-1161: tau <= k < tau+n_f, n_y = 3, theta_base(beta) as in sec:params.
- 03:1171-1173: W_r = diag(sigma_{y,j}/sigma_{e,r,j}). By hand, W_r(y^n - ŷ^n) = diag(1/sigma_{e,r})(y - ŷ), because mu_y cancels. This is the residual in metres divided by the servo-error std. Code: controller.py:353 sets weights ystd/servo_sigma; data.py:224-235 takes the population std of e = y - r.
- 03:1179-1180: the encoder uses samples up to tau-1. Hoekstra LFR Eq. (22e) uses y_{k-na}^{k-1}, u_{k-nb}^{k-1}. Code: model.py:47-51 (data_lsq: na = nb = na_nb, na_right = 0).
- eq:aug_encoder: chi_tau stacks y_{tau-n-1}..y_{tau-1} and u_{tau-n-1}..u_{tau-1}, n+1 samples, so chi_tau is in R^{6(n+1)}.
  - Code: model.py:253 builds the encoder with na-1, so n = na-1. ps2_opening.py:174 takes the window u[k-nb:k], y[k-na:k].
- 03:1190: x̂^n_{tau|tau} is the normalised state of Fig. 3.
- 03:1202-1205: the encoder estimates all 6+n_xbar states, positions included.
- eq:aug_enc_lin: psi = [W^b; W^a] chi + psi_tilde(chi). Matches encoder paper Eq. (8), p. 3, including the y-then-u block order. Code: pre_encoder.py:503-524.
- 03:1221-1225: W^b is 6 x 6(n+1) and W^a is n_xbar x 6(n+1). theta_enc consists of W^b, W^a and psi_tilde. In data_lsq there are no offset buffers (fix_enabled false), so there are no further parameters.
- 03:1243-1247: SUBNET assumes the input is independent of the noise (Beintema Sec. 2.1, p. 3) and incremental exponential output stability (Beintema Cond. 1, p. 7).
- 03:1248-1249: Hoekstra Sec. 5.5 (p. 10) relies on the conditions of [5] = Beintema 2023. Hoekstra names incremental exponential output stability explicitly. Input independence enters through Beintema's consistency theorem ("u independent of the white noise", p. 7). Note: Hoekstra writes "Conditions 2.1 to 2.4", but Beintema numbers its conditions 1 to 6. That is Hoekstra's labelling, not a thesis error.
- 03:1250-1251: the open-loop gantry is not incrementally output stable. By hand: with K = 0 on X and Y and positive damping, a velocity mismatch dv decays, so the position difference converges to a nonzero constant (m dv / c) and does not decay to zero.
- 03:1264-1267: each window starts from a new encoder estimate with x^c_tau = 0, so model errors before tau do not enter. Code: closed_loop.py:284.
- 03:1275-1277: the baseline/network split is not unique. Matches Hoekstra Sec. 5.2, p. 8 to 9.
- 03:1290-1292: Adam on minibatches with the gradient over the full window. Code: closed_loop.py:295-311 uses exact (non-truncated) checkpointed BPTT.
- 03:1298-1301: L-BFGS on a fixed set of windows. Code: lbfgs_polish.py:633 (FixedBatch) and :461-496 (the same fit_sys.loss). Drenth Sec. 4.1, p. 4 uses Adam then L-BFGS-B.
- 03:1302-1303: unconstrained L-BFGS needs no bound. By hand from eq:mdelta_map: |m_Delta| < gamma (2/L_b) sqrt(m_Sigma J_eff) with gamma < 1, which gives (13) for every beta, because exp keeps m_Sigma and J_eff positive. Code: lbfgs_polish.py:650-654 (torch LBFGS, no bounds) and blocks.py:1404-1405.
- 03:1308-1311: the lowest validation error is kept during Adam. Code: interconnect.py:902.
- 03:1322-1324: validation is a closed-loop free run with the encoder on the first n+1 samples, scored on the rest. Code: closed_loop.py:756 (k0 = max(na, nb) = n+1) and :787-792.
- 03:1329: the polish is kept only if it lowers the validation error. Code: lbfgs_polish.py:121-127 with meters_gate False (config) gives a strict improvement test; training.py:793 compares against bestfit.

IV-D (sec:aug_encoder)
- 03:1349-1352: random initialisation can be unstable and optimise poorly. Matches Hoekstra Sec. 5.4, p. 9.
- 03:1358-1359: the baseline is nonlinear through M(Y)^{-1} with Y a state. Consistent with eq:base_ss.
- 03:1360-1364: a model-based map is accurate only around the linearisation point. Matches encoder paper Sec. 3.1, p. 4 ("only be accurate around the equilibrium point").
- 03:1365-1368: fit the encoder to baseline-simulated states. Matches encoder paper Sec. 3.2 / Eq. (35) and Hoekstra LFR Eq. (30), p. 9.
- eq:aug_enc_sim: consistent with eq:aug_norm_maps and eq:aug_cl_step at theta^0 with f_aug = 0. Code: model.py:389-398 (recorder on fit_sys.hfn inside closed_loop_rollout).
- 03:1390-1391: Hoekstra Eq. (30) / D̂_N uses the recorded u_k (LFR p. 9; encoder paper p. 5); the thesis uses the closed-loop input instead.
- 03:1400-1403: initial conditions q = P^{-T} y_0 (consistent with eq:Ptransform), zero velocities, zero added states and x^c = 0. Code: model.py:384-386.
- eq:aug_enc_fit: "∈ argmin", D and chi^0. Code: model.py:401-404 (windows at the training stride, target state at tau = start + na) and :435-437. The minimum-norm lstsq in standardised regressors is an element of the argmin up to the gelsd rank cutoff (machine precision).
- 03:1429-1430: linear fit for LTI baselines and a network for nonlinear baselines. Matches encoder paper Eqs. (33), (35), p. 5.
- 03:1446-1449: physical output bias of psi_tilde = b^{b,0} and weights = 0, which gives psi_b = W^{b,0} chi + b^{b,0}. Code: model.py:447-449; enc(0,0) has no offset in data_lsq (pre_encoder.py:467-469).
- eq:aug_init, network row:
  - Physical rows: W^xtilde_L = 0 and b_L = 0 (torch_nets.py:197-201).
  - Added-state rows: W^xbar_L keeps the nn.Linear draw U(-1/sqrt(n_h), 1/sqrt(n_h)) (kaiming_uniform a = sqrt(5) gives bound 1/sqrt(fan_in)).
- eq:aug_init, encoder rows:
  - W^b = W^{b,0}.
  - Physical rows of psi_tilde: weight 0, bias b^{b,0}.
  - Added-state rows of psi_tilde: weight 0 and bias 0 (pre_encoder.py:454, 459; asserted at model.py:461-463).
  - W^a ~ U(-a, a) with a = sqrt(6/(n_xbar + 6(n+1))). This is Xavier-uniform with fan_in 6(n+1), fan_out n_xbar, gain 1, as one joint draw (pre_encoder.py:432-435).
- 03:1503-1517: definitions of W_L, b_L, row split, n_h and the meaning of U(-a, a).
- 03:1518-1520: hidden layers of phi_aug and psi_tilde use nn.Linear defaults U(-1/sqrt(m), 1/sqrt(m)).
- 03:1526-1531: at the start, theta_base = theta^0 and f^n_aug = 0, so x̃ and ŷ match the baseline (also in closed loop, since eq:aug_cl_step depends only on ŷ), while x̄ is updated by g^n_aug. Matches Hoekstra Eq. (29), p. 9.

IV-E (sec:aug_ps2)
- 03:1555-1559: the physical states have a data-based start and a direct output map; the added states have neither.
- 03:1569-1578: x̄ enters x̃_{k+1} only through f^n_aug, and V has no term that forces x̄ to predict the error (by inspection of eq:aug_loss).
- 03:1602-1608: two stages, then V continues.
- 03:1614-1623:
  - Hefny Sec. 2, pp. 3 to 4: S1A is a "possibly non-linear" regression of future features; S2 is a linear operator W. Fig. 2 layout, p. 2.
  - Downey Sec. 4.2, p. 5: 2SR initialisation, then BPTT refinement.
- 03:1663-1673: S1 uses the vartheta after the latest V update. theta^a_enc = {W^a, the added-state output rows of psi_tilde}. Code: ps2_opening.py:151-156 passes all final-layer rows, but the physical rows get an exactly zero gradient, so Adam leaves them unchanged; the hidden layers are excluded.
- 03:1683-1684: S1 minibatch from D_N without the left-out records. Code: ps2_opening.py:195 (fit_ix), :59-66 (last record of each class with at least 3 records).
- 03:1694-1698: closed-loop simulation from the encoder state, residual over H = n+1 samples tau..tau+H-1. Code: ps2_opening.py:146 (M = na + na_right = na = n+1), :175 and :187.
- eq:ps2_s1: decoder on psi_a only. The 1/(|B| H n_y) mean matches `.pow(2).mean()` over B x M x n_y (ps2_opening.py:198). e_tau is the time-major stack of normalised residuals (reshape at :188). sigma_e is the batch RMS (ps2_opening.py:189).
- 03:1737-1741: ŷ is held fixed in the S1 update (no_grad and a detached x at ps2_opening.py:185-197).
- 03:1756-1759: S1 updates only theta_p and theta^a_enc; V still trains all of vartheta.
- 03:1763-1764: the decoder sees only x̄, which encourages predictive content (stated as encouragement, not a guarantee).
- 03:1820-1822: left-out records decide the end of S1 and whether S2 runs, and remain in V (V uses data.train_data, which holds all records).
- eq:ps2_alt and 03:1828-1845:
  - Each V step is followed by one S1 step, with separate Adam instances (interconnect.py:1039-1046; ps2_opening.py:156).
  - theta^a_enc from A_S1 overwrites the copy in vartheta^{(i+1)} (in-place step).
  - S1 performs i_1 steps for i = 0..i_1-1 (ps2_opening.py:106-107).
- 03:1852-1858: S2 trains g^n_aug from x̂^n_k and u^n_k to the next x̄.
- 03:1866-1875 and eq:ps2_s2, the set P:
  - x̂_{k|k} = psi(chi_k) and x̄_{k+1|k+1} = psi_a(chi_{k+1}), the window shifted by one sample (ps2_opening.py:177, 223).
  - Recorded u^n_k = uf[:, 0].
  - Computed once after S1; theta_enc is frozen (no_grad).
  - Only fit records are used (ps2_opening.py:224).
- 03:1878-1881: in S2 there is no closed-loop simulation and no V step; one-step regression with its own Adam (ps2_opening.py:229). The x_a rows of hfn equal g^n_aug.
- 03:1885 and app:421-425: with x^c = 0, the model input at the window start is û = u + D_fb(y - ŷ_{k|k}), from eq:aug_residual_loop. It equals the recorded u iff ŷ_{k_i|k_i} = y_{k_i}, and the condition is stated. Code: closed_loop.py:237 with xc = 0 gives u_fb = D eps.
- 03:1898-1901: B_P and m_j are defined (the code samples B_P with replacement, which is immaterial).
- 03:1919-1921: keep the theta_aug with the lowest cost on the left-out records, with the start as a candidate. Code: ps2_opening.py:224, 232-247.
- 03:1942-1951: S2 updates all of theta_aug. The physical rows of W_L and b_L get a zero gradient and do not change. The shared hidden layers change f^n_aug.
- eq:ps2_handover and 03:1961-1964:
  - vartheta_PS = (beta^{(i_1)}, theta^{S2}_aug, theta^{(i_1)}_enc).
  - On fallback, the parameters after S1 are used (ps2_opening.py:118-128).
- 03:1971-1972: PS adds no parameters; the head is discarded (ps2_opening.py:127-128).

Appendix (app:ps_stop)
- app:339-340: sigma_e is the RMS of all residual values in B (ps2_opening.py:189).
- app:347 decoder p = W_{p,2} tanh(W_{p,1} xbar + b_{p,1}) + b_{p,2}, with W_{p,2}, b_{p,2} = 0 at the start. Code: ps2_opening.py:148-150.
- app:362-363: the last record of each class with at least 3 records is held out (ps2_opening.py:59-66).
- eq:ps2_nu: the ratio of sums equals the code's ratio of means over the same count (ps2_opening.py:211). It is invariant to the sigma_e scaling. C is a fixed set of 128 windows drawn once (:157).
- app:385, the i_1 display, nu^new and nu^old, and nu^old = infinity for the first w values. These match ps2_opening.py:108-118:
  - eval_every = i_c = 25, s1_window = w = 4, s1_tol = delta_nu, s1_min = i_min, s1_cap = i_max, screen_at = i_s.
  - Plateau test `min(last w) > (1 - tol) * best_before`; best_before is None while len(history) <= w.
  - The screen (i = i_s and nu > nu_max) ends PS with no S2 (ps2_opening.py:118 `if (self.it == self.spec.screen_at and not end and u > self.spec.screen_max) or (end and u > self.spec.screen_max):`).
  - i_max and i_s are multiples of i_c (350, 150), so "i = i_max" and the code's ">=" agree.
- app:409-410: the S2 condition is correct. The clause "when i_1 > i_s, also nu_{i_s} <= nu_max" is implied by the definition of i_1 (otherwise i_1 = i_s), so it is redundant but not wrong.
- app:421-425: the û equation, as above.


### Slice E


- Opening, "pair ... can be non-unique: different pairs can give the same fit [Sec. 3.1]": matches Gyorok 2026 Sec. 3.1, PDF p. 3.
- Opening, the definition of orthogonality (no component along the directions in which a change of $\theta_{\mathrm{base}}$ changes the next state): consistent with eq:obc_coef. $\ran\Phi_\beta=\ran\Phi_{\theta_{\mathrm{base}}}$ because the coordinate map is invertible, and $\Phi_\Gamma^\top\tilde F=0$ implies $\Phi^\top\tilde F=0$.
- Opening, "unique estimate ... up to first order" and "equals the true parameters only if the missing dynamics are orthogonal": consistent with eq:obc_recovery and the iff at 406 to 409, under the full-rank condition of IV-B (see finding 2).
- Opening, "Gyorok ... input-output baseline models linear in their parameters": Gyorok 2026 Eq. (2), Sec. 6 (PDF pp. 2, 9).
- Inline $\hat y_k=\varphi(x_k)\theta_{\mathrm b}$, changes along columns of $\varphi$ [Eqs. (2), (39)]: Eq. (2) PDF p. 2; Eq. (39) $\partial\hat y_k/\partial\theta_b=\varphi(x_k)$, PDF p. 6.
- eq:obc_gyorok row 1 ($\hat Y=\Phi\theta_b+\tilde F$): Gyorok 2026 Eq. (9).
- eq:obc_gyorok row 2 ($\tilde F=F-\Phi\theta_{\mathrm{aux}}$): Eq. (9).
- eq:obc_gyorok row 3 ($\theta_{\mathrm{aux}}=(\Phi^\top\Phi)^{-1}\Phi^\top F$): Eq. (8).
- eq:obc_gyorok row 4 ($0=\Phi^\top\tilde F$): Eq. (11), Lemma 3. By hand: $\Phi^\top F-\Phi^\top\Phi(\Phi^\top\Phi)^{-1}\Phi^\top F=0$.
- "$x_k$ past measured outputs and past and current inputs [Eq. (1)]": $x_k=\mathrm{vec}(y^{k-1}_{k-n_a},u^k_{k-n_b})$, PDF p. 2.
- $\hat Y$, $\Phi$, $F^{\mathrm{ANN}}$ stacked [Eq. (5)] and $\tilde F$ [Eqs. (9), (10)]: PDF pp. 2 to 3.
- "$(\Phi^\top\Phi)^{-1}$ exists only if $\Phi$ has full column rank [Assumption 1]": Assumption 1 Eq. (7), PDF p. 3; Gram matrix invertible iff full column rank.
- "different values of $\theta_b$ are unable to give the same stacked prediction": $\Pi\hat Y=\Phi\theta_b$ with $\Phi$ injective, so $\theta_b$ is unique.
- "identifiable baseline, different $\theta_b$ give different functions $\varphi(\cdot)\theta_b$": Gyorok 2026 PDF p. 3.
- "nonlinear in the parameters, because ... $M(Y)^{-1}$ in eq:base_ss": eq:base_ss; also the exp/tanh coordinate map and RK4.
- "This requires identifiable parameters, which is why we use $\beta$": structural identifiability is necessary for full column rank (Section II-C, eq:phi).
- eq:obc_sens and the domain/codomain $\varphi:\R^{10}\times\R^6\times\R^3\to\R^{6\times10}$, $\beta_{\mathrm{lin}}\in\R^{10}$: dimensions agree with $\tilde x^{\mathrm n}\in\R^6$, $u^{\mathrm n}\in\R^3$, ten coordinates. Code: `obc.py:361` jacfwd of `transition_from_free` (normalised RK4 next state, `blocks.py:1429 to 1445`) in the free coordinate.
- eq:obc_offset row 1 ($f\approx\varphi\beta+\gamma$) and row 2 ($\gamma=f(\beta_{\mathrm{lin}})-\varphi\beta_{\mathrm{lin}}$): first-order Taylor expansion, Gyorok 2025 Eqs. (15), (16), PDF p. 7.
- "the first term is linear in $\beta$ and $\gamma:\R^{10}\times\R^6\times\R^3\to\R^6$ is an offset independent of $\beta$": Gyorok 2025 text after Eq. (16); dimensions right.
- "add this offset as an extra column ... make the neural network also orthogonal to the next states predicted by the baseline [Eq. (18)]": Gyorok 2025 Eq. (18), PDF p. 7; $\Phi\beta_{\mathrm{lin}}+\Gamma\in\ran\Phi_\Gamma$, so $\tilde F$ is orthogonal to the stacked baseline next state at $\beta_{\mathrm{lin}}$.
- "Jacobian depends on the state, while the regressor ... uses only measured inputs and outputs [Eqs. (1), (2)]": Gyorok 2026 PDF p. 2.
- eq:obc_tuples row 1, $q_i=P^{-\top}y_i$: eq:Ptransform $q=P^{-\top}q_{\mathrm{act}}$. Code `obc_gantry.py:131` `q = y @ PinvT.T` (MATCH OK).
- eq:obc_tuples row 2, the fourth-order central difference: standard weights $(1,-8,0,8,-1)/12$ over $T_s$, error $O(T_s^4)$. Code `obc_gantry.py:88` `return (-q[4:] + 8.0 * q[3:-1] - 8.0 * q[1:-3] + q[:-4]) / (12.0 * ts)` (MATCH OK).
- "$\bar x_i=0$", normalisation by eq:aug_norm: `obc_gantry.py:148 to 154` (zeros for the added states, $\mu_x,\sigma_x,\mu_u,\sigma_u$ of the pipeline).
- "calculated once before training, using all samples for which eq:obc_tuples can be calculated at $i$ and $i+1$": `obc_gantry.py:135` with `obc_ref_successor=True` (MATCH OK); built in `attach_obc` at model build (`obc_gantry.py:376 to 380`); per record, never across a record boundary.
- eq:obc_coef rows 1 to 5: dimensions $\Phi_\Gamma\in\R^{6N\times11}$, $F,\tilde F\in\R^{6N}$, $\theta_{\mathrm{aux}}\in\R^{11}$ are consistent (all six physical rows routed: `ann_route_ix=range(6+n_a)`). $0=\Phi_\Gamma^\top\tilde F$ follows by the Lemma 3 algebra under rank 11 (finding 2). Code: SVD pseudoinverse `obc.py:247` `theta = self.Vh.T @ ((self.U.T @ F64) / self.S)` (MATCH OK), equal to $(\Phi_\Gamma^\top\Phi_\Gamma)^{-1}\Phi_\Gamma^\top F$ at full rank; field at $\bar x=0$, `obc_gantry.py:153 to 154, 357 to 358`.
- "no component along any column of $\Phi_\Gamma$, which satisfies our orthogonality definition": the column inner products are zero, and $\Phi$ is a sub-block.
- "holds for the stacked tuples at $\bar x=0$, not for each tuple separately": the normal equations are a sum over tuples.
- "fitted part removed ... at every step in the closed-loop simulation [Eqs. (9), (13)]": Gyorok 2026 Eq. (13), PDF p. 3, applies $\varphi(x_k)\hat\theta_{\mathrm{aux}}$ at each new point. Code `interconnect.py:170` `xp = xp - _obc(x, u, obc_pass)` (MATCH OK), applied in every rollout step.
- "added-state rows: Jacobian and offset zero, $g^{\mathrm n}_{\mathrm{aug}}$ orthogonal by definition; only physical rows corrected": the expansion matrix `E` is zero on the added-state rows, `obc_gantry.py:213 to 216, 308`.
- eq:obc_corr row 1 ($f^{\mathrm n}_{\mathrm{obc}}=[\varphi\ \gamma]\theta_{\mathrm{aux}}$), row 2 (base + aug - obc), row 3 ($\bar x_{k+1|\tau}=g^{\mathrm n}_{\mathrm{aug}}$), evaluated at $(\tilde x^{\mathrm n}_{k|\tau},\hat u^{\mathrm n}_{k|\tau})$: `GantryOBCCorrection.forward`, `obc_gantry.py:286 to 308` (RK4 tangent at the current state and plant input; see finding 10 for the eleventh column).
- "differentiate $\theta_{\mathrm{aux}}$ with respect to $\theta_{\mathrm{aug}}$ [Eqs. (9), (10)]": Eq. (10) makes $\tilde F$ a function of $\theta_a$. Code: the coefficient keeps autograd through the reference field (`obc.py:242 to 248, 523 to 531`); the basis and `vbar` are detached (`obc.py:356 to 363`, `obc_gantry.py:217, 231 to 232`).
- "solve eq:obc_coef at every evaluation of $V$": `interconnect.py:1407 to 1408`, `self.obc.maybe_refresh(_epoch, _corr.expansion_point)` / `_theta = self.obc.coefficient(...)` (MATCH OK), once per `loss` call.
- "At the start of each epoch $\beta_{\mathrm{lin}}$ is set to the current $\beta$ and held fixed": `OBCLifecycle.maybe_refresh`, `obc.py:501 to 509`, `obc_refresh='epoch'`. Code exception: on a single rank loss the previous basis is retained (`obc.py:460 to 478`); the section deliberately leaves the rank fallback out.
- "Gyorok update $\beta_{\mathrm{lin}}$ at every iteration or keep it fixed at the nominal parameters [Sec. 4]": Gyorok 2025 PDF p. 8.
- "adapt their recomputation at the start of each epoch [Remark 2]": Gyorok 2025 Remark 2, PDF p. 7 (states, not the linearisation point; "adapt" is stated).
- eq:obc_eps, $\Xi\in\R^{6N}$, one-step discrepancy at $\beta^\star$: dimensions consistent. "unmodelled terms of Eqs. (14), (15)": Gyorok 2026 PDF pp. 3 to 4 ($\delta$, $\Delta$, $E$). Off-path `eval_overlap.py` forms $\Xi$ from the reconstructed successor `x_phys_next_rec` minus the baseline at $\beta^\star$, as in the text.
- Inline "one-step error $\Xi-\Phi(\beta-\beta^\star)-\tilde F$": sign and terms derived by hand (see finding 1 for the condition).
- eq:obc_recovery row 1 ($\hat\beta$) and row 2 ($\hat\beta_{\mathrm{np}}$): normal equations of $\min_\beta\lVert\Xi-\tilde F-\Phi(\beta-\beta^\star)\rVert^2$ and of the version with $F$; correct given finding 1.
- "With the projection $\hat\beta$ does not depend on the network output, since $\Phi^\top\tilde F=0$": first ten rows of $\Phi_\Gamma^\top\tilde F=0$ (under finding 1).
- Moved sentence "The offset does not change the first-order estimate": without $\Gamma$, $\Phi^\top\tilde F'=0$ still holds, so the same $\hat\beta$ results.
- "$\Phi^\top F$ still in $\hat\beta_{\mathrm{np}}$ [Eq. (25)]": Gyorok 2026 Eq. (25), PDF p. 4, $\lVert\theta^\star-\hat\theta\rVert=\lVert(\Phi^\top\Phi)^{-1}\Phi^\top(F-\Delta)\rVert$; same sign structure.
- "same form as [Eq. (21)] ... assuming the model exactly recovers the dynamics on $\mathcal D_N$ [Assumption 6]": Gyorok 2026 Eqs. (20), (21), Assumption 6 (Eq. (18)), PDF p. 4.
- "estimate equals $\beta^\star$ iff $\Phi^\top\Xi=0$ [Condition 4, Remark 5]": $(\Phi^\top\Phi)^{-1}$ nonsingular gives the iff; Gyorok 2026 Condition 4 Eq. (17) and Remark 5, PDF p. 4.
- "This holds for the one-step error, not for the closed-loop loss eq:aug_loss": correct limitation.
- eq:obc_overlap row 1, $\rho(Z)=\max_{Q\in\ran\Phi\setminus\{0\}}\langle Q,Z\rangle/(\lVert Q\rVert\lVert Z\rVert)=\lVert\Pi Z\rVert/\lVert Z\rVert$: Cauchy-Schwarz with $\langle Q,Z\rangle=\langle Q,\Pi Z\rangle$, equality at $Q=\Pi Z$; if $\Pi Z=0$ every cosine is 0. $Z\in\R^{6N}$ nonzero.
- eq:obc_overlap row 2, $\Pi=\Phi(\Phi^\top\Phi)^{-1}\Phi^\top$: symmetric, idempotent, rank 10 under full column rank, so it is the orthogonal projector onto $\ran\Phi$.
- "zero if and only if $\Phi^\top Z=0$": $\Pi Z=0\iff\Phi^\top Z=0$ (since $\Phi(\Phi^\top\Phi)^{-1}$ is injective). The Section V measure in `eval_overlap.py` uses $\Phi$ (10 columns) by least squares, consistent. The main-path validation diagnostic `rho_overlap` (`obc.py:571 to 575`) uses the 11-column basis in the affine arm: a different quantity with the same name, not the thesis $\rho$.
- Fig. app_gantry_floating caption (99_appendix.tex:881 to 889): parallel torsion springs and dampers acting on the same $\Theta$ add, giving $k_{b,\Sigma}=k_{b1}+k_{b2}$, $c_{b,\Sigma}=c_{b1}+c_{b2}$; consistent with eq:damping_matrix, eq:stiffness_matrix and eq:phi. "Both ends held locks $\Theta$ for a rigid cross-arm": exact rigid-body kinematics (the joint distance would have to grow as $L_b/\cos\Theta$).
- 99_appendix.tex:864 wip bullet (stacked projector, rank-deficient pseudoinverse, epoch refresh, zero extension): a to-do list, no claim to check beyond finding 12.


### Slice F


- 05:52 to 54 and implicit f_b: all f_b ratios are consistent with f_b = 100 Hz (tdg_config.m cfg.fbw = 100): 106 Hz = 1.06 f_b, 75 Hz = 0.75 f_b, f_a = 150 Hz = 1.5 f_b.
- eq:setup_fric (05:99 to 100): actuator velocities P^T qdot (q_act = P^T q from eq:Ptransform), generalised friction force P F_fr, F_fr,i = c_c,i tanh(v_i/v_fr): matches gantrySystemExtendedTanh.m:74 to 80 (`F = cc .* tanh(g * v_stage)`, :75, MATCH OK), g = 1000 s/m so v_fr = 1/g = 1 mm/s = \SetupFricVzero (tdg_config.m:37).
- 05:107 to 110 limit to c_c sign(v): tanh(v/v_fr) to sign(v) for |v| >> v_fr; Garcia Eq. (6) (PDF p. 3) uses c_c sign of each actuator velocity, mapped by the same P: matches.
- 05:111 to 112 v_fr in 0.1 to 10 mm/s: 1 mm/s in range.
- 99:708 to 709: tanh(3) = 0.99505, so within 0.495\,% < 0.5\,% of the Coulomb law for |v| >= 3 v_fr.
- 05:156 to 157 c_c from Garcia Table I: 16.8, 18.35, 11.6 N (Garcia Table I, PDF p. 4) = tab:nominal_parameters and tdg_config.m:28.
- eq:setup_abs and M_e, M_a, C_e, K_e (05:126 to 142): derived from the kinetic energy of m_a at (X - (Y+ell)Theta, Y + q_a - d Theta) minus the share already in M(Y) with total m_h: X-Theta -m_a ell, Theta-Theta m_a ell(2Y+ell), q_a column m_a[0 -d 1], q_a-q_a m_a; C_e, K_e block diagonal. Matches gantrySystemExtendedTanh.m:34 to 53 with mh = mh_rigid (tdg_replica.m:21, gtd_run_simulation.m:68).
- 05:120 to 123 (Lagrange, no Coriolis): the q_a-dependent M_e would give extra velocity-product terms; stated as omitted, and the code has none.
- 05:144 to 145 ell = L_0 + q_a: matches code (Y + L0 + delta_a).
- 05:146 to 148 k_a = m_a(2 pi f_a)^2, c_a = 2 zeta_a sqrt(k_a m_a): tdg_config.m:46, :48.
- 05:151 to 152 fixed-step RK4 at 20 kHz: tdg_make_model.m:108 (ode4, FixedStep cfg.ts = 1/20e3).
- 05:158 to 160 anti-resonance at f_a, resonance f_a sqrt(1 + m_a/m_hr): 2-DOF derivation Y/F_Y = (m_a s^2 + c_a s + k_a)/(s^2[m_hr m_a s^2 + m_h(c_a s + k_a)]), zeros at omega_a, poles at omega_a^2 m_h/m_hr; with m_a = 0.5 m_h: 150 sqrt(2) = 212.1 Hz = 2.12 f_b; full linearisation gives 213.4 Hz (setup_band_features.csv), consistent with the "approximately".
- 05:161 to 162 both above crossover: max crossover 106 Hz < 150 Hz (setup_band_edge.csv).
- 05:165 to 168 K_fb = K_1 at T_s by Tustin: controller.py:94 to 116 (bilinear at ts).
- 05:179 to 181 SNR ranges 58 to 74 dB (X), 71 to 80 dB (Y): macros from snr.py over training records, definition at snr.py:22 to 26 = 99:563.
- 05:200, 99:617 to 620: five motion types and 12 s records; run log 47960 samples = 12 s x 4 kHz minus 2 x 20 trimmed.
- 05:203 "at least 59 dB": \SetupGapSnrMinMotion.
- 05:204 to 206 multisine in X, Theta, Y through P^{-1}: consistent with u = P u_act (eq:Ptransform).
- 05:207 to 208 band start round(106/100,2) f_b = 1.06 f_b; 99:583 to 584 Delta f = 0.005 f_b, L = {212..594}: 212 x 0.005 = 1.06, 594 x 0.005 = 2.97; tdg_config.m:100 to 101 (106 to 297 Hz, 0.5 Hz step).
- 05:213 to 216 band holds anti-resonance (150.3 to 150.5 Hz) and resonance (213.4 Hz) at all five training Y: setup_band_features.csv.
- 05:218 to 219 split 18 / 6 / 11; tab:setup_records 14 of 18 and 4 of 6 with multisine: run log (18 train, 6 val; 12 test loaded, E5 excluded from the thesis) and res_structure.csv record lists.
- 05:220 to 221 seven twins: test records without multisine are I3 to I6, E2, E3, E4, E6 (8), twins E2M, E3M, E4M, E6M, I3M, I5M, I6M (7), I4 none.
- 05:222 to 224 |Y| <= Y_tr = 0.30 m, 75\,\% acceleration and velocity: setup_coverage.csv training rows (max 0.30 m, 75.0\,\%).
- 05:227 to 230 I within, E outside one limit: E1 Y 0.39, E2 Y stroke to 0.39, E3 acceleration 99.8 to 99.97\,\%, E4 velocity 97.5\,\%, E6 acceleration 131\,\% (see finding 9 on "limit").
- 05:235 T_s = 0.25 ms, 0.1 s windows: run log nf=400 at 4 kHz.
- 05:240 to 244 K1-at-T_s mismatch 0.5\,\%, hold 10.4\,\%, reduction 2.3e-9 m: setup_numbers.tex macros.
- tab:setup_models and 05:302 to 304 detuning: run log detune = [0.9, 1.1, 1.1, 1.1, 0.9, 0.9, 1.1, 1.1, 0.9, 0.9] equals the signs of the Start column; m_Delta start: 0.1 x (-0.5)/10.45 = -0.48\,\% printed -0.5.
- 05:316 to 318 11 configurations: 9 (3 n_x x 3 widths at 3e-3) + 2 learning rates = 11 (res_bb_selection.csv).
- eq:setup_rms RMS_j and RMS_j^fl count: N-n-1 terms in the sum; matches code scoring from k0 = na = 30 (closed_loop.py:756, :801 to 808; eval_blackbox.py:147, :156 to 158).
- 05:359 to 361 sigma_e,j population std over the scored samples: data.py:233 `s = np.stack([np.std(np.asarray(e, np.float64)[k0:], axis=0) for e in e_list])` (MATCH OK).
- 05:362 NRMS 100\,\% = predicting the mean servo error: yhat = r + mean(e) gives RMS = sigma_e.
- 05:363 to 364 floor definition: eval_blackbox.py:156 to 158 (noisy minus noise-free twin, same k0).
- revnote 05:373 to 374: test floors 0.0053 to 0.105\,\%, smallest model value 2.35\,\% (baseline E3 Y): res_generalisation.csv.
- eq:setup_delta and 05:396 to 397: res_obc_combinations.py:35, :99 to 101 (mean actuator mass 10.45 kg for m_Delta, |true| otherwise); true m_Delta = -0.5 kg.
- 05:399 to 404 overlap on tuples of train/val/test (I1 to I6), zero on training tuples: res_obc_overlap.csv n_tuples 863190 = 18 x 47955, 287730 = 6 x 47955; OBC train overlap 2.5e-15, 2.3e-15, 1.1e-14.
- tab:res_structure all cells: means over I1, I2 and over I3 to I6 of res_generalisation.csv per-record NRMS_e (e.g. OBC Y (6.268 + 6.166)/2 = 6.22; floor X1 (0.0373 + 0.0833)/2 = 0.060); caption "mean over test records I1 to I6" correct.
- 06:66 to 67 "more than an order of magnitude on Y, halves X": 116.6/6.2 = 18.8; 9.7/18.8 = 0.52, 9.5/20.7 = 0.46.
- 06:68 "without multisine similar": 4.7/4.8/3.8 against 4.5/4.6/3.4.
- 06:123 to 124 dampers deviate most, masses within 3.4\,\%: tab:res_obc.
- 06:125 to 126 equal performance: full-model NRMS_e within 0.8 points (see finding 8 for the conclusion wording).
- tab:res_obc NRMS_e block "test records" = mean over I1 to I6 of all six records, e.g. OBC X1 (2 x 9.7 + 4 x 4.7)/6 = 6.37, U Y (2 x 6.3 + 4 x 4.0)/6 = 4.77; train block (14 x 9.4 + 4 x 4.1)/18 = 8.22.
- tab:res_obc Norm: start sqrt(900.25) = 30.0, U 83.08, OBC 47.72; tab:app_obc_full correct-start norms consistent.
- 06:127 to 129 overlaps 0.13 against 0.84, truth 0.59: table (provenance in finding 6).
- 06:130 correct start OBC closer: 35.4 against 58.3. 06:131 n_a = 0 reversed: 68.4 against 48.8.
- 06:155 to 159 without multisine "as well or slightly above": final minus baseline per record and axis -0.19 to +1.18 points; black box 74.5 to 193.4\,\% on the eight records without multisine; black box below the final model only on I2 X2 (10.19 against 11.42; E1 X2 13.221 against 13.216 is above, by 0.005); res_generalisation.csv.
- 06:160 to 161 twins: final Y 4.0 to 9.3\,\% (all below 10), baseline Y 47.7 to 86.8, black box Y 74.6 to 111.7 (res_errors.csv split twins).
- 06:184 to 193 (VII-D first paragraph): added states and PS needed: tab:app_structure val Y 6.4 (PS, n=2) against 78.1 (plain) and 86.5 (n=0).
- 06:194 to 195 revnote 3.8 against 3.4\,\% on Y: tab:res_structure.
- 06:201 to 202 47.7 against 83.1: tab:res_obc.
- 06:215 to 218 starts closer with OBC: difference norms 25.1 (OBC) against 36.4 (U) from tab:app_obc_full.
- 06:220 to 222 black box poor on training (32 to 104\,\%) and not converged: best update 78000 of 80000 (res_bb_selection.csv).
- 06:249 to 252 SUBNET independence assumption and correlation: consistent with eq:system_loop text (02:210 to 212).
- 08:84 to 85 116.6 to 6.2\,\%; 08:90 to 92 black box only better on X2 of I2: as above. 08:86 at baseline level without multisine: as above.
- tab:app_splits all rows: res_structure.csv (train, val, test) to one decimal; "physical part" = learned block and OBC theta_frozen zeroed (closed_loop_report.py:38 to 64).
- tab:app_bb_search: selection rule (48.2 lowest, next 52.5 not within 2\,\%); parameter count 16518 for n_x = 16, width 16, encoder 32, na = nb = 30, inputs r and f (6), outputs 3: f 1280 + h 646 + encoder 14592 = 16518 under MLP-plus-linear-bypass nets (arithmetic check only).
- eq:setup_K1 (99:536 to 543): ruleOfThumb.m:2 to 14 (omega_b/6, omega_b/3, 3 omega_b, 10 omega_b, gain 1/|sys Cnorm| at omega_b, Tustin at ts); designed on the rigid plant at Y = 0 per actuator diagonal (gtd_build_plant.m:22 to 27, :46 to 49).
- 99:544 to 546 Tustin at f_sim and at T_s: ruleOfThumb.m:14, controller.py:116.
- eq:setup_noise (99:552 to 561): tdg_noise_psd.m:31 to 37 (g^2 |H_butter2|^2 + floor^2, cutoff fc); units m/sqrt(Hz) consistent with a one-sided PSD in m^2/Hz; floor = median above 5 kHz (tdg_config.m:126 to 130); fit of the simulated standstill error at Y = 0 to the measured one (NOISE-INJECTION.md 10.3).
- 99:562 to 564 SNR_j definition: snr.py:22 to 26.
- eq:setup_ms (99:578 to 582): unit-RMS multisine times A_c (gtd_make_multisine.m:96 to 98, :51): RMS of A_c sqrt(2/|L|) sum of |L| unit cosines on distinct lines is A_c.
- 99:586 to 588: 2 zeta_a f_a = 2 x 0.03 x 150 = 9 Hz = 0.09 f_b; 0.09/0.005 = 18 lines; period 1/(0.005 f_b) = 2 s.
- 99:589 30 phase draws, lowest crest factor: tdg_config.m:99.
- 99:594 to 597 crossover 0.75 to 1.06 f_b: setup_band_edge.csv (74.76 to 106.08 Hz).
- 99:645 E1 |Y| = round(0.39/0.30, 1) = 1.3 Y_tr; 99:647 to 648 E3 100\,\%, E4 98\,\% (97.5).
- eq:setup_decim (99:686 to 692): data.py:171 (block mean, MATCH OK), data.py:184 `return r[::D] + sosfiltfilt(sos, d['y'] - r, axis=0)[::D]` (MATCH OK), butter order 8 at 1600 Hz (config.py:37), forward and backward (sosfiltfilt).
- 99:701 to 705: filter acts on y - r only (code above); 2.6 ms < 5 ms trim; trim 5 ms = 20 samples at 4 kHz (run log).
- tab:setup_checks rows: macros of setup_numbers.tex; noise above 2 kHz is above the 4 kHz Nyquist; 1.6 kHz = 0.8 of the Nyquist frequency.
- 99:742 to 747: same macros as 05:240 to 243.
- eq:setup_bb (99:780 to 787): train_bb_clmap.py target e = y - r (:11 to 13, :105), inputs (r, f) (:103), prediction r + e_hat (:114 to 120), encoder on past inputs and servo errors, na = 30 (log).
- tab:hyper grey-box rows: run logs (4 kHz, up_sample 1, nf 400, stride 10, na_nb 30, 16x2 tanh, float64, lr 1e-5, eps 1e-16, batch 512, 200 epochs; 167 updates per epoch x 200 = 33400; 26 validations = 33400/1300 + initial), L-BFGS (m = 50, max 100, 16384 windows, tol 2.6e-5, 2.6e-7; inner 20 from config default), gamma 0.99 (blocks.py:1301), encoder least squares (rank 128 of 180 regressors, 86274 windows in the log); Hoekstra encoder Eq. (33) is the linear least-squares encoder fit (PDF p. 5): matches.
- tab:hyper PS rows: run log spec (aux_batch 64, lr 1e-3, eval_every 25, s1_min 100, s1_cap 350, s1_tol 0.02, s1_window 4, screen_at 150, screen_max 0.9, n_cal 128, n_snap 8192, n_snap_cal 2048, s2_batch 512, s2_check 100, s2_min 500, s2_cap 8000, s2_tol 0.01), "M 30" = H = n+1, S2 plateau 5 checks (ps2_opening.py:248), held-out records TR-S5, TR-Y3, TR-P4, TR-T4 = last of each class with at least 3 records (L has 2).
- tab:hyper black-box rows: bb log (n_its 80000, batch 256, e-width 32 = 2 x 16, na 30); 34236 windows over 18 records consistent with stride 25.
- revnote 99:801 Hoekstra Tables 5 and 7 are hyperparameter tables (PDF pp. 11, 14); revnote 06:143 Gyorok Fig. 2 shows estimates over Monte Carlo runs (PDF p. 7): matches.
- 01_introduction printed text: no formula or number besides finding 2.
- 00_abstract, 07_discussion: no printed formula or claim (wip / commented).


## Not checkable

Resolved by the lead and moved to Correct: E's items on the rank of $\Phi_\Gamma$ at every refresh and on which OBC space the Section VI runs used (all six 18-10-2026 OBC logs: space=affine, 52 refreshes at rank 11/11, no rank loss).

### Slice A

- 02:175, $v_k$ zero-mean and coloured, and 02:210 independence from $r,u^{\mathrm{ff}}$: properties of the generated data (Section V noise model), not of this slice; would be settled by the noise generator settings and a sample mean/spectrum of $v$.
- 02:269, "This model does not capture all dynamics of the system": settled by the data (baseline output error versus noise floor), Section V/VI quantity.
- 02:197 to 199 versus the data: whether the data's controller and hold actually run at $T_s$ (the comment says deviations are named in Section V); outside this slice.
- Fig. 1 caption colour statement (red actuator, blue generalised coordinates, $Y$ in both): figure content, not mathematics; the figure source draws $Y$ in red only (figures/gantry_system_with_absorber.tex:123 to 128), so "with $Y$ in both" may refer to its double role, not checked further.
- app:frf, size of the model curve deviations: needs the numerical run (res_frf_full.csv); not a claim of the printed text in this slice.

### Slice B

- gautier1990minimum (02:798-799, \ucite): PDF in References/ is a scan with no text layer (PyMuPDF returns only the IEEE footer); passage not verifiable without OCR.
- Fig. fig:lfr (figures/lfr_loop): figure source not in the snapshot; only the caption was checked.
- wip block 99:320-328 (Drenth Thm. 6, $\rho(D_{zw})<1$, overbounding): treated as not printed (excluded in the review build); not audited.
- Whether $\mathcal D_N$ makes $\theta_{\mathrm{base}}$ unique (finding 2): needs a numerical run; settling quantity is the rank and condition number of the stacked output sensitivity $\partial\hat y/\partial\theta_{\mathrm{base}}$ over $\mathcal D_N$ at $\theta^0_{\mathrm{base}}$.
- Garcia-Herreros source of eq:mass_matrix (outside this slice, 02:446-453): only used as given; code gantry_ss.py matches it.

### Slice C

- 99:743-747 median force mismatch \SetupKmismatchMed % and hold change \SetupKholdMed %: values come from `Thesis-writeup/Code/Setup/controller_rate.py` (not on the main path, not run); settling finding 4 needs RMS of $u_{\mathrm{act},k}-\check u_{\mathrm{act},k}$ on the training records.
- 03:509 Bishop 1995 claim: no PDF in References/ and no bib entry.
- 03:761-767 "the feedback controller is able to correct this drift": quantitative claim needs the open-loop vs closed-loop drift on a record (the todo at 03:797 names it); qualitatively consistent with a stabilising $K_{\mathrm{fb}}$.
- 03:563-566 "the scale of $\bar x$ is set by the network initialisation and its normalised inputs": a property of training, not of an equation; needs the trained $\bar x$ statistics.

### Slice D

- 03:1251-1252 (also finding 3): whether K_fb corrects the offset for the closed-loop model. This would be settled by the spectral radius (< 1) of the linearised closed loop of eq:aug_residual_loop over the Y range at theta^0 and at the final theta.
- 03:1577-1578: whether minimising V actually leaves the missing dynamics out of x̄. This is a possibility statement. It would be settled by the Section VI comparison of the added-state content (for example, output error with and without PS on held-out records).
- The quality of the start eq:aug_enc_fit (how well W^{b,0} reconstructs the states). Numerical: the per-state unexplained fraction printed by least_squares_start (model.py:451-460; the code stops the run above 1e-3).
- The S1 conditional-mean statement in the population sense (finding 2): whether p∘psi_a reaches E[e given chi]. Numerical: the calibration nu at the end of S1, compared with a nu from an unrestricted regressor on chi_tau.
- Venkatraman 2017 (03:1784): only inside \todo, not printed.

### Slice E

- Quantitative validity of eq:obc_recovery for this system: settled by comparing the predicted bias $(\Phi^{\star\top}\Phi^\star)^{-1}\Phi^{\star\top}\Xi$ with the observed $\hat\beta-\beta^\star$ (the revnote reports a mismatch of 100 to 1300 percent against a few to 30 percent).
- Velocity reconstruction under noise: eq:obc_tuples applies the stencil to the measured $y$, which in the noisy thesis mode is the anti-alias-filtered, decimated pipeline $y$ (`obc_gantry.py:91 to 103`, D-232). How much reconstruction error this puts in $\Xi$ needs a numerical comparison of the reconstructed against the true velocity.
- Kessels Sec. 5.3.1.3 only defines negation qualitatively. Whether the $\Gamma$ column reduces negation in the closed loop has no code counterpart (the measure was cut).

### Slice F

- 05:113 to 114 (higher v_fr gives viscous friction under the multisine, lower keeps the standstill error above the noise): needs the D-223 sweep; quantity: standstill servo-error RMS and rail-velocity share vs v_fr.
- 05:209 to 212 and 99:598 to 604 upper band edge 2.97 f_b keeping 90\,\% of the squared BLA difference: needs the BLA computation; quantity: cumulative share of |G_sys - G_base|^2 above the crossover at 297 Hz; also "a rerun with eq:setup_fric gives the same edges".
- 05:163 to 164 "the feedback does not suppress them": qualitative; quantity |S_33| over 140 to 220 Hz (NOISE-INJECTION.md 10.5 reports 0.78 to 1.05).
- tab:setup_checks phase-margin loss, time constants, decay times, alias RMS, K1 errors: generated macros; recomputation not allowed.
- 99:589 to 590 crest factor on the simulated yaw response for Theta; 99:591 to 592 every record within force and stroke limits without scaling: generator internals (gtd_size_anti_amp.m, gtd_enforce_limits.m) not run.
- eq:setup_bb input $u^{\mathrm{ms}}_k$: the code feeds the record field f_sim; whether f_sim is the generalised u^ms or the actuator u^ff = P^{-1}u^ms was not traced in the generator.
- 06:211 to 214 (orthogonality only first order and at xbar = 0; added states enter through f_aug unprojected): method statement, agent for Section IV.
- tab:res_all_splits physical-part rows for U and the train/val physical parts: not in the CSV columns extracted (res_errors.csv run R4 physics rows exist; not cross-checked cell by cell).
- Whether HEAD reproduces the 18-10 numbers: run logs carry no git provenance (header note).
