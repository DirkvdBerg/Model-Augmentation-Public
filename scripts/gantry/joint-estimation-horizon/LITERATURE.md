# Literature for G3 (deep-research skill, D-121; two subagents, 2026-09-24)

Scratch files, helper scripts and downloaded PDFs: `outputs/lit/A/`, `outputs/lit/B/`.
TU/e browser route: not run (excluded for subagents); no item needed it except Borkar 1997,
Golub and Pereyra 2003 and Gaskin's thesis (marked needs-browser-route, cited second-hand or not
at all). Verification level is given per claim; "grep" means a full-text search of the PDF with the
passage read in context.

## A. Horizon, informativity, closed loop

| Source | Claim used here | Level |
|-|-|-|
| Beintema, Schoukens, Toth, Automatica 156 (2023) 111210, DOI 10.1016/j.automatica.2023.111210 | Sec. 3.4: choose T "a few times the largest characteristic time scale"; Sec. 5.4 (Fig. 3): overfitting when T is below the dominant time scale; T tuned on k-step NRMS, never against parameter bias or variance. Thm. 9: loss Lipschitz constants grow as O(L^2T) once L > 1. Sec. 4.3: overlapping subsections are more data-efficient than non-overlapping multiple shooting | grep, passages read |
| Beintema, Toth, Schoukens, L4DC 2021, PMLR 144:241-250 | T = 80, about 4x the system time scale, checked post hoc | grep |
| Gyorok et al. 2025 L4DC (held as `literature/Orthogonality/Hoekstra - Orthogonal projection-based regularization...pdf`) | Sec. 5.3: T = 15, "roughly 1.5 times the largest characteristic time scale"; p. 4 describes an OPTIONAL encoder for partial state measurement | grep. Note: the project memory says this paper uses measured states and no encoder; both can hold (experiments with measured states, encoder described as an option); reconcile before citing |
| Hoekstra, Verhoek, Toth, Schoukens, EJC 86 (2025) 101304 | SUBNET truncated loss verbatim; T = 200 samples (3-DOF MSD); no T rule, no sensitivity study | grep |
| Hoekstra, Gyorok, Toth, Schoukens, LFR augmentation of FP models (preprint, `literature/closed-loop-id/`) | T = 200 (MSD), T = 40 (F1Tenth, "based on previous black-box results"); no sensitivity study | grep |
| Ribeiro, Tiels, Umenberger, Schon, Aguirre, Automatica 121 (2020) 109158 | In non-contractive regions the Lipschitz and smoothness constants of the simulation loss grow exponentially with the simulation length; multiple shooting as the fix | abstract page read |
| Galioto, Gorodetsky, Physica D (2024), DOI 10.1016/j.physd.2024.134146 | Sec. 3.2.4: the horizon acts like a process-noise prior; longer T concentrates the likelihood for contractive modes, adds little for rho(A) >= 1 | grep |
| Forssell, Ljung, Automatica 35(7) (1999) 1215-1241 | Eq. 34-35: closed-loop informativity needs reference-driven input power; Eq. 39: under feedback the covariance couples input and noise spectra and is generically worse than the open-loop diagonal form | pages read (1998 preprint text) |

**Gap, graded provisional but consistent (three independent null searches):** no source derives the
Fisher information of a truncated or multiple-shooting objective as a function of the window length
and a mode's time constant, and none ablates the horizon against parameter bias or variance. The
field rule is "T a few times the slowest time scale", set on output error. G1 is therefore a
measurement the literature does not have.

## B. Staging, timescales, projection

| Source | Claim used here | Level |
|-|-|-|
| Tuo, Wu, Ann. Statist. 43(6) (2015), DOI 10.1214/14-AOS1315 | Eq. 4.1: OLS calibration, a physics-only least-squares fit with no discrepancy term, is CONSISTENT for the L2-projection calibration parameter (the same point as L2 calibration) but not semiparametric-efficient | grep |
| Plumlee, JASA 112(519) (2017), DOI 10.1080/01621459.2016.1211016 | Defining the parameter "implicitly requires orthogonality of the bias function and an aspect of the computer model"; the orthogonalising kernel needs "the gradient of the computer model" (Sec. 4.1) | grep |
| Tuo, SIAM/ASA JUQ 7(2) (2019); Xie, Xu, JASA (2020); Wang, SIAM/ASA JUQ 10(2) (2022) | Residual orthogonal to span(d y_s / d theta) is the first-order condition of the L2 projection; with a nonparametric discrepancy theta is only defined by that projection | read in earlier sessions (repo) |
| Chernozhukov et al., Econometrics J. 21(1) (2018), DOI 10.1111/ectj.12097 | Def. 2.1: Neyman orthogonality = the moment condition is first-order insensitive to the nuisance | grep |
| Konda, Tsitsiklis, Ann. Appl. Probab. 14(2) (2004), DOI 10.1214/105051604000000116 | Two-timescale SA: the auxiliary block is FAST and the parameter quasi-static (Assumption 2.3, beta_k/gamma_k -> eps); Borkar 1997 second-hand | grep |
| Gyorok et al., IFAC J. Syst. Control 35 (2026), arXiv 2511.01321 | Physical and network parameters trained jointly with one Adam chain, then L-BFGS; no staging, no separate rate | grep |
| Kon et al., ACC 2022, DOI 10.23919/acc53348.2022.9867653 | Physical parameters and network optimised jointly under one criterion | grep |
| Newman, Ruthotto, Hart, van Bloemen Waanders, SIAM J. Math. Data Sci. 3(4) (2021), DOI 10.1137/20m1359511 | Variable projection eliminates a block that enters LINEARLY in closed form; it does not apply to a whole nonlinear network | grep |
| von Stosch et al., Comput. Chem. Eng. 60 (2014), DOI 10.1016/j.compchemeng.2013.08.008 | Taxonomy of hybrid-model training: direct (staged), indirect (joint through sensitivities), incremental (staged sub-problems, then simultaneous refinement) | grep |
| Bradley et al., Comput. Chem. Eng. 166 (2022), DOI 10.1016/j.compchemeng.2022.107898 | p. 15: whether to estimate mechanistic parameters simultaneously or sequentially with the data-driven part is an open question needing further research | grep |

**Negative, graded moderate (two Scholar queries, no venue enumeration):** no verified source uses a
separate learning rate for the physical parameters of a hybrid model with a measured effect (one
thesis abstract claims a "component-wise learning rate", unverified).

## What the two sweeps imply for G3
1. The staged estimator has a direct statistical precedent in the STATIC setting: physics-only least
   squares is consistent for the projection-defined parameter (Tuo and Wu), which is the static form
   of R2. No source extends it to a dynamic, simulation-error objective; G2 measures that extension.
2. Two-timescale theory says the network should be the fast block, which is the observed situation;
   the gap is not an optimiser pathology that a rate split is known to fix, and the rate split has
   no verified precedent.
3. The horizon rule of the SUBNET line is set on output error at a few times the slowest time
   scale; for the gantry that is 3 to 4 s, 30 to 40 times the production window, and nobody has
   measured what it buys for parameters. In closed loop the informativity argument (Forssell and
   Ljung) says the feedback, not the window, may be what limits the slow parameters. G1 measures it.

## C. Window, encoder lag, training settings, friction memory (deep-research skill, one subagent plus parent reading, 2026-09-27)
Session: `tasks/handoffs/2026-09-27-training-hyperparameters-and-window.md`. Scratch and PDFs: session scratchpad `lit-train/` (not kept). TU/e browser route not run; needs-browser-route: Hensen 2003, Retzler et al. AMC 2022 (DOI 10.1109/amc51637.2022.9729299, grey-box shooting, likely on-question, unread), Aeyels 1981, Takens 1981, Stark 1999, Levin and Narendra 1995, Van Essche et al. Nonlinear Dyn. 2026 (DOI 10.1007/s11071-025-12134-8, unread).

### C1. Window T
| Source | Claim used here | Level |
|-|-|-|
| Hoekstra, Gyorok, Toth, Schoukens, arXiv 2602.17297 (2026), held `literature/closed-loop-id/hoekstra2026_lfr-augmentation-fp-models.pdf` | Eq. 22a: truncated loss scored from l = 0 (no burn-in). MSD Table 5: T = 200 at Ts 0.02 s (4 s); F1Tenth Table 7: T = 40 at 40 Hz (1 s), lag "based on previous black-box results"; no T rule. Integrators detached on F1Tenth: "the integrator dynamics make it difficult to obtain accurate long-term predictions". Sec. 5.5: consistency via Beintema 2023 conditions, incl. an incrementally exponentially output stable system | read in full (parent) |
| Hoekstra et al. EJC 86 (2025) 101304 (held) | Table 6: T = 200 (3-DOF MSD); Table 8: T = 500 (Bouc-Wen); ANN-SS "trained for 10 000 epochs" against 3000 for the augmentations | grep (parent) |
| Hoekstra, Gyorok, Toth, Schoukens, arXiv 2602.13108 (2026), held `literature/augmentation/Encoder initialisation methods in the model augmentation setting.pdf` | Table 2: T = 200 at Ts 0.1 s (20 s); Fig. 2: encoder initialisation matters most at short horizons ("the decaying impact of the initial state") | read in full (parent) |
| Beintema, PhD thesis TU/e 2024 (held), Sec. 2.4 (PDF p38) | "For oscillatory systems (i.e. systems with an integrator) one should pick T to be close to the characteristic timescale of the input and for unstable systems, one should pick T to be close to the Lyapunov time of the system." Not in the Automatica version | grep, passage read (parent verified) |
| Beintema thesis Sec. 2.5.2 (PDF p41) | "we will limit ourselves to open-loop identification as is done in Ljung 1978": the SUBNET consistency proof, and so Hoekstra's Sec. 5.5, does not cover closed-loop data | grep, passage read (parent verified) |
| Beintema et al. Automatica 2023 Sec. 5.4 (held); Beintema 2021 Sec. 3.1 | T read off the k-step NRMS curve: "the transient error has a length of 30 time steps, which approximately corresponds to the characteristic timescale"; T = 80 validated "as the transient does not dominate the loss function". Output error only | grep (agent) |
| Ribeiro et al. Automatica 121 (2020) 109158 (held), Sec. 6.2, Fig. 5, Table 2 | Linear OE, 100 Monte Carlo runs, PARAMETER histograms: multiple shooting with continuity constraints gives parameter distributions "Independent of Delta m_max"; truncated simulation without continuity (MSA-PEM) is non-monotone in K for the slow system ("K = 10 or K = 20 ... bimodal ... K = 7 yields the best results") | read (agent) |
| Forgione, Mejari, Piga, arXiv 2206.12928 (2022) | Factorial designs (768 and 432 runs): "The training time has by far the largest impact"; the longest fitting length 320 has "a negative impact"; long sequences need "more time/iterations for convergence". Output FIT only | read (agent) |
| Forgione, Piga, EJC 59 (2021) 69-81, DOI 10.1016/j.ejcon.2021.01.008 | Table 1: test R2 vs subsequence length 2 to 256, rising then flat; rule "m chosen within [32 128]"; free initial states, no encoder | read (agent) |
| Aicher, Foti, Fox, UAI 2019, arXiv 1905.07473 (held) | Eq. 9: smallest truncation with relative gradient bias below delta, from the geometric decay of backpropagated gradients. Beintema thesis p27: SUBNET avoids truncated-BPTT gradient bias | grep (agent) |
| Chiuso, Annu. Rev. Control 41 (2016) 24-38, DOI 10.1016/j.arcontrol.2016.04.013 | FIR truncation "just supposed to be 'large'"; the stable-spline decay rate "could be estimated from data using the marginal likelihood": a data-derived memory length | grep (agent) |

### C2. Encoder lag and fair comparison across orders
| Source | Claim used here | Level |
|-|-|-|
| Beintema thesis Sec. 2.5 (PDF p41) | Unbiased reconstruction needs n = na = nb >= nx - 1; "for general nonlinear systems reconstructability using Taken's embedding it can require lags up to 2·nx + 1 (Aeyels 1981)"; the variance "can be ... further reduced by increasing n". Source of the code's `2(nx_phys + nx_ann) + 1` (config.py `na_nb`), which the code labels THEORY without this citation | grep, passage read (parent verified) |
| Hoekstra 2026 (both), EJC 2025 | Lags 7 (6 states), 9 (4 states), 12 (8 states, from black-box results), 13 (Bouc-Wen); no rule stated. Within each study T, lag, width, epochs, batch are FIXED across structures and state counts; EJC gives the black box more epochs | read (parent) |
| Beintema 2023 Sec. 5, 5.6 | One-at-a-time study with the rest fixed; "20 >= n = na = nb > 4 perform much better"; "the choice of lag windows has a less strong impact on performance than the choice of T and nx" | grep (agent) |
| Kessels et al., Nonlinear Dyn. 113 (2025) 17335-17363, DOI 10.1007/s11071-025-11092-5; Kessels thesis TU/e 2025 (held) Ch. 5, App. D.3 | Closed-loop augmentation with added states n_ext in {0, 2, 6, 10, 14, 24, 34}: "hyperparameter values are manually tuned and identical for all EA models"; widths by a fixed rule; validation loss saturates from n_ext = 6, non-monotone beyond; SUBNET comparator capacity-matched and tuned "with an equal amount of effort" | grep, pages read (agent) |
| Melis, Dyer, Blunsom, ICLR 2018, arXiv 1707.05589; Lucic et al., NeurIPS 2018, arXiv 1711.10337 | ML standard: per-model tuning at matched parameter count; report distributions over seeds, not the best run | grep (agent) |
| Schoukens, Champneys, Beintema, Rogers, Data-Centric Eng. (2026), DOI 10.1017/dce.2026.10067 | Black-box baselines tuned per benchmark on validation, 5 to 10 initialisations | read (agent) |

### C3. Training settings
| Source | Claim used here | Level |
|-|-|-|
| Beintema 2023 Sec. 3.4; thesis Alg. 1, p46 to p50 | Zero-mean unit-std normalisation; batch "the smallest size which only marginally compromises the data throughput" (256); Adam lr 1e-3 "sufficient"; early stopping on validation simulation error; final full-batch search "unnecessary" (0.101 % to 0.0987 % NRMS) | grep (agent) |
| Hoekstra 2026 Eq. 28 | u, y and baseline states normalised to unit std; Adam only; 3000 epochs | read (parent) |
| Hoekstra encoder paper Table 3 | jax-sysid encoder pre-fit: 50 Adam epochs, 200 L-BFGS-B epochs, memory 50 | read (parent) |
| Gyorok et al. IFAC JSC 35 (2026), arXiv 2511.01321 (held) | Adam 500 epochs then L-BFGS 1000 iterations; MSD with Stribeck friction: 2500 Adam and 500 L-BFGS epochs; whole-record loss | grep (agent) |
| Kessels thesis App. D.3 | Adam lr 1e-3, at most 1000 epochs, T = 200 (4 s), patience 50; min-max scaling gives a non-zero steady state in an augmented model, so plain scaling is recommended | read (agent) |

### C4. Friction and memory in closed loop
| Source | Claim used here | Level |
|-|-|-|
| Armstrong-Helouvry, Dupont, Canudas de Wit, Automatica 30(7) (1994) (held) | Table 2: Coulomb only gives "No hunting for PID control"; static + Coulomb + viscous "predicts hunting under PID control" | grep (agent) |
| Olsson, Astrom, IEEE TCST 9(4) (2001) 629-636, DOI 10.1109/87.930974 | Stick-slip needs F_S > F_C; a limit cycle with F_S = F_C is sustained by an unstable controller | read (agent) |
| Beerens et al., Automatica 107 (2019) 483-492, DOI 10.1016/j.automatica.2019.06.017 (TU/e Pure) | Under PID with friction, settling needs "the integrator buffer ... to deplete and refill", taking "increasingly more time with a decreasing position error"; stick periods of 10 to 20 s on an industrial stage | read (agent) |
| EMPS practice (CT-SUBNET T = 0.2 s; Forgione-Piga 0.32 s; dynoNet whole record) | No paper sets its window from friction or loop memory | read (agent) |

### C5. Negative claims, graded
- No study varies T and reports physical-parameter bias or variance with co-estimated parameters, a learned part, or closed-loop data (nearest: Ribeiro, linear open loop). Moderate, provisional; agrees with section A.
- No stated rule for encoder lag against added-state count; 2nx + 1 appears only as Beintema's worst-case bound. Moderate.
- No identification paper sets its window from the closed-loop time scale or friction dwell. Provisional (dblp bot-walled, OpenAlex search paused).
- No citable sampling-rate rule verified. Coverage gap.

### C6. What it implies here
1. The line's T rules are output-error rules; for integrator plants Beintema ties T to the input time scale. The thesis trains in closed loop, where the plant integrators sit inside a stable loop, so the loop's response duration is the relevant scale; the SUBNET consistency proof does not cover this closed-loop setting.
2. A window sweep must compare runs at convergence (Forgione 2022) and report parameters per window, which need not improve monotonically (Ribeiro).
3. Fair state-count comparison in the augmentation line fixes lag, width and budget (Hoekstra, Beintema, Kessels); 2nx + 1 scaling is a bound, not a requirement.
4. Coulomb friction alone predicts no hunting under PID, but integrator deplete-and-refill makes stick dwell last seconds at small errors (Beerens); the residual rollout carries the controller's own memory in `u_data`.
