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
