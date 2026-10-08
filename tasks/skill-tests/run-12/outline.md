# Outline: Section IV, Interpretability by Orthogonal Construction

Opening (one paragraph): Section III-A left the split between f_base and f_aug non-unique; this section removes the one-step overlap by adapting OBC [gyorok2026obc] to the self-scheduled state equation; names IV-A, IV-B, IV-C.

## IV-A Negation in the augmented state equation
Job: state precisely which directions the network can negate, for the gantry.
- P1 topic: the physical parameter directions of the one-step transition are the columns of the sensitivity Phi at an expansion point. Eq. obc_sens (adapted from [gyorok2025l4dc, Eq. (15)]). Then Eq. obc_negation: a learned write cancels a displacement to first order; the data cannot attribute it; this is negation [gyorok2026obc, Ex. 2], [gyorok2025l4dc, Ex. 1]. Aim: an estimate not selected by the initial pair.
- P2 topic: for the gantry, each direction is a generalized mass, damping or stiffness force. Eq. obc_gantry_sens (own derivation from (eom)). M(Y)^-1 makes the directions depend on Y, so the protected space must be sampled over the Y range.
- P3 topic: the protected coordinates are the ten combinations of Section II-C, not the fourteen raw parameters. Reason: four structural null directions violate the rank assumption [gyorok2026obc, Ass. 1] (D-190). Training coordinates of II-C give the same range (own: invertible coordinate map).

## IV-B Orthogonal-by-construction correction
Job: construct the correction and state its gradient treatment and refresh.
- P1 topic: for a baseline linear in its parameters, Gyorok et al. subtract the least-squares fit of the stacked network output on the regressor. Eq. obc_gyorok [gyorok2026obc, Eqs. (8), (9), (11)]. Difference to the penalty [gyorok2025l4dc]: no weight trading orthogonality against fit (research-plan anchor).
- P2 topic: the regressor is the sensitivity (obc_sens), since f_base is rational in theta_base; the affine offset of [gyorok2025l4dc, Eqs. (17), (18)] is not protected because it multiplies no estimated parameter (D-192). (?) supervisor condition in D-190 is non-overlap with the output.
- P3 topic: the stack is evaluated on fixed reference tuples reconstructed from every training sample. Eq. obc_tuples. Reason: only positions measured, a baseline simulation drifts on free X and Y (Section III-C); a coefficient fitted on the rollout would depend on later rollout states. Auxiliary set permitted by [gyorok2026obc, after Lemma 3]. Controller does not enter.
- P4 topic: on the tuples the construction gives orthogonality and gradient orthogonality. Eq. obc_ours. Full column rank required, checked per rebuild; theta_aux differentiated through (D-190), the ingredient of [gyorok2026obc, Eq. (39)].
- P5 topic: the expansion point follows theta_base per epoch. [gyorok2025l4dc, Sec. 4] updates every evaluation or fixes at nominal; updating rebuilds the stack every evaluation; nominal does not apply from a detuned start (Section V). Phi and theta_bar carry no gradient (D-190, D-192).
- P6 topic: the corrected augmented model. Eq. obc_model. Correction at the rollout point with frozen coefficient as in [gyorok2026obc, Eq. (13)]; zero baseline rows leave g_aug uncorrected; prediction recomputes theta_aux once from the trained network (D-198).

## IV-C Scope and condition for true parameter recovery
Job: delimit the guarantee and state when uniqueness becomes true recovery.
- P1 topic: the guarantee is one-step, stacked over the tuples, local at theta_bar, in the normalised Euclidean metric.
- P2 topic: it does not cover the additional states or the output trajectory (zero slice, uncorrected g_aug, encoder, closed-loop multistep objective); [gyorok2026obc, Thms. 16, 18] do not transfer.
- P3 topic: uniqueness is not true recovery. Eq. obc_delta (D-202, D-203), Eq. obc_bias (state-level counterpart of [gyorok2026obc, Thm. 7, Eq. (21)], own). One-step surrogate of the closed-loop objective; the condition is aggregate and needs designed excitation [gyorok2026obc, Remark 5].
- P4 topic: one overlap measure checks the condition and the construction. Eq. obc_overlap (rho with Phi and with [Phi F_base], as in obc_diagnostics.py). Simulation only for Delta*. Section VI tests start independence and, where the condition holds, recovery.

## Moved or left out
- Hand-written forward sensitivity through RK4 (D-194): implementation detail, the model is the same; left out.
- Fourth-order stencil formula: textbook, now a phrase.
- Rank tolerance value, rank-loss policy (retain once, abort on two): Setup or appendix.
- Anti-alias filtering of positions before differencing (D-232): Setup.
- cos-angle with F_base: replaced by rho with [Phi F_base], which the code computes.
- Pointwise no-go and Coulomb argument: already in Section VII; not stated here (measured on superseded data).
- Planned geometric projection figure: not yet drawn; open point.
