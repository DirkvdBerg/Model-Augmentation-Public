# Outline: Section IV, Interpretability by Orthogonal Construction (pre-approved, test run 10)

Opening (one paragraph): Section III-A left the split between f_base and f_aug non-unique; this section makes it unique by adapting Gyorok's orthogonal-by-construction (OBC) augmentation from input-output models linear in their parameters to the self-scheduled gantry state equation. Names IV-A (negation), IV-B (construction), IV-C (scope and recovery condition).

## IV-A Negation of the Physical Parameters
Job: define the non-unique direction that OBC must remove, in gantry terms.
- P1. Joint estimation lets f_aug write into the same physical rows as f_base, so different parameter pairs give the same one-step map. Eq: transition sensitivity Phi (obc_sens). Sources: gyorok2026obc Ex. 2, gyorok2025l4dc Ex. 1.
- P2. A learned write cancels a parameter displacement to first order; the data cannot attribute it; this is negation; the aim is a split whose physical estimate does not depend on the initial pair. Eq: obc_negation.
- P3. For the gantry the directions are added mass, damping and stiffness forces, scheduled by M(Y)^{-1}, so the protected space depends on payload position. Eq: obc_gantry_sens (own derivation from eq:eom).
- P4. The ten combinations of Section II-C are protected, not the fourteen raw parameters (four structural null directions are not negation). The log/bounded training coordinates leave range(Phi) unchanged (invertible coordinate Jacobian; own one-line argument).
- P5. The parameter regulariser of hoekstra2026lfr (Sec. 5.2, Eq. 26) does not remove the direction; it settles it at the a priori values, which are detuned here (D-193). No equation.

## IV-B Orthogonal-by-Construction State Update
Job: give the construction, its regressor, its evaluation set, its gradient treatment and its training lifecycle.
- P1. Problem then remedy: the projection penalty of gyorok2025l4dc (Eq. 13), proposed in the research plan, needs a weight that trades fit against orthogonality; OBC (gyorok2026obc) removes it. Eq: obc_gyorok (Eqs. 8, 10, 11, 13). Lemma 3 as consequence.
- P2. Adaptation: apply it to the physical rows with the tangent Phi as regressor (f_base rational in theta_base, gyorok2025l4dc Eq. 15); the tangent defines only the protected space, the rollout keeps the nonlinear f_base. Offset column Gamma (gyorok2025l4dc Eqs. 17, 18) excluded: by (obc_negation) a negating write lies in range(Phi); Gamma multiplies no parameter (D-192).
- P3. Reference tuples from measured positions, because the open-loop baseline drifts on the free axes (Section III-C) and gyorok2025l4dc's simulated-state fallback does not apply. Eq: obc_tuples (stencil itself as a phrase, textbook). Measured Y_i per tuple gives the scheduling dependence; recorded input contains the feedback action; fixed auxiliary set permitted by gyorok2026obc Sec. 3.2.
- P4. Stacked construction. Eq: obc_ours (stacks, coefficient, Lemma 3 identity, gradient identity). Full column rank (counterpart of Assumption 1) checked at every rebuild, values in Results; the gradient line holds because Phi is fixed in theta_aug and the coefficient is differentiated through.
- P5. Lifecycle: expansion point options of gyorok2025l4dc Sec. 4 do not fit (cost, detuned start); Phi rebuilt per epoch at current theta_base and held fixed; coefficient once per objective from the fixed tuples, not from the rollout (causality) (D-190, D-192).
- P6. Corrected model. Eq: obc_model with the closed-loop input u-hat. Correction at the epoch's expansion point and the objective's coefficient as in gyorok2026obc Eq. 13; g_aug uncorrected; after training Phi and theta_aux rebuilt once at the selected parameters and fixed (D-198).

## IV-C Scope and Condition for True Parameter Recovery
Job: delimit the guarantee and state when it yields the true parameters.
- P1. Guarantee is the identity of (obc_ours) only: local, one-step, aggregate, normalised metric, slice x-bar = 0; added states uncorrected and re-enter one step later; hence no trajectory-output orthogonality and no transfer of gyorok2026obc Thms. 16, 18. No equation.
- P2. Unique split is not true split. Eq: obc_delta (true one-step discrepancy, same stencil, normalised; D-202, D-203). Assumptions (noise-free, affine between theta* and theta-hat, exact one-step recovery, counterpart of Assumption 6). Eq: obc_bias, state-level counterpart of Cond. 4 and Eq. 21 (own derivation). One-step surrogate, not the bias of the closed-loop multistep fit.
- P3. The condition is aggregate, so it needs cancellation over the stack and in general designed excitation (Remark 5). Eq: obc_overlap (rho* only), zero iff the condition holds, simulation only. Pointer to Results.

## Moved or left out
- Stencil formula: textbook machinery, now a phrase.
- Second overlap measure (cosine with F_base) and the held-out-tuple use of rho: evaluation metrics, to Setup/Results.
- Hand-written RK4 forward sensitivity (D-194), SVD solve, rank-loss policy, validation coefficient synchronisation mechanics, D-number references in prose: implementation detail, left out (reproducibility appendix candidate).
- Affine comparison arm: removed from the campaign; only the exclusion and its reason stay.
- Pointwise no-go result and Coulomb inner-product argument: already in Section VII (Limitations).
- Planned geometric figure: not drawn in this run (open point).
