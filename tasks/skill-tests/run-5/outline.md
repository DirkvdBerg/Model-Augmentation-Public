# Outline: Section IV, Interpretability by Orthogonal Construction (pre-approved)

Roadmap: negation (IV-A), protected space and reference tuples (IV-B), corrected model and its training (IV-C), what is guaranteed and when the true combinations are recovered (IV-D).

## Roadmap paragraph
- Topic: Section III leaves the split between baseline and network non-unique; this section removes the first-order part of that freedom by adapting Gyorok's orthogonal-by-construction (OBC) parametrisation to the gantry state transition.
- Content: OBC exists for discrete-time IO models linear in their parameters; its authors name state-space baselines without full-state measurement as open; the four subsections in order.
- Sources: gyorok2026obc (Sec. 6, conclusion).

## IV-A Negation in joint estimation (sec:obc_negation)
Job: define negation and the parameter directions it acts along.
1. Topic: joint estimation lets the additive network reproduce a change of theta_base. Eq. obc_sens (one-step sensitivity Phi at expansion point), eq. obc_negation (first-order cancellation). Sources: gyorok2026obc Ex. 2, gyorok2025l4dc Ex. 1 and Eq. (15).
2. Topic: on the gantry these directions are added mass, damping and stiffness forces that depend on Y. Eq. obc_gantry_sens (acceleration sensitivity from eq:eom). Source: own derivation from Section II.
3. Topic: OBC protects the ten identifiable combinations, not the fourteen raw parameters. No equation. Sources: Section II-C (four null directions); code (`Reduced_Gantry_State_Block`, training coordinates with nonsingular map, so same range). Ends on the goal: a start-independent split; true values need IV-D.

## IV-B Protected space on reference tuples (sec:obc_space)
Job: state the source construction, what replaces its regressor, and where it is evaluated.
1. Topic: Gyorok et al. remove the overlap by construction rather than by a penalty. Eq. obc_gyorok (theta_aux, corrected stack, orthogonality). States the departure from the research plan's projection penalty (weight trade-off). Sources: gyorok2026obc Eqs. (8), (10), Lemma 3; gyorok2025l4dc Eq. (13).
2. Topic: on the gantry the regressor is the one-step sensitivity and only the physical rows are protected. No new equation (uses obc_sens). Reasons: rational in theta through M(Y)^{-1} (D-190); the offset of the Taylor form is excluded because it multiplies no estimated parameter, so negation (obc_negation) cannot act along it, unlike gyorok2025l4dc Eq. (18) (D-192, RESULTS-DESIGN change 30).
3. Topic: the stack is built on fixed tuples formed from the measured positions. Eq. obc_tuples. Reasons: only positions measured; open-loop baseline simulation (gyorok2025l4dc's fallback) drifts on the free axes X and Y (Section III-C); an auxiliary evaluation set is permitted (gyorok2026obc after Lemma 3). Code: `build_reference_set`.
4. Topic: each tuple carries its own scheduling value and recorded input. Consequences: Y dependence enters the stack; the controller does not enter; x_bar = 0 because added states are unmeasured; tuples fixed for the run and normalised. Code: `obc_gantry.py`.

## IV-C Corrected augmented model (sec:obc_map)
Job: the stacked construction, its training lifecycle, and the model it yields.
1. Topic: on the tuples the construction gives orthogonality of the network write and of its gradient. Eq. obc_ours. Full column rank required, checked at every rebuild. Sources: gyorok2026obc Lemma 3; code (`OBCBasis`, SVD solve).
2. Topic: the coefficient is fitted per objective evaluation and the basis is frozen per epoch. Reasons: fitting on the rollout would make the correction depend on later states (causality); range turns slowly with the expansion point (D-190); consequence: no gradient reaches theta_base through the basis. Sources: D-190, D-192.
3. Topic: the corrected model subtracts the fitted parameter change at each rollout point. Eq. obc_model. Prediction uses the coefficient of the final network, as gyorok2026obc Eq. (13) (D-198). Zero baseline rows leave g_aug uncorrected.

## IV-D Scope and condition for true parameter recovery (sec:obc_scope)
Job: delimit the guarantee and state when non-overlap gives the true combinations.
1. Topic: the guarantee is local, one-step and aggregate over the tuples, in the metric fixed by the state normalisation. Sources: D-190, D-192 Constrains.
2. Topic: the additional states fall outside the guarantee. No trajectory-output orthogonality; consistency and zero-covariance theorems do not transfer. Sources: gyorok2026obc Thms. 16, 18; D-190.
3. Topic: non-overlap gives a unique split, not the true one. Eq. obc_delta (true one-step discrepancy, same stencil, normalised), eq. obc_bias (derivation and parameter error). Sources: gyorok2026obc Cond. 4, Thm. 7, Eq. (21); D-202, D-203.
4. Topic: on the gantry the condition can only hold through cancellation over the record. Source: gyorok2026obc Remark 5; Discussion bullet (pointwise rank).
5. Topic: two overlap measures check the condition and the unprotected overlap. Eq. obc_overlap. Sources: RESULTS-DESIGN R4, R5; obc_diagnostics.py (rho).

## Moved or left out
- To Setup: number of tuples and records, rank tolerance value, refresh setting, anti-alias filtering before differentiation (D-232), arm list (Aug-U, Aug-OBC).
- Left out (reader test): how the code stores the actuator force and applies P; the hand-written forward sensitivity through RK4 (implementation of the same directional derivative); D-number citations in prose.
- Left out (decided): the affine comparison arm (removed from the campaign); the soft-penalty route.
- Planned figure: no asset exists and this run may not create one; listed as open point.
