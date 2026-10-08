# Outline: Section IV, Interpretability by Orthogonal Construction (run 14)

Pre-approved for this test run. Delivers contribution 3 of the Introduction: state-level
orthogonality for the identifiable physical-parameter tangent space and its true-parameter
recovery condition.

**Opening (one paragraph).** Section III-A left the split between f_base and f_aug non-unique;
this section makes the split unique by construction, extending Gyorok et al.'s OBC from
linear-in-parameters IO models to the self-scheduled gantry state transition; names IV-A, IV-B, IV-C.

## IV-A Negation of the physical parameters (sec:obc_negation)
Job: state the problem precisely, in the gantry's terms.
- P1. Topic: a learned write can cancel a parameter change to first order. Eq. obc_sens
  (transition sensitivity at the expansion point; adapted from Gyorok 2025 Eq. 15, state
  transition instead of their full-state model) and eq. obc_negation (first-order cancellation).
  Linear special case = Gyorok 2025 Ex. 1. Name: negation. Adapted.
- P2. Topic: on the gantry the directions are forces. Eq. obc_gantry_sens (differentiated EOM):
  mass via qdd, damping via qd, stiffness via q, all through M(Y)^-1, so Y-dependent. This work's.
- P3. Topic: protected coordinates are the ten combinations of Sec. II-C. Raw parameters would
  add four null columns (rank-deficient, violates Gyorok 2026 Assumption 1). Training coordinates
  differ by an invertible map, so the range is unchanged. Target: start-independent split; true
  values need IV-C. This work's.

## IV-B Orthogonal-by-construction transition (sec:obc_map)
Job: the construction and its gantry realisation.
- P1. Topic: Gyorok 2026 removes the overlap without a penalty weight. Eq. obc_gyorok (Eqs. 8, 9,
  11). Departure from research-plan Aspect 3 (penalty of Gyorok 2025) stated here. Adopted.
- P2. Topic: the regressor is the stacked transition sensitivity, tangent only. Rational in
  theta through M(Y)^-1, so Taylor regressor as Gyorok 2025 Sec. 4; offset column excluded
  (multiplies no parameter; D-186 reason 1, D-192 item 1); expansion point refreshed per epoch,
  between Gyorok 2025's two options, because the start is detuned (D-190, D-192). Adapted.
- P3. Topic: reference tuples from data. Eq. obc_tuples. Gyorok 2026 names missing full-state
  measurement as open; the FP-simulation fallback drifts on the free axes (Sec. III-C; D-111,
  D-185). Recorded input includes feedback; each tuple carries its measured Y, so the basis is
  scheduled. Fixed auxiliary set permitted (Gyorok 2026 p. 3). This work's.
- P4. Topic: stacked construction. Eq. obc_ours. Full column rank required and checked at each
  refresh (SVD, standard tolerance, rank-loss policy, code obc.py). Last line = Jacobian identity:
  every change of theta_aug moves the corrected stack orthogonally. Adapted.
- P5. Topic: corrected model. Eq. obc_model. Coefficient once per objective on the tuples, not on
  the rollout (causality, D-190/D-192); differentiated through; prediction as Gyorok 2026 Eq. 13;
  coefficient re-solved before validation and basis rebuilt at the selected checkpoint (D-198);
  hand-written forward sensitivity (D-194) to appendix; g_aug rows uncorrected. This work's.

## IV-C Scope and recovery condition (sec:obc_scope)
Job: what the guarantee covers, and when true values follow.
- P1. Topic: the guarantee is one-step, stacked, local, in the normalised metric, on xbar = 0.
  Not pointwise, not trajectory-output; added-state use unconstrained (D-190); Gyorok Thms. 16, 18
  do not transfer (objective is closed-loop multistep, Eq. aug_loss). This work's.
- P2. Topic: uniqueness is not truth. Eqs. obc_delta, obc_bias (state-level counterpart of
  Cond. 4, Thm. 7, Eq. 21; Delta* frame and stencil D-202, D-203). One-step surrogate. Adapted.
- P3. Topic: one measure decides the condition. Eq. obc_overlap (rho, D-190, D-201) with the
  bound on the parameter error (this work's). Needs theta*, simulation only. Cancellation over the
  stack, designed excitation (Gyorok 2026 Remark 5). Results tests start independence and
  conditional recovery.

## Moved or left out
- Test-data tangent share and output-level overlap (R4 metrics): Setup.
- The F_base cosine measure of the old draft: dropped (not computed by the pipeline).
- Pointwise no-go and friction violation of the condition: Discussion (already there).
- Rank, condition number, column correlations of the stack: Results (R4).
- Forward-sensitivity derivation, rank-deficient pseudoinverse: Appendix (OBC details).
- Implementation storage of u_act and P inside the block: left out (Sec. III fixes u).
- Affine arm: out of the campaign (RESULTS-DESIGN change 30), not mentioned.
- Planned geometric figure: not yet produced; stays in the header plan.
