# Outline: Section IV, Interpretability by Orthogonal Construction (run 8)

Pre-approved for this test run. Symbols kept from the existing draft and Sections II and III
($\theta_{\mathrm{base}}$, $\theta_{\mathrm{aug}}$, $\tilde x$, $\bar x$, $\hat x$, $f_{\mathrm{aug}}$,
$g_{\mathrm{aug}}$, $\Phi_{\bar\theta}$, $\theta_{\mathrm{aux}}$, $\Delta^*$).

**Opening (one paragraph).** Joint estimation (Section III) leaves the split between baseline and
network non-unique; this section adapts Gyorok's orthogonal-by-construction (OBC) parametrisation
to the self-scheduled state-space gantry model, replacing the projection regularisation of the
research plan, in three steps: negation (A), the construction (B), its scope and the recovery
condition (C).

## A. Negation of the baseline (`sec:obc_negation`)
Job: show what goes wrong in joint estimation and which directions must be protected.

1. Topic: the additive write $f_{\mathrm{aug}}$ can realise every first-order change of
   $\theta_{\mathrm{base}}$. Equations: sensitivity $\Phi_{\bar\theta}$ (eq:obc_sens) and the
   first-order cancellation (eq:obc_negation). Sources: G26 Ex. 2, L4DC Ex. 1.
2. Topic: the data cannot attribute these directions, so the estimate depends on the start; a
   penalty on the deviation from the initial values bounds the drift but ties the estimate to
   uncertain initial values. Sources: G26 Sec. 1 and Sec. 5.2, D-193.
3. Topic: for the gantry the protected directions are generalised forces of mass, damping and
   stiffness type, scheduled by $Y$. Equation: acceleration sensitivity (eq:obc_gantry_sens).
   Source: this work's derivation from (eq:eom).
4. Topic: OBC protects the ten identifiable combinations of Section II-C, because the raw stack
   has four null columns; the training coordinates give the same range. Sources: D-190, D-229.

## B. Orthogonal-by-construction transition (`sec:obc_map`)
Job: define the implemented construction and every choice an examiner would question.

1. Topic: Gyorok's IO construction removes the regressor fit from the network output without a
   weight, unlike the L4DC penalty proposed in the research plan. Equation: eq:obc_gyorok. Sources:
   G26 Eqs. (8), (10), (11), (13), Lemma 3; L4DC Eq. (13).
2. Topic: three properties of the gantry model do not match the source (nonlinear in parameters,
   only positions measured, additional states); the sensitivity replaces the regressor (L4DC
   Eq. (15)) and the affine offset (L4DC Eq. (18)) is not protected, with its reason. Sources:
   D-192 item 1, implementation plan.
3. Topic: evaluation points are fixed tuples from the training records. Equation: eq:obc_tuples.
   Reasons: drift of a baseline replay on free axes (Section III-C), fourth-order stencil because
   damping directions multiply velocity, sensitivity at each tuple's own $Y$, recorded input
   carries the feedback. Sources: D-192 item 2, D-203, G26 Sec. 3.2.
4. Topic: the stacked construction on the tuples. Equation: eq:obc_ours. Rank requirement and
   tolerance, why stacked and not per tuple, coefficient differentiated through, basis detached.
   Sources: D-190, D-192 items 3 and 4, G26 Lemma 3.
5. Topic: lifecycle: coefficient once per objective, basis refreshed per epoch, prediction with
   the current network's coefficient. Sources: D-190, D-192, D-198, L4DC Sec. 4.
6. Topic: the corrected augmented model. Equation: eq:obc_model. Correction at the rollout point
   with the frozen expansion point (G26 Eq. (13)), exact forward sensitivity through RK4 (D-194),
   additional-state rows uncorrected.

## C. Scope and condition for true parameter recovery (`sec:obc_scope`)
Job: delimit what is guaranteed and state when the true parameters are recovered.

1. Topic: the guarantee is one stacked identity: local, one-step, in the normalised metric, on the
   $\bar x=0$ slice, not pointwise and not at trajectory output level; G26 Thms. 16, 18 do not
   transfer. Sources: D-190, D-192 constraints, D-202.
2. Topic: non-overlap does not imply the true split. Equations: eq:obc_delta, eq:obc_bias
   (state-level counterpart of G26 Cond. 4, Eq. (21), under its Assumption 6). Consequence: a
   one-step surrogate of the closed-loop objective; aggregate cancellation needs designed
   excitation (G26 Remark 5).
3. Topic: overlap measures to check the condition and the unprotected offset. Equation:
   eq:obc_overlap with $B\in\{\Phi,[\Phi\ F_{\mathrm{base}}]\}$. Sources: D-201, D-202, D-203,
   `obc_diagnostics.rho_table`. Pointer to Results.

## Moved or left out
- Numbers (rank, condition, tolerance value, refresh timing, filter cutoff of D-232) to Setup.
- Forward-sensitivity derivation, stacked projector and rank-deficient case to the appendix.
- Implementation detail of storing $u_{\mathrm{act}}$ and applying $P$ inside the block: left out
  (reader test).
- Pointwise no-go measurement and Coulomb argument stay in Discussion (as already noted).
- Affine comparison arm: left out (header: no affine comparison arm).
- Figure: planned geometric figure kept as a LaTeX comment, listed as open point.
