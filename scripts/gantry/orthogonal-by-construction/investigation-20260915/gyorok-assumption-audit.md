# Assumption-by-assumption audit of Györök et al. (2026) against our candidate estimator

Source read directly for this audit: `literature/Orthogonality/gyorok2026_orthogonal-by-construction-augmentation_IFAC-JSC_arXiv2511.01321.pdf`.
The local copy is **v3, 9 January 2026** (the arXiv stamp on the first page reads
`arXiv:2511.01321v3 [eess.SY] 9 Jan 2026`), not the v2 named in the handoff. Every
equation, assumption, condition, lemma and theorem number below is from that v3 file and
was read on the page, not recalled. Where v2 and v3 could differ, the numbering used
here is the one that is on disk and reproducible.

Released implementation read alongside it:
`orthogonal-IO-augm-main/orthogonalx_augm/src.py`, class `IO_orthogonal_augmentation`.

Predecessor read for the nonlinear-parameter construction:
`literature/Orthogonality/Hoekstra - Orthogonal projection-based regularization for efficient model.pdf`
(arXiv:2501.05842), which is a **regularizer** with a trade-off weight, not the exact
reparameterization; the two must not be conflated.

## Source corrections established here

**C1. Missing transpose in the prose after Eq. (8).** The displayed Eq. (8) is
`theta_aux = (Phi^T Phi)^{-1} Phi^T F^ANN`, which is correct. The sentence immediately
below it reads "In fact, `(Phi^T Phi)^{-1} Phi` corresponds to the Moore-Penrose pseudo
inverse of `Phi`". The transpose is missing on the trailing factor. The correct identity
is `(Phi^T Phi)^{+} Phi^T = Phi^{+}`, which holds at any rank, not only at full rank.
The released code uses the correct form
(`src.py:248`: `K_phi = jnp.linalg.pinv(Phi_train.T @ Phi_train) @ Phi_train.T`).
This investigation uses the displayed equation.

**C2. Eq. (40) is displayed per data point but only holds stacked.** The proof of
Theorem 18 writes, "Then, for all data points in `D_N`",

    psi_k^T psi_k = [[ phi^T(x_k) phi(x_k), 0 ], [ 0, J_f^T(x_k) J_f(x_k) ]],

that is, it asserts `phi^T(x_k) J_f(x_k) = 0` **at each k**. Lemma 3 gives only the
stacked identity `Phi^T F̃ = 0`, which is a statement about the sum over k. Measured
directly on our prototype (`results/s3_projection_prototype.log`, T4):

| quantity | value |
|---|---|
| per-sample `‖phi_k^T J_f,k‖`, min / median / max | 3.64e-04 / 1.88e-03 / 4.51e-03 |
| stacked `‖sum_k phi_k^T J_f,k‖` | 1.29e-13 (2.02e-13 on the rerun) |

The per-sample product is not small; only the sum vanishes, to machine precision. The
**conclusion** of Theorem 18 is unaffected, because Eq. (38) contains only
`sum_k psi_k^T psi_k`, and that sum is block diagonal. The per-sample display in Eq. (40)
is nevertheless incorrect as written. This matters for us directly: any claim we make
about block structure must be stated over the stacked reference set, never pointwise.

**C3. Theorem 12 assumes the input is independent of the noise.** Its statement reads
"with a quasi-stationary input `u` independent of the white noise process `e`". This is
an open-loop condition. It is not listed among the numbered Conditions, so it is easy to
miss when auditing Conditions 10, 11 and 14 alone. Our data are closed-loop logs, so
this hypothesis is the one that decides whether the consistency chain
(Theorem 12 -> Lemma 15 -> Theorem 16) is available to us at all.

## Status legend

- **satisfied**: holds in our setting, with the evidence named.
- **structurally violated**: cannot hold given the model class or the data; the
  conclusion that depends on it is not available.
- **conditional**: holds only under a stated restriction we can choose to impose.
- **unknown**: not established by this investigation; the test that would settle it is
  named.
- **not invoked**: we do not use the result that depends on it.

## The matrix

| # | Source hypothesis | Anchor | Our setting | Status |
|---|---|---|---|---|
| 1 | Data-generating system is a DT input-output NARX process `y_k = f(x_k) + e_k` with `x_k` a lagged measured IO vector | Eq. (1) | Ours is a state-space LPV-LFR model with 6 unmeasured physical states plus `n_a` learned states; the outputs are 3 of the 6 positions through `C = [P^T 0]` | **structurally violated**. This is the extension the paper's own conclusion names as open, citing the absence of full-state measurement. Our condition is imposed on the state equation instead (`decisions.md` OBC-1), so the object made orthogonal is not the object the paper makes orthogonal. |
| 2 | Baseline is linear in its parameters, `ŷ_k = phi(x_k) theta_b` | Eq. (2) | `qddot = M(Y)^{-1}(Pu - C qdot - K q)`, and `det M(Y)` was shown symbolically to depend on `m1, m2, mb, mh, Jb, Jh, Lb` (`results/s12_parameterization_and_rank.log`, S1a). Every velocity row is a ratio whose denominator carries the unknowns | **structurally violated** for the explicit map. Weakens to orthogonality against the affine tangent set `[Phi_p , c]` at a declared expansion point. Theorem 7 does not transfer as an exact statement. An exactly linear form exists only for the *implicit* force balance (S1b), at the price of a different estimand. |
| 3 | Assumption 1: `rank(Phi) = n_theta`, Eq. (7) | Eq. (7) | Rank 10 of 14 parameter columns, 11 of 15 with the offset column, at every rank tolerance from 1e-4 to 1e-14, at three admissible parameter sets and two reference grids (S2). The gap factor between `sigma_9` and `sigma_10` is 7.9e12 | **structurally violated**, and provably so, not numerically. Cost is bounded: by Prop. 3.1 of `state-level-obc-derivation.tex` orthogonality holds at any rank. Measured: the coefficient moved by 3.70 along a null direction while the projected write moved by 1.19e-15 (S3 T2). What is lost is the uniqueness argument of Theorem 7 / Eq. (21), not the orthogonality. |
| 4 | Lemma 3 / Eq. (11): `Phi^T F̃ = 0` on the training set | Eqs. (10)-(12) | Reproduced exactly on the real RK4 transition: residual ratio 1.96e-14 on the reference set (S3 T1) | **satisfied**, on the reference set only. Off it the residual is not controlled; see row 12. |
| 5 | Condition 4 / Eq. (17): `sum_i phi^T(x_i) delta(x_i) = 0`, the true unmodeled term is orthogonal to the regressor on the data | Eq. (17), Eq. (16) | `delta` is the true model error. Checked here on two synthetic residuals: a friction law gives `rho(Phi; delta) = 0.9465`, a gross violation; the same residual projected out of the span gives `2.7e-15`, satisfied by construction (`results/s3e_condition4_control.log`). On real Telica data `delta` is unknown by definition | **conditional, and it is the binding hypothesis.** Checkable in simulation, structurally unverifiable on real data. With everything else held fixed the sign of the method's effect on parameter recovery FLIPS across it: mean combination error 0.4175 when violated against 0.0206 when satisfied, against 0.1122 for no training at all. Theorem 7 cannot be invoked as an exact-recovery claim on real data; what survives is the error expression Eq. (21), which this experiment makes visible, plus the comparative statement under Assumption 8 (Eq. 22). See `decision-report.md`, the opening section. |
| 6 | Assumption 6 / Eq. (18): the identified model recovers the data-generating dynamics on `D_N` | Eq. (18) | Only realistic for finite `N` with noiseless data, as the paper itself says immediately after stating it. Our training data are noiseless simulation for the augmentation experiments and noisy closed-loop logs for Telica | **conditional**: available in the noiseless synthetic phase, not on real data. |
| 7 | Theorem 7: exact recovery `theta_b_hat = theta_b*` | Theorem 7, Eqs. (19)-(21) | Depends on rows 2, 3, 5, 6, all of which fail or are unknown | **not available**. Do not claim exact parameter recovery from the construction. |
| 8 | Assumption 8 / Eq. (22): `‖Phi theta_b*‖ > ‖Delta‖`, the baseline carries the dominant dynamics | Eq. (22) | Plausible for our baseline, which reproduces the gantry to a free-run floor well below the residual it is meant to learn, but this has not been measured inside this investigation | **unknown**, cheaply measurable: compare the stacked baseline write to the stacked truth-minus-baseline write on the training records. |
| 9 | Condition 10 / Eq. (26): the data-generating system is exponentially forgetting | Eq. (26) | The gantry has two rigid-body integrator axes, `K[0,0] = K[2,2] = 0` in `gantry_ss.py:98-100`, so the open-loop plant has poles at `z = 1` and does not forget its initial condition | **structurally violated in open loop**, and it is a *physical* property that must not be altered to make a theorem apply. Under the closed loop the records are generated in (`gtd_run_simulation.m`), the *closed-loop* system is stable, so the condition can hold for the data-generating system as operated even though it fails for the isolated plant. Which object the condition applies to must be stated explicitly in the thesis. |
| 10 | Condition 11 / Eq. (28): stable predictor, uniform exponential forgetting of the predictor | Eq. (28) | Our predictor is a free-run rollout of the augmented model with an encoder initialization and multiple shooting, on a plant with poles at `z = 1`. The augmented rollout is marginally stable on those two axes | **structurally violated**. Remark 13 points at stable-by-design parameterizations as the remedy and calls the extension to augmentation an open question. Note the scope: this kills the *asymptotic* results (Theorem 12, 16), not the finite-horizon projection identities of rows 4 and 11, which never referred to a limit. |
| 11 | Theorem 18 / Eq. (40): block-diagonal `psi^T psi`, zero covariance between `theta_b_hat` and `theta_a_hat` | Theorem 18, Eqs. (37)-(40) | The block structure itself is reproduced exactly with a frozen projector, because the proof only needs the projector to be independent of `theta_a` (Remark 2.2 of the derivation). Measured: `‖Phi^T dG̃/d eta‖` falls from 4.49e-02 to 1.29e-13 when the coefficient is differentiated through, and stays at 4.49e-02 when it is detached (S3 T3) | **satisfied for the block structure, over the stacked reference set**, subject to correction C2. The covariance *conclusion* additionally needs `Sigma_e` diagonal and both blocks full rank; the second fails here by row 3, and noiseless simulation has no `Sigma_e` at all. So: block structure yes, covariance statement **not invoked**. |
| 12 | The projector transfers to prediction data, Eq. (13) | Eq. (13) | The released code recomputes `Phi_test` at the test points but keeps the frozen `theta_aux` (`src.py:364-369`). Measured off our reference set the orthogonality residual does **not** improve, and on the raw metric it degrades by ~5x (S3 T1, S3b) | **structurally violated as a transfer claim** for a learned write that is not itself in the baseline span. Control E of S3b shows that a write which *is* in the span is removed at any point set, so the gap is a generalization failure of the fitted coefficient, not a defect of the projector. |
| 13 | Static learning component (no learned states, no recursion) | Sect. 2, Eq. (3) | Ours carries `n_a` additional states with no baseline equation, and the write into them is unconstrained by the condition (Prop. 4.2 of the derivation, verified on the real gantry equations) | **structurally violated**. Consequence recorded in OBC-1: one-step non-overlap does not transfer to the accumulated output. Do not word any claim as horizon-level non-overlap. |
| 14 | No encoder; the regressor argument is measured | Sect. 2 | Our initial state comes from a SUBNET encoder with parameters `rho` that appear in no row the condition touches | **not covered**. A contribution injected purely through `x(0)` is invisible to the condition. This is a coverage gap, not a violation of a stated hypothesis, because the paper has no encoder to make a hypothesis about. |
| 15 | Condition 14 / Eq. (30): persistency of excitation, distinguishability of non-equivalent models | Eq. (30) | With 14 raw parameters and rank 10, non-equivalent raw parameter vectors are *not* distinguishable: exact invariance was measured, `max|dx_next| = 0.00e+00` along every null direction at displacements up to 1e-1 (S2). In the reduced 10-combination coordinates the one-step map has full rank 10 with condition number 1.33e4 | **structurally violated in raw coordinates, satisfiable in reduced coordinates.** This is the single strongest argument for reporting and, if possible, estimating in the 10 combinations. |
| 16 | Theorem 12 / Eq. (29) and Theorem 16 / Eqs. (34)-(36): convergence and consistency | Theorems 12, 16 | Depend on rows 9, 10, 15 and on the unnumbered "input independent of the noise" hypothesis (correction C3), which closed-loop data violate | **not available**. No consistency claim may be made for our estimator from this source, and none may be inferred from numerical experiments. |
| 17 | One-step prediction cost, Eq. (4) | Eq. (4) | Ours is free-run simulation error with multiple shooting | **not invoked**. The orthogonality condition never referred to the cost, so nothing in rows 3, 4, 11 depends on this. It does remove comparability with the paper's numerical results. |

## What the audit leaves standing

Three statements survive every row above, and they are the honest content of the method
for our setting:

1. **Exact non-overlap of the one-step physical write with the affine baseline tangent
   set, over a frozen reference point set, at any rank.** Rows 3, 4. Measured 1.96e-14.
2. **Exact block structure of the joint parameter Jacobian over that same reference set,
   provided the auxiliary coefficient is differentiated through.** Row 11. Measured
   1.29e-13 against 4.49e-02 for the detached control.
3. **Elimination of the flat direction of the source's Example 2 at the level of the
   one-step augmented map**, which is the identifiability pathology the method exists to
   remove.

Everything asymptotic, everything about recovering the true physical parameters, and
everything about the accumulated output over a horizon is outside what this source
supports in our setting.

## The row that matters most

Of the seventeen rows, row 5 is the one that changes what the method DOES rather than
what may be said about it. Rows 1, 2, 3, 13 and 14 are structural mismatches that the
existing derivation already handles by weakening the claim; rows 9, 10 and 16 remove the
asymptotic results, which were never going to be available for a marginally stable plant
under free-run training. Row 5 is different: it is a property of the true residual and of
the data, it is checkable in simulation, and the measured effect of violating it is that
the construction makes physical parameter recovery four times worse than not training at
all. No amount of care in the implementation compensates for it, because the behaviour is
exactly what Eq. (21) says it should be.
