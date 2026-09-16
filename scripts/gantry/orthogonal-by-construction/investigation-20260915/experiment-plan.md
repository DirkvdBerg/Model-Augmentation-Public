# Experiment plan: orthogonal-by-construction on the nonlinear self-scheduled gantry

Written before any experiment in this folder was run. Each stage records its hypothesis,
the alternatives it must separate, and the acceptance criterion, followed by the outcome
once the stage has run. Failures stay in the record.

Environment: `conda run -n GraduationProject python`, torch 2.5.1, numpy 2.0.1,
scipy 1.17.0, sympy 1.14.0. MATLAB R2025a and R2021b are installed.
All Python numerical work is float64 unless a stage says otherwise.

## Governing model as read from source (not assumed)

`model_augmentation/systems/gantry_ss.py` and
`model_augmentation/fit_systems/blocks.py` (`Gantry_State_Block`,
`Parameterized_Gantry_State_Block`) define, in logical coordinates
`q = (q1, q2, q3)`, `x = (q, qdot)`, stage input `u`:

    M(Y) qddot + C qdot + K q = P u,      Y = q3 = x[2]
    M(Y) = M0 + M1 Y + M2 Y^2
    y     = [P^T  0] x

with `M0, M1, M2, C, K, P` as listed in `gantry_ss.py` lines 70-128. The discrete
transition is RK4 with `up_sample = 10` substeps of `Ts/up_sample`, self-scheduled
through `Y = x[2]` re-evaluated at every substep and at every RK4 stage
(`blocks.py:778-784`, `blocks.py:825`). `Lb` is frozen; the 14 trained raw scalars are
`kb1, kb2, cg1, cg2, cy, cb1, cb2, mh, m1, m2, mb, Jb, Jh, d`, trained in log
coordinates `theta = theta_init * exp(l)`.

## Stage 1 - baseline parameterization and scheduling

**H1a** There is no exact representation of the explicit discrete transition
`x(k+1) = f_theta(x(k), u(k))` that is linear in a finite set of unknown parameter
combinations, because `M(Y)^{-1}` makes every velocity row a ratio of polynomials in
theta whose denominator depends on theta.

**H1b** The *implicit* force balance
`r = M(Y) qddot + C qdot + K q - P u` is exactly linear in a finite set of lumped
coefficients, and those coefficients are the entries of `M0, M1, M2, C, K` subject to
algebraic admissibility constraints that tie shared scalars (`mh` appears in `M0[2,2]`,
`M1[0,1]`, `M1[1,0]`, `M2[1,1]` and through `mh*d`, `mh*d^2`).

**H1c** The three scheduling conventions (measured/exogenous `Y`, simulated endogenous
`Y`, frozen `Y`) give materially different transitions over the operating range, so a
frozen-scheduling basis is a diagnostic only.

Acceptance: H1a is accepted only if a symbolic check shows the denominator
`d(Y, theta)` depends on theta (then no finite linear-in-theta form of the explicit map
can exist without also fixing the denominator). H1b is accepted if the residual built
from lumped coefficients reproduces the symbolic force balance identically. H1c is
accepted if the transition difference between frozen and endogenous scheduling exceeds
the float64 round-off floor by orders of magnitude on a designed excitation inside the
documented Y range, and is rejected if it does not.

## Stage 2 - identifiable parameter combinations

**H2** The map `theta (14) -> (M0, M1, M2, C, K)` factors exactly through 10 quantities
`{kb_sum, cg1, cg2, cy, cb_sum, mh, m_total, m_diff, J_eff, d}` as coded in
`Parameterized_Gantry_State_Block._combos_from_raw`, so the one-step parameter Jacobian
has rank 10 and nullity 4, and the four null directions are
`kb1-kb2`, `cb1-cb2`, and the two-dimensional family
`(dm1, dm2, dmb, dJb, dJh) = (-1/2, -1/2, 1, Lb^2/4, 0)` and `(-1/2, -1/2, 1, 0, Lb^2/4)`
whose difference is `Jb-Jh`.

**H2-alt** The prior MATLAB Euler check (`code/logs/gantry_euler_regressor_structure.log`)
and the derivation's closing remark disagree with the project's documented set; the
remark claims the project lists "the inertia sum on its own". This must be checked
against the actual code rather than the prose.

Tests, all on the real RK4 transition, not on the Euler surrogate:
1. Analytic autodiff Jacobian `d x(k+1) / d theta` stacked over designed points; SVD
   spectrum; rank at a relative threshold swept over three decades.
2. Independent central differences with a step sweep, compared to the autodiff Jacobian.
3. Physical, normalized and log parameter coordinates, to show rank is invariant and
   conditioning is not.
4. Exact null-vector identification: project the numerical null space onto the predicted
   four directions and report the principal angles.
5. A direct invariance test: construct raw parameter vectors differing by a null
   direction, then check equality of `M0, M1, M2, C, K` and of the full RK4 transition
   at independent points.
6. Measured-output sensitivity over a trajectory with known initialization, stacked, to
   separate the structural one-step rank from what the output actually sees.
7. Small noiseless multi-start recovery on reduced (10) versus raw (14) coordinates,
   three seeds, reporting recovered combinations and the spread of raw splits.

Acceptance: H2 accepted if the autodiff and finite-difference null spaces agree to a
principal angle below 1e-6 rad at three admissible parameter sets and both match the
predicted directions. Rejected if any extra null direction appears at RK4 fidelity that
was absent at Euler fidelity.

## Stage 3 - protected basis and linearization/reference policy

**H3a** Because Stage 1 rejects an exact linear-in-parameter form for the explicit map,
the G26 exact parameter-span projection is not available exactly; the available object
is the affine tangent set `[Phi_p, c]` at a declared expansion point, which is what the
derivation already adopts.

**H3b** The subtraction `G - Phi K G` with `K` a fixed Moore-Penrose solve of the normal
equations is an exact projector on the reference set at any rank, and the derivative
identity `d/d eta (G - Phi eta_aux) = (I - P) dG/d eta` holds only if the coefficient is
differentiated through. A detached-coefficient variant must be shown to break it.

**H3c** The protected span depends on the expansion point and on the scheduling coverage
of the reference set; the approximation error for a parameter displacement `delta` grows
like `||delta||^2`.

Tests: an isolated prototype implementing the subtraction with a declared SVD
convention; orthogonality residual on and off the reference set; rank-deficient
coefficient equivalence (two different normal-equation solutions give the same
projected write); derivative identity by autodiff versus a detached-gradient control;
second-order error sweep in `||delta||`; principal angles between protected spans at
different expansion points and different Y coverages; and a small additive competition
example comparing no projection, the 2025 penalty and the 2026 subtraction from the same
initialization.

Acceptance: exactness to 1e-12 relative on the reference set; the detached control must
produce a nonzero `Phi^T d(G-tilde)/d eta`; the displacement error must show a clean
slope 2 on a log-log fit over at least two decades. Anything else is reported as is.

## Stage 4 - additional states and encoder

**H4** With a baseline-only reference set the fitted coefficient is provably blind to the
latent path (derivation Prop. 7.1); with a lagged augmented rollout (R2) or a designed
latent set (R1) it is not. The sensitivity of the coefficient to latent-path parameters
is the discriminator.

Tests: zero versus representative latent reference coverage; sensitivity of
`eta_aux` to latent-path weights under each reference policy; fixed versus oracle versus
learnable initial state; and a small joint recovery with and without encoder freedom
under identical excitation.

Acceptance: a nonzero coefficient sensitivity under R1/R2 and an exactly zero one under
the baseline-only set reproduces Prop. 7.1 on the real block rather than on the
surrogate. Encoder confounding is reported as measured rank/conditioning, never as a
universal claim.

## Stage 5 - theorem audit, marginal stability, closed loop

**H5** The G26 assumptions split into: satisfied, structurally violated, conditional,
unknown, not invoked. The marginal modes are the two rigid-body integrator axes
(`K[0,0] = K[2,2] = 0`), which are physical and must not be altered to invoke a theorem.

Tests: an assumption-by-assumption table with source equation numbers; a finite
perturbation study of the marginal modes; and a small correctly specified closed-loop
synthetic plant with the same estimator, open loop versus closed loop, noiseless first.

Acceptance: every row carries a status and an anchor. No consistency claim is made from
numerical tests.

---

## Outcomes

| hypothesis | verdict | where |
|---|---|---|
| H1a no exact linear-in-parameters explicit map | **accepted**, symbolically | `results/s12_...log` S1a |
| H1b implicit force balance exactly affine in 11 lumped coefficients | **accepted**, with the redundancy `MD^2 = MH MD2` named | `s12` S1b |
| H1c the three scheduling conventions are materially different | **accepted**, 9 to 13 percent of the state increment against a round-off floor of exactly zero | `s12` S1c |
| H2 rank 10, nullity 4, the four predicted null directions | **accepted**, max principal angle 2.58e-08 rad and below 4e-08 rad across three parameter sets and two grids, stable across tolerances | `s12` S2 |
| H2-alt the derivation's flagged discrepancy is real | **rejected**. The code already uses `J_eff = Jb+Jh+(m1+m2)Lb^2/4`; the remark's premise is false and there is nothing to reconcile | `parameterization-and-scheduling.md` Sect. 4 |
| H3a exact G26 projection unavailable, affine tangent set is the object | **accepted**, follows from H1a | `s3` |
| H3b exact projector at any rank, derivative identity needs differentiating through | **accepted**, 1.96e-14 and 1.29e-13 against 4.49e-02 detached | `s3` T1, T2, T3 |
| H3c protected span depends on expansion point and coverage; displacement error is second order | **accepted**, with the correction that the RELATIVE residual against a first-order signal has slope 1, not 2 | `s3` T5, T6 |
| H4 baseline-only reference set makes the coefficient blind to the latent path | **accepted**, exactly zero across all 66 latent-path parameters on the real gantry; R1 and R2 both repair it | `s4` T1, T2 |
| H5 the assumption split, and the marginal modes are physical | **accepted**, plus three corrections to the source | `gyorok-assumption-audit.md` |

**Unplanned experiments the results forced.** Stage 3's T1 metric inverted its own
conclusion, which required `s3b`. Stage 3c's competition example returned a negative for
the 2026 construction, which required `s3d` to refute the obvious explanation and `s3e`
to establish the real one: G26's Condition 4 is the hypothesis that decides the sign of
the effect. Stage 4's encoder test ran on a window 20 times shorter than the plant's
slowest resonance, which required `s4b`. All four are recorded in `continuation.md`.

See `results/` for logs and `decision-report.md` for the recommendation per priority.
