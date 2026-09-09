# Verification report: revised-ownership trajectory orthogonality, small cases and production gantry

Date: 2026-09-08. **Revised the same day after review.**

> **STATUS: NOT VALIDATED.** Review found a blocking defect in the shared function
> `compressed_nuisance_basis`, which the passing suite did not catch because the small
> reference case implemented the construction a second time and compared that private copy
> against explicit `E` rather than calling the production function (Sect. 2.4). The defect is
> fixed and both suites now call the shared function, but the gantry geometry, penalty-gradient
> and chunking stages remain unexecuted (Sect. 3.4), and the production accumulation hook is
> not installed. What is supported is the mathematical direction and several production
> interfaces, not a correct implementation of the complete trajectory regulariser.
>
> **UPDATE, later the same day.** V3 to V7 have now COMPLETED (Sect. 3.5): 41 checks, 41 passed,
> 0 failed, exit 0, in 440 s of wall clock, on 2 windows. The gantry geometry exists and the
> Sect. 6.5 rank prerequisite passes. The accumulation hook is now installed and tested. Sect.
> 3.4 is kept as the record of the four attempts that did not complete, and its diagnosis is
> corrected there. The remaining gaps are the pilot window set, the GPU/compiled path and
> everything in Sect. 5.4.

Specification: [fixed-reference-rollout-orthogonality-implementation.md](../plan/fixed-reference-rollout-orthogonality-implementation.md).
Prior report (shared-encoder ownership): [nonlinear-closed-loop-orthogonality.md](nonlinear-closed-loop-orthogonality.md).
Live theory and claim ledger: [gaps.md](../gaps.md). This report owns no theory.

Entry point: `scripts/gantry/orthogonality/verify.py`.
Engines: `code/trajectory_ownership_geometry.py` (SymPy), `code/trajectory_ownership_geometry.m`
(MATLAB), `code/trajectory_ownership_geometry_compare.py`.
Gantry integration: `scripts/gantry/orthogonality/gantry/`.

## 0. What this is, and the four levels it keeps apart

The specification defines the method; this report checks that an implementation satisfies it.
Nothing here is an efficacy result, and no training was run: no beta sweep, no parameter-recovery
study, no joint optimisation. Implementation correctness does **not** prove parameter recovery,
statistical consistency, or that no parameter/ANN substitution is possible.

| Level | Meaning | Where |
|-|-|-|
| L1 exact symbolic identity | proved for all parameter values over the rationals, both engines | small cases |
| L2 float64 numerical evaluation at a witness point | **corrected**: previously written as "exact rational point evaluation", which it is not. Every L2 row goes through `numpy.linalg.svd` or equivalent, so it carries round-off and a rank tolerance | small cases |
| L3 numerical / finite-difference | depends on example, dtype and step size | both suites |
| L4 measured gantry geometry | a property of THIS checkpoint, THESE windows, THIS excitation | gantry suite |
| L5 training or recovery guarantee | **nothing here establishes any** | not claimed |

RULES.md Rule 1 categories are carried per check in the JSON. The small suite is **structural**
throughout. The gantry suite is **data-computable** throughout: no simulation ground truth is
read anywhere in it, so every gantry number in this report is a quantity that could also be
computed on Telica.

## 1. Compatibility assessment of the code that already existed

Requested first, and done before any new code was written. Three bodies of code were inspected
separately: the deployed pointwise penalty, the newer generic trajectory core, and the testbed
adapter.

### 1.1 What already matched the specification

| Component | Status | Basis |
|-|-|-|
| `trajectory_orth_projection.TrajectoryProjectionGeometry` profiling and basis algebra | **reused unchanged** | `M_E = I - Q_E Q_E^T`, `Q = orth(M_E L S)`, `Q^T M_E = Q^T` are the specification's Eq. (16); the pre-profiling rank floor fixed on 2026-09-07 is exactly the Sect. 6.4 `tau_S` rule |
| `ProjectedResidualAccumulator` two-pass chunking | **reused unchanged** | implements Eqs. (23)-(24) including the frozen residual and the surrogate; the per-chunk-sum trap is documented in the same module |
| `assert_geometry_detached` / `assert_gradient_policy` split | **reused unchanged** | the second is the actual Eq. (22) check; the first is honestly scoped to the stored basis |
| `subspace_drift` principal angles | **reused unchanged** | Eq. (26) diagnostics, with the rank change reported separately |
| `TrajectoryAdapter` three-intervention contract and `check_ownership` | **reused, contract extended** | the three modes and identity-level disjointness are what the specification asks for |
| `contribution_metrics` profiled-numerator/profiled-denominator convention | **reused unchanged** | Eq. (25) |

### 1.2 What violated the specification and had to change

| Violation | Why it is a violation | Resolution |
|-|-|-|
| The `encoder` parameter group is the WHOLE encoder, so the latent initialisation is profiled away as nuisance | Spec Eqs. (7)-(8) put latent initialisation in `eta`, trainable under both the base loss and the penalty | Encoder split (Sect. 2.1 below); the adapter's `augmentation` group now contains `eta_a` |
| `linear_encoder_init_aug` shares ONE nonlinear correction net between physical and latent rows | Disjoint ownership is then impossible; slicing or detaching an output does not separate the parameters | `SplitEncoderInitAug`, an output-preserving migration, checked for parity AND for identity-level and behavioural disjointness |
| `build_geometry` differentiates the rollout with respect to encoder parameters directly | 6774 encoder scalars means 6774 rollouts; and the file's own comment already says it must not be reused for the gantry | Compressed construction `compressed_nuisance_basis`, Eq. (15), verified against explicit `E` |
| `penalty` is `beta \|\|Q^T L d\|\|^2` with no `1/N` | The production loss is `mse_loss`, a mean; a summing penalty is in a different metric by the factor `N` | Added `penalty_mse`, Eq. (11). The sum convention is kept for existing callers, and a beta measured under one does not transfer to the other |
| Zero profiled rank raises `ValueError` | Spec Sect. 6.5 makes `r_bar = 0` a result to diagnose; a probe cannot report a rank if constructing the object throws | `strict=False` reports it as `zero_rank` |
| No intervention mechanism on the production ANN path | Modes were testbed-only rollout flags | Non-persistent `out_gate` buffer on `Static_ANN_Block`, an exact no-op at its default |
| The verification suite's own first draft formed dense `N x N` projectors for the robustness comparison | The specification prohibits a dense `N x N` projector outright, and gaps.md Sect. 7h records this exact bug crashing the machine at `N = 7200` | `gantry/subspace.py`: `\|\|P1 - P2\|\|_2` computed in the factors |

### 1.3 What was deliberately NOT duplicated

The gantry adapter has no rollout of its own. It calls `fit_sys.encoder`, `fit_sys.simulate` and
`closed_loop_rollout`, and its primary parity target is `fit_sys.loss` rather than another
private loop. The shared algebra is imported, not re-transcribed: a bug found on the testbed is
fixed for the gantry and conversely.

### 1.4 Mathematical inconsistencies found in the specification

None. Every equation checked (4, 9, 10, 11, 13, 14, 15, 16, 19, 21, 22, 23, 24) is consistent
with the others as written. Two places where the specification is right and an implementation was
wrong are recorded in Sect. 2.3 below.

## 2. Small mathematical reference cases

Model: closed loop, two physical states, one latent state, one controller state, three windows of
three scored outputs, `N = 9`; output computed before the state transition, matching production
timing. Ownership is DISJOINT by construction, which is the whole point of re-deriving these:

    lambda = (kap1, kap2, nu, sig)   physical, deliberately rank deficient: only kap1+kap2 enters
    eta    = (Kk, Ff, Gg, Pp, ca)    augmentation, INCLUDING the latent initialisation ca
    xi_p   = (xp1, xp2, xp3)         physical-initialisation nuisance, nonlinear in xi_p

`kap1 + kap2` is the small analogue of the gantry's `kb1+kb2`, `cb1+cb2`, `Jb+Jh`, so the rank
policy of Sect. 6.5 has a known structural null direction to detect rather than a conditioning
artefact.

### 2.1 Results

**SymPy: 57 checks, 57 passed, 0 failed, 0 undecided, 183.6 s** (54 before the review; +2 for the production-function check `A5b`, +1 for splitting the corrected route row).
**MATLAB: 23 checks, 23 passed, 0 failed, 0 undecided, 37.0 s.**
**Shared: 23 of 23 in agreement, all true in both. 0 disagreements. 0 undecided in either.**

The MATLAB script transcribes the model equations from the written specification and reads
nothing the SymPy run produces. MATLAB is about 5x faster in-engine on this workload, between the 2.5x of
algebra-tooling.md Sect. 2 and the 7x of 2026-09-07; engine startup is excluded from both.
That is a timing observation, not a correctness argument. Note also that the 23 shared
checks do NOT cover the compressed numerical construction, which only SymPy exercises, so
their agreement must not be quoted as two-engine verification of it.

| Check | Level | Result | What passing establishes | What it does not |
|-|-|-|-|-|
| Sensitivity recurrence Eq. (21) equals direct differentiation, for `lambda`, for `eta` including `ca`, and for `xi_p` | L1 | identity, both engines | the recurrence is the exact derivative of the rollout, latent and controller paths included | nothing about the gantry blocks |
| Same for the off-mode physical sensitivity `S_off` | L1 | identity, both engines | Eq. (29)'s two sensitivities are both computable from one recurrence | that either is the right target for a given claim |
| Negative control: severing the controller-state coupling changes `S` | L1 | non-vanishing, witness, both engines | an omitted controller derivative path is detectable | that no other omission is silent |
| `p_off` independent of every augmentation parameter, `ca` included | L1 | identity, both engines | the premise licensing a no-grad `off` | that a future ownership change inherits it |
| `p_clamped` independent of `ca`; clamped latent state null at every step | L1 | identity, both engines | the clamped intervention isolates the direct route | that an ungated state-write path could not break it |
| `d = d_lat + d_dir` | L1 | identity | **nothing about simulator semantics**; algebraically trivial for any three vectors | see gaps.md Sect. 10, which already records the opposite claim as FALSE |
| `E_w = Gamma_w J_w`, hence `E = blockdiag(Gamma_w) stack(J_w)` | L1 | identity, both engines | D-184 is exact for an encoder entering only through the initial state | that the gantry encoder has that property (checked separately, Sect. 3) |
| `xi_p` has no residual dependence with the initial states held free | L1 | identity, both engines | the chain rule's premise holds in this model by construction | applicability to an encoder that also feeds an output or scheduling path |
| `D_ca p = (D_a0 p)(D_ca a_0)` | L1 | identity, both engines | the latent-initialisation derivative factorises the same way | that the implementation routes it (V4) |
| `D_ca p` not identically zero | L1 | witness, both engines | `ca` is a real degree of freedom, so profiling or detaching it changes the method | anything about its magnitude |
| `E = U_G Z`, `U_G^T U_G = I`, compressed `Q_E` spans exactly `range(E)` | L2 | max abs `1.14e-13`; `U_G^T U_G - I` `6.66e-16`; projector difference `5.48e-16`, ranks 3 = 3 | the compressed build of Eq. (15) is exact before truncation | that truncation at gantry conditioning keeps it exact, or that `Z` is affordable |
| **The PRODUCTION function `compressed_nuisance_basis` reproduces both the inline reference and explicit `E`** | L2 | distance to the reference `0.000e+00`, to explicit `E` `5.48e-16`, ranks 3 = 3 = 3; per-window ranks and `Z` shape agree | that the function the gantry adapter actually calls computes the construction this file proves | that it stays affordable, or that its truncation is right at gantry conditioning |
| Nuisance range invariant under `xi' = T xi` | L3 | 8 diagonal rescalings in `[0.1, 10]`, worst projector difference `1.22e-15` | the exact invariance of Sect. 6.6 survives truncation here | invariance at arbitrary conditioning |
| `M_E` symmetric idempotent; `Q^T M_E = Q^T`; `Q_E^T Q = 0` | L2 | all at round-off | profiling is embedded in the basis | |
| `V = (beta/N)\|\|Q^T d\|\|^2` differs from the sum convention by exactly `N` | L1 | `N = 9`, `V_mse 3.404034e+01`, `V_sum 3.063631e+02` | the recorded beta convention; an old sum-based beta needs conversion by `N` | |
| `S n = 0` for `n = e_kap1 - e_kap2` | L1 | identity, both engines | the raw coordinate set is structurally rank deficient, as the gantry's is | the gantry rank, which is measured |
| Encoder overlap removes a physical direction | L2 | `r_bar` 3 -> 2 when an exact copy of an `S` column is appended to the nuisance space | this is the mechanism the `0 < r_bar < r_S` branch detects | that the gantry encoder overlaps anything |
| `P_(eps E) = P_E` | L2 | projector difference at round-off for `eps = 1e-9` | a small encoder block erases exactly as much as a large one; burn-in magnitude decay cannot remove the encoder projector | |
| `r` splits over row chunks; `beta\|\|sum r_b\|\|^2 != beta sum\|\|r_b\|\|^2`; an explicit cancelling pair exists | L2 | split residual `< 1e-13`; a constructed `r_1 = -r_2 != 0` | chunking is a memory device and a per-chunk sum is a different objective | |
| Penalty gradient reaches `ca`, `Kk`, `Ff`, `Gg`, `Pp` | L1 | witnesses, `ca` in both engines | the latent initialisation is not silently detached | anything about training |
| `D_lambda V` not identically zero | L1 | witness, both engines | routing only `D_eta V` is not the gradient of `J_base + V` | mixed-partial asymmetry on its own |
| **`D_lambda D_eta V` not identically zero** | L1 | witness, both engines | **the selectively routed field `(0, 0, D_eta V)` has an asymmetric Jacobian, so it is the gradient of NO scalar objective.** This is the witness the 2026-09-08 correction said was missing | anything about convergence of the resulting update |
| Zero projected value with nonzero tangent overlap | L2 | `\|\|Q^T d_perp\|\| 1.66e-14` against `\|\|Q^T R\|\| 3.03e+01` | `V = 0` does not imply local inability to substitute | |
| Projected route amplitudes are not additive | L2 | `2.60e+01` + `6.45e+00` = `3.25e+01` against a total of `2.88e+01` | a route ratio is not a share of an energy budget | **corrected**: the earlier version of this row claimed these numbers show a route exceeding the total. They do not, and at this checkpoint the total exceeds both routes |
| Route cancellation is realisable in this geometry | L2 | rescaling the latent route by the least-squares coefficient (alpha = -0.080) gives `6.10e+00` against a direct-route `6.45e+00` | the geometry PERMITS cancellation, so both routes must be logged | that the trained augmentation exhibits it; it does not here |
| Analytic derivatives vs central finite differences, all 12 parameters | L3 | worst relative `1.54e-10` (`xp1`); `ca` `6.63e-12` | the symbolic derivatives are the derivatives of the evaluated map | |

Retained scope statements, checked in as rows rather than prose: a finite `beta` gives only
`||Q^T d||/sqrt(N) <= sqrt(eps/beta)`, never zero; and every identity here is about a noiseless
deterministic map and bounds no estimator bias.

### 2.2 One engine failure, recorded for algebra-tooling.md clause 6

**2026-09-08, SymPy 1.14.0, symbolic `Matrix.rank()`.** `E.rank()` on a `9 x 3` matrix of
degree-6 polynomials in twelve symbols did not terminate in over ten minutes; the run was killed
twice before the cause was located. The fix is not a faster rank: a rank is a property of a
POINT, not an identity, so it moved to a numerical evaluation at an integer witness, which is
where algebra-tooling.md rule 5 puts data-dependent quantities anyway. MATLAB was not asked for
a symbolic rank and so did not hit this.

A second, milder one: `sp.expand(Matrix) == 0` is `False` for every matrix, zero or not, so a
zero test written against a scalar silently reports every matrix identity as a failure. Caught
because it failed ALL of them at once rather than one.

### 2.3 Two implementation defects the small suite found in its own author's code

Both are recorded because they are the kind the suite exists for.

1. **The clamped intervention left `a_0` at its learned value.** The specification says clamped
   starts from `a_0 = 0`; the first draft gated the latent WRITE but not the latent
   INITIALISATION, so `p_clamped` still depended on `eta_a` and the first latent state was
   nonzero. `A3.latent_init_absent_from_clamped` failed and located it.
2. **The robustness check materialised a dense `N x N` projector.** Prohibited by the
   specification's own storage table, and the same bug that crashed the machine at `N = 7200` on
   2026-09-07. It was found by the run stalling, not by a check, which is itself worth noting:
   there is no automated guard against reintroducing it, only `subspace.py` being the one place
   the comparison is implemented.

### 2.4 The defect this suite did not catch, and why

**Found in review, 2026-09-08, not by any check here.** `compressed_nuisance_basis` computed

    Z_w = U_w^T J_w          instead of        Z_w = (U_w^T Gamma_w) J_w

omitting `Gamma_w` entirely. It is not a small numerical error: with the gantry's shapes `U_w^T`
is `r_w x 1200` and `J_w` is `6 x 6774`, so the product is dimensionally invalid and raises.
V3 would therefore have failed even on a machine with headroom, which means the resource failure
of Sect. 3.4 was masking a second, independent defect.

**Why nothing caught it, and this is the part worth keeping.** The small reference case built the
compressed factorisation INLINE, correctly, and compared that private copy against an explicitly
assembled `E`. Two independent implementations agreeing is exactly what the dual-engine rule
asks for at the level of MATHEMATICS, and it establishes nothing whatever about the third
implementation, which is the one the gantry adapter calls. The suite reported the construction
verified while never executing the function under test. The dual-engine rule is about
independence of TRANSCRIPTION; it does not substitute for exercising the production code.

The 23 shared MATLAB/SymPy checks likewise do not cover this construction at all, and their
agreement must not be quoted as two-engine verification of it.

**The fix, and the regression guard.** `C_w = U_w^T Gamma_w` is restored, and two callers now
exercise the shared function directly: `A5b` in the SymPy suite, on the same inputs its own
inline reference uses, and `test_trajectory_projection.v3b_compressed_nuisance`, which is
dependency-free and runs in a second over four shapes plus a rank-deficient `Gamma_w` and a zero
block.

**A second defect, introduced by the fix and caught by that new test within a minute.** The
repair initially dropped `U_blocks.append(U)`, so `Q_E` came back all zeros with the correct
shape and the correct rank recorded in the info dict. A rank check alone would have passed it.
The orthonormality and projector-distance assertions are what failed.

## 3. Gantry geometry and production integration

Checkpoint `gantry_ckpt_81757`, run
`scripts/gantry/meeting/meeting-07-09-2026/server/augmentation_ma50_b140-230_a6_z03_linear_map/81757`.
Configuration read from its own `config.json` and reproduced, with three deliberate deviations,
all recorded in the manifest: `device='cpu'`, `compile_mode=None` (eager, fixed shapes, as
Sect. 13 requires for the first measurements), and `joint_estimation=True`.

| | |
|-|-|
| `nx_phys` / `nx_ann` / `nu` / `ny` | 6 / 8 / 3 / 3 |
| ANN outputs, routing | 14 outputs; `route_ix = 0..13`; physical output columns `0..5`, latent `6..13` |
| `nf` / `burn_in` / scored `T` | 400 / 0 / 400 |
| Encoder | `linear_map`, `na = nb = 29`, `na_right = nb_right = 1` |
| Controller | `ControllerBank`, 9 states, 8 distinct controllers over the 14 training records |
| dtype | float32 checkpoint, widened to float64 for the geometry reference |
| Parameter counts after the split | `lambda` 14, `eta` 9166 (ANN 1982 + `eta_a` 7184), `xi_p` 6774 |
| Windows | 4, one per record, records T1 / T6 / T9 / T13 (standstill, y-sweep, aprbs, lissajous), seed 20260908 |
| `N = n_W T n_y` | 4800 |

### 3.1 Conversions, P0

| Check | Result | Tolerance | Establishes | Does not establish |
|-|-|-|-|-|
| `Parameterized_Gantry_State_Block` at `log_params = 0` reproduces the trained `Gantry_State_Block` | max relative `9.75e-11` | `1e-10` | `S` is the physical sensitivity of the predictor that was ACTUALLY trained, not of a different model | that the run optimised those coordinates; it did not |
| No parameter on the `1e-6` clamp | `theta/theta_init` in `[1.000000, 1.000000]`, min `theta = 5.0e-02` | 0 active | the smooth-argument requirement of Eq. (2) holds at the reference | that it holds during training; clamp activation must be monitored there too |
| Split encoder reproduces the shared-net encoder's initial states | max abs `0.000e+00` | bit-identical | the migration preserves the function at the reference | that the two OPTIMISATION problems are the same; sharing changed |
| Split groups disjoint by tensor identity | `xi_p` 6774 scalars, `eta_a` 7184 scalars, no shared id | exact | identity-level disjointness | behavioural disjointness |
| Perturbing `eta` leaves the physical initial state bit-identical, and conversely | both `0.000e+00`, against own changes `4.4e-02` and `3.4e-02` | exactly zero | the two initialisations have genuinely disjoint dependencies | anything about the rest of the predictor |

### 3.2 Production parity and interventions, V1

| Check | Result | Establishes | Does not establish |
|-|-|-|-|
| Adapter scored loss vs `fit_sys.loss` on the same windows | `1.212719557427e-09` both, relative `0.00e+00` | the stack order, burn-in slice and averaging match the trained objective | that the interventions are correct |
| Gate restored after an exception; default all-ones | restored, all ones | exception-safe scoping; loading starts in full mode | compiled-path parity |
| `d = d_lat + d_dir` | `0.00e+00` | one shared rollout path | **nothing about simulator semantics** |
| Latent route nonzero | `\|d\| 8.61e-03`, `\|d_lat\| 5.59e-03`, `\|d_dir\| 1.22e-02` | the latent route carries contribution at this checkpoint | any share-of-energy reading; these are not percentages of an additive budget |
| Clamped latent state zero at every one of 400 steps | `0.000e+00` | the clamped intervention isolates the direct route under this routing | that a future ungated write path could not break it |
| `p_off` unchanged by perturbing all 18 augmentation tensors | `0.000e+00` | the premise licensing a no-grad `off` | that a future ownership change inherits it |
| AD-enabled `off` has no gradient into `eta` | `0.000e+00` | the derivative half of the same premise | |

### 3.3 Derivatives, V2

Central finite differences of the production rollout, three step sizes, best reported.

| Direction | Best relative error | Step | Note |
|-|-|-|-|
| Physical, random `log_params` direction | `6.15e-08` | `1e-4` | forward-mode JVP through the 400-step closed loop |
| Augmentation group, random direction | `1.06e-08` | `1e-6` | includes `eta_a` |
| Encoder group `xi_p`, random direction | `5.15e-08` | `1e-6` | |
| `E_w = Gamma_w J_w` against a direct `xi_p` perturbation | `2.25e-08` | `1e-6` | **D-184 verified on the production predictor** |
| `xi_p` with `x_0` held fixed | `0.000e+00` | exact | the chain rule's premise holds for THIS encoder |
| Negative control: controller state frozen inside the rollout | relative change `2.48e+01` | vs a `6.15e-08` check floor | an omitted controller path would be caught by nine orders of magnitude |

The error decreasing by about a factor 10 per decade of step size is the expected first-order
behaviour of a central difference against a float64 rollout; it is reported across steps rather
than at one step for that reason.

**Replicated on a second window set.** The same stages were run with two windows (records T1 and
T9) and gave the same verdicts, with the derivative errors moving as they should when the window
set changes: physical `5.07e-08`, augmentation `1.87e-09`, encoder `6.11e-08`, chain rule
`1.08e-08`, controller negative control `2.44e+01`. The bit-exact rows (production loss parity,
clamp, off-independence, `xi_p` held-`x_0`) were bit-exact on both sets. That run is the saved
record: 21 checks, 21 passed, 0 failed, 0 skipped, in
`results/20260908_run/gantry_nominal.json`.

**Two windows, not four, in the saved record.** The window count was reduced while diagnosing
the V3 block (Sect. 3.4). It changes `N` from 4800 to 2400 and it changes nothing about which
identities are being checked; the four-window numbers quoted in the tables above come from the
earlier run whose console log is saved alongside.

### 3.4 Stages implemented but NOT executed to completion: V3, V4, V5, V6, V7

This is an unresolved failure on this machine and it is reported as one.

**What the timing actually says.** The saved run record settles a question that watching the
console could not. P0, V0, V1 and V2 complete in **178.7 s of wall clock** in total, of which V2
is **158.5 s** and building the environment is **5.0 s**. The executed part of the ladder is
fast. `stage_V3` then never returns: across four attempts it produced no output for 20 to 100
minutes while the process accumulated 16 to 40 s of CPU per 20-minute window, roughly one per
cent utilisation. It is BLOCKED, not slow.

**What was ruled out, and what was not.**

| Hypothesis | Test | Outcome |
|-|-|-|
| The explicit-`E` SVD, `4800 x 6774`, about `1.6e11` flops and a gigabyte | replaced by a row-subsampled instance of the same identity | a real bottleneck, correctly removed; **did not** unblock V3 |
| The dense `N x N` projector in the robustness check | replaced by `gantry/subspace.py` | a real violation of the specification, correctly removed; **did not** unblock V3 |
| The second fit system built for the P0 conversion check | released explicitly | correct regardless; **did not** unblock V3 |
| Problem size | window set halved, `n_W` 4 to 2, `N` 4800 to 2400 | **did not** unblock V3 |
| Thread count | `torch.set_num_threads(1)` | **made it worse**: a no-grad rollout costing under two seconds sat at 9 s of CPU over ten minutes. Reverted; the thread count is now a recorded run parameter, and the run that produced the saved record used 6 |
| Competing Python processes left by earlier attempts | killed | helped, and is why V1 and V2 completed at all in the final attempt |
| Memory pressure | free physical memory measured at 1.6 to 2.7 GB of 16 GB throughout | consistent with blocking on paging, but **not established** |

**The cause is not identified, and naming one would be a guess.** V3 begins with fourteen
forward-mode physical JVPs, and V2 performs three of the identical call at the same problem size
without trouble, so "the JVP is too expensive" does not fit the evidence either.

**What this does and does not mean.** It is not evidence that the geometry stages are wrong, and
it is not a measurement of their cost, because no completed timing for them exists. It IS the
specification's own stop condition: Sect. 12 makes an unexplained resource failure a blocker on
scale-up, and Sect. 13 requires per-stage memory to be measured rather than assumed. The next
step is `--stages V3 V4 V6 V7` on a machine with headroom, or on the GPU, with a stack dump taken
at the block. Until then no rank, no conditioning number, no chunking result and no
gradient-ownership result exists for the gantry, and none is claimed.

**A gap in the machine-readable deliverable.** `peak_memory_mb` is `nan`. `psutil` is present in
the base interpreter but NOT in the `GraduationProject` environment, and Windows has no
`resource` module, so the per-stage peak memory the specification asks for by name was not
captured. Installing `psutil` into the environment is a prerequisite for the next attempt.

**One cost figure that IS measured**, from the four-window attempt that reached V3: a forward-mode
physical JVP through the 400-step closed-loop rollout took about 13 s against a 1.2 to 1.5 s
median for a plain rollout, so `S` is roughly fourteen of those. The closed-loop simulator's
recompile detector fires on every one of them, which is a false positive here, since
`compile_mode` is `None` and nothing is compiled.

### 3.5 V3 to V7, completed

2 windows (records T1 and T9), `N = 2400`, float64, eager CPU, checkpoint 81757.
**41 checks, 41 passed, 0 failed, 0 skipped, exit 0, 440 s wall clock.**
Results: `results/20260908_diag2/gantry_nominal.json`.

| Quantity | Value | What it means |
|-|-|-|
| `rank(S)` on 14 raw log-coordinates | **10** | exactly the structural bound the specification derives from the factorisation through `kappa` (Eq. 3). Measured on the real predictor, not inherited from the small example |
| `rank(Q_E)`, from 6774 encoder scalars | **12** | `r_G = 12 = 2 windows x 6`, so every `Gamma_w` has full rank 6 |
| `rank(Sbar)` after profiling | **10** | **the shared encoder removed no supported physical direction** |
| tolerance sweep 0.1x / 1x / 10x | (12, 10) at all three | the ranks are not a threshold artefact |
| nuisance coordinate robustness | `2.49e-15` over 6 diagonal rescalings | Sect. 6.6 invariance survives truncation here |
| free per-window initial-state bound | rank 12 -> `r_bar` 10 | consistent with the shared-encoder result; still only a labelled BOUND |
| supported spectrum of `Sbar` | `1.30e-02` down to `2.79e-06`, then `7.7e-18` and below | ~4.6 decades across the supported directions; the last four are the exact null directions |
| compressed vs explicit `E` | `Z` is `12 x 6774`; an explicit `E` would be `2400 x 6774` | the factorisation of Eq. (15) doing what it is for |

**This is the Sect. 6.5 branch `r_bar = r_S = 10` with robust spectra: the rank prerequisite for
a ten-combination pilot passes.** It establishes neither tangent separation (Eq. 20) nor
recovery, and the specification requires confirming the same gates on the final window set.

Gradient ownership, chunking and execution:

| Check | Value |
|-|-|
| `max\|g_lambda\|`, `max\|g_xi_p\|` from the penalty | `0.000e+00`, `0.000e+00` |
| `max\|g_eta_d\|` / `max\|g_eta_a\|` | `4.72e-06` / `8.84e-11` |
| chunked (2 chunks) vs full stack | rel value `8.1e-13`, rel grad `3.9e-12` |
| checkpointed vs un-checkpointed gradient | `0.000e+00` (chunk 100) |
| out-of-scope checkpoint replay detected | `1.0` |
| accumulation-order difference | `1.93` |
| feature-disabled no-op | bit-identical, gate absent from the state dict |

**Flagged rather than buried:** `g_eta_a` sits about five orders of magnitude below `g_eta_d`.
It is nonzero, which is what the contract requires and what the specification asks to be
measured, but "the penalty reaches the latent initialisation" and "it reaches it usefully" are
different claims and only the first is established.

Route metrics at this checkpoint (Eq. 25), for the record and not as a result:
`ell` total `0.539`, latent `0.802`, direct `0.666`.

**Correction to Sect. 3.4's diagnosis.** The stage completed on the SAME machine that produced
four stalls, after a CUDA-initialisation defect in the new `diag.py` was fixed. That defect
post-dates the original stalls, so it cannot be their cause and no claim is made that it was.
What can be said is narrower and worth saying: "a bigger machine" was not the answer, and the
memory-pressure reading in Sect. 3.4 is weaker than it appeared. The per-operation ledger is what
made the difference legible: each operation ran at 105% CPU while the process accrued 2%, which
localised the cost to the instrumentation between the blocks rather than to the Jacobians.

## 4. Reproduction

```text
conda run -n GraduationProject python scripts/gantry/orthogonality/verify.py --suite small
"C:\Program Files\MATLAB\R2025a\bin\matlab.exe" -batch "addpath('scripts/gantry/orthogonality/code'); trajectory_ownership_geometry"
conda run -n GraduationProject python scripts/gantry/orthogonality/code/trajectory_ownership_geometry_compare.py
conda run -n GraduationProject python scripts/gantry/orthogonality/verify.py --suite gantry --windows 4 --perturbed
conda run -n GraduationProject python scripts/gantry/orthogonality/testbed/testbed_adapter.py
```

Machine-readable results: `scripts/gantry/orthogonality/results/<stamp>/` for the gantry suite
and `code/trajectory_ownership_geometry/` for the two engines and their comparison. Each row
carries its statement, reference equation, expected value, tolerance, category, and the
`establishes` / `not_establishes` pair quoted in the tables above.

## 5. What is established, and what is not

### 5.1 Mathematics established, under the stated assumptions

Exact identities, proved in two engines reading two independent transcriptions, for a smooth
closed-loop model with a controller, an internal learned state, disjoint parameter ownership and
a trainable latent initialisation:

- The forward sensitivity recurrence equals direct differentiation, for physical, augmentation
  and nuisance parameters, in full and off modes.
- `p_off` is independent of every augmentation parameter, the latent initialisation included.
  `p_clamped` is independent of the latent INITIALISATION only, and the clamped latent state is
  null at every step; it still depends on the direct-route ANN parameters by construction, which
  is the point of the clamped intervention. **Corrected**: an earlier version of this line said
  `p_clamped` was independent of every augmentation parameter, which is false and is broader
  than the check it summarised.
- `E_w = Gamma_w J_w`, and `E = blockdiag(Gamma_w) stack(J_w)`, given that the nuisance enters
  only through the physical initial state; that premise is itself checked, not assumed.
- `D_ca p = (D_a0 p)(D_ca a_0)`, and `D_ca p` is not identically zero.
- The projection identities, `Q^T M_E = Q^T`, and zero penalty value iff contribution
  orthogonality for `beta > 0`.
- The plain-MSE normalisation differs from the sum convention by exactly `N`.
- Exact two-pass chunking, and a constructed case where a per-chunk sum is a different objective.
- `S n = 0` on the structural null direction of a rank-deficient coordinate set.
- **`D_lambda D_eta V` is not identically zero**, so the selectively routed update is the
  gradient of no scalar objective. This closes the gap the 2026-09-08 correction left open;
  the earlier report's weaker nonzero-physical-gradient check was not sufficient for it.

Verified numerically at L2/L3 in one engine: the compressed range construction against explicit
`E` (projector difference `5.5e-16`), nuisance-coordinate invariance under rescaling, the
encoder-overlap rank drop, projector scale invariance, and analytic derivatives against central
differences for all twelve parameters (worst `1.5e-10`).

### 5.2 Implementation behaviour verified on the production gantry

Against checkpoint 81757, on two independently drawn window sets: the physical-block and encoder
conversions are output preserving; the parameter groups are disjoint by tensor identity AND
behaviourally; the adapter reproduces `fit_sys`'s own scored loss exactly; the gate is
exception safe, defaults to full mode and is absent from the state dict; the clamped latent state
is exactly zero at every step; `p_off` is bit-identical under perturbation of all 18 augmentation
tensors and carries no autograd dependence on them; and every parameter group's derivative,
including the D-184 chain rule, agrees with central finite differences of the production rollout.
The controller negative control fires at a relative `2.5e+01` against a `6e-08` check floor.

### 5.3 Gantry-specific geometry measured

**Measured, on a 2-window smoke set** (records T1 and T9, `N = 2400`, float64, eager CPU,
checkpoint 81757). See Sect. 3.5 for the table. `rank(S) = 10` of 14 raw log-coordinates, which
is exactly the structural bound; `rank(Q_E) = 12`; `rank(Sbar) = 10`, so **encoder profiling
removed no supported physical direction**. Stable across a 0.1x/1x/10x tolerance sweep.

That is the Sect. 6.5 branch `r_bar = r_S = 10 with robust spectra`: the rank prerequisite for a
ten-combination pilot **passes on this set**. It does not establish tangent separation, and it
does not establish recovery. The specification requires the same gates on the FINAL window set
before training, and two windows is not that set.

### 5.4 Remaining mathematical and statistical questions

Unchanged by this work, and none of them is addressed by it:

- Whether a finite `beta` selects usefully. `beta` remains a chosen constant with no
  justification until an L-curve is run; the suite's default of 1.0 is a placeholder for
  checking identities, not a selection.
- Local non-substitution, `range(M_E S) intersect range(M_E R) = {0}`, and the Sect. 7.4
  capability diagnostic. `V = 0` does not imply it, and the small suite exhibits a case with zero
  projected value and a tangent overlap of `3.03e+01`.
- Whether the constraint preserves correction that improves prediction. Efficacy is V8 and is
  out of scope here.
- Noise. **The perturbed/noisy arm was NEVER EXECUTED**; there is no completed JSON for it, and
  every claim about it must remain unexecuted. **Corrected**: an earlier version of this line
  said the arm establishes that the identities and derivatives hold under changed inputs. The
  machinery for it exists (`--perturbed`, with the noise drawn once and frozen across every
  paired evaluation) and it has not been run. Even when it is, it would say nothing about
  estimator bias, errors in variables, or physical recovery under noise, and the geometry stages
  are deliberately excluded from it, because a rank measured under one frozen draw is not a
  statistical statement.
- Frozen-basis adequacy, and re-anchoring, which remain a separate algorithm.

### 5.5 Remaining implementation work and unexecuted checks

1. ~~V3 to V7 on the gantry.~~ **DONE**, 2 windows, CPU (Sect. 3.5). Still to do: the pilot
   window set, and a GPU run (`runners/run_verify_ladder.sh`).
2. ~~The accumulation hook is not wired into the production closure.~~ **DONE.**
   `SSE_Interconnect_Composed.fit` wraps `optimizer.step` so the penalty lands between the
   closure returning and the update; Jan's `fit()` is untouched and no loop is duplicated.
   `traj_orth=False` returns to the parent immediately, so an existing run is unchanged.
   `testbed/test_penalty_hook.py` covers it (12/12), including that the wrong order loses the
   penalty and that a `backward=False` line-search evaluation adds nothing.
3. **Compiled and CUDA parity.** Not measured. Everything here is eager CPU.
4. **The perturbed arm** was launched but is gated behind the same resource failure; its
   P0/V0/V1/V2 rows are the selected repeats Part D asks for, and its geometry stages are
   excluded by design rather than by the failure.
5. **A guard against reintroducing a dense `N x N` projector.** There is none; `subspace.py`
   being the single implementation is the only protection, and this report records that the bug
   was reintroduced once already.
6. **The recompile false positive** in `ClosedLoopSimulator._note_update_time`, which fires on
   every JVP in an uncompiled run.
7. **Directional derivative checks are SAMPLED evidence.** V2 checks one random direction per
   group against finite differences. It does not verify every Jacobian entry, and it is a
   derivative of the PREDICTION, not of the projected penalty; the penalty gradient is V4 and
   has not run on the gantry.
8. **The accumulation-hook demonstration is weaker than it should be.** Showing that two orders
   give different updates is a necessary condition, not the check that matters. What is needed
   is the correct gradient SUM and one complete optimizer step against a reference, which is
   what `stage_V4_V5` does and which has not run on the gantry.
9. **The suggested GPU retry is not plug-and-play.** `config_from_run` forces `device='cpu'` and
   several tensor constructors in the adapter and checks assume CPU tensors. Moving the suite to
   the GPU is its own piece of work, not a flag.
10. **`psutil` gives no stage-local attribution.** Installing it fixes `peak_memory_mb` being
    `nan`, but locating the V3 block needs per-operation instrumentation inside `stage_V3` and a
    stack dump at the stall, not a single peak figure.


## Addendum, 2026-09-08 (later): the two decisive checks, and a stale-reference defect

`check_update.py` runs the two checks that decide whether the real training path executes the
specified update. Reference run `81757`, 2 regularisation windows, batch 8, `beta = 1`, float32,
CPU, `lr = 1e-4`, Adam `eps = 1e-16`.

### Check 1: one production update against an independent reference

The reference assembles `V` from scratch (encode, `p_off` under the gate, `p_full`, `d`,
`Q_S^T d`, `beta ||.||^2 / N`) and backpropagates it, with no use of the hook's own machinery.

| Quantity | Production | Reference |
|-|-|-|
| base loss | `9.176311932535e-10` | identical |
| penalty `V` | `3.771410739e-09` | identical |
| gradient, all three groups | relative `0.000e+00` | |
| post-step parameters, all three groups | relative `0.000e+00` | |

The penalty's own contribution is `0.000e+00` to physical and to encoder, and `2.257e-05` to
augmentation. **This supports a narrow statement: for this configuration, one production update
equals the specified update, and the penalty moves only `eta`.** It is not evidence about
parameter recovery, about statistical consistency, or about any beta other than 1.

### Check 2: a confirmed defect, found by review, in the production resume path

The first version of check 2 failed at `1.152e+00` on the physical group. That was **not** a
tolerance issue. Review identified the cause, and it is confirmed:

`GantryTrajectoryAdapter.__init__` cached `enc`, `ann` and `phys` as snapshots.
`checkpoint_load_system` does `self.__dict__ = torch.load(file)`, which preserves the system
object's identity while replacing every module inside it. The adapter was therefore spliced
across two model generations: driving the NEWLY LOADED dynamics through `fit_sys.simulate` while
gating the OLD ANN block and owning the OLD parameters. Nothing raises. The failure modes are
silent:

- `ann_gate` gates a block that is not in the rollout, so the gating half of the `off` and
  `clamped` interventions does nothing and **the intervention becomes incorrect**. CORRECTED
  2026-09-08 after review: an earlier version of this paragraph claimed `p_off` then equals
  `p_full` and the penalty vanishes. That is wrong. `scored_stack` (adapter.py:236) zeros the
  latent initialisation for every non-full mode independently of the gate, so `d` is generally
  still nonzero. The penalty does not become a no-op; it becomes **the penalty of a different,
  unspecified intervention**, which is harder to notice, not easier.
- gradients land on tensors the live model no longer contains

The original test also **accepted the defect**. It asserted that the pre-load group list and
`hook.groups()` held the same tensors, but both were read from the same retained hook, so it
passed precisely by demonstrating that the hook still held its old references. It then built the
resumed parameter vector from that same stale list.

**Fix.** `GantryTrajectoryAdapter.rebind(fit_sys)` re-resolves all three references from the live
system, plus `assert_bound()`; `checkpoint_load_system` calls both before anything else touches
the adapter, and refreshes the hook's cached `_groups`.

**Test.** Check 2 was rewritten to resolve everything from the loaded model by stable parameter
NAME, and to verify what the reviewer specified: values by name across the load, the adapter
bound to the loaded modules, no orphaned group, the optimizer owning the tensors the hook writes
into, the wrapper reinstated, the same frozen geometry, and optimizer moments and step counters
preserved.

### Attribution

The fix and the test changed together, so neither run alone attributes the result. A control run
(`scratchpad/attr_no_rebind.py`) neutralises **only** `rebind` and `assert_bound`, leaving all
other production code and the corrected test in place.

| | pre-fix behaviour (control) | fixed |
|-|-|-|
| parameter groups orphaned from the loaded model | physical, encoder, augmentation | none |
| optimizer owns the tensors the hook writes into | no, all three groups | yes |
| resumed vs uninterrupted, worst absolute | `2.517e-04` at `hfn.connected_blocks.2.net.net.2.weight` | `0.000e+00` |
| the same, relative to the total movement of `4.000e-04` | `6.292e-01` | `0.000e+00` |
| result | 20 passed, 4 failed | 24 passed, 0 failed |

The stale references displaced the resumed trajectory by **63 per cent of the size of the update
itself**, and the rebind removes it exactly. One row is uninformative in the control column: "the
adapter is bound to the loaded modules" passes there only because the control neutralises the
assertion that would report it.

The unit tests are also not vacuous. With the rebind disabled in `interconnect.py`,
`test_fit_override.py` goes from 31/31 to 28 passed, 3 failed, and the three failures are exactly
the new rebind rows.

### What this addendum still does NOT establish

1. **GPU, compiled and CUDA paths.** Everything above is eager CPU. Bit-exact agreement is a
   CPU-determinism result and should not be expected to transfer unchanged.
2. **Beta.** Fixed at 1, a chosen constant with no justification. Nothing here selects it.
3. **Multi-epoch behaviour, and pilot window counts.** Two windows and a handful of updates.
4. **`gantry/latent_init_probe.py` has never been run.**
5. **The perturbed arm (Part D) has never been executed.**
6. Bit-exact resume is evidence that the update is *reproduced*, not that it is *correct*;
   check 1 is what speaks to correctness, and only for this configuration.

## Addendum 2, 2026-09-08: two weak assertions replaced, and a vacuity control

Review found two further gaps in check 2. Both are confirmed and fixed.

### Gap 1: optimizer restoration was partly bypassed

The check called `optimizer.load_state_dict(opt_state)` **before** comparing. That can repair a
broken restoration and then report the restoration as correct. The comparison now happens with no
manual repair applied first, keyed by parameter NAME (the load replaces every tensor, so an
id-keyed snapshot cannot be compared across it), covering `exp_avg`, `exp_avg_sq`, `step`, the
parameter-to-group association and the group hyper-parameters. The saved state dict is retained
only as a last-resort repair, applied **after** the comparison, printed as a reported defect, and
recorded in the JSON as `manual_optimizer_repair_was_needed`.

**Result: `manual_optimizer_repair_was_needed = False`.** The checkpoint restores all 29
parameters' moments, step counters and group associations bit-exactly on its own. The previous
manual call was masking nothing, but that could only be established by removing it.

### Gap 2: the frozen-geometry assertion was vacuous

It read `geometry.N == cfg.nf * 0 + geometry.N` and checked that a rank was positive. Neither can
fail. It is replaced by a content comparison across the load of `Q_S` and `Q_E` (sha and shape),
both spectra, `rank_S`, `rank_E`, `rank_S_before_profiling`, `rank_rtol`, `rank_atol`, `scale_LS`,
`zero_rank`, `beta`, `chunk` and the metadata; plus a context comparison of every window tensor,
`ctrl_ix`, `burn_in`, `T`, the simulator's `record_rows`, and **the controller bank's own rows**.
The bank matters because `ctrl_ix` is an index INTO it: identical indices against a reordered bank
scores the right windows through the wrong controllers, silently.

### Vacuity control

Replacing a vacuous assertion is worth nothing unless the replacement can fail. Eleven corruptions
were injected one at a time inside `checkpoint_load_system`, after the model is replaced and
before anything compares (`scratchpad/vacuity_control.py`).

| Corruption | Caught by | Caught by the trajectory comparison too? |
|-|-|-|
| `Q_S[0,0] += 1e-3` | frozen geometry | no |
| `rank_rtol` doubled | frozen geometry | no |
| `beta` scaled by 1.01 | frozen geometry | yes |
| metadata key injected | frozen geometry | no |
| `ufuture[0,0,0] += 1e-3` | context | no |
| `ctrl_ix[0]` changed | context | yes |
| one controller bank row perturbed | context | yes |
| `record_rows` reversed | context | no |
| one Adam `exp_avg += 1e-6` | optimizer as restored | no |
| one Adam `step += 1` | optimizer as restored | no |
| all optimizer state cleared | optimizer as restored | no |

**All eleven were caught, each by the intended assertion.** The third column is the substantive
finding: **eight of the eleven were invisible to the resumed-vs-uninterrupted parameter
comparison** at its 1e-5 tolerance. A resume check built only on comparing the two trajectories
would have passed on a corrupted frozen basis, a corrupted window set, a reversed record mapping
and wholly lost optimizer state. The content assertions are not redundant with the trajectory
comparison; they carry most of the coverage.

Check 2 now reports **31 passed, 0 failed**, up from 24 assertions.

### Correction to Addendum 1

Addendum 1 claimed that a stale ANN reference makes `p_off` equal `p_full` and the penalty vanish.
**That is wrong**, and it has been corrected there and in the three code comments that repeated
it. `scored_stack` (`gantry/adapter.py:236`) zeros the latent initialisation for every non-full
mode independently of the gate, so `d` generally remains nonzero. A stale gate does not empty the
intervention; it makes the intervention **incorrect**, penalising a different and unspecified
quantity while continuing to report a plausible value. That is a harder failure to notice, not an
easier one, so the correction strengthens the case for the rebind rather than weakening it.
