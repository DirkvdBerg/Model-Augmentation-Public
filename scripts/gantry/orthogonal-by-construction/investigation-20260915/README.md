# Orthogonal-by-construction augmentation on the nonlinear self-scheduled gantry

Investigation of 15 September 2026, run autonomously against
`tasks/handoffs/2026-09-15-autonomous-gantry-orthogonal-construction-investigation.md`.
Everything in this folder is new. No existing derivation, decision log, result or
production file was modified.

## Executive recommendation, one per priority

**0. The finding that outranks the others.** Györök's Condition 4 (Eq. 17), that the true
unmodeled term is orthogonal to the baseline regressor on the data, is what decides
whether the construction helps or harms. With everything else held fixed, a residual
lying 94.6 percent inside the baseline tangent span makes the exact subtraction recover
the physical combinations **four times worse than doing nothing** (0.4175 against 0.1122
mean relative error), while the same residual projected out of the span makes it **four
times better than no projection and ten times more reproducible** (0.0206, seed spread
1.0e-03). Condition the claim on this, measure it, and choose the validation residual on
this basis.

**1. Parameterization and Y scheduling.** Keep the explicit self-scheduled RK4 map and
protect the affine tangent set `[Phi_p | c]` at a declared expansion point. There is no
exact linear-in-parameters form of the explicit map, proved symbolically. Build the
reference set with **endogenous** scheduling over the full `Y in [-0.30, 0.30] m` range:
only 0.108 of a frozen-scheduling error lies in the parameter span, so the learning
component would otherwise be free to absorb the scheduling convention itself.

**2. Identifiable combinations.** The 10 quantities already coded in
`Parameterized_Gantry_State_Block._combos_from_raw` are exactly right and exactly
complete: rank 10, nullity 4, verified at full RK4 fidelity. Keep training raw in log
coordinates (52x better conditioned) and reporting only the 10. Two pieces of project
prose need correcting, listed in `parameterization-and-scheduling.md` Sect. 4.

**3. Protected basis and reference policy.** Tangent plus offset, log coordinates,
Moore-Penrose coefficient, **differentiated through** (detaching it destroys the property
entirely: 4.49e-02 against 1.29e-13). Report `rho = ‖P G‖ / ‖G‖`, never `‖Phi^T G̃‖`,
which is dominated by the offset column and inverts the conclusion. Column-scale before
the pseudo-inverse: at `rcond = 1e-4` against `sigma_0` the rank silently falls from 11 to
6.

**4. Additional states and encoder.** Never use the baseline-only rollout as the reference
set. Its coefficient is **exactly** blind to all 66 latent-path parameters, reproduced on
the real gantry, not just on the surrogate. Use R2 (lagged augmented rollout) or R1
(designed latent set). Never constrain the latent rows; the gauge invariance is exact.
Evaluate any encoder claim on a window longer than the 197 ms yaw period: at 10 ms
the median encoder-parameter overlap is 0.99, at 200 ms it is 0.75, but `mh`, `cy` and
`cg1` stay above 0.97 throughout.

**5. Theorem assumptions.** Claim exactly three things: one-step non-overlap over a frozen
reference set at any rank, block structure of the joint Jacobian over that set, and
removal of the source's Example 2 flat direction at the one-step level. Nothing
asymptotic, nothing about true-parameter recovery, nothing at horizon level. Full matrix
in `gyorok-assumption-audit.md`.

## Documents

| file | what it holds |
|---|---|
| `decision-report.md` | the recommendations above with the evidence, the limits, and the next implementation step |
| `parameterization-and-scheduling.md` | governing map, the linear-in-parameters question, the scheduling comparison, the identifiable combinations and the candidate table |
| `gyorok-assumption-audit.md` | 17-row assumption matrix with source equation anchors and a status per row, plus three corrections to the source |
| `experiment-plan.md` | hypotheses and acceptance criteria, written before the experiments ran |
| `continuation.md` | what was not run, how to run it, and the four corrections this investigation made to its own earlier steps |

## Scripts and how to run them

All from the repository root, all float64, all under 20 minutes each.

```
conda run -n GraduationProject python -u scripts/gantry/orthogonal-by-construction/investigation-20260915/s12_parameterization_and_rank.py
conda run -n GraduationProject python -u scripts/gantry/orthogonal-by-construction/investigation-20260915/s3_projection_prototype.py
conda run -n GraduationProject python -u scripts/gantry/orthogonal-by-construction/investigation-20260915/s3b_offreference_behaviour.py
conda run -n GraduationProject python -u scripts/gantry/orthogonal-by-construction/investigation-20260915/s3c_competition.py
conda run -n GraduationProject python -u scripts/gantry/orthogonal-by-construction/investigation-20260915/s3d_refset_identity.py
conda run -n GraduationProject python -u scripts/gantry/orthogonal-by-construction/investigation-20260915/s3e_condition4_control.py
conda run -n GraduationProject python -u scripts/gantry/orthogonal-by-construction/investigation-20260915/s4_latent_and_encoder.py
conda run -n GraduationProject python -u scripts/gantry/orthogonal-by-construction/investigation-20260915/s4b_encoder_window.py
```

| script | question | log |
|---|---|---|
| `obc_common.py` | the standalone float64 gantry, built from source and value-checked against `Gantry_State_Block` | (library) |
| `obc_proj.py` | the learned block, the stacked regressor, the declared pseudo-inverse, `rho` | (library) |
| `s12_...` | is there an exact linear-in-parameters form; what are the identifiable combinations; how different are the scheduling conventions | `results/s12_parameterization_and_rank.log` |
| `s3_...` | exactness, rank deficiency, the derivative identity, the per-sample versus stacked question, expansion point and coordinates, the offset column | `results/s3_projection_prototype.log` |
| `s3b_...` | construction or weak penalty off the reference set, on a scale-free metric | `results/s3b_offreference_behaviour.log` |
| `s3c_...` | the competition example: no projection against the 2025 penalty against the 2026 subtraction | `results/s3c_competition.log` |
| `s3d_...` | is a foreign reference set the cause of 3c's negative result (it is not) | `results/s3d_refset_identity.log` |
| `s3e_...` | the same competition inside Condition 4 | `results/s3e_condition4_control.log` |
| `s4_...` | the zero slice on the real gantry, gauge invariance, encoder sensitivity | `results/s4_latent_and_encoder.log` |
| `s4b_...` | the encoder question on a window long enough to contain the dynamics | `results/s4b_encoder_window.log` |

Machine-readable tables accompany each log as `results/s*_tables.json`.

## Reading order

`decision-report.md` first. Then `gyorok-assumption-audit.md` for what may and may not be
claimed. Then `parameterization-and-scheduling.md` for the identifiability detail.
`continuation.md` last, which also records the mistakes this investigation made and
corrected, so they are not repeated from the logs.

## Two results that supersede earlier text in this folder

1. **`s3` T1's raw orthogonality residual must not be quoted.** `‖Phi^T G̃‖` is dominated
   by the offset column and reported a 5x degradation off the reference set where the
   scale-free `rho` reports a genuine improvement. `s3b` supersedes it.
2. **`s4` T4's encoder numbers must not be quoted.** They come from a 10 ms window against
   a 197 ms yaw period. `s4b` supersedes them, and the "exact dependency" between the
   parameter and initial-state spans that `s4` appeared to show is a rank-threshold
   artifact on a matrix with condition number 1.9e9.

## Scope

Noiseless, float64, one-step, synthetic. No real data were read. No production code,
baseline library, immutable MATLAB source, existing derivation or prior result was
touched. No consistency, optimality, stability or novelty claim is made anywhere.
