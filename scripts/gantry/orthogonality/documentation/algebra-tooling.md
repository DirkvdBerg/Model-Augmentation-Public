# Rule: which CAS for which algebra, in this subfolder

**Set** 2026-09-03, from the measurement in Section 2 rather than from preference.
**Scripts** `../code/cas_benchmark.py`, `../code/cas_benchmark.m`,
**result** `../code/cas_benchmark_sympy.json`.

## 1. The rule

1. **Every structural proof about the FP model is written twice**, once in MATLAB
   transcribing from `kamtin-fp-model/` and once in sympy transcribing from the Python
   implementation under `model_augmentation/`. Agreement between two engines reading two
   independent sources is the check; either alone is weaker than it looks. This is not
   redundancy: the sympy proof of the null-space claim, run alone, would have established
   only that the *Python* model factors through the identifiable combinations, which is not
   the claim the thesis needs.
2. **MATLAB is authoritative on FP-model structure.** It reads the ground truth. A sympy
   proof always goes through a transcription into Python first, and a transcription error is
   exactly what the dual proof exists to catch. Where the two disagree, MATLAB is right about
   the model and the Python implementation has drifted.
3. **sympy is the default for anything that must reach the training pipeline**, because it
   shares a process with it: symbolic derivation, `lambdify`, and comparison against autograd
   can be one script, whereas crossing to MATLAB requires a file handoff.
4. **For simplification-heavy work, prefer MATLAB.** Measured below: the entire measured
   advantage sits in `simplify`, and that is what the heavy upcoming derivations lean on.
5. **Data-dependent claims never go to a CAS at all.** Whether the point set excites all ten
   combinations, the singular spectrum, conditioning: these are properties of the data and
   are numerical by necessity. The general split is **structural claims symbolic,
   data-dependent claims numerical**.
6. **Record any case where one engine chokes and the other does not**, in this file. The rule
   is revised by measurement, not by argument.

`kamtin-fp-model/` is read-only. Everything here derives from it and writes nothing to it.

## 2. The measurement that justifies rule 4

Identical task list, both engines, timings exclude engine startup. Tasks are the ingredients
of the LFR rational rewrite of `M(Y)^-1`, chosen to be useful rather than synthetic and
graded from what has already been done to the heaviest thing coming.

| Task | sympy 1.14.0 | MATLAB R2025a | ratio |
|-|-|-|-|
| T1 `det(M(Y))` expanded | 0.018 s | 0.131 s | sympy 7x faster |
| T2 `adj(M(Y))` expanded | 0.020 s | 0.029 s | comparable |
| T3 `simplify(M(Y)^-1)` | 1.798 s | 0.799 s | **MATLAB 2.3x** |
| T4 `simplify(M^-1 X)`, `X` in `K,C,M1,M2` | 4.497 s | 1.580 s | **MATLAB 2.8x** |
| T5 degree of `det(M(Y))` in `Y` | 0.136 s | 0.083 s | comparable |
| **total** | **6.469 s** | **2.622 s** | **MATLAB 2.5x** |

**What this shows, stated narrowly.** MATLAB is about 2.5x faster overall on this workload,
and **the entire advantage is `simplify`**. On structural operations that do not simplify
(expand, determinant, adjugate, polynomial degree) the two are comparable and sympy is
actually faster on T1.

**What it does not show.** At these absolute times, seconds either way, the difference is not
decisive on today's tasks and neither engine was stressed. The reason it still sets the rule
is the mechanism: simplification cost grows superlinearly with expression size, so a 2.5x gap
on a 3x3 problem is a reasonable predictor of a much larger gap on the 6x6 `A` matrix or the
Eq. 21 derivation, which involve products and inverses of substantially bigger expressions.

**A practical counterweight to record.** MATLAB startup under `matlab -batch` is roughly 30 to
40 seconds of wall clock, which is not in the table. For a small one-off check, sympy
round-trips faster end to end even though it is slower at the algebra. Rule 4 is about the
heavy derivations, not about every check.

## 3. Side result, and it is worth more than the timings

Both engines independently return: **`det(M(Y))` has degree exactly 2 in `Y`, and the entries
of `adj(M(Y))` have degrees in `{0, 1, 2}`.**

That confirms the LFR rational rewrite closes at second order, which is precisely the shape
`build_poly_constants` assumes when it returns `N0, N1, N2` and forms `M0inv = N0/d0`. The
assumption was implicit in the implementation and is now checked symbolically against the
ground truth as well as against the Python.

## 4. Reproduce

```
conda run -n GraduationProject python scripts/gantry/orthogonality/code/cas_benchmark.py
"/c/Program Files/MATLAB/R2025a/bin/matlab.exe" -batch "cd('scripts/gantry/orthogonality/code'); cas_benchmark"
```

MATLAB R2025a and R2021b are both installed at `C:\Program Files\MATLAB\`; the Symbolic Math
Toolbox is present in R2025a. **A caveat on rule 1: it depends on a local MATLAB licence.** If
this work moves to a machine or cluster without one, rule 1 degrades to sympy only, and the
provenance check of rule 2 is lost rather than merely slower. Note that in any handoff.

## 4b. Recorded engine failures (clause 6)

2026-09-06, finite-horizon separation check: MATLAB rejected `syms beta positive` inside a
function because `beta` is also a MATLAB function name. SymPy accepted its corresponding
symbol. Replaced the MATLAB declaration with `beta_reg = sym('beta_reg','positive')`.
This was a declaration failure, not an algebra disagreement. Sandboxed MATLAB also failed
during startup with a filesystem inconsistency; the approved execution outside the sandbox
started successfully. After the declaration fix, both scripts completed and agreed on all
18 shared symbolic checks (structural verification examples; results linked from `gaps.md`
Section 9.7).

2026-09-07, nonlinear closed-loop orthogonality verification
([report](proofs/nonlinear-closed-loop-orthogonality.md)). Three failures, all engine
behaviour rather than algebra disagreement; the two engines never disagreed on a result
(60 shared checks, all true in both, no undecided identities).

| Date | Engine | Failure | Fix |
|-|-|-|-|
| 2026-09-07 | MATLAB R2025a `syms` | "Attempt to add `kap` to a static workspace". A function containing **nested functions** has a static workspace, so `syms` cannot inject variables into it. SymPy has no equivalent restriction | Declare every symbol explicitly with `sym('name','real')`. Same class of failure as the 2026-09-06 `beta` incident: `syms` is convenient at the prompt and fragile inside functions |
| 2026-09-07 | Both | Expression swell. The sensitivity recurrence built deeply nested **unexpanded** products of the growing state; downstream `expand`/`simplify` then ran over 15 minutes without finishing in either engine | `expand` the state AND the sensitivity matrix at every step of the recurrence, plus a syntactic-zero fast path (a zero expanded polynomial is already a proof) before any decision-procedure call |
| 2026-09-07 | **SymPy 1.14.0** | `OverflowError: 'mpz' too large to convert to float` inside `sympy.ntheory.factor_._perfect_power`: sympy tried to factor a huge integer under a radical. Triggered twice, when expanding a degree-16 expression containing an orthonormal basis's surds, and when calling `pinv()` on a surd matrix. **MATLAB R2025a handled both without complaint.** This is the first recorded case of an engine failing outright where the other did not | Work in the basis-free **rational** projector residual `rho = P_S L d` instead of `Q_0' L d` (proved equivalent symbolically), build the chunk-cancellation example in the rational chunk operators, and use small **integer** witness points instead of fractions, whose powers produce astronomically large numerators |
| 2026-09-04 | MATLAB R2025a `solve` | Returned the **trivial** branch for an underdetermined homogeneous linear system: "solved 8 of 8 entries" for 4 equations in 8 unknowns, giving `L = M = bf = 0`, and `K = 0` likewise. Two rows of `affine_leakage` came out wrong and disagreed with sympy. | Both engines moved off `solve` to an explicit null-space parameterisation (`null` / `Matrix.nullspace`), which has no branch to pick, plus an assertion that the constraint holds identically after substitution. Full agreement after. |

2026-09-08, revised-ownership reference cases
([report](proofs/trajectory-orthogonality-verification-2026-09-08.md)). Two failures, both SymPy,
both located because they failed loudly rather than quietly.

| Date | Engine | Failure | Fix |
|-|-|-|-|
| 2026-09-08 | **SymPy 1.14.0** `Matrix.rank()` | Symbolic rank of a 9x3 matrix of degree-6 polynomials in twelve symbols did not terminate in over ten minutes; two runs were killed before the cause was located. MATLAB was never asked for a symbolic rank here, so this is not an engine disagreement | Evaluate the rank numerically at an integer witness instead, and **say that the claim changed**. Corrected 2026-09-08: the first version of this row argued that "a rank is a property of a point, not an identity", which is too broad. Generic rank is a perfectly good symbolic notion, and so is a structural rank BOUND; what a witness evaluation gives is a lower bound at one point, which is weaker. Substituting the cheap check is legitimate; presenting it as the same statement is not |
| 2026-09-08 | SymPy | `expand(Matrix) == 0` is `False` for every matrix, zero or not, so a zero test written for scalars reports every MATRIX identity as a failure | Decide matrices ENTRY BY ENTRY. Caught only because it failed all of them at once; a helper that had failed one would have looked like a real result |

**A note on rule 4 that this run does not support.** MATLAB finished in `37.0 s` against SymPy's
`199.9 s`, roughly 5.4x. That sits between the 2.5x of Section 2 and the 7x of the 2026-09-07
run, and both engines returned every shared result true with no undecided identities, so the
margin is a timing observation and not a correctness argument.

**The general lesson, and it applies to every proof in this folder.** For a homogeneous
constraint, never ask a CAS to *solve* it. Ask for its **null space** and parameterise. `solve`
is free to return any point of the solution set, and for a homogeneous system the origin is
always in it, so a trivial answer is a legal answer and is silently indistinguishable from a
real one. sympy did not fall into it here, but nothing prevents it from doing so.

Note also what the dual-engine rule bought: sympy alone would have shipped the right numbers
with no evidence they were right, and MATLAB alone would have shipped two wrong ones. The
disagreement, not either result, is what located the bug.

## 5. Revision log

| Date | Change | Evidence |
|-|-|-|
| 2026-09-03 | Rule set | The Section 2 benchmark, and the dual proof in `nullspace-check.md` Sect. 6b |
| 2026-09-04 | Clause 6 entry added; homogeneous constraints go via null space, never `solve` | The `affine_leakage` disagreement in Sect. 4b |
| 2026-09-07 | Three clause 6 entries added; rule 4's margin re-measured at ~7x on substitution-heavy work, not just `simplify`; new working rule below | The nonlinear closed-loop verification, Sect. 4b |

## 6. Working rule added 2026-09-07: prefer the projector to the orthonormal basis

Measured, not argued. An orthonormal basis `Q` is the right object for *stating* projection
geometry and the wrong object for *computing* with it. Normalisation introduces square roots,
which block exact rational arithmetic, and in SymPy they reach a code path that overflows
outright. Where a projector suffices, use the projector: `P_S = Sbar Sbar^+` stays rational,
carries the same penalty value (`||Q'v||^2 = v' P_S v`), and splits over row chunks the same
way. Keep `Q` only for the statements that are genuinely about an orthonormal basis, where
the matrices are small and constant.

Second measurement from the same run, on rule 4's margin. MATLAB finished the whole workload
in `39.2 s` against SymPy's `270.7 s`, roughly **7x**, engine startup excluded from both. The
2.5x in Section 2 was measured on `simplify` alone; this workload is dominated by substitution
and expansion, so the gap is not confined to simplification as Section 2's wording suggested.

Third, a pure performance note for SymPy: `xreplace` is dramatically faster than `subs` for a
fully numeric substitution. Swapping one for the other took the three slowest checks in that
run from `132 s`, `326 s` and `482 s` to `75 s`, `9 s` and `43 s`.
