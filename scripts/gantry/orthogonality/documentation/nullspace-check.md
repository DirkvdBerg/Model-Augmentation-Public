# Null-space check: does `ker(Phi)` equal the four structural degeneracies?

**Run** 2026-09-03
**Scripts**
- `../code/claimA_symbolic.py` (sympy, from the Python implementation) -> `claimA_symbolic.json`
- `../code/claimA_symbolic.m` (MATLAB Symbolic Toolbox, from the FP ground truth)
- `../code/nullspace_check.py` (numerical, autograd + SVD) -> `nullspace_check.json`

**Env** `conda run -n GraduationProject`, float64, `RunConfig(use_f64=True, snr=None, up_sample=2)`;
MATLAB R2025a headless via `matlab -batch`.

**Status: claim A is PROVED symbolically, twice and independently. Claim B is numerical by
necessity.** Sections 3 to 6 below record the numerical run, which came first and which is now
corroboration rather than the primary evidence; Section 6b records the proofs.

## 1. The question, and why it was asked

The baseline carries 14 raw physical parameters but only 10 data-identifiable
combinations (D-077, "train raw, trust combinations"). The proposal on the table is to
stop training the raw 14 and parameterise directly in the 10, which would do three things:
make `rank(Phi)` full and therefore satisfy Assumption 1 of the orthogonal-by-construction
paper, remove the four flat directions along which `theta` currently wanders unseen by the
loss (gap P3), and let the `param_loss` anchor be dropped without leaving anything
unrestrained.

None of that is legitimate unless the model output really does depend on `theta` only
through the 10 combinations. If it does not, the raw splits are not a free gauge and the
reparameterisation changes the model. This check settles that, and it is truth-free, so
whatever it concludes carries over to real data.

## 2. Method

`Phi[(j,r), i] = d x_next[r] / d theta_i` at sample `j`, by autograd through
`Parameterized_Gantry_State_Block`, stacked over samples and over the six baseline state
rows. Point set: `simulations/gantry_subnet/diagnostics/orth_projection/step0/point_set.npz`,
671,762 samples, strided evenly so every record contributes.

**The frame matters and is easy to get wrong.** The block differentiates with respect to
`log_params`, so the raw autograd result is `d f / d log theta_j = theta_j * d f / d theta_j`.
In log coordinates the belt-stiffness null direction is `[k_{b2}, -k_{b1}]`, not `[1, -1]`.
Columns are therefore rescaled to physical units, `d f / d theta_j = (d f / d log theta_j) / theta_j`,
which is also the frame `orth_penalty.py` deploys.

## 3. The predicted null space, in closed form

From `Parameterized_Gantry_State_Block._combos_from_raw`, the 10 combinations are
`kb_sum = kb1 + kb2`, `cg1`, `cg2`, `cy`, `cb_sum = cb1 + cb2`, `mh`,
`m_total = m1 + m2 + mb`, `m_diff = m1 - m2`, `J_eff = Jb + Jh + (m1 + m2) Lb^2 / 4`, `d`.

The kernel of the 10x14 Jacobian `d kappa / d theta` is 4-dimensional:

| | direction |
|-|-|
| `v1` | `kb1 - kb2` |
| `v2` | `cb1 - cb2` |
| `v3` | `Jb - Jh` |
| `v4` | `m1 +1, m2 +1, mb -2, Jh -(Lb^2)/2` |

**CORRECTION, 2026-09-03.** An earlier version of this file called `v4` newly named, on the
grounds that the gap analysis says only "and the fourth". **That was wrong.** `v4` was already
derived, named and experimentally confirmed in
`literature/identification/structural-identifiability-gantry.md` (18 May 2026), Sect. 4, as
direction `n_4`. See Sect. 8b. The derivation below is a re-derivation, and it agrees.

Derivation: `m_diff` fixed forces `dm1 = dm2 = a`; `m_total` fixed
then forces `dmb = -2a`; `J_eff` fixed forces `dJb + dJh = -a Lb^2 / 2`. The
`dJb - dJh` freedom in that last equation is `v3`, so the remaining independent direction
takes `dJb = 0` and `dJh = -a Lb^2 / 2`. With `Lb = 0.725` the coefficient is `-0.2628`.

## 4. Two claims, tested separately

They can fail independently, because `Phi = (d f / d kappa)(d kappa / d theta)` gives
`ker(Phi) >= ker(d kappa)` always, with equality only when the data excites all ten
combinations.

**A. Factorization.** Is `span(v1..v4)` inside `ker(Phi)`? A property of the model
structure alone.

**B. Excitation.** Is `rank(Phi) = 10`? A property of the point set.

## 5. Results

**A. CONFIRMED at machine precision.**

| direction | `\|\|Phi v\|\| / \|\|Phi\|\|_2` |
|-|-|
| `kb1 - kb2` | 2.80e-21 |
| `cb1 - cb2` | 0.00 |
| `Jb - Jh` | 3.33e-18 |
| `v4` (mass-inertia) | 2.67e-18 |
| joint | 3.46e-18 |

Principal angles between the numerically computed `ker(Phi)` and the predicted span:
**0.0000 degrees on all four**.

**B. CONFIRMED.** `rank(Phi) = 10` at the deployed `1e-12` relative cut. The spectrum
(60 samples, relative to `s[0]`):

```
s[0] 1.000e+00   s[5] 1.320e-03   s[10] 1.881e-17
s[1] 2.143e-01   s[6] 1.199e-03   s[11] 5.449e-18
s[2] 4.956e-02   s[7] 2.421e-04   s[12] 2.252e-18
s[3] 1.348e-02   s[8] 1.996e-04   s[13] 2.821e-20
s[4] 2.144e-03   s[9] 1.469e-05
```

The gap at the cut is `s[9]/s[10] ~ 8e11`, so **any threshold between `1e-12` and `1e-6`
returns rank 10**. The rank is not threshold-sensitive.

**Conclusion: `ker(Phi) = ker(d kappa)` exactly.**

## 6. Sample-count stability

`N_SAMPLES = 60` is labelled `# HEURISTIC:` in the script; this is the sweep that
justifies it.

| n | rank | `s[9]` | `s[10]` | gap | joint residual |
|-|-|-|-|-|-|
| 10 | 10 | 9.308e-07 | 3.119e-17 | 2.98e10 | 1.02e-17 |
| 20 | 10 | 4.732e-06 | 1.010e-17 | 4.68e11 | 5.60e-18 |
| 60 | 10 | 1.469e-05 | 1.881e-17 | 7.81e11 | 3.46e-18 |
| 150 | 10 | 1.526e-05 | 1.248e-17 | 1.22e12 | 5.25e-18 |
| 400 | 10 | 1.449e-05 | 1.776e-17 | 8.15e11 | 5.34e-18 |

Rank is 10 everywhere and the factorization residual is machine precision everywhere.
`s[9]` is still climbing below n=60 and flat above it, which is why 60 is adequate.

## 6b. Claim A, proved symbolically (the primary evidence)

The numerical run above evaluates the Jacobian **at `theta_0` and at 60 sampled `Y` values**.
The claim has to hold for all `theta`, because `theta` moves during training, and for all `Y`.
A symbolic identity gives that; a machine-precision residual at one operating point does not.
So claim A was redone symbolically.

**What is proved.** Every parameter-dependent matrix of the baseline is annihilated by
`v1..v4` identically in all 14 parameters and in `Lb` and `Y`: `M(Y)`, `C`, `K`, `det(M)` and
`adj(M)`. Everything downstream is a function of these alone, since
`build_G_matrix_entries` forms `M0inv = adj/det` and its products with `K`, `C`, `M1`, `M2`,
and the RK4 discretisation is a fixed composition of the resulting field. By the chain rule
the discrete transition `f_base` is annihilated too, so differentiating through RK4
symbolically is unnecessary rather than skipped.

**The structural reading, which the symbolics then confirm.** `build_poly_constants` defines
`alpha = m1+m2+mb+mh`, `beta = (m1-m2) Lb/2`, `gamma = Jb+Jh+(m1+m2) Lb^2/4`, and `M0/M1/M2`
depend on the raw parameters only through `alpha, beta, gamma, mh, d`. Against the
identifiable combinations: `alpha = m_total + mh`, `beta = m_diff * Lb / 2`, and
`gamma = J_eff` exactly. `K` uses only `kb_sum`; `C` uses `cg1, cg2, cb_sum, cy`. Every one is
a function of `kappa`.

**Two independent proofs, and why both were run.** `claimA_symbolic.py` transcribes from
`model_augmentation/systems/gantry_ss.py`, so alone it proves only that the *Python* model
factors. `claimA_symbolic.m` transcribes from the ground truth,
`kamtin-fp-model/03 Simulink gantry/main.m` lines 51-63, which that folder holds read-only.
Both return **PROVED**, so the factorization holds for the FP model *and* the Python
implementation agrees with it on this property. A drift between the two would have shown up
here.

**Both report `rank(d kappa) = 10`, hence `dim ker = 4`**, and four independent directions are
exhibited, so `v1..v4` are the whole kernel and not a subset of it.

**Negative controls fire in both.** `kb1+kb2` moves `K`; `m1` alone and `Jb+Jh` move `M`,
`det` and `adj`. The tight one: **`v4` with the coefficient `-Lb^2/3` instead of `-Lb^2/2`
moves the matrices**, so the coefficient is exactly right rather than approximately. A test
that cannot fail proves nothing, and these show it can.

## 7. One number that does not match the record, and why

The record states the relative spectrum reaches `2.6e-6`, with `v[10]` carrying belt
stiffness. This run bottoms out at `1.47e-5`. **These are different matrices and the
difference is expected**: this is the 14-column raw `Phi` in physical units, whereas the
deployed object is the extended regressor `Phi_tilde = [Phi | Gamma]` with 15 columns, on
the stride-100 Y-stratified point set, in the normalised frame. The project already
records that `Q` is frame-invariant while `Sigma` and `V` are not
(`improvement-trials/implementation-plan/PROGRESS.md`, "Frame dependence").

So the `2.6e-6` is **neither reproduced nor contradicted here**. It should be re-derived in
the deployed frame before any conditioning argument leans on it.

One consistency check that does hold: rank 10 on the parameter block plus the `Gamma`
column gives 11, matching the recorded "rank 11 of 15".

## 8. What this settles

- **The reparameterisation `theta -> kappa(theta)` is exact.** The matrices depend on
  `theta` only through the 10 combinations, so the raw splits are a free gauge and moving
  to 10 coordinates changes no prediction.
- **Assumption 1 of `gyorok2026obc` becomes satisfiable by construction** in the
  reparameterised coordinates, which is what would let its theorems describe the penalty's
  large-`beta` endpoint.
- **Gap P3's invisible wander disappears** under the reparameterisation, because the flat
  directions cease to exist rather than being restrained.
- Consequently the `Lambda` anchor can be dropped without leaving anything unheld, which
  was previously unsafe.
- The fourth degeneracy is confirmed in closed form. **Not newly named: see Sect. 8b.**

**Read Sect. 8b before using this list.** Two items above are narrower than stated there: the
`Lambda` consequence applies cleanly only to `n_4`, and the reparameterisation is a standard
base-parameter computation rather than a result of ours.

## 8b. Prior art, found after the run, and it changes what this file claims

**The project already held this result.**
`literature/identification/structural-identifiability-gantry.md`, dated 18 May 2026, contains
the same observable map `f: R^14 -> R^10`, the same `rank = 10`, the same 4-dimensional
kernel, all four directions including `n_4 = v_4`, and the same list of ten combinations. It
was missed here because the local-holdings pass covered `literature/Orthogonality/`,
`docs/references.md` and the orth-projection tree, and never `literature/identification/`.
The gap analysis's "and the fourth" is a failure to find that file, not an open question.

**That document also holds two things this run does not.**

*Experimental confirmation.* Its Sect. 5 records a training run from a 10% detune drifting
along `n_4` with `eps = 0.252`: measured `dm_b = -0.504` against the predicted `-2 eps`, and
`dJ_sum = -0.066` against the predicted `-(Lb^2/2) eps = -0.066`. The flat direction was
observed in an actual optimizer trajectory, not only in the algebra.

*A better treatment rule than the one this file first gave.* It distinguishes the four
directions instead of treating them alike:

| Direction | Treatment | Reason given |
|-|-|-|
| `n_1` `kb1-kb2`, `n_2` `cb1-cb2`, `n_3` `Jb-Jh` | **regularize** | the prior encodes a known physical symmetry, the joints being equal by design, so it is physically motivated rather than arbitrary |
| `n_4` mass-inertia | **reparameterize** | no symmetry argument exists; "regularizing `n_4` would make the recovered values depend on the regularization weight, not on the measured data" |

**This supersedes the blanket "reparameterise to the 10" that Sect. 8 below implied.** Only
`n_4` requires the coordinate change on its own merits; `Lambda` may legitimately hold
`n_1` to `n_3` because the equal-joint prior is a physical statement. That matters for the
`Lambda = 0` decision, which is therefore narrower than this file first suggested.

**The reduction is also a classic under another name.** In robot dynamic-parameter
identification the ten combinations are the **base parameter set** or minimal inertial
parameter set, computed symbolically since Gautier and Khalil, *IEEE Trans. Robotics and
Automation* 6(3):368-373, 1990, DOI `10.1109/70.56655`, with a dedicated symbolic tool
(SYMORO+). The 14-to-10 reduction is not a contribution and should be cited, not presented as
machinery of ours. What remains ours is what is done with the kernel, not its existence.

**What no version of this result yet establishes: completeness.** Neither the May document,
nor sympy, nor MATLAB can certify that no eleventh identifiable combination exists.
`StructuralIdentifiability.jl` implements the Ovchinnikov, Pillay, Pogudin and Scanlon
algorithm (*Systems and Control Letters* 157:105030, 2021, arXiv 2004.07774) whose
`find_identifiable_functions` returns generators of the field of identifiable functions and
would settle it. Estimated half a day. Its documented scope limit is that it is restricted to
rational models and explicitly excludes neural-network augmented dynamics, so it validates the
baseline only.

## 9. What it does not settle

**Conditioning is untouched, and rank and conditioning are different problems.**
`s[0]/s[9] ~ 6.8e4` on the identifiable block, so the Gram condition number is around
`5e9`. The argument that a hard projection is risky along directions known to 11 to 38
degrees stands unchanged; removing exact zeros does nothing to the surviving tail.

Nothing here bears on the added-state channel, the point-set measure, or the application
mode. Those are separate decisions.

## 10. Reproduce

```
conda run -n GraduationProject python scripts/gantry/orthogonality/code/claimA_symbolic.py
conda run -n GraduationProject python scripts/gantry/orthogonality/code/nullspace_check.py
"/c/Program Files/MATLAB/R2025a/bin/matlab.exe" -batch "cd('scripts/gantry/orthogonality/code'); claimA_symbolic"
```

## 11. Assurance

| Claim | Level | Basis |
|-|-|-|
| A, factorization | **PROVED** | Symbolic identity in all 14 parameters, `Lb` and `Y`, independently in sympy from the Python implementation and in MATLAB from the FP ground truth. Negative controls fire in both. |
| `dim ker(d kappa) = 4`, and `v1..v4` span it | **PROVED** | `rank(d kappa) = 10` symbolically in both, with four independent directions exhibited. |
| B, excitation, `rank(Phi) = 10` | **MEASURED** | Numerical, at `theta_0`, 60 samples, stable from 10 to 400. Cannot be symbolic: whether the point set excites all ten combinations is a property of the data. |
| Spectrum, conditioning, the `2.6e-6` | **OPEN** | Frame-dependent and not re-derived in the deployed frame. See Section 7. |
| Completeness: no eleventh combination exists | **NOT ESTABLISHED** | Outside what any CAS proof here can show. Needs `StructuralIdentifiability.jl`. See Sect. 8b. |
| Novelty of the 14-to-10 reduction | **NONE, and it never had any** | Held in this repo since 2026-05-18, and a base-parameter classic since 1990. See Sect. 8b. |

**Provenance correction.** An earlier version of this section said the predicted directions
were derived from `blocks.py::_combos_from_raw` "not read off any prior record". The first
half is true; the second was an error of ignorance rather than of transcription, since the
prior record existed and was not found. The general rule this run establishes:
**structural claims go to symbolic algebra, data-dependent claims stay numerical**, and where
a ground-truth implementation exists the symbolic proof is run against it as well as against
ours.
