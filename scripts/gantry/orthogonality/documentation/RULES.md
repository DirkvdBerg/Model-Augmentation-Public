# Hard rules for this folder

Binding on every script, document and reported number under `scripts/gantry/orthogonality/`.
These are constraints, not guidance. A result that violates one is not a result.

| # | Rule | Detail |
|-|-|-|
| 1 | **Extendability** | Below |
| 2 | **Algebra tooling** | [algebra-tooling.md](algebra-tooling.md) |

## Rule 1: Extendability. Nothing that needs the truth is a deliverable.

The thesis has to work on the real gantry, where there is no known ground truth, no `Delta_true`,
no true parameter vector and no hidden-absorber trajectory. **Every deliverable must therefore
be a general object that survives the move to Telica.** A number that can only be computed in
simulation is validation, never a result.

### The test, applied before writing any script

> **Could this be computed on Telica?**

If yes, it is a deliverable. If no, it is validation only, and the real question has not been
asked yet: **what is the general object that this number is an instance of?** Derive that
object first, then use the simulation instance to check it.

### The three categories, and every reported quantity carries its label

| Category | Needs | Extends? | Examples here |
|-|-|-|-|
| **Structural** | the model only, no data | Yes, trivially | `ker(d kappa)`, span invariance under reparameterisation, the `beta sigma_i^2` pricing, the exact final-layer proximal step, the bias formula `(Phi^T Phi)^-1 Phi^T Delta` |
| **Data-computable** | measured data, no truth | Yes | `Phi` and `Q` from reconstructed states, `sigma_min(Phi)`, `\|\|Q^T f_ANN\|\| / \|\|f_ANN\|\|`, `\|\|x_aug\|\|`, split-half subspace stability, `Lambda_phy` by the Bolderman rule (see the note below; this is the PARAMETER-ANCHOR weight, NOT `beta`), `beta` by L-curve, cross-record parameter consistency |
| **Truth-requiring** | the simulation ground truth | **No** | `\|\|Q^T Delta_true\|\| / \|\|Delta_true\|\|`, parameter error against `theta_true`, anything comparing to the 8-state truth |

Category 3 is not banned. It is **demoted**: it may validate a category 1 or 2 object and may
never be quoted as a finding on its own.

### What to do when a computation lands in category 3

Do not compute it and then caveat it. Convert it first:

1. Write the general relation the number is an instance of.
2. Split it into parts that are separately computable without truth.
3. Make the general relation the deliverable, with a bound or an identity.
4. Then run the simulation instance, and use it to check the relation holds and is not vacuous.

### The worked example, which is also the incident that produced this rule

**2026-09-03.** The recovery floor was computed as `\|\|Q^T Delta_true\|\| / \|\|Delta_true\|\| = 0.826`
on the augmentation simulation data and reported as the headline result. It is category 3: it
needs the 8-state truth, so it cannot be computed on Telica and does not extend. Caught by the
user, not by the author.

The category 1 object it is an instance of is the bias relation

```
theta_error = (Phi^T Phi)^-1 Phi^T Delta ,      ||theta_error|| <= ||Delta|| / sigma_min(Phi)
```

whose two parts are separately computable on real data: `sigma_min(Phi)` from the model and the
excitation, and `\|\|Delta\|\|` estimated as data minus baseline simulation. **That relation is the
deliverable.** It states what determines the bias for any unmodelled dynamics rather than what
the bias happens to be for this absorber, and it connects the floor to the conditioning problem
already on record, since `sigma_min` is the amplifier and our relative spectrum reaches
`2.6e-6`.

The 0.826 keeps exactly one role: a validation instance, in simulation, checking that the
predicted `theta_error` matches the observed one and that the bound is not vacuous.

Note the side effect, and it is the point of the rule: the general form is **more** useful than
the instance. It explains rather than reports, it reuses on every dataset, and it supplies a
third independent argument for the reparameterisation, because `sigma_min = 0` on the raw 14
parameters makes the bound vacuous while `kappa` coordinates make it finite.

### Note on `beta`, added 2026-09-07 to correct this table

The row above previously read "`beta` by the Bolderman rule". That conflated two different
regularisers and is wrong.

**What Bolderman gives.** `bolderman2024iss` Eq. (38), verified from the rendered p19:

```
Lambda_phy = [ (1/(eps*n_phy)) (1/N) sum_i (u_i - f_phy(theta*_phy, phi_i))^2 ]^(1/2)
             * diag(theta*_phy)^-1        with eps = 1 in every reported use
```

This weights the **parameter-anchor** term of his Eq. (19),
`||diag(Lambda_NN, Lambda_phy)(theta - [0; theta*_phy])||^2`, which pulls `theta` toward the
physics-only fit. In our code that is `param_loss` / `Lambda`. It is **not** the orthogonality
penalty weight. His Remark 3.1 presents orthogonal projection as an *alternative* regulariser,
so he never pairs the two and gives no rule for `beta`.

**What we actually have for `beta`.** No closed-form rule. Bolderman selects his network
regulariser `Lambda_NN = lambda I` by **L-curve**: 20 log-spaced values over `[1e-18, 1e8]`,
warm-started, selecting `1e-5`, citing Hansen and O'Leary (1993). That procedure is the
candidate for `beta`, it is data-computable and therefore Rule 1 clean, and it is a legitimate
answer to the objection in `gyorok2026obc` p2 that finding the trade-off parameter "may not be
intuitive": a principled selection procedure answers that without requiring a formula.

**Until an L-curve is actually run, `beta` is a chosen constant with no justification**, and any
result quoting a particular `beta` must say so.

### Reporting requirement

Every number in a document or a script output states its category. A results table without
category labels is incomplete. Where a category 3 number appears, the category 1 or 2 object it
validates is named in the same place.

## Revision log

| Date | Change | Cause |
|-|-|-|
| 2026-09-03 | Rule 1 set | The recovery-floor incident above |
| 2026-09-03 | Rule 2 set | The CAS benchmark in `algebra-tooling.md` |
| 2026-09-07 | Corrected the `beta` entry: Bolderman Eq. (38) sets the parameter-anchor weight, not the orthogonality weight | Misattribution found while specifying the penalty; `gyorok2026obc` p2 makes `beta` selection the stated reason for abandoning the penalty route |
