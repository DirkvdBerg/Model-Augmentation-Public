# Construction Spec: hitting ALL FIVE requirements (the assembly = the contribution)

**Date**: 2026-07-11. **Status**: construction target, unbuilt. **Why**: the literature search concluded no
published method meets all five requirements (`docs/literature-search-conclusion.md`, D-108); the survey
(Sivaranjani et al. 2025) confirms the area is open. So the way to hit all five is to BUILD the assembly and
DEMONSTRATE it — that assembly + the R4/R5 handling is the contribution. This spec is the unambiguous build
target: each requirement -> its mechanism -> how it is validated. Companions: `docs/open-loop-solution-
decision.md` (D-107), `docs/gns-encoder-diagnostic-plan.md`, `docs/data-silent-regularization-concept.md`,
`docs/drift-diagnosis-status.md` §5 (the five requirements).

## 0. The key fact — all five ARE jointly achievable
The impossibility blocks ONLY the pair {structural for-all-weights R4} AND {full R2 expressivity}
SIMULTANEOUSLY. It does NOT block all five. Two routes, each hits all five:
- **Route B (R4 EMPIRICAL):** fully-expressive ANN; R4 demonstrated, not guaranteed. R2 fully preserved.
- **Route A (R4 STRUCTURAL):** bounded-impulse ANN + friction in `f_base` (grey-box); R4 is a for-all-weights
  guarantee; R2 preserved AT THE MODEL LEVEL (friction lives in physics), conditional on the parametric
  friction model being adequate.
**Recommendation: Route B primary** (keeps R2 fully intact = the non-negotiable; R4-empirical is the honest
ceiling the whole search confirmed). **Route A = the comparison arm** (does structural-R4 beat empirical-R4
when friction is in `f_base`? — itself a thesis result).

## 1. Route B (PRIMARY) — requirement -> mechanism -> validation
| Req | Mechanism | Validated by |
|---|---|---|
| **R1 knowledge-free** | conditioning + data-silent regularization act on DATA properties (the unexcited direction), not the unknown residual; orthogonal projection uses the KNOWN FP subspace | by construction; recompute target subspace under different excitation (moves with data) |
| **R2 full expressivity / friction** | ANN unconstrained in what it represents; ALL fixes are SOFT (regularizers / conditioning), NO hard class restriction | injected-friction sim: the ANN LEARNS the friction (held-out free-run + residual-force error in the excited subspace) |
| **R3 marginal-preserving** | ANN writes force to the velocity/accel rows only; no stiffness/damping on the position row; soft fixes add no restoring term | **eigen-check: linearize trained model, X/Y rigid-body \|lambda\| = 1** (not < 1) |
| **R4 non-drifting (EMPIRICAL)** | **Layer 1** long-horizon / free-run conditioning (unroll PAST the ~0.5 s drift onset; state-consistency term, Sertbas-Kumbasar Eq 13, WITHOUT Schur) so the loss SEES the drift + **Layer 2** data-silent projection (= Gyorok orthogonal projection re-aimed at the unexcited DC subspace) pins the drift direction | 12 s free-run position-ENVELOPE ratio ~1.0 (not slope) |
| **R5 scheduling-integrity (Y)** | **Layer 3** schedule `M` off a DE-DRIFTED / EXOGENOUS (measured) Y, breaking the drift->detune->feedback loop (self-scheduling off drifting `x[2]` confirmed, R5) | held-out Y positions / `M(Y)` dependence retained AND Y-pole \|lambda\|=1 |

## 2. Route A (COMPARISON) — structural R4 via grey-box
| Req | Mechanism | Validated by |
|---|---|---|
| R1 | bounded-impulse form holds for ALL weights (data-independent) | by construction |
| R2 | friction in `f_base` (grey-box Coulomb/LuGre) so the ANN residual is zero-DC; R2 at MODEL level, conditional on friction-model adequacy | injected-friction sim: model captures friction via `f_base`; residual offset if model inadequate |
| R3 | ANN force on velocity rows, no position stiffness | eigen-check \|lambda\|=1 |
| R4 (STRUCTURAL) | bounded-impulse ANN: `Sum g = psi(z_N)-psi(z_0)` bounded for all weights -> position bounded (`c>0`) | proof + 12 s envelope ~1 |
| R5 | de-drifted/exogenous Y-scheduling (same as Route B) | held-out Y / `M(Y)` |
**Route A honest limit:** R2 is conditional — if the parametric friction model is inadequate, leftover
dissipative DC is REFUSED (bounded offset), not learned. That is the price of structural R4.

## 3. Build / validate sequence (each step maps to an existing diagnostic)
1. **D-107 clean re-run (Layer 1)** — correct post-D-101 lr; long-horizon conditioning + state-consistency;
   unroll PAST 0.5 s on a curriculum subset. De-confounds Optuna 69399. Tests R2/R4-train (nf-RMS down),
   R4 (envelope), R3 (eigen-check) on X. [`gns-encoder-diagnostic-plan.md` machinery + long-horizon.]
2. **Add Layer 2** — data-silent projection (the orthogonal-projection contribution re-aimed) for the
   residual DC. [`data-silent-regularization-concept.md`.]
3. **Add Layer 3 (Y)** — de-drifted / exogenous scheduling; test R5 (scheduling integrity) + R3 on Y. REQUIRES
   the Y-scheduling decision (exogenous/measured vs self-scheduled — supervisor call; the keystone).
4. **Build the injected-friction sim** — the ONLY way to test R2/friction (current sim has none). Then
   demonstrate ALL FIVE together, Route B vs Route A. [§5g Phase 3-4 / D-D2.]
5. **Real Telica data** — held-out free-run BFR; Y measured -> exogenous scheduling natural; R5 dissolves.

## 4. What "hitting all five" means here (honest)
- It is DEMONSTRATED on the injected-friction sim + diagnostics (H1-H4 + eigen-check R3 + R5 check), not
  proven on paper. That demonstration IS the contribution (no published method does it; survey-confirmed open).
- R4 is EMPIRICAL in Route B (the impossibility ceiling); STRUCTURAL in Route A at the cost of conditional R2.
- Every mechanism is IN-FRAMEWORK (Verhoek LPV-SUBNET + Gyorok projection + rollout-stability conditioning +
  cyclo/EID marginal-storage language) — the novelty is the ASSEMBLY + the R4 (empirical no-drift) and R5
  (scheduling-integrity on a drifting self-scheduling variable) handling.

## 5. The one open decision (supervisor)
**Y-scheduling: exogenous/measured vs self-scheduled.** Self-scheduling (current, `M(Y=x[2])`) makes Y the
hard axis (drift detunes `M`). Exogenous/measured Y (natural on real data) dissolves the R5 conflict and makes
Y as tractable as X. This is the keystone of Layer 3 and a legitimate standard LPV formulation either way.

## 6. First concrete action
**D-107 clean re-run (step 1)** — cheap, de-confounds 69399, tests R2/R3/R4 on X in one shot. Then layer up,
validating each requirement against its diagnostic. Do NOT add friction to the current sim (build the
injected-friction sim as a separate step, §5g Phase 3).
