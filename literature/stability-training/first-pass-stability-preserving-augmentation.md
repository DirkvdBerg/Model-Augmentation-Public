# First-pass literature scan: stability-preserving augmentation (2026-07-18)

Web first-pass (WebSearch + WebFetch, Claude) ahead of the full deep-research run. Reframing trigger:
the v5 DC-null counterfactual showed the dominant long-horizon drift is the LEARNED augmentation
DESTABILIZING the free-run on the marginally-stable axis (ANN makes Y ~50x worse than the physics
baseline over 2 s), NOT a constant DC. So the clean fix is stability-preserving augmentation BY
CONSTRUCTION, not post-hoc DC pins / longer windows / SGD. This doc = what already exists; the fix
choice belongs in `docs/decisions.md`.

## KEY DISTINCTION (user, 2026-07-18): well-posedness =/= stability; we HAVE the former
Drenth, Hoekstra, Schoukens, Tóth 2025 ("Efficient Learning of Affine and Rational Dependency LPV
Models with LFR", `literature/lpv-lfr/drenth2025_lpv-lfr-rational.pdf`) already gives our LPV-LFR
structure WELL-POSEDNESS by construction: Def 1 (well-posed <=> det(I - D_zw·Δ(p)) != 0), Thm 6
(guaranteed if Δ diagonal + ‖p‖∞<=1 + `ρ(D_zw)<1`), via the direct parameterization `D_zw = e^{-N}`,
`N = Ψ(DₐᵀDₐ + D_B - D_Bᵀ + εI) ≻ 0` (unconstrained; the SAME Revay-based parameterization Györök
uses). **But well-posedness only makes the LFR loop SOLVABLE (z computable); it does NOT bound the
state trajectory.** The v5 divergence is a STABILITY failure -- a condition on {A, B_w, C_z}, not on
D_zw. So: we already have well-posedness; the MISSING guarantee is STABILITY. The stability-preserving
papers below all ADD stability on top of Drenth's well-posedness.

## Ranked shortlist

### 1. Györök, Drenth, Verhoek, Schoukens, Tóth, Péni (2026) -- CLOSEST TO A DROP-IN
"Data-driven augmentation of first-principles models under constraint-free well-posedness and
stability guarantees." arXiv 2604.11421. **Our exact framework (LFR augmentation), our supervisors,
co-authors already in the project (Drenth, Györök).**
- Guarantee: **contraction (incremental stability) of the FULL augmented interconnection** (physics
  baseline + learned LFR block), enforced by a CONSTRAINT-FREE parameterization of {A, B_w, C_z}
  (sigmoid/exp scaling so `‖A‖₂ + κ‖B_w‖₂‖C_z‖₂ < ᾱ ≤ 1`, Corollary 13 / Eqs. 42-44) + a
  generalized Cayley-transform parameterization of `D_zw` (`‖D_zw‖₂ < 1/L`) for well-posedness (no
  algebraic loops). No penalty, no constrained optimization.
- Baseline assumption: **only local Lipschitz (Assumption 3) -- baseline stability NOT required**;
  a baseline with integrator/z=1 poles is admissible as input.
- Structure: parallel/additive (our case) + multiplicative/hybrid via the LFR.
- Validation: SIMULATION ONLY (F1Tenth, Cascaded Tanks). No real noisy data.
- **GAP for us:** the AUGMENTED model is forced contracting with rate `ᾱ ≤ 1`. `ᾱ < 1` (strict)
  would pull our genuine z=1 integrator inward -> leaky integrator -> physics corruption (same risk
  the v2 optimizer report flagged for spectral reg). The knob is `ᾱ → 1` (MARGINAL contraction),
  which MIGHT preserve the integrator -- unproven/unclear = THE open research question. Also does
  not explicitly treat LPV.

### 2. Moradi, Beintema, Jaensson, Tóth, Schoukens (2025) -- THE REAL-DATA / MARGINAL-MODE ROUTE
"Port-Hamiltonian Neural Networks with Output-Error Noise Models." arXiv 2502.14432. **Also our
group (Beintema = SUBNET).**
- Guarantee: **passivity by construction** (skew-symmetric `J_θ`, PSD dissipation `R_θ` -> dissipation
  inequality -> stable). Passivity admits a LOSSLESS/marginal mode (an integrator is a passive,
  lossless element) -- so it can hold the z=1 mode where STRICT contraction cannot.
- Real noise: output-error (OE) model + the SUBNET subspace-encoder truncated-simulation training we
  already use; validated on real system-ID benchmarks.
- Structure: STANDALONE pH model (not an augmentation); requires casting in pH form (mechanical
  systems are naturally pH).
- Connects to the project's existing passivity scope ([[project_passivity_gap_scope]]).

### 3. Sertbaş & Kumbasar (2025) -- TECHNIQUE REFERENCE ONLY
"Stable-by-Design NN-Based LPV State-Space Models." arXiv 2510.24757.
- Guarantee: Schur parameterization of `A(ρ)` (hard, eigenvalues inside radius γ) -> stable NN-LPV.
  Matches our LPV setting as a TECHNIQUE for parameterizing a stable NN-LPV block.
- Limits: FULLY BLACK-BOX (no physics augmentation); EXCLUDES integrators (`|λ|<1` strict). Real
  benchmark data (two-tank, robot arm, power plant) but vs classical SysID, no closed-loop deploy.

### 4. Revay, Wang, Manchester (2021) -- FOUNDATION
"Recurrent Equilibrium Networks: Flexible Dynamic Models with Guaranteed Stability and Robustness."
arXiv 2104.05942 / IEEE TAC.
- Contraction + IQC (incremental passivity, Lipschitz) BY CONSTRUCTION, parameterized by an
  unconstrained vector in R^N. The general stability-by-construction toolbox behind #1/#2.

### 5. Ghanipoor, Murguia, Mohajerin Esfahani, van de Wouw (2026) -- SDP / ISS ROUTE (TU/e)
"Model updating for nonlinear systems with stability guarantees." Automatica 184 (2026) 112729.
(read from `literature/stability-training/Model updating for nonlinear systems with stability
guarantees.pdf`).
- Structure: **augments a known physics-based model with a black-box CORRECTION term** (exactly our
  parallel augmentation).
- Guarantee: **input-to-state stability (ISS) + set invariance** of the EXTENDED model, obtained via
  two tractable **semidefinite programs (SDPs)** for locally / globally Lipschitz nonlinear models.
  The CONVEX/LMI route (vs Györök's ML-parameterization route).
- Real-data half: a **noise-robust filter** (with an approximated internal model of the uncertainty)
  that asymptotically estimates the uncertainty + state from I/O data; synthesized via SDP with
  robustness to model mismatch, disturbance, noise. Uses an "ultra-local model" for the uncertainty.
- Validation: simulation (vehicle roll-plane model, large dataset).
- GAP for us: ISS bounds the state w.r.t. input, but a PURE INTEGRATOR is NOT ISS -> the guarantee
  likely must be SCOPED to the correction (correction doesn't destabilize) rather than ISS of the
  whole augmented model; check whether set-invariance admits a marginal mode.

### 6. Liu, Tóth, Schoukens (2024) -- WEIGHTED-REGULARIZATION ROUTE (our group)
"Physics-Guided State-Space Model Augmentation Using Weighted Regularized Neural Networks (W-PGNN)."
IFAC (arXiv 2405.10429). (read from the PDF under stability-training).
- Method: augments a prior physics SS model with an SS-NN under a **weighted, data-adaptive
  regularization** penalizing the difference between baseline and identified STATE + OUTPUT functions;
  trusts data where informative, **keeps the augmentation near the physics baseline where data has low
  information content**.
- Relevance: NOT a hard stability guarantee (penalty-based), but directly relevant to our drift -- the
  integrator's long-horizon behavior is a low-information region, so W-PGNN holds the correction near
  the (marginally-stable) baseline there. Also the closest to our INTERPRETABILITY (orthogonal-
  projection) regularizer. Data-adaptive, not a blunt DC pin.

### Secondary (not fetched in depth)
- Stable Port-Hamiltonian Neural Networks, arXiv 2502.02480 (global Lyapunov stability, energy +
  dissipation, sparse data).
- Physics-Guided State-Space Model Augmentation Using Weighted Regularized NNs, IFAC 2024
  (S2405896324013247) -- 403 on fetch; likely same group, weighted regularization for PG-SS
  augmentation; pull later.
- Physics-guided NNs for feedforward control with input-to-state-stability guarantees, arXiv
  2301.08568 (ISS via Lipschitz bounds on a PGNN).

## The reframed conclusion (interpretation -> also to decisions.md)
The field (specifically our own group) already has stability-preserving augmentation: **Györök =
contraction-guaranteed LFR augmentation (near drop-in, our framework)**, **Moradi/pHNN =
passivity-guaranteed + real-noise via OE+SUBNET**. The clean fix is a stability-BY-CONSTRUCTION
parameterization of the learned LFR block, NOT the DC pin. The single genuine research question is
the **marginally-stable integrator modes**: strict contraction (Györök `ᾱ<1`, Schur) corrupts the
true z=1 physics; the resolution is either **marginal contraction `ᾱ→1`** (needs validation) or a
**passivity / lossless-mode carve-out** (pH). This is a cleaner, real-data-transferable story than
post-hoc symptom fixes.

## Sources
- https://arxiv.org/html/2604.11421 (Györök et al. 2026)
- https://arxiv.org/html/2502.14432v1 (Moradi/pHNN-OE 2025)
- https://arxiv.org/abs/2104.05942 (RENs, Revay et al. 2021)
- https://arxiv.org/html/2510.24757v1 (Sertbaş & Kumbasar 2025)
- https://arxiv.org/html/2502.02480v2 (Stable pHNN)
- https://www.sciencedirect.com/science/article/pii/S2405896324013247 (PG-SS weighted-reg augmentation, IFAC 2024)
