# Literature Verdict: is there a REAL method satisfying all four augmentation requirements?

**Date**: 2026-07-11. **Question (user, this thread)**: find a real, literature-grounded method that
satisfies the four requirements for the X/Y free-integrator augmentation. **Answer (grounded, exhaustive)**:
NO single published method satisfies all four; every real method satisfies at most 3 of 4, and there is a
structural reason. The closest real, buildable candidate is Tustin-Net (integrator factoring), which
satisfies 1-3 with a position loss and no class restriction. This doc is the consolidated verdict with the
requirement table and per-method provenance. Companions: `docs/dissipativity-limits.md`,
`docs/data-silent-regularization-limits.md`, `docs/drift-diagnosis-status.md` (§5, §5m).

## The four requirements (precise)
1. **Knowledge-free** — the guarantee holds without knowing the true residual/dynamics (real machine unknown).
2. **Friction-permitting / full expressivity** — can represent ANY dissipative state-dependent residual
   (Coulomb/static friction, cogging); no restriction on the class of learnable dynamics.
3. **Marginal-preserving** — keeps the zero-stiffness free-integrator pole at the origin; must NOT damp X/Y.
4. **Non-drifting (STRUCTURAL)** — a for-all-weights guarantee that the free-run position stays bounded.
   (If req 4 is read as EMPIRICAL only, the picture changes; see the verdict.)
5. **Scheduling-integrity (the Y conflict)** — Y is SIMULTANEOUSLY a K=0 free-integrator (drifts) AND the
   LPV scheduling variable (`M(Y)`). A method must not corrupt the Y-scheduling dependence: not train out
   `M(Y)` when robustifying against Y-drift, and not achieve no-drift on Y via a restoring action that damps
   the Y pole (violates R3). Y is the HARDEST axis: on X drift-fixes are clean; on Y, drift-robustness and
   scheduling-dependence CONFLICT. **CONFIRMED (code-read 2026-07-11): the pipeline uses LPV SELF-SCHEDULING
   — `M(Y)` reads the PROPAGATED, drifting `x[2]` (`model.py` Y_op=None; `blocks.py:659`), so the conflict is
   real/active, AND Y-drift DETUNES `M(Y)` (a scheduling feedback X lacks -> Y strictly worse than X).**
   Per-method Y outcome must be MEASURED (held-out Y / `M(Y)` dependence + Y-pole eigen-check), not assumed.
   (Detail: `drift-diagnosis-status.md` §5 criterion 5; `gns-encoder-diagnostic-plan.md`.)

## The exhaustive requirement table
Coverage: primary-read (PR) or search-verified (SV) this thread. Each row cites its detailed treatment.

| Method | 1 KF | 2 Expressive/friction | 3 Marginal | 4 Non-drift (structural) | Source |
|---|---|---|---|---|---|
| Pure dissipativity `int F.v<=0` | yes | **NO** (forbids storage/absorber) | yes | (n/a) | dissipativity-limits B1 (PR) |
| Passivity `int F.v<=V` | yes | partial | yes | **NO** (bounds velocity, O(sqrt T) position) | dissipativity-limits A1 (PR §5j/G10) |
| Net-impulse / bounded-integral | yes | **NO** (forbids dissipative friction) | yes | yes | dissipative-block-spec; limits B2 |
| Contraction (RENs, `P>0`,`||A||<1`) | yes | yes | **NO** (damps the pole) | yes | passivity-lit G2 (PR 2104.05942) |
| Stable-by-design LPV NN-SS (Schur) | yes | yes | **NO** (Schur = strictly stable) | yes | SV arXiv:2510.24757 |
| Negative-Imaginary (NI) | yes | yes | yes | **NO** (closed-loop / partner only) | dissipativity-limits A5; passivity-lit G8 |
| Cyclo / EID / shifted passivity | yes | yes | yes | **NO** ("only instability results") | dissipativity-limits A2-A3 (PR H1/H2) |
| Momentum-conservation (Dynami-CAL) | yes | partial | partial | **NO** (external channel unconstrained) | drift-diagnosis §5m (PR 2501.07373) |
| **Tustin-Net (integrator factoring)** | **yes** | **yes** | **yes** | **NO** (DC in accel still drifts) | THIS doc (PR 1911.01310) |
| Latent-force GP (Rogers-Friis) | yes | yes | yes | **NO** (soft prior, conditional) | validation-design §3 (PR 2109.10681) |
| Data-silent regularization (ours) | yes | yes | yes | **NO** (soft, training-conditional) | data-silent-*-limits (synthesis) |
| Marginally-stable ID (Ghai) | yes | **NO** (linear only) | yes | yes (bounded, linear) | THIS doc (PR ghai20a PMLR 2020) |
| ISS dissipative residual (DiLaR-PINN) | yes | yes | **NO** (needs ISS baseline) | (excluded) | passivity-lit G1 (PR 2604.18277) |

**Every real method has at least one NO.** The ones that get req 4 as a STRUCTURAL guarantee (net-impulse,
contraction/Schur, marginally-stable ID) all fail req 2 or req 3. The ones that keep req 1+2+3 (Tustin-Net,
latent-force GP, data-silent) fail req 4 as a structural guarantee (only empirical/conditional).

## The two new primary reads (this thread)

### Tustin-Net -- the best real partial (satisfies 1, 2, 3; keeps a POSITION loss)
**Reference [PRIMARY-READ]:** D. Masov/Mavkov, M. Forgione, D. Piga, "Tustin neural networks: a class of
recurrent nets for adaptive MPC of mechanical systems", arXiv:1911.01310; applied study "Accounts of using
the Tustin-Net architecture on a rotary inverted pendulum", arXiv:2408.12266.
- **Architecture (their Eqs 2-3):** the position update is the FIXED exact trapezoidal integrator
  `x(k+1) = x(k) + Ts*(xdot(k+1)+xdot(k))/2`; the velocity update `xdot(k+1) = xdot(k) + Ts*f(x,xdot,u)` has
  the acceleration `f` as a FREE feedforward NN; output `y(k)=x(k)`.
- **Against the requirements:** 1 knowledge-free (integrator = known physics, `f` unconstrained); 2 full
  expressivity (`f` any acceleration incl. friction); 3 marginal-preserving (integrator kept EXACTLY, no
  stiffness/damping on the pole); **4 NO** -- a DC in the learned acceleration `f` still integrates to drift.
- **Why 4 fails / not demonstrated:** their long-horizon accuracy is shown on a double pendulum (restoring
  gravity, stable/unstable equilibria), NOT a free integrator. The architecture removes the encoder-
  reconstructability pathology (our d3) and keeps a POSITION loss (so it is NOT the forbidden velocity-domain
  loss; it is the §5 SECONDARY, un-gated), but it does not structurally forbid a DC in `f`.
- **Verdict:** the strongest real, buildable, class-preserving candidate. Reduces the whole problem to "the DC
  of the learned acceleration". Req 4 on top must be empirical (conditioning).

### Marginally-stable ID -- real bounded guarantee, but LINEAR
**Reference [PRIMARY-READ]:** U. Ghai et al., "No-Regret Prediction in Marginally Stable Systems", PMLR v125
(2020). Also SV: "Identification of Linear Marginally Stable Dynamical Systems Using a Single Rollout"
(IEEE 2025, ARX + Cayley-Hamilton, bounded-noise regression).
- **What it gives (abstract + intro):** least squares has SUBLINEAR (polylog in the stochastic case) regret
  for prediction in a MARGINALLY STABLE LINEAR system whose "state can grow polynomially with time", via a
  "structural volume-doubling lemma showing x_t to be a small linear combination of past states".
- **Against the requirements:** bounded ESTIMATION/prediction for the marginal mode (req 3+4) WITHOUT
  restricting to stable -- but it is LINEAR prediction, not learning a nonlinear friction residual, so it
  FAILS req 2. Confirms marginal systems CAN be identified with bounded guarantees; not a nonlinear-
  augmentation solution.

## Verdict
1. **No single literature method satisfies all four.** This is now grounded by exhaustive primary/search
   coverage, not asserted. It is a DEFENSIBLE negative result (a real thesis finding: the gap is genuine).
2. **Structural reason:** a for-all-weights structural no-drift guarantee (req 4) and full expressivity
   (req 2) are logically incompatible (a universal approximator can represent a drifting force). So any
   method with structural req 4 must restrict the class (fail req 2 or 3), and any fully-expressive method
   gives req 4 only empirically. The table is this dichotomy made concrete.
3. **Best real partial = Tustin-Net (integrator factoring):** satisfies 1, 2, 3, keeps a POSITION loss, no
   class restriction; reduces the problem to the DC of the learned acceleration.
4. **Recommendation:** Tustin-Net + EMPIRICAL conditioning (multiple shooting + excitation) for the drift,
   and accept that req 4 is EMPIRICAL, not structural. That is the closest the literature gets without
   sacrificing friction (req 2) or the marginal mode (req 3). The alternative is to relax req 2 (accept a
   class-restricting structural constraint) -- a supervisor/user choice, not a technical one.

## The one setting where all four hold (closed-loop)
The impossibility (req 4-structural vs req 2) is specific to the OPEN-LOOP free-run. The real machine runs
CLOSED-LOOP, where the servo provides requirement 4 (bounded position) FOR FREE: a spurious model DC is
counteracted by control action instead of ramping the position (the servo is the interconnection partner of
dissipativity-limits A5, NI free-body). So in closed-loop:
- **1, 2, 3** are given by any fully-expressive augmentation (Tustin-Net, latent-force GP, or an unconstrained
  ANN with the integrator factored) that permits friction and preserves the marginal mode;
- **4** is provided by the LOOP, not by constraining the model.
So **all four are satisfiable in the closed-loop deployment** with a fully-expressive model. This matches how
the machine actually operates, and it DISSOLVES the open-loop impossibility.

**Mandatory caveat (also a STANDING CONSTRAINT in `drift-diagnosis-status.md`):** closed-loop no-drift MASKS
model defects (the servo bounds position for a wrong model too), so it certifies the LOOP, not the MODEL.
Closed-loop no-drift MUST be paired with a FIDELITY metric feedback cannot hide: (a) prediction-error /
short-horizon residual analysis, and/or (b) a control-effort / bias audit (a spurious model DC appears as a
steady-state control-effort offset). Never report closed-loop no-drift as model validation on its own.

**Consequence.** The blocking question is not "which method" but "is open-loop free-run the right acceptance
metric, given closed-loop deployment?" If the deliverable accepts closed-loop (or a drift-tolerant/fidelity
metric), a fully-expressive augmentation is the approach that has all four. This is the open supervisor
question (drift-diagnosis §5m).

## Honest caveats / provenance
- PRIMARY-READ this thread: Tustin-Net (1911.01310), Ghai (PMLR v125 2020), plus the dissipativity family
  (2003.10143, 1709.06986, 2604.18277, 2104.05942, 2112.03339, 1305.1079, 2011.14610, 1907.07420),
  momentum (2501.07373), latent-force (2109.10681), identifiability (Little/Rothenberg PLoS ONE 2010).
- SEARCH-VERIFIED (not primary-read): stable-by-design LPV NN-SS (2510.24757), single-rollout marginal ID
  (IEEE 2025), MPINeuralODE, kernel-ID survey (Pillonetto, Automatica 2014, not on arXiv).
- Req 4 read as STRUCTURAL throughout. If the deliverable accepts EMPIRICAL no-drift (held-out free-run),
  Tustin-Net + conditioning and the data-silent route become admissible; that reading is the productive one
  and is the open supervisor question (open-loop free-run vs closed-loop metric, drift-diagnosis §5m).
- All quotes/equation refs: re-verify character-exact against the PDF before thesis use.
