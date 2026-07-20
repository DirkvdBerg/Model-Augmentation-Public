# Deep-research prompt: stability-preserving augmentation (transfers to real nonlinear data)

Focused deep-research prompt (run on the web). Reframing trigger: v5 DC-null counterfactual showed
the dominant long-horizon drift is the LEARNED augmentation destabilizing the free-run on the
marginally-stable axis, not a DC. Pre-populated with a first-pass web scan (see
`first-pass-stability-preserving-augmentation.md`) so the run starts warm -- CONFIRM/EXTEND/CHALLENGE
that shortlist rather than start cold.

---

## FRAMING (settled first-pass): well-posedness =/= stability; we HAVE well-posedness
Drenth et al. 2025 (LPV-LFR, `literature/lpv-lfr/drenth2025_lpv-lfr-rational.pdf`) gives our structure
WELL-POSEDNESS by construction (`D_zw = e^{-N}`, ρ(D_zw)<1, Thm 6) -- the LFR loop is solvable. That
does NOT bound the state trajectory. The v5 divergence is a STABILITY failure (condition on
{A,B_w,C_z}, not D_zw). So the missing guarantee is STABILITY, added ON TOP of Drenth's well-posedness.
The central integrator question is which stability route preserves the genuine z=1 mode.

## PRE-POPULATED SHORTLIST (first-pass, 2026-07-18 -- verify + go deeper)
1. **Györök, Drenth, Verhoek, Schoukens, Tóth, Péni (2026), arXiv 2604.11421** -- constraint-free
   well-posedness (Cayley `D_zw`) + CONTRACTION of the full LFR augmentation `{A,B_w,C_z}`; baseline
   need not be stable (Lipschitz only); parallel/additive; SIM-only. GAP: strict contraction `ᾱ<1`
   would corrupt our genuine z=1 integrator; `ᾱ→1` marginal contraction is the unproven knob.
2. **Moradi, Beintema, Jaensson, Tóth, Schoukens (2025), arXiv 2502.14432** -- passivity-by-construction
   pHNN + output-error noise + SUBNET; real benchmarks; STANDALONE (not augmentation). Passivity
   admits the lossless/marginal integrator mode.
3. **Sertbaş & Kumbasar (2025), arXiv 2510.24757** -- Schur-parameterized stable NN-LPV; black-box;
   excludes integrators. Technique reference for stable NN-LPV blocks.
4. **Revay, Wang, Manchester (2021), arXiv 2104.05942** -- RENs; contraction/IQC by construction;
   foundation.
5. **Ghanipoor, Murguia, Mohajerin Esfahani, van de Wouw (2026), Automatica 184:112729** -- augments a
   known physics model with a black-box correction; ISS + set-invariance via two SDPs; noise-robust
   uncertainty-estimation filter (real-data half). CONVEX/LMI route. GAP: a pure integrator is not ISS
   -> scope the guarantee to the correction? check set-invariance for the marginal mode.
6. **Liu, Tóth, Schoukens (2024), arXiv 2405.10429** -- W-PGNN: weighted, data-adaptive regularization
   keeping the correction near the physics baseline where data is uninformative. Penalty-based (no hard
   guarantee) but data-adaptive; closest to the interpretability regularizer; relevant to the drift.

Integrator handling by route: contraction (Györök/REN) needs marginal `ᾱ→1`; ISS (Ghanipoor) may fight
the integrator (scope to correction?); passivity (pHNN) admits the lossless mode natively; W-PGNN soft.

## SHARPENED CENTRAL QUESTION
How to parameterize a learned PARALLEL augmentation of a physics LPV state-space baseline so the
AUGMENTED model is stability-guaranteed BY CONSTRUCTION and transfers to real noisy nonlinear data,
**WITHOUT corrupting the baseline's genuine marginally-stable (z=1 integrator) modes** -- i.e., the
guarantee must admit a lossless/marginal mode, not force strict contraction. Is `ᾱ→1` marginal
contraction (Györök) valid/clean, or is passivity / a lossless-mode carve-out (pH, REN incremental
passivity) the correct tool? What is proven vs open?

---

ROLE
Research assistant with expertise in stability-guaranteed ML for dynamical systems, neural
state-space / system identification, and physics-informed (grey-box) modeling. Ranked, citation-backed
answer. REUSE established methods; do not invent.

CONTEXT (setup)
- Baseline: physics-based LPV state-space model of a high-precision motion system. Some modes are
  pure integrators (positions, no stiffness -> discrete poles at z=1, marginally stable) -- these are
  PHYSICALLY REAL and must be preserved, not damped away.
- Augmentation: parallel neural correction, x[k+1] = f_physics(x[k],u[k]) + w_NN(x[k],u[k]),
  interconnected in an LFR.
- Training: free-run / simulation-error loss over short truncated windows (~400 steps), learned
  encoder for the initial state (SUBNET), truncated BPTT.
- Deployment target: real closed-loop, noisy, nonlinear measured data.

PROBLEM (diagnosed)
The learned augmentation fits the short window but DESTABILIZES the long free-run: on the integrator
axes the augmented simulation diverges ~50x worse than the physics baseline alone over 2 s, while the
short-window loss looks fine. A counterfactual shows the dominant drift is the NN's STATE-DEPENDENT
output on the marginally-stable axis (removing a constant offset does not fix it and raises the loss).
Post-hoc mitigations (zero-mean/DC penalty, longer windows, SGD) are refuted / partial / unprincipled
and we doubt they transfer to real noisy data (a genuine bounded nonzero-mean correction may be
physically required, e.g. Coulomb friction).

QUESTIONS
1. STABILITY BY CONSTRUCTION. Parameterizations making a learned recurrent/state-space augmentation
   contracting / dissipative / Lyapunov-stable by construction (RENs; contracting SSMs;
   port-Hamiltonian/dissipative NNs; Lyapunov-constrained; S4D/S5). Which apply to a PARALLEL
   correction on a baseline with marginally-stable modes?
2. STABILITY-PRESERVING AUGMENTATION. For physics + learned-residual (grey-box): how is the
   augmentation guaranteed to PRESERVE the baseline's stability/passivity (augmented model inherits
   it) rather than free to inject drift? (Györök LFR-contraction; passivity-preserving
   interconnection; dissipative residuals.)
3. INTEGRATOR / MARGINALLY-STABLE MODES [CENTRAL]. How is a learned correction on a genuine
   pure-integrator (z=1) mode constrained so it cannot inject drift, WITHOUT forcing the mode strictly
   inside the unit circle (which corrupts the physics)? Compare: marginal contraction (rate -> 1) vs
   passivity/lossless-mode handling vs a protected/structured integrator parameterization. What is
   proven, what is open?
4. ROLLOUT-STABLE TRAINING (structural). Multiple shooting / SUBNET / scheduled sampling as the
   principled route to long-horizon-stable models; documented limits.
5. TRANSFER TO REAL NOISY DATA. Which methods are validated on real, noisy, nonlinear system-ID
   benchmarks; what breaks going from noiseless simulation to real data (esp. output-error noise
   models, closed-loop data)?

DELIVERABLE
- RANKED shortlist (confirm/extend/challenge the pre-populated one) fitting our setup, each with: the
  guarantee, stability-by-construction vs training-side vs penalty-based, whether it PRESERVES a
  marginal/integrator mode, real-data validation status, and integration cost into a physics-baseline
  + LFR + encoder + BPTT pipeline.
- Resolve the CENTRAL question (marginal-mode preservation) as far as the literature allows; flag
  where our exact configuration is uncovered.
- Cite peer-reviewed sources; separate proven guarantees from empirical results.
