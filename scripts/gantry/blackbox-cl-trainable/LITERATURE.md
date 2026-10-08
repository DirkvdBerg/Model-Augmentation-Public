# blackbox-cl-trainable: literature

Deep-research run 2026-09-29, three agents, one sub-question each (agreed with the user). Full
bibliographic data for held items in `docs/references.md`; earlier sweep on SQ1:
`docs/closed-loop-objective-literature-2026-08-28.md`, `docs/closed-loop-training-findings-2026-08-28.md`.

## SQ1 SUBNET / neural state-space trained with a known controller in the rollout
- No published work trains a SUBNET-type model through a fixed controller in the rollout and
  reports or solves training-time destabilisation. PROVISIONAL: 78 forward citers of two lineages
  enumerated (2026-08-28) + 4 arXiv abstract queries today; Google Scholar never worked for this
  question; model-based RL vocabulary not enumerated.
- Mechanism is known: Ribeiro, Tiels, Aguirre, Schoukens, Automatica 2020 (read): a system held
  near instability by a linear controller gives a simulation-error cost full of local minima;
  cost and gradient Lipschitz constants grow with horizon. Matches our G4 fragility.
- Boroujeni et al. CDC 2025 (read): our rollout is their "Strategy 2", drawback D.1 (diverges when
  the model is not stable under K); their fix is architectural (ICI, SQ2), not an optimiser trick.
- Abbas, Ali, Werner, CDC 2010, DOI 10.1109/CDC.2010.5717855: known controller as fixed weights
  inside a trained recurrent net (LPV). Abstract only (?) (`needs-browser-route`).
- Beintema, Schoukens, Toth, Automatica 2023; Beintema PhD 2024; deepSI: open loop only.

## SQ2 models stabilised by a given controller
- Boroujeni, Meroi, Massai, Galimberti, Ferrari-Trecate, "Neural Identification of
  Feedback-Stabilized Nonlinear Systems", IEEE CDC 2025, DOI 10.1109/CDC57313.2025.11312335,
  arXiv 2503.22601 (read in full):
  - model: y = S(u_hat - K(y)), S strictly causal and L_p-stable (a REN in their examples);
  - Theorem 1: every such S gives a model K stabilises, and every G that K stabilises has one,
    S = G(I - KG)^-1, G = S(I + KS)^-1; needs K only, no nominal plant, no coprime factors;
  - hypothesis: K itself incrementally L_p-stable. **K1 has an integrator, so not covered as
    published.**
  - examples: REN, nx 8, horizon 100, Gaussian excitation; controller transfer (our K2) not shown.
- Linear dual Youla (Sugie and Maruta, IFAC 2020; Van den Hof and Schrama, Automatica 1995;
  Landau and Vau, ECC/EJC 2022): handles integrators through doubly coprime factors, but needs a
  nominal plant stabilised by K; no nonlinear extension keeps that (De Bruyne, Anderson, Linard,
  IDC 1999 unread (?)).
- Furieri, Galimberti, Ferrari-Trecate, IEEE OJCSYS 2024, DOI 10.1109/OJCSYS.2024.3441768:
  nonlinear Youla with RENs, ancestor of ICI; not read (?).
- Chen, Srivastava, Yin, Smith, EJC 2024, dual-IOP without coprime factors: whether it needs a
  stable K not resolved (?).
- Our reading (not in the literature, to verify): write K1 = K_s I_int with K_s stable and I_int
  the integrator; the loop G K_s I_int is the loop of G' = I_int G with the stable controller
  K_s, so ICI applies to G', and G = (1 - z^-1) G' needs no plant information.

## SQ3 stable-by-construction parameterisations
- RENs (Revay, Wang, Manchester, IEEE TAC 2024): contraction for every parameter; sufficient, not
  necessary; biases lightly damped modes towards extra damping.
- Hoekstra, Gyorok et al., constraint-free well-posedness and stability (held, read): contracting
  variant clearly less accurate (Cascaded Tanks RMSE 0.64 vs 0.25 well-posed-only, 0.31 CT-SUBNET),
  2.3x slower; explicitly avoids the SUBNET encoder (free initial state per sequence).
- Kon et al., IEEE TAC 2026 (held, read): IO models; exact in LTI, Lyapunov-conservative in LPV.
- LRU (Orvieto et al., ICML 2023, arXiv 2303.06349, read incl. Appendix A code): diagonal complex
  recurrence Lambda = exp(-exp(nu_log) + i exp(theta_log)), |Lambda| < 1 for every parameter,
  exact (no LMI); ring init r in [r_min, r_max] places poles near the unit circle without plant
  data; cheap (parallel scan).
- S4D and "Spectral Bias of Diagonal SSMs" (arXiv 2508.20441): frequency-targeted init from the
  sample rate alone (non-aliasing bound |Delta Im(lambda)| < pi).
- None of these has been used with an encoder or in closed loop (0 hits, moderate confidence).
- Note for the design: a stable LRU recurrence with a static nonlinear readout, or a cascade of
  such blocks, is L_p-stable and strictly causal if the direct term is dropped, i.e. a valid S
  for ICI; a nonlinearity inside the state feedback would lose the guarantee.

## What this means for the build
- Remedy class supported: ICI (Boroujeni) with S a stable-by-construction LRU-type SUBNET; the
  integrator of K1 moved to the plant side (our reading, to verify in Phase B/C).
- New relative to the literature: SUBNET encoder with a stable S, closed-loop training with a
  marginally stable controller, a real motion system. Rejected: REN/contraction as S (damps the
  264 Hz mode, accuracy cost measured by Hoekstra/Gyorok).

## Access status
TU/e browser access: not preflighted (fan-out rule); `needs-browser-route`: Abbas 2010,
De Bruyne 1999, Anderson 1998. Everything else read from local holdings or arXiv.

## Research Log (merged)
- Local holdings answered most of SQ1 and SQ2 (earlier sweeps in `docs/`); new this run: the
  ICI Theorem 1 hypothesis on K, the LRU ring-init recipe, the Hoekstra/Gyorok accuracy cost.
- APIs: SQ1 4 arXiv; SQ2 1 OpenAlex; SQ3 3 arXiv (genuine empty feeds), 2 OpenAlex; dblp blocked
  once (shared IP); Scholar not run.
- Gaps: Scholar pass for SQ1; model-based RL vocabulary; four 2024 to 2025 REN citers title-only
  (SIMBa, stable subspace ID, BiLipREN, REN reduction); Furieri 2024, Chen 2024 full text.
- Suggested skill fixes (not applied, for the user): delta-first rule when a dated local sweep
  already covers the sub-question; separate arXiv "well-formed empty feed" (genuine zero) from the
  0-byte transport failure.

## Second run (2026-09-29): keeping Jan's plain black box
Agreed sub-questions after the ICI design failed at B3 (BB2-012) and the user asked for Jan's
construction without added structure.

### SQ4 model-based RL: learned models rolled out under a controller
- The field sidesteps the problem: one-step models (MBPO, Janner et al. NeurIPS 2019,
  arXiv 1906.08253) or short fixed rollouts of 5 to 15 steps with gradient clipping (Dreamer,
  TD-MPC, held). No fixed high-gain controller unstable at init anywhere.
- Growing-horizon curriculum beaten by randomised horizons in one world-model paper (Bardhan
  et al., arXiv 2603.25685, 2026; abstract only).
- Multi-step beats one-step training under model mismatch, linear systems (CDC 2025,
  DOI 10.1109/CDC57313.2025.11312354; abstract only).

### SQ5 plant from a learned closed-loop map (the E5 question)
- A closed-loop fit is controller-specific (Karimi and Landau, SCL 1998, held).
- Indirect method recovers the plant with the known controller: G = S (I - K S)^-1
  (Forssell and Ljung, Automatica 1999, held); nonlinear two-step version Linard, Anderson,
  De Bruyne (IEEE, ~1998, abstract only, `needs-browser-route`); RNN indirect example Sharma
  et al., Arab. J. Sci. Eng., DOI 10.1007/s13369-023-08356-w (abstract only).
- Our reading: the inversion needs S learned from the plant-input exogenous signal f + K1 r,
  which grows without bound under K1's integrator for nonzero references, i.e. the same
  integrator obstacle as ICI. So a plain closed-loop map gives no plant without added structure.
- No neural example with a controller-transfer test found (arXiv zeros, provisional).

### SQ6 simple remedies that keep Jan's model unconstrained
- Windowed simulation error is already SUBNET's stabiliser (Beintema et al. L4DC 2021, held,
  Sec. 2.3, citing Ribeiro et al. 2020).
- Unconstrained multiple shooting for unstable initialisation: short free-x0 subrecords, their
  number reduced during training (Decuyper, Tiels, Noel, Schoukens, IFAC-PapersOnLine 2020,
  held, read): pure schedule change, the encoder already gives each window its x0.
- Curriculum plus multiple shooting most robust over 100 random inits; plain curriculum or plain
  shooting sometimes diverged (arXiv 2608.05777, 2026, held, read).
- Generalized teacher forcing bounds gradients without constraining f (Hess et al. ICML 2023,
  held, read); aimed at chaotic divergence, transfer to a controller loop unverified.
- Lead: Ger and Barak, "Learning dynamics of RNNs in closed-loop environments", NeurIPS 2025
  (abstract only) (?).

### What this means
- Simplest literature-backed option that keeps a plant model (and E5): Jan's SUBNET unchanged,
  trained in closed loop with a window curriculum, short windows first, grown to nf = 0.1 s
  (Decuyper 2020, curriculum 2026): a schedule change only.
- Fallback without E5: Jan's SUBNET on the closed-loop map (r, f) -> y.
