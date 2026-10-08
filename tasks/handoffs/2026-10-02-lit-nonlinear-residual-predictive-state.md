# Handoff: literature search for a NONLINEAR, theoretically grounded way to give added latent states a predictive target (replacing the linear residual-prediction head)
**From**: session of 2026-10-02 | **Branch**: Augmentation | **Effort suggested**: xhigh (multi-agent literature research with equation-level reading)

## 1. Task
Run the `deep-research` skill (step 0 FRAME mandatory) to find and assess literature for a nonlinear version of "residual-predictive state shaping": training the added latent states of a grey-box neural state-space model so that they become a nonlinear summary of the past that predicts the baseline model's future residual, and then remain state-like (propagatable by the model's own dynamics). Deliver one report, `literature/added-states-training/deep-research-nonlinear-predictive-state.md`, ranking candidate formulations by (a) theoretical grounding for NONLINEAR systems, (b) compatibility with the constraints in section 2, (c) reported evidence of recovering lightly damped or oscillatory latent dynamics. For each candidate give the exact objective (equation, page), what it guarantees (identifiability, consistency, collapse-freedom) and under which assumptions, and its known failure modes. Then answer section 8 in one paragraph.

## 2. Out of scope
- Any code, run or experiment; editing files other than the report (and PDFs saved next to it).
- Remedies that change the model's parameterisation or initialisation (complex / polar / modal / oscillator structures, BLA or residual-based init, pole or band priors): record each in one line under "Excluded" with the reason, do not develop them. The live-ResNet initialisation question (D-233) is handled separately.
- Re-searching what is already covered (section 4, "Already read").

## 3. Where things stand
Branch Augmentation, tree dirty (unrelated files), no runs in flight. Model (Hoekstra-style LFR augmentation, following Jan's published code): physics block + one zero-init tanh MLP writing an additive correction to the physical rows and the next value of n_a = 2 added states from z = [x, x_a, u]; SUBNET-style encoder (linear W^b map + random linear W^a on a 30-sample past window, plus a zero-init NL net); training loss = closed-loop 400-step simulation error with a known controller; data simulated, noisy, closed loop. The missing dynamics is a lightly damped mode (212 Hz, zeta 0.043); plain training never learns it (zero-init dead start).

Local evidence for the current, LINEAR approach (scripts/gantry/absorber-learning-diagnosis/DECISIONS.md section 4): a temporary training-only LINEAR head predicting the model's next 40 output residuals from x_a(k) (trains encoder W^a), then a one-step fit of the ANN's added-state map to the encoder states (encoder detached), then the plain loss, learned an added pair at 208 to 209 Hz, zeta 0 to 0.09, in 2 of 4 local runs. The user rejects the linear head as "monkey patching": it borrows LINEAR subspace-identification theory (CVA / N4SID: state = linear summary of the past predicting the future, readout linear) into a nonlinear framework, whereas SUBNET (Beintema, Schoukens, Toth 2023) is nonlinear throughout and justifies its encoder only by the model's own prediction.

## 4. Established and verified
- Linear-case theory of the current approach (two independent audits, user-provided, not on disk): Phase A = reduced-rank past-to-future regression of the residual; recovers the error system's predictor state up to similarity only for linear, open-loop, exogenous non-periodic input, exact rigid-body baseline, fixed target. In closed loop with the controller simulated, the residual carries closed-loop poles. A NONLINEAR head loses the "state up to a linear transform" identifiability: it can decode a scrambled x_a; only a propagation (dynamics-consistency) requirement keeps x_a state-like. No theorem covers the A->B->C ordering.
- Already read (cite, do not redo): `literature/added-states-training/deep-research-LA-latent-mode-training.md` (Ouala 2020 free latents; Otto and Rowley 2019 multi-step self-prediction; Ni et al. 2024; Hefny, Downey, Gordon 2015 two-stage regression; Azencot 2020; Demirkaya 2026; Kostic 2024), `...-LB-ee-bias-real-poles.md` (Dawson 2016; Hemati 2017; Gilson and Van den Hof 2005; Wu and Noe 2020 VAMP; Tang 2023; Tian 2021), `...-LC-closed-loop-innovation.md` (Forssell and Ljung; Van den Hof and Schrama; Gonzalez 2024/2025; Kon, Heertjes, Oomen 2022 IVNN; Bolderman 2022). Also Beintema, Schoukens, Toth 2023 (arXiv:2210.14816) and Hoekstra et al. 2026 (arXiv:2602.17297, `literature/closed-loop-id/hoekstra2026_lfr-augmentation-fp-models.pdf`).

## 5. Assumed but not verified
- That a nonlinear predictive-state objective exists with guarantees beyond "approximates the reconstructability map".
- That a joint objective (nonlinear residual prediction + one-step or multi-step propagation consistency) is enough to keep x_a state-like without a linear readout.
- That such objectives apply with exogenous inputs and closed-loop data (most predictive-state and Koopman work is autonomous or uncontrolled).

## 6. Tried and failed
None in literature for this sub-question (new). Locally: linear-head approach as in section 3; random live initialisation alone places a near-band pair in 0.42 % of draws (AE1).

## 7. Achieved
See section 3; reports listed in section 4.

## 8. The open question
Is there a nonlinear, literature-grounded objective that makes latent states a predictive (sufficient) statistic of the past for the model's future residual and keeps them propagatable by the model's state map, with stated guarantees and applicable to input-driven, closed-loop, noisy data, that a TU/e system-identification audience (Schoukens, Toth, Hoekstra) would accept as an extension of SUBNET / LFR augmentation? Candidate answers: (i) yes, an established formulation exists (name it); (ii) partial: nonlinear analogues exist with weaker guarantees (which, and what is lost); (iii) no: only linear theory exists, and the nonlinear version is a heuristic.

## 9. Next action
Run deep-research step 0 (FRAME) with these sub-questions, then up to 4 subagents, one per sub-question:
- **N1 Nonlinear subspace / predictive-state learning:** nonlinear and kernel CCA/CVA and kernel subspace identification; predictive state representations with nonlinear features (Hefny, Downey, Gordon line; Boots; spectral learning); VAMPnets (Mardt et al. 2018) and VAMP-type past/future feature learning; deep CCA; contrastive predictive coding. Objective, identifiability, collapse handling, controlled (input-driven) extensions.
- **N2 Joint prediction + dynamics consistency for neural latent state-space models:** latent ODE / deep state-space models, Koopman and linearly recurrent autoencoders with nonlinear readouts, deep Kalman / variational SSMs, identifiable nonlinear latent dynamics (nonlinear ICA for time series, e.g. Hyvarinen, Halva, Yao et al.), multi-step consistency; which make the latent a minimal sufficient statistic / Markov state, and with what guarantees.
- **N3 Nonlinear residual / model-error dynamics in hybrid grey-box models:** learning latent residual dynamics on top of a physics model (hybrid modelling, physics-guided neural networks, model-error modelling with nonlinear latent states, closed-loop settings), including how added states are trained or initialised.
- **N4 SUBNET / encoder lineage and extra states:** what Beintema, Schoukens, Toth, Hoekstra and follow-ups (e.g. arXiv:2602.13108 encoder initialisation; CT-SUBNET arXiv:2204.09405) say about encoders for added states, auxiliary encoder objectives, and identifiability of extra states; whether any of them trains the encoder with a prediction target other than the simulation loss.
Write the report, merge the Research Logs.

## 10. Acceptance criterion
Every candidate in the report has: citation with DOI or arXiv id, objective with equation and page, guarantees with assumptions, applicability to (input-driven, closed loop, noisy, nonlinear, zero-init model unchanged) marked yes / no / unknown, and evidence on oscillatory or lightly damped latents. Section 8 answered with (i), (ii) or (iii) and the deciding sources. Equation-level claims only from full text actually read; abstract-only sources marked as such.

## 11. Read these first
1. `.claude/skills/deep-research/SKILL.md`: the procedure (FRAME first, Research Log).
2. `literature/added-states-training/deep-research-LA-latent-mode-training.md`: closest prior sweep; do not repeat its sources.
3. `scripts/gantry/absorber-learning-diagnosis/CAUSE-AND-DIRECTION.md` sections 1 and 2: the problem and the current linear approach (its tone overclaims; use it for the setup only).
4. Beintema et al. 2023 (arXiv:2210.14816) Secs. 3.1 to 3.2: what SUBNET's encoder is and is justified by.

## 12. Do not
- Develop parameterisation or initialisation remedies (section 2).
- Cite aggregators (Lacuna, Wikipedia, ResearchGate summaries) as sources.
- Present an analogy as a theorem; label each connection as formal equivalence, close analogue, or conceptual analogy.

## 13. Operational
Report and PDFs in `literature/added-states-training/`. Use the scratchpad for temporary files. No conda runs needed.

## 14. Delegation
Up to 4 deep-research subagents in parallel, one per sub-question N1 to N4 (user-requested literature sweep). No code subagents.
