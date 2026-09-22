# Candidate thesis impact by section

These are proposed prompts for later writing sessions, not changes already made to the thesis guide.
Each section begins with what the meeting scan says to retrieve or verify before drafting.

## Cross-section priorities from Quinten's outline feedback

- Organize the evidence around relevant ASMPT motion, tracking and a fixed 200--400 ms post-move settling interval; an unseen multisine phase alone is not application validation.
- Build a validation ladder: interpolation inside the represented range, multisine-free operational motion, controller transfer, and only then operating-range extrapolation.
- Ask Jasper and Dragan to approve the concrete motion set: ETEL/Jasper setpoint-generator profiles, useful ILC references, minimum and maximum moves, long stroke, and representative accelerations.
- Treat a changed controller as a separate closed-loop distribution shift and digital-twin use case, not automatically as plant extrapolation.
- Explain every material modelling, routing, data, metric and hyperparameter choice in one sentence or more, proportional to its effect on the conclusion.
- Keep exact-model parameter recovery as implementation or initialization support, not a main result. Joint-estimation/OBC parameter error remains relevant only where it directly tests the interpretability claim.
- Give the black box enough validation-selected capacity to be a genuine capability comparator.
- Use several neural-initialization seeds for final learned-model comparisons when feasible; distinguish these from multisine phase seeds.
- Select one primary predictive metric. Physical RMS is the default candidate for tracking and settling; add NRMS/BFR only for a distinct supporting purpose.
- Add measurement noise only when its level and injection point are grounded in machine or ILC data. Otherwise state the noiseless scope plainly.

## Abstract

- State the actual final problem: augmenting an interpretable position-dependent gantry model from
  closed-loop data.
- Name only contributions supported by final experiments: exact LPV baseline, dynamic augmentation,
  closed-loop estimation and OBC/joint estimation if the final comparison succeeds.
- Do not mention burn-in, SVD reduction or physical recovery of latent states.
- Verify final quantitative claims against the final result artefacts.

## Introduction

- Use the ASMPT notes for industrial motivation: digital twins, offline controller evaluation,
  tracking/settling and cross-coupling prediction.
- Explain the gap between an interpretable but incomplete FP model and an unconstrained black box.
- State the research question after the method scope stabilized; do not retain the early inverse/
  feedforward question.
- Lead with the controller-evaluation/digital-twin use case: the model should predict relevant motion
  before a changed controller is deployed on the machine.
- Verify any general claim with literature, not meeting notes.
- Figure to reuse or create: one problem-overview diagram from machine to FP baseline, augmentation and
  intended digital-twin use.

## Gantry system and baseline model

- Start from the physical gantry coordinates and map to measured/logical coordinates.
- Explain why payload position `Y` schedules the dynamics and state the admissible operating range.
- Derive the polynomial mass equation first, then the exact rational/LFR form.
- Include the well-posedness/invertibility check and clarify continuous-time model versus RK4
  implementation.
- Exclude the explored six-to-four SVD reduction.
- Figure to reuse or create: physical coordinate schematic and `G-Delta` LFR diagram.

## Model augmentation

- Explain why the augmentation is parallel and dynamic.
- Document the exact route of the final reported run, not a route shown in an older meeting deck.
- State why corrections reach position/velocity-related dynamics and what marginal-mode risk this
  creates; this is a design choice that needs a reason.
- Explain baseline-preserving initialization and the limits of interpreting latent states.
- Figure to create/refine: detailed routing diagram with physical and latent rows labelled.

## Encoder and training objective

- Separate encoder history from rollout/scoring horizon and from free-run validation.
- Justify the chosen history from reconstructability/implementation and the rollout from the targeted
  fast dynamics, while explicitly acknowledging uncovered slow/marginal behaviour.
- State that the full window is scored: `burn_in=0`.
- Explain why joint estimation may need a longer sensitivity horizon than absorber learning.
- Figure to reuse/refine: honest-scale history/window/validation timeline.

## Closed-loop identification

- Write the controller-in-the-loop residual equation and sign convention.
- Show where the recorded trajectory, controller correction and total model input enter.
- Explain controller state initialization and the absence of an artificial delay.
- Motivate closed-loop identification first from the fact that relevant machine data must be measured
  under feedback; use the integrating axes to explain the numerical difficulty that follows.
- State the risk that feedback masks plant error and pair closed-loop results with plant-domain checks.
- Define a controller-transfer test if a credible alternative controller can be frozen before the
  final experiment campaign.
- Figure to create/refine: aligned open-loop replay and closed-loop training diagrams with the
  continuous/discrete boundary.

## Orthogonal-by-construction

- Begin from the identifiability/negation problem, then define the protected one-step space.
- Derive the construction at sufficient depth to expose coordinates, reference stack, rank decision,
  projection and coefficient refresh.
- State explicitly what happens to additional-state rows and why latent dynamics are not thereby
  physically identified.
- Restrict claims to the measured aggregate one-step property and its assumptions.
- Figure to create: projection geometry plus latent-route exception.

## Experimental setup and data

- For each dataset, state the purpose, operating positions, reference trajectory, excitation band,
  phase/seed separation and constraint checks.
- Explain the corrected all-channel amplitude issue and ensure no result uses the superseded mismatched
  friction dataset.
- State whether data are noiseless and whether friction is in truth only.
- Explain why the hidden absorber was strengthened and distinguish this from machine realism.
- Identify the operational-motion source and provenance: reconstructed ETEL/Jasper setpoint generator,
  ILC reference, or another approved machine motion. Record stroke, acceleration and settling dwell.
- Make the 200--400 ms post-move settling interval an explicit evaluation region; do not treat it as
  part of an undifferentiated full-record metric.
- Keep the current `snr=None` status visible until a noise estimate is obtained from Jasper or ILC/
  stationary data.
- Separate multisine phase realizations from neural initialization seeds in the reproducibility plan.
- Figures to create: operating-region coverage; spectra/crest factor; total-input headroom; record-purpose
  table.

## Results

- Use final artefacts, not plots copied from meeting slides.
- Show baseline, augmentation and black-box comparisons on consistent metrics and initialization.
- Include both training-window and long-horizon/settling behaviour.
- Present results in the order interpolation, ASMPT-relevant motion, controller transfer, and
  operating-range extrapolation; omit any rung that was not actually completed.
- Show a full tracking view and a dedicated settling zoom over a fixed post-move interval.
- If claiming plant improvement, include open-loop or controller-free evidence beside closed-loop fit.
- Use truth-only state/contribution plots only as simulation diagnostics and label them accordingly.
- Figures: representative output/error; settling zoom; residual spectrum; per-record summary; OBC versus
  no-projection parameter/orthogonality comparison; latent contribution ablation.
- Do not allocate a main result figure to exact-model parameter recovery.

## Discussion

- Discuss the 0.1 s versus slow/marginal dynamics limitation directly.
- Discuss feedback masking, data-region dependence and lack of physical identity for latent states.
- Separate structural identifiability from practical identifiability under the selected excitation and
  horizon.
- State that the simulation mismatch and friction benchmark are deliberately controlled and may not
  reproduce every real-machine effect.
- Revisit which meeting-era ideas were not used: SVD reduction, burn-in, inverse/feedforward scope and
  abandoned routing schemes.

## Conclusion

- Answer the final research question using only results that survived the final pipeline.
- Keep method limitations visible: sampled operating region, finite horizon, closed-loop data and
  simulation-to-machine gap.
- Do not elevate parameter-recovery or latent-state interpretation beyond what was demonstrated.

## Appendix

- Place long LPV-LFR coefficient expansions and symbolic verification here.
- Include the raw-to-identifiable-parameter mapping if too large for the main method.
- Put supplementary controller/timing conventions, dataset tables and additional ablations here.
- Do not use the appendix to hide an assumption needed to understand the main method.
