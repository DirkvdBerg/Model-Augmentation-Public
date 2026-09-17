# Handoff: choose the MSc workshop presentation title

**From**: session of 2026-09-17 | **Branch**: `Augmentation` | **Effort suggested**: `medium`

## 1. Task

Help the user choose one title for a 20-minute MSc workshop presentation about the graduation
project. Start from the user's original title, identify the central message that can be defended at
the current project stage, propose a small set of close refinements, and recommend one final title.
Treat this as an interactive editorial decision rather than a technical research task.

The title must be sent before Friday, September 18, 2026. The workshop is on October 2 from 10:00
to 12:30 in Flux 0.150. The 20-minute slot includes questions.

## 2. Out of scope

* Do not create presentation slides or an abstract.
* Do not change the thesis scope or invent a new research contribution.
* Do not claim that orthogonal-by-construction training has already improved parameter recovery.
* Do not edit repository files.
* Do not send or draft the email unless the user asks after choosing the title.
* Do not produce a long brainstorming list. The user needs a decision.

## 3. Where things stand

The original working title is:

> Model Augmentation for Dual-Gantry High-Precision Motion Syste

The last word is an incomplete spelling of `System`. The current project identity in `CLAUDE.md`
uses:

> Model Augmentation for a Dual-Gantry High-Precision Motion System

A grammatically smoother version is:

> Model Augmentation for a High-Precision Dual-Gantry Motion System

The project augments a physics-based nonlinear LPV gantry model with learned dynamics. Its main
scientific concern is retaining the interpretability of the physical parameters by preventing the
learned component from duplicating baseline parameter directions.

The one-step OBC implementation exists and has passed implementation diagnostics, but no OBC
training comparison has established improved parameter recovery yet.

## 4. Established and verified

1. The application is a high-precision dual-gantry motion system.
2. The overall research topic is model augmentation, not only orthogonal projection.
3. The baseline is physics based, nonlinear and LPV.
4. The augmentation contains learned dynamic states.
5. Orthogonal-by-construction separation is the intended method for preserving physical parameter
   interpretability.
6. The first implemented construction concerns one-step state-equation orthogonality.
7. No completed training result yet proves that OBC improves parameter recovery on the gantry.
8. The presentation is short and likely addresses a broader MSc audience than the thesis committee.

## 5. Assumed but not verified

1. The workshop audience probably includes students and staff outside the narrow model-augmentation
   literature.
2. The presentation will probably cover the complete project motivation and model structure, with
   OBC as the main current methodological contribution.
3. The workshop title does not have to equal the final thesis title.

Ask at most one short question if the answer materially changes the recommendation. A useful
question is whether the presentation will focus mostly on OBC or introduce the broader model
augmentation project. If the user is unsure, recommend the broader title.

## 6. Tried and failed

None. The current task is choosing among accurate levels of specificity.

## 7. Achieved

The original title already has the correct broad subject. The remaining editorial choices are:

* whether to add the article `a`;
* whether `high-precision` should directly modify `dual-gantry motion system`;
* whether to add `interpretable` to signal the scientific contribution;
* whether naming OBC is appropriate before the comparative training result exists.

## 8. The open question

Should the workshop title remain broad, add the goal of interpretability, or name the OBC method?

Use these principles:

* A broad title is clearest and safest for a mixed MSc audience.
* `Interpretable` identifies the research problem without claiming a successful experimental result.
* `Orthogonal-by-Construction` is precise but specialized and may make the presentation sound
  narrower and more complete than the current evidence supports.

## 9. Next action

Compare exactly these three title levels:

1. **Broad:** `Model Augmentation for a High-Precision Dual-Gantry Motion System`
2. **Contribution focused:** `Interpretable Model Augmentation for a High-Precision Dual-Gantry Motion System`
3. **Method focused:** `Orthogonal-by-Construction Model Augmentation for a High-Precision Dual-Gantry Motion System`

For each, explain in one or two sentences what expectation it creates for the audience. Then
recommend exactly one title based on the present maturity of the work and the 20-minute workshop
format. If a shorter grammatical refinement is clearly stronger, include only that one additional
candidate and explain what it replaces.

After the user responds, refine wording around their preferred emphasis. Do not reopen the entire
title search.

## 10. Acceptance criterion

The task is complete when the user has one final title that:

* accurately describes the project;
* reads naturally in English;
* is understandable to a mixed MSc workshop audience;
* fits a 20-minute presentation;
* does not claim a result that has not been measured;
* remains consistent with the likely presentation content.

## 11. Read these first

1. `CLAUDE.md` for the official project identity and overall objective.
2. `presentations/Research_plan___Graduation_project_AI_ES___Feedback_Processed (3).pdf` only if
   the original presentation scope is needed to resolve the title.
3. `scripts/gantry/orthogonal-by-construction/documentation/obc-gantry-one-step-implementation-plan.tex`
   only if the user wants the method named explicitly.
4. D-192 through D-195 in `docs/decisions.md` only to avoid overstating implementation results.

Do not expand this into a general repository review.

## 12. Do not

* Do not give more than four candidates.
* Do not use unexplained acronyms such as LPV, LFR, ANN or OBC in the title.
* Do not include `parameter recovery`, `guaranteed`, `preserving`, `successful` or `improved` unless
  the wording clearly states a goal rather than an established result.
* Do not make the title so technical that the application disappears.
* Do not replace the user's original direction with a generic machine-learning title.
* Do not contact the workshop organizers.

## 13. Operational

This is a read-only discussion. No code, command, external search or long document audit is needed.

Open with the recommended title, then explain why it is the best refinement of the user's original
title. Show the other two levels underneath for comparison.

## 14. Delegation

No subagent is needed.
