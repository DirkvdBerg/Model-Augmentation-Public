# Handoff: first test of the figure-design skill on the PS S1 figure (what the figure must achieve, not how)
**From**: session of 2026-10-30 | **Branch**: Augmentation | **Effort suggested**: high (design judgement on a method that is easy to misdraw; no derivations, no runs)

## 1. Task
Run the `figure-design` skill (`.claude/skills/figure-design/SKILL.md`) end to end on one presentation figure: **S1 of PS**, the step that trains the encoder's added states through a temporary decoder. This is the skill's first real use; its point is that the design comes out of the skill's steps, not out of this brief. This brief therefore gives you only **what** the figure must make the viewer understand, the verified method facts, and why earlier attempts were rejected. It gives no layout, glyph or build order, and you must not take one from older files (section 12). Start at skill step 1 by putting the proposed claim and inferences (section 4) to Dirk for agreement.

## 2. Out of scope
- R (the residual figure) and S2 (one-step regression): later figures, each its own run of the skill. S1 may assume the viewer already knows what the residual e_τ is.
- Slide composition, titles, placement in the deck: not this skill.
- The thesis: do not read or write `Thesis-writeup/` except the method subsection named in section 11, read only.
- Editing the skill itself. If a step of the skill does not work, note it for Dirk in one line at the end; do not patch it mid-run.
- No training runs, no real-data curves; schematic curves only.

## 3. Where things stand
Tree dirty from other sessions; commit nothing. No run in flight. No approved S1 figure exists. The skill was written 2026-10-30 (D-291) and has never been run.

## 4. Established and verified
**Proposed claim (for step 1, Dirk to agree or change):** S1 trains the added part of the encoder so that, from the past data alone, its state lets a temporary decoder predict the residual the current model will make over the next H samples.

**Proposed inferences (what the viewer must be able to say afterwards), from Dirk's own list of what earlier figures failed to show (storyboard "PS method in three slides", user remark of 2026-10-28):**
- I1. Every time τ gives one training example: input the recorded past window χ_τ (u and y, samples τ−30 to τ−1), target the residual e_τ of that same τ over the next H = 30 samples.
- I2. The decoder p sees only the added state x̄_τ = ψ_a(χ_τ), nothing from after τ, and outputs all H residual values at once.
- I3. The target and the prediction are separate things, and their mismatch is what is minimised.
- I4. That mismatch trains p and the added-state part of the encoder (the gradient goes back through p into ψ_a). This is how the encoder gets trained.
- I5. p is a training device only and is dropped afterwards.
- I6 (math, Maarten asked for it): min over θ_p, θ^a_enc of Σ_τ ‖p(ψ_a(χ_τ)) − e_τ‖².

**Method facts, checked against code and thesis on 2026-10-29:** `tasks/handoffs/2026-10-29-ps-figures-round-2.md` section 4 (window, residual target, S1, timescale, claim strength). Use it as the fact base. Two of those facts matter most for not drawing something false:
- The residual comes from the current model simulated in **closed loop** with the controller, not from replaying the recorded force u.
- The thesis says the decoder **encourages** predictive information in x̄ (`03_augmentation.tex:1753`, changed on purpose from "forces"). Nothing shows x̄ equals physical absorber coordinates.

**Notation:** exactly as in the thesis: χ_τ, ψ, ψ_a, x̂_τ|τ, x̄_τ|τ, p, e_τ, H, θ_p, θ^a_enc. Qualifications (alternation with V, fixed target within a step, σ_e scaling) go in speaker notes, not on the figure.

## 5. Assumed but not verified
- That I1 to I6 are the right list. Settled by Dirk in skill step 1.
- Whether all six belong on one figure or I5 belongs to a later figure. Dirk decides in step 1.

## 6. Tried and failed
Reasons only, to be handed to the concept agents as constraints (skill step 4). Do not show them the figures.
- Objects drawn as a checklist (window, encoder, state, decoder, wave) without showing what is compared with what -> Dirk: "they dont actually describe method that well … we need to make more clear what each step actually does".
- The residual drawn as a free-standing wave, or the model output as a near-straight line -> Dirk: "why for theresidual are you basicly drawing a straight line". The residual must read as measured minus model.
- Target and prediction not visibly separate, so the fit could not be seen.
- Easier objects swapped in for the real ones (state ring, phasor, poles, loss landscape, training time-lapse, Hankel block): each claims something the method does not do -> Dirk: "claude is currently failing badly each time".
- Words doing the work: a box filled with "past samples", block diagrams whose meaning is in their labels -> Dirk: "I dont like that block with past samples".
- Captions, legends and conclusion sentences added while the comparison itself was missing -> Dirk: "simultaniously too much and too less at the same time".
- Math missing, then math dumped -> Dirk: "the visualization should carry the explanation. But the math is also important".
- "N genuinely different approaches" quotas: new silhouettes for the same unexplained content -> Dirk: "youre doing the same one with just a different bit each time". The skill's named-axis divergence replaces this.
- Overclaiming ("must carry the missing dynamics") carried from brief to brief.

What Dirk said works (preferences, not a design): minimal and clean, his thesis figure style, stepwise builds that add one thing at a time, clear formulas without detail.

## 7. Achieved
The skill, and the fact base in section 4. Nothing else for S1.

## 8. The open question
Nothing blocked. Dirk's answers in steps 1 and 6 of the skill drive the run.

## 9. Next action
Invoke the `figure-design` skill and do step 1: show Dirk the claim and I1 to I6 from section 4, ask him to agree or change them, then continue through the skill's steps, stopping where it says the user decides: step 1 (spec) and step 6, where Dirk sees about three different rendered figures side by side and picks one. Spec and outputs go in `../Final thesis Presentation/drafts-2026-10-30/figure-design-S1/` (relative to the repository's parent folder), starting with `spec.md`.

## 10. Acceptance criterion
Two parts, both needed:
- Dirk gets about three different rendered S1 figures to choose from, and the fresh reader of skill step 5 recovers the agreed inferences from the one he picks.
- Dirk approves the finished figure in his words.
Separately, one line for Dirk on how the skill itself performed: which step helped, which did not.

## 11. Read these first
1. `.claude/skills/figure-design/SKILL.md`: the procedure you run.
2. `tasks/handoffs/2026-10-29-ps-figures-round-2.md` **section 4 only**: verified method facts.
3. `Thesis-writeup/Writing/sections/03_augmentation.tex` lines 1522 to 1965, read only: the method and its notation.
4. `Clean/model_augmentation/fit_systems/ps2_opening.py` (`_sample`, `_residual`, `_s1_step`): only if a fact in section 4 needs its source.
5. `docs/writeup/figure-style.md`: style for the rendered figures (steps 4 and 7).

## 12. Do not
- **Do not read old designs.** That means: sections 9 to 13 of the round-2 handoff; the "Visual" and per-slide lines of the storyboard's "PS method in three slides" block; `codex-review.md` section C and its `sketches/`; everything under `tasks/handoffs/2026-10-29-codex-ps-figure-review/attempts/` and `review-evidence/`; `../Final thesis Presentation/drafts-2026-10-28/ps-round2/`. They are the designs the skill is meant to get away from, and showing them causes fixation (skill, "Why this skill exists").
- **The one exception:** the rejected renders above may go to the step 5 fresh reader, to check the candidates do not repeat them.
- Never pass section 6's figures, or any example figure, to a concept agent; only its reasons.
- Do not draw these objects: state ring or phasor, poles, loss landscape, training time-lapse, Hankel block, word boxes.
- Do not imitate Dirk's V2 deck figures.

## 13. Operational
- TikZ standalone, compiled to PDF then PNG. A working preamble and toolchain: `tasks/handoffs/2026-10-29-codex-ps-figure-review/generate-sketches.py` (copy its preamble only, not its layouts; do not open the sketch outputs).
- Show renders to Dirk with SendUserFile.
- No runs, no data processing.

## 14. Delegation
As the skill sets: up to 4 fresh sub-agents for this figure (Dirk, 2026-10-30), normally 3 concept agents in step 4 and 1 fresh reader in step 5. No Explore agent, no Workflow.
