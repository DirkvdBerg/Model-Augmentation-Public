# Handoff: get the defense storyboard right before any slide is built
**From**: session of 2026-10-29 | **Branch**: Augmentation | **Effort suggested**: high (story and design judgement, no mechanical work)

## 1. Task
Dirk gives a 20-minute MSc thesis defense. In his words: "I think the storyboard is most important and we need to get that down, because the slides are not it". Work through `Final thesis Presentation/storyboard.md` with him until it is a storyboard he would present from. For every slide it should say:
- the one thing the committee understands after that slide;
- what is on screen;
- the order in which things appear;
- whether a video or animation earns its place.

Form your own view from the sources in section 11, starting with the thesis and Jan Hoekstra's two decks, and propose that view to him. Discuss one part at a time. Write each agreed decision into `storyboard.md` in the same turn.

## 2. Out of scope
- Building slides, pptx files or figures: the storyboard comes first, and slides follow later in a separate step.
- The thesis text in `Thesis-writeup/`: it is submitted. Read only, and never compile it.
- New experiments or runs: all numbers come from the thesis.

## 3. Where things stand
- `storyboard.md` is a draft from earlier sessions. Treat its slide cards and dated notes as a proposal to question, not as settled. Only Dirk's own decisions in section 4 below are settled.
- Another session ("Defense deck walkthrough") may still be editing `storyboard.md`. Dirk decides which session continues.

## 4. Established and verified
These are Dirk's decisions, with his words where quoted:
- Length: 20 minutes, on the TU/e template.
- Intro idea: explain white box and black box first, so that combining them (augmentation) reads as the obvious choice. Do not show the gantry equation in the intro: "i wouldnt already state our phyiscs baseline there already".
- Take Jan's principles, not his layouts: "im not a fan of just copying him".
- From his V2 deck he likes the idea of introducing each element step by step ("that idea, the slides are ai and also not that good"), not the slides themselves.
- Videos: "they should actually add value and overdoing it for the sake of a video is quite bad". Position-dependent effects interest him.
- The open-loop drift animation is liked: "the animation is actually quite nice".

## 5. Assumed but not verified
None.

## 6. Tried and failed
- Designing slides before the story was fixed. Dirk rejected the output repeatedly as "ai generated", and the cause is that the visuals were invented before it was settled what each slide must convey. That is why this task is the storyboard only.

## 7. Achieved
None that binds you. Drafts in `Final thesis Presentation/drafts-2026-10-28/` exist only for reference, for example `videos/A0_drift.mp4` for the liked drift animation.

## 8. The open question
What story, told over about 20 minutes of slides, makes the committee understand the thesis contribution? Your reading of the thesis and of Jan's decks should drive the answer.

## 9. Next action
1. Read the thesis PDF and Jan's two decks.
2. Give Dirk a short story arc: the parts, the minutes per part, and the one message per part.
3. Agree on it, then go through it part by part.

## 10. Acceptance criterion
Dirk approves the storyboard in his own words, part by part.

## 11. Read these first
1. `Thesis-writeup/Writing/build/main.pdf`: the content. The talk tells this thesis.
2. `Final thesis Presentation/Example presentations - Jan/benelux_2026.pptx` and `ecc_2025 1.pptx`: the quality bar for how an augmentation talk is structured and paced. Look at them as images; to render them, use `powershell -File "Final thesis Presentation/drafts-2026-10-28/scripts/render.ps1" <pptx> <outdir>`, which writes PNGs.
3. `Final thesis Presentation/Control group presentation/Dirk presentation notes.txt`: Jan's feedback on Dirk's previous talk.
4. `Final thesis Presentation/Control group presentation/Interpretable Model Augmentation for a High-Precision Dual-Gantry Motion System - V2.pptx`: that previous talk. Use it for the step-by-step idea only.
5. `Final thesis Presentation/storyboard.md`: the draft to revise.

## 12. Do not
- Build slides or figures, or write pptx files.
- Copy Jan's slide layouts one to one.
- Write anywhere other than `storyboard.md`.

## 13. Operational
None.

## 14. Delegation
None. This is a conversation with Dirk. If rendering Jan's decks takes long, one subagent may do it.
