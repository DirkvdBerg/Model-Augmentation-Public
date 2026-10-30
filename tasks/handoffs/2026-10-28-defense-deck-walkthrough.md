# Handoff: walk through the defense deck with Dirk and design it properly
**From**: session of 2026-10-28 | **Branch**: Augmentation | **Effort suggested**: high (design judgement and critique, not mechanical building)

## 1. Task
Be Dirk's presentation designer and sparring partner for his 20-minute MSc thesis defense. Walk through the deck with him part by part, starting at the intro (slides 1 to 4):
- discuss what each part must convey;
- decide how to spread it over slides;
- decide which presenting concepts and methods to take from Jan Hoekstra's decks;
- decide where a video or a position-dependent animation adds real value.

Form your own design judgement and critique your own output before showing it. Dirk's complaint about the last session, in his words: "apparently i need to spell out everyting else you give me dogshit". He wants a session "capable at understanding and determining for itself". Build a slide only after the two of you agree on what it shows. Every agreed decision and its reasoning goes into `storyboard.md` in the same turn, unasked (his rule: "you need to write things like this down").

This is a design conversation first, building second. In the first turn, read section 11, look at the reference decks as images, and open with your own short reading of what the intro must convey and how Jan does it. Then discuss.

## 2. Out of scope
- The thesis text (`Thesis-writeup/`): finished and submitted; never write there, never compile it.
- PS method figures (slides 14b-1 to 14b-3): the separate session "PS visual design" (`tasks/handoffs/2026-10-28-ps-slide-design.md`) owns them. Its conclusion is in section 6. Coordinate, do not redo.
- New experiments or training runs: all numbers come from the thesis tables and `Results/24-10-2026/CSV/`.
- Writing the final deck into `Final thesis Presentation/` root: only after Dirk approves, with the file name confirmed. Drafts go in `drafts-2026-10-28/` or a new drafts folder.

## 3. Where things stand
- Branch Augmentation, tree dirty (Dirk commits himself). Nothing running.
- Plan of record: `Final thesis Presentation/storyboard.md`:
  - section 3: slide cards with dated decision notes;
  - section 4: animations;
  - section 5: backups;
  - section 7: visual plan.
- The 2026-10-28 notes under slides 3 and 13 are the newest state.
- Draft assets copied from the last session's scratchpad to `Final thesis Presentation/drafts-2026-10-28/`:
  - `slides/`: partA to partD .pptx, s12_test, s3_jan (the Jan copy), intro_s3 (latest slide 3 and 3b), review PNGs;
  - `videos/`: A0_drift, A1_rollout, A3_inertia, A4_obc, as .mp4 plus first and last stills;
  - `scripts/`: the python-pptx builders, drift_sim.py and drift_anim.py with drift_I1.npz, render.ps1 (PowerPoint COM, pptx to PNG), pdf2png.py, stack.py;
  - `reference/`: Jan Benelux contact sheets, V2 intro sheets.
- The scripts still point at the old scratchpad paths in their `HERE`/output constants; repoint them when you reuse one.

## 4. Established and verified
Dirk's decisions (his words where quoted):
- **Template.** The real TU/e template `TUe PowerPoint 16x9 template/TUe_16x9_ppt_with_instructions.potx`, with its own warm grey background as in his V2 deck ("i prefer that background"). This replaces the white-background rule in storyboard section 7.
- **Diagrams.** Native PowerPoint shapes so slides stay editable; a LaTeX PNG for equations (Cambria Math renders the dot accents of q̇ and q̈ badly).
- **Jan: principles, not copies.** A slide copied from Jan's layout was "a really good job at copying jan's style, but im not a fan of just copying him".
- **Intro order** (agreed: "this sounds good"):
  1. machine;
  2. why a model (two loops);
  3. white box (first principles) and black box first, so that combining them reads as the obvious choice;
  3b. augmentation as our block scheme, its own slide, returning later as a map;
  4. goal, with the three aspects placed on that map.
- **Slide 3 does not show our gantry equation yet:** "i wouldnt already state our phyiscs baseline there already".
- **Stepwise builds.** From V2 Dirk likes the stepwise introduction of each element ("that idea, the slides are ai and also not that good").
- **Videos only where they add value.** "I like videos but they should actually add value and overdoing it for the sake of a video is quite bad." Position-dependent effects interest him ("with like position dependent effects also we could do cool stuff to visualize things").
- **Open-loop drift video (A0) is liked:** "the animation is actually quite nice". It needs polish (storyboard section 4, A0 to-do list). The WHY is agreed and written on the slide 13 card and in backups B4, B16 and B17:
  - no spring in X and Y, so a force error integrates into a position offset;
  - the machine never drifts because its controller corrects it;
  - so the model is simulated in the same loop.
- **Naming.** "Digital twin" may be said, but the thesis dropped the term. Say "a plant model used to test controllers".
- **Rule conflict to know about.** The memory rule "never V2 figures or style" still holds for V2 figures and colours. Dirk on V2: the coloured white and black box figures "look ai", and V2 is "compared to jan his slides ... not up to standard".

## 5. Assumed but not verified
- That slide 3b may sum the network at the output for the intro. In the thesis the network adds to the state update (thesis Fig. 3, `figures/tex/augmentation_structure.tex`). This is my reading; ask Dirk.
- Highlight colour blue 0092B5 versus scarlet C81919 (storyboard open point).
- A0 drift: the replay used a 4 kHz block-mean force. That the drift is not an artefact of that needs the 20 kHz replay (A0 to-do).

## 6. Tried and failed
Each attempt produced an "AI look" or a copy:
- **White rounded cards on a tinted panel, pastel fills with coloured outlines, tiny sans labels** (V2 style, and a Claude Design test deck) -> Dirk: "ai slides", "the design is terrible" -> generic web-UI styling, not presentation design -> Claude Design dropped for real slides.
- **Bold column headings with mirrored + and - rows, cons in grey, two identical u -> box -> y rows, arrows converging into a centred pink question box** (intro_s3 v1) -> "the output is quite ai generated" -> stock pros/cons template and forced symmetry.
- **Literal solid black block and white block with a tiny spring sketch, plus a half-white, half-black "augmentation" block** (intro_s3 v2) -> rejected -> big empty shapes, a gimmick, and bullets and diagram saying the same thing twice.
- **Exact replica of Jan's Benelux slide 3 layout** (s3_jan) -> well done but a copy -> Dirk wants his own deck.
- **Scaled-down thesis figures on slides** (2026-10-27) -> "if these are the figures you will be showing I can't trust you" -> too many labels at slide size.
- **AI-invented metaphors for PS** (spirals, made-up signals, a "bridge"), from the PS session -> every round failed the same way: the picture came from the model, not from the method or from Dirk -> figures must come from the method itself or from Dirk's own sketch.
- **Common cause.** The previous session kept producing a generic layout first and polishing details nobody asked for. Design starts from what the slide must make the audience understand, then the simplest honest picture of that.

## 7. Achieved
- Storyboard with all 24 slide cards, backups B1 to B17, and the animation list. Message-first, approved per part on 2026-10-27, revised 2026-10-28.
- Videos (drafts): A0 open-loop drift (8 s open loop, then the closed-loop line, Y/Y_tr), A3 inertia versus Y, A4 OBC projection, A1 free-run rollout. Style of A0 and A4 approved; content of A0 approved with to-dos.
- No slide is approved as final.

## 8. The open question
How should the intro (slides 1 to 4) look so that it is clearly Dirk's own, at Jan's standard, and not AI-looking?

Candidate approaches:
- Dirk sketches by hand or points at a reference slide, and you execute.
- You propose with your own critique, using concrete Jan principles.

Decide with him in the walkthrough. Evidence that chooses: his reaction to your first proposal for slide 3.

## 9. Next action
Open the walkthrough on the intro:
1. Look at Jan's Benelux slides 1 to 6 and ECC slides 1 to 6 as images, and at Dirk's V2 slides 1 to 6 (`reference/`).
2. Read the slide 1 to 4 cards in the storyboard.
3. Give Dirk, in a few lines:
   - what the intro must make the committee understand;
   - which Jan methods serve that, named concretely;
   - your proposal for how many slides and what each shows, as a short visual overview Dirk can read quickly (he asked for "short and not a lot of text").

Then discuss. Build nothing before he agrees.

## 10. Acceptance criterion
Dirk approves each slide in his own words. Self-check before showing anything:
- Would a careful human designer have made exactly this?
- Does every element carry the slide's one point?
- Is anything there only for symmetry or decoration?
- Is it free of the section 6 patterns?

## 11. Read these first
1. `Final thesis Presentation/storyboard.md`: sections 1, 1b, 2, the Part A cards, the slide 13 card, section 4 (A0), section 7.
2. `Final thesis Presentation/Example presentations - Jan/benelux_2026.pptx` and `ecc_2025 1.pptx`: render to PNG with `drafts-2026-10-28/scripts/render.ps1` and look at them. Jan is the quality bar.
3. `Final thesis Presentation/Control group presentation/Interpretable Model Augmentation ... - V2.pptx` and `Dirk presentation notes.txt`: his previous deck and Jan's comments on it ("could spend more time on the different models and explain that better").
4. `drafts-2026-10-28/slides/intro_review.png` and `slides/s3_review/`: the rejected attempts, so you see the failure modes.
5. `Thesis-writeup/Writing/sections/01_introduction.tex`: the content source for the intro.

## 12. Do not
- Copy Jan's slides one to one; reuse V2 figures or colours; put white cards on the grey background; use custom +/- pro/con tables or forced symmetric columns.
- Invent visual metaphors that are not in the method; show the gantry equation before Part B.
- Add a video where a still does the job.
- Publish to claude.ai for slide design (Claude Design was rejected on quality). Artifacts for quick previews only if Dirk asks.
- Run git write operations; write in `Thesis-writeup/` or `kamtin-fp-model/`.

## 13. Operational
- Python: `conda run -n GraduationProject python <script>`. Render: `powershell -File drafts-2026-10-28/scripts/render.ps1 <pptx> <outdir>` (PowerPoint COM, 1920x1080 PNG per slide).
- Equations: standalone LaTeX, pdflatex, then `pdf2png.py <pdf> <stem> <dpi>`.
- Videos: matplotlib animation to MP4 (see `drift_anim.py`), always with a last-frame still as a fallback.
- Thesis numbers: `Thesis-writeup/Writing/tables/*.tex` and `Results/24-10-2026/CSV/`. Errors as NRMS_e %, frequencies in Hz, no metres.

## 14. Delegation
None for the walkthrough: it is a conversation. At most one subagent, only to render or inspect a whole reference deck if that is faster.
