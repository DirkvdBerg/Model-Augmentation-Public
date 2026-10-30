# Brief for Codex: why the PS figures keep failing, and what to draw instead
**From**: Dirk, prepared by a Claude session on 2026-10-29 | **Package**: `tasks/handoffs/2026-10-29-codex-ps-figure-review/`

## 1. The situation
Over three days (27 to 29 Oct 2026), four Claude sessions tried to design the defense slides that explain PS, the predictive-state initialisation of the added states. I (Dirk) rejected every attempt. I am not satisfied with any of them, and I don't understand why Claude keeps failing at this, even though I think I gave enough context.

I want your own, independent judgement. This brief deliberately contains no diagnosis by Claude. Where Claude-written text appears in the inputs (briefs, handoffs, notes on canvases), treat it as part of what you are assessing, not as a conclusion.

## 2. What I ask, in this order
**A. Diagnose.**
- Look at every attempt next to the brief it was made from and my reaction to it.
- Explain why they failed: in the figures themselves, in how the method was understood, and in the briefs Claude wrote for itself and for other sessions. Find the causes yourself; do not limit yourself to my words below.
- Be critical and concrete: point at specific elements of specific figures.

**B. Criteria.**
- From the method (thesis section, code, sources), state what a figure explaining PS must make understood to a mixed control and ML committee in about a minute per slide.
- Ground every criterion in the method or in my reactions. Do not take them from the earlier briefs unchecked.

**C. Concepts.**
- Propose several concepts for showing S1 and S2.
- Each concept rests on its own explanatory idea; merge concepts that rest on the same idea. Distinct because they explain differently, not distinct for the sake of it.
- For each concept give:
  - the idea in one sentence;
  - a sketch of the final figure (SVG or TikZ, or a precise description if you cannot render);
  - the build order, one new element per build;
  - the equation that appears with each build;
  - which criteria from B it meets, and where it is weak.
- Then say which one you would choose and why.

**Note.** Every attempt in this package was made by Claude. If your diagnosis shows the task is underspecified, or that the method itself is hard to draw for a reason no layout fixes, say so.

## 3. My reactions, verbatim (chronological)
Typos are kept as I typed them.

27 Oct, deck session, on the first PS figure (attempt 01):
- "im not a big fan of this figure i think we should start a subsession just purely focused on visual design for PS, and optimize it to be best at visualizing and being critical about how to best show this in a visual format"
- "like still keep the current style with clean and not ai"

28 Oct, PS visual design session (attempt 02):
- "I dont think the current images are clear. but i also dont have good idea in mind yet how it should be done. I think we separate the 2 steps clearly and motivate the steps and clearly make the residual prediction clear"
- "i do think we can show the next step prediction with the figure of the curve or another motion profile but separating over 2 slides sounds smarter"
- "but this is a figure for demonstrating the concept"
- "I dont like that block with past samples and in general its just way too messy, i dont think youre doing a good job at this"
- "but i do like a clear stepwise building up of the figure,but that can still be one slide"
- "no i just think these are really bad"
- "I think its best if we start a new session for this you are too much influenced by the current design."

28 Oct, PS slide design session (attempts 03 and 03b), plus a Claude web session in between:
- "Bowtie is probably the best but i still dont think its clean. especially the error figure, the slides here do the residual better: [meeting 08-10 slides] but these figures are also not clean"
- "do you not see how the slides i referenced made the residual plot?"
- "i just think all the figures you have shown me dont actually describe method that well. we have like residual learning and one step regression. we need to make more clear what each step actually does. currently i dont find these figures clear at aall"
- "and how to properly show this with the figures, so that the figures actually HELP understand the method."
- "This is terrible i will start trying on the web"
- On the web session: "claude on the web is not better ... and its figures were also not good."
- "I want to make use of existing literature examples for similar methods. I think the current figure in my thesis is a lacking for a presentation i think we are able to explain it much more clearly visually"
- "the original I-JEPA figure doenst look nice please dont use this"
- "I dont like the examples you gave. I prefer like minimalistic in the svg style instead of the figures you found. to me that minimalisms keeps it nicely to follow, not too much clutter."
- "and also the slides here didnt show enough of the math to maarten: [meeting 08-10 slides]"
- "I hvae the issue we dont clearly show the one step prediction, how i see a one step prediction is we have our samples and then we want to predict the next sample. so we show the motion profiles (it concists out of data points) and we want to predict the next data point. and we need to establish how we train, how we obtain the data with the decoder, what is this one sample difference with the encoder, how do we train the encoder, what is the residual. you fail at these"
- "i think the idea is much better. We dont properly introduce the decoder and the figures still look quite bad also we should do smoother sine waves BUT the idea is improving of how to show it"
- "but the predicted y is depenent on u right? i dont know how to show it cleanly but for a control audience they will ask questions or they find it unclear if we dont show it"
- "it is simultaniously too much and too less at the same time. we can spread it over more figures but i dont like it. it doesnt explain it actually at all."
- "but is the dissetisfection with those meeting slides also clear to you or not? and maarten didnt see the math so we explained the general concept but not the actual math. SO We need to find like the minimalistic style method with clear formulas (dont have to go into detail) to describe the method"
- "no the visualization should carry the explanation. But the math is also important. but you dont get me at all"
- "i like this step wise explanation introducing it one step at a time, building up the complexitity in a logical structured way" / "i meant for the slides, like in my v2, i build up the eventual figures"
- "but do you also understand why these meeting figures dont cut it, not only due to lacking formulas"
- "i dont like the residual why for theresidual are you basicly drawing a straight line."
- "somehow this is just not workign AND I HAVE DEFINITELY GIVEN ENOUGH CONTEXT"
- "i want a prompt for another claude design session that will give me 15 different approaches of drawing the figure. because youre doing the same one with just a different bit each time"

28 Oct, figure-guidelines session (the same problem, approached through tooling):
- "ow god no you disgust me" (on a test figure)
- "look at the figures i use in my thesis for the style"

27 Oct, deck session, on figures in general (context for the style):
- "In general if these are the figures you will be showing i cant trust you or the roadmap, they didnt actually look at jan his slides or mine"
- "I didnt want you to use these figures from v2"

29 Oct, after the 15-approach canvas (attempt 04):
- "claude is currently failing badly each time"

What I said works:
- the residual on slide 13 of the 08-10 meeting deck (time axis split at τ, y continuous, ŷ starting on y, the gap visible);
- stepwise builds, one element at a time;
- minimal SVG style;
- my thesis figure style.

## 4. The attempts (`attempts/`)
For each attempt, the brief it was made from, what it produced, and my reaction (section 3).

| Folder | Made by | Brief it came from | Contents |
|-|-|-|-|
| `01_deck-session_2026-10-27` | deck build session | storyboard block "14b-2" (see 6.4), line about 249 | `psb.tex`, renders `psb_1..4.png`, `psb_steps.png` |
| `02_ps-visual-design_2026-10-27-28` | PS visual design session | `brief_given.txt` (full text) | `renders/`: TikZ sources and review sheets of concepts opt1 window, opt2 block, opt3 bridge, ps1, s1, s2, sketch, `concepts_ABC.png`; `canvas-PS-slide-design/`: canvas A (error ringing), B (without vs with PS), C (stage cards, the closing slide I accepted) |
| `03_ps-slide-design_2026-10-28` | PS slide design session | `tasks/handoffs/2026-10-28-ps-slide-design.md` | `canvas-PS-slide-concepts/`: bowtie (A1, A2), Hefny time strip (B1, B2), I-JEPA two panels (C); `renders/`: later rounds (`sheet_R/S1/S2.png`, `s1_all.png`, `s2_all.png`, residual from real data `ps4_resid.png`, `ps5_real.png`, web examples sheet) |
| `03b_canvas-PS-stages-figures_2026-10-28` | a 28 Oct session (not established which) | unknown | S1 builds 1 to 4; S2 option 1 time axis; S2 option 2 phase plane |
| `04_15-approaches_2026-10-29` | session of 29 Oct | `brief_given.txt` (my pasted prompt; its attachments were never uploaded) | `canvas-source/`: 15 artboards, each with the final S1 and S2 figures, build order and a one-line self-assessment |

Notes on the files:
- The `.dc.html` files are Claude Design artboards: HTML plus inline SVG, run by a runtime that is not included. Most of the drawing is readable in the SVG; attempt 04 computes its curves in the script block.
- If you need renders of 03b or 04, ask me for screenshots.
- `.html` files under `renders/` were rendered by those sessions themselves; the PNGs are their output.
- The 28 Oct handoff (attempt 03's brief) contains Claude's own diagnosis of attempts 01 and 02 in its section 6. That is part of what you assess.

## 5. Ground truth for the method
Read these yourself. Where they disagree, report it; do not pick one silently.
1. Thesis method: `Thesis-writeup/Writing/sections/03_augmentation.tex`, subsection `sec:aug_ps2` (from about line 1522; S1 objective near line 1702, alternation with V near 1825, S2 near 1890, hand-over near 1952). Read only.
2. Thesis figure: `Thesis-writeup/Writing/figures/tex/ps2_windows.tex`.
3. Code: `Clean/model_augmentation/fit_systems/ps2_opening.py` (`_sample` defines the windows, `_residual` the residual, `_s1_step` the S1 update, `_fit_added_state_transition` the S2 fit and its target).
4. Decisions: `docs/decisions.md`, D-250 (encoder window ends at τ−1).
5. Sources:
   - Hefny, Downey, Gordon 2015 (two-stage regression, also named S1/S2): `Thesis-writeup/References/Hefny-2015-Supervised-learning-dynamical-systems.pdf`, Sec. 2 and Fig. 2.
   - Venkatraman et al. 2017 (predictive-state decoders): `Thesis-writeup/References/Venkatraman-2017-Predictive-state-decoders.pdf`.
   - Downey et al. 2017 (two-stage initialisation before BPTT): `literature/thesis-referenced/downey2017psrnn.pdf`, Sec. 4.2.
6. Data for real curves: `Thesis-writeup/Results/24-10-2026/CSV/res_time_trace.csv`, record I1, axis Y:
   - columns `t_s`, `y_Y`, `baseline_Y`; residual = `y_Y − baseline_Y`;
   - Ts = 0.25 ms, so H = 30 samples = 7.5 ms;
   - errors are shown normalised only, never in metres.

## 6. The slides around it and the style
1. Meeting deck of 8 Oct (the residual picture I point to, and the slides Maarten found lacking in math):
   - `scripts/gantry/meeting/meeting-08-10-2026/slides/08-10-2026-additional-states-black-box-noise_live-g.pdf`;
   - page renders in `reference/meeting-08-10-pages/`.
2. Defense deck so far (slides 12 and 14a show the step block and the encoder window that the PS slides follow): `../Final thesis Presentation/drafts-2026-10-28/slides/partC.pptx`, relative to the repository's parent folder.
3. Storyboard: `../Final thesis Presentation/storyboard.md`. The "PS in three slides" block and section 7 hold the agreed slide order and visual rules. They were written by Claude sessions with me, so they are inputs, not settled answers.
4. Style:
   - black on white, greys carry meaning;
   - one highlight colour #0092B5 for what is new in a build;
   - LaTeX serif math;
   - short labels on the elements, no boxes of words;
   - 16:9;
   - no em-dashes anywhere;
   - house style in `docs/writeup/figure-style.md`;
   - never imitate my V2 deck figures.

## 7. Constraints
- Read only: do not edit the thesis, the code, `kamtin-fp-model/`, or anything in `attempts/`. Never read `kamtin-data/Data Telica/`.
- Write your output into this package folder, as `codex-review.md` plus any sketch files.
