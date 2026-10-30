# Handoff: design how the defense deck explains PS (predictive-state initialisation), grounded in how the literature shows it
**From**: session of 2026-10-28 | **Branch**: Augmentation | **Effort suggested**: high (open-ended visual design plus a short literature search; no derivations)

## 1. Task
Design the slides that explain PS, the predictive-state initialisation of the added states, for Dirk's 20-minute MSc thesis defense. Start fresh: first find out how the literature and good teaching material demonstrate predictive states, two-stage regression and encoder or decoder pretraining visually. Then propose 2 or 3 genuinely different visual concepts on a new Claude Design canvas (Artifact tool, Design type), converge with Dirk on one, and work it out into clear step-by-step builds. The PS explanation sits between slide 14b-1 (Y error spectrum without PS, the 212 Hz peak stays) and 14b-3 (with PS, peak gone, 53.6 % to 6.1 %). It ends with a three-stage overview slide: S1, then S2, then train V as usual (decided by Dirk, see section 4).

## 2. Out of scope
- Building the .pptx, and placing figures in the deck: the deck session does that.
- Final production to 300 dpi PNG or TikZ: a follow-up once Dirk approves a design.
- Changing the method, the thesis text or any result: read-only sources.
- A recorded-run animation of PS learning: a possible option, not decided, in the storyboard right under the "PS in three slides" block. Do not start it. A concept may leave room for it, for example a static frame that a later animation could replace.
- Slides 14b-1 and 14b-3: already agreed, not part of this task.
- Never write in `Thesis-writeup/` or `kamtin-fp-model/`; never read `kamtin-data/Data Telica/`.

## 3. Where things stand
- Branch `Augmentation`; the tree is dirty from other sessions. Commit nothing (standing rule).
- No run in flight.
- Earlier design attempts exist and were all rejected by Dirk (see section 6). This session should NOT open them, to stay unanchored. The single exception is the three-stage overview card layout (section 4).

## 4. Established and verified
**The method** (thesis `Thesis-writeup/Writing/sections/03_augmentation.tex`, subsection `sec:aug_ps2` from line 1522; eqs. `ps2_s1` near line 1702, `ps2_s2` near line 1881, `ps2_handover` near line 1949; figure `Writing/figures/tex/ps2_windows.tex`):
- **Problem.** The added states x̄ start from random encoder rows. The loss V scores only the output, and f_aug can lower part of the error with the physical states and force alone. So training can leave the missing dynamics (the 212 Hz absorber mode) out of x̄.
- **S1.** The encoder ψ maps the past window χ_τ (n + 1 = 30 samples of recorded y and u) to the state; its added part is x̄_τ = ψ_a(χ_τ). A temporary decoder p (one hidden layer) predicts the model's residual e_τ over the next H = 30 samples from x̄_τ alone. The minimiser is the conditional mean, so x̄_τ becomes a predictive state. S1 trains p and the added-state encoder rows.
- **S2.** g_aug is trained to predict the next added state. Its input is x̂_τ and the recorded u_τ; its target is ψ_a(χ_{τ+1}), from the same encoder on the window one sample later. S2 trains g_aug.
- **Hand-over.** p is dropped and V is minimised as before from the PS parameters. PS changes the start, not what is minimised.
- **Sources.** PS is adapted from two-stage regression, Hefny et al. 2015 Sec. 2 (`literature/thesis-referenced/hefny2015supervised.pdf`; its Fig. 2 inspired the thesis windows figure). It is used before BPTT as in Downey et al. 2017 Sec. 4.2 (`literature/thesis-referenced/downey2017psrnn.pdf`). Both are also under `literature/added-states-training/`.

**Dirk's decisions (2026-10-27/28, his words in quotes):**
- The figures are concept figures: "this is a figure for demonstrating the concept". Illustrative curves are allowed, such as a predicted error drawn over the real one.
- "Separate the 2 steps clearly and motivate the steps", and "clearly make the residual prediction clear".
- "I do like a clear stepwise building up of the figure."
- "I do think its good to have the 3 stages as an last slide for these, to also show that we train V as usual." The approved layout for that closing slide is three cards in a row (S1 → S2 → then train V as usual). Each card shows what is learned, why in one line, a small picture, and what it trains. It is artboard `Stages.dc.html` on canvas https://claude.ai/artifact/SxhsLgd6oVJmfA5o7aNYvp. That is the only earlier design you may look at, and only for this closing slide.

**Data for curves (real):**
- Record I1, axis Y, from `Thesis-writeup/Results/24-10-2026/CSV/res_time_trace.csv` (columns t_s, r_Y, y_Y, baseline_Y, final_Y). Residual = y_Y minus baseline_Y. Ts = 0.25 ms, so H = 30 samples = 7.5 ms.
- A 7.5 ms window cannot resolve 212 Hz (its dominant FFT bin lands at 196 Hz). Do not label a 30-sample curve "212 Hz"; slide 14b-3 carries the mode evidence.
- Normalised curves already exist: `curves.py` and `curves.tex` in `C:\Users\20203253\AppData\Local\Temp\claude\C--Users-20203253-OneDrive---TU-Eindhoven-Graduation-Project-Baseline-FP-model-Baseline-LPV-Augmentation\ffaed620-117e-4144-ac0f-bedc90562873\scratchpad\ps\`. They hold the residual over τ to τ+H-1 (`rescurve`), over τ-n to τ+H-1 (`rescurvelong`), and an illustrative prediction (damped sinusoid fit) in both normalisations (`predcurve`, `predlong`). Reuse the numbers, not any layout.

**Deck rules** (storyboard `C:\Users\20203253\OneDrive - TU Eindhoven\Graduation Project\Final thesis Presentation\storyboard.md`, "PS in three slides" block and section 7):
- Thesis figure style (`docs/writeup/figure-style.md`): black on white, greys carry meaning.
- Exactly one highlight colour, template blue #0092B5, for what is new in each build step.
- Jan Hoekstra's slide patterns: a diagram with its equation, one block reused, a question box.
- Never use or imitate Dirk's V2 deck figures, colours or layouts.
- Errors normalised, never metres; frequencies in Hz.
- 16:9 TU/e slide; content area about 21.5 x 9 cm.
- Earlier slides use the symbols ψ, χ_τ, x̃, x̄, x̂, g_aug, f_aug, f_base and V. Slide 12 shows the step block with φ_aug giving f_aug and g_aug. Slide 14a shows the encoder window: past samples, encoder ψ, start state.

## 5. Assumed but not verified
- That a Design canvas can export something the deck can use (PNG or SVG). If it cannot, the approved design is reproduced in TikZ by a follow-up session. Check the canvas Share or Export menu when it matters, not before.
- Whether Dirk wants the PS explanation as one slide or two before the closing three-stage slide. Settle it together with the chosen concept.

## 6. Tried and failed
All rejected by Dirk for clarity. They share these failure mechanisms:
- One slide carrying S1, S2 and the hand-over together, as one time-axis schematic (4 builds) -> "not a big fan" -> neither stage got its reason, and the residual was a small wiggle among arrows.
- Variants: the 14a sample-cell row continued, a bridge (past band to x̄ to future band), the slide 12 block with stage costs -> "not clear" -> too many symbols per build (ψ, ψ_a, χ_τ, χ_{τ+1}, p, x̂, u, e_τ, H and tick labels), plus text in three places (motivation line, legends, cost line).
- A grey "past window" block labelled with words -> "I dont like that block with past samples" -> a large box that shows nothing.
- Two slides (S1, S2) with motivation, real residual, dashed prediction and an x̄ sketch -> "way too messy" -> label collisions, legends, and three kinds of text per figure.
- One minimal slide, five builds (error, prediction, x̄ dots, g_aug step, hand-over) -> "really bad" -> stripped of meaning: no visible reason why each stage exists, nor how x̄ connects to the error.
- The pattern behind all of these: concepts were invented from the thesis figure instead of from how others explain predictive states, and each round patched the previous layout instead of rethinking it. That is why this session starts from the literature and must not open the old files.

## 7. Achieved
- The method facts, data and curves in section 4.
- The decided closing slide: three-stage overview cards, `Stages.dc.html` on the existing canvas.

## 8. The open question
What visual metaphor makes "x̄_τ is trained to predict the coming error, so it must carry the missing dynamics" obvious to a mixed control and ML committee in about one minute? Candidates come from the search in section 9, for example:
- how Hefny, Downey, Boots, Venkatraman or Sun et al. illustrate predictive states (past features, future features, the state as the bridge);
- how SUBNET or encoder papers (Beintema, Hoekstra) show the encoder;
- general ML teaching visuals for auxiliary prediction tasks and pretraining.

Dirk chooses among 2 or 3 concepts on the canvas.

## 9. Next action
Agree the search sub-questions with Dirk in one short message, per the memory rule on literature agents: claims and sub-questions are agreed before launching. Then run the `deep-research` skill. Suggested sub-questions:
1. Which figures do the predictive-state and two-stage regression papers (Hefny 2015, Downey 2017, Boots, Venkatraman, Sun 2016 PSIM) use to explain a predictive state?
2. How do talks, tutorials and slides explain predictive state representations or encoder pretraining visually?
3. How do SUBNET and encoder-based system identification papers draw the encoder and the future-prediction target?

Return the best 5 to 8 figure ideas, each with its source and what makes it clear. Then create a NEW Design canvas (do not reuse the existing one, except to read `Stages.dc.html`). Put 2 or 3 concepts on it, each a final-build artboard at slide size, with its builds listed in a note. Send Dirk the link.

## 10. Acceptance criterion
Dirk approves one concept and its build steps on the canvas. Each build adds exactly one idea, shown in blue. The two stages are visibly separate and each carries its reason. The residual prediction is the visual centre of S1. The closing three-stage slide shows that V is trained as usual afterwards.

## 11. Read these first
1. `Thesis-writeup/Writing/sections/03_augmentation.tex` from line 1522 (sec:aug_ps2): the method as written; read only.
2. `literature/thesis-referenced/hefny2015supervised.pdf`, Sec. 2 and Fig. 2: the source concept and its own figure.
3. `literature/thesis-referenced/downey2017psrnn.pdf`, Sec. 4.2: two-stage initialisation before BPTT.
4. Storyboard `Final thesis Presentation/storyboard.md`, the "PS in three slides" block and section 7: the deck's visual rules and neighbouring slides.
5. `docs/writeup/figure-style.md`: the house style.

## 12. Do not
- Do not open the rejected attempts: `...\e1023128-...\scratchpad\deck\psb*`, `partC.tex` pages 11 to 14, the `PS_14b-2_*` folders in the scratchpad named in section 4, and the artboards `Main.dc.html` and `Contrast.dc.html` on the existing canvas.
- Do not put a box of past samples, legends, or text in three places on one figure (section 6).
- Do not publish anything outside private claude.ai artifacts. Do not share artifacts; sharing is Dirk's choice.
- No em-dashes anywhere, figures included (standing rule).

## 13. Operational
- Claude Design: Artifact tool, `action: "quickstart"` with `intent: "design"`, then publish with the returned `type_url` and a `title`. Follow the type's instructions for `project/canvas.json` and `project/*.dc.html`.
- Curves: `conda run -n GraduationProject python <file>` (multi-line snippets in a file).
- A Bash heredoc in this environment can fail with a Git Bash fatal error. Use PowerShell or the Write tool for files.

## 14. Delegation
The literature search runs through the `deep-research` skill, after Dirk agrees the sub-questions: at most 3 subagents, one per sub-question. No subagents for the design work itself.
