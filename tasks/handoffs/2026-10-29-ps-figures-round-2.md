# Handoff: draw the three PS method slides (R, S1, S2) for the defense, one slide at a time, against a fixed spec
**From**: session of 2026-10-29 | **Branch**: Augmentation | **Effort suggested**: high (design judgement plus a method that is easy to misdraw; no derivations, no runs)

## 1. Task
Draw the build sequences of three defense slides that explain PS, the predictive-state initialisation, as standalone TikZ figures. The slides are:
- **R:** what the residual is.
- **S1:** residual regression trains the encoder's added states through a temporary decoder.
- **S2:** one-step regression of the added-state transition.

The acceptance spec is criteria C1 to C11 of Codex's review (`tasks/handoffs/2026-10-29-codex-ps-figure-review/codex-review.md`, section B).

Work one slide at a time, starting with R:
1. Render every build.
2. Look at the renders yourself against the spec and against Dirk's reactions (section 6).
3. Show Dirk the slide and stop for his verdict before starting the next.

Four days of attempts failed. This round succeeds only if each build shows the actual training operation at a readable scale. A new metaphor does not count.

## 2. Out of scope
- **Any new concept search, literature search, metaphor, or "N different approaches" exercise.** That was done twice; see section 6.
- **The closing three-stage card slide.** Dirk approved it (`Stages.dc.html` on https://claude.ai/artifact/SxhsLgd6oVJmfA5o7aNYvp); leave it as is.
- **Slides 14b-1 and 14b-3** (spectra without and with PS). Already agreed.
- **Placing figures in the .pptx deck.** The deck session does that once Dirk approves.
- **A recorded-run animation of PS.** An open idea in the storyboard, not decided.
- **Editing anything** in `Thesis-writeup/`, `Clean/`, `kamtin-fp-model/`, the storyboard, or the review package `tasks/handoffs/2026-10-29-codex-ps-figure-review/`. All read only.
- **Making real PS checkpoints or training curves.** No runs.

## 3. Where things stand
- Branch `Augmentation`; the tree is dirty from other sessions; commit nothing (standing rule). No run in flight.
- No approved figure exists for R, S1 or S2.
- The storyboard block "PS method in three slides (agreed with the user 2026-10-28)" fixes the slide order R, S1, S2 and the closing card. Its build lists were written by Claude and are not approved designs.

## 4. Established and verified
Checked on 2026-10-29 against the code and the thesis. Build on these without re-deriving them.
- **Window.** χ_τ is the recorded (u, y) at τ−30 … τ−1. The encoder never sees y_τ.
  - Evidence: `Clean/model_augmentation/fit_systems/ps2_opening.py:186`: `u[k - nb:k]`, `y[k - na:k]`.
  - D-250.
- **Residual target.**
  - The current model is simulated in closed loop from x̂_τ|τ = ψ(χ_τ) over H = 30 samples: `ps2_opening.py:211`, `fs.simulate(x, uf, yf, ctrl_ix=...)`.
  - The loop input is the recorded force plus the feedback-controller correction on y − ŷ (thesis `eq:aug_residual_loop`; comment at `03_augmentation.tex:1904`). It is not a replay of the recorded u.
  - e_τ = (y − ŷ)/σ_e, with σ_e the RMS over the batch (`ps2_opening.py:213-215`).
  - The target is computed under `no_grad`, so it is fixed within the S1 step.
- **S1.**
  - The decoder p is `Linear(n_x̄, width) → tanh → Linear(width, H·n_y)`. Its last layer starts at zero, so the first prediction is zero (`ps2_opening.py:156-158`).
  - Its input is the added states only (`:237`).
  - S1 updates θ_p and the encoder's added-state parameters W^a and the last encoder layer. Only the added-state rows receive gradient (`:160`, `:239`).
  - S1 runs one step after every Adam step on V (`:98-118`; thesis `eq:ps2_alt`, near `03_augmentation.tex:1825`). So the residual is always the current model's.
- **S2.**
  - Runs once, after S1 ends.
  - Pairs: (x̂_k|k = ψ(χ_k), recorded u_k, x̄_{k+1|k+1} = ψ_a(χ_{k+1})), computed once with the encoder frozen (`:280-287`).
  - The network gets the full state and u: `ann(torch.cat((X, U)))`. Only its added-state output columns enter the loss (`:293`).
  - V is not minimised during S2 (`03_augmentation.tex:1870`).
  - The shared hidden layers also change f_aug (`:1943-1945`).
  - Thesis cost: `eq:ps2_s2` (`03_augmentation.tex:1881-1890`), a squared error divided by the target variance over the minibatch.
- **Hand-over.** p is dropped and V continues from the PS parameters (`eq:ps2_handover`, `:1949-1959`). "PS does not change what is minimised."
- **Strength of the claim.** The thesis says the decoder "encourages" the encoder to put predictive information in x̄; it was deliberately changed from "forces" (`03_augmentation.tex:1753`). The method does not establish that x̄ equals the absorber coordinates.
- **Timescale.** Ts = 0.25 ms, so H = 30 samples = 7.5 ms, about 1.6 periods of 212 Hz.

## 5. Assumed but not verified
- **Dirk has not judged Codex's concepts or attempt 03b.** Codex recommends concept 1 and calls 03b the strongest earlier attempt. Dirk's verdict on 03b was never recorded, and he has not seen Codex's sketches with a decision. The starting points in section 9 are therefore proposals. Settled by Dirk's verdict on R and his answers to section 8.
- **Slide equation for S2.**
  - Dirk asked for "clear formulas (dont have to go into detail)".
  - Whether that means the thesis quotient or the numerator ‖g_aug(x̂_τ|τ, u_τ) − ψ_a(χ_{τ+1})‖² with the normalisation in speaker notes is his call (section 8).
  - Codex's extra weight w_B on the slide is not settled either way.
- **The 196 Hz claim in the 28 Oct handoff and the storyboard** ("dominant FFT bin lands at 196 Hz") is inconsistent with 30 samples at 4 kHz (bin spacing 133 Hz). Unresolved. Put no frequency label on any 30-sample curve.
- **Renders.** Codex compiled TikZ in this repo (`sketches/*.tex`), so a LaTeX toolchain exists. Which engine and command it used is not recorded; check `generate-sketches.py` before your first compile.

## 6. Tried and failed
Full per-figure diagnosis: `codex-review.md` section A. Mechanisms, and what Dirk said:
- **Object checklist instead of the operation.** Window, encoder, state, decoder and wave were drawn as labelled things, but the figures never showed what is compared with what.
  - Attempts 01 and 02.
  - Dirk: "they dont actually describe method that well … we need to make more clear what each step actually does".
  - Mechanism: one blue wave with an arrow from p into it. There is no target to see as distinct from the prediction.
- **Residual as a free-standing wave.** The gap y − ŷ was never constructed, or ŷ was nearly a straight line.
  - Dirk: "do you not see how the slides i referenced made the residual plot?" and "why for theresidual are you basicly drawing a straight line".
  - Meeting slide 13 (`tasks/handoffs/2026-10-29-codex-ps-figure-review/reference/meeting-08-10-pages/p29.png`) is the reference he endorses: one time axis split at τ, y continuing, ŷ from τ, the gap visible.
- **S2 hidden in a tiny arrow under a large signal.**
  - The visually dominant curve (y) was not the variable being predicted.
  - Dirk: "we dont clearly show the one step prediction, how i see a one step prediction is we have our samples and then we want to predict the next sample".
- **Easier objects swapped in for the real ones.** State ring, phasor, pole pair at 212 Hz, loss landscape, training time-lapse, Hankel residual block (attempt 04, artboards 02, 04, 08, 09, 14, 15).
  - Each makes a claim the method does not support (`codex-review.md`, attempt 04 table).
  - Dirk: "claude is currently failing badly each time".
- **Novelty quota.** "15 genuinely different approaches" and "2 or 3 genuinely different concepts" rewarded new silhouettes (bowtie, strip, panels) for the same unexplained content.
  - Dirk: "youre doing the same one with just a different bit each time".
- **Too much and too little at once.** Labels, legends, captions and conclusion sentences were added while the comparison itself was missing.
  - Dirk: "it is simultaniously too much and too less at the same time" and "way too messy".
- **Block diagrams.** Boxes and wiring with words doing the work. Codex's own concept-1 S2 page has this problem (a g_aug box, "one sample ahead", "compare").
  - Dirk: "I dont like that block with past samples".
- **Math missing, then math dumped.** Maarten did not see the math in the 08-10 meeting. Dirk: "the visualization should carry the explanation. But the math is also important".
- **Overclaiming in the briefs.** "Must carry the missing dynamics" was carried from brief to brief and into figure captions.
- **Brief told the session not to look at the failures** (28 Oct handoff section 12). The next session repeated the same mistakes without seeing them.

## 7. Achieved
- The method facts in section 4, checked against code and thesis.
- Codex's criteria C1 to C11, accepted here as the spec (`codex-review.md` section B).
- Codex's sketches, compiled:
  - `tasks/handoffs/2026-10-29-codex-ps-figure-review/sketches/ps-concepts.pdf` (three concept pairs);
  - `concept-1-builds.pdf` (8 + 8 builds);
  - `ps-concepts.tex`, `generate-sketches.py` (sources).
  - These are review sketches, not approved slides.
- Attempt 03b renders, reconstructed by Codex: `review-evidence/group-03b-0.png` (S1 builds) and `group-03b-4.png` (S2 time-axis builds).
- Approved: the closing card slide (section 2). Nothing else.

## 8. The open question
Two decisions are Dirk's. Put both to him in the same message as the R renders:
1. **Starting design for S1 and S2.** Candidates:
   - (a) Codex concept 1, redrawn without its captions and block-diagram S2;
   - (b) attempt 03b S1 builds 2 to 4 (residual on its own zero line, ψ_a triangle, p, dashed prediction), with the overclaiming caption removed;
   - (c) either one with Codex concept 3's two-history build added as the "why" step of S1.

   His verdict on 03b and on concept 1 decides. Ask him to look at `group-03b-0.png` and `ps-concepts.pdf` pages 1, 2 and 5.
2. **S2 equation on the slide.** The thesis quotient, or the numerator with the normalisation in speaker notes.

Until he answers, draw only R.

## 9. Next action
Draw slide R as TikZ builds, render each to PNG, and show Dirk the builds with the two questions of section 8.

R builds, one new element each (a proposal; keep the positions fixed across builds):
1. Time axis with the split at τ; measured output y_k as dots, one per sample (draw the sample pitch so one sample is visible; mark the window length 30 in a label, not with 30 dots if they become unreadable).
2. ŷ_{k|τ} of the current model from τ onward: smooth, with its own dynamics (not a straight line or a flat trend). It may start on or near y_τ.
3. The residual as vertical sticks between y_k and ŷ_{k|τ} for k = τ … τ+H−1 (meeting slide 13 pattern).
4. The equation e_τ = [y_τ − ŷ_{τ|τ}, …, y_{τ+H−1} − ŷ_{τ+H−1|τ}], under the figure.

Curves:
- Schematic and smooth, in normalised units: Dirk allowed concept curves and asked for smoother sine waves.
- The residual must oscillate visibly over the 30 samples. Do not label a frequency.
- Do not draw u as the force driving ŷ: the rollout is closed loop with the controller.

Write the figures to `../Final thesis Presentation/drafts-2026-10-28/ps-round2/`, relative to the repository's parent folder: `R_build1.png` … and `R.tex`.

## 10. Acceptance criterion
Done when Dirk approves R, S1 and S2, each as a build sequence, in his words. There is no numeric target.

Dirk's own test (28 Oct): "you fail at these". Each answer must be visible in the figure, not in a caption:
- "what is the residual": R.
- "how we obtain the data with the decoder": S1, the past window to x̄ to p, against the residual of that same τ.
- "how do we train the encoder": S1's backward arrow.
- "what is this one sample difference with the encoder": S2.
- "we want to predict the next data point": S2.

The criteria below are what you hold your own renders to before showing them:
- **Every build adds exactly one element.** The highlight #0092B5 marks only that build's new element; earlier elements return to black or grey.
- **Target and prediction are visibly separate objects** in S1 and in S2 (C4). The comparison between them is the largest thing on the slide.
- **S2 shows the state as points, one per sample.** The prediction lands next to the next point. x̂_τ|τ (full state) and u_τ visibly enter the prediction (C8). The one-sample shift is visible at the drawn scale without a caption (C7).
- **S1 shows the learning route.** A dashed backward arrow from the comparison through p to ψ_a, and the minimisation variables θ_p, θ^a_enc in the equation (C5). p is visibly temporary.
- **Text on the slide:**
  - symbols on elements, plus one equation line per build under the figure;
  - no captions, footnotes, legends or conclusion sentences;
  - no "must", "forces", "212 Hz" or "absorber".

  Qualifications go in a speaker-notes file next to the figures: closed loop, fixed targets, alternation with V, V not minimised in S2, shared layers.
- **Notation exactly as in the thesis:** χ_τ, ψ, ψ_a, x̂_τ|τ, x̄_τ|τ, p, e_τ, H, g_aug, θ_p, θ^a_enc, θ_aug. Mark normalised quantities once in the notes, not on the slide.
- **Style:**
  - per `docs/writeup/figure-style.md` and the thesis figure `Thesis-writeup/Writing/figures/tex/ps2_windows.tex`;
  - black on white, greys carry meaning, LaTeX math;
  - 16:9, sized for a content area of about 21.5 × 9 cm;
  - no em-dashes.

Positive example, in words, of the S1 final build:
- top: the R picture (y dots, ŷ, sticks), reused as is;
- below τ, on its own zero line aligned with the future interval: the residual e_τ drawn as black dots joined thinly;
- left of it, the past window bracket, a ψ_a arrow to a small x̄_τ|τ node;
- a dashed p arrow from x̄_τ|τ that ends in a dashed blue curve over the residual dots;
- a dashed backward arrow from that pair to ψ_a;
- under all of it, one line: min over θ_p, θ^a_enc of Σ_τ ‖p(ψ_a(χ_τ)) − e_τ‖².

Nothing else is on the slide.

## 11. Read these first
1. `tasks/handoffs/2026-10-29-codex-ps-figure-review/codex-review.md`: the diagnosis (section A) and the spec C1 to C11 (section B). Read all of it.
2. `tasks/handoffs/2026-10-29-codex-ps-figure-review.md` section 3: Dirk's verbatim reactions, the ground truth of what failed.
3. `tasks/handoffs/2026-10-29-codex-ps-figure-review/reference/meeting-08-10-pages/p29.png` (slide 13, the residual Dirk endorses), and `review-evidence/group-03b-0.png` and `group-03b-4.png`.
4. `tasks/handoffs/2026-10-29-codex-ps-figure-review/sketches/ps-concepts.pdf` pages 1, 2, 5, 6.
5. `Thesis-writeup/Writing/sections/03_augmentation.tex` lines 1522 to 1965 (the method), read only.

## 12. Do not
- **Do not propose new metaphors or concept families**, and do not offer a menu of designs (section 6).
- **Do not draw these objects:** a state ring or phasor, poles, loss landscapes, training time-lapses, Hankel residual blocks, a box filled with "past samples", or block diagrams whose meaning is in words.
- **Do not draw x̄ as a measured signal** or as physical coordinates. If a state trace is drawn, it is encoder output, introduced as such in the build that adds it.
- **Do not show S1, S2 and the hand-over on one slide.**
- **Do not start S1 or S2** before Dirk has given his verdict on R.
- **Do not imitate Dirk's V2 deck figures**, and do not reuse I-JEPA or bowtie layouts (rejected).

## 13. Operational
- **Toolchain:** TikZ standalone documents, one per build (or one file with a build counter), compiled to PDF and converted to PNG for viewing.
  - Codex's `tasks/handoffs/2026-10-29-codex-ps-figure-review/generate-sketches.py` and `sketches/ps-concepts.tex` show a working setup in this environment. Reuse its preamble, not its layouts.
- **Python:** `conda run -n GraduationProject python <file>` (multi-line snippets in a file).
- **Showing renders:** send the PNGs to Dirk with SendUserFile, all builds of one slide in one message.
- **No runs and no data processing** beyond schematic curves.

## 14. Delegation
None. The work is one design thread that needs the full context; no subagents.
