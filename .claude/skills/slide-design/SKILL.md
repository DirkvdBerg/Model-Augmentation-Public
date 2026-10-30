---
name: slide-design
description: Designs and builds slides for Dirk's MSc defense (20 min, TU/e template, python-pptx). Use for any request to design, build, redesign or fix a slide or figure of the final presentation. The method is Dirk's V2 approach (stepwise builds, one diagram per slide) drawn to Jan Hoekstra's quality, with a render-and-critique gate before anything is shown.
---

# Slide design for the defense

## The method in one line
**V2's approach, drawn to Jan's quality.** Dirk's control-group deck (V2) has the right approach: claim titles, one diagram per slide built element by element, glyphs reused across slides, few words. Its rendering looks AI-generated. Jan Hoekstra's decks have the drawing quality: large diagrams, black outlines, LaTeX math, the equation beside the diagram, question boxes. Every slide takes the first from V2 and the second from Jan.

Dirk's words: "i do like how we approach it and the style with the clear stepwise explanation" (V2); "im not a fan of just copying him" (Jan: take his principles, not his layouts).

## Sources (read before designing)
- Story and per-slide content: `Final thesis Presentation/storyboard.md` (in `Graduation Project/`, one level above this repo). Section 2 holds the agreed arc; a slide without an agreed line there is not designed, ask Dirk first.
- Content and notation: the thesis, `Thesis-writeup/Writing/sections/` and `Thesis-writeup/Results/24-10-2026/CSV/`. Read only, never compile.
- References, rendered: `references/` next to this file (list below). Full decks: `Final thesis Presentation/Example presentations - Jan/` and `Control group presentation/...V2.pptx`.
- Template: `Final thesis Presentation/TUe PowerPoint 16x9 template/TUe_16x9_ppt_with_instructions.potx`, layout "Text slide", warm grey background (Dirk's choice).
- Code base for template loading, footer, bullets, question box: `Final thesis Presentation/drafts-2026-10-28/scripts/build_s3_frf.py`. LaTeX math as PNG: a `.tex` file plus `drafts-2026-10-28/scripts/pdf2png.py`.

## References and what each teaches
| File | Take |
|-|-|
| `v2_22`, `v2_23`, `v2_24` | Build one element per step: state stack, then the step block, then the next state. Elements keep their position. |
| `v2_25` to `v2_27` | The diagram grows: network added, then learned states, new parts in the accent. |
| `v2_29`, `v2_30` | A concept (encoder) explained as a picture on a time axis, "?" boxes filled in the next step. |
| `v2_13` to `v2_15` | An equation with arrows to its terms, built term by term. |
| `v2_38` to `v2_40` | A claim shown as a picture ("same shift"), not as text. |
| `jan_benelux_03` | Short bullets left, one diagram right, question box as the hook. |
| `jan_benelux_04` | The map: the whole approach on one slide, open questions boxed on it. |
| `jan_benelux_07` | Three views: application photo, block diagram, equation, same symbols. |
| `jan_benelux_11`, `_12` | One small glyph chained over time to explain simulation and windows. |
| `jan_benelux_14` | Encoder: past window into a bracket, start state out. |
| `jan_ecc_15` | System above baseline, the missing part in red with a pointer. |
| `jan_ecc_18` | A plot annotated with a circle and a short grey note. |
| `bad_s2_frf`, `bad_intro_boxes`, `bad_intro_map` | Rejected by Dirk ("ugly", "so much text", "ai"). See the failure list. |

**V2 rendering to fix, always:** white rounded card on the grey background (drop it, draw on the slide); pastel orange and blue fills with coloured outlines (black outlines, white fill); diagrams at a third of the slide with tiny text (make them fill the slide).

## Process
Each step has a done-criterion for each goal: **clear** (one message, understood without reading), **clean** (the rules below), **faithful** (thesis content and notation). A step is done only when all three hold. Write the criteria check into your reply as a short pass/fail list.

1. **Card.** Copy the slide's line from storyboard section 2: what the committee understands after it.
   - Done: one sentence, a claim; if none is agreed, stop and ask.
2. **Pick the references.** Name the V2 slide whose approach it follows and the Jan slide whose drawing quality it matches, and what is taken from each. Open both PNGs.
   - Done: two references named with one line each; if no V2 counterpart exists, say so and name the closest pattern.
3. **Build sequence, in text first.** One line per click, each adding exactly one element; under it the sentence Dirk says at that click (goes to the notes).
   - Clear: the last step shows the claim of step 1. Clean: every step adds one thing; on-screen prose at most about 15 words in total (labels and math not counted). Faithful: every symbol and number traced to a thesis file.
4. **Draw.** Native PowerPoint shapes, plots from matplotlib at slide size, math as LaTeX PNG at the slide font size. Duplicated slides for builds, same title, identical positions.
5. **Render** with `powershell -File .claude/skills/slide-design/render.ps1 <pptx> <outdir>` and open every build step PNG at full size.
6. **Critique the render**, against every rule and every item of the failure list, as a pass/fail list. Fix and re-render until all pass. Compare it next to its two reference PNGs: would it sit in Jan's deck without standing out?
   - Done: all pass, written out. Never show a slide that fails a rule; if a rule cannot be met, say which and why.
7. **Show** Dirk the last build step as an image, plus the build steps as a strip, with one line on what it shows. Output in a new dated folder `Final thesis Presentation/drafts-YYYY-MM-DD/`, never overwriting an earlier draft. Record Dirk's verdict in storyboard.md the same turn.

## Rules (checkable)
**Layout**
- One focal diagram or plot per slide, at least half of the area under the title.
- No panel or card behind figures. No quarter of the slide left empty without reason.
- Elements on a common grid; a build never moves an element that is already on screen.

**Text**
- Title = the claim, at most about 8 words, sentence case.
- On-screen prose at most about 15 words; the explanation goes in the notes.
- Calibri, body at least 18 pt, diagram labels at least 14 pt; no small caps, no coloured label words, no symmetric word pairs ("Accurate / Interpretable").
- No em-dashes anywhere, notes included.

**Colour**
- Black lines, white or no fill, greys for context and for parts no longer in focus (V2 "putting it together" greying).
- One accent, TU/e blue 0092B5, only for what is new in this step. Nothing else coloured except the template's logo.
- Question box (Jan's pattern): one per part end, the question the next part answers; fill colour still open (?), ask Dirk on first use.

**Plots**
- Drawn at slide size, transparent background, labels on the lines, no legend, no grid.
- Only the axis range the claim needs; the eye must land on the point of the claim (annotate it as `jan_ecc_18`), and every visible feature must be explained or cut.
- Errors as NRMS_e in %, frequencies in Hz, never metres or plant gain.

**Math**
- Symbols identical to the thesis. One key equation per concept, beside or under the diagram; each term appears with its block.
- No derivations on main slides; they go to backup.

**Content**
- Jan's method (augmentation structure, encoder, truncated loss) is background: short. Time goes to Dirk's additions: the gantry (MIMO, LPV), closed-loop training with the known controller, added states with PS, the OBC extension.
- Only the current method; never superseded approaches.

## Failure list (rejected before, check every render against it)
- Bullet list beside a figure carrying the message in text (bad_s2_frf: eight bullets).
- A plot whose most visible feature is not the point (the unexplained 150 Hz dip drew the eye, not the 212 Hz peak).
- Several accents competing: ring, coloured label and coloured box on one slide.
- Icon-sized pictures in a large empty field (bad_intro_boxes).
- A heading plus bullet hierarchy: reads as a document, not a slide.
- A symbolic picture that shows nothing ("augmentation" as two glued rectangles).
- Thin coloured lines with no meaning; a small diagram in the middle of grey (bad_intro_map).
- Generic glyphs (stock network icon, m-c-k as text) where a real object or real behaviour can be shown.
- V2 figures pasted, or thesis figures scaled down.
- Showing chat wireframes or mock-ups instead of rendered slides in the real template.
