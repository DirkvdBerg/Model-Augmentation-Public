---
name: figure-design
description: Designs a new figure for Dirk's defense presentation (method diagrams and result plots, standalone TikZ/SVG/matplotlib files): a spec of what the viewer must infer, then about three genuinely different rendered figures from separate contexts for Dirk to choose from, each checked by a fresh reader. Use whenever a figure has no agreed design yet, the user is unsure how to depict something, or earlier attempts were rejected. Not for slide layout, and never for the thesis.
---

# Figure design

## Why this skill exists
Claude kept producing variations of one bad design. Two causes, both measured:
- **Content was never specified.** Briefs listed objects ("window, encoder, state, wave") instead of what the viewer must be able to tell apart, so easier objects were swapped in for the real ones and the comparison the figure existed for disappeared (Codex review, `tasks/handoffs/2026-10-29-codex-ps-figure-review/codex-review.md`, "Causes across briefs and sessions").
- **Concepts were generated in one context.** An LLM fixates on its own earlier output in the conversation, and feeding it previous ideas makes the next ones *less* diverse (Deng et al. 2026; Meincke et al. 2024). Asking for novelty as such buys novelty at the cost of feasibility (Si et al. 2024).

The fix is a written spec first, divergence along a named axis in separate contexts, and a filter against the spec. Style is not this skill's job: `docs/writeup/figure-style.md` (ink, type, scale honesty) and the presentation highlight #0092B5 for the element new in a build.

## Scope
- In: one figure at a time for the defense presentation, as a standalone file plus PNG render.
- Out: slide composition, titles, bullets, placement in the .pptx; anything in `Thesis-writeup/` (do not read or write it except the method sources the user points to, read only).

## Sub-agents
The user allowed **up to 4 fresh-context sub-agents per figure** (2026-10-30), above the general 2-agent cap: typically 3 concept agents (step 4) and 1 fresh reader (step 5). No Workflow tool.

## Procedure
Each step ends with its done-criterion. Write the spec into `spec.md` in the figure's draft folder; it is the single source every sub-agent receives.

### 1. Claim and inferences
- Claim: one sentence the figure proves.
- Inferences I1..In: what the viewer must be able to say or distinguish after looking, in the viewer's words ("the target is built from data and model; the dashed curve is the prediction"). For a comparison, name what is compared with what, and whether item to item or set to set.
- Done: user agreed the claim and the list. If the user does not know yet, propose one claim and its inferences from the method sources, not a design.

### 2. Element map
- Table: element, the thesis/code object it stands for (file:line or equation label), the inference it serves.
- Expressiveness check: the glyph asserts only properties the object has. A wave asserts periodicity, a ring asserts an orbit, four copies assert "four", an arrow asserts causal or temporal flow, a box asserts a separate entity (Mackinlay 1986; Heiser and Tversky 2006; Marshall et al. 2020).
- Claim strength matches the source ("encourages", not "forces").
- Done: every element maps to a real object and an inference; every inference has an element; no substitute objects.

### 3. Channel legend
- One meaning per value of each channel (colour, grey level, line style, arrow type, box style), across the figure (Moody 2009).
- Done: the legend is written in `spec.md`; no value carries two meanings.

### 4. Diverge on a named axis, one fresh context per value
- Pick the axis by figure type and assign one value per concept agent:
  - Method figure, explanatory strategy: one worked example constructed step by step; the dataset/table as the explanation; a contrast or counterexample; before/after; zoom on one sample.
  - Result plot, comparison strategy (Gleicher et al. 2011): overlay on one axis; juxtaposed panels on aligned axes; explicit encoding (plot the difference or ratio itself).
- Each agent gets: `spec.md`, its assigned value, the method sources, the rejection reasons as constraints ("must not imply X", with the user's words). **Never the rejected figures, never another concept, never example figures to copy** (Chrysikou and Weisberg 2005; Wadinambiarachchi et al. 2024).
- Each agent **draws and renders its concept**: the final figure as a standalone file plus PNG, clean and in the house style (`figure-style.md`, thesis figure `Thesis-writeup/Writing/figures/tex/ps2_windows.tex` as the style reference, read only), each build step as its own PNG. It also returns the idea in one sentence and which inferences it serves weakly.
- Done: 3 rendered figures that differ on the axis, not in layout.

### 5. Fresh-reader test of every candidate
- One fresh sub-agent gets only the candidate renders, no spec, no conversation, and writes per figure what it claims and what each build adds. Reading several renders in one context is fine; fixation is a generation effect.
- Compare each with I1..In. Your own preference is not evidence (Hegarty et al. 2012).
- The rejected figures may be given to this reader only, to check the candidates do not repeat their failures.
- Fix inference gaps in the candidates before showing them (the fix stays within that concept; it does not borrow from the others).
- Done: per candidate, the inferences recovered and missed.

### 6. Show the user the candidates side by side
- All candidates in one message (SendUserFile), final build of each plus its build strip, with one line per candidate (its idea) and the reader's result. Add one recommendation with its reason. The user picks, combines or rejects (Dow et al. 2010: parallel then compare).
- On rejection, write the user's reasons into `spec.md` as constraints and rerun step 4 with fresh agents; never revise the rejected figures in place.
- Done: the user's choice is recorded in `spec.md` with their words.

### 7. Finish the chosen figure
- Polish only the chosen one. Builds: each build adds exactly one inference, earlier elements keep their position (segmenting meta-analysis, Rey et al. 2019). The comparison the claim rests on is the largest thing in the figure.
- For result plots use the claim-to-encoding table below.
- Done: compiles clean, style per `figure-style.md`, no em-dashes, and the user approves it.

## Claim-to-encoding table (result plots)
| Claim | Encoding | Source |
|-|-|-|
| A beats B at each point (per position, per frequency) | overlay A and B on one axis, or plot A minus B / A over B with a reference line | Jardine et al. 2020 (read) |
| Size or sign of a difference | plot the difference or ratio itself; never make the viewer subtract two curves | Gleicher 2011; Cleveland and McGill 1984 |
| Sets agree (train/val/test, means and spread) | vertically stacked panels, identical aligned axes; not superposed | Jardine et al. 2020 (read) |
| Recovered within the noise floor (many parameters) | one dot per parameter, normalised error on one shared axis, floor as a band | inference from the two rows above |
| Shape or location over a continuous x (frequency, time) | line; log frequency axis; mark the feature; add values if exact numbers matter | Saket et al. 2019 (read) |
| Any compared value | position on a common aligned scale beats length, angle, area, colour | Cleveland and McGill 1984; Heer and Bostock 2010 |

## Sources
Read in this skill's research (2026-10-30): Jardine et al. 2020 TVCG 10.1109/tvcg.2019.2934786; Saket et al. 2019 TVCG arXiv:1709.08546; Rougier et al. 2014 PLOS CB 10.1371/journal.pcbi.1003833; Marshall et al. arXiv:2008.12566; Meincke et al. arXiv:2402.01727; Deng et al. arXiv:2602.20408; Si et al. arXiv:2409.04109; Doshi and Hauser 2024 Sci. Adv. 10.1126/sciadv.adn5290; Wadinambiarachchi et al. CHI 2024 10.1145/3613904.3642919; Zhang et al. (verbalized sampling) arXiv:2510.01171.
From metadata or recall only: Cleveland and McGill 1984; Heer and Bostock 2010; Gleicher et al. 2011; Mackinlay 1986; Heiser and Tversky 2006; Moody 2009; Rey et al. 2019; Hegarty et al. 2012; Chrysikou and Weisberg 2005; Dow et al. 2010. Fine for guiding Claude; read before citing anywhere.
Evidence limit: all LLM diversity evidence is text ideation; none tested on figures.
