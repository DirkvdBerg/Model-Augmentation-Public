# Critic (examiner lens, Maarten Schoukens), cycle 13, section III

Score: 7/10

## Verdict
On one read, Maarten would see what is built. The physical-units model (eq:aug_transition), the network and its zero start, the encoder and W^b, the residual closed loop, the training coordinates and one problem display with every constraint line are all there. The display states what is trained (vartheta), what is fixed (controller, P, L_b, normalisation) and the cost V. The two PS2 stages are displayed. Criterion B is largely met, and the roadmap and the one-job-per-subsection split follow Hoekstra 2026 Sec. 5.

The section loses points on "why" (criterion D, Maarten's "justify choices") and on algebra load (C). Core choices of III-A and III-B end in \todos even where the source map already gives the reason. The normalisation display is heavier than Hoekstra Eq. 28. PS2 is shown as mechanics without the idea that makes it work. Several displays are followed by runs of sentences or short paragraphs, which turns III-A to III-C back towards formula-plus-remarks blocks (A).

## Issues

1. **Major (D, F). The reason for h_aug = 0 is left as a \todo (draft line ~276(a)) although the source map holds it.** The README source map row III-A lists "kessels2025ai (Remarks 5.1, 5.3: reason for h_aug=0; verified 2026-10-09)". The skill says a reason found in the source-map row is stated in the prose with its source, not left as a candidate. An examiner asks at once why the output map is not learned.
   Rule cause: skill "Reasons and claims" ("Before writing a \todo for a missing reason, search the part's source-map row ...") was not followed. Nothing in the review loop checks each \todo against the source-map row.

2. **Major (D). The header contradicts the draft on the W^b reason.** The CORRECTIONS line says "why model-based rather than data-based: resolved by encoder paper Sec. 4.4 (citation-log)". Yet the encoder paragraph cites the data-based caveat against its own choice ("may be preferred for nonlinear baselines") and ends in a \todo asking for the reason. The reader gets a literature paragraph that argues against the thesis's own choice and then gives no answer.
   Rule cause: skill review item "nothing a \todo asks could have been resolved in this session" has no check that the header's resolved items match the prose. Skill "Justify only what an examiner would question" gives no rule against citing a caveat with no answer next to it.

3. **Major (A, D). PS2 shows the mechanics but not the idea.** The draft says S1 regresses the closed-loop residual of the current model on the encoded added state, and that S2 then regresses g_aug on encoder pairs. No sentence says why this makes the added states carry the missing dynamics: S1 makes x̄ predictive of the future output error (a predictive state, Hefny Sec. 3), and S2 makes g_aug propagate that predictive state. Maarten's PS2 objection was exactly that "the model was not clear". Without this link he still sees a procedure, not a construction with a consequence.
   Rule cause: the README "Example (PS2, sketch agreed 2026-10-08)" fixes the sentence jobs (problem, lead-in, what S1 trains, S2, S3, unchanged), but none of them is the consequence of the construction. The rule "problem, construction, consequence" (README "Prose around the math") is overridden in practice by the fixed six-sentence sketch.

4. **Minor (C). The normalisation display eq:aug_norm is overloaded.** It has five lines, one with a manual \hspace break that writes out the conjugation of f_base, plus h^n_base. Three sentences follow the "where" clause: the data source of the states, the domains of the normalised maps, and "μ_y = P^T[I 0]μ_x". The last is an intermediate step. Hoekstra Eq. 28 is one line. An examiner wants the scalings and which signals are centred, not the conjugated maps.
   Rule cause: README "Math standard" item 4 ("one line shows how each physical-unit map acts on them") makes writers display every map's conjugation. The skill review item ("states the invertibility of every matrix the display inverts") adds the filler about positive standard deviations and invertible scalings.

5. **Minor (B, F). The encoder display contradicts its own prose.** Eq. aug_encoder applies [W^b; W^a] to y^n and u^n, which eq:aug_norm defines as centred. The next sentences say "The linear maps act on the scaled, uncentred histories. The term T_x μ_x is subtracted from their physical rows." As written, the display does not compute what the code computes. These are also two sentences after the "where" clause.
   Rule cause: README "Math standard" item 4's carve-out ("for the encoder's history maps, stated in the where clause rather than as offset terms") allows a display that is literally false for the symbols it uses. It should require a scaled-only history symbol, or the offset written in the display.

6. **Minor (A, D). III-C runs short paragraphs after the residual-loop display, and the step-order choice sits after the display.** After eq:aug_residual_loop come the figure and three short paragraphs: x^c_τ = 0, the step order with no algebraic loop, and "a closed-loop fit can hide plant error". The step order fixes an element of the display, so it belongs in the lead-in. The plant-error sentence is a separate point placed at the subsection's end.
   Rule cause: README "Prose around the math" ("A choice that fixes an element of a display (... step order ...) is argued in that display's lead-in"; "never ... a run of short paragraphs after it") was not followed. The skill review "Count the sentences between a display and the next paragraph break" counts only up to the paragraph break, so this run of paragraphs escapes it.

7. **Minor (A). III-C opens with a literature block before its construction.** The first two paragraphs (about 14 sentences) cite Hoekstra Eq. 22, Kessels twice, Forssell twice, Hoekstra Sec. 5.5, Beintema Cond. 1 and Kessels Ch. 5 before any controller equation. The argument (feedback data, free integrators make open-loop consistency conditions fail) is right and is MUST ESTABLISH item 12. The positioning is still a block, not one sentence per method at the component.
   Rule cause: skill "Order" ("Literature positioning sits at the component it contrasts, ..., never as a block before the construction") conflicts with an agreed MUST ESTABLISH item that lists four sources for one motivation. No rule says how to compress such an item.

8. **Minor (E). The same well-posedness fact is stated three times.** Line ~274 ("M(Y) is invertible for every Y when θ_base satisfies (lfr_admissibility)"), the well-posedness paragraph of III-A ("which (lfr_admissibility) ensures for every Y"), and III-D ("Every ξ gives M(Y) ≻ 0 for every Y, so the LFR is well posed"). The O_n full-rank fact is also stated twice: after eq:aug_wb, then re-derived in the next paragraph.
   Rule cause: the skill review item requiring the "where" clause to state "the invertibility of every matrix the display inverts" produces the first copy. The second then breaks "Say each fact once", and the review does not check the two against each other.

9. **Minor (A). Paragraph topics in III-A are mixed.** The paragraph that opens "the correction acts on every physical row" warns that a persistent correction accumulates in position, then switches to the zero output layer and its display. The accumulation risk is left without a consequence. The network is described in words one paragraph earlier ("One feedforward network ... supplies both learned maps") and displayed only here, so its description and its display are split.
   Rule cause: skill "Topic sentences" ("every later sentence of a paragraph serves its first") and README "Prose around the math" (a display's reasoning sits in its lead-in) were not applied. Nothing tells the writer that a construction described in words must be the lead-in of its display.

10. **Minor (D). The observability argument for W^b follows the display.** "The map exists although the baseline is marginally stable" is a condition the display needs (O_n left-invertible), but it comes after eq:aug_wb and its "where" clause.
    Rule cause: skill review "No reason for a display's inputs arrives after a later display; a choice that fixes an element of a display sits in its lead-in" covers choices. It does not explicitly cover an existence condition the display relies on.

11. **Minor (D, E). The \todo load hides the argument.** About 17 \todos, most on core modelling choices of III-A and III-B: h_aug = 0, the correction after RK4, the position rows, an unrestricted vs LPV-structured network, centring, Y = 0, nominal W^b. Most are honest open decisions. Together, Maarten reading the draft build sees "justify choices" unanswered in the first two subsections. Issues 1 and 2 show that at least two could have been resolved.
    Rule cause: skill "Open points are \todo{}s" says to resolve everything possible first, but no review step re-checks each \todo against the source map, the citation log and the header corrections before delivery.

## Not counted against the draft
- Length (about 3.5 pages against 1.55) is explained in the header against the required displays (skill "Length").
- The problem display repeats the constraint lines, as README Math standard item 3 requires.
- The oracle polish rule is named and has a \todo, as the skill requires.
