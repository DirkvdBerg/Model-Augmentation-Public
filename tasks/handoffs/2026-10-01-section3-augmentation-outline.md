# Handoff: outline thesis Section III (Dynamic LPV-LFR Augmentation) so later sections can reference its definitions
**From**: session of 2026-10-01 | **Branch**: Augmentation | **Effort suggested**: high (reading and structuring across papers, decisions and code; no derivations)

## 1. Task
Run stage 1 of the README "structure" phase for `Thesis-writeup/Writing/sections/03_augmentation.tex`: a concise outline, checked for coverage and sources, and nothing else. The thesis writing instructions in `Thesis-writeup/Writing/README.md` govern this work; follow its "Writing a section" procedure (source order, six-question brief), "Derivation policy" (choice records for nontrivial choices), "Mandatory methodological reading", and "Reference verification gate".

Return exactly four things, in this order:
1. The six-question section brief (README "Writing a section"), one or two lines per question.
2. The outline: subsection headers plus one-line bullets, one claim per bullet, open points marked (?). Format example (match it):
   ```
   ### III-A Augmented state equation
   - Dynamic parallel structure: f_base + f_aug on x̃, g_aug on x̄ (Hoekstra 2025, Table 1)
   - Output reads x̃ only, h_aug = 0 (?)
   ```
   No meta layers, no label blocks, no restatement of the brief.
3. The coverage table (section 4, "Coverage").
4. The citation table (section 4, "Citations").

Then stop and wait. Stage 2 (claim lists per subsection, the displayed equations with their single job, and the two figure concepts) starts only after the user approves stage 1.

## 2. Out of scope
- Stage 2 and any prose in 03_augmentation.tex. Do not edit the file unless the user asks.
- Section IV (`04_obc.tex`): the user is rewriting it in phase 3. Read only.
- Drawing figures in TikZ.
- Adding `refs.bib` entries (flag missing ones; the README reference gate applies).
- Changing `.vscode/settings.json`, `main.tex`, `.latexmkrc`.
- Compiling (section 12).

## 3. Where things stand
Branch `Augmentation`, dirty tree. `03_augmentation.tex` holds a bullet outline from an earlier pass (four subsections: parallel augmentation with added states; encoder and initialisation; closed-loop rollout objective; why augmentation) and a WRITING GUIDE header. Section II is prose (`02_system_baseline.tex`, notation x̃ and θ_base). Section IV is prose being rewritten by the user. Setup, results and training design live in `Thesis-writeup/Documentation/{TRAINING,DATA,RESULTS}-DESIGN.md`. No runs in flight.

## 4. Established and verified (the user's standpoints; build on these)

**Writing instructions.** `Thesis-writeup/Writing/README.md` is the authority: source order, six-question brief, phases (this is phase 1, structure), authorial-voice rules, derivation policy, mandatory reading, reference gate, figure pipeline. In addition, the user's own standpoints: one subsection at a time; outline approved before claim lists, claim lists approved before prose; formulas displayed, never compressed into prose to save space; no word budgets, but concise and to the point; display the working formulas of methods used (inherited ones cited, no derivations) and do not repeat displays owned by other sections; one claim per sentence.

**Coverage.** Before outlining, build an inventory of what Section III must cover, from: the WRITING GUIDE in `03_augmentation.tex` ("Purpose", "Choices to justify", "Horizon check", "Section is finished when"); research-plan Aspect 2 (`docs/Thesis-documentation/Research_plan___Graduation_project_AI_ES___Feedback_Processed/Sections/Aspects/2.tex`); the matching sections of `docs/Thesis-documentation/Meeting-audit/candidate-thesis-impact.md` and themes 4 to 6 of `thematic-thesis-audit.md`; the newest relevant entries in `docs/decisions.md`; and the Section IV needs below. Read Sections II, IV and V in full, including their "Owns:" headers, so boundaries come from the files. Return the inventory as:

| Required item | Source | Covered by (bullet) or owned by (Sec. II / IV / V / VI / appendix) |
|-|-|-|

Every row has a bullet or a named owner; an empty cell is a gap, a row claimed twice is a duplicate.

**Citations.** For every bullet that rests on literature, read the passage in the PDF itself (not summaries, decisions or memory) and return:

| Bullet | Cite key | Exact location (Eq., Sec., page) | Passage read in PDF | In `refs.bib` |
|-|-|-|-|-|

A bullet with no source is tagged "ours". Reading list (README mandatory reading):
- Hoekstra et al. 2025 EJC, `literature/augmentation/hoekstra2025_lfr-augmentation-ejc.pdf`: the main inspiration (LFR interconnection Fig. 1, Table 1 "dynamic parallel", Eqs. 2 to 4, normalization Eq. 9, SUBNET encoder Fig. 2, truncated loss Eq. 5). Check whether `refs.bib` has a key for it; `hoekstra2026lfr` is a different paper.
- Hoekstra et al. 2026, `literature/closed-loop-id/hoekstra2026_lfr-augmentation-fp-models.pdf`, key `hoekstra2026lfr` (arXiv 2602.17297): extended first-principles formulation, identification procedure, initialization.
- Encoder initialization, `literature/augmentation/Encoder initialisation methods in the model augmentation setting.pdf` (key `hoekstra2026encoder` is a placeholder in `refs.bib`).
- Kessels, `literature/augmentation/kessels2025_ai-control.pdf`, key `kessels2025ai`: reference where useful: closed-loop-trained augmentation on an industrial motion system, and direct closed-loop augmentation as the fallback for nonlinear controllers. Known limits recorded in `01_introduction.tex` todos: no LFR/LPV, frozen FP parameters, no orthogonality; the thesis names a wire bonder, not ASMPT.
- Drenth, `literature/lpv-lfr/drenth2025_lpv-lfr-rational.pdf`, only where the LPV-LFR form of Section II is referenced.

**Notation** (user decision; Hoekstra 2025 Table 1 and Eqs. 2 to 4; same symbols as `docs/writeup/README.md` and Section IV):
- x̃ ∈ R^6 physical state col(q, q̇) (Sec. II), x̄ ∈ R^{n_a} added states, x̂ = col(x̃, x̄).
- f_base(θ_base, x̃, u): one-step RK4 transition of the Section II model, normalized; f_aug(θ_aug, x̂, u): ANN write into the physical rows; g_aug(θ_aug, x̂, u): ANN write into the added rows; y = C_y x̃ (h_aug = 0).
- θ_base = the ten identifiable combinations (Sec. II-C), θ_aug = ANN weights. Do not use φ (Györök's regressor, Hoekstra's component functions), η, ρ, J, S for these objects.
- u = P u_act: the code feeds the recorded actuator force and the physical block applies P internally (`model_augmentation/fit_systems/blocks.py` lines 828 and 916).

**Ownership** (agreed): Section III owns which states exist, what the ANN reads (x̂, u) and writes (f_aug into all six physical rows, g_aug into the n_a added rows), that the output reads x̃ only, the continuous/discrete boundary (RK4 at T_s), normalization, joint estimation of θ_base and θ_aug, the encoder, and the closed-loop rollout objective with û = u + C_fb(y − ŷ). Section V owns numbers (n_a ∈ {0, 2, 8}, 16 x 2 tanh network, encoder lag 29, window 0.1 s, budgets; `TRAINING-DESIGN.md`). Section IV owns only the OBC correction on the f_aug part.

**Section IV needs from Section III** (it currently re-explains these): the augmented state equation as one labelled display; the symbols above; f_base as the normalized RK4 map of Sec. II; the normalization (Hoekstra Eq. 9); joint estimation; the closed-loop multistep objective; u = P u_act.

**Figures** (stage 2, but the outline must place them):
1. Overview figure, adapted from `docs/writeup/jan-blockscheme-v5.tex` (and v4): remove the blue A_aa "memory" and B_u "drive" blocks (leftover of the D-150 bypass experiment); the router writes all six physical rows plus the n_a added rows (thesis arms D-222; D-103: X and Y always included), not only X, Θ, Y as in v5; one shaded continuous-time region (the LPV-LFR G and Δ(Y) loop of Section II, Fig. `lfr`, giving the derivative of x̃), one RK4 block at T_s as the boundary producing f_base, everything else discrete (ANN, router, additions, state register labelled x̂_{k+1} to x̂_k so it is not read as a plant delay, encoder ψ, output); controller not in this figure. Visual language after Hoekstra 2025 Fig. 1 and `docs/writeup/figure-style.md`; README "Figure and table pipeline".
2. Closed-loop figure: reuse `docs/writeup/closed-loop-form-v4.tex` (caption in `docs/writeup/README.md`) in the closed-loop objective subsection, simplified so the residual rollout equation is the visual conclusion.

## 5. Assumed but not verified
- That D-129, D-140, D-141, D-150, D-151, D-160, D-178, D-180 (named in the WRITING GUIDE) are still final: several are superseded (D-150 bypass gone; windows and burn-in revised by D-220/D-222). Check newest-first; the later decision wins.
- That `jan-blockscheme-v5` matches the final interconnection apart from the items in section 4: compare with `scripts/gantry/gantry_dynamic/model.py` and `model_augmentation/fit_systems/interconnect.py`.
- Where the closed-loop input enters the code path (`model_augmentation/fit_systems/closed_loop.py`).

## 6. Tried and failed (from the Section IV sessions; do not repeat)
- Generic method text without gantry-specific formulas: rejected as not a proper state-space contribution.
- Word budgets per subsection: rejected.
- One long answer with brief, claim lists, equations and figures at once: too much to review; hence the two stages.
- Writing LaTeX through bash heredocs: `\\` collapsed into `\`. Use the Edit/Write tools.
- latexmk test compile: `.latexmkrc` forces `build/`, overwrote the editor build. Do not compile.

## 7. Achieved
None for Section III. Section IV prose exists and is the consumer of III's definitions.

## 8. The open question
Nothing blocked. Open for the user: whether "Why augmentation rather than competing residual models" stays in III or moves to Introduction/Discussion (its own todo in the file).

## 9. Next action
Read the sources (README order, plus Sections II, IV, V), build the coverage inventory, then return the four stage-1 items of section 1 and wait.

## 10. Acceptance criterion
The user approves stage 1. Before returning it: the outline matches the format example; the coverage table has no empty owner cell and no item owned twice; every Section IV need is a bullet in III; every literature-based bullet has a location read in the PDF; missing `refs.bib` keys are flagged.

## 11. Read these first
1. `Thesis-writeup/Writing/README.md`: the thesis writing instructions.
2. `Thesis-writeup/Writing/sections/03_augmentation.tex`: current bullets and WRITING GUIDE.
3. `literature/augmentation/hoekstra2025_lfr-augmentation-ejc.pdf` Secs. 2 to 3.
4. `Thesis-writeup/Writing/sections/04_obc.tex` (read only): what it expects Section III to define.
5. `Thesis-writeup/Documentation/TRAINING-DESIGN.md`: what Section V owns.

## 12. Do not
- Compile, or write to `Thesis-writeup/Writing/build/`.
- Edit `04_obc.tex`.
- Reintroduce φ, η, ρ, J, S for the objects in section 4, or the A_aa/B_u blocks.
- Start stage 2 or prose before stage 1 is approved.

## 13. Operational
No runs. Reading and outlining only.

## 14. Delegation
At most one Explore subagent, only if locating the closed-loop and encoder code paths takes more than a few targeted searches. Read the listed papers inline; the deep-research skill is for finding new literature, not for reading these.
