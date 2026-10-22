# Handoff: make thesis III-B ("Baseline state-space model") state why each part is there, in Dirk's wording, without forward references
**From**: Section III session of 2026-10-21/22 | **Branch**: Augmentation | **Effort suggested**: high (judging narrative and justification against supervisor feedback; replies stay short)

## 1. Task
Revise `\subsection{Baseline state-space model}\label{sec:lfr}` in `Thesis-writeup/Writing/sections/02_system_baseline.tex` with Dirk so that every sentence has a reason the reader can see at that point. The current text says what the baseline is (position-dependent inertia, explicit state equation, rational in Y, self-scheduled LPV, exact, LFR in the appendix) but not why the reader needs each statement. First read the supervisor feedback in its PDF context (13) and the sections before III-B (Section I research questions, Section II, III-A), then post to Dirk, in one short block, your diagnosis of what is vague in III-B and the reason for each part that can be drawn from text the reader has already seen. Then work paragraph by paragraph (final text with marks applied, comments that belong to it, your findings, the smallest fix in Dirk's wording as a marked LaTeX block, wait). Use the `thesis-section` skill for structure (it models sections on Hoekstra's papers). Apply only on Dirk's explicit "apply" / "yes, apply".

## 2. Out of scope
- III-C (`sec:params`, identifiability, Maarten v2 p5 notes (a) to (d)): next step, its own work. III-B may only provide what III-C will cite (the explicit state equation, the split if kept).
- Moving the LFR back into III-B: decided (see 4). Appendix A.2 content is done unless III-B needs a reference fix.
- III-A (finished on 2026-10-21/22 except the ASMPT revnote, which Dirk leaves open).
- Section II text (another session), Sections IV onwards: edit only a cross-reference that III-B changes break, and say so.
- LaTeX builds (Dirk's editor rebuilds), git writes, `kamtin-fp-model/`.

## 3. Where things stand
Branch `Augmentation`, last commit `b4a985b`; uncommitted edits in `02_system_baseline.tex`, `03_augmentation.tex`, `04_obc.tex`, `06_results.tex`, `99_appendix.tex`, `main.tex`. Dirk edits the same files in his editor during the session: re-read the passage right before every edit. Backups of the 2026-10-22 pre-edit states are in the old session's scratchpad (not needed). No runs in flight.

## 4. Established and verified (decisions with Dirk, 2026-10-22)
- III-B was "Self-scheduled LPV-LFR representation" (about one page). The LFR construction (Fig. `fig:lfr`, `eq:lfr_mass_factorization`, `eq:lfr_chain`, `eq:lfr_constant`, `eq:delta`, `eq:lfr_G`, loop closing, well-posedness via Schur complement, closed form) now lives in Appendix A.2 (`app:lfr-details`, `99_appendix.tex`), with an opening sentence. Reasons: the augmentation, RK4 and OBC use the explicit transition, never the LFR loop (`06_results.tex`, LFR-role paragraph); the introduction no longer claims the LFR as a result (`01_introduction.tex`, commented contribution near `:462`, and `:339` Jan26); Roland asked for the LFR and accepted it as "derivation and structural analysis only" (`docs/decisions.md` D-005, D-017, D-020).
- Current III-B order (read the file; marks show what moved): Dirk's physics sentence (Y changes M(Y)); split `eq:Msplit` M(Y)=M_0+YM_1+Y^2M_2 with "where M_0, M_1, M_2 are given in Appendix A.2"; M(Y) > 0 (Appendix A.3) so q'' can be computed (Dirk reworded this himself); state x~=[q;q'] and `eq:base_ss` x~' = F(x~,u) = [q'; M(Y)^{-1}(u - Cq' - Kq)] (moved from IV-A, Dirk's sentences); "Because of M(Y)^{-1}, (base_ss) is rationally dependent on Y"; merged LPV sentence "Treating Y as an externally specified scheduling signal, (base_ss) is linear in x~ and u, with matrices that depend on Y: it is an LPV model ... [Toth]"; Dirk's "self-scheduled" and "coincides with (3) for every motion of Y and needs no assumption on Y'" sentences; "Equation (base_ss) defines the continuous-time baseline from which the discrete-time transition used in Section IV is obtained"; LFR pointer ("exact self-scheduled LPV realisation as a linear fractional representation (LFR). Appendix A.2 derives ...").
- Cross-references already updated: IV-A says "The baseline state derivative F is given by (base_ss), where M, C, K depend on theta_base through (12)" (`03_augmentation.tex`); `04_obc.tex:168` cites `eq:base_ss`; `06_results.tex` LFR paragraph cites Appendix A.2. `eq:aug_ct` and `eq:qdd` no longer exist. Each label is defined once.
- Terms checked: F is the state derivative (vector field), not a transition; the discrete-time transition comes from one RK4 step in IV-A. Maarten's p5 note (b) "i.e. state derivative." means x~' = [q'; q''], so (base_ss) shows q'' is computed from the model.
- `eq:Msplit` is used by III-C ("Since M(Y) is quadratic in Y, this also implies M_0=M_0' ...", `02_system_baseline.tex` around `:676`) and by A.2 ("In (Msplit), M_0 is the part of (4a) without Y").
- Section IV gives Y to the learned correction (`03_augmentation.tex:442-444`), but citing that from III-B is a forward reference (rejected, see 6).
- Research question on generalisation to "held-out operating points and unseen motion profiles": `01_introduction.tex:424` (and `:496` in a draft block).

## 5. Assumed but not verified
- That "held-out operating points" in Section I means held-out payload positions Y, which would let III-B justify carrying Y explicitly by a backward reference. Settled by reading Section I and Section V's split definition, and by Dirk.
- Whether III-B needs the split (6) at all, or whether it belongs where III-C uses it. Settled by Dirk after the diagnosis.
- Whether "self-scheduled" needs a source (Dirk's todo "source, or do i state it earlier..?"); Toth 2010 p. 165 calls a scheduling variable that is a function of the state quasi-LPV. Settled by checking `Thesis-writeup/References/` (`index.csv`).

## 6. Tried and failed (this session)
- Proposals that cited Section IV from III-B ("Section IV also gives Y to the learned correction", "obtained by one RK4 step") -> Dirk rejected: forward references, which Maarten's mail explicitly criticises -> justify only from what the reader has already read.
- Rewording Dirk's sentences beyond a few words, and offering new phrasings in my own style -> Dirk: "doesn't feel like me" -> fixes of at most about five words; new text only where needed, in `\kwtwo{}`, in his sentence structure (short sentences, no semicolons, no "we" openers he did not write).
- Flip-flopping recommendations on the LFR's place (main text, appendix, back) and the figure -> lost Dirk's trust -> the LFR placement is decided (4); do not reopen.
- Using a term loosely ("baseline transition" for the continuous-time F) -> corrected -> check each technical word against its definition.
- Applying an edit after a direction like "so drop X" -> Dirk was angry -> only "apply"/"yes, apply" authorises an edit.

## 7. Achieved
III-A finished (except ASMPT revnote, left open). Matrices as `subequations` (4a) to (4c). Appendix parameter table: caption, paragraph, `[!ht]`; appendix subsections numbered A.1, A.2 (`main.tex`). A.2 rebuilt as the complete LFR derivation. III-B restructured as in 4.

## 8. The open question
What reason does III-B give the reader for (a) the explicit state equation, (b) the LPV characterisation (rational, self-scheduled, exact, no assumption on Y'), and (c) the split (6), using only Sections I, II and III-A? Candidates: (a) it is the model the rest of the thesis builds on (state that without naming Section IV, or name it as the purpose in III-B's first sentence); (b) Y changes during operation and the research question asks for held-out operating points (Section I), so the model must carry Y explicitly and exactly; (c) move to III-C where it is used, or state its use here without a forward reference. Evidence: Section I text, Maarten's mail, Dirk.

## 9. Next action
Read Maarten's v2 PDF p4 to p5 and his mail, the v1 pages on old II-B (Jasper on "LPV ... how quickly Y changes", Quinten "Explain LPV ... earlier"), Section I's research questions and Section II, then post to Dirk your diagnosis of III-B (one line per vague point) with the backward-sourced reason for each part, before any LaTeX.

## 10. Acceptance criterion
Every sentence of III-B either states a fact the next sentence needs or gives a reason drawn from earlier text; no forward reference to Sections IV to VII in live III-B text; at least most sentences are Dirk's, with changes visible through the marks; Dirk says it reads clearly in his build.

## 11. Read these first
1. `02_system_baseline.tex` from `\subsection{Baseline state-space model}` to `\label{sec:params}`, with its source comments (RESTRUCTURE, REORDER).
2. Maarten v2: `Thesis-writeup/Feedback/v2/Feedback-Maarten-General-v2.txt` (mail) and the PDF p4 to p5 in context.
3. `01_introduction.tex` research questions (around `:420-430`) and Section II (`02_system_baseline.tex`, `\section{Identification problem}`).
4. `.claude/skills/thesis-section/SKILL.md`.
5. `tasks/handoffs/2026-10-21-section3-physics-baseline.md` sections 4.4 (mark macros, reply shape) and 13 (feedback dump method).

## 12. Do not
- No forward references in live text (Sections IV to VII), not even "Section IV augments it".
- Do not reopen the LFR placement or move the figure back.
- Do not reword Dirk's sentences beyond about five words; no semicolons; do not invent openers.
- No edits without "apply"; no LaTeX builds; no git writes.

## 13. Operational
Feedback dump (PyMuPDF, annotations plus page PNGs): the method in `tasks/handoffs/2026-10-21-section3-physics-baseline.md` section 13; run with `conda run -n GraduationProject python -I <script>` from the session scratchpad. Reply shape that worked: diagnosis in one line, smallest fix as a marked LaTeX block, "Apply?". Text for Dirk to paste: LaTeX in a code block, one sentence per line, about 75 characters.

## 14. Delegation
None: known files, a dialogue with Dirk.
