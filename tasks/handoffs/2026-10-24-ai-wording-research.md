# Handoff: research how to recognise AI-sounding sentences, then write the prompt for a session that audits the thesis for them
**From**: Section IV session (2026-10-22 to 10-24) | **Branch**: Augmentation | **Effort suggested**: high (literature research and turning it into checkable criteria)

**Dirk's current message always overrides this plan. When he asks a question, answer that question first, in plain words, in a few sentences.**

## 1. Task
Dirk (the author) wants to find every sentence in his thesis that reads as AI-written, so he can rewrite those himself. His words (2026-10-24): "start a new session that will create a prompt and detect all the sentences that read as ai. But it first needs to determine online how to find these ai sentences. And that session will need to make a prompt for a new session that will read over my thesis writeup (for the sentences actually used) and make an audit of ai sentences and wording, so i can focus on rewriting those."

Your job has two parts. You do not audit the thesis yourself.
1. **Research.** Find out from the literature and other reliable sources which features of a sentence or wording make English academic text read as written by a large language model, as judged by human readers and by studies of word and phrase frequency. Include vocabulary (words and phrases that are overused), sentence construction, paragraph structure, and the habits of technical or scientific writing in particular. Use the `deep-research` skill (`.claude/skills/deep-research/SKILL.md`, per the literature rule in `CLAUDE.md`). Before launching any search agent, post your sub-questions to Dirk and wait for his OK.
2. **Write the audit prompt.** Turn the findings into a list of checkable criteria. Each criterion gets a name, what to look for, one example, and its source. Merge them with Dirk's own rejected patterns (4.2). Then write a handoff for the audit session to `tasks/handoffs/2026-10-24-ai-wording-audit.md`, following `docs/session-handoff-protocol.md`, with the criteria list inside it. Its task is set out in 4.3.

Show Dirk the criteria list before you write the audit prompt.

## 2. Out of scope
- Rewriting or editing any thesis text. The audit session does not edit either: Dirk rewrites.
- Making text "undetectable": no evasion tricks (typos, homoglyphs, humanizer tools). The goal is text that is clear and in Dirk's own voice.
- Running AI-detector tools on the thesis. Their scores are not evidence of authorship (4.1).
- Thesis LaTeX builds, git writes, `kamtin-data/`.

## 3. Where things stand
The thesis is in `Thesis-writeup/Writing/sections/` (`00_abstract.tex` to `08_conclusion.tex`, plus `99_appendix.tex`). Sections I to V were walked with Dirk on 2026-10-20 to 10-23, and `05_setup.tex` is about to be restructured in another session (`tasks/handoffs/2026-10-24-section6-setup.md`).

## 4. Established and verified

### 4.1 Earlier findings (memory `project_ai_writing_pipeline.md`, 2026-09-29)
- The TU/e "Most Urgent Rules for AI Use in Education" (25 Nov 2024), rule 4: "Scores from AI detection tools do not count as sufficient evidence of fraud".
- An earlier deep-research run (2026-09-29) found that no prompt makes a rewrite reliably undetectable, and that expert human readers recognised AI text well (Russell et al., ACL 2025, as recorded in that memory). Its full output is not in the repo: re-find any source you rely on.

### 4.2 Patterns Dirk has already rejected as "AI" (from memory `feedback_keep_dirk_wording.md` and the 2026-10-22/23 sessions)
- "so" / "so that" result clauses, "This makes ...", "may";
- participle tails ("..., making ..."), semicolons, comma chains with several clauses, appositions, opening adverbials ("In normalised coordinates, ...");
- invented labels instead of saying what happens ("early check", "plateau tolerance");
- vague abstractions ("to include all components", "leaves an offset", "acts", "operates");
- section overviews and summaries nobody asked for, and sentences that restate the equation beside them.
His own style, as a positive reference: subject first, stating a fact; the reason at the end after "because"; plain verbs (is, are, uses, gives). Examples he wrote or kept: "The system used to obtain the data is simulated." "We use the dynamic parallel structure of Hoekstra et al. [Table 1]." "A subset of records is left out of S1 and S2 to evaluate the performance of each stage."

### 4.3 What the audit session must do (put this in its prompt)
- Read every sentence that is printed in the thesis: the final reading with the change marks applied (`Thesis-writeup/Writing/util/format.tex`). `\chg{old}{new}` prints new; `\del`/`\delmv` are not printed; `\mv`/`\add` are printed; `\kw`/`\kwtwo` are plain text. Skip `%` comments, `\todo`, `\revnote`, `\ifdraft` blocks, and equations themselves (but do check the prose around them). Captions count.
- Write the audit to one file, `tasks/ai-wording-audit.md`. For each section: every flagged sentence, quoted, with its `file:line`, the criterion it matches, and how strong the match is (clear / borderline). End with a short count per section and the most frequent criteria.
- No rewrites and no proposed wordings. Flag only, so that Dirk can rewrite.
- Flag Dirk's own sentences too when they match: the audit is about how a sentence reads, not who wrote it.
- Read-only on `Thesis-writeup/`. No builds.

## 5. Assumed but not verified
- That the literature gives sentence-level features usable by a reader, not only detector statistics. If it mostly gives statistics, say so to Dirk and keep only the features a person can check in a sentence.

## 6. Tried and failed
- Lists of banned words alone did not stop AI phrasing in the thesis sessions: Dirk said a word list cannot catch it. The criteria must also cover construction (clause structure, labels, abstraction, redundancy), not only vocabulary.

## 7. Achieved
None yet for this task.

## 8. The open question
Nothing blocked.

## 9. Next action
Read `.claude/skills/deep-research/SKILL.md`, then post your sub-questions for the research to Dirk.

## 10. Acceptance criterion
Dirk has approved the criteria list, every criterion has a source (a paper or study, or "Dirk, 2026-10-xx"), and `tasks/handoffs/2026-10-24-ai-wording-audit.md` exists and can be started by a fresh session without questions.

## 11. Read these first
1. `.claude/skills/deep-research/SKILL.md`.
2. Memories `project_ai_writing_pipeline.md` and `feedback_keep_dirk_wording.md`.
3. `docs/session-handoff-protocol.md` (for the audit prompt).

## 12. Do not
- Edit any `.tex` file.
- Launch research agents before Dirk approves the sub-questions.

## 13. Operational
None.

## 14. Delegation
Research: up to about 5 agents, one per sub-question, after Dirk approves the sub-questions. The audit prompt you write may let the audit session use at most 2 agents (for example one per half of the sections).
