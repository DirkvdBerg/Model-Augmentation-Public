# Handoff: finish VI-D "Compared models" and VI-E "Metrics" in `05_setup.tex` with Dirk: flag problems, propose cuts, keep his wording
**From**: Section IV session, after reading the failed setup session `5af4cab4` (2026-10-24) | **Branch**: Augmentation | **Effort suggested**: high

**Dirk's current message always overrides this plan. When he asks a question, answer that question first, in plain words, in a few sentences.**

## 1. Task
Finish subsections VI-D `\subsection{Compared models}\label{sec:setup_models}` and VI-E `\subsection{Metrics}\label{sec:setup_metrics}` of `Thesis-writeup/Writing/sections/05_setup.tex` (Dirk calls this "section 5" after the file name) with Dirk, the author. VI-A to VI-C were restructured earlier on 2026-10-24. Walk D, then E, in reading order. In each turn:
- show one paragraph (or caption or table) as it reads in the final text;
- name its problems, each with the few words it is about and the reason;
- give the smallest fix as Now / After in rendered text (4.1).

Prefer a cut, or a move to the appendix, over new words. When a fix needs words, take them from Dirk's own sentences (4.3). When Dirk writes a sentence himself, check only its facts and grammar, and apply it as he wrote it. Apply only on his "yes", "apply" or "do it", and only what he approved. When he asks what is open, give the complete list for D and E at once.

## 2. Out of scope
- VI-A to VI-C, other sections and the appendix. If a fix needs a change there, say so in one line and leave it. Exception: the last paragraph of VI-C is still open (4.5). Raise it once at the start and do it only if Dirk says so.
- Which results and figures the Results section shows: another session (`tasks/handoffs/2026-10-24-results-what-to-show.md`). Metrics in E must not contradict its decisions. Ask Dirk if unsure.
- Thesis LaTeX builds, git writes, `kamtin-data/`, the `\ifdraft` blocks.

## 3. Where things stand
Branch `Augmentation`. The `05_setup.tex` subsections are A Simulated system (`:69`), B Closed loop and excitation (`:213`), C Records (`:320`), D Compared models (`:414`) and E Metrics (`:508`), as of 2026-10-24. Dirk edits in parallel and saves often. Re-read a paragraph right before editing it, and if an edit does not match, re-read instead of guessing.

## 4. Established and verified

### 4.1 How to show a proposal
Rendered text, never LaTeX, until Dirk says apply. Always show the whole sentence or block that changes, so he sees exactly what is different. Example of a reply he accepted:

> **Now:**
> > ..., where i_s < i_max is a fixed early check.
>
> **After:**
> > ..., where S1 already stops at update i_s < i_max if ν_{i_s} > ν_max.
>
> "Early check" was a name that explained nothing; the After says what happens.

On apply, use the marks of `Thesis-writeup/Writing/util/format.tex`:
- `\chg[own]{old}{\kwtwo{new}}`, `\del[own]{...}`, `\add[own]{\kwtwo{...}}`; tag `Dirk` for text he wrote in the chat.
- Use `\mv`/`\delmv` only to move text, and tag it by who wrote it. Do not assume that existing text is Dirk's.
- No marks inside math, captions or `\cite[...]`, and none nested in struck text (use `\mbox{\ref{...}}` there).
- Add a `% FEEDBACK (<who>, 2026-10-2x): <reason>` line above each change.
- Check that `{` and `}` still balance as before.

### 4.2 Decisions Dirk made on 2026-10-24 (do not reopen them)
- Confidentiality: errors and noise only in normalised form (NRMS); no metres, no plant gain, no noise model. Frequencies may be given in Hz, and f_b = 100 Hz may be stated. Limits are relative values (memory `feedback_units_scientific_metres.md`).
- No section or subsection roadmap (memory `feedback_no_section_roadmap.md`).
- "Plain training" is the standard start of Hoekstra et al., without PS.
- "Data is", not "data are".
- In D he dislikes: the word "compared" in the subsection title; the current models table (the "start" column makes no sense, and the caption does not say what the table is); and the caption "Compared models. Start: ...; final model: ...". The `\todo` on model notation (`:452`) is open.
- E6 exceeds multiple limits.
- The final results are in `Thesis-writeup/Results/24-10-2026/CSV` (memory `project_final_run_priority.md`). Whether the black box is scored in RMS or NRMS was asked and not settled. Check the E text against the CSVs.

### 4.3 Dirk's wording
Sentences he wrote in this section (re-read them in VI-A to VI-C before proposing):
- "The data used to validate our methods is simulated, with limits and noise level taken from a gantry of ASMPT."
- "The records are downsampled to T_s = 0.25 ms."
- "We separate the test records into interpolation and extrapolation. Interpolation stays within the training range, while each extrapolation record exceeds at least one of the limits on Y, acceleration or velocity."

His pattern:
- subject first, stating a fact; the reason at the end after "because"; plain verbs;
- one clause per sentence, few commas.

Things he rejected in the setup session: "so", semicolons, comma chains, "paired records", "twins", "at most about", "above the noise" (vague), "and their checks", and invented labels in general. A word list is not enough: the failed session kept adding a sentence for every reason, inventing labels, and rewriting a whole paragraph when he pointed at one word. Change only what he points at.

### 4.4 Revnotes in E (Dirk's own notes, to settle with him)
- `:529`: the research plan names BFR as the primary measure.
- `:568`: the LTI model noise floor does not change per record; take the noise floor of the noise model.
- `:576`: the noise floor is 0.005 to 0.11 % on the test records.
- `:609`: also show the unconstrained model in this table.
Treat them as items in the walk. Do not delete them without his yes.

### 4.5 Open end of VI-C (from session `5af4cab4`)
The test-record paragraph was not settled. The last version shown to Dirk:
> Each extrapolation record uses the same motion profile as an interpolation record. The difference in error between them shows the effect of leaving the training range. We also simulate seven of these records with a multisine, to test the added mass on motion not used in training.
His earlier objections: "We also simulate these records with a multisine in the training band ..." "doesn't make sense", and he asked for one formulation "without monkey patching". He also said the test sets should have a different crest factor. Ask him whether to settle it first.

## 5. Assumed but not verified
- That the current D and E text matches the final CSVs (models, splits, metric names). Settled by reading the CSV headers.

## 6. Tried and failed (setup session `5af4cab4`, 2026-10-24)
- It re-added a roadmap after being told not to; my handoff had left that question open -> now decided (4.2).
- It marked AI-written sentences as Dirk's moved text (`\mv[own]`), for example the `K_1` sentence with a semicolon -> marks by author (4.1).
- It kept "frequencies relative to f_b" after Dirk said the bandwidth may be stated -> 4.2.
- It proposed new wording again and again ("paired records", "learned added mass") and rewrote whole paragraphs -> cut first; change only the words he names; reuse his sentences.
- Its proposals did not show clearly what changed -> whole block, Now / After.

## 7. Achieved
VI-A to VI-C restructured and walked (VI-C's last paragraph open, 4.5).

## 8. The open question
Nothing blocked.

## 9. Next action
Read `05_setup.tex` from VI-A to the end, then ask Dirk in one line whether to settle 4.5 first. Then post the first item of D.

## 10. Acceptance criterion
Every paragraph, caption, table, todo and revnote in D and E is done, or left by Dirk's choice, in a closing list. Every change is visible through marks. No metres or unstated confidential quantity in the printed text.

## 11. Read these first
1. `05_setup.tex` in full.
2. Memories `feedback_keep_dirk_wording.md`, `feedback_proposal_now_after_format.md`, `feedback_no_section_roadmap.md`, `feedback_units_scientific_metres.md`.

## 12. Do not
- Edit before an explicit yes. A pasted phrase or a question gets an answer, not an edit.
- Show LaTeX before apply.
- Rewrite more than the words Dirk points at.

## 13. Operational
None.

## 14. Delegation
None.
