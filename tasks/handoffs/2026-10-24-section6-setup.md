# Handoff: restructure the thesis section "Experiment design" (`05_setup.tex`) on the model of Hoekstra et al. 2026, with Dirk, in his own wording
**From**: Section IV session (2026-10-22 to 10-24) | **Branch**: Augmentation | **Effort suggested**: high (structural judgement against a reference paper, then careful wording; replies stay short)

**Dirk's current message always overrides this plan. When he asks a question, answer that question first, in plain words, in a few sentences.**

## 1. Task
Restructure `\section{Experiment design}\label{sec:setup}` in `Thesis-writeup/Writing/sections/05_setup.tex` (Section VI in the build; Dirk calls it "section 5" after the file name) with Dirk, the author. His words (2026-10-24): "I'm not too happy with the current setup section, I think it looks messy. There is not a lot of text but it doesn't look clean. I think we should take inspiration from Hoekstra his 2026 paper. I would like the next session to state what is wrong with the current version, and propose a better section for the setup." Use the `thesis-section` skill. Work in three steps, each confirmed by Dirk before the next:
1. **What is wrong.** Read `05_setup.tex` in full in its final reading (marks applied) and the supervisor feedback on it (4.2). Post a concrete list of what makes the current version messy, each point tied to a passage (quote a few words), for example order, things that belong elsewhere (appendix, Results), repeats of Sections II to V, todo/revnote clutter in the printed text, missing "why" per subsection, uneven paragraph lengths, floating tables or figures. Include the feedback comments that apply.
2. **The model.** Read the experiment or simulation-study section of Hoekstra et al. 2026 (`hoekstra2026lfr`, file `Thesis-writeup/References/Hoekstra-2026-LFR-model-augmentation-arXiv-v1.pdf`; also check `hoekstra2026encoder`, `Hoekstra-2026-Encoder-initialisation-arXiv-v1.pdf`, and say which one fits better). Give its structure in a few lines, with PDF pages: which subsections, in which order, what each one tells the reader, and how much text it takes.
3. **The proposal.** Propose the new section as an outline: subsection headers with short one-line bullets (what each says and why it exists), open points marked (?). For every piece of current text, say where it goes: kept, moved (to which subsection, or to the appendix), or cut (with the reason). After Dirk approves the outline, carry it out one subsection per turn. Move his sentences with `\mv`/`\delmv` rather than rewriting them, and write new text only where the outline needs it, in his pattern (4.3), shown as rendered Now / After (4.1).

Apply only on Dirk's "yes", "apply" or "do it", and only what he approved. When he asks what is open, give the complete list for the current scope at once.

## 2. Out of scope
- All other sections and the appendix. If a fix here needs a change elsewhere, say so in one line and leave it.
- Results numbers and the `\revnote` items about the noise floor and tables (`:344`, `:365`, `:373`, `:405`): data questions for Dirk, not wording.
- The `\todo` items on confidentiality (`:183`, `:184`, `:233`): Dirk decides with ASMPT. Raise them only when he asks.
- The `\ifdraft` bullet blocks: never mention or edit them.
- Thesis LaTeX builds, git writes, `kamtin-data/`.

## 3. Where things stand
Branch `Augmentation`. The Section IV and V walks of 2026-10-22/23 edited `03_augmentation.tex`, `04_obc.tex` and `99_appendix.tex`. In `05_setup.tex` the last edits are from 2026-10-20/21 (five FEEDBACK/RESTRUCTURE comments). Dirk edits in parallel and saves often. Re-read a paragraph right before editing it, and if an edit does not match, re-read instead of guessing.

## 4. Established and verified

### 4.1 How to propose (this worked in the Section IV session; its absence failed in Section V)
Show the rendered text, never LaTeX, until Dirk says apply. Show the whole sentence or block that changes. Example of a reply he accepted:

> **Now:**
> > ..., where i_s < i_max is a fixed early check.
>
> **After:**
> > ..., where S1 already stops at update i_s < i_max if ν_{i_s} > ν_max.
>
> "Early check" was a name that explained nothing; the After says what happens.
>
> Should I apply this?

On apply, use the marks of `Thesis-writeup/Writing/util/format.tex`: `\chg[own]{old}{\kwtwo{new}}`, `\del[own]{...}`, `\add[own]{\kwtwo{...}}`, `\mv`/`\delmv` for moves (tag `Dirk` for his rewrites, or the comment ID). Add a `% FEEDBACK (<who>, 2026-10-2x): <reason>` line above each change. No marks inside math, captions or `\cite[...]`. No mark nested inside struck text, and use `\mbox{\ref{...}}` there. Check that `{` and `}` in the file still balance as before.

### 4.2 Feedback sources
- Maarten v2 (main supervisor, final say): `Thesis-writeup/Feedback/v2/Feedback-Maarten-Main-supervisor-v2-...pdf`, the pages of his build that hold "Experiment design" (find them, do not assume), plus `Feedback-Maarten-General-v2.txt` (order and narrative, no forward references, "het kan wel wat korter", appendix not read).
- v1: Jan (`Feedback-Jan-1/2-...pdf`, `Feedback-Jan-General-1-and-2.txt`: every subsection says why it exists), Jasper (`Feedback-Jasper-1-...v2.pdf`, `Feedback-Jasper-General.txt`: undefined variables), Quinten (`Feedback-Quinten-1-...pdf`).
- Dump method: `tasks/handoffs/2026-10-23-section5-obc.md` section 13 (PyMuPDF annotations plus page PNGs, UTF-8 file, never print PDF text to the console). Map a note to a paragraph by its wording, never by line number.

### 4.3 Dirk's wording (read his sentences, then write like them)
His own sentences in this section, for example `:46-53`: "The system used to obtain the data is simulated. When referring to the system in this section, we refer to this simulated model." Sentences he kept in Section IV: "We use the dynamic parallel structure of Hoekstra et al. [Table 1]." "Fig. 3 shows how the network is connected in parallel with the baseline model." "A subset of records is left out of S1 and S2 to evaluate the performance of each stage." The pattern:
- subject first, stating a fact; the reason at the end after "because"; purpose with "to";
- plain verbs (is, are, uses, gives, is computed), "we" structure, one clause per sentence, few commas;
- say what a thing does instead of giving it a new name;
- for new wording, reuse his own earlier wording first: struck text in the file, `\delmv` originals, `git show <commit>:<file>`;
- when unsure, propose a cut, not a sentence; a pointer such as "(Section II-C)" beats a definition sentence.
Memory: `feedback_keep_dirk_wording.md`, `feedback_proposal_now_after_format.md`, `feedback_paragraph_walk_method.md`.

### 4.4 Notation and terms already fixed elsewhere
𝒟_N is the training set and consists of multiple records ("the records of 𝒟_N", never "training data"); 𝒟_val is the validation set; the test set appears first in this section. The controller is K_fb (matrices A_fb to D_fb); K_1 is its continuous design. The simulation rate is f_sim = 20 kHz (not f_s), and the models run at T_s (1/T_s = 4 kHz). β are the training coordinates (Section III-C). Before introducing any symbol, grep `Thesis-writeup/Writing/sections/` for it: d, ξ, ζ_a and every physical parameter are taken.

## 5. Assumed but not verified
- Which supervisor notes sit on this section, and whether the five FEEDBACK comments in the file handle any of them. Settled by the dump in 4.2.
- Whether the roadmap sentence (`:57-60`) stays. Dirk added it on 2026-10-20 "as in Sections III to V", but the Section V session struck a section overview. Decide it in the outline (step 3).
- That Hoekstra 2026 LFR is the paper Dirk means ("his 2026 paper"). Most of the thesis follows it, but the encoder paper is also from 2026. Settled in step 2.

## 6. Tried and failed (Section V session, 2026-10-23, transcript `e1d5304e`)
- It proposed in LaTeX blocks -> Dirk could not read the result ("why are you not showing what the new text will look like") -> show rendered Now / After (4.1).
- It changed more words than Dirk named (he asked about "since training may start", or about one "it") -> anger -> change only the words he names.
- It added sentences and a section overview -> "you're making it longer and only adding ai text" -> cut or point instead.
- It used words he had rejected ("so", "may", "leaves an offset", "take the arguments of", a semicolon sentence) -> "terrible" -> write in the pattern of 4.3.
- It proposed symbols already in use (d, ξ) and the term "training data" -> 4.4.
- It stated what a paper says before reading it -> quote the paper (with the PDF page) before any claim about it.

## 7. Achieved
Sections I to V have been walked. Nothing in this section has been walked yet.

## 8. The open question
Nothing blocked.

## 9. Next action
Read `05_setup.tex` in full and dump the feedback (4.2), then post step 1 of section 1: the list of what is wrong.

## 10. Acceptance criterion
Dirk has approved an outline for the section, and it is carried out with marks. Every current passage is kept, moved or cut by his decision. Every supervisor comment on this section is done, or deferred with Dirk's reason, in a closing list. No symbol here conflicts with Sections II to V.

## 11. Read these first
1. `05_setup.tex` in full.
2. The feedback (4.2).
3. The experiment section of Hoekstra et al. 2026 (step 2).
4. Section 4.3 and the memories it names, plus `feedback_outline_format.md` for the outline.

## 12. Do not
- Edit before an explicit yes. A pasted phrase or a question gets an answer, not an edit.
- Show LaTeX before apply.
- Raise extra points after the item asked about.

## 13. Operational
None beyond the dump in 4.2.

## 14. Delegation
None: a dialogue with Dirk on known files.
