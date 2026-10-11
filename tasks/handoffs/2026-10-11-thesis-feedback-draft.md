# Handoff: bring the thesis paper to a supervisor-feedback draft (intro, todos, appendix, outlines of V to VIII with results)
**From**: session of 2026-10-11 | **Branch**: Augmentation | **Effort suggested**: high

## 1. Task
Get the IEEE thesis paper in `Thesis-writeup/Writing/` to a state Dirk can send to his supervisors for
feedback. In Dirk's words: "this section will be focusing on finishing so i can get actual feedback from
my supervisors." Four parts, in this order:
1. **Introduction** (`sections/01_introduction.tex`): look at what can be copied 1:1 from the introduction
   of the research plan (`presentations/Research_plan___Graduation_project_AI_ES___Feedback_Processed (3).pdf`,
   pages 2 to 3), and reuse it. The one content change Dirk named: the application motivation moves from
   condition/health monitoring to **process and root cause analysis**, where interpretability is the value
   (his own todo at `01_introduction.tex:27`).
2. **Todos**: work through the `\todo{}` items with Dirk, one at a time (see section 8).
3. **Appendix** (`sections/99_appendix.tex`): bring it in line with what Sections II to IV now reference.
4. **Sections V Experiment design (`05_setup.tex`), VI Results (`06_results.tex`), VII Discussion
   (`07_discussion.tex`), VIII Conclusion (`08_conclusion.tex`)**: Dirk will NOT write these in prose now.
   Write each as **clean, clear bullet points of what the section will discuss**, and put the results
   figures and tables of `Thesis-writeup/Results/11-10-2026/thesis-results-2/` into Section VI. The current
   bullets in these files that are not clean may be left in the file (hidden, see section 8); they do not
   have to be deleted.

## 2. Out of scope
- Sections II, III, IV prose (`02_system_baseline.tex`, `03_augmentation.tex`, `04_obc.tex`): III was
  restructured with Dirk on 2026-10-09 to 10-11 and committed (`724f79a`); do not rewrite it. Only touch
  them if an intro or appendix change needs a label fixed, and then ask first.
- Writing V to VIII as prose: Dirk explicitly said he will not write these now. Bullets only.
- Code changes and reruns: the fixes for the final runs are listed in
  `Thesis-writeup/Documentation/FINAL-RUN-FIXES.md`; they belong to a later session.
- `main.tex`, `.latexmkrc`, `.vscode/settings.json`, `util/include.tex` (README editor-build invariant).
- New docs under `docs/`: none needed.

## 3. Where things stand
- Branch `Augmentation`, last commit `724f79a Update section 3`.
- Dirty: `Writing/sections/01_introduction.tex`, `Writing/sections/06_results.tex`, `Writing/build/*`;
  untracked: `Thesis-writeup/Documentation/THESIS-RESULTS.md`. These are Dirk's; build files are written
  by his editor.
- No runs in flight from this session.

## 4. Established and verified
- Paper ownership and the rule "experiment design never explains training": `Thesis-writeup/Writing/README.md`,
  "Paper structure and ownership". Section V owns the black-box comparator, compared-model table, metrics and
  the hyperparameter table (method values; implementation settings go to the appendix, agreed 2026-10-10).
- `05_setup.tex` already holds a bullet list "Training protocol, comparators, and metrics"; one bullet was
  added 2026-10-11: the n_x̄ = 0 static-parallel case moved there from III-A (run R6).
- `05_setup.tex` and `06_results.tex` still describe superseded items (Karnopp friction, disabled noise,
  Telica split, affine OBC arm): README "Open conflicts". Where they disagree with
  `Documentation/THESIS-RESULTS.md`, THESIS-RESULTS wins (README rule).
- `Documentation/THESIS-RESULTS.md` sections 3 to 7: the presentation order of results (steps 1 to 6 by
  research-plan aspect), the server runs (section 4) and open points. It is a design of what is run and what
  each step answers; it does not list the numbers of the 2026-10-11 runs (no run ID 88755 to 88757 found in it).
- `Results/11-10-2026/thesis-results-2/` contains only SLURM logs: `thesis-obc_88755_{201,202}.out`,
  `thesis-u_88756_{203..207}.out`, `bb_search_88757_{0..10}.out`. No figures, no CSV, no npz in that folder.
- Dataset of the reported runs: `Coulomb-tanh-and-MSD-lowpass-noise` (D-238, THESIS-RESULTS section 7).
- PS2 was renamed **PS** ("predictive state (PS) initialisation") in the thesis text; model names are
  calligraphic: $\mathcal M_{\mathrm U}$, $\mathcal M_{\mathrm{PS}}$, $\mathcal M_{\mathrm{OBC}}$, $\mathcal M_{\mathrm{BB}}$.
- The research plan's introduction motivates the model by "digital-twin development, offline evaluation,
  control design, and condition monitoring" (PDF p. 3). The thesis intro carries Dirk's todo to replace
  health monitoring by process and root cause analysis (`01_introduction.tex:27`) and an open todo on which
  benefit leads (`:26`).

## 5. Assumed but not verified
- **Where Dirk's description of the results lives.** Dirk said the results "should be clearly described" in
  `Thesis-writeup/Documentation`. THESIS-RESULTS.md is the design, not the outcomes. Settle by searching
  `Thesis-writeup/Documentation` and `scripts/gantry/thesis-results/` for an outcome summary of runs
  88755 to 88757; if none exists, ask Dirk once where it is before extracting numbers from the `.out` logs.
- **How the figures are made.** The figure pipeline is `Code/Figures/<stem>.py` reading `Results/<stem>.csv`
  and writing `Writing/figures/<stem>.pdf` (README "Figure and table pipeline"). Whether scripts exist for
  these runs is not checked. Evaluation code: `scripts/gantry/thesis-results/` (README source map, `TR/`).
- **Which runs map to which THESIS-RESULTS step** (R-numbers vs array indices 201 to 207): not checked; read
  the run table in THESIS-RESULTS section 4 against the `[thesis run]` header line of each `.out` log.
- **"Leave the bullets in but they don't have to be removed"**: my reading is that unclean bullets are hidden
  from the PDF but kept in the source. To confirm with Dirk; default if unconfirmed: wrap them in
  `\iffalse ... \fi` with a one-line comment, never delete.

## 6. Tried and failed
- Rewriting Dirk's sentences "for clarity" -> he rejected it repeatedly on 2026-10-10/11 because it turns his
  rewritten text back into recognisably AI-styled prose (he rewrites to avoid AI detection) -> the edit
  replaced his wording, not just the error -> see section 12, rule 1.
- Editing a `.tex` file while Dirk had unsaved changes -> VS Code refused his save ("content is newer") and
  his edits were nearly lost -> file written on disk under an open buffer -> section 12, rule 2.
- Claiming things "in Jan's words" from memory -> wrong section numbers twice (EJC vs 2026 paper, Sec. 2 vs 3)
  -> not read at the source -> read the passage in the PDF before citing (CITATION-LOG rule in README).

## 7. Achieved
- Section III restructured and committed (`724f79a`); its open items are `\todo{}`s in the file.
- `Thesis-writeup/Documentation/FINAL-RUN-FIXES.md`: code fixes for the final runs (W^b from the detuned
  start, polish acceptance without truth meters, PS target from τ+1) and checks (D-232 leak check passed on
  the lowpass data, 0.22 to 0.50 nm).

## 8. The open question
How to handle the todos: Dirk wants to fix them himself, with help. Candidate approaches: (a) list every
`\todo{}` in sections 01 to 08 and 99 grouped as "needs Dirk's decision", "needs a source check",
"mechanical", and go through the first group with him; (b) only the todos in 01 and 99. Ask Dirk at the start
which; default (a) as a list in chat, no edits.

## 9. Next action
Start with the introduction: read the research-plan introduction (PDF pp. 2 to 3) and
`01_introduction.tex`, and show Dirk in chat, per paragraph, which research-plan text can be copied 1:1,
which needs the root-cause-analysis change, and which thesis-intro text it replaces. No file edits until he
says which to take.

## 10. Acceptance criterion
Dirk can send the PDF to his supervisors: the introduction reads as his text; sections V to VIII each
have clean bullets covering what the section will discuss, consistent with THESIS-RESULTS.md; Section VI
shows the figures/tables of the 2026-10-11 runs with their numbers; the appendix has no dangling labels;
after his editor rebuild `build/main.log` has no line starting with `!` (README "Done when").

## 11. Read these first
1. `Thesis-writeup/Writing/README.md`: ownership, Math standard, phases, authorial-voice rules, figure pipeline.
2. `Thesis-writeup/Documentation/THESIS-RESULTS.md`: what each result step shows and which runs feed it.
3. `presentations/Research_plan___Graduation_project_AI_ES___Feedback_Processed (3).pdf`, pp. 2 to 5:
   introduction, research questions, aspects.
4. `Thesis-writeup/Writing/sections/01_introduction.tex` and `05_setup.tex`.
5. `Thesis-writeup/Documentation/FINAL-RUN-FIXES.md`: what the current results are still missing (for VII).

## 12. Do not
Dirk's working rules from the 2026-10-09 to 10-11 session (all confirmed by him):
1. **Never rewrite Dirk's sentences.** Point out a problem with a fix of at most five words, or show a full
   proposal in chat. Example of the wanted format for a proposal (line format, markers, nothing in the file yet):
   ```latex
   The positions are corrected as well because the correction acts on the   % NEW
   discrete-time step.                                                       % NEW
   ```
   When he asks you to put text in, put in exactly his wording or the shown proposal, keep his line breaks.
2. **No file edit unless he asks**, and before editing check he has saved (ask "saved?"); after editing tell
   him to run "File: Revert File" in VS Code.
3. Short, concise answers; bullets help him. Plain words; he dislikes AI-sounding phrasing ("evolve",
   "counteract", "act on centred signals").
4. No `\cite` claim without reading the passage in the PDF this session.
5. Do not compile `main.tex`; standalone test renders of a figure go to the scratchpad only.
6. No git write operations (CLAUDE.md standing rule).

## 13. Operational
- Python: `conda run -n GraduationProject python <script>` (multi-line snippets via a scratch file).
- PDF text: `fitz` (PyMuPDF) in that env; for pages with equations use the Read tool on the PDF.
- C: has filled up before; check free space before rendering figures.

## 14. Delegation
None for the intro, todos and appendix (targeted edits). At most one Explore subagent if locating the
results description or figure scripts takes more than a few searches.
