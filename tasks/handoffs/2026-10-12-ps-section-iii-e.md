# Handoff: write thesis Section III-E (PS initialisation of the added states), correct version, current runs in a todo
**From**: session of 2026-10-11/12 | **Branch**: Augmentation | **Effort suggested**: high (math and implementation must agree; prose in Dirk's style)

## 1. Task
Write subsection III-E "Added-state initialisation" (`\label{sec:aug_ps2}`) of
`Thesis-writeup/Writing/sections/03_augmentation.tex` (starts at l.847, runs to the end of the file) together with
Dirk, as finished prose with its displayed equations. Start by invoking the `thesis-section` skill. III-E describes
PS initialisation as it SHOULD work: the S1 target starts at tau+1, disjoint from the encoder window
(`Thesis-writeup/Documentation/FINAL-RUN-FIXES.md` item 5). Wherever the text describes a better version than the
code that produced the reported results, a `\todo{}` directly next to it states exactly how the current code and the
2026-10-11 runs do it and what that means for the results. Every step and every equation must match the
implementation (or the stated improvement); read the code, do not infer it from documents.

In Dirk's words: "i want to describe PS I want to state the correct version, because in the current run i have a
sample issue i think, and i would just like to state that in a todo how it is currently if i did it wrong" and
"make sure it is correct with the actual implementation, and the math is correct. but we can improve and state in
todo how we currently have it".

## 2. Out of scope
- The section intro and III-A to III-D, and every other section file. Only touch them if a label in III-E needs it, and ask first.
- `refs.bib` (a bibliography check just finished; add an entry only after the passage is read in the PDF and Dirk agrees), `util/format.tex`, `main.tex`, `.latexmkrc`, `.vscode/`.
- Code changes and reruns (fix 5 itself belongs to a later session).
- New docs. `CITATION-LOG.md` gets a row per passage you read.

## 3. Where things stand
- III-E currently holds an `outline` (shown in the review build) between its equations, with the earlier prose in `wip` blocks
  (mechanism: `util/format.tex`, README "Draft switch"). Labels in III-E: `eq:ps2_s1`, `eq:ps2_nu`, `eq:ps2_s2`,
  `fig:ps2_windows` (figure source `Writing/figures/tex/ps2_windows.tex`); all equations are outside `wip`.
- Existing todos in III-E already flag the issue: l.~899 (figure caption: "The figure shows the target from tau+1 ... the
  current runs start it at tau") and l.~936 (Hefny's disjointness argument, sample tau on both sides of `eq:ps2_s1`).
- The 2026-10-11 runs (R1 to R7, job ids 88762, 88755, 88763, 88764, 88767, 88768, 88756) used PS with the target
  from tau. Results that show PS works: Table I (`Writing/tables/res_structure.tex`): PS init n_a=2 Y error
  1.39e-6 m against 1.06e-5 m plain training and 1.24e-5 m static n_a=0; learned added-state pole 212 Hz,
  zeta about 0.05, against the system absorber pole 212.1 Hz, zeta 0.03 (`Thesis-writeup/Results/res_obc_overlap.csv`).
- The preview toggle `\def\ReviewBuild{}` (l.9 of `util/format.tex`) is Dirk's; leave it as he has it.

## 4. Established and verified
- Bibliography metadata of `hefny2015supervised`, `downey2017psrnn`, `beintema2023subnet` verified by three independent
  checks on 2026-10-11 (Hefny Sec. 2 and Downey Sec. 4.2 confirmed to exist and match). Hoekstra 2026 encoder passages
  are logged in `Thesis-writeup/Documentation/CITATION-LOG.md`.
- The review build strips citation locators (format.tex), so supervisors see only "[n]"; the locators must still be right.

## 5. Assumed but not verified
- That fix 5 (target from tau+1) is the only deviation between the intended PS and the code. Check the whole PS path; any
  further deviation is handled the same way: describe the correct version, current behaviour in a todo, after Dirk agrees.
- Whether Venkatraman et al. 2017 (Predictive-State Decoders) fits the S1 decoder idea well enough to cite. Not cited yet.

## 6. Tried and failed
- Pasting proposals that end in `% CHANGED` or `% NEW`: anything after `%` on the same line becomes a comment; twice this
  swallowed an `\end{outline}` / `\item` and broke the build. Give pasteable lines without markers.
- Bash heredocs or printf for .tex edits turned `\b` and `\e` into control characters. Use the Edit or Write tool.

## 7. Achieved
- None for III-E prose. The outline and equation check were done by the Section III session on 2026-10-11.

## 8. The open question
How deep III-E argues WHY PS works (the predictive-state argument: plain training has no gradient pressure to give the
added states a predictive role; S1 gives them one) and which sources carry it. Decide with Dirk after reading the
papers in section 11, before drafting.

## 9. Next action
Invoke the `thesis-section` skill. Then read the code in section 11 and write in chat, per PS stage (encoder sampling,
S1 target and decoder, S1 stopping rule, S2 regression, S3 hand-over), what the code does, with `file:function`, and
where it differs from the intended version. Show Dirk that list before any prose.

## 10. Acceptance criterion
III-E reads as Dirk's prose with its equations; each equation matches the code or a stated improvement with a todo
giving the current behaviour; `eq:ps2_s1` and the window figure show the target from tau+1; every `\cite` locator was
read in the PDF this session and logged in `CITATION-LOG.md`; a scratchpad copy of `Writing/` builds `review.tex`
and `main.tex` with no `!` lines and no undefined references.

## 11. Read these first
Code (the source of truth):
1. `model_augmentation/fit_systems/ps2_opening.py`: `calibration_split`, `_start`, `_sample` (encoder window vs target
   window), `_residual` (S1 target), `_s1_step`, `_cal_unexplained` (stopping rule, `eq:ps2_nu`), `_fit_map` (S2), `after_step`, `done`.
2. `scripts/gantry/gantry_dynamic/config.py` (the `ps2` field, l.~284, and its settings) and where `training.py` calls PS.
3. `Thesis-writeup/Results/11-10-2026/thesis_taf_lowpass_noise_linear_map/88767/config.json`: the values used in R5.
Papers (all local; read the passages, cite only what you read):
4. Hefny et al. 2015, `literature/added-states-training/hefny2015_supervised-learning-dynamical-system-learning.pdf`: two-stage regression, Sec. 2 (history and future disjoint).
5. Downey et al. 2017 (PSRNN), `literature/added-states-training/1705.09353.pdf`: Sec. 4.2, two-stage regression as initialisation, then BPTT.
6. Venkatraman et al. 2017 (Predictive-State Decoders), `literature/added-states-training/1709.08520.pdf`: decoder predicting the future to shape the latent state.
7. Hoekstra et al. 2026, encoder, `literature/augmentation/Encoder initialisation methods in the model augmentation setting.pdf`: Secs. 3.1, 3.2, 4.4.
8. Hoekstra et al. 2026, LFR augmentation, `literature/augmentation/hoekstra2026_lfr-augmentation-fp-models.pdf`: zero output layer, cost Eq. (22).
9. Beintema et al. 2023, `literature/experiment-design/Papers/Beintema_Schoukens_Toth_2023_Deep_subspace_encoders.pdf`: encoder and truncated simulation loss.
Context only, never evidence: `Thesis-writeup/Documentation/PS2-METHOD.md`, `FINAL-RUN-FIXES.md` item 5, `Thesis-writeup/Writing/README.md` (Math standard, source map for III-E).

## 12. Do not
Dirk's working rules (confirmed by him):
1. Never rewrite Dirk's sentences. Point out a problem with a fix of at most five words, or show a full proposal in chat as pasteable lines, for example:
   ```latex
   The target starts one sample after the encoder window, so the two share no sample.
   ```
2. No file edit unless he asks; ask "saved?" before every .tex edit, and after it tell him to run "File: Revert File".
3. Short answers, bullets, plain words, no em dashes.
4. Do not compile in `Writing/`; test in a scratchpad copy (copy `Writing/` without `build/`, put `refs.bib` in its `build/`, `latexmk -pdf -interaction=nonstopmode review.tex` and `main.tex`).
5. No git write operations.

## 13. Operational
- Python: `conda run -n GraduationProject python <script>`; multi-line snippets via a script file in the scratchpad.
- PDFs: Read tool with `pages`.

## 14. Delegation
None. At most one agent if a source passage cannot be located after a few searches.
