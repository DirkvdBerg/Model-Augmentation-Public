# Handoff: insert four agreed additions into thesis Section IV-B, changing nothing else
**From**: session of 2026-10-14 to 16 (Section IV) | **Branch**: Augmentation | **Effort suggested**: medium (precise, small edits; the content is decided)

## 1. Task
Dirk wants "another session to add the correct sentences". In
`Thesis-writeup/Writing/sections/04_obc.tex`, subsection IV-B (`\subsection{Extension to the gantry}`,
`\label{sec:obc_map}`), make exactly the four changes of section 9 and the one typo fix, and nothing else. The
paragraph is Dirk's own text: keep his sentences and words verbatim; insert only the bold parts given below. Before
editing, ask Dirk "is 04_obc.tex saved?" and wait for yes. After editing, tell him to run "File: Revert File".

## 2. Out of scope
- Any other wording change, restructuring, shortening or "improvement", in IV-B or anywhere else. Dirk has said he
  cannot trust edits that change more than asked; this is the main risk of the task.
- The section opening, IV-A, IV-C, the supervisor `\revnote`s, the draft-bullet lists, the outline blocks (moved to
  the end of the file on 2026-10-15 on purpose), other sections, code, git.
- Compiling `main.tex`.

## 3. Where things stand
Section IV has three subsections: IV-A Gyorok's construction (adopted), IV-B extension to the gantry (ours), IV-C
identification and recovery. Dirk has rewritten the IV-B opening paragraph himself (2026-10-15/16). Its current text
(read the file; it may have changed since):

```latex
Our baseline violates three requirements of the projection in
\eqref{eq:obc_gyorok}.
Its state update is nonlinear in the parameters, because
the next state update depends on $M(Y)^{-1}$ in \eqref{eq:aug_ct},
violating the linear requirement for the regressor.
Instead of the regressor the Jacobian of the baseline state update is taken
with respect to $\beta$,
as in the orthogonal regularisation method of
Gy\"or\"ok et al.~\cite[Sec.~4, Eq.~(15)]{gyorok2025l4dc}.
Like the regressor, the stacked Jacobian must have full column rank.
This requires identifiable parameters, which is why we use $\beta$.
The Jacobian is taken at linearisation point $\beta_{\mathrm{lin}}$ as
\end{wip}
\begin{equation} ... \label{eq:obc_sens} \end{equation}
where $\beta_{\mathrm{lin}}\in\R^{10}$ and
$\varphi:\R^{10}\times\R^6\times\R^3\to\R^{6\times10}$.
The Jacobian also depends on the state, while the regressor of
Gy\"or\"ok et al.\ uses only measured inputs and
outputs~\cite[Eqs.~(1), (2)]{gyorok2026obc}.
Since only the positions are measured, we evaluate the Jacobian at these
positions with the velocities reconstructed from them.
We use a state-space model predicting the next state instead of the output.
The projection is removed from the neural network correction of the
physical state at every step in the closed-loop simulation.
```
Further down IV-B: `eq:obc_tuples` (reconstructed tuples), `eq:obc_coef` (stacked Jacobian Phi, F_aug, theta_aux,
projected output, 0 = Phi^T F~_aug), `eq:obc_corr` (correction at every step, physical rows; added states not
corrected). Locate them in the file; their order may differ from the paragraph above.

## 4. Established and verified (the reasons behind each insertion; do not re-argue them)
- Gamma: Gyorok 2025 (L4DC, `gyorok2025l4dc`) Eqs. (17), (18) add an offset column Gamma to the linearised regressor.
  We leave it out; to first order it does not enter the parameter estimate (IV-C recovery result). Decided in D-254
  item 1 (amended 2026-10-14); a supervisor `\revnote` already explains it in detail, but the method text must state
  the choice in one sentence.
- Data, not simulation: Gyorok 2025 Sec. 3 (p. 7, paragraph before Remark 2) simulates the baseline when states are
  not measured. We reconstruct them from the measured positions instead (D-254 item 2).
- Full rank: Gyorok 2026 Assumption 1 assumes full column rank of his regressor on his data. For us, rank depends on
  the tuples (and beta_lin); identifiability alone does not guarantee it, and the thesis does not verify it for the
  real data, so it is stated as an assumption, like Gyorok's.
- Placement: the last two sentences of Dirk's paragraph introduce the third requirement (state update in closed loop)
  and belong as the lead-in of `eq:obc_corr`.

## 5. Assumed but not verified
- That the file still contains Dirk's paragraph as in section 3 and that `eq:obc_tuples`, `eq:obc_coef`,
  `eq:obc_corr` are present. Check first; if the text differs, apply the insertions to the corresponding sentences
  and tell Dirk what differed.
- `wip` markers: after `\end{equation}` of `eq:obc_sens` there may be no `\begin{wip}` while a later `\end{wip}`
  exists. If the `wip` environments are unbalanced, report it to Dirk; fix only if he says so.

## 6. Tried and failed
- Offering "improved" versions of Dirk's sentences or merging/restructuring them made him angry repeatedly; he wants
  his sentences copied verbatim with only agreed insertions (memory `feedback_keep_dirk_wording.md`).
- Inserting a citation where it does not support the exact claim (e.g. Gyorok Assumption 1 after a claim about the
  Jacobian or identifiability) was rejected; cite only what the passage says.

## 7. Achieved
IV-B structure and content agreed with Dirk; only these insertions remain.

## 8. The open question
Nothing blocked.

## 9. Next action
After "saved?" is confirmed, make exactly these edits (bold = added; everything else verbatim):

1. Typo: `The Jacobian is taken at linearisation point` -> `The Jacobian is taken at **a** linearisation point`.
2. New sentence directly after the where-clause of `eq:obc_sens` (after `...\to\R^{6\times10}$.`):
   `The offset column of~\cite[Eq.~(18)]{gyorok2025l4dc} is left out, because to first order it does not enter the estimate (Section~\ref{sec:obc_ident}).`
3. Append to Dirk's reconstruction sentence, before its final period:
   `... with the velocities reconstructed from them**, rather than simulating the baseline as in~\cite[Sec.~3]{gyorok2025l4dc}**.`
4. New sentence directly after the where-clause of `eq:obc_tuples`:
   `As Gy\"or\"ok et al.\ assume of their data~\cite[Assumption~1]{gyorok2026obc}, we assume that the tuples give the stacked Jacobian full column rank.`
5. Move, unchanged, the two sentences `We use a state-space model predicting the next state instead of the output.` and
   `The projection is removed from the neural network correction of the physical state at every step in the
   closed-loop simulation.` to directly before `\begin{equation}` of `eq:obc_corr` (inside the `wip` text that
   precedes that display), replacing nothing; if an older lead-in to `eq:obc_corr` with the same meaning is there,
   show both to Dirk and ask which stays.

## 10. Acceptance criterion
A diff of `04_obc.tex` shows exactly the five edits of section 9 and no other change (verify with
`git diff --word-diff Thesis-writeup/Writing/sections/04_obc.tex` restricted to IV-B); environments still balance
(`grep -c` of begin/end for wip, equation, aligned); Dirk confirms.

## 11. Read these first
1. `Thesis-writeup/Writing/sections/04_obc.tex`, subsection IV-B.
2. Memory `feedback_keep_dirk_wording.md` (auto-loaded index): copy Dirk's sentences verbatim.

## 12. Do not
- Reword, merge, split or reorder any of Dirk's sentences beyond item 5.
- Add explanations, citations or sentences beyond section 9.
- Touch the outline blocks at the end of the file or any `\revnote`.

## 13. Operational
Use the Edit tool for each change (exact-string replacement). The file has LF line endings. OneDrive can briefly lock
the file; retry the Edit. No compile, no git writes.

## 14. Delegation
None.
