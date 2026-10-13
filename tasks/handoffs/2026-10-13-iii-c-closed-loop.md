# Handoff: write thesis Section III-C (closed-loop simulation) with Dirk, concise, math first, every choice justified
**From**: session of 2026-10-12/13 | **Branch**: Augmentation | **Effort suggested**: high (math, code and sources must agree; prose in Dirk's style)

## 1. Task
Write subsection III-C "Closed-loop simulation" (`\label{sec:aug_loop}`, `Thesis-writeup/Writing/sections/03_augmentation.tex`
l.574 to l.717, up to `\subsection{Identification problem}`) together with Dirk, as finished prose with its displayed
equations. Invoke the `thesis-section` skill first. Before anything else, read the full current thesis (all files in
`Writing/sections/`, `00_abstract` to `99_appendix`, in order), so III-C builds only on what precedes it and keeps what
later sections use from it. The subsection must, concisely:
1. state the math: the machine loop and the model loop (`eq:aug_two_loops`), their difference, the residual loop
   (`eq:aug_residual_loop`), in which the reference $r$ and $u^{\mathrm{ff}}$ cancel so the simulation runs with respect to
   the recorded data, and the step order (`eq:aug_cl_step`): output first, then error and controller input, then the
   next state, which avoids an algebraic loop through $D_{\mathrm{fb}}$;
2. justify why this works: the controller is known, linear and time invariant, so the loops can be subtracted;
3. state that no controller state has to be initialised: $x^c_\tau=0$ is the difference of model and machine controller
   states, not the machine controller state, and state the benefit of that;
4. justify simulating in closed loop instead of driving the model with the recorded input (both reasons, see sec. 4);
5. reference Kessels properly: what we adopt and exactly how we differ, read in the PDF.

Dirk's words (2026-10-13): "most important is stating the math, and why this works. what the benefit is of not having
to initialize the controller states, and reference what we have from bas kessels properly", "make clear its done with
respect to the reference, linear controller, no controller states init needed now", "dont state short, just concise".

## 2. Out of scope
- III-E (`sec:aug_ps2`): finished with Dirk on 2026-10-12/13. It relies on `eq:aug_residual_loop` at the window start
  ($x^c=0$, only $D_{\mathrm{fb}}$ acts); keep that label and form.
- All other subsections and section files: only touch them if a label needs it, and ask first.
- `refs.bib` (add an entry only after Dirk agrees), `util/format.tex`, `main.tex`, `.latexmkrc`, `.vscode/`.
- Code changes and runs.
- New docs. `Thesis-writeup/Documentation/CITATION-LOG.md` gets a row per passage read.

## 3. Where things stand
- III-C currently has `outline` lines (shown in the review build) and `wip` prose blocks around four displays:
  `eq:aug_two_loops` (l.~626), `eq:aug_residual_loop` (l.~647), `eq:aug_cl_step`, and the figure `fig:aug_closed_loop`
  (l.~715). Open todos in III-C: section number for `forssell1999revisited`; evidence on the thesis data that open-loop
  rollouts drift on $X$ and $Y$.
- The current prose already argues closed loop with Kessels (Eqs. (5.12), (5.13), Remark 5.6), the indirect approach
  (Forssell) and the SUBNET stability condition (`beintema2023subnet` Cond. 1, via `hoekstra2026lfr` Sec. 5.5).
- Dirk's decisions on content (2026-10-13):
  - Keep the current figure `fig:aug_closed_loop`; Dirk likes it, no extra figure.
  - Keep both displays: the two loops (`eq:aug_two_loops`) and the residual loop (`eq:aug_residual_loop`). The
    subtraction of the machine loop from the model loop is the argument: it shows that $r$ and $u^{\mathrm{ff}}$ cancel
    (simulation with respect to the recorded data), and that only the difference of the controller states
    $x^c=\hat x^{\mathrm{fb}}-x^{\mathrm{fb}}$ remains, which can start at zero, so the machine controller state never
    appears. With only the residual loop the reader sees the result but not why it is valid.
  - Keep both justifications for the closed loop (noise bias and the free axes / stability condition), and first check
    what earlier sections already state (e.g. Section II on the free integrators), so III-C points back instead of
    repeating.
  - Do not mention the multisine. It must be clear that the model input is the actuator force $u_{\mathrm{act}}$, the
    force into the plant (as III-A defines it).
  - Concise, not "short": everything needed, nothing the reader would not miss.

## 4. Established and verified
- Main-pipeline code path of the closed loop (the source of truth for every display):
  1. `scripts/gantry/gantry_interconnect_dynamic.py`: entry point.
  2. `scripts/gantry/gantry_dynamic/controller.py`: builds the controller bank and the `ClosedLoopSimulator` for the
     gantry (l.270 to 340); its docstring explains the forced step order, output before input (l.32). Read it for how the
     controller enters the pipeline (and how it is discretised over $T_s$).
  3. `model_augmentation/fit_systems/closed_loop.py`: `ControllerBank` (l.76), `closed_loop_rollout` and
     `_rollout_segment` (l.216 to 300: residual loop, $x^c=0$), `ClosedLoopSimulator` (l.435).
  4. `model_augmentation/fit_systems/interconnect.py`: `loss()` and `simulate()` hand the windows to the simulator.
- Code of the rollout: `model_augmentation/fit_systems/closed_loop.py::_rollout_segment` (l.216 to 242): per step
  `y_model = output_only(x)`, then `u_fb, xc = step(xc, y_t - y_model, ctrl)`, then `hfn(x, u_t + u_fb)`. So the model
  input is the recorded force plus the controller response to $y-\hat y$ (residual form), and the output is computed from
  the state before the input (no feedthrough in the plant output).
- `closed_loop.py::closed_loop_rollout` l.276 to 285: `xc = 0` at every window start, labelled HEURISTIC in the code. The
  code comment states: not Kessels' Remark 5.4, which reconstructs the absolute controller state from $(\bar y,\bar r)$
  for the lumped-$r$ form; in the residual form $y_{\text{data}}-y_{\text{model}}$ does not exist before the window, so
  $x^c=0$ defines the initial condition, and it is the unique value for which the correction vanishes when the model is
  exact.
- `u_act` in the data is the total force into the plant: controller output plus the injected multisine, which sits in
  the feedforward slot; there is no real feedforward (Dirk, 2026-10-13; base generator
  `Matlab-scripts/Augmentation/data/gtd_run_simulation.m` l.33 to 34: `u_total = u_fb + f_ms`). So $u^{\mathrm{ff}}$ in
  `eq:aug_two_loops` is that injected force; per Dirk, do not name the multisine in III-C.
- `CITATION-LOG.md` row (2026-10-09): `kessels2025ai` Eqs. (5.12), (5.13a) to (5.13d), thesis pp. 156 to 157 (closed-loop
  truncated-window criterion, encoder, controller in the rollout, error against the model output).
- Code comment `closed_loop.py` l.234: Kessels Eq. (5.13d), p. 157, forms the controller error against the model output.

## 5. Assumed but not verified
- `docs/references.md` l.899 summarises Kessels: Remark 5.4 reconstructs the absolute feedback-controller state;
  Remark 5.6 reports that direct open-loop SUBNET identification from his closed-loop data failed, attributed to noise
  or input correlation bias. Read Remarks 5.4 and 5.6 and Eqs. (5.12), (5.13) in
  `literature/augmentation/kessels2025_ai-control.pdf` before citing them; the contrast "Kessels reconstructs the
  controller state, we need none" is the main Kessels point Dirk wants and rests on Remark 5.4.
- The claim in the current text (l.~662) that the residual form equals the model loop "when the recorded input was
  produced by this controller from the same output samples and initial state": derive it from the two loops and state
  exactly what the $x^c_\tau=0$ choice drops (the difference of the controller states at $\tau$).
- `forssell1999revisited`: the section number of the indirect approach is unverified in the Automatica version (todo in
  the file); the local PDF is the LiU report, with a ligature-corrupted text layer (`docs/references.md` l.555).

## 6. Tried and failed
- Proposals ending in `% CHANGED` / `% NEW` comment out the rest of the line and twice broke Dirk's build. Give pasteable
  lines without markers.
- A `\\` as the first line of a paragraph (e.g. right after `\begin{wip}`) gives "There's no line here to end", stops
  latexmk after one pass, leaves "??" references and stops the editor PDF from refreshing. New paragraph = empty line.
- Bash heredocs for .tex edits turned `\b` and `\e` into control characters; use the Edit or Write tool.
- In III-E, rewriting Dirk's sentences in my own words was rejected repeatedly; so were extra todos and analyses when he
  asked a wording question.

## 7. Achieved
None for III-C. III-E done (prose with Dirk, displays `eq:ps2_s1`, `eq:ps2_nu`, `eq:ps2_s2`, `eq:ps2_handover`, stop rule
in Appendix `app:ps_stop`, notation $\hat x^{\mathrm n}_{\tau|\tau}$ and $\bar x_{k_i+1|k_i+1}$ as in `eq:aug_loss`).

## 8. The open question
Nothing blocked. Decide with Dirk, after reading the thesis and Kessels: which claims III-C must establish (skill step
2a), starting from the five points in sec. 1.

## 9. Next action
Read the full thesis in order, then the code path of sec. 4 (`gantry_dynamic/controller.py`, `closed_loop.py` l.216 to
300) and Kessels Eqs. (5.12), (5.13), Remarks 5.4 and 5.6.
Then write in chat, for Dirk, the 2a list: the claims III-C must establish, what earlier sections already state that
III-C should point to, and where the current III-C text and the code or Kessels disagree. No prose before he agrees.

## 10. Acceptance criterion
III-C reads as Dirk's prose; each display matches `closed_loop.py`; the text states with respect to the reference, the
known linear time-invariant controller, $x^c_\tau=0$ with its benefit and what it drops, and both closed-loop reasons
without repeating earlier sections; every `\cite` locator was read in the PDF this session and logged in
`CITATION-LOG.md`; labels used by III-D and III-E (`eq:aug_cl_step`, `eq:aug_residual_loop`, `eq:aug_two_loops`,
`fig:aug_closed_loop`, `sec:aug_loop`) still resolve; a scratchpad copy of `Writing/` builds `main.tex` and `review.tex`
with no `!` lines.

## 11. Read these first
1. `Thesis-writeup/Writing/sections/` all files in order: what is established before III-C and what uses it after.
2. `model_augmentation/fit_systems/closed_loop.py` l.216 to 300: the rollout and the $x^c=0$ reset with its reasoning.
3. `literature/augmentation/kessels2025_ai-control.pdf`, Ch. 5: Eqs. (5.12), (5.13), Remarks 5.4, 5.6 (pp. ~156 to 158).
4. `scripts/gantry/gantry_dynamic/controller.py` l.1 to 60 and 270 to 340: how the controller enters the main pipeline.
5. `Thesis-writeup/Writing/README.md` "Math standard" and the III-C row of the source map.

## 12. Do not
Dirk's working rules (confirmed by him in this and earlier sessions):
1. Use Dirk's own wording wherever possible. When a fix needs new words, reuse his verbs and phrases from the surrounding
   text; he would never write "evolve", for example, and an example is never the full scope of a rule.
2. Fixes of at most a few words, or a full proposal as pasteable lines. For example, when he asks "is this clear?":
   ```latex
   When S2 is completed the optimization continues minimizing \eqref{eq:aug_loss}.
   ```
   Answer exactly the wording question asked; no extra analyses, todos or alternatives unless he asks.
3. No file edit unless he asks; ask "saved?" before every .tex edit, and after it tell him to run "File: Revert File".
4. Do not compile in `Writing/`; test in a scratchpad copy (copy `Writing/` without `build/`, put `refs.bib` in its
   `build/`, `latexmk -pdf -interaction=nonstopmode main.tex` and `review.tex`).
5. No git write operations.

## 13. Operational
- Python: `conda run -n GraduationProject python <script>`; multi-line snippets via a script file in the scratchpad.
- PDFs: Read tool with `pages`.

## 14. Delegation
None. At most one agent if a Kessels passage cannot be located after a few searches.
