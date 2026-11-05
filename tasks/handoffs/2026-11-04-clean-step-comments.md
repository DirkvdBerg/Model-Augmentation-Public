# Handoff: write purpose-first step comments in Clean/ that a reader of the method can follow
**From**: session of 2026-11-04 | **Branch**: Augmentation (uncommitted changes in Clean/) | **Effort suggested**: medium (well-specified, small batches with approved examples; raise to high only if Dirk keeps rejecting the wording)

## 1. Task
Write short step comments inside the functions of five `Clean/` files, so that Dirk, who knows the thesis method
well but reads Python poorly, can follow each step of the code and say what it is for. Each comment explains the
**purpose of a step in the method** (what the encoder, S1, S2, the closed loop or OBC needs from it), not the
mechanics of the code. Write each comment yourself, from scratch, as a plain sentence a person would say when
explaining the code; read it back before showing it. Present proposals in small batches (about five) on a
full-file review page (tool in section 13), apply only on Dirk's "apply", and prove after each file that only
comments changed. Files, in order: `Clean/model_augmentation/fit_systems/ps2_opening.py`,
`Clean/model_augmentation/fit_systems/closed_loop.py`, `Clean/model_augmentation/fit_systems/interconnect.py`,
`Clean/model_augmentation/fit_systems/obc.py`, `Clean/gantry_dynamic/obc_gantry.py`. Comment only the steps that
are hard to follow without one; about 3 to 6 per file, not one per line.

## 2. Out of scope
- Code changes of any kind, including printed strings and function names (e.g. `closed_loop_free_run_rms_batch`):
  only comments and docstrings change.
- Docstring first lines and `# CHANGED:` reasons in these files: already reviewed and applied this session.
- Renaming terms across files ("PS opening" in code vs "PS initialisation" in the thesis; "calibration records" vs
  the thesis's "left out" records): Dirk has not decided; raise it at most once, in one line.
- The suspected code issue that the polish of a fresh run may run uncompiled after deepSI reloads `_best` (see 5):
  report only, never fix here.
- `kamtin-fp-model/`, `Clean/data_generation/`, any git write (standing rules).

## 3. Where things stand
Branch `Augmentation`, last commit 344b3aa. The working tree is dirty in `Clean/` (about 150 comment and docstring
changes from this session, all applied by Dirk's "apply"), `Clean/README.md` (data generation and data paths
removed), new `Clean/run_grey_box.sh` (Dirk edited it after creation: `CODE_DIR`, runs from the code folder), and
`Clean_for_Jan.zip` in the repo root (Clean/ without data_generation/, .ruff_cache/, __pycache__/). Nothing runs.
The 42 step-comment proposals of this session for the five files were NOT applied: Dirk found them messy (see 6).

## 4. Established and verified
- Every applied change was proven code-neutral with an AST compare that strips docstrings (tool in 13); all 28
  Python files in `Clean/` compile; both `.sh` files pass `bash -n`.
- Docstring first lines and `# CHANGED:` reasons in `interconnect.py`, `blocks.py`, `pre_encoder.py`,
  `torch_nets.py`, `utils.py`, `deepSI_corrections.py` were checked against Jan's original in
  `Model-Augmentation-Public-main/model_augmentation/` (same relative paths).
- Thesis terms confirmed in the .tex: "state equation" (`Thesis-writeup/Writing/sections/02_system_baseline.tex:670`),
  "PS initialisation" (4 uses, "PS opening" 0), calibration = last record of each class with at least 3 records,
  called "left out" (`03_augmentation.tex:1677`), score name NRMS_e (`05_setup.tex:561`).
- Compile speed-up "about 6.5x" is from `docs/decisions.md` (search "6.5x").

## 5. Assumed but not verified
- After `fit` reloads `_best` (deepSI `checkpoint_load_system` replaces `__dict__`), the simulator may come back with
  `_compiled = None`, so the polish of a fresh job would run uncompiled. Settled by a finished run's
  `lbfgs_outcome['used_compiled']`. Out of scope (section 2).
- `interconnect.py` fit's `validation` comment "validate on the training device instead of the CPU" has no
  documented reason; none found in `docs/`. Leave as is unless Dirk gives one.

## 6. Tried and failed
- Mechanical step comments (42 proposals, two reviewer agents, then shortened) -> Dirk: "messy", "this doesn't speak
  to me" -> they described what the code computes ("weighted by its number of starts", "zero rows above it skip the
  rows of the sources already placed") instead of what the step is for in the method; every rewrite under the
  constraints (one line, <=120 chars, no brackets, no symbols) compressed them further -> proposal JSONs
  `r6_*.json` in the scratchpad (section 13).
- Purpose-first comments on `_sample` (ps2_opening.py) -> Dirk: "the idea is better but the formulation has gone
  downhill" -> written fast after many rounds; choppy phrasing ("what the encoder turns into the state at k").
  The direction is right; the sentences must read naturally.
- Explaining syntax in comments ("zip pairs two lists", "with block") -> rejected as vague or beside the point.
- Rewriting a whole comment when Dirk asked about one word -> made it worse twice; change only what he points at.

## 7. Achieved
- Implemented and validated: about 150 comment/docstring changes over 20 files in `Clean/` (code-neutral, compiles).
- Side-by-side code pages (code left, notes right, line-by-line readings, rendered formulas) for interconnect,
  blocks, obc, obc_gantry, ps2_opening, closed_loop and the five black-box files: scratchpad `annotate/*_annotated.html`.
  These are Dirk's study material for the walkthrough with Jan; not shipped.

## 8. The open question
Nothing blocked. The style is set by the examples in section 9; if Dirk rejects a batch, ask which sentence and
rewrite only that one.

## 9. Next action
Read `Clean/model_augmentation/fit_systems/ps2_opening.py` in full, pick its 3 to 6 hardest steps, and write a
purpose-first comment for each in the style of these examples, which Dirk approved:

```python
    # step through the window one sample at a time: each step gives y_hat_k and the next model and controller state
    x, ys = x0, []
    for u_t, y_t in zip(u_data.unbind(1), y_data.unbind(1)):
        y_model, x, xc = step(x, xc, u_t, y_t, ctrl, obc_pass, pass_mats)
        ys.append(y_model)
    # all predictions as (windows, samples, outputs), and the final states
    return torch.stack(ys, dim=1), x, xc
```
```python
        def snap(ix, n):
            """S2 pairs from records ix (fit or calibration): X, U = state and input at k;
            T = added states at k + 1, the target."""
```
```python
        # CHANGED: loop moved to _propagate, so output_only can compute y_k before the controller sets u_k
```
What makes them good: a reader who cannot parse `zip`/`unbind`/`stack` still sees what the step means; plain words;
the purpose is in the sentence. Write the batch to an items JSON, build the full-file review page, give Dirk the
path, and wait for his "apply".

## 10. Acceptance criterion
Per file: Dirk says the comments read clearly and approves them ("apply"); the AST compare prints "code unchanged";
the file compiles. Done when all five files are approved, then `Clean_for_Jan.zip` is rebuilt (command in 13).

## 11. Read these first
1. `C:/Users/20203253/.claude/projects/C--Users-20203253-OneDrive---TU-Eindhoven-Graduation-Project-Baseline-FP-model-Baseline-LPV-Augmentation/memory/feedback_walkthrough_priorities.md`: Dirk's rejected words and the review-page format.
2. `memory/feedback_clean_comment_style.md` and `memory/feedback_code_comment_minimal_edits.md`: house comment style.
3. `Clean/README.md`: what each file is for, in thesis terms.
4. The scratchpad notes `annotate/ps2_notes.json` (and the other `*_notes.json`): verified plain explanations of every
   step, a good source of meaning (not of wording).
5. `Thesis-writeup/Writing/sections/03_augmentation.tex` (PS and closed loop) and `04_obc.tex`: the method words.

## 12. Do not
- Do not use: "free run" (say "closed-loop simulation of the whole record"), "eager" (say "uncompiled"),
  "contiguous", "pickles", "own design", "with block", filler such as "names that say what they do", thesis
  equation labels (eq:...), em-dashes, unexplained superscripts or symbols.
- Do not describe mechanics (slices, rows, offsets) when the purpose can be said instead.
- Do not batch more than about five proposals, and do not paste long Now/Proposed blocks in chat: use the page.
- Do not change "PS initialization" in `gantry_interconnect_dynamic.py` (Dirk's decision).
- Do not invent reasons; every claim in a comment must hold for the code.

## 13. Operational
Tools (Python, run with `conda run -n GraduationProject python -I <script>` from the folder below):
`C:/Users/20203253/AppData/Local/Temp/claude/64e1351a3d90a18a/3ea1a672-ac73-41ec-b4de-6491fe44170f/scratchpad/annotate/`
- `review_full.py <out.html> <source> <items.json> [...]`: one page with whole files and each change in place.
- `review.py apply <source> <items.json> all|1,3`: exact-text replacement; `review.py ast <before> <after>`: proof.
- Items JSON: list of `{"old": exact current text, "new": proposed text, "rule": what it fixes, "basis": the code it
  relies on}`. Before applying, `cp <source> before_x.py` for the AST proof.
- The pane may refuse to open new local HTML files; then give Dirk the absolute path.
This folder is temporary. If it is gone, recreate `review.py` and `review_full.py` (about 150 lines together:
side-by-side diff page with red/green word diff, checks for line length <= 120, em-dashes and eq: labels).
Rebuild the zip from the repo root:
`cd Clean && powershell.exe -NoProfile -Command "$src=(Get-Location).Path; $tmp=Join-Path $env:TEMP 'clean_for_jan'; if(Test-Path $tmp){Remove-Item -Recurse -Force $tmp}; New-Item -ItemType Directory $tmp | Out-Null; $dst=Join-Path $tmp 'Clean'; robocopy $src $dst /E /XD data_generation .ruff_cache __pycache__ /NFL /NDL /NJH /NJS | Out-Null; Compress-Archive -Path $dst -DestinationPath (Join-Path (Split-Path $src) 'Clean_for_Jan.zip') -Force; Remove-Item -Recurse -Force $tmp"`

## 14. Delegation
None for writing: the comments must be written by the session itself, one batch at a time. A fresh-reader subagent
only if Dirk asks for one, at most two, read-only, reporting findings that the session then rewrites itself.
