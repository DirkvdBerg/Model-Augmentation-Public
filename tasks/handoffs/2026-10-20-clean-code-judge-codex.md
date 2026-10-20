# Prompt for Codex: judge the clean minimal implementation in `Clean/` (step 4b)
**From**: Claude session of 2026-10-20, for Dirk to give to Codex | **Branch**: Augmentation |
**Effort suggested**: high

## 1. Task
Dirk's goal: "a new self contained folder only needing the required python files and to be written with minimal
code" for the `scripts/gantry/gantry_interconnect_dynamic.py` pipeline. A Claude session has written it in
`<repo>/Clean/` (24 files, about 4,450 lines) and an equivalence harness shows it reproduces mainline bit for bit on
every local test. You judge the code itself, as one reader holding the whole picture. Answer four questions:
1. **Faithful to the design:** does every file of `Clean/` implement `tasks/clean-code/design.md` (revised; its
   changelog is at the top) as amended by section 4? Is every deviation in the implementation report justified?
2. **Faithful to the thesis:** does the code implement the method the thesis describes, per the thesis-to-code
   tables of step 1 and the rulings since (section 4)? Report any place where code and thesis now disagree.
3. **Minimal:** what is still more code than the method needs: dead branches, unused parameters or fields,
   generality no spec path uses, duplicated logic, leftover diagnostics, defensive code that re-checks values the
   code builds itself (rule G of the design, D5)?
4. **Robust where the harness cannot look:** paths no local test reaches (GPU and `torch.compile`, the polish
   compiled try and eager fallback, CUDA OOM handling, OBC rank-loss, resume and polish-only on the server, the
   periodic `_last.pth`, CLI and root arguments). Read them against mainline: is each copied faithfully, or could it
   fail or diverge?
Write the verdict to `tasks/clean-code/judge-codex.md` (the only file you create).

## 2. Out of scope
- Comment style: comments get their own pass later (Dirk). Judge only that the required markers are present (see
  section 3) and that kept `# THEORY:` / `# HEURISTIC:` labels survived. Do not report comment wording.
- Numerical equivalence on the tested paths: the harness has settled it (section 5). Do not re-derive it.
- Editing any file, running training or the harness, the evaluation scripts, the thesis text, server jobs.
- Re-deciding the decisions of section 4.

## 3. Rules for this repository
- Read-only: no edits, no training or simulation runs. Quick imports, `pyflakes` or syntax checks are fine.
- No git write operations of any kind. Read-only git is fine.
- Never read `kamtin-data/Data Telica/` or `kamtin-data/Telica.mat`. `kamtin-fp-model/` is read only.
- No em dashes in your output (not the Unicode character, not a double hyphen).
- Markers in `model_augmentation/` (and so in `Clean/model_augmentation/`): `@added` on a new class or function,
  `__project_origin__ = "added"` on a new file, `# CHANGED: <reason>` on a changed block inside Jan's original code;
  per D-275 D4, every Jan definition that is only trimmed carries `# CHANGED (clean): trimmed to the spec path`.
  Missing or wrong markers are findings.

## 4. Decisions that bind the code (all in `docs/decisions.md`, end of file)
- **D-274**: PS2 S2 keeps its start as a candidate; only K1 records (18/6/11); the OBC polish is kept only if it
  beats the selected checkpoint's score, otherwise the rebuilt model is kept; the S2 cost keeps `T.var(0)`.
- **D-275 and amendments 1 to 4**: D1 optimizer property kept; D2 `RunConfig.from_config_json` reads a mainline
  `config.json`; D3 each code resumes only its own deepSI dumps, evaluation uses the component pair; D4 markers; D5
  guard rule applied consistently; D6 a rerun continues inside its own run folder (previous outcome to
  `previous/<rerun_id>/`, `RESUMES` list in `config.json`); amendment 2 periodic `_last.pth` at every validation;
  amendment 3 Adam moments restored from a dump (`d.get('optimizer') or d.get('_optimizer')`); amendment 4 the
  evaluation scripts are untouched, no design-time name checker.
- **Dirk, 2026-10-20**: folder `<repo>/Clean/`; `data_root` and `out_root` arguments with defaults
  (`<repo>/Thesis-writeup/Data`, `<repo>/simulations/gantry_clean/`); CLI with default R1 lowpass on cuda; resume and
  polish-only kept.
- **Thesis wording settled 2026-10-19** (so the code is not wrong on these): the appendix says the numerator uses
  Horner and d(Y) is evaluated term by term; "measured" was removed before the scheduling variable Y; the NRMS of
  100 % means predicting the mean servo error.

## 5. What is established
- `tasks/clean-code/implementation-report.md`: per layer PASS, bitwise, on CPU float64 single thread: static check
  (pyflakes, imports from `Clean/` only), layers 1 and 2 (16 cases), layer 3 (5 cases, both arms, n_a 0/2/8, PS2
  on/off, both noise modes), 3b (polish rejection, 2), 3c (resume from `_best` and `_last`, polish-only, 5),
  `R6 lowpass --cpu --smoke`, and a local test of the layer-4 loader. Its "Deviations from design.md" section lists
  every place the code differs from the design, with the cause; judge each one.
- Not yet run (server): layers 4 and 4b (final checkpoints) and the compiled GPU smokes of design 6.4. These are the
  paths your question 4 covers by reading.
- The harness lives in `Thesis-writeup/Code/Verification/clean-code/`; it is verification tooling, not part of
  `Clean/`. Judge it only where it could make a failing `Clean/` look like a pass.

## 6. Read these first
1. `tasks/clean-code/implementation-report.md`.
2. `tasks/clean-code/design.md` (changelog first, then sections 0, 1, 2, 3).
3. `docs/decisions.md` D-274 and D-275 with its amendments.
4. `tasks/clean-code/comparison.md` (step-1 thesis-to-code findings and rulings) and Part B of
   `tasks/clean-code/inventory-codex.md` (your own step-1 table).
5. `Thesis-writeup/Writing/sections/02_system_baseline.tex`, `03_augmentation.tex`, `04_obc.tex`, `05_setup.tex`.

## 7. Done when
- Each of the 24 files of `Clean/` has a short verdict (faithful, minimal, robust) and its findings.
- Every finding cites the `Clean/` `file:line`, the mainline `file:line` or design line it is judged against, the
  concrete consequence (what breaks, diverges, or is superfluous), the fix, and a severity (high, medium, low).
- Every deviation in the implementation report is marked justified or not, with the reason.
- Question 2 ends with a list of every remaining code-thesis disagreement, or "none found".
Report everything you find, including small items, ranked by consequence; Dirk filters.

Format example (format only; file, line and content are invented): `Clean/gantry_dynamic/example.py:NNN | keeps a keyword argument that no caller in Clean/ passes (mainline example.py:MMM) | superfluous parameter | remove the argument and its default | low`

Final message: the path of the verdict and the three findings with the largest consequence.
