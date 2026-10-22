# Prompt for Codex: check that every comment and docstring in `Clean/` is correct (after the comment pass)
**From**: Claude session of 2026-10-22, for Dirk to give to Codex | **Branch**: Augmentation |
**Effort suggested**: high

## 1. Task
A Claude session rewrote every comment and docstring in `<repo>/Clean/` and applied the renames you recommended in
`tasks/clean-code/comments-naming-review-codex.md` (decision D-282 at the end of `docs/decisions.md`). Comments do not
change numbers, so the equivalence harness cannot catch a wrong comment. Check every comment and docstring for
correctness and write `tasks/clean-code/comment-check-codex.md`. Per comment or docstring:
1. **Formula correctness:** does every formula match the code beside it exactly (operations, scaling, which
   quantities are normalised, shapes, indices, signs)? Use the thesis notation of
   `Thesis-writeup/Writing/sections/02_system_baseline.tex`, `03_augmentation.tex`, `04_obc.tex`, `05_setup.tex` and
   `99_appendix.tex`, and say where the comment's symbols differ from the thesis.
2. **Claim correctness:** is every statement about behaviour true (what a function computes, when it rebuilds,
   what it returns, what an error means, what the fix in an error message would do)?
3. **Standard (section 4):** anything that breaks it.
4. **Your earlier review:** is every correction of `comments-naming-review-codex.md` sections 5 and 6 applied, in
   particular the column scaling `D^-1` in the OBC coefficient and the affine basis `[Phi F_base(beta_lin)]` described
   as what it is (same span as `[Phi Gamma]`, different coefficients: code `(a, c)` corresponds to
   `(a + beta_lin c, c)` on the thesis basis), and every rename of your section 1 and 3?
Rank findings by consequence: a wrong formula or false claim first, then misleading wording, then style.

## 2. Out of scope
Editing any file except the check; behaviour changes; renames beyond reporting a missed one; the evaluation scripts;
the thesis; running training or the harness (it runs after this check).

## 3. Rules for this repository and where things stand
- Read-only except the check file. Quick imports, `pyflakes` or AST checks are fine.
- No git write operations; read-only git is fine. The pass is in commit `1107587`; the diff to review is
  `git diff 4e3c4c9 1107587 -- Clean Thesis-writeup/Code/Verification/clean-code`. Jan's last own
  `model_augmentation` commit is `c9f913a`.
- Never read `kamtin-data/Data Telica/` or `kamtin-data/Telica.mat`.
- No em dashes in your output.
- The writing session reports: syntax tree unchanged in 24 of 24 files apart from renames, 7 new `@added` decorators,
  error and print texts; strings that reach output files unchanged; pyflakes clean; no D-numbers, job ids, section
  numbers or dashes in comments; 25 `# CHANGED:`, 19 `@added`, 20 `THEORY`, 17 `HEURISTIC`. Spot-check these claims.

## 4. The comment standard (Dirk, 2026-10-21)
Clean, minimal, not AI-sounding; a first-time reader can follow the code without clutter. Module docstring of one or
two lines; one-line docstrings for functions called from outside their module; short step comments only in long
functions; inline comments only for a real why (a convention, a deliberate choice that looks wrong, the formula the
code implements). Formulas in short plain text with the thesis symbols; a paper name (Beintema et al., Hoekstra et
al., Gyorok et al., Drenth et al., Kessels for the known-controller rollout) only where the method comes from that
paper, once. Never: D-numbers, job ids, history, thesis section numbers or equation labels, restating the line,
capitals for emphasis, "Note:", "we", filler words, em dashes. Error messages: one sentence, what is wrong and what
to do. Markers: `@added`, `__project_origin__ = "added"`, five file headers on Jan's files, `# CHANGED:` only on
changed blocks in Jan's code and import groups, `# THEORY:` / `# HEURISTIC:` labels with short plain content.

## 5. Read these first
1. The diff of section 3, then `Clean/` itself (all 24 files).
2. `tasks/clean-code/comments-naming-review-codex.md` (your review).
3. The thesis sections of section 1, for notation.

## 6. Done when
- Every file has a verdict; every finding cites `Clean/file:line`, quotes the comment, states what is wrong with the
  code line or thesis equation it contradicts, and proposes the corrected text.
- Section 1 point 4 has an explicit applied or not applied per item of your earlier review.
Report everything; Dirk filters. Final message: the check path and the findings that are wrong formulas or false
claims (or "none").
