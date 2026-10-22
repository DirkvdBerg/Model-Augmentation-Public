# Prompt for Codex: review the comment and naming plan for `Clean/` (names first)
**From**: Claude session of 2026-10-22, for Dirk to give to Codex | **Branch**: Augmentation |
**Effort suggested**: high

## 1. Task
A Claude session wrote `tasks/clean-code/comments-naming-plan.md`: a rename list (R1 to R12), a marker plan, and two
sample files (`obc.py`, `model.py`) rewritten in a new comment style. Nothing in `Clean/` is edited yet; Dirk approves
the plan first. Dirk thinks the names are not yet right (for example `SSE_Interconnect_Weighted`). Review the plan
and write `tasks/clean-code/comments-naming-review-codex.md` with:
1. **Names (the main part).** For every row R1 to R12 and for every name the plan "considered and kept": is the new
   name clear to a reader who knows the method but not the history? Propose a better name wherever you can, with one
   line why. Check consistency across the folder: one convention for our class names (Jan's names such as
   `SSE_Interconnect`, `Static_ANN_Block` stay as they are), names that use the method's own terms (augmented model,
   physical baseline, added states, encoder, orthogonal projection, polish, servo error), no names that describe a
   removed option or history. Also list names the plan did not touch but should (only our own names, see section 4).
2. **Safety of each rename** against section 4: verify the plan's claims by grep, do not take them on trust.
3. **Marker plan** against the repository rule in section 3.
4. **Sample files:** do the comments follow the standard of section 5? Is every formula in a comment correct for the
   code beside it and written in the thesis notation (check `Thesis-writeup/Writing/sections/`)? Is the claim "same
   syntax tree as `Clean/` apart from the renames, only four error-message strings change" true?
5. **The plan's open decisions:** naming Kessels once in `closed_loop.py`; moving the `gantry_ss.py` docstring to the
   top of the file; adding `@added` decorators to methods (marker item 4). Give a recommendation for each.
Change nothing else.

## 2. Out of scope
Editing `Clean/` or any other file except the review; renaming Jan's names; changing behaviour; the evaluation
scripts; thesis text; server jobs.

## 3. Rules for this repository
- Read-only except the review file. Quick imports or syntax checks are fine; no training or harness runs.
- No git write operations. Read-only git is fine (Jan's last own commit of `model_augmentation` is `c9f913a`).
- Never read `kamtin-data/Data Telica/` or `kamtin-data/Telica.mat`.
- No em dashes in your output (not the Unicode character, not a double hyphen).
- Marker rule (`CLAUDE.md`): in `model_augmentation/`, a new class or function gets `@added`, a new file gets
  `__project_origin__ = "added"`, a changed block inside Jan's original code gets `# CHANGED: <reason>`.

## 4. What may be renamed
Our own names only: definitions marked `@added`, files marked `__project_origin__ = "added"`, our subclasses (such as
`SSE_Interconnect_Composed`) and our locals. Must stay: Jan's names (Dirk's design choice: keep Jan's classes and
files 1:1 with mainline); every name the evaluation scripts in `Thesis-writeup/Code/Evaluation/` import or read;
`state_dict` keys (so attribute names that hold modules stay); `config.json` keys; CLI flags. Pickled class paths do
not block a rename: the only dumps `Clean/` wrote are harness and smoke outputs, regenerated afterwards, and the final
checkpoints are component pairs stored by module position.

## 5. The comment standard (Dirk, 2026-10-21)
Clean, minimal, not AI-sounding; a first-time reader can follow the code, without clutter. Module docstring of one
or two lines; one-line docstrings for functions called from outside their module; short step comments only in long
functions; inline comments only for a real why (a convention, a deliberate choice that looks wrong, the formula the
code implements). Formulas in short plain text with the thesis symbols, not references; a paper name (Beintema et al.
for SUBNET, Hoekstra et al., Gyorok et al., Drenth et al.) only where the method comes from that paper, once. Never:
D-numbers, job ids, history, thesis section numbers or equation labels, restating the line, capitals for emphasis,
"Note:", "we", filler words, em dashes. Error messages: one sentence, what is wrong and what to do.

## 6. Read these first
1. `tasks/clean-code/comments-naming-plan.md`.
2. `Clean/` (all 24 files; the names in context).
3. The imports and attribute reads of `Thesis-writeup/Code/Evaluation/*.py` and of the harness in
   `Thesis-writeup/Code/Verification/clean-code/`.
4. `Thesis-writeup/Writing/sections/03_augmentation.tex` and `04_obc.tex` for the method's terms and symbols.

## 7. Done when
- Every rename row and every kept name has a verdict (keep the proposal, better name X, or do not rename) with a
  reason; the list of extra names to rename is complete for our own code.
- Every safety claim of the plan was checked by grep; every wrong claim is listed.
- The marker plan, the two samples and the three open decisions each have a verdict.
Report everything; Dirk filters. Final message: the review path and your top five naming changes.
