# Handoff: propose simplifications that make `Clean/` smaller and clearer with identical results (optimisation, Claude)
**From**: session of 2026-10-21 | **Branch**: Augmentation | **Effort suggested**: xhigh (a design-level read of 24
files with a numerical-safety argument per proposal)

## 1. Task
Dirk's definition of this step (2026-10-21): make `<repo>/Clean/` as small and clear as possible while every result
stays bit for bit identical. Read all of `Clean/` and PROPOSE simplifications; change nothing. Look beyond dead code
(the Codex judge already removed that, F8 to F15): duplicated logic that can merge, functions or classes that can be
simpler, structure inherited from mainline that `Clean/` no longer needs in that form, parameters and indirection no
caller needs, control flow that can be flatter, names that hide what the code does. For each proposal give: the
`Clean/` `file:line` range, the change, the lines saved or the clarity gained, and the argument that results stay
identical (section 4). Write the proposals to `tasks/clean-code/optimise-claude.md`, ranked by gain. A Codex session
does the same independently; do not read `tasks/clean-code/optimise-codex.md`.

## 2. Out of scope
- Editing `Clean/` or anything else: Dirk approves the merged list first, then one session implements it.
- Speed optimisation that changes floating-point operation order, solver, vectorisation or batching (Dirk: identical
  results, not faster code).
- Comments and docstrings: their own pass comes later. Propose no comment wording.
- Mainline, the harness, the evaluation scripts, thesis text, server jobs, git.

## 3. Where things stand
`Clean/` (24 files, about 4,500 lines) passes every local harness layer bitwise against mainline on the current tree
(`tasks/clean-code/implementation-report.md`, sections for step 4 and "Judge fixes (step 4c)"). The Codex judge
(`tasks/clean-code/judge-codex.md`) and its fixes are done. Server layers run last, after this step, the ultrareview
and the comment pass. No run in flight.

## 4. Constraints every proposal must respect (or flag explicitly)
1. **Identical numbers:** same statements in the same order on the same values: no reordering of sums or products,
   no change of dtype or of where a float32 constant is materialised, no change of solver driver (gelsd stays), no
   change in the number or order of random draws (the draw order of `design.md` 6.1, including draws whose values are
   overwritten and the draws with n_a = 0), the fixed `Dtype_DataLoader` `default_rng(0)` shuffle unchanged.
2. **Interfaces that must not change:** checkpoint `state_dict` keys and the component pair format, `config.json`
   keys, `RunConfig.from_config_json`, the CLI, the rerun folder rules (D-275 D6, periodic `_last.pth`), the names the
   evaluation scripts import (design section 5 and `implementation-report.md`).
3. **Markers:** `@added`, `__project_origin__ = "added"`, `# CHANGED:` and D4's `# CHANGED (clean): trimmed to the
   spec path` stay meaningful; `# THEORY:` / `# HEURISTIC:` labels stay beside their formulas.
4. **Design choice 3 of `design.md`** keeps Jan's classes, names and files 1:1 so each file diffs against its mainline
   original. A proposal that merges or renames Jan's classes or files is allowed, but mark it "breaks choice 3" so
   Dirk can weigh the gain against the marker clarity.
5. Robustness fixes of step 4c (F1, F4, F5, F6, F16) stay in substance; their code may be simplified.
For each proposal state which constraints it touches and why it still holds them. If you are unsure that a change is
bit-for-bit safe, say so and say which harness layer would show it.

## 5. Established and verified
- Decisions: D-274, D-275 with amendments 1 to 4, D-278, D-279 (end of `docs/decisions.md`).
- The harness (`Thesis-writeup/Code/Verification/clean-code/`) will prove each accepted proposal; its freshness check
  (H2) refuses old outputs once `Clean/` changes, so every implemented proposal is retested.

## 6. Assumed but not verified
- That a large share of the remaining size is mainline structure rather than method: settle by reading.

## 7. Tried and failed
None for this task.

## 8. The open question
Nothing blocked.

## 9. Next action
Start the two subagents of section 14, then read part A yourself.

## 10. Acceptance criterion
- Every file of `Clean/` was read; each proposal carries `file:line`, the change, the gain, the constraint check of
  section 4 and a risk (low, medium, high).
- A summary at the top: total lines the proposals would save, and the five with the largest gain.
- Proposals that break design choice 3 are listed separately.
Report every proposal, small ones too; Dirk filters.

Format example (format only; invented content): `Clean/gantry_dynamic/example.py:NNN-MMM | two helpers compute the same index map; keep one | -18 lines | order of operations unchanged, no RNG, no interface | low`

## 11. Read these first
1. `tasks/clean-code/implementation-report.md`.
2. `tasks/clean-code/design.md` sections 0, 1, 3 and 6.1.
3. `tasks/clean-code/judge-codex.md` (what was already trimmed and why some code must stay, its "Minimality" section).
4. `docs/decisions.md` D-275 and D-278.

## 12. Do not
- Do not edit any file except the proposals file; do not run training or the harness.
- Do not read `kamtin-data/Data Telica/` or the Codex proposals.

## 13. Operational
Quick imports or `pyflakes` checks are fine. Final message: the proposals path, the total saving and the top five.

## 14. Delegation
Split by package, Dirk asked for multiple agents: two general-purpose subagents, each with this brief limited to its
part, while you take part A:
- Part A (you): `Clean/gantry_interconnect_dynamic.py` and `Clean/gantry_dynamic/` (about 1,650 lines).
- Part B (subagent 1): `Clean/model_augmentation/fit_systems/interconnect.py`, `blocks.py`, `closed_loop.py`
  (about 1,220 lines).
- Part C (subagent 2): `fit_systems/lbfgs_polish.py`, `obc.py`, `pre_encoder.py`, `ps2_opening.py`, and
  `model_augmentation/systems/` and `utils/` (about 960 lines).
Then check their proposals yourself against section 4, and add proposals that cross part boundaries (logic
duplicated between parts, an indirection between `gantry_dynamic/` and `fit_systems/`), which no single part sees.
