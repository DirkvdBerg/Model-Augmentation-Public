# Prompt for Codex: propose simplifications that make `Clean/` smaller and clearer with identical results (optimisation)
**From**: Claude session of 2026-10-21, for Dirk to give to Codex | **Branch**: Augmentation |
**Effort suggested**: high

## 1. Task
Dirk's definition of this step (2026-10-21): make `<repo>/Clean/` as small and clear as possible while every result
stays bit for bit identical. Read all of `Clean/` and PROPOSE simplifications; change nothing. You judged `Clean/`
before (`tasks/clean-code/judge-codex.md`, findings F8 to F15 are fixed now); this time look beyond dead code:
duplicated logic that can merge, functions or classes that can be simpler, structure inherited from mainline that
`Clean/` no longer needs in that form, parameters and indirection no caller needs, control flow that can be flatter,
names that hide what the code does. For each proposal give: the `Clean/` `file:line` range, the change, the lines
saved or the clarity gained, and the argument that results stay identical (section 4). Write the proposals to
`tasks/clean-code/optimise-codex.md`, ranked by gain. A Claude session does the same independently; do not read
`tasks/clean-code/optimise-claude.md`.

## 2. Out of scope
- Editing `Clean/` or anything else: Dirk approves the merged list first, then one session implements it.
- Speed optimisation that changes floating-point operation order, solver, vectorisation or batching (Dirk: identical
  results, not faster code).
- Comments and docstrings: their own pass comes later. Propose no comment wording.
- Mainline, the harness, the evaluation scripts, thesis text, server jobs.

## 3. Rules for this repository and where things stand
- Read-only: no edits, no training or simulation runs. Quick imports, `pyflakes` or syntax checks are fine.
- No git write operations of any kind. Read-only git is fine.
- Never read `kamtin-data/Data Telica/` or `kamtin-data/Telica.mat`. `kamtin-fp-model/` is read only.
- No em dashes in your output (not the Unicode character, not a double hyphen).
- `Clean/` (24 files, about 4,500 lines) passes every local harness layer bitwise against mainline on the current
  tree (`tasks/clean-code/implementation-report.md`, step 4 and "Judge fixes (step 4c)"). Your judge findings F1 to
  F16 and H1, H2 are implemented (D-278, D-279 at the end of `docs/decisions.md`).

## 4. Constraints every proposal must respect (or flag explicitly)
1. **Identical numbers:** same statements in the same order on the same values: no reordering of sums or products,
   no change of dtype or of where a float32 constant is materialised, no change of solver driver (gelsd stays), no
   change in the number or order of random draws (the draw order of `tasks/clean-code/design.md` 6.1, including draws
   whose values are overwritten and the draws with n_a = 0), the fixed `Dtype_DataLoader` `default_rng(0)` shuffle
   unchanged.
2. **Interfaces that must not change:** checkpoint `state_dict` keys and the component pair format, `config.json`
   keys, `RunConfig.from_config_json`, the CLI, the rerun folder rules (D-275 D6, periodic `_last.pth`), the names the
   evaluation scripts import (design section 5 and the implementation report).
3. **Markers:** `@added`, `__project_origin__ = "added"`, `# CHANGED:` and `# CHANGED (clean): trimmed to the spec
   path` stay meaningful; `# THEORY:` / `# HEURISTIC:` labels stay beside their formulas.
4. **Design choice 3** keeps Jan's classes, names and files 1:1 so each file diffs against its mainline original. A
   proposal that merges or renames Jan's classes or files is allowed, but mark it "breaks choice 3" so Dirk can weigh
   the gain against the marker clarity.
5. The step-4c robustness fixes (F1, F4, F5, F6, F16) stay in substance; their code may be simplified.
For each proposal state which constraints it touches and why it still holds them. If you are unsure that a change is
bit-for-bit safe, say so and say which harness layer would show it.

## 5. How to split the work
Work through three parts in turn (or in parallel, if your environment runs sub-agents; at most two besides
yourself):
- Part A: `Clean/gantry_interconnect_dynamic.py` and `Clean/gantry_dynamic/` (about 1,650 lines).
- Part B: `Clean/model_augmentation/fit_systems/interconnect.py`, `blocks.py`, `closed_loop.py` (about 1,220 lines).
- Part C: `fit_systems/lbfgs_polish.py`, `obc.py`, `pre_encoder.py`, `ps2_opening.py`, and
  `model_augmentation/systems/` and `utils/` (about 960 lines).
Then add the proposals that cross part boundaries (logic duplicated between parts, an indirection between
`gantry_dynamic/` and `fit_systems/`).

## 6. Read these first
1. `tasks/clean-code/implementation-report.md`.
2. `tasks/clean-code/design.md` sections 0, 1, 3 and 6.1.
3. Your `tasks/clean-code/judge-codex.md`, its "Minimality" section in particular (what must stay and why).
4. `docs/decisions.md` D-275, D-278, D-279.

## 7. Done when
- Every file of `Clean/` was read; each proposal carries `file:line`, the change, the gain, the constraint check of
  section 4 and a risk (low, medium, high).
- A summary at the top: total lines the proposals would save, and the five with the largest gain.
- Proposals that break design choice 3 are listed separately.
Report every proposal, small ones too; Dirk filters.

Format example (format only; invented content): `Clean/gantry_dynamic/example.py:NNN-MMM | two helpers compute the same index map; keep one | -18 lines | order of operations unchanged, no RNG, no interface | low`

Final message: the proposals path, the total saving and the top five.
