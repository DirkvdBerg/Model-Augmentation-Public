# Prompt for Codex: your own thesis-to-code map, with equation tests (step 4a)
**From**: Claude session of 2026-10-28, for Dirk to give to Codex at the same time as the Claude step-4 session |
**Branch**: Augmentation | **Effort**: high

## 1. Task
Read `tasks/handoffs/2026-10-26-clean-code-opt/00-overview.md` and item 2 of `04-comments-claude.md` (the map's
scope rules) first. Build, independently of Claude, the map between the thesis and the grey-box code, and prove the
math parts with tests. Claude builds its own map at the same time and merges both afterwards; do not read
`tasks/clean-code/thesis-code-map-claude.md`.
1. **Map** `tasks/clean-code/thesis-code-map-codex.md`, with the scope of `04-comments-claude.md` item 2: the active
   displayed equations and algorithm steps of `Thesis-writeup/Writing/sections/02_system_baseline.tex`,
   `03_augmentation.tex`, `04_obc.tex`, `05_setup.tex` and the method subsections of `99_appendix.tex` (state which
   you included and excluded); the code is the 24 grey-box files (`Clean/gantry_interconnect_dynamic.py`,
   `Clean/gantry_dynamic/`, `Clean/model_augmentation/`), not `Clean/blackbox/` or `Clean/data_generation/`.
   - forward: each thesis entry to the callable(s) that implement it (qualified name, `file:line`), or external with
     a pointer;
   - backward: every function, method, property and constructor to one or more entries, or infrastructure;
   - mismatches: symbols, signs, indices, normalisation, order of operations, defaults; a thesis step without code,
     code without an entry or infrastructure role. Say whether code or thesis would change.
2. **Equation tests** in `Thesis-writeup/Code/Verification/clean-code/equation_tests.py` (new file, the only file you
   create besides the map): for every mapped entry that is a formula a callable computes, one test that writes the
   thesis formula out directly in torch or numpy, evaluates the callable on random inputs of the right shape and
   dtype (float64, fixed seed, small batch), and compares, at 1e-12 relative unless the entry's order of operations
   differs (then state the tolerance and why). Name each test after its map row. A formula you cannot test without a
   training run is marked "not testable here" in the map with the reason.
   Run it once (`conda run -n GraduationProject python -m pytest -q <file>` or plain `python <file>`; seconds, CPU) and
   record per row: pass, fail with the largest difference, or not testable.

## 2. Rules for this repository
Read-only on everything except your two output files: no edits to `Clean/`, the thesis, mainline or the harness. No
training, no harness runs. No git write operations. Never read `kamtin-data/Data Telica/` or
`kamtin-data/Telica.mat`. Never compile the thesis. No em dashes or double hyphens as dashes.

## 3. Done when
Every in-scope thesis entry and every grey-box callable is in the map; every testable formula has a test that was
run, with its result in the map. Report every mismatch, small ones too. Final message: the two paths, the mismatch
count (code side, thesis side), and tests passed, failed and not testable.
