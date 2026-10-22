# Handoff: comment and naming pass over `Clean/`, then the final local harness rerun
**From**: session of 2026-10-21 | **Branch**: Augmentation | **Effort suggested**: xhigh (every comment and our own
names across 24 files, two approval points with Dirk, then the full harness)

## 1. Task
Make `<repo>/Clean/` read well for someone opening it from scratch: clear names and few, helpful comments, without
changing any number. Three phases, with Dirk's approval between them:
1. **Proposal (no edits to `Clean/` yet).** Write `tasks/clean-code/comments-naming-plan.md` with:
   (a) the rename list: every one of our own names you would rename, with the new name, one line why, and the check
   of section 4 that it may be renamed;
   (b) the marker plan of section 4;
   (c) a style sample: two files rewritten in full to the standard of section 3, as diffs in the plan (suggested:
   `Clean/model_augmentation/fit_systems/obc.py` and `Clean/gantry_dynamic/model.py`).
   Then stop and give Dirk the plan path. He approves, edits or rejects items.
2. **Apply** the approved renames (with every call site, the harness scripts that name them, and the `_dumps()` file
   name pattern) and the comment standard to all 24 files.
3. **Re-prove:** `pyflakes` and imports, then the final local harness rerun, layers 1, 2, 3, 3b, 3c and the
   `R6 lowpass --cpu --smoke` run, bitwise, with the H2 freshness check. Report in a section "Comments and naming"
   of `tasks/clean-code/implementation-report.md`, including the updated server command list.

## 2. Out of scope
- Any change to executable behaviour: statement order, values, dtypes, random draws, control flow. Renames and
  comments only (plus the edits renames force at call sites).
- Jan's classes, methods and names (design choice 3), and every name the evaluation scripts in
  `Thesis-writeup/Code/Evaluation/` import (the evaluation stays untouched).
- Mainline, the evaluation scripts, the thesis, server jobs, git.

## 3. The comment standard (Dirk, 2026-10-21)
Goal: clean, minimal, not AI-sounding; a reader who opens the code for the first time can follow it, but the code is
not cluttered.
- **Module docstring:** one or two lines on what the file does.
- **Docstring** for functions and classes called from outside their module: one line on what it computes; arguments
  only where shape or units are not obvious.
- **Step comments** only in the few long functions (`main`, `train`, the polish): short headers such as `# Adam`,
  `# rebuild the basis at the selected parameters`, `# L-BFGS polish`.
- **Inline comments only for a real why:** a convention (normalised coordinates, units), a deliberate choice that
  looks wrong (float32 constants, a random draw whose values are overwritten but must stay, `var` with B-1, the fixed
  loader seed), or the formula the code implements.
- **Formulas, not references:** where code implements a formula, write it in short plain text with the thesis
  symbols, e.g. `theta = (J^T J)^-1 J^T F`, `w = sigma_y / sigma_e`, `x_{k+1} = f_base(x_k, u_k) + f_aug(x_k, u_k)`.
  A paper name only where the method comes from that paper, once, where it is first used: SUBNET encoder (Beintema
  et al.), the augmentation structure (Hoekstra et al.), the orthogonal projection (Gyorok et al.), the LPV-LFR model
  (Drenth et al.).
- **Never:** D-numbers, job ids, history ("was", "replaces", "removed"), thesis section numbers or equation labels,
  comments that restate the line, capitals for emphasis, "Note:", "Important:", "we", filler words (ensure, robust,
  crucial, simply, explicitly), em dashes.
- **Error messages:** one sentence, what is wrong and what to do; no history. **Prints:** short and factual.
- Example, `build_model` today:
  `# CHANGED: C_aug / Parameterized_Linear_Output_Block removed: diag_gradient_routing confirms near-zero C_aug init (Frobenius ~1e-2) chokes ANN gradient by same factor.`
  becomes `out_phys = Linear_Output_Block(C=norm.Cd_norm, D=norm.Dd_np)   # fixed output map, not learned`.

## 4. Renames and markers
**May be renamed:** our own classes, functions, methods and module-level names (files marked
`__project_origin__ = "added"`, definitions marked `@added`, and our subclasses such as `SSE_Interconnect_Composed`),
provided the evaluation scripts do not import the name (grep `Thesis-writeup/Code/Evaluation/`). Pickled class paths
do not block a rename: the only dumps `Clean/` has written are harness and smoke outputs, regenerated in phase 3, and
the final checkpoints are component pairs (state_dict by module position, unaffected). Examples Dirk named:
`OBCLifecycle` (says nothing about its job), `SSE_Interconnect_Composed`. Also consider `linear_encoder_init_aug`,
`live_g_feed_forward_nn`, `ValHistory`, `NX_ANN`, `_combo`, `hp` naming. Keep `state_dict` keys unchanged: a rename of
an attribute that holds a module would change keys, so attribute names of modules stay.
**Must stay:** Jan's names (choice 3); evaluation-imported names; `state_dict` keys; `config.json` keys; CLI flags.
**Marker plan (propose, Dirk decides):** `CLAUDE.md` asks for `@added`, `__project_origin__ = "added"` and
`# CHANGED: <reason>` in `model_augmentation/`. Proposal to put in the plan: keep `@added` and `__project_origin__`
as they are; replace the per-definition `# CHANGED (clean): trimmed to the spec path` lines by one line at the top of
each of Jan's files (`# CHANGED: trimmed to the thesis runs; the full original is in model_augmentation/`); keep a
`# CHANGED:` only where behaviour differs from Jan's original, with a short reason about the current code and no
D-number. Keep `# THEORY:` / `# HEURISTIC:` labels (repo rule for identification code), written in the same plain
style: `# THEORY: <formula or paper>`, `# HEURISTIC: <short reason>`.

## 5. Established and verified
- `Clean/` (3,911 lines) passes every local layer bitwise; mainline goldens recaptured after D-280
  (`implementation-report.md`, "Optimisation (D-281)"). The harness bugs (meta written last, open `.npz` handles,
  missing `_clean/L3c/`) are fixed; 93 existing cases pass the new check.
- Stale comments already listed for this pass: end of `tasks/clean-code/optimise-claude.md` and the optimisation
  section of the report.

## 6. Assumed but not verified
- That no evaluation script imports a name on the rename list: settle by grep before proposing it.
- Runtime of phase 3: about 2 to 3 h, as before; time the first case of each layer.

## 7. Tried and failed
- Windows paths through cygpath created a stray folder; use literal Windows paths. OneDrive locks files right after
  a move; run layer-3c clean cases in the unsynced temp folder as before.

## 8. The open question
The marker plan and the rename list: Dirk decides at the end of phase 1.

## 9. Next action
Read `Clean/` and the evaluation imports, then write the plan of phase 1.

## 10. Acceptance criterion
- Phase 1: the plan has the full rename list with its checks, the marker plan, and the two sample files.
- Phase 3: `pyflakes` clean; every module imports from `Clean/` only; no comment in `Clean/` contains a D-number, a
  job id, a thesis section number or an em dash (a grep in the report shows it); layers 1 to 3c and the CLI smoke pass
  bitwise with H2 active.

## 11. Read these first
1. Section 3 and 4 of this handoff.
2. `tasks/clean-code/implementation-report.md` (latest sections).
3. `Clean/` itself.
4. The imports of `Thesis-writeup/Code/Evaluation/*.py` from `gantry_dynamic` and `model_augmentation`.

## 12. Do not
- Do not edit `Clean/` before Dirk approves the phase-1 plan.
- Do not change executable behaviour; do not edit mainline, the evaluation, `design.md`, the thesis.
- One numerical job at a time, in the background with streamed output.
- Do not read `kamtin-data/Data Telica/`.

## 13. Operational
Harness conventions as in the report (env `GraduationProject`, single-threaded CPU float64, goldens under
`simulations/clean-golden/`). Final message of phase 1: the plan path. Final message of phase 3: results, the grep
check, the server command list.

## 14. Delegation
None for phase 1 and the comment writing (one voice across the folder). At most one general-purpose subagent in phase
2 for the mechanical part of the renames (call sites, harness references). No Workflow.
