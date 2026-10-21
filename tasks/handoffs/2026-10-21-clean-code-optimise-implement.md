# Handoff: implement the approved optimisation of `Clean/` and re-prove equivalence (optimisation, step 2)
**From**: session of 2026-10-21 | **Branch**: Augmentation | **Effort suggested**: xhigh (about 150 small edits across
24 files, each of which must keep results bit for bit, then the full local harness)

## 1. Task
Implement every accepted item of `tasks/clean-code/optimise-merged.md` in `<repo>/Clean/` (Dirk's goal: smaller and
clearer, identical results), then prove with the local harness that results are still bit for bit identical to
mainline. In this order:
1. **Recapture the mainline goldens.** Mainline `scripts/gantry/gantry_dynamic/training.py` changed on 2026-10-21
   (D-280), so the harness fingerprint (`hcommon.py`, the 19 copied mainline files) refuses every existing golden.
   Recapture layers 1, 2, 3, 3b and 3c from mainline before editing `Clean/`. Expect the same numbers as before: the
   D-280 change only adds an eager validation on a path no local case takes, except the spent-budget case of 3c,
   where it adds one validation call before the polish.
2. **Edit `Clean/`** item by item, in groups by file. After each file group, run `pyflakes` and the import check, and
   run layers 1 and 2 on two cases (one obc, one u) as a quick gate before the next group.
3. **Harness edits** for the O8 items only (B5, B41, B51, C3, C19b): each follows a renamed or dropped field.
4. **Re-prove:** layers 1, 2, 3, 3b, 3c and the `R6 lowpass --cpu --smoke` run, bitwise, with the H2 freshness check.
5. **Report:** a section "Optimisation (D-281)" in `tasks/clean-code/implementation-report.md`: per item implemented,
   reverted (with the failing layer and cause) or skipped (with the reason); the new line count per file; layer
   results and runtimes; the updated server command list.

## 2. Out of scope
- The rejected items O1 to O6 and O9 (D-281): do not implement them, not even partly.
- Comments and docstrings beyond what an edit forces (moving a `# THEORY:`/`# HEURISTIC:` label with its formula,
  keeping the markers). The stale-text list of `optimise-claude.md` is for the later comment pass.
- Mainline, the evaluation scripts, thesis text, server jobs, git.

## 3. Where things stand
`Clean/` (24 files, 4,506 lines) passes every local layer bitwise against mainline as of step 4c, but those outputs
are now refused by the fingerprint (mainline D-280). Mainline carries D-274, D-275 amendment 3, D-278 and D-280.
Server layers 4/4b and the GPU smokes run last, after the ultrareview and the comment pass. No run in flight.

## 4. Decisions that bind this step
- **D-281** (end of `docs/decisions.md`): O1 to O6 rejected (keep the `optimizer` property, the `least_squares_start`
  self-checks, `_json_safe`, the PS2 `.grad` save/restore, the KeyboardInterrupt handler, `fixed_System_data`);
  O7 update the two `reason` strings naming removed options; O8 harness edits allowed for B5, B41, B51, C3, C19b;
  O9 nothing that breaks design choice 3 (Jan's classes, names and files stay 1:1; take Codex's P06/P07 instead of
  Claude's choice-3 parse alternatives); O10 B1, B2, B26 accepted.
- **Constraints of every item** (both proposal handoffs, section 4): same statements in the same order on the same
  values; same dtype and float32 materialisation points; gelsd stays; same random draws in number and order
  (including overwritten draws and n_a = 0); `Dtype_DataLoader` `default_rng(0)` unchanged; checkpoint keys, the
  component pair, `config.json` keys, `from_config_json`, the CLI, the rerun rules and the evaluation import names
  unchanged; markers and labels kept; the step-4c robustness fixes kept in substance.
- **A failing item is reverted, never toleranced.** If an item breaks a layer bitwise, revert it, record the layer
  and the cause, and continue. Do not loosen a tolerance or edit a golden to make an item pass.

## 5. Established and verified
- The proposals with their safety arguments: `optimise-claude.md` (primary IDs) and `optimise-codex.md`; the merged,
  approved list and ID mapping: `optimise-merged.md`.
- Harness method and conventions: `implementation-report.md` (steps 4 and 4c), including the OneDrive workaround for
  layer-3c clean reruns (deviation 7(f)) and the lstsq solver pin (7(e)).

## 6. Assumed but not verified
- That every accepted item is bitwise safe: the proposals argue it statically; the harness decides.
- B1, B2 and B26 change the compiled graph: the CPU layers cannot prove them; the 6.4 GPU smokes at the end do.
  List them prominently in the report as "compiled path unproven until 6.4".
- Runtimes: the full local harness took about 2 to 3 h; mainline recapture adds about as much. Time the first case
  of every layer before launching the rest.

## 7. Tried and failed
- Windows paths through cygpath created a stray folder; use literal Windows paths for `LOCALAPPDATA` and roots.
- OneDrive locks `val_hist.npz` right after a rerun archives it; run layer-3c clean cases in the unsynced temp folder
  as step 4c does.

## 8. The open question
Nothing blocked.

## 9. Next action
Read `optimise-merged.md`, then recapture the first mainline golden (layer 1, 221-lowpass) and time it.

## 10. Acceptance criterion
- Every accepted item of `optimise-merged.md` appears in the report as implemented, reverted (with layer and cause)
  or skipped (with reason); no rejected item is implemented.
- `pyflakes` reports no warning in `Clean/`; every module imports with only `Clean/` on the path.
- Layers 1, 2, 3, 3b, 3c (all cases) and the CLI smoke pass bitwise against the recaptured mainline goldens, with
  the H2 freshness check active.
- The report gives the new total line count and the line count per file.

## 11. Read these first
1. `tasks/clean-code/optimise-merged.md`.
2. `tasks/clean-code/optimise-claude.md` (the item details and safety arguments).
3. `docs/decisions.md` D-281 and D-280.
4. `tasks/clean-code/implementation-report.md`.
5. `tasks/clean-code/optimise-codex.md` (for the P and Q items taken from Codex).

## 12. Do not
- Do not edit mainline, the evaluation scripts, `design.md` or the thesis; harness edits only for O8.
- Do not run more than one numerical job at a time; launch runs in the background with streamed output.
- Do not write into `Thesis-writeup/Results/`, the mainline run folders, `~/.deepSI/checkpoints` or data folders.
- Do not read `kamtin-data/Data Telica/`.

## 13. Operational
Same environment and conventions as steps 4 and 4c: env `GraduationProject`, single-threaded CPU float64 harness
processes, goldens under `simulations/clean-golden/`. Final message: the report section, the line counts, the
reverted items and the leftovers for Dirk.

## 14. Delegation
At most one general-purpose subagent, for the O8 harness edits and the per-file line counts. None for the `Clean/`
edits, which need one context to keep operation order. No Workflow.
