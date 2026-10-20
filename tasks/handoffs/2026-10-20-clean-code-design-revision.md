# Handoff: revise the clean-code design document with the merged critique (end of step 3)
**From**: session of 2026-10-20 | **Branch**: Augmentation | **Effort suggested**: xhigh (many coordinated edits to
a 1000-line design, each to be checked against mainline code)

## 1. Task
Revise `tasks/clean-code/design.md` so that it carries every item of `tasks/clean-code/critique-merged.md`: the six
rulings D1 to D6 (section A, accepted by Dirk, D-275) and the 23 fixes (section B). Edit `design.md` in place. Each
fix becomes a concrete change in the section it belongs to (layout, mapping rows, config, data, evaluation interface,
equivalence plan, open choices), with the mainline `file:line` it rests on; where a fix says "keep X, or list the
edits", choose the smaller exact option and state the choice. Then add a short changelog at the top of `design.md`
(one line per merged item, pointing to where it landed). The revised document is the input for step 4
(implementation); it must be implementable literally, without dangling names.

## 2. Out of scope
- Writing or copying code, creating `<repo>/Clean/`, capturing golden outputs, running anything beyond quick imports.
- Re-deciding D1 to D6, D-274 or the four decisions of 2026-10-20 (folder `<repo>/Clean/`, roots as arguments, CLI
  with default R1 lowpass, resume and polish-only kept).
- A comment standard: Dirk decided comments get their own pass after the reviews; step 4 keeps only the required
  markers (`@added`, `__project_origin__`, `# CHANGED:`) and the `# THEORY:` / `# HEURISTIC:` labels.
- Mainline code, the critiques, the inventories, thesis text, git writes.

## 3. Where things stand
Branch `Augmentation`, dirty tree; leave it. Mainline carries D-274 (smoke-tested rows 229 and 224). `design.md`
was written in step 2; two independent critiques (`critique-claude.md`, `critique-codex.md`) were merged by the main
session into `critique-merged.md`, which is the authority for this revision. No run in flight.

## 4. Established and verified
- Both critics agree that no trim checked on the CPU float64 path changes a training number, and that the RNG order
  of design section 6.1 holds.
- Re-checked by the main session: `scripts/gantry/thesis-results/inspect_checkpoint.py:30-31` puts the mainline root
  on `sys.path` at import, and `Thesis-writeup/Code/Evaluation/eval_greybox.py:105` and `eval_overlap.py:141`
  import it (merged item 14).
- The rulings: `docs/decisions.md` D-275 (end of file); D-274 above it.
- D6 is defined by the D-275 amendment (Dirk 2026-10-20), and this definition replaces any other reading of D6: a
  run folder is one model; `--resume <run_dir>` and `--polish-only <run_dir>` take the run folder, find the
  checkpoint in it (`_best`; `_last` only when named, as mainline `training.py:97-98`; `pre_lbfgs` for polish-only)
  and write into the same folder under the same run id; the previous outcome (final pair, `.npz`, `val_hist.npz`)
  first moves to `<run_dir>/previous/<rerun_id>/`; `config.json` gets a `RESUMES` list (time, type, source
  checkpoint, code); the evaluation is unchanged. The CLI of merged item B22 takes run folders for both flags.
- D-275 amendment 2 (Dirk 2026-10-20): the clean code also saves `_last.pth` at every validation, so a run stopped
  by the time limit continues from where it stopped (`--resume <run_dir> --from last`; the default stays `_best`).
  An addition beyond mainline: the design states where the save sits, shows from the code that it changes no
  training state (no basis sync, no RNG, no optimizer change), and the equivalence plan adds a layer-3 check with and
  without the save. Mainline only writes `_last.pth` at the end of `fit` (`interconnect.py:1122`).

## 5. Assumed but not verified
- The optimizer pickling detail behind D1 (`interconnect.py:1287, 1521-1524, 1605-1627`, deepSI `fit_system.py:496`):
  D1 keeps the property either way, so only the resume-test wording depends on it; read it before writing layer 3c.
- Line numbers in the critiques are against the 2026-10-20 working tree; re-read each before citing it.

## 6. Tried and failed
None for this task.

## 7. Achieved
`design.md` (step 2), the two critiques, the merged list and D-275.

## 8. The open question
Nothing blocked. If a fix turns out impossible as stated (the code says otherwise), write the reason in the changelog
line and pick the nearest exact alternative; Dirk reviews the changelog.

## 9. Next action
Read `critique-merged.md`, then both critiques for the detail behind each item, then edit `design.md` item by item.

## 10. Acceptance criterion
- Every item of `critique-merged.md` (D1 to D6, B1 to B23) has a changelog line and a concrete edit in `design.md`.
- The mapping has no row whose kept code still uses a dropped name: a script check over design section 2 confirms
  that every name the kept mainline statements read is kept, renamed with its call sites listed, or listed as removed.
- Open choices 1, 4, 12 and 16 are rewritten or deleted per the 2026-10-20 decisions; layer 3c and the compiled GPU
  smoke are in the equivalence plan.

## 11. Read these first
1. `tasks/clean-code/critique-merged.md`.
2. `tasks/clean-code/critique-claude.md` and `critique-codex.md`: the evidence per item.
3. `tasks/clean-code/design.md`.
4. `docs/decisions.md` D-274 and D-275.

## 12. Do not
- Do not create any file other than edits to `design.md` (scratch scripts go in the session scratchpad).
- Do not run training; do not read `kamtin-data/Data Telica/`.

## 13. Operational
Quick checks follow the CLAUDE.md environment rule. Final message: the changelog, and any item where the code forced
a different fix than the merged list proposes.

## 14. Delegation
One general-purpose subagent for the dangling-name check of section 10 (mechanical, wide), started early. None for
the edits; they need one context.
