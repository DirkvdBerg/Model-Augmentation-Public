# Handoff: independent critique of the clean-code design document (step 3 of 4, Claude)
**From**: session of 2026-10-20 | **Branch**: Augmentation | **Effort suggested**: xhigh (checks a 1000-line design
against two packages and an equivalence plan)

## 1. Task
Critique `tasks/clean-code/design.md`, the design of the clean minimal implementation of the
`scripts/gantry/gantry_interconnect_dynamic.py` pipeline. Find every place where implementing the document as written
would (a) change a number relative to mainline, (b) break the evaluation interface, (c) miss or wrongly drop a
required function, branch or constant, (d) contradict a decision of section 4, or (e) be more code than the method
needs. Check claims against the mainline code, not against the document's own reasoning. Write the critique to
`tasks/clean-code/critique-claude.md` (the only file you create). A Codex session critiques the same document
independently; do not look for or read `tasks/clean-code/critique-codex.md`.

## 2. Out of scope
- Rewriting `design.md` or writing any code: Dirk and the main session merge the two critiques into the document.
- Re-deciding Dirk's decisions of section 4; check only that the document can carry them.
- Diagnostics, the table and figure scripts, the black box R8, thesis text, server jobs, git writes.

## 3. Where things stand
Branch `Augmentation`, dirty tree; leave it. Mainline carries the D-274 changes (polish comparison in
`scripts/gantry/gantry_dynamic/training.py`, PS2 S2 start kept as a candidate in
`model_augmentation/fit_systems/ps2_opening.py`, K1-only records in `scripts/gantry/gantry_dynamic/data.py`),
smoke-tested on rows 229 and 224. Mainline is the oracle. The clean folder does not exist yet. No run in flight.

## 4. Decisions by Dirk (2026-10-20) that override the document
1. **Folder: `<repo>/Clean/`** (replaces open choice 1, `Thesis-writeup/Code/Training/`), with the tree of design
   section 1 inside it (`gantry_interconnect_dynamic.py`, `gantry_dynamic/`, `model_augmentation/`).
2. **Self-contained**: the data root and the output root are arguments with defaults
   (`<repo>/Thesis-writeup/Data` and `<repo>/simulations/gantry_clean/`), not fixed relative paths.
3. **Launch by CLI** as in open choice 4, **with a default**: no arguments runs R1 on lowpass.
4. **Resume is kept, including the polish-only start** (replaces open choice 12): the clean code continues a run
   that stopped (time limit) and runs polish-only reruns from a `pre_lbfgs` checkpoint (the D-274 amendment allows
   these for 223, 225, 226, 230). So `resolve_resume_hp`, `load_optimizer_moments`, `start_phase='lbfgs'` and the
   resume path of `main` come back; the equivalence plan needs a resume test against mainline resume.
Every other open choice of design section 7 stands as proposed unless your critique gives a reason against it.

## 5. Established and verified (inputs the document builds on)
- Spec: rows 221 to 230 of `Thesis-writeup/Code/Server/tables/runs.tsv` (no 228), each on `lowpass` or
  `noisefree` data; presets R1 to R7, R13, R14.
- Step 1: `tasks/clean-code/inventory-claude.md` (coverage trace, data in `tasks/clean-code/coverage/`),
  `tasks/clean-code/inventory-codex.md` (static), compared and ruled in `tasks/clean-code/comparison.md`.
- D-274 and its amendment at the end of `docs/decisions.md`.
- Agreed harness: four layers (forward pass; loss and gradients; short seeded training; final checkpoint metrics per
  split), golden outputs from mainline, CPU, float64.
- The server environment pins pytorch 2.5.1 (`environment.server-train.yml:18`); deepSI loads `_best.pth` with a
  plain `torch.load`.

## 6. Assumed but not verified (by the document; test these)
- Layer 1 and 2 bit-for-bit equality after the trims of rule G (guards removed) and open choice 14 (device
  self-heals removed): any removed line that changes an operation, its order or a dtype breaks it.
- Open choices 11 and 21: no evaluation script reads truth fields or the six dropped reference-set fields from a
  clean checkpoint or reference set.
- Open choice 9 (`ValHistory`) reproduces the `bestfit` and the selected/final events that the evaluation reads.
- The RNG draw order of design section 6.1 survives every trim (live g_aug MLP, W^b draw, W^a Xavier, encoder net,
  PS2 decoder; `Dtype_DataLoader` `default_rng(0)`; draws with n_a = 0).
- The polish outcome of 221 and 222 is still unknown (server logs); it only sets their layer-4 reference.

## 7. Tried and failed
- Monkey patching the framework at run time: rejected by Dirk; the clean code is copied and trimmed code.
- Comparing full training runs bit for bit: not a criterion; the layered harness replaces it.

## 8. The open question
Nothing blocked. Where you disagree with an open choice, state the alternative and the evidence; Dirk decides.

## 9. Next action
Read `design.md` in full, then check it section by section against the mainline code and the inputs of section 5.

## 10. Acceptance criterion
- Every finding cites the design line and the mainline `file:line` you read yourself, states the concrete
  consequence (which number changes, which import breaks, which definition is missing), and proposes a fix.
- Spot-check the mapping (design section 2) for at least every file of `fit_systems/`, `gantry_dynamic/training.py`
  and `model.py`: report each definition whose destination or "dropped" is wrong.
- Say for each of Dirk's four decisions (section 4) what the document must change to carry it.
Report every finding, including small ones, ranked by consequence; Dirk filters.

Format example (illustrative lines): `design.md:412 | removes the device check in Static_ANN_Block.forward (blocks.py:76) | changes nothing on CPU, but on GPU the out_gate buffer stays on CPU after .cuda() because ... | keep the two lines | high`

## 11. Read these first
1. `tasks/clean-code/design.md`: the document under review.
2. `tasks/clean-code/comparison.md`: the agreed step-1 result and Dirk's rulings.
3. `docs/decisions.md` D-274 (end of file).
4. Part C of `tasks/clean-code/inventory-claude.md` and `inventory-codex.md`: the equivalence hazards.
5. `Thesis-writeup/Code/Evaluation/eval_common.py` and `eval_greybox.py`: the evaluation interface.

## 12. Do not
- Do not edit `design.md`, mainline, the inventories or `comparison.md`; create only the critique file.
- Do not run training; quick imports and reads only.
- Do not read `kamtin-data/Data Telica/` or the Codex critique.

## 13. Operational
No runs. Quick Python checks follow the CLAUDE.md environment rule. Final message: the path of the critique and the
three findings with the largest consequence.

## 14. Delegation
One general-purpose subagent for the mapping spot-check of section 10 (design section 2 against the mainline
definitions: is each destination or "dropped" right?), started at the beginning so it runs while you do the
numerical-equivalence and evaluation-interface checks yourself. None for the rest; those need one context.
