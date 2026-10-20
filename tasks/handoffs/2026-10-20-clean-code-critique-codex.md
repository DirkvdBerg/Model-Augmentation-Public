# Prompt for Codex: independent critique of the clean-code design document (step 3 of 4)
**From**: Claude session of 2026-10-20, for Dirk to give to Codex | **Branch**: Augmentation |
**Effort suggested**: high

## 1. Task
Critique `tasks/clean-code/design.md`, the design of the clean minimal implementation of the
`scripts/gantry/gantry_interconnect_dynamic.py` pipeline. Find every place where implementing the document as written
would (a) change a number relative to mainline, (b) break the evaluation interface, (c) miss or wrongly drop a
required function, branch or constant, (d) contradict a decision of section 4, or (e) be more code than the method
needs. Check claims against the mainline code, not against the document's own reasoning. Write the critique to
`tasks/clean-code/critique-codex.md` (the only file you create). A Claude session critiques the same document
independently; do not look for or read `tasks/clean-code/critique-claude.md`. You wrote the static step-1 inventory
(`tasks/clean-code/inventory-codex.md`); use it, but check the design against the code, since mainline changed since
then (section 3).

## 2. Out of scope
- Rewriting `design.md` or writing any code: Dirk merges the two critiques into the document.
- Re-deciding Dirk's decisions of section 4; check only that the document can carry them.
- Diagnostics, the table and figure scripts, the black box R8, thesis text, server jobs.

## 3. Rules for this repository and where things stand
- Read-only: no edits, no training or simulation runs. Quick import or syntax checks are fine.
- No git write operations of any kind. Read-only git is fine.
- Never read `kamtin-data/Data Telica/` or `kamtin-data/Telica.mat`. `kamtin-fp-model/` is read only.
- No em dashes in your output (not the Unicode character, not a double hyphen).
- Markers in `model_augmentation/`: `@added`, `__project_origin__ = "added"`, `# CHANGED: <reason>`; the clean code keeps them.
- Mainline changed on 2026-10-19 (decision D-274, end of `docs/decisions.md`), smoke-tested on rows 229 and 224:
  the OBC polish is compared with the selected checkpoint's score `bestfit` in
  `scripts/gantry/gantry_dynamic/training.py`; PS2 S2 keeps its start as a candidate in
  `model_augmentation/fit_systems/ps2_opening.py`; `record_files` in `scripts/gantry/gantry_dynamic/data.py` loads
  K1 records only (18/6/11). Mainline is the oracle. The clean folder does not exist yet.

## 4. Decisions by Dirk (2026-10-20) that override the document
1. **Folder: `<repo>/Clean/`** (replaces open choice 1, `Thesis-writeup/Code/Training/`), with the tree of design
   section 1 inside it (`gantry_interconnect_dynamic.py`, `gantry_dynamic/`, `model_augmentation/`).
2. **Self-contained**: the data root and the output root are arguments with defaults
   (`<repo>/Thesis-writeup/Data` and `<repo>/simulations/gantry_clean/`), not fixed relative paths.
3. **Launch by CLI** as in open choice 4, **with a default**: no arguments runs R1 on lowpass.
4. **Resume is kept, including the polish-only start** (replaces open choice 12): the clean code continues a run
   that stopped (time limit) and runs polish-only reruns from a `pre_lbfgs` checkpoint (allowed for 223, 225, 226,
   230). So `resolve_resume_hp`, `load_optimizer_moments`, `start_phase='lbfgs'` and the resume path of `main` come
   back; the equivalence plan needs a resume test against mainline resume.
Every other open choice of design section 7 stands as proposed unless your critique gives a reason against it.

## 5. Context
- Spec: rows 221 to 230 of `Thesis-writeup/Code/Server/tables/runs.tsv` (no 228), each on `lowpass` or
  `noisefree` data; presets R1 to R7, R13, R14.
- Step 1 comparison and Dirk's rulings: `tasks/clean-code/comparison.md`.
- Agreed harness: four layers (forward pass; loss and gradients; short seeded training; final checkpoint metrics per
  split), golden outputs from mainline, CPU, float64.
- Server environment: pytorch 2.5.1 (`environment.server-train.yml:18`).
- Claims of the document to test hardest: bit-for-bit equality after the guard trims (rule G, section 0) and the
  removed device self-heals (open choice 14); open choices 11 and 21 (truth fields and six reference-set fields not
  kept) against what the evaluation reads; open choice 9 (`ValHistory`) against the `bestfit` and selected/final
  events the evaluation reads; the RNG draw order of design section 6.1 after every trim.

## 6. Read these first
1. `tasks/clean-code/design.md`.
2. `tasks/clean-code/comparison.md`.
3. `docs/decisions.md` D-274 (end of file).
4. `Thesis-writeup/Code/Evaluation/eval_common.py` and `eval_greybox.py`.
5. Your own `tasks/clean-code/inventory-codex.md` Part C.

## 7. Done when
- Every finding cites the design line and the mainline `file:line` you read yourself, states the concrete
  consequence (which number changes, which import breaks, which definition is missing), and proposes a fix.
- The mapping (design section 2) is spot-checked for at least every file of `fit_systems/`,
  `gantry_dynamic/training.py` and `model.py`; each wrong destination or wrong "dropped" is reported.
- For each of Dirk's four decisions, the critique says what the document must change to carry it.
Report every finding, including small ones, ranked by consequence; Dirk filters.

Format example (illustrative lines): `design.md:412 | removes the device check in Static_ANN_Block.forward (blocks.py:76) | changes nothing on CPU, but on GPU the out_gate buffer stays on CPU after .cuda() because ... | keep the two lines | high`

Final message: the path of the critique and the three findings with the largest consequence.
