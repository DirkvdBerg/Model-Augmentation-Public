# Handoff: residue audit of `Clean/` (what the keep audit could not see)
**From**: session of 2026-10-31 | **Branch**: Augmentation | **Effort suggested**: high (reading and judging about
5,000 lines; no derivations, no runs)

Two independent audits use this file: a Claude session (sections 1 to 14) and a Codex session (section 15, to paste).
Neither reads the other's output.

## 1. Task
Dirk, after reading `Clean/gantry_dynamic/config.py`: "i see so many trash variables", "if the code needs these trash
variables that points to a bigger issue", "what else bad code habits have not been caugth". Audit the grey-box tree of
`Clean/` for residue: everything that exists because of mainline's history, mainline compatibility or verification,
rather than because a user of `Clean/` (ASMPT) needs it. The keep audit (D-284 amendment 8) gave a row only to
callables; give a row to every item of the kinds in section 4 and judge it by the yardstick there. Write
`tasks/clean-code/residue-audit-claude.md`: a summary at the top (counts per category and verdict, the proposals ranked
by how much they clean, the patterns), then one row per item. This is reading and judging only: the table is the
deliverable, Dirk rules on it afterwards.

## 2. Out of scope
- Any edit under `Clean/`, the harness or `clean-up-verification/`; any training, harness or suite run (Dirk
  2026-10-31: no long local runs). Static tools (grep, ruff, vulture, radon) are fine.
- `Clean/blackbox/` (D-285) and `Clean/data_generation/` (D-286): other sessions.
- Re-judging callables the merged keep audit already ruled (`tasks/clean-code/keep-audit-merged.md`, amendment 9),
  unless this yardstick gives new evidence; then say so in the row.
- The step-7 proof of D-284 and the step-5 server job.
- Reading the other audit's output file before yours is written.

## 3. Where things stand
Tree dirty (Dirk commits). `Clean/` = snapshot `final` (D-284) plus the rename "logical" to "generalised", the K items
K3, K8, K13, K14, K15, K23, K24, K26, K31, K41, K42, and E1 to E4 of amendment 9 (new `gantry_dynamic/evaluation.py`;
`closed_loop_report.py` removed), plus the fixes of `tasks/clean-code/keep-audit-merged.md` section 6. 24 grey-box files:
`gantry_interconnect_dynamic.py`, `gantry_dynamic/`, `model_augmentation/`. Nothing running.

## 4. The yardstick
**Items that get a row** (each with `file:line`):
- module constants; dataclass fields and ClassVars of `RunConfig`;
- every key written by `config_json_dict` and every key in `_ENV_KEYS`, `_CLEAN_KEYS` and the `from_config_json` check;
- registered buffers and parameters (checkpoint keys), and the meta keys of the checkpoint pair `.npz`;
- the `hp` dict keys; the accepted checkpoint and dump formats (`resolve_checkpoint`);
- function parameters and their defaults; branches, options and flags (a value that selects code);
- names and comments that describe mainline, history or a removed feature rather than this code;
- numbers in code without a `# THEORY:` or `# HEURISTIC:` label.

**Category** (one or more per row):
- `compute`: read by the computation of a `Clean/` run (name the reader `file:line`);
- `write-only`: written (config.json, checkpoint, print) and never read as a choice;
- `fixed-option`: an option that every thesis run and every `Clean/` path uses at one value;
- `mainline-compat`: exists so `Clean/` reads or writes mainline's run folders, names or formats;
- `verification`: exists for the harness or a check of the cleanup;
- `misleading`: the name or comment says something untrue of this code;
- `unlabelled`: a number without a label.

**Verdict**: `keep` (reason), `remove`, `literal` (inline the one value), `move` (to `clean-up-verification/`, for
example a converter of thesis run folders), `rename`, `label`, `comment`. Give the concrete change in one line, the
`bitwise` or `rounding` label and what else it touches (harness readers, `config.json` content, loading of existing
run folders).

**Keep reasons that count**: the item is read by a `Clean/` computation; a user of `Clean/` would set it (a real
option, D-284 amendment 9 point 2: "functionality that can be actual useful"); it guards invalid input with an
actionable message. **Reasons that do not count by themselves**: "interface" (`config.json`, checkpoint keys, CLI,
rerun rules), "mainline has it", "the harness reads it". Those interfaces are themselves under audit here: for each,
state what changing it would cost (does a thesis checkpoint still load, does a rerun of an existing `Clean/` folder
still work, which harness reader changes).

**Jan's code** (`Model-Augmentation-Public-main/` is his reference): his classes, names and files stayed 1:1 by an
earlier design choice. My reading, to confirm with Dirk: residue inside Jan's code is still listed, marked
`Jan's code`, with the same verdict; Dirk decides.

## 5. Established and verified (2026-10-31, read in the code)
Samples of the kind, not the scope:
- `config_json_dict` writes about 80 keys; 25 are fixed literals, several of mainline options `Clean/` does not have:
  `ORTH`, `ORTH_BETA=4.66e-4`, `ORTH_POINT_STRIDE`, `ORTH_RANK_TOL` (mainline's soft orthogonality penalty, off in every
  thesis run), `LBFGS_CHUNK`, `LBFGS_FREEZE`, `LBFGS_TIME_BUDGET_S`, `LBFGS_METER_REL_TOL`, `LBFGS_METERS_GATE`,
  `CHECKPOINT_CHUNK`, `STATIC_PASS_BUFFERS`, `SNR`, `PARAM_INIT_DETUNE`, `NF_PROBE_PRINT`. Their mainline meaning is
  in `scripts/gantry/gantry_dynamic/config.py` (comments at each field).
- `RunConfig.from_config_json` accepts a folder only if every non-skipped key equals `config_json_dict`'s value; it has
  a separate branch for mainline folders (`CODE` not `'clean'`). The thesis run folders were written by mainline.
- `obc_rank_rtol` and `obc_ref_successor` are ClassVars written only to `config.json`; the rank tolerance of
  `Clean/` is always tol = S_max max(M, N) eps (`model_augmentation/fit_systems/obc.py`, `OBCBasis`).
- `obc_ref_successor = True` is mainline's D-252 setting (each tuple has a successor); `Clean/` builds no successors
  since K15 but keeps `k = np.arange(2, len(u) - 3)` in `build_reference_set` to keep the thesis tuple set, with no
  comment saying so.
- `obc_ref_chunk = 16384` has no label; the same number is the default `chunk` of `build_stacked_sensitivity` in
  `obc.py`. It bounds memory only.
- `param_rmse_baseline` only feeds the `Lambda_combo` buffer, and its own comment says no loss reads it.
- K29 (eight write-only ClassVars), K7, K11 and K32 of the merged audit were neither implemented nor kept by an
  explicit ruling in amendment 9.

## 6. Assumed but not verified
- That `Clean/`'s own resume path needs the deepSI whole-dump format of `resolve_checkpoint` (it reads `_best.pth`
  and `_last.pth`); settle by reading `gantry_interconnect_dynamic.py` `_rerun_source` and `training.py`.
- That the state_dict keys of the thesis checkpoints equal those of `Clean/`, so a converter of thesis folders needs
  only `config.json`; settle by reading `training._assert_same_architecture` and the layer-4 harness loader.
- Which `config.json` keys anything in the harness or `clean-up-verification/` reads.

## 7. Tried and failed
- Keep audit (D-284 amendment 8): two independent audits under one criteria file (`06c-audit-criteria.md`) -> both
  missed every item of section 5 -> the fixed rules made `config.json`, checkpoint keys and the CLI an automatic keep
  reason "(c) interface", the gates made "equal to mainline" look like a requirement, and rows existed only for
  callables -> `keep-audit-merged.md` has no row for any constant, key or buffer.

## 8. The open question
Nothing blocked. The one interpretation to confirm is in section 4 (Jan's code); until confirmed, list and mark.

## 9. Next action
Enumerate the items of section 4 in all 24 files (your own pass: AST for constants, fields, parameters and defaults;
`config_json_dict` and `register_buffer` by reading), then find each item's readers by search, then write the rows and
the summary.

## 10. Acceptance criterion
Every item of the kinds in section 4 in the 24 files has a row with category, verdict, reader or `none`, and the cost
of its change; every `keep` has a reason that counts; the summary ranks the proposals and names the patterns (for
example "mainline compatibility in production code") with the items behind each.

## 11. Read these first
1. `Clean/gantry_dynamic/config.py`: the densest residue.
2. `tasks/clean-code/keep-audit-merged.md` sections 2 and 3: what was ruled, so you do not re-litigate it.
3. `docs/decisions.md` D-284 amendment 9 and the implementation-choices paragraph after it.
4. `scripts/gantry/gantry_dynamic/config.py`: what each mainline key meant.
5. `tasks/handoffs/2026-10-26-clean-code-opt/00-overview.md` section 6: the interface rules now under audit.

## 12. Do not
Edit or run anything (section 2). Treat "interface" or "mainline has it" as a keep reason. Leave out an item because
it is small: a two-line constant gets a row. Read `kamtin-data/Data Telica/` or `kamtin-data/Telica.mat`.

## 13. Operational
Read-only. Static tools via `conda run -n GraduationProject ...` on the 24 files only. Output:
`tasks/clean-code/residue-audit-claude.md`. Row format, one line per item:
`Clean/gantry_dynamic/config.py:121 | obc_ref_successor [ClassVar] | write-only, mainline-compat, misleading | reader: none (config_json_dict only) | literal | write OBC_REF_SUCCESSOR=True as a literal, comment the tuple range in build_reference_set; bitwise; config.json unchanged`

## 14. Delegation
None, or at most one Explore subagent for reader searches outside `Clean/`.

## 15. Prompt for Codex (Dirk pastes this)
```
Residue audit of Clean/, independent of a Claude session doing the same; do not read
tasks/clean-code/residue-audit-claude.md.

Dirk: "i see so many trash variables", "if the code needs these trash variables that points to a bigger issue",
"what else bad code habits have not been caugth". The earlier keep audit (tasks/clean-code/keep-audit-merged.md)
gave rows only to callables and treated config.json, checkpoint keys and the CLI as fixed interfaces, so it missed
residue such as config_json_dict writing about 25 literals of mainline options Clean/ does not have (ORTH_BETA,
LBFGS_FREEZE, ...).

Read tasks/handoffs/2026-10-31-residue-audit.md sections 1 to 14 and do that task exactly as written there; the
only difference is your output file: tasks/clean-code/residue-audit-codex.md instead of the -claude file named
there. Use your own enumeration and static tools (ruff, vulture, radon; read-only). Report every item, small ones
too.

Rules: read-only on everything except your output file. No training, harness or suite runs. No git write
operations. Never read kamtin-data/Data Telica/ or kamtin-data/Telica.mat. No em dashes or double hyphens as dashes.
Final message: the path, counts per category and verdict, number of proposals.
```
