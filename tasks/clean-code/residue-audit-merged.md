# Residue audit, merged (2026-10-31)

Inputs: `residue-audit-claude.md` (1,150 rows, proposals P1 to P8) and `residue-audit-codex.md` (3,053 rows, one row
per occurrence including every numeric literal; proposals P01 to P31), written independently by
`tasks/handoffs/2026-10-31-residue-audit.md`. Merged at proposal level; IDs `R<n>`. Both audits are static: "bitwise"
means the numerical operations are unchanged, reasoned from the readers, not run. Nothing is implemented.

## 1. The finding both audits make
The residue has one cause: `Clean/` was built to read the thesis run folders that mainline wrote and to produce
files the harness compares key by key with mainline. That pulled mainline's `config.json` schema, its checkpoint
state and its names into production code. Both propose the same structural fix: one converter in
`clean-up-verification/` that turns a thesis folder into a Clean folder; `Clean/` then owns its formats.

## 2. Agreed proposals (both audits), ranked by how much they clean

| R | Change | Claude | Codex | Cost |
|-|-|-|-|-|
| R1 | Converter `clean-up-verification/` for thesis folders; `from_config_json` keeps only the Clean branch and checks only real settings; the mainline branch, `_CLEAN_KEYS`, the skip list and the PS2_SMOKE note leave `Clean/`; `config_json_dict` stops writing the fixed literals of options `Clean/` does not have (ORTH*, LBFGS_CHUNK, LBFGS_FREEZE, CHECKPOINT_CHUNK, SNR, ...) and the mainline names (MODE, ENCODER_INIT, NX_ANN, OBC, COMBO_INIT_DETUNE, ANN_ROUTE_IX); `main` stops writing `hp` into config.json | P1 | P01, P02, P07 | about 45 to 49 keys and 60 lines; thesis folders load only through the converter (r1_check, L4 clean side); existing Clean folders still load; compare_l3 compares common keys only |
| R2 | Write-only ClassVars go: param_prior, use_f64, joint_estimation (and the ignored `joint_estimation` argument of `load_checkpoint`, K7), lbfgs, controller_gain, obc_rank_rtol, obc_ref_successor, encoder_init with the `mode` property (run family folder named after `noise`); obc_space and obc_refresh as literals in the one print; `route_ix` of the OBC correction and PS2 (K32) | P2 | P03, P09, P11, P07 | 9 to 12 settings, 3 parameters; harness callers drop keywords; new runs land in a renamed folder |
| R3 | Unread checkpoint state goes: the 13 nominal matrix buffers of `GantryStateBlock` (the live matrices come from `_mats_at`; `P_mat` stays), the parameter-prior chain (`param_rmse_baseline`, `RMSE_baseline`, `Lambda_combo`), then `gantry_ss` M1, M2, C, K | P3 | P05, P04 | 14 state_dict keys; every existing checkpoint, thesis and Clean, needs the converter to drop them; compare_l3 needs a dropped set; decide before the step-5 server run |
| R4 | Pair `.npz` keys nobody reads go: bestfit_selected, elapsed, hp, orig_run_id, start_phase, resumed_from | P4 | P06 | l4_stage and compare_l3 adapted |
| R5 | Defaults that cannot work or name the wrong thing: `mats=None`, `pass_mats=None` after K14, `params_init=None`, `combo_init=None`, the ignored `validation_measure='sim-NRMS'` parameter of `cal_validation_error` | P5 | P12, P08 (part) | direct harness callers pass the argument |
| R6 | `build_closed_loop` loses `val_data` ("kept for callers" is no reason; K6 narrowed) | E rows | P10 | main and harness keyword |
| R7 | History in comments and messages rewritten as what the code does: the tuple range of `build_reference_set` (ends at N - 4 to keep the thesis tuple set; no successors), "run this folder with the mainline code", "the mainline implementation", the false clause about whole dumps in `load_checkpoint`, the SNR remnant comment in data.py, the mainline-order text of config.py | P6, P4 (print) | P27, P28, P30 | comments only |
| R8 | Unlabelled numbers get `# THEORY:` or `# HEURISTIC:`: RunConfig budgets and sizes, gantry_ss constants, tolerances (1e-6, 1e-8, 1e-9, 1e-12), PS2Spec fields and seed offsets, `max_consecutive_rank_loss`; `obc_ref_chunk` and the duplicate default `chunk=16384` become one labelled constant | P7 | P21 to P24 | comments only (Codex also labels plotting and terminal layout numbers, P25) |

## 3. Disagreements for Dirk (one line per side)

| R | Item | Claude | Codex |
|-|-|-|-|
| R9 | `CODE` key of config.json | remove: no reader once the mainline branch leaves | keep: the native schema discriminator |
| R10 | `optimizer_state_dropped` in the pair | remove: the pair holds no optimizer exactly when it is True | keep: `load_checkpoint` reads it for its message |
| R11 | `resumed_at_epoch`, `ps2_summary` in the pair | keep: the only file record of where a continuation started and which PS2 path ran | remove: written, never read by code |
| R12 | `val_hist.npz` | keep the file; drop only `val_hist_sim_rms` and `last_sim_rms` | move the disk write out of `Clean/` (only verification reads it); keep the in-memory rows that select the checkpoint |
| R13 | `RunConfig.validation_measure` | keep: the one display name of the score | remove the setting and its key; a fixed label in the prints |
| R14 | `**kw` of `ClosedLoopSimulator.__call__` | comment: it absorbs nf and stride of the loss keywords | remove the sink, pass ctrl_ix, obc_pass, pass_mats explicitly |
| R15 | the `optimizer` property and its `_optimizer` field, `d.get('optimizer') or d.get('_optimizer')` | remove (mainline's backing field) | not proposed; Clean's own resume dumps store the optimizer as `_optimizer`, so removal needs the resume path checked |

## 4. Found by one audit only

Claude only:
- R16 `lbfgs_outcome['max_iter']` and the fourth return value of `estimation_parameters`: write-only, remove.
- R17 duplicate sources: hp key `NX_ANN` beside `n_a`; `fs` from `dt` beside `fs_new`; `raw_init[10..12]` by index; the
  dtype print slice; the S2 plateau window 5 as a `PS2Spec` field `s2_window` beside `s1_window`.
- R18 `P_inv_T` of `compute_normalization` (data.py, the MM16 line) to reuse `obc_gantry.P_INV_T64`. Note: equal in the
  final form, but the step-5 server procedure reverts that exact line with `sed`; changing it breaks arm A. Do after
  step 5 or not at all.

Codex only:
- R19 `allow_pickle=False` for the pair `.npz` reader (the pair holds only primitive arrays and JSON strings).
- R20 small habits: tuples for `PARAM_NAMES`, `COMBO_NAMES`; three redundant `int(...)`; 14 unused unpacked raw
  parameters named `_`; `p.data.copy_` inside `no_grad` as `p.copy_`.
- R21 Jan's code: the discarded `x` argument of `WindowedSystemData` and `transform`; shared mutable defaults
  (`optimizer_kwargs={}`, `loss_kwargs={}`, a default tensor); the deepSI keyword seam documented and
  `online_construct=True` refused with a message.
- R22 `__project_origin__ = "added"` in 13 files: unread at runtime; removal reverses the D-282 marker rule.

## 5. Policy questions behind the rulings
- **Jan's code**: both list residue inside it, marked; Claude 10 non-keep rows, Codex R21. Keep his code 1:1 (no R21),
  or allow residue fixes with `# CHANGED:` markers?
- **Markers** (R22): keep the `__project_origin__` rule (D-282) or rely on the license header and the attribution
  ledger?
- **Order with step 5**: R1 and R3 change config.json and checkpoint keys. If they go in before the server run, the
  layer-4 harness and `r1_check.py` load the thesis folders through the converter; if after, the server run proves
  the current tree and R1/R3 need their own proof later.

## 6. Proof, without long local runs
Every proposal is labelled bitwise in both audits. The checks that fit Dirk's rule: compile, ruff F, the 72 equation
tests, `check_imports.py`, one short smoke run with `evaluate()` and `val_check.py`, and the converter tested on one
thesis folder by loading its final checkpoint. The layer 1 to 4 comparisons run once, in the step-5 server job.
