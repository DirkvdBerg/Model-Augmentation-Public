# Keep audit of `Clean/blackbox/` (D-285 amendment 6, phase 1)

Date 2026-11-01. Rules: D-284 amendments 1, 9, 10 (an option stays only if it selects a working alternative; no
write-only values, pass-throughs, verification code or picked tolerances in `Clean/`; settings at the top of the entry
script, CLI may override; thesis and grey-box names). Style: `ps2_opening.py`. Nothing is edited yet.

## 0. Reference golden (closes todo D-285 4b)
`simulations/clean-golden/blackbox/clean2`: the unedited `Clean/blackbox/train_bb.py`, flags `--truncate-s 3 --n-its
200 --its-per-val 100`, 240 s, val NRMS_e 0.6874. `compare_bb.py clean1 clean2`: weights, optimizer state, losses,
`result.json`, `val_hist.json` identical; only deepSI's `unique_code`, `_hist_path` (output folder) and `batch_feq`
differ, as in `clean1` vs `run1`. `clean1` predates the move of the nets to `jan_model.py`, so its checkpoints pickle
`bb.JanEncoder`; the comparator ran unchanged through a scratch wrapper that aliases `bb.JanEncoder` and
`bb.JanStateNet` to `jan_model`'s. `clean2` is the reference for phase 2.

## 1. Fixed by existing files (not renamed)
- `jan_model`, `JanEncoder`, `JanStateNet`: pickled by the thesis checkpoints (D-285 amendment 2).
- `ServoWeightedEncoder`: NOT pickled as a class (the R8 `_best.pth` holds only `jan_model.*` and deepSI classes; deepSI
  saves `__dict__`), but it is the stem of every checkpoint file name and of `result.json` `checkpoint`. Keep: a rename
  gains nothing and renames every new checkpoint.
- `result.json` keys `args.nx`, `args.fh_width`, `args.e_width`, `args.lr`, `k0`, `checkpoint`, `best_update`,
  `status`, `nrms_e_per_record`: read by `select_bb.py` and `eval_bb.py` from the 11 thesis grid folders. So the
  settings `nx`, `fh_width`, `e_width` keep these names (grey box `n_nodes_per_layer` is a different net anyway).
- `preds_BB*.npz` keys: read by the thesis figures (`make_results.py`: records, k0, y, r, sigma_e, bb, y_nf;
  `eval_common.py`: code).

## 2. Rows
Verdicts: keep (reason), **remove**, **rename old -> new** (match), **move**, **rewrite** (comment). Bold = change.

### bb.py
| ID | Item | Verdict |
|-|-|-|
| B1 | module docstring (5 lines, "Conventions:" legend) | **rewrite** to 2 lines, no legend (B11 renames make it unneeded) |
| B2 | `__project_origin__` | keep (D-282 marker, R22) |
| B3 | `sys.path.insert` of `Clean/` | keep: the scripts run as `python Clean/blackbox/<x>.py`, so `gantry_dynamic` is not on the path otherwise; comment `# Clean/` -> step comment |
| B4 | re-export of `RunConfig`, `DATA_ROOT` to `train_bb`, `eval_bb` | **move**: those import `gantry_dynamic.config` directly (pass-through) |
| B5 | import of private `_record_at_fs_new` | keep (D-285 point 1, same loader as the grey box); wish for the other session in section 6 |
| B6 | `NA = 30`, "(Table hyper)" | **remove**, use `RunConfig.na_nb` (same value and meaning, "past samples of y and u the encoder sees"; the thesis black box uses the grey box's window, as `nf` already comes from `RunConfig.nf`). Fallback if Dirk prefers a separate value: **rename NA -> NA_NB** |
| B7 | `model_kwargs` | keep (2 callers, train and select; am. 9 point 1); docstring to style |
| B8 | literal `n_hidden_layers=2` (twice) | **rename** to constant `N_HIDDEN_LAYERS = 2` (grey box `n_hidden_layers`), labelled HEURISTIC thesis value |
| B9 | `fkw` | **rename fkw -> fh_kwargs** |
| B10 | imports inside `model_kwargs` (deepSI, jan_model) | **move** to module top: no reason found (train sets LOCALAPPDATA before it imports `bb`); the phase-2 golden shows it |
| B11 | `load_records`: `d`, `f` | **rename d -> mat** (`d` is also `RunConfig.d`, the decimation), **f -> u_ms** (removes the "code `f` is u^ms" legend) |
| B12 | `load_records` `folder` parameter | keep: eval passes the twin and noise-free folders |
| B13 | `{**d, 'u_total': d['f_sim']}` | keep; comment says why: the grey-box loader reads the plant input from `u_total`, here it gets the injected force |
| B14 | `data_cfg` and its fixed `noise='lowpass'` | keep (3 callers; the thesis black box trains on the lowpass set, the noise-free set is read by folder) |

### jan_model.py
| ID | Item | Verdict |
|-|-|-|
| J1 | module docstring (3 lines, legend "Batch rows are windows; u = ...") | **rewrite** to 2 lines, no legend |
| J2 | ".reshape replaces deepSI's .view" | **move** from the docstring to one inline comment at the two `reshape` lines (a marked change to Jan's net) |
| J3 | `forward` docstrings, Args/Returns | keep, trimmed to style (formula line, one context line, Args/Returns as in `ps2_opening._sample`) |

### train_bb.py
| ID | Item | Verdict |
|-|-|-|
| T1 | module docstring (8 lines, "Conventions:", Local/Server usage) | **rewrite** to 2 lines; usage is in the settings block and `Clean/README.md` |
| T2 | argparse-only settings, `--out` required | **move** to a settings block at the top (section 3); CLI flags keep their names and override |
| T3 | `--nx`, `--fh-width`, `--e-width`, `--lr` | keep: the grid axes, each a working alternative (`run_r8.sh` uses them) |
| T4 | `--n-its` | keep, new reason: the training budget, a working option (grey box `n_its`), not "for the golden runs" |
| T5 | `--its-per-val` | keep (grey box `its_per_val`) |
| T6 | `--truncate-s` | keep (grey box `truncate_s`, "for trial runs") |
| T7 | `--out` | **rename -> OUT_DIR** setting; default `<OUT_ROOT>/blackbox/<SLURM job id or time>`, the grey box's run-id rule |
| T8 | LOCALAPPDATA redirect + `makedirs` | keep: on Windows deepSI writes checkpoints below LOCALAPPDATA; the redirect keeps them in the run folder, and deepSI's own `mkdir` is not recursive. On Linux deepSI ignores it (checkpoints go to `~/.deepSI`, which `eval_bb` reads) and an empty folder remains; noted, not changed |
| T9 | imports after the env line (`noqa: E402`) | keep (order needed by T8) |
| T10 | `SEED, BATCH, STRIDE = 1, 256, 25` | keep values; **rename BATCH -> BATCH_SIZE** (grey box `batch_size`); into the settings block as fixed thesis values (HEURISTIC) |
| T11 | `torch.set_num_threads(4)` | keep; value as setting `N_THREADS = 4` (the server's cores per task), shared with E15 |
| T12 | `per_record_rms` | keep (2 callers); **rename sds -> sys_data**; the one-line comprehension split into two lines (same operations) |
| T13 | `np.errstate(all='ignore')` | keep: a diverging configuration gives inf/NaN; it is scored and `select_bb` skips it (3 thesis configurations failed) |
| T14 | class docstring, `__dict__` reload note | **move** the reason to one step comment in `set_objective` ("lists owned by main: deepSI's reload of `_best` replaces this model's attributes, and `status` needs the failure after it") |
| T15 | `set_objective` `run_lists` tuple | **rename** to two parameters `val_hist, failure` |
| T16 | attributes `_sigma_train`, `_val_sigma`, `_dump` | **rename -> _train_sigma_e, _val_sigma_e** (grey box `val_sigma_e`), **_dump -> _write_val_hist** (pickled in `__dict__`: the comparator then lists them as keys only in A/B; their effect is in the weights, which it still compares) |
| T17 | weight-row `RuntimeError` in `make_training_data` | keep (judgment): exact count, no tolerance; catches a deepSI windowing change that would silently pair weights with the wrong windows; same class as grey-box guards K19, K22, K39. Alternative: remove (deepSI pinned at 0.3.29) |
| T18 | `loss`: `yhat` | **rename yhat -> e_hat** (it is e_hat^n); `**kw` gets a sink comment (grey box R14) |
| T19 | failure record in `loss` | keep: deepSI stops on NaN and reloads `_best` without a record |
| T20 | `cal_validation_error(..., validation_measure=None)` | keep the parameter (deepSI's `fit` always passes it), **remove the default** (grey box R5: a default that names nothing) |
| T21 | `fit(..., validation_measure='sim-NRMSe')` | keep: deepSI's display name in its prints, same string as grey-box `SCORE_NAME` |
| T22 | `main`: `tr_f, va_f`, `sds`, `sigma_val`, `pa` | **rename -> train_files, val_files** (grey box), **sys_data**, **val_sigma_e** (grey box), **val_rms** |
| T23 | prints `[B]` | **rename -> [BB]** (as `eval_bb`) |
| T24 | result.json `args`, `k0`, `checkpoint`, `best_update`, `status`, `nrms_e_per_record` | keep (readers, section 1); `args` is built from the settings with the same keys, so it stays bitwise |
| T25 | result.json `val_rms_hist` | **remove**: the same list as `val_hist.json` `history` |
| T26 | result.json `failure` | **remove**: duplicate of `val_hist.json` `failure`; `status` says failed |
| T27 | result.json `per_record` | **remove** if E8 goes: its only reader is the E8 check; it equals `nrms_e_per_record x val_sigma_e` |
| T28 | result.json `val_sigma_e`, `wall_s` | keep (record scale for reading NRMS_e in metres; run time) |
| T29 | val_hist.json history entries `rms` and top key `val_sigma_e` | **remove** (grey box R12 removed `val_hist_sim_rms`; no reader; `val_sigma_e` only converted `rms`); history keeps `update`, `nrms_e`; `failure` stays |
| T30 | numbered phase comments 1 to 4 in `main`, 1 to 3 in `loss` | **rewrite** to style (what and why, word first) |

### eval_bb.py
| ID | Item | Verdict |
|-|-|-|
| E1 | module docstring (5 lines, usage) | **rewrite** to 2 lines |
| E2 | `sys.argv` only | **move** to settings `RUN_DIR = None` (required), `OUT_DIR = None` -> `<RUN_DIR>/evaluation` (the grey box's `<run>/evaluation/`); argv overrides |
| E3 | `from bb import RunConfig, DATA_ROOT` | **move** (B4) |
| E4 | `TWINS` | keep (multisine twins, evaluation only); **rename -> TWINS_DIR** |
| E5 | `predict` | keep (readable name, am. 9); `pr` **rename -> preds** |
| E6 | checkpoint fallback to `~/.deepSI` | keep: Linux runs (the thesis grid) write there; `ck` **rename -> ckpt** |
| E7 | `__new__` + `__dict__ = torch.load` | keep: deepSI's own reload form; prediction needs no subclass |
| E8 | val check print + `assert rel < 1e-4` | **remove**: verification in `Clean/` with a picked tolerance (am. 9 point 6, E1 precedent for the grey box); the bitwise preds comparison outside `Clean/` covers it |
| E9 | splits dict with noise-free folders, `y_nf` | keep (`make_results` reads `y_nf`, the noise floor) |
| E10 | npz keys records, k0, y, r, sigma_e, bb, y_nf, code | keep (readers, section 1) |
| E11 | npz key `split` | keep (judgment): no reader, but the file then has the grey-box preds format; alternative: remove (one key less vs `eval_clean`) |
| E12 | `preds_BB.npz` for test | keep (the name the figures read) |
| E13 | `o`, `e` (an RMS, not the servo error), `n, a, sg` | **rename -> saved, rms, name, rms_r, sigma_r** |
| E14 | `torch.set_num_threads(4)` | keep as `N_THREADS` (T11) |
| E15 | `json.load(open(...))` | **rewrite** with `with` (file closed) |

### select_bb.py
| ID | Item | Verdict |
|-|-|-|
| S1 | module docstring (7 lines, usage) | **rewrite** to 2 lines |
| S2 | `sys.argv` only | **move** to settings `GRID_DIR = None` (required), `OUT_CSV = None` -> `<GRID_DIR>/selection.csv`; argv overrides |
| S3 | `NU, NY = 6, 3` | **rewrite** as `RunConfig.ny + RunConfig.nu` (r and u^ms) and `RunConfig.ny` (one source) |
| S4 | `n_params`, `nu/ny` set before `init_nets` | keep (deepSI needs both before it builds the nets) |
| S5 | glob `*_n*_*` and name parsing | keep (the `run_r8.sh` folder form) |
| S6 | `d`, `a` | **rename -> run_dir, args** |
| S7 | `isfinite` filter | keep (failed configurations) |
| S8 | CSV columns and `lr` format | keep (shared columns of thesis `res_bb_selection.csv`, the selection proof) |
| S9 | `json.load(open(...))` | **rewrite** with `with` |

### run_r8.sh
| ID | Item | Verdict |
|-|-|-|
| L1 | `-J`, `-N`, `--ntasks`, `-c 4`, `--mem`, `-t`, `--array=0-10` | keep |
| L2 | `-p molokai,lanai` | **remove** (our cluster's partitions; the site default applies) |
| L3 | `-o /home/dirk_van_den_berg/logs/...` | **rewrite** `-o bb_r8_%A_%a.out` (submit folder; SLURM does not create log folders) |
| L4 | `source $HOME/miniconda3/...`, `conda activate /dataB1/...`, `cd /dataB1/...` | **rewrite** as settings at the top: `CONDA_ENV=gantry-augmentation` (`Clean/environment.yml`), `conda activate "$CONDA_ENV"` via `$(conda info --base)`, `cd "$SLURM_SUBMIT_DIR"` (submit from the repo root) |
| L5 | reads `Thesis-writeup/Code/Server/blackbox/thesis_r8_grid.conf` | **move** the 11 lines into the script (section 4) |
| L6 | `n80000` literal in the folder name | **rewrite** as `N_ITS=80000`, passed as `--n-its` and used in the name, so the name cannot lie |
| L7 | git and job prints, thread exports, `set -eo pipefail`, `${SLURM_ARRAY_TASK_ID:?}` | keep (run log; prints `unavailable` without git) |
| L8 | header comments (runtime, usage lines) | **rewrite** to style; keep the two `sbatch` usage lines and the runtime as the reason for CPU |

## 3. Proposed settings blocks
`train_bb.py` (CLI flags keep their names and override; the settings must precede the deepSI import, T8):
```python
# one configuration of the grid; the defaults are the selected R8 (grid index 9)
NX = 16                 # state dimension n_x
FH_WIDTH = 16           # hidden width of f_bb and h_bb
E_WIDTH = 32            # hidden width of the encoder psi
LR = 1e-3               # Adam learning rate
# run budget and data
N_ITS = 80000           # Adam updates
ITS_PER_VAL = 1000      # updates between validations; the best validated checkpoint is kept
TRUNCATE_S = None       # [s] use only the start of each record, for trial runs
OUT_DIR = None          # run folder; None: <OUT_ROOT>/blackbox/<SLURM job id or time>
# fixed (HEURISTIC: thesis values)
SEED = 1                # initial weights and batch order
BATCH_SIZE = 256        # windows per update
STRIDE = 25             # samples between window starts
N_THREADS = 4           # CPU threads, the server's cores per task
```
`eval_bb.py`: `RUN_DIR = None` (run folder with `result.json`; required), `OUT_DIR = None` (None:
`<RUN_DIR>/evaluation`). `select_bb.py`: `GRID_DIR = None` (folder of one `run_r8.sh` job; required), `OUT_CSV = None`
(None: `<GRID_DIR>/selection.csv`). `bb.py`: `N_HIDDEN_LAYERS = 2` (B8).

## 4. Grid file outside `Clean/` (L5)
Replace the `grep` of `thesis_r8_grid.conf` by the 11 lines as a bash array, index = array position:
```bash
# R8 grid: name and arguments per array index (indices 0 to 8: nx x width at lr 3e-3; 9, 10: lr at nx 16, w 16)
GRID=(
  "nx08_w08        --nx 8  --fh-width 8  --e-width 16 --lr 3e-3"
  ...
  "nx16_w16_lr1e-2 --nx 16 --fh-width 16 --e-width 32 --lr 1e-2"
)
LINE=${GRID[$IDX]:-}
```
The conf file's header (2 % band, plateau rule, budget) is superseded or verification text and stays where it is. No
new file. `run_r8.sh` then reads nothing outside `Clean/` except the data root and the run folder.

## 5. Proof plan (phase 2)
Edit, then `train_bb.py` golden into `clean3`, `compare_bb.py clean2 clean3`: bitwise except the volatile fields and
the keys removed or renamed by T16, T25 to T27, T29 (the comparator lists those as "only in A/B"). `select_bb.py` on
`grid_89756` against `res_bb_selection.csv` (seconds). The evaluation golden (about 24 min locally) against
`eval_clean`: Dirk decides local or server.

## 6. Out of scope, for the session that owns `Clean/gantry_dynamic/`
- `_record_at_fs_new` is imported by the black box; a public name (`record_at_fs_new`) would end the private import.
