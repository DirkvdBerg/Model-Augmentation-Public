# Final-run code changes, 2026-10-16

Every code change for the final thesis runs, per fix. Status: **implemented 2026-10-16** (section 12 lists
where the implementation differs from this plan and the verification results).
Approved scope: `tasks/final-run-implementation-review-2026-10-16.md` (merged proposal) and
`Thesis-writeup/Documentation/FINAL-RUN-FIXES.md`. Revision 2 (2026-10-16) resolves the first independent review
(seven findings and five refinements, section 11); one decision for Dirk remains open (section 10).

Abbreviations: EP = `scripts/gantry/gantry_interconnect_dynamic.py`, GD = `scripts/gantry/gantry_dynamic/`,
FS = `model_augmentation/fit_systems/`, BB = `scripts/gantry/blackbox-cl-trainable/`,
EV = `Thesis-writeup/Code/Evaluation/`. Line numbers are of the current files, read 2026-10-16.

## 1. Rules for the implementation

- **Old behaviour stays the default** of every changed function, script argument and config field, including the
  black-box script and the OBC reference set. The final method is switched on in ONE place for the grey box, the
  `_THESIS_FIXED` block (EP:331-338), and by explicit arguments in `run_bb_search.sh` for the black box. Jan's
  benchmarks, the ~20 diagnostics, the 2026-10-11 checkpoints and the 2026-10-11 runs keep working unchanged.
- **Reuse existing seams**, no new data paths: the named extra-array seam (`extra_array_names`) for the loss
  weights, `validation_measure` for selection, `PolishSpec` for the polish, `obc_space='affine'` for Gamma,
  `live_g_feed_forward_nn` for g_aug, `_sync_prediction_state_for_validation` for the OBC coefficient.
- **One definition per quantity**: sigma_e and the record-axis weights come from one GD function used by the grey
  box, the black box and the evaluation.
- **No new files**, no monkey patching, no copied scripts or training loops. Superseded code is not deleted; the
  final path does not call it.
- **Markers** (CLAUDE.md): `# CHANGED: <reason>` on changed blocks in `model_augmentation/`, `@added` on new
  functions there; GD files are already `__project_origin__ = "added"`. Every numerical rule carries `# THEORY:`
  or `# HEURISTIC:`.
- **Proof per fix group** before the next group (section 7); old paths must stay bitwise equal.

## 2. Decision entries, written before the code

| D | Content |
|-|-|
| D-260 | FIX-5 superseded by FIX-8: PS2 S1 target stays at tau to tau+H-1 (reason in FINAL-RUN-FIXES item 5) |
| D-261 | FIX-10 setup: initial state, fitted parameters, rollout recorder, fixed budget from a pilot (HEURISTIC) |
| D-262 | sigma_e = population std (ddof 0) of y - r over the scored samples k0..N-1; measure name `sim-NRMSe`; training weight w = sigma_y / sigma_e per training record and axis over k0..N-1 |
| D-263 | FIX-2/16: meters logged but not gating in thesis runs; tol_grad and tol_change set from the pilot |
| D-264 | R8: full 11-point BB grid at 80000 updates, CPU on the fast molokai/lanai nodes; NaN breaks and nonconvergence reported |
| D-265 | Controller-rate check: diagnostic correction only, no preprocessing change |
| D-266 | Polish keeps the OBC coefficient paired with the parameters it reports: re-solve after every restore and before every validation (pre-existing defect, found in review 1; R1/R2 of 2026-10-11 accepted their polish, so unaffected) |
| D-267 | Validation history records four event types (selected, rebuilt, polish candidate, final) and is written to disk at every validation |
| D-268 | Predicted first-order bias and the overlaps computed on three tuple stacks, training, validation and test (I1 to I6) (Dirk 2026-10-16); the training stack is the prediction of eq:obc_recovery, the other two show how it carries over |
| D-269 | Every reported model metric is available for training, validation and test (Dirk 2026-10-16: supervisors want all three) |

Existing entries cover the rest: D-248 (live g_aug), D-250 (window), D-251 (data-based encoder), D-252 (tuples),
D-253 (polish score after rebuild), D-255 (polish acceptance), D-256 (NRMS_e), D-257 (Gamma), D-258 (twins),
D-259 (m_Delta scale).

## 3. Configuration (one place)

**GD/config.py** (`RunConfig`, defaults = old behaviour):
- `encoder_init` accepts a new value `'data_fit'` (validation at GD/config.py:363-374 extended; the
  `wa_encoder_init` check at :373 allows `'data_fit'` too).
- New fields, each added to `config_json_dict` (GD/config.py:645-662; the coverage assertion at :765-783 enforces it):
  - `encoder_fit_epochs: int = 0`, `encoder_fit_lr: float = 1e-3` (HEURISTIC, set from the pilot, D-261)
  - `loss_weighting: str = 'none'` (`'none' | 'servo'`)
  - `validation_measure: str = 'sim-RMS'` (`'sim-RMS' | 'sim-NRMSe'`)
  - `lbfgs_meters_gate: bool = True`
  - `obc_ref_successor: bool = False` (FIX-11; False reproduces the 2026-10-11 reference set exactly)
- The window length is recorded as the existing resolved key `NA_NB` (GD/config.py:707); `na_nb_override` stays
  exempt (:641). The evaluator already reads `NA_NB` (EV/eval_greybox.py:36).

**EP `_THESIS_FIXED`** (EP:331-338), the only place the final grey-box method is switched on:
- `encoder_init='data_fit'`, `na_nb_override=30` (FIX-8), `encoder_fit_epochs=<pilot>`.
- `obc_space='affine'` (FIX-19, replaces `'tangent'` at :333), `obc_ref_successor=True` (FIX-11).
- `loss_weighting='servo'`, `validation_measure='sim-NRMSe'` (FIX-14, 15, 17).
- `lbfgs_meters_gate=False` (FIX-2), `lbfgs_tol_grad=<pilot>`, `lbfgs_tol_change=<pilot>` (FIX-16; today the thesis
  runs inherit the freeze-pair diagnostic `tol_change=0.0` from EP:203).
- `wa_encoder_init='xavier_uniform_gain1'` stays.
- `THESIS_SMOKE=1` (EP:354-358) also sets `encoder_fit_epochs=1` on a 2 s truncation, so a smoke run passes the
  pre-fit in seconds.
- `THESIS_INIT` default at EP:346 stays `'zero'`; the final runs pass `THESIS_INIT=live` in runs.tsv (FIX-6).
- Run folder: `save_dir` uses `{mode}_{encoder_init}` (GD/config.py:583), so final runs land in
  `thesis_taf_lowpass_noise_data_fit/` and never overwrite the 2026-10-11 `..._linear_map/` results.

## 4. Grey-box training changes, per fix

### Data: servo error without changing `load_traj` (for FIX-14, 15)
- **GD/data.py**: new internal `_load_record(filename, cfg, folder=None)` reads the MAT file ONCE and returns
  `(System_data, e)`, with `e = y - r` from the same `_record_at_fs_new` call extended by field `'r_sim'`
  (:181-196), so trim and grid are identical.
- `load_traj(filename, cfg, folder=None)` (:199-204) keeps returning only the `System_data` (it calls
  `_load_record` and drops `e`); its three callers at :246-248 are unchanged in meaning.
- `load_datasets` (:228-292) calls `_load_record` directly and fills new `DataBundle` fields `e_train`, `e_val`,
  `e_test` (default None, :102-118).
- **GD/data.py**, new `servo_sigma(e_list, k0)`: per record and axis `np.std(e[k0:], axis=0)` (ddof 0); raises on a
  zero or non-finite value. The single definition used everywhere.

### FIX-8: encoder window tau-30 .. tau-1 (D-250)
- **GD/model.py:39-52 `get_encoder_dims`**: new branch for `'data_fit'`: `na = nb = hp['na_nb']` (30),
  `na_right = nb_right = 0`. The `'linear_map'` branch is untouched.
- **GD/model.py:205-240**: the `'data_fit'` branch builds `linear_encoder_init_aug` with `na = nb = na_window - 1`
  (the class takes n and reads n+1 samples, FS/pre_encoder.py:364, 407-408, 431), stated in one comment.
- **Already correct, no change**: validation window `[k0-na : k0+na_r]` with `k0 = max(na, nb)`
  (FS/closed_loop.py:727-730, 761-762); training window offsets (FS/closed_loop.py:346, 424); PS2 `_sample` and
  `M = na + na_right` (FS/ps2_opening.py:146, 165-174). With na = 30, right 0, the first prediction is at sample 30.

### FIX-10: data-based physical encoder (D-251 option A, D-261); covers FIX-1, 7, 9
- **FS/pre_encoder.py `linear_encoder_init_aug.__init__`** (:321-352), one new argument
  `wb_init='reconstructability'` (`# CHANGED:`):
  - `'random'`: skip the map construction (:366-404); draw `Wb_psi_y`, `Wb_psi_u` from U(-1/sqrt(m), 1/sqrt(m)),
    m = (n+1)(nu+ny) = 180 (`# THEORY:` PyTorch nn.Linear default; randomly initialised encoder, encoder paper
    Eq. 35).
  - `'random'`: final layer of psi_tilde (:438-442): physical rows keep the nn.Linear draw, added-state rows and the
    whole bias zero (thesis eq:aug_init).
  - The six mean arguments are not passed in the `'data_fit'` branch, so `fix_enabled=False` (:448-450) and no
    D-055 offsets act (FIX-7 by construction).
- **GD/model.py**, new function `prefit_physical_encoder(fit_sys, cfg, data, norm)` beside `build_model`:
  1. Initial state per training record: positions `q = P^-T y[0]` (as GD/obc_gantry.py:126), zero velocities,
     zero added states, zero controller state; normalised with the stored `norm` scales (FIX-9 satisfied).
  2. One `closed_loop_rollout` (FS/closed_loop.py:245) over all 18 training records in one batch, CPU, `no_grad`,
     at theta^0 with f_aug = 0. A temporary recorder callable is passed as the `hfn` argument; it stores the state
     BEFORE each step and the corrected input it receives (`u_t + u_fb`, FS/closed_loop.py `_rollout_segment`),
     then calls the real step. `fit_sys.hfn` is never replaced and no rollout code is copied. The OBC correction is
     zero here (`theta_frozen` starts at zero, GD/obc_gantry.py:209-210).
  3. Windows at stride `cfg.stride`: input `(y_hat, u_cl)` over tau-30..tau-1, target the physical state at tau.
  4. Adam (`cfg.encoder_fit_lr`, `cfg.encoder_fit_epochs`, batch `hp['batch_size']`) on `Wb_psi_*`, the hidden
     layers and the physical rows of the final layer; gradients of `Wa_psi_*` and the added rows masked; loss = MSE
     on the 6 physical states. Fixed budget from the pilot, no truth gate, no LS warm start (D-261, HEURISTIC).
  5. Prints the start and end fit error.
- **EP `main()`**: call it after `build_closed_loop` (EP:423-425; it needs the controller bank), only when
  `cfg.encoder_init == 'data_fit'` and no `RESUME_CHECKPOINT` is set (a resume restores the fitted encoder).
- The optimizer built in `init_model` (GD/model.py:272) holds the same parameter objects; Adam state is lazy, so the
  pre-fit leaves it valid.

### FIX-6: live g_aug (D-248)
- No code. `live_g_feed_forward_nn` (`model_augmentation/utils/torch_nets.py:170-204`) already implements
  eq:aug_init. runs.tsv rows pass `THESIS_INIT=live`; config.json shows `G_INIT live`.

### FIX-11: OBC tuples with a successor (D-252)
- **GD/obc_gantry.py:128-135 `build_reference_set`**, under `cfg.obc_ref_successor` only (False = today's set):
  `k = np.arange(2, len(u) - 3)`; `x_ref = np.hstack([q[k], qd[k - 2]])`; new stored field
  `x_phys_next_rec = np.hstack([q[k + 1], qd[k - 1]])` (reconstructed successor) beside `x_phys_next` (simulator
  successor, kept).
- **GD/obc_gantry.py `GantryReferenceSet`**: one optional field for the reconstructed successor (None when off).
- **GD/obc_diagnostics.py:~235 `discrepancy_fields`**: when the field exists, the `'data'` discrepancy uses it; the
  simulator successor stays available as a separate simulation-only key.

### FIX-19: span[Phi Gamma] (D-257)
- Config only (`obc_space='affine'`). The affine builder exists (GD/obc_gantry.py:309-330): its eleventh column is
  `F_base(vbar)`, which spans the same space as Gamma = F_base - Phi beta_lin. Evaluation conversion: section 6.

### FIX-15: loss weighted per training record and axis (D-256, D-262)
- **FS/closed_loop.py `WindowControllerIndex`** (:354-432): optional `record_weights` (one `(ny,)` array per record).
  `build` (:374-414) appends a `loss_w` array `(n_windows, ny)` beside `ctrl_ix` from the same per-record window
  counts, so the existing content check covers it (`# CHANGED:`).
- **FS/closed_loop.py `ClosedLoopSimulator`**: `extra_array_names` (:470) becomes an instance attribute,
  `('ctrl_ix',)` or `('ctrl_ix', 'loss_w')` (`# CHANGED:`).
- **FS/interconnect.py `SSE_Interconnect_Composed.loss`** (:1383-1440): `w = Loss_kwargs.pop('loss_w', None)` before
  `simulate`; the existing `L = nn.functional.mse_loss(yfuture, y_pred)` line runs unchanged when `w is None`
  (native reduction order kept); otherwise `L = ((y_pred - yfuture) * w[:, None, :]).square().mean()`
  (`# CHANGED:`, `# THEORY: D-256`). `w = sigma_y / sigma_e` turns the normalised error into error / sigma_e.
- **GD/controller.py `build_closed_loop`** (:269-306): when `cfg.loss_weighting == 'servo'`, computes
  `k0 = max(fs.na, fs.nb)` from the built model and the training weights with `servo_sigma(data.e_train, k0)` and
  `norm.ystd`; passes them to the simulator. Needs `e_train`, `e_val` as new keyword arguments.
- **GD/training.py `_NfProbe.__init__`** (:387-405): the existing row swap for the val batch also swaps
  `record_weights`.
- **Polish and PS2**: the polish's fixed batch takes every named extra array (FS/lbfgs_polish.py:283-305) and calls
  `loss`, so it uses the same weights. PS2 S1 calls `simulate`, not `loss` (to verify in the PS2 smoke).

### FIX-14: selection by NRMS_e (D-256, D-262)
- **GD/controller.py `build_closed_loop`**: computes `val_sigma_e = servo_sigma(data.e_val, k0)` once, from the built
  model's k0, and hands it to the simulator. No lazy computation.
- **FS/closed_loop.py `ClosedLoopSimulator.validation_error`** (:649-690): accepts `'sim-NRMSe'` besides `'sim-RMS'`
  (`# CHANGED:`). Keeps the per-channel RMS that `closed_loop_free_run_rms_batch` already returns (:678, today
  discarded as `_`) in `self.last_per_channel`; keeps the old scalar in `self.last_sim_rms`; returns the mean over
  records and axes of `per_channel / val_sigma_e` for `'sim-NRMSe'`.
- **GD/model.py:349 `train_model`** and **GD/training.py:821**: pass `cfg.validation_measure`; the print at
  GD/training.py:791 names it.

### `[nf gap]` diagnostic units (review finding 7)
- **GD/training.py `_NfProbe.__call__`** (:601-611): the ratios divide `sim.last_sim_rms` (metres) by the window RMS
  (metres), not the selector `value`, which is dimensionless under NRMS_e. The header text at :647 states that the
  objective now weights by 1/sigma_e when `loss_weighting='servo'`.

### FIX-13: validation history with events (D-256, D-267)
- **GD/training.py `_NfProbe.__call__`** (:586-621): at every Adam validation append one row
  `(event='adam', update, epoch, per_channel 6x3 [m], score)` to `self.val_hist`, a list on the probe object, and
  rewrite `val_hist.npz` in the run folder (a few KB, so a crashed run keeps its history).
- **GD/training.py `train_model_with_diagnostics`** (:755-926) keeps a direct reference to the probe (returned by
  `_install_nf_val_probe`, :624-649), so `fit`'s best-checkpoint reload (FS/interconnect.py:1139) cannot truncate
  the list, and appends the other events:
  - `'selected'`: the best Adam index (from `fit_sys.bestfit` and the history);
  - `'rebuilt'`: OBC only, the score of the restored model after the basis rebuild (`val_before` measured by the
    polish, FIX-12);
  - `'polish_candidate'`: `val_after` of the polish and whether it was accepted;
  - `'final'`: the model that is saved and evaluated.
  Each event row stores the per-channel RMS from `sim.last_per_channel`.
- **GD/evaluation.py `_build_save_dict`** (:238-250): saves `val_hist` (events and arrays) and `val_sigma_e` in the
  results npz.

### FIX-12, FIX-17, FIX-2: polish (D-253, D-255, D-263)
- **GD/training.py `run_lbfgs_polish`** (:702-752): `validation_measure=cfg.validation_measure` (:748, FIX-17);
  `val_before=None` when `cfg.obc` (:746-747, FIX-12); `PolishSpec(..., meters_gate=cfg.lbfgs_meters_gate)`.
- **FS/lbfgs_polish.py `PolishSpec`** (:49-64): new field `meters_gate: bool = True` (`# CHANGED:`).
- **FS/lbfgs_polish.py** decide call (:761-765): passes the meters only when `spec.meters_gate`, else `{}`; meters
  are still measured and stored in the outcome (`# CHANGED:`).
- `param_loss` is zero in thesis runs (`param_prior=False`), so it needs no separate handling.

### Polish OBC coefficient after rollback (D-266, review finding 1)
- Defect today: `_restore` (FS/lbfgs_polish.py:162-165) copies parameters only; every closure's `loss()` calls
  `set_frozen` (FS/interconnect.py:1409), so after a rejection (:680, :720, :769) `theta_frozen` still belongs to the
  rejected network, and `_validate` (:584-596) never re-solves it either.
- **FS/lbfgs_polish.py** (`# CHANGED:`): a helper `_resync(fit_sys)` calls
  `fit_sys._sync_prediction_state_for_validation(refresh_basis=False)` when the method exists (no-op without OBC,
  FS/interconnect.py:1340-1341; it re-solves the coefficient and keeps the basis, :1364-1366). Called after each of
  the three `_restore` calls and at the start of `_validate`.

### FIX-16: polish tolerances (D-263)
- No code. The pilot (section 7) measures the weighted opening loss, the gradient infinity norm and the iteration at
  which each tolerance would fire; `lbfgs_tol_grad` and `lbfgs_tol_change` are then set in `_THESIS_FIXED` with a
  `# HEURISTIC:` comment stating the measured values.

### FIX-5: superseded (D-260)
- No code. Check in the PS2 smoke: `M = 30`, decoder output `M * ny`, sampling bound in `_sample` unchanged.

## 5. Black box (FIX-18, D-256, D-264)

- **BB/train_bb_clmap.py**, all defaults unchanged (BB2-018 runs reproduce):
  - `--na` default stays 29 (:38); the runner passes `--na 30`. `k0 = NA` (:129) then scores from 30.
  - New argument `--objective {native, nrmse}`, default `native` (deepSI's own loss, reduction and `sim-RMS`).
  - `nrmse`: a subclass of `SS_encoder_general_hf` in this file. `make_training_data` appends a weight array per
    window (record-axis weight from `servo_sigma` and `model.norm.ystd`, which undoes deepSI's own output
    normalisation); `loss(uhist, yhist, ufuture, yfuture, w=None)` weights as in FIX-15 and appends the first
    non-finite loss with its update count to a run-owned failure list; `cal_validation_error` implements
    `'sim-NRMSe'` and appends per-record, per-axis RMS to a run-owned history list. Both lists are created in
    `main()` and handed to the instance as references, so deepSI's `_best` reload (:155-162) cannot erase them.
  - `nrms_e` (:151, 172) divides by `servo_sigma` instead of RMS(r - y) (:132) under `nrmse`.
  - `result.json` (:168-176) gains `val_rms_hist`, `val_sigma_e`, `failure` (`None` or
    `{'update': n, 'loss': value}`) and `status` (`'failed'` when a failure was recorded, even though an earlier
    checkpoint is reported).
- **BB/runners/run_bb_search.sh**: `SHARED` (:60-61) gains `--na 30 --objective nrmse`; `BB_NITS` default
  20000 -> 80000 (:43); wall time `-t 12:00:00` (measured 1.1 h per 20000 on blade6, so about 4.5 h). CPU
  (`cuda=0` in the 2026-10-11 logs); partitions molokai, lanai; not blade1/oahu (about 5x slower).
- **BB/runners/thesis_r8_grid.conf**: header only: SELECTION by lowest best-validation NRMS_e; limit 80000; "the
  selected search run is R8"; failed configurations listed. The 11 index lines are unchanged.

## 6. Evaluation (no training change)

- **EV/eval_common.py:25**: `K0 = 29` removed; k0 per run from its config `NA_NB` (grey box, checked against the
  built model's `max(na, nb)`) and from `result.json` `k0` (black box). New constants: `TWINS =
  TW/'Data'/'Multisine-test-twins'`, the final run folder and job ids (filled after the runs).
- **Twins**: `load_traj(..., folder=TWINS/...)` loads the 7 twins (I3M, I5M, I6M, E2M, E3M, E4M, E6M) through the
  same resampling, trim and controller registration. Never added to train or validation lists.
- **Splits (D-269)**: every evaluation script scores the training, validation and test records (plus the twins as
  part of test), each with its split label, through the same closed-loop scorer. The records come from
  `data.train_files`, `val_files`, `test_files`; no new loader.
- **EV/eval_greybox.py** (:88, 98, 111): loops over the three splits plus twins; the
  `assert max(fs.na, fs.nb) == K0` compares with the run's own k0.
- **EV/eval_blackbox.py** (:29-35 already takes a folder): the three splits plus twins (it loads train and val
  already, :54); scores from the run's k0.
- **EV/make_results.py** (:40-53): NRMS_e column per record and axis with `servo_sigma`; RMS in metres and floor kept
  (floor only where a noise-free twin exists); multisine and motion-only records reported separately (D-256, revnote
  V:356); one table per split, the main text uses test, the appendix shows training and validation.
- **EV/eval_overlap.py** (:61-68, 82-91):
  - deployed correction from the run's own basis: `make_basis_builder(..., space=cfg.obc_space)` gives `[J]` or
    `[J, F_base]`, so `F - B @ theta_frozen` works for 10 and 11 coefficients.
  - network overlap, truth overlap and predicted first-order bias (eq:obc_recovery, revnote V:384) on three tuple
    stacks built with `build_reference_set(files=...)`: training (the OBC reference set), validation, test I1 to I6
    (D-268). Bias: `dbeta = lstsq(Phi, Xi)` at beta*, mapped to the ten combinations with the block's combination
    Jacobian, scaled per combination with the m_Delta mean-mass rule (D-259), written beside the observed change.
    The training-stack row is labelled as the prediction; validation and test rows as carry-over. Note: on the
    training stack the OBC network overlap is zero by construction.
  - exported Gamma coordinates: theta_Phi' = theta_Phi + c * vbar_lin, theta_Gamma = c (exact for the affine basis).
- **EV/eval_negation.py** (:57-67): same deployed-correction fix; new column `cos(F_aug, F_base - x_tilde)`
  (baseline one-step increment, FIX-19 evaluation).

## 7. Proof per fix group (all local, CPU, seconds to minutes)

| Group | Proof |
|-|-|
| Config | config.json of a smoke run shows every new field and `NA_NB = 30`; coverage assertion passes |
| Data | `load_traj` returns the same `System_data` as before, bitwise; each MAT file is read once |
| FIX-8 | content check passes; first window = samples 0..29; first prediction at sample 30 |
| FIX-10 | pre-fit error falls; `Wa_psi_*` bitwise unchanged; the NONLINEAR added-state output rows of psi_tilde stay zero (the added states themselves are W^a chi, nonzero by design); augmented physical step equals the baseline step from the same state; a resume skips the pre-fit |
| FIX-11 | with the flag off the reference set is bitwise today's; with it on, one tuple fewer per record, Phi of kept tuples bitwise unchanged, reconstructed successor of tuple k equals the state of tuple k+1 |
| FIX-19 | rank 11; `[Phi F_base]^T F_tilde ~ 0` on the training tuples |
| FIX-15 | `loss_weighting='none'` gives the old loss bitwise; one weighted batch equals an offline weighted MSE |
| FIX-14 | NRMS_e of a saved checkpoint equals the offline mean of `cl_val_rms / sigma_e`; `[nf gap]` ratios stay in metres |
| FIX-13 | history survives the `_best` reload and is on disk after every validation; for U the `'selected'` row equals `cl_val_rms` when no polish is accepted; for OBC the `'rebuilt'` row equals a fresh validation of the restored, rebuilt model |
| FIX-12/2/17 | OBC smoke log shows `val_before` measured by the polish (not passed in); a worse `combo_err` cannot veto a better NRMS_e |
| D-266 | predictions identical before the polish and after a forced rejection, also after save and reload |
| FIX-16 | pilot: weighted loss, gradient norm, iteration of the first tolerance trigger, logged in D-263 |
| FIX-18 | `--objective native` reproduces a BB2-018 4-update run bitwise; `nrmse` with unit weights equals deepSI's loss; an injected NaN sets `status: failed` with its update |
| Evaluation | `eval_greybox`/`eval_blackbox` reproduce their run reports; all 7 twins load; a synthetic perturbation recovers a known bias |

Tests go into the existing `scripts/gantry/thesis-results/test_ps2_pipeline.py` and `test_live_comparison.py` where
they belong; one-off checks stay in the session scratchpad.

Final smoke: `THESIS_SMOKE=1 THESIS_DEVICE=cpu` for OBC+PS2, U+PS2, U plain and n_a = 0, plus a 4-update BB run with
`--objective nrmse`. CPU smoke does not test GPU compilation; the first server job does.

## 8. Not changed, with reason

| Item | Reason |
|-|-|
| FIX-1, FIX-7, FIX-9 | superseded by FIX-10; the analytical W^b path stays for old runs, unused by the final path |
| FIX-5 | superseded by FIX-8 (D-260) |
| FIX-6 | config value only |
| Checks 3, 4 | done 2026-10-09 / no reader of the stale `pair` field |
| PS2 S2 input | recorded u_k is correct (FINAL-RUN-FIXES "Checked, no change") |
| OBC refresh cadence | per epoch, confirmed in the R1/R2 logs (201 refresh lines) |
| Git provenance | dropped by Dirk 2026-10-16 |
| Data, data generator | unchanged (Dirk 2026-10-16) |
| Controller-rate preprocessing | unchanged; only the diagnostic (below) |

**Controller-rate diagnostic (D-265)**, `Thesis-writeup/Code/Setup/controller_rate.py:43-50, 58-63`: compare the
4 kHz K1 output with `u_fb[::D]` (same instant), and report separately the in-hold force variation
`RMS(block mean - u_fb[::D]) / RMS(block mean)` as the cost of closing the loop at 4 kHz. The macro values change;
the thesis sentence that uses them is Dirk's to update.

## 9. Runs of the final campaign (job list to be finalised after the smoke runs)

Dirk 2026-10-16: include the appendix runs, and launch the main-result runs first. All grey-box runs: lowpass data,
seed 1, `THESIS_ARCH=mlp`, `THESIS_INIT=live`, new run ids, run-table rows written before launch.

| Priority | Run | Arm | PS2 | n_a | Start | Used in | Hardware, expected time |
|-|-|-|-|-|-|-|-|
| 1 | R1 | OBC | yes | 2 | detuned | main (final model, Aspects 2 to 4) | A100 (lanai, molokai), 13 to 20 h |
| 1 | R4 | U | yes | 2 | detuned | main (Aspect 3) | GPU, about 7 h |
| 1 | R8 | black box | n/a | grid | n/a | main (Aspects 2, 4) + appendix search table | CPU, molokai/lanai, about 4.5 h per point, 11 points |
| 2 | R2 | OBC | yes | 2 | correct | appendix (OBC vs U, correct start) | A100, 13 to 20 h |
| 2 | R5 | U | yes | 2 | correct | appendix (structure, spectrum, OBC vs U) | GPU, about 7 h |
| 2 | R3 | U | no | 2 | correct | appendix (plain vs PS, spectrum) | GPU, about 7 h |
| 2 | R6 | U | no | 0 | correct | appendix (structure) | GPU, about 7 h |
| 2 | R7 | U | yes | 8 | correct | appendix (structure) | GPU, about 7 h |
| 2 | R13 | OBC | no | 0 | detuned | appendix (recovery without added states, Jasper point 5) | A100, 13 to 20 h |
| 2 | R14 | U | no | 0 | detuned | appendix (same) | GPU, about 7 h |

Not planned: optional R9, R10 (n_a 1, 4) and R11, R12 (black-box seeds 2, 3). R13 and R14 need no code change.
`THESIS-RESULTS.md` section 4 was updated to these final settings on 2026-10-16.

## 10. Decisions for Dirk

- **Predicted first-order bias (D-268), decided 2026-10-16:** compute it, and the overlaps, on the training,
  validation and test stacks. The training stack is the prediction of eq:obc_recovery (the stack where theta_aux is
  formed); validation and test show how it carries over. Section V-D ("measured on the one-step tuples of I1 to I6")
  is Dirk's text to update.
- **All splits (D-269), decided 2026-10-16:** every reported metric available for training, validation and test.
- **Validation decides (decided 2026-10-16):** every design choice in the appendix is decided on the validation
  records; test is used only for the final results; training and test numbers are reported beside it.
- **Open:** the agreement criterion between predicted and observed delta_j in table C, stated before the results.

## 11. Review 1 (2026-10-16): findings and resolution

| Finding | Resolution in this revision |
|-|-|
| 1 High: polish rollback leaves `theta_frozen` of the rejected network | D-266, `_resync` after every restore and before every validation (section 4) |
| 2 High: history incomplete; selected entry not equal to the final OBC model | D-267, four event types, written to disk at every validation; proof split by arm (section 7) |
| 3 High: `load_traj` return contract | `_load_record` internal; `load_traj` unchanged; one read per file |
| 4 High: BB NaN recording | run-owned failure list filled in `loss`, persisted in `result.json` with `status: failed` |
| 5 Medium: compatibility contradicted | BB defaults unchanged (`--na 29`, `--objective native`); `obc_ref_successor` flag; promise now covers both |
| 6 Medium: `NA_NB_OVERRIDE` key | existing resolved `NA_NB` used and checked against the model |
| 7 Medium: `[nf gap]` units | ratios use `last_sim_rms`; header text updated |
| Recorder | pre-step state and corrected input; `fit_sys.hfn` never replaced; CPU, one batch; skipped on resume; smoke budget |
| Lazy sigma | removed; computed in `build_closed_loop` from the built model's k0 |
| BB history | run-owned list, not module-global; native reduction order kept |
| Bias stack | decided: training, validation and test stacks, training labelled as the prediction (D-268) |
| Proof wording | nonlinear added-state rows zero; `val_before` checked by how it was measured |

## 12. Implementation notes (2026-10-16)

Where the code differs from sections 3 to 9, and why:
- **FIX-12**: `run_lbfgs_polish` measures the rebuilt model's validation score itself (one free run, the same
  cost as letting the polish measure it) and passes it as `val_before`, so the `rebuilt` history row has its
  per-record values. For U runs `bestfit` is still reused.
- **FIX-11 diagnostic**: the reconstructed successor enters `discrepancy_fields` as a new key `reconstructed`;
  the existing `data` key (simulator successor) keeps its meaning.
- **Gamma-coordinate export**: not implemented; nothing exports theta_aux. The evaluation applies the deployed
  basis `[J F_base]` directly (`eval_overlap.deployed_basis`), so no conversion is needed.
- **Black box**: `--na 30 --objective nrmse` reach the grid through a new runner variable `BB_SHARED_EXTRA`
  (default empty), so `run_bb_search.sh` defaults and the 11 grid lines are unchanged.
- **Final rows**: runs.tsv gains tiers `final-obc-main` (221 = R1), `final-u-main` (224 = R4), `final-obc` (222 =
  R2, 229 = R13), `final-u` (223 = R3, 225 = R5, 226 = R6, 227 = R7, 230 = R14) and `final-smoke` (231 to 240);
  `submit.sh` and `smoke_all.sbatch` (`SMOKE_TIER`) know them. The THESIS block now runs the final method for
  every THESIS_ARM row, so rows 201 to 217 no longer reproduce their 2026-10-11 runs if resubmitted (stated in
  the submit.sh header).
- **Pilot values**: `_ENCODER_FIT_EPOCHS`, `_LBFGS_TOL_GRAD`, `_LBFGS_TOL_CHANGE` at the top of the THESIS block
  are provisional (20, 1e-12, 1e-14) until the pilot (D-261, D-263).
- **Evaluation**: `THESIS_EVAL=final` selects the final runs; `RUNS_FINAL` and the final black-box paths in
  `eval_common.py` are empty until the runs exist, and the final run folders are expected under
  `Thesis-writeup/Results/final/thesis_taf_lowpass_noise_data_fit/`. Old preds caches lack `k0` and `sigma_e`;
  `make_results.py` skips a stem until `eval_greybox.py`/`eval_blackbox.py` are rerun.
- **controller_rate.py**: changed, not run: running it rewrites `SetupKmismatch*` in `Writing/tables/setup_numbers.tex`
  (new `SetupKholdMed`), and the appendix sentence that uses them is Dirk's to update first.
- **Tests**: the proofs of section 7 are one scratch script (`proofs.py`, session scratchpad), not added to
  `test_ps2_pipeline.py`/`test_live_comparison.py`.
