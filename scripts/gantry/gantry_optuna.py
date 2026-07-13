"""
gantry_optuna.py — learning-rate search (Phase 1) for gantry augmentation.

Run: conda run -n GraduationProject python scripts/gantry/gantry_optuna.py

TPE + MedianPruner search over ONE dimension: the learning rate, for the
X+Theta+Y routing at a fixed nf (=400) and a large stride (fast). nf is NOT
searched: against val sim-RMS it has no interior optimum (longer horizon just
sees more drift), so it is a fixed regime knob here, swept separately if needed.

Design (clean vs gantry_interconnect_dynamic.py):
  * ONE config surface. Each trial is dataclasses.replace(CFG, lr=...) and the
    hp dict is the derived view cfg_t.hp -- no hand-built hp (D-100).
  * The regime (routing, nf, stride) is set once in the module-level CFG below.
  * Trials are lightweight: prints + the SQLite study DB + a trials CSV.
  * The single best config is retrained via train_model_with_diagnostics +
    evaluate_and_save, so its output is identical in shape to one entry-file run
    (same figures, including the train/val nf-RMS meter figure, D-102).
  * Two search figures: lr-vs-objective scatter (the lr-basin plot) and the
    best-so-far optimization history.
"""
__project_origin__ = "added"

import os
import json
import dataclasses
from datetime import datetime

import numpy as np
import torch
import deepSI
import optuna
from optuna.samplers import TPESampler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gantry_dynamic.config import RunConfig, save_dir
from gantry_dynamic.data import load_datasets, compute_normalization
from gantry_dynamic.model import build_model, train_model
from gantry_dynamic.training import train_model_with_diagnostics, _install_nf_val_probe
from gantry_dynamic.evaluation import evaluate_and_save
from gantry_dynamic.diagnostics import state_recovery_diagnostic

# ── MODE ─────────────────────────────────────────────────────────────────────
MODE = 'curriculum'   # 'curriculum' = warm-started nf-ladder (base X+Theta+Y learning run, 2026-07-12);
                      # 'orth_smoke' = the previous orth-projection single-trial run (preserved below)

# ── NF CURRICULUM (MODE='curriculum'): warm-started ladder, ONE fit_sys ───────
# Supervisors recommended increasing nf; the ANN is not learning at nf=400 because
# the FP residual there is near the floor (problem log Sect. 7 weak-signal). Longer
# nf accumulates more absorber signal (Sect. 8) -> real gradient. This is a CURRICULUM,
# not a search: nf has no interior optimum, so each rung trains at a longer nf,
# WARM-STARTED from the previous rung's trained weights (build_model ONCE; after fit()
# reloads _best under the sim-RMS selector, we reload _last to recover the trained
# weights, proven in make_drift_checkpoint.py). lr is FIXED at CFG.lr (per-rung lr is
# ignored by fit once init_model_done, D-101); overshoot handled REACTIVELY by watching
# the per-epoch train nf-RMS print, not scheduled (user 2026-07-12).
# WATCH (pre-declared): do train nf-RMS (windows) and full sim-RMS improve TOGETHER, or
# SPLIT? Split = drift is separate from signal (d8-d12) and needs Layer 2 on top; the
# nf climb at lr=1e-7 is also the clean never-run test of nf-conditioning (69399 was
# confounded by the lr bug).
NF_LADDER = [        # (nf, epochs) warm-started in order; ~26 epochs total, budget 12h
    (400,  8),
    (800,  7),
    (1600, 6),
    (2000, 5),       # trim if memory/wall-clock exceed budget; nf=4000 stays off (566MB wall)
]

# ── Search / orth-smoke knobs (MODE != 'curriculum') ──────────────────────────
# ORTH-PROJECTION SMOKE RUN (2026-07-12): joint estimation (detuned) + Theta routing
# + orth penalty at beta_center=4.66e-4. Judged on loss-component health only.
N_TRIALS        = 1           # single config, no search
OPTUNA_EPOCHS   = 5           # smoke horizon
CHUNK_SIZE      = 5           # one chunk = the whole run
LR_LOW, LR_HIGH = 1e-5, 1e-5  # FIXED lr=1e-5: Theta-routing rate (D-101 era)
NF_LOW, NF_HIGH, NF_STEP = 400, 400, 1000     # FIXED nf=400
SEARCH_VAL_SAMPLES = 8000     # trials validate on one cropped val (see swap below); full val for final
FINAL_EPOCHS    = 5           # short final (full-val) pass: yields the joint param table
STUDY_BASE_NAME = "gantry_orth_smoke"

# ── Regime: set once, single source of truth (mirrors the entry file's CFG) ──
if MODE == 'curriculum':
    CFG = RunConfig(
        ann_route_ix=(0, 1, 2, 3, 4, 5, 6, 7),  # FULL X+Theta+Y: the deliverable routing (D-103)
        stride=100,                             # large -> fast epochs (consistent with prior runs)
        lr=1e-7,                                # K=0 routing rate (override the 1e-4 default!)
        joint_estimation=False,                 # free ANN: base learning run, no negation
        param_init_detune=None,                 # nominal theta: correct baseline, ANN learns the residual
        # orth_beta default 0.0 = off: no penalty-basis build (~6 min skipped)
    )
else:
    CFG = RunConfig(
        ann_route_ix=(1, 4, 6, 7),   # Theta+absorber: the approved machinery-validation routing (D-103 guard)
        stride=100,                  # large -> fast smoke (user 07-12)
        nf_seconds=0.100,            # nf = 400
        joint_estimation=True,       # negation only exists with trainable theta
        # param_init_detune: RunConfig default 14-vector (+-10%) -> detuned start (run-D style)
        orth_beta=4.66e-4,           # beta_center (D7.9 layer 1, measured 2026-07-12)
    )

RUN_ID = os.environ.get('SLURM_JOB_ID') or datetime.now().strftime('%Y%m%d_%H%M%S')
SDIR   = save_dir(CFG)
os.makedirs(SDIR, exist_ok=True)

# Build data + normalization once (seeded, as the entry file does before data/noise).
np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)
DATA = load_datasets(CFG)
NORM = compute_normalization(CFG, DATA)

# Cheap validation for the SEARCH only. Trials validate on ONE val trajectory
# cropped to SEARCH_VAL_SAMPLES instead of the full val_ckpt_data (4 files x 48k
# = 192k samples, ~10 min/validation on CPU -> infeasible for 30 trials). The
# full selector is restored before the final retrain so the winner is chosen and
# reported on the real validation set. NOTE: a short cropped sim-RMS is closer to
# the training horizon (less K=0-drift-dominated), so the found lr is "lr that
# trains well on this horizon" -- confirm the winner on full val afterwards.
_FULL_VAL_CKPT = DATA.val_ckpt_data
_v0 = DATA.val_list[0]
DATA.val_ckpt_data = deepSI.System_data(
    u=_v0.u[:SEARCH_VAL_SAMPLES], y=_v0.y[:SEARCH_VAL_SAMPLES], dt=_v0.dt)
print(f"Search validation: {DATA.val_ckpt_data.y.shape[0]} samples "
      f"(cropped from {sum(v.y.shape[0] for v in DATA.val_list)} over {len(DATA.val_list)} files)")

# Cropped train trajectory for the per-epoch nf-RMS probe (kept cheap for the search).
_t0 = DATA.train_list[0]
SEARCH_TRAIN = deepSI.System_data(
    u=_t0.u[:SEARCH_VAL_SAMPLES], y=_t0.y[:SEARCH_VAL_SAMPLES], dt=_t0.dt)


def next_study_name(base_name, directory):
    """First unused study name so re-runs create fresh SQLite DBs, not collisions."""
    version = 1
    while True:
        name = base_name if version == 1 else f"{base_name}_v{version}"
        if not os.path.exists(os.path.join(directory, f"optuna_{name}.db")):
            return name
        version += 1


def objective(trial):
    lr    = trial.suggest_float("lr", LR_LOW, LR_HIGH, log=True)
    nf    = trial.suggest_int("nf", NF_LOW, NF_HIGH, step=NF_STEP)
    cfg_t = dataclasses.replace(CFG, lr=lr, nf_override=nf, epochs=OPTUNA_EPOCHS)
    hp    = cfg_t.hp   # derived view (D-100), not a hand-built dict

    print(f"\n{'='*70}\nTrial {trial.number}:  lr={lr:.3e}  nf={nf}  "
          f"(routing={cfg_t.ann_route_ix}, stride={cfg_t.stride})\n{'='*70}")

    trial_seed = CFG.seed + trial.number   # distinct init per trial, reproducible
    np.random.seed(trial_seed)
    torch.manual_seed(trial_seed)

    try:
        fit_sys = build_model(hp, cfg_t, DATA, NORM)
        # Chunked so MedianPruner can kill a diverging lr early. bestfit persists
        # across fit() calls (init_model_done=True -> no reset) and the end-of-fit
        # _best reload keeps the global best, so trial.report sees the running best.
        for chunk in range(OPTUNA_EPOCHS // CHUNK_SIZE):
            # Per-epoch train/val nf-RMS prints (D-102). Reinstalled each chunk: the
            # end-of-fit _best reload restores cal_validation_error to the probe's
            # no-op, so it must be re-attached before every train_model call.
            _orig_cve = _install_nf_val_probe(fit_sys, hp, cfg_t, SEARCH_TRAIN, DATA.val_ckpt_data)
            try:
                train_model(fit_sys, hp, cfg_t, DATA, epochs=CHUNK_SIZE)
            finally:
                fit_sys.cal_validation_error = _orig_cve
            # A diverging lr can make bestfit NaN/inf without raising; report a large
            # finite value so the pruner/sampler stay well-defined, then prune.
            # Orth-smoke loss decomposition (plan Step 9): print the penalty and
            # param_loss components alongside the total each chunk.
            with torch.no_grad():
                _pl = sum(float(m.param_loss()) for m in fit_sys.hfn.connected_blocks
                          if hasattr(m, 'param_loss'))
                if getattr(fit_sys, 'orth_penalty', None) is not None:
                    from model_augmentation.fit_systems.blocks import Static_ANN_Block as _SAB
                    _ann = next(m for m in fit_sys.hfn.connected_blocks if isinstance(m, _SAB))
                    _ov = float(fit_sys.orth_penalty(_ann))
                else:
                    _ov = 0.0
            print(f"  [orth-probe] V_orth={_ov:.4e}  param_loss={_pl:.4e}  "
                  f"bestfit={fit_sys.bestfit:.6e}")
            bf = fit_sys.bestfit if np.isfinite(fit_sys.bestfit) else 1e6
            trial.report(bf, chunk)
            if not np.isfinite(fit_sys.bestfit):
                print(f"  Trial {trial.number} non-finite bestfit -> treated as failed")
                return float('inf')
            if trial.should_prune():
                print(f"  Trial {trial.number} PRUNED at chunk {chunk+1}")
                raise optuna.TrialPruned()
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial {trial.number} FAILED: {e}")
        return float('inf')

    print(f"Trial {trial.number}: bestfit (val sim-RMS) = {fit_sys.bestfit:.6f}")
    return fit_sys.bestfit


def _save_search_figures(study, out_dir, rid):
    """lr-vs-objective scatter (colored by nf) + best-so-far history (no plotly dep)."""
    try:
        done = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
                and np.isfinite(t.value)]
        if not done:
            print("No completed trials to plot."); return
        lrs  = [t.params["lr"] for t in done]
        nfs  = [t.params["nf"] for t in done]
        vals = [t.value for t in done]
        order = np.argsort([t.number for t in done])
        running_best = np.minimum.accumulate([vals[i] for i in order])

        fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4))
        sc = axL.scatter(lrs, vals, c=nfs, cmap='viridis', s=30)
        fig.colorbar(sc, ax=axL, label='nf')
        best = study.best_trial
        axL.scatter([best.params["lr"]], [best.value], c='C3', s=110, marker='*',
                    label=f'best lr={best.params["lr"]:.2e}, nf={best.params["nf"]}')
        axL.set_xscale('log'); axL.set_yscale('log')
        axL.set_xlabel('learning rate'); axL.set_ylabel('val sim-RMS (bestfit)')
        axL.set_title('lr vs objective (color = nf)'); axL.legend(fontsize=8); axL.grid(True, which='both')

        axR.plot([done[i].number for i in order], running_best, 'C1-o', ms=4)
        axR.set_yscale('log'); axR.set_xlabel('trial'); axR.set_ylabel('best val sim-RMS so far')
        axR.set_title('Optimization history'); axR.grid(True, which='both')

        fig.tight_layout()
        p = os.path.join(out_dir, f'optuna_lr_nf_search_{rid}.png')
        fig.savefig(p, dpi=150); plt.close('all')
        print(f"Saved search figure: {p}")
    except Exception as e:
        print(f"Warning: search-figure plotting failed: {e}")


def run_curriculum_main():
    """Warm-started nf curriculum on ONE fit_sys (base X+Theta+Y learning run).

    build_model ONCE, then climb NF_LADDER. Each rung warm-starts from the previous
    rung's TRAINED weights: fit() reloads _best at its end (=possibly epoch 0 under the
    sim-RMS selector on the drift route), so after each rung we reload _last to recover
    the actually-trained weights before the next rung (proven in make_drift_checkpoint.py).
    lr is fixed at CFG.lr (per-rung lr is ignored by fit once init_model_done, D-101);
    overshoot is watched via the per-epoch train nf-RMS print, not scheduled.
    """
    curr_dir = os.path.join(SDIR, f'curriculum_{RUN_ID}')
    os.makedirs(curr_dir, exist_ok=True)
    print(f"\n{'='*70}\nNF CURRICULUM  routing={CFG.ann_route_ix}  lr={CFG.lr:.1e}  "
          f"ladder={NF_LADDER}\n  -> {curr_dir}\n{'='*70}")

    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)
    fit_sys = build_model(CFG.hp, CFG, DATA, NORM)   # built ONCE; warm-started across rungs

    for rung, (nf, epochs) in enumerate(NF_LADDER):
        cfg_r = dataclasses.replace(CFG, nf_override=nf, epochs=epochs)
        hp = cfg_r.hp
        print(f"\n{'-'*70}\nRUNG {rung}: nf={nf} ({nf*CFG.ts_new:.2f}s)  epochs={epochs}  "
              f"lr={CFG.lr:.1e}  routing={CFG.ann_route_ix}\n{'-'*70}")
        fit_sys.bestfit = float('inf')   # per-rung _best; _last = this rung's final weights
        _orig = _install_nf_val_probe(fit_sys, hp, cfg_r, SEARCH_TRAIN, DATA.val_ckpt_data)
        try:
            train_model(fit_sys, hp, cfg_r, DATA, epochs=epochs, nf=nf)
        finally:
            fit_sys.cal_validation_error = _orig
        # Recover the trained weights (fit reloaded _best) so the next rung warm-starts.
        fit_sys.checkpoint_load_system(name='_last')
        # The nf-probe (D-095) was pickled into _last as _noop_cve (returns None);
        # checkpoint_load_system replaces __dict__ wholesale, so it shadows the real
        # cal_validation_error. The next rung's probe would then wrap a None-returning
        # function -> validation() crashes on `bestfit >= None`. Drop the instance
        # attr so lookup falls back to the class method (System_torch.cal_validation_error).
        fit_sys.__dict__.pop('cal_validation_error', None)
        tr = np.array(getattr(fit_sys, 'Loss_train_nf', []), dtype=float)
        vl = np.array(getattr(fit_sys, 'Loss_val_nf', []), dtype=float)
        if tr.size and vl.size:
            print(f"  [rung {rung} summary] train nf-RMS {tr[0]:.4e} -> {tr[-1]:.4e}   "
                  f"val nf-RMS {vl[0]:.4e} -> {vl[-1]:.4e}  (@nf={nf})")
        torch.save(fit_sys.__dict__, os.path.join(curr_dir, f'rung{rung}_nf{nf}_last.pth'))

    # Final deliverable evaluation on the FULL validation set + figures.
    DATA.val_ckpt_data = _FULL_VAL_CKPT
    final_cfg = dataclasses.replace(CFG, nf_override=NF_LADDER[-1][0], epochs=NF_LADDER[-1][1])
    rid = f'curriculum_{RUN_ID}'
    try:
        evaluate_and_save(fit_sys, final_cfg.hp, rid, final_cfg, DATA, NORM, curr_dir)
    except Exception as e:
        print(f"evaluate_and_save failed ({e}); rung checkpoints are saved in {curr_dir}.")
    print(f"\nCurriculum complete -> {curr_dir}")


def run_search_main():
    study_name = next_study_name(STUDY_BASE_NAME, SDIR)
    storage    = f"sqlite:///{os.path.join(SDIR, f'optuna_{study_name}.db')}"
    # GridSampler for a DETERMINISTIC nf sweep (2026-07-11): TPE + 3 trials could repeat an nf and miss one;
    # GridSampler runs exactly {lr}x{nf grid}. lr is a singleton (fixed 1e-7). n_trials auto = #grid points.
    _nf_grid = list(range(NF_LOW, NF_HIGH + 1, NF_STEP))   # {2000,3000,4000}
    study = optuna.create_study(
        study_name=study_name, storage=storage,
        sampler=optuna.samplers.GridSampler({"lr": [LR_LOW], "nf": _nf_grid}),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
        direction="minimize",
    )
    print(f"\nOptuna lr+nf search '{study_name}' — {N_TRIALS} trials, "
          f"lr in [{LR_LOW:.0e},{LR_HIGH:.0e}], nf in [{NF_LOW},{NF_HIGH}] step {NF_STEP}, "
          f"stride={CFG.stride}\n")
    study.optimize(objective, n_trials=N_TRIALS)

    print(f"\n{'='*70}\nSEARCH COMPLETE — {len(study.trials)} trials")
    print(f"Best trial #{study.best_trial.number}:  lr={study.best_params['lr']:.3e}  "
          f"nf={study.best_params['nf']}  val sim-RMS={study.best_value:.6f}\n{'='*70}")
    for t in sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf')):
        val = f"{t.value:.6f}" if t.value is not None else "None"
        lr  = t.params.get("lr", float('nan')); nf = t.params.get("nf", -1)
        print(f"  #{t.number:3d}  lr={lr:.2e}  nf={nf}  val={val}  [{t.state.name}]")

    # Artifacts: trials CSV + best-params JSON + search figures.
    study.trials_dataframe().to_csv(os.path.join(SDIR, f'optuna_trials_{RUN_ID}.csv'), index=False)
    with open(os.path.join(SDIR, f'optuna_best_params_{RUN_ID}.json'), 'w') as f:
        json.dump(dict(best_trial=study.best_trial.number,
                       best_value=study.best_value, best_params=study.best_params), f, indent=2)
    _save_search_figures(study, SDIR, RUN_ID)

    # Retrain the single best (lr, nf) as a full run: identical output shape to the
    # entry file (evaluate_and_save + state_recovery_diagnostic). Baselines are
    # omitted (evaluate_and_save is None-safe) to keep the search self-contained.
    DATA.val_ckpt_data = _FULL_VAL_CKPT   # restore the real val selector for the honest final run
    best_rid  = f"optuna_best_{RUN_ID}"
    best_cfg  = dataclasses.replace(CFG, lr=study.best_params["lr"],
                                    nf_override=study.best_params["nf"], epochs=FINAL_EPOCHS)
    final_dir = os.path.join(SDIR, best_rid)
    os.makedirs(final_dir, exist_ok=True)
    print(f"\nRetraining best lr={best_cfg.lr:.3e}, nf={best_cfg.nf} for {FINAL_EPOCHS} epochs -> {final_dir}")
    np.random.seed(CFG.seed + study.best_trial.number)
    torch.manual_seed(CFG.seed + study.best_trial.number)
    fit_sys = build_model(best_cfg.hp, best_cfg, DATA, NORM)
    _, diag_conv = train_model_with_diagnostics(
        fit_sys, best_cfg.hp, best_cfg, DATA, NORM,
        checkpoint_dir=final_dir, run_id=best_rid)
    evaluate_and_save(fit_sys, best_cfg.hp, best_rid, best_cfg, DATA, NORM, final_dir,
                      diag_conv=diag_conv)
    state_recovery_diagnostic(fit_sys, best_cfg.hp, best_rid, best_cfg, DATA, NORM, final_dir)


if __name__ == '__main__':
    (run_curriculum_main if MODE == 'curriculum' else run_search_main)()
