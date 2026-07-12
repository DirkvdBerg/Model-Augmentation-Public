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

# ── Search knobs (the only tunables) ─────────────────────────────────────────
# ORTH-PROJECTION SMOKE RUN (2026-07-12, plan Step 9 via optuna per user; D-111 basis).
# Single trial, no search: joint estimation (detuned start) + Theta routing +
# orth penalty at beta_center=4.66e-4 (D7.9: V_MSE/E_drift = 1e-4 / 2.15e-1).
# Judged ONLY on loss-component health (run-table row, problem log Sect. 12):
# (1) mse / param_loss / V_orth all finite every chunk; (2) V_orth responds to
# training (changes once the ANN moves; exactly 0 at zero-init is correct);
# (3) no optimizer collapse (train nf-RMS not rising monotonically -> lr ok).
# Model quality is explicitly NOT judged here. lr=1e-5 = Theta-routing rate
# (D-101 era); WATCH: if train nf-RMS rises from epoch 1 -> lr overshoot.
# First build_model call triggers a fresh ~6 min penalty-basis build at the
# DETUNED theta_bar (D7.5; cache key includes theta_bar + states='data').
N_TRIALS        = 1           # single config, no search
OPTUNA_EPOCHS   = 5           # smoke horizon
CHUNK_SIZE      = 5           # one chunk = the whole run
LR_LOW, LR_HIGH = 1e-5, 1e-5  # FIXED lr=1e-5: Theta-routing rate (D-101 era)
NF_LOW, NF_HIGH, NF_STEP = 400, 400, 1000     # FIXED nf=400
SEARCH_VAL_SAMPLES = 8000     # trials validate on one cropped val (see swap below); full val for final
FINAL_EPOCHS    = 5           # short final (full-val) pass: yields the joint param table
STUDY_BASE_NAME = "gantry_orth_smoke"

# ── Regime: set once, single source of truth (mirrors the entry file's CFG) ──
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


if __name__ == '__main__':
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
