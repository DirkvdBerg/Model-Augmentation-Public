"""
gantry_optuna.py — Optuna hyperparameter search for gantry augmentation.

Run: conda run -n GraduationProject python scripts/gantry/gantry_optuna.py

Uses the gantry_dynamic package: builds RunConfig + data + normalization once,
then drives a TPE study with median-pruner chunked training.
"""
__project_origin__ = "added"

import os
import json
from datetime import datetime

import numpy as np
import torch
import optuna
from optuna.samplers import TPESampler

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gantry_dynamic.config import RunConfig, save_dir
from gantry_dynamic.data import load_datasets, compute_normalization
from gantry_dynamic.model import build_model, train_model
from gantry_dynamic.evaluation import evaluate_and_save
from gantry_dynamic.diagnostics import state_recovery_diagnostic

# ── Study config ─────────────────────────────────────────────────────────────
N_TRIALS        = 40
STUDY_BASE_NAME = "gantry_subnet_augmented"
OPTUNA_EPOCHS   = 30
CHUNK_SIZE      = 10

# ── Build config, data, normalization once (import-time cost was the same before) ─
CFG    = RunConfig()
SDIR   = save_dir(CFG)
RUN_ID = os.environ.get('SLURM_JOB_ID') or datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs(SDIR, exist_ok=True)

np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)
DATA = load_datasets(CFG)
NORM = compute_normalization(CFG, DATA)
DEFAULT_HP = CFG.hp


def next_study_name(base_name, directory):
    version = 1
    while True:
        name = base_name if version == 1 else f"{base_name}_v{version}"
        if not os.path.exists(os.path.join(directory, f"optuna_{name}.db")):
            return name
        version += 1


def objective(trial):
    hp = dict(
        NX_ANN            = trial.suggest_categorical("NX_ANN", [2, 4]),
        n_nodes_per_layer = trial.suggest_categorical("n_nodes_per_layer", [64, 128, 256]),
        n_hidden_layers   = trial.suggest_int("n_hidden_layers", 1, 3),
        na_nb             = trial.suggest_int("na_nb", 10, 50, step=5),
        nf                = trial.suggest_int("nf", 50, 500, step=50),
        batch_size        = trial.suggest_categorical("batch_size", [1000, 2000, 4000]),
        lr                = trial.suggest_float("lr", 5e-5, 5e-3, log=True),
        epochs            = OPTUNA_EPOCHS,
        up_sample         = DEFAULT_HP['up_sample'],
    )
    hp['na_nb'] = (CFG.nx_phys + hp['NX_ANN']) * 2 + 1

    print(f"\n{'='*70}\nTrial {trial.number}")
    for k, v in hp.items():
        print(f"  {k}: {v}")
    print('='*70)

    trial_seed = CFG.seed + trial.number
    np.random.seed(trial_seed)
    torch.manual_seed(trial_seed)

    try:
        fit_sys = build_model(hp, CFG, DATA, NORM)
        n_chunks = OPTUNA_EPOCHS // CHUNK_SIZE
        for chunk in range(n_chunks):
            train_model(fit_sys, hp, CFG, DATA, epochs=CHUNK_SIZE)
            trial.report(fit_sys.bestfit, chunk)
            if trial.should_prune():
                print(f"  Trial {trial.number} PRUNED at chunk {chunk+1}/{n_chunks}")
                raise optuna.TrialPruned()
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial {trial.number} FAILED: {e}")
        return float('inf')

    print(f"\nTrial {trial.number} finished: bestfit = {fit_sys.bestfit:.6f}")
    return fit_sys.bestfit


if __name__ == '__main__':
    study_name = next_study_name(STUDY_BASE_NAME, SDIR)
    db_path    = os.path.join(SDIR, f"optuna_{study_name}.db")
    storage    = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=TPESampler(seed=CFG.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
        direction="minimize",
    )

    print(f"\nOptuna study '{study_name}' — {N_TRIALS} trials")
    print(f"DB: {db_path}\n")
    study.optimize(objective, n_trials=N_TRIALS)

    print(f"\n{'='*70}")
    print(f"OPTUNA COMPLETE — {len(study.trials)} trials")
    print(f"{'='*70}")
    print(f"Best trial:  #{study.best_trial.number}")
    print(f"Best value:  {study.best_value:.6f}")
    print(f"Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    print(f"\nAll trials (sorted by value):")
    for t in sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf')):
        val_str = f"{t.value:.6f}" if t.value is not None else "None"
        print(f"  #{t.number:3d}  val={val_str}  [{t.state.name}]  {t.params}")

    csv_path = os.path.join(SDIR, f'optuna_trials_{RUN_ID}.csv')
    study.trials_dataframe().to_csv(csv_path, index=False)
    print(f"\nSaved trials CSV: {csv_path}")

    best_params_path = os.path.join(SDIR, f'optuna_best_params_{RUN_ID}.json')
    with open(best_params_path, 'w') as f:
        json.dump(dict(
            best_trial=study.best_trial.number,
            best_value=study.best_value,
            best_params=study.best_params,
        ), f, indent=2)
    print(f"Saved best params: {best_params_path}")

    # Retrain best
    best_hp = {**study.best_params, 'epochs': 200,
               'up_sample': DEFAULT_HP['up_sample']}
    best_hp['na_nb'] = (CFG.nx_phys + best_hp['NX_ANN']) * 2 + 1
    print(f"\nRetraining best configuration for {best_hp['epochs']} epochs...")
    np.random.seed(CFG.seed + study.best_trial.number)
    torch.manual_seed(CFG.seed + study.best_trial.number)
    fit_sys = build_model(best_hp, CFG, DATA, NORM)
    train_model(fit_sys, best_hp, CFG, DATA)
    evaluate_and_save(fit_sys, best_hp, f"optuna_best_{RUN_ID}", CFG, DATA, NORM, SDIR)
    state_recovery_diagnostic(fit_sys, best_hp, f"optuna_best_{RUN_ID}", CFG, DATA, NORM, SDIR)
