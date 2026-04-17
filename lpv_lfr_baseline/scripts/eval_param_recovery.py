"""
eval_param_recovery.py
----------------------
Evaluate a param-recovery checkpoint.

Prints:
  1. Parameter table  — init / recovered / true / % error
  2. Prediction RMSE  — per trajectory and overall, vs RMSE_baseline

Run:
    conda run -n GraduationProject python -m lpv_lfr_baseline.scripts.eval_param_recovery
"""

import contextlib
import io
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from lpv_lfr_baseline.blocks.lfr_param_block import (
    ParameterizedLFRBlock,
    _PARAM_NAMES,
    _TRUE_PARAMS,
    _DETUNED_PARAMS,
)
from lpv_lfr_baseline.scripts.train_param_recovery import (
    TRAJ_SPECS,
    TRAJ_DIR,
    SAVE_DIR,
    _load_trajectory,
    _load_or_build_state_traj,
    _run_no_grad,
    _aggregate_grouped_rmse,
)
from lpv_lfr_baseline.core.physics import P as _P, ts as _TS

# ── Config ──────────────────────────────────────────────────────────────────
CHECKPOINT = os.path.join(SAVE_DIR, 'checkpoint_e100.pt')
RMSE_BASELINE_CACHE = os.path.join(SAVE_DIR, 'rmse_baseline_cache_v1.pt')


def _sep(widths, char='-'):
    return '  '.join(char * w for w in widths)


def main():
    device = torch.device('cpu')

    # ── Load checkpoint ──────────────────────────────────────────────────────
    ck = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    epoch = ck['epoch']
    print(f'\nCheckpoint  epoch={epoch}  file={os.path.basename(CHECKPOINT)}')

    # ── Build block and restore params ──────────────────────────────────────
    block = ParameterizedLFRBlock(RMSE_baseline=1.0).to(device)
    with torch.no_grad():
        block.log_params.copy_(ck['log_params'])

    # ── 1. Parameter table ───────────────────────────────────────────────────
    recovered = block._recover_params()
    W = [6, 12, 12, 12, 8]
    print(f'\n{"=" * 56}  Parameter recovery')
    print(f'{"Param":<{W[0]}}  {"Init":>{W[1]}}  {"Recovered":>{W[2]}}  {"True":>{W[3]}}  {"Err%":>{W[4]}}')
    print(_sep(W))
    for i, name in enumerate(_PARAM_NAMES):
        init_v = _DETUNED_PARAMS[name]
        rec_v  = recovered[i].item()
        true_v = _TRUE_PARAMS[name]
        err    = (rec_v - true_v) / true_v * 100
        print(f'{name:<{W[0]}}  {init_v:>{W[1]}.4e}  {rec_v:>{W[2]}.4e}  {true_v:>{W[3]}.4e}  {err:>+{W[4]}.2f}%')
    print(_sep(W))

    # ── 2. Prediction RMSE vs baseline ───────────────────────────────────────
    baseline_cache = torch.load(RMSE_BASELINE_CACHE, map_location='cpu', weights_only=False)
    baseline_per_traj = baseline_cache['per_traj']

    eval_entries = []
    W2 = [6, 14, 14, 14, 7]
    print(f'\n{"=" * 62}  Prediction RMSE')
    print(f'{"Traj":<{W2[0]}}  {"RMSE [m]":>{W2[1]}}  {"Baseline [m]":>{W2[2]}}  {"Ratio":>{W2[3]}}')
    print(_sep(W2[:4]))

    for spec in TRAJ_SPECS:
        mat_path = os.path.join(TRAJ_DIR, spec['file'])
        u_i, q1_i, _ = _load_trajectory(mat_path)
        with contextlib.redirect_stdout(io.StringIO()):
            state_traj_i = _load_or_build_state_traj(
                spec['id'], q1_i, _P, _TS, device, SAVE_DIR, load=True
            )
        result = _run_no_grad(block, state_traj_i[:1].to(device), u_i.to(device))
        y_pred = result.Y[0]
        rmse = F.mse_loss(y_pred, q1_i.to(device)).sqrt().item()

        base_entry = baseline_per_traj.get(spec['id'], {})
        base_rmse  = base_entry.get('rmse_total', float('nan'))
        ratio      = rmse / base_rmse if base_rmse > 0 else float('nan')

        eval_entries.append({
            'id':        spec['id'],
            'group':     spec['group'],
            'mse_total': rmse ** 2,
            'rmse_total': rmse,
        })
        print(f'{spec["id"]:<{W2[0]}}  {rmse:>{W2[1]}.4e}  {base_rmse:>{W2[2]}.4e}  {ratio:>{W2[3]}.4f}')

    print(_sep(W2[:4]))
    overall      = _aggregate_grouped_rmse(eval_entries)
    base_entries = [baseline_per_traj[s['id']] for s in TRAJ_SPECS if s['id'] in baseline_per_traj]
    base_overall = _aggregate_grouped_rmse([{**e, 'mse_total': e['rmse_total']**2} for e in base_entries])
    print(f'{"Overall":<{W2[0]}}  {overall:>{W2[1]}.4e}  {base_overall:>{W2[2]}.4e}  {overall/base_overall:>{W2[3]}.4f}')


if __name__ == '__main__':
    main()
