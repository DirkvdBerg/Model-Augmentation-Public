"""
replot_telica_eval.py
---------------------
Regenerate evaluation plots from a saved eval_data_*.pt file without
rerunning the model.

Usage — specific file:
    conda run -n GraduationProject python \\
        scripts/gantry/real-data-verification/replot_telica_eval.py \\
        simulations/param_recovery_telica_xpos_-60_ypos-40/eval_data_best_ETEL_20260629_123456.pt

Usage — replot all eval_data_*.pt in the default save directory:
    conda run -n GraduationProject python \\
        scripts/gantry/real-data-verification/replot_telica_eval.py
"""

__project_origin__ = "added"

import os
import sys
import glob

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch

# Import plot helpers from the main script — also sets matplotlib backend to Agg.
# Module-level patches in run_telica_param_recovery are harmless for plot-only use.
from run_telica_param_recovery import (
    _plot_traj_comparison,
    _plot_residual,
    _SAVE_DIR,
)


def replot(pt_path):
    data     = torch.load(pt_path, map_location='cpu', weights_only=False)
    t_s      = data['t_s']
    q1_eval  = data['q1_eval']
    y_sim    = data['y_sim']
    diff     = data['diff']
    rmse_ch  = data['rmse_ch']
    nrmse_ch = data['nrmse_ch']
    rmse_tot = data['rmse_tot']
    label    = data['label']
    traj_id  = data['traj_id']
    run_id   = data['run_id']
    axes_lbl = data['axes_lbl']
    save_dir = os.path.dirname(pt_path)

    print(f'Replotting: {os.path.basename(pt_path)}')
    _plot_traj_comparison(t_s, q1_eval, y_sim, rmse_ch, nrmse_ch, rmse_tot,
                          axes_lbl, traj_id, save_dir, 'RMS',  label, run_id)
    _plot_traj_comparison(t_s, q1_eval, y_sim, rmse_ch, nrmse_ch, rmse_tot,
                          axes_lbl, traj_id, save_dir, 'NRMS', label, run_id)
    _plot_residual(t_s, diff, rmse_ch, nrmse_ch, rmse_tot,
                   axes_lbl, traj_id, save_dir, 'RMS',  label, run_id)
    _plot_residual(t_s, diff, rmse_ch, nrmse_ch, rmse_tot,
                   axes_lbl, traj_id, save_dir, 'NRMS', label, run_id)
    print(f'  Done — 4 plots written to {save_dir}')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        replot(sys.argv[1])
    else:
        pt_files = glob.glob(os.path.join(_SAVE_DIR, 'eval_data_*.pt'))
        if not pt_files:
            print(f'No eval_data_*.pt found in {_SAVE_DIR}')
            sys.exit(1)
        for pt in sorted(pt_files):
            replot(pt)
