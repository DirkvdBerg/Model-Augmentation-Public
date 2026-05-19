"""
plot_trajectories.py
--------------------
Motion profile plots for all trajectories in a dataset.

Each figure shows one trajectory:
  Row 1  — Position [m]:  X1 | X2 | Y
  Row 2  — Force   [N]:  FX1 | FX2 | FY

Run as:
    conda run -n GraduationProject python -m lpv_lfr_baseline.plots.plot_trajectories
"""

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.io import loadmat

# ── Dataset selector — mirrors train_param_recovery.py ───────────────────────
DATASET = 'base_extended'

_BASE = os.path.join(os.path.dirname(__file__), '..', '..')

_TRAJ_BASE = (
    {'id': 'T1', 'file': 'T1_Y_sweep_conservative.mat'},
    {'id': 'T2', 'file': 'T2_X_sym_Y030.mat'},
    {'id': 'T3', 'file': 'T3_X_sym_Y000.mat'},
    {'id': 'T4', 'file': 'T4_X_antisym_Y020.mat'},
    {'id': 'T5', 'file': 'T5_X_sym_Y_sweep.mat'},
    {'id': 'T6', 'file': 'T6_Y_sweep_aggressive.mat'},
)
_TRAJ_EXTENDED = _TRAJ_BASE + (
    {'id': 'T7', 'file': 'T7_X_antisym_Y_sweep.mat'},
    {'id': 'T8', 'file': 'T8_X_sym_anti_Y_sweep.mat'},
)

_DATASETS = {
    'base': dict(
        traj_dir   = os.path.join(_BASE, 'Matlab-output', 'parameter-recovery'),
        traj_specs = _TRAJ_BASE,
    ),
    'multisine': dict(
        traj_dir   = os.path.join(_BASE, 'Matlab-output', 'parameter-recovery-multisine'),
        traj_specs = _TRAJ_EXTENDED,
    ),
    'ref_injection': dict(
        traj_dir   = os.path.join(_BASE, 'Matlab-output', 'parameter-recovery-ref-injection'),
        traj_specs = _TRAJ_EXTENDED,
    ),
    'identification': dict(
        traj_dir   = os.path.join(_BASE, 'Matlab-output', 'identification-trajectories'),
        traj_specs = _TRAJ_EXTENDED,
    ),
    'base_extended': dict(
        traj_dir   = os.path.join(_BASE, 'Matlab-output', 'identification-trajectories-no-multisine'),
        traj_specs = _TRAJ_EXTENDED,
    ),
}

SAVE_DIR = os.path.join(os.path.dirname(__file__), 'figures', DATASET)

# ── Style ─────────────────────────────────────────────────────────────────────
_C_POS   = '#1f77b4'   # position — blue
_C_FORCE = '#d62728'   # force    — red

plt.rcParams.update({
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.linewidth':     0.4,
    'grid.alpha':         0.5,
    'font.size':          9,
})


def _human_title(spec):
    """'T1_Y_sweep_conservative.mat'  →  'T1 — Y sweep conservative'"""
    stem   = spec['file'].replace('.mat', '')          # T1_Y_sweep_conservative
    suffix = stem[len(spec['id']) + 1:]                # Y_sweep_conservative
    return f'{spec["id"]} \u2014 {suffix.replace("_", " ")}'


_MIN_POS_RANGE   = 1e-3   # m  — don't zoom into sub-mm noise on stationary axes
_MIN_FORCE_RANGE = 1.0    # N  — don't zoom into sub-Newton noise


def _tidy_axis(ax, min_range):
    """Suppress offset notation and enforce a minimum y-axis range."""
    fmt = mticker.ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    ax.yaxis.set_major_formatter(fmt)
    lo, hi = ax.get_ylim()
    if hi - lo < min_range:
        mid = (lo + hi) / 2
        ax.set_ylim(mid - min_range / 2, mid + min_range / 2)


def plot_trajectory(spec, traj_dir, save_dir):
    mat = loadmat(os.path.join(traj_dir, spec['file']))
    t   = mat['t_sim'].squeeze()   # (T,)  seconds
    q1  = mat['q1']                # (T, 3)  measured position [X1, X2, Y]  m
    u   = mat['u_q1']              # (T, 3)  plant force       [FX1, FX2, FY]  N

    fig, axes = plt.subplots(
        2, 3,
        figsize=(13, 5.5),
        sharex=True,
        gridspec_kw={'hspace': 0.12, 'wspace': 0.40},
    )
    fig.suptitle(_human_title(spec), fontsize=12, fontweight='bold', y=1.01)

    pos_labels   = ['X1  [m]',  'X2  [m]',  'Y  [m]']
    force_labels = ['FX1  [N]', 'FX2  [N]', 'FY  [N]']

    for col in range(3):
        # ── Row 0: position ───────────────────────────────────────────────
        ax = axes[0, col]
        ax.plot(t, q1[:, col], color=_C_POS, lw=1.0)
        ax.set_ylabel(pos_labels[col])
        _tidy_axis(ax, _MIN_POS_RANGE)

        # ── Row 1: force ──────────────────────────────────────────────────
        ax = axes[1, col]
        ax.plot(t, u[:, col], color=_C_FORCE, lw=1.0)
        ax.set_ylabel(force_labels[col])
        ax.set_xlabel('Time  [s]')
        _tidy_axis(ax, _MIN_FORCE_RANGE)

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'{spec["id"]}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'  Saved: {path}')
    plt.close(fig)


def main():
    ds = _DATASETS[DATASET]
    print(f'Dataset : {DATASET}')
    print(f'Saving  : {SAVE_DIR}\n')
    for spec in ds['traj_specs']:
        plot_trajectory(spec, ds['traj_dir'], SAVE_DIR)
    print('\nDone.')


if __name__ == '__main__':
    main()
