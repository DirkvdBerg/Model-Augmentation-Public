"""
Plot training curves from Step 1 server job logs.

Parses the 4 log files (nf=20/40 × direct/apply_experiment) and produces:
  - Plot 1: validation loss vs epoch (nf=20 vs nf=40)
  - Plot 2: state NRMS vs epoch - direct method (6-subplot grid)
  - Plot 3: state NRMS vs epoch - apply_experiment (6-subplot grid)

Output: simulations/gantry_subnet/encoder/step1_server_*.png
"""

import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).parents[3] / "simulations" / "server-output"
OUT_DIR = Path(__file__).parents[3] / "simulations" / "gantry_subnet" / "encoder"

LOGS = {
    ('nf20', 'direct'):           LOG_DIR / "step1_baseline_equals_system_67696.log",
    ('nf20', 'apply_experiment'): LOG_DIR / "step1_baseline_equals_system_67697.log",
    ('nf40', 'direct'):           LOG_DIR / "step1_baseline_equals_system_67698.log",
    ('nf40', 'apply_experiment'): LOG_DIR / "step1_baseline_equals_system_67699.log",
}

STATE_NAMES = ['q1', 'q2', 'q3', 'dq1', 'dq2', 'dq3']
STATE_UNITS = ['m',  'm',  'm',  'm/s', 'm/s', 'm/s']

# Analytical baseline NRMS (P_inv + FD, from any log - same across all)
ANALYTICAL_BASELINE = {
    'q1':  3.6896e-08,
    'q2':  2.7302e-04,
    'q3':  None,        # 0.0, skip on log scale
    'dq1': 1.4291e-02,
    'dq2': 2.5232e-01,
    'dq3': 4.7236e-03,
}

COLORS = {
    'nf20': '#1f77b4',   # blue
    'nf40': '#d62728',   # red
}
LS = {
    'direct':           '-',
    'apply_experiment': '--',
}

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
RE_EPOCH = re.compile(
    r'^\s+(\d+)\s*\|\s*([\d.e+\-]+)\s*\|'
    r'\s*([\d.e+\-]+)\s*\|\s*([\d.e+\-]+)\s*\|\s*([\d.e+\-]+)\s*\|'
    r'\s*([\d.e+\-]+)\s*\|\s*([\d.e+\-]+)\s*\|\s*([\d.e+\-]+)'
)
RE_INIT = re.compile(
    r'^\s*init\s*\|\s*\|'
    r'\s*([\d.e+\-]+)\s*\|\s*([\d.e+\-]+)\s*\|\s*([\d.e+\-]+)\s*\|'
    r'\s*([\d.e+\-]+)\s*\|\s*([\d.e+\-]+)\s*\|\s*([\d.e+\-]+)'
)


def parse_log(path):
    """Return dict with keys: init (array len 6), epochs (int array), val_loss, states (6 x N)."""
    init = None
    epochs, val_loss = [], []
    states = {s: [] for s in STATE_NAMES}

    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            m = RE_INIT.match(line)
            if m:
                init = np.array([float(m.group(i)) for i in range(1, 7)])
                continue
            m = RE_EPOCH.match(line)
            if m:
                epochs.append(int(m.group(1)))
                val_loss.append(float(m.group(2)))
                for i, s in enumerate(STATE_NAMES):
                    states[s].append(float(m.group(3 + i)))

    return {
        'init':      init,
        'epochs':    np.array(epochs),
        'val_loss':  np.array(val_loss),
        'states':    {s: np.array(v) for s, v in states.items()},
    }


def load_all():
    return {key: parse_log(path) for key, path in LOGS.items()}


# ---------------------------------------------------------------------------
# Plot 1: validation loss vs epoch
# ---------------------------------------------------------------------------
def plot_val_loss(data):
    fig, ax = plt.subplots(figsize=(7, 4))

    for nf_tag, color in COLORS.items():
        # Direct and apply_exp share identical val_loss - just use direct
        d = data[(nf_tag, 'direct')]
        label = 'nf=20 (50 ms)' if nf_tag == 'nf20' else 'nf=40 (100 ms)'
        ax.semilogy(d['epochs'], d['val_loss'], color=color, lw=2, label=label)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation loss (nf-step output RMS)')
    ax.set_title('Validation loss vs epoch')
    ax.legend()
    ax.grid(True, which='both', ls=':', alpha=0.5)
    fig.tight_layout()
    out = OUT_DIR / 'step1_server_val_loss.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2 & 3: state NRMS per eval method
# ---------------------------------------------------------------------------
def plot_states(data, eval_method, title_suffix, out_name):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()

    for ax_i, (state, unit) in enumerate(zip(STATE_NAMES, STATE_UNITS)):
        ax = axes[ax_i]
        baseline_val = ANALYTICAL_BASELINE[state]

        # Plot epoch curves for both nf settings
        for nf_tag, color in COLORS.items():
            d = data[(nf_tag, eval_method)]
            label = 'nf=20' if nf_tag == 'nf20' else 'nf=40'
            ax.semilogy(d['epochs'], d['states'][state],
                        color=color, lw=1.5, label=label)

        # Init reference - use nf=20 (same init for both)
        init_val = data[('nf20', eval_method)]['init'][STATE_NAMES.index(state)]
        ax.axhline(init_val, color='gray', ls='--', lw=1.2, label='init')

        # Analytical baseline (skip if zero)
        if baseline_val is not None and baseline_val > 0:
            ax.axhline(baseline_val, color='k', ls=':', lw=1.2, label='analytical')

        ax.set_title(f'{state}  [{unit}]')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('NRMS')
        ax.grid(True, which='both', ls=':', alpha=0.4)
        if ax_i == 0:
            ax.legend(fontsize=8)

    fig.suptitle(f'State NRMS vs epoch  ({title_suffix})', fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all()

    plot_val_loss(data)
    plot_states(data, 'direct',           'EVAL_METHOD=direct',           'step1_server_states_direct.png')
    plot_states(data, 'apply_experiment', 'EVAL_METHOD=apply_experiment', 'step1_server_states_apply_exp.png')

    print("Done.")
