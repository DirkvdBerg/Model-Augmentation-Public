"""
plot_diag_nf_lr_400hz.py
------------------------
Parse a diag_nf_lr_400hz .log or .out file and reproduce the three
diagnostic figures that diag_nf_lr_400hz.py saves during training.

Figures produced (saved next to the input log):
  1. diag_nf_lr_400hz_heatmap_<run_id>.png   — val-loss ratio + worst obs-state RMS ratio
  2. diag_nf_lr_400hz_curves_<run_id>.png    — val and train loss curves per nf
  3. diag_nf_lr_400hz_states_<run_id>.png    — all 6 state NRMS vs epoch (direct eval)

Usage:
    conda run -n GraduationProject python \\
        scripts/gantry/encoder-baseline/figures/plot_diag_nf_lr_400hz.py \\
        simulations/gantry_subnet/encoder-baseline/diag_nf_lr_400hz_67920.log

    # or use the default (run 67920):
    conda run -n GraduationProject python \\
        scripts/gantry/encoder-baseline/figures/plot_diag_nf_lr_400hz.py
"""

import os
import sys
import re
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', '..'))

DEFAULT_LOG = os.path.join(
    PROJECT_ROOT, 'simulations', 'gantry_subnet', 'encoder-baseline',
    'diag_nf_lr_400hz_67920.log',
)

STATE_NAMES = ['q1', 'q2', 'q3', 'dq1', 'dq2', 'dq3']
OBS_IDX     = [0, 2, 3, 5]
OUT_NAMES   = ['y1', 'y2', 'y3']

# ---------------------------------------------------------------------------
# Log cleaning
# ---------------------------------------------------------------------------

_NOISE_SUBSTRINGS = [
    'FutureWarning',
    'torch.load',
    'self.__dict__',
    'Gymnasium',
    'gymnasium',
    'Gym has been',
    'Users of this version',
    'See the migration',
    'Please upgrade',
    'weights_only',
    'allowlisted',
    'arbitrary code',
    'unpickling',
    'pickle module',
    'import gym',
    'deprecated',
]


def _clean_lines(raw):
    result = []
    for line in raw.split('\n'):
        s = line.strip()
        if s in ('', '(0,)'):
            continue
        if any(ns in line for ns in _NOISE_SUBSTRINGS):
            continue
        result.append(line)
    return result


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def _safe_float(s):
    s = s.strip()
    if not s or s.lower() == 'nan':
        return float('nan')
    # Extract leading number, tolerating trailing annotations (e.g. "EARLY STOP ...")
    m = re.match(r'^[+-]?[\d\.]+(?:e[+\-]?\d+)?', s)
    if m:
        return float(m.group(0))
    return float('nan')


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_log(path):
    with open(path, encoding='utf-8', errors='replace') as f:
        raw = f.read()

    lines = _clean_lines(raw)

    # ---- header ----
    cfg = {
        'run_id':    None,
        'fs_new':    None,
        'nf_values': [],
        'lr_values': [],
        'n_epochs':  None,
    }
    for line in lines[:40]:
        m = re.match(r'\s+run_id\s+=\s+(\S+)', line)
        if m:
            cfg['run_id'] = m.group(1)

        m = re.match(r'\s+FS_NEW\s+=\s+(\d+)', line)
        if m:
            cfg['fs_new'] = int(m.group(1))

        m = re.match(r'\s+NF_VALUES\s+=\s+(\[.*\])', line)
        if m:
            cfg['nf_values'] = [int(x) for x in re.findall(r'\d+', m.group(1))]

        m = re.match(r'\s+LR_VALUES\s+=\s+(\[.*\])', line)
        if m:
            cfg['lr_values'] = [
                float(x)
                for x in re.findall(r'[\d\.]+(?:e[+\-]?\d+)?', m.group(1))
                if x
            ]

        m = re.match(r'\s+N_DIAG_EPOCHS\s+=\s+(\d+)', line)
        if m:
            cfg['n_epochs'] = int(m.group(1))

    # ---- reference NRMS ----
    ref_20k = {}
    ref_400 = {}
    for line in lines:
        if '20kHz native' in line and '|' in line:
            vals = [p.strip() for p in line.split('|')[1:] if p.strip()]
            if len(vals) >= 6:
                for i, v in enumerate(vals[:6]):
                    ref_20k[STATE_NAMES[i]] = _safe_float(v)
        if '400Hz native' in line and '|' in line:
            vals = [p.strip() for p in line.split('|')[1:] if p.strip()]
            if len(vals) >= 6:
                for i, v in enumerate(vals[:6]):
                    ref_400[STATE_NAMES[i]] = _safe_float(v)

    # ---- analytical baseline ----
    nrms_ana = {}
    rms_ana  = {}
    in_ana   = False
    for line in lines:
        if '--- Analytical baseline' in line:
            in_ana = True
            continue
        if in_ana:
            m = re.match(
                r'\s+(q[123]|dq[123])\s*\*?\s+'
                r'([\d\.e\+\-]+)\s+([\d\.e\+\-]+)\s+\S',
                line,
            )
            if m:
                nrms_ana[m.group(1)] = float(m.group(2))
                rms_ana [m.group(1)] = float(m.group(3))
            if line.strip().startswith('(*'):
                in_ana = False

    # ---- find combo block boundaries ----
    combo_starts = []
    for idx, line in enumerate(lines):
        m = re.match(r'^nf=(\d+),\s*lr=([\d\.e\+\-]+)\s*$', line.strip())
        if m:
            combo_starts.append((idx, int(m.group(1)), float(m.group(2))))

    results = {}
    for ci, (start, nf, lr) in enumerate(combo_starts):
        end   = combo_starts[ci + 1][0] if ci + 1 < len(combo_starts) else len(lines)
        block = lines[start:end]
        results[(nf, lr)] = _parse_combo_block(block)

    return cfg, ref_20k, ref_400, nrms_ana, rms_ana, results


# Compiled patterns for combo block parsing
_INIT_RE  = re.compile(r'^\s+init\s*\|\s*\|\s*\|(.+)')
_EPOCH_RE = re.compile(
    r'^\s+(\d+)\s*\|\s*(nan|[\d\.e\+\-]+)\s*\|\s*(nan|[\d\.e\+\-]+)\s*\|(.+)'
)
_VERDICT_RE = re.compile(
    r'\s+verdict:\s+(BETTER|STABLE|WORSE)\s+\(worst obs ratio\s+([\d\.]+)x\)'
)
_SUMMARY_RE = re.compile(
    r'\s+(q[123]|dq[123])\s*\*?\s+'
    r'([\d\.e\+\-]+)\s+([\d\.e\+\-]+)\s+[\d\.]+x\s+'
    r'([\d\.e\+\-]+)\s+([\d\.e\+\-]+)\s+([\d\.e\+\-]+)'
)
_OUTRMS_RE = re.compile(
    r'\s+(y\d):\s+init=([\d\.e\+\-]+)\s+best=([\d\.e\+\-]+)'
)


def _parse_combo_block(block):
    res = {
        'loss_val':        [],
        'loss_train':      [],
        'state_curves':    {n: [] for n in STATE_NAMES},
        'nrms_init_ae':    {},
        'nrms_best_ae':    {},
        'rms_init_ae':     {},
        'rms_best_ae':     {},
        'rms_ana_ae':      {},
        'output_rms_init': {},
        'output_rms_best': {},
        'verdict':         None,
        'worst_obs_ratio': None,
        'early_stopped':   False,
    }

    for line in block:
        # init row
        m = _INIT_RE.match(line)
        if m:
            vals = [_safe_float(v) for v in m.group(1).split('|')]
            if len(vals) >= 6:
                for i, name in enumerate(STATE_NAMES):
                    res['state_curves'][name].append(vals[i])
            continue

        # epoch row
        m = _EPOCH_RE.match(line)
        if m:
            res['loss_train'].append(_safe_float(m.group(2)))
            res['loss_val'].append(_safe_float(m.group(3)))
            vals = [_safe_float(v) for v in m.group(4).split('|')]
            if len(vals) >= 6:
                for i, name in enumerate(STATE_NAMES):
                    res['state_curves'][name].append(vals[i])
            if 'EARLY STOP' in line:
                res['early_stopped'] = True
            continue

        # verdict
        m = _VERDICT_RE.match(line)
        if m:
            res['verdict'] = m.group(1)
            res['worst_obs_ratio'] = float(m.group(2))
            continue

        # summary table
        m = _SUMMARY_RE.match(line)
        if m:
            n = m.group(1)
            res['nrms_init_ae'][n] = float(m.group(2))
            res['nrms_best_ae'][n] = float(m.group(3))
            res['rms_init_ae'][n]  = float(m.group(4))
            res['rms_best_ae'][n]  = float(m.group(5))
            res['rms_ana_ae'][n]   = float(m.group(6))
            continue

        # output RMS
        m = _OUTRMS_RE.match(line)
        if m:
            res['output_rms_init'][m.group(1)] = float(m.group(2))
            res['output_rms_best'][m.group(1)] = float(m.group(3))

    return res


# ---------------------------------------------------------------------------
# Plot 1 — heatmaps
# ---------------------------------------------------------------------------

def plot_heatmaps(cfg, results, out_path):
    nf_vals = cfg['nf_values']
    lr_vals = cfg['lr_values']
    run_id  = cfg['run_id']
    fs_new  = cfg['fs_new']
    n_ep    = cfg['n_epochs']

    grid_loss = np.full((len(nf_vals), len(lr_vals)), np.nan)
    grid_obs  = np.full((len(nf_vals), len(lr_vals)), np.nan)

    for i, nf in enumerate(nf_vals):
        for j, lr in enumerate(lr_vals):
            key = _find_key(results, nf, lr)
            if key is None:
                continue
            v  = results[key]
            lc = v['loss_val']
            if len(lc) >= 2 and lc[0] > 0 and not np.isnan(lc[0]):
                # Use first non-nan val and last non-nan val
                first = next((x for x in lc if not np.isnan(x)), None)
                last  = next((x for x in reversed(lc) if not np.isnan(x)), None)
                if first and last and first > 0:
                    grid_loss[i, j] = last / first
            rms_i = np.array([v['rms_init_ae'].get(STATE_NAMES[k], np.nan) for k in OBS_IDX])
            rms_b = np.array([v['rms_best_ae'].get(STATE_NAMES[k], np.nan) for k in OBS_IDX])
            with np.errstate(invalid='ignore'):
                grid_obs[i, j] = float(np.nanmax(rms_b / (rms_i + 1e-12)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, grid, title, vmax in zip(
        axes,
        [grid_loss, grid_obs],
        ['Val loss ratio (end / start)',
         'Worst obs-state RMS ratio (best / init)\nq1, q3, dq1, dq3  [apply_experiment]'],
        [2.0, 5.0],
    ):
        im = ax.imshow(grid, cmap='RdYlGn_r', aspect='auto', vmin=0.0, vmax=vmax)
        ax.set_xticks(range(len(lr_vals)))
        ax.set_xticklabels([f'{lr:.0e}' for lr in lr_vals])
        ax.set_yticks(range(len(nf_vals)))
        ax.set_yticklabels([str(nf) for nf in nf_vals])
        ax.set_xlabel('Learning rate')
        ax.set_ylabel('nf (rollout horizon)')
        ax.set_title(title)
        for ii in range(len(nf_vals)):
            for jj in range(len(lr_vals)):
                val = grid[ii, jj]
                if np.isnan(val):
                    ax.text(jj, ii, 'N/A', ha='center', va='center',
                            fontsize=9, color='gray')
                else:
                    color = 'white' if val > vmax * 0.7 else 'black'
                    ax.text(jj, ii, f'{val:.2f}', ha='center', va='center',
                            fontsize=9, color=color, fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(
        f'Baseline encoder diagnostic ({fs_new} Hz native init, {n_ep} epochs)'
        f'\nrun_id: {run_id}',
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ---------------------------------------------------------------------------
# Plot 2 — loss curves
# ---------------------------------------------------------------------------

def plot_loss_curves(cfg, results, out_path):
    nf_vals   = cfg['nf_values']
    lr_vals   = cfg['lr_values']
    run_id    = cfg['run_id']
    fs_new    = cfg['fs_new']
    lr_colors = ['tab:red', 'tab:orange', 'tab:blue']

    fig, axes = plt.subplots(len(nf_vals), 1,
                             figsize=(10, 3 * len(nf_vals)), sharex=True)
    if len(nf_vals) == 1:
        axes = [axes]

    for i, nf in enumerate(nf_vals):
        ax = axes[i]
        for j, lr in enumerate(lr_vals):
            key = _find_key(results, nf, lr)
            if key is None:
                continue
            color       = lr_colors[j % len(lr_colors)]
            val_curve   = results[key]['loss_val']
            train_curve = results[key]['loss_train']
            epochs = list(range(1, len(val_curve) + 1))

            # val: solid line with circle markers
            ax.plot(epochs, val_curve, 'o-',
                    color=color, label=f'val lr={lr:.0e}',
                    markersize=5, linewidth=1.8)

            # train: dotted line, same color — NaN gaps show naturally
            if train_curve and any(not np.isnan(l) for l in train_curve):
                train_plot = [l if not np.isnan(l) else np.nan for l in train_curve]
                ax.plot(epochs, train_plot, ':',
                        color=color, label=f'train lr={lr:.0e}',
                        linewidth=2.0)

        ax.set_ylabel(f'nf={nf}\nLoss', fontsize=12)
        ax.set_yscale('log')
        ax.legend(fontsize=9, loc='upper right', ncol=2)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

    axes[-1].set_xlabel('Epoch', fontsize=12)
    fig.suptitle(f'Loss curves ({fs_new} Hz, run {run_id})', fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ---------------------------------------------------------------------------
# Plot 3 — state NRMS vs epoch
# ---------------------------------------------------------------------------

def plot_state_nrms(cfg, ref_20k, ref_400, nrms_ana, results, out_path):
    nf_vals   = cfg['nf_values']
    lr_vals   = cfg['lr_values']
    run_id    = cfg['run_id']
    fs_new    = cfg['fs_new']
    n_ep      = cfg['n_epochs']

    nf_colors = plt.cm.tab10(np.linspace(0, 0.5, len(nf_vals)))
    lr_styles = ['-', '--', ':']

    fig, axes = plt.subplots(len(STATE_NAMES), 1,
                             figsize=(12, 3 * len(STATE_NAMES)), sharex=True)

    for si, name in enumerate(STATE_NAMES):
        ax = axes[si]

        for ni, nf in enumerate(nf_vals):
            for li, lr in enumerate(lr_vals):
                key = _find_key(results, nf, lr)
                if key is None:
                    continue
                sc = results[key]['state_curves'].get(name, [])
                if not sc:
                    continue
                ax.plot(range(len(sc)), sc,
                        color=nf_colors[ni],
                        linestyle=lr_styles[li % len(lr_styles)],
                        marker='o', markersize=2, linewidth=1.0,
                        label=f'nf={nf},lr={lr:.0e}')

        if nrms_ana.get(name) is not None:
            ax.axhline(y=nrms_ana[name], color='purple', linestyle='-.',
                       linewidth=1.5, alpha=0.8, label='analytical (P_inv+FD)')

        unobs_tag = '  [unobservable]' if si in (1, 4) else ''
        ax.set_ylabel(f'{name}\nNRMS{unobs_tag}', fontsize=12)
        ax.set_yscale('log')
        ax.legend(fontsize=8, loc='upper right', ncol=3)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

    axes[-1].set_xlabel('Epoch (0 = init)', fontsize=12)
    fig.suptitle(
        f'State NRMS vs epoch — direct eval (run {run_id})',
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_key(results, nf, lr):
    """Find the results dict key matching (nf, lr), tolerating float precision."""
    if (nf, lr) in results:
        return (nf, lr)
    # Fallback: match by nf and closest lr
    candidates = [(k, abs(k[1] - lr)) for k in results if k[0] == nf]
    if candidates:
        best = min(candidates, key=lambda x: x[1])
        if best[1] < lr * 0.01:  # within 1%
            return best[0]
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('log', nargs='?', default=DEFAULT_LOG,
                        help='Path to .log or .out file (default: run 67920)')
    args = parser.parse_args()

    log_path = os.path.abspath(args.log)
    if not os.path.isfile(log_path):
        print(f'ERROR: file not found: {log_path}', file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.dirname(log_path)
    print(f'Parsing: {log_path}')

    cfg, ref_20k, ref_400, nrms_ana, rms_ana, results = parse_log(log_path)

    # Restrict nf/lr lists to combos that actually completed (partial log safety)
    ran_nf = sorted({k[0] for k in results if results[k]['verdict'] is not None})
    ran_lr = sorted({k[1] for k in results if results[k]['verdict'] is not None})
    if ran_nf:
        cfg['nf_values'] = ran_nf
    if ran_lr:
        cfg['lr_values'] = ran_lr

    run_id = cfg.get('run_id') or 'unknown'
    print(f'  run_id     = {run_id}')
    print(f'  fs_new     = {cfg.get("fs_new")} Hz')
    print(f'  nf_values  = {cfg.get("nf_values")}')
    print(f'  lr_values  = {cfg.get("lr_values")}')
    print(f'  n_epochs   = {cfg.get("n_epochs")}')
    print(f'  combos parsed = {len(results)}')
    for key, v in sorted(results.items()):
        es = '  EARLY STOP' if v['early_stopped'] else ''
        print(f'    nf={key[0]:3d} lr={key[1]:.0e}  verdict={v["verdict"]}  '
              f'worst={v["worst_obs_ratio"]}{es}')

    plot_heatmaps(
        cfg, results,
        os.path.join(out_dir, f'diag_nf_lr_400hz_heatmap_{run_id}.png'),
    )
    plot_loss_curves(
        cfg, results,
        os.path.join(out_dir, f'diag_nf_lr_400hz_curves_{run_id}.png'),
    )
    plot_state_nrms(
        cfg, ref_20k, ref_400, nrms_ana, results,
        os.path.join(out_dir, f'diag_nf_lr_400hz_states_{run_id}.png'),
    )

    print('Done.')


if __name__ == '__main__':
    main()
