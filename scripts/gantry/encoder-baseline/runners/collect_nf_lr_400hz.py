"""
collect_nf_lr_400hz.py
----------------------
Merge per-task partial JSONs written by diag_nf_lr_400hz.py --task_idx N
into the combined JSON + heatmap + curve + state plots.

Run after all SLURM array tasks have completed:
    conda run -n GraduationProject python scripts/gantry/encoder-baseline/runners/collect_nf_lr_400hz.py

Optionally restrict to a subset of tasks:
    ... collect_nf_lr_400hz.py --tasks 0 1 2 3 4
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', '..'))

OUT_DIR = os.path.join(
    PROJECT_ROOT, 'simulations', 'gantry_subnet', 'diagnostics',
)

# Must match diagnostic_nf_lr_400hz.py exactly
NF_VALUES  = [20, 40, 80, 160, 200]
LR_VALUES  = [5e-4, 1e-4, 5e-5]
ALL_COMBOS = [(nf, lr) for nf in NF_VALUES for lr in LR_VALUES]

STATE_NAMES = ['q1', 'q2', 'q3', 'dq1', 'dq2', 'dq3']
NX_PHYS     = 6
OBS_IDX     = [0, 2, 3, 5]
UNOBS_IDX   = [1, 4]

REF_20K = np.array(
    [9.181e-08, 4.614e-02, 1.859e-07, 2.543e-05, 7.003e+00, 1.079e-03],
    dtype=np.float32,
)
REF_400 = np.array(
    [5.840e-04, 1.336e+02, 4.586e-04, 5.169e-03, 3.344e+02, 4.633e-02],
    dtype=np.float32,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--tasks', type=int, nargs='*', default=None,
        help='Subset of task indices to collect (default: all found in OUT_DIR)',
    )
    parser.add_argument(
        '--out_dir', default=OUT_DIR,
        help=f'Directory with partial JSONs (default: {OUT_DIR})',
    )
    return parser.parse_args()


def load_partials(out_dir, task_filter=None):
    """Load all partial JSON files and return results dict keyed by (nf, lr)."""
    pattern = os.path.join(out_dir, 'diagnostic_nf_lr_400hz_task_*.json')
    files = sorted(glob.glob(pattern))

    if not files:
        print(f'No partial JSONs found in {out_dir}', flush=True)
        sys.exit(1)

    results = {}
    loaded_tasks = []

    for fpath in files:
        with open(fpath) as f:
            partial = json.load(f)

        task_idx = partial['task_idx']
        if task_filter is not None and task_idx not in task_filter:
            continue

        nf  = partial['nf']
        lr  = partial['lr']
        results[(nf, lr)] = partial['result']
        loaded_tasks.append(task_idx)
        print(f'  loaded task {task_idx:02d}: nf={nf}, lr={lr:.0e}  '
              f'verdict={partial["result"]["verdict"]}', flush=True)

    return results, loaded_tasks


def print_summary(results):
    print('\n' + '=' * 70, flush=True)
    print('SUMMARY', flush=True)
    print('=' * 70, flush=True)

    for category in ['BETTER', 'STABLE', 'WORSE']:
        matches = [(k, v) for k, v in results.items() if v['verdict'] == category]
        if not matches:
            continue
        print(f'\n{category} ({len(matches)}):', flush=True)
        for (nf, lr), v in sorted(matches, key=lambda x: x[1]['worst_obs_ratio']):
            sc = v['state_curves']
            obs_detail = '  '.join(
                f'{STATE_NAMES[i]}:{sc[STATE_NAMES[i]][0]:.2e}->{sc[STATE_NAMES[i]][-1]:.2e}'
                for i in OBS_IDX
            )
            print(f'  nf={nf:3d} lr={lr:.0e}  worst={v["worst_obs_ratio"]:.2f}x  '
                  f'{obs_detail}', flush=True)

    if all(v['verdict'] == 'WORSE' for v in results.values()):
        print('\nNO combination preserved state quality.', flush=True)
        print('Output MSE alone cannot constrain states. Consider regularization.',
              flush=True)


def save_combined_json(results, out_dir):
    FS_NEW = 400
    ts     = 1.0 / FS_NEW
    N_DIAG_EPOCHS = 10  # informational only

    json_out = {
        'config': {
            'fs_new': FS_NEW, 'ts': ts,
            'nf_values': NF_VALUES, 'lr_values': LR_VALUES,
            'n_epochs': N_DIAG_EPOCHS,
            'obs_states':   [STATE_NAMES[i] for i in OBS_IDX],
            'unobs_states': [STATE_NAMES[i] for i in UNOBS_IDX],
        },
        'references': {
            '20khz_native': {n: float(REF_20K[i]) for i, n in enumerate(STATE_NAMES)},
            '400hz_native': {n: float(REF_400[i]) for i, n in enumerate(STATE_NAMES)},
        },
        'results': {
            f'nf={nf}_lr={lr:.0e}': {
                'loss_curve':      v['loss_curve'],
                'state_curves':    v['state_curves'],
                'verdict':         v['verdict'],
                'worst_obs_ratio': v['worst_obs_ratio'],
                'early_stopped':   v['early_stopped'],
                'elapsed_s':       v['elapsed_s'],
            }
            for (nf, lr), v in results.items()
        },
    }
    path = os.path.join(out_dir, 'diagnostic_nf_lr_400hz.json')
    with open(path, 'w') as f:
        json.dump(json_out, f, indent=2)
    print(f'Saved combined JSON: {path}', flush=True)


def make_heatmap(results, out_dir):
    # Only plot nf/lr combos present in results
    nf_vals = sorted({nf for nf, _ in results})
    lr_vals = sorted({lr for _, lr in results}, reverse=True)

    grid_loss = np.full((len(nf_vals), len(lr_vals)), np.nan)
    grid_obs  = np.full((len(nf_vals), len(lr_vals)), np.nan)

    for i, nf in enumerate(nf_vals):
        for j, lr in enumerate(lr_vals):
            if (nf, lr) not in results:
                continue
            v  = results[(nf, lr)]
            lc = v['loss_curve']
            grid_loss[i, j] = lc[-1] / lc[0] if len(lc) >= 2 and lc[0] > 0 else 1.0
            sc0    = np.array([v['state_curves'][STATE_NAMES[k]][0]  for k in OBS_IDX])
            sc_fin = np.array([v['state_curves'][STATE_NAMES[k]][-1] for k in OBS_IDX])
            grid_obs[i, j]  = float(np.max(sc_fin / (sc0 + 1e-12)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, grid, title, vmax in zip(
        axes,
        [grid_loss, grid_obs],
        ['Loss ratio (end / start)',
         'Worst obs-state ratio (final / init)\nq1, q3, dq1, dq3'],
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
                    continue
                color = 'white' if val > vmax * 0.7 else 'black'
                ax.text(jj, ii, f'{val:.2f}', ha='center', va='center',
                        fontsize=9, color=color, fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('Baseline encoder diagnostic (400 Hz native init)', fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, 'diagnostic_nf_lr_400hz_heatmap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}', flush=True)


def make_loss_curves(results, out_dir):
    nf_vals    = sorted({nf for nf, _ in results})
    lr_colors  = ['tab:red', 'tab:orange', 'tab:blue']

    fig, axes = plt.subplots(len(nf_vals), 1,
                             figsize=(10, 3 * len(nf_vals)), sharex=True)
    if len(nf_vals) == 1:
        axes = [axes]

    for i, nf in enumerate(nf_vals):
        ax = axes[i]
        for j, lr in enumerate(LR_VALUES):
            if (nf, lr) not in results:
                continue
            curve = results[(nf, lr)]['loss_curve']
            ax.plot(range(1, len(curve) + 1), curve, 'o-',
                    color=lr_colors[j % len(lr_colors)], label=f'lr={lr:.0e}',
                    markersize=4, linewidth=1.2)
        ax.set_ylabel(f'nf={nf}\nVal loss')
        ax.set_yscale('log')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Epoch')
    fig.suptitle('Loss curves (400 Hz native init)', fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, 'diagnostic_nf_lr_400hz_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}', flush=True)


def make_state_plots(results, out_dir):
    nf_vals   = sorted({nf for nf, _ in results})
    nf_colors = plt.cm.tab10(np.linspace(0, 0.5, len(nf_vals)))
    lr_styles = ['-', '--', ':']

    fig, axes = plt.subplots(NX_PHYS, 1,
                             figsize=(12, 3 * NX_PHYS), sharex=True)

    for si, name in enumerate(STATE_NAMES):
        ax = axes[si]
        for ni, nf in enumerate(nf_vals):
            for li, lr in enumerate(LR_VALUES):
                if (nf, lr) not in results:
                    continue
                sc = results[(nf, lr)]['state_curves'][name]
                ax.plot(range(len(sc)), sc,
                        color=nf_colors[ni], linestyle=lr_styles[li % len(lr_styles)],
                        marker='o', markersize=2, linewidth=1.0,
                        label=f'nf={nf},lr={lr:.0e}')

        ax.axhline(y=float(REF_20K[si]), color='green', linestyle='--',
                   linewidth=1.2, alpha=0.8, label='20kHz ceiling')
        ax.axhline(y=float(REF_400[si]), color='orange', linestyle=':',
                   linewidth=1.2, alpha=0.8, label='400Hz floor')

        unobs_tag = '  [unobservable]' if si in UNOBS_IDX else ''
        ax.set_ylabel(f'{name}\nNRMS{unobs_tag}')
        ax.set_yscale('log')
        ax.legend(fontsize=5, loc='upper right', ncol=3)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Epoch (0 = init)')
    fig.suptitle('All state NRMS vs epoch (400 Hz native init, no regularization)',
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, 'diagnostic_nf_lr_400hz_states.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}', flush=True)


def main():
    args = parse_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print('=' * 70, flush=True)
    print('Collecting diagnostic_nf_lr_400hz partial results', flush=True)
    print('=' * 70, flush=True)
    print(f'  OUT_DIR = {out_dir}', flush=True)

    task_filter = set(args.tasks) if args.tasks is not None else None
    results, loaded = load_partials(out_dir, task_filter)

    n_expected = len(ALL_COMBOS) if task_filter is None else len(task_filter)
    print(f'\nLoaded {len(loaded)}/{n_expected} tasks.', flush=True)

    missing = [i for i, (nf, lr) in enumerate(ALL_COMBOS)
               if (nf, lr) not in results
               and (task_filter is None or i in task_filter)]
    if missing:
        print(f'WARNING: missing tasks {missing} — partial plots only.', flush=True)

    if not results:
        print('No results to collect. Exiting.', flush=True)
        sys.exit(1)

    print_summary(results)
    save_combined_json(results, out_dir)
    make_heatmap(results, out_dir)
    make_loss_curves(results, out_dir)
    make_state_plots(results, out_dir)

    print('\nDone.', flush=True)


if __name__ == '__main__':
    main()
