"""Create presentation-ready figures from the saved downsampling study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_rate_sweep import (
    DATA_DIR, OUT_DIR, FS_MASTER, RATES, block_mean, controller_rollout_corate,
    deriv8, reconstruct_truth, y_op_for,
)


FIG_DIR = OUT_DIR / 'supervisor_figures'
COLORS = {'open': '#0072B2', 'controller': '#D55E00'}


def load_results():
    with open(OUT_DIR / 'open_loop.json', encoding='utf-8') as f:
        open_records = json.load(f)['records']
    with open(OUT_DIR / 'controller.json', encoding='utf-8') as f:
        controller_records = json.load(f)['records']
    return open_records, controller_records


def ratios(records, protocol, horizon, domain):
    values = {fs: [] for fs in RATES}
    for rec in records:
        for fs in RATES:
            arms = rec['rates'][str(fs)]
            if protocol != 'open_loop':
                arms = arms[protocol]
            oracle = arms['oracle'][horizon][domain]['aggregate_rms']
            fp = arms['fp'][horizon][domain]['aggregate_rms']
            if oracle is not None and fp is not None and fp > 0:
                values[fs].append(oracle / fp)
    return values


def save_figure(fig, stem):
    fig.savefig(FIG_DIR / f'{stem}.png', dpi=220, bbox_inches='tight')
    fig.savefig(FIG_DIR / f'{stem}.pdf', bbox_inches='tight')
    plt.close(fig)


def decision_figure(open_records, controller_records):
    open_values = ratios(open_records, 'open_loop', 'windows', 'band')
    ctrl_values = ratios(controller_records, 'corate', 'windows', 'time')
    rate_hz = np.array([1000, 2000, 4000])

    # RSS aggregation matches the study's main tables.
    def rss_ratio(records, protocol, domain, fs):
        eo, eb = [], []
        for rec in records:
            arms = rec['rates'][str(fs)]
            if protocol != 'open_loop':
                arms = arms[protocol]
            eo.append(arms['oracle']['windows'][domain]['aggregate_rms'])
            eb.append(arms['fp']['windows'][domain]['aggregate_rms'])
        return np.sqrt(np.mean(np.square(eo)) / np.mean(np.square(eb)))

    open_rss = np.array([rss_ratio(open_records, 'open_loop', 'band', fs) for fs in rate_hz])
    ctrl_rss = np.array([rss_ratio(controller_records, 'corate', 'time', fs) for fs in rate_hz])

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for values, label, color, marker, y_offset in (
        (open_rss, 'Open loop, 130–180 Hz', COLORS['open'], 'o', -18),
        (ctrl_rss, 'Closed loop, time domain', COLORS['controller'], 's', 8),
    ):
        ax.plot(rate_hz / 1000, 100 * values, marker=marker, lw=2.2,
                ms=7, color=color, label=label)
        for x, y in zip(rate_hz / 1000, 100 * values):
            ax.annotate(f'{y:.1f}%', (x, y), xytext=(0, y_offset),
                        textcoords='offset points', ha='center', color=color,
                        fontsize=9, fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim(0.35, 90.0)
    ax.set_xticks(rate_hz / 1000)
    ax.set_xlabel('Model and controller sampling rate [kHz]')
    ax.set_ylabel('Oracle downsampling error / MSD discrepancy [%]')
    ax.set_title('Downsampling error relative to the dynamics to be learned')
    ax.grid(True, which='both', alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, '01_rate_comparison')


def trajectory_figure(record='V2_aprbs_Ylow'):
    raw = loadmat(DATA_DIR / f'{record}.mat', squeeze_me=True)
    raw_u = np.asarray(raw['u_total'], dtype=float)
    raw_y = np.asarray(raw['y'], dtype=float)
    truth20, _, _ = reconstruct_truth(raw)
    y_op = y_op_for(record)
    runs = {}
    for fs in (2000, 1000):
        down = FS_MASTER // fs
        u = block_mean(raw_u, down)
        y = raw_y[::down][:len(u)]
        runs[fs] = (y, controller_rollout_corate(
            deriv8, 'oracle', truth20[0], u, y, fs, y_op))

    t_detail = 0.12
    t_error = 0.50
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.3))
    truth_stride = FS_MASTER // 4000
    n_truth = min(len(raw_y), int(t_detail * FS_MASTER))
    tt = np.arange(0, n_truth, truth_stride) / FS_MASTER
    trajectory_ax = axes[0]
    trajectory_ax.plot(tt, raw_y[:n_truth:truth_stride, 0], color='black', lw=1.8,
                       label='20 kHz truth')
    for fs, color, ls in ((2000, COLORS['open'], '-'),
                          (1000, COLORS['controller'], '--')):
        y, run = runs[fs]
        n = min(len(y), int(t_detail * fs))
        trajectory_ax.plot(np.arange(n) / fs, run['y'][:n, 0], color=color,
                           ls=ls, lw=1.5, label=f'{fs // 1000} kHz oracle')
    trajectory_ax.set_ylabel(r'$X_1$ position [m]')
    trajectory_ax.set_xlabel('Time [s]')
    trajectory_ax.set_title('Trajectory detail: onset of the 1 kHz instability')
    trajectory_ax.grid(True, alpha=0.22)
    trajectory_ax.legend(frameon=False, ncol=3)

    err_ax = axes[1]
    for fs, color, ls in ((2000, COLORS['open'], '-'),
                          (1000, COLORS['controller'], '--')):
        y, run = runs[fs]
        n = min(len(y), int(t_error * fs))
        err = np.linalg.norm(run['y'][:n] - y[:n], axis=1)
        err_ax.semilogy(np.arange(n) / fs, np.maximum(err, 1e-12),
                       color=color, ls=ls, lw=1.5, label=f'{fs // 1000} kHz')
    div = runs[1000][1]['divergence_seconds']
    if div is not None:
        err_ax.axvline(div, color=COLORS['controller'], lw=1.1, ls=':')
        err_ax.annotate(f'1 kHz divergence\n{div:.3f} s', xy=(div, 2e-4),
                        xytext=(-72, 18), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color=COLORS['controller']),
                        color=COLORS['controller'], fontsize=9)
    err_ax.set_ylabel(r'$\|y_{model}-y_{truth}\|_2$ [m]')
    err_ax.grid(True, which='both', alpha=0.22)
    err_ax.set_xlabel('Time [s]')
    err_ax.set_title('Tracking-error growth and divergence')
    fig.suptitle(f'Closed-loop oracle comparison: {record}', y=1.01)
    fig.tight_layout()
    save_figure(fig, '02_closed_loop_trajectory')


def distribution_figure(open_records, controller_records):
    open_values = ratios(open_records, 'open_loop', 'windows', 'band')
    ctrl_values = ratios(controller_records, 'corate', 'windows', 'time')
    rates = (1000, 2000, 4000)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), sharey=True)
    rng = np.random.default_rng(7)
    panels = (
        (axes[0], open_values, COLORS['open'], 'Open loop: 130–180 Hz', '22 records'),
        (axes[1], ctrl_values, COLORS['controller'], 'Closed loop: time domain', '5 held-out records'),
    )
    for ax, values, color, title, subtitle in panels:
        data = [100 * np.asarray(values[fs]) for fs in rates]
        bp = ax.boxplot(data, positions=np.arange(3), widths=0.52, patch_artist=True,
                        showfliers=False, medianprops=dict(color='black', lw=1.4))
        for box in bp['boxes']:
            box.set(facecolor=color, alpha=0.28, edgecolor=color)
        for i, vals in enumerate(data):
            jitter = rng.uniform(-0.10, 0.10, len(vals))
            ax.scatter(i + jitter, vals, s=23, color=color, alpha=0.72,
                       edgecolor='white', linewidth=0.35)
        ax.set_xticks(np.arange(3), ['1', '2', '4'])
        ax.set_xlabel('Sampling rate [kHz]')
        ax.set_title(title + '\n' + subtitle, fontsize=11)
        ax.set_yscale('log')
        ax.grid(True, which='both', axis='y', alpha=0.22)
    axes[0].set_ylabel('Oracle downsampling error / MSD discrepancy [%]')
    fig.suptitle('Variation across trajectories', y=1.02)
    fig.tight_layout()
    save_figure(fig, '03_across_record_distribution')


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    open_records, controller_records = load_results()
    decision_figure(open_records, controller_records)
    trajectory_figure()
    distribution_figure(open_records, controller_records)
    print(f'Wrote supervisor figures to {FIG_DIR}')


if __name__ == '__main__':
    main()
