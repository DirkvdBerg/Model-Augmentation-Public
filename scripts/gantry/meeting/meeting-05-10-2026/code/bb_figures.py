"""Meeting 05-10-2026: figures of the best black-box model (option B, run e8).

Model: Jan's SUBNET on the closed-loop servo error (Thesis-writeup/Documentation/BLACKBOX-METHOD.md),
run e8 (noisy Coulomb-tanh data, seed 1, lr 3e-3, 9576 updates), best validated checkpoint.

Per validation record (3 kinds):
  <rec>_1_trajectory.png   measured stage positions X1, X2, Y from the data file (read only)
  <rec>_2_error.png        free-run prediction error y_hat - y of the model, per axis
And:
  e8_loss_training.png     training loss per validation interval
  e8_loss_validation.png   validation free-run RMS per validation interval

The model's RMS per record and axis is recomputed here and asserted equal to the run's
result.json before any figure is written.

Run:  python -u scripts/gantry/meeting/meeting-05-10-2026/code/bb_figures.py
"""
__project_origin__ = "added"

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', '..'))
BB = os.path.join(REPO, 'scripts', 'gantry', 'blackbox-cl-trainable')
RUN = os.path.join(BB, 'outputs', 'e', 'e8_noisy_s1')
FIG = os.path.join(HERE, '..', 'figures')
for p in (REPO, os.path.join(REPO, 'scripts', 'gantry'), os.path.join(BB, 'bbcl')):   # bbcl first
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                                     # noqa: E402
import torch                                                           # noqa: E402
import deepSI                                                          # noqa: E402
import matplotlib                                                      # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.ticker import ScalarFormatter                          # noqa: E402
from deepSI.fit_systems.encoders import SS_encoder_general_hf, default_output_net  # noqa: E402
from scipy.io import loadmat                                           # noqa: E402

from setup import cfg_for                                              # noqa: E402
from jan_model import JanEncoder, JanStateNet                          # noqa: E402
from gantry_dynamic.data import record_files, traj_dir, _record_at_fs_new  # noqa: E402

RECORDS = {                                     # the three validation kinds shown
    'VA-S1': 'standstill at Y = +0.10 m, multisine force',
    'VA-P1': 'S-curve moves, multisine force',
    'VA-T1': 'ILC-shape moves at Y = +0.04 m, no multisine',
}
AXES = ('X1', 'X2', 'Y')
COL = ('#2a78d6', '#eb6834', '#1baf7a')         # dataviz reference palette, slots 1 to 3 (validated)
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'

plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True,
                     'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.facecolor': 'white',
                     'axes.facecolor': 'white', 'savefig.dpi': 200})


def sci(ax):
    f = ScalarFormatter(useMathText=True)
    f.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(f)


def build(args):
    fkw = dict(n_hidden_layers=args['fh_depth'], n_nodes_per_layer=args['fh_width'])
    ekw = dict(n_hidden_layers=args['e_depth'], n_nodes_per_layer=args['e_width'])
    m = SS_encoder_general_hf(nx=args['nx'], na=args['na'], nb=args['na'],
                              hf_net_kwargs=dict(f_net=JanStateNet, f_net_kwargs=fkw,
                                                 h_net=default_output_net, h_net_kwargs=fkw),
                              e_net=JanEncoder, e_net_kwargs=ekw,
                              na_right=args['na_right'], nb_right=args['na_right'])
    ck = os.path.join(RUN, 'localappdata', 'deepSI', 'checkpoints')
    best = [f for f in os.listdir(ck) if f.endswith('_best.pth')]
    assert len(best) == 1, best
    m.__dict__ = torch.load(os.path.join(ck, best[0]), weights_only=False)   # deepSI's own format
    return m


def load(cfg, fn, args):
    """Exactly the signals of train_bb_clmap.py (--target e --in64 --dr)."""
    d = loadmat(os.path.join(traj_dir(cfg), fn), squeeze_me=True)
    _, y, r, f = _record_at_fs_new(d, cfg, ('r_sim', 'f_sim'))
    r, y = np.asarray(r, float), np.asarray(y, float)
    assert args['target'] == 'e' and args['in64'] and args['dr']
    u = np.hstack([r, np.asarray(f, float), np.diff(r, axis=0, prepend=r[:1])])
    return deepSI.System_data(u=u, y=y - r, dt=cfg.ts_new), r, y


def main():
    os.makedirs(FIG, exist_ok=True)
    res = json.load(open(os.path.join(RUN, 'result.json')))
    args, k0 = res['args'], res['k0']
    cfg = cfg_for(args['version'], args['seed'])
    _, va_f, _ = record_files(cfg)                                     # validation names only
    model = build(args)
    ts = cfg.ts_new

    for rec, kind in RECORDS.items():
        i = next(j for j, f in enumerate(va_f) if os.path.basename(f).startswith(rec))
        sd, r, y = load(cfg, va_f[i], args)
        with np.errstate(all='ignore'):
            yhat = r + np.asarray(model.apply_experiment(sd).y, float)
        err = (yhat - y)[k0:]
        rms = np.sqrt(np.mean(err ** 2, 0))
        assert np.allclose(rms, res['per_record'][i], rtol=1e-6), (rec, rms, res['per_record'][i])
        triv = np.sqrt(np.mean((r - y)[k0:] ** 2, 0))
        t = np.arange(len(y)) * ts

        # 1: measured trajectory (data)
        fig, axs = plt.subplots(3, 1, figsize=(9, 6.2), sharex=True)
        for a, ax in enumerate(axs):
            ax.plot(t, y[:, a], color=COL[a], lw=1.2)
            ax.set_ylabel('%s [m]' % AXES[a])
            ax.set_title('%s position' % AXES[a], loc='left', fontsize=10, color=INK)
        axs[-1].set_xlabel('time [s]')
        fig.suptitle('%s: measured trajectory (%s)' % (rec, kind), x=0.01, ha='left', color=INK)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, '%s_1_trajectory.png' % rec))
        plt.close(fig)

        # 2: black-box free-run prediction error
        fig, axs = plt.subplots(3, 1, figsize=(9, 6.2), sharex=True)
        for a, ax in enumerate(axs):
            ax.plot(t[k0:], err[:, a], color=COL[a], lw=0.7)
            sci(ax)
            ax.set_ylabel('error [m]')
            ax.set_title('%s: RMS %.2e m   (no model, y_hat = r: %.2e m)' % (AXES[a], rms[a], triv[a]),
                         loc='left', fontsize=10, color=INK)
        axs[-1].set_xlabel('time [s]')
        fig.suptitle('%s: black-box prediction error y_hat - y, free run over the record' % rec,
                     x=0.01, ha='left', color=INK)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, '%s_2_error.png' % rec))
        plt.close(fig)
        print('[fig] %s RMS %s m (matches result.json)' % (rec, np.array2string(rms, precision=3)),
              flush=True)

    # loss curves from the run log (every interval as printed during training). result.json is not
    # used here: deepSI reloads the best checkpoint at the end, which drops the intervals after it
    # and back-fills the update-0 training loss with the first interval's value.
    import re
    txt = open(os.path.join(BB, 'outputs', 'e', 'e8.log'), encoding='utf-8', errors='ignore').read()
    v0 = float(re.search(r'Initial Validation sim-RMS=\s*([0-9.eE+-]+)', txt).group(1))
    rows = [(int(n), float(tl), float(vl)) for n, tl, vl in
            re.findall(r'It\s+(\d+), sqrt loss\s+([0-9.eE+-]+), Val sim-RMS\s+([0-9.eE+-]+)', txt)]
    assert len(rows) == 12, len(rows)
    upd = np.array([0] + [n for n, _, _ in rows], float)
    lv = np.array([v0] + [v for _, _, v in rows])
    lt_upd = np.array([n for n, _, _ in rows], float)
    lt_rms = np.array([t for _, t, _ in rows])                        # deepSI prints sqrt(mean loss)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(lt_upd, lt_rms, color=COL[0], lw=2, marker='o', ms=5)
    ax.set_xlabel('Adam updates')
    ax.set_ylabel('training loss, RMS [normalised]')
    ax.set_title('e8 training loss (mean over each %d-update interval, plotted at its end)' % args['its_per_val'],
                 loc='left', fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'e8_loss_training.png'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(upd, lv, color=COL[0], lw=2, marker='o', ms=5)
    b = int(np.nanargmin(lv))
    ax.plot(upd[b], lv[b], 'o', ms=9, mfc='none', mec=INK, mew=1.5)
    ax.annotate('selected model: %.2e m' % lv[b], (upd[b], lv[b]), xytext=(-10, 18),
                textcoords='offset points', ha='center', color=INK)
    sci(ax)
    ax.set_xlabel('Adam updates')
    ax.set_ylabel('validation free-run RMS [m]')
    ax.set_title('e8 validation loss (6 validation records, all axes; circle = selected model)', loc='left', fontsize=10,
                 color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'e8_loss_validation.png'))
    plt.close(fig)
    print('[fig] loss curves written to %s' % os.path.abspath(FIG), flush=True)


if __name__ == '__main__':
    main()
