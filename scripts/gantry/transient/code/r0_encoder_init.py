"""R0: what does the encoder know BEFORE any training? (D-197)

The supervisors' first question, "encoder init compare to actual states": run the production
encoder at its `linear_map` initialisation over every window of a record and compare its output
to the state that is actually there. No training, no rollout, no gradients.

Three numbers per channel, all on the same windows:
    encoder vs EXACT      what the encoder actually gets wrong,
    encoder vs stored     what the existing diagnostic reports (`x_logical`, D-053), and
    stored vs EXACT       how much of that is the TARGET being wrong, not the encoder.
The third column is the reason this arm exists: the record's velocities are `gradient()`
finite differences and are provably wrong at the endpoints (`truth.py`).

Also reported: the same comparison against the truth at k-1 (a one-sample lag shows up as a
BETTER score there, `diagnostics.py::state_recovery_diagnostic`), and the liveness of the
`nx_ann` augmented channels, which have no ground truth and are never scored.

Run:
  PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output \
      -n GraduationProject python -u scripts/gantry/transient/code/r0_encoder_init.py
"""
__project_origin__ = "added"

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import enc_common as ec                                          # noqa: E402
from enc_common import CHANNELS, UNITS                           # noqa: E402


def analyse(cfg, norm, fit_sys, dims, names, stride, k_lo=None):
    std_x = norm.std_x.flatten()
    fit_sys.eval()
    out = {}
    for nm in names:
        w = ec.record_windows(nm, cfg, norm, dims, stride=stride, k_lo=k_lo, gate=True)
        x_hat = ec.encoder_forward(fit_sys.encoder, w['upast'], w['ypast'], cfg)
        xp = x_hat[:, :cfg.nx_phys]
        out[nm] = dict(
            window=w, x_hat=x_hat,
            enc_vs_exact=ec.scores(xp, w['xt'], std_x),
            enc_vs_stored=ec.scores(xp, w['xfd'], std_x),
            stored_vs_exact=ec.scores(w['xfd'], w['xt'], std_x),
            enc_vs_exact_lag1=ec.scores(xp[1:], w['xt'][:-1], std_x) if stride == 1 else None,
            aug_std=x_hat[:, cfg.nx_phys:].std(axis=0),
            aug_nan=int(np.isnan(x_hat).sum()),
            aug_true_std=w['x_aug_true'].std(axis=0),
        )
    return out


def figure(res, name, cfg, norm, path):
    """Truth vs encoder estimate on a 0.25 s slice, one panel per physical channel."""
    w, x_hat = res[name]['window'], res[name]['x_hat']
    std_x, x_mean = norm.std_x.flatten(), norm.x_mean.flatten()
    n = min(1000, len(w['k_ix']))
    sl = slice(len(w['k_ix']) // 2, len(w['k_ix']) // 2 + n)
    t = w['k_ix'][sl] * cfg.ts_new
    fig, ax = plt.subplots(3, 2, figsize=(11, 7.5), sharex=True)
    for i, a in enumerate(ax.T.flatten()):
        true = w['xt'][sl, i] * std_x[i] + x_mean[i]
        stored = w['xfd'][sl, i] * std_x[i] + x_mean[i]
        est = x_hat[sl, i] * std_x[i] + x_mean[i]
        a.plot(t, true, 'k', lw=1.1, label='exact truth')
        a.plot(t, stored, color='tab:orange', lw=0.8, ls=':', label='stored x_logical')
        a.plot(t, est, color='tab:blue', lw=0.9, label='encoder at init')
        a.set_ylabel(f'{CHANNELS[i]} [{UNITS[i]}]')
        a.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=8, ncol=3)
    for a in ax[-1]:
        a.set_xlabel('time [s]')
    fig.suptitle(f'R0: encoder state estimate at initialisation, {name}  '
                 f'({cfg.mode}, na=nb={cfg.na_nb})')
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--records', nargs='*', default=None, help='default = the validation set')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--tag', default='r0_init')
    args = ap.parse_args()

    ec.ensure_dirs()
    cfg, data, norm, fit_sys, hp, dims = ec.build()
    from gantry_dynamic.data import VAL_FILES
    names = args.records or [f[:-4] for f in VAL_FILES]
    na, nb, na_right, nb_right = dims

    print(f'R0 encoder-initialisation check (D-197)')
    print(f'  dataset   {cfg.mode}')
    print(f'  encoder   {cfg.encoder_init}, na=nb={na} (window {na + na_right} samples = '
          f'{(na + na_right) * cfg.ts_new * 1e3:.2f} ms), nx_ann={cfg.nx_ann}, seed={cfg.seed}')
    print(f'  records   {", ".join(names)}   stride {args.stride}')
    print('\n  replay gate (exact truth vs the record\'s own positions):')

    res = analyse(cfg, norm, fit_sys, dims, names, args.stride)

    for nm in names:
        r = res[nm]
        print(f'\n{"=" * 78}\n{nm}   {len(r["window"]["k_ix"])} windows')
        ec.print_scores('encoder vs EXACT truth', r['enc_vs_exact'])
        ec.print_scores('encoder vs STORED x_logical (what D-053 reports)', r['enc_vs_stored'])
        ec.print_scores('STORED x_logical vs EXACT truth (the target error alone)',
                        r['stored_vs_exact'])
        if r['enc_vs_exact_lag1'] is not None:
            lag = r['enc_vs_exact_lag1']['r2']
            now = r['enc_vs_exact']['r2']
            print(f'\n    lag check: R2 against truth at k-1 minus R2 at k = '
                  f'{np.array2string(lag - now, precision=4)}')
            print('      positive on every channel would mean the encoder is aligned to k-1')
        print(f'\n    augmented channels (gauge, never scored): std '
              f'{np.array2string(r["aug_std"], precision=3)}  NaN {r["aug_nan"]}')
        print(f'      true absorber [delta_a, vdelta_a] std '
              f'{np.array2string(r["aug_true_std"], precision=3)}')

    fig_path = os.path.join(ec.FIGDIR, f'{args.tag}_{names[0]}.png')
    figure(res, names[0], cfg, norm, fig_path)
    print(f'\nFigure: {fig_path}')

    npz = os.path.join(ec.OUTDIR, f'{args.tag}.npz')
    payload = {'records': np.array(names)}
    for nm in names:
        for key in ('enc_vs_exact', 'enc_vs_stored', 'stored_vs_exact'):
            for k, v in res[nm][key].items():
                payload[f'{nm}|{key}|{k}'] = np.asarray(v)
        payload[f'{nm}|k_ix'] = res[nm]['window']['k_ix']
        payload[f'{nm}|x_hat'] = res[nm]['x_hat']
        payload[f'{nm}|xt'] = res[nm]['window']['xt']
        payload[f'{nm}|xfd'] = res[nm]['window']['xfd']
    np.savez_compressed(npz, **payload)
    print(f'Saved:   {npz}')


if __name__ == '__main__':
    main()
