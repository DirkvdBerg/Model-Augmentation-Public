"""R2: what does the encoder's state error COST in free run? (D-197)

A state-space error norm is not interpretable on its own: an error in a heavily damped or
near-unobservable direction decays, while a small error on a K = 0 axis never does. This script
converts the R0/R1 state error into the units the pipeline is actually judged in, by seeding the
FROZEN model with three different initial states and running it open loop:

  exact      the true state from `truth.py`              -> MODEL error alone
  encoder    the encoder's estimate, all 14 rows         -> model + encoder error
  mixed      exact physical rows, encoder augmented rows -> how much of it is the augmented gauge

A rollout, but no gradients and no training: the model is exactly as `build_model` returns it
(ANN zero-initialised unless a trained checkpoint is passed), and only the initial condition
changes between the three arms.

Open loop, with the recorded plant input replayed. The production training loss is CLOSED loop
(`cfg.closed_loop`), so these numbers are not comparable with `bestfit`; open loop is the clean
isolation, because a controller would partly correct the very initial-state error being measured.

Run:
  PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output \
      -n GraduationProject python -u scripts/gantry/transient/code/r2_freerun.py \
      --horizon 400 4000
"""
__project_origin__ = "added"

import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import enc_common as ec                                          # noqa: E402


def rollout(fit_sys, x0, u_seq, cfg):
    """Open-loop free run of the interconnect. x0 (B, nx), u_seq (B, H, nu) -> y (B, H, ny)."""
    ys = []
    x = torch.as_tensor(x0, dtype=cfg.dtype_pt)
    with torch.no_grad():
        for t in range(u_seq.shape[1]):
            u = torch.as_tensor(u_seq[:, t], dtype=cfg.dtype_pt)
            y, x = fit_sys.hfn(x, u)
            ys.append(y.numpy())
    return np.stack(ys, axis=1)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--records', nargs='*', default=None, help='default = the validation set')
    ap.add_argument('--horizon', nargs='*', type=int, default=[400, 4000],
                    help='rollout lengths in samples (400 = the production nf, 0.100 s)')
    ap.add_argument('--n-starts', type=int, default=20)
    ap.add_argument('--win-stride', type=int, default=50,
                    help='stride of the CANDIDATE start grid. Only --n-starts of these are '
                         'rolled out, so building every window of the record (stride 1) '
                         'allocates ~50x the memory for nothing, and on this machine that '
                         'is what made a concurrent run die with a MemoryError.')
    ap.add_argument('--encoder', default=None, help='trained encoder .pt from R1')
    ap.add_argument('--ckpt', default=None,
                    help='production checkpoint base path (the `<base>.pt` + `<base>.npz` '
                         'pair or a deepSI `_best.pth`), loaded through the pipeline via '
                         '`training.resolve_checkpoint`. '
                         'With a TRAINED ANN the model term '
                         'shrinks while the encoder term does not, which is the comparison '
                         'that decides whether the encoder is the dominating error. No '
                         'checkpoint of the production dataset exists on this machine; the '
                         'trained runs live on the cluster.')
    ap.add_argument('--tag', default='r2_freerun')
    args = ap.parse_args()

    ec.ensure_dirs()
    cfg, data, norm, fit_sys, hp, dims = ec.build()
    na, nb, na_right, nb_right = dims
    nphys, nxd = cfg.nx_phys, cfg.nx_phys + cfg.nx_ann
    fit_sys.eval()

    if args.ckpt:
        from gantry_dynamic.training import resolve_checkpoint
        hfn_sd, enc_sd, _optim, _meta = resolve_checkpoint(args.ckpt)
        fit_sys.hfn.load_state_dict(hfn_sd)
        fit_sys.encoder.load_state_dict(enc_sd)
        print(f'Loaded production checkpoint (hfn + encoder): {args.ckpt}')
    if args.encoder:
        fit_sys.encoder.load_state_dict(torch.load(args.encoder, map_location='cpu'))
        print(f'Loaded trained encoder: {args.encoder}')

    from gantry_dynamic.data import VAL_FILES, load_traj
    names = args.records or [f[:-4] for f in VAL_FILES]

    print('R2 free run from three initial states (D-197)')
    enc_lab = 'R1-trained' if args.encoder else ('checkpoint' if args.ckpt else 'at init')
    ann_lab = 'checkpoint' if args.ckpt else 'zero-init'
    print(f'  dataset {cfg.mode}   encoder {enc_lab}   ANN {ann_lab}')
    print(f'  open loop, recorded plant input replayed; horizons {args.horizon} samples '
          f'({[round(h * cfg.ts_new, 4) for h in args.horizon]} s)')

    payload, rows = {}, []
    for nm in names:
        w = ec.record_windows(nm, cfg, norm, dims, stride=args.win_stride)
        x_hat = ec.encoder_forward(fit_sys.encoder, w['upast'], w['ypast'], cfg)
        sd = load_traj(nm + '.mat', cfg)
        un = ec.fit_norm_transform(sd, norm, cfg)['u']
        yn = ec.fit_norm_transform(sd, norm, cfg)['y']
        N = len(yn)

        for H in args.horizon:
            valid = w['k_ix'][w['k_ix'] + H <= N]
            starts_ix = np.linspace(0, len(valid) - 1, args.n_starts).astype(int)
            ks = valid[starts_ix]
            pos = np.searchsorted(w['k_ix'], ks)

            u_seq = np.stack([un[k:k + H] for k in ks])
            y_ref = np.stack([yn[k:k + H] for k in ks])

            x_exact = np.zeros((len(ks), nxd), dtype=cfg.dtype_np)
            x_exact[:, :nphys] = w['xt'][pos]
            x_enc = x_hat[pos].astype(cfg.dtype_np)
            x_mix = x_enc.copy()
            x_mix[:, :nphys] = w['xt'][pos]

            t0 = time.time()
            res = {}
            for lab, x0 in (('exact', x_exact), ('encoder', x_enc), ('mixed', x_mix)):
                y_hat = rollout(fit_sys, x0, u_seq, cfg)
                e = (y_hat - y_ref) * norm.ystd                     # metres, stage frame
                res[lab] = dict(
                    rms=np.sqrt((e ** 2).mean(axis=(0, 1))),        # per output channel
                    # CHANGED: per time step AND per output channel. The channel mean hid
                    # the Y axis, which is the one the whole comparison is about.
                    rms_t=np.sqrt((e ** 2).mean(axis=0)),           # (H, ny)
                )
            sig = (y_ref * norm.ystd).std(axis=(0, 1))
            print(f'\n  {nm}  H={H} ({H * cfg.ts_new:.3f} s), {len(ks)} starts '
                  f'[{time.time() - t0:.0f} s]')
            print(f'    {"init state":<10}{"RMS y1 [m]":>13}{"RMS y2 [m]":>13}'
                  f'{"RMS y3 [m]":>13}{"NRMS mean":>12}')
            for lab in ('exact', 'encoder', 'mixed'):
                r = res[lab]['rms']
                print(f'    {lab:<10}{r[0]:>13.3e}{r[1]:>13.3e}{r[2]:>13.3e}'
                      f'{np.mean(r / sig):>12.4f}')
            ratio = np.mean(res['encoder']['rms'] / sig) / max(np.mean(res['exact']['rms'] / sig),
                                                              1e-30)
            print(f'    encoder-init cost: {ratio:.1f}x the free-run error of the exact state')
            rows.append((nm, H, np.mean(res['exact']['rms'] / sig),
                         np.mean(res['encoder']['rms'] / sig),
                         np.mean(res['mixed']['rms'] / sig)))
            for lab in res:
                payload[f'{nm}|{H}|{lab}|rms'] = res[lab]['rms']
                payload[f'{nm}|{H}|{lab}|rms_t'] = res[lab]['rms_t']
            payload[f'{nm}|{H}|sig'] = sig

    # Error growth with horizon, the thing that decides whether the encoder dominates.
    H = max(args.horizon)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    t = np.arange(H) * cfg.ts_new
    # `mixed` sits exactly on `exact` whenever the ANN is zero-initialised (the augmented rows then
    # have no path to the output), so it is drawn wide and pale UNDER the dashed exact curve
    # rather than hiding it.
    for lab, c, lw, ls in (('mixed', 'tab:orange', 3.0, '-'),
                           ('exact', 'k', 1.4, '--'),
                           ('encoder', 'tab:blue', 1.1, '-')):
        key = f'{names[0]}|{H}|{lab}|rms_t'
        if key in payload:
            ax.semilogy(t, payload[key][:, 2], color=c, lw=lw, ls=ls, alpha=0.9, label=f'x0 = {lab}')
    ax.set_xlabel('time into the free run [s]')
    ax.set_ylabel('Y output RMS error over starts [m]')
    ax.set_title(f'R2: free-run error by initial state, {names[0]}\n'
                 f'encoder {"trained" if args.encoder else "at init"}, open loop')
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fpath = os.path.join(ec.FIGDIR, f'{args.tag}.png')
    fig.savefig(fpath, dpi=140)
    plt.close(fig)

    np.savez_compressed(os.path.join(ec.OUTDIR, f'{args.tag}.npz'),
                        summary=np.array(rows, dtype=object), **payload)
    print(f'\nFigure: {fpath}')
    print(f'Saved:  {os.path.join(ec.OUTDIR, args.tag)}.npz')


if __name__ == '__main__':
    main()
