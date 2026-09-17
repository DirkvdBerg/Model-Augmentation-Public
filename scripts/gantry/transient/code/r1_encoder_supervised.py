"""R1: train the encoder ALONE against the exact state (D-197).

This is the experiment the supervisors described: "only train the encoder ... regression loss ...
past data, what is the difference between truth and encoder ... to verify the structure is good
enough". The baseline and the ANN are frozen and never simulated; there is no rollout and no
gradient through the model. The encoder is a static map from an I/O window to a state, so fitting
it is an ordinary supervised regression.

WHAT IT ANSWERS. If a DIRECTLY supervised encoder still cannot reproduce the true state, the
problem is structural (window length, capacity, the reconstructability init, or the state simply
not being reconstructable from that window) and no amount of end-to-end training fixes it. If it
can, the encoder structure is adequate and the free-run failure lives elsewhere.

SCOPE. The loss is on the six PHYSICAL channels only, in the encoder's own frame
`(x - x_mean)/std_x`. The `nx_ann` augmented channels have no ground truth (arbitrary gauge), so
they are left free and only monitored. Weighting every physical channel equally in that frame is
# HEURISTIC: it is the frame the rollout state lives in, so no channel is favoured by its unit.

Run (GPU, minutes):
  PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output \
      -n GraduationProject python -u scripts/gantry/transient/code/r1_encoder_supervised.py \
      --epochs 200
Sweep the window:  --na-nb 15 / 29 / 59      Contrast the target:  --target fd
"""
__project_origin__ = "added"

import json
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
from enc_common import CHANNELS, UNITS                           # noqa: E402


def stack(names, cfg, norm, dims, stride, target):
    """Concatenate windows over records. `target` selects the exact or the stored truth."""
    U, Y, X, meta = [], [], [], []
    for nm in names:
        w = ec.record_windows(nm, cfg, norm, dims, stride=stride)
        U.append(w['upast'])
        Y.append(w['ypast'])
        X.append(w['xt'] if target == 'exact' else w['xfd'])
        meta.append((nm, len(w['k_ix'])))
    return (np.concatenate(U), np.concatenate(Y), np.concatenate(X), meta)


def evaluate(encoder, U, Y, X, cfg, std_x, device):
    xh = ec.encoder_forward(encoder, U, Y, cfg, device=device)
    return ec.scores(xh[:, :cfg.nx_phys], X, std_x), xh


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--stride', type=int, default=4, help='training window stride')
    ap.add_argument('--val-stride', type=int, default=20)
    ap.add_argument('--na-nb', type=int, default=None, help='override the encoder window length')
    ap.add_argument('--nodes', type=int, default=None, help='override n_nodes_per_layer')
    ap.add_argument('--layers', type=int, default=None, help='override n_hidden_layers')
    ap.add_argument('--target', choices=['exact', 'fd'], default='exact')
    ap.add_argument('--weights', choices=['init', 'unit'], default='init',
                    help='per-channel loss weights. "unit" is a plain MSE in the encoder frame '
                         'and is DOMINATED by the channels the reconstructability init already '
                         'gets nearly exact (X and Y sit at a relative 2e-05 of their own std, '
                         'so their squared error is invisible next to Theta at 1e-01): the '
                         'optimiser then buys Theta by selling X, which answers nothing. '
                         '"init" (default) divides each channel by its own mean squared error '
                         'AT INITIALISATION, so every channel starts at a loss of 1 and the '
                         'question becomes the one that was asked -- can this structure improve '
                         'on the theory init everywhere at once')
    ap.add_argument('--part', choices=['all', 'linear', 'net'], default='all',
                    help='which encoder parameters may move: the W matrices, the correction '
                         'net, or both')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--tag', default=None)
    args = ap.parse_args()

    over = {}
    if args.na_nb is not None:
        over['na_nb_override'] = args.na_nb
    if args.nodes is not None:
        over['n_nodes_per_layer'] = args.nodes
    if args.layers is not None:
        over['n_hidden_layers'] = args.layers

    ec.ensure_dirs()
    cfg, data, norm, fit_sys, hp, dims = ec.build(over)
    na, nb, na_right, nb_right = dims
    std_x = norm.std_x.flatten()
    tag = args.tag or f'r1_{args.target}_na{na}_{args.part}'

    from gantry_dynamic.data import TRAIN_FILES, VAL_FILES
    train_names = [f[:-4] for f in TRAIN_FILES]
    val_names = [f[:-4] for f in VAL_FILES]

    print('R1 supervised encoder-only regression (D-197)')
    print(f'  dataset {cfg.mode}   encoder {cfg.encoder_init}   seed {cfg.seed}')
    print(f'  window  na=nb={na} ({na + na_right} samples = '
          f'{(na + na_right) * cfg.ts_new * 1e3:.2f} ms)   nx_ann={cfg.nx_ann}')
    print(f'  net     {cfg.n_hidden_layers} x {cfg.n_nodes_per_layer}   trainable part: {args.part}')
    print(f'  target  {args.target}   device {args.device}   lr {args.lr}  batch {args.batch}')

    t0 = time.time()
    Ut, Yt, Xt, meta_t = stack(train_names, cfg, norm, dims, args.stride, args.target)
    Uv, Yv, Xv, meta_v = stack(val_names, cfg, norm, dims, args.val_stride, args.target)
    print(f'  data    {len(Ut)} train windows, {len(Uv)} val windows '
          f'({time.time() - t0:.0f} s)')

    enc = fit_sys.encoder
    linear_names = ('Wb_psi_y', 'Wb_psi_u', 'Wa_psi_y', 'Wa_psi_u')
    params = []
    for n, p in enc.named_parameters():
        is_linear = n in linear_names
        keep = (args.part == 'all') or (args.part == 'linear' and is_linear) \
            or (args.part == 'net' and not is_linear)
        p.requires_grad_(bool(keep))
        if keep:
            params.append(p)
    print(f'  params  {sum(p.numel() for p in params)} trainable of '
          f'{sum(p.numel() for p in enc.parameters())}')

    dev = torch.device(args.device)
    sc0, _ = evaluate(enc, Uv, Yv, Xv, cfg, std_x, 'cpu')
    ec.print_scores('BEFORE training, validation windows', sc0)
    sc0t, _ = evaluate(enc, Ut[::7], Yt[::7], Xt[::7], cfg, std_x, 'cpu')

    if args.weights == 'init':
        w_ch = 1.0 / np.maximum((sc0t['nrms'] * sc0t['ref_std_phys'] / std_x) ** 2, 1e-30)
    else:
        w_ch = np.ones(cfg.nx_phys)
    w_ch = w_ch / w_ch.mean()
    print(f'\n  loss weights ({args.weights}): '
          f'{np.array2string(w_ch, precision=3, max_line_width=200)}')

    enc = enc.to(dev)
    Utt = torch.as_tensor(Ut, dtype=cfg.dtype_pt, device=dev)
    Ytt = torch.as_tensor(Yt, dtype=cfg.dtype_pt, device=dev)
    Xtt = torch.as_tensor(Xt, dtype=cfg.dtype_pt, device=dev)
    opt = torch.optim.Adam(params, lr=args.lr)
    wt = torch.as_tensor(w_ch, dtype=cfg.dtype_pt, device=dev)
    nphys = cfg.nx_phys
    N = len(Utt)
    g = torch.Generator(device='cpu').manual_seed(cfg.seed)

    # Epoch 0 IS a candidate. The reconstructability init is a theory-derived map already
    # accurate to a relative 2e-05 on X and Y, while Adam's step is scale-free (its first
    # step is ~lr on every parameter however small the gradient), so training can leave the
    # init worse than it found it. A run that never beats epoch 0 is a result, not a bug,
    # and it is only visible if epoch 0 is in the curve and in the checkpoint selection.
    mse0 = sc0['nrms'] ** 2 * (sc0['ref_std_phys'] / std_x) ** 2
    val_mse0 = float((w_ch * mse0).mean())
    hist = [(0, np.nan, val_mse0, float(sc0['r2'].min()), float(sc0['nrms'].mean()))]
    best = (val_mse0, {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}, 0)
    print(f'  epoch 0 (init) weighted validation MSE {val_mse0:.4e}')
    print(f'\n  {"epoch":>6}{"train MSE":>13}{"val MSE":>13}{"val worst R2":>14}'
          f'{"val mean NRMS":>15}{"s":>7}')
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        enc.train()
        perm = torch.randperm(N, generator=g).to(dev)
        tot = 0.0
        for i in range(0, N, args.batch):
            ix = perm[i:i + args.batch]
            opt.zero_grad(set_to_none=True)
            xh = enc(Utt[ix], Ytt[ix])
            loss = (wt * (xh[:, :nphys] - Xtt[ix]) ** 2).mean()
            loss.backward()
            opt.step()
            tot += float(loss) * len(ix)
        train_mse = tot / N
        enc.eval()
        sc, _ = evaluate(enc, Uv, Yv, Xv, cfg, std_x, dev)
        # Same weighting as the loss, so checkpoint selection and training agree.
        mse_ch = sc['nrms'] ** 2 * (sc['ref_std_phys'] / std_x) ** 2
        val_mse = float((w_ch * mse_ch).mean())
        hist.append((ep, train_mse, val_mse, float(sc['r2'].min()), float(sc['nrms'].mean())))
        if val_mse < best[0]:
            best = (val_mse, {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()},
                    ep)
        if ep <= 5 or ep % max(1, args.epochs // 20) == 0 or ep == args.epochs:
            print(f'  {ep:>6}{train_mse:>13.4e}{val_mse:>13.4e}{sc["r2"].min():>14.5f}'
                  f'{sc["nrms"].mean():>15.4e}{time.time() - t0:>7.0f}')

    enc.load_state_dict(best[1])
    enc = enc.to('cpu')
    print(f'\n  best validation MSE {best[0]:.4e} at epoch {best[2]}'
          + ('   <-- TRAINING NEVER BEAT THE INITIALISATION' if best[2] == 0 else ''))
    sc1, xh_v = evaluate(enc, Uv, Yv, Xv, cfg, std_x, 'cpu')
    ec.print_scores('AFTER training, validation windows', sc1)
    sc1t, _ = evaluate(enc, Ut[::7], Yt[::7], Xt[::7], cfg, std_x, 'cpu')
    ec.print_scores('AFTER training, training windows (every 7th)', sc1t)

    # What the achieved state error IS, next to the two references that bound it.
    print('\n  per-channel RMS state error [physical units]')
    print(f'    {"channel":<9}{"init":>12}{"trained":>12}{"stored FD":>12}   unit')
    w0 = ec.record_windows(val_names[0], cfg, norm, dims, stride=args.val_stride)
    fd_err = np.sqrt(((w0['xfd'] - w0['xt']) ** 2).mean(axis=0)) * std_x
    for i, ch in enumerate(CHANNELS):
        print(f'    {ch:<9}{sc0["rms_phys"][i]:>12.3e}{sc1["rms_phys"][i]:>12.3e}'
              f'{fd_err[i]:>12.3e}   {UNITS[i]}')

    h = np.array(hist)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogy(h[:, 0], h[:, 1], label='train')
    ax[0].semilogy(h[:, 0], h[:, 2], label='validation')
    ax[0].set_xlabel('epoch'); ax[0].set_ylabel('MSE (normalised state)')
    ax[0].legend(); ax[0].grid(alpha=0.3)
    x = np.arange(len(CHANNELS))
    ax[1].bar(x - 0.2, sc0['nrms'], 0.4, label='at init')
    ax[1].bar(x + 0.2, sc1['nrms'], 0.4, label='trained')
    ax[1].set_yscale('log'); ax[1].set_xticks(x); ax[1].set_xticklabels(CHANNELS)
    ax[1].set_ylabel('NRMS vs exact truth'); ax[1].legend(); ax[1].grid(alpha=0.3, axis='y')
    fig.suptitle(f'R1 encoder-only regression, target={args.target}, na=nb={na}, '
                 f'part={args.part}')
    fig.tight_layout()
    fpath = os.path.join(ec.FIGDIR, f'{tag}.png')
    fig.savefig(fpath, dpi=140)
    plt.close(fig)

    torch.save(enc.state_dict(), os.path.join(ec.OUTDIR, f'{tag}_encoder.pt'))
    np.savez_compressed(
        os.path.join(ec.OUTDIR, f'{tag}.npz'), hist=h,
        **{f'init|{k}': v for k, v in sc0.items()},
        **{f'trained|{k}': v for k, v in sc1.items()},
        **{f'trained_train|{k}': v for k, v in sc1t.items()},
        fd_err=fd_err, x_hat_val=xh_v, x_true_val=Xv)
    with open(os.path.join(ec.OUTDIR, f'{tag}.json'), 'w') as f:
        json.dump(dict(args=vars(args), na=na, nb=nb, mode=cfg.mode, seed=cfg.seed,
                       n_train=len(Ut), n_val=len(Uv), best_epoch=best[2],
                       best_val_mse=best[0],
                       r2_init=sc0['r2'].tolist(), r2_trained=sc1['r2'].tolist(),
                       nrms_init=sc0['nrms'].tolist(), nrms_trained=sc1['nrms'].tolist()),
                  f, indent=2)
    print(f'\nFigure: {fpath}')
    print(f'Saved:  {os.path.join(ec.OUTDIR, tag)}.npz / .json / _encoder.pt')


if __name__ == '__main__':
    main()
