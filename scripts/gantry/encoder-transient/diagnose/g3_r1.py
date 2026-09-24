"""G3 R1: train the G2 encoder ALONE against the exact state; how much structure headroom is left? (ET-005)

The vendored R1 recipe (D-197), started from the G2 encoder instead of the production one:
Adam lr 1e-4, batch 4096, per-channel weight 1 / MSE at init, epoch 0 a candidate, best val kept.
Arms: --arm planted (8 channels incl. the exact absorber pair) or --arm baseline (6 channels).
No rollout, no model gradient. Output: outputs/g3/r1_<arm>.json and _encoder.pt
"""
__project_origin__ = "added"

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import et_paths                                                  # noqa: E402
sys.path.insert(0, os.path.join(et_paths.ET, 'encoder'))
import enc_common as ec                                          # noqa: E402
import sched_encoder as se                                       # noqa: E402
import truth                                                     # noqa: E402
from g1_state import windows as state_windows                    # noqa: E402

OUT = os.path.join(et_paths.OUT, 'g3')
G2 = os.path.join(et_paths.OUT, 'g2', 'g2_attempt1_encoders.pt')
CH8 = ec.CHANNELS + ['da', 'vda']
UNITS8 = ec.UNITS + ['m', 'm/s']


def load_g2(cfg, norm, enc_prod, dims, sa, dtype=torch.float64):
    """Rebuild the two G2 encoders exactly as saved by g2_fix.py (state dicts loaded)."""
    sd = torch.load(G2, weights_only=False)
    na, dt, hp = dims[0], cfg.ts_new, cfg.hp
    sx6 = np.asarray(norm.x_all, float).std(0)
    su = np.asarray(norm.u_all, float).std(0)
    sy = np.asarray(norm.y_all, float).std(0)
    torch.manual_seed(cfg.seed)
    enc_U = se.ScheduledReconEncoder(
        se.make_parent('baseline', enc_prod, norm, dt, na, sx6, su, sy, sa, dtype, hp),
        sd['Cb'], na, 3, 3, norm.y0, norm.ystd).to(dtype)
    enc_PL = se.ScheduledReconEncoder(
        se.make_parent('planted', enc_prod, norm, dt, na, sx6, su, sy, sa, dtype, hp),
        sd['Cp'], na, 3, 3, norm.y0, norm.ystd).to(dtype)
    enc_U.load_state_dict(sd['enc_U'])
    enc_PL.load_state_dict(sd['enc_PL'])
    return enc_U, enc_PL


def stack(names, cfg, norm, na, stride, nch, sa):
    """Normalised encoder inputs and the exact target in the encoder's OUTPUT frame."""
    U, Y, X = [], [], []
    xm = np.asarray(norm.x_mean).ravel()
    sx = np.asarray(norm.std_x).ravel()
    for nm in names:
        yw, uw, x8, _, _ = state_windows(nm, cfg, na, stride=stride)
        U.append((uw - np.asarray(norm.u_mean).ravel()) / np.asarray(norm.std_u).ravel())
        Y.append((yw - np.asarray(norm.y0).ravel()) / np.asarray(norm.ystd).ravel())
        t = (x8[:, :6] - xm) / sx
        if nch == 8:
            t = np.concatenate([t, x8[:, 6:8] / sa], 1)
        X.append(t)
    return (np.ascontiguousarray(np.concatenate(U)), np.ascontiguousarray(np.concatenate(Y)),
            np.concatenate(X))


def rms_phys(enc, U, Y, X, nch, sx_out, dev, batch=16384):
    out = []
    with torch.no_grad():
        for i in range(0, len(U), batch):
            out.append(enc(torch.as_tensor(U[i:i + batch], device=dev),
                           torch.as_tensor(Y[i:i + batch], device=dev))[:, :nch].cpu().numpy())
    e = np.concatenate(out) - X
    return np.sqrt((e ** 2).mean(0)) * sx_out, (e ** 2).mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['planted', 'baseline'], required=True)
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--stride', type=int, default=4)
    ap.add_argument('--val-stride', type=int, default=20)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    t0 = time.time()
    et_paths.check()
    cfg, data, norm, fit_sys, hp, dims = ec.build()
    na = dims[0]
    from gantry_dynamic.data import TRAIN_FILES, VAL_FILES
    xa = np.concatenate([truth.exact_truth(f[:-4], cfg.mode)['x8'][:, [3, 7]] for f in TRAIN_FILES])
    sa = xa.std(0)
    enc_U, enc_PL = load_g2(cfg, norm, fit_sys.encoder.to(torch.float64), dims, sa)
    enc, nch = (enc_PL, 8) if args.arm == 'planted' else (enc_U, 6)
    sx_out = np.concatenate([np.asarray(norm.std_x).ravel(), sa])[:nch]
    tr = [f[:-4] for f in TRAIN_FILES]
    va = [f[:-4] for f in VAL_FILES]
    Ut, Yt, Xt = stack(tr, cfg, norm, na, args.stride, nch, sa)
    Uv, Yv, Xv = stack(va, cfg, norm, na, args.val_stride, nch, sa)
    print(f'R1 from the G2 encoder, arm {args.arm}, {nch} channels, na={na}, {len(Ut)} train / '
          f'{len(Uv)} val windows, device {args.device}, float64  [{time.time() - t0:.0f} s]')
    dev = torch.device(args.device)
    enc = enc.to(dev)
    params = [p for p in enc.parameters()]
    for p in params:
        p.requires_grad_(True)
    print(f'  params {sum(p.numel() for p in params)}')
    r0v, ms0v = rms_phys(enc, Uv, Yv, Xv, nch, sx_out, dev)
    r0t, ms0t = rms_phys(enc, Ut[::7], Yt[::7], Xt[::7], nch, sx_out, dev)
    w = 1.0 / np.maximum(ms0t, 1e-300)
    w = w / w.mean()
    wt = torch.as_tensor(w, device=dev)
    val0 = float((w * ms0v).mean())
    best = (val0, {k: v.detach().clone() for k, v in enc.state_dict().items()}, 0)
    Utt, Ytt, Xtt = (torch.as_tensor(a, device=dev) for a in (Ut, Yt, Xt))
    opt = torch.optim.Adam(params, lr=args.lr)
    g = torch.Generator(device='cpu').manual_seed(cfg.seed)
    N = len(Utt)
    hist = [(0, float('nan'), val0)]
    print(f'  epoch 0 weighted val MSE {val0:.4e}')
    for ep in range(1, args.epochs + 1):
        enc.train()
        perm = torch.randperm(N, generator=g).to(dev)
        tot = 0.0
        for i in range(0, N, args.batch):
            ix = perm[i:i + args.batch]
            opt.zero_grad(set_to_none=True)
            loss = (wt * (enc(Utt[ix], Ytt[ix])[:, :nch] - Xtt[ix]) ** 2).mean()
            loss.backward()
            opt.step()
            tot += float(loss) * len(ix)
        enc.eval()
        _, msv = rms_phys(enc, Uv, Yv, Xv, nch, sx_out, dev)
        vm = float((w * msv).mean())
        hist.append((ep, tot / N, vm))
        if vm < best[0]:
            best = (vm, {k: v.detach().clone() for k, v in enc.state_dict().items()}, ep)
        if ep <= 3 or ep % 50 == 0 or ep == args.epochs:
            print(f'  {ep:>5} train {tot / N:.4e}  val {vm:.4e}  best {best[0]:.4e}@{best[2]}  '
                  f'[{time.time() - t0:.0f} s]')
    enc.load_state_dict(best[1])
    r1v, _ = rms_phys(enc, Uv, Yv, Xv, nch, sx_out, dev)
    print(f'\n  best epoch {best[2]}' + ('  <-- training never beat the G2 initialisation' if best[2] == 0 else ''))
    print(f'  {"ch":<7}{"G2 init":>12}{"trained":>12}{"headroom":>10}   unit   (validation V1-V4)')
    for i in range(nch):
        print(f'  {CH8[i]:<7}{r0v[i]:>12.3e}{r1v[i]:>12.3e}{r0v[i] / r1v[i]:>10.2f}   {UNITS8[i]}')
    os.makedirs(OUT, exist_ok=True)
    torch.save(enc.cpu().state_dict(), os.path.join(OUT, f'r1_{args.arm}_encoder.pt'))
    with open(os.path.join(OUT, f'r1_{args.arm}.json'), 'w') as f:
        json.dump(dict(args=vars(args), best_epoch=best[2], init_val_rms=r0v.tolist(),
                       trained_val_rms=r1v.tolist(), headroom=(r0v / r1v).tolist(),
                       hist=hist, n_train=len(Ut), n_val=len(Uv)), f, indent=1)
    print(f'  saved outputs/g3/r1_{args.arm}.json  [{time.time() - t0:.0f} s]')


if __name__ == '__main__':
    main()
