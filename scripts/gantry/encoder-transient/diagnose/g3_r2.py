"""G3 R2: open-loop free run from different initial states, untrained and planted model (ET-005).

As D-197 R2: recorded plant input replayed, no controller, no gradients, 20 starts per record.
Untrained model: x0 in {exact, P0, G2 (baseline source)}; planted model: {exact, P0 W^a random,
P0 W^a zero, G2 (planted source)}. Output: outputs/g3/r2.json
"""
__project_origin__ = "added"

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import et_paths                                                  # noqa: E402
import cl_common as cc                                           # noqa: E402
import truth                                                     # noqa: E402
import recon                                                     # noqa: E402
from g1_transient import setup, prod_encoder                     # noqa: E402
from g3_r1 import load_g2                                        # noqa: E402

HORIZONS = [400, 4000]
N_STARTS = 20
OUT = os.path.join(et_paths.OUT, 'g3')


def rollout_ol(hfn, x0, un):
    """Open loop. x0 (B, nx) normalised, un (B, H, nu) normalised -> y (B, H, ny) normalised."""
    x = torch.as_tensor(x0, dtype=cc.DT)
    ys = []
    with torch.no_grad():
        for t in range(un.shape[1]):
            y, x = hfn(x, un[:, t])
            ys.append(y)
    return torch.stack(ys, 1).numpy()


def enc_out(enc, yw, uw, norm):
    un = (uw - np.asarray(norm.u_mean).ravel()) / np.asarray(norm.std_u).ravel()
    yn = (yw - np.asarray(norm.y0).ravel()) / np.asarray(norm.ystd).ravel()
    with torch.no_grad():
        return enc(torch.as_tensor(np.ascontiguousarray(un), dtype=cc.DT),
                   torch.as_tensor(np.ascontiguousarray(yn), dtype=cc.DT)).numpy()


def main():
    t0 = time.time()
    cfg, norm, fit_sys, hfn, enc_prod, bank, rowmap, planted, sa, dims = setup()
    na = dims[0]
    enc_U, enc_PL = load_g2(cfg, norm, enc_prod, dims, sa)
    enc_U.eval(); enc_PL.eval()
    from gantry_dynamic.data import load_traj
    ystd = np.asarray(norm.ystd).ravel()
    arms = ['U|exact', 'U|P0', 'U|G2', 'PL|exact', 'PL|P0|random', 'PL|P0|zero', 'PL|G2']
    sq = {(a, H): np.zeros(3) for a in arms for H in HORIZONS}
    cnt = {H: 0 for H in HORIZONS}
    per = {}
    for nm in cc.VAL:
        sd = load_traj(nm + '.mat', cfg)
        u = np.ascontiguousarray(sd.u, dtype=float)
        y = np.ascontiguousarray(sd.y, dtype=float)
        x8 = truth.exact_truth(nm, cfg.mode)['x8'][:, recon.T2M]
        N = len(y)
        for H in HORIZONS:
            ks = np.linspace(na + 1, N - H, N_STARTS).astype(int)
            yw = np.stack([y[k - na:k + 1] for k in ks])
            uw = np.stack([u[k - na:k + 1] for k in ks])
            un = torch.as_tensor(np.ascontiguousarray(np.stack([
                (u[k:k + H] - np.asarray(norm.u_mean).ravel()) / np.asarray(norm.std_u).ravel()
                for k in ks])), dtype=cc.DT)
            yref = np.stack([y[k:k + H] for k in ks])
            w = dict(yw=yw, uw=uw)
            xe = prod_encoder(enc_prod, w, norm)
            xg_u = enc_out(enc_U, yw, uw, norm)
            xg_p = enc_out(enc_PL, yw, uw, norm)
            xu_ex = np.zeros_like(xe); xu_ex[:, :6] = cc.x0_norm(x8[ks, :6], norm)
            x0s = {
                'U|exact': (hfn, xu_ex), 'U|P0': (hfn, xe), 'U|G2': (hfn, xg_u),
                'PL|exact': (planted, (x8[ks] - planted.xm.numpy()) / planted.sx.numpy()),
                'PL|P0|random': (planted, np.concatenate([xe[:, :6], xe[:, 6:8]], 1)),
                'PL|P0|zero': (planted, np.concatenate([xe[:, :6], np.zeros((len(ks), 2))], 1)),
                'PL|G2': (planted, xg_p),
            }
            for a, (m, x0) in x0s.items():
                yhat = rollout_ol(m, x0, un) * ystd + np.asarray(norm.y0).ravel()
                e2 = ((yhat - yref) ** 2).sum(axis=(0, 1))
                sq[(a, H)] += e2
                per[f'{nm}|{H}|{a}'] = np.sqrt(e2 / (len(ks) * H)).tolist()
            cnt[H] += len(ks) * H
            print(f'  {nm} H={H} done [{time.time() - t0:.0f} s]')
    res = dict(per_record=per, pooled={})
    print(f'\n  pooled open-loop free-run RMS [m], V1-V4, {N_STARTS} starts per record')
    print(f'  {"arm":<14}' + ''.join(f'{"H=" + str(H) + " X1/X2/Y":>40}' for H in HORIZONS))
    for a in arms:
        row = []
        for H in HORIZONS:
            r = np.sqrt(sq[(a, H)] / cnt[H])
            res['pooled'][f'{a}|{H}'] = r.tolist()
            row.append(' '.join(f'{v:.3e}' for v in r))
        print(f'  {a:<14}' + ''.join(f'{s:>40}' for s in row))
    ok = all(np.all(np.asarray(res['pooled'][f'PL|G2|{H}'])
                    <= np.asarray(res['pooled'][f'PL|P0|{lat}|{H}']))
             for H in HORIZONS for lat in ('random', 'zero'))
    res['pass'] = bool(ok)
    print(f'\n  G3 R2 criterion (planted: G2 <= P0 random and zero, every channel, both H): {ok}')
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'r2.json'), 'w') as f:
        json.dump(res, f, indent=1)
    print(f'  saved outputs/g3/r2.json  [{time.time() - t0:.0f} s]')


if __name__ == '__main__':
    main()
