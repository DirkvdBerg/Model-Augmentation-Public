"""D2 (diagnostic, no training): rank-2 reduced-rank regression of the update-0 residual (as D1) fitted on the 16 fit
records (all but TR-T3, TR-T4), scored per record: can ANY linear 2-d predictive state transfer to the T class?
Usage: python d2_rank2.py
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import n_phase1 as n1                                                          # noqa: E402
import u_probe as up                                                           # noqa: E402
from ps2 import sample                                                         # noqa: E402


def main():
    os.environ['U_INIT'] = 'zero'
    cfg, data, norm, fs, ann, others = up.setup()
    recs, na_r, nb_r = n1.windows(cfg, data, fs)
    names = [os.path.basename(str(f)).split('_')[0] for f in data.train_files]
    X, E = [], []
    for r in range(len(recs)):
        uh, yh, uf, yf, cix, _ = sample(recs, fs, na_r, nb_r, 768, np.random.default_rng(50 + r), cfg.dtype_pt, allowed=[r])
        M = yh.shape[1]
        with torch.no_grad():
            x = fs.encoder(uh, yh)
            yp, _ = fs.simulate(x, uf[:, :M], yf[:, :M], ctrl_ix=cix, nf=M)
        e = (yf[:, :M] - yp).reshape(len(x), -1).numpy()
        e = e / np.sqrt((e ** 2).mean())            # per-window-batch unit RMS, as the head's target
        X.append(np.hstack([uh.reshape(len(x), -1).numpy(), yh.reshape(len(x), -1).numpy(), np.ones((len(x), 1))]))
        E.append(e)
    fit = list(range(len(recs) - 2))
    Xt = np.vstack([X[i] for i in fit]); Et = np.vstack([E[i] for i in fit])
    G = Xt.T @ Xt; G += 1e-6 * np.trace(G) / len(G) * np.eye(len(G))
    W = np.linalg.solve(G, Xt.T @ Et)
    print('\n######## D2 rank-r reduced-rank regression fitted on the 16 fit records')
    for rk in (2, 4):
        _, _, Vt = np.linalg.svd(Xt @ W, full_matrices=False)
        P = Vt[:rk].T @ Vt[:rk]                     # projection onto the top-rk predicted-target directions
        Wr = W @ P
        row = []
        for r in range(len(recs)):
            row.append('%s %.3f' % (names[r], ((X[r] @ Wr - E[r]) ** 2).mean() / (E[r] ** 2).mean()))
        print('  rank %d: %s' % (rk, ' | '.join(row)), flush=True)


if __name__ == '__main__':
    main()
