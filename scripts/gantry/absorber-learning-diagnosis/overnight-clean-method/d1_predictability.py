"""D1 (diagnostic, no training): is the closed-loop residual predictable from the encoder window ACROSS records, per
record class? Decides whether P1's FAIL-SCREEN (calibration records TR-T3/T4) reflects the method or an unrepresentative
calibration class.

Model at its update-0 state (= the baseline; zero-init ANN). For every training record: 768 random windows, target
e = next M closed-loop residuals (M = encoder past window, as ps2.py), regressors = the encoder's flattened (u, y)
history plus a constant. Ridge regression (relative ridge 1e-6 of the trace) fitted on all records EXCEPT the one
evaluated (leave-one-record-out), unexplained fraction = mean squared error / mean squared target on that record.
Also the residual RMS per record (relative to the median record). Data only: no pole, band or truth quantity.

Usage: python d1_predictability.py
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
    X, E, R = [], [], []
    for r in range(len(recs)):
        uh, yh, uf, yf, cix, _ = sample(recs, fs, na_r, nb_r, 768, np.random.default_rng(50 + r), cfg.dtype_pt, allowed=[r])
        M = yh.shape[1]
        with torch.no_grad():
            x = fs.encoder(uh, yh)
            yp, _ = fs.simulate(x, uf[:, :M], yf[:, :M], ctrl_ix=cix, nf=M)
        e = (yf[:, :M] - yp).reshape(len(x), -1).numpy()
        X.append(np.hstack([uh.reshape(len(x), -1).numpy(), yh.reshape(len(x), -1).numpy(), np.ones((len(x), 1))]))
        E.append(e); R.append(np.sqrt((e ** 2).mean()))
    R = np.array(R); print('\n######## D1 leave-one-record-out linear predictability of the M-step residual (update-0 model)')
    for r in range(len(recs)):
        Xt = np.vstack([X[i] for i in range(len(recs)) if i != r]); Et = np.vstack([E[i] for i in range(len(recs)) if i != r])
        G = Xt.T @ Xt; G += 1e-6 * np.trace(G) / len(G) * np.eye(len(G))
        W = np.linalg.solve(G, Xt.T @ Et)
        u_out = ((X[r] @ W - E[r]) ** 2).mean() / (E[r] ** 2).mean()
        u_in = ((Xt @ W - Et) ** 2).mean() / (Et ** 2).mean()
        print('  %-6s residual rms %.2fx median | held-out unexplained %.3f (in-sample on the others %.3f)' % (
            names[r] if r < len(names) else r, R[r] / np.median(R), u_out, u_in), flush=True)


if __name__ == '__main__':
    main()
