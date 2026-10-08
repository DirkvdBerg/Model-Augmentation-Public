"""Is training SPEED-limited? Adam's update per weight is about lr (|m / sqrt(v)| <= ~1), so after
N updates no weight can have moved more than about lr * N. The ANN's final layer is zero-initialised,
so its trained values ARE the displacements. Per checkpoint (best, last):
  - |W_out| per output row (6 physical, n_a added) against the budget lr * N_updates;
  - displacement of the hidden layers from the seeded init (build_model with the run's seed);
  - spectral radius of the added self-map A_aa at the encoder points (as h_learned_poles.py).
Reading: added-row weights near the budget and a radius still rising best -> last = the path is
followed at the maximum speed Adam allows (step-size budget). Weights far below the budget = the
landscape stops them (flat or opposing gradient).

Usage: python j_weight_budget.py [job]   (default 86894; needs <job>_last.pth for the trend)
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from e_thesis_checkpoints import JOBS, CK, cfg_for                        # noqa: E402  (sets sys.path)
from h_learned_poles import encoder_points, step_jac                      # noqa: E402

LR = 1e-5
UPD = {'best': 29900, 'last': 33400}      # run 42 log: best at 29900 updates, 200 epochs x 167


def main(job):
    from gantry_dynamic.data import load_datasets, compute_normalization
    from gantry_dynamic.model import build_model
    from gantry_dynamic.controller import build_closed_loop
    from gantry_dynamic.training import load_checkpoint
    from model_augmentation.fit_systems.blocks import Static_ANN_Block
    run, arm, na, seed = JOBS[job]
    cfg = cfg_for(arm, na, seed)
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    data = load_datasets(cfg); norm = compute_normalization(cfg, data)
    lin = lambda m: [l for l in next(b for b in m.hfn.connected_blocks if isinstance(b, Static_ANN_Block)).net.net
                     if isinstance(l, torch.nn.Linear)]                                  # noqa: E731
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    fs0 = build_model(cfg.hp, cfg, data, norm)
    init = [(l.weight.detach().clone(), l.bias.detach().clone()) for l in lin(fs0)]
    print('\n######## run %d (job %s): Adam budget lr * N = %.3f (best) / %.3f (last)' % (
        run, job, LR * UPD['best'], LR * UPD['last']))
    for kind in ('best', 'last'):
        path = os.path.join(CK, 'SSE_Interconnect_Composed_%s_%s.pth' % (job, kind))
        if not os.path.exists(path):
            continue
        np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        fs = build_model(cfg.hp, cfg, data, norm)
        fs.simulator = build_closed_loop(fs, norm, cfg, train_files=data.train_files,
                                         val_files=data.val_files, val_data=data.val_ckpt_data, verbose=False)
        load_checkpoint(fs, path, True, 'lbfgs')
        L = lin(fs)
        B = LR * UPD[kind]
        W = L[-1].weight.detach().numpy(); b = L[-1].bias.detach().numpy()
        print('  [%s, %d updates] output layer (init 0), per row: max|w| / budget, rms|w| / budget, |b| / budget' % (kind, UPD[kind]))
        for r in range(W.shape[0]):
            print('    row %d %-8s  %.2f  %.2f  %.2f' % (r, 'added' if r >= 6 else 'phys', np.abs(W[r]).max() / B,
                                                       np.sqrt(np.mean(W[r] ** 2)) / B, abs(b[r]) / B))
        for i, (l, (w0, b0)) in enumerate(zip(L[:-1], init[:-1])):
            dw = (l.weight.detach() - w0).abs().numpy()
            print('    hidden layer %d: |dW| / budget max %.2f, median %.2f, frac > 0.5 budget %.2f; x_a input columns max %.2f'
                  % (i, dw.max() / B, np.median(dw) / B, (dw > 0.5 * B).mean(),
                     dw[:, 6:6 + na].max() / B if i == 0 else float('nan')))
        x, u, _, _ = encoder_points(cfg, data, fs)
        J = step_jac(fs, x, u)
        rho = np.array([np.abs(np.linalg.eigvals(j[6:, 6:])).max() for j in J])
        print('    A_aa spectral radius over %d points: median %.4f, p90 %.4f' % (len(rho), np.median(rho), np.percentile(rho, 90)), flush=True)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '86894')
