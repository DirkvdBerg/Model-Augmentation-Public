"""Test 2: does the WINDOWED training objective prefer the damped learned shape over the truth?

The training loss (SSE_Interconnect_Composed.loss, prior off) is mse over (window, step, channel) of
the normalised closed-loop rollout error, nf = 400 (0.1 s), the model starting from its encoder.
Evaluated on the same windows (to_hist_future_data, stride 10, subsampled) of every training record:
  model   trained checkpoint, its encoder x0                     (checked against fs.loss)
  T1      truth (FP + absorber + tanh friction), true x0
  T2      truth, true gantry state, absorber state ZERO at the window start (unknown absorber state)
  noabs   truth without absorber, true x0 (best a model without added dynamics can do)
Decision (DECISIONS.md, G1): T2 < model -> a better solution exists even without knowing the
absorber state: optimisation. T2 >= model -> the windowed objective with an unknown initial
absorber state favours the damped shape: objective / encoder.

Usage: python g_window_loss.py [job]   (default 86894 = run 42)
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from e_thesis_checkpoints import JOBS, CK, cfg_for                        # noqa: E402  (sets sys.path)
from b_error_budget import variant, oracle_cl_batch, K0                   # noqa: E402

N_PER_REC = 48          # HEURISTIC: windows per record, spread evenly; 18 x 48 = 864 windows
NF = 400


def main(job):
    from scipy.io import loadmat
    from gantry_dynamic.data import load_datasets, compute_normalization, load_mat_aug, traj_dir
    from gantry_dynamic.model import build_model
    from gantry_dynamic.controller import build_closed_loop, controller_ss, y_op_for, gain_for
    from gantry_dynamic.training import load_checkpoint
    from common import oracle

    run, arm, na, seed = JOBS[job]
    cfg = cfg_for(arm, na, seed)
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    data = load_datasets(cfg); norm = compute_normalization(cfg, data)
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    fs = build_model(cfg.hp, cfg, data, norm)
    fs.simulator = build_closed_loop(fs, norm, cfg, train_files=data.train_files,
                                     val_files=data.val_files, val_data=data.val_ckpt_data, verbose=False)
    load_checkpoint(fs, os.path.join(CK, 'SSE_Interconnect_Composed_%s_best.pth' % job), True, 'lbfgs')
    ystd = np.asarray(norm.ystd, dtype=float).ravel()
    ctrl_rows = fs.simulator.train_ctrl_rows if hasattr(fs.simulator, 'train_ctrl_rows') else None
    na_r, nb_r = getattr(fs, 'na_right', 0), getattr(fs, 'nb_right', 0)

    res = {k: [] for k in ('model', 'T1', 'T2', 'noabs')}
    cls_of = []
    loss_check = []
    for r, (f, sd) in enumerate(zip(data.train_files, data.train_list)):
        sdn = fs.norm.transform(sd)
        uh, yh, uf, yf = sdn.to_hist_future_data(na=fs.na, nb=fs.nb, nf=NF, na_right=na_r, nb_right=nb_r, stride=cfg.stride)[:4]
        pick = np.linspace(0, len(uf) - 1, N_PER_REC).round().astype(int)
        uh, yh, uf, yf = uh[pick], yh[pick], uf[pick], yf[pick]
        # window start sample k of each pick, verified on content
        yn = np.asarray(sdn.y)
        k = max(fs.na, fs.nb) + pick * cfg.stride
        assert all(np.allclose(yn[kk:kk + NF], yf[i]) for i, kk in enumerate(k)), 'window start mapping wrong'
        # model: encoder + closed-loop rollout, exactly the loss path
        T = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=cfg.dtype_pt)   # noqa: E731
        row = 0 if ctrl_rows is None else int(np.asarray(ctrl_rows)[r])
        cix = torch.full((len(pick),), row, dtype=torch.long)
        with torch.no_grad():
            x0 = fs.encoder(T(uh), T(yh))
            yp, _ = fs.simulator(fs, x0, T(uf), T(yf), ctrl_ix=cix)
            loss_check.append(float(fs.loss(T(uh), T(yh), T(uf), T(yf), ctrl_ix=cix, nf=NF)))
        res['model'].append((yp.numpy() - yf))                                    # normalised
        # truth variants, physical, same windows
        u_ph, y_ph = np.asarray(sd.u, float), np.asarray(sd.y, float)
        _, _, xl, xa = load_mat_aug(f, cfg)
        meta = loadmat(os.path.join(traj_dir(cfg), f), variable_names=['meta'], squeeze_me=True,
                       struct_as_record=False)['meta']
        tp = oracle.truth_from_meta(meta.truth)
        ctrl = controller_ss(y_op_for(f), cfg.ts_new, gain_for(f))
        U = np.stack([u_ph[kk:kk + NF] for kk in k]); Y = np.stack([y_ph[kk:kk + NF] for kk in k])
        X0 = np.stack([[xl[kk, 0], xl[kk, 1], xl[kk, 2], xa[kk, 0], xl[kk, 3], xl[kk, 4], xl[kk, 5], xa[kk, 1]] for kk in k])
        X0z = X0.copy(); X0z[:, 3] = 0.0; X0z[:, 7] = 0.0
        for name, x0v, var in (('T1', X0, 'full'), ('T2', X0z, 'full'), ('noabs', X0, 'noabs')):
            yo = oracle_cl_batch(U, Y, x0v, [variant(tp, var)] * len(k), [ctrl] * len(k), cfg.ts_new)
            res[name].append((yo - Y) / ystd)
        cls_of += [os.path.basename(f).split('-')[1].split('_')[0][0]] * len(k)   # S Y P L T
        print('  %s done (%d windows)' % (f, len(k)), flush=True)

    cls_of = np.array(cls_of)
    route = cls_of == 'T'
    print('\n######## run %d (job %s) arm=%s n_a=%d: windowed objective, %d windows, nf %d' % (
        run, job, arm, na, len(cls_of), NF))
    E = {k: np.concatenate(v) for k, v in res.items()}
    mse_model = np.mean(E['model'] ** 2)
    print('  check: my model mse %.4e vs fs.loss mean %.4e' % (mse_model, np.mean(loss_check)))
    print('  %-6s  mse all     mse multisine  mse routes | Y rms [m] all   steps 0-40   steps 40-400' % '')
    for k, e in E.items():
        yr = lambda a: np.sqrt(np.mean(a[..., 2] ** 2)) * ystd[2]                  # noqa: E731
        print('  %-6s  %.4e  %.4e     %.4e | %.3e      %.3e    %.3e' % (
            k, np.mean(e ** 2), np.mean(e[~route] ** 2), np.mean(e[route] ** 2),
            yr(e), yr(e[:, :40]), yr(e[:, 40:])))
    np.savez(os.path.join(HERE, 'outputs', 'g_window_loss_%s.npz' % job),
             **{'mse_%s' % k: np.mean(e ** 2, axis=(1,)) for k, e in E.items()}, cls=cls_of,
             loss_check=np.array(loss_check))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '86894')
