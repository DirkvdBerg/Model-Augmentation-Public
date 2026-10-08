"""Which poles did the trained model realise? (mechanism search, phase H)

The one-step map of the trained model, x_{k+1} = F(x_k, u_k) (normalised coordinates, physics RK4 +
Static_ANN_Block), linearised at the encoder states of the training windows: J = dF/dx. Its
eigenvalues are coordinate-free, so they compare directly with the truth's discrete poles
z = exp(lambda Ts) (truth without friction, linearised at the same Y). Also: the added-state
self-map A_aa = d x_a+ / d x_a, and the physics-only Jacobian (ANN output removed).
Reading: the truth has a pair at |z| 0.977, angle 0.333 rad (212 Hz, zeta about 0.07). A learned pair
near that angle but far inside the circle = the over-damped mode is REALISED in the model.

Usage: python h_learned_poles.py [job ...]   (default 86894 = run 42; U runs only, OBC needs obc_pass)
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from e_thesis_checkpoints import JOBS, CK, cfg_for                        # noqa: E402  (sets sys.path)
from b_error_budget import _deriv_b, _stack, variant                     # noqa: E402

N_PER_REC = 48          # HEURISTIC: same window pick as g_window_loss.py
TS = 1 / 4000.0


def build(job):
    from gantry_dynamic.data import load_datasets, compute_normalization
    from gantry_dynamic.model import build_model
    from gantry_dynamic.controller import build_closed_loop
    from gantry_dynamic.training import load_checkpoint
    run, arm, na, seed = JOBS[job]
    cfg = cfg_for(arm, na, seed)
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    data = load_datasets(cfg); norm = compute_normalization(cfg, data)
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    fs = build_model(cfg.hp, cfg, data, norm)
    fs.simulator = build_closed_loop(fs, norm, cfg, train_files=data.train_files,
                                     val_files=data.val_files, val_data=data.val_ckpt_data, verbose=False)
    load_checkpoint(fs, os.path.join(CK, 'SSE_Interconnect_Composed_%s_best.pth' % job), True, 'lbfgs')
    return cfg, data, norm, fs


def encoder_points(cfg, data, fs, multisine_only=True):
    """Encoder states x0 and the input u at the window starts (normalised), plus physical Y."""
    X, U, Yph, rec = [], [], [], []
    na_r, nb_r = getattr(fs, 'na_right', 0), getattr(fs, 'nb_right', 0)
    for r, (f, sd) in enumerate(zip(data.train_files, data.train_list)):
        cls = os.path.basename(f).split('-')[1].split('_')[0][0]
        if multisine_only and cls == 'T':
            continue
        sdn = fs.norm.transform(sd)
        uh, yh, uf, yf = sdn.to_hist_future_data(na=fs.na, nb=fs.nb, nf=1, na_right=na_r, nb_right=nb_r,
                                                 stride=cfg.stride)[:4]
        pick = np.linspace(0, len(uf) - 1, N_PER_REC).round().astype(int)
        T = lambda a: torch.as_tensor(np.ascontiguousarray(a[pick]), dtype=cfg.dtype_pt)   # noqa: E731
        with torch.no_grad():
            X.append(fs.encoder(T(uh), T(yh)))
        U.append(T(uf)[:, 0])
        Yph.append(np.asarray(sd.y)[max(fs.na, fs.nb) + pick * cfg.stride, 2])
        rec += [r] * len(pick)
    return torch.cat(X), torch.cat(U), np.concatenate(Yph), np.array(rec)


def step_jac(fs, x, u, ann_off=False):
    """Per-sample Jacobian dF/dx of the one-step map, (B, nx, nx)."""
    from model_augmentation.fit_systems.blocks import Static_ANN_Block
    ann = next(b for b in fs.hfn.connected_blocks if isinstance(b, Static_ANN_Block))
    gate = ann.out_gate.clone()
    if ann_off:
        ann.out_gate = torch.zeros_like(gate)
    try:
        f = lambda xx: fs.hfn(xx, u)[1].reshape(x.shape[0], -1).sum(0)          # noqa: E731
        J = torch.autograd.functional.jacobian(f, x.clone())                      # (nx, B, nx)
    finally:
        ann.out_gate = gate
    return J.permute(1, 0, 2).detach().numpy()


def zinfo(z):
    lr, th = np.log(np.abs(z)), np.abs(np.angle(z))
    return th / (2 * np.pi * TS), -lr / np.sqrt(lr ** 2 + th ** 2 + 1e-300)       # f [Hz], zeta


def truth_poles(tp, Y):
    """Discrete poles exp(lambda Ts) of the truth (no friction) linearised at rest, head at Y."""
    p = _stack([variant(tp, 'nofr')])
    x0 = np.zeros((1, 8)); x0[0, 2] = Y
    A = np.zeros((8, 8)); h = 1e-7
    for j in range(8):
        dx = np.zeros((1, 8)); dx[0, j] = h
        A[:, j] = (_deriv_b(x0 + dx, np.zeros((1, 3)), p) - _deriv_b(x0 - dx, np.zeros((1, 3)), p))[0] / (2 * h)
    return np.exp(np.linalg.eigvals(A) * TS)


def band_pairs(ev, lo=0.15, hi=0.6):
    """Eigenvalues with angle in [lo, hi] rad (about 95 to 380 Hz), upper half plane."""
    return [z for z in ev if lo <= np.angle(z) <= hi]


def main(jobs):
    from scipy.io import loadmat
    from gantry_dynamic.data import traj_dir
    from common import oracle
    for job in jobs:
        run, arm, na, seed = JOBS[job]
        cfg, data, norm, fs = build(job)
        x, u, Yph, rec = encoder_points(cfg, data, fs)
        meta = loadmat(os.path.join(traj_dir(cfg), data.train_files[0]), variable_names=['meta'],
                       squeeze_me=True, struct_as_record=False)['meta']
        tp = oracle.truth_from_meta(meta.truth)
        J = step_jac(fs, x, u)
        J0 = step_jac(fs, x, u, ann_off=True)
        nphy = 6
        print('\n######## run %d (job %s) arm=%s n_a=%d: %d encoder points (multisine train windows), Y %.3f to %.3f m'
              % (run, job, arm, na, len(x), Yph.min(), Yph.max()))
        zt = band_pairs(truth_poles(tp, float(np.median(Yph))))
        print('  truth (no friction, Y median): ' + '  '.join('|z| %.4f ang %.3f -> %.1f Hz zeta %.3f'
              % ((abs(z), np.angle(z)) + zinfo(z)) for z in zt))
        res = {'full': [], 'phys': [], 'aa': []}
        for i in range(len(x)):
            res['full'].append(np.linalg.eigvals(J[i]))
            res['phys'].append(np.linalg.eigvals(J0[i]))
            if na > 0:
                res['aa'].append(np.linalg.eigvals(J[i][nphy:, nphy:]))
        for k in ('full', 'phys'):
            ev = np.array(res[k])
            bp = [band_pairs(e) for e in ev]
            nb = np.array([len(b) for b in bp])
            print('  %-5s: pairs in 95 to 380 Hz per point: %s' % (k, dict(zip(*np.unique(nb, return_counts=True)))))
            flat = np.concatenate([np.array(b) for b in bp if len(b)]) if nb.sum() else np.array([])
            if len(flat):
                f, z = zinfo(flat)
                for q in (10, 50, 90):
                    print('         pct %2d: |z| %.4f  f %.1f Hz  zeta %.3f' % (
                        q, np.percentile(np.abs(flat), q), np.percentile(f, q), np.percentile(z, q)))
            # all eigenvalue moduli, median over points (sorted)
            print('         median sorted |eig|: %s' % ' '.join('%.4f' % v for v in np.median(np.sort(np.abs(ev), 1), 0)))
        if na > 0:
            ev = np.array(res['aa'])
            print('  A_aa (added self-map) eig, median over points: |z| %s, angle %s rad' % (
                ' '.join('%.4f' % v for v in np.median(np.sort(np.abs(ev), 1), 0)),
                ' '.join('%.3f' % v for v in np.median(np.sort(np.abs(np.angle(ev)), 1), 0))))
        # the mean Jacobian (a single linear model of the learned dynamics)
        evm = np.linalg.eigvals(J.mean(0))
        print('  mean-Jacobian eig (|z|, angle, f, zeta): ' + '  '.join('(%.4f, %.3f, %.0f Hz, %.3f)'
              % ((abs(z), np.angle(z)) + zinfo(z)) for z in sorted(evm, key=lambda v: -abs(v)) if np.angle(z) >= 0))
        np.savez(os.path.join(HERE, 'outputs', 'h_poles_%s.npz' % job), J=J, J0=J0, Y=Yph, rec=rec,
                 zt=np.array(zt))


if __name__ == '__main__':
    main(sys.argv[1:] or ['86894'])
