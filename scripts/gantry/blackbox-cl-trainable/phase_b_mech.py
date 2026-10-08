"""Phase B follow-up (DECISIONS.md BB2-012): mechanism of the B3 failure, plus B5 and B6, which
pb3 did not reach (import clash on `common`).

M1: small-signal poles of the closed-loop step map at the initial model, seeds 1 to 3:
    (a) model + K1 (the training loop), (b) model + Kt (by construction the loop reduces to S).
    Jacobian of z = [x, xc] -> z_next at z = 0, u_data = y_data = 0, by autograd.
Writes outputs/phase_b_mech.json.
"""
__project_origin__ = "added"

import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bbcl'))
from setup import cfg_for, load, ROOT, REPO                           # noqa: E402
import ici_model as M                                                   # noqa: E402
from model_augmentation.fit_systems.closed_loop import ControllerBank   # noqa: E402
from gantry_dynamic.controller import controller_ss                     # noqa: E402

torch.set_num_threads(4)
OUT = os.path.join(ROOT, 'outputs')
res = {}


def loop_poles(fs, bank):
    hf = fs.hfn
    ctrl = bank.gather(torch.tensor([0]))
    nx, nc = hf.nx, bank.nc

    def F(z):
        x, xc = z[None, :nx], z[None, nx:]
        y = hf.output_only(x)
        u_fb, xc2 = bank.step(xc, -y, ctrl)
        _, x2 = hf(x, u_fb)
        return torch.cat([x2, xc2], 1)[0]
    J = torch.autograd.functional.jacobian(F, torch.zeros(nx + nc, dtype=torch.float64))
    ev = np.linalg.eigvals(J.detach().numpy())
    i = int(np.argmax(np.abs(ev)))
    return float(np.abs(ev[i])), float(np.angle(ev[i])), int(np.sum(np.abs(ev) > 1))


def main():
    t0 = time.time()
    cfg = cfg_for('noisy', 1)
    data, norm = load(cfg)
    ts = cfg.ts_new
    Kt = M.kt_physical(ts)
    kt_bank = ControllerBank(*(np.asarray(m)[None] for m in Kt), ystd=norm.ystd, std_u=norm.std_u,
                             dtype=torch.float64)
    sigma_w, m1 = None, []
    for seed in (1, 2, 3):
        fs = M.build(cfg, data, norm, seed, sigma_w=sigma_w, verbose=False)
        sigma_w = fs.hfn.sigma_w.numpy()
        a = loop_poles(fs, fs.simulator.bank)
        b = loop_poles(fs, kt_bank)
        m1.append(dict(seed=seed, K1=a, Kt=b, S_max_radius=fs.hfn.max_radius()))
        print('[M1] seed %d  loop with K1: max|z| %.6f angle %.2e rad (%d outside) | loop with Kt: '
              'max|z| %.6f (%d outside) | S alone %.6f'
              % (seed, a[0], a[1], a[2], b[0], b[2], fs.hfn.max_radius()), flush=True)
    res['M1'] = m1

    # B5: truth as a CHECK only, imported by file path (the pipeline's `common` package shadows it)
    spec = importlib.util.spec_from_file_location(
        'excl_common', os.path.join(REPO, 'scripts', 'gantry', 'excitation-closed-loop', 'common.py'))
    TRUTH = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(TRUTH)
    from scipy.signal import cont2discrete
    K1 = controller_ss(0.0, ts)

    def frf(K, f):
        A, B, C, D = K
        z = np.exp(1j * 2 * np.pi * f * ts)
        return np.array([C @ np.linalg.solve(zz * np.eye(A.shape[0]) - A, B) + D for zz in z])

    f = np.linspace(50, 300, 251)
    rel = np.abs(frf(Kt, f) - frf(K1, f)).max(axis=(1, 2)) / np.abs(frf(K1, f)).max(axis=(1, 2))

    def rho(K, Y):
        A, B, Cy = TRUTH.ss_stage(*TRUTH.truth_mck(Y))
        Ad, Bd, Cd, _, _ = cont2discrete((A, B, Cy, np.zeros((3, 3))), ts, method='zoh')
        Ak, Bk, Ck, Dk = K
        Acl = np.block([[Ad - Bd @ Dk @ Cd, Bd @ Ck], [-Bk @ Cd, Ak]])
        return float(np.max(np.abs(np.linalg.eigvals(Acl))))
    res['B5'] = dict(Kt_vs_K1_rel_max_50_300=float(rel.max()),
                     rho={('%+.2f' % Y): dict(Kt=rho(Kt, Y), K1=rho(K1, Y)) for Y in (-0.3, 0.0, 0.3)})
    print('[B5] |Kt - K1| / |K1| max 50-300 Hz %.2e; truth loop radius %s'
          % (rel.max(), json.dumps(res['B5']['rho'])), flush=True)

    # B6 (reduced): Kt folding against the pipeline's ControllerBank, and strict causality
    hf = fs.hfn
    Kt_f = M.fold(*Kt, norm.ystd, norm.std_u)
    fold_err = max(
        float(torch.max(torch.abs(kt_bank.M_state[0] - torch.cat([torch.as_tensor(Kt_f[2]),
                                                                   torch.as_tensor(Kt_f[0])])))),
        float(torch.max(torch.abs(kt_bank.M_error[0] - torch.cat([torch.as_tensor(Kt_f[3]),
                                                                   torch.as_tensor(Kt_f[1])])))))
    with torch.no_grad():
        xx = torch.randn(1, hf.nx, dtype=torch.float64)
        ya, _ = hf(xx, torch.zeros(1, 3, dtype=torch.float64))
        yb, _ = hf(xx, torch.ones(1, 3, dtype=torch.float64))
    res['B6'] = dict(fold_err=fold_err, causal_dy=float(torch.max(torch.abs(ya - yb))))
    print('[B6] fold err %.1e, y[k] vs u[k] %.1e' % (fold_err, res['B6']['causal_dy']), flush=True)
    res['wall_s'] = time.time() - t0
    with open(os.path.join(OUT, 'phase_b_mech.json'), 'w') as fh:
        json.dump(res, fh, indent=1)
    print('[B] wrote outputs/phase_b_mech.json (%.0f s)' % res['wall_s'], flush=True)


if __name__ == '__main__':
    main()
