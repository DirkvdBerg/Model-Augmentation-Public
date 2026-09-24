"""G2 gate (TR-007): Karnopp torch port vs MATLAB references, and the cc = 0 no-op.

    python telica-real/friction/check_friction.py prep      writes outputs/g2_friction/ref_inputs.mat
    matlab -sd telica-real/friction/matlab -batch make_ref_g2
    python telica-real/friction/check_friction.py compare   prints the verdict, writes g2_metrics.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tr_env                                                          # noqa: E402

import numpy as np                                                     # noqa: E402
import torch                                                           # noqa: E402
from scipy.io import loadmat, savemat                                  # noqa: E402

from params import telica_params as tp                                 # noqa: E402
from friction.karnopp import make_friction_block, make_plain_block     # noqa: E402
tr_env.check_no_leak()

OUT = tr_env.out_dir('g2_friction')
N, H = 8000, tp.TS
X0 = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])


def u_stage_profile(n=N, h=H):
    t = np.arange(n) * h
    u = np.empty((n, 3))
    hold = t < 0.05
    u[hold] = [20.0, -15.0, 25.0]                     # below cc on every rail: must stay stuck
    tt = t[~hold]
    u[~hold, 0] = 250 * np.sin(2 * np.pi * 5 * tt) + 30
    u[~hold, 1] = 250 * np.sin(2 * np.pi * 5 * tt + 0.5) - 20
    u[~hold, 2] = 120 * np.sin(2 * np.pi * 7 * tt + 1.0)
    return t, u


def prep():
    _, u = u_stage_profile()
    p = {n: float(getattr(tp, n)) for n in ('m1', 'm2', 'mb', 'mh', 'Jb', 'Jh', 'Lb', 'd', 'cg1',
                                            'cg2', 'cy', 'cb1', 'cb2', 'kb1', 'kb2')}
    p.update(cc1=tp.CC[0], cc2=tp.CC[1], ccy=tp.CC[2])
    savemat(os.path.join(OUT, 'ref_inputs.mat'),
            dict(p=p, u_stage=u, x0=X0.reshape(6, 1), h=H, ma1=1e-6, ma2=2e-6,
                 f_abs=200.0, zeta_abs=0.5))
    print('[prep] wrote', os.path.relpath(os.path.join(OUT, 'ref_inputs.mat'), tr_env.REPO))


def rollout(blk, u, x0, census=False):
    dt = blk.P_mat.dtype
    x = torch.as_tensor(x0, dtype=dt).reshape(1, 6, 1)
    ut = torch.as_tensor(u, dtype=dt)
    xs = np.empty((len(u), 6))
    held = np.zeros((len(u), 3), bool)
    stuck0 = np.zeros((len(u), 3), bool)
    with torch.no_grad():
        for k in range(len(u)):
            xs[k] = x[0, :, 0].double().numpy()
            uk = ut[k].reshape(1, 3, 1)
            if census:
                blk.record_census = True
                blk._deriv_with(x, uk, blk._mats())
                blk.record_census = False
                held[k] = blk.last_census['held'][0].numpy()
                stuck0[k] = blk.last_census['stuck0'][0].numpy()
            x = blk.nonlinear_function(torch.cat([x, uk], dim=1))
    return xs, held, stuck0


def stage_pos(xs):
    Lb = tp.Lb
    return np.stack([xs[:, 0] + Lb / 2 * xs[:, 1], xs[:, 0] - Lb / 2 * xs[:, 1], xs[:, 2]], 1)


def compare():
    res, ok = {}, True
    _, u = u_stage_profile()

    # (a) cc = 0 bit identity, LPV and frozen, float64 and float32 (2000 steps)
    for dt in (torch.float64, torch.float32):
        for yop in (None, 0.1):
            a = make_plain_block(Y_op=yop, dtype=dt)
            b = make_friction_block(Y_op=yop, cc=(0.0, 0.0, 0.0), mode='karnopp', dtype=dt)
            xa, _, _ = rollout(a, u[:2000], X0)
            xb, _, _ = rollout(b, u[:2000], X0)
            dmax = float(np.abs(xa - xb).max())
            key = f'a_cc0_{str(dt).split(".")[1]}_{"lpv" if yop is None else "frozen"}'
            res[key] = dmax
            ok &= (dmax == 0.0)
            print(f'(a) cc=0 vs parent, {key}: max|diff| = {dmax:.3e}  '
                  f'{"PASS" if dmax == 0.0 else "FAIL"}')

    # torch Karnopp, float64, LPV branch, the gate trajectory
    blk = make_friction_block(Y_op=None, mode='karnopp', dtype=torch.float64)
    import time
    t0 = time.time()
    xt, held_t, stuck0_t = rollout(blk, u, X0, census=True)
    print(f'[compare] torch rollout {N} steps in {time.time() - t0:.1f} s')

    R = loadmat(os.path.join(OUT, 'ref_outputs.mat'))
    xR2 = R['x_R2']
    held_R2 = R['held_R2'].astype(bool)
    stuck0_R2 = R['stuck0_R2'].astype(bool)

    # (b) R2
    e_pos = np.abs(stage_pos(xt) - stage_pos(xR2)).max()
    e_all = np.abs(xt - xR2).max(axis=0)
    census_same = bool(np.array_equal(held_t, held_R2) and np.array_equal(stuck0_t, stuck0_R2))
    n_diff = int((held_t != held_R2).sum() + (stuck0_t != stuck0_R2).sum())
    res.update(b_max_stage_pos_err_m=float(e_pos), b_state_err=e_all.tolist(),
               b_census_identical=census_same, b_census_mismatches=n_diff)
    okb = e_pos <= 1e-12 and census_same
    ok &= okb
    print(f'(b) R2 6-state copy: max|y_torch - y_R2| = {e_pos:.3e} m (tol 1e-12), '
          f'census identical = {census_same} ({n_diff} mismatches)  {"PASS" if okb else "FAIL"}')

    # (c) R1 Richardson
    cols = [0, 1, 2, 4, 5, 6]
    x1, x2 = R['x8_ma1'][:, cols], R['x8_ma2'][:, cols]
    y0 = 2 * x1 - x2
    first_order = np.abs(stage_pos(x1) - stage_pos(x2)).max()
    e_rich = np.abs(stage_pos(xt) - stage_pos(y0)).max()
    e_raw = np.abs(stage_pos(xt) - stage_pos(x1)).max()
    tol_c = max(first_order, 1e-12)
    okc = e_rich <= tol_c
    ok &= okc
    res.update(c_first_order_absorber_effect_m=float(first_order),
               c_err_vs_richardson_m=float(e_rich), c_err_vs_ma1_m=float(e_raw),
               c_tol_m=float(tol_c))
    print(f'(c) R1 original .m, ma->0: max|y_torch - y0| = {e_rich:.3e} m, tol '
          f'{tol_c:.3e} (|y(1e-6)-y(2e-6)|), raw vs ma=1e-6 {e_raw:.3e}  '
          f'{"PASS" if okc else "FAIL"}')

    # (d) census
    n_hold = int(round(0.05 / H))
    held_n = held_t.sum(0)
    slide_n = (~stuck0_t).sum(0)
    broke_n = (stuck0_t & ~held_t).sum(0)
    first_held = bool(held_t[:n_hold].all())
    okd = bool((held_n > 0).all() and (slide_n > 0).all() and first_held)
    ok &= okd
    res.update(d_held=held_n.tolist(), d_sliding=slide_n.tolist(), d_broke=broke_n.tolist(),
               d_all_held_first_50ms=first_held)
    print(f'(d) census per rail [X1, X2, Y]: held {held_n.tolist()}, sliding {slide_n.tolist()}, '
          f'broke away {broke_n.tolist()}, all held in first 50 ms = {first_held}  '
          f'{"PASS" if okd else "FAIL"}')

    # figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = np.arange(N) * H
    fig, axs = plt.subplots(3, 2, figsize=(12, 7), sharex=True)
    v = (xt[:, 3:] @ np.array([[1, 1, 0], [tp.Lb / 2, -tp.Lb / 2, 0], [0, 0, 1]]))
    for i, nm in enumerate(('X1', 'X2', 'Y')):
        axs[i, 0].plot(t, stage_pos(xt)[:, i] * 1e3, 'k', lw=0.8)
        axs[i, 0].set_ylabel(f'{nm} [mm]')
        axs[i, 1].plot(t, v[:, i] * 1e3, 'C0', lw=0.8)
        axs[i, 1].fill_between(t, -3, 3, where=held_t[:, i], color='C3', alpha=0.3, lw=0,
                               label='held (stuck)')
        axs[i, 1].set_ylabel(f'd{nm} [mm/s]')
        axs[i, 1].set_ylim(-3 * max(1, np.abs(v[:, i]).max() * 1e3 / 3), None)
    axs[0, 1].legend(fontsize=8)
    axs[-1, 0].set_xlabel('t [s]'); axs[-1, 1].set_xlabel('t [s]')
    fig.suptitle(f'G2 Karnopp port: torch vs MATLAB max err {e_pos:.1e} m (R2)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'g2_trajectory.png'), dpi=110)
    res['PASS'] = bool(ok)
    json.dump(res, open(os.path.join(OUT, 'g2_metrics.json'), 'w'), indent=2)
    print('G2 CHECK', 'PASS' if ok else 'FAIL')


if __name__ == '__main__':
    {'prep': prep, 'compare': compare}[sys.argv[1]]()
