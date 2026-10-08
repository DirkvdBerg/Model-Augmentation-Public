"""bla2 (DECISIONS.md BB2-004): the plant BLA below the multisine band, 1 to 106 Hz, from the 18
training records by the local polynomial method on the full-record DFT, with a control run of the
same estimator on a friction-free fixed-Y LTI simulation of the truth in the K1 loop, driven by the
records' own references and injections.

The truth is used only as the scoring reference and as the control's data generator (user
2026-09-29); nothing here is imported by any black-box code.

Run: python -u bla/bla_lowband.py   (writes bla/outputs/bla_lowband.npz and .json)
"""
__project_origin__ = "added"

import json
import os
import sys

import numpy as np
from scipy.signal import cont2discrete, dlsim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bla_check as B                                             # noqa: E402  loaders, K1, truth
from gantry_dynamic.controller import controller_ss               # noqa: E402  K1 state space

N = 240000
DF = 1.0 / (N * B.TS)                                             # 1/12 Hz
KB = np.arange(int(round(1.0 / DF)), int(round(106.0 / DF)))      # 1 .. 106 Hz
FB = KB * DF
LPM_N, LPM_R = 7, 2                                               # BB2-004


def lti_loop_y0():
    """Friction-free truth at Y = 0 (ZOH 20 kHz) in the K1 loop. Inputs [r_dev; f], outputs [y; u]."""
    A, Bm, Cy = B.TRUTH.ss_stage(*B.TRUTH.truth_mck(0.0))
    Ad, Bd, Cd, _, _ = cont2discrete((A, Bm, Cy, np.zeros((3, 3))), B.TS, method='zoh')
    Ac, Bc, Cc, Dc = controller_ss(0.0, B.TS)
    npx, ncx = Ad.shape[0], Ac.shape[0]
    Acl = np.block([[Ad - Bd @ Dc @ Cd, Bd @ Cc], [-Bc @ Cd, Ac]])
    Bcl = np.block([[Bd @ Dc, Bd], [Bc, np.zeros((ncx, 3))]])
    Ccl = np.block([[Cd, np.zeros((3, ncx))], [-Dc @ Cd, Cc]])
    Dcl = np.block([[np.zeros((3, 3)), np.zeros((3, 3))], [Dc, np.eye(3)]])
    rho = np.max(np.abs(np.linalg.eigvals(Acl)))
    return (Acl, Bcl, Ccl, Dcl, B.TS), rho


def spectra(y, u, f, r):
    """Full-record DFT at KB of outputs [y; u] (offset removed) and of the exogenous force
    W = Cfb(z) R + F (reference deviation from its start)."""
    rd = r - r[0]
    Y = np.fft.rfft(y - y[0], axis=0)[KB]
    U = np.fft.rfft(u - u[0], axis=0)[KB]
    W = B.cfb_freq(FB) * np.fft.rfft(rd, axis=0)[KB] + np.fft.rfft(f, axis=0)[KB]
    return W, np.concatenate([Y, U], axis=1)


def lpm_pooled(recs, n=LPM_N, R=LPM_R):
    """recs: list of (W (nk, 3), Z (nk, 6)). LPM per bin, pooled, transient polynomial per record.
    Returns G = S_yw S_uw^-1 (nk, 3, 3) and the input-regressor condition number per bin."""
    nk, nrec = recs[0][0].shape[0], len(recs)
    G = np.empty((nk, 3, 3), complex)
    cond = np.empty(nk)
    ncol = 3 * (R + 1) + nrec * (R + 1)
    for k in range(nk):
        lo = min(max(k - n, 0), nk - 2 * n - 1)
        rr = np.arange(lo, lo + 2 * n + 1) - k
        A = np.zeros((nrec * len(rr), ncol), complex)
        Z = np.empty((nrec * len(rr), 6), complex)
        for i, (W, Zr) in enumerate(recs):
            for j, r in enumerate(rr):
                row = i * len(rr) + j
                for s in range(R + 1):
                    A[row, 3 * s:3 * s + 3] = W[k + r] * r ** s
                    A[row, 3 * (R + 1) + i * (R + 1) + s] = r ** s
                Z[row] = Zr[k + r]
        sc = np.max(np.abs(A), axis=0); sc[sc == 0] = 1
        th = np.linalg.lstsq(A / sc, Z, rcond=None)[0] / sc[:, None]
        Th = th[:3].T                                              # (6, 3)
        G[k] = Th[:3] @ np.linalg.inv(Th[3:])
        cond[k] = np.linalg.cond((A / sc)[:, :3])
    return G, cond


def score(G):
    E = np.stack([B.rel_err(G, B.g_true(Yg, FB)) for Yg in B.Y_GRID])
    return E.min(0), E[list(B.Y_GRID).index(0.0)]


def bands(e):
    out = {}
    for lo, hi in ((1, 10), (10, 20), (20, 40), (40, 60), (60, 80), (80, 106)):
        m = (FB >= lo) & (FB < hi)
        out['%d-%d' % (lo, hi)] = float(np.median(e[m]))
    m = FB >= 20
    out['med_20_106'] = float(np.median(e[m]))
    return out


def main():
    rows = B.training_records()
    sysd, rho = lti_loop_y0()
    print('control loop (truth Y 0 + K1, 20 kHz) spectral radius %.6f' % rho, flush=True)
    recs = {'noisefree': [], 'noisy': [], 'control': []}
    for row in rows:
        for ver in ('noisefree', 'noisy'):
            d = B.load(ver, row)
            r = np.broadcast_to(d['r_sim'], d['y'].shape).astype(float) if np.ndim(d['r_sim']) else np.zeros_like(d['y'])
            recs[ver].append(spectra(d['y'], d['u_total'], d['f_sim'], r))
            if ver == 'noisefree':
                _, out, _ = dlsim(sysd, np.hstack([r - r[0], d['f_sim']]))
                yc, uc = out[:, :3] + r[0], out[:, 3:]
                assert np.all(np.isfinite(out))
                recs['control'].append(spectra(yc, uc, d['f_sim'], r))
            del d
        print('  %s loaded' % row['record'], flush=True)
    res, summ = {'freq': FB}, {'control_rho': float(rho)}
    for ver, rl in recs.items():
        G, cond = lpm_pooled(rl)
        e_env, e0 = score(G)
        res['G_' + ver], res['cond_' + ver], res['e_env_' + ver], res['e_Y0_' + ver] = G, cond, e_env, e0
        summ[ver] = dict(env=bands(e_env), Y0=bands(e0), cond_median=float(np.median(cond)),
                         cond_max=float(np.max(cond)))
        print('%-9s envelope %s' % (ver, json.dumps(summ[ver]['env'])), flush=True)
        print('%-9s Y 0      %s | cond med %.1e max %.1e' % (ver, json.dumps(summ[ver]['Y0']),
                                                             summ[ver]['cond_median'], summ[ver]['cond_max']),
              flush=True)
    ctrl = summ['control']['Y0']['med_20_106'] <= 0.10
    data = all(summ[v]['env']['med_20_106'] <= 0.10 for v in ('noisefree', 'noisy'))
    summ['verdict'] = dict(control_PASS=ctrl, data_PASS=data,
                           reading=('proper BLA with LPM' if ctrl and data else
                                    'data BLA is not the linear plant in 20-106 Hz (friction / LPV)' if ctrl else
                                    'training references do not excite 20-106 Hz enough'))
    print('VERDICT', json.dumps(summ['verdict']), flush=True)
    np.savez_compressed(os.path.join(B.OUT, 'bla_lowband.npz'), **res)
    with open(os.path.join(B.OUT, 'bla_lowband.json'), 'w') as fh:
        json.dump(summ, fh, indent=1)
    print('wrote bla/outputs/bla_lowband.npz, bla_lowband.json')


if __name__ == '__main__':
    main()
