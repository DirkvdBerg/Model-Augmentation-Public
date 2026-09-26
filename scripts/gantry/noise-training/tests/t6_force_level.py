"""NT-T6: white input-force noise that reproduces the measured Telica standstill error covariance
in the simulation's closed loop (DECISIONS NT-008). Read-only; no dataset, no MATLAB, no training.

Loop: the frictionless 8-state truth (`transient/code/truth.py::Plant8`, production absorber
ma_frac 0.50, fa 150 Hz, zeta 0.03, L0 0.10), linearised at rest at Y_op, ZOH at 20 kHz, with the
generator's controller Cfb at 20 kHz (`gantry_dynamic/controller.py`, verified equal to the
generator in T2). Force d: white per 20 kHz sample, covariance Sd (3 x 3, stage axes), added at the
plant input. Error e = -y. Target: the measured zero-lag error covariance from `g1.npz`.
Writes outputs/<run>/t6.json.
"""
__project_origin__ = "added"

import json
import os
import sys

import numpy as np
from scipy.io import loadmat
from scipy.linalg import solve_discrete_lyapunov
from scipy.signal import cont2discrete

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import nt_env                                                      # noqa: E402
sys.path.insert(0, os.path.join(nt_env.REPO, 'scripts', 'gantry', 'transient', 'code'))
from truth import Plant8, P_np                                     # noqa: E402
from gantry_dynamic import controller as ctl                       # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else 't6_force_level'
OUT = nt_env.out_dir(RUN)
TS = 1.0 / 20000
CN = os.path.join(nt_env.REPO, 'scripts', 'gantry', 'closed-loop-noise', 'outputs')
IU = np.triu_indices(3)


def plant_ss(Y_op):
    """Frictionless linearisation at rest, stage force in, stage position out (as recipe_sensitivity.m)."""
    p = Plant8(ma_frac=0.50, fa=150.0, zeta_a=0.03, L0=0.10)
    x0 = np.array([0, 0, Y_op, 0, 0, 0, 0, 0], float)
    h = 1e-7
    A = np.zeros((8, 8)); B = np.zeros((8, 3))
    for i in range(8):
        e = np.zeros(8); e[i] = h
        A[:, i] = (p.deriv(x0 + e, np.zeros(3)) - p.deriv(x0 - e, np.zeros(3))) / (2 * h)
    for i in range(3):
        e = np.zeros(3); e[i] = h
        B[:, i] = (p.deriv(x0, e) - p.deriv(x0, -e)) / (2 * h)
    B = B @ P_np                                    # stage force -> logical generalised force
    C = np.hstack([P_np.T, np.zeros((3, 5))])       # logical q -> stage position
    Ad, Bd, Cd, _, _ = cont2discrete((A, B, C, np.zeros((3, 3))), TS, method='zoh')
    return Ad, Bd, Cd


def closed_loop(Y_op):
    Ad, Bd, Cd = plant_ss(Y_op)
    Ac, Bc, Cc, Dc = ctl.controller_ss(Y_op, TS)
    n, nc = Ad.shape[0], Ac.shape[0]
    Acl = np.block([[Ad - Bd @ Dc @ Cd, Bd @ Cc], [-Bc @ Cd, Ac]])
    Bcl = np.vstack([Bd, np.zeros((nc, 3))])        # force noise at the plant input
    Ccl = np.hstack([-Cd, np.zeros((3, nc))])       # e = r - y with r = 0
    Scl = (np.block([[Ad - Bd @ Dc @ Cd, Bd @ Cc], [-Bc @ Cd, Ac]]),
           np.vstack([-Bd @ Dc, -Bc]), np.hstack([Cd, np.zeros((3, nc))]), np.eye(3))
    assert np.max(np.abs(np.linalg.eigvals(Acl))) < 1
    return Acl, Bcl, Ccl, Scl


def err_cov(Acl, Bcl, Ccl, Sd):
    X = solve_discrete_lyapunov(Acl, Bcl @ Sd @ Bcl.T)
    return Ccl @ X @ Ccl.T


def lin_map(Acl, Bcl, Ccl):
    """6 x 6 map vec_sym(Sd) -> vec_sym(Se)."""
    L = np.zeros((6, 6))
    for k, (i, j) in enumerate(zip(*IU)):
        E = np.zeros((3, 3)); E[i, j] = E[j, i] = 1.0
        L[:, k] = err_cov(Acl, Bcl, Ccl, E)[IU]
    return L


def unvec(v):
    M = np.zeros((3, 3)); M[IU] = v; return M + np.triu(M, 1).T


def corr(M):
    s = np.sqrt(np.diag(M)); return M / np.outer(s, s)


def freq_resp(A, B, C, D, f):
    z = np.exp(2j * np.pi * f * TS)
    return np.stack([C @ np.linalg.solve(zz * np.eye(A.shape[0]) - A, B) + D for zz in z])


def main():
    g = np.load(os.path.join(CN, 'g1_spectra', 'g1.npz'), allow_pickle=True)
    f, P = g['f'], g['Ppool']
    Se_meas = np.real(P).sum(0) * (f[1] - f[0])
    res = {'Se_meas_nm2': (Se_meas * 1e18).tolist(), 'rms_meas_nm': (np.sqrt(np.diag(Se_meas)) * 1e9).tolist(),
           'corr_meas': corr(Se_meas).tolist(), 'Y_op': {}}
    print('measured rms %s nm, corr X1X2 %.3f X1Y %.3f X2Y %.3f' % (
        np.round(np.sqrt(np.diag(Se_meas)) * 1e9, 3), *corr(Se_meas)[[0, 0, 1], [1, 2, 2]]))
    # check: the S_sim built here equals the noise session's S_sim (recipe.mat) at the same Y_op
    rec = loadmat(os.path.join(CN, 'recipe', 'recipe.mat'), squeeze_me=True, struct_as_record=False)['SS']
    fchk = np.array([10.0, 100.0, 212.0, 1000.0, 5000.0])
    for Y_op in sorted({float(s.Y_op) for s in rec}):
        Acl, Bcl, Ccl, Scl = closed_loop(Y_op)
        s = next(r for r in rec if abs(float(r.Y_op) - Y_op) < 1e-12)
        Hm = freq_resp(np.atleast_2d(s.A), np.atleast_2d(s.B), np.atleast_2d(s.C), np.atleast_2d(s.D), fchk)
        Hp = freq_resp(*Scl, fchk)
        chk = float(np.max(np.abs(Hp - Hm)) / np.max(np.abs(Hm)))
        L = lin_map(Acl, Bcl, Ccl)
        sd_full = np.linalg.solve(L, Se_meas[IU])
        Sd_full = unvec(sd_full)
        eig = np.linalg.eigvalsh(Sd_full)
        # diagonal-only (independent axes): least squares on the three variances
        Ld = L[np.ix_([0, 3, 5], [0, 3, 5])]
        var_diag = np.linalg.solve(Ld, Se_meas[IU][[0, 3, 5]])
        Sd_diag = np.diag(var_diag)
        Se_diag = err_cov(Acl, Bcl, Ccl, Sd_diag)
        # if the full solution is not a covariance, clip to the nearest PSD one and report
        w, V = np.linalg.eigh(Sd_full)
        Sd_psd = (V * np.maximum(w, 0)) @ V.T
        Se_psd = err_cov(Acl, Bcl, Ccl, Sd_psd)
        out = dict(S_sim_check_rel=chk, Sd_full_N2=Sd_full.tolist(), Sd_full_eig=eig.tolist(),
                   full_is_psd=bool(eig.min() >= 0), sigma_full_N=np.sqrt(np.maximum(np.diag(Sd_full), 0)).tolist(),
                   corr_force_full=corr(Sd_full).tolist() if eig.min() > 0 else None,
                   Sd_psd_N2=Sd_psd.tolist(), rms_psd_nm=(np.sqrt(np.diag(Se_psd)) * 1e9).tolist(),
                   corr_err_psd=corr(Se_psd).tolist(),
                   sigma_diag_N=np.sqrt(var_diag).tolist(), rms_diag_nm=(np.sqrt(np.diag(Se_diag)) * 1e9).tolist(),
                   corr_err_diag=corr(Se_diag).tolist())
        res['Y_op'][str(Y_op)] = out
        print('Y_op %+.2f  S_sim check %.1e | full: psd %s, sigma %s N, eig %s | psd: rms %s nm corr %s | diag: sigma %s N, corr %s' % (
            Y_op, chk, out['full_is_psd'], np.round(out['sigma_full_N'], 3), np.round(eig, 5),
            np.round(out['rms_psd_nm'], 2), np.round(corr(Se_psd)[[0, 0, 1], [1, 2, 2]], 3),
            np.round(out['sigma_diag_N'], 3), np.round(corr(Se_diag)[[0, 0, 1], [1, 2, 2]], 3)), flush=True)
    # time-domain check at Y_op = 0 with the retained (PSD) covariance: 20 s of the linear loop
    Acl, Bcl, Ccl, _ = closed_loop(0.0)
    Sd = np.array(res['Y_op']['0.0']['Sd_psd_N2'])
    w, V = np.linalg.eigh(Sd)
    Lc = V * np.sqrt(np.maximum(w, 0))
    rng = np.random.default_rng(1)
    N = 400000
    d = rng.standard_normal((N, 3)) @ Lc.T
    x = np.zeros(Acl.shape[0]); e = np.empty((N, 3))
    for k in range(N):
        e[k] = Ccl @ x
        x = Acl @ x + Bcl @ d[k]
    e = e[20000:]
    Se_td = np.cov(e.T)
    res['time_domain_check'] = dict(rms_nm=(np.sqrt(np.diag(Se_td)) * 1e9).tolist(), corr=corr(Se_td).tolist(),
                                    force_rms_N=np.sqrt(np.diag(np.cov(d.T))).tolist())
    print('time domain (20 s, Y_op 0, PSD covariance): rms %s nm, corr %s' % (
        np.round(np.sqrt(np.diag(Se_td)) * 1e9, 2), np.round(corr(Se_td)[[0, 0, 1], [1, 2, 2]], 3)))
    with open(os.path.join(OUT, 't6.json'), 'w') as fh:
        json.dump(res, fh, indent=1)
    print('T6 done')


if __name__ == '__main__':
    main()
