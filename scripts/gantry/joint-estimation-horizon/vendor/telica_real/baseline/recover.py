"""G4 (1): baseline parameter recovery by equation error on TRAIN records (TR-014).

Logical EOM, linear in
    theta = [m_total, mh, m_diff, M11 = J_eff + mh d^2, mh d, cg1, cg2, cy, cb_sum, kb_sum,
             cc1, cc2, ccy]
on sliding samples only, 200 Hz zero-phase low-pass on positions and total force, rows decimated
10x, 5 ms guard around |v| < V_BRK. A per-record constant in the Theta row absorbs the rail
encoder skew (kb * Theta_0). A parameter moves from its TR-003 value only if its relative SE
(n_eff-corrected, rows weighted per row type) is < 20 %.
Writes baseline/recovered_params.json (used by g4_replay and G6) and outputs/g4_recover/.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tr_env                                                          # noqa: E402

import numpy as np                                                     # noqa: E402
from scipy.ndimage import binary_erosion                               # noqa: E402
from scipy.signal import butter, sosfiltfilt                           # noqa: E402

from params import telica_params as tp                                 # noqa: E402
from baseline.records import iter_records                              # noqa: E402
tr_env.check_no_leak()

OUT = tr_env.out_dir('g4_recover')
HERE = os.path.dirname(os.path.abspath(__file__))
FS = tp.FS
LP = butter(4, 200.0, fs=FS, output='sos')      # HEURISTIC 200 Hz (TR-014)
DEC = 10
GUARD = 100                                     # 5 ms at 20 kHz
Lb = tp.Lb
NAMES = ['m_total', 'mh', 'm_diff', 'M11', 'mh_d', 'cg1', 'cg2', 'cy', 'cb_sum', 'kb_sum',
         'cc1', 'cc2', 'ccy']
SE_RULE = 0.20                                  # HEURISTIC (TR-014)


def prior():
    c = tp.combos10()
    return dict(m_total=c['m_total'], mh=c['mh'], m_diff=c['m_diff'],
                M11=c['J_eff'] + c['mh'] * c['d'] ** 2, mh_d=c['mh'] * c['d'],
                cg1=c['cg1'], cg2=c['cg2'], cy=c['cy'], cb_sum=c['cb_sum'], kb_sum=c['kb_sum'],
                cc1=tp.CC[0], cc2=tp.CC[1], ccy=tp.CC[2])


def rows_of(d):
    q = sosfiltfilt(LP, d['q1'], axis=0)
    u = sosfiltfilt(LP, d['u'], axis=0)
    X = (q[:, 0] + q[:, 1]) / 2; Th = (q[:, 0] - q[:, 1]) / Lb; Y = q[:, 2]
    g = lambda s: np.gradient(s, 1 / FS)                               # noqa: E731
    Xd, Thd, Yd = g(X), g(Th), g(Y)
    Xdd, Thdd, Ydd = g(Xd), g(Thd), g(Yd)
    v1 = Xd + Lb / 2 * Thd; v2 = Xd - Lb / 2 * Thd; vy = Yd
    s1 = sosfiltfilt(LP, np.sign(v1)); s2 = sosfiltfilt(LP, np.sign(v2))
    sy = sosfiltfilt(LP, np.sign(vy))
    k0 = d['motion_idx']
    mot = np.zeros(len(X), bool); mot[k0:] = True
    er = lambda m: binary_erosion(m, iterations=GUARD)                 # noqa: E731
    mX = mot & er(np.abs(v1) >= tp.V_BRK) & er(np.abs(v2) >= tp.V_BRK)
    mY = mot & er(np.abs(vy) >= tp.V_BRK)
    iX = np.where(mX)[0][::DEC]; iY = np.where(mY)[0][::DEC]
    z = lambda i: np.zeros(len(i))                                     # noqa: E731
    AX = np.column_stack([Xdd[iX], Xdd[iX] - Y[iX] * Thdd[iX], Lb / 2 * Thdd[iX], z(iX), z(iX),
                          v1[iX], v2[iX], z(iX), z(iX), z(iX), s1[iX], s2[iX], z(iX)])
    bX = u[iX, 0] + u[iX, 1]
    AT = np.column_stack([z(iX), -Y[iX] * Xdd[iX] + Y[iX] ** 2 * Thdd[iX], Lb / 2 * Xdd[iX],
                          Thdd[iX], -Ydd[iX], Lb / 2 * v1[iX], -Lb / 2 * v2[iX], z(iX), Thd[iX],
                          Th[iX], Lb / 2 * s1[iX], -Lb / 2 * s2[iX], z(iX)])
    bT = Lb / 2 * (u[iX, 0] - u[iX, 1])
    AY = np.column_stack([z(iY), Ydd[iY], z(iY), z(iY), -Thdd[iY], z(iY), z(iY), vy[iY], z(iY),
                          z(iY), z(iY), z(iY), sy[iY]])
    bY = u[iY, 2]
    return (AX, bX), (AT, bT), (AY, bY)


def n_eff_factor(r, maxlag=50):
    """1 + 2 sum rho_k (Bartlett-truncated) of a residual sequence."""
    r = r - r.mean()
    c0 = r @ r
    s = 0.0
    for k in range(1, maxlag + 1):
        s += (1 - k / (maxlag + 1)) * (r[:-k] @ r[k:]) / c0
    return max(1.0, 1 + 2 * s)


CC_FROM_HOLDING = '--cc-from-holding' in sys.argv     # G4 attempt 2 (TR-015)


def held_cc_train():
    """Per-rail held friction (F_end - F_start)/2, median over TRAIN records (run g4_holding)."""
    R = json.load(open(os.path.join(tr_env.OUTPUTS, 'g4_holding', 'holding.json')))
    c = np.array([[(r['F_end'][j] - r['F_start'][j]) / 2 for j in range(3)]
                  for r in R if r['rest_ok'] and r['split'] == 'train'])
    return np.median(c, axis=0)


def main():
    blocks = {'X': [], 'T': [], 'Y': []}
    n_rec = 0
    for s, op, it, d in iter_records(splits=('train',)):
        (AX, bX), (AT, bT), (AY, bY) = rows_of(d)
        blocks['X'].append((AX, bX)); blocks['T'].append((AT, bT)); blocks['Y'].append((AY, bY))
        n_rec += 1
    print(f'[recover] {n_rec} train records; rows X {sum(len(b) for _, b in blocks["X"])}, '
          f'Y {sum(len(b) for _, b in blocks["Y"])}')
    nT = len(blocks['T'])
    # stack; Theta rows get one constant column per record (encoder skew)
    A_list, b_list, kind = [], [], []
    for (A, b) in blocks['X']:
        A_list.append(np.hstack([A, np.zeros((len(b), nT))])); b_list.append(b)
        kind += ['X'] * len(b)
    for r, (A, b) in enumerate(blocks['T']):
        C = np.zeros((len(b), nT)); C[:, r] = 1.0
        A_list.append(np.hstack([A, C])); b_list.append(b); kind += ['T'] * len(b)
    for (A, b) in blocks['Y']:
        A_list.append(np.hstack([A, np.zeros((len(b), nT))])); b_list.append(b)
        kind += ['Y'] * len(b)
    A = np.vstack(A_list); b = np.concatenate(b_list); kind = np.array(kind)
    cc_fixed = None
    if CC_FROM_HOLDING:
        cc_fixed = held_cc_train()
        b = b - A[:, 10:13] @ cc_fixed                  # friction force moved to the target
        A[:, 10:13] = 0.0
        print(f'[recover] ATTEMPT 2: cc fixed to the train held-friction medians '
              f'{np.round(cc_fixed, 2).tolist()} N; linear parameters re-fitted')
    # pass 1 OLS, pass 2 weighted by per-row-type residual std
    keep = np.ones(A.shape[1], bool)
    if cc_fixed is not None:
        keep[10:13] = False
    th = np.zeros(A.shape[1])
    th[keep], *_ = np.linalg.lstsq(A[:, keep], b, rcond=None)
    r = b - A @ th
    w = np.ones_like(b)
    sd = {}
    for k in ('X', 'T', 'Y'):
        sd[k] = r[kind == k].std(); w[kind == k] = 1 / sd[k]
    Aw, bw = A * w[:, None], b * w
    th = np.zeros(A.shape[1])
    th[keep], *_ = np.linalg.lstsq(Aw[:, keep], bw, rcond=None)
    rw = bw - Aw @ th
    infl = max(n_eff_factor(rw[kind == k]) for k in ('X', 'T', 'Y'))
    se = np.zeros(A.shape[1])
    Ak = Aw[:, keep]
    cov = np.linalg.inv(Ak.T @ Ak) * (rw @ rw) / (len(bw) - keep.sum()) * infl
    se[keep] = np.sqrt(np.diag(cov))
    if cc_fixed is not None:
        th[10:13] = cc_fixed
        se[10:13] = 0.0                                  # fixed from the holding measurement
    est = dict(zip(NAMES, th[:13])); ses = dict(zip(NAMES, se[:13]))
    pr = prior()
    final, moved = {}, {}
    print(f'[recover] residual std per row: X {sd["X"]:.2f} N, Theta {sd["T"]:.3f} Nm, '
          f'Y {sd["Y"]:.2f} N; autocorrelation inflation {infl:.1f}; cond(Aw) '
          f'{np.linalg.cond(Aw[:, :13]):.2e}')
    print(f'  {"param":8s} {"prior":>10s} {"estimate":>11s} {"SE":>10s} {"relSE":>7s}  used')
    for n in NAMES:
        rel = abs(ses[n] / est[n]) if est[n] != 0 else np.inf
        ok = rel < SE_RULE and est[n] > 0 if n != 'm_diff' else rel < SE_RULE
        if cc_fixed is not None and n in ('cc1', 'cc2', 'ccy'):
            ok = True                                    # measured at rest, not regressed
        final[n] = float(est[n]) if ok else float(pr[n]); moved[n] = bool(ok)
        print(f'  {n:8s} {pr[n]:10.4g} {est[n]:11.4g} {ses[n]:10.3g} {rel:7.3f}  '
              f'{"RECOVERED" if ok else "prior"}')
    # back to the ten combinations and the block's raw 14 (gauge: mb, Jb:Jh from TR-003)
    mh = final['mh']; d = final['mh_d'] / mh if moved['mh_d'] else tp.d
    J_eff = final['M11'] - mh * d ** 2 if moved['M11'] else tp.combos10()['J_eff']
    combos = dict(kb_sum=final['kb_sum'], cg1=final['cg1'], cg2=final['cg2'], cy=final['cy'],
                  cb_sum=final['cb_sum'], mh=mh, m_total=final['m_total'],
                  m_diff=final['m_diff'], J_eff=J_eff, d=d)
    mb = tp.mb
    m_sum = combos['m_total'] - mb
    m1 = (m_sum + combos['m_diff']) / 2; m2 = (m_sum - combos['m_diff']) / 2
    J_sum = J_eff - m_sum * Lb ** 2 / 4
    r_ = tp.Jb / (tp.Jb + tp.Jh)
    raw = dict(kb1=combos['kb_sum'] / 2, kb2=combos['kb_sum'] / 2, cg1=combos['cg1'],
               cg2=combos['cg2'], cy=combos['cy'], cb1=combos['cb_sum'] / 2,
               cb2=combos['cb_sum'] / 2, mh=mh, m1=m1, m2=m2, mb=mb, Jb=J_sum * r_,
               Jh=J_sum * (1 - r_), d=d)
    cc = [final['cc1'], final['cc2'], final['ccy']]
    # admissibility of M(Y)
    Ys = np.linspace(-0.25, 0.25, 201)
    alpha = combos['m_total'] + mh; beta = combos['m_diff'] * Lb / 2
    mins = []
    for Y in Ys:
        M = np.array([[alpha, beta - mh * Y, 0], [beta - mh * Y, J_eff + mh * d ** 2 + mh * Y ** 2,
                                                  -mh * d], [0, -mh * d, mh]])
        mins.append(np.linalg.eigvalsh(M).min())
    out = dict(estimate=est, se=ses, moved=moved, final=final, combos=combos, raw14=raw, cc=cc,
               row_resid_std=sd, inflation=infl, n_records=n_rec, min_eig_M=float(min(mins)),
               theta_skew_const=[float(v) for v in th[13:]])
    out['cc_source'] = 'held friction (train median)' if cc_fixed is not None else 'regression'
    tag = '_a2' if cc_fixed is not None else ''
    json.dump(out, open(os.path.join(HERE, f'recovered_params{tag}.json'), 'w'), indent=1)
    json.dump(out, open(os.path.join(OUT, f'recover{tag}.json'), 'w'), indent=1)
    print(f'[recover] combos: ' + ', '.join(f'{k} {v:.4g}' for k, v in combos.items()))
    print(f'[recover] cc [N]: {np.round(cc, 2).tolist()}; min eig M(Y) over +-0.25 m: '
          f'{min(mins):.4g}')


if __name__ == '__main__':
    main()
