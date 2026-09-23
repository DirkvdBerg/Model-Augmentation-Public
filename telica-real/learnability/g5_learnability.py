"""G5: learnability of the baseline's closed-loop residual (TR-017, pre-registered).

M1 repeatable residual power across exact repeats (iter0 group, iterETEL group): per-bin F test,
   FDR q = 0.01, SP/NP. M2 cross-validated model-error model (FIR on external regressors: reference
   acceleration and velocity, logged feedforward, Y_ref x reference acceleration), train -> held-out,
   headroom H = mean d_n / NP. M3 residual-reference correlation (descriptive). ANN settings from the
   residual: routing, nx_ann (Hankel rank of the MEM impulse response), horizon, band.
Reads outputs/g5_residuals/{train,validation,test}.npz. Writes outputs/g5_learnability/g5.json,
learnability/ann_settings.json and PNGs. Summaries only.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tr_env                                                          # noqa: E402

import numpy as np                                                     # noqa: E402
from scipy import stats                                                # noqa: E402
from scipy.signal import butter, sosfiltfilt                           # noqa: E402

from params import telica_params as tp                                 # noqa: E402
tr_env.check_no_leak()

OUT = tr_env.out_dir('g5_learnability')
RES = os.path.join(tr_env.OUTPUTS, 'g5_residuals')
ANN_FILE = os.path.join(tr_env.TR, 'learnability', 'ann_settings.json')
FS = tp.FS / 4
LAGS = 100                     # TR-017: FIR lags 0..99 at 5 kHz
Q_FDR = 0.01
LP_R = butter(4, 500.0, fs=FS, output='sos')      # HEURISTIC: smooth the um-quantised setpoint
DEGEN = ('train/xpos_-210_ypos-200/iter0', 2)      # MF230_Y constant (G3)
SIG = ('X1', 'X2', 'Y', 'X', 'Theta')              # stage axes gate; logical X, Theta for routing


def load(split):
    z = np.load(os.path.join(RES, f'{split}.npz'), allow_pickle=True)
    return {k: z[k] for k in z.files}


def signals(eps):
    """(N,3) stage residual -> (N,5): X1, X2, Y, X = mean of rails, Theta = difference / Lb."""
    return np.column_stack([eps, (eps[:, 0] + eps[:, 1]) / 2, (eps[:, 0] - eps[:, 1]) / tp.Lb])


def regressors(r, uff):
    """External regressors (never u_data): r_dd (3), r_d (3), u_ff (3), Y_ref * r_dd (3)."""
    rf = sosfiltfilt(LP_R, r.astype(np.float64), axis=0)
    rd = np.gradient(rf, 1 / FS, axis=0)
    rdd = np.gradient(rd, 1 / FS, axis=0)
    return np.column_stack([rdd, rd, uff.astype(np.float64), rf[:, 2:3] * rdd])


def lagged(X, L=LAGS):
    N, C = X.shape
    out = np.zeros((N, C * L))
    for l in range(L):
        out[l:, l * C:(l + 1) * C] = X[:N - l]
    return out


def bh(p, q):
    """Benjamini-Hochberg: boolean mask of discoveries."""
    n = len(p)
    o = np.argsort(p)
    thr = q * np.arange(1, n + 1) / n
    ok = p[o] <= thr
    k = np.max(np.where(ok)[0]) + 1 if ok.any() else 0
    m = np.zeros(n, bool)
    m[o[:k]] = True
    return m


def m1_group(D, it):
    idx = [i for i, n in enumerate(D['names']) if n.endswith('/' + it)]
    km = D['k_motion']
    Lc = min(len(D['res'][i]) - km[i] for i in idx)
    r0 = None
    ref_dev = 0.0
    W = []
    for i in idx:
        r = D['r'][i][km[i]:km[i] + Lc].astype(np.float64)
        rr = r - r[0]
        if r0 is None:
            r0 = rr
        ref_dev = max(ref_dev, float(np.abs(rr - r0).max()))
        W.append(signals(D['res'][i][km[i]:km[i] + Lc].astype(np.float64)))
    W = np.stack(W)                                                   # (M, Lc, 5)
    f = np.fft.rfftfreq(Lc, 1 / FS)
    E = np.fft.rfft(W, axis=1) / Lc                                   # (M, K, 5)
    w = np.full(len(f), 2.0); w[0] = 1.0
    if Lc % 2 == 0:
        w[-1] = 1.0
    band = (f >= 1.0) & (f <= 2500.0)
    out = dict(M=len(idx), Lc=int(Lc), ref_max_dev_m=ref_dev, valid=ref_dev <= 1e-6, sig={})
    names = [D['names'][i] for i in idx]
    for c, s in enumerate(SIG):
        keep = np.ones(len(idx), bool)
        if s in ('Y',):
            keep = np.array([not (n == DEGEN[0]) for n in names])
        Ec = E[keep, :, c]
        M = Ec.shape[0]
        Eb = Ec.mean(0)
        s2 = (np.abs(Ec - Eb) ** 2).sum(0) / (M - 1)
        T = M * np.abs(Eb) ** 2 / np.maximum(s2, 1e-300)
        p = stats.f.sf(T, 2, 2 * M - 2)
        disc = np.zeros(len(f), bool)
        disc[band] = bh(p[band], Q_FDR)
        SPk = w * (np.abs(Eb) ** 2 - s2 / M)
        NPk = w * s2
        SP, NP = float(SPk[band].sum()), float(NPk[band].sum())
        strong = disc & (SPk >= NPk)
        fs_ = f[strong]
        out['sig'][s] = dict(M=int(M), n_sig=int(disc.sum()), n_strong=int(strong.sum()),
                             SP=SP, NP=NP, SP_over_NP=SP / NP,
                             SP_over_NP_sigbins=float(SPk[disc].sum() / max(NPk[disc].sum(), 1e-300)),
                             band_hz=[float(fs_.min()), float(fs_.max())] if len(fs_) else None,
                             PASS=bool(disc.sum() >= 1 and SP / NP >= 1.0),
                             rms_total=float(np.sqrt(np.mean((np.abs(Ec) ** 2 * w).sum(1)))))
    out['_spec'] = (f, E)
    return out


def main():
    TR, VA, TE = load('train'), load('validation'), load('test')
    res = dict(M1={}, M2={}, M3={})
    # ================ M1 ================
    ALL = {k: np.concatenate([TR[k], VA[k], TE[k]]) for k in ('names', 'k_motion', 'res', 'r')}
    for it in ('iter0', 'iterETEL'):
        g = m1_group(ALL, it)
        g.pop('_spec')
        res['M1'][it] = g
        print(f'[M1] group {it}: M={g["M"]}, window {g["Lc"] / FS * 1e3:.0f} ms, reference max '
              f'deviation {g["ref_max_dev_m"] * 1e6:.3f} um (valid={g["valid"]})')
        for s in SIG:
            v = g['sig'][s]
            print(f'[M1]   {s:5s}: significant bins {v["n_sig"]:4d} (strong {v["n_strong"]:4d}), '
                  f'SP/NP {v["SP_over_NP"]:.3f} (sig bins {v["SP_over_NP_sigbins"]:.2f}), '
                  f'SP {np.sqrt(v["SP"]) * 1e9:.1f} nm rms, NP {np.sqrt(v["NP"]) * 1e9:.1f} nm rms, '
                  f'band {v["band_hz"]} -> {"PASS" if v["PASS"] else "FAIL"}')
    # standstill floor (per-sample variance of the logged servo error before motion)
    floor = {}
    fl = np.concatenate([np.concatenate(TR['floor']), np.concatenate(VA['floor']),
                         np.concatenate(TE['floor'])]).astype(np.float64)
    fl5 = signals(fl)
    for c, s in enumerate(SIG):
        floor[s] = float(fl5[:, c].var())
    res['standstill_floor_var'] = floor
    print('[floor] standstill servo-error rms [nm]: '
          + ', '.join(f'{s} {np.sqrt(floor[s]) * 1e9:.1f}' for s in SIG))
    # ================ M2 ================
    def seg(D, i):
        km = D['k_motion'][i]
        X = regressors(D['r'][i], D['uff'][i])[km:]
        Y = signals(D['res'][i].astype(np.float64))[km:]
        return X, Y
    # standardise channels on train
    Xs = [seg(TR, i)[0] for i in range(len(TR['names']))]
    mu = np.concatenate(Xs).mean(0); sd = np.concatenate(Xs).std(0) + 1e-30
    p_dim = Xs[0].shape[1] * LAGS + 1
    XtX = np.zeros((p_dim, p_dim)); XtY = np.zeros((p_dim, len(SIG))); n_tot = 0
    XtX_y = np.zeros((p_dim, p_dim)); XtY_y = np.zeros(p_dim)        # Y axis without DEGEN
    for i in range(len(TR['names'])):
        X, Y = seg(TR, i)
        A = np.hstack([lagged((X - mu) / sd), np.ones((len(X), 1))])
        G = A.T @ A
        XtX += G; XtY += A.T @ Y; n_tot += len(A)
        if TR['names'][i] != DEGEN[0]:
            XtX_y += G; XtY_y += A.T @ Y[:, 2]
    lam = 1e-6 * np.trace(XtX) / p_dim                                # TR-017 ridge (numerical)
    Ainv = np.linalg.inv(XtX + lam * np.eye(p_dim))
    B = Ainv @ XtY                                                    # (p, 5)
    B[:, 2] = np.linalg.solve(XtX_y + lam * np.eye(p_dim), XtY_y)     # DEGEN Y axis excluded
    del XtX_y
    # train residual stats for covariance / autocorrelation inflation
    rss = np.zeros(len(SIG)); infl = np.ones(len(SIG)); rtr = []
    for i in range(len(TR['names'])):
        X, Y = seg(TR, i)
        A = np.hstack([lagged((X - mu) / sd), np.ones((len(X), 1))])
        R = Y - A @ B
        rss += (R ** 2).sum(0)
        if i < 20:
            rtr.append(R)
    s2_res = rss / (n_tot - p_dim)
    Rc = np.concatenate(rtr)
    for c in range(len(SIG)):
        r = Rc[:, c] - Rc[:, c].mean(); c0 = r @ r
        acc = sum((1 - k / 101) * (r[:-k] @ r[k:]) / c0 for k in range(1, 101))
        infl[c] = max(1.0, 1 + 2 * acc)
    # held-out
    NPs = {s: res['M1']['iter0']['sig'][s]['NP'] for s in SIG}
    d = {s: [] for s in SIG}; r2 = {s: [] for s in SIG}
    for D in (VA, TE):
        for i in range(len(D['names'])):
            X, Y = seg(D, i)
            A = np.hstack([lagged((X - mu) / sd), np.ones((len(X), 1))])
            R = Y - A @ B
            for c, s in enumerate(SIG):
                e0 = np.mean(Y[:, c] ** 2); e1 = np.mean(R[:, c] ** 2)
                d[s].append(e0 - e1); r2[s].append(1 - e1 / e0)
    for c, s in enumerate(SIG):
        dn = np.array(d[s])
        H = float(dn.mean() / NPs[s]); Hf = float(dn.mean() / floor[s])
        t = stats.ttest_1samp(dn, 0.0, alternative='greater')
        ok = bool(H >= 1.0 and t.pvalue < 0.01)
        res['M2'][s] = dict(H=H, H_standstill=Hf, p=float(t.pvalue), n=int(len(dn)),
                            R2_median=float(np.median(r2[s])), R2_mean=float(np.mean(r2[s])),
                            PASS=ok, s2_res_train=float(s2_res[c]), infl=float(infl[c]))
        print(f'[M2] {s:5s}: held-out H = {H:.2f} (vs standstill floor {Hf:.1f}), p = {t.pvalue:.2e}, '
              f'R2 median {np.median(r2[s]):.3f} over {len(dn)} records -> {"PASS" if ok else "FAIL"}')
    # ================ M3 (descriptive) ================
    for c, s in enumerate(SIG[:3]):
        row = []
        for j in range(3):
            vals = []
            for D in (VA, TE):
                for i in range(len(D['names'])):
                    X, Y = seg(D, i)
                    x = X[:, j] - X[:, j].mean(); y = Y[:, c] - Y[:, c].mean()
                    cc = np.correlate(y, x, 'full') / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-300)
                    vals.append(np.abs(cc).max())
            row.append(float(np.median(vals)))
        res['M3'][s] = dict(max_abs_xcorr_with_rdd=row)
        print(f'[M3] {s}: median max |xcorr| with r_dd of [X1, X2, Y] = {np.round(row, 3).tolist()}')
    # ================ verdict ================
    stage = ('X1', 'X2', 'Y')
    m1 = {s: res['M1']['iter0']['sig'][s]['PASS'] or res['M1']['iterETEL']['sig'][s]['PASS']
          for s in SIG}
    m2 = {s: res['M2'][s]['PASS'] for s in SIG}
    go_axes = [s for s in stage if m1[s] and m2[s]]
    if not any(m1[s] for s in stage):
        verdict = 'NO-GO'
    elif go_axes:
        verdict = 'GO'
    else:
        verdict = 'CONDITIONAL'
    res['verdict'] = dict(verdict=verdict, go_axes=go_axes, m1=m1, m2=m2,
                          server_run_warranted='yes' if verdict == 'GO' else 'no')
    print(f'[G5] verdict {verdict}; axes passing M1 and M2: {go_axes}; server run warranted: '
          f'{res["verdict"]["server_run_warranted"]}')
    # ================ ANN settings (TR-017 rules) ================
    C = Xs[0].shape[1]
    def fir_block(coef):
        """lag x (out 3 stage) x (in 3 r_dd) impulse response, physical input units."""
        h = np.zeros((LAGS, 3, 3))
        for l in range(LAGS):
            for j in range(3):
                h[l, :, j] = coef[l * C + j, :3] / sd[j]
        return h
    h = fir_block(B)
    nb = LAGS // 2
    def hankel(hh):
        return np.block([[hh[i + j] for j in range(nb)] for i in range(nb)])
    sv = np.linalg.svd(hankel(h), compute_uv=False)
    rng = np.random.default_rng(0)
    idx_rdd = np.array([l * C + j for l in range(LAGS) for j in range(3)])
    cov_base = Ainv[np.ix_(idx_rdd, idx_rdd)]
    Lchol = np.linalg.cholesky(cov_base + 1e-30 * np.eye(len(idx_rdd)))
    smax = []
    for _ in range(200):
        pert = np.zeros_like(B)
        for c in range(3):
            pert[idx_rdd, c] = Lchol @ rng.standard_normal(len(idx_rdd)) * np.sqrt(s2_res[c] * infl[c])
        smax.append(np.linalg.svd(hankel(fir_block(pert)), compute_uv=False)[0])
    tau = float(np.percentile(smax, 95))
    order = int((sv > tau).sum())
    nx = int(min(16, max(2, order + (order % 2))))
    en = (h ** 2).sum((1, 2)); cum = np.cumsum(en) / en.sum()
    l99 = int(np.searchsorted(cum, 0.99)) + 1
    nf_s = max(0.005, np.ceil(l99 / FS / 0.005) * 0.005)
    routed_dofs = []
    if any(m1[s] or m2[s] for s in ('X1', 'X2', 'X', 'Theta', 'Y')):
        routed_dofs = ['X', 'Y']                                       # D-103: never Theta alone
        if m1['Theta'] or m2['Theta']:
            routed_dofs.insert(1, 'Theta')
    vel_rows = {'X': 3, 'Theta': 4, 'Y': 5}
    route = sorted(vel_rows[d_] for d_ in routed_dofs) + list(range(6, 6 + nx))
    bands = [res['M1'][g]['sig'][s]['band_hz'] for g in ('iter0', 'iterETEL') for s in stage
             if res['M1'][g]['sig'][s]['band_hz']]
    band = [min(b[0] for b in bands), max(b[1] for b in bands)] if bands else None
    ann = dict(nx_ann=nx, ann_route_ix=route, routed_dofs=routed_dofs, nf_seconds=float(nf_s),
               band_hz=band, hankel_order=order, hankel_sv_top=sv[:12].tolist(), hankel_tau=tau,
               fir_energy_99_lag_ms=l99 / FS * 1e3, fir_edge_warning=bool(l99 >= LAGS - 2),
               verdict=verdict, source='TR-017, outputs/g5_learnability/g5.json',
               # kept from the vendored pipeline, not derived here (G6 sets batch from memory):
               n_nodes_per_layer=24, n_hidden_layers=3, lr=1e-5, stride=20, burn_in=0,
               checkpoint_chunk=500, batch_size=256, epochs=100)
    json.dump(ann, open(ANN_FILE, 'w'), indent=1)
    res['ann_settings'] = ann
    print(f'[ann] Hankel SVs top {np.round(sv[:8] / sv[0], 4).tolist()} (rel), noise tau/sv0 '
          f'{tau / sv[0]:.4f} -> order {order} -> nx_ann {nx}; FIR 99 % energy at {l99 / FS * 1e3:.1f} ms '
          f'-> nf_seconds {nf_s}; routed DOFs {routed_dofs} -> rows {route}; band {band}')
    json.dump(res, open(os.path.join(OUT, 'g5.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
