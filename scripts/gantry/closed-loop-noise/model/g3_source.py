"""G3: source spectrum Phi_v = S^-1 Phi_e S^-H, S-only bound, interpretations, shaping filter H_v.

    python model/g3_source.py
Pre-registered in CN-006. Reads outputs/g1_spectra/g1.npz. Writes outputs/g3_source/{g3.npz,
g3.json, hv_<reading>.npz, hv_<reading>.mat, g3_source.png}.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cn_env                                                           # noqa: E402

import numpy as np                                                      # noqa: E402
from scipy.io import savemat                                            # noqa: E402

from measure import spec                                                # noqa: E402
from model import noise_model as nm                                     # noqa: E402
from sensitivity.g2_sensitivity import model_S                          # noqa: E402
cn_env.check_no_leak()

OUT = cn_env.out_dir('g3_source')
G1 = os.path.join(cn_env.OUTPUTS, 'g1_spectra', 'g1.npz')
AX = cn_env.AX
FS = spec.FS
ORDERS = (16, 32, 64, 128, 256, 512, 1024)   # CN-006 + CN-006a
N_BOOT = 1000
RNG = np.random.default_rng(20260925)
READINGS = ('sliding', 'tanh', 'stuck')
PRODUCTION = 'tanh'                      # CN-007 (deviation from the CN-005 default, see there)


def op_of(name):
    return name.split('/')[1]


def pooled_with_ci(Pr, Kr, idx, band_of, nb):
    """Pr (R, F, 3, 3) per-log sums -> pooled matrix, band means of autos, bootstrap CI."""
    K = Kr[idx].sum()
    P = Pr[idx].sum(0) / K
    Pb = np.stack([spec.band_mean(spec.autos(Pr[i]), band_of, nb) for i in range(len(Pr))])
    bv = Pb[idx].sum(0) / K
    boot = np.empty((N_BOOT, nb, 3))
    for b in range(N_BOOT):
        s_ = RNG.choice(idx, size=len(idx), replace=True)
        boot[b] = Pb[s_].sum(0) / Kr[s_].sum()
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return P, bv, lo, hi


def var_weighted_coh(P, f, fmin):
    A = spec.autos(P)
    m = f >= fmin
    out = {}
    for i, j in ((0, 1), (0, 2), (1, 2)):
        c = np.abs(P[m, i, j]) ** 2 / np.maximum(A[m, i] * A[m, j], 1e-300)
        w = np.sqrt(A[m, i] * A[m, j])
        out[f'{AX[i]}-{AX[j]}'] = float((c * w).sum() / w.sum())
    return out


def main():
    t0 = time.time()
    g1 = np.load(G1, allow_pickle=True)
    f = g1['f']; names = list(g1['names']); xy = g1['xy']; keep = g1['keep']
    Psum = g1['Psum']; Kr = g1['Kr']; band_of = g1['band_of']; fc = g1['fc']; fe = g1['fe']
    nb = len(fc); df = f[1] - f[0]
    idx = np.where(keep)[0]
    ops = sorted(set(op_of(n) for n in names))
    pos = [xy[[i for i, n in enumerate(names) if op_of(n) == op][0]] for op in ops]
    rec_pos = np.array([ops.index(op_of(n)) for n in names])
    Sm, SGm, _ = model_S(pos, f, readings=('sliding', 'tanh'))
    inband = (f >= fe[0, 0]) & (f <= fe[-1, 1])
    e_rms = np.sqrt(spec.autos(Psum[idx].sum(0) / Kr[idx].sum())[inband].sum(0) * df)
    summary = dict(e_rms_band_nm=(e_rms * 1e9).tolist())
    res = {}
    for rd in READINGS:
        if rd == 'stuck':
            Pv_r = Psum.copy()
        else:
            Sinv = np.linalg.inv(Sm[rd])
            Pv_r = np.stack([spec.shape(Sinv[rec_pos[i]], Psum[i]) for i in range(len(names))])
        Pv, bv, lo, hi = pooled_with_ci(Pv_r, Kr, idx, band_of, nb)
        Av = spec.autos(Pv)
        v_rms = np.sqrt(Av[inband].sum(0) * df)
        bound = bv.min(0)                                       # density, m^2/Hz
        bound_rms = np.sqrt(bound * FS / 2)
        coh = var_weighted_coh(Pv, f, fe[0, 0])
        rows = dict(v_rms_nm=(v_rms * 1e9).tolist(), v_over_e=(v_rms / e_rms).tolist(),
                    S_only_bound_density=bound.tolist(), S_only_bound_rms_nm=(bound_rms * 1e9).tolist(),
                    bound_band_hz=[float(fc[k]) for k in bv.argmin(0)], coherence_vw=coh,
                    band_rms_nm={f'{a}-{b}': (np.sqrt(Av[(f >= a) & (f < b)].sum(0) * df) * 1e9).round(3).tolist()
                                 for a, b in ((19.5, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 10001))})
        if rd != 'stuck':
            SGinv = np.linalg.inv(SGm[rd])
            Pd_r = np.stack([spec.shape(SGinv[rec_pos[i]], Psum[i]) for i in range(len(names))])
            Pd = Pd_r[idx].sum(0) / Kr[idx].sum()
            Ad = spec.autos(Pd)
            rows['d_rms_N'] = np.sqrt(Ad[inband].sum(0) * df).tolist()
            rows['d_band_rms_N'] = {f'{a}-{b}': np.sqrt(Ad[(f >= a) & (f < b)].sum(0) * df).round(5).tolist()
                                    for a, b in ((19.5, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 10001))}
            res[rd + '_Pd'] = Pd
        # ==== shaping filter ===============================================================
        correlated = any(c > 0.1 for c in coh.values())
        target = Pv.copy()
        first = band_of == 0
        target[f < fe[0, 0]] = Pv[first].mean(0)                # CN-006 extrapolation below 19.5 Hz
        R = nm.autocov(target, FS)
        fit = None
        tried = []
        for p in ORDERS:
            A, Sig = nm.fit_var(R, p, diagonal=not correlated)
            rho = nm.is_stable(A)
            Pm = nm.model_psd(A, Sig, f, FS)
            Am = spec.autos(Pm)
            bm = spec.band_mean(Am, band_of, nb)
            inside = ((bm >= lo) & (bm <= hi)).mean(0)
            rms_m = np.sqrt(Am[inband].sum(0) * df)
            rel = rms_m / v_rms - 1
            ok = bool(np.all(inside >= 0.9) and np.all(np.abs(rel) < 0.02) and rho < 1)
            tried.append(dict(p=p, inside=inside.round(3).tolist(), rms_rel=rel.round(4).tolist(),
                              max_abs_root=rho, ok=ok))
            print(f'[g3] {rd:8s} p={p:3d} ({"VAR" if correlated else "3x AR"}): bands inside CI '
                  f'{np.round(inside, 3)}, rms rel {np.round(rel * 100, 2)} %, max root {rho:.5f}, '
                  f'PASS {ok}')
            if ok and fit is None:
                fit = dict(p=p, A=A, Sig=Sig, Pm=Pm, bm=bm)
                break
        if fit is None:
            best = max(tried, key=lambda t: min(t['inside']))
            A, Sig = nm.fit_var(R, best['p'], diagonal=not correlated)
            Pm = nm.model_psd(A, Sig, f, FS)
            fit = dict(p=best['p'], A=A, Sig=Sig, Pm=Pm, bm=spec.band_mean(spec.autos(Pm), band_of, nb))
        rows.update(correlated=correlated, filter_order=fit['p'], filter_tries=tried,
                    filter_pass=any(t['ok'] for t in tried))
        np.savez(os.path.join(OUT, f'hv_{rd}.npz'), A=fit['A'], Sig=fit['Sig'], fs=FS, reading=rd)
        savemat(os.path.join(OUT, f'hv_{rd}.mat'), dict(A=fit['A'], Sig=fit['Sig'], fs=FS,
                                                        p=fit['p'], reading=rd))
        res[rd] = dict(Pv=Pv, bv=bv, lo=lo, hi=hi, Pm=fit['Pm'], bm=fit['bm'])
        summary[rd] = rows
        print(f'[g3] {rd:8s}: v rms (19.5 Hz-10 kHz) {np.round(v_rms * 1e9, 2)} nm = '
              f'{np.round(v_rms / e_rms, 3)} x e; S-only bound on white n: '
              f'{np.round(bound_rms * 1e9, 2)} nm rms (density {np.array2string(bound, precision=2)} m^2/Hz'
              f' at {[round(float(fc[k])) for k in bv.argmin(0)]} Hz); coherence {coh}')
        print(f'[g3] {rd:8s}: band rms [nm] {rows["band_rms_nm"]}')
        if 'd_rms_N' in rows:
            print(f'[g3] {rd:8s}: interpretation B force rms {np.round(rows["d_rms_N"], 4)} N; bands '
                  f'{rows["d_band_rms_N"]}')
    summary['production'] = PRODUCTION
    summary['PASS'] = summary[PRODUCTION]['filter_pass']
    json.dump(summary, open(os.path.join(OUT, 'g3.json'), 'w'), indent=1)
    np.savez_compressed(os.path.join(OUT, 'g3.npz'), f=f, fc=fc,
                        **{f'{rd}_{k}': res[rd][k] for rd in READINGS for k in res[rd]},
                        **{k: v for k, v in res.items() if k.endswith('_Pd')})
    plot(f, fc, res, g1)
    print(f'[g3] production ({PRODUCTION}) filter PASS: {summary["PASS"]}; done in {time.time() - t0:.0f} s')


def plot(f, fc, res, g1):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for i in range(3):
        a = ax[i]
        a.loglog(fc, g1['band_val'][:, i], color='#c3c2b7', lw=2, label='measured Phi_e (G1)')
        for rd, ls, c in (('sliding', '-', spec.COLORS[i]), ('tanh', 'dashed', spec.INK)):
            r = res[rd]
            if rd == 'sliding':
                a.fill_between(fc, r['lo'][:, i], r['hi'][:, i], color=c, alpha=0.25, lw=0)
            a.loglog(fc, r['bv'][:, i], color=c, ls=ls, lw=1.3, label=f'Phi_v, {rd}')
            a.loglog(fc, r['bm'][:, i], color=c, ls=':', lw=1.5, label=f'H_v fit, {rd}')
        a.set(xlabel='band centre [Hz]', ylabel='band-mean PSD [m^2/Hz]',
              title=f'{AX[i]}: source spectrum (de-shaped) and shaping filter')
        a.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(OUT, 'g3_source.png'), dpi=110)


if __name__ == '__main__':
    main()
