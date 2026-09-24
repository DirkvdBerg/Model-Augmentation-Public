"""G2: model S (three standstill-plant readings) vs empirical S from the ILC feedforward (CN-005).

    python sensitivity/g2_sensitivity.py
Reads outputs/g1_spectra/g1.npz. Writes outputs/g2_sensitivity/{g2.npz, g2.json, *.png}.
Summaries only.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cn_env                                                           # noqa: E402

import numpy as np                                                      # noqa: E402

from measure import spec                                                # noqa: E402
from sensitivity import loop                                            # noqa: E402
from real_data_verification.telica_loader import _A_TO_N                # noqa: E402
from params import telica_params as tp                                  # noqa: E402
cn_env.check_no_leak()

OUT = cn_env.out_dir('g2_sensitivity')
G1 = os.path.join(cn_env.OUTPUTS, 'g1_spectra', 'g1.npz')
AX = cn_env.AX
FS = spec.FS
PRE = 400                     # window starts PRE samples before j0 (feedforward leads by 257)
COH = 0.9                     # CN-005
EPS_K = np.array([0.045, 0.045, 0.245])   # CN-005a: sqrt(1 - held-out VAF), telica-real G3
EPS_M = np.array([0.20, 0.15, 0.38])      # CN-005b: telica-real G4 held-out iter0 replay NRMSE
A2N = np.asarray(tp.a_to_n())


def op_of(name):
    return name.split('/')[1]


def model_S(positions, f, readings=('sliding', 'tanh')):
    """{reading: S (P, F, 3, 3)} and {reading: SG}, K (F, 3, 3) N/m."""
    K = loop.frf(*loop.ctrl_ss(), f)
    I = np.eye(3)[None]
    out, outSG = {}, {}
    for rd in readings:
        S_, SG_ = [], []
        for y in positions:
            Ap, Bp, Cp = loop.plant_ss(y, rd)
            G = loop.frf(Ap, Bp, Cp, np.zeros((3, 3)), f)
            S = np.linalg.inv(I + G @ K)
            S_.append(S); SG_.append(S @ G)
        out[rd] = np.stack(S_); outSG[rd] = np.stack(SG_)
    return out, outSG, K


def main():
    t0 = time.time()
    g1 = np.load(G1, allow_pickle=True)
    f = g1['f']; names = list(g1['names']); xy = g1['xy']; keep = g1['keep']
    ops = sorted(set(op_of(n) for n in names))
    pos = {op: xy[[i for i, n in enumerate(names) if op_of(n) == op][0]] for op in ops}
    P_list = [pos[op] for op in ops]
    print(f'[g2] {len(ops)} positions; Y range {min(p[2] for p in P_list):.3f} .. '
          f'{max(p[2] for p in P_list):.3f} m')
    summary = {}

    # ==== 1. model S and closed-loop modes ====================================================
    Sm, SGm, Kf = model_S(P_list, f)
    modes = {}
    for rd in ('sliding', 'tanh', 'stuck'):
        ld = [loop.least_damped(loop.closed_loop(y, rd)[0]) for y in P_list]
        modes[rd] = dict(f_hz=[x[0] for x in ld], zeta=[x[1] for x in ld],
                         max_abs_eig=max(x[2] for x in ld))
        print(f'[g2] {rd:8s}: least-damped closed-loop mode {np.median(modes[rd]["f_hz"]):.1f} Hz, '
              f'zeta {np.min(modes[rd]["zeta"]):.4f} .. {np.max(modes[rd]["zeta"]):.4f}; '
              f'max |eig| {modes[rd]["max_abs_eig"]:.6f}')
    summary['closed_loop_modes'] = modes
    for rd in Sm:
        S = Sm[rd]
        mag = np.abs(S)
        dg = np.sqrt((mag[..., [0, 1, 2], [0, 1, 2]] ** 2).sum(-1))
        od = np.sqrt((mag ** 2).sum((-1, -2)) - dg ** 2)
        ratio = od / dg
        peak = mag[..., [0, 1, 2], [0, 1, 2]].max(1)                    # (P, 3)
        fpk = f[mag[..., [0, 1, 2], [0, 1, 2]].argmax(1)]
        summary[f'S_{rd}'] = dict(
            offdiag_over_diag_max=float(ratio.max()), f_at_max=float(f[ratio.max(0).argmax()]),
            offdiag_over_diag_median=float(np.median(ratio)),
            S_peak=np.median(peak, 0).tolist(), S_peak_f=np.median(fpk, 0).tolist(),
            X1X2_max=float(mag[..., 0, 1].max()), X1Y_max=float(mag[..., 0, 2].max()),
            YX1_max=float(mag[..., 2, 0].max()))
        print(f'[g2] S {rd:8s}: |S_ii| peak {np.round(np.median(peak, 0), 2)} at '
              f'{np.round(np.median(fpk, 0), 0)} Hz; offdiag/diag max {ratio.max():.3f} '
              f'(at {f[ratio.max(0).argmax()]:.0f} Hz), median {np.median(ratio):.4f}; '
              f'max |S12| {mag[..., 0, 1].max():.3f}, |S13| {mag[..., 0, 2].max():.4f}, '
              f'|S31| {mag[..., 2, 0].max():.4f}')

    # ==== 2. diagonal vs MIMO de-shaping on the pooled standstill spectrum =====================
    Psum = g1['Psum']; Kr = g1['Kr']
    band_of = g1['band_of']; nb = len(g1['fc'])
    half = (g1['b_hi'] - g1['b_lo']) / 2 / g1['band_val']
    rec_pos = [ops.index(op_of(n)) for n in names]
    diag_ok = {}
    for rd in Sm:
        Sinv = np.linalg.inv(Sm[rd])                                     # (P, F, 3, 3)
        Pv = np.zeros_like(Psum[0]); Pd = np.zeros((len(f), 3))
        for i in np.where(keep)[0]:
            Pv += spec.shape(Sinv[rec_pos[i]], Psum[i])
            Pd += spec.autos(Psum[i]) / np.abs(np.einsum('fii->fi', Sm[rd][rec_pos[i]])) ** 2
        Kt = Kr[keep].sum()
        Av = spec.autos(Pv / Kt); Ad = Pd / Kt
        bv = spec.band_mean(Av, band_of, nb); bd = spec.band_mean(Ad, band_of, nb)
        rel = np.abs(bd / bv - 1)
        frac = (rel < half).mean(0)
        diag_ok[rd] = dict(frac_bands_within_ci=frac.tolist(), max_rel_diff=rel.max(0).tolist(),
                           suffices=bool(np.all(frac >= 0.9)))
        print(f'[g2] diagonal-only de-shaping ({rd}): bands within G1 CI {np.round(frac, 3)}, '
              f'max rel diff {np.round(rel.max(0), 3)} -> diagonal suffices: {diag_ok[rd]["suffices"]}')
    summary['diagonal_vs_mimo'] = diag_ok

    # ==== 3. empirical S from the ILC feedforward ==============================================
    recs = spec.all_records()
    segs = {}
    pre_ff = []
    for (s, op, it, p) in recs:
        try:
            d = spec.load(p)
        except Exception:
            continue
        j0 = d['j0']
        i_ff = d['u_ff'] / _A_TO_N
        segs.setdefault(op, []).append((it, d['e'][j0 - PRE:], i_ff[j0 - PRE:]))
        nz = np.where(np.any(d['u_ff'][:j0] != 0, 1))[0]
        if len(nz) and it != 'iterETEL':
            a = int(nz[0])
            pre_ff.append(dict(name=f'{s}/{op}/{it}', op=op, e=d['e'][a - 2200:j0].copy(),
                               u=d['u_ff'][a - 2200:j0].copy(), a=2200))
    L = min(len(x[1]) for v in segs.values() for x in v)
    fe = np.fft.rfftfreq(L, 1 / FS)
    print(f'[g2] empirical: {sum(len(v) for v in segs.values())} records, window {L} samples '
          f'({L / FS * 1e3:.0f} ms), {len(fe)} bins')
    dE, dU = [], []
    for op, v in segs.items():
        E = np.fft.rfft(np.stack([x[1][:L] for x in v]), axis=1)
        U = np.fft.rfft(np.stack([x[2][:L] for x in v]), axis=1)
        dE.append(E - E.mean(0)); dU.append(U - U.mean(0))
    dof = sum(len(v) - 1 for v in segs.values())
    dE = np.concatenate(dE); dU = np.concatenate(dU)                   # (n, F, 3)
    Ef = np.transpose(dE, (1, 2, 0)); Uf = np.transpose(dU, (1, 2, 0)) # (F, 3, n)
    M = Uf @ np.conj(np.transpose(Uf, (0, 2, 1)))
    Minv = np.linalg.inv(M)
    SG = -(Ef @ np.conj(np.transpose(Uf, (0, 2, 1)))) @ Minv          # (F, 3, 3) m/A
    R = Ef + SG @ Uf
    rss = (np.abs(R) ** 2).sum(-1)
    coh = 1 - rss / np.maximum((np.abs(Ef) ** 2).sum(-1), 1e-300)
    sig2 = rss / max(dof - 3, 1)
    KA = loop.frf(*loop.ctrl_ss(amps=True), fe)
    S_emp = np.eye(3)[None] - SG @ KA
    kMk = np.real(np.einsum('fmj,fmn,fnj->fj', KA.conj(), Minv, KA))  # (F, 3) k_j^H M^-1 k_j
    sd = np.sqrt(sig2[:, :, None] * kMk[:, None, :])                  # (F, 3, 3)
    Sm_e, SGm_e, _ = model_S(P_list, fe, readings=("sliding", "tanh"))
    sd_SG = np.sqrt(sig2[:, :, None] * np.real(np.einsum("fjj->fj", Minv))[:, None, :])
    agree = {}
    for rd in ('sliding', 'tanh'):
        Smod = Sm_e[rd]
        mid = Smod.mean(0)
        spread = (np.abs(Smod - mid[None]).max(0))
        T_mid = np.eye(3)[None] - mid
        tol = (2 * sd + spread + EPS_K[None, None, :] * np.abs(T_mid)      # CN-005a
               + EPS_M[None, :, None] * np.abs(mid))                     # CN-005b
        dif = np.abs(S_emp - mid)
        cohm = np.repeat((coh >= COH)[:, :, None], 3, 2)
        ok = (dif <= tol) & cohm
        frac = ok.sum() / max(cohm.sum(), 1)
        SGA = SGm_e[rd] * A2N[None, None, None, :]                      # m/N -> m/A
        SGmid = SGA.mean(0)
        tolSG = 2 * sd_SG + np.abs(SGA - SGmid[None]).max(0)
        fracSG = ((np.abs(SG - SGmid) <= tolSG) & cohm).sum() / max(cohm.sum(), 1)
        agree[rd] = dict(n_coherent=int(cohm.sum()), frac_agree=float(frac),
                         frac_agree_SG=float(fracSG))
        print(f"[g2] empirical vs model ({rd}): coherent (bin, element) pairs {int(cohm.sum())}, "
              f"S agree {frac * 100:.1f} % (CN-005a tolerance); SG agree {fracSG * 100:.1f} %")
        if rd == 'sliding':
            Smod_s, tol_s, ok_s, dif_s = mid, tol, ok, dif
    cf = {AX[i]: [float(fe[coh[:, i] >= COH].min()) if (coh[:, i] >= COH).any() else None,
                  float(fe[coh[:, i] >= COH].max()) if (coh[:, i] >= COH).any() else None,
                  int((coh[:, i] >= COH).sum())] for i in range(3)}
    print(f'[g2] coherent range per output [f_min, f_max, n_bins]: {cf}')
    # disagreement bands (sliding) per output row, in the G1-style bands
    bo, fcb, feb = spec.bands(fe)
    dis = {}
    for i in range(3):
        rows = []
        for b in range(len(fcb)):
            m = (bo == b) & (coh[:, i] >= COH)
            if m.sum() == 0:
                continue
            fr = ok_s[m, i, :].mean()
            if fr < 0.8:
                rel = (dif_s[m, i, :] / np.maximum(np.abs(Smod_s[m, i, :]), 1e-12)).mean(0)
                rows.append(dict(band=[float(feb[b, 0]), float(feb[b, 1])], frac_agree=float(fr),
                                 mean_rel_diff_row=rel.round(3).tolist()))
        dis[AX[i]] = rows
        print(f'[g2] row {AX[i]}: {len(rows)} coherent bands with < 80 % agreement: '
              + '; '.join(f'{r["band"][0]:.0f}-{r["band"][1]:.0f} Hz ({r["frac_agree"]:.2f})' for r in rows))
    summary['empirical'] = dict(n_records=int(dE.shape[0]), dof=int(dof), window_ms=L / FS * 1e3,
                                coherent=cf, agree=agree, disagreement_bands_sliding=dis,
                                PASS_frac=bool(agree['sliding']['frac_agree'] >= 0.8))

    # ==== 4. standstill plant from the pre-motion feedforward window ===========================
    sysd = {rd: {} for rd in ('sliding', 'tanh')}
    res = {rd: [] for rd in ('sliding', 'tanh', 'stuck')}
    amp = []
    for r_ in pre_ff:
        y = pos[r_['op']]
        meas = r_['e'] - r_['e'][:r_['a'] - 200].mean(0)
        seg = slice(r_['a'] - 200, len(meas))
        mw = meas[seg]
        amp.append(np.sqrt((mw[200:] ** 2).mean(0)))
        for rd in ('sliding', 'tanh'):
            key = r_['op']
            if key not in sysd[rd]:
                Acl, Bcl, Ccl, Dcl = loop.closed_loop(y, rd)
                sysd[rd][key] = (Acl, Bcl[:, 3:], Ccl[:3], Dcl[:3, 3:])
            A_, B_, C_, D_ = sysd[rd][key]
            em = loop.lsim(A_, B_, C_, D_, r_['u'][seg])
            res[rd].append(np.sqrt(((mw - em) ** 2).mean(0)) / np.sqrt((mw ** 2).mean(0)))
        res['stuck'].append(np.ones(3))
    amp = np.array(amp)
    nrm = {rd: np.median(np.array(v), 0).tolist() for rd, v in res.items()}
    ss_rms = np.sqrt(spec.autos(g1['Ppool']).sum(0) * (f[1] - f[0]))
    decidable = (np.median(amp, 0) > 3 * ss_rms)
    best = min(("sliding", "tanh", "stuck"), key=lambda rd: np.mean(nrm[rd]))
    if not decidable.any():
        best = "sliding"                      # CN-005: undecidable -> sliding (default)
    print(f'[g2] pre-motion window: {len(pre_ff)} ILC logs; measured rms in [nz0, j0) median '
          f'{np.round(np.median(amp, 0) * 1e9, 1)} nm vs standstill {np.round(ss_rms * 1e9, 1)} nm '
          f'(decidable: {decidable.tolist()})')
    for rd in res:
        print(f'[g2]   NRMSE {rd:8s}: median {np.round(nrm[rd], 3)}')
    print(f'[g2] standstill plant chosen: {best} (adequate: {bool(np.mean(nrm[best]) < 0.5)})')
    summary['standstill_plant'] = dict(n=len(pre_ff), meas_rms_nm=(np.median(amp, 0) * 1e9).tolist(),
                                       standstill_rms_nm=(ss_rms * 1e9).tolist(),
                                       decidable=decidable.tolist(), nrmse_median=nrm, chosen=best,
                                       adequate=bool(np.mean(nrm[best]) < 0.5))
    json.dump(summary, open(os.path.join(OUT, 'g2.json'), 'w'), indent=1)
    np.savez_compressed(os.path.join(OUT, 'g2.npz'), f=f, ops=np.array(ops),
                        positions=np.array(P_list), S_sliding=Sm['sliding'], S_tanh=Sm['tanh'],
                        K=Kf, fe=fe, S_emp=S_emp, sd=sd, coh=coh, S_mod_e=Smod_s, tol=tol_s,
                        SG_emp=SG)
    plot(f, Sm, fe, S_emp, sd, coh, Smod_s, pre_ff, sysd, pos)
    print(f'[g2] done in {time.time() - t0:.0f} s')


def plot(f, Sm, fe, S_emp, sd, coh, Smod, pre_ff, sysd, pos):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    for i in range(3):
        a = ax[0, i]
        m = coh[:, i] >= COH
        a.loglog(f[1:], np.abs(Sm['sliding'][:, 1:, i, i]).mean(0), color=spec.INK, lw=1.2,
                 label='model, sliding')
        a.loglog(f[1:], np.abs(Sm['tanh'][:, 1:, i, i]).mean(0), color=spec.INK, lw=1, ls='dashed',
                 label='model, tanh at v = 0')
        a.loglog(fe[m], np.abs(S_emp[m, i, i]), '.', color=spec.COLORS[i], ms=3,
                 label=f'empirical (coherence >= {COH})')
        a.loglog(fe[~m & (fe > 0)], np.abs(S_emp[~m & (fe > 0), i, i]), '.', color='#c3c2b7', ms=2,
                 label='empirical, not coherent')
        a.set(xlabel='frequency [Hz]', ylabel=f'|S_{AX[i]}{AX[i]}|', ylim=(1e-3, 20),
              xlim=(3, 1e4), title=f'Sensitivity {AX[i]} -> {AX[i]}')
        a.legend(frameon=False, fontsize=8)
    a = ax[1, 0]
    for k, (i, j) in enumerate(((0, 1), (1, 0), (0, 2), (2, 0))):
        a.loglog(f[1:], np.abs(Sm['sliding'][:, 1:, i, j]).max(0), lw=1,
                 color=(spec.COLORS + ('#e87ba4',))[k], label=f'|S_{AX[i]}{AX[j]}| model max over Y')
    a.set(xlabel='frequency [Hz]', ylabel='|S_ij|', title='Off-diagonal sensitivity (sliding)',
          xlim=(3, 1e4))
    a.legend(frameon=False, fontsize=8)
    a = ax[1, 1]
    for i in range(3):
        a.semilogx(fe[1:], coh[1:, i], color=spec.COLORS[i], lw=0.8, label=AX[i])
    a.axhline(COH, color=spec.INK, ls=':', lw=1, label='threshold')
    a.set(xlabel='frequency [Hz]', ylabel='multiple coherence', ylim=(0, 1.02), xlim=(3, 1e4),
          title='Empirical S: coherence on the three feedforward inputs')
    a.legend(frameon=False, fontsize=8)
    a = ax[1, 2]
    r_ = pre_ff[0]
    meas = r_['e'] - r_['e'][:r_['a'] - 200].mean(0)
    seg = slice(r_['a'] - 200, len(meas))
    t = np.arange(len(meas[seg])) / spec.FS * 1e3
    a.plot(t, meas[seg][:, 0] * 1e9, color=spec.COLORS[0], lw=1.2, label='measured X1')
    for rd, ls in (('sliding', '-'), ('tanh', 'dashed')):
        A_, B_, C_, D_ = sysd[rd][r_['op']]
        em = loop.lsim(A_, B_, C_, D_, r_['u'][seg])
        a.plot(t, em[:, 0] * 1e9, color=spec.INK, ls=ls, lw=1, label=f'model {rd}')
    a.axhline(0, color='#c3c2b7', lw=0.8, label='stuck (G = 0)')
    a.set(xlabel='time from 10 ms before feedforward onset [ms]', ylabel='error [nm]',
          title=f'Pre-motion feedforward response, {r_["name"]}')
    a.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(OUT, 'g2_sensitivity.png'), dpi=110)


if __name__ == '__main__':
    main()
