"""G1: standstill error spectra from every log's pre-motion window + non-repeatable motion spectrum.

    python measure/g1_spectra.py
Pre-registered in CN-004. Writes outputs/g1_spectra/{g1.npz, g1.json, g1_spectra.png}. Summaries
only (data access policy): no raw samples are stored, only spectra and per-record statistics.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cn_env                                                           # noqa: E402

import numpy as np                                                      # noqa: E402
from scipy import stats                                                 # noqa: E402

from measure import spec                                                # noqa: E402
cn_env.check_no_leak()

OUT = cn_env.out_dir('g1_spectra')
Q_STEP = 9.765625e-10                     # m, encoder step (Telica 1.mat, 09-22 handoff)
FS = spec.FS
AX = cn_env.AX
N_BOOT = 1000
RNG = np.random.default_rng(20260924)


def main():
    t0 = time.time()
    recs = spec.all_records()
    print(f'[g1] {len(recs)} candidate records')
    names, Psum, Kr, var_r, nsamp, xy, qratio, ff_trunc, skipped = [], [], [], [], [], [], [], 0, []
    ff_info = []
    ref_bad = 0
    motion = {'iter0': [], 'iterETEL': []}
    f = None
    for n_, (s, op, it, p) in enumerate(recs):
        try:
            d = spec.load(p)
        except Exception as exc:
            skipped.append(f'{s}/{op}/{it}: {type(exc).__name__}')
            print(f'[g1] SKIP unreadable {s}/{op}/{it}: {type(exc).__name__}')
            continue
        e, j0 = d["e"], d["j0"]
        if it in motion:
            motion[it].append(e[j0:].copy())
        end = j0 - spec.GUARD
        nz = np.where(np.any(d["u_ff"][:j0] != 0.0, axis=1))[0]
        if len(nz):
            ff_trunc += 1
            ff_info.append(dict(name=f"{s}/{op}/{it}", lead_samples=int(j0 - nz[0]),
                                max_abs_N=np.abs(d["u_ff"][nz[0]:j0]).max(0).round(4).tolist()))
            end = min(end, int(nz[0]) - spec.GUARD)
        if end < spec.NPER:
            skipped.append(f'{s}/{op}/{it}: window {end} < {spec.NPER}')
            continue
        if np.any(d['r'][:end] != d['r'][0]):
            ref_bad += 1
        w = e[:end]
        q = end // 4
        rq = np.std(w[:q], 0) / np.maximum(np.std(w[-q:], 0), 1e-30)
        f, P, K = spec.welch_sum(w)
        names.append(f'{s}/{op}/{it}')
        Psum.append(P); Kr.append(K)
        var_r.append(w.var(0)); nsamp.append(end)
        xy.append(d['r'][0].copy()); qratio.append(rq)
        if n_ % 20 == 0:
            print(f'[g1] {n_ + 1}/{len(recs)} {s}/{op}/{it}: window {end} samples, K {K}, '
                  f'rms {np.round(np.sqrt(w.var(0)) * 1e9, 2)} nm ({time.time() - t0:.0f} s)')
    Psum = np.stack(Psum); Kr = np.array(Kr); var_r = np.stack(var_r)
    nsamp = np.array(nsamp); xy = np.stack(xy); qratio = np.stack(qratio)
    rms_r = np.sqrt(var_r)
    med = np.median(rms_r, 0)
    bad_q = np.any((qratio > 2) | (qratio < 0.5), 1)
    bad_h = np.any(rms_r > 3 * med, 1)
    keep = ~(bad_q | bad_h)
    excluded = [dict(name=names[i], qratio=qratio[i].round(3).tolist(),
                     rms_nm=(rms_r[i] * 1e9).round(3).tolist()) for i in np.where(~keep)[0]]
    print(f'[g1] {len(names)} usable records; feedforward nonzero in window: {ff_trunc}; '
          f'reference not constant: {ref_bad}; excluded by screen: {len(excluded)}')
    for x in excluded:
        print(f'[g1]   excluded {x}')
    leads = np.array([x["lead_samples"] for x in ff_info])
    by_it = {}
    for x in ff_info:
        by_it.setdefault(x["name"].split("/")[-1], []).append(x["lead_samples"])
    print("[g1] feedforward lead before j0 [samples] by iteration: " + ", ".join(
        f"{k}: n {len(v)} min {min(v)} med {int(np.median(v))} max {max(v)}" for k, v in sorted(by_it.items())))
    print(f"[g1] skipped: {skipped}")

    # ==== pooled spectra ====================================================================
    K = int(Kr[keep].sum())
    Ppool = Psum[keep].sum(0) / K
    A = spec.autos(Ppool)
    nu = spec.nu_welch(K)
    lo, hi = spec.chi2_ci(A, nu)
    df = f[1] - f[0]
    rms_psd = np.sqrt(A.sum(0) * df)
    rms_td = np.sqrt((Kr[keep, None] * var_r[keep]).sum(0) / K)
    rms_ts = np.sqrt((nsamp[keep, None] * var_r[keep]).sum(0) / nsamp[keep].sum())
    ratio = rms_psd / rms_td - 1
    print(f'[g1] pooled K = {K} segments, nu = {nu:.0f}, per-bin 95 % CI factor '
          f'[{lo[5, 0] / A[5, 0]:.3f}, {hi[5, 0] / A[5, 0]:.3f}]')
    print(f'[g1] rms from PSD {np.round(rms_psd * 1e9, 3)} nm; time-domain (segment-weighted) '
          f'{np.round(rms_td * 1e9, 3)}; sample-weighted {np.round(rms_ts * 1e9, 3)}; '
          f'ratio-1 {np.round(ratio * 100, 3)} %')

    # ==== coherence ===========================================================================
    coh = np.zeros((len(f), 3, 3))
    for i in range(3):
        for j in range(3):
            coh[:, i, j] = np.abs(Ppool[:, i, j]) ** 2 / np.maximum(A[:, i] * A[:, j], 1e-300)
    thr = 1 - 0.01 ** (1 / (nu / 2 - 1))

    # ==== bands and bootstrap ===================================================================
    band_of, fc, fe = spec.bands(f)
    nb = len(fc)
    Pb_r = np.stack([spec.band_mean(spec.autos(Psum[i]), band_of, nb) for i in range(len(names))])
    idx = np.where(keep)[0]
    band_val = Pb_r[idx].sum(0) / K
    boot = np.empty((N_BOOT, nb, 3))
    for b in range(N_BOOT):
        s_ = RNG.choice(idx, size=len(idx), replace=True)
        boot[b] = Pb_r[s_].sum(0) / Kr[s_].sum()
    b_lo, b_hi = np.percentile(boot, [2.5, 97.5], axis=0)
    nbins = np.array([(band_of == b).sum() for b in range(nb)])
    c_lo, c_hi = spec.chi2_ci(band_val, nu * nbins[:, None] / 1.5)   # ENBW of Hann = 1.5 bins
    half_boot = (b_hi - b_lo) / 2 / band_val
    print(f'[g1] {nb} bands {fe[0, 0]:.1f}-{fe[-1, 1]:.0f} Hz; bootstrap 95 % CI half-width '
          f'median {np.median(half_boot) * 100:.1f} % (max {half_boot.max() * 100:.1f} %)')

    # ==== non-repeatable motion spectrum (telica-real G5 M1 split) ============================
    nr = {}
    for it, segs in motion.items():
        M = len(segs)
        Lc = min(len(x) for x in segs)
        X = np.fft.rfft(np.stack([x[:Lc] for x in segs]), axis=1)          # (M, F, 3)
        s2 = (np.abs(X - X.mean(0)) ** 2).sum(0) / (M - 1)
        Pnr = 2 * s2 / (FS * Lc)
        Pnr[0] /= 2
        fm = np.fft.rfftfreq(Lc, 1 / FS)
        rms_nr = np.sqrt(Pnr.sum(0) * (fm[1] - fm[0]))
        dev = np.stack([x[:Lc] for x in segs]); dev = dev - dev.mean(0)
        rms_nr_t = np.sqrt((dev ** 2).sum(0).sum(0) / ((M - 1) * Lc))
        bo, fcm, _ = spec.bands(fm)
        nr[it] = dict(M=M, Lc=Lc, f=fm, P=Pnr, rms=rms_nr, rms_t=rms_nr_t,
                      band=spec.band_mean(Pnr, bo, len(fcm)), fc=fcm)
        print(f'[g1] non-repeatable motion {it}: M {M}, window {Lc / FS * 1e3:.0f} ms, rms '
              f'{np.round(rms_nr * 1e9, 1)} nm (time domain {np.round(rms_nr_t * 1e9, 1)})')

    # ==== summaries =========================================================================
    qfloor = Q_STEP ** 2 / 12 / (FS / 2)
    split_bands = [(10, 100), (100, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 5000),
                   (5000, 10000)]
    band_rms = {f'{a}-{b}': (np.sqrt(A[(f >= a) & (f < b)].sum(0) * df) * 1e9).round(3).tolist()
                for a, b in split_bands}
    frac_above_200 = (A[f >= 200].sum(0) / A.sum(0)).round(3).tolist()
    ok = bool(np.all(np.abs(ratio) < 0.02))
    summary = dict(
        n_candidates=len(recs), n_usable=len(names), n_pooled=int(keep.sum()), skipped=skipped,
        excluded=excluded, ff_nonzero_before_motion=ff_trunc, ff_info=ff_info, ref_not_constant=ref_bad,
        K=K, nu=nu, rms_psd_nm=(rms_psd * 1e9).tolist(), rms_td_nm=(rms_td * 1e9).tolist(),
        rms_sample_weighted_nm=(rms_ts * 1e9).tolist(), ratio_minus_1=ratio.tolist(),
        per_record_rms_nm=dict(median=(med * 1e9).tolist(),
                               p10=(np.percentile(rms_r, 10, 0) * 1e9).tolist(),
                               p90=(np.percentile(rms_r, 90, 0) * 1e9).tolist()),
        band_rms_nm=band_rms, var_fraction_above_200Hz=frac_above_200,
        quant_floor_m2Hz=qfloor, psd_max_over_quant_dB=(10 * np.log10(A[1:].max(0) / qfloor)).tolist(),
        coherence_null_threshold_1pct=thr,
        coherence_max={f'{AX[i]}-{AX[j]}': float(coh[1:, i, j].max())
                       for i, j in ((0, 1), (0, 2), (1, 2))},
        coherence_var_weighted={f'{AX[i]}-{AX[j]}': float(
            (coh[1:, i, j] * np.sqrt(A[1:, i] * A[1:, j])).sum() / np.sqrt(A[1:, i] * A[1:, j]).sum())
            for i, j in ((0, 1), (0, 2), (1, 2))},
        boot_ci_halfwidth_median=float(np.median(half_boot)),
        nonrep={it: dict(M=v['M'], window_ms=v['Lc'] / FS * 1e3, rms_nm=(v['rms'] * 1e9).tolist(),
                         rms_time_nm=(v['rms_t'] * 1e9).tolist()) for it, v in nr.items()},
        PASS=ok)
    print(f'[g1] band rms [nm]: {band_rms}')
    print(f'[g1] variance fraction above 200 Hz: {frac_above_200}')
    print(f'[g1] coherence max {summary["coherence_max"]}, variance-weighted '
          f'{summary["coherence_var_weighted"]} (null 1 % threshold {thr:.4f})')
    print(f'[g1] PASS (rms closure < 2 %): {ok}')
    json.dump(summary, open(os.path.join(OUT, 'g1.json'), 'w'), indent=1)
    np.savez_compressed(os.path.join(OUT, 'g1.npz'), f=f, Ppool=Ppool, Psum=Psum, Kr=Kr,
                        keep=keep, names=np.array(names), xy=xy, var_r=var_r, lo=lo, hi=hi,
                        nu=nu, band_of=band_of, fc=fc, fe=fe, band_val=band_val, b_lo=b_lo,
                        b_hi=b_hi, c_lo=c_lo, c_hi=c_hi, Pb_r=Pb_r, coh=coh,
                        nr_f0=nr['iter0']['f'], nr_P0=nr['iter0']['P'],
                        nr_fE=nr['iterETEL']['f'], nr_PE=nr['iterETEL']['P'])
    plot(f, A, lo, hi, fc, band_val, b_lo, b_hi, coh, thr, nr, qfloor)
    print(f'[g1] done in {time.time() - t0:.0f} s')


def plot(f, A, lo, hi, fc, bv, blo, bhi, coh, thr, nr, qfloor):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    a = ax[0, 0]
    for j in range(3):
        a.fill_between(f[1:], lo[1:, j], hi[1:, j], color=spec.COLORS[j], alpha=0.25, lw=0)
        a.loglog(f[1:], A[1:, j], color=spec.COLORS[j], lw=1.0, label=AX[j])
    a.axhline(qfloor, color=spec.INK, lw=1, ls=':', label='quantisation q^2/12 / (fs/2)')
    a.set(xlabel='frequency [Hz]', ylabel='PSD [m^2/Hz]',
          title='Standstill error PSD, all records pooled (95 % chi-square CI)')
    a.legend(frameon=False)
    a = ax[0, 1]
    for j in range(3):
        a.fill_between(fc, blo[:, j], bhi[:, j], color=spec.COLORS[j], alpha=0.25, lw=0)
        a.loglog(fc, bv[:, j], color=spec.COLORS[j], lw=1.5, marker='o', ms=3, label=AX[j])
    a.set(xlabel='band centre [Hz]', ylabel='band-mean PSD [m^2/Hz]',
          title='Band means with record-bootstrap 95 % CI (used by G3-G5)')
    a.legend(frameon=False)
    a = ax[1, 0]
    for k, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
        a.semilogx(f[1:], coh[1:, i, j], color=spec.COLORS[k], lw=0.8, label=f'{AX[i]}-{AX[j]}')
    a.axhline(thr, color=spec.INK, lw=1, ls=':', label='1 % null threshold')
    a.set(xlabel='frequency [Hz]', ylabel='magnitude-squared coherence', ylim=(0, 1),
          title='Cross-axis coherence at standstill')
    a.legend(frameon=False)
    a = ax[1, 1]
    v = nr['iter0']
    for j in range(3):
        a.loglog(v['fc'], v['band'][:, j], color=spec.COLORS[j], lw=1.5, label=f'{AX[j]} motion, non-repeatable')
        a.loglog(fc, bv[:, j], color=spec.COLORS[j], lw=1, ls='dashed')
    a.set(xlabel='frequency [Hz]', ylabel='band-mean PSD [m^2/Hz]',
          title='Non-repeatable motion (iter0, solid) vs standstill (dashed)')
    a.legend(frameon=False)
    fig.savefig(os.path.join(OUT, 'g1_spectra.png'), dpi=120)


if __name__ == '__main__':
    main()
