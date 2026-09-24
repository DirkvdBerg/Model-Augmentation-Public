"""G3 (i): controller replay on every iter0 record vs the logged MF230 (TR-009).

Per record and axis: zero-state replay i_K of the LOGGED error, then
    M0  i_log = g0 i_K                    (the historic statistic, D-073: 1.16 / 1.33 / 3.55)
    M1  i_log = g1 i_K + Phi c            (Phi = the controller's zero-input response: its unknown
                                           state when the log starts; integrator -> constant)
plus: lag scan of M1, cross-axis regression, saturation, feedforward leakage, quantization, H1.
Prints summaries only (data policy). Writes outputs/g3_replay/g3_replay.json and PNGs.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tr_env                                                          # noqa: E402

import numpy as np                                                     # noqa: E402
from scipy.signal import csd, welch, sosfilt                           # noqa: E402

import data_index                                                      # noqa: E402
from controller import telica_ctrl as tc                               # noqa: E402
from real_data_verification.telica_loader import (load_telica_log_full,  # noqa: E402
                                                  load_telica_log_cl)
tr_env.check_no_leak()

OUT = tr_env.out_dir('g3_replay')
AX = ('X1', 'X2', 'Y')
IP = 27.9          # THEORY: datasheet p.3 peak current Ip [Arms], X and Y motors
LAGS = range(-100, 101)


def ls(X, y):
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ c
    return c, r


def perp(v, Phi):
    """v with its projection on span(Phi) removed."""
    c, *_ = np.linalg.lstsq(Phi, v, rcond=None)
    return v - Phi @ c


def main():
    t0 = time.time()
    recs = data_index.records(iters=('iter0',))
    print(f'[g3] {len(recs)} iter0 records')
    sos = tc.sos_bank()
    _, _, snapped = tc.load_zpk()
    print('[g3] snapped integrator poles (|p| before snap):', snapped)
    res = {'records': [], 'snapped': snapped}
    pooled_num = np.zeros(3); pooled_den = np.zeros(3)
    S_xy = None; S_xx = None; S_yy = None; nseg = 0
    Phi_cache = {}
    for (split, op, it, p) in recs:
        d = load_telica_log_full(p)
        e, i_log, i_tot = d['e'], d['i_fb'], d['i_tot']
        T = len(e)
        if T not in Phi_cache:
            Phi_cache = {T: tc.zir_basis(T)}
        Phi = Phi_cache[T]
        iK = tc.replay(e)
        rec = dict(split=split, op=op, T=T, motion_idx=int(d['motion_idx']),
                   ff_leak_max_A=float(np.abs(i_tot - i_log).max()),
                   std_i_A=i_log.std(0).tolist(), std_e_m=e.std(0).tolist())
        if not (i_log.std(0) > 0).all() or not (e.std(0) > 0).all():
            print(f'[g3] DEGENERATE record {split}/{op}: std(MF230) = {i_log.std(0)} A, '
                  f'std(e) = {e.std(0)} m, T = {T}; excluded from the gain statistics', flush=True)
            rec['degenerate'] = True
            res.setdefault('degenerate', []).append(rec)
            continue
        for j, ax in enumerate(AX):
            y = i_log[:, j]; x = iK[:, j]; Pj = Phi[tc.AXES[j]]
            g0 = float(x @ y / (x @ x))
            corr = float(np.corrcoef(x, y)[0, 1])
            X1 = np.column_stack([x, Pj])
            c1, r1 = ls(X1, y)
            vaf1 = float(1 - r1.var() / y.var())
            # lag scan (d > 0: logged current lags the replay)
            best = (None, -np.inf, None)
            for lag in LAGS:
                if lag >= 0:
                    yy, xx, PP = y[lag:], x[:T - lag], Pj[lag:]
                else:
                    yy, xx, PP = y[:T + lag], x[-lag:], Pj[:T + lag]
                c, r = ls(np.column_stack([xx, PP]), yy)
                v = 1 - r.var() / yy.var()
                if v > best[1]:
                    best = (lag, v, float(c[0]))
            # cross-axis
            Xc = np.column_stack([iK, Pj])
            cc_, _ = ls(Xc, y)
            off = [k for k in range(3) if k != j]
            share = float(sum(abs(cc_[k]) * iK[:, k].std() for k in off)
                          / (abs(cc_[j]) * iK[:, j].std()))
            # quantization steps and saturation
            m2 = d['M2_um'][:, j]
            dm2 = np.abs(np.diff(m2)); dm2 = dm2[dm2 > 0]
            di = np.abs(np.diff(y)); di = di[di > 0]
            ymax = float(np.abs(y).max())
            plateau = float(np.mean(np.abs(y) >= 0.999 * ymax))
            xp = perp(x, Pj)
            pooled_num[j] += xp @ y
            pooled_den[j] += xp @ xp
            rec[ax] = dict(g0=g0, corr=corr, g1=float(c1[0]), vaf1=vaf1,
                           c_int_A=float(Pj[-1] @ c1[1:]),
                           best_lag=best[0], vaf_lag=float(best[1]), g_lag=best[2],
                           cross_gains=[float(v) for v in cc_[:3]], off_share=share,
                           dM2_min_um=float(dm2.min()) if len(dm2) else None,
                           di_min_A=float(di.min()) if len(di) else None,
                           max_abs_i_A=ymax, plateau_frac=plateau,
                           var_iK=float(x.var()))
        # H1 over records (Welch), motion part only
        k0 = rec['motion_idx']
        f, sxy = csd(iK[k0:], i_log[k0:], fs=tc.tp.FS, nperseg=2048, axis=0)
        _, sxx = welch(iK[k0:], fs=tc.tp.FS, nperseg=2048, axis=0)
        _, syy = welch(i_log[k0:], fs=tc.tp.FS, nperseg=2048, axis=0)
        S_xy = sxy if S_xy is None else S_xy + sxy
        S_xx = sxx if S_xx is None else S_xx + sxx
        S_yy = syy if S_yy is None else S_yy + syy
        nseg += 1
        res['records'].append(rec)
        print(f'[g3] {split:10s} {op:18s} T={T:6d} ff_leak={rec["ff_leak_max_A"]:.1e} A  '
              + '  '.join(f'{ax}: g0={rec[ax]["g0"]:.3f} g1={rec[ax]["g1"]:.4f} '
                          f'vaf1={rec[ax]["vaf1"]:.4f} lag={rec[ax]["best_lag"]:+d}'
                          for ax in AX), flush=True)

    # ==== pooled gains, interval, EIV bound ====
    R = res['records']
    n_r = len(R)
    summ = {}
    rng = np.random.default_rng(0)
    for j, ax in enumerate(AX):
        g_r = np.array([r[ax]['g1'] for r in R])
        g_pool = float(pooled_num[j] / pooled_den[j])
        se = float(g_r.std(ddof=1) / np.sqrt(n_r))
        dE = min(r[ax]['dM2_min_um'] for r in R if r[ax]['dM2_min_um']) * 1e-6
        # s2: variance of K applied to uniform quantization noise of step dE, record length
        T = R[0]['T']
        q = rng.uniform(-dE / 2, dE / 2, size=(T, 3))
        s2 = float(sosfilt(sos[tc.AXES[j]], q[:, j]).var())
        var_iK = float(np.mean([r[ax]['var_iK'] for r in R]))
        b = s2 / (var_iK + s2)
        delta = max(3 * se, b)
        summ[ax] = dict(g_pooled=g_pool, g_rec_mean=float(g_r.mean()), g_rec_std=float(g_r.std(ddof=1)),
                        g_rec_min=float(g_r.min()), g_rec_max=float(g_r.max()), se=se,
                        dE_m=dE, s2_A2=s2, b_eiv=b, delta=delta,
                        g0_mean=float(np.mean([r[ax]['g0'] for r in R])),
                        vaf1_mean=float(np.mean([r[ax]['vaf1'] for r in R])),
                        corr_mean=float(np.mean([r[ax]['corr'] for r in R])),
                        lag_mode=int(np.bincount(np.array([r[ax]['best_lag'] for r in R]) + 100).argmax() - 100),
                        g_lag_mean=float(np.mean([r[ax]['g_lag'] for r in R])),
                        off_share_max=float(max(r[ax]['off_share'] for r in R)),
                        max_abs_i_A=float(max(r[ax]['max_abs_i_A'] for r in R)),
                        plateau_max=float(max(r[ax]['plateau_frac'] for r in R)),
                        di_min_A=float(min(r[ax]['di_min_A'] for r in R if r[ax]['di_min_A'])),
                        PASS_i=bool(abs(g_pool - 1) <= delta))
    res['summary'] = summ
    res['ff_leak_max_A'] = float(max(r['ff_leak_max_A'] for r in R))

    # ==== the historic statistic on the trimmed record, for continuity with D-073 ====
    dcl = load_telica_log_cl(data_index.path('train', 'xpos_-60_ypos-40'))
    e_tr = (dcl['r'] - dcl['q1']).numpy(); i_tr = dcl['i_fb'].numpy()
    iK_tr = tc.replay(e_tr)
    res['historic_trimmed_g0'] = [float(iK_tr[:, j] @ i_tr[:, j] / (iK_tr[:, j] @ iK_tr[:, j]))
                                  for j in range(3)]

    # ==== H1 ====
    H1 = S_xy / S_xx
    coh = np.abs(S_xy) ** 2 / (S_xx * S_yy)
    bands = [(1, 10), (10, 50), (50, 150), (150, 400), (400, 1000), (1000, 3000), (3000, 10000)]
    res['H1_bands'] = {}
    for j, ax in enumerate(AX):
        rows = []
        for lo, hi in bands:
            m = (f >= lo) & (f < hi)
            rows.append(dict(band=[lo, hi], absH1=float(np.median(np.abs(H1[m, j]))),
                             phase_deg=float(np.degrees(np.median(np.angle(H1[m, j])))),
                             coh=float(np.median(coh[m, j]))))
        res['H1_bands'][ax] = rows

    json.dump(res, open(os.path.join(OUT, 'g3_replay.json'), 'w'), indent=1)
    print()
    print('historic g0 on the trimmed xpos_-60_ypos-40 record:',
          ['%.3f' % v for v in res['historic_trimmed_g0']], '(D-073: 1.16 / 1.33 / 3.55)')
    print(f'feedforward leakage max |MF30 - MF230| over all iter0: {res["ff_leak_max_A"]:.3e} A')
    for ax in AX:
        s = summ[ax]
        print(f'{ax}: g_pooled={s["g_pooled"]:.4f} (records {s["g_rec_min"]:.4f}..{s["g_rec_max"]:.4f}, '
              f'se {s["se"]:.2e}), g0_mean={s["g0_mean"]:.3f}, vaf1_mean={s["vaf1_mean"]:.4f}, '
              f'corr_mean={s["corr_mean"]:.4f}, lag_mode={s["lag_mode"]:+d} (g_lag {s["g_lag_mean"]:.4f}), '
              f'off_share_max={s["off_share_max"]:.3e}, max|i|={s["max_abs_i_A"]:.2f} A (Ip {IP}), '
              f'plateau={s["plateau_max"]:.2e}, dE={s["dE_m"]:.2e} m, di={s["di_min_A"]:.2e} A, '
              f'b_eiv={s["b_eiv"]:.2e}, delta={s["delta"]:.2e}  -> (i) {"PASS" if s["PASS_i"] else "FAIL"}')
    for ax in AX:
        print(f'H1 {ax}: ' + '  '.join(f'{r["band"][0]}-{r["band"][1]}Hz |H1|={r["absH1"]:.3f} '
                                        f'ph={r["phase_deg"]:+.1f} coh={r["coh"]:.3f}'
                                        for r in res['H1_bands'][ax]))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for j, ax in enumerate(AX):
        axs[j].semilogx(f[1:], np.abs(H1[1:, j]), 'k', lw=0.8, label='|H1| logged / replay')
        axs[j].semilogx(f[1:], coh[1:, j], 'C1', lw=0.8, label='coherence')
        axs[j].axhline(1, color='grey', lw=0.5)
        axs[j].set_ylim(0, 2); axs[j].set_ylabel(ax)
    axs[0].legend(fontsize=8); axs[-1].set_xlabel('f [Hz]')
    fig.suptitle('G3 iter0: logged MF230 vs replay of logged error (all records, motion part)')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'g3_H1.png'), dpi=110)
    print(f'[g3] done in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
