"""TR-022 (A): the logged controller at 5 kHz.

Init: matched pole-zero map of the 20 kHz identified controller (p -> p^4, z -> z^4, gain matched at
100 Hz, integrator stays at z = 1). Refined by the TR-011 output-error fit on decimated TRAIN iter0
(decimated error -> block-mean MF230), lag in {-1, 0, 1}. Held-out val + test iter0: pooled gain
within 1 +- delta (TR-009 construction). Writes controller/telica_sos_identified_5k.npz and
outputs/g3_5k/identify_5k.json.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tr_env                                                          # noqa: E402

import numpy as np                                                     # noqa: E402
from scipy.optimize import least_squares                               # noqa: E402
from scipy.signal import sosfilt, sosfreqz, sos2zpk, zpk2sos           # noqa: E402

import data_index                                                      # noqa: E402
import rate                                                            # noqa: E402
from controller import telica_ctrl as tc                               # noqa: E402
from controller.identify_logged import SosParam, shift, zir, m1_gain   # noqa: E402
from real_data_verification.telica_loader import load_telica_log_full  # noqa: E402
tr_env.check_no_leak()

OUT = tr_env.out_dir('g3_5k')
AX = ('X1', 'X2', 'Y')
K0 = 250                     # 50 ms at 5 kHz, as TR-011's 1000 samples at 20 kHz (HEURISTIC)
LAGS = (-1, 0, 1)


def load5(recs):
    out = []
    for (_, op, _, p) in recs:
        d = load_telica_log_full(p, engine='c')
        e5 = rate.dec_pos(d['r']) - rate.dec_pos(d['q1'])
        i5 = rate.dec_force(d['i_fb'])
        out.append((op, e5, i5))
    return out


def mapped_init(sos20):
    z, p, k = sos2zpk(sos20)
    z5, p5 = z ** rate.D, p ** rate.D
    p5 = np.where(np.abs(p5 - 1.0) < 1e-12, 1.0 + 0j, p5)
    sos5 = zpk2sos(z5, p5, 1.0, pairing='nearest')
    f = np.array([100.0])
    _, h20 = sosfreqz(sos20, worN=f, fs=rate.tp.FS)
    _, h5 = sosfreqz(sos5, worN=f, fs=rate.FS5)
    sos5[0, :3] *= abs(h20[0]) / abs(h5[0])
    return sos5


def residuals(th, prm, E, I, d):
    sos = prm.sos(th)
    res = []
    for e, i in zip(E, I):
        r = i[K0:-2] - shift(sosfilt(sos, e), d)[K0:-2]
        res.append(r - r.mean())
    return np.concatenate(res)


def main():
    t0 = time.time()
    TR = load5(data_index.records(splits=('train',), iters=('iter0',)))
    HO = load5(data_index.records(splits=('validation', 'test'), iters=('iter0',)))
    z20 = np.load(os.path.join(tc.HERE, 'telica_sos_identified.npz'))
    print(f'[5k] train iter0 {len(TR)}, held-out {len(HO)} records at {rate.FS5:.0f} Hz')
    out, sos_hat, lags = {'axes': {}}, {}, []
    for j, ax in enumerate(AX):
        name = tc.AXES[j]
        E = [e[:, j] for _, e, i in TR if i[:, j].std() > 0]
        I = [i[:, j] for _, e, i in TR if i[:, j].std() > 0]
        prm = SosParam(mapped_init(z20[name]))
        y_all = np.concatenate([(i[K0:-2] - i[K0:-2].mean()) for i in I])
        best = None
        for lag in LAGS:
            r0 = residuals(prm.theta0, prm, E, I, lag)
            sol = least_squares(residuals, prm.theta0, args=(prm, E, I, lag), x_scale='jac',
                                method='trf', max_nfev=400, xtol=1e-10, ftol=1e-10)
            vaf = 1 - np.mean(sol.fun ** 2) / y_all.var()
            vaf0 = 1 - np.mean(r0 ** 2) / y_all.var()
            print(f'[5k] {ax} lag {lag:+d}: train VAF mapped init {vaf0:.4f} -> refined {vaf:.4f}',
                  flush=True)
            if best is None or vaf > best[1]:
                best = (lag, vaf, vaf0, sol.x)
        lag, vaf, vaf0, th = best
        sh = prm.sos(th)
        sos_hat[name] = sh; lags.append(lag)
        g_r, v_r, varK = [], [], []
        num = den = 0.0
        for _, e, i in HO:
            ee, ii = e[:, j], i[:, j]
            pred = shift(sosfilt(sh, ee), lag)
            Phi = zir(sh, len(ee))
            g, v = m1_gain(ii, pred, Phi)
            g_r.append(g); v_r.append(v); varK.append(pred.var())
            c, *_ = np.linalg.lstsq(Phi, pred, rcond=None)
            xp = pred - Phi @ c
            num += xp @ ii; den += xp @ xp
        g_r = np.array(g_r)
        se = float(g_r.std(ddof=1) / np.sqrt(len(g_r)))
        q = np.random.default_rng(0).uniform(-0.5e-9, 0.5e-9, size=len(HO[0][1]))
        s2 = float(sosfilt(sh, q).var())
        b = s2 / (np.mean(varK) + s2)
        delta = max(3 * se, b)
        g_pool = float(num / den)
        ok = abs(g_pool - 1) <= delta
        p = sos2zpk(sh)[1]
        out['axes'][ax] = dict(lag=lag, train_vaf_init=float(vaf0), train_vaf=float(vaf),
                               heldout_g_pooled=g_pool, heldout_g_records=g_r.tolist(),
                               delta=delta, heldout_vaf=float(np.mean(v_r)), PASS=bool(ok),
                               max_abs_pole=float(np.abs(p).max()))
        print(f'[5k] {ax}: lag {lag:+d}; HELD-OUT pooled gain {g_pool:.4f} (records '
              f'{np.round(g_r, 4).tolist()}), delta {delta:.2e} -> {"PASS" if ok else "FAIL"}; '
              f'held-out VAF {np.mean(v_r):.4f}; max|pole| {np.abs(p).max():.8f}', flush=True)
    np.savez(os.path.join(tc.HERE, 'telica_sos_identified_5k.npz'), **sos_hat,
             lags=np.array(lags), ts=rate.TS5)
    out['PASS'] = bool(all(v['PASS'] for v in out['axes'].values()))
    json.dump(out, open(os.path.join(OUT, 'identify_5k.json'), 'w'), indent=1)
    print(f'[5k] TR-022 (A): {"PASS" if out["PASS"] else "FAIL"} ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
