"""G3 attempt 3 (TR-011): identify the logged path M2 -> MF230 in the zpk's own structure.

Output-error least squares on TRAIN iter0 records, initialised at the zpk file (integrator fixed
at z = 1, stable pole parameterization), then the TR-009 (i) test on HELD-OUT val + test iter0.
Writes controller/telica_sos_identified.npz (the identified controller, derived constants) and
outputs/g3_identify/{identify.json, frf_identified.png}. Prints summaries only.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tr_env                                                          # noqa: E402

import numpy as np                                                     # noqa: E402
from scipy.optimize import least_squares                               # noqa: E402
from scipy.signal import sosfilt, sosfreqz, csd, welch                 # noqa: E402

import data_index                                                      # noqa: E402
from controller import telica_ctrl as tc                               # noqa: E402
from real_data_verification.telica_loader import load_telica_log_full  # noqa: E402
tr_env.check_no_leak()

OUT = tr_env.out_dir('g3_identify')
AX = ('X1', 'X2', 'Y')
K0 = 1000                 # HEURISTIC: 50 ms; the non-integrator controller states decay in < 25 ms
LAGS = range(-3, 4)


# ================================================================ parameterization
class SosParam:
    """Stable parameterization of a cascade of biquads with a fixed integrator."""

    def __init__(self, sos):
        self.spec = []          # per section: ('c', ) complex pair or ('r', fixed_mask) real pair
        theta = []
        for s, sec in enumerate(sos):
            b0, b1, b2, _, a1, a2 = sec / sec[3]
            if s == 0:
                theta += [b0, b1, b2]
            else:
                theta += [b1 / b0, b2 / b0]
            roots = np.roots([1.0, a1, a2]) if abs(a2) > 0 else np.array([-a1, 0.0])
            if abs(roots[0].imag) > 1e-12:
                r = abs(roots[0]); th = abs(np.angle(roots[0]))
                self.spec.append(('c',))
                theta += [np.log(1 - r), th]
            else:
                ps = np.sort(roots.real)[::-1]
                fixed = [bool(abs(p - 1.0) < 1e-12) for p in ps]
                self.spec.append(('r', fixed, [float(p) for p in ps]))
                for p, fx in zip(ps, fixed):
                    if not fx:
                        theta.append(np.arctanh(np.clip(p, -0.999999, 0.999999)))
        self.theta0 = np.array(theta, float)
        self.n_sec = len(sos)

    def sos(self, th):
        out = np.zeros((self.n_sec, 6))
        i = 0
        for s in range(self.n_sec):
            if s == 0:
                b = th[i:i + 3]; i += 3
            else:
                b = np.array([1.0, th[i], th[i + 1]]); i += 2
            sp = self.spec[s]
            if sp[0] == 'c':
                r = 1 - np.exp(th[i]); w = th[i + 1]; i += 2
                a = [1.0, -2 * r * np.cos(w), r * r]
            else:
                ps = []
                for fx in sp[1]:
                    if fx:
                        ps.append(1.0)
                    else:
                        ps.append(np.tanh(th[i])); i += 1
                a = [1.0, -(ps[0] + ps[1]), ps[0] * ps[1]]
            out[s, :3] = b
            out[s, 3:] = a
        return out


def shift(x, d):
    """pred[k] = x[k - d] (zero-padded)."""
    if d == 0:
        return x
    y = np.zeros_like(x)
    if d > 0:
        y[d:] = x[:-d]
    else:
        y[:d] = x[-d:]
    return y


def residuals(th, prm, E, I, d):
    sos = prm.sos(th)
    res = []
    for e, i in zip(E, I):
        p = shift(sosfilt(sos, e), d)[K0:-3]
        y = i[K0:-3]
        r = y - p
        res.append(r - r.mean())          # per-record constant = integrator initial state
    return np.concatenate(res)


def zir(sos, T):
    A, B, C, D = tc.axis_ss(sos)
    Phi = np.empty((T, A.shape[0]))
    row = C.copy()
    for k in range(T):
        Phi[k] = row
        row = row @ A
    return Phi


def m1_gain(i, pred, Phi):
    X = np.column_stack([pred, Phi])
    c, *_ = np.linalg.lstsq(X, i, rcond=None)
    r = i - X @ c
    return float(c[0]), float(1 - r.var() / i.var())


def main():
    t0 = time.time()
    tr = data_index.records(splits=('train',), iters=('iter0',))
    ho = data_index.records(splits=('validation', 'test'), iters=('iter0',))
    load = lambda recs: [(op, load_telica_log_full(p)) for (_, op, _, p) in recs]   # noqa: E731
    TR, HO = load(tr), load(ho)
    print(f'[id] train iter0 {len(TR)} records, held-out iter0 {len(HO)} records')
    sos0 = tc.sos_bank()
    out = {'axes': {}}
    sos_hat = {}
    for j, ax in enumerate(AX):
        name = tc.AXES[j]
        E = [d['e'][:, j] for _, d in TR if d['i_fb'][:, j].std() > 0]
        I = [d['i_fb'][:, j] for _, d in TR if d['i_fb'][:, j].std() > 0]
        prm = SosParam(sos0[name])
        best = None
        for lag in LAGS:
            r0 = residuals(prm.theta0, prm, E, I, lag)
            sol = least_squares(residuals, prm.theta0, args=(prm, E, I, lag), x_scale='jac',
                                method='trf', max_nfev=400, xtol=1e-10, ftol=1e-10)
            y_all = np.concatenate([(i[K0:-3] - i[K0:-3].mean()) for i in I])
            vaf = 1 - np.mean(sol.fun ** 2) / y_all.var()
            vaf0 = 1 - np.mean(r0 ** 2) / y_all.var()
            print(f'[id] {ax} lag {lag:+d}: train VAF zpk {vaf0:.4f} -> identified {vaf:.4f} '
                  f'(nfev {sol.nfev}, status {sol.status})', flush=True)
            if best is None or vaf > best[1]:
                best = (lag, vaf, vaf0, sol.x)
        lag, vaf, vaf0, th = best
        sh = prm.sos(th)
        sos_hat[name] = sh
        # poles / zeros of the identified controller
        from scipy.signal import sos2zpk
        z, p, k = sos2zpk(sh)
        # ==== held-out TR-009 (i) with the identified replay ====
        g_r, v_r, g_r0, v_r0, varK = [], [], [], [], []
        for op, d in HO:
            e = d['e'][:, j]; i = d['i_fb'][:, j]
            T = len(e)
            pred = shift(sosfilt(sh, e), lag)
            g, v = m1_gain(i, pred, zir(sh, T))
            g0, v0 = m1_gain(i, sosfilt(sos0[name], e), zir(sos0[name], T))
            g_r.append(g); v_r.append(v); g_r0.append(g0); v_r0.append(v0); varK.append(pred.var())
        g_r = np.array(g_r)
        se = float(g_r.std(ddof=1) / np.sqrt(len(g_r)))
        rng = np.random.default_rng(0)
        dE = 1e-9                     # smallest logged M2 step (g3_replay: 1.00e-09 m, all axes)
        q = rng.uniform(-dE / 2, dE / 2, size=len(HO[0][1]['e']))
        s2 = float(sosfilt(sh, q).var())
        b = s2 / (np.mean(varK) + s2)
        delta = max(3 * se, b)
        # pooled gain over held-out records (projection on each record's ZIR removed)
        num = den = 0.0
        for op, d in HO:
            e = d['e'][:, j]; i = d['i_fb'][:, j]
            pred = shift(sosfilt(sh, e), lag)
            Phi = zir(sh, len(e))
            c, *_ = np.linalg.lstsq(Phi, pred, rcond=None)
            xp = pred - Phi @ c
            num += xp @ i; den += xp @ xp
        g_pool = float(num / den)
        ok = abs(g_pool - 1) <= delta
        out['axes'][ax] = dict(lag=lag, train_vaf_zpk=float(vaf0), train_vaf_id=float(vaf),
                               heldout_g_records=g_r.tolist(), heldout_g_pooled=g_pool,
                               heldout_se=se, b_eiv=b, delta=delta, PASS=bool(ok),
                               heldout_vaf_id=float(np.mean(v_r)),
                               heldout_vaf_zpk=float(np.mean(v_r0)),
                               heldout_g_zpk=[float(v) for v in g_r0],
                               poles=[[float(c.real), float(c.imag)] for c in p],
                               zeros=[[float(c.real), float(c.imag)] for c in z], k=float(k),
                               max_abs_pole=float(np.abs(p).max()))
        print(f'[id] {ax}: lag {lag:+d}, train VAF {vaf0:.4f} -> {vaf:.4f}; HELD-OUT: '
              f'g_pooled {g_pool:.4f} (records {np.round(g_r, 4).tolist()}), delta {delta:.2e} '
              f'-> {"PASS" if ok else "FAIL"}; VAF id {np.mean(v_r):.4f} vs zpk {np.mean(v_r0):.4f}; '
              f'max|pole| {np.abs(p).max():.8f}', flush=True)

    np.savez(os.path.join(tc.HERE, 'telica_sos_identified.npz'),
             **{k: v for k, v in sos_hat.items()},
             lags=np.array([out['axes'][a]['lag'] for a in AX]))
    # ==== held-out H1 flatness, identified vs zpk ====
    See = Sei = Sii = None
    for op, d in HO:
        k0 = d['motion_idx']
        e, i = d['e'][k0:], d['i_fb'][k0:]
        f, a = welch(e, fs=tc.tp.FS, nperseg=4096, axis=0)
        _, b_ = csd(e, i, fs=tc.tp.FS, nperseg=4096, axis=0)
        _, c = welch(i, fs=tc.tp.FS, nperseg=4096, axis=0)
        See = a if See is None else See + a
        Sei = b_ if Sei is None else Sei + b_
        Sii = c if Sii is None else Sii + c
    H = Sei / See
    coh = np.abs(Sei) ** 2 / (See * Sii)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    for j, ax in enumerate(AX):
        name = tc.AXES[j]
        _, K0f = sosfreqz(sos0[name], worN=f, fs=tc.tp.FS)
        _, Kh = sosfreqz(sos_hat[name], worN=f, fs=tc.tp.FS)
        lag = out['axes'][ax]['lag']
        Kh = Kh * np.exp(-2j * np.pi * f * lag / tc.tp.FS)
        m = (f >= 5) & (f <= 5000) & (coh[:, j] > 0.9)
        flat_id = float(np.median(np.abs(np.log(np.abs(H[m, j] / Kh[m])))))
        flat_zpk = float(np.median(np.abs(np.log(np.abs(H[m, j] / K0f[m])))))
        out['axes'][ax].update(heldout_flat_id=flat_id, heldout_flat_zpk=flat_zpk)
        print(f'[id] {ax}: held-out median |log(H/K)|: identified {flat_id:.3f}, zpk {flat_zpk:.3f}')
        axs[j, 0].loglog(f[1:], np.abs(H[1:, j]), 'k', lw=0.8, label='held-out logged e->i')
        axs[j, 0].loglog(f[1:], np.abs(K0f[1:]), 'C3', lw=0.8, label='zpk file')
        axs[j, 0].loglog(f[1:], np.abs(Kh[1:]), 'C0', lw=0.8, label='identified (train)')
        axs[j, 0].set_ylabel(f'{ax} |K| [A/m]')
        axs[j, 1].semilogx(f[1:], np.degrees(np.angle(H[1:, j])), 'k', lw=0.8)
        axs[j, 1].semilogx(f[1:], np.degrees(np.angle(K0f[1:])), 'C3', lw=0.8)
        axs[j, 1].semilogx(f[1:], np.degrees(np.angle(Kh[1:])), 'C0', lw=0.8)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle('G3 attempt 3: identified logged controller vs zpk, on held-out iter0')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'frf_identified.png'), dpi=110)
    out['PASS_i'] = bool(all(out['axes'][a]['PASS'] for a in AX))
    json.dump(out, open(os.path.join(OUT, 'identify.json'), 'w'), indent=1)
    print(f'[id] G3 (i) attempt 3: {"PASS" if out["PASS_i"] else "FAIL"}  ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
