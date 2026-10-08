"""Window memory of the closed loop, methods A and B (JH-013, D-220).

h = 20 kHz impulse response of PS = (I + G K1)^-1 G (logical force injection -> stage position): the filter
through which a force residual of the model reaches the servo error in the residual rollout
(model_augmentation/fit_systems/closed_loop.py, controller state zero at the window start).
  A-full : discrete closed-loop state space of the linearised thesis truth (absorber, no friction) with K1.
  A-band : the analytic PS on the measured lines, inverted exactly like B.
  B      : the measured closed-loop BLA of the Coulomb + absorber truth under K1 (j2_bla.npz, B6 to B10).
Metric: window blindness B(T), the share of the free-run error energy a window of length T does not see.
Prints the tables and writes outputs/wm_ab/window_memory.json; nothing else is written.
"""
__project_origin__ = "added"

import json
import os
import sys
import time

import numpy as np
from scipy.linalg import block_diag
from scipy.signal import butter, cont2discrete, fftconvolve, lfilter, tf2ss

HERE = os.path.dirname(os.path.abspath(__file__))
JH = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(JH, '..', '..', '..'))
ECL = os.path.join(REPO, 'scripts', 'gantry', 'excitation-closed-loop')
sys.path.insert(0, ECL)
import common as C  # noqa: E402  (read only)
sys.path.pop(0)

OUT = os.path.join(JH, 'outputs', 'wm_ab')
TS = C.TS
YS = [-0.30, -0.15, 0.0, 0.15, 0.30]
DF = 1.5                     # line spacing per input column in j2_bla (zippered 0.5 Hz grid, 3 columns)
T_PER = 1.0 / DF             # period of the line inversion [s]
T_MAX_FULL, T_MAX_BAND = 1.6, 0.6
T_READ = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
LG = int(round(0.4 / TS))    # shaping-filter impulse response length (both filters decay well inside 0.4 s)
T0 = time.time()


def log(msg):
    print('[%6.1f s] %s' % (time.time() - T0, msg), flush=True)


def shaping_filters():
    imp = np.zeros(LG); imp[0] = 1.0
    b, a = butter(1, 10.0, fs=1.0 / TS)                  # HEURISTIC: velocity-type residual, move content < 10 Hz
    lp = lfilter(b, a, imp)
    b, a = butter(2, [106.0, 297.0], btype='bandpass', fs=1.0 / TS)  # HEURISTIC: B_tr (D-219), absorber residual
    bp = lfilter(b, a, imp)
    return {'white': None, 'lowpass10': lp, 'band106_297': bp}


def k1_ss():
    cfb, _ = C.build_cfb_at(0.0, TS)                     # K1: designed once at Y = 0 (DATA-DESIGN 5.11)
    ss = [tf2ss(b, a) for b, a in cfb]
    Ac = block_diag(*[s[0] for s in ss]); Bc = block_diag(*[s[1] for s in ss])
    Cc = block_diag(*[s[2] for s in ss]); Dc = np.diag([float(np.ravel(s[3])[0]) for s in ss])
    return Ac, Bc, Cc, Dc


def closed_loop(Y):
    A, B, Cy = C.ss_stage(*C.truth_mck(Y))
    Ad, Bd, Cd, _, _ = cont2discrete((A, B, Cy, np.zeros((3, 3))), TS, method='zoh')
    Ac, Bc, Cc, Dc = k1_ss()
    nc = Ac.shape[0]
    # THEORY: negative feedback with r = 0: e = -y, u = Cc xc + Dc e + PINV w, y = Cd x (ZOH plant, no feedthrough)
    Acl = np.block([[Ad - Bd @ Dc @ Cd, Bd @ Cc], [-Bc @ Cd, Ac]])
    Bcl = np.vstack([Bd @ C.PINV, np.zeros((nc, 3))])
    Ccl = np.hstack([Cd, np.zeros((3, nc))])
    return Acl, Bcl, Ccl


def gate_a(Acl, Bcl, Ccl, Y):
    f = np.logspace(0, np.log10(3000.0), 50)
    n = Acl.shape[0]
    H = np.array([Ccl @ np.linalg.solve(z * np.eye(n) - Acl, Bcl) for z in np.exp(1j * 2 * np.pi * f * TS)])
    _, PSt = C.loop(C.frf_zoh(*C.truth_mck(Y), f), C.ctrl_frf(0.0, f))
    ref = PSt @ C.PINV
    return float(np.max(np.linalg.norm(H - ref, axis=(1, 2)) / np.linalg.norm(ref, axis=(1, 2))))


def slow_poles(Acl, k=8):
    z = np.linalg.eigvals(Acl)
    lam, th = np.log(np.abs(z)), np.angle(z)
    keep = th >= 0                                       # one pole of each conjugate pair
    lam, th = lam[keep], th[keep]
    tau = -TS / lam                                      # THEORY: z = exp(s Ts), tau = -1 / Re(s)
    idx = np.argsort(-tau)[:k]
    return [dict(tau_s=float(tau[i]), f_hz=float(th[i] / (2 * np.pi * TS)),
                 zeta=float(-lam[i] / np.hypot(lam[i], th[i]))) for i in idx], float(np.abs(z).max())


def impulse_ss(Acl, Bcl, Ccl, n):
    h = np.zeros((n, 3, 3))
    X = Bcl.copy()
    for k in range(1, n):                                # h[0] = 0: no direct feedthrough
        h[k] = Ccl @ X
        X = Acl @ X
    return h


def taper(f):
    fhi = f.max(); f0 = 0.8 * fhi                        # HEURISTIC: raised cosine over the top 20 % against Gibbs ringing
    w = np.ones_like(f)
    m = f > f0
    w[m] = 0.5 * (1 + np.cos(np.pi * (f[m] - f0) / (fhi - f0)))
    return w


def invert_lines(PS, fl, vt=None):
    n_t = int(round(T_PER / TS))
    t = np.arange(n_t) * TS
    h = np.zeros((n_t, 3, 3)); s2 = np.zeros((3, 3))
    for c in range(3):
        f = fl[:, c]; w = taper(f)
        E = np.exp(1j * 2 * np.pi * np.outer(t, f))
        # THEORY: h[n] = Ts * integral over (-fs/2, fs/2) of H e^{j 2 pi f n Ts} df; real h -> 2 Re of the positive half
        h[:, :, c] = 2 * TS * DF * np.real(E @ (PS[:, :, c] * w[:, None]))
        if vt is not None:
            # THEORY: independent circular line errors, Var Re(dH e^{j phi}) = E|dH|^2 / 2
            s2[:, c] = (2 * TS * DF) ** 2 * np.sum(vt[:, :, c] * w[:, None] ** 2, axis=0) / 2
    return h, s2


def missed_curve(h, g, t_max):
    """missed[t] = energy of (h * 1[j > t]) conv g for t = 0..t_max, and the full energy (JH-013)."""
    nt = min(int(round(t_max / TS)), h.shape[0] - 1)
    if g is None:
        e2 = (h ** 2).sum(axis=(1, 2))
        tail = np.cumsum(e2[::-1])[::-1]
        return np.append(tail[1:nt + 2], 0.0)[:nt + 1], float(tail[0])
    r = np.empty((h.shape[0] + len(g) - 1, 3, 3))       # residual = full response, lags removed one by one
    for i in range(3):
        for j in range(3):
            r[:, i, j] = fftconvolve(h[:, i, j], g)
    full = float((r ** 2).sum())
    missed = np.empty(nt + 1); E = full
    L = len(g)
    for t in range(nt + 1):
        seg = r[t:t + L]
        d = h[t][None] * g[:len(seg), None, None]
        E += float(((seg - d) ** 2).sum() - (seg ** 2).sum())
        seg -= d
        missed[t] = E
    return missed, full


def readouts(missed, full, noise=None):
    B = np.cumsum(missed) / np.arange(1, len(missed) + 1) / full   # mean over t in [0, T) of missed / full
    out = {}
    for lvl, key in ((0.10, 'T10_s'), (0.01, 'T1_s')):
        k = np.argmax(B <= lvl)
        out[key] = float((k + 1) * TS) if B[k] <= lvl else None
    for T in T_READ:
        k = int(round(T / TS)) - 1
        if k < len(B):
            out['B(%.2f)' % T] = float(B[k])
            if noise is not None:
                out['res(%.2f)' % T] = bool(missed[k] > noise[k])
    return out


def fmt_t(x, cap):
    return ('>%.2f' % cap) if x is None else '%.3f' % x


def main():
    os.makedirs(OUT, exist_ok=True)
    j2 = np.load(os.path.join(ECL, 'outputs', 'j2_bla.npz'))
    filt = shaping_filters()
    res = {}
    for Y in YS:
        tag = 'K1_Y%+.2f' % Y
        Acl, Bcl, Ccl = closed_loop(Y)
        ga = gate_a(Acl, Bcl, Ccl, Y)
        poles, zmax = slow_poles(Acl)
        log('%s: G-A max rel err %.2e (%s), max |z| %.9f, slowest tau %.4f s at %.2f Hz (zeta %.3f)'
            % (tag, ga, 'PASS' if ga < 1e-6 else 'FAIL', zmax, poles[0]['tau_s'], poles[0]['f_hz'], poles[0]['zeta']))
        if ga >= 1e-6 or zmax >= 1.0:
            raise SystemExit('G-A failed or unstable loop at %s: stop (JH-013)' % tag)
        n_a = int(round(min(30.0, max(2.0, 15 * poles[0]['tau_s'])) / TS))
        hA = impulse_ss(Acl, Bcl, Ccl, n_a)
        log('%s: A-full impulse response, %d samples (%.1f s)' % (tag, n_a, n_a * TS))
        fl, PSm, vt = j2['fl_' + tag], j2['PS_' + tag], j2['vt_' + tag]
        PSa = np.empty_like(PSm)
        for c in range(3):
            _, PSt = C.loop(C.frf_zoh(*C.truth_mck(Y), fl[:, c]), C.ctrl_frf(0.0, fl[:, c]))
            PSa[:, :, c] = (PSt @ C.PINV)[:, :, c]
        hAb, _ = invert_lines(PSa, fl)
        hB, s2 = invert_lines(PSm, fl, vt)
        log('%s: line inversions done (A-band, B), period %.3f s' % (tag, T_PER))
        entry = dict(gate_A=ga, max_abs_z=zmax, slow_poles=poles,
                     lit_multiples_s=dict(x1_5=1.5 * poles[0]['tau_s'], x4=4 * poles[0]['tau_s']))
        for name, g in filt.items():
            gE = 1.0 if g is None else float((g ** 2).sum())
            mA, fA = missed_curve(hA, g, T_MAX_FULL)
            mAb, fAb = missed_curve(hAb, g, T_MAX_BAND)
            mB, fB = missed_curve(hB, g, T_MAX_BAND)
            # noise energy of hB over lags (t, n_t), passed through g (white noise per element)
            n_t = hB.shape[0]
            noise = (n_t - 1 - np.arange(len(mB))) * float(s2.sum()) * gE
            entry[name] = dict(A_full=readouts(mA, fA), A_band=readouts(mAb, fAb), B=readouts(mB, fB, noise),
                               B_noise_share_of_full=float(n_t * s2.sum() * gE / fB))
            e = entry[name]
            log('%s %-11s | A-full T10 %s T1 %s B(0.1) %.3f | A-band T10 %s T1 %s | B T10 %s T1 %s B(0.1) %.3f%s'
                % (tag, name, fmt_t(e['A_full']['T10_s'], T_MAX_FULL), fmt_t(e['A_full']['T1_s'], T_MAX_FULL),
                   e['A_full']['B(0.10)'], fmt_t(e['A_band']['T10_s'], T_MAX_BAND),
                   fmt_t(e['A_band']['T1_s'], T_MAX_BAND), fmt_t(e['B']['T10_s'], T_MAX_BAND),
                   fmt_t(e['B']['T1_s'], T_MAX_BAND), e['B']['B(0.10)'],
                   '' if e['B']['res(0.10)'] else ' (B(0.1) below noise)'))
        res[tag] = entry
        print('  slow poles %s: ' % tag + '; '.join('tau %.4f s, %.2f Hz, zeta %.3f' % (p['tau_s'], p['f_hz'], p['zeta'])
                                                  for p in poles), flush=True)
    with open(os.path.join(OUT, 'window_memory.json'), 'w') as fh:
        json.dump(res, fh, indent=1)
    log('wrote %s' % os.path.join(OUT, 'window_memory.json'))


if __name__ == '__main__':
    main()
