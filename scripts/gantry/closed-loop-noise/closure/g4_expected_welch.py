"""G4 (c) re-score (CN-008a): analytic S Phi_v S^H passed through the expectation of the Welch
estimator (Hann 2048, 50 %), compared with the saved simulation of run g4_closure_<reading>.

    python closure/g4_expected_welch.py [reading=tanh]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cn_env                                                           # noqa: E402

import numpy as np                                                      # noqa: E402
from scipy.signal import get_window                                     # noqa: E402

from measure import spec                                                # noqa: E402
from model import noise_model as nm                                     # noqa: E402
from sensitivity import loop                                            # noqa: E402
from closure.g4_closure import model_psd_fft, NFINE, op_of               # noqa: E402
cn_env.check_no_leak()

READING = sys.argv[1] if len(sys.argv) > 1 else 'tanh'
RUN = os.path.join(cn_env.OUTPUTS, f'g4_closure_{READING}')
FS = spec.FS


def main():
    g1 = np.load(os.path.join(cn_env.OUTPUTS, 'g1_spectra', 'g1.npz'), allow_pickle=True)
    z = np.load(os.path.join(RUN, 'g4.npz'))
    hv = np.load(os.path.join(cn_env.OUTPUTS, 'g3_source', f'hv_{READING}.npz'))
    names = list(g1['names']); keep = g1['keep']; Kr = g1['Kr']; xy = g1['xy']
    band_of = g1['band_of']; nb = len(g1['fc']); f = g1['f']; df = f[1] - f[0]
    ops = sorted(set(op_of(n) for n in names))
    pos = np.stack([xy[[i for i, n in enumerate(names) if op_of(n) == op][0]] for op in ops])
    w = np.array([Kr[[i for i, n in enumerate(names) if op_of(n) == op and keep[i]]].sum() for op in ops], float)
    w /= w.sum()
    ff, PHv = model_psd_fft(hv['A'], hv['Sig'], NFINE)
    K = loop.frf(*loop.ctrl_ss(), ff)
    I = np.eye(3)[None]
    Pe = np.zeros_like(PHv)
    for p_, y in enumerate(pos):
        Ap, Bp, Cp = loop.plant_ss(y, READING)
        S = np.linalg.inv(I + loop.frf(Ap, Bp, Cp, np.zeros((3, 3)), ff) @ K)
        Pe += w[p_] * spec.shape(S, PHv)
    A = spec.autos(Pe)                                       # (NFINE/2+1, 3)
    R = nm.autocov(A[:, :, None] * np.eye(3)[None][:, :, :1].reshape(1, 3, 1), FS)[:, :, 0] \
        if False else np.stack([nm.autocov(A[:, i:i + 1, None], FS)[:, 0, 0] for i in range(3)], 1)
    N = spec.NPER
    win = get_window('hann', N)
    rho = np.correlate(win, win, 'full')[N - 1:] / np.sum(win ** 2)      # lags 0..N-1
    Rt = np.zeros((N, 3))
    Rt[:N // 2] = R[:N // 2] * rho[:N // 2, None]
    Rt[N // 2 + 1:] = (R[1:N // 2][::-1] * rho[1:N // 2][::-1, None])  # negative lags
    Ew = np.real(np.fft.fft(Rt, axis=0))[:N // 2 + 1] / FS
    Ew[1:] *= 2
    Ew[-1] /= 2
    band_ew = spec.band_mean(Ew, band_of, nb)
    bs, lo, hi = z["band_sim"], z["s_lo"], z["s_hi"]
    if "bb_lo" in z.files:                                   # CN-008b: block-bootstrap CI
        lo, hi = z["bb_lo"], z["bb_hi"]
        print("[g4c] using the block-bootstrap CI of the simulation (CN-008b)")
    inside = ((band_ew >= lo) & (band_ew <= hi)).mean(0)
    nbins = np.array([(band_of == b).sum() for b in range(nb)])
    rms_ew = np.sqrt((band_ew * nbins[:, None]).sum(0) * df)
    rms_sim = np.sqrt((bs * nbins[:, None]).sum(0) * df)
    rel = rms_sim / rms_ew - 1
    old = ((z['band_an'] >= lo) & (z['band_an'] <= hi)).mean(0)
    ok = bool(np.all(inside >= 0.9) and np.all(np.abs(rel) <= 0.01))
    print(f'[g4c] expected-Welch analytic inside simulated CI: {np.round(inside, 3)} (was {np.round(old, 3)} '
          f'with the raw analytic); rms sim vs expected-Welch {np.round(rel * 100, 3)} %; PASS (c) {ok}')
    print(f'[g4c] largest band deviations sim/expected-1 [%]: '
          + '; '.join(f'{g1["fc"][b]:.0f} Hz {np.round((bs[b] / band_ew[b] - 1) * 100, 1)}'
                      for b in np.argsort(-np.abs(bs / band_ew - 1).max(1))[:5]))
    J = json.load(open(os.path.join(RUN, 'g4.json')))
    J.update(bands_expwelch_in_sim_ci=inside.tolist(), rms_sim_vs_expwelch=rel.tolist(), PASS_c_rescored=ok,
             PASS_rescored=bool(J['PASS_a'] and J['PASS_b'] and ok))
    json.dump(J, open(os.path.join(RUN, 'g4.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
