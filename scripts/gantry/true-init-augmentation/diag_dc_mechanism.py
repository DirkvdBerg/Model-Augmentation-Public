"""Why is the per-window target STILL dirty when the six physical states are exact?

`diag_window_target.py` measured that per-window re-seeding from the exact 6-state
truth IC leaves the Y per-window DC scatter at `1.028e-04 m` against a
`3.147e-08 m` free-run floor, and that this is 1.0x the scatter you get from the
record's finite-difference rows. The exact velocities bought nothing on Y. The
handoff's section 5 assumption ("the clean target survives per-window
re-seeding") is therefore false, and section 10 says to establish whether the
cause is a defect in this code or a genuine property of re-seeding.

THE HYPOTHESIS. The truth has EIGHT states. Re-seeding the six physical ones
exactly still starts the window with the absorber at rest, while the truth's
absorber is mid-oscillation and carries momentum `vdelta_a(s)`. On a `K = 0` axis
that momentum error integrates into a per-window ramp instead of decaying, which
is the F3 mechanism from the coulomb-offset thread (`corr = -1.000` with
`vDelta_a(0)`, slope `-(ma/mh) * nf/2`) reappearing per window rather than once.

THREE ARMS, and the third is the decisive one:

  R  regression of the measured per-window mean against `[delta_a(s), vdelta_a(s)]`.
     If the DC is the absorber IC, this is R^2 ~ 1 with the predicted slope.
  T8 the TRUTH model re-seeded per window from the exact EIGHT-state IC.
     Scatter must collapse to the integrator floor: this is the harness gate,
     and it also proves the window grid and the statistics are not the problem.
  T6 the TRUTH model re-seeded per window from the exact SIX-state IC with the
     absorber zeroed. Same model, same window grid, same integrator, and the ONLY
     difference from T8 is the absorber initial condition. If T6 reproduces the
     baseline's scatter, then the entire per-window DC is the absorber IC and
     nothing about the baseline, the CoG term or the LFR realization is involved.

T6 is a MEASUREMENT of the mechanism, not an implementation of the rejected
"seed rows 6-7" fix (handoff section 8). It is run on the TRUTH's own absorber
coordinates, where they mean what they say; the model's rows 6-7 are latent and
are not touched anywhere in this file.

Run:
  PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output \\
      -n GraduationProject python -u \\
      scripts/gantry/true-init-augmentation/diag_dc_mechanism.py
"""
__project_origin__ = "added"

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry'))

from data_exact import exact_truth, rollout8, x6_from_x8            # noqa: E402
from gantry_dynamic.oracle import MA, MH_RIGID, L0                  # noqa: E402
from model_augmentation.systems.gantry_ss import mh as _mh          # noqa: E402

OUT = os.path.join(REPO, 'simulations', 'gantry_subnet', 'diagnostics')
RECORDS = ('V1_standstill_Yp10',)
NF, STRIDE, K0 = 400, 100, 17
STATES = ['X', 'Theta', 'Y', 'dX', 'dTheta', 'dY']
MH = float(_mh)


def r2_and_slope(y, X):
    A = np.hstack([X, np.ones((len(X), 1))])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ c
    ss = 1.0 - (res ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-300)
    return float(ss), c


def main():
    res = {}
    for name in RECORDS:
        print(f'=== {name} ===')
        tr = exact_truth(name)
        rec, x6, x8 = tr['rec'], tr['x6'], tr['x8']
        ts, N = rec['ts'], len(x6)
        starts = np.arange(K0, N - NF + 1, STRIDE)
        da, vda = x8[:, 3], x8[:, 7]
        print(f'  {len(starts)} windows of {NF} ({NF*ts*1e3:.0f} ms), stride {STRIDE}')
        print(f'  truth absorber: delta_a std {da.std():.4e} m, '
              f'vdelta_a std {vda.std():.4e} m/s')

        # ---- R: is the measured per-window DC the absorber IC? --------------
        wm_path = os.path.join(HERE, 'figures', f'_winmeans_{name}.npz')
        if not os.path.exists(wm_path):
            print(f'  {wm_path} missing -- run diag_window_target.py first')
            return 1
        z = np.load(wm_path)
        wm, st = z['exact'], z['starts']
        assert np.array_equal(st, starts), 'window grids differ'
        feats = np.stack([da[starts], vda[starts]], axis=1)
        print(f'\n  R  regression of the per-window mean on [delta_a(s), vdelta_a(s)]')
        print(f'  {"state":<8}{"R^2":>10}{"corr vda":>12}{"slope vda":>14}'
              f'{"predicted":>14}{"ratio":>9}')
        rr = {}
        for c in range(6):
            s2, coef = r2_and_slope(wm[:, c], feats)
            corr = float(np.corrcoef(wm[:, c], vda[starts])[0, 1])
            # THEORY (coulomb-offset F3 / msd-offset mechanism): a missing absorber
            # momentum vdelta_a(0) acts on the payload row as a velocity deficit
            # (ma/mh)*vdelta_a(0); on a K = 0 axis that integrates, so the mean over
            # an nf-window is -(ma/mh)*vdelta_a(0)*nf*ts/2. Position rows only.
            pred = (-(float(MA) / MH) * NF * ts / 2 if c == 2 else
                    (-(float(MA) / MH) if c == 5 else np.nan))
            rr[STATES[c]] = dict(r2=s2, corr_vda=corr, slope_vda=float(coef[0 + 1]),
                                 slope_da=float(coef[0]), predicted=float(pred))
            ratio = coef[1] / pred if np.isfinite(pred) and pred != 0 else np.nan
            print(f'  {STATES[c]:<8}{s2:>10.4f}{corr:>12.4f}{coef[1]:>14.4e}'
                  f'{pred:>14.4e}{ratio:>9.3f}')
        print('  (predicted is written only for the two rows the mechanism predicts:')
        print('   Y position -(ma/mh)*nf*ts/2 and dY -(ma/mh); the others have no')
        print('   closed form here and are left blank)')

        # ---- T8 / T6: the truth model against itself ------------------------
        print(f'\n  T8/T6  the TRUTH model re-seeded per window, 8-state vs 6-state IC')
        t0 = time.time()
        out = {}
        for tag, zero_abs in (('T8_exact8', False), ('T6_abs_zeroed', True)):
            wmeans = np.empty((len(starts), 6))
            for i, s in enumerate(starts):
                x0 = x8[s].copy()
                if zero_abs:
                    x0[3] = 0.0        # delta_a
                    x0[7] = 0.0        # vdelta_a
                sim = rollout8(x0, rec['u_log'][s:s + NF], ts, up_sample=1)
                wmeans[i] = (x6_from_x8(sim) - x6[s:s + NF]).mean(axis=0)
            out[tag] = wmeans
            print(f'    {tag:<14} {time.time()-t0:6.1f} s')

        print(f'\n  {"state":<8}{"T8 exact-8":>15}{"T6 abs zeroed":>16}'
              f'{"baseline exact-6":>18}{"T6 / baseline":>15}')
        cmp_ = {}
        for c in range(6):
            a = float(out['T8_exact8'][:, c].std())
            b = float(out['T6_abs_zeroed'][:, c].std())
            d = float(wm[:, c].std())
            cmp_[STATES[c]] = dict(t8=a, t6=b, baseline=d, t6_over_base=b / max(d, 1e-300))
            print(f'  {STATES[c]:<8}{a:>15.4e}{b:>16.4e}{d:>18.4e}{b/max(d,1e-300):>15.3f}')

        print('\n  READING')
        print('   T8 near zero        -> the harness, the grid and the statistics are sound;')
        print('                          a per-window re-seed with the COMPLETE state is clean.')
        print('   T6 ~ baseline       -> the whole per-window DC is the absorber initial')
        print('                          condition. The baseline model, the CoG term and the')
        print('                          LFR realization contribute nothing to it.')
        res[name] = dict(n_win=int(len(starts)), nf=NF, stride=STRIDE, ts=float(ts),
                         da_std=float(da.std()), vda_std=float(vda.std()),
                         regression=rr, truth_arms=cmp_)

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, 'true_init_dc_mechanism.json')
    with open(p, 'w') as f:
        json.dump(res, f, indent=2, default=float)
    print(f'\n  wrote {p}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
