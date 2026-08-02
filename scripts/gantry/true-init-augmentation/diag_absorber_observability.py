"""Where IS the absorber state observable from, and where is it not?

This is the sharpest form of the whole thread, and it is two least-squares fits.

The coulomb-offset thread's F6 measured that `vdelta_a` is recoverable from the
ENCODER's inputs, a window of `na+1 = 18` past outputs and `nb+1 = 18` past
inputs, at `R^2 = 1.0000`, held out across record classes. That is what makes the
encoder-initialisation thread look promising.

The ANN sees something else entirely: the CURRENT state and the CURRENT input,
one sample, with rows 6-7 identically zero (gate G6). So the question that
decides whether a static augmentation can ever absorb the absorber is

    how much of vdelta_a(k) is a function of [x_phys(k), u(k)] alone?

Both fits are run here on identical data with identical machinery, plus a
held-out split, so the contrast is a measurement rather than a juxtaposition of
two numbers from two sources.

Run:
  PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output \\
      -n GraduationProject python -u \\
      scripts/gantry/true-init-augmentation/diag_absorber_observability.py
"""
__project_origin__ = "added"

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry'))

from data_exact import exact_truth                                   # noqa: E402

OUT = os.path.join(REPO, 'simulations', 'gantry_subnet', 'diagnostics')
RECORDS = ('V1_standstill_Yp10', 'V2_aprbs_Ylow',
           'V3_ysweep_Yp10', 'V4_lissajous_Ym10')
NA = 17          # = 2*(nx_phys + nx_ann) + 1, Jan's rule; the encoder window is na+1
LAB = ['delta_a', 'vdelta_a']


def fit_r2(F, T, split=0.7):
    """Least squares with a held-out tail. Returns (R2_train, R2_test) per column."""
    n = len(F)
    k = int(split * n)
    A = np.hstack([F, np.ones((n, 1))])
    c, *_ = np.linalg.lstsq(A[:k], T[:k], rcond=None)
    out = []
    for sl in (slice(0, k), slice(k, n)):
        res = T[sl] - A[sl] @ c
        ss_tot = ((T[sl] - T[sl].mean(axis=0)) ** 2).sum(axis=0)
        out.append(1.0 - (res ** 2).sum(axis=0) / np.maximum(ss_tot, 1e-300))
    return out[0], out[1]


def build(name):
    tr = exact_truth(name)
    rec, x6, x8 = tr['rec'], tr['x6'], tr['x8']
    N = len(x6)
    ix = np.arange(NA, N)
    # (a) what the ANN sees: the current physical state and the current input.
    #     Rows 6-7 of the model state are identically zero (G6) and carry nothing,
    #     so they are omitted rather than padded with zero columns.
    inst = np.hstack([x6[ix], rec['u'][ix]])
    # (b) what the ENCODER sees: y[k-na .. k] and u[k-na .. k]  (na_right = nb_right = 1)
    win = np.hstack([np.hstack([rec['y'][ix - j] for j in range(NA, -1, -1)]),
                     np.hstack([rec['u'][ix - j] for j in range(NA, -1, -1)])])
    return inst, win, x8[ix][:, [3, 7]]


def main():
    print('Where is the absorber state observable from?\n')
    print(f'  instantaneous features: x_phys(k) (6) + u(k) (3) = 9')
    print(f'  encoder-window features: y[k-{NA}..k] (3x{NA+1}) + u[k-{NA}..k] (3x{NA+1}) '
          f'= {6*(NA+1)}')
    print(f'  70/30 split, the 30 % is the TAIL of the record (no interleaving)\n')
    print(f'  {"record":<22}{"target":<10}{"inst train":>12}{"inst test":>12}'
          f'{"window train":>15}{"window test":>13}')
    res = {}
    for name in RECORDS:
        inst, win, T = build(name)
        ai, bi = fit_r2(inst, T)
        aw, bw = fit_r2(win, T)
        res[name] = {LAB[c]: dict(inst_train=float(ai[c]), inst_test=float(bi[c]),
                                  win_train=float(aw[c]), win_test=float(bw[c]))
                     for c in range(2)}
        for c in range(2):
            print(f'  {name if c == 0 else "":<22}{LAB[c]:<10}{ai[c]:>12.4f}{bi[c]:>12.4f}'
                  f'{aw[c]:>15.4f}{bw[c]:>13.4f}')

    print('\n  READING')
    print('   window >> inst  ->  the absorber state IS in the data the encoder reads and')
    print('                       is NOT in the instantaneous state the ANN reads. The')
    print('                       information exists; the architecture cannot route it,')
    print('                       because the only rows that could carry it from one to')
    print('                       the other are overwritten every step (gate G6).')
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, 'true_init_absorber_observability.json')
    with open(p, 'w') as f:
        json.dump(dict(na=NA, records=res), f, indent=2)
    print(f'\n  wrote {p}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
