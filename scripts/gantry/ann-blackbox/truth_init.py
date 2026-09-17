"""Write the ORACLE linear initialisation: the exact frozen truth, discretised.

This is the phase-1 diagnostic arm of `tasks/handoffs/2026-09-17-full-bla-blackbox.md`. It is
deliberately NOT a defensible identification result: it reads the truth model directly, so no
thesis claim may rest on it. It exists to answer one upstream question, namely whether the
black box can learn this plant at all when the linear start is correct by construction.

WHAT IT PRODUCES. `results/truth_init/truth_ss_fs<FS>.npz`, in the same key format that
`results/frf_init/frf_init_ss.npz` uses, so `bla_init.apply_frf_bla_init(..., ss_path=...)`
consumes it unchanged and the oracle arm differs from the `--bla frf` arm in exactly one file.

OPERATING POINT. Y = 0, matching the standstill record `T3_standstill_Y000` that the frequency
domain chain in `frf_init.py` is estimated from, so oracle and estimated init describe the same
frozen plant and the only difference between them is estimation error.

PLANT PARAMETERISATION. `truthmodel.truth_ct` reads `plant.py`'s module constants, which are
`MA_FRAC = 0.10`, `ZETA_A = 0.05`. Those describe the `augmentation` dataset and nothing else
(handoff fact 2). `ann_blackbox.py`, `data.py` and `bars.py` all read that same folder, so on
this dataset the truth, the bars and every prior run agree. Running this script against any
other trajectory folder would silently build the truth of a different plant.
"""
__project_origin__ = "added"

import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, HERE)
from truthmodel import truth_ct, discretise
import plant                                    # imported by truthmodel; re-used here for the report

FS = 800.0                                      # matches every prior black-box run (handoff fact 11)
Y_OP = 0.0                                      # matches frf_init.py's T3_standstill_Y000
OUT = os.path.join(HERE, 'results', 'truth_init')


def build(fs=FS, y_op=Y_OP):
    A, B, C = truth_ct(y_op)
    Ad, Bd = discretise(A, B, 1.0 / fs)
    return A, B, C, Ad, Bd


def report(A, Ad, fs):
    """Continuous poles as (f_n, zeta) pairs, plus the discrete spectral radius."""
    pc = np.linalg.eigvals(A)
    print(f'continuous-time poles at Y = {Y_OP:+.3f} m (plant.py MA_FRAC={plant.MA_FRAC}, '
          f'ZETA_A={plant.ZETA_A}):')
    for p in sorted(pc, key=lambda z: abs(z)):
        wn, z = abs(p), (-p.real / abs(p) if abs(p) > 0 else np.nan)
        fd = abs(p.imag) / (2 * np.pi)
        print(f'  {p.real:+.6e} {p.imag:+.6e}j   f_n {wn/(2*np.pi):9.4f} Hz   '
              f'zeta {z:8.5f}   f_d {fd:9.4f} Hz')
    ev = np.linalg.eigvals(Ad)
    print(f'\ndiscretised at {fs:.0f} Hz: max|eig| = {np.max(np.abs(ev)):.6f}, '
          f'n outside unit circle = {int(np.sum(np.abs(ev) > 1 + 1e-12))}')
    print(f'|eig(A_d)| = {np.array2string(np.sort(np.abs(ev))[::-1], precision=6)}')
    return pc, ev


if __name__ == '__main__':
    A, B, C, Ad, Bd = build()
    pc, ev = report(A, Ad, FS)

    # Sanity, all three from the handoff section 9 step 1. Each catches a real error: a wrong
    # ma_frac moves the absorber, a wrong Y moves the rigid-body block, a wrong ts breaks the ZOH.
    fa_und = sorted(abs(p) / (2 * np.pi) for p in pc if p.imag > 0)
    fa_dmp = sorted(abs(p.imag) / (2 * np.pi) for p in pc if p.imag > 0)
    print('\nchecks:')
    print(f'  absorber undamped  {fa_und[-1]:8.3f} Hz   expect 158.11')
    print(f'  absorber damped    {fa_dmp[-1]:8.3f} Hz   expect 157.89')
    print(f'  slow pair undamped {fa_und[0]:8.3f} Hz   expect ~5.12')
    print(f'  max|eig(A_d)|      {np.max(np.abs(ev)):.6f}   expect 1.000000')

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f'truth_ss_fs{int(FS)}.npz')
    np.savez(path, A=Ad, B=Bd, C=C, Ac=A, Bc=B, fs=float(FS), poles_c=pc,
             Y_op=float(Y_OP), ma_frac=float(plant.MA_FRAC), zeta_a=float(plant.ZETA_A))
    print(f'\nwrote {path}')
