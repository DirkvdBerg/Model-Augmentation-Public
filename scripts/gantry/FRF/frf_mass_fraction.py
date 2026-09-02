"""3x3 MIMO FRF of the 8-state augmented gantry at two hidden-MSD mass fractions.

WHAT IT SHOWS, and why it is worth a figure.
`ma_frac` splits a FIXED payload mass `mh_total = 10.1 kg` between the rigid head and the hidden
absorber, and `ka` is recomputed as `ma*(2*pi*fa)^2` so the absorber's STANDALONE frequency stays
at `fa = 150 Hz`. Two consequences that pull in opposite directions:

  ANTI-RESONANCE of `Y <- F_Y` stays at `sqrt(ka/ma) = 150 Hz` for every `ma_frac`, because that
  is the frequency at which the absorber's inertia force cancels the applied force.

  RESONANCE moves, because the head is a FREE mass that recoils, so the coupled mode is the
  free-free two-mass root `f = fa*sqrt(1 + ma/mhr)`:  158.11 Hz at 0.10, 212.13 Hz at 0.50.

So the two features SEPARATE as `ma_frac` grows. `gtd_config.m` picked the multisine band
`[130, 180]` as "targets fa=150 +/- margin", i.e. around the STANDALONE number, which happens to
be the anti-resonance. At `ma_frac = 0.10` the real mode at 158 Hz still fell inside that window;
at 0.50 it walks out to 212 Hz and the band is left centred on the anti-resonance. Measured on
the records, input at the absorber is 45 to 57 dB down on the `ma50` datasets.

The other eight panels carry the second point for free: the absorber appears ONLY in `Y <- F_Y`
(participation 1e-4 elsewhere, per multisine_frequency_range_MSD.m), which is why an
energy-ranked estimator such as N4SID discards it (D-148).

MODEL-BASED, deliberately. The `ma50` records carry almost no input at 212 Hz, so a measured
overlay would show noise exactly where the interesting feature is. Validation against the MATLAB
diagnostic (`Matlab-scripts/Augmentation/diagnostics/multisine_frequency_range_MSD.m`) is printed
at the end and must pass before any number here is trusted.

Run:
  PYTHONIOENCODING=utf-8 conda run --no-capture-output -n GraduationProject python -u \
      scripts/gantry/FRF/frf_mass_fraction.py
"""
__project_origin__ = "added"

import os
import sys

import numpy as np
import scipy.linalg as sla
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry', 'msd-offset'))
import plant                                                            # noqa: E402

# ── configuration ───────────────────────────────────────────────────────────
MA_FRACS = (0.10, 0.50)          # 0.10 = augmentation/joint datasets, 0.50 = *_ma50*
Y_OP = 0.0                       # frozen Y [m]. T3_standstill_Y000, the identification record.
FA = 150.0                       # absorber standalone frequency [Hz], held fixed by construction
ZETA_A = 0.05
F_LO, F_HI, N_F = 1.0, 300.0, 3000
BANDS = {                        # multisine bands actually used, for the shaded overlay
    'augmentation [130,180]': (130.0, 180.0),
    'joint(_lowf) [~0,200]': (1.0, 200.0),
}
CH_IN = ('F_X1', 'F_X2', 'F_Y')
CH_OUT = ('x1', 'x2', 'Y')
COL = {0.10: '#1f77b4', 0.50: '#d62728'}
OUT = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'figures')


def build(ma_frac, Y=Y_OP):
    """Frozen 8-state augmented model at (Y, da=0), in STAGE coordinates.

    Replicates plant.M8 / _K4 / _C4 with the absorber parameterised, because plant.py hard-codes
    MA_FRAC = 0.10 at module level and every truth quantity downstream inherits it.
    """
    m1, m2, mb, Lb, d = plant.m1, plant.m2, plant.mb, plant.Lb, plant.d
    Jb, Jh, l0 = plant.Jb, plant.Jh, plant.L0
    mh_total = plant.mh
    ma = ma_frac * mh_total
    mhr = mh_total - ma
    ka = ma * (2 * np.pi * FA) ** 2          # THEORY: k = m*(2*pi*f)^2, as gtd_config.m
    ca = 2 * ZETA_A * np.sqrt(ka * ma)       # THEORY: c = 2*zeta*sqrt(k*m)

    B12 = (m1 - m2) * Lb / 2
    off = B12 - (mhr + ma) * Y - ma * l0
    M = np.array([
        [m1 + m2 + mb + mhr + ma, off, 0., 0.],
        [off, Jb + Jh + (m1 + m2) * Lb ** 2 / 4 + (mhr + ma) * d ** 2
              + mhr * Y ** 2 + ma * (Y + l0) ** 2, -(mhr + ma) * d, -ma * d],
        [0., -(mhr + ma) * d, mhr + ma, ma],
        [0., -ma * d, ma, ma]])
    C4 = plant._C4.copy(); C4[3, 3] = ca
    K4 = plant._K4.copy(); K4[3, 3] = ka

    Minv = np.linalg.inv(M)
    A = np.zeros((8, 8))
    A[:4, 4:] = np.eye(4)
    A[4:, :4] = -Minv @ K4
    A[4:, 4:] = -Minv @ C4
    B_log = np.zeros((8, 3))
    B_log[4:] = Minv @ plant._E43
    C_log = np.zeros((3, 8))
    C_log[:, :3] = np.eye(3)

    P = plant.P_np
    return A, B_log @ P, P.T @ C_log        # u_logical = P u_stage, y_stage = P^T y_logical


def frf(A, B, C, f):
    """G(j2*pi*f), shape (len(f), ny, nu). Solved per frequency, no explicit inverse."""
    n = A.shape[0]
    G = np.empty((len(f), C.shape[0], B.shape[1]), complex)
    for i, fi in enumerate(f):
        G[i] = C @ np.linalg.solve(1j * 2 * np.pi * fi * np.eye(n) - A, B)
    return G


def tzeros(A, b, c):
    """Transmission zeros of a SISO (A,b,c,0) by the Rosenbrock pencil generalized eigenproblem.

    THEORY: the invariant zeros are the finite generalized eigenvalues of
        [[A, b], [c, 0]]  -  lambda * [[I, 0], [0, 0]]
    (Rosenbrock system matrix). Infinite eigenvalues are dropped, which is what MATLAB's `tzero`
    reports as the finite zero set.
    """
    n = A.shape[0]
    Mp = np.block([[A, b.reshape(-1, 1)], [c.reshape(1, -1), np.zeros((1, 1))]])
    Np = np.zeros((n + 1, n + 1)); Np[:n, :n] = np.eye(n)
    z = sla.eig(Mp, Np, right=False)
    return z[np.isfinite(z)]


def modes(A):
    """(f_damped [Hz], zeta) for each oscillatory conjugate pair, sorted by frequency."""
    lam = np.linalg.eigvals(A)
    out = []
    for L in lam:
        if L.imag > 1e-6:
            out.append((abs(L.imag) / (2 * np.pi), -L.real / abs(L)))
    return sorted(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    f = np.logspace(np.log10(F_LO), np.log10(F_HI), N_F)

    data, report = {}, []
    for mf in MA_FRACS:
        A, B, C = build(mf)
        G = frf(A, B, C, f)
        ms = modes(A)
        z = tzeros(A, B[:, 2], C[2])                       # Y <- F_Y driving point
        zf = sorted({round(abs(zz.imag) / (2 * np.pi), 2) for zz in z if abs(zz.imag) > 1})
        data[mf] = dict(G=G, modes=ms, antires=zf)
        ma = mf * plant.mh
        pred = FA * np.sqrt(1 + ma / (plant.mh - ma))
        report.append((mf, ma, plant.mh - ma, pred, ms, zf))
        print(f'\nma_frac = {mf:.2f}   ma = {ma:.3f} kg   mhr = {plant.mh - ma:.3f} kg')
        print(f'  coupled absorber, predicted  f = fa*sqrt(1+ma/mhr) = {pred:.2f} Hz (undamped)')
        print(f'  eigenvalue modes [Hz, zeta]  : ' +
              ', '.join(f'{fn:.2f} ({zt:.4f})' for fn, zt in ms))
        print(f'  anti-resonances Y<-F_Y [Hz]  : ' + ', '.join(f'{v:.2f}' for v in zf))

    # ── figure ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True)
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            for lo, hi in BANDS.values():
                ax.axvspan(lo, hi, color='0.85', alpha=0.45, lw=0, zorder=0)
            for mf in MA_FRACS:
                ax.loglog(f, np.abs(data[mf]['G'][:, i, j]), color=COL[mf], lw=1.4,
                          label=f'ma_frac = {mf:.2f}', zorder=3)
            ax.grid(True, which='both', alpha=0.25, lw=0.4)
            ax.set_title(f'{CH_OUT[i]}  <-  {CH_IN[j]}', fontsize=10)
            if i == 2:
                ax.set_xlabel('frequency [Hz]')
            if j == 0:
                ax.set_ylabel('|G|  [m/N]')

    ax = axes[2, 2]                                         # Y <- F_Y, where the absorber lives
    for mf in MA_FRACS:
        g = np.abs(data[mf]['G'][:, 2, 2])
        for fn, _ in data[mf]['modes']:
            if fn > 50:                                     # the absorber pair, not the 5 Hz one
                k = int(np.argmin(np.abs(f - fn)))
                ax.plot(f[k], g[k], 'o', color=COL[mf], ms=7, mfc='none', mew=1.8, zorder=5)
                ax.annotate(f'{fn:.1f} Hz', (f[k], g[k]), textcoords='offset points',
                            xytext=(6, 8), color=COL[mf], fontsize=9, fontweight='bold')
        for za in data[mf]['antires']:
            if za > 50:
                k = int(np.argmin(np.abs(f - za)))
                ax.plot(f[k], g[k], 'v', color=COL[mf], ms=7, zorder=5)
                ax.annotate(f'{za:.1f} Hz', (f[k], g[k]), textcoords='offset points',
                            xytext=(-42, -14), color=COL[mf], fontsize=9)
    axes[0, 0].legend(loc='lower left', fontsize=9, framealpha=0.9)
    fig.suptitle(f'Augmented gantry FRF, stage frame, frozen at Y = {Y_OP:.2f} m.  '
                 f'Circles = absorber resonance, triangles = anti-resonance.  '
                 f'Shaded = multisine bands in use.', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    for ext in ('png', 'pdf'):
        p = os.path.join(FIG, f'frf-mass-fraction-Y{Y_OP:+.2f}.{ext}')
        fig.savefig(p, dpi=160)
    print(f'\nwrote {p}')

    # ── validation against the MATLAB diagnostic, which must agree ──────────
    print('\n=== cross-check vs multisine_frequency_range_MSD.m (Y = 0.30 m there) ===')
    for mf, expect in ((0.10, 157.89), (0.50, 211.60)):
        A, _, _ = build(mf, Y=0.30)
        hi = max(modes(A))[0]
        ok = 'OK' if abs(hi - expect) < 0.05 else 'MISMATCH'
        print(f'  ma_frac {mf:.2f}: python {hi:.2f} Hz vs MATLAB {expect:.2f} Hz   {ok}')


if __name__ == '__main__':
    main()
