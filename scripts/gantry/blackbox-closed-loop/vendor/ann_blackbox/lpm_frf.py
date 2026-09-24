"""Step 1 of PLAN-BLA.md: nonparametric MIMO FRF by the Local Polynomial Method.

QUESTION THIS ANSWERS. `bla_decimation_test.py` showed the time-domain BLA has no complex pole at
either physical resonance. Before replacing the estimator we must know whether the resonances are
recoverable from this data AT ALL. A nonparametric FRF answers that without fitting any model
order, so it cannot hide a mode by spending states elsewhere. If the 157.9 Hz peak is here, the
data is fine and the estimator was the problem; if it is not, no estimator can help and the plan
branches to new excitation (PLAN-BLA.md step 4).

METHOD, and why this one.
`THEORY` Pintelon, Vandersteen, Schoukens, Rolain, "Fast FRF Measurement of Multivariable Systems
Using Periodic Excitations", I2MTC 2011, DOI 10.1109/imtc.2011.5944005, held at
literature/experiment-design/Papers/. The DFT of a finite record contains the steady-state term,
a leakage/transient term, and noise. The transient term T(Omega) is a SMOOTH function of
frequency while the excitation is not, so both are separated by fitting a local polynomial in the
frequency index over a sliding window of bins:

    Y_i(k) = sum_j G_ij(k) U_j(k) + T_i(k),
    G_ij(k) ~ sum_{r=0..R} g_ijr (k-k0)^r,      T_i(k) ~ sum_{r=0..R} t_ir (k-k0)^r

and G_ij(k0) = g_ij0. Their validation case is an aluminium plate under FREE-FREE boundary
conditions, i.e. with rigid-body modes present, which is our structure. Their stated sizing rule
for the excited-bin fit is 2*n_R + 1 > (R+1)*n_u; with R=2 and n_u=3 that is a window of at least
13 bins. We use more for conditioning and report the condition number at every bin.

`THEORY` Pintelon and Schoukens, System Identification: A Frequency Domain Approach, 2nd ed.,
Sec. 13.11.2: in the frequency domain there is no problem modelling plants whose poles lie on the
unit circle, because the transfer function is evaluated only on a grid; lines coinciding with a
pole are dropped or regularized. That is why the band below F_LO is excluded rather than fitted:
our two rigid-body poles sit at s = 0.

`HEURISTIC` the band 0.5 to 250 Hz. Chosen from our own measured spectrum (`frf_diagnostic.py`):
above 250 Hz the output sits at solver-residual level (1e-21), so content there is numerical, not
physical. No literature source sets these numbers.

WHY 4 kHz AND NOT 800 Hz. The decimation to 800 Hz applies a block mean to u and a non-causal
zero-phase FIR to y (data.py:29-30). Neither is needed here, and the FIR has no causal pole-zero
realisation, which `bla_decimation_test.py` showed produces near-Nyquist artefacts. Working at the
native rate removes the whole question.

Run:
  PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output \
      -n GraduationProject python -u scripts/gantry/ann-blackbox/lpm_frf.py
"""
__project_origin__ = "added"

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# VENDOR-PATCH (BB-002): repo five levels up; plant.py from vendor/msd_offset.
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'msd_offset'))   # VENDOR-PATCH (BB-002)

import plant                                    # noqa: E402

# WHICH TRACK. `plant.TRAJ` defaults to the 'augmentation' folder, whose multisine band is
# [130, 180] Hz (gtd_config.m:107). The 'joint' folder was generated with the SAME record names
# but a broadband [1, 200] Hz multisine (gtd_config.m:105), which is strictly better here:
# one periodic, leakage-free record set covers BOTH the 5.12 Hz pair and the 157.9 Hz absorber,
# so the low band no longer has to come from non-periodic trajectory records with a window.
# Measured on joint/T3_standstill_Y000: excited lines run to a sharp cutoff at exactly 200.00 Hz,
# and the 5.08 Hz line is present at -49 dB of the peak (the controller's high loop gain
# suppresses it, not the design). In a noiseless simulation that is still ample.
TRACK = 'joint'                                 # 'joint' or 'augmentation'
plant.TRAJ = os.path.join(REPO, 'data', 'gantry', 'matlab', 'trajectory', TRACK)
from plant import load_record                   # noqa: E402  (must follow the TRAJ override)

FS = 4000.0
# MIMO IDENTIFIABILITY, and why this list. The gantry has 3 inputs driven SIMULTANEOUSLY, so one
# record gives 3 equations per frequency for 9 unknown FRM entries. The classical fix is n_u
# experiments with uncorrelated excitations (5SMB0 Lecture 9 pp. 46-52; Pintelon et al. 2011
# calls the same thing the "robust" method). Measured input cross-correlations: T10 has
# corr(x1,x2) = +0.895, i.e. the two gantry drives are nearly collinear in a single record, which
# is why a one-record solve was ill-conditioned (median cond 1.7e6, 67 % median error vs truth).
# Every record below sits at Y_op = 0.0000 m, so combining them does NOT mix operating points.
# E2_multisine_Yp22 is EXCLUDED: it is at Y = +0.22 m and would mix scheduling into a fit that is
# meant to be at one frozen Y. E1 has corr(x1,x2) = -1.000, a pure differential drive, so it adds
# the direction the APRBS records are weakest in.
#
# WHAT THE GENERATOR ACTUALLY DID, read from Matlab-scripts/Augmentation/data/ rather than
# assumed. This folder is TRACK='augmentation' (gtd_config.m:117 keys out_dir on TRACK), so
# gtd_config.m:107 gives a multisine band of [130, 180] Hz, deliberately aimed at the absorber.
# gtd_make_multisine.m:53-55 puts the lines on `cfg.f_low:df:cfg.f_high` with df = fs/N and
# `bins = round(freqs/df)+1`, i.e. EXACTLY on FFT bins with period = the full record, synthesised
# by IFFT of a unit-magnitude spectrum. The multisine component is therefore leakage-free.
# gtd_build_records.m:103-104 gives every aprbs record excitation='multisine', amp_frac=1.0;
# E4 is pnone(), excitation='none'; E1 is a 130-180 Hz sinesweep.
#
# CONSEQUENCE FOR THIS SCRIPT. Two disjoint problems needing two different record sets:
#   absorber band  the STANDSTILL records carry the multisine and NOTHING else, so they are
#                  exactly periodic and leakage-free. No window needed; the only non-periodic
#                  term is the startup transient, which is what the LPM models.
#   slow mode      only the trajectory excites 5 Hz, and the trajectory is non-periodic, so this
#                  band needs the APRBS records plus a window.
# Combining standstill records taken at different Y is defensible for the absorber band because
# the absorber pair is Y-invariant to six digits (frf_diagnostic.py, truth model); it would NOT
# be defensible for the 5 Hz pair, which moves 4.83 to 5.12 Hz over the Y range.
_STANDSTILL = ('T1_standstill_Ym30', 'T2_standstill_Ym15', 'T3_standstill_Y000',
               'T4_standstill_Yp15', 'T5_standstill_Yp30', 'V1_standstill_Yp10')
_APRBS = ('T9_aprbs_30', 'T10_aprbs_60', 'T11_aprbs_100', 'T12_aprbs_yaw',
          'T13_lissajous', 'T14_lissajous_yaw', 'E3_aprbs_above')

# THE LPM WINDOW MUST BE NARROWER THAN THE NARROWEST FEATURE IT HAS TO REPRESENT.
# The local polynomial models G as smooth ACROSS the window, so a resonance narrower than the
# window is smoothed away. Half-power bandwidths from the frozen truth, 2*zeta*f_n:
#     absorber  2*0.0528*157.894 = 16.7 Hz   -> a 25-bin (2.08 Hz) window is 8x smaller, fine
#     slow pair 2*0.0921*  5.122 =  0.94 Hz  -> a 25-bin window is 2.2x WIDER, so it smears it
# Measured: with n_half=12 the 5 Hz estimate is ~99 % in error in every channel.
# Shrinking the window is affordable because the sizing rule counts EQUATIONS, not bins:
# M records x W bins > (n_u+1)*(R+1) = 12, so at M=6 even W=5 gives 30 equations.
# SINGLE OPERATING POINT, and this REVERSES the multi-record argument above.
# Combining the six standstill records mixes SIX DIFFERENT Y values (-0.30 to +0.30 m). The
# absorber is Y-invariant so that band survives, but the low-frequency rigid-body response goes
# as 1/(M(Y) s^2) and M(Y) carries the mh*Y^2 term, so mixing Y biases the low band badly.
# Measured against the frozen truth at Y = 0:
#     T3 alone (Y=0), n_half=6 :  low band 0.0149,  absorber band 0.122
#     all six Y,      n_half=2 :  low band 0.2814,  absorber band 0.156
# Single-Y is better in BOTH bands. The MIMO identifiability worry that motivated stacking
# records does not bite here, because the sizing rule counts M*W and the LPM's smoothness
# constraint already supplies enough equations from one record: 1 x 13 = 13 > 12.
# For a Y-dependent FRF, estimate one FRF PER standstill record; that is PLAN-BLA.md step 5.
_ONE_Y = ('T3_standstill_Y000',)

if TRACK == 'joint':
    # ONE broadband periodic record at ONE operating point; two passes differing only in the
    # LPM window, because the two resonances have very different bandwidths.
    PASSES = (
        # Band starts at 1.0 Hz, not 3.0: the frozen truth has two REAL poles at 0.103 and
        # 0.158 Hz, and a fit that starts at 3 Hz has no data constraining them, which
        # `frf_to_ss.py` showed spends two model poles on spurious 14.6 Hz content instead.
        dict(name='absorber', band=(1.0, 200.0), window=False, n_half=12, records=_ONE_Y),
        dict(name='slow', band=(3.0, 30.0), window=False, n_half=6, records=_ONE_Y),
    )
else:
    PASSES = (
        dict(name='absorber', band=(130.0, 180.0), window=False, n_half=12,
             records=_STANDSTILL),
        dict(name='slow', band=(1.0, 50.0), window=True, n_half=2, records=_APRBS),
    )
RECORDS = PASSES[0]['records']        # kept for backwards compatibility of the loader below
F_LO, F_HI = 0.5, 250.0              # HEURISTIC, see docstring
R_ORDER = 2                          # local polynomial degree
N_HALF = 12                          # half window in bins -> 25 bins, well over the 13 required
CHAN = ('x1', 'x2', 'Y')
OUT = os.path.join(HERE, 'results', 'lpm_frf')
FIG = os.path.join(HERE, 'figures')


def truth_frf(f_hz, Y=0.0):
    """Stage-force to stage-position FRF of the frozen 8-state truth, as a DIAGNOSTIC overlay.

    Coordinates, verified against plant.py rather than assumed:
      state  x = [X, Th, Y, da, dX, dTh, dY, vda]  (logical)
      input  deriv8 uses _E43 @ u_logical, and load_record sets u_log = P @ u_stage
      output y_stage = P^T @ q_logical[0:3]        (plant.to_stage)
    """
    Minv = np.linalg.inv(plant.M8(Y, 0.0, freeze=True))
    A = np.block([[np.zeros((4, 4)), np.eye(4)],
                  [-Minv @ plant._K4, -Minv @ plant._C4]])
    B = np.vstack([np.zeros((4, 3)), Minv @ plant._E43])      # logical force in
    C = np.hstack([np.eye(3, 4), np.zeros((3, 4))])           # q_logical[0:3] out
    P = plant.P_np
    G = np.empty((len(f_hz), 3, 3), complex)
    I8 = np.eye(8)
    for i, f in enumerate(f_hz):
        s = 2j * np.pi * f
        G[i] = P.T @ C @ np.linalg.solve(s * I8 - A, B) @ P   # stage -> stage
    return G


def baseline_frf(f_hz, Y=0.0):
    """Same FRF for the 6-state ABSORBER-FREE baseline (full mh, no hidden MSD).

    This is the discriminating reference. Asking "is there a peak" is the wrong question for a
    10 % absorber mass: measured, the truth and the baseline differ by only 0.55 % median over
    130-180 Hz across all nine channels. The right question is whether the DATA prefers the
    model that has the absorber, and that is answered channel by channel.
    """
    mh, m1, m2, mb = plant.mh, plant.m1, plant.m2, plant.mb
    off = plant._B12 - mh * Y
    M = np.array([[m1 + m2 + mb + mh, off, 0.],
                  [off, plant.Jb + plant.Jh + (m1 + m2) * plant.Lb ** 2 / 4
                   + mh * plant.d ** 2 + mh * Y ** 2, -mh * plant.d],
                  [0., -mh * plant.d, mh]])
    Minv = np.linalg.inv(M)
    A = np.block([[np.zeros((3, 3)), np.eye(3)], [-Minv @ plant._K3, -Minv @ plant._C3]])
    B = np.vstack([np.zeros((3, 3)), Minv])
    C = np.hstack([np.eye(3), np.zeros((3, 3))])
    P = plant.P_np
    G = np.empty((len(f_hz), 3, 3), complex)
    I6 = np.eye(6)
    for i, fr in enumerate(f_hz):
        G[i] = P.T @ C @ np.linalg.solve(2j * np.pi * fr * I6 - A, B) @ P
    return G


def lpm(U, Y, k_idx, n_half=N_HALF, order=R_ORDER):
    """MIMO local polynomial FRF.

    U : (M, K, nu) complex DFT of the inputs, M records
    Y : (M, K, ny) complex DFT of the outputs
    Returns G (len(k_idx), ny, nu) and the per-bin condition number.
    """
    M, K, nu = U.shape
    ny = Y.shape[2]
    G = np.empty((len(k_idx), ny, nu), complex)
    cond = np.empty(len(k_idx))
    npoly = order + 1
    for a, k0 in enumerate(k_idx):
        lo, hi = k0 - n_half, k0 + n_half + 1
        d = np.arange(lo, hi) - k0                            # local frequency offset
        # regressor: [U_j * d^r  for j,r] and [d^r for r]  (the transient column block)
        powers = np.stack([d.astype(float) ** r for r in range(npoly)], axis=1)   # (W, npoly)
        blocks = []
        for j in range(nu):
            uj = U[:, lo:hi, j]                               # (M, W)
            blocks.append((uj[:, :, None] * powers[None, :, :]).reshape(M * len(d), npoly))
        blocks.append(np.tile(powers, (M, 1)))                # transient, same for every record
        Phi = np.hstack(blocks)                               # (M*W, (nu+1)*npoly)
        cond[a] = np.linalg.cond(Phi)
        for i in range(ny):
            rhs = Y[:, lo:hi, i].reshape(-1)
            theta, *_ = np.linalg.lstsq(Phi, rhs, rcond=None)
            G[a, i, :] = theta[0::npoly][:nu]                 # the r=0 coefficient of each G_ij
    return G, cond


def run_pass(spec):
    """One (record set, band, window) configuration. Returns f, G, G_truth, cond."""
    RECORDS = spec['records']
    F_LO, F_HI = spec['band']
    Us, Ys = [], []
    for name in RECORDS:
        rec = load_record(name, fs_new=int(FS))
        u, y = np.asarray(rec['u'], float), np.asarray(rec['y'], float)
        n = len(u)
        # LEAKAGE SUPPRESSION, and this is load-bearing, not cosmetic.
        # `HEURISTIC` a Hann window on u and y before the DFT. The book's preferred route is the
        # LPM's transient term ALONE, with no window. Measured on these records that fails: with a
        # rectangular window the median relative error against the frozen truth FRF over 1-200 Hz
        # is 1.9e+02, and 1.4e+02 above 100 Hz; with a Hann window it is 4.6e-01 and 1.8e-01.
        # Reason: the records are NON-PERIODIC and the output spectrum spans ~8 decades (y is
        # 5.7e-2 m at 5 Hz and the plant rolls off as 1/f^2), so leakage from the low-frequency
        # content swamps the true high-frequency response, and that leakage is not a low-order
        # polynomial across the LPM window. The window is applied IDENTICALLY to u and y, so the
        # ratio is unbiased apart from a 3-bin smearing; at df = 0.083 Hz that is 0.25 Hz against
        # an absorber half-power bandwidth of 16.7 Hz, i.e. negligible.
        # The clean fix is periodic excitation measured over an integer number of periods, which
        # removes leakage entirely. That is PLAN-BLA.md step 4.
        w = np.hanning(n) if spec['window'] else np.ones(n)
        u = (u - u.mean(0)) * w[:, None]
        y = (y - y.mean(0)) * w[:, None]
        Us.append(np.fft.rfft(u, axis=0))
        Ys.append(np.fft.rfft(y, axis=0))
        print(f'loaded {name}: N={n}, Y_op={rec["Y_op"]:+.4f} m, df={FS/n:.4f} Hz')
    U = np.stack(Us); Y = np.stack(Ys)
    N = 2 * (U.shape[1] - 1)
    freqs = np.fft.rfftfreq(N, d=1.0 / FS)
    df = freqs[1]

    band = np.where((freqs >= F_LO) & (freqs <= F_HI))[0]
    band = band[(band >= N_HALF) & (band < len(freqs) - N_HALF - 1)]
    nh = spec.get('n_half', N_HALF)
    print(f'band {F_LO} to {F_HI} Hz -> {len(band)} bins, window {2*nh+1} bins '
          f'({(2*nh+1)*df:.3f} Hz), local polynomial order {R_ORDER}, {len(RECORDS)} records')
    # Sizing rule counts EQUATIONS (M records x W bins), not bins, so a narrow window is legal.
    neq, nunk = len(RECORDS) * (2 * nh + 1), (3 + 1) * (R_ORDER + 1)
    print(f'sizing rule  M*W > (n_u+1)*(R+1) : {neq} > {nunk}  '
          f'{"OK" if neq > nunk else "VIOLATED"}')

    G, cond = lpm(U, Y, band, n_half=nh)
    f = freqs[band]
    print(f'condition number: median {np.median(cond):.2e}, max {np.max(cond):.2e}')
    Gt = truth_frf(f)
    m = np.ones(len(f), bool)
    rel = np.abs(G[m] - Gt[m]) / np.maximum(np.abs(Gt[m]), 1e-30)
    print(f'vs frozen truth FRF over this band: median relative error {np.median(rel):.3e}, '
          f'90th pct {np.percentile(rel, 90):.3e}')
    return f, G, Gt, cond, float(np.median(rel)), float(np.percentile(rel, 90))


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    F_ABS, F_SLOW = 157.894, 5.122
    TARGET = {'absorber': (F_ABS, 2.0), 'slow': (F_SLOW, 0.3)}
    summary, store = {}, {}
    for spec in PASSES:
        print('\n' + '=' * 78)
        print(f"PASS '{spec['name']}'  band {spec['band'][0]}-{spec['band'][1]} Hz  "
              f"window={'hann' if spec['window'] else 'none (periodic multisine)'}")
        print('=' * 78)
        f, G, Gt, cond, relmed, relp90 = run_pass(spec)
        store[spec['name']] = (f, G, Gt)
        f0, tol = TARGET[spec['name']]
        w = np.abs(f - f0) <= tol
        best = None
        for i in range(3):
            for j in range(3):
                mag = np.abs(G[:, i, j])
                if w.sum() < 3:
                    continue
                idx = np.where(w)[0]
                # Shoulder = the OUTER 20 % of the band on each side, not a few bins either
                # side of the peak. The absorber's half-power bandwidth is 2*zeta*f_n = 16.7 Hz,
                # i.e. ~200 bins, so a +-40-bin shoulder sits INSIDE the resonance and
                # under-reports prominence by design. Measured: that mistake reported +1.7 dB
                # for a peak whose FRF matches the truth to 14.5 % across the whole band.
                q = max(3, int(0.2 * len(f)))
                shoulder = np.median(np.concatenate([mag[:q], mag[-q:]]))
                pk = mag[idx].max(); fpk = f[idx][int(np.argmax(mag[idx]))]
                prom = 20 * np.log10(pk / max(shoulder, 1e-300))
                if best is None or prom > best[0]:
                    best = (prom, i, j, fpk, pk)
        prom, i, j, fpk, pk = best
        err = 100 * (fpk - f0) / f0
        print(f"  peak location: truth {f0:.3f} Hz, estimate {fpk:.3f} Hz ({err:+.2f} %) "
              f"in G[{CHAN[i]}<-{CHAN[j]}], prominence {prom:+.1f} dB")

        # ---- the real acceptance test: does the DATA prefer the model WITH the absorber? ----
        Gb = baseline_frf(f)
        k = int(np.argmin(np.abs(f - f0)))
        print(f'  model discrimination at {f[k]:.2f} Hz, per channel '
              f'(|est-truth8| vs |est-base6|, both relative to the model):')
        best_ch, best_sep = None, -np.inf
        for a in range(3):
            for b in range(3):
                e8 = abs(G[k, a, b] - Gt[k, a, b]) / max(abs(Gt[k, a, b]), 1e-30)
                e6 = abs(G[k, a, b] - Gb[k, a, b]) / max(abs(Gb[k, a, b]), 1e-30)
                feat = abs(Gt[k, a, b] - Gb[k, a, b]) / max(abs(Gb[k, a, b]), 1e-30)
                if feat > best_sep:
                    best_sep, best_ch = feat, (a, b, e8, e6, feat)
                print(f'    G[{CHAN[a]}<-{CHAN[b]}]  err vs truth8 {e8:7.4f}   '
                      f'err vs base6 {e6:7.4f}   absorber feature size {feat:7.4f}')
        a, b, e8, e6, feat = best_ch
        ok = (abs(fpk - f0) <= tol) and (feat > 0.05) and (e8 < 0.5 * e6)
        print(f'  ACCEPTANCE {spec["name"]}: most discriminating channel G[{CHAN[a]}<-{CHAN[b]}], '
              f'absorber changes it by {100*feat:.1f} %; estimate is {100*e8:.2f} % from the '
              f'8-state truth and {100*e6:.2f} % from the absorber-free baseline  -> '
              f'{"PASS" if ok else "FAIL"}')
        summary[spec['name']] = dict(
            records=list(spec['records']), band=list(spec['band']), window=spec['window'],
            f_truth=f0, f_peak=float(fpk), err_pct=float(err), prominence_db=float(prom),
            peak_channel=f'{CHAN[i]}<-{CHAN[j]}',
            discriminating_channel=f'{CHAN[a]}<-{CHAN[b]}',
            feature_size=float(feat), err_vs_truth8=float(e8), err_vs_baseline6=float(e6),
            passed=bool(ok), cond_median=float(np.median(cond)),
            rel_err_median=relmed, rel_err_p90=relp90)

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(3, 3, figsize=(11, 8))
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            for nm, (f, G, Gt) in store.items():
                ax.loglog(f, np.abs(Gt[:, i, j]), color='0.75', lw=2.0)
                ax.loglog(f, np.abs(G[:, i, j]), color='k', lw=0.8)
            for f0 in (F_SLOW, F_ABS):
                ax.axvline(f0, color='k', lw=0.5, ls=(0, (3, 3)))
            ax.set_title(f'{CHAN[i]} <- {CHAN[j]}', fontsize=8)
            ax.grid(True, which='both', lw=0.3, alpha=0.5)
            if i == 2:
                ax.set_xlabel('Hz', fontsize=8)
            if j == 0:
                ax.set_ylabel('|G| [m/N]', fontsize=8)
    fig.suptitle('Step 1: nonparametric MIMO FRF, local polynomial method. '
                 'grey = frozen truth, black = estimate. '
                 'Left cluster: APRBS 1-50 Hz. Right cluster: standstill multisine 130-180 Hz.',
                 fontsize=8)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG, f'lpm-frf-step1.{ext}'), dpi=180)
    plt.close(fig)
    print(f'\nwrote {os.path.join(FIG, "lpm-frf-step1.png")}')
    np.savez(os.path.join(OUT, 'lpm_frf.npz'),
             **{f'{k}_{n}': v for k, (f_, G_, Gt_) in store.items()
                for n, v in (('f', f_), ('G', G_), ('Gt', Gt_))})
    with open(os.path.join(OUT, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f'wrote {os.path.join(OUT, "summary.json")}')


if __name__ == '__main__':
    main()
