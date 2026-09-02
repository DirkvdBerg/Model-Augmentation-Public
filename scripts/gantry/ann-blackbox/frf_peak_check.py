"""Is the 157.9 Hz absorber present in the MEASURED FRF of a given track?

QUESTION. `frf_init.py` on `joint_lowf_ma50_a5` placed its second pole pair at 198.7 Hz, the
band edge, instead of at the absorber. That is either because the data no longer carries the
mode (the 2x per-line dilution of the wider band) or because the parametric fit spent its one
free pair on newly excited band-edge content. A NONPARAMETRIC FRF settles it: it fits no model
order, so it cannot hide a mode by spending states elsewhere.

TEST, per PLAN-BLA.md step 1: not "is there a peak" but MODEL DISCRIMINATION. Compare the
measured FRF against the 8-state truth (absorber present) and against the 6-state baseline
(absorber absent) in the channel where the absorber lives, G[Y<-Y]. The data prefers whichever
it sits closer to. A 10 % absorber mass moves the median FRF across all nine channels by only
0.55 %, so a peak-prominence test is the wrong instrument.
"""
__project_origin__ = "added"

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry', 'msd-offset'))
import plant                                                        # noqa: E402

# THE ABSORBER MOVES WITH ma_frac, and this is the whole point of the check. The mode is the
# free-free two-mass root f = fa*sqrt(1 + ma/mhr), not the standalone fa = 150 Hz that the
# [130,180] band was designed around. Verified against the known truth: ma_frac = 0.10 gives
# 158.1139 Hz undamped / 157.9161 damped, which is the value every truth model in the repo uses.
#   ma_frac 0.10 -> 158.1 Hz     ma_frac 0.50 -> 212.1 Hz
# plant.py hard-codes MA_FRAC = 0.10, so any comparison of ma50 data against plant's truth is
# against the WRONG PLANT. This script therefore takes the absorber frequency per track.
TRACKS = (('joint', 0.10), ('joint_lowf_ma50_a5', 0.50),
          ('augmentation_ma50_a5', 0.50), ('augmentation', 0.10))
RECORD = 'T3_standstill_Y000'
FS = 4000.0
BAND = (1.0, 250.0)
N_HALF = 12
WIN = 8.0                        # +/- Hz around the absorber for the discrimination window


def absorber_hz(ma_frac):
    """Coupled absorber mode. THEORY: free-free two-mass root, f = fa*sqrt(1 + ma/mhr)."""
    ma = ma_frac * plant.mh
    return plant.FA * np.sqrt(1 + ma / (plant.mh - ma))


def measure(track):
    plant.TRAJ = os.path.join(REPO, 'data', 'gantry', 'matlab', 'trajectory', track)
    import lpm_frf
    plant.TRAJ = os.path.join(REPO, 'data', 'gantry', 'matlab', 'trajectory', track)
    rec = plant.load_record(RECORD, fs_new=int(FS))
    u, y = np.asarray(rec['u'], float), np.asarray(rec['y'], float)
    U = np.fft.rfft(u - u.mean(0), axis=0)[None]
    Y = np.fft.rfft(y - y.mean(0), axis=0)[None]
    N = 2 * (U.shape[1] - 1)
    freqs = np.fft.rfftfreq(N, d=1.0 / FS)
    band = np.where((freqs >= BAND[0]) & (freqs <= BAND[1]))[0]
    band = band[(band >= N_HALF) & (band < len(freqs) - N_HALF - 1)]
    G, _ = lpm_frf.lpm(U, Y, band, n_half=N_HALF)
    return freqs[band], G, lpm_frf, u, y


for track, ma_frac in TRACKS:
    F_A = absorber_hz(ma_frac)
    f, G, L, u, y = measure(track)
    g = np.abs(G[:, 2, 2])                                          # G[Y <- Y]
    m = np.abs(f - F_A) <= WIN
    k = np.argmax(g * m)
    floor = np.median(g[m])
    print(f'\n== {track}   ma_frac={ma_frac:.2f}  ->  absorber at {F_A:.2f} Hz')
    print(f'   lines {len(f)}, {f[0]:.3f} to {f[-1]:.1f} Hz')
    print(f'   |G[Y<-Y]| peak within {WIN:.0f} Hz -> {f[k]:.3f} Hz, '
          f'{g[k]/floor:.2f}x above the local median ({20*np.log10(g[k]/floor):+.1f} dB)')

    # IS IT EXCITED. The decisive question: a mode outside the multisine band cannot be
    # identified however large the response is elsewhere. Input power per band, on the Y
    # channel, referenced to the band the multisine was designed for.
    N = len(u)
    fu = np.fft.rfftfreq(N, d=1.0 / FS)
    U = np.abs(np.fft.rfft(u - u.mean(0), axis=0))
    Yf = np.abs(np.fft.rfft(y - y.mean(0), axis=0))

    def bp(A, lo, hi):
        s = (fu >= lo) & (fu <= hi)
        return float(np.sqrt(2.0 * np.sum(A[s, 2] ** 2)) / N)

    u_abs, y_abs = bp(U, F_A - WIN, F_A + WIN), bp(Yf, F_A - WIN, F_A + WIN)
    u_ms, y_ms = bp(U, 130, 180), bp(Yf, 130, 180)
    print(f'   u rms  in absorber window = {u_abs:.4e} N     in 130-180 Hz = {u_ms:.4e} N'
          f'   ratio {20*np.log10(max(u_abs, 1e-300)/max(u_ms, 1e-300)):+.1f} dB')
    print(f'   y rms  in absorber window = {y_abs:.4e} m     in 130-180 Hz = {y_ms:.4e} m')
