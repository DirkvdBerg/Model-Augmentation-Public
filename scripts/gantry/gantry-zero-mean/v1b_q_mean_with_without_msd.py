"""v1b: does adding the hidden MSD shift the MEAN of any measured signal?

README section 8, V1 extension (Jan's literal ask, 2026-07-15 meeting, Theme A):
compare the per-channel time-means of the recorded signals on matched records,
with-MSD vs no-MSD. This is definition-independent: it does not rely on how
delta_a is defined. If the offset mass (L0 = 0.10 m in +Y) exerted any static
one-sided effect, it must appear as a mean shift in the outputs q and/or in the
closed-loop force u (a static pull is split between a position offset and a
holding force by the loop; checking BOTH closes every path).

Method: for each matched record (T1..T12 present in both folders), load the raw
.mat at native 20 kHz (no training pipeline; means need no resampling), align to
the common length, and compute dmean = mean(with) - mean(without) per channel:
  stage positions  y        = [X1, X2, Y]        [m]
  logical states   x_logical = [X, Theta, Y, dX, dTheta, dY]
  stage forces     u_total  = [F_X1, F_X2, F_Y]  [N]
Uncertainty: the record is split into time-paired segments; dmean is computed
per segment and the standard error of the record-level dmean is
std(segment dmeans)/sqrt(n_seg).
  # THEORY: standard error of the mean, SE = s/sqrt(n) (independent segments)
  # HEURISTIC: 1 s segments (12 per record); long vs the 150 Hz ripple, short
  #            enough for n=12 spread estimate
Error bars in the figures are 2*SE.
  # THEORY: ~95% normal-approximation interval

Hypothesis test posed by the figures: "with-MSD minus no-MSD mean = 0 within
2*SE on every channel". A real static effect of the mass shows as points off
zero beyond their error bars, consistently across records.

Caveat printed with the results: the two datasets use different multisine bands
(with-MSD 130-180 Hz, no-MSD 1-7 Hz; gtd_config.m). Both are zero-mean, so the
MEANS are comparable; the ripple content is not, and the segment spread absorbs
that difference into the error bars.

Outputs (folder convention, README header):
  figures -> scripts/gantry/gantry-zero-mean/figures/v1b_dmean_{positions,velocities,forces}.png
  data    -> scripts/gantry/gantry-zero-mean/data/v1b_dmean.npz  (+ printed table)
"""

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DIR_W = os.path.join(REPO, 'data', 'gantry', 'matlab', 'trajectory', 'augmentation')
DIR_N = os.path.join(DIR_W, 'baseline')
FIG_DIR = os.path.join(HERE, 'figures')
DAT_DIR = os.path.join(HERE, 'data')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DAT_DIR, exist_ok=True)

FS = 20e3                       # native rate of the .mat records [Hz]
SEG_SEC = 1.0                   # HEURISTIC: segment length for the SE estimate [s]

# channel table: (group, panel label, unit)
CHANNELS = [
    ('stage_pos', 'X1 [m]'), ('stage_pos', 'X2 [m]'), ('stage_pos', 'Y [m]'),
    ('log_pos', 'X [m]'), ('log_pos', 'Theta [rad]'), ('log_pos', 'Y [m]'),
    ('log_vel', 'dX [m/s]'), ('log_vel', 'dTheta [rad/s]'), ('log_vel', 'dY [m/s]'),
    ('force', 'F_X1 [N]'), ('force', 'F_X2 [N]'), ('force', 'F_Y [N]'),
]


def load_record(directory, filename):
    """Raw signals at native rate: (N,12) matrix in CHANNELS order, plus delta_a."""
    d = loadmat(os.path.join(directory, filename), squeeze_me=True)
    y = np.asarray(d['y'], dtype=np.float64)                    # (N,3) stage positions
    xl = np.asarray(d['x_logical'], dtype=np.float64)           # (N,6) logical states
    u = np.asarray(d['u_total'], dtype=np.float64)              # (N,3) stage forces
    n = min(len(y), len(xl), len(u))
    sig = np.hstack([y[:n], xl[:n, :3], xl[:n, 3:6], u[:n]])    # (n,12)
    da = np.asarray(d['delta_a'], dtype=np.float64)[:n] if 'delta_a' in d else None
    return sig, da


def matched_records():
    in_n = {f for f in os.listdir(DIR_N) if f.endswith('.mat')}
    recs = sorted((f for f in os.listdir(DIR_W) if f.endswith('.mat') and f in in_n),
                  key=lambda f: (len(f), f))                    # T1..T9 before T10..T12
    if not recs:
        raise RuntimeError(f'no matched .mat records between {DIR_W} and {DIR_N}')
    return recs


def paired_dmean(sig_w, sig_n):
    """Record-level dmean per channel + SE from time-paired segment dmeans."""
    n = min(len(sig_w), len(sig_n))
    seg = int(round(SEG_SEC * FS))
    n_seg = n // seg
    dm_segs = np.empty((n_seg, sig_w.shape[1]))
    for s in range(n_seg):
        sl = slice(s * seg, (s + 1) * seg)
        dm_segs[s] = sig_w[sl].mean(axis=0) - sig_n[sl].mean(axis=0)
    dmean = sig_w[:n_seg * seg].mean(axis=0) - sig_n[:n_seg * seg].mean(axis=0)
    se = dm_segs.std(axis=0, ddof=1) / np.sqrt(n_seg)   # THEORY: SE of the mean
    return dmean, se, n_seg


recs = matched_records()
labels = [f.replace('.mat', '') for f in recs]
n_ch = len(CHANNELS)
DM = np.empty((len(recs), n_ch))
SE = np.empty((len(recs), n_ch))
STD_W = np.empty((len(recs), n_ch))
DA_MEAN = np.full(len(recs), np.nan)
DA_STD = np.full(len(recs), np.nan)

print(f'matched records ({len(recs)}): {", ".join(labels)}')
print(f'with-MSD: {DIR_W}\nno-MSD:   {DIR_N}')
print('NOTE: multisine bands differ by design (with-MSD 130-180 Hz, no-MSD 1-7 Hz,'
      ' gtd_config.m); both zero-mean, so means are comparable.')

for i, f in enumerate(recs):
    sig_w, da = load_record(DIR_W, f)
    sig_n, _ = load_record(DIR_N, f)
    DM[i], SE[i], n_seg = paired_dmean(sig_w, sig_n)
    STD_W[i] = sig_w.std(axis=0)
    if da is not None:
        DA_MEAN[i], DA_STD[i] = da.mean(), da.std()
    print(f'  {labels[i]:24s} N={min(len(sig_w), len(sig_n))} segs={n_seg}')

print('\n=== dmean = mean(with-MSD) - mean(no-MSD), per channel per record ===')
hdr = f'{"record":24s}' + ''.join(f'{lab.split(" ")[0]:>12s}' for _, lab in CHANNELS)
print(hdr)
for i, lab in enumerate(labels):
    print(f'{lab:24s}' + ''.join(f'{DM[i, c]:>12.2e}' for c in range(n_ch)))
print('\n=== |dmean| / (2*SE)  (>1 = significant at ~95%) ===')
print(hdr)
for i, lab in enumerate(labels):
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.abs(DM[i]) / (2 * SE[i])
    print(f'{lab:24s}' + ''.join(f'{ratio[c]:>12.2f}' for c in range(n_ch)))
print('\n=== delta_a (with-MSD only): mean vs std ===')
for i, lab in enumerate(labels):
    print(f'  {lab:24s} mean={DA_MEAN[i]:+.3e}  std={DA_STD[i]:.3e}'
          f'  |mean|/std={abs(DA_MEAN[i]) / DA_STD[i]:.2e}')

np.savez(os.path.join(DAT_DIR, 'v1b_dmean.npz'),
         records=np.array(labels), channels=np.array([lab for _, lab in CHANNELS]),
         dmean=DM, se=SE, std_with=STD_W, da_mean=DA_MEAN, da_std=DA_STD,
         seg_sec=SEG_SEC, fs=FS)
print(f'\nsaved {os.path.join(DAT_DIR, "v1b_dmean.npz")}')


def plot_group(group, fname, suptitle):
    idx = [c for c, (g, _) in enumerate(CHANNELS) if g in group]
    ncols = 3
    nrows = int(np.ceil(len(idx) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.4 * nrows),
                             sharex=True, squeeze=False)
    x = np.arange(len(labels))
    for k, c in enumerate(idx):
        ax = axes[k // ncols][k % ncols]
        ax.axhline(0.0, color='k', lw=0.8)
        ax.errorbar(x, DM[:, c], yerr=2 * SE[:, c], fmt='o', ms=4, capsize=3,
                    color='tab:red', ecolor='tab:gray')
        ax.set_title(CHANNELS[c][1], fontsize=10)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.grid(alpha=0.3)
        if k // ncols == nrows - 1:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=7)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(FIG_DIR, fname)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'saved {out}')


TEST = 'Does adding the MSD shift the mean?  dmean = mean(with) - mean(without), bars = 2*SE'
plot_group({'stage_pos', 'log_pos'}, 'v1b_dmean_positions.png',
           f'{TEST}\npositions: stage [X1, X2, Y] (top), logical [X, Theta, Y] (bottom)')
plot_group({'log_vel'}, 'v1b_dmean_velocities.png',
           f'{TEST}\nlogical velocities [dX, dTheta, dY]')
plot_group({'force'}, 'v1b_dmean_forces.png',
           f'{TEST}\nrecorded total force, stage frame [F_X1, F_X2, F_Y]')
