"""What does the open-loop record actually look like? Input, output, drift, spectra.

OL1_multisine_Yp10 is the one existing open-loop record: random-phase multisine in [130,180] Hz,
no reference, no controller, no feedforward, 12 s at 20 kHz, launched from rest at Y = 0.10.
This plots it as a record rather than as a model comparison, to judge the construction before
generating a full set the same way.
"""
__project_origin__ = "added"

import os
import numpy as np
from scipy.io import loadmat
from scipy.signal import welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
REC = os.path.join(REPO, 'data', 'gantry', 'matlab', 'trajectory', 'openloop',
                   'OL1_multisine_Yp10.mat')
CH = ['X1', 'X2', 'Y']
C_U, C_Y = '#0072B2', '#D55E00'

dm = loadmat(REC, squeeze_me=True)
ts = float(dm['dt'])
u = np.asarray(dm['u_total'], float)
y = np.asarray(dm['y'], float)
t = np.arange(len(u)) * ts
Y0 = float(dm['Y_op'])

print('record %s' % os.path.basename(REC))
print('  %d samples, %.1f s at %.0f kHz, Y0 = %.4f m' % (len(t), t[-1], 1e-3 / ts, Y0))
print('  u  rms [%.3f %.3f %.3f] N   DC [%+.2e %+.2e %+.2e] N' % (*u.std(axis=0), *u.mean(axis=0)))
print('  u  peak [%.1f %.1f %.1f] N' % tuple(np.abs(u).max(axis=0)))
print('  y  start [%+.6f %+.6f %+.6f] m' % tuple(y[0]))
print('  y  end   [%+.6f %+.6f %+.6f] m' % tuple(y[-1]))
print('  drift    [%+.3e %+.3e %+.3e] m' % tuple(y[-1] - y[0]))
# AC content after the settling transient, detrended by a moving mean over one 130 Hz period
m = (t >= 6.0) & (t <= t[-1] - 0.05)
w = int(round((1 / 130.0) / ts)) | 1
k = np.ones(w) / w
ac = np.stack([y[:, c] - np.convolve(y[:, c], k, mode='same') for c in range(3)], axis=1)
print('  y  AC rms after 6 s [%.3e %.3e %.3e] m' % tuple(ac[m].std(axis=0)))

fig, axes = plt.subplots(3, 3, figsize=(14.0, 8.4))
iz = slice(int(8.0 / ts), int(8.02 / ts))
for c in range(3):
    ax = axes[0, c]
    ax.plot(t, u[:, c], color=C_U, lw=0.4)
    ax.set_title('%s' % CH[c], fontsize=11)
    ax.grid(alpha=0.25, lw=0.5)
    if c == 0:
        ax.set_ylabel('input force [N]\nfull record')
    ax.text(0.97, 0.05, 'rms %.1f N' % u[:, c].std(), transform=ax.transAxes,
            ha='right', fontsize=8)

    ax = axes[1, c]
    ax.plot(t, (y[:, c] - y[0, c]) * 1e6, color=C_Y, lw=0.6)
    ax.axhline(0, color='#999999', lw=0.8, ls=':')
    ax.grid(alpha=0.25, lw=0.5)
    if c == 0:
        ax.set_ylabel('position - start [$\\mu$m]\nfull record')
    ax.text(0.97, 0.10, 'drift %+.0f $\\mu$m' % ((y[-1, c] - y[0, c]) * 1e6),
            transform=ax.transAxes, ha='right', fontsize=8)

    ax = axes[2, c]
    f, Pu = welch(u[:, c], fs=1 / ts, nperseg=8192, detrend=False)
    f2, Py = welch(y[m, c], fs=1 / ts, nperseg=8192, detrend=False)
    ax.loglog(f, np.sqrt(Pu) / np.sqrt(Pu).max(), color=C_U, lw=0.8, label='input')
    ax.loglog(f2, np.sqrt(Py) / np.sqrt(Py).max(), color=C_Y, lw=0.8, label='output')
    ax.axvspan(130, 180, color='#999999', alpha=0.2)
    ax.set_xlim(1, 1e4)
    ax.set_ylim(1e-10, 3)
    ax.grid(alpha=0.25, lw=0.5, which='both')
    ax.set_xlabel('frequency [Hz]')
    if c == 0:
        ax.set_ylabel('normalised spectrum\n(shaded = 130-180 Hz)')
        ax.legend(fontsize=8, frameon=False, loc='lower left')
for c in range(3):
    axes[1, c].set_xlabel('') if c else None
axes[1, 0].set_xlabel('time [s]')
axes[1, 1].set_xlabel('time [s]')
axes[1, 2].set_xlabel('time [s]')

fig.suptitle('OL1_multisine_Yp10: open-loop record, multisine only, no reference and no '
             'controller.\nRow 2 shows the rectification drift; it settles and is common mode.',
             fontsize=11.5, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.945])
out = os.path.join(HERE, 'figures')
os.makedirs(out, exist_ok=True)
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(out, 'ol1_record.%s' % ext), dpi=160, bbox_inches='tight')
print('wrote %s' % os.path.join(out, 'ol1_record.png'))
