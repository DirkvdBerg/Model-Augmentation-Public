"""make_v1f_figure.py -- the PHYSICS-clean "system is zero-mean": open-loop, same input.

v1f drove each logical axis open-loop with a sustained DC offset + a 150 Hz tone (the MSD resonance),
SAME input to the truth plant (with MSD) and the baseline (without), from 4 operating points Y0.
Read the steady DC by harmonic least-squares on all three logical channels (dX, Theta, dY). The
quantity that matters is dc_diff_mechB = DC(truth) - DC(baseline) = the DC the truth carries that the
baseline LACKS -- i.e. exactly what the ANN would need to reproduce as a constant.

No controller (open loop) -> a real static DC on the K=0 axes would integrate into an unmissable ramp;
this is the physics-clean version of the closed-loop v1b comparison. Result: the largest such DC is
~1e-7 (static mass-split asymmetry) and the genuine MSD resonance-rectification DC is ~3e-10 -- both
far below the absorber signal (std ~2.2e-5) they live in, and orders below the ANN's parked DC.

Output: figures/meeting/meeting_fig0b_v1f_openloop.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat

plt.rcParams.update({'font.size': 13, 'axes.titlesize': 15, 'axes.labelsize': 13})
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'figures', 'meeting'); os.makedirs(OUT, exist_ok=True)
DRIVES = ['X', 'Theta', 'Y']; CHANS = ['dX', 'Theta', 'dY']


def main():
    m = loadmat(os.path.join(HERE, 'data', 'v1f_results.mat'), squeeze_me=True, struct_as_record=False)
    # max over the 4 operating points of |truth - baseline DC|, per (drive, read-channel)
    vals = np.zeros((len(DRIVES), len(CHANS)))
    da_rms = {}
    for di, d in enumerate(DRIVES):
        s = m[d]
        diff = np.abs(np.asarray(s.dc_diff_mechB, float))     # (3 chan, 4 Y0)
        vals[di] = diff.max(axis=1)
        da_rms[d] = float(np.asarray(s.delta_a_rms, float).max())
    da_scale = da_rms['Y']                                     # absorber excited only on the Y drive

    x = np.arange(len(DRIVES)); w = 0.26
    cols = ['#95a5a6', '#95a5a6', '#95a5a6']
    fh, ax = plt.subplots(figsize=(10.5, 5.6))
    for ci, ch in enumerate(CHANS):
        bars = ax.bar(x + (ci - 1) * w, vals[:, ci], w, label=f'read {ch}',
                      color=['#e67e22' if (DRIVES[j] == 'Y' and ch == 'Theta') else
                             ('#c0392b' if vals[j, ci] == vals.max() else '#7f8c8d') for j in range(len(DRIVES))])
    ax.set_yscale('log'); ax.set_ylim(1e-10, 1e-4)
    ax.axhline(da_scale, color='#2980b9', ls='--', lw=1.6)
    ax.text(len(DRIVES) - 0.5, da_scale * 1.15, f'absorber signal scale ({da_scale:.1e}) = the effect being modeled',
            ha='right', va='bottom', fontsize=10, color='#2980b9')
    # annotate the two physically meaningful bars
    jY = DRIVES.index('Y'); cT = CHANS.index('Theta')
    ax.annotate('MSD resonance rectification\n(real, ~3e-10)', xy=(jY + (cT - 1) * w, vals[jY, cT]),
                xytext=(1.4, 5e-10), fontsize=10, color='#e67e22',
                arrowprops=dict(arrowstyle='->', color='#e67e22'))
    jmax, cmax = np.unravel_index(np.argmax(vals), vals.shape)
    ax.annotate('largest physics DC\n= static mass-split asymmetry (~1e-7)',
                xy=(jmax + (cmax - 1) * w, vals[jmax, cmax]), xytext=(-0.35, 6e-7),
                fontsize=10, color='#c0392b', arrowprops=dict(arrowstyle='->', color='#c0392b'))
    ax.set_xticks(x); ax.set_xticklabels([f'drive {d}' for d in DRIVES])
    ax.set_ylabel('|DC(truth) - DC(baseline)|  per read channel   [m/s or rad]')
    ax.set_title('Open-loop, same input: the DC the truth carries that the baseline lacks is <= 1e-7')
    ax.legend(loc='upper center', ncol=3, fontsize=11)
    ax.grid(True, axis='y', alpha=0.3, which='both')
    fh.text(0.5, -0.02, 'Each plant driven open-loop with DC + 150 Hz tone, identical input, 4 operating '
            'points. Every physics DC sits far below the absorber signal it lives in\n(the MSD was excited '
            f'only on the Y drive: delta_a rms {da_scale:.1e} vs ~1e-9 on X/Theta drives). No constant the '
            'baseline lacks -> the physics is zero-mean.', ha='center', fontsize=9.5, style='italic')
    fh.tight_layout()
    p = os.path.join(OUT, 'meeting_fig0b_v1f_openloop.png')
    fh.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fh)
    print('saved', p)
    for di, d in enumerate(DRIVES):
        print(f'  drive {d}: max|truth-base DC| per chan {dict(zip(CHANS, [f"{v:.1e}" for v in vals[di]]))} '
              f'| delta_a rms {da_rms[d]:.1e}')


if __name__ == '__main__':
    main()
