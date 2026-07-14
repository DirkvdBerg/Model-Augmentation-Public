"""
f5_horizon.py -- F5/C5 (plan doc §12): a horizon that CAN see the drift exists; the training
window cannot. Zero simulation: reuses the saved F2 traces (figures/f2_dc.npz).

CLAIM: the drift is below detectability at the 0.1 s training window yet rises above the
absorber scale at T > T*, so the window's failure is horizon-limited BLINDNESS to the
free-run COST (d8: in-window the DC is even preferred), not fundamental invisibility.
FALSIFIED IF the drift contribution never exceeded the floor at any horizon.

Curves (cumulative Y-error RMS over [0, T]):
  full ANN            -- everything (encoder transient + absorber ripple + drift)
  DC removed          -- the bounded reference (same run minus the measured DC)
  drift contribution  -- full minus DC-removed traces, RMS: isolates the DC's effect
Marks: sigma(delta_a) floor; the 0.1 s training window; the ~0.5 s max feasible BPTT window
(nf=2000; nf=4000 = 566 MB wall); T* where the drift contribution crosses the floor.

Run: conda run -n GraduationProject python scripts/gantry/drift-demo/f5_horizon.py  (seconds)
"""
__project_origin__ = "added"

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import demo_common as dm

d = np.load(os.path.join(dm.OUT_DIR, 'f2_dc.npz'), allow_pickle=True)
e_full, e_deb = d['e_full'][:, 2], d['e_deb'][:, 2]      # Y channel
ts, floor, tag = float(d['ts']), float(d['absorber_rms']), str(d['ckpt'])
e_drift = e_full - e_deb                                  # the DC's isolated effect
M = len(e_full)

n_grid = np.unique(np.geomspace(8, M, 240).astype(int))
T = n_grid * ts
cum = lambda e: np.array([np.sqrt(np.mean(e[:n] ** 2)) for n in n_grid])
c_full, c_deb, c_drift = cum(e_full), cum(e_deb), cum(e_drift)

TRAIN_WIN = 0.1                                           # nf=400 training window [s]
BPTT_MAX  = 0.5                                           # nf=2000; nf=4000 = 566 MB wall
ix = np.argmax(c_drift > floor)
T_star = T[ix] if c_drift[ix] > floor else np.nan
print(f'checkpoint {tag}:  drift-contribution RMS crosses sigma(delta_a)={floor:.2e} m '
      f'at T* = {T_star:.2f} s  (training window {TRAIN_WIN} s, max feasible BPTT ~{BPTT_MAX} s)')
for Tq in (TRAIN_WIN, BPTT_MAX, T[-1]):
    i = np.argmin(np.abs(T - Tq))
    print(f'  T={T[i]:6.2f}s   full={c_full[i]:.3e}   DC-removed={c_deb[i]:.3e}   '
          f'drift-contrib={c_drift[i]:.3e}  ({c_drift[i]/floor:5.2f}x floor)')

fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(T, c_full, 'C3', lw=1.3, label='full ANN (everything)')
ax.loglog(T, c_deb, 'C0', lw=1.3, label='DC removed (bounded reference)')
ax.loglog(T, c_drift, 'C4', lw=1.6, label='DRIFT CONTRIBUTION (full minus DC-removed)')
ax.axhline(floor, color='0.4', ls=':', lw=1.2,
           label=f'sigma(delta_a) = {floor:.1e} m (absorber scale = detectability floor; '
                 'on real data: the noise floor)')
ax.axvline(TRAIN_WIN, color='k', ls='--', lw=1.0, label=f'training window ({TRAIN_WIN} s, nf=400)')
ax.axvline(BPTT_MAX, color='k', ls='-.', lw=1.0,
           label=f'max feasible BPTT window (~{BPTT_MAX} s, nf=2000; nf=4000 = 566 MB)')
if np.isfinite(T_star):
    ax.axvline(T_star, color='C4', ls=':', lw=1.2)
    ax.annotate(f'T* = {T_star:.2f} s\n(first horizon that SEES the drift)',
                xy=(T_star, floor), xytext=(T_star * 1.6, floor * 0.25),
                arrowprops=dict(arrowstyle='->', color='C4'), fontsize=8, color='C4')
ax.set_xlabel('evaluation horizon T [s]')
ax.set_ylabel('cumulative Y-error RMS over [0, T] [m]  (V1 free-run)')
ax.grid(True, which='both')
ax.legend(fontsize=7, loc='upper left')
ax.set_title('At which horizon does the drift become visible?\n'
             '(below T*: genuinely BLIND (0.11x floor at the 0.1 s window); above T*: visible '
             'but the windowed loss still PREFERS/tolerates the DC, d8/d12)')
dm.add_provenance(fig, f'{tag} | traces from f2_dc.npz (12 s V1 free-run) | '
                       f'drift contribution = full-ANN trace minus DC-removed trace')
fig.tight_layout()
p = os.path.join(dm.OUT_DIR, 'f5_horizon.png')
fig.savefig(p, dpi=150)
np.savez(os.path.join(dm.OUT_DIR, 'f5_horizon.npz'),
         T=T, c_full=c_full, c_deb=c_deb, c_drift=c_drift, floor=floor, T_star=T_star)
print(f'Saved: {p}')
