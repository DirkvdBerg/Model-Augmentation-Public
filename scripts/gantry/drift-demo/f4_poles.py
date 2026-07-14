"""
f4_poles.py -- F4/C1 (plan doc §12): the discrete baseline poles at pipeline conditions.

CLAIM: each K=0 axis (X, Y) carries ONE marginal POSITION pole AT z=1 (free integrator,
s(ms+c) structure) plus one DAMPED velocity pole inside (z=exp(-ts/tau), the bounded tau*dv
settling) -- the marginal poles are there BY CONSTRUCTION, not pushed OUTSIDE by
discretization (the supervisors' "discretizatie op de rand" hypothesis), and not strictly
inside (which would make drift impossible). Theta is sprung: complex pair strictly inside.

Method: Jacobian (torch autograd) of the ACTUAL pipeline discrete step (Gantry_State_Block,
RK4, pipeline ts and up_sample) at the T1 equilibrium (Y_op = -0.3, rest, u = 0), then its
eigenvalues. This is the eigen-check at pipeline conditions, not an idealized ZOH formula.
FALSIFIED IF any X/Y pole were strictly inside (no drift possible) or outside (a
discretization instability, a different problem).

Run: conda run -n GraduationProject python scripts/gantry/drift-demo/f4_poles.py  (seconds)
"""
__project_origin__ = "added"

import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import demo_common as dm
from demo_common import CFG
import drift_common as dc

ts = CFG.ts_new
up = CFG.hp['up_sample']
Y_OP = -0.3                       # T1 operating point

blk = dc.build_baseline_block(ts, up)
x_eq = torch.zeros(6, dtype=torch.float64)
x_eq[2] = Y_OP
u0 = torch.zeros(3, dtype=torch.float64)


def step(x):
    z = torch.cat([x, u0]).view(1, 9, 1)
    return blk(z).view(6)


J = torch.autograd.functional.jacobian(step, x_eq).numpy()   # discrete A at (x_eq, u=0)
lam = np.linalg.eigvals(J)
order = np.argsort(-np.abs(lam))
lam = lam[order]

print(f'Discrete baseline poles at pipeline conditions (ts={ts:.2e}s, up_sample={up}, '
      f'Y_op={Y_OP}):')
print(f"  {'#':>2s} {'Re':>12s} {'Im':>12s} {'|lambda|':>12s} {'|lambda|-1':>12s}")
for i, l in enumerate(lam):
    print(f'  {i:2d} {l.real:>12.8f} {l.imag:>12.4e} {abs(l):>12.8f} {abs(l)-1:>12.3e}')

# Physics: each K=0 axis is s(ms+c) -> ONE marginal pole (position, z=1) + ONE damped
# velocity pole z = exp(-ts/tau) (the tau*dv settling of F1). Theta: sprung complex pair.
marg = lam[np.abs(lam - 1.0) < 1e-4]              # position integrators (expect exactly 2: X, Y)
vel  = lam[(np.abs(lam - 1.0) >= 1e-4) & (np.abs(lam.imag) < 1e-6)]   # real damped velocity poles
osc  = lam[np.abs(lam.imag) >= 1e-6]              # Theta complex pair
int_dev = np.abs(np.abs(marg) - 1.0)
print(f'\n  X/Y POSITION poles (marginal): {len(marg)} found, '
      f'max | |lambda|-1 | = {int_dev.max():.2e}  (ON the circle to numerical precision)')
print(f'  X/Y VELOCITY poles (damped, inside): |lambda| = {np.abs(vel)}  '
      f'(= exp(-ts/tau): tau_X={dc.tau_X:.2f}s, tau_Y={dc.tau_Y:.2f}s -> '
      f'{np.exp(-ts/dc.tau_X):.8f}, {np.exp(-ts/dc.tau_Y):.8f})')
print(f'  Theta (yaw) pair: |lambda| = {np.abs(osc)}  (strictly inside = sprung)')
assert len(marg) == 2, f'expected exactly 2 marginal position poles, found {len(marg)}'
assert int_dev.max() < 1e-6, 'position poles not on the unit circle?!'

# ── Figure: full view + zoom at z=1 ───────────────────────────────────────────
th = np.linspace(0, 2 * np.pi, 720)
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5.5))
for ax, (xl, yl) in zip((axL, axR), [((-1.15, 1.15), (-1.15, 1.15)),
                                     ((0.9975, 1.0007), (-0.012, 0.012))]):
    ax.plot(np.cos(th), np.sin(th), '0.6', lw=0.8, label='unit circle' if ax is axL else None)
    ax.plot(marg.real, marg.imag, 'C3x', ms=12, mew=2.5,
            label='X/Y POSITION poles: AT z=1 (marginal free integrators) -- the drift path')
    ax.plot(vel.real, vel.imag, 'C2s', ms=7,
            label='X/Y VELOCITY poles: damped, inside (the bounded tau*dv settling)')
    ax.plot(osc.real, osc.imag, 'C0o', ms=7,
            label='Theta (yaw) pair: sprung, strictly INSIDE')
    ax.set_xlim(*xl); ax.set_ylim(*yl)
    ax.set_xlabel('Re'); ax.grid(True); ax.set_aspect('auto')
axL.set_ylabel('Im'); axL.set_title('full view')
axR.set_title(f'zoom at z = 1   (X/Y: | |z|-1 | < {int_dev.max():.0e};  '
              f'Theta strictly inside)')
axL.legend(fontsize=7, loc='lower left')
fig.suptitle('Discrete baseline poles at PIPELINE discretization: on the circle (marginal), '
             'inside (stable), or outside (unstable)?')
dm.add_provenance(fig, f'Jacobian of the pipeline Gantry_State_Block step (RK4, ts={ts:.1e}, '
                       f'up_sample={up}) at Y_op={Y_OP}, u=0 | autograd, float64')
fig.tight_layout()
p = os.path.join(dm.OUT_DIR, 'f4_poles.png')
fig.savefig(p, dpi=150)
np.savez(os.path.join(dm.OUT_DIR, 'f4_poles.npz'), J=J, lam=lam, ts=ts, up=up, Y_op=Y_OP)
print(f'\nSaved: {p}')
