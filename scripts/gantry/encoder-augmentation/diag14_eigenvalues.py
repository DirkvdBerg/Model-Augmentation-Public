"""
diag14_eigenvalues.py
---------------------
Prove that the gantry at 4kHz has poles exactly on the unit circle,
while Jan's MSD at 50Hz has all poles strictly inside.

This is the root cause of why Jan can route ANN -> all physical states
(ecc_2025/msd_ndof_interconnect_dynamic.py:91) and we cannot:

  Jan (ECC): interconnect.connect_block_signals(ANN_state_block, ["x","u"], ["xp"])
             nw=nxd  -> ANN output dimension = full state dim -> all states

  Jan (journal): state_augment_specific_states = False (line 54) -> nx_aug_model_out
                 overrides to nx_aug_model -> all states

  Gantry: same routing causes 800x blowup in 1 step (diag13).

Structural argument:
  Gantry K matrix (gantry_ss.py:97-98):
    K = zeros(3,3), K[1,1] = kb1 + kb2

  K is zero for q1 and q3 -> no spring restoring force -> CT poles at s=0
  (pure rigid-body integrators). DT poles: z = exp(0 * Ts) = 1.000000 exactly.

  Any additive ANN term in the state update is equivalent to injecting a
  persistent bias into an integrator: position error grows as nf * bias.
  Over nf=400 steps this amplifies by 400x before the optimizer can compensate.

  Jan's MSD (msd_ndof_data_generation.py:38-41):
    k = [100, 100, 100], m = [0.5, 0.4, 0.1], c = [0.5, 0.5, 0.5], dt = 0.02s
  All k > 0 -> no integrators -> all DT poles strictly inside unit circle.
"""

import os
import sys
import numpy as np
import scipy.signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from model_augmentation.systems.gantry_linearization import gantry_linearize_and_discretize

OUT_DIR = os.path.join(PROJECT_ROOT, 'simulations', 'gantry_subnet',
                       'encoder-augmentation', 'diagnostics')
os.makedirs(OUT_DIR, exist_ok=True)

## ═══════════════════════════════════════════════════════════════════════════
## 1. Gantry discrete eigenvalues at 4kHz
## ═══════════════════════════════════════════════════════════════════════════

print('=' * 60)
print('Gantry discrete eigenvalues at 4 kHz (dt = 1/4000 s)')
print('=' * 60)

Ad_gantry, _, _, _ = gantry_linearize_and_discretize(dt=1.0 / 4000)
eigs_g = np.linalg.eigvals(Ad_gantry)
abs_g  = np.abs(eigs_g)

print(f'  {"|z|":>10}  {"1 - |z|":>12}  {"angle [deg]":>12}')
for z in sorted(eigs_g, key=lambda z: abs(z), reverse=True):
    print(f'  {abs(z):10.8f}  {1 - abs(z):12.3e}  {np.degrees(np.angle(z)):12.4f}')

n_on_circle = np.sum(np.abs(1.0 - abs_g) < 1e-6)
print(f'\n  Poles with |z| = 1 (within 1e-6): {n_on_circle} / {len(abs_g)}')
print(f'  min(|z|): {abs_g.min():.8f}')
print(f'  max(1 - |z|): {(1 - abs_g).max():.3e}')
print(f'  min(1 - |z|): {(1 - abs_g).min():.3e}')

## ═══════════════════════════════════════════════════════════════════════════
## 2. Jan's 2-DOF MSD discrete eigenvalues at 50Hz
##    Parameters from scripts/journal_model_augmentation/msd_ndof_data_generation.py
##    m=[0.5, 0.4], k=[100,100], c=[0.5,0.5], a=[0,50] (linear approx: a=0)
##    FP_dof=2 -> 4 physical states
## ═══════════════════════════════════════════════════════════════════════════

print()
print('=' * 60)
print("Jan's 2-DOF MSD discrete eigenvalues at 50 Hz (dt = 0.02 s)")
print('=' * 60)
print("  Parameters (msd_ndof_data_generation.py:38-41, linearised a=0):")
print("    m=[0.5, 0.4]  k=[100, 100]  c=[0.5, 0.5]  dt=0.02")

m1, m2 = 0.5, 0.4
k1, k2 = 100.0, 100.0
c1, c2 = 0.5, 0.5

# CT A matrix for 2-DOF linear MSD (a=0):
#   x = [z1, z2, z3, z4]  (positions and velocities interleaved per mass_spring_damper.py)
#   dz1 = z2
#   dz2 = -(k1+k2)/m1 * z1 + k2/m1 * z3 - (c1+c2)/m1 * z2 + c2/m1 * z4
#   dz3 = z4
#   dz4 = k2/m2 * (z1-z3) + c2/m2 * (z2-z4)
Ac_msd = np.array([
    [0,            1,          0,       0],
    [-(k1+k2)/m1, -(c1+c2)/m1, k2/m1,  c2/m1],
    [0,            0,          0,       1],
    [k2/m2,       c2/m2,      -k2/m2, -c2/m2],
])

dt_msd = 0.02
Ad_msd = scipy.linalg.expm(Ac_msd * dt_msd)
eigs_m = np.linalg.eigvals(Ad_msd)
abs_m  = np.abs(eigs_m)

print(f'  {"|z|":>10}  {"1 - |z|":>12}  {"angle [deg]":>12}')
for z in sorted(eigs_m, key=lambda z: abs(z), reverse=True):
    print(f'  {abs(z):10.8f}  {1 - abs(z):12.3e}  {np.degrees(np.angle(z)):12.4f}')

print(f'\n  min(|z|): {abs_m.min():.8f}')
print(f'  max(1 - |z|): {(1 - abs_m).max():.3e}')
print(f'  min(1 - |z|): {(1 - abs_m).min():.3e}')

## ═══════════════════════════════════════════════════════════════════════════
## 3. Compare
## ═══════════════════════════════════════════════════════════════════════════

print()
print('=' * 60)
print('Comparison')
print('=' * 60)
print(f'  Gantry 4kHz  : min(1-|z|) = {(1 - abs_g).min():.3e}   '
      f'max(1-|z|) = {(1 - abs_g).max():.3e}')
print(f"  Jan's MSD 50Hz: min(1-|z|) = {(1 - abs_m).min():.3e}   "
      f"max(1-|z|) = {(1 - abs_m).max():.3e}")
print()
print('  Gantry: K[q1]=0, K[q3]=0 (gantry_ss.py:97-98) -> rigid-body integrators')
print('          CT poles at s=0 -> DT z = exp(0 * 1/4000) = 1.000000 exactly')
print('  MSD:    k=[100,100] (all >0) -> all modes have restoring force')
print('          no integrators -> all DT poles strictly inside unit circle')
print()
print('  Jan routes ANN -> ALL states (ecc_2025/msd_ndof_interconnect_dynamic.py:91):')
print('    interconnect.connect_block_signals(ANN_state_block, ["x","u"], ["xp"])')
print('  This is safe for MSD because |z| < 1 everywhere.')
print('  For the gantry, ANN perturbation injects into integrators -> nf*bias drift.')
print(f'  At nf=400: 400 * (lr * grad) accumulation makes any update catastrophic.')

## ═══════════════════════════════════════════════════════════════════════════
## 4. Quantify the 1-step amplification
## ═══════════════════════════════════════════════════════════════════════════

print()
print('=' * 60)
print('1-step amplification of ANN state perturbation over nf=400 rollout')
print('=' * 60)

nf = 400
# For a pole z, a unit perturbation at step k accumulates as sum_{j=0}^{nf-k-1} z^j
# Worst case: perturbation at step 0 -> sum = (1-z^nf)/(1-z) for |z|<1, or nf for z=1

for label, eigs, dt in [('Gantry (4kHz)', eigs_g, 1/4000),
                         ("Jan's MSD (50Hz)", eigs_m, 0.02)]:
    print(f'\n  {label}:')
    for z in sorted(eigs, key=lambda z: abs(z), reverse=True)[:4]:
        if abs(z - 1.0) < 1e-6:
            amp = nf  # pure integrator: linear growth
        elif abs(z) >= 1.0:
            amp = nf  # unstable
        else:
            amp = abs((1 - z**nf) / (1 - z))
        print(f'    |z|={abs(z):.6f}  amp(nf={nf}) = {amp:.1f}x')

## ═══════════════════════════════════════════════════════════════════════════
## 5. Plot
## ═══════════════════════════════════════════════════════════════════════════

theta = np.linspace(0, 2 * np.pi, 500)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for ax, eigs, title in [
    (axes[0], eigs_g, 'Gantry at 4 kHz'),
    (axes[1], eigs_m, "Jan's 2-DOF MSD at 50 Hz"),
]:
    ax.plot(np.cos(theta), np.sin(theta), 'k-', lw=0.8, label='Unit circle')
    ax.scatter(eigs.real, eigs.imag, color='C0', s=80, zorder=5, label='DT poles')
    ax.axhline(0, color='k', lw=0.4)
    ax.axvline(0, color='k', lw=0.4)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('Re(z)')
    ax.set_ylabel('Im(z)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle('Discrete-time poles: gantry vs Jan\'s MSD')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'diag14_eigenvalues.png'), dpi=150)
print(f'\nSaved plot: {os.path.join(OUT_DIR, "diag14_eigenvalues.png")}')
print('Done.')
