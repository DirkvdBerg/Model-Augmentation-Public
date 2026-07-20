"""
d13_scheduling_detune_tolerance.py -- Layer-3 NECESSITY diagnostic: how much
Y-scheduling error can M(Y) absorb before the dynamics are materially detuned?

Layer 3 (de-drifted/exogenous Y-scheduling) is a CONDITIONAL fallback, not a
proven necessity: the R5 drift->detune conflict is second-order if the residual
Y-drift after Layer 2 is small. This probe measures the tolerance side:

  epsilon_tol = the Y-scheduling error at which the scheduling-induced output
                error reaches the absorber scale (the smallest dynamics the
                augmentation must learn).

Decision rule (with the Layer-2 build later): residual 12 s Y-drift << eps_tol
-> Layer 3 unnecessary (self-scheduling keeps R5 intact); residual ~ eps_tol
-> Layer 3 needed, variant discussion goes to the supervisor with numbers.

Two parts, no training, truth-simulator only (drift_common 8-state EOM):
  P1 closed form: linearized 4-DOF eigenfrequencies at Y0 vs Y0+dY over the
     data's Y range -> frequency detune per unit scheduling error.
  P2 forward sim: full-record truth rollout where the MASS MATRIX ONLY is
     evaluated at (Y + dY) while the true state propagates -- the exact
     mechanism of R5 (M reads a wrong Y) isolated from everything else.
     Output deviation vs dY, crossing point with the absorber RMS = eps_tol.

Reference scale: absorber displacement RMS (measured from the stored GT).
# HEURISTIC: absorber RMS as the materiality threshold -- the absorber effect is
# the signal the augmentation must capture; scheduling-induced error at that
# level corrupts the learning target. Data-derived scale, not oracle-based.

Run:
  conda run -n GraduationProject python scripts/gantry/diagnostics-drift/d13_scheduling_detune_tolerance.py
Env: REC (default V1 record), DY_LIST (m, default "1e-5,1e-4,5e-4,1e-3,5e-3,2.6e-2"
     -- last value = the measured drifted-checkpoint 12 s Y drift, d6),
     N_SEC (sim length seconds, default full record).
Outputs -> simulations/gantry_subnet/diagnostics/ (npz + png)
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import drift_common as dc

REC     = os.environ.get('REC', 'V1_standstill_Yp10.mat')
DY_LIST = [float(x) for x in os.environ.get(
    'DY_LIST', '1e-5,1e-4,5e-4,1e-3,5e-3,2.6e-2').split(',')]
N_SEC   = os.environ.get('N_SEC', None)

rec = dc.load_record(REC)
ts  = rec.ts
N   = len(rec.u_stage) if N_SEC is None else min(int(float(N_SEC) / ts), len(rec.u_stage))
u_log = dc.stage_force_to_logical(rec.u_stage[:N])
x0 = dc.true_x0_8(rec)
absorber_rms = rec.delta_a.std()
Y_true = rec.x_logical[:N, 2]
print(f'{REC}: {N} samples @ {1/ts:.0f} Hz ({N*ts:.1f}s)  '
      f'Y range [{Y_true.min():+.4f}, {Y_true.max():+.4f}] m  '
      f'absorber RMS={absorber_rms:.3e} m')

# ── P1: closed-form eigenfrequency detune of the linearized 4-DOF ────────────
# THEORY: undamped natural frequencies from det(K - w^2 M(Y)) = 0; the
# Y-dependence of M is the scheduling dependence whose corruption we quantify.
def natfreqs(Y, da=0.0):
    M = dc._M(Y, da)
    w2 = np.linalg.eigvals(np.linalg.solve(M, dc._K4))
    w2 = np.sort(np.real(w2))
    return np.sqrt(np.clip(w2, 0.0, None)) / (2 * np.pi)   # Hz; two zeros (K=0 axes)

Y0 = float(np.median(Y_true))
f0 = natfreqs(Y0)
print(f'\nP1: natural frequencies at Y0={Y0:+.4f} m: {np.array2string(f0, precision=2)} Hz')
print(f"  {'dY [m]':>10s} {'|df/f| theta-mode':>18s} {'|df/f| absorber':>16s}")
p1 = {}
for dY in DY_LIST:
    f1 = natfreqs(Y0 + dY)
    # modes 2,3 are the nonzero ones (theta-ish, absorber-ish); 0,1 are K=0 zeros
    rel = np.abs(f1[2:] - f0[2:]) / f0[2:]
    p1[dY] = rel
    print(f'  {dY:>10.1e} {rel[0]:>18.3e} {rel[1]:>16.3e}')

# ── P2: forward sim with M scheduled at (Y + dY), state propagates truly ─────
def _deriv_detuned(x, u_log_k, dY):
    """drift_common._deriv but with the MASS MATRIX evaluated at Y + dY.
    Isolates the R5 mechanism: dynamics assembled at a wrong scheduling Y."""
    Y, da = x[2], x[3]
    M = dc._M(Y + dY, da)                      # <-- the scheduling corruption
    q, qd = x[:4], x[4:]
    qdd = np.linalg.solve(M, dc._E43 @ u_log_k - dc._K4 @ q - dc._C4 @ qd)
    dx = np.empty(8)
    dx[:4] = qd
    dx[4:] = qdd
    return dx


def simulate_detuned(dY):
    x = x0.astype(dc.DTYPE).copy()
    h = ts / dc.UP_SAMPLE
    q_out = np.empty((N, 4))
    for k in range(N):
        q_out[k] = x[:4]
        uk = u_log[k]
        for _ in range(dc.UP_SAMPLE):
            k1 = _deriv_detuned(x, uk, dY)
            k2 = _deriv_detuned(x + 0.5 * h * k1, uk, dY)
            k3 = _deriv_detuned(x + 0.5 * h * k2, uk, dY)
            k4 = _deriv_detuned(x + h * k3, uk, dY)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return q_out


print(f'\nP2: reference rollout (dY=0) ...')
q_ref = simulate_detuned(0.0)
rows = []
print(f"  {'dY [m]':>10s} {'RMS dX':>10s} {'RMS dTh':>10s} {'RMS dY-ch':>10s} "
      f"{'max/absRMS':>11s}")
for dY in DY_LIST:
    q_d = simulate_detuned(dY)
    dev = q_d[:, :3] - q_ref[:, :3]
    rms = np.sqrt((dev ** 2).mean(axis=0))
    rows.append((dY, rms))
    print(f'  {dY:>10.1e} {rms[0]:>10.3e} {rms[1]:>10.3e} {rms[2]:>10.3e} '
          f'{rms.max()/absorber_rms:>11.2f}')

# eps_tol: log-log interpolate the worst-channel curve to the absorber RMS level
dys  = np.array([r[0] for r in rows])
worst = np.array([r[1].max() for r in rows])
if worst.min() < absorber_rms < worst.max():
    eps_tol = float(np.exp(np.interp(np.log(absorber_rms),
                                     np.log(worst), np.log(dys))))
    print(f'\n  eps_tol (worst-channel deviation = absorber RMS): ~{eps_tol:.2e} m')
else:
    eps_tol = float('nan')
    print(f'\n  absorber RMS not bracketed by the swept dY range; widen DY_LIST.')
print(f'  measured drifted-checkpoint 12 s Y drift (d6): 2.6e-2 m '
      f'-> ratio to eps_tol: {2.6e-2/eps_tol:.1f}x' if np.isfinite(eps_tol) else '')

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].loglog(dys, [p1[d][0] for d in dys], 'o-', label='theta-mode |df/f|')
axes[0].loglog(dys, [p1[d][1] for d in dys], 's-', label='absorber-mode |df/f|')
axes[0].set_xlabel('Y scheduling error dY [m]'); axes[0].set_ylabel('relative frequency detune')
axes[0].set_title('P1: closed-form eigenfrequency detune'); axes[0].grid(True, which='both')
axes[0].legend(fontsize=8)
for ch, lbl in enumerate(['X', 'Theta', 'Y']):
    axes[1].loglog(dys, [r[1][ch] for r in rows], 'o-', label=f'RMS dev {lbl}')
axes[1].axhline(absorber_rms, color='0.4', ls=':', label='absorber RMS (materiality)')
if np.isfinite(eps_tol):
    axes[1].axvline(eps_tol, color='C3', ls='--', label=f'eps_tol ~ {eps_tol:.1e} m')
axes[1].axvline(2.6e-2, color='k', ls='-.', lw=0.8, label='measured 12 s drift (d6)')
axes[1].set_xlabel('Y scheduling error dY [m]'); axes[1].set_ylabel('output deviation RMS [m]')
axes[1].set_title('P2: forward-sim detune effect vs scheduling error')
axes[1].grid(True, which='both'); axes[1].legend(fontsize=8)
fig.suptitle(f'd13: how much Y-scheduling error does M(Y) tolerate? ({REC}, {N*ts:.1f}s)')
fig.tight_layout()
stem = os.path.join(dc.OUT_DIR, f'd13_scheduling_detune_tolerance_{REC.split(".")[0]}')
fig.savefig(stem + '.png', dpi=150)
np.savez(stem + '.npz', dy_list=dys, rms=np.array([r[1] for r in rows]),
         p1=np.array([p1[d] for d in dys]), f0=f0, Y0=Y0, eps_tol=eps_tol,
         absorber_rms=absorber_rms, ts=ts, N=N, rec=REC)
print(f'\nSaved: {stem}.png')
print(f'Saved: {stem}.npz')
