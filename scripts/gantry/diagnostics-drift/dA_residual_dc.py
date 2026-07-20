"""
dA_residual_dc.py -- D-A: is the TRUE residual the ANN must learn on X/Y zero-DC?
(compatibility check for the bounded-integral construction, §5b of drift-diagnosis-status.md)

The chosen solution routes the free-axis coupling through a ZERO-DC (bounded-integral) channel.
That only works if the TRUE residual the ANN must supply on X/Y is itself ~zero-DC. This diagnostic
measures it directly on the sim data where we KNOW the truth -- no training, no fitting.

Residual = the acceleration the ANN must add to the baseline to reproduce truth, evaluated ALONG the
true trajectory:
    a_res(k) = q̈_truth(x_true(k), u(k))  -  q̈_baseline(x_true(k), u(k))     [logical coords]
  q̈_truth  : from the 8-state truth EOM (drift_common._deriv), rows [X,Th,Y].
  q̈_base   : from the 3-DOF baseline EOM = the truth EOM with the absorber removed
              (ma=0, mh_rigid->mh, drop the da DOF). Built here and VERIFIED against
              drift_common.simulate_baseline (Gantry_State_Block) by an open-loop free-run match.

Readout on the K=0 axes (X, Y):
  |mean|/rms small AND running integral bounded  -> residual is ~zero-DC -> the bounded-integral
     channel CAN represent it -> construction viable.
  |mean|/rms not small OR running integral ramps -> a rectified DC (nonlinear M(Y); cf. the S3
     open-loop side-finding) -> that DC must live in f_base, not the learned residual.

Run:
  conda run -n GraduationProject python scripts/gantry/diagnostics-drift/dA_residual_dc.py
Env: REC (V1 default), FS_NEW (4000), N_WIN (full).
Outputs -> simulations/gantry_subnet/diagnostics/
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import drift_common as dc

REC    = os.environ.get('REC', 'V1_standstill_Yp10.mat')
FS_NEW = int(os.environ.get('FS_NEW', str(dc.DEFAULT_FS_NEW)))
N_WIN  = os.environ.get('N_WIN', None)

# ── Baseline 3-DOF EOM = truth EOM with the absorber removed (ma=0, mh_rigid->mh) ──
m1, m2, mb, mh = dc.m1f, dc.m2f, dc.mbf, dc.mhf
Jb, Jh, Lb, d  = dc.Jbf, dc.Jhf, dc.Lbf, dc.df
_C3 = np.array([
    [dc.cg1f + dc.cg2f,          (dc.cg1f - dc.cg2f) * Lb / 2,                      0.0],
    [(dc.cg1f - dc.cg2f) * Lb/2, dc.cb1f + dc.cb2f + (dc.cg1f + dc.cg2f) * Lb**2/4, 0.0],
    [0.0, 0.0, dc.cyf],
], dtype=np.float64)
_K3 = np.array([[0.0, 0.0, 0.0],
                [0.0, dc.kb1f + dc.kb2f, 0.0],
                [0.0, 0.0, 0.0]], dtype=np.float64)


def _M3(Y):
    off = (m1 - m2) * Lb / 2 - mh * Y
    return np.array([
        [m1 + m2 + mb + mh, off, 0.0],
        [off, Jb + Jh + (m1 + m2) * Lb**2 / 4 + mh * d**2 + mh * Y**2, -mh * d],
        [0.0, -mh * d, mh],
    ], dtype=np.float64)


def base_qdd(x6, u_log3):
    """Baseline continuous accel [X,Th,Y] at logical state x6, logical force u_log3."""
    Y = x6[2]; q = x6[:3]; qd = x6[3:]
    return np.linalg.solve(_M3(Y), u_log3 - _K3 @ q - _C3 @ qd)


def base_deriv6(x6, u_log3):
    dx = np.empty(6); dx[:3] = x6[3:]; dx[3:] = base_qdd(x6, u_log3); return dx


# ── Load record + true 8-state trajectory ────────────────────────────────────
rec = dc.load_record(REC, FS_NEW)
ts = rec.ts
N  = rec.u_stage.shape[0] if N_WIN is None else min(int(N_WIN), rec.u_stage.shape[0])
u_stage = rec.u_stage[:N]
u_log   = dc.stage_force_to_logical(u_stage)              # (N,3) logical force
x8 = np.column_stack([rec.x_logical[:N, :3], rec.delta_a[:N],
                      rec.x_logical[:N, 3:], rec.vdelta_a[:N]])   # [X,Th,Y,da,dX,dTh,dY,vda]
x6 = rec.x_logical[:N]                                    # [X,Th,Y,dX,dTh,dY]
t  = np.arange(N) * ts
print(f'Record {rec.name}: {N} samples @ {1/ts:.0f} Hz')

# ── VERIFY the baseline EOM matches simulate_baseline (Gantry_State_Block) ────
# Open-loop free-run from x0 with both; they must agree to integration tolerance.
n_chk = min(4000, N)
x0_6 = x6[0].copy()
# our base_deriv6 via RK4 (up_sample matched)
xchk = x0_6.copy(); y_ours = np.empty((n_chk, 3))
h = ts / dc.UP_SAMPLE
for k in range(n_chk):
    y_ours[k] = xchk[:3]
    uk = u_log[k]
    for _ in range(dc.UP_SAMPLE):
        k1 = base_deriv6(xchk, uk); k2 = base_deriv6(xchk + 0.5*h*k1, uk)
        k3 = base_deriv6(xchk + 0.5*h*k2, uk); k4 = base_deriv6(xchk + h*k3, uk)
        xchk = xchk + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
_, xb = dc.simulate_baseline(x0_6, u_stage[:n_chk], ts, return_state=True)
chk = np.abs(y_ours - xb[:, :3]).max()
print(f'  baseline-EOM verification vs Gantry_State_Block: max |dpos| = {chk:.2e} m '
      f'({"OK" if chk < 1e-6 else "MISMATCH -- residual not trustworthy"})')

# ── Residual acceleration the ANN must supply, along the true trajectory ─────
a_res = np.empty((N, 3))
for k in range(N):
    qdd_t = dc._deriv(x8[k], u_log[k])[4:7]      # truth accel [X,Th,Y]
    qdd_b = base_qdd(x6[k], u_log[k])            # baseline accel [X,Th,Y]
    a_res[k] = qdd_t - qdd_b
v_log = x6[:, 3:]                                # logical velocity [dX,dTh,dY]

# ── Analysis ──────────────────────────────────────────────────────────────────
names = ['X (K=0)', 'Theta (spring)', 'Y (K=0)']
mean_a = a_res.mean(axis=0)
rms_a  = np.sqrt((a_res**2).mean(axis=0))
ratio  = np.abs(mean_a) / (rms_a + 1e-30)
integ  = np.cumsum(a_res, axis=0) * ts           # running integral (velocity contribution)
# implied position drift from a residual DC the bounded-integral channel would forbid.
# DAMPED K=0 axis: a constant accel a_dc gives steady velocity v_ss = a_dc*tau (tau=m/c), so
# position drifts ~ a_dc*tau*T after the tau transient. (Undamped 0.5*a*T^2 would over-estimate.)
T = N * ts
tau_ax = np.array([dc.tau_X, np.nan, dc.tau_Y])   # Theta has a spring -> N/A
pos_from_dc = mean_a * tau_ax * T
# dissipativity proxy: sign of residual-accel . velocity (per axis, summed)
energy_proxy = (a_res * v_log).sum(axis=0) * ts

print(f'\n=== Residual acceleration the ANN must supply (along true {rec.name}, {T:.1f}s) ===')
print(f"  {'axis':16s} {'mean(DC)':>12s} {'rms':>12s} {'|mean|/rms':>11s} {'intdt end':>12s} "
      f"{'implied pos drift':>18s}")
for c, nm in enumerate(names):
    print(f'  {nm:16s} {mean_a[c]:>12.3e} {rms_a[c]:>12.3e} {ratio[c]:>11.4f} '
          f'{integ[-1,c]:>12.3e} {pos_from_dc[c]:>18.3e}')
print(f'  energy proxy sum(a_res*v)*dt [m^2/s^2-ish]:  X={energy_proxy[0]:+.2e}  '
      f'Th={energy_proxy[1]:+.2e}  Y={energy_proxy[2]:+.2e}  (<0 = dissipative-like)')

absorber_rms = rec.delta_a[:N].std()
print(f'\n  reference: absorber delta_a RMS = {absorber_rms:.3e} m')

# ── v-projection: separate DISSIPATIVE (velocity-dependent) DC from EXTERNAL (drifting) DC ──
# Affine fit a_res = alpha*v + beta per axis. alpha*v = velocity-dependent force (damping if
# alpha<0 -> self-limiting on the integrator, does NOT run away). beta = velocity-INDEPENDENT
# constant = the part that acts as a fixed external force -> THIS is what drifts.
# NOTE: linear-in-v; a nonlinear-in-v dissipation (Coulomb/quadratic) would be under-credited,
# making beta a conservative (over-)estimate of the true external DC.
print('\n=== v-projection: is the residual DC dissipative (self-limiting) or external (drifts)? ===')
beta_ext = np.zeros(3); alpha_damp = np.zeros(3)
for c, nm in enumerate(names):
    v = v_log[:, c]; a = a_res[:, c]
    A = np.column_stack([v, np.ones_like(v)])
    (alpha, beta), *_ = np.linalg.lstsq(A, a, rcond=None)
    R2 = 1.0 - ((a - (alpha * v + beta)).var() / (a.var() + 1e-30))
    alpha_damp[c], beta_ext[c] = alpha, beta
    tau_c = tau_ax[c] if c in (0, 2) else np.nan
    ext_drift = beta * tau_c * T
    frac_ext = abs(beta) / (abs(mean_a[c]) + 1e-30)
    print(f'  {nm:16s} alpha(damping)={alpha:+.3e} 1/s  beta(ext DC)={beta:+.3e}  R2={R2:.3f}')
    print(f'      total DC mean(a_res)={mean_a[c]:+.3e};  external fraction |beta|/|mean|={frac_ext:.2f};  '
          f'external-DC drift over {T:.0f}s = {ext_drift:+.2e} m')
print('  -> |beta| << |mean(a_res)| : DC is velocity-explained (dissipative, self-limiting) -> SAFE.')
print('     |beta| ~ |mean(a_res)|  : DC is external -> would drift -> belongs in f_base.')
print('\n=== Verdict (K=0 axes X, Y) ===')
print('  The DOMINANT residual (absorber coupling) is zero-DC iff |mean|/rms << 1 (bounded-integral')
print('  channel represents it perfectly). Separately, the tiny absolute DC still drifts on the free')
print('  integrator (drift ~ a_dc*tau*T); compare that to the absorber signal to see if it matters.')
for c in (0, 2):
    drift = pos_from_dc[c]; osc = 'zero-DC (viable)' if ratio[c] < 0.01 else 'has relative DC'
    matt = 'NEGLIGIBLE' if abs(drift) < absorber_rms else f'{abs(drift)/absorber_rms:.0f}x absorber -> real'
    print(f'  {names[c]:16s} osc part: {osc} (|mean|/rms={ratio[c]:.5f}); '
          f'DC-drift over {T:.0f}s = {drift:+.2e} m ({matt})')

# ── Plot: residual accel (+mean) and running integral, per axis ──────────────
fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
for c, nm in enumerate(names):
    axes[c, 0].plot(t, a_res[:, c], lw=0.5, color='C0')
    axes[c, 0].axhline(mean_a[c], color='C3', lw=1.3, label=f'mean={mean_a[c]:.2e}')
    axes[c, 0].axhline(0, color='k', lw=0.5)
    axes[c, 0].set_ylabel(f'{nm}\nresidual accel'); axes[c, 0].grid(True); axes[c, 0].legend(fontsize=7)
    axes[c, 1].plot(t, integ[:, c], lw=0.9, color='C1')
    axes[c, 1].plot(t, mean_a[c] * t, 'C3--', lw=1.0, label='pure-DC ramp')
    axes[c, 1].set_ylabel('int residual dt'); axes[c, 1].grid(True); axes[c, 1].legend(fontsize=7)
axes[0, 0].set_title('residual acceleration + its mean (DC)')
axes[0, 1].set_title('running integral: bounded (zero-DC) vs ramp (rectified DC)')
axes[-1, 0].set_xlabel('Time [s]'); axes[-1, 1].set_xlabel('Time [s]')
fig.suptitle(f'D-A: is the true X/Y residual zero-DC? (compatibility with the bounded-integral channel)  '
             f'[{rec.name}]')
fig.tight_layout()
stem = os.path.join(dc.OUT_DIR, f'dA_residual_dc_{rec.name.split(".")[0]}')
fig.savefig(stem + '.png', dpi=150)
np.savez(stem + '.npz', t=t, a_res=a_res, v_log=v_log, mean_a=mean_a, rms_a=rms_a,
         ratio=ratio, integ=integ, pos_from_dc=pos_from_dc, energy_proxy=energy_proxy,
         alpha_damp=alpha_damp, beta_ext=beta_ext, absorber_rms=absorber_rms, verify_max=chk, ts=ts)
print(f'\nSaved: {stem}.png')
print(f'Saved: {stem}.npz')
