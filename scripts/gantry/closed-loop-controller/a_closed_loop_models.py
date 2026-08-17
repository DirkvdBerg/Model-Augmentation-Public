"""A: closed-loop evaluation of the models (PLAN-controller-in-the-loop.md).

Runs the same models through the same loop, driven by r_sim, and compares against the open-loop
replay of the same record. This is simultaneously the A experiment and a direct test of the
prediction in controller-in-derivation.tex section 6.3:

    e_ol = Delta Si w          e_cl = So Delta So_hat w

so the closed-loop discrepancy should be the open-loop one shaped by So, i.e. suppressed below
about 45 Hz and amplified by roughly 1.8 across [130, 180] Hz, which is where the absorber sits.
If the measured ratio follows sigma_max(So), the theory is confirmed on data.

Models available without leaving this folder:
  truth     PL.deriv8, 8-state, the plant that generated the record       -> the floor
  baseline  PL.deriv6, 6-state, no absorber                               -> the model under test
A trained ANN checkpoint lives in the training pipeline, outside this folder. The hook is
MODEL_HOOK below: give it a callable with the deriv signature and it is included automatically.

Criteria (S-A1 to S-A4 in the plan). S-A2 compares the closed-loop NRMS of baseline against
truth; with no ANN checkpoint present, S-A4's ordering reduces to truth < baseline.
"""
__project_origin__ = "added"

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch

import closed_loop as CL
import plant as PL

FORCE_PEAK = np.array([2000., 2000., 1420.])     # cfg.lim.force_peak
FORCE_RMS = np.array([916., 916., 656.])         # cfg.lim.force_rms
T_SKIP = 0.5
CH = ['X1', 'X2', 'Y']
MODEL_HOOK = {}          # name -> (deriv, n_state); fill to add a trained checkpoint


def open_loop_replay(deriv, x0, u_stage, ts=CL.TS):
    """Same record, same forces, no controller. The existing arm, for comparison."""
    N = len(u_stage)
    out = np.empty((N, 3))
    x = np.asarray(x0, float).copy()
    Pt = PL.P_np.T
    for k in range(N):
        out[k] = Pt @ x[:3]
        ul = PL.P_np @ u_stage[k]
        k1 = deriv(x, ul)
        k2 = deriv(x + .5 * ts * k1, ul)
        k3 = deriv(x + .5 * ts * k2, ul)
        k4 = deriv(x + ts * k3, ul)
        x = x + (ts / 6.) * (k1 + 2 * k2 + 2 * k3 + k4)
    return out


def nrms(err, ref):
    return np.sqrt(np.mean(err ** 2, axis=0)) / np.std(ref, axis=0)


def run(record_name):
    rec = CL.load_record(record_name)
    Ac, Bc, Cc, Dc, Y_op = CL.load_controller(record_name)
    ctrl = (Ac, Bc, Cc, Dc)
    t = np.arange(len(rec['r'])) * CL.TS
    m = t >= T_SKIP
    print('=' * 78)
    print('%s   Y_op = %.2f m' % (record_name, Y_op))

    models = {'truth': PL.deriv8, 'baseline': PL.deriv6}
    models.update({k: v for k, v in MODEL_HOOK.items()})

    out = {}
    for nm, deriv in models.items():
        x0 = CL.x0_for('truth' if nm == 'truth' else 'baseline', Y_op)
        y_cl, u_cl, _ = CL.simulate(deriv, x0, rec['r'], rec['f_ms'], ctrl,
                                    force_clip=FORCE_PEAK * 10)
        diverged = len(y_cl) < len(rec['r'])
        y_ol = open_loop_replay(deriv, x0, rec['u_total'])
        out[nm] = dict(y_cl=y_cl, u_cl=u_cl, y_ol=y_ol, diverged=diverged)

        if diverged:
            print('  %-9s DIVERGED at t = %.3f s   -> S-A1 FAIL' % (nm, len(y_cl) * CL.TS))
            continue
        e_cl = y_cl - rec['y']
        e_ol = y_ol - rec['y']
        peak = np.abs(u_cl[m]).max(axis=0)
        urms = u_cl[m].std(axis=0)
        out[nm].update(e_cl=e_cl, e_ol=e_ol)
        print('  %-9s NRMS closed [%.3e %.3e %.3e]  open [%.3e %.3e %.3e]'
              % (nm, *nrms(e_cl[m], rec['y'][m]), *nrms(e_ol[m], rec['y'][m])))
        print('            |u| peak [%7.1f %7.1f %7.1f] N  vs lim [%.0f %.0f %.0f]  %s'
              % (*peak, *FORCE_PEAK, 'ok' if np.all(peak <= FORCE_PEAK) else 'OVER'))
        print('            u rms    [%7.1f %7.1f %7.1f] N  vs lim [%.0f %.0f %.0f]  %s'
              % (*urms, *FORCE_RMS, 'ok' if np.all(urms <= FORCE_RMS) else 'OVER'))
        print('            ramp fraction of closed-loop error [%5.2f%% %5.2f%% %5.2f%%]'
              % tuple(100 * CL.ramp_fraction(e_cl[m])))

    # ---- S-A1, S-A2, S-A4 --------------------------------------------------
    ok = {}
    ok['S-A1'] = not any(v['diverged'] for v in out.values())
    if ok['S-A1']:
        n_base = nrms(out['baseline']['e_cl'][m], rec['y'][m])
        n_truth = nrms(out['truth']['e_cl'][m], rec['y'][m])
        ok['S-A4'] = bool(np.all(n_truth < n_base))
        print('  S-A1 stability          %s' % ('PASS' if ok['S-A1'] else 'FAIL'))
        print('  S-A4 ordering truth<base %s' % ('PASS' if ok['S-A4'] else 'FAIL'))
    return rec, t, out, Y_op


def spectral_ratio(e_cl, e_ol, fs):
    """|e_cl| / |e_ol| against frequency. Should track sigma_max(So) if section 6.3 holds."""
    f, Pcl = welch(e_cl, fs=fs, nperseg=8192, axis=0, detrend=False)
    _, Pol = welch(e_ol, fs=fs, nperseg=8192, axis=0, detrend=False)
    return f, np.sqrt(Pcl / np.maximum(Pol, 1e-300))


if __name__ == '__main__':
    names = CL.available_records()
    results = [run(n) for n in names]

    fig, axes = plt.subplots(3, len(results) * 2, figsize=(7.0 * len(results), 7.8),
                             squeeze=False)
    for c, (rec, t, out, Y_op) in enumerate(results):
        m = t >= T_SKIP
        for j in range(3):
            ax = axes[j, 2 * c]
            if 'e_ol' in out['baseline']:
                ax.plot(t, out['baseline']['e_ol'][:, j] * 1e6, color='#999999', lw=0.6,
                        label='open loop')
                ax.plot(t, out['baseline']['e_cl'][:, j] * 1e6, color='#D55E00', lw=0.6,
                        label='closed loop')
            ax.grid(alpha=0.25, lw=0.5)
            ax.set_ylabel('%s err [$\\mu$m]' % CH[j])
            if j == 0:
                ax.set_title('baseline error, open vs closed loop', fontsize=10)
                ax.legend(fontsize=8, frameon=False, loc='upper right')
            if j == 2:
                ax.set_xlabel('time [s]')

            ax = axes[j, 2 * c + 1]
            if 'e_ol' in out['baseline']:
                f, ratio = spectral_ratio(out['baseline']['e_cl'][m],
                                          out['baseline']['e_ol'][m], 1 / CL.TS)
                ax.loglog(f, ratio[:, j], color='#0072B2', lw=0.9)
            ax.axhline(1.0, color='#333333', lw=0.8, ls=':')
            ax.axvspan(130, 180, color='#D55E00', alpha=0.15)
            ax.set_ylim(1e-3, 1e1)
            ax.grid(alpha=0.25, lw=0.5, which='both')
            if j == 0:
                ax.set_title('|e_cl| / |e_ol|, shaded = absorber band', fontsize=10)
            ax.set_ylabel('ratio')
            if j == 2:
                ax.set_xlabel('frequency [Hz]')
    fig.suptitle('A: models in the loop. Right column tests section 6.3: the ratio should follow '
                 'sigma_max(So),\nabout 0.02 at 10 Hz and about 1.8 across the absorber band.',
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    outdir = os.path.join(CL.HERE, 'figures')
    os.makedirs(outdir, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(outdir, 'closed_loop_models.%s' % ext), dpi=160,
                    bbox_inches='tight')
    print('wrote %s' % os.path.join(outdir, 'closed_loop_models.png'))
