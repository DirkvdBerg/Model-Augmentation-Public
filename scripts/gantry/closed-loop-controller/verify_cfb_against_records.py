"""Read-only verification of Cfb against every stored closed-loop record.

Two measurements, per record and per channel, nothing summarised to a pass/fail:

  (1) the additivity identity  u_total - (u_fb + f_sim).  gtd_run_simulation.m:33-34 computes
      u_total = u_fb + f_ms and gtd_save_record.m stores the three of them as `single`, so any
      residual above the float32 storage step means the stored signals are not what the
      generator's block diagram says.

  (2) the controller residual  u_fb_recomputed - u_fb, where u_fb_recomputed is Cfb rebuilt in
      Python at that record's Y_op (loss_variants.controller_ss -> p2_rate_compare.build_cfb_at
      -> verify_controller's closed-form Cnorm and ruleOfThumb gain) driven by r_sim - y from a
      zero initial state.  MATLAB produced u_fb with lsim(plant.Cfb, r_sim - q_with), also from
      zero state, and `y` is exactly q_with, so the two are comparable sample for sample.

Both are reported over the full record and over the record minus its first 0.5 s, which
separates a controller initial-state transient from a structural mismatch.

The controller is simulated through its state space (loss_variants.controller_ss), not through
lfilter on the (b, a) pairs, precisely because the state space is the object the interconnect
would host; this run therefore also prints the per-channel order and the total n_FB that the
placement decision in the handoff's section 8 needs.

Storage floor.  The records are float32.  The relative agreement that a correct Cfb can reach is
set by that quantisation, not by the formula: Cnorm has a pole at s = 0, so the tustin controller
has a pole at z = 1 and any constant bias in e = r_sim - y left by float32 rounding is integrated
into a ramp rather than showing up as broadband noise.  RESULT.md documents this.  The residual
is therefore also decomposed into a fitted ramp and the rest, and the fitted slope is compared
against kappa_j * w / 54 * mean(e_j), the integral gain times the measured bias.

That floor is also MEASURED rather than argued, because the whole verdict turns on it.  MATLAB
computed u_fb from a double-precision q_with; Python can only supply the single-precision y that
was stored, so the two controllers see inputs that differ by the rounding that `single()` applied.
Re-applying an independent draw of exactly that rounding, y + u*ulp(y) with u uniform on
[-1/2, 1/2], and re-running the same controller gives the spread of u_fb values that are all
equally consistent with what was stored.  Any residual inside that spread is not evidence about
the controller.  This uses no MATLAB and is the confound-free version of the record-level check;
test_controller_exact.py's L1 (coefficients, no simulation) and L4 (MATLAB's own lsim on the
identical input bits) are the same statement from the other direction, for two records.

Nothing is written; no model, no training, no MATLAB.
"""
__project_origin__ = "added"

import os
import sys

import numpy as np
from scipy.io import loadmat
from scipy.signal import dlsim

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from loss_variants import controller_ss                      # noqa: E402
from p2_rate_compare import build_cfb_at                     # noqa: E402
from verify_controller import W                              # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
TRAJ = os.path.join(REPO, 'data', 'gantry', 'matlab', 'trajectory', 'augmentation')

TS = 1.0 / 20e3          # design and storage rate of the records
T_SKIP = 0.5             # [s] discarded in the "settled" columns
CH = ['X1', 'X2', 'Y']   # stage channels, y = q_with is in stage coordinates

# Y_op per record, Matlab-scripts/Augmentation/data/gtd_build_records.m:36-67.
# The controller is frozen at this Y for the whole record (D-039); lsim is LTI, so there is no
# scheduling along Y even on T6-T14 where Y actually sweeps.
RECORDS = [
    ('T1_standstill_Ym30',  -0.30, 'train'),
    ('T2_standstill_Ym15',  -0.15, 'train'),
    ('T3_standstill_Y000',   0.00, 'train'),
    ('T4_standstill_Yp15',   0.15, 'train'),
    ('T5_standstill_Yp30',   0.30, 'train'),
    ('T6_ysweep_slow',       0.00, 'train'),
    ('T7_ysweep_fast',       0.00, 'train'),
    ('T8_ysweep_xmix',       0.00, 'train'),
    ('T9_aprbs_30',          0.00, 'train'),
    ('T10_aprbs_60',         0.00, 'train'),
    ('T11_aprbs_100',        0.00, 'train'),
    ('T12_aprbs_yaw',        0.00, 'train'),
    ('T13_lissajous',        0.00, 'train'),
    ('T14_lissajous_yaw',    0.00, 'train'),
    ('V1_standstill_Yp10',   0.10, 'val'),
    ('V2_aprbs_Ylow',       -0.22, 'val'),
    ('V3_ysweep_Yp10',       0.10, 'val'),
    ('V4_lissajous_Ym10',   -0.10, 'val'),
    ('E1_resonance_sweep',   0.00, 'test'),
    ('E2_multisine_Yp22',    0.22, 'test'),
    ('E3_aprbs_above',       0.00, 'test'),
    ('E4_multisine_off',     0.00, 'test'),
]


def _stats(res, ref, k0):
    """(max abs, rms, rms relative to rms(ref)) per channel, full record and from sample k0."""
    out = {}
    for tag, sl in (('full', slice(None)), ('settled', slice(k0, None))):
        r = res[sl]
        f = ref[sl]
        rms = np.sqrt(np.mean(r ** 2, axis=0))
        rms_ref = np.sqrt(np.mean(f ** 2, axis=0))
        with np.errstate(divide='ignore', invalid='ignore'):
            rel = np.where(rms_ref > 0, rms / rms_ref, np.nan)
        out[tag] = (np.abs(r).max(axis=0), rms, rms_ref, rel)
    return out


def _row(label, v, fmt='%.4e'):
    return '    %-26s [' % label + ' '.join(fmt % x for x in v) + ']'


def main():
    # ---- the controller object, once per distinct Y_op -----------------------
    print('Cfb rebuilt in Python: loss_variants.controller_ss(Y_op, ts) at ts = %.6e s (%g Hz)'
          % (TS, 1.0 / TS))
    print('  -> p2_rate_compare.build_cfb_at -> verify_controller closed-form Cnorm + '
          'ruleOfThumb gain')
    print('MATLAB side: gtd_run_simulation.m:33  u_fb = lsim(plant.Cfb, r_sim - q_with), '
          'zero initial state')
    print('Stored y is exactly q_with (gtd_save_record.m:31), both cast to single.\n')

    cache = {}
    for _, Y_op, _ in RECORDS:
        if Y_op in cache:
            continue
        cache[Y_op] = (controller_ss(Y_op, TS), build_cfb_at(Y_op, TS))

    print('=' * 100)
    print('CONTROLLER ORDER (identical structure at every Y_op; only the gains kappa_j differ)')
    print('=' * 100)
    for Y_op in sorted(cache):
        (A, B, C, D), (cfb, gains) = cache[Y_op]
        orders = [len(a) - 1 for _, a in cfb]
        print('  Y_op = %+5.2f m   per-channel order [%s]   n_FB = %d   '
              'D diag [%.4e %.4e %.4e]'
              % (Y_op, ' '.join(str(o) for o in orders), A.shape[0], *np.diag(D)))
        print('                   kappa = [%.6e %.6e %.6e]' % tuple(gains))
    (A0, _, _, _), _ = cache[0.00]
    print('\n  n_FB = %d states total (3 diagonal SISO channels, block-diagonal state space).'
          % A0.shape[0])
    print('  For the section-8 placement question: joining the interconnect state vector would'
          '\n  take nxd from 8 to %d, a %+.0f %% growth.\n' % (8 + A0.shape[0],
                                                              100.0 * A0.shape[0] / 8.0))

    k0 = int(round(T_SKIP / TS))
    add_tab, ctl_tab = [], []

    for name, Y_op, split in RECORDS:
        path = os.path.join(TRAJ, name + '.mat')
        print('=' * 100)
        if not os.path.exists(path):
            print('%s  MISSING at %s' % (name, path))
            continue
        d = loadmat(path, squeeze_me=True)
        missing = [k for k in ('r_sim', 'y', 'u_fb', 'f_sim', 'u_total') if k not in d]
        if missing:
            print('%s  record lacks %s, skipped' % (name, missing))
            continue

        r = np.asarray(d['r_sim'], float)
        y = np.asarray(d['y'], float)
        u_fb = np.asarray(d['u_fb'], float)
        f_sim = np.asarray(d['f_sim'], float)
        u_tot = np.asarray(d['u_total'], float)
        dt = float(d['dt'])
        N = len(y)

        print('%s   split=%-5s  Y_op = %+5.2f m   N = %d   dt = %.6e s (design %.6e)'
              % (name, split, Y_op, N, dt, TS))
        # dt is stored as single (gtd_save_record.m:35), so it can only agree with the design ts
        # to a float32 ulp; anything larger is a real rate mismatch.
        if abs(dt - TS) > 2 * np.spacing(np.float32(TS)):
            print('    NOTE stored dt differs from the design ts; Cfb is rebuilt at the '
                  'design ts anyway')

        # ---- (1) additivity of the stored signals ---------------------------
        res_add = u_tot - (u_fb + f_sim)
        s = _stats(res_add, u_tot, k0)
        print('  [1] u_total - (u_fb + f_sim)                 channels  %s' % '      '.join(CH))
        for tag in ('full', 'settled'):
            mx, rms, rms_ref, rel = s[tag]
            lab = 'full record' if tag == 'full' else 'from t = %.1f s' % T_SKIP
            print(_row('%s  max abs [N]' % lab, mx))
            print(_row('%s  rms [N]' % lab, rms))
            print(_row('%s  rel to rms(u_total)' % lab, rel))
        # float32 step on the largest stored value in the sum, the floor this identity can reach
        q_add = np.array([np.spacing(np.float32(np.abs(u_tot[:, j]).max())) for j in range(3)])
        print(_row('float32 step on u_total [N]', q_add))
        add_tab.append((name, split, s['settled'][0], s['settled'][3], q_add))

        # ---- (2) recomputed u_fb --------------------------------------------
        (A, B, C, D), (cfb, gains) = cache[Y_op]
        e = r - y
        _, u_hat, _ = dlsim((A, B, C, D, TS), e, x0=np.zeros(A.shape[0]))
        u_hat = np.asarray(u_hat, float)
        res_c = u_hat - u_fb
        sc = _stats(res_c, u_fb, k0)
        print('  [2] u_fb_recomputed - u_fb                   channels  %s' % '      '.join(CH))
        print(_row('rms(u_fb) [N]', sc['full'][2]))
        print(_row('rms(e = r_sim - y) [m]', np.sqrt(np.mean(e ** 2, axis=0))))
        print(_row('mean(e) [m]', e.mean(axis=0)))
        for tag in ('full', 'settled'):
            mx, rms, rms_ref, rel = sc[tag]
            lab = 'full record' if tag == 'full' else 'from t = %.1f s' % T_SKIP
            print(_row('%s  max abs [N]' % lab, mx))
            print(_row('%s  rms [N]' % lab, rms))
            print(_row('%s  rel to rms(u_fb)' % lab, rel))

        # ---- residual decomposition: ramp from the integrator vs the rest ----
        t = np.arange(N) * TS
        Afit = np.vstack([t, np.ones_like(t)]).T
        frac = np.empty(3); slope_fit = np.empty(3); slope_pred = np.empty(3)
        rest_rms = np.empty(3)
        for j in range(3):
            coef, *_ = np.linalg.lstsq(Afit, res_c[:, j], rcond=None)
            ramp = Afit @ coef
            rest = res_c[:, j] - ramp
            frac[j] = 1.0 - np.var(rest) / np.var(res_c[:, j]) if np.var(res_c[:, j]) > 0 else 0.0
            slope_fit[j] = coef[0]
            # kappa_j * w / 54 is the integral gain of kappa_j*Cnorm(s) as s -> 0
            slope_pred[j] = gains[j] * W / 54.0 * e[:, j].mean()
            rest_rms[j] = np.sqrt(np.mean(rest ** 2))
        print(_row('ramp explains [%]', 100 * frac, '%8.3f'))
        print(_row('slope fitted [N/s]', slope_fit, '%+.4e'))
        print(_row('slope predicted [N/s]', slope_pred, '%+.4e'))
        print(_row('residual less ramp, rms [N]', rest_rms))
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_rest = rest_rms / sc['full'][2]
        print(_row('  the same, rel to rms(u_fb)', rel_rest))

        # ---- measured storage floor: re-apply the rounding that single() applied ----
        # ulp of the stored y at each sample; the information lost is uniform on +-1/2 ulp.
        rng = np.random.default_rng(0)
        ulp = np.spacing(np.abs(y).astype(np.float32)).astype(float)
        y_alt = y + (rng.random(y.shape) - 0.5) * ulp
        _, u_alt, _ = dlsim((A, B, C, D, TS), r - y_alt, x0=np.zeros(A.shape[0]))
        d_store = np.asarray(u_alt, float) - u_hat
        s_store = _stats(d_store, u_fb, k0)
        print('    -- storage floor, measured: same Cfb on an independent redraw of the '
              'single() rounding')
        for tag in ('full', 'settled'):
            mx, rms, _, rel = s_store[tag]
            lab = 'full record' if tag == 'full' else 'from t = %.1f s' % T_SKIP
            print(_row('%s  max abs [N]' % lab, mx))
            print(_row('%s  rms [N]' % lab, rms))
            print(_row('%s  rel to rms(u_fb)' % lab, rel))
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = sc['settled'][1] / s_store['settled'][1]
        print(_row('residual / storage floor', ratio, '%9.3f'))
        ctl_tab.append((name, split, sc['settled'][0], sc['settled'][3], rel_rest,
                        s_store['settled'][3], ratio))

    # ---- compact per-record, per-channel recap -------------------------------
    print('\n' + '=' * 100)
    print('RECAP 1  u_total - (u_fb + f_sim), first %.1f s discarded' % T_SKIP)
    print('=' * 100)
    print('%-22s %-6s %-32s %-32s' % ('record', 'split', 'max abs [N]  X1  X2  Y',
                                      'rel to rms(u_total)  X1  X2  Y'))
    for name, split, mx, rel, q in add_tab:
        print('%-22s %-6s %s   %s' % (name, split,
                                      ' '.join('%.3e' % v for v in mx),
                                      ' '.join('%.3e' % v for v in rel)))

    print('\n' + '=' * 100)
    print('RECAP 2  u_fb_recomputed - u_fb, first %.1f s discarded' % T_SKIP)
    print('=' * 100)
    print('%-22s %-6s %-32s %-32s %-32s %-32s %s'
          % ('record', 'split', 'max abs [N]  X1  X2  Y', 'rel to rms(u_fb)  X1  X2  Y',
             'rel, ramp removed', 'storage floor, rel', 'residual / storage floor'))
    for name, split, mx, rel, relr, relf, ratio in ctl_tab:
        print('%-22s %-6s %s   %s   %s   %s   %s'
              % (name, split, ' '.join('%.3e' % v for v in mx),
                 ' '.join('%.3e' % v for v in rel), ' '.join('%.3e' % v for v in relr),
                 ' '.join('%.3e' % v for v in relf), ' '.join('%8.3f' % v for v in ratio)))
    print('\nA ratio at or below 1 means the residual is inside the spread of u_fb values that '
          'are all\nequally consistent with the single()-rounded signals the record stores, i.e. '
          'it is not\nevidence about the controller.')


if __name__ == '__main__':
    main()
