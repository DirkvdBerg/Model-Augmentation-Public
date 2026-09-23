"""G4 verdict: TR-014 (a)-(c) evaluated from the replay metrics and the recovered parameters.

    python telica-real/baseline/g4_verdict.py <candidate variant> <candidate params tag>
e.g. rec_a2 _a2. Reference variant for (b): 70821. Writes outputs/g4_verdict_<variant>.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tr_env                                                          # noqa: E402

import numpy as np                                                     # noqa: E402

from params import telica_params as tp                                 # noqa: E402
tr_env.check_no_leak()

VAR, TAG = sys.argv[1], sys.argv[2]
O = tr_env.OUTPUTS
HERE = os.path.dirname(os.path.abspath(__file__))


def metrics(v):
    return json.load(open(os.path.join(O, f'g4_replay_{v}', 'metrics.json')))


def main():
    C, R = metrics(VAR), metrics('70821')
    P = json.load(open(os.path.join(HERE, f'recovered_params{TAG}.json')))
    H = json.load(open(os.path.join(O, 'g4_holding', 'holding.json')))
    held = np.array([[(r['F_end'][j] - r['F_start'][j]) / 2 for j in range(3)]
                     for r in H if r['rest_ok'] and r['split'] == 'train'])
    p975 = np.percentile(held, 97.5, axis=0)
    out = {}
    # (a) stable, both forms, every record
    s = C['summary']
    a = bool(s['all_stable'])
    out['a'] = dict(PASS=a, n=s['n'], unstable_direct=s['n_unstable_direct'],
                    unstable_resid=s['n_unstable_resid'])
    # (b) better than 70821 on held-out, median over (record, axis) of NRMSE_direct
    def med_ho(M):
        v = np.array([r['nrmse_direct'] for r in M['per_record'] if r['split'] != 'train'], float)
        return float(np.nanmedian(v)), int(np.isnan(v).any(axis=1).sum())
    mc, nc = med_ho(C); mr, nr = med_ho(R)
    b = mc < mr
    out['b'] = dict(PASS=bool(b), candidate=mc, ref_70821=mr, ref_nonfinite_records=nr,
                    ref_all_stable=R['summary']['all_stable'],
                    ref_unstable=[R['summary']['n_unstable_direct'], R['summary']['n_unstable_resid']])
    # (c) physical
    cb = P['combos']; cc = np.array(P['cc'])
    Ys = np.linspace(-0.25, 0.25, 201)
    mh, d, J = cb['mh'], cb['d'], cb['J_eff']
    alpha = cb['m_total'] + mh; beta = cb['m_diff'] * tp.Lb / 2
    eig = min(np.linalg.eigvalsh(np.array([[alpha, beta - mh * Y, 0],
                                           [beta - mh * Y, J + mh * d ** 2 + mh * Y ** 2, -mh * d],
                                           [0, -mh * d, mh]])).min() for Y in Ys)
    # LFR denominator, gantry_ss.build_poly_constants: gamma = Jb + Jh + (m1+m2) Lb^2/4 = J_eff
    dY = [mh * (alpha * J - beta ** 2 + 2 * beta * mh * Y + mh * (alpha - mh) * Y ** 2)
          for Y in Ys]
    pos = all(v > 0 for k, v in cb.items() if k != 'm_diff')
    visc = all(cb[k] >= 0 for k in ('cg1', 'cg2', 'cy', 'cb_sum')) and bool((cc >= 0).all())
    cc_ok = bool((cc <= p975).all())
    mass_x = (cb['m_total'] + mh) / 91.0 - 1; mass_y = mh / 19.0 - 1
    mass_ok = abs(mass_x) <= 0.5 and abs(mass_y) <= 0.5
    c = pos and eig > 0 and min(abs(v) for v in dY) > 0 and visc and cc_ok and mass_ok
    out['c'] = dict(PASS=bool(c), combos_positive=pos, min_eig_M=float(eig),
                    min_abs_dY=float(min(abs(v) for v in dY)), viscous_coulomb_nonneg=visc,
                    cc=cc.tolist(), cc_bound_p975=p975.tolist(), cc_ok=cc_ok,
                    mass_dev_X=float(mass_x), mass_dev_Y=float(mass_y), mass_ok=mass_ok,
                    cc_source=P.get('cc_source'))
    out['PASS'] = bool(a and b and c)
    json.dump(out, open(os.path.join(O, f'g4_verdict_{VAR}.json'), 'w'), indent=1)
    for k in ('a', 'b', 'c'):
        print(f'({k}) {"PASS" if out[k]["PASS"] else "FAIL"}: {out[k]}')
    print('G4', 'PASS' if out['PASS'] else 'FAIL')


if __name__ == '__main__':
    main()
