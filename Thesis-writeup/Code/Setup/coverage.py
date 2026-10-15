"""Coverage of position, velocity and acceleration per record (Section V-B, splits; coverage figure).

Justifies: "Training covers |Y| <= 0.30 m, and accelerations and velocities up to 75 % of the machine
maximum" and the one-property extrapolation of E1 to E4 and E6, from the references of the records.
Per record: Y range, and peak |velocity| and |acceleration| of the reference r per axis (X = max of the
two X rails), as a percentage of the machine limits read from the generator (tdg_config.m lim).
Velocity and acceleration by central differences of r at 20 kHz (r is the smooth setpoint).
Input: all 36 records (references only). Output: Results/setup_coverage.csv (also the data of the
coverage figure); macros for the training range and the E records.
"""
import os
import re

import numpy as np

import _setup_common as sc

TDG_CONFIG = os.path.join(sc.TW, 'Code', 'Data', 'tdg_config.m')


def limits():
    """lim.vel, lim.acc_X, lim.acc_Y from the generator config (never retyped)."""
    txt = open(TDG_CONFIG).read()
    get = lambda k: float(re.search(r'lim\.%s\s*=\s*([0-9.eE+-]+)' % k, txt).group(1))
    return get('vel'), get('acc_X'), get('acc_Y')


def main():
    ds = sc.dataset_arg()
    vmax, ax, ay = limits()
    rows, out = [], {}
    for row in sc.manifest(ds):
        r = sc.load(sc.record_path(ds, row), ['r_sim'])['r_sim']
        # THEORY: central differences, np.gradient, at the 20 kHz sample time
        v = np.gradient(r, 1.0 / 20e3, axis=0)
        a = np.gradient(v, 1.0 / 20e3, axis=0)
        Ymin, Ymax = r[:, 2].min(), r[:, 2].max()
        vX, vY = np.abs(v[:, :2]).max(), np.abs(v[:, 2]).max()
        aX, aY = np.abs(a[:, :2]).max(), np.abs(a[:, 2]).max()
        rec = dict(record=row['record'], block=row['block'], Ymin=Ymin, Ymax=Ymax,
                   vX_pct=100 * vX / vmax, vY_pct=100 * vY / vmax, aX_pct=100 * aX / ax, aY_pct=100 * aY / ay)
        out[row['record']] = rec
        rows.append([rec[k] for k in ('record', 'block', 'Ymin', 'Ymax', 'vX_pct', 'vY_pct', 'aX_pct', 'aY_pct')])
        print('%-6s %-11s Y [%+.3f %+.3f]  v %% [%5.1f %5.1f]  a %% [%5.1f %5.1f]' % (
            row['record'], row['block'], Ymin, Ymax, rec['vX_pct'], rec['vY_pct'], rec['aX_pct'], rec['aY_pct']))
    sc.write_csv('coverage', ['record', 'block', 'Ymin_m', 'Ymax_m', 'vX_pct', 'vY_pct', 'aX_pct', 'aY_pct'], rows)
    tr = [o for o in out.values() if o['block'] == 'training']
    pct = lambda o, k: max(o[k + 'X_pct'], o[k + 'Y_pct'])
    sc.set_macros(os.path.splitext(os.path.basename(__file__))[0], {
        'SetupTrainY': ('%.2f' % max(max(abs(o['Ymin']), abs(o['Ymax'])) for o in tr), 'largest |Y| of a training reference [m]'),
        'SetupTrainVel': ('%.0f' % max(pct(o, 'v') for o in tr), 'largest training velocity [% of max]'),
        'SetupTrainAcc': ('%.0f' % max(pct(o, 'a') for o in tr), 'largest training acceleration [% of max]'),
        'SetupEY': ('%.2f' % max(max(abs(out[k]['Ymin']), abs(out[k]['Ymax'])) for k in ('E1', 'E2')), 'largest |Y| of E1, E2 [m]'),
        'SetupEAcc': ('%.0f' % pct(out['E3'], 'a'), 'E3 acceleration [% of max]'),
        'SetupEVel': ('%.0f' % pct(out['E4'], 'v'), 'E4 velocity [% of max]'),
        'SetupESixAcc': ('%.0f' % pct(out['E6'], 'a'), 'E6 acceleration [% of max]'),
    })


if __name__ == '__main__':
    main()
