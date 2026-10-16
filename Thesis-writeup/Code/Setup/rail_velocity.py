"""Rail velocities of the training records against the friction scale 1/g (Section V-A, eq:setup_truth).

Justifies: "With g = 1000 s/m the friction is Coulomb-like at the rail velocities of the excited records"
(D-223, D-226; D-259 item 1). The earlier velocity figure (7 to 8 mm/s) was simulated at g = 100 (D-225).
For |v| >= 3/g, tanh(g v) is within 0.5 % of sign(v), so the law equals Coulomb friction there.
Per noise-free training record (simulated states, g as in meta.truth) and rail i in {X1, X2, Y}, with the
rail velocities v = P^T qdot from x_logical:
  v_rms  = RMS of v_i
  share  = fraction of samples with |v_i| >= 3/g
Input: noise-free training records. Output: Results/setup_rail_velocity.csv; macros SetupFricVzero [mm/s],
SetupRailShareMin, SetupRailShareMed [%], SetupRailVrmsMin [mm/s] (records with multisine only).
"""
import os

import numpy as np

import _setup_common as sc

# THEORY: tanh(3) = 0.9951, so c tanh(g v) is within 0.5 % of c sign(v) for |v| >= 3/g
K_COUL = 3.0


def main():
    ds = sc.NOISE_FREE
    T = sc.truth_params(ds)
    g = float(T['friction_gain'])
    P = sc.P_matrix(T)
    rows, shares, vrms = [], [], []
    for row in sc.manifest(ds, ('training',)):
        x = sc.load(sc.record_path(ds, row), ['x_logical'])['x_logical']
        v = x[:, 3:6] @ P                                   # rail velocities P^T qdot, columns X1, X2, Y
        s = np.mean(np.abs(v) >= K_COUL / g, axis=0)
        vr = sc.rms(v)
        ms = row['excitation'] != 'none' and float(row['A_Y_N'] or 0) > 0
        for j, a in enumerate(sc.AXES):
            rows.append([row['file'], row['excitation'], a, vr[j], s[j]])
        if ms:
            shares.append(s); vrms.append(vr)
        print('%-9s %-10s v_rms [mm/s] %s  share>=3/g %s' % (row['file'], row['excitation'],
              np.array2string(1e3 * vr, precision=1), np.array2string(100 * s, precision=1)))
    shares, vrms = np.array(shares), np.array(vrms)
    sc.write_csv('rail_velocity', ['record', 'excitation', 'axis', 'v_rms_m_s', 'share_coulomb'], rows)
    sc.set_macros(os.path.splitext(os.path.basename(__file__))[0], {
        'SetupFricVzero': ('%.0f' % (1e3 / g), 'friction velocity scale 1/g [mm/s]'),
        'SetupRailShareMin': ('%.0f' % (100 * shares.min()), 'smallest share of samples with |v| >= 3/g, multisine training records, any rail [%]'),
        'SetupRailShareMed': ('%.0f' % (100 * np.median(shares)), 'median share of samples with |v| >= 3/g, multisine training records [%]'),
        'SetupRailVrmsMin': ('%.1f' % (1e3 * vrms.min()), 'smallest rail velocity RMS, multisine training records [mm/s]'),
    })


if __name__ == '__main__':
    main()
