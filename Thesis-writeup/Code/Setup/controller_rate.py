"""Controller-rate mismatch: K1 at the 4 kHz training rate against K1 at the 20 kHz record rate
(Section V-B; the open \todo "size of the controller mismatch").

Per training record (noisy version, as trained on):
  check    : K1 Tustin at 20 kHz on the recorded r - y gives the recorded u_fb (validates this K1)
  mismatch : K1 Tustin at 4 kHz on the decimated servo error (eq:setup_decim) against the recorded
             u_fb at the same instant kD, relative RMS per axis (D-265)
  hold     : the block-mean force (what the training data holds, D-087) against u_fb at kD: the
             force change within one 4 kHz hold, i.e. the cost of closing the loop at 4 kHz
The first 0.5 s (the hold before the motion, tdg_config t_hold) is left out, so both filters start
from the same rest state. HEURISTIC: 0.5 s, the generator's hold time.
Input: training records. Output: Results/setup_controller_rate.csv; macros SetupKcheck,
SetupKmismatchMed, SetupKmismatchMax (relative RMS).
"""
import os

import numpy as np
from scipy.signal import butter, cont2discrete, lfilter, sosfiltfilt

import _setup_common as sc

FS, D = 20000, 5
SOS = butter(8, 1600.0, fs=FS, output='sos')        # HEURISTIC (D-232), as filter_leak.py
SKIP = int(0.5 * FS)


def k1_discrete(T, fs):
    num, den, kappa = sc.k1_continuous(T)
    out = []
    for j in range(3):
        b, a, _ = cont2discrete((kappa[j] * num, den), 1.0 / fs, method='bilinear')
        out.append((np.ravel(b), np.ravel(a)))
    return out


def main():
    ds = sc.dataset_arg()
    T = sc.truth_params(ds)
    K20, K4 = k1_discrete(T, FS), k1_discrete(T, FS // D)
    rows, chk, mis, hold, motion = [], [], [], [], []
    for row in sc.manifest(ds, ('training',)):
        d = sc.load(sc.record_path(ds, row), ['y', 'r_sim', 'u_fb'])
        e20 = d['r_sim'] - d['y']
        u20 = np.column_stack([lfilter(*K20[j], e20[:, j]) for j in range(3)])
        # THEORY: eq:setup_decim, y_dec = r[::D] + BW(y - r)[::D], so e_dec = -BW(y - r)[::D]
        e4 = -sosfiltfilt(SOS, d['y'] - d['r_sim'], axis=0)[::D]
        u4 = np.column_stack([lfilter(*K4[j], e4[:, j]) for j in range(3)])
        n = u4.shape[0]
        ufb4 = d['u_fb'][:n * D].reshape(n, D, 3).mean(axis=1)      # block mean, as GD/data.py::_resample_u
        # D-265: the 4 kHz output at sample kD is compared with u_fb at the SAME instant kD. The block
        # mean is centred 0.1 ms later; its difference from u_fb(kD) is the force change within one
        # hold, reported separately as the cost of closing the loop at 4 kHz.
        ufb_pt = d['u_fb'][:n * D:D]
        s20, s4 = slice(SKIP, None), slice(SKIP // D, None)
        c = sc.rms(u20[s20] - d['u_fb'][s20]) / sc.rms(d['u_fb'][s20])
        m = sc.rms(u4[s4] - ufb_pt[s4]) / sc.rms(ufb_pt[s4])
        h = sc.rms(ufb4[s4] - ufb_pt[s4]) / sc.rms(ufb4[s4])
        chk.append(c); mis.append(m); hold.append(h); motion.append(row['excitation'] == 'none')
        for j, a in enumerate(sc.AXES):
            rows.append([row['file'], a, c[j], m[j], h[j]])
        print('%-8s check %s  mismatch %s  hold %s' % (row['file'], np.array2string(c, precision=1),
              np.array2string(m, precision=3), np.array2string(h, precision=3)))
    chk, mis, hold, motion = np.array(chk), np.array(mis), np.array(hold), np.array(motion)
    sc.write_csv('controller_rate', ['record', 'axis', 'rel_rms_20kHz_check', 'rel_rms_4kHz_mismatch',
                                     'rel_rms_hold_variation'], rows)
    sc.set_macros(os.path.splitext(os.path.basename(__file__))[0], {
        'SetupKcheck': (sc.sci(chk.max()), 'largest relative RMS, K1 at 20 kHz vs recorded u_fb'),
        'SetupKmismatchMed': ('%.1f' % (100 * np.median(mis)), 'median relative RMS [%], K1 at 4 kHz vs u_fb at the same instant'),
        'SetupKmismatchMotion': ('%.1f' % (100 * mis[motion].max()), 'largest relative RMS [%] in the records without multisine'),
        'SetupKmismatchMax': ('%.1f' % (100 * mis.max()), 'largest relative RMS [%], same'),
        'SetupKholdMed': ('%.1f' % (100 * np.median(hold)), 'median relative RMS [%] of the force change within one 4 kHz hold (block mean vs u_fb at kD)'),
    })


if __name__ == '__main__':
    main()
