"""SNR of the servo error per stage axis over the training records (Section V-A, noise paragraph).

Justifies: the noise level stated as representative of the machine, by the SNR it gives in the data.
Input: the 18 training records of the dataset and their noise-free twins.
Output: Results/setup_snr.csv (per record and axis), macros SetupSnr{Min,Med,Max}<axis> [dB] and
SetupNoiseRms<axis> [m, median over records].
"""
import os

import numpy as np

import _setup_common as sc


def main():
    ds = sc.dataset_arg()
    rows, snr, nrms = [], {a: [] for a in sc.AXES}, {a: [] for a in sc.AXES}
    for row in sc.manifest(ds, ('training',)):
        d = sc.load(sc.record_path(ds, row), ['y', 'r_sim'])
        d0 = sc.load(sc.record_path(sc.NOISE_FREE, row), ['y', 'r_sim'])
        assert np.array_equal(d['r_sim'], d0['r_sim']), row['file']   # twins share the reference
        # THEORY: SNR_j = 20 log10( RMS(y0_j - r_j) / RMS(y_j - y0_j) ), Section V-A; the noise term
        # is the noisy minus the noise-free record, so it holds the loop's response to the noise
        sig = sc.rms(d0['y'] - d0['r_sim'])
        noi = sc.rms(d['y'] - d0['y'])
        s = 20 * np.log10(sig / noi)
        for j, a in enumerate(sc.AXES):
            snr[a].append(s[j]); nrms[a].append(noi[j])
            rows.append([row['file'], a, sig[j], noi[j], s[j]])
        print('%-8s SNR [dB] %s' % (row['file'], np.round(s, 1)))
    sc.write_csv('snr', ['record', 'axis', 'rms_servo_error_m', 'rms_noise_m', 'snr_dB'], rows)
    mac = {}
    for a in sc.AXES:
        t = sc.AXIS_TAG[a]
        v = np.array(snr[a])
        mac['SetupSnrMin' + t] = ('%.0f' % v.min(), 'min SNR [dB] over training records, axis ' + a)
        mac['SetupSnrMed' + t] = ('%.0f' % np.median(v), 'median SNR [dB], axis ' + a)
        mac['SetupSnrMax' + t] = ('%.0f' % v.max(), 'max SNR [dB], axis ' + a)
        mac['SetupNoiseRms' + t] = (sc.sci(np.median(nrms[a])), 'median RMS(y - y0) [m], axis ' + a)
    vx = np.array(snr['X1'] + snr['X2'])
    mac['SetupSnrMinX'] = ('%.0f' % vx.min(), 'min SNR [dB] over training records, both X rails')
    mac['SetupSnrMaxX'] = ('%.0f' % vx.max(), 'max SNR [dB], both X rails')
    sc.set_macros(os.path.splitext(os.path.basename(__file__))[0], mac)


if __name__ == '__main__':
    main()
