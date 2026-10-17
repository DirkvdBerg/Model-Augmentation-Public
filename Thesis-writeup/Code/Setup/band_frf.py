"""Frequency responses behind the multisine band (Section V-B, Appendix app:setup Fig. setup_band).

Justifies: "The band is placed with the frequency response of the simulated system under K1 at the five
training positions, so that the anti-resonance, the coupled mode and the closed-loop peak lie inside it"
(D-219, D-226; simulation-only design information, D-259 item 1).
Per training position Y and frequency f, for the stage axis Y (the absorber acts along Y), linearised
(absorber on, friction off), for the simulated system (true) and the rigid baseline (base):
  G = |G_YY| open loop, stage force to stage position
  T = |(I + G K1)^-1 G K1|_YY, the complementary sensitivity of the loop with K1
Features per Y: anti-resonance = minimum of G_true/G_base, coupled mode = maximum of G_true/G_base,
closed-loop peak = maximum of T_true, all searched above 100 Hz.
Input: meta.truth. Output: Results/setup_band.csv (responses), Results/setup_band_features.csv.
"""
import numpy as np

import _setup_common as sc

TRAIN_Y = (-0.30, -0.15, 0.0, 0.15, 0.30)
F = np.geomspace(20.0, 1000.0, 4000)
J = 2                                                   # stage axis Y in (X1, X2, Y)


def comp_sens(G, Kc, kappa):
    out = np.empty(len(F))
    for i in range(len(F)):
        L = G[i] @ np.diag(kappa * Kc[i])
        # THEORY: complementary sensitivity T = (I + L)^-1 L of the loop u = K1 (r - y)
        out[i] = abs(np.linalg.solve(np.eye(3) + L, L)[J, J])
    return out


def proc_sens(G, Kc, kappa):
    out = np.empty(len(F))
    for i in range(len(F)):
        L = G[i] @ np.diag(kappa * Kc[i])
        # THEORY: process sensitivity S G = (I + G K1)^-1 G, injected force to position in the loop with K1
        out[i] = abs(np.linalg.solve(np.eye(3) + L, G[i])[J, J])
    return out


def main():
    ds = sc.dataset_arg()
    T = sc.truth_params(ds)
    num, den, kappa = sc.k1_continuous(T)
    Kc = np.polyval(num, 2j * np.pi * F) / np.polyval(den, 2j * np.pi * F)
    rows, feats, ps_lo, ps_hi = [], [], [], []
    for Y in TRAIN_Y:
        G = sc.frf_stage(T, Y, F, truth=True)
        Gb = sc.frf_stage(T, Y, F, truth=False)
        gt, gb = np.abs(G[:, J, J]), np.abs(Gb[:, J, J])
        tt, tb = comp_sens(G, Kc, kappa), comp_sens(Gb, Kc, kappa)
        pt, pb = proc_sens(G, Kc, kappa), proc_sens(Gb, Kc, kappa)
        for i in range(len(F)):
            rows.append([Y, F[i], gt[i], gb[i], tt[i], tb[i], pt[i], pb[i]])
        hi = F > 100.0
        band = (F >= 100.0) & (F <= 300.0)
        # THEORY: difference in dB, 20 log10(|S G|_true / |S G|_base)
        dps = 20 * np.log10(pt[band] / pb[band])
        ps_lo.append(dps.min()); ps_hi.append(dps.max())
        i_ar = np.argmin(np.where(hi, gt / gb, np.inf))
        i_cm = np.argmax(np.where(hi, gt / gb, -np.inf))
        i_pk = np.argmax(np.where(hi, tt, -np.inf))
        feats.append([Y, F[i_ar], F[i_cm], F[i_pk]])
        print('Y %+.2f  anti-resonance %.1f Hz  coupled mode %.1f Hz  closed-loop peak %.1f Hz'
              % (Y, F[i_ar], F[i_cm], F[i_pk]))
    sc.write_csv('band', ['Y', 'f_Hz', 'G_true', 'G_base', 'T_true', 'T_base', 'PS_true', 'PS_base'], rows)
    sc.write_csv('band_features', ['Y', 'anti_resonance_Hz', 'coupled_mode_Hz', 'closed_loop_peak_Hz'], feats)
    pk = np.array(feats)[:, 3]
    print('closed loop F_Y -> Y, system against baseline, 100-300 Hz: %.1f to %.1f dB' % (min(ps_lo), max(ps_hi)))
    sc.set_macros('band_frf', {
        'SetupClPeak': ('%.0f' % np.median(pk), 'closed-loop peak of T_YY with K1, median over training Y [Hz]'),
        'SetupAddedMassDbLo': ('%.0f' % min(ps_lo), 'lowest closed-loop |S G|_YY of the system against the baseline, 100-300 Hz, training Y [dB]'),
        'SetupAddedMassDbHi': ('%.0f' % max(ps_hi), 'highest closed-loop |S G|_YY of the system against the baseline, 100-300 Hz, training Y [dB]'),
    })


if __name__ == '__main__':
    main()
