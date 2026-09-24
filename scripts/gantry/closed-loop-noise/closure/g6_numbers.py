"""G6 numbers: the formula table per band, aliasing, differencing, precision, the CL_NOISE_CONSISTENT
shortcut against the in-loop injection, and the motion upper bound.

    python closure/g6_numbers.py
Reads G1, G3 outputs, outputs/recipe/recipe.mat, the twin and the noise-off references. Summaries only.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cn_env                                                           # noqa: E402

import numpy as np                                                      # noqa: E402
from scipy.io import loadmat                                            # noqa: E402

from measure import spec                                                # noqa: E402
from sensitivity import loop                                            # noqa: E402
from sensitivity.g2_sensitivity import model_S                          # noqa: E402
from closure.g5_check import model_psd_fft, ld, TWIN, PROD, REF          # noqa: E402
cn_env.check_no_leak()

OUT = cn_env.out_dir('g6_numbers')
FS = spec.FS; AX = cn_env.AX
BANDS = ((19.5, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 10001))
SIGMA_SNR60 = 1.4e-4                    # data.py load_datasets, SNR 60 (D-078)
P = np.array([[1.0, 1.0, 0.0], [0.3625, -0.3625, 0.0], [0.0, 0.0, 1.0]])   # Garcia Lb = 0.725


def band_rms(f, A):
    df = f[1] - f[0]
    return {f'{a:g}-{b:g}': (np.sqrt(A[(f >= a) & (f < b)].sum(0) * df) * 1e9).round(3).tolist() for a, b in BANDS}


def tot(f, A, lo=19.5, hi=10001):
    df = f[1] - f[0]
    return np.sqrt(A[(f >= lo) & (f < hi)].sum(0) * df)


def main():
    out = {}
    g1 = np.load(os.path.join(cn_env.OUTPUTS, 'g1_spectra', 'g1.npz'), allow_pickle=True)
    g3 = np.load(os.path.join(cn_env.OUTPUTS, 'g3_source', 'g3.npz'), allow_pickle=True)
    f = g1['f']; names = list(g1['names']); keep = g1['keep']; Kr = g1['Kr']; xy = g1['xy']
    Pe = g1['Ppool']; Ae = spec.autos(Pe)
    ops = sorted(set(n.split('/')[1] for n in names))
    pos = [xy[[i for i, n in enumerate(names) if n.split('/')[1] == op][0]] for op in ops]
    w = np.array([Kr[[i for i, n in enumerate(names) if n.split('/')[1] == op and keep[i]]].sum() for op in ops], float)
    w /= w.sum()
    Sm, _, _ = model_S(pos, f, readings=('tanh',))
    Pv = g3['tanh_Pv']
    # ==== A. Telica loop ===================================================================
    twice = sum(w[p] * spec.shape(Sm['tanh'][p], Pe) for p in range(len(pos)))
    reshaped = sum(w[p] * spec.shape(Sm['tanh'][p], Pv) for p in range(len(pos)))
    out['telica'] = dict(measured_e=band_rms(f, Ae), source_v_tanh=band_rms(f, spec.autos(Pv)),
                         source_v_sliding=band_rms(f, spec.autos(g3['sliding_Pv'])),
                         inject_v_gives_e=band_rms(f, spec.autos(reshaped)),
                         inject_measured_e_gives_e=band_rms(f, spec.autos(twice)),
                         totals_nm=dict(measured=(tot(f, Ae) * 1e9).tolist(),
                                        inject_v=(tot(f, spec.autos(reshaped)) * 1e9).tolist(),
                                        inject_measured=(tot(f, spec.autos(twice)) * 1e9).tolist()))
    # ==== B. simulated (Garcia) loop ===========================================================
    rc = loadmat(os.path.join(cn_env.OUTPUTS, 'recipe', 'recipe.mat'), squeeze_me=True, struct_as_record=False)
    SS = np.atleast_1d(rc['SS'])
    Pvs = np.zeros_like(Pe); Pes = np.zeros_like(Pe); Kv = np.zeros_like(Pe); KSv = np.zeros_like(Pe); Tv = np.zeros_like(Pe)
    for s in SS:
        S = loop.frf(s.A, s.B, s.C, s.D, f)
        K = loop.frf(s.KA, s.KB, s.KC, s.KD, f)
        K3 = K                                   # Cfb: 3x3 diagonal, error [m] -> stage force [N]
        Pvs += spec.shape(S, Pv) / len(SS)
        Pes += spec.shape(S, Pe) / len(SS)
        Kv += spec.shape(K3, Pv) / len(SS)
        KSv += spec.shape(K3 @ S, Pv) / len(SS)
        Tv += spec.shape(np.eye(3)[None] - S, Pv) / len(SS)
    white = np.full_like(Ae, 2 * SIGMA_SNR60 ** 2 / FS)
    out['simulated'] = dict(
        inject_v_error=band_rms(f, spec.autos(Pvs)), inject_measured_error=band_rms(f, spec.autos(Pes)),
        datapy_snr60_white=band_rms(f, white),
        totals_nm=dict(inject_v=(tot(f, spec.autos(Pvs)) * 1e9).tolist(),
                       inject_measured=(tot(f, spec.autos(Pes)) * 1e9).tolist(),
                       v_itself=(tot(f, spec.autos(Pv)) * 1e9).tolist(),
                       true_position_Tv=(tot(f, spec.autos(Tv)) * 1e9).tolist(),
                       datapy_snr60=(tot(f, white) * 1e9).tolist()),
        shortcut_vs_inloop=dict(
            y_noise_shortcut_v_nm=(tot(f, spec.autos(Pv)) * 1e9).tolist(),
            y_noise_inloop_Sv_nm=(tot(f, spec.autos(Pvs)) * 1e9).tolist(),
            u_noise_shortcut_Kv_N=tot(f, spec.autos(Kv)).tolist(),
            u_noise_inloop_KSv_N=tot(f, spec.autos(KSv)).tolist(),
            true_position_inloop_Tv_nm=(tot(f, spec.autos(Tv)) * 1e9).tolist()))
    # ==== C. aliasing of the twin's y noise (S_sim v) when point-sampled =======================
    Ay = spec.autos(Pvs); Av = spec.autos(Pv)
    out['aliasing'] = {f'D{D}_{FS / D / 1e3:g}kHz': dict(
        nyquist_hz=FS / D / 2,
        y_noise_var_fraction_above=(Ay[f > FS / D / 2].sum(0) / Ay[f > 0].sum(0)).round(4).tolist(),
        v_var_fraction_above=(Av[f > FS / D / 2].sum(0) / Av[f > 0].sum(0)).round(4).tolist())
        for D in (4, 5)}
    # ==== D. differencing and precision, from the twin (8 standstill + 7 Telica-profile) =======
    vel = {4: [], 5: []}; q32 = []; true_vel = []
    for fn in sorted(os.listdir(REF)):
        if not fn.endswith('.mat'):
            continue
        n = ld(os.path.join(TWIN, fn), ['y', 'q_true', 'v_enc']); r = ld(os.path.join(REF, fn), ['q_true'])
        ny = n['y'] - r['q_true']                                 # noise part of the recorded y
        for D in vel:
            d = np.diff((np.linalg.solve(P.T, ny[::D].T)).T, axis=0) * FS / D
            vel[D].append(d.std(0))
        q32.append(np.sqrt(((n['y'].astype(np.float32).astype(np.float64) - n['y']) ** 2).mean(0)))
    for fn in sorted(os.listdir(PROD)):
        if fn.startswith('T') and fn.endswith('.mat') and not fn.startswith('TP'):
            true_vel.append(ld(os.path.join(PROD, fn), ['x_logical'])['x_logical'][:, 3:].std(0))
    out['differencing'] = {f'D{D}': dict(noise_velocity_std_logical=np.median(vel[D], 0).tolist())
                           for D in vel}
    out['differencing']['true_velocity_std_logical_train_median'] = np.median(true_vel, 0).tolist()
    out['float32_rounding_rms_nm'] = (np.median(q32, 0) * 1e9).tolist()
    # ==== F. motion upper bound ==================================================================
    j1 = json.load(open(os.path.join(cn_env.OUTPUTS, 'g1_spectra', 'g1.json')))
    fm = g1['nr_f0']; Pnr = g1['nr_P0']
    mb = {}
    for rd in ('tanh', 'sliding'):
        Sr, _, _ = model_S(pos, fm[1:], readings=(rd,))
        S2 = np.mean(np.abs(np.einsum('pfii->pfi', Sr[rd])) ** 2, 0)
        vr = np.sqrt((Pnr[1:] / S2)[(fm[1:] >= 19.5)].sum(0) * (fm[1] - fm[0]))
        mb[rd] = (vr * 1e9).round(1).tolist()
    out['motion_bound'] = dict(nonrep_e_iter0_nm=j1['nonrep']['iter0']['rms_nm'],
                               nonrep_e_ETEL_nm=j1['nonrep']['iterETEL']['rms_nm'],
                               ilc_factor=[1, 2],
                               implied_source_diag_nm=mb,
                               note='diagonal |S_ii|^2 de-shaping (the motion split keeps autos only)')
    json.dump(out, open(os.path.join(OUT, 'g6.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
