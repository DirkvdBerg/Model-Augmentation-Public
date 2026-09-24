"""G5 (c) and (d) (CN-009): the simulated noise part of the error against S_sim Phi_v S_sim^H,
plus integrator, headroom and stick-fraction reports.

    python closure/g5_check.py
Reads the twin (data/.../..._noise_ss), the noise-off double references (outputs/g5c_records),
the production records, outputs/recipe/recipe.mat (S_sim, Cfb) and H_v. Summaries only.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cn_env                                                           # noqa: E402

import numpy as np                                                      # noqa: E402
from scipy.io import loadmat                                            # noqa: E402
from scipy.signal import dlsim, get_window                              # noqa: E402

from measure import spec                                                # noqa: E402
from model import noise_model as nm                                     # noqa: E402
from sensitivity import loop                                            # noqa: E402
cn_env.check_no_leak()

TAG = os.environ.get('G5_TAG', '')                      # '' = the gate run; otherwise a variant
EXCLUDE = [s for s in os.environ.get('G5_EXCLUDE', '').split(',') if s]
OUT = cn_env.out_dir('g5_check' + (('_' + TAG) if TAG else ''))
TRAJ = os.path.join(cn_env.REPO, 'data', 'gantry', 'matlab', 'trajectory')
TWIN = os.environ.get('G5_TWIN') or os.path.join(TRAJ, 'augmentation_ma50_z03_b140-230_a6all_telica_coulomb_noise_ss')
PROD = os.path.join(TRAJ, 'augmentation_ma50_z03_b140-230_a6all_telica_coulomb')
REF = os.environ.get('G5_REF') or os.path.join(cn_env.OUTPUTS, 'g5c_records')
FS = spec.FS; TS = 1 / FS; AX = cn_env.AX
DISCARD = 10000; LBLK = 8000; NFINE = 131072
LIM_PEAK = np.array([2000.0, 2000.0, 1420.0]); LIM_RMS = np.array([916.0, 916.0, 656.0])   # gtd_config


def ld(path, keys):
    m = loadmat(path, variable_names=keys)
    return {k: np.asarray(m[k], float) for k in keys if k in m}


def model_psd_fft(A, Sig, N):
    p, m, _ = A.shape
    B = np.zeros((N, m, m)); B[0] = np.eye(m); B[1:p + 1] = -A
    H = np.linalg.inv(np.fft.fft(B, axis=0)[:N // 2 + 1])
    P = 2.0 / FS * H @ Sig[None] @ np.conj(np.transpose(H, (0, 2, 1)))
    P[0] /= 2
    return np.fft.rfftfreq(N, 1 / FS), P


def expected_welch(A_fine):
    """Autos on the fine grid -> expected Hann-2048 Welch autos (CN-008a)."""
    R = np.stack([nm.autocov(A_fine[:, i:i + 1, None], FS)[:, 0, 0] for i in range(3)], 1)
    N = spec.NPER
    win = get_window('hann', N)
    rho = np.correlate(win, win, 'full')[N - 1:] / np.sum(win ** 2)
    Rt = np.zeros((N, 3))
    Rt[:N // 2] = R[:N // 2] * rho[:N // 2, None]
    Rt[N // 2 + 1:] = R[1:N // 2][::-1] * rho[1:N // 2][::-1, None]
    Ew = np.real(np.fft.fft(Rt, axis=0))[:N // 2 + 1] / FS
    Ew[1:] *= 2; Ew[-1] /= 2
    return Ew


def stick(q):
    v = np.gradient(q, TS, axis=0)
    return (np.abs(v) < 1e-3).mean(0)


def ctrl_states(KA, KB, KC, KD, e):
    """Per-axis Cfb states driven by e (T, 3); returns (states (T, n), index of the integrator)."""
    _, _, x = dlsim((KA, KB, KC, KD, TS), e)
    lam, V = np.linalg.eig(KA)
    integ = int(np.argmin(np.abs(lam - 1)))
    z = np.real(np.linalg.solve(V, x.T).T)            # modal coordinates
    return z, integ


def main():
    g1 = np.load(os.path.join(cn_env.OUTPUTS, 'g1_spectra', 'g1.npz'), allow_pickle=True)
    band_of, fc = g1['band_of'], g1['fc']; nb = len(fc); f = g1['f']; df = f[1] - f[0]
    nbins = np.array([(band_of == b).sum() for b in range(nb)])
    rc = loadmat(os.path.join(cn_env.OUTPUTS, 'recipe', 'recipe.mat'), squeeze_me=True, struct_as_record=False)
    SS = [s for s in np.atleast_1d(rc['SS']) if s.id not in EXCLUDE and os.path.isfile(os.path.join(TWIN, s.id + '.mat'))]
    TPc = {t.id: t for t in np.atleast_1d(rc['TP']) if os.path.isfile(os.path.join(REF, t.id + '.mat'))
           and os.path.isfile(os.path.join(TWIN, t.id + '.mat'))}
    print(f'[g5] standstill records used: {[s.id for s in SS]}')
    hv = np.load(os.path.join(cn_env.OUTPUTS, 'g3_source', 'hv_tanh.npz'))
    ff, PHv = model_psd_fft(hv['A'], hv['Sig'], NFINE)
    summ = dict(records={})
    # ==== (c) standstill records ===========================================================
    blocks, Kb, rec_of = [], [], []
    Pe_an = np.zeros_like(PHv); Ktot = 0
    for k, s in enumerate(SS):
        n = ld(os.path.join(TWIN, s.id + '.mat'), ['q_true', 'v_enc', 'y', 'u_total', 'u_fb'])
        r = ld(os.path.join(REF, s.id + '.mat'), ['q_true', 'u_total', 'u_fb'])
        p = ld(os.path.join(PROD, s.id + '.mat'), ['y', 'u_total'])
        e_n = -(n['q_true'] + n['v_enc'] - r['q_true'])            # noise part of e = r - y
        assert np.allclose(n['y'], n['q_true'] + n['v_enc'], atol=0, rtol=0) or \
            np.abs(n['y'] - n['q_true'] - n['v_enc']).max() < 1e-15
        off_vs_prod = float(np.abs(r['q_true'] - p['y']).max())
        Kr = 0
        for b0 in range(DISCARD, len(e_n) - LBLK + 1, LBLK):
            _, Pb, K_ = spec.welch_sum(e_n[b0:b0 + LBLK])
            blocks.append(spec.band_mean(spec.autos(Pb), band_of, nb)); Kb.append(K_); rec_of.append(k)
            Kr += K_
        S = loop.frf(s.A, s.B, s.C, s.D, ff)
        Pe_an += Kr * spec.shape(S, PHv); Ktot += Kr
        # (d) integrator: noise part of the Cfb states, first vs second half
        z, integ = ctrl_states(s.KA, s.KB, s.KC, s.KD, e_n[DISCARD:])
        h = len(z) // 2
        summ['records'][s.id] = dict(
            e_noise_rms_nm=(e_n[DISCARD:].std(0) * 1e9).tolist(),
            covar_rms_nm=(np.sqrt(np.asarray(s.var_e)) * 1e9).tolist(),
            off64_vs_production_max_abs_m=off_vs_prod,
            integrator_state_std_first_second_half=[float(z[:h, integ].std()), float(z[h:, integ].std())],
            u_fb_noise_rms_N=(n['u_fb'] - r['u_fb'])[DISCARD:].std(0).tolist(),
            peak_force_noisy_N=np.abs(n['u_total']).max(0).tolist(),
            peak_force_prod_N=np.abs(p['u_total']).max(0).tolist(),
            stick_noisy=stick(n['q_true']).tolist(), stick_off=stick(r['q_true']).tolist())
        print(f'[g5] {s.id}: noise part of e {np.round(e_n[DISCARD:].std(0) * 1e9, 2)} nm '
              f'(covar {np.round(np.sqrt(np.asarray(s.var_e)) * 1e9, 2)}); off64 vs production max '
              f'{off_vs_prod:.2e} m; integrator std halves {z[:h, integ].std():.3e} / {z[h:, integ].std():.3e}')
    blocks = np.stack(blocks); Kb = np.array(Kb); rec_of = np.array(rec_of)
    band_sim = blocks.sum(0) / Kb.sum()
    rng = np.random.default_rng(13)
    boot = []
    for _ in range(1000):
        idx = np.concatenate([rng.choice(np.where(rec_of == k)[0], (rec_of == k).sum()) for k in range(len(SS))])
        boot.append(blocks[idx].sum(0) / Kb[idx].sum())
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    Ew = expected_welch(spec.autos(Pe_an / Ktot))
    band_an = spec.band_mean(Ew, band_of, nb)
    inside = ((band_an >= lo) & (band_an <= hi)).mean(0)
    rms_sim = np.sqrt((band_sim * nbins[:, None]).sum(0) * df)
    rms_an = np.sqrt((band_an * nbins[:, None]).sum(0) * df)
    rms_boot = np.sqrt((boot * nbins[None, :, None]).sum(1) * df)
    h = (np.percentile(rms_boot, 97.5, 0) - np.percentile(rms_boot, 2.5, 0)) / 2 / rms_sim
    rel = rms_sim / rms_an - 1
    ok = bool(np.all(inside >= 0.85) and np.all(np.abs(rel) <= 0.02 + h))
    summ.update(c_bands_inside=inside.tolist(), c_rms_sim_nm=(rms_sim * 1e9).tolist(),
                c_rms_analytic_nm=(rms_an * 1e9).tolist(), c_rms_rel=rel.tolist(), c_tol=(0.02 + h).tolist(),
                PASS_c=ok, n_blocks=int(len(Kb)))
    print(f'[g5] (c) analytic inside the simulated CI in {np.round(inside, 3)} of bands (>= 0.85); rms sim '
          f'{np.round(rms_sim * 1e9, 3)} vs analytic {np.round(rms_an * 1e9, 3)} nm ({np.round(rel * 100, 2)} %, '
          f'tol {np.round((0.02 + h) * 100, 2)} %) -> PASS {ok}')
    worst = np.argsort(-np.abs(band_sim / band_an - 1).max(1))[:6]
    print('[g5] (c) largest band deviations sim/analytic-1 [%]: ' + '; '.join(
        f'{fc[b]:.0f} Hz {np.round((band_sim[b] / band_an[b] - 1) * 100, 1)}' for b in worst))
    # ==== (d) Telica-profile records ==========================================================
    for rid, t in TPc.items():
        n = ld(os.path.join(TWIN, rid + '.mat'), ['q_true', 'v_enc', 'u_total', 'u_fb'])
        r = ld(os.path.join(REF, rid + '.mat'), ['q_true', 'u_total', 'u_fb'])
        p = ld(os.path.join(PROD, rid + '.mat'), ['u_total'])
        e_n = -(n['q_true'] + n['v_enc'] - r['q_true'])
        z, integ = ctrl_states(t.KA, t.KB, t.KC, t.KD, e_n[DISCARD:])
        hh = len(z) // 2
        summ['records'][rid] = dict(
            e_noise_rms_nm=(e_n[DISCARD:].std(0) * 1e9).tolist(),
            q_true_diff_rms_nm=((n['q_true'] - r['q_true'])[DISCARD:].std(0) * 1e9).tolist(),
            integrator_state_std_first_second_half=[float(z[:hh, integ].std()), float(z[hh:, integ].std())],
            u_fb_noise_rms_N=(n['u_fb'] - r['u_fb'])[DISCARD:].std(0).tolist(),
            peak_force_noisy_N=np.abs(n['u_total']).max(0).tolist(),
            peak_force_prod_N=np.abs(p['u_total']).max(0).tolist(),
            stick_noisy=stick(n['q_true']).tolist(), stick_off=stick(r['q_true']).tolist())
        R = summ['records'][rid]
        print(f'[g5] {rid}: noise part of e {np.round(np.array(R["e_noise_rms_nm"]), 2)} nm, true-position '
              f'change {np.round(np.array(R["q_true_diff_rms_nm"]), 2)} nm; stick noisy '
              f'{np.round(np.array(R["stick_noisy"]) * 100, 1)} % vs off {np.round(np.array(R["stick_off"]) * 100, 1)} %; '
              f'integrator std halves {R["integrator_state_std_first_second_half"]}')
    # ==== headroom, all 29 records ==============================================================
    pk_n, pk_p, rm_n = [], [], []
    for fn in sorted(os.listdir(TWIN)):
        if not fn.endswith('.mat') or not os.path.isfile(os.path.join(PROD, fn)):
            continue
        n = ld(os.path.join(TWIN, fn), ['u_total']); p = ld(os.path.join(PROD, fn), ['u_total'])
        pk_n.append(np.abs(n['u_total']).max(0)); pk_p.append(np.abs(p['u_total']).max(0))
        rm_n.append(np.sqrt((n['u_total'] ** 2).mean(0)))
    pk_n, pk_p, rm_n = map(np.array, (pk_n, pk_p, rm_n))
    summ['headroom'] = dict(n_records=len(pk_n), peak_noisy_max_N=pk_n.max(0).tolist(),
                            peak_prod_max_N=pk_p.max(0).tolist(), peak_change_max_N=np.abs(pk_n - pk_p).max(0).tolist(),
                            limit_peak_N=LIM_PEAK.tolist(), rms_noisy_max_N=rm_n.max(0).tolist(),
                            limit_rms_N=LIM_RMS.tolist())
    print(f'[g5] (d) headroom over {len(pk_n)} records: peak |u_total| noisy {np.round(pk_n.max(0), 1)} N vs '
          f'production {np.round(pk_p.max(0), 1)} (limit {LIM_PEAK}); largest per-record peak change '
          f'{np.round(np.abs(pk_n - pk_p).max(0), 2)} N; rms max {np.round(rm_n.max(0), 1)} (limit {LIM_RMS})')
    json.dump(summ, open(os.path.join(OUT, 'g5.json'), 'w'), indent=1)
    np.savez(os.path.join(OUT, 'g5.npz'), fc=fc, band_sim=band_sim, band_an=band_an, lo=lo, hi=hi)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for i in range(3):
        a = ax[i]
        a.fill_between(fc, lo[:, i], hi[:, i], color=spec.COLORS[i], alpha=0.25, lw=0, label='simulated, 95 % CI')
        a.loglog(fc, band_sim[:, i], 'o', color=spec.COLORS[i], ms=3.5, label='simulated (twin - noise-off)')
        a.loglog(fc, band_an[:, i], color=spec.INK, lw=1.2, label='S_sim Phi_v S_sim^H (expected Welch)')
        a.loglog(fc, g1['band_val'][:, i], color='#c3c2b7', lw=1, ls='dashed', label='Telica measured Phi_e')
        a.set(xlabel='band centre [Hz]', ylabel='band-mean PSD [m^2/Hz]', title=f'{AX[i]}: generator noise part of e')
        a.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(OUT, 'g5_check.png'), dpi=110)


if __name__ == '__main__':
    main()
