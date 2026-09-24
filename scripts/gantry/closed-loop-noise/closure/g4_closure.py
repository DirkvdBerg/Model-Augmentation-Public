"""G4: closure on the Telica loop. Inject v from H_v at the encoder node; analytic S Phi_v S^H vs
nonlinear time simulation vs the measured standstill spectrum (CN-008).

    python closure/g4_closure.py [reading=tanh] [seconds=16.5] [seeds=2]
Writes outputs/g4_closure_<reading>/{g4.json, g4.npz, g4_closure.png}.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cn_env                                                           # noqa: E402

import numpy as np                                                      # noqa: E402
import torch                                                            # noqa: E402
from scipy import stats                                                 # noqa: E402

from measure import spec                                                # noqa: E402
from model import noise_model as nm                                     # noqa: E402
from sensitivity import loop                                            # noqa: E402
from baseline.cl_sim import P as P_STAGE                                # noqa: E402
cn_env.check_no_leak()

READING = sys.argv[1] if len(sys.argv) > 1 else 'tanh'
SECONDS = float(sys.argv[2]) if len(sys.argv) > 2 else 16.5
SEEDS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
OUT = cn_env.out_dir(f'g4_closure_{READING}')
FS = spec.FS
AX = cn_env.AX
DISCARD = 10000
NFINE = 131072                     # 0.153 Hz grid for the analytic rms
D = torch.float64


def op_of(name):
    return name.split('/')[1]


def model_psd_fft(A, Sig, N):
    """VAR one-sided spectral matrix on the grid k fs / N, k = 0..N/2, via FFT of the AR polynomial."""
    p, m, _ = A.shape
    B = np.zeros((N, m, m)); B[0] = np.eye(m); B[1:p + 1] = -A
    H = np.linalg.inv(np.fft.fft(B, axis=0)[:N // 2 + 1])
    P = 2.0 / FS * H @ Sig[None] @ np.conj(np.transpose(H, (0, 2, 1)))
    P[0] /= 2
    return np.fft.rfftfreq(N, 1 / FS), P


def simulate(positions, V, log=print):
    """Nonlinear closed loop (cl_sim.py's step) with v at the encoder node, zero reference.

    positions (B, 3) stage; V (B, T, 3) noise. Returns e (B, T, 3), u (B, T, 3), xc_std (B, nc),
    vmax (B, 3) max |stage velocity|.
    """
    A, Bc, C, Dc = (torch.as_tensor(m, dtype=D) for m in loop.ctrl_ss())
    blk = loop.block(READING) if READING != 'stuck' else None
    Pt = torch.as_tensor(P_STAGE, dtype=D)
    Bn, T, _ = V.shape
    q0 = torch.as_tensor(np.stack([loop.stage_to_logical(y) for y in positions]), dtype=D)
    x = torch.cat([q0, torch.zeros_like(q0)], 1).unsqueeze(-1)
    xc = torch.zeros(Bn, A.shape[0], dtype=D)
    y0 = x[:, :3, 0] @ Pt
    Vt = torch.as_tensor(V, dtype=D)
    e_out = torch.empty(Bn, T, 3, dtype=D); u_out = torch.empty(Bn, T, 3, dtype=D)
    xc_s = torch.zeros(Bn, A.shape[0], dtype=D); xc_ss = torch.zeros_like(xc_s)
    vmax = torch.zeros(Bn, 3, dtype=D)
    t0 = time.time()
    with torch.no_grad():
        for k in range(T):
            ym = x[:, :3, 0] @ Pt
            e = -((ym - y0) + Vt[:, k])                     # reference = start position
            u = xc @ C.T + e @ Dc.T
            e_out[:, k] = e; u_out[:, k] = u
            if k >= DISCARD:
                xc_s += xc; xc_ss += xc ** 2
                vmax = torch.maximum(vmax, (x[:, 3:, 0] @ Pt).abs())
            xc = xc @ A.T + e @ Bc.T
            x = blk.nonlinear_function(torch.cat([x, u.unsqueeze(-1)], 1))
            if k % 20000 == 0 and k:
                log(f'[g4] step {k}/{T} ({time.time() - t0:.0f} s)')
    n = T - DISCARD
    xc_std = torch.sqrt(torch.clamp(xc_ss / n - (xc_s / n) ** 2, min=0))
    return e_out.numpy(), u_out.numpy(), xc_std.numpy(), vmax.numpy()


def main():
    t0 = time.time()
    g1 = np.load(os.path.join(cn_env.OUTPUTS, 'g1_spectra', 'g1.npz'), allow_pickle=True)
    hv = np.load(os.path.join(cn_env.OUTPUTS, 'g3_source', f'hv_{READING}.npz'))
    A, Sig = hv['A'], hv['Sig']
    f = g1['f']; names = list(g1['names']); keep = g1['keep']; Kr = g1['Kr']; xy = g1['xy']
    band_of, fc, fe = g1['band_of'], g1['fc'], g1['fe']
    nb = len(fc); df = f[1] - f[0]
    ops = sorted(set(op_of(n) for n in names))
    pos = np.stack([xy[[i for i, n in enumerate(names) if op_of(n) == op][0]] for op in ops])
    w = np.array([Kr[[i for i, n in enumerate(names) if op_of(n) == op and keep[i]]].sum()
                  for op in ops], float)
    w /= w.sum()
    inband = (f >= fe[0, 0]) & (f <= fe[-1, 1])
    # ==== measured =======================================================================
    idx = np.where(keep)[0]
    nbins = np.array([(band_of == b).sum() for b in range(nb)])
    meas_bv, meas_lo, meas_hi = g1['band_val'], g1['b_lo'], g1['b_hi']
    rms_meas = np.sqrt((meas_bv * nbins[:, None]).sum(0) * df)
    Pb_r = g1['Pb_r']
    rng_b = np.random.default_rng(7)
    boot = []
    for _ in range(1000):
        s_ = rng_b.choice(idx, len(idx))
        boot.append(np.sqrt(((Pb_r[s_].sum(0) / Kr[s_].sum()) * nbins[:, None]).sum(0) * df))
    h = (np.percentile(boot, 97.5, 0) - np.percentile(boot, 2.5, 0)) / 2 / rms_meas
    # ==== analytic =====================================================================
    ff, PHv = model_psd_fft(A, Sig, NFINE)
    K = loop.frf(*loop.ctrl_ss(), ff)
    I = np.eye(3)[None]
    Pe_an = np.zeros_like(PHv)
    for p_, y in enumerate(pos):
        if READING == 'stuck':
            S = np.repeat(I, len(ff), 0)
        else:
            Ap, Bp, Cp = loop.plant_ss(y, READING)
            G = loop.frf(Ap, Bp, Cp, np.zeros((3, 3)), ff)
            S = np.linalg.inv(I + G @ K)
        Pe_an += w[p_] * spec.shape(S, PHv)
    Aan = spec.autos(Pe_an)
    dff = ff[1] - ff[0]
    band_an = np.stack([Aan[(ff >= f[band_of == b][0] - df / 2) & (ff < f[band_of == b][-1] + df / 2)].mean(0)
                        for b in range(nb)])
    fin = (ff >= fe[0, 0] - df / 2) & (ff < fe[-1, 1] + df / 2)
    rms_an = np.sqrt(Aan[fin].sum(0) * dff)
    print(f'[g4] analytic ({READING}): rms {np.round(rms_an * 1e9, 3)} nm (measured '
          f'{np.round(rms_meas * 1e9, 3)} nm, bootstrap half-width {np.round(h * 100, 2)} %)')
    # ==== time simulation ================================================================
    T = int(SECONDS * FS)
    runs_pos = np.repeat(np.arange(len(ops)), SEEDS)
    Psi = nm.impulse_fft(A)
    print(f'[g4] VAR impulse truncated at {len(Psi)} lags; generating {len(runs_pos)} x {T} samples')
    V = np.stack([nm.generate_fft(A, Sig, T, np.random.default_rng(1000 * int(p_) + s_), Psi=Psi)
                  for p_, s_ in zip(runs_pos, np.tile(np.arange(SEEDS), len(ops)))])
    e, u, xc_std, vmax = simulate(pos[runs_pos], V, log=lambda m: print(m, flush=True))
    NBLK = 10; LBLK = (T - DISCARD) // NBLK                           # CN-008b
    blk_band = np.zeros((len(runs_pos), NBLK, nb, 3)); blk_K = np.zeros((len(runs_pos), NBLK))
    for r in range(len(runs_pos)):
        for b_ in range(NBLK):
            _, Pb_, Kb_ = spec.welch_sum(e[r, DISCARD + b_ * LBLK:DISCARD + (b_ + 1) * LBLK])
            blk_band[r, b_] = spec.band_mean(spec.autos(Pb_), band_of, nb); blk_K[r, b_] = Kb_
    rng_bb = np.random.default_rng(11)
    bb = []
    for _ in range(1000):
        acc = np.zeros((nb, 3))
        for p_ in range(len(ops)):
            B_ = blk_band[runs_pos == p_].reshape(-1, nb, 3); K_ = blk_K[runs_pos == p_].reshape(-1)
            s_ = rng_bb.integers(0, len(K_), len(K_))
            acc += w[p_] * B_[s_].sum(0) / K_[s_].sum()
        bb.append(acc)
    bb_lo, bb_hi = np.percentile(bb, [2.5, 97.5], axis=0)
    np.savez_compressed(os.path.join(OUT, "g4_blocks.npz"), blk_band=blk_band, blk_K=blk_K, runs_pos=runs_pos,
                        w=w, bb_lo=bb_lo, bb_hi=bb_hi)
    Psim = np.zeros((len(f), 3, 3), complex); Ksim = 0
    for p_ in range(len(ops)):
        Pp = np.zeros_like(Psim); Kp = 0
        for r in np.where(runs_pos == p_)[0]:
            _, P_, K_ = spec.welch_sum(e[r, DISCARD:])
            Pp += P_; Kp += K_
        Psim += w[p_] * Pp / Kp; Ksim += Kp
    Asim = spec.autos(Psim)
    band_sim = spec.band_mean(Asim, band_of, nb)
    rms_sim = np.sqrt((band_sim * nbins[:, None]).sum(0) * df)
    nu_b = spec.nu_welch(Ksim) * nbins[:, None] / 1.5
    s_lo, s_hi = spec.chi2_ci(band_sim, nu_b)
    in_meas = ((band_sim >= meas_lo) & (band_sim <= meas_hi)).mean(0)
    an_in_sim = ((band_an >= s_lo) & (band_an <= s_hi)).mean(0)
    rel_meas = rms_sim / rms_meas - 1
    rel_an = rms_sim / rms_an - 1
    ok_a = bool(np.all(in_meas >= 0.85)); ok_b = bool(np.all(np.abs(rel_meas) <= 0.02 + h))
    ok_c = bool(np.all(np.abs(rel_an) <= 0.01) and np.all(an_in_sim >= 0.9))
    stats_ = dict(
        reading=READING, seconds=SECONDS, seeds=SEEDS, K_sim=int(Ksim), filter_order=int(A.shape[0]),
        rms_meas_nm=(rms_meas * 1e9).tolist(), rms_meas_boot_halfwidth=h.tolist(),
        rms_analytic_nm=(rms_an * 1e9).tolist(), rms_sim_nm=(rms_sim * 1e9).tolist(),
        sim_vs_meas_rel=rel_meas.tolist(), sim_vs_analytic_rel=rel_an.tolist(),
        bands_sim_in_meas_ci=in_meas.tolist(), bands_analytic_in_sim_ci=an_in_sim.tolist(),
        v_rms_injected_nm=(V[:, DISCARD:].std(1).mean(0) * 1e9).tolist(),
        u_rms_N=u[:, DISCARD:].std(1).mean(0).tolist(),
        ctrl_state_std_max=float(xc_std.max()), ctrl_state_std_median=float(np.median(xc_std)),
        stage_speed_max_mps=vmax.max(0).tolist(), v0_tanh=1e-3,
        PASS_a=ok_a, PASS_b=ok_b, PASS_c=ok_c, PASS=ok_a and ok_b and ok_c)
    print(f'[g4] simulated rms {np.round(rms_sim * 1e9, 3)} nm: vs measured {np.round(rel_meas * 100, 2)} % '
          f'(tol {np.round((0.02 + h) * 100, 2)} %), vs analytic {np.round(rel_an * 100, 2)} %')
    print(f'[g4] bands: simulated inside measured CI {np.round(in_meas, 3)} (>= 0.85); analytic inside '
          f'simulated CI {np.round(an_in_sim, 3)} (>= 0.9)')
    print(f'[g4] injected v rms {np.round(np.array(stats_["v_rms_injected_nm"]), 2)} nm; force rms '
          f'{np.round(np.array(stats_["u_rms_N"]), 4)} N; controller-state std max {xc_std.max():.3e}; '
          f'max stage speed {np.array2string(vmax.max(0), precision=2)} m/s (v0 1e-3)')
    print(f'[g4] PASS (a) {ok_a} (b) {ok_b} (c) {ok_c} -> {stats_["PASS"]}')
    json.dump(stats_, open(os.path.join(OUT, 'g4.json'), 'w'), indent=1)
    np.savez_compressed(os.path.join(OUT, 'g4.npz'), f=f, fc=fc, band_sim=band_sim, band_an=band_an,
                        meas_bv=meas_bv, meas_lo=meas_lo, meas_hi=meas_hi, s_lo=s_lo, s_hi=s_hi,
                        bb_lo=bb_lo, bb_hi=bb_hi)
    plot(fc, band_sim, band_an, meas_bv, meas_lo, meas_hi)
    print(f'[g4] done in {time.time() - t0:.0f} s')


def plot(fc, bs, ba, bm, lo, hi):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for i in range(3):
        a = ax[i]
        a.fill_between(fc, lo[:, i], hi[:, i], color='#c3c2b7', alpha=0.6, lw=0, label='measured 95 % CI')
        a.loglog(fc, bm[:, i], color=spec.INK, lw=1, label='measured')
        a.loglog(fc, ba[:, i], color=spec.INK, lw=1.2, ls='dashed', label='analytic S Phi_v S^H')
        a.loglog(fc, bs[:, i], 'o', color=spec.COLORS[i], ms=3.5, label='time simulation')
        a.set(xlabel='band centre [Hz]', ylabel='band-mean PSD [m^2/Hz]',
              title=f'{AX[i]}: Telica-loop closure ({READING} reading)')
        a.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(OUT, 'g4_closure.png'), dpi=110)


if __name__ == '__main__':
    main()
