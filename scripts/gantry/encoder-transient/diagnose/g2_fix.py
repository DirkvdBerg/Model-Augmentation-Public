"""G2: the fix in simulation, judged on the exact planted model (ET-004).

1. Degree sweep 0..10 for both sources (baseline, planted): map residual and T1-T5 state error
   against the exact truth, relative to the exact grid-scheduled map. Degree rule of ET-004.
2. Collapse gate: source='baseline', deg=0 vs the pipeline encoder, bitwise, float32 and float64.
3. Module vs numpy check for the chosen planted degree.
4. Closed loop, V1-V4, nf = 400, K = 0 and 100: untrained with its own (baseline) G2 map, planted
   with its own (planted) G2 map, plus decomposition arms.
5. Criterion (c): per-record physical-state error, G2 (planted source) vs P0, T1-T5 and V1-V4.
Output: outputs/g2/<tag>.json
"""
__project_origin__ = "added"

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import et_paths                                                  # noqa: E402
sys.path.insert(0, os.path.join(et_paths.ET, 'encoder'))
import cl_common as cc                                           # noqa: E402
import enc_common as ec                                          # noqa: E402
import recon                                                     # noqa: E402
import sched_encoder as se                                       # noqa: E402
from g1_transient import setup, prod_encoder                     # noqa: E402
from g1_state import windows as state_windows, RECORDS, STAND    # noqa: E402

NF, K = 400, 100
DEGS = list(range(0, 11))
DISC_BURNIN = 19.23          # ET-004: burn-in discrimination re-measured in G1 (R004), W^a random
OUT = os.path.join(et_paths.OUT, 'g2')


def norm_windows(yw, uw, norm, dtype):
    un = (uw - np.asarray(norm.u_mean).ravel()) / np.asarray(norm.std_u).ravel()
    yn = (yw - np.asarray(norm.y0).ravel()) / np.asarray(norm.ystd).ravel()
    return (torch.as_tensor(np.ascontiguousarray(un), dtype=dtype).contiguous(),
            torch.as_tensor(np.ascontiguousarray(yn), dtype=dtype).contiguous())


def module_phys(enc, yw, uw, norm, sa, dtype=cc.DT, batch=8192):
    """Encoder module on physical windows -> physical state (6 or 8 rows)."""
    out = []
    with torch.no_grad():
        for i in range(0, len(yw), batch):
            u, y = norm_windows(yw[i:i + batch], uw[i:i + batch], norm, dtype)
            out.append(enc(u, y).double().numpy())
    x = np.concatenate(out)
    ph = x[:, :6] * np.asarray(norm.std_x).ravel() + np.asarray(norm.x_mean).ravel()
    if x.shape[1] == 8 and sa is not None:
        return np.concatenate([ph, x[:, 6:8] * sa], 1), x
    return ph, x


def numpy_sched(C, W0, yw, uw, sx, su, sy, n):
    """The ScheduledReconEncoder's linear part in numpy (pure frame -> physical)."""
    N = len(yw)
    wp = np.concatenate([(yw / sy).reshape(N, -1), (uw / su).reshape(N, -1)], 1)
    x = wp @ W0.T
    if len(C):
        phi = se.cheb_basis(yw[:, -1, 2] / se.S_SCALE, len(C))
        x = x + np.einsum('bk,kxm,bm->bx', phi, C, wp)
    return x * sx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='g2_attempt1')
    ap.add_argument('--deg-planted', type=int, default=None, help='override the degree rule')
    ap.add_argument('--deg-baseline', type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()
    cfg, norm, fit_sys, hfn, enc_prod, bank, rowmap, planted, sa, dims = setup()
    na, dt, hp = dims[0], cfg.ts_new, cfg.hp
    sx6 = np.asarray(norm.x_all, float).std(0)
    su = np.asarray(norm.u_all, float).std(0)
    sy = np.asarray(norm.y_all, float).std(0)
    sx8 = np.concatenate([sx6, sa])
    res = dict(tag=args.tag, na=na, degs=DEGS)

    # ==== 1. degree sweep ====
    src = {'baseline': (recon.baseline_dt, sx6), 'planted': (recon.planted_dt, sx8)}
    grids = {s: se.grid_maps(b, dt, na, sx, su, sy) for s, (b, sx) in src.items()}
    exact = {s: recon.ScheduledMap(b, dt, na, sx, su, sy) for s, (b, sx) in src.items()}
    print(f'  grids built [{time.time() - t0:.0f} s]')
    Wst = {r: state_windows(r, cfg, na) for r in RECORDS}
    fits, sweep = {}, {}
    for s, (b, sx) in src.items():
        nx = len(sx)
        ms_ex = np.zeros(6); cnt = 0
        for r in STAND:
            yw, uw, xt, Yk, k = Wst[r]
            ms_ex += ((exact[s].apply(yw, uw, Yk)[:, :6] - xt[:, :6]) ** 2).sum(0); cnt += len(k)
        ms_ex /= cnt
        sweep[s] = {}
        for d in DEGS:
            C, W0, rr = se.fit_schedule(grids[s], d)
            fits[(s, d)] = (C, W0)
            ms = np.zeros(6)
            for r in STAND:
                yw, uw, xt, Yk, k = Wst[r]
                ms += ((numpy_sched(C, W0, yw, uw, sx, su, sy, na)[:, :6] - xt[:, :6]) ** 2).sum(0)
            ms /= cnt
            ratio = np.sqrt(ms / ms_ex)
            sweep[s][d] = dict(resid=rr, rms=np.sqrt(ms).tolist(), ratio_to_exact=ratio.tolist(),
                               worst=float(ratio.max()))
            print(f'  {s:<8} deg {d:>2}  map resid max/W {rr["max_rel_to_W"]:.2e}  '
                  f'/corr {rr["rel_to_correction"]:.2e}  worst RMS ratio to exact {ratio.max():.3f}  '
                  + ' '.join(f'{c} {v:.2f}' for c, v in zip(ec.CHANNELS, ratio)))
        sweep[s]['exact_rms'] = np.sqrt(ms_ex).tolist()
    res['sweep'] = {s: {str(k): v for k, v in d.items()} for s, d in sweep.items()}

    def pick(s):
        ok = [d for d in DEGS if sweep[s][d]['worst'] <= 1.10]
        return (min(ok), True) if ok else (min(DEGS, key=lambda d: sweep[s][d]['worst']), False)

    deg_b, okb = pick('baseline')
    deg_p, okp = pick('planted')
    if args.deg_baseline is not None:
        deg_b, okb = args.deg_baseline, 'override'
    if args.deg_planted is not None:
        deg_p, okp = args.deg_planted, 'override'
    res['deg'] = dict(baseline=deg_b, baseline_rule_met=okb, planted=deg_p, planted_rule_met=okp)
    print(f'  degree rule: baseline {deg_b} ({okb}), planted {deg_p} ({okp})')

    # ==== 2. collapse gate ====
    yv, uv, xv, Yv, kv = state_windows('V1_standstill_Yp10', cfg, na, stride=1)
    gate = {}
    for dtp in (torch.float32, torch.float64):
        prod = se.make_parent('baseline', enc_prod, norm, dt, na, sx6, su, sy, sa, dtp, hp)
        var = se.ScheduledReconEncoder(se.make_parent('baseline', enc_prod, norm, dt, na, sx6, su,
                                                      sy, sa, dtp, hp),
                                       np.zeros((0, 14, 1)), na, 3, 3, norm.y0, norm.ystd).to(dtp)
        with torch.no_grad():
            u, y = norm_windows(yv, uv, norm, dtp)
            gate[str(dtp)] = bool(torch.equal(prod(u, y), var(u, y)))
    # float32: a fresh float32 copy of the pipeline encoder as the reference
    res['collapse_bitwise'] = gate
    print(f'  collapse gate (deg 0, baseline source vs pipeline encoder, V1, {len(kv)} windows): {gate}')

    # ==== 3. modules ====
    Cb, W0b = fits[('baseline', deg_b)]
    Cp, W0p = fits[('planted', deg_p)]
    enc_U = se.ScheduledReconEncoder(
        se.make_parent('baseline', enc_prod, norm, dt, na, sx6, su, sy, sa, cc.DT, hp),
        Cb, na, 3, 3, norm.y0, norm.ystd).to(cc.DT).eval()
    enc_PL = se.ScheduledReconEncoder(
        se.make_parent('planted', enc_prod, norm, dt, na, sx6, su, sy, sa, cc.DT, hp),
        Cp, na, 3, 3, norm.y0, norm.ystd).to(cc.DT).eval()
    xm_, _ = module_phys(enc_PL, yv[::5], uv[::5], norm, sa)
    xn_ = numpy_sched(Cp, W0p, yv[::5], uv[::5], sx8, su, sy, na)
    err_ = np.sqrt(((xn_ - xv[::5]) ** 2).mean(0))
    res['module_vs_numpy_rel'] = (np.sqrt(((xm_ - xn_) ** 2).mean(0)) / err_).tolist()
    print('  module vs numpy (planted, V1), RMS diff / RMS error: '
          + ' '.join(f'{v:.1e}' for v in res['module_vs_numpy_rel']))

    # ==== 4. closed loop ====
    arm_names = ['U|P0', 'U|G2', 'PL|P0|random', 'PL|G2', 'PL|B0', 'PL|A|zero', 'PL|A|random',
                 'PL|ABgrid', 'PL|exact']
    errs = {a: [] for a in arm_names}
    for nm in cc.VAL:
        w = cc.windows_cl(nm, cfg, norm, na, NF)
        xe = prod_encoder(enc_prod, w, norm)
        _, xU = module_phys(enc_U, w['yw'], w['uw'], norm, None)
        _, xP = module_phys(enc_PL, w['yw'], w['uw'], norm, sa)
        C0, W00 = fits[('planted', 0)]
        xB0 = numpy_sched(C0, W00, w['yw'], w['uw'], sx8, su, sy, na)
        xA = exact['baseline'].apply(w['yw'], w['uw'], w['Y'])
        xAB = exact['planted'].apply(w['yw'], w['uw'], w['Y'])
        lat = lambda x8: x8[:, 6:8] / sa                                        # noqa: E731
        z2 = np.zeros((len(xe), 2))
        arms = {
            'U|P0': (hfn, xe),
            'U|G2': (hfn, xU),
            'PL|P0|random': (planted, np.concatenate([xe[:, :6], xe[:, 6:8]], 1)),
            'PL|G2': (planted, xP),
            'PL|B0': (planted, np.concatenate([cc.x0_norm(xB0[:, :6], norm), lat(xB0)], 1)),
            'PL|A|zero': (planted, np.concatenate([cc.x0_norm(xA, norm), z2], 1)),
            'PL|A|random': (planted, np.concatenate([cc.x0_norm(xA, norm), xe[:, 6:8]], 1)),
            'PL|ABgrid': (planted, np.concatenate([cc.x0_norm(xAB[:, :6], norm), lat(xAB)], 1)),
            'PL|exact': (planted, (w['x8'] - planted.xm.numpy()) / planted.sx.numpy()),
        }
        for a, (m, x0) in arms.items():
            errs[a].append(cc.rollout_cl(m, x0, w, bank, rowmap[nm], norm.ystd))
        print(f'  {nm}: {len(w["k"])} windows [{time.time() - t0:.0f} s]')
    cl = {a: cc.metrics(errs[a], K) for a in arm_names}
    res['closed_loop'] = cl
    disc = cl['U|G2']['rms0'] / cl['PL|G2']['rms0']
    res['disc'] = dict(official_K0=disc, official_K100=cl['U|G2']['rmsK'] / cl['PL|G2']['rmsK'],
                       vs_UP0_K0=cl['U|P0']['rms0'] / cl['PL|G2']['rms0'],
                       today_K0=cl['U|P0']['rms0'] / cl['PL|P0|random']['rms0'],
                       today_K100=cl['U|P0']['rmsK'] / cl['PL|P0|random']['rmsK'])
    print(f'\n  {"arm":<14}{"RMS[0:400]":>12}{"RMS[100:]":>12}{"share":>8}{"grow":>8}   chan X1/X2/Y')
    for a in arm_names:
        r = cl[a]
        print(f'  {a:<14}{r["rms0"]:>12.4e}{r["rmsK"]:>12.4e}{r["share"]:>8.1%}{r["grow"]:>8.2f}   '
              + ' '.join(f'{v:.3e}' for v in r['chan0']))
    print('  discrimination: ' + '  '.join(f'{k} {v:.3f}x' for k, v in res['disc'].items()))

    # ==== 5. criterion (c): per-record physical-state error, G2 (planted) vs P0 ====
    W0prod = np.concatenate(se.pure_map(*recon.baseline_dt(dt, 0.0), na, sx6, su, sy), 1)
    per = {}
    worst_ratio = 0.0
    pooled = {'P0': np.zeros(6), 'G2': np.zeros(6), 'G2U': np.zeros(6)}
    n_tot = 0
    for r in RECORDS:
        yw, uw, xt, Yk, k = Wst[r]
        xg, _ = module_phys(enc_PL, yw, uw, norm, sa)
        xgu, _ = module_phys(enc_U, yw, uw, norm, None)
        xp0 = numpy_sched(np.zeros((0,)), W0prod, yw, uw, sx6, su, sy, na)
        e_g = np.sqrt(((xg[:, :6] - xt[:, :6]) ** 2).mean(0))
        e_u = np.sqrt(((xgu[:, :6] - xt[:, :6]) ** 2).mean(0))
        e_p = np.sqrt(((xp0 - xt[:, :6]) ** 2).mean(0))
        e_lat = np.sqrt(((xg[:, 6:8] - xt[:, 6:8]) ** 2).mean(0))
        per[r] = dict(G2=e_g.tolist(), G2_baseline_source=e_u.tolist(), P0=e_p.tolist(),
                      G2_latent=e_lat.tolist(), latent_std=xt[:, 6:8].std(0).tolist())
        worst_ratio = max(worst_ratio, float((e_g / e_p).max()))
        pooled['P0'] += e_p ** 2 * len(k); pooled['G2'] += e_g ** 2 * len(k)
        pooled['G2U'] += e_u ** 2 * len(k); n_tot += len(k)
    res['state'] = dict(per_record=per, worst_ratio_G2_over_P0=worst_ratio,
                        pooled={a: np.sqrt(v / n_tot).tolist() for a, v in pooled.items()})
    print('\n  physical-state RMS, pooled over 9 records [P0 | G2 planted | G2 baseline-source]')
    for i, c in enumerate(ec.CHANNELS):
        print(f'    {c:<7}{res["state"]["pooled"]["P0"][i]:>11.3e}{res["state"]["pooled"]["G2"][i]:>11.3e}'
              f'{res["state"]["pooled"]["G2U"][i]:>11.3e}')
    print(f'  worst per-record per-channel ratio G2/P0: {worst_ratio:.4f}')

    # ==== verdict ====
    share = cl['PL|G2']['share']
    va = share <= 0.30
    vb = disc >= 0.85 * DISC_BURNIN
    vc = worst_ratio < 1.0
    res['verdict'] = dict(a_share=share, a_pass=va, b_disc=disc, b_threshold=0.85 * DISC_BURNIN,
                          b_pass=vb, c_worst_ratio=worst_ratio, c_pass=vc,
                          collapse_pass=all(gate.values()), PASS=bool(va and vb and vc and all(gate.values())))
    print(f'\n  G2 verdict: share {share:.1%} (<= 30 %: {va}); discrimination {disc:.2f}x '
          f'(>= {0.85 * DISC_BURNIN:.2f}x: {vb}); state worst ratio {worst_ratio:.4f} (< 1: {vc}); '
          f'collapse {gate} -> {"PASS" if res["verdict"]["PASS"] else "FAIL"}')
    os.makedirs(OUT, exist_ok=True)
    torch.save(dict(enc_U=enc_U.state_dict(), enc_PL=enc_PL.state_dict(), deg_b=deg_b, deg_p=deg_p,
                    Cb=Cb, W0b=W0b, Cp=Cp, W0p=W0p, sa=sa),
               os.path.join(OUT, f'{args.tag}_encoders.pt'))
    with open(os.path.join(OUT, f'{args.tag}.json'), 'w') as f:
        json.dump(res, f, indent=1, default=float)
    print(f'  saved outputs/g2/{args.tag}.json  [{time.time() - t0:.0f} s]')


if __name__ == '__main__':
    main()
