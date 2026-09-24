"""G1 part 1: attribute the encoder's physical-state error to operating point, model, window (ET-003).

No training, no rollout. Four maps (P0 production, A own-Y, B planted model, AB both) applied to
the same windows of T1-T5 and V1-V4, scored against the exact 8-state truth (D-197), plus a window
sweep na in {15, 29, 59, 119} for P0 and AB. Output: outputs/g1/g1_state.json and a printed table.
"""
__project_origin__ = "added"

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import et_paths                                                  # noqa: E402
sys.path.insert(0, os.path.join(et_paths.ET, 'encoder'))

import enc_common as ec                                          # noqa: E402
import truth                                                     # noqa: E402
import recon                                                     # noqa: E402

RECORDS = ['T1_standstill_Ym30', 'T2_standstill_Ym15', 'T3_standstill_Y000',
           'T4_standstill_Yp15', 'T5_standstill_Yp30',
           'V1_standstill_Yp10', 'V2_aprbs_Ylow', 'V3_ysweep_Yp10', 'V4_lissajous_Ym10']
STAND = RECORDS[:5]
WINDOWS = [15, 29, 59, 119]
STRIDE = 5                     # HEURISTIC (ET-003): memory; >= 9000 windows per record
CH8 = ec.CHANNELS + ['da', 'vda']
OUT = os.path.join(et_paths.OUT, 'g1')


def stds(norm, cfg):
    """The stds normalize_linear_ss_matrices uses, plus the exact absorber pair's training std."""
    from gantry_dynamic.data import TRAIN_FILES
    sx6 = np.asarray(norm.x_all, float).std(axis=0)
    su = np.asarray(norm.u_all, float).std(axis=0)
    sy = np.asarray(norm.y_all, float).std(axis=0)
    xa = np.concatenate([truth.exact_truth(f[:-4], cfg.mode)['x8'][:, [3, 7]] for f in TRAIN_FILES])
    return sx6, su, sy, xa.std(axis=0)


def windows(name, cfg, n, stride=STRIDE):
    """Physical windows (production data path) and the exact state at k, MODEL order (8)."""
    from gantry_dynamic.data import load_traj
    sd = load_traj(name + '.mat', cfg)
    u, y = np.ascontiguousarray(sd.u, dtype=float), np.ascontiguousarray(sd.y, dtype=float)
    x8 = truth.exact_truth(name, cfg.mode)['x8'][:, recon.T2M]
    N = len(y)
    k = np.arange(n + 1, N - 1, stride)
    yw = np.lib.stride_tricks.sliding_window_view(y, n + 1, axis=0)[k - n].transpose(0, 2, 1)
    uw = np.lib.stride_tricks.sliding_window_view(u, n + 1, axis=0)[k - n].transpose(0, 2, 1)
    return np.ascontiguousarray(yw), np.ascontiguousarray(uw), x8[k], y[k, 2], k


def consistency_gate(dt):
    """The planted linearisation with a vanishing absorber must reduce to the baseline map."""
    p = truth.Plant8(ma_frac=1e-12, fa=150.0, zeta_a=0.03, L0=0.10)
    worst = 0.0
    for Y in (-0.36, -0.1, 0.0, 0.2, 0.36):
        Ab, Bb, _, _ = recon.baseline_dt(dt, Y)
        Ap, Bp, _, _ = recon.planted_dt(dt, Y, plant=p)
        ra = np.abs(Ap[:6, :6] - Ab).max() / np.abs(Ab).max()
        rb = np.abs(Bp[:6] - Bb).max() / np.abs(Bb).max()
        worst = max(worst, ra, rb)
    print(f'  consistency gate: planted(ma->0) vs baseline, worst rel diff over 5 Y = {worst:.2e}')
    return worst


def main():
    t0 = time.time()
    et_paths.check()
    cfg, data, norm, fit_sys, hp, dims = ec.build()
    dt = cfg.ts_new
    sx6, su, sy, sa = stds(norm, cfg)
    sx8 = np.concatenate([sx6, sa])
    print(f'  mode {cfg.mode}  dt {dt}  production na={dims[0]}  absorber train std '
          f'da {sa[0]:.3e} m  vda {sa[1]:.3e} m/s')
    gate = consistency_gate(dt)

    res = {'records': RECORDS, 'windows': WINDOWS, 'stride': STRIDE, 'consistency_gate': gate,
           'rms': {}, 'meanabsY': {}, 'n_win': {}}

    # Collapse check: P0 in float64 vs the pipeline's own float32 encoder, V1.
    na = dims[0]
    yw, uw, xt, Yk, k = windows('V1_standstill_Yp10', cfg, na)
    P0 = recon.physical_map(*recon.baseline_dt(dt, 0.0), na, sx6, su, sy)
    x_p0 = recon.apply_map(*P0, yw, uw)
    wv = ec.record_windows('V1_standstill_Yp10', cfg, norm, dims, stride=STRIDE)
    fit_sys.eval()
    xm = ec.encoder_forward(fit_sys.encoder, wv['upast'], wv['ypast'], cfg)[:, :6]
    xm = xm * norm.std_x.flatten() + norm.x_mean.flatten()
    assert np.array_equal(wv['k_ix'], k), 'window index mismatch with enc_common'
    d = np.sqrt(((x_p0 - xm) ** 2).mean(0))
    e = np.sqrt(((x_p0 - xt[:, :6]) ** 2).mean(0))
    res['collapse_rel'] = (d / e).tolist()
    print('  collapse P0(float64) vs pipeline encoder(float32), RMS diff / RMS error: '
          + ' '.join(f'{c} {v:.1e}' for c, v in zip(ec.CHANNELS, d / e)))

    for n in WINDOWS:
        t1 = time.time()
        maps = {
            'P0': recon.physical_map(*recon.baseline_dt(dt, 0.0), n, sx6, su, sy),
            'B': recon.physical_map(*recon.planted_dt(dt, 0.0), n, sx8, su, sy),
            'AB': recon.ScheduledMap(recon.planted_dt, dt, n, sx8, su, sy),
        }
        if n == na:
            maps['A'] = recon.ScheduledMap(recon.baseline_dt, dt, n, sx6, su, sy)
            # interpolation check at a mid-node Y against the exact map there
            Ym = 0.1025
            Wex = recon.physical_map(*recon.planted_dt(dt, Ym), n, sx8, su, sy)
            yv, uv, xv, _, _ = windows("V1_standstill_Yp10", cfg, n)
            yw1, uw1 = yv[:2000], uv[:2000]
            xi = maps['AB'].apply(yw1, uw1, np.full(len(yw1), Ym))
            xe = recon.apply_map(*Wex, yw1, uw1)
            xt1 = xv[:2000]
            res['interp_rel'] = (np.sqrt(((xi - xe) ** 2).mean(0))
                                 / np.sqrt(((xe - xt1) ** 2).mean(0))).tolist()
            print('  interpolation check (mid-node, AB vs exact map), diff / error: '
                  + ' '.join(f'{v:.1e}' for v in res['interp_rel']))
        print(f'  na={n}: maps built [{time.time() - t1:.0f} s]')
        for nm in RECORDS:
            yw, uw, xt, Yk, k = windows(nm, cfg, n)
            res['meanabsY'][nm] = float(np.abs(xt[:, 2]).mean())
            res['n_win'][nm] = len(k)
            for v, W in maps.items():
                x = W.apply(yw, uw, Yk) if isinstance(W, recon.ScheduledMap) else recon.apply_map(*W, yw, uw)
                m = x.shape[1]
                err = x - xt[:, :m]
                res['rms'][f'{v}|{n}|{nm}'] = np.sqrt((err ** 2).mean(0)).tolist()
                res['rms'][f'{v}|{n}|{nm}|bias'] = err.mean(0).tolist()
            res['rms'][f'std|{n}|{nm}'] = xt.std(0).tolist()

    # ==== attribution (pooled MS over the nine records, na = production) ====
    def pooled(v, n):
        ms = np.array([np.asarray(res['rms'][f'{v}|{n}|{r}'][:6]) ** 2 * res['n_win'][r]
                       for r in RECORDS]).sum(0)
        return ms / sum(res['n_win'][r] for r in RECORDS)

    MS = {v: pooled(v, na) for v in ('P0', 'A', 'B', 'AB')}
    op = ((MS['P0'] - MS['A']) + (MS['B'] - MS['AB'])) / 2 / MS['P0']
    mm = ((MS['P0'] - MS['B']) + (MS['A'] - MS['AB'])) / 2 / MS['P0']
    rem = MS['AB'] / MS['P0']
    ab_n = {n: pooled('AB', n) for n in WINDOWS}
    best = np.min(np.stack([ab_n[n] for n in WINDOWS]), axis=0)
    best_n = [WINDOWS[i] for i in np.argmin(np.stack([ab_n[n] for n in WINDOWS]), axis=0)]
    win = (MS['AB'] - best) / MS['P0']

    def corr(v, recs):
        a = np.array([res['rms'][f'{v}|{na}|{r}'][:6] for r in recs])
        b = np.array([res['meanabsY'][r] for r in recs])
        return [float(np.corrcoef(a[:, i], b)[0, 1]) for i in range(6)]

    res['attribution'] = dict(op=op.tolist(), model=mm.tolist(), remainder=rem.tolist(),
                              window=win.tolist(), best_window=best_n,
                              MS={v: MS[v].tolist() for v in MS},
                              MS_AB_by_window={n: ab_n[n].tolist() for n in WINDOWS},
                              MS_P0_by_window={n: pooled('P0', n).tolist() for n in WINDOWS})
    res['corr'] = {f'{v}|{s}': corr(v, recs) for v in ('P0', 'A', 'B', 'AB')
                   for s, recs in (('T1-T5', STAND), ('all9', RECORDS))}

    print(f'\n  pooled RMS error over 9 records, na={na} [phys units]')
    print(f'    {"ch":<7}' + ''.join(f'{v:>11}' for v in ('P0', 'A', 'B', 'AB'))
          + f'{"op":>7}{"model":>7}{"rem":>7}{"window":>8}{"best na":>8}')
    for i, c in enumerate(ec.CHANNELS):
        print(f'    {c:<7}' + ''.join(f'{np.sqrt(MS[v][i]):>11.3e}' for v in ('P0', 'A', 'B', 'AB'))
              + f'{op[i]:>7.2f}{mm[i]:>7.2f}{rem[i]:>7.3f}{win[i]:>8.3f}{best_n[i]:>8d}')
    print('\n  corr(record RMS, mean|Y|)')
    for key, v in res['corr'].items():
        print(f'    {key:<10}' + ' '.join(f'{c} {x:+.3f}' for c, x in zip(ec.CHANNELS, v)))
    print('\n  per-record RMS, na=%d: P0 / A / B / AB' % na)
    for r in RECORDS:
        print(f'    {r:<22} |Y| {res["meanabsY"][r]:.3f}')
        for i, c in enumerate(ec.CHANNELS):
            vals = [res['rms'][f'{v}|{na}|{r}'][i] for v in ('P0', 'A', 'B', 'AB')]
            print(f'      {c:<7}' + ''.join(f'{x:>11.3e}' for x in vals)
                  + f'   std {res["rms"][f"std|{na}|{r}"][i]:.3e}')
        la = [res['rms'][f'{v}|{na}|{r}'][6:] for v in ('B', 'AB')]
        print(f'      da/vda B {la[0][0]:.3e} {la[0][1]:.3e}  AB {la[1][0]:.3e} {la[1][1]:.3e}   '
              f'std {res["rms"][f"std|{na}|{r}"][6]:.3e} {res["rms"][f"std|{na}|{r}"][7]:.3e}')
    print('\n  window sweep, pooled RMS [P0 | AB]')
    for n in WINDOWS:
        p, a = np.sqrt(pooled('P0', n)), np.sqrt(ab_n[n])
        print(f'    na={n:<4}' + ' '.join(f'{c} {x:.2e}|{z:.2e}' for c, x, z in zip(ec.CHANNELS, p, a)))

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'g1_state.json'), 'w') as f:
        json.dump(res, f, indent=1)
    print(f'\n  saved {os.path.join(OUT, "g1_state.json")}   [{time.time() - t0:.0f} s]')


if __name__ == '__main__':
    main()
