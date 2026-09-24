"""G4: the encoder on real Telica data, against measured positions and filtered FD velocities (ET-006).

Runs on telica-real's vendored stack (Telica parameters). Records: iter0 of all 15 operating points,
5 kHz (TR-022), cut [motion - 100 ms, end], na = 29. Encoders: P0-T (the telica-real production
encoder, Y_op = 0) and G2-T (Y-scheduled, baseline source, degree by the truth-free ET-006 rule).
Reference: q = P^-T y (raw 20 kHz measured positions), v = P^-T d/dt LP_fc(y); error bound per
channel from the standstill noise and the filter's distortion of the motion spectrum.
Only summaries are printed. Output: outputs/telica/g4/g4_telica.json
"""
__project_origin__ = "added"

import json
import os
import sys
import time

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt, sosfreqz, welch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'vendor_telica'))
import tr_env                                                    # noqa: E402  (vendor_telica first)
sys.path.append(os.path.join(HERE, '..', 'encoder'))             # core.py: pure algebra only
import core                                                      # noqa: E402

import data_index                                                # noqa: E402
import rate                                                      # noqa: E402
from params import telica_params as tp                           # noqa: E402
from baseline.records import iter_records                        # noqa: E402
from pipeline.real_data import frozen_norm, PRE, P               # noqa: E402
from pipeline.telica_model import _synthetic_sysdata             # noqa: E402
from model_augmentation.systems.gantry_linearization import gantry_linearize_and_discretize  # noqa: E402
from model_augmentation.utils.utils import normalize_linear_ss_matrices                      # noqa: E402
from model_augmentation.fit_systems.pre_encoder import linear_encoder_init_aug              # noqa: E402

CH = ['X', 'Theta', 'Y', 'dX', 'dTheta', 'dY']
NA = 29                          # Jan's rule 2 (6 + 8) + 1, the Telica config (ann_settings nx_ann 8)
FC_GRID = [50.0, 100.0, 200.0, 400.0, 800.0, 1600.0]
DEGS = list(range(0, 11))
FS20 = float(tp.FS)
PT_INV = np.linalg.inv(P.T)      # stage -> logical: q = P^-T y
OUT = os.path.join(tr_env.OUTPUTS, 'g4')


class _Cfg:
    dtype_np = np.float64


def build_p0(norm, dt):
    """Exactly telica_model.build_model_real's encoder (float64)."""
    Ad, Bd, Cd, Dd = gantry_linearize_and_discretize(dt=dt)
    Ab, Bb, Cb, Db = normalize_linear_ss_matrices(Ad, Bd, Cd, Dd, _synthetic_sysdata(norm))
    torch.manual_seed(42)
    return linear_encoder_init_aug(
        A=Ab, B=Bb, C=Cb, D=Db, nx=6, nu=3, ny=3, na=NA, nb=NA, nx_aug=8,
        n_nodes_per_layer=24, n_hidden_layers=3, flag_linear_only=False,
        u_mean=norm.u_mean, std_u=norm.std_u, y0=norm.y0, ystd=norm.ystd,
        x_mean=norm.x_mean, std_x=norm.std_x, dtype=torch.float64).to(torch.float64).eval()


def enc_phys(enc, yw, uw, norm):
    un = (uw - norm.u_mean.ravel()) / norm.std_u.ravel()
    yn = (yw - norm.y0.ravel()) / norm.ystd.ravel()
    with torch.no_grad():
        x = enc(torch.as_tensor(np.ascontiguousarray(un)), torch.as_tensor(np.ascontiguousarray(yn))).numpy()
    return x[:, :6] * norm.std_x.ravel() + norm.x_mean.ravel()


def load_all():
    """[(split, op, ypos, dict)] with the 20 kHz and 5 kHz views; only summaries printed."""
    recs = []
    for s, op, it, d in iter_records(splits=('train', 'validation', 'test'), iters=('iter0',),
                                     optional=False, log=print):
        d5 = rate.decimate_record(d)
        recs.append((s, op, data_index.op_xy_mm(op)[1], d, d5))
    return recs


def fd_velocity(y20, fc):
    """P^-T d/dt LP_fc(y), zero-phase Butterworth order 4, central difference at 20 kHz."""
    sos = butter(4, fc, fs=FS20, output='sos')
    return np.gradient(sosfiltfilt(sos, y20, axis=0), 1 / FS20, axis=0) @ PT_INV.T


def bounds(d, fc_list):
    """Per fc: noise part and FD velocity; the PSDs for the pooled distortion term (ET-009);
    position bound (3,). # THEORY: Parseval."""
    y = np.asarray(d['q1'], float)
    m = int(d['motion_idx'])
    st = slice(200, max(400, m - 400))                                   # HEURISTIC margins (ET-006)
    q_st = y[st] @ PT_INV.T
    t = np.arange(len(q_st))
    pos_b = np.array([np.std(q_st[:, j] - np.polyval(np.polyfit(t, q_st[:, j], 1), t)) for j in range(3)])
    v_raw = np.gradient(y, 1 / FS20, axis=0) @ PT_INV.T
    nper = 2048
    f, S_mot = welch(v_raw[m:], fs=FS20, nperseg=min(nper, len(v_raw) - m), axis=0)
    _, S_st = welch(v_raw[st], fs=FS20, nperseg=min(nper, st.stop - st.start), axis=0)
    if len(S_st) != len(S_mot):
        S_st = np.stack([np.interp(f, np.linspace(0, FS20 / 2, len(S_st)), S_st[:, j]) for j in range(3)], 1)
    out = dict(psd=(f, S_mot, S_st))
    for fc in fc_list:
        v = fd_velocity(y, fc)
        out[fc] = dict(noise=v[st].std(axis=0), v=v)
    return pos_b, out


def pooled_distortion(refs, fc_list):
    """ET-009: delta per fc from the motion and standstill PSDs averaged over records."""
    f = refs[0]['psd'][0]
    S_mot = np.mean([r['psd'][1] for r in refs], axis=0)
    S_st = np.mean([r['psd'][2] for r in refs], axis=0)
    S_true = np.clip(S_mot - S_st, 0, None)
    df = f[1] - f[0]
    out = {}
    for fc in fc_list:
        _, h = sosfreqz(butter(4, fc, fs=FS20, output='sos'), worN=f, fs=FS20)
        G = np.abs(h) ** 2                                               # zero-phase gain
        out[fc] = np.sqrt((((1 - G) ** 2)[:, None] * S_true).sum(0) * df)
    return out


def windows5(d5, na):
    """Encoder windows over the cut record at 5 kHz, and the 20 kHz index of each window's k."""
    y, u = np.asarray(d5['q1'], float), np.asarray(d5['u'], float)
    ks0 = max(0, int(d5['motion_idx']) - PRE // rate.D)
    k = np.arange(ks0 + na + 1, len(y))
    yw = np.stack([y[i - na:i + 1] for i in k])
    uw = np.stack([u[i - na:i + 1] for i in k])
    return np.ascontiguousarray(yw), np.ascontiguousarray(uw), k, rate.D * k


def standstill_windows5(d5, na):
    y, u = np.asarray(d5['q1'], float), np.asarray(d5['u'], float)
    m = int(d5['motion_idx'])
    k = np.arange(na + 60, max(na + 61, m - 120))                  # HEURISTIC: clear of the edges
    return (np.ascontiguousarray(np.stack([y[i - na:i + 1] for i in k])),
            np.ascontiguousarray(np.stack([u[i - na:i + 1] for i in k])))


def main():
    t0 = time.time()
    tr_env.check_no_leak()
    import model_augmentation, gantry_dynamic                     # noqa: E401
    for mod in (model_augmentation, gantry_dynamic, core):
        print(f'  [paths] {mod.__name__:<20} {os.path.relpath(mod.__file__, HERE)}')
    norm = frozen_norm(_Cfg)
    for k in ('x_mean', 'std_x', 'std_u', 'u_mean'):
        setattr(norm, k, np.asarray(getattr(norm, k), float))
    dt = float(rate.TS5)
    sx = norm.std_x.ravel(); su = norm.std_u.ravel(); sy = np.asarray(norm.ystd, float)
    print(f'  dt {dt} (5 kHz), na {NA}, window {(NA + 1) * dt * 1e3:.1f} ms')
    recs = load_all()
    print(f'  {len(recs)} iter0 records loaded [{time.time() - t0:.0f} s]')

    # ==== reference bounds and fc choice (before any encoder is scored) ====
    per_fc = {fc: [] for fc in FC_GRID}
    posb, refs = [], []
    for s, op, ypos, d, d5 in recs:
        pb, bo = bounds(d, FC_GRID)
        posb.append(pb)
        refs.append(bo)
    if len({len(r['psd'][0]) for r in refs}) != 1:
        raise RuntimeError('records gave different Welch grids; ET-009 pooling needs one grid')
    dist = pooled_distortion(refs, FC_GRID)
    for bo in refs:
        for fc in FC_GRID:
            bo[fc]['distortion'] = dist[fc]
            bo[fc]['bound'] = np.sqrt(bo[fc]['noise'] ** 2 + dist[fc] ** 2)
            per_fc[fc].append(bo[fc]['bound'])
    pooled_fc = {fc: np.sqrt((np.array(per_fc[fc]) ** 2).mean(0)) for fc in FC_GRID}
    fc_best = [min(FC_GRID, key=lambda fc: pooled_fc[fc][j]) for j in range(3)]
    print('  pooled velocity-reference bound by fc [m/s, rad/s, m/s]:')
    for fc in FC_GRID:
        print(f'    fc {fc:>6.0f} Hz  ' + ' '.join(f'{v:.3e}' for v in pooled_fc[fc]))
    print(f'  chosen fc per velocity channel: {fc_best}')
    pos_bound = np.sqrt((np.array(posb) ** 2).mean(0))
    print(f'  position-reference bound (standstill std, logical): ' + ' '.join(f'{v:.3e}' for v in pos_bound))

    # ==== encoders ====
    p0 = build_p0(norm, dt)
    grid = core.grid_maps(lambda dt_, Y: gantry_linearize_and_discretize(dt=dt_, Y_op=Y), dt, NA, sx, su, sy)
    W0p = np.concatenate(core.pure_map(*gantry_linearize_and_discretize(dt=dt), NA, sx, su, sy), 1)
    train = [(i, r) for i, r in enumerate(recs) if r[0] == 'train']
    tw = [windows5(r[4], NA) for _, r in train]
    x_p0 = [core.numpy_sched(np.zeros((0,)), W0p, yw, uw, sx, su, sy) for yw, uw, _, _ in tw]
    x_gr = [core.grid_apply(grid, yw, uw, yw[:, -1, 2], sx, su, sy) for yw, uw, _, _ in tw]
    gain = np.sqrt(sum(((a - b) ** 2).sum(0) for a, b in zip(x_gr, x_p0)))
    deg_sel, deg_tab = None, {}
    for dg in DEGS:
        C, W0, rr = core.fit_schedule(grid, dg)
        xp = [core.numpy_sched(C, W0, yw, uw, sx, su, sy) for yw, uw, _, _ in tw]
        miss = np.sqrt(sum(((a - b) ** 2).sum(0) for a, b in zip(xp, x_gr)))
        ratio = miss / np.maximum(gain, 1e-300)
        deg_tab[dg] = dict(ratio=ratio.tolist(), resid=rr)
        print(f'  deg {dg:>2}: poly-vs-grid / grid-vs-P0 per channel ' + ' '.join(f'{v:.3f}' for v in ratio)
              + f'   map resid {rr["max_rel_to_W"]:.1e}')
        if deg_sel is None and np.all(ratio <= 0.1):
            deg_sel = dg
    rule_met = deg_sel is not None
    if not rule_met:
        deg_sel = min(DEGS, key=lambda dg: max(deg_tab[dg]['ratio']))
    C, W0, _ = core.fit_schedule(grid, deg_sel)
    import copy
    g2 = core.ScheduledReconEncoder(copy.deepcopy(p0), C, NA, 3, 3, norm.y0, norm.ystd).to(torch.float64).eval()
    print(f'  G2-T degree {deg_sel} (rule met: {rule_met}); scheduling range of the data: '
          f'Y in [{min(r[4]["q1"][:, 2].min() for r in recs):.3f}, {max(r[4]["q1"][:, 2].max() for r in recs):.3f}] m')

    # ==== score ====
    res = dict(fc=fc_best, pooled_fc={str(k): v.tolist() for k, v in pooled_fc.items()},
               pos_bound=pos_bound.tolist(), deg=deg_sel, deg_rule_met=rule_met,
               deg_table={str(k): v for k, v in deg_tab.items()}, records={})
    agg = {e: dict(sq=np.zeros(6), n=0, bsq=np.zeros(6)) for e in ('P0', 'G2')}
    by_y = {}
    noise_floor = {'P0': [], 'G2': []}
    for (s, op, ypos, d, d5), bo, pb in zip(recs, refs, posb):
        yw, uw, k5, k20 = windows5(d5, NA)
        qref = (np.asarray(d['q1'], float) @ PT_INV.T)[k20]
        vref = np.stack([bo[fc_best[j]]['v'][k20, j] for j in range(3)], 1)
        xref = np.concatenate([qref, vref], 1)
        bnd = np.concatenate([pb, [bo[fc_best[j]]['bound'][j] for j in range(3)]])
        rr = dict(split=s, ypos=ypos, n=len(k5), bound=bnd.tolist())
        for e, enc in (('P0', p0), ('G2', g2)):
            x = enc_phys(enc, yw, uw, norm)
            err = x - xref
            ms = (err ** 2).mean(0)
            rr[e] = np.sqrt(ms).tolist()
            agg[e]['sq'] += ms * len(k5); agg[e]['n'] += len(k5); agg[e]['bsq'] += bnd ** 2 * len(k5)
            by_y.setdefault((ypos, e), [np.zeros(6), 0])
            by_y[(ypos, e)][0] += ms * len(k5); by_y[(ypos, e)][1] += len(k5)
            ysw, usw = standstill_windows5(d5, NA)
            noise_floor[e].append(enc_phys(enc, ysw, usw, norm)[:, 3:].std(0))
        rr['Yrange'] = [float(d5['q1'][:, 2].min()), float(d5['q1'][:, 2].max())]
        res['records'][f'{s}/{op}'] = rr
        print(f'  {s:<10} {op:<22} n {len(k5):>5}  P0 ' + ' '.join(f'{v:.2e}' for v in rr['P0'])
              + '  G2 ' + ' '.join(f'{v:.2e}' for v in rr['G2']))
    pooled = {e: np.sqrt(agg[e]['sq'] / agg[e]['n']) for e in agg}
    bpool = np.sqrt(agg['P0']['bsq'] / agg['P0']['n'])
    resolvable = bpool < pooled['P0']
    debiased = {e: np.sqrt(np.clip(pooled[e] ** 2 - bpool ** 2, 0, None)) for e in pooled}
    ylev = sorted({r[2] for r in recs})
    per_y = {str(y): {e: np.sqrt(by_y[(y, e)][0] / by_y[(y, e)][1]).tolist() for e in ('P0', 'G2')}
             for y in ylev}
    worst_y = max(max(np.asarray(per_y[str(y)]['G2'])[resolvable] / np.asarray(per_y[str(y)]['P0'])[resolvable])
                  if resolvable.any() else 0.0 for y in ylev)
    lower = bool(np.all(pooled['G2'][resolvable] < pooled['P0'][resolvable])) if resolvable.any() else False
    res.update(pooled={e: v.tolist() for e, v in pooled.items()}, bound_pooled=bpool.tolist(),
               resolvable=resolvable.tolist(), debiased={e: v.tolist() for e, v in debiased.items()},
               per_ypos=per_y, worst_ypos_ratio=float(worst_y),
               noise_floor={e: np.sqrt((np.array(v) ** 2).mean(0)).tolist() for e, v in noise_floor.items()},
               PASS=bool(resolvable.any() and lower and worst_y <= 1.01))
    print(f'\n  {"ch":<8}{"bound":>11}{"P0-T":>11}{"G2-T":>11}{"G2/P0":>8}{"resolv":>8}{"P0 deb":>11}{"G2 deb":>11}')
    for j, c in enumerate(CH):
        print(f'  {c:<8}{bpool[j]:>11.3e}{pooled["P0"][j]:>11.3e}{pooled["G2"][j]:>11.3e}'
              f'{pooled["G2"][j] / pooled["P0"][j]:>8.4f}{str(bool(resolvable[j])):>8}'
              f'{debiased["P0"][j]:>11.3e}{debiased["G2"][j]:>11.3e}')
    print('\n  per ypos [mm]: G2/P0 per channel')
    for y in ylev:
        a, b = np.asarray(per_y[str(y)]['G2']), np.asarray(per_y[str(y)]['P0'])
        print(f'    ypos {y:>6.0f}  ' + ' '.join(f'{c} {v:.4f}' for c, v in zip(CH, a / b)))
    print(f'  encoder velocity noise floor on standstill windows (std): P0 '
          + ' '.join(f'{v:.2e}' for v in res['noise_floor']['P0']) + '  G2 '
          + ' '.join(f'{v:.2e}' for v in res['noise_floor']['G2']))
    print(f'\n  G4 verdict: resolvable {dict(zip(CH, resolvable.tolist()))}; G2 lower on all resolvable: '
          f'{lower}; worst per-ypos ratio {worst_y:.4f} (<= 1.01) -> {"PASS" if res["PASS"] else "FAIL"}')
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'g4_telica.json'), 'w') as f:
        json.dump(res, f, indent=1, default=float)
    torch.save(dict(g2_state=g2.state_dict(), C=C, W0=W0, deg=deg_sel), os.path.join(OUT, 'g2_telica_encoder.pt'))
    print(f'  saved {os.path.relpath(OUT, HERE)}/g4_telica.json  [{time.time() - t0:.0f} s]')


if __name__ == '__main__':
    main()
