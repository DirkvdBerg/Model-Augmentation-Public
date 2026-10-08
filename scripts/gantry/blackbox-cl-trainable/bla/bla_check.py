"""BLA check (DECISIONS.md BB2-001 to BB2-003): can a proper best linear approximation of the plant
(stage force u_total -> stage position, 3 x 3) be estimated from the 18 thesis TRAINING records,
noise-free and noisy?

Estimation uses the training records and the known controller K1 only. The frictionless T_AF
truth enters ONLY as the scoring reference (user 2026-09-29: compare with the real model, never
feed it to the black box). Nothing here is imported by any black-box code.

Run: python -u bla/bla_check.py   (writes bla/outputs/bla_check.npz and bla_check.json)
"""
__project_origin__ = "added"

import csv
import json
import os
import sys

import numpy as np
from scipy.io import loadmat
from scipy.signal import lfilter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
DATA = os.path.join(REPO, 'Thesis-writeup', 'Data')
VERSIONS = {'noisefree': 'Coulomb-tanh-and-MSD-noise-free', 'noisy': 'Coulomb-tanh-and-MSD'}
OUT = os.path.join(HERE, 'outputs')

sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry'))
from gantry_dynamic.controller import build_cfb_at            # noqa: E402  K1, read only
sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry', 'excitation-closed-loop'))
import common as TRUTH                                           # noqa: E402  scoring reference only

TS = 5e-5
NP = 40000                     # 2 s multisine period at 20 kHz (DATA-DESIGN B_tr)
LPM_N, LPM_R = 7, 2            # BB2-002: window 2n+1 = 15 lines, degree 2
KAPPA_MIN = 1e-2               # BB2-003
LOW_SEG = 20000                # BB2-002: 1 s Hann segments, 50 % overlap -> 1 Hz bins
LOW_BINS = np.arange(1, 106)   # 1 .. 105 Hz
Y_GRID = np.round(np.arange(-0.30, 0.3001, 0.05), 3)


def training_records():
    with open(os.path.join(DATA, VERSIONS['noisy'], 'MANIFEST.csv'), newline='') as fh:
        rows = [r for r in csv.DictReader(fh) if r['block'] == 'training']
    assert len(rows) == 18, len(rows)
    return rows


def load(version, row):
    p = os.path.join(DATA, VERSIONS[version], row['folder'], row['file'] + '.mat')
    d = loadmat(p, mat_dtype=True, simplify_cells=True)
    return {k: d[k] for k in ('u_total', 'u_fb', 'f_sim', 'y', 'y_true', 'r_sim', 'meta')}


# K1
CFB, _ = build_cfb_at(0.0, TS)                                   # K1: designed once at Y = 0


def cfb_time(e):
    return np.stack([lfilter(b, a, e[:, j]) for j, (b, a) in enumerate(CFB)], axis=1)


def cfb_freq(f):
    z = np.exp(1j * 2 * np.pi * np.asarray(f) * TS)
    return np.stack([np.polyval(b, z) / np.polyval(a, z) for b, a in CFB], axis=-1)   # (nf, 3) diag


# truth (scoring only)
def check_truth_params(meta):
    t = meta['truth']
    raw = TRUTH.RAW0
    for k in ('mb', 'm1', 'm2', 'Jb', 'Jh', 'cg1', 'cg2', 'cy', 'cb1', 'cb2', 'kb1', 'kb2', 'd'):
        assert abs(float(t[k]) - raw[k]) <= 1e-9 * max(1, abs(raw[k])), (k, t[k], raw[k])
    for k, v in (('ma', TRUTH.MA), ('mh_rigid', TRUTH.MH_R), ('L0', TRUTH.L0), ('fa', TRUTH.FA),
                 ('zeta_a', TRUTH.ZETA_A), ('ka', TRUTH.KA), ('ca', TRUTH.CA)):
        assert abs(float(t[k]) - v) <= 1e-6 * max(1, abs(v)), (k, t[k], v)


def g_true(Y, f):
    return TRUTH.frf_zoh(*TRUTH.truth_mck(Y), f)                  # (nf, 3, 3) stage u -> stage q


def rel_err(Gh, Gt):
    return np.linalg.norm(Gh - Gt, 2, axis=(1, 2)) / np.linalg.norm(Gt, 2, axis=(1, 2))


# conventions (C0)
def c0(d):
    u, ufb, f, y, r = d['u_total'], d['u_fb'], d['f_sim'], d['y'], d['r_sim']
    r = np.broadcast_to(r, y.shape) if np.ndim(r) else np.zeros_like(y)
    a = np.max(np.abs(u - ufb - f)) / np.max(np.abs(u))
    b = np.max(np.abs(cfb_time(r - y) - ufb)) / np.max(np.abs(ufb))
    return a, b


# band B_tr: period DFTs at the multisine lines
def period_spectra(d):
    f = d['f_sim']
    s0 = int(np.argmax(np.any(f != 0, axis=1)))
    nper = (f.shape[0] - s0) // NP
    F0 = np.fft.rfft(f[s0:s0 + NP], axis=0)
    lines = np.where(np.max(np.abs(F0), axis=1) > 1e-6 * np.max(np.abs(F0)))[0]
    per = lambda x, p: np.fft.rfft(x[s0 + p * NP:s0 + (p + 1) * NP], axis=0)[lines]
    Fs = np.stack([per(f, p) for p in range(1, nper)])            # period 0 dropped (transient)
    Ys = np.stack([per(d['y'], p) for p in range(1, nper)])
    Us = np.stack([per(d['u_total'], p) for p in range(1, nper)])
    fper = np.max(np.abs(Fs - Fs[0])) / np.max(np.abs(Fs[0]))    # periodicity of the excitation
    return dict(lines=lines, freq=lines / (NP * TS), F=Fs.mean(0), Y=Ys.mean(0), U=Us.mean(0),
                Yvar=Ys.var(0, ddof=1) / len(Ys), s0=s0, nper=nper, fper=fper)


def lpm(recs, nl, n=LPM_N, R=LPM_R):
    """Local polynomial method, pooled over records. recs: list of (F, Z) with F (nl, 3) inputs and
    Z (nl, 6) outputs [Y; U] at the excited lines. Per line k: Z_rec(k+r) = sum_s Th_s r^s F_rec(k+r)
    + sum_s T_rec,s r^s (transient / leakage per record). Returns Th_0 (nl, 6, 3)."""
    out = np.empty((nl, 6, 3), complex)
    nrec = len(recs)
    for k in range(nl):
        lo = min(max(k - n, 0), nl - 2 * n - 1)
        rr = np.arange(lo, lo + 2 * n + 1) - k
        rows, rhs = [], []
        for i, (F, Z) in enumerate(recs):
            for j, r in enumerate(rr):
                x = np.zeros(3 * (R + 1) + nrec * (R + 1), complex)
                for s in range(R + 1):
                    x[3 * s:3 * s + 3] = F[k + r] * r ** s
                    x[3 * (R + 1) + i * (R + 1) + s] = r ** s
                rows.append(x); rhs.append(Z[k + r])
        A, B = np.array(rows), np.array(rhs)
        sc = np.max(np.abs(A), axis=0); sc[sc == 0] = 1
        th = np.linalg.lstsq(A / sc, B, rcond=None)[0] / sc[:, None]
        out[k] = th[:3].T
    return out


def g_from(Th):
    Syf, Suf = Th[:, :3], Th[:, 3:]
    return Syf @ np.linalg.inv(Suf)                                # indirect: G = S_yf S_uf^-1


def g_direct(recs_yu, n=LPM_N):
    """Direct G = S_yu S_uu^-1 over the same LPM windows (degree 0, pooled): biased in closed loop
    with noise; reported, not used for the verdict."""
    nl = recs_yu[0][0].shape[0]
    out = np.empty((nl, 3, 3), complex)
    for k in range(nl):
        lo = min(max(k - n, 0), nl - 2 * n - 1)
        U = np.concatenate([u[lo:lo + 2 * n + 1] for u, _ in recs_yu])
        Y = np.concatenate([y[lo:lo + 2 * n + 1] for _, y in recs_yu])
        out[k] = np.linalg.lstsq(U, Y, rcond=None)[0].T
    return out


# low band: reference as instrument
def low_spectra(d):
    y, u, f = d['y'], d['u_total'], d['f_sim']
    r = d['r_sim']
    r = np.broadcast_to(r, y.shape).astype(float) if np.ndim(r) else np.zeros_like(y)
    win = np.hanning(LOW_SEG)[:, None]
    C = cfb_freq(LOW_BINS)
    W, Yl, Ul = [], [], []
    for s in range(0, y.shape[0] - LOW_SEG + 1, LOW_SEG // 2):
        sl = slice(s, s + LOW_SEG)
        R = np.fft.rfft(win * (r[sl] - r[sl].mean(0)), axis=0)[LOW_BINS]
        Fq = np.fft.rfft(win * f[sl], axis=0)[LOW_BINS]
        W.append(C * R + Fq)                                        # exogenous force Cfb r + f
        Yl.append(np.fft.rfft(win * (y[sl] - y[sl].mean(0)), axis=0)[LOW_BINS])
        Ul.append(np.fft.rfft(win * (u[sl] - u[sl].mean(0)), axis=0)[LOW_BINS])
    return np.stack(W), np.stack(Yl), np.stack(Ul)                  # (nseg, nbins, 3)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = training_records()
    res, summ = {}, {}
    for ver in VERSIONS:
        print('=== %s ===' % ver, flush=True)
        ms_recs, low = [], []
        for row in rows:
            d = load(ver, row)
            check_truth_params(d['meta'])
            a, b = c0(d)
            yop = float(row['Y_op'])
            print('  %-6s Y_op %+.2f  C0 |u - u_fb - f|/|u| %.1e  |Cfb(r - y) - u_fb|/|u_fb| %.1e'
                  % (row['record'], yop, a, b), flush=True)
            low.append(low_spectra(d))
            if row['excitation'] == 'multisine':
                ps = period_spectra(d)
                print('         multisine start %d, %d periods kept, %d lines %.1f..%.1f Hz, '
                      'excitation periodicity %.1e' % (ps['s0'], ps['nper'] - 1, len(ps['lines']),
                                                      ps['freq'][0], ps['freq'][-1], ps['fper']))
                ps.update(record=row['record'], Y_op=yop, cls=row['class'])
                ms_recs.append(ps)
            del d
        freq = ms_recs[0]['freq']
        assert all(np.array_equal(p['lines'], ms_recs[0]['lines']) for p in ms_recs)
        nl = len(freq)
        # per standstill Y
        for p in ms_recs:
            if p['cls'] != 'standstill':
                continue
            Th = lpm([(p['F'], np.concatenate([p['Y'], p['U']], 1))], nl)
            G = g_from(Th)
            Gd = g_direct([(p['U'], p['Y'])])
            e = rel_err(G, g_true(p['Y_op'], freq))
            ed = rel_err(Gd, g_true(p['Y_op'], freq))
            key = '%s_%s' % (ver, p['record'])
            res['G_' + key], res['Gd_' + key], res['e_' + key] = G, Gd, e
            summ[key] = dict(Y_op=p['Y_op'], med=float(np.median(e)), p90=float(np.percentile(e, 90)),
                             med_direct=float(np.median(ed)), p90_direct=float(np.percentile(ed, 90)))
            print('  %s Y %+.2f  indirect e med %.3f p90 %.3f | direct e med %.3f p90 %.3f'
                  % (p['record'], p['Y_op'], summ[key]['med'], summ[key]['p90'],
                     summ[key]['med_direct'], summ[key]['p90_direct']), flush=True)
        # pooled over all multisine records
        Th = lpm([(p['F'], np.concatenate([p['Y'], p['U']], 1)) for p in ms_recs], nl)
        G = g_from(Th)
        Gd = g_direct([(p['U'], p['Y']) for p in ms_recs])
        E = np.stack([rel_err(G, g_true(Yg, freq)) for Yg in Y_GRID])
        Ed = np.stack([rel_err(Gd, g_true(Yg, freq)) for Yg in Y_GRID])
        e_env, e0 = E.min(0), E[list(Y_GRID).index(0.0)]
        res['G_%s_pooled' % ver], res['Gd_%s_pooled' % ver] = G, Gd
        res['e_%s_pooled_env' % ver], res['e_%s_pooled_Y0' % ver] = e_env, e0
        summ['%s_pooled' % ver] = dict(n_records=len(ms_recs), med_env=float(np.median(e_env)),
                                       p90_env=float(np.percentile(e_env, 90)),
                                       med_Y0=float(np.median(e0)), p90_Y0=float(np.percentile(e0, 90)),
                                       med_direct_env=float(np.median(Ed.min(0))),
                                       p90_direct_env=float(np.percentile(Ed.min(0), 90)))
        print('  pooled (%d records) indirect e vs Y envelope med %.3f p90 %.3f, vs Y 0 med %.3f p90 %.3f'
              ' | direct vs envelope med %.3f p90 %.3f' % tuple(
                  [len(ms_recs)] + [summ['%s_pooled' % ver][k] for k in
                                    ('med_env', 'p90_env', 'med_Y0', 'p90_Y0', 'med_direct_env',
                                     'p90_direct_env')]), flush=True)
        # low band
        W = np.concatenate([w for w, _, _ in low]); Yl = np.concatenate([y for _, y, _ in low])
        Ul = np.concatenate([u for _, _, u in low])
        sv = np.stack([np.linalg.svd(W[:, b, :], compute_uv=False) for b in range(len(LOW_BINS))])
        kappa = sv[:, -1] / sv[:, 0]
        ok = kappa >= KAPPA_MIN
        Gl = np.full((len(LOW_BINS), 3, 3), np.nan, complex)
        for b in range(len(LOW_BINS)):
            Wb = W[:, b, :]
            Syw = Yl[:, b, :].T @ Wb.conj(); Suw = Ul[:, b, :].T @ Wb.conj()
            Gl[b] = Syw @ np.linalg.pinv(Suw)
        El = np.stack([rel_err(Gl, g_true(Yg, LOW_BINS.astype(float))) for Yg in Y_GRID]).min(0)
        res['kappa_%s' % ver], res['sv_%s' % ver], res['Glow_%s' % ver], res['elow_%s' % ver] = kappa, sv, Gl, El
        band = (LOW_BINS >= 20)
        summ['%s_low' % ver] = dict(
            bins_identifiable=int(ok.sum()), bins_total=int(len(LOW_BINS)),
            bins_identifiable_20_105=int(ok[band].sum()), bins_20_105=int(band.sum()),
            kappa_median_20_105=float(np.median(kappa[band])),
            sv_median_20_105=np.median(sv[band], axis=0).tolist(),
            e_median_identifiable=float(np.median(El[ok])) if ok.any() else None,
            e_median_20_105_all=float(np.median(El[band])))
        print('  low band: identifiable bins %d/%d (20..105 Hz: %d/%d), median kappa 20..105 %.1e, '
              'median e (envelope) 20..105 %.3f' % (ok.sum(), len(LOW_BINS), ok[band].sum(), band.sum(),
                                                    np.median(kappa[band]), np.median(El[band])), flush=True)
        res['freq'] = freq
    res['low_bins'] = LOW_BINS
    # verdict (BB2-003)
    v = {}
    for ver in VERSIONS:
        ss = [summ[k] for k in summ if k.startswith(ver + '_TR-S')]
        v['%s_Btr_per_Y' % ver] = all(s['med'] <= 0.10 and s['p90'] <= 0.25 for s in ss)
        pl = summ['%s_pooled' % ver]
        v['%s_Btr_pooled' % ver] = pl['med_env'] <= 0.10 and pl['p90_env'] <= 0.25
        lo = summ['%s_low' % ver]
        v['%s_low_identifiable_20_105' % ver] = lo['bins_identifiable_20_105'] == lo['bins_20_105']
    v['Btr_PASS_both'] = all(v['%s_Btr_per_Y' % x] and v['%s_Btr_pooled' % x] for x in VERSIONS)
    v['low_identifiable_both'] = all(v['%s_low_identifiable_20_105' % x] for x in VERSIONS)
    v['proper_BLA'] = v['Btr_PASS_both'] and v['low_identifiable_both'] and all(
        summ['%s_low' % x]['e_median_20_105_all'] <= 0.10 for x in VERSIONS)
    summ['verdict'] = v
    print('VERDICT', json.dumps(v, indent=1), flush=True)
    np.savez_compressed(os.path.join(OUT, 'bla_check.npz'), **res)
    with open(os.path.join(OUT, 'bla_check.json'), 'w') as fh:
        json.dump(summ, fh, indent=1)
    print('wrote bla/outputs/bla_check.npz, bla_check.json')


if __name__ == '__main__':
    main()
