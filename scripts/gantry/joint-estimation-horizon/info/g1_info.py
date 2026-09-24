"""G1 (JH-003): Cramer-Rao relative standard deviation of the ten combinations against window length,
closed-loop windowed output model, production simulated data, at the TRUE combinations.

    python -u info/g1_info.py [--nf 100,200,400,800,1600,3200] [--records all|N] [--checks-only]

Writes outputs/<run>/g1.json and g1_F.npz (run folder from --out, default outputs/g1_sim).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jh_env  # noqa: E402
import jh_model as jm  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--nf', default='100,200,400,800,1600,3200')
ap.add_argument('--records', default='all')
ap.add_argument('--chunk-samples', type=int, default=204800)
ap.add_argument('--checks-only', action='store_true')
ap.add_argument('--out', default=None)
args = ap.parse_args()
OUT = args.out or jh_env.out_dir('g1_sim')
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))

# THEORY/DATA: Telica closed-loop standstill measured error, BHL X1 / X2 / Y (JH-003).
SIGMA = np.array([8.69e-9, 10.63e-9, 5.94e-9])
BURN = 100            # D-178
THR = (0.01, 0.05)    # JH-003: primary 1 %, secondary 5 %
sw = jm.Stopwatch()

ctx = jm.build(combo_init_detune=None)
blk = ctx.block
free0 = torch.zeros(10, dtype=torch.float64)
print('[g1] mode=%s  %d train records  nf-grid %s  dtype %s' % (ctx.cfg.mode, len(ctx.train_files),
      args.nf, ctx.cfg.dtype_pt))
print('[g1] true combinations:', {k: round(v, 6) for k, v in blk.identifiable_combinations().items()})
print('[g1] ystd [m]:', ctx.ystd)
recs = jm.normalised_records(ctx)
if args.records != 'all':
    recs = recs[:int(args.records)]
print('[g1] records:', [(r[0], len(r[1])) for r in recs])

# ---------------------------------------------------------------- V1: equals production rollout
from model_augmentation.fit_systems.closed_loop import closed_loop_rollout  # noqa: E402
W = jm.windows(recs[:2], 400)
sel = np.linspace(0, len(W['k']) - 1, 8).astype(int)
x0full = jm.encoder_x0(ctx, W['uh'][sel], W['yh'][sel])
u, y, c = jm.T(W['u'][sel]), jm.T(W['y'][sel]), torch.as_tensor(W['ctrl'][sel])
with torch.no_grad():
    y_prod, _, _ = closed_loop_rollout(ctx.fit_sys.hfn, ctx.fit_sys.hfn.output_only, u, y, x0full,
                                       ctx.bank, c, pass_mats=blk.pass_mats())
    y_mine = jm.rollout(ctx, jm.mats_of(blk, free0), x0full[:, :6], u, y, c)
v1 = float((y_prod - y_mine).abs().max())
print('[g1] V1 max|y_prod - y_ann_free| = %.3e (normalised, i.e. fraction of ystd; tol 1e-12) %s'
      % (v1, 'PASS' if v1 <= 1e-12 else 'FAIL'))

# ---------------------------------------------------------------- V2: forward mode vs central FD
sel4 = sel[:4]
x0 = x0full[:4, :6]
u4, y4, c4 = u[:4], y[:4], c[:4]
yp, S = jm.sensitivities(ctx, free0, x0, u4, y4, c4, want_x0=True)
# JH-008 (attempt 2): a step ladder, because a single h = 1e-6 was rounding-limited on the small
# damping/stiffness sensitivities. Pass = min over the ladder <= 1e-4 for every direction.
HS = (1e-3, 1e-4, 1e-5, 1e-6)
names16 = ['x0_%d' % k for k in range(6)] + jm.COMBO
v2 = {h: [] for h in HS}
for h in HS:
    for d in range(16):
        if d < 6:
            e = torch.zeros(6, dtype=torch.float64); e[d] = h
            yplus = jm.rollout(ctx, jm.mats_of(blk, free0), x0 + e, u4, y4, c4)
            yminus = jm.rollout(ctx, jm.mats_of(blk, free0), x0 - e, u4, y4, c4)
        else:
            e = torch.zeros(10, dtype=torch.float64); e[d - 6] = h
            yplus = jm.rollout(ctx, jm.mats_of(blk, free0 + e), x0, u4, y4, c4)
            yminus = jm.rollout(ctx, jm.mats_of(blk, free0 - e), x0, u4, y4, c4)
        fd = (yplus - yminus) / (2 * h)
        ad = S[..., d]
        v2[h].append(float((fd - ad).abs().max() / ad.abs().max().clamp_min(1e-300)))
amax = [float(S[..., d].abs().max()) for d in range(16)]
best = [min(v2[h][d] for h in HS) for d in range(16)]
print('[g1] V2 forward mode vs central FD, max rel diff per direction and step (tol 1e-4 on the best):')
print('     %-8s %10s ' % ('dir', 'max|S|') + ' '.join('%9s' % ('h=%.0e' % h) for h in HS))
for d, n in enumerate(names16):
    print('     %-8s %10.3e ' % (n, amax[d]) + ' '.join('%9.1e' % v2[h][d] for h in HS))
print('     best: ' + '  '.join('%s %.1e' % (n, v) for n, v in zip(names16, best)),
      'PASS' if max(best) <= 1e-4 else 'FAIL')
checks = dict(V1=v1, V1_pass=v1 <= 1e-12, V2={('%.0e' % h): dict(zip(names16, v2[h])) for h in HS},
              V2_best=dict(zip(names16, best)), V2_maxS=dict(zip(names16, amax)),
              V2_pass=max(best) <= 1e-4)
json.dump(dict(checks=checks), open(os.path.join(OUT, 'g1_checks.json'), 'w'), indent=1)
if args.checks_only:
    print('[g1] checks only, done in %.0f s' % sw())
    sys.exit(0)


# ---------------------------------------------------------------- coloured noise (secondary)
def coloured_autocorr(nlag):
    """R(m) at 4 kHz lags m = 0..nlag-1 per stage channel, from the pooled Telica error PSD."""
    z = np.load(os.path.join(jh_env.REPO, 'scripts', 'gantry', 'closed-loop-noise', 'outputs',
                             'g1_spectra', 'g1.npz'), allow_pickle=True)
    f, P = np.asarray(z['f'], float), np.asarray(z['Ppool'], float)
    if P.shape[0] != len(f):
        P = P.T
    assert P.shape == (len(f), 3), P.shape
    df = f[1] - f[0]
    L = int(round(20000.0 / df))                      # Welch segment length at 20 kHz
    m20 = np.arange(0, 5 * nlag, 5)                   # every fifth 20 kHz lag = 4 kHz lags
    tau = m20 / 20000.0
    wts = np.full(len(f), df); wts[0] = wts[-1] = df / 2   # trapezoid on the one-sided grid
    R = np.cos(2 * np.pi * np.outer(tau, f)) @ (P * wts[:, None])          # (nlag, 3)
    R *= np.clip(1 - m20 / L, 0, None)[:, None]                            # Bartlett taper
    return R, dict(df=df, L=L, rms_nm=(np.sqrt(R[0]) * 1e9).tolist())


NF = [int(s) for s in args.nf.split(',')]
Rcol, cinfo = coloured_autocorr(max(NF))
v4 = np.abs(np.sqrt(Rcol[0]) * 1e9 / np.array([9.848, 10.400, 6.338]) - 1)
print('[g1] coloured noise: Welch df %.3f Hz, segment %d, R(0) rms %s nm; V4 rel dev %s %s'
      % (cinfo['df'], cinfo['L'], np.round(cinfo['rms_nm'], 3), np.round(v4, 4),
         'PASS' if v4.max() <= 0.02 else 'FAIL'))
checks.update(V4=v4.tolist(), V4_pass=bool(v4.max() <= 0.02), coloured=cinfo)


def coloured_whitened(Sb):
    """Per channel L^-1 S with L the Cholesky factor of the n x n Toeplitz noise covariance of the
    scored samples; channels stacked. Built per call and freed (memory, not speed, is scarce)."""
    from scipy.linalg import toeplitz
    n = Sb.shape[1]
    out = []
    for ch in range(3):
        L = torch.linalg.cholesky(torch.from_numpy(toeplitz(Rcol[:n, ch])))
        out.append(torch.linalg.solve_triangular(L, Sb[:, :, ch, :], upper=False))
        del L
    return torch.cat(out, dim=1)


# ---------------------------------------------------------------- the information per nf
VARIANTS = [(nz, b, x) for nz in ('white', 'coloured') for b in (0, BURN) for x in ('free', 'known')]
res = {}
Fsave = {}
ystd = torch.as_tensor(ctx.ystd)
for nf in NF:
    t_nf = time.time()
    W = jm.windows(recs, nf)
    B = len(W['k'])
    Bc = max(1, args.chunk_samples // nf)
    F = {v: torch.zeros(10, 10, dtype=torch.float64) for v in VARIANTS if not (v[1] >= nf)}
    minRxx = np.inf
    for s in range(0, B, Bc):
        ix = slice(s, min(B, s + Bc))
        x0 = jm.encoder_x0(ctx, W['uh'][ix], W['yh'][ix])[:, :6]
        _, S = jm.sensitivities(ctx, free0, x0, jm.T(W['u'][ix]), jm.T(W['y'][ix]),
                                torch.as_tensor(W['ctrl'][ix]), want_x0=True)
        S = S * ystd[None, None, :, None]                    # metres per unit direction
        for b in (0, BURN):
            if b >= nf:
                continue
            Sb = S[:, b:]                                    # (Bc, n, 3, 16)
            n = Sb.shape[1]
            for nz in ('white', 'coloured'):
                if nz == 'white':
                    A = (Sb / torch.as_tensor(SIGMA)[None, None, :, None]).reshape(Sb.shape[0], -1, 16)
                else:
                    A = coloured_whitened(Sb)
                R = torch.linalg.qr(A, mode='r')[1]
                F[(nz, b, 'free')] += torch.einsum('bij,bik->jk', R[:, 6:, 6:], R[:, 6:, 6:])
                At = A[..., 6:]
                F[(nz, b, 'known')] += torch.einsum('bij,bik->jk', At, At)
                d = R[:, :6, :6].diagonal(dim1=1, dim2=2).abs()
                minRxx = min(minRxx, float((d.min(dim=1).values / d.max(dim=1).values).min()))
        del S
    dt = time.time() - t_nf
    for v, Fv in F.items():
        Fv = 0.5 * (Fv + Fv.T)
        ev = torch.linalg.eigvalsh(Fv)
        C = torch.linalg.inv(Fv)
        sd = C.diagonal().sqrt()
        corr = C / torch.outer(sd, sd)
        corr.fill_diagonal_(0)
        k = int(corr.abs().argmax())
        key = '%s|b%d|x0%s|nf%d' % (v[0], v[1], v[2], nf)
        res[key] = dict(nf=nf, noise=v[0], burn_in=v[1], x0=v[2], n_windows=B,
                        sd=dict(zip(jm.COMBO, sd.tolist())), cond=float(ev.max() / ev.min()),
                        min_eig=float(ev.min()), spd=bool(ev.min() > 0),
                        max_corr=(jm.COMBO[k // 10], jm.COMBO[k % 10], float(corr.flatten()[k])),
                        seconds=dt, peak_rss_mb=jm.peak_rss_mb())
        Fsave[key] = Fv.numpy()
    p = res['white|b0|x0free|nf%d' % nf]
    print('[g1] nf %4d (%5.0f ms)  %5d windows  %6.1f s  peak RSS %.0f MB  cond %.2e  '
          'min Rxx ratio %.1e' % (nf, nf / 4, B, dt, jm.peak_rss_mb(), p['cond'], minRxx))
    print('      sd %% (white, x0 free, b0): ' + '  '.join('%s %.3g' % (n, 100 * p['sd'][n])
                                                       for n in jm.COMBO))
    json.dump(dict(checks=checks, results=res, sigma=SIGMA.tolist(), thresholds=THR,
                   mode=ctx.cfg.mode, records=[r[0] for r in recs]),
              open(os.path.join(OUT, 'g1.json'), 'w'), indent=1)
    np.savez(os.path.join(OUT, 'g1_F.npz'), **{k.replace('|', '__'): v for k, v in Fsave.items()})


# ---------------------------------------------------------------- the PASS tables
def table(noise, b, x0):
    rows = []
    for n in jm.COMBO:
        sds = {nf: res.get('%s|b%d|x0%s|nf%d' % (noise, b, x0, nf), {}).get('sd', {}).get(n)
               for nf in NF}
        first = {thr: next((nf for nf in NF if sds[nf] is not None and sds[nf] < thr), None)
                 for thr in THR}
        rows.append((n, sds, first))
    return rows


for (nz, b, x) in VARIANTS:
    print('\n[g1] TABLE noise=%s burn_in=%d x0=%s: relative CRB sd [%%] per nf, smallest nf < 1 %% / < 5 %%'
          % (nz, b, x))
    print('  %-8s ' % 'combo' + ' '.join('%9d' % nf for nf in NF) + '   <1%    <5%')
    for n, sds, first in table(nz, b, x):
        print('  %-8s ' % n + ' '.join('%9s' % ('-' if sds[nf] is None else '%.3g' % (100 * sds[nf]))
                                        for nf in NF)
              + '   %-6s %-6s' % (first[0.01] or 'never', first[0.05] or 'never'))
print('\n[g1] done in %.0f s' % sw())
jh_env.check_no_leak()
