"""G1 on Telica (JH-004): Cramer-Rao relative sd of the 13 joint coordinates (ten combinations +
three log Coulomb levels) against window length, telica-real pipeline, G4 attempt-2 point, 5 kHz.

    python -u info/g1_telica.py [--nf 125,250,500,1000,2000] [--max-records N]

Bootstraps with the VENDORED telica-real (vendor/telica_real/tr_env.py): its own model_augmentation
copy and packages. Telica logs are read only through the vendored loader, nothing is cached.
Writes outputs/<run>/g1_telica.json (run folder from --out, default outputs/g1_telica).
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'vendor', 'telica_real'))
import tr_env  # noqa: E402  (vendored telica-real bootstrap, JH-004)
from dataclasses import replace  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.autograd.forward_ad as fwAD  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--nf', default='125,250,500,1000,2000')
ap.add_argument('--span', type=int, default=2000)
ap.add_argument('--burn', type=int, default=125)
ap.add_argument('--max-records', type=int, default=None)
ap.add_argument('--chunk-samples', type=int, default=102400)
ap.add_argument('--out', default=None)
args = ap.parse_args()
OUT = args.out or os.path.join(HERE, 'outputs', 'g1_telica')
os.makedirs(OUT, exist_ok=True)

os.environ.pop('TELICA_SMOKE', None)
os.environ.pop('OBC_ARM', None)
from pipeline import telica_augment as ta  # noqa: E402
from pipeline.real_data import load_datasets_real, frozen_norm  # noqa: E402
from pipeline.telica_model import build_model_real  # noqa: E402
from controller.telica_bank import build_closed_loop_telica  # noqa: E402
from friction.karnopp import ReducedGantryFrictionBlockCC  # noqa: E402
from model_augmentation.fit_systems.closed_loop import closed_loop_rollout  # noqa: E402
tr_env.check_no_leak()
assert tr_env.TR.startswith(os.path.join(HERE, 'vendor')), tr_env.TR

SIGMA = np.array([8.69e-9, 10.63e-9, 5.94e-9])     # JH-003 / JH-004: Telica standstill error
THR = (0.01, 0.05)
NA = 29
t_start = time.time()

cfg = replace(ta.make_cfg(), device='cpu', compile_mode=None, checkpoint_chunk=0, fs_new=5000,
              joint_estimation=True, physics_parameterization='reduced', param_prior=False,
              combo_init_detune=None, param_init_detune=None, obc=False, burn_in=0)
print('[g1t] cfg: fs %s Hz, nx_ann %d, na_nb %d, dtype %s' % (cfg.fs_new, cfg.nx_ann, cfg.na_nb,
                                                            cfg.dtype_pt))
assert cfg.na_nb == NA, cfg.na_nb
data = load_datasets_real(cfg, train_iters=None, val_iters=('iter0',), test_iters=('iter0',),
                          max_records=args.max_records)
norm = frozen_norm(cfg)
torch.manual_seed(cfg.seed)
fs = build_model_real(cfg.hp, cfg, data, norm, baseline_tag='_a2')
fs.simulator = build_closed_loop_telica(fs, norm, cfg, train_files=data.train_names,
                                        val_files=data.val_names, val_data=data.val_ckpt_data,
                                        verbose=True)
fs.eval()
bank = fs.simulator.bank
phy = next(b for b in fs.hfn.connected_blocks if isinstance(b, ReducedGantryFrictionBlockCC))
NAMES = list(phy.COMBO_NAMES) + ['cc1', 'cc2', 'ccy']
NP = phy.free_params.numel()
assert NP == 13, NP
free0 = phy.free_params.detach().clone()
tr_env.check_no_leak()
print('[g1t] block %s, %d free coordinates at the G4 point; %d train records'
      % (type(phy).__name__, NP, len(data.train_list)))

u0, us = np.ravel(fs.norm.u0), np.ravel(fs.norm.ustd)
y0, ys = np.ravel(fs.norm.y0), np.ravel(fs.norm.ystd)
recs = [((np.asarray(sd.u, float) - u0) / us, (np.asarray(sd.y, float) - y0) / ys)
        for sd in data.train_list]
lens = np.array([len(r[0]) for r in recs])
keep = [i for i, n in enumerate(lens) if n >= NA + args.span]
print('[g1t] record lengths at 5 kHz: min %d median %d max %d; %d of %d hold [29, 29+%d)'
      % (lens.min(), np.median(lens), lens.max(), len(keep), len(recs), args.span))


def windows(nf):
    UH, YH, U, Y = [], [], [], []
    for i in keep:
        un, yn = recs[i]
        for k in range(NA, NA + args.span, nf):
            UH.append(un[k - NA:k + 1]); YH.append(yn[k - NA:k + 1])
            U.append(un[k:k + nf]); Y.append(yn[k:k + nf])
    T = lambda a: torch.as_tensor(np.stack(a), dtype=torch.float64)  # noqa: E731
    return T(UH), T(YH), T(U), T(Y)


def rollout(mats, x0, u, y):
    ctrl = torch.zeros(x0.shape[0], dtype=torch.long)
    yp, _, _ = closed_loop_rollout(fs.hfn, fs.hfn.output_only, u, y, x0, bank, ctrl,
                                   pass_mats=mats)
    return yp


def sens(x0, u, y):
    """(y (B,nf,3), S (B,nf,3,6+13)) normalised; x0 (B, nx) full encoder state."""
    B = x0.shape[0]
    cols = []
    x0r = x0.repeat(6, 1)
    tan = torch.zeros_like(x0r)
    for k in range(6):
        tan[k * B:(k + 1) * B, k] = 1.0
    with fwAD.dual_level():
        yd = rollout(phy._mats_from_free(free0), fwAD.make_dual(x0r, tan), u.repeat(6, 1, 1),
                     y.repeat(6, 1, 1))
        p, t = fwAD.unpack_dual(yd)
    yprim = p[:B].clone()
    cols += [t[k * B:(k + 1) * B] for k in range(6)]
    for j in range(NP):
        e = torch.zeros(NP, dtype=torch.float64); e[j] = 1.0
        with fwAD.dual_level():
            yd = rollout(phy._mats_from_free(fwAD.make_dual(free0, e)), x0, u, y)
            cols.append(fwAD.unpack_dual(yd).tangent)
    return yprim, torch.stack(cols, -1)


def enc(uh, yh):
    with torch.no_grad():
        return fs.encoder(uh, yh).detach()

# ---------------------------------------------------------------- V2: forward mode vs FD
UH, YH, U, Y = windows(500)
sel = np.linspace(0, len(U) - 1, 4).astype(int)
x0 = enc(UH[sel], YH[sel])
_, S = sens(x0, U[sel], Y[sel])
h, v2 = 1e-6, []
for d in range(6 + NP):
    if d < 6:
        e = torch.zeros_like(x0); e[:, d] = h
        fd = (rollout(phy._mats_from_free(free0), x0 + e, U[sel], Y[sel])
              - rollout(phy._mats_from_free(free0), x0 - e, U[sel], Y[sel])) / (2 * h)
    else:
        e = torch.zeros(NP, dtype=torch.float64); e[d - 6] = h
        fd = (rollout(phy._mats_from_free(free0 + e), x0, U[sel], Y[sel])
              - rollout(phy._mats_from_free(free0 - e), x0, U[sel], Y[sel])) / (2 * h)
    ad = S[..., d]
    v2.append(float((fd - ad).abs().max() / ad.abs().max().clamp_min(1e-300)))
names19 = ['x0_%d' % k for k in range(6)] + NAMES
print('[g1t] V2 forward vs FD max rel diff (tol 1e-4): '
      + '  '.join('%s %.1e' % (n, v) for n, v in zip(names19, v2)),
      'PASS' if max(v2) <= 1e-4 else 'FAIL')

# ---------------------------------------------------------------- information per nf
NF = [int(s) for s in args.nf.split(',')]
ystd = torch.as_tensor(ys)
res = {}
for nf in NF:
    t0 = time.time()
    UH, YH, U, Y = windows(nf)
    B = len(U)
    Bc = max(1, args.chunk_samples // nf)
    F = {(b, x): torch.zeros(NP, NP, dtype=torch.float64)
         for b in (0, args.burn) if b < nf for x in ('free', 'known')}
    for s in range(0, B, Bc):
        ix = slice(s, min(B, s + Bc))
        _, S = sens(enc(UH[ix], YH[ix]), U[ix], Y[ix])
        S = S * ystd[None, None, :, None] / torch.as_tensor(SIGMA)[None, None, :, None]
        for b in (0, args.burn):
            if b >= nf:
                continue
            A = S[:, b:].reshape(S.shape[0], -1, 6 + NP)
            R = torch.linalg.qr(A, mode='r')[1]
            F[(b, 'free')] += torch.einsum('bij,bik->jk', R[:, 6:, 6:], R[:, 6:, 6:])
            F[(b, 'known')] += torch.einsum('bij,bik->jk', A[..., 6:], A[..., 6:])
        del S
    for (b, x), Fv in F.items():
        Fv = 0.5 * (Fv + Fv.T)
        ev = torch.linalg.eigvalsh(Fv)
        C = torch.linalg.pinv(Fv, hermitian=True) if ev.min() <= 0 else torch.linalg.inv(Fv)
        sd = C.diagonal().clamp_min(0).sqrt()
        res['b%d|x0%s|nf%d' % (b, x, nf)] = dict(
            nf=nf, burn_in=b, x0=x, n_windows=B, sd=dict(zip(NAMES, sd.tolist())),
            cond=float(ev.max() / ev.min()) if ev.min() > 0 else float('inf'),
            spd=bool(ev.min() > 0), seconds=time.time() - t0)
    p = res['b0|x0free|nf%d' % nf]
    print('[g1t] nf %4d (%3.0f ms) %4d windows %6.1f s cond %.2e  sd %%: ' % (
        nf, nf / 5, B, time.time() - t0, p['cond'])
        + ' '.join('%s %.3g' % (n, 100 * p['sd'][n]) for n in NAMES), flush=True)
    json.dump(dict(V2=dict(zip(names19, v2)), V2_pass=max(v2) <= 1e-4, results=res,
                   sigma=SIGMA.tolist(), n_records=len(keep), n_train=len(recs),
                   record_len=dict(min=int(lens.min()), max=int(lens.max()))),
              open(os.path.join(OUT, 'g1_telica.json'), 'w'), indent=1)

for b in (0, args.burn):
    for x in ('free', 'known'):
        print('\n[g1t] TABLE burn_in=%d x0=%s: relative CRB sd [%%], smallest nf < 1 %% / < 5 %%' % (b, x))
        print('  %-8s ' % 'coord' + ' '.join('%9d' % nf for nf in NF) + '   <1%    <5%')
        for n in NAMES:
            sds = {nf: res.get('b%d|x0%s|nf%d' % (b, x, nf), {}).get('sd', {}).get(n) for nf in NF}
            first = [next((nf for nf in NF if sds[nf] is not None and sds[nf] < t), None) for t in THR]
            print('  %-8s ' % n + ' '.join('%9s' % ('-' if sds[nf] is None else '%.3g' % (100 * sds[nf]))
                                            for nf in NF)
                  + '   %-6s %-6s' % (first[0] or 'never', first[1] or 'never'))
print('\n[g1t] done in %.0f s' % (time.time() - t_start))
tr_env.check_no_leak()
