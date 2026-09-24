"""G2 (JH-005): baseline-only parameter fits (no ANN) against the OBC-predicted optimum J+ Delta*.

    python -u staged/g2_staged.py --arm a                 one-step fit; also writes c = (I-P0) r0
    python -u staged/g2_staged.py --arm i  --nf 3200      long windows, unrestricted
    python -u staged/g2_staged.py --arm ii --nf 3200      long windows + orthogonal correction c

Dataset augmentation_ma50_b140-230_a6_z03 (JH_FILESET=obc14), detuned start of the OBC arms,
float64 CPU. Results: outputs/<run>/g2_<arm>_<nf>.json.
"""
import argparse
import json
import os
import sys
import time

os.environ['JH_FILESET'] = 'obc14'          # before any gantry_dynamic import (JH-001 patch 3)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jh_env  # noqa: E402
import jh_model as jm  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--arm', required=True, choices=['a', 'i', 'ii'])
ap.add_argument('--nf', type=int, default=3200)
ap.add_argument('--burn', type=int, default=100)
ap.add_argument('--maxit', type=int, default=30)
ap.add_argument('--chunk-samples', type=int, default=204800)
ap.add_argument('--out', default=None)
args = ap.parse_args()
OUT = args.out or jh_env.out_dir('g2_%s_%d' % (args.arm, args.nf) if args.arm != 'a' else 'g2_a')
C_FILE = os.path.join(jh_env.OUT, 'g2_a', 'c_orth.npz')
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))

MODE = 'augmentation_ma50_b140-230_a6_z03'
DETUNE = [1.10, 1.10, 0.90, 1.10, 0.90, 0.90, 1.10, 1.10, 0.90, 1.10]   # OBC arms, 84032-84037
ROWS = (0, 1, 2, 3, 4, 5)
REF_JSON = os.path.join(jh_env.REPO, 'scripts', 'gantry', 'orthogonal-by-construction',
                        'baseline-and-added-dynamics', 'delta_orthogonality.json')
TOL_EACH, TOL_RMS = 1.2, 0.74           # JH-005
sw = jm.Stopwatch()

ctx = jm.build(mode=MODE, combo_init_detune=DETUNE, epochs=150)
blk = ctx.block
from gantry_dynamic.data import TRAIN_FILES  # noqa: E402
assert len(TRAIN_FILES) == 14 and ctx.cfg.mode == MODE, (len(TRAIN_FILES), ctx.cfg.mode)
from model_augmentation.fit_systems.blocks import (  # noqa: E402
    Parameterized_Gantry_State_Block, Reduced_Gantry_State_Block)
from model_augmentation.systems import gantry_ss as gss  # noqa: E402

# ------------------------------------------------------------------ truth, meter, reference
_raw = torch.stack([getattr(gss, n) for n in Parameterized_Gantry_State_Block.PARAM_NAMES]
                   ).to(torch.float64)
TRUE = Reduced_Gantry_State_Block.combos_of(_raw, float(gss.Lb)).numpy()
SCALE = np.abs(TRUE).copy()
SCALE[7] = 0.5 * float(gss.m1 + gss.m2)                # the meter's m_diff scale (training.py)


def combos(free):
    return blk.combinations_from_free(torch.as_tensor(free, dtype=torch.float64)).detach().numpy()


def pct(free):
    return 100.0 * (combos(free) - TRUE) / SCALE


def pct_own(free):
    return 100.0 * (combos(free) - TRUE) / np.abs(TRUE)


pct0 = pct(np.zeros(10))
expect0 = 100.0 * (TRUE * np.asarray(DETUNE) - TRUE) / SCALE
assert np.abs(pct0 - expect0).max() < 1e-6, (pct0, expect0)
ref = json.load(open(REF_JSON))
prim = next(p for p in ref['predictions'] if p['tag'] == 'J @ theta*, Delta* (stencil)')
REF_PCT = np.asarray(prim['pct'])
assert np.allclose(np.asarray(ref['true_combo']), TRUE, rtol=1e-6), 'truth vector mismatch'
print('[g2] mode %s, %d records, arm %s, nf %d; start combo-err %.3f %%; reference J+Delta* %.3f %%'
      % (MODE, len(TRAIN_FILES), args.arm, args.nf, np.sqrt((pct0 ** 2).mean()), prim['rms']))


def verdict(free, tag, extra=None):
    p = pct(free)
    gap = p - REF_PCT
    rms_gap = float(np.sqrt((gap ** 2).mean()))
    ok = bool(np.abs(gap).max() <= TOL_EACH and rms_gap <= TOL_RMS)
    print('\n[g2] RESULT %s: combo-err %.3f %%, gap to J+Delta* RMS %.3f pp, max %.3f pp -> %s'
          % (tag, np.sqrt((p ** 2).mean()), rms_gap, np.abs(gap).max(),
             'WITHIN tolerance' if ok else 'OUTSIDE tolerance'))
    print('  %-8s %10s %10s %10s %12s' % ('combo', 'fit %', 'J+D* %', 'gap pp', 'own-scale %'))
    po = pct_own(free)
    for j, n in enumerate(jm.COMBO):
        print('  %-8s %+10.3f %+10.3f %+10.3f %+12.3f' % (n, p[j], REF_PCT[j], gap[j], po[j]))
    adm = None
    with torch.no_grad():
        saved = blk.free_params.detach().clone()
        blk.free_params.data.copy_(torch.as_tensor(free))
        adm = blk.admissibility()
        blk.free_params.data.copy_(saved)
    d = dict(tag=tag, free=list(map(float, free)), combos=combos(free).tolist(), pct=p.tolist(),
             pct_own_scale=po.tolist(), combo_err=float(np.sqrt((p ** 2).mean())),
             ref_pct=REF_PCT.tolist(), gap_pp=gap.tolist(), gap_rms_pp=rms_gap,
             gap_max_pp=float(np.abs(gap).max()), within_tolerance=ok, admissibility=adm,
             seconds=sw())
    if extra:
        d.update(extra)
    return d


# ------------------------------------------------------------------ arm (a): one-step
def stencil_next(norm):
    """Truth's next state [q_{k+1}; D4 q_{k+1}] on the reference tuples, normalised, + valid mask.
    Adapted from measure_delta_orthogonality.py::next_states (the 'stencil' variant, D-203)."""
    from gantry_dynamic.obc_gantry import fourth_order_central_difference, _load_record_float64
    from model_augmentation.systems.gantry_ss import P as P_pt
    x_mean = np.asarray(norm.x_mean, float).reshape(1, 6)
    std_x = np.asarray(norm.std_x, float).reshape(1, 6)
    PinvT = np.linalg.inv(P_pt.detach().cpu().numpy().astype(np.float64).T)
    nxt, valid = [], []
    for f in TRAIN_FILES:
        u, y, _x, _a = _load_record_float64(f, ctx.cfg)
        q = y @ PinvT.T
        qd = fourth_order_central_difference(q, ctx.cfg.ts_new)
        k = np.arange(2, len(u) - 2)
        idx = k + 1 - 2
        ok = idx < len(qd)
        nxt.append(np.hstack([q[np.clip(k + 1, 0, len(q) - 1)], qd[np.clip(idx, 0, len(qd) - 1)]]))
        valid.append(ok & (k + 1 < len(q)))
    return (np.concatenate(nxt) - x_mean) / std_x, np.concatenate(valid)


def arm_a():
    from gantry_dynamic.obc_gantry import build_reference_set
    from model_augmentation.fit_systems.obc import build_stacked_sensitivity, evaluate_step_rows
    refset = build_reference_set(ctx.cfg, ctx.norm, nx_ann=ctx.cfg.nx_ann, stride=1, verbose=True)
    Z = refset.Z_step.to(torch.float64)
    xn, valid = stencil_next(ctx.norm)
    assert xn.shape[0] == refset.n == Z.shape[0]
    XN = torch.from_numpy(np.ascontiguousarray(xn)).reshape(-1)
    MASK = torch.from_numpy(np.repeat(valid, 6))
    step = lambda v, z: blk.transition_from_free(v, z)  # noqa: E731
    CH = 16384

    def normal_eq(v):
        JtJ = torch.zeros(10, 10, dtype=torch.float64)
        Jtr = torch.zeros(10, dtype=torch.float64)
        cost = 0.0
        for s in range(0, Z.shape[0], CH):
            Zc = Z[s:s + CH]
            Jc = build_stacked_sensitivity(step, v, Zc, ROWS, chunk=CH, device='cpu')
            fc = evaluate_step_rows(step, v, Zc, ROWS, chunk=CH, device='cpu')
            m = MASK[6 * s:6 * (s + Zc.shape[0])]
            rc = torch.where(m, XN[6 * s:6 * (s + Zc.shape[0])] - fc, torch.zeros_like(fc))
            Jc = Jc * m[:, None]
            JtJ += Jc.T @ Jc
            Jtr += Jc.T @ rc
            cost += float(rc @ rc)
        return JtJ, Jtr, cost

    v = torch.zeros(10, dtype=torch.float64)
    hist = []
    delta0 = None
    for it in range(args.maxit):
        t0 = time.time()
        JtJ, Jtr, cost = normal_eq(v)
        dv = torch.linalg.solve(JtJ, Jtr)
        if it == 0:
            delta0 = dv.clone()
            ev = torch.linalg.eigvalsh(JtJ)
            print('[g2a] cond(J0) = %.4e (reference 8.925e+02 at the detuned start)'
                  % float(torch.sqrt(ev.max() / ev.min())))
        hist.append(dict(it=it, cost=cost, step=float(dv.abs().max()), s=time.time() - t0))
        print('[g2a] it %2d  cost %.8e  |dv|max %.3e  combo-err after %.3f %%  (%.0f s)'
              % (it, cost, float(dv.abs().max()), np.sqrt((pct((v + dv).numpy()) ** 2).mean()),
                 time.time() - t0), flush=True)
        v = v + dv
        if float(dv.abs().max()) < 1e-7:
            break
    res = verdict(v.numpy(), 'a: one-step baseline-only LS', dict(history=hist, iterations=it + 1))
    # the orthogonal correction for arm (ii): c = r0 - J0 delta0 = (I - P0) r0, at the start
    print('[g2a] building c = (I - P0) r0 at the detuned start ...', flush=True)
    v0 = torch.zeros(10, dtype=torch.float64)
    cs = []
    for s in range(0, Z.shape[0], CH):
        Zc = Z[s:s + CH]
        Jc = build_stacked_sensitivity(step, v0, Zc, ROWS, chunk=CH, device='cpu')
        fc = evaluate_step_rows(step, v0, Zc, ROWS, chunk=CH, device='cpu')
        m = MASK[6 * s:6 * (s + Zc.shape[0])]
        rc = torch.where(m, XN[6 * s:6 * (s + Zc.shape[0])] - fc, torch.zeros_like(fc))
        cs.append(torch.where(m, rc - Jc @ delta0, torch.zeros_like(rc)).reshape(-1, 6))
    c = torch.cat(cs).numpy()
    r_rms = float(np.sqrt((c ** 2).mean()))
    print('[g2a] c rms (normalised) %.4e; delta0 combo-err %.3f %% (the one-step linear estimate)'
          % (r_rms, np.sqrt((pct(delta0.numpy()) ** 2).mean())))
    os.makedirs(os.path.dirname(C_FILE), exist_ok=True)
    np.savez(C_FILE, c=c, record_ix=refset.record_ix.numpy(), sample_ix=refset.sample_ix.numpy(),
             delta0=delta0.numpy(), files=np.array(TRAIN_FILES))
    res.update(delta0_pct=pct(delta0.numpy()).tolist(), c_rms=r_rms)
    return res


# ------------------------------------------------------------------ arms (i), (ii): long windows
def arm_long(with_c):
    nf, burn = args.nf, args.burn
    recs = jm.normalised_records(ctx)
    assert [r[0] for r in recs] == list(TRAIN_FILES)
    W = jm.windows(recs, nf)
    B = len(W['k'])
    corr = None
    if with_c:
        z = np.load(C_FILE)
        assert list(z['files']) == list(TRAIN_FILES)
        dense = [np.zeros((len(r[1]), 6)) for r in recs]
        for ri in range(len(recs)):
            sel = z['record_ix'] == ri
            dense[ri][z['sample_ix'][sel]] = z['c'][sel]
        corr = np.stack([dense[ri][k:k + nf] for ri, k in zip(W['rec'], W['k'])])
        print('[g2] correction c: rms %.4e over the windows' % np.sqrt((corr ** 2).mean()))
    U, Y, CT = jm.T(W['u']), jm.T(W['y']), torch.as_tensor(W['ctrl'])
    CR = None if corr is None else jm.T(corr)
    x0 = jm.encoder_x0(ctx, W['uh'], W['yh'])[:, :6].clone()
    Bc = max(1, args.chunk_samples // nf)
    print('[g2] %d windows of nf %d (%.2f s), burn-in %d, chunks of %d windows'
          % (B, nf, nf / 4000, burn, Bc))

    def cost_of(v, x0):
        c = 0.0
        with torch.no_grad():
            for s in range(0, B, Bc):
                ix = slice(s, min(B, s + Bc))
                yp = jm.rollout(ctx, jm.mats_of(blk, v), x0[ix], U[ix], Y[ix], CT[ix],
                                None if CR is None else CR[ix])
                c += float(((Y[ix, burn:] - yp[:, burn:]) ** 2).sum())
        return c

    def linearise(v, x0):
        H = torch.zeros(10, 10, dtype=torch.float64)
        g = torch.zeros(10, dtype=torch.float64)
        Nxx_l, Nxt_l, gx_l = [], [], []
        c = 0.0
        for s in range(0, B, Bc):
            ix = slice(s, min(B, s + Bc))
            yp, S = jm.sensitivities(ctx, v, x0[ix], U[ix], Y[ix], CT[ix], want_x0=True,
                                     corr=None if CR is None else CR[ix])
            r = (Y[ix, burn:] - yp[:, burn:]).reshape(yp.shape[0], -1)
            A = S[:, burn:].reshape(yp.shape[0], -1, 16)
            N = torch.einsum('bij,bik->bjk', A, A)
            gg = torch.einsum('bij,bi->bj', A, r)
            Nxx, Nxt, Ntt = N[:, :6, :6], N[:, :6, 6:], N[:, 6:, 6:]
            X = torch.linalg.solve(Nxx, torch.cat([Nxt, gg[:, :6, None]], dim=2))
            H += (Ntt - Nxt.transpose(1, 2) @ X[:, :, :10]).sum(0)
            g += (gg[:, 6:] - (Nxt.transpose(1, 2) @ X[:, :, 10:]).squeeze(-1)).sum(0)
            Nxx_l.append(Nxx); Nxt_l.append(Nxt); gx_l.append(gg[:, :6])
            c += float((r ** 2).sum())
            del S, A
        return H, g, torch.cat(Nxx_l), torch.cat(Nxt_l), torch.cat(gx_l), c

    v = torch.zeros(10, dtype=torch.float64)
    lam = 1e-3
    hist = []
    cost = None
    for it in range(args.maxit):
        t0 = time.time()
        H, g, Nxx, Nxt, gx, cost = linearise(v, x0)
        accepted = False
        for _ in range(12):
            dv = torch.linalg.solve(H + lam * torch.diag(H.diagonal()), g)
            dx = torch.linalg.solve(Nxx, (gx - (Nxt @ dv)).unsqueeze(-1)).squeeze(-1)
            c_new = cost_of(v + dv, x0 + dx)
            if c_new < cost:
                accepted = True
                break
            lam *= 5.0
        rel = (cost - c_new) / cost if accepted else 0.0
        if accepted:
            v, x0 = v + dv, x0 + dx
            lam = max(lam / 3.0, 1e-9)
        p = pct(v.numpy())
        hist.append(dict(it=it, cost=cost, cost_new=c_new, accepted=accepted, lam=lam,
                         step=float(dv.abs().max()), combo_err=float(np.sqrt((p ** 2).mean())),
                         s=time.time() - t0))
        print('[g2] it %2d  cost %.8e -> %.8e  %s  lam %.1e  |dv|max %.2e  combo-err %.3f %%  (%.0f s)'
              % (it, cost, c_new, 'acc' if accepted else 'REJ', lam, float(dv.abs().max()),
                 np.sqrt((p ** 2).mean()), time.time() - t0), flush=True)
        if not accepted or rel < 1e-9 or float(dv.abs().max()) < 1e-6:
            break
    tag = '%s: long-window baseline-only, nf %d, burn-in %d%s' % (
        'ii' if with_c else 'i', nf, burn, ', + c = (I-P0) r0' if with_c else ', unrestricted')
    stop = ('stalled: 12 LM trials rejected' if not accepted else
            'converged' if (rel < 1e-9 or float(dv.abs().max()) < 1e-6) else 'iteration cap')
    print('[g2] stop reason: %s' % stop)
    return verdict(v.numpy(), tag, dict(history=hist, iterations=it + 1, n_windows=B, nf=nf,
                                        burn_in=burn, final_cost=cost_of(v, x0), stop=stop))


res = arm_a() if args.arm == 'a' else arm_long(args.arm == 'ii')
name = 'g2_a.json' if args.arm == 'a' else 'g2_%s_%d.json' % (args.arm, args.nf)
json.dump(res, open(os.path.join(OUT, name), 'w'), indent=1)
print('[g2] wrote %s; done in %.0f s' % (os.path.join(OUT, name), sw()))
jh_env.check_no_leak()
