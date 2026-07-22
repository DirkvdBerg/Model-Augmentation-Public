"""train_artbp.py -- Phase D: does the poly-tail truncation reduce ARTBP's variance vs geometric,
at matched average cost, while still collapsing the dY DC? (reframed per D-120)

v12 already proved geometric ARTBP collapses the DC. The open question is the paper's central claim
(Sec. 4): the geometric distribution has EXPONENTIALLY-growing compensation weights (high variance),
the poly-tail (Eq. 14) has POLYNOMIALLY-growing weights (finite variance for alpha>3). We used the
high-variance geometric in v12. Phase D compares four training conditions differing ONLY in the
truncation distribution, at matched mean horizon (K_bar ~ nf) and cap (H_max):

  fixed      standard nf window                                        biased control (DC -> -4.5e-6)
  geom       geometric, c_t = 1/nf                                     v12's variant (high variance)
  poly4      poly-tail Eq. 14, c_t=(a-1)/((a-2)L0+t), alpha=4          finite-variance candidate (primary)
  poly6      poly-tail, alpha=6                                        secondary

Unbiased loss-term reweighting (README section 2), UNIFIED across distributions:
  loss = (1/nf) sum_{t=0..K-1} w_t * mse_t,  w_t = 1 / P(K>t) = 1 / prod_{s<t}(1-c_s).
  E_K[loss] = (1/nf) sum_t mse_t = the H_max-horizon loss, for ANY c-schedule (exchange argument).

Instrumentation per step: mean ann(Z_pts) dY (the DC), dLoss/d(bias) on dY via a patched ann.forward
+ zero probe leaf (v3b machinery; its VARIANCE OVER STEPS is the DC-direction estimator variance =
the primary metric), sampled K, and the max compensation weight. Per epoch: held-out windowed nf-RMS
(the fit gate). Convention: data -> ./data/train_<mode>_seed*.npz.

# THEORY labels: geometric weights = paper Sec.4 constant-c_t; poly-tail c_t = paper Eq. 14.
"""
import os
import sys
import time
import argparse
import dataclasses

import numpy as np
import torch

HERE   = os.path.dirname(os.path.abspath(__file__))
GANTRY = os.path.dirname(HERE)
REPO   = os.path.dirname(os.path.dirname(GANTRY))          # repo root
sys.path.insert(0, REPO)                                   # so `import model_augmentation` works on any launch
sys.path.insert(0, os.path.join(GANTRY, 'drift-demo'))

import demo_common as dm
from demo_common import CFG
from gantry_dynamic.data import TRAIN_FILES
from model_augmentation.fit_systems.blocks import Static_ANN_Block

# ── one config surface ─────────────────────────────────────────────────────────
MODES     = tuple(os.environ.get('TA_MODES', 'fixed,geom,poly4,poly6').split(','))
SEEDS     = tuple(int(s) for s in os.environ.get('TA_SEEDS', '0').split(','))
# Cluster grid (job array): task_idx -> (mode, seed), ordered mode-major. 4 modes x 5 seeds = 20 tasks.
GRID_MODES = ('fixed', 'geom', 'poly4', 'poly6')
GRID_SEEDS = (0, 1, 2, 3, 4)
LR        = float(os.environ.get('TA_LR', '1e-7'))
BATCH     = int(os.environ.get('TA_BATCH', '256'))
STRIDE    = int(os.environ.get('TA_STRIDE', '10'))
H_MAX     = int(os.environ.get('TA_HMAX', '1600'))        # truncation cap (4x nf), matched across modes
EPOCHS    = int(os.environ.get('TA_EPOCHS', '1'))
HELDOUT   = float(os.environ.get('TA_HELDOUT', '0.2'))    # window fraction held out for the fit gate
PRINT_EVERY = 25
STATE_NAMES = ['X', 'Theta', 'Y', 'dX', 'dTheta', 'dY', 'delta_a', 'vdelta_a']
IY = 5
ALPHA = {'poly4': 4.0, 'poly6': 6.0}

datDir = os.path.join(HERE, 'data'); os.makedirs(datDir, exist_ok=True)


def c_schedule(mode, t_arr, q, L0):
    """Per-step truncation probability c_t for t=0..len-1. # THEORY: geom=const (paper Sec.4),
    poly=Eq.14 c_t=(alpha-1)/((alpha-2)*L0 + t)."""
    if mode == 'geom':
        return np.full_like(t_arr, q, dtype=np.float64)
    a = ALPHA[mode]
    return (a - 1.0) / ((a - 2.0) * L0 + t_arr)           # THEORY: Tallec-Ollivier Eq. 14


def sample_K(c_full, rng):
    """First step (0-based t) where a Bernoulli(c_t) fires -> K=t+1; capped H_MAX, floored 2 (K>=2:
    a length-1 window has only the detached encoder-init output, no gradient)."""
    u = rng.random(len(c_full))
    fired = np.nonzero(u < c_full)[0]
    K = (int(fired[0]) + 1) if fired.size else len(c_full)
    return max(2, min(K, len(c_full)))


def weights(c_full, K):
    """w_t = 1 / P(K>t) = 1 / prod_{s<t}(1-c_s), t=0..K-1 (w_0=1)."""
    surv = np.concatenate([[1.0], np.cumprod(1.0 - c_full[:K - 1])])
    return (1.0 / surv).astype(np.float32)


def run_seed(mode, seed, first):
    cfg = dataclasses.replace(CFG, seed=seed)
    np.random.seed(seed); torch.manual_seed(seed)
    fit_sys, norm, K0, na, nb, na_right, nb_right = dm.build_pipeline(cfg=cfg, verbose=first)
    nf = cfg.hp['nf']; q = 1.0 / nf; L0 = float(nf)
    rng = np.random.default_rng(seed + 100)               # truncation RNG, seed-dependent

    ann = next(m for m in fit_sys.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
    pen = getattr(fit_sys, 'orth_penalty', None)
    assert pen is not None and ann.nw == 8, 'need orth_observe Z_pts and nw=8'
    Zpts = pen.Z_pts
    route_col = list(int(i) for i in np.asarray(cfg.ann_route_ix).ravel()).index(IY)

    # v3b probe: patch ann.forward to add a zero per-row bias; its .grad after backward =
    # dLoss/d(bias). probe_bias is NOT an optimizer param (probe only). Custom loop (no deepSI fit),
    # so no checkpoint_save pickle issue.
    probe_bias = torch.zeros(ann.nw, requires_grad=True)
    orig_forward = ann.forward
    ann.forward = lambda z: orig_forward(z) + probe_bias.view(1, -1, 1)

    # per-record normalized arrays + window starts (train/heldout split by window)
    um = norm.u_mean.flatten()[None, :]; us = norm.std_u.flatten()[None, :]
    y0 = np.asarray(norm.y0).flatten()[None, :]; ys = np.asarray(norm.ystd).flatten()[None, :]
    recs, starts = [], []
    for ri, f in enumerate(TRAIN_FILES):
        u, y, _, _ = dm.load_T(f, cfg)
        un = ((u - um) / us).astype(np.float32); yn = ((y - y0) / ys).astype(np.float32)
        recs.append((un, yn)); N = len(un)
        for p in range(max(K0, na, nb), N - H_MAX, STRIDE):
            starts.append((ri, p))
    starts = np.array(starts)
    split_rng = np.random.default_rng(seed)               # same split every mode at a given seed
    split_rng.shuffle(starts)
    n_hold = int(len(starts) * HELDOUT)
    hold_starts, train_starts = starts[:n_hold], starts[n_hold:]

    def gather(bs, length):
        B = len(bs)
        U = np.empty((B, length, 3), np.float32); Y = np.empty((B, length, 3), np.float32)
        UH = np.empty((B, nb + 1, 3), np.float32); YH = np.empty((B, na + 1, 3), np.float32)
        for i, (ri, p) in enumerate(bs):
            un, yn = recs[ri]
            U[i] = un[p:p + length]; Y[i] = yn[p:p + length]
            UH[i] = un[p - nb:p + 1]; YH[i] = yn[p - na:p + 1]
        return (torch.from_numpy(U), torch.from_numpy(Y),
                torch.from_numpy(np.ascontiguousarray(UH)), torch.from_numpy(np.ascontiguousarray(YH)))

    def heldout_nfrms():
        with torch.no_grad():
            se = 0.0; nb_ = 0
            for b0 in range(0, len(hold_starts), BATCH):
                bs = hold_starts[b0:b0 + BATCH]
                U, Y, UH, YH = gather(bs, nf)
                x = fit_sys.encoder(UH.contiguous(), YH.contiguous())
                sse = 0.0
                for t in range(nf):
                    yhat, x = fit_sys.hfn(x, U[:, t, :])
                    sse = sse + torch.mean((Y[:, t, :] - yhat) ** 2)
                se = se + float(sse / nf); nb_ += 1
            return float(np.sqrt(se / max(nb_, 1)))

    params = list(ann.parameters()) + list(fit_sys.encoder.parameters())
    opt = torch.optim.Adam(params, lr=LR)
    c_full = c_schedule(mode, np.arange(H_MAX, dtype=np.float64), q, L0) if mode != 'fixed' else None

    rec = {'dc': [], 'dcgrad': [], 'K': [], 'loss': [], 'wmax': []}
    t0 = time.time()
    print(f'\n=== train seed {seed} | MODE={mode} | {len(train_starts)} train / {len(hold_starts)} '
          f'heldout windows | lr={LR:.0e} nf={nf} H_max={H_MAX} ===')
    step = 0
    for ep in range(EPOCHS):
        order = np.arange(len(train_starts)); rng.shuffle(order)
        for b0 in range(0, len(train_starts), BATCH):
            bs = train_starts[order[b0:b0 + BATCH]]
            if mode == 'fixed':
                K = nf; w = np.ones(nf, np.float32)
            else:
                K = sample_K(c_full, rng); w = weights(c_full, K)
            U, Y, UH, YH = gather(bs, K)
            with torch.no_grad():
                x = fit_sys.encoder(UH.contiguous(), YH.contiguous()).detach()
            opt.zero_grad()
            acc = 0.0
            for t in range(K):
                yhat, x = fit_sys.hfn(x, U[:, t, :])
                acc = acc + float(w[t]) * torch.mean((Y[:, t, :] - yhat) ** 2)
            loss = acc / nf
            if not loss.requires_grad:
                continue
            loss.backward()
            with torch.no_grad():
                rec['dc'].append(float(ann(Zpts)[..., 0].mean(0)[IY]))
            rec['dcgrad'].append(float(probe_bias.grad[route_col]) if probe_bias.grad is not None else np.nan)
            probe_bias.grad = None
            opt.step()
            rec['loss'].append(float(loss.item())); rec['K'].append(K); rec['wmax'].append(float(w[-1]))
            step += 1
            if step <= 6 or step % PRINT_EVERY == 0:
                print(f'    [step {step:4d}] K={K:5d} wmax={w[-1]:.2e} loss={rec["loss"][-1]:.3e} '
                      f'| DC dY={rec["dc"][-1]:+.3e} dLoss/dbias dY={rec["dcgrad"][-1]:+.3e}')
        hr = heldout_nfrms()
        print(f'  [nf-probe] epoch {ep+1}/{EPOCHS} heldout nf-RMS={hr:.4e}  (train windows={len(train_starts)})')

    ann.forward = orig_forward
    dcg = np.asarray(rec['dcgrad']); half = dcg[len(dcg)//2:]
    out = {k: np.asarray(v) for k, v in rec.items()}
    out.update(labels=np.array(STATE_NAMES), mode=mode, seed=seed, nf=nf, hmax=H_MAX,
               heldout_nfrms=hr, dcgrad_var=float(np.nanvar(half)),
               dc_endpoint=float(np.mean([r for r in rec['dc'][-50:]])), walltime=time.time() - t0)
    np.savez(os.path.join(datDir, f'train_{mode}_seed{seed}.npz'), **out)
    print(f'  seed {seed} {mode}: endpoint DC dY={out["dc_endpoint"]:+.3e} | '
          f'dcgrad var(2nd half)={out["dcgrad_var"]:.3e} | heldout nf-RMS={hr:.4e} | '
          f'{out["walltime"]:.0f}s')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task_idx', type=int, default=None,
                    help='SLURM array index -> one (mode,seed) from the 4x5 grid (mode-major); '
                         'overrides TA_MODES/TA_SEEDS. Each task writes its own train_<mode>_seed<seed>.npz.')
    args = ap.parse_args()
    if args.task_idx is not None:
        grid = [(m, s) for m in GRID_MODES for s in GRID_SEEDS]
        mode, seed = grid[args.task_idx]
        combos = [(mode, seed)]
        print(f'train_artbp | task_idx={args.task_idx}/{len(grid)-1} -> mode={mode} seed={seed} | '
              f'lr={LR:.0e} H_max={H_MAX} epochs={EPOCHS}')
    else:
        combos = [(m, s) for m in MODES for s in SEEDS]
        print(f'train_artbp | modes={MODES} | seeds={SEEDS} | lr={LR:.0e} | H_max={H_MAX} | epochs={EPOCHS}')
    summary = {}
    first = True
    for mode, seed in combos:
        o = run_seed(mode, seed, first); first = False
        summary[(mode, seed)] = o
    if len(combos) > 1:                                       # cross-combo summary only when >1 ran here
        print('\n==== SUMMARY (endpoint DC dY | dcgrad var 2nd-half | heldout nf-RMS | walltime) ====')
        modes_here = [m for m in GRID_MODES if any(k[0] == m for k in summary)]
        for mode in modes_here:
            ss  = [k[1] for k in summary if k[0] == mode]
            dcs = [summary[(mode, s)]['dc_endpoint'] for s in ss]
            vs  = [summary[(mode, s)]['dcgrad_var'] for s in ss]
            hrs = [summary[(mode, s)]['heldout_nfrms'] for s in ss]
            wt  = [summary[(mode, s)]['walltime'] for s in ss]
            print(f'  {mode:6s}: DC {np.mean(dcs):+.3e} (sd {np.std(dcs):.2e}) | '
                  f'var {np.mean(vs):.3e} | nf-RMS {np.mean(hrs):.4e} | {np.mean(wt):.0f}s/seed')


if __name__ == '__main__':
    main()
