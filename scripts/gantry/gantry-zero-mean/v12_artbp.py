"""v12_artbp.py -- ARTBP (unbiased gradient at fixed average cost): does it remove the DC drive?

THE mechanism test. The DC's DRIVE is a small sign-consistent gradient on the dY constant (v3b:
+2.4e-5, same sign 3 seeds), hypothesised to be the FINITE-WINDOW (truncated-BPTT) bias: on a z=1
axis the loss summed over only nf steps rewards a constant velocity "lean" that is optimal over the
window but wrong asymptotically. Longer FIXED windows do not fix it (DC ~ 1/nf, nonzero at every nf
up to 3200). ARTBP attacks the BIAS directly, at the SAME average horizon.

Two modes, identical except the loss:
  fixed  -> standard nf=400 window, L = (1/nf) sum_{k=1..nf} mse_k     (CONTROL, reproduces v3b)
  artbp  -> random horizon K ~ geometric(q=1/nf) capped at H_max (MEAN K = nf, matched cost);
            L = (1/nf) sum_{k=1..K} w_k * mse_k,  w_k = (1-q)^{-(k-1)}  (Tallec-Ollivier 2017
            reweighting, forward-window adaptation). E_K[L] is unbiased for the long-horizon loss,
            so the rare long rollouts (upweighted) inject the z=1 drift penalty the fixed window
            cannot see.

Read: fixed DC ~ -4e-6 sign-locked (harness OK); artbp DC -> ~1e-7 / sign-scattered = the drive WAS
truncation bias (mechanism confirmed) and ARTBP is a candidate fix at nf. artbp DC ~ -4e-6 unchanged
= not truncation bias. Convention: data -> ./data/v12_<mode>_seed*.npz, figure -> ./figures/.
"""
import os
import sys
import dataclasses

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE   = os.path.dirname(os.path.abspath(__file__))
GANTRY = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(GANTRY, 'drift-demo'))

import demo_common as dm
from demo_common import CFG
from gantry_dynamic.data import TRAIN_FILES
from model_augmentation.fit_systems.blocks import Static_ANN_Block

MODE     = os.environ.get('V12_MODE', 'artbp')            # 'fixed' (control) | 'artbp'
SEEDS    = tuple(int(s) for s in os.environ.get('V12_SEEDS', '0,1,2').split(','))
LR       = float(os.environ.get('V12_LR', '1e-7'))
BATCH    = 256
STRIDE   = 10
H_MAX    = int(os.environ.get('V12_HMAX', '1600'))        # cap on the random horizon (4x nf)
PRINT_EVERY = 25
MAX_STEPS = int(os.environ.get('V12_MAXSTEPS', '0'))      # 0 = full epoch; >0 caps (smoke)
STATE_NAMES = ['X', 'Theta', 'Y', 'dX', 'dTheta', 'dY', 'delta_a', 'vdelta_a']
IY = 5
figDir = os.path.join(HERE, 'figures'); os.makedirs(figDir, exist_ok=True)
datDir = os.path.join(HERE, 'data');    os.makedirs(datDir, exist_ok=True)


def run_seed(seed):
    cfg = dataclasses.replace(CFG, seed=seed)
    np.random.seed(seed); torch.manual_seed(seed)
    fit_sys, norm, K0, na, nb, na_right, nb_right = dm.build_pipeline(cfg=cfg, verbose=(seed == SEEDS[0]))
    nf = cfg.hp['nf']; q = 1.0 / nf
    ann = next(m for m in fit_sys.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
    pen = getattr(fit_sys, 'orth_penalty', None)
    assert pen is not None and ann.nw == 8, 'need orth_observe Z_pts and nw=8'
    Zpts = pen.Z_pts

    # Per-record normalized full arrays (real with-MSD data); index slices on the fly.
    um = norm.u_mean.flatten()[None, :]; us = norm.std_u.flatten()[None, :]
    y0 = np.asarray(norm.y0).flatten()[None, :]; ys = np.asarray(norm.ystd).flatten()[None, :]
    recs = []
    starts = []
    for ri, f in enumerate(TRAIN_FILES):
        u, y, xl, _ = dm.load_T(f, cfg)
        un = ((u - um) / us).astype(np.float32); yn = ((y - y0) / ys).astype(np.float32)
        recs.append((un, yn))
        N = len(un)
        for p in range(max(K0, na), N - H_MAX, STRIDE):     # room for H_MAX + encoder history
            starts.append((ri, p))
    starts = np.array(starts)
    nwin = len(starts)

    # ANN-output DC probe. (No bias-grad probe here; v3b already has it. Keep it lean.)
    rec = {'mean': [], 'loss': [], 'K': []}
    params = list(ann.parameters()) + list(fit_sys.encoder.parameters())
    opt = torch.optim.Adam(params, lr=LR)

    order = np.arange(nwin); np.random.shuffle(order)
    nbatch = (nwin + BATCH - 1) // BATCH
    print(f'\n=== v12 seed {seed} | MODE={MODE} | {nwin} windows | {nbatch} steps | lr={LR:.0e} '
          f'| nf={nf} H_max={H_MAX} ===')

    def gather(batch_starts, length):
        B = len(batch_starts)
        U = np.empty((B, length, 3), np.float32); Y = np.empty((B, length, 3), np.float32)
        UH = np.empty((B, nb + 1, 3), np.float32); YH = np.empty((B, na + 1, 3), np.float32)
        for i, (ri, p) in enumerate(batch_starts):
            un, yn = recs[ri]
            U[i] = un[p:p + length]; Y[i] = yn[p:p + length]
            UH[i] = un[p - nb:p + 1]; YH[i] = yn[p - na:p + 1]
        return (torch.from_numpy(U), torch.from_numpy(Y),
                torch.from_numpy(np.ascontiguousarray(UH)), torch.from_numpy(np.ascontiguousarray(YH)))

    step = 0
    for bi in range(nbatch):
        bs = starts[order[bi * BATCH:(bi + 1) * BATCH]]
        if MODE == 'artbp':
            # K >= 2: a length-1 window has only the detached encoder-init output (no gradient);
            # P(K=1)=q=0.25% so clamping to 2 is negligible for the reweighting/unbiasedness.
            K = max(2, int(min(np.random.geometric(q), H_MAX)))   # mean ~ 1/q = nf; capped
        else:
            K = nf
        U, Y, UH, YH = gather(bs, K)
        with torch.no_grad():
            x = fit_sys.encoder(UH.contiguous(), YH.contiguous()).detach()
        opt.zero_grad()
        if MODE == 'artbp':
            # w_k = (1-q)^{-(k-1)}: inverse prob step k is included -> unbiased long-horizon loss.
            wk = torch.from_numpy(((1.0 - q) ** (-np.arange(K))).astype(np.float32))
            acc = 0.0
            for t in range(K):
                yhat, x = fit_sys.hfn(x, U[:, t, :])
                acc = acc + wk[t] * torch.mean((Y[:, t, :] - yhat) ** 2)
            loss = acc / nf
        else:
            errs = []
            for t in range(K):
                yhat, x = fit_sys.hfn(x, U[:, t, :])
                errs.append(torch.mean((Y[:, t, :] - yhat) ** 2))
            loss = torch.stack(errs).mean()
        if not loss.requires_grad:
            continue                                          # safety: degenerate window, skip
        loss.backward()
        with torch.no_grad():
            rec['mean'].append(ann(Zpts)[..., 0].mean(0).cpu().numpy().copy())
        opt.step()
        rec['loss'].append(float(loss.item())); rec['K'].append(K)
        step += 1
        if step <= 8 or step % PRINT_EVERY == 0:
            m = rec['mean'][-1]
            print(f'    [step {step:4d}] K={K:5d} loss={rec["loss"][-1]:.3e} '
                  f'| DC dX={m[3]:+.3e} dY={m[5]:+.3e}')
        if MAX_STEPS and step >= MAX_STEPS:
            break

    out = {k: np.asarray(v) for k, v in rec.items()}
    out.update(labels=np.array(STATE_NAMES), mode=MODE, seed=seed, nf=nf, hmax=H_MAX)
    np.savez(os.path.join(datDir, f'v12_{MODE}_seed{seed}.npz'), **out)
    print(f'  seed {seed} final DC dY = {out["mean"][-1, IY]:+.3e} '
          f'(v3b fixed-nf reference: -3.5..-4.2e-6)')
    return out


def main():
    print(f'v12_artbp | MODE={MODE} | seeds={SEEDS} | lr={LR:.0e}'
          + (f' | SMOKE maxsteps={MAX_STEPS}' if MAX_STEPS else ''))
    outs = [run_seed(s) for s in SEEDS]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for o in outs:
        k = np.arange(o['mean'].shape[0])
        ax[0].plot(k, o['mean'][:, IY], lw=0.9, label=f'seed {int(o["seed"])}')
    ax[0].axhline(0, color='k', lw=0.6)
    ax[0].axhline(-4e-6, color='tab:red', ls='--', lw=0.8, label='v3b fixed-nf DC (-4e-6)')
    ax[0].set_title(f'DC on dY vs step (MODE={MODE})'); ax[0].set_xlabel('update step')
    ax[0].set_ylabel('mean ann(Z_pts) dY'); ax[0].grid(True); ax[0].legend(fontsize=7)
    for o in outs:
        ax[1].plot(np.arange(len(o['K'])), o['K'], lw=0.5, alpha=0.7)
    ax[1].set_title('sampled horizon K per step'); ax[1].set_xlabel('update step')
    ax[1].set_ylabel('K (steps)'); ax[1].grid(True)
    fig.suptitle(f'v12 ARTBP | MODE={MODE} | mean horizon={outs[0]["nf"]}, H_max={H_MAX}')
    fig.tight_layout(); fig.savefig(os.path.join(figDir, f'v12_{MODE}_dc.png'), dpi=150)
    plt.close(fig)
    print(f'\ndone | data -> {datDir}\\v12_{MODE}_seed*.npz')


if __name__ == '__main__':
    main()
