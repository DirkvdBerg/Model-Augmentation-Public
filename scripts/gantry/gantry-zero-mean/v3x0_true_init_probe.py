"""v3x0_true_init_probe.py -- encoder-bypass (true-x0) DC-birth probe (encoder-init test).

Question: does the ANN learn the systematic DC (v3) BECAUSE of the encoder's velocity init error
(v4: the encoder-init error dominates the within-window K=0 ramp)? Intervention: train each window
from the TRUE initial state instead of the encoder estimate. If the DC vanishes, encoder-init is the
cause; if it persists, encoder-init is NOT the cause (drop that story, per
`causal-claim-needs-intervention-not-observation`).

Why a custom loop: deepSI inits each window from `x = self.encoder(uhist, yhist)` inside its loss
(interconnect.py:433) and does not thread the true state, so we replicate the exact rollout
(`yhat, x = fit_sys.hfn(x, u)`, mean per-step MSE) with a controllable init.

INIT (the key knob), V3X0_INIT env:
  'true'    -> x0 = [ (x_logical[p]-x_mean)/std_x  (6 physical, TRUE);  0, 0 (2 aug latents) ].
               The 2 aug states are LATENTS with no fixed physical scale (W^a random-init,
               pre_encoder.py:396; related to true delta_a only by a fitted map, hence R2_linmap),
               so a "true aug init" is not well-defined; they are tiny and start ~equilibrium, so 0.
  'encoder' -> x0 = fit_sys.encoder(uhist, yhist)  == the CONTROL: must reproduce v3's encoder-init
               DC (sign+rough magnitude). If the control does not match v3, the loop is wrong; do
               NOT trust the 'true' result until the control passes.

Everything else identical to v3 (lr=1e-7, nf=400, augmentation band, full X+Theta+Y routing, 3
unfixed seeds). Per-optimizer-step DC instrumentation as v3b: per-row mean of ann(Z_pts) + the loss
gradient along a constant per-row correction (dLoss/d(bias)).

Convention: lives in scripts/gantry/gantry-zero-mean/; data -> ./data/v3x0_<init>_seed*.npz,
figures -> ./figures/. This DOES train (short). Smoke via env caps before the full run.
"""
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE   = os.path.dirname(os.path.abspath(__file__))
GANTRY = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(GANTRY, 'drift-demo'))    # demo_common (sets up the rest of the path)

import deepSI
import demo_common as dm
from demo_common import CFG
from gantry_dynamic.data import TRAIN_FILES
from model_augmentation.fit_systems.blocks import Static_ANN_Block

# ─────────────────────────────────────────────────────────────────────────────
# Config knobs (one surface)
# ─────────────────────────────────────────────────────────────────────────────
INIT     = os.environ.get('V3X0_INIT', 'true')          # 'true' (intervention) | 'encoder' (control)
OPT      = os.environ.get('V3X0_OPT', 'adam')           # 'adam' | 'sgd' (SGD-vs-Adam mechanism test)
SEEDS    = tuple(int(s) for s in os.environ.get('V3X0_SEEDS', '0,1,2').split(','))
LR       = float(os.environ.get('V3X0_LR', '1e-7'))     # V3X0_LR to match effective descent for SGD
BATCH    = 256
STRIDE   = 10
PRINT_EVERY = 50
MAX_WINDOWS = int(os.environ.get('V3X0_MAXWIN', '0'))   # 0 = all; >0 caps windows (smoke)
MAX_STEPS   = int(os.environ.get('V3X0_MAXSTEPS', '0')) # 0 = full epoch; >0 caps steps (smoke)
PREFIX   = os.environ.get('V3X0_PREFIX', 'v3x0')

STATE_NAMES = ['X', 'Theta', 'Y', 'dX', 'dTheta', 'dY', 'delta_a', 'vdelta_a']
K0_DRIFT = (0, 2, 3, 5)

figDir = os.path.join(HERE, 'figures'); os.makedirs(figDir, exist_ok=True)
datDir = os.path.join(HERE, 'data');    os.makedirs(datDir, exist_ok=True)


def build_windows(norm, cfg, K0, na, nb):
    """Normalized train windows: for each start p (stride) return the arrays needed by both init
    modes. u=stage force, y=stage pos, x_logical=(N,6). Normalization matches build_model."""
    nf = cfg.hp['nf']
    um = norm.u_mean.flatten()[None, :]; us = norm.std_u.flatten()[None, :]
    y0 = np.asarray(norm.y0).flatten()[None, :]; ys = np.asarray(norm.ystd).flatten()[None, :]
    xm = norm.x_mean.flatten(); xs = norm.std_x.flatten()
    U_n, Y_n, X0p, UH, YH = [], [], [], [], []
    for f in TRAIN_FILES:
        u, y, xl, _ = dm.load_T(f, cfg)
        u_n = ((u - um) / us).astype(np.float32)
        y_n = ((y - y0) / ys).astype(np.float32)
        xphys_n = ((xl - xm[None, :]) / xs[None, :]).astype(np.float32)   # (N,6) true physical, normalized
        N = len(u)
        for p in range(K0, N - nf, STRIDE):
            U_n.append(u_n[p:p + nf])                    # (nf,3) rollout input
            Y_n.append(y_n[p:p + nf])                    # (nf,3) rollout target
            X0p.append(xphys_n[p])                       # (6,) true physical init
            UH.append(u_n[p - nb:p + 1])                 # (nb+1,3) history for the encoder control
            YH.append(y_n[p - na:p + 1])                 # (na+1,3) history for the encoder control
    return (np.stack(U_n), np.stack(Y_n), np.stack(X0p), np.stack(UH), np.stack(YH))


def run_seed(seed):
    import dataclasses
    # build_pipeline reseeds np/torch from cfg.seed (demo_common.py:61,78), overriding any global
    # seed set here. So vary the seed AT the config, else every "seed" builds the identical model.
    cfg = dataclasses.replace(CFG, seed=seed)
    np.random.seed(seed); torch.manual_seed(seed)
    fit_sys, norm, K0, na, nb, na_right, nb_right = dm.build_pipeline(cfg=cfg, verbose=(seed == SEEDS[0]))
    nf = cfg.hp['nf']
    ann = next(m for m in fit_sys.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
    pen = getattr(fit_sys, 'orth_penalty', None)
    assert pen is not None and ann.nw == 8, 'need orth_observe Z_pts and nw=8'
    Zpts = pen.Z_pts
    labels = STATE_NAMES

    U_n, Y_n, X0p, UH, YH = build_windows(norm, cfg, K0, na, nb)
    if MAX_WINDOWS:
        U_n, Y_n, X0p, UH, YH = U_n[:MAX_WINDOWS], Y_n[:MAX_WINDOWS], X0p[:MAX_WINDOWS], UH[:MAX_WINDOWS], YH[:MAX_WINDOWS]
    nwin = len(U_n)
    U_n, Y_n, X0p, UH, YH = (torch.tensor(a) for a in (U_n, Y_n, X0p, UH, YH))

    # B: bias added in ann.forward (interconnect calls block.forward directly -> hooks bypassed);
    # its .grad after backward = dLoss/d(constant correction). No-op checkpoint not needed (no deepSI fit).
    probe_bias = torch.zeros(ann.nw, dtype=torch.float32, requires_grad=True)
    orig_forward = ann.forward
    def forward_with_bias(z):
        return orig_forward(z) + probe_bias.view(1, -1, 1)
    ann.forward = forward_with_bias

    # deepSI shadows nn.Module.parameters with a list attribute, so gather trainable params
    # explicitly: ANN + encoder (joint=False -> physical/output blocks have none). Encoder gets
    # no gradient in 'true' mode (not in the graph), so it simply does not update there.
    params = list(ann.parameters()) + list(fit_sys.encoder.parameters())
    opt = torch.optim.SGD(params, lr=LR) if OPT == 'sgd' else torch.optim.Adam(params, lr=LR)
    rec = {'mean': [], 'std': [], 'bias_grad': [], 'loss': []}

    order = np.arange(nwin); np.random.shuffle(order)
    nbatch = (nwin + BATCH - 1) // BATCH
    print(f'\n=== seed {seed} | INIT={INIT} | OPT={OPT} | {nwin} windows | {nbatch} steps | lr={LR:.0e} | nf={nf} ===')
    step = 0
    for bi in range(nbatch):
        idx = order[bi * BATCH:(bi + 1) * BATCH]
        idx_t = torch.as_tensor(idx)
        u_b = U_n[idx_t]; y_b = Y_n[idx_t]                # (B,nf,3)
        if INIT == 'true':
            x = torch.zeros(len(idx), ann.nw, dtype=torch.float32)   # (B,8)
            x[:, :6] = X0p[idx_t]                          # true physical; aug rows stay 0
        else:                                             # 'encoder' control
            with torch.no_grad():
                x = fit_sys.encoder(UH[idx_t].contiguous(), YH[idx_t].contiguous()).detach()

        opt.zero_grad()
        errs = []
        for t in range(nf):
            yhat, x = fit_sys.hfn(x, u_b[:, t, :])        # (B,3),(B,8)  same call as interconnect loss
            errs.append(torch.nn.functional.mse_loss(y_b[:, t, :], yhat))
        loss = torch.stack(errs).mean()
        loss.backward()
        with torch.no_grad():
            wr = ann(Zpts)[..., 0]
            rec['mean'].append(wr.mean(0).cpu().numpy().copy())
            rec['std'].append(wr.std(0).cpu().numpy().copy())
        rec['bias_grad'].append(probe_bias.grad.detach().cpu().numpy().copy()
                                if probe_bias.grad is not None else np.full(ann.nw, np.nan))
        probe_bias.grad = None
        opt.step()
        rec['loss'].append(float(loss.item()))
        step += 1
        if step <= 10 or step % PRINT_EVERY == 0:
            m = rec['mean'][-1]; g = rec['bias_grad'][-1]
            print(f'    [step {step:5d}] loss={rec["loss"][-1]:.4e} | DC dX={m[3]:+.3e} dY={m[5]:+.3e}'
                  f' | dLoss/dbias dX={g[3]:+.3e} dY={g[5]:+.3e}')
        if MAX_STEPS and step >= MAX_STEPS:
            break

    ann.forward = orig_forward
    out = {k: np.asarray(v) for k, v in rec.items()}
    out.update(labels=np.array(labels), init=INIT, seed=seed, lr=LR, nf=nf, nwin=nwin)
    np.savez(os.path.join(datDir, f'{PREFIX}_{INIT}_seed{seed}.npz'), **out)
    return out


def plot_multiseed(outs):
    rows = [STATE_NAMES[i] for i in K0_DRIFT]
    fh, ax = plt.subplots(1, len(rows), figsize=(5 * len(rows), 4.2), squeeze=False)
    for c, rn in enumerate(rows):
        j = STATE_NAMES.index(rn)
        for o in outs:
            k = np.arange(o['mean'].shape[0])
            ax[0, c].plot(k, o['mean'][:, j], lw=0.9, label=f'seed {int(o["seed"])}')
        ax[0, c].axhline(0, color='k', lw=0.6); ax[0, c].grid(True)
        ax[0, c].set_title(f'DC on {rn} (INIT={INIT})'); ax[0, c].set_xlabel('update step')
        if c == 0:
            ax[0, c].set_ylabel('mean of ann(Z_pts)'); ax[0, c].legend(fontsize=7)
    fh.suptitle(f'v3x0 true-init probe | INIT={INIT} (control=encoder must match v3; true=intervention)')
    fh.tight_layout(); fh.savefig(os.path.join(figDir, f'{PREFIX}_{INIT}_multiseed.png'), dpi=150)
    plt.close(fh)


def main():
    print(f'v3x0_true_init_probe | INIT={INIT} | seeds={SEEDS} | lr={LR:.0e}'
          + (f' | SMOKE maxwin={MAX_WINDOWS} maxsteps={MAX_STEPS}' if (MAX_WINDOWS or MAX_STEPS) else ''))
    outs = [run_seed(s) for s in SEEDS]
    plot_multiseed(outs)
    print(f'\ndone | data -> {datDir}\\{PREFIX}_{INIT}_seed*.npz | figure -> {figDir}')


if __name__ == '__main__':
    main()
