"""Task item (ii): CAN the ANN learn the absorber when the initial condition is exact?

WHAT IS DIFFERENT FROM `scripts/gantry/gantry_interconnect_dynamic.py`
----------------------------------------------------------------------
Two changes, and nothing else:

  1. The SUBNET encoder is GONE. Every training window is initialised from the
     truth's six physical states, with velocities taken from the 8-state truth's
     own integrator (`data_exact.py`), not from the record's `gradient()` rows.
     The model's augmented rows 6-7 start at ZERO, as they do today, and are
     deliberately NOT seeded: they are the model's own latent coordinates, the
     ANN may represent anything it likes in them, and at initialisation the ANN
     output is exactly zero so anything seeded there is overwritten at step 1
     anyway (`verify_ms_gradient.py` gate G6). The handoff settles this in
     section 8; it is not reopened here.
  2. The baseline carries the truth's static mass distribution at `delta_a = 0`
     (`plant_cog.py`), so the centre-of-gravity mismatch stops polluting the X
     and Theta rows. Confound removal, not a fix (coulomb-offset F4).

After both, the ONLY difference between truth and model is the absorber's
DYNAMICS. If the ANN cannot learn from that, the cause is structural, and
which structure is what the log has to attribute.

WHY THE MODEL IS BUILT HERE RATHER THAN IMPORTED
------------------------------------------------
`gantry_dynamic/model.py::build_model` also builds the encoder, the multiple
shooting subclass, the orth penalty, the ReZero gate and the Lipschitz cap. With
the encoder bypassed, deepSI's `fit()` is not usable either (its loss calls the
encoder and its validation measure simulates from the encoder, so checkpoint
selection would be made on a metric this experiment has deleted). Rather than
subclass around all of that, the interconnect is assembled directly from the
same three blocks, wired with the same five calls, and trained with a loop that
does exactly what `SSE_Interconnect.loss` does minus the encoder. Every line is
one that can be pointed at. The interconnect wiring below is a verbatim copy of
`gantry_dynamic/model.py:96-138`.

THE VALIDATION MEASURE
----------------------
`sim-RMS` in the pipeline means a whole-record simulation initialised by the
encoder. There is no encoder here, so that measure does not exist. The measure
used instead is the natural one for this target: a free run of `nf` samples from
the exact IC, over a grid of window starts on the held-out records, RMS of the
output error in METRES. `--freerun` additionally reports the whole-record free
run from the exact IC at K0. The ANN-off value of the same measure is the
comparison the handoff asks for, and it is exact rather than approximate,
because the zero-initialised final layer makes the ANN output identically zero
before the first optimizer step.

Run (probe the learning rate first, it is cheap):
  ... python -u scripts/gantry/true-init-augmentation/true_init_train.py --probe
  ... python -u scripts/gantry/true-init-augmentation/true_init_train.py --epochs 20 --lr 1e-7
"""
__project_origin__ = "added"

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry'))

from model_augmentation.utils.utils import expansion_matrix, selection_matrix   # noqa: E402
from model_augmentation.utils.torch_nets import zero_init_feed_forward_nn       # noqa: E402
from model_augmentation.fit_systems.interconnect import Interconnect            # noqa: E402
from model_augmentation.fit_systems.blocks import (                             # noqa: E402
    Gantry_State_Block, Linear_Output_Block, Static_ANN_Block)

from gantry_dynamic.config import RunConfig                                     # noqa: E402
from gantry_dynamic.data import (                                               # noqa: E402
    load_datasets, compute_normalization, TRAIN_FILES, VAL_FILES)
from plant_cog import Gantry_State_Block_CoG                                    # noqa: E402
from data_exact import exact_truth                                             # noqa: E402

OUT = os.path.join(REPO, 'simulations', 'gantry_subnet', 'diagnostics')
SDIR = os.path.join(REPO, 'simulations', 'gantry_subnet', 'true_init_augmentation')
PHY_IX = np.arange(6)
CH = ['X1', 'X2', 'Y']

# Mirrors the pipeline CFG in gantry_interconnect_dynamic.py for everything that
# still applies. nf_seconds, nx_ann, routing, net size, activation, batch, seed,
# fs and up_sample are identical; stride is 100 (the handoff's figure) and lr is
# a command-line argument because the pipeline's 1e-7 was tuned WITH an encoder
# in the loop and that loop no longer exists.
CFG = RunConfig(
    mode='augmentation', encoder_init='linear_map', ann_activation='tanh',
    joint_estimation=False, snr=None, seed=42,
    fs_orig=20000, fs_new=4000, stride=100, use_f64=False,
    nx_ann=2, ann_route_ix=(0, 1, 2, 3, 4, 5, 6, 7),
    n_nodes_per_layer=16, n_hidden_layers=2, up_sample=1,
    batch_size=256, lr=1e-7, epochs=20, nf_seconds=0.100,
)


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════
def build_interconnect(cfg, norm, cog=True, dtype=torch.float32):
    """Physics + output + ANN, wired exactly as gantry_dynamic/model.py does."""
    nxd = cfg.nx_phys + cfg.nx_ann
    route_ix = np.asarray(cfg.ann_route_ix)
    ic = Interconnect(nxd, cfg.nu, cfg.ny, debugging=False)

    kw = dict(Y_op=None, std_x=norm.std_x, std_u=norm.std_u,
              x_mean=norm.x_mean, u_mean=norm.u_mean,
              Ts=cfg.ts_new, up_sample=cfg.up_sample)
    phy_block = (Gantry_State_Block_CoG(**kw) if cog else Gantry_State_Block(**kw)).to(dtype)
    out_phys = Linear_Output_Block(C=norm.Cd_norm, D=norm.Dd_np).to(dtype)
    ann_block = Static_ANN_Block(
        nz=nxd + cfg.nu, nw=len(route_ix),
        n_nodes_per_layer=cfg.n_nodes_per_layer,
        n_hidden_layers=cfg.n_hidden_layers,
        net=zero_init_feed_forward_nn,
        activation=torch.nn.Identity if cfg.ann_activation == 'linear' else torch.nn.Tanh,
    ).to(dtype)

    ic.add_block(phy_block)
    ic.add_block(out_phys)
    ic.add_block(ann_block)
    ic.connect_block_signals(ann_block, ["x", "u"], [])
    ic.connect_signals(ann_block, "xp", "additive", expansion_matrix(route_ix, nxd))
    ic.connect_signals("x", phy_block, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(phy_block, ["u"], [])
    ic.connect_signals(phy_block, "xp", "additive", expansion_matrix(PHY_IX, nxd))
    ic.connect_signals("x", out_phys, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(out_phys, ["u"], ["y"])
    return ic.to(dtype), ann_block, phy_block


# ══════════════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════════════
class Windows:
    """Normalised u/y and exact normalised x0 per record, plus a (rec, start) list."""

    def __init__(self, files, cfg, norm, nf, stride, k0, dtype=torch.float32):
        self.nf, self.k0, self.stride = nf, k0, stride
        self.U, self.Y, self.X0, self.idx = [], [], [], []
        u0, ustd = norm.u_mean.flatten(), norm.std_u.flatten()
        xm, xs = norm.x_mean.flatten(), norm.std_x.flatten()
        for r, f in enumerate(files):
            tr = exact_truth(f[:-4])
            rec, x6 = tr['rec'], tr['x6']
            N = len(rec['u'])
            self.U.append(torch.as_tensor((rec['u'] - u0) / ustd, dtype=dtype))
            self.Y.append(torch.as_tensor((rec['y'] - norm.y0) / norm.ystd, dtype=dtype))
            x0 = np.zeros((N, cfg.nx_phys + cfg.nx_ann))
            x0[:, :6] = (x6 - xm) / xs            # rows 6-7 stay ZERO by design
            self.X0.append(torch.as_tensor(x0, dtype=dtype))
            self.idx += [(r, s) for s in range(k0, N - nf + 1, stride)]
        self.idx = np.array(self.idx)
        self.ystd = torch.as_tensor(norm.ystd, dtype=dtype)

    def __len__(self):
        return len(self.idx)

    def batch(self, sel):
        rs = self.idx[sel]
        u = torch.stack([self.U[r][s:s + self.nf] for r, s in rs])
        y = torch.stack([self.Y[r][s:s + self.nf] for r, s in rs])
        x0 = torch.stack([self.X0[r][s] for r, s in rs])
        return x0, u, y


def rollout_loss(ic, x0, u, y):
    """Exactly SSE_Interconnect.loss minus the encoder: mean over time of MSE."""
    x, errs = x0, []
    for t in range(u.shape[1]):
        yhat, x = ic(x, u[:, t])
        errs.append(torch.nn.functional.mse_loss(y[:, t], yhat))
    return torch.mean(torch.stack(errs))


@torch.no_grad()
def evaluate(ic, W, batch=256, max_windows=None):
    """Windowed free run from the exact IC. Returns metres, not normalised units.

    rms         RMS output error over all windows and steps           [m]
    dc_scatter  std ACROSS windows of the per-window mean error       [m]
    """
    n = len(W) if not max_windows else min(len(W), max_windows)
    sel_all = np.linspace(0, len(W) - 1, n).astype(int)
    se = torch.zeros(3, dtype=torch.float64)
    cnt = 0
    wms = []
    for i in range(0, n, batch):
        sel = sel_all[i:i + batch]
        x0, u, y = W.batch(sel)
        x = x0
        acc = []
        for t in range(u.shape[1]):
            yhat, x = ic(x, u[:, t])
            acc.append((yhat - y[:, t]) * W.ystd)          # -> metres
        e = torch.stack(acc, dim=1)                        # (B, nf, 3)
        se += (e.double() ** 2).sum(dim=(0, 1))
        cnt += e.shape[0] * e.shape[1]
        wms.append(e.mean(dim=1).double())
    wm = torch.cat(wms, dim=0)
    return dict(rms=float(torch.sqrt(se.sum() / (cnt * 3))),
                rms_ch=(torch.sqrt(se / cnt)).tolist(),
                dc_scatter=wm.std(dim=0).tolist(),
                dc_bias=wm.mean(dim=0).tolist(),
                n_win=int(wm.shape[0]))


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════
def train(args):
    cfg = CFG
    dtype = torch.float64 if args.f64 else torch.float32
    os.makedirs(SDIR, exist_ok=True)
    run_id = args.tag or time.strftime('%Y%m%d_%H%M%S')

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    data = load_datasets(cfg)
    norm = compute_normalization(cfg, data)

    nf, k0 = cfg.nf, cfg.na_nb
    print(f'\nTrue-init augmentation run {run_id}')
    print(f'  nf {nf} ({cfg.nf_seconds} s)  stride {args.stride}  k0 {k0}  '
          f'batch {args.batch}  lr {args.lr:g}  epochs {args.epochs}  '
          f'dtype {"f64" if args.f64 else "f32"}  CoG {"ON" if not args.no_cog else "off"}')
    print(f'  routing {cfg.ann_route_ix}  nx_ann {cfg.nx_ann}  '
          f'net {cfg.n_hidden_layers}x{cfg.n_nodes_per_layer} {cfg.ann_activation}')

    Wtr = Windows(TRAIN_FILES, cfg, norm, nf, args.stride, k0, dtype)
    Wva = Windows(VAL_FILES, cfg, norm, nf, args.val_stride, k0, dtype)
    print(f'  {len(Wtr)} train windows, {len(Wva)} val windows')

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    ic, ann, phy = build_interconnect(cfg, norm, cog=not args.no_cog, dtype=dtype)

    # ANN-off reference: the zero-initialised final layer makes the ANN output
    # exactly zero, so the untrained model IS the baseline. Verified, not assumed.
    with torch.no_grad():
        z = torch.zeros(4, cfg.nx_phys + cfg.nx_ann + cfg.nu, dtype=dtype)
        z[:, 2] = 0.1
        w0 = float(ann.net(z).abs().max())
    print(f'  ANN output at init, max |w| = {w0:.3e}  '
          f'{"(exactly zero: ANN-off == untrained)" if w0 == 0.0 else "NOT ZERO"}')

    base = evaluate(ic, Wva, batch=args.batch, max_windows=args.val_windows)
    print(f'\n  [ANN off] val nf-RMS {base["rms"]:.6e} m   per channel '
          f'{["%.3e" % v for v in base["rms_ch"]]}')
    print(f'  [ANN off] val per-window DC scatter [m] '
          f'{["%.3e" % v for v in base["dc_scatter"]]}')

    opt = torch.optim.Adam(ann.parameters(), lr=args.lr)
    rng = np.random.default_rng(cfg.seed)
    nb = len(Wtr) // args.batch
    hist = []
    best = dict(rms=base['rms'], epoch=-1)
    t0 = time.time()
    for ep in range(args.epochs):
        ic.train()
        acc, seen = 0.0, 0
        order = rng.permutation(len(Wtr))
        for b in range(nb):
            sel = order[b * args.batch:(b + 1) * args.batch]
            x0, u, y = Wtr.batch(sel)
            opt.zero_grad()
            L = rollout_loss(ic, x0, u, y)
            L.backward()
            gn = float(torch.sqrt(sum((p.grad ** 2).sum() for p in ann.parameters()
                                      if p.grad is not None)))
            opt.step()
            acc += float(L)
            seen += 1
            if args.probe:
                print(f'    step {b:3d}  train MSE {float(L):.6e}  '
                      f'sqrt {float(L)**0.5:.6e}  |g ANN| {gn:.3e}')
                if b + 1 >= args.probe_steps:
                    break
        ic.eval()
        v = evaluate(ic, Wva, batch=args.batch, max_windows=args.val_windows)
        wmax = float(max(p.abs().max() for p in ann.parameters()))
        hist.append(dict(epoch=ep, train_mse=acc / max(seen, 1), val=v,
                         ann_wmax=wmax, grad=gn))
        flag = ''
        if v['rms'] < best['rms']:
            best = dict(rms=v['rms'], epoch=ep, val=v)
            flag = '  <-- best'
            torch.save(ann.state_dict(), os.path.join(SDIR, f'ann_{run_id}_best.pt'))
        print(f'  ep {ep:3d}  train sqrtMSE {(acc/max(seen,1))**0.5:.6e}  '
              f'val nf-RMS {v["rms"]:.6e} m  ({100*(v["rms"]/base["rms"]-1):+.2f} % vs ANN off)'
              f'  |g| {gn:.2e}  max|W| {wmax:.2e}  {time.time()-t0:6.0f} s{flag}')
        if args.probe:
            break

    verdict = ('IMPROVES' if best['rms'] < base['rms'] * 0.99 else
               'no improvement' if best['rms'] <= base['rms'] * 1.01 else 'DEGRADES')
    print(f'\n  ANN off  val nf-RMS {base["rms"]:.6e} m')
    print(f'  ANN on   val nf-RMS {best["rms"]:.6e} m  (best at epoch {best["epoch"]})')
    print(f'  VERDICT: {verdict}   ratio on/off = {best["rms"]/base["rms"]:.4f}')

    res = dict(run_id=run_id, lr=args.lr, epochs=args.epochs, batch=args.batch,
               stride=args.stride, nf=nf, k0=k0, cog=not args.no_cog,
               dtype='f64' if args.f64 else 'f32',
               n_train_windows=len(Wtr), n_val_windows=len(Wva),
               ann_zero_at_init=(w0 == 0.0),
               ann_off=base, best=best, history=hist, verdict=verdict)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f'true_init_train_{run_id}.json')
    with open(p, 'w') as f:
        json.dump(res, f, indent=2, default=float)
    print(f'  wrote {p}')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=CFG.epochs)
    ap.add_argument('--lr', type=float, default=CFG.lr)
    ap.add_argument('--batch', type=int, default=CFG.batch_size)
    ap.add_argument('--stride', type=int, default=CFG.stride)
    ap.add_argument('--val-stride', type=int, default=400, dest='val_stride')
    ap.add_argument('--val-windows', type=int, default=0, dest='val_windows',
                    help='0 = every val window')
    ap.add_argument('--no-cog', action='store_true', dest='no_cog')
    ap.add_argument('--f64', action='store_true')
    ap.add_argument('--probe', action='store_true',
                    help='a few steps only, printing per-step loss and |g| (lr probe)')
    ap.add_argument('--probe-steps', type=int, default=8, dest='probe_steps')
    ap.add_argument('--tag', type=str, default=None)
    return train(ap.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
