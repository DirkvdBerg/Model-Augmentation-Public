"""Does float32 corrupt the CLOSED-LOOP training gradient? Measured on the trained checkpoint.

THE QUESTION
------------
The closed loop turns an output difference into an input force through `Dc ~ 3e6` N/m. The
residual `y_data - y_model` is a difference of two numbers of size ~0.1 m, so in float32 it carries
about 1e-8 m of rounding regardless of how small the residual itself is; multiplied by `Dc` that is
~0.02 N injected at the model input every step (measured as a flat 2.400e-02 N in
`cl_direct_vs_residual.py` T1, identical at every ANN gain, i.e. model-independent). The relative
corruption is `eps*|y| / |y_data - y_model|`, so it GROWS as the model improves, which is exactly
the regime the thesis cares about.

That argues for float64. It does not settle it, because SGD tolerates a great deal of noise. What
matters is not whether the loss is noisy but whether the GRADIENT is misdirected.

WHAT IS MEASURED
----------------
On the trained closed-loop checkpoint, over one fixed batch of `nf`-length training windows:

  loss32 vs loss64        relative difference of the loss value
  cos(g32, g64)           cosine similarity of the full trainable-parameter gradient
  |g32| / |g64|           gradient magnitude ratio

and the SAME three for the open-loop loss, as a control: if the closed loop is no worse than open
loop, the precision issue is not a property of the loop.

THE CALIBRATION, without which the numbers mean nothing
-------------------------------------------------------
"Is cos = 0.997 good?" is unanswerable in the abstract. The reference scale is the disagreement
training already tolerates: the cosine between gradients of two DIFFERENT minibatches, in float64.
Stochastic gradient descent is built to work with that much disagreement. So the verdict is

  cos(g32, g64) much closer to 1 than cos(batch A, batch B)  ->  float32 is not the problem
  cos(g32, g64) comparable to or worse than that             ->  train the closed loop in float64

CHECKPOINT
----------
`server-results/deep-SI-checkpoints/FitSys_ClosedLoop_*_best.pth`, the step-6 run (76573):
val 2.187e-06 -> 1.393e-06, 36.3 % improvement over the baseline, 12 epochs. Loaded state-only into
a freshly built system, following `cl_plot_step6.py`: adopting the pickled `__dict__` leaves
captured handles pointing at stale modules.

Usage
-----
  PYTHONUNBUFFERED=1 python -u cl_precision_gradient.py
"""
__project_origin__ = "added"

import dataclasses
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
GANTRY = os.path.join(REPO, 'scripts', 'gantry')
for p in (REPO, GANTRY, HERE, os.path.join(GANTRY, 'drift-demo'),
          os.path.join(GANTRY, 'msd-offset')):
    if p not in sys.path:
        sys.path.insert(0, p)

import demo_common as dm                                                  # noqa: E402
from demo_common import CFG                                               # noqa: E402
from gantry_dynamic.data import load_traj, VAL_FILES                      # noqa: E402
import cl_plant as PLANT                                                  # noqa: E402
import cl_validation as CV                                                # noqa: E402
from cl_controller import ControllerBank, rollout, open_loop_rollout      # noqa: E402
import cl_fitsys as CLF                                                   # noqa: E402

SRV = os.path.join(HERE, 'server-results')
RECORD = 'V1_standstill_Yp10'
N_WIN = 16               # windows per batch
t0 = time.time()


def find_ckpt():
    g = sorted(glob.glob(os.path.join(SRV, '**', '*_best.pth'), recursive=True))
    if not g:
        raise SystemExit('no *_best.pth under %s' % SRV)
    return g[0]


def build(use_f64, lr):
    cfg = dataclasses.replace(CFG, seed=0, ann_route_ix=tuple(range(8)), lr=lr,
                              use_f64=use_f64)
    fs, norm, K0, na, nb, na_r, nb_r = dm.build_pipeline(cfg=cfg, verbose=False)
    nx = cfg.nx_phys + cfg.nx_ann
    C_out, b_out = PLANT.identify_output_map(fs.hfn, nx, cfg.nu, dtype=cfg.dtype_pt)
    step_fn, out_fn = PLANT.make_fns(fs, C_out, b_out)
    vnames = [f[:-4] for f in VAL_FILES]
    bank = ControllerBank(vnames, cfg.ts_new, dtype=cfg.dtype_pt,
                          ystd=norm.ystd, std_u=norm.std_u)
    CLF.attach(fs, bank, step_fn, out_fn)      # BEFORE torch.load, see cl_plot_step6.py
    return cfg, fs, norm, K0, na, nb, na_r, nb_r, bank, vnames


def load_state(fs, ckpt):
    ck = torch.load(ckpt, weights_only=False)
    # state only, cast by copy_ into whatever dtype this system was built with
    fs.hfn.load_state_dict(ck['hfn'].state_dict())
    fs.encoder.load_state_dict(ck['encoder'].state_dict())


def make_batch(cfg, fs, K0, na, nb, na_r, nb_r, starts, rec_slot):
    """One fixed batch of nf-length windows from RECORD, in this cfg's dtype.

    Normalisation comes from `fs.norm`, deepSI's own norm object, which is what the encoder and
    hfn were built with. (The pipeline's separate `Norm` dataclass stores (nu, 1) columns and is
    used only to feed the ControllerBank; mixing the two silently rescales the loop.) Same choice
    as cl_step5_reset_cost.py.
    """
    sd = load_traj(RECORD + '.mat', cfg)
    rv = lambda a: np.asarray(a).ravel()                                  # noqa: E731
    un = ((sd.u - rv(fs.norm.u0)) / rv(fs.norm.ustd)).astype(cfg.dtype_np)
    yn = ((sd.y - rv(fs.norm.y0)) / rv(fs.norm.ystd)).astype(cfg.dtype_np)
    uh, yh, uf, yf = [], [], [], []
    for s in starts:
        a, b = CV.encoder_window(un, yn, int(s), na, nb, na_r, nb_r, cfg.dtype_pt)
        uh.append(a); yh.append(b)
        uf.append(un[s:s + cfg.nf])
        yf.append(yn[s:s + cfg.nf])
    T = lambda A: torch.as_tensor(np.stack(A), dtype=cfg.dtype_pt)        # noqa: E731
    return (torch.cat(uh, 0), torch.cat(yh, 0), T(uf), T(yf),
            torch.full((len(starts),), rec_slot, dtype=torch.long))


def grad_of(fs, batch, closed):
    """Flat gradient of the loss over every trainable parameter."""
    for p in fs.hfn.parameters():
        p.grad = None
    for p in fs.encoder.parameters():
        p.grad = None
    uh, yh, uf, yf, rix = batch
    L = fs.loss(uh, yh, uf, yf, rec_ix=rix) if closed \
        else fs.loss_open_loop(uh, yh, uf, yf, rec_ix=rix)
    L.backward()
    gs = [p.grad.detach().reshape(-1).double() for p in
          list(fs.hfn.parameters()) + list(fs.encoder.parameters())
          if p.grad is not None]
    return float(L.detach()), torch.cat(gs)


def compare(g_a, g_b):
    cos = float(torch.dot(g_a, g_b) / (g_a.norm() * g_b.norm()))
    return cos, float(g_a.norm() / g_b.norm())


# ---------------------------------------------------------------------------------------------
print('=' * 96)
print('PRECISION vs the CLOSED-LOOP GRADIENT, on the trained checkpoint')
print('=' * 96)

ckpt = find_ckpt()
res_path = sorted(glob.glob(os.path.join(SRV, 'step6_result_*.json')))
lr = json.load(open(res_path[0]))['lr'] if res_path else 1e-7
print('checkpoint : %s' % os.path.basename(ckpt))
print('lr         : %g' % lr)

print('\nbuilding float32 ...', flush=True)
c32, fs32, n32, K32, na, nb, na_r, nb_r, bank32, vnames = build(False, lr)
load_state(fs32, ckpt)
print('building float64 ...', flush=True)
c64, fs64, n64, K64, na4, nb4, nar4, nbr4, bank64, _ = build(True, lr)
load_state(fs64, ckpt)
print('nf = %d, windows per batch = %d, record = %s' % (c32.nf, N_WIN, RECORD))

slot = vnames.index(RECORD)
sd = load_traj(RECORD + '.mat', c32)
hi = len(sd.u) - c32.nf - 2
startsA = np.linspace(K32, hi, N_WIN).astype(int)
startsB = np.linspace(K32 + c32.nf // 2, hi - c32.nf // 2, N_WIN).astype(int)   # disjoint-ish

bA32 = make_batch(c32, fs32, K32, na, nb, na_r, nb_r, startsA, slot)
bA64 = make_batch(c64, fs64, K64, na4, nb4, nar4, nbr4, startsA, slot)
bB64 = make_batch(c64, fs64, K64, na4, nb4, nar4, nbr4, startsB, slot)

rows = []
for closed in (True, False):
    tag = 'closed loop' if closed else 'open loop (control)'
    L32, g32 = grad_of(fs32, bA32, closed)
    L64, g64 = grad_of(fs64, bA64, closed)
    cos_p, rat_p = compare(g32, g64)
    _, gB64 = grad_of(fs64, bB64, closed)
    cos_b, rat_b = compare(g64, gB64)
    rows.append((tag, L32, L64, cos_p, rat_p, cos_b, rat_b))
    print('\n%s' % tag)
    print('  loss float32 / float64          : %.9e / %.9e   (rel %.3e)'
          % (L32, L64, abs(L32 - L64) / abs(L64)))
    print('  cos(g32, g64)                   : %.9f' % cos_p)
    print('  |g32| / |g64|                   : %.6f' % rat_p)
    print('  cos(batch A, batch B) in f64    : %.9f      <- the scale SGD already tolerates'
          % cos_b)
    print('  |gA| / |gB| in f64              : %.6f' % rat_b)

print('\n' + '=' * 96)
print('VERDICT')
print('=' * 96)
for tag, L32, L64, cos_p, rat_p, cos_b, rat_b in rows:
    d_prec = 1.0 - cos_p
    d_batch = 1.0 - cos_b
    print('%-22s  1-cos precision = %.3e   1-cos batch = %.3e   ratio = %.3g'
          % (tag, d_prec, d_batch, d_prec / max(d_batch, 1e-30)))
cl = rows[0]
if (1 - cl[3]) < 0.1 * (1 - cl[5]):
    print('\n-> The float32/float64 gradient disagreement is far inside the batch-to-batch')
    print('   scatter that SGD already copes with. Training in float32 is not the problem;')
    print('   keep float64 for gates and validation only.')
else:
    print('\n-> The float32/float64 disagreement is comparable to (or worse than) the')
    print('   batch-to-batch scatter. The closed-loop training path should run in float64.')
print('[%.0fs]' % (time.time() - t0))
