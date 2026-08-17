"""STEP 4 GATES: the closed-loop training loss.

  G11 EXACT NO-OP.  The new loss with the loop DISABLED must equal the production open-loop loss
                    to float32 precision. This is the load-bearing gate: it proves the closed-loop
                    path differs from production ONLY by the loop, and not by some accidental
                    change in normalisation, reduction, window handling, or the param_loss and
                    orthogonality terms. Without it, any comparison between open and closed loop
                    is uninterpretable. Same discipline as MultipleShooting and ParamLoss, which
                    are exact no-ops when unconfigured.
  G12 rec_ix ALIGNMENT.  Every window must be attached to the record it actually came from. A
                    silent misalignment would give most windows the wrong Cfb and would present as
                    a training problem rather than a bookkeeping one. Checked by content, not by
                    counting: each window's ufuture is matched back against the source records.
  G13 GRADIENT.     The loss must produce finite gradients that reach the ANN through the loop.
  G14 CONTROLLER ACTIVE.  The closed-loop loss must actually differ from the open-loop loss. If it
                    does not, the controller is wired in but inert and every gate above would
                    still pass.

Usage: python -u cl_gate_loss.py
"""
__project_origin__ = "added"

import dataclasses
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

import deepSI                                                             # noqa: E402
import demo_common as dm                                                  # noqa: E402
from demo_common import CFG                                               # noqa: E402
from gantry_dynamic.data import load_traj, TRAIN_FILES                    # noqa: E402
from model_augmentation.fit_systems.blocks import Static_ANN_Block        # noqa: E402

import cl_plant as PLANT                                                  # noqa: E402
import cl_fitsys as CLF                                                   # noqa: E402
from cl_controller import ControllerBank                                  # noqa: E402

results = {}
t0 = time.time()
NF = 100          # short windows keep the gate fast; the claims are structural, not horizon-dependent
N_REC = 4         # first four training records

print('=' * 92)
print('STEP 4 GATES: closed-loop training loss')
print('=' * 92)
cfg = dataclasses.replace(CFG, seed=0)
fs, norm, K0, na, nb, na_r, nb_r = dm.build_pipeline(cfg=cfg, verbose=False)
nx = cfg.nx_phys + cfg.nx_ann
C_out, b_out = PLANT.identify_output_map(fs.hfn, nx, cfg.nu, dtype=cfg.dtype_pt)
step_fn, out_fn = PLANT.make_fns(fs, C_out, b_out)
names = [f[:-4] for f in TRAIN_FILES[:N_REC]]
bank = ControllerBank(names, cfg.ts_new, dtype=cfg.dtype_pt, ystd=norm.ystd, std_u=norm.std_u)
print('records %s' % names)
print('nf = %d, na = %d, nb = %d, na_right = %d, nb_right = %d   [%.0fs]'
      % (NF, na, nb, na_r, nb_r, time.time() - t0), flush=True)

train_list = [load_traj(f, cfg) for f in TRAIN_FILES[:N_REC]]
train_data = deepSI.System_data_list(train_list)
normed = fs.norm.transform(train_data)

# production data BEFORE grafting, so the comparison is against the real production path
prod_data = fs.make_training_data(normed, nf=NF)
prod_loss_fn = type(fs).loss
CLF.attach(fs, bank, step_fn, out_fn)
cl_data = fs.make_training_data(normed, nf=NF)

print('\nproduction arrays %d, closed-loop arrays %d (the extra one is rec_ix)'
      % (len(prod_data), len(cl_data)))
print('window count: production %d, closed-loop %d, per-record %s'
      % (len(prod_data[0]), len(cl_data[0]), fs._cl_counts))

# ---------------------------------------------------------------- G12
print('\n' + '-' * 92)
print('G12 rec_ix ALIGNMENT   (checked by CONTENT, not by counting)')
rec_ix = cl_data[4]
uf = cl_data[2]
un_by_rec = [((sd.u - fs.norm.u0) / fs.norm.ustd).astype(cfg.dtype_np) for sd in train_list]
rng = np.random.default_rng(0)
probe = rng.choice(len(rec_ix), size=200, replace=False)
bad = 0
for w in probe:
    claimed = int(rec_ix[w])
    win = np.asarray(uf[w])
    # the window must appear verbatim somewhere in the record rec_ix claims
    src = un_by_rec[claimed]
    # window j of a record starts at kmid = k0 + j*stride (system_data.py:322-326); stride is 1
    # here, but write it out so the check stays valid if the pipeline ever sets stride > 1
    STRIDE = 1
    k0 = max(na, nb)
    off = k0 + (w - sum(fs._cl_counts[:claimed])) * STRIDE
    if off + len(win) > len(src) or not np.allclose(src[off:off + len(win)], win, atol=0, rtol=0):
        bad += 1
print('    probed %d windows, mismatched %d' % (len(probe), bad))
g12 = bad == 0
print('    %s   (a mismatch means the wrong Cfb on most windows)'
      % ('PASS' if g12 else 'FAIL'))
results['G12 rec_ix alignment'] = g12

# ---------------------------------------------------------------- G11
print('\n' + '-' * 92)
print('G11 EXACT NO-OP   (loop disabled => production loss)')
B = 64
sl = slice(0, B)
uh = torch.as_tensor(prod_data[0][sl], dtype=cfg.dtype_pt)
yh = torch.as_tensor(prod_data[1][sl], dtype=cfg.dtype_pt)
ufu = torch.as_tensor(prod_data[2][sl], dtype=cfg.dtype_pt)
yfu = torch.as_tensor(prod_data[3][sl], dtype=cfg.dtype_pt)
rix = torch.as_tensor(cl_data[4][sl])
with torch.no_grad():
    L_prod = float(prod_loss_fn(fs, uh, yh, ufu, yfu))
    L_open = float(fs.loss_open_loop(uh, yh, ufu, yfu, rix))
    L_closed = float(fs.loss(uh, yh, ufu, yfu, rix))
print('    production loss (open loop)      %.12e' % L_prod)
print('    new loss, loop DISABLED          %.12e' % L_open)
print('    new loss, loop ENABLED           %.12e' % L_closed)
rel = abs(L_open - L_prod) / max(abs(L_prod), 1e-30)
print('    |disabled - production| / prod   %.3e' % rel)
g11 = rel < 1e-6
print('    %s   (anything else means something OTHER than the loop changed)'
      % ('PASS' if g11 else 'FAIL'))
results['G11 exact no-op'] = g11

# ---------------------------------------------------------------- G14
print('\n' + '-' * 92)
print('G14 CONTROLLER ACTIVE   (closed must differ from open, or the loop is inert)')
d = abs(L_closed - L_open) / max(abs(L_open), 1e-30)
print('    |closed - open| / open           %.3e' % d)
g14 = d > 1e-6
print('    %s' % ('PASS' if g14 else 'FAIL'))
results['G14 controller active'] = g14

# ---- LOSS SCALE, reported because it silently rescales every regularisation weight ----
# The loop suppresses the within-window error, so the DATA term shrinks by a large factor while
# param_loss and the orthogonality penalty do not. Any Lambda or orth_beta tuned against the
# open-loop loss is therefore effectively multiplied by this ratio when the loop is switched on.
# Not a defect, but it must be re-tuned or the data term renormalised before reading anything
# into a closed-loop run's regularisation behaviour.
print('\n    LOSS SCALE  open %.4e -> closed %.4e   data term shrinks %.1fx'
      % (L_open, L_closed, L_open / max(L_closed, 1e-30)))
pl = sum(float(m.param_loss()) for m in fs.hfn.connected_blocks if hasattr(m, 'param_loss'))
ob = 0.0 if fs.orth_penalty is None else float(fs.orth_penalty.beta)
print('    param_loss now %.4e, orth beta %.4e' % (pl, ob))
print('    relative weight of param_loss: open %.3e -> closed %.3e'
      % (pl / max(L_open, 1e-30), pl / max(L_closed, 1e-30)))

# ---------------------------------------------------------------- G13
print('\n' + '-' * 92)
print('G13 GRADIENT   (finite, and reaching the ANN through the loop)')
# SSE_Interconnect inherits deepSI's System, NOT nn.Module, so it has no zero_grad (D-070).
# Only fs.hfn and fs.encoder are torch modules, and together they hold every trainable parameter.
fs.hfn.zero_grad(set_to_none=True)
fs.encoder.zero_grad(set_to_none=True)
L = fs.loss(uh, yh, ufu, yfu, rix)
L.backward()
ann = next(m for m in fs.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
gn = [(n, float(p.grad.norm())) for n, p in ann.named_parameters() if p.grad is not None]
enc_gn = [float(p.grad.norm()) for p in fs.encoder.parameters() if p.grad is not None]
print('    loss %.6e' % float(L))
for n, v in gn:
    print('      ANN %-22s grad norm %.6e' % (n, v))
print('    encoder grad norms: %s' % ' '.join('%.3e' % v for v in enc_gn))
finite = all(np.isfinite(v) for _, v in gn) and all(np.isfinite(v) for v in enc_gn)
nonzero = any(v > 0 for _, v in gn)
g13 = finite and nonzero
print('    finite=%s  ANN receives gradient=%s' % (finite, nonzero))
print('    %s' % ('PASS' if g13 else 'FAIL'))
results['G13 gradient'] = g13

print('\n' + '=' * 92)
for k in sorted(results):
    print('%-28s %s' % (k, 'PASS' if results[k] else 'FAIL'))
allok = all(results.values())
print('=' * 92)
print('STEP 4 %s   [%.0fs]' % ('PASSED' if allok else 'HAS FAILURES', time.time() - t0))
sys.exit(0 if allok else 1)
