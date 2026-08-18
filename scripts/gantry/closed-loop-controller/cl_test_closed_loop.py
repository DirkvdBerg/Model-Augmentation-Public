"""MIGRATION STEPS 5-6 gate: the framework closed loop against the one it replaces (R2).

This is reference B of plan 5.1: the closed loop as it existed before the move, recorded on a
fixed batch before any edit, in `references/step1_reference.{json,npz}`. It catches a change in
the PLUMBING that gate A cannot see, because A tests the controller in isolation and says nothing
about whether the right controller reaches the right window.

What is deliberately NOT bit-identical here, and why the tolerances in plan 5.1 exist:

  * the controller step is now two matmuls on stacked matrices with the normalisation folded into
    B, C and D, instead of four einsums with a denormalise/renormalise sandwich per timestep;
  * `y = h(x)` comes from `Interconnect.output_only` (the output signal's dependency cone) instead
    of an affine map identified with nx + 1 probe forward passes.

Both are the same arithmetic in a different order, so the yardstick is the size of a difference
already known to be pure arithmetic: float32 versus float64 on this same loop gives a loss
difference of 3.1e-4 and a gradient 1 - cos of 2.2e-6, against a batch-to-batch 1 - cos of about
1.2. Hence loss rel <= 1e-3 and 1 - cos <= 1e-5.

  C1  the batch is the SAME batch     rebuilt from the manifest, hashes checked, or nothing below
                                      is a comparison at all
  C2  controller identity             each training record maps to the SAME physical controller as
                                      before, even though the row INDEX differs (one bank over 22
                                      records renumbers what two per-split banks numbered locally)
  C3  loss and gradient               both arms, against R2
  C4  y_pred                          a loss is one number and can agree by accident; a 400-step
                                      trajectory cannot
  C5  units                           a perturbed-state gate. The zero-ANN replay gate CANNOT
                                      catch a scale error on Cfb: with the ANN off the residual is
                                      identically zero and any scale factor is multiplied by zero
  C6  ctrl_ix plumbing                augment_training_data's per-window index selects, for every
                                      window, the controller that record actually had
  C7  selection scalar                the closed-loop free run over the four validation records,
                                      the number that decides checkpoints

Usage:
  PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python -u cl_test_closed_loop.py
  CL_SKIP_VAL=1 ... to skip C7 (the slow one)
"""
__project_origin__ = "added"

import copy
import dataclasses
import hashlib
import json
import os
import sys

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
from gantry_dynamic.data import load_traj, TRAIN_FILES, VAL_FILES         # noqa: E402
import cl_validation as CV                                                # noqa: E402
from cl_controller import y_op_for, controller_ss                         # noqa: E402
from cl_pipeline import build_closed_loop                                 # noqa: E402


THREADS = int(os.environ.get('CL_THREADS', 1))
torch.set_num_threads(THREADS)
SKIP_VAL = bool(int(os.environ.get('CL_SKIP_VAL', 0)))
# The OLD-implementation recording, kept deliberately. `step1_reference.{json,npz}` is
# re-recorded on the framework path after the migration, so comparing against it would compare
# the new code with itself and pass no matter what. This file is the frozen "before": R2 as
# `cl_fitsys.ClosedLoopLoss` computed it, at the corrected closed-loop perturbation, on code that
# no longer exists. It cannot be regenerated, which is exactly why it is checked in.
REF_JSON = os.path.join(HERE, 'references', 'step1_reference.old_impl.json')
REF_NPZ = os.path.join(HERE, 'references', 'step1_reference.old_impl.npz')
if not os.path.exists(REF_JSON):
    sys.exit('missing %s: the pre-migration recording is the whole point of this test and it '
             'cannot be regenerated, since the implementation it measured has been deleted.'
             % REF_JSON)
TOL_LOSS_REL, TOL_ONE_MINUS_COS, TOL_SEL_M = 1e-3, 1e-5, 1e-10

ref = json.load(open(REF_JSON))
refa = np.load(REF_NPZ)
ok = True


def check(tag, passed, detail=''):
    global ok
    ok &= bool(passed)
    print('   %-46s %s %s' % (tag, 'PASS' if passed else 'FAIL', detail))


def arr_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def cmp_grad(tag, g_new, key):
    g_old = np.asarray(refa[key], dtype=np.float64)
    g = np.asarray(g_new, dtype=np.float64)
    den = np.linalg.norm(g) * np.linalg.norm(g_old)
    omc = 1.0 - float(g @ g_old / den) if den > 0 else 0.0
    ratio = float(np.linalg.norm(g) / max(np.linalg.norm(g_old), 1e-300))
    check(tag, abs(omc) <= TOL_ONE_MINUS_COS,
          '1-cos %.3e   |g_new|/|g_old| %.9f' % (omc, ratio))


print('=' * 96)
print('MIGRATION STEPS 5-6: the framework closed loop against R2')
print('=' * 96)
if ref['threads'] != THREADS:
    sys.exit('reference recorded at %d threads, running at %d' % (ref['threads'], THREADS))

cfg = dataclasses.replace(CFG, seed=0, ann_route_ix=tuple(range(8)), lr=1e-7)
fs, norm, K0, na, nb, na_r, nb_r = dm.build_pipeline(cfg=cfg, verbose=False)
PARAMS = list(fs.hfn.parameters()) + list(fs.encoder.parameters())
h = hashlib.sha256()
for p in PARAMS:
    h.update(p.detach().cpu().numpy().tobytes())
if h.hexdigest() != ref['build_param_sha256']:
    sys.exit('build fingerprint changed; no comparison below would mean anything')
print('build fingerprint reproduced')

# ---- C1: the same batch -----------------------------------------------------------------------
print('\nC1  the batch is the same batch')
rv = lambda a: np.asarray(a).ravel()                                      # noqa: E731
train_names = [f[:-4] for f in TRAIN_FILES]
uh, yh, uf, yf, rec_names = [], [], [], [], []
for nm_rec in ref['batch_records']:
    sd = load_traj(nm_rec + '.mat', cfg)
    un = ((sd.u - rv(fs.norm.u0)) / rv(fs.norm.ustd)).astype(cfg.dtype_np)
    yn = ((sd.y - rv(fs.norm.y0)) / rv(fs.norm.ystd)).astype(cfg.dtype_np)
    for s in ref['starts'][nm_rec]:
        a, b = CV.encoder_window(un, yn, int(s), na, nb, na_r, nb_r, cfg.dtype_pt)
        uh.append(a); yh.append(b)
        uf.append(un[s:s + cfg.nf])
        yf.append(yn[s:s + cfg.nf])
        rec_names.append(nm_rec)
T = lambda A: torch.as_tensor(np.stack(A), dtype=cfg.dtype_pt)            # noqa: E731
UH, YH, UF, YF = torch.cat(uh, 0), torch.cat(yh, 0), T(uf), T(yf)
check('ufuture hash', arr_hash(UF.numpy()) == ref['batch_u_sha256'])
check('yfuture hash', arr_hash(YF.numpy()) == ref['batch_y_sha256'])

# ---- build the closed loop --------------------------------------------------------------------
val_data = deepSI.System_data_list([load_traj(f, cfg) for f in VAL_FILES])
sim = build_closed_loop(fs, norm, cfg, train_files=TRAIN_FILES, val_files=VAL_FILES,
                          val_data=val_data, verbose=True)
fs.simulator = sim
bank = sim.bank

# ---- C2: controller identity ------------------------------------------------------------------
print('\nC2  every training record still maps to its own physical controller')
worst = 0.0
for i, nm_rec in enumerate(train_names):
    row = sim.train_ctrl_rows[i]
    A, B, C, D = controller_ss(y_op_for(nm_rec), cfg.ts_new)
    got = bank.physical_D()[row].numpy()
    worst = max(worst, float(np.abs(got - np.asarray(D)).max()
                             / max(np.abs(D).max(), 1e-30)))
check('D matches controller_ss(y_op_for(record)) for all %d' % len(train_names), worst < 1e-6,
      'worst rel %.3e' % worst)

# ---- C3 / C4: loss, gradient, trajectory ------------------------------------------------------
print('\nC3/C4  loss, gradient and the predicted trajectory, against R2')
ctrl_ix_batch = torch.tensor([sim.train_ctrl_rows[train_names.index(n)] for n in rec_names],
                             dtype=torch.long)
ann_state = copy.deepcopy(fs.hfn.state_dict())


def perturb():
    """The CLOSED-LOOP perturbation amplitude, which is smaller than the open-loop one.

    At 1e-2 the closed-loop rollout is chaotic and the gradient is not a well-defined quantity:
    the same implementation gives loss 4.81e-02 in float32 and 2.29e-03 in float64, and two
    implementations that agree to 1 - cos = 8.9e-16 at 1e-4 give 1 - cos = 1.87 at 1e-2 IN
    FLOAT64. See PERTURB_SIGMA_CL in cl_step1_reference.py for the measured sweep.
    """
    from model_augmentation.fit_systems.blocks import Static_ANN_Block
    ann = next(m for m in fs.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
    g_ = torch.Generator().manual_seed(int(ref['perturb_seed']))
    sigma = float(ref['perturb_sigma_closed_loop'])
    with torch.no_grad():
        for p in ann.parameters():
            p.add_(torch.randn(p.shape, generator=g_, dtype=p.dtype) * sigma)


for arm in ('zero', 'pert'):
    if arm == 'pert':
        perturb()
    for p in PARAMS:
        p.grad = None
    L = fs.loss(UH, YH, UF, YF, ctrl_ix=ctrl_ix_batch)
    L.backward()
    g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().reshape(-1)
                   for p in PARAMS]).numpy()
    v_old = float(ref['scalars']['R2_loss_%s' % arm])
    rel = abs(float(L.detach()) - v_old) / abs(v_old)
    check('loss, ANN %-5s' % arm, rel <= TOL_LOSS_REL,
          'new %.12e  old %.12e  rel %.3e' % (float(L.detach()), v_old, rel))
    cmp_grad('gradient, ANN %-5s' % arm, g, 'R2_grad_%s' % arm)
    if arm == 'pert':
        fs.hfn.load_state_dict(ann_state)

with torch.no_grad():
    x0 = fs.encoder(UH, YH)
    y_new, _ = fs.simulate(x0, UF, YF, ctrl_ix=ctrl_ix_batch)
y_old = np.asarray(refa['R2_ypred_zero'], dtype=np.float64)
d = np.abs(y_new.numpy().astype(np.float64) - y_old)
check('y_pred (32, %d, 3)' % cfg.nf, float(d.max() / max(np.abs(y_old).max(), 1e-30)) <= 1e-3,
      'max rel %.3e' % float(d.max() / max(np.abs(y_old).max(), 1e-30)))

# ---- C5: units --------------------------------------------------------------------------------
print('\nC5  units gate (a perturbed-state gate; zero-ANN replay cannot catch a scale error)')
u_norm, u_expect, u_rel = bank.check_units(ctrl_ix=0)
check('u_fb round-trips through the folded scaling', float(np.max(u_rel)) <= 1e-6,
      'worst rel %.3e   Dc @ e = [%s] N' % (float(np.max(u_rel)),
                                            ' '.join('%.4e' % v for v in u_expect)))

# ---- C6: ctrl_ix plumbing ---------------------------------------------------------------------
print('\nC6  per-window controller index over the real training set')
train_list = [load_traj(f, cfg) for f in TRAIN_FILES]
sdl = deepSI.System_data_list(train_list)
data = fs.make_training_data(fs.norm.transform(sdl), nf=cfg.nf, stride=cfg.stride)
ctrl_ix = np.asarray(data[4])
check('a fifth array arrived', len(data) == 5, '%d arrays' % len(data))
check('one index per window', len(ctrl_ix) == len(data[0]),
      '%d vs %d' % (len(ctrl_ix), len(data[0])))
check('window counts match the recording',
      sim.last_window_counts == list(np.asarray(refa['cl_counts'])),
      str(sim.last_window_counts[:3]) + ' ...')
# every window must carry the controller of the record it came from
old_rec_ix = np.asarray(refa['cl_rec_ix'] if 'cl_rec_ix' in refa.files
                        else refa['cl_ctrl_ix'])
want = np.array([sim.train_ctrl_rows[r] for r in old_rec_ix], dtype=np.int64)
check('every window carries its own record\'s controller', np.array_equal(ctrl_ix, want),
      '%d/%d agree' % (int((ctrl_ix == want).sum()), len(want)))

# ---- C7: the selection scalar -----------------------------------------------------------------
if SKIP_VAL:
    print('\nC7  SKIPPED (CL_SKIP_VAL=1)')
else:
    print('\nC7  closed-loop selection scalar over the four validation records ...', flush=True)
    sel = fs.cal_validation_error(val_data, validation_measure='sim-RMS')
    sel_old = float(ref['scalars']['R2_selection_untrained'])
    check('selection scalar', abs(sel - sel_old) <= TOL_SEL_M,
          'new %.12e  old %.12e  |d| %.3e m' % (sel, sel_old, abs(sel - sel_old)))
    per_old = np.asarray(refa['R2_selection_per_record'])
    per_new = np.asarray(sim.last_per_record)
    check('per record', float(np.abs(per_new - per_old).max()) <= TOL_SEL_M,
          'max |d| %.3e m' % float(np.abs(per_new - per_old).max()))

print('\n' + '=' * 96)
print('OVERALL: %s' % ('PASS' if ok else 'FAIL'))
sys.exit(0 if ok else 1)
