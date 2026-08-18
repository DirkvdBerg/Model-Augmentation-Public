"""MIGRATION STEP 1: the reference sets, and the gate that every later step is checked against.

`PLAN-move-to-model-augmentation.md` section 5.1 defines the references and orders them by
strength. A (MATLAB) is external ground truth and is checked by `test_controller_exact.py`,
`verify_controller.py`, `verify_cfb_against_records.py` and `p1_equivalence.py`, captured to
`references/gateA_*.txt`. This script records the rest, which are regression nets rather than
truth: they prove that nothing changed, and would faithfully preserve an error if one existed.
That is why A comes first and why none of these is allowed to stand in for it.

  R1  the seams must be inert.  The PRODUCTION loss with no closed loop anywhere:
      SSE_Interconnect_MultipleShooting.loss (n_seg = 1, so it falls through OrthLoss and
      ParamLoss to SSE_Interconnect.loss). BIT-IDENTICAL through migration step 2a; from 2b
      onward the reduction change costs exactly one float32 ulp, deliberately and once.
  R2  the closed loop.  Its loss and gradient, the rollout's trajectory, the units gate, the
      per-window controller index and the closed-loop selection scalar. Recorded on the framework
      implementation since migration step 7; the comparison against the implementation it replaced
      is frozen in `cl_test_closed_loop.py`.
  D   the SEGMENTED loss at n_seg > 1 with both defect terms live. Migration step 4 routes
      SSE_Interconnect_MultipleShooting's inner loop through self.simulate(), and n_seg > 1
      WITHOUT a simulator has to keep working because that is what the defect diagnostics use.
      Section 5.1 did not list this and no step recorded it; that gap is closed here.
  E   the nf-probe histories. Migration step 2d converts `_install_nf_val_probe` from a monkey
      patch into a `validation_probes` entry and asserts "the probe histories are identical",
      which needs a pre-change history. Same gap, same fix.

WHY THIS IS RECORDED ON THE UNTRAINED BUILD, not on the step-6 checkpoint
------------------------------------------------------------------------
Plan section 8 accepts that `FitSys_ClosedLoop_Go1qTA_{best,last}.pth` stop loading after the
move: they pickle a class that `attach()` creates at runtime, and no shim will be written. A
reference that can only be evaluated on those files is useless as a regression net, because it
cannot be re-evaluated after the thing it guards. The untrained build is reproducible on both
sides: `build_pipeline` seeds before the data load and again before `build_model`, so the
parameters are a deterministic function of the config. The sha256 of every parameter tensor is
stored and `--check` refuses to compare anything if it does not reproduce.

TWO ARMS, and the second is not optional
----------------------------------------
At initialisation the ANN's final layer is zero, so its output is exactly zero. That is the
production starting point and it is arm 1. But a zero output layer also zeroes the gradient into
every layer behind it (measured: 852 of 3616 entries nonzero), so a gradient recorded there is
partly a vector of structural zeros and would not notice a change in those paths. Arm 2 adds a
fixed seeded perturbation to the ANN parameters (1600 of 3616 nonzero) and records the same
quantities. A seam that is inert on arm 1 and not on arm 2 is still broken.

The loop is active in BOTH arms: with the ANN at zero the augmented model is the baseline, which
does not reproduce the data (the absorber is missing from it), so `y_data - y_model` is nonzero
and `Cfb` is genuinely in the loop.

DETERMINISM
-----------
Bit-identity is only meaningful against a fixed reduction order, and torch's threaded reductions
do not guarantee one across thread counts. The thread count is pinned (CL_THREADS, default 1) and
stored in the manifest; `--check` refuses to compare across a different value, because that is
comparing two arithmetic orders and calling the difference a regression.

Usage:
  python -u cl_step1_reference.py              record  -> references/step1_reference.{json,npz}
  python -u cl_step1_reference.py --check      compare the current code against that recording
  CL_SKIP_VAL=1 ...                            skip the four full-length validation free runs
"""
__project_origin__ = "added"

import copy
import dataclasses
import hashlib
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

import deepSI                                                             # noqa: E402
import demo_common as dm                                                  # noqa: E402
from demo_common import CFG                                               # noqa: E402
from gantry_dynamic.data import load_traj, TRAIN_FILES, VAL_FILES         # noqa: E402
import cl_plant as PLANT                                                  # noqa: E402
import cl_validation as CV                                                # noqa: E402
from model_augmentation.fit_systems.closed_loop import window_controller_index  # noqa: E402

CHECK = '--check' in sys.argv
THREADS = int(os.environ.get('CL_THREADS', 1))
torch.set_num_threads(THREADS)
SKIP_VAL = bool(int(os.environ.get('CL_SKIP_VAL', 0)))
# Four records, three distinct controllers (Y_op -0.30 / +0.30 / 0.00), eight windows each. The
# batch has to span several records or it cannot detect a ctrl_ix/gather regression at all, which
# is precisely the failure mode plan 3.5 warns about and the one gate A cannot see.
BATCH_RECORDS = ['T1_standstill_Ym30', 'T5_standstill_Yp30', 'T10_aprbs_60', 'T13_lissajous']
WIN_PER_RECORD = 8
PERTURB_SIGMA = 1e-2
# The CLOSED-LOOP perturbed arm needs a SMALLER perturbation, and this is measured rather than
# tuned. At sigma = 1e-2 the closed-loop rollout is chaotic: the same implementation gives loss
# 4.81e-02 in float32 and 2.29e-03 in float64, a factor 20, and two implementations that agree to
# machine precision elsewhere give gradients with 1 - cos = 1.87 IN FLOAT64. A quantity that two
# precisions of the same code disagree on by 20x is not a regression net, it is a coin flip.
# Sweeping the amplitude (float64, old versus new implementation):
#     sigma 1e-2   1 - cos 1.87        loss rel 1.3e-03     unusable
#     sigma 1e-3   1 - cos 4.6e-05     loss rel 3.9e-08
#     sigma 1e-4   1 - cos 8.9e-16     loss rel 5.4e-14     machine precision
# 1e-4 is used: it still perturbs every ANN parameter, so no gradient path is structurally dead,
# and the closed-loop loss is 4.57e-06 against 2.54e-10 at zero, i.e. four orders above the
# zero-ANN arm, so the ANN is unmistakably active. The open-loop arms (R1, D) keep 1e-2, where
# there is no loop gain to amplify and the numbers are reproducible.
PERTURB_SIGMA_CL = 1e-4
PERTURB_SEED = 12345
# Baseline D. nf must divide by n_seg and each segment must be at least max(na, nb) = 17 long, so
# 4 x 100 is the coarsest split that exercises three interior boundaries. Both weights are nonzero
# because the guard in SSE_Interconnect_MultipleShooting.loss covers BOTH: with either at zero the
# segmented path is never entered and D would silently record the unsegmented loss.
D_N_SEG, D_DEFECT_W, D_DEFECT_ACC_W = 4, 1.0, 1.0
# Baseline E. The probe's cost is one n_step_error pass per call over each record, so the records
# are truncated. Valid because the comparison is before-versus-after at the SAME configuration.
E_TRUNC, E_CALLS = 6000, 2
OUTDIR = os.path.join(HERE, 'references')
REF_JSON = os.path.join(OUTDIR, 'step1_reference.json')
REF_NPZ = os.path.join(OUTDIR, 'step1_reference.npz')
t0 = time.time()

print('=' * 96)
print('MIGRATION STEP 1: reference sets  %s'
      % ('CHECK against the recording' if CHECK else 'RECORD'))
print('=' * 96)
print('torch %s   threads %d (pinned)   dtype float32' % (torch.__version__, THREADS))

cfg = dataclasses.replace(CFG, seed=0, ann_route_ix=tuple(range(8)), lr=1e-7)
fs, norm, K0, na, nb, na_r, nb_r = dm.build_pipeline(cfg=cfg, verbose=True)
nx = cfg.nx_phys + cfg.nx_ann
print('nf %d   nx %d   n_seg %d   orth_beta %g   orth_penalty %s'
      % (cfg.nf, nx, cfg.n_seg, cfg.orth_beta,
         'attached' if getattr(fs, 'orth_penalty', None) is not None else 'None'))
print('production fit-system class: %s' % type(fs).__name__)

PARAMS = list(fs.hfn.parameters()) + list(fs.encoder.parameters())


def param_hash():
    """sha256 over every trainable tensor's raw bytes, in a fixed order. The build fingerprint."""
    h = hashlib.sha256()
    for p in PARAMS:
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def arr_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


H_BUILD = param_hash()
print('build fingerprint (sha256 over %d parameter tensors): %s' % (len(PARAMS), H_BUILD))

# ---- the eval stack, exactly as cl_step6_run.py builds it ------------------------------------
train_names = [f[:-4] for f in TRAIN_FILES]
val_names = [f[:-4] for f in VAL_FILES]

# ---- the fixed batch -------------------------------------------------------------------------
rv = lambda a: np.asarray(a).ravel()                                      # noqa: E731
uh, yh, uf, yf, rix = [], [], [], [], []
starts_log = {}
for nm_rec in BATCH_RECORDS:
    sd = load_traj(nm_rec + '.mat', cfg)
    un = ((sd.u - rv(fs.norm.u0)) / rv(fs.norm.ustd)).astype(cfg.dtype_np)
    yn = ((sd.y - rv(fs.norm.y0)) / rv(fs.norm.ystd)).astype(cfg.dtype_np)
    ss_ = np.linspace(K0, len(un) - cfg.nf - 2, WIN_PER_RECORD).astype(int)
    starts_log[nm_rec] = [int(v) for v in ss_]
    for s in ss_:
        a, b = CV.encoder_window(un, yn, int(s), na, nb, na_r, nb_r, cfg.dtype_pt)
        uh.append(a); yh.append(b)
        uf.append(un[s:s + cfg.nf])
        yf.append(yn[s:s + cfg.nf])
        rix.append(train_names.index(nm_rec))
T = lambda A: torch.as_tensor(np.stack(A), dtype=cfg.dtype_pt)            # noqa: E731
UH, YH, UF, YF = torch.cat(uh, 0), torch.cat(yh, 0), T(uf), T(yf)
RIX = torch.tensor(rix, dtype=torch.long)
print('batch: %d windows over %d records, ufuture %s'
      % (UF.shape[0], len(BATCH_RECORDS), tuple(UF.shape)))


def value_and_grad(fn):
    """(loss value, flat float32 gradient over every trainable tensor) for a loss callable."""
    for p in PARAMS:
        p.grad = None
    L = fn()
    L.backward()
    g = torch.cat([(p.grad if p.grad is not None
                    else torch.zeros_like(p)).detach().reshape(-1) for p in PARAMS])
    return float(L.detach()), g.numpy().copy()


def report(tag, val, grad):
    print('  %-34s loss %.12e   |g| %.9e   sha %s'
          % (tag, val, float(np.linalg.norm(grad.astype(np.float64))), arr_hash(grad)[:16]))


store, scal = {}, {}

# =============================================================================================
# R1: the production path. No closed loop, no simulator, nothing attached.
# =============================================================================================
print('\nR1  PRODUCTION LOSS, closed loop absent')
v, g = value_and_grad(lambda: fs.loss(UH, YH, UF, YF))
report('production loss, ANN at zero', v, g)
store['R1_grad_zero'] = g
scal['R1_loss_zero'] = v

# a short open-loop rollout of the production configuration, kept as a separate witness: a loss
# is one number and can agree by accident, a trajectory cannot.
with torch.no_grad():
    x0 = fs.encoder(UH, YH)
    yop, _ = fs.simulate(x0, UF, YF)          # simulator absent: the open-loop default
store['R1_ypred_zero'] = yop.numpy().copy()
store['R1_x0_zero'] = x0.numpy().copy()
print('  %-34s sha %s' % ('open-loop y_pred', arr_hash(store['R1_ypred_zero'])[:16]))

ann_state = copy.deepcopy(fs.hfn.state_dict())


def perturb_ann(sigma=PERTURB_SIGMA):
    """Fixed, seeded perturbation of every ANN parameter. Reproducible from the manifest."""
    from model_augmentation.fit_systems.blocks import Static_ANN_Block
    ann = next(m for m in fs.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
    g_ = torch.Generator().manual_seed(PERTURB_SEED)
    n = 0
    with torch.no_grad():
        for p in ann.parameters():
            p.add_(torch.randn(p.shape, generator=g_, dtype=p.dtype) * sigma)
            n += p.numel()
    return n


def restore_ann():
    fs.hfn.load_state_dict(ann_state)
    assert param_hash() == H_BUILD, 'restoring the ANN did not reproduce the build fingerprint'


n_pert = perturb_ann()
v, g = value_and_grad(lambda: fs.loss(UH, YH, UF, YF))
report('production loss, ANN perturbed', v, g)
store['R1_grad_pert'] = g
scal['R1_loss_pert'] = v
print('  nonzero gradient entries: zero-ANN arm %d / %d, perturbed arm %d / %d'
      % (int((store['R1_grad_zero'] != 0).sum()), store['R1_grad_zero'].size,
         int((g != 0).sum()), g.size))
restore_ann()

# =============================================================================================
# D: the SEGMENTED loss, n_seg > 1, both defect terms live. Recorded BEFORE attach, because
# ClosedLoopLoss.loss overrides SSE_Interconnect_MultipleShooting.loss in the MRO and the
# segmented path would never run.
# =============================================================================================
print('\nD   SEGMENTED LOSS, n_seg = %d, defect_weight = %g, defect_acc_weight = %g'
      % (D_N_SEG, D_DEFECT_W, D_DEFECT_ACC_W))

# D1. FINDING, and the reason D exists in two parts. Before the contiguity fix in
# `multiple_shooting.py`, the segmented path did not run AT ALL, with either encoder:
# `SSE_Interconnect_MultipleShooting.loss` builds each interior node with
# `self.encoder(ufuture[:, s-nb : s+nb_right], ...)`, and a time-axis slice of a contiguous
# (batch, nf, nu) tensor keeps dim-0 stride nf*nu, i.e. it is not contiguous. Both encoders then
# reshape with `.view`, which requires contiguity:
#     encoder_init='linear_map' (production) -> RuntimeError at pre_encoder.py:450
#     encoder_init='default'                 -> RuntimeError at interconnect.py:384
# deepSI's `to_hist_future_data` hands the encoder contiguous windows, so the n_seg = 1 path never
# touches this and nothing in production ever noticed. It does mean the defect diagnostics that
# plan rule 2 cites as the reason to keep `multiple_shooting.py` were not runnable through this
# method, and that migration step 4's stated gate ("assert n_seg > 1 against the pre-change defect
# diagnostics") had nothing to compare against. Fixed at the caller, marked `# CHANGED (contiguity)`,
# which does not touch the n_seg = 1 path and so cannot move R1.
#
# D1 below re-runs the production config and records what it now does, so a later step that
# reintroduces the contiguity fault is caught rather than rediscovered.
_keep = (fs.n_seg, fs.defect_weight, fs.defect_acc_weight)
fs.n_seg, fs.defect_weight, fs.defect_acc_weight = D_N_SEG, D_DEFECT_W, D_DEFECT_ACC_W
try:
    for arm in ('zero', 'pert'):
        if arm == 'pert':
            perturb_ann()
        v, g = value_and_grad(lambda: fs.loss(UH, YH, UF, YF))
        store['D1_grad_%s' % arm] = g
        scal['D1_loss_%s' % arm] = v
        scal['D1_mse_%s' % arm] = float(fs.last_mse)
        scal['D1_defect_rms_%s' % arm] = float(fs.last_defect_rms)
        scal['D1_defect_acc_%s' % arm] = float(fs.last_defect_acc)
        print('  D1 linear_map, ANN %-5s loss %.12e  mse %.12e  defect_rms %.9e  '
              'defect_acc %.9e' % (arm, v, fs.last_mse, fs.last_defect_rms, fs.last_defect_acc))
        if arm == 'pert':
            restore_ann()
    scal['D1_production_status'] = 'ran'
except Exception as exc:
    tb = sys.exc_info()[2]
    while tb.tb_next is not None:
        tb = tb.tb_next
    scal['D1_production_status'] = '%s at %s:%d' % (
        type(exc).__name__, os.path.basename(tb.tb_frame.f_code.co_filename), tb.tb_lineno)
    print('  D1 production config (linear_map encoder): %s' % scal['D1_production_status'])
    restore_ann()
fs.n_seg, fs.defect_weight, fs.defect_acc_weight = _keep
for p in PARAMS:
    p.grad = None

# D2. A config where the segmented path DOES run, so step 4 has numbers and not just an exception.
# `encoder_init = 'default'` is deepSI's learned encoder, which does not reshape its input, and
# na_right = nb_right = 0. orth_observe is off so this build does not trigger a fresh penalty-basis
# Jacobian; the penalty contributes nothing at orth_beta = 0 either way, so the segmented loss is
# unaffected. This is a self-contained before/after baseline with its own build fingerprint.
print('  D2 building the default-encoder arm ...', flush=True)
# nf_override sets nf_seg, and RunConfig.nf is n_seg * nf_seg, so without it n_seg = 4 would give
# nf = 1600 rather than 400 and D2 would not be scored on the same window length as R1 and R2.
cfg2 = dataclasses.replace(CFG, seed=0, ann_route_ix=tuple(range(8)), lr=1e-7,
                           encoder_init='default', orth_observe=False,
                           n_seg=D_N_SEG, nf_override=CFG.nf // D_N_SEG,
                           defect_weight=D_DEFECT_W, defect_acc_weight=D_DEFECT_ACC_W)
fs2, norm2, K02, na2, nb2, na_r2, nb_r2 = dm.build_pipeline(cfg=cfg2, verbose=False)
PARAMS2 = list(fs2.hfn.parameters()) + list(fs2.encoder.parameters())
h2 = hashlib.sha256()
for p in PARAMS2:
    h2.update(p.detach().cpu().numpy().tobytes())
H_BUILD2 = h2.hexdigest()
uh2, yh2, uf2, yf2 = [], [], [], []
for nm_rec in BATCH_RECORDS:
    sd = load_traj(nm_rec + '.mat', cfg2)
    un = ((sd.u - rv(fs2.norm.u0)) / rv(fs2.norm.ustd)).astype(cfg2.dtype_np)
    yn = ((sd.y - rv(fs2.norm.y0)) / rv(fs2.norm.ystd)).astype(cfg2.dtype_np)
    for s in starts_log[nm_rec]:
        a, b = CV.encoder_window(un, yn, int(s), na2, nb2, na_r2, nb_r2, cfg2.dtype_pt)
        uh2.append(a); yh2.append(b)
        uf2.append(un[s:s + cfg2.nf])
        yf2.append(yn[s:s + cfg2.nf])
UH2, YH2, UF2, YF2 = (torch.cat(uh2, 0), torch.cat(yh2, 0),
                      T(uf2).to(cfg2.dtype_pt), T(yf2).to(cfg2.dtype_pt))
ann2_state = copy.deepcopy(fs2.hfn.state_dict())
print('  D2 n_seg %d  nf_seg %d  encoder %s  fingerprint %s'
      % (fs2.n_seg, cfg2.nf // fs2.n_seg, cfg2.encoder_init, H_BUILD2[:16]))


def value_and_grad2(fn):
    for p in PARAMS2:
        p.grad = None
    L = fn()
    L.backward()
    g = torch.cat([(p.grad if p.grad is not None
                    else torch.zeros_like(p)).detach().reshape(-1) for p in PARAMS2])
    return float(L.detach()), g.numpy().copy()


for arm in ('zero', 'pert'):
    if arm == 'pert':
        from model_augmentation.fit_systems.blocks import Static_ANN_Block
        _ann2 = next(m for m in fs2.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
        _g2 = torch.Generator().manual_seed(PERTURB_SEED)
        with torch.no_grad():
            for p in _ann2.parameters():
                p.add_(torch.randn(p.shape, generator=_g2, dtype=p.dtype) * PERTURB_SIGMA)
    v, g = value_and_grad2(lambda: fs2.loss(UH2, YH2, UF2, YF2))
    store['D_grad_%s' % arm] = g
    scal['D_loss_%s' % arm] = v
    scal['D_mse_%s' % arm] = float(fs2.last_mse)
    scal['D_defect_rms_%s' % arm] = float(fs2.last_defect_rms)
    scal['D_defect_acc_%s' % arm] = float(fs2.last_defect_acc)
    print('  segmented loss, ANN %-5s loss %.12e  |g| %.9e' % (arm, v, float(np.linalg.norm(
        g.astype(np.float64)))))
    print('  %-24s mse %.12e  defect_rms %.9e  defect_acc %.9e'
          % ('', fs2.last_mse, fs2.last_defect_rms, fs2.last_defect_acc))
    if arm == 'pert':
        fs2.hfn.load_state_dict(ann2_state)
del fs2, PARAMS2

# =============================================================================================
# E: the nf-probe histories. The probe is driven directly with a stub selector, which is the
# right isolation: migration step 2d changes only HOW the probe is invoked, not what it records,
# and the real selector is exercised by R2 and by the short fit in step 2c's gate.
# =============================================================================================
print('\nE   NF-PROBE HISTORIES, %d calls on %d-sample records' % (E_CALLS, E_TRUNC))
from gantry_dynamic.training import _NfProbe                              # noqa: E402


def _trunc(sd, n):
    return deepSI.System_data(u=np.array(sd.u[:n]), y=np.array(sd.y[:n]), dt=sd.dt)


e_train = _trunc(load_traj(TRAIN_FILES[0], cfg), E_TRUNC)
e_val = _trunc(load_traj(VAL_FILES[0], cfg), E_TRUNC)
probe = _NfProbe(fs, cfg.nf, e_train, e_val, do_print=False)
for _ in range(E_CALLS):
    # The validation_probes signature: (fit_sys, val_sys_data, value), side effects only. Before
    # migration step 2d this was a monkey patch called as (val_sys_data, validation_measure=...)
    # with a stub selector standing in for the wrapped original. What the probe RECORDS is
    # invocation-independent, which is exactly why the recorded histories still have to match.
    probe(fs, deepSI.System_data_list([e_val]), 0.0)
for key in ('Loss_train_nf', 'Loss_val_nf', 'Probe_combo_err', 'Probe_orth_frac',
            'Probe_V_orth', 'Probe_param_loss'):
    vals = np.asarray(getattr(fs, key, []), dtype=float)
    store['E_%s' % key] = vals
    print('  %-22s %s' % (key, ' '.join('%.9e' % v for v in vals)))

# =============================================================================================
# R2: the closed loop as it exists today.
# =============================================================================================
print('\nR2  THE CLOSED LOOP')
# MIGRATION step 7: this used to graft `cl_fitsys.ClosedLoopLoss` onto the instance with
# `fit_sys.__class__ = type(...)`. It is now one assignment of a declared attribute. The
# old-versus-new comparison that mattered is frozen in `cl_test_closed_loop.py`, which measured
# the framework implementation against the recording made before any edit: loss rel 1.4e-05,
# gradient 1 - cos 5.3e-09, trajectory 2.8e-07, selection scalar within 1e-10 m.
from cl_pipeline import build_closed_loop                                 # noqa: E402
val_data = deepSI.System_data_list([load_traj(f, cfg) for f in VAL_FILES])
fs.simulator = build_closed_loop(fs, norm, cfg, train_files=TRAIN_FILES, val_files=VAL_FILES,
                                   val_data=val_data, verbose=False)
CTRL_IX = torch.tensor([fs.simulator.train_ctrl_rows[i] for i in RIX.tolist()], dtype=torch.long)
print('  simulator: %s over %d controllers, class attribute not a grafted type'
      % (type(fs.simulator).__name__, fs.simulator.bank.n_controllers))

v, g = value_and_grad(lambda: fs.loss(UH, YH, UF, YF, ctrl_ix=CTRL_IX))
report('closed-loop loss, ANN at zero', v, g)
store['R2_grad_zero'] = g
scal['R2_loss_zero'] = v

# The G11 contract used to need a `loss_open_loop` twin of the closed-loop loss, to prove the
# closed-loop file had not changed anything except the loop. It is now STRUCTURAL: with
# `simulator = None` the very same `loss()` IS the production loss, so R1 above already is that
# statement and a separate twin would only be testing that None means None.
fs.simulator, _sim = None, fs.simulator
v_off, _ = value_and_grad(lambda: fs.loss(UH, YH, UF, YF))
fs.simulator = _sim
print('  G11 contract, now structural: loss with simulator = None - R1 = %+.3e'
      % (v_off - scal['R1_loss_zero']))
scal['R2_loss_simulator_off'] = v_off

with torch.no_grad():
    x0 = fs.encoder(UH, YH)
    ycl, xend = fs.simulate(x0, UF, YF, ctrl_ix=CTRL_IX)
store['R2_ypred_zero'] = ycl.numpy().copy()
store['R2_xfinal_zero'] = xend.numpy().copy()
print('  %-34s sha %s' % ('closed-loop y_pred', arr_hash(store['R2_ypred_zero'])[:16]))

perturb_ann(PERTURB_SIGMA_CL)          # smaller, see PERTURB_SIGMA_CL: 1e-2 is chaotic in the loop
v, g = value_and_grad(lambda: fs.loss(UH, YH, UF, YF, ctrl_ix=CTRL_IX))
report('closed-loop loss, ANN perturbed', v, g)
store['R2_grad_pert'] = g
scal['R2_loss_pert'] = v
restore_ann()

# ---- the units gate and the record-index derivation ------------------------------------------
u_norm, u_expect, u_rel = fs.simulator.bank.check_units(ctrl_ix=0)
print('\n  units gate  rel err [%s]' % ' '.join('%.3e' % v for v in u_rel))
scal['units_rel_max'] = float(np.max(u_rel))
store['units_u_norm'] = np.asarray(u_norm, dtype=np.float64)

# Per-record window counts, checked PER RECORD against the real to_hist_future_data rather than
# against their sum: today's code asserts only sum(counts) == len(data[0]), which two compensating
# errors would pass. One record at a time, because the concatenated training set is ~700 MB of
# window copies and none of it is needed to count.
train_list = [load_traj(f, cfg) for f in TRAIN_FILES]
derived, counts = window_controller_index(
    deepSI.System_data_list(train_list), fs.simulator.train_ctrl_rows,
    na, nb, cfg.nf, na_r, nb_r, cfg.stride)
real = []
for sd in train_list:
    hf = sd.to_hist_future_data(na=na, nb=nb, na_right=na_r, nb_right=nb_r,
                                nf=cfg.nf, stride=cfg.stride)
    real.append(len(hf[0]))
    del hf
ok_counts = list(counts) == real
print('  window counts per record  derived == actual: %s  (%d windows)'
      % (ok_counts, sum(real)))
assert ok_counts, 'per-record window counts disagree; _record_index is wrong for this config'
scal['n_windows'] = int(sum(real))
store['cl_counts'] = np.asarray(real, dtype=np.int64)
# Renamed from cl_rec_ix at migration step 7: it holds CONTROLLER ROWS in the one
# global bank, not positions in a per-split record list. Same plumbing, different
# numbers, and the rename is so a comparison against the old recording fails loudly
# on the name rather than quietly on the values.
store['cl_ctrl_ix'] = derived.astype(np.int64)

# ---- the selection scalar --------------------------------------------------------------------
if SKIP_VAL:
    print('\n  validation free runs SKIPPED (CL_SKIP_VAL=1)')
else:
    print('\n  four full-length closed-loop validation free runs ...', flush=True)
    # Through the SEAM, not through a patched attribute: this is the same call fit() makes to
    # decide checkpoints, so what is recorded here is the selection scalar itself.
    sel = fs.cal_validation_error(val_data, validation_measure='sim-RMS')
    per_rec = fs.simulator.last_per_record
    scal['R2_selection_untrained'] = sel
    store['R2_selection_per_record'] = np.asarray(per_rec, dtype=np.float64)
    print('  selection scalar (untrained, closed loop) %.12e m' % sel)

# =============================================================================================
# Record, or check against the recording.
# =============================================================================================
# Per-key comparison class. The point of naming these individually is that a step which legally
# reorders one quantity must not silently widen the gate on every other one.
#   exact    bit-identical. Anything on the production path through migration step 2a.
#   reorder  the same arithmetic in a different order: bounded by what step 1 measured for
#            exactly that change (1 - cos 6.1e-15, one float32 ulp elementwise).
#   loose    plan 5.1's tolerances for a genuinely re-implemented closed loop.
#   sel      the validation selection scalar, in metres.
CLASS = {'exact': dict(rel=0.0, one_minus_cos=0.0),
         'reorder': dict(rel=1e-6, one_minus_cos=1e-13),
         'loose': dict(rel=1e-3, one_minus_cos=1e-5),
         'sel': dict(abs_m=1e-10)}
# Default classes. A later step that legally changes one of these passes CL_RELAX="key=class,..."
# so the widening is visible on the command line and in the log rather than edited into the file.
DEFAULT_CLASS = {}
for k in ('R1_loss_zero', 'R1_loss_pert', 'R1_grad_zero', 'R1_grad_pert',
          'R1_ypred_zero', 'R1_x0_zero',
          'D_loss_zero', 'D_loss_pert', 'D_grad_zero', 'D_grad_pert',
          'D_mse_zero', 'D_mse_pert', 'D_defect_rms_zero', 'D_defect_rms_pert',
          'D_defect_acc_zero', 'D_defect_acc_pert', 'D1_production_status',
          'D1_loss_zero', 'D1_loss_pert', 'D1_grad_zero', 'D1_grad_pert',
          'D1_mse_zero', 'D1_mse_pert', 'D1_defect_rms_zero', 'D1_defect_rms_pert',
          'D1_defect_acc_zero', 'D1_defect_acc_pert',
          'E_Loss_train_nf', 'E_Loss_val_nf', 'E_Probe_combo_err', 'E_Probe_orth_frac',
          'E_Probe_V_orth', 'E_Probe_param_loss',
          'cl_counts', 'cl_ctrl_ix', 'n_windows', 'units_rel_max', 'units_u_norm'):
    DEFAULT_CLASS[k] = 'exact'
for k in ('R2_loss_zero', 'R2_loss_pert', 'R2_mse_zero', 'R2_grad_zero', 'R2_grad_pert',
          'R2_lossOL_zero', 'R2_lossOL_pert', 'R2_gradOL_zero', 'R2_gradOL_pert',
          'R2_ypred_zero', 'R2_xfinal_zero', 'R2_xcfinal_zero'):
    DEFAULT_CLASS[k] = 'loose'
for k in ('R2_selection_untrained', 'R2_selection_per_record'):
    DEFAULT_CLASS[k] = 'sel'

manifest = dict(
    torch=torch.__version__, numpy=np.__version__, threads=THREADS,
    dtype='float32', seed=cfg.seed, nf=cfg.nf, stride=cfg.stride,
    ann_route_ix=list(cfg.ann_route_ix), n_seg=cfg.n_seg, orth_beta=cfg.orth_beta,
    fit_system_class_production='SSE_Interconnect_MultipleShooting',
    batch_records=BATCH_RECORDS, win_per_record=WIN_PER_RECORD, starts=starts_log,
    perturb_sigma=PERTURB_SIGMA, perturb_sigma_closed_loop=PERTURB_SIGMA_CL,
    perturb_seed=PERTURB_SEED, n_perturbed_params=n_pert,
    d_config=dict(n_seg=D_N_SEG, defect_weight=D_DEFECT_W, defect_acc_weight=D_DEFECT_ACC_W),
    e_config=dict(trunc=E_TRUNC, calls=E_CALLS),
    build_param_sha256=H_BUILD, d2_build_param_sha256=H_BUILD2,
    batch_u_sha256=arr_hash(UF.numpy()), batch_y_sha256=arr_hash(YF.numpy()),
    array_sha256={k: arr_hash(v) for k, v in store.items()},
    scalars=scal, classes=DEFAULT_CLASS, seconds=time.time() - t0)


def compare():
    """Compare what this run computed against the recording. Returns True if everything passes."""
    ref = json.load(open(REF_JSON))
    refa = np.load(REF_NPZ)
    print('\n' + '=' * 96)
    print('CHECK against %s' % os.path.basename(REF_JSON))
    print('=' * 96)
    if ref['threads'] != THREADS:
        sys.exit('recorded at %d threads, running at %d: that compares two arithmetic orders '
                 'and calls the difference a regression. Set CL_THREADS=%d.'
                 % (ref['threads'], THREADS, ref['threads']))
    if ref['build_param_sha256'] != H_BUILD:
        sys.exit('BUILD FINGERPRINT CHANGED\n  recorded %s\n  now      %s\nThe parameters are '
                 'not the ones the reference was taken on, so no loss comparison below would '
                 'mean anything. Fix the build before reading any other number.'
                 % (ref['build_param_sha256'], H_BUILD))
    print('build fingerprint reproduced, threads match')
    relax = {}
    for item in filter(None, os.environ.get('CL_RELAX', '').split(',')):
        k, _, c = item.partition('=')
        relax[k.strip()] = c.strip()
    if relax:
        print('RELAXED by CL_RELAX: %s' % relax)

    rows, ok_all = [], True
    for key in sorted(set(list(scal) + list(store))):
        cls = relax.get(key, DEFAULT_CLASS.get(key, 'loose'))
        tol = CLASS[cls]
        if key in scal:
            if key not in ref['scalars']:
                rows.append((key, cls, 'MISSING in reference', True)); continue
            if isinstance(scal[key], str) or isinstance(ref['scalars'][key], str):
                ok = scal[key] == ref['scalars'][key]
                rows.append((key, cls, ('same: %s' % scal[key]) if ok
                             else ('%r vs recorded %r' % (scal[key], ref['scalars'][key])), ok))
                ok_all &= ok
                continue
            a, b = float(scal[key]), float(ref['scalars'][key])
            if np.isnan(a) or np.isnan(b):
                ok = np.isnan(a) and np.isnan(b)
                rows.append((key, cls, 'nan on both sides' if ok else 'nan mismatch', ok))
                ok_all &= ok
                continue
            if 'abs_m' in tol:
                d, ok = abs(a - b), abs(a - b) <= tol['abs_m']
                detail = '|d| %.3e m' % d
            else:
                d = abs(a - b) / max(abs(b), 1e-300)
                ok = (a == b) if tol['rel'] == 0.0 else d <= tol['rel']
                detail = 'rel %.3e%s' % (d, '' if a != b else ' (bit-identical)')
        else:
            if key not in refa.files:
                rows.append((key, cls, 'MISSING in reference', True)); continue
            a, b = np.asarray(store[key]), np.asarray(refa[key])
            if a.shape != b.shape:
                rows.append((key, cls, 'SHAPE %s vs %s' % (a.shape, b.shape), False))
                ok_all = False
                continue
            # equal_nan: the probe records nan on purpose (combo-err is n/a with theta frozen,
            # orth-frac is n/a while the ANN output is identically zero). nan != nan would make
            # those rows fail forever and train everyone to ignore the table.
            if np.array_equal(a, b, equal_nan=not np.issubdtype(a.dtype, np.integer)):
                detail, ok = 'bit-identical', True
            elif 'abs_m' in tol:
                d = float(np.abs(a.astype(float) - b.astype(float)).max())
                ok = d <= tol['abs_m']
                detail = 'max |d| %.3e m' % d
            else:
                af, bf = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
                nrm = max(np.abs(bf).max(), 1e-300)
                d = float(np.abs(af - bf).max() / nrm)
                den = np.linalg.norm(af) * np.linalg.norm(bf)
                omc = 1.0 - float(af @ bf / den) if den > 0 else 0.0
                ok = (tol['rel'] > 0.0 and d <= tol['rel']
                      and abs(omc) <= tol['one_minus_cos'])
                detail = 'max rel %.3e  1-cos %.3e' % (d, omc)
        ok_all &= ok
        rows.append((key, cls, detail, ok))

    for key, cls, detail, ok in rows:
        print('  %-28s %-8s %-42s %s' % (key, cls, detail, 'PASS' if ok else 'FAIL'))

    # Keys the RECORDING has but this run did not compute. Without this they vanish from the
    # table and a check that quietly stopped running reads exactly like a check that passed.
    computed = set(scal) | set(store)
    dropped = sorted((set(ref['scalars']) | set(refa.files)) - computed)
    if dropped:
        print('\n  NOT COMPUTED THIS RUN (present in the recording): %s' % ', '.join(dropped))
        print('  Each is either a deliberate removal, in which case re-record, or a check that '
              'silently stopped running.')
    print('=' * 96)
    print('OVERALL: %s' % ('PASS' if ok_all else 'FAIL'))
    return ok_all


if CHECK:
    sys.exit(0 if compare() else 1)

os.makedirs(OUTDIR, exist_ok=True)
np.savez(REF_NPZ, **store)
json.dump(manifest, open(REF_JSON, 'w'), indent=2)
print('\nwrote %s' % os.path.join(OUTDIR, 'step1_reference.{json,npz}'))
print('[%.0fs]' % (time.time() - t0))
