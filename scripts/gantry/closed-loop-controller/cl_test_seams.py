"""MIGRATION STEPS 2a-2d and 4 gate: do the four seams BEHAVE, not just produce the same numbers?

`cl_step1_reference.py --check` proves the seams are numerically inert. That is necessary and not
sufficient: a seam that is never reached is also inert. These checks prove each one is actually
wired, that its default is a no-op, and that the one combination whose semantics are undecided is
refused rather than guessed.

  S1  simulate() delegates       a simulator on the instance IS consulted, and its return value
                                 is what the loss uses. Otherwise the closed loop would train the
                                 open-loop objective and every number would still look plausible.
  S2  simulate() contract        returns (y_pred, x_final), and x_final is the state the rollout
                                 actually ended on. Multiple shooting forms its defects from it.
  S3  make_training_data seam    with no simulator, the arrays are deepSI's unchanged; with one,
                                 augment_training_data is called and its extra array arrives.
  S4  cal_validation_error seam  with no simulator, deepSI's own value; with one, the simulator's.
  S5  validation_probes          probes are called with (fit_sys, val_sys_data, value) and CANNOT
                                 change the returned value. This is the property that removes the
                                 old ordering hazard, where whichever monkey patch was installed
                                 last decided checkpoint selection.
  S6  multiple shooting guard    n_seg > 1 with a simulator attached raises, because whether the
                                 simulator resets its own state at a segment boundary is an open
                                 modelling question. A silently wrong combination is worse than an
                                 unsupported one.
  S7  no monkey patches left     the greps from plan 5.1, over the production path.

Usage:
  PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python -u cl_test_seams.py
"""
__project_origin__ = "added"

import dataclasses
import os
import subprocess
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
from gantry_dynamic.data import load_traj, TRAIN_FILES                    # noqa: E402
import cl_validation as CV                                                # noqa: E402

torch.set_num_threads(int(os.environ.get('CL_THREADS', 1)))
# These checks are about wiring, not about the model, so keep them cheap. The floor is set by S6:
# a segment must be at least max(na, nb) = 17 long or SSE_Interconnect_MultipleShooting rejects
# the configuration before reaching anything this file is testing, so nf >= 4 * 17.
NF_SMALL = 80
BATCH = 4

print('=' * 96)
print('MIGRATION STEPS 2a-2d and 4: do the seams behave?')
print('=' * 96)
cfg = dataclasses.replace(CFG, seed=0, ann_route_ix=tuple(range(8)), lr=1e-7)
fs, norm, K0, na, nb, na_r, nb_r = dm.build_pipeline(cfg=cfg, verbose=False)
nx, nu, ny = cfg.nx_phys + cfg.nx_ann, cfg.nu, cfg.ny
ok = True


def check(tag, passed, detail=''):
    global ok
    ok &= bool(passed)
    print('   %-52s %s %s' % (tag, 'PASS' if passed else 'FAIL', detail))


# ---- a small fixed batch ----------------------------------------------------------------------
rv = lambda a: np.asarray(a).ravel()                                      # noqa: E731
sd = load_traj(TRAIN_FILES[0], cfg)
un = ((sd.u - rv(fs.norm.u0)) / rv(fs.norm.ustd)).astype(cfg.dtype_np)
yn = ((sd.y - rv(fs.norm.y0)) / rv(fs.norm.ystd)).astype(cfg.dtype_np)
uh, yh, uf, yf = [], [], [], []
for s in np.linspace(K0, 5000, BATCH).astype(int):
    a, b = CV.encoder_window(un, yn, int(s), na, nb, na_r, nb_r, cfg.dtype_pt)
    uh.append(a); yh.append(b)
    uf.append(un[s:s + NF_SMALL])
    yf.append(yn[s:s + NF_SMALL])
T = lambda A: torch.as_tensor(np.stack(A), dtype=cfg.dtype_pt)            # noqa: E731
UH, YH, UF, YF = torch.cat(uh, 0), torch.cat(yh, 0), T(uf), T(yf)


class _SpySimulator:
    """Minimal stand-in for a driving strategy: records that it was reached, returns a marked
    prediction so the loss cannot accidentally agree with the open-loop one."""

    def __init__(self):
        self.calls = 0
        self.saw_kwargs = None

    def __call__(self, fit_sys, x, ufuture, yfuture, **kw):
        self.calls += 1
        self.saw_kwargs = kw
        y_pred, x_final = super_simulate(fit_sys, x, ufuture, yfuture)
        return y_pred + 1.0, x_final          # marked, so agreement cannot be accidental

    def validation_error(self, fit_sys, val_sys_data, validation_measure='sim-RMS'):
        self.calls += 1
        return -12345.0                        # unmistakable, and not a plausible RMS

    def augment_training_data(self, data, sys_data, fit_sys, **kw):
        self.calls += 1
        return list(data) + [np.arange(len(data[0]), dtype=np.int64)]


def super_simulate(fit_sys, x, ufuture, yfuture):
    """The open-loop rollout, called directly so the spy can mark its output."""
    hfn = fit_sys.hfn
    ys = []
    for u in ufuture.unbind(1):
        yhat, x = hfn(x, u)
        ys.append(yhat)
    return torch.stack(ys, dim=1), x


# ---- S1 / S2 ----------------------------------------------------------------------------------
print('\nS1/S2  simulate(): delegation and the (y_pred, x_final) contract')
with torch.no_grad():
    x0 = fs.encoder(UH, YH)
    out = fs.simulate(x0, UF, YF, nf=NF_SMALL)
check('default returns a 2-tuple', isinstance(out, tuple) and len(out) == 2)
y_open, x_open = out
check('y_pred shape (batch, nf, ny)', tuple(y_open.shape) == (BATCH, NF_SMALL, ny),
      str(tuple(y_open.shape)))
with torch.no_grad():
    x_manual = x0
    for t in range(NF_SMALL):
        _, x_manual = fs.hfn(x_manual, UF[:, t])
check('x_final is the state the rollout ended on', bool(torch.equal(x_open, x_manual)))

spy = _SpySimulator()
fs.simulator = spy
with torch.no_grad():
    y_spy, _ = fs.simulate(x0, UF, YF, nf=NF_SMALL)
check('an attached simulator IS consulted', spy.calls == 1)
check('its return value is what comes back',
      bool(torch.allclose(y_spy, y_open + 1.0)))
L_spy = float(fs.loss(UH, YH, UF, YF, nf=NF_SMALL).detach())
fs.simulator = None
L_open = float(fs.loss(UH, YH, UF, YF, nf=NF_SMALL).detach())
check('loss() uses the simulator, not the open loop', L_spy != L_open,
      'closed %.6e vs open %.6e' % (L_spy, L_open))

# ---- S3 ---------------------------------------------------------------------------------------
print('\nS3  make_training_data seam')
small = deepSI.System_data(u=np.array(sd.u[:3000]), y=np.array(sd.y[:3000]), dt=sd.dt)
sdl = deepSI.System_data_list([small])
base = fs.make_training_data(fs.norm.transform(sdl), nf=NF_SMALL, stride=100)
check('no simulator: deepSI\'s arrays unchanged', len(base) == 4, '%d arrays' % len(base))
fs.simulator = spy
aug = fs.make_training_data(fs.norm.transform(sdl), nf=NF_SMALL, stride=100)
fs.simulator = None
check('with a simulator: the extra array arrives', len(aug) == 5, '%d arrays' % len(aug))
check('the first four arrays are untouched',
      all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(base, aug[:4])))

# ---- S4 / S5 ----------------------------------------------------------------------------------
print('\nS4/S5  cal_validation_error seam and validation_probes')
seen = []


def probe_a(fit_sys, val_sys_data, value):
    seen.append(('a', value))
    return 999.0                     # a return value here must be IGNORED


def probe_b(fit_sys, val_sys_data, value):
    seen.append(('b', value))


fs.simulator = spy
fs.validation_probes = (probe_a, probe_b)
v = fs.cal_validation_error(sdl, validation_measure='sim-RMS')
fs.simulator = None
fs.validation_probes = ()
check('with a simulator: the simulator\'s value is returned', v == -12345.0, str(v))
check('every probe was called', [t for t, _ in seen] == ['a', 'b'], str([t for t, _ in seen]))
check('probes see the value', all(val == -12345.0 for _, val in seen))
check('a probe CANNOT replace the value', v == -12345.0)

# ---- S6 ---------------------------------------------------------------------------------------
print('\nS6  multiple shooting with a simulator is refused, not guessed')
keep = (fs.n_seg, fs.defect_weight, fs.defect_acc_weight)
fs.n_seg, fs.defect_weight, fs.defect_acc_weight = 4, 1.0, 1.0
fs.simulator = spy
raised = None
try:
    fs.loss(UH, YH, UF, YF, nf=NF_SMALL)
except NotImplementedError as e:
    raised = str(e)
except Exception as e:                                     # any other error is a different bug
    raised = 'WRONG EXCEPTION: %s: %s' % (type(e).__name__, e)
fs.simulator = None
check('n_seg > 1 + simulator raises NotImplementedError',
      raised is not None and not raised.startswith('WRONG'),
      (raised or 'nothing raised')[:60])
# and the same configuration WITHOUT a simulator still works, because the defect diagnostics use it
try:
    L_seg = float(fs.loss(UH, YH, UF, YF, nf=NF_SMALL).detach())
    seg_ok = np.isfinite(L_seg)
except Exception as e:
    L_seg, seg_ok = None, False
    print('       segmented loss raised: %s: %s' % (type(e).__name__, e))
check('n_seg > 1 without a simulator still runs', seg_ok,
      '' if L_seg is None else 'loss %.6e' % L_seg)
fs.n_seg, fs.defect_weight, fs.defect_acc_weight = keep

# ---- S7 ---------------------------------------------------------------------------------------
print('\nS7  no monkey patches left on the production path (plan 5.1 greps)')
for pattern, where, expect in (
        (r'__class__ = type', ['model_augmentation'], 0),
        (r'cal_validation_error = ', ['model_augmentation', 'scripts/gantry/gantry_dynamic'], 0),
        (r'def closed_loop_rollout', ['model_augmentation'], 1)):
    n = 0
    for w in where:
        r = subprocess.run(['grep', '-rn', pattern, os.path.join(REPO, w)],
                           capture_output=True, text=True)
        n += len([l for l in r.stdout.splitlines() if l.strip()])
    check('grep %-28s -> %d (want %d)' % (pattern, n, expect), n == expect)

print('\n' + '=' * 96)
print('OVERALL: %s' % ('PASS' if ok else 'FAIL'))
sys.exit(0 if ok else 1)
