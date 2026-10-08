"""Todo 7b check: report rows 2 to 4 on a model COPY (closed_loop_report._without_learned) against the D-227
in-place context managers, on the two saved noisy smoke models (validation split, rows 1 to 4, no oracle).

  u    rows 2 to 4 bitwise equal old vs new; the trained model's state bitwise unchanged by the new path
  obc  with theta_frozen = 0 old == new bitwise; with the trained theta old != new (the D-227 rows carried
       -E J theta); doubling theta moves the old rows and leaves the new rows bitwise unchanged

Run (one arm per process, the entry reads its config at import):
  conda run -n GraduationProject python check_report_copy.py u
  conda run -n GraduationProject python check_report_copy.py obc
"""
__project_origin__ = "added"

import contextlib
import os
import sys

import numpy as np
import torch

ARM = sys.argv[1]
RUN = {'u': '20260928_230134', 'obc': '20260928_231124'}[ARM]
os.environ.update(THESIS_ARM=ARM, THESIS_NOISE='noisy', THESIS_START='detuned', THESIS_NA='2',
                  THESIS_SEED='1', THESIS_SMOKE='1', THESIS_DEVICE='cpu')
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
for p in (REPO, os.path.join(REPO, 'scripts', 'gantry')):
    sys.path.insert(0, p)
import deepSI                                                                          # noqa: E402
import gantry_interconnect_dynamic as E                                                # noqa: E402
from gantry_dynamic.data import load_datasets, compute_normalization                   # noqa: E402
from gantry_dynamic.controller import build_controller_bank                             # noqa: E402
from gantry_dynamic import closed_loop_report as R                                      # noqa: E402
from model_augmentation.fit_systems.blocks import Static_ANN_Block                      # noqa: E402


# ---- the D-227 context managers, verbatim from the committed closed_loop_report.py ----
@contextlib.contextmanager
def ann_silenced(fit_sys):
    ann = next((b for b in fit_sys.hfn.connected_blocks if isinstance(b, Static_ANN_Block)), None)
    if ann is None:
        yield
        return
    last = [m for m in ann.net.net if isinstance(m, torch.nn.Linear)][-1]
    saved = [p.detach().clone() for p in (last.weight, last.bias) if p is not None]
    with torch.no_grad():
        for p in (last.weight, last.bias):
            if p is not None:
                p.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            for p, s in zip([p for p in (last.weight, last.bias) if p is not None], saved):
                p.copy_(s)


@contextlib.contextmanager
def physics_at(phy, free):
    saved = phy.free_params.detach().clone()
    with torch.no_grad():
        phy.free_params.copy_(free)
    try:
        yield
    finally:
        with torch.no_grad():
            phy.free_params.copy_(saved)


cfg = E.CFG
data = load_datasets(cfg)
norm = compute_normalization(cfg, data)
fit_sys = deepSI.load_system(os.path.join(REPO, 'simulations', 'gantry_subnet', 'thesis_taf_noisy_linear_map',
                                          RUN, 'gantry_' + RUN))
fit_sys.eval()
sdl, files = list(data.val_list), list(data.val_files)
bank, rows, _ = build_controller_bank(files, cfg.ts_new, ystd=norm.ystd, std_u=norm.std_u, dtype=cfg.dtype_pt,
                                      use_gain=cfg.controller_gain)
phy = next(b for b in fit_sys.hfn.connected_blocks if hasattr(b, 'free_params'))
corr = getattr(fit_sys.hfn, 'obc_correction', None)
fails = []


def check(name, ok):
    print('  %-86s %s' % (name, 'PASS' if ok else 'FAIL'), flush=True)
    if not ok:
        fails.append(name)


def state():
    return {k: v.detach().clone() for mod in (fit_sys.hfn, fit_sys.encoder) for k, v in mod.state_dict().items()}


def unchanged(s0):
    s1 = state()
    return s0.keys() == s1.keys() and all(torch.equal(s0[k], s1[k]) for k in s0)


def score(m):
    return np.asarray(R._score(m, sdl, bank, rows)[0])


def old_rows():
    with ann_silenced(fit_sys):
        r2 = score(fit_sys)
        with physics_at(phy, torch.zeros_like(phy.free_params)):
            r3 = score(fit_sys)
        with physics_at(phy, R._true_free(phy)):
            r4 = score(fit_sys)
    return r2, r3, r4


def new_rows():
    return (score(R._without_learned(fit_sys)),
            score(R._without_learned(fit_sys, torch.zeros_like(phy.free_params))),
            score(R._without_learned(fit_sys, R._true_free(phy))))


def eq(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a, b))


print('[%s] model %s, %d validation records, OBC correction %s' % (ARM, RUN, len(sdl), corr is not None))
s0 = state()
r1 = score(fit_sys)
new = new_rows()
check('new path leaves the trained model bitwise unchanged', unchanged(s0))
old = old_rows()
check('old path restored the model bitwise (reference)', unchanged(s0))
check('row 1 unchanged by scoring rows 2 to 4', np.array_equal(r1, score(fit_sys)))
for i, name in enumerate(('physics part', 'baseline, start', 'baseline, true')):
    print('    %-16s old %s  new %s' % (name, np.array2string(old[i].ravel()[:3], precision=4),
                                        np.array2string(new[i].ravel()[:3], precision=4)))

if corr is None:
    check('rows 2 to 4 bitwise equal old vs new (no OBC)', eq(old, new))
else:
    theta = corr.theta_frozen.detach().clone()
    print('    trained theta_frozen norm %.3e' % float(theta.norm()))
    check('rows 2 to 4 differ old vs new with the trained theta (the D-227 rows carried -E J theta)',
          float(theta.norm()) > 0 and not eq(old, new))
    with torch.no_grad():
        corr.theta_frozen.mul_(2.0)
    old2, new2 = old_rows(), new_rows()
    check('doubling theta moves the OLD rows 2 to 4', not eq(old, old2))
    check('doubling theta leaves the NEW rows 2 to 4 bitwise unchanged', eq(new, new2))
    with torch.no_grad():
        corr.theta_frozen.zero_()
    old0 = old_rows()
    check('with theta_frozen = 0, old == new bitwise', eq(old0, new))
    with torch.no_grad():
        corr.theta_frozen.copy_(theta)
    check('trained model restored after the test', unchanged(s0))

print('\n[%s] %s (%d failures)' % (ARM, 'ALL PASS' if not fails else 'FAILURES', len(fails)))
