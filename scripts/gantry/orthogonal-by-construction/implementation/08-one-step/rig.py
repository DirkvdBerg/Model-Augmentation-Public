"""Shared rig for the one-step OBC probes and tests (D-192). Named `rig`, not `common`: the
pipeline has a `scripts/gantry/common` package that evaluation.py imports.

Builds the PRODUCTION model through `gantry_dynamic.model.build_model` with the entry point's
config (`gantry_interconnect_dynamic.CFG`) switched to reduced joint estimation, attaches the
OBC objects, and hands out batches shaped exactly as `fit()` shapes them. Nothing here is
library code; the library lives in `model_augmentation/fit_systems/obc.py` and
`scripts/gantry/gantry_dynamic/obc_gantry.py`.
"""
__project_origin__ = "added"

import contextlib
import importlib.util
import io
import os
import subprocess
import sys
from dataclasses import replace
from datetime import date

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_HERE, '..', '..', '..', '..', '..'))
for p in (REPO, os.path.join(REPO, 'scripts', 'gantry')):
    if p not in sys.path:
        sys.path.insert(0, p)

RESULTS_DIR = os.path.join(_HERE, '..', 'results', date.today().isoformat())
os.makedirs(RESULTS_DIR, exist_ok=True)
START_COMMIT = 'a52dd5584065f87bdeceef4b6f4a3ccdf05fb6cd'

from gantry_interconnect_dynamic import CFG                                  # noqa: E402
from gantry_dynamic.data import (load_datasets, compute_normalization,       # noqa: E402
                                 VAL_FILES, TRAIN_FILES)
from gantry_dynamic.model import build_model                                 # noqa: E402
from gantry_dynamic.controller import build_closed_loop                      # noqa: E402
from gantry_dynamic.obc_gantry import attach_obc                             # noqa: E402


class Tee:
    """stdout mirror to a log file (same helper as the earlier stages)."""

    def __init__(self, path):
        self.f = open(path, 'w', encoding='utf-8')

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()

    # torch._dynamo inspects sys.stdout (isatty/fileno) when it formats its logs; without
    # these the compile probe dies inside Dynamo with an AttributeError on this object.
    def isatty(self):
        return False

    def fileno(self):
        return sys.__stdout__.fileno()


def obc_config(**over):
    """The entry point's config as an OBC-capable reduced joint-estimation run.

    `obc` itself is left to the caller: tests attach with a decimated reference set through
    `build_system(ref_stride=...)`, the qualification uses the production attach (`obc=True`).
    """
    base = dict(joint_estimation=True, physics_parameterization='reduced',
                param_init_detune=None, orth=False, lbfgs=False, save_flag=False,
                nf_probe_print=False)
    base.update(over)
    return replace(CFG, **base)


class Rig:
    def __init__(self, cfg, data, norm, hp, fs, life, corr, ref):
        self.cfg, self.data, self.norm, self.hp = cfg, data, norm, hp
        self.fs, self.life, self.corr, self.ref = fs, life, corr, ref

    @property
    def ann(self):
        return self.fs.hfn.connected_blocks[self.fs._ann_ix]

    @property
    def phy(self):
        return self.corr._phy[0]


def build_system(cfg, ref_stride=1, attach=True, verbose=False, closed_loop=True):
    """Production build + closed loop + OBC attach. `ref_stride > 1` is for probes only."""
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    with contextlib.redirect_stdout(io.StringIO()) if not verbose else contextlib.nullcontext():
        data = load_datasets(cfg)
        norm = compute_normalization(cfg, data)
        hp = cfg.hp
        np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        # build with the attach switched OFF so the decimated set can be attached by hand
        fs = build_model(hp, replace(cfg, obc=False), data, norm)
        if closed_loop:
            fs.simulator = build_closed_loop(fs, norm, cfg, train_files=TRAIN_FILES,
                                             val_files=VAL_FILES, val_data=data.val_ckpt_data,
                                             verbose=verbose)
        life = corr = ref = None
        if attach:
            life, corr, ref = attach_obc(fs, cfg, data, norm, ref_stride=ref_stride,
                                         verbose=verbose)
    fs.epoch_counter = 0.0
    return Rig(cfg, data, norm, hp, fs, life, corr, ref)


def perturb_output_layer(ann, scale=1e-3, seed=1):
    """Give the zero-initialised ANN output layer a small deterministic perturbation.

    Only the LAST linear layer: the pipeline zero-initialises it so the write is exactly zero
    at init, and a projection that removes nothing is not evidence. Small, so the write stays a
    correction and the closed loop does not leave its linear regime (a 0.05 perturbation of every
    layer sent a 400-step rollout to NaN gradients in the first smoke test).
    """
    last = [m for m in ann.modules() if isinstance(m, torch.nn.Linear)][-1]
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        last.weight.add_(scale * torch.randn(last.weight.shape, generator=g).to(last.weight))
        last.bias.add_(scale * torch.randn(last.bias.shape, generator=g).to(last.bias))


def make_batch(rig, B=8, seed=0, device=None):
    """`B` training windows exactly as `fit()` would hand them to `loss()`: (arrays, kwargs)."""
    fs, cfg, hp = rig.fs, rig.cfg, rig.hp
    dt = fs.make_training_data(fs.norm.transform(rig.data.train_data),
                               nf=hp['nf'], stride=cfg.stride)
    n = len(dt[0])
    rs = np.random.RandomState(seed)
    ix = np.sort(rs.choice(n, size=min(B, n), replace=False))
    dev = device or next(fs.hfn.parameters()).device
    batch = [torch.as_tensor(a[ix]).to(cfg.dtype_pt if a.dtype.kind == 'f' else torch.long).to(dev)
             for a in dt]
    kw = dict(nf=hp['nf'], stride=cfg.stride, ctrl_ix=batch[4])
    return batch, kw


def loss_and_backward(rig, batch, kw):
    fs = rig.fs
    fs.optimizer.zero_grad()
    L = fs.loss(*batch[:4], **kw)
    L.backward()
    return L


def head_module(rel_path, name):
    """Import `rel_path` as it stood at the START commit, under module name `name`."""
    src = subprocess.run(['git', 'show', '%s:%s' % (START_COMMIT, rel_path)], cwd=REPO,
                         capture_output=True, check=True).stdout.decode('utf-8')
    d = os.path.join(RESULTS_DIR, '_head_snapshot')
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name + '.py')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def fmt(x):
    return '%.4e' % float(x)
