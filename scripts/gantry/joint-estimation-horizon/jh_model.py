"""The vendored production model, the ANN-free closed-loop rollout and its forward-mode
sensitivities. Shared by G1 (info/) and G2 (staged/). JH-003, JH-005.

Set JH_FILESET in the environment BEFORE importing this module if the obc14 file lists are wanted
(the vendored data.py reads it at import).
"""
import importlib.util
import os
import sys
import time
from dataclasses import dataclass, replace

import jh_env  # noqa: F401  (puts vendor/ first on sys.path)

import numpy as np
import torch
import torch.autograd.forward_ad as fwAD

COMBO = ['kb_sum', 'cg1', 'cg2', 'cy', 'cb_sum', 'mh', 'm_total', 'm_diff', 'J_eff', 'd']
STAGE = ['X1', 'X2', 'Y']
NA = 29          # encoder window: na = nb = 29 (production na_nb), na_right = nb_right = 1


def load_entry():
    """The vendored entry file as a module (its CFG and env overrides), never main()."""
    if 'gantry_interconnect_dynamic' in sys.modules:
        return sys.modules['gantry_interconnect_dynamic']
    path = os.path.join(jh_env.VENDOR, 'gantry_interconnect_dynamic.py')
    spec = importlib.util.spec_from_file_location('gantry_interconnect_dynamic', path)
    m = importlib.util.module_from_spec(spec)
    sys.modules['gantry_interconnect_dynamic'] = m
    spec.loader.exec_module(m)
    return m


# The configuration of every G1/G2 computation: the reduced ten-combination block, prior off,
# float64 on the CPU, nothing compiled. `combo_init_detune` is set by the caller (None = TRUE).
BASE_OVERRIDES = dict(joint_estimation=True, physics_parameterization='reduced',
                      param_init_detune=None, param_prior=False, obc=False, orth=False,
                      use_f64=True, device='cpu', compile_mode=None)


@dataclass
class Ctx:
    cfg: object
    data: object
    norm: object
    fit_sys: object
    block: object
    bank: object
    Cn: torch.Tensor
    train_files: list
    train_rows: list
    ystd: np.ndarray


def build(**overrides):
    """Build the model exactly as main() does, with BASE_OVERRIDES plus `overrides`."""
    entry = load_entry()
    from gantry_dynamic.data import load_datasets, compute_normalization, TRAIN_FILES, VAL_FILES
    from gantry_dynamic.model import build_model
    from gantry_dynamic.controller import build_closed_loop
    cfg = replace(entry.CFG, **{**BASE_OVERRIDES, **overrides})
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    data = load_datasets(cfg)
    norm = compute_normalization(cfg, data)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    fit_sys = build_model(cfg.hp, cfg, data, norm)
    fit_sys.simulator = build_closed_loop(fit_sys, norm, cfg, train_files=TRAIN_FILES,
                                          val_files=VAL_FILES, val_data=data.val_ckpt_data,
                                          verbose=False)
    fit_sys.eval()
    block = fit_sys.hfn.connected_blocks[0]
    out = fit_sys.hfn.connected_blocks[1]
    assert type(block).__name__ == 'Reduced_Gantry_State_Block', type(block).__name__
    assert torch.count_nonzero(out.D) == 0, 'output feedthrough must be zero (D_d = 0)'
    jh_env.check_no_leak(verbose=False)
    return Ctx(cfg=cfg, data=data, norm=norm, fit_sys=fit_sys, block=block,
               bank=fit_sys.simulator.bank, Cn=out.C.detach().clone(),
               train_files=list(TRAIN_FILES), train_rows=list(fit_sys.simulator.train_ctrl_rows),
               ystd=np.asarray(norm.ystd, float).ravel())


def mats_of(block, free):
    """The M(Y) structure at an explicit free coordinate (same path as pass_mats())."""
    return block._mats_from_raw(block.gauge_section(block.combinations_from_free(free),
                                                    block.params_init, block.Lb))


def normalised_records(ctx, files=None, which='train'):
    """[(name, un (N,3), yn (N,3), ctrl_row)] for the training records, normalised as the model."""
    fs = ctx.fit_sys
    u0, us = np.ravel(fs.norm.u0), np.ravel(fs.norm.ustd)
    y0, ys = np.ravel(fs.norm.y0), np.ravel(fs.norm.ystd)
    out = []
    for i, (name, sd) in enumerate(zip(ctx.train_files, ctx.data.train_list)):
        if files is not None and name not in files:
            continue
        out.append((name, (np.asarray(sd.u, float) - u0) / us, (np.asarray(sd.y, float) - y0) / ys,
                    ctx.train_rows[i]))
    return out


def windows(recs, nf, span=3200, k_first=NA):
    """Non-overlapping windows of nf inside [k_first, k_first + span*M) of each record.

    Returns dict of stacked arrays: uh, yh (B, NA+1, 3) encoder history, u, y (B, nf, 3),
    ctrl (B,), rec (B,), k (B,) window start sample. The same samples for every nf dividing span.
    """
    assert span % nf == 0
    UH, YH, U, Y, C, R, K = [], [], [], [], [], [], []
    for ri, (name, un, yn, row) in enumerate(recs):
        M = (len(un) - k_first) // span
        for k in range(k_first, k_first + span * M, nf):
            UH.append(un[k - NA:k + 1]); YH.append(yn[k - NA:k + 1])
            U.append(un[k:k + nf]); Y.append(yn[k:k + nf])
            C.append(row); R.append(ri); K.append(k)
    return dict(uh=np.stack(UH), yh=np.stack(YH), u=np.stack(U), y=np.stack(Y),
                ctrl=np.asarray(C, np.int64), rec=np.asarray(R), k=np.asarray(K))


def T(a, dtype=torch.float64):
    return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype)


def encoder_x0(ctx, uh, yh):
    """Production encoder output (B, 14); the physical part is [:, :6]."""
    with torch.no_grad():
        return ctx.fit_sys.encoder(T(uh), T(yh)).detach()


def rollout(ctx, mats, x0, u, y, ctrl, corr=None):
    """ANN-free closed loop, residual form, xc = 0, the step order of `_rollout_segment`.

    x0 (B, 6) normalised, u, y (B, nf, 3) normalised tensors, ctrl (B,) long.
    corr: optional (B, nf, 6) additive correction on the NEXT physical state (G2 arm ii).
    Returns y_model (B, nf, 3) normalised.
    """
    block, bank = ctx.block, ctx.bank
    ctl = bank.gather(ctrl)
    x = x0.unsqueeze(-1)
    xc = torch.zeros(x0.shape[0], bank.nc, dtype=x0.dtype)
    ys = []
    for t in range(u.shape[1]):
        ym = torch.matmul(ctx.Cn, x).squeeze(-1)          # Linear_Output_Block: C @ x
        ys.append(ym)
        ufb, xc = bank.step(xc, y[:, t] - ym, ctl)
        x = block._rk4(x, (u[:, t] + ufb).unsqueeze(-1), mats)
        if corr is not None:
            x = x + corr[:, t].unsqueeze(-1)
    return torch.stack(ys, dim=1)


def sensitivities(ctx, free, x0, u, y, ctrl, want_x0=True, corr=None):
    """Forward-mode dy/dfree (10 directions) and dy/dx0 (6), float64.

    Returns (y_model (B,nf,3), S (B, nf, 3, 16 or 10)) in NORMALISED output units, columns
    [x0_0..x0_5, free_0..free_9] (x0 first) or free only.
    """
    block = ctx.block
    free = free.detach().to(torch.float64)
    B = x0.shape[0]
    cols = []
    y_prim = None
    if want_x0:
        # one pass, the batch replicated six times with a unit tangent on each initial state
        x0r = x0.repeat(6, 1)
        tan = torch.zeros_like(x0r)
        for k in range(6):
            tan[k * B:(k + 1) * B, k] = 1.0
        with fwAD.dual_level():
            yd = rollout(ctx, mats_of(block, free), fwAD.make_dual(x0r, tan), u.repeat(6, 1, 1),
                         y.repeat(6, 1, 1), ctrl.repeat(6),
                         None if corr is None else corr.repeat(6, 1, 1))
            p, t = fwAD.unpack_dual(yd)
        y_prim = p[:B].clone()
        cols += [t[k * B:(k + 1) * B] for k in range(6)]
    for j in range(10):
        e = torch.zeros(10, dtype=torch.float64)
        e[j] = 1.0
        with fwAD.dual_level():
            yd = rollout(ctx, mats_of(block, fwAD.make_dual(free, e)), x0, u, y, ctrl, corr)
            p, t = fwAD.unpack_dual(yd)
        if y_prim is None:
            y_prim = p.clone()
        cols.append(t)
    return y_prim, torch.stack(cols, dim=-1)


class Stopwatch:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self):
        return time.time() - self.t0


def peak_rss_mb():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return getattr(mi, 'peak_wset', mi.rss) / 2 ** 20
    except Exception:
        return float('nan')
