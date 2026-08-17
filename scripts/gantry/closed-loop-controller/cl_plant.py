"""Adapter from the interconnect to the (step_fn, out_fn) pair the closed-loop rollout needs.

WHY THIS EXISTS
---------------
`fs.hfn(x, u)` returns `(y, x_next)` together. The closed loop needs `y` BEFORE it can form `u`,
so a naive implementation calls `hfn` twice per step: once to read `y`, once to advance the state
with the corrected input. That is what `ClosedLoopLossMixin` does, and it doubles the FP-plus-ANN
forward cost of every training step.

It is avoidable because the output has no feedthrough, so `y` is a fixed affine function of the
normalised state and can be evaluated without touching the model:

    y_norm = x_norm @ C.T + b

`C` and `b` are identified from the model itself with `nx + 1` forward passes at setup
(the same identification `dc-accumulation/compare_annoff_replay.py` uses for its `x0`), so they
cannot drift from whatever `Cd_norm`, `std_x`, `x_mean`, `ystd` and `y0` happen to be. The
rollout then costs ONE `hfn` call per step.

THREE THINGS THIS ASSUMES, ALL CHECKED RATHER THAN ASSERTED IN A COMMENT
-----------------------------------------------------------------------
1. **No feedthrough.** `model_augmentation.systems.gantry_ss.Dd` is exactly zero, but
   `model.py:140` still wires `u` into `out_phys` (`connect_block_signals(out_phys, ["u"], ["y"])`),
   so the wiring permits a nonzero `D`. `check_no_feedthrough` runs the real interconnect on the
   same `x` with two very different `u` and requires identical `y`. If that ever fails, the
   closed-loop step order is invalid and there is a genuine algebraic loop.
2. **The output map is affine.** `check_affine` compares `x @ C.T + b` against `hfn` on random
   states, not just on the basis vectors used to identify it.
3. **The output map is not trainable.** `C` and `b` are captured once as constants. If any
   parameter of the output path required grad, freezing them would silently cut the gradient.
   D-066 removed the trainable `C_aug` so the output is solely the fixed `Cd_norm`, but
   `check_output_frozen` verifies it instead of relying on that still being true.
"""
__project_origin__ = "added"

import numpy as np
import torch

from model_augmentation.fit_systems.blocks import Static_ANN_Block


def identify_output_map(hfn, nx, nu, dtype=torch.float32):
    """(C, b) such that y_norm = x_norm @ C.T + b. Costs nx + 1 forward passes."""
    u0 = torch.zeros(1, nu, dtype=dtype)
    with torch.no_grad():
        b = hfn(torch.zeros(1, nx, dtype=dtype), u0)[0][0].clone()
        ny = b.shape[0]
        C = torch.zeros(ny, nx, dtype=dtype)
        for j in range(nx):
            e = torch.zeros(1, nx, dtype=dtype)
            e[0, j] = 1.0
            C[:, j] = hfn(e, u0)[0][0] - b
    return C, b


def check_no_feedthrough(hfn, nx, nu, dtype=torch.float32, scale=1e3, seed=0):
    """y must not depend on u. Returns the worst absolute difference over random states."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(8, nx, generator=g, dtype=dtype)
    ua = torch.zeros(8, nu, dtype=dtype)
    ub = torch.randn(8, nu, generator=g, dtype=dtype) * scale
    with torch.no_grad():
        ya = hfn(x, ua)[0]
        yb = hfn(x, ub)[0]
    return float((ya - yb).abs().max())


def check_affine(hfn, C, b, nx, nu, dtype=torch.float32, seed=1):
    """x @ C.T + b must reproduce hfn's y on states OTHER than the identification basis."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(16, nx, generator=g, dtype=dtype)
    with torch.no_grad():
        y_model = hfn(x, torch.zeros(16, nu, dtype=dtype))[0]
    y_map = x @ C.T + b
    denom = y_model.abs().max().clamp_min(1e-30)
    return float((y_model - y_map).abs().max()), float((y_model - y_map).abs().max() / denom)


def check_output_frozen(fs):
    """Number of trainable parameters on the output path. Must be 0 for (C, b) to be constants.

    Identifies the output block as the one whose forward produces 'y'; falls back to reporting
    every block's trainable count so a failure is diagnosable rather than just a number.
    """
    from model_augmentation.fit_systems.blocks import Linear_Output_Block
    counts = {}
    for m in fs.hfn.connected_blocks:
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        counts[type(m).__name__] = counts.get(type(m).__name__, 0) + n
    out_n = sum(sum(p.numel() for p in m.parameters() if p.requires_grad)
                for m in fs.hfn.connected_blocks if isinstance(m, Linear_Output_Block))
    return out_n, counts


def zero_the_ann(fs):
    """Force the ANN output to exactly zero. Returns a restore callable.

    The same monkeypatch `compare_annoff_replay.py` uses. With the ANN off the augmented model
    IS the baseline, which is what the replay gate needs.
    """
    ann = next(m for m in fs.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
    orig = ann.forward
    ann.forward = lambda z: torch.zeros_like(orig(z))

    def restore():
        ann.forward = orig
    return restore


class ModelStep:
    """Picklable `x_{k+1} = hfn(x, u)`.

    NOT a closure. These end up in `fit_sys.__dict__`, and deepSI's `checkpoint_save_system`
    does `torch.save(self.__dict__, file)` at EVERY validation, so a local function here raises
    `Can't pickle local object 'make_fns.<locals>.step_fn'` and kills the run. That is the same
    failure mode that lost variant B's verdict block after 3.2 h of training
    (`loss_variants.py:165-171`), arriving from a different direction.

    Holds `hfn` rather than the fit system: `hfn` is already in `fit_sys.__dict__` so pickle
    memoises it, and it avoids a needless reference cycle back through the fit system.
    """

    def __init__(self, hfn):
        self.hfn = hfn

    def __call__(self, x, u):
        return self.hfn(x, u)[1]


class AffineOutput:
    """Picklable `y = x @ C.T + b`, the identified no-feedthrough output map."""

    def __init__(self, C, b):
        self.C = C
        self.b = b

    def __call__(self, x):
        return x @ self.C.T + self.b


def make_fns(fs, C, b):
    """(step_fn, out_fn) for cl_controller.rollout. One hfn call per step."""
    return ModelStep(fs.hfn), AffineOutput(C, b)


def normalise_u(u_raw, norm, dtype_np):
    """Recorded plant input -> normalised model input."""
    u_mean = np.asarray(norm.u_mean).flatten()
    std_u = np.asarray(norm.std_u).flatten()
    return ((u_raw - u_mean) / std_u).astype(dtype_np)


def denormalise_y(y_norm, norm):
    """Normalised model output -> physical [m]."""
    return np.asarray(y_norm) * np.asarray(norm.ystd) + np.asarray(norm.y0)


def normalise_y(y_raw, norm, dtype_np):
    return ((y_raw - np.asarray(norm.y0)) / np.asarray(norm.ystd)).astype(dtype_np)
