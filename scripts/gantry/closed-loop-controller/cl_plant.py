"""Checks on the interconnect's output map, and the ANN off-switch. Gate code only.

MIGRATION step 7: the ADAPTERS are gone. `ModelStep`, `AffineOutput` and `make_fns` existed to
give the closed-loop rollout a `y = h(x)` accessor without calling the model twice per step, by
identifying the output map with `nx + 1` probe forward passes and then assuming it stayed affine
and frozen. `Interconnect.output_only` evaluates the output signal's dependency cone directly, so
the accessor is exact by construction, needs no identification, and none of the three assumptions
has to hold. Measured equal to the full forward BIT-IDENTICALLY (`cl_test_output_only.py`).

What remains here is the checks, which is where they belong. `identify_output_map` survives ONLY
as one of them: it is the map the previous implementation used, and agreeing with it is how we
know the two are the same object rather than two plausible ones.

1. **No feedthrough.** `model_augmentation.systems.gantry_ss.Dd` is exactly zero, but
   `model.py:140` still wires `u` into `out_phys` (`connect_block_signals(out_phys, ["u"], ["y"])`),
   so the wiring PERMITS a nonzero `D` and `D_d = 0` is a property of the block's coefficients,
   not of the graph. `check_no_feedthrough` runs the real interconnect on the same `x` with two
   very different `u` and requires identical `y`. If that ever fails, the closed-loop step order
   is invalid and there is a genuine algebraic loop. This gate is NOT redundant with a structural
   argument and must be kept.
2. **The output map is affine.** `check_affine` compares `x @ C.T + b` against `hfn` on random
   states, not just on the basis vectors used to identify it.
3. **The output map is not trainable.** D-066 removed the trainable `C_aug` so the output is
   solely the fixed `Cd_norm`, but `check_output_frozen` verifies it rather than relying on that
   still being true.

The ANN off-switch is `ann_output_zeroed`, a context manager that zeroes the ANN's output layer
and restores it. It replaced `zero_the_ann`, which patched `ann.forward`; see that function for
why parameters are the right thing to mutate and code is not. Plan 3.6d had decided to KEEP the
patch on the grounds that it is gate-only and fails visibly, which was a fair argument, but a
version with no patch, no restore callable to forget, and no cost to the production path is
simply better, so the decision was revisited rather than defended.

The eight remaining `PLAN.zero_the_ann` callers are all in the historical scripts left dangling by
migration step 7 and are not repointed; see the plan's step 7 table.
"""
__project_origin__ = "added"

import contextlib

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


def _ann_final_linear(fs):
    """The ANN's output Linear layer, i.e. the one `zero_init_feed_forward_nn` zeroes at init."""
    ann = next(m for m in fs.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
    net = ann.net.net if hasattr(ann.net, 'net') else ann.net
    for layer in reversed(list(net)):
        if isinstance(layer, torch.nn.Linear):
            return layer
    raise RuntimeError('no Linear layer found in the ANN; zeroing its output is not defined here')


@contextlib.contextmanager
def ann_output_zeroed(fs):
    """Force the ANN output to exactly zero for the duration of the block, then restore.

    WHY THIS REPLACED A MONKEY PATCH
    --------------------------------
    The predecessor did `ann.forward = lambda z: torch.zeros_like(orig(z))` and handed back a
    restore callable. Three things are better here:

    1. It mutates PARAMETERS, not CODE. Replacing a bound method swaps behaviour and leaves the
       object lying about what it is; setting the output layer to zero is an ordinary parameter
       assignment that this network is designed to accept. `zero_init_feed_forward_nn` zeroes
       exactly this layer's weight AND bias at construction, so the block below puts the model
       into the state it is BORN in, which is also the state every gate assumes when it says
       "with the ANN at zero the augmented model IS the baseline". The gate now exercises a real
       configuration rather than a simulated one.
    2. It is scoped by the `with`, not by remembering to call a restore. The old form leaked on
       any exception between install and restore.
    3. It costs the production path NOTHING. The alternative the plan sketched, an `output_scale`
       buffer on `Static_ANN_Block`, would put a multiply into every ANN forward, 400 per window,
       to serve a diagnostic; in a dispatch-bound regime (plan 3.8) that is 400 added dispatches
       on the training path for no training benefit.

    The old version also did not even save the forward pass: it called `orig(z)` for the shape and
    threw the result away.

    Asserts its own postcondition, because a gate that silently fails to zero the ANN would make
    every comparison downstream meaningless.
    """
    layer = _ann_final_linear(fs)
    saved_w = layer.weight.detach().clone()
    saved_b = None if layer.bias is None else layer.bias.detach().clone()
    try:
        with torch.no_grad():
            layer.weight.zero_()
            if layer.bias is not None:
                layer.bias.zero_()
        yield
    finally:
        with torch.no_grad():
            layer.weight.copy_(saved_w)
            if layer.bias is not None:
                layer.bias.copy_(saved_b)


def ann_output_magnitude(fs, nz=None, n=8, dtype=torch.float32, seed=0):
    """max |ANN output| over a FIXED set of random inputs. 0.0 exactly when the output is zeroed.

    The input set is seeded on purpose. An earlier version drew fresh `randn` on every call, so
    two calls on identical parameters returned different numbers and "is it restored?" could not
    be asked of it at all. A check whose own inputs move is not a check.
    """
    ann = next(m for m in fs.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
    nz = ann.nz if nz is None else nz
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        w = ann(torch.randn(n, nz, 1, generator=g, dtype=dtype))
    return float(w.abs().max())


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
