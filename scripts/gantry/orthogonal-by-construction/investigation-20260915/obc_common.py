"""Standalone float64 gantry baseline for the orthogonal-by-construction investigation.

__project_origin__ = "added"

Built from scratch (per the diagnostic-independence rule) rather than by calling the
production block, then cross-checked against `Gantry_State_Block` in `check_against_block()`.
Nothing here imports the training pipeline; the only project import is the scalar parameter
table in `model_augmentation/systems/gantry_ss.py`, which is the single source of truth for
the nominal values and is read as plain floats.

Model, as read from `gantry_ss.py` lines 70-128 and `blocks.py:796-849`:

    M(Y) qddot + C qdot + K q = P u,   Y = q3,   M(Y) = M0 + M1 Y + M2 Y^2
    x = (q1, q2, q3, qd1, qd2, qd3),   y = [P^T  0] x

The discrete transition is RK4 with `up_sample` substeps, `Y` re-read from the current
state at every RK4 stage (`blocks.py:778-784`).
"""

__project_origin__ = "added"

import os
import sys

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

PARAM_NAMES = ["kb1", "kb2", "cg1", "cg2", "cy", "cb1", "cb2",
               "mh", "m1", "m2", "mb", "Jb", "Jh", "d"]

COMBO_NAMES = ["kb_sum", "cg1", "cg2", "cy", "cb_sum", "mh",
               "m_total", "m_diff", "J_eff", "d"]


def nominal():
    """Nominal (14,) raw parameter vector and Lb, as float64, from gantry_ss."""
    from model_augmentation.systems import gantry_ss as g
    theta = torch.tensor([float(getattr(g, n)) for n in PARAM_NAMES], dtype=torch.float64)
    return theta, float(g.Lb)


TS = 1.0 / 20000.0
UP_SAMPLE = 10

# Documented scheduling range: RECORD_Y_OP in scripts/gantry/gantry_dynamic/controller.py
# spans -0.30 to +0.30 m over the train/val/test records.
Y_RANGE = (-0.30, 0.30)


def unpack(theta):
    return dict(zip(PARAM_NAMES, theta))


def build_matrices(theta, Lb):
    """M0, M1, M2, C, K, P from the raw 14-vector. Autograd safe."""
    p = unpack(theta)
    z = theta.new_zeros(())

    alpha = p["m1"] + p["m2"] + p["mb"] + p["mh"]
    beta = (p["m1"] - p["m2"]) * Lb / 2
    gamma = p["Jb"] + p["Jh"] + (p["m1"] + p["m2"]) * Lb ** 2 / 4
    mh, d = p["mh"], p["d"]

    M0 = torch.stack([
        torch.stack([alpha, beta, z]),
        torch.stack([beta, gamma + mh * d ** 2, -mh * d]),
        torch.stack([z, -mh * d, mh]),
    ])
    M1 = torch.stack([
        torch.stack([z, -mh, z]),
        torch.stack([-mh, z, z]),
        torch.stack([z, z, z]),
    ])
    M2 = torch.stack([
        torch.stack([z, z, z]),
        torch.stack([z, mh, z]),
        torch.stack([z, z, z]),
    ])
    C = torch.stack([
        torch.stack([p["cg1"] + p["cg2"], (p["cg1"] - p["cg2"]) * Lb / 2, z]),
        torch.stack([(p["cg1"] - p["cg2"]) * Lb / 2,
                     p["cb1"] + p["cb2"] + (p["cg1"] + p["cg2"]) * Lb ** 2 / 4, z]),
        torch.stack([z, z, p["cy"]]),
    ])
    K = torch.stack([
        torch.stack([z, z, z]),
        torch.stack([z, p["kb1"] + p["kb2"], z]),
        torch.stack([z, z, z]),
    ])
    one = theta.new_ones(())
    P = torch.stack([
        torch.stack([one, one, z]),
        torch.stack([one * Lb / 2, -one * Lb / 2, z]),
        torch.stack([z, z, one]),
    ])
    return M0, M1, M2, C, K, P


def combos(theta, Lb):
    """The 10 quantities coded in Parameterized_Gantry_State_Block._combos_from_raw."""
    p = unpack(theta)
    return torch.stack([
        p["kb1"] + p["kb2"],
        p["cg1"], p["cg2"], p["cy"],
        p["cb1"] + p["cb2"],
        p["mh"],
        p["m1"] + p["m2"] + p["mb"],
        p["m1"] - p["m2"],
        p["Jb"] + p["Jh"] + (p["m1"] + p["m2"]) * Lb ** 2 / 4,
        p["d"],
    ])


def deriv(theta, Lb, x, u, Y_frozen=None, f_extra=None, matrices=None):
    """xdot in physical units.

    x : (B, 6), u : (B, 3) stage forces.
    Y_frozen : None for endogenous self-scheduling (Y = x[:, 2]); a float or (B,) tensor
               to schedule on an exogenous value instead.
    f_extra  : optional callable (x) -> (B, 3) extra LOGICAL force, added to fnet. Used
               only to build a ground truth that the baseline cannot represent; the
               baseline itself never has one.
    matrices : optional prebuilt (M0, M1, M2, C, K, P). These do not depend on the state
               or on the substep, so hoisting them out of the RK4 loop is an exact
               optimization, not an approximation; `rk4_step` does it automatically.
    """
    M0, M1, M2, C, K, P = build_matrices(theta, Lb) if matrices is None else matrices
    q, qd = x[:, :3], x[:, 3:]
    u_log = u @ P.T
    fnet = -(q @ K.T) - (qd @ C.T) + u_log            # (B, 3)
    if f_extra is not None:
        fnet = fnet + f_extra(x)

    if Y_frozen is None:
        Y = x[:, 2]
    elif torch.is_tensor(Y_frozen):
        Y = Y_frozen.expand(x.shape[0]) if Y_frozen.ndim == 0 else Y_frozen
    else:
        Y = torch.full((x.shape[0],), float(Y_frozen), dtype=x.dtype, device=x.device)

    MY = M0.unsqueeze(0) + M1.unsqueeze(0) * Y[:, None, None] \
        + M2.unsqueeze(0) * (Y ** 2)[:, None, None]    # (B, 3, 3)
    a = torch.linalg.solve(MY, fnet.unsqueeze(-1)).squeeze(-1)
    return torch.cat([qd, a], dim=-1)


def rk4_step(theta, Lb, x, u, Ts=TS, up_sample=UP_SAMPLE, Y_frozen=None, f_extra=None,
             matrices=None):
    """One sample-period RK4 transition with `up_sample` substeps.

    The parameter-dependent matrices are built ONCE per call and reused across substeps
    and stages, which is what `Parameterized_Gantry_State_Block.nonlinear_function`
    (blocks.py:976-989) also does. Only the scheduling evaluation `M(Y)` is inside.
    """
    m = build_matrices(theta, Lb) if matrices is None else matrices
    h = Ts / up_sample
    for _ in range(up_sample):
        k1 = h * deriv(theta, Lb, x, u, Y_frozen, f_extra, m)
        k2 = h * deriv(theta, Lb, x + k1 / 2, u, Y_frozen, f_extra, m)
        k3 = h * deriv(theta, Lb, x + k2 / 2, u, Y_frozen, f_extra, m)
        k4 = h * deriv(theta, Lb, x + k3, u, Y_frozen, f_extra, m)
        x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x


def output(theta, Lb, x):
    _, _, _, _, _, P = build_matrices(theta, Lb)
    return x[:, :3] @ P            # y = P^T q  ->  (B,3) @ P is q^T P^T transposed
                                   # (P^T q)_i = sum_j P[j, i] q_j = (q @ P)_i


def simulate(theta, Lb, x0, U, Ts=TS, up_sample=UP_SAMPLE, Y_frozen=None):
    """Free-run rollout. x0 (B,6), U (B,T,3) -> X (B,T+1,6), Y_out (B,T,3)."""
    m = build_matrices(theta, Lb)
    xs = [x0]
    x = x0
    for k in range(U.shape[1]):
        x = rk4_step(theta, Lb, x, U[:, k, :], Ts, up_sample, Y_frozen, matrices=m)
        xs.append(x)
    X = torch.stack(xs, dim=1)
    P = m[5]
    Yout = torch.stack([X[:, k, :3] @ P for k in range(U.shape[1])], dim=1)
    return X, Yout


# ----------------------------------------------------------------------
# Designed evaluation points
# ----------------------------------------------------------------------

def design_points(n, seed, y_range=Y_RANGE, scale=None):
    """Independent (x, u) points covering the documented scheduling range.

    `scale` gives the per-channel spread; the defaults are physical magnitudes taken from
    the operating envelope (positions ~0.3 m, velocities ~0.5 m/s, forces ~100 N), not
    from any derived diagnostic artifact.
    """
    rng = np.random.default_rng(seed)
    if scale is None:
        scale = np.array([0.30, 0.02, 0.30, 0.50, 0.05, 0.50])
    x = rng.uniform(-1, 1, size=(n, 6)) * scale
    # Y = x[:, 2] is swept deterministically across the documented range so coverage is
    # a property of the design, not of the draw.
    x[:, 2] = np.linspace(y_range[0], y_range[1], n)
    u = rng.uniform(-1, 1, size=(n, 3)) * 100.0
    return (torch.tensor(x, dtype=torch.float64),
            torch.tensor(u, dtype=torch.float64))


def check_against_block(verbose=True):
    """Value check of this float64 model against the production Gantry_State_Block."""
    from model_augmentation.fit_systems.blocks import Gantry_State_Block

    theta, Lb = nominal()
    x, u = design_points(8, seed=0)

    blk = Gantry_State_Block(Y_op=None, Ts=TS, up_sample=UP_SAMPLE).float()
    z = torch.cat([x, u], dim=-1).to(torch.float32).unsqueeze(-1)   # (B, 9, 1)
    with torch.no_grad():
        x_blk = blk.nonlinear_function(z).squeeze(-1).to(torch.float64)
    x_here = rk4_step(theta, Lb, x, u)

    dx = (x_blk - x_here)
    rel = torch.linalg.norm(dx) / torch.linalg.norm(x_here)
    if verbose:
        print(f"  block vs standalone: rel 2-norm difference = {rel.item():.3e} "
              f"(float32 block vs float64 here)")
    return rel.item()


def mass_pd_margin(theta, Lb, y_range=Y_RANGE, n=201):
    """Minimum eigenvalue of M(Y) over the scheduling range."""
    M0, M1, M2, _, _, _ = build_matrices(theta, Lb)
    Ys = torch.linspace(y_range[0], y_range[1], n, dtype=torch.float64)
    mins = []
    for Y in Ys:
        MY = M0 + M1 * Y + M2 * Y ** 2
        mins.append(torch.linalg.eigvalsh((MY + MY.T) / 2).min())
    mins = torch.stack(mins)
    return mins.min().item(), Ys[mins.argmin()].item()
