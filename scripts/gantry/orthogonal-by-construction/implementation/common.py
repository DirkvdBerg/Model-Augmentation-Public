"""Shared rig for the OBC state-space implementation stages.

__project_origin__ = "added"

Block construction at the production pipeline's settings, the designed evaluation points, the
ten identifiable combinations and the declared gauge section from the ten back to the fourteen
raw scalars. Nothing here runs on import; nothing here is library code (the library lives in
`model_augmentation/`).

Settings are read off `scripts/gantry/gantry_dynamic/config.py`: `fs_new = 4000` so
`Ts = 2.5e-4 s`, `up_sample = 2`, `Y_op = None` (LPV self-scheduling), `nx_ann = 2`,
`ann_route_ix = (0..7)`, `nf = 400`. Identity normalisation (`std_x = 1`, `x_mean = 0`) is used
deliberately so every reported number is in physical units and does not depend on a dataset.
"""

__project_origin__ = "added"

import os
import sys

import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

F64 = torch.float64

# --- pipeline settings (config.py: fs_new=4000, up_sample=2, nx_ann=2, nf_seconds=0.100) ----
TS = 1.0 / 4000.0
UP_SAMPLE = 2
NF = 400
N_P = 6                 # physical states
N_A = 2                 # additional (ANN latent) states, cfg.nx_ann
N_U = 3

# Documented scheduling range of the records: RECORD_Y_OP in gantry_dynamic/controller.py.
Y_RANGE = (-0.30, 0.30)

RAW_NAMES = ["kb1", "kb2", "cg1", "cg2", "cy", "cb1", "cb2",
             "mh", "m1", "m2", "mb", "Jb", "Jh", "d"]
COMBO_NAMES = ["kb_sum", "cg1", "cg2", "cy", "cb_sum", "mh",
               "m_total", "m_diff", "J_eff", "d"]
M_DIFF_IX = COMBO_NAMES.index("m_diff")


class Tee:
    """stdout mirror to a log file."""

    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


def nominal_raw():
    """Nominal (14,) raw parameter vector and Lb, float64, from gantry_ss."""
    from model_augmentation.systems import gantry_ss as g
    theta = torch.tensor([float(getattr(g, n)) for n in RAW_NAMES], dtype=F64)
    return theta, float(g.Lb)


def combos_from_raw(theta, Lb):
    """The ten identifiable combinations, exactly `_combos_from_raw` (blocks.py:1046-1058)."""
    p = dict(zip(RAW_NAMES, theta))
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


def gauge_section(vartheta, raw_init, Lb):
    """The declared gauge, delegated to the library so there is exactly ONE implementation.

    `Reduced_Gantry_State_Block.gauge_section` is canonical (D-190). Kept here as a thin alias
    because stage 0 was written against this name before the library class existed.
    """
    from model_augmentation.fit_systems.blocks import Reduced_Gantry_State_Block
    return Reduced_Gantry_State_Block.gauge_section(vartheta, raw_init, Lb)


def make_raw_block(Ts=TS, up_sample=UP_SAMPLE, params_init=None):
    """`Parameterized_Gantry_State_Block` at the pipeline's settings, float64, eval mode.

    `params_init` defaults to the nominal vector AS FLOAT64, so the block is built in float64
    from the start rather than constructed at float32 and cast afterwards. This matters: the
    reduced block forms `m_total` and `J_eff` at construction, and forming them at float32 and
    upcasting loses about 1e-7 relative, which reaches the velocity rows as a 2.3e-11 absolute
    difference. Measured in `01-reduced/`.
    """
    from model_augmentation.fit_systems.blocks import Parameterized_Gantry_State_Block
    if params_init is None:
        params_init = nominal_raw()[0]
    blk = Parameterized_Gantry_State_Block(
        Y_op=None, Ts=Ts, up_sample=up_sample,
        RMSE_baseline=1.0, flag_loss_reg=False, params_init=params_init,
    ).to(F64)
    blk.eval()
    return blk


def make_reduced_block(Ts=TS, up_sample=UP_SAMPLE, params_init=None, combo_init=None):
    """`Reduced_Gantry_State_Block` at the same settings, float64 from construction."""
    from model_augmentation.fit_systems.blocks import Reduced_Gantry_State_Block
    if params_init is None:
        params_init = nominal_raw()[0]
    blk = Reduced_Gantry_State_Block(
        Y_op=None, Ts=Ts, up_sample=up_sample,
        RMSE_baseline=1.0, flag_loss_reg=False, params_init=params_init,
        combo_init=combo_init,
    ).to(F64)
    blk.eval()
    return blk


def design_points(n, seed, y_range=Y_RANGE, scale=None, u_scale=100.0):
    """Independent physical (x, u) points covering the documented scheduling range.

    `scale` is the per-channel spread from the operating envelope (positions ~0.3 m,
    velocities ~0.5 m/s, forces ~100 N), the same design the investigation used so its numbers
    remain comparable. `Y = x[:, 2]` is swept deterministically, so coverage is a property of
    the design and not of the draw.
    """
    rng = np.random.default_rng(seed)
    if scale is None:
        scale = np.array([0.30, 0.02, 0.30, 0.50, 0.05, 0.50])
    x = rng.uniform(-1, 1, size=(n, N_P)) * scale
    x[:, 2] = np.linspace(y_range[0], y_range[1], n)
    u = rng.uniform(-1, 1, size=(n, N_U)) * u_scale
    return torch.tensor(x, dtype=F64), torch.tensor(u, dtype=F64)


def zu(x, u):
    """(B, nx+nu, 1) block input from (B, nx), (B, nu)."""
    return torch.cat([x, u], dim=1).unsqueeze(-1)
