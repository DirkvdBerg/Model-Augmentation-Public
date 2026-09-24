"""The G2 encoder variant: a Y-scheduled reconstructability map built from the model it feeds (ET-004).

    x(k) = parent(u, y) + sum_{k=1..deg} (T_k(s) - T_k(0)) (C^y_k y_pure + C^u_k u_pure),  s = Y / S

* `parent` is a `linear_encoder_init_aug` (the production class). Its parameters ARE `W_0`, so with
  `deg = 0` the forward returns `parent(u, y)` untouched: the collapse gate is bitwise by
  construction, not by tolerance.
* `Y` is the MEASURED stage `Y` at the window's last sample, de-normalised from the encoder's own
  `yhist` (channel 2 = logical `q3`), so the map stays deployable (D-167).
* `C_k` are fitted by least squares to EXACT frozen-Y maps with `W(0)` pinned (D-167 (a0)); the basis
  is Chebyshev in `s` (ET-004) and `T_k(s) - T_k(0)` vanishes at `Y = 0`.
* `source='planted'`: the parent's W^b (6 rows) and W^a (2 rows) are overwritten with the planted
  8-state model's map at `Y = 0`, so the latent rows are initialised in the planted realisation
  (`[da, vda] / sa`), consistently with the model the encoder feeds.
"""
__project_origin__ = "added"

import os
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import et_paths                                                  # noqa: E402,F401
import recon                                                     # noqa: E402
from model_augmentation.fit_systems.pre_encoder import linear_encoder_init_aug   # noqa: E402
from core import (S_SCALE, FIT_LO, FIT_HI, FIT_STEP, cheb_basis, pure_map, grid_maps,   # noqa: E402,F401
                  fit_schedule, numpy_sched, ScheduledReconEncoder)   # ET-006: moved to core.py

def make_parent(source, pipeline_encoder, norm, dt, n, sx6, su, sy, sa, dtype, hp):
    """`baseline`: a deep copy of the pipeline's own encoder (W_0 = production, bitwise).
    `planted`: same class, 6 + 2 rows, W^b and W^a overwritten with the planted map at Y = 0."""
    import copy
    if source == 'baseline':
        return copy.deepcopy(pipeline_encoder).to(dtype)
    Ad, Bd, Cd, Dd = recon.baseline_dt(dt, 0.0)
    # Only the construction path is shared with production; the maps are replaced below.
    enc = linear_encoder_init_aug(
        A=Ad, B=Bd, C=Cd, D=Dd, nx=6, nu=3, ny=3, na=n, nb=n, nx_aug=2,
        n_nodes_per_layer=hp['n_nodes_per_layer'], n_hidden_layers=hp['n_hidden_layers'],
        flag_linear_only=False, u_mean=norm.u_mean, std_u=norm.std_u, y0=norm.y0, ystd=norm.ystd,
        x_mean=norm.x_mean, std_x=norm.std_x, dtype=dtype).to(dtype)
    Wy, Wu = pure_map(*recon.planted_dt(dt, 0.0), n, np.concatenate([sx6, sa]), su, sy)
    with torch.no_grad():
        enc.Wb_psi_y.copy_(torch.as_tensor(Wy[:6], dtype=dtype))
        enc.Wb_psi_u.copy_(torch.as_tensor(Wu[:6], dtype=dtype))
        enc.Wa_psi_y.copy_(torch.as_tensor(Wy[6:], dtype=dtype))
        enc.Wa_psi_u.copy_(torch.as_tensor(Wu[6:], dtype=dtype))
    return enc
