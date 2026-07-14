"""Small shared helper: capture the trained ANN's dY-row output over a short V1 free-run
(the d6 shadow pattern; used by g_deck.py for the G6 force panel)."""
__project_origin__ = "added"

import os
import sys

import numpy as np
import torch
import deepSI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import demo_common as dm
from demo_common import CFG, REPO
from model_augmentation.fit_systems.blocks import Static_ANN_Block

CKPT = os.path.join(REPO, 'simulations', 'gantry_subnet', 'diagnostics',
                    'checkpoints', 'gantry_drift_last.pth')


def capture_force_trace(fit_sys, seconds=2.0, file='V1_standstill_Yp10.mat'):
    """Load gantry_drift_last into fit_sys, free-run `seconds` of V1, return (t, w_dY)."""
    fit_sys.__dict__ = torch.load(CKPT, map_location='cpu', weights_only=False)
    fit_sys.hfn.eval()
    ann = next(m for m in fit_sys.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
    u, y, _, _ = dm.load_T(file)
    n = int(seconds / CFG.ts_new)
    records = []
    orig = ann.forward

    def shadow(z):
        w = orig(z)
        records.append(w.detach().view(-1, ann.nw).cpu().numpy())
        return w

    ann.forward = shadow
    try:
        fit_sys.apply_experiment(deepSI.System_data(u=u[:n], y=y[:n], dt=CFG.ts_new))
    finally:
        ann.forward = orig
    w = np.concatenate(records, axis=0)[:, 5]
    return np.arange(len(w)) * CFG.ts_new, w
