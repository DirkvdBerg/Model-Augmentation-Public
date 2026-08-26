"""The controller and sensitivity as state-space objects at a chosen sample rate.

MIGRATION step 7: what this file was FOR is gone. It held loss variants A, B and C as mixins
grafted onto the fit-system instance by rebinding its class to one built at runtime, which is the
machinery the migration removes: the closed loop is now
`model_augmentation/fit_systems/closed_loop.py`, attached as a declared `simulator` attribute, and
variant C's So-weighted loss was never taken past the experiment. Both mixins and `attach()` are
deleted.

What remains is the state-space construction, which several verification scripts and
`cl_controller.build_controller_bank` still use, and which plan 3.9 keeps on the gantry side
because it is the FP model's controller and not a framework capability:

  controller_ss(Y_op, ts)   Cfb at the given rate, as one 3-in 3-out state space
  sensitivity_ss(Y_op, ts)  So = (I + Gop Cfb)^-1 at the given rate

SAMPLE RATE. The pipeline runs at cfg.fs_new (4 kHz by default) while Cfb was designed at 20 kHz.
p2_rate_compare.py measured the 4 kHz loop's sensitivity peak 15.3 % higher in the absorber band.
The controller is re-discretised at the pipeline rate, and `test_controller_exact.py` L5 checks
that re-discretisation against MATLAB at exactly that rate.
"""
__project_origin__ = "added"

import numpy as np
import torch

from p2_rate_compare import build_cfb_at
import so_filter as SOF
from scipy.signal import cont2discrete, tf2ss


def _tf_to_ss_batch(cfb):
    """Per-channel (b, a) -> block-diagonal (A, B, C, D) for the 3-channel diagonal Cfb."""
    As, Bs, Cs, Ds = [], [], [], []
    for b, a in cfb:
        A, B, C, D = tf2ss(b, a)
        As.append(A); Bs.append(B); Cs.append(C); Ds.append(D)
    n = sum(A.shape[0] for A in As)
    A = np.zeros((n, n)); B = np.zeros((n, 3)); C = np.zeros((3, n)); D = np.zeros((3, 3))
    i = 0
    for j, (Aj, Bj, Cj, Dj) in enumerate(zip(As, Bs, Cs, Ds)):
        m = Aj.shape[0]
        A[i:i + m, i:i + m] = Aj
        B[i:i + m, j] = Bj.ravel()
        C[j, i:i + m] = Cj.ravel()
        D[j, j] = Dj.ravel()[0]
        i += m
    return A, B, C, D


def controller_ss(Y_op, ts):
    """Cfb at the given rate, as one 3-in 3-out state space."""
    cfb, _ = build_cfb_at(Y_op, ts)
    return _tf_to_ss_batch(cfb)


def sensitivity_ss(Y_op, ts):
    """So = (I + Gop Cfb)^-1 at the given rate."""
    ctrl = controller_ss(Y_op, ts)
    return SOF.so_ss(Y_op, ctrl, ts=ts)
