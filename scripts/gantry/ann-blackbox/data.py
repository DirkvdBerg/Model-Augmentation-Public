"""Record loading and rate change, shared by ann_blackbox.py and oversampling_diagnostic.py.

DEV(Jan L10-L12): Jan's script loads a prepared .npz inline in 3 lines and needs no module.
This is its own file for one reason: the trainer and the diagnostic must decimate identically,
and a duplicated loader would let the sweep arms and the diagnostic drift apart silently.
"""
__project_origin__ = "added"

import os
import sys
import numpy as np
from scipy.signal import decimate            # DEV: the brief forbids decimating y without an anti-alias filter; plant.load_record point-samples it
import deepSI

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry', 'msd-offset'))
from plant import load_record                # DEV(Jan L11-12): gantry records are MATLAB .mat, not deepSI .npz; brief says import this loader, do not reimplement

FS_NATIVE = 4000.0                           # DEV: plant.load_record's own rate; all decimation happens from here so every arm shares one source


def load(name, fs):
    """One deepSI.System_data per record, decimated to fs with an anti-alias filter."""
    rec = load_record(name, fs_new=int(FS_NATIVE))   # always load at 4 kHz, then decimate here
    q = int(round(FS_NATIVE / fs))
    u, y = rec['u'], rec['y']                        # u = stage force [N] (3), y = stage position [m] (3): [x1, x2, Y]
    if q > 1:
        n = (len(u) // q) * q
        u = u[:n].reshape(-1, q, 3).mean(axis=1)     # block mean on u, consistent with D-087 (u is ZOH, so the mean is the energy-preserving reduction)
        y = decimate(y, q, ftype='fir', zero_phase=True, axis=0)[:len(u)]   # DEV: zero-phase FIR anti-alias before point-sampling y
    # DEV: ascontiguousarray is load-bearing, not cosmetic. .mat arrays come back Fortran-ordered,
    # deepSI's default_encoder_net.forward calls .view (encoders.py:122), and .view on the resulting
    # non-C-contiguous tensor raises. Verified: without this, fit() dies on its first validation.
    return deepSI.System_data(u=np.ascontiguousarray(u, dtype=float),
                              y=np.ascontiguousarray(y, dtype=float), dt=1.0 / fs)
