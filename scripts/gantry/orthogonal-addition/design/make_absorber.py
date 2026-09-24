"""OA-011 / OA-019: an absorber-only addition, lossless, along Y at the payload (b = [0, -d, 1]).

Usage: make_absorber.py [f_hz=150] [mass_kg=5.05] [out=design/addition_c_abs.mat]
Default = the production absorber made lossless and centred: m = 0.50 * mh = 5.05 kg, f = 150 Hz
(gtd_config: fa = 150, ka = ma (2 pi fa)^2), L0 = 0, zeta = 0.
"""
__project_origin__ = "added"

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import addition_io as aio                              # noqa: E402

MH, D = 10.1, 0.1                                      # gtd_config.m, gantry_ss
f_hz = float(sys.argv[1]) if len(sys.argv) > 1 else 150.0
mass = float(sys.argv[2]) if len(sys.argv) > 2 else 0.50 * MH
out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, 'addition_c_abs.mat')
oa = aio.empty()
oa['oa_m'][0, 0] = mass
oa['oa_w'][0, 0] = 2.0 * np.pi * f_hz
oa['oa_b0'][:, 0] = [0.0, -D, 1.0]
oa['oa_b1'][:, 0] = [0.0, 0.0, 0.0]
aio.save_mat(out, oa, 'absorber only: %.4g kg, %.4g Hz, Y at the payload, b = [0, -d, 1], lossless'
             % (mass, f_hz))
print(out, oa['oa_m'][0, 0], oa['oa_w'][0, 0] / (2 * np.pi), oa['oa_b0'][:, 0])
