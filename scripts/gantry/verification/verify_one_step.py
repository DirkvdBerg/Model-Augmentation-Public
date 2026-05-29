"""
verify_one_step.py
------------------
Compare one full RK4 step from Gantry_State_Block against a numpy
reference implementation of the same collapsed ODE.

Reference: collapsed form  xdot = [q_dot;  M(Y_op)^{-1} @ fnet]
Block:     LFR form        xdot = Ax@x + Bw@w + Bu@u_log

Both should be numerically equivalent (proven analytically — see plan).
Tolerance is float32 precision (~1e-5).

Run from project root:
    conda run -n GraduationProject python scripts/gantry/verification/verify_one_step.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch
import numpy as np
from model_augmentation.fit_systems.blocks import Gantry_State_Block
from model_augmentation.systems.gantry_ss import (
    M0, M1, M2, K, C, P, ts,
)

TOL = 1e-5   # float32 single-step

def check(name, value, tol):
    status = 'PASS' if value < tol else 'FAIL'
    print(f'  {name:<55s}  max|err| = {value:.2e}   {status}')
    return status == 'PASS'

# ---------------------------------------------------------------------------
# Numpy reference: collapsed ODE, one RK4 step (no normalisation)
# ---------------------------------------------------------------------------
M0_np = M0.numpy(); M1_np = M1.numpy(); M2_np = M2.numpy()
K_np  = K.numpy();  C_np  = C.numpy();  P_np  = P.numpy()
Ts    = float(ts)

def _xdot_np(x, u_stage, Y_op):
    """Collapsed xdot = [q_dot; M(Y_op)^{-1} @ fnet]."""
    u_log  = P_np @ u_stage
    q, qdot = x[:3], x[3:]
    fnet   = -K_np @ q - C_np @ qdot + u_log
    M_Y    = M0_np + M1_np * Y_op + M2_np * Y_op**2
    a      = np.linalg.solve(M_Y, fnet)
    return np.concatenate([qdot, a])

def rk4_np(x0, u_stage, Y_op, Ts, up_sample=10):
    x = x0.copy()
    h = Ts / up_sample
    for _ in range(up_sample):
        k1 = h * _xdot_np(x,        u_stage, Y_op)
        k2 = h * _xdot_np(x + k1/2, u_stage, Y_op)
        k3 = h * _xdot_np(x + k2/2, u_stage, Y_op)
        k4 = h * _xdot_np(x + k3,   u_stage, Y_op)
        x  = x + (k1 + 2*k2 + 2*k3 + k4) / 6
    return x

# ---------------------------------------------------------------------------

def run():
    print('=' * 60)
    print('verify_one_step.py')
    print('=' * 60)

    results = []

    test_cases = [
        ('small motion',   np.array([0.05, 0.01, 0.30,  0.02, -0.01,  0.05]),
                           np.array([10.0, -5.0, 3.0])),
        ('zero state',     np.zeros(6),
                           np.array([5.0, 5.0, 2.0])),
        ('large forces',   np.array([0.1, 0.02, 0.25, 0.05, 0.01, 0.03]),
                           np.array([50.0, -50.0, 20.0])),
    ]

    Y_op = 0.3
    std_x = np.ones((6, 1))   # no normalisation scaling
    std_u = np.ones((3, 1))

    block = Gantry_State_Block(Y_op=Y_op, std_x=std_x, std_u=std_u)
    block.eval()

    print()
    for name, x0_np, u_np in test_cases:
        # --- numpy reference ---
        xnext_ref = rk4_np(x0_np, u_np, Y_op, Ts)

        # --- block (normalised == physical since std=1) ---
        x_t = torch.tensor(x0_np, dtype=torch.float32).reshape(1, 6, 1)
        u_t = torch.tensor(u_np,  dtype=torch.float32).reshape(1, 3, 1)
        z_t = torch.cat([x_t, u_t], dim=1)   # (1, 9, 1)

        with torch.no_grad():
            xnext_block = block(z_t).squeeze().numpy()

        err = np.abs(xnext_block - xnext_ref).max()
        results.append(check(name, err, TOL))

    print()
    overall = all(results)
    print(f'Overall: {"ALL PASS" if overall else "SOME FAILED"}')
    return overall

if __name__ == '__main__':
    ok = run()
    sys.exit(0 if ok else 1)
