"""
verify_interconnect.py
----------------------
Verify that the full Interconnect pipeline (Gantry_State_Block + Linear_Output_Block)
correctly propagates states and outputs over a short trajectory when driven from a
known initial state — bypassing the encoder entirely.

Approach
--------
1. Generate a reference trajectory by stepping Gantry_State_Block directly.
2. Build Interconnect(nx=6, nu=3, ny=3) with the same block instances wired up.
3. Step the interconnect from true x0 for T steps.
4. Compare: shape, NaN/Inf, non-flat channels, numerical match.

Because both the reference and the interconnect use the SAME block instance, any
wiring error will produce a mismatch (wrong z layout, wrong output routing, etc.),
while correct wiring gives bit-identical outputs.

Run from project root:
    conda run -n GraduationProject python scripts/gantry/verification/verify_interconnect.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch
import numpy as np
from model_augmentation.fit_systems.blocks import Gantry_State_Block, Linear_Output_Block
from model_augmentation.fit_systems.interconnect import Interconnect
from model_augmentation.systems.gantry_ss import Cd, Dd

TOL = 1e-5   # float32 — same block instance → bit-identical, so tolerance is generous

T = 200  # simulation horizon

# ---------------------------------------------------------------------------

def check_bool(name, condition):
    status = 'PASS' if condition else 'FAIL'
    print(f'  {name:<55s}  {status}')
    return bool(condition)

def check_err(name, value, tol):
    status = 'PASS' if value < tol else 'FAIL'
    print(f'  {name:<55s}  max|err| = {value:.2e}   {status}')
    return status == 'PASS'

# ---------------------------------------------------------------------------

def build_interconnect(gantry_block, output_block):
    """Wire gantry_block and output_block into a bare Interconnect."""
    interconnect = Interconnect(nx=6, nu=3, ny=3, debugging=False)
    interconnect.add_block(gantry_block)
    interconnect.add_block(output_block)

    # State block: z = [x(6); u(3)] → xp
    interconnect.connect_signals("x", gantry_block)
    interconnect.connect_block_signals(gantry_block, ["u"], [])
    interconnect.connect_signals(gantry_block, "xp")

    # Output block: z = [x(6); u(3)] → y
    interconnect.connect_signals("x", output_block)
    interconnect.connect_block_signals(output_block, ["u"], ["y"])

    return interconnect


def run():
    print('=' * 60)
    print('verify_interconnect.py')
    print('=' * 60)

    results = []

    Y_op  = 0.3
    std_x = np.ones((6, 1))   # no normalisation — physical == normalised
    std_u = np.ones((3, 1))

    # -----------------------------------------------------------------------
    # 1. Reference trajectory — step Gantry_State_Block directly
    # -----------------------------------------------------------------------
    gantry_block  = Gantry_State_Block(Y_op=Y_op, std_x=std_x, std_u=std_u)
    gantry_block.eval()
    output_block  = Linear_Output_Block(C=Cd, D=Dd)
    output_block.eval()

    x0_np = np.array([0.05, 0.01, 0.30, 0.02, -0.01, 0.05], dtype=np.float32)
    u_np  = np.array([10.0, -5.0, 3.0], dtype=np.float32)

    Cd_np = Cd.numpy()
    Dd_np = Dd.numpy()

    x_ref = np.zeros((T + 1, 6), dtype=np.float32)
    y_ref = np.zeros((T, 3),     dtype=np.float32)
    x_ref[0] = x0_np

    with torch.no_grad():
        for t in range(T):
            x_t = torch.from_numpy(x_ref[t]).reshape(1, 6, 1)
            u_t = torch.from_numpy(u_np).reshape(1, 3, 1)
            z_t = torch.cat([x_t, u_t], dim=1)          # (1, 9, 1)
            x_ref[t + 1] = gantry_block(z_t).squeeze().numpy()
            y_ref[t]     = Cd_np @ x_ref[t] + Dd_np @ u_np

    # -----------------------------------------------------------------------
    # 2. Build Interconnect and step from true x0
    # -----------------------------------------------------------------------
    interconnect = build_interconnect(gantry_block, output_block)
    interconnect.eval()

    x_t      = torch.from_numpy(x0_np).reshape(1, 6)
    u_const  = torch.from_numpy(u_np).reshape(1, 3)

    y_pred = np.zeros((T, 3), dtype=np.float32)

    with torch.no_grad():
        for t in range(T):
            y_t, x_next = interconnect(x_t, u_const)  # y_t: (1,3), x_next: (1,6)
            y_pred[t] = y_t.squeeze().numpy()
            x_t = x_next

    x_final_pred = x_t.squeeze().numpy()

    # -----------------------------------------------------------------------
    # 3. Checks
    # -----------------------------------------------------------------------
    print()

    # Shape
    results.append(check_bool('y_pred shape == (T=200, 3)',  y_pred.shape == (T, 3)))

    # Numerical health
    results.append(check_bool('no NaN in y_pred',  not np.isnan(y_pred).any()))
    results.append(check_bool('no Inf in y_pred',  not np.isinf(y_pred).any()))

    # All 3 channels carry signal (non-flat)
    for ch in range(3):
        results.append(check_bool(f'channel {ch} is non-flat (std > 1e-8)',
                                  float(y_pred[:, ch].std()) > 1e-8))

    # Numerical match: output
    err_y = float(np.abs(y_pred - y_ref).max())
    results.append(check_err('y_pred matches y_ref over T=200 steps', err_y, TOL))

    # Numerical match: final state
    err_x = float(np.abs(x_final_pred - x_ref[T]).max())
    results.append(check_err('final state matches reference',           err_x, TOL))

    print()
    overall = all(results)
    print(f'Overall: {"ALL PASS" if overall else "SOME FAILED"}')
    return overall


if __name__ == '__main__':
    ok = run()
    sys.exit(0 if ok else 1)
