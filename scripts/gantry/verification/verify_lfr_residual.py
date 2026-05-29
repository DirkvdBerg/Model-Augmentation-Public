"""
verify_lfr_residual.py
----------------------
Verify that the rational form N(Y)/d(Y) correctly inverts M(Y):
    residual = M(Y) @ a - fnet  should be ≈ 0

Mirrors lfr_forward.py Check 1. Validates that build_poly_constants()
produces the correct adjugate/determinant decomposition in float32.

Run from project root:
    conda run -n GraduationProject python scripts/gantry/verification/verify_lfr_residual.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch
from model_augmentation.systems.gantry_ss import (
    mh, m1, m2, mb, Jb, Jh, Lb, d,
    M0, M1, M2, K, C, P,
    build_poly_constants,
)

TOL = 1e-4   # float32 — looser than float64 (1e-10 in lfr_forward.py Check 1)

def check(name, value, tol):
    status = 'PASS' if value < tol else 'FAIL'
    print(f'  {name:<55s}  max|res| = {value:.2e}   {status}')
    return status == 'PASS'

def run():
    print('=' * 60)
    print('verify_lfr_residual.py')
    print('=' * 60)

    alpha, beta, gamma, N0, N1, N2 = build_poly_constants(m1, m2, mb, mh, Jb, Jh, Lb, d)

    # Test a range of Y values including the typical operating point
    test_Y = [0.0, 0.1, 0.3, -0.2, 0.35]
    batch = len(test_Y)

    # Fixed test state and stage input (logical coordinates after P)
    x_test    = torch.tensor([0.05, 0.01, 0.30, 0.02, -0.01, 0.05], dtype=torch.float32)
    u_stage   = torch.tensor([10.0, -5.0, 3.0], dtype=torch.float32)
    u_logical = P @ u_stage   # stage -> logical

    fnet = -(x_test[:3] @ K.T) - (x_test[3:] @ C.T) + u_logical  # (3,)

    print()
    results = []
    for y_val in test_Y:
        Y_t = torch.tensor(y_val, dtype=torch.float32)

        # Rational inverse: N(Y)/d(Y)
        dY  = mh * (alpha * gamma - beta**2
                    + 2 * beta * mh * Y_t
                    + mh * (alpha - mh) * Y_t**2)
        N_Y = N0 + N1 * Y_t + N2 * Y_t**2
        a   = N_Y @ fnet / dY   # (3,)

        # Check: M(Y) @ a should equal fnet
        M_Y      = M0 + M1 * Y_t + M2 * Y_t**2
        residual = (M_Y @ a - fnet).abs().max().item()

        results.append(check(f'Y = {y_val:+.2f} m   M(Y)@a - fnet', residual, TOL))

    print()
    overall = all(results)
    print(f'Overall: {"ALL PASS" if overall else "SOME FAILED"}')
    return overall

if __name__ == '__main__':
    ok = run()
    sys.exit(0 if ok else 1)
