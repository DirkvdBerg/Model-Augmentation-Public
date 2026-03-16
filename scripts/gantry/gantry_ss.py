"""
gantry_ss.py
------------
Computes the discrete-time state-space matrices (A, B, C, D) for the ASMPT
dual-gantry FP model in stage coordinates.

Replicates the following MATLAB pipeline from main.m + getss.m:
    1. Define physical parameters
    2. Build M(Y), C, K matrices                        [main.m]
    3. Build continuous-time SS in logical coordinates   [getss.m]
    4. Transform I/O to stage coordinates via P          [main.m]
    5. Discretize with ZOH at fs = 16 kHz               [main.m]

Coordinate systems:
    Internal states : logical [X, Theta, Y, dX, dTheta, dY]
    Inputs          : stage   [F_X1, F_X2, F_Y]
    Outputs         : stage   [X1, X2, Y]

Dimensions: nx=6, nu=3, ny=3

Reference: kamtin-fp-model/03 Simulink gantry/main.m
           kamtin-fp-model/03 Simulink gantry/functions/getss.m
"""

import numpy as np
from scipy.signal import cont2discrete


def gantry_discrete_ss(Y=0.3, fs=16e3):
    """
    Compute discrete-time A, B, C, D matrices for the gantry FP model.

    Parameters
    ----------
    Y  : float, payload position (m). Operating point for linearisation.
          Default: 0.3 m  (matches main.m)
    fs : float, sample frequency (Hz). Default: 16000 Hz (matches main.m)

    Returns
    -------
    A, B, C, D : np.ndarray
        Discrete-time state-space matrices (ZOH).
        A : (6, 6), B : (6, 3), C : (3, 6), D : (3, 3)
    """
    ts = 1.0 / fs

    # ------------------------------------------------------------------
    # Step 1: Physical parameters (from main.m)
    # ------------------------------------------------------------------
    mb  = 22.8    # Mass of moving cross-arm            [kg]
    mh  = 10.1    # Mass of payload (Y-axis)            [kg]
    m1  = 10.2    # Mass of actuator X1                 [kg]
    m2  = 10.7    # Mass of actuator X2                 [kg]

    Jb  = 1.0     # Rotary inertia of cross-arm         [kg.m^2]
    Jh  = 0.05    # Rotary inertia of payload            [kg.m^2]

    cg1 = 14.5    # Viscous friction actuator X1        [N/(m/s)]
    cg2 = 20.3    # Viscous friction actuator X2        [N/(m/s)]
    cy  = 10.0    # Viscous friction payload Y          [N/(m/s)]

    cb1 = 9.0     # Viscous friction elastic joint 1   [Nm/(rad/s)]
    cb2 = 9.0     # Viscous friction elastic joint 2   [Nm/(rad/s)]

    kb1 = 1987.5  # Stiffness elastic joint 1           [N.m/rad]
    kb2 = 1987.5  # Stiffness elastic joint 2           [N.m/rad]

    Lb  = 0.725   # Length of moving cross-arm          [m]
    d   = 0.1     # Distance cross-arm to payload       [m]

    # ------------------------------------------------------------------
    # Step 2: Build M(Y), C, K  (from main.m)
    # Logical coordinates: q = [X, Theta, Y]
    # ------------------------------------------------------------------
    M = np.array([
        [m1 + m2 + mb + mh,
         (m1 - m2) * Lb / 2 - mh * Y,
         0],
        [(m1 - m2) * Lb / 2 - mh * Y,
         Jb + Jh + (m1 + m2) * Lb**2 / 4 + mh * d**2 + mh * Y**2,
         -mh * d],
        [0,
         -mh * d,
         mh]
    ])

    C = np.array([
        [cg1 + cg2,              (cg1 - cg2) * Lb / 2,                    0],
        [(cg1 - cg2) * Lb / 2,   cb1 + cb2 + (cg1 + cg2) * Lb**2 / 4,   0],
        [0,                       0,                                        cy]
    ])

    K = np.array([
        [0,  0,          0],
        [0,  kb1 + kb2,  0],
        [0,  0,          0]
    ])

    # ------------------------------------------------------------------
    # Step 3: Continuous-time SS in logical coordinates  (from getss.m)
    #
    #   A_c = [  0    I  ]      B_c = [    0    ]
    #         [-M\K  -M\C]             [  M^{-1} ]
    #
    #   C_c = [I_3  0_3],  D_c = 0
    # ------------------------------------------------------------------
    n = 3
    MiK = np.linalg.solve(M, K)   # M^{-1} K
    MiC = np.linalg.solve(M, C)   # M^{-1} C
    Mi  = np.linalg.inv(M)        # M^{-1}

    A_c = np.block([
        [np.zeros((n, n)),  np.eye(n)],
        [-MiK,             -MiC      ]
    ])

    B_c = np.block([
        [np.zeros((n, n))],
        [Mi               ]
    ])

    C_c = np.block([np.eye(n), np.zeros((n, n))])

    D_c = np.zeros((n, n))

    # ------------------------------------------------------------------
    # Step 4: Transform I/O to stage coordinates  (from main.m)
    #
    #   P maps logical forces to stage forces: f_logical = P * f_stage
    #   Stage positions from logical:          q_stage   = P.T * q_logical
    #
    #   StageCoordinatesSystem = P.T * sys * P
    #   → B_stage = B_c @ P          (input side)
    #   → C_stage = P.T @ C_c        (output side)
    #   → A unchanged (internal state stays in logical coordinates)
    # ------------------------------------------------------------------
    P = np.array([
        [1,      1,       0],
        [Lb / 2, -Lb / 2, 0],
        [0,      0,       1]
    ])

    B_c_stage = B_c @ P
    C_c_stage = P.T @ C_c
    D_c_stage = P.T @ D_c @ P   # zero, but kept for completeness

    # ------------------------------------------------------------------
    # Step 5: Discretize with ZOH at ts = 1/fs  (from main.m)
    #   Equivalent to: G = c2d(StageCoordinatesSystem, ts, 'zoh')
    # ------------------------------------------------------------------
    A, B, C, D, _ = cont2discrete(
        (A_c, B_c_stage, C_c_stage, D_c_stage),
        dt=ts,
        method='zoh'
    )

    return A, B, C, D


# ----------------------------------------------------------------------
# Validation: compare against MATLAB G matrices
# Run from repo root: conda run -n GraduationProject python scripts/gantry/gantry_ss.py
# ----------------------------------------------------------------------
if __name__ == '__main__':
    import os
    from scipy.io import loadmat

    A, B, C, D = gantry_discrete_ss()

    print("Discrete-time matrices computed.")
    print(f"  A : {A.shape},  B : {B.shape},  C : {C.shape},  D : {D.shape}")

    # Structural checks
    eigs = np.linalg.eigvals(A)
    all_stable = np.all(np.abs(eigs) < 1.0)
    print(f"\nEigenvalue check (all inside unit circle): {'PASS' if all_stable else 'FAIL'}")
    print(f"  Max |eigenvalue| = {np.max(np.abs(eigs)):.6f}")

    print(f"\nD matrix is zero: {'PASS' if np.allclose(D, 0) else 'FAIL'}")

    # Numerical comparison against MATLAB G matrices
    mat_path = os.path.join(
        os.path.dirname(__file__), '..', '..',
        'Matlab-output', 'gantry_G_matrices.mat'
    )

    if os.path.exists(mat_path):
        mat = loadmat(mat_path)
        A_matlab = mat['A']
        B_matlab = mat['B']
        C_matlab = mat['C']
        D_matlab = mat['D']

        tol = 1e-10
        results = {
            'A': np.max(np.abs(A - A_matlab)),
            'B': np.max(np.abs(B - B_matlab)),
            'C': np.max(np.abs(C - C_matlab)),
            'D': np.max(np.abs(D - D_matlab)),
        }

        print("\nComparison against MATLAB G matrices (max absolute error):")
        all_pass = True
        for name, err in results.items():
            status = 'PASS' if err < tol else 'FAIL'
            if status == 'FAIL':
                all_pass = False
            print(f"  {name}: {err:.2e}  ->  {status}")

        print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED — check parameters or transform'}")
    else:
        print(f"\nMATLAB comparison skipped — file not found at:\n  {mat_path}")
        print("Run main.m in MATLAB and save G matrices first.")
