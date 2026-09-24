__project_origin__ = "added"  # entire file is project-specific, not Jan's original framework

"""
gantry_linearization.py
-----------------------
Linearize the gantry CT-LPV model at a frozen operating point and discretize.

Returns (Ad, Bd, Cd, Dd) as numpy arrays, ready for ``linear_encoder_init``.

THEORY: At frozen Y_op the CT state-space is
    x = [q; qdot]  (6 states, logical coordinates)
    xdot = Ac @ x + Bc @ u_logical
    y    = Cc @ x + Dc @ u

where
    Ac = [[0, I]; [-M0inv@K, -M0inv@C]]   (from build_G_matrix_entries → Ax)
    Bc = [[0]; [M0inv @ P]]               (from build_G_matrix_entries → Bu)
    Cc = [P^T | 0]                        (= gantry_ss.Cd)
    Dc = 0                                (= gantry_ss.Dd)

Note: Ac equals Ax from build_G_matrix_entries **only at Y_op=0** where Bw@w
vanishes (w depends on Y * z, and z depends on Y, so at Y=0 the LFR feedback
is zero).

CHANGED (D-167): Y_op != 0 IS now implemented, and the earlier claim that it "must account for
the LFR loop gain" was wrong as stated. The LFR is a REALISATION of M(Y)^{-1} with constant
matrices plus a feedback path; it is not extra dynamics. Freezing Y lets M(Y)^{-1} be evaluated
in the closed form the LFR itself encodes,

    M(Y)^{-1} = N(Y) / d(Y),
    N(Y) = N0 + Y N1 + Y^2 N2                                  (gantry_ss.build_poly_constants)
    d(Y) = mh (alpha gamma - beta^2 + 2 beta mh Y + mh (alpha - mh) Y^2)

so the frozen-Y linearisation is simply

    Ac(Y) = [[0, I], [-M(Y)^{-1} K, -M(Y)^{-1} C]],   Bu(Y) = [[0], [M(Y)^{-1}]]

with Bu in LOGICAL forces, exactly as build_G_matrix_entries defines it. Verified against the
pipeline's own Gantry_State_Block by central-difference Jacobian at Y = -0.36 .. +0.36:
max|Ad_fd - Ad_zoh| = 6.2e-09 to 9.6e-09, no worse off-centre than at Y = 0.
At Y = 0 the polynomial reduces to N0/d0 term by term, so this path is numerically identical to
the previous one and the change is an exact no-op wherever it is already used.
"""

import numpy as np
import scipy.signal
import torch

from model_augmentation.systems import gantry_ss as gss


def gantry_linearize_and_discretize(dt: float, Y_op: float = 0.0):
    """
    Linearize the gantry model at frozen Y_op and discretize at sample period dt.

    Parameters
    ----------
    dt : float
        Discrete-time sample period [s], e.g. 1/4000 for FS_NEW = 4 kHz.
    Y_op : float
        Frozen operating point for q3 [m]. Any value in the machine's operating
        range is supported (D-167); the model is well posed wherever d(Y) != 0,
        which over the measured range Y in [-0.36, +0.36] holds with a large
        margin (min d(Y) = 2062, min eig M(Y) = 3.74).

    Returns
    -------
    Ad, Bd, Cd, Dd : np.ndarray
        Discrete-time state-space matrices (float64).
    """
    # --- Compute CT matrices from gantry_ss ---
    # THEORY: build_poly_constants gives N0, N1, N2 for M(Y)^{-1} = N(Y)/d(Y)
    alpha, beta, gamma, N0, N1, N2 = gss.build_poly_constants(
        gss.m1, gss.m2, gss.mb, gss.mh, gss.Jb, gss.Jh, gss.Lb, gss.d
    )
    # CHANGED (D-167): the FROZEN-Y numerator and denominator, evaluated in the same torch dtype
    # flow as before so that Y_op = 0 reduces term by term to the previous N0 / d0 and the change
    # is an exact no-op there. THEORY: lpv_lfr_baseline/core/lfr_forward.py:70-76.
    #   d(Y) = mh (alpha gamma - beta^2 + 2 beta mh Y + mh (alpha - mh) Y^2)
    #   N(Y) = N0 + Y N1 + Y^2 N2
    dY = gss.mh * (alpha * gamma - beta ** 2
                   + 2 * beta * gss.mh * Y_op
                   + gss.mh * (alpha - gss.mh) * Y_op ** 2)
    NY = N0 + Y_op * N1 + Y_op ** 2 * N2

    # THEORY: build_G_matrix_entries returns Ax = [[0, I]; [-Minv@K, -Minv@C]]
    # and Bu = [[0]; [Minv]]  (input in logical coordinates). It takes the numerator and the
    # denominator as arguments and forms Minv = N/d internally, so passing the frozen-Y pair
    # gives the frozen-Y linearisation with no second implementation of the same algebra.
    Ax, _Bw, Bu, _A_combined = gss.build_G_matrix_entries(
        NY, dY, gss.M1, gss.M2, gss.K, gss.C
    )

    Ac = Ax.numpy().astype(np.float64)  # (6, 6)
    Bc = Bu.numpy().astype(np.float64)  # (6, 3) — maps logical forces to state
    # THEORY: Bu maps logical forces. The encoder sees stage forces u_stage,
    # and logical = P @ stage. So Bc_stage = Bu @ P.
    Bc_stage = Bc @ gss.P.numpy().astype(np.float64)  # (6, 3) — maps stage forces

    Cc = gss.Cd.numpy().astype(np.float64)  # (3, 6): y_stage = P^T @ q
    Dc = gss.Dd.numpy().astype(np.float64)  # (3, 3): zero

    # --- Discretize with ZOH ---
    Ad, Bd, Cd, Dd, _ = scipy.signal.cont2discrete(
        (Ac, Bc_stage, Cc, Dc), dt, method='zoh'
    )

    return Ad, Bd, Cd, Dd


def verify_linearization(dt: float, atol: float = 1e-4):
    """
    Quick sanity check: compare Ad@x + Bd@u vs the analytical CT solution
    for a small perturbation at Y_op=0.

    Prints max absolute error. Should be O(dt^2) or smaller.
    """
    Ad, Bd, Cd, Dd = gantry_linearize_and_discretize(dt)
    nx, nu, ny = 6, 3, 3

    # Small perturbation around equilibrium
    rng = np.random.default_rng(42)
    x0 = rng.standard_normal(nx) * 1e-3
    u0 = rng.standard_normal(nu) * 1.0

    # Linear prediction
    x1_lin = Ad @ x0 + Bd @ u0
    y0_lin = Cd @ x0 + Dd @ u0

    # RK4 prediction using CT matrices
    Ac = np.zeros((nx, nx))
    Ac[:3, 3:] = np.eye(3)
    alpha, beta, gamma, N0, _, _ = gss.build_poly_constants(
        gss.m1, gss.m2, gss.mb, gss.mh, gss.Jb, gss.Jh, gss.Lb, gss.d
    )
    d0 = gss.mh * (alpha * gamma - beta ** 2)
    M0inv = (N0 / d0).numpy().astype(np.float64)
    K_np = gss.K.numpy().astype(np.float64)
    C_np = gss.C.numpy().astype(np.float64)
    P_np = gss.P.numpy().astype(np.float64)
    Ac[3:, :3] = -M0inv @ K_np
    Ac[3:, 3:] = -M0inv @ C_np
    Bc = np.zeros((nx, nu))
    Bc[3:, :] = M0inv @ P_np

    def deriv(x, u):
        return Ac @ x + Bc @ u

    # RK4
    k1 = deriv(x0, u0)
    k2 = deriv(x0 + 0.5 * dt * k1, u0)
    k3 = deriv(x0 + 0.5 * dt * k2, u0)
    k4 = deriv(x0 + dt * k3, u0)
    x1_rk4 = x0 + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    err = np.max(np.abs(x1_lin - x1_rk4))
    print(f"Linearization verification (dt={dt:.6f}s):")
    print(f"  Max |x1_zoh - x1_rk4| = {err:.2e}")
    print(f"  (should be small, O(dt^2) = {dt**2:.2e})")
    return err


if __name__ == "__main__":
    dt = 1.0 / 4000  # FS_NEW = 4 kHz
    Ad, Bd, Cd, Dd = gantry_linearize_and_discretize(dt)
    print("Discrete-time state-space at Y_op=0, dt=1/4000:")
    print(f"  Ad shape: {Ad.shape}, rank: {np.linalg.matrix_rank(Ad)}")
    print(f"  Bd shape: {Bd.shape}")
    print(f"  Cd shape: {Cd.shape}, rank: {np.linalg.matrix_rank(Cd)}")
    print(f"  Dd shape: {Dd.shape}")

    # Check observability rank
    from numpy.linalg import matrix_rank
    O = np.vstack([Cd @ np.linalg.matrix_power(Ad, i) for i in range(6)])
    print(f"  Observability matrix rank: {matrix_rank(O)} (need 6 for full observability)")

    print()
    verify_linearization(dt)
