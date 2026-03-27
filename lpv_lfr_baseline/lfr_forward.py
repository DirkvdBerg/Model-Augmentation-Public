"""
lfr_forward.py
--------------
Resolve-and-retain forward pass for the dual-gantry LPV-LFR baseline.

Implements the step-by-step computation described in README.md and
docs/lfr-baseline-implementation-method.md.

The algebraic loop z = Cz*x + Dzw*w + Dzu*u, w = Y*I6*z (Dzw != 0) is resolved
analytically in a forward sequence. z and w are retained as explicit tensors
rather than discarded — this is the key distinction from a full collapse to A_c(Y)*x + B_c(Y)*u.

Forward pass steps:
    1. fnet   = [-K, -C] @ x + u
    2. v      = M(Y)^{-1} @ fnet          (loop resolved: unique fixed point)
    3. v1     = Y * v
       v2     = Y * v1
    4. z      = cat([v;  v1])  in R^6
       w      = cat([v1; v2])  in R^6
    5. xdot   = Ax @ x + Bw @ w + Bu @ u  (G matrix applied explicitly)
    6. y      = Cy @ x

Provides:
    lfr_forward(x, u, Y, G, M1, M2, M0) -> (xdot, z, w, y)
        All inputs and outputs are torch tensors, dtype=float64.
        x      : (6,)  state [q; qdot] in logical coordinates
        u      : (3,)  input [f_X1, f_X2, f_Y] in stage coordinates (after P transform)
        Y      : ()    scheduling variable (payload Y-position) [m]
        G      : GMatrix  constant G matrix entries from lfr_matrices.py
        returns xdot (6,), z (6,), w (6,), y (3,)

Note on coordinates:
    The derivation is in logical coordinates (x = [X, Theta, Y, dX, dTheta, dY]).
    Stage coordinate transform (P matrix) is applied to u before entering this function
    and to y after — see lfr_simulate.py.
    Open question: whether z and w should be exposed in logical or stage coordinates
    for Jan's framework integration (see docs/lfr-baseline-implementation-method.md).
"""
