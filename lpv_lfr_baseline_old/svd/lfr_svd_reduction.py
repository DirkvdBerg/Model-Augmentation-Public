"""
lfr_svd_reduction.py
--------------------
SVD-based latent dimension reduction of the dual-gantry LPV-LFR realization.

Derived from: LPV/LFR-SVD-derivation.tex, Method 2.

Starting point: the 6-channel LFR with Δ(Y) = Y·I₆ from lfr_matrices.py.
Result:         a 4-channel LFR with Δ(Y) = Y·I₄.

The reduction is lossless (exact, not approximate) and structural: it follows
from rank(D_zw) = 4 for all physically valid parameters (mh > 0, M0 ≻ 0).

Two-stage procedure (Method 2 from the derivation):

  Stage 1 — Pre-compress D_zw
      D_zw = S2 @ P       (compact SVD: S2 ∈ R^{6×4}, P ∈ R^{4×6})
      B_w  = E @ S2 @ P   (structural identity, E selects lower block)
      Compressed loop variable: ŵ = P @ w  ∈ R^4

  Stage 2 — SVD on the compressed stacked matrix
      S_new = [B_w; P] @ [C_z, S2, D_zu]  ∈ R^{10×13},  rank = 4
      S_new = L_new @ R_new  (compact SVD, split by row/col structure)
      L_new = [Lx; Lz],    Lx ∈ R^{6×4},  Lz ∈ R^{4×4}
      R_new = [Rx, Rw, Ru], Rx ∈ R^{4×6},  Rw ∈ R^{4×4},  Ru ∈ R^{4×3}

  Reduced G̃ matrix entries:
      B̃_w  = Lx              (6×4)
      C̃_z  = Rx              (4×6)
      D̃_zw = Rw @ Lz         (4×4)
      D̃_zu = Ru              (4×3)
      A_x, B_u, C_y          unchanged from G

Provides:
    GMatrixReduced  dataclass holding all entries as plain torch.float64 tensors.
    build_reduced_G_matrix(G) -> GMatrixReduced
    G_reduced       module-level singleton computed from the fixed G.
"""

from dataclasses import dataclass

import torch

from lpv_lfr_baseline.core.lfr_matrices import G, GMatrix


@dataclass
class GMatrixReduced:
    """
    Reduced G̃ matrix entries for the 4-channel LPV-LFR realization.

    Scheduling block: Δ(Y) = Y·I₄  (4 latent channels, down from 6).
    Unchanged entries (A_x, B_u, C_y) are carried from the original G.

    Shapes:
        Ax  : (6, 6)   — unchanged
        Bw  : (6, 4)   — reduced  (was 6×6)
        Bu  : (6, 3)   — unchanged
        Cz  : (4, 6)   — reduced  (was 6×6)
        Dzw : (4, 4)   — reduced  (was 6×6)
        Dzu : (4, 3)   — reduced  (was 6×3)
        Cy  : (3, 6)   — unchanged

    Also stores the intermediate SVD factors for inspection/verification:
        S2  : (6, 4)   — left factor of D_zw = S2 @ P
        P   : (4, 6)   — right factor (row-compressed loop projection)
    """
    Ax:  torch.Tensor   # (6, 6)
    Bw:  torch.Tensor   # (6, 4)
    Bu:  torch.Tensor   # (6, 3)
    Cz:  torch.Tensor   # (4, 6)
    Dzw: torch.Tensor   # (4, 4)
    Dzu: torch.Tensor   # (4, 3)
    Cy:  torch.Tensor   # (3, 6)
    S2:  torch.Tensor   # (6, 4)  intermediate factor
    P:   torch.Tensor   # (4, 6)  intermediate factor


def build_reduced_G_matrix(G: GMatrix) -> GMatrixReduced:
    """
    Compute the reduced G̃ matrix from the full G via two-stage SVD.

    Parameters
    ----------
    G : GMatrix  — full 6-channel LFR from lfr_matrices.build_G_matrix()

    Returns
    -------
    GMatrixReduced with 4-channel entries as plain torch.float64 tensors.
    """
    # ------------------------------------------------------------------
    # Stage 1: compact SVD of D_zw  →  D_zw = S2 @ P
    # rank(D_zw) = 4 (proved in derivation for all valid physical params)
    # ------------------------------------------------------------------
    U_hat, Sigma_hat, Vh_hat = torch.linalg.svd(G.Dzw, full_matrices=False)
    # Sigma_hat : (6,) — only first 4 are nonzero; truncate to rank 4
    rank_Dzw = 4
    U_hat  = U_hat[:, :rank_Dzw]        # (6, 4)
    Sigma_hat = Sigma_hat[:rank_Dzw]    # (4,)
    Vh_hat = Vh_hat[:rank_Dzw, :]       # (4, 6)

    S2 = U_hat * Sigma_hat.unsqueeze(0)  # (6, 4)  = Û Σ̂
    P  = Vh_hat                          # (4, 6)  = V̂ᵀ

    # ------------------------------------------------------------------
    # Stage 2: compressed stacked matrix
    # S_new = [B_w; P] @ [C_z, S2, D_zu]  ∈ R^{10×13}
    # rank(S_new) = 4 (proved in derivation)
    # ------------------------------------------------------------------
    left  = torch.cat([G.Bw, P], dim=0)                    # (10, 6)
    right = torch.cat([G.Cz, S2, G.Dzu], dim=1)            # (6, 13)
    S_new = left @ right                                     # (10, 13)

    # Compact SVD of S_new, truncated to rank 4
    U_new, Sigma_new, Vh_new = torch.linalg.svd(S_new, full_matrices=False)
    rank_Snew = 4
    U_new    = U_new[:, :rank_Snew]         # (10, 4)
    Sigma_new = Sigma_new[:rank_Snew]       # (4,)
    Vh_new   = Vh_new[:rank_Snew, :]        # (4, 13)

    Sigma_sqrt = Sigma_new.sqrt()                            # (4,)
    L_new = U_new * Sigma_sqrt.unsqueeze(0)                  # (10, 4)
    R_new = Sigma_sqrt.unsqueeze(1) * Vh_new                 # (4, 13)

    # Split L_new: top 6 rows → Lx (for B̃_w), bottom 4 rows → Lz
    Lx = L_new[:6, :]   # (6, 4)
    Lz = L_new[6:, :]   # (4, 4)

    # Split R_new by column blocks: x(6) | ŵ(4) | u(3)
    Rx = R_new[:, :6]    # (4, 6)
    Rw = R_new[:, 6:10]  # (4, 4)
    Ru = R_new[:, 10:]   # (4, 3)

    # ------------------------------------------------------------------
    # Assemble reduced G̃ entries
    # ------------------------------------------------------------------
    return GMatrixReduced(
        Ax  = G.Ax,          # (6, 6)  unchanged
        Bw  = Lx,            # (6, 4)
        Bu  = G.Bu,          # (6, 3)  unchanged
        Cz  = Rx,            # (4, 6)
        Dzw = Rw @ Lz,       # (4, 4)
        Dzu = Ru,            # (4, 3)
        Cy  = G.Cy,          # (3, 6)  unchanged
        S2  = S2,            # (6, 4)  stored for inspection
        P   = P,             # (4, 6)  stored for inspection
    )


# Module-level singleton — precomputed from the fixed G.
# If physical parameters become trainable in future, call
#   build_reduced_G_matrix(build_G_matrix(M0, M1, M2, K, C))
# inside forward() instead of using this singleton. No other change needed.
G_reduced = build_reduced_G_matrix(G)


# ----------------------------------------------------------------------
# Verification
# Run as: conda run -n GraduationProject python -m lpv_lfr_baseline.lfr_svd_reduction
# ----------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from lpv_lfr_baseline.physics import M0, M1, M2, K, C, build_M

    dtype = torch.float64
    eye3  = torch.eye(3, dtype=dtype)
    eye4  = torch.eye(4, dtype=dtype)
    z33   = torch.zeros(3, 3, dtype=dtype)

    Gr = G_reduced

    def check(name, actual, expected, tol=1e-12):
        err = (actual - expected).abs().max().item()
        status = 'PASS' if err < tol else f'FAIL (max|err|={err:.2e})'
        print(f"  {name:50s}  {status}")
        return 'PASS' in status

    results = []

    # ------------------------------------------------------------------
    # Check 1 — rank(D_zw) = 4
    # ------------------------------------------------------------------
    print("=" * 65)
    print("Check 1: rank(D_zw) = 4")
    print("=" * 65)
    rank_Dzw = torch.linalg.matrix_rank(G.Dzw).item()
    ok = rank_Dzw == 4
    print(f"  rank(D_zw) = {rank_Dzw}   {'PASS' if ok else 'FAIL'}")
    results.append(ok)

    # ------------------------------------------------------------------
    # Check 2 — Stage 1 factorization: S2 @ P ≈ D_zw
    # ------------------------------------------------------------------
    print()
    print("=" * 65)
    print("Check 2: Stage 1 factorization  S2 @ P == D_zw")
    print("=" * 65)
    results.append(check("S2 @ P == D_zw", Gr.S2 @ Gr.P, G.Dzw))

    # ------------------------------------------------------------------
    # Check 3 — Structural identity: B_w = E @ S2 @ P
    # E = [0; I3] as a (6,6) selector (lower-left identity block)
    # ------------------------------------------------------------------
    print()
    print("=" * 65)
    print("Check 3: Structural identity  B_w == E @ S2 @ P")
    print("=" * 65)
    E = torch.zeros(6, 6, dtype=dtype)
    E[3:, :3] = torch.eye(3, dtype=dtype)
    results.append(check("B_w == E @ S2 @ P", E @ Gr.S2 @ Gr.P, G.Bw))

    # ------------------------------------------------------------------
    # Check 4 — rank(S_new) = 4
    # ------------------------------------------------------------------
    print()
    print("=" * 65)
    print("Check 4: rank(S_new) = 4")
    print("=" * 65)
    left  = torch.cat([G.Bw, Gr.P], dim=0)
    right = torch.cat([G.Cz, Gr.S2, G.Dzu], dim=1)
    S_new = left @ right
    rank_Snew = torch.linalg.matrix_rank(S_new).item()
    ok = rank_Snew == 4
    print(f"  rank(S_new) = {rank_Snew}   {'PASS' if ok else 'FAIL'}")
    results.append(ok)

    # ------------------------------------------------------------------
    # Check 5 — Reduced shapes
    # ------------------------------------------------------------------
    print()
    print("=" * 65)
    print("Check 5: Reduced G̃ matrix shapes")
    print("=" * 65)
    expected_shapes = {
        'Ax':  (6, 6), 'Bw':  (6, 4), 'Bu': (6, 3),
        'Cz':  (4, 6), 'Dzw': (4, 4), 'Dzu': (4, 3),
        'Cy':  (3, 6),
    }
    for name, exp_shape in expected_shapes.items():
        actual = tuple(getattr(Gr, name).shape)
        ok = actual == exp_shape
        print(f"  {name:6s}  {str(actual):12s}  {'PASS' if ok else f'FAIL (expected {exp_shape})'}")
        results.append(ok)

    # ------------------------------------------------------------------
    # Check 6 — Dynamics equivalence: reduced LFR reproduces A_c(Y)@x + B_c(Y)@u
    #
    # Resolve the reduced algebraic loop:
    #   z̃ = C̃z@x + D̃zw@(Y z̃) + D̃zu@u
    #   (I4 - Y D̃zw) z̃ = C̃z@x + D̃zu@u
    # Then:
    #   w̃ = Y z̃
    #   ẋ = Ax@x + B̃w@w̃ + Bu@u
    # Compare against A_c(Y)@x + B_c(Y)@u for several Y values.
    # ------------------------------------------------------------------
    print()
    print("=" * 65)
    print("Check 6: Dynamics equivalence vs A_c(Y)@x + B_c(Y)@u  (5 Y values)")
    print("=" * 65)

    torch.manual_seed(0)
    x_test    = torch.tensor([0.05, 0.01, 0.30, 0.02, -0.01, 0.05], dtype=dtype)
    u_test    = torch.tensor([1.0, -0.5, 0.2], dtype=dtype)
    test_Y    = [0.0, 0.1, 0.3, -0.2, 0.35]
    all_pass  = True

    for y_val in test_Y:
        Y = torch.tensor(y_val, dtype=dtype)

        # Reference: collapsed A_c(Y)@x + B_c(Y)@u
        M_Y    = build_M(Y)
        MYinvK = torch.linalg.solve(M_Y, K)
        MYinvC = torch.linalg.solve(M_Y, C)
        MYinv  = torch.linalg.solve(M_Y, eye3)
        A_c    = torch.cat([torch.cat([z33, eye3], dim=1),
                            torch.cat([-MYinvK, -MYinvC], dim=1)], dim=0)
        B_c    = torch.cat([z33, MYinv], dim=0)
        xdot_ref = A_c @ x_test + B_c @ u_test

        # Reduced LFR loop resolution
        lhs = eye4 - Y * Gr.Dzw                              # (4,4)
        rhs = Gr.Cz @ x_test + Gr.Dzu @ u_test              # (4,)
        z_tilde = torch.linalg.solve(lhs, rhs)               # (4,)
        w_tilde = Y * z_tilde                                 # (4,)
        xdot_red = Gr.Ax @ x_test + Gr.Bw @ w_tilde + Gr.Bu @ u_test

        err    = (xdot_red - xdot_ref).abs().max().item()
        status = 'PASS' if err < 1e-10 else f'FAIL (err={err:.2e})'
        if 'FAIL' in status:
            all_pass = False
        print(f"  Y = {y_val:+.2f} m   max|xdot error| = {err:.2e}   {status}")

    results.append(all_pass)
    print(f"\nCheck 6: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 65)
    print(f"Overall: {'ALL PASS' if all(results) else 'SOME FAILED'}")
    print(f"Latent channels: 6 -> 4  (Delta(Y) = Y*I6  ->  Delta(Y) = Y*I4)")
    print("=" * 65)
