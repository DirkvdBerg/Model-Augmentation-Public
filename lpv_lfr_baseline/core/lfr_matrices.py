"""
lfr_matrices.py
---------------
Precomputes the constant G matrix entries for the dual-gantry LPV-LFR realization.

Derived from: LPV/LFR-derivation-supervisor.tex, Section "Constant Interconnection Matrices".

All entries are built from M0_inv = inverse(M0), where M0 is the Y-independent
part of the mass matrix decomposition M(Y) = M0 + M1*Y + M2*Y^2.
M0 is constant, so M0_inv is precomputed once.

G matrix structure (rows: xdot, z, y  |  cols: x, w, u):

    Ax  = [0,       I3      ]   (6x6)
          [-M0invK, -M0invC ]

    Bw  = [0,        0       ]  (6x6)
          [-M0invM1, -M0invM2]

    Bu  = [0    ]               (6x3)
          [M0inv]

    Cz  = [-M0invK, -M0invC]   (6x6)
          [0,       0      ]

    Dzw = [-M0invM1, -M0invM2]  (6x6)
          [I3,       0       ]

    Dzu = [M0inv]               (6x3)
          [0    ]

    Cy  = [I3, 0]               (3x6)

Where:
    M0invK  = M0_inv @ K
    M0invC  = M0_inv @ C
    M0invM1 = M0_inv @ M1
    M0invM2 = M0_inv @ M2

Provides:
    GMatrix dataclass holding all entries as plain torch tensors (not nn.Parameter).
    build_G_matrix(M0, M1, M2, K, C) -> GMatrix

Note on build_G_matrix signature:
    M0, M1, M2, K, C are passed explicitly (not read from physics.py inside this
    function). This allows build_G_matrix to be called inside a forward() pass if
    physical parameters ever become trainable — no signature change needed.
"""

from dataclasses import dataclass

import torch

from lpv_lfr_baseline.core.physics import M0, M1, M2, K, C


@dataclass
class GMatrix:
    """
    Constant G matrix entries for the LPV-LFR realization.

    All fields are plain torch.float64 tensors — precomputed constants,
    not trainable parameters.

    Shapes:
        Ax  : (6, 6)
        Bw  : (6, 6)
        Bu  : (6, 3)
        Cz  : (6, 6)
        Dzw : (6, 6)
        Dzu : (6, 3)
        Cy  : (3, 6)
    """
    Ax:  torch.Tensor   # (6, 6)
    Bw:  torch.Tensor   # (6, 6)
    Bu:  torch.Tensor   # (6, 3)
    Cz:  torch.Tensor   # (6, 6)
    Dzw: torch.Tensor   # (6, 6)
    Dzu: torch.Tensor   # (6, 3)
    Cy:  torch.Tensor   # (3, 6)


def build_G_matrix(
    M0: torch.Tensor,
    M1: torch.Tensor,
    M2: torch.Tensor,
    K:  torch.Tensor,
    C:  torch.Tensor,
) -> GMatrix:
    """
    Build the constant G matrix from M0, M1, M2, K, C.

    Parameters
    ----------
    M0, M1, M2 : (3, 3) torch.float64 — mass matrix decomposition
    K, C       : (3, 3) torch.float64 — stiffness and damping matrices

    Returns
    -------
    GMatrix with all entries as (plain) torch.float64 tensors.
    """
    dtype = torch.float64
    eye3  = torch.eye(3, dtype=dtype)
    z33   = torch.zeros(3, 3, dtype=dtype)

    # M0^{-1} and products — use solve, not inv
    M0inv   = torch.linalg.solve(M0, eye3)       # (3,3)  M0^{-1}
    M0invK  = torch.linalg.solve(M0, K)          # (3,3)  M0^{-1} K
    M0invC  = torch.linalg.solve(M0, C)          # (3,3)  M0^{-1} C
    M0invM1 = torch.linalg.solve(M0, M1)         # (3,3)  M0^{-1} M1
    M0invM2 = torch.linalg.solve(M0, M2)         # (3,3)  M0^{-1} M2

    # ------------------------------------------------------------------
    # Ax = [  0,       I3    ]  (6x6)
    #      [ -M0invK, -M0invC]
    # ------------------------------------------------------------------
    Ax = torch.cat([
        torch.cat([z33,      eye3    ], dim=1),
        torch.cat([-M0invK, -M0invC  ], dim=1),
    ], dim=0)

    # ------------------------------------------------------------------
    # Bw = [  0,        0      ]  (6x6)
    #      [ -M0invM1, -M0invM2]
    # ------------------------------------------------------------------
    Bw = torch.cat([
        torch.cat([z33,       z33      ], dim=1),
        torch.cat([-M0invM1, -M0invM2  ], dim=1),
    ], dim=0)

    # ------------------------------------------------------------------
    # Bu = [  0    ]  (6x3)
    #      [ M0inv ]
    # ------------------------------------------------------------------
    Bu = torch.cat([z33, M0inv], dim=0)

    # ------------------------------------------------------------------
    # Cz = [ -M0invK, -M0invC]  (6x6)
    #      [  0,       0     ]
    # ------------------------------------------------------------------
    Cz = torch.cat([
        torch.cat([-M0invK, -M0invC], dim=1),
        torch.cat([z33,      z33    ], dim=1),
    ], dim=0)

    # ------------------------------------------------------------------
    # Dzw = [ -M0invM1, -M0invM2]  (6x6)
    #       [  I3,       0      ]
    # ------------------------------------------------------------------
    Dzw = torch.cat([
        torch.cat([-M0invM1, -M0invM2], dim=1),
        torch.cat([eye3,      z33     ], dim=1),
    ], dim=0)

    # ------------------------------------------------------------------
    # Dzu = [ M0inv]  (6x3)
    #       [  0   ]
    # ------------------------------------------------------------------
    Dzu = torch.cat([M0inv, z33], dim=0)

    # ------------------------------------------------------------------
    # Cy = [I3, 0]  (3x6)
    # ------------------------------------------------------------------
    Cy = torch.cat([eye3, z33], dim=1)

    return GMatrix(Ax=Ax, Bw=Bw, Bu=Bu, Cz=Cz, Dzw=Dzw, Dzu=Dzu, Cy=Cy)


# Module-level singleton — precomputed from the fixed physical parameters.
# If physical parameters become trainable in future, call build_G_matrix()
# inside forward() instead of using this singleton.
G = build_G_matrix(M0, M1, M2, K, C)


# ----------------------------------------------------------------------
# Verification  (run as: conda run -n GraduationProject python lpv_lfr_baseline/lfr_matrices.py)
# ----------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from lpv_lfr_baseline.core.physics import M0, M1, M2, K, C

    dtype = torch.float64
    eye3  = torch.eye(3, dtype=dtype)
    z33   = torch.zeros(3, 3, dtype=dtype)

    # Recompute products independently for comparison
    M0inv   = torch.linalg.solve(M0, eye3)
    M0invK  = torch.linalg.solve(M0, K)
    M0invC  = torch.linalg.solve(M0, C)
    M0invM1 = torch.linalg.solve(M0, M1)
    M0invM2 = torch.linalg.solve(M0, M2)

    def check(name, actual, expected):
        err = (actual - expected).abs().max().item()
        status = 'PASS' if err == 0.0 else 'FAIL'
        print(f"  {name:30s}  max|error| = {err:.2e}   {status}")
        return status == 'PASS'

    print("=" * 60)
    print("G matrix algebraic assembly check")
    print("=" * 60)

    results = []

    # Ax blocks
    results.append(check("Ax[0:3, 0:3] == 0",        G.Ax[:3, :3],  z33         ))
    results.append(check("Ax[0:3, 3:6] == I3",        G.Ax[:3, 3:],  eye3        ))
    results.append(check("Ax[3:6, 0:3] == -M0invK",   G.Ax[3:, :3], -M0invK     ))
    results.append(check("Ax[3:6, 3:6] == -M0invC",   G.Ax[3:, 3:], -M0invC     ))

    # Bw blocks
    results.append(check("Bw[0:3, 0:3] == 0",         G.Bw[:3, :3],  z33         ))
    results.append(check("Bw[0:3, 3:6] == 0",         G.Bw[:3, 3:],  z33         ))
    results.append(check("Bw[3:6, 0:3] == -M0invM1",  G.Bw[3:, :3], -M0invM1    ))
    results.append(check("Bw[3:6, 3:6] == -M0invM2",  G.Bw[3:, 3:], -M0invM2    ))

    # Bu blocks
    results.append(check("Bu[0:3, :]   == 0",          G.Bu[:3, :],   z33         ))
    results.append(check("Bu[3:6, :]   == M0inv",      G.Bu[3:, :],   M0inv       ))

    # Cz blocks
    results.append(check("Cz[0:3, 0:3] == -M0invK",   G.Cz[:3, :3], -M0invK     ))
    results.append(check("Cz[0:3, 3:6] == -M0invC",   G.Cz[:3, 3:], -M0invC     ))
    results.append(check("Cz[3:6, 0:3] == 0",         G.Cz[3:, :3],  z33         ))
    results.append(check("Cz[3:6, 3:6] == 0",         G.Cz[3:, 3:],  z33         ))

    # Dzw blocks
    results.append(check("Dzw[0:3, 0:3] == -M0invM1", G.Dzw[:3, :3], -M0invM1   ))
    results.append(check("Dzw[0:3, 3:6] == -M0invM2", G.Dzw[:3, 3:], -M0invM2   ))
    results.append(check("Dzw[3:6, 0:3] == I3",       G.Dzw[3:, :3],  eye3       ))
    results.append(check("Dzw[3:6, 3:6] == 0",        G.Dzw[3:, 3:],  z33        ))

    # Dzu blocks
    results.append(check("Dzu[0:3, :]   == M0inv",    G.Dzu[:3, :],   M0inv      ))
    results.append(check("Dzu[3:6, :]   == 0",        G.Dzu[3:, :],   z33        ))

    # Cy blocks
    results.append(check("Cy[0:3, 0:3]  == I3",       G.Cy[:, :3],    eye3       ))
    results.append(check("Cy[0:3, 3:6]  == 0",        G.Cy[:, 3:],    z33        ))

    print()
    print(f"Overall: {'ALL PASS' if all(results) else 'SOME FAILED'}")

    # Shape check
    print()
    print("Shape check:")
    shapes = {
        'Ax': (6,6), 'Bw': (6,6), 'Bu': (6,3),
        'Cz': (6,6), 'Dzw': (6,6), 'Dzu': (6,3), 'Cy': (3,6),
    }
    for name, expected_shape in shapes.items():
        actual = tuple(getattr(G, name).shape)
        status = 'PASS' if actual == expected_shape else 'FAIL'
        print(f"  {name:6s}  {str(actual):12s}  {status}")
