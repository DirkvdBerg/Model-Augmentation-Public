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
    GMatrix dataclass (or namedtuple) holding all entries as torch tensors.
    build_G_matrix() -> GMatrix
"""
