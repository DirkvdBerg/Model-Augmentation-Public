"""
physics.py
----------
Physical constants for the dual-gantry FP model as torch tensors.

Source: kamtin-fp-model/03 Simulink gantry/main.m (immutable ground truth).
All values must match main.m exactly. Do not modify without checking main.m first.

Provides:
    - Scalar physical parameters (masses, inertias, damping, stiffness, geometry)
    - Mass matrix decomposition: M(Y) = M0 + M1*Y + M2*Y^2
    - Constant damping matrix C
    - Constant stiffness matrix K
    - Coordinate transform P  (logical -> stage)
    - Sampling constants: fs, ts

All tensors are dtype=torch.float64.
"""
