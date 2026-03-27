"""
validate_lfr.py
---------------
Validation script for the dual-gantry LPV-LFR baseline.

Runs three checks to verify the implementation against analytical expectations
and the existing ZOH reference (gantry_lpv_torch.py).

Check 1 — G matrix algebraic consistency:
    Verify that the G matrix entries satisfy the structural constraints from
    LFR-derivation-supervisor.tex. Concretely: that Bw, Cz, Dzw, Dzu are all
    derived consistently from M0_inv @ {K, C, M1, M2}.

    Expected outcome: max absolute entry-wise error = 0.0 (exact arithmetic).

Check 2 — Algebraic loop resolution:
    For a fixed Y and a given state x and input u, verify that:
        M(Y) @ v = fnet
    where fnet = [-K, -C] @ x + u and v is the resolved acceleration from
    lfr_forward. This is the closed-form check that the loop resolution is correct.

    Expected outcome: residual norm < 1e-12 (float64 machine precision).

Check 3 — Trajectory comparison vs ZOH reference:
    Simulate a short trajectory (N steps, zero or step input) with both:
        - This module: lfr_simulate (RK4, CT)
        - Reference:   gantry_lpv_torch.py (ZOH, discrete-time)

    RK4 and ZOH are different discretization methods — differences are expected.
    The check is that differences are bounded (not diverging) and small relative to
    the signal magnitude at fs=16kHz (ts=62.5us). Large divergence would indicate
    a bug, not a method difference.

    Expected outcome: max position error < 1e-6 m over a 0.1s window (heuristic bound).

Usage:
    python -m lpv_lfr_baseline.validate_lfr
    or
    python lpv_lfr_baseline/validate_lfr.py

All checks print PASS/FAIL with residuals. No assertions that abort silently —
every check reports its metric so the caller can judge.
"""
