"""
lfr_param_block.py
------------------
Jan-compatible Block wrapper for the dual-gantry LPV-LFR baseline
with trainable physical parameters.

Mirrors LFRBaselineBlock (lfr_block.py) but makes 10 physical scalars
trainable via log/exp reparameterization. All other design choices are
identical: stateless, (batch, 9, 1) -> (batch, 18, 1), float64 physics.

Trainable scalars (13, stored as log_params):
    kb1                     [N.m/rad]   stiffness joint 1
    kb2                     [N.m/rad]   stiffness joint 2
    cg1                     [N/(m/s)]   X1 viscous friction
    cg2                     [N/(m/s)]   X2 viscous friction
    cy                      [N/(m/s)]   Y  viscous friction (isolated in C[2,2])
    cb1                     [N.m/(rad/s)] rotational friction joint 1
    cb2                     [N.m/(rad/s)] rotational friction joint 2
    mh                      [kg]        payload mass -- the sole LPV parameter
    m1                      [kg]        actuator X1 mass
    m2                      [kg]        actuator X2 mass
    mb                      [kg]        cross-arm mass
    Jb                      [kg.m^2]    rotary inertia of cross-arm
    Jh                      [kg.m^2]    rotary inertia of payload
    d                       [m]         cross-arm to payload distance

    Note: dynamics only observes kb1+kb2, cb1+cb2, Jb+Jh (sums). The individual
    components are regularized toward physical priors to resolve the flat ridge
    in the loss landscape (see param_loss).

Fixed buffers (not trainable):
    Lb      [m]     cross-arm length  -- enters P transform, cannot train

Positivity guarantee (D-035):
    Parameters stored as self.log_params = nn.Parameter(zeros), representing
    log(theta / params_init). Physical values recovered as
    params_init * exp(log_params), clamped to (1e-6, inf).
    Initialized at zero so all parameters start at their reference value (= 1
    in normalized space). This centers the log-space landscape around 0 and
    normalizes each parameter to be ~1 at init, giving Adam uniform gradient
    scaling across parameters that otherwise span 1-4000 in physical units.

Regularization (D-034):
    param_loss() computes Lambda-weighted L2 toward params_init (physical space).
    Lambda[i] = RMSE_baseline / params_init[i]  (tighter for small params_init).
    RMSE_baseline is passed at construction from a pre-training forward-pass
    measurement on the MATLAB data (D-034).

Detuned initial values:
    Fixed ±10% applied per parameter (see _DETUNING); sum pairs share the same sign
    so identifiable sums are detuned. Signs are fixed for reproducibility without a seed.

Block interface (identical to LFRBaselineBlock):
    nz = 9   (nx=6 state + nu=3 stage-coord input)
    nw = 18  (x_next=6 + z_lfr=6 + w_lfr=6)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lpv_lfr_baseline.core.physics import P, ts, build_poly_constants
from lpv_lfr_baseline.core.lfr_matrices import build_G_matrix
from lpv_lfr_baseline.core.lfr_simulate import rk4_step

try:
    from model_augmentation.fit_systems.blocks import Block as _JanBlock
    _HAS_JAN_BLOCK = True
    _BASE = _JanBlock
except ImportError:
    _HAS_JAN_BLOCK = False
    _BASE = nn.Module

# ----------------------------------------------------------------------
# True physical values (from physics.py / main.m)
# Used only for verification in __main__. Training uses detuned inits.
# ----------------------------------------------------------------------
_TRUE_PARAMS = {
    'kb1':    1987.50,
    'kb2':    1987.50,
    'cg1':      14.50,
    'cg2':      20.30,
    'cy':       10.00,
    'cb1':       9.00,
    'cb2':       9.00,
    'mh':       10.10,
    'm1':       10.20,
    'm2':       10.70,
    'mb':       22.80,
    'Jb':        1.00,
    'Jh':        0.05,
    'd':         0.10,
}

_PARAM_NAMES = ['kb1', 'kb2', 'cg1', 'cg2', 'cy', 'cb1', 'cb2', 'mh', 'm1', 'm2', 'mb', 'Jb', 'Jh', 'd']

# Detuned initial values — fixed ±10% per parameter, reproducible without a seed.
# Sum pairs (kb1+kb2, cb1+cb2, Jb+Jh) share the same sign so the sum is detuned;
# individual params within a pair are not data-identifiable anyway.
_DETUNING = {
    'kb1': +0.10, 'kb2': +0.10,   # kb_sum starts +10% from true
    'cg1': +0.10,
    'cg2': -0.10,
    'cy':  +0.10,
    'cb1': -0.10, 'cb2': -0.10,   # cb_sum starts -10% from true
    'mh':  -0.10,
    'm1':  +0.10,
    'm2':  -0.10,
    'mb':  +0.10,
    'Jb':  -0.10, 'Jh':  -0.10,   # J_sum  starts -10% from true
    'd':   +0.10,
}
_DETUNED_PARAMS = {n: _TRUE_PARAMS[n] * (1 + _DETUNING[n]) for n in _PARAM_NAMES}

# Fixed geometry — Lb enters P (coordinate transform) and cannot be trained.
_Lb = torch.tensor(0.725, dtype=torch.float64)


def _build_matrices(
    params: Tensor,   # (10,) physical scalars -- already exp'd and clamped
    Lb:     Tensor,   # () scalar
    d:      Tensor,   # () scalar
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Differentiably reconstruct M0, M1, M2, K, C from physical scalars.

    Parameters
    ----------
    params : (10,) float64 -- [kb_sum, cg1, cg2, cy, cb_sum, mh, m1, m2, mb, J_sum]
    Lb, d  : scalar float64 -- fixed geometry

    Returns
    -------
    M0, M1, M2 : (3, 3)  mass matrix decomposition  M(Y) = M0 + M1*Y + M2*Y^2
    K          : (3, 3)  stiffness matrix
    C          : (3, 3)  damping matrix
    """
    kb_sum, cg1, cg2, cy, cb_sum, mh, m1, m2, mb, J_sum = params

    # Scalar zero with correct device/dtype -- used for structurally-zero entries
    z = params.new_zeros(())

    # ------------------------------------------------------------------
    # M0 -- constant part of M(Y)
    #   [m1+m2+mb+mh,        (m1-m2)*Lb/2,                    0    ]
    #   [(m1-m2)*Lb/2,   J_sum+(m1+m2)*Lb^2/4+mh*d^2,          -mh*d]
    #   [0,                  -mh*d,                            mh   ]
    # ------------------------------------------------------------------
    M0 = torch.stack([
        torch.stack([m1+m2+mb+mh,   (m1-m2)*Lb/2,                        z     ]),
        torch.stack([(m1-m2)*Lb/2,  J_sum+(m1+m2)*Lb**2/4+mh*d**2,      -mh*d  ]),
        torch.stack([z,             -mh*d,                                 mh   ]),
    ])

    # ------------------------------------------------------------------
    # M1 -- coefficient of Y
    #   [ 0,   -mh,  0]
    #   [-mh,   0,   0]
    #   [ 0,    0,   0]
    # ------------------------------------------------------------------
    M1 = torch.stack([
        torch.stack([z,   -mh,  z]),
        torch.stack([-mh,  z,   z]),
        torch.stack([z,    z,   z]),
    ])

    # ------------------------------------------------------------------
    # M2 -- coefficient of Y^2
    #   [0,   0,   0]
    #   [0,  mh,   0]
    #   [0,   0,   0]
    # ------------------------------------------------------------------
    M2 = torch.stack([
        torch.stack([z,   z,   z]),
        torch.stack([z,   mh,  z]),
        torch.stack([z,   z,   z]),
    ])

    # ------------------------------------------------------------------
    # K -- stiffness (only K[1,1] non-zero)
    #   [0,      0,      0]
    #   [0,  kb_sum,     0]
    #   [0,      0,      0]
    # ------------------------------------------------------------------
    K = torch.stack([
        torch.stack([z,      z,       z]),
        torch.stack([z,      kb_sum,  z]),
        torch.stack([z,      z,       z]),
    ])

    # ------------------------------------------------------------------
    # C -- damping
    #   [cg1+cg2,          (cg1-cg2)*Lb/2,                   0 ]
    #   [(cg1-cg2)*Lb/2,   cb_sum+(cg1+cg2)*Lb^2/4,           0 ]
    #   [0,                0,                                 cy]
    # ------------------------------------------------------------------
    C = torch.stack([
        torch.stack([cg1+cg2,          (cg1-cg2)*Lb/2,                    z ]),
        torch.stack([(cg1-cg2)*Lb/2,   cb_sum+(cg1+cg2)*Lb**2/4,          z ]),
        torch.stack([z,                 z,                                  cy]),
    ])

    return M0, M1, M2, K, C


class ParameterizedLFRBlock(_BASE):
    """
    LPV-LFR baseline block with trainable physical parameters.

    Identical interface to LFRBaselineBlock (nz=9, nw=18) but physical
    scalars are nn.Parameter (log-reparameterized) instead of fixed buffers.

    Usage
    -----
    block = ParameterizedLFRBlock(RMSE_baseline=<measured>)
    # RMSE_baseline: RMS prediction error of the detuned baseline on training
    # data, computed before training (D-034). Default 1.0 is a safe placeholder
    # that will over-regularize; always replace with the measured value.

    Diagnostics
    -----------
    block.physical_params()     -> dict  {name: value}  current physical values
    block.param_table()         -> str   formatted comparison table
    """

    def __init__(self, RMSE_baseline: float = 1.0, **kwargs):
        if _HAS_JAN_BLOCK:
            super().__init__(nz=9, nw=18, **kwargs)
        else:
            super().__init__(**kwargs)
            self.nz = 9
            self.nw = 18

        dtype = torch.float64

        # ------------------------------------------------------------------
        # Detuned initial values -- what we start training from
        # ------------------------------------------------------------------
        params_init = torch.tensor(
            [_DETUNED_PARAMS[n] for n in _PARAM_NAMES], dtype=dtype
        )   # (10,)

        # ------------------------------------------------------------------
        # Trainable parameters -- stored in log space (D-035)
        # torch.exp(log_params) gives physical values, always > 0
        # ------------------------------------------------------------------
        self.log_params = nn.Parameter(torch.zeros_like(params_init))

        # ------------------------------------------------------------------
        # Frozen reference for regularization -- physical space (D-034, D-035)
        # ------------------------------------------------------------------
        self.register_buffer('params_init', params_init)

        # Lambda[i] = RMSE_baseline / params_init[i]  (D-034)
        Lambda = torch.tensor(RMSE_baseline, dtype=dtype) / params_init   # (10,)
        self.register_buffer('Lambda', Lambda)

        # ------------------------------------------------------------------
        # Fixed geometry -- Lb only (enters P transform, cannot train)
        # ------------------------------------------------------------------
        self.register_buffer('_Lb', _Lb.clone())

        # ------------------------------------------------------------------
        # Coordinate transform and sample period -- fixed buffers
        # ------------------------------------------------------------------
        self.register_buffer('_P',  P.clone())
        self.register_buffer('_ts', ts.clone())

    # ------------------------------------------------------------------
    # Physical parameter access helpers
    # ------------------------------------------------------------------

    def _recover_params(self) -> Tensor:
        """Return physical parameters: params_init * exp(log_params), clamped > 0."""
        return (self.params_init * torch.exp(self.log_params)).clamp(min=1e-6)

    def physical_params(self) -> dict:
        """Return current physical parameter values as a dict."""
        vals = self._recover_params().detach()
        return {name: vals[i].item() for i, name in enumerate(_PARAM_NAMES)}

    # Identifiable sums and the non-split individually identifiable params, in physics order.
    _SUM_PAIRS   = [('kb_sum', 'kb1', 'kb2'), ('cb_sum', 'cb1', 'cb2'), ('J_sum', 'Jb', 'Jh')]
    # Table 1: identifiable combinations (Jacobian rank analysis — 10 recoverable scalars).
    # m_total, m_diff, J_eff replace individual m1/m2/mb/J_sum which are not separately identifiable.
    _IDENT_ORDER = ['kb_sum', 'cg1', 'cg2', 'cy', 'cb_sum', 'mh', 'm_total', 'm_diff', 'J_eff', 'd']
    # Table 2: not data-identifiable — individual splits and mass-inertia components.
    _SPLIT_NAMES = ['kb1', 'kb2', 'cb1', 'cb2', 'Jb', 'Jh', 'm1', 'm2', 'mb', 'J_sum']

    def param_table(self) -> str:
        """Two-table comparison: identifiable quantities, then split diagnostics."""
        hdr = f"{'Parameter':<10} {'True':>10} {'Detuned':>10} {'Learned':>10} {'delta':>10}"
        sep = "-" * 55
        cur = self.physical_params()
        Lb  = self._Lb.item()

        # Pairwise sums (kb_sum, cb_sum, J_sum)
        sums = {
            sn: (
                _TRUE_PARAMS[a]    + _TRUE_PARAMS[b],
                _DETUNED_PARAMS[a] + _DETUNED_PARAMS[b],
                cur[a]             + cur[b],
            )
            for sn, a, b in self._SUM_PAIRS
        }

        # Derived identifiable combinations for the mass-inertia group (n4 analysis).
        # These are the quantities the data can actually determine; m1/m2/mb/J_sum are not
        # individually identifiable (flat direction: Δm1=Δm2=ε, Δmb=-2ε, ΔJ_sum=-Lb²/2·ε).
        sums['m_total'] = (
            _TRUE_PARAMS['m1']    + _TRUE_PARAMS['m2']    + _TRUE_PARAMS['mb'],
            _DETUNED_PARAMS['m1'] + _DETUNED_PARAMS['m2'] + _DETUNED_PARAMS['mb'],
            cur['m1']             + cur['m2']             + cur['mb'],
        )
        sums['m_diff'] = (
            _TRUE_PARAMS['m1']    - _TRUE_PARAMS['m2'],
            _DETUNED_PARAMS['m1'] - _DETUNED_PARAMS['m2'],
            cur['m1']             - cur['m2'],
        )
        # J_eff = J_sum + (m1+m2)*Lb²/4  [= M0[1,1] - mh*d²]
        sums['J_eff'] = (
            sums['J_sum'][0] + (_TRUE_PARAMS['m1']    + _TRUE_PARAMS['m2'])    * Lb**2 / 4,
            sums['J_sum'][1] + (_DETUNED_PARAMS['m1'] + _DETUNED_PARAMS['m2']) * Lb**2 / 4,
            sums['J_sum'][2] + (cur['m1']             + cur['m2'])             * Lb**2 / 4,
        )

        lines = ['  Table 1 — identifiable quantities', hdr, sep]
        for name in self._IDENT_ORDER:
            if name in sums:
                true_v, det_v, lrn_v = sums[name]
            else:
                true_v, det_v, lrn_v = _TRUE_PARAMS[name], _DETUNED_PARAMS[name], cur[name]
            delta = (lrn_v - true_v) / true_v * 100 if abs(true_v) > 1e-12 else float('nan')
            lines.append(f"{name:<10} {true_v:>10.4f} {det_v:>10.4f} {lrn_v:>10.4f} {delta:>+9.2f}%")

        lines += ['', '  Table 2 — split diagnostics (not data-identifiable)', hdr, sep]
        for name in self._SPLIT_NAMES:
            if name == 'J_sum':
                true_v = sums['J_sum'][0]
                det_v  = sums['J_sum'][1]
                lrn_v  = sums['J_sum'][2]
            else:
                true_v, det_v, lrn_v = _TRUE_PARAMS[name], _DETUNED_PARAMS[name], cur[name]
            delta = (lrn_v - true_v) / true_v * 100 if abs(true_v) > 1e-12 else float('nan')
            lines.append(f"{name:<10} {true_v:>10.4f} {det_v:>10.4f} {lrn_v:>10.4f} {delta:>+9.2f}%")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Regularization loss (called by LFRFitSystem.loss())
    # ------------------------------------------------------------------

    def param_loss(self) -> Tensor:
        """
        Lambda-weighted L2 regularization toward params_init.

        Both sides are in physical space (not log space) for interpretability.
        Returns a scalar tensor.
        """
        params = self._recover_params()
        return F.mse_loss(
            self.Lambda * params,
            self.Lambda * self.params_init,
            reduction='sum',
        )

    def matrix_table(self) -> str:
        """Comparison of physics matrix entries: true vs learned (non-zero, upper triangle)."""
        dtype = torch.float64

        p_true = torch.tensor([_TRUE_PARAMS[n] for n in _PARAM_NAMES], dtype=dtype)
        kb1, kb2, cg1, cg2, cy, cb1, cb2, mh, m1, m2, mb, Jb, Jh, d = p_true
        M0_t, M1_t, M2_t, K_t, C_t = _build_matrices(
            torch.stack([kb1+kb2, cg1, cg2, cy, cb1+cb2, mh, m1, m2, mb, Jb+Jh]),
            self._Lb.to(dtype), d,
        )

        p = self._recover_params().detach().to(dtype)
        kb1, kb2, cg1, cg2, cy, cb1, cb2, mh, m1, m2, mb, Jb, Jh, d = p
        M0_l, M1_l, M2_l, K_l, C_l = _build_matrices(
            torch.stack([kb1+kb2, cg1, cg2, cy, cb1+cb2, mh, m1, m2, mb, Jb+Jh]),
            self._Lb.to(dtype), d,
        )

        true_mats    = {'M0': M0_t, 'M1': M1_t, 'M2': M2_t, 'K': K_t, 'C': C_t}
        learned_mats = {'M0': M0_l, 'M1': M1_l, 'M2': M2_l, 'K': K_l, 'C': C_l}

        _ENTRIES = [
            ('M0', 0, 0, 'm1+m2+mb+mh'),
            ('M0', 0, 1, '(m1-m2)*Lb/2'),
            ('M0', 1, 1, 'J+(m1+m2)*Lb^2/4+mh*d^2'),
            ('M0', 1, 2, '-mh*d'),
            ('M0', 2, 2, 'mh'),
            ('M1', 0, 1, '-mh'),
            ('M2', 1, 1, 'mh'),
            ('K',  1, 1, 'kb_sum'),
            ('C',  0, 0, 'cg1+cg2'),
            ('C',  0, 1, '(cg1-cg2)*Lb/2'),
            ('C',  1, 1, 'cb_sum+(cg1+cg2)*Lb^2/4'),
            ('C',  2, 2, 'cy'),
        ]

        hdr = f"  {'Matrix':<6}  {'[i,j]':<6}  {'True':>12}  {'Learned':>12}  {'delta':>9}  Expression"
        sep = "  " + "-" * 72
        lines = ['  Table 3 — Physics matrix entries (non-zero, upper triangle)', hdr, sep]
        for mat, i, j, expr in _ENTRIES:
            tv = true_mats[mat][i, j].item()
            lv = learned_mats[mat][i, j].item()
            delta = (lv - tv) / tv * 100 if abs(tv) > 1e-12 else float('nan')
            lines.append(
                f"  {mat:<6}  [{i},{j}]  {tv:>12.4f}  {lv:>12.4f}  {delta:>+9.2f}%  {expr}"
            )
        return '\n'.join(lines)

    def split_loss(self) -> Tensor:
        """
        Scale-invariant penalty on degenerate parameter splits (D-037).

        kb1/kb2, cb1/cb2: normalised squared difference -- prefers equal split.
          (true values are symmetric by design: kb1=kb2, cb1=cb2)
        Jb/Jh: log-space squared difference -- prefers proportional fractional
          change rather than equal split (true values differ: Jb=1.0, Jh=0.05).

        Returns a scalar tensor.
        """
        p = self._recover_params()
        kb1, kb2 = p[0], p[1]
        cb1, cb2 = p[5], p[6]
        return (
            ((kb1 - kb2) / (kb1 + kb2)).pow(2)
            + ((cb1 - cb2) / (cb1 + cb2)).pow(2)
            + (self.log_params[11] - self.log_params[12]).pow(2)
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, z_in: Tensor) -> Tensor:
        """One RK4 step. (batch, 9, 1) -> (batch, 18, 1)."""
        in_dtype = z_in.dtype
        z_flat   = z_in.squeeze(-1)
        work_dtype = self.log_params.dtype

        if z_flat.dtype != work_dtype:
            z_flat = z_flat.to(dtype=work_dtype)

        x       = z_flat[:, :6]
        u_stage = z_flat[:, 6:]

        u_logical = u_stage @ self._P.T

        # Rebuild G and polynomial constants from current trainable params each forward call.
        # Cannot cache as module-level constants — physical params are nn.Parameter objects,
        # so G and poly constants must be recomputed here to preserve gradient flow.
        params = self._recover_params()
        kb1, kb2, cg1, cg2, cy, cb1, cb2, mh, m1, m2, mb, Jb, Jh, d = params
        _, M1, M2, K, C = _build_matrices(
            torch.stack([kb1+kb2, cg1, cg2, cy, cb1+cb2, mh, m1, m2, mb, Jb+Jh]),
            self._Lb, d,
        )
        alpha, beta, gamma, N0, N1, N2 = build_poly_constants(
            m1, m2, mb, mh, Jb, Jh, self._Lb, d
        )
        d0 = mh * (alpha * gamma - beta ** 2)
        G = build_G_matrix(N0, d0, M1, M2, K, C)

        x_next, z_lfr, w_lfr, _ = rk4_step(
            x, u_logical,
            G, K, C, mh, alpha, beta, gamma, N0, N1, N2,
            self._ts,
        )

        w_out = torch.cat([x_next, z_lfr, w_lfr], dim=-1)   # (batch, 18)
        out   = w_out if in_dtype == work_dtype else w_out.to(in_dtype)
        return out.unsqueeze(-1)


# ----------------------------------------------------------------------
# Verification  (run as: conda run -n GraduationProject python -m lpv_lfr_baseline.lfr_param_block)
# ----------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from lpv_lfr_baseline.core.physics import M0 as M0_ref, M1 as M1_ref, M2 as M2_ref
    from lpv_lfr_baseline.core.physics import K as K_ref, C as C_ref

    dtype = torch.float64

    # ------------------------------------------------------------------
    # Check 1 -- _build_matrices at true params matches physics.py exactly
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Check 1: _build_matrices(true params) matches physics.py")
    print("=" * 60)

    p = torch.tensor([_TRUE_PARAMS[n] for n in _PARAM_NAMES], dtype=dtype)
    kb1, kb2, cg1_, cg2_, cy_, cb1, cb2, mh_, m1_, m2_, mb_, Jb_, Jh_, d_ = p
    true_params_10 = torch.stack([kb1+kb2, cg1_, cg2_, cy_, cb1+cb2, mh_, m1_, m2_, mb_, Jb_+Jh_])
    M0_t, M1_t, M2_t, K_t, C_t = _build_matrices(true_params_10, _Lb, d_)

    tol = 1e-10
    results_1 = {}
    for name, got, ref in [
        ('M0', M0_t, M0_ref), ('M1', M1_t, M1_ref), ('M2', M2_t, M2_ref),
        ('K',  K_t,  K_ref),  ('C',  C_t,  C_ref),
    ]:
        err = (got - ref).abs().max().item()
        ok  = err < tol
        results_1[name] = ok
        print(f"  {name}  max|error| = {err:.2e}   {'PASS' if ok else 'FAIL'}")

    status_1 = all(results_1.values())
    print(f"\nCheck 1: {'PASS' if status_1 else 'FAIL'}")

    # ------------------------------------------------------------------
    # Check 2 -- Detuned init: exp(log_params) == params_init exactly
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Check 2: _recover_params() reproduces detuned init values; log_params all zero")
    print("=" * 60)

    block = ParameterizedLFRBlock(RMSE_baseline=1.0)
    recovered = block._recover_params()

    all_ok = True
    # log_params must be all zeros at init
    log_zero_ok = (block.log_params.abs().max().item() == 0.0)
    print(f"  log_params all zeros at init     : {log_zero_ok}")
    if not log_zero_ok:
        all_ok = False

    for i, name in enumerate(_PARAM_NAMES):
        expected = _DETUNED_PARAMS[name]
        got_val  = recovered[i].item()
        err      = abs(got_val - expected)
        ok       = err < 1e-10
        if not ok:
            all_ok = False
        print(f"  {name:<10}  expected={expected:.6f}  got={got_val:.6f}  err={err:.2e}  {'PASS' if ok else 'FAIL'}")

    status_2 = all_ok
    print(f"\nCheck 2: {'PASS' if status_2 else 'FAIL'}")

    # ------------------------------------------------------------------
    # Check 3 -- Output shape and dtype
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Check 3: Output shape and dtype")
    print("=" * 60)

    x_test = torch.tensor([[0.05, 0.01, 0.30, 0.02, -0.01, 0.05]], dtype=torch.float32)
    u_test = torch.tensor([[10.0, -5.0, 3.0]], dtype=torch.float32)
    z_in   = torch.cat([x_test, u_test], dim=1).unsqueeze(-1)   # (1, 9, 1)

    with torch.no_grad():
        w_out = block.forward(z_in)

    shape_ok = w_out.shape == (1, 18, 1)
    dtype_ok = w_out.dtype == torch.float32
    print(f"  Output shape (1, 18, 1) : {shape_ok}  got {tuple(w_out.shape)}")
    print(f"  Output dtype float32    : {dtype_ok}  got {w_out.dtype}")
    status_3 = shape_ok and dtype_ok
    print(f"\nCheck 3: {'PASS' if status_3 else 'FAIL'}")

    # ------------------------------------------------------------------
    # Check 4 -- Physical consistency: detuned block vs direct rk4_step
    #           with the same detuned matrices
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Check 4: Physical consistency -- block vs direct rk4_step (detuned params)")
    print("=" * 60)

    from lpv_lfr_baseline.core.physics import P as P_ref, ts as ts_ref

    x_f64        = x_test.double()
    u_stage_f64  = u_test.double()
    u_logical_f64 = u_stage_f64 @ P_ref.T

    # Build detuned matrices directly for reference
    dp = torch.tensor([_DETUNED_PARAMS[n] for n in _PARAM_NAMES], dtype=dtype)
    kb1d, kb2d, cg1d, cg2d, cyd, cb1d, cb2d, mhd, m1d, m2d, mbd, Jbd, Jhd, dd = dp
    detuned_p_10 = torch.stack([kb1d+kb2d, cg1d, cg2d, cyd, cb1d+cb2d, mhd, m1d, m2d, mbd, Jbd+Jhd])
    _, M1_d, M2_d, K_d, C_d = _build_matrices(detuned_p_10, _Lb, dd)
    alpha_d, beta_d, gamma_d, N0_d, N1_d, N2_d = build_poly_constants(
        m1d, m2d, mbd, mhd, Jbd, Jhd, _Lb, dd
    )
    d0_d = mhd * (alpha_d * gamma_d - beta_d ** 2)
    G_d = build_G_matrix(N0_d, d0_d, M1_d, M2_d, K_d, C_d)

    with torch.no_grad():
        x_next_ref, _, _, _ = rk4_step(
            x_f64, u_logical_f64,
            G_d, K_d, C_d, mhd, alpha_d, beta_d, gamma_d, N0_d, N1_d, N2_d,
            ts_ref,
        )
        w_out_block = block.forward(z_in)

    x_next_block = w_out_block[0, :6, 0].double()
    err_4 = (x_next_block - x_next_ref[0]).abs().max().item()
    tol_4 = 1e-6
    status_4 = err_4 < tol_4
    print(f"  Max |x_next error| (block vs direct rk4_step) : {err_4:.2e}   {'PASS' if status_4 else 'FAIL'}")
    print(f"  (Expected: float32 rounding ~1e-7, tolerance {tol_4:.0e})")
    print(f"\nCheck 4: {'PASS' if status_4 else 'FAIL'}")

    # ------------------------------------------------------------------
    # Check 5 -- Gradient flows to log_params through full forward pass
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Check 5: Gradient flows to log_params through forward pass")
    print("=" * 60)

    z_grad = z_in.clone().float()
    w_out_g = block.forward(z_grad)
    w_out_g.sum().backward()

    grad_ok   = block.log_params.grad is not None
    grad_norm = block.log_params.grad.norm().item() if grad_ok else 0.0
    all_nonzero = grad_ok and (block.log_params.grad.abs() > 0).all().item()

    print(f"  log_params.grad is not None      : {grad_ok}")
    print(f"  All 10 grad entries non-zero     : {all_nonzero}")
    print(f"  log_params.grad norm             : {grad_norm:.6e}")
    status_5 = grad_ok and all_nonzero
    print(f"\nCheck 5: {'PASS' if status_5 else 'FAIL'}")

    # ------------------------------------------------------------------
    # Check 6 -- param_loss: zero when at init, positive when perturbed
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Check 6: param_loss is ~0 at init, positive when perturbed")
    print("=" * 60)

    # At init: log_params == log(params_init) -> exp(log_params) == params_init
    loss_at_init = block.param_loss().item()
    at_init_ok   = loss_at_init < 1e-20

    # Perturb: shift one param away from init
    block_p = ParameterizedLFRBlock(RMSE_baseline=1.0)
    with torch.no_grad():
        block_p.log_params[0] += 0.1   # shift kb_sum in log space
    loss_perturbed = block_p.param_loss().item()
    perturbed_ok   = loss_perturbed > 0.0

    print(f"  param_loss at init (expect ~0)   : {loss_at_init:.2e}   {'PASS' if at_init_ok else 'FAIL'}")
    print(f"  param_loss perturbed (expect >0) : {loss_perturbed:.6f}   {'PASS' if perturbed_ok else 'FAIL'}")
    status_6 = at_init_ok and perturbed_ok
    print(f"\nCheck 6: {'PASS' if status_6 else 'FAIL'}")

    # ------------------------------------------------------------------
    # Check 7 -- param_table printout
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Check 7: param_table() at init shows detuned vs true values")
    print("=" * 60)
    print(block.param_table())
    print(f"\nCheck 7: PASS (visual)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    all_pass = all([status_1, status_2, status_3, status_4, status_5, status_6])
    for label, ok in [
        ("Check 1 (_build_matrices vs physics.py)", status_1),
        ("Check 2 (detuned init round-trip)",       status_2),
        ("Check 3 (output shape/dtype)",            status_3),
        ("Check 4 (physical consistency)",          status_4),
        ("Check 5 (gradient to log_params)",        status_5),
        ("Check 6 (param_loss behavior)",           status_6),
    ]:
        print(f"  {label:<45} {'PASS' if ok else 'FAIL'}")
    print()
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print(f"nz={block.nz}, nw={block.nw}  |  Base class: {_BASE.__name__}")
    print("=" * 60)
