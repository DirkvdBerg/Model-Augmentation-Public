"""Gates C1-C5 for the CoG-corrected LFR baseline and for the exact truth replay.

Nothing downstream is trustworthy until these pass, so they run first and they
run on their own.

  C1  ma = 0 reproduces the parent Gantry_State_Block                  (no-op gate)
  C2  N_c(Y)/d_c(Y) is inv(M_c(Y))                                     (algebra gate)
  C3  M_c(Y) is the truth's own 3x3 block at delta_a = 0               (target gate)
  C4  the CoG correction actually CHANGES the derivative               (a no-op gate
      that passes because nothing happened is the classic trap; T2 in the
      coulomb-offset log. C4 exists so C1 cannot pass for the wrong reason.)
  C5  the exact 8-state replay reproduces the record's positions       (truth gate)

Run:
  PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output \\
      -n GraduationProject python -u \\
      scripts/gantry/true-init-augmentation/check_plant_cog.py
"""
__project_origin__ = "added"

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))

from plant_cog import (                                              # noqa: E402
    Gantry_State_Block_CoG, cog_constants, mass_matrix_cog, make_block)
from data_exact import exact_truth, gate_replay, fd_velocity_error   # noqa: E402

sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry'))
from gantry_dynamic.oracle import _M as _M8, MA, L0                  # noqa: E402
from model_augmentation.fit_systems.blocks import Gantry_State_Block  # noqa: E402

OUT = os.path.join(REPO, 'simulations', 'gantry_subnet', 'diagnostics')
Y_SWEEP = np.linspace(-0.35, 0.35, 71)
REC = 'V1_standstill_Yp10'


def _rand_state(n, rng):
    x = np.zeros((n, 6))
    x[:, 0] = rng.normal(0, 1e-4, n)          # X    [m]
    x[:, 1] = rng.normal(0, 1e-4, n)          # Th   [rad]
    x[:, 2] = rng.uniform(-0.35, 0.35, n)     # Y    [m]   full operating range
    x[:, 3:] = rng.normal(0, 1e-2, (n, 6 - 3))
    return x


def main():
    res = {}
    print('Gates for the CoG-corrected LFR baseline\n')
    rng = np.random.default_rng(0)
    x = _rand_state(64, rng)
    u = rng.normal(0, 10.0, (64, 3))
    xt = torch.as_tensor(x, dtype=torch.float64).reshape(64, 6, 1)
    ut = torch.as_tensor(u, dtype=torch.float64).reshape(64, 3, 1)

    # ---- C1a: at ma = 0 the DERIVED CONSTANTS are the framework's own ------
    # Compared in float64 against build_poly_constants / build_G_matrix_entries
    # called with float64 inputs. This is the exact algebraic gate: the parent
    # BLOCK carries float32 constants (gantry_ss is float32), so comparing
    # against the block instead would measure float32 rounding, not the algebra.
    n0r, n1r, n2r, dp_ref, _, m1r, m2r = None, None, None, None, None, None, None
    from model_augmentation.systems.gantry_ss import (
        build_poly_constants, build_G_matrix_entries as _bG,
        m1 as g_m1, m2 as g_m2, mb as g_mb, mh as g_mh, Jb as g_Jb, Jh as g_Jh,
        Lb as g_Lb, d as g_d, M1 as g_M1, M2 as g_M2, K as g_K, C as g_C)
    f64 = torch.float64
    a_r, b_r, g_r, n0r, n1r, n2r = build_poly_constants(
        *(t.to(f64) for t in (g_m1, g_m2, g_mb, g_mh, g_Jb, g_Jh, g_Lb, g_d)))
    d0r = g_mh.to(f64) * (a_r * g_r - b_r ** 2)
    A_ref = _bG(n0r, d0r, g_M1.to(f64), g_M2.to(f64), g_K.to(f64), g_C.to(f64))[3]

    N0z, N1z, N2z, dpz, _, M1z, M2z = cog_constants(0.0, float(L0))
    d0z = g_mh.to(f64) * dpz[0]
    A_z = _bG(N0z, d0z, M1z, M2z, g_K.to(f64), g_C.to(f64))[3]

    def _rel(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))

    c1a = max(_rel(N0z, n0r), _rel(N1z, n1r), _rel(N2z, n2r),
              _rel(float(d0z), float(d0r)), _rel(A_z, A_ref),
              _rel(M1z, g_M1.to(f64)), _rel(M2z, g_M2.to(f64)))
    res['C1a_ma0_constants_rel'] = c1a
    print(f'  C1a ma = 0 constants vs framework, max rel  {c1a:.3e}   '
          f'{"PASS" if c1a < 1e-14 else "FAIL"}  (tol 1e-14, float64 both sides)')

    # ---- C1b: and the resulting derivative matches the stock block ---------
    parent = make_block(cog=False, dtype=torch.float64)
    off = Gantry_State_Block_CoG(
        ma=0.0, l0=float(L0), Y_op=None, Ts=1 / 4000, up_sample=1,
        std_x=np.ones((6, 1)), std_u=np.ones((3, 1)),
        x_mean=np.zeros((6, 1)), u_mean=np.zeros((3, 1))).to(torch.float64)
    with torch.no_grad():
        dp = parent.deriv(xt, ut).squeeze(-1).numpy()
        d0 = off.deriv(xt, ut).squeeze(-1).numpy()
    scale = np.abs(dp).max(axis=0) + 1e-300
    c1 = float((np.abs(dp - d0) / scale).max())
    res['C1b_ma0_vs_parent_rel'] = c1
    print(f'  C1b ma = 0 vs stock block, max rel |dxdot|  {c1:.3e}   '
          f'{"PASS" if c1 < 1e-6 else "FAIL"}  (tol 1e-6 = float32 constant'
          f' precision; gantry_ss stores float32)')

    # ---- C2: the rational form IS the inverse ------------------------------
    N0, N1, N2, dp_, M0c, M1c, M2c = cog_constants(float(MA), float(L0))
    mh = float(torch.as_tensor(M0c[2, 2]))
    worst_inv, worst_M = 0.0, 0.0
    for Y in Y_SWEEP:
        Nn = (N0 + N1 * Y + N2 * Y ** 2).numpy()
        dd = mh * float(dp_[0] + dp_[1] * Y + dp_[2] * Y ** 2)
        Mc = mass_matrix_cog(Y)
        Mpoly = (M0c + M1c * Y + M2c * Y ** 2).numpy()
        worst_M = max(worst_M, float(np.abs(Mpoly - Mc).max()))
        worst_inv = max(worst_inv, float(np.abs(Mc @ (Nn / dd) - np.eye(3)).max()))
    res['C2_max_MinvN_minus_I'] = worst_inv
    res['C2_max_Mpoly_minus_M'] = worst_M
    print(f'  C2  max |M_c(Y) N_c(Y)/d_c(Y) - I|          {worst_inv:.3e}   '
          f'{"PASS" if worst_inv < 1e-12 else "FAIL"}  (tol 1e-12, 71 Y in [-.35,.35])')
    print(f'      max |M0+YM1+Y^2M2 - M_c(Y)|             {worst_M:.3e}')

    # ---- C3: M_c is the truth's own block at delta_a = 0 -------------------
    worst_t = 0.0
    for Y in Y_SWEEP:
        truth33 = _M8(Y, 0.0)[:3, :3]
        worst_t = max(worst_t, float(np.abs(truth33 - mass_matrix_cog(Y)).max()))
    res['C3_max_vs_truth_M_at_da0'] = worst_t
    print(f'  C3  max |M_c(Y) - M_truth(Y, da=0)[:3,:3]|  {worst_t:.3e}   '
          f'{"PASS" if worst_t < 1e-12 else "FAIL"}  (tol 1e-12)')

    # ---- C4: it is not a no-op --------------------------------------------
    on = make_block(cog=True, dtype=torch.float64)
    with torch.no_grad():
        don = on.deriv(xt, ut).squeeze(-1).numpy()
    c4 = float((np.abs(don - dp) / scale).max())
    res['C4_cog_effect_rel'] = c4
    print(f'  C4  CoG on vs off, max rel |d xdot|         {c4:.3e}   '
          f'{"PASS" if c4 > 1e-6 else "FAIL"}  (must be NON-zero)')
    print(f'      ma*L0 = {float(MA)*float(L0):.4f} kg*m, '
          f'Theta inertia delta at Y=0.10 = {float(MA)*(2*0.10*float(L0)+float(L0)**2):.4f} kg*m^2')

    # ---- C5: the exact truth replay is the record -------------------------
    print(f'\n  C5  exact 8-state replay vs record positions, {REC}')
    tr = exact_truth(REC)
    ok5 = gate_replay(tr)
    res['C5_replay_max_abs'] = [float(v) for v in tr['gate']]
    res['C5_pass'] = bool(ok5)

    fd = fd_velocity_error(tr)
    print(f'\n  How wrong are the record\'s finite-difference velocities?')
    print(f'  {"":8}{"dX [m/s]":>14}{"dTheta [rad/s]":>16}{"dY [m/s]":>14}')
    print(f'  {"max":8}' + ''.join(f'{v:>14.4e}' if i != 1 else f'{v:>16.4e}'
                                   for i, v in enumerate(fd['max_abs'])))
    print(f'  {"rms":8}' + ''.join(f'{v:>14.4e}' if i != 1 else f'{v:>16.4e}'
                                   for i, v in enumerate(fd['rms'])))
    print(f'  {"rel":8}' + ''.join(f'{v:>14.4e}' if i != 1 else f'{v:>16.4e}'
                                   for i, v in enumerate(fd['rel'])))
    res['fd_velocity_rms'] = [float(v) for v in fd['rms']]
    res['fd_velocity_rel'] = [float(v) for v in fd['rel']]
    print('  (this is the error a K0-style seed injects; on a K = 0 axis it does '
          'not decay,\n   it displaces the window by ~dv * nf * ts)')

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, 'true_init_gates.json')
    with open(p, 'w') as f:
        json.dump(res, f, indent=2)
    print(f'\n  wrote {p}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
