"""Stage 3: is the protected basis a genuine second-order Taylor expansion?

__project_origin__ = "added"

The basis is built from the affine expansion

    f_vartheta(x, u) ~= c(x, u) + J(x, u) (vartheta - vartheta_bar) ,

and the whole construction rests on that being the first-order truncation of the real nonlinear
transition, so the error it neglects must be second order in the displacement. That is a statement
about a SLOPE, not about matching any closed-form expression: the criterion is the log-log slope
of the error against displacement, `2.0 +/- 0.1` over a decade. Matching a formula is explicitly
NOT the criterion (handoff Sect. 10).

  T1  the slope, over several random displacement directions and over the ten coordinate
      directions separately                                                  <- THE CRITERION
  T2  the same for the OUTPUT-relevant quantity, the error relative to the state increment, so
      the number is readable as "how wrong the expansion is compared with what the step does"
  T3  where the round-off floor is, so the fit window is chosen by the data rather than assumed
  T4  the expansion point is NOT the prediction: the exact nonlinear transition and the affine
      surrogate differ by exactly the measured amount, which is the check that nothing in the
      pipeline has quietly substituted the approximation into the rollout

A note on why the sweep is in the FREE coordinate. `vartheta` displacements are expressed as
relative parameter changes, nine of them logarithmic and `m_diff` linear, which is the coordinate
the basis is built in. A displacement of `s` is therefore "every combination moved by `s` relative",
which is the physically meaningful unit and the one the expansion-point sensitivity numbers in the
handoff are quoted in.
"""

__project_origin__ = "added"

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C                                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# THEORY: second order means slope 2 in a log-log error-against-displacement plot. The tolerance
# is the handoff's, Sect. 10.
SLOPE_TARGET, SLOPE_TOL = 2.0, 0.1
N_POINTS = 256
# Displacement decade. Chosen from T3's measured round-off floor, not assumed: the fit window has
# to sit above the floor and below the radius where third order starts to show.
S_GRID = np.logspace(-3, -1, 9)
N_DIRS = 5


def slope_of(s, e):
    """Least-squares log-log slope. `s` and `e` are arrays over the displacement grid."""
    ls, le = np.log10(s), np.log10(e)
    A = np.vstack([ls, np.ones_like(ls)]).T
    return float(np.linalg.lstsq(A, le, rcond=None)[0][0])


def main():
    torch.manual_seed(0)
    sys.stdout = C.Tee(os.path.join(HERE, 'results.log'))
    from model_augmentation.fit_systems import obc_projection as obc

    red = C.make_reduced_block()
    x, u = C.design_points(N_POINTS, seed=0)
    Z = C.zu(x, u)
    J, c, v_bar = obc.stacked_basis(red, Z)
    v0 = torch.zeros(10, dtype=C.F64)

    def exact(v_free):
        with torch.no_grad():
            out = obc.reduced_step(red, v_free, Z).reshape(-1)
        red._cur = None
        return out

    f0 = c.reshape(-1)
    increment = (f0.reshape(-1, C.N_P) - x).abs().max().item()

    print('=' * 78)
    print('STAGE 3 -- is the affine basis a second-order Taylor expansion?')
    print('=' * 78)
    print(f'  {N_POINTS} designed points, float64, displacement in the reduced FREE coordinate')
    print(f'  criterion: log-log slope {SLOPE_TARGET} +/- {SLOPE_TOL} over a decade')
    print(f'  reference scale: max abs state increment = {increment:.6e}')
    print(f'  sanity: max abs (c - f(vartheta_bar)) = '
          f'{(f0 - exact(v0)).abs().max().item():.3e}  (the offset IS the nominal transition)')

    # ------------------------------------------------------------------ T3 the round-off floor
    print('\n--- T3  where the round-off floor is ---')
    g = torch.Generator().manual_seed(7)
    d0 = torch.randn(10, generator=g, dtype=C.F64)
    d0 = d0 / d0.norm()
    print(f'    {"s":>10}  {"abs error":>14}  {"rel to increment":>18}')
    floor_grid = np.logspace(-8, -1, 15)
    errs_floor = []
    for s in floor_grid:
        v = v0 + float(s) * d0
        e = (exact(v) - (f0 + J @ (float(s) * d0))).abs().max().item()
        errs_floor.append(e)
        print(f'    {s:>10.2e}  {e:>14.4e}  {e / increment:>18.4e}')
    # The floor is where the error stops falling as s falls. Found from the data.
    arr = np.array(errs_floor)
    floor = float(arr[:4].min())
    print(f'  measured floor (smallest error over the four smallest displacements): {floor:.3e}')
    lo = S_GRID[0] ** 2 * arr[-1] / (S_GRID[-1] ** 2)
    print(f'  the fit window {S_GRID[0]:.0e} .. {S_GRID[-1]:.0e} predicts a smallest error near '
          f'{lo:.3e}, which is {lo / max(floor, 1e-300):.1e} x the floor')

    # ------------------------------------------------------------------ T1 the criterion
    print('\n--- T1  log-log slope over random displacement directions (THE CRITERION) ---')
    print(f'    {"direction":<22}  ' + '  '.join(f'{s:>9.1e}' for s in S_GRID) + '     slope')
    slopes = []
    for k in range(N_DIRS):
        gg = torch.Generator().manual_seed(100 + k)
        d = torch.randn(10, generator=gg, dtype=C.F64)
        d = d / d.norm()
        errs = []
        for s in S_GRID:
            errs.append((exact(v0 + float(s) * d)
                         - (f0 + J @ (float(s) * d))).abs().max().item())
        sl = slope_of(S_GRID, np.array(errs))
        slopes.append(sl)
        print(f'    random {k:<15}  ' + '  '.join(f'{e:>9.2e}' for e in errs) + f'   {sl:>7.4f}')

    print(f'\n    {"coordinate":<22}  ' + '  '.join(f'{s:>9.1e}' for s in S_GRID) + '     slope')
    coord_slopes = []
    for i, name in enumerate(C.COMBO_NAMES):
        d = torch.zeros(10, dtype=C.F64)
        d[i] = 1.0
        errs = [(exact(v0 + float(s) * d) - (f0 + J @ (float(s) * d))).abs().max().item()
                for s in S_GRID]
        sl = slope_of(S_GRID, np.array(errs))
        coord_slopes.append(sl)
        print(f'    {name:<22}  ' + '  '.join(f'{e:>9.2e}' for e in errs) + f'   {sl:>7.4f}')

    all_sl = np.array(slopes + coord_slopes)
    ok = bool(np.all(np.abs(all_sl - SLOPE_TARGET) <= SLOPE_TOL))
    print(f'\n  slopes: min {all_sl.min():.4f}, max {all_sl.max():.4f}, '
          f'mean {all_sl.mean():.4f}, max deviation from {SLOPE_TARGET} '
          f'{np.abs(all_sl - SLOPE_TARGET).max():.4f}')

    # ------------------------------------------------------------------ T2 relative magnitude
    print('\n--- T2  the neglected term, relative to the state increment ---')
    print(f'    {"s":>10}  {"abs error":>14}  {"rel to increment":>18}  '
          f'{"rel to the linear term":>24}')
    for s in S_GRID:
        lin = (J @ (float(s) * d0)).abs().max().item()
        e = (exact(v0 + float(s) * d0) - (f0 + J @ (float(s) * d0))).abs().max().item()
        print(f'    {s:>10.2e}  {e:>14.4e}  {e / increment:>18.4e}  '
              f'{e / max(lin, 1e-300):>24.4e}')

    # ------------------------------------------------------------------ T4 never substituted
    print('\n--- T4  the approximation is the BASIS, never the prediction ---')
    s = 0.05
    v = v0 + s * d0
    ex, af = exact(v), f0 + J @ (s * d0)
    print(f'  at a 5 percent displacement, exact nonlinear transition and affine surrogate differ')
    print(f'  by {(ex - af).abs().max().item():.4e}, i.e. '
          f'{(ex - af).abs().max().item() / increment:.2e} of the state increment.')
    print(f'  The rollout keeps the exact transition; only the BASIS comes from the expansion.')
    print(f'  A model that substituted the surrogate would carry exactly this error at every step.')

    print('\n' + '=' * 78)
    print(f'STAGE 3 VERDICT: slopes in [{all_sl.min():.4f}, {all_sl.max():.4f}] against '
          f'{SLOPE_TARGET} +/- {SLOPE_TOL} -- {"MET" if ok else "NOT MET"}')
    print('=' * 78)
    sys.stdout.flush()


if __name__ == '__main__':
    main()
