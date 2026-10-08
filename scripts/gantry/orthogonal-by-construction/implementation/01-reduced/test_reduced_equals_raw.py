"""Stage 1: does the reduced ten-combination block reproduce the raw fourteen-scalar transition?

__project_origin__ = "added"

`Reduced_Gantry_State_Block` (blocks.py, D-190) trains the ten identifiable combinations instead
of the fourteen raw scalars. The claim it rests on is that the map from the fourteen to
`(M0, M1, M2, C, K)` factors exactly through the ten, so the two transitions are the same
function and not merely close. This stage measures that, and three things around it.

  T1  the tensor combination map agrees with the reporting path `_combos_from_raw`
  T2  the gauge section is an exact section, `combos_of(gauge_section(v)) == v`
  T3  the reduced transition equals the raw transition at the nominal parameters, over designed
      points spanning `Y in [-0.30, 0.30]`                                    <- THE CRITERION
  T4  the same away from nominal, at displaced combinations, so T3 is not an artefact of both
      paths sitting at their initial values
  T5  GAUGE INVARIANCE: a different `mb` and a different `Jb:Jh` split at the SAME combinations
      leave the transition unchanged. This is what makes the gauge a gauge; if it failed, the
      arbitrary split would be entering the model.
  T6  admissibility over the scheduling range AND at the analytic worst Y (where det M(Y) is
      smallest, possibly far outside the grid), including saturation of the D-229 bounded
      mass-imbalance coordinate in both signs
  T7  that the reduced block's parameter is a ten-vector and the parent's fourteen-dimensional
      log coordinate is gone, so no checkpoint carries untrained tensors
  T8  the D-229 coordinate has the declared initial sensitivity and an exact inverse, batched and
      row-wise alike; the +-10 % detuned starts are reproduced; unrepresentable starts are refused

Criterion (handoff Sect. 10): `max abs (x_next(reduced) - x_next(raw)) / max abs (x_next - x)`
below `1e-12` over at least 24 designed points spanning the scheduling range.
"""

__project_origin__ = "added"

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C                                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# THEORY: the criterion is float64 round-off relative to the state increment, per handoff
# Sect. 10. It is a round-off threshold, not a tuned tolerance.
TOL = 1e-12
N_POINTS = 256          # well above the 24 the criterion asks for
Y_GRID = 24             # designed sweep points used for the per-Y table


def rel_err(a, b, x):
    """max abs difference, scaled by the max state INCREMENT (the handoff's reference scale)."""
    scale = (a - x[:, :C.N_P]).abs().max().item()
    return (a - b).abs().max().item() / scale, scale


def main():
    torch.manual_seed(0)
    sys.stdout = C.Tee(os.path.join(HERE, 'results.log'))
    from model_augmentation.fit_systems.blocks import (
        Parameterized_Gantry_State_Block, Reduced_Gantry_State_Block)

    theta0, Lb = C.nominal_raw()
    raw = C.make_raw_block()
    red = C.make_reduced_block()

    print('=' * 78)
    print('STAGE 1 -- reduced ten-combination block against the raw fourteen-scalar block')
    print('=' * 78)
    print(f'  Ts = {C.TS:.6e} s, up_sample = {C.UP_SAMPLE}, Y_op = None (LPV), float64, CPU')
    print(f'  criterion: relative difference below {TOL:.0e} over at least 24 designed points')

    # ---------------------------------------------------------------- T7 the parameter itself
    print('\n--- T7  the trainable parameter ---')
    names = [n for n, _ in red.named_parameters()]
    print(f'  reduced block parameters: {names}')
    print(f'  free_params shape {tuple(red.free_params.shape)}')
    assert names == ['free_params'], names
    assert 'log_params' not in dict(red.named_parameters())
    assert 'log_params' not in dict(red.named_buffers())
    print('  the parent fourteen-dimensional log coordinate is GONE from parameters and buffers')
    sd = [k for k in red.state_dict() if 'log_params' in k]
    print(f'  state_dict keys mentioning log_params: {sd}   (expected none)')
    assert not sd
    assert int(red.coordinate_version) == 2

    # ---------------------------------------------------------------- T1, T2 the maps
    print('\n--- T1  tensor combination map against the reporting path ---')
    v_tensor = Reduced_Gantry_State_Block.combos_of(theta0, Lb)
    v_report = Parameterized_Gantry_State_Block._combos_from_raw(
        dict(zip(Parameterized_Gantry_State_Block.PARAM_NAMES, [float(t) for t in theta0])), Lb)
    v_report = torch.tensor([v_report[n] for n in Reduced_Gantry_State_Block.COMBO_NAMES],
                            dtype=C.F64)
    d1 = (v_tensor - v_report).abs().max().item()
    for n, a in zip(Reduced_Gantry_State_Block.COMBO_NAMES, v_tensor):
        print(f'    {n:<9} {a.item():>14.8f}')
    print(f'  max abs difference to _combos_from_raw = {d1:.3e}')
    assert d1 == 0.0

    print('\n--- T8  constrained-coordinate sensitivity and inverse ---')
    zero = torch.zeros(10, dtype=C.F64)
    jac = torch.func.jacfwd(red.combinations_from_free)(zero)
    print(f'  d m_diff / d free_diff = {jac[red.M_DIFF_IX, red.M_DIFF_IX].item():+.9f} kg')
    print(f'  expected m_diff_init   = {red.combo_init[red.M_DIFF_IX].item():+.9f} kg')
    assert torch.allclose(jac[red.M_DIFF_IX, red.M_DIFF_IX],
                          red.combo_init[red.M_DIFF_IX], rtol=1e-14, atol=0)
    for seed in range(4):
        free = torch.randn(10, generator=torch.Generator().manual_seed(seed), dtype=C.F64) * 0.2
        recovered = red.free_from_combinations(red.combinations_from_free(free))
        err = (recovered - free).abs().max().item()
        print(f'  inverse round trip seed {seed}: max abs {err:.3e}')
        assert err < 1e-12
    batch = torch.randn(5, 10, generator=torch.Generator().manual_seed(9), dtype=C.F64) * 0.2
    rows = torch.stack([red.combinations_from_free(f) for f in batch])
    assert torch.allclose(red.combinations_from_free(batch), rows, rtol=1e-14, atol=0), \
        'batched map != row-wise'
    assert (red.free_from_combinations(rows) - batch).abs().max().item() < 1e-12
    print('  batched (5, 10) map and inverse equal the row-wise ones')
    # the D-222 detuned starts: +-10 % on every combination, m_diff included
    for sign in (1.0, -1.0):
        c0 = red.combo_init * (1.0 + 0.1 * sign)
        blk = C.make_reduced_block(combo_init=c0)
        j0 = torch.func.jacfwd(blk.combinations_from_free)(zero)
        e_start = ((blk.combinations_from_free(zero) - c0).abs() / c0.abs()).max().item()
        print(f'  start x{1 + 0.1 * sign:.1f}: max rel |c(0) - c0| = {e_start:.1e}, '
              f'd m_diff / d free_diff = {j0[blk.M_DIFF_IX, blk.M_DIFF_IX].item():+.6f} kg')
        assert e_start < 1e-14
        assert torch.allclose(j0[blk.M_DIFF_IX, blk.M_DIFF_IX], c0[blk.M_DIFF_IX], rtol=1e-14, atol=0)
    # starts the coordinate cannot represent are refused at construction
    for tag, md in (('zero m_diff', 0.0), ('m_diff beyond the bound', 40.0)):
        c0 = red.combo_init.clone()
        c0[red.M_DIFF_IX] = md
        try:
            C.make_reduced_block(combo_init=c0)
        except ValueError as exc:
            print(f'  {tag}: refused ({exc})')
        else:
            raise AssertionError(f'{tag} was accepted')

    print('\n--- T2  the gauge section is an exact section ---')
    for tag, v in (('nominal', v_tensor),
                   ('displaced x1.3', v_tensor * 1.3),
                   ('displaced, m_diff sign flipped',
                    v_tensor * torch.tensor([1.1, .9, 1.2, .8, 1.3, .95, 1.05, -1.4, .85, 1.15],
                                            dtype=C.F64))):
        r = Reduced_Gantry_State_Block.gauge_section(v, theta0, Lb)
        # Judged RELATIVE to each combination's own magnitude. The section subtracts `mb` and
        # `m_sum * Lb^2 / 4` and the inverse map adds them back, so off the initial values the
        # round-trip carries float64 cancellation error by construction; only at nominal, where
        # nothing is displaced, is it exactly zero.
        e = ((Reduced_Gantry_State_Block.combos_of(r, Lb) - v).abs()
             / v.abs().clamp(min=1e-300)).max().item()
        print(f'    {tag:<32} max REL (combos(section(v)) - v) = {e:.3e}')
        assert e < 1e-14, tag

    # ---------------------------------------------------------------- T3 the criterion
    print('\n--- T3  reduced vs raw transition at the nominal parameters (THE CRITERION) ---')
    x, u = C.design_points(N_POINTS, seed=0)
    z = C.zu(x, u)
    with torch.no_grad():
        xn_raw = raw(z).squeeze(-1)
        raw._cur = None
        xn_red = red(z).squeeze(-1)
        red._cur = None
    e3, scale3 = rel_err(xn_raw, xn_red, x)
    print(f'  {N_POINTS} points, Y swept over {C.Y_RANGE}')
    print(f'  max abs state increment (reference scale) = {scale3:.6e}')
    print(f'  max abs (reduced - raw)                   = '
          f'{(xn_raw - xn_red).abs().max().item():.6e}')
    print(f'  RELATIVE                                  = {e3:.3e}   '
          f'{"MET" if e3 < TOL else "NOT MET"} (< {TOL:.0e})')

    # per-Y table so the scheduling range is visibly covered, not just aggregated over
    print('\n  per-Y breakdown over the designed sweep:')
    print(f'    {"Y [m]":>8}  {"max abs diff":>14}  {"relative":>12}')
    xg, ug = C.design_points(Y_GRID, seed=1)
    for i in range(Y_GRID):
        zi = C.zu(xg[i:i + 1], ug[i:i + 1])
        with torch.no_grad():
            a = raw(zi).squeeze(-1)
            raw._cur = None
            b = red(zi).squeeze(-1)
            red._cur = None
        ei, si = rel_err(a, b, xg[i:i + 1])
        print(f'    {xg[i, 2].item():>8.4f}  {(a - b).abs().max().item():>14.4e}  {ei:>12.3e}')
        assert ei < TOL, f'Y = {xg[i, 2].item()}'

    # ---------------------------------------------------------------- T4 away from nominal
    print('\n--- T4  the same at DISPLACED combinations ---')
    print(f'    {"displacement":>14}  {"max abs diff":>14}  {"relative":>12}')
    for frac in (0.01, 0.05, 0.20, 0.50):
        # Same physical parameters reached from both sides: displace the combinations, then set
        # the raw block to the gauge preimage of those same combinations.
        v = v_tensor * (1.0 + frac)
        with torch.no_grad():
            red.free_params.copy_(red.free_from_combinations(v))
            raw_pre = Reduced_Gantry_State_Block.gauge_section(v, theta0, Lb)
            raw.log_params.copy_(torch.log(raw_pre / raw.params_init))
            a = raw(z).squeeze(-1)
            raw._cur = None
            b = red(z).squeeze(-1)
            red._cur = None
        e, s = rel_err(a, b, x)
        print(f'    {frac:>13.0%}  {(a - b).abs().max().item():>14.4e}  {e:>12.3e}')
        assert e < TOL, f'displacement {frac}'
    with torch.no_grad():
        red.free_params.zero_()
        raw.log_params.zero_()

    # ---------------------------------------------------------------- T5 gauge invariance
    print('\n--- T5  gauge invariance: a DIFFERENT gauge, the same combinations ---')
    print('  If the arbitrary split entered the model these would differ. The reference is the')
    print('  nominal gauge (mb = 22.8, Jb:Jh = 1.0:0.05).')
    print(f'    {"gauge":<34}  {"max abs diff":>14}  {"relative":>12}')
    for tag, mb_new, jratio in (('mb = 5.0, Jb:Jh unchanged', 5.0, None),
                                ('mb = 40.0, Jb:Jh unchanged', 40.0, None),
                                ('mb unchanged, Jb:Jh = 1:1', None, 1.0),
                                ('mb = 1.0, Jb:Jh = 1:9', 1.0, 9.0)):
        alt = theta0.clone()
        if mb_new is not None:
            alt[10] = mb_new
        if jratio is not None:
            tot = alt[11] + alt[12]
            alt[11] = tot / (1 + jratio)
            alt[12] = tot * jratio / (1 + jratio)
        red_alt = C.make_reduced_block(params_init=alt, combo_init=v_tensor)
        # The alternative gauge must give the SAME ten combinations, or the test is vacuous.
        dv = ((red_alt.recover_combinations() - red.recover_combinations()).abs()
              / red.recover_combinations().abs()).max().item()
        with torch.no_grad():
            b = red_alt(z).squeeze(-1)
            red_alt._cur = None
        e, s = rel_err(xn_red, b, x)
        print(f'    {tag:<34}  {(xn_red - b).abs().max().item():>14.4e}  {e:>12.3e}'
              f'   (relative combination difference {dv:.1e})')
        assert dv < 1e-14, tag
        assert e < TOL, tag

    # ---------------------------------------------------------------- T6 admissibility
    print('\n--- T6  admissibility over the scheduling range and at the worst Y ---')
    # The guarantee holds for EVERY Y, not only on the +-0.30 m grid. det M(Y) is quadratic in Y
    # with its minimum at Y* = -Lb m_diff / (2 m_total), which leaves the grid when m_diff nears
    # the bound and J_eff / m_total grows; so M is also checked at Y* itself.
    sat = torch.zeros(10, dtype=C.F64)
    sat[red.M_DIFF_IX] = 1e6
    wide = torch.zeros(10, dtype=C.F64)
    wide[red.J_EFF_IX], wide[red.M_TOTAL_IX] = 5.0, -5.0      # J_eff up, m_total down: Y* far out
    for tag, free in (('nominal', torch.zeros(10, dtype=C.F64)),
                      ('+20% all', torch.full((10,), 0.2, dtype=C.F64)),
                      ('-20% all', torch.full((10,), -0.2, dtype=C.F64)),
                      ('saturated m_diff +', sat),
                      ('saturated m_diff -', -sat),
                      ('saturated, Y* far out', wide + sat),
                      ('saturated, Y* far out -', wide - sat)):
        with torch.no_grad():
            red.free_params.copy_(free)
        c = red.recover_combinations().detach()
        y_star = float(-red.Lb * c[red.M_DIFF_IX] / (2 * c[red.M_TOTAL_IX]))
        a = red.admissibility()
        a_star = red.admissibility(y_range=(y_star, y_star), n=1)
        print(f'    {tag:<24} grid min eig {a["min_eig_M"]:>11.4e} at Y = {a["min_eig_M_at_Y"]:+.3f} m;'
              f'  Y* = {y_star:+9.3f} m: min eig {a_star["min_eig_M"]:>11.4e}, det {a_star["min_abs_d"]:>11.4e}')
        assert a['admissible'] and a_star['admissible'], tag
    with torch.no_grad():
        red.free_params.zero_()

    print('\n' + '=' * 78)
    print(f'STAGE 1 VERDICT: relative difference {e3:.3e} against a criterion of {TOL:.0e} -- '
          f'{"MET" if e3 < TOL else "NOT MET"}')
    print('=' * 78)
    sys.stdout.flush()


if __name__ == '__main__':
    main()
