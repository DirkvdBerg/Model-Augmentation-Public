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
  T6  admissibility over the scheduling range, `min eig M(Y) > 0` and `d(Y) != 0`, which
      positivity of the ten combinations does not imply
  T7  that the reduced block's parameter is a ten-vector and the parent's fourteen-dimensional
      log coordinate is gone, so no checkpoint carries untrained tensors

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
            red.free_params.copy_(torch.where(
                red._m_diff_mask,
                torch.full((10,), frac, dtype=C.F64),
                torch.log(v / red.combo_init)))
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
    print('\n--- T6  admissibility over the scheduling range ---')
    for tag, free in (('nominal', torch.zeros(10, dtype=C.F64)),
                      ('+20% all', torch.full((10,), 0.2, dtype=C.F64)),
                      ('-20% all', torch.full((10,), -0.2, dtype=C.F64))):
        with torch.no_grad():
            red.free_params.copy_(free)
        a = red.admissibility()
        print(f'    {tag:<10} min eig M(Y) = {a["min_eig_M"]:>10.4f} kg at Y = '
              f'{a["min_eig_M_at_Y"]:+.3f} m,  min abs d(Y) = {a["min_abs_d"]:>12.4e} at Y = '
              f'{a["min_abs_d_at_Y"]:+.3f} m,  admissible = {a["admissible"]}')
        assert a['admissible'], tag
    with torch.no_grad():
        red.free_params.zero_()

    print('\n' + '=' * 78)
    print(f'STAGE 1 VERDICT: relative difference {e3:.3e} against a criterion of {TOL:.0e} -- '
          f'{"MET" if e3 < TOL else "NOT MET"}')
    print('=' * 78)
    sys.stdout.flush()


if __name__ == '__main__':
    main()
