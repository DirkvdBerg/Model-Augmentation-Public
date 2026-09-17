"""Validate the hand-written forward sensitivity against two independent oracles (D-194).

The correction `J_b(vbar, x, u) theta` is now computed by propagating a tangent through the RK4
stages in plain tensor operations (`transition_and_tangent_from_free`), because `torch.func.jvp`
inside a compiled step miscompiles to an illegal memory access on this stack. That replacement is
a SECOND implementation of a derivative, so it is only safe if it is gated.

Three legs, all local, CPU, float64, seconds to run:

  A  per-term against `torch.func.jvp`   which term is wrong, not just that something is
  B  end to end against `torch.func.jvp`  the definition it must reproduce
  C  end to end against a central finite difference of the transition itself, which uses no
     autodiff at all, so a shared misreading of the code cannot hide in it
  D  linearity and scaling in the direction, which a derivative must satisfy by definition
  E  float32 agreement and timing, reported rather than gated

Leg A is the one that earns its keep during development: with twenty lines of hand-derived
calculus, knowing that `a` is right and `dY` is wrong is most of the debugging.

Run:
    conda run -n GraduationProject python scripts/gantry/orthogonal-by-construction/\\
implementation/08-one-step/test_tangent.py
"""
__project_origin__ = "added"

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rig import RESULTS_DIR, Tee                                      # noqa: E402
from model_augmentation.fit_systems.blocks import (                   # noqa: E402
    Reduced_Gantry_State_Block, Parameterized_Gantry_State_Block)
from model_augmentation.systems import gantry_ss as gss               # noqa: E402

F64 = torch.float64
BATCH = 64
TS = 1.0 / 4000.0        # production rate
UP_SAMPLE = 1            # production value (CFG.up_sample)
TERMS = ('fnet', 'dY', 'a', 'z', 'w', 'xdot_phys', 'xdot')
RESULTS = {}


def verdict(name, ok, detail=None):
    RESULTS[name] = {'met': bool(ok), 'detail': detail}
    print('  -> %s: %s' % ('PASS' if ok else 'FAIL', name))
    return ok


def make_block(detune=None):
    """A reduced block at the production settings, float64, identity normalisation.

    Identity `std_x` / `x_mean` is deliberate: it keeps every printed quantity in physical units
    and makes a term-by-term comparison readable. The tangent is linear in those constants, so
    they cannot hide an error.
    """
    names = Parameterized_Gantry_State_Block.PARAM_NAMES
    nominal = torch.stack([getattr(gss, n) for n in names]).to(F64)
    combo = Reduced_Gantry_State_Block.combos_of(nominal, gss.Lb)
    if detune is not None:
        combo = combo * combo.new_tensor(detune)
    blk = Reduced_Gantry_State_Block(
        params_init=nominal, combo_init=combo, Y_op=None,
        std_x=np.ones((6, 1)), std_u=np.ones((3, 1)),
        x_mean=np.zeros((6, 1)), u_mean=np.zeros((3, 1)),
        Ts=TS, up_sample=UP_SAMPLE, RMSE_baseline=0.01).to(F64)
    return blk


def sample_point(seed=0, batch=BATCH):
    """A physically plausible (x, u) batch: Y across its whole scheduling range."""
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(batch, 6, 1, dtype=F64)
    x[:, 0, 0] = torch.rand(batch, generator=g, dtype=F64) * 0.4 - 0.2      # X   [m]
    x[:, 1, 0] = torch.rand(batch, generator=g, dtype=F64) * 0.02 - 0.01    # Th  [rad]
    x[:, 2, 0] = torch.rand(batch, generator=g, dtype=F64) * 0.6 - 0.3      # Y   [m], full range
    x[:, 3:, 0] = torch.randn(batch, 3, generator=g, dtype=F64) * 0.1       # velocities
    u = torch.randn(batch, 3, 1, generator=g, dtype=F64) * 50.0             # stage forces [N]
    return torch.cat([x, u], dim=1)


def rel(a, b):
    """Relative 2-norm difference, scaled by the OPERANDS so a near-zero term is not flattered."""
    num = torch.linalg.vector_norm((a - b).reshape(-1))
    den = 0.5 * (torch.linalg.vector_norm(a.reshape(-1))
                 + torch.linalg.vector_norm(b.reshape(-1))) + 1e-300
    return float(num / den)


def main():
    sys.stdout = Tee(os.path.join(RESULTS_DIR, 'test_tangent.log'))
    torch.manual_seed(0)
    blk = make_block()
    z = sample_point(0)
    g = torch.Generator().manual_seed(7)
    free = torch.randn(10, generator=g, dtype=F64) * 0.05      # a displaced expansion point
    dfree = torch.randn(10, generator=g, dtype=F64)            # the direction (the coefficient)
    print('block: Y_op=None, Ts=%g, up_sample=%d, batch=%d, float64' % (TS, UP_SAMPLE, BATCH))
    print('free  = ' + ' '.join('%+.4f' % v for v in free))
    print('dfree = ' + ' '.join('%+.4f' % v for v in dfree))

    def mats_of(v):
        return blk._mats_from_raw(
            blk.gauge_section(blk.combinations_from_free(v), blk.params_init, blk.Lb))

    # ---- 0. the once-per-objective matrix tangent ---------------------------------------
    print('\n[0] dmats against torch.func.jvp of the parameter-to-matrices map')
    mats, dmats = blk.mats_and_tangent_from_free(free, dfree)
    mats_ref, dmats_ref = torch.func.jvp(mats_of, (free,), (dfree,))
    names = ('K', 'C', 'A_combined', 'mh', 'alpha', 'beta', 'gamma', 'N0', 'N1', 'N2')
    worst0 = 0.0
    for n, mv, mr, dv, dr in zip(names, mats, mats_ref, dmats, dmats_ref):
        e_p, e_t = rel(mv, mr), rel(dv, dr)
        worst0 = max(worst0, e_p, e_t)
        print('    %-11s primal %.3e   tangent %.3e' % (n, e_p, e_t))
    verdict('T0_dmats', worst0 < 1e-12, worst0)

    # ---- A. per-term, one deriv call ----------------------------------------------------
    print('\n[A] per-term tangent of one _deriv_with call against torch.func.jvp')
    x, u = z[:, :6, :], z[:, 6:, :]
    sx0 = torch.zeros_like(x)
    tp, tt = {}, {}
    blk._deriv_with(x, u, mats, terms=tp)
    blk._deriv_tangent_with(x, sx0, u, mats, dmats, terms=tt)

    def term_of(v, key):
        t = {}
        blk._deriv_with(x, u, mats_of(v), terms=t)
        return t[key]

    worstA, rowsA = 0.0, {}
    for key in TERMS:
        prim_ref, tan_ref = torch.func.jvp(lambda v, k=key: term_of(v, k), (free,), (dfree,))
        e_p, e_t = rel(tp[key], prim_ref), rel(tt[key], tan_ref)
        scale = float(torch.linalg.vector_norm(tan_ref.reshape(-1)))
        rowsA[key] = {'primal_rel': e_p, 'tangent_rel': e_t, 'tangent_norm': scale}
        worstA = max(worstA, e_t)
        print('    %-10s primal %.3e   tangent %.3e   (||tangent|| = %.3e)' % (key, e_p, e_t, scale))
    verdict('A_per_term', worstA < 1e-12, rowsA)

    # ---- B. end to end against the jvp --------------------------------------------------
    print('\n[B] full transition and tangent against torch.func.jvp of transition_from_free')
    xp, sp = blk.transition_and_tangent_from_free(free, dfree, z)
    xp_ref, sp_ref = torch.func.jvp(lambda v: blk.transition_from_free(v, z), (free,), (dfree,))
    eB_p, eB_t = rel(xp, xp_ref), rel(sp, sp_ref)
    print('    primal  %.3e' % eB_p)
    print('    tangent %.3e   (||tangent|| = %.3e, ||state increment|| = %.3e)'
          % (eB_t, float(sp.norm()), float((xp - x).norm())))
    verdict('B_end_to_end_jvp', eB_p < 1e-14 and eB_t < 1e-12, {'primal': eB_p, 'tangent': eB_t})

    # ---- C. end to end against a central finite difference ------------------------------
    # Uses no autodiff at all: it only EVALUATES the transition, so an error shared between the
    # hand derivation and functorch cannot hide here.
    print('\n[C] against a central finite difference of the transition (no autodiff)')
    best, rowsC = 1.0, {}
    with torch.no_grad():
        for eps in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
            h = eps / max(float(dfree.norm()), 1e-300)
            fd = (blk.transition_from_free(free + h * dfree, z)
                  - blk.transition_from_free(free - h * dfree, z)) / (2 * h)
            e = rel(sp, fd)
            rowsC['eps%g' % eps] = e
            best = min(best, e)
            print('    eps %.0e   relative %.3e' % (eps, e))
    print('    best over the sweep %.3e. A central difference is O(eps^2) and floors out where '
          'round-off takes over;' % best)
    print('    anything below 1e-8 rules out a sign error or a missing term.')
    verdict('C_finite_difference', best < 1e-8, rowsC)

    # ---- D. the tangent must be linear in the direction ---------------------------------
    print('\n[D] linearity and scaling in the direction')
    d2 = torch.randn(10, generator=torch.Generator().manual_seed(11), dtype=F64)
    _, s_a = blk.transition_and_tangent_from_free(free, dfree, z)
    _, s_b = blk.transition_and_tangent_from_free(free, d2, z)
    _, s_sum = blk.transition_and_tangent_from_free(free, dfree + d2, z)
    _, s_scaled = blk.transition_and_tangent_from_free(free, 3.7 * dfree, z)
    _, s_zero = blk.transition_and_tangent_from_free(free, torch.zeros_like(dfree), z)
    eD_add, eD_scale = rel(s_sum, s_a + s_b), rel(s_scaled, 3.7 * s_a)
    eD_zero = float(s_zero.abs().max())
    print('    additivity %.3e | homogeneity %.3e | zero direction gives max|s| = %.3e'
          % (eD_add, eD_scale, eD_zero))
    verdict('D_linearity', eD_add < 1e-12 and eD_scale < 1e-12 and eD_zero == 0.0,
            {'add': eD_add, 'scale': eD_scale, 'zero': eD_zero})

    # ---- E. across the scheduling range, several points and displacements ---------------
    print('\n[E] robustness: 5 point sets x 3 expansion points, worst relative tangent error')
    worstE = 0.0
    for si in range(5):
        zz = sample_point(100 + si)
        for j, fscale in enumerate((0.0, 0.05, 0.15)):
            gg = torch.Generator().manual_seed(200 + si * 10 + j)
            fr = torch.randn(10, generator=gg, dtype=F64) * fscale
            dr = torch.randn(10, generator=gg, dtype=F64)
            _, s_h = blk.transition_and_tangent_from_free(fr, dr, zz)
            _, s_r = torch.func.jvp(lambda v: blk.transition_from_free(v, zz), (fr,), (dr,))
            worstE = max(worstE, rel(s_h, s_r))
    print('    worst over 15 combinations: %.3e' % worstE)
    verdict('E_robustness', worstE < 1e-12, worstE)

    # ---- F. float32, which is what the rollout runs, plus timing ------------------------
    print('\n[F] float32 agreement and cost (reported, not gated)')
    blk32 = make_block().to(torch.float32)
    z32, f32, d32 = z.float(), free.float(), dfree.float()
    _, s32 = blk32.transition_and_tangent_from_free(f32, d32, z32)
    e32 = rel(s32.double(), sp)
    print('    float32 hand tangent vs float64 reference: %.3e' % e32)
    m32, dm32 = blk32.mats_and_tangent_from_free(f32, d32)
    x32, u32 = z32[:, :6, :], z32[:, 6:, :]
    # The NON-JOINT block: identical physics, but the M(Y) structure comes from fixed buffers
    # instead of being rebuilt from trainable parameters every timestep. This is what a
    # joint_estimation=False run executes, and the difference is what joint estimation costs.
    from model_augmentation.fit_systems.blocks import Gantry_State_Block   # noqa: E402
    blk_fix = Gantry_State_Block(
        Y_op=None, std_x=np.ones((6, 1)), std_u=np.ones((3, 1)),
        x_mean=np.zeros((6, 1)), u_mean=np.zeros((3, 1)),
        Ts=TS, up_sample=UP_SAMPLE).to(torch.float32)

    def bench(fn, n=50):
        fn(); t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n

    t_fix = bench(lambda: blk_fix.nonlinear_function(z32))                  # no joint estimation
    t_joint = bench(lambda: blk32.nonlinear_function(z32))                  # + matrix rebuild
    t_rk4 = bench(lambda: blk32._rk4(x32, u32, m32))                        # RK4 alone
    t_tan = bench(lambda: blk32._rk4_with_tangent(x32, torch.zeros_like(x32), u32, m32, dm32))
    t_jvp = bench(lambda: torch.func.jvp(lambda v: blk32.transition_from_free(v, z32),
                                         (f32,), (d32,)))
    print('    step, no joint estimation (fixed buffers)   %.4f ms' % (t_fix * 1e3))
    print('    step, joint estimation (rebuilds M(Y))      %.4f ms   %.2fx  <- joint costs this'
          % (t_joint * 1e3, t_joint / t_fix))
    print('    RK4 alone, matrices given                   %.4f ms' % (t_rk4 * 1e3))
    print('    RK4 + hand tangent, fused                   %.4f ms   %.2fx of RK4 alone'
          % (t_tan * 1e3, t_tan / t_rk4))
    print('    RK4 + torch.func.jvp                        %.4f ms   %.2fx of RK4 alone'
          % (t_jvp * 1e3, t_jvp / t_rk4))
    # What a rollout timestep actually costs with OBC on: the block's own forward at the current
    # parameters, PLUS the correction at the frozen expansion point. The matrix tangent is not in
    # here: it is computed once per objective, not per step.
    step_off, step_on = t_joint, t_joint + t_tan
    step_on_jvp = t_joint + t_jvp
    print('    ROLLOUT STEP  OBC off %.4f ms | OBC on (hand) %.4f ms  %.2fx | OBC on (jvp) '
          '%.4f ms  %.2fx' % (step_off * 1e3, step_on * 1e3, step_on / step_off,
                              step_on_jvp * 1e3, step_on_jvp / step_off))
    RESULTS['F_float32_and_cost'] = {
        'float32_vs_float64': e32, 'step_nojoint_ms': t_fix * 1e3, 'step_joint_ms': t_joint * 1e3,
        'joint_ratio': t_joint / t_fix, 'rk4_ms': t_rk4 * 1e3, 'tangent_ms': t_tan * 1e3,
        'jvp_ms': t_jvp * 1e3, 'tangent_ratio_of_rk4': t_tan / t_rk4,
        'jvp_ratio_of_rk4': t_jvp / t_rk4, 'rollout_step_obc_ratio': step_on / step_off,
        'rollout_step_obc_ratio_jvp': step_on_jvp / step_off}

    # ---- G. the frozen-Y branch must refuse rather than guess ---------------------------
    print('\n[G] frozen Y_op refuses instead of returning an underived tangent')
    blk_frozen = Reduced_Gantry_State_Block(
        params_init=torch.stack([getattr(gss, n) for n in
                                 Parameterized_Gantry_State_Block.PARAM_NAMES]).to(F64),
        Y_op=0.1, std_x=np.ones((6, 1)), std_u=np.ones((3, 1)),
        x_mean=np.zeros((6, 1)), u_mean=np.zeros((3, 1)),
        Ts=TS, up_sample=UP_SAMPLE, RMSE_baseline=0.01).to(F64)
    try:
        blk_frozen._deriv_tangent_with(x, sx0, u, mats, dmats)
        refused = False
    except NotImplementedError as e:
        refused = True
        print('    raised: %s' % str(e)[:96])
    verdict('G_frozen_refuses', refused)

    print('\n==== summary ====')
    for k, v in RESULTS.items():
        if isinstance(v, dict) and 'met' in v:
            print('  %-22s %s' % (k, 'PASS' if v['met'] else 'FAIL'))
    allok = all(v['met'] for v in RESULTS.values() if isinstance(v, dict) and 'met' in v)
    print('\n%s' % ('ALL GATES PASS: the hand-written tangent reproduces torch.func.jvp and a '
                    'finite difference.' if allok else 'FAILURES ABOVE; leg A localises the term.'))
    with open(os.path.join(RESULTS_DIR, 'test_tangent.json'), 'w') as f:
        json.dump(RESULTS, f, indent=2, default=str)
    return 0 if allok else 1


if __name__ == '__main__':
    sys.exit(main())
