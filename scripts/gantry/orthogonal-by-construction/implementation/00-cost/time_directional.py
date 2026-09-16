"""Stage 0c: the rollout needs one DIRECTIONAL derivative, not the ten-column Jacobian.

__project_origin__ = "added"

Stages 0 and 0b priced the wrong object, and the handoff's candidate (a) specifies the wrong
object with it ("`Phi(z_k)` recomputed inside the rollout by `jacfwd`"). Read the correction term
of the construction:

    g~(z_k) = g(z_k) - [J(z_k) | c(z_k)] a ,      a in R^11 fixed for the pass

so what the rollout applies is

    Delta(z_k) = J(z_k) a[0:10] + c(z_k) a[10] .

`J(z_k) a[0:10]` is a single Jacobian-vector product in the direction `a[0:10]`: one tangent, not
ten. And `c(z_k)` in the reduced/log coordinate is `f_vartheta_bar(z_k)`, the nominal baseline
transition, which the rollout has ALREADY computed at that step. So the marginal cost of the
subtraction is one jvp, and the ten-column Jacobian is needed only on the construction
evaluations, where it is built once per epoch under `no_grad`.

This is an exact algebraic identity, not an approximation: the same `Delta` to round-off, which
this script asserts against the explicit `J @ a` before timing anything.

Timed here:
  (vii)  one jvp in a fixed direction, batched          -- the APPLY cost per rollout step
  (viii) the same with `a` requiring grad, plus backward -- the training-time cost
and, for the record, the full ten-column Jacobian from stage 0b as the thing avoided.
"""

__project_origin__ = "added"

import os
import sys
import time

import torch
from torch.func import functional_call, jacfwd, jvp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C                                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BATCH = 512
N_WARM, N_REP_STEP, N_REP_JAC = 3, 30, 5


def timeit(fn, n_rep, n_warm=N_WARM):
    for _ in range(n_warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_rep):
        fn()
    return (time.perf_counter() - t0) / n_rep


def main():
    torch.manual_seed(0)
    sys.stdout = C.Tee(os.path.join(HERE, 'results_directional.log'))

    theta0, Lb = C.nominal_raw()
    vartheta0 = C.combos_from_raw(theta0, Lb)
    blk = C.make_raw_block()
    lp0 = blk.log_params.detach().clone()
    x, u = C.design_points(BATCH, seed=0)
    z = C.zu(x, u)
    m_mask = torch.arange(10) == C.M_DIFF_IX

    print('=' * 78)
    print('STAGE 0c -- the rollout needs ONE directional derivative, not the Jacobian')
    print('=' * 78)
    print(f'  batch {BATCH}, float64, CPU, torch {torch.__version__}, '
          f'threads {torch.get_num_threads()}')

    def red_to_lp(v_free):
        v = torch.where(m_mask, vartheta0 * (1.0 + v_free), vartheta0 * torch.exp(v_free))
        return torch.log(C.gauge_section(v, theta0, Lb) / blk.params_init)

    def f_batched(v_free):
        return functional_call(blk, {'log_params': red_to_lp(v_free)}, (z,)).reshape(-1)

    v0 = torch.zeros(10, dtype=C.F64)
    # An arbitrary but fixed coefficient direction, standing in for `a[0:10]`. Its VALUE is
    # irrelevant to both the identity and the cost; only its being a single vector matters.
    a10 = torch.linspace(-1.0, 1.0, 10, dtype=C.F64)

    with torch.no_grad():
        J = jacfwd(f_batched)(v0)
        blk._cur = None
        ref = J @ a10
        _, dir_out = jvp(f_batched, (v0,), (a10,))
        blk._cur = None
        scale = ref.abs().max().item()
        rel = (dir_out - ref).abs().max().item() / scale
        print(f'\n  identity check   max abs (J @ a) = {scale:.6e}')
        print(f'  jvp(f, v0, a) vs J @ a   max rel diff = {rel:.3e}')
        assert rel < 1e-12, 'the directional derivative is not J @ a'

        t_step = timeit(lambda: functional_call(blk, {'log_params': lp0}, (z,)), N_REP_STEP)
        blk._cur = None
        t_jac = timeit(lambda: jacfwd(f_batched)(v0), N_REP_JAC)
        blk._cur = None
        t_dir = timeit(lambda: jvp(f_batched, (v0,), (a10,)), N_REP_STEP)
        blk._cur = None

    # --- (viii) the training-time cost: `a` carries gradient, and backward runs -------------
    # `a` is a function of the ANN parameters, so the tangent of the jvp requires grad and the
    # rollout backwards through it. This is forward-over-reverse; it is what the optimiser pays.
    a_var = a10.clone().requires_grad_(True)

    def fwd_bwd():
        _, out = jvp(f_batched, (v0,), (a_var,))
        out.pow(2).sum().backward()
        a_var.grad = None
    t_fb = timeit(fwd_bwd, N_REP_JAC)
    blk._cur = None

    def fwd_bwd_plain():
        lp = lp0.clone().requires_grad_(True)
        out = functional_call(blk, {'log_params': lp}, (z,))
        out.pow(2).sum().backward()
    t_fb_plain = timeit(fwd_bwd_plain, N_REP_STEP)
    blk._cur = None

    print(f'\n  --- seconds per call at batch {BATCH} ---')
    print(f'  plain block step                            {t_step:.6f}   '
          f'{1.0:7.2f}x')
    print(f'  (iv)  jacfwd, full 10-column J              {t_jac:.6f}   '
          f'{t_jac/t_step:7.2f}x   <- avoided in the rollout')
    print(f'  (vii) jvp, ONE direction a                  {t_dir:.6f}   '
          f'{t_dir/t_step:7.2f}x   <- what the rollout pays')
    print(f'\n  plain step forward + backward               {t_fb_plain:.6f}   '
          f'{1.0:7.2f}x')
    print(f'  (viii) jvp with grad-carrying a, + backward  {t_fb:.6f}   '
          f'{t_fb/t_fb_plain:7.2f}x of plain fwd+bwd')

    r_dir = t_dir / t_step
    print(f'\n  --- cost of one nf = {C.NF} window at batch {BATCH} ---')
    print(f'  rollout, steps only                            {C.NF * t_step:8.3f} s')
    print(f'  + one jvp per step (the subtraction)           {C.NF * (t_step + t_dir):8.3f} s'
          f'   ({1 + r_dir:5.2f}x)')
    print(f'  + full Jacobian per step (what 0b priced)      {C.NF * (t_step + t_jac):8.3f} s'
          f'   ({1 + t_jac/t_step:5.2f}x)')
    print('\n  the construction evaluations, built ONCE per epoch under no_grad:')
    for m in (32, 64, 256, 1024):
        # The reference-set Jacobian is one batched jacfwd over M points; its cost scales with M
        # against the BATCH the timing used, so it is priced per point.
        print(f'    M = {m:5d} points   {m / BATCH * t_jac:8.4f} s')

    print('\n  --- verdict ---')
    if r_dir <= 3.0:
        print(f'  {r_dir:.2f}x: the APPLY cost is one jvp and it is affordable at nf = {C.NF}. '
              f'Candidate (a) proceeds, with the correction that the rollout evaluates a '
              f'DIRECTIONAL derivative and not the Jacobian. The analytic Jacobian (b) is an '
              f'optimisation, not a prerequisite.')
    else:
        print(f'  {r_dir:.2f}x: even the single directional derivative is expensive. Price the '
              f'analytic route (b) before building the projection.')
    print('=' * 78)
    sys.stdout.flush()


if __name__ == '__main__':
    main()
