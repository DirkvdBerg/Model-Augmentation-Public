"""Stage 0b: the 28x of `vmap(jacfwd(...))` is overhead, not tangent count. How cheap can it get?

__project_origin__ = "added"

Stage 0 measured `vmap(jacfwd(f_single))` at 28.2x a plain block step for ten parameter
directions and 26.2x for fourteen. Ten directions costing the SAME as fourteen is the diagnostic:
the cost cannot be proportional to the tangent count, so it is dominated by the per-sample
`vmap` composition, which evaluates the block at batch one and loses every BLAS economy the
batched step has.

`gantry_orth_matrix` uses that composition because it builds the regressor ONCE on a fixed point
set, where 28x of a cheap thing is irrelevant. Inside a rollout it is not. This script measures
the alternatives that keep the batch intact:

  (iv) `jacfwd` over the BATCHED step, 10 tangents, no `vmap`. Forward mode over the whole batch
       at once: the expected cost is roughly the tangent count, not 28x.
  (v)  ten explicit `torch.func.jvp` calls on the batched step, one per direction. Same
       arithmetic, no `jacfwd` machinery, and it is the form an analytic forward-sensitivity
       implementation would take.

Every variant is checked against the stage-0 Jacobian for exact agreement before it is timed: a
faster way of computing a different matrix is not a result.
"""

__project_origin__ = "added"

import os
import sys
import time

import torch
from torch.func import functional_call, jacfwd, jvp, vmap

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
    sys.stdout = C.Tee(os.path.join(HERE, 'results_variants.log'))

    theta0, Lb = C.nominal_raw()
    vartheta0 = C.combos_from_raw(theta0, Lb)
    blk = C.make_raw_block()
    lp0 = blk.log_params.detach().clone()
    x, u = C.design_points(BATCH, seed=0)
    z = C.zu(x, u)
    m_mask = torch.arange(10) == C.M_DIFF_IX

    print('=' * 78)
    print('STAGE 0b -- how cheap can the per-timestep reduced Jacobian get?')
    print('=' * 78)
    print(f'  batch {BATCH}, float64, CPU, torch {torch.__version__}, '
          f'threads {torch.get_num_threads()}')

    def red_to_lp(v_free):
        v = torch.where(m_mask, vartheta0 * (1.0 + v_free), vartheta0 * torch.exp(v_free))
        return torch.log(C.gauge_section(v, theta0, Lb) / blk.params_init)

    def f_batched(v_free):                          # (10,) -> (B*n_p,)
        return functional_call(blk, {'log_params': red_to_lp(v_free)}, (z,)).reshape(-1)

    def f_single(v_free, z1):                       # (10,), (nx+nu,1) -> (n_p,)
        return functional_call(blk, {'log_params': red_to_lp(v_free)},
                               (z1.unsqueeze(0),)).reshape(C.N_P)

    v0 = torch.zeros(10, dtype=C.F64)
    jac_vmap = vmap(jacfwd(f_single, argnums=0), in_dims=(None, 0))

    with torch.no_grad():
        # --- reference and value checks -----------------------------------------------------
        J_vmap = jac_vmap(v0, z).reshape(BATCH * C.N_P, 10)
        blk._cur = None
        J_batched = jacfwd(f_batched)(v0)
        blk._cur = None
        cols = []
        for i in range(10):
            e = torch.zeros(10, dtype=C.F64)
            e[i] = 1.0
            cols.append(jvp(f_batched, (v0,), (e,))[1])
            blk._cur = None
        J_jvp = torch.stack(cols, dim=1)

        n = torch.linalg.matrix_norm(J_vmap)
        print(f'\n  reference J from vmap(jacfwd(f_single)):  shape {tuple(J_vmap.shape)}, '
              f'Frobenius norm {n.item():.6e}')
        # Agreement is judged RELATIVE to the largest entry of the reference Jacobian, not by
        # exact equality: the batched and the per-sample variant reduce the same sums in a
        # different order, so they differ at float64 round-off by construction. The tolerance is
        # the same float64 round-off criterion the other stages use.
        scale = J_vmap.abs().max().item()
        d_b = (J_batched - J_vmap).abs().max().item() / scale
        d_j = (J_jvp - J_vmap).abs().max().item() / scale
        print(f'  largest reference entry {scale:.3e}')
        print(f'  (iv) jacfwd(batched)   max rel diff vs reference = {d_b:.3e}')
        print(f'  (v)  10 x jvp(batched) max rel diff vs reference = {d_j:.3e}')
        assert max(d_b, d_j) < 1e-12, 'a variant computes a DIFFERENT matrix; not a speed-up'

        # --- timings ------------------------------------------------------------------------
        t_step = timeit(lambda: functional_call(blk, {'log_params': lp0}, (z,)), N_REP_STEP)
        blk._cur = None
        t_vmap = timeit(lambda: jac_vmap(v0, z), N_REP_JAC)
        blk._cur = None
        t_batched = timeit(lambda: jacfwd(f_batched)(v0), N_REP_JAC)
        blk._cur = None

        def ten_jvps():
            for i in range(10):
                e = torch.zeros(10, dtype=C.F64)
                e[i] = 1.0
                jvp(f_batched, (v0,), (e,))
        t_jvp = timeit(ten_jvps, N_REP_JAC)
        blk._cur = None

    print(f'\n  --- seconds per call at batch {BATCH} ---')
    print(f'  plain block step                          {t_step:.6f}')
    print(f'  (ii)  vmap(jacfwd(f_single)), 10 dirs     {t_vmap:.6f}   '
          f'{t_vmap/t_step:7.2f}x')
    print(f'  (iv)  jacfwd(f_batched),      10 dirs     {t_batched:.6f}   '
          f'{t_batched/t_step:7.2f}x')
    print(f'  (v)   10 x jvp(f_batched)                 {t_jvp:.6f}   '
          f'{t_jvp/t_step:7.2f}x')

    best = min(t_batched, t_jvp)
    best_name = '(iv) jacfwd(batched)' if t_batched <= t_jvp else '(v) 10 x jvp'
    r = best / t_step
    print(f'\n  cheapest exact variant: {best_name} at {r:.2f}x a plain step')

    # --- cost decomposition -----------------------------------------------------------------
    # A strided reference set does NOT make the rollout orthogonal at the skipped steps; it makes
    # the CONSTRUCTION EVALUATIONS a strided subset, which is what the handoff candidate (a)
    # already is. The subtraction itself still needs the basis at every step, so stride reduces
    # the cost of FITTING the coefficient, never of APPLYING it. Kept separate here so the two
    # are not conflated.
    print(f'\n  --- cost decomposition over one nf = {C.NF} window, batch {BATCH} ---')
    print(f'  rollout, steps only                          {C.NF * t_step:8.3f} s')
    print(f'  APPLY: basis at every step (needed always)   {C.NF * best:8.3f} s'
          f'   -> total {C.NF * (t_step + best):8.3f} s  ({1 + r:5.2f}x)')
    for stride in (1, 4, 10, 40):
        m = max(1, C.NF // stride)
        print(f'  FIT:   coefficient on stride {stride:3d} ({m:4d} pts)      '
              f'{m * best:8.3f} s')

    print('\n  --- verdict ---')
    if r <= 2.0:
        print(f'  {r:.2f}x at or below 2x: candidate (a) as written. Recompute the basis inside '
              f'the rollout with the batched variant.')
    elif r <= 12.0:
        print(f'  {r:.2f}x: candidate (a) is affordable with the BATCHED variant and is NOT '
              f'affordable with the per-sample vmap composition ({t_vmap/t_step:.1f}x). '
              f'Proceed with (a) using {best_name}; the analytic Jacobian (b) becomes an '
              f'optimisation, not a prerequisite.')
    else:
        print(f'  {r:.2f}x above 12x: even the batched variant is expensive. The analytic '
              f'Jacobian (b) is a prerequisite.')
    print('=' * 78)
    sys.stdout.flush()


if __name__ == '__main__':
    main()
