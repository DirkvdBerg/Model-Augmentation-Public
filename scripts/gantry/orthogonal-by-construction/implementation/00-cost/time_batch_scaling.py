"""Stage 0d: is the Jacobian overhead per-op or per-element? A batch sweep decides.

__project_origin__ = "added"

Stage 0c found a SINGLE-tangent jvp at 8.5x a plain block step and the full ten-column Jacobian at
14.3x. One tangent costing 60 percent of ten is only possible if a fixed per-call cost dominates
both, and the block makes that plausible: `nonlinear_function` rebuilds the whole `M(Y)` rational
structure from the parameters on every call, which is some fifty tiny tensor operations on 3x3
blocks, and then runs `up_sample * 4 = 8` `deriv` evaluations of a handful of small matmuls each.
Under a `torch.func` transform every one of those operations is wrapped.

If the cost is per-OPERATION then the ratio falls as the batch grows, because the arithmetic grows
and the wrapper count does not, and the production ratio is better than the batch-512 number. If
the cost is per-ELEMENT the ratio is flat and 8.5x is what the design pays.

This distinction decides whether the analytic forward-sensitivity route (candidate (b)) is worth
building: (b) removes wrapper overhead and adds arithmetic, so it only pays if the overhead is
per-operation AND the production batch is small.
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
BATCHES = (64, 256, 512, 2048)


def timeit(fn, n_rep, n_warm=3):
    for _ in range(n_warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_rep):
        fn()
    return (time.perf_counter() - t0) / n_rep


def main():
    torch.manual_seed(0)
    sys.stdout = C.Tee(os.path.join(HERE, 'results_batch_scaling.log'))

    theta0, Lb = C.nominal_raw()
    vartheta0 = C.combos_from_raw(theta0, Lb)
    blk = C.make_raw_block()
    lp0 = blk.log_params.detach().clone()
    m_mask = torch.arange(10) == C.M_DIFF_IX
    v0 = torch.zeros(10, dtype=C.F64)
    a10 = torch.linspace(-1.0, 1.0, 10, dtype=C.F64)

    print('=' * 78)
    print('STAGE 0d -- per-operation or per-element? batch scaling of the Jacobian cost')
    print('=' * 78)
    print(f'  float64, CPU, torch {torch.__version__}, threads {torch.get_num_threads()}')
    print(f'  production batch is 256 (config.py batch_size)')
    print(f'\n  {"batch":>6}  {"step [s]":>10}  {"jvp 1 dir":>10}  {"ratio":>7}  '
          f'{"jacfwd 10":>10}  {"ratio":>7}  {"fwd+bwd":>10}  {"jvp f+b":>10}  {"ratio":>7}')

    rows = []
    for B in BATCHES:
        x, u = C.design_points(B, seed=0)
        z = C.zu(x, u)

        def red_to_lp(v_free):
            v = torch.where(m_mask, vartheta0 * (1.0 + v_free), vartheta0 * torch.exp(v_free))
            return torch.log(C.gauge_section(v, theta0, Lb) / blk.params_init)

        def f_batched(v_free):
            return functional_call(blk, {'log_params': red_to_lp(v_free)}, (z,)).reshape(-1)

        with torch.no_grad():
            t_step = timeit(lambda: functional_call(blk, {'log_params': lp0}, (z,)), 20)
            blk._cur = None
            t_dir = timeit(lambda: jvp(f_batched, (v0,), (a10,)), 10)
            blk._cur = None
            t_jac = timeit(lambda: jacfwd(f_batched)(v0), 5)
            blk._cur = None

        def fwd_bwd_plain():
            lp = lp0.clone().requires_grad_(True)
            functional_call(blk, {'log_params': lp}, (z,)).pow(2).sum().backward()
        t_fb = timeit(fwd_bwd_plain, 10)
        blk._cur = None

        a_var = a10.clone().requires_grad_(True)

        def fwd_bwd_jvp():
            _, out = jvp(f_batched, (v0,), (a_var,))
            out.pow(2).sum().backward()
            a_var.grad = None
        t_fbj = timeit(fwd_bwd_jvp, 5)
        blk._cur = None

        rows.append((B, t_step, t_dir, t_jac, t_fb, t_fbj))
        print(f'  {B:>6}  {t_step:>10.6f}  {t_dir:>10.6f}  {t_dir/t_step:>6.2f}x  '
              f'{t_jac:>10.6f}  {t_jac/t_step:>6.2f}x  {t_fb:>10.6f}  {t_fbj:>10.6f}  '
              f'{t_fbj/t_fb:>6.2f}x')

    print('\n  --- reading ---')
    r_lo, r_hi = rows[0][2] / rows[0][1], rows[-1][2] / rows[-1][1]
    print(f'  jvp/step ratio: {r_lo:.2f}x at batch {rows[0][0]} -> {r_hi:.2f}x at '
          f'batch {rows[-1][0]}')
    if r_hi < 0.7 * r_lo:
        print('  the ratio FALLS with batch: the cost is per-OPERATION (transform wrapper and '
              'the per-call M(Y) rebuild), not per-element. The production ratio is the one at '
              'the production batch, and an analytic implementation would attack exactly this '
              'overhead.')
    elif r_hi > 1.3 * r_lo:
        print('  the ratio RISES with batch: the cost is per-ELEMENT and grows faster than the '
              'plain step. Forward mode over the batched step is the wrong shape.')
    else:
        print('  the ratio is FLAT in batch: the cost is proportional to the work, so the '
              'overhead is not a fixed per-call charge and an analytic implementation would '
              'save arithmetic rather than dispatch.')
    prod = [r for r in rows if r[0] == 256]
    if prod:
        B, t_step, t_dir, t_jac, t_fb, t_fbj = prod[0]
        print(f'\n  AT THE PRODUCTION BATCH 256, which is the number the design is judged on:')
        print(f'    forward only:        one jvp is {t_dir/t_step:.2f}x a plain step')
        print(f'    forward + backward:  {t_fbj/t_fb:.2f}x, i.e. a training step costs that '
              f'much more with the subtraction than without')
        print(f'    one nf = {C.NF} window: {C.NF*t_fb:.2f} s -> {C.NF*t_fbj:.2f} s')
    print('=' * 78)
    sys.stdout.flush()


if __name__ == '__main__':
    main()
