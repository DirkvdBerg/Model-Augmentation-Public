"""Stage 0: what does a per-timestep parameter Jacobian cost?

__project_origin__ = "added"

This is the feasibility question for the whole OBC design (handoff Sect. 8). The construction
needs the protected basis `[J | c]` at EVERY rollout point, not only on a reference set, because
a model that subtracts on a reference set and rolls out with the raw learned write is not the
orthogonal model. So the per-timestep cost of the parameter Jacobian decides between:

  (a) recompute `J` inside the rollout with `vmap(jacfwd(step))`   -- affordable if under ~2x
  (b) build an analytic `J` from `M^{-1}` and constant derivative matrices -- required if ~10x

Timed, at batch 512 in float64 on the CPU, all at the pipeline's settings:
  (i)   a plain block step
  (ii)  `vmap(jacfwd(step))` over the TEN reduced combination directions
  (iii) `vmap(jacfwd(step))` over the FOURTEEN raw directions

(ii) is measured by composing the declared gauge section `R^10 -> R^14` with the existing raw
block, which is the same tangent count as the reduced block of stage 1 will have and therefore
the same operation count; the section is an exactly-satisfied reparameterisation
(`combos(section(v)) == v`, asserted here), so this is a measurement of the ten-direction cost
and not a proxy for it.

Reported: the per-call ratios and the extrapolation to one `nf = 400` window.
"""

__project_origin__ = "added"

import os
import sys
import time

import torch
from torch.func import functional_call, jacfwd, vmap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C                                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BATCH = 512
# HEURISTIC: timing repeat counts. Chosen so each timed quantity accumulates at least a few
# hundred ms of wall on this CPU, which puts the perf_counter resolution well below the noise.
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
    sys.stdout = C.Tee(os.path.join(HERE, 'results.log'))

    theta0, Lb = C.nominal_raw()
    vartheta0 = C.combos_from_raw(theta0, Lb)
    blk = C.make_raw_block()
    lp0 = blk.log_params.detach().clone()

    x, u = C.design_points(BATCH, seed=0)
    z = C.zu(x, u)

    print('=' * 78)
    print('STAGE 0 -- per-timestep parameter Jacobian cost')
    print('=' * 78)
    print(f'  settings: Ts = {C.TS:.6e} s, up_sample = {C.UP_SAMPLE}, Y_op = None (LPV), '
          f'batch = {BATCH}, float64, CPU')
    print(f'  torch {torch.__version__}, threads {torch.get_num_threads()}')
    print(f'  theta_bar (raw 14)  = {theta0.numpy()}')
    print(f'  vartheta_bar (10)   = {vartheta0.numpy()}')

    # --- the gauge section, asserted exact before it is used for anything -------------------
    err = (C.combos_from_raw(C.gauge_section(vartheta0, theta0, Lb), Lb) - vartheta0).abs().max()
    print(f'\n  gauge section round-trip  max |combos(section(v)) - v| = {err.item():.3e}')
    assert err.item() == 0.0, 'gauge section is not a section of the combination map'

    # --- the three timed callables ----------------------------------------------------------
    def step_raw(lp, z_):
        return functional_call(blk, {'log_params': lp}, (z_,))

    def f_single_raw(lp, z1):                      # z1 (nx+nu, 1) -> (nx,)
        return step_raw(lp, z1.unsqueeze(0)).reshape(C.N_P)

    def f_single_red(v_free, z1):
        """One step as a function of the TEN reduced free coordinates at `vartheta_bar`.

        Nine combinations in log, `m_diff` in a relative linear coordinate (it is signed, so a
        log coordinate does not exist for it). Free zero gives `vartheta_bar` exactly.
        """
        v = torch.where(
            torch.arange(10) == C.M_DIFF_IX,
            vartheta0 * (1.0 + v_free),            # linear, relative: signed m_diff
            vartheta0 * torch.exp(v_free),         # log
        )
        raw = C.gauge_section(v, theta0, Lb)
        lp = torch.log(raw / blk.params_init)
        return step_raw(lp, z1.unsqueeze(0)).reshape(C.N_P)

    jac_raw = vmap(jacfwd(f_single_raw, argnums=0), in_dims=(None, 0))
    jac_red = vmap(jacfwd(f_single_red, argnums=0), in_dims=(None, 0))
    v_free0 = torch.zeros(10, dtype=C.F64)

    with torch.no_grad():
        # A value check before any timing: the reduced path at free = 0 must reproduce the raw
        # step exactly, or the thing being timed is not the thing being built.
        x_raw = step_raw(lp0, z).reshape(BATCH, C.N_P)
        x_red = vmap(f_single_red, in_dims=(None, 0))(v_free0, z)
        blk._cur = None
        dx = (x_raw - x_red).abs().max().item()
        scale = (x_raw - x[:, :C.N_P]).abs().max().item()
        print(f'  reduced path at free = 0 vs raw step: max |diff| = {dx:.3e} '
              f'(state increment scale {scale:.3e})')

        t_step = timeit(lambda: step_raw(lp0, z), N_REP_STEP)
        blk._cur = None
        t_j10 = timeit(lambda: jac_red(v_free0, z), N_REP_JAC)
        blk._cur = None
        t_j14 = timeit(lambda: jac_raw(lp0, z), N_REP_JAC)
        blk._cur = None

    print('\n  --- timings, seconds per call at batch %d ---' % BATCH)
    print(f'  (i)   plain block step                     {t_step:.6f}')
    print(f'  (ii)  vmap(jacfwd) over 10 reduced dirs    {t_j10:.6f}   ratio {t_j10/t_step:6.2f}x')
    print(f'  (iii) vmap(jacfwd) over 14 raw dirs        {t_j14:.6f}   ratio {t_j14/t_step:6.2f}x')

    # --- extrapolation to one training window ----------------------------------------------
    # One window is `nf` sequential block steps. The batch dimension is already 512, i.e. wider
    # than the production batch of 256, so this is an upper bound per window at that batch.
    print(f'\n  --- extrapolation to one nf = {C.NF} window at batch {BATCH} ---')
    print(f'  baseline rollout, steps only                {C.NF * t_step:8.3f} s')
    print(f'  + 10-direction Jacobian at every step       {C.NF * (t_step + t_j10):8.3f} s'
          f'   ({1 + t_j10/t_step:5.2f}x)')
    print(f'  + 14-direction Jacobian at every step       {C.NF * (t_step + t_j14):8.3f} s'
          f'   ({1 + t_j14/t_step:5.2f}x)')

    r = t_j10 / t_step
    print('\n  --- verdict ---')
    if r <= 2.0:
        print(f'  ratio {r:.2f}x <= 2x: candidate (a), recompute J inside the rollout with '
              f'jacfwd. No analytic Jacobian needed.')
    elif r <= 4.0:
        print(f'  ratio {r:.2f}x is between 2x and 4x: candidate (a) is affordable but not free. '
              f'Proceed with (a); record the cost.')
    else:
        print(f'  ratio {r:.2f}x > 4x: candidate (a) is expensive. Consider the analytic '
              f'Jacobian (b) before building the projection, OR restrict the construction '
              f'evaluations to a strided subset of the rollout.')
    print('=' * 78)
    sys.stdout.flush()


if __name__ == '__main__':
    main()
