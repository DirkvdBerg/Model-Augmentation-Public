"""V6 execution safety: functional AD hygiene, checkpoint replay, no-op parity, hook ordering.

Split from `checks.py` only for length. These are the checks that are about HOW the computation
runs rather than about what it computes, and each one exists because the corresponding failure
is silent: a deleted registry entry, a checkpoint segment replayed under the wrong intervention,
a gate that is not quite a no-op, a penalty gradient wiped by the closure's own zero_grad.
"""
__project_origin__ = "added"

import torch

from .adapter import ann_gate


def check_functional_call_is_clean(L, env, ad):
    """The functional parameter substitution must leave the module exactly as it found it.

    Specification Sect. 3.2 and 9.2: "Never remove log_params from the Parameter registry to
    accommodate forward-mode duals. Use an explicit pure parameter evaluation or tested
    functional substitution, without a stale `_cur` cache." Both halves are checked: the
    registry, state dict and values; and the physical block's per-forward matrix cache.
    """
    sd_before = {k: v.detach().clone() for k, v in env.fit_sys.hfn.state_dict().items()}
    names_before = [n for n, _ in env.fit_sys.hfn.named_parameters()]
    lp_before = ad.phys.log_params.detach().clone()
    cur_before = ad.phys._cur

    e = torch.zeros(ad.phys.log_params.numel(), dtype=ad.phys.log_params.dtype,
                    device=ad.phys.log_params.device)
    e[0] = 1.0
    _ = ad.physical_directional(e)

    names_after = [n for n, _ in env.fit_sys.hfn.named_parameters()]
    sd_after = env.fit_sys.hfn.state_dict()
    same_keys = set(sd_before) == set(sd_after) and names_before == names_after
    worst = max((float((sd_before[k] - sd_after[k]).abs().max())
                 for k in sd_before if sd_before[k].is_floating_point()), default=0.0)
    lp_same = bool(torch.equal(lp_before, ad.phys.log_params.detach()))
    L.add('V6', 'functional_substitution_leaves_registry_intact',
          same_keys and worst == 0.0 and lp_same,
          'after a forward-mode Jacobian call the module keeps every registered parameter and '
          'buffer, with unchanged values',
          'spec Sect. 3.2 and 9.2', 'identical registry and values',
          dict(same_keys=same_keys, max_value_change=worst, log_params_unchanged=lp_same),
          tol=0.0, category='data-computable',
          detail=f'{len(names_after)} parameters, max value change {worst:.3e}',
          establishes='no parameter was deleted from the registry to make forward-mode AD work',
          not_establishes='that the derivative is correct; that is V2')

    with torch.no_grad():
        _ = ad.scored_stack(mode='full')
    rebuilt = ad.phys._cur is not cur_before
    L.add('V6', 'physical_block_cache_is_rebuilt_per_forward', rebuilt,
          'the parameterised physical block rebuilds its M(Y) structure on every forward, so a '
          'functional parameter substitution cannot be served a stale matrix set',
          'blocks.py Parameterized_Gantry_State_Block.nonlinear_function sets self._cur',
          'a new cache object', rebuilt, category='structural',
          detail=f'cache identity changed: {rebuilt}')


def check_gate_checkpoint_replay(L, env, ad):
    """Gradient checkpointing re-runs the forward during backward; the gate must replay right.

    The failure mode is silent and specific. `torch.utils.checkpoint` recomputes a segment in the
    backward pass, at which point the gate buffer holds whatever the CURRENT global mode is. A
    graph-bearing backward taken after the gate scope has exited therefore recomputes under the
    wrong intervention, and the gradient belongs to a different function than the value.
    """
    sim = env.fit_sys.simulator
    if sim is None:
        return L.add('V6', 'gate_checkpoint_replay', False,
                     'requires the closed-loop simulator', 'spec Sect. 4.3', '-', None,
                     skipped=True, detail='no simulator attached')
    saved = sim.checkpoint_chunk
    ps = ad.parameter_groups()['augmentation']
    chunk = max(1, int(ad.w['ufuture'].shape[1]) // 4)
    try:
        sim.checkpoint_chunk = 0
        for p in ps:
            p.grad = None
        ad.scored_stack(mode='full').pow(2).sum().backward()
        g_ref = torch.cat([p.grad.reshape(-1).clone() for p in ps if p.grad is not None])

        sim.checkpoint_chunk = chunk
        for p in ps:
            p.grad = None
        xp, xa = ad.encode()
        with ann_gate(env.ann_block, ad._masks['full']):
            y = ad.simulate(ad.x0_from(xp, xa))
            ad.score(y).pow(2).sum().backward()
        g_in = torch.cat([p.grad.reshape(-1).clone() for p in ps if p.grad is not None])
        r_in = float((g_in - g_ref).norm() / max(float(g_ref.norm()), 1e-300))

        for p in ps:
            p.grad = None
        xp, xa = ad.encode()
        with ann_gate(env.ann_block, ad._masks['full']):
            y2 = ad.simulate(ad.x0_from(xp, xa))
            s2 = ad.score(y2).pow(2).sum()
        with ann_gate(env.ann_block, ad._masks['off']):
            s2.backward()
        g_out = torch.cat([p.grad.reshape(-1).clone() for p in ps if p.grad is not None])
        r_out = float((g_out - g_ref).norm() / max(float(g_ref.norm()), 1e-300))
    finally:
        sim.checkpoint_chunk = saved
        for p in ps:
            p.grad = None

    L.add('V6', 'checkpointed_gradient_matches_uncheckpointed', r_in < 1e-8,
          'with gradient checkpointing on, a backward taken INSIDE the gate scope reproduces the '
          'un-checkpointed gradient',
          'spec Sect. 4.3 "Checkpoint recomputation must replay the original mode"',
          'relative difference at round-off', r_in, tol=1e-8, category='data-computable',
          detail=f'chunk={chunk}, rel {r_in:.3e}',
          establishes='the gate is safe under checkpointing for the supported call pattern',
          not_establishes='safety for a backward taken after the scope exits; see the next row')
    L.add('V6', 'checkpoint_replay_outside_scope_is_detected', r_out > 1e-6,
          'taking the backward AFTER leaving the gate scope, with the global mode changed in '
          'between, gives a measurably different gradient, so the unsupported pattern is '
          'detectable rather than silently wrong',
          'spec Sect. 4.3', 'a large relative difference', r_out, category='data-computable',
          detail=f'relative difference against the reference gradient {r_out:.3e}',
          establishes='the constraint is real and enforced by a check, not by a comment',
          not_establishes='that the supported pattern is the only safe one in every future '
                          'configuration')


def check_feature_disabled_is_a_no_op(L, env, ad):
    """With the gate at its default and no penalty, production behaviour must be unchanged.

    The gate multiplies by exactly 1.0, which is exact in IEEE arithmetic, so this is a
    BIT-IDENTITY claim and is checked as one rather than with a tolerance.
    """
    with torch.no_grad():
        base = ad.production_loss()
        g0 = env.ann_block.out_gate.detach().clone()
        env.ann_block.out_gate = torch.ones_like(g0)
        again = ad.production_loss()
        env.ann_block.out_gate = g0
    same = bool(torch.equal(base, again))
    L.add('V6', 'feature_disabled_behaviour_unchanged', same,
          'the all-ones gate is a bit-identical no-op on the production loss',
          'blocks.py Static_ANN_Block.forward; multiplication by 1.0 is exact in IEEE',
          'bit-identical', float(base), tol=0.0, category='data-computable',
          detail=f'loss {float(base):.12e}',
          establishes='every run predating the gate remains reproducible',
          not_establishes='parity of the compiled path, which is a separate measurement')
    keys = [k for k in env.fit_sys.hfn.state_dict() if 'out_gate' in k]
    L.add('V6', 'gate_absent_from_state_dict', not keys,
          'the gate is a non-persistent buffer, so no checkpoint gains a key and every existing '
          'checkpoint still loads',
          'spec Sect. 4.3 and 9.3', 'absent', keys, category='structural')


def check_accumulation_hook_survives_zero_grad(L, env, ad, geom, beta=1.0):
    """The penalty gradient must be added AFTER the closure's zero_grad, not inside the loss.

    `SSE_Interconnect.fit`'s closure clears gradients and then backwards the loss, so gradients
    prepopulated inside `loss()` are discarded. The specification's order is: zero_grad;
    evaluate and backward J_base; no-grad residual pass; recompute chunks and add eta-only
    gradients; clip; step. This check runs BOTH orders and shows the wrong one loses the penalty.
    """
    ps = ad.parameter_groups()['augmentation']
    lp = ad.phys.log_params

    def add_penalty():
        lp.requires_grad_(False)
        try:
            xp, xa = ad.encode()
            xp = xp.detach()
            with torch.no_grad():
                p_off = ad.scored_stack(mode='off', x0=ad.x0_from(xp, torch.zeros_like(xa)))
            d = ad.scored_stack(mode='full', x0=ad.x0_from(xp, xa)) - p_off
            geom.penalty_mse(d, beta).backward()
        finally:
            lp.requires_grad_(True)

    for p in ps:
        p.grad = None
    add_penalty()
    for p in ps:                       # exactly what the production closure does next
        p.grad = None
    ad.production_loss().backward()
    g_wrong = torch.cat([p.grad.reshape(-1).clone() for p in ps if p.grad is not None])

    for p in ps:
        p.grad = None
    ad.production_loss().backward()
    add_penalty()
    g_right = torch.cat([p.grad.reshape(-1).clone() for p in ps if p.grad is not None])

    diff = float((g_right - g_wrong).norm() / max(float(g_wrong.norm()), 1e-300))
    L.add('V6', 'accumulation_hook_must_follow_zero_grad', diff > 1e-10,
          'adding the penalty gradient before the closure clears gradients loses it entirely; '
          'the two orders give measurably different updates',
          'spec Sect. 8.1; the fit() closure order in interconnect.py',
          'a nonzero relative difference', diff, category='data-computable',
          detail=f'relative difference between the two orders {diff:.3e}',
          establishes='the hook has to run after zero_grad and after the base backward',
          not_establishes='that the production closure has been modified; it has not, and wiring '
                          'the hook into it is the remaining integration step')
    for p in ps:
        p.grad = None
