"""Is the small latent-initialisation penalty gradient correct, and does it matter?

WHY THIS EXISTS. The ladder reports `max|g_eta_a| = 8.8e-11` against `max|g_eta_d| = 4.7e-06`.
I first presented that ratio as something to be concerned about. Review was right that it says
nothing on its own: `eta_a` (the encoder's latent-initialisation branch) and `eta_d` (the ANN)
are DIFFERENTLY PARAMETERISED, so a ratio between their raw gradient magnitudes measures the
parameterisation at least as much as the geometry. A small number could be any of

    - correct, and simply the scale of those parameters
    - a weak influence of `a_0` on the scored stack
    - a small projected residual, which is the state the method is trying to reach
    - cancellation inside the projection
    - an actual defect in the routing

and a magnitude cannot separate them. So this probe asks the four questions that can.

  1. FINITE DIFFERENCE OF THE PENALTY along `eta_a`, against the analytic directional
     derivative. This is the one that can find a wrong gradient, and it differentiates the
     PENALTY specifically, not the prediction: V2's checks are prediction derivatives and do not
     cover the projection.
  2. RELATIVE to the base loss in the SAME group. `|g_eta_a^penalty| / |g_eta_a^base|` is
     dimensionless and comparable, where the cross-group ratio is not.
  3. The RESULTING OPTIMIZER UPDATE difference, which is what actually moves the model, and
     under Adam is scale-free by construction.
  4. `D_{a_0} p` against `D_{x_0} p`, to see whether a small number reflects the latent initial
     state simply having less influence on the scored output than the physical one.

NOT AN EFFICACY MEASUREMENT. It says whether the gradient is right and whether it is negligible
in the update, not whether the penalty helps.
"""
__project_origin__ = "added"

import numpy as np
import torch


def _flat(ps):
    return torch.cat([p.reshape(-1) for p in ps])


def _flat_grad(ps):
    return torch.cat([(torch.zeros_like(p).reshape(-1) if p.grad is None else p.grad.reshape(-1))
                      for p in ps])


def _add_flat(ps, v):
    i = 0
    for p in ps:
        n = p.numel()
        p.data.add_(v[i:i + n].reshape(p.shape).to(p.dtype))
        i += n


def latent_init_probe(ad, geom, beta=1.0, seed=20260908, steps=(1e-3, 1e-4, 1e-5), lr=1e-4):
    """Run the four comparisons. `ad` is a GantryTrajectoryAdapter, `geom` the frozen geometry."""
    enc = ad.enc
    eta_a = list(enc.latent_parameters())
    eta_d = list(ad.ann.parameters())
    lam = [ad.phys.log_params]
    xip = enc.physical_parameters()
    out = {}

    def penalty_value():
        """V at the current parameters, with lambda and the physical initial state constant."""
        xp, xa = ad.encode()
        xp = xp.detach()
        with torch.no_grad():
            p_off = ad.scored_stack(mode='off', x0=ad.x0_from(xp, torch.zeros_like(xa)))
        p_full = ad.scored_stack(mode='full', x0=ad.x0_from(xp, xa))
        return geom.penalty_mse(p_full - p_off, beta)

    # ---- 1. analytic vs finite difference of the PENALTY along eta_a -------------------
    for p in eta_a + eta_d + lam + xip:
        p.grad = None
    saved_rg = [(p, p.requires_grad) for p in lam + xip]
    for p, _ in saved_rg:
        p.requires_grad_(False)
    try:
        V0 = penalty_value()
        V0.backward()
        g_a = _flat_grad(eta_a).clone()
        g_d = _flat_grad(eta_d).clone()

        gen = torch.Generator().manual_seed(seed)
        v = torch.randn(g_a.numel(), generator=gen, dtype=torch.float64)
        v = (v / v.norm()).to(g_a.dtype)
        analytic = float(g_a @ v)

        fd = {}
        saved_vals = [p.detach().clone() for p in eta_a]
        with torch.no_grad():
            for h in steps:
                _add_flat(eta_a, h * v)
                vp = float(penalty_value())
                for p, s in zip(eta_a, saved_vals):
                    p.data.copy_(s)
                _add_flat(eta_a, -h * v)
                vm = float(penalty_value())
                for p, s in zip(eta_a, saved_vals):
                    p.data.copy_(s)
                d = (vp - vm) / (2 * h)
                fd[f'{h:g}'] = dict(fd=d, rel_err=abs(d - analytic) / max(abs(analytic), 1e-300))
        out['penalty_directional_derivative'] = dict(
            analytic=analytic, by_step=fd,
            best_rel_err=min(x['rel_err'] for x in fd.values()),
            note='differentiates the PENALTY along eta_a; V2 only covers prediction derivatives')
    finally:
        for p, was in saved_rg:
            p.requires_grad_(was)

    out['raw_magnitudes'] = dict(
        g_eta_a_max=float(g_a.abs().max()), g_eta_d_max=float(g_d.abs().max()),
        g_eta_a_norm=float(g_a.norm()), g_eta_d_norm=float(g_d.norm()),
        n_eta_a=int(g_a.numel()), n_eta_d=int(g_d.numel()),
        note='NOT comparable across groups: differently parameterised. Kept for the record only')

    # ---- 2. penalty gradient RELATIVE to the base gradient, within each group ----------
    for p in eta_a + eta_d:
        p.grad = None
    ad.production_loss().backward()
    b_a, b_d = _flat_grad(eta_a).clone(), _flat_grad(eta_d).clone()
    out['relative_to_base_loss'] = dict(
        eta_a=float(g_a.norm() / max(float(b_a.norm()), 1e-300)),
        eta_d=float(g_d.norm() / max(float(b_d.norm()), 1e-300)),
        base_eta_a_norm=float(b_a.norm()), base_eta_d_norm=float(b_d.norm()),
        note='dimensionless and comparable ACROSS groups, unlike the raw magnitudes')

    # ---- 3. the update the optimizer actually takes ------------------------------------
    def adam_step_norm(params, grads):
        """One Adam step from a cold state, which is scale-free by construction."""
        clones = [torch.nn.Parameter(p.detach().clone()) for p in params]
        opt = torch.optim.Adam(clones, lr=lr)
        i = 0
        for c in clones:
            n = c.numel()
            c.grad = grads[i:i + n].reshape(c.shape).clone()
            i += n
        before = _flat([c for c in clones]).clone()
        opt.step()
        return float((_flat(clones) - before).norm())

    out['optimizer_update'] = dict(
        eta_a_base_only=adam_step_norm(eta_a, b_a),
        eta_a_base_plus_penalty=adam_step_norm(eta_a, b_a + g_a),
        eta_d_base_only=adam_step_norm(eta_d, b_d),
        eta_d_base_plus_penalty=adam_step_norm(eta_d, b_d + g_d),
        lr=lr,
        note=('Adam normalises by the gradient scale, so a small RAW gradient can still move a '
              'parameter as far as a large one. This is the number that says whether the '
              'penalty is negligible IN THE UPDATE, which the raw magnitude does not.'))
    ua, ub = out['optimizer_update']['eta_a_base_plus_penalty'], \
        out['optimizer_update']['eta_a_base_only']
    out['optimizer_update']['eta_a_relative_change'] = abs(ua - ub) / max(ub, 1e-300)

    # ---- 4. influence of a_0 versus x_0 on the scored stack ---------------------------
    G_x = ad.initial_state_jacobians()
    G_a = ad.latent_init_jacobians()
    out['initial_state_influence'] = dict(
        sv_Gamma_x_first_window=np.linalg.svd(G_x[0], compute_uv=False).tolist(),
        sv_Gamma_a_first_window=np.linalg.svd(G_a[0], compute_uv=False).tolist(),
        ratio_top_singular=float(np.linalg.svd(G_a[0], compute_uv=False)[0] /
                                 max(np.linalg.svd(G_x[0], compute_uv=False)[0], 1e-300)),
        note=('how much the LATENT initial state moves the scored output, against the physical '
              'one. A small penalty gradient on eta_a is expected if a_0 simply has less '
              'influence, and that is a property of the model, not a defect'))

    for p in eta_a + eta_d:
        p.grad = None
    return out
