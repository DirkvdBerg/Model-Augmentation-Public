"""The executable verification ladder against the PRODUCTION gantry predictor.

Stages follow the specification's table in Sect. 12. Every check records: the mathematical
statement, the reference equation, the expected result, the tolerance and why it is that
tolerance, what passing establishes and what it does not. A check that cannot be run records
itself as SKIPPED with the reason, which is not the same thing as a pass.

CATEGORY LABELS (RULES.md Rule 1) are carried per check:
    structural       needs the model only
    data-computable  needs measured data, no truth
    truth-requiring  needs the simulation ground truth   (nothing here is in this category)
"""
__project_origin__ = "added"

import os
import time

import numpy as np
import torch

from model_augmentation.fit_systems.trajectory_orth_projection import (
    TrajectoryProjectionGeometry, ProjectedResidualAccumulator, compressed_nuisance_basis,
    penalty_full_stack, assert_geometry_detached)
from model_augmentation.fit_systems.trajectory_adapter import check_ownership
from .adapter import ann_gate, gate_masks
from .encoder_split import split_encoder, check_split_parity, SplitEncoderInitAug
from .subspace import projector_distance, residual_outside, principal_angles_deg


class Ledger:
    def __init__(self):
        self.rows = []

    def add(self, stage, name, ok, statement, reference, expected, got, tol=None,
            category='structural', establishes='', not_establishes='', detail='',
            skipped=False):
        r = dict(stage=stage, name=name, ok=bool(ok) and not skipped, skipped=bool(skipped),
                 statement=statement, reference=reference, expected=expected,
                 got=_j(got), tol=tol, category=category, establishes=establishes,
                 not_establishes=not_establishes, detail=detail)
        self.rows.append(r)
        tag = 'SKIP' if skipped else ('PASS' if ok else 'FAIL')
        print(f'  [{tag}] {stage} {name}' + (f'   {detail}' if detail else ''))
        return r

    def summary(self):
        p = sum(1 for r in self.rows if r['ok'])
        f = sum(1 for r in self.rows if not r['ok'] and not r['skipped'])
        s = sum(1 for r in self.rows if r['skipped'])
        return dict(passed=p, failed=f, skipped=s, total=len(self.rows),
                    failures=[r['name'] for r in self.rows
                              if not r['ok'] and not r['skipped']])


def _j(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().tolist()
    if isinstance(x, (list, tuple)):
        return [_j(v) for v in x]
    if isinstance(x, dict):
        return {k: _j(v) for k, v in x.items()}
    return x


def seeded_like(n, ref, seed):
    """`n` standard normals from a CPU generator, moved to `ref`'s device and dtype.

    SEEDED ON THE CPU AND THEN MOVED, deliberately. A CUDA generator produces a DIFFERENT stream
    for the same seed, so a GPU run and a CPU run of this suite would probe different directions
    and their finite-difference errors would not be comparable. Comparability across devices is
    the one property these checks need; the cost of the transfer is a few kilobytes.
    """
    g = torch.Generator().manual_seed(int(seed))
    return torch.randn(n, generator=g, dtype=torch.float64).to(
        device=ref.device, dtype=ref.dtype)


def rel(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    den = max(float(np.abs(b).max()), 1e-300)
    return float(np.abs(a - b).max() / den)


# =========================================================================== V0 OWNERSHIP
def stage_V0(L, env, adapter_cls, win_t, orig_encoder):
    """Encoder conversion parity and disjoint parameter ownership."""
    enc = env.fit_sys.encoder
    par = check_split_parity(orig_encoder, enc, win_t['uhist'], win_t['yhist'])
    L.add('V0', 'encoder_conversion_output_parity', par['max_rel'] <= 1e-6,
          'the split encoder reproduces the shared-net encoder\'s initial-state outputs',
          'spec Sect. 4.2 default migration',
          'max relative difference at or below float32 round-off', par['max_rel'], tol=1e-6,
          category='data-computable',
          detail=f"max abs {par['max_abs']:.3e}, max rel {par['max_rel']:.3e}",
          establishes='the conversion preserves the function at the reference point',
          not_establishes='that the two optimisation problems are the same; parameter sharing '
                          'changed, so optimiser state must be reset identically in every arm '
                          'and a converted-versus-original unregularised control is required')
    L.add('V0', 'encoder_groups_disjoint_by_identity', par['disjoint'],
          'the physical- and latent-initialisation parameter sets share no tensor',
          'spec Eq. (7)', 'no shared tensor id',
          dict(n_phys_tensors=par['n_phys'], n_lat_tensors=par['n_lat'],
               n_phys_scalars=par['n_phys_scalars'], n_lat_scalars=par['n_lat_scalars']),
          detail=f"xi_p {par['n_phys_scalars']} scalars, eta_a {par['n_lat_scalars']} scalars",
          establishes='identity-level disjointness',
          not_establishes='behavioural disjointness, which the perturbation checks below test')

    ad = adapter_cls(env, win_t)
    counts = check_ownership(ad)
    L.add('V0', 'parameter_groups_disjoint', True,
          'physical / augmentation / encoder groups are disjoint by tensor identity across the '
          'whole model, not only inside the encoder',
          'trajectory_adapter.check_ownership', 'no overlap', counts,
          detail=str(counts))

    # BEHAVIOURAL disjointness: perturb one group, require the other initialisation unchanged.
    with torch.no_grad():
        xp0, xa0 = ad.encode()
        xp0, xa0 = xp0.clone(), xa0.clone()
        for gname, other in (('augmentation', 'physical initial state'),
                             ('encoder', 'latent initial state')):
            ps = [p for p in ad.parameter_groups()[gname]]
            saved = [p.detach().clone() for p in ps]
            for p in ps:
                p.add_(1e-3 * torch.randn_like(p))
            xp1, xa1 = ad.encode()
            d_phys = float((xp1 - xp0).abs().max())
            d_lat = float((xa1 - xa0).abs().max())
            for p, s in zip(ps, saved):
                p.copy_(s)
            watched = d_phys if gname == 'augmentation' else d_lat
            moved = d_lat if gname == 'augmentation' else d_phys
            L.add('V0', f'perturb_{gname}_leaves_{other.replace(" ", "_")}_fixed',
                  watched == 0.0,
                  f'perturbing the {gname} group leaves the {other} bit-identical',
                  'spec Sect. 4.2 "Tensor identity checks alone are insufficient"',
                  'exactly zero', watched, tol=0.0, category='data-computable',
                  detail=f'watched change {watched:.3e}, own change {moved:.3e}',
                  establishes='the two initialisations have genuinely disjoint dependencies',
                  not_establishes='anything about the rest of the predictor')
    return ad


# ======================================================================== V1 SIMULATOR
def stage_V1(L, env, ad):
    """Production parity of the full stack, off semantics, gate restoration."""
    fs = env.fit_sys
    with torch.no_grad():
        L_prod = float(ad.production_loss())
        p_full = ad.scored_stack(mode='full')
        y = ad.w['yfuture'][:, ad.burn_in:].reshape(-1)
        L_adapt = float(((p_full - y) ** 2).mean())
    L.add('V1', 'scored_loss_matches_production', rel(L_adapt, L_prod) <= 1e-12,
          'the adapter\'s scored stack reproduces fit_sys\'s OWN loss on the same windows',
          'spec Sect. 9.2 primary parity target; interconnect.py SSE_Interconnect_Composed.loss',
          'relative difference at round-off', rel(L_adapt, L_prod), tol=1e-12,
          category='data-computable',
          detail=f'production {L_prod:.12e}, adapter {L_adapt:.12e}, '
                 f'rel {rel(L_adapt, L_prod):.2e}',
          establishes='the stack, the burn-in slice and the averaging match the trained objective',
          not_establishes='that the off and clamped interventions are correct')

    # gate restoration, including on an exception
    g0 = env.ann_block.out_gate.detach().clone()
    try:
        with ann_gate(env.ann_block, np.zeros(env.ann_block.nw)):
            raise RuntimeError('deliberate')
    except RuntimeError:
        pass
    restored = bool(torch.equal(env.ann_block.out_gate, g0))
    L.add('V1', 'gate_restored_on_exception', restored,
          'the intervention scope restores the previous gate even when the body raises',
          'spec Sect. 4.3 "exception-safe restoration"', 'gate unchanged',
          restored, detail=f'gate after exception = {env.ann_block.out_gate.tolist()[:4]}...')
    L.add('V1', 'gate_default_is_full', bool(torch.all(g0 == 1)),
          'the non-persistent gate initialises to full mode, so loading a checkpoint never '
          'silently starts in an intervention',
          'spec Sect. 4.3', 'all ones', g0.tolist()[:4])

    with torch.no_grad():
        c = ad.contributions()
    err = float((c['total'] - c['lat'] - c['direct']).abs().max())
    scale = max(float(c['total'].abs().max()), 1e-300)
    L.add('V1', 'intervention_decomposition_identity', err <= 1e-9 * scale,
          'd = d_lat + d_dir',
          'spec Eq. (9)', 'exact for any three vectors', err / scale, tol=1e-9,
          category='data-computable',
          detail=f'max |d - d_lat - d_dir| / |d| = {err/scale:.2e}',
          establishes='the three stacks came from one code path, nothing more',
          not_establishes='ANY simulator semantics; this identity is algebraically trivial and '
                          'gaps.md Sect. 10 already records the opposite claim as FALSE')
    L.add('V1', 'latent_route_is_nonzero', float(c['lat'].norm()) > 0,
          'the latent route carries a nonzero contribution at this checkpoint',
          'spec Sect. 4.3', 'nonzero', float(c['lat'].norm()), category='data-computable',
          detail=f"|d| {float(c['total'].norm()):.4e}  |d_lat| {float(c['lat'].norm()):.4e}  "
                 f"|d_dir| {float(c['direct'].norm()):.4e}")

    # clamped latent state must be zero at every step, asserted at runtime
    zero_lat = _clamped_latent_trace(env, ad)
    L.add('V1', 'clamped_latent_state_is_zero_everywhere', zero_lat['max_abs'] == 0.0,
          'under the clamped gate the latent state is exactly zero at initialisation and after '
          'every transition',
          'spec Sect. 4.3 runtime assertion', 'exactly zero', zero_lat['max_abs'], tol=0.0,
          category='data-computable',
          detail=f"max |a_k| over {zero_lat['n_steps']} steps = {zero_lat['max_abs']:.3e}",
          establishes='the clamped intervention isolates the direct route under THIS routing',
          not_establishes='that a future ungated state-write path could not break it')

    # off independence from every declared augmentation parameter
    off_dep = _off_independence(ad)
    L.add('V1', 'off_independent_of_every_augmentation_parameter',
          off_dep['max_change'] == 0.0,
          'perturbing every declared augmentation parameter, including the latent '
          'initialisation, leaves p_off bit-identical',
          'spec Sect. 4.3', 'exactly zero', off_dep['max_change'], tol=0.0,
          category='data-computable',
          detail=f"{off_dep['n_tensors']} tensors perturbed, max |dp_off| = "
                 f"{off_dep['max_change']:.3e}",
          establishes='the premise that licenses evaluating off under no_grad',
          not_establishes='that a future ownership change inherits it; this check must be re-run')
    L.add('V1', 'off_has_no_autograd_dependence_on_eta', off_dep['grad_max'] == 0.0,
          'an AD-enabled off evaluation produces no gradient into any augmentation parameter',
          'spec Sect. 4.3 "a zero derivative at one special parameter value is insufficient on '
          'its own"; this is the AD half of that pair',
          'no gradient', off_dep['grad_max'], tol=0.0, category='data-computable',
          detail=f"max |d(sum p_off)/d eta| = {off_dep['grad_max']:.3e}")
    return c


def _clamped_latent_trace(env, ad):
    """Run the clamped rollout with a hook on the interconnect state to watch the latent rows."""
    cfg = env.cfg
    nxp = cfg.nx_phys
    seen = []
    root = env.fit_sys.hfn
    orig_forward = root.forward

    def watched(x, u, *a, **k):
        seen.append(float(x[:, nxp:].abs().max()))
        return orig_forward(x, u, *a, **k)

    xp, xa = ad.encode()
    with torch.no_grad():
        x0 = ad.x0_from(xp.detach(), torch.zeros_like(xa))
        root.forward = watched
        try:
            with ann_gate(env.ann_block, ad._masks['clamped']):
                ad.simulate(x0)
        finally:
            root.forward = orig_forward
    return dict(max_abs=(max(seen) if seen else float('nan')), n_steps=len(seen))


def _off_independence(ad):
    ps = ad.parameter_groups()['augmentation']
    with torch.no_grad():
        base = ad.scored_stack(mode='off').clone()
        saved = [p.detach().clone() for p in ps]
        for p in ps:
            p.add_(1e-2 * torch.randn_like(p))
        pert = ad.scored_stack(mode='off')
        change = float((pert - base).abs().max())
        for p, s in zip(ps, saved):
            p.copy_(s)
    for p in ps:
        p.grad = None
    out = ad.scored_stack(mode='off').sum()
    gm = 0.0
    if out.requires_grad:
        gs = torch.autograd.grad(out, ps, allow_unused=True, retain_graph=False)
        gm = max((float(g.abs().max()) for g in gs if g is not None), default=0.0)
    return dict(max_change=change, grad_max=gm, n_tensors=len(ps))


# ======================================================================== V2 DERIVATIVES
def stage_V2(L, env, ad, ix=None, steps=(1e-4, 1e-5, 1e-6)):
    """Directional finite differences for every group, and the initialisation chain rule."""
    res = {}
    # -- physical
    lp = ad.phys.log_params
    dirn = seeded_like(lp.numel(), lp, 20260908)
    dirn = dirn / dirn.norm()
    an = ad.physical_directional(dirn, ix=ix).cpu().numpy()
    best, tab = np.inf, {}
    with torch.no_grad():
        for h in steps:
            saved = lp.detach().clone()
            lp.copy_(saved + h * dirn.reshape(lp.shape))
            pp = ad.scored_stack(mode='full', ix=ix).cpu().numpy()
            lp.copy_(saved - h * dirn.reshape(lp.shape))
            pm = ad.scored_stack(mode='full', ix=ix).cpu().numpy()
            lp.copy_(saved)
            fd = (pp - pm) / (2 * h)
            e = rel(fd, an)
            tab[f'{h:g}'] = e
            best = min(best, e)
    res['physical'] = tab
    L.add('V2', 'physical_directional_derivative', best < 1e-4,
          'the forward-mode physical sensitivity agrees with central finite differences of the '
          'production rollout along a random log-coordinate direction',
          'spec Eq. (21) and Sect. 8', 'relative error at the float32 differencing floor', best,
          tol=1e-4, category='data-computable',
          detail='best over steps ' + ', '.join(f'{k}:{v:.2e}' for k, v in tab.items()),
          establishes='S is the derivative of the map production actually evaluates',
          not_establishes='anything about the rank or the usefulness of that derivative')

    # -- latent initialisation (eta_a), the case the earlier work did not separate
    for gname in ('augmentation', 'encoder'):
        ps = ad.parameter_groups()[gname]
        flat = torch.cat([p.reshape(-1) for p in ps])
        v = seeded_like(flat.numel(), flat, 7 + len(gname))
        v = v / v.norm()
        # analytic: reverse-mode VJP of a scalar projection of the stack
        for p in ps:
            p.grad = None
        out = ad.scored_stack(mode='full', ix=ix)
        wgt = seeded_like(out.numel(), out, 3)
        s = (out * wgt).sum()
        gs = torch.autograd.grad(s, ps, allow_unused=True)
        gflat = torch.cat([(torch.zeros_like(p).reshape(-1) if gi is None else gi.reshape(-1))
                           for p, gi in zip(ps, gs)])
        an_s = float(gflat @ v)
        bestg, tabg = np.inf, {}
        with torch.no_grad():
            for h in steps:
                saved = [p.detach().clone() for p in ps]
                _add_flat(ps, h * v)
                sp_ = float((ad.scored_stack(mode='full', ix=ix) * wgt).sum())
                for p, sv in zip(ps, saved):
                    p.copy_(sv)
                _add_flat(ps, -h * v)
                sm_ = float((ad.scored_stack(mode='full', ix=ix) * wgt).sum())
                for p, sv in zip(ps, saved):
                    p.copy_(sv)
                fd = (sp_ - sm_) / (2 * h)
                e = abs(fd - an_s) / max(abs(an_s), 1e-300)
                tabg[f'{h:g}'] = e
                bestg = min(bestg, e)
        res[gname] = tabg
        L.add('V2', f'{gname}_directional_derivative', bestg < 1e-3,
              f'the reverse-mode {gname}-group derivative of a random projection of the stack '
              f'agrees with central finite differences',
              'spec Sect. 8 "Finite-difference eta, including latent initialization"',
              'relative error at the differencing floor', bestg, tol=1e-3,
              category='data-computable',
              detail='best over steps ' + ', '.join(f'{k}:{v:.2e}' for k, v in tabg.items()))

    # -- initialisation chain rule, E_w = Gamma_w J_w against a direct perturbation
    G = ad.initial_state_jacobians(ix)
    Jb, names, sizes = ad.encoder_jacobians(ix)
    E_chain = np.vstack([G[w] @ Jb[w] for w in range(len(G))])
    ps = ad.parameter_groups()['encoder']
    flat_n = sum(p.numel() for p in ps)
    v = seeded_like(flat_n, ps[0], 11)
    v = v / v.norm()
    an_dir = E_chain @ v.numpy().astype(np.float64)
    bestc, tabc = np.inf, {}
    with torch.no_grad():
        saved = [p.detach().clone() for p in ps]
        for h in steps:
            _add_flat(ps, h * v)
            pp = ad.scored_stack(mode='full', ix=ix).cpu().numpy()
            for p, sv in zip(ps, saved):
                p.copy_(sv)
            _add_flat(ps, -h * v)
            pm = ad.scored_stack(mode='full', ix=ix).cpu().numpy()
            for p, sv in zip(ps, saved):
                p.copy_(sv)
            e = rel((pp - pm) / (2 * h), an_dir)
            tabc[f'{h:g}'] = e
            bestc = min(bestc, e)
    res['chain_rule'] = tabc
    L.add('V2', 'initialisation_chain_rule_E_eq_Gamma_J', bestc < 1e-3,
          'E_w = Gamma_w J_w reproduces the directional derivative of the production rollout '
          'with respect to the physical-encoder parameters',
          'spec Eq. (13), D-184', 'relative error at the differencing floor', bestc, tol=1e-3,
          category='data-computable',
          detail='best over steps ' + ', '.join(f'{k}:{v:.2e}' for k, v in tabc.items()),
          establishes='the factorised construction is the same derivative as the direct one, '
                      'for the encoder actually attached',
          not_establishes='that the encoder has no other route into the prediction; that is the '
                          'separate premise check below')

    # the premise: xi_p enters ONLY through the physical initial state
    with torch.no_grad():
        xp, xa = ad.encode(ix)
        x0 = ad.x0_from(xp.detach().clone(), xa.detach().clone())
        base = ad.scored_stack(mode='full', ix=ix, x0=x0).clone()
        saved = [p.detach().clone() for p in ps]
        for p in ps:
            p.add_(1e-2 * torch.randn_like(p))
        pert = ad.scored_stack(mode='full', ix=ix, x0=x0)
        d = float((pert - base).abs().max())
        for p, sv in zip(ps, saved):
            p.copy_(sv)
    L.add('V2', 'xi_p_enters_only_through_initial_state', d == 0.0,
          'with the initial state held fixed, perturbing xi_p leaves the prediction '
          'bit-identical, which is the premise that makes the chain rule exact',
          'spec Sect. 6.2 warrant', 'exactly zero', d, tol=0.0, category='data-computable',
          detail=f'max |dp| with x_0 held = {d:.3e}',
          establishes='D-184 is applicable to this encoder',
          not_establishes='applicability to a future encoder that also feeds an output '
                          'correction or a scheduling signal; that would void it silently')

    # negative control: an omitted controller path must be DETECTED
    nc = _controller_negative_control(env, ad, ix, dirn)
    L.add('V2', 'negative_control_controller_path', nc['detected'],
          'freezing the controller state inside the rollout changes the physical sensitivity by '
          'far more than the finite-difference floor, so an omitted controller derivative path '
          'would be caught rather than absorbed',
          'spec Sect. 12 V2 "scheduling/controller negative controls"',
          'a large relative difference', nc['rel'], category='data-computable',
          detail=f"relative change {nc['rel']:.3e} against a derivative check floor of {best:.2e}")
    return res


def _add_flat(ps, v):
    i = 0
    for p in ps:
        n = p.numel()
        p.add_(v[i:i + n].reshape(p.shape).to(p.dtype))
        i += n


def _controller_negative_control(env, ad, ix, dirn):
    """Sever the controller feedback and show the physical sensitivity changes."""
    ref = ad.physical_directional(dirn, ix=ix).cpu().numpy()
    sim = env.fit_sys.simulator
    bank = sim.bank
    orig_step = bank.step

    def frozen(xc, e, ctrl):
        u_fb, xcn = orig_step(xc, e, ctrl)
        return u_fb, xc            # controller state does not evolve

    bank.step = frozen
    try:
        bad = ad.physical_directional(dirn, ix=ix).cpu().numpy()
    finally:
        bank.step = orig_step
    r = rel(bad, ref)
    return dict(rel=r, detected=r > 1e-3)


# ========================================================================= V3 GEOMETRY
def stage_V3(L, env, ad, ix=None, rank_rtol=1e-12, ops=None, op_timeout=None, dump_dir=None):
    """S, Gamma, J, compressed E, ranks, structural null tests, coordinate robustness.

    EVERY OPERATION IS TIMED AND WATCHDOGGED. This stage has never completed: four attempts
    produced no output for 20 to 100 minutes at about one per cent CPU, the stall point moved
    between them, and nothing in the saved record said which call was running. `diag.timed`
    records wall clock, CPU time, peak RSS and CUDA peak per operation and dumps every thread's
    stack when one overruns, so a fifth stall names its call instead of repeating the mystery.

    The 14 physical JVPs are timed INDIVIDUALLY rather than as a block: if one of them is the
    problem, a single aggregate figure hides it, and the earlier console logs suggest the stall
    lands mid-sequence rather than at the start.
    """
    from . import diag
    ops = ops if ops is not None else diag.OpLedger('V3')
    dump = os.path.join(dump_dir, 'v3_stacks.log') if dump_dir else None

    def op(label):
        return diag.timed(label, ledger=ops, timeout=op_timeout,
                          device=getattr(env, 'device', None), dump_to=dump, hard=True)

    out = {}
    n_lam = ad.phys.log_params.numel()
    cols = []
    for j in range(n_lam):
        with op(f'S: physical JVP {j + 1}/{n_lam}'):
            e = torch.zeros(n_lam, dtype=ad.phys.log_params.dtype,
                            device=ad.phys.log_params.device)
            e[j] = 1.0
            cols.append(ad.physical_directional(e, ix=ix).cpu().numpy())
    S = np.stack(cols, axis=1).astype(np.float64)
    N = S.shape[0]
    out['S_shape'] = list(S.shape)
    out['t_S_seconds'] = sum(r['wall_s'] for r in ops.rows)

    with op('Gamma_w: initial-state JVPs (batched over windows)'):
        G = ad.initial_state_jacobians(ix)
    with op('J_w: encoder-only Jacobian (jacrev)'):
        Jb, names, sizes = ad.encoder_jacobians(ix)
    t_geo = sum(r['wall_s'] for r in ops.rows[n_lam:])
    with op('compressed_nuisance_basis'):
        Q_E, info = compressed_nuisance_basis(G, Jb, rtol=rank_rtol, N=N)
    out['compressed'] = {k: v for k, v in info.items() if k != 'sv_Gamma'}
    out['E_explicit_shape_avoided'] = [N, int(Jb[0].shape[1])]
    out['sv_Gamma_first_window'] = info['sv_Gamma'][0]
    out['t_geometry_seconds'] = t_geo
    out['n_xi_p'] = int(sum(sizes))

    # COMPRESSED VERSUS EXPLICIT E, on a genuinely tiny row set. The specification says to
    # compare against explicit E "on a tiny set", and the reason is measurable: the explicit E
    # here is 4800-by-6774, and its SVD is ~1.6e11 flops and about a gigabyte. It was measured
    # running past 25 minutes without finishing, which is exactly the cost the factorisation of
    # Eq. (15) exists to avoid, so paying it in full to check the factorisation would be
    # self-defeating.
    #
    # Row subsampling is EXACT for this comparison rather than an approximation of it: E is
    # block diagonal over windows, so selecting a subset of each window's scored rows gives
    # E_sub = blockdiag(Gamma_w[rows]) stack(J_w), and the compressed construction rebuilt from
    # the SAME row-restricted Gamma blocks is the matching object. What is checked is the
    # identity, on a small instance of it; the ranks reported for the FULL set come from the
    # compressed build, which is the one that scales.
    step = max(1, G[0].shape[0] // 30)
    G_sub = [B[::step] for B in G]
    n_sub = sum(B.shape[0] for B in G_sub)
    with op('explicit E on the row-subsampled set (+ its SVD)'):
        E_exp = np.vstack([G_sub[w] @ Jb[w] for w in range(len(G_sub))])
        Q_E_sub, info_sub = compressed_nuisance_basis(G_sub, Jb, rtol=rank_rtol, N=n_sub)
        U, s, _ = np.linalg.svd(E_exp, full_matrices=False)
    rE = int(np.sum(s > max(rank_rtol * s[0], 0.0))) if s.size and s[0] > 0 else 0
    QE_exp = U[:, :rE]
    Q_E_cmp = Q_E_sub
    # NEVER `Q @ Q.T`: at this N a dense projector is hundreds of megabytes and the
    # specification prohibits it outright. `projector_distance` computes the same norm in the
    # factors (see subspace.py).
    dP = projector_distance(Q_E_cmp, QE_exp)
    L.add('V3', 'compressed_E_equals_explicit_E', dP < 1e-8 and rE == Q_E_cmp.shape[1],
          'the compressed nuisance basis spans exactly range(E) built explicitly, on a '
          'row-subsampled instance of the same identity',
          'spec Eq. (15)', 'equal projectors and equal ranks', dP, tol=1e-8,
          category='data-computable',
          detail=f'rows {n_sub} of {N} (every {step}th scored row); rank compressed '
                 f'{Q_E_cmp.shape[1]}, rank explicit {rE}, projector distance {dP:.3e}; '
                 f'full-set Z is {info["Z_shape"]} against a full E of '
                 f'{[N, Jb[0].shape[1]]}',
          establishes='the factorised build is exact before truncation on the real predictor',
          not_establishes='that Z stays affordable at pilot scale; that is measured, not implied')

    with op('profiling + orth(M_E S)'):
        geom = TrajectoryProjectionGeometry(S, Q_E=Q_E, L=ad.loss_weight(), rank_rtol=rank_rtol,
                                            strict=False, metadata=ad.metadata())
    assert_geometry_detached(geom)
    out['rank_S_raw'] = int(geom.rank_S_before_profiling)
    out['rank_E'] = int(geom.rank_E)
    out['rank_S_profiled'] = int(geom.rank_S)
    out['sv_S_profiled'] = geom.sv_S.tolist()
    sv_S_raw = np.linalg.svd(S, compute_uv=False)
    out['sv_S_raw'] = sv_S_raw.tolist()
    out['cond_S_raw'] = float(sv_S_raw[0] / max(sv_S_raw[-1], 1e-300))

    L.add('V3', 'ranks_measured', True,
          'physical rank before and after profiling by the actual shared encoder',
          'spec Sect. 6.4-6.5', 'a measurement, not a target',
          dict(rank_S=out['rank_S_raw'], rank_E=out['rank_E'],
               rank_Sbar=out['rank_S_profiled']),
          category='data-computable',
          detail=f"r_S={out['rank_S_raw']}  r_E={out['rank_E']}  r_bar={out['rank_S_profiled']} "
                 f"of 14 columns, N={N}",
          establishes='the rank prerequisite status of Sect. 6.5 on THIS window set',
          not_establishes='the rank on the final pilot window set, nor tangent separation, nor '
                          'recovery')

    # structural null directions from D_lambda kappa, spec Sect. 6.4
    nulls = _kappa_null_directions(ad)
    rows = {}
    nS = float(np.linalg.norm(S, 2))
    for nm, v in nulls.items():
        rows[nm] = float(np.linalg.norm(S @ v) / max(nS * np.linalg.norm(v), 1e-300))
    worst = max(rows.values()) if rows else 0.0
    L.add('V3', 'structural_null_directions', worst < 1e-6,
          'the reference log-coordinate null vectors derived from D_lambda kappa are annihilated '
          'by S, i.e. the predictor factors through the identifiable combinations',
          'spec Eq. (3) and Sect. 6.4', 'relative residual at round-off', worst, tol=1e-6,
          category='structural',
          detail=', '.join(f'{k} {v:.2e}' for k, v in rows.items()),
          establishes='rank(S) <= 10 is realised structurally and not by conditioning',
          not_establishes='that the surviving 10 are all supported by THIS excitation')

    # free per-window initial-state profiling: an explicitly labelled BOUND
    with op('free-initial-state nuisance bound (blockdiag Gamma + SVD)'):
        F = _block_diag_gamma(G, N)
        Uf, sf, _ = np.linalg.svd(F, full_matrices=False)
    rf = int(np.sum(sf > max(rank_rtol * sf[0], 0.0))) if sf.size and sf[0] > 0 else 0
    Qf = Uf[:, :rf]
    Sbar_f = S - Qf @ (Qf.T @ S)
    ss = np.linalg.svd(Sbar_f, compute_uv=False)
    floor = rank_rtol * float(sv_S_raw[0])
    r_free = int(np.sum(ss > floor))
    out['rank_free_nuisance'] = rf
    out['rank_S_after_free_profiling'] = r_free
    L.add('V3', 'free_initial_state_bound', True,
          'profiling a FREE per-window initial state removes at least as much as profiling the '
          'shared encoder, so this rank is a lower bound on the shared-encoder one',
          'spec Eq. (14) range inclusion', 'a labelled bound',
          dict(rank_free_nuisance=rf, rank_S_after=r_free),
          category='data-computable',
          detail=f'free-nuisance rank {rf}, r_bar under it {r_free} vs shared-encoder '
                 f"{out['rank_S_profiled']}",
          establishes='an upper bound on how much the shared encoder can remove',
          not_establishes='collapse of the shared-encoder geometry; the specification says so '
                          'explicitly and this row exists to keep the two apart')

    # range inclusion, checked rather than assumed
    inc = residual_outside(Q_E, Qf)
    L.add('V3', 'range_E_inside_range_blockdiag_Gamma', inc < 1e-8,
          'range(E) is contained in range(blockdiag Gamma_w)',
          'spec Eq. (14)', 'residual at round-off', inc, tol=1e-8, category='data-computable',
          detail=f'max residual of Q_E outside the free-nuisance range: {inc:.3e}')

    # coordinate robustness of the nuisance projector and rank-tolerance sweep
    with op('coordinate robustness (6 rebuilds)'):
        rob = _coordinate_robustness(G, Jb, N, rank_rtol, Q_E)
    out['robustness'] = rob
    L.add('V3', 'nuisance_coordinate_robustness', rob['worst_projector_diff'] < 1e-6,
          'diagonal rescalings of the encoder coordinates, xi\' = T xi, leave the TRUNCATED '
          'nuisance projector unchanged to the reported accuracy',
          'spec Sect. 6.6, E\' = E T^{-1}', 'projector distance near round-off',
          rob['worst_projector_diff'], tol=1e-6, category='data-computable',
          detail=f"worst projector diff {rob['worst_projector_diff']:.3e} over "
                 f"{rob['n_trials']} rescalings; ranks seen {sorted(set(rob['ranks']))}",
          establishes='the exact invariance survives truncation at this conditioning',
          not_establishes='invariance under arbitrary conditioning or at pilot scale')
    with op('rank-tolerance sweep'):
        swp = _tolerance_sweep(S, G, Jb, N, rank_rtol)
    out['tolerance_sweep'] = swp
    L.add('V3', 'rank_tolerance_sweep', True,
          'physical and nuisance ranks at rank tolerances 0.1x, 1x and 10x the default',
          'spec Sect. 6.4 "save threshold sweeps"', 'a report, not a target', swp,
          category='data-computable',
          detail='; '.join(f"{k}: r_E={v['rank_E']} r_bar={v['rank_Sbar']}"
                           for k, v in swp.items()),
          not_establishes='a rank claim at any single tolerance; a tolerance is never tuned '
                          'until a desired rank appears')
    out['operations'] = ops.summary()
    return geom, S, Q_E, out


def _block_diag_gamma(G, N):
    """blockdiag_w(Gamma_w) from blocks already computed.

    Built from the SAME `Gamma_w` used for the compressed construction rather than by calling
    `free_initial_state_nuisance`, which would repeat six forward-mode rollouts of the whole
    batch for no new information.
    """
    cols, off = [], 0
    for B in G:
        padded = np.zeros((N, B.shape[1]))
        padded[off:off + B.shape[0]] = B
        cols.append(padded)
        off += B.shape[0]
    return np.hstack(cols) if cols else np.zeros((N, 0))


def _kappa_null_directions(ad):
    """Null vectors of D_lambda kappa in log coordinates, from the block's own combination map.

    `kappa` depends on the raw parameters only through kb1+kb2, cb1+cb2 and Jb+Jh, so in RAW
    coordinates `e_kb1 - e_kb2` and the other two are exact null directions. In LOG coordinates
    `d theta_i = theta_i d lambda_i`, so the corresponding log direction is
    `e_i/theta_i - e_j/theta_j`, up to scale `theta_j e_i - theta_i e_j`.
    """
    names = ad.phys.PARAM_NAMES
    theta = ad.phys._recover_params().detach().cpu().numpy().astype(np.float64)
    pairs = {'kb1-kb2': ('kb1', 'kb2'), 'cb1-cb2': ('cb1', 'cb2'), 'Jb-Jh': ('Jb', 'Jh')}
    out = {}
    for nm, (a, b) in pairs.items():
        i, j = names.index(a), names.index(b)
        v = np.zeros(len(names))
        v[i] = theta[j]
        v[j] = -theta[i]
        out[nm] = v / np.linalg.norm(v)
    return out


def _coordinate_robustness(G, Jb, N, rtol, Q_ref, n=6, seed=20260908):
    rng = np.random.default_rng(seed)
    n_xi = Jb[0].shape[1]
    worst, ranks = 0.0, []
    for _ in range(n):
        t = 10.0 ** rng.uniform(-1, 1, size=n_xi)
        Jp = [J / t[None, :] for J in Jb]          # E' = E T^{-1} with T = diag(t)
        Qp, info = compressed_nuisance_basis(G, Jp, rtol=rtol, N=N)
        ranks.append(int(Qp.shape[1]))
        worst = max(worst, projector_distance(Qp, Q_ref))
    return dict(worst_projector_diff=worst, ranks=ranks, n_trials=n,
                rank_reference=int(Q_ref.shape[1]))


def _tolerance_sweep(S, G, Jb, N, base):
    sv0 = float(np.linalg.svd(S, compute_uv=False)[0])
    out = {}
    for f in (0.1, 1.0, 10.0):
        rt = base * f
        Q, info = compressed_nuisance_basis(G, Jb, rtol=rt, N=N)
        Sb = S - Q @ (Q.T @ S) if Q.shape[1] else S
        ss = np.linalg.svd(Sb, compute_uv=False)
        out[f'{f:g}x'] = dict(rtol=rt, rank_E=int(Q.shape[1]),
                              rank_Sbar=int(np.sum(ss > rt * sv0)),
                              sv_Sbar=ss.tolist())
    return out


# ================================================================ V4/V5 GRADIENT + CHUNKING
def stage_V4_V5(L, env, ad, geom, beta=1.0, ix=None):
    """Penalty gradient ownership, latent-initialisation reachability, exact chunking."""
    groups = ad.parameter_groups()
    named = ad.group_named()
    for p in sum(groups.values(), []):
        p.grad = None

    # penalty with the physical parameters and the PHYSICAL initial state held as constants
    def penalty(ix_=None, rows=None, chunk_ix=None):
        sel = ix if chunk_ix is None else chunk_ix
        xp, xa = ad.encode(sel)
        xp = xp.detach()                                    # tilde x_0 constant, spec Sect. 8
        z = torch.zeros_like(xa)
        with torch.no_grad():
            p_off = ad.scored_stack(mode='off', ix=sel, x0=ad.x0_from(xp, z))
        p_full = ad.scored_stack(mode='full', ix=sel, x0=ad.x0_from(xp, xa))
        return p_full - p_off

    # the physical parameters must be detached too: reparametrise them as constants
    lp = ad.phys.log_params
    lp_saved = lp.detach().clone()
    lp.requires_grad_(False)
    try:
        d = penalty()
        V = geom.penalty_mse(d, beta)
        V.backward()
        rep = {}
        for gname, items in named.items():
            rep[gname] = {n: (0.0 if p.grad is None else float(p.grad.abs().max()))
                          for n, p in items}
        g_aug = ad.parameter_groups()['augmentation']
        g_full = torch.cat([(torch.zeros_like(p).reshape(-1) if p.grad is None
                             else p.grad.reshape(-1)) for p in g_aug]).clone()
    finally:
        lp.requires_grad_(True)
        with torch.no_grad():
            lp.copy_(lp_saved)

    worst_phys = max(rep['physical'].values())
    worst_enc = max(rep['encoder'].values())
    L.add('V4', 'no_penalty_gradient_into_physical_or_encoder',
          worst_phys == 0.0 and worst_enc == 0.0,
          'the penalty puts no gradient into lambda or xi_p',
          'spec Eq. (22)', 'exactly zero', dict(physical=worst_phys, encoder=worst_enc),
          tol=0.0, category='data-computable',
          detail=f'max |g_lambda| {worst_phys:.3e}, max |g_xi_p| {worst_enc:.3e}',
          establishes='the declared selective-update policy is what the code does',
          not_establishes='that this update is the gradient of any scalar objective; the small '
                          'engines show it is not')
    lat_names = [n for n in rep['augmentation'] if n.startswith('enc.')]
    lat_max = max((rep['augmentation'][n] for n in lat_names), default=0.0)
    ann_max = max((v for n, v in rep['augmentation'].items() if n.startswith('ann.')), default=0.0)
    L.add('V4', 'penalty_gradient_reaches_latent_initialisation', lat_max > 0.0,
          'the penalty gradient reaches eta_a, the trainable latent initialisation',
          'spec Sect. 4.2 and Eq. (22)', 'strictly positive', lat_max,
          category='data-computable',
          detail=f'max |g_eta_a| {lat_max:.3e} over {len(lat_names)} tensors; '
                 f'max |g_eta_d| {ann_max:.3e}',
          establishes='the latent initialisation is not silently profiled away or detached',
          not_establishes='that the gradient is the right size, or that training uses it well')
    L.add('V4', 'penalty_gradient_reaches_ann', ann_max > 0.0,
          'the penalty gradient reaches eta_d', 'spec Eq. (22)', 'strictly positive', ann_max,
          category='data-computable', detail=f'max |g_eta_d| {ann_max:.3e}')

    # V5: exact two-pass chunking over WINDOWS
    n_w = ad.n_win if ix is None else len(ix)
    per = ad.T * ad.ny
    results = {}
    for k in (2, 4):
        if n_w % k:
            continue
        wins = np.array_split(np.arange(n_w), k)
        acc = ProjectedResidualAccumulator(geom, beta / geom.N)
        with torch.no_grad():
            for wsub in wins:
                rows = _row_slice(wsub, per)
                sub = torch.as_tensor(wsub, device=ad.w['ufuture'].device)
                acc.accumulate_value(penalty(chunk_ix=sub), rows=rows)
        acc.freeze()
        for p in g_aug:
            p.grad = None
        lp.requires_grad_(False)
        try:
            for wsub in wins:
                rows = _row_slice(wsub, per)
                sub = torch.as_tensor(wsub, device=ad.w['ufuture'].device)
                acc.backward_chunk(penalty(chunk_ix=sub), rows=rows)
        finally:
            lp.requires_grad_(True)
        g_ch = torch.cat([(torch.zeros_like(p).reshape(-1) if p.grad is None
                           else p.grad.reshape(-1)) for p in g_aug])
        ev = abs(acc.value - float(V)) / max(abs(float(V)), 1e-300)
        eg = float((g_ch - g_full).norm() / max(float(g_full.norm()), 1e-300))
        results[k] = dict(rel_value=ev, rel_grad=eg,
                          windows_per_chunk=[len(w) for w in wins])
        L.add('V5', f'chunked_{k}_matches_full_stack', ev < 1e-8 and eg < 1e-5,
              f'the two-pass form over {k} genuine window subsets reproduces the full-stack '
              f'penalty value and augmentation gradient',
              'spec Eqs. (23)-(24)', 'relative differences at round-off',
              dict(value=ev, grad=eg), tol=1e-5, category='data-computable',
              detail=f'rel value {ev:.2e}, rel grad {eg:.2e}, windows per chunk '
                     f'{[len(w) for w in wins]}',
              establishes='chunking is a memory device that does not change the objective',
              not_establishes='that a SUM of per-chunk penalties would; it is a different '
                              'objective and the small engines exhibit a case where they differ')
    return dict(gradients=rep, chunking=results, V=float(V))


def _row_slice(wsub, per):
    """Rows of the stack belonging to a contiguous window subset."""
    a, b = int(wsub[0]), int(wsub[-1]) + 1
    assert np.array_equal(wsub, np.arange(a, b)), 'window chunks must be contiguous'
    return slice(a * per, b * per)


# ================================================================== V7 INTERPRETATION
def stage_V7(L, env, ad, geom, c):
    """The counterexamples that stop a zero metric from being read as a guarantee."""
    m = {k: geom.contribution_metrics(c[k]) for k in ('total', 'lat', 'direct')}
    L.add('V7', 'route_metrics_reported_separately', True,
          'the projected total, latent and direct contributions are reported separately, because '
          'projected routes can cancel and a small total can conceal two aligned routes',
          'spec Eq. (9) and Sect. 10', 'three numbers, not one', m,
          category='data-computable',
          detail='; '.join(f"{k}: ell={v['ell']:.4f} a_par={v['a_par']:.3e} "
                           f"|M_E d|={v['norm_profiled']:.3e}" for k, v in m.items()),
          not_establishes='that a small ell means the augmentation is not substituting')
    dperp = c['total'] - geom.Q_S @ (geom.Q_S.T @ c['total'])
    ov = float((geom.Q_S.T @ dperp).norm())
    L.add('V7', 'zero_value_overlap_is_constructible', ov < 1e-8 * float(c['total'].norm()),
          'a contribution with exactly zero projected value exists on this geometry; its '
          'existence says nothing about the augmentation tangent space',
          'spec Sect. 7.3', 'zero projection', ov, tol=1e-8, category='data-computable',
          detail=f'||Q^T d_perp|| = {ov:.3e}',
          not_establishes='local inability to substitute, which needs Eq. (20) or the Sect. 7.4 '
                          'diagnostic')
    return m
