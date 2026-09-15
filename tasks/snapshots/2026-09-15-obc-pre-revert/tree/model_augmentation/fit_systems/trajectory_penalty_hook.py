"""The trajectory-orthogonality penalty as a POST-BACKWARD GRADIENT CONTRIBUTION.

Specification Eq. (22) and Sect. 8.1. This is the piece that makes the method reachable from a
training run; everything below it (geometry, projection, exact chunking) already exists in
`trajectory_orth_projection.py` and is reused, not re-implemented.

WHY IT IS A HOOK AND NOT A LOSS TERM. The production closure is

    Loss = self.loss(...)  ->  self.optimizer.zero_grad()  ->  Loss.backward()

so a penalty added inside `loss()` is backwarded and then IMMEDIATELY WIPED by the closure's own
`zero_grad`. That is not a hypothetical: `checks_v6.check_accumulation_hook_survives_zero_grad`
runs both orders and shows the wrong one loses the penalty entirely. The gradient therefore has
to be accumulated AFTER the base backward and BEFORE the optimizer's update, which is exactly
where this hook runs.

WHAT IT ADDS, and to whom:

    g_eta  +=  (2 beta / N) (D_eta d)^T Q Q^T d          only eta
    g_lambda, g_xi_p                                     untouched

`lambda` and the physical initial state are held as CONSTANTS while the penalty is differentiated
(`requires_grad_(False)` on the physical parameters, `.detach()` on the physical initial state),
which is what makes the routing a property of the computation rather than of a promise. The
resulting update is NOT the gradient of `J_base + V`: the small engines exhibit a nonzero mixed
derivative `D_lambda D_eta V`, so the selectively routed field is the gradient of no scalar
objective at all. That is a deliberate, recorded design choice, not an oversight, and it means no
convergence theory covers this update.

CHUNKING is the exact two-pass form of Eqs. (23)-(24), NOT a per-chunk penalty: `V` is
`beta||sum_b r_b||^2` and a sum of per-chunk penalties is a different objective. The accumulator
enforces that; `test_trajectory_projection` constructs a case where the two differ by six orders
of magnitude.
"""
__project_origin__ = "added"

import contextlib

import torch


@contextlib.contextmanager
def _frozen(params):
    """Hold parameters as constants for the duration, restoring `requires_grad` exactly.

    `requires_grad_(False)` rather than a detached copy: the penalty rollout must run through the
    SAME modules the base loss used, so that it is the same predictor, and a copy would be a
    second implementation of the thing this whole design avoids.
    """
    saved = [(p, bool(p.requires_grad)) for p in params]
    try:
        for p, _ in saved:
            p.requires_grad_(False)
        yield
    finally:
        for p, was in saved:
            p.requires_grad_(was)


class TrajectoryPenaltyHook:
    """Accumulate the Eq. (22) penalty gradient into the augmentation group.

    Parameters
    ----------
    adapter : a `TrajectoryAdapter`, supplying `contributions`-style scored stacks on the FIXED
        regularisation window set. Note this set is fixed and independent of the training batch:
        the geometry was built on it, and evaluating the penalty on a different set each step
        would be a different objective with a basis that does not belong to it.
    geometry : a frozen `TrajectoryProjectionGeometry`.
    beta : penalty weight. A CHOSEN CONSTANT WITH NO JUSTIFICATION until an L-curve is run
        (RULES.md, 2026-09-07 note); it must be reported as one.
    chunk : windows per chunk, or 0 for a single stack. Memory only; the value is unchanged.
    """

    def __init__(self, adapter, geometry, beta, chunk=0, groups=None, verbose_every=0):
        self.adapter = adapter
        self.geometry = geometry
        self.beta = float(beta)
        self.chunk = int(chunk or 0)
        self._groups = groups
        self.verbose_every = int(verbose_every)
        self.calls = 0
        self.last = {}
        # An optional callback invoked after every application, for live diagnostics. A named
        # attribute rather than a monkey-patched `__call__`: patching it with a local closure is
        # what made the whole fit system unpicklable and killed the first real run at its first
        # validation. Signature: observer(hook, call_index).
        self.observer = None

    # ------------------------------------------------------------------ groups
    def groups(self):
        return self._groups if self._groups is not None else self.adapter.parameter_groups()

    def _eta(self):
        return list(self.groups()['augmentation'])

    def _frozen_params(self):
        g = self.groups()
        return list(g['physical']) + list(g['encoder'])

    # ------------------------------------------------------------------ contribution
    def _contribution(self, ix=None):
        """`d = p_full - p_off` with the physical initial state held constant.

        `off` is evaluated under `no_grad`: it is independent of every augmentation parameter,
        which `checks.stage_V1` asserts by perturbation AND by an AD check rather than assuming.
        Its VALUES still use the current physical parameters and initial state, so it is
        recomputed every call rather than cached across steps.
        """
        ad = self.adapter
        xp, xa = ad.encode(ix)
        xp = xp.detach()                       # tilde x_0 constant, spec Sect. 8
        with torch.no_grad():
            p_off = ad.scored_stack(mode='off', ix=ix,
                                    x0=ad.x0_from(xp, torch.zeros_like(xa)))
        p_full = ad.scored_stack(mode='full', ix=ix, x0=ad.x0_from(xp, xa))
        return p_full - p_off

    # ------------------------------------------------------------------ the hook
    def __call__(self, fit_sys=None, *args, **kwargs):
        """Add the penalty gradient to `eta`. Returns the penalty VALUE for logging."""
        from model_augmentation.fit_systems.trajectory_orth_projection import (
            ProjectedResidualAccumulator)

        ad = self.adapter
        eta = self._eta()
        before = [None if p.grad is None else p.grad.detach().clone() for p in eta]

        with _frozen(self._frozen_params()):
            if not self.chunk or self.chunk >= ad.n_win:
                d = self._contribution()
                V = self.geometry.penalty_mse(d, self.beta)
                V.backward()
                value = float(V.detach())
            else:
                # EXACT two-pass form, Eqs. (23)-(24). Pass 1 accumulates the residual with no
                # graph; pass 2 freezes it and recomputes each chunk. Chunks are contiguous
                # window blocks so the row slice of the stack is contiguous too.
                per = ad.T * ad.ny
                edges = list(range(0, ad.n_win, self.chunk)) + [ad.n_win]
                blocks = [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]
                acc = ProjectedResidualAccumulator(self.geometry, self.beta / self.geometry.N)
                dev = ad.w['ufuture'].device
                with torch.no_grad():
                    for a, b in blocks:
                        sub = torch.arange(a, b, device=dev)
                        acc.accumulate_value(self._contribution(sub), rows=slice(a * per, b * per))
                acc.freeze()
                for a, b in blocks:
                    sub = torch.arange(a, b, device=dev)
                    acc.backward_chunk(self._contribution(sub), rows=slice(a * per, b * per))
                value = acc.value

        added = 0.0
        for p, g0 in zip(eta, before):
            if p.grad is not None:
                delta = p.grad if g0 is None else (p.grad - g0)
                added = max(added, float(delta.abs().max()))
        self.calls += 1
        # `d` is PUBLISHED, detached, so a diagnostic can compute alignment and amplitudes
        # without re-running the rollout. Under chunking the contribution was never assembled as
        # one vector, so it is absent there rather than silently partial.
        self.last = dict(V=value, max_added_eta_grad=added, chunked=bool(self.chunk),
                         beta=self.beta, N=int(self.geometry.N),
                         d=(d.detach() if (not self.chunk or self.chunk >= ad.n_win) else None))
        if self.observer is not None:
            self.observer(self, self.calls)
        if self.verbose_every and self.calls % self.verbose_every == 0:
            print(f'[traj-orth] call {self.calls}  V {value:.6e}  '
                  f'max |dg_eta| {added:.3e}  beta {self.beta:g} (a chosen constant)', flush=True)
        return value

    # ------------------------------------------------------------------ self-check
    def check_ownership(self, tol=0.0, require_nonzero=False):
        """The declared routing, measured on the live model rather than promised.

        Returns a report; RAISES only for a gradient that must not exist.

        CORRECTED 2026-09-08 after review. The first version also raised when the augmentation
        gradient was zero, on the reasoning that an inert penalty is a bug. That is
        mathematically wrong: `D_eta V = 0` is the CORRECT value at several legitimate points,
        and at least three of them are reachable in normal use.

          - the augmentation is off, where `d = 0`, so `V = 0` and `D_eta V = 0`. The
            specification says so in as many words and warns against reading it as a defect
          - the projected residual `Q^T d` is already zero, which is the state the method is
            trying to reach
          - a stationary point of the penalty in `eta`

        Treating any of those as a broken implementation would abort a run for succeeding. The
        zero case is REPORTED, and `require_nonzero=True` is available for a caller that has
        deliberately constructed a non-degenerate reference and wants reachability enforced
        there. `check_reachability` below is the honest test of "can the penalty move eta",
        because it perturbs to a point where a nonzero gradient is expected.

        Gradients are restored in a `finally`, so a raised assertion does not also leave the
        caller's gradients cleared. The first version restored them only on the success path.
        """
        g = self.groups()
        saved = {id(p): (None if p.grad is None else p.grad.detach().clone())
                 for grp in g.values() for p in grp}
        report = {}
        try:
            for grp in g.values():
                for p in grp:
                    p.grad = None
            value = self()
            for name in ('physical', 'encoder'):
                worst = max((float(p.grad.abs().max()) for p in g[name] if p.grad is not None),
                            default=0.0)
                report[name] = worst
                if worst > tol:
                    raise AssertionError(
                        f'the trajectory penalty put gradient {worst:.3e} into "{name}" '
                        f'(tolerance {tol:g}). The optimiser could then reduce the penalty by '
                        f'moving the subspace instead of correcting the augmentation, which '
                        f'corrupts the quantity the method exists to protect while the penalty '
                        f'value falls.')
            got = max((float(p.grad.abs().max()) for p in g['augmentation']
                       if p.grad is not None), default=0.0)
            report['augmentation'] = got
            report['V'] = float(value)
            report['augmentation_gradient_is_zero'] = (got == 0.0)
            if got == 0.0 and require_nonzero:
                raise AssertionError(
                    'the trajectory penalty produced no gradient into the augmentation at a '
                    'reference the caller declared non-degenerate. Note this is NOT a defect in '
                    'general: D_eta V = 0 is correct at augmentation-off, at zero projected '
                    'residual, and at a stationary point.')
        finally:
            for grp in g.values():
                for p in grp:
                    p.grad = saved[id(p)]
        return report

    # Kept so existing callers do not silently change behaviour; the name overstated what it did.
    assert_ownership = check_ownership

    def check_reachability(self, scale=1e-2, seed=0, tol=0.0):
        """CAN the penalty move eta, tested where a nonzero gradient is actually expected.

        Perturbs the augmentation away from whatever point the checkpoint sits at, measures the
        gradient there, and restores. A zero gradient at the reference is uninformative (see
        `check_ownership`); a zero gradient after a deliberate perturbation, with a nonzero
        penalty value, would be a real finding.

        Reports the two groups separately, because `eta_d` and `eta_a` are differently
        parameterised and their raw magnitudes are NOT comparable: a ratio between them measures
        the parameterisation as much as the geometry.
        """
        g = self.groups()
        eta = list(g['augmentation'])
        gen = torch.Generator().manual_seed(int(seed))
        saved_vals = [p.detach().clone() for p in eta]
        saved_grads = {id(p): (None if p.grad is None else p.grad.detach().clone())
                       for grp in g.values() for p in grp}
        try:
            with torch.no_grad():
                for p in eta:
                    z = torch.randn(p.shape, generator=gen, dtype=torch.float64)
                    # AN ABSOLUTE FLOOR, or the probe cannot leave the point it exists to leave.
                    # Corrected 2026-09-08 after review: the step was `scale * mean|p|`, which is
                    # exactly ZERO for an all-zero tensor -- and zero initialisation is the very
                    # degeneracy this check is meant to escape (the ANN's final layer is
                    # zero-initialised by construction). The floor makes the perturbation
                    # unconditional; it does not make a nonzero gradient guaranteed.
                    step = max(scale * float(p.detach().abs().mean()), scale)
                    p.add_(step * z.to(device=p.device, dtype=p.dtype))
            rep = self.check_ownership(tol=tol)
        finally:
            with torch.no_grad():
                for p, v in zip(eta, saved_vals):
                    p.copy_(v)
            for grp in g.values():
                for p in grp:
                    p.grad = saved_grads[id(p)]
        rep['perturbation_scale'] = scale
        # A DIAGNOSTIC, NOT A DECISION. A nonzero perturbation does not guarantee a nonzero
        # projected penalty gradient: the perturbed point can still sit where `Q^T d = 0`, or the
        # perturbation can lie in a direction the projection annihilates. A zero here is worth
        # investigating and is not by itself a defect; a decisive reachability test needs a
        # deliberately constructed non-degenerate reference, which this is not.
        rep['is_diagnostic_not_decisive'] = True
        return rep
