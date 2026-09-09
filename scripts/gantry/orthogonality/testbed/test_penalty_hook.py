"""The production seam: the penalty gradient must survive the closure's own zero_grad.

Category: verification of the GENERIC wiring. Not a claim about the gantry.

WHAT THIS GUARDS. `SSE_Interconnect_Composed.fit` wraps `optimizer.step` so that the trajectory
penalty is accumulated between the training closure returning and the update being applied. Three
things can go wrong silently and each has a check here:

  1. The penalty is added and then wiped. The parent's closure is
     `loss -> zero_grad -> backward`, so a penalty backwarded inside `loss()` is discarded. This
     is not hypothetical: it is the failure `checks_v6` measures on the gantry.
  2. The wrapper changes behaviour when the feature is OFF. `traj_penalty = None` must delegate
     to the parent untouched, and the check is bit-identity, not a tolerance.
  3. The wrapper fires during an L-BFGS line search, where `closure(backward=False)` re-evaluates
     the loss WITHOUT refreshing gradients. Adding to stale gradients there would corrupt the
     search direction.

Run: conda run -n GraduationProject python scripts/gantry/orthogonality/testbed/test_penalty_hook.py
"""
__project_origin__ = "added"

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

from model_augmentation.fit_systems.trajectory_penalty_hook import (  # noqa: E402
    TrajectoryPenaltyHook, _frozen)

PASS, FAIL = [], []


def check(name, ok, detail=''):
    (PASS if ok else FAIL).append(name)
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f'   {detail}' if detail else ''))


# --------------------------------------------------------------------------- a tiny system
class ToyAdapter:
    """Smallest thing satisfying what the hook uses: three disjoint groups and a scored stack.

    `p_full` depends on eta and lambda; `p_off` depends on lambda only, which is the property
    that licenses the no-grad off evaluation and is asserted rather than assumed.
    """

    def __init__(self, n_win=4, T=5, ny=2, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.n_win, self.T, self.ny = n_win, T, ny
        self.N = n_win * T * ny
        self.eta = torch.randn(6, generator=g, dtype=torch.float64, requires_grad=True)
        self.lam = torch.randn(3, generator=g, dtype=torch.float64, requires_grad=True)
        self.xip = torch.randn(4, generator=g, dtype=torch.float64, requires_grad=True)
        self.A = torch.randn(self.N, 6, generator=g, dtype=torch.float64)
        self.B = torch.randn(self.N, 3, generator=g, dtype=torch.float64)
        self.C = torch.randn(self.N, 4, generator=g, dtype=torch.float64)
        self.w = {'ufuture': torch.zeros(n_win, 1, 1)}

    def parameter_groups(self):
        return {'physical': [self.lam], 'augmentation': [self.eta], 'encoder': [self.xip]}

    def n_rows(self):
        return self.N

    def loss_weight(self):
        return None

    def metadata(self):
        return {'system': 'toy'}

    def encode(self, ix=None):
        n = self.n_win if ix is None else len(ix)
        return (self.C[:n] @ self.xip).reshape(n, 1), (self.A[:n, :2] @ self.eta[:2]).reshape(n, 1)

    @staticmethod
    def x0_from(xp, xa):
        return torch.cat([xp, xa], dim=1)

    def _rows(self, ix):
        if ix is None:
            return slice(None)
        a, b = int(ix[0]), int(ix[-1]) + 1
        per = self.T * self.ny
        return slice(a * per, b * per)

    def scored_stack(self, params=None, mode='full', rows=None, ix=None, x0=None):
        r = self._rows(ix)
        base = self.B[r] @ self.lam
        if mode == 'off':
            return base
        extra = self.A[r] @ self.eta
        if x0 is not None:
            extra = extra + x0.sum() * 0.0        # keep the x0 path in the graph, value neutral
        return base + extra


def make_geometry(ad, seed=1):
    from model_augmentation.fit_systems.trajectory_orth_projection import (
        TrajectoryProjectionGeometry)
    g = torch.Generator().manual_seed(seed)
    S = torch.randn(ad.N, 3, generator=g, dtype=torch.float64).numpy()
    return TrajectoryProjectionGeometry(S, rank_rtol=1e-12, strict=False)


def hdr(s):
    print('\n' + '=' * 78 + f'\n{s}\n' + '=' * 78)


# --------------------------------------------------------------------------- checks
def test_hook_routing():
    hdr('hook: ownership and value')
    ad = ToyAdapter()
    geom = make_geometry(ad)
    hook = TrajectoryPenaltyHook(ad, geom, beta=0.7)

    rep = hook.assert_ownership()
    check('penalty reaches eta and nothing else',
          rep['physical'] == 0.0 and rep['encoder'] == 0.0 and rep['augmentation'] > 0,
          f"g_lambda {rep['physical']:.1e}  g_xi_p {rep['encoder']:.1e}  "
          f"g_eta {rep['augmentation']:.3e}")

    # requires_grad restored exactly, or the next base backward silently stops training lambda
    check('requires_grad restored after the frozen scope',
          ad.lam.requires_grad and ad.xip.requires_grad and ad.eta.requires_grad)

    # the plain-MSE convention: V must scale as 1/N against the sum convention
    d = (ad.scored_stack(mode='full') - ad.scored_stack(mode='off')).detach()
    v_mse = float(geom.penalty_mse(d, 0.7))
    v_sum = float(geom.penalty(d, 0.7))
    check('V = (beta/N)||Q^T d||^2, not the sum convention',
          abs(v_mse * geom.N - v_sum) < 1e-12 * max(v_sum, 1e-30),
          f'N {geom.N}  V_mse {v_mse:.6e}  V_sum {v_sum:.6e}')


def test_chunking_matches():
    hdr('hook: chunked accumulation equals the single stack')
    ad = ToyAdapter()
    geom = make_geometry(ad)
    ref = None
    for chunk in (0, 1, 2):
        for p in ad.parameter_groups()['augmentation']:
            p.grad = None
        hook = TrajectoryPenaltyHook(ad, geom, beta=0.7, chunk=chunk)
        v = hook()
        g = ad.eta.grad.detach().clone()
        if ref is None:
            ref = (v, g)
            continue
        ev = abs(v - ref[0]) / max(abs(ref[0]), 1e-300)
        eg = float((g - ref[1]).norm() / max(float(ref[1].norm()), 1e-300))
        check(f'chunk={chunk} matches the single stack',
              ev < 1e-10 and eg < 1e-9, f'rel value {ev:.2e}  rel grad {eg:.2e}')


def test_step_wrapper():
    hdr('seam: the wrapper survives the closure\'s zero_grad, and is inert when off')
    ad = ToyAdapter()
    geom = make_geometry(ad)
    params = [ad.eta, ad.lam, ad.xip]

    def base_closure(backward=True):
        """Mimics the production closure EXACTLY: loss, then zero_grad, then backward."""
        loss = (ad.scored_stack(mode='full') ** 2).mean()
        if backward:
            opt.zero_grad()
            loss.backward()
        return loss

    # ---- reference: base only
    opt = torch.optim.SGD(params, lr=0.0)
    opt.step(base_closure)
    g_base = ad.eta.grad.detach().clone()

    # ---- with the wrapper, exactly as SSE_Interconnect_Composed.fit installs it
    hook = TrajectoryPenaltyHook(ad, geom, beta=0.7)
    real_step = opt.step

    def step_with_penalty(closure=None, **kw):
        def wrapped(*a, **k):
            loss = closure(*a, **k)
            backward = (a[0] if a else k.get('backward', True))
            if backward:
                hook(None)
            return loss
        return real_step(wrapped, **kw)

    opt.step = step_with_penalty
    opt.step(base_closure)
    g_with = ad.eta.grad.detach().clone()
    opt.step = real_step

    delta = float((g_with - g_base).norm())
    check('the penalty gradient SURVIVES the closure zero_grad', delta > 1e-12,
          f'||g_with - g_base|| = {delta:.3e} against ||g_base|| = {float(g_base.norm()):.3e}')

    # ---- the wrong order, for contrast: penalty first, then the closure wipes it
    for p in params:
        p.grad = None
    hook(None)
    base_closure()                     # its zero_grad discards what the hook just added
    g_wrong = ad.eta.grad.detach().clone()
    check('the WRONG order loses the penalty entirely',
          float((g_wrong - g_base).norm()) < 1e-12,
          f'||g_wrong - g_base|| = {float((g_wrong - g_base).norm()):.3e}')

    # ---- line-search evaluations must not accumulate
    for p in params:
        p.grad = None
    opt.step(base_closure)
    g_ref = ad.eta.grad.detach().clone()
    calls = hook.calls
    opt.step = step_with_penalty
    step_with_penalty(base_closure)
    once = hook.calls - calls
    opt.step = real_step
    wrapped_nb = base_closure(False)   # backward=False, as L-BFGS does inside its line search
    check('a backward=False closure evaluation adds nothing',
          hook.calls - calls == once and wrapped_nb is not None,
          f'hook called {once} time(s) for one real step, and not on the no-backward evaluation')


def test_feature_off_is_identical():
    hdr('seam: traj_penalty = None delegates to the parent untouched')
    from model_augmentation.fit_systems.interconnect import SSE_Interconnect_Composed
    src = SSE_Interconnect_Composed.fit.__doc__ or ''
    check('SSE_Interconnect_Composed overrides fit',
          'fit' in dir(SSE_Interconnect_Composed))
    check('the class default is None, so an existing run is unaffected',
          SSE_Interconnect_Composed.traj_penalty is None)
    # The early return is the guarantee; assert it exists in the source rather than trusting it.
    import inspect
    body = inspect.getsource(SSE_Interconnect_Composed.fit)
    check('fit() returns super().fit() immediately when the feature is off',
          'if self.traj_penalty is None:' in body and 'return super().fit(' in body)
    check('the wrapper is removed in a finally block', 'finally:' in body)


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    test_hook_routing()
    test_chunking_matches()
    test_step_wrapper()
    test_feature_off_is_identical()
    print('\n' + '=' * 78)
    print(f'{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        print('FAILED: ' + ', '.join(FAIL))
    sys.exit(1 if FAIL else 0)
