"""The ACTUAL `SSE_Interconnect_Composed.fit` override, driven with a controlled parent.

WHY THIS FILE EXISTS. `test_penalty_hook.py` reproduces the step-wrapper logic locally and
inspects `fit()`'s source for strings like `finally:`. Review called that what it is: another
reference-to-production gap, the same class of gap that let `compressed_nuisance_basis` ship with
`Gamma_w` omitted while a suite reported the construction verified. A test that re-implements the
thing under test cannot fail when the thing under test is wrong.

So this file calls `SSE_Interconnect_Composed.fit` itself. The PARENT `fit` is replaced by a
minimal stand-in that reproduces only the three behaviours the override contracts with:

    1. it may replace `self.optimizer` before stepping   (deepSI `_refresh_optimizer`, which does
       `self.optimizer = optimizer_new` on a system loaded from a file)
    2. it calls `optimizer.step(closure)` with a closure of the shape
       `loss -> zero_grad -> backward -> return loss`
    3. it may call `closure(backward=False)`             (an L-BFGS line search)

Replacing the parent rather than running deepSI's real loop is deliberate: the real loop needs
data, a norm, a validation set and minutes of rollout, and none of that exercises the override
any harder than these three behaviours do. What is NOT stubbed is the override itself.

Run: conda run -n GraduationProject python scripts/gantry/orthogonality/testbed/test_fit_override.py
"""
__project_origin__ = "added"

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

from model_augmentation.fit_systems.interconnect import (  # noqa: E402
    SSE_Interconnect_Composed)

PASS, FAIL = [], []


def check(name, ok, detail=''):
    (PASS if ok else FAIL).append(name)
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f'   {detail}' if detail else ''))


class CountingHook:
    """Stands in for TrajectoryPenaltyHook: adds a KNOWN constant to eta, and counts calls."""

    def __init__(self, eta, delta=0.25):
        self.eta, self.delta, self.calls = eta, delta, 0

    def groups(self):
        return {'physical': [], 'augmentation': list(self.eta), 'encoder': []}

    def __call__(self, fit_sys=None, *a, **k):
        self.calls += 1
        for p in self.eta:
            g = torch.full_like(p, self.delta)
            p.grad = g if p.grad is None else p.grad + g
        return float(self.delta)


class Harness(SSE_Interconnect_Composed):
    """Only `fit` is stubbed; the override under test is inherited unchanged."""

    def __init__(self, n=3, refresh_before_step=False, line_search=False, steps=2):
        # deliberately NOT calling SSE_Interconnect_Composed.__init__: it wants an Interconnect.
        self.p = torch.nn.Parameter(torch.ones(n, dtype=torch.float64))
        self.traj_penalty = None
        self._refresh_before_step = refresh_before_step
        self._line_search = line_search
        self._steps = steps
        self.base_grad = 2.0
        self._optimizer = torch.optim.SGD([self.p], lr=0.0)
        self.parent_calls = 0

    # ---- the minimal parent -------------------------------------------------------
    def _parent_fit(self, *a, **k):
        self.parent_calls += 1
        for _ in range(self._steps):
            if self._refresh_before_step:
                # EXACTLY what deepSI's _refresh_optimizer does: build a new optimizer over the
                # same parameters and rebind the attribute.
                new = torch.optim.SGD([self.p], lr=0.0)
                new.load_state_dict(self._optimizer.state_dict())
                self.optimizer = new
                self._refresh_before_step = False

            def closure(backward=True):
                loss = (self.p ** 2).sum()
                if backward:
                    self.optimizer.zero_grad()
                    self.p.grad = torch.full_like(self.p, self.base_grad)
                return loss

            if self._line_search:
                self.optimizer.step(closure)
                closure(False)              # a no-backward re-evaluation, as L-BFGS does
            else:
                self.optimizer.step(closure)
        return 'fitted'


# `super().fit(...)` inside the override must reach the stand-in.
SSE_Interconnect_Composed.__bases__[0].fit = Harness._parent_fit


def run(**kw):
    h = Harness(**kw)
    hook = CountingHook([h.p])
    h.traj_penalty = hook
    out = h.fit()
    return h, hook, out


def test_feature_off_is_untouched():
    h = Harness()
    h.traj_penalty = None
    h.fit()
    check('traj_penalty=None delegates to the parent and leaves gradients as the base set them',
          float(h.p.grad.min()) == 2.0 and float(h.p.grad.max()) == 2.0,
          f'grad {float(h.p.grad[0])}')
    check('no wrapper is left on the optimizer when the feature is off',
          not getattr(h._optimizer, '_traj_wrapped', False))


def test_penalty_reaches_the_gradient_through_the_real_override():
    h, hook, out = run(steps=2)
    check('the parent still ran', out == 'fitted' and h.parent_calls == 1)
    check('the hook fired once per optimizer step', hook.calls == 2, f'{hook.calls} calls')
    # base 2.0 set INSIDE the closure after zero_grad, penalty +0.25 added after it returns
    check('the final gradient is base + penalty, i.e. the penalty survived zero_grad',
          abs(float(h.p.grad[0]) - 2.25) < 1e-12, f'grad {float(h.p.grad[0])}')
    check('the wrapper is removed when fit returns',
          not getattr(h._optimizer, '_traj_wrapped', False))


def test_optimizer_refresh_mid_fit_is_survived():
    """THE DEFECT THIS FILE WAS WRITTEN FOR.

    The first version wrapped `self.optimizer` before `super().fit()`. deepSI replaces that
    object on a resumed system, so the wrapper was dropped and the penalty silently never fired
    for the rest of the run.
    """
    h, hook, _ = run(refresh_before_step=True, steps=2)
    check('the penalty still fires after the optimizer is REPLACED mid-fit',
          hook.calls == 2, f'{hook.calls} calls across 2 steps')
    check('the gradient still carries the penalty after the refresh',
          abs(float(h.p.grad[0]) - 2.25) < 1e-12, f'grad {float(h.p.grad[0])}')
    check('the replacement optimizer is the one that was wrapped',
          getattr(h._optimizer, '_traj_real_step', None) is None,
          'unwrapped cleanly on exit')


def test_reconciliation_catches_a_lost_wrapper():
    """If a future change loses the wrapper, fit must RAISE rather than train without it."""
    h = Harness(steps=2)
    hook = CountingHook([h.p])
    h.traj_penalty = hook

    # Simulate the failure: a replacement optimizer that the property does not re-wrap.
    orig_wrap = h._wrap_optimizer_step
    calls = {'n': 0}

    def wrap_once(opt):
        calls['n'] += 1
        return opt if calls['n'] > 1 else orig_wrap(opt)

    h._wrap_optimizer_step = wrap_once
    h._refresh_before_step = True
    raised = ''
    try:
        h.fit()
    except RuntimeError as e:
        raised = str(e)
    check('a lost wrapper is DETECTED rather than silently training without the penalty',
          'NOT the wrapped one' in raised, raised[:90])
    # WHY THIS NEEDLE AND NOT A COUNT. Reconciling hook calls against a step counter cannot see
    # this failure: the counter lives inside the wrapper, so losing the wrapper zeroes BOTH sides
    # and the comparison passes vacuously. That was the first guard, and this test is what showed
    # it was useless. The working check is the state of the optimizer the fit ended with.


def test_line_search_evaluation_adds_nothing():
    h, hook, _ = run(line_search=True, steps=2)
    check('a backward=False re-evaluation does not accumulate, through the real wrapper',
          hook.calls == 2, f'{hook.calls} calls for 2 steps plus 2 no-backward evaluations')


def test_unsupported_modes_are_refused():
    for name, setup, needle in (
        ('L-BFGS', lambda h: setattr(h, '_optimizer', torch.optim.LBFGS([h.p])), 'L-BFGS'),
        ('AMP', lambda h: setattr(h, 'scaler', object()), 'mixed precision'),
        ('no optimizer', lambda h: setattr(h, '_optimizer', None), 'init_model'),
    ):
        h = Harness()
        h.traj_penalty = CountingHook([h.p])
        setup(h)
        msg = ''
        try:
            h.fit()
        except RuntimeError as e:
            msg = str(e)
        check(f'{name} is refused with an explanation', needle in msg, msg[:80])

    h = Harness()
    h.traj_penalty = CountingHook([h.p])
    msg = ''
    try:
        h.fit(concurrent_val=True)
    except RuntimeError as e:
        msg = str(e)
    check('concurrent_val is refused', 'concurrent_val' in msg, msg[:80])


def test_step_without_closure_is_refused():
    h = Harness()
    h.traj_penalty = CountingHook([h.p])
    h._traj_wrapping = True
    h._wrap_optimizer_step(h._optimizer)
    msg = ''
    try:
        h._optimizer.step()
    except RuntimeError as e:
        msg = str(e)
    check('stepping without a closure is refused', 'without a closure' in msg, msg[:70])


def test_replacement_optimizer_is_validated_too():
    """Unsupported-mode validation must happen where an optimizer is ACCEPTED, not only at entry.

    Added 2026-09-08 after review. The rejection lived at the top of `fit()`, so a REPLACEMENT
    optimizer -- the whole reason the property exists -- was wrapped without being checked. An
    L-BFGS refresh would have been accepted mid-run by the very mechanism added to make refreshes
    safe.
    """
    h = Harness(steps=1)
    h.traj_penalty = CountingHook([h.p])
    h._traj_wrapping = True
    for cls, needle in ((torch.optim.LBFGS, 'L-BFGS'), (torch.optim.Adadelta, 'not been analysed')):
        msg = ''
        try:
            h.optimizer = cls([h.p])          # goes through the property -> _wrap_optimizer_step
        except RuntimeError as e:
            msg = str(e)
        check(f'a replacement {cls.__name__} is refused at assignment', needle in msg, msg[:70])
    h._traj_wrapping = False

    # and an allowed one is accepted and wrapped
    h._traj_wrapping = True
    h.optimizer = torch.optim.Adam([h.p])
    check('an allowed replacement optimizer is wrapped',
          getattr(h._optimizer, '_traj_wrapped', False))
    h._unwrap_optimizer_step(h._optimizer)
    h._traj_wrapping = False


def test_legacy_pickle_keeps_its_optimizer():
    """A system pickled BEFORE `optimizer` became a property must not lose it.

    `optimizer` is now a data descriptor, and a data descriptor SHADOWS an instance `__dict__`
    entry of the same name. Without migration, unpickling an old system leaves the real optimizer
    stranded in `__dict__['optimizer']` while `self.optimizer` reads `self._optimizer`, which is
    None. The penalty path would then refuse to start, and a caller tolerating None would train
    with a freshly initialised optimizer while believing it had restored moments.
    """
    h = Harness()
    real = torch.optim.Adam([h.p])
    legacy_state = dict(h.__dict__)
    legacy_state.pop('_optimizer', None)
    legacy_state['optimizer'] = real          # exactly how an old pickle looks

    revived = Harness.__new__(Harness)
    revived.__setstate__(legacy_state)
    check('a legacy `optimizer` entry is migrated into the backing field',
          revived.optimizer is real)
    check('the stale __dict__ entry is removed, so the descriptor is not shadowed',
          'optimizer' not in revived.__dict__)

    # a modern pickle round-trips unchanged
    h2 = Harness()
    modern = dict(h2.__dict__)
    revived2 = Harness.__new__(Harness)
    revived2.__setstate__(modern)
    check('a modern state dict round-trips', revived2.optimizer is h2._optimizer)

def _swap_super_load(fn):
    """Replace the PARENT's checkpoint_load_system for the duration of one call."""
    base = SSE_Interconnect_Composed.__bases__[0]
    had = hasattr(base, 'checkpoint_load_system')
    saved = getattr(base, 'checkpoint_load_system', None)
    base.checkpoint_load_system = fn
    return base, had, saved


def _restore_super_load(base, had, saved):
    if had:
        base.checkpoint_load_system = saved
    else:
        del base.checkpoint_load_system


def test_checkpoint_load_during_fit_keeps_the_wrapper():
    """deepSI loads the best checkpoint at the END of every fit, and mid-run too.

    `checkpoint_load_system` does `self.__dict__ = torch.load(file)`, installing a freshly
    unpickled optimizer that carries no wrapper. Two failures came from that, one visible and
    one not: the end-of-fit reconciliation declared a CORRECT run invalid (3 penalty
    applications across 3 steps), and a load DURING a fit would have trained the remainder with
    no penalty at all while reporting that it had one. The load now re-wraps while a fit is in
    progress.
    """
    h = Harness(steps=2)
    hook = CountingHook([h.p])
    h.traj_penalty = hook
    h._traj_wrapping = True
    h._wrap_optimizer_step(h._optimizer)

    replacement = torch.optim.Adam([h.p])

    def fake_load(self, *a, **k):
        self.__dict__['_optimizer'] = replacement      # what the real one does, in miniature
        return 'loaded'

    ctx = _swap_super_load(fake_load)
    try:
        h.checkpoint_load_system()
    finally:
        _restore_super_load(*ctx)

    check('a load during a fit re-wraps the replacement optimizer',
          getattr(h._optimizer, '_traj_wrapped', False),
          'otherwise the rest of the run trains with no penalty')
    check('the penalty hook survives the __dict__ replacement', h.traj_penalty is hook)
    h._traj_wrapping = False
    h._unwrap_optimizer_step(h._optimizer)


def test_checkpoint_load_outside_a_fit_leaves_a_plain_optimizer():
    h = Harness()
    h.traj_penalty = CountingHook([h.p])
    replacement = torch.optim.Adam([h.p])

    def fake_load(self, *a, **k):
        self.__dict__['_optimizer'] = replacement
        return 'loaded'

    ctx = _swap_super_load(fake_load)
    try:
        h.checkpoint_load_system()
    finally:
        _restore_super_load(*ctx)
    check('a load OUTSIDE a fit leaves the optimizer unwrapped',
          not getattr(h._optimizer, '_traj_wrapped', False),
          'a restored system should look plain until a fit starts')


def test_save_does_not_pickle_the_hook():
    """`checkpoint_save_system` pickles `self.__dict__`, so __getstate__ never runs."""
    h = Harness()
    h.traj_penalty = CountingHook([h.p])
    seen = {}

    base = SSE_Interconnect_Composed.__bases__[0]
    had = hasattr(base, 'checkpoint_save_system')
    saved = getattr(base, 'checkpoint_save_system', None)

    def fake_save(self, *a, **k):
        seen['keys'] = set(self.__dict__)
        return 'saved'

    base.checkpoint_save_system = fake_save
    try:
        h.checkpoint_save_system()
    finally:
        if had:
            base.checkpoint_save_system = saved
        else:
            del base.checkpoint_save_system

    check('traj_penalty is absent from what gets pickled',
          'traj_penalty' not in seen.get('keys', set()),
          'deepSI pickles self.__dict__, so __getstate__ is never consulted')
    check('the hook is put back after the save', h.traj_penalty is not None)


def test_checkpoint_load_rebinds_the_adapter():
    """A load must re-resolve the adapter's cached modules, not just restore the hook.

    THE DEFECT THIS PINS. `GantryTrajectoryAdapter.__init__` caches the encoder, the ANN block and
    the physical block as snapshots. `checkpoint_load_system` does `self.__dict__ = torch.load(f)`,
    which keeps the system object's identity but replaces every module inside it. Restoring the
    hook therefore leaves the adapter spliced across two generations: it drives the NEW dynamics
    through `fit_sys.simulate` while gating the OLD ANN block and owning the OLD parameters.

    That does not raise, and it is not a tolerance issue. The `off` intervention would gate a
    block outside the rollout, so the intervention becomes incorrect while still producing a
    plausible value (it does not vanish: the latent initialisation is zeroed independently of the
    gate), and the gradients land on tensors the live model no longer contains. The load now calls `rebind` before anything else.
    """
    class FakeAdapter:
        def __init__(self):
            self.bound_to, self.rebinds = None, 0

        def rebind(self, fit_sys=None):
            self.bound_to, self.rebinds = fit_sys, self.rebinds + 1
            return self

        def assert_bound(self):
            return True

        def parameter_groups(self):
            return {'eta_d': []}

    h = Harness()
    hook = CountingHook([h.p])
    ad = FakeAdapter()
    hook.adapter = ad
    hook._groups = {'eta_d': ['STALE']}
    h.traj_penalty = hook

    def fake_load(self, *a, **k):
        self.__dict__['_optimizer'] = torch.optim.Adam([h.p])
        return 'loaded'

    ctx = _swap_super_load(fake_load)
    try:
        h.checkpoint_load_system()
    finally:
        _restore_super_load(*ctx)

    check('the load rebinds the adapter exactly once', ad.rebinds == 1, f'{ad.rebinds} rebind(s)')
    check('the adapter is rebound to the loaded system', ad.bound_to is h)
    check("the hook's cached parameter groups are rebuilt from the loaded model",
          hook._groups == {'eta_d': []},
          'a cached group list is stale for the same reason the module references are')


def test_load_without_an_adapter_is_harmless():
    """Not every hook carries an adapter; the rebind must not assume one."""
    h = Harness()
    h.traj_penalty = CountingHook([h.p])

    def fake_load(self, *a, **k):
        self.__dict__['_optimizer'] = torch.optim.Adam([h.p])
        return 'loaded'

    ctx = _swap_super_load(fake_load)
    ok = True
    try:
        h.checkpoint_load_system()
    except Exception as e:                                          # noqa: BLE001
        ok = False
        print('        ' + repr(e))
    finally:
        _restore_super_load(*ctx)
    check('a hook with no adapter loads without error', ok)

if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    print('=' * 78 + '\nthe real fit() override\n' + '=' * 78)
    test_feature_off_is_untouched()
    test_penalty_reaches_the_gradient_through_the_real_override()
    test_optimizer_refresh_mid_fit_is_survived()
    test_reconciliation_catches_a_lost_wrapper()
    test_line_search_evaluation_adds_nothing()
    test_unsupported_modes_are_refused()
    test_step_without_closure_is_refused()
    test_replacement_optimizer_is_validated_too()
    test_legacy_pickle_keeps_its_optimizer()
    test_checkpoint_load_during_fit_keeps_the_wrapper()
    test_checkpoint_load_outside_a_fit_leaves_a_plain_optimizer()
    test_save_does_not_pickle_the_hook()
    test_checkpoint_load_rebinds_the_adapter()
    test_load_without_an_adapter_is_harmless()
    print('\n' + '=' * 78)
    print(f'{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        print('FAILED: ' + ', '.join(FAIL))
    sys.exit(1 if FAIL else 0)
