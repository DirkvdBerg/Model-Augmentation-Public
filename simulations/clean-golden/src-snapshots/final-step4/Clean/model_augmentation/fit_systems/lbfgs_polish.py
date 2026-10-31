"""The polish: L-BFGS on V over one fixed set of windows from the selected parameters, kept only if the validation
error drops."""
__project_origin__ = "added"

import time
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch

from model_augmentation.fit_systems.closed_loop import window_counts


@dataclass(frozen=True)
class PolishSpec:
    """The settings of the polish."""
    n_windows: int
    max_iter: int
    inner_iter: int
    history_size: int
    tol_grad: float
    tol_change: float
    seed: int


def _restore(fit_sys, params, snap):
    """Restore the parameter snapshot and recompute theta_frozen for that network with the existing OBC basis
    (every loss() overwrites theta_frozen, so after a restore it can describe another network); no OBC work on the
    unconstrained arm."""
    with torch.no_grad():
        for p, saved in zip(params, snap):
            p.data.copy_(saved)
    fit_sys._sync_obc_coefficient(refresh_basis=False)


class FixedWindowBatch:
    """The one deterministic batch the polish optimises: windows at a stride scaled to about n_windows, then a
    seeded subset if there are still more."""

    def __init__(self, fit_sys, sys_data, loss_kwargs, n_windows, seed):
        """The windows at the scaled stride, and a seeded subset of n_windows if there are more."""
        nf = loss_kwargs['nf']
        base_stride = loss_kwargs['stride']
        self.n_total = sum(window_counts(fit_sys, sys_data.sdl, nf, base_stride))

        want = min(n_windows, self.n_total)
        scale = max(1, self.n_total // max(1, want))
        self.effective_stride = base_stride * scale

        build_kwargs = dict(loss_kwargs)
        build_kwargs['stride'] = self.effective_stride
        # augment_training_data checks the built windows against window_counts at this stride
        built = fit_sys.make_training_data(fit_sys.norm.transform(sys_data), **build_kwargs)
        n_built = len(built[0])

        self.n_used = min(want, n_built)
        if self.n_used < n_built:
            ix = np.random.default_rng(seed).choice(n_built, size=self.n_used, replace=False)
            ix.sort()
        else:
            ix = np.arange(n_built)

        p = next(fit_sys.hfn.parameters())

        def _to_tensor(a):
            """The selected windows of one array, on the model's device and dtype."""
            t = torch.as_tensor(np.asarray(a)[ix])
            return (t.to(p.dtype) if t.is_floating_point() else t).to(p.device)

        cols = [_to_tensor(a) for a in built]
        names = fit_sys.simulator.extra_array_names
        # loss(*args, **kw): the four window arrays, then the loss kwargs and the simulator's extras
        self.args = tuple(cols[:4])
        self.kw = {**loss_kwargs, **dict(zip(names, cols[4:]))}

        print('  [lbfgs] fixed batch: %d of %d windows (stride %d -> %d), %.1f MB on %s, '
              'extras=%s'
              % (self.n_used, self.n_total, base_stride, self.effective_stride,
                 sum(c.element_size() * c.nelement() for c in cols) / 1024 ** 2, p.device,
                 names or '()'))


def _make_closure(fit_sys, batch, opt, counter):
    """L-BFGS closure: V and its gradient on the fixed batch; counter[0] counts its calls."""
    def closure():
        """Write grad V to the parameters' .grad slots and return a detached scalar V on the fixed batch."""
        counter[0] += 1
        opt.zero_grad(set_to_none=False)
        loss = fit_sys.loss(*batch.args, **batch.kw)
        loss.backward()
        # a detached value, so the graph is freed before L-BFGS calls again
        return torch.as_tensor(float(loss))
    return closure


def _lbfgs_counts(opt, params):
    """(iterations, closure evaluations) as torch counts them, cumulative over step() calls."""
    st = opt.state[params[0]]
    return int(st.get('n_iter', 0)), int(st.get('func_evals', 0))


def _is_oom(exc):
    """Recognise CUDA OOM exceptions and RuntimeErrors containing 'out of memory'."""
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (isinstance(exc, RuntimeError)
                                                           and 'out of memory' in str(exc).lower())


def _probe_closure(closure, label):
    """Evaluate the closure twice without a step; returns (identical, first value, error or None)."""
    try:
        a = float(closure())
        b = float(closure())
    except Exception as e:
        print('  [lbfgs] %s rollout FAILED (%s): %s'
              % (label, type(e).__name__, str(e).splitlines()[0][:160]))
        # without its traceback, so the frames and tensors of the failed rollout are freed
        return False, float('nan'), e.with_traceback(None)
    ok = (a == b)
    print('  [lbfgs] determinism check (%s): %.17e vs %.17e -> %s'
          % (label, a, b, 'OK' if ok else 'FAILED'))
    return ok, a, None


def _acceptance(out, validation_measure):
    """(keep, why): keep the polished weights only if V is finite and the validation error drops."""
    if not np.isfinite(out['loss_last']):
        return False, 'non-finite loss'
    if not (np.isfinite(out['val_before']) and np.isfinite(out['val_after'])):
        return False, 'no validation measure available; refusing to keep an unchecked weight change'
    if not (out['val_after'] < out['val_before']):
        return False, '%s did not improve (%.6e -> %.6e)' % (validation_measure, out['val_before'], out['val_after'])
    return True, 'accepted'


def lbfgs_polish(fit_sys, train_sys_data, *, loss_kwargs, spec, val_sys_data, val_before=None,
                 validation_measure):
    """Polish of the selected model.

    L-BFGS (strong Wolfe) on V over one fixed batch of windows, rolled back unless the validation error drops; it
    refines the Adam result without risking a worse model.

    Args:
        fit_sys: the augmented model, changed in place; train_sys_data, val_sys_data: records.
        loss_kwargs: nf and stride; spec: PolishSpec; val_before: score to beat; validation_measure: its name.
    Returns:
        dict with the outcome (accepted, iterations, losses, validation before and after).
    """
    t0 = time.time()
    out = dict(ran=False, accepted=False, reason='', n_iter=0, max_iter=spec.max_iter,
               n_closure=0, func_evals=0, wall_s=0.0,
               loss_first=float('nan'), loss_last=float('nan'),
               val_before=float('nan'), val_after=float('nan'),
               n_windows_used=0, n_windows_total=0,
               effective_stride=0, deterministic=None, used_compiled=None,
               params_polished=0)

    # every parameter deepSI hands its optimizer, in its order (encoder, then hfn)
    params = [p for item in fit_sys.parameters_with_names.values() for p in item['params']]
    # the start weights, for an exact rollback
    snap = [p.detach().clone() for p in params]
    batch = opt = closure = failure = None
    n_closure = [0]

    def _validate():
        """The validation error at the current weights, theta_aux synced first."""
        was_training = fit_sys.hfn.training
        fit_sys.eval()
        try:
            fit_sys._sync_obc_coefficient(refresh_basis=False)
            return float(fit_sys.cal_validation_error(val_sys_data, validation_measure))
        finally:
            if was_training:
                fit_sys.train()

    def _record_counts():
        """Store the L-BFGS iteration and closure counts in out."""
        out['n_iter'], out['func_evals'] = _lbfgs_counts(opt, params)
        out['n_closure'] = n_closure[0]

    def _abort(reason, oom, word):
        """Abandon the polish: restore and resync the start weights, and release the CUDA cache after an
        out-of-memory error; the pre-polish result stands."""
        out['reason'] = reason
        out['wall_s'] = time.time() - t0
        _restore(fit_sys, params, snap)
        if oom:
            torch.cuda.empty_cache()
        print('  [lbfgs] ABORTED, weights %s: %s' % (word, reason))
        return out

    # one restore-and-resync boundary from the start validation to the final scoring: an exception aborts the
    # polish with the weights restored
    try:
        # 1. Score to beat and the fixed batch.
        out['val_before'] = _validate() if val_before is None else float(val_before)

        out['params_polished'] = sum(p.numel() for p in params)

        batch = FixedWindowBatch(fit_sys, train_sys_data, loss_kwargs, spec.n_windows, spec.seed)
        out.update(n_windows_used=batch.n_used, n_windows_total=batch.n_total,
                   effective_stride=batch.effective_stride)

        def _new_opt():
            """A fresh L-BFGS optimizer with the spec's settings, strong-Wolfe line search."""
            return torch.optim.LBFGS(
                params, lr=1.0, max_iter=spec.inner_iter, history_size=spec.history_size,
                tolerance_grad=spec.tol_grad, tolerance_change=spec.tol_change,
                line_search_fn='strong_wolfe')

        # 2. Optimizer and determinism probe.
        # determinism probe, compiled first if available: the strong-Wolfe line search needs a loss that
        # reproduces
        compiled_available = fit_sys.simulator._compiled is not None

        opt = _new_opt()
        closure = _make_closure(fit_sys, batch, opt, n_closure)
        ok, first, err = _probe_closure(closure, 'compiled' if compiled_available else 'eager')

        # a failed compiled probe is retried once on the eager rollout
        force_eager = compiled_available and not ok
        if force_eager:
            print('  [lbfgs] falling back to the eager rollout (%s): the strong-Wolfe line search compares f at '
                  'several points, so it needs a loss that both runs and reproduces'
                  % ('the compiled rollout raised' if err is not None
                     else 'the compiled loss is not reproducible'))
            # the failed attempt's optimizer and allocations go before the eager retry
            opt = closure = None
            if err is not None and _is_oom(err):
                torch.cuda.empty_cache()

        with (fit_sys.simulator.eager() if force_eager else nullcontext()):
            if force_eager:
                opt = _new_opt()
                closure = _make_closure(fit_sys, batch, opt, n_closure)
                ok, first, err = _probe_closure(closure, 'eager')
            out['deterministic'], out['used_compiled'] = ok, compiled_available and not force_eager
            if not ok:
                return _abort('the eager rollout raised (%s: %s)' % (type(err).__name__, err)
                              if err is not None else
                              'the loss is not reproducible even eagerly on this device; a line search '
                              'cannot be run. Try torch.use_deterministic_algorithms(True).',
                              False, 'unchanged')

            out['loss_first'] = first
            print('  [lbfgs] %d parameters in one group, m=%d, strong-Wolfe, '
                  'tol_grad=%.1e tol_change=%.1e (%.1e relative to the opening loss), '
                  'max %d iterations, %s rollout'
                  % (out['params_polished'], spec.history_size, spec.tol_grad,
                     spec.tol_change, spec.tol_change / max(abs(first), 1e-300),
                     spec.max_iter, 'compiled' if out['used_compiled'] else 'eager'))

            # 3. L-BFGS iterations.
            # allow ceil(max_iter / inner_iter) blocks, stopping early if a block uses fewer than inner_iter iterations
            n_outer = max(1, -(-spec.max_iter // max(1, spec.inner_iter)))
            for _ in range(n_outer):
                it_before, _ = _lbfgs_counts(opt, params)
                opt.step(closure)
                _record_counts()
                done = out['n_iter'] - it_before
                print('  [lbfgs] iter %4d (+%d of %d)  (%d closures, %.0f s)'
                      % (out['n_iter'], done, spec.inner_iter, n_closure[0],
                         time.time() - t0))
                if done < spec.inner_iter:
                    out['reason'] = ('converged: L-BFGS stopped internally after %d of %d '
                                     'iterations in this block (tol_grad=%.1e, tol_change=%.1e)'
                                     % (done, spec.inner_iter, spec.tol_grad, spec.tol_change))
                    break
            else:
                out['reason'] = 'iteration cap %d reached' % spec.max_iter

            # the objective at the final weights: one forward pass over the batch, without gradient
            with torch.no_grad():
                out['loss_last'] = float(fit_sys.loss(*batch.args, **batch.kw))
            if not np.isfinite(out['loss_last']):
                out['reason'] += ' | non-finite loss at the final point'

        out['ran'] = True
        out['val_after'] = _validate()
        out['wall_s'] = time.time() - t0
    except Exception as e:
        if opt is not None:
            _record_counts()
        failure = (_is_oom(e), 'CUDA out of memory' if _is_oom(e) else type(e).__name__, out['n_iter'],
                   'Lower lbfgs_windows.' if _is_oom(e) else str(e).splitlines()[0][:200] if str(e) else '')
    if failure is not None:
        # handled here, once the except block has released the failed frames and their tensors
        batch = opt = closure = None
        out.update(ran=False, accepted=False, val_after=float('nan'))
        return _abort('%s after %d iterations in the polish; phase abandoned, weights restored. %s' % failure[1:],
                      failure[0], 'restored')

    # 4. Keep the polish only if the validation error dropped.
    out['accepted'], why = _acceptance(out, validation_measure)
    out['reason'] = '%s | %s' % (out['reason'], why)

    if not out['accepted']:
        _restore(fit_sys, params, snap)
    print('  [lbfgs] %s: %s' % ('ACCEPTED' if out['accepted'] else 'ROLLED BACK', why))
    print('  [lbfgs] loss %.6e -> %.6e | %d iterations, %d closures in %.0f s'
          % (out['loss_first'], out['loss_last'], out['n_iter'], out['n_closure'],
             out['wall_s']))
    if np.isfinite(out['val_before']):
        print('  [lbfgs] %s %.6e -> %.6e'
              % (validation_measure, out['val_before'], out['val_after']))
    return out
