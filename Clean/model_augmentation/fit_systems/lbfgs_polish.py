"""The L-BFGS polish after Adam: one fixed batch of windows, kept only if the validation score improves."""
__project_origin__ = "added"

import time
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch


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


def _snapshot(params):
    """Clone every parameter for an exact rollback."""
    return [p.detach().clone() for p in params]


def _restore(params, snap):
    with torch.no_grad():
        for p, saved in zip(params, snap):
            p.data.copy_(saved)


def _resync(fit_sys):
    """Pair theta_frozen of the OBC arm with the parameters as they are now, keeping the basis; every loss()
    overwrites it, so after a restore it can describe another network. A no-op without OBC."""
    fit_sys._sync_obc_coefficient(refresh_basis=False)


def window_count(fit_sys, sys_data, nf, stride):
    """How many windows make_training_data would return, without building them."""
    first = max(fit_sys.nb, fit_sys.na) + nf
    return sum(len(range(first, len(sd.u) + 1, stride)) for sd in sys_data.sdl)


class FixedWindowBatch:
    """The one deterministic batch the polish optimises: windows at a stride scaled to about n_windows, then a
    seeded subset if there are still more."""

    def __init__(self, fit_sys, sys_data, loss_kwargs, n_windows, seed):
        nf = loss_kwargs['nf']
        base_stride = loss_kwargs['stride']
        self.n_total = window_count(fit_sys, sys_data, nf, base_stride)

        want = min(n_windows, self.n_total)
        scale = max(1, self.n_total // max(1, want))
        self.effective_stride = base_stride * scale

        build_kwargs = dict(loss_kwargs)
        build_kwargs['stride'] = self.effective_stride
        built = fit_sys.make_training_data(fit_sys.norm.transform(sys_data), **build_kwargs)
        n_built = len(built[0])

        n_pred = window_count(fit_sys, sys_data, nf, self.effective_stride)
        if n_pred != n_built:
            raise RuntimeError(
                'window_count predicts %d windows at stride %d but make_training_data built %d; make '
                'window_count follow the windowing of WindowedSystemData'
                % (n_pred, self.effective_stride, n_built))

        self.n_used = min(want, n_built)
        if self.n_used < n_built:
            ix = np.random.default_rng(seed).choice(n_built, size=self.n_used, replace=False)
            ix.sort()
        else:
            ix = np.arange(n_built)

        p = next(fit_sys.hfn.parameters())

        def _to_tensor(a):
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
    """The L-BFGS closure over the fixed batch; counter[0] counts its calls."""
    def closure():
        counter[0] += 1
        opt.zero_grad(set_to_none=False)
        loss = fit_sys.loss(*batch.args, **batch.kw)
        loss.backward()
        # a detached value, so the graph is freed before L-BFGS calls again
        return torch.as_tensor(float(loss))
    return closure


def _loss_value(fit_sys, batch):
    """The objective at the current weights: one forward pass over the batch, without gradient."""
    with torch.no_grad():
        return float(fit_sys.loss(*batch.args, **batch.kw))


def _lbfgs_counts(opt, params):
    """(iterations, closure evaluations) as torch counts them, cumulative over step() calls."""
    st = opt.state[params[0]]
    return int(st.get('n_iter', 0)), int(st.get('func_evals', 0))


def _is_oom(exc):
    """True for a CUDA out-of-memory error."""
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


def lbfgs_polish(fit_sys, train_sys_data, *, loss_kwargs, spec, val_sys_data, val_before=None,
                 validation_measure):
    """Polish fit_sys in place with L-BFGS on one fixed batch; returns the outcome dict."""
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
    snap = _snapshot(params)
    batch = opt = closure = failure = None
    force_eager = False

    def _validate():
        was_training = fit_sys.hfn.training
        fit_sys.eval()
        try:
            _resync(fit_sys)
            return float(fit_sys.cal_validation_error(val_sys_data, validation_measure))
        finally:
            if was_training:
                fit_sys.train()

    def _abort(reason, oom, word):
        """Abandon the polish: restore and resync the start weights, and release the CUDA cache after an
        out-of-memory error; the pre-polish result stands."""
        out['reason'] = reason
        out['wall_s'] = time.time() - t0
        _restore(params, snap)
        _resync(fit_sys)
        if oom:
            torch.cuda.empty_cache()
        print('  [lbfgs] ABORTED, weights %s: %s' % (word, reason))
        return out

    # one restore-and-resync boundary from the start validation to the final scoring: an exception the inner
    # handlers do not catch aborts the polish with the weights restored
    try:
        # start validation and the fixed batch
        out['val_before'] = _validate() if val_before is None else float(val_before)

        out['params_polished'] = sum(p.numel() for p in params)

        batch = FixedWindowBatch(fit_sys, train_sys_data, loss_kwargs, spec.n_windows, spec.seed)
        out.update(n_windows_used=batch.n_used, n_windows_total=batch.n_total,
                   effective_stride=batch.effective_stride)

        def _new_opt():
            return torch.optim.LBFGS(
                params, lr=1.0, max_iter=spec.inner_iter, history_size=spec.history_size,
                tolerance_grad=spec.tol_grad, tolerance_change=spec.tol_change,
                line_search_fn='strong_wolfe')

        # determinism probe, compiled first if available: the strong-Wolfe line search needs a loss that
        # reproduces
        n_closure = [0]
        compiled_available = fit_sys.simulator._compiled is not None

        opt = _new_opt()
        closure = _make_closure(fit_sys, batch, opt, n_closure)
        label = 'compiled' if compiled_available else 'eager'
        ok, first, err = _probe_closure(closure, label)
        out['deterministic'], out['used_compiled'] = ok, (label == 'compiled' and ok)

        if not ok and compiled_available:
            print('  [lbfgs] falling back to the eager rollout (%s): the strong-Wolfe line search compares f at '
                  'several points, so it needs a loss that both runs and reproduces'
                  % ('the compiled rollout raised' if err is not None
                     else 'the compiled loss is not reproducible'))
            force_eager = True
            # the failed attempt's optimizer and allocations go before the eager retry
            opt = closure = None
            if err is not None and _is_oom(err):
                torch.cuda.empty_cache()

        with (fit_sys.simulator.eager() if force_eager else nullcontext()):
            if not out['deterministic']:
                opt = _new_opt()
                closure = _make_closure(fit_sys, batch, opt, n_closure)
                ok, first, err = _probe_closure(closure, 'eager')
                out['deterministic'], out['used_compiled'] = ok, False
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

            # L-BFGS in blocks of inner_iter iterations, until it stops inside a block or max_iter is reached
            n_outer = max(1, -(-spec.max_iter // max(1, spec.inner_iter)))
            for k in range(n_outer):
                it_before, _ = _lbfgs_counts(opt, params)
                try:
                    opt.step(closure)
                except Exception as e:
                    out['n_iter'], out['func_evals'] = _lbfgs_counts(opt, params)
                    out['n_closure'] = n_closure[0]
                    return _abort('%s after %d iterations; phase abandoned, weights restored. %s'
                                  % ('CUDA out of memory' if _is_oom(e) else type(e).__name__,
                                     out['n_iter'],
                                     'Lower lbfgs_windows.' if _is_oom(e)
                                     else str(e).splitlines()[0][:200]),
                                  _is_oom(e), 'restored')
                out['n_iter'], out['func_evals'] = _lbfgs_counts(opt, params)
                out['n_closure'] = n_closure[0]
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

            out['loss_last'] = _loss_value(fit_sys, batch)
            if not np.isfinite(out['loss_last']):
                out['reason'] += ' | non-finite loss at the final point'

        # final validation
        out['ran'] = True
        out['val_after'] = _validate()
        out['wall_s'] = time.time() - t0
    except Exception as e:
        failure = (_is_oom(e), 'CUDA out of memory' if _is_oom(e) else type(e).__name__,
                   str(e).splitlines()[0][:200] if str(e) else '')
    if failure is not None:
        # handled here, once the except block has released the failed frames and their tensors
        batch = opt = closure = None
        out.update(ran=False, accepted=False, val_after=float('nan'))
        return _abort('%s in the polish; phase abandoned, weights restored. %s' % failure[1:],
                      failure[0], 'restored')

    # keep the polished weights only with a finite loss and a validation score that improved
    if not np.isfinite(out['loss_last']):
        out['accepted'], why = False, 'non-finite loss'
    elif not (np.isfinite(out['val_before']) and np.isfinite(out['val_after'])):
        out['accepted'], why = False, ('no validation measure available; refusing to keep an '
                                       'unchecked weight change')
    elif not (out['val_after'] < out['val_before']):
        out['accepted'], why = False, ('%s did not improve (%.6e -> %.6e)'
                                       % (validation_measure, out['val_before'], out['val_after']))
    else:
        out['accepted'], why = True, 'accepted'
    out['reason'] = '%s | %s' % (out['reason'], why)

    if not out['accepted']:
        _restore(params, snap)
        _resync(fit_sys)
    print('  [lbfgs] %s: %s' % ('ACCEPTED' if out['accepted'] else 'ROLLED BACK', why))
    print('  [lbfgs] loss %.6e -> %.6e | %d iterations, %d closures in %.0f s'
          % (out['loss_first'], out['loss_last'], out['n_iter'], out['n_closure'],
             out['wall_s']))
    if np.isfinite(out['val_before']):
        print('  [lbfgs] %s %.6e -> %.6e'
              % (validation_measure, out['val_before'], out['val_after']))
    return out
