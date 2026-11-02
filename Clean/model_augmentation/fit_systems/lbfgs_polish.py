"""The polish: L-BFGS on the training loss V over one fixed set of windows, from the selected parameters, kept only
if the validation error drops."""
__project_origin__ = "added"

import time
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch

from model_augmentation.fit_systems.closed_loop import window_counts


@dataclass(frozen=True)
class PolishSpec:
    """Settings of the polish; values come from the run config (lbfgs_*)."""
    n_windows: int       # windows in the fixed batch; fewer lowers memory use
    max_iter: int        # L-BFGS iterations in total
    inner_iter: int      # iterations per step() call; a call that stops early ends the polish
    history_size: int    # curvature pairs kept by L-BFGS
    tol_grad: float      # stop when the largest loss gradient is below this
    tol_change: float    # stop when the loss or the step changes less than this
    seed: int            # seed of the random window subset


def _restore(fit_sys, params, snap):
    """Restore the parameter snapshot and recompute theta_aux for it with the existing OBC basis."""
    with torch.no_grad():
        for p, saved in zip(params, snap):
            p.copy_(saved)
    fit_sys._sync_obc_coefficient(refresh_basis=False)


class FixedWindowBatch:
    """The one fixed batch the polish optimises; L-BFGS needs the same loss at every evaluation, so no minibatches."""

    def __init__(self, fit_sys, sys_data, loss_kwargs, n_windows, seed):
        """Windows at a stride scaled to give about n_windows, then a random subset of n_windows if there are more.

        Args:
            fit_sys: the augmented model.
            sys_data: training records.
            loss_kwargs: nf (window length [samples]) and stride (samples between window starts).
            n_windows: wanted number of windows.
            seed: seed of the subset.
        """
        # 1. Stride: the base stride times the factor that brings the window count down to about n_windows.
        nf = loss_kwargs['nf']
        base_stride = loss_kwargs['stride']
        self.n_total = sum(window_counts(fit_sys, sys_data.sdl, nf, base_stride))

        want = min(n_windows, self.n_total)
        scale = max(1, self.n_total // max(1, want))
        self.effective_stride = base_stride * scale

        # 2. Build the windows at that stride, as for Adam.
        build_kwargs = dict(loss_kwargs)
        build_kwargs['stride'] = self.effective_stride
        built = fit_sys.make_training_data(fit_sys.norm.transform(sys_data), **build_kwargs)
        n_built = len(built[0])

        # 3. If still more than wanted, a random subset, kept in the original order.
        self.n_used = min(want, n_built)
        if self.n_used < n_built:
            ix = np.random.default_rng(seed).choice(n_built, size=self.n_used, replace=False)
            ix.sort()
        else:
            ix = np.arange(n_built)

        # 4. Keep the batch as tensors on the model's device, ready as loss(*args, **kw).
        p = next(fit_sys.hfn.parameters())

        def _to_tensor(a):
            """The selected windows of one array, on the model's device and dtype."""
            t = torch.as_tensor(np.asarray(a)[ix])
            return (t.to(p.dtype) if t.is_floating_point() else t).to(p.device)

        cols = [_to_tensor(a) for a in built]
        names = fit_sys.simulator.extra_array_names
        # args: uhist, yhist, ufuture, yfuture; kw: the loss kwargs and the simulator's extras (ctrl_ix, loss_w)
        self.args = tuple(cols[:4])
        # kw keeps the base stride; the rollout does not use it
        self.kw = {**loss_kwargs, **dict(zip(names, cols[4:]))}

        print('  [lbfgs] fixed batch: %d of %d windows (stride %d -> %d), %.1f MB on %s, '
              'extras=%s'
              % (self.n_used, self.n_total, base_stride, self.effective_stride,
                 sum(c.element_size() * c.nelement() for c in cols) / 1024 ** 2, p.device,
                 names or '()'))


def _make_closure(fit_sys, batch, opt, counter):
    """L-BFGS closure: V and its gradient on the fixed batch; counter[0] counts its calls."""
    def closure():
        """Write grad V to .grad and return V on the fixed batch, detached."""
        counter[0] += 1
        opt.zero_grad(set_to_none=False)
        loss = fit_sys.loss(*batch.args, **batch.kw)
        loss.backward()
        # detached, so the graph is freed before the next call
        return torch.as_tensor(float(loss))
    return closure


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
        # drop the traceback so the tensors of the failed rollout are freed
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

    L-BFGS (strong Wolfe) on V over one fixed batch of windows, rolled back unless the validation error drops, so it
    can only improve on the Adam result.

    Args:
        fit_sys: the augmented model, changed in place.
        train_sys_data, val_sys_data: training and validation records.
        loss_kwargs: nf and stride.
        spec: PolishSpec.
        val_before: score to beat; None measures it.
        validation_measure: name of the score.
    Returns:
        dict with the outcome: ran, accepted, deterministic, used_compiled, iteration and closure counts, loss and
        validation before and after, and reason.
    """
    t0 = time.time()
    out = dict(ran=False, accepted=False, reason='', n_iter=0,
               n_closure=0, func_evals=0, wall_s=0.0,
               loss_first=float('nan'), loss_last=float('nan'),
               val_before=float('nan'), val_after=float('nan'),
               n_windows_used=0, n_windows_total=0,
               effective_stride=0, deterministic=None, used_compiled=None,
               params_polished=0)

    # all trainable parameters in deepSI's order (encoder, then hfn), one group; the OBC basis stays at its beta_lin
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
            return float(fit_sys.cal_validation_error(val_sys_data))
        finally:
            if was_training:
                fit_sys.train()

    def _record_counts():
        """Store the L-BFGS iteration and closure counts in out."""
        out['n_iter'], out['func_evals'] = _lbfgs_counts(opt, params)
        out['n_closure'] = n_closure[0]

    def _abort(reason, oom, word):
        """Abandon the polish and restore the start weights; the pre-polish result stands."""
        out['reason'] = reason
        out['wall_s'] = time.time() - t0
        _restore(fit_sys, params, snap)
        if oom:
            torch.cuda.empty_cache()
        print('  [lbfgs] ABORTED, weights %s: %s' % (word, reason))
        return out

    # any exception from here on aborts the polish with the weights restored
    try:
        # 1. Score to beat (measured here if not given) and the fixed batch of training windows.
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

        # 2. Determinism check: the strong-Wolfe line search compares the loss at several points, so evaluating it
        #    twice must give the same value. Tried on the compiled rollout first if there is one, else eager.
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
            # not reproducible even eagerly: abort with the weights unchanged
            if not ok:
                return _abort('the eager rollout raised (%s: %s)' % (type(err).__name__, err)
                              if err is not None else
                              'the loss is not reproducible even eagerly on this device; a line search '
                              'cannot be run. Try torch.use_deterministic_algorithms(True).',
                              False, 'unchanged')

            out['loss_first'] = first
            # HEURISTIC: 1e-300 guards the printed ratio tol_change / |V| against V = 0
            print('  [lbfgs] %d parameters in one group, m=%d, strong-Wolfe, '
                  'tol_grad=%.1e tol_change=%.1e (%.1e relative to the opening loss), '
                  'max %d iterations, %s rollout'
                  % (out['params_polished'], spec.history_size, spec.tol_grad,
                     spec.tol_change, spec.tol_change / max(abs(first), 1e-300),
                     spec.max_iter, 'compiled' if out['used_compiled'] else 'eager'))

            # 3. L-BFGS in blocks of inner_iter iterations, logged after each block, up to max_iter in total;
            #    a block that stops early has met tol_grad or tol_change and ends the polish.
            # number of blocks: ceil(max_iter / inner_iter)
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

        # validation score of the polished weights, compared in step 4
        out['ran'] = True
        out['val_after'] = _validate()
        out['wall_s'] = time.time() - t0
    except Exception as e:
        if opt is not None:
            _record_counts()
        failure = (_is_oom(e), 'CUDA out of memory' if _is_oom(e) else type(e).__name__, out['n_iter'],
                   'Lower lbfgs_windows.' if _is_oom(e) else str(e).splitlines()[0][:200] if str(e) else '')
    if failure is not None:
        # handled after the except block, so the tensors of the failed step are released
        batch = opt = closure = None
        out.update(ran=False, accepted=False, val_after=float('nan'))
        return _abort('%s after %d iterations in the polish; phase abandoned, weights restored. %s' % failure[1:],
                      failure[0], 'restored')

    # 4. Keep the polished weights only if the loss is finite and the validation score dropped; else restore.
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
