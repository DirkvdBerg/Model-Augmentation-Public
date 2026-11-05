"""Refine the selected model with L-BFGS on fixed training windows; accept only if validation improves.
L-BFGS needs no bounds here, since every beta gives an admissible baseline."""
__project_origin__ = "added"

import time
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch

from model_augmentation.fit_systems.closed_loop import window_counts


HISTORY_SIZE = 50    # past parameter and gradient changes used to approximate curvature
INNER_ITER = 20      # maximum iterations per optimizer call
# Dividing output errors by the servo error increased the loss and largest absolute gradient
# by about 2.6e7 on 128 training windows. We use this measured factor to scale the stopping
# tolerances from 1e-12 and 1e-14, set below the unscaled loss of about 1e-9, to 2.6e-5 and 2.6e-7.
TOL_GRAD = 2.6e-5    # threshold for the largest absolute gradient entry
TOL_CHANGE = 2.6e-7  # shared threshold for loss change, parameter step and directional derivative


@dataclass(frozen=True)
class PolishSpec:
    """Run settings for the fixed training batch and L-BFGS iteration budget."""
    n_windows: int       # maximum windows in the fixed batch
    max_iter: int        # iteration budget, rounded up to blocks of INNER_ITER (at least one block)
    seed: int            # seed for selecting windows


def _restore(fit_sys, params, snap):
    """Restore starting parameters and recompute theta_aux using the unchanged OBC basis."""
    with torch.no_grad():
        for p, saved in zip(params, snap):
            p.copy_(saved)
    fit_sys.sync_obc_coefficient(refresh_basis=False)


class FixedWindowBatch:
    """Keep training windows fixed so line-search evaluations compare the same objective."""

    def __init__(self, fit_sys, sys_data, loss_kwargs, n_windows, seed):
        """Increase the stride to limit window construction, then select at most n_windows.

        Args:
            fit_sys: the augmented model.
            sys_data: training records.
            loss_kwargs: training-data and loss options, including nf and stride in samples.
            n_windows: maximum number of windows.
            seed: seed for sampling without replacement.
        """
        # Reduce the number of windows built before selecting the fixed batch.
        nf = loss_kwargs['nf']
        base_stride = loss_kwargs['stride']
        self.n_total = sum(window_counts(fit_sys, sys_data.sdl, nf, base_stride))

        want = min(n_windows, self.n_total)
        scale = max(1, self.n_total // max(1, want))
        self.effective_stride = base_stride * scale

        build_kwargs = dict(loss_kwargs)
        build_kwargs['stride'] = self.effective_stride
        built = fit_sys.make_training_data(fit_sys.norm.transform(sys_data), **build_kwargs)
        n_built = len(built[0])

        # Sample without replacement and preserve the original window order.
        self.n_used = min(want, n_built)
        if self.n_used < n_built:
            ix = np.random.default_rng(seed).choice(n_built, size=self.n_used, replace=False)
            ix.sort()
        else:
            ix = np.arange(n_built)

        p = next(fit_sys.hfn.parameters())

        def _to_tensor(a):
            """Move selected windows to the model's device; preserve integer index types."""
            t = torch.as_tensor(np.asarray(a)[ix])
            return (t.to(p.dtype) if t.is_floating_point() else t).to(p.device)

        cols = [_to_tensor(a) for a in built]
        names = fit_sys.simulator.extra_array_names
        # The loss takes four history/future arrays, plus simulator extras such as ctrl_ix and loss_w.
        self.args = tuple(cols[:4])
        # Stride controls window construction; the loss does not use it.
        self.kw = {**loss_kwargs, **dict(zip(names, cols[4:]))}

        print('  [lbfgs] fixed batch: %d of %d windows (stride %d -> %d), %.1f MB on %s, '
              'extras=%s'
              % (self.n_used, self.n_total, base_stride, self.effective_stride,
                 sum(c.element_size() * c.nelement() for c in cols) / 1024 ** 2, p.device,
                 names or '()'))


def _make_closure(fit_sys, batch, opt, counter):
    """Build a loss-and-gradient evaluation; counter[0] includes probes and line-search trials."""
    def closure():
        counter[0] += 1
        opt.zero_grad(set_to_none=False)
        loss = fit_sys.loss(*batch.args, **batch.kw)
        loss.backward()
        # Return only the loss value so the optimizer does not retain its computation graph.
        return torch.as_tensor(float(loss))
    return closure


def _lbfgs_counts(opt, params):
    """Return cumulative optimizer iterations and loss evaluations; exclude separate closure probes."""
    st = opt.state[params[0]]
    return int(st.get('n_iter', 0)), int(st.get('func_evals', 0))


def _is_oom(exc):
    """Recognise torch out-of-memory errors and RuntimeErrors containing 'out of memory'."""
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (isinstance(exc, RuntimeError)
                                                           and 'out of memory' in str(exc).lower())


def _probe_closure(closure, label):
    """Compare two losses at unchanged parameters; return (equal, first loss, exception or None).

    Both calls compute gradients, but only the loss values are compared.
    """
    try:
        a = float(closure())
        b = float(closure())
    except Exception as e:
        print('  [lbfgs] %s rollout FAILED (%s): %s'
              % (label, type(e).__name__, str(e).splitlines()[0][:160]))
        # Release traceback references to tensors from the failed evaluation.
        return False, float('nan'), e.with_traceback(None)
    ok = (a == b)
    print('  [lbfgs] determinism check (%s): %.17e vs %.17e -> %s'
          % (label, a, b, 'OK' if ok else 'FAILED'))
    return ok, a, None


def _acceptance(out, validation_measure):
    """Return (accept, reason): require finite final loss and validation scores, with lower validation error."""
    if not np.isfinite(out['loss_last']):
        return False, 'non-finite loss'
    if not (np.isfinite(out['val_before']) and np.isfinite(out['val_after'])):
        return False, 'no validation measure available; refusing to keep an unchecked weight change'
    if not (out['val_after'] < out['val_before']):
        return False, '%s did not improve (%.6e -> %.6e)' % (validation_measure, out['val_before'], out['val_after'])
    return True, 'accepted'


def lbfgs_polish(fit_sys, train_sys_data, *, loss_kwargs, spec, val_sys_data, val_before=None,
                 validation_measure):
    """Refine the model with L-BFGS on fixed training windows, choosing the step size by line search.

    Restore starting parameters on failure or rejection. Acceptance requires a finite final training loss and
    finite validation scores with a strict improvement; the training loss need not decrease.

    Args:
        fit_sys: the augmented model, changed in place.
        train_sys_data, val_sys_data: training and validation records.
        loss_kwargs: training-data and loss options, including nf and stride in samples.
        spec: fixed-batch size, iteration budget and sampling seed.
        val_before: validation score to beat; None evaluates the starting model.
        validation_measure: score name for messages; fit_sys computes the score.
    Returns:
        dict with acceptance, reason, losses, validation scores, window counts, optimizer counts and runtime.
        n_closure includes probes; func_evals counts optimizer evaluations only. deterministic records exact
        equality of the two probe losses. Final scores describe the candidate even when it is rolled back.
    """
    t0 = time.time()
    out = dict(ran=False, accepted=False, reason='', n_iter=0,
               n_closure=0, func_evals=0, wall_s=0.0,
               loss_first=float('nan'), loss_last=float('nan'),
               val_before=float('nan'), val_after=float('nan'),
               n_windows_used=0, n_windows_total=0,
               effective_stride=0, deterministic=None, used_compiled=None,
               params_polished=0)

    # Optimise the encoder and model together; keep the OBC basis fixed at beta_lin.
    params = [p for item in fit_sys.parameters_with_names.values() for p in item['params']]
    # Save parameters before any trial steps so failure or rejection can restore them.
    snap = [p.detach().clone() for p in params]
    batch = opt = closure = failure = None
    n_closure = [0]

    def _validate():
        """Evaluate validation after updating theta_aux; restore the previous training mode."""
        was_training = fit_sys.hfn.training
        fit_sys.eval()
        try:
            fit_sys.sync_obc_coefficient(refresh_basis=False)
            return float(fit_sys.cal_validation_error(val_sys_data))
        finally:
            if was_training:
                fit_sys.train()

    def _record_counts():
        out['n_iter'], out['func_evals'] = _lbfgs_counts(opt, params)
        out['n_closure'] = n_closure[0]

    def _abort(reason, oom, word):
        """Restore starting parameters and report why the polish failed."""
        out['reason'] = reason
        out['wall_s'] = time.time() - t0
        _restore(fit_sys, params, snap)
        if oom:
            torch.cuda.empty_cache()
        print('  [lbfgs] ABORTED, weights %s: %s' % (word, reason))
        return out

    try:
        out['val_before'] = _validate() if val_before is None else float(val_before)

        out['params_polished'] = sum(p.numel() for p in params)

        batch = FixedWindowBatch(fit_sys, train_sys_data, loss_kwargs, spec.n_windows, spec.seed)
        out.update(n_windows_used=batch.n_used, n_windows_total=batch.n_total,
                   effective_stride=batch.effective_stride)

        def _new_opt():
            return torch.optim.LBFGS(
                params, lr=1.0, max_iter=INNER_ITER, history_size=HISTORY_SIZE,
                tolerance_grad=TOL_GRAD, tolerance_change=TOL_CHANGE,
                line_search_fn='strong_wolfe')

        # A line search needs repeatable evaluations. Probe the active rollout before taking a step.
        compiled_available = fit_sys.simulator._compiled is not None

        opt = _new_opt()
        closure = _make_closure(fit_sys, batch, opt, n_closure)
        ok, first, err = _probe_closure(closure, 'compiled' if compiled_available else 'eager')

        # Retry without compilation if the compiled rollout raises or returns unequal probe losses.
        force_eager = compiled_available and not ok
        if force_eager:
            print('  [lbfgs] falling back to the eager rollout (%s): the strong-Wolfe line search compares f at '
                  'several points, so it needs a loss that both runs and reproduces'
                  % ('the compiled rollout raised' if err is not None
                     else 'the compiled loss is not reproducible'))
            # Release the first optimizer before allocating the eager retry.
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
            # Round the budget up to full optimizer calls, with at least one call.
            n_outer = max(1, -(-spec.max_iter // max(1, INNER_ITER)))
            # Avoid division by zero in the displayed tolerance-to-loss ratio.
            print('  [lbfgs] %d parameters in one group, m=%d, strong-Wolfe, '
                  'tol_grad=%.1e tol_change=%.1e (%.1e relative to the opening loss), '
                  'max %d iterations, %s rollout'
                  % (out['params_polished'], HISTORY_SIZE, TOL_GRAD,
                     TOL_CHANGE, TOL_CHANGE / max(abs(first), 1e-300),
                     n_outer * INNER_ITER, 'compiled' if out['used_compiled'] else 'eager'))

            # A call can stop on a tolerance or its evaluation limit; the optimizer does not report which.
            for _ in range(n_outer):
                it_before, _ = _lbfgs_counts(opt, params)
                opt.step(closure)
                _record_counts()
                done = out['n_iter'] - it_before
                print('  [lbfgs] iter %4d (+%d of %d)  (%d closures, %.0f s)'
                      % (out['n_iter'], done, INNER_ITER, n_closure[0],
                         time.time() - t0))
                if done < INNER_ITER:
                    out['reason'] = ('L-BFGS stopped after %d of %d iterations in this call; '
                                     'internal stopping condition not reported'
                                     % (done, INNER_ITER))
                    break
            else:
                out['reason'] = 'iteration cap %d reached' % (n_outer * INNER_ITER)

            # Measure the final loss explicitly: step() can return the loss from before its updates.
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
        # Leave the exception handler and release failed-step tensors before restoring parameters.
        batch = opt = closure = None
        out.update(ran=False, accepted=False, val_after=float('nan'))
        return _abort('%s after %d iterations in the polish; phase abandoned, weights restored. %s' % failure[1:],
                      failure[0], 'restored')

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
