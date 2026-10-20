"""Terminal L-BFGS polish phase, staged after Adam (D-171)."""
__project_origin__ = "added"

import math
import time
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class PolishSpec:
    """Everything the phase can be tuned by, in one object."""
    n_windows: int = 4096
    max_iter: int = 500
    inner_iter: int = 20
    history_size: int = 50
    tol_grad: float = 1e-12
    tol_change: float = 1e-14
    seed: int = 42


def decide(val_before, val_after, *, loss_finite, validation_measure='sim-RMS'):
    """Should the polished weights be kept? Pure: no tensors, no system, no side effects."""
    if not loss_finite:
        return False, 'non-finite loss'
    have_val = np.isfinite(val_before) and np.isfinite(val_after)
    if not have_val:
        return False, ('no validation measure available and accept_without_validation=False; '
                       'refusing to keep an unchecked weight change')
    if not (val_after < val_before):
        return False, ('%s did not improve (%.6e -> %.6e)'
                       % (validation_measure, val_before, val_after))
    return True, 'accepted'


def _named_params(fit_sys):
    """[(group_name, Parameter)] over everything deepSI would hand its optimizer, DEDUPLICATED."""
    out, seen = [], set()
    for name, item in fit_sys.parameters_with_names.items():
        p_iter = [item['params']] if isinstance(item['params'], torch.nn.Parameter) \
            else list(item['params'])
        for p in p_iter:
            if id(p) not in seen:
                seen.add(id(p))
                out.append((name, p))
    return out


def _snapshot(named):
    """Clone every parameter for an exact rollback."""
    return [p.detach().clone() for _, p in named]


def _restore(named, snap):
    with torch.no_grad():
        for (_, p), saved in zip(named, snap):
            p.data.copy_(saved)


def _resync(fit_sys):
    """D-266: pair a prediction coefficient held OUTSIDE the parameters (the OBC `theta_frozen`)
    with the parameters as they are now. Every closure's loss() overwrites it, so after a restore,
    or before a validation, it may still describe another network. Keeps the basis; a no-op for a
    system without such state."""
    sync = getattr(fit_sys, '_sync_prediction_state_for_validation', None)
    if sync is not None:
        sync(refresh_basis=False)


def window_count(fit_sys, sys_data, nf, stride):
    """How many windows `make_training_data` WOULD return, without building any of them."""
    na, nb = fit_sys.na, fit_sys.nb
    na_r = getattr(fit_sys, 'na_right', 0)
    nb_r = getattr(fit_sys, 'nb_right', 0)
    records = sys_data.sdl if hasattr(sys_data, 'sdl') else [sys_data]
    total = 0
    for sd in records:
        n = len(sd.u)
        first = max(nb, na) + max(nf, na_r, nb_r)
        total += max(0, -(-(n + 1 - first) // stride))
    return total


class FixedBatch:
    """The ONE deterministic batch the phase optimises, and nothing else."""

    def __init__(self, fit_sys, sys_data, loss_kwargs, n_windows, seed, verbose=True):
        nf = loss_kwargs['nf']
        base_stride = int(loss_kwargs.get('stride', 1))
        self.n_total = window_count(fit_sys, sys_data, nf, base_stride)

        want = self.n_total if n_windows in (0, None) else min(int(n_windows), self.n_total)
        scale = max(1, self.n_total // max(1, want))
        self.effective_stride = base_stride * scale

        build_kwargs = dict(loss_kwargs)
        build_kwargs['stride'] = self.effective_stride
        built = fit_sys.make_training_data(fit_sys.norm.transform(sys_data), **build_kwargs)
        n_built = len(built[0])

        n_pred = window_count(fit_sys, sys_data, nf, self.effective_stride)
        if n_pred != n_built:
            raise RuntimeError(
                'window_count predicts %d windows at stride %d but make_training_data built %d. '
                'The two must agree, or the stride scaling that sizes this batch is guessing. '
                'deepSI\'s System_data.to_hist_future_data has changed, or na/nb/na_right/'
                'nb_right no longer mean what window_count assumes.'
                % (n_pred, self.effective_stride, n_built))

        self.n_used = min(want, n_built)
        if self.n_used < n_built:
            ix = np.random.default_rng(seed).choice(n_built, size=self.n_used, replace=False)
            ix.sort()
        else:
            ix = np.arange(n_built)

        dtype = next(fit_sys.hfn.parameters()).dtype
        device = next(fit_sys.hfn.parameters()).device

        def _to_tensor(a):
            t = torch.as_tensor(np.asarray(a)[ix])
            return (t.to(dtype) if t.is_floating_point() else t).to(device)

        cols = [_to_tensor(a) for a in built]

        names = tuple(getattr(getattr(fit_sys, 'simulator', None), 'extra_array_names', ()))

        self._cols = cols
        self._names = names
        self._loss_kwargs = dict(loss_kwargs)
        self.bytes_on_device = sum(c.element_size() * c.nelement() for c in cols)
        self.device = device

        if verbose:
            print('  [lbfgs] fixed batch: %d of %d windows (stride %d -> %d), %.1f MB on %s, '
                  'extras=%s'
                  % (self.n_used, self.n_total, base_stride, self.effective_stride,
                     self.bytes_on_device / 1024 ** 2, device, names or '()'))

    def __iter__(self):
        uh, yh, uf, yf = self._cols[:4]
        kw = dict(self._loss_kwargs)
        kw.update(zip(self._names, self._cols[4:]))
        yield uh, yh, uf, yf, kw


def _eager(fit_sys, active=True):
    """Optionally run through the EAGER rollout instead of the compiled one."""
    sim = getattr(fit_sys, 'simulator', None)
    return sim.eager() if (active and hasattr(sim, 'eager')) else nullcontext()


def _make_closure(fit_sys, batch, params, opt, counter):
    """One closure over the fixed batch, accumulating over chunks when there is more than one."""
    def closure():
        counter[0] += 1
        opt.zero_grad(set_to_none=False)
        total = 0.0
        for uh, yh, uf, yf, kw in batch:
            loss = fit_sys.loss(uh, yh, uf, yf, **kw)
            loss.backward()
            total += float(loss)
        return torch.as_tensor(total)
    return closure


def _loss_value(fit_sys, batch):
    """The objective at the CURRENT weights: one forward over the batch, no graph, no gradient."""
    with torch.no_grad():
        return float(sum(fit_sys.loss(uh, yh, uf, yf, **kw)
                         for uh, yh, uf, yf, kw in batch))


def _lbfgs_counts(opt, params):
    """(iterations, closure evaluations) as torch ITSELF counted them, cumulative over step()s."""
    st = opt.state[params[0]]
    return int(st.get('n_iter', 0)), int(st.get('func_evals', 0))


def _is_oom(exc):
    """True for a CUDA out-of-memory error, across torch versions."""
    oom_cls = getattr(torch.cuda, 'OutOfMemoryError', ())
    return isinstance(exc, oom_cls) or (isinstance(exc, RuntimeError)
                                        and 'out of memory' in str(exc).lower())


def _probe_closure(closure, label, verbose):
    """Evaluate twice without stepping; return (ok, first_value, error_or_None)."""
    try:
        a = float(closure())
        b = float(closure())
    except Exception as e:
        if verbose:
            print('  [lbfgs] %s rollout FAILED (%s): %s'
                  % (label, type(e).__name__, str(e).splitlines()[0][:160]))
        return False, float('nan'), e
    ok = (a == b)
    if verbose:
        print('  [lbfgs] determinism check (%s): %.17e vs %.17e -> %s'
              % (label, a, b, 'OK' if ok else 'FAILED'))
    return ok, a, None


def lbfgs_polish(fit_sys, train_sys_data, *, loss_kwargs, spec=None,
                 val_sys_data=None, val_before=None, validation_measure='sim-RMS',
                 force_eager=False, verbose=True):
    """Polish `fit_sys` in place with L-BFGS on one fixed batch. Returns an outcome dict."""
    spec = spec or PolishSpec()
    t0 = time.time()
    out = dict(ran=False, accepted=False, reason='', n_iter=0, max_iter=spec.max_iter,
               n_closure=0, func_evals=0, wall_s=0.0,
               loss_first=float('nan'), loss_last=float('nan'),
               val_before=float('nan'), val_after=float('nan'),
               n_windows_used=0, n_windows_total=0,
               effective_stride=0, deterministic=None, used_compiled=None,
               params_polished=0)

    named = _named_params(fit_sys)
    snap = _snapshot(named)

    def _validate():
        was_training = fit_sys.hfn.training
        fit_sys.eval()
        try:
            _resync(fit_sys)
            return float(fit_sys.cal_validation_error(val_sys_data, validation_measure))
        finally:
            if was_training:
                fit_sys.train()

    out['val_before'] = _validate() if val_before is None else float(val_before)

    params = [p for n, p in named if p.requires_grad]
    out['params_polished'] = sum(p.numel() for p in params)

    batch = FixedBatch(fit_sys, train_sys_data, loss_kwargs, spec.n_windows, spec.seed,
                       verbose=verbose)
    out.update(n_windows_used=batch.n_used, n_windows_total=batch.n_total,
               effective_stride=batch.effective_stride)

    def _new_opt():
        return torch.optim.LBFGS(
            params, lr=1.0, max_iter=spec.inner_iter, history_size=spec.history_size,
            tolerance_grad=spec.tol_grad, tolerance_change=spec.tol_change,
            line_search_fn='strong_wolfe')

    n_closure = [0]
    sim = getattr(fit_sys, 'simulator', None)
    compiled_available = sim is not None and getattr(sim, '_compiled', None) is not None

    with _eager(fit_sys, active=force_eager):
        opt = _new_opt()
        closure = _make_closure(fit_sys, batch, params, opt, n_closure)
        label = 'eager' if (force_eager or not compiled_available) else 'compiled'
        ok, first, err = _probe_closure(closure, label, verbose)
        out['deterministic'], out['used_compiled'] = ok, (label == 'compiled' and ok)

    if not ok and not force_eager and compiled_available:
        print('  [lbfgs] falling back to the EAGER rollout (%s). The strong-Wolfe line '
              'search compares f at several points, so it needs a loss that both runs and '
              'reproduces.'
              % ('the compiled rollout raised' if err is not None
                 else 'the compiled loss is not reproducible'))
        force_eager = True

    with _eager(fit_sys, active=force_eager):
        if not out['deterministic']:
            opt = _new_opt()
            closure = _make_closure(fit_sys, batch, params, opt, n_closure)
            ok, first, err = _probe_closure(closure, 'eager', verbose)
            out['deterministic'], out['used_compiled'] = ok, False
        if not ok:
            out['reason'] = (
                'the eager rollout raised (%s: %s)' % (type(err).__name__, err)
                if err is not None else
                'the loss is not reproducible even eagerly on this device; a line search '
                'cannot be run. Try torch.use_deterministic_algorithms(True).')
            out['wall_s'] = time.time() - t0
            _restore(named, snap)
            _resync(fit_sys)
            if verbose:
                print('  [lbfgs] ABORTED, weights unchanged: %s' % out['reason'])
            return out

        out['loss_first'] = first
        if verbose:
            print('  [lbfgs] %d parameters in one group, m=%d, strong-Wolfe, '
                  'tol_grad=%.1e tol_change=%.1e (%.1e relative to the opening loss), '
                  'max %d iterations, %s rollout'
                  % (out['params_polished'], spec.history_size, spec.tol_grad,
                     spec.tol_change, spec.tol_change / max(abs(first), 1e-300),
                     spec.max_iter, 'compiled' if out['used_compiled'] else 'eager'))

        n_outer = max(1, int(math.ceil(spec.max_iter / max(1, spec.inner_iter))))
        for k in range(n_outer):
            it_before, _ = _lbfgs_counts(opt, params)
            try:
                opt.step(closure)
            except Exception as e:
                kind = 'CUDA out of memory' if _is_oom(e) else type(e).__name__
                out['n_iter'], out['func_evals'] = _lbfgs_counts(opt, params)
                out['reason'] = ('%s after %d iterations; phase abandoned, weights restored. %s'
                                 % (kind, out['n_iter'],
                                    'Lower lbfgs_windows, or set lbfgs_chunk.'
                                    if _is_oom(e) else str(e).splitlines()[0][:200]))
                out['wall_s'] = time.time() - t0
                out['n_closure'] = n_closure[0]
                _restore(named, snap)
                _resync(fit_sys)
                if _is_oom(e):
                    torch.cuda.empty_cache()
                if verbose:
                    print('  [lbfgs] ABORTED, weights restored: %s' % out['reason'])
                return out
            out['n_iter'], out['func_evals'] = _lbfgs_counts(opt, params)
            out['n_closure'] = n_closure[0]
            done = out['n_iter'] - it_before
            if verbose:
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
        out['n_closure'] = n_closure[0]
        if not np.isfinite(out['loss_last']):
            out['reason'] += ' | non-finite loss at the final point'

    out['ran'] = True
    out['val_after'] = _validate()
    out['wall_s'] = time.time() - t0

    out['accepted'], why = decide(
        out['val_before'], out['val_after'],
        loss_finite=bool(np.isfinite(out['loss_last'])),
        validation_measure=validation_measure)
    out['reason'] = '%s | %s' % (out['reason'], why)

    if not out['accepted']:
        _restore(named, snap)
        _resync(fit_sys)
    if verbose:
        print('  [lbfgs] %s: %s' % ('ACCEPTED' if out['accepted'] else 'ROLLED BACK', why))
        print('  [lbfgs] loss %.6e -> %.6e | %d iterations, %d closures in %.0f s'
              % (out['loss_first'], out['loss_last'], out['n_iter'], out['n_closure'],
                 out['wall_s']))
        if np.isfinite(out['val_before']):
            print('  [lbfgs] %s %.6e -> %.6e'
                  % (validation_measure, out['val_before'], out['val_after']))
    return out
