"""Terminal L-BFGS polish phase, staged after Adam (D-171).

Adam explores, a quasi-Newton phase converges. That split is the recipe of Bemporad's
`jax_sysid` (IEEE TAC 70(7):4857-4864, 2025) and of Drenth's LPV-LFR thesis, which this
project's method descends from: a fixed number of Adam iterations, then a MAXIMUM number of
L-BFGS-B iterations that normally ends on its own tolerances. `docs/references.md`, section
"Adam to L-BFGS handover", carries the evidence; D-171 the decisions;
`tasks/lbfgs-known-issues.md` the review this revision implements.

WHY THIS IS A SEPARATE PHASE AND NOT AN OPTIMIZER SWAP INSIDE `fit()`
--------------------------------------------------------------------
deepSI's update loop draws a fresh random minibatch every iteration (`interconnect.py:910`).
L-BFGS cannot live there. It builds curvature from `y = g(x+s) - g(x)`, a difference of two
gradients, and Byrd, Hansen, Nocedal and Singer (SIOPT 26(2):1008-1031, 2016) reject
differencing gradients taken on different samples outright; Bollapragada et al. (ICML 2018)
reach the same requirement from the other side with their overlap construction. The
strong-Wolfe line search needs it too: it compares f at several points and those comparisons
are meaningless if f changes underneath it. So the phase owns ONE deterministic batch, held
fixed for its whole duration, and runs outside `fit()` where nothing perturbs it.

That requirement is CHECKED rather than assumed (`_assert_deterministic`): the phase evaluates
its closure twice before taking a step, and if the two values differ it falls back to the eager
rollout and says so, rather than running a line search on a function that moves.

FRAMEWORK LEVEL, DELIBERATELY
-----------------------------
Nothing here knows about the gantry. The system is touched only through `loss`,
`make_training_data`, `norm`, `cal_validation_error`, `parameters_with_names` and `simulator`,
so the MSD and encoder-initialisation pipelines can call it unchanged. Anything
problem-specific, such as the gantry's identifiable-combination recovery meter, arrives through
the `meters_fn` callback rather than an import.
"""
__project_origin__ = "added"

import math
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch


# =================================================================================================
# Knobs
# =================================================================================================

@dataclass(frozen=True)
class PolishSpec:
    """Everything the phase can be tuned by, in one object.

    Exists so the knobs are declared ONCE. They previously appeared in three places (`RunConfig`,
    the caller's call site, and this function's signature) and adding one meant editing all
    three. `RunConfig` still keeps flat fields, because `config.json` and its coverage assertion
    need them flat; the caller builds one spec from them, in one place.
    """
    n_windows: int = 4096          # 0 = every window (Drenth's full-batch setting; needs `chunk`)
    max_iter: int = 500            # a MAXIMUM, in L-BFGS ITERATIONS; tolerances fire first
    inner_iter: int = 20           # iterations per opt.step() call = logging + guard granularity
    history_size: int = 50         # m; 50 matches jax_sysid and Drenth, torch defaults to 100
    tol_grad: float = 1e-12        # NOT torch's 1e-7: the loss here is of order 1e-9
    tol_change: float = 1e-14      # NOT torch's 1e-9, same reason
    freeze: Tuple[str, ...] = ()   # parameter-group names to hold fixed, e.g. ('encoder',)
    meter_rel_tol: float = 1e-3    # relative worsening tolerated on a meter before rollback
    time_budget_s: Optional[int] = None
    chunk: Optional[int] = None    # windows per gradient-accumulation chunk; None = one chunk
    seed: int = 42


# =================================================================================================
# The acceptance decision: pure, so it can be tested without a rollout
# =================================================================================================

def meter_regression(before, after, rel_tol):
    """Names of the meters that got WORSE by more than `rel_tol`, relative.

    Lower is better for all of them: `param_loss` is drift from the initial parameters,
    `orth_frac` and `V_orth` are the negating component, `combo_err` is parameter-recovery error.
    A meter that is NaN on either side is skipped rather than treated as a regression:
    `orth_frac` reads NaN while the learned block is still ~0, which is a legitimate state.
    """
    bad = []
    for k, b in before.items():
        a = after.get(k)
        if a is None or not np.isfinite(a) or not np.isfinite(b):
            continue
        scale = abs(b) if abs(b) > 1e-30 else 1.0
        if (a - b) / scale > rel_tol:
            bad.append('%s %.3e -> %.3e' % (k, b, a))
    return bad


def decide(val_before, val_after, meters_before, meters_after, *,
           loss_finite, meter_rel_tol, validation_measure='sim-RMS',
           accept_without_validation=False):
    """Should the polished weights be kept? Pure: no tensors, no system, no side effects.

    Two conditions, both required (D-171). The validation measure must improve, and no meter may
    regress. The second is not optional: a polish phase can lower the loss by negating harder,
    where the learned block cancels more of the baseline, the fit improves and the physical
    parameters drift. Selecting on the fit alone would accept exactly that, which is the failure
    this project exists to prevent.

    Returns (accepted, reason).
    """
    if not loss_finite:
        return False, 'non-finite loss'
    have_val = np.isfinite(val_before) and np.isfinite(val_after)
    if not have_val:
        if not accept_without_validation:
            return False, ('no validation measure available and accept_without_validation=False; '
                           'refusing to keep an unchecked weight change')
        regressed = meter_regression(meters_before, meters_after, meter_rel_tol)
        if regressed:
            return False, 'meters regressed: ' + '; '.join(regressed)
        return True, 'accepted WITHOUT a validation check (accept_without_validation=True)'
    if not (val_after < val_before):
        return False, ('%s did not improve (%.6e -> %.6e)'
                       % (validation_measure, val_before, val_after))
    regressed = meter_regression(meters_before, meters_after, meter_rel_tol)
    if regressed:
        return False, 'meters regressed: ' + '; '.join(regressed)
    return True, 'accepted'


# =================================================================================================
# Parameters, snapshots, meters
# =================================================================================================

def _named_params(fit_sys):
    """[(group_name, Parameter)] over everything deepSI would hand its optimizer, DEDUPLICATED.

    `parameters_with_names` is deepSI's own accessor (`fit_system.py:187`), so it finds the
    encoder, the interconnect and any bare `nn.Parameter` attribute without this module
    hardcoding a single module name. Two things it does not do, and both matter here:
    its `params` entries for modules are GENERATORS, exhausted by one pass; and it walks
    `dir(self)`, so a parameter reachable under two attribute names appears twice. A duplicate
    would be handed to L-BFGS twice and counted twice, so it is removed by identity.
    """
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
    """Clone every parameter for an exact rollback.

    Clones rather than `state_dict()`: a state dict is keyed by module structure and this has to
    restore bare `nn.Parameter` attributes too. `copy_` back into the same tensors is bit-exact
    and leaves every dtype, device and optimizer reference untouched.
    """
    return [p.detach().clone() for _, p in named]


def _restore(named, snap):
    with torch.no_grad():
        for (_, p), saved in zip(named, snap):
            p.data.copy_(saved)


def generic_meters(fit_sys):
    """Scalars that say whether the fit improved for the RIGHT reason.

    Computed here rather than read from `fit_sys.validation_probes`, because `fit()` ends with
    `checkpoint_load_system` (`interconnect.py:1032`), which replaces `__dict__` wholesale; the
    probes that come back are unpickled copies bound to a stale system object.

    Only meters derivable from the system through a DUCK-TYPED interface live here, which today
    is exactly one: any block exposing `param_loss()`. Anything that has to know what the blocks
    ARE arrives through `meters_fn` (D-176). The orthogonality meters used to be computed here by
    reaching into `fit_sys.orth_penalty` and isinstance-testing for `Static_ANN_Block`, which
    made this module gantry-aware and, worse, gave `orth_frac` a second definition beside
    `training.py::_joint_probe`; a metric that gates ACCEPTANCE may not exist twice.
    """
    m = {}
    blocks = getattr(getattr(fit_sys, 'hfn', None), 'connected_blocks', ())
    if any(hasattr(b, 'param_loss') for b in blocks):
        m['param_loss'] = sum(float(b.param_loss()) for b in blocks if hasattr(b, 'param_loss'))
    return m


# =================================================================================================
# The fixed batch
# =================================================================================================

def window_count(fit_sys, sys_data, nf, stride):
    """How many windows `make_training_data` WOULD return, without building any of them.

    Arithmetic on record lengths, mirroring `System_data.to_hist_future_data`
    (`deepSI/system_data/system_data.py:305-330`), which has two branches:
      stride == 1 : sliding windows, `len(u) - max(na,nb) - nf + 1` per record
      stride > 1  : `range(k0 + k0_right, len(u) + 1, stride)` with `k0 = max(nb, na)` and
                    `k0_right = max(nf, na_right, nb_right)`

    Verified against the real pipeline: na = nb = 29, na_right = 1, nf = 400, stride = 10, 14
    records of 48000 samples gives 4758 per record and 66612 total, which is exactly the number
    the phase reported before this function existed.
    """
    na, nb = fit_sys.na, fit_sys.nb
    na_r = getattr(fit_sys, 'na_right', 0)
    nb_r = getattr(fit_sys, 'nb_right', 0)
    records = sys_data.sdl if hasattr(sys_data, 'sdl') else [sys_data]
    total = 0
    for sd in records:
        n = len(sd.u)
        if stride == 1:
            total += max(0, n - max(na, nb) - nf + 1)
        else:
            first = max(nb, na) + max(nf, na_r, nb_r)
            total += max(0, -(-(n + 1 - first) // stride))     # ceil division
    return total


class FixedBatch:
    """The ONE deterministic batch the phase optimises, and nothing else.

    Owns the STRIDE SCALING, which is the whole reason this is a class. The caller asks for a
    window COUNT; asking `make_training_data` for every window and then keeping 4096 of them
    means building, and discarding, the entire window set: measured 9.6 kB per window at nf=400,
    so ~640 MB there and roughly 19 GB at nf=12000, for a batch that ends up a few hundred MB.
    Scaling the stride up first makes the allocation follow the request instead of the dataset.

    The windows chosen are more widely spaced than the caller's own stride, and that is fine:
    the phase's requirement is that the set be FIXED and deterministic, not that it be a uniform
    random sample of the stride-10 set. `effective_stride` is recorded because two phases asking
    for 4096 windows out of different totals used different spacings and are otherwise not
    comparable.

    Iterating yields `(uh, yh, uf, yf, kwargs, weight)` chunks whose weights sum to 1. With
    `chunk=None` there is one chunk of weight 1 and the closure is a single forward and backward.
    With `chunk=N` the closure accumulates over chunks, which is what makes `n_windows=0` (full
    batch) reachable at all. The accumulated gradient is EXACT, not an approximation: every
    window contributes the same element count to the MSE, so a weighted sum of per-chunk means
    with weights `n_c / N` is the full mean, and any regulariser inside `loss()` is likewise
    weighted by `sum(w_c) = 1` and therefore counted exactly once.
    """

    def __init__(self, fit_sys, sys_data, loss_kwargs, n_windows, seed, chunk=None, verbose=True):
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

        # `window_count` mirrors deepSI's window arithmetic, and a mirror that drifts fails
        # SILENTLY: the stride scaling above would simply pick the wrong spacing and the phase
        # would optimise a differently sized batch than it reports. The check is pure arithmetic
        # against the count just built, so it costs nothing and turns that into an error.
        n_pred = window_count(fit_sys, sys_data, nf, self.effective_stride)
        if n_pred != n_built:
            raise RuntimeError(
                'window_count predicts %d windows at stride %d but make_training_data built %d. '
                'The two must agree, or the stride scaling that sizes this batch is guessing. '
                'deepSI\'s System_data.to_hist_future_data has changed, or na/nb/na_right/'
                'nb_right no longer mean what window_count assumes.'
                % (n_pred, self.effective_stride, n_built))

        # The scaled stride lands near `want` but not on it; take the remainder deterministically.
        self.n_used = min(want, n_built)
        if self.n_used < n_built:
            ix = np.random.default_rng(seed).choice(n_built, size=self.n_used, replace=False)
            ix.sort()          # ordered memory access; the SET is what matters, not the order
        else:
            ix = np.arange(n_built)

        dtype = next(fit_sys.hfn.parameters()).dtype
        device = next(fit_sys.hfn.parameters()).device

        def _to_tensor(a):
            t = torch.as_tensor(np.asarray(a)[ix])
            # Mirrors Dtype_DataLoader._conv (interconnect.py:558): floats take the model dtype,
            # integer arrays keep their own. `ctrl_ix` is an index, not a number to interpolate.
            return (t.to(dtype) if t.is_floating_point() else t).to(device)

        cols = [_to_tensor(a) for a in built]

        # Arrays a simulator asked for travel BY NAME, the convention fit() fixes at
        # interconnect.py:921-935. Re-expressed here rather than shared, so that adding this
        # phase cannot touch the Adam path; the guards below are the same two fit() makes.
        names = tuple(getattr(getattr(fit_sys, 'simulator', None), 'extra_array_names', ()))
        n_extra = len(cols) - 4
        if n_extra != len(names):
            raise RuntimeError(
                'the training data carries %d array(s) beyond deepSI\'s four but the simulator '
                'names %d of them (%s). make_training_data and extra_array_names must agree, or '
                'an array reaches loss() unnamed.' % (n_extra, len(names), names))
        clash = set(names) & set(loss_kwargs)
        if clash:
            raise RuntimeError(
                'simulator array name(s) %s collide with loss_kwargs of the same name; one would '
                'silently overwrite the other' % sorted(clash))

        self._cols = cols
        self._names = names
        self._loss_kwargs = dict(loss_kwargs)
        self.chunk = self.n_used if chunk in (0, None) else min(int(chunk), self.n_used)
        self.n_chunks = int(math.ceil(self.n_used / self.chunk))
        self.bytes_on_device = sum(c.element_size() * c.nelement() for c in cols)
        self.device = device

        if verbose:
            print('  [lbfgs] fixed batch: %d of %d windows (stride %d -> %d), %.1f MB on %s, '
                  '%d chunk(s), extras=%s'
                  % (self.n_used, self.n_total, base_stride, self.effective_stride,
                     self.bytes_on_device / 1024 ** 2, device, self.n_chunks, names or '()'))

    def __iter__(self):
        for lo in range(0, self.n_used, self.chunk):
            hi = min(lo + self.chunk, self.n_used)
            uh, yh, uf, yf = (c[lo:hi] for c in self._cols[:4])
            kw = dict(self._loss_kwargs)
            kw.update({n: c[lo:hi] for n, c in zip(self._names, self._cols[4:])})
            yield uh, yh, uf, yf, kw, (hi - lo) / self.n_used


# =================================================================================================
# Context managers
# =================================================================================================

def _eager(fit_sys, active=True):
    """Optionally run through the EAGER rollout instead of the compiled one.

    Delegates to the simulator's own `eager()` context, so this module holds no knowledge of how
    a simulator stores its compiled artefact. A simulator without one is simply always eager.

    NOT unconditional. The earlier version always disabled compilation, on the grounds that a
    compiled run is not bit-identical to eager. That is the wrong test: the line search does not
    need the compiled value to equal the eager value, it needs the loss to be self-consistent
    across evaluations, which is what `_probe_closure` checks.

    WHICH ROLLOUT THIS PHASE ACTUALLY GETS, and why it is not the compiled one. `fit()` ends by
    reloading its best checkpoint (`interconnect.py`, `checkpoint_load_system`), which replaces
    `__dict__` wholesale with unpickled objects, and `ClosedLoopSimulator.__getstate__` drops
    `_compiled` on purpose. So after a normal Adam run the simulator comes back EAGER and the
    compiled branch here cannot fire; it fires only on a `start_phase='lbfgs'` resume, where
    `fit()` never ran. That is structural, not a fallback, and on the measured numbers it is also
    the better outcome: eager cost 4.37 s per closure in job 81655 against roughly 0.67 s
    compiled (~6.5x, jobs 80610/80634/80652) plus a one-off compilation of order 1400 s at this
    batch shape, so compiling only pays past roughly 380 closures and the phase used 138.
    """
    sim = getattr(fit_sys, 'simulator', None)
    return sim.eager() if (active and hasattr(sim, 'eager')) else nullcontext()


@contextmanager
def _frozen(named, names):
    """Hold the named parameter groups fixed for the duration, and restore what was changed.

    Leaving them out of the optimizer is NOT freezing. Backward still computes their gradients,
    and `opt.zero_grad` only touches the optimizer's own parameters, so a frozen group
    accumulated gradient over every closure of the phase and was never cleared: no compute saved,
    and a stale `.grad` left on the model afterwards. Clearing `requires_grad` is what actually
    keeps them out of the backward.

    Only the parameters this manager changed are restored, so a group the caller had already
    frozen stays frozen.
    """
    touched = [p for n, p in named if n in names and p.requires_grad]
    for p in touched:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p in touched:
            p.requires_grad_(True)


@contextmanager
def _quiet_probes(fit_sys):
    """Suppress `validation_probes` and `loss_stats` for the duration.

    Two different reasons, both silent corruption rather than an error. The probes that survive
    `fit()` are unpickled copies bound to a stale system (see `generic_meters`), so firing them
    writes to an object nobody reads. And `loss_stats` accumulates inside `loss()`, so every
    line-search evaluation would enter the nf probe's running mean and make the reported train
    RMS a mixture of measurements and trial points.
    """
    probes = getattr(fit_sys, 'validation_probes', ())
    stats = getattr(fit_sys, 'loss_stats', None)
    fit_sys.validation_probes = ()
    fit_sys.loss_stats = None
    try:
        yield
    finally:
        fit_sys.validation_probes = probes
        fit_sys.loss_stats = stats


# =================================================================================================
# The phase
# =================================================================================================

def warm_up_lazy_init(fit_sys, batch, max_windows=8):
    """One SMALL eager forward, so lazy initialisation happens outside any compiled region.

    `Interconnect.output_only` builds its block execution order on first call
    (`interconnect.py:288-290`, guarded by `initialized_forward_function`), and that build runs
    numpy graph analysis with data-dependent control flow (`utils.py:38`, `if np.trace(An) != 0`).
    torch.compile cannot trace it and raises "Dynamic control flow is not supported".

    A normal training run never meets this, because `fit()` performs an eager validation before
    the first compiled update (`interconnect.py:892`) and the flag is set by then. A
    `start_phase='lbfgs'` resume has run nothing at all, so the polish would be the first caller
    and would trip it. Measured on job 81462.

    Deliberately generic: one real forward under `no_grad`, not a poke at the flag, so any other
    lazy initialisation in the chain is warmed too.

    SLICED to `max_windows`, and the reason is MEMORY, not time.

    The lazy state this builds depends on the interconnection graph, not on how many windows pass
    through, so a handful of windows warms it exactly as well as the whole batch. What slicing
    avoids is allocating and immediately freeing a FULL-SIZE eager activation graph (~5 GB at
    16384 windows in float32) right before the compiled allocation, which is a fragmentation risk
    exactly where memory is tightest.

    It saves almost no TIME, and an earlier version of this comment claimed otherwise. The
    rollout is dispatch-bound: its cost is `nf` sequential steps of ~2400 tiny ops and is FLAT in
    batch size, measured in job 81463 as ~4.1 s per closure from 512 to 8192 windows. So 8
    windows and 16384 windows both cost about 4 s here, and that 4 s is paid once per phase
    either way.

    Cheaper options exist and are deliberately not taken. Calling `hfn.init_forward()` directly
    would cost nothing, but assumes this is the ONLY lazy state in the chain; slicing the TIME
    axis as well would cut the cost by `nf`, but `loss_kwargs` still carries `nf`, which the
    attached simulator may index against. Against a ~500 s compilation and two ~3 minute
    validations, 4 s once is not worth either risk.
    """
    with _eager(fit_sys, active=True), torch.no_grad():
        uh, yh, uf, yf, kw, _w = next(iter(batch))
        n = min(max_windows, uh.shape[0])
        kw = {k: (v[:n] if torch.is_tensor(v) and v.shape[:1] == uh.shape[:1] else v)
              for k, v in kw.items()}
        fit_sys.loss(uh[:n], yh[:n], uf[:n], yf[:n], **kw)


def _make_closure(fit_sys, batch, params, opt, counter):
    """One closure over the fixed batch, accumulating over chunks when there is more than one."""
    def closure():
        counter[0] += 1
        opt.zero_grad(set_to_none=False)
        total = 0.0
        for uh, yh, uf, yf, kw, w in batch:
            loss = fit_sys.loss(uh, yh, uf, yf, **kw) * w
            loss.backward()
            total += float(loss)
        # A parameter that never reaches the loss comes back with grad None, and torch's
        # flat-gradient gather would raise mid-phase. Zero is the correct value for it.
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        return torch.as_tensor(total)
    return closure


def _loss_value(fit_sys, batch):
    """The objective at the CURRENT weights: one forward over the batch, no graph, no gradient.

    Exists because `opt.step()` cannot answer this question. It returns `orig_loss`, the value
    its closure produced at the START of the block (torch's `LBFGS.step`: `orig_loss = closure()`
    ... `return orig_loss`), so reading it leaves the loss at the returned weights unmeasured.
    The last value the CLOSURE saw is no better: a strong-Wolfe line search evaluates trial points
    and the last of them may be one it rejected.

    Same expression as the closure, minus the backward, so it is cheaper than one closure and is
    comparable with `loss_first` term for term.
    """
    with torch.no_grad():
        return float(sum(fit_sys.loss(uh, yh, uf, yf, **kw) * w
                         for uh, yh, uf, yf, kw, w in batch))


def _lbfgs_counts(opt, params):
    """(iterations, closure evaluations) as torch ITSELF counted them, cumulative over step()s.

    `torch.optim.LBFGS` keeps both in `state[self._params[0]]` and exposes neither through a
    return value, so a caller that wants them has to read them here. Inferring them instead, as
    `(k + 1) * inner_iter`, is wrong by construction: `max_eval` defaults to `max_iter * 1.25`,
    an iteration under a line search costs one to three evaluations, and torch breaks out of a
    step on any of three internal conditions. Run 81655 reported 500 iterations for 138 closures;
    since every iteration costs at least one evaluation and every step() spends one more on
    `orig_loss`, the true count there was at most 113.
    """
    st = opt.state[params[0]]
    return int(st.get('n_iter', 0)), int(st.get('func_evals', 0))


def _is_oom(exc):
    """True for a CUDA out-of-memory error, across torch versions."""
    oom_cls = getattr(torch.cuda, 'OutOfMemoryError', ())
    return isinstance(exc, oom_cls) or (isinstance(exc, RuntimeError)
                                        and 'out of memory' in str(exc).lower())


def _probe_closure(closure, label, verbose):
    """Evaluate twice without stepping; return (ok, first_value, error_or_None).

    This is the precondition the whole phase rests on, so it is checked rather than assumed, on
    the device and backend actually in use.

    It also catches ANY exception, not only non-determinism, and that is deliberate. Every server
    failure this phase has had was the compiled path raising where eager would not: job 81462 was
    `torch._dynamo.exc.UserError: Dynamic control flow is not supported` from a lazy init inside
    the traced region. A compile problem must degrade the phase to eager, never kill a run that
    has already spent hours in Adam.
    """
    try:
        a = float(closure())
        b = float(closure())
    except Exception as e:                       # noqa: BLE001 - see the docstring
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
                 meters_fn=None, force_eager=False, accept_without_validation=False,
                 verbose=True):
    """Polish `fit_sys` in place with L-BFGS on one fixed batch. Returns an outcome dict.

    The weights change ONLY if the phase is accepted (see `decide`). On rejection every parameter
    is restored bit-exactly, so calling this can never make a run worse than not calling it.

    Parameters
    ----------
    loss_kwargs : dict
        Passed to BOTH `make_training_data` and `loss`, exactly as `fit()` does. Carries `nf` and
        `stride`, so the phase optimises the objective the run trained on.
    spec : PolishSpec
        Every tuning knob. Defaults to `PolishSpec()`.
    val_before : float or None
        The pre-phase validation measure when the caller already has it (`fit_sys.bestfit`).
        None means measure it, which costs one full closed-loop free run (~162 s on the GPU).
    force_eager : bool
        Disable the compiled rollout up front. Not normally needed: the phase checks determinism
        and falls back on its own.
    """
    spec = spec or PolishSpec()
    t0 = time.time()
    # `n_iter` is the count torch reports, and `max_iter` the cap it was allowed. Both are
    # recorded because runs before D-174 wrote the CAP under `n_iter`, so the two eras are only
    # distinguishable if the cap is on disk beside the count.
    out = dict(ran=False, accepted=False, reason='', n_iter=0, max_iter=spec.max_iter,
               n_closure=0, func_evals=0, wall_s=0.0,
               loss_first=float('nan'), loss_last=float('nan'),
               val_before=float('nan'), val_after=float('nan'),
               meters_before={}, meters_after={}, n_windows_used=0, n_windows_total=0,
               effective_stride=0, n_chunks=0, deterministic=None, used_compiled=None,
               peak_mem_bytes=0, params_polished=0, frozen=tuple(spec.freeze))

    named = _named_params(fit_sys)
    snap = _snapshot(named)

    def _all_meters():
        m = generic_meters(fit_sys)
        if meters_fn is not None:
            try:
                m.update(meters_fn(fit_sys))
            except Exception as e:                      # a meter must never break the phase
                print('  [lbfgs] meters_fn failed (non-fatal): %s' % e)
        return m

    def _validate():
        if val_sys_data is None:
            return float('nan')
        was_training = fit_sys.hfn.training
        fit_sys.eval()
        try:
            with _quiet_probes(fit_sys):
                return float(fit_sys.cal_validation_error(val_sys_data, validation_measure))
        finally:
            if was_training:
                fit_sys.train()

    # The caller usually knows this already (`fit_sys.bestfit`), and measuring it again costs a
    # full free run. Reusing it compares a compiled-GPU number against an eager one, which this
    # project measured as agreeing to 5 digits, far inside the validation noise band.
    out['val_before'] = _validate() if val_before is None else float(val_before)
    out['meters_before'] = _all_meters()

    # --- the parameter vector -----------------------------------------------------------------
    # ONE group. torch.optim.LBFGS refuses more than one ("doesn't support per-parameter
    # options"), which is also what the reference method does: Drenth's theta = [theta_G;
    # theta_H; theta_psi] puts the LFR matrices and the scheduling-map network in a single vector
    # under one step size (thesis p14), extended by the free initial states (p20).
    frozen = set(spec.freeze)
    unknown = frozen - {n for n, _ in named}
    if unknown:
        raise ValueError('freeze names %s are not parameter groups; available: %s'
                         % (sorted(unknown), sorted({n for n, _ in named})))
    params = [p for n, p in named if n not in frozen and p.requires_grad]
    if not params:
        out['reason'] = 'no trainable parameters left after freeze=%s' % (tuple(spec.freeze),)
        return out
    out['params_polished'] = sum(p.numel() for p in params)

    with _quiet_probes(fit_sys), _frozen(named, frozen):
        batch = FixedBatch(fit_sys, train_sys_data, loss_kwargs, spec.n_windows, spec.seed,
                           chunk=spec.chunk, verbose=verbose)
        out.update(n_windows_used=batch.n_used, n_windows_total=batch.n_total,
                   effective_stride=batch.effective_stride, n_chunks=batch.n_chunks)

        # BEFORE anything compiled runs. See warm_up_lazy_init: on a start_phase='lbfgs' resume
        # this phase is the first caller of the model, and the lazy init it triggers cannot be
        # traced by torch.compile.
        warm_up_lazy_init(fit_sys, batch)

        # The batch line above reports the TENSORS. What decides whether `lbfgs_windows` can be
        # raised is the peak, which is dominated by the autograd graph over nf sequential steps
        # and was measured at roughly 34x the tensors at nf=400. Reset here so the number
        # reported at the end belongs to this phase and not to the Adam run before it.
        if torch.cuda.is_available() and batch.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(batch.device)

        def _new_opt():
            return torch.optim.LBFGS(
                params, lr=1.0, max_iter=spec.inner_iter, history_size=spec.history_size,
                tolerance_grad=spec.tol_grad, tolerance_change=spec.tol_change,
                line_search_fn='strong_wolfe')

        n_closure = [0]
        sim = getattr(fit_sys, 'simulator', None)
        compiled_available = sim is not None and getattr(sim, '_compiled', None) is not None

        # Run compiled when it is available, and PROVE the loss is reproducible before stepping.
        # If it is not, fall back to eager and check again; only then give up. This replaces the
        # old unconditional `_eager`, which paid ~6.5x on every closure to avoid a problem that
        # had never been shown to exist.
        # Try the compiled rollout, then fall back to eager on EITHER a failure or a
        # non-reproducible loss. Both outcomes are survivable and neither may kill the run.
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
                if verbose:
                    print('  [lbfgs] ABORTED, weights unchanged: %s' % out['reason'])
                return out

            out['loss_first'] = first
            if verbose:
                # tol_change is printed BOTH ways because torch applies it absolutely, in three
                # places (`gtd > -tol`, `max|t*d| <= tol`, `|loss - prev_loss| < tol`), while a
                # reader judges it against the loss. At 1e-14 on a loss of order 1e-9 it is a
                # relative 1e-5, which is what ends each block after a couple of iterations.
                print('  [lbfgs] %d parameters in one group, m=%d, strong-Wolfe, '
                      'tol_grad=%.1e tol_change=%.1e (%.1e relative to the opening loss), '
                      'max %d iterations, %s rollout'
                      % (out['params_polished'], spec.history_size, spec.tol_grad,
                         spec.tol_change, spec.tol_change / max(abs(first), 1e-300),
                         spec.max_iter, 'compiled' if out['used_compiled'] else 'eager'))

            n_outer = max(1, int(math.ceil(spec.max_iter / max(1, spec.inner_iter))))
            for k in range(n_outer):
                it_before, _ = _lbfgs_counts(opt, params)
                # An OOM or any other mid-phase failure must NOT kill the job. This phase runs
                # after Adam, so a crash here throws away hours of training that succeeded; and
                # on this pipeline the phase is where a large fixed batch first meets the
                # compiled rollout, which is the likeliest place to run out of memory. Whatever
                # happens, the run keeps the weights it arrived with.
                try:
                    # The return value is deliberately NOT bound: it is the loss at the START of
                    # this block, not at the point the block reached. `loss_last` is measured
                    # once after the loop instead (see _loss_value).
                    opt.step(closure)
                except Exception as e:                   # noqa: BLE001 - see the comment above
                    kind = 'CUDA out of memory' if _is_oom(e) else type(e).__name__
                    out['n_iter'], out['func_evals'] = _lbfgs_counts(opt, params)
                    out['reason'] = ('%s after %d iterations; phase abandoned, weights restored. %s'
                                     % (kind, out['n_iter'],
                                        'Lower lbfgs_windows, or set lbfgs_chunk.'
                                        if _is_oom(e) else str(e).splitlines()[0][:200]))
                    out['wall_s'] = time.time() - t0
                    out['n_closure'] = n_closure[0]
                    _restore(named, snap)
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
                    # torch stopped inside step() on one of its own conditions, so every further
                    # call costs a closure and moves nothing. Exact, unlike the previous test
                    # (`loss == prev`), which compared start-of-block values that never repeat.
                    out['reason'] = ('converged: L-BFGS stopped internally after %d of %d '
                                     'iterations in this block (tol_grad=%.1e, tol_change=%.1e)'
                                     % (done, spec.inner_iter, spec.tol_grad, spec.tol_change))
                    break
                if spec.time_budget_s is not None and (time.time() - t0) > spec.time_budget_s:
                    out['reason'] = 'wall-clock budget %.0f s reached' % spec.time_budget_s
                    break
            else:
                out['reason'] = 'iteration cap %d reached' % spec.max_iter

            # The loss at the weights the phase is about to be judged on. Cheaper than a closure
            # (no backward) and, unlike step()'s return value, actually measured HERE.
            out['loss_last'] = _loss_value(fit_sys, batch)
            out['n_closure'] = n_closure[0]
            if not np.isfinite(out['loss_last']):
                out['reason'] += ' | non-finite loss at the final point'
            if torch.cuda.is_available() and batch.device.type == 'cuda':
                out['peak_mem_bytes'] = int(torch.cuda.max_memory_allocated(batch.device))

    out['ran'] = True
    out['val_after'] = _validate()
    out['meters_after'] = _all_meters()
    out['wall_s'] = time.time() - t0

    out['accepted'], why = decide(
        out['val_before'], out['val_after'], out['meters_before'], out['meters_after'],
        loss_finite=bool(np.isfinite(out['loss_last'])), meter_rel_tol=spec.meter_rel_tol,
        validation_measure=validation_measure,
        accept_without_validation=accept_without_validation)
    out['reason'] = '%s | %s' % (out['reason'], why)

    if not out['accepted']:
        _restore(named, snap)
    if verbose:
        print('  [lbfgs] %s: %s' % ('ACCEPTED' if out['accepted'] else 'ROLLED BACK', why))
        print('  [lbfgs] loss %.6e -> %.6e | %d iterations, %d closures in %.0f s%s'
              % (out['loss_first'], out['loss_last'], out['n_iter'], out['n_closure'],
                 out['wall_s'],
                 '' if not out['peak_mem_bytes']
                 else ' | peak %.1f GB (batch %.1f MB)'
                      % (out['peak_mem_bytes'] / 1024 ** 3,
                         batch.bytes_on_device / 1024 ** 2)))
        if np.isfinite(out['val_before']):
            print('  [lbfgs] %s %.6e -> %.6e'
                  % (validation_measure, out['val_before'], out['val_after']))
    return out
