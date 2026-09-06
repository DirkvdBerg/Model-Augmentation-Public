# L-BFGS polish phase: known problems in the current code

Written 2026-09-04, after the implementation in D-171 passed its 27 gates. Every item below is a
defect in code that IS committed and IS passing tests, found by re-reading rather than by a failure.
The tests pass because the fixture runs at `nf=400` on the CPU with `use_f64=False`, and every issue
here needs a bigger `nf`, a GPU, or float64 to appear.

Companion documents: design and rationale in **D-171** (`docs/decisions.md:22`); file-by-file plan in
`tasks/lbfgs-implementation-plan.md`; literature in `docs/references.md`, section "Adam to L-BFGS
handover".

**Nothing here is a reason to revert.** With `lbfgs=False` the phase does not run and the Adam path
is unchanged, proven by `test_lbfgs_false_is_an_exact_no_op`. Items 1 and 2 must be fixed before any
GPU run; items 3 and 4 must be measured on the cluster before the phase is trusted there.

---

## 1. BLOCKING. The fixed batch is materialised in full before it is subsampled

**Where:** `model_augmentation/fit_systems/lbfgs_polish.py:149`

```python
d = fit_sys.make_training_data(fit_sys.norm.transform(train_sys_data), **loss_kwargs)
```

This builds **every** window, then line 155 keeps `n_windows` of them and discards the rest. The
whole point of `lbfgs_windows` is to work on a subset, and the subset is taken after paying for the
whole.

**Evidence.** The smoke run reports "fixed batch: 32 of 66612 windows, 0.3 MB", i.e. **9.6 kB per
window at nf=400**, so the full set is ~640 MB built and thrown away on every call. At the target
`nf=12000` each window is 30x larger, giving roughly **19 GB of host allocation** for a batch that
ends up a few hundred MB. `docs/gpu-nf12000-feasibility-2026-08-31.md` records the same arithmetic
from the other side ("window arrays 145 GB" before the stride correction).

**Consequence.** At `nf=12000` the phase very likely dies in `make_training_data` before it reaches
the optimizer, on a machine that has enough memory for the training run itself.

**Fix.** Choose the windows before materialising them. `make_training_data` has no index argument,
but it has `stride`: scale the caller's stride so the returned count is approximately `n_windows`,
then subsample the small remainder. The windows are differently spaced but equally valid, and the
phase's requirement is only that the set be FIXED and deterministic, not that it be a uniform random
sample of the stride-10 set. Record the effective stride in the outcome dict so two runs remain
comparable.

**Why the tests missed it.** Every fixture test runs at `nf=400`, where 640 MB is invisible on a
32 GB host, and no test asserts anything about peak memory.

---

## 2. BLOCKING. `lbfgs_windows=4096` and `use_f64=True` contradict each other on the target card

**Where:** `scripts/gantry/gantry_interconnect_dynamic.py`, the `lbfgs_windows=4096` default, against
D-171's recommendation of `use_f64=True` for the whole run.

The closure holds the activation graph for the entire fixed batch at once, so it costs what a
training update at that batch size costs. `docs/gpu-nf12000-feasibility-2026-08-31.md:155` records
batch 4096 as "the practical ceiling on the RTX 2080" and batch 8192 as ~9 GB, i.e. over the 8 GB
card, **in float32**. float64 doubles activation memory. So the two defaults this project now
recommends together are, on the card the pipeline targets, probably an OOM.

**Mitigation that already exists and was never stated.** The phase inherits
`simulator.checkpoint_chunk`, because the rollout runs through the attached `ClosedLoopSimulator`,
so gradient checkpointing applies to the polish for free at ~+33% compute. That is the lever, and
neither the config comment nor D-171 mentions it.

**Fix.** Say so in the config comment, and either lower the default or make the default depend on
nothing and force the caller to choose. A default that OOMs on the reference hardware is worse than
no default.

---

## 3. Forcing the eager rollout gives up the 6.5x compile speedup, on an untested premise

**Where:** `model_augmentation/fit_systems/lbfgs_polish.py:193`, `_eager`, which sets
`simulator._compiled = None` for the duration of the phase.

The Adam route runs `compile_mode='reduce-overhead'` at a measured ~6.5x (jobs 80610/80634/80652).
The phase disables it, so **every closure is about 6.5x more expensive than the equivalent Adam
update**, and one L-BFGS iteration costs one to three closures.

**The premise is the wrong test.** The justification given, in the docstring and in D-171, is that a
compiled run is not bit-identical to eager (max |dg| = 7.5e-10, job 80610). But the strong-Wolfe
line search does not need the compiled value to equal the eager value; it needs the loss to be
**self-consistent across evaluations within the phase**. A compiled callable that returns the same
number for the same input satisfies that completely.

**Fix.** Measure it: run the closure twice under `reduce-overhead` on the cluster and compare bit
for bit. If it is self-consistent, drop `_eager` to an opt-out and take the 6.5x. If it is not, keep
it and cite the measurement instead of the current argument, which does not support the conclusion
it is used for.

---

## 4. The determinism evidence is CPU-only, and the whole design rests on it

**Where:** `test_lbfgs_polish.py::test_closure_is_bit_deterministic`.

It asserts two identical closure evaluations return the same float, and it passes
(`1.28890442763207602e-09` twice). It has only ever run **on the CPU**. CUDA backward kernels use
atomics in places and are not guaranteed run-to-run reproducible without
`torch.use_deterministic_algorithms`. Bit-determinism of the loss is the precondition for the line
search and therefore for the entire phase.

**Fix.** Run this one test on the cluster before trusting the phase there, and record the result in
D-171. If it fails, the remedies in order are: deterministic algorithms mode, or accepting an
inexact line search with a looser Wolfe tolerance. Until then, the claim "the closure is
deterministic" is **proven on CPU and assumed on GPU**, and should be written that way everywhere it
appears.

---

## 5. Two full validations per phase, one of which is nearly free to remove

**Where:** `lbfgs_polish.py:299` (`val_before`) and the matching call after the loop.

Each is a closed-loop free run over V1-V4, ~162 s on the GPU (job 80713), so the phase pays ~324 s
of validation regardless of how short it is. The "before" number is already known:
`fit_sys.bestfit` holds it, and this session measured eager-CPU 5.774347e-06 against the cluster's
compiled-GPU 5.774376e-06, agreeing to 5 digits, which is 0.0005% relative, far inside the +-0.1%
validation noise band of run 81262.

**Fix.** Accept `val_before` as an argument, defaulting to None meaning "measure it". The caller
passes `fit_sys.bestfit` when it has it. Keep the measured path for a standalone call.

**Counter-argument, and it is not weak:** two numbers produced by the same code path are more
trustworthy than one taken from a checkpoint written by a different device and dtype. If the phase
is expected to run for tens of minutes, 162 s is 10% of it and the extra rigour is worth it. Decide
per run rather than hardcoding.

---

## 6. `_named_params` does not deduplicate

**Where:** `lbfgs_polish.py:45-60`.

`parameters_with_names` walks `dir(self)` and returns one entry per attribute. Any parameter
reachable under two attribute names is returned twice, handed to `torch.optim.LBFGS` twice, and
counted twice in `params_polished`. No current model does this, so it is latent rather than active.

**Fix.** Deduplicate by `id(p)` while preserving order. Two lines.

---

## 7. `val_sys_data=None` means unconditional acceptance

**Where:** `lbfgs_polish.py:385-386`.

```python
improved = (np.isnan(out['val_before']) and np.isnan(out['val_after'])
            or out['val_after'] < out['val_before'])
```

With no validation data both sides are NaN and `improved` is True, so the phase accepts whatever it
produced provided no meter regressed. That is defensible (there is nothing to check against) and it
is what the fast tests rely on, but a caller who simply forgets to pass `val_sys_data` gets silent
unconditional acceptance of a weight change. `run_lbfgs_polish` always passes it, so the production
path is safe.

**Fix.** Either require an explicit `accept_without_validation=True`, or log a loud line saying
acceptance was not checked. Also add the parentheses: the expression is correct by operator
precedence but reads as though it might not be.

---

## What holds up

Stated so the list above is not read as a verdict on the whole file:

- the framework/gantry split: `model_augmentation/` imports nothing gantry-specific, and
  `combo_err` arrives through the `meters_fn` hook;
- rollback is bit-exact, asserted tensor by tensor;
- `lbfgs=False` is a proven exact no-op;
- the two-format loader plus the architecture guard that names the mismatching key;
- `interconnect.py` and `model.py` untouched, so the Adam path carries no risk from this work;
- the acceptance rule fired correctly on its first real trial, rolling back a phase that improved
  the training loss while sim-RMS worsened.

---

## Order of work

1. Item 1 (batch materialisation), then item 2 (the memory-contradicting defaults). Both are
   required before a GPU run at any interesting `nf`.
2. Items 3 and 4 together, as one cluster measurement: run the determinism test under both eager and
   `reduce-overhead`. That single job resolves whether the phase can keep the 6.5x and whether the
   line search is safe on CUDA at all.
3. Items 5, 6, 7 are cleanups; none blocks a run.

---

# How to get to code that is actually optimal

The section above lists what is wrong. This one says what the file should look like instead. It is
specific to this code: no generic advice, and every item names the function it changes.

An eighth defect surfaces here rather than above, because it is a design gap rather than a bug:
**`lbfgs_windows=0` ("full batch") is advertised and cannot work as written.** It would materialise
every window (item 1) and then hold one autograd graph over all of them. Drenth's own setting is
therefore unreachable today. It needs gradient accumulation, which is design work, not a patch.

## A. The batch becomes an object that owns its own scaling

`_build_fixed_batch` currently does four things in one function: it materialises windows,
subsamples, casts and homes tensors, and binds the simulator's extra arrays by name. Split the first
from the rest, because only the first is where the memory problem lives.

```python
class FixedBatch:
    """The ONE deterministic batch a polish phase optimises, and nothing else.

    Owns the stride scaling, so the caller asks for a window COUNT and never computes a stride.
    """
    def __init__(self, fit_sys, sys_data, loss_kwargs, n_windows, seed, chunk=None): ...
    def __iter__(self):        # yields (uh, yh, uf, yf, kwargs) chunks; ONE chunk when chunk=None
    n_used: int                # windows actually used
    effective_stride: int      # what it asked make_training_data for; goes in the outcome dict
    bytes_on_device: int
```

Two properties make this worth a class rather than a tuple:

1. **`effective_stride` is a run parameter and has to be recorded.** Two polish runs asking for 4096
   windows out of different totals used different window spacings, and without this in the outcome
   dict they are not comparable.
2. **`__iter__` is where full batch becomes possible.** With `chunk=None` it yields once and the
   closure is what it is today. With `chunk=N` the closure loops, accumulating gradients, and
   `lbfgs_windows=0` stops being a lie. The accumulated gradient is exact, not an approximation, so
   nothing in the L-BFGS theory changes; only peak activation memory does.

The stride scaling itself is three lines and removes item 1:

```python
# Ask make_training_data for ~n_windows rows instead of building all of them and discarding 99%.
n_total_est = _count_windows(sys_data, nf, stride=loss_kwargs['stride'])
scale = max(1, n_total_est // max(1, n_windows))
built = fit_sys.make_training_data(..., stride=loss_kwargs['stride'] * scale)
```

`_count_windows` is arithmetic on record lengths, not an allocation.

## B. The acceptance decision becomes a pure function

Right now the decision is inline at the end of `lbfgs_polish`, so testing "does a 0.05% sim-RMS
worsening with flat meters get rejected" requires running a rollout. Extract it:

```python
def decide(val_before, val_after, meters_before, meters_after, *,
           loss_finite, meter_rel_tol):
    """Pure. No tensors, no fit_sys. Returns (accepted, human-readable reason)."""
```

This is the highest-value refactor in the list for testability. The acceptance rule is the part that
protects the thesis contribution, and it currently has exactly one integration test behind it. As a
pure function it gets a table of cases in milliseconds, including the joint-estimation and orth
combinations the local fixture cannot produce, which is precisely what D-171 has to leave "verified
by construction" today.

## C. The knobs are declared once, not three times

`lbfgs_max_iter`, `lbfgs_history`, `lbfgs_windows`, the two tolerances, `freeze`, the budget and the
meter tolerance are currently written out in three places: `RunConfig`, the `run_lbfgs_polish` call,
and the `lbfgs_polish` signature. Adding a knob means editing all three plus the `config.json` dict.
Collapse the last two:

```python
@dataclass(frozen=True)
class PolishSpec:
    n_windows: int = 4096
    max_iter: int = 500
    inner_iter: int = 20
    history_size: int = 50
    tol_grad: float = 1e-12
    tol_change: float = 1e-14
    freeze: tuple = ()
    meter_rel_tol: float = 1e-3
    time_budget_s: Optional[int] = None
    chunk: Optional[int] = None

def lbfgs_polish(fit_sys, train_sys_data, *, loss_kwargs, spec=PolishSpec(), ...)
```

`RunConfig` keeps its flat fields, because they must stay flat for `config.json` and for
`_assert_json_coverage`. `run_lbfgs_polish` builds one `PolishSpec` from them, in one place.

## D. The phase self-checks determinism instead of assuming it

Items 3 and 4 come from the same gap: bit-determinism is a precondition, it is asserted in a test on
one device, and the phase itself never checks. Make it check, for the price of one extra closure
evaluation:

```python
l1, l2 = closure_value(), closure_value()          # no step between them
if l1 != l2:
    raise RuntimeError(
        'the loss is not reproducible on this device/backend (%r vs %r). The strong-Wolfe line '
        'search compares f at several points and cannot work here. Try compile_mode=None, or '
        'torch.use_deterministic_algorithms(True).' % (l1, l2))
```

This is better than either current option. It removes the need to decide eager-versus-compiled in
advance: run compiled, and if the backend is not self-consistent the phase says so in one line
instead of quietly taking bad steps. `_eager` then becomes an opt-out (`force_eager=True`) rather
than an unconditional cost, which is where the 6.5x comes back.

## E. The duplication with `fit()` is removed, once there is a test to protect it

`_build_fixed_batch` re-expresses the extra-array binding from `interconnect.py:921-935`. Duplication
was the right call for a first version and is not the clean end state. The clean end state is one
function in `interconnect.py`:

```python
def bind_extra_arrays(simulator, cols, loss_kwargs):   # plus the two RuntimeError guards
```

called by both `fit()` and the batch builder, carrying a `# CHANGED:` marker per the CLAUDE.md
tracking rule. The precondition is a test that pins `fit()`'s current binding behaviour, so the
extraction is provably a no-op. Do the test first, the extraction second, never the other way round.

## F. The test that would have caught item 1

Every existing test asserts behaviour. None asserts cost, which is why a 19 GB allocation passed 27
gates. Add one:

```python
def test_batch_build_allocates_with_n_windows_not_with_the_full_set(self):
    small = _peak_host_bytes(lambda: FixedBatch(..., n_windows=32))
    large = _peak_host_bytes(lambda: FixedBatch(..., n_windows=256))
    self.assertLess(small * 4, large)        # scales with the ASK
    self.assertLess(large, 0.1 * full_set_bytes)
```

`tracemalloc` gives the host figure and `torch.cuda.max_memory_allocated` the device one. The
assertion is deliberately loose: the point is to fail when the build stops scaling with the request,
not to pin a number that will drift.

## G. Priority, if only part of this gets done

1. **A** (stride scaling) and **D** (self-check). Together they remove both blocking issues and the
   eager-versus-compiled question in one pass, and D is about fifteen lines.
2. **F**, immediately after A, or the defect returns the next time the builder is touched.
3. **B**. Cheap, and it is what turns D-171's "verified by construction" into "tested".
4. **C**. Ergonomics, worth doing before the knob list grows again.
5. **E**. Last, and only with the pinning test first.

Defect items 5, 6 and 7 fold into this work: `val_before` becomes an argument on the way through B,
the `id()` deduplication belongs in `_named_params` whenever it is next opened, and the
`val_sys_data=None` acceptance becomes explicit inside `decide`.

## What "optimal" does not mean here

Worth stating, because the pull is real. The phase runs once per training run, for minutes, over ten
thousand parameters. Micro-optimising the closure, fusing the meter computation, or caching the
batch across calls buys nothing measurable. The cost is entirely in rollout evaluations, and only
three levers move it: the window count, gradient checkpointing, and whether the compiled rollout can
be kept. Everything else above is about being correct at scale and being testable, not about being
fast.

---

# STATUS 2026-09-04: the rewrite above is implemented

`model_augmentation/fit_systems/lbfgs_polish.py` was rewritten to this design.
**49 tests pass** (`python -m unittest discover -s scripts/gantry/gantry_dynamic/tests -p
test_lbfgs_polish.py`, ~148 s), up from 27, and the suite is now in three tiers: pure, checkpoint
I/O, and fixture-gated.

| Item | Status |
|-|-|
| 1. Batch materialised in full | **FIXED.** `FixedBatch` raises `make_training_data`'s stride so the build scales with the request. `window_count` computes the exact count from record lengths, mirroring `system_data.py:305-330`, and is pinned by a unit test against the measured 66,612. |
| 2. `lbfgs_windows=4096` vs `use_f64=True` | **OVERSTATED, and fixed anyway.** See the correction below. |
| 2 (as fixed) | **FIXED, two ways.** `lbfgs_chunk` splits the batch for gradient accumulation, and the config comments now state the interaction and that `checkpoint_chunk` applies here for free. |
| 3. Eager forced, 6.5x given up | **FIXED.** `_eager` takes `active=`, defaults to off, and the phase runs compiled when compilation is present. |
| 4. Determinism assumed on GPU | **FIXED.** `_assert_deterministic` evaluates the closure twice before any step, and on failure falls back to eager, re-checks, and aborts with the roll-back intact if that also fails. `deterministic` and `used_compiled` are recorded in the outcome. |
| 5. Two validations per phase | **FIXED.** `val_before` is an argument; `run_lbfgs_polish` passes `bestfit`, which `fit()` already paid for. |
| 6. `_named_params` duplicates | **FIXED.** Deduplicated by `id()`, with a test. |
| 7. `val_sys_data=None` accepts unconditionally | **FIXED, and the default reversed.** Without a validation measure the phase now ROLLS BACK unless `accept_without_validation=True` is passed explicitly. |
| 8. `lbfgs_windows=0` unimplementable | **FIXED.** Chunked accumulation makes full batch reachable, and `__post_init__` refuses `lbfgs_windows=0` with `lbfgs_chunk=None` rather than letting it OOM. |
| A. `FixedBatch` | done, with `effective_stride` and `bytes_on_device` in the outcome dict |
| B. `decide` as a pure function | done; 9 cases including the negation case and both no-validation branches |
| C. `PolishSpec` | done; `RunConfig` stays flat, `run_lbfgs_polish` maps once |
| D. Determinism self-check | done, with automatic eager fallback |
| E. Share the extra-array binding with `fit()` | **NOT done, deliberately.** Still requires the pinning test on `fit()` first. This is the one open item from the rewrite. |
| F. A test that asserts cost | done: `test_batch_build_scales_with_the_REQUEST_not_the_dataset` |

New evidence from the rewrite, worth keeping:

- **Chunked accumulation is exact, and it is now measured, not argued.**
  `test_chunked_gradient_equals_the_single_chunk_gradient` splits 64 windows into 4 chunks and
  compares against the single-chunk gradient: relative difference below 1e-4 in float32, which is
  reduction-order noise. So `lbfgs_windows=0` costs memory, not accuracy.
- The window arithmetic is exact rather than estimated. At `n_windows=4096` out of 66,612 the phase
  now builds ~4,172 windows instead of 66,612, a 16x reduction in the build, and the ratio grows
  with `nf` because the discarded windows are the expensive ones.

Remaining, and both need the cluster, not this machine:

1. Run the suite on the GPU. `test_closure_is_bit_deterministic` and the phase's own self-check are
   the two that matter; the self-check means a non-deterministic CUDA backend now degrades to eager
   with a printed line instead of silently corrupting the line search.
2. Size `lbfgs_windows` and `lbfgs_chunk` against the wall and the card, then run the polish from
   the job-81262 checkpoint. The bar is a sim-RMS improvement larger than that run's own +-0.1%
   validation oscillation, so roughly 0.3%.

---

## Correction 2026-09-04: item 2 was aimed at the wrong card

Item 2 called the `lbfgs_windows=4096` plus `use_f64=True` combination "probably an OOM", citing
`docs/gpu-nf12000-feasibility-2026-08-31.md:155`, which measures an **8 GB RTX 2080** on `oahu`.

Job 81262, the run this whole exercise is about, did not run there. Its own log records
`device=cuda (Quadro RTX 6000)`, i.e. the `hawaii` node with **24 GB**. On that card 4096 windows
in float64 has roughly 3x the headroom the item assumed, so the concern is real but not blocking,
and calling it BLOCKING was wrong.

Two things follow, and neither is undone by the correction:

1. `lbfgs_chunk` and the `lbfgs_windows=0` fix stand on their own merits. Chunking is what makes
   Drenth's full-batch setting reachable at all, independently of any card.
2. **The memory table is a property of the card**, so the probe must run on the partition the
   production run will use. `hawaii` = RTX 6000 24 GB, `oahu`/`mpi` = RTX 2080 8 GB,
   `lanai`/`molokai` = A100 40-80 GB. The runner defaults to `hawaii` for that reason.

---

# MEASURED ON THE GPU, job 81463 (blade3, Quadro RTX 6000 24 GB, nf=400)

The probe (`scripts/gantry/lbfgs/lbfgs_gpu_probe.py`) ran clean after two failures worth recording,
both mine and both the same shape: an assumption about an environment this machine cannot exercise.

* **81461** built the model from the cluster's `CFG` (`nx_ann=2`, 16x2) and tried to load the
  `nx_ann=8`, 24x3 checkpoint into it. Fixed by `infer_arch_from_hfn` / `config_for_checkpoint`:
  the architecture now comes from the checkpoint, never from whichever revision of the entry file
  is checked out.
* **81462** hit `torch._dynamo.exc.UserError: Dynamic control flow is not supported`.
  `Interconnect.output_only` initialises lazily on first call (`interconnect.py:288-290`) and that
  build runs numpy graph analysis with data-dependent control flow (`utils.py:38`). A training run
  never meets it because `fit()` validates eagerly first; anything going straight to the compiled
  rollout does, INCLUDING a real `start_phase='lbfgs'` resume. Fixed by `warm_up_lazy_init`, called
  inside the phase itself as well as the probe.

## Q1 determinism: all four combinations pass, loss AND gradient

```
float32  compiled  1.24820853653062613e-09 x3   deterministic, gradient bit-identical
float32  eager     1.24818888558309027e-09 x3   deterministic, gradient bit-identical
float64  compiled  1.24820887956275600e-09 x3   deterministic, gradient bit-identical
float64  eager     1.24820887956274731e-09 x3   deterministic, gradient bit-identical
```

The strong-Wolfe precondition holds on CUDA, compiled included, so `force_eager=False` stays and
the phase keeps the compile speedup. Item 3 and item 4 of the defect list are both closed by this,
and by measurement rather than by argument.

Incidental: compiled and eager agree to 8 significant figures in float32 and 15 in float64, which
is the same "not bit-identical, numerically identical" result D-169 recorded for training.

## Q2/Q3 cost and memory, eager, per closure

| windows | built | stride | MB peak f32 | MB peak f64 | s, 1 chunk | s, chunk=512 |
|-|-|-|-|-|-|-|
| 512  | 518  | 1300 | 178.8  | 339.8  | 4.05 | 4.05 |
| 1024 | 1036 | 650  | 339.7  | 661.6  | 4.06 | 8.20 |
| 2048 | 2086 | 320  | 661.5  | 1306.2 | 4.13 | 16.59 |
| 4096 | 4172 | 160  | 1306.1 | 2595.3 | 5.56 | 33.75 |
| 8192 | 8330 | 80   | 2595.2 | 5406.0 | 4.12 | 69.06 |

Full window set at the config stride: 66612.

**Three results, and the first one inverts the design assumption.**

1. **Closure time is FLAT in window count.** 512 and 8192 windows both cost ~4.1 s, because the
   rollout is dispatch-bound: nf=400 sequential steps of ~2400 tiny ops each, and the batch axis
   rides along for free. So a bigger batch is nearly costless and the only limit is memory. That
   reverses the usual reasoning: **take as many windows as fit**, and the 256-window CPU experiment
   that improved the training loss while worsening sim-RMS was a sampling artefact, fixable at
   almost no cost.
2. **Chunking costs LINEARLY in chunk count and should stay off here.** At 8192 windows, chunk=512
   turns 4.12 s into 69.06 s for a 21x memory saving this card does not need. Each chunk is another
   full sequential rollout. It also introduces a second tensor shape (the remainder), i.e. a second
   torch.compile compilation. Keep `lbfgs_chunk=None` unless memory forces it; it remains the only
   route to `lbfgs_windows=0`.
3. **float64 is free in time and exactly 2x in memory.** Both dtypes give ~4.1 s per closure.
   This settles the self-contradiction in `docs/gpu-nf12000-feasibility-2026-08-31.md`: line 17
   ("float64 measured free (0-4%)") is right and line 203 ("a ~30x cliff" on flipping `use_f64`) is
   wrong for this workload, because the FP64 arithmetic rate never binds on a dispatch-bound loop.
   That document should be corrected; it is not this work's to edit without a separate go-ahead.

## Defaults set from this

`lbfgs_windows=16384` (~10.8 GB in float64, ~45% of the card, same 4.1 s per closure as 512) and
`lbfgs_chunk=None`. 8192 (~5.4 GB) is the conservative fallback. The full 66612-window set would
need ~44 GB in one graph and stays out of reach without chunking.

Budget for the real phase, extrapolating: `lbfgs_max_iter=500` at 1 to 3 closures per iteration is
500 to 1500 closures. Eager that is 35 to 100 minutes; compiled, at the ~6.5x D-169 measured for
training, roughly 5 to 15 minutes plus a one-off compilation of order 500 s. Two validations
(~162 s each, one of which is now skipped by passing `bestfit`) sit on top. A 2 h wall is ample.
