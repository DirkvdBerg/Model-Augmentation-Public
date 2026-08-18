# Closed-loop controller: clean implementation in `model_augmentation/`

Design target for lifting the closed-loop training loop out of `scripts/gantry/closed-loop-controller/`
and into the framework. This describes the implementation we **want**, not a refactor of the one we
have. Where the two disagree, this document wins and the current code is the thing that changes.

Scope: the controller as a stepped subsystem, the residual-form rollout, and the fit-system class
that uses it. Not the design of `Cfb` itself, which is the FP model's and stays in `scripts/gantry/`.

---

## 1. Rules this design follows

1. **No monkey patching.** No `attach()`, no `fit_sys.__class__ = type(...)`, no classes created at
   runtime. Every class is defined at module import time, so `pickle` resolves it by name and
   checkpoints load without a shim.
2. **No dependency on `multiple_shooting.py`.** That file is ours, is a documented exact no-op in
   every production run (`n_seg=1`, `defect_weight=0`, `defect_acc_weight=0`), is kept only for a
   set of defect diagnostics, and is already marked CANDIDATE FOR REMOVAL in
   `gantry_dynamic/model.py`. The closed-loop class must not sit on top of it, must not inherit
   from it, and must not be affected by its removal.
3. **One implementation of the loop.** Training, validation and checkpoint selection call the same
   rollout function. They cannot disagree about what the loop is because there is only one.
4. **No copy-pasted loss terms.** `param_loss` and the orthogonality penalty are picked up in
   exactly one place. The current code re-implements both in three files; a fourth copy is how the
   thesis contribution gets silently dropped from the objective.
5. **The framework does not know about the gantry.** `model_augmentation/` knows how to step a
   controller alongside a model. It does not know what `ruleOfThumb` is, what `Y_op` means, or
   which records exist.
6. **Composition over inheritance for behaviour.** How the model is driven is a value on the
   instance, not a position in a class chain. No new link is added to
   `ParamLoss -> OrthLoss -> MultipleShooting`, and the closed loop must not know that chain exists.
7. **Markers.** New files carry `__project_origin__ = "added"` at module top. Any edit inside an
   existing file carries `# CHANGED: <reason>` inline.
8. **Attribution in the code.** The formulation is Kessels'. Every line that implements one of his
   equations carries a `# THEORY:` label naming the equation and thesis page, and the one place we
   deviate says so explicitly. Section 7.
9. **Correct, then clean, then fast.** Runtime improvements are welcome and never bought with
   clarity. No optimisation without a profile; no optimisation that adds a second code path for
   the same thing. Section 3.8.

---

## 2. The structural problem to solve

`SSE_Interconnect.loss` does four things in one method: encode the initial state, roll the model
forward, reduce to an MSE, and add parameter regularisation. Each subclass in the chain adds its
term by calling `super().loss()`:

```
SSE_Interconnect            encode + rollout + MSE + Jan's isinstance param pickup
  SSE_Interconnect_ParamLoss    + generic param_loss sweep
    SSE_Interconnect_OrthLoss     + orthogonality penalty
```

Closing the loop changes **only the rollout**, which is the one thing buried in the middle of the
base method with no seam. That is the entire reason the current code overrides `loss()` wholesale
and then re-adds `param_loss` and the orth penalty by hand.

**The fix is to create the seam, not to work around its absence.**

---

## 3. Target structure

### 3.0 Feasibility, checked against the code

All three call sites the design needs are inside `SSE_Interconnect.fit`, which lives in
`interconnect.py` and is already a `# CHANGED:` replacement of deepSI's:

```
interconnect.py:568   data_train = self.make_training_data(...)
interconnect.py:623   Loss       = self.loss(*train_batch, **loss_kwargs)
interconnect.py:518   Loss_val   = self.cal_validation_error(val_sys_data, ...)
```

`make_training_data` is defined on deepSI's `SS_encoder_general` and `cal_validation_error` on
deepSI's `System_fittable`; **neither needs to be edited**. `SSE_Interconnect` overrides both in
our file, calls `super()`, and delegates. `loss` is already overridden there. So the plan requires
edits to `interconnect.py` only, and no change to the installed deepSI package.

### 3.1 The seam, in `interconnect.py`

Extract the rollout from `SSE_Interconnect.loss` into an overridable method, and have `loss()` call
it. One `# CHANGED:` edit in Jan's file, no behaviour change:

```python
def simulate(self, x, ufuture, yfuture=None, **kw):
    """Roll the model forward from x. Returns y_pred, (batch, nf, ny).

    Overridable seam: a subclass that changes HOW the model is driven (closed loop,
    teacher forcing, a different integrator) overrides this and inherits every loss
    term unchanged. yfuture is passed so a driven rollout can use it; the open-loop
    default ignores it.
    """
    hfn = self.hfn                       # bound once: 400 Module.__getattr__ otherwise
    ys = []
    for u_t in ufuture.unbind(1):        # ONE dispatch for all nf views, not nf selects
        yhat, x = hfn(x, u_t)
        ys.append(yhat)
    return torch.stack(ys, dim=1)

def loss(self, uhist, yhist, ufuture, yfuture, **kw):
    x = self.encoder(uhist, yhist)
    y_pred = self.simulate(x, ufuture, yfuture, **kw)
    loss_MSE = torch.nn.functional.mse_loss(y_pred, yfuture)
    ...   # parameter regularisation exactly as today
```

`ufuture.unbind(1)` rather than `ufuture[:, t]`: the latter is `nf` separate `select` dispatches,
the former is one. `ufuture` carries no grad so this changes no autograd node, only dispatch count,
but it is also the shorter line. Binding `self.hfn` to a local before the loop removes `nf`
`nn.Module.__getattr__` calls, measured at 1.3 us each over 31623 calls in the step-0 profile.

Three constraints on this edit:

- **Parity, and it is one ulp rather than exact.** MEASURED, migration step 1
  (`references/step1_reference.npz`): the base loss's mean of per-timestep `mse_loss` values and a
  single `mse_loss` over the stacked tensor give a **bit-identical value** (difference exactly
  0.000e+00 on both the zero-ANN and perturbed-ANN arms) but **not** a bit-identical gradient:
  `1 - cos = 6.1e-15`, gradient-norm ratio off by 4.9e-09, largest elementwise difference
  1.207e-07 relative to `max|g|`, i.e. exactly one float32 ulp. So section 3.1's original claim of
  exact parity was optimistic, and R1's bit-identical contract cannot survive the reduction change.
- **Therefore the seam and the reduction are two steps, not one** (5.2a and 5.2b). Step 2a extracts
  `simulate()` and keeps the per-timestep reduction verbatim, so R1 stays bit-identical and the
  gate can still separate reordering from a real change. Step 2b then replaces the reduction with
  the single `mse_loss`, as its own change, whose entire content is a reordering and whose evidence
  is the one-ulp measurement above. The end state is the single `mse_loss`: it is one dispatch and
  one autograd node against 400 plus a stack and a mean, so it is both the cleaner and the faster
  form, and backward is 53 % of a step (3.8) so deleting 400 forward nodes deletes 400 backward
  nodes with them. The per-timestep form never survives into the final code.
- **Nothing else moves.** The isinstance-based parameter pickup in the base stays exactly where it
  is, including its double-count caveat.

One cost the seam adds regardless of reduction: `simulate()` materialises the stacked
`(batch, nf, ny)` prediction, which today's loss never forms because it reduces to scalars per
timestep. 1.2 MB at batch 256. Negligible, but it is a real change in what the graph holds.

### 3.2 No new fit-system subclass. The driving strategy is composed in.

**Rejected: `SSE_Interconnect_ClosedLoop(SSE_Interconnect_OrthLoss)`.** Subclassing the orthogonality
class to change how the model is driven couples two unrelated concerns. The existing chain already
has this defect: `ParamLoss -> OrthLoss -> MultipleShooting` makes every feature a subclass of the
previous one, so a new feature inherits everything before it whether or not it is related, the
order of the chain becomes load-bearing, and deleting a link in the middle is a breaking change.
Adding a fourth link would make the closed loop depend on the orth penalty for no reason. Do not.

**Instead, `simulate()` delegates to a simulator object held on the instance.** How the model is
driven becomes a value, not a position in an inheritance chain:

```python
# in SSE_Interconnect, alongside the existing `orth_penalty = None` pattern
simulator = None            # None = open loop, the default, an exact no-op

def simulate(self, x, ufuture, yfuture=None, **kw):
    if self.simulator is None:
        return open_loop_simulate(self.hfn, x, ufuture)
    return self.simulator(self, x, ufuture, yfuture, **kw)
```

This is not monkey patching: `simulator` is a declared class attribute with a documented default,
set after construction exactly as `orth_penalty` already is (D7.1/D7.8). The object assigned to it
is an ordinary class defined at import time, so `pickle` resolves it by name.

Consequences, and they are the reason for the change:

- The closed loop works with **whatever** loss class the pipeline builds. It does not know about
  `OrthLoss`, `ParamLoss` or `MultipleShooting`, and is unaffected when any of them is removed or
  reordered.
- `param_loss`, the orthogonality penalty and the MSE reduction are never mentioned anywhere in the
  closed-loop code. They cannot be dropped because they are never touched.
- Open loop, closed loop and any future variant (teacher forcing, a different integrator) are
  siblings, not a chain.

### 3.3 New file: `model_augmentation/fit_systems/closed_loop.py`

```
__project_origin__ = "added"

DiscreteController      one biproper LTI controller as a batched state-space step
ControllerBank          the distinct controllers for ALL records, gathered per batch
closed_loop_rollout     the residual-form rollout, free of deepSI and of any plant
ClosedLoopSimulator     the simulator object assigned to fit_sys.simulator
```

**The bank stores the controller in NORMALISED coordinates, not physical.** `Cfb` is linear, so the
denormalise / filter / renormalise sandwich the current `step` performs every timestep can be
absorbed into the matrices once at construction:

```
B' = B * ystd[None, :]         C' = C / stdu[:, None]         D' = D * ystd[None, :] / stdu[:, None]
```

This removes one multiply, one divide and two `nn.Module.__getattr__` per timestep (2 of the 8
tensor ops in `step`, see 3.8) and it is the cleaner object: one controller in one coordinate
system, instead of a physical controller wrapped in a per-step conversion. The
`# THEORY: Kessels (2025) Eq. (5.13c)` label moves to `__init__`, where the scaling is now applied
once, which states the physics in the place it happens rather than restating it 400 times per
window.

The units gate does not weaken, it strengthens: `check_units` **unfolds** the stored matrix,
`D_phys = stdu[:, None] * D' / ystd[None, :]`, and asserts against `Dc @ e` exactly as today. One
line, no duplicated state, and it now tests the folding as well as the units. Do NOT keep a second
set of physical buffers for the gate; two representations of one object is how they drift apart.

**Stack `[C'; A]` and `[D'; B]` once, and step with two batched matmuls.** With the normalisation
folded in, one timestep is

```python
out = torch.baddbmm(torch.bmm(M2, e.unsqueeze(-1)), M1, xc.unsqueeze(-1)).squeeze(-1)
u_fb, xc = out[:, :nu], out[:, nu:]
```

`baddbmm` computes `input + b1 @ b2` in a single dispatch, so two matmuls and an add cost two
operations rather than five. Together with the folding this takes `step` from **8 tensor ops, 4
einsum string parses and 2 attribute lookups down to 2 real ops plus free views**. Acceptable only
if `M1` and `M2` are assembled from named slices in `__init__` and the two outputs are named at the
call site; if it turns into index arithmetic, stop at plain `bmm` plus add (3 ops) and keep the
clarity. `einsum` is rejected here for a specific reason, not a stylistic one: it re-parses its
subscript string on every call, measured at 15 us per call on top of the 63 us kernel.

**ONE bank, over every record.** `Cfb` is per trajectory. Train versus validation is not an axis:
a controller belongs to a record, and a record is used for training or for validation, which is a
property of the split and not of the controller. The current code builds two banks
(`cl_step6_run.py`: `bank_tr` from `train_names`, `bank_va` from `val_names`) only because
`rec_ix` is a position in a per-list array, so index 0 means `T1` in one context and `V1` in the
other, and each context then needs its own map.

Build the bank over all 22 records and index it globally and the second bank disappears. One
object then serves training, validation and selection, which is what lets `ClosedLoopSimulator`
own both the rollout and the validation score.

```python
class ClosedLoopSimulator:
    """Cfb closed around the model during the rollout, residual form.

    Assigned to fit_sys.simulator. Knows nothing about the loss, the penalties, or
    which fit-system class it is attached to. Holds no model handles: fit_sys.hfn and
    fit_sys.output_only are resolved at call time, see 3.6c.
    """

    def __init__(self, bank, val_records=()):
        self.bank = bank                       # ALL records, not a train/val subset
        self.val_records = val_records         # ordered (name, sys_data), see 3.6

    def __call__(self, fit_sys, x, ufuture, yfuture, ctrl_ix=None, **kw):
        # `simulate()` returns y_pred ONLY. closed_loop_rollout keeps returning
        # (y_pred, x_final, xc_final); the loss drops the last two here and
        # validation_error calls the rollout directly for what it needs. No result
        # object, no state parked on the simulator, nothing to keep in sync.
        y_pred, _, _ = closed_loop_rollout(fit_sys.hfn, fit_sys.output_only,
                                           ufuture, yfuture, x, self.bank, ctrl_ix)
        return y_pred

    def validation_error(self, fit_sys, val_sys_data, validation_measure):
        """Closed-loop free run per record, scored in metres. Same rollout as __call__."""

    def augment_training_data(self, data, sys_data, fit_sys, **kw):
        """Append the per-window controller index. Identity for simulators needing nothing."""
        return list(data) + [window_controller_index(sys_data, self.bank, fit_sys, **kw)]
```

**`ctrl_ix`, not `rec_ix`.** The fifth array carries the row in the deduplicated controller stack,
resolved at data-build time by the code that is slicing a named record and therefore knows the
answer. After that nothing downstream carries a record concept, and `ControllerBank` loses
`rec_to_ctrl` and does one job.

Be clear about what this does and does not buy: it removes one indirection and the second bank. It
does **not** make the window-count derivation safer, because that derivation is needed either way
to build the array at all. The safety comes from the assertions in 3.5.

The last method needs the `make_training_data` seam: the base ends with
`return self.simulator.augment_training_data(data, sys_data, self, **kw)` when a simulator is
present, and returns `data` unchanged otherwise.

### 3.4 The output map: evaluate the output's dependency cone

`closed_loop_rollout` needs `y = h(x)` **before** it can form `u`, and `hfn(x, u)` returns
`(y, x_next)` together. The current code reverse-engineers `h` with `nx + 1` probe forward passes.

**Measured structure of the interconnect** (from the built gantry model, `Interconnect.forward`
computes `output_signals[k]` in `order_output_signal_computation` order):

```
order: [2, 3, 4, 0, 1]
out 0 (xp)                  <- [4, 2]        ANN and Gantry_State_Block
out 1 (y)                   <- [3]           Linear_Output_Block only
out 2 (Gantry_State_Block)  <- [0, 1]        x, u
out 3 (Linear_Output_Block) <- [0, 1]        x, u
out 4 (Static_ANN_Block)    <- [0, 1]        x, u
```

So `y` has a small dependency cone: signal 3 and, through it, `x` and `u`. Computing `y` requires
one `Linear_Output_Block` forward, a matmul, and does **not** require the ANN or the state block.
That is the real cost saving, not the avoidance of a second call.

Clean version: `Interconnect` gains a method that evaluates only the dependency cone of the output
signal. The cone is computed once at init from `output_ix_sorted_input_ix_dependencies`, so the
method is generic and does not hard-code this model's graph:

```python
def output_only(self, x, u=None):
    """y from the output signal's dependency cone alone. u defaults to zeros; see D_d = 0."""
```

`identify_output_map` then survives only as a **test** that this agrees with the full forward,
which is where it belongs.

**Two rules on the implementation, both about cost.** The cone must be resolved **once at
`__init__`** into a plain tuple of signal indices; re-deriving it inside the method means a graph
traversal `nf` times per window, which would cost more than the forward it is replacing. And once
it is a fixed short list, **do not optimise it further**. The step-0 profile puts the current
`AffineOutput` (one matmul plus one add) at 0.4 % of a training step, so even a generic cone
evaluation three times slower than that is about 1 % of a step. That 1 % buys the deletion of three
standing assumptions (output affine, output frozen, no feedthrough) and the three `cl_plant` gates
that currently guard them. Take the general version and stop.

**One finding that changes an earlier claim.** `u` *is* wired into the output block's input:
`sig3 <- u` is a (9, 3) matrix with entries of magnitude 1, i.e.
`connect_block_signals(out_phys, ["u"], ["y"])` in `model.py` really does route the input into the
output path. `D_d = 0` therefore holds because the `Linear_Output_Block`'s own coefficients are
zero on those columns, **not** because the graph forbids a feedthrough. So the numerical
`check_no_feedthrough` gate is not redundant with a structural argument and must be kept. The
physical argument (position output, forces reach it through two integrators) explains why the
coefficients are zero; it does not prove the wiring is.

### 3.5 Record identity, and asserting it

The controller differs per record, so building `ctrl_ix` requires knowing how many windows each
record produced. That derivation is unavoidable and it is the fragile part: `_record_index`
re-implements the two branches of deepSI's `system_data.py` window construction, and it has
already been wrong once, an off-by-one from the right-hand encoder extension that would have
attached the wrong controller to most of the training set. Nothing crashes when that happens; the
loss still decreases and the model is fitted inside the wrong loop.

Keep the fifth-array mechanism, because deepSI's `My_Simple_DataLoader` slices every array in the
list by the same shuffled ids. Replace the single total-count assert with three checks, cheap and
run once at setup, each failing differently:

1. **Per-record counts, not the sum.** Call `to_hist_future_data` on each record individually and
   compare each length against the derived count. Today only `sum(counts) == len(data[0])` is
   asserted, so two compensating per-record errors pass.
2. **Identity by name.** Build the array from `[(name, count), ...]` and map names to controller
   rows, rather than relying on a position meaning the same thing in two places.
3. **Verify by content.** For the first and last window of each record, assert the window's
   `ufuture`/`yfuture` slice equals that named record's raw data at the expected offset. This is
   the one that makes the identity explicit: it checks against the data instead of against a
   re-derivation of deepSI's conventions, so it catches a stride change, an off-by-one, or a
   reordering inside `System_data_list`, none of which a count check can see.

The count-derivation comment stays, since it documents why the arithmetic is what it is. It just
stops being the only line of defence.

**Rejected alternative, and it should be recorded because it is the obvious clean idea.** Derive
the controller from the window's own measured `Y` instead of a record index, which would delete
the plumbing entirely. This is wrong. The machine's controller was frozen at the record's nominal
`Y_op` for the whole record. Deriving it per window would give a *different operator* on the model
side than the machine used, which breaks the condition the residual identity depends on. It would
be silently wrong on exactly the `ysweep` records where `Y` moves. Do not do it.

### 3.6 Validation: the third seam, and the second monkey patch removed

`ClosedLoopValidator.install()` currently replaces `fit_sys.cal_validation_error` at runtime. That
is the same pattern banned for the loss and it goes too.

The reason it exists is real, and it decides the design. deepSI's `cal_validation_error` calls
`self.apply_experiment(val_sys_data)`, which drives the model through the `System` interface, `u`
in and `y` out, one step at a time via `measure_act_multi`. That interface **cannot** carry
`y_data`, and the closed-loop free run needs `y_data` to form the residual. So the closed loop
cannot be expressed as an `apply_experiment` and the override point has to be
`cal_validation_error` itself. deepSI's own docstring anticipates this: "User given callback.
(overwrite this function?)".

Clean version: make it a seam on the same object that owns the training rollout.

```python
# in SSE_Interconnect, # CHANGED:
def cal_validation_error(self, val_sys_data, validation_measure='sim-NRMS'):
    if self.simulator is not None and hasattr(self.simulator, 'validation_error'):
        return self.simulator.validation_error(self, val_sys_data, validation_measure)
    return super().cal_validation_error(val_sys_data, validation_measure)
```

`ClosedLoopSimulator` then carries `validation_error` alongside `__call__`. This is stronger than
merely removing a monkey patch: **training and validation become two methods on one object sharing
one `closed_loop_rollout`**, so design rule 3, one implementation of the loop, is enforced by
construction rather than by discipline. `ClosedLoopValidator` and its `install`/restore pair
disappear.

#### Which controller belongs to which validation record

`cal_validation_error` receives only `val_sys_data`, a `System_data_list` carrying no record
names, so the simulator cannot tell which trajectory it is scoring and therefore which `Cfb` to
use. In training this is solved at data-build time by `ctrl_ix`; validation needs the same idea.

**The principle is the one from 3.5: resolve identity once, at the boundary where the pipeline
constructs the data and therefore knows the answer, and assert it by content wherever it is used.**

The pipeline already holds the ordered validation list, it is the same object it built `val_names`
and the bank from and the same object it passes into `fit()`, so the mapping is free:

```python
ClosedLoopSimulator(bank, val_records=[(name, sys_data), ...])
```

Then `validation_error` checks it rather than trusting it, once, cached after the first call so it
costs nothing per epoch:

1. **Count.** The number of records in the incoming `val_sys_data` equals the number registered.
   Catches a changed split.
2. **Position to name.** Record `i` in the incoming list corresponds to registered entry `i`.
3. **Content.** The incoming record's `u` and `y` match the registered one over the first and last
   `nf` samples. This is what makes it explicit: if deepSI reorders, or the validation list is
   rebuilt in a different order, or `VAL_FILES` is changed, it fails loudly instead of scoring
   every record through the wrong controller.

The framework learns nothing about the gantry by this: it receives `(name, sys_data)` pairs and a
bank, never `Y_op`, `ruleOfThumb`, or a record list. Rule 5 holds.

**Two alternatives, rejected and recorded.**

*Tag the `System_data` objects with a name at load time*, so identity travels with the data.
Cleanest in principle, but `fit()` calls `norm.transform(train_sys_data)`, which returns new
`System_data` objects, so a plain attribute is dropped in the middle of the pipeline and surfaces
as a silent `None` rather than an error. Surviving that needs a subclass of deepSI's data type,
which is a larger change to the data path than this problem justifies.

*Derive the controller from the record's own measured `Y`*, which would delete the plumbing
entirely. Rejected for the same reason as in 3.5: the machine froze `Cfb` at the record's nominal
`Y_op`, so a per-trajectory derivation from measured `Y` is silently wrong on exactly the `ysweep`
records, where `Y` moves.

### 3.6b The other `cal_validation_error` patch, in the production path

The closed-loop validator is not the only thing patching that attribute.
`gantry_dynamic/training.py`'s `_install_nf_val_probe` replaces `fit_sys.cal_validation_error` with
an `_NfProbe` on **every production run**, restores it in a `finally`, and carries a `__reduce__`
that returns a no-op so the object survives pickling. `cl_validation.py` documents the collision:
whichever of the two is installed last decides selection, which is why the validator has to be
outermost and call the probe for its side effects.

Two different concerns are riding on one attribute:

- **selection**, the scalar that decides the best checkpoint;
- **diagnostics**, extra measurements logged per validation whose return value is discarded
  (`_NfProbe` returns `self.orig(...)` untouched).

Only the first belongs on the simulator. The second gets its own declared extension point:

```python
# in SSE_Interconnect, # CHANGED:
validation_probes = ()          # class default, empty = no-op

def cal_validation_error(self, val_sys_data, validation_measure='sim-NRMS'):
    if self.simulator is not None and hasattr(self.simulator, 'validation_error'):
        value = self.simulator.validation_error(self, val_sys_data, validation_measure)
    else:
        value = super().cal_validation_error(val_sys_data, validation_measure)
    for probe in self.validation_probes:
        probe(self, val_sys_data, value)      # side effects only, value never replaced
    return value
```

That removes a monkey patch from the production training path, removes the `__reduce__` hack, and
removes the ordering hazard: probes cannot change the selection value because the seam does not let
them.

### 3.6c Resolve `hfn` at call time, not at attach time

Today `ModelStep` and `AffineOutput` capture `hfn` when `attach()` runs. deepSI's
`checkpoint_load_system` does `self.__dict__ = torch.load(file)`, so after a `_best` reload those
captured handles point at the old modules. That trap is documented in `cl_lr_probe.py` and
`cl_plot_step6.py`, and it is why `cl_plot_step6.py` loads state dicts rather than adopting the
pickled `__dict__`. The same replacement silently loses the monkeypatched `cal_validation_error`,
documented in `cl_sanity.py` and `cl_step6_run.py`.

`ClosedLoopSimulator` must therefore hold **no** model handles. It takes `fit_sys` as its first
argument and reads `fit_sys.hfn` and `fit_sys.output_only` at call time. It holds only the
`ControllerBank`, which is parameter-free.

This is not a detail. It means the whole stale-handle class of bug disappears, and because
`ClosedLoopSimulator` is an ordinary importable class it is pickled and restored by
`checkpoint_load_system` like any other attribute, instead of being silently dropped the way the
patched method is today.

### 3.6d The ANN off-switch: REVISITED, and the patch is gone after all

`cl_plant.zero_the_ann` replaced `ann.forward` with a lambda and handed back a restore callable.
This section originally DECIDED to keep it, on an argument that still looks fair: it is used only
by gates, baselines and diagnostics, never in the training or validation path; it is scoped and
restored immediately by its caller; and if it failed, the gate it serves would be visibly wrong
rather than silently wrong, unlike the patches removed elsewhere.

It was revisited anyway, because a version with none of those caveats turned out to be available.
`ann_output_zeroed(fs)` is a context manager that zeroes the ANN's output `Linear` layer, weight
and bias, and restores it in a `finally`:

- it mutates PARAMETERS, not CODE. `zero_init_feed_forward_nn` zeroes exactly that layer at
  construction, so the block puts the model into the state it is BORN in, which is also the state
  every gate assumes when it says "with the ANN at zero the augmented model IS the baseline". The
  gate now exercises a real configuration instead of a simulated one;
- it is scoped by the `with`, so it cannot leak on an exception, which the install/restore pair
  could;
- it costs the production path NOTHING.

That last point is why the `output_scale` buffer this section used to suggest as the "nice to
have" was NOT taken: a buffer multiplied into every ANN forward is 400 extra dispatches per
window, on the training path, to serve a diagnostic, and 3.8 measured that dispatch count is the
entire cost in this regime.

Gated (`scratchpad/zero_ann_test.py`): output exactly 0.0 inside the block, `forward` not
replaced, parameters restored BIT-FOR-BIT afterwards, and restored correctly when the body raises.
One thing that test caught about itself first: its magnitude helper originally drew fresh `randn`
per call, so "is it restored?" was unanswerable; the inputs are seeded now.

### 3.7 Multiple shooting must route through the same seam

`SSE_Interconnect_MultipleShooting.loss` overrides `loss()` and calls `self.hfn` directly inside
its segment loop, so it would never reach `self.simulate()`. Leaving that alone reproduces today's
bug in mirror image: today the closed-loop loss silently drops the defect terms, and after the move
multiple shooting would silently ignore the controller.

Bypassing is not clean. Change its inner loop to call `self.simulate()` on each segment. It is our
file, it is a few lines, and the result is that **exactly one rollout implementation exists in the
codebase**. A guard that raises INSTEAD of routing would also remove the trap, but it would leave
two rollout paths, which is the thing that caused the problem in the first place.

**Route, and then also raise, and they are not the same decision.** Routing is about there being one
implementation; the raise is about the SEMANTICS of a combination that routing makes reachable for
the first time. Once each segment goes through `self.simulate()`, a closed loop at `n_seg > 1` runs,
and nothing here says whether the controller state resets at a segment boundary as `x` is
re-anchored to the encoder. That is an unanswered modelling question, not an implementation gap, so
the combination raises until someone answers it (5.4, 9.3b). One rollout path, one refused
configuration, zero silent behaviour.

This does not make the closed loop depend on multiple shooting. The dependency runs the other way:
multiple shooting uses the seam, like everything else.

### 3.8 Computational cost: where it goes, and what is worth doing

**Rule: no optimisation without a profile, and none that costs clarity.** Every item below is either
a measured win, a cheap win that also makes the code more honest about the object, or explicitly
rejected. The order is: correct, then clean, then fast.

**The regime, and it decides every question below. MEASURED, migration step 0**
(`cl_step0_profile.py`, `runs/step0_profile.json`; batch 32, `nf = 400`, 6 threads, one
`loss()` plus `backward()`):

| op | calls | tottime | per call |
|-|-|-|-|
| `torch.matmul` | 4400 | 0.203 s | ~46 us profiled, ~33 us real |
| `torch.einsum` kernel | 1600 | 0.101 s | ~63 us |
| `einsum` Python wrapper, i.e. the subscript parse | 1600 | 0.024 s | ~15 us on top |
| `nn.Module.__getattr__` | 31623 | 0.041 s | ~1.3 us |
| `blocks.py:781 deriv` | 1600 | 0.903 s | ~560 us |

A `(32,8) @ (8,8)` matmul is 2048 FLOPs. At 33 us that is 60 MFLOP/s on a CPU capable of tens of
GFLOP/s, so **the arithmetic is free and we are paying 100 % overhead**: Python, the dispatcher,
autograd node construction, allocation. The consequence is a rule, not a preference:

> **Reduce the NUMBER OF TENSOR OPERATIONS, never the number of FLOPs.** An optimisation that cuts
> FLOPs while leaving the op count alone buys nothing in this regime.

**Where the time goes, measured rather than assumed:**

| item | prior | measured | verdict |
|-|-|-|-|
| `hfn` forward plus its backward | dominant | backward alone 53 % tottime, `blocks.py` 26 % | confirmed; inherent |
| controller step, 400x per window | "a few percent" | **8.1 %** of the forward (interleaved, IQR 7.4 to 12.7 %), 5.2 % of the full step but noise-limited | confirmed |
| ceiling on ANY controller optimisation | not stated | **7.6 %** (400 x `bank.step` alone, forward plus backward) | see below |
| `bank.gather`, once per batch | negligible | **0.002 %** | dead item |
| encoder, once per batch | negligible | **0.012 %** | dead item |
| `AffineOutput`, the `y = h(x)` accessor | not stated | **0.4 %** | see 3.4 |

Two caveats on those numbers, both real. The full-step difference is noise-limited on this machine:
per-pair differences ranged -0.6 to +3.0 s on a 3.3 s step, and a first attempt that timed the two
arms in separate blocks returned a physically impossible negative. Interleaving the arms fixed it
and the forward-only comparison is the low-noise version, which is why 8.1 % is the figure to quote.
And the profile ran at **batch 32 while production is 256**: dispatch count is batch independent
while FLOPs scale 8x, so every overhead-bound share above is an UPPER BOUND on its production share.
Re-run with `CL_BATCH=256` before spending anything here.

**The largest single item is not in scope.** `blocks.py:781 deriv` is 22 % of the profiled step in
1600 calls, four per timestep, four times larger than anything the closed loop touches. It is inside
the model. If runtime ever becomes the objective, that is the target, and the controller is a
rounding error beside it. Recording it here so that nobody spends a week fusing einsums while 22 %
sits in a per-timestep derivative helper.

**Already taken, do not lose it.** `cl_controller.rollout` calls `hfn` ONCE per step. The
predecessor `ClosedLoopLossMixin` called it twice, to work around the `y = h(x)` ordering, which
doubled the FP-plus-ANN forward cost of every step. Whatever else changes, the one-call property is
load-bearing and `output_only` exists to preserve it. The other property already exploited, and the
single most important one, is that the whole window batch rolls at once: batching is what PyTorch is
good at, and 256 windows cost barely more than one.

**Worth doing, and the justification is that each is also SHORTER code:**

- **Fold the normalisation into `B, C, D`** (3.3). Removes a multiply, a divide and two attribute
  lookups per timestep, and replaces a per-step conversion sandwich with one coordinate system.
- **Stack `[C; A]` and `[D; B]`, step with `bmm` plus `baddbmm`** (3.3). `baddbmm` does
  `input + b1 @ b2` in one dispatch, so `step` goes from 8 tensor ops, 4 subscript parses and 2
  attribute lookups to 2 real ops plus free views.
- **`unbind(1)` and locals bound outside the rollout loop** (3.1). One dispatch instead of `nf`
  selects, and `nf` fewer `Module.__getattr__` per rollout.

Together these take the controller path from roughly 8 % of the forward to roughly 2 to 3 %, i.e.
about 5 % of the forward and 2 to 3 % of a full step. That is small, and it is the honest reason to
take them: **if the only argument for a change here were speed, the profile says do not bother.**

**Explicitly rejected, with reasons:**

- **The block-diagonal `Cfb` storage.** Three SISO channels of order 3 means the `9x9` `A_c` has 27
  nonzeros of 81, so storing it as `(K,3,3,3)` cuts FLOPs 3x and the operation count by ZERO. In a
  dispatch-bound regime that buys nothing, and it is mutually exclusive with the stacking above,
  which is worth strictly more. This reverses the earlier entry in this section, which recommended
  it on a FLOP argument before the profile existed.
- **Preallocating and reusing the output-signal buffers in `Interconnect.forward`.** It allocates
  zeros for every signal on every call. Reuse means in-place writes, which breaks autograd. Not
  available.
- **Materialising per-window controller matrices.** Would be roughly 150 floats per window across
  ~47000 windows to avoid a gather measured at 0.002 % of a step. Precompute the *index*, not the
  matrices (3.3).
- **Optimising `output_only`** (3.4). 0.4 % today; the generic version buys three deleted
  assumptions.
- **Anything that trades the four seams for speed.** The seams are why there is one rollout
  implementation; a faster design with two would be a worse design.

**To measure, not assume:**

- **`torch.compile`.** PARTLY MEASURED. The stated risk, that `Interconnect.forward` builds its
  signal list dynamically and may not trace, **does not hold**: with `backend='eager'` (dynamo
  traces, codegen skipped) and with `aot_eager`, `hfn` traces cleanly and returns a BIT-IDENTICAL
  state, `max |dx| = 0.000e+00` on one step. What could not be measured is whether fusion helps,
  because `inductor` fails on this machine with `Compiler: cl is not found`, i.e. MSVC is absent
  and it cannot build the kernel it generated. That is an environment limit, not a property of the
  code, and the Linux cluster has a compiler.

  One number worth carrying: `aot_eager` ran at **0.22x, i.e. 4.5x SLOWER** than eager. Tracing
  overhead without fusion is a real cost, so "it compiles" is not "it is faster" and the inductor
  measurement is the only one that would settle it. Compile the **single step function**, not the
  400-iteration rollout, which would unroll into an enormous graph. Keep it optional, default off,
  never required for correctness, and treat anything compiled as a SEPARATE arithmetic path: R1
  bit-identity is not the right gate for it.
- **The step-0 profile at batch 256**, per the caveat above.

### 3.9 What stays in `scripts/gantry/`

- `RECORD_Y_OP` and `y_op_for`: a lookup table for one machine.
- `build_cfb_at` / `controller_ss`: the `ruleOfThumb` design rule, the frozen design plant, Tustin.
  This is the FP model's controller, not a framework capability.
- Every verification script in this folder.

The gantry pipeline constructs `Cfb` and hands the matrices to `ControllerBank`. The framework
never learns where they came from.

---

## 4. What disappears

| Today | Why it exists | After |
|-|-|-|
| `loss_variants.attach`, `cl_fitsys.attach` | avoid touching `build_model` | `fit_sys.simulator = ClosedLoopSimulator(bank)`, one assignment |
| runtime `type()` class + manual globals binding | so `pickle` can find the grafted class | no class is created at runtime |
| `loss()` overridden wholesale in 3 files | no rollout seam | nothing overrides `loss()` at all |
| `param_loss` + orth pickup copied 3x | consequence of the above | never mentioned by closed-loop code |
| a new subclass per feature | the `ParamLoss -> OrthLoss -> MultipleShooting` chain | the closed loop adds no link to the chain |
| two `hfn` calls per step (`ClosedLoopLossMixin`) | no `h(x)` accessor | one partial forward, the output's dependency cone |
| `identify_output_map` as mechanism | no `h(x)` accessor | survives as a test |
| `ClosedLoopValidator.install()` replacing `cal_validation_error` | `apply_experiment` cannot carry `y_data` | `cal_validation_error` seam delegating to the simulator |
| `_install_nf_val_probe` patching the same attribute in production | no diagnostics extension point | `validation_probes` tuple, side effects only |
| `_NfProbe.__reduce__` no-op | a patched bound method cannot pickle | nothing to pickle; probes are declared attributes |
| `ModelStep`/`AffineOutput` capturing `hfn` | handles taken at attach time | resolved from `fit_sys` at call time |
| training and validation as separate objects | no shared owner | two methods on one `ClosedLoopSimulator`, one rollout |
| `sys.path.insert` in every script | code lives outside the package | package import |

Eleven of the thirteen are consequences of missing seams, and there are four of them:
`simulate`, `make_training_data`, `cal_validation_error`, `validation_probes`.

Originally not in the table, deliberately: `zero_the_ann`, on the grounds that a patch in a GATE
fails visibly while a patch in the training or validation path fails silently and changes what
gets optimised or selected. That distinction is still the right one, and the patch went anyway,
because a context manager that zeroes the ANN's output layer has none of its drawbacks and no
cost. See 3.6d.

---

## 5. Migration steps, in order

Each step is separately verifiable, and the run in step 1 is the safety net for all the others.

0. ~~**Profile one training step**~~ **DONE**, `cl_step0_profile.py` ->
   `runs/step0_profile.json`. 3.8 now carries measurements. Outcome: priors confirmed, the
   controller path is 8.1 % of the forward with a 7.6 % ceiling on any optimisation of it, the
   block-diagonal storage is rejected, and the regime is dispatch-bound rather than FLOP-bound,
   which is what actually decides the remaining design choices.
1. ~~**Reference sets first.**~~ **DONE**, `cl_step1_reference.py` -> `references/`. R1 and R2
   recorded on the untrained production build (build fingerprint pinned, threads pinned to 1,
   arrays hashed); gate A re-run and captured to `references/gateA_*.txt`, all passing; the 4 kHz
   `export_controller.m` export added and checked. Two findings that change later steps: the
   reduction is one ulp from bit-identical (3.1, splitting step 2), and the 4 kHz coefficient
   comparison needed an exact-arithmetic third reference to be conclusive (5.1).
   R1 and R2 are regression nets; A remains the criterion that matters.
2a. **Extract the `simulate()` seam only**, `# CHANGED:` in `interconnect.py`, with the
   per-timestep reduction kept VERBATIM. R1 must be bit-identical, gradient included. This isolates
   the seam so the gate can still tell a reordering from a real change.
2b. **Replace the reduction with the single `mse_loss`** over the stacked prediction. Its own
   change, its own evidence: the diff is definitionally a reordering, and step 1 measured what that
   reordering costs (value bit-identical, `1 - cos = 6.1e-15`, one float32 ulp on the largest
   gradient entries). This is where R1 stops being bit-identical, deliberately and once.
2c. **Add the remaining three seams**: `make_training_data` and `cal_validation_error` as overrides
   on `SSE_Interconnect` that call `super()` and then delegate, and the `validation_probes` tuple.
   All default to `simulator = None` / `validation_probes = ()`, i.e. exactly today's behaviour.
   deepSI itself is not touched. Bit-identical against 2b.
2d. **Convert `_install_nf_val_probe` to a `validation_probes` entry** and delete the
   `__reduce__` hack. Assert the probe histories are identical on a two-epoch run. Not optional and
   not deferrable: once `cal_validation_error` is a seam, a patch on the same attribute can still
   silently override it, which is the ordering hazard 3.6b exists to remove.
3. **Add `output_only`** to `Interconnect`, evaluating the output signal's dependency cone, with
   the cone resolved once at `__init__` (3.4). Assert it equals the full forward's `y` to machine
   precision on random states, and that `check_no_feedthrough` still passes through the real
   interconnect.
4. **Route `SSE_Interconnect_MultipleShooting.loss` through `self.simulate()`** so one rollout
   implementation exists. Assert parity against step 1 with `n_seg = 1` and with `n_seg > 1`
   against the pre-change defect diagnostics. **Then raise on `n_seg > 1` with a simulator
   present.** Routing multiple shooting through the seam makes that combination reachable for the
   first time, and nothing in this document says whether the controller state resets at a segment
   boundary. A raise is not a second rollout path; a silently wrong combination is worse than an
   unsupported one. Decide and implement the semantics only when something needs it.
5. **Create `closed_loop.py`** with `DiscreteController`, `ControllerBank`, `closed_loop_rollout`,
   `ClosedLoopSimulator` (both `__call__` and `validation_error`). Port the units gate and the
   record-index assert with it. Three things are written differently from the current
   `cl_controller.py`, all specified in 3.3 and all justified by the step-0 profile: the
   normalisation is folded into `B, C, D` at construction and the units gate unfolds to check it;
   the matrices are stacked so a step is `bmm` plus `baddbmm` rather than four einsums; and
   `closed_loop_rollout` returns `(y_pred, x_final, xc_final)` while `__call__` returns `y_pred`
   alone. `augment_training_data` is the one invented API in this file (9.2) and is the reason this
   step, not step 2, is where the data path can still surprise us.
6. **Set `fit_sys.simulator`** in the gantry pipeline when the loop is enabled, following the
   `orth_penalty` precedent: attached after construction, absent by default, exact no-op when
   absent. Assert against step 1 again with it absent.
7. **Delete** `cl_fitsys.py`, `loss_variants.py`'s B and C mixins, `ClosedLoopValidator` and both
   `attach()` functions, plus `cl_controller.py`'s `ControllerBank`, `rollout`,
   `open_loop_rollout` and `check_units`, which are now the framework's. `cl_controller.py` keeps
   `RECORD_Y_OP`, `y_op_for` and the new `build_controller_bank`, i.e. exactly the boundary of 3.9;
   `loss_variants.py` keeps `controller_ss` and `sensitivity_ss`, which the MATLAB gates import.
   Repoint what is still live and STATE what is not, rather than leaving it to be discovered:

   | still live, repointed | left with dangling imports |
   |-|-|
   | `cl_validation.py` (`free_run` on the framework rollout), `cl_step0_profile.py`, `cl_step1_reference.py`, `cl_pipeline.py`, the four gate A scripts | `cl_diag_step3`, `cl_direct_vs_residual`, `cl_gate_loss`, `cl_gate_replay`, `cl_gate_validation`, `cl_lr_probe`, `cl_plot_step6`, `cl_precision_gradient`, `cl_precision_validation`, `cl_sanity`, `cl_smoke`, `cl_step5_reset_cost`, `cl_step6_run` |

   Those thirteen are historical: every result they produced is already recorded (section 4 of the
   handoff, `RESULT*.md`, `server-results/step6_result_76573.json`, `references/`), and the handoff
   explicitly says not to re-run the equivalence and precision experiments. Repointing scripts that
   will never run again is churn with real risk, since a break would not be noticed. This is the
   same call section 8 already makes for the checkpoints, made once more and written down.
   `cl_step6_run.py` is superseded by `cl_train.py` and kept as the record of what run 76573 did.

   **One grep in 5.1 does not pass repo-wide, and the criterion is the thing to fix, not the code.**
   `grep -rn "cal_validation_error = " scripts/` still returns hits in
   `scripts/gantry/augmentation-error/{diag_nf_curriculum,diag_theta_lr_sweep,diag_xy_routing_blowup}.py`,
   which patch it for their own diagnostics and predate this work. Over the path this migration owns
   (`model_augmentation/` and `scripts/gantry/gantry_dynamic/`) the count is 0. Those three are now
   candidates for the same `validation_probes` treatment, which is a separate piece of work.
8. **Retrain.** The existing checkpoints will not load; see section 8. Runner: `cl_train.py`, which
   replaces `cl_step6_run.py` at the same configuration. `CL_SMOKE=1` runs the whole path end to
   end on truncated data in about 90 s and has passed: the simulator pickles, training reduces the
   loss, validation goes through the seam, and the simulator SURVIVES `fit()`'s closing
   `checkpoint_load_system('_best')`, which is the trap that forced the old implementation to
   rebuild its entire eval stack afterwards. The real 12-epoch run is a cluster job: extrapolating
   the step-0 profile it is on the order of a day of wall clock on this machine.

**Status: steps 0 to 7 are DONE and gated.** Every step's gate is recorded above; the four gate A
scripts, `cl_test_seams.py`, `cl_test_output_only.py`, `cl_test_closed_loop.py` and
`cl_step1_reference.py --check` all pass. Step 8 is the only one outstanding and it needs hardware
this machine does not have.

No fit-system class is created, subclassed or swapped anywhere in this list. That is the test of
whether the design held: if a step needs a new class in the loss chain, the seam is in the wrong
place.

### 5.1 Acceptance criteria: what "good" means, and against what

There are three references, and they are not equal. Ordered by strength:

**A. MATLAB's `Cfb`. External ground truth, the primary criterion.** This is the only reference that
proves the controller *is* the one that generated the records. The others prove only that nothing
changed, so if the current implementation were wrong they would faithfully preserve the error. The
gates already exist and must pass after the move:

| gate | what it pins | current value |
|-|-|-|
| `test_controller_exact.py` L1 | our `num`/`den`, poles, zeros, `kappa_j`, `sys_jj(i w_b)` against MATLAB `tfdata`. No simulation, so a pass here IS the formula being right | 9.6e-12 on coefficients |
| L2 | MATLAB's exported `(A,B,C,D)` run in Python, calibrating the arithmetic floor | |
| L3 | our `num`/`den` on MATLAB's `e_test` against MATLAB's `u_test`; the L2-L3 gap is `tf2ss` conditioning, not the formula | |
| `verify_controller.py` | `Cfb` driven by the stored `r_sim - y` against the stored `u_fb` | 4.5e-09 relative |
| `verify_cfb_against_records.py` | the same on every record, plus the additivity identity `u_total - (u_fb + f_sim)` | |
| `p1_equivalence.py` | the whole loop driven from `r_sim`: sign convention, sample alignment, controller init, with a ramp-fraction criterion that catches integrator faults specifically | |

**A gap to close while doing this. CLOSED in migration step 1, and it needed a third reference.**
Every gate above runs at the RECORD rate, 20 kHz, because that is what MATLAB produced. The training
path steps `Cfb` at 4 kHz (`cfg.ts_new`), and nothing checked the 4 kHz controller against MATLAB at
all: it was verified at 20 kHz and then re-discretised in Python. `export_controller.m` now exports
`c2d(kappa*Cnorm, 1/4000, 'tustin')` and its `ss`, and `test_controller_exact.py` gained L5.

The two-way comparison at 4 kHz gives **2.7e-10**, worse than the 9.6e-12 the same comparison gives
at 20 kHz and above the 1e-11 this table demanded. A two-way comparison cannot say which side moved,
so L5 measures both against the bilinear transform computed in **exact rational arithmetic**
(`fractions.Fraction`, no rounding at all, applied to the same double-valued continuous
coefficients). Measured:

| | 20 kHz | 4 kHz |
|-|-|-|
| MATLAB `c2d` vs exact | 5.7e-14 | **6.9e-14** |
| scipy `cont2discrete` vs exact | 5.0e-12 | **2.05e-10** |
| the two-way gap being explained | 4.99e-12 | 2.051e-10 |

`|scipy - MATLAB|` tracks `|scipy - exact|` to three digits, so the entire gap is scipy's numerator
arithmetic, and the formula is right at both rates. The mechanism is cancellation: tustin sends
`z = inf` to the finite `s = 2/ts`, and at 20 kHz `2/ts = 40000` rad/s dominates every other term in
the expansion while at 4 kHz it is 8000, comparable to `10w = 6283`, so the alternating-sign sums
cancel far more. 2.05e-10 relative is three decades below float32 eps (1.19e-07), so it cannot reach
the training path.

Consequence for this table: **"L1 coefficients <= 1e-11 at both rates" is achievable against the
formula, not against scipy.** L5 therefore asserts the formula (MATLAB vs exact, 1e-11) and L5py
asserts the implementation the loop actually uses (scipy vs exact) at 1e-09, one decade above its
measured floor rather than a number chosen to make it pass. Replacing scipy's `cont2discrete` with
the exact construction would collapse L5py to 7e-14 as well; it is not done here because it changes
the controller the reference sets were recorded against, and it buys nothing measurable.

**B. R2, the closed loop as it exists today. Regression, captured BEFORE any edit. RECORDED**,
`references/step1_reference.{json,npz}`. On a fixed batch of 32 windows spanning four records
(T1, T5, T10, T13, i.e. three distinct controller rows, so a `ctrl_ix`/gather regression is
observable at all): `cl_fitsys.ClosedLoopLoss.loss` value and full gradient, `cl_controller.rollout`'s
`y_pred`, `loss_open_loop`'s value for the G11 no-op contract, the units gate, the per-record window
counts, and `ClosedLoopValidator`'s selection scalar over the four validation records. This catches
a change in the plumbing that A cannot see, because A tests the controller in isolation and says
nothing about whether the right controller reaches the right window.

**Recorded on the UNTRAINED build, not on `Go1qTA_best`.** Section 8 accepts that those checkpoints
stop loading after the move, so a reference that can only be evaluated on them cannot be
re-evaluated after the thing it guards. `build_pipeline` seeds before the data load and again before
`build_model`, so the untrained parameters are a deterministic function of the config; the sha256 of
every parameter tensor is stored and must reproduce before any loss comparison is meaningful.
Cross-check that it is the right object anyway: the untrained selection scalar came out at
**2.186602663e-06 m** against the step-6 server run's **2.186550806e-06 m**, a difference of
5.2e-11 m, below the 7.6e-11 m float32-vs-float64 shift already measured for that same quantity.

**Two arms, and the second is not optional.** At initialisation the ANN's last layer is zero, so its
output is exactly zero and the gradient into every layer behind it is zero too: only 852 of 3616
gradient entries are nonzero. A reference recorded there is partly a vector of structural zeros and
would not notice a change in those paths. Arm 2 adds a fixed seeded perturbation (N(0, 1e-2), seed
12345, 600 ANN parameters), which lifts it to 1600 of 3616. A seam that is inert on arm 1 and not on
arm 2 is still broken. The loop is genuinely active in both: with the ANN at zero the augmented model
is the baseline, which does not reproduce the data, so the residual is nonzero. Evidence that the
batch is sensitive at all: `1 - cos(g_closed, g_open) = 1.053` at identical parameters.

**Determinism.** Bit-identity is meaningless without a fixed reduction order, and torch does not
guarantee one across thread counts. Threads are pinned to 1 and recorded in the manifest; a later
comparison at a different thread count is comparing two arithmetic orders and calling the difference
a regression.

**C. R1, simulator absent. Proves the seams are inert. RECORDED**, same file. Loss value, full
gradient, encoder state and a full open-loop rollout of the production configuration on the same
fixed batch. Bit-identical required through step 2a; from 2b onward the reduction change costs one
ulp on the gradient, by design and once (3.1).

**D. The SEGMENTED loss, `n_seg > 1` with both defect terms live. RECORDED**, same file. Step 4
routes `SSE_Interconnect_MultipleShooting`'s inner loop through `self.simulate()`, and `n_seg > 1`
WITHOUT a simulator has to keep working because that is what the defect diagnostics use. Recorded
at `n_seg = 4`, `nf_seg = 100`, `defect_weight = defect_acc_weight = 1` (both nonzero: the guard in
that method covers both, so with either at zero the segmented path is never entered and D would
silently record the unsegmented loss), on the same fixed batch, both arms: loss, `last_mse`,
`last_defect_rms`, `last_defect_acc` and the gradient. Two arms of its own, D1 at the production
`linear_map` encoder and D2 at `encoder_init = 'default'` with its own build fingerprint, because a
defect that showed up while recording this turned out to be encoder-independent and both needed
pinning. See section 8.

**E. The nf-probe histories. RECORDED**, same file. Step 2d converts `_install_nf_val_probe` from a
monkey patch into a `validation_probes` entry and asserts the histories are identical, which needs
a pre-change history. The probe is driven directly with a stub selector over truncated records,
which is the right isolation: step 2d changes how the probe is INVOKED, not what it records, and
the real selector is covered by R2 and by step 2c's short-run gate. Records `Loss_train_nf`,
`Loss_val_nf`, `Probe_combo_err`, `Probe_orth_frac`, `Probe_V_orth`, `Probe_param_loss`.

**Why D and E are listed here at all.** They were not in the original version of this section, while
steps 4 and 2d both said "assert against the pre-change X" and no step recorded X, and the preamble
to section 5 claimed step 1 was "the safety net for all the others". That was an internal
contradiction rather than a missing run, and it was caught two steps late. Any future step that
compares against a baseline states that baseline HERE, or it does not have one.

What each cannot catch, so none is dropped: A cannot see plumbing (wrong controller on the right
window), B cannot see a wrong controller that was already wrong, C cannot see the closed loop at
all, and D and E see only the paths they name.

**The gate is a mode, not a ritual.** `cl_step1_reference.py --check` rebuilds, recomputes
everything above and prints a per-key PASS/FAIL table against the recording. It refuses to compare
at all if the build fingerprint or the thread count differ, because then the numbers below it would
be measuring the build rather than the change. Each key carries a comparison CLASS (`exact`,
`reorder`, `loose`, `sel`) rather than one global tolerance, so a step that legally reorders one
quantity cannot silently widen the gate on every other one; a step that needs a key relaxed passes
`CL_RELAX="key=class"` on the command line, where it is visible in the log, instead of editing the
default.

**Thresholds, derived from measurement rather than invented.** Some steps change the order of
operations (a single `mse_loss` over a stacked prediction instead of a mean of per-step values;
`output_only` instead of the probe-identified affine map), so bit-identical is the wrong
requirement there. The right yardstick is the size of a difference already known to be pure
arithmetic, which this folder has measured:

| quantity | tolerance | where the number comes from |
|-|-|-|
| loss, relative | `<= 1e-3` | the float32-vs-float64 closed-loop loss differs by 3.1e-4, so anything below this is indistinguishable from reordering |
| gradient, `1 - cos(g_new, g_old)` | `<= 1e-5` | float32-vs-float64 gives 2.2e-6; batch-to-batch scatter is 1.2, so this is four orders inside what SGD tolerates |
| validation selection scalar | `<= 1e-10` m | the float32-vs-float64 shift is 7.6e-11 m and the checkpoint gap is 1.39e-09 m |
| `output_only` vs full forward `y` | machine precision | no reordering, it is the same graph |
| seam no-op (R1), step 2a | **bit-identical** | nothing about the arithmetic changes |
| reduction change (R1), step 2b only | `1 - cos <= 1e-13`, one ulp elementwise | measured on the same batch: `1 - cos = 6.1e-15`, max elementwise 1.207e-07 relative = one float32 ulp, value bit-identical |
| MATLAB coefficients vs exact arithmetic (A), both rates | `<= 1e-11` | 5.7e-14 at 20 kHz, 6.9e-14 at 4 kHz |
| scipy `cont2discrete` vs exact arithmetic (A), 4 kHz | `<= 1e-9` | 2.05e-10 measured, one decade of margin; three decades below float32 eps |

Where bit-identical is expected it is required; the table above applies only to the steps that
genuinely reorder operations, and each such step must say which.

**Non-numerical criteria, greppable, part of done:**

```
grep -rn "__class__ = type\|def attach(" scripts/ model_augmentation/     -> no hits
grep -rn "cal_validation_error = "        scripts/ model_augmentation/     -> no hits
grep -c  "def closed_loop_rollout"        model_augmentation/              -> exactly 1
```

plus the installed deepSI package unmodified. `zero_the_ann` was the one patch this document
planned to keep; it is gone too (3.6d), so `grep -rn "\.forward = " model_augmentation/ scripts/gantry/closed-loop-controller/`
returns only the comment in `cl_plant.py` explaining what used to be there.

**End-to-end confirmation, last.** Retrain and check the result reproduces the reference run: 36.3 %
improvement over the baseline, val 2.187e-06 to 1.393e-06 m in 12 epochs. This is stochastic, so it
is a sanity band and not a parity assert: a result within a few percent of that improvement
confirms the move; a materially worse one means something in the loop changed that the batch-level
checks missed.

---

## 6. Attribution: where Kessels is cited in the code

The closed-loop training formulation is Kessels' method, not ours. It must be attributed **in the
code**, at the lines that implement it, not only in the write-up. This follows the existing
`# THEORY: <source>` convention: source, variable and context must all match, or the label is not
earned.

Reference, verified against the PDF in this repository:

> B.M. Kessels, PhD thesis, TU/e, 2025, Chapter 5 "Extension and augmentation-based model
> structure updating". `literature/augmentation/kessels2025_ai-control.pdf`.
> **Page offset: PDF page = thesis page + 26.**

`closed_loop.py` module header carries the reference block once. Then, inline:

| Code | Label |
|-|-|
| the truncated-window loss, if it is ever written out here rather than inherited | `# THEORY: Kessels (2025) Eq. (5.12), p156 -- truncated-window loss V_T, C = n_TW*T := (N-T+1-n_o)*T` |
| `y = h(x)` before the controller, controller error formed against the MODEL output | `# THEORY: Kessels (2025) Eq. (5.13d), p157 -- e_hat = r_bar - y_hat, controller driven by the model output` |
| the controller stepped as its own state equation, outside the model state vector | `# THEORY: Kessels (2025) Eq. (5.13d), p157 -- FB controller as a separate constraint, not part of the model state` |
| `ControllerBank.__init__`, where `ystd` and `stdu` are folded into `B, C, D` (3.3). The scaling is applied ONCE there rather than at every timestep, so this is where the label belongs; the rollout line is then just `u = u_data + u_fb` in one coordinate system | `# THEORY: Kessels (2025) Eq. (5.13c), p157 -- u_hat = S_u(u_FB + u_FF), scaling at the control interface` |
| the encoder call, which returns model states only | `# THEORY: Kessels (2025) Eq. (5.13a), p156 -- encoder returns FP and extension states; FB states are NOT encoder outputs` |
| `xc = 0` at each window start | see below, this one is **not** a Kessels citation |

**The one place where we deviate, and it must say so.** Kessels' Remark 5.4, p157, reconstructs the
machine's feedback state from the measured output, the known reference and the known controller
(assuming the FB state is zero at k=1) and, in his words, uses it "to initialize the feedback
states for each TW". His controller filters `r_bar - y_hat`, both of which exist before a window
opens, so his `xc(tau)` is a real quantity with a data-derived value and zeroing it would be an
approximation.

Our residual form filters `y_data - y_model`, which does not exist before the window opens, so
there is nothing to reconstruct. The comment at that line must therefore read as a **contrast**,
not a citation:

```python
# HEURISTIC: xc = 0 at every window start. NOT Kessels' Remark 5.4 (p157), which
# reconstructs xc from (y_bar, r_bar, controller) for the lumped-r form where the
# controller filters y_hat against the reference. In the residual form the controller
# filters (y_data - y_model), which does not exist before the window opens, so this is
# the definition of an initial condition rather than an estimate of an unknown. It is
# also the unique value for which the correction vanishes when the model is exact.
# Cost: lost integral memory against the validation free run, measured, see
# cl_step5_reset_cost.py.
```

Claiming Remark 5.4 here would be a misattribution: it would credit him with an assumption he does
not make and would hide the fact that the two forms differ structurally.

---

## 7. Gates that must survive the move

These already exist and are the reason the current implementation is trustworthy. They are not
optional and they port with the code.

- **Exact no-op.** With the controller absent, the loss is bit-identical to the production one.
  This is the same contract D-076 and D7.1 established for `ParamLoss` and `OrthLoss`.
- **No feedthrough.** `y(x, u=0) == y(x, u=1e3 randn)` through the real interconnect, not by
  reading `Dd` from a matrix. The wiring passes `u` into the output block, so this permits
  feedthrough and must be measured.
- **Units.** A perturbed-state gate. The zero-ANN replay gate *cannot* catch a scale error on
  `Cfb`, because with the ANN off the residual is identically zero and any scale factor is
  multiplied by zero.
- **Record index alignment.** The derived per-record window count asserted against the real
  `to_hist_future_data` output. This has already caught one off-by-one that would have attached the
  wrong controller to most of the training set.
- **Controller against the records.** The closed-form `Cfb` reproduces MATLAB's stored `u_fb`.

---

## 8. Known costs, accepted going in

- **Existing `.pth` checkpoints will not load, and no shim will be written.** They pickle
  `FitSys_ClosedLoop`, a class created at runtime by `attach()`, which stops existing. A
  compatibility shim would mean keeping the runtime-class machinery alive purely to read old
  files, i.e. coding around the structure being removed. DECIDED: no shim, retrain after the move.
  The affected artefacts are `server-results/deep-SI-checkpoints/FitSys_ClosedLoop_Go1qTA_{best,
  last}.pth` from run 76573; the numbers already extracted from them (val 2.187e-06 -> 1.393e-06,
  the reset-cost and precision measurements) survive in the result JSON and in this folder's
  scripts.
- **`interconnect.py` gets edited.** DECIDED: permitted. Three seams plus `output_only`, each a
  delegation with an exact no-op default, each marked `# CHANGED:` and each covered by the parity
  assert of step 1.
- **`rec_ix` still depends on a deepSI dataloader invariant.** Not removable without owning the
  data path. It shrinks from a derivation spread across files to one asserted assumption.
- **One pre-existing defect was fixed to make step 4 gateable, and it is worth stating plainly.**
  `SSE_Interconnect_MultipleShooting.loss` could not run at `n_seg > 1` with ANY encoder in this
  codebase. It builds each interior node with `self.encoder(ufuture[:, s-nb : s+nb_right], ...)`,
  and a time-axis slice of a contiguous `(batch, nf, nu)` tensor keeps dim-0 stride `nf*nu`, i.e. it
  is not contiguous, while both encoders reshape their input with `.view`:
  `RuntimeError` at `pre_encoder.py:450` for `encoder_init='linear_map'` (the production encoder)
  and at `interconnect.py:384` for `'default'`. deepSI's `to_hist_future_data` hands the encoder
  contiguous windows, so the `n_seg = 1` production path never touches it and nothing noticed.
  Fixed at the caller with `.contiguous()`, marked `# CHANGED (contiguity)`, which cannot move R1
  because it is not on the `n_seg = 1` path. Two consequences worth carrying: rule 2's statement
  that `multiple_shooting.py` is "kept only for a set of defect diagnostics" is weaker than it
  reads, since those diagnostics cannot have been produced through this method; and step 4's gate
  now compares against a numeric D recorded on the fixed code rather than against an exception.

---

## 9. Settled, and what remains

**Settled, with the evidence:**

| question | answer |
|-|-|
| may `interconnect.py` be edited | yes; three seams plus `output_only`, each `# CHANGED:` with a no-op default |
| validation path | `cal_validation_error` seam delegating to the simulator, section 3.6. No monkey patch |
| is bypassing the seam in multiple shooting acceptable | no; route it through `self.simulate()`, section 3.7 |
| how invasive is the output map | not invasive; the output's dependency cone is one block, section 3.4 |
| one bank or two | ONE, over all records, indexed globally. Two exist today only because `rec_ix` is a per-list position; `Cfb` is per trajectory and train/val is not an axis |
| `rec_ix` or `ctrl_ix` | `ctrl_ix`, resolved at data-build time. Removes one indirection and the second bank; does NOT make the count derivation safer, see 3.5 |
| checkpoint compatibility | no shim, retrain, section 8 |
| which rate the loop steps at | 4 kHz (`cfg.ts_new`). `diag(D_c) = [8.055, 8.253, 4.275]e6` N/m, computed through `controller_ss` at both rates |
| the `xc = 0` justification | `cl_controller.py`'s version is correct; Kessels Remark 5.4 does not apply, section 6 |
| how validation identifies a record | ordered `(name, sys_data)` registered with the simulator, then count / position / content asserted at first use, section 3.6 |
| training precision | float32. `cos(g32, g64) = 0.9999978` against a batch-to-batch cosine of -0.218, i.e. a disagreement 1.8e-6 of what SGD already tolerates |
| what regime the runtime is in | dispatch-bound, not FLOP-bound: a `(32,8)@(8,8)` matmul runs at 60 MFLOP/s. Cut operation COUNT, never FLOPs (3.8) |
| is the block-diagonal `Cfb` storage worth it | NO. Cuts FLOPs 3x and the op count by zero, and it conflicts with the stacking that does cut the op count. Reversed on the step-0 measurement |
| how the controller is stored | in normalised coordinates, scalings folded into `B, C, D` at construction; the units gate unfolds to check (3.3) |
| what `simulate()` returns | `(y_pred, x_final)`. REVISED during step 4: multiple shooting forms each defect from the segment's final state, so returning `y_pred` alone would force it to keep its own rollout, i.e. the second implementation this seam exists to prevent. `closed_loop_rollout` still returns all three and the simulator drops `xc` |
| where `output_only` lives | on `Interconnect`, so it is `fit_sys.hfn.output_only`. Section 3.3's sketch said `fit_sys.output_only`, which is wrong and raises `AttributeError`; 3.4 is the correct statement |
| the closed-loop perturbed test point | `sigma = 1e-4`, not the `1e-2` used on the open-loop arms. At 1e-2 the closed-loop rollout is chaotic: the SAME implementation gives loss 4.81e-02 in float32 and 2.29e-03 in float64, and two implementations that agree to `1 - cos = 8.9e-16` at 1e-4 give `1 - cos = 1.87` at 1e-2 IN FLOAT64. A quantity two precisions of one code disagree on by 20x is not a regression net |
| is the 4 kHz controller verified against MATLAB | yes, and it needed an exact-arithmetic third reference: MATLAB is 6.9e-14 from exact, scipy's `cont2discrete` 2.05e-10 (5.1) |

**Remaining:**

1. ~~**Whether the `simulate()` extraction is bit-identical.**~~ MEASURED, migration step 1, and the
   answer is "the value yes, the gradient no". Mean of per-timestep `mse_loss` versus one `mse_loss`
   over the stacked prediction: value difference exactly 0.000e+00 on both arms, gradient
   `1 - cos = 6.1e-15` with the largest elementwise difference at one float32 ulp. Consequence:
   the extraction and the reduction became two steps (5.2a and 5.2b) so that bit-identity guards
   the seam and a measured reordering bound guards the reduction. Caveat carried forward: the two
   quantities compared differ in two ways, not one (reduction order, and `loss_open_loop` rolling
   through `out_fn` rather than `hfn`'s own `y`), so which of the two costs the ulp is unresolved.
   Step 2a's seam keeps `self.hfn`, so if the reduction is also preserved it should be exactly zero;
   if step 2a is not bit-identical, the cause is the rollout path and not the reduction, and that is
   a real finding rather than a tolerance to widen.
2. **The `augment_training_data` signature.** An invented API. `make_training_data` returns the
   four arrays from `to_hist_future_data` and `fit()` calls `self.loss(*train_batch)`, so the
   mechanism is proven by the existing fifth array; only the method shape is unchecked. Low risk.
3. ~~**`concurrent_val`.**~~ SETTLED, and it was the artefact, not the fundamental. `cl_sanity.py`
   recorded that it MUST be False for the closed-loop path: the concurrent branch ships the model
   to a subprocess by pickle, a monkeypatched `cal_validation_error` does not survive that, and
   the child validated with deepSI's default, i.e. the OPEN-loop measure, so selection silently
   optimised one objective and chose on another. The seam removes the mechanism, because
   `simulator` is a declared attribute holding an importable class. MEASURED with
   `concurrent_val=True` on full-length validation records: the child returned
   **2.1866011034e-06 m** against the recorded untrained closed-loop scalar 2.1866026634e-06 m,
   rel 7.13e-07, and selection picked a trained checkpoint through `remote_recv`.

   Two caveats, and they are why the retrain should still start with it OFF. **The verification ran
   on Windows, whose multiprocessing start method is spawn; the cluster is Linux, which forks.**
   Those are different code paths: spawn re-imports the module, which is how the missing `__main__`
   guard in `cl_train.py` surfaced (fixed), while fork copies memory and has its own hazards. And
   **child-crash behaviour is untested**: whether `fit()` hangs on `remote_recv`, records a nan, or
   propagates is unknown, which matters more than the saving on a day-long job.

   The saving is smaller than first claimed. It is bounded by the overlap, so at this machine's
   numbers, roughly 18 min of training and 8.5 min of validation per epoch, serial 26.5 min against
   concurrent 18 min: about a THIRD, not a half. It would approach a half only with more frequent
   validation. Cost: the child holds its own copy of the model and data, so peak memory roughly
   doubles.
3b. **Multiple shooting combined with a closed loop.** Not a question today, because production runs
   `n_seg = 1`, and not a question before step 4, because `SSE_Interconnect_MultipleShooting.loss`
   bypasses the seam entirely. Step 4 makes the combination REACHABLE, and this document never says
   whether the controller state `xc` resets at a segment boundary the way `x` is re-anchored to the
   encoder. Both readings are defensible and they are different objectives. Step 4 therefore raises
   on the combination rather than picking one silently; decide the semantics when something needs
   it, and note that the `xc = 0` argument in section 6 is a statement about a WINDOW start, so it
   does not settle a SEGMENT start for free.
4. ~~**Validation precision.**~~ SETTLED. `cl_precision_validation.py`, both checkpoints, four val
   records, full-record closed-loop free runs: the ranking does not flip and the precision shift on
   the selection scalar is 7.6e-11 m against a checkpoint gap of 1.39e-09 m, i.e. 5.5 %. No dtype
   policy is needed: training, validation and selection all run in float32. Float64 stays for the
   gates only, where the thresholds are at machine precision. Caveat to state once in the write-up:
   the margin is 5.5 %, so separating checkpoints closer than about 1.4e-09 m would require
   redoing this measurement.
5. ~~**The `D_c` correction.**~~ DONE. `cl_controller.py` now quotes
   `[8.055, 8.253, 4.275]e6` N/m at `ts = 1/4000` s and explains the rate dependence
   (`Dc_jj = kappa_j * Cnorm(2/ts)`). The D-140 entries in `docs/decisions.md` and the problem log
   keep the 20 kHz value, which is correct in their context. Nothing outstanding: the figure does not
   appear in either file in `documentation/`, only in prose outside the repo.
