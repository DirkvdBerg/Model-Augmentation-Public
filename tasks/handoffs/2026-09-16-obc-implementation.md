# Handoff: implement one-step orthogonal-by-construction augmentation for the gantry

Run this session on **Claude Fable 5.1**, effort **`high`** (not `xhigh` or `max`: this task
produces long code files, which is the shape that makes the higher settings draft the output in
reasoning and then write it again). **Unattended**: nobody can answer a question while it runs.
Every decision that would otherwise need asking is pinned in section 7.

---

## 0. Three standing rules that would otherwise stall this session

`CLAUDE.md` is auto-loaded and is not repeated here, with three exceptions that would block an
unattended implementation run on their plain reading:

1. **"Answer before code" is satisfied by this document.** The direction is given. The
   "Commit after direction is given" rule applies: execute, do not re-request approval.
2. **"No new files unless asked" is lifted for the files named in section 10** and only those.
3. **Do not enter plan mode.** This document is the plan. Plan mode needs an approval you cannot
   receive.

Everything else in `CLAUDE.md` applies unchanged. Two rules will bite immediately and are worth
naming: the `@added` / `__project_origin__` / `# CHANGED:` marker rule for anything added to
`model_augmentation/`, and the `# THEORY:` / `# HEURISTIC:` labelling rule, which applies to the
FD4 stencil coefficients, the SVD rank tolerance, and both cost thresholds in section 7F.

---

## 1. Why this exists

The thesis contribution is preserving physical-parameter interpretability while a learned dynamic
component is co-estimated. Gyorok's orthogonal-by-construction (OBC) parametrization is the
mechanism: the learning component is reparametrized so it cannot represent anything the baseline's
own parameters could have represented, which makes the physical parameter estimate unique instead
of one of a continuum. Gyorok published it for input-output models that are linear in their
parameters. The gantry is a nonlinear LPV state-space model discretized by RK4 and trained in
closed loop, so the transfer is not automatic, and the mathematical form it takes here is settled
in

    scripts/gantry/orthogonal-by-construction/documentation/obc-gantry-one-step-implementation-plan.tex

Read that document. It is the specification; this handoff is the implementation contract for it.

**What this session delivers:** a tested projection path and no-training diagnostics. Not a
training run. The next session decides whether to train, using the numbers this one measures.

---

## 2. The invariant

    +-------------------------------------------------------------------------+
    |  theta_aux is computed ONCE per loss/objective evaluation, from the      |
    |  current ANN over the fixed reference set, and reused unchanged at       |
    |  every timestep of that rollout. Gradients flow through it to the ANN    |
    |  weights. It is never recomputed inside the per-timestep block call.     |
    +-------------------------------------------------------------------------+

Everything else in this document is negotiable under measurement. This is not. Getting it wrong in
either direction produces a method that looks like OBC and is not:

- Recomputed per timestep: same numbers (the ANN weights do not change within one objective), but
  the reference set is evaluated `nf` times and the autograd graph holds `nf` copies of that
  subgraph. A runtime and memory defect, not a numerical one.
- Frozen across optimizer updates: `J^T (F - J a) = 0` stops holding after the first update, the
  correction degenerates into a fixed subtracted field that does not constrain the model class at
  all, and none of Gyorok's results apply. This is what the withdrawn epoch-wise variant did.

---

## 3. What you need from Gyorok, without reading the papers

The comprehension of the papers is a separate session. Four semantics are all this implementation
needs, and all four are verifiable in the local reference repository:

1. The regressor and its factorization are built **outside** the objective, once, because they do
   not depend on the learning component (`orthogonal-IO-augm-main/orthogonalx_augm/src.py:248`).
2. The auxiliary coefficient is recomputed **inside** every objective evaluation from the current
   learning-component output (`src.py:255`).
3. Gradients pass through that coefficient: `jax.value_and_grad` over the objective, no
   `stop_gradient` anywhere (`src.py:261-262`).
4. After training the coefficient is frozen and the regressor is re-evaluated at each new
   prediction point (`src.py:364-368`; Gyorok 2026 Eq. 13).

Paper equations, if you want them: Eq. (7) is the full-column-rank assumption, Eq. (8) is the
coefficient, Eq. (10) the projected component, Eq. (11) the stacked orthogonality statement, Eq.
(13) the prediction rule. The local PDF is
`literature/Orthogonality/gyorok2026_orthogonal-by-construction-augmentation_IFAC-JSC_arXiv2511.01321.pdf`.
The local-sensitivity substitution used here (a Jacobian in place of an exact regressor) is
Gyorok et al., L4DC 2025, Eqs. (15) to (19), at
`literature/Orthogonality/Hoekstra - Orthogonal projection-based regularization for efficient model.pdf`.

Do not open either PDF unless a specific equation is in dispute. The four semantics above are the
contract.

---

## 4. The construction, with dimensions

Model dimensions in the active configuration: `n_b = 6` physical states, `n_a = 8` additional
learned states, augmented state 14, `n_u = 3`, `n_y = 3`, `n_theta = 10` identifiable parameter
combinations, `n_r = 6` routed physical rows (`ann_route_ix = tuple(range(14))`, so all of them).

**Reference set.** Every valid sample of the 14 training records (`gantry_dynamic/data.py:21-27`),
roughly `N = 6.7e5` samples:

    q_k     = P^-T y_k                     exact static inversion; P is constant and invertible
                                           (gantry_ss.py:120-136, Cd = [P^T | 0])
    qdot_k  = fourth-order central difference of q
    x_a     = 0
    u_k     = u_k^data                     the recorded plant input, not a controller output

The position inversion is exact. The velocity is a data-derived finite-difference **estimate**, not
a state measurement; say so wherever it is described.

**Basis.** At expansion point `vbar` (the current reduced free coordinate, detached):

    J_{b,k} = d f_base^d / d theta |_{vbar}  evaluated at (x_b,k^ref, u_k^ref)   in R^{6 x 10}
    J       = sample-major stack, rows restricted to r_phys                       in R^{N*n_r x 10}
            ~ 4.0e6 x 10

`f_base^d` is the complete discrete RK4 transition of `Reduced_Gantry_State_Block`, differentiated
with respect to its **free coordinates** (`free_params`), not the physical combinations. The free
coordinates are log and relative-linear, which is what makes the ten columns comparably scaled; the
block's own docstring records a factor 52 improvement in condition number over physical
coordinates.

The block already works in normalized state and input coordinates and returns the normalized next
state, and the ANN writes into those same normalized rows. **Differentiating the block gives the
normalized Jacobian directly. Do not apply any further `1/std_x` scaling to it or to the ANN
output.** `orth_penalty.py:144-157` is the existing precedent for this and applies no such factor.

**Projection.** Economy SVD of `J`, rank tolerance declared, pseudo-inverse from the SVD directly.
Never form or invert the Gram matrix.

    K_J   = J^+                                          in R^{10 x N*n_r}
    a     = K_J @ F_aug                                  in R^{10},  once per objective
    Ftil  = F_aug - J @ a                                stacked orthogonality: J^T Ftil = 0

where `F_aug` is the current ANN's physical-row output over the reference set, stacked in the same
sample-major order as `J`.

**Correction inside the rollout.** At rollout point `(xhat_k, u_k)`:

    ftil_aug^b(xhat_k, u_k) = f_aug^b(xhat_k, u_k) - J_b(vbar, x_b,k, u_k) @ a

`J_b @ a` is a directional derivative: one forward-mode JVP of the transition in direction `a`, not
a formed Jacobian. The additional-state rows `f_aug^a` are left untouched.

**What this does and does not establish.** Orthogonality holds on the reference stack, in the
normalized-state metric restricted to the routed physical rows, against the tangent space of the
baseline family at `vbar`. It is exact for the current ANN at each objective evaluation and local
in the parameters. Nothing follows about pointwise behaviour, about nonzero `x_a`, about multistep
or closed-loop behaviour, or about any of Gyorok's recovery, consistency, covariance or stability
results.

---

## 5. Module boundary

**Generic, in `model_augmentation/fit_systems/obc.py`** (new file, `__project_origin__ = "added"`):
SVD-based basis object with rank reporting, the differentiable coefficient solve, the
objective-lifecycle plumbing, and the orthogonality and rank diagnostics.

This half must not know what a gantry is. The seam is a **callable contract**: the caller supplies
`step(v, z) -> x_plus` and a frozen `vbar`, and the generic code differentiates and JVPs through
that callable. It must not name `free_params` or any other block attribute, or the framework half
is unusable for the MSD and Bouc-Wen benchmarks and the boundary is decorative.

**Gantry-specific, in `scripts/gantry/gantry_dynamic/`**: construction of the reference set
(position transform, FD4 velocities), the `Reduced_Gantry_State_Block` callable that satisfies the
contract above, the ten-parameter configuration, the epoch-boundary refresh, and the
`J` versus `[J | Gamma]` diagnostic.

---

## 6. The compile constraint

Compilation is what makes the server run viable and the design must accommodate it, not disable it.

**What is compiled.** `controller.py:257-264` compiles **one model step**, not the rollout:

    torch.compile(fs.hfn,             backend='inductor', mode='reduce-overhead', fullgraph=True)
    torch.compile(fs.hfn.output_only, backend='inductor', mode='reduce-overhead', fullgraph=True)

`Interconnect.forward(self, x, u)` is at `interconnect.py:92` and `output_only(self, x, u=None)` at
`:266`. The step is invoked `nf` times inside `_rollout_segment` (`closed_loop.py:216-235`), and
`closed_loop.py:561` calls `torch.compiler.cudagraph_mark_step_begin()` once per update.
`closed_loop.py:550-556` selects the compiled callables only when `ufuture.is_cuda` and grad is
enabled, so every `no_grad` path is eager by construction.

**What follows.** `reduce-overhead` means CUDA graphs, which replay captured kernels and want
stable input addresses. `fullgraph=True` means nothing can silently graph-break: an untraceable
construction raises on the first update rather than quietly costing you the speedup.

So the coefficient **cannot** be a Python attribute rebound each objective. It must reach the
compiled step in a form that satisfies three requirements simultaneously:

- a stable graph input under CUDA graphs,
- differentiable, so `dL/da` comes back and chains to the ANN weights,
- no recompile per objective.

A tensor argument threaded to the compiled callable is the shape that satisfies all three, because
AOTAutograd lifts arguments into the joint graph. The natural seam is an optional third argument
defaulting to `None` through `Interconnect.forward`, `closed_loop_rollout`, `_rollout_segment` and
the simulator call, so every existing caller, benchmark and eager path is an exact no-op.

**That seam is a target, not an instruction.** I have confirmed the signatures and the call sites
but not how cleanly a third argument reaches one specific block through the interconnect's signal
routing. Choose the seam after tracing the loss closure, satisfy the three requirements, and record
in the `D-` entry which seam you chose and why. Two shapes already considered and rejected:
compiling `_rollout_segment` instead of the step, which means tracing a 400-to-12000 iteration
Python loop under `fullgraph=True`; and carrying the coefficient as ten constant extra states,
which pollutes the state vector, the encoder, the normalization and every diagnostic.

**The larger risk is the JVP, not the coefficient.** Whether

    torch.func.jvp(lambda v: functional_call(block, {'free_params': v}, (z,)), (vbar,), (a,))

traces under inductor with `reduce-overhead` and `fullgraph=True` on torch 2.5.1 is unknown. The
block rebuilds the entire `M(Y)` rational structure on every call and stashes it in the mutable
`_cur` attribute, and `blocks.py:955-983` documents that a `torch.func` transform over this block
leaves functorch `TensorWrapper` duals in `_cur` that break the first validation checkpoint.

**The only JVP cost number in the repository does not transfer.** `implementation/00-cost/`
measured 3.19x on a forward-plus-backward step, but that was CPU, float64, eager, batch 256, and
the stage-0 analysis traced the cost to per-operation dispatch overhead. Production is GPU,
float32, batch 512, with CUDA graphs, which exist to eliminate exactly that overhead. The outcome
is bimodal: cheap if the JVP folds into the captured graph, a cliff if it breaks it. Measure it;
do not extrapolate.

**Fallback ladder, in order, if the probe fails.** Record which rung you took.

1. Traces, no per-objective recompile. The target.
2. Traces but recompiles per objective: the coefficient is not being lifted as an input. Inspect
   `torch._dynamo` recompile logs. `closed_loop.py:604` already carries a recompile detector.
3. Does not trace: measure `fullgraph=False`. The repository's own measurement says losing CUDA
   graphs costs 6.5x down to about 2.6x (`config.py:118-123`, jobs 80610/80634/80652). A 2.6x
   speedup is a real fallback, not a failure.
4. Still too slow: replace the functorch JVP with pure-torch arithmetic. Either a hand-written
   forward-mode pass through the RK4 and the `M(Y)` rebuild, or a directional finite difference
   `(f(vbar + eps*a) - f(vbar)) / eps`, which is two forward calls, compiles without functorch, and
   is differentiable in `a` because `a` enters the primal. Note the costs honestly: the FD form has
   roughly `sqrt(eps_machine)` relative accuracy, about 1e-4 in float32, against an exact JVP
   already validated at 4.6e-16; and the hand-written forward mode is a second implementation of
   the transition, which `docs/pytorch-optimization-guidelines.md` forbids. Validate either against
   the exact JVP on the `implementation/00-cost/` harness before adopting, and say in the `D-`
   entry which rule you are trading against which.

---

## 7. Pinned decisions

These are decided. Log them as one `D-` entry in `docs/decisions.md` before implementing, per the
standing rule, and do not reopen them.

**A. Rank loss at an epoch refresh.** If the rank of `J` is below 10, hold the previous basis and
its `vbar`, log the full singular-value spectrum loudly, and continue. Abort only if it recurs at
two consecutive refreshes. Truncating silently changes the protected space mid-run and makes the
experiment uninterpretable; aborting on a single dip discards a long run for what is more likely a
conditioning artifact of the current expansion point.

**B. Compilation.** `compile_mode='reduce-overhead'` stays on. Section 6 is the contract.

**C. Dtype.** Build `J`, the SVD and `K_J` in **float64** under `no_grad`, off the hot path. Cast
the stored basis and `K_J` to the pipeline dtype (**float32**, `use_f64=False`) once at
construction. `theta_aux`, the JVP and the correction are float32, which the compiled CUDA-graph
region forces anyway. This matches the existing D7.7 convention in
`orth_penalty.py`, which matters because you are comparing against that penalty's basis and a dtype
difference would be a confound. `Reduced_Gantry_State_Block`'s own docstring records why the
construction half wants float64: forming `m_total` and `J_eff` at float32 loses about 1e-7 relative,
reaching the velocity rows as 2.3e-11 absolute, a thousand times the criterion. One line for the
`D-` entry: `training.py::_assert_same_dtype` refuses a resume across a dtype change, and an L-BFGS
polish phase wants float64, so this convention must be revisited rather than inherited if a polish
phase is ever added to an OBC run.

**C2. Orthogonality acceptance threshold.** Do not hard-code one. Measure the floor by projecting a
random field of the same norm through the same basis, and use that as the threshold. At float32
with a condition number around 3e2 expect roughly 1e-6 to 1e-5, not 1e-14. This is data-derived and
re-derives itself when the conditioning or dtype changes, per the acceptance-criteria rule in the
control-engineering stance.

**D. Experiment arms.** Build three configurations: no projection, the existing 2025 penalty behind
`cfg.orth`, and OBC. All three with `joint_estimation=True`, `physics_parameterization='reduced'`,
`param_init_detune=None`, and `combo_init_detune=[1.10, 1.10, 0.90, 1.10, 0.90, 0.90, 1.10, 1.10,
0.90, 1.10]` in `COMBO_NAMES` order, as the entry point already carries. Build them. Run none of
them.

**E. The rho diagnostic.** Report a 2x2 table: basis `J` and `[J | Gamma]`, crossed with the
absorber state at its recorded value and at zero.

    rho(B; delta) = ||P_B delta|| / ||delta||

with `delta` the one-step discrepancy between the eight-state truth and the baseline transition,
restricted to the routed physical rows. The recorded absorber states load through
`gantry_dynamic/data.py:165` (`load_mat_aug`). `delta` is not a function of `(x_b, u)` alone, so the
absorber-state choice is part of the definition and both values must be reported; the difference
between them measures how much the `{x_a = 0}` reference slice restricts what the condition sees.

Why this matters more than its position in the work suggests:
`scripts/gantry/orthogonal-by-construction/investigation-20260915/decision-report.md` measured that
this quantity decides the **sign** of the effect. At `rho = 0.9465` the 2026 subtraction gave mean
combination error 0.4175 against 0.0891 for no projection, roughly four times worse, with the damage
concentrated on the damping combinations. At `rho` near zero it gave 0.0206 against 0.0825, four
times better and ten times more reproducible across seeds. Note that those experiments used
`with_offset=True` (`investigation-20260915/obc_proj.py:81`, called at `s3c_competition.py:94,97`
and `s3e_condition4_control.py:77`), so they measured the **11-column** extended span, while the
plan's primary construction is the 10-column `range(J)`. The primary construction has never been
measured. That is the gap this diagnostic closes, and it gates the next session's training runs
even though it does not gate this session's code.

**F. Reference set and the two cost terms.** Build the complete training set as the definition.
Measure two costs **separately** and report both against the 0.50 s/update compiled baseline
(`gantry_interconnect_dynamic.py:122-123`, job 80713, batch 512, which is a
reduce-overhead number consistent with 2.1 to 2.4 ms/step at nf 200, not an eager one):

- the per-objective ANN pass over the reference set plus the coefficient solve, which happens once
  per objective and is independent of `nf`;
- the per-step JVP, which happens `nf` times per objective.

Decimation reduces the first and **cannot reduce the second by a single operation**. If the first
is the binding cost, decimate as an explicit approximation and report principal angles and singular
values against the full basis. If the second is the binding cost, the lever is the section 6
ladder, never decimation. Report the total as updates per ten-hour wall. Do not act on it: the
threshold is the next session's decision.

# HEURISTIC: the two thresholds worth flagging if crossed, neither derived from anything but
# convenience: a per-objective cost above roughly twice the 0.50 s/update baseline, and resident
# basis plus reference-set memory above roughly 4 GB. The memory estimate is about 0.7 GB
# (J and K_J at 161 MB each in float32, Z_ref 46 MB, retained ANN activations about 194 MB), which
# is minor on a 24 GB card and material on an 11 GB RTX 2080 Ti next to a BPTT graph the config
# already budgets at up to 25 GB. Measure on the card the run will use.

**G. The remnant. Do not touch it.** `model_augmentation/fit_systems/obc_projection.py` is untracked
and is a **remnant**, not the current implementation. It recomputes the coefficient inside
`forward`, which violates the section 2 invariant, and it builds an 11-column `[J | c]` basis while
the plan's primary construction is 10-column. Do not import it, adopt it, extend it, or delete it.
Five committed tests currently import it (`implementation/02-rank`, `03-taylor`, `04-projection`,
`05-gradient`, `06-integration`); **do not modify those either**, and do not cite their results as
evidence about the new construction. Reconciling that pairing is a separate attended job.

Two things in it are worth reading rather than reusing, and both are recorded here so you do not
have to: the column-scaling note that the offset column's singular value is about 1030x the largest
parameter column, and the note that a raw inner product is dominated by the offset direction and
inverted a conclusion where the scale-free `rho` showed an improvement. Both are evidence for the
plan's decision to make `range(J)` primary.

**H. New module name.** `model_augmentation/fit_systems/obc.py`. A new name, so nothing inherits the
remnant's identity.

**I. FD4 at record boundaries.** Fourth-order central difference in the interior only. Trim the two
samples at each end of each record from the reference set rather than mixing stencil orders. That is
56 samples out of roughly 672000 and it keeps the velocity estimator uniform across the stack.
# THEORY: the fourth-order central first-derivative stencil
# (-q[k+2] + 8 q[k+1] - 8 q[k-1] + q[k-2]) / (12 Ts) has gain (8 sin(w Ts) - sin(2 w Ts)) / (6 w Ts),
# which is 0.14% low at 230 Hz at fs = 4000, against 2.2% for the second-order central difference.
# The excitation band of the active mode is 140 to 230 Hz and the damping columns of J multiply
# velocity, so the second-order stencil biases exactly the fragile columns.

**J. Configuration surface.** Add dedicated OBC fields to `RunConfig` and dedicated experiment
configurations. Do **not** change the standard training defaults in `RunConfig` or in
`gantry_interconnect_dynamic.py`. A run that did not ask for OBC must be bit-identical to what it is
today, and the disabled path must return the wrapped block's output tensor itself, not a
numerically equal copy.

**K. Documentation deliverable.** One `D-` entry in `docs/decisions.md` covering every decision in
this section plus the seam and ladder rung you chose. One stage entry in
`scripts/gantry/orthogonal-by-construction/implementation/PROGRESS.md` in its existing format. No
`tasks/todo.md` churn.

---

## 8. Ordering constraints

Three, not a procedure. Sequence the rest yourself.

1. **The compile-feasibility probe comes first and gates the correction design.** A standalone
   script, no dataset, no training: wrap a step that applies `f_aug - jvp(...)` with the coefficient
   as a tensor input, compile it with `backend='inductor', mode='reduce-overhead', fullgraph=True`,
   run it at least twice on CUDA with different coefficient values, and report whether it traces,
   whether it recompiles on the second call, and the per-step time against the uncorrected compiled
   step. Everything in section 6 depends on the answer. Reuse the `implementation/00-cost/` harness.
2. **The basis builder precedes the rho diagnostic**, which needs it.
3. **The cost benchmark precedes any statement about whether the full reference set is viable.**

Tracing the loss closure in `SSE_Interconnect_Composed` (where one objective evaluation begins and
ends, including the checkpointed and compiled paths) is what tells you where the coefficient can be
created and cleared. Do it early; it is not a gate and it will not make us discontinue.

---

## 9. Verification gates

All eight must pass and be reported with the number that shows it.

1. **Gradient flow.** `d(loss)/d(ANN weights)` is nonzero through the coefficient path. The
   discriminating test is that detaching the coefficient changes the gradient: compute both and
   report the norms.
2. **Numerical orthogonality.** `||J^T Ftil||` at or below the measured floor from C2, checked at
   several objective evaluations across at least one optimizer update, not only at construction.
3. **Invariant held.** The reference set is evaluated exactly once per objective. Assert it by
   counting, not by inspection.
4. **Additional-state passthrough.** The rows the ANN writes into `x_a` are bit-identical with the
   correction enabled and disabled. The basis has no rows there; assert it rather than special-case
   it.
5. **Disabled-path equivalence.** With OBC off, the pipeline is bit-identical to today, not merely
   close.
6. **Checkpoint compatibility.** Save and reload across a validation, including the `_cur` cache
   interaction documented at `blocks.py:955-983`. This is a known failure with a known symptom
   (`NotImplementedError: Cannot access storage of TensorWrapper`), so it is a test, not an unknown.
7. **Physical parameters still update.** The ten combinations move under training with OBC enabled.
   A projection that silently zeroes the parameter gradient would pass gates 1 to 6.
8. **No per-objective recompile.** Over at least ten consecutive objective evaluations.

---

## 10. Files you may create or modify

Create: `model_augmentation/fit_systems/obc.py`; new modules under
`scripts/gantry/gantry_dynamic/` for the reference set and basis construction; test and probe
scripts under `scripts/gantry/orthogonal-by-construction/implementation/`.

Modify: `scripts/gantry/gantry_dynamic/config.py` (new fields only), the seam files identified in
section 6, `docs/decisions.md`, and `implementation/PROGRESS.md`.

---

## 11. Out of scope

- **No training run.** Not a short one, not a smoke test that trains, not even if every gate passes.
- No changes to standard training defaults in `RunConfig` or `gantry_interconnect_dynamic.py`.
- No reading of the Gyorok papers beyond the four semantics in section 3.
- No `[J | Gamma]` implementation beyond what the rho diagnostic in 7E needs. The primary
  construction is `range(J)`.
- No touching `obc_projection.py` or the five committed stage tests that import it (7G).
- No touching `kamtin-fp-model/`.
- No multistep, input-output, or closed-loop-Jacobian orthogonality. One-step plant state equation
  only. This is the scope the supervisor set.
- Do not act on the cost numbers from 7F. Measure and report; the threshold is the next session's.

---

## 12. Acceptance criterion

Done when all eight gates in section 9 pass with reported numbers, the 2x2 rho table from 7E is
produced, the two cost terms from 7F are measured on the target card with the ladder rung recorded,
the three configurations from 7D exist and none has been run, and the `D-` entry and `PROGRESS.md`
stage entry are written.

---

## 13. How to work

You are operating autonomously. The user is not watching in real time and cannot answer questions
mid-task, so asking 'Want me to...?' or 'Shall I...?' will block the work. For reversible actions
that follow from the original request, proceed without asking. Stop only for destructive actions or
genuine scope changes the user must decide. Offering follow-ups after the task is done is fine;
asking permission before doing the work is not.

Before ending your turn, check your last paragraph. If it is a plan, an analysis, a question, a list
of next steps, or a promise about work you have not done ('I'll...', 'let me know when...'), do that
work now with tool calls. That includes retrying after errors and gathering missing information
yourself. Do not stop because the context or session is long. You have ample context remaining; do
not stop, summarize, or suggest a new session on account of context limits. End your turn only when
the task is complete or you are blocked on input only the user can provide.

The scope above is the deliverable: do not quietly narrow, widen, or swap it. If one part turns out
to be blocked, complete every other part in full and say exactly what you left out and why. Where
the task is ambiguous, implement the reading its wording and the surrounding code most directly
support, state that assumption in your summary, and do not build for the other readings as well.

Do not add features, refactor, or introduce abstractions beyond what the task requires. Do not
design for hypothetical future requirements. Do not add error handling or validation for scenarios
that cannot happen; validate at boundaries only. If you find a pre-existing bug or a performance
concern the task does not mention, do not fix it in this change unless the requested behaviour
cannot work without it; report it as a follow-up in your summary. Verify your work however you like;
scratch scripts and quick checks need not be kept, and should live in the session scratchpad rather
than the repository. Commit tests only where this handoff asks for them or the repository already
keeps tests for this kind of change, sized like the neighbouring files, and do not turn scratch
checks into additional permanent test files. This is about extras only: implement every behaviour
section 9 asks for, completely.

The number of tokens used to edit files is best minimized, all else being equal. When it will not
affect the end result, surgically edit a file rather than rewrite the entire thing. `blocks.py` is
1271 lines and `interconnect.py` is larger.

Before reporting progress, audit each claim against a tool result from this session. Only report
work you can point to evidence for; if something is not yet verified, say so explicitly. If a gate
fails, say so with the output. If a step was skipped, say that. When something is done and verified,
state it plainly without hedging.

Your final message is the user's first look at all of this. Write it as a re-grounding, not a
continuation of your working thread: the outcome first, then the one or two things you need from
them, each explained as if new. Drop the working shorthand. Write complete sentences and spell out
terms instead of abbreviating them. Do not use arrow chains or labels you invented while working.
When you mention a file, a gate, or a number, give it its own plain-language clause saying what it
is. Lead with what happened, then the supporting detail.

Use lists and tables where the content is multifaceted enough that they help. Mannered prose
substitutes metaphor and flourish for direct statement; say what you mean, and when a literal phrase
is available, use it.
