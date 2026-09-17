# Handoff: build a clean probe for whether CUDA graphs can run the one-step OBC rollout, and search the PyTorch issue tracker for the failure
**From**: session of 2026-09-17 | **Branch**: `Augmentation` | **Effort suggested**: `high`

## 1. Task

Two deliverables, in this order. First, **find out online whether our failure is a known
PyTorch/inductor issue** and whether a workaround exists: the compiled closed-loop rollout uses
`mode='reduce-overhead'` and never completes a single optimizer update at `nf=400` because
cudagraph trees re-records instead of replaying. Use the `deep-research` skill only if the search
turns academic; for an issue-tracker search, targeted web search is correct. Second, **write one
small probe script** that answers the CUDA-graph question in a single short job, and run it. The
probe this session produced (`probe_compile.py`) is not that script: it grew to 300 lines, it
toggles configurations inside one process, and that toggling is itself a confound (section 6).
Write a new one, or cut the existing one down to the single measurement. The measurement wanted
is: per-update wall time at the production shape, `nf=400`, `batch=512`, in one fixed
configuration per process, with the first two updates discarded as compile and record.

## 2. Out of scope

* Do not change the OBC mathematics, the hand-written forward sensitivity, or the reference set.
  All three are validated (section 7) and are not suspected.
* Do not start the paired training runs. That is the user's call and is section 8's alternative.
* Do not touch `docs/decisions.md` D-192 through D-195; they are current. At the time this handoff
  was written the static-buffer decision was deliberately unwritten; it is now D-199.
* Do not modify `kamtin-fp-model/`, and do not touch job 83795's outputs (a 400-epoch
  `joint_estimation=True`, no-projection run on blade3 that has been going since 2026-09-16
  23:29 and predates every change made today).
* Do not pursue the batch-size or TF32 levers here. Both are real and both are experiment-design
  decisions the user has not taken.

## 3. Where things stand

Branch `Augmentation`, last commit `a52dd55`. The tree is dirty across
`model_augmentation/fit_systems/` (`blocks.py`, `interconnect.py`, `closed_loop.py`, new
`obc.py`), `scripts/gantry/gantry_dynamic/` (`config.py`, `model.py`, `obc_gantry.py`, new
`obc_diagnostics.py`), `scripts/gantry/gantry_interconnect_dynamic.py`,
`scripts/gantry/orthogonal-by-construction/implementation/08-one-step/` (new) and
`scripts/gantry/orthogonal-by-construction/runners/` (new). Treat everything else in the dirty
tree as unrelated user work.

Jobs 83962 and 83978 were the two CUDA-graph attempts and were cancelled (section 6). Job 83795
is still running on blade3 and is unrelated to today's code.

`CFG` in `gantry_interconnect_dynamic.py` currently reads `compile_mode='reduce-overhead'`,
`static_pass_buffers=True`, `epochs=400`. If the user launches the pair before this task starts,
those become `'default'`, `False` and `300`.

## 4. Established and verified

1. **The projection works and is correct.** Fourteen gates pass in
   `08-one-step/test_behavior.py`, including: coefficient computed once per objective, gradients
   reaching the ANN through it, orthogonality residual 9.25e-09 against a measured float32 floor
   of 5.97e-08, additional-state rows exactly zero, and the disabled path bit-identical to commit
   `a52dd55`.
2. **The hand-written forward sensitivity equals `torch.func.jvp`.** `08-one-step/test_tangent.py`:
   per-term agreement `0.000e+00` on all seven terms, end to end 2.36e-17, and 4.16e-10 against a
   central finite difference that uses no autodiff at all.
3. **`default` works at the production shape.** Job 83949, A100, `nf=400`, `batch=512`: control
   0.97 s/update, OBC 1.89 s/update, ratio 1.95x, peak memory 1.32 GB of 85, 98 percent headroom.
4. **`reduce-overhead` does not.** Job 83951 probe: 525.7 s then 1472.8 s for successive updates
   at `nf=400`, degrading. Jobs 83962 and 83978: no single update completed in 25+ minutes.
5. **At `nf=20` `reduce-overhead` is the faster mode**, 0.08 s/objective against `default`'s 0.11,
   same card, same code (job 83949). So the failure is specific to the long horizon.
6. **The parameter prior would have pinned the physical parameters.** `probe_prior_scale.py` on
   the A100: `||g_prior||/||g_data||` is 49 at ten percent along the recovery path and 1433 at the
   true values, exceeding the data term on all ten combinations. Hence `param_prior=False`
   (D-193).
7. **The dataset qualifies structurally.** Rank 10 of 10 at both parameter points, condition
   number 892.5, every leave-one-record-out retains rank 10. `cg1` against `cg2` correlate at
   minus 0.98 and are the weak direction.

## 5. Assumed but not verified

1. **That threading freshly allocated tensors is what breaks cudagraph replay.** This is the
   hypothesis behind the now-D-199 proposal and it is NOT established. Evidence for: the failure appeared only
   after D-194 and D-195 introduced the threading, and job 83795 runs `reduce-overhead` at
   `nf=400` at 0.77 s/update without it. Evidence against: the D-199 fix did not help (section 6),
   which either falsifies the hypothesis or means the fix is incomplete. **The diagnostic that
   settles it exists and was never run**: `probe_compile.py --mode reduce-overhead --no-thread`
   disables both the hoist and the correction, reproducing 83795's configuration. If that is fast,
   threading is the cause; if it is also slow, the cause is elsewhere and D-199 is dead.
2. That 400 sequential invocations of the compiled step inside one
   `cudagraph_mark_step_begin` window is within what cudagraph trees supports. Job 83795 suggests
   yes, but 83795's step is a different graph.
3. That the static buffers in D-199 actually reach inductor as static. `mark_static_address` is
   called but its effect was never confirmed; it is possible the buffers are correct and something
   downstream still allocates.

## 6. Tried and failed

- **`torch.func.jvp` inside the compiled step** -> `CUDA error: an illegal memory access was
  encountered` at the first objective, in every inductor mode including `default` (jobs 83796,
  83913) -> inductor miscompiles codegen over a forward-mode transform; `aot_eager` traced the
  same graph cleanly, so it is codegen, not tracing -> D-194. Replaced by the hand-written
  sensitivity; do not reintroduce the transform into the step.
- **`fullgraph=False` to let the transform graph-break out** -> Dynamo reported one graph and
  zero breaks, so nothing broke out and the codegen was identical -> job 83913.
- **Caching the per-pass tensors on the module and choosing by `cached[0] is theta`** -> 34.71 s
  per objective at `nf=20` -> under Dynamo the identity test compares a graph input against a
  closed-over constant, is False at trace time, and the compiled graph bakes in the fallback
  branch that calls `torch.func.jvp` per timestep. Eager took the other branch, so an eager
  counter could not see it -> job 83947. Gate `T13` in `test_behavior.py` now counts functorch
  calls inside a compiled step and requires zero.
- **D-199 static-address buffers** (`StaticPassBuffers` and `_StaticCopy` in
  `model_augmentation/fit_systems/obc.py`) -> job 83978 still produced no update in 25 minutes ->
  mechanism unknown; either the hypothesis in 5.1 is wrong, or the buffers are not reaching
  inductor as static, or something else in the step still allocates per call. **The fix is
  correct but ineffective**: with it on, loss and both gradients are bit-identical to with it off,
  over three consecutive updates in both arms.
- **`probe_compile.py` as the measurement vehicle** -> it toggles `obc_correction` on and off
  inside one process to compare states, which forces a recompile and a fresh CUDA-graph recording
  mid-run, and its first `reduce-overhead` numbers were partly that artefact -> this is why
  section 1 asks for a new probe with one fixed configuration per process.

## 7. Achieved

Implemented and validated: the one-step OBC path end to end (D-192), the parameter-prior switch
(D-193), the hand-written forward sensitivity replacing `torch.func.jvp` (D-194), and the
per-pass hoisting of the M(Y) rebuild (D-195). Artefacts:
`08-one-step/test_behavior.py` (14 gates), `08-one-step/test_tangent.py` (derivative gates),
`results/2026-09-17/`.

Measured gains today, all at `nf=400`, `batch=512`, A100: the corrected step went from not
compiling at all, to 937 s per update, to 1.89. The M(Y) hoist cut 51 rebuilds per objective to
2 and is worth 1.36x to 1.52x locally; it applies to every joint-estimation run, not only the
projected one.

Implemented but, at the time of this handoff, NOT performance-validated: D-199 static buffers. Correct (bit-identical loss and gradients)
but with no demonstrated effect. Off by default via `RunConfig.static_pass_buffers`.

## 8. The open question

**Is `reduce-overhead` recoverable for this rollout, and if so how?** Candidate answers:
(a) threading is the cause and the D-199 buffers are the right fix but incomplete, which the
`--no-thread` diagnostic plus an inductor-level check of whether the buffers are treated as
static would settle; (b) threading is irrelevant and something about 400 sequential invocations
or this specific graph defeats cudagraph trees, which the same `--no-thread` run settles in the
opposite direction; (c) it is a known upstream issue with a documented workaround, which the web
search in section 1 settles.

The alternative the user may prefer: stop here and launch the pair on `default`, which is
measured and works, at 0.97 and 1.89 s per update. CUDA graphs are worth roughly 1.4x, about
eight hours on a 300-epoch OBC arm. That is the user's decision, not the successor's.

## 9. Next action

**Search for the failure signature before writing or running anything.** It costs nothing, it is
local, and it may make the probe unnecessary. The signature to search, in the user's terms: an
inductor `mode='reduce-overhead'` compiled step invoked several hundred times sequentially inside
one `cudagraph_mark_step_begin` window, given freshly allocated input tensors each iteration,
which re-records the cudagraph tree instead of replaying and slows down monotonically (525 s then
1472 s per update, then no completion at all). Start with `github.com/pytorch/pytorch` issues on
cudagraph trees, changing input addresses, `mark_static_address`, and re-recording, and with the
torch.compile troubleshooting docs on CUDA graph requirements. Torch here is 2.5.1, which is many
releases old, so check whether the behaviour changed upstream.

Then write the probe, informed by what the search says the right instrument is. The probe should
be small enough to read in one screen and should answer one question: median seconds per update
at the production shape in one fixed configuration. The existing `probe_compile.py` is available
as a source of working setup code (building the rig, the production batch, the guards) but it is
not a good template and should not be extended further; see section 6.

Submitting anything to the cluster is the step after that, and the user decides when.

## 10. Acceptance criterion

The probe is done when one number exists: **median seconds per update at `nf=400`, `batch=512`,
over at least three updates after discarding the first two**, for `reduce-overhead` in a single
fixed configuration. The threshold is `default`'s measured 1.89 s/update for the OBC arm and
0.97 s for the control (job 83949, same card, same code). Below 1.89 means CUDA graphs are worth
keeping; at or above means they are not, and that is a complete and reportable answer.

The `--no-thread` rung has its own reference: job 83795 runs that configuration at 0.77 s/update,
so anything near 0.77 confirms threading as the cause and anything in minutes refutes it.

## 11. Read these first

1. `docs/decisions.md` D-192 to D-195, newest first. The construction, the prior, why
   `torch.func.jvp` had to go, and the per-pass hoisting rule.
2. `scripts/gantry/orthogonal-by-construction/implementation/08-one-step/probe_compile.py`. The
   existing probe, including the `--no-thread` and `--static-pass` flags and the guards.
3. `model_augmentation/fit_systems/obc.py`, `_StaticCopy` and `StaticPassBuffers`. The D-199 fix
   whose effect is unknown.
4. `model_augmentation/fit_systems/interconnect.py`, `Interconnect.forward` and
   `SSE_Interconnect_Composed.loss`. Where the two per-pass objects are built and threaded.
5. `scripts/gantry/GPU/gantry_interconnect_dynamic_gpu.sh`. The runner for job 83795, the working
   `reduce-overhead` configuration, and its comment recording ~500 s of one-off compile.

## 12. Do not

* Do not reintroduce `torch.func.jvp` into the compiled step (section 6, first entry).
* Do not judge a rung on its first production update: the first, and under `reduce-overhead`
  often the second, is compile and CUDA-graph recording, not per-update cost. This session
  misread 623.5 s as a steady cost and lost a job to it.
* Do not build a probe that changes configuration inside one process. One configuration per
  process, as `probe_compile.py --mode` already does per rung.
* Do not use an eager counter to verify behaviour of a compiled path (section 6, third entry).
* Do not modify `CFG` between the two paired launches if the user decides to launch.

## 13. Operational

**Every compile measurement must run on the cluster.** The development PC's Quadro P2000 is
compute capability 6.1 and Triton refuses anything below 7.0, so inductor cannot run locally at
all and `torch.compile` there only reaches `aot_eager`. What IS worth doing locally: the
correctness gates, `08-one-step/test_behavior.py` and `test_tangent.py`, both CPU, both a few
minutes. Nothing about CUDA graphs can be answered without a job.

Conda env `GraduationProject`. Cluster partition `molokai`, node blade6, one A100 80 GB, torch
2.5.1, triton 3.1.0, gcc 11.4. Logs go to
`/home/dirk_van_den_berg/logs/augmentation/orthogonal-by-construction/`; that directory exists.
Probe JSON and logs land in
`scripts/gantry/orthogonal-by-construction/implementation/results/<date>/`.

The repo on the server is `/dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation` and is
updated by the user copying files from the Windows checkout; it is not a git remote, so verify
what is deployed rather than assuming. Runs record no git provenance unless
`export GIT_SHA=$(git rev-parse --short HEAD)` precedes `sbatch`.

Expected runtime: the three-rung diagnostic is about 30 minutes inside a 1 hour limit. Each rung
pays roughly 500 s of CUDA-graph recording before its first usable number.

## 14. Delegation

None. The search is a handful of targeted queries against the PyTorch issue tracker and the
torch.compile documentation, which is inline work. Writing the probe is a single small file. An
Explore subagent is not warranted for either, and the project default is one only for a genuinely
wide codebase sweep.
