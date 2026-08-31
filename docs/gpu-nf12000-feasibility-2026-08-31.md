# GPU feasibility for a 3 s training horizon (nf = 12000)

**Question.** Can `scripts/gantry/gantry_interconnect_dynamic.py` train at `nf_seconds = 3.0`
(nf = 12000 at 4 kHz), on all 14 train and 4 validation trajectories, for 5 epochs, inside a
10 hour wall, by moving to GPU?

**Answer.** Yes, at an estimated ~7 h on a single GPU, under two conditions that are not optional:
chunked gradient checkpointing over the rollout, and a batch size of ~2900-4096 instead of 256.
An 8 GB RTX 2080 is sufficient; an A100 buys almost nothing.

General PyTorch rules (compile the step body not the loop, `reduce-overhead`, hot-path bans) live in
`docs/pytorch-optimization-guidelines.md` and are not repeated here. That document was written for the
`lpv_lfr_baseline/` pipeline and its "keep float64 throughout" rule does **not** apply to this pipeline,
which sets `use_f64=False` on measured grounds (see the `cl_update_precision` note in the entry file).

Every number below marked ESTIMATE is unmeasured and is closed by the probe in section 8.

---

## 1. Why the pipeline is dispatch-bound, not compute-bound

Per batch update at the current config (`closed_loop=True`, nf = 400, batch 256, stride 10):

| Quantity | Value |
|---|---|
| Windows / batch updates per epoch | 66,626 / 260 |
| Measured wall cost | 4.9-6.5 s per batch; loss 68-77%, validation 23-31% |
| Rollout structure | Python `for` loop over nf timesteps in `closed_loop.py::closed_loop_rollout` |
| State / input / output dims | nx = 6 physical + 2 ANN = 8, nu = 3, ny = 3 |
| ANN | 11 -> 16 -> 16 -> 8, tanh |
| Per timestep | RK4 with `up_sample=1` (4 `deriv` calls), LPV `M(Y)^-1` as a Horner rational (no matrix solve), controller `baddbmm` on (256,12,9) |
| Tensor ops per timestep | ~140 |
| Largest single matmul | (256,15) @ (15,6), ~23 kFLOP |

Arithmetic per update is ~1.2 GFLOP including backward, which is roughly **0.1 ms** of work on a
mid-range GPU. The measured cost is 4-6 s. So >99.9% of runtime is the fixed per-operation cost of
~170k tiny operations passing through the Python interpreter, the PyTorch dispatcher, and autograd
graph construction.

**That cost is device independent.** Moving to CUDA removes no Python call and no autograd node; it
replaces a ~1 us CPU kernel call on L1-resident data with a ~5 us asynchronous launch. This is why
earlier GPU attempts on this pipeline produced no improvement, and it is consistent with the framework's
own note in `closed_loop.py` that for these matrix sizes "the arithmetic is free and the dispatch count
is the whole cost".

The relevant confirmation for a port: there are **no** `.numpy()`, `.item()`, or NumPy calls inside the
rollout, and **no** `torch.linalg.solve`/`inverse` per step. The rollout is pure batched tensor ops.

---

## 2. The budget equation

```
T_train  =  U x nf x t_step          U = batch updates,  t_step = cost of one rollout step for the whole batch
U        =  N_windows x epochs / B
```

The decisive property, which follows from section 1: **`t_step` is essentially independent of `B`** up
to a few thousand, because the cost is one kernel launch per op regardless of how wide the tensor is.

Three consequences:

1. **Batch size is free on GPU.** It does not appear in `T_train` except by reducing `U`.
2. **Using all the data gets cheaper as `B` grows**, which is the inverse of CPU intuition.
3. **Neither more GPUs nor a bigger GPU shortens the run.** Under DDP each rank still walks all 12,000
   sequential steps, so an update still costs `nf x t_step`. Extra GPUs buy a larger effective batch at
   equal wall time, never a shorter wall time. They are for running independent jobs in parallel.

A 10 h wall therefore purchases a fixed number of updates, and nothing else does:

| `t_step` (ESTIMATE) | Updates in 10 h |
|---|---|
| 3.4 ms, current op count | ~880 |
| 1.7 ms, ARTBP (backward halved) | ~1,760 |
| 1.0 ms, fused / compiled step | ~3,000 |

The 3.4 ms figure is built as ~140 launches x ~6 us forward, doubled for backward, plus one forward for
checkpoint recompute.

---

## 3. Coverage: stride is a redundancy knob, not a coverage knob

All 14 train and 4 validation trajectories are used at any stride. Stride sets only how densely window
*start points* are sampled inside each record. At nf = 12000 with stride 10, every sample already appears
in 1,200 distinct windows.

Window count at stride 1: `(48000 - 12000 - 17) x 14 = 503,762`.

---

## 4. Configuration that fits

| Setting | Value | Rationale |
|---|---|---|
| `nf_seconds` | 3.0 (nf = 12000) | the target |
| `batch_size` | 2900-4096 | `B = 504k x 5 / 880`; free in wall time, capped by VRAM |
| `stride` | 1 | all windows; affordable precisely because `B` is large |
| checkpoint chunk | 200 steps | 60 boundaries; +33% compute |
| `epochs` | 5 | as specified |
| `lr` | **re-probe** | `1e-6` was measured at B = 256 (`cl_update_lr.py`); an 8-16x wider batch changes the gradient noise it was tuned against |
| validation | stack the 4 records into one batch-4 free run | ~48k steps, ~2 min/epoch versus ~6 min sequential today |
| `use_f64` | False | already justified numerically; also a 1/32-rate cliff on Turing |

Budget: 615 updates x 12000 x 3.4 ms = **7.0 h** training, +0.2 h validation, +~1 h post-training
baselines and diagnostics, total **~8.2 h** against a 10 h wall.

---

## 5. Memory arithmetic

The binding constraint, and the reason checkpointing is mandatory rather than an optimisation.

Without checkpointing the BPTT graph stores activations for all 12,000 steps. At an ESTIMATED 4 kB of
stored intermediates per sample per timestep:

| Configuration | Activation memory |
|---|---|
| now: 256 x 400 | ~0.4 GB |
| nf = 12000, B = 256 | ~12 GB (already past most cards, at the *current* batch) |
| nf = 12000, B = 4096 | ~200 GB (impossible on any hardware) |

With chunked checkpointing, peak memory is one chunk plus the batch tensors:

| Item | at B = 4096 |
|---|---|
| `ufuture` + `yfuture`, each (B, 12000, 3) float32 | 2 x 590 MB = 1.2 GB |
| checkpointed activations, chunk 200 | ~3.3 GB |
| model, optimiser, orth basis `Q` | <0.1 GB |
| **total** | **~4.6 GB** |

Fits 8 GB with margin. B = 8192 does not (~9 GB), so **4096 is the practical ceiling on the RTX 2080
nodes**; chunk 100 buys back ~1.6 GB if needed.

Two hard requirements that follow:

- The raw dataset is 18 records x 48000 x 6 floats = ~20 MB. Keep all of it resident on the GPU and slice
  windows as **views**. Materialising 504k windows of length 12000 would be ~9.5 TB.
- The batch load into a static buffer costs ~3 ms against a ~40 s update, i.e. free.

---

## 6. Hardware: the card choice barely matters

Cluster inventory (snapshot 2026-08-31, occupancy ignored):

| Node | Partition | GPU | Count |
|---|---|---|---|
| blade1, blade2 | `oahu`, `mpi` | RTX 2080 (Turing, consumer, 8 GB) | 14 |
| blade3 | `hawaii` | RTX 6000 (Turing, 24 GB) | 1 |
| blade5, blade6 | `lanai`, `molokai` | A100 (Ampere, 40-80 GB) | 13 |
| blade4 | `kauai`, `mpi` | none | 0 |
| quad | `maui` | DOWN+DRAIN | unusable |

For a launch-bound workload of tiny kernels the A100's advantages are the wrong ones. Its 4x memory
bandwidth and 6x SM count are unused by a (4096,16)@(16,16) matmul; what matters is per-kernel latency,
which tracks **clock**, and there the RTX 2080 (~1.8 GHz boost) is *ahead* of the A100 (~1.41 GHz).

The A100 wins on exactly two axes, neither currently binding:

1. **VRAM**, 40-80 GB vs 8 GB. Irrelevant once checkpointing is in, and pushing `B` past ~4096 is
   counterproductive anyway: it cuts `U` below the point where any optimisation happens.
2. **FP64**, 1/2 rate vs 1/32 on Turing. Only matters if `use_f64` is ever flipped, which on `oahu`
   would be a ~30x cliff rather than merely expensive.

**Conclusion:** target whichever card frees first. Use the surplus GPUs for parallel *jobs* (lr re-probe,
stride, seed replicates, ablations), not for one faster run.

---

## 7. Optimisation levers, ranked

Ranked by expected value for this specific profile: fixed shapes, no data-dependent branching, no RNG in
the rollout, ~140 tiny ops x 12,000 sequential steps.

**1. CUDA Graphs over a chunk of steps. Largest win.**
A replay submits the whole captured block with a single `cudaGraphLaunch`, eliminating the CPU-side
dispatch that is ~99% of the cost. Capturing 200 steps as one graph takes launches per rollout from ~1.7M
to ~60. Requirements are all met by this rollout: static buffers updated with `.copy_()` rather than
reassignment, fixed shapes, no CPU-GPU sync inside, no dynamic control flow. Because `ufuture`/`yfuture`
are preallocated `(B, nf, 3)` tensors, the per-step slices sit at fixed addresses inside a captured chunk
and need no copying at all.

> **Known conflict.** Graph capture and gradient checkpointing do not compose. `make_graphed_callables`
> is documented incompatible with reentrant checkpointing, and non-reentrant checkpoint raises
> `RuntimeError: Checkpointing is not compatible with .grad()` (pytorch#82465). Since this plan needs
> both, it requires **manual** chunk recomputation (forward-only capture for the no-grad pass, a separate
> forward+backward capture for the recompute) rather than `torch.utils.checkpoint`. This is the main
> engineering risk in the plan.

**2. `torch.compile` on the step function. Lower risk, still large.**
Fuses ~140 ops into perhaps 10-20 kernels, and composes with non-reentrant checkpointing (Dynamo has
first-class handling for it) unlike raw capture. Caveat: `torch.compile` carries a per-call overhead that
can make it *slower* than eager for very small kernels (KernelBench Level 1). Measure, do not assume.

> **Trap.** Dynamo unrolls Python loops without bound. Compiling the 12,000-step loop yields an FX graph
> of ~1.7M nodes; reported symptoms are compile times of minutes to hours and stack-explosion crashes
> (pytorch#97155, #111441, #102839), and there is a logged case of **silently wrong results** on unrolled
> loops (pytorch#96064), which is unacceptable in an identification setting. Compile the step **body**;
> leave the Python loop intact. This is already the rule in
> `docs/pytorch-optimization-guidelines.md`.

**3. `mode="reduce-overhead"`.** Applies CUDA Graphs automatically via CUDAGraph Trees, i.e. the cheap
version of lever 1. PyTorch maintainers state that manually capturing a compiled graph is unsupported, so
it is this or manual capture, not both. NVIDIA notes it can fragment into many small graphs rather than
few large ones, which is precisely the failure mode here, so verify rather than trust.

**4. Hand-fusing the `Interconnect`. Most predictable, no compiler dependency.**
`Interconnect.forward` and `output_only` allocate fresh `torch.zeros` per signal per call and re-check
device/dtype on the connection matrices every call. For this fixed 8-state graph a single precomputed
dense update skips the generic machinery. Likely worth 2-3x on op count alone.

**Levers that will not help, recorded so they are not retried:**

| Lever | Why not |
|---|---|
| fp16 / bf16 / AMP | launch-bound, not bandwidth-bound; 16x16 matrices do not engage tensor cores; numerically risky here |
| TF32 | Turing does not have it |
| More GPUs for one run | a sequential rollout does not shard (section 2) |
| `cudnn.benchmark` | no convolutions in this model |
| `B` beyond ~4096 | costs updates, saves no wall time |

---

## 8. What is estimated, and the probe that closes it

Unmeasured quantities the plan rests on: `t_step` (~3.4 ms), activation bytes per sample per step
(~4 kB), and the claim that `t_step` is batch-independent.

The probe, one array job across the idle GPUs, ~20 min, must answer four things:

1. Baseline eager `t_step` and peak memory for nf in {400, 1600, 3200} and B in {256, 2048}. Both
   quantities are linear in nf because the rollout is a `for` loop, so extrapolation to 12,000 is valid
   and never requires running 12,000. Curvature is itself a finding.
2. Whether `t_step` really is flat in `B`. The whole plan rests on this.
3. Whether `torch.compile` on the **step function** works and helps.
4. Whether a 200-step chunk captures cleanly as a CUDA graph.

If levers 1 and 4 land, `t_step` plausibly falls toward ~0.5-1.0 ms, turning the 10 h budget from ~880
updates into several thousand and making nf = 12000 a comfortable regime rather than a tight one.

---

## 9. Open risks that are not about compute

**Update count.** 5 epochs at B = 4096 is ~615 updates, versus 1300 in the current nf = 400 shakedown.
The run fits, but it performs *less* optimisation on much wider batches. Treat the first one as a
feasibility result, not a converged model. This is the strongest argument for pursuing levers 1-2 rather
than accepting 880 updates.

**Independent-window ceiling.** 14 records of 12 s give only **56 genuinely independent 3 s windows**.
The 504k strided windows overlap by up to 99.99%. A batch of 4096 is therefore not 4096 independent
experiments, and gradient noise may not fall in proportion to `B`, which blunts the batch-scaling
argument that the whole plan leans on. Additionally, if the multisine period is 1 s
(`generate_multisine_data.m`, `N_period = 20000` at 20 kHz), a 3 s window contains three repeats of the
same excitation. **This is a data question, not an engineering one, and it is the largest open risk.**

**Gradient depth.** A 30x deeper unroll through a lightly damped closed-loop system risks exploding or
vanishing sensitivity, and it interacts with the measured `lr = 1e-6`. `scripts/gantry/ARTBP/` already
implements randomly truncated BPTT, which is the standard answer and also halves the backward cost.
Decide up front between true 12,000-step BPTT and an ARTBP-truncated gradient; they cost very differently.

**Cheaper alternative if the motivation is Y coverage.** If 3 s is wanted because at 0.1 s the scheduling
variable `Y` barely moves within a window, then 4 kHz resolution across all 3 s is not required.
`gantry_dynamic/multirate_data.py` and `multirate_premise.py` already exist and would deliver the
scheduling variation at a fraction of 12,000 steps. Which motivation applies decides whether the GPU port
is the right work at all.

---

## 10. References

- Accelerating PyTorch with CUDA Graphs: https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
- NVIDIA, PyTorch CUDA Graph Integration: https://docs.nvidia.com/dl-cuda-graph/latest/torch-cuda-graph/torch-integration.html
- NVIDIA, CUDA Graph Best Practice, Introduction: https://docs.nvidia.com/dl-cuda-graph/latest/cuda-graph-basics/introduction.html
- `torch.cuda.make_graphed_callables`: https://docs.pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html
- pytorch#82465, checkpoint vs `make_graphed_callables`: https://github.com/pytorch/pytorch/issues/82465
- pytorch#97155, custom RNN slow to compile for long sequences: https://github.com/pytorch/pytorch/issues/97155
- pytorch#102839, Dynamo unbounded loop unrolling: https://github.com/pytorch/pytorch/issues/102839
- pytorch#96064, wrong answer on unrolled loops: https://github.com/pytorch/pytorch/issues/96064
- PyTorch forum, compiling an RNN loop once: https://discuss.pytorch.org/t/how-to-compile-a-rnn-loop-once/191455
- PyTorch forum, `torch.compile` + CUDA graphs + activation checkpointing: https://discuss.pytorch.org/t/how-to-use-torch-compile-with-cuda-graphs-when-using-gradient-activation-checkpointing/179466
- KernelBench, compile overhead on small kernels: https://arxiv.org/pdf/2502.10517
