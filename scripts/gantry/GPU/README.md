# GPU routing for the gantry augmentation rollout: measurements and justification

Every number here was measured on 2026-09-01, on `oahu` (blade1, RTX 2080 Ti, sm_75) or on the
development PC (Quadro P2000, sm_61). SLURM job IDs are given so each claim is traceable to a log.
Anything still unmeasured says so.

**Question.** `nf_seconds = 0.100` (nf = 400) was the training horizon. Is nf = 12000 (3 s)
trainable, and what GPU configuration makes it so?

**Answer.** Yes, but at a smaller margin than the first estimate. Measured **at** nf = 12000
(job 80712), not extrapolated to it:

| mode | `t_step` | one-off compile | updates in 10 h |
|---|---|---|---|
| eager | 17.190 ms | none | 174 |
| **`inductor`** | **6.428 ms** | 88 s | **466** |
| `reduce-overhead` | unmeasured | **> 50 min, did not finish** | unknown |

Sections 2 to 10 were measured at nf = 200 and are correct there. Sections 13 to 16 are the
2026-09-01 evening results at the real horizon and in the real `fit()`, and they **supersede** the
extrapolated headline this file previously carried (~1300 updates). Read 13 to 16 first if you are
about to launch a run.

---

## 1. Recommended configuration, and the evidence for each field

| Setting | Value | Evidence |
|---|---|---|
| `device` | `'cuda'` | Sect. 4. Eager GPU is *slower* than CPU; the GPU only pays once compiled |
| `use_f64` | `True` at nf = 12000, **`False` at nf = 400** | Sect. 8 for the cost (0-4%). At nf = 400 the trained residual sits ~280x above the float32 floor (Sect. 16), so it buys nothing there |
| `checkpoint_chunk` | **`0`** (off) | Sect. 5. 1.8x faster, and ~3.5 GB at nf = 12000 fits the 11 GB card |
| `batch_size` | `512` | Sect. 3. `t_step` flat across a 64x batch range |
| `stride` | `10` | 50,372 windows; every sample still appears in 1200 windows at nf = 12000 |
| `compile_mode` | **`'reduce-overhead'` at nf = 400, `'inductor'` at nf = 12000** | Sect. 6 for the ranking; Sect. 14 for why the long horizon differs: CUDA-graph capture scales steeply and did not finish in 50 min at nf = 12000 |
| partition | `oahu` (RTX 2080 Ti) | Sect. 4. A100 measured irrelevant |

Total speedup from where this started (benchmark, nf = 200):

```
27.5 ms    eager, chunk=50          (the starting point)
15.0 ms    eager, chunk=0            1.8x   checkpointing off
 2.12 ms   + inductor reduce-overhead, float64
─────────────────────────────────────────────
~13x
```

And end to end against the CPU, which is the number that matters. Same data, same objective, same
entry point, per **gradient update**:

| model | CPU (jobs 80498-80557) | GPU | per update | per sample |
|---|---|---|---|---|
| `nx_ann=2`, 16x2 | 4.89 s @ batch 256 | 0.50 s @ batch 512 (80713) | **9.8x** | 19.6x |
| `nx_ann=8`, 24x3 | 5.00 s @ batch 256 | 0.97 s @ batch 512 (80717) | **5.2x** | 10.3x |

Note the asymmetry: on the CPU the two models cost the same (4.89 vs 5.00 s), on the GPU the larger
one costs 1.9x. Both are dispatch-bound, but the CPU's per-op cost dominates the op *count* while
the GPU's does not, so capacity is nearly free on the CPU and is not free on the GPU.

---

## 2. The bottleneck: host dispatch, not the GPU

Profiled on the P2000 (nf = 50, batch 8, `torch.profiler` with CUDA activity):

```
2404 tensor operations per timestep, of which 62 are matrix multiplies
21.076 ms / 2404 ops = 8.8 us per op        <- kernel-launch latency
```

97% of operations are allocation, striding, slicing, transposing. Per timestep: `as_strided` 340,
`zeros`+`empty`+`fill_`+`zero_` 256, `mul` 127, `slice` 113, `transpose` 83, `mm` 62.

Arithmetic per update is ~1.2 GFLOP including backward, about 0.1 ms of GPU work against a measured
4-6 s. The device is idle almost the whole time.

---

## 3. Batch size is free (the load-bearing measurement)

**P2000**, nf = 100, chunk 50:

| batch | `t_step` | per window | peak MB |
|---|---|---|---|
| 128 | 34.660 ms | 270.78 us | 22.1 |
| 512 | 33.761 ms | 65.94 us | 36.0 |
| 2048 | 33.187 ms | 16.20 us | 91.3 |
| 8192 | 31.963 ms | **3.90 us** | 313.8 |

**2080 Ti (job 80606)**, nf = 200, chunk 50:

| batch | `t_step` | per window |
|---|---|---|
| 256 | 27.963 ms | 109.23 us |
| 1024 | 27.968 ms | 27.31 us |
| 4096 | 27.971 ms | 6.83 us |

A **64x** batch increase changes `t_step` by under 3% on the P2000 and by **0.03%** on the 2080 Ti.
Per-window cost falls 69x. This is what makes a long horizon affordable, and no CPU can do it.

---

## 4. Hardware: the card does not matter, the host does

| Card | clock | FP32 | `t_step` (chunk-adjusted) |
|---|---|---|---|
| Quadro P2000 (sm_61) | ~1480 MHz | ~3 TFLOP | ~27.9 ms |
| RTX 2080 Ti (sm_75) | 2100 MHz | ~13.4 TFLOP | 27.97 ms |

A 4.5x FLOP gap and a 1.4x clock gap produce **0.3%**. `t_step` tracks neither GPU throughput nor
GPU clock; it tracks the **host** issuing operations (blade1: 2x Xeon E5-2650 v4, 48 threads,
2.20 GHz base / 2.90 max).

**Consequences.** The A100 partitions are not worth queuing for: bandwidth, SM count and VRAM
address a bottleneck this model does not have, and their ~1.41 GHz clock is the lowest of the set.
Data-parallel across GPUs is equally useless: batch is already free (Sect. 3) and every rank would
still walk all 12,000 sequential steps.

**Toolchain.** `inductor` was untestable on the development PC for two independent reasons: Windows
lacks MSVC, and Triton needs CUDA capability >= 7.0 while the P2000 is 6.1. The server has gcc 9.4.0
and triton 3.1.0.

---

## 5. Gradient checkpointing: correct, but turned off

`closed_loop_rollout` gained a `chunk` argument (`model_augmentation/fit_systems/closed_loop.py`).

**It is exact.** At chunk 1, 7, 20, 40, nf and nf+50: `max|dy| = 0`, `max|dloss| = 0`,
`max|dgrad| = 0`, `cos = 1.000000000000000`. Segment boundaries are **not** detached and
`dLoss/dx0` is nonzero over the full horizon, so the gradient still spans all nf steps. That is the
difference from truncated BPTT, which detaches and changes the objective.

**It works.** At nf = 4000 the tensors saved for backward drop from **20,010 to 34**, gradient norm
identical to ten digits.

**But it is not needed at nf = 12000**, and it is expensive:

| chunk | batch 512 `t_step` | measured across 3 runs |
|---|---|---|
| 0 (off) | **14.6 - 15.1 ms** | 14.611 / 14.997 / 15.075 |
| 50 | 26.8 - 27.5 ms | 26.782 / 27.273 / 27.459 |

**1.8x**, not the +33% pure recomputation predicts; the rest is per-segment bookkeeping in
`torch.utils.checkpoint`. At ~590 bytes/sample/step with chunking off, batch 512 projects to
**~3.5 GB** at nf = 12000, inside 11 GB. Keep the feature for horizons where memory does bind.

---

## 6. Compilation: the 6-7x

`Interconnect.forward` was **not compilable**: `graphs=12, breaks=11`. Causes, from
`torch._dynamo.explain`:

- ~5 breaks: `generic_jump NumpyNdarrayVariable` at `interconnect.py:91`. Root cause at `:160-163`,
  where `order_output_signal_computation` was filled from `np.argwhere(...).flatten()`, so the main
  forward loop branched on a **numpy scalar**.
- 2 breaks: "Dynamic control flow" at `:81-85`, the per-call device/dtype check.

**Fix: `append(int(element))`.** One word. Result: **`graphs=1, breaks=0, ops=254`**, on CPU and
CUDA. Proven inert by re-running with the list cast back to `numpy.int64`: loss and gradient norm
**bit-identical to 17 digits**, `max|dy| = 0`, `max|dg| = 0`.

Backends at batch 512, chunk 0 (each row against its own run's eager reference):

| backend | 80610 | 80634 | 80652 | accuracy vs eager |
|---|---|---|---|---|
| eager | 15.075 | 14.611 | 14.997 ms | - |
| `aot_eager` | 0.52x (local) | | | bit-identical |
| `inductor` | 5.819 (2.59x) | 5.446 (2.68x) | 6.026 (2.49x) | `max|dy|=9.5e-07`, `max|dg|=7.5e-10` |
| **`reduce-overhead`** | 2.487 (6.06x) | **2.129 (6.86x)** | 2.400 (6.25x) | same |
| `max-autotune` | 2.235 | 2.150 | 2.452 | `max|dy|=7.2e-07`, `max|dg|=4.7e-10` |

Only `inductor` helps: it is the sole backend that **fuses kernels** and so reduces the op count.
`aot_eager` traces without codegen (pure overhead). `cudagraphs` alone was 0.02x, because per-step
capture is the wrong granularity for a callable invoked 12,000 times in a loop.

`reduce-overhead` = inductor **plus** CUDA graphs, i.e. dispatch collapsed into graph replay, which
is why it is 2.5x better than inductor alone. It requires
`torch.compiler.cudagraph_mark_step_begin()` before every invocation and `.clone()` on retained
outputs; without those it fails with "accessing tensor output of CUDAGraphs that has been
overwritten by a subsequent run".

**`max-autotune` is settled and rejected.** Re-measured in job 80652 with the CUDA-graph flag fixed:
2.452 ms against `reduce-overhead`'s 2.400 in float32, and clearly worse in float64 (2.307 vs
2.120). Slower to compile, no faster to run.

**Run-to-run variance is +/-8% on the compiled path** (2.129 to 2.487 ms across three runs) against
+/-2% for eager. CUDA-graph replay is nearly pure host dispatch and therefore sensitive to CPU
contention on a shared node; `nvidia-smi` reported two GPUs on some runs. Quote **~1300 +/- 100
updates**, never a single run's best.

---

## 7. Compilation does not change where training goes (job 80614)

40 optimiser steps per arm, identical parameters, identical batches, nf = 200, batch 512,
`reduce-overhead`:

```
after     cos(dtheta)      |dtheta| ratio
    1   1.000000000000       1.000000000
   10   0.999999940395       0.999995232
   40   1.000000000000       0.999996483

parameter separation / |dtheta| = 1.616e-04
loss  eager 1.320568e-08 -> 6.322958e-09
   compiled 1.320569e-08 -> 6.322956e-09
```

Reference: the float32-vs-float64 result this project already **accepted** measured
`cos = 0.999042`, `ratio = 1.0048`. Compilation is ~1000x tighter, and the agreement does **not
decay with N**, which was the failure mode worth testing for.

---

## 8. Precision: the float32 floor, and why float64 wins

The floor is `|y_f32 - y_f64|` for the same rollout: the width of float32's uncertainty about its
own answer. No amount of training produces a residual below it.

**Job 80617**, batch 64:

| nf | max\|y32-y64\| | rms | floor/eps | t32 | t64 |
|---|---|---|---|---|---|
| 200 | 1.1426e-06 | 1.2763e-07 | 1.1 | 1.85 s | 1.14 s |
| 1000 | 1.1233e-06 | 1.1511e-07 | 1.0 | 5.22 s | 5.16 s |
| 4000 | 1.2019e-06 | 1.1307e-07 | 0.9 | 20.82 s | 20.55 s |
| 12000 | 1.2830e-06 | 1.1487e-07 | 1.0 | 62.05 s | 61.85 s |

**The floor is exactly 1 float32 eps and does not grow: 0.9x over a 60x longer rollout**, where a
random walk predicts 7.7x. The closed loop **contracts** rounding perturbations as fast as they are
created. That is a stability property of the controlled system, worth stating in the thesis in its
own right, and it means the 30x horizon increase costs nothing in conditioning.

**In metres.** Measured `ystd = [0.03233, 0.03233, 0.19013] m`, so the floor is:

| channel | floor |
|---|---|
| X, Theta | 3.71e-09 m |
| **Y** | **2.18e-08 m** |

Y binds: its `ystd` is 6x the others, so its absolute floor is 6x coarser. Against a **trained** RMS
of ~1e-07 m:

| target | vs X/Theta floor | vs Y floor |
|---|---|---|
| 1e-06 m | 270x | 46x |
| **1e-07 m** | 27x | **4.6x** |
| 1e-08 m | 2.7x | **0.5x, below the floor** |

**float64 is free, even compiled (job 80652, Part 3, batch 512, chunk 0):**

| config | f64 | f32 | ratio |
|---|---|---|---|
| eager | 16.605 ms | 14.997 ms | 1.11x |
| `inductor` | 6.238 ms | 6.026 ms | 1.04x |
| `reduce-overhead` | **2.120 ms** | 2.400 ms | 0.88x |
| `max-autotune` | 2.307 ms | 2.452 ms | 0.94x |

**The 0.88x is not a speedup.** This run's float32 sample (2.400 ms) was the slowest of three, so
the ratio is flattered by its denominator; against the best float32 sample it is 1.00x. The honest
statement is **float64 costs 0 to 4%**. The 2080 Ti runs FP64 at 1/32 rate, but that rate applies to
arithmetic throughput which is not the bottleneck (Sect. 2), and even compiled the kernels stay
latency-bound.

**Decision: `use_f64 = True`.** Up to 4% for a floor that drops from ~1.15e-07 to ~1e-16
normalised. The Y-channel margin question disappears and 1e-08 m becomes reachable.

---

## 9. Corrections to figures quoted before they were measured

| Claimed | Measured | Factor |
|---|---|---|
| ~140 ops/timestep | 2404 | 17x |
| `t_step` ~3.4 ms | 15.0 ms eager, 2.12 ms compiled | ~4x |
| "nf = 12000 in ~7 h" | 28-35 h eager; ~10 h compiled | 4-5x |
| window arrays "145 TB" | 145 GB | 1000x |
| 4 kB/sample/step | ~590 B | 7x, in our favour |
| "compile touches nothing Jan owns" | blocked by 11 breaks inside `Interconnect` | - |
| "clock is the relevant spec" | host dispatch is; clock is not | - |
| "FP64 catastrophic (1/32 rate)" | free, 0-4% | - |
| "958x headroom to the floor" | that used the UNTRAINED residual; trained is 4.6x on Y | 200x |
| `max-autotune` beats `reduce-overhead` | ties in f32, worse in f64, 10x slower to compile | - |
| float64 "0.88x, faster than float32" | noise; the f32 denominator was the slowest of three | - |
| "~1300 updates in 10 h at nf = 12000" | 466 with `inductor`; `reduce-overhead` unmeasured there | 2.8x |
| 2.12 ms/step is the end-to-end update cost | that was `closed_loop_rollout` alone; the real update adds the optimiser step, param loss, statistics and the H2D batch move | - |
| "0.55-0.65 s/update at `nx_ann=8`" | 0.97 s (80717); the extra layer costs ~1.9x, not 10-25% | 1.6x |
| "validation is ~14 min on the GPU" | ~162 s; the 18:32 to the first validation was the window build, not validation | 5x |
| "eager CUDA validation will be slower than CPU" | 2x faster: the free run is batched over the 4 records | - |

---

## 10. Pipeline changes, and their verification

| File | Change | Verified by |
|---|---|---|
| `fit_systems/interconnect.py` | `append(int(element))` — breaks 11 → 0 | bit-identical loss + gradient to 17 digits |
| " | `_apply` override + placement at `init_forward`, so the connection matrices follow the module's device/dtype | `.cuda()`, `.cpu()`, `.double()`, lazy-init-on-CUDA all confirmed |
| `fit_systems/closed_loop.py` | `closed_loop_rollout(..., chunk=)` | bit-exact at every chunk size |
| " | bank re-homed to the data's device per rollout | would otherwise have crashed on the first GPU batch: `ClosedLoopSimulator` is a plain object, so the bank is not a submodule and did not move with `fit_sys.cuda()` |
| " | `closed_loop_free_run_rms_batch`, one rollout for all records | 3.15x; agreement 2.6e-10 (BLAS reassociation, not bit-identical) |
| " | simulator holds the compiled pair; routing guard; `cudagraph_mark_step_begin()`; output clone; recompile detector | `compiled=None` an exact no-op; routing verified training→compiled, `_NfProbe`/validation→eager |
| `gantry_dynamic/config.py` | `device`, `checkpoint_chunk`, `compile_mode` | validators tested; all three in `config.json` |
| `gantry_dynamic/model.py` | device into `init_model`, `cuda=` into `fit`, `COMPILE_BACKEND` | |
| `gantry_dynamic/controller.py` | compiles both callables, passes them to the simulator | |
| `gantry_interconnect_dynamic.py` | the three fields + `EXEC:` banner | |

**`fit_sys.hfn` is never replaced.** The compiled pair lives on the simulator, so
`closed_loop_free_run_rms_batch` and the ~20 diagnostics stay eager by construction. That mattered
originally because deepSI's `fit()` moved the model to the CPU around every validation and a
compiled callable would have recompiled per device, twice an epoch. `fit()` no longer flips
(Sect. 13), but the split is still what keeps validation off the compiled artefact.

**Compiling both callables is required.** `torch.compile(module)` returns an `OptimizedModule` that
proxies attribute access, so `.output_only` on it is still the *uncompiled* bound method; compiling
only the module leaves the rollout half compiled.

---

## 11. Open items

Closed on 2026-09-01 (evening): `t_step` and memory at nf = 12000 (Sect. 14), end-to-end wiring
with compilation (Sect. 13), `flip_safety` (Sect. 13).

Still open:

1. **`reduce-overhead` at nf = 12000.** Capture did not finish in 50 min (Sect. 14). Unknown
   whether it completes at all, and what it would give. Needs a 6 h wall to settle. Until then the
   long-horizon recommendation is plain `inductor`.
2. **`lr` at batch 512.** `3e-5` is interpolated between a stable `1e-5` and an oscillating `1e-4`
   measured at batch 256 (Sect. 16). It is the only unmeasured number in the launch config.
3. **Capacity.** `nx_ann=8`, 24x3 rests on one paired comparison that moves three settings at once
   (Sect. 16), and it costs 1.9x per update on the GPU. Worth an A/B at the rate actually used.
4. **`main()`'s post-training block on CUDA.** The four baseline sims and per-record NRMS have
   never run with `device='cuda'`. The CPU restore (Sect. 15) exists for it but is untested there.
5. **Window build time and host RAM at nf = 12000.** `make_training_data` takes the pure-Python
   `stride != 1` path (`system_data.py:316-330`) and holds the list *and* the output array at peak.
   At stride 10 that projects to ~36 GB and ~72 GB transiently. Raise `stride` with `nf`
   (80 gives ~4.5 GB); windows overlap 99.9% at that horizon anyway. Unmeasured.

---

## 12. The scripts

| File | What it measures | Jobs |
|---|---|---|
| `gpu_bench.py` / `.sh` | graph structure, batch flatness, first inductor test | 80606 |
| `gpu_bench2.py` / `.sh` | checkpointing on/off, all inductor modes, float64 arm | 80610, 80634, 80652 |
| `compile_update_precision.py` / `.sh` | does compilation change the training trajectory | 80614 |
| `float32_floor.py` / `.sh` | the arithmetic floor vs horizon | 80617 |
| `flip_safety.py` / `.sh` | does the per-validation device flip corrupt compiled state | 80688, 80695 |
| `wiring_e2e.py` / `.sh` | the real `train_model` -> `fit()` path with `n_its` capped: does compilation survive an actual run | 80706, 80708, 80710, 80713 |
| `gantry_interconnect_dynamic_gpu.sh` | not a diagnostic: the GPU launcher for the full entry point | 80717 |

All are diagnostics: nothing is trained, nothing is written. Knobs are environment variables
(`BENCH_*`, `PREC_*`, `FLOOR_*`, `FLIP_*`), so no edit is needed to re-run a variant.

**A consistency check worth recording.** `gpu_bench2` reached the CUDA-graph fast path on the same
pre-`_apply` code where `flip_safety` did not. The reason: `gpu_bench2` runs its eager Part 1 first,
and `Interconnect.forward`'s device check re-homes the matrices to CUDA during that pass, so by the
time Part 2 compiles no CPU tensor can enter the graph. `flip_safety` compiles *after* a CPU
validation, which puts them back. One root cause explains both the failure and the non-failure, and
the `_apply` fix makes the ordering irrelevant. The benchmark numbers were never wrong; they were
measured in the lucky ordering.
