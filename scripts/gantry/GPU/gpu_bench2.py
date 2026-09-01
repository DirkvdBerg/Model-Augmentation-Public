"""GPU benchmark round 2 (D-169): how many UPDATES can one nf=12000 run afford?

Round 1 (job 80606, blade1 RTX 2080 Ti) established:
    graphs=1 breaks=0 ops=254          after the int(element) fix
    eager                27.971 ms/step, FLAT from batch 256 to 4096 (0.03%)
    inductor             10.453 ms/step  2.68x   max|dy|=9.5e-07 max|dg|=2.1e-10
    inductor reduce-ovh  FAILED "accessing tensor output of CUDAGraphs that has been
                                overwritten by a subsequent run"
    aot_eager            53.862 ms/step  0.52x   bit-identical (control)

That FAILURE is a bug in round 1's harness, not in the model: it retained y and g from one
iteration while starting the next, which is precisely what the message describes.
`reduce-overhead` IS inductor + CUDA graphs, i.e. the ~30-dispatches-to-one-replay step, so it
is the single biggest untested lever. This script tests it properly:
    - torch.compiler.cudagraph_mark_step_begin() before every invocation
    - every retained output .clone()d out of the graph's memory pool

It also tests CHECKPOINTING OFF. Checkpointing costs exactly one extra forward (+33%, confirmed:
20.95 ms local without vs 27.97 ms server with). At nf=12000 it is only needed if the graph does
not fit, so this measures the memory ceiling directly.

The metric is UPDATES IN A 10 h WALL, because T = U * nf * t_step and the goal is to train
nf=12000 properly, not merely to run it. Reference: the nf=400 pipeline trains at 1300 updates.

Run:  python scripts/gantry/GPU/gpu_bench2.py
"""
__project_origin__ = "added"

import os
import sys
import time
from dataclasses import replace

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..')))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..', '..')))

from gantry_dynamic.data import load_datasets, compute_normalization, VAL_FILES, TRAIN_FILES  # noqa: E402
from gantry_dynamic.model import build_model                                # noqa: E402
from gantry_dynamic.controller import build_closed_loop                     # noqa: E402
from model_augmentation.fit_systems.closed_loop import closed_loop_rollout  # noqa: E402
import gantry_interconnect_dynamic as entry                                 # noqa: E402

NF = int(os.environ.get('BENCH_NF', 200))
BATCHES = [int(b) for b in os.environ.get('BENCH_BATCHES', '256,512,1024').split(',')]
CHUNKS = [int(c) for c in os.environ.get('BENCH_CHUNKS', '0,50').split(',')]   # 0 = OFF
REPS = int(os.environ.get('BENCH_REPS', 3))
WALL_H = float(os.environ.get('BENCH_WALL_H', 10))
TARGET_NF, BASELINE_UPDATES = 12000, 1300


def banner(s):
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}", flush=True)


banner("ENVIRONMENT")
if not torch.cuda.is_available():
    print("no CUDA device (need sbatch --gres=gpu:1)")
    sys.exit(1)
p = torch.cuda.get_device_properties(0)
print(f"  torch {torch.__version__}  cuda {torch.version.cuda}")
print(f"  GPU {p.name}  {p.total_memory / 2**30:.1f} GB  sm_{p.major}{p.minor}")
print(f"  nf={NF}  batches={BATCHES}  chunks={CHUNKS} (0=checkpointing OFF)  reps={REPS}")

torch.manual_seed(42)
np.random.seed(42)
cfg = replace(entry.CFG, orth=False, device='cuda', nf_override=NF, stride=20, save_flag=False)
data = load_datasets(cfg)
norm = compute_normalization(cfg, data)
fs = build_model(cfg.hp, cfg, data, norm)
fs.simulator = build_closed_loop(fs, norm, cfg, train_files=TRAIN_FILES, val_files=VAL_FILES,
                                 val_data=data.val_ckpt_data, verbose=False)
bank = fs.simulator.bank
arrs = fs.make_training_data(fs.norm.transform(data.train_data), nf=NF, stride=cfg.stride)
N_WIN = len(arrs[0])
print(f"  windows={N_WIN}")

_cache = {}


def batch_of(B):
    if B not in _cache:
        ix = np.random.default_rng(0).choice(N_WIN, size=B, replace=B > N_WIN)
        out = []
        for a in arrs:
            t = torch.as_tensor(np.ascontiguousarray(np.asarray(a)[ix]))
            out.append(t.to('cuda', cfg.dtype_pt) if t.is_floating_point() else t.to('cuda'))
        _cache[B] = out
    return _cache[B]


def step(b, hfn, out_fn, chunk, graphs):
    """One training step. `graphs` -> the CUDA-graph discipline round 1 omitted."""
    if graphs:
        # Round 1's failure: without this, a retained output from the previous iteration is
        # still pointing into the graph's memory pool when the next run overwrites it.
        torch.compiler.cudagraph_mark_step_begin()
    uh, yh, uf, yf, cix = b
    for q in fs.hfn.parameters():
        q.grad = None
    x = fs.encoder(uh, yh)
    y, _, _ = closed_loop_rollout(hfn, out_fn, uf, yf, x, bank, cix.long(), chunk=chunk)
    L = torch.nn.functional.mse_loss(yf, y)
    L.backward()
    g = torch.cat([q.grad.reshape(-1) for q in fs.hfn.parameters() if q.grad is not None])
    # .clone() lifts anything we keep OUT of the graph's pool, so holding it across the next
    # invocation is safe. This is the other half of the round-1 fix.
    return float(L), y.detach().clone(), g.detach().clone()


def timed(b, hfn, out_fn, chunk, graphs, reps=REPS):
    step(b, hfn, out_fn, chunk, graphs)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(reps):
        step(b, hfn, out_fn, chunk, graphs)
    torch.cuda.synchronize()
    return ((time.perf_counter() - t0) / reps / NF,
            torch.cuda.max_memory_allocated() / 2**20)


def truncate(b, n):
    """The same batch shortened in TIME. uhist/yhist are encoder windows and are left alone."""
    uh, yh, uf, yf, cix = b
    return uh, yh, uf[:, :n].contiguous(), yf[:, :n].contiguous(), cix


def peak_for(b, chunk):
    """Peak CUDA memory of one training step on this batch. Memory only, no timing."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    step(b, fs.hfn, fs.hfn.output_only, chunk, graphs=False)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**20


def updates_in_wall(ts):
    return int(WALL_H * 3600 / (TARGET_NF * ts))


def mem_at_target(peak_lo, peak_hi, B, chunk):
    """Peak memory at nf=TARGET_NF, from a TWO-POINT measurement.

    Scaling the whole peak by nf is wrong: a large part of it is fixed (model, optimiser, cuBLAS
    workspace) and does not grow with the horizon. Measured at nf=50 the fixed part dominated so
    badly that batch 64 -> 128 moved the peak by only 9%, and a single-point extrapolation
    produced bogus "will not fit" verdicts.

    Two points separate the terms: peak(nf) = fixed + slope*nf, so
        slope = (peak_hi - peak_lo) / (NF - NF//2)
        fixed = peak_lo - slope*(NF//2)
    and the extrapolation uses only the part that actually scales.
    """
    nf_lo, nf_hi = NF // 2, NF
    slope = (peak_hi - peak_lo) / max(nf_hi - nf_lo, 1)
    fixed = peak_lo - slope * nf_lo
    return max(fixed, 0.0) + slope * TARGET_NF, slope / B * 2**20   # MB, bytes/sample/step


banner("PART 1: EAGER, CHECKPOINTING ON vs OFF (does the graph fit without it?)")
print("  memory is extrapolated from TWO nf points, so the fixed part (model, cuBLAS workspace)")
print("  is separated from the part that scales with the horizon.\n")
print(f"  {'chunk':>6} {'batch':>7} {'t_step':>11} {'peak MB':>10} {'B/sample/step':>14} "
      f"{'est MB @nf=12000':>18} {'updates/10h':>12}")
eager = {}
VRAM_MB = p.total_memory / 2**20
for chunk in CHUNKS:
    for B in BATCHES:
        try:
            b = batch_of(B)
            # LOW nf point, for the memory slope only: same batch truncated in TIME, so the
            # fixed term is identical and the difference is purely the per-step activations.
            mb_lo = peak_for(truncate(b, NF // 2), chunk)
            ts, mb = timed(b, fs.hfn, fs.hfn.output_only, chunk, graphs=False)
        except torch.cuda.OutOfMemoryError:
            print(f"  {chunk:>6} {B:>7}   OOM at nf={NF}")
            torch.cuda.empty_cache()
            continue
        eager[(chunk, B)] = ts
        est, bps = mem_at_target(mb_lo, mb, B, chunk)
        flag = '' if est < 0.85 * VRAM_MB else '  <-- exceeds this card'
        print(f"  {chunk:>6} {B:>7} {ts * 1e3:10.3f} ms {mb:9.1f} {bps:13.0f} {est:17.0f} "
              f"{updates_in_wall(ts):>12}{flag}", flush=True)
        torch.cuda.empty_cache()

if not eager:
    print("no eager point succeeded; aborting")
    sys.exit(1)
BEST_CHUNK, BEST_B = min(eager, key=lambda k: eager[k])
print(f"\n  fastest eager: chunk={BEST_CHUNK} batch={BEST_B} -> {eager[(BEST_CHUNK, BEST_B)]*1e3:.3f} ms")

banner("PART 2: INDUCTOR, AND reduce-overhead DONE PROPERLY")
b = batch_of(BEST_B)
_, y_ref, g_ref = step(b, fs.hfn, fs.hfn.output_only, BEST_CHUNK, graphs=False)
base = eager[(BEST_CHUNK, BEST_B)]
print(f"  reference: eager {base * 1e3:.3f} ms  |g|={float(g_ref.norm()):.6e}\n")

# The third field is whether the config uses CUDA graphs and therefore needs
# cudagraph_mark_step_begin() before every invocation.
# CHANGED: max-autotune was flagged False. It enables CUDA graphs internally exactly as
# reduce-overhead does, so without the mark it never reaches the replay fast path and PyTorch
# says so ("Unable to hit fast path of CUDAGraphs because of pending, uninvoked backwards",
# job 80634). Its 2.235 ms in job 80610 was therefore measured HANDICAPPED -- a floor, not its
# speed -- and it still beat reduce-overhead's 2.487 ms. Plain 'inductor' stays False: it does
# no graph capture, so the mark would be meaningless there.
CONFIGS = [
    ("inductor",                dict(backend='inductor', fullgraph=True), False),
    ("inductor reduce-overhead", dict(backend='inductor', mode='reduce-overhead',
                                      fullgraph=True), True),
    ("inductor max-autotune",   dict(backend='inductor', mode='max-autotune',
                                     fullgraph=True), True),
]
res = {}
for tag, kw, graphs in CONFIGS:
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    t0 = time.perf_counter()
    try:
        hfn_c = torch.compile(fs.hfn, **kw)
        out_c = torch.compile(fs.hfn.output_only, **kw)
        _, y, g = step(b, hfn_c, out_c, BEST_CHUNK, graphs)
        torch.cuda.synchronize()
        t_comp = time.perf_counter() - t0
        ts, mb = timed(b, hfn_c, out_c, BEST_CHUNK, graphs)
    except Exception as e:
        print(f"  {tag:<26} FAILED {type(e).__name__}: {str(e)[:130]}", flush=True)
        torch.cuda.empty_cache()
        continue
    res[tag] = ts
    print(f"  {tag:<26} {ts * 1e3:8.3f} ms  {base / ts:5.2f}x  peak={mb:7.1f} MB  "
          f"compile={t_comp:6.1f}s  max|dy|={float((y - y_ref).abs().max()):.1e} "
          f"max|dg|={float((g - g_ref).abs().max()):.1e}  "
          f"updates/10h={updates_in_wall(ts)}", flush=True)
    torch.cuda.empty_cache()

# ── PART 3: is float64 still free once compiled? ─────────────────────────────
# WHY. Job 80617 measured the float32 arithmetic floor at 1 eps, FLAT across a 60x horizon
# (the closed loop contracts rounding rather than accumulating it). Converted to metres with
# the measured ystd = [0.0323, 0.0323, 0.1901] m, that floor is 3.7e-09 m on X/Theta and
# 2.18e-08 m on Y. Against a TRAINED rms of ~1e-07 m, the Y channel therefore sits only ~4.6x
# above the floor, and 1e-08 m is BELOW it. So float64 is a live question, not a formality.
#
# In EAGER, float64 was measured FREE (61.85 s vs 62.05 s at nf=12000, job 80617): the workload
# is launch-bound, so the RTX 2080 Ti's 1/32 FP64 rate applies to arithmetic throughput that is
# not the bottleneck. The open question is whether that survives compilation, which removes most
# of the dispatch overhead and pushes the workload toward compute-bound -- exactly the regime
# where 1/32 bites. If float64 stays free, switch and precision closes permanently. If not, the
# 1e-07 m result is float32-limited on Y and that has to be stated rather than discovered later.
#
# Deliberately reuses BEST_CHUNK / BEST_B from Part 1 so the comparison is like-for-like.
banner("PART 3: FLOAT64 -- free in eager, but is it free COMPILED?")
try:
    torch.manual_seed(42)
    np.random.seed(42)
    cfg64 = replace(entry.CFG, orth=False, device='cuda', nf_override=NF, stride=20,
                    use_f64=True, save_flag=False)
    data64 = load_datasets(cfg64)
    norm64 = compute_normalization(cfg64, data64)
    fs64 = build_model(cfg64.hp, cfg64, data64, norm64)
    fs64.simulator = build_closed_loop(fs64, norm64, cfg64, train_files=TRAIN_FILES,
                                       val_files=VAL_FILES, val_data=data64.val_ckpt_data,
                                       verbose=False)
    bank64 = fs64.simulator.bank
    arrs64 = fs64.make_training_data(fs64.norm.transform(data64.train_data),
                                     nf=NF, stride=cfg64.stride)
    ix64 = np.random.default_rng(0).choice(len(arrs64[0]), size=BEST_B,
                                           replace=BEST_B > len(arrs64[0]))
    b64 = []
    for a in arrs64:
        t = torch.as_tensor(np.ascontiguousarray(np.asarray(a)[ix64]))
        b64.append(t.to('cuda', cfg64.dtype_pt) if t.is_floating_point() else t.to('cuda'))

    def step64(hfn, out_fn, graphs):
        if graphs:
            torch.compiler.cudagraph_mark_step_begin()
        uh, yh, uf, yf, cix = b64
        for q in fs64.hfn.parameters():
            q.grad = None
        x = fs64.encoder(uh, yh)
        y, _, _ = closed_loop_rollout(hfn, out_fn, uf, yf, x, bank64, cix.long(),
                                      chunk=BEST_CHUNK)
        torch.nn.functional.mse_loss(yf, y).backward()
        return y.detach().clone()

    def timed64(hfn, out_fn, graphs, reps=REPS):
        step64(hfn, out_fn, graphs)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(reps):
            step64(hfn, out_fn, graphs)
        torch.cuda.synchronize()
        return ((time.perf_counter() - t0) / reps / NF,
                torch.cuda.max_memory_allocated() / 2**20)

    print(f"  batch={BEST_B} chunk={BEST_CHUNK}  (same as the float32 arms above)\n")
    print(f"  {'config':<28} {'f64 t_step':>12} {'f32 t_step':>12} {'f64/f32':>9} "
          f"{'peak MB':>9} {'updates/10h':>12}")
    ts64_eager, mb = timed64(fs64.hfn, fs64.hfn.output_only, False)
    print(f"  {'eager':<28} {ts64_eager*1e3:11.3f} ms {base*1e3:11.3f} ms "
          f"{ts64_eager/base:8.2f}x {mb:8.1f} {updates_in_wall(ts64_eager):>12}", flush=True)

    res64 = {}
    for tag, kw, graphs in CONFIGS:
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        try:
            h = torch.compile(fs64.hfn, **kw)
            o = torch.compile(fs64.hfn.output_only, **kw)
            step64(h, o, graphs)
            torch.cuda.synchronize()
            ts, mb = timed64(h, o, graphs)
        except Exception as e:
            print(f"  {tag:<28} FAILED {type(e).__name__}: {str(e)[:80]}", flush=True)
            torch.cuda.empty_cache()
            continue
        res64[tag] = ts
        f32 = res.get(tag, base)
        print(f"  {tag:<28} {ts*1e3:11.3f} ms {f32*1e3:11.3f} ms {ts/f32:8.2f}x "
              f"{mb:8.1f} {updates_in_wall(ts):>12}", flush=True)
        torch.cuda.empty_cache()

    if res64:
        b64_tag = min(res64, key=lambda k: res64[k])
        b32_ts = min(res.values()) if res else base
        pen = res64[b64_tag] / b32_ts
        print(f"\n  best float64: {b64_tag} at {res64[b64_tag]*1e3:.3f} ms "
              f"({updates_in_wall(res64[b64_tag])} updates/10h)")
        print(f"  float64 penalty vs best float32: {pen:.2f}x")
        if pen <= 1.15:
            print("  -> float64 is FREE compiled too. Use it: the Y channel stops sitting 4.6x")
            print("     above the arithmetic floor, and precision closes permanently.")
        elif pen <= 2.0:
            print("  -> float64 costs a modest amount. Worth paying if 1e-08 m is a real target;")
            print("     otherwise float32 with the 4.6x Y-channel margin stated explicitly.")
        else:
            print("  -> float64 is EXPENSIVE compiled (dispatch overhead gone -> 1/32 FP64 rate")
            print("     now bites). Either stay float32 and STATE the Y-channel floor margin, or")
            print("     move to lanai/molokai: the A100 runs FP64 at 1/2 rate, 16x better.")
except torch.cuda.OutOfMemoryError:
    print("  OOM building the float64 pipeline (float64 doubles activation memory).")
    torch.cuda.empty_cache()
except Exception as e:
    print(f"  FLOAT64 ARM UNAVAILABLE: {type(e).__name__}: {str(e)[:160]}")
    torch.cuda.empty_cache()

banner(f"VERDICT: can nf={TARGET_NF} be TRAINED (not just run) in {WALL_H:g} h?")
best_tag, best_ts = 'eager', base
for tag, ts in res.items():
    if ts < best_ts:
        best_tag, best_ts = tag, ts
U = updates_in_wall(best_ts)
print(f"  best: {best_tag}  chunk={BEST_CHUNK}  batch={BEST_B}  t_step={best_ts * 1e3:.3f} ms")
print(f"  one update at nf={TARGET_NF}: {TARGET_NF * best_ts:.1f} s")
print(f"  UPDATES IN {WALL_H:g} h: {U}    (nf=400 baseline trains at {BASELINE_UPDATES})")
print(f"  ratio to baseline: {U / BASELINE_UPDATES:.2f}x"
      f"   {'-> a proper training run' if U >= BASELINE_UPDATES else '-> still short; next lever is manual CUDA-graph chunk capture'}")
print(f"\n  {'stride':>7} {'windows':>9} {'epochs at that update count':>30}")
for stride in (1, 5, 10, 20):
    wins = ((48000 - TARGET_NF - 17) // stride) * 14
    print(f"  {stride:>7} {wins:>9} {U * BEST_B / wins:>29.1f}")
print("\ngpu_bench2 complete")
