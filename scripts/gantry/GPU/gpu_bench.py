"""GPU benchmark for the gantry augmentation rollout (D-169). Throwaway diagnostic.

WHY THIS EXISTS. The pipeline is DISPATCH-bound, not FLOP-bound: measured 8.8 us per tensor op,
with ~62 of ~254 ops per timestep being matmuls. Wall time obeys

    T = U * nf * t_step        U = batch updates,  t_step independent of batch size

so the only things that matter are the update count and t_step. This script measures t_step, its
(non-)dependence on batch, and whether `inductor` -- the only torch.compile backend that FUSES
kernels, and therefore the only one that can reduce the op count -- moves it.

`inductor` cannot be tested on the Windows development PC (needs MSVC/cl.exe; Triton has no
Windows support), which is the entire reason this has to run on the cluster. Measured locally on
a Quadro P2000 for reference:

    eager                       20.95 ms/step   (batch-flat from 128 to 8192, 64x range)
    aot_eager  fullgraph=True   43.66 ms/step   0.48x   no fusion, pure tracing overhead
    cudagraphs fullgraph=True 1208.42 ms/step   0.02x   wrong granularity for a per-step call
    inductor                    NOT TESTABLE ON WINDOWS

Run:
    python scripts/gantry/gpu_bench.py
Nothing is trained and nothing is written; it prints a report.
"""
__project_origin__ = "added"

import os
import sys
import time
from dataclasses import replace

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
# This file lives in scripts/gantry/GPU/, so the pipeline package (scripts/gantry) and the repo
# root (for model_augmentation) are one and three levels up respectively.
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..')))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..', '..')))

from gantry_dynamic.data import load_datasets, compute_normalization, VAL_FILES, TRAIN_FILES  # noqa: E402
from gantry_dynamic.model import build_model                                    # noqa: E402
from gantry_dynamic.controller import build_closed_loop                         # noqa: E402
from model_augmentation.fit_systems.closed_loop import closed_loop_rollout      # noqa: E402
import gantry_interconnect_dynamic as entry                                     # noqa: E402

NF = int(os.environ.get('BENCH_NF', 200))          # op counts and t_step are nf-independent
CHUNK = int(os.environ.get('BENCH_CHUNK', 50))     # gradient-checkpoint segment
BATCHES = [int(b) for b in os.environ.get('BENCH_BATCHES', '256,1024,4096').split(',')]
REPS = int(os.environ.get('BENCH_REPS', 2))
TARGET_NF, TARGET_EPOCHS = 12000, 5


def banner(s):
    print(f"\n{'=' * 74}\n{s}\n{'=' * 74}", flush=True)


banner("ENVIRONMENT")
print(f"  torch {torch.__version__}   CUDA build {torch.version.cuda}")
if not torch.cuda.is_available():
    print("  no CUDA device -- this benchmark needs a GPU (sbatch --gres=gpu:1)")
    sys.exit(1)
p = torch.cuda.get_device_properties(0)
print(f"  GPU: {p.name}  {p.total_memory / 2**30:.1f} GB  sm_{p.major}{p.minor}")
try:
    import triton
    print(f"  triton {triton.__version__}  -> inductor CAN generate GPU kernels")
    HAVE_TRITON = True
except Exception as e:
    print(f"  triton MISSING ({type(e).__name__}) -> inductor will fall back or fail")
    HAVE_TRITON = False
print(f"  nf={NF}  chunk={CHUNK}  batches={BATCHES}  reps={REPS}")

banner("BUILD")
torch.manual_seed(42)
np.random.seed(42)
cfg = replace(entry.CFG, orth=False, device='cuda', nf_override=NF, stride=20,
              checkpoint_chunk=CHUNK, save_flag=False)
data = load_datasets(cfg)
norm = compute_normalization(cfg, data)
fs = build_model(cfg.hp, cfg, data, norm)
fs.simulator = build_closed_loop(fs, norm, cfg, train_files=TRAIN_FILES, val_files=VAL_FILES,
                                 val_data=data.val_ckpt_data, verbose=False)
bank = fs.simulator.bank
arrs = fs.make_training_data(fs.norm.transform(data.train_data), nf=NF, stride=cfg.stride)
N_WIN = len(arrs[0])
print(f"  windows={N_WIN}  params={sum(q.numel() for q in fs.hfn.parameters())}")


def batch_of(B):
    ix = np.random.default_rng(0).choice(N_WIN, size=B, replace=B > N_WIN)
    out = []
    for a in arrs:
        t = torch.as_tensor(np.ascontiguousarray(np.asarray(a)[ix]))
        out.append(t.to('cuda', cfg.dtype_pt) if t.is_floating_point() else t.to('cuda'))
    return out


def step(b, hfn, out_fn):
    uh, yh, uf, yf, cix = b
    for q in fs.hfn.parameters():
        q.grad = None
    x = fs.encoder(uh, yh)
    y, _, _ = closed_loop_rollout(hfn, out_fn, uf, yf, x, bank, cix.long(), chunk=CHUNK)
    L = torch.nn.functional.mse_loss(yf, y)
    L.backward()
    g = torch.cat([q.grad.reshape(-1) for q in fs.hfn.parameters() if q.grad is not None])
    return float(L), y.detach(), g.detach()


def timed(b, hfn, out_fn, reps=REPS):
    step(b, hfn, out_fn)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(reps):
        step(b, hfn, out_fn)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps / NF, torch.cuda.max_memory_allocated() / 2**20


# ── graph structure: must be graphs=1 / breaks=0, or no backend can help ──────
banner("DYNAMO GRAPH STRUCTURE")
b0 = batch_of(min(256, N_WIN))
with torch.no_grad():
    fs.hfn(b0[0].new_zeros(b0[0].shape[0], fs.hfn.nx), b0[2][:, 0].contiguous())
torch._dynamo.reset()
ex = torch._dynamo.explain(fs.hfn)(b0[0].new_zeros(b0[0].shape[0], fs.hfn.nx),
                                   b0[2][:, 0].contiguous())
print(f"  Interconnect.forward: graphs={ex.graph_count} breaks={ex.graph_break_count} "
      f"ops={ex.op_count}   (expected 1 / 0 after the D-169 int() fix)")
for i, r in enumerate(ex.break_reasons[:5]):
    print(f"      break {i + 1}: {str(getattr(r, 'reason', r))[:90]}")

# ── eager baseline + batch flatness ──────────────────────────────────────────
banner("EAGER: t_step VS BATCH (is the batch axis free on this card?)")
print(f"  {'batch':>7} {'t_step':>11} {'per-window':>12} {'peak MB':>9} {'flatness':>9}")
eager, ref, ref_out = {}, None, {}
for B in BATCHES:
    try:
        b = batch_of(B)
        L, y, g = step(b, fs.hfn, fs.hfn.output_only)
        ts, mb = timed(b, fs.hfn, fs.hfn.output_only)
    except torch.cuda.OutOfMemoryError:
        print(f"  {B:>7}   OOM")
        torch.cuda.empty_cache()
        continue
    ref = ts if ref is None else ref
    eager[B], ref_out[B] = ts, (y, g)
    print(f"  {B:>7} {ts * 1e3:10.3f} ms {ts / B * 1e6:10.2f} us {mb:8.1f} {ts / ref:8.2f}x",
          flush=True)
    torch.cuda.empty_cache()

if not eager:
    print("\nno eager measurement succeeded; aborting")
    sys.exit(1)
B_REF = max(eager)

# ── the point of the exercise: does inductor fuse the op count down? ─────────
banner("torch.compile ON THE STEP BODY (loop left intact: Dynamo unrolls loops)")
CONFIGS = [("inductor", dict(backend='inductor', fullgraph=True)),
           ("inductor reduce-overhead", dict(backend='inductor', mode='reduce-overhead',
                                             fullgraph=True)),
           ("aot_eager (control)", dict(backend='aot_eager', fullgraph=True))]
results = {}
b = batch_of(B_REF)
y_ref, g_ref = ref_out[B_REF]
for tag, kw in CONFIGS:
    torch._dynamo.reset()
    t0 = time.perf_counter()
    try:
        hfn_c = torch.compile(fs.hfn, **kw)
        out_c = torch.compile(fs.hfn.output_only, **kw)
        L, y, g = step(b, hfn_c, out_c)
        torch.cuda.synchronize()
        t_comp = time.perf_counter() - t0
        ts, mb = timed(b, hfn_c, out_c)
    except Exception as e:
        print(f"  {tag:<26} FAILED {type(e).__name__}: {str(e)[:110]}", flush=True)
        torch.cuda.empty_cache()
        continue
    results[tag] = ts
    print(f"  {tag:<26} t_step={ts * 1e3:9.3f} ms  {eager[B_REF] / ts:5.2f}x  "
          f"peak={mb:7.1f} MB  compile={t_comp:5.1f} s  "
          f"max|dy|={float((y - y_ref).abs().max()):.1e} "
          f"max|dg|={float((g - g_ref).abs().max()):.1e}", flush=True)
    torch.cuda.empty_cache()

# ── what it means for the real run ───────────────────────────────────────────
banner(f"PROJECTION: nf={TARGET_NF}, {TARGET_EPOCHS} epochs")
best_tag, best_ts = 'eager', eager[B_REF]
for tag, ts in results.items():
    if ts < best_ts:
        best_tag, best_ts = tag, ts
print(f"  best t_step = {best_ts * 1e3:.3f} ms  ({best_tag}) at batch {B_REF}")
print(f"  one update at nf={TARGET_NF} costs {TARGET_NF * best_ts / 60:.1f} min")
print(f"  a 10 h wall therefore buys {int(10 * 3600 / (TARGET_NF * best_ts))} updates "
      f"(batch size does NOT change this; it changes how much data each update sees)\n")
print(f"  {'stride':>7} {'windows':>9} {'updates/5ep':>12} {'hours':>8}")
for stride in (1, 10, 20, 35):
    wins = ((48000 - TARGET_NF - 17) // stride) * 14
    upd = TARGET_EPOCHS * wins // B_REF
    hrs = upd * TARGET_NF * best_ts / 3600
    print(f"  {stride:>7} {wins:>9} {upd:>12} {hrs:>7.1f} h  {'FITS 10h' if hrs <= 10 else ''}")
print("\ngpu_bench complete")
