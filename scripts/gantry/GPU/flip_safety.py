"""Does deepSI's per-epoch CPU/CUDA flip corrupt or slow the compiled training path? (D-169)

THE SITUATION. `fit()` moves the whole model to the CPU before every validation and back
afterwards (interconnect.py:716, :734). With `mode='reduce-overhead'` the training rollout is
backed by CUDAGraph Trees, which hold a GPU memory pool with captured pointers. Flipping devices
underneath that is not a well-trodden path, and this project has already hit the adjacent error
twice ("accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run").

THREE THINGS CAN GO WRONG, and only one of them is loud:
  1. SILENT WRONG NUMBERS - a replay reads a stale pointer and returns garbage. Worst case.
  2. A CRASH - the same fault, caught. This is the good outcome.
  3. SILENT SLOWDOWN - Dynamo recompiles per device, twice an epoch, at 12-150 s each.

WHAT THIS MEASURES. Several cycles of {K updates on CUDA} -> {flip to CPU, rollout, flip back},
in three arms that differ ONLY in how the rollout is driven:

    eager            reference trajectory; no compilation anywhere
    compiled-safe    training compiled, VALIDATION ROUTED THROUGH THE UNCOMPILED hfn
    compiled-naive   training compiled, validation ALSO compiled (the obvious wiring)

Divergence of the parameter trajectory from `eager` detects (1). A raised exception detects (2).
Per-update timing straight after each flip detects (3): a recompile shows as a large spike.

The point is to choose between compiled-safe and compiled-naive with evidence rather than by
argument, before wiring either into train_model.

Run:  python scripts/gantry/GPU/flip_safety.py
Env:  FLIP_NF (200), FLIP_BATCH (256), FLIP_CYCLES (3), FLIP_UPDATES_PER_CYCLE (5),
      FLIP_VAL_STEPS (2000)  -- the CPU rollout length standing in for a full validation
"""
__project_origin__ = "added"

import copy
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

NF = int(os.environ.get('FLIP_NF', 200))
BATCH = int(os.environ.get('FLIP_BATCH', 256))
CYCLES = int(os.environ.get('FLIP_CYCLES', 3))
K = int(os.environ.get('FLIP_UPDATES_PER_CYCLE', 5))
VAL_STEPS = int(os.environ.get('FLIP_VAL_STEPS', 2000))
USE_F64 = os.environ.get('FLIP_F64', '1') not in ('0', 'false', 'False')
COMPILE_KW = dict(backend='inductor', mode='reduce-overhead', fullgraph=True)


def banner(s):
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}", flush=True)


banner("SETUP")
if not torch.cuda.is_available():
    print("no CUDA device (need sbatch --gres=gpu:1)")
    sys.exit(1)
print(f"  GPU {torch.cuda.get_device_properties(0).name}")
print(f"  nf={NF} batch={BATCH} cycles={CYCLES} updates/cycle={K} "
      f"cpu_rollout_steps={VAL_STEPS} use_f64={USE_F64}")
print(f"  compile: {COMPILE_KW}")

torch.manual_seed(42)
np.random.seed(42)
cfg = replace(entry.CFG, orth=False, device='cuda', nf_override=NF, stride=20,
              use_f64=USE_F64, checkpoint_chunk=0, save_flag=False)
data = load_datasets(cfg)
norm = compute_normalization(cfg, data)
fs = build_model(cfg.hp, cfg, data, norm)
fs.simulator = build_closed_loop(fs, norm, cfg, train_files=TRAIN_FILES, val_files=VAL_FILES,
                                 val_data=data.val_ckpt_data, verbose=False)
bank = fs.simulator.bank
arrs = fs.make_training_data(fs.norm.transform(data.train_data), nf=NF, stride=cfg.stride)
N_WIN = len(arrs[0])

_rng = np.random.default_rng(7)
BATCH_IX = [_rng.choice(N_WIN, size=BATCH, replace=BATCH > N_WIN) for _ in range(CYCLES * K)]
INIT = {'hfn': copy.deepcopy(fs.hfn.state_dict()),
        'enc': copy.deepcopy(fs.encoder.state_dict()),
        'opt': copy.deepcopy(fs.optimizer.state_dict())}
TRACKED = [p for g in fs.optimizer.param_groups for p in g['params']]
print(f"  windows={N_WIN}  tracked={sum(p.numel() for p in TRACKED)} scalars")

# One fixed CPU input for the stand-in validation, so every arm validates on identical data.
_vix = _rng.choice(N_WIN, size=4, replace=False)
VAL_CPU = [torch.as_tensor(np.ascontiguousarray(np.asarray(a)[_vix])) for a in arrs]


def to_dev(b, dev):
    return [t.to(dev, cfg.dtype_pt) if t.is_floating_point() else t.to(dev) for t in b]


def batch_at(k, dev='cuda'):
    return to_dev([torch.as_tensor(np.ascontiguousarray(np.asarray(a)[BATCH_IX[k]]))
                   for a in arrs], dev)


def reset():
    fs.hfn.load_state_dict(INIT['hfn'])
    fs.encoder.load_state_dict(INIT['enc'])
    fs.optimizer.load_state_dict(copy.deepcopy(INIT['opt']))


def theta():
    return torch.cat([p.detach().reshape(-1).double().cpu() for p in TRACKED]).clone()


def update(k, hfn, out_fn, graphs):
    if graphs:
        torch.compiler.cudagraph_mark_step_begin()
    uh, yh, uf, yf, cix = batch_at(k)
    fs.optimizer.zero_grad(set_to_none=True)
    x = fs.encoder(uh, yh)
    y, _, _ = closed_loop_rollout(hfn, out_fn, uf, yf, x, bank, cix.long())
    L = torch.nn.functional.mse_loss(yf, y)
    L.backward()
    fs.optimizer.step()
    return float(L)


def cpu_validation(hfn, out_fn):
    """Stand-in for deepSI's validation: flip to CPU, roll out, flip back.

    Shortened (FLIP_VAL_STEPS instead of 48000) because the mechanism under test is the DEVICE
    FLIP with live compiled state, not the length of the run.
    """
    fs.hfn.cpu(); fs.encoder.cpu()
    uh, yh, uf, yf, cix = to_dev(VAL_CPU, 'cpu')
    uf, yf = uf[:, :VAL_STEPS].contiguous(), yf[:, :VAL_STEPS].contiguous()
    with torch.no_grad():
        x = fs.encoder(uh, yh)
        y, _, _ = closed_loop_rollout(hfn, out_fn, uf, yf, x, bank, cix.long())
        score = float(y.pow(2).mean().sqrt())
    fs.hfn.cuda(); fs.encoder.cuda()
    return score


def run_arm(tag, train_pair, val_pair, graphs):
    """train_pair/val_pair = (hfn, output_only) used for training and for the CPU validation."""
    reset()
    traj, losses, scores, spikes = [theta()], [], [], []
    k = 0
    for c in range(CYCLES):
        for j in range(K):
            t0 = time.perf_counter()
            losses.append(update(k, *train_pair, graphs))
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if j == 0 and c > 0:        # first update AFTER a flip: recompile shows up here
                spikes.append(dt)
            k += 1
            traj.append(theta())
        scores.append(cpu_validation(*val_pair))
    print(f"  {tag:<16} loss {losses[0]:.6e} -> {losses[-1]:.6e}   "
          f"cpu-scores {' '.join('%.6f' % s for s in scores)}")
    if spikes:
        print(f"  {'':<16} first-update-after-flip: "
              f"{' '.join('%.2fs' % s for s in spikes)}")
    return traj, losses, scores


banner("ARM 1: EAGER (reference)")
traj_e, loss_e, score_e = run_arm("eager", (fs.hfn, fs.hfn.output_only),
                                  (fs.hfn, fs.hfn.output_only), graphs=False)

results = {}
for tag, naive in (("compiled-safe", False), ("compiled-naive", True)):
    banner(f"ARM: {tag.upper()}  (validation {'COMPILED' if naive else 'uncompiled'})")
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    try:
        hfn_c = torch.compile(fs.hfn, **COMPILE_KW)
        out_c = torch.compile(fs.hfn.output_only, **COMPILE_KW)
        val_pair = (hfn_c, out_c) if naive else (fs.hfn, fs.hfn.output_only)
        results[tag] = run_arm(tag, (hfn_c, out_c), val_pair, graphs=True)
    except Exception as e:
        print(f"  CRASHED: {type(e).__name__}: {str(e)[:200]}")
        print("  -> this is failure mode 2 (loud). Better than silent corruption.")
        results[tag] = None
    torch.cuda.empty_cache()

banner("VERDICT")
th0 = traj_e[0]
de = traj_e[-1] - th0
print(f"  {'arm':<16} {'cos(dtheta)':>18} {'|dtheta| ratio':>16} {'max|dscore|':>14}")
for tag, r in results.items():
    if r is None:
        print(f"  {tag:<16} {'CRASHED':>18}")
        continue
    traj_c, _, score_c = r
    dc = traj_c[-1] - th0
    cos = float(torch.dot(de, dc) / (de.norm() * dc.norm())) if float(dc.norm()) else float('nan')
    ratio = float(dc.norm() / de.norm()) if float(de.norm()) else float('nan')
    dscore = max(abs(a - b) for a, b in zip(score_c, score_e))
    print(f"  {tag:<16} {cos:>18.12f} {ratio:>16.9f} {dscore:>14.3e}")

print("\n  Reference: the compile-vs-eager gate WITHOUT flips (job 80614) gave")
print("  cos=1.000000000000, ratio=0.999996483. A materially worse cos here means the")
print("  device flip is corrupting CUDA-graph state -- failure mode 1, the silent one.")
print("  A large first-update-after-flip time means Dynamo is recompiling -- mode 3.")
print("\nflip_safety complete")
