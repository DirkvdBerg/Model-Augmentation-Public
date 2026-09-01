"""Does torch.compile change WHERE TRAINING GOES? (D-169 acceptance gate)

`gpu_bench2` established that inductor + reduce-overhead is 6.06x and agrees with eager to
max|dy|=9.5e-07, max|dg|=7.5e-10. That is ONE forward/backward. It does not answer the question
that matters before committing a 10 h run: after N optimiser steps, do the two arms end up at the
same parameters?

METHOD, deliberately identical to `cl_direct_vs_residual` / `cl_update_precision`, which this
project already used to settle float32-vs-float64 (measured cos(dtheta_32, dtheta_64) = 0.999042,
|dtheta| ratio 1.0048, and concluded Adam's per-parameter normalisation absorbs rollout noise):

    run N updates in each arm, FROM IDENTICAL PARAMETERS, ON IDENTICAL BATCHES,
    then compare the cumulative parameter displacement dtheta = theta_N - theta_0
    by cosine similarity and norm ratio.

Cosine near 1.0 and ratio near 1.0 means compilation moves the model along the same trajectory,
i.e. the 6.06x is free. A cosine that decays with N means it does not, and the speedup is not
usable as-is.

Run:  python scripts/gantry/GPU/compile_update_precision.py
Env:  PREC_NF (default 200; set 12000 for the definitive test at the real horizon, ~40 min)
      PREC_BATCH (512), PREC_UPDATES (40), PREC_CHUNK (0 = checkpointing off)
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

NF = int(os.environ.get('PREC_NF', 200))
BATCH = int(os.environ.get('PREC_BATCH', 512))
UPDATES = int(os.environ.get('PREC_UPDATES', 40))
CHUNK = int(os.environ.get('PREC_CHUNK', 0))


def banner(s):
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}", flush=True)


banner("SETUP")
if not torch.cuda.is_available():
    print("no CUDA device (need sbatch --gres=gpu:1)")
    sys.exit(1)
print(f"  GPU {torch.cuda.get_device_properties(0).name}")
print(f"  nf={NF}  batch={BATCH}  updates={UPDATES}  chunk={CHUNK} (0=checkpointing off)")

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
print(f"  windows={N_WIN}   lr={cfg.lr:g}  adam_eps={cfg.adam_eps:.1e}")

# The exact same batch SEQUENCE for both arms. Indices are drawn once; tensors are built per
# update rather than all at once, because at nf=12000 forty pre-materialised batches would be
# several GB. Same indices -> same batches, which is what the comparison requires.
_rng = np.random.default_rng(7)
BATCH_IX = [_rng.choice(N_WIN, size=BATCH, replace=BATCH > N_WIN) for _ in range(UPDATES)]


def batch_at(k):
    out = []
    for a in arrs:
        t = torch.as_tensor(np.ascontiguousarray(np.asarray(a)[BATCH_IX[k]]))
        out.append(t.to('cuda', cfg.dtype_pt) if t.is_floating_point() else t.to('cuda'))
    return out


# Snapshot the exact starting point so arm 2 begins where arm 1 did, optimiser state included.
INIT = {
    'hfn': copy.deepcopy(fs.hfn.state_dict()),
    'enc': copy.deepcopy(fs.encoder.state_dict()),
    'opt': copy.deepcopy(fs.optimizer.state_dict()),
}
TRACKED = [p for g in fs.optimizer.param_groups for p in g['params']]
print(f"  tracked parameters: {len(TRACKED)} tensors, "
      f"{sum(p.numel() for p in TRACKED)} scalars")


def reset():
    fs.hfn.load_state_dict(INIT['hfn'])
    fs.encoder.load_state_dict(INIT['enc'])
    fs.optimizer.load_state_dict(copy.deepcopy(INIT['opt']))


def theta():
    return torch.cat([p.detach().reshape(-1) for p in TRACKED]).clone()


def run_arm(tag, hfn, out_fn, graphs):
    reset()
    th0 = theta()
    losses, traj = [], [th0]
    t0 = time.perf_counter()
    for k in range(UPDATES):
        if graphs:
            torch.compiler.cudagraph_mark_step_begin()
        b = batch_at(k)
        uh, yh, uf, yf, cix = b
        fs.optimizer.zero_grad(set_to_none=True)
        x = fs.encoder(uh, yh)
        y, _, _ = closed_loop_rollout(hfn, out_fn, uf, yf, x, bank, cix.long(), chunk=CHUNK)
        L = torch.nn.functional.mse_loss(yf, y)
        L.backward()
        fs.optimizer.step()
        losses.append(float(L))
        traj.append(theta())
        del b, x, y, L
    torch.cuda.synchronize()
    print(f"  {tag:<12} {UPDATES} updates in {time.perf_counter() - t0:6.1f} s   "
          f"loss {losses[0]:.6e} -> {losses[-1]:.6e}", flush=True)
    return losses, traj


banner("ARM 1: EAGER")
loss_e, traj_e = run_arm("eager", fs.hfn, fs.hfn.output_only, graphs=False)

banner("ARM 2: INDUCTOR + reduce-overhead")
torch._dynamo.reset()
t0 = time.perf_counter()
try:
    hfn_c = torch.compile(fs.hfn, backend='inductor', mode='reduce-overhead', fullgraph=True)
    out_c = torch.compile(fs.hfn.output_only, backend='inductor', mode='reduce-overhead',
                          fullgraph=True)
    loss_c, traj_c = run_arm("compiled", hfn_c, out_c, graphs=True)
    print(f"  (compile included in the arm time; first call ~{time.perf_counter() - t0:.0f} s)")
except Exception as e:
    # Kept non-fatal so arm 1 is never lost to an environment problem. Expected on any card
    # below sm_70 (Triton's floor): the development PC's Quadro P2000 is sm_61, which is why
    # this test only produces a verdict on the cluster.
    print(f"  COMPILED ARM UNAVAILABLE: {type(e).__name__}: {str(e)[:160]}")
    print("\n  Arm 1 (eager) completed; there is nothing to compare it against.")
    print(f"  eager loss trajectory: {' '.join('%.4e' % v for v in loss_e)}")
    print("\n  This is expected on sm_61 hardware. Run on oahu (RTX 2080 Ti, sm_75).")
    sys.exit(0)


def cos_ratio(a, b):
    na, nb = a.norm(), b.norm()
    if float(na) == 0 or float(nb) == 0:
        return float('nan'), float('nan')
    return float(torch.dot(a, b) / (na * nb)), float(nb / na)


banner("RESULT")
th0 = traj_e[0]
print(f"  {'after':>7} {'cos(dtheta)':>18} {'|dtheta| ratio':>16} "
      f"{'loss rel diff':>15} {'|dtheta|_eager':>16}")
for k in (1, 2, 5, 10, 20, UPDATES):
    if k > UPDATES:
        continue
    de, dc = traj_e[k] - th0, traj_c[k] - th0
    c, r = cos_ratio(de, dc)
    lrel = abs(loss_c[k - 1] - loss_e[k - 1]) / max(abs(loss_e[k - 1]), 1e-300)
    print(f"  {k:>7} {c:>18.12f} {r:>16.9f} {lrel:>15.3e} {float(de.norm()):>16.6e}")

de, dc = traj_e[-1] - th0, traj_c[-1] - th0
c, r = cos_ratio(de, dc)
sep = float((traj_c[-1] - traj_e[-1]).norm() / max(float(de.norm()), 1e-300))

# Reference: this project's float32-vs-float64 comparison, which was ACCEPTED, measured
# cos = 0.999042 and ratio = 1.0048 (see the use_f64 note in gantry_interconnect_dynamic.py).
REF_COS, REF_RATIO = 0.999042, 1.0048
print(f"\n  cumulative over {UPDATES} updates:  cos={c:.12f}  ratio={r:.9f}")
print(f"  parameter separation / |dtheta|: {sep:.6e}")
print(f"\n  reference, the ACCEPTED float32-vs-float64 result on this pipeline:"
      f"  cos={REF_COS:.6f}  ratio={REF_RATIO:.4f}")
ok = (c >= REF_COS) and (abs(r - 1) <= abs(REF_RATIO - 1))
print(f"\n  VERDICT: compilation is {'TIGHTER' if ok else 'LOOSER'} than a change this project "
      f"already accepted\n           -> {'safe to use for the nf=12000 run' if ok else 'inspect before committing a 10 h run'}")
print("\ncompile_update_precision complete")
