"""What is the smallest residual float32 can REPRESENT at nf=12000? (D-169)

Chasing a normalised residual of 1e-7 or 1e-8 is an optimisation problem only if the arithmetic
can represent it. This measures the arithmetic floor directly: the SAME rollout, same parameters,
same inputs, run in float32 and in float64, differenced.

    floor = |y_f32 - y_f64|

That difference is not an error in either arm; it is the width of float32's uncertainty about the
answer. No amount of training can drive a residual below it, because the model cannot tell two
outputs that close apart. If the target residual sits under the floor, the problem is arithmetic,
not optimisation, and the fix is use_f64 rather than more updates.

WHY THIS IS ASKED NOW. The entry file justifies use_f64=False from `cl_update_precision`, which
measured cos(dtheta_32, dtheta_64) = 0.999042 at nf = 400. At nf = 12000 the rollout accumulates
30x more rounding, so that conclusion does not automatically carry over. A random-walk estimate
gives eps*sqrt(N) = 1.19e-7 * sqrt(12000) ~ 1.3e-5 normalised; this measures the real number.

HARDWARE NOTE. If the floor turns out to matter, float64 changes which partition is correct:
RTX 2080 Ti (`oahu`) runs FP64 at 1/32 rate, A100 (`lanai`/`molokai`) at 1/2. That is the ONE
case in this whole investigation where the A100 is the right machine. This script therefore also
times both dtypes so the cost of the fix is on the record alongside its benefit.

Run:  python scripts/gantry/GPU/float32_floor.py
Env:  FLOOR_NF (default 12000 -- the horizon actually in question), FLOOR_BATCH (64),
      FLOOR_HORIZONS (a comma list, e.g. 200,1000,4000,12000, to see the floor GROW with nf)
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

BATCH = int(os.environ.get('FLOOR_BATCH', 64))
HORIZONS = [int(h) for h in os.environ.get('FLOOR_HORIZONS', '200,1000,4000,12000').split(',')]
EPS32 = float(np.finfo(np.float32).eps)      # 1.1920929e-07


def banner(s):
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}", flush=True)


banner("ENVIRONMENT")
if not torch.cuda.is_available():
    print("no CUDA device (need sbatch --gres=gpu:1)")
    sys.exit(1)
p = torch.cuda.get_device_properties(0)
print(f"  GPU {p.name}  sm_{p.major}{p.minor}")
print(f"  float32 eps = {EPS32:.6e}   batch={BATCH}   horizons={HORIZONS}")
print("  NOTE: FP64 is 1/32 rate on RTX 2080 Ti, 1/2 on A100. The float64 arm is a REFERENCE,")
print("        not a proposal; its runtime here is not what a float64 training run would cost.")


def build(use_f64, nf):
    """A pipeline at the requested dtype. Same seed -> same parameters in both arms."""
    torch.manual_seed(42)
    np.random.seed(42)
    cfg = replace(entry.CFG, orth=False, device='cuda', nf_override=nf, stride=200,
                  use_f64=use_f64, save_flag=False)
    data = load_datasets(cfg)
    norm = compute_normalization(cfg, data)
    fs = build_model(cfg.hp, cfg, data, norm)
    fs.simulator = build_closed_loop(fs, norm, cfg, train_files=TRAIN_FILES,
                                     val_files=VAL_FILES, val_data=data.val_ckpt_data,
                                     verbose=False)
    arrs = fs.make_training_data(fs.norm.transform(data.train_data), nf=nf, stride=cfg.stride)
    return cfg, fs, arrs


def rollout(fs, cfg, arrs, ix):
    """One free rollout, no grad. Returns y (batch, nf, ny) at the pipeline dtype."""
    out = []
    for a in arrs:
        t = torch.as_tensor(np.ascontiguousarray(np.asarray(a)[ix]))
        out.append(t.to('cuda', cfg.dtype_pt) if t.is_floating_point() else t.to('cuda'))
    uh, yh, uf, yf, cix = out
    with torch.no_grad():
        x0 = fs.encoder(uh, yh)
        y, _, _ = closed_loop_rollout(fs.hfn, fs.hfn.output_only, uf, yf, x0,
                                      fs.simulator.bank, cix.long())
    return y, yf


banner("MEASURING THE FLOOR")
print(f"  {'nf':>7} {'max|y32-y64|':>15} {'rms|y32-y64|':>15} {'floor/eps':>11} "
      f"{'rms residual':>14} {'floor/residual':>15} {'t32':>8} {'t64':>9}")
rows = []
for nf in HORIZONS:
    try:
        cfg32, fs32, arrs32 = build(False, nf)
        n_win = len(arrs32[0])
        ix = np.random.default_rng(0).choice(n_win, size=min(BATCH, n_win), replace=False)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        y32, yf32 = rollout(fs32, cfg32, arrs32, ix)
        torch.cuda.synchronize(); t32 = time.perf_counter() - t0
        del fs32, arrs32
        torch.cuda.empty_cache()

        cfg64, fs64, arrs64 = build(True, nf)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        y64, yf64 = rollout(fs64, cfg64, arrs64, ix)
        torch.cuda.synchronize(); t64 = time.perf_counter() - t0

        d = (y32.double() - y64).abs()
        floor_max, floor_rms = float(d.max()), float(d.pow(2).mean().sqrt())
        # The residual the model is actually fitting, in the SAME normalised units, so the two
        # are directly comparable. This ratio is the whole answer.
        res = float((y64 - yf64).pow(2).mean().sqrt())
        rows.append((nf, floor_max, floor_rms, res))
        print(f"  {nf:>7} {floor_max:>15.4e} {floor_rms:>15.4e} {floor_rms / EPS32:>11.1f} "
              f"{res:>14.4e} {floor_rms / max(res, 1e-300):>15.3e} "
              f"{t32:>7.2f}s {t64:>8.2f}s", flush=True)
        del fs64, arrs64, y32, y64
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        print(f"  {nf:>7}   OOM (reduce FLOOR_BATCH)")
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  {nf:>7}   FAILED {type(e).__name__}: {str(e)[:90]}")
        torch.cuda.empty_cache()

banner("WHAT THIS MEANS")
if not rows:
    print("  no measurement succeeded")
    sys.exit(1)
nf_t, fmax, frms, res = rows[-1]
print(f"  At nf={nf_t}, float32 cannot distinguish outputs closer than ~{frms:.3e} (rms,")
print(f"  normalised). That is the ARITHMETIC FLOOR: no amount of training goes below it.\n")
print(f"  {'target normalised residual':<34} {'reachable in float32?':>24}")
for tgt in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
    ok = tgt > 10 * frms
    print(f"  {tgt:<34.0e} {('yes' if ok else 'NO - below the float32 floor'):>24}")
print(f"\n  current residual at this horizon: {res:.4e}  "
      f"({res / max(frms, 1e-300):.1f}x above the floor)")
if len(rows) > 1:
    n0, _, f0, _ = rows[0]
    print(f"\n  growth with horizon: floor went {f0:.3e} (nf={n0}) -> {frms:.3e} (nf={nf_t}), "
          f"{frms / max(f0, 1e-300):.1f}x for a {nf_t / n0:.0f}x longer rollout")
    print(f"  (a random walk would predict {np.sqrt(nf_t / n0):.1f}x; "
          f"much faster than that means the rollout AMPLIFIES rounding, not just accumulates it)")
print("\n  If a target sits below the floor, the fix is use_f64=True, NOT more updates --")
print("  and float64 makes the A100 partitions (lanai/molokai, FP64 1/2 rate) the correct")
print("  hardware instead of oahu (1/32 rate).")
print("\nfloat32_floor complete")
