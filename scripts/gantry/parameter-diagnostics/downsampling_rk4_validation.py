"""Downsampling and RK4 step-size validation for the gantry model.

Determines the minimum viable sampling rate (FS_NEW) using the LPV model
(Gantry_State_Block with Y_op=None) to match the actual training pipeline.

Test A: LPV downsampling sweep (NRMS vs FS_NEW, up_sample=1 vs reference).
Test B: RK4 sub-step sweep at key rates from Test A.

Usage:
    conda run -n GraduationProject python scripts/gantry/parameter-diagnostics/downsampling_rk4_validation.py
"""

import json
import os
import sys
import time
from datetime import datetime
from math import gcd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import resample_poly

# ── Path setup ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from model_augmentation.fit_systems.blocks import Gantry_State_Block

# ── Configuration ───────────────────────────────────────────────────────
# Toggle: "baseline" or "msd_narrowband"
CONFIG = "msd_narrowband"

FS_ORIG = 20000  # THEORY: original sample rate (Hz), from main.m line 164
# HEURISTIC: 1% conservative bound for acceptable discretization error
NRMS_THRESHOLD = 0.01
# Reference uses high up_sample to approximate continuous-time integration
REF_UP_SAMPLE = 20

# Per-config settings: data path, MAT file, JSON dynamics key, and rates to sweep.
# Rates that are NOT integer divisors of FS_ORIG (i.e. 3000, 6000 Hz) use
# resample_poly for decimation.  All others use stride decimation.
CONFIGS = {
    "baseline": {
        "data_dir": os.path.join(
            PROJECT_ROOT, "data", "gantry", "matlab", "multisine", "baseline-v2"
        ),
        "mat_file": "T7_full_MIMO.mat",
        "json_key": "baseline",
        # Only exact divisors of 20000 in this range:
        "sweep_rates": [500, 1000, 2000, 4000, 5000],
    },
    "msd_narrowband": {
        "data_dir": os.path.join(
            PROJECT_ROOT, "data", "gantry", "matlab", "multisine", "m50", "narrowband"
        ),
        "mat_file": "T7_full_MIMO.mat",
        "json_key": "msd",   # MSD poles for Nyquist marker (fa≈150 Hz → f_nyquist≈300 Hz)
        # 3000 and 6000 are not exact divisors of 20000 → resampled via resample_poly
        "sweep_rates": [500, 1000, 2000, 3000, 4000, 5000, 6000],
    },
}

OUT_DIR = os.path.join(
    PROJECT_ROOT, "simulations", "gantry_subnet", "diagnostics"
)
JSON_PATH = os.path.join(OUT_DIR, "system_dynamics.json")

STATE_NAMES = ["X", "theta", "Y", "dX", "dtheta", "dY"]


# =====================================================================
# Helpers
# =====================================================================
def compute_nrms_range(x_sim, x_ref, channel_names):
    """Range-based NRMS per channel: rms(err) / (max - min)."""
    nrms = {}
    for i, name in enumerate(channel_names):
        err = x_sim[:, i] - x_ref[:, i]
        rng = x_ref[:, i].max() - x_ref[:, i].min()
        if rng < 1e-12:
            nrms[name] = 0.0 if np.allclose(err, 0, atol=1e-12) else float("inf")
        else:
            nrms[name] = float(np.sqrt(np.mean(err**2)) / rng)
    return nrms


def decimate_signal(sig, fs_orig, fs_new):
    """Downsample sig (N, C) from fs_orig to fs_new.

    Uses stride for exact integer divisors; resample_poly (anti-aliased,
    rational ratio) for non-integer cases like 20000→3000 or 20000→6000.
    """
    if fs_orig % fs_new == 0:
        D = fs_orig // fs_new
        return sig[::D]
    g = gcd(int(fs_orig), int(fs_new))
    up   = int(fs_new)   // g
    down = int(fs_orig)  // g
    return resample_poly(sig, up, down, axis=0)


def compute_normalization(u_stage, x_logical):
    """Compute normalization constants from full-rate data."""
    std_x = x_logical.std(axis=0).reshape(6, 1).astype(np.float64) + 1e-8
    std_u = u_stage.std(axis=0).reshape(3, 1).astype(np.float64) + 1e-8
    x_mean = x_logical.mean(axis=0).reshape(6, 1).astype(np.float64)
    u_mean = u_stage.mean(axis=0).reshape(3, 1).astype(np.float64)
    return std_x, std_u, x_mean, u_mean


def forward_sim_lpv(state_block, u_stage, x0_phys, std_x, std_u, x_mean, u_mean):
    """Forward-simulate using Gantry_State_Block (LPV), return physical states."""
    N = u_stage.shape[0]
    std_x_f = std_x.flatten()
    std_u_f = std_u.flatten()
    x_mean_f = x_mean.flatten()
    u_mean_f = u_mean.flatten()

    x_phys = np.zeros((N, 6))
    x_phys[0] = x0_phys

    # Normalize initial state
    x0_norm = (x0_phys - x_mean_f) / std_x_f
    x_t = torch.tensor(x0_norm, dtype=torch.float32).reshape(1, 6, 1)

    with torch.no_grad():
        for k in range(N - 1):
            u_norm_k = (u_stage[k] - u_mean_f) / std_u_f
            u_t = torch.tensor(u_norm_k, dtype=torch.float32).reshape(1, 3, 1)
            z = torch.cat([x_t, u_t], dim=1)  # (1, 9, 1)
            x_t = state_block.nonlinear_function(z)
            x_phys[k + 1] = x_t.squeeze().numpy() * std_x_f + x_mean_f

    return x_phys


# =====================================================================
# Test A: LPV downsampling sweep
# =====================================================================
def run_lpv_sweep(u_stage, x_logical, sweep_rates, std_x, std_u, x_mean, u_mean):
    """Sweep FS_NEW using LPV model.  Reference = FS_ORIG with high up_sample.

    Self-reference approach: same LPV model at different rates. Isolates
    discretization error.
    """
    results = {}

    # Reference: LPV at FS_ORIG with high up_sample
    print(f"  Computing reference: fs={FS_ORIG} Hz, up_sample={REF_UP_SAMPLE} ...")
    t0 = time.time()
    ref_block = Gantry_State_Block(
        Y_op=None, std_x=std_x, std_u=std_u,
        x_mean=x_mean, u_mean=u_mean,
        Ts=1.0 / FS_ORIG, up_sample=REF_UP_SAMPLE,
    )
    ref_block.eval()
    x_ref_hi = forward_sim_lpv(
        ref_block, u_stage, x_logical[0],
        std_x, std_u, x_mean, u_mean,
    )
    print(f"  Reference done ({time.time() - t0:.1f}s)\n")

    for fs_new in sweep_rates:
        u_ds  = decimate_signal(u_stage,  FS_ORIG, fs_new)
        x_ref = decimate_signal(x_ref_hi, FS_ORIG, fs_new)

        t0 = time.time()
        block = Gantry_State_Block(
            Y_op=None, std_x=std_x, std_u=std_u,
            x_mean=x_mean, u_mean=u_mean,
            Ts=1.0 / fs_new, up_sample=1,
        )
        block.eval()
        x_sim = forward_sim_lpv(
            block, u_ds, x_ref[0],
            std_x, std_u, x_mean, u_mean,
        )
        elapsed = time.time() - t0

        nrms = compute_nrms_range(x_sim, x_ref, STATE_NAMES)
        max_nrms = max(nrms.values())
        status = "PASS" if max_nrms < NRMS_THRESHOLD else "FAIL"
        print(f"  fs={fs_new:6d} Hz  max_nrms={max_nrms:.2e}  [{status}]  ({elapsed:.1f}s)")
        results[fs_new] = nrms

    return results, x_ref_hi


# =====================================================================
# Test B: RK4 sub-step sweep
# =====================================================================
def run_rk4_sweep(u_stage, x_ref_hi, test_rates, std_x, std_u, x_mean, u_mean):
    """Sweep up_sample at each test rate, return nested dict.

    Tests up_sample = 1..4.  up_sample=20 is the hidden reference (continuous-time
    approximation); it is not plotted but used to compute NRMS.

    Why up_sample may not improve monotonically:
      - At HIGH fs (e.g. 5000 Hz), h=Ts is already tiny; all errors are near
        machine epsilon.  The ordering of up_sample=1..4 vs. 20 is numerical noise.
      - At LOW fs with high-frequency inputs (e.g. 500 Hz + 150 Hz multisine),
        ZOH input aliasing dominates.  More sub-steps can't recover aliased content.
      - Monotonic improvement only shows up at intermediate rates where integration
        error (not input aliasing) is the main source.
    """
    UP_SAMPLES_TEST = [1, 2, 3, 4]
    results = {}

    for fs_new in test_rates:
        u_ds  = decimate_signal(u_stage,  FS_ORIG, fs_new)
        x_ref = decimate_signal(x_ref_hi, FS_ORIG, fs_new)

        # Simulate test variants + hidden reference (up_sample=20)
        sims = {}
        for us in UP_SAMPLES_TEST + [20]:
            t0 = time.time()
            block = Gantry_State_Block(
                Y_op=None, std_x=std_x, std_u=std_u,
                x_mean=x_mean, u_mean=u_mean,
                Ts=1.0 / fs_new, up_sample=us,
            )
            block.eval()
            sims[us] = forward_sim_lpv(
                block, u_ds, x_ref[0],
                std_x, std_u, x_mean, u_mean,
            )
            elapsed = time.time() - t0
            label = " [ref]" if us == 20 else ""
            print(f"  fs={fs_new:6d} Hz  up_sample={us:2d}{label}  ({elapsed:.1f}s)")

        # Reference: up_sample=20 at this rate
        x_ref_rk4 = sims[20]
        rate_results = {}
        for us in UP_SAMPLES_TEST:
            nrms = compute_nrms_range(sims[us], x_ref_rk4, STATE_NAMES)
            max_nrms = max(nrms.values())
            status = "PASS" if max_nrms < NRMS_THRESHOLD else "FAIL"
            print(f"    -> up_sample={us:2d}  max_nrms={max_nrms:.2e}  [{status}]")
            rate_results[us] = nrms

        results[fs_new] = rate_results

    return results


# =====================================================================
# Plots
# =====================================================================
def plot_downsampling_sweep(sweep_results, f_nyquist, f_practical, config):
    """Plot NRMS vs FS_NEW with Nyquist markers."""
    rates = sorted(sweep_results.keys())
    channels = list(sweep_results[rates[0]].keys())

    fig, ax = plt.subplots(figsize=(10, 6))
    for ch in channels:
        vals = [sweep_results[r][ch] for r in rates]
        ax.plot(rates, vals, "o-", label=ch, markersize=4)

    ax.axhline(NRMS_THRESHOLD, color="k", ls="--", alpha=0.5, label=f"threshold={NRMS_THRESHOLD}")
    ax.axvline(f_nyquist, color="r", ls=":", alpha=0.7, label=f"f_nyquist={f_nyquist:.0f} Hz")
    ax.axvline(f_practical, color="orange", ls=":", alpha=0.7, label=f"f_practical={f_practical:.0f} Hz")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("FS_NEW (Hz)")
    ax.set_ylabel("NRMS (range-based)")
    ax.set_title(f"Downsampling sweep ({config}, LPV, up_sample=1 vs ref={REF_UP_SAMPLE})")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    ax.invert_xaxis()

    path = os.path.join(OUT_DIR, f"downsampling_nrms_vs_fs_{config}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_rk4_sweep(rk4_results):
    """Plot NRMS vs up_sample (1..4 vs ref=20) for each rate.

    Non-monotonic appearance at high fs is expected: errors are near float
    precision and ordering is noise, not physics.  At low fs dominated by
    ZOH input aliasing, up_sample has little effect for the same reason.
    """
    rates = sorted(rk4_results.keys())
    n_rates = len(rates)
    if n_rates == 0:
        return

    # Two rows: top = per-channel lines, bottom = max-channel summary
    fig, axes = plt.subplots(2, n_rates, figsize=(4 * n_rates, 8), squeeze=False)
    for i, fs_new in enumerate(rates):
        rate_data = rk4_results[fs_new]
        up_samples = sorted(rate_data.keys())
        channels = list(rate_data[up_samples[0]].keys())
        max_vals = [max(rate_data[us].values()) for us in up_samples]

        # Top: per-channel
        ax = axes[0, i]
        for ch in channels:
            vals = [rate_data[us][ch] for us in up_samples]
            ax.plot(up_samples, vals, "o-", label=ch, markersize=4)
        ax.axhline(NRMS_THRESHOLD, color="k", ls="--", alpha=0.5, label="threshold")
        ax.set_xlabel("up_sample")
        ax.set_ylabel("NRMS vs up_sample=20")
        ax.set_title(f"fs={fs_new} Hz")
        ax.set_xticks(up_samples)
        ax.set_yscale("log")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, which="both", alpha=0.3)

        # Bottom: max channel
        ax2 = axes[1, i]
        ax2.plot(up_samples, max_vals, "s-", color="black", markersize=5, label="max channel")
        ax2.axhline(NRMS_THRESHOLD, color="k", ls="--", alpha=0.5)
        ax2.set_xlabel("up_sample")
        ax2.set_ylabel("max NRMS")
        ax2.set_xticks(up_samples)
        ax2.set_yscale("log")
        ax2.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"RK4 sub-step sweep (LPV, {CONFIG}) — reference = up_sample=20\n"
        "Non-monotonic at high fs = float-precision noise; at low fs = ZOH aliasing dominates",
        fontsize=9,
    )
    path = os.path.join(OUT_DIR, f"rk4_substep_nrms_{CONFIG}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# =====================================================================
# Main
# =====================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = CONFIGS[CONFIG]
    data_dir   = cfg["data_dir"]
    mat_file   = cfg["mat_file"]
    json_key   = cfg["json_key"]
    sweep_rates = cfg["sweep_rates"]

    print(f"Config: {CONFIG}")
    print(f"  Data: {data_dir}/{mat_file}")
    print(f"  Rates: {sweep_rates}\n")

    # ── 1. Load system dynamics JSON ────────────────────────────────
    with open(JSON_PATH) as f:
        dyn = json.load(f)

    f_nyquist   = dyn[json_key]["f_nyquist_hz"]
    f_practical = dyn[json_key]["f_practical_hz"]
    print(f"  f_nyquist   = {f_nyquist:.1f} Hz")
    print(f"  f_practical = {f_practical:.1f} Hz\n")

    # ── 2. Load reference data ──────────────────────────────────────
    mat_path = os.path.join(data_dir, mat_file)
    d = loadmat(mat_path, squeeze_me=True)
    u_stage   = d["u_total"].astype(np.float64)    # (N, 3) at 20 kHz
    x_logical = d["x_logical"].astype(np.float64)  # (N, 6) at 20 kHz
    N_orig = u_stage.shape[0]
    print(f"  Loaded {mat_file}: {N_orig} samples at {FS_ORIG} Hz "
          f"({N_orig / FS_ORIG:.2f} s)\n")

    # ── 3. Normalization (from full-rate data) ──────────────────────
    std_x, std_u, x_mean, u_mean = compute_normalization(u_stage, x_logical)

    # ── 4. Test A: LPV downsampling sweep ───────────────────────────
    print(f"=== Test A: LPV downsampling sweep (up_sample=1 vs ref={REF_UP_SAMPLE}) ===")
    sweep_results, x_ref_hi = run_lpv_sweep(
        u_stage, x_logical, sweep_rates, std_x, std_u, x_mean, u_mean,
    )

    # Find lowest passing rate
    passing = [
        fs for fs in sorted(sweep_results.keys(), reverse=True)
        if max(sweep_results[fs].values()) < NRMS_THRESHOLD
    ]
    lowest_passing = min(passing) if passing else None
    print(f"\n  Lowest passing rate: {lowest_passing} Hz "
          f"(threshold={NRMS_THRESHOLD})\n")

    # ── 5. Test B: RK4 sub-step sweep (all sweep rates) ────────────
    # Run on every rate so we can see where up_sample=1..4 matters.
    print(f"=== Test B: RK4 sub-step sweep (up_sample=1..4 vs ref=20) at all rates ===")
    rk4_results = run_rk4_sweep(
        u_stage, x_ref_hi, sweep_rates, std_x, std_u, x_mean, u_mean,
    )
    print()

    # ── 6. Save JSON results ────────────────────────────────────────
    print("=== Saving results ===")
    sweep_json = {
        "config": CONFIG,
        "fs_orig": FS_ORIG,
        "f_nyquist_hz": f_nyquist,
        "f_practical_hz": f_practical,
        "nrms_threshold": NRMS_THRESHOLD,
        "ref_up_sample": REF_UP_SAMPLE,
        "nrms_formula": "rms(err) / (max(ref) - min(ref)) per channel",
        "data_file": mat_file,
        "lowest_passing_fs": lowest_passing,
        "sweep": {str(k): v for k, v in sweep_results.items()},
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    sweep_path = os.path.join(OUT_DIR, f"downsampling_sweep_{CONFIG}.json")
    with open(sweep_path, "w") as f:
        json.dump(sweep_json, f, indent=2)
    print(f"  Saved {sweep_path}")

    rk4_json = {
        "config": CONFIG,
        "reference_up_sample": 20,
        "up_samples_tested": [1, 2, 3, 4],
        "nrms_formula": "rms(err) / (max(ref) - min(ref)) per channel",
        "sweep": {
            str(fs): {str(us): v for us, v in rate_data.items()}
            for fs, rate_data in rk4_results.items()
        },
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    rk4_path = os.path.join(OUT_DIR, f"rk4_substep_sweep_{CONFIG}.json")
    with open(rk4_path, "w") as f:
        json.dump(rk4_json, f, indent=2)
    print(f"  Saved {rk4_path}")

    # ── 7. Plots ────────────────────────────────────────────────────
    print("\n=== Generating plots ===")
    plot_downsampling_sweep(sweep_results, f_nyquist, f_practical, CONFIG)
    plot_rk4_sweep(rk4_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
