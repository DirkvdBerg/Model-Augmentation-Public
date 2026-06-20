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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat

# ── Path setup ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from model_augmentation.fit_systems.blocks import Gantry_State_Block

# ── Configuration ───────────────────────────────────────────────────────
CONFIG = "baseline"  # Toggle: "baseline" or "msd"

FS_ORIG = 20000  # THEORY: original sample rate (Hz), from main.m line 164
# THEORY: integer divisors of 20000 for alias-free decimation (ZOH inputs)
# Transition region for baseline f_high=7 Hz (Nyquist=14 Hz)
ALL_FS_NEW = [100, 80, 50, 40, 25, 20]
# HEURISTIC: 1% conservative bound for acceptable discretization error
NRMS_THRESHOLD = 0.01
# Reference uses high up_sample to approximate continuous-time integration
REF_UP_SAMPLE = 20

# T7: full MIMO (X sym + X anti + Y sweep), excites all modes + LPV scheduling
MAT_FILE = "T7_full_MIMO.mat"

DATA_DIR = os.path.join(
    PROJECT_ROOT, "data", "gantry", "matlab", "multisine", "baseline-v2"
)
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
        if fs_new == FS_ORIG:
            # Compare FS_ORIG with up_sample=1 vs reference (up_sample=REF_UP_SAMPLE)
            D = 1
        else:
            D = FS_ORIG // fs_new

        u_ds = u_stage[::D]
        x_ref = x_ref_hi[::D]

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
    """Sweep up_sample at each test rate, return nested dict."""
    UP_SAMPLES = [1, 2, 5, 10, 20]
    results = {}

    for fs_new in test_rates:
        D = FS_ORIG // fs_new
        u_ds = u_stage[::D]
        x_ref = x_ref_hi[::D]

        sims = {}
        for us in UP_SAMPLES:
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
            print(f"  fs={fs_new:6d} Hz  up_sample={us:2d}  ({elapsed:.1f}s)")

        # Reference: up_sample=20 at this rate
        x_ref_rk4 = sims[20]
        rate_results = {}
        for us in UP_SAMPLES:
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

    path = os.path.join(OUT_DIR, "downsampling_nrms_vs_fs.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_rk4_sweep(rk4_results):
    """Plot NRMS vs up_sample for each test rate."""
    rates = sorted(rk4_results.keys())
    n_rates = len(rates)
    if n_rates == 0:
        return

    fig, axes = plt.subplots(1, n_rates, figsize=(5 * n_rates, 5), squeeze=False)
    for i, fs_new in enumerate(rates):
        ax = axes[0, i]
        rate_data = rk4_results[fs_new]
        up_samples = sorted(rate_data.keys())
        channels = list(rate_data[up_samples[0]].keys())

        for ch in channels:
            vals = [rate_data[us][ch] for us in up_samples]
            ax.plot(up_samples, vals, "o-", label=ch, markersize=4)

        ax.axhline(NRMS_THRESHOLD, color="k", ls="--", alpha=0.5)
        ax.set_xlabel("up_sample (RK4 sub-steps)")
        ax.set_ylabel("NRMS vs up_sample=20")
        ax.set_title(f"fs={fs_new} Hz")
        ax.set_yscale("log")
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("RK4 sub-step sweep (LPV)")
    path = os.path.join(OUT_DIR, "rk4_substep_nrms.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# =====================================================================
# Main
# =====================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Config: {CONFIG}\n")

    # ── 1. Load system dynamics JSON ────────────────────────────────
    with open(JSON_PATH) as f:
        dyn = json.load(f)

    cfg_key = "baseline" if CONFIG == "baseline" else "msd"
    f_nyquist = dyn[cfg_key]["f_nyquist_hz"]
    f_practical = dyn[cfg_key]["f_practical_hz"]
    print(f"  f_nyquist   = {f_nyquist:.1f} Hz")
    print(f"  f_practical = {f_practical:.1f} Hz")

    # Filter sweep: go slightly below Nyquist to show breakdown
    sweep_rates = [fs for fs in ALL_FS_NEW if fs >= f_nyquist / 2]
    print(f"  Sweep rates: {sweep_rates}\n")

    # ── 2. Load reference data ──────────────────────────────────────
    mat_path = os.path.join(DATA_DIR, MAT_FILE)
    d = loadmat(mat_path, squeeze_me=True)
    u_stage = d["u_total"].astype(np.float64)     # (N, 3) at 20 kHz
    x_logical = d["x_logical"].astype(np.float64)  # (N, 6) at 20 kHz
    N_orig = u_stage.shape[0]
    print(f"  Loaded {MAT_FILE}: {N_orig} samples at {FS_ORIG} Hz "
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

    # ── 5. Test B: RK4 sub-step sweep ──────────────────────────────
    rk4_results = {}
    if passing:
        test_rates = sorted(passing)[:3]
        print(f"=== Test B: RK4 sub-step sweep at {test_rates} ===")
        rk4_results = run_rk4_sweep(
            u_stage, x_ref_hi, test_rates, std_x, std_u, x_mean, u_mean,
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
        "data_file": MAT_FILE,
        "lowest_passing_fs": lowest_passing,
        "sweep": {str(k): v for k, v in sweep_results.items()},
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    sweep_path = os.path.join(OUT_DIR, "downsampling_sweep.json")
    with open(sweep_path, "w") as f:
        json.dump(sweep_json, f, indent=2)
    print(f"  Saved {sweep_path}")

    if rk4_results:
        rk4_json = {
            "config": CONFIG,
            "reference_up_sample": 20,
            "nrms_formula": "rms(err) / (max(ref) - min(ref)) per channel",
            "sweep": {
                str(fs): {str(us): v for us, v in rate_data.items()}
                for fs, rate_data in rk4_results.items()
            },
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        rk4_path = os.path.join(OUT_DIR, "rk4_substep_sweep.json")
        with open(rk4_path, "w") as f:
            json.dump(rk4_json, f, indent=2)
        print(f"  Saved {rk4_path}")

    # ── 7. Plots ────────────────────────────────────────────────────
    print("\n=== Generating plots ===")
    plot_downsampling_sweep(sweep_results, f_nyquist, f_practical, CONFIG)
    if rk4_results:
        plot_rk4_sweep(rk4_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
