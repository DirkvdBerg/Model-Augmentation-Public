"""
plot_convergence.py  —  training / validation convergence from a checkpoint file

Usage:
    conda run -n GraduationProject python lpv_lfr_baseline/scripts/plot_convergence.py <checkpoint.pt>
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="Plot train/val loss curves from checkpoint")
    parser.add_argument("checkpoint", help="Path to .pt checkpoint file")
    parser.add_argument("--no-show", action="store_true", help="Save only, don't display")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    history = ckpt["history"]

    # Extract training loss (every epoch) and validation RMSE (sparse)
    train_epochs = np.array([h["epoch"] for h in history])
    train_mse = np.array([h["mse_loss"] for h in history])

    val_mask = [h for h in history if "full_traj_rmse_m" in h]
    val_epochs = np.array([h["epoch"] for h in val_mask])
    val_rmse = np.array([h["full_traj_rmse_m"] for h in val_mask])

    # Run metadata for title
    n_epochs = ckpt.get("epochs", train_epochs[-1] + 1)
    run_id = ckpt.get("run_id", "")
    dataset = ckpt.get("active_traj_ids", "")
    lr = ckpt.get("lr", "")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: training MSE loss
    ax1.semilogy(train_epochs, train_mse, linewidth=0.8, color="C0")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training MSE loss")
    ax1.set_title("Training loss (segment MSE)")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    # Right: validation full-trajectory RMSE
    ax2.semilogy(val_epochs, val_rmse, marker="o", markersize=3, linewidth=0.8, color="C1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Full-trajectory RMSE [m]")
    ax2.set_title("Validation RMSE (full trajectory)")
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    fig.suptitle(
        f"Convergence — {n_epochs} epochs, lr={lr}, trajs={dataset}, run={run_id}",
        fontsize=10,
    )
    fig.tight_layout()

    out_dir = os.path.dirname(args.checkpoint)
    out_path = os.path.join(out_dir, f"convergence_{run_id}.pdf")
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
