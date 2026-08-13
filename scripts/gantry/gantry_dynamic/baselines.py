"""Baseline FP-model open-loop simulation (no MSD, no ANN).

The generic step-by-step rollout is shared with the x_logical-init sim in
evaluation.py. Per-step operations are verbatim from the pre-refactor loop.
"""
__project_origin__ = "added"

import os

import numpy as np
import torch

from model_augmentation.fit_systems.blocks import Gantry_State_Block

from .config import RunConfig


def stepwise_rollout(step_fn, x0, u_seq):
    """Open-loop rollout: at each t, y_t, x = step_fn(x, u_seq[t]); collect y_t.

    step_fn computes the output from the CURRENT state/input, then returns the
    next state -- matching the pre-refactor loops (output before state update).
    """
    y_list = []
    x = x0
    with torch.no_grad():
        for t in range(len(u_seq)):
            y_t, x = step_fn(x, u_seq[t])
            y_list.append(y_t)
    return y_list


def compute_baseline_fp_nrms(hp, cfg: RunConfig, data, norm, data_sd=None, x0_phys=None,
                             x0_norm=None, start_ix=0, avg_from=0, label='val',
                             phy_block=None):
    """Simulate baseline FP model (no MSD, no ANN) on val data (default) or given data.

    Initialization (D-072):
      x0_phys  physical initial state (default: true x0 from val_x_logical) — 'true x0'
      x0_norm  normalized initial state (overrides x0_phys) — for encoder-init baselines
      start_ix simulate from data sample start_ix onward (encoder init estimates x(k0))
      avg_from average the error only from this sample of the simulated window
               (aligns with the model metric, which excludes the encoder warm-up)
      phy_block optional pre-built state block. None (default) = plain
               Gantry_State_Block at nominal theta (behavior unchanged).
               Pass a trained Parameterized_Gantry_State_Block to run the FP
               model STANDALONE with learned theta_hat — the negation test
               (orthogonal-projection plan Step 10, D7.9 layer 2).

    Runs Gantry_State_Block alone (zero ANN contribution) starting from the
    true state (callers pass x_logical[K0], the first interior sample, D-087).
    Compares to val_data.y which contains the augmented simulation output
    (y WITH MSD effect). The NRMS gap measures how much the hidden MSD
    degrades the baseline-only prediction -- the trained augmented model must
    beat this to justify augmentation.

    Returns:
        nrms_baseline  (ny,)  per-channel NRMS
        y_hat_baseline (N, ny) simulated y in physical units [m]
    """
    NX_PHYS, nu = cfg.nx_phys, cfg.nu
    TS_NEW = cfg.ts_new
    DTYPE_NP, DTYPE_PT = cfg.dtype_np, cfg.dtype_pt
    std_x, std_u = norm.std_x, norm.std_u
    x_mean, u_mean = norm.x_mean, norm.u_mean
    Cd_norm, Dd_np = norm.Cd_norm, norm.Dd_np
    ystd, y0 = norm.ystd, norm.y0

    if data_sd is None:
        data_sd = data.val_data

    if phy_block is None:
        phy_block = Gantry_State_Block(
            Y_op=None, std_x=std_x, std_u=std_u,
            x_mean=x_mean, u_mean=u_mean, Ts=TS_NEW,
            up_sample=hp['up_sample'],
        ).to(DTYPE_PT)
    phy_block.eval()

    # Initial state: normalized estimate (encoder-init) or physical true x0 (D-072)
    if x0_norm is not None:
        x_norm_np = np.asarray(x0_norm, dtype=DTYPE_NP).flatten()  # (NX_PHYS,)
    else:
        if x0_phys is None:
            x0_phys = data.val_x_logical[0]
        x0_phys   = np.asarray(x0_phys, dtype=DTYPE_NP)
        x_norm_np = (x0_phys - x_mean.flatten()) / std_x.flatten()  # (NX_PHYS,)

    u_val_norm = ((data_sd.u[start_ix:] - u_mean.flatten()) / std_u.flatten()).astype(DTYPE_NP)

    def _fp_step(x_norm_np_t, u_norm_np):
        # Output: y_norm = Cd_norm @ x_norm + Dd_np @ u_norm (Dd_np ~ 0)
        y_norm = Cd_norm @ x_norm_np_t + Dd_np @ u_norm_np   # (ny,)
        y_phys = y_norm * ystd + y0
        # State transition: x_{k+1} = phy_block(x_k, u_k)
        x_t = torch.tensor(x_norm_np_t, dtype=DTYPE_PT).view(1, NX_PHYS, 1)
        u_t = torch.tensor(u_norm_np, dtype=DTYPE_PT).view(1, nu, 1)
        z   = torch.cat([x_t, u_t], dim=1)     # (1, NX_PHYS+nu, 1)
        x_norm_next = phy_block(z)              # (1, NX_PHYS, 1) or (1, NX_PHYS)
        return y_phys, x_norm_next.view(NX_PHYS).cpu().numpy()

    y_hat_list = stepwise_rollout(_fp_step, x_norm_np, u_val_norm)

    y_hat = np.array(y_hat_list, dtype=DTYPE_NP)   # (N-start_ix, ny)
    y_ref  = data_sd.y[start_ix:]
    nrms   = np.sqrt(((y_hat[avg_from:] - y_ref[avg_from:])**2).mean(axis=0)) / ystd

    rms = nrms * ystd   # [m]
    # THEORY: deepSI System_data.RMS — sqrt(mean sq. error over all samples AND channels)
    rms_agg = float(np.sqrt(np.mean(rms ** 2)))
    print(f'\n=== Baseline FP model ({label}, no MSD, reference to beat) ===')
    print(f"  {'':4s}  {'NRMS':>8s}  {'RMS':>11s}")
    for ch, lbl in enumerate(['X1', 'X2', 'Y ']):
        print(f'  {lbl}:  {nrms[ch]:8.4f}  {rms[ch]:.3e} m')
    print(f'  aggregate sim-RMS (same formula as validation loss): {rms_agg:.4e} m')

    return nrms, y_hat
