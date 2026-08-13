"""Diagnostics: R2 helpers, encoder-window state recovery, gradient norms.

The R2 formula, the encoder-window builder, and the affine-map R2 are shared
here (they appeared 2-3 times pre-refactor with identical bodies). All numerics
are verbatim; the extractions are the only structural change.
"""
__project_origin__ = "added"

import os

import numpy as np
import torch
import torch.nn as nn

from .config import RunConfig
from .model import get_encoder_dims


def r2_per_channel(ref, est):
    """Per-channel coefficient of determination between reference and estimate."""
    # THEORY: R^2 = 1 - SS_res/SS_tot (standard OLS); 1e-12 guards constant channels
    ss_res = ((ref - est) ** 2).sum(axis=0)
    ss_tot = ((ref - ref.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def best_affine_r2(X, target, dtype):
    """Best affine map target ~ [X, 1] @ W and its per-channel R2.

    # THEORY: ordinary least squares -- uses all X channels jointly.
    """
    A = np.hstack([X, np.ones((len(X), 1), dtype=dtype)])
    W, *_ = np.linalg.lstsq(A, target, rcond=None)
    return W, r2_per_channel(target, A @ W)


def encoder_state_estimates(fit_sys, sysdata, na, nb, na_right, nb_right, cfg: RunConfig,
                            max_windows=2000):
    """Run the encoder on strided validation windows -> (k_ix, stride, x_hat).

    deepSI hist convention: ypast = y[k-na : k+na_right], upast = u[k-nb : k+nb_right];
    the encoder output initializes the state at time k.
    """
    DTYPE_NP, DTYPE_PT = cfg.dtype_np, cfg.dtype_pt
    val_norm = fit_sys.norm.transform(sysdata)
    yn = np.ascontiguousarray(val_norm.y, dtype=DTYPE_NP)
    un = np.ascontiguousarray(val_norm.u, dtype=DTYPE_NP)
    N = len(yn)
    k0 = max(na, nb) + 1                                      # +1 so k-1 exists for the lag column
    stride = max(1, (N - k0) // max_windows)                  # HEURISTIC: cap window count to bound memory
    k_ix = np.arange(k0, N, stride)
    ypast = np.stack([yn[k - na : k + na_right] for k in k_ix])  # (Nk, na+na_right, ny)
    upast = np.stack([un[k - nb : k + nb_right] for k in k_ix])  # (Nk, nb+nb_right, nu)
    with torch.no_grad():
        x_hat = fit_sys.encoder(
            torch.tensor(upast, dtype=DTYPE_PT),
            torch.tensor(ypast, dtype=DTYPE_PT),
        ).numpy()                                             # (Nk, nxd)
    return k_ix, stride, x_hat


def aug_state_r2(fit_sys, hp, cfg: RunConfig, data, norm):
    """Encoder augmented state R2 vs delta_a / vdelta_a from mat file.

    Runs the encoder on strided validation windows and computes:
      R2_raw    -- direct comparison (encoder output vs normalized GT)
      R2_linmap -- best affine map from ALL encoder outputs to GT channel
                   (catches arbitrary scale/offset, shows information content)

    Both quantities returned as arrays of shape (NX_ANN,).
    """
    NX_PHYS = cfg.nx_phys
    na, nb, na_right, nb_right = get_encoder_dims(hp, cfg)
    fit_sys.eval()

    k_ix, stride, x_hat = encoder_state_estimates(
        fit_sys, data.val_data, na, nb, na_right, nb_right, cfg)   # (Nk, NX_PHYS + NX_ANN)

    x_ann = x_hat[:, NX_PHYS:]                    # (Nk, NX_ANN)

    # Normalize GT so encoder (dimensionless) and GT are on comparable axes
    gt_raw  = data.val_x_aug[k_ix]                # (Nk, NX_ANN) physical units
    gt_mean = gt_raw.mean(axis=0)
    gt_std  = gt_raw.std(axis=0) + 1e-8
    gt_norm = (gt_raw - gt_mean) / gt_std          # (Nk, NX_ANN) normalized

    r2_raw = r2_per_channel(gt_norm, x_ann)

    # Best affine map: gt_norm ~ [x_ann_all_channels, 1] @ W (per GT channel)
    W_aug, r2_lin = best_affine_r2(x_ann, gt_norm, cfg.dtype_np)

    return r2_raw, r2_lin


def state_recovery_diagnostic(fit_sys, hp, rid, cfg: RunConfig, data, norm, save_dir,
                              max_windows=2000):
    """Linear-map state recovery test: basis rotation vs lost information.

    Compares encoder state estimates x_hat(k) on the validation set against
    physical states reconstructed from measurements:
      R2_raw      x_hat[:, :6] read directly as normalized physical states
      R2_linmap   best least-squares linear map x_true ~ x_hat @ W + b
      R2_raw_lag1 raw comparison against x_true(k-1) (detects one-sample lag)
    Interpretation:
      R2_linmap ~ 1, R2_raw low  -> information present, basis rotated
      R2_linmap low              -> information absent from encoder state
      R2_raw_lag1 > R2_raw       -> encoder aligned to k-1 (history off-by-one)
    """
    NX_PHYS = cfg.nx_phys
    NX_ANN = hp['NX_ANN']
    nxd = NX_PHYS + NX_ANN
    na, nb, na_right, nb_right = get_encoder_dims(hp, cfg)
    fit_sys.eval()
    DTYPE_NP = cfg.dtype_np
    x_mean, std_x = norm.x_mean, norm.std_x

    # True physical states from augmented simulation mat file (more accurate than P_inv+FD)
    x_true = data.val_x_logical.astype(DTYPE_NP)                             # (N,6) x_logical from mat
    x_true_norm = (x_true - x_mean.flatten()) / std_x.flatten()             # (N,6)

    # Encoder estimates. deepSI hist convention: the encoder output initializes
    # the state at time k. k0 = max(na,nb)+1 so k-1 exists for the lag column.
    k_ix, stride, x_hat = encoder_state_estimates(
        fit_sys, data.val_data, na, nb, na_right, nb_right, cfg, max_windows)

    xt   = x_true_norm[k_ix]                                  # x_true(k)
    xt_l = x_true_norm[k_ix - 1]                              # x_true(k-1)

    r2_raw = r2_per_channel(xt,   x_hat[:, :NX_PHYS])
    r2_lag = r2_per_channel(xt_l, x_hat[:, :NX_PHYS])

    # Best affine map x_true ~ [x_hat, 1] @ W
    W, r2_lin = best_affine_r2(x_hat, xt, cfg.dtype_np)

    labels = ['q1 ', 'q2 ', 'q3 ', 'dq1', 'dq2', 'dq3']
    print('\n=== State recovery diagnostic (D-053) ===')
    print(f'  {len(k_ix)} windows (stride {stride}), na=nb={na}, '
          f'encoder={cfg.encoder_init}')
    print('  channel   R2_raw      R2_linmap   R2_raw_lag1')
    for ch in range(NX_PHYS):
        print(f'  {labels[ch]}     {r2_raw[ch]:+10.4f}  {r2_lin[ch]:+10.4f}  {r2_lag[ch]:+10.4f}')
    print('  R2_linmap ~ 1 & R2_raw low -> basis rotation;')
    print('  R2_linmap low              -> information absent from encoder state;')
    print('  R2_raw_lag1 > R2_raw       -> encoder aligned to k-1 (one-sample lag)')

    # --- Augmented state R2 vs delta_a / vdelta_a ---
    r2_aug_raw, r2_aug_lin = aug_state_r2(fit_sys, hp, cfg, data, norm)
    aug_labels = ['delta_a  ', 'vdelta_a ']
    aug_notes  = ['(mat file)', '(FD estimate)']
    print('\n=== Augmented state R2 vs saved GT (delta_a/vdelta_a from mat file) ===')
    print(f'  {"state":<12s}  {"R2_raw":>10s}  {"R2_linmap":>10s}  note')
    for ch in range(hp['NX_ANN']):
        lbl  = aug_labels[ch] if ch < len(aug_labels) else f'x_ann[{ch}]'
        note = aug_notes[ch]  if ch < len(aug_notes)  else ''
        print(f'  {lbl}  {r2_aug_raw[ch]:+10.4f}  {r2_aug_lin[ch]:+10.4f}  {note}')
    print('  R2_linmap ~ 1 -> augmented state captured MSD dynamics')
    print('  R2_linmap ~ 0 -> augmented state did not learn delta_a')

    if cfg.save_flag:
        np.savez(os.path.join(save_dir, f'gantry_state_recovery_{rid}.npz'),
                 r2_raw=r2_raw, r2_lin=r2_lin, r2_lag=r2_lag,
                 W=W, k_ix=k_ix, x_hat=x_hat, x_true_norm=xt,
                 r2_aug_raw=r2_aug_raw, r2_aug_lin=r2_aug_lin)
        print(f'Saved state recovery diagnostic: gantry_state_recovery_{rid}.npz')


def compute_gradient_norms(fit_sys, hp, cfg: RunConfig, data):
    """Single forward+backward pass on training data, return gradient norms per parameter group."""
    DTYPE_PT = cfg.dtype_pt
    fit_sys.train()
    # CHANGED (2026-07-29): pass `stride`, which was missing. deepSI's
    # make_training_data does `stride = Loss_kwargs.get('stride', 1)`, so omitting
    # it built EVERY window instead of every cfg.stride-th, and then kept 256.
    # Two consequences, both real:
    #   MEMORY. Training at nf=6400/stride=100 builds 5824 windows (855 MB). At
    #   stride=1 that is ~582,000 windows, ~86 GB, against a 32 GB job. This
    #   SIGKILLed the T1 rungs nf=1600, 3200 and 6400 at the very end of the run,
    #   after training and the per-record NRMS had completed, so all three lost
    #   their results.json. 400 and 800 were under the limit and survived.
    #   MEANING. The slice below takes the FIRST batch_size windows. At stride=1
    #   those are offsets 0,1,2,... of record 1, i.e. 256 near-identical windows
    #   overlapping by nf-1 samples, so the gradient norm was measured on an
    #   almost degenerate sample. With the training stride they span 100x more
    #   data. Numbers reported by this function before this date, on long-horizon
    #   runs, describe that degenerate sample.
    # Verified by preflight/check_grad_norm_stride.py.
    data_train = fit_sys.make_training_data(fit_sys.norm.transform(data.train_data),
                                            nf=hp['nf'], stride=cfg.stride)
    batch_size = min(hp['batch_size'], len(data_train[0]))
    batch = [torch.tensor(d[:batch_size], dtype=DTYPE_PT) for d in data_train]

    fit_sys.optimizer.zero_grad()
    loss = fit_sys.loss(*batch, nf=hp['nf'])
    loss.backward()

    grad_norms = {}
    for attr_name, item in fit_sys.parameters_with_names.items():
        params = item['params']
        if isinstance(params, nn.Parameter):
            params = [params]
        else:
            params = list(params)
        for i, p in enumerate(params):
            pname = f'{attr_name}.{i}'
            if p.grad is not None:
                grad_norms[pname] = float(torch.norm(p.grad).item())
            else:
                grad_norms[pname] = 0.0

    # Aggregate by parameter group (encoder, hfn.blocks.0=physics, hfn.blocks.2=ANN)
    group_norms = {}
    for name, norm_val in grad_norms.items():
        if name.startswith('encoder'):
            group = 'encoder'
        elif name.startswith('hfn'):
            group = 'hfn'
        else:
            group = 'other'
        group_norms[group] = group_norms.get(group, 0.0) + norm_val ** 2
    group_norms = {k: float(np.sqrt(v)) for k, v in group_norms.items()}

    print('\n=== Gradient norms (single batch, post-training) ===')
    for group, norm_val in sorted(group_norms.items()):
        print(f'  {group:20s}: {norm_val:.4e}')

    return grad_norms, group_norms


def encoder_init_state(fit_sys, sysdata, K0, na, nb, na_right, nb_right, cfg: RunConfig):
    """Untrained linear-map encoder estimate of x(K0) from the first I/O window (D-072).

    Runs BEFORE training, so fit_sys.encoder is still the pure reconstructability
    map built from the baseline linearization -- a baseline-only quantity.
    """
    DTYPE_NP, DTYPE_PT = cfg.dtype_np, cfg.dtype_pt
    dn = fit_sys.norm.transform(sysdata)
    yp = np.ascontiguousarray(dn.y, dtype=DTYPE_NP)[K0 - na: K0 + na_right][None]
    up = np.ascontiguousarray(dn.u, dtype=DTYPE_NP)[K0 - nb: K0 + nb_right][None]
    with torch.no_grad():
        x0 = fit_sys.encoder(torch.tensor(up, dtype=DTYPE_PT),
                             torch.tensor(yp, dtype=DTYPE_PT)).numpy()[0]
    return x0[:cfg.nx_phys]
