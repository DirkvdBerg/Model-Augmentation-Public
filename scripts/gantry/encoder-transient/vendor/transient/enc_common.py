"""Shared harness for the encoder-only experiments (D-197).

Everything here is the PRODUCTION path, imported and not re-implemented:

  * the run configuration is `gantry_interconnect_dynamic.CFG` itself, so the encoder under
    test is the encoder that run trains (same `na_nb`, `nx_ann`, `encoder_init`, seed, dtype),
  * the model is built by `gantry_dynamic.model.build_model`,
  * the window convention is `gantry_dynamic.diagnostics.encoder_state_estimates`'s:
    `ypast = y[k-na : k+na_right]`, `upast = u[k-nb : k+nb_right]`, and the encoder's output
    initialises the state at time k,
  * the I/O normalisation is `fit_sys.norm`, and the state frame is `(x - x_mean) / std_x`
    with the pipeline's own constants, which is the frame the encoder outputs in.

The one thing that is NOT the production path is the TARGET: the exact RK4 truth from
`truth.py` rather than the record's finite-difference `x_logical` (D-197).

SCOPE. Only the six physical channels are scored. The `nx_ann` augmented channels are an
arbitrary gauge with no ground truth, so they are reported for liveness only and never fitted.
"""
__project_origin__ = "added"

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# VENDORED: the vendor dir (gantry_dynamic, model_augmentation, the entry file) replaces REPO.
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', '..'))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
sys.path.insert(0, HERE)

from truth import exact_truth, gate_replay                       # noqa: E402

CHANNELS = ['X', 'Theta', 'Y', 'dX', 'dTheta', 'dY']
UNITS = ['m', 'rad', 'm', 'm/s', 'rad/s', 'm/s']
FIGDIR = os.path.join(HERE, '..', '..', 'outputs', 'vendored_r0', 'figure')   # VENDORED
OUTDIR = os.path.join(HERE, '..', '..', 'outputs', 'vendored_r0', 'results')  # VENDORED


def build(overrides=None, quiet=True):
    """Build the production model. Returns (cfg, data, norm, fit_sys, hp, dims).

    `overrides` are applied to the entry file's own RunConfig with `dataclasses.replace`, so
    anything not named here is exactly what a production run uses.
    """
    from dataclasses import replace
    import gantry_interconnect_dynamic as entry                  # noqa: E402
    from gantry_dynamic.data import load_datasets, compute_normalization
    from gantry_dynamic.model import build_model, get_encoder_dims

    cfg = entry.CFG
    # The encoder test never trains the rollout, so it never needs the GPU for the model
    # itself, and a CPU build keeps `build_model`'s device assertion out of the way on any
    # machine. R1 moves the ENCODER alone to the GPU.
    # compile_mode is wired for the CUDA training rollout only, and RunConfig refuses it on the
    # CPU. Nothing here ever calls that rollout, so it is switched off with the device.
    # VENDORED: pin the D-197 dataset; the working-tree entry file names the friction folder.
    cfg = replace(cfg, mode='augmentation_ma50_b140-230_a6_z03')
    cfg = replace(cfg, device='cpu', compile_mode=None, save_flag=False, **(overrides or {}))

    stdout = sys.stdout
    if quiet:
        sys.stdout = open(os.devnull, 'w')
    try:
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        data = load_datasets(cfg)
        norm = compute_normalization(cfg, data)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        fit_sys = build_model(cfg.hp, cfg, data, norm)
    finally:
        if quiet:
            sys.stdout.close()
            sys.stdout = stdout
    return cfg, data, norm, fit_sys, cfg.hp, get_encoder_dims(cfg.hp, cfg)


def record_windows(name, cfg, norm, dims, stride=1, k_lo=None, k_hi=None, gate=False):
    """Encoder inputs and the EXACT target for one record.

    Returns dict with
      upast (Nk, nb+nb_right, nu), ypast (Nk, na+na_right, ny)  encoder inputs, pipeline norm
      xt    (Nk, 6)  exact truth at k, in the encoder's frame (x - x_mean)/std_x
      xfd   (Nk, 6)  the record's finite-difference x_logical at k, same frame (for contrast)
      k_ix  (Nk,)    sample index the encoder output initialises
    """
    from gantry_dynamic.data import load_traj

    na, nb, na_right, nb_right = dims
    res = exact_truth(name[:-4] if name.endswith('.mat') else name, cfg.mode)
    if gate:
        gate_replay(res)

    sysdata = load_traj(name if name.endswith('.mat') else name + '.mat', cfg)
    sd = fit_norm_transform(sysdata, norm, cfg)
    yn, un = sd['y'], sd['u']
    N = len(yn)
    # +1 so that k-1 exists, matching diagnostics.encoder_state_estimates exactly.
    k0 = max(na, nb) + 1 if k_lo is None else k_lo
    k1 = N - max(na_right, nb_right) if k_hi is None else k_hi
    k_ix = np.arange(k0, k1, stride)

    yw = np.lib.stride_tricks.sliding_window_view(yn, na + na_right, axis=0)   # (M, ny, W)
    uw = np.lib.stride_tricks.sliding_window_view(un, nb + nb_right, axis=0)   # (M, nu, W)
    ypast = np.ascontiguousarray(yw[k_ix - na].transpose(0, 2, 1))
    upast = np.ascontiguousarray(uw[k_ix - nb].transpose(0, 2, 1))

    x_mean, std_x = norm.x_mean.flatten(), norm.std_x.flatten()
    xt = ((res['x6'][k_ix] - x_mean) / std_x).astype(cfg.dtype_np)
    xfd = ((res['rec']['x_logical'][k_ix] - x_mean) / std_x).astype(cfg.dtype_np)
    return dict(name=name, upast=upast.astype(cfg.dtype_np), ypast=ypast.astype(cfg.dtype_np),
                xt=xt, xfd=xfd, k_ix=k_ix, gate=res['gate'],
                x_aug_true=res['x8'][k_ix][:, [3, 7]])


def fit_norm_transform(sysdata, norm, cfg):
    """(y - y0)/ystd and (u - u_mean)/std_u, the transform `fit_sys.norm` applies."""
    y = (np.asarray(sysdata.y) - norm.y0) / norm.ystd
    u = (np.asarray(sysdata.u) - norm.u_mean.flatten()) / norm.std_u.flatten()
    return dict(y=np.ascontiguousarray(y, dtype=cfg.dtype_np),
                u=np.ascontiguousarray(u, dtype=cfg.dtype_np))


def encoder_forward(encoder, upast, ypast, cfg, batch=8192, device='cpu'):
    """x_hat = encoder(upast, ypast), batched, no grad. Returns (Nk, nx_phys + nx_ann)."""
    out = []
    encoder = encoder.to(device)
    with torch.no_grad():
        for i in range(0, len(upast), batch):
            u = torch.as_tensor(upast[i:i + batch], dtype=cfg.dtype_pt, device=device)
            y = torch.as_tensor(ypast[i:i + batch], dtype=cfg.dtype_pt, device=device)
            out.append(encoder(u, y).cpu().numpy())
    return np.concatenate(out, axis=0)


def scores(x_hat_phys, xt, std_x):
    """Per-channel R2, NRMS and RMS error in physical units, all against the same target."""
    e = x_hat_phys - xt
    var = xt.var(axis=0) + 1e-30
    return dict(
        r2=1.0 - (e ** 2).mean(axis=0) / var,
        nrms=np.sqrt((e ** 2).mean(axis=0)) / np.sqrt(var),
        rms_phys=np.sqrt((e ** 2).mean(axis=0)) * std_x,
        bias_phys=e.mean(axis=0) * std_x,
        ref_std_phys=np.sqrt(var) * std_x,
    )


def print_scores(title, sc, extra=None):
    print(f'\n  {title}')
    print(f'    {"channel":<8}{"R2":>10}{"NRMS":>10}{"RMS err":>13}{"bias":>13}'
          f'{"signal std":>13}   unit')
    for i, ch in enumerate(CHANNELS):
        print(f'    {ch:<8}{sc["r2"][i]:>10.4f}{sc["nrms"][i]:>10.4f}'
              f'{sc["rms_phys"][i]:>13.3e}{sc["bias_phys"][i]:>13.3e}'
              f'{sc["ref_std_phys"][i]:>13.3e}   {UNITS[i]}')
    if extra:
        print(f'    {extra}')


def ensure_dirs():
    for d in (FIGDIR, OUTDIR):
        os.makedirs(d, exist_ok=True)
