"""Training orchestration with per-run diagnostics and checkpoint I/O.

Checkpoint formats (`.pt` component state_dicts, `.npz` meta keys) and the
D-076 pre-JE resume guard are a frozen contract, unchanged from pre-refactor.
"""
__project_origin__ = "added"

import os
import json
import time

import numpy as np
import torch

from .config import RunConfig
from .model import train_model
from .diagnostics import aug_state_r2


def save_checkpoint_weights(fit_sys, base_path):
    """Save the torch components (SSE_Interconnect is a deepSI System, no state_dict). D-070."""
    torch.save({
        'hfn':       fit_sys.hfn.state_dict(),
        'encoder':   fit_sys.encoder.state_dict(),
        'optimizer': fit_sys.optimizer.state_dict(),
    }, base_path + '.pt')


def load_checkpoint(fit_sys, base_path, joint_estimation):
    """Restore weights + optimizer from a checkpoint; return the meta npz handle."""
    meta = np.load(base_path + '.npz', allow_pickle=True)
    # D-070: SSE_Interconnect has no state_dict; checkpoint holds component state_dicts
    ckpt = torch.load(base_path + '.pt', map_location='cpu')
    # D-076: JE checkpoints carry log_params; pre-JE checkpoints cannot resume a JE run
    if joint_estimation and not any('log_params' in k for k in ckpt['hfn']):
        raise RuntimeError(
            'RESUME_CHECKPOINT points at a pre-JE checkpoint (no log_params); '
            'JOINT_ESTIMATION runs must start from fresh checkpoints (D-076)')
    fit_sys.hfn.load_state_dict(ckpt['hfn'])
    fit_sys.encoder.load_state_dict(ckpt['encoder'])
    if 'optimizer' in ckpt:
        fit_sys.optimizer.load_state_dict(ckpt['optimizer'])
    return meta


def _install_nf_val_probe(fit_sys, hp):
    """Piggyback an nf-window RMS on each epoch's validation (D-095).

    deepSI validates once per epoch through `self.cal_validation_error`
    (concurrent_val=False path). We wrap that instance method: return the
    selector value untouched, and also record the nf-window RMS (same nf as
    training, encoder re-init per window) into `fit_sys.Loss_val_nf`, aligned
    with `fit_sys.Loss_val`. Physical-meter RMS (mode='RMS'), so it is directly
    comparable to the 'sim-RMS' selector. Non-overlapping windows (stride=nf)
    keep the probe at ~one sim-pass. Returns the original method to restore.
    """
    nf = hp['nf']
    probe_stride = max(1, nf)   # non-overlapping windows -> ~one sim-pass cost
    fit_sys.Loss_val_nf = []
    orig = fit_sys.cal_validation_error

    def wrapped(val_sys_data, validation_measure='sim-NRMS'):
        sel = orig(val_sys_data, validation_measure=validation_measure)  # selector, untouched
        try:
            with torch.no_grad():
                e = fit_sys.n_step_error(val_sys_data, nf=nf, stride=probe_stride,
                                         mode='RMS', mean_channels=True)
            fit_sys.Loss_val_nf.append(float(np.mean(e)))
        except Exception:
            fit_sys.Loss_val_nf.append(float('nan'))
        return sel

    fit_sys.cal_validation_error = wrapped
    return orig


def train_model_with_diagnostics(fit_sys, hp, cfg: RunConfig, data, norm,
                                 resume_ckpt=None, checkpoint_dir=None, run_id=None):
    """Train for hp['epochs'] epochs with sim-RMS validation; record aug-state R2 after.

    resume_ckpt : base path (no extension) of a prior checkpoint to resume from.
    checkpoint_dir : directory for checkpoint; None = no saving.
    """
    diag_epochs, diag_r2_raw, diag_r2_lin = [], [], []
    done_epochs = 0
    bestfit = float('inf')

    # --- Resume ---
    if resume_ckpt is not None:
        meta = load_checkpoint(fit_sys, resume_ckpt, cfg.joint_estimation)
        done_epochs = int(meta['done_epochs'])
        bestfit     = float(meta['bestfit'])
        diag_epochs = list(meta['diag_epochs'])
        diag_r2_raw = [meta['diag_r2_raw'][i]  for i in range(len(meta['diag_r2_raw']))]
        diag_r2_lin = [meta['diag_r2_linmap'][i] for i in range(len(meta['diag_r2_linmap']))]
        print(f'Resumed from {resume_ckpt}  ({done_epochs} epochs done)')

    epochs_remaining = hp['epochs'] - done_epochs
    print(f'\nTraining: nf={hp["nf"]}  epochs={hp["epochs"]}  val=sim-RMS  NX_ANN={hp["NX_ANN"]}')
    if done_epochs > 0:
        print(f'  Resuming: {done_epochs} done, {epochs_remaining} remaining')

    fit_sys.bestfit = float('inf')
    t0 = time.time()
    # D-095: piggyback the nf-window RMS diagnostic; selection stays full-traj sim-RMS.
    _orig_cve = _install_nf_val_probe(fit_sys, hp)
    try:
        bestfit = train_model(fit_sys, hp, cfg, data,
                              epochs=epochs_remaining,
                              nf=hp['nf'],
                              validation_measure='sim-RMS')
    finally:
        fit_sys.cal_validation_error = _orig_cve   # restore always
    loss_val_nf = np.array(getattr(fit_sys, 'Loss_val_nf', []), dtype=float)
    elapsed = time.time() - t0
    done_epochs = hp['epochs']

    _fin = loss_val_nf[np.isfinite(loss_val_nf)] if loss_val_nf.size else loss_val_nf
    if _fin.size:
        print(f'  val nf-window RMS ({hp["nf"]}-step): first={loss_val_nf[0]:.4e}  '
              f'best={_fin.min():.4e}  last={loss_val_nf[-1]:.4e} [m]  (sim-RMS selector unchanged)')

    # --- Checkpoint weights first: diagnostics below must not be able to lose them (D-070)
    ckpt_base = None
    if checkpoint_dir is not None:
        ckpt_base = os.path.join(checkpoint_dir, f'gantry_ckpt_{run_id}')
        save_checkpoint_weights(fit_sys, ckpt_base)
        print(f'  Checkpoint weights: {ckpt_base}.pt')

    fit_sys.eval()
    try:
        r2_raw, r2_lin = aug_state_r2(fit_sys, hp, cfg, data, norm)
    except Exception as e:
        print(f'Warning: aug_state_r2 failed ({e}); recording NaN')
        r2_raw = np.full(hp['NX_ANN'], np.nan)
        r2_lin = np.full(hp['NX_ANN'], np.nan)
    diag_epochs.append(done_epochs)
    diag_r2_raw.append(r2_raw.copy())
    diag_r2_lin.append(r2_lin.copy())

    r2_str = '  '.join([f'{n}={r2_lin[i]:+.4f}'
                        for i, n in enumerate(['delta_a', 'vdelta_a'][:hp['NX_ANN']])])
    print(f'Training done | {done_epochs} ep | {elapsed:.0f}s | '
          f'bestfit={bestfit:.5f}  R2_linmap: {r2_str}')

    # --- Checkpoint metadata (weights already saved above) ---
    if ckpt_base is not None:
        np.savez(ckpt_base + '.npz',
                 done_epochs    = np.array(done_epochs),
                 bestfit        = np.array(bestfit),
                 elapsed        = np.array(elapsed),
                 diag_epochs    = np.array(diag_epochs),
                 diag_r2_raw    = np.array(diag_r2_raw),
                 diag_r2_linmap = np.array(diag_r2_lin),
                 hp             = np.array(json.dumps(hp)),
                 orig_run_id    = np.array(run_id),
        )
        print(f'  Checkpoint meta: {ckpt_base}.npz')

    return bestfit, dict(
        epochs      = np.array(diag_epochs),
        r2_raw      = np.array(diag_r2_raw),
        r2_linmap   = np.array(diag_r2_lin),
        loss_val_nf = loss_val_nf,   # D-095: per-epoch nf-window RMS (aligns with Loss_val tail)
    )
