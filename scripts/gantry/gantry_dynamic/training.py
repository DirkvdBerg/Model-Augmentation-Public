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


def _noop_cve(*args, **kwargs):
    """Placeholder restored when a checkpointed _NfProbe is unpickled (D-095)."""
    return None


def _restore_noop_cve():
    """Unpickle target for _NfProbe: yields the _noop_cve callable (not None)."""
    return _noop_cve


class _NfProbe:
    """Per-epoch train+val nf-window RMS piggybacked on validation (D-095).

    deepSI validates once per epoch through `self.cal_validation_error`
    (concurrent_val=False path). This wraps that instance method: return the
    selector value untouched, and additionally record the nf-window RMS (same nf
    as training, encoder re-init per window, physical meters via mode='RMS') for
    BOTH a train trajectory and the val data into `fit_sys.Loss_train_nf` /
    `Loss_val_nf`, aligned with `fit_sys.Loss_val`. Non-overlapping windows
    (stride=nf) keep each probe at ~one sim-pass.

    A module-level class (not a closure) so `checkpoint_save_system`'s
    `torch.save(self.__dict__)` can pickle `fit_sys` while the probe is installed;
    `__reduce__` serialises it back to a no-op (the probe is transient and
    re-installed each fit). Compare train vs val nf-RMS to read generalization
    (train low/val high) vs long-rollout drift (both bounded, sim-RMS grows).
    """

    def __init__(self, fit_sys, orig, nf, train_sd, val_sd, do_print=True):
        self.fit_sys = fit_sys
        self.orig = orig
        self.nf = nf
        self.stride = max(1, nf)
        self.train_sd = train_sd
        self.val_sd = val_sd
        self.do_print = do_print
        fit_sys.Loss_train_nf = []
        fit_sys.Loss_val_nf = []
        # Joint/orth probe state (user 07-12: live recovery + negation meters).
        # Nominal combos computed ONCE; sim-study meter -- on real data the
        # reference must become params_init (no nominal truth exists there).
        self._pblock = next((m for m in fit_sys.hfn.connected_blocks
                             if hasattr(m, 'identifiable_combinations')), None)
        self._combo_nom = None
        if self._pblock is not None:
            from model_augmentation.systems import gantry_ss as _gss
            _true_raw = {n: getattr(_gss, n).item() for n in self._pblock.PARAM_NAMES}
            self._combo_nom = self._pblock._combos_from_raw(_true_raw,
                                                            self._pblock.Lb.item())
            # Per-combo error scale: |nominal|, EXCEPT m_diff.
            # HEURISTIC: m_diff = m1-m2 has near-zero nominal (-0.5 kg vs ~10 kg
            # masses); relative-to-itself error dominates the RMS meaninglessly
            # (run 70783: -418% -> combo-err 132%). Scale by the mean actuator
            # mass instead, so a 10% detune reads ~20% on m_diff, not 418%.
            self._combo_scale = {n: abs(t) for n, t in self._combo_nom.items()
                                 if abs(t) > 1e-12}
            self._combo_scale['m_diff'] = 0.5 * (_true_raw['m1'] + _true_raw['m2'])
        if self._pblock is not None or getattr(fit_sys, 'orth_penalty', None) is not None:
            fit_sys.Probe_combo_err = []
            fit_sys.Probe_orth_frac = []
            fit_sys.Probe_V_orth = []
            fit_sys.Probe_param_loss = []

    def _nf_rms(self, sd):
        try:
            with torch.no_grad():
                e = self.fit_sys.n_step_error(sd, nf=self.nf, stride=self.stride,
                                              mode='RMS', mean_channels=True)
            return float(np.mean(e))
        except Exception:
            return float('nan')

    def _joint_probe(self):
        """One line: recovery + negation meters. Cost: scalars + one batched ANN
        forward over the fixed penalty points (measured 0.03 s, plan Step 3).
        combo part requires a parameterized block (joint runs); orth part only
        an attached penalty (also fires for joint=False + orth_observe)."""
        fs = self.fit_sys
        combo_err, worst, rels = float('nan'), None, {}
        if self._pblock is not None:
            # combo-err: RMS scaled error of the 10 identifiable combos vs nominal
            # (scale = |nominal| per combo; m_diff uses the mass scale -- __init__).
            combos = self._pblock.identifiable_combinations()
            rels = {n: (combos[n] - self._combo_nom[n]) / s
                    for n, s in self._combo_scale.items()}
            combo_err = float(np.sqrt(np.mean([r ** 2 for r in rels.values()])))
            worst = max(rels, key=lambda n: abs(rels[n]))
        # orth meters: available when a penalty object is attached (beta>0 or observe).
        pen = getattr(fs, 'orth_penalty', None)
        frac, v_orth = float('nan'), float('nan')
        if pen is not None:
            from model_augmentation.fit_systems.blocks import Static_ANN_Block
            ann = next(m for m in fs.hfn.connected_blocks
                       if isinstance(m, Static_ANN_Block))
            with torch.no_grad():
                w = ann(pen.Z_pts)
                f = w[:, pen.route_cols, 0].reshape(-1)
                f2 = float(torch.linalg.vector_norm(f) ** 2)
                q2 = float(torch.linalg.vector_norm(pen.Q.T @ f) ** 2)
            frac = q2 / f2 if f2 > 1e-30 else float('nan')   # n/a while ANN ~ 0
            v_orth = pen.beta * q2
        pl = sum(float(m.param_loss()) for m in fs.hfn.connected_blocks
                 if hasattr(m, 'param_loss'))
        fs.Probe_combo_err.append(combo_err)
        fs.Probe_orth_frac.append(frac)
        fs.Probe_V_orth.append(v_orth)
        fs.Probe_param_loss.append(pl)
        if self.do_print:
            frac_s = f'{frac:.3f}' if np.isfinite(frac) else 'n/a (ANN~0)'
            combo_s = (f'combo-err {100*combo_err:.2f}% (worst {worst} '
                       f'{100*rels[worst]:+.1f}%)' if worst is not None
                       else 'combo-err n/a (theta frozen)')
            print(f'    [joint-probe] {combo_s} | orth-frac {frac_s} | '
                  f'V_orth {v_orth:.3e} | param_loss {pl:.3e}')

    def __call__(self, val_sys_data, validation_measure='sim-NRMS'):
        sel = self.orig(val_sys_data, validation_measure=validation_measure)  # selector, untouched
        tr = self._nf_rms(self.train_sd)
        vl = self._nf_rms(self.val_sd)
        self.fit_sys.Loss_train_nf.append(tr)
        self.fit_sys.Loss_val_nf.append(vl)
        if self.do_print:
            print(f'    [nf-probe] train nf-RMS={tr:.4e}   val nf-RMS={vl:.4e} [m]  (@nf={self.nf})')
        if (self._pblock is not None
                or getattr(self.fit_sys, 'orth_penalty', None) is not None):
            try:
                self._joint_probe()
            except Exception as e:
                print(f'    [joint-probe] failed (non-fatal): {e}')
        return sel

    def __reduce__(self):
        return (_restore_noop_cve, ())


def _install_nf_val_probe(fit_sys, hp, cfg, train_sd, val_sd):
    """Install an `_NfProbe` on `fit_sys.cal_validation_error`; return the original to restore."""
    orig = fit_sys.cal_validation_error
    fit_sys.cal_validation_error = _NfProbe(
        fit_sys, orig, hp['nf'], train_sd, val_sd, do_print=cfg.nf_probe_print)
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
    # D-095: piggyback the nf-window RMS diagnostic (train + val); selection stays full-traj sim-RMS.
    _orig_cve = _install_nf_val_probe(fit_sys, hp, cfg, data.train_list[0], data.val_ckpt_data)
    try:
        bestfit = train_model(fit_sys, hp, cfg, data,
                              epochs=epochs_remaining,
                              nf=hp['nf'],
                              validation_measure='sim-RMS')
    finally:
        fit_sys.cal_validation_error = _orig_cve   # restore always
    loss_val_nf   = np.array(getattr(fit_sys, 'Loss_val_nf', []), dtype=float)
    loss_train_nf = np.array(getattr(fit_sys, 'Loss_train_nf', []), dtype=float)
    elapsed = time.time() - t0
    done_epochs = hp['epochs']

    _fin = loss_val_nf[np.isfinite(loss_val_nf)] if loss_val_nf.size else loss_val_nf
    if _fin.size:
        print(f'  val nf-window RMS ({hp["nf"]}-step): first={loss_val_nf[0]:.4e}  '
              f'best={_fin.min():.4e}  last={loss_val_nf[-1]:.4e} [m]  (sim-RMS selector unchanged)')
    _fintr = loss_train_nf[np.isfinite(loss_train_nf)] if loss_train_nf.size else loss_train_nf
    if _fintr.size:
        print(f'  train nf-window RMS ({hp["nf"]}-step): first={loss_train_nf[0]:.4e}  '
              f'best={_fintr.min():.4e}  last={loss_train_nf[-1]:.4e} [m]')

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
        loss_val_nf   = loss_val_nf,    # D-095: per-epoch val nf-window RMS (aligns with Loss_val tail)
        loss_train_nf = loss_train_nf,  # D-095: per-epoch train nf-window RMS (same horizon, meters)
    )
