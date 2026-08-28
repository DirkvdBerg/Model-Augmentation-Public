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
from model_augmentation.fit_systems.interconnect import WindowErrorStats


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


class _NfProbe:
    """Per-epoch nf-window error statistics, train and val, recorded alongside validation (D-095).

    CHANGED (2026-08-28): both numbers came from deepSI's `System.n_step_error`, the last deepSI
    call on the training path, and both were wrong for the comparison they invited:
      - it drives the model through `measure_act_multi`, i.e. OPEN LOOP always, never
        `self.simulate`. In a closed-loop run it measured a different system than the one being
        optimised, printed next to a closed-loop `bestfit`, unlabelled;
      - the train side was ONE trajectory, `train_list[0]` = T1_standstill_Ym30, the least
        excited record, standing in for 14;
      - it returned `np.mean` over the n = 1..nf growth curve, not the RMS over an nf window;
      - and each side cost a full extra pass per epoch.

    Now: the TRAIN statistics come from `fit_sys.loss_stats`, accumulated inside loss() over
    every training window of all 14 trajectories, through whatever rollout the run uses, at no
    extra compute. The VAL statistics come from ONE fixed 256-window batch through
    `fit_sys.simulate`, so both sides are in the run's own loop mode by construction.

    That val number is worth its ~1/260-of-an-epoch because it DECOMPOSES the reading:
    train-nf vs val-nf is generalisation at fixed horizon, val-nf vs val-sim is horizon at fixed
    generalisation, and train-nf vs val-sim (the only comparison available before) is both at once.

    The train row is a RUNNING MEAN over weights that moved during the epoch, not a snapshot. For
    the mismatch reading the bias is conservative: it reads slightly high, which makes the gap
    look smaller, so it cannot manufacture a mismatch that is not there.

    # CHANGED (closed-loop seam): this was a MONKEY PATCH on `fit_sys.cal_validation_error`,
    # installed on every production run, wrapping the original and returning its value untouched.
    # It is now an entry in `fit_sys.validation_probes`, called for its side effects only.
    # Three things go away with the patch:
    #   - the ordering hazard. Two different patches wanted the same attribute (this one and the
    #     closed-loop validator), and whichever was installed LAST decided checkpoint selection.
    #     A probe cannot decide selection any more: the seam ignores its return value.
    #   - the `__reduce__` no-op. A patched bound method cannot be pickled, and
    #     `checkpoint_save_system` does `torch.save(self.__dict__)` at every validation, so the
    #     class had to serialise itself back into a placeholder. A declared attribute needs none
    #     of that.
    #   - the try/finally restore, and the silent failure when a checkpoint reload replaced
    #     `__dict__` and dropped the patched method without anyone noticing.
    """

    N_VAL_WINDOWS = 256      # a tight RMS estimate; all windows costs ~7x for no more precision

    def __init__(self, fit_sys, nf, cfg, val_sd, do_print=True):
        self.fit_sys = fit_sys
        self.nf = nf
        self.do_print = do_print
        fit_sys.Loss_train_nf = []
        fit_sys.Loss_val_nf = []

        # Train side: statistics accumulate inside loss() over every training window, so there is
        # nothing to build here and no extra rollout. Attaching the object IS the switch.
        fit_sys.loss_stats = WindowErrorStats(fit_sys.norm.ystd, dtype=cfg.dtype_pt)

        # Val side: ONE fixed batch, drawn once. Fixed and not resampled per epoch, because a
        # changing window set puts selection noise on top of the learning signal and the two
        # cannot be separated afterwards.
        self._val_stats_obj = WindowErrorStats(fit_sys.norm.ystd, dtype=cfg.dtype_pt)
        self._val_batch, self._val_kwargs = None, {}
        try:
            d = fit_sys.make_training_data(fit_sys.norm.transform(val_sd), nf=nf, stride=nf)
            n = min(self.N_VAL_WINDOWS, len(d[0]))
            ix = np.random.default_rng(cfg.seed).choice(len(d[0]), size=n, replace=False)
            cols = [torch.as_tensor(np.asarray(a)[ix], dtype=cfg.dtype_pt) for a in d]
            self._val_batch = tuple(cols[:4])
            # Arrays a simulator asked for travel BY NAME, the same convention fit() uses.
            names = getattr(fit_sys.simulator, 'extra_array_names', ())
            self._val_kwargs = dict(zip(names, cols[4:]))
        except Exception as e:
            print(f'    [nf val  ] window batch unavailable (non-fatal): {e}')
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

    def _val_stats(self):
        """The fixed val window batch, through the SAME rollout the loss uses.

        `fit_sys.simulate` and not `closed_loop_free_run_rms`: the mode then follows the attached
        simulator exactly as the loss does, so a closed-loop run gives closed-loop numbers and an
        open-loop run open-loop ones, with no branch here and no second rollout to drift.
        """
        if self._val_batch is None:
            return None
        uh, yh, uf, yf = self._val_batch
        try:
            with torch.no_grad():
                x = self.fit_sys.encoder(uh, yh)
                y_pred, _ = self.fit_sys.simulate(x, uf, yf, **self._val_kwargs)
            self._val_stats_obj.reset()
            self._val_stats_obj.update(y_pred, yf)
            return self._val_stats_obj.summary()
        except Exception as e:
            print(f'    [nf val  ] failed (non-fatal): {e}')
            return None

    @staticmethod
    def _fmt(tag, s):
        return ('    [nf %-5s] rms %.3e  chan %s  grow %s  bias %s  (%d win)'
                % (tag, s['rms'],
                   '/'.join('%.1e' % v for v in s['chan']),
                   ('%.2fx' % s['grow']) if np.isfinite(s['grow']) else 'n/a',
                   '/'.join('%+.1e' % v for v in s['bias']),
                   s['n_win']))

    def _joint_probe(self):
        """One line: recovery + negation meters. Cost: scalars + one batched ANN
        forward over the fixed penalty points (measured 0.03 s, plan Step 3).
        combo part requires a parameterized block (joint runs); orth part only
        an attached penalty, i.e. cfg.orth, independently of joint estimation)."""
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

    def __call__(self, fit_sys, val_sys_data, value):
        """validation_probes entry: side effects only, `value` is read-only here."""
        # Train: read and RESET, so each line covers the windows trained since the last
        # validation rather than the whole run.
        tr_s = fit_sys.loss_stats.summary() if fit_sys.loss_stats is not None else None
        if fit_sys.loss_stats is not None:
            fit_sys.loss_stats.reset()
        va_s = self._val_stats()
        self.fit_sys.Loss_train_nf.append(tr_s['rms'] if tr_s else float('nan'))
        self.fit_sys.Loss_val_nf.append(va_s['rms'] if va_s else float('nan'))
        if self.do_print:
            if tr_s:
                print(self._fmt('train', tr_s))
            if va_s:
                print(self._fmt('val', va_s))
            # The comparison, stated rather than left as arithmetic. `value` is the 12 s free-run
            # sim-RMS that selects the checkpoint, so both ratios are against the same scalar and
            # each is labelled with what it confounds.
            if np.isfinite(value):
                parts = []
                if va_s and va_s['rms'] > 0:
                    parts.append('%.2fx val_nf (horizon)' % (value / va_s['rms']))
                if tr_s and tr_s['rms'] > 0:
                    parts.append('%.2fx train_nf (horizon+gen)' % (value / tr_s['rms']))
                if parts:
                    print('    [nf gap  ] val_sim %.3e = %s' % (value, ' = '.join(parts)))
        if (self._pblock is not None
                or getattr(self.fit_sys, 'orth_penalty', None) is not None):
            try:
                self._joint_probe()
            except Exception as e:
                print(f'    [joint-probe] failed (non-fatal): {e}')


def _install_nf_val_probe(fit_sys, hp, cfg, val_sd):
    """Append an `_NfProbe` to `fit_sys.validation_probes`; return the previous tuple to restore."""
    prev = fit_sys.validation_probes
    probe = _NfProbe(fit_sys, hp['nf'], cfg, val_sd, do_print=cfg.nf_probe_print)
    fit_sys.validation_probes = tuple(prev) + (probe,)
    if cfg.nf_probe_print:
        mode = 'closed loop' if getattr(fit_sys, 'simulator', None) is not None else 'open loop'
        print(f"\n[nf] rms = RMS over nf={hp['nf']} ({cfg.nf_seconds:.3f} s) windows, "
              f"{mode}, metres")
        print('     chan = per-channel X/Theta/Y | grow = RMS(last step)/RMS(first) | '
              'bias = mean residual')
        print(f'     train = all windows since the last validation, running mean | '
              f'val = {probe.N_VAL_WINDOWS} fixed windows, snapshot')
        print('     metre weighting is NOT the objective\'s, which weights channels by 1/ystd^2;')
        print('     numbers are not comparable ACROSS loop modes, the [nf gap] ratios are')
    return prev


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
    _prev_probes = _install_nf_val_probe(fit_sys, hp, cfg, data.val_ckpt_data)
    try:
        bestfit = train_model(fit_sys, hp, cfg, data,
                              epochs=epochs_remaining,
                              nf=hp['nf'],
                              validation_measure='sim-RMS')
    finally:
        fit_sys.validation_probes = _prev_probes   # restore always
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
