"""Training: Adam with the PS2 opening, selection of the best validated model, the L-BFGS polish, and the checkpoint
files that let a run resume."""
__project_origin__ = "added"

import os
import json
import time

import numpy as np
import torch

from .config import RunConfig
from .model import on_device
from model_augmentation.fit_systems.closed_loop import SCORE_NAME
from model_augmentation.fit_systems.lbfgs_polish import PolishSpec, lbfgs_polish
from model_augmentation.fit_systems.ps2_opening import PS2Opening, PS2Spec


def _json_safe(obj):
    """Convert NumPy integer and floating values recursively to native values for JSON metadata."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def save_checkpoint_pair(fit_sys, base_path, include_optimizer=True, **meta):
    """Save the weights (and Adam state) to <base>.pt and the meta values to <base>.npz.

    Args:
        fit_sys: the augmented model.
        base_path: file path without extension.
        include_optimizer: also save Adam's moment estimates, needed to continue Adam exactly.
        **meta: values for <base>.npz (done_epochs, bestfit, ...).
    """
    # encoder psi and interconnect hfn together form the whole trainable model
    payload = {
        'hfn':     fit_sys.hfn.state_dict(),
        'encoder': fit_sys.encoder.state_dict(),
    }
    if include_optimizer:
        payload['optimizer'] = fit_sys.optimizer.state_dict()
    torch.save(payload, base_path + '.pt')
    np.savez(base_path + '.npz', **{k: np.array(v) for k, v in meta.items()})


def resolve_checkpoint(base_path):
    """Read a checkpoint in either format.

    Args:
        base_path: '<base>' of a pair (<base>.pt, <base>.npz) written by save_checkpoint_pair, or the path of a
            deepSI dump (.pth) written during Adam.
    Returns:
        (hfn weights, encoder weights, Adam state or None, meta dict, format 'pair' or 'whole').
    """
    # 1. Pair format: weights in <base>.pt, meta in <base>.npz.
    pair_pt, pair_npz = base_path + '.pt', base_path + '.npz'
    if os.path.exists(pair_pt) and os.path.exists(pair_npz):
        ckpt = torch.load(pair_pt, map_location='cpu')
        # file closed at once, since a rerun moves it after loading
        with np.load(pair_npz, allow_pickle=False) as z:
            meta = {k: z[k] for k in z.files}
        return ckpt['hfn'], ckpt['encoder'], ckpt.get('optimizer'), meta, 'pair'

    # 2. Otherwise a deepSI dump: the whole model object, with the meta read from its attributes.
    if not (base_path.endswith('.pth') and os.path.exists(base_path)):
        raise FileNotFoundError(
            'no checkpoint at %r: neither the pair %r, %r nor a deepSI dump %r exists; check the run folder '
            'and the --from choice' % (base_path, pair_pt, pair_npz, base_path))
    d = torch.load(base_path, map_location='cpu', weights_only=False)
    # done_epochs: epochs done over all continuations; bestfit: best validation score so far;
    # ps2_opening: (phase 'S1', 'S2' or 'S3' = ended, updates) of the PS2 opening when the dump was written
    meta = {
        'done_epochs': np.array(round(float(d.get('epochs_before', 0)) + float(d.get('epoch_counter', 0)))),
        'bestfit': np.array(float(d.get('bestfit', float('inf')))),
        'ps2_opening': d.get('ps2_opening'),
    }
    optimizer = d.get('optimizer')
    optimizer_state = None if optimizer is None else optimizer.state_dict()
    return d['hfn'].state_dict(), d['encoder'].state_dict(), optimizer_state, meta, 'whole'


def load_optimizer_moments(optimizer, saved):
    """Restore the saved Adam moment estimates into optimizer, keeping this run's lr, eps and betas."""
    live = optimizer.state_dict()
    if len(live['param_groups']) != len(saved['param_groups']):
        raise RuntimeError(
            'optimizer parameter groups do not match (this run %d, the checkpoint %d); resume with the '
            'configuration that wrote the checkpoint'
            % (len(live['param_groups']), len(saved['param_groups'])))
    # saved moments per parameter; per group only the parameter ids are taken from the save, the settings from this run
    merged = {**saved, 'param_groups': [
        {**live_g, 'params': saved_g['params']}
        for live_g, saved_g in zip(live['param_groups'], saved['param_groups'])
    ]}
    optimizer.load_state_dict(merged)


def _assert_same_architecture(target_sd, ckpt_sd, what):
    """Refuse a checkpoint that does not fit this model, with a message naming the mismatch.

    Args:
        target_sd: state dict of this run's module.
        ckpt_sd: state dict from the checkpoint.
        what: module name for the message ('hfn' or 'encoder').
    """
    # 1. Same parameter coordinates: coordinate_version names the map beta -> theta_base, so another version means
    #    the same numbers stand for other physical parameters.
    target_coord = {int(v) for k, v in target_sd.items() if k.endswith('coordinate_version')}
    ckpt_coord = {int(v) for k, v in ckpt_sd.items() if k.endswith('coordinate_version')}
    if target_coord != ckpt_coord:
        raise RuntimeError(
            'checkpoint coordinate version does not match this config for %s (config %s, checkpoint %s): its '
            'free_params mean other parameters, so start a new run instead of resuming it'
            % (what, sorted(target_coord), sorted(ckpt_coord)))

    # 2. Same keys and shapes, so n_a and the network size match.
    missing = sorted(set(target_sd) - set(ckpt_sd))
    extra = sorted(set(ckpt_sd) - set(target_sd))
    shape = [(k, tuple(target_sd[k].shape), tuple(ckpt_sd[k].shape))
             for k in sorted(set(target_sd) & set(ckpt_sd))
             if tuple(target_sd[k].shape) != tuple(ckpt_sd[k].shape)]
    if missing or extra or shape:
        raise RuntimeError(
            'checkpoint does not match this config for %s: %d key(s) missing %s, %d extra %s, %d shape '
            'mismatch(es) (key, config, checkpoint) %s, first 3 of each; resume with the n_a, network size and '
            'parameter coordinates of the run that wrote it'
            % (what, len(missing), missing[:3], len(extra), extra[:3], len(shape), shape[:3]))


def _assert_same_dtype(fit_sys, hfn_sd):
    """Refuse a checkpoint whose floating dtype differs from the model's."""
    want = next(fit_sys.hfn.parameters()).dtype
    got = next((v.dtype for v in hfn_sd.values()
                if torch.is_tensor(v) and v.is_floating_point()), None)
    if got is not None and got != want:
        raise RuntimeError(
            'checkpoint dtype %s does not match this run\'s %s; use a float64 checkpoint' % (got, want))


def load_checkpoint(fit_sys, base_path, start_phase='adam'):
    """Restore a checkpoint into fit_sys.

    Args:
        fit_sys: the augmented model, built with the run's config.
        base_path: checkpoint path, as for resolve_checkpoint.
        start_phase: 'adam' also restores the Adam moments; 'lbfgs' (polish only) restores the weights only.
    Returns:
        meta: done_epochs, bestfit and, by format, ps2_opening (deepSI dump) or lbfgs_outcome (final pair).
    """
    # 1. Read; refuse a dump written inside an unfinished PS2 opening, whose own state a dump does not hold.
    hfn_sd, enc_sd, optim_sd, meta, fmt = resolve_checkpoint(base_path)
    opening = meta.get('ps2_opening')
    if start_phase == 'adam' and opening is not None and opening[0] != 'S3' and opening[1] > 0:
        raise RuntimeError(
            '%s was written %d updates into an unfinished PS2 opening (phase %s); continue from a dump '
            'written before the opening started or after it ended' % (base_path, opening[1], opening[0]))
    # 2. Check that the checkpoint fits this model, then load the weights.
    _assert_same_dtype(fit_sys, hfn_sd)
    _assert_same_architecture(fit_sys.hfn.state_dict(), hfn_sd, 'hfn')
    _assert_same_architecture(fit_sys.encoder.state_dict(), enc_sd, 'encoder')
    fit_sys.hfn.load_state_dict(hfn_sd)
    fit_sys.encoder.load_state_dict(enc_sd)
    # 3. On an Adam start, the saved moments if the checkpoint has them; else Adam restarts its moments.
    if optim_sd is not None and start_phase == 'adam':
        load_optimizer_moments(fit_sys.optimizer, optim_sd)
        print("  Adam moment estimates restored; lr/eps/betas come from this run's config.")
    print(f'  Checkpoint format: {fmt}')
    if start_phase == 'adam' and optim_sd is None:
        print('  This checkpoint has no Adam state'
              + (' (saved without it: polished weights or a polish-only job)' if fmt == 'pair'
                 else ' (the dump holds no optimizer)')
              + '; Adam starts from fresh moment estimates at these weights, which suits polished weights'
                ' but does not continue the earlier Adam trajectory. To continue it, resume'
                ' gantry_ckpt_<run_id>_pre_lbfgs, which keeps the matching state.')
    return meta


class ValidationHistory:
    """Every validation (RMS per record and axis) and every selection step, saved to val_hist.npz at each entry; kept
    outside the model, since fit reloads the model from its best checkpoint."""

    def __init__(self, fit_sys, hist_path):
        """An empty history for fit_sys, saved to hist_path (val_hist.npz)."""
        self.fit_sys = fit_sys
        self.hist_path = hist_path
        self.val_hist = []

    def record(self, event, fit_sys, score, row=None, **extra):
        """Append one row and save the history.

        Args:
            event: 'adam' (validation during Adam), 'rebuilt' (rescore after the OBC basis rebuild), 'polish_candidate'
                (validation of the polished weights), 'selected' (the chosen Adam row) or 'final' (the kept model).
            fit_sys: the model just validated.
            score: its sim-NRMSe.
            row: an existing row to copy ('selected', 'final'); None builds one from the last validation.
            **extra: more fields, e.g. accepted (1 or 0) for a polish candidate.
        """
        if row is None:
            sim = fit_sys.simulator
            # per_channel: (n_val, 3) RMS of y_hat - y per validation record and axis [m]; update and epoch: where
            # Adam was at this validation
            row = dict(per_channel=np.array(sim.last_per_channel, dtype=float), score=float(score),
                       update=int(getattr(fit_sys, 'batch_counter', -1)),
                       epoch=float(getattr(fit_sys, 'epoch_counter', float('nan'))))
        self.val_hist.append(dict(row, event=event, **extra))
        np.savez(self.hist_path, **self.hist_arrays())

    def hist_arrays(self):
        """The history as arrays, one entry per row, for val_hist.npz."""
        h = self.val_hist
        # val_hist_accepted: 1 or 0 for a polish candidate, NaN for other rows; val_sigma_e: (n_val, 3) sigma_e per
        # validation record and axis [m], so val_hist_rms / val_sigma_e gives the NRMS_e behind the score
        return dict(
            val_hist_event=np.array([r['event'] for r in h]),
            val_hist_update=np.array([r['update'] for r in h]),
            val_hist_epoch=np.array([r['epoch'] for r in h]),
            val_hist_score=np.array([r['score'] for r in h]),
            val_hist_rms=np.stack([r['per_channel'] for r in h]),
            val_hist_accepted=np.array([r.get('accepted', np.nan) for r in h], dtype=float),
            val_sigma_e=np.asarray(self.fit_sys.simulator.val_sigma_e))

    def __call__(self, fit_sys, val_sys_data, value):
        """Called by fit after every validation during Adam: record it as an 'adam' row with score value."""
        self.record('adam', fit_sys, value)


def run_lbfgs_polish(fit_sys, cfg: RunConfig, data, bestfit, history, adam_ran=True):
    """The L-BFGS polish of the selected model, kept only if the validation score drops.

    Args:
        fit_sys, cfg, data: model, settings and records.
        bestfit: score of the Adam-selected model, the score to beat (inf if unknown).
        history: the ValidationHistory that receives the 'rebuilt' and 'polish_candidate' rows.
        adam_ran: False when this job ran no Adam update (polish only, or Adam budget already done).
    Returns:
        the polish outcome dict of lbfgs_polish.
    """
    print('\n[lbfgs] polish phase: Adam is done, L-BFGS from here')
    # 1. Polish settings from the config.
    spec = PolishSpec(
        n_windows=cfg.lbfgs_windows, max_iter=cfg.lbfgs_max_iter,
        inner_iter=cfg.lbfgs_inner_iter, history_size=cfg.lbfgs_history,
        tol_grad=cfg.lbfgs_tol_grad, tol_change=cfg.lbfgs_tol_change,
        seed=cfg.seed)
    with on_device(fit_sys, cfg):
        if cfg.device == 'cuda':
            torch.cuda.empty_cache()
        # 2. Score to beat: bestfit; None lets the polish measure it itself.
        val_before = (bestfit if np.isfinite(bestfit) else None)
        if not adam_ran and not cfg.obc:
            # without Adam in this job, one eager (uncompiled) validation initialises the model before the
            # compiled polish
            fit_sys.cal_validation_error(data.val_ckpt_data)
        # on the OBC arm, fit rebuilt the basis at the selected beta, which changes the score a little: rescore
        if cfg.obc:
            val_rebuilt = float(fit_sys.cal_validation_error(data.val_ckpt_data))
            print('[lbfgs] rebuilt-basis validation %s %.6e (selected checkpoint: %.6e)'
                  % (SCORE_NAME, val_rebuilt, bestfit))
            history.record('rebuilt', fit_sys, val_rebuilt)
            if val_before is None:
                val_before = val_rebuilt
        # 3. Polish on the training windows; rolled back inside unless the validation score drops.
        out = lbfgs_polish(
            fit_sys, data.train_data,
            loss_kwargs={'nf': cfg.nf, 'stride': cfg.stride}, spec=spec,
            val_sys_data=data.val_ckpt_data,
            val_before=val_before,
            validation_measure=SCORE_NAME)
        # 4. Record the polished weights' validation, also when they were rolled back.
        if out.get('ran') and np.isfinite(out.get('val_after', np.nan)):
            history.record('polish_candidate', fit_sys, out['val_after'],
                           accepted=float(bool(out.get('accepted'))))
        return out


def _adam_phase(fit_sys, cfg: RunConfig, data, checkpoint_dir, meta, done_epochs, epochs_remaining, polish_only):
    """Adam phase of one job.

    Args:
        fit_sys, cfg, data: model, settings and records.
        checkpoint_dir: run folder, receives val_hist.npz.
        meta: meta of the resumed checkpoint, None for a fresh run.
        done_epochs: Adam epochs already done before this job.
        epochs_remaining: Adam epochs this job runs.
        polish_only: True skips Adam (start_phase 'lbfgs').
    Returns:
        (PS2 opening or None, ValidationHistory, seconds spent).
    """
    # 1. Selection restarts in this job; the initial validation in fit keeps the resumed weights a candidate.
    fit_sys.bestfit = float('inf')
    t0 = time.time()
    opening = None
    # 2. PS2 opening: on a fresh run, or on a resume from a dump written before the opening ended.
    ps2_state = None if meta is None else meta.get('ps2_opening')
    if cfg.ps2 and not polish_only and (
            done_epochs == 0 if ps2_state is None else ps2_state[0] != 'S3'):
        # it runs after every Adam update: S1 steps, then the S2 fit, then phase S3 (ended)
        spec = PS2Spec(**cfg.ps2_opening, seed=cfg.seed)
        opening = PS2Opening(fit_sys, data, n_xb=cfg.nx_phys, n_a=cfg.n_a,
                             width=cfg.n_nodes_per_layer, spec=spec)
        fit_sys.training_phase = opening
    # 3. Validation history: called by fit after every validation, removed again afterwards.
    prev = fit_sys.validation_probes
    history = ValidationHistory(fit_sys, os.path.join(checkpoint_dir, 'val_hist.npz'))
    fit_sys.validation_probes = tuple(prev) + (history,)
    # 4. Adam on the remaining budget; fit ends with the best validated weights loaded.
    try:
        if polish_only:
            print('  start_phase=lbfgs: skipping Adam, polishing the resumed weights directly')
        elif epochs_remaining <= 0:
            print('  Adam budget done (%d of %d epochs): skipping Adam' % (done_epochs, cfg.epochs))
        else:
            with on_device(fit_sys, cfg):
                fit_sys.fit(
                    train_sys_data=data.train_data, val_sys_data=data.val_ckpt_data,
                    batch_size=cfg.batch_size, epochs=epochs_remaining,
                    n_its=cfg.n_its,
                    its_per_val=(cfg.n_its if cfg.n_its else cfg.its_per_val),
                    loss_kwargs={'nf': cfg.nf, 'stride': cfg.stride},
                    cuda=(cfg.device == 'cuda'),
                    validation_measure=SCORE_NAME,
                )
    finally:
        fit_sys.validation_probes = prev
        fit_sys.training_phase = None
    if opening is not None:
        print('[ps2] summary: %s' % opening.summary_json(), flush=True)
    return opening, history, time.time() - t0


def train(fit_sys, cfg: RunConfig, data, checkpoint_dir, run_id, resume=None):
    """Run the remaining Adam budget, then polish and save the selected model.

    Args:
        fit_sys, cfg, data: model, settings and records; cfg.epochs is the total Adam budget over all
            continuations, so a resume runs only what its checkpoint has not done.
        checkpoint_dir: the run folder.
        run_id: run id in the checkpoint names.
        resume: (archived path, meta) of the checkpoint loaded into fit_sys; None for a fresh run.
    Returns:
        bestfit: sim-NRMSe (mean validation NRMS_e over records and axes) of the final model: the accepted polish,
        else the Adam-selected model.
    """
    # 1. Resume state: epochs done and the score to beat, from the checkpoint meta.
    done_epochs = 0
    resumed_at_epoch = -1
    polish_only = (cfg.start_phase == 'lbfgs')

    fit_sys.unique_code = str(run_id)
    bestfit = float('inf')

    resume_ckpt, meta = (None, None) if resume is None else resume
    if meta is not None:
        done_epochs = int(meta['done_epochs'])
        resumed_at_epoch = done_epochs
        bestfit     = float(meta['bestfit'])
        print(f'Resumed from {resume_ckpt}  ({done_epochs} epochs done)')
        # deepSI dumps of this job count epochs on from here
        fit_sys.epochs_before = done_epochs

    epochs_remaining = cfg.epochs - done_epochs
    print(f'\nTraining: nf={cfg.nf}  epochs={cfg.epochs}  val={SCORE_NAME}  n_a={cfg.n_a}')
    if done_epochs > 0:
        print(f'  Resuming: {done_epochs} done, {epochs_remaining} remaining')

    # 2. Adam with the PS2 opening; fit leaves the best validated weights loaded.
    adam_ran = not polish_only and epochs_remaining > 0
    opening, history, elapsed = _adam_phase(fit_sys, cfg, data, checkpoint_dir, meta, done_epochs,
                                            epochs_remaining, polish_only)
    if adam_ran:
        bestfit = fit_sys.bestfit
    # the pairs record the whole budget as done, also after an n_its cap or an early stop
    if not polish_only:
        done_epochs = cfg.epochs

    # 3. Pre-polish checkpoint: the selected weights with their Adam moments, so Adam can be continued from here.
    pre_base = os.path.join(checkpoint_dir, f'gantry_ckpt_{run_id}_pre_lbfgs')
    save_checkpoint_pair(fit_sys, pre_base, done_epochs=done_epochs, bestfit=bestfit)
    print(f'  Pre-polish checkpoint: {pre_base}.pt')
    # mark the selected Adam validation in the history; on equal scores the latest, as fit keeps the latest best
    adam_rows = [r for r in history.val_hist if r['event'] == 'adam']
    selected = min(reversed(adam_rows), key=lambda r: r['score']) if adam_rows else None
    if selected is not None:
        history.record('selected', fit_sys, selected['score'], row=selected)

    # 4. Polish, then the final model: the polished weights if accepted, else the (rescored) Adam selection.
    lbfgs_outcome = run_lbfgs_polish(fit_sys, cfg, data, bestfit, history, adam_ran=adam_ran)
    last_by_event = {r['event']: r for r in history.val_hist}
    final = (last_by_event.get('polish_candidate') if lbfgs_outcome.get('accepted')
             else last_by_event.get('rebuilt') or selected)
    if final is not None:
        history.record('final', fit_sys, final['score'], row=final)
        bestfit = final['score']
    fit_sys.bestfit = bestfit

    # 5. Final checkpoint, without Adam state after an accepted polish or in a polish-only job, since the saved
    #    moments would not belong to these weights.
    polish_accepted = bool(lbfgs_outcome.get('accepted'))
    drop_optimizer_state = polish_accepted or polish_only
    ckpt_base = os.path.join(checkpoint_dir, f'gantry_ckpt_{run_id}')
    save_checkpoint_pair(
        fit_sys, ckpt_base, include_optimizer=not drop_optimizer_state,
        done_epochs=done_epochs, bestfit=bestfit, lbfgs_outcome=json.dumps(_json_safe(lbfgs_outcome)),
        # resumed_at_epoch: -1 for a fresh run; ps2_summary: the opening's record, for reading only
        resumed_at_epoch=resumed_at_epoch, ps2_summary='' if opening is None else opening.summary_json())
    drop_reason = ('weights were polished' if polish_accepted else
                   'no Adam phase ran in this job' if polish_only else '')
    print(f'  Checkpoint weights: {ckpt_base}.pt'
          + (f'  (Adam state omitted: {drop_reason})' if drop_optimizer_state else ''))
    print(f'  Checkpoint meta: {ckpt_base}.npz')

    # 6. Training loss to loss.npz: fit's mean loss per validation interval, with its update, epoch and time
    #    [s since the Adam start]; empty when this job ran no Adam.
    fit_history = getattr(fit_sys, 'fit_history', {}) if adam_ran else {}
    np.savez(os.path.join(checkpoint_dir, 'loss.npz'),
             update=np.asarray(fit_history.get('batch_id', []), dtype=np.int64),
             epoch=np.asarray(fit_history.get('epoch_id', []), dtype=float),
             time=np.asarray(fit_history.get('time', []), dtype=float),
             train_loss=np.asarray(fit_history.get('Loss_train', []), dtype=float))

    fit_sys.eval()
    print(f'Training done | {done_epochs} ep | {elapsed:.0f}s | bestfit={bestfit:.5f}')
    return bestfit
