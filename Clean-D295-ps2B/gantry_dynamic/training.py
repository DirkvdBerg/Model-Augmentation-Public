"""Training of one run: Adam with the PS opening, the choice of the weights with the lowest validation score, the
L-BFGS polish, and the checkpoints from which a run can be continued.

A job is one launch of gantry_interconnect_dynamic.py. A run folder is trained in one job, or in more when it is
resumed (e.g. after the server's time limit) or its polish is rerun. Two kinds of checkpoint: deepSI saves the whole
model at every validation during Adam (_best.pth, _last.pth), from which a resume starts; this module writes the
pairs below. The Adam moments are Adam's running averages of the gradients; with them, Adam continues exactly where
it stopped. The validation score sim-NRMSe is the free-run error on the validation records relative to their servo
error sigma_e, averaged over records and axes.

Files written to the run folder:
  gantry_ckpt_<run_id>_pre_lbfgs   (.pt, .npz) the weights Adam selected
  gantry_ckpt_<run_id>             (.pt, .npz) the final weights: the polished ones if kept, else Adam's
  val_hist.npz                     every validation score, and which one was selected and kept
  loss.npz                         the mean training loss between validations
Each .pt holds the weights, each .npz the run's progress: epochs done (done_epochs) and the lowest validation
score (bestfit).
"""
__project_origin__ = "added"

import os
import json
import time

import deepSI
import numpy as np
import torch

from .config import RunConfig
from .model import on_device
from model_augmentation.fit_systems.closed_loop import SCORE_NAME
from model_augmentation.fit_systems.lbfgs_polish import PolishSpec, lbfgs_polish
from model_augmentation.fit_systems.ps2_opening import PS2Opening, PS2Spec


def _json_safe(obj):
    """The polish result obj, with each NumPy number turned into an int or float so json.dumps can write it into the
    final checkpoint."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def save_checkpoint_pair(fit_sys, base_path, save_adam_moments=True, **meta):
    """Save the weights, and the Adam moments if save_adam_moments, to base_path.pt; the run's progress to
    base_path.npz.

    Args:
        fit_sys: the augmented model.
        base_path: file path without extension.
        save_adam_moments: also save the Adam moments, so Adam can continue where it stopped.
        **meta: values for base_path.npz: done_epochs (epochs done in all jobs) and bestfit (lowest validation
            score); the final pair also lbfgs_outcome (the polish result), resumed_at_epoch (epochs done when
            this job started, -1 for a fresh run) and ps2_summary (the PS opening's record).
    """
    # hfn (the model's one-step map) and the encoder psi hold all trained weights
    payload = {
        'hfn':     fit_sys.hfn.state_dict(),
        'encoder': fit_sys.encoder.state_dict(),
    }
    if save_adam_moments:
        payload['optimizer'] = fit_sys.optimizer.state_dict()
    torch.save(payload, base_path + '.pt')
    np.savez(base_path + '.npz', **{k: np.array(v) for k, v in meta.items()})


def resolve_checkpoint(base_path):
    """Weights, Adam moments and progress of either checkpoint: a pair from save_checkpoint_pair, or a deepSI
    checkpoint saved at a validation during Adam.

    Args:
        base_path: '<base>' of a pair (<base>.pt, <base>.npz) written by save_checkpoint_pair, or the path of a
            deepSI checkpoint (.pth) written during Adam.
    Returns:
        (hfn weights, encoder weights, Adam moments or None, progress).
    """
    # 1. A pair: weights in base_path.pt, done_epochs, bestfit and the other values in base_path.npz.
    pair_pt, pair_npz = base_path + '.pt', base_path + '.npz'
    if os.path.exists(pair_pt) and os.path.exists(pair_npz):
        ckpt = torch.load(pair_pt, map_location='cpu')
        # file closed at once, since a rerun moves it after loading
        with np.load(pair_npz, allow_pickle=False) as z:
            meta = {k: z[k] for k in z.files}
        return ckpt['hfn'], ckpt['encoder'], ckpt.get('optimizer'), meta

    # 2. Otherwise a deepSI checkpoint: the whole saved model, which also holds the progress.
    if not (base_path.endswith('.pth') and os.path.exists(base_path)):
        raise FileNotFoundError(
            'no checkpoint at %r: neither the pair %r, %r nor a deepSI checkpoint %r exists; check the run folder '
            'and the --from choice' % (base_path, pair_pt, pair_npz, base_path))
    d = torch.load(base_path, map_location='cpu', weights_only=False)
    # done_epochs: Adam epochs of the run up to this checkpoint, rounded, since deepSI saves at validations,
    # which can fall inside an epoch
    # ps2_opening: (phase, Adam updates in the PS opening so far), phase 'S1' while the opening runs; missing in a
    # run without the opening
    meta = {
        'done_epochs': np.array(round(float(d['epochs_before']) + float(d['epoch_counter']))),
        'bestfit': np.array(float(d['bestfit'])),
        'ps2_opening': d.get('ps2_opening'),
    }
    return d['hfn'].state_dict(), d['encoder'].state_dict(), d['optimizer'].state_dict(), meta


def load_checkpoint(fit_sys, base_path, polish_only=False):
    """Load a checkpoint into fit_sys: the weights and, unless polish_only, the Adam moments; returns its progress.

    Args:
        fit_sys: the augmented model, built with the run's config.
        base_path: checkpoint path, as for resolve_checkpoint.
        polish_only: True loads the weights only, for a polish rerun or an evaluation; False also loads the Adam
            moments, so a resume continues Adam where it stopped.
    Returns:
        progress: done_epochs, bestfit, and ps2_opening (deepSI checkpoint) or lbfgs_outcome (final pair).
    """
    # 1. Refuse a deepSI checkpoint saved during the PS opening: the opening's own network and Adam are not saved,
    #    so the opening cannot continue.
    hfn_sd, enc_sd, optim_sd, meta = resolve_checkpoint(base_path)
    opening = meta.get('ps2_opening')
    if not polish_only and opening is not None and opening[0] != 'S3' and opening[1] > 0:
        raise RuntimeError(
            '%s was written %d updates into an unfinished PS opening (phase %s); continue from a checkpoint '
            'written before the opening started or after it ended' % (base_path, opening[1], opening[0]))
    # 2. Load the weights; a rerun reads its settings from the run folder, so the checkpoint fits this model.
    fit_sys.hfn.load_state_dict(hfn_sd)
    fit_sys.encoder.load_state_dict(enc_sd)
    # 3. Before more Adam: the saved Adam moments; a checkpoint without them starts Adam from zero moments.
    if not polish_only:
        if optim_sd is not None:
            fit_sys.optimizer.load_state_dict(optim_sd)
        else:
            print('  no Adam moments in this checkpoint: Adam starts from zero moments')
    return meta


class ValidationHistory:
    """One row per validation of this job (score, RMS error per validation record and axis), plus a row for the
    selected and one for the final model; written to val_hist.npz after every row. Not stored on the model, since
    fit reloads the model from its best checkpoint."""

    def __init__(self, fit_sys, hist_path):
        """An empty history; each row takes its RMS errors from fit_sys's last validation and is written to hist_path
        (val_hist.npz)."""
        self.fit_sys = fit_sys
        self.hist_path = hist_path
        self.val_hist = []

    def record(self, event, score, row=None, **extra):
        """Add one row and save val_hist.npz.

        Args:
            event: what the row is:
                'adam'              a validation during Adam
                'rebuilt'           the selected weights validated again after the OBC basis J is rebuilt (OBC arm)
                'polish_candidate'  the polished weights
                'selected'          the Adam validation that was chosen
                'final'             the model that is kept
            score: sim-NRMSe of the validation; unused when row is given.
            row: for 'selected' and 'final', the earlier row to copy; None builds the row from the last validation.
            **extra: further fields; only accepted (1 or 0) for a polish candidate.
        """
        fit_sys = self.fit_sys
        if row is None:
            sim = fit_sys.simulator
            # rms: for each validation record and axis, the RMS of y_hat - y over the record's samples [m]
            # update, epoch: Adam updates and epochs done at this validation; -1 and NaN in a job without Adam
            row = dict(rms=np.array(sim.last_per_channel, dtype=float), score=float(score),
                       update=int(getattr(fit_sys, 'batch_counter', -1)),
                       epoch=float(getattr(fit_sys, 'epoch_counter', float('nan'))))
        self.val_hist.append(dict(row, event=event, **extra))
        np.savez(self.hist_path, **self.hist_arrays())

    def hist_arrays(self):
        """The rows as one array per field, plus val_sigma_e, for val_hist.npz."""
        # accepted: 1 or 0 for a polish candidate, NaN for the other rows
        arrays = {'val_hist_' + name: np.array([row.get(name, np.nan) for row in self.val_hist])
                  for name in ('event', 'update', 'epoch', 'score', 'rms', 'accepted')}
        # val_sigma_e: the standard deviation of the servo error y - r per validation record and axis [m];
        # val_hist_rms / val_sigma_e is the NRMS per record and axis, whose mean is the score
        arrays['val_sigma_e'] = np.asarray(self.fit_sys.simulator.val_sigma_e)
        return arrays

    def __call__(self, _model, _validation_records, score):
        """Called by fit after every validation during Adam; records the score as an 'adam' row."""
        self.record('adam', score)


def run_lbfgs_polish(fit_sys, cfg: RunConfig, data, bestfit, history):
    """L-BFGS polish of the selected weights, kept only if they lower the validation score.

    Args:
        fit_sys, cfg, data: model, settings and records.
        bestfit: validation score of the selected weights, which the polish must beat; inf if Adam never validated.
        history: this job's validation history; this function adds the 'rebuilt' and 'polish_candidate' rows.
    Returns:
        the polish result of lbfgs_polish (ran, accepted, scores before and after, ...).
    """
    print('\n[lbfgs] polish phase: Adam is done, L-BFGS from here')
    # 1. Polish settings from the config.
    spec = PolishSpec(n_windows=cfg.lbfgs_windows, max_iter=cfg.lbfgs_max_iter, seed=cfg.seed)
    # deepSI takes all records of a split as one System_data_list
    train_data, val_data = deepSI.System_data_list(data.train_list), deepSI.System_data_list(data.val_list)
    with on_device(fit_sys, cfg):
        # release the GPU memory Adam no longer uses, before the polish allocates its windows
        if cfg.device == 'cuda':
            torch.cuda.empty_cache()
        # 2. Score to beat: bestfit; with None, lbfgs_polish measures it first.
        val_before = (bestfit if np.isfinite(bestfit) else None)
        # OBC arm: during Adam, J is built at the beta of each epoch's start; at the end of Adam it is rebuilt at the
        # selected beta and theta_aux recomputed, so the correction matches the final parameters. The score then
        # differs slightly from the selected validation: validate again and record it
        if cfg.obc:
            val_rebuilt = float(fit_sys.cal_validation_error(val_data))
            print('[lbfgs] rebuilt-J validation %s %.6e (selected checkpoint: %.6e)'
                  % (SCORE_NAME, val_rebuilt, bestfit))
            history.record('rebuilt', val_rebuilt)
            if val_before is None:
                val_before = val_rebuilt
        # 3. Polish on the training windows; lbfgs_polish restores the start weights unless the validation score drops.
        out = lbfgs_polish(
            fit_sys, train_data,
            loss_kwargs={'nf': cfg.nf, 'stride': cfg.stride}, spec=spec,
            val_sys_data=val_data,
            val_before=val_before,
            validation_measure=SCORE_NAME)
        # 4. Record the polished weights' score, also when they were not kept.
        if out['ran'] and np.isfinite(out['val_after']):
            history.record('polish_candidate', out['val_after'], accepted=float(out['accepted']))
        return out


def _adam_phase(fit_sys, cfg: RunConfig, data, meta, done_epochs, epochs_remaining, history):
    """Adam for the epochs this job has left, with the PS opening on a fresh run or on a resume from before the
    opening ended; fit ends with the weights of the lowest validation score loaded.

    Args:
        fit_sys, cfg, data: model, settings and records.
        meta: progress of the checkpoint resumed from; None for a fresh run.
        done_epochs: Adam epochs done before this job.
        epochs_remaining: Adam epochs this job runs, at least 1.
        history: this job's validation history; fit adds an 'adam' row at every validation.
    Returns:
        the PS opening, or None without one.
    """
    # 1. Reset the lowest score for this job; fit validates the loaded weights before its first update, so they can
    #    still be selected.
    fit_sys.bestfit = float('inf')
    # 2. PS opening: on a fresh run, or on a resume from a checkpoint saved before the opening ended.
    ps2_state = None if meta is None else meta.get('ps2_opening')
    opening = None
    if cfg.ps2 and (done_epochs == 0 if ps2_state is None else ps2_state[0] != 'S3'):
        # the opening runs after every Adam update until it ends
        opening = PS2Opening(fit_sys, data, n_xb=cfg.nx_phys, n_a=cfg.n_a, width=cfg.n_nodes_per_layer,
                             spec=PS2Spec(**cfg.ps2_opening, seed=cfg.seed))
        fit_sys.training_phase = opening
    # 3. Adam on the remaining budget, with the history and the opening removed from the model afterwards.
    fit_sys.validation_probes = (history,)
    try:
        with on_device(fit_sys, cfg):
            fit_sys.fit(
                train_sys_data=deepSI.System_data_list(data.train_list),
                val_sys_data=deepSI.System_data_list(data.val_list),
                batch_size=cfg.batch_size, epochs=epochs_remaining,
                n_its=cfg.n_its,
                # with an update cap n_its, one validation at its end
                its_per_val=(cfg.n_its if cfg.n_its else cfg.its_per_val),
                loss_kwargs={'nf': cfg.nf, 'stride': cfg.stride},
                cuda=(cfg.device == 'cuda'),
                validation_measure=SCORE_NAME,
            )
    finally:
        fit_sys.validation_probes = ()
        fit_sys.training_phase = None
    if opening is not None:
        print('[ps2] summary: %s' % opening.summary_json(), flush=True)
    return opening


def train(fit_sys, cfg: RunConfig, data, checkpoint_dir, run_id, resume=None, polish_only=False):
    """One job of a run: Adam for the epochs left, then the polish; saves both checkpoint pairs, val_hist.npz and
    loss.npz, and returns the validation score of the final model.

    Args:
        fit_sys, cfg, data: model, settings and records; cfg.epochs is the Adam budget over all jobs, so a resume
            runs only the epochs not done.
        checkpoint_dir: the run folder.
        run_id: run id in the checkpoint names.
        resume: (path of the loaded checkpoint, now in previous/, its progress); None for a fresh run.
        polish_only: skip Adam and polish the loaded weights.
    Returns:
        bestfit: the validation score of the final model: the polished weights if kept, else Adam's.
    """
    # deepSI names its checkpoints AugmentedModel_<unique_code>_best/_last.pth
    fit_sys.unique_code = str(run_id)

    # 1. Progress so far: epochs done and the score to beat; 0 and inf for a fresh run.
    resume_ckpt, meta = (None, None) if resume is None else resume
    done_epochs = 0 if meta is None else int(meta['done_epochs'])
    bestfit = float('inf') if meta is None else float(meta['bestfit'])
    resumed_at_epoch = -1 if meta is None else done_epochs
    # deepSI checkpoints of this job count epochs on from here
    fit_sys.epochs_before = done_epochs
    epochs_remaining = cfg.epochs - done_epochs
    print(f'\nTraining: nf={cfg.nf}  epochs={cfg.epochs}  val={SCORE_NAME}  n_a={cfg.n_a}')
    if meta is not None:
        print(f'  Resumed from {resume_ckpt}: {done_epochs} epochs done, {epochs_remaining} remaining')

    # 2. Adam, if this job has epochs left; afterwards the model holds the weights with the lowest validation score.
    history = ValidationHistory(fit_sys, os.path.join(checkpoint_dir, 'val_hist.npz'))
    adam_ran = not polish_only and epochs_remaining > 0
    opening, elapsed = None, 0.0
    if polish_only:
        print('  polish only: skipping Adam, polishing the resumed weights directly')
    elif not adam_ran:
        print('  Adam budget done (%d of %d epochs): skipping Adam' % (done_epochs, cfg.epochs))
    else:
        t0 = time.time()
        opening = _adam_phase(fit_sys, cfg, data, meta, done_epochs, epochs_remaining, history)
        elapsed = time.time() - t0
        bestfit = fit_sys.bestfit
    # the checkpoint files record the whole Adam budget as done, also when Adam stopped early (n_its cap, NaN loss)
    if not polish_only:
        done_epochs = cfg.epochs

    # 3. Pre-polish checkpoint: the selected weights, from which a polish-only job reruns the polish.
    pre_base = os.path.join(checkpoint_dir, f'gantry_ckpt_{run_id}_pre_lbfgs')
    save_checkpoint_pair(fit_sys, pre_base, save_adam_moments=False, done_epochs=done_epochs, bestfit=bestfit)
    print(f'  Pre-polish checkpoint: {pre_base}.pt')
    # mark the selected Adam validation in the history; on equal scores the latest, as fit keeps the latest best
    adam_rows = [r for r in history.val_hist if r['event'] == 'adam']
    selected = min(reversed(adam_rows), key=lambda r: r['score']) if adam_rows else None
    if selected is not None:
        history.record('selected', selected['score'], row=selected)

    # 4. Polish; the final model is the polished one if kept, else the Adam selection (validated again on the OBC
    #    arm).
    lbfgs_outcome = run_lbfgs_polish(fit_sys, cfg, data, bestfit, history)
    last_by_event = {r['event']: r for r in history.val_hist}
    final = (last_by_event.get('polish_candidate') if lbfgs_outcome['accepted']
             else last_by_event.get('rebuilt') or selected)
    if final is not None:
        history.record('final', final['score'], row=final)
        bestfit = final['score']
    fit_sys.bestfit = bestfit

    # 5. Final checkpoint, without Adam moments after an accepted polish or in a polish-only job, since the saved
    #    moments would not belong to these weights.
    drop_adam_moments = lbfgs_outcome['accepted'] or polish_only
    ckpt_base = os.path.join(checkpoint_dir, f'gantry_ckpt_{run_id}')
    save_checkpoint_pair(
        fit_sys, ckpt_base, save_adam_moments=not drop_adam_moments,
        done_epochs=done_epochs, bestfit=bestfit, lbfgs_outcome=json.dumps(_json_safe(lbfgs_outcome)),
        # resumed_at_epoch: -1 for a fresh run; ps2_summary: the PS opening's record, read by no code
        resumed_at_epoch=resumed_at_epoch, ps2_summary='' if opening is None else opening.summary_json())
    print(f'  Checkpoint weights: {ckpt_base}.pt' + ('  (without Adam moments)' if drop_adam_moments else ''))
    print(f'  Checkpoint progress: {ckpt_base}.npz')

    # 6. loss.npz: the mean training loss over the its_per_val updates before each validation, with the update
    #    count, epoch and time [s since Adam started] at that validation; empty when this job ran no Adam.
    fit_history = fit_sys.fit_history if adam_ran else {}
    np.savez(os.path.join(checkpoint_dir, 'loss.npz'),
             update=np.asarray(fit_history.get('batch_id', []), dtype=np.int64),
             epoch=np.asarray(fit_history.get('epoch_id', []), dtype=float),
             time=np.asarray(fit_history.get('time', []), dtype=float),
             train_loss=np.asarray(fit_history.get('Loss_train', []), dtype=float))

    fit_sys.eval()
    print(f'Training done | {done_epochs} ep | {elapsed:.0f}s | bestfit={bestfit:.5f}')
    return bestfit
