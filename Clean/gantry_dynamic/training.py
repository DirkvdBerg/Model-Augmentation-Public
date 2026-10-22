"""Training: Adam with the PS2 opening, checkpoint selection, the L-BFGS polish and the checkpoint files."""
__project_origin__ = "added"

import os
import json
import time

import numpy as np
import torch

from .config import RunConfig
from .model import on_device
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
    """Save component state dictionaries to <base>.pt and metadata arrays to <base>.npz."""
    payload = {
        'hfn':     fit_sys.hfn.state_dict(),
        'encoder': fit_sys.encoder.state_dict(),
    }
    if include_optimizer:
        payload['optimizer'] = fit_sys.optimizer.state_dict()
    torch.save(payload, base_path + '.pt')
    np.savez(base_path + '.npz', **{k: np.array(v) for k, v in meta.items()})


def resolve_checkpoint(base_path):
    """Read either checkpoint format to (hfn_sd, encoder_sd, optimizer_sd or None, meta, format)."""
    pair_pt, pair_npz = base_path + '.pt', base_path + '.npz'
    if os.path.exists(pair_pt) and os.path.exists(pair_npz):
        ckpt = torch.load(pair_pt, map_location='cpu')
        # read and closed: a rerun moves this file right after loading it
        with np.load(pair_npz, allow_pickle=True) as z:
            meta = {k: z[k] for k in z.files}
        return ckpt['hfn'], ckpt['encoder'], ckpt.get('optimizer'), meta, 'pair'

    if not (base_path.endswith('.pth') and os.path.exists(base_path)):
        raise FileNotFoundError(
            'no checkpoint at %r: neither the pair %r, %r nor a deepSI dump %r exists; check the run folder '
            'and the --from choice' % (base_path, pair_pt, pair_npz, base_path))
    d = torch.load(base_path, map_location='cpu', weights_only=False)
    # the epoch count is cumulative over chained continuations (epochs_before is the count the dump's job
    # started from); ps2_opening = (phase, updates) of the opening at these weights
    meta = {
        'done_epochs': np.array(int(round(float(d.get('epochs_before', 0))
                                          + float(d.get('epoch_counter', 0))))),
        'bestfit': np.array(float(d.get('bestfit', float('inf')))),
        'ps2_opening': d.get('ps2_opening'),
    }
    optimizer = d.get('optimizer') or d.get('_optimizer')
    optimizer_state = (optimizer.state_dict() if optimizer is not None and hasattr(optimizer, 'state_dict')
                       else None)
    return d['hfn'].state_dict(), d['encoder'].state_dict(), optimizer_state, meta, 'whole'


def load_optimizer_moments(optimizer, saved):
    """Restore the optimizer's moment estimates and keep this run's hyperparameters."""
    live = optimizer.state_dict()
    if len(live['param_groups']) != len(saved['param_groups']):
        raise RuntimeError(
            'optimizer parameter groups do not match (this run %d, the checkpoint %d); resume with the '
            'configuration that wrote the checkpoint'
            % (len(live['param_groups']), len(saved['param_groups'])))
    merged = {**saved, 'param_groups': [
        {**live_g, 'params': saved_g['params']}
        for live_g, saved_g in zip(live['param_groups'], saved['param_groups'])
    ]}
    optimizer.load_state_dict(merged)


def _assert_same_architecture(target_sd, ckpt_sd, what):
    """Raise naming the mismatch, rather than load_state_dict's list of keys."""
    target_coord = {int(v) for k, v in target_sd.items() if k.endswith('coordinate_version')}
    ckpt_coord = {int(v) for k, v in ckpt_sd.items() if k.endswith('coordinate_version')}
    if target_coord != ckpt_coord:
        raise RuntimeError(
            'checkpoint coordinate version does not match this config for %s (config %s, checkpoint %s): its '
            'free_params mean other parameters, so start a new run instead of resuming it'
            % (what, sorted(target_coord), sorted(ckpt_coord)))

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
    """Refuse a checkpoint whose dtype differs from the run's."""
    want = next(fit_sys.hfn.parameters()).dtype
    got = next((v.dtype for v in hfn_sd.values()
                if torch.is_tensor(v) and v.is_floating_point()), None)
    if got is not None and got != want:
        raise RuntimeError(
            'checkpoint dtype %s does not match this run\'s %s; use a matching float64 checkpoint or the mainline '
            'implementation configured for the checkpoint precision' % (got, want))


def load_checkpoint(fit_sys, base_path, joint_estimation, start_phase='adam'):
    """Restore the weights, and on an 'adam' start the Adam moments, from either format; returns the meta."""
    hfn_sd, enc_sd, optim_sd, meta, fmt = resolve_checkpoint(base_path)
    opening = meta.get('ps2_opening')
    if start_phase == 'adam' and opening is not None and opening[0] != 'S3' and opening[1] > 0:
        # a dump holds no state of the opening itself (decoder, optimizers, calibration, random state), so a
        # continuation can neither resume it nor restart it on weights it already changed
        raise RuntimeError(
            '%s was written %d updates into an unfinished PS2 opening (phase %s); continue from a dump '
            'written before the opening started or after it ended' % (base_path, opening[1], opening[0]))
    _assert_same_dtype(fit_sys, hfn_sd)
    _assert_same_architecture(fit_sys.hfn.state_dict(), hfn_sd, 'hfn')
    _assert_same_architecture(fit_sys.encoder.state_dict(), enc_sd, 'encoder')
    fit_sys.hfn.load_state_dict(hfn_sd)
    fit_sys.encoder.load_state_dict(enc_sd)
    if optim_sd is not None and start_phase == 'adam':
        load_optimizer_moments(fit_sys.optimizer, optim_sd)
        print("  Adam moment estimates restored; lr/eps/betas come from this run's config.")
    print(f'  Checkpoint format: {fmt}')
    if start_phase == 'adam' and optim_sd is None:
        optimizer_state_dropped = bool(meta.get('optimizer_state_dropped', False))
        print('  This checkpoint has no Adam state'
              + (' (dropped on purpose: optimizer_state_dropped=True)' if optimizer_state_dropped
                 else ' (whole-system format: deepSI stores a live optimizer, not a state dict)'
                 if fmt == 'whole' else '')
              + '; Adam starts from fresh moment estimates at these weights, which suits polished weights'
                ' but does not continue the earlier Adam trajectory. To continue it, resume'
                ' gantry_ckpt_<run_id>_pre_lbfgs, which keeps the matching state.')
    return meta


class ValidationHistory:
    """The validation RMS per record and axis at every validation, and the selection events, in val_hist.npz.

    Held here and not on the model, since fit's reload of the best checkpoint overwrites the model's __dict__;
    hist_path is rewritten at every entry."""

    def __init__(self, fit_sys, hist_path):
        self.fit_sys = fit_sys
        self.hist_path = hist_path
        self.val_hist = []

    def record(self, event, fit_sys, score, row=None, **extra):
        """Append a row: from the simulator's last validation (events adam, rebuilt, polish_candidate), or a
        copy of an earlier row passed as row (selected, final)."""
        if row is None:
            sim = fit_sys.simulator
            row = dict(per_channel=np.array(sim.last_per_channel, dtype=float),
                       sim_rms=float(sim.last_sim_rms), score=float(score),
                       update=int(getattr(fit_sys, 'batch_counter', -1)),
                       epoch=float(getattr(fit_sys, 'epoch_counter', float('nan'))))
        self.val_hist.append(dict(row, event=event, **extra))
        np.savez(self.hist_path, **self.hist_arrays())

    def hist_arrays(self):
        """The history as arrays, one entry per row, for val_hist.npz."""
        h = self.val_hist
        return dict(
            val_hist_event=np.array([r['event'] for r in h]),
            val_hist_update=np.array([r['update'] for r in h]),
            val_hist_epoch=np.array([r['epoch'] for r in h]),
            val_hist_score=np.array([r['score'] for r in h]),
            val_hist_sim_rms=np.array([r['sim_rms'] for r in h]),
            val_hist_rms=np.stack([r['per_channel'] for r in h]),
            val_hist_accepted=np.array([r.get('accepted', np.nan) for r in h], dtype=float),
            val_sigma_e=np.asarray(self.fit_sys.simulator.val_sigma_e))

    def __call__(self, fit_sys, val_sys_data, value):
        """Validation-probe entry: record the validation that just ran."""
        self.record('adam', fit_sys, value)


def run_lbfgs_polish(fit_sys, hp, cfg: RunConfig, data, bestfit, history, adam_ran=True):
    """The L-BFGS polish on cfg.device; adam_ran is False when this job ran no Adam update."""
    print('\n[lbfgs] polish phase: Adam is done, L-BFGS from here')
    spec = PolishSpec(
        n_windows=cfg.lbfgs_windows, max_iter=cfg.lbfgs_max_iter,
        inner_iter=cfg.lbfgs_inner_iter, history_size=cfg.lbfgs_history,
        tol_grad=cfg.lbfgs_tol_grad, tol_change=cfg.lbfgs_tol_change,
        seed=cfg.seed)
    with on_device(fit_sys, cfg):
        if cfg.device == 'cuda':
            torch.cuda.empty_cache()
        # the polish must beat the selected checkpoint's score bestfit, on the OBC arm too. There fit() has
        # rebuilt the basis at the restored parameters, so the rebuilt model scores slightly differently: it is
        # measured once for the log and the history, and it is the model kept when the polish is rejected.
        val_before = (bestfit if np.isfinite(bestfit) else None)
        if not adam_ran and not cfg.obc:
            # without Adam in this job, one eager validation completes the lazy initialisation before the
            # compiled probe of the polish
            fit_sys.cal_validation_error(data.val_ckpt_data, cfg.validation_measure)
        if cfg.obc:
            val_rebuilt = float(fit_sys.cal_validation_error(data.val_ckpt_data, cfg.validation_measure))
            print('[lbfgs] rebuilt-basis validation %s %.6e (selected checkpoint: %.6e)'
                  % (cfg.validation_measure, val_rebuilt, bestfit))
            history.record('rebuilt', fit_sys, val_rebuilt)
            if val_before is None:
                val_before = val_rebuilt
        out = lbfgs_polish(
            fit_sys, data.train_data,
            loss_kwargs={'nf': hp['nf'], 'stride': cfg.stride}, spec=spec,
            val_sys_data=data.val_ckpt_data,
            val_before=val_before,
            validation_measure=cfg.validation_measure)
        if out.get('ran') and np.isfinite(out.get('val_after', np.nan)):
            history.record('polish_candidate', fit_sys, out['val_after'],
                           accepted=float(bool(out.get('accepted'))))
        return out


def train(fit_sys, hp, cfg: RunConfig, data, checkpoint_dir, run_id, resume=None):
    """Run the remaining Adam budget when requested, then polish and save the selected model; return its final score."""
    done_epochs = 0
    resumed_at_epoch = -1

    fit_sys.unique_code = str(run_id)
    bestfit = float('inf')

    # resume = (path, meta) of the checkpoint main has loaded into fit_sys
    resume_ckpt, meta = (None, None) if resume is None else resume
    if meta is not None:
        done_epochs = int(meta['done_epochs'])
        resumed_at_epoch = done_epochs
        bestfit     = float(meta['bestfit'])
        print(f'Resumed from {resume_ckpt}  ({done_epochs} epochs done)')
        # the dumps of this job count on from here; weights and Adam moments continue, the batch stream and an
        # unfinished PS2 opening do not
        fit_sys.epochs_before = done_epochs

    epochs_remaining = hp['epochs'] - done_epochs
    print(f'\nTraining: nf={hp["nf"]}  epochs={hp["epochs"]}  val={cfg.validation_measure}  '
          f'NX_ANN={hp["NX_ANN"]}')
    if done_epochs > 0:
        print(f'  Resuming: {done_epochs} done, {epochs_remaining} remaining')

    # Adam, with the PS2 opening after each update and the validation history as a probe
    fit_sys.bestfit = float('inf')
    t0 = time.time()
    opening = None
    # a dump written after the opening ended never restarts it, even when its count rounds to 0; without a
    # record, the count decides
    ps2_state = None if meta is None else meta.get('ps2_opening')
    if cfg.ps2 and cfg.start_phase == 'adam' and (
            done_epochs == 0 if ps2_state is None else ps2_state[0] != 'S3'):
        spec = PS2Spec(seed=cfg.seed, **(dict(
            eval_every=2, s1_min=0, s1_cap=4, screen_at=-1, screen_max=float('inf'), n_cal=32,
            n_snap=256, n_snap_cal=64, s2_batch=64, s2_check=50, s2_min=50, s2_cap=100) if cfg.ps2_smoke else {}))
        opening = PS2Opening(fit_sys, data, n_xb=cfg.nx_phys, n_a=cfg.nx_ann,
                             width=hp['n_nodes_per_layer'], spec=spec)
        fit_sys.training_phase = opening
    prev = fit_sys.validation_probes
    history = ValidationHistory(fit_sys, os.path.join(checkpoint_dir, 'val_hist.npz'))
    fit_sys.validation_probes = tuple(prev) + (history,)
    adam_ran = cfg.start_phase != 'lbfgs' and epochs_remaining > 0
    try:
        if cfg.start_phase == 'lbfgs':
            print('  start_phase=lbfgs: skipping Adam, polishing the resumed weights directly')
        elif epochs_remaining <= 0:
            # skip fit's validation and checkpoint cycle when the Adam budget is spent
            print('  Adam budget done (%d of %d epochs): skipping Adam' % (done_epochs, hp['epochs']))
        else:
            with on_device(fit_sys, cfg):
                fit_sys.fit(
                    train_sys_data=data.train_data, val_sys_data=data.val_ckpt_data,
                    batch_size=hp['batch_size'], epochs=epochs_remaining,
                    n_its=cfg.n_its,
                    its_per_val=(cfg.n_its if cfg.n_its else cfg.its_per_val),
                    loss_kwargs={'nf': hp['nf'], 'stride': cfg.stride},
                    cuda=(cfg.device == 'cuda'),
                    validation_measure=cfg.validation_measure,
                )
            bestfit = fit_sys.bestfit
    finally:
        fit_sys.validation_probes = prev
        fit_sys.training_phase = None
    if opening is not None:
        print('[ps2] summary: %s' % opening.summary_json(), flush=True)
    elapsed = time.time() - t0
    if cfg.start_phase != 'lbfgs':
        done_epochs = hp['epochs']

    # pre-polish checkpoint and the selected row (the latest of equal scores, as fit keeps the latest best)
    pre_base = os.path.join(checkpoint_dir, f'gantry_ckpt_{run_id}_pre_lbfgs')
    save_checkpoint_pair(fit_sys, pre_base, done_epochs=done_epochs, bestfit=bestfit,
                         hp=json.dumps(hp), orig_run_id=run_id)
    print(f'  Pre-polish checkpoint: {pre_base}.pt')
    adam_rows = [r for r in history.val_hist if r['event'] == 'adam']
    selected = min(reversed(adam_rows), key=lambda r: r['score']) if adam_rows else None
    if selected is not None:
        history.record('selected', fit_sys, selected['score'], row=selected)

    # L-BFGS polish; the final row is the accepted polish, else the rebuilt basis (OBC), else the selection
    lbfgs_outcome = run_lbfgs_polish(fit_sys, hp, cfg, data, bestfit, history, adam_ran=adam_ran)
    last_by_event = {r['event']: r for r in history.val_hist}
    final = (last_by_event.get('polish_candidate') if lbfgs_outcome.get('accepted')
             else last_by_event.get('rebuilt') or selected)
    bestfit_selected = bestfit
    if final is not None:
        history.record('final', fit_sys, final['score'], row=final)
        bestfit = final['score']
    fit_sys.bestfit = bestfit

    # final checkpoint, without Adam state after an accepted polish or in a polish-only job (it would not
    # belong to these weights)
    polish_accepted = bool(lbfgs_outcome.get('accepted'))
    polish_only = (cfg.start_phase == 'lbfgs')
    drop_optimizer_state = polish_accepted or polish_only
    ckpt_base = os.path.join(checkpoint_dir, f'gantry_ckpt_{run_id}')
    save_checkpoint_pair(
        fit_sys, ckpt_base, include_optimizer=not drop_optimizer_state,
        done_epochs=done_epochs, bestfit=bestfit, bestfit_selected=bestfit_selected, elapsed=elapsed,
        hp=json.dumps(hp), orig_run_id=run_id, lbfgs_outcome=json.dumps(_json_safe(lbfgs_outcome)),
        optimizer_state_dropped=drop_optimizer_state, start_phase=cfg.start_phase,
        resumed_from='' if resume_ckpt is None else str(resume_ckpt), resumed_at_epoch=resumed_at_epoch,
        ps2_summary='' if opening is None else opening.summary_json())
    drop_reason = ('weights were polished' if polish_accepted else
                   'no Adam phase ran in this job' if polish_only else '')
    print(f'  Checkpoint weights: {ckpt_base}.pt'
          + (f'  (Adam state omitted: {drop_reason})' if drop_optimizer_state else ''))
    print(f'  Checkpoint meta: {ckpt_base}.npz')

    fit_sys.eval()
    print(f'Training done | {done_epochs} ep | {elapsed:.0f}s | bestfit={bestfit:.5f}')
    return bestfit
