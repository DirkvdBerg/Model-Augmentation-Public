"""Training orchestration (Adam, PS2, selection, L-BFGS polish) and checkpoint I/O."""
__project_origin__ = "added"

import os
import json
import time

import numpy as np
import torch

from .config import RunConfig
from model_augmentation.fit_systems.lbfgs_polish import PolishSpec, lbfgs_polish
from model_augmentation.fit_systems.ps2_opening import PS2Opening, PS2Spec


def _json_safe(obj):
    """numpy scalars out of the outcome dict, so json.dumps cannot fail on the meta write."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def save_checkpoint_weights(fit_sys, base_path, include_optimizer=True):
    """Save the torch components (SSE_Interconnect is a deepSI System, no state_dict). D-070."""
    payload = {
        'hfn':     fit_sys.hfn.state_dict(),
        'encoder': fit_sys.encoder.state_dict(),
    }
    if include_optimizer:
        payload['optimizer'] = fit_sys.optimizer.state_dict()
    torch.save(payload, base_path + '.pt')


def resolve_checkpoint(base_path):
    """Normalise EITHER checkpoint format to (hfn_sd, encoder_sd, optim_sd_or_None, meta)."""
    pair_pt, pair_npz = base_path + '.pt', base_path + '.npz'
    if os.path.exists(pair_pt) and os.path.exists(pair_npz):
        ckpt = torch.load(pair_pt, map_location='cpu')
        # D-279 (F6): read and closed, since a rerun moves this file right after loading it
        with np.load(pair_npz, allow_pickle=True) as z:
            meta = {k: z[k] for k in z.files}
        return ckpt['hfn'], ckpt['encoder'], ckpt.get('optimizer'), meta, 'pair'

    candidates = [base_path] if base_path.endswith('.pth') else [
        base_path + '.pth',
        base_path + '_best.pth',
    ]
    whole = next((c for c in candidates if os.path.exists(c)), None)
    if whole is not None:
        d = torch.load(whole, map_location='cpu', weights_only=False)
        # D-279 (F4): the count is cumulative over chained continuations (epochs_before = the
        # count the dump's job started from); ps2_opening = (phase, updates) of the opening at
        # these weights
        meta = {
            'done_epochs': np.array(int(round(float(d.get('epochs_before', 0))
                                              + float(d.get('epoch_counter', 0))))),
            'bestfit': np.array(float(d.get('bestfit', float('inf')))),
            'ps2_opening': d.get('ps2_opening'),
        }
        _opt = d.get('optimizer') or d.get('_optimizer')
        _opt_sd = _opt.state_dict() if _opt is not None and hasattr(_opt, 'state_dict') else None
        return d['hfn'].state_dict(), d['encoder'].state_dict(), _opt_sd, meta, 'whole'

    raise FileNotFoundError(
        'no checkpoint at %r. Tried the pair %r + %r, and the deepSI whole-system dump(s): %s'
        % (base_path, pair_pt, pair_npz, ', '.join(repr(c) for c in candidates)))


def load_optimizer_moments(optimizer, saved):
    """Restore an optimizer's MOMENT ESTIMATES while keeping this run's hyperparameters (D-173)."""
    live = optimizer.state_dict()
    if len(live['param_groups']) != len(saved['param_groups']):
        raise RuntimeError(
            'optimizer parameter groups do not match: this run has %d, the checkpoint has %d. '
            'The moments cannot be mapped onto these parameters.'
            % (len(live['param_groups']), len(saved['param_groups'])))
    merged = {**saved, 'param_groups': [
        {**live_g, 'params': saved_g['params']}
        for live_g, saved_g in zip(live['param_groups'], saved['param_groups'])
    ]}
    optimizer.load_state_dict(merged)


def _assert_same_architecture(target_sd, ckpt_sd, what):
    """Raise NAMING the mismatch, instead of letting load_state_dict raise a wall of keys."""
    target_coord = {int(v) for k, v in target_sd.items() if k.endswith('coordinate_version')}
    ckpt_coord = {int(v) for k, v in ckpt_sd.items() if k.endswith('coordinate_version')}
    if target_coord != ckpt_coord:
        raise RuntimeError(
            'checkpoint coordinate version does not match this config for %s: config=%s, '
            'checkpoint=%s. Pre-D-229 reduced-coordinate checkpoints cannot be resumed because '
            'the same free_params values have a different physical meaning.'
            % (what, sorted(target_coord), sorted(ckpt_coord)))

    missing = sorted(set(target_sd) - set(ckpt_sd))
    extra = sorted(set(ckpt_sd) - set(target_sd))
    shape = [(k, tuple(target_sd[k].shape), tuple(ckpt_sd[k].shape))
             for k in sorted(set(target_sd) & set(ckpt_sd))
             if tuple(target_sd[k].shape) != tuple(ckpt_sd[k].shape)]
    if not (missing or extra or shape):
        return
    parts = []
    if shape:
        parts.append('%d shape mismatch(es), first 3: %s'
                     % (len(shape), '; '.join('%s config%s != ckpt%s' % s for s in shape[:3])))
    if missing:
        parts.append('%d key(s) the config expects and the checkpoint lacks, first 3: %s'
                     % (len(missing), missing[:3]))
    if extra:
        parts.append('%d key(s) the checkpoint has and the config does not, first 3: %s'
                     % (len(extra), extra[:3]))
    raise RuntimeError(
        'checkpoint does not match this config for %s: %s. The usual cause is a different '
        'nx_ann / n_nodes_per_layer / n_hidden_layers or parameter coordinate than the run that '
        'wrote it.'
        % (what, '; '.join(parts)))


def _assert_same_dtype(fit_sys, hfn_sd):
    """Refuse to load a checkpoint whose dtype differs from the run's."""
    want = next(fit_sys.hfn.parameters()).dtype
    got = next((v.dtype for v in hfn_sd.values()
                if torch.is_tensor(v) and v.is_floating_point()), None)
    if got is not None and got != want:
        raise RuntimeError(
            'checkpoint dtype %s does not match this run\'s %s. Precision must stay coherent: '
            'set use_f64=%s to match the checkpoint, or retrain at %s from scratch. Casting here '
            'would make the pre- and post-phase numbers incomparable, which is the one comparison '
            'the polish phase exists to make.'
            % (got, want, got == torch.float64, want))


def load_checkpoint(fit_sys, base_path, joint_estimation, start_phase='adam'):
    """Restore weights (+ Adam state on an 'adam' start) from either format; return the meta."""
    hfn_sd, enc_sd, optim_sd, meta, fmt = resolve_checkpoint(base_path)
    opening = meta.get('ps2_opening')
    if start_phase == 'adam' and opening is not None and opening[0] != 'S3' and opening[1] > 0:
        # D-279 (F4): the opening's own state (head, optimizers, calibration, RNG) is not in a
        # dump, so a continuation can neither resume it nor restart it on weights it already changed
        raise RuntimeError(
            '%s was written %d updates into an unfinished PS2 opening (phase %s). A continuation '
            'can neither resume nor restart the opening from it; continue from a dump written '
            'before the opening started or after it ended.' % (base_path, opening[1], opening[0]))
    _assert_same_dtype(fit_sys, hfn_sd)
    # CHANGED (D-191): raw JE checkpoints carry log_params; reduced JE checkpoints carry
    # free_params. The exact architecture check below rejects cross-parameterization resumes.
    if joint_estimation and not any(
            k.endswith(('log_params', 'free_params')) for k in hfn_sd):
        raise RuntimeError(
            'RESUME_CHECKPOINT points at a pre-JE checkpoint (no physical parameter vector); '
            'JOINT_ESTIMATION runs must start from a matching JE checkpoint')
    _assert_same_architecture(fit_sys.hfn.state_dict(), hfn_sd, 'hfn')
    _assert_same_architecture(fit_sys.encoder.state_dict(), enc_sd, 'encoder')
    fit_sys.hfn.load_state_dict(hfn_sd)
    fit_sys.encoder.load_state_dict(enc_sd)
    if optim_sd is not None and start_phase == 'adam':
        load_optimizer_moments(fit_sys.optimizer, optim_sd)
        print("  Adam moment estimates restored; lr/eps/betas come from this run's config.")
    print(f'  Checkpoint format: {fmt}')
    if start_phase == 'adam' and optim_sd is None:
        _dropped = bool(meta['optimizer_state_dropped']) \
            if 'optimizer_state_dropped' in meta else None
        print('  NOTE: this checkpoint carries NO Adam optimizer state'
              + (' (recorded as deliberate: optimizer_state_dropped=True)' if _dropped
                 else ' (whole-system format: deepSI stores a live optimizer, not a state dict)'
                 if fmt == 'whole' else '')
              + '.\n        Adam will start from fresh moment estimates at these weights. That is'
                ' correct for\n        polished weights, but it is NOT a continuation of the'
                ' earlier Adam trajectory.\n        For that, resume'
                ' `gantry_ckpt_<run_id>_pre_lbfgs`, which keeps the matching state.')
    return meta


class ValHistory:
    """D-267: the per-record, per-axis validation RMS of every validation and the selection events.
    Held HERE rather than on fit_sys, so fit()'s best-checkpoint reload (which replaces
    fit_sys.__dict__) cannot truncate it; `hist_path` is rewritten at every entry."""

    def __init__(self, fit_sys, hist_path):
        self.fit_sys = fit_sys
        self.hist_path = hist_path
        self.val_hist = []

    def record(self, event, fit_sys, score, **extra):
        """D-267: one history row from the simulator's LAST validation, which must be the one the
        row describes ('adam', 'rebuilt', 'polish_candidate'), or a copy of an earlier row
        ('selected', 'final', passed as `row`)."""
        row = extra.pop('row', None)
        if row is None:
            sim = getattr(fit_sys, 'simulator', None)
            row = dict(per_channel=np.array(sim.last_per_channel, dtype=float),
                       sim_rms=float(sim.last_sim_rms), score=float(score),
                       update=int(getattr(fit_sys, 'batch_counter', -1)),
                       epoch=float(getattr(fit_sys, 'epoch_counter', float('nan'))))
        self.val_hist.append(dict(row, event=event, **extra))
        if self.hist_path is not None:
            np.savez(self.hist_path, **self.hist_arrays())

    def hist_arrays(self):
        """The history as arrays, one entry per row, for val_hist.npz and the results npz."""
        if not self.val_hist:
            return {}
        sim = getattr(self.fit_sys, 'simulator', None)
        h = self.val_hist
        return dict(
            val_hist_event=np.array([r['event'] for r in h]),
            val_hist_update=np.array([r['update'] for r in h]),
            val_hist_epoch=np.array([r['epoch'] for r in h]),
            val_hist_score=np.array([r['score'] for r in h]),
            val_hist_sim_rms=np.array([r['sim_rms'] for r in h]),
            val_hist_rms=np.stack([r['per_channel'] for r in h]),
            val_hist_accepted=np.array([r.get('accepted', np.nan) for r in h], dtype=float),
            val_sigma_e=(np.asarray(sim.val_sigma_e) if getattr(sim, 'val_sigma_e', None) is not None
                         else np.array([])))

    def __call__(self, fit_sys, val_sys_data, value):
        """validation_probes entry: side effects only, `value` is read-only here."""
        self.record('adam', fit_sys, value)


def run_lbfgs_polish(fit_sys, hp, cfg: RunConfig, data, bestfit=float('nan'), history=None,
                     adam_ran=True):
    """The D-171 phase, with the device round-trip wired in. `adam_ran`: False when this job ran no
    Adam update (polish-only, or a resume with the budget spent)."""
    print('\n[lbfgs] polish phase (D-171): Adam is done, quasi-Newton from here')
    spec = PolishSpec(
        n_windows=cfg.lbfgs_windows, max_iter=cfg.lbfgs_max_iter,
        inner_iter=cfg.lbfgs_inner_iter, history_size=cfg.lbfgs_history,
        tol_grad=cfg.lbfgs_tol_grad, tol_change=cfg.lbfgs_tol_change,
        seed=cfg.seed)
    if cfg.device == 'cuda':
        fit_sys.cuda()
        torch.cuda.empty_cache()
    try:
        # D-274 (replaces the D-253 comparison): the polish must beat the selected checkpoint's
        # score `bestfit`, for OBC as well. With OBC, fit() rebuilt the basis at the restored
        # parameters, so the rebuilt model scores slightly differently; it is measured once for the
        # log and the history, and it is the model kept when the polish is rejected.
        # D-278: a polish-only start uses the selected score of its checkpoint too (Dirk:
        # best checkpoint), no longer the start model's remeasured score.
        val_before = (bestfit if np.isfinite(bestfit) else None)
        if not adam_ran and not cfg.obc:
            # The remeasure the polish skips when val_before is given ran the eager forward that
            # completes lazy initialisation before the compiled probe (lbfgs_polish `_validate`).
            fit_sys.cal_validation_error(data.val_ckpt_data, cfg.validation_measure)
        if cfg.obc:
            val_rebuilt = float(fit_sys.cal_validation_error(data.val_ckpt_data, cfg.validation_measure))
            print('[lbfgs] rebuilt-basis validation %s %.6e (selected checkpoint: %.6e)'
                  % (cfg.validation_measure, val_rebuilt, bestfit))
            if history is not None:
                history.record('rebuilt', fit_sys, val_rebuilt)
            if val_before is None:
                val_before = val_rebuilt
        out = lbfgs_polish(
            fit_sys, data.train_data,
            loss_kwargs={'nf': hp['nf'], 'stride': cfg.stride}, spec=spec,
            val_sys_data=data.val_ckpt_data,
            val_before=val_before,
            validation_measure=cfg.validation_measure,
            verbose=True)
        if history is not None and out.get('ran') and np.isfinite(out.get('val_after', np.nan)):
            history.record('polish_candidate', fit_sys, out['val_after'],
                           accepted=float(bool(out.get('accepted'))))
        return out
    finally:
        if cfg.device == 'cuda':
            fit_sys.cpu()


def train(fit_sys, hp, cfg: RunConfig, data, resume=None, checkpoint_dir=None, run_id=None):
    """Train for hp['epochs'] epochs with sim-NRMSe validation, then the L-BFGS polish (D-171).
    `resume` = (path, meta) of the checkpoint main() has already loaded into fit_sys (D-279 F6)."""
    done_epochs = 0
    resumed_at_epoch = -1
    lbfgs_outcome = None

    if run_id:
        fit_sys.unique_code = str(run_id)
    bestfit = float('inf')

    resume_ckpt, meta = (None, None) if resume is None else resume
    if meta is not None:
        done_epochs = int(meta['done_epochs'])
        resumed_at_epoch = done_epochs
        bestfit     = float(meta['bestfit'])
        print(f'Resumed from {resume_ckpt}  ({done_epochs} epochs done)')
        # D-279 (F4): this job's dumps count on from here. Weights and Adam moments continue; the
        # batch stream and an unfinished PS2 opening do not, so this is no uninterrupted trajectory.
        fit_sys.epochs_before = done_epochs

    epochs_remaining = hp['epochs'] - done_epochs
    print(f'\nTraining: nf={hp["nf"]}  epochs={hp["epochs"]}  val={cfg.validation_measure}  '
          f'NX_ANN={hp["NX_ANN"]}')
    if done_epochs > 0:
        print(f'  Resuming: {done_epochs} done, {epochs_remaining} remaining')

    fit_sys.bestfit = float('inf')
    t0 = time.time()
    opening = None
    # D-279 (F4): a dump written after the opening ended never restarts it, even when its count
    # rounds to 0 (load_checkpoint refuses one written inside it); without a record, the count decides
    ps2_state = None if meta is None else meta.get('ps2_opening')
    if cfg.ps2 and cfg.start_phase == 'adam' and (
            done_epochs == 0 if ps2_state is None else ps2_state[0] != 'S3'):
        spec = PS2Spec(seed=cfg.seed)
        if cfg.ps2_smoke:
            spec = PS2Spec(seed=cfg.seed, eval_every=2, s1_min=0, s1_cap=4, screen_at=-1, screen_max=float('inf'),
                           n_cal=32, n_snap=256, n_snap_cal=64, s2_batch=64, s2_check=50, s2_min=50, s2_cap=100)
        opening = PS2Opening.for_pipeline(fit_sys, data, n_xb=cfg.nx_phys, n_a=cfg.nx_ann,
                                          width=hp['n_nodes_per_layer'], spec=spec)
        fit_sys.training_phase = opening
    prev = fit_sys.validation_probes
    probe = ValHistory(fit_sys, os.path.join(checkpoint_dir, 'val_hist.npz'))
    fit_sys.validation_probes = tuple(prev) + (probe,)
    _prev_probes = prev
    history = fit_sys.validation_probes[-1]
    adam_ran = cfg.start_phase != 'lbfgs' and epochs_remaining > 0
    try:
        if cfg.start_phase == 'lbfgs':
            print('  start_phase=lbfgs: skipping Adam, polishing the resumed weights directly')
        elif epochs_remaining <= 0:
            # D-278: the Adam budget is already spent; fit(epochs=0) would not be a no-op. Polish
            # directly.
            print('  Adam budget done (%d of %d epochs): skipping Adam' % (done_epochs, hp['epochs']))
        else:
            eps_values = {float(group['eps']) for group in fit_sys.optimizer.param_groups}
            if eps_values != {float(cfg.adam_eps)}:
                raise RuntimeError(
                    f'Adam eps mismatch before training: config={cfg.adam_eps}, '
                    f'optimizer={sorted(eps_values)}. Resume with the checkpoint epsilon or '
                    'start a fresh optimizer.')
            if cfg.device == 'cuda':
                fit_sys.cuda()
            fit_sys.fit(
                train_sys_data=data.train_data, val_sys_data=data.val_ckpt_data,
                batch_size=hp['batch_size'], epochs=epochs_remaining,
                n_its=cfg.n_its,
                its_per_val=(cfg.n_its if cfg.n_its else (cfg.its_per_val or 'epoch')),
                loss_kwargs={'nf': hp['nf'], 'stride': cfg.stride},
                cuda=(cfg.device == 'cuda'),
                validation_measure=cfg.validation_measure,
            )
            if cfg.device == 'cuda':
                fit_sys.cpu()
            bestfit = fit_sys.bestfit
    finally:
        fit_sys.validation_probes = _prev_probes
        fit_sys.training_phase = None
    if opening is not None:
        print('[ps2] summary: %s' % opening.summary_json(), flush=True)
    elapsed = time.time() - t0
    if cfg.start_phase != 'lbfgs':
        done_epochs = hp['epochs']

    if cfg.lbfgs and checkpoint_dir is not None:
        pre_base = os.path.join(checkpoint_dir, f'gantry_ckpt_{run_id}_pre_lbfgs')
        save_checkpoint_weights(fit_sys, pre_base)
        np.savez(pre_base + '.npz', done_epochs=np.array(done_epochs),
                 bestfit=np.array(bestfit), hp=np.array(json.dumps(hp)),
                 orig_run_id=np.array(run_id))
        print(f'  Pre-polish checkpoint: {pre_base}.pt')
    _adam = [r for r in history.val_hist if r['event'] == 'adam']
    selected = min(reversed(_adam), key=lambda r: r['score']) if _adam else None
    if selected is not None:
        history.record('selected', fit_sys, selected['score'], row=selected)
    lbfgs_outcome = run_lbfgs_polish(fit_sys, hp, cfg, data, bestfit=bestfit, history=history,
                                     adam_ran=adam_ran)
    _last = {r['event']: r for r in history.val_hist}
    final = (_last.get('polish_candidate') if lbfgs_outcome and lbfgs_outcome.get('accepted')
             else _last.get('rebuilt') or selected)
    if final is not None:
        history.record('final', fit_sys, final['score'], row=final)
    bestfit_selected = bestfit
    if final is not None:
        bestfit = final['score']
    elif lbfgs_outcome is not None and lbfgs_outcome.get('accepted'):
        bestfit = lbfgs_outcome['val_after']
    fit_sys.bestfit = bestfit

    ckpt_base = None
    _polished = bool(lbfgs_outcome is not None and lbfgs_outcome.get('accepted'))
    _no_adam_ran = (cfg.start_phase == 'lbfgs')
    _drop_optimizer = _polished or _no_adam_ran
    if checkpoint_dir is not None:
        ckpt_base = os.path.join(checkpoint_dir, f'gantry_ckpt_{run_id}')
        save_checkpoint_weights(fit_sys, ckpt_base, include_optimizer=not _drop_optimizer)
        _why = ('weights were polished' if _polished else
                'no Adam phase ran in this job' if _no_adam_ran else '')
        print(f'  Checkpoint weights: {ckpt_base}.pt'
              + (f'  (Adam state omitted: {_why})' if _drop_optimizer else ''))

    fit_sys.eval()
    print(f'Training done | {done_epochs} ep | {elapsed:.0f}s | bestfit={bestfit:.5f}')

    if ckpt_base is not None:
        np.savez(ckpt_base + '.npz',
                 done_epochs    = np.array(done_epochs),
                 bestfit        = np.array(bestfit),
                 bestfit_selected = np.array(bestfit_selected),
                 elapsed        = np.array(elapsed),
                 hp             = np.array(json.dumps(hp)),
                 orig_run_id    = np.array(run_id),
                 lbfgs_outcome  = np.array(json.dumps(_json_safe(lbfgs_outcome))),
                 optimizer_state_dropped = np.array(_drop_optimizer),
                 start_phase    = np.array(cfg.start_phase),
                 resumed_from   = np.array('' if resume_ckpt is None else str(resume_ckpt)),
                 resumed_at_epoch = np.array(resumed_at_epoch),
                 ps2_summary    = np.array('' if opening is None else opening.summary_json()),
        )
        print(f'  Checkpoint meta: {ckpt_base}.npz')

    return bestfit


def resolve_resume_hp(cfg: RunConfig, default_hp: dict, resume_ckpt) -> dict:
    """@added (2026-09-15). The hp a resumed run actually trains with (D-179)."""
    if resume_ckpt:
        _STRUCTURAL = ('NX_ANN', 'n_nodes_per_layer', 'n_hidden_layers', 'na_nb', 'up_sample')
        print(f'\nResuming checkpoint: {resume_ckpt}')
        if os.path.exists(resume_ckpt + '.npz'):
            _meta = np.load(resume_ckpt + '.npz', allow_pickle=True)
            _saved_hp = json.loads(str(_meta['hp']))
            hp = {**default_hp,
                  **{k: _saved_hp[k] for k in _STRUCTURAL if k in _saved_hp}}
            print('  hp: architecture from the checkpoint, everything else from this config')
            _diff = [(k, _saved_hp[k], hp[k]) for k in hp
                     if k in _saved_hp and _saved_hp[k] != hp[k]]
            for _k, _was, _now in _diff:
                print(f'      {_k:<20s} checkpoint {_was!r:>10s}  ->  this run {_now!r}'
                      + ('   [OBJECTIVE CHANGE]' if _k in ('nf', 'batch_size') else ''))
            if not _diff:
                print('      (identical on every field)')
        else:
            hp = default_hp
            print('  No .npz alongside it (deepSI whole-system format); using this config\'s hp')
            print(f"  hp from THIS config: nf={hp['nf']} stride={cfg.stride} "
                  f"na_nb={hp['na_nb']}. If the checkpoint was trained at another nf, the "
                  f"resumed run optimises a different objective than the one that made it.")
        print(f'  start_phase: {cfg.start_phase}')
    else:
        hp = default_hp
    return hp
