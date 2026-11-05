"""Evaluation of a finished run: the whole-record simulation error of the augmented model and of the baseline with the
trained and with the true parameters on the training, validation and test records, with the figures and the data
they are drawn from.

The score NRMS_e is the RMS of the prediction error y_hat - y over the standard deviation sigma_e of the servo error
e = y - r, per record and axis; with the predicted servo error e_hat = y_hat - r, the prediction error is also
e_hat - e. Scores are given in the stage frame (X1, X2, Y), as measured, and in the generalised frame (X, Theta, Y),
with Theta the beam rotation, mapped by P_INV_T of gantry_ss; each frame uses sigma_e of the servo error in that
frame.
"""
__project_origin__ = "added"

import copy
import csv
import json
import os
import time

import numpy as np
import torch
from matplotlib.figure import Figure

from model_augmentation.fit_systems.blocks import Static_ANN_Block
from model_augmentation.fit_systems.closed_loop import closed_loop_free_run_rms_batch
from model_augmentation.systems.gantry_ss import P_INV_T

from .config import RunConfig, DATA_ROOT
from .controller import build_controller_bank
from .data import load_datasets, compute_normalization, dataset_dir, servo_sigma, load_record, record_name
from .model import build_model
from .training import load_checkpoint

SPLITS = ('train', 'val', 'test')
# per model: its key, its label in the tables and figures, and its colour; the record figures draw the first two
# (augmented and baseline (trained))
MODELS = (('augmented', 'augmented', '#1f77b4'), ('trained', 'baseline (trained)', '#ff7f0e'),
          ('true', 'baseline (true)', '#9e9e9e'))
# per frame: its axis names, and the map from stage positions (one row per sample) to positions in that frame
FRAMES = {'stage': (('X1', 'X2', 'Y'), lambda a: a),
          'generalised': (('X', 'Theta', 'Y'), lambda a: a @ P_INV_T.T)}
# LaTeX form of an axis name in the figure labels; other names are written as they are
TEX = {'Theta': r'$\Theta$'}


def _baseline_block(fit_sys):
    """The identifiable baseline block of a model: the block that holds beta, the trained parameter coordinates."""
    return next(b for b in fit_sys.hfn.connected_blocks if hasattr(b, 'beta'))


def _without_learned(fit_sys, beta=None):
    """Copy of the augmented model that simulates the baseline alone.

    The network output is multiplied by zero (its output gate) and the OBC coefficient theta_aux is set to zero, so
    neither adds to the baseline; optionally the baseline is set to another beta. The copy keeps the
    trained encoder for its initial state.

    Args:
        fit_sys: the augmented model.
        beta: (10,) beta of the baseline, or None to keep the trained beta.
    Returns:
        the copy.
    """
    baseline_model = copy.deepcopy(fit_sys)
    network = next(b for b in baseline_model.hfn.connected_blocks if isinstance(b, Static_ANN_Block))
    correction = getattr(baseline_model.hfn, 'obc_correction', None)
    with torch.no_grad():
        network.out_gate.zero_()
        if correction is not None:
            correction.theta_aux.zero_()
        if beta is not None:
            _baseline_block(baseline_model).beta.copy_(beta)
    return baseline_model


def _true_combinations(baseline_block):
    """Identifiable combinations of the parameters the data were generated with: the nominal values of gantry_ss,
    held by the baseline block as params_init.

    Args:
        baseline_block: the identifiable baseline block.
    Returns:
        (10,) combinations in physical units, in COMBO_NAMES order.
    """
    # in double precision, so the combinations are rounded to the model's precision only once
    return type(baseline_block).combos_of(baseline_block.params_init.double(), float(baseline_block.Lb)).to(
        baseline_block.beta.dtype)


def _true_beta(baseline_block):
    """beta at which the baseline's ten identifiable combinations equal the true ones; beta is what training
    changes, with each combination = its start value * exp(beta) (m_Delta through a tanh instead).

    Args:
        baseline_block: the identifiable baseline block.
    Returns:
        (10,) beta.
    """
    return baseline_block.beta_from_combinations(_true_combinations(baseline_block))


def _noise_free_twins(cfg: RunConfig):
    """For each record file, the file of its noise-free twin (pair_file in MANIFEST.csv); None when the run uses the
    noise-free data set."""
    if cfg.noise == 'noisefree':
        return None
    with open(os.path.join(dataset_dir(cfg), 'MANIFEST.csv'), newline='') as file:
        return {'%s/%s.mat' % (row['folder'], row['file']): row['pair_file'] for row in csv.DictReader(file)}


def _split_scores(models, record_signals, e_list, e_noise_free_list, bank, rows, k0):
    """Scores of one split in the stage and in the generalised frame.

    Each model simulates each whole record once, in closed loop. Per frame and axis, NRMS_e is the RMS of the prediction
    error y_hat - y from sample k0 on, over the standard deviation sigma_e of the record's servo error in that frame.
    The noise floor is the same score for a model that predicts the noise-free servo error exactly: the RMS of
    e - e_noise_free over sigma_e.

    Args:
        models: the augmented model and the trained and true baseline, by key.
        record_signals: the split's records at fs_new, unnormalised.
        e_list, e_noise_free_list: per record the servo error of the record and of its noise-free twin, one row per
            sample [m].
        bank, rows: controller bank and one bank row per record.
        k0: first scored sample.
    Returns:
        preds: for each model, per record the predicted positions y_hat from sample k0 on [m].
        frame_scores: for each frame a dict with, for each model, residual (per record the prediction error over
            sigma_e at every sample from k0 on), rms (per record and axis [m], Theta in [rad]) and nrms (rms over
            sigma_e); and, per record and axis, sigma (sigma_e) and floor (the noise floor).
    """
    preds = {key: closed_loop_free_run_rms_batch(model, record_signals, bank, rows, return_pred=True)[2]
             for key, model in models.items()}
    y = [np.asarray(signals.y)[k0:] for signals in record_signals]
    frame_scores = {}
    for frame, (_, to_frame) in FRAMES.items():
        sigma = servo_sigma([to_frame(e) for e in e_list], k0)
        error = {key: [to_frame(pred - y_record) for pred, y_record in zip(model_preds, y)]
                 for key, model_preds in preds.items()}
        rms = {key: np.stack([np.sqrt(np.mean(record_error ** 2, axis=0)) for record_error in error[key]])
               for key in error}
        floor = np.stack([np.sqrt(np.mean(to_frame(e - e_noise_free)[k0:] ** 2, axis=0))
                          for e, e_noise_free in zip(e_list, e_noise_free_list)]) / sigma
        frame_scores[frame] = dict(
            residual={key: [record_error / record_sigma for record_error, record_sigma in zip(error[key], sigma)]
                      for key in error},
            rms=rms, nrms={key: model_rms / sigma for key, model_rms in rms.items()}, sigma=sigma, floor=floor)
    return preds, frame_scores


def _table(split, names, records, scores):
    """Terminal table of one split and frame: a row per record, the mean row and the mean over records and axes."""
    cols = [scores['nrms'][key] for key, _, _ in MODELS] + [scores['floor']]
    format_values = lambda v: ' '.join('%6.4f' % x for x in v)
    row = '%-7s | %s | %s | %s | %s'
    lines = ['', '%s  (%d records)   NRMS_e per record and axis' % (split.upper(), len(records)),
             '%-7s | %-20s | %-20s | %-20s | %-20s' % ('record', *[label for _, label, _ in MODELS], 'noise floor'),
             row % ('', *[' '.join('%6s' % name for name in names)] * 4), '-' * 97]
    lines += [row % (record, *(format_values(c[i]) for c in cols)) for i, record in enumerate(records)]
    lines += ['-' * 97, row % ('mean', *(format_values(c.mean(0)) for c in cols)),
              'mean over records and axes: ' + ' | '.join('%s %.4f' % (label, scores['nrms'][key].mean())
                                                           for key, label, _ in MODELS)]
    return lines


def _bar_figure(path, title, names, records, scores):
    """NRMS_e bars per record and a set-apart mean group, one panel per axis, the three models side by side."""
    fig = Figure(figsize=(max(7, 0.5 * len(records) + 2.5), 7))
    panels = fig.subplots(3, 1, sharex=True)
    x = np.r_[np.arange(len(records)), len(records) + 0.6]
    # bar width: the three bars of a record fill 0.81 of the record spacing, leaving a gap between records
    bar_width = 0.27
    for i, name in enumerate(names):
        for j, (key, label, colour) in enumerate(MODELS):
            v = scores['nrms'][key][:, i]
            panels[i].bar(x + (j - 1) * bar_width, np.r_[v, v.mean()], bar_width, color=colour, label=label)
        panels[i].set_ylabel('NRMS$_e$ %s' % TEX.get(name, name))
        panels[i].axvline(len(records) - 0.2, color='0.5', lw=0.8)
        panels[i].grid(alpha=0.3)
    panels[-1].set_xticks(x, records + ['mean'], rotation=45, ha='right')
    fig.legend(*panels[0].get_legend_handles_labels(), ncol=3, loc='upper center', bbox_to_anchor=(0.5, 0.95))
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=110)


def _record_figure(path, title, names, t, residual, ref, nrms, floor):
    """Prediction error over sigma_e per axis from sample k0 on, for baseline (trained) and augmented, with the
    record's NRMS_e beside each panel and the reference below."""
    fig = Figure(figsize=(10, 7.8))
    panels = fig.subplots(4, 1, sharex=True, gridspec_kw=dict(height_ratios=[1, 1, 1, 0.6]))
    for i, name in enumerate(names):
        for key, label, colour in (MODELS[1], MODELS[0]):
            panels[i].plot(t, residual[key][:, i], color=colour, lw=0.6, label=label)
        panels[i].set_ylabel(r'$(\hat e - e)/\sigma_e$  %s' % TEX.get(name, name))
        panels[i].text(1.008, 0.5, 'NRMS$_e$\n%s %.3f\n%s %.3f\nnoise floor %.4f'
                       % (MODELS[1][1], nrms['trained'][i], MODELS[0][1], nrms['augmented'][i], floor[i]),
                       transform=panels[i].transAxes, va='center', fontsize=7.5)
    # the reference of axis 2 (Y) and of axis 0 (X1 in the stage frame, X in the generalised frame)
    panels[3].plot(t, ref[:, 2], color='k', lw=1, label=names[2])
    panels[3].plot(t, ref[:, 0], color='0.55', lw=1, label=names[0])
    panels[3].set_ylabel('reference (m)')
    panels[3].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=7.5)
    panels[3].set_xlabel('t (s)')
    panels[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    for panel in panels:
        panel.grid(alpha=0.3)
    fig.suptitle(title)
    fig.align_ylabels(panels)
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    fig.savefig(path, dpi=110)


def _loss_figure(path, title, loss, hist):
    """Training loss (loss.npz) and validation NRMS_e (the Adam rows of val_hist.npz) against the Adam update count,
    with the selected checkpoint and the L-BFGS result."""
    fig = Figure(figsize=(8, 5.5))
    panels = fig.subplots(2, 1, sharex=True)
    panels[0].semilogy(loss['update'], loss['train_loss'], color=MODELS[0][2])
    panels[0].set_ylabel('training loss\nmean of $((\\hat y - y)/\\sigma_e)^2$\nover n$_f$-step windows')
    event = hist['val_hist_event']
    updates, scores = hist['val_hist_update'][event == 'adam'], hist['val_hist_score'][event == 'adam']
    panels[1].semilogy(updates, scores, color='#2ca02c', label='validation, Adam')
    selected = np.isin(updates, hist['val_hist_update'][event == 'selected'])
    panels[1].scatter(updates[selected], scores[selected], s=50, color='r', zorder=5, label='selected checkpoint')
    polish_rows = event == 'polish_candidate'
    if polish_rows.any():
        panels[1].scatter(updates[-1:], hist['val_hist_score'][polish_rows][-1:], s=70, marker='*',
                          color='purple', zorder=5, label='after L-BFGS polish (%s)'
                          % ('accepted' if hist['val_hist_accepted'][polish_rows][-1] else 'rejected'))
    panels[1].set_ylabel('validation NRMS$_e$\n(free run, mean of records, axes)')
    panels[1].set_xlabel('Adam update')
    panels[1].legend()
    for panel in panels:
        panel.grid(alpha=0.3)
    panels[0].set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def evaluate(run_dir, data_root=DATA_ROOT):
    """Evaluate the final checkpoint of a run folder on every split.

    NRMS_e per record and axis, in a closed-loop simulation of the whole record from the first sample after the
    encoder's input, for the augmented model, the baseline with the trained parameters and the baseline with the
    true parameters. Prints the tables and
    writes the figures and their data to <run_dir>/evaluation/.

    Args:
        run_dir: run folder with config.json and the final checkpoint pair gantry_ckpt_<run_id>.
        data_root: folder of the data sets.
    """
    t_start = time.time()
    # 1. The run on the CPU: configuration, records, final checkpoint.
    with open(os.path.join(run_dir, 'config.json')) as file:
        meta = json.load(file)
    run_id = str(meta['run_id'])
    cfg = RunConfig.from_config_json(meta, device='cpu', compile_mode=None, data_root=data_root)
    data = load_datasets(cfg)
    norm = compute_normalization(cfg, data)
    fit_sys = build_model(cfg, data, norm)
    checkpoint = os.path.join(run_dir, 'gantry_ckpt_%s' % run_id)
    # weights only, no Adam moments
    lbfgs_outcome = json.loads(str(load_checkpoint(fit_sys, checkpoint, polish_only=True)['lbfgs_outcome']))
    if lbfgs_outcome['accepted']:
        checkpoint_state = 'after L-BFGS polish, accepted'
    elif lbfgs_outcome['ran']:
        checkpoint_state = 'L-BFGS polish rejected'
    else:
        checkpoint_state = 'no L-BFGS polish'
    baseline_block = _baseline_block(fit_sys)
    models = {'augmented': fit_sys, 'trained': _without_learned(fit_sys),
              'true': _without_learned(fit_sys, _true_beta(baseline_block))}
    # first scored sample: the encoder uses the first max(na, nb) samples
    k0 = max(fit_sys.na, fit_sys.nb)
    files = [getattr(data, split + '_files') for split in SPLITS]
    bank, rows, _ = build_controller_bank([f for split_files in files for f in split_files], data.controllers,
                                          cfg.ts_new, ystd=norm.ystd, std_u=norm.std_u, dtype=cfg.dtype_pt)
    twins = _noise_free_twins(cfg)
    output_dir = os.path.join(run_dir, 'evaluation')
    print('\n'.join([
        '=' * 100, '[evaluate] run %s  (%s)' % (run_id, run_dir),
        '[evaluate] checkpoint %s.pt (%s)  device cpu  %s'
        % (os.path.basename(checkpoint), checkpoint_state, str(cfg.dtype_pt).replace('torch.', '')),
        '[evaluate] models: augmented | baseline (trained): trained parameters | baseline (true): true parameters;'
        ' both baselines without the network and the OBC correction',
        '[evaluate] NRMS_e = RMS(e_hat - e) / sigma_e per record and axis, from k0 = %d, free-run closed loop' % k0,
        '=' * 100]), flush=True)

    # 2. Per split: whole-record simulations, scores in both frames, figures.
    saved, score_rows, tables = {}, [], {frame: [] for frame in FRAMES}
    offset = 0
    for split, split_files in zip(SPLITS, files):
        record_signals, e_list = getattr(data, split + '_list'), getattr(data, 'e_' + split)
        # the twin is resampled and trimmed like its record, so e - e_noise_free compares the same samples
        e_noise_free = (e_list if twins is None else
                        [load_record(twins[f], cfg, folder=cfg.data_root)[1] for f in split_files])
        preds, split_scores = _split_scores(models, record_signals, e_list, e_noise_free, bank,
                                            rows[offset:offset + len(record_signals)], k0)
        offset += len(record_signals)
        records = [record_name(f) for f in split_files]
        # reference r = y - e of each record [m]
        refs = [np.asarray(signals.y) - e for signals, e in zip(record_signals, e_list)]
        # t = 0 at the first sample kept after the edge trim
        t = [np.arange(len(reference)) * cfg.ts_new for reference in refs]
        os.makedirs(os.path.join(output_dir, split, 'records'), exist_ok=True)
        for i, record in enumerate(records):
            key_prefix = '%s/%s/' % (split, record)
            saved.update({key_prefix + 't': t[i], key_prefix + 'r': refs[i], key_prefix + 'e': e_list[i],
                          key_prefix + 'e_noise_free': e_noise_free[i]})
            saved.update({key_prefix + 'e_hat_' + key: preds[key][i] - refs[i][k0:] for key in models})
        for frame, (names, to_frame) in FRAMES.items():
            scores = split_scores[frame]
            tables[frame] += _table(split, names, records, scores)
            score_rows += [(split, record, frame, key, axis, scores['rms'][key][i, j], scores['nrms'][key][i, j],
                            scores['sigma'][i, j], scores['floor'][i, j])
                           for i, record in enumerate(records) for key in models for j, axis in enumerate(names)]
            _bar_figure(os.path.join(output_dir, split, 'nrms_%s.png' % frame),
                        'run %s, %s: NRMS$_e$ per record and axis, %s frame' % (run_id, split, frame),
                        names, records, scores)
            for i, record in enumerate(records):
                _record_figure(os.path.join(output_dir, split, 'records', '%s_%s.png' % (record, frame)),
                               '%s (%s), run %s, %s frame: prediction error over the record'
                               % (record, split, run_id, frame), names, t[i][k0:],
                               {key: scores['residual'][key][i] for key in ('augmented', 'trained')},
                               to_frame(refs[i][k0:]),
                               {key: scores['nrms'][key][i] for key in ('augmented', 'trained')},
                               scores['floor'][i])
    print('\n'.join(tables['stage'] + ['', 'GENERALISED FRAME (X, Theta, Y)'] + tables['generalised']), flush=True)

    # 3. Physical parameters: true, start and learned identifiable combinations.
    with torch.no_grad():
        names, start, learned = baseline_block.estimation_parameters()
        true = _true_combinations(baseline_block)
    params = [(name, float(true_value), float(start_value), float(learned_value))
              for name, true_value, start_value, learned_value in zip(names, true, start, learned)]
    print('\n'.join(['', 'PHYSICAL PARAMETERS  (identifiable combinations; true = data-generating value)',
                     '%-9s | %12s | %12s | %12s | %9s' % ('', 'true', 'start', 'learned', 'error [%]'), '-' * 66]
                    + ['%-9s | %12.5g | %12.5g | %12.5g | %+8.1f%%'
                       % (name, true_value, start_value, learned_value,
                          100 * (learned_value - true_value) / true_value)
                       for name, true_value, start_value, learned_value in params]), flush=True)

    # 4. Data files and the loss curves.
    # 17 significant digits, so every value in scores.csv and params.csv reads back as the same number
    with open(os.path.join(output_dir, 'scores.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(('split', 'record', 'frame', 'model', 'axis', 'rms', 'nrms_e', 'sigma_e', 'floor_nrms_e'))
        writer.writerows(row[:5] + tuple('%.17g' % v for v in row[5:]) for row in score_rows)
    np.savez(os.path.join(output_dir, 'predictions.npz'), k0=k0, P_inv_T=P_INV_T, **saved)
    with open(os.path.join(output_dir, 'params.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(('name', 'true', 'start', 'learned'))
        writer.writerows((name,) + tuple('%.17g' % v for v in values) for name, *values in params)
    with open(os.path.join(output_dir, 'evaluation.json'), 'w') as file:
        json.dump(dict(checkpoint=os.path.basename(checkpoint) + '.pt', run_id=run_id), file, indent=2)
    loss_file, hist_file = os.path.join(run_dir, 'loss.npz'), os.path.join(run_dir, 'val_hist.npz')
    curves = os.path.exists(loss_file) and os.path.exists(hist_file)
    if curves:
        with np.load(loss_file) as loss, np.load(hist_file) as hist:
            # a job without Adam (polish only, or the Adam budget already done) saves an empty loss.npz
            curves = loss['update'].size > 0
            if curves:
                _loss_figure(os.path.join(output_dir, 'loss.png'), 'run %s: loss curves' % run_id, dict(loss),
                             dict(hist))
    print('\n'.join([
        '', 'saved %s' % output_dir,
        '  scores.csv          %d rows (split, record, frame, model, axis: rms, NRMS_e, sigma_e, floor)'
        % len(score_rows),
        '  predictions.npz     per record: t, r, e, e of the noise-free twin, e_hat per model; k0, P^-T',
        '  params.csv          true, start and learned physical combinations',
        '  evaluation.json     checkpoint, run id',
        '  loss.png            ' + ('training loss and validation NRMS_e, from loss.npz and val_hist.npz' if curves
                                    else 'not drawn: no loss.npz or val_hist.npz, or the last job ran no Adam (the '
                                         'Adam curves are then in previous/)'),
        '  train/ val/ test/   nrms_stage.png, nrms_generalised.png, records/<record>_{stage,generalised}.png '
        '(%d + %d + %d records)' % tuple(len(split_files) for split_files in files),
        '[evaluate] done in %.0f s' % (time.time() - t_start)]), flush=True)
