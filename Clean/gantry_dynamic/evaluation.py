"""Evaluation of a finished run: NRMS_e of the augmented model and of the baseline with the trained and with the
true parameters on the training, validation and test records, with the figures and the data they are drawn from.

Conventions: e = y - r the servo error and e_hat = y_hat - r, so e_hat - e = y_hat - y; samples are rows. Stage
frame (X1, X2, Y) as measured; generalised frame (X, Theta, Y) with q = P^-T y, each axis over its own sigma_e.
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

from .config import RunConfig, DATA_ROOT, git_provenance
from .controller import build_controller_bank
from .data import load_datasets, compute_normalization, traj_dir, servo_sigma, _load_record
from .model import build_model
from .obc_gantry import P_INV_T64
from .training import load_checkpoint

SPLITS = ('train', 'val', 'test')
# key, label, colour; the record figures draw the first two
MODELS = (('augmented', 'augmented', '#1f77b4'), ('trained', 'baseline (trained)', '#ff7f0e'),
          ('true', 'baseline (true)', '#9e9e9e'))
# frame -> (axis names, map of (N, 3) stage arrays into the frame)
FRAMES = {'stage': (('X1', 'X2', 'Y'), lambda a: a),
          'generalised': (('X', 'Theta', 'Y'), lambda a: a @ P_INV_T64.T)}
# axis name in the figures
TEX = {'Theta': r'$\Theta$'}


def _without_learned(fit_sys, free=None):
    """Baseline-only copy of a model.

    The network gate and theta_aux set to zero, optionally at another beta, so the copy simulates the baseline alone.

    Args:
        fit_sys: the augmented model; free: (10,) beta, or None to keep the current one.
    Returns:
        The modified copy.
    """
    m = copy.deepcopy(fit_sys)
    blocks = m.hfn.connected_blocks
    ann = next(b for b in blocks if isinstance(b, Static_ANN_Block))
    corr = getattr(m.hfn, 'obc_correction', None)
    with torch.no_grad():
        ann.out_gate.zero_()
        if corr is not None:
            corr.theta_frozen.zero_()
        if free is not None:
            next(b for b in blocks if hasattr(b, 'free_params')).free_params.copy_(free)
    return m


def _true_combinations(phy):
    """Identifiable combinations of the data-generating parameters.

    theta_base(theta^0), the combinations of the raw parameters theta^0 the data were generated with.

    Args:
        phy: the identifiable baseline block.
    Returns:
        (10,) theta_base, physical, in COMBO_NAMES order and the dtype of beta.
    """
    return type(phy).combos_of(phy.params_init.double(), float(phy.Lb)).to(phy.free_params.dtype)


def _true_free(phy):
    """Training coordinate of the true parameters.

    beta_star = beta(theta_base(theta^0)), the coordinate at which the baseline holds the true combinations.

    Args:
        phy: the identifiable baseline block.
    Returns:
        (10,) beta_star.
    """
    return phy.free_from_combinations(_true_combinations(phy))


def _noise_free_twins(cfg: RunConfig):
    """Record file -> its noise-free twin (MANIFEST.csv column pair_file, relative to the data root); None on the
    noise-free data set, whose records are their own twins."""
    if cfg.noise == 'noisefree':
        return None
    with open(os.path.join(traj_dir(cfg), 'MANIFEST.csv'), newline='') as fh:
        return {'%s/%s.mat' % (row['folder'], row['file']): row['pair_file'] for row in csv.DictReader(fh)}


def _split_scores(models, sdl, e_list, e_nf_list, bank, rows, k0):
    """Scores of one split in both frames.

    One closed-loop free run per record and model gives y_hat; per frame and axis j, with T the frame map
    (identity or P^-T): RMS_j = (mean over k >= k0 of (T(y_hat - y))_j,k^2)^(1/2), NRMS_e,j = RMS_j / sigma_e,j,
    sigma_e = servo_sigma of T e, and the floor = RMS of T(e - e_nf) over sigma_e.

    Args:
        models: model key -> model (augmented, trained and true baseline).
        sdl: R records (physical, at 1 / T_s).
        e_list, e_nf_list: R arrays (N_r, 3), the servo error e = y - r of each record and of its noise-free twin,
            metres.
        bank, rows: controller bank and one bank row per record.
        k0: first scored sample.
    Returns:
        preds {model: R arrays (N_r - k0, 3) of y_hat, metres}, and per frame dict(residual={model: R arrays
        (N_r - k0, 3) of T(e_hat - e) / sigma_e}, rms={model: (R, 3)} in metres (rad for Theta), nrms={model: (R, 3)},
        sigma (R, 3), floor (R, 3) in NRMS_e).
    """
    runs = {key: closed_loop_free_run_rms_batch(m, sdl, bank, rows, return_pred=True) for key, m in models.items()}
    y = [np.asarray(sd.y)[k0:] for sd in sdl]
    out = {}
    for frame, (_, to_frame) in FRAMES.items():
        sigma = servo_sigma([to_frame(e) for e in e_list], k0)
        error = {key: [to_frame(p - yr) for p, yr in zip(run[2], y)] for key, run in runs.items()}
        # stage: the scorer's RMS, the numerator of the validation score; generalised: from the same predictions
        rms = {key: (np.asarray(run[0]) if frame == 'stage'
                     else np.stack([np.sqrt(np.mean(d ** 2, axis=0)) for d in error[key]]))
               for key, run in runs.items()}
        floor = np.stack([np.sqrt(np.mean(to_frame(e - e_nf)[k0:] ** 2, axis=0))
                          for e, e_nf in zip(e_list, e_nf_list)]) / sigma
        out[frame] = dict(residual={key: [d / s for d, s in zip(error[key], sigma)] for key in error}, rms=rms,
                          nrms={key: r / sigma for key, r in rms.items()}, sigma=sigma, floor=floor)
    return {key: run[2] for key, run in runs.items()}, out


def _table(split, names, records, s):
    """Terminal table of one split and frame: a row per record, the mean row and the mean over records and axes."""
    cols = [s['nrms'][key] for key, _, _ in MODELS] + [s['floor']]
    fmt = lambda v: ' '.join('%6.4f' % x for x in v)
    row = '%-7s | %s | %s | %s | %s'
    lines = ['', '%s  (%d records)   NRMS_e per record and axis' % (split.upper(), len(records)),
             '%-7s | %-20s | %-20s | %-20s | %-20s' % ('record', *[lab for _, lab, _ in MODELS], 'noise floor'),
             row % ('', *[' '.join('%6s' % n for n in names)] * 4), '-' * 97]
    lines += [row % (rec, *(fmt(c[i]) for c in cols)) for i, rec in enumerate(records)]
    lines += ['-' * 97, row % ('mean', *(fmt(c.mean(0)) for c in cols)),
              'mean over records and axes: ' + ' | '.join('%s %.4f' % (lab, s['nrms'][key].mean())
                                                           for key, lab, _ in MODELS)]
    return lines


def _bar_figure(path, title, names, records, s):
    """NRMS_e bars per record and a set-apart mean group, one panel per axis, the three models side by side."""
    fig = Figure(figsize=(max(7, 0.5 * len(records) + 2.5), 7))
    axs = fig.subplots(3, 1, sharex=True)
    x = np.r_[np.arange(len(records)), len(records) + 0.6]
    w = 0.27
    for i, name in enumerate(names):
        for j, (key, lab, colour) in enumerate(MODELS):
            v = s['nrms'][key][:, i]
            axs[i].bar(x + (j - 1) * w, np.r_[v, v.mean()], w, color=colour, label=lab)
        axs[i].set_ylabel('NRMS$_e$ %s' % TEX.get(name, name))
        axs[i].axvline(len(records) - 0.2, color='0.5', lw=0.8)
        axs[i].grid(alpha=0.3)
    axs[-1].set_xticks(x, records + ['mean'], rotation=45, ha='right')
    fig.legend(*axs[0].get_legend_handles_labels(), ncol=3, loc='upper center', bbox_to_anchor=(0.5, 0.95))
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=110)


def _record_figure(path, title, names, t, residual, ref, nrms, floor):
    """(e_hat - e) / sigma_e over the whole record per axis for baseline (trained) and augmented, the record NRMS_e
    beside each panel and the reference below."""
    fig = Figure(figsize=(10, 7.8))
    axs = fig.subplots(4, 1, sharex=True, gridspec_kw=dict(height_ratios=[1, 1, 1, 0.6]))
    for i, name in enumerate(names):
        for key, lab, colour in (MODELS[1], MODELS[0]):
            axs[i].plot(t, residual[key][:, i], color=colour, lw=0.6, label=lab)
        axs[i].set_ylabel(r'$(\hat e - e)/\sigma_e$  %s' % TEX.get(name, name))
        axs[i].text(1.008, 0.5, 'NRMS$_e$\nbaseline %.3f\naugmented %.3f\nfloor %.4f'
                    % (nrms['trained'][i], nrms['augmented'][i], floor[i]),
                    transform=axs[i].transAxes, va='center', fontsize=7.5)
    axs[3].plot(t, ref[:, 2], color='k', lw=1, label=names[2])
    axs[3].plot(t, ref[:, 0], color='0.55', lw=1, label=names[0])
    axs[3].set_ylabel('reference (m)')
    axs[3].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=7.5)
    axs[3].set_xlabel('t (s)')
    axs[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    for ax in axs:
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.align_ylabels(axs)
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    fig.savefig(path, dpi=110)


def _loss_figure(path, title, loss):
    """Training loss and validation NRMS_e against the update, with the selected checkpoint and the L-BFGS result."""
    fig = Figure(figsize=(8, 5.5))
    axs = fig.subplots(2, 1, sharex=True)
    upd, val = loss['update'], loss['val_score']
    axs[0].semilogy(upd, loss['train_loss'], color=MODELS[0][2])
    axs[0].set_ylabel('training loss\n(windowed MSE, n$_f$ steps)')
    axs[1].semilogy(upd, val, color='#2ca02c', label='validation, Adam')
    selected = upd == loss['selected_update']
    axs[1].scatter(upd[selected], val[selected], s=50, color='r', zorder=5, label='selected checkpoint')
    if loss['polish_score'].size:
        axs[1].scatter(upd[-1:] if upd.size else [0], loss['polish_score'][-1:], s=70, marker='*', color='purple', zorder=5,
                       label='after L-BFGS polish (%s)' % ('accepted' if loss['polish_accepted'][-1] else 'rejected'))
    axs[1].set_ylabel('validation NRMS$_e$\n(free run, mean of records, axes)')
    axs[1].set_xlabel('Adam update')
    axs[1].legend()
    for ax in axs:
        ax.grid(alpha=0.3)
    axs[0].set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def evaluate(run_dir, data_root=DATA_ROOT):
    """Evaluate the final checkpoint of a run folder on every split.

    NRMS_e = RMS(e_hat - e) / sigma_e per record and axis, from k0, in closed-loop free run, of the augmented model,
    the baseline with the trained parameters (learned part removed) and the baseline with the true parameters; prints
    the tables and writes the figures and their data under <run_dir>/evaluation/.

    Args:
        run_dir: run folder with config.json and the final checkpoint pair gantry_ckpt_<run_id>.
        data_root: folder of the data sets.
    Returns:
        dict split -> records, and per frame rms and nrms per model, sigma and floor.
    """
    t_start = time.time()
    # 1. The run on the CPU: configuration, records and the final checkpoint.
    with open(os.path.join(run_dir, 'config.json')) as fh:
        meta = json.load(fh)
    run_id = str(meta['run_id'])
    cfg = RunConfig.from_config_json(meta, device='cpu', compile_mode=None, data_root=data_root)
    data = load_datasets(cfg)
    norm = compute_normalization(cfg, data)
    fit_sys = build_model(cfg.hp, cfg, data, norm)
    ckpt = os.path.join(run_dir, 'gantry_ckpt_%s' % run_id)
    polish = json.loads(str(load_checkpoint(fit_sys, ckpt, cfg.joint_estimation, start_phase='lbfgs')
                            .get('lbfgs_outcome', '{}')))
    phy = next(b for b in fit_sys.hfn.connected_blocks if hasattr(b, 'free_params'))
    models = {'augmented': fit_sys, 'trained': _without_learned(fit_sys),
              'true': _without_learned(fit_sys, _true_free(phy))}
    k0 = max(fit_sys.na, fit_sys.nb)
    files = [getattr(data, s + '_files') for s in SPLITS]
    bank, rows, _ = build_controller_bank([f for fs in files for f in fs], cfg.ts_new, ystd=norm.ystd,
                                          std_u=norm.std_u, dtype=cfg.dtype_pt)
    twins = _noise_free_twins(cfg)
    out_dir = os.path.join(run_dir, 'evaluation')
    print('\n'.join([
        '=' * 100, '[evaluate] run %s  (%s)' % (run_id, run_dir),
        '[evaluate] checkpoint %s.pt (%s)  device cpu  %s'
        % (os.path.basename(ckpt), 'after L-BFGS polish, accepted' if polish.get('accepted') else
           'L-BFGS polish rejected' if polish.get('ran') else 'no L-BFGS polish', str(cfg.dtype_pt)[6:]),
        '[evaluate] models: augmented | baseline, trained params (learned part removed) | baseline, true params',
        '[evaluate] NRMS_e = RMS(e_hat - e) / sigma_e per record and axis, from k0 = %d, free-run closed loop' % k0,
        '=' * 100]), flush=True)

    # 2. Per split: the free runs, the scores of both frames, the figures; one simulation per record and model.
    results, saved, score_rows, tables = {}, {}, [], {frame: [] for frame in FRAMES}
    offset = 0
    for split, split_files in zip(SPLITS, files):
        sdl, e_list = getattr(data, split + '_list'), getattr(data, 'e_' + split)
        e_nf = (e_list if twins is None else
                [_load_record(twins[f], cfg, folder=cfg.data_root)[1] for f in split_files])
        preds, res = _split_scores(models, sdl, e_list, e_nf, bank, rows[offset:offset + len(sdl)], k0)
        offset += len(sdl)
        records = [os.path.basename(f).split('_')[0] for f in split_files]
        results[split] = dict(res, records=records)
        refs = [np.asarray(sd.y) - e for sd, e in zip(sdl, e_list)]
        t = [np.arange(len(r)) * cfg.ts_new for r in refs]
        os.makedirs(os.path.join(out_dir, split, 'records'), exist_ok=True)
        for i, rec in enumerate(records):
            pre = '%s/%s/' % (split, rec)
            saved.update({pre + 't': t[i], pre + 'r': refs[i], pre + 'e': e_list[i], pre + 'e_noise_free': e_nf[i]})
            saved.update({pre + 'e_hat_' + key: preds[key][i] - refs[i][k0:] for key in models})
        for frame, (names, to_frame) in FRAMES.items():
            s = res[frame]
            tables[frame] += _table(split, names, records, s)
            score_rows += [(split, rec, frame, key, ax, s['rms'][key][i, j], s['nrms'][key][i, j], s['sigma'][i, j],
                            s['floor'][i, j])
                           for i, rec in enumerate(records) for key in models for j, ax in enumerate(names)]
            _bar_figure(os.path.join(out_dir, split, 'nrms_%s.png' % frame),
                        'run %s, %s: NRMS$_e$ per record and axis, %s frame' % (run_id, split, frame),
                        names, records, s)
            for i, rec in enumerate(records):
                _record_figure(os.path.join(out_dir, split, 'records', '%s_%s.png' % (rec, frame)),
                               '%s (%s), run %s, %s frame: prediction error over the record'
                               % (rec, split, run_id, frame), names, t[i][k0:],
                               {key: s['residual'][key][i] for key in ('augmented', 'trained')},
                               to_frame(refs[i][k0:]), {key: s['nrms'][key][i] for key in ('augmented', 'trained')},
                               s['floor'][i])
            del s['residual']
    print('\n'.join(tables['stage'] + ['', 'GENERALISED FRAME (X, Theta, Y)'] + tables['generalised']), flush=True)

    # 3. Physical parameters: true, start and learned identifiable combinations.
    with torch.no_grad():
        names, start, learned, _ = phy.estimation_parameters()
        true = _true_combinations(phy)
    params = [(n, float(a), float(b), float(c)) for n, a, b, c in zip(names, true, start, learned)]
    print('\n'.join(['', 'PHYSICAL PARAMETERS  (identifiable combinations; true = data-generating value)',
                     '%-9s | %12s | %12s | %12s | %9s' % ('', 'true', 'start', 'learned', 'rel. err'), '-' * 66]
                    + ['%-9s | %12.5g | %12.5g | %12.5g | %+8.1f%%' % (n, a, b, c, 100 * (c - a) / a)
                       for n, a, b, c in params]), flush=True)

    # 4. Data files, floats with 17 significant digits so they read back exactly, and the loss curves.
    with open(os.path.join(out_dir, 'scores.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(('split', 'record', 'frame', 'model', 'axis', 'rms', 'nrms_e', 'sigma_e', 'floor_nrms_e'))
        w.writerows(r[:5] + tuple('%.17g' % v for v in r[5:]) for r in score_rows)
    np.savez(os.path.join(out_dir, 'predictions.npz'), k0=k0, P_inv_T=P_INV_T64, **saved)
    with open(os.path.join(out_dir, 'params.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(('name', 'true', 'start', 'learned'))
        w.writerows((n,) + tuple('%.17g' % v for v in p) for n, *p in params)
    with open(os.path.join(out_dir, 'evaluation.json'), 'w') as fh:
        json.dump(dict(checkpoint=os.path.basename(ckpt) + '.pt', run_id=run_id, git=git_provenance()), fh, indent=2)
    loss_file = os.path.join(run_dir, 'loss.npz')
    if os.path.exists(loss_file):
        with np.load(loss_file) as loss:
            _loss_figure(os.path.join(out_dir, 'loss.png'), 'run %s: loss curves' % run_id, dict(loss))
    print('\n'.join([
        '', 'saved %s' % out_dir,
        '  scores.csv          %d rows (split, record, frame, model, axis: rms, NRMS_e, sigma_e, floor)'
        % len(score_rows),
        '  predictions.npz     per record: t, r, e, e of the noise-free twin, e_hat per model; k0, P^-T',
        '  params.csv          true, start and learned physical combinations',
        '  evaluation.json     checkpoint, run id, git commit',
        '  loss.png            ' + ('training loss and validation NRMS_e, from loss.npz' if os.path.exists(loss_file)
                                    else 'not drawn: the run folder has no loss.npz'),
        '  train/ val/ test/   nrms_stage.png, nrms_generalised.png, records/<record>_{stage,generalised}.png '
        '(%d + %d + %d records)' % tuple(len(fs) for fs in files),
        '[evaluate] done in %.0f s' % (time.time() - t_start)]), flush=True)
    return results
