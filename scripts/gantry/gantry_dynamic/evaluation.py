"""Evaluation: the closed-loop report, plots, and the npz save.

D-227: every number is a closed-loop free run through the known controller, scored the way
training and checkpoint selection score (`closed_loop_report.py`): one table per split (validation,
test), five rows (model, its physics part, the baseline at the start and at the true parameters,
the oracle), per record, in metres. The open-loop replays this module used to print (section A,
the true-x0 run, the open-loop oracle, the baselines computed in baselines.py) are gone from it,
and D-228 removed the added-state-versus-truth R2. The npz keys changed with D-227; the current
set is listed in `_build_save_dict`.

It does not RELOAD the best checkpoint before simulating: `fit()` already returns the
best-validating weights, and reloading them here silently discarded an accepted L-BFGS polish
(D-172).
"""
__project_origin__ = "added"

import os
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from .config import RunConfig, config_json_dict, record_trim
from .model import get_encoder_dims
from .closed_loop_report import closed_loop_table, ROWS


def capture_loss_history(fit_sys, cfg: RunConfig, save_dir, rid):
    """Save the model and return this run's loss history, reading it off `fit_sys`.

    It used to reconstruct the history from `_last.pth` and then reload `_best.pth` to undo that.
    Both loads are gone: `fit()` now keeps its own history across the best-checkpoint restore
    (D-172), so there is nothing to reconstruct, and the `_best` reload was silently replacing
    the weights, which after an accepted L-BFGS polish meant every diagnostic below reported the
    unpolished model.

    `fit_sys.eval()` stays: that is a property this function is expected to establish.
    """
    if cfg.save_flag:
        model_path = os.path.join(save_dir, f'gantry_{rid}')
        fit_sys.save_system(model_path)
        print(f'Saved model: {model_path}')

    fit_sys.eval()
    return (np.asarray(fit_sys.epoch_id).copy(),
            np.asarray(fit_sys.Loss_val).copy(),
            np.asarray(fit_sys.Loss_train).copy())


def report_joint_estimation(fit_sys):
    """Report and return the coordinates actually optimized by the physical block."""
    _pblocks = [m for m in fit_sys.hfn.connected_blocks
                if hasattr(m, 'estimation_parameters')]
    param_names = None
    params_init_np = None
    params_learned_np = None
    if _pblocks:
        _pb = _pblocks[0]
        param_names, _init, _learned, _coordinate_label = _pb.estimation_parameters()
        params_init_np = _init.cpu().numpy().copy()
        params_learned_np = _learned.cpu().numpy().copy()
        print('\n=== Joint estimation: identifiable combinations (best checkpoint) ===')
        print(_pb.param_table())
        if hasattr(_pb, 'admissibility'):
            with torch.no_grad():
                _adm = _pb.admissibility()
            print('  admissibility: min eig M(Y) = %.6g at Y = %+.3f m; '
                  'min abs d(Y) = %.6g at Y = %+.3f m; %s'
                  % (_adm['min_eig_M'], _adm['min_eig_M_at_Y'],
                     _adm['min_abs_d'], _adm['min_abs_d_at_Y'],
                     'PASS' if _adm['admissible'] else 'FAIL'))
        print(f'\n=== Joint estimation: {_coordinate_label} (init vs learned) ===')
        print(f"  {'param':8s} {'init':>12s} {'learned':>12s} {'delta':>9s}")
        for _i, _n in enumerate(param_names):
            _d = 100.0 * (params_learned_np[_i] - params_init_np[_i]) / params_init_np[_i]
            print(f'  {_n:8s} {params_init_np[_i]:12.4f} {params_learned_np[_i]:12.4f} {_d:+8.2f}%')
    return param_names, params_init_np, params_learned_np


def _record_mean(rms):
    """Mean over records of each row's per-record RMS (all channels): (5, n, 3) -> (5,). NaN rows
    (not available) stay NaN."""
    per_record = np.sqrt(np.mean(np.asarray(rms, float) ** 2, axis=2))            # (5, n)
    out = np.full(per_record.shape[0], np.nan)
    for r in range(per_record.shape[0]):
        v = per_record[r][np.isfinite(per_record[r])]
        if v.size:
            out[r] = v.mean()
    return out


def evaluate_and_save(fit_sys, hp, rid, cfg: RunConfig, data, norm, save_dir,
                      diag_conv=None, **legacy_baselines):
    """The closed-loop report (D-227), plots and the npz save.

    `legacy_baselines`: older experiment scripts still pass the open-loop baseline results of
    baselines.py; they are accepted and not reported (D-227)."""
    if legacy_baselines:
        print('Note: open-loop baseline arguments %s are no longer reported (D-227)'
              % sorted(legacy_baselines))
    na, nb, _, _ = get_encoder_dims(hp, cfg)
    epoch_id_full, loss_val_full, loss_train_full = capture_loss_history(fit_sys, cfg, save_dir, rid)
    param_names, params_init_np, params_learned_np = report_joint_estimation(fit_sys)

    # The closed-loop runs are the expensive part; on the GPU they run there (the diagnostics
    # around them are CPU by construction, D-169).
    if cfg.device == 'cuda':
        fit_sys.cuda()
    try:
        tabs = {s: closed_loop_table(fit_sys, cfg, data, norm, hp['up_sample'], s)
                for s in ('val', 'test')}
    finally:
        if cfg.device == 'cuda':
            fit_sys.cpu()
    means = {s: _record_mean(tabs[s]['rms']) for s in tabs}
    print('\n' + '=' * 72)
    print(f'VERDICT (run {rid}): closed loop, mean over records of the per-record RMS [m]')
    for s, lbl in (('val', 'validation'), ('test', 'test')):
        print('  %-10s ' % lbl + ' | '.join('%s %s' % (n, 'n/a' if not np.isfinite(v) else '%.3e' % v)
                                          for n, v in zip(ROWS, means[s])))
    print('=' * 72)

    # plots: the first validation record's closed-loop trajectory (row 1)
    vd = data.val_list[0]
    y_ref = np.asarray(vd.y)
    k0 = tabs['val']['k0']
    y_hat = np.full(y_ref.shape, np.nan)
    if k0 is not None and tabs['val']['pred_model'][0] is not None:
        y_hat[k0:] = tabs['val']['pred_model'][0]
    t_val = (np.arange(len(y_ref)) + record_trim(cfg)) * vd.dt      # record time (D-232 trim)
    _make_plots(save_dir=save_dir, rid=rid, hp=hp, epoch_id_full=epoch_id_full,
                loss_val_full=loss_val_full, loss_train_full=loss_train_full, t_val=t_val,
                y_ref=y_ref, y_hat=y_hat, k0=(k0 or 0), baseline_rms=means['val'][2],
                sigma_n=data.sigma_n, diag_conv=diag_conv)

    if cfg.save_flag:
        save_dict = _build_save_dict(cfg=cfg, data=data, norm=norm, hp=hp, na=na, nb=nb,
                                     y_ref=y_ref, y_hat=y_hat, t_val=t_val, k0=(k0 or 0),
                                     epoch_id_full=epoch_id_full, loss_val_full=loss_val_full,
                                     loss_train_full=loss_train_full, tabs=tabs,
                                     param_names=param_names, params_init_np=params_init_np,
                                     params_learned_np=params_learned_np)
        np.savez(os.path.join(save_dir, f'gantry_results_{rid}.npz'), **save_dict)
        print(f'Saved results: gantry_results_{rid}.npz')


def _make_plots(save_dir, rid, hp, epoch_id_full, loss_val_full, loss_train_full, t_val, y_ref,
                y_hat, k0, baseline_rms, sigma_n, diag_conv):
    """Loss curves, the closed-loop validation trajectory, its residual and the Y residual spectrum."""
    NX_ANN = hp['NX_ANN']
    plot_dir = os.path.join(save_dir, 'plots')
    val_dir = os.path.join(plot_dir, 'val')
    os.makedirs(val_dir, exist_ok=True)
    k0_t = float(t_val[min(k0, len(t_val) - 1)])        # read off the axis: t_val need not start at 0 (D-232)
    loss_val_nf = diag_conv.get('loss_val_nf') if diag_conv is not None else None
    loss_train_nf = diag_conv.get('loss_train_nf') if diag_conv is not None else None

    # Plot 1: loss convergence in metres (D-095/D-102): the validation selector (closed-loop
    # sim-RMS), the validation and training nf-window RMS, the start-parameter baseline (row 3 of
    # the closed-loop table) and the output-noise floor where one is set.
    fig1, ax1 = plt.subplots(figsize=(7, 3.5))
    ax1.semilogy(epoch_id_full, loss_val_full, color='C0', label='Val sim-RMS (selector)')
    if loss_val_nf is not None and len(loss_val_nf) > 0:
        _n = min(len(epoch_id_full), len(loss_val_nf))   # resume-safe: align to the tail
        ax1.semilogy(epoch_id_full[-_n:], np.asarray(loss_val_nf)[-_n:], color='C0',
                     linestyle=':', label=f'Val nf-RMS ({hp["nf"]}-step)')
    if loss_train_nf is not None and len(loss_train_nf) > 0:
        _m = min(len(epoch_id_full), len(loss_train_nf))
        ax1.semilogy(epoch_id_full[-_m:], np.asarray(loss_train_nf)[-_m:], color='C1',
                     linestyle='--', alpha=0.8, label=f'Train nf-RMS ({hp["nf"]}-step)')
    if np.isfinite(baseline_rms):
        ax1.axhline(baseline_rms, color='C2', linestyle=':', lw=1.2,
                    label=f'Baseline, start parameters ({baseline_rms:.2e} m)')
    if sigma_n is not None:
        ax1.axhline(sigma_n, color='r', linestyle='--', lw=1.2, label=f'Noise floor sigma_n={sigma_n:.2e} m')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('RMS [m]')
    ax1.set_title(f'Loss convergence, closed loop (NX_ANN={NX_ANN})')
    ax1.legend(fontsize=7); ax1.grid(True, which='both')
    fig1.tight_layout()
    fig1.savefig(os.path.join(plot_dir, f'gantry_val_loss_{rid}.png'), dpi=150)

    # Plot 1b: the normalized per-batch training loss the optimizer minimizes, on its own axis.
    figT, axT = plt.subplots(figsize=(7, 3.5))
    axT.semilogy(epoch_id_full, loss_train_full, color='C1', linestyle='--', label='Train loss (sqrt-MSE)')
    axT.set_xlabel('Epoch'); axT.set_ylabel('train loss (normalized)')
    axT.set_title(f'Training loss (NX_ANN={NX_ANN})')
    axT.legend(fontsize=7); axT.grid(True, which='both')
    figT.tight_layout()
    figT.savefig(os.path.join(plot_dir, f'gantry_train_loss_{rid}.png'), dpi=150)

    # Plot 2: the first validation record, closed loop, and Plot 5: its residual y_model - y_data
    ch_labels = ['X1 [m]', 'X2 [m]', 'Y [m]']
    err = y_hat - y_ref
    for fname, series, title in (
            (f'gantry_simulation_{rid}.png', None, 'Validation record 1, closed-loop free run'),
            (f'V1_error_trace_{rid}.png', err, 'Validation record 1, residual y_model - y_data (closed loop)')):
        fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
        for ch, (ax, lab) in enumerate(zip(axes, ch_labels)):
            if series is None:
                ax.plot(t_val, y_ref[:, ch], 'k', lw=0.8, label='Reference data')
                ax.plot(t_val, y_hat[:, ch], 'C0', lw=0.9, label='Model, closed loop (encoder x0)')
                ax.set_ylabel(lab)
            else:
                ax.plot(t_val, series[:, ch], 'C0', lw=0.8, label='model - data')
                ax.axhline(0, color='k', lw=0.5)
                ax.set_ylabel(f'{lab} error')
            ax.axvspan(t_val[0], k0_t, alpha=0.10, color='steelblue',
                       label=f'Encoder window ({k0} samples)' if ch == 0 else '_nolegend_')
            ax.grid(True)
            if ch == 0:
                ax.legend(fontsize=7, loc='upper right')
        axes[-1].set_xlabel('Time [s]')
        fig.suptitle(f'{title} (NX_ANN={NX_ANN})')
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir if series is None else val_dir, fname), dpi=150)

    # Plot 6: spectrum of the Y residual.
    r_y = err[k0:, 2]
    r_y = r_y[np.isfinite(r_y)]
    if r_y.size > 8:
        dt = float(t_val[1] - t_val[0])
        freq = np.fft.rfftfreq(r_y.size, dt)
        mag = np.abs(np.fft.rfft(r_y * np.hanning(r_y.size))) / r_y.size
        fig6, ax6 = plt.subplots(figsize=(8, 3.5))
        ax6.semilogy(freq, mag + 1e-30, 'C0', lw=0.8, label='Y residual spectrum')
        ax6.set_xlim(0, min(freq[-1], 400))
        ax6.set_xlabel('Frequency [Hz]'); ax6.set_ylabel('|Y error| [m]')
        ax6.set_title('Y residual spectrum, closed loop')
        ax6.legend(fontsize=7); ax6.grid(True, which='both')
        fig6.tight_layout()
        fig6.savefig(os.path.join(val_dir, f'V1_error_spectrum_{rid}.png'), dpi=150)
    plt.close('all')


def _build_save_dict(cfg, data, norm, hp, na, nb, y_ref, y_hat, t_val, k0, epoch_id_full,
                     loss_val_full, loss_train_full, tabs, param_names, params_init_np,
                     params_learned_np):
    """The npz contents (D-227 set): the closed-loop table of both splits (`cl_*`), the first
    validation and test record's closed-loop trajectory, loss curves, normalisation, dimensions,
    configuration and the estimated coordinates."""
    from model_augmentation.systems.gantry_ss import P
    vd = data.val_list[0]
    save_dict = dict(
        y_ref=y_ref, y_hat_cl=y_hat, t_val=t_val, u_val=vd.u,
        epoch_id=epoch_id_full, loss_val=loss_val_full, loss_train=loss_train_full,
        cl_rows=np.asarray(ROWS),
        cl_val_files=np.asarray([str(f) for f in tabs['val']['files']]), cl_val_rms=tabs['val']['rms'],
        cl_test_files=np.asarray([str(f) for f in tabs['test']['files']]), cl_test_rms=tabs['test']['rms'],
        std_x=norm.std_x, x_mean=norm.x_mean, std_u=norm.std_u, u_mean=norm.u_mean,
        ystd=norm.ystd, y0=norm.y0, Cd_norm=norm.Cd_norm, Dd_np=norm.Dd_np, P_matrix=P.numpy(),
        noise_snr=np.array(cfg.snr if cfg.snr is not None else -1),
        noise_sigma=np.array(data.sigma_n if data.sigma_n is not None else 0.0),
        cheat_n=np.array(k0), dt=np.array(vd.dt),
        na=np.array(na), nb=np.array(nb), nf=np.array(hp['nf']),
        NX_PHYS=np.array(cfg.nx_phys), NX_ANN=np.array(hp['NX_ANN']),
        nxd=np.array(cfg.nx_phys + hp['NX_ANN']),
        hp=json.dumps(hp), config=json.dumps(config_json_dict(cfg)),
    )
    tk0 = tabs['test']['k0']
    if tk0 is not None and tabs['test']['pred_model'] and tabs['test']['pred_model'][0] is not None:
        y_test = np.asarray(data.test_list[0].y)
        y_hat_test = np.full(y_test.shape, np.nan)
        y_hat_test[tk0:] = tabs['test']['pred_model'][0]
        save_dict['y_hat_test_cl'] = y_hat_test
        save_dict['y_test_ref'] = y_test
    if params_learned_np is not None:
        save_dict['param_names'] = np.asarray(param_names)
        save_dict['params_init'] = params_init_np
        save_dict['params_learned'] = params_learned_np
    return save_dict


def per_record_nrms(fit_sys, files, sdlist, norm, K0: int, tag: str) -> list:
    """@added (2026-09-15). Per-record augmented NRMS over held-out records (D-098).

    OPEN loop (`apply_experiment`); not used by the main pipeline since D-227, kept for the
    experiment scripts that import it (orthogonal-addition/runners/oa_entry.py)."""
    print(f'{tag} NRMS per record (augmented, avg from K0):')
    rows = []
    for _f, _td in zip(files, sdlist):
        _yh = fit_sys.apply_experiment(_td).y
        rows.append(np.sqrt(((_yh[K0:] - _td.y[K0:]) ** 2).mean(axis=0)) / norm.ystd)
        print(f'  {_f}: {rows[-1]}')
    print(f'  mean: {np.mean(rows, axis=0)}')
    return rows
