"""Evaluation: best-checkpoint simulation, metrics, four plots, and the npz save.

`evaluate_and_save` keeps its pre-refactor orchestration order exactly; the
internals are factored into capture_loss_history / report_joint_estimation /
_make_plots / _build_save_dict. The npz key set, names, conditional inclusion,
and dtypes are a frozen contract.
"""
__project_origin__ = "added"

import os
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .config import RunConfig, config_json_dict
from .model import get_encoder_dims
from .baselines import stepwise_rollout
from .diagnostics import r2_per_channel, best_affine_r2
from . import oracle as _oracle


def capture_loss_history(fit_sys, cfg: RunConfig, save_dir, rid):
    """Save the model, then capture the FULL loss history before best-checkpoint restore."""
    if cfg.save_flag:
        model_path = os.path.join(save_dir, f'gantry_{rid}')
        fit_sys.save_system(model_path)
        print(f'Saved model: {model_path}')

    # Capture full loss history before best-checkpoint restore truncates it.
    fit_sys.checkpoint_load_system(name='_last')
    epoch_id_full   = fit_sys.epoch_id.copy()
    loss_val_full   = fit_sys.Loss_val.copy()
    loss_train_full = fit_sys.Loss_train.copy()
    fit_sys.checkpoint_load_system(name='_best')
    fit_sys.eval()
    return epoch_id_full, loss_val_full, loss_train_full


def report_joint_estimation(fit_sys):
    """Joint estimation report (D-076/D-077); returns (params_init_np, params_learned_np)."""
    _pblocks = [m for m in fit_sys.hfn.connected_blocks if hasattr(m, 'physical_params')]
    params_init_np = None
    params_learned_np = None
    if _pblocks:
        _pb = _pblocks[0]
        params_init_np = _pb.params_init.detach().cpu().numpy().copy()
        params_learned_np = np.array([_pb.physical_params()[n] for n in _pb.PARAM_NAMES])
        # Trusted view: the 10 identifiable combinations (raw are trained, combos
        # are what the data can determine -- D-077).
        print('\n=== Joint estimation: identifiable combinations (best checkpoint) ===')
        print(_pb.param_table())
        # Diagnostic view: all 14 raw params (splits held near init by param_loss).
        print('\n=== Joint estimation: raw parameters (init vs learned) ===')
        print(f"  {'param':8s} {'init':>12s} {'learned':>12s} {'delta':>9s}")
        for _i, _n in enumerate(_pb.PARAM_NAMES):
            _d = 100.0 * (params_learned_np[_i] - params_init_np[_i]) / params_init_np[_i]
            print(f'  {_n:8s} {params_init_np[_i]:12.4f} {params_learned_np[_i]:12.4f} {_d:+8.2f}%')
    return params_init_np, params_learned_np


def _print_same_init_comparison(title, nrms_aug, nrms_base, ystd,
                                base_label='baseline FP (enc-init)', refs=None):
    """Same-init augmented-vs-baseline table (D-094/D-098).

    Improvement % is aug-vs-base at the SAME init. `refs` is a list of
    (label, nrms) reference columns (true-x0 baseline, oracle FP+MSD) shown as
    RMS[m] with no % (different init -- reference only). NRMS = RMS/ystd.
    """
    refs = [(lbl, v) for (lbl, v) in (refs or []) if v is not None]
    print(f'\n{title}')
    print(f"  base = {base_label}"
          + (''.join(f'   ref[{i}] = {lbl}' for i, (lbl, _) in enumerate(refs)) if refs else ''))
    print(f"  {'':4s} {'aug RMS[m]':>11s} {'aug NRMS':>9s}  "
          f"{'base RMS[m]':>11s} {'base NRMS':>9s} {'aug/base':>9s}"
          + ''.join(f"   {'ref['+str(i)+'] RMS':>13s}" for i in range(len(refs))))
    for ch, lbl in enumerate(['X1', 'X2', 'Y ']):
        if nrms_base is not None:
            improv = 100.0 * (nrms_base[ch] - nrms_aug[ch]) / (nrms_base[ch] + 1e-12)
            base_s = f"{nrms_base[ch]*ystd[ch]:>11.3e} {nrms_base[ch]:>9.4f} {improv:>+8.1f}%"
        else:
            base_s = f"{'-':>11s} {'-':>9s} {'-':>9s}"
        ref_s = ''.join(f"   {v[ch]*ystd[ch]:>13.3e}" for (_, v) in refs)
        print(f"  {lbl:4s} {nrms_aug[ch]*ystd[ch]:>11.3e} {nrms_aug[ch]:>9.4f}  {base_s}{ref_s}")


def _oracle_nrms(u_stage, x_logical, x_aug, cfg, up_sample, start_ix, y_ref, ystd):
    """Run the FP+MSD oracle open-loop and return (nrms, y_hat_oracle). D-098.

    Reference only (true-x0 init; the encoder cannot observe the absorber). Runs at
    the pipeline rate and up_sample so it is on the same footing as baseline/aug.
    """
    try:
        y_hat, _ = _oracle.oracle_open_loop(u_stage, x_logical, x_aug, cfg, up_sample, start_ix)
        nrms = np.sqrt(((y_hat[start_ix:] - y_ref[start_ix:]) ** 2).mean(axis=0)) / ystd
        return nrms, y_hat
    except Exception as e:
        print(f'Warning: oracle sim failed: {e}')
        return None, None


def evaluate_and_save(fit_sys, hp, rid, cfg: RunConfig, data, norm, save_dir,
                      diag_conv=None, baseline_nrms=None,
                      baseline_test_nrms=None, baseline_encinit_nrms=None,
                      baseline_test_encinit_nrms=None):
    """Load best checkpoint, simulate, compute NRMS, plot, save."""
    NX_PHYS = cfg.nx_phys
    ny = cfg.ny
    NX_ANN = hp['NX_ANN']
    nxd = NX_PHYS + NX_ANN
    na, nb, na_right, nb_right = get_encoder_dims(hp, cfg)
    DTYPE_NP, DTYPE_PT = cfg.dtype_np, cfg.dtype_pt
    std_x, x_mean = norm.std_x, norm.x_mean
    ystd, y0 = norm.ystd, norm.y0
    val_data, test_data = data.val_data, data.test_data
    val_x_logical, val_x_aug = data.val_x_logical, data.val_x_aug

    epoch_id_full, loss_val_full, loss_train_full = capture_loss_history(fit_sys, cfg, save_dir, rid)

    params_init_np, params_learned_np = report_joint_estimation(fit_sys)

    # ── Encoder-initialised simulation ──────────────────────────────────────
    fit_sys.hfn.reset_saved_signals()
    sim_result = fit_sys.apply_experiment(val_data)
    cheat_n   = sim_result.cheat_n
    y_hat_enc = sim_result.y       # (T, 3) physical [m]
    y_ref     = val_data.y

    x_enc_norm = np.array(fit_sys.hfn.saved_output_signals)
    x_enc_phys = np.full((len(y_ref), NX_PHYS), np.nan, dtype=DTYPE_NP)
    x_enc_phys[cheat_n:] = (x_enc_norm[:NX_PHYS, :] * std_x + x_mean).T
    x_enc_ann  = np.full((len(y_ref), NX_ANN), np.nan, dtype=DTYPE_NP)
    x_enc_ann[cheat_n:]  = x_enc_norm[NX_PHYS:nxd, :].T

    nrms_enc = np.sqrt(((y_hat_enc[cheat_n:] - y_ref[cheat_n:]) ** 2).mean(axis=0)) / ystd
    rms_enc  = nrms_enc * ystd   # [m]
    ann_rms_enc = np.sqrt((x_enc_ann[cheat_n:] ** 2).mean(axis=0))  # early: needed for the verdict

    # ── Oracle FP+MSD reference (D-098): best-case, true-x0 init, pipeline rate ──
    nrms_oracle, y_hat_oracle = _oracle_nrms(
        val_data.u, val_x_logical, val_x_aug, cfg, hp['up_sample'], cheat_n, y_ref, ystd)

    # ── Verdict (D-094) ───────────────────────────────────────────────────────
    # Same-init comparison: augmented (encoder-init) vs the encoder-init baseline,
    # NOT the true-x0 baseline (different init -> its % is an init artifact).
    _base_si  = baseline_encinit_nrms if baseline_encinit_nrms is not None else baseline_nrms
    _base_lbl = 'baseline FP (enc-init)' if baseline_encinit_nrms is not None else 'baseline FP (true-x0)'
    _ref_si   = baseline_nrms if (baseline_encinit_nrms is not None and baseline_nrms is not None) else None
    ann_active = bool(np.nanmax(ann_rms_enc) > 1e-9) if NX_ANN > 0 else False
    print('\n' + '=' * 72)
    print(f"VERDICT (run {rid}): ANN {'ACTIVE' if ann_active else 'inactive (aug states ~0)'}")
    if _base_si is not None:
        _si = [100.0 * (_base_si[c] - nrms_enc[c]) / (_base_si[c] + 1e-12) for c in range(ny)]
        print('  augmentation vs baseline (same init): '
              + '  '.join(f'{l} {_si[c]:+.1f}%' for c, l in enumerate(['X1', 'X2', 'Y'])))
    print('=' * 72)

    # A. MODEL QUALITY -- validation (encoder-init; aug vs baseline same init)
    if _base_si is not None:
        _print_same_init_comparison(
            '== A. MODEL QUALITY: validation (V1) ==',
            nrms_enc, _base_si, ystd, base_label=_base_lbl,
            refs=[('baseline FP (true-x0)', _ref_si), ('oracle FP+MSD (true-x0)', nrms_oracle)])
    else:
        print('\n== A. MODEL QUALITY: validation (V1), encoder-init ==')
        for ch, lbl in enumerate(['X1', 'X2', 'Y ']):
            print(f'  {lbl}: RMS {rms_enc[ch]:.3e} m  NRMS {nrms_enc[ch]:.4f}')

    print('\n== C. AUGMENTATION / ABSORBER ==')
    print('=== ANN latent state RMS ===')
    for ch in range(NX_ANN):
        print(f'  x[{NX_PHYS+ch}]: enc={ann_rms_enc[ch]:.4e}')

    # ── Rollout aug-state R2 vs GT absorber (closed-loop simulation) ─────────
    # Complements the encoder-based R2 (D-053/aug_state_r2): this tests whether
    # the MODEL's simulated x[6:7] trajectories track delta_a, i.e. whether the
    # augmented states became the absorber in rollout. Activation-agnostic.
    r2_roll_raw, r2_roll_lin = None, None
    try:
        x_ann_roll = x_enc_ann[cheat_n:]                    # (N-cheat_n, NX_ANN) simulated
        gt_roll    = val_x_aug[cheat_n:]                    # (N-cheat_n, NX_ANN) physical
        gt_norm = (gt_roll - gt_roll.mean(axis=0)) / (gt_roll.std(axis=0) + 1e-8)

        r2_roll_raw = r2_per_channel(gt_norm, x_ann_roll)
        # best affine map from all rollout aug channels
        W_roll, r2_roll_lin = best_affine_r2(x_ann_roll, gt_norm, DTYPE_NP)

        aug_names = ['delta_a ', 'vdelta_a']
        print('\n=== Rollout aug-state R2 vs GT (closed-loop simulation) ===')
        for ch in range(NX_ANN):
            lbl = aug_names[ch] if ch < len(aug_names) else f'x_ann[{ch}]'
            print(f'  {lbl}  R2_raw={r2_roll_raw[ch]:+.4f}  R2_linmap={r2_roll_lin[ch]:+.4f}')
        print('  R2_linmap ~ 1 -> simulated aug states carry the absorber dynamics;')
        print('  R2_linmap ~ 0 -> aug states unused in rollout (encoder R2 tests encoder only)')
    except Exception as e:
        print(f'Warning: rollout aug-state R2 failed: {e}')

    # ── Test-set simulation (E1, unseen excitation) — generalization (D-071) ─
    nrms_test  = None
    y_hat_test = None
    nrms_oracle_test = None   # D-098: set inside the try when GT states are present
    try:
        fit_sys.hfn.reset_saved_signals()
        test_result = fit_sys.apply_experiment(test_data)
        y_hat_test  = test_result.y
        tc = test_result.cheat_n
        nrms_test = np.sqrt(((y_hat_test[tc:] - test_data.y[tc:]) ** 2).mean(axis=0)) / ystd
        # Oracle reference on the test record (true-x0 init), when GT states are available.
        nrms_oracle_test = None
        if data.test_x_logical is not None and data.test_x_aug is not None:
            nrms_oracle_test, _ = _oracle_nrms(
                test_data.u, data.test_x_logical, data.test_x_aug, cfg, hp['up_sample'],
                tc, test_data.y, ystd)
        # Same-init (D-094): aug vs encoder-init baseline; true-x0 baseline as reference.
        _tbase = baseline_test_encinit_nrms if baseline_test_encinit_nrms is not None else baseline_test_nrms
        _tlbl  = 'baseline FP (enc-init)' if baseline_test_encinit_nrms is not None else 'baseline FP (true-x0)'
        _tref  = baseline_test_nrms if (baseline_test_encinit_nrms is not None and baseline_test_nrms is not None) else None
        if _tbase is not None:
            _print_same_init_comparison(
                '== A. MODEL QUALITY: test (E1, unseen excitation) ==',
                nrms_test, _tbase, ystd, base_label=_tlbl,
                refs=[('baseline FP (true-x0)', _tref), ('oracle FP+MSD (true-x0)', nrms_oracle_test)])
        else:
            print('\n== A. MODEL QUALITY: test (E1), encoder-init ==')
            for ch, lbl in enumerate(['X1', 'X2', 'Y ']):
                print(f'  {lbl}: RMS {nrms_test[ch]*ystd[ch]:.3e} m  NRMS {nrms_test[ch]:.4f}')
    except Exception as e:
        print(f'Warning: test-set (E1) evaluation failed: {e}')

    # ── x_logical-initialised simulation (model from TRUE x0, D-072) ────────
    # Symmetric counterpart to the true-x0 baseline: isolates model quality from
    # encoder quality. Seeded from val_x_logical (val_data.x is never set by load_traj).
    if val_x_logical is not None:
        val_norm = fit_sys.norm.transform(val_data)
        u_val_norm = torch.tensor(np.ascontiguousarray(val_norm.u), dtype=DTYPE_PT)

        # D-087: seed from the interior sample cheat_n, not sample 0 — the stored
        # qdot at sample 0 is a one-sided gradient() artifact (V1 starts at rest yet
        # v0 != 0, worth tau*dv = -1e-4 m on Y); interior samples carry central
        # differences. Also starts at the same instant as the encoder-init sim.
        x_xlog = torch.zeros(1, nxd)
        x_xlog[0, :NX_PHYS] = torch.tensor(
            (val_x_logical[cheat_n] - x_mean.flatten()) / std_x.flatten(), dtype=DTYPE_PT)

        def _xlog_step(x, u_row):
            y_t, x_next = fit_sys.hfn(x, u_row.view(1, -1))
            return y_t.squeeze().numpy(), x_next

        y_xlog_list = stepwise_rollout(_xlog_step, x_xlog, u_val_norm[cheat_n:])
        y_hat_xlog = np.full((len(y_ref), ny), np.nan, dtype=DTYPE_NP)
        y_hat_xlog[cheat_n:] = np.array(y_xlog_list) * ystd + y0

        # Averaged from cheat_n like the encoder-init model metric (D-072 alignment)
        nrms_xlog = np.sqrt(((y_hat_xlog[cheat_n:] - y_ref[cheat_n:]) ** 2).mean(axis=0)) / ystd
        print('\n=== Model, true-x0 init: sim-NRMS + RMS ===')
        for ch, lbl in enumerate(['X1', 'X2', 'Y ']):
            print(f'  {lbl}: {nrms_xlog[ch]:.4f}  {nrms_xlog[ch] * ystd[ch]:.3e} m')
        HAS_ORACLE = True
    else:
        print('\n=== x_logical-initialised simulation skipped (no state data) ===')
        y_hat_xlog = None
        nrms_xlog  = None
        HAS_ORACLE = False

    # ── Plots ───────────────────────────────────────────────────────────────
    t_val   = np.arange(len(y_ref)) * val_data.dt
    cheat_t = cheat_n * val_data.dt

    _make_plots(
        save_dir=save_dir, rid=rid, cfg=cfg, hp=hp, nxd=nxd,
        epoch_id_full=epoch_id_full, loss_val_full=loss_val_full, loss_train_full=loss_train_full,
        t_val=t_val, y_ref=y_ref, y_hat_enc=y_hat_enc, nrms_enc=nrms_enc, rms_enc=rms_enc,
        HAS_ORACLE=HAS_ORACLE, y_hat_xlog=y_hat_xlog, nrms_xlog=nrms_xlog,
        cheat_n=cheat_n, cheat_t=cheat_t, x_enc_ann=x_enc_ann, ann_rms_enc=ann_rms_enc,
        val_x_aug=val_x_aug, diag_conv=diag_conv, baseline_nrms=baseline_nrms,
        norm=norm, sigma_n=data.sigma_n,
        loss_val_nf=(diag_conv.get('loss_val_nf') if diag_conv is not None else None),  # D-095/Step 4
        loss_train_nf=(diag_conv.get('loss_train_nf') if diag_conv is not None else None),  # D-102
        y_hat_oracle=y_hat_oracle,   # D-098: FP+MSD oracle reference line on the error trace
    )

    # ── Results npz ─────────────────────────────────────────────────────────
    if cfg.save_flag:
        save_dict = _build_save_dict(
            cfg=cfg, data=data, norm=norm, hp=hp, na=na, nb=nb,
            y_ref=y_ref, y_hat_enc=y_hat_enc, t_val=t_val,
            epoch_id_full=epoch_id_full, loss_val_full=loss_val_full, loss_train_full=loss_train_full,
            nrms_enc=nrms_enc, rms_enc=rms_enc, x_enc_phys=x_enc_phys, x_enc_ann=x_enc_ann,
            cheat_n=cheat_n, nxd=nxd, HAS_ORACLE=HAS_ORACLE, y_hat_xlog=y_hat_xlog, nrms_xlog=nrms_xlog,
            diag_conv=diag_conv, baseline_nrms=baseline_nrms, nrms_test=nrms_test, y_hat_test=y_hat_test,
            r2_roll_raw=r2_roll_raw, r2_roll_lin=r2_roll_lin,
            baseline_test_nrms=baseline_test_nrms, baseline_encinit_nrms=baseline_encinit_nrms,
            baseline_test_encinit_nrms=baseline_test_encinit_nrms,
            params_init_np=params_init_np, params_learned_np=params_learned_np,
        )
        # D-098: oracle FP+MSD reference (conditionally, like the other optional keys).
        if nrms_oracle is not None:
            save_dict['nrms_oracle']  = nrms_oracle
            save_dict['rms_oracle']   = nrms_oracle * ystd
            save_dict['y_hat_oracle'] = y_hat_oracle
        if nrms_oracle_test is not None:
            save_dict['nrms_oracle_test'] = nrms_oracle_test
        np.savez(os.path.join(save_dir, f'gantry_results_{rid}.npz'), **save_dict)
        print(f'Saved results: gantry_results_{rid}.npz')


def _make_plots(save_dir, rid, cfg, hp, nxd, epoch_id_full, loss_val_full, loss_train_full,
                t_val, y_ref, y_hat_enc, nrms_enc, rms_enc, HAS_ORACLE, y_hat_xlog, nrms_xlog,
                cheat_n, cheat_t, x_enc_ann, ann_rms_enc, val_x_aug, diag_conv, baseline_nrms,
                norm, sigma_n, loss_val_nf=None, loss_train_nf=None, y_hat_oracle=None):
    NX_PHYS = cfg.nx_phys
    NX_ANN = hp['NX_ANN']
    ystd = norm.ystd

    # Step 4: all figures go into a per-run plots/ subtree.
    plot_dir = os.path.join(save_dir, 'plots')
    val_dir  = os.path.join(plot_dir, 'val')
    os.makedirs(val_dir, exist_ok=True)

    # Plot 1: Loss convergence, all physical meters (D-095/D-102). Val full-traj sim-RMS
    # (solid, selector) + val nf-window RMS (dotted) + train nf-window RMS (dashed, same nf
    # horizon). Directly comparable: train-vs-val gap = generalization; val nf-vs-sim gap =
    # long-rollout drift. The normalized deepSI train loss lives in its own figure below.
    fig1, ax1 = plt.subplots(figsize=(7, 3.5))
    ax1.semilogy(epoch_id_full, loss_val_full,   color='C0', label='Val sim-RMS (selector)')
    if loss_val_nf is not None and len(loss_val_nf) > 0:
        _n = min(len(epoch_id_full), len(loss_val_nf))   # resume-safe: align to the tail
        ax1.semilogy(epoch_id_full[-_n:], np.asarray(loss_val_nf)[-_n:], color='C0',
                     linestyle=':', label=f'Val nf-RMS ({hp["nf"]}-step)')
    if loss_train_nf is not None and len(loss_train_nf) > 0:
        _m = min(len(epoch_id_full), len(loss_train_nf))
        ax1.semilogy(epoch_id_full[-_m:], np.asarray(loss_train_nf)[-_m:], color='C1',
                     linestyle='--', alpha=0.8, label=f'Train nf-RMS ({hp["nf"]}-step)')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('RMS [m]')
    ax1.set_title(f'Loss convergence - dynamic parallel (NX_ANN={NX_ANN})')
    ax1.legend(fontsize=7); ax1.grid(True, which='both')
    fig1.tight_layout()
    fig1.savefig(os.path.join(plot_dir, f'gantry_val_loss_{rid}.png'), dpi=150)

    # Plot 1b: deepSI training loss on its own axis. This is the NORMALIZED, per-batch
    # sqrt-MSE the optimizer minimizes -- NOT physical meters, so it is kept separate from
    # the meter figure above to avoid a units-mismatched y-axis.
    figT, axT = plt.subplots(figsize=(7, 3.5))
    axT.semilogy(epoch_id_full, loss_train_full, color='C1', linestyle='--', label='Train loss (sqrt-MSE)')
    axT.set_xlabel('Epoch'); axT.set_ylabel('train loss (normalized)')
    axT.set_title(f'Training loss - dynamic parallel (NX_ANN={NX_ANN})')
    axT.legend(fontsize=7); axT.grid(True, which='both')
    figT.tight_layout()
    figT.savefig(os.path.join(plot_dir, f'gantry_train_loss_{rid}.png'), dpi=150)

    # Plot 2: Validation simulation
    ch_labels = ['X1 [m]', 'X2 [m]', 'Y [m]']
    fig2, axes2 = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for ch, (ax, lab) in enumerate(zip(axes2, ch_labels)):
        ax.plot(t_val, y_ref[:, ch], 'k', lw=0.8, label='Reference')
        ax.plot(t_val, y_hat_enc[:, ch], 'C0', lw=0.9,
                label=f'Encoder-init (NRMS={nrms_enc[ch]:.3f}, RMS={rms_enc[ch]:.2e} m)')
        if HAS_ORACLE:
            ax.plot(t_val, y_hat_xlog[:, ch], 'C1', lw=0.9, linestyle='--',
                    label=f'x_logical-init (NRMS={nrms_xlog[ch]:.3f}, RMS={nrms_xlog[ch]*ystd[ch]:.2e} m)')
        enc_lbl = f'Encoder warmup ({cheat_n} samples)' if ch == 0 else '_nolegend_'
        ax.axvspan(t_val[0], cheat_t, alpha=0.10, color='steelblue', label=enc_lbl)
        ax.axvline(cheat_t, color='steelblue', linestyle='--', lw=0.8)
        ax.set_ylabel(lab); ax.legend(fontsize=7, loc='upper right'); ax.grid(True)
    axes2[-1].set_xlabel('Time [s]')
    fig2.suptitle(f'Validation simulation - dynamic parallel (NX_ANN={NX_ANN})')
    fig2.tight_layout()
    fig2.savefig(os.path.join(plot_dir, f'gantry_simulation_{rid}.png'), dpi=150)

    # Plot 3: ANN latent state trajectories vs ground-truth absorber states
    aug_gt_labels  = ['delta_a [m]', 'vdelta_a [m/s]']
    aug_gt_names   = ['delta_a', 'vdelta_a']
    if NX_ANN == 1:
        fig3, axes3 = plt.subplots(1, 1, figsize=(12, 3), sharex=True)
        axes3 = [axes3]
    else:
        fig3, axes3 = plt.subplots(NX_ANN, 1, figsize=(12, 4), sharex=True)
    for ch, ax in enumerate(axes3):
        ax.plot(t_val, x_enc_ann[:, ch], 'C0', lw=0.8,
                label=f'x_ann[{NX_PHYS+ch}] (RMS={ann_rms_enc[ch]:.2e})')
        ax.axvspan(t_val[0], cheat_t, alpha=0.10, color='steelblue')
        ax.axvline(cheat_t, color='steelblue', linestyle='--', lw=0.8)
        ax.set_ylabel(f'x[{NX_PHYS+ch}] (dim-less)', color='C0')
        ax.tick_params(axis='y', labelcolor='C0')
        if ch < len(aug_gt_labels):
            ax2 = ax.twinx()
            ax2.plot(t_val, val_x_aug[:, ch], 'C1', lw=0.8, alpha=0.7,
                     label=f'GT {aug_gt_names[ch]}')
            ax2.set_ylabel(aug_gt_labels[ch], color='C1')
            ax2.tick_params(axis='y', labelcolor='C1')
            lines1, labs1 = ax.get_legend_handles_labels()
            lines2, labs2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labs1 + labs2, fontsize=7, loc='upper right')
        else:
            ax.legend(fontsize=7)
        ax.grid(True)
    axes3[-1].set_xlabel('Time [s]')
    fig3.suptitle(f'ANN latent states x[{NX_PHYS}:{nxd}] vs GT absorber (dimensionless vs physical)')
    fig3.tight_layout()
    fig3.savefig(os.path.join(plot_dir, f'gantry_ann_states_{rid}.png'), dpi=150)

    # Plot 4: Training convergence (loss + R2_linmap) — only when diag_conv available
    if diag_conv is not None and len(diag_conv['epochs']) > 0:
        fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 4))

        # Left: loss curves with baseline FP NRMS as horizontal reference
        ax4a.semilogy(epoch_id_full, loss_val_full,   color='C0', label='Val loss')
        ax4a.semilogy(epoch_id_full, loss_train_full, color='C1', linestyle='--',
                      alpha=0.7, label='Train loss')
        if baseline_nrms is not None:
            # Aggregate sim-RMS [m]: same formula as the plotted validation loss,
            # so the reference line and the curve share units (NRMS lines were not comparable).
            rms_agg_baseline = float(np.sqrt(np.mean((baseline_nrms * ystd) ** 2)))
            ax4a.axhline(rms_agg_baseline, color='C2', linestyle=':', lw=1.2,
                         label=f'Baseline FP sim-RMS={rms_agg_baseline:.2e} m')
        if sigma_n is not None:
            ax4a.axhline(sigma_n, color='r', linestyle='--', lw=1.2,
                         label=f'Noise floor sigma_n={sigma_n:.2e} m')
        ax4a.set_xlabel('Epoch'); ax4a.set_ylabel('sim-RMS')
        ax4a.set_title('Loss + baseline reference')
        ax4a.legend(fontsize=7); ax4a.grid(True, which='both')

        # Right: R2_linmap for each augmented state channel over epochs
        aug_labels_short = ['delta_a', 'vdelta_a']
        conv_epochs = diag_conv['epochs']
        conv_r2lin  = diag_conv['r2_linmap']   # (n_chunks, NX_ANN)
        for ch in range(hp['NX_ANN']):
            lbl = aug_labels_short[ch] if ch < len(aug_labels_short) else f'x_ann[{ch}]'
            ax4b.plot(conv_epochs, conv_r2lin[:, ch], marker='o', ms=4, label=lbl)
        ax4b.axhline(0.0, color='k', lw=0.5, linestyle='--')
        ax4b.axhline(1.0, color='k', lw=0.5, linestyle=':')
        ax4b.set_xlabel('Epoch'); ax4b.set_ylabel('R2_linmap')
        ax4b.set_title('Aug state convergence (R2_linmap vs delta_a)')
        ax4b.legend(fontsize=8); ax4b.grid(True)
        ax4b.set_ylim([-0.1, 1.05])

        fig4.suptitle(f'Training convergence (NX_ANN={NX_ANN})')
        fig4.tight_layout()
        fig4.savefig(os.path.join(plot_dir, f'gantry_convergence_{rid}.png'), dpi=150)

    # Plot 5: error-vs-time (diagnostic). The overlay (Plot 2) hides sub-mm structure on a
    # 0.24 m axis; the residual y_model - y_data reveals it. Ramp = drift, oscillation =
    # absorber, flat = fine. (Baseline/oracle lines added in Step 6.)
    err_enc = y_hat_enc - y_ref                                    # augmented (encoder-init) residual
    fig5, axes5 = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for ch, (ax, lab) in enumerate(zip(axes5, ch_labels)):
        ax.plot(t_val, err_enc[:, ch], 'C0', lw=0.8, label='augmented (encoder-init)')
        if HAS_ORACLE:
            ax.plot(t_val, y_hat_xlog[:, ch] - y_ref[:, ch], 'C1', lw=0.8, linestyle='--',
                    label='augmented (true-x0 init)')
        if y_hat_oracle is not None:
            ax.plot(t_val, y_hat_oracle[:, ch] - y_ref[:, ch], 'C2', lw=0.8, linestyle=':',
                    label='oracle FP+MSD (true-x0)')
        ax.axhline(0, color='k', lw=0.5)
        ax.axvspan(t_val[0], cheat_t, alpha=0.10, color='steelblue')
        ax.axvline(cheat_t, color='steelblue', linestyle='--', lw=0.8)
        ax.set_ylabel(f'{lab} error'); ax.grid(True)
        if ch == 0:
            ax.legend(fontsize=7, loc='upper left')
    axes5[-1].set_xlabel('Time [s]')
    fig5.suptitle('Validation residual y_model - y_data (ramp=drift, oscillation=absorber)')
    fig5.tight_layout()
    fig5.savefig(os.path.join(val_dir, f'V1_error_trace_{rid}.png'), dpi=150)

    # Plot 6: error spectrum of the Y residual. The absorber lives at ~157 Hz (band 130-180);
    # if augmentation removes it, this peak flattens. (With the ANN at zero it is fully present.)
    r_y = err_enc[cheat_n:, 2]
    r_y = r_y[np.isfinite(r_y)]
    if r_y.size > 8:
        dt = float(t_val[1] - t_val[0])
        freq = np.fft.rfftfreq(r_y.size, dt)
        mag  = np.abs(np.fft.rfft(r_y * np.hanning(r_y.size))) / r_y.size
        fig6, ax6 = plt.subplots(figsize=(8, 3.5))
        ax6.semilogy(freq, mag + 1e-30, 'C0', lw=0.8, label='Y residual spectrum')
        ax6.axvspan(130, 180, alpha=0.12, color='orange', label='absorber band 130-180 Hz')
        ax6.axvline(157, color='r', linestyle='--', lw=0.8, label='resonance ~157 Hz')
        ax6.set_xlim(0, min(freq[-1], 400))
        ax6.set_xlabel('Frequency [Hz]'); ax6.set_ylabel('|Y error| [m]')
        ax6.set_title('Y residual spectrum - is the absorber left in the error?')
        ax6.legend(fontsize=7); ax6.grid(True, which='both')
        fig6.tight_layout()
        fig6.savefig(os.path.join(val_dir, f'V1_error_spectrum_{rid}.png'), dpi=150)

    plt.close('all')


def _build_save_dict(cfg, data, norm, hp, na, nb, y_ref, y_hat_enc, t_val,
                     epoch_id_full, loss_val_full, loss_train_full,
                     nrms_enc, rms_enc, x_enc_phys, x_enc_ann,
                     cheat_n, nxd, HAS_ORACLE, y_hat_xlog, nrms_xlog,
                     diag_conv, baseline_nrms, nrms_test, y_hat_test,
                     r2_roll_raw, r2_roll_lin, baseline_test_nrms,
                     baseline_encinit_nrms, baseline_test_encinit_nrms,
                     params_init_np, params_learned_np):
    from model_augmentation.systems.gantry_ss import P
    NX_PHYS = cfg.nx_phys
    ystd = norm.ystd
    val_data = data.val_data
    SNR = cfg.snr
    sigma_n = data.sigma_n

    save_dict = dict(
        # Predictions and targets
        y_ref=y_ref, y_hat_enc=y_hat_enc, t_val=t_val,
        u_val=val_data.u,
        # Loss curves
        epoch_id=epoch_id_full, loss_val=loss_val_full, loss_train=loss_train_full,
        # Per-channel metrics (rms_* = nrms_* x ystd, [m])
        nrms_enc=nrms_enc, rms_enc=rms_enc,
        x_enc_phys=x_enc_phys, x_enc_ann=x_enc_ann,
        val_x_aug=data.val_x_aug,
        # Normalization constants (for reconstruction)
        std_x=norm.std_x, x_mean=norm.x_mean, std_u=norm.std_u, u_mean=norm.u_mean,
        ystd=ystd, y0=norm.y0, Cd_norm=norm.Cd_norm, Dd_np=norm.Dd_np,
        P_matrix=P.numpy(),
        # Output-noise floor (D-078): SNR in dB, sigma_n [m]; -1/0.0 = noiseless
        noise_snr=np.array(SNR if SNR is not None else -1),
        noise_sigma=np.array(sigma_n if sigma_n is not None else 0.0),
        # Model dimensions
        cheat_n=np.array(cheat_n), dt=np.array(val_data.dt),
        na=np.array(na), nb=np.array(nb), nf=np.array(hp['nf']),
        NX_PHYS=np.array(NX_PHYS), NX_ANN=np.array(hp['NX_ANN']), nxd=np.array(nxd),
        # Config metadata
        hp=json.dumps(hp),
        config=json.dumps(config_json_dict(cfg)),
    )
    if HAS_ORACLE:
        save_dict['y_hat_xlog'] = y_hat_xlog
        save_dict['nrms_xlog'] = nrms_xlog
    if diag_conv is not None:
        save_dict['diag_conv_epochs']    = diag_conv['epochs']
        save_dict['diag_conv_r2_raw']    = diag_conv['r2_raw']
        save_dict['diag_conv_r2_linmap'] = diag_conv['r2_linmap']
    if baseline_nrms is not None:
        save_dict['baseline_nrms'] = baseline_nrms
        save_dict['baseline_rms']  = baseline_nrms * ystd
    if nrms_test is not None:
        save_dict['nrms_test']  = nrms_test
        save_dict['rms_test']   = nrms_test * ystd
        save_dict['y_hat_test'] = y_hat_test
        save_dict['y_test_ref'] = data.test_data.y
    if r2_roll_lin is not None:
        save_dict['r2_rollout_raw']    = r2_roll_raw
        save_dict['r2_rollout_linmap'] = r2_roll_lin
    if baseline_test_nrms is not None:
        save_dict['baseline_test_nrms'] = baseline_test_nrms
        save_dict['baseline_test_rms']  = baseline_test_nrms * ystd
    if baseline_encinit_nrms is not None:
        save_dict['baseline_encinit_nrms'] = baseline_encinit_nrms
        save_dict['baseline_encinit_rms']  = baseline_encinit_nrms * ystd
    if baseline_test_encinit_nrms is not None:
        save_dict['baseline_test_encinit_nrms'] = baseline_test_encinit_nrms
        save_dict['baseline_test_encinit_rms']  = baseline_test_encinit_nrms * ystd
    if params_learned_np is not None:
        save_dict['params_init']    = params_init_np
        save_dict['params_learned'] = params_learned_np
    return save_dict
