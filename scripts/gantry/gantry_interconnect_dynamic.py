"""Gantry augmentation training entry point.

This is the thin entry file: the run knobs (RunConfig below) and the main()
orchestration. All logic lives in the gantry_dynamic package (config, data,
model, baselines, diagnostics, evaluation, training). Behavior is identical to
the pre-refactor monolith (see D-092); the split only reorganizes where code
lives and passes RunConfig/DataBundle/Norm explicitly instead of via globals.

Run: conda run -n GraduationProject python scripts/gantry/gantry_interconnect_dynamic.py
"""
import os
import sys
import json
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gantry_dynamic.config import RunConfig, save_dir, config_json_dict
from gantry_dynamic.data import load_datasets, compute_normalization, VAL_FILES, TEST_FILES
from gantry_dynamic.model import build_model, get_encoder_dims
from gantry_dynamic.baselines import compute_baseline_fp_nrms
from gantry_dynamic.diagnostics import (
    state_recovery_diagnostic, compute_gradient_norms, encoder_init_state,
)
from gantry_dynamic.evaluation import evaluate_and_save
from gantry_dynamic.training import train_model_with_diagnostics

## ═══════════════════════════════════════════════════════════════════════════════
## ALL run parameters -- edit here. Field docs live on RunConfig in
## gantry_dynamic/config.py. Derived values (cfg.d, cfg.ts_new, cfg.nf, cfg.na_nb,
## cfg.hp) are computed from these fields; nf/na_nb can be pinned via
## nf_override / na_nb_override.
## ═══════════════════════════════════════════════════════════════════════════════

CFG = RunConfig(
    # --- Experiment identity ---
    # Track: 'joint' (broadband [1,200] Hz) or 'augmentation' (narrowband [130,180] Hz)
    mode='augmentation',
    # 'linear_map' = Hoekstra 2026 reconstructability init (trainable); 'default' = deepSI learned encoder
    encoder_init='linear_map',
    ann_activation='tanh',        # 'linear' = Identity (Jan's ECC, D-071); 'tanh' = nonlinear ANN
    joint_estimation=True,        # D-076: True = trainable damping/stiffness scalars (orth shakedown, 07-12)
    param_rmse_baseline=0.01,     # HEURISTIC: measured initial sqrt-loss, jobs 68675/68676 (D-076 Lambda scale)
    # Orthogonal-projection penalty (docs/orthogonal-projection-plan.md; D-111 basis).
    # beta_center = V_MSE/E_drift = 1e-4/2.15e-1 (D7.9, measured 07-12). First entry-file
    # run triggers one fresh ~6 min basis build at up_sample=1 (cached thereafter).
    orth_beta=4.66e-4,
    # None = start at true values (run T); 14-vector aligned to PARAM_NAMES = detuned start (run D, D-076).
    param_init_detune=[1.10, 1.10, 1.10, 0.90, 1.10, 0.90, 0.90,
                       0.90, 1.10, 0.90, 1.10, 0.90, 0.90, 1.10],
    snr=None,                     # dB: 50/55/60; None = noiseless (supervisor 07-07)
    seed=42,
    # --- Sampling / data conditioning ---
    fs_orig=20000,
    fs_new=4000,                  # None = no downsampling (use fs_orig)
    stride=100,                   # keep every STRIDE-th BPTT window; 100 matches 69399 (fewer windows -> ~10x faster epoch)
    use_f64=False,
    save_flag=True,
    nf_probe_print=True,          # print per-epoch train/val nf-window RMS [m] (D-095)
    # --- Model + training hyperparameters ---
    nx_ann=2,
    # ANN routing rows: (1,4,6,7)=Theta+absorber (D-068); (0,1,2,3,4,5,6,7)=X+Theta+Y+absorber.
    # WARNING: X/Y (K=0) routing needs lr ~1e-7, not 1e-4, or it diverges after init (D-101/D-102).
    ann_route_ix=(0,1,2,3,4,5,6,7),
    n_nodes_per_layer=16,
    n_hidden_layers=2,
    up_sample=1,
    batch_size=256,
    lr=1e-5,                       # Theta-routing rate (D-101 era; smoke 07-12 healthy at 1e-5).
                                   # NB: 1e-7 is the K=0 X/Y routing value -- restore it when
                                   # switching ann_route_ix back to (0..7) (D-101/D-102).
    epochs=1,                      # entry-file shakedown (user 07-12); ~30 for the real Step 10 pair
    nf_seconds=0.100,             # [s] rollout horizon (5*tau_msd); nf = nf_seconds / ts_new
    # nf_override=None,           # set an int to pin nf directly (bypasses nf_seconds)
    # na_nb_override=None,        # set an int to pin encoder history (bypasses Jan's rule)
)


def main():
    cfg = CFG
    run_id = os.environ.get('SLURM_JOB_ID') or datetime.now().strftime('%Y%m%d_%H%M%S')
    # D-093: per-run subfolder; save_dir(cfg) stays the family dir (shared cache home, D-098).
    sdir = os.path.join(save_dir(cfg), run_id)
    os.makedirs(sdir, exist_ok=True)

    # Seed before data/noise (matches the pre-refactor top-of-module seeding).
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    data = load_datasets(cfg)
    norm = compute_normalization(cfg, data)

    default_hp_dict = cfg.hp

    print(f"\nConfiguration:")
    print(f"  MODE:           {cfg.mode}")
    print(f"  ENCODER_INIT:   {cfg.encoder_init}")
    print(f"  ANN_ACTIVATION: {cfg.ann_activation}")
    print(f"  JOINT_ESTIM:    {cfg.joint_estimation}")
    print(f"  SNR (noise):    {cfg.snr if cfg.snr is not None else 'None (noiseless)'}"
          + (f"  ->  sigma_n={data.sigma_n:.2e} m" if data.sigma_n is not None else ""))
    print(f"  save_dir:       {sdir}")
    print(f"  NF_SECONDS:     {cfg.nf_seconds}")
    print(f"  na_nb (samples): {default_hp_dict['na_nb']}  "
          f"({default_hp_dict['na_nb']/cfg.fs_new_hz*1000:.2f} ms)")
    print(f"  Sampling rate:  {cfg.fs_new_hz} Hz (D={cfg.d})")
    print(f"  Dtype:          {'float64' if cfg.use_f64 else 'float32'}")
    print(f"\nDefault hyperparameters (may be overridden by checkpoint):")
    for k, v in default_hp_dict.items():
        print(f"  {k}: {v}")

    resume_ckpt = os.environ.get('RESUME_CHECKPOINT')  # e.g. /path/to/gantry_ckpt_68458_stage3
    if resume_ckpt:
        _meta = np.load(resume_ckpt + '.npz', allow_pickle=True)
        hp = json.loads(str(_meta['hp']))
        print(f'\nResuming checkpoint: {resume_ckpt}')
        print(f'  Saved hp: {hp}')
    else:
        hp = default_hp_dict

    # D-093: self-documenting run folder — config + resolved hp.
    if cfg.save_flag:
        with open(os.path.join(sdir, 'config.json'), 'w') as _f:
            json.dump({**config_json_dict(cfg), 'hp': hp, 'run_id': run_id}, _f, indent=2)

    # Seed again before model construction (matches pre-refactor second seeding).
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    fit_sys = build_model(hp, cfg, data, norm)

    _na, _nb, _na_right, _nb_right = get_encoder_dims(hp, cfg)
    K0 = max(_na, _nb)   # first sample with a full encoder window (~ model cheat_n)

    # D-089: capture untrained-encoder x0 estimates now (must happen before fit()
    # trains the encoder); the four slow baseline sims run post-training — nothing
    # in training consumes them, and they don't touch fit_sys or the RNG stream.
    if cfg.encoder_init == 'linear_map':
        x0_encinit_val  = encoder_init_state(fit_sys, data.val_data,  K0, _na, _nb, _na_right, _nb_right, cfg)
        x0_encinit_test = encoder_init_state(fit_sys, data.test_data, K0, _na, _nb, _na_right, _nb_right, cfg)
    else:
        x0_encinit_val = x0_encinit_test = None

    ckpt_dir = sdir if cfg.save_flag else None
    bestfit, diag_conv = train_model_with_diagnostics(
        fit_sys, hp, cfg, data, norm, resume_ckpt=resume_ckpt,
        checkpoint_dir=ckpt_dir, run_id=run_id)
    print(f"\nTraining complete. Best validation sim-RMS: {bestfit:.6f}")

    print('\nComputing baseline FP RMS/NRMS (fixed reference, no MSD)...')
    # True-x0 (oracle) baselines — start at interior sample K0 (D-087: sample-0 qdot is a
    # one-sided FD artifact); simulated window matches the model metric (D-072).
    baseline_nrms, _ = compute_baseline_fp_nrms(
        hp, cfg, data, norm, x0_phys=data.val_x_logical[K0], start_ix=K0, label='val, true x0 @K0')
    if data.test_x_logical is not None:
        baseline_test_nrms, _ = compute_baseline_fp_nrms(
            hp, cfg, data, norm, data_sd=data.test_data, x0_phys=data.test_x_logical[K0],
            start_ix=K0, label='test E1, true x0 @K0')
    else:
        baseline_test_nrms = None

    # Encoder-init baselines — same init information as the model, no oracle (D-072).
    # x0 vectors were captured pre-training from the untrained reconstructability map (D-089).
    if x0_encinit_val is not None:
        baseline_encinit_nrms, _ = compute_baseline_fp_nrms(
            hp, cfg, data, norm, x0_norm=x0_encinit_val, start_ix=K0,
            label='val, encoder-init (untrained linear map)')
        baseline_test_encinit_nrms, _ = compute_baseline_fp_nrms(
            hp, cfg, data, norm, data_sd=data.test_data, x0_norm=x0_encinit_test, start_ix=K0,
            label='test E1, encoder-init (untrained linear map)')
    else:
        baseline_encinit_nrms = None
        baseline_test_encinit_nrms = None

    # Per-record augmented NRMS coverage over all held-out records (D-098).
    def _per_record_nrms(files, sdlist, tag):
        print(f'{tag} NRMS per record (augmented, avg from K0):')
        rows = []
        for _f, _td in zip(files, sdlist):
            _yh = fit_sys.apply_experiment(_td).y
            rows.append(np.sqrt(((_yh[K0:] - _td.y[K0:]) ** 2).mean(axis=0)) / norm.ystd)
            print(f'  {_f}: {rows[-1]}')
        print(f'  mean: {np.mean(rows, axis=0)}')
        return rows

    _per_record_nrms(VAL_FILES, data.val_list, 'Validation-set')
    _per_record_nrms(TEST_FILES, data.test_list, 'Test-set')

    evaluate_and_save(fit_sys, hp, run_id, cfg, data, norm, sdir,
                      diag_conv=diag_conv, baseline_nrms=baseline_nrms,
                      baseline_test_nrms=baseline_test_nrms,
                      baseline_encinit_nrms=baseline_encinit_nrms,
                      baseline_test_encinit_nrms=baseline_test_encinit_nrms)

    print('\n== B. ENCODER QUALITY ==')   # D-094 output grouping
    state_recovery_diagnostic(fit_sys, hp, run_id, cfg, data, norm, sdir)

    # Gradient norm snapshot (after evaluation, non-critical)
    print('\n== D. TRAINING HEALTH ==')   # D-094 output grouping
    try:
        grad_norms, group_norms = compute_gradient_norms(fit_sys, hp, cfg, data)
        if cfg.save_flag:
            np.savez(os.path.join(sdir, f'gantry_grad_norms_{run_id}.npz'),
                     grad_norms=json.dumps(grad_norms),
                     group_norms=json.dumps(group_norms))
    except Exception as e:
        print(f"Warning: gradient norm computation failed: {e}")


if __name__ == '__main__':
    main()
