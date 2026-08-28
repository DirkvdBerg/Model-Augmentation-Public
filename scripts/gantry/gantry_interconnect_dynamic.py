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
from dataclasses import replace

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
    # Track: 'joint' (broadband [1,200] Hz) or 'augmentation' (narrowband [130,180] Hz).
    # Doubles as the dataset folder key (data/gantry/matlab/trajectory/<mode>) and the
    # save_dir / orth-basis-cache key.
    #   'augmentation'         MA_FRAC=0.10
    #   'augmentation_ma50'    MA_FRAC=0.50
    #   'augmentation_ma50_a5' MA_FRAC=0.50 AND 5x excitation (AMP_SCALE=5 in
    #                          generate_trajectory_data.m). Measured 2026-08-27: delta_a scales
    #                          exactly 5.00x, no record hit the limiter, and the augmentation
    #                          target is 91.1 N of which 99.4% is absorber dynamics (the rest
    #                          is the static L0 centre-of-mass offset).
    # NOTE gtd_config sets mh_rigid = mh - ma, so the total below-absorber rigid mass is
    # conserved. That explains the insensitive rigid response, but absorber dynamics are not
    # globally invariant to mass ratio. The 50/50 intervention is closed because it reduced the
    # measured augmentation signal, not because the ratio is absent from the dynamics.
    mode='augmentation_ma50_a5',
    # 'linear_map' = Hoekstra 2026 reconstructability init (trainable); 'default' = deepSI learned encoder
    encoder_init='linear_map',
    ann_activation='tanh',        # 'linear' = Identity (Jan's ECC, D-071); 'tanh' = nonlinear ANN
    joint_estimation=False,        # D-076: True = trainable damping/stiffness scalars (orth shakedown, 07-12)
    param_rmse_baseline=0.01,     # HEURISTIC: measured initial sqrt-loss, jobs 68675/68676 (D-076 Lambda scale)
    # Orthogonal-projection penalty (docs/orthogonal-projection-plan.md; D-111 basis).
    # beta_center = V_MSE/E_drift = 1e-4/2.15e-1 (D7.9, measured 07-12). First entry-file
    # run triggers one fresh ~6 min basis build at up_sample=1 (cached thereafter).
    # MEASURED 2026-08-27 (79502/79503, joint_estimation=False): this penalty is ACTIVE but
    # INERT. It is gated on orth_beta alone, never on joint_estimation, so it does apply with
    # theta frozen -- but the probe reads orth-frac 0.000 and V_orth 1e-14..1e-12 (already
    # beta-weighted, orth_projection.py:11) against a fit loss of order 1e-9+, so it is ~0.01%
    # of the loss. Its gradient is 2*beta*Q Q^T f_ANN, and orth-frac ~ 0 means that is ~0 too.
    # I.e. the ANN's correction already lies almost entirely OUTSIDE the baseline's parameter
    # span without the penalty pushing it there.
    orth_beta=4.66e-4,
    # Set orth_beta=0 AND orth_observe=True for provably zero penalty while KEEPING the
    # orth-frac / V_orth meter. With orth_beta=0 and orth_observe=False the penalty object is
    # never attached (model.py:249) and the meter goes silent, which is how you lose the
    # evidence that the term is inert.
    orth_observe=False,
    # None = start at true values (run T); 14-vector aligned to PARAM_NAMES = detuned start (run D, D-076).
    param_init_detune=None,
    # param_init_detune=[1.10, 1.10, 1.10, 0.90, 1.10, 0.90, 0.90,
    #                    0.90, 1.10, 0.90, 1.10, 0.90, 0.90, 1.10],
    snr=None,                     # dB: 50/55/60; None = noiseless (supervisor 07-07)
    seed=42,
    # Training-loss rollout. True = closed loop (known controller around the model, the
    # cl_train.py objective); False = open loop (recorded plant input replayed).
    closed_loop=True,
    # --- Sampling / data conditioning ---
    fs_orig=20000,
    fs_new=4000,                  # None = no downsampling (use fs_orig)
    stride=10,                   # keep every STRIDE-th BPTT window; 100 matches 69399 (fewer windows -> ~10x faster epoch)
    # KEEP False. cl_direct_vs_residual T4 does show a large float32 ROLLOUT sensitivity once
    # the ANN is active (gap/err 1.4% at gain 0, but 835% at 1e-2 and 867% at 1e-1, i.e.
    # gain-independent once the loop is nonlinear), which looked like a reason to switch.
    # It is not: cl_update_precision.py ran 40 updates in each dtype from identical parameters
    # on identical batches and got cos(dtheta_32, dtheta_64) = 0.999042 with |dtheta| ratio
    # 1.0048. Adam's per-parameter normalisation absorbs the rollout noise, so the UPDATE is
    # unaffected. float64 costs runtime and buys nothing here. The toggle works if ever needed.
    use_f64=False,
    save_flag=True,
    nf_probe_print=True,          # print per-epoch train/val nf-window RMS [m] (D-095)
    # --- Model + training hyperparameters ---
    nx_ann=2,
    # ANN routing rows: (1,4,6,7)=Theta+absorber (D-068); (0,1,2,3,4,5,6,7)=X+Theta+Y+absorber.
    # WARNING: X/Y (K=0) routing cannot use the old 1e-5 rate. The controlled short sweep favors
    # 1e-6, but no completed five-epoch run has yet established its long-run behavior.
    ann_route_ix=(0,1,2,3,4,5,6,7),
    n_nodes_per_layer=16,
    n_hidden_layers=2,
    up_sample=1,
    batch_size=256,
    # MEASURED 2026-08-27, do not raise without re-measuring. With ann_route_ix=(0..7) the
    # old 1e-5 (a Theta-routing value that was never lowered when the routing changed, exactly
    # as the WARNING above says) makes the model WORSE from step one. Controlled 40-update
    # sweep, identical batches, common start asserted (cl_update_lr.py):
    #     1e-5 -> 1.684x worse | 1e-6 -> 0.606x BETTER | 1e-7 -> 0.675x better
    # Server A/B agrees: 79502 (1e-5) degraded val nf-RMS 2.5x, 79503 (1e-7) 1.6x.
    # CONSEQUENCE: 79421, 79422, and 79502 used 1e-5 and are contaminated. Run 79503 used
    # 1e-7 and still degraded, so lowering the rate was necessary but not sufficient. None is
    # a valid 1e-6 baseline for the paired dataset test.
    lr=1e-6,
    adam_eps=1e-16,                # D-148: keep 1e-11..1e-14 augmented-writer gradients live.
    epochs=5,                      # entry-file shakedown (user 07-12); ~30 for the real Step 10 pair
    n_its=None,                    # None = epochs decide (exact no-op). Set an int to cap BATCH
                                   # UPDATES for a smoke test; model.py then also shortens
                                   # its_per_val, because one validation is a ~6 min closed-loop
                                   # free run over V1-V4 and would otherwise dwarf the updates.
    nf_seconds=0.100,             # [s] rollout horizon (5*tau_msd); nf = nf_seconds / ts_new
    # nf_override=None,           # set an int to pin nf directly (bypasses nf_seconds)
    # na_nb_override=None,        # set an int to pin encoder history (bypasses Jan's rule)
)

# Dataset-learnability paired experiment. These two optional overrides let every
# arm import the same entry file while varying only the preregistered dataset and
# common seed. They are deliberately narrow so an accidental environment variable
# cannot change any other run parameter.
_dataset_ab_mode = os.environ.get('DATASET_AB_MODE')
_dataset_ab_seed = os.environ.get('DATASET_AB_SEED')
if _dataset_ab_mode is not None:
    if _dataset_ab_mode not in {'augmentation_ma50', 'augmentation_ma50_a5'}:
        raise ValueError(
            'DATASET_AB_MODE must be augmentation_ma50 or augmentation_ma50_a5, '
            f'got {_dataset_ab_mode!r}')
    CFG = replace(CFG, mode=_dataset_ab_mode)
if _dataset_ab_seed is not None:
    try:
        _dataset_ab_seed_int = int(_dataset_ab_seed)
    except ValueError as exc:
        raise ValueError(
            f'DATASET_AB_SEED must be an integer, got {_dataset_ab_seed!r}') from exc
    CFG = replace(CFG, seed=_dataset_ab_seed_int)


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
    print(f"  LOSS ROLLOUT:   {'closed loop' if cfg.closed_loop else 'open loop'}")
    print(f"  SNR (noise):    {cfg.snr if cfg.snr is not None else 'None (noiseless)'}"
          + (f"  ->  sigma_n={data.sigma_n:.2e} m" if data.sigma_n is not None else ""))
    print(f"  save_dir:       {sdir}")
    print(f"  NF_SECONDS:     {cfg.nf_seconds}")
    print(f"  ADAM_EPS:       {cfg.adam_eps:.1e}")
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

    # ── close the known controller around the model (cfg.closed_loop) ────────────────
    # cfg.closed_loop = False leaves interconnect.py's `simulator = None` default (line 476)
    # in force and the loss is the OPEN-loop rollout. Setting the simulator routes loss()
    # through the closed-loop rollout (interconnect.py:549) and appends ctrl_ix to the
    # training arrays (:600), so the objective becomes the same one cl_train.py trains on
    # and `bestfit` becomes the V1-V4 closed-loop free-run RMS in metres. The two `bestfit`
    # numbers are therefore NOT comparable across the toggle.
    #
    # The gantry-specific bank (Y_op -> controller row per record) lives in scripts/gantry/ by
    # design and cannot move into the framework: closed_loop.py's header states it "does NOT know
    # what a gantry is ... receives stacked (A, B, C, D) matrices and an integer row index per
    # window, and that is all". The import is LOCAL so an open-loop run never touches the
    # controller module at all.
    # CHANGED (2026-08-28): the whole gantry side is now ONE module, gantry_dynamic/controller.py,
    # sitting in the same package as config/data/model/training. What this replaces was a
    # sys.path.insert onto closed-loop-controller/core/, itself a verbatim COPY of six modules in
    # the parent folder, taken so the training path would not sit downstream of the 43 diagnostic
    # scripts there. Both the copy and the path hack are gone; those files are kept for reference
    # and are no longer imported by anything on this path.
    if cfg.closed_loop:
        from gantry_dynamic.controller import build_closed_loop    # noqa: E402
        from gantry_dynamic.data import TRAIN_FILES                # noqa: E402
        # Keyword-only by design: build_closed_loop's docstring records that swapping train_files
        # and val_files "attaches the wrong controller to every record, produces a plausible loss".
        fit_sys.simulator = build_closed_loop(
            fit_sys, norm, cfg,
            train_files=TRAIN_FILES, val_files=VAL_FILES, val_data=data.val_ckpt_data)
        print('\nLoss rollout: CLOSED loop (known controller wrapped around the model)')
    else:
        print('\nLoss rollout: OPEN loop (plant input replayed; simulator = None)')
    # ─────────────────────────────────────────────────────────────────────────────────

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
