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

from gantry_dynamic.config import RunConfig, save_dir, config_json_dict, git_provenance
from gantry_dynamic.data import load_datasets, compute_normalization, VAL_FILES, TEST_FILES
from gantry_dynamic.model import build_model, get_encoder_dims
from gantry_dynamic.baselines import compute_all_baselines
from gantry_dynamic.diagnostics import (
    state_recovery_diagnostic, compute_gradient_norms, encoder_init_state,
    print_run_banner,
)
from gantry_dynamic.evaluation import evaluate_and_save, per_record_nrms
from gantry_dynamic.training import train_model_with_diagnostics, resolve_resume_hp

## ═══════════════════════════════════════════════════════════════════════════════
## ALL run parameters -- edit here. Field docs live on RunConfig in
## gantry_dynamic/config.py. Derived values (cfg.d, cfg.ts_new, cfg.nf, cfg.na_nb,
## cfg.hp) are computed from these fields; nf/na_nb can be pinned via
## nf_override / na_nb_override.
## ═══════════════════════════════════════════════════════════════════════════════

CFG = RunConfig(
    # D-188: band [140,230] holds BOTH the 212.13 Hz free-free pole and the 150 Hz
    # anti-resonance; AMP_SCALE=6 is the largest unscaled value, zeta_a=0.03 the JPE floor.
    mode='augmentation_ma50_b140-230_a6_z03',
    # 'linear_map' = Hoekstra 2026 reconstructability init (trainable); 'default' = deepSI learned encoder
    encoder_init='linear_map',
    ann_activation='tanh',        # 'linear' = Identity (Jan's ECC, D-071); 'tanh' = nonlinear ANN
    joint_estimation=False,        # D-076: True = trainable damping/stiffness scalars (orth shakedown, 07-12)
    # CHANGED (D-191): joint estimation uses the ten identifiable combinations directly.
    physics_parameterization='reduced',
    param_rmse_baseline=0.01,     # HEURISTIC: measured initial sqrt-loss, jobs 68675/68676 (D-076 Lambda scale)
    # D7.1/D-111. False = no penalty object and no basis build. The Gyorok 2025
    # regularisation has its own entry point: orthogonality/Gyorok/train_gyorok.py.
    orth=False,
    orth_beta=4.66e-4,
    # Raw detuning is disabled for the reduced block.
    param_init_detune=None,
    # HEURISTIC (D-191): 10% perturbations in COMBO_NAMES order:
    # [kb_sum, cg1, cg2, cy, cb_sum, mh, m_total, m_diff, J_eff, d].
    combo_init_detune=[1.10, 1.10, 0.90, 1.10, 0.90,
                       0.90, 1.10, 1.10, 0.90, 1.10],
    snr=None,                     # dB: 50/55/60; None = noiseless (supervisor 07-07)
    seed=42,
    # Training-loss rollout. True = closed loop (known controller around the model, the
    # cl_train.py objective); False = open loop (recorded plant input replayed).
    closed_loop=True,
    # --- Sampling / data conditioning ---
    fs_orig=20000,
    fs_new=4000,                  # None = no downsampling (use fs_orig)
    stride=10,                   # keep every STRIDE-th BPTT window; 100 matches 69399 (fewer windows -> ~10x faster epoch)
    # D-178: samples dropped from the SCORE at each window start. 0 = the pre-D-178
    # objective, bit-identically. K=100 (25 ms) is the loop settling time, not a fit.
    burn_in=0,
    # KEEP False. cl_direct_vs_residual T4 does show a large float32 ROLLOUT sensitivity once
    # the ANN is active (gap/err 1.4% at gain 0, but 835% at 1e-2 and 867% at 1e-1, i.e.
    # gain-independent once the loop is nonlinear), which looked like a reason to switch.
    # It is not: cl_update_precision.py ran 40 updates in each dtype from identical parameters
    # on identical batches and got cos(dtheta_32, dtheta_64) = 0.999042 with |dtheta| ratio
    # 1.0048. Adam's per-parameter normalisation absorbs the rollout noise, so the UPDATE is
    # unaffected. float64 costs runtime and buys nothing here. The toggle works if ever needed.
    # SCOPE (D-171): that measurement is about ADAM, and its stated mechanism is Adam's
    # per-parameter normalisation absorbing the rollout noise. An L-BFGS phase has no such
    # normalisation and forms its curvature from y = g(x+s) - g(x), a difference of two nearly
    # equal gradients, which is catastrophic cancellation that gets worse as the phase succeeds;
    # the strong-Wolfe line search must resolve loss differences of a relative 1e-6 or below on a
    # loss of order 1e-9, at the edge of float32's ~7 digits. So a run with lbfgs=True wants
    # use_f64=True.
    #
    # COHERENCE IS THE RULE, and it outranks the preference above. One dtype for the Adam phase,
    # the polish phase, and any checkpoint being resumed. training.py::_assert_same_dtype REFUSES
    # a resume across a dtype change rather than casting silently, because the phase's whole
    # output is a before/after comparison and a cast puts a dtype change inside it.
    # CONSEQUENCE: an existing float32 checkpoint (all three in meeting-07-09-2026) is polished at
    # float32. To polish at float64, retrain at float64 -- which job 81463 measured as costing
    # nothing in time on this workload, ~4.1 s per closure in either dtype, 2x memory.
    use_f64=False,
    # D-169: 'cpu' or 'cuda'. NOT bit-identical across the two (float32 reduction order),
    # which is why DEVICE is in config.json. The win needs a larger batch_size with it.
    device='cuda',
    # Steps per gradient-checkpoint segment; 0 = off. EXACT (boundaries are not detached, the
    # gradient still spans all nf steps), unlike truncated BPTT. Required above roughly nf = 2000:
    # at nf = 12000, batch 512 the un-checkpointed graph is ~25 GB and does not fit on CPU either.
    # Costs ~+33% compute for ~nf/chunk times less activation memory.
    checkpoint_chunk=0,
    # D-169: torch.compile mode for the TRAINING rollout only; None = eager.
    # 'reduce-overhead' measured ~6.5x; validation and diagnostics stay eager.
    compile_mode='reduce-overhead',
    save_flag=True,
    nf_probe_print=True,          # print per-epoch train/val nf-window RMS [m] (D-095)
    # D-189: capacity only pays once the rate can use it (2x at lr=1e-5, 6% at 1e-6).
    nx_ann=8,
    # ANN routing rows: (1,4,6,7)=Theta+absorber (D-068); (0,1,2,3,4,5,6,7)=X+Theta+Y+absorber.
    # WARNING: X/Y (K=0) routing cannot use the old 1e-5 rate. The controlled short sweep favors
    # 1e-6, but no completed five-epoch run has yet established its long-run behavior.
    # 14 rows, not 8: with nx_ann=8 the interconnected state is 6 baseline + 8 augmented. The
    # sweep runs that used nx_ann=8 (80499-80501, 80555, 80557) all logged ann_route_ix=(0..13);
    # leaving the 8-row tuple here would silently route only part of the state.
    ann_route_ix=tuple(range(14)),
    n_nodes_per_layer=24,
    n_hidden_layers=3,
    up_sample=1,
    # 512, not 256. Two reasons, both measured. (1) The batch axis is nearly free on the GPU: the
    # rollout is dispatch-bound, so t_step is flat in batch over this range, and 80713 measured
    # 0.50 s/update at 512. (2) The sweep's failure mode at a usable rate is gradient NOISE, not
    # bias: 80557 (lr=1e-4, batch 256) oscillated 3.68e-06 <-> 1.355e-05 between validations.
    # Doubling the batch halves that variance, which is what makes the rate below usable.
    batch_size=512,
    # D-189/D-148: 1e-5 is the rate at which added capacity pays. X/Y (K=0) routing
    # cannot use the old 1e-5 blindly; see the problem log's run table before changing.
    lr=1e-5,
    adam_eps=1e-16,                # D-148: keep 1e-11..1e-14 augmented-writer gradients live.
    # Wall-clock budget, not a convergence criterion. n_its below silently overrides it.
    epochs=400,
    # Batch UPDATES between validations; None = 'epoch' (the historical default). 650 = every 5
    # epochs at batch 512 (130 updates/epoch). On the GPU one validation costs ~162 s against 65 s
    # of training per epoch, so validating every epoch would spend 70% of the wall on validation
    # and cut a 10 h run from ~364 epochs to ~156. MOVES WITH batch_size: change that and this
    # number no longer means 5 epochs.
    its_per_val=650,
    n_its=None,                    # None = epochs decide (exact no-op). Set an int to cap BATCH
    # [s] rollout horizon; nf = nf_seconds / ts_new. 0.100 s = 5*tau_msd.
    nf_seconds=0.100,
    # D-171: polish phase after Adam. Wants use_f64=True (no Adam normalisation to
    # absorb float32 rollout noise), but dtype COHERENCE across phases outranks that.
    lbfgs=True,
    # Outer polish iterations; inner_iter is the strong-Wolfe budget per step.
    lbfgs_max_iter=100,
    lbfgs_inner_iter=20,
    lbfgs_history=50,             # m; 50 = jax_sysid and Drenth. torch defaults to 100.
    # Windows in the polish batch. L-BFGS needs one FIXED batch: same batch, same
    # weights, same loss, or the line search is inconsistent.
    lbfgs_windows=16384,
    # Windows per chunk, None = one stack. MEMORY ONLY; value and gradient unchanged.
    lbfgs_chunk=None,
    # Measured on the job-81262 checkpoint: the loss is ~1.29e-9, so torch's defaults
    # (1e-7 / 1e-9) would end the phase at iteration one.
    lbfgs_tol_grad=1e-12,
    # CHANGED for the freeze pair: 1e-14 -> 0.0, the diagnostic setting named in config.py.
    # torch applies tolerance_change ABSOLUTELY, in three places inside one step()
    # (`gtd > -tol`, `max|t*d| <= tol`, `|loss - prev_loss| < tol`). On a loss of 1.15e-09 the
    # production 1e-14 is a RELATIVE 8.7e-06, which is what ended each of job 81655's blocks
    # after about two iterations. That is correct for a production polish and wrong for THIS
    # experiment, which asks whether the training loss keeps falling and therefore needs the
    # phase to iterate rather than exit. 0.0 disables all three, leaving tol_grad and the cap.
    lbfgs_tol_change=0.0,
    # Parameter groups held fixed during the polish; () = the whole vector.
    lbfgs_freeze=(),
    lbfgs_time_budget_s=None,
    lbfgs_meter_rel_tol=1e-3,
    # 'adam' = normal run, and with RESUME_CHECKPOINT set it CONTINUES that run's Adam phase
    # (D-173 restores the moments) before polishing. 'lbfgs' = skip Adam and polish the restored
    # weights directly; that spelling requires RESUME_CHECKPOINT, enforced in main().
    # CHANGED BACK for the burn-in screen: 'adam'. The freeze pair (81692/81693) is done and its
    # question is answered; this run trains, and `lbfgs=True` runs the polish at the end of the
    # SAME job, so the burn-in objective is exercised by both phases without a second launch.
    start_phase='adam',
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

# Encoder-freeze pair (D-175). Same narrow form and the same reason as the two above: the two
# arms differ ONLY in which parameters the polish may move, and editing this file between the
# launches is how a paired experiment silently stops being paired. The arm that ran is recorded
# in config.json as LBFGS_FREEZE, because the override rewrites CFG before it is serialised.
# Burn-in pair (D-178), same narrow form and the same reason as the overrides above: the two arms
# differ ONLY in whether the window's startup transient is scored, and the arm that ran is
# recorded in config.json as BURN_IN. An integer rather than named arms, because unlike the
# freeze this knob has a meaningful range and a sweep over K is a likely follow-up; the
# construction check in RunConfig.__post_init__ rejects anything outside [0, nf).
_burn_in_ab = os.environ.get('BURN_IN_AB')
if _burn_in_ab is not None:
    try:
        _burn_in_ab_int = int(_burn_in_ab)
    except ValueError as exc:
        raise ValueError(
            'BURN_IN_AB is the number of samples dropped from the score at each window start, '
            f'so it must be an integer; got {_burn_in_ab!r}') from exc
    CFG = replace(CFG, burn_in=_burn_in_ab_int)

_lbfgs_ab_freeze = os.environ.get('LBFGS_AB_FREEZE')
if _lbfgs_ab_freeze is not None:
    _freeze_arms = {'none': (), 'encoder': ('encoder',)}
    if _lbfgs_ab_freeze not in _freeze_arms:
        raise ValueError(
            'LBFGS_AB_FREEZE selects the arm of the encoder-freeze pair and must be '
            f'"none" (control, the whole vector) or "encoder", got {_lbfgs_ab_freeze!r}')
    CFG = replace(CFG, lbfgs_freeze=_freeze_arms[_lbfgs_ab_freeze])


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

    print_run_banner(cfg, default_hp_dict, data, run_id, sdir)

    resume_ckpt = os.environ.get('RESUME_CHECKPOINT')  # e.g. /path/to/gantry_ckpt_68458_stage3
    hp = resolve_resume_hp(cfg, default_hp_dict, resume_ckpt)

    # D-093: self-documenting run folder — config + resolved hp.
    if cfg.save_flag:
        with open(os.path.join(sdir, 'config.json'), 'w') as _f:
            json.dump({**config_json_dict(cfg, git=git_provenance()),
                       'hp': hp, 'run_id': run_id}, _f, indent=2)

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

    baselines = compute_all_baselines(hp, cfg, data, norm, K0,
                                      x0_encinit_val, x0_encinit_test)

    per_record_nrms(fit_sys, VAL_FILES, data.val_list, norm, K0, 'Validation-set')
    per_record_nrms(fit_sys, TEST_FILES, data.test_list, norm, K0, 'Test-set')

    evaluate_and_save(fit_sys, hp, run_id, cfg, data, norm, sdir,
                      diag_conv=diag_conv, **baselines)

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
