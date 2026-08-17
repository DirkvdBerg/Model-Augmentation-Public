"""Does the ANN learn the absorber, under each of the three losses?

Production settings, not a smoke test. Each variant is trained and then evaluated with the
pipeline's own evaluation, which reports the three numbers that answer the question:

    baseline NRMS   FP model, no MSD, no ANN          the starting point
    augmented NRMS  the trained model                 did it move?
    oracle NRMS     FP + MSD, true absorber           the target it should reach

"Learned" means augmented lands materially below baseline and toward oracle. Anything else is
"did not learn", however cleanly the code ran.

Variants, differing ONLY in loss() via the mixins in loss_variants.py:
    A   production loss, open loop
    C   residual weighted by So before the norm
    B   controller closed around the model during the rollout

    python train_variants.py A|B|C|all

Run-table note (D-090): the canonical row belongs in
docs/gantry-augmentation-problem-log.md section 12, outside this folder, and has NOT been
written. Hypothesis under test: the ANN reduces the free-run residual toward the FP+MSD oracle,
and the loss variant changes whether it does.
"""
__project_origin__ = "added"

import os
import sys
import json
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry'))
sys.path.insert(0, HERE)

from gantry_dynamic.config import RunConfig
from gantry_dynamic.data import load_datasets, compute_normalization
from gantry_dynamic.model import build_model, train_model, get_encoder_dims
from gantry_dynamic.baselines import compute_baseline_fp_nrms
from gantry_dynamic.diagnostics import encoder_init_state
from gantry_dynamic.evaluation import evaluate_and_save

import loss_variants as LV

# Production settings, matching gantry_interconnect_dynamic.py's CFG.
EPOCHS = int(os.environ.get('CLV_EPOCHS', 12))
STRIDE = int(os.environ.get('CLV_STRIDE', 10))
NF_SECONDS = float(os.environ.get('CLV_NF', 0.100))
LR = float(os.environ.get('CLV_LR', 1e-7))
# Operating point the fixed Cfb is built at, for variants B and C.
#
# Was 0.10, which is the VALIDATION point (V1, V3) and matches NO training record.
# gtd_build_records.m:36-56 gives the training Y_op values: T1-T5 at
# -0.30/-0.15/0.00/+0.15/+0.30, and T6-T14 all at 0.00. So 0.00 matches nine of the
# fourteen exactly and is nearer the other five than 0.10 is.
#
# A single fixed Cfb is structurally correct here: generate_gantry_lti_augmented.m:90-91
# builds Cfb per record from the plant linearised at that record's Y_op, and
# gtd_run_simulation.m:33 applies it with lsim, an LTI simulation. The generating
# controller does NOT gain-schedule along Y, not even on T6-T14 where Y sweeps the full
# range. Per-record Cfb would be more faithful still; that is a separate change.
Y_OP = float(os.environ.get('CLV_YOP', 0.00))


def make_cfg():
    return RunConfig(
        mode='augmentation', encoder_init='linear_map', ann_activation='tanh',
        joint_estimation=False, param_init_detune=None, snr=None, seed=42,
        fs_orig=20000, fs_new=4000, stride=STRIDE, use_f64=False, save_flag=True,
        nf_probe_print=True,
        nx_ann=2, ann_route_ix=(3, 4, 5, 7),
        n_nodes_per_layer=16, n_hidden_layers=2, up_sample=1,
        batch_size=256, lr=LR, epochs=EPOCHS, nf_seconds=NF_SECONDS,
        orth_beta=0.0, orth_observe=False,
    )


def run(variant, data, norm, cfg, baseline_nrms):
    tag = '%s_lr%g_Yop%+.2f' % (variant, cfg.hp['lr'], Y_OP)
    sdir = os.path.join(HERE, 'runs', 'variant_%s' % tag)
    os.makedirs(sdir, exist_ok=True)
    print('=' * 78, flush=True)
    print('VARIANT %s   epochs=%d nf=%d stride=%d lr=%g  -> %s'
          % (variant, EPOCHS, cfg.nf, STRIDE, cfg.hp['lr'], sdir), flush=True)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    fit_sys = build_model(cfg.hp, cfg, data, norm)
    fit_sys = LV.attach(fit_sys, variant, norm, cfg, Y_op=Y_OP)
    print('  class: %s' % type(fit_sys).__name__, flush=True)

    # ENCODER-INIT BASELINE, the reference this experiment is actually judged against.
    #
    # D-089: the x0 estimate must be captured from the UNTRAINED encoder, i.e. here, before
    # train_model touches it. D-094: the verdict must compare the augmented model against the
    # baseline given the SAME initial information. Comparing against a true-x0 baseline makes
    # the percentage an initialisation artefact rather than a statement about the ANN, and
    # today's replay work (D-139) is a direct demonstration of how large that class of artefact
    # gets: a wrong x0 on the K=0 axes bought +1.3e-04 m of permanent offset, 6400x the real
    # error on V1.
    #
    # Without this, evaluate_and_save falls back to `baseline_nrms` and silently labels the
    # verdict line 'baseline FP (true-x0)', answering a different question than the one asked.
    _na, _nb, _na_r, _nb_r = get_encoder_dims(cfg.hp, cfg)
    K0 = max(_na, _nb)
    baseline_encinit_nrms = None
    if cfg.encoder_init == 'linear_map':
        x0_encinit_val = encoder_init_state(fit_sys, data.val_data, K0, _na, _nb,
                                            _na_r, _nb_r, cfg)
        baseline_encinit_nrms, _ = compute_baseline_fp_nrms(
            cfg.hp, cfg, data, norm, x0_norm=x0_encinit_val, start_ix=K0,
            label='val, encoder-init (untrained linear map) -- THE reference to beat')

    t0 = time.time()
    err = None
    try:
        train_model(fit_sys, cfg.hp, cfg, data, epochs=EPOCHS, nf=cfg.nf)
    except Exception as exc:
        err = '%s: %s' % (type(exc).__name__, exc)
        print('  TRAINING RAISED %s' % err, flush=True)
    wall = time.time() - t0
    tl = np.asarray(getattr(fit_sys, 'Loss_train', []), float)
    print('  wall %.1f s (%.1f s/epoch)   train loss %s'
          % (wall, wall / max(len(tl), 1), np.array2string(tl, precision=6)), flush=True)

    res = dict(variant=variant, tag=tag, lr=cfg.hp['lr'], Y_op=Y_OP, err=err, wall=wall,
               train=tl.tolist(), finite=bool(len(tl) and np.all(np.isfinite(tl))),
               baseline_encinit_nrms=(None if baseline_encinit_nrms is None
                                      else np.asarray(baseline_encinit_nrms,
                                                      float).ravel().tolist()))
    try:
        out = evaluate_and_save(fit_sys, cfg.hp, 'variant_%s' % tag, cfg, data, norm, sdir,
                                baseline_nrms=baseline_nrms,
                                baseline_encinit_nrms=baseline_encinit_nrms)
        res['eval'] = str(out)[:2000] if out is not None else None
    except Exception as exc:
        print('  EVALUATION RAISED %s: %s' % (type(exc).__name__, exc), flush=True)
        res['eval_err'] = '%s: %s' % (type(exc).__name__, exc)
    return res


def main():
    which = (sys.argv[1] if len(sys.argv) > 1 else 'all').upper()
    variants = ['A', 'C', 'B'] if which == 'ALL' else [which]
    cfg = make_cfg()
    print('loading datasets at %d Hz ...' % (cfg.fs_new or cfg.fs_orig), flush=True)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    data = load_datasets(cfg)
    norm = compute_normalization(cfg, data)
    print('  nf = %d samples (%.1f ms), na_nb = %d, stride = %d'
          % (cfg.nf, 1e3 * cfg.nf / (cfg.fs_new or cfg.fs_orig), cfg.hp['na_nb'], STRIDE),
          flush=True)

    print('computing baseline FP NRMS (no MSD, no ANN) ...', flush=True)
    # True-x0 reference, started at the first INTERIOR sample K0, not at sample 0.
    # D-087: the velocity at sample 0 comes from a one-sided finite difference, so a sample-0
    # true-x0 start injects a velocity error, and on the K=0 axes a velocity error integrates
    # to a permanent position offset (D-139). This is a secondary reference only; the verdict
    # uses the encoder-init baseline computed per variant in run().
    _na0, _nb0, _, _ = get_encoder_dims(cfg.hp, cfg)
    _K0 = max(_na0, _nb0)
    _b = compute_baseline_fp_nrms(cfg.hp, cfg, data, norm,
                                  x0_phys=data.val_x_logical[_K0], start_ix=_K0,
                                  label='val, true x0 @K0')
    # returns (nrms, y_hat) or similar; keep only the NRMS vector
    baseline_nrms = _b[0] if isinstance(_b, tuple) else _b
    print('  baseline NRMS vector: %s'
          % np.array2string(np.asarray(baseline_nrms, float).ravel(), precision=6), flush=True)

    # One JSON per (variant set, lr, Y_op) so a rate sweep never overwrites an earlier run.
    jname = 'train_variants_result_%s_lr%g_Yop%+.2f.json' % (which, cfg.hp['lr'], Y_OP)
    out = []
    for v in variants:
        out.append(run(v, data, norm, cfg, baseline_nrms))
        with open(os.path.join(HERE, jname), 'w') as f:
            json.dump({'baseline_nrms': np.asarray(baseline_nrms, float).ravel().tolist(),
                       'epochs': EPOCHS, 'stride': STRIDE, 'nf_seconds': NF_SECONDS,
                       'lr': cfg.hp['lr'], 'Y_op': Y_OP, 'variants': out}, f, indent=2)
    print('=' * 78, flush=True)
    print('done. per-variant evaluation printed above; JSON in %s' % jname, flush=True)


if __name__ == '__main__':
    main()
