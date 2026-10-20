"""Polish-only rerun of one final-campaign grey-box run (D-276; D-171, D-255).

The L-BFGS polish of six final runs aborted on GPU memory (R1, R2, R3, R5, R6, R14). This job
repeats ONLY that polish, from the run's own pre-polish checkpoint, with the campaign code unchanged:
  * the configuration is the entry's THESIS block (same runs.tsv row, so the same THESIS_*
    variables), with the start phase set to the polish (D-171): no Adam, no PS2, no encoder start;
  * the steps are those of the entry's main() (gantry_interconnect_dynamic.py at 4207001), copied,
    every function called unchanged;
  * it writes into the ORIGINAL run folder under the ORIGINAL run id, so the evaluation still finds
    one run per design (D-275 D6). The previous outcome must already be in <run>/previous/campaign/;
    the polish starts from the pre-polish pair there, which this job only reads.

    POLISH_RUN_ID=<run folder id> sbatch -a <runs.tsv row id> ... run_polish_only.sbatch
"""
import json
import os
import sys
from dataclasses import replace

import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *['..'] * 4))
sys.path.insert(0, os.path.join(_REPO, 'scripts', 'gantry'))

# Importing the entry runs its THESIS block, which builds the campaign configuration from the
# THESIS_* variables. CFG is read here, never modified.
from gantry_interconnect_dynamic import CFG                                          # noqa: E402
from gantry_dynamic.config import save_dir, config_json_dict, git_provenance         # noqa: E402
from gantry_dynamic.data import load_datasets, compute_normalization, traj_dir       # noqa: E402
from gantry_dynamic.model import build_model                                         # noqa: E402
from gantry_dynamic.controller import build_closed_loop                              # noqa: E402
from gantry_dynamic.diagnostics import (                                             # noqa: E402
    state_recovery_diagnostic, compute_gradient_norms, print_run_banner,
)
from gantry_dynamic.evaluation import evaluate_and_save                              # noqa: E402
from gantry_dynamic.training import train_model_with_diagnostics, resolve_resume_hp  # noqa: E402

# config.json keys allowed to differ from the campaign run; every other key must be identical,
# which is the check that this job runs the campaign's settings.
_MAY_DIFFER = ('START_PHASE', 'GIT', 'POLISH_ONLY')


def main():
    cfg = replace(CFG, start_phase='lbfgs')
    if not (cfg.closed_loop and cfg.lbfgs and cfg.save_flag):
        raise RuntimeError('a final-campaign row must be closed loop, with the polish, and saving')
    run_id = os.environ['POLISH_RUN_ID']
    sdir = os.path.join(save_dir(cfg), run_id)
    prev = os.path.join(sdir, 'previous', 'campaign')
    resume_ckpt = os.path.join(prev, 'gantry_ckpt_%s_pre_lbfgs' % run_id)
    for f in (resume_ckpt + '.pt', resume_ckpt + '.npz', os.path.join(prev, 'config.json')):
        if not os.path.exists(f):
            raise FileNotFoundError('%s is missing: move the previous outcome to %s first' % (f, prev))
    if os.path.exists(os.path.join(sdir, 'gantry_ckpt_%s.pt' % run_id)):
        raise FileExistsError('%s still holds a final checkpoint; nothing is overwritten' % sdir)

    # From here the entry's main(), with run_id, sdir and resume_ckpt set above.
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    data = load_datasets(cfg)
    norm = compute_normalization(cfg, data)
    print_run_banner(cfg, cfg.hp, data, run_id, sdir)
    hp = resolve_resume_hp(cfg, cfg.hp, resume_ckpt)

    # Same settings as the campaign run, key by key, before anything is written.
    now = {**config_json_dict(cfg, git=git_provenance()), 'hp': hp, 'run_id': run_id,
           'data_dir': traj_dir(cfg),
           'POLISH_ONLY': {'from': resume_ckpt, 'job': os.environ.get('SLURM_JOB_ID')}}
    now = json.loads(json.dumps(now))                 # tuples -> lists, as in the stored file
    was = json.load(open(os.path.join(prev, 'config.json'), encoding='utf-8'))
    diff = sorted(k for k in set(was) | set(now)
                  if k not in _MAY_DIFFER and was.get(k) != now.get(k))
    if diff:
        raise ValueError('folder %s and this row differ in %s' % (run_id, diff))
    print('[polish-only] config.json identical to the campaign run except %s' % (_MAY_DIFFER,))
    with open(os.path.join(sdir, 'config.json'), 'w') as fh:
        json.dump(now, fh, indent=2)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    fit_sys = build_model(hp, cfg, data, norm)
    fit_sys.simulator = build_closed_loop(
        fit_sys, norm, cfg,
        train_files=data.train_files, val_files=data.val_files, val_data=data.val_ckpt_data,
        e_train=data.e_train, e_val=data.e_val)
    bestfit, diag_conv = train_model_with_diagnostics(
        fit_sys, hp, cfg, data, norm, resume_ckpt=resume_ckpt, checkpoint_dir=sdir, run_id=run_id)
    print(f"\nPolish-only rerun complete. Validation {cfg.validation_measure} of the saved model: {bestfit:.6g}")

    evaluate_and_save(fit_sys, hp, run_id, cfg, data, norm, sdir, diag_conv=diag_conv)
    print('\n== B. ENCODER QUALITY ==')
    state_recovery_diagnostic(fit_sys, hp, run_id, cfg, data, norm, sdir)
    print('\n== D. TRAINING HEALTH ==')
    try:
        grad_norms, group_norms = compute_gradient_norms(fit_sys, hp, cfg, data)
        np.savez(os.path.join(sdir, f'gantry_grad_norms_{run_id}.npz'),
                 grad_norms=json.dumps(grad_norms), group_norms=json.dumps(group_norms))
    except Exception as e:
        print(f"Warning: gradient norm computation failed: {e}")


if __name__ == '__main__':
    main()
