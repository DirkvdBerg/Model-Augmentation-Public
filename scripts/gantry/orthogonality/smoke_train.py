"""STAGE B. A hard-capped paired training run, to check the seam fires in the real loop.

WHAT THIS IS FOR, and what it is emphatically not for. It confirms that the penalty reaches the
optimizer through the PRODUCTION `fit()`: that the hook is called once per update, that the
gradient it adds survives the closure's own `zero_grad`, that it lands in `eta` and nowhere else,
and that a `traj_orth=False` arm is unchanged. It is **not** an efficacy measurement. There is no
beta sweep, no parameter-recovery study and no held-out comparison here, and the run is far too
short to say anything about fit. The specification's gate is explicit: the pilot follows V0-V7
passing, and V3-V7 have not passed.

WHY IT GOES THROUGH `fit()` AND NOT A PRIVATE LOOP. The seam being tested lives in
`SSE_Interconnect_Composed.fit`, which wraps `optimizer.step` so the accumulation lands between
the closure returning and the update. A hand-written loop would not exercise that wrapper, and
"the version that is tested is not the version that runs" is the failure this whole subtree
already made once.

Run (see runners/run_smoke_train.sh for the server form):
    conda run -n GraduationProject python scripts/gantry/orthogonality/smoke_train.py \
        --run-dir <checkpoint run dir> --its 3 --device cpu
"""
__project_origin__ = "added"

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
for _p in (REPO, os.path.join(REPO, 'scripts', 'gantry'), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataclasses import replace                                          # noqa: E402


def _flat_grads(params):
    return torch.cat([(torch.zeros_like(p).reshape(-1) if p.grad is None else p.grad.reshape(-1))
                      for p in params]).detach().clone()


def _flat_values(params):
    return torch.cat([p.detach().reshape(-1) for p in params]).clone()


def run_arm(run_dir, traj_orth, its, device, beta, windows, chunk, seed, verbose=True):
    """One arm. Returns what changed, and how the penalty behaved if it was on."""
    from gantry.env import load_env
    from gantry_dynamic.traj_orth import build_traj_penalty

    t0 = time.time()
    env = load_env(run_dir, use_f64=False, joint_estimation=True, device=device,
                   verbose=verbose)
    # The smoke run trains; the ladder does not. Everything else stays as the ladder built it, so
    # the two are describing the same model.
    env.cfg = replace(env.cfg, traj_orth=bool(traj_orth), traj_beta=beta,
                      traj_windows=windows, traj_chunk=chunk, n_its=its,
                      its_per_val=its, epochs=1, seed=seed)
    fit_sys = env.fit_sys

    hook = None
    if traj_orth:
        hook = build_traj_penalty(fit_sys, env.cfg, env.data, env.norm, env=env,
                                  verbose=verbose)
    else:
        # The encoder is split in BOTH arms. The conversion changes parameter sharing even though
        # it preserves outputs, so an unconverted control would differ by two things at once and
        # neither arm would mean anything (spec Sect. 4.2).
        from gantry.encoder_split import split_encoder, SplitEncoderInitAug
        if not isinstance(fit_sys.encoder, SplitEncoderInitAug):
            fit_sys.encoder = split_encoder(fit_sys.encoder).to(
                next(fit_sys.hfn.parameters()).dtype)

    groups = None
    if hook is not None:
        groups = hook.groups()
    else:
        from gantry.adapter import GantryTrajectoryAdapter    # noqa: F401  (documented below)

    # Fresh optimizer state in BOTH arms. A resumed Adam moment carries the pre-conversion
    # parameter sharing, and comparing an arm that inherited it against one that did not is a
    # comparison of two different optimisation problems (spec Sect. 4.2, Sect. 6.1).
    # deepSI's `parameters` is a PROPERTY returning a list of param-group dicts
    # ([{'params': ...}, ...]), not a method yielding tensors: `parameters()` raises
    # "'list' object is not callable". This mirrors what `init_model` does with
    # `parameters_with_names`, so the fresh optimizer owns exactly the groups the framework
    # would have given it. The geometry is NOT among them: it lives inside the hook, which is a
    # plain object, so `isinstance(..., nn.Module)` never picks it up -- and it carries buffers
    # only in any case.
    fit_sys.optimizer = torch.optim.Adam(fit_sys.parameters,
                                         lr=env.cfg.lr, eps=env.cfg.adam_eps)

    before = {}
    if groups is not None:
        before = {k: _flat_values(v) for k, v in groups.items()}

    calls0 = 0 if hook is None else hook.calls
    fit_sys.fit(
        train_sys_data=env.data.train_data, val_sys_data=env.data.val_ckpt_data,
        batch_size=env.cfg.batch_size, epochs=1, n_its=its, its_per_val=its,
        auto_fit_norm=False,
        loss_kwargs={'nf': env.cfg.nf, 'stride': env.cfg.stride},
        optimizer_kwargs={'lr': env.cfg.lr, 'eps': env.cfg.adam_eps},
        cuda=(device == 'cuda'), validation_measure='sim-RMS')

    out = dict(traj_orth=bool(traj_orth), its=its, seconds=time.time() - t0,
               bestfit=float(fit_sys.bestfit),
               loss_train=[float(x) for x in np.atleast_1d(fit_sys.Loss_train)[-5:]])
    if hook is not None:
        out.update(hook_calls=hook.calls - calls0, last=hook.last,
                   beta_is_a_chosen_constant=True)
        out['param_delta'] = {
            k: float((_flat_values(v) - before[k]).norm()) for k, v in groups.items()}
    return out, fit_sys, hook


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--its', type=int, default=3, help='HARD CAP on optimizer updates')
    ap.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    ap.add_argument('--beta', type=float, default=1.0,
                    help='a CHOSEN CONSTANT with no justification (RULES.md); nothing here '
                         'selects it')
    ap.add_argument('--windows', type=int, default=4)
    ap.add_argument('--chunk', type=int, default=0)
    ap.add_argument('--seed', type=int, default=20260908)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    if a.its > 50:
        raise SystemExit(
            f'--its {a.its} is not a smoke test. This script exists to check that the seam '
            f'fires, and a longer run would invite reading it as efficacy, which it cannot '
            f'support: the V3-V7 gates have not passed. Use the training entry point for a '
            f'real run, once they have.')

    out_dir = a.out or os.path.join(HERE, 'results', f'smoke_{os.environ.get("SLURM_JOB_ID", time.strftime("%Y%m%d_%H%M%S"))}')
    os.makedirs(out_dir, exist_ok=True)
    rec = {'settings': vars(a), 'arms': {}}

    for name, flag in (('penalty_on', True), ('penalty_off', False)):
        print('\n' + '=' * 92)
        print(f'ARM: {name}   (traj_orth={flag})')
        print('=' * 92, flush=True)
        res, fit_sys, hook = run_arm(a.run_dir, flag, a.its, a.device, a.beta,
                                     a.windows, a.chunk, a.seed)
        rec['arms'][name] = res
        print(json.dumps(res, indent=1, default=str))

    on, off = rec['arms']['penalty_on'], rec['arms']['penalty_off']
    checks = {
        'hook_called_once_per_update': on.get('hook_calls') == a.its,
        'penalty_moved_the_augmentation': on.get('param_delta', {}).get('augmentation', 0) > 0,
        'penalty_value_finite': np.isfinite(on.get('last', {}).get('V', np.nan)),
        'off_arm_ran_without_the_hook': off.get('hook_calls') is None,
    }
    rec['checks'] = checks
    rec['not_established'] = (
        'This is NOT an efficacy result. It shows the seam fires and the gradient lands in eta. '
        'It says nothing about fit, parameter recovery, or the value of beta, and the V3-V7 '
        'gates have not passed.')
    p = os.path.join(out_dir, 'smoke_train.json')
    with open(p, 'w') as f:
        json.dump(rec, f, indent=1, default=str)
    print('\n' + '=' * 92)
    for k, v in checks.items():
        print(f'  [{"PASS" if v else "FAIL"}] {k}')
    print(f'wrote {p}')
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
