"""THE training entry point for trajectory orthogonality, with live diagnostics.

This is a STANDALONE entry point, not a patch on `gantry_interconnect_dynamic.py`. That file
builds a model, attaches a simulator and trains; it knows nothing about a frozen reference
geometry, an encoder conversion, or a gradient contribution applied between the base backward and
the optimizer step, and teaching it would put five conditionals on a path every other experiment
in this project shares. The sequence this method needs is its own, so it lives here:

    load the reference checkpoint  ->  attach the closed loop  ->  convert the encoder
    ->  build or load the frozen geometry  ->  apply the rank gate  ->  attach the hook
    ->  train through the production fit(), with the penalty in the loop

`gantry_dynamic.model.build_model` REFUSES `traj_orth=True` for exactly this reason: a config
that claims an objective the run does not optimise is a wrong experiment that looks like a right
one.

WHAT IS SHARED WITH THE VERIFICATION LADDER, and this is the point. The environment, the adapter,
the geometry, the penalty hook and the fit seam are the SAME objects `verify.py` exercises. There
is no second implementation of the rollout, the projection or the accumulation, so what the
ladder verifies is what this trains with.

LIVE DIAGNOSTICS, printed every `--log-every` updates, because a penalty that is silently inert
or silently dominant is the failure mode here and neither is visible in the loss alone:

    V            the penalty value, (beta/N)||Q^T d||^2
    ell          ||Q^T d|| / ||M_E d||, the ALIGNMENT. Falls if the contribution rotates out of
                 the protected subspace; flat under uniform shrinkage
    |d|          the contribution size. Reported WITH ell, because a penalty can lower the
                 alignment ratio by switching the augmentation off, and the ratio alone cannot
                 tell that apart from the rotation the method is for
    g_pen/g_base per group, dimensionless. The cross-group magnitude ratio is not comparable;
                 this is
    drift        principal angles between the current physical sensitivity basis and the frozen
                 reference, when --drift-every is set. The frozen-basis assumption is the one
                 this method rests on and never measures during training

NOT AN EFFICACY RUN unless you give it paired arms and a selected beta, and `traj_beta` is a
CHOSEN CONSTANT WITH NO JUSTIFICATION until an L-curve is run (RULES.md, 2026-09-07).

Run:
    conda run -n GraduationProject python scripts/gantry/orthogonality/train_orth.py \
        --run-dir <reference run dir> --its 20 --device cpu --log-every 1
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


# ---------------------------------------------------------------------------- diagnostics
class LiveMeters:
    """Printed per update, from what the hook already computed.

    CORRECTED 2026-09-08. The first version of this docstring claimed "nothing here re-runs a
    rollout" while `snapshot` called `adapter.contributions()`, which is THREE rollouts
    (full/off/clamped) over the regularisation windows. At `--log-every 1` the diagnostics
    therefore cost more than the penalty they were measuring, which is exactly the failure the
    sentence denied.

    Now the hook publishes `d` from the evaluation it already performed, and the meters are
    algebra on that vector: `ell`, the two amplitudes and the gradient ratios are all
    projections of `d` against a frozen basis. The ROUTE SPLIT (`d_lat`, `d_dir`) genuinely
    needs the clamped rollout, so it is gated behind `--route-every` and off by default; the
    same is true of the basis drift, which needs 14 JVPs.
    """

    def __init__(self, hook, adapter, geom, groups, every=1, drift_every=0, route_every=0):
        self.hook, self.ad, self.geom, self.groups = hook, adapter, geom, groups
        self.every, self.drift_every = int(every), int(drift_every)
        self.route_every = int(route_every)
        self.rows = []
        self.Q0 = geom.Q_S.detach().cpu().numpy().copy()

    def _grad_ratios(self, base, after):
        out = {}
        for name, ps in self.groups.items():
            b = base.get(name, 0.0)
            a = after.get(name, 0.0)
            out[name] = dict(base=b, total=a,
                             penalty_over_base=(abs(a - b) / b if b > 0 else float('nan')))
        return out

    def snapshot(self, it, train_loss, base_grads, total_grads):
        h = self.hook.last or {}
        row = dict(it=it, train_loss=float(train_loss), V=h.get('V'),
                   grads=self._grad_ratios(base_grads, total_grads))
        d = h.get('d')                      # published by the hook; NO extra rollout
        if d is not None:
            m = self.geom.contribution_metrics(d)
            row.update(ell=m['ell'], d_norm=m['norm_profiled'], d_norm_raw=m['norm_raw'],
                       a_par=m['a_par'], a_perp=m['a_perp'])
        # THE ROUTE SPLIT COSTS A CLAMPED ROLLOUT and is off by default. It is worth paying for
        # occasionally: the projected total can be small while the two routes are individually
        # aligned and cancelling, and only the split can see that.
        if self.route_every and it % self.route_every == 0:
            with torch.no_grad():
                c = self.ad.contributions()
            mm = {k: self.geom.contribution_metrics(c[k]) for k in ('lat', 'direct')}
            row['ell_lat'] = mm['lat']['ell']
            row['ell_dir'] = mm['direct']['ell']
        if self.drift_every and it % self.drift_every == 0:
            row['drift'] = self._drift()
        self.rows.append(row)
        return row

    def _drift(self):
        """Principal angles between the CURRENT physical sensitivity span and the frozen one.

        Costs 14 forward JVPs, so it is off by default and gated behind `--drift-every`. It is
        the only number that says whether the frozen-basis assumption is still standing; the
        specification's Eq. (26) bound is stated in terms of exactly this.
        """
        from model_augmentation.fit_systems.trajectory_orth_projection import (
            subspace_drift, _orth)
        S_now = self.ad.physical_jacobian()
        Q_now, _sv, _r = _orth(S_now, rtol=self.geom.rank_rtol)
        return subspace_drift(self.Q0, Q_now)

    def line(self, row):
        g = row['grads']
        parts = [f"it {row['it']:>4d}", f"loss {row['train_loss']:.4e}"]
        if row['V'] is not None:
            parts.append(f"V {row['V']:.3e}")
        if 'ell' in row:
            parts += [f"ell {row['ell']:.4f}", f"|d| {row['d_norm']:.3e}"]
        if 'ell_lat' in row:
            parts.append(f"ell_lat/dir {row['ell_lat']:.3f}/{row['ell_dir']:.3f}")
        for name in ('augmentation', 'physical', 'encoder'):
            r = g.get(name, {}).get('penalty_over_base')
            if r is not None and np.isfinite(r):
                parts.append(f"g_pen/g_base[{name[:3]}] {r:.2e}")
        if 'drift' in row and np.isfinite(row['drift'].get('max_deg', float('nan'))):
            parts.append(f"drift {row['drift']['max_deg']:.2f}deg")
        return '  '.join(parts)


def _group_grad_norms(groups):
    return {k: float(torch.cat([(torch.zeros_like(p).reshape(-1) if p.grad is None
                                 else p.grad.reshape(-1)) for p in ps]).norm())
            for k, ps in groups.items() if ps}


class _Observer:
    """Live-diagnostic callback. A CLASS so it is picklable and inspectable, not a closure."""

    def __init__(self, meters, fit_sys, groups, every):
        self.meters, self.fit_sys, self.groups, self.every = meters, fit_sys, groups, int(every)

    def __call__(self, hook, call_index):
        if not self.every or call_index % self.every:
            return
        losses = self.fit_sys.Loss_train
        row = self.meters.snapshot(
            call_index, losses[-1] if len(losses) else float('nan'),
            getattr(hook._contribution, 'base', {}), _group_grad_norms(self.groups))
        print('[traj] ' + self.meters.line(row), flush=True)


class _BaseGradCapture:
    """Wraps the hook's contribution call to record the BASE gradient norms.

    They have to be read before the penalty touches `.grad`, and by the time the observer runs
    the penalty is already in there, so the capture point is the first thing the hook does. Also
    a class rather than a closure, for the reason above.
    """

    def __init__(self, hook, groups):
        self._inner = hook._contribution
        self.groups = groups
        self.base = {}

    def __call__(self, ix=None):
        if ix is None:
            self.base = _group_grad_norms(self.groups)
        return self._inner(ix)


# ---------------------------------------------------------------------------- the run
def build(args, traj_orth=True, verbose=True):
    """Assemble the environment, geometry and hook. Returns (env, adapter, geom, hook)."""
    from gantry.env import load_env
    from gantry.encoder_split import split_encoder, SplitEncoderInitAug
    from gantry_dynamic.traj_orth import build_traj_penalty

    # Optional on `args` so the default stays EAGER for every existing caller; only the GPU
    # ladder sets it. RunConfig's own guard rejects a compile mode on a non-CUDA device.
    env = load_env(args.run_dir, use_f64=args.f64, joint_estimation=True,
                   device=args.device, verbose=verbose,
                   compile_mode=getattr(args, 'compile_mode', None))
    env.cfg = replace(env.cfg, traj_orth=bool(traj_orth), traj_beta=args.beta,
                      traj_windows=args.windows, traj_chunk=args.chunk,
                      traj_rank_rtol=args.rank_rtol, traj_log_every=0,
                      n_its=args.its, its_per_val=args.its, epochs=1, seed=args.seed,
                      lr=args.lr)
    fit_sys = env.fit_sys

    if traj_orth:
        hook = build_traj_penalty(fit_sys, env.cfg, env.data, env.norm, env=env,
                                  verbose=verbose)
        return env, hook.adapter, hook.geometry, hook
    # THE CONTROL ARM CONVERTS THE ENCODER TOO. The conversion preserves outputs but changes
    # parameter sharing, so an unconverted control would differ by two things at once.
    if not isinstance(fit_sys.encoder, SplitEncoderInitAug):
        fit_sys.encoder = split_encoder(fit_sys.encoder).to(
            next(fit_sys.hfn.parameters()).dtype)
    return env, None, None, None


def run_arm(args, traj_orth, verbose=True):
    from gantry.adapter import GantryTrajectoryAdapter                    # noqa: F401

    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env, ad, geom, hook = build(args, traj_orth=traj_orth, verbose=verbose)
    fit_sys = env.fit_sys

    # FRESH OPTIMIZER STATE IN BOTH ARMS. A resumed Adam moment carries the pre-conversion
    # parameter sharing, so comparing an arm that inherited it against one that did not compares
    # two different optimisation problems (spec Sect. 4.2, 6.1).
    # deepSI's `parameters` is a PROPERTY returning param-group dicts, not a method yielding
    # tensors; `parameters()` raises "'list' object is not callable". Mirrors `init_model`.
    fit_sys.optimizer = torch.optim.Adam(fit_sys.parameters, lr=args.lr, eps=args.adam_eps)

    meters = None
    if hook is not None:
        groups = hook.groups()
        meters = LiveMeters(hook, ad, geom, groups, every=args.log_every,
                            drift_every=args.drift_every, route_every=args.route_every)
        # THE HOOK'S `observer` ATTRIBUTE, not a monkey-patched `__call__`. Patching it with a
        # local closure made the whole fit system unpicklable and killed the first real run at
        # its first validation ("Can't pickle local object 'run_arm.<locals>.logged'"). The hook
        # is now excluded from checkpoints as well, but the observer is the right seam either
        # way: it does not change what the hook IS.
        #
        # The base gradient norms are captured by a pre-hook on the optimizer step, because by
        # the time the observer runs the penalty is already in `.grad` and the base value is no
        # longer recoverable.
        # A MODULE-LEVEL CLASS, not a closure. Even with the hook excluded from checkpoints, a
        # local function attached to a long-lived object is one refactor away from being pickled
        # again; that has now happened twice in this file's short life.
        hook.observer = _Observer(meters, fit_sys, groups, args.log_every)
        hook._contribution = _BaseGradCapture(hook, groups)

    print(f'\n=== training: traj_orth={traj_orth}, {args.its} updates, '
          f'beta={args.beta:g} (a chosen constant) ===', flush=True)
    fit_sys.fit(train_sys_data=env.data.train_data, val_sys_data=env.data.val_ckpt_data,
                batch_size=args.batch, epochs=1, n_its=args.its, its_per_val=args.its,
                auto_fit_norm=False,
                loss_kwargs={'nf': env.cfg.nf, 'stride': env.cfg.stride},
                optimizer_kwargs={'lr': args.lr, 'eps': args.adam_eps},
                cuda=(args.device == 'cuda'), validation_measure='sim-RMS')

    out = dict(traj_orth=bool(traj_orth), its=args.its, seconds=time.time() - t0,
               bestfit=float(fit_sys.bestfit),
               loss_train=[float(x) for x in np.atleast_1d(fit_sys.Loss_train)])
    if hook is not None:
        # TWO DIFFERENT COUNTS, and conflating them produced a spurious "hook_calls=5 vs its=3".
        # `hook.calls` counts every evaluation of the penalty over the hook's whole life, which
        # includes the attach-time ownership and reachability probes. `_traj_hook_calls` counts
        # only the evaluations made inside a wrapped optimizer step, which is the number that has
        # to equal the update count. The probe count is the difference, reported so the gap is
        # explained rather than merely tolerated.
        in_step = int(getattr(fit_sys, '_traj_hook_calls', 0))
        out.update(hook_calls_in_step=in_step,
                   hook_calls_total=int(hook.calls),
                   hook_calls_before_training=int(hook.calls) - in_step,
                   optimizer_steps=int(getattr(fit_sys, '_traj_steps', 0)),
                   last=hook.last,
                   meters=meters.rows,
                   beta_is_a_chosen_constant_with_no_justification=True,
                   geometry=dict(N=int(geom.N), rank_S=int(geom.rank_S),
                                 rank_E=int(geom.rank_E),
                                 rank_S_before_profiling=int(geom.rank_S_before_profiling)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True, help='the REFERENCE run directory')
    ap.add_argument('--its', type=int, default=20, help='optimizer updates')
    ap.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    ap.add_argument('--beta', type=float, default=1.0,
                    help='CHOSEN CONSTANT, no justification (RULES.md); nothing here selects it')
    ap.add_argument('--windows', type=int, default=4, help='fixed regularisation windows')
    ap.add_argument('--chunk', type=int, default=0)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--adam-eps', type=float, default=1e-16)
    ap.add_argument('--rank-rtol', type=float, default=1e-12)
    ap.add_argument('--seed', type=int, default=20260908)
    ap.add_argument('--f64', action='store_true')
    ap.add_argument('--log-every', type=int, default=1)
    ap.add_argument('--drift-every', type=int, default=0,
                    help='principal angles against the frozen basis every N updates. Costs 14 '
                         'JVPs, so it is off by default; it is the only measurement of whether '
                         'the frozen-basis assumption still holds during the run')
    ap.add_argument('--route-every', type=int, default=0,
                    help='log the latent/direct route split every N updates. Costs one extra '
                         'clamped rollout, so it is off by default; it is the only thing that '
                         'can see two aligned routes cancelling inside a small total')
    ap.add_argument('--control', action='store_true',
                    help='also run a paired traj_orth=False arm. Same converted encoder, same '
                         'seed, same fresh optimizer state')
    ap.add_argument('--compile-mode', default=None,
                    choices=(None, 'default', 'reduce-overhead', 'max-autotune'),
                    help='torch.compile mode for the training rollout. Default eager. Parity '
                         'against eager is NOT checked here; run_gpu_ladder.py does that and '
                         'gates on it')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    stamp = os.environ.get('SLURM_JOB_ID') or time.strftime('%Y%m%d_%H%M%S')
    out_dir = a.out or os.path.join(HERE, 'results', f'train_{stamp}')
    os.makedirs(out_dir, exist_ok=True)

    rec = dict(settings=vars(a), stamp=stamp, arms={})
    rec['arms']['penalty_on'] = run_arm(a, True)
    if a.control:
        rec['arms']['penalty_off'] = run_arm(a, False)

    on = rec['arms']['penalty_on']
    rec['checks'] = {
        # The count that must match the update count is the IN-STEP one. `hook_calls_total`
        # legitimately exceeds it by the attach-time probes.
        'hook_fired_once_per_update': (on.get('hook_calls_in_step') == a.its
                                       and on.get('optimizer_steps') == a.its),
        'probe_calls_account_for_the_difference':
            on.get('hook_calls_total', 0) - on.get('hook_calls_in_step', 0)
            == on.get('hook_calls_before_training'),
        'penalty_value_finite': bool(np.isfinite(on.get('last', {}).get('V', np.nan))),
    }
    rec['not_established'] = (
        'This run shows the penalty is applied and what it does to the meters. It is NOT an '
        'efficacy result: beta is unselected, the window set is small, and the V3-V7 gates '
        'certify the geometry, not the training outcome.')
    p = os.path.join(out_dir, 'train_orth.json')
    with open(p, 'w') as f:
        json.dump(rec, f, indent=1, default=str)
    print('\n' + '=' * 92)
    for k, v in rec['checks'].items():
        print(f'  [{"PASS" if v else "FAIL"}] {k}')
    print(f'wrote {p}')
    return 0 if all(rec['checks'].values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
