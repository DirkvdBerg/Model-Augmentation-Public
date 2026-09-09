"""THE TWO DECISIVE CHECKS: does the production path execute the specified update?

Everything before this established that the code RUNS. Neither a passing component test, nor a
completed short training run, nor a hook-call count says the update applied to the parameters is
the one the specification defines. These two checks do.

CHECK 1 -- ONE PRODUCTION UPDATE AGAINST AN INDEPENDENT REFERENCE.

    production:  the real `SSE_Interconnect_Composed` step wrapper, driving the real
                 `TrajectoryPenaltyHook`, on a fixed batch
    reference:   the same batch, with the base gradient taken from `fit_sys.loss(...)` and the
                 penalty assembled HERE from first principles --
                     d = p_full - p_off,  V = (beta/N)||Q^T d||^2,
                     backward into eta only, with lambda and tilde x_0 held constant
                 then one Adam step from an identical, cold optimizer state.

    The reference deliberately does NOT call `hook.__call__`, `hook._contribution`, or the
    instrumentation wrappers in `train_orth.py`. It DOES call `adapter.scored_stack`, because
    that is the production rollout and writing a second one is the thing this whole design
    forbids: a reference that re-implements the physics would test a different model. So what is
    independently assembled is the PENALTY and its GRADIENT ROUTING, which is what is in
    question; the rollout is shared and is verified separately by V1/V2.

    SCOPE. This covers one update on a fixed batch. It does NOT cover deepSI's batch selection,
    shuffling or epoch bookkeeping; those are data plumbing, not the update mathematics.

CHECK 2 -- SAVE / LOAD / RESUME AGAINST UNINTERRUPTED TRAINING.

    arm A:  N updates, uninterrupted
    arm B:  N/2 updates, `checkpoint_save_system`, `checkpoint_load_system`, geometry and hook
            reattached, N/2 more updates
    compare the resulting parameters.

    Excluding the hook from serialisation is only acceptable if resumption cannot silently
    disable or change the method, and that is a claim about the RESULT, not about the exclusion.
    deepSI replaces `self.__dict__` wholesale on load, which has already produced two defects
    here, so this is measured rather than argued.

Run:
    conda run -n GraduationProject python scripts/gantry/orthogonality/check_update.py \
        --run-dir <reference run dir> --windows 2 --batch 16
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

PASS, FAIL = [], []


def check(name, ok, detail=''):
    (PASS if ok else FAIL).append(name)
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f'   {detail}' if detail else ''))
    return ok


def flat(ps):
    return torch.cat([p.detach().reshape(-1) for p in ps]).clone()


def flat_grad(ps):
    return torch.cat([(torch.zeros_like(p).reshape(-1) if p.grad is None else p.grad.reshape(-1))
                      for p in ps]).clone()


def rel(a, b):
    d = float((a - b).norm())
    n = max(float(b.norm()), 1e-300)
    return d / n


# --------------------------------------------------------------------------- setup
def setup(args, verbose=True):
    from gantry.env import load_env, production_windows, to_tensors
    from gantry_dynamic.traj_orth import build_traj_penalty

    # `compile_mode` is optional on `args` so every existing caller keeps running EAGER, which
    # is what the verification stages require. Only the GPU ladder passes it.
    env = load_env(args.run_dir, use_f64=args.f64, joint_estimation=True,
                   device=args.device, verbose=verbose,
                   compile_mode=getattr(args, 'compile_mode', None))
    env.cfg = replace(env.cfg, traj_orth=True, traj_beta=args.beta,
                      traj_windows=args.windows, traj_chunk=0, seed=args.seed)
    hook = build_traj_penalty(env.fit_sys, env.cfg, env.data, env.norm, env=env,
                              verbose=verbose)
    # A FIXED BATCH, selected once and reused by every arm. deepSI's loader shuffles; the update
    # mathematics must be compared on identical data or nothing else is comparable.
    recs = list(range(min(2, len(env.data.train_list))))
    win = production_windows(env, recs, max(1, args.batch // len(recs)), seed=args.seed + 7,
                             verbose=False)
    keep = np.arange(min(args.batch, win['n_windows']))
    batch = to_tensors(win, env.dtype, device=args.device)
    batch = {k: (v[keep] if hasattr(v, 'shape') and v.shape[0] == win['n_windows'] else v)
             for k, v in batch.items()}
    return env, hook, batch


def base_loss(fit_sys, batch):
    kw = {'ctrl_ix': batch['ctrl_ix']} if 'ctrl_ix' in batch else {}
    return fit_sys.loss(batch['uhist'], batch['yhist'], batch['ufuture'], batch['yfuture'], **kw)


def fresh_adam(groups, lr, eps):
    return torch.optim.Adam([p for ps in groups.values() for p in ps], lr=lr, eps=eps)


# --------------------------------------------------------------------------- check 1
def reference_penalty_gradient(env, hook, groups):
    """Assemble V and its eta-only gradient HERE, without touching the hook's own machinery."""
    ad, geom, beta = hook.adapter, hook.geometry, hook.beta
    lam, xip, eta = groups['physical'], groups['encoder'], groups['augmentation']

    saved_rg = [(p, bool(p.requires_grad)) for p in list(lam) + list(xip)]
    for p, _ in saved_rg:
        p.requires_grad_(False)
    try:
        xp, xa = ad.encode()
        xp = xp.detach()                                  # tilde x_0 constant, spec Sect. 8
        with torch.no_grad():
            p_off = ad.scored_stack(mode='off', x0=ad.x0_from(xp, torch.zeros_like(xa)))
        p_full = ad.scored_stack(mode='full', x0=ad.x0_from(xp, xa))
        d = p_full - p_off
        # V = (beta/N) ||Q^T d||^2, written out rather than delegated to penalty_mse
        r = geom.Q_S.T @ d
        V = beta * (r * r).sum() / geom.N
        V.backward()
    finally:
        for p, was in saved_rg:
            p.requires_grad_(was)
    return float(V.detach()), d.detach()


def check_one_update(env, hook, batch, args):
    print('\n' + '=' * 92)
    print('CHECK 1: one production update against an independently assembled reference')
    print('=' * 92, flush=True)
    fit_sys = env.fit_sys
    groups = hook.groups()
    all_p = [p for ps in groups.values() for p in ps]
    theta0 = {k: flat(v) for k, v in groups.items()}
    start = [p.detach().clone() for p in all_p]

    def restore():
        with torch.no_grad():
            for p, v in zip(all_p, start):
                p.copy_(v)
            for p in all_p:
                p.grad = None

    out = {}

    # ---------------- PRODUCTION ARM: the real wrapper driving the real hook ------------
    restore()
    fit_sys._optimizer = fresh_adam(groups, args.lr, args.adam_eps)
    fit_sys._traj_wrapping = True
    fit_sys._traj_steps = 0
    calls0 = hook.calls
    fit_sys._wrap_optimizer_step(fit_sys._optimizer)

    grads_prod = {}

    def closure(backward=True):
        loss = base_loss(fit_sys, batch)
        if backward:
            fit_sys._optimizer.zero_grad()
            loss.backward()
        return loss

    loss_prod = float(fit_sys._optimizer.step(closure))
    for k, ps in groups.items():
        grads_prod[k] = flat_grad(ps)
    theta_prod = {k: flat(v) for k, v in groups.items()}
    fit_sys._unwrap_optimizer_step(fit_sys._optimizer)
    fit_sys._traj_wrapping = False
    out['production'] = dict(loss=loss_prod, hook_calls=hook.calls - calls0,
                             V=hook.last.get('V'))

    # ---------------- REFERENCE ARM: base + independently assembled penalty -------------
    restore()
    ref_opt = fresh_adam(groups, args.lr, args.adam_eps)
    ref_opt.zero_grad()
    lb = base_loss(fit_sys, batch)
    loss_ref = float(lb.detach())
    lb.backward()
    grads_base = {k: flat_grad(ps) for k, ps in groups.items()}
    V_ref, d_ref = reference_penalty_gradient(env, hook, groups)
    grads_ref = {k: flat_grad(ps) for k, ps in groups.items()}
    ref_opt.step()
    theta_ref = {k: flat(v) for k, v in groups.items()}
    out['reference'] = dict(loss=loss_ref, V=V_ref)

    # ---------------- COMPARE -----------------------------------------------------------
    check('the two arms saw the same base loss',
          abs(loss_prod - loss_ref) <= 1e-12 * max(abs(loss_ref), 1e-30),
          f'production {loss_prod:.12e}  reference {loss_ref:.12e}')
    Vp, Vr = out['production']['V'], V_ref
    check('the two arms computed the same penalty value',
          Vp is not None and abs(Vp - Vr) <= 1e-9 * max(abs(Vr), 1e-30),
          f'production {Vp:.9e}  reference {Vr:.9e}')

    tol = 1e-10 if args.f64 else 1e-5
    for name in ('physical', 'encoder', 'augmentation'):
        g_rel = rel(grads_prod[name], grads_ref[name])
        t_rel = rel(theta_prod[name], theta_ref[name])
        check(f'gradient matches the reference [{name}]', g_rel <= tol,
              f'relative {g_rel:.3e} (tol {tol:g})')
        check(f'post-step parameters match the reference [{name}]', t_rel <= tol,
              f'relative {t_rel:.3e}')
        out.setdefault('per_group', {})[name] = dict(
            grad_rel=g_rel, theta_rel=t_rel,
            base_grad_norm=float(grads_base[name].norm()),
            total_grad_norm=float(grads_ref[name].norm()),
            penalty_share=float((grads_ref[name] - grads_base[name]).norm()
                                / max(float(grads_base[name].norm()), 1e-300)),
            theta_moved=float((theta_ref[name] - theta0[name]).norm()))

    # the penalty must be the ONLY difference between base and total, and only in eta
    for name in ('physical', 'encoder'):
        dd = float((grads_ref[name] - grads_base[name]).norm())
        check(f'the penalty added nothing to [{name}]', dd == 0.0, f'||delta|| {dd:.3e}')
    de = float((grads_ref['augmentation'] - grads_base['augmentation']).norm())
    check('the penalty added something to [augmentation]', de > 0.0, f'||delta|| {de:.3e}')

    restore()
    return out


# --------------------------------------------------------------------------- check 2
def named_state(fit_sys):
    """Every parameter and buffer of the live model, keyed by STABLE NAME.

    Comparison across a checkpoint load must go by name, never by tensor identity. The load
    replaces every module, so identity is guaranteed to differ and asserting it either fails
    trivially or, worse, passes because both sides were read from the same stale object. That
    second case is what the first version of this check did.
    """
    out = {}
    for prefix, mod in (('hfn', fit_sys.hfn), ('enc', fit_sys.encoder)):
        for n, t in list(mod.named_parameters()) + list(mod.named_buffers()):
            out[f'{prefix}.{n}'] = t
    return out


def compare_named(a, b, label, tol):
    """Value comparison by name, reporting absolute AND relative error."""
    missing = sorted(set(a) ^ set(b))
    check(f'{label}: the same names are present', not missing,
          f'differing keys: {missing[:5]}' if missing else f'{len(a)} entries')
    worst_abs, worst_rel, worst_key = 0.0, 0.0, ''
    for k in sorted(set(a) & set(b)):
        x, y = a[k].detach().reshape(-1), b[k].detach().reshape(-1)
        if x.shape != y.shape or not x.is_floating_point():
            continue
        e = float((x - y).abs().max())
        r = e / max(float(y.abs().max()), 1e-30)
        if e > worst_abs:
            worst_abs, worst_key = e, k
        worst_rel = max(worst_rel, r)
    check(f'{label}: values match by name', worst_abs <= tol,
          f'worst abs {worst_abs:.3e} at {worst_key or "-"}, worst rel {worst_rel:.3e}')
    return dict(worst_abs=worst_abs, worst_rel=worst_rel, worst_key=worst_key)


def _sha(t):
    import hashlib
    a = t.detach().cpu().numpy() if hasattr(t, 'detach') else np.asarray(t)
    return hashlib.sha1(np.ascontiguousarray(a)).hexdigest()[:12]


def snapshot_optimizer(opt, fit_sys):
    """Adam moments, step counters and group hyper-parameters, keyed by PARAMETER NAME.

    Keyed by name because the load replaces every tensor, so an id-keyed snapshot cannot be
    compared across it. This captures what a resume actually has to restore: `exp_avg`,
    `exp_avg_sq` and `step` per parameter, plus each parameter's group association.
    """
    name_of = {id(t): n for n, t in named_state(fit_sys).items()}
    out = {'groups': [], 'state': {}}
    for gi, g in enumerate(opt.param_groups):
        out['groups'].append({k: v for k, v in g.items() if k != 'params'})
        for prm in g['params']:
            nm = name_of.get(id(prm), '<unnamed:%d>' % gi)
            st = opt.state.get(prm, {})
            ent = {'group': gi, 'in_state': bool(st)}
            for k in ('step', 'exp_avg', 'exp_avg_sq'):
                if k in st:
                    v = st[k]
                    ent[k] = float(v) if not hasattr(v, 'numel') or v.numel() == 1 else _sha(v)
            out['state'][nm] = ent
    return out


def compare_optimizer(a, b, label):
    """Every moment and counter, compared with NO repair applied first.

    THE ORDERING IS THE POINT. An earlier version called `load_state_dict(saved)` before
    comparing, which would repair a broken restoration and then report that restoration as
    correct. Whatever this reports is what the checkpoint round trip actually produced.
    """
    ok_g = a['groups'] == b['groups']
    check('%s: optimizer group hyper-parameters and count survive' % label, ok_g,
          '%d group(s); %s' % (len(a['groups']),
                               'identical' if ok_g else '%s -> %s' % (a['groups'], b['groups'])))
    only = sorted(set(a['state']) ^ set(b['state']))
    check('%s: the same parameters carry optimizer state' % label, not only,
          ('differing: %s' % only[:6]) if only else '%d parameters' % len(a['state']))
    bad_moment, bad_step, bad_assoc, missing = [], [], [], []
    for k in sorted(set(a['state']) & set(b['state'])):
        x, y = a['state'][k], b['state'][k]
        if x['in_state'] and not y['in_state']:
            missing.append(k)
            continue
        if x.get('group') != y.get('group'):
            bad_assoc.append(k)
        if x.get('step') != y.get('step'):
            bad_step.append(k)
        for m in ('exp_avg', 'exp_avg_sq'):
            if x.get(m) != y.get(m):
                bad_moment.append('%s.%s' % (k, m))
    check('%s: every parameter that had optimizer state still has it' % label, not missing,
          ('lost state: %s' % missing[:6]) if missing else '%d parameters' % len(a['state']))
    check('%s: parameter-to-group associations survive' % label, not bad_assoc,
          ('reassociated: %s' % bad_assoc[:6]) if bad_assoc else 'unchanged')
    check('%s: step counters survive' % label, not bad_step,
          ('differing: %s' % bad_step[:6]) if bad_step else 'unchanged')
    check('%s: Adam first and second moments survive bit-exactly' % label, not bad_moment,
          ('differing: %s' % bad_moment[:6]) if bad_moment else 'all moments identical')
    return dict(groups_ok=ok_g, lost_state=missing, reassociated=bad_assoc,
                bad_step=bad_step, bad_moment=bad_moment)


def snapshot_geometry(hook):
    """The frozen geometry, BY CONTENT.

    The previous version of this assertion was vacuous: it compared `N` with itself
    (`geometry.N == cfg.nf * 0 + geometry.N`) and checked that the rank was positive. Neither can
    fail. The bases, spectra, tolerances and metadata are what must be identical across a load,
    because the whole method rests on the basis being FROZEN.
    """
    g = hook.geometry
    return dict(N=int(g.N), n_phys=int(g.n_phys),
                rank_S=int(g.rank_S), rank_E=int(g.rank_E),
                rank_S_before_profiling=int(g.rank_S_before_profiling),
                rank_rtol=float(g.rank_rtol), rank_atol=float(g.rank_atol),
                scale_LS=float(g.scale_LS), zero_rank=bool(g.zero_rank),
                beta=float(hook.beta), chunk=int(hook.chunk),
                Q_S_sha=_sha(g.Q_S), Q_E_sha=_sha(g.Q_E),
                Q_S_shape=tuple(g.Q_S.shape), Q_E_shape=tuple(g.Q_E.shape),
                sv_S_sha=_sha(np.asarray(g.sv_S)), sv_E_sha=_sha(np.asarray(g.sv_E)),
                metadata=json.dumps(g.metadata, sort_keys=True, default=str))


def snapshot_context(env, adapter):
    """Windows, controller assignments and the record mapping, BY CONTENT.

    `ctrl_ix` is an INDEX INTO THE CONTROLLER BANK, and the bank lives inside the model the load
    replaces. Identical indices against a differently ordered bank is a silent way to score the
    right windows through the wrong controllers, so the bank's own rows are hashed too, not just
    the indices that select them.
    """
    out = {'win.%s' % k: _sha(v) for k, v in adapter.w.items()
           if hasattr(v, 'shape') and getattr(v, 'ndim', 0) > 0}
    out['win.n'] = int(adapter.n_win)
    out['burn_in'] = int(adapter.burn_in)
    out['T'] = int(adapter.T)
    idx = getattr(env.fit_sys.simulator, 'indexer', None)
    out['record_rows'] = json.dumps(
        None if idx is None else
        [list(map(int, r)) if hasattr(r, '__iter__') else int(r) for r in idx.record_rows],
        default=str)
    bank = (getattr(env.fit_sys.simulator, 'controllers', None)
            or getattr(env.fit_sys.simulator, 'bank', None))
    if bank is not None and hasattr(bank, 'state_dict'):
        for k, v in sorted(bank.state_dict().items()):
            out['bank.%s' % k] = _sha(v)
    out['bank_present'] = bank is not None
    return out


def compare_context(a, b, label):
    diff = sorted(k for k in set(a) & set(b) if a[k] != b[k])
    only = sorted(set(a) ^ set(b))
    check('%s: the same context keys are present' % label, not only,
          ('differing: %s' % only[:6]) if only else '%d keys' % len(a))
    check('%s: windows, controller bank rows and record mapping are unchanged' % label, not diff,
          ('CHANGED: %s' % diff[:8]) if diff else
          '%d content hashes identical (bank present: %s)' % (len(a), a.get('bank_present')))
    return dict(changed=diff, missing=only)


def check_resume(env, hook, batch, args):
    """Save / load / resume against uninterrupted training, compared BY NAME.

    REWRITTEN 2026-09-08 after review found the first version testing the wrong identity. It
    asserted that the pre-load `groups` list and `hook.groups()` held the same tensor objects.
    Both were read from the same retained hook, so it passed by demonstrating that the hook still
    held its OLD references, which is precisely the defect. It also built the resumed parameter
    vector from the pre-load `groups` variable, so it may have compared the uninterrupted model
    against stale tensors rather than the resumed one.

    Everything after the load is now resolved from the LOADED model, by name.
    """
    print('\n' + '=' * 92)
    print('CHECK 2: save / load / resume against uninterrupted training')
    print('=' * 92, flush=True)
    fit_sys = env.fit_sys
    n = args.resume_steps
    tol = 1e-10 if args.f64 else 1e-5
    out = {}

    groups0 = hook.groups()
    all_p = [p for ps in groups0.values() for p in ps]
    start_vals = {k: v.detach().clone() for k, v in named_state(fit_sys).items()}

    def restore_start():
        cur = named_state(fit_sys)
        with torch.no_grad():
            for k, v in start_vals.items():
                if k in cur and cur[k].shape == v.shape:
                    cur[k].copy_(v)
        for p in [q for ps in hook.groups().values() for q in ps]:
            p.grad = None

    def run_steps(k, opt):
        def closure(backward=True):
            loss = base_loss(fit_sys, batch)
            if backward:
                opt.zero_grad()
                loss.backward()
            return loss
        for _ in range(k):
            opt.step(closure)

    # ---- arm A: uninterrupted -----------------------------------------------------------
    restore_start()
    fit_sys._optimizer = fresh_adam(hook.groups(), args.lr, args.adam_eps)
    fit_sys._traj_wrapping = True
    fit_sys._wrap_optimizer_step(fit_sys._optimizer)
    run_steps(2 * n, fit_sys._optimizer)
    state_A = {k: v.detach().clone() for k, v in named_state(fit_sys).items()}
    fit_sys._unwrap_optimizer_step(fit_sys._optimizer)
    fit_sys._traj_wrapping = False

    # ---- arm B: n steps, save, load, n steps --------------------------------------------
    restore_start()
    fit_sys._optimizer = fresh_adam(hook.groups(), args.lr, args.adam_eps)
    fit_sys._traj_wrapping = True
    fit_sys._wrap_optimizer_step(fit_sys._optimizer)
    run_steps(n, fit_sys._optimizer)

    pre = {k: v.detach().clone() for k, v in named_state(fit_sys).items()}
    pre_opt = snapshot_optimizer(fit_sys._optimizer, fit_sys)
    pre_geom = snapshot_geometry(hook)
    pre_ctx = snapshot_context(env, hook.adapter)
    opt_state = fit_sys._optimizer.state_dict()      # kept only as a LAST-RESORT repair, below

    tmp = os.path.join(args.out or HERE, 'resume_ckpt')
    os.makedirs(tmp, exist_ok=True)
    fit_sys.checkpoint_save_system(name='_resume_probe', directory=tmp)
    check('save succeeded with the penalty attached', True)
    check('the hook survived the save', fit_sys.traj_penalty is hook)

    fit_sys.checkpoint_load_system(name='_resume_probe', directory=tmp)

    # ---- 1. values match by name across the load ----------------------------------------
    out['across_load'] = compare_named(pre, named_state(fit_sys), 'across the load', tol)

    # ---- 2. the adapter is bound to the LOADED modules ----------------------------------
    ad = hook.adapter
    bound = True
    try:
        ad.assert_bound()
    except RuntimeError as e:
        bound = False
        print('        ' + str(e)[:150])
    check('the adapter is bound to the LOADED encoder, ANN and physical block', bound,
          'identity against the live model, not against its own cached snapshot')

    live = named_state(fit_sys)
    gp = {id(t) for t in live.values()}
    orphan = [k for k, ps in hook.groups().items()
              for p in ps if id(p) not in gp]
    check('no parameter group is orphaned from the loaded model', not orphan,
          f'orphaned in: {sorted(set(orphan))}' if orphan else 'all groups reachable')

    optp = {id(p) for g in fit_sys._optimizer.param_groups for p in g['params']}
    not_opt = [k for k, ps in hook.groups().items() for p in ps if id(p) not in optp]
    check('the optimizer owns the same tensors the hook writes into', not not_opt,
          f'missing from the optimizer: {sorted(set(not_opt))}' if not_opt else 'all owned')

    check('the optimizer is wrapped after a load during a fit',
          getattr(fit_sys._optimizer, '_traj_wrapped', False))

    # ---- 3. the frozen geometry, BY CONTENT ---------------------------------------------
    post_geom = snapshot_geometry(hook)
    gdiff = sorted(k for k in pre_geom if pre_geom[k] != post_geom[k])
    check('the frozen geometry is unchanged across the load', not gdiff,
          ('CHANGED: %s' % [(k, pre_geom[k], post_geom[k]) for k in gdiff][:4]) if gdiff else
          'Q_S %s sha %s, Q_E %s sha %s, rtol %.2e, atol %.2e, beta %s, plus spectra and metadata'
          % (post_geom['Q_S_shape'], post_geom['Q_S_sha'], post_geom['Q_E_shape'],
             post_geom['Q_E_sha'], post_geom['rank_rtol'], post_geom['rank_atol'],
             post_geom['beta']))
    out['geometry'] = dict(changed=gdiff, post=post_geom)

    # ---- 4. windows, controller assignments and the record mapping ----------------------
    out['context'] = compare_context(pre_ctx, snapshot_context(env, hook.adapter),
                                     'across the load')

    # ---- 5. OPTIMIZER STATE, WITH NO MANUAL REPAIR APPLIED FIRST ------------------------
    # Deliberately BEFORE any `load_state_dict`. Calling that first would repair a broken
    # restoration and then report the restoration as correct, which is what the previous version
    # of this check did.
    out['optimizer_as_restored'] = compare_optimizer(
        pre_opt, snapshot_optimizer(fit_sys._optimizer, fit_sys), 'as restored by the checkpoint')

    restored_clean = not any(out['optimizer_as_restored'][k] for k in
                             ('lost_state', 'reassociated', 'bad_step', 'bad_moment'))
    out['manual_optimizer_repair_was_needed'] = not restored_clean
    if not restored_clean:
        # Recorded as a finding, not hidden. The resume comparison below then measures the
        # remaining question (does the adapter resume correctly) instead of stopping here.
        print('        NOTE: the checkpoint did not restore the optimizer state cleanly; '
              'applying the saved state dict so the resume comparison can still proceed. '
              'This is a REPORTED DEFECT, not a repair that makes the check pass.')
        fit_sys._optimizer.load_state_dict(opt_state)

    # ---- 6. the NEXT update matches the uninterrupted branch ----------------------------
    for p in [q for ps in hook.groups().values() for q in ps]:
        p.grad = None
    run_steps(n, fit_sys._optimizer)
    state_B = {k: v.detach().clone() for k, v in named_state(fit_sys).items()}
    fit_sys._unwrap_optimizer_step(fit_sys._optimizer)
    fit_sys._traj_wrapping = False

    out['resumed_vs_uninterrupted'] = compare_named(
        state_A, state_B, 'resumed vs uninterrupted', tol)
    # Update-relative error too: a relative error against a parameter that STARTS AT ZERO
    # (log_params does) is uninterpretable, so it is reported against how far the arm moved.
    moved = 0.0
    for k in set(state_A) & set(start_vals):
        if state_A[k].is_floating_point() and state_A[k].shape == start_vals[k].shape:
            moved = max(moved, float((state_A[k] - start_vals[k]).abs().max()))
    wa = out['resumed_vs_uninterrupted']['worst_abs']
    out['resumed_vs_uninterrupted']['update_relative'] = wa / max(moved, 1e-30)
    check('the divergence is small relative to how far training moved',
          wa <= tol * max(moved, 1e-30) or wa <= tol,
          f'worst abs {wa:.3e} against a total movement of {moved:.3e} '
          f'(update-relative {wa / max(moved, 1e-30):.3e})')
    return out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--windows', type=int, default=2)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--beta', type=float, default=1.0)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--adam-eps', type=float, default=1e-16)
    ap.add_argument('--seed', type=int, default=20260908)
    ap.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    ap.add_argument('--f64', action='store_true')
    ap.add_argument('--resume-steps', type=int, default=2)
    ap.add_argument('--skip-resume', action='store_true')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    t0 = time.time()
    env, hook, batch = setup(a)
    rec = dict(settings=vars(a),
               geometry=dict(N=int(hook.geometry.N), rank_S=int(hook.geometry.rank_S),
                             rank_E=int(hook.geometry.rank_E)),
               batch_windows=int(batch['ufuture'].shape[0]))
    rec['check1'] = check_one_update(env, hook, batch, a)
    if not a.skip_resume:
        rec['check2'] = check_resume(env, hook, batch, a)
    rec['seconds'] = time.time() - t0
    rec['summary'] = dict(passed=len(PASS), failed=len(FAIL), failures=FAIL)
    rec['not_established'] = (
        'These checks cover ONE update on a fixed batch and one save/load/resume cycle. They do '
        'not cover deepSI batch selection, the compiled or CUDA paths, multi-epoch behaviour, '
        'or anything about whether the penalty helps.')

    out_dir = a.out or os.path.join(HERE, 'results', 'check_update')
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, 'check_update.json')
    with open(p, 'w') as f:
        json.dump(rec, f, indent=1, default=str)
    print('\n' + '=' * 92)
    print(f'{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        print('FAILED: ' + ', '.join(FAIL))
    print(f'wrote {p}')
    return 1 if FAIL else 0


if __name__ == '__main__':
    raise SystemExit(main())
