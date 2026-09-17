"""The ten behavioral checks of handoff sect. 9.10 for the one-step OBC path (D-192), plus the
synthetic rank-loss lifecycle test of sect. 9.5.

CPU, production model, decimated reference set (stride 200: the checks are about the
construction, not the dataset; the full set is qualified in run_qualification.py). Every check
prints its number and a MET / NOT MET verdict against a stated criterion.

  T1  coefficient evaluated once per objective
  T2  gradient passes through the coefficient to the ANN weights
  T3  detaching the coefficient changes the ANN gradient
  T4  stacked orthogonality reaches the MEASURED float32 floor
  T5  orthogonality still holds after an optimizer update (and the stale coefficient does not)
  T6  additional-state outputs bit-identical with the correction on and off
  T7  disabled path bit-identical to the starting commit a52dd55
  T8  all ten reduced physical parameters keep nonzero training gradients
  T9  save and reload leave no functorch wrappers in the block cache; reload reproduces
  T10 ten consecutive objective evaluations cause no recompilation (CUDA; aot_eager, see probe)
  T11 synthetic lifecycle: rank loss retains the previous basis, two in a row abort
  T15 rank-zero reporting and reference-field dtype-cache invalidation
"""
__project_origin__ = "added"

import copy
import json
import os
import pickle
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rig import (RESULTS_DIR, Tee, obc_config, build_system, perturb_output_layer,   # noqa: E402
                    make_batch, head_module, REPO)
from model_augmentation.fit_systems.obc import (OBCLifecycle, OBCBasis,               # noqa: E402
                                                OBCRankLossAbort)

NF = 50
B = 8
RESULTS = {}


def verdict(name, ok, detail):
    RESULTS[name] = {'met': bool(ok), 'detail': detail}
    print('  -> %s: %s' % ('MET' if ok else 'NOT MET', name))


def jvp_path_field(rig, theta, chunk=4096):
    """`F_tilde` exactly as the ROLLOUT forms it: ANN write minus the JVP correction, at the
    reference tuples, pipeline dtype, routed physical rows, stacked sample-major."""
    fs, corr, ref = rig.fs, rig.corr, rig.ref
    ann = rig.ann
    dev = next(ann.parameters()).device
    outs = []
    with torch.no_grad():
        pair = corr.tangent_pair(theta)     # once, as the objective does
        for s in range(0, ref.n, chunk):
            Zf = ref.Z_field[s:s + chunk].to(dev)
            w = ann(Zf)[:, list(corr.field_cols), 0]                        # (n, nr)
            x = Zf[:, :fs.hfn.nx]; u = Zf[:, fs.hfn.nx:]
            c = corr(x, u, pair)[:, list(corr.row_ix), 0]                   # (n, nr)
            outs.append((w - c).reshape(-1))
    return torch.cat(outs)


def orthogonality_floor(rig, n_draws=10, seed=0):
    """r_perp of RANDOM fields with the shape, norm AND in-span fraction of the current ANN
    field, through the same float64 solve -> pipeline-dtype coefficient -> float32 JVP path.

    The in-span fraction matters: a white random field has an in-span fraction of about
    sqrt(p / n_rows) (1.6e-3 on the full set), so the float32 correction it subtracts is tiny
    and its rounding floor with it, whereas the ANN field has about 18 percent in span. A floor
    measured on white fields would therefore be optimistic by the ratio of the two. Each draw is
    `rho * ||F||` of a random in-span direction `J theta_r` plus `sqrt(1 - rho^2) * ||F||` of a
    random out-of-span field, with `rho = r_overlap(F)` of the current ANN field.
    """
    life, corr = rig.life, rig.corr
    F = life.reference_field(rig.ann).detach()
    basis = life.basis
    rho = basis.r_overlap(F)
    g = torch.Generator().manual_seed(seed)
    vals = []
    with torch.no_grad():
        for _ in range(n_draws):
            th_r = torch.randn(basis.n_cols, generator=g, dtype=torch.float64).to(basis.S.device)
            pin = basis.apply(th_r)
            R = torch.randn(F.shape, generator=g).to(F)
            pout = R.to(torch.float64) - basis.project_onto(R)
            R = (pin * (rho * F.norm() / pin.norm())
                 + pout * ((1 - rho ** 2) ** 0.5 * F.norm() / pout.norm())).to(F)
            theta = basis.coefficient(R).to(rig.cfg.dtype_pt)
            # F_tilde through the rollout path but with the random field in place of the ANN write
            c = jvp_correction_only(rig, theta)
            vals.append(basis.r_perp(R - c))
    return vals


def jvp_correction_only(rig, theta, chunk=4096):
    fs, corr, ref = rig.fs, rig.corr, rig.ref
    dev = next(rig.ann.parameters()).device
    outs = []
    with torch.no_grad():
        pair = corr.tangent_pair(theta)
        for s in range(0, ref.n, chunk):
            Zf = ref.Z_field[s:s + chunk].to(dev)
            x = Zf[:, :fs.hfn.nx]; u = Zf[:, fs.hfn.nx:]
            outs.append(corr(x, u, pair)[:, list(corr.row_ix), 0].reshape(-1))
    return torch.cat(outs)


def main():
    sys.stdout = Tee(os.path.join(RESULTS_DIR, 'test_behavior.log'))
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    cfg = obc_config(device='cpu', compile_mode=None, nf_override=NF, stride=3000,
                     static_pass_buffers=bool(os.environ.get('STATIC_PASS')))
    rig = build_system(cfg, ref_stride=200)
    fs, life, corr, ref, ann, phy = rig.fs, rig.life, rig.corr, rig.ref, rig.ann, rig.phy
    perturb_output_layer(ann, 1e-3)
    batch, kw = make_batch(rig, B=B, seed=0)
    print('rig: nf=%d batch=%d N_ref=%d dtype=%s' % (NF, B, ref.n, cfg.dtype_pt))

    # ---- T1 once per objective ---------------------------------------------------------
    print('\nT1 coefficient evaluated once per objective')
    calls = {'ref': 0, 'roll': 0}
    orig_forward = ann.forward
    def counting_forward(z):
        calls['ref' if z.shape[0] == ref.n else 'roll'] += 1
        return orig_forward(z)
    ann.forward = counting_forward
    n0 = life.n_objective_evals
    fs.optimizer.zero_grad(); L = fs.loss(*batch[:4], **kw); L.backward()
    ann.forward = orig_forward
    print('  objective evals +%d, reference-set ANN passes %d, rollout ANN passes %d (nf=%d)'
          % (life.n_objective_evals - n0, calls['ref'], calls['roll'], NF))
    verdict('T1', life.n_objective_evals - n0 == 1 and calls['ref'] == 1 and calls['roll'] == NF,
            calls)
    theta_a = life.last_theta.clone()

    # ---- T2 / T3 gradient through the coefficient --------------------------------------
    print('\nT2 gradient passes through the coefficient to the ANN weights')
    th = life.coefficient(ann)
    g = torch.autograd.grad(th.sum(), list(ann.parameters()), allow_unused=True)
    gn = float(sum((gi ** 2).sum() for gi in g if gi is not None) ** 0.5)
    print('  ||d theta.sum() / d ANN weights|| = %.4e' % gn)
    verdict('T2', gn > 0 and np.isfinite(gn), gn)

    print('\nT3 detaching the coefficient changes the ANN gradient')
    fs.optimizer.zero_grad(); L1 = fs.loss(*batch[:4], **kw); L1.backward()
    g_att = torch.cat([p.grad.reshape(-1).clone() for p in ann.parameters()])
    orig_coef = life.coefficient
    life.coefficient = lambda a: orig_coef(a).detach()
    fs.optimizer.zero_grad(); L2 = fs.loss(*batch[:4], **kw); L2.backward()
    life.coefficient = orig_coef
    g_det = torch.cat([p.grad.reshape(-1).clone() for p in ann.parameters()])
    rel = float((g_att - g_det).norm() / (g_att.norm() + 1e-300))
    cos = float((g_att @ g_det) / (g_att.norm() * g_det.norm() + 1e-300))
    print('  loss identical: %s (%.6e vs %.6e); ||g_att - g_det|| / ||g_att|| = %.4e, cos = %.6f'
          % (bool(torch.equal(L1.detach(), L2.detach())), float(L1), float(L2), rel, cos))
    verdict('T3', bool(torch.equal(L1.detach(), L2.detach())) and rel > 1e-6, {'rel': rel, 'cos': cos})

    # ---- T4 stacked orthogonality at the measured floor --------------------------------
    print('\nT4 stacked orthogonality reaches the measured numerical floor')
    theta = life.coefficient(ann).detach()
    Ft = jvp_path_field(rig, theta)
    r_roll = life.basis.r_perp(Ft)
    d = life.stacked_diagnostics(ann)
    floor = orthogonality_floor(rig)
    # HEURISTIC: acceptance = 3 x the worst of ten random-field draws through the same path.
    C_FLOOR = 3.0
    thr = C_FLOOR * max(floor)
    kappa = life.basis.report.cond
    tau32 = kappa * torch.finfo(torch.float32).eps
    tau64 = kappa * torch.finfo(torch.float64).eps
    F_raw = life.reference_field(ann).detach()
    print('  r_perp, rollout (float32 JVP) path        = %.4e' % r_roll)
    print('  r_perp, float64 stored-factor construction = %.4e' % d['r_perp'])
    print('  r_overlap before projection                = %.4e   (||F|| = %.4e)' % (d['r_overlap'], float(F_raw.norm())))
    print('  measured float32 floor over 10 random fields: min %.3e median %.3e max %.3e'
          % (min(floor), sorted(floor)[5], max(floor)))
    print('  acceptance threshold = %g x max floor = %.3e; for reference kappa*eps: float32 %.3e, float64 %.3e'
          % (C_FLOOR, thr, tau32, tau64))
    verdict('T4', r_roll <= thr and d['r_perp'] <= C_FLOOR * tau64 * 10,
            {'r_perp_rollout': r_roll, 'r_perp_f64': d['r_perp'], 'floor': floor, 'thr': thr,
             'r_overlap': d['r_overlap'], 'kappa': kappa})

    # ---- T5 after an optimizer update --------------------------------------------------
    print('\nT5 orthogonality after one optimizer update (fresh coefficient) and with the stale one')
    fs.optimizer.zero_grad(); L = fs.loss(*batch[:4], **kw); L.backward()
    theta_before = life.last_theta.clone()
    fs.optimizer.step(); fs.optimizer.zero_grad()
    theta_after = life.coefficient(ann).detach()
    r_fresh = life.basis.r_perp(jvp_path_field(rig, theta_after))
    r_stale = life.basis.r_perp(jvp_path_field(rig, theta_before.to(cfg.dtype_pt)))
    n_eval = life.n_objective_evals
    n_refresh = life.n_refresh
    epoch_saved = fs.epoch_counter
    fs.epoch_counter = float(life.last_refresh_epoch + 1)  # validation sits at the next boundary
    fs._sync_prediction_state_for_validation(refresh_basis=False)
    fs.epoch_counter = epoch_saved
    theta_synced = corr.theta_frozen.detach().clone()
    r_synced = life.basis.r_perp(jvp_path_field(rig, theta_synced))
    print('  |theta_after - theta_before| / |theta| = %.3e' % float((theta_after.double() - theta_before).norm() / theta_before.norm()))
    print('  r_perp fresh coefficient = %.4e  (threshold %.3e);  stale coefficient = %.4e; validation-synced = %.4e'
          % (r_fresh, thr, r_stale, r_synced))
    print('  validation sync matches fresh coefficient: %s; objective counter unchanged: %s; epoch basis retained: %s'
          % (bool(torch.equal(theta_synced, theta_after.to(theta_synced.dtype))),
             life.n_objective_evals == n_eval, life.n_refresh == n_refresh))
    verdict('T5', r_fresh <= thr and r_stale > thr and r_synced <= thr
            and bool(torch.equal(theta_synced, theta_after.to(theta_synced.dtype)))
            and life.n_objective_evals == n_eval and life.n_refresh == n_refresh,
            {'fresh': r_fresh, 'stale': r_stale, 'synced': r_synced})

    # ---- T6 additional-state rows untouched -------------------------------------------
    print('\nT6 additional-state outputs bit-identical with the correction on and off')
    x = batch[0].new_zeros(B, fs.hfn.nx, 1).normal_() * 0.3
    u = batch[0].new_zeros(B, 3, 1).normal_() * 0.3
    with torch.no_grad():
        y_on, xp_on = fs.hfn(x, u, corr.tangent_pair(theta_after.to(cfg.dtype_pt)))
        saved = fs.hfn.obc_correction; fs.hfn.obc_correction = None
        y_off, xp_off = fs.hfn(x, u)
        fs.hfn.obc_correction = saved
    aug_eq = bool(torch.equal(xp_on[:, 6:], xp_off[:, 6:]))
    phys_diff = float((xp_on[:, :6] - xp_off[:, :6]).abs().max())
    print('  y equal: %s; augmented rows equal: %s; physical rows max|diff| = %.3e (so the correction acts)'
          % (bool(torch.equal(y_on, y_off)), aug_eq, phys_diff))
    verdict('T6', aug_eq and bool(torch.equal(y_on, y_off)) and phys_diff > 0, {'phys_diff': phys_diff})

    # ---- T7 disabled path bit-identical to the starting commit --------------------------
    print('\nT7 disabled path bit-identical to commit a52dd55 (blocks, Interconnect.forward, closed-loop rollout)')
    hb = head_module('model_augmentation/fit_systems/blocks.py', 'head_blocks')
    hi = head_module('model_augmentation/fit_systems/interconnect.py', 'head_interconnect')
    hc = head_module('model_augmentation/fit_systems/closed_loop.py', 'head_closed_loop')
    from model_augmentation.fit_systems.closed_loop import closed_loop_rollout as new_rollout
    from model_augmentation.fit_systems.interconnect import Interconnect as NewIC
    from model_augmentation.utils.utils import expansion_matrix, selection_matrix
    t7 = {}
    for dt in (torch.float32, torch.float64):
        # (a) blocks: old vs new Reduced block from the same init, same inputs
        gss = __import__('model_augmentation.systems.gantry_ss', fromlist=['x'])
        names = hb.Parameterized_Gantry_State_Block.PARAM_NAMES
        nominal = torch.stack([getattr(gss, n) for n in names]).to(torch.float64)
        combo = hb.Reduced_Gantry_State_Block.combos_of(nominal, gss.Lb) * torch.tensor(cfg.combo_init_detune, dtype=torch.float64)
        common = dict(Y_op=None, std_x=rig.norm.std_x, std_u=rig.norm.std_u, x_mean=rig.norm.x_mean,
                      u_mean=rig.norm.u_mean, Ts=cfg.ts_new, up_sample=rig.hp['up_sample'],
                      RMSE_baseline=cfg.param_rmse_baseline)
        old_blk = hb.Reduced_Gantry_State_Block(params_init=nominal, combo_init=combo, **common).to(dt)
        new_blk = type(phy)(params_init=nominal, combo_init=combo, **common).to(dt)
        with torch.no_grad():
            fp = torch.randn(10, dtype=dt) * 0.05
            old_blk.free_params.copy_(fp); new_blk.free_params.copy_(fp)
            z = torch.randn(64, 9, 1, dtype=dt) * 0.5
            eq_blk = bool(torch.equal(old_blk(z), new_blk(z)))
            # transition_from_free at the live free coordinate must equal the block forward
            eq_pure = bool(torch.equal(new_blk.transition_from_free(new_blk.free_params, z), new_blk(z)))
        # (b) Interconnect.forward old vs new, sharing the NEW blocks of the production model
        fs_dt = fs if dt == cfg.dtype_pt else None
        if fs_dt is None:
            eq_ic = eq_roll = None
        else:
            blocks = list(fs.hfn.connected_blocks)
            nxd, nu, ny = fs.hfn.nx, fs.hfn.nu, fs.hfn.ny
            def wire(IC):
                ic = IC(nxd, nu, ny)
                phy_b, out_b, ann_b = blocks[0], blocks[1], blocks[2]
                route_ix = np.asarray(cfg.ann_route_ix); PHY_IX = np.arange(6)
                ic.add_block(phy_b); ic.add_block(out_b); ic.add_block(ann_b)
                ic.connect_block_signals(ann_b, ["x", "u"], [])
                ic.connect_signals(ann_b, "xp", "additive", expansion_matrix(route_ix, nxd))
                ic.connect_signals("x", phy_b, "concat", selection_matrix(PHY_IX, nxd))
                ic.connect_block_signals(phy_b, ["u"], [])
                ic.connect_signals(phy_b, "xp", "additive", expansion_matrix(PHY_IX, nxd))
                ic.connect_signals("x", out_b, "concat", selection_matrix(PHY_IX, nxd))
                ic.connect_block_signals(out_b, ["u"], ["y"])
                return ic.to(dt)
            old_ic, new_ic = wire(hi.Interconnect), wire(NewIC)
            with torch.no_grad():
                xx = torch.randn(B, nxd, dtype=dt) * 0.3; uu = torch.randn(B, nu, dtype=dt) * 0.3
                yo, xo = old_ic(xx, uu); yn, xn = new_ic(xx, uu)
                eq_ic = bool(torch.equal(yo, yn) and torch.equal(xo, xn))
                # also the production hfn (which has obc_correction attached) with correction OFF
                saved = fs.hfn.obc_correction; fs.hfn.obc_correction = None
                yp, xpp = fs.hfn(xx, uu); fs.hfn.obc_correction = saved
                eq_ic = eq_ic and bool(torch.equal(yp, yn) and torch.equal(xpp, xn))
                # (c) closed-loop rollout old vs new over the same eager hfn
                bank = fs.simulator.bank
                x0 = fs.encoder(batch[0], batch[1])
                fs.hfn.obc_correction = None
                yo, xo, _ = hc.closed_loop_rollout(fs.hfn, fs.hfn.output_only, batch[2], batch[3], x0, bank, batch[4].long())
                yn, xn, _ = new_rollout(fs.hfn, fs.hfn.output_only, batch[2], batch[3], x0, bank, batch[4].long())
                fs.hfn.obc_correction = saved
                eq_roll = bool(torch.equal(yo, yn) and torch.equal(xo, xn))
        t7[str(dt)] = {'block': eq_blk, 'pure_transition': eq_pure, 'interconnect': eq_ic, 'rollout': eq_roll}
        print('  %s: block %s, transition_from_free==forward %s, Interconnect.forward %s, closed-loop rollout (nf=%d) %s'
              % (dt, eq_blk, eq_pure, eq_ic, NF, eq_roll))
    verdict('T7', all(v for d_ in t7.values() for v in d_.values() if v is not None), t7)

    # ---- T8 physical parameter gradients ---------------------------------------------
    print('\nT8 all ten reduced physical parameters retain nonzero training gradients (OBC on)')
    fs.optimizer.zero_grad(); L = fs.loss(*batch[:4], **kw); L.backward()
    gp = phy.free_params.grad.detach().clone()
    print('  free_params.grad = ' + ' '.join('%+.3e' % v for v in gp))
    verdict('T8', bool((gp != 0).all()) and bool(torch.isfinite(gp).all()), gp.tolist())

    # ---- T9 save / reload -------------------------------------------------------------
    print('\nT9 save and reload: no functorch wrappers in the block cache; reload reproduces')
    is_wrapped = getattr(torch._C._functorch, 'is_functorch_wrapped_tensor', None)
    cur = phy._cur
    wrapped = [bool(is_wrapped(t)) for t in cur] if (is_wrapped and cur is not None) else None
    print('  block _cur entries functorch-wrapped: %s' % wrapped)
    with torch.no_grad():
        y_ref, xp_ref = fs.hfn(x, u)                      # frozen coefficient path
    d = tempfile.mkdtemp()
    fs.unique_code = 'obctest'
    fs.checkpoint_save_system(name='_t9', directory=d)
    saved = torch.load(os.path.join(d, fs.name + '_t9.pth'), weights_only=False)
    excluded = 'obc' not in saved and 'obc_reference' not in saved
    blob = pickle.dumps(fs.hfn)
    hfn2 = pickle.loads(blob)
    # reload on the LIVE object, as deepSI does at the end of every fit (a copy.copy would go
    # through __getstate__, which strips the lifecycle, and so cannot test the restore)
    hfn_before = fs.hfn
    fs.checkpoint_load_system(name='_t9', directory=d)
    restored = fs.__dict__.get('obc') is life and fs.__dict__.get('obc_reference') is ref
    with torch.no_grad():
        y2, xp2 = fs.hfn(x, u); y3, xp3 = hfn2(x, u)
    eq = bool(torch.equal(y_ref, y2) and torch.equal(xp_ref, xp2) and torch.equal(xp_ref, xp3))
    keys = [k for k in fs.hfn.state_dict().keys() if 'obc' in k]
    print('  checkpoint %.1f MB; lifecycle + reference set excluded from it: %s; restored on the live '
          'system after load: %s; hfn replaced by the load: %s'
          % (os.path.getsize(os.path.join(d, fs.name + '_t9.pth')) / 1e6, excluded, restored,
             fs.hfn is not hfn_before))
    print('  round trip reproduces (frozen theta path), checkpoint and pickle: %s; obc keys in state_dict: %s'
          % (eq, keys))
    # the lifecycle must keep working against the RELOADED correction module (it resolves the
    # adapter through fs.hfn at call time, never through a captured reference)
    fs.optimizer.zero_grad(); L = fs.loss(*batch[:4], **kw); L.backward()
    post = bool(torch.isfinite(L)) and fs.hfn.obc_correction is not None
    print('  objective after reload runs against the reloaded adapter: %s (loss %.4e)' % (post, float(L)))
    verdict('T9', eq and excluded and restored and post and (wrapped is None or not any(wrapped)),
            {'wrapped': wrapped, 'keys': keys, 'excluded': excluded, 'restored': restored})

    # ---- T10 no recompilation (CUDA only) ---------------------------------------------
    print('\nT10 ten consecutive objective evaluations cause no recompilation')
    if not torch.cuda.is_available():
        print('  no CUDA device here; see probe_compile.py on the training host')
        verdict('T10', False, 'no CUDA')
    else:
        pj = os.path.join(RESULTS_DIR, 'probe_compile.json')
        if os.path.exists(pj):
            pr = json.load(open(pj))['rung1_aot_eager']
            print('  from probe_compile.json: ok=%s, objectives=%s, refreshes=%s, error=%s'
                  % (pr.get('ok'), pr.get('n_objectives_no_recompile'), pr.get('refreshes_during'), pr.get('error')))
            verdict('T10', bool(pr.get('ok')), pr)
        else:
            print('  run probe_compile.py first'); verdict('T10', False, 'probe not run')

    # ---- T12 the two correction implementations agree, in the real module ---------------
    print('\nT12 hand-written tangent vs torch.func.jvp, through GantryOBCCorrection itself')
    xr = batch[0].new_zeros(B, fs.hfn.nx, 1).normal_() * 0.3
    ur = batch[0].new_zeros(B, 3, 1).normal_() * 0.3
    th12 = life.coefficient(ann).detach()
    with torch.no_grad():
        c_hand = corr.correction_from_theta(xr, ur, th12, use_jvp=False)
        c_jvp = corr.correction_from_theta(xr, ur, th12, use_jvp=True)
        c_threaded = corr(xr, ur, corr.tangent_pair(th12))
        c_other = corr(xr, ur, corr.tangent_pair(th12 * 2.0))
        c_other_ref = corr.correction_from_theta(xr, ur, th12 * 2.0, use_jvp=True)
    r_hand = float((c_hand - c_jvp).norm() / (c_jvp.norm() + 1e-300))
    r_thread = float((c_threaded - c_hand).norm() / (c_hand.norm() + 1e-300))
    r_other = float((c_other - c_other_ref).norm() / (c_other_ref.norm() + 1e-300))
    print('  hand vs jvp (%s): %.3e' % (cfg.dtype_pt, r_hand))
    print('  threaded pair vs correction_from_theta: %.3e' % r_thread)
    print('  a DIFFERENT coefficient tracks the jvp too: %.3e' % r_other)
    print('  additional-state rows of the correction: max %.3e' % float(c_hand[:, 6:].abs().max()))
    verdict('T12', r_hand < 1e-5 and r_thread == 0.0 and r_other < 1e-5
            and float(c_hand[:, 6:].abs().max()) == 0.0,
            {'hand_vs_jvp': r_hand, 'threaded': r_thread, 'other': r_other})

    # ---- T13 NO functorch inside the COMPILED step ---------------------------------------
    # The check job 83947 needed and did not have. Under Dynamo an attribute-based cache silently
    # took the fallback branch and ran torch.func.jvp per timestep; eager counting could not see
    # it. Count inside a compiled call instead. `aot_eager` traces the same graph as inductor and
    # needs no Triton, so this runs on the development machine.
    print('\nT13 the compiled step contains no functorch transform')
    n_mt = {'n': 0}
    _orig_mt = phy.mats_and_tangent_from_free
    def _counting_mt(free, dfree):
        n_mt['n'] += 1
        return _orig_mt(free, dfree)
    phy.mats_and_tangent_from_free = _counting_mt
    try:
        torch._dynamo.reset()
        cstep = torch.compile(fs.hfn, backend='aot_eager', fullgraph=True)
        pair = corr.tangent_pair(life.coefficient(ann))
        n_at_trace = n_mt['n']
        with torch.no_grad():
            for _ in range(5):
                cstep(x, u, pair)
        n_in_steps = n_mt['n'] - n_at_trace
        print('  tangent_pair calls during 5 compiled steps: %d (must be 0)' % n_in_steps)
        ok13 = n_in_steps == 0
    except Exception as e:  # noqa: BLE001
        print('  compiled trace failed: %s' % str(e)[:160]); ok13 = False
    finally:
        phy.mats_and_tangent_from_free = _orig_mt
        torch._dynamo.reset()
    verdict('T13', ok13, n_mt['n'])

    # ---- T14 the hoisted M(Y) structure is used, and is exact (D-195) ---------------------
    print('\nT14 the M(Y) rebuild is hoisted out of the timestep loop and changes nothing')
    n_rb = {'n': 0}
    # RE-RESOLVE from the live model: T9 reloads a checkpoint, which replaces `fs.hfn` wholesale,
    # so the `phy` captured at the top of this run is a detached copy and patching it counts
    # nothing. This is the same stale-handle hazard T9 itself exists to catch.
    phy = fs.hfn.connected_blocks[fs._mats_ix]
    _orig_rb = phy._mats_from_raw
    phy._mats_from_raw = lambda p: (n_rb.__setitem__('n', n_rb['n'] + 1), _orig_rb(p))[1]
    try:
        saved_ix, fs._mats_ix = fs._mats_ix, None       # the pre-D-195 behaviour
        fs.optimizer.zero_grad(); L_old = fs.loss(*batch[:4], **kw); L_old.backward()
        g_old = phy.free_params.grad.detach().clone(); n_old = n_rb['n']
        fs._mats_ix = saved_ix                          # hoisted
        n_rb['n'] = 0
        fs.optimizer.zero_grad(); L_new = fs.loss(*batch[:4], **kw); L_new.backward()
        g_new = phy.free_params.grad.detach().clone(); n_new = n_rb['n']
    finally:
        phy._mats_from_raw = _orig_rb
    g_rel = float((g_old - g_new).abs().max() / (g_old.abs().max() + 1e-300))
    print('  rebuilds per objective: %d without the hoist (nf=%d), %d with it' % (n_old, NF, n_new))
    print('  loss bit-identical: %s (%.10e)' % (bool(torch.equal(L_old.detach(), L_new.detach())),
                                                float(L_new)))
    print('  physical gradient, worst relative change: %.3e (reduction order only)' % g_rel)
    verdict('T14', n_new < n_old and n_new <= 2
            and bool(torch.equal(L_old.detach(), L_new.detach())) and g_rel < 1e-6,
            {'rebuilds_before': n_old, 'rebuilds_after': n_new, 'grad_rel': g_rel})

    # ---- T11 synthetic lifecycle ------------------------------------------------------
    print('\nT11 synthetic lifecycle: rank loss retains the previous basis; two consecutive abort')
    g = torch.Generator().manual_seed(3)
    Jfull = torch.randn(300, 10, generator=g, dtype=torch.float64)
    Jdef = Jfull.clone(); Jdef[:, 9] = Jdef[:, 0]                   # rank 9
    plan = {'k': 0}
    def builder(vbar):
        k = plan['k']; plan['k'] += 1
        return (Jfull if plan['seq'][k] == 'full' else Jdef), 0.0
    field = lambda m: torch.randn(300, generator=g, dtype=torch.float64)
    plan['seq'] = ['full', 'def', 'full', 'def', 'def']
    lc = OBCLifecycle(field, builder, refresh='epoch', out_dtype=torch.float64, expected_rank=10, verbose=False)
    vb = lambda: torch.zeros(10, dtype=torch.float64)
    lc.maybe_refresh(0, vb); b0 = lc.basis
    lc.maybe_refresh(1, vb); retained = lc.basis is b0 and lc.history[-1].get('retained_previous')
    lc.maybe_refresh(2, vb); accepted = lc.basis is not b0 and lc.consecutive_rank_loss == 0
    lc.maybe_refresh(3, vb); retained2 = lc.history[-1].get('retained_previous')
    try:
        lc.maybe_refresh(4, vb); aborted = False
    except OBCRankLossAbort as e:
        aborted = True; msg = str(e)[:120]
    print('  refresh 1 (rank 9): retained previous = %s; refresh 2 (rank 10): accepted = %s; '
          'refresh 3 (rank 9): retained = %s; refresh 4 (rank 9): aborted = %s'
          % (bool(retained), bool(accepted), bool(retained2), aborted))
    if aborted: print('   ', msg)
    # first-build rank loss must abort immediately
    plan['k'] = 0; plan['seq'] = ['def']
    lc2 = OBCLifecycle(field, builder, out_dtype=torch.float64, expected_rank=10, verbose=False)
    try:
        lc2.maybe_refresh(0, vb); first_abort = False
    except OBCRankLossAbort:
        first_abort = True
    zero = OBCBasis.from_matrix(torch.zeros(30, 10, dtype=torch.float64), vb())
    zero_ok = zero.rank == 0 and np.isfinite(zero.report.rtol) and np.isinf(zero.report.cond)
    print('  rank loss at the FIRST build aborts: %s; rank-zero report is finite/clean: %s; '
          'rank loss counter %d' % (first_abort, zero_ok, lc.n_rank_loss))
    verdict('T11', bool(retained and accepted and retained2 and aborted and first_abort and zero_ok),
            lc.state_summary())

    # ---- T15 reference-field cache follows dtype as well as device --------------------
    print('\nT15 reference-field cache invalidates when ANN dtype changes on the same device')
    ann_live = fs.hfn.connected_blocks[fs._ann_ix]
    dtype0 = next(ann_live.parameters()).dtype
    with torch.no_grad():
        f0 = life.reference_field(ann_live)
        ann_live.to(torch.float64)
        f64 = life.reference_field(ann_live)
        ann_live.to(dtype0)
        fback = life.reference_field(ann_live)
    dtype_ok = f0.dtype == dtype0 and f64.dtype == torch.float64 and fback.dtype == dtype0
    print('  field dtypes: %s -> %s -> %s; accepted: %s'
          % (f0.dtype, f64.dtype, fback.dtype, dtype_ok))
    verdict('T15', dtype_ok, {'initial': str(f0.dtype), 'converted': str(f64.dtype),
                              'restored': str(fback.dtype)})

    # ---- summary ----------------------------------------------------------------------
    print('\n==== summary ====')
    for k, v in RESULTS.items():
        print('  %-4s %s' % (k, 'MET' if v['met'] else 'NOT MET'))
    with open(os.path.join(RESULTS_DIR, 'test_behavior.json'), 'w') as f:
        json.dump(RESULTS, f, indent=2, default=str)


if __name__ == '__main__':
    main()
