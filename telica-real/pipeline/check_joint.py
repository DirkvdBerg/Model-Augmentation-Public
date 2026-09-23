"""TR-025 gates T1-T5 for the joint-estimation / OBC arms on real data. NO optimizer step.

T1 hand-written friction tangent vs torch.func.jvp (float64, real reference tuples)
T2 transition_from_free(free_params) == the block's own step, exactly
T3 stick-zone mask: cc gradient exactly 0 when every rail is in |v| < 3 v0, nonzero when sliding;
   stick-zone share of the cc gradient without the mask (closed-loop windows at init)
T4 OBC: the corrected ANN field is orthogonal to the retained basis (relative <= 1e-8), and the
   per-step correction module (hand tangent) equals the jvp version
T5 per arm (noproj, obc, obc_affine): build, basis at the G4 point (rank, singular values),
   forward + backward of the training loss on 8 windows
Reference set at stride REF_STRIDE (probe only; the server uses every sample).
Writes outputs/g6_joint/check_joint.json.
"""
import json
import os
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tr_env                                                          # noqa: E402

import numpy as np                                                     # noqa: E402
import torch                                                           # noqa: E402

os.environ.pop('TELICA_SMOKE', None)
os.environ.pop('OBC_ARM', None)
from pipeline import telica_augment as ta                              # noqa: E402
from pipeline.real_data import load_datasets_real, frozen_norm         # noqa: E402
from pipeline.telica_model import build_model_real                     # noqa: E402
from pipeline.memory import rss_mb, peak_rss_mb                        # noqa: E402
from controller.telica_bank import build_closed_loop_telica            # noqa: E402
from friction import karnopp                                           # noqa: E402
from friction.karnopp import ReducedGantryFrictionBlockCC              # noqa: E402
from model_augmentation.fit_systems.blocks import Static_ANN_Block     # noqa: E402
from params import telica_params as tp                                 # noqa: E402
tr_env.check_no_leak()

OUT = tr_env.out_dir('g6_joint')
REF_STRIDE = int(os.environ.get('REF_STRIDE', '20'))
res = {}
ok_all = True


def check(label, ok, detail=''):
    global ok_all
    ok_all &= bool(ok)
    print(f'{"PASS" if ok else "FAIL"}  {label}  {detail}', flush=True)
    res[label] = dict(PASS=bool(ok), detail=str(detail))


def arm_cfg(arm):
    base = replace(ta.make_cfg(), device='cpu', compile_mode=None, checkpoint_chunk=0, batch_size=8)
    arms = {'noproj': dict(obc=False, obc_space='tangent'),
            'obc': dict(obc=True, obc_space='tangent'),
            'obc_affine': dict(obc=True, obc_space='affine')}
    return replace(base, joint_estimation=True, physics_parameterization='reduced',
                   param_prior=False, combo_init_detune=None, param_init_detune=None, **arms[arm])


def build(arm, data, norm):
    cfg = arm_cfg(arm)
    torch.manual_seed(cfg.seed)
    fs = build_model_real(cfg.hp, cfg, data, norm, baseline_tag='_a2', obc_ref_stride=REF_STRIDE,
                          obc_expected_rank=1)            # measured, then reported (T5)
    fs.simulator = build_closed_loop_telica(fs, norm, cfg, train_files=data.train_names,
                                            val_files=data.val_names,
                                            val_data=data.val_ckpt_data, verbose=False)
    phy = next(b for b in fs.hfn.connected_blocks if isinstance(b, ReducedGantryFrictionBlockCC))
    ann = next(b for b in fs.hfn.connected_blocks if isinstance(b, Static_ANN_Block))
    return cfg, fs, phy, ann


def windows(fs, cfg, data, n=8):
    arrs = fs.make_training_data(fs.norm.transform(data.train_data), nf=cfg.nf, stride=97)
    pick = np.linspace(0, len(arrs[0]) - 1, n).astype(int)
    b = [torch.as_tensor(a[pick]) for a in arrs]
    return [t.to(cfg.dtype_pt) for t in b[:4]] + [b[4].long()]


def main():
    t0 = time.time()
    cfg0 = arm_cfg('obc')
    data = load_datasets_real(cfg0, max_records=4)
    norm = frozen_norm(cfg0)

    # ---------------- T1-T4 on the tangent OBC arm ----------------
    cfg, fs, phy, ann = build('obc', data, norm)
    ref = fs.obc_reference
    Z = ref.Z_step.to(torch.float64)
    v = ref.x_phys[:, 3:].numpy() @ np.array([[1, 1, 0], [tp.Lb / 2, -tp.Lb / 2, 0], [0, 0, 1.0]])
    slide_all = (np.abs(v) > karnopp.STICK_FACTOR * tp.V0_TANH).all(1)
    stick_all = (np.abs(v) < karnopp.STICK_FACTOR * tp.V0_TANH).all(1)
    rng = np.random.default_rng(0)
    ix = np.concatenate([rng.choice(np.where(slide_all)[0], 128),
                         rng.choice(np.where(stick_all)[0], 128),
                         rng.choice(len(Z), 256)])
    z = Z[ix]
    free = phy.free_params.detach().clone()
    dfree = torch.as_tensor(rng.standard_normal(13) * 1e-2, dtype=torch.float64)
    # T1
    xp_h, jv_h = phy.transition_and_tangent_from_free(free, dfree, z)
    xp_r, jv_r = torch.func.jvp(lambda vv: phy.transition_from_free(vv, z), (free,), (dfree,))
    e1 = float((jv_h - jv_r).abs().max() / jv_r.abs().max())
    e1p = float((xp_h - xp_r).abs().max())
    check('T1 friction tangent vs jvp (rel <= 1e-10)', e1 <= 1e-10 and e1p == 0.0,
          f'tangent rel err {e1:.2e}, primal diff {e1p:.1e}')
    # T2
    with torch.no_grad():
        xs = phy.nonlinear_function(z)
        xt = phy.transition_from_free(phy.free_params, z)
    e2 = float((xs - xt).abs().max())
    check('T2 transition_from_free == block step (exact)', e2 == 0.0, f'max diff {e2:.1e}')
    # T3 one-step mask
    def cc_grad(zz):
        phy.free_params.grad = None
        out = phy.transition_from_free(phy.free_params, zz)
        out.pow(2).sum().backward()
        return phy.free_params.grad[10:].clone()
    def cc_grad_deriv(zz):
        # TR-025 T3a amendment: the mask acts per derivative evaluation
        phy.free_params.grad = None
        mats = phy._mats_from_free(phy.free_params)
        xd = phy._deriv_with(zz[:, :6], zz[:, 6:], mats)
        xd.pow(2).sum().backward()
        return phy.free_params.grad[10:].clone()
    g_step = cc_grad(Z[np.where(stick_all)[0][:512]])
    g_stick = cc_grad_deriv(Z[np.where(stick_all)[0][:512]])
    g_slide = cc_grad(Z[np.where(slide_all)[0][:512]])
    res['T3a_one_rk4_step_grad'] = g_step.tolist()
    check('T3a cc gradient exactly 0 in the stick zone (per derivative evaluation)',
          float(g_stick.abs().max()) == 0.0,
          f'max |grad| {float(g_stick.abs().max()):.1e} (one full RK4 step: '
          f'{float(g_step.abs().max()):.1e}, stages leave the zone)')
    check('T3b cc gradient nonzero while sliding', float(g_slide.abs().min()) > 0.0,
          f'|grad| {g_slide.abs().numpy()}')
    # T3 share on closed-loop windows (with vs without the mask)
    W = windows(fs, cfg, data, n=8)
    def win_grad():
        fs.hfn.zero_grad(set_to_none=True)
        loss = fs.loss(*W[:4], ctrl_ix=W[4])
        loss.backward()
        return phy.free_params.grad[10:].clone(), float(loss)
    g_mask, L = win_grad()
    saved = karnopp.STICK_FACTOR
    karnopp.STICK_FACTOR = 0.0                                   # mask off: every sample trains cc
    g_full, _ = win_grad()
    karnopp.STICK_FACTOR = saved
    share = (g_full - g_mask).abs() / g_full.abs().clamp_min(1e-300)
    res['T3_stick_share'] = share.tolist()
    print(f'[T3] closed-loop windows (nf {cfg.nf}): cc gradient with mask {g_mask.numpy()}, '
          f'without {g_full.numpy()}; stick-zone share of the unmasked gradient per rail '
          f'{np.round(share.numpy(), 4).tolist()}', flush=True)
    # T4 orthogonality with a non-zero ANN field
    with torch.no_grad():
        for p in ann.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    life = fs.obc
    life.refresh(phy.free_params, epoch=0)
    rep = life.basis.report
    F = life.reference_field(ann).detach().to(torch.float64)
    theta = life.coefficient(ann).detach().to(torch.float64)
    J, _ = life.basis_builder(phy.free_params.detach().to(torch.float64))
    r = F - J @ theta
    U = life.basis.U.to(torch.float64)
    ortho = float(torch.linalg.vector_norm(U.T @ r) / torch.linalg.vector_norm(F))
    check('T4a corrected field orthogonal to the retained basis (<= 1e-8)', ortho <= 1e-8,
          f'|U^T r| / |F| = {ortho:.2e}; |r|/|F| = '
          f'{float(torch.linalg.vector_norm(r) / torch.linalg.vector_norm(F)):.3f}')
    corr = fs.hfn.obc_correction
    corr.set_frozen(theta, phy.free_params.detach())
    xz = torch.cat([z[:64, :6], torch.zeros(64, cfg.nx_ann, 1, dtype=z.dtype)], 1)
    uz = z[:64, 6:]
    c_h = corr.correction_from_theta(xz, uz, theta.to(cfg.dtype_pt), use_jvp=False)
    c_r = corr.correction_from_theta(xz, uz, theta.to(cfg.dtype_pt), use_jvp=True)
    e4 = float((c_h - c_r).abs().max() / c_r.abs().max().clamp_min(1e-300))
    check('T4b per-step correction: hand tangent == jvp (rel <= 1e-10)', e4 <= 1e-10,
          f'rel err {e4:.2e}')
    res['basis_obc'] = dict(rank=rep.rank, cond=rep.cond, sv=[float(x) for x in rep.singular_values])
    print('[T4] basis at G4 (tangent arm):', flush=True)
    for ln in rep.lines(list(phy.COMBO_NAMES) + list(phy.CC_NAMES)):
        print('   ' + ln, flush=True)
    del fs, phy, ann, life

    # ---------------- T5 per arm ----------------
    for arm in ('noproj', 'obc', 'obc_affine'):
        t1 = time.time()
        cfg, fs, phy, ann = build(arm, data, norm)
        W = windows(fs, cfg, data, n=8)
        fs.hfn.zero_grad(set_to_none=True)
        loss = fs.loss(*W[:4], ctrl_ix=W[4])
        loss.backward()
        g = phy.free_params.grad
        finite = bool(torch.isfinite(loss)) and g is not None and bool(torch.isfinite(g).all())
        rank = None
        if cfg.obc:
            rep = fs.obc.basis.report
            rank = rep.rank
            res[f'basis_{arm}'] = dict(rank=rep.rank, cond=rep.cond,
                                       sv=[float(x) for x in rep.singular_values])
        check(f'T5 {arm}: builds, forward + backward finite', finite,
              f'loss {float(loss):.3e}, |grad free| {float(g.norm()):.2e}, rank {rank}, '
              f'{time.time() - t1:.0f} s, rss {rss_mb():.0f} MB')
        del fs, phy, ann
    res['peak_rss_mb'] = peak_rss_mb()
    res['PASS'] = bool(ok_all)
    json.dump(res, open(os.path.join(OUT, 'check_joint.json'), 'w'), indent=1)
    print(f'TR-025 GATES {"PASS" if ok_all else "FAIL"} ({time.time() - t0:.0f} s, peak {peak_rss_mb():.0f} MB)')


if __name__ == '__main__':
    main()
