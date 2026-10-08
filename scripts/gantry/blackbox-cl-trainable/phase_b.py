"""Phase B diagnosis on the thesis data (DECISIONS.md BB2-007, BB2-009). Noisy version, K1, 4 kHz.

B1 folded K1 feedthrough; B2 plain Jan SUBNET under K1; B3 new model at init, seeds 1 to 3;
B4 random-sign perturbations; B5 Kt against the truth (CHECK only) and against K1; B6 wiring.
Writes outputs/phase_b.json.
"""
__project_origin__ = "added"

import copy
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bbcl'))
from setup import cfg_for, load, ROOT, REPO                           # noqa: E402
import ici_model as M                                                   # noqa: E402
from model_augmentation.fit_systems.closed_loop import (closed_loop_free_run_rms_batch,  # noqa: E402
                                                        ControllerBank)

torch.set_num_threads(4)
OUT = os.path.join(ROOT, 'outputs')
os.makedirs(OUT, exist_ok=True)
res = {}


def score(fit_sys, data):
    """Per-record per-axis RMS [m] and the BB-007 stability flag on the 6 validation records."""
    sim = fit_sys.simulator
    sdl = data.val_ckpt_data.sdl
    k0 = max(fit_sys.na, fit_sys.nb)
    with np.errstate(all='ignore'):
        pc, agg, pred = closed_loop_free_run_rms_batch(
            fit_sys, sdl, sim.bank, [r for (_, _, r) in sim.val_records], return_pred=True)
    stable = []
    for p, sd in zip(pred, sdl):
        y = np.asarray(sd.y)[k0:]
        ok = np.all(np.isfinite(p)) and np.max(np.abs(p - y)) <= 10 * np.max(np.abs(y))
        stable.append(bool(ok))
    return np.array(pc), stable


def perturbed(fit_sys, delta, seed):
    fs = copy.deepcopy(fit_sys)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for net in (fs.encoder, fs.hfn):
            for p in net.parameters():
                p.add_(delta * (2 * torch.randint(0, 2, p.shape, generator=g) - 1).to(p.dtype))
    return fs


def main():
    t0 = time.time()
    cfg = cfg_for('noisy', 1)
    data, norm = load(cfg)
    print('[B] data loaded %.0f s; train %d, val %d records' % (time.time() - t0,
          len(data.train_files), len(data.val_files)), flush=True)

    # B3 first (it builds the simulator used by B1 and B2)
    sigma_w = None
    b3 = []
    for seed in (1, 2, 3):
        fs = M.build(cfg, data, norm, seed, sigma_w=sigma_w, verbose=(seed == 1))
        sigma_w = fs.hfn.sigma_w.numpy()
        t = time.time()
        pc, st = score(fs, data)
        b3.append(dict(seed=seed, stable=st, rms_axis_mean=np.nanmean(pc, 0).tolist(),
                       t_val_s=time.time() - t))
        print('[B3] seed %d stable %s  per-axis mean RMS %s m  (%.0f s)'
              % (seed, st, np.array2string(np.nanmean(pc, 0), precision=3), time.time() - t), flush=True)
        if seed == 1:
            fs1 = fs
    res['B3'] = b3
    res['sigma_w'] = sigma_w.tolist()

    # B1
    bank = fs1.simulator.bank
    Dn = bank.M_error[:, :bank.nu, :].numpy()
    res['B1_folded_D_max'] = float(np.abs(Dn).max())
    res['B1_physical_D_diag'] = bank.physical_D()[0].diagonal().tolist()
    print('[B1] folded K1 feedthrough max|D| = %.3e (physical diag %s N/m)'
          % (res['B1_folded_D_max'], np.array2string(bank.physical_D()[0].diagonal().numpy(),
                                                     precision=3)), flush=True)

    # B4 perturbations of the new model
    b4 = {}
    for delta in (1e-3, 1e-2):
        n_bad = 0
        for d in range(10):
            _, st = score(perturbed(fs1, delta, 100 + d), data)
            n_bad += int(not all(st))
        b4[str(delta)] = n_bad
        print('[B4] new model, delta %g: %d / 10 draws unstable' % (delta, n_bad), flush=True)
    res['B4_new_unstable_of_10'] = b4

    # B2 plain Jan SUBNET (old adapter's classes, read-only import), nx 32, random seed 1
    sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry', 'blackbox-closed-loop', 'adapter'))
    from bb_model import BlackBoxHF, ReshapeEncoder, E_KW
    from model_augmentation.fit_systems.interconnect import SSE_Interconnect_Composed
    b2 = []
    for seed in (1, 2, 3):
        torch.manual_seed(seed); np.random.seed(seed)
        hf = BlackBoxHF(32, cfg.nu, cfg.ny)
        fj = SSE_Interconnect_Composed(interconnect=hf, na=29, nb=29, na_right=1, nb_right=1,
                                       e_net=ReshapeEncoder, e_net_kwargs=dict(E_KW))
        fj.norm.u0, fj.norm.ustd = norm.u_mean.flatten(), norm.std_u.flatten()
        fj.norm.y0, fj.norm.ystd = norm.y0, norm.ystd
        fj.init_model(sys_data=data.train_data, device='cpu', auto_fit_norm=False,
                      optimizer_kwargs={'lr': 1e-3, 'eps': 1e-8})
        for net in (fj.encoder, fj.hfn):
            net.to(torch.float64)
        fj.simulator = fs1.simulator
        pc, st = score(fj, data)
        b2.append(dict(seed=seed, stable=st))
        print('[B2] plain SUBNET seed %d stable %s' % (seed, st), flush=True)
    res['B2'] = b2

    # B5 Kt vs K1, and Kt with the frictionless truth (CHECK only)
    from scipy.signal import cont2discrete
    sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry', 'excitation-closed-loop'))
    import common as TRUTH
    from gantry_dynamic.controller import controller_ss
    ts = cfg.ts_new
    Kt = M.kt_physical(ts)
    K1 = controller_ss(0.0, ts)

    def frf(K, f):
        A, B, C, D = K
        z = np.exp(1j * 2 * np.pi * f * ts)
        return np.array([C @ np.linalg.solve(zz * np.eye(A.shape[0]) - A, B) + D for zz in z])

    f = np.linspace(50, 300, 251)
    rel = np.abs(frf(Kt, f) - frf(K1, f)).max(axis=(1, 2)) / np.abs(frf(K1, f)).max(axis=(1, 2))
    res['B5_Kt_vs_K1_rel_max_50_300'] = float(rel.max())

    def rho(K, Y):
        A, B, Cy = TRUTH.ss_stage(*TRUTH.truth_mck(Y))
        Ad, Bd, Cd, _, _ = cont2discrete((A, B, Cy, np.zeros((3, 3))), ts, method='zoh')
        Ak, Bk, Ck, Dk = K
        Acl = np.block([[Ad - Bd @ Dk @ Cd, Bd @ Ck], [-Bk @ Cd, Ak]])
        return float(np.max(np.abs(np.linalg.eigvals(Acl))))

    res['B5_rho'] = {('%+.2f' % Y): dict(Kt=rho(Kt, Y), K1=rho(K1, Y)) for Y in (-0.3, 0.0, 0.3)}
    print('[B5] |Kt - K1| / |K1| max over 50-300 Hz = %.2e; loop spectral radius with truth %s'
          % (rel.max(), json.dumps(res['B5_rho'])), flush=True)

    # B6 wiring
    hf = fs1.hfn
    Kt_f = M.fold(*Kt, norm.ystd, norm.std_u)
    kb = ControllerBank(*(np.asarray(m)[None] for m in Kt), ystd=norm.ystd, std_u=norm.std_u,
                        dtype=torch.float64)
    fold_err = max(float(torch.max(torch.abs(kb.M_state[0] - torch.cat([torch.as_tensor(Kt_f[2]),
                                                                         torch.as_tensor(Kt_f[0])])))),
                   float(torch.max(torch.abs(kb.M_error[0] - torch.cat([torch.as_tensor(Kt_f[3]),
                                                                         torch.as_tensor(Kt_f[1])])))))
    # record what the simulator feeds hfn on one validation record, then re-step outside it
    sd = data.val_ckpt_data.sdl[0]
    rec = []
    orig_forward = hf.forward

    def spy(x, u):
        rec.append((x.detach().clone(), u.detach().clone()))
        return orig_forward(x, u)
    hf.forward = spy
    k0 = 29
    n = 2000
    un = (sd.u - norm.u_mean.ravel()) / norm.std_u.ravel()
    yn = (sd.y - norm.y0.ravel()) / norm.ystd.ravel()
    from model_augmentation.fit_systems.closed_loop import closed_loop_rollout
    T = lambda a: torch.as_tensor(np.ascontiguousarray(a)[None], dtype=torch.float64)   # noqa: E731
    with torch.no_grad():
        x0 = fs1.encoder(T(un[k0 - 29:k0 + 1]), T(yn[k0 - 29:k0 + 1]))
        yp, _, _ = closed_loop_rollout(hf, hf.output_only, T(un[k0:k0 + n]), T(yn[k0:k0 + n]), x0,
                                       fs1.simulator.bank, torch.tensor([0]))
    hf.forward = orig_forward
    with torch.no_grad():
        ns = sum(hf.ns)
        ss_x, xk = x0[:, :ns], x0[:, ns:]
        xk_b = xk.clone()
        worst = 0.0
        for k in range(n):
            x = torch.cat([ss_x, xk_b], 1)
            y = hf.output_only(x)
            worst = max(worst, float(torch.max(torch.abs(y - yp[:, k]))))
            u_cl = rec[k][1]
            ukt, xk_b = kb.step(xk_b, y, kb.gather(torch.tensor([0])))
            w = u_cl + ukt
            ss, _ = hf.split(x)
            os_ = hf._outputs(ss)
            nxt = [hf.blocks[0].step(ss[0], w / hf.sigma_w)]
            for b in range(1, len(hf.blocks)):
                nxt.append(hf.blocks[b].step(ss[b], os_[b - 1]))
            ss_x = torch.cat(nxt, 1)
        # strict causality: y[k] must not depend on u[k]
        x = rec[100][0]
        y_a, _ = hf(x, rec[100][1])
        y_b, _ = hf(x, rec[100][1] + 1.0)
    res['B6'] = dict(fold_err=fold_err, restep_max_abs=worst,
                     causal_dy=float(torch.max(torch.abs(y_a - y_b))))
    print('[B6] fold err %.1e, re-step max |dy| %.1e (normalised), y[k] vs u[k] %.1e'
          % (fold_err, worst, res['B6']['causal_dy']), flush=True)

    res['wall_s'] = time.time() - t0
    with open(os.path.join(OUT, 'phase_b.json'), 'w') as fh:
        json.dump(res, fh, indent=1)
    print('[B] wrote outputs/phase_b.json (%.0f s)' % res['wall_s'], flush=True)


if __name__ == '__main__':
    main()
