"""Gate D: the PS2 rules on a genuinely nonlinear missing subsystem (REPORT.md Section 5).

True system: Jan's 3-DOF chain `Msd_ndof` (unchanged class) with a = [0, 0, a3]: the omitted third mass is a hidden
Duffing oscillator (cubic spring on q3 - q2), so the missing state transition is nonlinear in the missing state.
Baseline: Jan's ideal 2-DOF `Parameterized_MSD_State_Block` (masses 1 and 2), S-DP augmentation as
`scripts/ecc_2025/msd_ndof_interconnect_dynamic.py` (one zero-init tanh MLP writing all 6 rows; rows 4, 5 = x_a),
Jan's SSE encoder (simple_res_net), open loop (this benchmark tests the nonlinear transition, not the closed loop).
Normalisation fixed to the constants hard-coded in Jan's MSD block (u std 10, y std 1/5.8566), so the physical block
stays physically consistent on the new data.

Data (truth side only): multisine as Jan's generator (pmax 4999, crest-factor optimised, amp 10), dt 0.02, output q2,
SNR-30 noise sigma 5.2e-3. Records: A (fit: 2 periods of 10000 after one transient period), C (calibration: 1 new-
phase period; in the thesis loss, excluded from S1/S2 fitting), T (test, new phase), X (test at 1.5x amplitude).
# HEURISTIC: a3 is set so that the RMS cubic force of spring 3 is 0.5 of its RMS linear force on a noise-free linear
# pre-simulation of the A input (truth side, before any training; never used in training or selection).

Arms (same init draw and batch sequence per seed): sdp (ordinary S-DP), ps2_mlp, ps2_lin; seeds 0, 1, 2. PS2 rules
exactly as ps2.py: M = encoder past output window; S1 plateau (2 % over 4 evals of 25, >= 100, cap 350); screen at 150
(cal unexplained <= 0.90); S2 snapshot fit at Adam 1e-3 to a 1 % / 500-step plateau (cap 8000); thesis rates
identical in all arms (Adam 1e-3, Jan's deepSI default). Final model = fixed budget. Evaluation on T and X only.

Usage: python gate_d.py N_UPDATES     (writes outputs/gate_d/*.npz and prints the results table)
"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, ROOT)
OUT = os.path.join(HERE, 'outputs', 'gate_d')
os.makedirs(OUT, exist_ok=True)

import deepSI                                                                          # noqa: E402
from model_augmentation.systems.mass_spring_damper import Msd_ndof                    # noqa: E402
from model_augmentation.fit_systems.interconnect import Interconnect, SSE_Interconnect  # noqa: E402
from model_augmentation.fit_systems.blocks import (Parameterized_MSD_State_Block, Linear_Output_Block,  # noqa: E402
                                                   Static_ANN_Block)
from model_augmentation.utils.torch_nets import zero_init_feed_forward_nn             # noqa: E402
from model_augmentation.utils.utils import selection_matrix, expansion_matrix        # noqa: E402

DT, N, NF, B, EVAL = 0.02, 10000, 200, 256, 25
S1CAP = int(os.environ.get('GD_S1CAP', 350)); SCREEN = min(150, S1CAP)
THR = float(os.environ.get('GD_THR', 0.90))   # smoke tests only; 0.90 is the rule
M_, K_, C_ = [0.5, 0.4, 0.1], [100, 100, 100], [0.5, 0.5, 0.5]
SIGMA = 5.2e-3
ARMS = os.environ.get('GD_ARMS', 'sdp,ps2_mlp,ps2_lin').split(',')


def sim(a3, u, n_drop):
    s = Msd_ndof(n=3, m=M_, k=K_, c=C_, a=[0, 0, a3], dt=DT, input_ix=[0], output_ix=[1])
    d = s.apply_experiment(sys_data=deepSI.System_data(u=u), save_state=True)[n_drop:]
    d.y = d.y[:, 0]
    return d


def make_data():
    f = os.path.join(OUT, 'data.npz')
    if os.path.exists(f):
        z = np.load(f)
        return {k: deepSI.System_data(u=z[k + '_u'], y=z[k + '_y']) for k in 'ACTX'}, float(z['a3'])
    ms = lambda periods, seed: (np.random.seed(seed), deepSI.deepSI.exp_design.multisine(        # noqa: E731
        N, periods + 1, pmax=4999, n_crest_factor_optim=1) * 10)[1]
    uA, uC, uT = ms(2, 1), ms(1, 2), ms(1, 3)
    lin = sim(0.0, uA, N)
    dq = lin.x[:, 4] - lin.x[:, 2]
    a3 = 0.5 * K_[2] * np.sqrt(np.mean(dq ** 2)) / np.sqrt(np.mean(dq ** 6))
    rng = np.random.default_rng(7)
    out = {}
    for k, u in (('A', uA), ('C', uC), ('T', uT), ('X', 1.5 * uT)):
        d = sim(a3, u, N)
        out[k] = deepSI.System_data(u=d.u, y=d.y + rng.normal(0, SIGMA, d.y.shape))
    np.savez(f, a3=a3, **{k + '_u': v.u for k, v in out.items()}, **{k + '_y': v.y for k, v in out.items()})
    print('  data: a3 = %.4g (RMS cubic / linear spring-3 force 0.5 on the linear pre-simulation); y std A %.4f' % (
        a3, np.std(out['A'].y)), flush=True)
    return out, a3


def build(seed, data):
    torch.manual_seed(seed); np.random.seed(seed)
    nxd, nu, ny = 6, 1, 1
    ic = Interconnect(nxd, nu, ny, debugging=False)
    phys = Parameterized_MSD_State_Block(nz=5, nw=4, FP_type='ideal')
    C = torch.zeros(1, 4); C[0, 2] = 1.0        # y = q2 in the block's normalised coordinates (see check below)
    outb = Linear_Output_Block(C=phys.Ty @ C @ phys.Tix, D=torch.zeros(1, 1))
    ann = Static_ANN_Block(nz=nxd + nu, nw=nxd, n_nodes_per_layer=8, net=zero_init_feed_forward_nn,
                           activation=torch.nn.Tanh)
    for b in (phys, outb, ann):
        ic.add_block(b)
    ic.connect_block_signals(ann, ['x', 'u'], ['xp'])
    ic.connect_signals('x', phys, 'concat', selection_matrix(np.array([0, 1, 2, 3]), nxd))
    ic.connect_block_signals(phys, ['u'], [])
    ic.connect_signals(phys, 'xp', 'additive', expansion_matrix(np.array([0, 1, 2, 3]), nxd))
    ic.connect_signals('x', outb, 'concat', selection_matrix(np.array([0, 1, 2, 3]), nxd))
    ic.connect_block_signals(outb, ['u'], ['y'])
    fs = SSE_Interconnect(interconnect=ic, na=nxd * 2 + 1, nb=nxd * 2 + 1, e_net_kwargs={'n_nodes_per_layer': 16})
    fs.init_model(sys_data=data['A'], auto_fit_norm=False)
    fs.norm.u0, fs.norm.ustd, fs.norm.y0, fs.norm.ystd = 0.0, float(phys.Tiu[0, 0]), 0.0, 1.0 / float(phys.Ty[0, 0])
    return fs, ann


def windows(fs, sd):
    sdn = fs.norm.transform(sd)
    uh, yh, uf, yf = sdn.to_hist_future_data(na=fs.na, nb=fs.nb, nf=NF)[:4]
    T = lambda a: torch.as_tensor(a, dtype=torch.float32)                              # noqa: E731
    return T(uh), T(yh), T(uf), T(yf)


def free_run(fs, sd):
    with torch.no_grad():
        p = fs.apply_experiment(sd)
    return np.asarray(p.y).ravel() - np.asarray(sd.y).ravel()


def run(arm, seed, n_upd, data):
    fs, ann = build(seed, data)
    last = ann.net.net[-1]
    WA, WC = windows(fs, data['A']), windows(fs, data['C'])
    allw = [torch.cat([a, c]) for a, c in zip(WA, WC)]
    nA, nT = len(WA[0]), len(allw[0])
    M = int(WA[1].shape[1])
    params = list(fs.hfn.parameters()) + list(fs.encoder.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    head = None
    if arm != 'sdp':
        head = (torch.nn.Linear(2, M) if arm == 'ps2_lin' else
                torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, M)))
        hl = head if arm == 'ps2_lin' else head[-1]
        with torch.no_grad():
            hl.weight.zero_(); hl.bias.zero_()
        enc_out = [p for p in fs.encoder.parameters() if p.shape[0] == 6]        # output rows of lin + nonlin parts
        aux_p = list(head.parameters()) + enc_out
        aux_opt = torch.optim.Adam(aux_p, lr=1e-3)
    rng = np.random.default_rng(seed)
    crng = np.random.default_rng(321)
    cix = torch.as_tensor(crng.choice(len(WC[0]), 256, replace=False))

    def resid(uh, yh, uf, yf, x):
        with torch.no_grad():
            yp, _ = fs.simulate(x, uf[:, :M])
            e = (yf[:, :M] - yp).reshape(len(x), -1)
            return e / e.pow(2).mean().sqrt()

    def ucal():
        with torch.no_grad():
            b = [w[cix] for w in WC]
            x = fs.encoder(b[0], b[1]); e = resid(*b, x)
            return float((head(x[:, 4:6]) - e).pow(2).mean() / e.pow(2).mean())

    phase = 'S3' if arm == 'sdp' else 'S1'
    uh_ = [ucal()] if phase == 'S1' else []
    s1_end, s2 = None, None
    t0 = time.time()
    status = 'ok'
    for it in range(1, n_upd + 1):
        ix = torch.as_tensor(rng.integers(nT, size=B))
        uh, yh, uf, yf = [w[ix] for w in allw]
        opt.zero_grad()
        l = fs.loss(uh, yh, uf, yf, nf=NF)
        g = None
        if phase == 'S1':
            k = ix < nA                                  # S1 fits record A only (C = calibration)
            x = fs.encoder(uh[k], yh[k])
            e = resid(uh[k], yh[k], uf[k], yf[k], x.detach())
            g = torch.autograd.grad((head(x[:, 4:6]) - e).pow(2).mean(), aux_p)
        l.backward(); opt.step()
        if g is not None:
            for p, gr in zip(aux_p, g):
                p.grad = gr.clone()
            for p in enc_out:
                p.grad[:4] = 0
            aux_opt.step()
        if not np.isfinite(float(l)):
            status = 'non-finite'; break
        if phase == 'S1' and it % EVAL == 0:
            u = ucal(); uh_.append(u)
            plateau = it >= 100 and len(uh_) > 5 and min(uh_[-4:]) > 0.98 * min(uh_[:-4])
            if (it == SCREEN and not plateau and u > THR) or ((plateau or it >= S1CAP) and u > THR):
                status = 'FAIL-SCREEN %d (%.3f)' % (it, u); break
            if plateau or it >= S1CAP:
                s1_end = (it, u)
                s2 = fit_map(fs, ann, WA, WC, seed)
                phase = 'S3'
    res = {'arm': arm, 'seed': seed, 'status': status, 's1_end': s1_end, 's2': s2, 'train_loss': float(l),
           'sec': time.time() - t0}
    for key in 'TX':
        e_on = free_run(fs, data[key])
        w, b_ = last.weight.detach().clone(), last.bias.detach().clone()
        with torch.no_grad():
            last.weight[4:6] = 0; last.bias[4:6] = 0
        e_cl = free_run(fs, data[key])
        with torch.no_grad():
            last.weight.copy_(w); last.bias.copy_(b_)
        res[key + '_on'] = e_on; res[key + '_clamp'] = e_cl
    np.savez(os.path.join(OUT, '%s_s%d.npz' % (arm, seed)), **{k: np.asarray(v, dtype=object) if not isinstance(
        v, np.ndarray) else v for k, v in res.items()})
    print('  [%s seed %d] %s | S1 end %s | S2 %s | test RMS on %.4e clamp %.4e | extrap on %.4e clamp %.4e | %.0f s' % (
        arm, seed, status, s1_end, s2, rms(res['T_on']), rms(res['T_clamp']), rms(res['X_on']), rms(res['X_clamp']),
        res['sec']), flush=True)
    return res


def fit_map(fs, ann, WA, WC, seed):
    g = torch.Generator().manual_seed(seed)
    def pairs(W, n):
        ix = torch.randint(len(W[0]) - 1, (n,), generator=g)
        with torch.no_grad():
            x = fs.encoder(W[0][ix], W[1][ix])
            # window k+1 = the same record shifted by one sample: history of window ix+1
            x1 = fs.encoder(W[0][ix + 1], W[1][ix + 1])
        return x, W[2][ix, 0], x1[:, 4:6]
    Xf, Uf, Tf = pairs(WA, 8192); Xc, Uc, Tc = pairs(WC, 2048)
    gp = list(ann.parameters())
    o = torch.optim.Adam(gp, lr=1e-3)
    ee = lambda X, U, T: (fs.hfn(X, U)[1].reshape(len(X), -1)[:, 4:6] - T).pow(2).mean() / T.var(0).mean()  # noqa
    with torch.no_grad():
        r0 = float(ee(Xc, Uc, Tc))
    best, state, checks = r0, None, [r0]
    for s in range(1, 8001):
        j = torch.randint(len(Xf), (512,), generator=g)
        o.zero_grad(); ee(Xf[j], Uf[j], Tf[j]).backward(); o.step()
        if s % 100 == 0:
            with torch.no_grad():
                rc = float(ee(Xc, Uc, Tc))
            checks.append(rc)
            if rc < best:
                best, state = rc, [p.detach().clone() for p in gp]
            if s >= 500 and len(checks) > 5 and min(checks[-5:]) > 0.99 * min(checks[:-5]):
                break
    if state is not None:
        with torch.no_grad():
            for p, b in zip(gp, state):
                p.copy_(b)
    return (round(r0, 4), round(best, 4), s)


def rms(e):
    return float(np.sqrt(np.mean(np.asarray(e, dtype=float) ** 2)))


def seg_boot(e_on, e_cl, L=500, n=10000):
    a = np.asarray(e_on, float); c = np.asarray(e_cl, float); k = len(a) // L
    d = np.array([np.sqrt(np.mean(c[i * L:(i + 1) * L] ** 2)) - np.sqrt(np.mean(a[i * L:(i + 1) * L] ** 2)) for i in range(k)])
    r = np.random.default_rng(0)
    m = np.median(d[r.integers(k, size=(n, k))], axis=1)
    return np.median(d), np.percentile(m, 2.5), np.percentile(m, 97.5)


def main(n_upd):
    data, a3 = make_data()
    print('\n######## Gate D: hidden Duffing absorber (a3 %.4g), %d updates, batch %d, nf %d' % (a3, n_upd, B, NF), flush=True)
    res = {}
    seeds = [int(v) for v in os.environ.get('GD_SEEDS', '0,1,2').split(',')]
    for seed in seeds:
        for arm in ARMS:
            res[(arm, seed)] = run(arm, seed, n_upd, data)
    print('\n######## Gate D summary (free-run RMS of y, noisy test outputs; clamp = added-row output zeroed)')
    for key, name in (('T', 'test'), ('X', 'extrapolation 1.5x')):
        for arm in ARMS:
            row = []
            for seed in seeds:
                r = res[(arm, seed)]
                md, lo, hi = seg_boot(r[key + '_on'], r[key + '_clamp'])
                row.append('s%d %.4e (clamp +%.2e [%.2e, %.2e])' % (seed, rms(r[key + '_on']), md, lo, hi))
            print('  %-20s %-8s %s' % (name, arm, ' | '.join(row)), flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]))
