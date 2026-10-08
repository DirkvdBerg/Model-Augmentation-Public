"""Early Levenberg-Marquardt phase probe (optimiser family; thesis model, init, routing unchanged).

Start: a u_probe checkpoint of the zero-init thesis model (default early150: 150 Adam updates at U1 settings,
the point where Z2 found an early valley). Then N damped Gauss-Newton (Levenberg-Marquardt) steps on the
closed-loop simulation residual of a fixed batch of short windows:
    (J^T J + mu I) delta = -J^T r,   r = (y_pred - y) / sqrt(numel)  (so |r|^2 = the rollout MSE)
solved matrix-free by conjugate gradients: J p by a central finite difference of the residual (two forward
rollouts, float64), J^T u by one vector-Jacobian product. No forward-mode autodiff through the simulator.
# THEORY: Levenberg 1944 / Marquardt 1963 (damped Gauss-Newton with accept/reject and mu adaptation).
# HEURISTIC: parameters = the learned block (ANN) and the encoder's added-state map W^a only (physical
# combinations and W^b frozen, so the step cannot buy the fit by parameter drift); mu0 = the Rayleigh quotient
# of J^T J along the gradient; mu /3 on acceptance, x4 on rejection (max 4 tries); CG 20 iterations, rel. tol
# 1e-3; finite-difference step 1e-6 |theta| / |p|; batch 64 windows, nf 100 (25 ms), fixed during the phase.
Logged every step: batch loss, mu, CG iterations; added poles (as u_probe); every 5 steps the fixed 72-window
nf 400 eval loss (the U1 / E metric). Saves the final state.

Usage: python g3_lm.py START_TAG N_STEPS TAG
"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import n_phase1 as n1                                                          # noqa: E402
import u_probe as up                                                           # noqa: E402

NF_LM, B_LM = 100, 64


def main(start, n_steps, tag):
    os.environ['U_INIT'] = 'zero'
    cfg, data, norm, fs, ann, others = up.setup()
    st = torch.load(os.path.join(HERE, 'outputs', 'u_%s.pt' % start), weights_only=False)
    fs.hfn.load_state_dict(st['hfn']); fs.encoder.load_state_dict(st['enc'])
    params = list(ann.net.parameters()) + [fs.encoder.Wa_psi_u, fs.encoder.Wa_psi_y]
    for p in fs.hfn.parameters():
        p.requires_grad_(False)
    for p in fs.encoder.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    vec = lambda: torch.nn.utils.parameters_to_vector(params).detach().clone()        # noqa: E731
    setv = lambda v: torch.nn.utils.vector_to_parameters(v, params)                     # noqa: E731
    recs, na_r, nb_r = n1.windows(cfg, data, fs)
    uh, yh, uf, yf, cix = n1.batch(recs, fs, na_r, nb_r, B_LM, np.random.default_rng(11), cfg.dtype_pt)
    uf, yf = uf[:, :NF_LM], yf[:, :NF_LM]
    ev = n1.batch(recs, fs, na_r, nb_r, 72, np.random.default_rng(123), cfg.dtype_pt)

    def resid(grad=False):
        with torch.set_grad_enabled(grad):
            x0 = fs.encoder(uh, yh)
            y, _ = fs.simulate(x0, uf, yf, ctrl_ix=cix, nf=NF_LM)
            return ((y - yf) / np.sqrt(yf.numel())).reshape(-1)

    with torch.no_grad():
        chk = float(fs.loss(uh, yh, uf, yf, ctrl_ix=cix, nf=NF_LM))
    r0 = resid()
    print('\n######## LM probe [%s from %s]: %d params, %d residuals; check |r|^2 %.6e vs fs.loss %.6e'
          % (tag, start, vec().numel(), r0.numel(), float(r0 @ r0), chk), flush=True)

    def jvp(p, th):
        eps = 1e-6 * max(float(th.norm()), 1.0) / max(float(p.norm()), 1e-300)
        setv(th + eps * p); rp = resid()
        setv(th - eps * p); rm = resid()
        setv(th)
        return (rp - rm) / (2 * eps)

    def vjp(u):
        r = resid(grad=True)
        g = torch.autograd.grad(r, params, grad_outputs=u)
        return torch.cat([x.reshape(-1) for x in g]).detach()

    def evaluate(it, l, mu, ncg):
        with torch.no_grad():
            x0 = fs.encoder(ev[0], ev[1])
        Ef, Ea = up.jac_eigs(fs, x0[:24].detach(), ev[2][:24, 0])
        dom = Ea[np.arange(len(Ea)), np.abs(Ea).argmax(1)]
        pa = up.band_pair(Ea); ok = np.any(~np.isnan(pa[:, 0]))
        s = '  [step %3d] batch loss %.4e | mu %.2e cg %2d | added: rho med %.3f, dom %.0f Hz, complex %.0f %%, band pair %.0f %% (f %.0f, zeta %.2f)' % (
            it, l, mu, ncg, np.median(np.abs(dom)), np.median(np.abs(np.angle(dom))) * up.FS / (2 * np.pi),
            100 * np.mean(np.abs(Ea.imag).max(1) > 1e-9), 100 * np.mean(~np.isnan(pa[:, 0])),
            np.nanmedian(pa[:, 0]) if ok else np.nan, np.nanmedian(pa[:, 1]) if ok else np.nan)
        if it % 5 == 0:
            with torch.no_grad():
                s += ' | eval nf400 %.4e' % float(fs.loss(*ev[:4], ctrl_ix=ev[4], nf=n1.NF))
        print(s, flush=True)

    th = vec(); r = r0; l = float(r @ r); mu = None
    evaluate(0, l, float('nan'), 0)
    for it in range(1, n_steps + 1):
        t0 = time.time()
        g = vjp(r)
        if mu is None:
            Jg = jvp(g, th)
            mu = float(Jg @ Jg) / float(g @ g)
        for _ in range(4):
            # CG on (J^T J + mu I) d = -g
            d = torch.zeros_like(g); res = -g.clone(); p = res.clone(); rs = float(res @ res); ncg = 0
            for ncg in range(1, 21):
                Jp = jvp(p, th)
                Ap = vjp(Jp) + mu * p
                a = rs / float(p @ Ap)
                d = d + a * p; res = res - a * Ap
                rs_new = float(res @ res)
                if rs_new ** 0.5 < 1e-3 * float(g.norm()):
                    break
                p = res + (rs_new / rs) * p; rs = rs_new
            setv(th + d)
            r_new = resid(); l_new = float(r_new @ r_new)
            if l_new < l:
                th, r, l, mu = th + d, r_new, l_new, mu / 3
                break
            setv(th); mu *= 4
        evaluate(it, l, mu, ncg)
        print('    step %d: %.0f s' % (it, time.time() - t0), flush=True)
    torch.save({'hfn': fs.hfn.state_dict(), 'enc': fs.encoder.state_dict(), 'n_upd': st['n_upd']},
               os.path.join(HERE, 'outputs', 'g3_%s.pt' % tag))
    print('  saved outputs/g3_%s.pt' % tag, flush=True)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
