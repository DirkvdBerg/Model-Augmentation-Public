"""Equation-error to output-error continuation probe (objective family; thesis model and init unchanged).

Loss = fs.loss (the thesis closed-loop rollout, nf 400) + lambda(t) * EE, with
  EE = mean over the batch and the 8 state rows of (x_hat(k+1) - F(x_hat(k), u(k)))^2,
x_hat = the encoder's state estimate at window starts k and k+1 (both DETACHED: the term trains only the
model's state map F, so the encoder cannot collapse x_a to satisfy it; the encoder is trained by the
rollout loss as in the thesis). F = one step of the model (fs.hfn).
# THEORY: Shynk 1989 (IEEE ASSP Magazine), printed pp.6-9, Eqs. (3)-(5): equation error is quadratic in
# the denominator (pole) parameters for a linear model, output error is not; the classical route starts in
# equation error and finishes in output error.
# HEURISTIC: lambda(0) set so lambda * EE equals the rollout loss on the first batch; linear decay to 0 at
# 2/3 of the run; the last third is the pure thesis loss. No literature schedule for the neural case.
Optimiser as U1 (u_probe.py, U_LR_A=1e-3): added-row output layer 1e-3, everything else 1e-5. Logged every
EVAL updates: fixed-window rollout loss (nf 400), added poles (as u_probe), RMS of the encoder's x_a.

Usage: python f_eeoe.py N_UPDATES BATCH TAG
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

EVAL = 25


def batch_pair(recs, fs, na_r, nb_r, B, rng, dt):
    """n1.batch plus the encoder history one sample later (window start k+1)."""
    cnt = np.array([len(r[2]) for r in recs]); p = cnt / cnt.sum()
    ri = rng.choice(len(recs), size=B, p=p)
    UH, YH, UF, YF, C, UH1, YH1 = [], [], [], [], [], [], []
    for r in ri:
        u, y, st, row = recs[r]
        k = st[rng.integers(len(st))]
        UH.append(u[k - fs.nb:k + nb_r]); YH.append(y[k - fs.na:k + na_r])
        UF.append(u[k:k + n1.NF]); YF.append(y[k:k + n1.NF]); C.append(row)
        UH1.append(u[k + 1 - fs.nb:k + 1 + nb_r]); YH1.append(y[k + 1 - fs.na:k + 1 + na_r])
    T = lambda a: torch.as_tensor(np.ascontiguousarray(np.stack(a)), dtype=dt)   # noqa: E731
    return T(UH), T(YH), T(UF), T(YF), torch.as_tensor(C, dtype=torch.long), T(UH1), T(YH1)


def main(n_upd, B, tag):
    lr, lr_a = 1e-5, float(os.environ.get('U_LR_A', 1e-3))
    os.environ['U_INIT'] = 'zero'
    cfg, data, norm, fs, ann, others = up.setup()
    net = ann.net
    # F_LR_ENC_A: lr of the encoder's added-state map W^a (Wa_psi_u, Wa_psi_y); default = lr (E1)
    lr_wa = float(os.environ.get('F_LR_ENC_A', lr))
    wa = [fs.encoder.Wa_psi_u, fs.encoder.Wa_psi_y]
    rest = [p for p in others if all(p is not w for w in wa)]
    assert len(rest) == len(others) - 2, 'W^a not found among the optimiser parameters'
    opt = torch.optim.Adam([{'params': rest, 'lr': lr}, {'params': wa, 'lr': lr_wa},
                            {'params': list(net.hidden.parameters()) + list(net.out_p.parameters()), 'lr': lr},
                            {'params': list(net.out_a.parameters()), 'lr': lr_a}], lr=lr, eps=cfg.adam_eps)
    # F_INN=m: innovation target (training only). A linear head maps the encoder's x_hat_a(k) to the current
    # model's next m output errors e(k..k+m-1) = y - y_pred (normalised, DETACHED); its loss trains the head
    # and the encoder (W^a), so x_hat_a carries what the model currently gets wrong (as an observer's state
    # does). The head is discarded; the model is unchanged and equals the baseline at start.
    # HEURISTIC: m, the head's linear form, and its weight (matched to the rollout loss at update 1, same
    # decay as the EE term) have no literature value for this neural setting.
    m_inn = int(os.environ.get('F_INN', 0))
    head = None
    if m_inn:
        head = torch.nn.Linear(2, m_inn * 3).to(cfg.dtype_pt)
        with torch.no_grad():                                     # zero-init: starts predicting no error
            head.weight.zero_(); head.bias.zero_()
        opt.add_param_group({'params': list(head.parameters()), 'lr': lr_wa})
    lam_inn0 = None
    # F_EE_ROWS=added: equation error on the added rows only (default: all 8 rows, E1)
    ee_rows = slice(6, 8) if os.environ.get('F_EE_ROWS') == 'added' else slice(0, 8)
    print('  W^a lr %g, EE rows %s' % (lr_wa, ee_rows), flush=True)
    wa0 = [w.detach().clone() for w in wa]
    recs, na_r, nb_r = n1.windows(cfg, data, fs)
    rng = np.random.default_rng(0)
    ev = n1.batch(recs, fs, na_r, nb_r, 72, np.random.default_rng(123), cfg.dtype_pt)
    print('\n######## EE->OE probe [%s]: %d updates, batch %d, lr %g, lr added %g; EE weight decays to 0 at %d'
          % (tag, n_upd, B, lr, lr_a, int(2 * n_upd / 3)), flush=True)
    lam0 = None

    def evaluate(it):
        with torch.no_grad():
            l = float(fs.loss(*ev[:4], ctrl_ix=ev[4], nf=n1.NF))
            x0 = fs.encoder(ev[0], ev[1])
        Ef, Ea = up.jac_eigs(fs, x0[:24], ev[2][:24, 0])
        rho = np.abs(Ea).max(1)
        dom = Ea[np.arange(len(Ea)), np.abs(Ea).argmax(1)]
        pa = up.band_pair(Ea); ok = np.any(~np.isnan(pa[:, 0]))
        print('  [eval %5d] loss %.4e | added: rho med %.3f, dom %.0f Hz, complex %.0f %%, band pair %.0f %% (f %.0f, zeta %.2f)'
              ' | encoder x_a rms %.3e | W^a rel change %.3f | |W_a| %.2e' % (
                  it, l, np.median(rho), np.median(np.abs(np.angle(dom))) * up.FS / (2 * np.pi),
                  100 * np.mean(np.abs(Ea.imag).max(1) > 1e-9), 100 * np.mean(~np.isnan(pa[:, 0])),
                  np.nanmedian(pa[:, 0]) if ok else np.nan, np.nanmedian(pa[:, 1]) if ok else np.nan,
                  float(x0[:, 6:8].pow(2).mean().sqrt()),
                  float(sum((w - w0).pow(2).sum() for w, w0 in zip(wa, wa0)).sqrt() / sum(w0.pow(2).sum() for w0 in wa0).sqrt()),
                  net.out_a.weight.norm().item()), flush=True)
        if not np.isfinite(l):
            raise SystemExit('non-finite loss, stopping')

    evaluate(0)
    t0 = time.time()
    for it in range(1, n_upd + 1):
        uh, yh, uf, yf, cix, uh1, yh1 = batch_pair(recs, fs, na_r, nb_r, B, rng, cfg.dtype_pt)
        opt.zero_grad()
        l_oe = fs.loss(uh, yh, uf, yf, ctrl_ix=cix, nf=n1.NF)
        with torch.no_grad():
            xk = fs.encoder(uh, yh); xk1 = fs.encoder(uh1, yh1)
        xp = fs.hfn(xk, uf[:, 0])[1].reshape(xk.shape[0], -1)
        l_ee = (xk1 - xp)[:, ee_rows].pow(2).mean()
        if lam0 is None:
            lam0 = float(l_oe) / max(float(l_ee), 1e-300)
        lam = lam0 * max(0.0, 1 - it / (2 * n_upd / 3))
        loss = l_oe + lam * l_ee
        if head is not None:
            with torch.no_grad():
                y_pred, _ = fs.simulate(xk, uf[:, :m_inn], yf[:, :m_inn], ctrl_ix=cix, nf=m_inn)
                e_inn = (yf[:, :m_inn] - y_pred).reshape(xk.shape[0], -1)
            xa = fs.encoder(uh, yh)[:, 6:8]                       # with gradient into the encoder
            # target = innovation / its batch RMS (O(1), so Adam's lr-sized steps fit the head's scale);
            # the loss is then the unexplained fraction of the innovation energy (1 at the zero-init head)
            e_n = e_inn / e_inn.pow(2).mean().sqrt()
            l_inn = (head(xa) - e_n).pow(2).mean()
            if lam_inn0 is None:
                lam_inn0 = float(l_oe) / max(float(l_inn), 1e-300)
            loss = loss + lam_inn0 * max(0.0, 1 - it / (2 * n_upd / 3)) * l_inn
        loss.backward()
        if it <= 3 or it % 25 == 0:
            print('  upd %5d oe %.4e ee %.4e lam %.3e%s | %.2f s/upd' % (
                it, float(l_oe), float(l_ee), lam,
                (' inn unexplained %.4f' % float(l_inn))
                if head is not None else '', (time.time() - t0) / it), flush=True)
        opt.step()
        if it % EVAL == 0 or it == n_upd:
            evaluate(it)
    torch.save({'hfn': fs.hfn.state_dict(), 'enc': fs.encoder.state_dict(), 'n_upd': n_upd},
               os.path.join(HERE, 'outputs', 'f_%s.pt' % tag))
    print('  saved outputs/f_%s.pt' % tag, flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
