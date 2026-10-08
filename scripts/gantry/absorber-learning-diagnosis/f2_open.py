"""E3 opening with an opening-phase step budget for the state map (copy of f_eeoe.py; E4 and follow-ups).

Same objective as E3 (`f_eeoe.py`, F_INN, F_EE_ROWS=added): thesis rollout loss (nf 400) + w(t) * (lam_ee * EE
on the added rows + lam_inn * INN), EE with detached encoder states (trains the model's state map only), INN =
unexplained fraction of the current model's next F_INN output errors (RMS-normalised, detached) predicted
from the encoder's x_a(k) by a zero-init training-only linear head (trains head and encoder). Thesis model,
init (zero-init MLP output), routing unchanged; model equals the baseline at start.
Changes vs E3 (optimiser and schedule only):
  w(t) = 1 up to F_HOLD updates, linear to 0 at F_END, 0 after (E3: linear from 1 at 0 to 0 at 400).
  During the opening (t <= F_END): added-row output layer at F_LR_A_OPEN, shared hidden layers at F_LR_H_OPEN,
  encoder W^a and the head at F_LR_ENC_A. After F_END: added-row output at U_LR_A, hidden at the thesis lr
  (E3's pure-OE phase settings), W^a at F_LR_ENC_A_POST.
# HEURISTIC: F_LR_A_OPEN 1e-2 and F_LR_H_OPEN 1e-3 from AB1 (EE sub-problem fitted to its linear optimum
# within 500 updates at these rates, not at E3's); hold / end lengths: no literature value.
Logged every EVAL updates: fixed-window rollout loss, added poles (as u_probe), x_a RMS, W^a change, the EE
residual relative to the batch variance of x_a(k+1) (how well the state map fits its target).

Options (env): F_START=<checkpoint in outputs/> continue from a probe checkpoint (fresh zero-init head);
F_EE_FROM=n: innovation term from update 1, EE term and the fast state-map rates only from update n+1
(order: encoder content first, state map second).

Usage: python f2_open.py N_UPDATES BATCH TAG
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
from f_eeoe import batch_pair                                                  # noqa: E402

EVAL = 25


def main(n_upd, B, tag):
    lr = 1e-5
    env = lambda k, d: float(os.environ.get(k, d))                             # noqa: E731
    lr_a, lr_a_open, lr_h_open = env('U_LR_A', 1e-3), env('F_LR_A_OPEN', 1e-2), env('F_LR_H_OPEN', 1e-3)
    lr_wa, lr_wa_post = env('F_LR_ENC_A', 1e-3), env('F_LR_ENC_A_POST', 1e-3)
    hold, end = int(env('F_HOLD', 300)), int(env('F_END', 450))
    m_inn = int(env('F_INN', 40))
    ee_from = int(env('F_EE_FROM', 0))       # EE term (and the fast state-map rates) only from this update on
    start = os.environ.get('F_START', '')    # optional probe checkpoint to continue from (outputs/<file>)
    os.environ['U_INIT'] = 'zero'
    cfg, data, norm, fs, ann, others = up.setup()
    if start:
        st = torch.load(os.path.join(HERE, 'outputs', start), weights_only=False)
        fs.hfn.load_state_dict(st['hfn']); fs.encoder.load_state_dict(st['enc'])
        print('  continuing from outputs/%s (%d updates)' % (start, st['n_upd']), flush=True)
    net = ann.net
    seed = int(env('F_SEED', 0))
    if seed:
        # another draw of the SAME init method (robustness): torch's default Linear init of the hidden layers
        # (output layers stay zero, so the model still equals the baseline at start) and kaiming_uniform_ for
        # the encoder's W^a (pre_encoder.py l.406 to 411); also another batch sequence
        torch.manual_seed(1000 + seed)
        with torch.no_grad():
            for m in net.hidden:
                if isinstance(m, torch.nn.Linear):
                    m.reset_parameters()
            for w_ in (fs.encoder.Wa_psi_u, fs.encoder.Wa_psi_y):
                torch.nn.init.kaiming_uniform_(w_)
        print('  F_SEED %d: hidden layers and W^a redrawn (same init method), batch sequence %d' % (seed, seed), flush=True)
    wa = [fs.encoder.Wa_psi_u, fs.encoder.Wa_psi_y]
    rest = [p for p in others if all(p is not w for w in wa)]
    assert len(rest) == len(others) - 2, 'W^a not found among the optimiser parameters'
    head = torch.nn.Linear(2, m_inn * 3).to(cfg.dtype_pt)
    with torch.no_grad():                                         # zero-init: starts predicting no error
        head.weight.zero_(); head.bias.zero_()
    opt = torch.optim.Adam([{'params': rest, 'lr': lr},
                            {'params': wa + list(head.parameters()), 'lr': lr_wa},
                            {'params': list(net.out_p.parameters()), 'lr': lr},
                            {'params': list(net.hidden.parameters()), 'lr': lr_h_open if not ee_from else lr},
                            {'params': list(net.out_a.parameters()), 'lr': lr_a_open if not ee_from else lr_a}],
                           lr=lr, eps=cfg.adam_eps)
    wa0 = [w.detach().clone() for w in wa]
    recs, na_r, nb_r = n1.windows(cfg, data, fs)
    rng = np.random.default_rng(seed)
    ev = n1.batch(recs, fs, na_r, nb_r, 72, np.random.default_rng(123), cfg.dtype_pt)
    print('\n######## opening probe [%s]: %d updates, batch %d; opening: out_a %g, hidden %g, W^a %g, hold %d, end %d,'
          ' F_INN %d; after: out_a %g, hidden %g, W^a %g' % (tag, n_upd, B, lr_a_open, lr_h_open, lr_wa, hold, end,
                                                          m_inn, lr_a, lr, lr_wa_post), flush=True)

    def evaluate(it):
        with torch.no_grad():
            l = float(fs.loss(*ev[:4], ctrl_ix=ev[4], nf=n1.NF))
            x0 = fs.encoder(ev[0], ev[1])
        Ef, Ea = up.jac_eigs(fs, x0[:24], ev[2][:24, 0])
        dom = Ea[np.arange(len(Ea)), np.abs(Ea).argmax(1)]
        pa = up.band_pair(Ea); ok = np.any(~np.isnan(pa[:, 0]))
        pf = up.band_pair(Ef); okf = np.any(~np.isnan(pf[:, 0]))
        print('  [eval %5d] loss %.4e | added: rho med %.3f, dom %.0f Hz, complex %.0f %%, band pair %.0f %% (f %.0f, zeta %.2f)'
              ' | full J band pair %.0f %% (f %.0f, zeta %.2f) | x_a rms %.3e | W^a change %.3f | |W_a| %.2e' % (
                  it, l, np.median(np.abs(dom)), np.median(np.abs(np.angle(dom))) * up.FS / (2 * np.pi),
                  100 * np.mean(np.abs(Ea.imag).max(1) > 1e-9), 100 * np.mean(~np.isnan(pa[:, 0])),
                  np.nanmedian(pa[:, 0]) if ok else np.nan, np.nanmedian(pa[:, 1]) if ok else np.nan,
                  100 * np.mean(~np.isnan(pf[:, 0])), np.nanmedian(pf[:, 0]) if okf else np.nan,
                  np.nanmedian(pf[:, 1]) if okf else np.nan,
                  float(x0[:, 6:8].pow(2).mean().sqrt()),
                  float(sum((w - w0).pow(2).sum() for w, w0 in zip(wa, wa0)).sqrt() / sum(w0.pow(2).sum() for w0 in wa0).sqrt()),
                  net.out_a.weight.norm().item()), flush=True)
        if not np.isfinite(l):
            raise SystemExit('non-finite loss, stopping')

    evaluate(0)
    t0 = time.time()
    lam_ee = lam_inn = None
    switched = False
    for it in range(1, n_upd + 1):
        w = 1.0 if it <= hold else max(0.0, 1 - (it - hold) / max(end - hold, 1))
        if it > end and not switched:
            opt.param_groups[1]['lr'] = lr_wa_post
            opt.param_groups[3]['lr'] = lr
            opt.param_groups[4]['lr'] = lr_a
            switched = True
            print('  opening ended at update %d: out_a %g, hidden %g, W^a %g' % (it - 1, lr_a, lr, lr_wa_post), flush=True)
        if ee_from and it == ee_from + 1 and it <= end:
            opt.param_groups[3]['lr'] = lr_h_open
            opt.param_groups[4]['lr'] = lr_a_open
            print('  EE on from update %d: out_a %g, hidden %g' % (it, lr_a_open, lr_h_open), flush=True)
        uh, yh, uf, yf, cix, uh1, yh1 = batch_pair(recs, fs, na_r, nb_r, B, rng, cfg.dtype_pt)
        opt.zero_grad()
        l_oe = fs.loss(uh, yh, uf, yf, ctrl_ix=cix, nf=n1.NF)       # the exact thesis objective (as E3)
        loss = l_oe
        if w > 0:
            x_enc = fs.encoder(uh, yh)
            with torch.no_grad():
                xk = x_enc.detach(); xk1 = fs.encoder(uh1, yh1)
                y_pred, _ = fs.simulate(xk, uf[:, :m_inn], yf[:, :m_inn], ctrl_ix=cix, nf=m_inn)
                e = (yf[:, :m_inn] - y_pred).reshape(B, -1)
                e = e / e.pow(2).mean().sqrt()
            xp = fs.hfn(xk, uf[:, 0])[1].reshape(B, -1)
            l_ee = (xk1 - xp)[:, 6:8].pow(2).mean()
            w_ee = w if it > ee_from else 0.0
            l_inn = (head(x_enc[:, 6:8]) - e).pow(2).mean()
            if lam_inn is None:
                lam_inn = float(l_oe) / max(float(l_inn), 1e-300)
            if lam_ee is None and w_ee > 0:
                lam_ee = float(l_oe) / max(float(l_ee), 1e-300)
            loss = loss + w * lam_inn * l_inn + (w_ee * lam_ee * l_ee if w_ee > 0 else 0.0)
        loss.backward()
        if it <= 3 or it % 25 == 0:
            s = ''
            if w > 0:
                s = ' ee %.4e (rel %.3f) inn unexplained %.4f w %.3f' % (
                    float(l_ee), float(l_ee) / float(xk1[:, 6:8].var(0).mean()), float(l_inn), w)
            print('  upd %5d oe %.4e%s | %.2f s/upd' % (it, float(l_oe), s, (time.time() - t0) / it), flush=True)
        opt.step()
        if it % EVAL == 0 or it == n_upd:
            evaluate(it)
    torch.save({'hfn': fs.hfn.state_dict(), 'enc': fs.encoder.state_dict(), 'n_upd': n_upd},
               os.path.join(HERE, 'outputs', 'f2_%s.pt' % tag))
    print('  saved outputs/f2_%s.pt' % tag, flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
