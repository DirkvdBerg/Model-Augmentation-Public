"""PS2: two-stage predictive-state opening for the added states (candidate A, D-234), and its paired control.

Deployed model, routing, zero-init output, encoder and closed-loop thesis objective are those of the probes in the parent
folder (`u_probe.setup()`, run 42 settings, U arm, n_a 2). Only TRAINING changes, and only through a second optimiser,
so the thesis optimiser sees only the thesis-loss gradient (the runs diverge once the auxiliary step has
moved the encoder's added-state rows, which the thesis loss also reads).

  S1 (predictive state; Hefny, Downey, Gordon 2015, S1A regression, "possibly non-linear", p. 4):
      a training-only head h (one tanh layer, width = the deployed ANN width, zero-init output; P_HEAD=linear for the
      linear-head ablation) predicts e = the current model's next M closed-loop output residuals (detached, unit batch
      RMS) from the encoder's x_a(k). Trains h, W^a and the x_a rows of the encoder's final NL layer (x_b rows masked).
      # THEORY-ADJACENT: M = the encoder's past output window (equal past and future horizons, as in subspace ID); not
      # a gantry quantity.
  S2 (state map; Hefny 2015 S2 regression / PSRNN 2SR init, Downey et al. 2017 Sec. 4.2): the deployed g_aug (shared
      hidden layers + added-row output) is fitted to x_a(k+1) = g_aug(x(k), u(k)) on a FROZEN snapshot of detached
      encoder pairs, Adam at its default rate 1e-3 (Kingma and Ba 2015), to a plateau on held-out training records.
  S3 (refinement; PSRNN BPTT stage): the unchanged thesis loss only.
Switches are data-driven on two held-out TRAINING records (never validation, never poles):
  # HEURISTIC: S1 -> S2 when the best calibration unexplained fraction improved < 2 % (relative) over the last 4 evals
  # (100 updates), at least 100 updates, cap P_S1_CAP (350). S2 stops when the calibration EE residual improved < 1 % over
  # 5 checks (500 steps), at least 500 steps, cap 8000; the best calibration weights are kept.
Screen gate (handoff Gate C): if S1 has not switched by update 150 and the calibration unexplained fraction is > 0.90,
the run stops there (FAIL-SCREEN).
Control (P_MODE=control): same init draw, same batch sequence, same thesis rates, no S1/S2.
Thesis rates in both arms (= the U1 control `u_u1ctl600`): added-row output 1e-3, everything else 1e-5.
Poles are saved to the history file for post-selection diagnosis only; nothing decides on them.

Usage: python ps2.py N_UPDATES TAG     env: P_SEED (0|2|3), P_MODE (cand|control), P_HEAD (mlp|linear),
       P_CAL (last2|stratified), P_DATA (cubic: the cubic-absorber test set), P_FALLBACK (1: a failed screen
       continues with plain training instead of stopping)
"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
OUT = os.path.join(HERE, 'outputs')
sys.path.insert(0, PARENT)
import n_phase1 as n1                                                          # noqa: E402
import u_probe as up                                                           # noqa: E402

EVAL = 25


def select_data():
    """P_DATA=cubic: load the cubic-absorber test set (Thesis-writeup/Data/Cubic-absorber-test, 12 records) through the
    same thesis_taf_noisy processing chain (anti-alias, trim, controller bank); only the folder changes."""
    if os.environ.get('P_DATA') == 'cubic':
        import gantry_dynamic.data as gd
        gd.THESIS_DATASETS['thesis_taf_noisy'] = os.path.join('Cubic-absorber-test', 'Coulomb-tanh-and-MSD')
N_CAL_REC = 2          # last two training records = calibration (record-level split, fixed before any run)


def sample(recs, fs, na_r, nb_r, B, rng, dt, allowed=None, pair=False):
    """n1.batch with the SAME rng calls (so the batch sequence equals the control's / u_probe's for one seed), plus the
    record index and optionally the one-sample-shifted history. allowed: restrict to these record indices (own rng)."""
    cnt = np.array([len(r[2]) for r in recs], dtype=float)
    if allowed is not None:
        m = np.zeros(len(recs)); m[allowed] = 1; cnt = cnt * m
    p = cnt / cnt.sum()
    ri = rng.choice(len(recs), size=B, p=p)
    UH, YH, UF, YF, C, UH1, YH1 = [], [], [], [], [], [], []
    for r in ri:
        u, y, st, row = recs[r]
        k = st[rng.integers(len(st))]
        UH.append(u[k - fs.nb:k + nb_r]); YH.append(y[k - fs.na:k + na_r]); UF.append(u[k:k + n1.NF])
        YF.append(y[k:k + n1.NF]); C.append(row)
        if pair:
            UH1.append(u[k + 1 - fs.nb:k + 1 + nb_r]); YH1.append(y[k + 1 - fs.na:k + 1 + na_r])
    T = lambda a: torch.as_tensor(np.ascontiguousarray(np.stack(a)), dtype=dt)   # noqa: E731
    out = [T(UH), T(YH), T(UF), T(YF), torch.as_tensor(C, dtype=torch.long), ri]
    if pair:
        out += [T(UH1), T(YH1)]
    return out


def main(n_upd, tag):
    seed = int(os.environ.get('P_SEED', 0)); mode = os.environ.get('P_MODE', 'cand')
    head_kind = os.environ.get('P_HEAD', 'mlp'); s1_cap = int(os.environ.get('P_S1_CAP', 350))
    lr, lr_a, lr_aux = 1e-5, 1e-3, 1e-3
    thr = float(os.environ.get('P_THR', 0.90))     # screen threshold; only test_ps2.py overrides it
    os.environ['U_INIT'] = 'zero'
    select_data()
    cfg, data, norm, fs, ann, others = up.setup()
    net = ann.net
    if seed:
        # another draw of the same init method, exactly as f2_open.py (F_SEED): hidden layers torch default, W^a kaiming
        torch.manual_seed(1000 + seed)
        with torch.no_grad():
            for m in net.hidden:
                if isinstance(m, torch.nn.Linear):
                    m.reset_parameters()
            for w_ in (fs.encoder.Wa_psi_u, fs.encoder.Wa_psi_y):
                torch.nn.init.kaiming_uniform_(w_)
    # thesis optimiser: identical groups and rates to u_probe's U1 control
    opt = torch.optim.Adam([{'params': others, 'lr': lr}, {'params': list(net.out_p.parameters()), 'lr': lr},
                            {'params': list(net.hidden.parameters()), 'lr': lr},
                            {'params': list(net.out_a.parameters()), 'lr': lr_a}], lr=lr, eps=cfg.adam_eps)
    recs, na_r, nb_r = n1.windows(cfg, data, fs)
    n_rec = len(recs); cal = list(range(n_rec - N_CAL_REC, n_rec))
    if os.environ.get('P_CAL', 'last2') == 'stratified':
        # D2 (RUNS.md): no 2-d state transfers to the no-multisine T class alone; hold out the last record of every
        # record class with >= 3 records instead (class = record-name prefix, e.g. TR-S5 -> S)
        cls = [os.path.basename(str(f)).split('_')[0].split('-')[1].rstrip('0123456789') for f in data.train_files]
        cal = [max(i for i in range(n_rec) if cls[i] == c) for c in sorted(set(cls)) if cls.count(c) >= 3]
    fit = [i for i in range(n_rec) if i not in cal]
    rng = np.random.default_rng(seed)
    ev = n1.batch(recs, fs, na_r, nb_r, 72, np.random.default_rng(123), cfg.dtype_pt)
    cb = sample(recs, fs, na_r, nb_r, 128, np.random.default_rng(321), cfg.dtype_pt, allowed=cal)
    M = int(cb[1].shape[1])                       # encoder past output window length (framework quantity)
    nh = cfg.n_nodes_per_layer
    if head_kind == 'linear':
        head = torch.nn.Linear(2, M * 3)
    else:
        head = torch.nn.Sequential(torch.nn.Linear(2, nh), torch.nn.Tanh(), torch.nn.Linear(nh, M * 3))
    head = head.to(cfg.dtype_pt)
    with torch.no_grad():
        last = head if head_kind == 'linear' else head[-1]
        last.weight.zero_(); last.bias.zero_()
    enc_last = fs.encoder.net[-1] if hasattr(fs.encoder, 'net') else None
    aux_p = list(head.parameters()) + [fs.encoder.Wa_psi_u, fs.encoder.Wa_psi_y]
    if enc_last is not None:
        aux_p += [enc_last.weight, enc_last.bias]
    aux_opt = torch.optim.Adam(aux_p, lr=lr_aux, eps=cfg.adam_eps)
    check = os.environ.get('P_CHECK') == '1'      # test_ps2.py: assert the opening never writes protected tensors

    def protected():
        # physical block + physical-row output + shared hidden layers (aux step) / everything but hidden+out_a (S2),
        # and the encoder's x_b path: W^b, NL hidden layers, final-layer x_b rows
        d = {k: v.detach().clone() for k, v in fs.hfn.state_dict().items() if 'out_a' not in k}
        for k, v in fs.encoder.state_dict().items():
            if k.startswith('Wa_psi'):
                continue
            if enc_last is not None and (v.data_ptr() == enc_last.weight.data_ptr() or v.data_ptr() == enc_last.bias.data_ptr()):
                d['enc.' + k] = v.detach()[:6].clone()
            else:
                d['enc.' + k] = v.detach().clone()
        return d

    def same(d0, d1, skip=()):
        bad = [k for k in d0 if not any(t in k for t in skip) and not torch.equal(d0[k], d1[k])]
        return bad
    print('\n######## PS2 [%s]: mode %s, seed %d, head %s, %d updates, batch 64, M %d, S1 cap %d; thesis rates out_a %g, '
          'rest %g; aux %g; calibration records %s' % (tag, mode, seed, head_kind, n_upd, M, s1_cap, lr_a, lr, lr_aux, cal),
          flush=True)

    def residual(xk, uf, yf, cix):
        with torch.no_grad():
            y_pred, _ = fs.simulate(xk, uf[:, :M], yf[:, :M], ctrl_ix=cix, nf=M)
            e = (yf[:, :M] - y_pred).reshape(len(xk), -1)
            return e / e.pow(2).mean().sqrt()

    def cal_unexplained():
        with torch.no_grad():
            x = fs.encoder(cb[0], cb[1]); e = residual(x, cb[2], cb[3], cb[4])
            return float((head(x[:, 6:8]) - e).pow(2).mean() / e.pow(2).mean())

    hist = {'it': [], 'loss': [], 'ucal': [], 'phase': [], 'xa_rms': [], 'eig_a': [], 'eig_f': []}

    def evaluate(it, phase, ucal=np.nan):
        with torch.no_grad():
            l = float(fs.loss(*ev[:4], ctrl_ix=ev[4], nf=n1.NF))
            x0 = fs.encoder(ev[0], ev[1])
        Ef, Ea = up.jac_eigs(fs, x0[:24], ev[2][:24, 0])
        xr = float(x0[:, 6:8].pow(2).mean().sqrt())
        for k, v in zip(hist, (it, l, ucal, phase, xr, Ea, Ef)):
            hist[k].append(v)
        np.savez(os.path.join(OUT, 'ps2_%s_hist.npz' % tag), **{k: np.array(v) for k, v in hist.items()})
        print('  [eval %4d] phase %s loss %.4e | cal unexplained %.4f | enc x_a rms %.3e | |W_out_a| %.2e' % (
            it, phase, l, ucal, xr, net.out_a.weight.norm().item()), flush=True)
        if not np.isfinite(l):
            raise SystemExit('non-finite loss, stopping')

    def fit_state_map():
        """S2: supervised one-step fit of the deployed g_aug on a frozen snapshot of detached encoder pairs."""
        srng = np.random.default_rng(1000 + seed)
        def snap(n, allowed):
            parts = [sample(recs, fs, na_r, nb_r, 1024, srng, cfg.dtype_pt, allowed=allowed, pair=True) for _ in range(n)]
            with torch.no_grad():
                xk = torch.cat([fs.encoder(p[0], p[1]) for p in parts])
                xk1 = torch.cat([fs.encoder(p[6], p[7]) for p in parts])
            return xk, torch.cat([p[2][:, 0] for p in parts]), xk1[:, 6:8]
        Xf, Uf_, Tf = snap(8, fit); Xc, Uc, Tc = snap(2, cal)
        gp = list(net.hidden.parameters()) + list(net.out_a.parameters())
        g_opt = torch.optim.Adam(gp, lr=1e-3, eps=cfg.adam_eps)
        def ee(X, U, T):
            return (fs.hfn(X, U)[1].reshape(len(X), -1)[:, 6:8] - T).pow(2).mean() / T.var(0).mean()
        with torch.no_grad():
            r0 = float(ee(Xc, Uc, Tc))
        best, best_state, checks, t0 = r0, None, [r0], time.time()
        print('  S2 start: snapshot %d fit / %d cal pairs, cal EE rel residual %.4f' % (len(Xf), len(Xc), r0), flush=True)
        g = torch.Generator().manual_seed(seed)
        for s in range(1, int(os.environ.get('P_S2_CAP', 8000)) + 1):
            ix = torch.randint(len(Xf), (512,), generator=g)
            g_opt.zero_grad(); l = ee(Xf[ix], Uf_[ix], Tf[ix]); l.backward(); g_opt.step()
            if s % 100 == 0:
                with torch.no_grad():
                    rc = float(ee(Xc, Uc, Tc)); rf = float(ee(Xf, Uf_, Tf))
                checks.append(rc)
                if rc < best:
                    best = rc; best_state = [p.detach().clone() for p in gp]
                if s % 500 == 0:
                    print('    S2 step %5d: EE rel residual fit %.4f cal %.4f (best %.4f) | %.0f s' % (
                        s, rf, rc, best, time.time() - t0), flush=True)
                if s >= min(500, int(os.environ.get("P_S2_CAP", 8000))) and len(checks) > 5 and min(checks[-5:]) > (1 - 0.01) * min(checks[:-5]):
                    break
        if best_state is not None:
            with torch.no_grad():
                for p, b in zip(gp, best_state):
                    p.copy_(b)
        print('  S2 end after %d steps: best cal EE rel residual %.4f (start %.4f)' % (s, best, r0), flush=True)
        return best, r0

    phase = 'S3' if mode == 'control' else 'S1'
    ucal = cal_unexplained() if phase == 'S1' else np.nan
    evaluate(0, phase, ucal)
    u_hist, s1_end, s2_res, t0 = [ucal] if phase == 'S1' else [], None, None, time.time()
    for it in range(1, n_upd + 1):
        uh, yh, uf, yf, cix, ri = sample(recs, fs, na_r, nb_r, 64, rng, cfg.dtype_pt)
        opt.zero_grad(); aux_opt.zero_grad()
        l_oe = fs.loss(uh, yh, uf, yf, ctrl_ix=cix, nf=n1.NF)       # the exact thesis objective
        aux_g = None
        if phase == 'S1':
            keep = torch.as_tensor(np.isin(ri, fit))                  # calibration records never train the head
            x_enc = fs.encoder(uh[keep], yh[keep])
            e = residual(x_enc.detach(), uf[keep], yf[keep], cix[keep])
            l_inn = (head(x_enc[:, 6:8]) - e).pow(2).mean()
            aux_g = torch.autograd.grad(l_inn, aux_p, allow_unused=True)
        l_oe.backward()
        opt.step()                                                    # thesis step: thesis-loss gradient only
        if aux_g is not None:
            for p, gr in zip(aux_p, aux_g):
                p.grad = None if gr is None else gr.clone()
            if enc_last is not None and enc_last.weight.grad is not None:
                enc_last.weight.grad[:6] = 0; enc_last.bias.grad[:6] = 0   # x_b rows of the encoder untouched
            if check:
                d0 = protected()
            aux_opt.step()
            if check:
                bad = same(d0, protected())
                print('  CHECK aux step %d: %s' % (it, 'OK' if not bad else 'WROTE ' + str(bad)), flush=True)
        if it <= 3 or it % 25 == 0:
            print('  upd %4d oe %.4e%s | %.2f s/upd' % (it, float(l_oe), ' inn %.4f' % float(l_inn) if aux_g is not None
                                                       else '', (time.time() - t0) / it), flush=True)
        if it % EVAL == 0 or it == n_upd:
            ucal = cal_unexplained() if phase == 'S1' else np.nan
            evaluate(it, phase, ucal)
            if phase == 'S1':
                u_hist.append(ucal)
                plateau = it >= 100 and len(u_hist) > 5 and min(u_hist[-4:]) > (1 - 0.02) * min(u_hist[:-4])
                if it == 150 and not plateau and ucal > thr:
                    print('  FAIL-SCREEN at 150: cal unexplained %.4f > 0.90' % ucal, flush=True)
                    if os.environ.get('P_FALLBACK') == '1':
                        phase = 'S3'; print('  FALLBACK: plain training from here', flush=True); continue
                    break
                if (plateau or it >= s1_cap) and ucal > thr:
                    print('  FAIL-SCREEN at S1 end (update %d): cal unexplained %.4f > 0.90' % (it, ucal), flush=True)
                    if os.environ.get('P_FALLBACK') == '1':
                        phase = 'S3'; print('  FALLBACK: plain training from here', flush=True); continue
                    break
                if plateau or it >= s1_cap:
                    s1_end = it
                    print('  S1 -> S2 at update %d (%s), cal unexplained %.4f' % (
                        it, 'plateau' if plateau else 'cap', ucal), flush=True)
                    if check:
                        d0 = protected()
                    s2_res = fit_state_map()
                    if check:
                        bad = same(d0, protected(), skip=('hidden',))
                        print('  CHECK S2: %s' % ('OK' if not bad else 'WROTE ' + str(bad)), flush=True)
                    phase = 'S3'
                    evaluate(it, 'S2end')
    torch.save({'hfn': fs.hfn.state_dict(), 'enc': fs.encoder.state_dict(), 'n_upd': it, 's1_end': s1_end,
                's2_res': s2_res, 'mode': mode, 'seed': seed, 'head': head_kind},
               os.path.join(OUT, 'ps2_%s.pt' % tag))
    print('  saved overnight-clean-method/outputs/ps2_%s.pt (S1 end %s, S2 residual %s)' % (tag, s1_end, s2_res), flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]), sys.argv[2])
