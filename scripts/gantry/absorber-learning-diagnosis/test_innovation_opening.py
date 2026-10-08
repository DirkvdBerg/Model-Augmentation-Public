"""Pre-flight tests for the innovation-opening server runs (gantry_innovation_opening.py). CPU, local.

Every check asserts; the script ends with "ALL CHECKS PASSED" or raises at the first failure.
  1. RowScaledAdam == Adam at lr*factor on the scaled rows and lr elsewhere (two reference Adams), several steps.
  2. Build (per arm): the system is an InnovationOpening, Jan's ANN is untouched (net.net Sequential, last
     Linear 8 outputs, zero), and the loss with no opening configured equals SSE_Interconnect_Composed.loss.
  3. Shifted history: _shifted_history(window at k) == the encoder history built directly at k+1 (records).
  4. Baseline equality: with the opening configured, the thesis part of the loss equals the plain loss, and the
     opening terms are finite and non-zero; gradient reaches the ANN's added rows, W^a and the head.
  5. Schedule: after n_open grad-enabled evaluations the loss equals the plain loss exactly and the optimizer
     factor is 1; no-grad evaluations (validation) never advance the counter.
  6. Pickle: torch.save of the whole system, torch.load in a FRESH python process (only sys.path set), same loss.
Arms: u and obc (run 42 and run 41 settings).

Usage: python test_innovation_opening.py
"""
import importlib
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GANTRY = os.path.dirname(HERE)
for p in (os.path.abspath(os.path.join(GANTRY, '..', '..')), GANTRY, HERE):
    sys.path.insert(0, p)


def check(cond, msg):
    if not cond:
        raise AssertionError('FAIL: ' + msg)
    print('  ok  ' + msg, flush=True)


def test_row_scaled_adam():
    from innovation_opening import RowScaledAdam
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(8, 5, dtype=torch.float64)); v = torch.nn.Parameter(torch.randn(3, dtype=torch.float64))
    w1 = torch.nn.Parameter(w.detach().clone()); w2 = torch.nn.Parameter(w.detach().clone()); v2 = torch.nn.Parameter(v.detach().clone())
    opt = RowScaledAdam([w, v], lr=1e-5, eps=1e-16, scaled=[(w, slice(6, 8)), (v, None)], factor=100.0)
    ref_lo = torch.optim.Adam([w1], lr=1e-5, eps=1e-16)                 # rows 0-5 at lr
    ref_hi = torch.optim.Adam([w2, v2], lr=1e-3, eps=1e-16)            # rows 6-7 and v at lr*100
    for it in range(5):
        g = torch.randn(8, 5, dtype=torch.float64); gv = torch.randn(3, dtype=torch.float64)
        for p, gg in ((w, g), (v, gv), (w1, g), (w2, g), (v2, gv)):
            p.grad = gg.clone()
        opt.step(); ref_lo.step(); ref_hi.step()
    check(torch.allclose(w[:6], w1[:6], rtol=0, atol=1e-15), 'RowScaledAdam rows 0-5 == Adam at lr')
    check(torch.allclose(w[6:], w2[6:], rtol=0, atol=1e-13), 'RowScaledAdam rows 6-7 == Adam at lr x factor')
    check(torch.allclose(v, v2, rtol=0, atol=1e-13), 'RowScaledAdam whole tensor == Adam at lr x factor')
    opt.set_factor(1.0)
    check(opt.factor == 1.0, 'set_factor(1) switches the scaling off')


def build(arm):
    os.environ.update(THESIS_ARM=arm, THESIS_NOISE='noisy', THESIS_START='detuned', THESIS_NA='2',
                      THESIS_SEED='1', THESIS_DEVICE='cpu')
    for k in ('OBC_ARM', 'THESIS_SMOKE', 'THESIS_NITS'):
        os.environ.pop(k, None)
    mod = importlib.reload(importlib.import_module('gantry_innovation_opening'))
    cfg = mod.CFG
    from gantry_dynamic.data import load_datasets, compute_normalization
    from gantry_dynamic.controller import build_closed_loop
    from model_innovation import build_model
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    data = load_datasets(cfg); norm = compute_normalization(cfg, data)
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    fs = build_model(cfg.hp, cfg, data, norm)
    return cfg, data, norm, fs, build_closed_loop


def windows(cfg, data, fs, nf, n=24, shift=0):
    """Direct-index windows at starts k + shift (records), the n1.batch convention (checked against deepSI there)."""
    na_r, nb_r = getattr(fs, 'na_right', 0), getattr(fs, 'nb_right', 0)
    rows = np.asarray(fs.simulator.train_ctrl_rows)
    rng = np.random.default_rng(5)
    UH, YH, UF, YF, C = [], [], [], [], []
    for _ in range(n):
        r = int(rng.integers(len(data.train_list)))
        sdn = fs.norm.transform(data.train_list[r]); u, y = np.asarray(sdn.u), np.asarray(sdn.y)
        k = int(rng.integers(max(fs.na, fs.nb) + 1, len(u) - nf - 2)) + shift
        UH.append(u[k - fs.nb:k + nb_r]); YH.append(y[k - fs.na:k + na_r]); UF.append(u[k:k + nf]); YF.append(y[k:k + nf]); C.append(rows[r])
    T = lambda a: torch.as_tensor(np.ascontiguousarray(np.stack(a)), dtype=cfg.dtype_pt)    # noqa: E731
    return T(UH), T(YH), T(UF), T(YF), torch.as_tensor(C, dtype=torch.long)


def test_arm(arm):
    print('\n== arm %s ==' % arm, flush=True)
    from innovation_opening import InnovationOpening, RowScaledAdam
    from model_augmentation.fit_systems.interconnect import SSE_Interconnect_Composed
    from model_augmentation.fit_systems.blocks import Static_ANN_Block
    cfg, data, norm, fs, build_closed_loop = build(arm)
    check(isinstance(fs, InnovationOpening), 'build_model returns InnovationOpening')
    ann = next(b for b in fs.hfn.connected_blocks if isinstance(b, Static_ANN_Block))
    last = [m for m in ann.net.net if isinstance(m, torch.nn.Linear)][-1]
    check(isinstance(ann.net.net, torch.nn.Sequential) and last.out_features == 8 and float(last.weight.abs().max()) == 0.0,
          "Jan's ANN untouched (net.net Sequential, last Linear 8 outputs, zero-init)")
    fs.simulator = build_closed_loop(fs, norm, cfg, train_files=data.train_files, val_files=data.val_files,
                                     val_data=data.val_ckpt_data, verbose=False)
    nf = 400
    uh, yh, uf, yf, c = windows(cfg, data, fs, nf)
    with torch.no_grad():
        l_plain = float(SSE_Interconnect_Composed.loss(fs, uh, yh, uf, yf, ctrl_ix=c, nf=nf))
        l_sub = float(fs.loss(uh, yh, uf, yf, ctrl_ix=c, nf=nf))
    check(l_plain == l_sub, 'loss without opening == SSE_Interconnect_Composed.loss (%.6e)' % l_plain)
    # 3. shifted history
    u1, y1, _, _, _ = windows(cfg, data, fs, nf, shift=1)
    su, sy = fs._shifted_history(uh, yh, uf, yf)
    check(torch.equal(su, u1) and torch.equal(sy, y1), 'shifted history == direct history at k+1')
    # 4. configure, rebuild the optimizer exactly as the entry script does
    fs.configure_opening(5, 40, 6, cfg.nx_ann, cfg.ny, cfg.dtype_pt)
    enc = fs.encoder
    params = [p for g in fs.optimizer.param_groups for p in g['params']]
    fs.optimizer = RowScaledAdam(params + list(fs.inn_head.parameters()), lr=cfg.hp['lr'], eps=cfg.adam_eps,
                                 scaled=[(last.weight, slice(6, 8)), (last.bias, slice(6, 8)), (enc.Wa_psi_u, None),
                                         (enc.Wa_psi_y, None), (fs.inn_head.weight, None), (fs.inn_head.bias, None)],
                                 factor=100.0)
    check({float(g['eps']) for g in fs.optimizer.param_groups} == {float(cfg.adam_eps)}, 'optimizer eps == cfg.adam_eps')
    with torch.no_grad():
        l_nograd = float(fs.loss(uh, yh, uf, yf, ctrl_ix=c, nf=nf))
    check(fs._open_it == 0 and l_nograd == l_plain, 'no-grad evaluation (validation) does not advance the opening')
    fs.optimizer.zero_grad()
    L = fs.loss(uh, yh, uf, yf, ctrl_ix=c, nf=nf)
    ee, inn = fs._lam
    check(fs._open_it == 1 and np.isfinite(float(L)), 'opening evaluation 1 advances the counter, loss finite')
    check(abs(float(L) - l_plain * (1 + 0.8 * 2)) < 1e-6 * l_plain,
          'loss = thesis loss x (1 + w (1 + 1)) at evaluation 1 (terms matched to the thesis loss), w = 0.8')
    L.backward()
    gn = lambda t: float(t.grad.abs().max()) if t.grad is not None else 0.0                # noqa: E731
    check(gn(last.weight) > 0 and float(last.weight.grad[6:8].abs().max()) > 0, 'gradient reaches the ANN added rows')
    check(gn(enc.Wa_psi_u) == 0 and gn(enc.Wa_psi_y) == 0,
          'evaluation 1: no gradient on W^a (zero-init head and zero readout, by construction)')
    check(gn(fs.inn_head.weight) > 0, 'gradient reaches the innovation head')
    fs.optimizer.step()
    check(float(fs.inn_head.weight.abs().max()) > 0, 'the head is non-zero after the first step')
    fs.optimizer.zero_grad(); fs.loss(uh, yh, uf, yf, ctrl_ix=c, nf=nf).backward()
    check(gn(enc.Wa_psi_u) > 0 and gn(enc.Wa_psi_y) > 0, 'evaluation 2: gradient reaches the encoder W^a')
    fs.optimizer.step()
    # 5. run the schedule to its end
    for _ in range(3):
        fs.optimizer.zero_grad(); fs.loss(uh, yh, uf, yf, ctrl_ix=c, nf=nf).backward(); fs.optimizer.step()
    check(fs._open_it == 5 and fs.optimizer.factor == 1.0, 'after n_open evaluations: counter 5, row factor 1')
    with torch.no_grad():
        l_after_plain = float(SSE_Interconnect_Composed.loss(fs, uh, yh, uf, yf, ctrl_ix=c, nf=nf))
    fs.optimizer.zero_grad()
    l_after = float(fs.loss(uh, yh, uf, yf, ctrl_ix=c, nf=nf))
    check(l_after == l_after_plain, 'after the opening: loss == thesis loss exactly (%.6e)' % l_after)
    # 6. pickle round trip in a fresh process
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, 'fs.pt'); torch.save(fs, f)
        torch.save({'uh': uh, 'yh': yh, 'uf': uf, 'yf': yf, 'c': c}, os.path.join(td, 'b.pt'))
        code = ('import sys, torch\n'
                + ''.join('sys.path.insert(0, %r)\n' % p for p in (os.path.abspath(os.path.join(GANTRY, '..', '..')), GANTRY, HERE))
                + 'fs = torch.load(%r, weights_only=False); b = torch.load(%r, weights_only=False)\n' % (f, os.path.join(td, 'b.pt'))
                + 'with torch.no_grad(): print(repr(float(fs.loss(b["uh"], b["yh"], b["uf"], b["yf"], ctrl_ix=b["c"], nf=%d))))\n' % nf)
        sf = os.path.join(td, 'load.py'); open(sf, 'w').write(code)
        out = subprocess.run([sys.executable, sf], capture_output=True, text=True)
        check(out.returncode == 0, 'fresh-process torch.load of the whole system runs (stderr tail: %s)' % out.stderr[-300:].replace('\n', ' | '))
        with torch.no_grad():
            l_now = float(fs.loss(uh, yh, uf, yf, ctrl_ix=c, nf=nf))
        check(float(out.stdout.strip().splitlines()[-1]) == l_now, 'reloaded system gives the same loss')


if __name__ == '__main__':
    test_row_scaled_adam()
    for arm in (sys.argv[1:] or ['u', 'obc']):
        test_arm(arm)
    print('\nALL CHECKS PASSED', flush=True)
