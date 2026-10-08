"""Phase-1 pre-flight of the two-phase training (user proposal, 2026-09-30).

Jan's routing is untouched: the same Static_ANN_Block, the same inputs z = [x (6), x_a (2), u (3)],
the same 8 routed rows, the same additive connection, physics block, encoder, closed loop and loss.
Only the function INSIDE the block (ann.net) is replaced, after build_model, by a composite:
  added rows    = state(z)                               (x, x_a, u -> x_a+)
  physical rows = readout(x_a) - readout(0) + gate * static(x, u)
Phase 1: gate = 0 (no static correction), physical combinations and encoder train as usual.
All three nets zero-initialised in their output layer (the model equals the baseline at start);
readout(0) is subtracted so the readout cannot act as a constant static offset.
Training: fresh thesis model (U arm, n_a 2, seed 1, detuned start, noisy data), Adam (eps 1e-16),
random windows of nf 400 from the 18 training records at the pipeline's stride, loss = fs.loss
(the exact training objective). Logged: loss, gradient norms per group, output-layer norms, and
every EVAL updates: the added self-map eigenvalues (Jacobian of state wrt x_a) and a fixed-window
eval loss. Saves the model state at the end for the memory test (o_memory_test.py).

Usage: python n_phase1.py N_UPDATES LR BATCH [tag]
"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from e_thesis_checkpoints import cfg_for                                      # noqa: E402  (sets sys.path)

NF = 400
EVAL = 25


class SplitNet(torch.nn.Module):
    def __init__(self, n_phys, n_a, n_u, width, layers, act, act_dyn=None, gate=0.0):
        super().__init__()
        from model_augmentation.utils.torch_nets import zero_init_feed_forward_nn as ff
        self.n_phys, self.n_a = n_phys, n_a
        act_dyn = act_dyn or act                       # activation of the state and readout nets
        self.state = ff(n_in=n_phys + n_a + n_u, n_out=n_a, n_nodes_per_layer=width, n_hidden_layers=layers, activation=act_dyn)
        self.readout = ff(n_in=n_a, n_out=n_phys, n_nodes_per_layer=width, n_hidden_layers=layers, activation=act_dyn)
        self.static = ff(n_in=n_phys + n_u, n_out=n_phys, n_nodes_per_layer=width, n_hidden_layers=layers, activation=act)
        self.register_buffer('gate', torch.tensor(float(gate)))

    def forward(self, z):
        p, a, u = z[:, :self.n_phys], z[:, self.n_phys:self.n_phys + self.n_a], z[:, self.n_phys + self.n_a:]
        phys = self.readout(a) - self.readout(torch.zeros_like(a[:1]))
        phys = phys + self.gate * self.static(torch.cat([p, u], 1))
        return torch.cat([phys, self.state(z)], 1)


def setup(seed_arm=('u', 2, 1)):
    from gantry_dynamic.data import load_datasets, compute_normalization
    from gantry_dynamic.model import build_model
    from gantry_dynamic.controller import build_closed_loop
    from model_augmentation.fit_systems.blocks import Static_ANN_Block
    cfg = cfg_for(*seed_arm)
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    data = load_datasets(cfg); norm = compute_normalization(cfg, data)
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    fs = build_model(cfg.hp, cfg, data, norm)
    fs.simulator = build_closed_loop(fs, norm, cfg, train_files=data.train_files,
                                     val_files=data.val_files, val_data=data.val_ckpt_data, verbose=False)
    ann = next(b for b in fs.hfn.connected_blocks if isinstance(b, Static_ANN_Block))
    assert ann.nz == 6 + cfg.nx_ann + 3 and ann.nw == 6 + cfg.nx_ann, (ann.nz, ann.nw)
    old = {id(p) for p in ann.net.parameters()}
    torch.manual_seed(cfg.seed)
    # N_DYN_ACT=linear: linear state and readout nets (static net keeps tanh); N_GATE=1: static net on
    ann.net = SplitNet(6, cfg.nx_ann, 3, cfg.n_nodes_per_layer, cfg.n_hidden_layers,
                       torch.nn.Identity if cfg.ann_activation == 'linear' else torch.nn.Tanh,
                       act_dyn=torch.nn.Identity if os.environ.get('N_DYN_ACT') == 'linear' else None,
                       gate=float(os.environ.get('N_GATE', '0'))).to(cfg.dtype_pt)
    others = [p for g in fs.optimizer.param_groups for p in g['params'] if id(p) not in old]
    return cfg, data, norm, fs, ann, others


def windows(cfg, data, fs):
    """Normalised records, valid window starts, controller rows; the history convention is checked
    against deepSI's to_hist_future_data on the first record."""
    na_r, nb_r = getattr(fs, 'na_right', 0), getattr(fs, 'nb_right', 0)
    rows = np.asarray(fs.simulator.train_ctrl_rows) if hasattr(fs.simulator, 'train_ctrl_rows') else np.zeros(len(data.train_list), int)
    recs = []
    for r, sd in enumerate(data.train_list):
        sdn = fs.norm.transform(sd)
        u, y = np.asarray(sdn.u), np.asarray(sdn.y)
        k0 = max(fs.na, fs.nb)
        starts = np.arange(k0, len(u) - NF + 1, cfg.stride)
        recs.append((u, y, starts, int(rows[r])))
        if r == 0:
            uh, yh, uf, yf = sdn.to_hist_future_data(na=fs.na, nb=fs.nb, nf=NF, na_right=na_r, nb_right=nb_r, stride=cfg.stride)[:4]
            for j in (0, 7, len(uf) - 1):
                k = starts[j]
                assert np.allclose(uh[j], u[k - fs.nb:k + nb_r]) and np.allclose(yh[j], y[k - fs.na:k + na_r]), 'history'
                assert np.allclose(uf[j], u[k:k + NF]) and np.allclose(yf[j], y[k:k + NF]), 'future'
            assert len(uf) == len(starts), (len(uf), len(starts))
    return recs, na_r, nb_r


def batch(recs, fs, na_r, nb_r, B, rng, dt):
    cnt = np.array([len(r[2]) for r in recs]); p = cnt / cnt.sum()
    ri = rng.choice(len(recs), size=B, p=p)
    UH, YH, UF, YF, C = [], [], [], [], []
    for r in ri:
        u, y, st, row = recs[r]
        k = st[rng.integers(len(st))]
        UH.append(u[k - fs.nb:k + nb_r]); YH.append(y[k - fs.na:k + na_r]); UF.append(u[k:k + NF]); YF.append(y[k:k + NF]); C.append(row)
    T = lambda a: torch.as_tensor(np.ascontiguousarray(np.stack(a)), dtype=dt)   # noqa: E731
    return T(UH), T(YH), T(UF), T(YF), torch.as_tensor(C, dtype=torch.long)


def added_eigs(fs, ann, x, u):
    """Eigenvalues of d x_a+ / d x_a of the full one-step map at states x (B, 8), inputs u."""
    f = lambda xx: fs.hfn(xx, u)[1].reshape(x.shape[0], -1)[:, 6:].sum(0)            # noqa: E731
    J = torch.autograd.functional.jacobian(f, x.clone()).permute(1, 0, 2)[:, :, 6:].detach().numpy()
    return np.array([np.linalg.eigvals(j) for j in J])


def main(n_upd, lr, B, tag):
    cfg, data, norm, fs, ann, others = setup()
    net = ann.net
    groups = {'state': list(net.state.parameters()), 'readout': list(net.readout.parameters()),
              'static': list(net.static.parameters())}
    phy = next(b for b in fs.hfn.connected_blocks if hasattr(b, 'free_params'))
    groups['phys_params'] = [phy.free_params]
    groups['encoder'] = list(fs.encoder.parameters())
    # N_LR_REST: lr of the physical combinations and encoder (default: the same lr as the nets)
    lr_rest = float(os.environ.get('N_LR_REST', lr))
    opt = torch.optim.Adam([{'params': others, 'lr': lr_rest}, {'params': list(net.parameters()), 'lr': lr}],
                           lr=lr, eps=cfg.adam_eps)
    recs, na_r, nb_r = windows(cfg, data, fs)
    rng = np.random.default_rng(0)
    ev = batch(recs, fs, na_r, nb_r, 72, np.random.default_rng(123), cfg.dtype_pt)       # fixed eval windows
    print('\n######## pre-flight [%s]: %d updates, lr nets %g, lr combinations+encoder %g, batch %d, gate %.0f, dyn act %s, nf %d; %d windows'
          % (tag, n_upd, lr, lr_rest, B, float(net.gate), os.environ.get('N_DYN_ACT', 'tanh'), NF,
             sum(len(r[2]) for r in recs)), flush=True)

    def evaluate(it):
        with torch.no_grad():
            l = float(fs.loss(*ev[:4], ctrl_ix=ev[4], nf=NF))
            x0 = fs.encoder(ev[0], ev[1])
        e = added_eigs(fs, ann, x0[:24], ev[2][:24, 0])
        rho = np.abs(e).max(1); ang = np.abs(np.angle(e[np.arange(len(e)), np.abs(e).argmax(1)]))
        cplx = np.mean(np.abs(e.imag).max(1) > 1e-6)
        w_ro = net.readout.net[-1].weight.norm().item(); w_st = net.state.net[-1].weight.norm().item()
        print('  [eval %5d] loss %.4e | added eig: rho median %.4f max %.4f, angle of max %.3f rad (%.0f Hz), complex at %.0f %% | |W_out| state %.2e readout %.2e'
              % (it, l, np.median(rho), rho.max(), np.median(ang), np.median(ang) * 4000 / (2 * np.pi), 100 * cplx, w_st, w_ro), flush=True)

    evaluate(0)
    t0 = time.time()
    for it in range(1, n_upd + 1):
        uh, yh, uf, yf, cix = batch(recs, fs, na_r, nb_r, B, rng, cfg.dtype_pt)
        opt.zero_grad()
        loss = fs.loss(uh, yh, uf, yf, ctrl_ix=cix, nf=NF)
        loss.backward()
        if it <= 3 or it % 10 == 0:
            gn = {k: float(torch.sqrt(sum((p.grad ** 2).sum() for p in v if p.grad is not None))) for k, v in groups.items()}
            print('  upd %5d loss %.4e | grad %s | %.1f s/upd' % (
                it, float(loss), ' '.join('%s %.2e' % kv for kv in gn.items()), (time.time() - t0) / it), flush=True)
        opt.step()
        if it % EVAL == 0 or it == n_upd:
            evaluate(it)
    torch.save({'hfn': fs.hfn.state_dict(), 'enc': fs.encoder.state_dict(), 'opt': opt.state_dict(),
                'n_upd': n_upd, 'lr': lr, 'batch': B}, os.path.join(HERE, 'outputs', 'n_phase1_%s.pt' % tag))
    print('  saved outputs/n_phase1_%s.pt' % tag, flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else 'run')
