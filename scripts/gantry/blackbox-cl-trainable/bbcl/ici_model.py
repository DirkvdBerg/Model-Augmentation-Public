"""The closed-loop-trainable black box (DECISIONS.md BB2-006).

Plant model G_hat: u -> y_hat, all in the pipeline's normalised coordinates, built as
    y_hat = S(w),   w = u + Kt(y_hat)
# THEORY: Boroujeni, Meroi, Massai, Galimberti, Ferrari-Trecate, CDC 2025, Eq. (4) and Theorem 1
# (internal-controller interconnection, ICI): S strictly causal and L_p-stable, K incrementally
# stable. Kt is K1 with its integrator made leaky (BB2-006), so it is stable.

S is a cascade of LRU-type blocks, stable for every parameter value:
# THEORY: Orvieto et al., ICML 2023, "Resurrecting Recurrent Neural Networks for Long Sequences",
# Appendix A listing (p17): Lambda = exp(-exp(nu_log) + i exp(theta_log)), gamma = sqrt(1-|L|^2).

The encoder is Jan's (deepSI `default_encoder_net`), producing the S states and a correction to
the Kt state, which otherwise starts at its steady state for the last measured output.

This `hfn` plugs into the grey box's carrier (`SSE_Interconnect_Composed`) and the grey box's
closed-loop simulator unchanged, exactly as the old adapter (blackbox-closed-loop BB-003) did.
Nothing in here uses plant information.
"""
__project_origin__ = "added"

import numpy as np
import torch
from torch import nn
from scipy.signal import cont2discrete

from deepSI.fit_systems.encoders import default_encoder_net

# BB2-006 defaults (the hyperparameter search moves these).
N_BLOCKS = 2          # L
N_MODES = 8           # m complex modes per block -> 2m real states
BLOCK_WIDTH = 8       # d, dimension of z = C s and of the block output
MLP_WIDTH = 16
R_MIN, R_MAX, MAX_PHASE = 0.9, 0.999, np.pi     # HEURISTIC (BB2-006): rate and window only
OUT_SCALE = 0.1       # HEURISTIC (BB2-006): small but nonzero start of y_hat
F_LEAK_HZ = 100.0 / 6 / 10                      # HEURISTIC (BB2-006): a decade below W/6
E_KW = dict(n_hidden_layers=2, n_nodes_per_layer=16)   # Jan's encoder (msd_ndof_deepSI_encoder.py L28)


# Kt: K1 with a leaky integrator, from K1's own design constants (controller.py)
def kt_physical(ts, design_Y=0.0, f_leak_hz=F_LEAK_HZ):
    """Block-diagonal (A, B, C, D), physical m -> N, at sample time ts. Same per-channel gains as
    K1 (controller.build_cfb_at), denominator s -> s + w_leak in the integrator factor only."""
    from gantry_dynamic import controller as K
    _, gains = K.build_cfb_at(design_Y, ts)
    num, den = K.cnorm_coeffs()
    den_leak = np.polymul([1.0, 2 * np.pi * f_leak_hz], np.polymul([1., 3 * K.W], [1., 10 * K.W]))
    assert np.allclose(np.polymul([1., 0.], np.polymul([1., 3 * K.W], [1., 10 * K.W])), den)
    cfb = []
    for kj in gains:
        b, a, _ = cont2discrete((kj * num, den_leak), ts, method='bilinear')   # as build_cfb_at
        cfb.append((np.asarray(b).ravel(), np.asarray(a).ravel()))
    return K._tf_to_ss_batch(cfb)


def fold(A, B, C, D, ystd, std_u):
    """The ControllerBank folding (closed_loop.py lines 139 to 141): e_norm -> u_norm."""
    ystd, std_u = np.asarray(ystd, float).ravel(), np.asarray(std_u, float).ravel()
    return A, B * ystd[None, :], C / std_u[:, None], D / std_u[:, None] * ystd[None, :]


# S: cascade of stable LRU-type blocks
class LRUBlock(nn.Module):
    """s+ = Lambda s + gamma B v ; z = C s ; o = z + MLP(z). No direct term: o[k] from s[k]."""

    def __init__(self, d_in, m=N_MODES, d=BLOCK_WIDTH, width=MLP_WIDTH):
        super().__init__()
        u1, u2 = np.random.uniform(size=m), np.random.uniform(size=m)
        # THEORY: LRU listing p17, ring init uniform in area between r_min and r_max
        nu_log = np.log(-0.5 * np.log(u1 * (R_MAX ** 2 - R_MIN ** 2) + R_MIN ** 2))
        theta_log = np.log(MAX_PHASE * u2)
        self.nu_log = nn.Parameter(torch.tensor(nu_log))
        self.theta_log = nn.Parameter(torch.tensor(theta_log))
        lam_abs = np.exp(-np.exp(nu_log))
        self.gamma_log = nn.Parameter(torch.tensor(np.log(np.sqrt(1 - lam_abs ** 2))))
        self.B_re = nn.Parameter(torch.randn(m, d_in, dtype=torch.float64) / np.sqrt(2 * d_in))
        self.B_im = nn.Parameter(torch.randn(m, d_in, dtype=torch.float64) / np.sqrt(2 * d_in))
        self.C = nn.Parameter(torch.randn(d, 2 * m, dtype=torch.float64) / np.sqrt(2 * m))
        self.mlp = nn.Sequential(nn.Linear(d, width), nn.Tanh(), nn.Linear(width, d)).double()
        self.m, self.d = m, d

    def output(self, s):
        z = s @ self.C.T
        return z + self.mlp(z)

    def step(self, s, v):
        m = self.m
        sr, si = s[:, :m], s[:, m:]
        r = torch.exp(-torch.exp(self.nu_log))
        th = torch.exp(self.theta_log)
        g = torch.exp(self.gamma_log)
        c, sn = r * torch.cos(th), r * torch.sin(th)
        br, bi = (v @ self.B_re.T) * g, (v @ self.B_im.T) * g
        return torch.cat([c * sr - sn * si + br, sn * sr + c * si + bi], dim=1)

    def radius(self):
        return torch.exp(-torch.exp(self.nu_log))


class ICIPlant(nn.Module):
    """The `hfn`: (x, u) -> (y, x_next), x = [s_1, ..., s_L, x_Kt], normalised coordinates."""
    connected_blocks = ()

    def __init__(self, Kt_folded, sigma_w, nu=3, ny=3, n_blocks=N_BLOCKS, m=N_MODES,
                 d=BLOCK_WIDTH, width=MLP_WIDTH):
        super().__init__()
        A, B, C, D = (torch.as_tensor(np.asarray(M, float)) for M in Kt_folded)
        self.register_buffer('KA', A); self.register_buffer('KB', B)
        self.register_buffer('KC', C); self.register_buffer('KD', D)
        self.register_buffer('sigma_w', torch.as_tensor(np.asarray(sigma_w, float).ravel()))
        self.blocks = nn.ModuleList([LRUBlock(nu if b == 0 else d, m, d, width)
                                     for b in range(n_blocks)])
        self.W_out = nn.Parameter(OUT_SCALE * torch.randn(ny, d, dtype=torch.float64) / np.sqrt(d))
        self.b_out = nn.Parameter(torch.zeros(ny, dtype=torch.float64))
        self.ns = [2 * m] * n_blocks
        self.nk = A.shape[0]
        self.nx = sum(self.ns) + self.nk
        self.nu, self.ny = nu, ny

    def split(self, x):
        out, i = [], 0
        for n in self.ns:
            out.append(x[:, i:i + n]); i += n
        return out, x[:, i:]

    def _outputs(self, ss):
        return [blk.output(s) for blk, s in zip(self.blocks, ss)]

    def output_only(self, x, u=None):
        ss, _ = self.split(x)
        return self.blocks[-1].output(ss[-1]) @ self.W_out.T + self.b_out

    def forward(self, x, u):
        ss, xk = self.split(x)
        os_ = self._outputs(ss)
        y = os_[-1] @ self.W_out.T + self.b_out
        w = u + xk @ self.KC.T + y @ self.KD.T                    # w = u + Kt(y)
        xk_next = xk @ self.KA.T + y @ self.KB.T
        v = w / self.sigma_w
        nxt = [self.blocks[0].step(ss[0], v)]
        for b in range(1, len(self.blocks)):
            nxt.append(self.blocks[b].step(ss[b], os_[b - 1]))    # o_{b-1}[k] from s_{b-1}[k]
        return y, torch.cat(nxt + [xk_next], dim=1)

    def init_model(self, sys_data=None):
        return None

    def max_radius(self):
        return max(float(b.radius().max()) for b in self.blocks)


class ICIEncoder(default_encoder_net):
    """Jan's encoder (deepSI default_encoder_net, .reshape fix) for the S states; the Kt state is
    its steady state for y[k0] (the last sample of ypast, na_right = 1) plus a learned correction."""

    def __init__(self, nb, nu, na, ny, nx, n_s=None, M_ss=None, **kw):
        super().__init__(nb=nb, nu=nu, na=na, ny=ny, nx=nx, **kw)
        M = torch.as_tensor(np.asarray(M_ss, float))
        self.register_buffer('M_ss', M)
        self.register_buffer('k_scale', M.abs().sum(1).clamp_min(1e-12))
        self.n_s = n_s

    def forward(self, upast, ypast):
        net_in = torch.cat([upast.reshape(upast.shape[0], -1),
                            ypast.reshape(ypast.shape[0], -1)], dim=1)
        x = self.net(net_in)
        xk = ypast[:, -1, :] @ self.M_ss.T + x[:, self.n_s:] * self.k_scale
        return torch.cat([x[:, :self.n_s], xk], dim=1)


# build
def sigma_w_from_data(train_sdl, norm, Kt_folded):
    """std per channel of w = u_n + Kt(y_n) over the training records (data only, BB2-006)."""
    A, B, C, D = Kt_folded
    ws = []
    for sd in train_sdl:
        un = (sd.u - norm.u_mean.ravel()) / norm.std_u.ravel()
        yn = (sd.y - norm.y0.ravel()) / norm.ystd.ravel()
        xk = np.linalg.solve(np.eye(A.shape[0]) - A, B @ yn[0])     # start at steady state
        w = np.empty_like(un)
        for k in range(len(un)):
            w[k] = un[k] + C @ xk + D @ yn[k]
            xk = A @ xk + B @ yn[k]
        ws.append(w)
    return np.concatenate(ws).std(axis=0)


def build(cfg, data, norm, seed, train_files=None, attach=True, verbose=True, sigma_w=None,
          **sizes):
    """The black box on the grey box's carrier, data, norm, windows and closed loop."""
    from gantry_dynamic.controller import build_closed_loop
    from model_augmentation.fit_systems.interconnect import SSE_Interconnect_Composed

    torch.manual_seed(seed)
    np.random.seed(seed)
    Kt = fold(*kt_physical(cfg.ts_new), norm.ystd, norm.std_u)
    if sigma_w is None:
        sigma_w = sigma_w_from_data(data.train_data.sdl, norm, Kt)
    hf = ICIPlant(Kt, sigma_w, cfg.nu, cfg.ny, **sizes)
    A, B = Kt[0], Kt[1]
    M_ss = np.linalg.solve(np.eye(A.shape[0]) - A, B)
    na = nb = int(cfg.na_nb)
    fit_sys = SSE_Interconnect_Composed(interconnect=hf, na=na, nb=nb, na_right=1, nb_right=1,
                                        e_net=ICIEncoder,
                                        e_net_kwargs=dict(E_KW, n_s=sum(hf.ns), M_ss=M_ss))
    fit_sys.norm.u0 = norm.u_mean.flatten()
    fit_sys.norm.ustd = norm.std_u.flatten()
    fit_sys.norm.y0 = norm.y0
    fit_sys.norm.ystd = norm.ystd
    fit_sys.init_model(sys_data=data.train_data, device='cpu', auto_fit_norm=False,
                       optimizer_kwargs={'lr': cfg.lr, 'eps': cfg.adam_eps})
    for net in (fit_sys.encoder, fit_sys.hfn):
        net.to(cfg.dtype_pt)
    fit_sys.burn_in = cfg.burn_in
    if attach:
        fit_sys.simulator = build_closed_loop(
            fit_sys, norm, cfg,
            train_files=data.train_files if train_files is None else train_files,
            val_files=data.val_files, val_data=data.val_ckpt_data, verbose=verbose)
    if verbose:
        n_par = sum(p.numel() for mdl in (fit_sys.encoder, fit_sys.hfn) for p in mdl.parameters())
        print('[bb2] seed %d  S: %d blocks x %d modes, width %d  Kt states %d  nx %d  params %d  '
              'sigma_w %s  max|lambda| %.4f  lr %g'
              % (seed, len(hf.blocks), hf.ns[0] // 2, hf.blocks[0].d, hf.nk, hf.nx, n_par,
                 np.array2string(np.asarray(sigma_w), precision=3), hf.max_radius(), cfg.lr),
              flush=True)
    return fit_sys
