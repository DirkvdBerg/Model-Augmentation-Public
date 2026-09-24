"""Closed-loop window harness shared by G1, G2, G3 (ET-002, ET-003).

* `PlantedHfn`: the EXACT planted model (baseline plus the dataset's absorber in its known
  realisation). One step = RK4 of `truth.Plant8`'s EOM with the MODEL's own `Ts` and `up_sample`,
  input held over the step, exactly like `Gantry_State_Block._rk4`. State in the model's layout and
  frame: physical rows `(x - x_mean)/std_x` (pipeline constants), latent rows `[da, vda] / sa`
  (training-set std of the exact absorber pair). No fitted parameter.
* `windows_cl`: non-overlapping `nf` windows from `k = na + 1`, the encoder window
  `[k - na, k]` (na_right = 1) and the future `[k, k + nf)`, physical and normalised.
* `rollout_cl`: the PRODUCTION `closed_loop_rollout` (residual form, `xc = 0`) around any hfn.
* `metrics`: window RMS, transient share, grow, per-channel RMS (D-178 / interconnect definitions).
"""
__project_origin__ = "added"

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import et_paths                                                  # noqa: E402,F401
sys.path.insert(0, os.path.join(et_paths.ET, 'encoder'))

import truth                                                     # noqa: E402
import recon                                                     # noqa: E402
from model_augmentation.fit_systems.closed_loop import closed_loop_rollout      # noqa: E402

DT = torch.float64
VAL = ['V1_standstill_Yp10', 'V2_aprbs_Ylow', 'V3_ysweep_Yp10', 'V4_lissajous_Ym10']


class PlantedHfn(torch.nn.Module):
    """Exact planted model with the Interconnect's (x, u) -> (y, x_next) and output_only(x) API."""

    def __init__(self, norm, sa, ts, up_sample, plant=None):
        super().__init__()
        p = plant or truth.plant_for(et_paths.MODE)
        self.p, self.ts, self.up = p, float(ts), int(up_sample)
        t = lambda a: torch.as_tensor(np.asarray(a, float), dtype=DT)          # noqa: E731
        self.sx = t(np.concatenate([np.asarray(norm.std_x).ravel(), sa]))
        self.xm = t(np.concatenate([np.asarray(norm.x_mean).ravel(), [0.0, 0.0]]))
        self.su = t(np.asarray(norm.std_u).ravel())
        self.um = t(np.asarray(norm.u_mean).ravel())
        self.Cn = t(norm.Cd_norm)                                   # (3, 6), y_norm = Cn x_norm
        self.P = t(recon.P_np)
        self.K4, self.C4, self.E43 = t(p._K4), t(p._C4), t(p._E43)
        self.T2M = torch.as_tensor(recon.T2M)
        self.M2T = torch.as_tensor(np.argsort(recon.T2M))
        from truth import m1, m2, mb, Jb, Jh, Lb, dd
        self.c = dict(m1=m1, m2=m2, mb=mb, Jb=Jb, Jh=Jh, Lb=Lb, d=dd,
                      ma=p.ma, mhr=p.mh_rigid, L0=p.L0)

    def M(self, Y, da):
        c = self.c
        B = Y.shape[0]
        M = torch.zeros(B, 4, 4, dtype=DT)
        off = (c['m1'] - c['m2']) * c['Lb'] / 2 - (c['mhr'] + c['ma']) * Y - c['ma'] * c['L0'] - c['ma'] * da
        M[:, 0, 0] = c['m1'] + c['m2'] + c['mb'] + c['mhr'] + c['ma']
        M[:, 0, 1] = off
        M[:, 1, 0] = off
        M[:, 1, 1] = (c['Jb'] + c['Jh'] + (c['m1'] + c['m2']) * c['Lb'] ** 2 / 4
                      + (c['mhr'] + c['ma']) * c['d'] ** 2 + c['mhr'] * Y ** 2
                      + c['ma'] * (Y + c['L0'] + da) ** 2)
        M[:, 1, 2] = M[:, 2, 1] = -(c['mhr'] + c['ma']) * c['d']
        M[:, 1, 3] = M[:, 3, 1] = -c['ma'] * c['d']
        M[:, 2, 2] = c['mhr'] + c['ma']
        M[:, 2, 3] = M[:, 3, 2] = c['ma']
        M[:, 3, 3] = c['ma']
        return M

    def deriv_truth(self, xt, u_log):
        """Plant8.deriv, batched. xt (B, 8) truth order, u_log (B, 3) logical force."""
        q, qd = xt[:, :4], xt[:, 4:]
        rhs = u_log @ self.E43.T - q @ self.K4.T - qd @ self.C4.T
        qdd = torch.linalg.solve(self.M(xt[:, 2], xt[:, 3]), rhs.unsqueeze(-1)).squeeze(-1)
        return torch.cat([qd, qdd], dim=1)

    def to_phys(self, x):
        return (x * self.sx + self.xm)[:, self.M2T]                 # model-norm -> truth-phys

    def to_model(self, xt):
        return (xt[:, self.T2M] - self.xm) / self.sx

    def output_only(self, x, u=None):
        return x[:, :6] @ self.Cn.T

    def forward(self, x, u):
        y = self.output_only(x)
        xt = self.to_phys(x)
        u_log = (u * self.su + self.um) @ self.P.T
        h = self.ts / self.up
        for _ in range(self.up):                                    # same scheme as Gantry_State_Block._rk4
            k1 = h * self.deriv_truth(xt, u_log)
            k2 = h * self.deriv_truth(xt + k1 / 2, u_log)
            k3 = h * self.deriv_truth(xt + k2 / 2, u_log)
            k4 = h * self.deriv_truth(xt + k3, u_log)
            xt = xt + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return y, self.to_model(xt)


def selfcheck_planted(ph, n=64, seed=0):
    """deriv_truth must reproduce truth.Plant8.deriv (the replay-gated EOM) to machine precision."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 8)) * np.array([0.3, 0.05, 0.3, 1e-4, 1., 1., 1., 1e-1])
    u = rng.normal(size=(n, 3)) * np.array([500., 300., 400.])
    a = ph.deriv_truth(torch.as_tensor(x, dtype=DT), torch.as_tensor(u, dtype=DT)).numpy()
    b = np.stack([ph.p.deriv(xi, ui) for xi, ui in zip(x, u)])
    return float(np.abs(a - b).max() / np.abs(b).max())


def windows_cl(name, cfg, norm, na, nf):
    """Non-overlapping closed-loop windows. Physical arrays plus normalised torch tensors."""
    from gantry_dynamic.data import load_traj
    sd = load_traj(name + '.mat', cfg)
    u, y = np.ascontiguousarray(sd.u, dtype=float), np.ascontiguousarray(sd.y, dtype=float)
    x8 = truth.exact_truth(name, cfg.mode)['x8'][:, recon.T2M]     # model order, physical
    N = len(y)
    k = np.arange(na + 1, N - nf + 1, nf)
    yw = np.stack([y[i - na:i + 1] for i in k])
    uw = np.stack([u[i - na:i + 1] for i in k])
    uf = np.stack([u[i:i + nf] for i in k])
    yf = np.stack([y[i:i + nf] for i in k])
    un = np.ascontiguousarray((uf - np.asarray(norm.u_mean).ravel()) / np.asarray(norm.std_u).ravel())
    yn = np.ascontiguousarray((yf - np.asarray(norm.y0).ravel()) / np.asarray(norm.ystd).ravel())
    yw, uw = np.ascontiguousarray(yw), np.ascontiguousarray(uw)
    return dict(name=name, k=k, yw=yw, uw=uw, x8=np.ascontiguousarray(x8[k]), Y=y[k, 2],
                uf=torch.as_tensor(un, dtype=DT), yf=torch.as_tensor(yn, dtype=DT))


def x0_norm(x_phys6, norm):
    return (x_phys6 - np.asarray(norm.x_mean).ravel()) / np.asarray(norm.std_x).ravel()


def rollout_cl(hfn, x0, w, bank, row, ystd):
    """Production closed-loop rollout, no grad. Returns e (W, nf, ny) in metres."""
    with torch.no_grad():
        ctrl = torch.full((len(x0),), int(row), dtype=torch.long)
        y_pred, _, _ = closed_loop_rollout(hfn, hfn.output_only, w['uf'], w['yf'],
                                           torch.as_tensor(x0, dtype=DT), bank, ctrl)
    return (y_pred - w['yf']).numpy() * np.asarray(ystd).ravel()


def metrics(errs, K=100):
    """Pooled over windows and channels, as cl_burnin_sweep.criterion with w_burn = 0."""
    e = np.concatenate(errs, axis=0)
    ms0 = float((e ** 2).mean())
    msK = float((e[:, K:] ** 2).mean())
    first = float(np.sqrt((e[:, 0] ** 2).mean()))
    last = float(np.sqrt((e[:, -1] ** 2).mean()))
    return dict(rms0=np.sqrt(ms0), rmsK=np.sqrt(msK),
                share=(ms0 - msK) / ms0 if ms0 > 0 else float('nan'),
                grow=last / first if first > 0 else float('nan'),
                chan0=np.sqrt((e ** 2).mean(axis=(0, 1))).tolist(),
                chanK=np.sqrt((e[:, K:] ** 2).mean(axis=(0, 1))).tolist(),
                rms_t=np.sqrt((e ** 2).mean(axis=(0, 2))).tolist(), n_win=len(e))
