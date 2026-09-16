"""Stage 3d: is the 2026 subtraction harmful, or is a FOREIGN reference set harmful?

__project_origin__ = "added"

Stage 3c found arm C, the exact subtraction with a reference set independent of the
estimation set, recovering the ten combinations WORSE than doing nothing (mean relative
error 0.407 against an initial 0.112, seed spread 0.155). That is either

  (H-a) the subtraction itself is harmful for a rational baseline, or
  (H-b) the subtraction is harmful only when the reference set differs from the
        estimation set.

The mechanism behind (H-b): the corrected write at the estimation points is
`G_d - Phi_d eta_aux(eta)` with `eta_aux` fitted on OTHER points. The subtracted term
`Phi_d eta_aux` then lies in the baseline tangent span AT THE ESTIMATION POINTS with a
coefficient the learning component controls. That is precisely the flat direction of
G26's Example 2, handed back to the learning component. In G26 it cannot happen, because
the reference set IS the estimation set and `Phi_d = Phi_ref`, so the projected write is
exactly orthogonal to the regressor the loss sees.

This script runs the discriminating arms. Everything else, truth, detuning,
initialization, optimizer, seeds, is identical to `s3c_competition.py`.

Run:
    conda run -n GraduationProject python -u \
      scripts/gantry/orthogonal-by-construction/investigation-20260915/s3d_refset_identity.py
"""

__project_origin__ = "added"

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import obc_common as gc                                            # noqa: E402
import obc_proj as op                                              # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
sys.stdout = op.Tee(os.path.join(OUT, "s3d_refset_identity.log"))
RES = {}

theta_true, Lb = gc.nominal()
NP_ = 6
C_F = torch.tensor([6.0, 3.0, 4.0], dtype=torch.float64)
V0 = 0.05


def friction(x):
    return -C_F * torch.tanh(x[:, 3:] / V0)


DETUNE = {"cg1": 1.25, "cg2": 0.80, "cy": 1.30, "kb1": 1.10, "kb2": 1.10,
          "mh": 1.08, "mb": 0.92, "d": 1.15}
theta_init = theta_true.clone()
for k, v in DETUNE.items():
    theta_init[gc.PARAM_NAMES.index(k)] *= v

Xd, Ud = gc.design_points(256, seed=301)
with torch.no_grad():
    Xnext = gc.rk4_step(theta_true, Lb, Xd, Ud, f_extra=friction)
Yt = Xnext[:, :NP_].reshape(-1)
Xr, Ur = gc.design_points(256, seed=302)
THETA_BAR = theta_init.clone()


class StaticANN(torch.nn.Module):
    def __init__(self, nh=24, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.l1 = torch.nn.Linear(9, nh)
        self.l2 = torch.nn.Linear(nh, nh)
        self.l3 = torch.nn.Linear(nh, NP_)
        for m in (self.l1, self.l2, self.l3):
            torch.nn.init.normal_(m.weight, 0.0, 0.5, generator=g)
            torch.nn.init.zeros_(m.bias)
        torch.nn.init.zeros_(self.l3.weight)
        self.register_buffer("sx", torch.tensor([0.3, 0.02, 0.3, 0.5, 0.05, 0.5]))
        self.register_buffer("su", torch.tensor([100.0, 100.0, 100.0]))
        self.register_buffer("sw", torch.tensor([1e-6, 1e-6, 1e-6, 1e-3, 1e-3, 1e-3]))

    def forward(self, x, u):
        h = torch.tanh(self.l1(torch.cat([x / self.sx, u / self.su], dim=-1)))
        h = torch.tanh(self.l2(h))
        return self.l3(h) * self.sw


ct = gc.combos(theta_true, Lb)


def combos_err(l):
    return ((gc.combos(theta_true * torch.exp(l), Lb) - ct).abs() / ct.abs()).detach()


def run(ref, offset, seed, iters=150):
    """ref = 'same' (reference set = estimation set) or 'foreign'."""
    Xs, Us = (Xd, Ud) if ref == "same" else (Xr, Ur)
    Phi_ref = op.phi_stack(THETA_BAR, Lb, Xs, Us, with_offset=offset)
    K_ref, _, _ = op.pinv_report(Phi_ref)
    Phi_d = op.phi_stack(THETA_BAR, Lb, Xd, Ud, with_offset=offset)
    torch.manual_seed(seed)
    net = StaticANN(seed=seed).double()
    l = torch.nn.Parameter(torch.log(theta_init / theta_true).clone())
    opt = torch.optim.LBFGS(list(net.parameters()) + [l], max_iter=iters,
                            history_size=50, tolerance_grad=1e-14,
                            tolerance_change=1e-16, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        pred = gc.rk4_step(theta_true * torch.exp(l), Lb, Xd, Ud)[:, :NP_].reshape(-1)
        G = net(Xd, Ud).reshape(-1) - Phi_d @ (K_ref @ net(Xs, Us).reshape(-1))
        loss = ((Yt - pred - G) ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        pred = gc.rk4_step(theta_true * torch.exp(l), Lb, Xd, Ud)[:, :NP_].reshape(-1)
        G = net(Xd, Ud).reshape(-1) - Phi_d @ (K_ref @ net(Xs, Us).reshape(-1))
        return dict(fit=torch.linalg.norm(Yt - pred - G).item(),
                    ce=combos_err(l).numpy(),
                    rho=op.rho(Phi_d, G),
                    orth=(torch.linalg.norm(Phi_d.T @ G)
                          / torch.linalg.norm(Phi_d.T @ net(Xd, Ud).reshape(-1))).item())


init_err = combos_err(torch.log(theta_init / theta_true))
print("=" * 78)
print("Reference set identity: the discriminating arms")
print("=" * 78)
print(f"  initial combination error: mean {init_err.mean():.4f}, max {init_err.max():.4f}")
print(f"\n  {'arm':<44}{'fit':>12}{'combo mean':>12}{'combo max':>11}"
      f"{'rho(G)':>9}{'orth ratio':>12}")
print(f"  {'init (no training)':<44}{'-':>12}{init_err.mean().item():>12.4f}"
      f"{init_err.max().item():>11.4f}{'-':>9}{'-':>12}")
sys.stdout.flush()

rows = {}
for ref in ["same", "foreign"]:
    for offset in [True, False]:
        label = f"C subtraction, ref = {ref}, offset = {offset}"
        outs = [run(ref, offset, s) for s in [0, 1, 2]]
        cm = np.mean([o["ce"].mean() for o in outs])
        cx = np.mean([o["ce"].max() for o in outs])
        sd = np.std([o["ce"].mean() for o in outs])
        fit = np.mean([o["fit"] for o in outs])
        rh = np.mean([o["rho"] for o in outs])
        ortho = np.mean([o["orth"] for o in outs])
        print(f"  {label:<44}{fit:>12.3e}{cm:>12.4f}{cx:>11.4f}{rh:>9.4f}{ortho:>12.2e}")
        sys.stdout.flush()
        rows[label] = dict(fit=float(fit), combo_mean=float(cm), combo_max=float(cx),
                           combo_sd=float(sd), rho=float(rh), orth=float(ortho),
                           per_combo=[float(x) for x in
                                      np.mean([o["ce"] for o in outs], axis=0)])
RES["rows"] = rows
RES["init"] = [float(x) for x in init_err.numpy()]

print("\n  per-combination relative error, mean over 3 seeds")
labs = list(rows)
print(f"  {'combination':<10}{'INIT':>10}" +
      "".join(f"{'ref=' + l.split('ref = ')[1].replace(', offset = ', ' off='):>22}"
              for l in labs))
for i, n in enumerate(gc.COMBO_NAMES):
    print(f"  {n:<10}{init_err[i].item():>10.4f}" +
          "".join(f"{rows[l]['per_combo'][i]:>22.4f}" for l in labs))

print("\n  seed spread of the combination mean error:")
for l in labs:
    print(f"    {l:<44} sd = {rows[l]['combo_sd']:.4e}")

with open(os.path.join(OUT, "s3d_tables.json"), "w") as f:
    json.dump(RES, f, indent=2)
print(f"\nwrote {os.path.join(OUT, 's3d_tables.json')}")
