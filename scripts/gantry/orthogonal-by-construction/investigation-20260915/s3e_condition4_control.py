"""Stage 3e: the positive control the competition example was missing.

__project_origin__ = "added"

Stages 3c and 3d found the exact subtraction recovering the physical parameters WORSE
than no training at all, and 3d refuted the obvious explanation: making the reference set
identical to the estimation set, so that orthogonality holds exactly on the data the loss
sees (measured orthogonality ratio 3.0e-13), does not help. The per-combination breakdown
says what is actually happening:

    kb_sum, mh, m_total, J_eff, d      recovered to 3 or 4 digits
    cg1, cg2, cy, cb_sum               driven far worse than the initialization

The truth residual is a friction force `-c_f tanh(qdot / v0)`, which for small velocities
is `-(c_f / v0) qdot`, that is, extra viscous damping. Measured on the estimation set,
`rho(Phi; delta) = 0.9465`: 94.6 percent of that residual lies INSIDE the baseline
tangent span, concentrated on the damping directions. G26's Condition 4, Eq. (17),
requires the opposite. When it fails, the paper's own Eq. (21) predicts a baseline error
of `(Phi^T Phi)^{-1} Phi^T Delta`, which is exactly what is observed: the projection
forbids the learning component from representing anything in the span, so the span-aligned
part of the friction is pushed into the damping parameters.

So the competition example as built tests the method OUTSIDE its stated hypothesis. This
script adds the control that tests it INSIDE:

    delta_orth = (I - P_Phi) delta_friction

applied as an additive state increment on the physical rows, so Condition 4 holds by
construction on the estimation set. If the subtraction is sound, arm C should now recover
the ten combinations and arm A should not.

Arms, same truth, same detuned initialization, same optimizer, three seeds:
    A   no projection
    B   2025 penalty at the beta that won in 3c
    C   2026 exact subtraction, reference set = estimation set

Run:
    conda run -n GraduationProject python -u \
      scripts/gantry/orthogonal-by-construction/investigation-20260915/s3e_condition4_control.py
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
sys.stdout = op.Tee(os.path.join(OUT, "s3e_condition4_control.log"))
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
THETA_BAR = theta_init.clone()

Xd, Ud = gc.design_points(256, seed=301)
Phi_d = op.phi_stack(THETA_BAR, Lb, Xd, Ud, with_offset=True)
Qd, rd = op.proj_basis(Phi_d)

with torch.no_grad():
    base_true = gc.rk4_step(theta_true, Lb, Xd, Ud)
    fric_next = gc.rk4_step(theta_true, Lb, Xd, Ud, f_extra=friction)
    delta_raw = (fric_next - base_true)[:, :NP_].reshape(-1)
    delta_orth = delta_raw - Qd @ (Qd.T @ delta_raw)
    Yt = base_true[:, :NP_].reshape(-1) + delta_orth
    base_init = gc.rk4_step(theta_init, Lb, Xd, Ud)[:, :NP_].reshape(-1)

print("=" * 78)
print("Truth construction, and the Condition 4 check")
print("=" * 78)
print(f"  rho(Phi; delta_friction) = {op.rho(Phi_d, delta_raw):.6f}   "
      f"-> Condition 4 VIOLATED, this is the 3c and 3d truth")
print(f"  rho(Phi; delta_orth)     = {op.rho(Phi_d, delta_orth):.3e}   "
      f"-> Condition 4 SATISFIED by construction, this is the 3e truth")
print(f"  ||delta_orth|| / ||delta_friction|| = "
      f"{(torch.linalg.norm(delta_orth)/torch.linalg.norm(delta_raw)).item():.4f}")
print(f"  ||truth - baseline(theta_init)|| = "
      f"{torch.linalg.norm(Yt - base_init).item():.4e}")
print(f"  ||delta_orth||                   = "
      f"{torch.linalg.norm(delta_orth).item():.4e}")
RES["rho_friction"] = op.rho(Phi_d, delta_raw)
RES["rho_orth"] = op.rho(Phi_d, delta_orth)
sys.stdout.flush()


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
K_d, _, _ = op.pinv_report(Phi_d)


def combos_err(l):
    return ((gc.combos(theta_true * torch.exp(l), Lb) - ct).abs() / ct.abs()).detach()


def run(arm, seed, beta=0.0, iters=150):
    torch.manual_seed(seed)
    net = StaticANN(seed=seed).double()
    l = torch.nn.Parameter(torch.log(theta_init / theta_true).clone())
    opt = torch.optim.LBFGS(list(net.parameters()) + [l], max_iter=iters,
                            history_size=50, tolerance_grad=1e-14,
                            tolerance_change=1e-16, line_search_fn="strong_wolfe")

    def G(net_):
        raw = net_(Xd, Ud).reshape(-1)
        return (raw - Phi_d @ (K_d @ raw)) if arm == "C" else raw

    def closure():
        opt.zero_grad()
        pred = gc.rk4_step(theta_true * torch.exp(l), Lb, Xd, Ud)[:, :NP_].reshape(-1)
        loss = ((Yt - pred - G(net)) ** 2).sum()
        if arm == "B":
            loss = loss + beta * (Qd.T @ net(Xd, Ud).reshape(-1)).pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        pred = gc.rk4_step(theta_true * torch.exp(l), Lb, Xd, Ud)[:, :NP_].reshape(-1)
        g = G(net)
        # how well did the learning component recover the residual it was supposed to?
        rec = 1 - (torch.linalg.norm(delta_orth - g) / torch.linalg.norm(delta_orth)).item()
        return dict(fit=torch.linalg.norm(Yt - pred - g).item(),
                    ce=combos_err(l).numpy(), rho=op.rho(Phi_d, g), rec=rec)


init_err = combos_err(torch.log(theta_init / theta_true))
print("\n" + "=" * 78)
print("Arms, truth = baseline(theta_true) + delta_orth")
print("=" * 78)
print(f"  {'arm':<34}{'fit':>12}{'combo mean':>12}{'combo max':>11}"
      f"{'rho(G)':>9}{'residual recovered':>20}")
print(f"  {'init (no training)':<34}"
      f"{torch.linalg.norm(Yt - base_init).item():>12.3e}"
      f"{init_err.mean().item():>12.4f}{init_err.max().item():>11.4f}"
      f"{'-':>9}{'-':>20}")
sys.stdout.flush()

arms = [("A no projection", "A", 0.0),
        ("B 2025 penalty beta=1e2", "B", 1e2),
        ("C 2026 exact subtraction", "C", 0.0)]
rows = {}
for label, arm, beta in arms:
    outs = [run(arm, s, beta) for s in [0, 1, 2]]
    cm = np.mean([o["ce"].mean() for o in outs])
    cx = np.mean([o["ce"].max() for o in outs])
    sd = np.std([o["ce"].mean() for o in outs])
    print(f"  {label:<34}{np.mean([o['fit'] for o in outs]):>12.3e}{cm:>12.4f}"
          f"{cx:>11.4f}{np.mean([o['rho'] for o in outs]):>9.4f}"
          f"{np.mean([o['rec'] for o in outs]):>20.4f}")
    sys.stdout.flush()
    rows[label] = dict(fit=float(np.mean([o["fit"] for o in outs])),
                       combo_mean=float(cm), combo_max=float(cx), combo_sd=float(sd),
                       rho=float(np.mean([o["rho"] for o in outs])),
                       rec=float(np.mean([o["rec"] for o in outs])),
                       per_combo=[float(x) for x in
                                  np.mean([o["ce"] for o in outs], axis=0)])
RES["rows"] = rows

print("\n  per-combination relative error, mean over 3 seeds")
labs = list(rows)
print(f"  {'combination':<10}{'INIT':>10}" + "".join(f"{l.split()[0]:>14}" for l in labs))
for i, n in enumerate(gc.COMBO_NAMES):
    print(f"  {n:<10}{init_err[i].item():>10.4f}" +
          "".join(f"{rows[l]['per_combo'][i]:>14.4f}" for l in labs))
print("\n  seed spread of the combination mean error:")
for l in labs:
    print(f"    {l:<34} sd = {rows[l]['combo_sd']:.4e}")

with open(os.path.join(OUT, "s3e_tables.json"), "w") as f:
    json.dump(RES, f, indent=2)
print(f"\nwrote {os.path.join(OUT, 's3e_tables.json')}")
