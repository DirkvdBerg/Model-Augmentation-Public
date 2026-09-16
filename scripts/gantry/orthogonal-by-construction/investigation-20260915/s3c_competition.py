"""Stage 3c: the additive competition example.

__project_origin__ = "added"

Three arms on one baseline, one truth, one initialization, one optimizer, three seeds:

  A  no projection, joint estimation of (theta, eta)
  B  the 2025-style orthogonal-projection PENALTY, beta || Pi G_eta ||^2 with
     Pi = Q Q^T from the SVD of the stacked tangent regressor, swept over beta
  C  the 2026-style exact SUBTRACTION, G_eta - Phi_ref eta_aux(eta), no weight

What is measured is the thing the method exists to protect: the recovered values of the
ten identifiable combinations against their true values, next to the data fit. A method
that fits equally well and recovers the physics better is the one that works.

Truth: the baseline at the true parameters PLUS a dissipative friction force
`F = -c_f tanh(qdot / v0)` on the logical velocity rows, which is nonlinear, stable, and
outside the baseline's span. The baseline is started from DETUNED parameters, so there is
a genuine parameter error for the learning component to compete with.

The learning component here is STATIC (no additional states) so that the comparison
isolates the projection mechanism; additional states are covered in stage 4.

Level: one-step prediction on a fixed designed point set, which is the level the
orthogonality condition is stated at. This is deliberately NOT a free-run training run.

Run:
    conda run -n GraduationProject python -u \
      scripts/gantry/orthogonal-by-construction/investigation-20260915/s3c_competition.py
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
sys.stdout = op.Tee(os.path.join(OUT, "s3c_competition.log"))
RES = {}


def hdr(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)
    sys.stdout.flush()


theta_true, Lb = gc.nominal()
NP_ = 6

# ----------------------------------------------------------------------
# Truth: baseline at theta_true plus a dissipative friction the baseline cannot represent
# ----------------------------------------------------------------------
C_F = torch.tensor([6.0, 3.0, 4.0], dtype=torch.float64)   # N and N m, logical rows
V0 = 0.05                                                  # m/s, tanh knee


def friction(x):
    return -C_F * torch.tanh(x[:, 3:] / V0)


# Detuned start: a genuine parameter error, only on identifiable directions so the
# experiment is not testing the gauge.
DETUNE = {"cg1": 1.25, "cg2": 0.80, "cy": 1.30, "kb1": 1.10, "kb2": 1.10,
          "mh": 1.08, "mb": 0.92, "d": 1.15}
theta_init = theta_true.clone()
for k, v in DETUNE.items():
    theta_init[gc.PARAM_NAMES.index(k)] *= v
print("  detuning applied to theta_init:", DETUNE)

Xd, Ud = gc.design_points(256, seed=301)
with torch.no_grad():
    Xnext = gc.rk4_step(theta_true, Lb, Xd, Ud, f_extra=friction)
    Xbase_true = gc.rk4_step(theta_true, Lb, Xd, Ud)
    delta = (Xnext - Xbase_true)[:, :NP_].reshape(-1)
    base_err = (Xnext - gc.rk4_step(theta_init, Lb, Xd, Ud))[:, :NP_].reshape(-1)
print(f"  ||truth - baseline(theta_true)||  (the residual to learn) = "
      f"{torch.linalg.norm(delta).item():.4e}")
print(f"  ||truth - baseline(theta_init)||  (total initial error)   = "
      f"{torch.linalg.norm(base_err).item():.4e}")
print(f"  ratio: the parameter error is {torch.linalg.norm(base_err).item()/torch.linalg.norm(delta).item():.1f}x "
      f"the unmodelled residual, so both components have something to do")

# Reference set for the projection, and the declared expansion point.
Xr, Ur = gc.design_points(256, seed=302)
THETA_BAR = theta_init.clone()          # declared expansion point: the initial estimate
Phi_ref = op.phi_stack(THETA_BAR, Lb, Xr, Ur, coord="log", with_offset=True)
K_ref, r_ref, S_ref = op.pinv_report(Phi_ref, label="Phi_ref at theta_bar = theta_init")
Qpi, _ = op.proj_basis(Phi_ref)
Phi_d = op.phi_stack(THETA_BAR, Lb, Xd, Ud, coord="log", with_offset=True)
Qpi_d, _ = op.proj_basis(Phi_d)

# Check Condition 4 of G26 on this data: is the true residual orthogonal to the regressor?
c4 = torch.linalg.norm(Phi_d.T @ delta) / (torch.linalg.norm(Phi_d) * torch.linalg.norm(delta))
print(f"  G26 Condition 4 check on this data set: "
      f"||Phi^T delta|| / (||Phi|| ||delta||) = {c4.item():.4e}")
print(f"  rho(Phi_d; delta) = {op.rho(Phi_d, delta):.4f}  "
      f"-> the friction is {op.rho(Phi_d, delta)*100:.1f} percent absorbable by the "
      f"baseline tangent set, so Condition 4 does NOT hold and Theorem 7 does not apply")
RES["cond4_rho"] = op.rho(Phi_d, delta)


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
        torch.nn.init.zeros_(self.l3.weight)         # zero-output initialization
        self.register_buffer("sx", torch.tensor([0.3, 0.02, 0.3, 0.5, 0.05, 0.5]))
        self.register_buffer("su", torch.tensor([100.0, 100.0, 100.0]))
        self.register_buffer("sw", torch.tensor([1e-6, 1e-6, 1e-6, 1e-3, 1e-3, 1e-3]))

    def forward(self, x, u):
        h = torch.tanh(self.l1(torch.cat([x / self.sx, u / self.su], dim=-1)))
        h = torch.tanh(self.l2(h))
        return self.l3(h) * self.sw


def combos_err(th_log):
    c = gc.combos(theta_true * torch.exp(th_log), Lb)
    ct = gc.combos(theta_true, Lb)
    return ((c - ct).abs() / ct.abs()).detach()


def run_arm(arm, seed, beta=0.0, iters=150):
    torch.manual_seed(seed)
    net = StaticANN(seed=seed).double()
    l = torch.nn.Parameter(torch.log(theta_init / theta_true).clone())
    params = list(net.parameters()) + [l]
    opt = torch.optim.LBFGS(params, max_iter=iters, history_size=50,
                            tolerance_grad=1e-14, tolerance_change=1e-16,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        th = theta_true * torch.exp(l)
        pred = gc.rk4_step(th, Lb, Xd, Ud)[:, :NP_].reshape(-1)
        G_d = net(Xd, Ud).reshape(-1)
        if arm == "C":
            G_r = net(Xr, Ur).reshape(-1)
            G_d = G_d - Phi_d @ (K_ref @ G_r)
        loss = ((Xnext[:, :NP_].reshape(-1) - pred - G_d) ** 2).sum()
        if arm == "B":
            loss = loss + beta * (Qpi_d.T @ net(Xd, Ud).reshape(-1)).pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        th = theta_true * torch.exp(l)
        pred = gc.rk4_step(th, Lb, Xd, Ud)[:, :NP_].reshape(-1)
        G_d = net(Xd, Ud).reshape(-1)
        Gd_raw = G_d.clone()
        if arm == "C":
            G_d = G_d - Phi_d @ (K_ref @ net(Xr, Ur).reshape(-1))
        fit = torch.linalg.norm(Xnext[:, :NP_].reshape(-1) - pred - G_d).item()
        ce = combos_err(l)
        return dict(fit=fit, combo_err=ce.numpy(),
                    combo_mean=ce.mean().item(), combo_max=ce.max().item(),
                    rho_learned=op.rho(Phi_d, G_d),
                    rho_raw=op.rho(Phi_d, Gd_raw),
                    ann_norm=torch.linalg.norm(G_d).item())


hdr("Arms")
init_err = combos_err(torch.log(theta_init / theta_true))
print(f"  initial combination error: mean {init_err.mean():.4f}  max {init_err.max():.4f}")
print(f"  {'arm':<34}{'fit':>12}{'combo mean':>13}{'combo max':>12}"
      f"{'rho(learned)':>14}")
print(f"  {'init (no training)':<34}{torch.linalg.norm(base_err).item():>12.3e}"
      f"{init_err.mean().item():>13.4f}{init_err.max().item():>12.4f}{'-':>14}")

arms = [("A no projection", "A", 0.0),
        ("B 2025 penalty beta=1e0", "B", 1e0),
        ("B 2025 penalty beta=1e2", "B", 1e2),
        ("B 2025 penalty beta=1e4", "B", 1e4),
        ("C 2026 exact subtraction", "C", 0.0)]
table = {}
for label, arm, beta in arms:
    import time as _t; _t0 = _t.time()
    outs = [run_arm(arm, s, beta) for s in [0, 1, 2]]
    _dt = _t.time() - _t0
    fit = np.mean([o["fit"] for o in outs])
    cm = np.mean([o["combo_mean"] for o in outs])
    cx = np.mean([o["combo_max"] for o in outs])
    rl = np.mean([o["rho_learned"] for o in outs])
    sd = np.std([o["combo_mean"] for o in outs])
    print(f"  {label:<34}{fit:>12.3e}{cm:>13.4f}{cx:>12.4f}{rl:>14.4f}   [{_dt:.0f} s]")
    sys.stdout.flush()
    table[label] = dict(fit=float(fit), combo_mean=float(cm), combo_sd=float(sd),
                        combo_max=float(cx), rho_learned=float(rl),
                        per_combo=[float(x) for x in
                                   np.mean([o["combo_err"] for o in outs], axis=0)])
RES["arms"] = table

print(f"\n  per-combination relative error, mean over 3 seeds")
print(f"  {'combination':<10}" + "".join(f"{lab.split()[0]+lab.split()[-1][:6]:>14}"
                                          for lab, _, _ in arms))
for i, n in enumerate(gc.COMBO_NAMES):
    print(f"  {n:<10}" + "".join(f"{table[lab]['per_combo'][i]:>14.4f}"
                                 for lab, _, _ in arms))
print(f"  {'INIT':<10}" + "".join(f"{init_err[i].item():>14.4f}" for _ in arms))

print("\n  seed spread of the combination mean error:")
for lab in table:
    print(f"    {lab:<34} sd = {table[lab]['combo_sd']:.4e}")

with open(os.path.join(OUT, "s3c_tables.json"), "w") as f:
    json.dump(RES, f, indent=2)
print(f"\nwrote {os.path.join(OUT, 's3c_tables.json')}")
