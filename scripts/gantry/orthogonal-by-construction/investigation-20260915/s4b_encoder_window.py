"""Stage 4b: encoder versus physical parameters, with a window long enough to contain
the dynamics being identified.

__project_origin__ = "added"

Stage 4 T4 ran on a 200-sample window, 10 ms at Ts = 1/20000 s, and found that the six
initial-state directions reproduce 95 to 99.9 percent of every parameter tangent. That
window is far shorter than the plant's own timescales, so the finding is a property of
the window, not of the encoder. The timescales, computed here from the governing
matrices rather than quoted:

  the yaw resonance      sqrt(kb_sum / (J_eff + mh d^2))
  the X and Y axes       pure integrators, K[0,0] = K[2,2] = 0, so the relevant
                         timescale is the viscous one, m / c

This script recomputes both, sweeps the window over them, and additionally extracts the
exact linear dependency between the parameter tangent span and the initial-state span
that the joint rank in stage 4 revealed (rank 15 for 10 + 6 independent directions).

Run:
    conda run -n GraduationProject python -u \
      scripts/gantry/orthogonal-by-construction/investigation-20260915/s4b_encoder_window.py
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
sys.stdout = op.Tee(os.path.join(OUT, "s4b_encoder_window.log"))
RES = {}


def hdr(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


theta0, Lb = gc.nominal()
p = dict(zip(gc.PARAM_NAMES, theta0.tolist()))

hdr("Plant timescales, from the governing matrices")
M0, M1, M2, C, K, P = gc.build_matrices(theta0, Lb)
J_row = M0[1, 1].item()
kb_sum = K[1, 1].item()
w_yaw = np.sqrt(kb_sum / J_row)
print(f"  yaw row:  M0[1,1] = {J_row:.4f} kg m^2, K[1,1] = {kb_sum:.1f} N m/rad")
print(f"            omega = {w_yaw:.2f} rad/s = {w_yaw/2/np.pi:.2f} Hz, "
      f"period = {2*np.pi/w_yaw*1e3:.1f} ms")
tau_x = M0[0, 0].item() / C[0, 0].item()
tau_y = M0[2, 2].item() / C[2, 2].item()
print(f"  X axis:   K[0,0] = 0 (free integrator), m/c = "
      f"{M0[0,0].item():.2f} / {C[0,0].item():.2f} = {tau_x*1e3:.0f} ms")
print(f"  Y axis:   K[2,2] = 0 (free integrator), m/c = "
      f"{M0[2,2].item():.2f} / {C[2,2].item():.2f} = {tau_y*1e3:.0f} ms")
print(f"  So a window must reach at least {2*np.pi/w_yaw*1e3:.0f} ms to contain one yaw")
print(f"  period, and the integrator axes never forget at all.")
RES["yaw_period_ms"] = 2 * np.pi / w_yaw * 1e3
RES["tau_x_ms"] = tau_x * 1e3
RES["tau_y_ms"] = tau_y * 1e3

hdr("Window sweep: how much of each parameter tangent can the initial state reproduce?")
rng = np.random.default_rng(5)
x0 = torch.tensor([[0.0, 0.0, 0.10, 0.0, 0.0, 0.0]], dtype=torch.float64)
zero14 = torch.zeros(14, dtype=torch.float64)
rows = []
cache = {}
for T in [200, 1000, 4000]:
    print(f"  ... running T = {T}"); sys.stdout.flush()
    U = torch.tensor(rng.normal(size=(1, T, 3)) * 50.0)

    def y_of(th_log, x0_v, U=U):
        _, Y = gc.simulate(theta0 * torch.exp(th_log), Lb, x0_v, U)
        return Y.reshape(-1)

    Jth = torch.func.jacfwd(lambda l: y_of(l, x0))(zero14)
    Jx0 = torch.func.jacfwd(lambda z: y_of(zero14, z))(x0).reshape(Jth.shape[0], 6)
    Qx, _ = op.proj_basis(Jx0)
    Jj = torch.cat([Jth, Jx0], dim=1)
    sj = torch.linalg.svdvals(Jj)
    rj = int((sj > 1e-12 * sj[0]).sum())
    st = torch.linalg.svdvals(Jth)
    rt = int((st > 1e-12 * st[0]).sum())
    fr = {}
    for i, n in enumerate(gc.PARAM_NAMES):
        v = Jth[:, i]
        fr[n] = (torch.linalg.norm(Qx @ (Qx.T @ v)) / v.norm()).item() if v.norm() > 0 \
            else float("nan")
    med = float(np.nanmedian(list(fr.values())))
    mx = float(np.nanmax(list(fr.values())))
    print(f"  T = {T:6d} ({T*gc.TS*1e3:7.1f} ms): rank(dy/dtheta) = {rt:2d}, "
          f"rank(joint) = {rj:2d} of {rt + 6} independent  ->  "
          f"{rt + 6 - rj} exact dependenc{'y' if rt+6-rj == 1 else 'ies'}")
    print(f"      overlap with the initial-state span: median {med:.4f}, max {mx:.4f}, "
          f"min {min(x for x in fr.values() if not np.isnan(x)):.4f}")
    worst = sorted(((v, k) for k, v in fr.items() if not np.isnan(v)), reverse=True)[:4]
    print(f"      most confounded: "
          f"{', '.join(f'{k} {v:.4f}' for v, k in worst)}")
    sys.stdout.flush()
    rows.append(dict(T=T, ms=T * gc.TS * 1e3, rank_theta=rt, rank_joint=rj,
                     median=med, max=mx, per_param=fr))
    cache[T] = (Jth, Jx0)
RES["window_sweep"] = rows

hdr("The exact dependency between the parameter span and the initial-state span")
T = 4000
Jth, Jx0 = cache[T]
Jj = torch.cat([Jth, Jx0], dim=1)
Uj, Sj, Vhj = torch.linalg.svd(Jj, full_matrices=True)
rj = int((Sj > 1e-12 * Sj[0]).sum())
print(f"  joint rank {rj} of 20 columns; null dimension {20 - rj}")
print(f"  sigma tail = {np.array2string(Sj[rj-2:rj+3].numpy(), precision=3)}")
Nj = Vhj[rj:, :]
# The 4 known parameter null directions account for 4 of these; report the rest.
ixp = {n: i for i, n in enumerate(gc.PARAM_NAMES)}
known = torch.zeros(20, 4, dtype=torch.float64)
known[ixp["kb1"], 0] = 1; known[ixp["kb2"], 0] = -1
known[ixp["cb1"], 1] = 1; known[ixp["cb2"], 1] = -1
for j, Jn in enumerate(["Jb", "Jh"]):
    known[ixp["m1"], 2 + j] = -0.5; known[ixp["m2"], 2 + j] = -0.5
    known[ixp["mb"], 2 + j] = 1.0; known[ixp[Jn], 2 + j] = Lb ** 2 / 4
# log coordinates: a raw-parameter null direction v maps to diag(theta)^{-1} v
known[:14, :] = known[:14, :] / theta0[:, None]
Qk = torch.linalg.qr(known)[0]
resid = Nj.T - Qk @ (Qk.T @ Nj.T)
sr = torch.linalg.svdvals(resid)
extra = int((sr > 1e-8 * max(sr[0].item(), 1e-300)).sum())
print(f"  of the {20 - rj} null directions, {20 - rj - extra} are the known parameter")
print(f"  null directions and {extra} involve the initial state")
if extra:
    Ue, Se, _ = torch.linalg.svd(resid, full_matrices=False)
    for j in range(extra):
        v = Ue[:, j]
        v = v / v.abs().max()
        pn = {gc.PARAM_NAMES[i]: round(v[i].item(), 4) for i in range(14)
              if abs(v[i]) > 1e-3}
        xn = {f"x0[{i}]": round(v[14 + i].item(), 4) for i in range(6)
              if abs(v[14 + i]) > 1e-3}
        print(f"    extra direction {j}: log-parameter part {pn}")
        print(f"                        initial-state part {xn}")
        RES[f"extra_dir_{j}"] = dict(param=pn, x0=xn)

with open(os.path.join(OUT, "s4b_tables.json"), "w") as f:
    json.dump(RES, f, indent=2)
print(f"\nwrote {os.path.join(OUT, 's4b_tables.json')}")
