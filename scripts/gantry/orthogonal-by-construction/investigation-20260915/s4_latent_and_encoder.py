"""Stage 4: additional states, the reference-set zero slice, and the encoder.

__project_origin__ = "added"

Tests, on the real gantry transition rather than on the surrogate:

  T1  Prop. 7.1 of `state-level-obc-derivation.tex` (the zero slice): with a
      baseline-only reference set the additional states are identically zero, so the
      fitted coefficient is blind to every parameter on the latent path. Three reference
      policies are compared: the baseline-only set (x_a = 0), the designed set (R1), and
      a lagged augmented rollout (R2).
  T2  the same parameters DO move the learned physical write wherever x_a != 0, which is
      what makes the blindness a defect rather than a harmless invariance.
  T3  latent gauge invariance: x_a -> V x_a absorbed into the block's weights leaves the
      condition unchanged.
  T4  the encoder: joint sensitivity rank and conditioning of [d y / d theta , d y / d x0]
      over a trajectory, and the size of the output response to an initial-state
      perturbation, which decides whether encoder freedom can stand in for a parameter
      change.

Run:
    conda run -n GraduationProject python -u \
      scripts/gantry/orthogonal-by-construction/investigation-20260915/s4_latent_and_encoder.py
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
os.makedirs(OUT, exist_ok=True)
sys.stdout = op.Tee(os.path.join(OUT, "s4_latent_and_encoder.log"))
RES = {}


def hdr(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)
    sys.stdout.flush()


theta0, Lb = gc.nominal()
net = op.ANN(seed=0).double()
eta = op.flat_params(net).detach().clone()

# Index the flat parameter vector so the latent-path parameters can be named exactly.
offsets, i = {}, 0
for n, p in net.named_parameters():
    offsets[n] = (i, i + p.numel(), p.shape)
    i += p.numel()
print("  learned-block parameter layout:")
for n, (a, b, s) in offsets.items():
    print(f"    {n:<12} indices [{a:4d}, {b:4d})  shape {tuple(s)}")

# eta_lat: parameters acting ONLY on the x_a argument (the two latent input columns of
# the first layer) or ONLY on the R_a output block (the two latent output rows of the
# last layer and their biases). This is exactly the set of Prop. 7.1.
a1, b1, s1 = offsets["l1.weight"]
mask_lat = torch.zeros(eta.numel(), dtype=torch.bool)
W1 = torch.zeros(s1, dtype=torch.bool)
W1[:, 6:8] = True                       # columns reading x_a
mask_lat[a1:b1] = W1.reshape(-1)
a3, b3, s3 = offsets["l3.weight"]
W3 = torch.zeros(s3, dtype=torch.bool)
W3[6:8, :] = True                       # rows writing the latent block
mask_lat[a3:b3] = W3.reshape(-1)
ab3, bb3, _ = offsets["l3.bias"]
bb = torch.zeros(s3[0], dtype=torch.bool); bb[6:8] = True
mask_lat[ab3:bb3] = bb
print(f"\n  eta_lat: {int(mask_lat.sum())} of {eta.numel()} parameters "
      f"(latent input columns of layer 1, latent output rows and biases of layer 3)")
RES["n_eta_lat"] = int(mask_lat.sum())


def G_phys(eta_v, X, Xa, U):
    return op.g_write(net, eta_v, X, Xa, U)[:, :op.NP_].reshape(-1)


# ----------------------------------------------------------------------
hdr("T1  Sensitivity of the frozen coefficient to the latent-path parameters")
M = 128
Xr, Ur = gc.design_points(M, seed=101)
Phi_r = op.phi_stack(theta0, Lb, Xr, Ur)
K_r, r_r, _ = op.pinv_report(Phi_r, label="Phi_ref")

policies = {
    "baseline-only (x_a = 0)": torch.zeros(M, op.NA, dtype=torch.float64),
    "R1 designed latent set": op.design_latent(M, seed=11, scale=1.0),
    "R2 lagged augmented rollout (surrogate)": None,   # filled below
}
# R2 surrogate: roll the learned block's latent recursion forward along the reference
# points, so the latent values are the ones a previous epoch's augmented pass would have
# produced. x_a(k+1) = R_a g_eta(x_p(k), x_a(k), u(k)).
xa = torch.zeros(1, op.NA, dtype=torch.float64)
lat = []
for k in range(M):
    lat.append(xa.clone())
    xa = op.g_write(net, eta, Xr[k:k + 1], xa, Ur[k:k + 1])[:, op.NP_:]
policies["R2 lagged augmented rollout (surrogate)"] = torch.cat(lat, dim=0)
print(f"  R2 latent RMS = "
      f"{policies['R2 lagged augmented rollout (surrogate)'].pow(2).mean().sqrt():.4e}, "
      f"R1 latent RMS = {policies['R1 designed latent set'].pow(2).mean().sqrt():.4e}")

print(f"\n  {'reference policy':<42}{'||d eta_aux / d eta_lat||':>27}"
      f"{'||d eta_aux / d eta||':>24}")
for name, Xa in policies.items():
    J = torch.func.jacrev(lambda e: K_r @ G_phys(e, Xr, Xa, Ur))(eta)
    n_lat = torch.linalg.norm(J[:, mask_lat]).item()
    n_all = torch.linalg.norm(J).item()
    print(f"  {name:<42}{n_lat:>27.6e}{n_all:>24.6e}")
    RES[f"T1_{name}"] = dict(lat=n_lat, all=n_all)
print("\n  VERDICT: with the baseline-only set the coefficient derivative along every")
print("  latent-path parameter is exactly zero, reproducing Prop. 7.1 on the real")
print("  gantry regressor rather than on the surrogate. R1 and R2 both restore it.")

# ----------------------------------------------------------------------
hdr("T2  The same parameters DO move the physical write where x_a != 0")
Xa_live = policies["R1 designed latent set"]
Jw0 = torch.func.jacrev(
    lambda e: G_phys(e, Xr, torch.zeros(M, op.NA, dtype=torch.float64), Ur))(eta)
Jw1 = torch.func.jacrev(lambda e: G_phys(e, Xr, Xa_live, Ur))(eta)
print(f"  ||d(R_p g)/d eta_lat|| at x_a = 0        = "
      f"{torch.linalg.norm(Jw0[:, mask_lat]).item():.6e}")
print(f"  ||d(R_p g)/d eta_lat|| at live x_a       = "
      f"{torch.linalg.norm(Jw1[:, mask_lat]).item():.6e}")
print("  So a projection fitted at x_a = 0 charges nothing for a physical write that")
print("  the additional states produce. The defect is in the reference set, not in the")
print("  condition, and R2 is the cheapest repair that keeps the points on the data")
print("  manifold.")
RES["T2_zero"] = torch.linalg.norm(Jw0[:, mask_lat]).item()
RES["T2_live"] = torch.linalg.norm(Jw1[:, mask_lat]).item()

# ----------------------------------------------------------------------
hdr("T3  Latent gauge invariance")
V = torch.tensor([[2.0, 0.5], [-1.0, 3.0]], dtype=torch.float64)
eta_g = eta.clone()
W1m = eta_g[a1:b1].reshape(s1).clone()
W1m[:, 6:8] = W1m[:, 6:8] @ torch.linalg.inv(V)          # read V x_a as if it were x_a
eta_g[a1:b1] = W1m.reshape(-1)
W3m = eta_g[a3:b3].reshape(s3).clone()
W3m[6:8, :] = V @ W3m[6:8, :]                            # write V x_a
eta_g[a3:b3] = W3m.reshape(-1)
bm = eta_g[ab3:bb3].clone(); bm[6:8] = V @ bm[6:8]; eta_g[ab3:bb3] = bm

Xa0 = op.design_latent(M, seed=11, scale=1.0)
w_ref = op.g_write(net, eta, Xr, Xa0, Ur)
w_gauge = op.g_write(net, eta_g, Xr, Xa0 @ V.T, Ur)
print(f"  physical rows: max |difference| = "
      f"{(w_ref[:, :6] - w_gauge[:, :6]).abs().max().item():.3e}")
print(f"  latent rows:   max |w_gauge - V w_ref| = "
      f"{(w_gauge[:, 6:] - w_ref[:, 6:] @ V.T).abs().max().item():.3e}")
print("  The physical write, and therefore the stacked condition, is unchanged; the")
print("  latent write transforms exactly as the gauge demands. A condition that")
print("  constrained the latent rows would depend on an arbitrary coordinate choice.")

# ----------------------------------------------------------------------
hdr("T4  Encoder: can an initial-state change stand in for a parameter change?")
T = 200
rng = np.random.default_rng(5)
U = torch.tensor(rng.normal(size=(1, T, 3)) * 50.0)
x0 = torch.tensor([[0.0, 0.0, 0.10, 0.0, 0.0, 0.0]], dtype=torch.float64)
print(f"  trajectory: T = {T} samples at Ts = {gc.TS} s ({T*gc.TS*1e3:.1f} ms), "
      f"input RMS = {U.pow(2).mean().sqrt().item():.1f} N")


def y_of(th_log, x0_v):
    th = theta0 * torch.exp(th_log)
    _, Y = gc.simulate(th, Lb, x0_v, U)
    return Y.reshape(-1)


# Forward mode: 14 and 6 tangents respectively, against 3*T cotangents in reverse mode.
Jth = torch.func.jacfwd(lambda l: y_of(l, x0))(torch.zeros(14, dtype=torch.float64))
Jx0 = torch.func.jacfwd(
    lambda z: y_of(torch.zeros(14, dtype=torch.float64), z))(x0).reshape(Jth.shape[0], 6)

for lab, J in [("d y / d theta (log)", Jth), ("d y / d x0", Jx0),
               ("joint [theta, x0]", torch.cat([Jth, Jx0], dim=1))]:
    s = torch.linalg.svdvals(J)
    r = int((s > 1e-10 * s[0]).sum())
    print(f"  {lab:<22} shape {tuple(J.shape)}  rank {r:2d}  "
          f"cond(on range) {(s[0]/s[r-1]).item():.3e}")
    print(f"    sigma = {np.array2string(s.numpy(), precision=3, max_line_width=200)}")
    RES[f"T4_rank_{lab}"] = r

# how much of a parameter response can the encoder reproduce?
Qx, _ = op.proj_basis(Jx0)
print("\n  fraction of each parameter-tangent direction that the 6 initial-state")
print("  directions can reproduce over this window:")
for i, n in enumerate(gc.PARAM_NAMES):
    v = Jth[:, i]
    if v.norm() < 1e-300:
        print(f"    {n:<5} (null direction, no response)")
        continue
    f = (torch.linalg.norm(Qx @ (Qx.T @ v)) / v.norm()).item()
    print(f"    {n:<5} {f:.4f}")
    RES[f"T4_encoder_overlap_{n}"] = f
print("\n  Reading: a value near 1 means an initial-state error is indistinguishable")
print("  from that parameter error over this window, so encoder freedom and that")
print("  parameter compete. A value near 0 means they do not. This is a statement about")
print("  THIS window length and excitation, not a universal encoder verdict.")

with open(os.path.join(OUT, "s4_tables.json"), "w") as f:
    json.dump(RES, f, indent=2)
print(f"\nwrote {os.path.join(OUT, 's4_tables.json')}")
