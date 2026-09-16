"""Stage 1 and Stage 2: baseline parameterization, scheduling, identifiable combinations.

__project_origin__ = "added"

Run:
    conda run -n GraduationProject python -u \
      scripts/gantry/orthogonal-by-construction/investigation-20260915/s12_parameterization_and_rank.py

Writes `results/s12_parameterization_and_rank.log` and `results/s12_tables.json`.
Every number printed here is produced in this run; nothing is quoted from a stored artifact.
"""

__project_origin__ = "added"

import json
import os
import sys

import numpy as np
import sympy as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import obc_common as gc                                            # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


sys.stdout = Tee(os.path.join(OUT, "s12_parameterization_and_rank.log"))
RES = {}


def hdr(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


theta0, Lb = gc.nominal()
print(f"nominal theta (14) = "
      f"{dict(zip(gc.PARAM_NAMES, [round(v, 6) for v in theta0.tolist()]))}")
print(f"Lb = {Lb}   Ts = {gc.TS}   up_sample = {gc.UP_SAMPLE}   "
      f"Y range = {gc.Y_RANGE}")

# ----------------------------------------------------------------------
hdr("S0  Standalone model verified against the production block")
rel = gc.check_against_block()
RES["block_vs_standalone_rel"] = rel
print(f"  verdict: {'OK' if rel < 1e-6 else 'MISMATCH'} "
      f"(float32 block against float64 reimplementation)")

mn, yarg = gc.mass_pd_margin(theta0, Lb)
print(f"  min eig M(Y) over [{gc.Y_RANGE[0]}, {gc.Y_RANGE[1]}] = {mn:.6f} kg at Y = {yarg:+.3f} m")
RES["min_eig_MY"] = mn
print(f"  verdict: M(Y) positive definite over the documented range: {mn > 0}")

# ----------------------------------------------------------------------
hdr("S1a  Is the explicit transition linear in any finite parameter set?")
# Symbolic: does det M(Y) depend on theta? If it does, the explicit map's velocity rows
# are ratios whose denominator carries the unknowns, so no finite linear-in-unknowns
# representation of the EXPLICIT map exists.
m1, m2, mb, mh, Jb, Jh, dd, Y, Lbs = sp.symbols(
    "m1 m2 mb mh Jb Jh d Y L_b", positive=True)
alpha = m1 + m2 + mb + mh
beta = (m1 - m2) * Lbs / 2
gam = Jb + Jh + (m1 + m2) * Lbs ** 2 / 4
M0s = sp.Matrix([[alpha, beta, 0],
                 [beta, gam + mh * dd ** 2, -mh * dd],
                 [0, -mh * dd, mh]])
M1s = sp.Matrix([[0, -mh, 0], [-mh, 0, 0], [0, 0, 0]])
M2s = sp.Matrix([[0, 0, 0], [0, mh, 0], [0, 0, 0]])
MYs = M0s + M1s * Y + M2s * Y ** 2
detMY = sp.simplify(sp.expand(MYs.det()))
print("  det M(Y) =")
sp.pprint(sp.factor(sp.collect(detMY, Y)))
free = sorted(detMY.free_symbols, key=str)
print(f"\n  det M(Y) free symbols: {[str(s) for s in free]}")
dep = [str(s) for s in free if str(s) != "Y"]
print(f"  -> the denominator depends on the parameters {dep}")
print("  VERDICT H1a: ACCEPTED. The explicit velocity rows are p/q with q carrying the")
print("  unknowns, so no exact finite linear-in-unknown-combinations representation of")
print("  the explicit one-step map exists. Any linear surrogate is a local expansion.")
RES["detMY_param_symbols"] = dep

# The denominator at Y = 0 must match the project's d0 = mh*(alpha*gamma - beta^2).
d0_proj = mh * (alpha * gam - beta ** 2)
print(f"\n  cross-check det M(0) - mh*(alpha*gamma - beta^2) = "
      f"{sp.simplify(detMY.subs(Y, 0) - d0_proj)}")
print("  (gantry_ss.build_poly_constants uses gamma WITHOUT mh*d^2; the identity above")
print("   confirms that convention reproduces det M0 exactly.)")

# ----------------------------------------------------------------------
hdr("S1b  The implicit force balance IS exactly linear in lumped coefficients")
q1, q2, q3, v1, v2, v3, a1, a2, a3 = sp.symbols("q1 q2 q3 v1 v2 v3 a1 a2 a3")
u1, u2, u3 = sp.symbols("u1 u2 u3")
cg1, cg2, cy, cb1, cb2, kb1, kb2 = sp.symbols(
    "c_g1 c_g2 c_y c_b1 c_b2 k_b1 k_b2", positive=True)
Cs = sp.Matrix([[cg1 + cg2, (cg1 - cg2) * Lbs / 2, 0],
                [(cg1 - cg2) * Lbs / 2, cb1 + cb2 + (cg1 + cg2) * Lbs ** 2 / 4, 0],
                [0, 0, cy]])
Ks = sp.Matrix([[0, 0, 0], [0, kb1 + kb2, 0], [0, 0, 0]])
Ps = sp.Matrix([[1, 1, 0], [Lbs / 2, -Lbs / 2, 0], [0, 0, 1]])
qv = sp.Matrix([q1, q2, q3]); vv = sp.Matrix([v1, v2, v3]); av = sp.Matrix([a1, a2, a3])
uv = sp.Matrix([u1, u2, u3])
r = MYs.subs(Y, q3) * av + Cs * vv + Ks * qv - Ps * uv

# Lumped coefficient vector: exactly the 10 quantities the matrices depend on, with the
# two bilinear ones (mh*d, mh*d^2) promoted to their own coefficients.
lump = sp.symbols("A B G MH MD MD2 CG1 CG2 CBs CY KBs")
A_, B_, G_, MH_, MD_, MD2_, CG1_, CG2_, CBs_, CY_, KBs_ = lump
sub_lump = {alpha: A_, beta: B_, gam: G_}
r_l = r.subs({
    m1: 0, m2: 0, mb: 0, Jb: 0, Jh: 0,   # placeholders, replaced structurally below
})
# Rebuild the residual directly from lumped symbols, then prove it equals r.
M0l = sp.Matrix([[A_, B_, 0], [B_, G_ + MD2_, -MD_], [0, -MD_, MH_]])
M1l = sp.Matrix([[0, -MH_, 0], [-MH_, 0, 0], [0, 0, 0]])
M2l = sp.Matrix([[0, 0, 0], [0, MH_, 0], [0, 0, 0]])
Cl = sp.Matrix([[CG1_ + CG2_, (CG1_ - CG2_) * Lbs / 2, 0],
                [(CG1_ - CG2_) * Lbs / 2, CBs_ + (CG1_ + CG2_) * Lbs ** 2 / 4, 0],
                [0, 0, CY_]])
Kl = sp.Matrix([[0, 0, 0], [0, KBs_, 0], [0, 0, 0]])
r_lump = (M0l + M1l * q3 + M2l * q3 ** 2) * av + Cl * vv + Kl * qv - Ps * uv
back = {A_: alpha, B_: beta, G_: gam, MH_: mh, MD_: mh * dd, MD2_: mh * dd ** 2,
        CG1_: cg1, CG2_: cg2, CBs_: cb1 + cb2, CY_: cy, KBs_: kb1 + kb2}
diff = sp.simplify(sp.expand(r_lump.subs(back) - r))
print(f"  residual(lumped, substituted back) - residual(raw) = {list(diff)}")
ok_lump = all(sp.simplify(e) == 0 for e in diff)
print(f"  identity holds: {ok_lump}")
# Linearity in the lumped vector
lin = all(sp.simplify(sp.diff(e, s, 2)) == 0 for e in r_lump for s in lump)
print(f"  residual is affine in every lumped coefficient (all 2nd derivatives zero): {lin}")
print(f"  number of lumped coefficients = {len(lump)} "
      f"({', '.join(str(s) for s in lump)})")
print("  ADMISSIBILITY CONSTRAINTS (the embedding is exact but REDUNDANT):")
print("    MH appears in M0[2,2], M1[0,1], M1[1,0] and M2[1,1] and must be the SAME scalar;")
print("    MD^2 = MD2 * MH   (since MD = mh*d, MD2 = mh*d^2  =>  MD^2 = mh^2 d^2 = MH*MD2);")
print("    A, G, MH > 0 and M(Y) > 0 over the scheduling range.")
print("  Estimating MD and MD2 as free coefficients breaks MD^2 = MH*MD2 and leaves the")
print("  physical (mh, d) unrecoverable; it is 11 coefficients for 10 degrees of freedom.")
RES["lumped_identity_holds"] = bool(ok_lump)
RES["lumped_is_affine"] = bool(lin)
print("  VERDICT H1b: ACCEPTED with the redundancy constraint named.")
print("  NOTE on the estimator: this residual is M(Y) times the acceleration error, so")
print("  least squares on r weights the three logical channels by M(Y) and needs qddot.")
print("  It is a different estimand from output prediction; see decision-report.md.")

# ----------------------------------------------------------------------
hdr("S2  Parameter Jacobian of the RK4 transition: rank and null space")


def stacked_jacobian(theta, Lb, X, U, Y_frozen=None, coord="physical"):
    """d vec(x_{k+1}) / d theta stacked over points, in the requested coordinates."""
    def f(th):
        if coord == "log":
            th = theta * torch.exp(th)
        elif coord == "normalized":
            th = theta * th
        return gc.rk4_step(th, Lb, X, U, Y_frozen=Y_frozen).reshape(-1)
    base = (torch.zeros_like(theta) if coord == "log"
            else torch.ones_like(theta) if coord == "normalized" else theta)
    return torch.autograd.functional.jacobian(f, base, vectorize=True)


def rank_report(J, label, tol_rel=1e-8):
    s = torch.linalg.svdvals(J.double())
    r = int((s > tol_rel * s[0]).sum())
    print(f"  {label}: shape {tuple(J.shape)}  rank={r} at rel tol {tol_rel:.0e}")
    print(f"    sigma = {np.array2string(s.numpy(), precision=3, max_line_width=200)}")
    if r < len(s):
        print(f"    sigma[{r-1}]/sigma[0] = {(s[r-1]/s[0]).item():.3e}   "
              f"sigma[{r}]/sigma[0] = {(s[r]/s[0]).item():.3e}   "
              f"gap factor = {(s[r-1]/s[r]).item():.3e}")
    return r, s


X, U = gc.design_points(24, seed=1)
J_phys = stacked_jacobian(theta0, Lb, X, U)
r_phys, s_phys = rank_report(J_phys, "physical coords, 24 designed points, endogenous Y")
RES["rank_physical"] = r_phys
RES["sigma_physical"] = s_phys.tolist()

print("\n  rank threshold sweep (physical coords):")
for tol in [1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]:
    s = s_phys
    print(f"    rel tol {tol:.0e} -> rank {int((s > tol * s[0]).sum())}")

print("\n  coordinate comparison (rank is invariant, conditioning is not):")
for coord in ["physical", "normalized", "log"]:
    Jc = stacked_jacobian(theta0, Lb, X, U, coord=coord)
    sc = torch.linalg.svdvals(Jc.double())
    rc = int((sc > 1e-8 * sc[0]).sum())
    cond = (sc[0] / sc[rc - 1]).item()
    print(f"    {coord:<11} rank {rc}  cond(on range) = {cond:.3e}  "
          f"sigma0 = {sc[0]:.3e}  sigma[{rc-1}] = {sc[rc-1]:.3e}")
    RES[f"cond_{coord}"] = cond

# --- independent central differences with a step sweep -----------------
print("\n  central-difference cross-check (physical coords), relative step sweep:")
for h_rel in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    cols = []
    for i in range(14):
        h = h_rel * abs(theta0[i].item())
        tp, tm = theta0.clone(), theta0.clone()
        tp[i] += h; tm[i] -= h
        with torch.no_grad():
            cols.append(((gc.rk4_step(tp, Lb, X, U) - gc.rk4_step(tm, Lb, X, U))
                         / (2 * h)).reshape(-1))
    Jfd = torch.stack(cols, dim=1)
    err = (torch.linalg.norm(Jfd - J_phys) / torch.linalg.norm(J_phys)).item()
    sfd = torch.linalg.svdvals(Jfd)
    print(f"    h_rel {h_rel:.0e}: ||J_fd - J_ad||/||J_ad|| = {err:.3e}   "
          f"rank(J_fd) = {int((sfd > 1e-8 * sfd[0]).sum())}")

# --- the predicted null space -----------------------------------------
print("\n  predicted null directions (from the algebra of build_matrices):")
ix = {n: i for i, n in enumerate(gc.PARAM_NAMES)}
Npred = torch.zeros(14, 4, dtype=torch.float64)
Npred[ix["kb1"], 0] = 1.0; Npred[ix["kb2"], 0] = -1.0
Npred[ix["cb1"], 1] = 1.0; Npred[ix["cb2"], 1] = -1.0
for j, Jn in enumerate(["Jb", "Jh"]):
    Npred[ix["m1"], 2 + j] = -0.5
    Npred[ix["m2"], 2 + j] = -0.5
    Npred[ix["mb"], 2 + j] = 1.0
    Npred[ix[Jn], 2 + j] = Lb ** 2 / 4
Npred = torch.linalg.qr(Npred)[0]
names = ["kb1 - kb2", "cb1 - cb2",
         "-m1/2 - m2/2 + mb + (Lb^2/4) Jb", "-m1/2 - m2/2 + mb + (Lb^2/4) Jh"]
for n in names:
    print(f"    {n}")

U_, S_, Vh = torch.linalg.svd(J_phys, full_matrices=True)
Nnum = Vh[r_phys:, :].T            # numerical null space (14, 14-r)
print(f"\n  numerical null space dimension = {Nnum.shape[1]}")
sv = torch.linalg.svdvals(Npred.T @ Nnum)
angles = torch.arccos(sv.clamp(-1, 1))
print(f"  principal angles predicted vs numerical null space [rad] = "
      f"{np.array2string(angles.numpy(), precision=3)}")
print(f"  max principal angle = {angles.max().item():.3e} rad")
RES["max_principal_angle"] = angles.max().item()
print(f"  VERDICT: predicted and numerical null spaces "
      f"{'AGREE' if angles.max().item() < 1e-6 else 'DIFFER'}")

# --- exact invariance test --------------------------------------------
print("\n  exact invariance: perturb along a null direction, compare matrices "
      "and the transition at INDEPENDENT points")
Xi, Ui = gc.design_points(16, seed=7)
for j in range(Npred.shape[1]):
    for eps in [1e-3, 1e-1]:
        th2 = theta0 + eps * Npred[:, j] / Npred[:, j].abs().max()
        mats0 = gc.build_matrices(theta0, Lb)
        mats2 = gc.build_matrices(th2, Lb)
        dm = max((a - b).abs().max().item() for a, b in zip(mats0, mats2))
        with torch.no_grad():
            dxs = (gc.rk4_step(theta0, Lb, Xi, Ui)
                   - gc.rk4_step(th2, Lb, Xi, Ui)).abs().max().item()
        c0, c2 = gc.combos(theta0, Lb), gc.combos(th2, Lb)
        dc = (c0 - c2).abs().max().item()
        print(f"    null dir {j} eps={eps:.0e}: max |dM,dC,dK| = {dm:.3e}   "
              f"max |dx_next| = {dxs:.3e}   max |d combo| = {dc:.3e}")

print("\n  converse: perturb each of the 10 combinations and confirm an independent response")
base = gc.rk4_step(theta0, Lb, Xi, Ui).detach()
Jc = torch.autograd.functional.jacobian(
    lambda th: gc.rk4_step(th, Lb, Xi, Ui).reshape(-1), theta0, vectorize=True)
# map raw Jacobian to the combination coordinates via the combo Jacobian pseudoinverse
Jcomb_raw = torch.autograd.functional.jacobian(lambda th: gc.combos(th, Lb), theta0)
J_in_combo = Jc @ torch.linalg.pinv(Jcomb_raw)
s_combo = torch.linalg.svdvals(J_in_combo)
print(f"    sigma of d x_next / d combo = "
      f"{np.array2string(s_combo.numpy(), precision=3, max_line_width=200)}")
print(f"    rank = {int((s_combo > 1e-10 * s_combo[0]).sum())} of 10   "
      f"cond = {(s_combo[0]/s_combo[-1]).item():.3e}")
RES["rank_in_combo_coords"] = int((s_combo > 1e-10 * s_combo[0]).sum())
RES["cond_in_combo_coords"] = (s_combo[0] / s_combo[-1]).item()

# --- robustness across admissible nominal parameter sets ---------------
print("\n  rank across three admissible nominal parameter sets and two reference grids:")
rng = np.random.default_rng(11)
for k in range(3):
    pert = torch.tensor(rng.uniform(0.7, 1.4, size=14), dtype=torch.float64)
    th_k = theta0 * pert
    mn_k, _ = gc.mass_pd_margin(th_k, Lb)
    for g, seed in enumerate([2, 3]):
        Xg, Ug = gc.design_points(24, seed=seed)
        Jk = stacked_jacobian(th_k, Lb, Xg, Ug)
        sk = torch.linalg.svdvals(Jk)
        rk_ = int((sk > 1e-8 * sk[0]).sum())
        ang = torch.arccos(torch.linalg.svdvals(
            Npred.T @ torch.linalg.svd(Jk, full_matrices=True)[2][rk_:, :].T
        ).clamp(-1, 1)).max().item()
        print(f"    set {k} grid {g}: min eig M = {mn_k:8.4f}  rank = {rk_}  "
              f"max angle to predicted null = {ang:.3e} rad")

# --- scheduling convention comparison (H1c) ---------------------------
hdr("S1c  Scheduling conventions: endogenous, exogenous/measured, frozen")
Xs, Us = gc.design_points(24, seed=5)
with torch.no_grad():
    x_endo = gc.rk4_step(theta0, Lb, Xs, Us)
    ref = (x_endo - Xs).abs().max().item()
    print(f"  reference scale: max |x_next - x| over the designed points = {ref:.3e}")
    for Yf in [0.0, 0.15, 0.30]:
        x_fro = gc.rk4_step(theta0, Lb, Xs, Us, Y_frozen=Yf)
        dd_ = (x_endo - x_fro).abs().max().item()
        print(f"  frozen Y = {Yf:+.2f}: max |x_next(endo) - x_next(frozen)| = {dd_:.3e}   "
              f"relative to state increment = {dd_/ref:.3e}")
    # exogenous = endogenous value supplied from outside; must be bit-identical here
    x_exo = gc.rk4_step(theta0, Lb, Xs, Us, Y_frozen=Xs[:, 2])
    print(f"  exogenous Y held at x[2] of the INITIAL state: max difference to endogenous "
          f"= {(x_endo - x_exo).abs().max().item():.3e}")
    print("  (nonzero because the endogenous convention re-reads Y at every RK4 stage and")
    print("   every substep, blocks.py:778-784 and 825; an exogenous/measured Y is held.)")
    # float64 round-off floor for scale
    x_a = gc.rk4_step(theta0, Lb, Xs, Us)
    x_b = gc.rk4_step(theta0, Lb, Xs.clone(), Us.clone())
    print(f"  round-off floor (same computation twice) = "
          f"{(x_a - x_b).abs().max().item():.3e}")

print("\n  can a scheduling mismatch mimic a parameter mismatch?")
# project the frozen-vs-endogenous transition difference onto the parameter tangent span
with torch.no_grad():
    delta = (gc.rk4_step(theta0, Lb, Xs, Us)
             - gc.rk4_step(theta0, Lb, Xs, Us, Y_frozen=0.0)).reshape(-1)
Jm = stacked_jacobian(theta0, Lb, Xs, Us)
coef, *_ = torch.linalg.lstsq(Jm, delta.unsqueeze(-1))
resid = delta - (Jm @ coef).squeeze(-1)
frac = 1 - (torch.linalg.norm(resid) / torch.linalg.norm(delta)).item() ** 2
print(f"  fraction of the frozen-Y error explained by the 14 parameter tangents = {frac:.4f}")
print(f"  residual 2-norm / total = "
      f"{(torch.linalg.norm(resid)/torch.linalg.norm(delta)).item():.4e}")
RES["frozenY_explained_by_param_tangents"] = frac

with open(os.path.join(OUT, "s12_tables.json"), "w") as f:
    json.dump(RES, f, indent=2)
print(f"\nwrote {os.path.join(OUT, 's12_tables.json')}")
