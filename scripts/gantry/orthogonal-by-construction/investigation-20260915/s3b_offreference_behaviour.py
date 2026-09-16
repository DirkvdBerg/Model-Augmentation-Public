"""Stage 3b: construction or weak penalty off the reference set?

__project_origin__ = "added"

`state-level-obc-derivation.tex` Sect. 9 names the size of the OFF-reference residual as
not verified and as the quantity that decides whether the frozen construction behaves
like a construction or like a weak penalty. Stage 3 T1 found the raw residual going the
WRONG way off the reference set, so this script isolates the cause.

Scale-free metric:

    rho(Z; G) = || P_Z G || / || G ||,     P_Z = Phi_Z pinv(Phi_Z)

the fraction of the learned write that the baseline tangent set at the points Z could
absorb. rho = 0 is perfect non-overlap; rho = 1 is complete absorbability. Unlike
|| Phi^T G || it cannot be made small by shrinking G, so a cancellation cannot be
mistaken for orthogonality.

Arms separated: reference-set SIZE, the OFFSET column, scheduling COVERAGE, and the
write SCALE of the learned block.

Run:
    conda run -n GraduationProject python -u \
      scripts/gantry/orthogonal-by-construction/investigation-20260915/s3b_offreference_behaviour.py
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
sys.stdout = op.Tee(os.path.join(OUT, "s3b_offreference_behaviour.log"))
RES = {}


def hdr(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


theta0, Lb = gc.nominal()
net = op.ANN(seed=0).double()
eta = op.flat_params(net).detach().clone()


def G_of(X, Xa, U):
    return op.g_write(net, eta, X, Xa, U)[:, :op.NP_].reshape(-1)


# Evaluation set, fixed for every arm so the rows are comparable.
Xe, Ue = gc.design_points(48, seed=202)
Xae = op.design_latent(48, seed=202)
Ge = G_of(Xe, Xae, Ue)

hdr("A  Singular spectrum of the stacked regressor: the offset column dominates")
Phi_wo = op.phi_stack(theta0, Lb, Xe, Ue, with_offset=False)
Phi_w = op.phi_stack(theta0, Lb, Xe, Ue, with_offset=True)
s_wo = torch.linalg.svdvals(Phi_wo)
s_w = torch.linalg.svdvals(Phi_w)
print(f"  sigma, parameter columns only: {np.array2string(s_wo.numpy(), precision=3, max_line_width=200)}")
print(f"  sigma, with the offset column: {np.array2string(s_w.numpy(), precision=3, max_line_width=200)}")
print(f"  sigma_max(with offset) / sigma_max(parameters only) = "
      f"{(s_w[0] / s_wo[0]).item():.3e}")
print("  The offset column is the nominal one-step transition f(theta_bar), whose")
print("  position rows are q + Ts*qdot, order 0.3 m. Every parameter column is a")
print("  derivative of that map, order 1e-3 or less. So the affine protected set is")
print("  numerically dominated by ONE direction that is not a parameter direction.")
print("  Consequence: a pinv cutoff stated relative to sigma_0 is a cutoff relative to")
print("  the offset column, and at rcond >= 1e-4 it would silently discard REAL")
print("  parameter directions.")
for rc in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
    print(f"    rcond {rc:.0e}: rank(with offset) = "
          f"{int((s_w > rc * s_w[0]).sum()):2d}   rank(parameters only) = "
          f"{int((s_wo > rc * s_wo[0]).sum()):2d}")
RES["sigma_ratio_offset"] = (s_w[0] / s_wo[0]).item()

hdr("B  Reference-set size sweep, with and without the offset column")
print("  rho_eval before projection is the baseline number every row is compared to.")
print(f"  rho(eval; G) with offset    = {op.rho(Phi_w, Ge):.6f}")
print(f"  rho(eval; G) without offset = {op.rho(Phi_wo, Ge):.6f}")
RES["rho_raw_with_offset"] = op.rho(Phi_w, Ge)
RES["rho_raw_no_offset"] = op.rho(Phi_wo, Ge)

print(f"\n  {'M_ref':>7}{'offset':>9}{'rho_ref(after)':>16}{'rho_eval(after)':>17}"
      f"{'||Gtil||/||G||':>16}")
rows = []
for M in [32, 64, 256, 1024]:
    Xr, Ur = gc.design_points(M, seed=101)
    Xar = op.design_latent(M, seed=101)
    Gr = G_of(Xr, Xar, Ur)
    for use_off in [True, False]:
        Phi_r = op.phi_stack(theta0, Lb, Xr, Ur, with_offset=use_off)
        K_r, _, _ = op.pinv_report(Phi_r)
        a = K_r @ Gr
        Phi_e = op.phi_stack(theta0, Lb, Xe, Ue, with_offset=use_off)
        Gt_e = Ge - Phi_e @ a
        Gt_r = Gr - Phi_r @ a
        rr, re = op.rho(Phi_r, Gt_r), op.rho(Phi_e, Gt_e)
        gg = (torch.linalg.norm(Gt_e) / torch.linalg.norm(Ge)).item()
        print(f"  {M:>7}{str(use_off):>9}{rr:>16.6f}{re:>17.6f}{gg:>16.4f}")
        rows.append(dict(M=M, offset=use_off, rho_ref=rr, rho_eval=re, growth=gg))
RES["size_sweep"] = rows
print("\n  Reading: rho_ref is ~0 by construction at every size. rho_eval is the")
print("  quantity that decides the method. Compare it to the unprojected rho above.")

hdr("C  Scheduling coverage of the reference set")
print(f"  {'Y range':>18}{'rho_eval(after)':>17}{'||Gtil||/||G||':>16}")
cov = []
for yr in [(-0.30, 0.30), (-0.15, 0.15), (-0.05, 0.05), (0.0, 0.0)]:
    Xr, Ur = gc.design_points(256, seed=101, y_range=yr)
    Xar = op.design_latent(256, seed=101)
    Gr = G_of(Xr, Xar, Ur)
    Phi_r = op.phi_stack(theta0, Lb, Xr, Ur)
    K_r, _, _ = op.pinv_report(Phi_r)
    a = K_r @ Gr
    Gt_e = Ge - Phi_w @ a
    re = op.rho(Phi_w, Gt_e)
    gg = (torch.linalg.norm(Gt_e) / torch.linalg.norm(Ge)).item()
    print(f"  {str(yr):>18}{re:>17.6f}{gg:>16.4f}")
    cov.append(dict(y_range=list(yr), rho_eval=re, growth=gg))
RES["coverage"] = cov

hdr("D  Control: the reference set IS the evaluation set")
print("  If rho_eval stays near zero here and not in B, the gap is generalization of")
print("  the frozen coefficient, not a defect of the projector.")
Phi_e_only = op.phi_stack(theta0, Lb, Xe, Ue)
K_e, _, _ = op.pinv_report(Phi_e_only, label="Phi_eval")
a_e = K_e @ Ge
Gt_same = Ge - Phi_e_only @ a_e
print(f"  rho_eval(after), reference = evaluation = {op.rho(Phi_e_only, Gt_same):.3e}")
print(f"  ||G_tilde|| / ||G||                     = "
      f"{(torch.linalg.norm(Gt_same) / torch.linalg.norm(Ge)).item():.4f}")
RES["D_same_set_rho"] = op.rho(Phi_e_only, Gt_same)

hdr("E  Is the off-reference gap a property of the ANN or of the regressor?")
print("  Replace the random ANN write by a write that is EXACTLY in the baseline span")
print("  (a genuine parameter direction). A coefficient fitted on one point set must")
print("  then transfer to any other, because the same theta explains both.")
Phi_r256 = None
Xr, Ur = gc.design_points(256, seed=101)
Xar = op.design_latent(256, seed=101)
Phi_r = op.phi_stack(theta0, Lb, Xr, Ur)
lam = torch.zeros(15, dtype=torch.float64); lam[3] = 1.0e-2     # a cg1-like direction
G_in_r = Phi_r @ lam
G_in_e = Phi_w @ lam
K_r, _, _ = op.pinv_report(Phi_r)
a = K_r @ G_in_r
print(f"  rho(eval; G_in_span) before = {op.rho(Phi_w, G_in_e):.6f}   (must be 1)")
res_e = G_in_e - Phi_w @ a
print(f"  ||G_in - Phi_eval a|| / ||G_in|| at the EVALUATION points = "
      f"{(torch.linalg.norm(res_e) / torch.linalg.norm(G_in_e)).item():.3e}")
print("  VERDICT: a write that lies in the baseline span is removed at ANY point set,")
print("  because the fitted coefficient recovers the true parameter direction. The")
print("  off-reference gap in B is therefore specific to a write that is NOT in the")
print("  span: there the coefficient is a least-squares artefact of the reference")
print("  points and carries no meaning elsewhere.")
RES["E_inspan_residual"] = (torch.linalg.norm(res_e) / torch.linalg.norm(G_in_e)).item()

hdr("F  The operational question: does the projection reduce what the baseline could absorb?")
print("  Reported as rho_eval(after) / rho_eval(before) at the best configuration of B.")
best = min([r for r in rows if r["M"] == 1024], key=lambda r: r["rho_eval"])
print(f"  best row: M = {best['M']}, offset = {best['offset']}, "
      f"rho_eval(after) = {best['rho_eval']:.6f}")
base_rho = RES["rho_raw_with_offset"] if best["offset"] else RES["rho_raw_no_offset"]
print(f"  rho_eval(before) = {base_rho:.6f}   ratio = {best['rho_eval'] / base_rho:.3f}")
RES["F_ratio"] = best["rho_eval"] / base_rho

with open(os.path.join(OUT, "s3b_tables.json"), "w") as f:
    json.dump(RES, f, indent=2)
print(f"\nwrote {os.path.join(OUT, 's3b_tables.json')}")
