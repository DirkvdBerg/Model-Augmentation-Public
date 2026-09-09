"""Null-space check: does ker(Phi) equal the four structural degeneracies?

Question this answers (design decision 2 of the formal-definition spec):
whether the baseline transition factors exactly through the 10 identifiable
combinations, so that reparameterizing theta -> kappa(theta) is EXACT and the
gauge on the raw splits is free. If it is, Assumption 1 of the
orthogonal-by-construction paper can be satisfied by construction, which is
what makes its theorems describe our large-beta endpoint.

The 10 combinations are `Parameterized_Gantry_State_Block._combos_from_raw`:
    kb_sum  = kb1 + kb2                 m_total = m1 + m2 + mb
    cg1, cg2, cy                        m_diff  = m1 - m2
    cb_sum  = cb1 + cb2                 J_eff   = Jb + Jh + (m1+m2) Lb^2 / 4
    mh, d

Parameter order (14): kb1 kb2 cg1 cg2 cy cb1 cb2 mh m1 m2 mb Jb Jh d

PREDICTED null space of d kappa, derived in closed form from the combinations
above (kernel of the 10x14 Jacobian d kappa / d theta):
    v1 = e[kb1] - e[kb2]
    v2 = e[cb1] - e[cb2]
    v3 = e[Jb]  - e[Jh]
    v4 = e[m1] + e[m2] - 2 e[mb] - (Lb^2 / 2) e[Jh]
v4 is the direction that holds m_total, m_diff and J_eff fixed at once; it is
the fourth degeneracy, which the project record names only as "the fourth".
Derivation: m_diff fixed => dm1 = dm2 = a; m_total fixed => dmb = -2a;
J_eff fixed => dJb + dJh = -a Lb^2 / 2, and the dJb - dJh freedom is v3, so the
remaining direction takes dJb = 0, dJh = -a Lb^2 / 2.

TWO SEPARATE CLAIMS, tested separately, because they can fail independently:
  A. FACTORIZATION: span(v1..v4) is contained in ker(Phi). This is a property
     of the model structure alone. If it fails, the matrices do NOT depend on
     theta only through kappa, and the reparameterization is not exact.
  B. EXCITATION: rank(Phi) = 10, i.e. the point set excites all ten
     combinations. Phi = (df/dkappa)(dkappa/dtheta), so ker(Phi) can be LARGER
     than ker(dkappa) if the data does not excite some combination. Only when
     A and B both hold is ker(Phi) = ker(d kappa) exactly.

Frame: the Jacobian is taken w.r.t. log_params (the block's trainable
parameterization) and rescaled to PHYSICAL units by
    d f / d theta_j = (d f / d log theta_j) / theta_j,
which is the frame `orth_penalty.py` deploys. The rescale matters here: in log
coordinates the kb1/kb2 null direction is [kb2, -kb1], not [1, -1], so testing
the clean structural vectors requires physical columns.

Outputs -> scripts/gantry/orthogonality/code/nullspace_check.json
Run: conda run -n GraduationProject python scripts/gantry/orthogonality/code/nullspace_check.py
"""
import os
import sys
import json

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..', '..', '..')))  # repo root
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..')))              # scripts/gantry

from gantry_dynamic.config import RunConfig, REPO_ROOT
from model_augmentation.fit_systems.blocks import Parameterized_Gantry_State_Block

CFG = RunConfig(use_f64=True, snr=None, up_sample=2)
DIAG = os.path.join(REPO_ROOT, 'simulations', 'gantry_subnet', 'diagnostics',
                    'orth_projection')
OUT = os.path.join(os.path.dirname(__file__), 'nullspace_check.json')

NP = 14   # raw physical parameters
NX = 6    # baseline state rows
NK = 10   # identifiable combinations

# THEORY: rank truncation at the deployed cut. `orth_penalty.py` truncates the
# extended regressor at 1e-12 * sigma_0; the same relative cut is used here so
# the rank reported matches the one the penalty actually uses.
RANK_TOL_REL = 1e-12

# HEURISTIC: 60 samples spread over records and Y. More than step1's 6 because
# rank(Phi) = 10 needs the point set to excite ten combinations, and a handful
# of samples can be rank-deficient for reasons that have nothing to do with the
# model structure. No literature sets this number; it is raised until the
# reported rank stops changing (reported below as a stability check).
N_SAMPLES = 60

PNAMES = ["kb1", "kb2", "cg1", "cg2", "cy", "cb1", "cb2",
          "mh", "m1", "m2", "mb", "Jb", "Jh", "d"]
IX = {n: i for i, n in enumerate(PNAMES)}


def load_point_set():
    return np.load(os.path.join(DIAG, 'step0', 'point_set.npz'),
                   allow_pickle=True)


def make_block(d):
    blk = Parameterized_Gantry_State_Block(
        Y_op=None, std_x=d['std_x'], std_u=d['std_u'],
        x_mean=d['x_mean'], u_mean=d['u_mean'],
        Ts=CFG.ts_new, up_sample=CFG.up_sample).to(CFG.dtype_pt)
    blk.eval()
    return blk


def select_samples(d, n):
    """Stride evenly through the whole point set, so every record contributes."""
    ntot = len(d['rec_id'])
    return np.unique(np.linspace(0, ntot - 1, n).astype(int))


def jac_phys(blk, x6, u3, theta):
    """J[r, j] = d x_next[r] / d theta_j, physical units."""
    lp = blk.log_params
    z = torch.cat([x6.view(1, NX, 1), u3.view(1, 3, 1)], dim=1)
    xn = blk.nonlinear_function(z).view(NX)
    J = torch.zeros(NX, NP, dtype=CFG.dtype_pt)
    for r in range(NX):
        g = torch.autograd.grad(xn[r], lp, retain_graph=(r < NX - 1))[0]
        J[r] = g
    J = J.detach().numpy()
    return J / theta[None, :]          # d/dlog theta -> d/dtheta


def predicted_nullspace(Lb):
    V = np.zeros((NP, 4))
    V[IX['kb1'], 0], V[IX['kb2'], 0] = 1.0, -1.0
    V[IX['cb1'], 1], V[IX['cb2'], 1] = 1.0, -1.0
    V[IX['Jb'], 2], V[IX['Jh'], 2] = 1.0, -1.0
    V[IX['m1'], 3], V[IX['m2'], 3] = 1.0, 1.0
    V[IX['mb'], 3] = -2.0
    V[IX['Jh'], 3] = -(Lb ** 2) / 2.0
    Q, _ = np.linalg.qr(V)             # orthonormal basis of the predicted kernel
    return Q


def principal_angles(A, B):
    """Angles in degrees between the subspaces spanned by orthonormal A and B."""
    s = np.linalg.svd(A.T @ B, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def main():
    d = load_point_set()
    blk = make_block(d)
    # physical_params() returns a dict; order it by the block's own PARAM_NAMES
    # and assert that order matches this file's PNAMES, so a reordering upstream
    # cannot silently permute the predicted null vectors.
    pp = blk.physical_params()
    assert list(blk.PARAM_NAMES) == PNAMES, (list(blk.PARAM_NAMES), PNAMES)
    theta = np.array([float(pp[n]) for n in PNAMES], dtype=float)
    Lb = float(blk.Lb.item())
    print(f"Lb = {Lb:.6g}")
    print("theta_0 =", np.array2string(theta, precision=4))

    # The block consumes NORMALIZED states and inputs (std_x/x_mean are handed
    # to it at construction), so the point set's X_norm/U_norm feed it directly.
    # This is the frame step1_jacobian_fd.py uses.
    X = torch.as_tensor(np.asarray(d['X_norm']), dtype=CFG.dtype_pt)
    U = torch.as_tensor(np.asarray(d['U_norm']), dtype=CFG.dtype_pt)
    sel = select_samples(d, N_SAMPLES)
    print(f"samples: {len(sel)} of {len(d['rec_id'])}")

    blocks = []
    for i in sel:
        blocks.append(jac_phys(blk, X[i], U[i], theta))
    Phi = np.vstack(blocks)                       # (n_sel*6, 14)
    print(f"Phi shape = {Phi.shape}")

    U_, S, Vt = np.linalg.svd(Phi, full_matrices=True)
    s_rel = S / S[0]
    rank = int(np.sum(s_rel > RANK_TOL_REL))
    print("\nsingular values (relative):")
    for i, v in enumerate(s_rel):
        print(f"  s[{i:2d}] = {v:.6e}")
    print(f"\nrank at {RANK_TOL_REL:g} relative cut: {rank}")

    # --- claim A: factorization -------------------------------------------
    Vp = predicted_nullspace(Lb)
    resid = Phi @ Vp
    scale = np.linalg.norm(Phi, 2)
    relA = np.linalg.norm(resid, 2) / scale
    per_dir = [float(np.linalg.norm(Phi @ Vp[:, k]) / scale) for k in range(4)]
    print("\n[A] FACTORIZATION  ||Phi v|| / ||Phi||_2, per predicted direction:")
    for k, nm in enumerate(["kb1-kb2", "cb1-cb2", "Jb-Jh", "mass-inertia v4"]):
        print(f"  {nm:18s} {per_dir[k]:.6e}")
    print(f"  joint            {relA:.6e}")

    # --- claim B: excitation ----------------------------------------------
    print(f"\n[B] EXCITATION  rank(Phi) = {rank}, expected {NK}")

    # --- do the two subspaces coincide? -----------------------------------
    n_null = NP - rank
    angles = None
    if n_null > 0:
        Vnull = Vt[rank:, :].T                    # (14, n_null), orthonormal
        if n_null == Vp.shape[1]:
            angles = principal_angles(Vnull, Vp)
            print("\nprincipal angles between numerical ker(Phi) and predicted "
                  "span(v1..v4), degrees:")
            print("  " + np.array2string(angles, precision=4))
        else:
            print(f"\nnumerical kernel has dim {n_null}, predicted has 4: "
                  "subspaces not comparable directly")

    res = {
        "Lb": Lb,
        "theta_0": theta.tolist(),
        "param_names": PNAMES,
        "n_samples": int(len(sel)),
        "Phi_shape": list(Phi.shape),
        "singular_values_relative": s_rel.tolist(),
        "rank_tol_relative": RANK_TOL_REL,
        "rank": rank,
        "expected_rank": NK,
        "claim_A_factorization_rel_residual_per_direction": per_dir,
        "claim_A_joint": float(relA),
        "claim_B_rank_matches": bool(rank == NK),
        "kernel_dim_numerical": int(n_null),
        "principal_angles_deg": (angles.tolist() if angles is not None else None),
        "v4_definition": "m1 +1, m2 +1, mb -2, Jh -(Lb^2)/2",
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
