"""The geometry the gantry cannot give us: S, R, E explicitly, and the angles between them.

Category: DEVELOPMENT AND FALSIFICATION TOOL (see system.py). Not evidence about the gantry.

WHY THIS EXISTS. gaps.md Section 9 separates two objects that the deployed penalty conflates:

    the VALUE    d = the augmentation's current contribution to the stacked output
    the TANGENT  R = d yhat / d eta, the directions the network can MOVE in during training

Our penalty constrains the value. The non-substitution propositions are about the tangent. A
contribution can be orthogonal at one setting while its derivatives point straight into the
physical sensitivity space, so the two are genuinely different and connecting them is the open
proof task. At gantry scale `R` cannot be formed. Here it can, so this is the one place the
question is directly answerable.

Three parameter groups, matching Section 9's `dY = S dkappa + R deta + E dxi`:

    theta  physical, 4 raw / 3 identifiable          -> S
    eta    augmentation network weights              -> R
    xi     encoder weights (x0 from past u, y)       -> E

Everything is built in the WEIGHTED OUTPUT SPACE, not state space: substitution happens in the
loss, so the subspace belongs where the loss lives. `S` here is `d yhat / d theta`, and the
encoder is profiled out with `M_D = I - D D^+` before any angle is measured, per Section 9.3.

Every ratio is reported against a NULL. `span(S)` is 3-dimensional inside an ambient space of
`n_win * T`, so an unrelated vector already projects at `sqrt(3 / ambient)`. Ratios without
that baseline are meaningless; this is the lesson from leakage_measure.py.

Run: conda run -n GraduationProject python scripts/gantry/orthogonality/testbed/geometry.py
"""
__project_origin__ = "added"

import os
import sys
import json

import numpy as np
import torch
from torch.func import jacfwd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import system as S  # noqa: E402

F64 = torch.float64
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'geometry.json')

NX, NA, NU, NY = 2, 2, 1, 1          # physical / augmented / input / output
N_WIN, T_WIN = 48, 50                # windows are BATCHED and nearly free; steps are not.
                                     # 50 steps at 50 Hz = 1 s = 1.6 periods of the main mode.
N_LAG = 8                            # encoder lag: x0 from N_LAG past (u, y)
N_HID = 8                            # augmentation hidden width, 2 layers
RTOL = 1e-12


# ------------------------------------------------------------------ parameter packing
def init_eta(seed=0):
    """Small tanh MLP: [x_phys; x_aug; u] -> 4 rows (2 correct x_phys, 2 become x_aug)."""
    g = torch.Generator().manual_seed(seed)
    n_in, n_out = NX + NA + NU, NX + NA
    shapes = [(N_HID, n_in), (N_HID,), (N_HID, N_HID), (N_HID,), (n_out, N_HID), (n_out,)]
    flat = []
    for sh in shapes:
        fan = sh[-1] if len(sh) > 1 else N_HID
        flat.append(torch.randn(sh, generator=g, dtype=F64).reshape(-1) / np.sqrt(fan))
    return torch.cat(flat), shapes


def unpack(flat, shapes):
    out, i = [], 0
    for sh in shapes:
        n = int(np.prod(sh))
        out.append(flat[i:i + n].reshape(sh))
        i += n
    return out


def ann(z, eta, shapes, scale=None):
    """Rows 0..NX-1 are the physical correction, rows NX.. become the latent state.

    The two blocks get DIFFERENT scales. With a single scale eps on both, the latent is
    written at O(eps) and read back into the physical correction at O(eps), so the
    additional-state path starts at O(eps^2) against O(eps) for the direct correction. At
    eps = 1e-3 that suppressed the very path this testbed exists to study: measured
    ||d_state||/||d|| = 2.5e-4. Scaling the latent block at O(1) puts both paths at O(eps).
    """
    W1, b1, W2, b2, W3, b3 = unpack(eta, shapes)
    h = torch.tanh(W1 @ z + b1)
    h = torch.tanh(W2 @ h + b2)
    w = W3 @ h + b3
    return torch.cat([SCALE_PHYS * w[:NX], SCALE_LAT * w[NX:]])


# ------------------------------------------------------------------ the predictor
def predictor(theta, eta, xi, packs, shapes):
    """Stacked output over all windows. Pure function of the three parameter groups.

    packs: list of (u_win, lag_u, lag_y) per window. The encoder maps the lag vector to x0,
    which is what makes `E` nonzero and the encoder channel testable.
    """
    Ad, Bd = S.baseline_dt(theta)
    Wenc, benc = xi[:NX * 2 * N_LAG].reshape(NX, 2 * N_LAG), xi[NX * 2 * N_LAG:]
    ys = []
    for (u_win, lag) in packs:
        x = Wenc @ lag + benc                       # encoder sets x0  -> the E channel
        xa = torch.zeros(NA, dtype=F64)
        yw = []
        for k in range(u_win.shape[0]):
            yw.append(x[0])
            z = torch.cat([x, xa, u_win[k:k + 1]])
            w = ann(z, eta, shapes, ANN_SCALE)
            xn = Ad @ x + Bd @ u_win[k:k + 1] + w[:NX]
            xa = w[NX:]
            x = xn
        ys.append(torch.stack(yw))
    return torch.cat(ys)


SCALE_PHYS = 1e-3     # physical-correction rows: keeps the contribution a perturbation
SCALE_LAT = 1.0       # latent rows: O(1), so the state path is O(eps), not O(eps^2)
ANN_SCALE = SCALE_PHYS  # retained name for callers that scale the physical block only


def principal_cosines(Qa, Qb):
    """Cosines of the principal angles between two orthonormal ranges."""
    return np.linalg.svd(Qa.T @ Qb, compute_uv=False)


def orth(Mx, rtol=RTOL):
    U, s, _ = np.linalg.svd(Mx, full_matrices=False)
    r = int((s > rtol * s[0]).sum())
    return U[:, :r], s, r


def main():
    torch.set_default_dtype(F64)
    print('=' * 78)
    print('TESTBED GEOMETRY.  Category: development tool, not gantry evidence.')
    print('=' * 78)

    d = S.make_dataset(seed=0)
    u, y, delta = d['u'], d['y'], d['delta']

    rng = np.random.default_rng(0)
    starts = rng.choice(np.arange(N_LAG, len(u) - T_WIN - 1), size=N_WIN, replace=False)
    packs, dstack = [], []
    for k0 in starts:
        u_win = torch.tensor(u[k0:k0 + T_WIN], dtype=F64)
        lag = torch.tensor(np.concatenate([u[k0 - N_LAG:k0], y[k0 - N_LAG:k0]]), dtype=F64)
        packs.append((u_win, lag))
        dstack.append(delta[k0:k0 + T_WIN])
    dstack = np.concatenate(dstack)
    amb = N_WIN * T_WIN
    print(f'  {N_WIN} windows x {T_WIN} steps -> ambient dimension {amb}')

    theta = torch.tensor(S.THETA_TRUE, dtype=F64)
    eta, shapes = init_eta(0)
    xi = torch.zeros(NX * 2 * N_LAG + NX, dtype=F64)
    # Least-squares encoder init, so x0 is sensible rather than zero.
    L = torch.stack([p[1] for p in packs])
    Xb_all = S.baseline_states(u)                       # BUG FIX: true state at each k0
    X0 = torch.tensor(Xb_all[starts], dtype=F64)
    Wls = torch.linalg.lstsq(L, X0).solution.T
    xi[:NX * 2 * N_LAG] = Wls.reshape(-1)

    f = lambda th, et, x_: predictor(th, et, x_, packs, shapes)
    print('  building S, R, E by forward-mode autodiff ...', flush=True)
    Smat = jacfwd(f, argnums=0)(theta, eta, xi).detach().numpy()
    Rmat = jacfwd(f, argnums=1)(theta, eta, xi).detach().numpy()
    Emat = jacfwd(f, argnums=2)(theta, eta, xi).detach().numpy()
    print(f'  S {Smat.shape}   R {Rmat.shape}   E {Emat.shape}')

    res = {'category': 'development tool; testbed only, not gantry evidence',
           'ambient': amb, 'shapes': dict(S=list(Smat.shape), R=list(Rmat.shape),
                                          E=list(Emat.shape))}

    # ---------------- 1. the physical sensitivity, and the two pathologies -------------
    print('\n--- 1. physical sensitivity S = d yhat / d theta, in output space ---')
    QS, sS, rS = orth(Smat)
    print(f'  singular values : {"  ".join(f"{v:.3e}" for v in sS)}')
    print(f'  numerical rank  : {rS} of 4     condition (kept) {sS[0]/sS[rS-1]:.2e}')
    vnull = np.array([0.0, 0.0, 1.0, -1.0])
    print(f'  ||S v_null|| / ||S|| = {np.linalg.norm(Smat @ vnull)/np.linalg.norm(Smat):.3e}'
          '   <- the exact ka/kb null direction survives into output space')
    print('  READ: rank 3 of 4 with a graded spectrum. Both pathologies present, as designed.')
    res['S'] = dict(sv=[float(v) for v in sS], rank=rS,
                    null_residual=float(np.linalg.norm(Smat @ vnull) / np.linalg.norm(Smat)))

    # ---------------- 2. the recovery floor, computable ONLY here ----------------------
    print('\n--- 2. recovery floor (needs the truth; category 3, validation only) ---')
    floor = np.linalg.norm(QS.T @ dstack) / np.linalg.norm(dstack)
    chance = np.sqrt(rS / amb)
    print(f'  ||Q_S^T delta|| / ||delta|| = {floor:.4f}   (chance {chance:.4f})')
    print('  This is how much of the REAL missing physics a parameter change could mimic.')
    print('  It bounds recovery. Per gaps.md Sect. 0 it is NOT an argument against the method.')
    res['recovery_floor'] = dict(value=float(floor), chance=float(chance))

    # ---------------- 3. encoder channel, and profiling it out ------------------------
    print('\n--- 3. encoder channel E (Section 9.3 profiling) ---')
    QE, sE, rE = orth(Emat)
    print(f'  rank(E) = {rE} of {Emat.shape[1]}')
    cos_SE = principal_cosines(QS, QE)
    print(f'  principal cosines S vs E: {"  ".join(f"{c:.4f}" for c in cos_SE)}')
    MD = np.eye(amb) - QE @ QE.T
    QSb, sSb, rSb = orth(MD @ Smat)
    print(f'  after profiling the encoder out: rank(S_bar) = {rSb},  '
          f'||S_bar||/||S|| = {np.linalg.norm(MD@Smat)/np.linalg.norm(Smat):.4f}')
    if rSb < rS:
        print('  WARNING: the encoder absorbs a physical direction outright.')
    res['E'] = dict(rank=rE, cos_S_E=[float(c) for c in cos_SE], rank_S_after=rSb)

    # ---------------- 4. THE POINT: value versus tangent -------------------------------
    print('\n--- 4. value versus tangent (the open question in gaps.md Sect. 9.4) ---')
    dvec = f(theta, eta, xi).detach().numpy() - f(theta, eta * 0, xi).detach().numpy()
    r_val = np.linalg.norm(QSb.T @ (MD @ dvec)) / np.linalg.norm(MD @ dvec)
    QRb, sR, rR = orth(MD @ Rmat)
    cos_SR = principal_cosines(QSb, QRb)
    # Null for the tangent overlap: a random subspace of the same dimension.
    nulls = []
    g = np.random.default_rng(1)
    for _ in range(50):
        Z = g.standard_normal((amb, rR))
        Qz, _ = np.linalg.qr(Z)
        nulls.append(principal_cosines(QSb, Qz).max())
    nulls = np.array(nulls)
    print(f'  VALUE   ||Q_S^T d|| / ||d||        = {r_val:.4f}   '
          f'(chance {np.sqrt(rSb/amb):.4f})')
    print(f'  TANGENT rank(R_bar) = {rR} of {Rmat.shape[1]}')
    print(f'          principal cosines S vs R  : {"  ".join(f"{c:.6f}" for c in cos_SR)}')
    print(f'          1 - cos (how far from exact containment): '
          f'{"  ".join(f"{1-c:.2e}" for c in cos_SR)}')
    print(f'          R_bar singular values kept: {sR[0]:.3e} .. {sR[rR-1]:.3e}  '
          f'(rtol {RTOL:.0e}); next {sR[rR]:.3e}' if rR < len(sR) else '')
    print(f'          largest cosine            = {cos_SR.max():.4f}   '
          f'null {nulls.mean():.4f} (p95 {np.percentile(nulls,95):.4f})')
    print('  READ: the value ratio and the tangent overlap are different numbers measured on')
    print('  the SAME network. A penalty driving the first to zero need not move the second,')
    print('  which is exactly Proposition 2. This testbed can now settle that empirically.')
    res['value_vs_tangent'] = dict(
        r_value=float(r_val), value_chance=float(np.sqrt(rSb / amb)),
        rank_R=int(rR), cos_S_R=[float(c) for c in cos_SR],
        max_cos=float(cos_SR.max()), null_mean=float(nulls.mean()),
        null_p95=float(np.percentile(nulls, 95)))

    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=2)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
