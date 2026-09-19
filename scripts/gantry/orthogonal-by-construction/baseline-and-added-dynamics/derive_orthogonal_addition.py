"""Derive an addition whose contribution is ORTHOGONAL to the baseline on the training data.

STEP 2, the addition's equations of motion. A bank of `N` lossless tuned absorbers rides on the
payload. Absorber `i` has mass `m_i`, natural frequency `w_i`, no damping, and a mounting vector
`b_i` in the logical coordinates `q = [X, Theta, Y]` describing where it is attached and in which
direction it moves. Its relative displacement `d_i` obeys

    d_i''  +  w_i^2 d_i   =   - b_i^T q''                                          (EOM)

and, by Newton's third law through the spring, its reaction on the baseline coordinates is

    f_a  =  SUM_i  b_i * k_i * d_i,        k_i = m_i w_i^2.                        (FORCE)

Two facts make this the right parameterisation, and they are what turn the solve from a nonlinear
optimisation into a linear one:

  * `d_i` depends on `w_i` and `b_i` only, NOT on `m_i`. The mass cancels out of (EOM).
  * `f_a` is therefore LINEAR in the masses.

So with a dictionary of candidate `(w_i, b_i)` fixed, the ten orthogonality conditions are ten
LINEAR equations in the mass vector. Lossless is not a modelling convenience: a dissipative
element can never be orthogonal to the damping family, which has no boundary term and no symmetry
escape, so `zeta = 0` is forced.

STEP 3, the solve. The condition is Gyorok's Condition 4 on the records we generate,

    C x = 0,     C[j, i] = <J_j, Delta_i>,     x = the mass vector, x >= 0,

with `Delta_i` the one-step contribution of a unit-mass absorber `i`, computed through the
production block. Ten equations, `N` unknowns, so the null space is large; the remaining freedom
is spent maximising `||Delta||`, because an addition the augmentation cannot see is useless.

WHAT THIS IS AND IS NOT. The absorbers are driven here by the RECORDED trajectory, so this is the
first iterate of a fixed point: the true coupled plant moves differently once the bank is attached.
That is step 4, regenerating the data and re-solving, and it is not done here. What is proved here
is that a physically meaningful addition satisfying all ten conditions EXISTS and what it is.

Run:
  PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 OBC_REF_STRIDE=1 conda run --no-capture-output \
      -n GraduationProject python -u derive_orthogonal_addition.py
"""
__project_origin__ = "added"

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
for _p in (REPO, os.path.join(REPO, 'scripts', 'gantry')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_augmentation.fit_systems.blocks import (            # noqa: E402
    Parameterized_Gantry_State_Block, Reduced_Gantry_State_Block)
from model_augmentation.fit_systems.obc import (               # noqa: E402
    OBCBasis, build_stacked_sensitivity, evaluate_step_rows)
from model_augmentation.systems import gantry_ss as gss        # noqa: E402
from model_augmentation.systems.gantry_ss import P as P_pt     # noqa: E402

from gantry_dynamic.config import RunConfig                    # noqa: E402
from gantry_dynamic.data import (                              # noqa: E402
    TRAIN_FILES, compute_normalization, load_datasets)
from gantry_dynamic.obc_gantry import (                        # noqa: E402
    build_reference_set, fourth_order_central_difference, _load_record_float64)

COMBO_NAMES = Reduced_Gantry_State_Block.COMBO_NAMES
ROWS = (0, 1, 2, 3, 4, 5)
COMBO_INIT_DETUNE = [1.1, 1.1, 0.9, 1.1, 0.9, 0.9, 1.1, 1.1, 0.9, 1.1]
LINE = '=' * 92

# THEORY: the excitation band of this dataset is [140, 230] Hz (`generate_trajectory_data.m`,
# D-188). Absorber resonances are placed OUTSIDE it, some below and some above, for two reasons
# that are both structural rather than numerical convenience:
#   (a) an undamped absorber driven exactly at its own resonance grows without bound, and the
#       multisine grid here is dense (1/12 s fundamental), so a resonance inside the band cannot
#       reliably be placed between lines;
#   (b) the apparent mass w^2 w_a^2 / (w_a^2 - w^2) is POSITIVE below resonance and NEGATIVE
#       above it, so a bank straddling the band supplies both signs, which is what lets ten
#       conditions be met with nonnegative masses.
F_BELOW = [20.0, 40.0, 70.0, 100.0, 125.0]
F_ABOVE = [255.0, 300.0, 380.0, 520.0, 800.0]

# Mounting vectors. `b = [b_X, b_Theta, b_Y]` maps the absorber's line of action into the logical
# coordinates; `b_Theta` carries units of length and is the moment arm about the yaw axis.
# These are ordinary mechanical choices: move in Y, move in X, move at 45 degrees, and the same
# with a lever arm so the absorber also torques the beam.
#
# The lever arm and the line of action both carry a SIGN: an absorber can be mounted on either
# side of the beam and can act at either inclination. An earlier dictionary used only
# non-negative geometry, which is an arbitrary restriction and which made the problem infeasible
# by Gordan's theorem. Mirrored mountings are the physical fix.
MOUNTS = [
    ('Y at centre',          [0.0, 0.0, 1.0]),
    ('X at centre',          [1.0, 0.0, 0.0]),
    ('yaw only, 1 m arm',    [0.0, 1.0, 0.0]),
    ('Y, +0.10 m lever',     [0.0, +0.10, 1.0]),
    ('Y, -0.10 m lever',     [0.0, -0.10, 1.0]),
    ('X, +0.15 m lever',     [1.0, +0.15, 0.0]),
    ('X, -0.15 m lever',     [1.0, -0.15, 0.0]),
    ('+45 deg X-Y',          [0.7071, 0.0, +0.7071]),
    ('-45 deg X-Y',          [0.7071, 0.0, -0.7071]),
    ('+45 deg, +0.20 lever', [0.7071, +0.20, +0.7071]),
    ('+45 deg, -0.20 lever', [0.7071, -0.20, +0.7071]),
    ('-45 deg, +0.20 lever', [0.7071, +0.20, -0.7071]),
]

# A second mechanism class, structurally different from a tuned absorber and just as physical:
# a COMPLIANT MOUNT, i.e. a spring to ground along a line of action, contributing
# `f_a = -k (b^T q) b` with no internal state. Its signature is in phase with displacement
# rather than following an absorber's apparent-mass curve, so it reaches parts of the condition
# space that no bank of absorbers can. Flexure mounts on a precision stage are exactly this.
SPRING_MOUNTS = MOUNTS


def banner(t):
    print('\n' + LINE); print(t); print(LINE, flush=True)


def truth_combinations():
    names = Parameterized_Gantry_State_Block.PARAM_NAMES
    raw = torch.stack([getattr(gss, n) for n in names]).to(torch.float64)
    return Reduced_Gantry_State_Block.combos_of(raw, float(gss.Lb)), raw


def combo_scale():
    names = Parameterized_Gantry_State_Block.PARAM_NAMES
    raw = {n: float(getattr(gss, n)) for n in names}
    true_combo, _ = truth_combinations()
    scale = np.abs(true_combo.numpy()).copy()
    scale[Reduced_Gantry_State_Block.M_DIFF_IX] = 0.5 * (raw['m1'] + raw['m2'])
    return scale


def absorber_response(drive, w, ts):
    """Solve `d'' + w^2 d = drive` exactly under a zero-order hold on `drive`.

    # THEORY: exact ZOH discretisation of the undamped oscillator. With state [d, d'],
    # A = [[0, 1], [-w^2, 0]] has the closed form exp(A ts) = [[cos, sin/w], [-w sin, cos]] and
    # the input matrix integrates to [(1 - cos)/w^2 ; sin/w]. Standard (Franklin, Powell and
    # Workman, Digital Control of Dynamic Systems, sect. 4.3); used instead of RK4 so the
    # absorber carries no discretisation error of its own into the orthogonality conditions.
    """
    c, s = np.cos(w * ts), np.sin(w * ts)
    Ad = np.array([[c, s / w], [-w * s, c]])
    Bd = np.array([(1.0 - c) / w ** 2, s / w])
    n = len(drive)
    out = np.empty(n)
    x = np.zeros(2)
    for k in range(n):
        out[k] = x[0]
        x = Ad @ x + Bd * drive[k]
    return out


def build_candidate_forces(cfg, ref, stride):
    """Unit-mass absorber reaction forces on the reference tuples, one column per candidate.

    Returns `(F, labels)` with `F` of shape `(n_candidates, N, 3)` in PHYSICAL logical force,
    for `k_i = 1 * w_i^2`, i.e. per unit absorber mass.
    """
    P64 = P_pt.detach().cpu().numpy().astype(np.float64)
    PinvT = np.linalg.inv(P64.T)
    cands = [('absorber', f, nm, np.asarray(b, dtype=np.float64))
             for f in (F_BELOW + F_ABOVE) for nm, b in MOUNTS]
    cands += [('spring', None, nm, np.asarray(b, dtype=np.float64))
              for nm, b in SPRING_MOUNTS]
    labels = ['%-9s %7s %s' % (kind, ('%.1f Hz' % f) if f else '   -', nm)
              for kind, f, nm, _b in cands]
    per_record, t0 = [], time.perf_counter()
    for fname in TRAIN_FILES:
        u, y, _x, _a = _load_record_float64(fname, cfg)
        q = y @ PinvT.T
        qd = fourth_order_central_difference(q, cfg.ts_new)        # aligned q[2:-2]
        qdd = fourth_order_central_difference(qd, cfg.ts_new)      # aligned q[4:-4]
        k = np.arange(2, len(u) - 2)
        if stride > 1:
            k = k[::stride]
        # qdd exists for absolute indices 4 .. len-5; pad the two ends of the tuple window with
        # the nearest valid value so every reference tuple has a drive. The affected samples are
        # 4 per record out of ~48000 and they are inside the initial hold, where qdd ~ 0.
        idx = np.clip(k - 4, 0, len(qdd) - 1)
        rows = []
        for kind, w_hz, _nm, b in cands:
            if kind == 'absorber':
                w = 2.0 * np.pi * w_hz
                drive_full = -(qdd @ b)                            # -b^T qddot over the record
                d_full = absorber_response(drive_full, w, cfg.ts_new)
                # f_a = b * k * d with k = w^2 for unit mass
                rows.append(np.outer(d_full[idx] * w ** 2, b))     # (n_k, 3)
            else:
                # Compliant mount: f_a = -k (b^T q) b, unit stiffness, no internal state.
                rows.append(np.outer(-(q[k] @ b), b))
        per_record.append(np.stack(rows, axis=0))                  # (n_cand, n_k, 3)
    F = np.concatenate(per_record, axis=1)
    assert F.shape[1] == ref.n, 'candidate forces lost alignment with the reference set'
    print('[cand] %d candidates x %d tuples, %.1f s'
          % (F.shape[0], F.shape[1], time.perf_counter() - t0), flush=True)
    return F, labels


def main():
    stride = int(os.environ.get('OBC_REF_STRIDE', '1'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = RunConfig(
        mode='augmentation_ma50_b140-230_a6_z03', encoder_init='linear_map',
        joint_estimation=True, physics_parameterization='reduced',
        combo_init_detune=COMBO_INIT_DETUNE, param_init_detune=None,
        obc=True, obc_space='tangent', nx_ann=8, up_sample=1, stride=10,
        fs_new=4000, snr=None, save_flag=False)

    banner('SETUP (stride %d, device %s)' % (stride, device))
    data = load_datasets(cfg)
    norm = compute_normalization(cfg, data)
    ref = build_reference_set(cfg, norm, nx_ann=cfg.nx_ann, stride=stride, verbose=True)

    true_combo, raw_nominal = truth_combinations()
    combo_init = true_combo * torch.tensor(COMBO_INIT_DETUNE, dtype=torch.float64)
    blk = Reduced_Gantry_State_Block(
        params_init=raw_nominal, combo_init=combo_init, Y_op=None,
        std_x=norm.std_x, std_u=norm.std_u, x_mean=norm.x_mean, u_mean=norm.u_mean,
        Ts=cfg.ts_new, up_sample=cfg.up_sample, RMSE_baseline=cfg.param_rmse_baseline,
        flag_loss_reg=False).to(torch.float64)
    vbar = torch.log(true_combo / combo_init)
    vbar[Reduced_Gantry_State_Block.M_DIFF_IX] = (
        true_combo[Reduced_Gantry_State_Block.M_DIFF_IX]
        / combo_init[Reduced_Gantry_State_Block.M_DIFF_IX] - 1.0)
    blk.to(device)

    banner('STEP 2  the addition: a bank of lossless tuned absorbers')
    print("  d_i'' + w_i^2 d_i = -b_i^T q'',      f_a = SUM_i b_i k_i d_i,   k_i = m_i w_i^2")
    print('  zeta = 0 is FORCED: a dissipative element can never be orthogonal to the damping')
    print('  family (that condition is a positive quadratic form, with no boundary term and no')
    print('  symmetry escape). d_i is independent of m_i, so f_a is LINEAR in the masses.')
    print('  resonances below the band: %s Hz' % F_BELOW)
    print('  resonances above the band: %s Hz' % F_ABOVE)
    print('  mountings: %s' % [nm for nm, _ in MOUNTS])

    F, labels = build_candidate_forces(cfg, ref, stride)
    n_cand = F.shape[0]

    banner('STEP 3a  each candidate through the PRODUCTION block')
    print('  The one-step map is affine in u, so a candidate\'s contribution is')
    print('  Delta_i = step(x, u + P^-1 f_a_i) - step(x, u), evaluated by the same')
    print('  `evaluate_step_rows` the basis is built with. No second integrator.\n')
    P64 = P_pt.detach().cpu().numpy().astype(np.float64)
    Pinv = np.linalg.inv(P64)
    std_u = np.asarray(norm.std_u, np.float64).reshape(1, 3)

    step = lambda v, z: blk.transition_from_free(v, z)             # noqa: E731
    base = evaluate_step_rows(step, vbar.to(device), ref.Z_step, ROWS,
                              chunk=cfg.obc_ref_chunk, device=device).cpu()
    Deltas = []
    t0 = time.perf_counter()
    for i in range(n_cand):
        du_stage = F[i] @ Pinv.T                                   # logical -> stage force
        du_norm = torch.from_numpy(du_stage / std_u).to(torch.float64)
        Z = ref.Z_step.clone().to(torch.float64)
        Z[:, 6:9, 0] += du_norm
        di = evaluate_step_rows(step, vbar.to(device), Z, ROWS,
                                chunk=cfg.obc_ref_chunk, device=device).cpu() - base
        Deltas.append(di)
        del Z
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    print('  %d candidate fields in %.1f s' % (n_cand, time.perf_counter() - t0), flush=True)

    banner('STEP 3b  the basis, and the ten conditions')
    J = build_stacked_sensitivity(step, vbar.to(device), ref.Z_step, ROWS,
                                  chunk=cfg.obc_ref_chunk, device=device)
    basis = OBCBasis.from_matrix(J.cpu(), vbar.cpu())
    del J
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    for line in basis.report.lines(COMBO_NAMES):
        print('  ' + line)

    C = np.zeros((10, n_cand))
    norms = np.zeros(n_cand)
    for i, di in enumerate(Deltas):
        C[:, i] = basis.jt(di).numpy()
        norms[i] = float(torch.linalg.vector_norm(di))
    print('\n  condition matrix C is %s; column norms of Delta_i span %.3e to %.3e'
          % (C.shape, norms.min(), norms.max()))

    # DIAGNOSTIC, and it decides whether a nonnegative solution can exist at all. `C x = 0` with
    # `x >= 0` has only the trivial solution as soon as ONE row of C has all its entries of the
    # same sign: a sum of nonnegative multiples of same-sign numbers cannot vanish. So the sign
    # pattern, not the rank, is what governs feasibility here.
    print('\n  sign structure of C, which governs whether x >= 0 can work at all:')
    print('  %-9s %8s %8s %12s %12s' % ('condition', '#pos', '#neg', 'max |C|', 'row norm'))
    blocked = []
    Cn = C / np.maximum(norms[None, :], 1e-300)          # scale out the candidate magnitudes
    for j, n in enumerate(COMBO_NAMES):
        npos = int((Cn[j] > 1e-12 * np.abs(Cn[j]).max()).sum())
        nneg = int((Cn[j] < -1e-12 * np.abs(Cn[j]).max()).sum())
        flag = ''
        if npos == 0 or nneg == 0:
            flag = '  <== SINGLE SIGN, blocks any nonnegative solution'
            blocked.append(n)
        print('  %-9s %8d %8d %12.3e %12.3e%s'
              % (n, npos, nneg, np.abs(C[j]).max(), np.linalg.norm(C[j]), flag))
    if blocked:
        print('\n  Conditions with a single sign across the whole dictionary: %s' % blocked)
        print('  Every absorber pushes these the SAME way, so no bank of them, at any masses,')
        print('  can cancel them. The dictionary must contain a mechanism of the opposite sign')
        print('  before the LP can succeed.')

    banner('STEP 3c  solve  C x = 0,  x >= 0,  maximising the size of the addition')
    from scipy.optimize import linprog
    # The ten rows of C span four orders of magnitude, so equilibrate them before any LP: an
    # unscaled equality system lets the solver satisfy the large rows and treat the small ones
    # as noise. Scaling rows does not change the solution set of C x = 0.
    rs = np.linalg.norm(C, axis=1)
    Cs = C / rs[:, None]

    # GORDAN'S THEOREM, run FIRST, because it settles feasibility exactly rather than by
    # inference from a solver returning the trivial point. Exactly one of these holds:
    #   (i)  there is x >= 0, x != 0, with C x = 0;
    #   (ii) there is y with C^T y > 0 componentwise, i.e. a hyperplane with EVERY candidate
    #        strictly on one side.
    # If (ii) holds, no bank of these absorbers at any nonnegative masses can be orthogonal,
    # and the certificate y says which combination of the ten conditions they all push the
    # same way. That is a physical statement, not a numerical one.
    gor = linprog(c=np.zeros(10), A_ub=-Cs.T, b_ub=-np.ones(n_cand),
                  bounds=[(None, None)] * 10, method='highs')
    if gor.success:
        y = gor.x
        print('  GORDAN CERTIFICATE FOUND: every candidate has C^T y > 0 for')
        print('    y = %s' % np.array2string(y, precision=4, suppress_small=False))
        print('  so no NONNEGATIVE bank of these absorbers can satisfy the ten conditions.')
        print('  The certificate weights the conditions as:')
        for n, v in sorted(zip(COMBO_NAMES, y), key=lambda t: -abs(t[1]))[:5]:
            print('    %-9s %+.4f' % (n, v))
        print('  Read it as: every tuned absorber in this dictionary moves that particular')
        print('  combination of the ten conditions in one direction, so they cannot cancel.')
        print('  A mechanism of the OPPOSITE sign is needed, not more of the same.')
        return
    print('  No Gordan certificate: a nonnegative solution exists. Solving for it.')

    # MINIMUM ADDED MASS for a given size of addition. Maximising the size instead gives a
    # mathematically valid but mechanically absurd answer: the LP saturates every box constraint
    # and returns dozens of absorbers totalling several times the payload mass. Minimising the
    # total added mass subject to a FIXED size is the right objective, and because the optimum
    # of an LP sits at a vertex the solution is automatically sparse: at most eleven absorbers,
    # one per equality constraint plus the scale.
    A_eq = np.vstack([Cs, norms[None, :]])
    b_eq = np.concatenate([np.zeros(10), [1.0]])
    res = linprog(c=np.ones(n_cand), A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0.0, None)] * n_cand, method='highs')
    if not res.success:
        print('  LP failed: %s' % res.message)
        return
    x = res.x
    if x.max() <= 0:
        print('  LP returned the trivial x = 0 despite no Gordan certificate: numerical.')
        return
    # Rescale to a chosen physical size: put the total added mass at 10 percent of the payload,
    # matching the `ma_frac = 0.10` family this project already generates, so the result is
    # comparable to an existing dataset rather than to an arbitrary scale.
    target_mass = 0.10 * float(gss.mh)
    x = x * (target_mass / x.sum())
    print('  minimum-mass solution, rescaled to a total added mass of %.3f kg' % target_mass)
    active = np.where(x > 1e-9 * max(x.max(), 1e-30))[0]
    print('  LP solved. %d of %d candidates active.\n' % (len(active), n_cand))
    print('  %-26s %14s %14s' % ('absorber', 'mass [kg]', '||Delta_i||'))
    for i in active:
        print('  %-26s %14.6f %14.4e' % (labels[i], x[i], norms[i]))

    Delta = torch.zeros_like(Deltas[0])
    for i in active:
        Delta = Delta + float(x[i]) * Deltas[i]

    banner('VERIFY  the constructed addition against the production inner product')
    rho = basis.r_overlap(Delta)
    resid = basis.jt(Delta).numpy()
    dtheta = basis.coefficient(Delta)
    scale = combo_scale()
    free_hat = vbar.cpu() + dtheta
    combo_hat = blk.combinations_from_free(
        free_hat.to(blk.combo_init.device)).detach().cpu().numpy()
    pct = 100.0 * (combo_hat - true_combo.numpy()) / scale
    rms = float(np.sqrt((pct ** 2).mean()))
    print('  ||Delta||            = %.6e' % float(torch.linalg.vector_norm(Delta)))
    print('  rho* = ||P D||/||D|| = %.6e        (the current absorber gives 2.05e-01)' % rho)
    print('  ||J^T Delta||        = %.6e' % float(np.linalg.norm(resid)))
    print('  predicted combo-err  = %.6e %%      (the current absorber predicts 4.96 %%)' % rms)
    print('\n  per-parameter predicted bias, which should be numerically zero:')
    for n, p in zip(COMBO_NAMES, pct):
        print('    %-9s %+.3e %%' % (n, p))

    ok = rho < 1e-8
    print('\n  VERDICT: %s' % ('the addition IS orthogonal on this data, to numerical precision'
                               if ok else 'NOT orthogonal; inspect the residual above'))

    with open(os.path.join(HERE, 'orthogonal_addition.json'), 'w') as fh:
        json.dump(dict(stride=stride, n_reference=ref.n, candidates=labels,
                       masses=x.tolist(), active=[int(i) for i in active],
                       delta_norm=float(torch.linalg.vector_norm(Delta)),
                       rho=float(rho), residual=resid.tolist(),
                       predicted_pct=pct.tolist(), predicted_rms=rms,
                       f_below=F_BELOW, f_above=F_ABOVE,
                       mounts=[[nm, b] for nm, b in MOUNTS]), fh, indent=2)
    print('\n[out] wrote %s' % os.path.join(HERE, 'orthogonal_addition.json'))
    print('\n  NOT YET DONE (step 4): the absorbers were driven by the RECORDED trajectory, so')
    print('  this is the first iterate of a fixed point. Regenerating the data with the bank')
    print('  attached moves q, hence C, hence x. Iterate until the masses stop moving.')


if __name__ == '__main__':
    main()
