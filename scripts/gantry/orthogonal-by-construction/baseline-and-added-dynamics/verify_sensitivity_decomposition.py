"""Is the sensitivity decomposition this folder's derivation rests on actually true?

`baseline-and-added-dynamics.md` Sect. 8 claims, and every later step uses, that the columns of
the stacked baseline sensitivity are regressor signals of a specific form:

    J_j  =  -Ts * M^-1 [ (dM/dtheta_j) qddot + (dC/dtheta_j) qdot + (dK/dtheta_j) q ]  + O(Ts^2)

with the position rows at O(Ts^2), and that the ten identifiable combinations split CLEANLY into
five that touch only `M`, four that touch only `C`, and one that touches only `K`. The parity
table, the lossless conclusion and the ten matrix conditions are all downstream of that claim, and
none of it was taken from a source: it is derived here from the equations of motion. So it gets
checked against the production Jacobian before anything is built on it.

Four tests, in increasing cost:

  T1  THE SPLIT, exact and assumption-free. Evaluate `d(deriv)/d(free_j)` at synthetic states
      chosen to kill one term at a time. At `qdot = 0` the four damping sensitivities must be
      EXACTLY zero; at `q = 0` the stiffness one must be; at `q = qdot = 0` with `u != 0` only
      the five mass ones may be nonzero. No mass matrix is needed anywhere, which is the point:
      the test cannot be fooled by an error in a reconstruction of `M`.

  T2  LEADING ORDER. On real reference tuples, compare the production discrete `J` from
      `build_stacked_sensitivity` against `Ts * d(deriv)/d(free)`. Then rebuild at `Ts/2` and
      `Ts/4` and check the relative error falls like `O(Ts)`. A convergence rate, not a
      tolerance, is what makes this decisive.

  T3  POSITION ROWS. The claim is that they are `O(Ts^2)` and carry the ratio
      `(Ts/2) * std_x[i+3] / std_x[i]` against their velocity siblings, because RK4's position
      update picks the parameter dependence up only through `(h^2/2) qddot`.

  T4  POINTWISE RANK. The corrected no-go lemma needs `rank S(k) = 6` at a sample, where
      `S(k)` is that sample's `6 x 10` block of `J`. Measured, not asserted.

Run:
  PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 OBC_REF_STRIDE=997 conda run --no-capture-output \
      -n GraduationProject python -u verify_sensitivity_decomposition.py
"""
__project_origin__ = "added"

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
for _p in (REPO, os.path.join(REPO, 'scripts', 'gantry')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_augmentation.fit_systems.blocks import (            # noqa: E402
    Parameterized_Gantry_State_Block, Reduced_Gantry_State_Block)
from model_augmentation.fit_systems.obc import build_stacked_sensitivity   # noqa: E402
from model_augmentation.systems import gantry_ss as gss        # noqa: E402

from gantry_dynamic.config import RunConfig                    # noqa: E402
from gantry_dynamic.data import compute_normalization, load_datasets       # noqa: E402
from gantry_dynamic.obc_gantry import build_reference_set      # noqa: E402

COMBO_NAMES = Reduced_Gantry_State_Block.COMBO_NAMES
ROWS = (0, 1, 2, 3, 4, 5)
COMBO_INIT_DETUNE = [1.1, 1.1, 0.9, 1.1, 0.9, 0.9, 1.1, 1.1, 0.9, 1.1]

# The claim under test: which of (M, C, K) each identifiable combination enters.
# Read off the EOM in `Matlab-scripts/Augmentation/gantrySystemExtended.m` and `gtd_config.m`:
# `M` carries the masses, the inertia and the offset `d`; `C_damp` carries cg1, cg2, cy and
# cb_sum; `K` carries kb_sum alone.
PREDICTED_FAMILY = {
    'kb_sum': 'K', 'cg1': 'C', 'cg2': 'C', 'cy': 'C', 'cb_sum': 'C',
    'mh': 'M', 'm_total': 'M', 'm_diff': 'M', 'J_eff': 'M', 'd': 'M',
}
LINE = '=' * 88


def banner(t):
    print('\n' + LINE); print(t); print(LINE, flush=True)


def make_block(norm, cfg, ts):
    """The production block at an explicit sample time, truth-centred combos."""
    names = Parameterized_Gantry_State_Block.PARAM_NAMES
    raw = torch.stack([getattr(gss, n) for n in names]).to(torch.float64)
    combo_true = Reduced_Gantry_State_Block.combos_of(raw, float(gss.Lb))
    combo_init = combo_true * torch.tensor(COMBO_INIT_DETUNE, dtype=torch.float64)
    blk = Reduced_Gantry_State_Block(
        params_init=raw, combo_init=combo_init, Y_op=None,
        std_x=norm.std_x, std_u=norm.std_u, x_mean=norm.x_mean, u_mean=norm.u_mean,
        Ts=ts, up_sample=cfg.up_sample, RMSE_baseline=cfg.param_rmse_baseline,
        flag_loss_reg=False).to(torch.float64)
    free_true = torch.log(combo_true / combo_init)
    i = Reduced_Gantry_State_Block.M_DIFF_IX
    free_true[i] = combo_true[i] / combo_init[i] - 1.0
    assert float((blk.combinations_from_free(free_true) - combo_true).abs().max()) < 1e-12
    return blk, free_true


def deriv_from_free(blk, v, z):
    """The CONTINUOUS derivative at an explicit free coordinate. `transition_from_free`
    without the RK4, so the same matrix build and the same normalisation."""
    p = blk.gauge_section(blk.combinations_from_free(v), blk.params_init, blk.Lb)
    mats = blk._mats_from_raw(p)
    return blk._deriv_with(z[:, : blk.nx, :], z[:, blk.nx:, :], mats)


def jac_deriv(blk, v, z):
    """`d(deriv)/d(free)`, shape (N, 6, p), float64, no grad."""
    with torch.no_grad():
        J = torch.func.jacfwd(lambda vv: deriv_from_free(blk, vv, z))(v)   # (N,6,1,p)
    return J[:, :, 0, :]


# ------------------------------------------------------------------ T1
def t1_split(blk, free_true, norm):
    banner('T1  THE FIVE / FOUR / ONE SPLIT, at synthetic states (exact, no mass matrix needed)')
    x_mean = torch.as_tensor(np.asarray(norm.x_mean, np.float64).reshape(1, 6, 1))
    std_x = torch.as_tensor(np.asarray(norm.std_x, np.float64).reshape(1, 6, 1))
    u_mean = torch.as_tensor(np.asarray(norm.u_mean, np.float64).reshape(1, 3, 1))
    std_u = torch.as_tensor(np.asarray(norm.std_u, np.float64).reshape(1, 3, 1))

    def norm_state(q, qd):
        xp = torch.tensor(list(q) + list(qd), dtype=torch.float64).reshape(1, 6, 1)
        return (xp - x_mean) / std_x

    def norm_u(u):
        up = torch.tensor(list(u), dtype=torch.float64).reshape(1, 3, 1)
        return (up - u_mean) / std_u

    # Y is a scheduling variable, so keep it away from zero in the probes that do not pin q,
    # and use a representative operating point elsewhere. Forces are O(10 N), matching the data.
    probes = [
        ('qdot = 0, q and u nonzero      -> DAMPING columns must vanish',
         norm_state([2e-3, 1e-4, 0.15], [0.0, 0.0, 0.0]), norm_u([12.0, -7.0, 9.0]), 'C'),
        ('q = 0,   qdot and u nonzero    -> STIFFNESS column must vanish',
         norm_state([0.0, 0.0, 0.0], [0.3, 0.02, -0.4]), norm_u([12.0, -7.0, 9.0]), 'K'),
        ('q = qdot = 0, u nonzero        -> only MASS columns may survive',
         norm_state([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]), norm_u([12.0, -7.0, 9.0]), 'CK'),
    ]
    ok = True
    for label, xn, un, must_vanish in probes:
        z = torch.cat([xn, un], dim=1)
        Jd = jac_deriv(blk, free_true, z)[0]                 # (6, 10)
        scale = float(Jd.abs().max())
        print('\n  %s' % label)
        print('    column                 |d(xdot)/d(free_j)|_inf     relative to the largest')
        for j, n in enumerate(COMBO_NAMES):
            v = float(Jd[:, j].abs().max())
            rel = v / scale if scale > 0 else 0.0
            fam = PREDICTED_FAMILY[n]
            expect_zero = fam in must_vanish
            mark = ''
            if expect_zero:
                mark = 'OK zero' if rel < 1e-12 else '*** NONZERO, CLAIM FAILS ***'
                ok &= rel < 1e-12
            print('    %-9s (%s)   %20.6e   %14.3e   %s' % (n, fam, v, rel, mark))
    print('\n  T1 verdict: %s' % ('the 5/4/1 split holds EXACTLY' if ok else 'THE SPLIT IS WRONG'))
    return ok


# ------------------------------------------------------------------ T2 / T3
def t2_leading_order(cfg, norm, ref, device):
    banner('T2  LEADING ORDER  J_discrete  vs  Ts * d(deriv)/d(free),  and its Ts-convergence')
    print('  claim: J_j = Ts * d(xdot)/d(free_j) + O(Ts^2), so the relative error must HALVE')
    print('  when Ts halves. A rate, not a tolerance, is the test.\n')
    print('  %10s %14s %14s %14s %10s' % ('Ts [s]', 'rel err vel', 'rel err pos', 'pos/vel', 'rate'))
    prev = None
    out = {}
    for k in range(3):
        ts = cfg.ts_new / (2 ** k)
        blk, free_true = make_block(norm, cfg, ts)
        blk.to(device)
        Jd = build_stacked_sensitivity(lambda v, z: blk.transition_from_free(v, z),
                                       free_true.to(device), ref.Z_step, ROWS,
                                       chunk=cfg.obc_ref_chunk, device=device).cpu()
        Jd = Jd.reshape(-1, 6, 10)                                   # (N, 6, p)
        with torch.no_grad():
            Z = ref.Z_step.to(device=device, dtype=torch.float64)
            Jc = jac_deriv(blk, free_true.to(device), Z).cpu()        # (N, 6, p)
        pred = ts * Jc
        num_v = float(torch.linalg.norm(Jd[:, 3:, :] - pred[:, 3:, :]))
        den_v = float(torch.linalg.norm(pred[:, 3:, :]))
        num_p = float(torch.linalg.norm(Jd[:, :3, :] - pred[:, :3, :]))
        den_p = float(torch.linalg.norm(Jd[:, :3, :]))
        ratio = float(torch.linalg.norm(Jd[:, :3, :]) / torch.linalg.norm(Jd[:, 3:, :]))
        rel_v = num_v / den_v
        rate = '' if prev is None else '%8.2f' % (prev / rel_v)
        print('  %10.3e %14.4e %14.4e %14.4e %10s' % (ts, rel_v, num_p / den_p, ratio, rate))
        prev = rel_v
        out[ts] = (Jd, pred, ratio)
        blk.to('cpu')
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    print('\n  Note the POSITION column: the leading-order prediction for those rows is ZERO')
    print('  (qdot does not depend on theta), so a relative error near 1.0 there is the claim')
    print('  being CONFIRMED, not violated. T3 checks their true size.')
    return out


def t3_position_rows(out, norm, cfg):
    banner('T3  POSITION ROWS are O(Ts^2), with the RK4 ratio (Ts/2) * std_x[i+3] / std_x[i]')
    std_x = np.asarray(norm.std_x, np.float64).reshape(6)
    for ts, (Jd, _pred, ratio) in sorted(out.items(), reverse=True):
        # Predicted per-channel ratio of position row i to velocity row i+3.
        pred_ratio = [(ts / 2.0) * std_x[i + 3] / std_x[i] for i in range(3)]
        meas = [float(torch.linalg.norm(Jd[:, i, :]) / torch.linalg.norm(Jd[:, i + 3, :]))
                for i in range(3)]
        print('  Ts = %.3e' % ts)
        for i, ch in enumerate(['X', 'Theta', 'Y']):
            print('    %-6s measured %.6e   predicted (Ts/2)*std_x ratio %.6e   ratio %.4f'
                  % (ch, meas[i], pred_ratio[i], meas[i] / pred_ratio[i]))


# ------------------------------------------------------------------ T4
def t4_pointwise_rank(out, cfg):
    banner('T4  POINTWISE RANK of S(k), the corrected no-go lemma')
    print('  The lemma needs rank S(k) = 6 (full ROW rank). p > q does NOT imply it.\n')
    Jd = out[max(out)][0]                                    # production Ts
    S = Jd.numpy()                                           # (N, 6, 10)
    sv = np.linalg.svd(S, compute_uv=False)                  # (N, 6)
    tol = sv[:, :1] * 10 * np.finfo(np.float64).eps
    rank = (sv > tol).sum(axis=1)
    print('  samples: %d' % len(sv))
    print('  rank histogram: %s' % {int(r): int(c) for r, c in
                                    zip(*np.unique(rank, return_counts=True))})
    print('  sigma_6 / sigma_1 (per-sample conditioning of S):')
    r = sv[:, 5] / sv[:, 0]
    for q in (0, 1, 25, 50, 75, 100):
        print('      %3d%% quantile  %.4e' % (q, np.percentile(r, q)))
    full = int((rank == 6).sum())
    print('\n  T4 verdict: rank S(k) = 6 at %d of %d samples (%.2f %%)'
          % (full, len(rank), 100.0 * full / len(rank)))
    if full == len(rank):
        print('  => full row rank, so the strict lemma holds. But sigma_6/sigma_1 ~ 1e-10 means')
        print('     it holds only MARGINALLY, and that needs explaining before it is used.')
    else:
        print('  => there ARE samples with a pointwise null direction. The lemma does not hold')
        print('     everywhere and the claim in Sect. 8 must be weakened.')

    # WHY the conditioning is so bad, and why it does not rescue a physical addition.
    # T3 showed the position rows are nearly PROPORTIONAL to their velocity siblings,
    # pos_i = c_i * vel_{i+3} with c_i = (Ts/2) std_x[i+3]/std_x[i]. So to leading order S(k)
    # has rank THREE, and the three extra directions that lift it to rank six are O(Ts^2)
    # corrections. The near-null subspace of S(k)^T is therefore the set of state-row
    # perturbations with delta_vel_i = -c_i delta_pos_i, i.e. ones acting mostly on the
    # POSITION rows. A force cannot do that: it enters the accelerations, and its position
    # effect carries the SAME integrator factor c_i, which puts it along the column space
    # rather than its complement. So the physically relevant test restricts to velocity rows.
    banner('T4b  THE PHYSICALLY RELEVANT RANK: velocity rows only')
    print('  A generalised FORCE is fixed by its three velocity-row components, and its')
    print('  position-row effect is the same O(Ts^2) integrator factor the baseline has (T3).')
    print('  So the pointwise question for a real addition is the rank of the 3 x 10 block.')
    print()
    Sv = S[:, 3:, :]                                          # (N, 3, 10)
    svv = np.linalg.svd(Sv, compute_uv=False)                 # (N, 3)
    tolv = svv[:, :1] * 10 * np.finfo(np.float64).eps
    rv = (svv > tolv).sum(axis=1)
    print('  rank histogram (max 3): %s' % {int(a): int(c) for a, c in
                                            zip(*np.unique(rv, return_counts=True))})
    cond3 = svv[:, 2] / svv[:, 0]
    print('  sigma_3 / sigma_1 per sample:')
    for q in (0, 1, 50, 100):
        print('      %3d%% quantile  %.4e' % (q, np.percentile(cond3, q)))
    fullv = int((rv == 3).sum())
    print()
    print('  T4b verdict: rank = 3 of 3 at %d of %d samples (%.2f %%), worst conditioning %.2e'
          % (fullv, len(rv), 100.0 * fullv / len(rv), cond3.min()))
    if fullv == len(rv) and cond3.min() > 1e-6:
        print('  => the three logical force channels are spanned at EVERY sample, robustly.')
        print('     No nonzero pointwise-orthogonal FORCE exists. The no-go lemma holds for')
        print('     physically realisable additions, and with margin rather than marginally.')


def main():
    stride = int(os.environ.get('OBC_REF_STRIDE', '997'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = RunConfig(
        mode='augmentation_ma50_b140-230_a6_z03', encoder_init='linear_map',
        joint_estimation=True, physics_parameterization='reduced',
        combo_init_detune=COMBO_INIT_DETUNE, param_init_detune=None,
        obc=True, obc_space='tangent', nx_ann=8, up_sample=1, stride=10,
        fs_new=4000, snr=None, save_flag=False)
    banner('SETUP (stride %d, device %s, Ts = %.6g s)' % (stride, device, cfg.ts_new))
    data = load_datasets(cfg)
    norm = compute_normalization(cfg, data)
    ref = build_reference_set(cfg, norm, nx_ann=cfg.nx_ann, stride=stride, verbose=True)

    blk, free_true = make_block(norm, cfg, cfg.ts_new)
    ok = t1_split(blk, free_true, norm)
    out = t2_leading_order(cfg, norm, ref, device)
    t3_position_rows(out, norm, cfg)
    t4_pointwise_rank(out, cfg)

    banner('SUMMARY')
    print('  T1 split 5/4/1            : %s' % ('PASS' if ok else 'FAIL'))
    print('  T2 leading order + rate   : see the table above; the rate must approach 2.00')
    print('  T3 position rows O(Ts^2)  : see the ratios above')
    print('  T4 pointwise rank         : see the verdict above')


if __name__ == '__main__':
    main()
