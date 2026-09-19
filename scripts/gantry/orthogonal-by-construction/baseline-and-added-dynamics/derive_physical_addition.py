"""A PHYSICALLY MEANINGFUL addition that is orthogonal to the baseline.

Three lossless tuned mass absorbers on the payload, with their masses, natural frequencies and
mounting geometries solved from the ten orthogonality conditions.

WHY THIS OBJECT AND NOT ANOTHER. Every constraint below is a result established earlier in this
folder, not a modelling preference:

  * LOSSLESS. `zeta = 0` is forced. The condition pairing a dissipative addition with the damping
    family is a positive quadratic form in velocity, with no boundary term and no symmetry
    escape, so a dissipative element can never be orthogonal to `cg1, cg2, cy, cb_sum`.
  * AN ABSORBER, NOT A STIFFNESS OR INERTIA PERTURBATION. The memoryless solution
    `f_a = -(M_a q'' + K_a q)` is stateless, and its `M_a q''` term makes the interconnection
    CYCLIC, violating Condition 1 and Theorem 1 of Hoekstra et al. 2025 (EJC 86:101304). An
    absorber's reaction force depends on its OWN state, so the graph stays acyclic.
  * THREE OF THEM. Each absorber is one degree of freedom, hence two additional states. The
    production augmentation carries `nx_ann = 8`, so at most four are representable; Hoekstra's
    own 3-DOF example uses exactly the minimum (two states for one hidden DOF). Three leaves
    margin and still gives twelve free parameters against ten conditions.
  * DYNAMIC PARALLEL. In the Table 1 taxonomy this is `x_aug_{k+1} = g_aug(...)` feeding
    `f_aug(...)`, which is the structure the additional states exist for.

THE EQUATIONS.

    absorber i:   d_i'' + w_i^2 d_i = -b_i^T q'' ,        b_i = [cos(a_i), lever_i, sin(a_i)]
    reaction:     f_a = SUM_i b_i m_i w_i^2 d_i
    conditions:   <J_j, Delta(f_a)> = 0 ,  j = 1..10

`d_i` is independent of `m_i`, so the conditions are LINEAR in the masses and nonlinear only in
`(w_i, a_i, lever_i)`. The solve exploits that: for any geometry the masses come from a linear
least-norm step, and only six numbers are searched over.

Run:
  PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 OBC_REF_STRIDE=97 conda run --no-capture-output \
      -n GraduationProject python -u derive_physical_addition.py
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
DETUNE = [1.1, 1.1, 0.9, 1.1, 0.9, 0.9, 1.1, 1.1, 0.9, 1.1]
N_ABS = 3
LINE = '=' * 92


def banner(t):
    print('\n' + LINE); print(t); print(LINE, flush=True)


def absorber_response(drive, w, ts):
    """`d'' + w^2 d = drive`, exact zero-order-hold.

    # THEORY: exact ZOH of the undamped oscillator; exp(A ts) = [[cos, sin/w], [-w sin, cos]]
    # with input matrix [(1-cos)/w^2; sin/w] (Franklin, Powell and Workman, Digital Control of
    # Dynamic Systems, sect. 4.3). Used so the absorber contributes no discretisation error of
    # its own to the orthogonality conditions.
    """
    c, s = np.cos(w * ts), np.sin(w * ts)
    Ad = np.array([[c, s / w], [-w * s, c]])
    Bd = np.array([(1.0 - c) / w ** 2, s / w])
    out = np.empty(len(drive))
    x = np.zeros(2)
    for k in range(len(drive)):
        out[k] = x[0]
        x = Ad @ x + Bd * drive[k]
    return out


class Problem:
    """Everything fixed: the reference set, the basis, and the per-record accelerations."""

    def __init__(self, stride):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = RunConfig(
            mode='augmentation_ma50_b140-230_a6_z03', encoder_init='linear_map',
            joint_estimation=True, physics_parameterization='reduced',
            combo_init_detune=DETUNE, param_init_detune=None,
            obc=True, obc_space='tangent', nx_ann=8, up_sample=1, stride=10,
            fs_new=4000, snr=None, save_flag=False)
        cfg = self.cfg
        data = load_datasets(cfg)
        self.norm = compute_normalization(cfg, data)
        self.ref = build_reference_set(cfg, self.norm, nx_ann=cfg.nx_ann, stride=stride,
                                       verbose=True)
        names = Parameterized_Gantry_State_Block.PARAM_NAMES
        raw = torch.stack([getattr(gss, n) for n in names]).to(torch.float64)
        true_combo = Reduced_Gantry_State_Block.combos_of(raw, float(gss.Lb))
        det = torch.tensor(DETUNE, dtype=torch.float64)
        self.blk = Reduced_Gantry_State_Block(
            params_init=raw, combo_init=true_combo * det, Y_op=None,
            std_x=self.norm.std_x, std_u=self.norm.std_u, x_mean=self.norm.x_mean,
            u_mean=self.norm.u_mean, Ts=cfg.ts_new, up_sample=cfg.up_sample,
            RMSE_baseline=cfg.param_rmse_baseline, flag_loss_reg=False
        ).to(torch.float64).to(self.device)
        self.vbar = -torch.log(det)
        i = Reduced_Gantry_State_Block.M_DIFF_IX
        self.vbar[i] = 1.0 / float(det[i]) - 1.0

        self.step = lambda v, z: self.blk.transition_from_free(v, z)
        J = build_stacked_sensitivity(self.step, self.vbar.to(self.device), self.ref.Z_step,
                                      ROWS, chunk=cfg.obc_ref_chunk, device=self.device)
        self.basis = OBCBasis.from_matrix(J.cpu(), self.vbar.cpu())
        del J
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        self.base = evaluate_step_rows(self.step, self.vbar.to(self.device), self.ref.Z_step,
                                       ROWS, chunk=cfg.obc_ref_chunk,
                                       device=self.device).cpu()
        self.Pinv = np.linalg.inv(P_pt.detach().cpu().numpy().astype(np.float64))
        self.std_u = np.asarray(self.norm.std_u, np.float64).reshape(1, 3)
        self.stride = stride
        self._load_records()
        # Scale-free normalisation for the ten residuals, so no condition dominates by units.
        self.row_scale = np.maximum(np.abs(self.basis.jt(self.base)).numpy(), 1e-30)

    def _load_records(self):
        """`q''` per record plus the index map into the reference tuples."""
        P64 = P_pt.detach().cpu().numpy().astype(np.float64)
        PinvT = np.linalg.inv(P64.T)
        self.qdd_rec, self.idx_rec = [], []
        for f in TRAIN_FILES:
            u, y, _x, _a = _load_record_float64(f, self.cfg)
            q = y @ PinvT.T
            qd = fourth_order_central_difference(q, self.cfg.ts_new)
            qdd = fourth_order_central_difference(qd, self.cfg.ts_new)
            k = np.arange(2, len(u) - 2)
            if self.stride > 1:
                k = k[::self.stride]
            self.qdd_rec.append(qdd)
            self.idx_rec.append(np.clip(k - 4, 0, len(qdd) - 1))

    # ------------------------------------------------------------------ the model
    def unit_force(self, w_hz, alpha, lever):
        """Reaction force of ONE absorber of UNIT MASS, on the reference tuples."""
        b = np.array([np.cos(alpha), lever, np.sin(alpha)])
        w = 2.0 * np.pi * w_hz
        out = []
        for qdd, idx in zip(self.qdd_rec, self.idx_rec):
            d = absorber_response(-(qdd @ b), w, self.cfg.ts_new)
            out.append(np.outer(d[idx] * w ** 2, b))
        return np.concatenate(out)

    def delta_of(self, f_a):
        """The one-step contribution of a logical force, through the production block."""
        du = torch.from_numpy((f_a @ self.Pinv.T) / self.std_u).to(torch.float64)
        Z = self.ref.Z_step.clone().to(torch.float64)
        Z[:, 6:9, 0] += du
        out = evaluate_step_rows(self.step, self.vbar.to(self.device), Z, ROWS,
                                 chunk=self.cfg.obc_ref_chunk, device=self.device).cpu()
        del Z
        return out - self.base

    def conditions(self, geom):
        """The 10 x N_ABS condition matrix for a geometry, and the per-absorber Delta fields."""
        C = np.zeros((10, N_ABS))
        Ds, Fs = [], []
        for i in range(N_ABS):
            f = self.unit_force(*geom[i])
            D = self.delta_of(f)
            C[:, i] = self.basis.jt(D).numpy()
            Ds.append(D); Fs.append(f)
        return C, Ds, Fs

    def solve_masses(self, C, Ds):
        """Masses in the null space of C, scaled to unit total mass, sign-checked.

        With three absorbers and ten conditions the system is overdetermined, so an exact null
        vector exists only if C has rank <= 2. The geometry search below drives it there; here
        the least singular direction is returned together with how far from exact it is.
        """
        U, S, Vt = np.linalg.svd(C / self.row_scale[:, None])
        m = Vt[-1]
        if m.sum() < 0:
            m = -m
        return m, float(S[-1] / max(S[0], 1e-300))


def residual(problem, p):
    """Scale-free residual of the ten conditions at a packed parameter vector."""
    geom = [(np.exp(p[3 * i]), p[3 * i + 1], p[3 * i + 2]) for i in range(N_ABS)]
    C, Ds, _ = problem.conditions(geom)
    m, rel = problem.solve_masses(C, Ds)
    r = (C @ m) / problem.row_scale
    # Penalise negative masses: a negative absorber mass is not a mechanism.
    pen = 10.0 * np.minimum(m, 0.0)
    return np.concatenate([r, pen]), m, C, Ds


def main():
    stride = int(os.environ.get('OBC_REF_STRIDE', '97'))
    banner('SETUP (stride %d)' % stride)
    pb = Problem(stride)
    print('[setup] basis rank %d/%d, cond %.4e' % (pb.basis.rank, pb.basis.n_cols,
                                                   pb.basis.report.cond))

    banner('THE ADDITION: three lossless tuned absorbers, dynamic parallel, 6 states')
    print("  d_i'' + w_i^2 d_i = -b_i^T q'' ,   f_a = SUM_i b_i m_i w_i^2 d_i")
    print('  b_i = [cos(alpha_i), lever_i, sin(alpha_i)];  free: f_i [Hz], alpha_i, lever_i, m_i')
    print('  zeta = 0 forced by the damping-family obstruction; acyclic, so Hoekstra Condition 1')
    print('  holds, unlike the memoryless M_a q\'\' solution.')

    from scipy.optimize import least_squares
    rng = np.random.default_rng(0)
    best = None
    t0 = time.perf_counter()
    for trial in range(6):
        # Start from resonances spread around the excitation band and mixed mountings.
        f0 = np.array([40.0, 110.0, 300.0]) * np.exp(rng.normal(0, 0.25, N_ABS))
        a0 = rng.uniform(-np.pi / 2, np.pi / 2, N_ABS)
        l0 = rng.uniform(-0.3, 0.3, N_ABS)
        p0 = np.empty(3 * N_ABS)
        for i in range(N_ABS):
            p0[3 * i], p0[3 * i + 1], p0[3 * i + 2] = np.log(f0[i]), a0[i], l0[i]
        lo = np.concatenate([[np.log(5.0), -np.pi, -0.5]] * N_ABS)
        hi = np.concatenate([[np.log(1500.0), np.pi, 0.5]] * N_ABS)
        res = least_squares(lambda p: residual(pb, p)[0], p0, bounds=(lo, hi),
                            xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=120)
        cost = float(np.linalg.norm(res.fun))
        print('  trial %d: residual %.4e  (%.0f s elapsed)'
              % (trial, cost, time.perf_counter() - t0), flush=True)
        if best is None or cost < best[0]:
            best = (cost, res.x)

    banner('RESULT')
    r, m, C, Ds = residual(pb, best[1])
    geom = [(np.exp(best[1][3 * i]), best[1][3 * i + 1], best[1][3 * i + 2])
            for i in range(N_ABS)]
    # Physical scale: put the total added mass at 10 percent of the payload.
    m = m * (0.10 * float(gss.mh) / max(m.sum(), 1e-30))
    Delta = torch.zeros_like(Ds[0])
    for i in range(N_ABS):
        Delta = Delta + float(m[i]) * Ds[i]
    print('  %-4s %10s %10s %10s %12s %10s' % ('i', 'f [Hz]', 'alpha [deg]', 'lever [m]',
                                               'mass [kg]', 'states'))
    for i, (f, a, l) in enumerate(geom):
        print('  %-4d %10.2f %10.2f %10.4f %12.5f %10d'
              % (i, f, np.degrees(a), l, m[i], 2))
    print('  %-4s %10s %10s %10s %12.5f %10d' % ('', '', '', 'total', m.sum(), 2 * N_ABS))

    rho = pb.basis.r_overlap(Delta)
    print('\n  rho*        = %.6e    (the current dataset absorber gives 2.05e-01)' % rho)
    print('  ||J^T D||   = %.6e' % float(np.linalg.norm(pb.basis.jt(Delta).numpy())))
    print('  r_perp      = %.6e    (scale-free orthogonality residual)' % pb.basis.r_perp(Delta))
    print('  ||Delta||   = %.6e    (the current absorber at this stride: 1.73e+00)'
          % float(torch.linalg.vector_norm(Delta)))
    dth = pb.basis.coefficient(Delta).numpy()
    print('\n  predicted parameter bias, which should be numerically zero:')
    for n, v in zip(COMBO_NAMES, dth):
        print('    %-9s %+.3e' % (n, v))

    ok = rho < 1e-6
    print('\n  VERDICT: %s' % (
        'ORTHOGONAL, physically meaningful, and inside the state budget'
        if ok else 'not orthogonal to the target accuracy; see the residual above'))
    with open(os.path.join(HERE, 'physical_addition.json'), 'w') as fh:
        json.dump(dict(stride=stride, n_absorbers=N_ABS, states=2 * N_ABS,
                       geometry=[[float(f), float(a), float(l)] for f, a, l in geom],
                       masses=m.tolist(), rho=float(rho),
                       delta_norm=float(torch.linalg.vector_norm(Delta)),
                       predicted_bias=dth.tolist()), fh, indent=2)
    print('\n[out] wrote %s' % os.path.join(HERE, 'physical_addition.json'))


if __name__ == '__main__':
    main()
