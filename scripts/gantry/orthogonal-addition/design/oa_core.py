"""Shared machinery for the orthogonal-addition design: the production Jacobian, the fast force
operator, per-tuple signals, and candidate-addition columns.

Everything is evaluated on the OBC reference tuples of a dataset (`OA_MODE`), with the production
step `Reduced_Gantry_State_Block.transition_from_free` at the true parameters, exactly as
`measure/measure_delta.py` builds `J`. A candidate addition component `c` is a generalised force
field `phi_c[k]` in LOGICAL coordinates on those tuples; its one-step contribution is
`Delta_c = G phi_c` with `G[k]` the exact one-step response to a unit stage force (the block's RK4
is affine in `u`, `derive_physical_addition.py` verified it to 7.7e-09 relative). The condition
coefficient is `A[:, c] = J^T Delta_c`: the DISCRETE production Jacobian, never its continuous limit
(handoff sect. 6).
"""
__project_origin__ = "added"

import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OA = os.path.abspath(os.path.join(HERE, '..'))
REPO = os.path.abspath(os.path.join(OA, '..', '..', '..'))
for _p in (REPO, OA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_augmentation.fit_systems.blocks import (            # noqa: E402
    Parameterized_Gantry_State_Block, Reduced_Gantry_State_Block)
from model_augmentation.systems import gantry_ss as gss        # noqa: E402
from model_augmentation.systems.gantry_ss import P as P_pt     # noqa: E402
from vendor.obc import OBCBasis, build_stacked_sensitivity, evaluate_step_rows   # noqa: E402
from vendor.gantry_dynamic.config import RunConfig             # noqa: E402
from vendor.gantry_dynamic.data import TRAIN_FILES, compute_normalization, load_datasets  # noqa: E402
from vendor.gantry_dynamic.obc_gantry import (                 # noqa: E402
    build_reference_set, fourth_order_central_difference, _load_record_float64)

COMBO_NAMES = Reduced_Gantry_State_Block.COMBO_NAMES
ROWS = (0, 1, 2, 3, 4, 5)
DETUNE = [1.1, 1.1, 0.9, 1.1, 0.9, 0.9, 1.1, 1.1, 0.9, 1.1]
ORIGINAL_MODE = 'augmentation_ma50_b140-230_a6_z03'


def second_derivative4(q, ts):
    """`qdd[k] = (-q[k+2] + 16 q[k+1] - 30 q[k] + 16 q[k-1] - q[k-2]) / (12 ts^2)`, k in [2, N-3].

    # THEORY: fourth-order central second derivative, Fornberg (1988) Math. Comp. 51, Table 1
    # (second derivative, order 4, stencil -2..2 weights -1/12, 4/3, -5/2, 4/3, -1/12). Same
    # stencil support as the first-derivative stencil the reference set uses, so it exists on
    # exactly the reference tuples. Returns (N-4, ...) aligned with q[2:-2].
    """
    return (-q[4:] + 16.0 * q[3:-1] - 30.0 * q[2:-2] + 16.0 * q[1:-3] - q[:-4]) / (12.0 * ts * ts)


class Problem:
    """Reference set, basis at theta*, fast operator and per-tuple signals for one dataset."""

    def __init__(self, mode=None, stride=None, verbose=True):
        self.mode = mode or os.environ.get('OA_MODE', ORIGINAL_MODE)
        self.stride = int(stride if stride is not None else os.environ.get('OBC_REF_STRIDE', '1'))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        t0 = time.perf_counter()
        cfg = RunConfig(
            mode=self.mode, encoder_init='linear_map', joint_estimation=True,
            physics_parameterization='reduced', combo_init_detune=DETUNE, param_init_detune=None,
            obc=True, obc_space='tangent', nx_ann=8, up_sample=1, stride=10, fs_new=4000,
            snr=None, save_flag=False)
        self.cfg = cfg
        self.norm = compute_normalization(cfg, load_datasets(cfg))
        self.ref = build_reference_set(cfg, self.norm, nx_ann=cfg.nx_ann, stride=self.stride,
                                       verbose=verbose)
        names = Parameterized_Gantry_State_Block.PARAM_NAMES
        raw = torch.stack([getattr(gss, n) for n in names]).to(torch.float64)
        self.true_combo = Reduced_Gantry_State_Block.combos_of(raw, float(gss.Lb))
        det = torch.tensor(DETUNE, dtype=torch.float64)
        self.blk = Reduced_Gantry_State_Block(
            params_init=raw, combo_init=self.true_combo * det, Y_op=None,
            std_x=self.norm.std_x, std_u=self.norm.std_u, x_mean=self.norm.x_mean,
            u_mean=self.norm.u_mean, Ts=cfg.ts_new, up_sample=cfg.up_sample,
            RMSE_baseline=cfg.param_rmse_baseline, flag_loss_reg=False
        ).to(torch.float64).to(self.device)
        # The truth's free coordinate (log for nine, relative-linear for m_diff).
        self.vbar = -torch.log(det)
        i = Reduced_Gantry_State_Block.M_DIFF_IX
        self.vbar[i] = 1.0 / float(det[i]) - 1.0
        chk = self.blk.combinations_from_free(self.vbar.to(self.device)).detach().cpu()
        assert float((chk - self.true_combo).abs().max()) < 1e-10
        self.step = lambda v, z: self.blk.transition_from_free(v, z)
        J = build_stacked_sensitivity(self.step, self.vbar.to(self.device), self.ref.Z_step,
                                      ROWS, chunk=cfg.obc_ref_chunk, device=self.device)
        self.basis = OBCBasis.from_matrix(J.cpu(), self.vbar.cpu())
        self.Jcols = J.cpu().numpy() if self.stride > 1 else None   # dense only when small
        del J
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        self.base = evaluate_step_rows(self.step, self.vbar.to(self.device), self.ref.Z_step,
                                       ROWS, chunk=cfg.obc_ref_chunk, device=self.device).cpu()
        self.Pinv = np.linalg.inv(P_pt.detach().cpu().numpy().astype(np.float64))
        self.std_u = np.asarray(self.norm.std_u, np.float64).reshape(1, 3)
        self._signals()
        self._force_operator()
        if verbose:
            print('[core] %s stride %d: %d tuples, basis rank %d cond %.4e, %.1f s'
                  % (self.mode, self.stride, self.ref.n, self.basis.rank, self.basis.report.cond,
                     time.perf_counter() - t0), flush=True)

    # ==== signals
    def _signals(self):
        """Per-record full-grid q, qd, qdd (logical) and the tuple index map."""
        P64 = P_pt.detach().cpu().numpy().astype(np.float64)
        PinvT = np.linalg.inv(P64.T)
        ts = self.cfg.ts_new
        self.rec = []
        q_t, qd_t, qdd_t = [], [], []
        for f in TRAIN_FILES:
            u, y, _x, _a = _load_record_float64(f, self.cfg)
            q = y @ PinvT.T
            qd = np.zeros_like(q); qdd = np.zeros_like(q)
            qd[2:-2] = fourth_order_central_difference(q, ts)
            qdd[2:-2] = second_derivative4(q, ts)
            k = np.arange(2, len(u) - 2)
            if self.stride > 1:
                k = k[::self.stride]
            self.rec.append(dict(name=f, q=q, qd=qd, qdd=qdd, k=k, n=len(u)))
            q_t.append(q[k]); qd_t.append(qd[k]); qdd_t.append(qdd[k])
        self.q = np.concatenate(q_t); self.qd = np.concatenate(qd_t)
        self.qdd = np.concatenate(qdd_t)
        self.Y = self.q[:, 2]
        # The reference set's own x_phys must be these q and qd (same stencil, same samples).
        xp = self.ref.x_phys.numpy()
        err = max(np.abs(xp[:, :3] - self.q).max(), np.abs(xp[:, 3:] - self.qd).max())
        assert err < 1e-9, 'per-tuple signals lost alignment with the reference set (%.3e)' % err

    def _force_operator(self):
        """`G[k]` (6 x 3): the exact one-step response to a unit STAGE force at tuple k."""
        probe = 1.0e4
        cols = []
        for j in range(3):
            Z = self.ref.Z_step.clone().to(torch.float64)
            Z[:, 6 + j, 0] += probe / self.std_u[0, j]
            r = evaluate_step_rows(self.step, self.vbar.to(self.device), Z, ROWS,
                                   chunk=self.cfg.obc_ref_chunk, device=self.device).cpu()
            cols.append(((r - self.base).numpy()) / probe)
            del Z
        self.G = np.stack(cols, axis=1)                    # (n*6, 3)

    # ==== columns
    def delta_of(self, f_log):
        """One-step contribution of a LOGICAL force field (n, 3), stacked like J's rows."""
        f_stage = f_log @ self.Pinv.T
        return (self.G * np.repeat(f_stage, len(ROWS), axis=0)).sum(axis=1)

    def stateless_force(self, which, i, j, ypow=0):
        """Unit symmetric entry (i, j) of M_a / C_a / K_a times Y^ypow: f = -(E_ij v) Y^p."""
        E = np.zeros((3, 3)); E[i, j] = 1.0; E[j, i] = 1.0
        v = {'M': self.qdd, 'C': self.qd, 'K': self.q}[which]
        return -(v @ E.T) * (self.Y ** ypow)[:, None]

    def absorber_force(self, f_hz, b0, b1=(0.0, 0.0, 0.0)):
        """Reaction of ONE lossless absorber of UNIT mass: f = -b d'', d'' + w^2 d = -b^T qdd.

        Mass-conserving convention (OA-003): only the motion RELATIVE to the carrier is added.
        `d` from the exact ZOH recursion of the undamped oscillator, run on the full 4 kHz record
        from rest (records start at rest), then sampled on the tuples.
        """
        from scipy.signal import lfilter
        w = 2.0 * np.pi * f_hz
        ts = self.cfg.ts_new
        c, s = np.cos(w * ts), np.sin(w * ts)
        # THEORY: exact ZOH of d'' + w^2 d = drive (Franklin, Powell, Workman, Digital Control of
        # Dynamic Systems, sect. 4.3), as the IIR b = [0, 1-c, s^2 - c(1-c)]/w^2, a = [1, -2c, 1]
        # (derive_physical_addition.py, verified 1.3e-12 against the state recursion).
        bz = np.array([0.0, 1.0 - c, s * s - c * (1.0 - c)]) / (w * w)
        az = np.array([1.0, -2.0 * c, 1.0])
        b0 = np.asarray(b0, float); b1 = np.asarray(b1, float)
        out = []
        for r in self.rec:
            Yr = r['q'][:, 2]
            B = b0[None, :] + Yr[:, None] * b1[None, :]            # (n, 3)
            drive = -(B * r['qdd']).sum(axis=1)
            d = lfilter(bz, az, drive)
            dacc = drive - w * w * d                              # d'' = -b^T qdd - w^2 d
            out.append(-B[r['k']] * dacc[r['k']][:, None])
        return np.concatenate(out)

    def measured_delta(self):
        """The truth's recorded one-step residual on the tuples, stencil variant (D-203), exactly
        as measure_delta.py forms it: [q[k+1]; D4 q[k+1]] normalised, minus the baseline step at
        theta*. The last tuple of each record has no stencil at k+1 and is set to zero."""
        x_mean = np.asarray(self.norm.x_mean, np.float64).reshape(1, 6)
        std_x = np.asarray(self.norm.std_x, np.float64).reshape(1, 6)
        nxt, valid = [], []
        for r in self.rec:
            k1 = r['k'] + 1
            ok = k1 <= r['n'] - 3
            k1c = np.minimum(k1, r['n'] - 3)
            nxt.append(np.hstack([r['q'][k1c], r['qd'][k1c]]))
            valid.append(ok)
        xn = (np.concatenate(nxt) - x_mean) / std_x
        valid = np.repeat(np.concatenate(valid), len(ROWS))
        D = xn[:, list(ROWS)].reshape(-1) - self.base.numpy()
        return np.where(valid, D, 0.0)

    # ==== summaries
    def stats(self, D):
        """(J^T D, U^T D, ||D||) for a stacked one-step field D (numpy or torch)."""
        Dt = torch.as_tensor(np.asarray(D), dtype=torch.float64)
        jt = self.basis.jt(Dt).numpy()
        ut = (self.basis.U.T @ Dt).numpy()
        return jt, ut, float(torch.linalg.vector_norm(Dt))

    def rho(self, D):
        return float(self.basis.r_overlap(torch.as_tensor(np.asarray(D), dtype=torch.float64)))

    def bias_pct(self, D):
        """Predicted bias J^+ D in the combo-err percent convention, plus m_diff on its own scale."""
        Dt = torch.as_tensor(np.asarray(D), dtype=torch.float64)
        dth = self.basis.coefficient(Dt).cpu()
        combo = self.blk.combinations_from_free((self.vbar + dth).to(self.device)).detach().cpu().numpy()
        tc = self.true_combo.numpy()
        scale = np.abs(tc).copy()
        i = Reduced_Gantry_State_Block.M_DIFF_IX
        scale[i] = 0.5 * (float(gss.m1) + float(gss.m2))
        pct = 100.0 * (combo - tc) / scale
        own = 100.0 * (combo[i] - tc[i]) / abs(tc[i])
        return pct, own
