"""
diag_joint_estimation.py -- Gate checks for joint estimation (D-076).

Gate 1 (Phase 1):
  capture : save reference rollout of the UNMODIFIED fixed Gantry_State_Block.
            Must be run once BEFORE the _mats() refactor of the parent block;
            guarded against overwrite so it can never be silently regenerated
            with post-refactor code.
  A0      : refactored fixed block reproduces the pre-refactor reference.
  A1      : Parameterized_Gantry_State_Block at log_params=0 matches the
            fixed block (value equivalence) -- now also validates the full
            M(Y)-structure rebuild from nominal masses (D-077).
  B       : finite-difference vs autograd gradients of a rollout loss w.r.t.
            all 14 log_params (float64), incl. finite/nonzero assertions.
  M       : PD + inverse-consistency of the rebuilt M(Y) across the positive
            parameter orthant (off-nominal transcription guard, D-077).

Gate 2 (Phase 2) appends checks C and D.

Run:  conda run -n GraduationProject python scripts/gantry/diag_joint_estimation.py
Results: simulations/gantry_subnet/diagnostics/joint_estimation/
"""

import os
import sys
import json

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from model_augmentation.fit_systems.blocks import Gantry_State_Block

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'simulations',
                       'gantry_subnet', 'diagnostics', 'joint_estimation')
REF_PATH = os.path.join(OUT_DIR, 'ref_fixed_block.npz')
RESULT_PATH = os.path.join(OUT_DIR, 'gate1_results.json')

SEED = 1234
BATCH = 64
N_STEPS = 100          # reference rollout length
N_STEPS_GRAD = 10      # short rollout for the gradient check (small graph)

# Pipeline-representative block constants (multisine pipeline values)
TS = 1.0 / 4000.0
UP_SAMPLE = 2

# HEURISTIC: fixed plausible normalization stats -- exercise the denorm/renorm
# paths with realistic magnitudes; exact values are irrelevant, determinism is.
STD_X = np.array([0.05, 0.02, 0.09, 0.30, 0.15, 0.40]).reshape(6, 1)
X_MEAN = np.array([0.0, 0.0, 0.30, 0.0, 0.0, 0.0]).reshape(6, 1)
STD_U = np.array([30.0, 30.0, 20.0]).reshape(3, 1)
U_MEAN = np.zeros((3, 1))

# Tolerances
TOL_A0 = 1e-7   # same code path, same inputs -> expect bitwise 0
TOL_A1 = 2e-6   # float32 value equivalence fixed vs parameterized@nominal
# gradcheck-style combined tolerance |auto-fd| <= atol + rtol*|fd|. Pure relative
# error is unusable for Jh (gradient ~2.7e-7, ~1e3x smaller than kb): there the
# central-difference roundoff floor (~1e-11 on an O(0.1) loss) dominates. atol is
# set ~100x below the smallest true gradient, so a real gradient error still fails.
TOL_B_REL = 1e-5   # relative term
TOL_B_ATOL = 1e-9  # absolute floor (roundoff, not a bug)


def make_fixed_block(dtype=torch.float32):
    blk = Gantry_State_Block(Y_op=None, std_x=STD_X, std_u=STD_U,
                             x_mean=X_MEAN, u_mean=U_MEAN,
                             Ts=TS, up_sample=UP_SAMPLE).to(dtype)
    blk.eval()
    return blk


def make_param_block(dtype=torch.float32, **kwargs):
    from model_augmentation.fit_systems.blocks import Parameterized_Gantry_State_Block
    blk = Parameterized_Gantry_State_Block(
        Y_op=None, std_x=STD_X, std_u=STD_U, x_mean=X_MEAN, u_mean=U_MEAN,
        Ts=TS, up_sample=UP_SAMPLE, **kwargs).to(dtype)
    blk.eval()
    return blk


def make_inputs(dtype=torch.float32, n_steps=N_STEPS):
    g = torch.Generator().manual_seed(SEED)
    x0 = (torch.randn(BATCH, 6, 1, generator=g) * 0.5)
    u_seq = (torch.randn(n_steps, BATCH, 3, 1, generator=g) * 0.5)
    return x0.to(dtype), u_seq.to(dtype)


def rollout(block, x0, u_seq, grad=False):
    """Iterate nonlinear_function; returns (n_steps, batch, 6, 1) trajectory."""
    ctx = torch.enable_grad() if grad else torch.no_grad()
    xs = []
    x = x0
    with ctx:
        for k in range(u_seq.shape[0]):
            z = torch.cat([x, u_seq[k]], dim=1)
            x = block.nonlinear_function(z)
            xs.append(x)
        traj = torch.stack(xs)
    return traj


# ----------------------------------------------------------------------
# Reference capture (run once, pre-refactor)
# ----------------------------------------------------------------------

def capture_reference():
    if os.path.exists(REF_PATH):
        print(f'[capture] reference exists, NOT overwriting: {REF_PATH}')
        return False
    blk = make_fixed_block()
    x0, u_seq = make_inputs()
    traj = rollout(blk, x0, u_seq)
    np.savez(REF_PATH,
             x0=x0.numpy(), u_seq=u_seq.numpy(), traj=traj.numpy(),
             seed=np.array(SEED), ts=np.array(TS), up_sample=np.array(UP_SAMPLE),
             std_x=STD_X, x_mean=X_MEAN, std_u=STD_U, u_mean=U_MEAN)
    print(f'[capture] reference saved: {REF_PATH}  traj={tuple(traj.shape)}')
    return True


# ----------------------------------------------------------------------
# Check A0: fixed block (post-refactor) reproduces pre-refactor reference
# ----------------------------------------------------------------------

def check_A0():
    ref = np.load(REF_PATH)
    blk = make_fixed_block()
    x0 = torch.tensor(ref['x0'])
    u_seq = torch.tensor(ref['u_seq'])
    traj = rollout(blk, x0, u_seq).numpy()
    max_diff = float(np.max(np.abs(traj - ref['traj'])))
    ok = max_diff <= TOL_A0
    print(f'[A0] fixed block vs pre-refactor reference: max|diff|={max_diff:.3e} '
          f'(tol {TOL_A0:.0e})  {"PASS" if ok else "FAIL"}')
    return dict(name='A0_refactor_equivalence', max_diff=max_diff,
                tol=TOL_A0, status='PASS' if ok else 'FAIL')


# ----------------------------------------------------------------------
# Check A1: parameterized block @ log_params=0 == fixed block
# ----------------------------------------------------------------------

def check_A1():
    blk_fix = make_fixed_block()
    blk_par = make_param_block()
    x0, u_seq = make_inputs()
    traj_fix = rollout(blk_fix, x0, u_seq)
    traj_par = rollout(blk_par, x0, u_seq)
    max_diff = float((traj_fix - traj_par).abs().max())
    ok = max_diff <= TOL_A1
    print(f'[A1] parameterized@nominal vs fixed block: max|diff|={max_diff:.3e} '
          f'(tol {TOL_A1:.0e})  {"PASS" if ok else "FAIL"}')
    return dict(name='A1_init_equivalence', max_diff=max_diff,
                tol=TOL_A1, status='PASS' if ok else 'FAIL')


# ----------------------------------------------------------------------
# Check B: finite-difference vs autograd gradient, all 5 params (float64)
# ----------------------------------------------------------------------

def check_B():
    dtype = torch.float64
    blk = make_param_block(dtype=dtype)
    x0, u_seq = make_inputs(dtype=dtype, n_steps=N_STEPS_GRAD)

    def loss_value():
        traj = rollout(blk, x0, u_seq, grad=False)
        return float((traj ** 2).mean())

    # autograd
    blk.zero_grad()
    traj = rollout(blk, x0, u_seq, grad=True)
    loss = (traj ** 2).mean()
    loss.backward()
    g_auto = blk.log_params.grad.detach().clone().numpy()

    # central finite differences in log space
    n_p = blk.log_params.numel()
    eps = 1e-6
    g_fd = np.zeros(n_p)
    with torch.no_grad():
        for i in range(n_p):
            blk.log_params.data[i] += eps
            lp = loss_value()
            blk.log_params.data[i] -= 2 * eps
            lm = loss_value()
            blk.log_params.data[i] += eps
            g_fd[i] = (lp - lm) / (2 * eps)

    finite = np.all(np.isfinite(g_auto))
    nonzero = np.all(np.abs(g_auto) > 0)
    abs_err = np.abs(g_auto - g_fd)
    rel_err = abs_err / np.maximum(np.abs(g_fd), 1e-300)
    passes = abs_err <= (TOL_B_ATOL + TOL_B_REL * np.abs(g_fd))
    ok = finite and nonzero and bool(np.all(passes))

    names = list(blk.PARAM_NAMES)
    print(f'[B] FD vs autograd (float64, eps={eps:.0e}, {N_STEPS_GRAD}-step rollout):')
    for i, n in enumerate(names):
        print(f'    {n:7s} autograd={g_auto[i]:+.6e}  fd={g_fd[i]:+.6e}  '
              f'abs_err={abs_err[i]:.2e}  rel_err={rel_err[i]:.2e}  '
              f'{"ok" if passes[i] else "FAIL"}')
    print(f'[B] finite={finite}  nonzero={nonzero}  '
          f'max_abs_err={abs_err.max():.2e}  max_rel_err={rel_err.max():.2e} '
          f'(atol {TOL_B_ATOL:.0e} + rtol {TOL_B_REL:.0e}|fd|)  '
          f'{"PASS" if ok else "FAIL"}')
    return dict(name='B_gradient_correctness',
                g_auto=g_auto.tolist(), g_fd=g_fd.tolist(),
                abs_err=abs_err.tolist(), rel_err=rel_err.tolist(),
                max_abs_err=float(abs_err.max()), max_rel_err=float(rel_err.max()),
                finite=bool(finite), nonzero=bool(nonzero),
                tol_rel=TOL_B_REL, tol_atol=TOL_B_ATOL,
                status='PASS' if ok else 'FAIL')


# ----------------------------------------------------------------------
# Check M: rebuilt M(Y) is positive-definite AND N(Y)/d(Y) is its true
# inverse, across the positive parameter orthant (D-077). A1 only covers the
# nominal point; this guards the build_poly_constants transcription off-nominal
# -- exactly where a mass-rebuild bug would hide during training.
# ----------------------------------------------------------------------

TOL_M_INV = 1e-9   # float64 3x3: |M @ (N/d) - I| — expect ~1e-13*cond
N_SAMP_M = 200
SPREAD_M = 0.30    # log-uniform exp(+-0.30) ~ +-30% around truth (stays positive)


def check_matrix():
    from model_augmentation.systems.gantry_ss import build_poly_constants, Lb as _Lb
    dtype = torch.float64
    Lb = _Lb.to(dtype)
    # masses + geometry only (kb/cg/cy/cb do not enter M(Y))
    names = ['mh', 'm1', 'm2', 'mb', 'Jb', 'Jh', 'd']
    import model_augmentation.systems.gantry_ss as _gss
    truth = torch.tensor([getattr(_gss, n).item() for n in names], dtype=dtype)
    Y_grid = torch.linspace(0.0, 0.6, 7, dtype=dtype)   # operating Y range [m]
    eye3 = torch.eye(3, dtype=dtype)
    z = torch.zeros((), dtype=dtype)
    g = torch.Generator().manual_seed(SEED)

    max_inv_err, min_eig = 0.0, float('inf')
    for _ in range(N_SAMP_M):
        fac = torch.exp((torch.rand(7, generator=g, dtype=dtype) - 0.5) * 2 * SPREAD_M)
        mh_, m1_, m2_, mb_, Jb_, Jh_, d_ = truth * fac
        alpha, beta, gamma, N0, N1, N2 = build_poly_constants(
            m1_, m2_, mb_, mh_, Jb_, Jh_, Lb, d_)
        M0 = torch.stack([
            torch.stack([m1_+m2_+mb_+mh_, (m1_-m2_)*Lb/2,                       z      ]),
            torch.stack([(m1_-m2_)*Lb/2,  Jb_+Jh_+(m1_+m2_)*Lb**2/4+mh_*d_**2,  -mh_*d_]),
            torch.stack([z,               -mh_*d_,                               mh_    ]),
        ])
        M1 = torch.stack([torch.stack([z, -mh_, z]),
                          torch.stack([-mh_, z, z]),
                          torch.stack([z, z, z])])
        M2 = torch.stack([torch.stack([z, z, z]),
                          torch.stack([z, mh_, z]),
                          torch.stack([z, z, z])])
        for Y in Y_grid:
            MY = M0 + M1 * Y + M2 * Y ** 2
            NY = N0 + N1 * Y + N2 * Y ** 2
            dY = mh_ * (alpha * gamma - beta ** 2
                        + 2 * beta * mh_ * Y + mh_ * (alpha - mh_) * Y ** 2)
            max_inv_err = max(max_inv_err,
                              (MY @ (NY / dY) - eye3).abs().max().item())
            min_eig = min(min_eig, torch.linalg.eigvalsh(MY).min().item())

    ok = (max_inv_err <= TOL_M_INV) and (min_eig > 0.0)
    print(f'[M] rebuilt M(Y): max|M·N/d - I|={max_inv_err:.2e} (tol {TOL_M_INV:.0e})  '
          f'min eig(M)={min_eig:.4f}>0  ({N_SAMP_M} samples x {len(Y_grid)} Y)  '
          f'{"PASS" if ok else "FAIL"}')
    return dict(name='M_matrix_pd_inverse', max_inv_err=max_inv_err,
                min_eig=min_eig, n_samples=N_SAMP_M, spread=SPREAD_M,
                tol_inv=TOL_M_INV, status='PASS' if ok else 'FAIL')


# ======================================================================
# Gate 2 (Phase 2): C (loss integration), D (mini parameter recovery)
# ======================================================================

RESULT2_PATH = os.path.join(OUT_DIR, 'gate2_results.json')

N_DATA = 4000                     # 1 s at 4 kHz per trajectory
NA = 13                           # encoder history: nxd*2+1 with nxd=6
NF_C = 25                         # rollout horizon for check C
NF_D = 100                        # rollout horizon for check D training
BATCH_D = 128
EPOCHS_D = 15
LR_D = 2e-3
# 14 RAW physical params, order = Parameterized_Gantry_State_Block.PARAM_NAMES.
# P_TRUE derived from gantry_ss (single source of truth, no transcription).
from model_augmentation.systems import gantry_ss as _gss  # noqa: E402
RAW_NAMES = ["kb1", "kb2", "cg1", "cg2", "cy", "cb1", "cb2",
             "mh", "m1", "m2", "mb", "Jb", "Jh", "d"]
P_TRUE = np.array([getattr(_gss, n).item() for n in RAW_NAMES])
# HEURISTIC: fixed ±10% detune per raw param, reproducible (lfr_param_block _DETUNING).
# Sum pairs (kb1/kb2, cb1/cb2, Jb/Jh) share sign so the identifiable SUM is detuned.
DETUNE = np.array([1.10, 1.10,          # kb1, kb2  -> kb_sum +10%
                   1.10, 0.90, 1.10,    # cg1, cg2, cy
                   0.90, 0.90,          # cb1, cb2  -> cb_sum -10%
                   0.90,                # mh
                   1.10, 0.90, 1.10,    # m1, m2, mb
                   0.90, 0.90,          # Jb, Jh    -> J_sum -10%
                   1.10])               # d
# HEURISTIC: input std [N] small enough that K=0 position drift stays within
# the operating range over 1 s (Y stays in ~[0.2, 0.4] m)
U_STD2 = np.array([2.0, 2.0, 1.5])
TOL_C_REL = 1e-6
TOL_D_GAP = 0.5    # well-conditioned subset: |learned-true| <= 0.5*|detuned-true|
TOL_D_MASS = 1.5   # mass combos: reported, only gated against divergence


def gen_multisine_u(seed):
    """(N_DATA, 3) random-phase multisine, band 1-200 Hz, per-channel std U_STD2.

    THEORY: random-phase multisine -- flat-amplitude harmonics on the DFT grid
    within the excitation band, uniform random phases (standard sysid input).
    """
    rng = np.random.default_rng(seed)
    fs = 1.0 / TS
    t = np.arange(N_DATA) / fs
    df = fs / N_DATA
    ks = np.arange(max(1, int(round(1.0 / df))), int(round(200.0 / df)) + 1)
    u = np.zeros((N_DATA, 3))
    for ch in range(3):
        sig = np.zeros(N_DATA)
        phases = rng.uniform(0, 2 * np.pi, size=ks.shape)
        for k, ph in zip(ks, phases):
            sig += np.cos(2 * np.pi * k * df * t + ph)
        u[:, ch] = sig / sig.std() * U_STD2[ch]
    return u.astype(np.float32)


def make_block_p2(kind, **kwargs):
    """Fixed or parameterized block with phase-2 normalization stats."""
    common = dict(Y_op=None, std_x=STD_X, std_u=U_STD2.reshape(3, 1),
                  x_mean=X_MEAN, u_mean=U_MEAN, Ts=TS, up_sample=UP_SAMPLE)
    if kind == 'fixed':
        return Gantry_State_Block(**common)
    from model_augmentation.fit_systems.blocks import Parameterized_Gantry_State_Block
    return Parameterized_Gantry_State_Block(**common, **kwargs)


def simulate_y(block, u_phys, std_x=None, x_mean=None):
    """Simulate y[k] = Cd @ x_phys[k] BEFORE the step (pipeline convention).

    std_x/x_mean must match the block's construction stats (default: the
    phase-2 constants). Returns (y_phys (N,3), x_phys (N,6)).
    """
    from model_augmentation.systems.gantry_ss import Cd
    Cd_np = Cd.numpy()
    sx = (STD_X if std_x is None else std_x).flatten()
    xm = (X_MEAN if x_mean is None else x_mean).flatten()
    u_norm = torch.tensor(u_phys / U_STD2[None, :], dtype=torch.float32)
    x = torch.zeros(1, 6, 1)
    ys, xs = [], []
    with torch.no_grad():
        for k in range(u_phys.shape[0]):
            x_phys = x.squeeze(0).squeeze(-1).numpy() * sx + xm
            ys.append(Cd_np @ x_phys)
            xs.append(x_phys)
            z = torch.cat([x, u_norm[k].view(1, 3, 1)], dim=1)
            x = block.nonlinear_function(z)
    return np.array(ys, dtype=np.float32), np.array(xs, dtype=np.float32)


def build_fit_sys(phy_block, train_data, y0, ystd):
    """Minimal no-ANN interconnect (physics + linear output) under the new trainer."""
    import deepSI  # noqa: F401  (System_data already constructed by caller)
    from model_augmentation.fit_systems.interconnect import (
        Interconnect, SSE_Interconnect_ParamLoss)
    from model_augmentation.fit_systems.blocks import Linear_Output_Block
    from model_augmentation.utils.utils import selection_matrix, expansion_matrix
    from model_augmentation.systems.gantry_ss import Cd

    Cd_norm = Cd.numpy() * STD_X.flatten()[None, :] / ystd[:, None]
    ic = Interconnect(6, 3, 3, debugging=False)
    out = Linear_Output_Block(C=Cd_norm, D=np.zeros((3, 3)))
    ic.add_block(phy_block)
    ic.add_block(out)
    ix = np.arange(6)
    ic.connect_signals("x", phy_block, "concat", selection_matrix(ix, 6))
    ic.connect_block_signals(phy_block, ["u"], [])
    ic.connect_signals(phy_block, "xp", "additive", expansion_matrix(ix, 6))
    ic.connect_signals("x", out, "concat", selection_matrix(ix, 6))
    ic.connect_block_signals(out, ["u"], ["y"])

    fs = SSE_Interconnect_ParamLoss(
        interconnect=ic, na=NA, nb=NA,
        e_net_kwargs={"n_nodes_per_layer": 16, "n_hidden_layers": 2})
    fs.norm.u0 = np.zeros(3)
    fs.norm.ustd = U_STD2.copy()
    fs.norm.y0 = y0
    fs.norm.ystd = ystd
    fs.init_model(sys_data=train_data, auto_fit_norm=False)
    return fs


def _batch_from(fit_sys, data, nf, n):
    arrays = fit_sys.make_training_data(fit_sys.norm.transform(data), nf=nf)
    return [torch.tensor(a[:n], dtype=torch.float32) for a in arrays]


def check_C():
    import deepSI
    from model_augmentation.fit_systems.interconnect import SSE_Interconnect

    u = gen_multisine_u(seed=777)
    y, _ = simulate_y(make_block_p2('fixed'), u)
    data = deepSI.System_data(u=u, y=y, dt=TS)
    y0, ystd = y.mean(axis=0), y.std(axis=0) + 1e-8

    phy = make_block_p2('param', RMSE_baseline=0.02)
    fit_sys = build_fit_sys(phy, data, y0, ystd)
    with torch.no_grad():
        phy.log_params += 0.05          # make param_loss > 0, params off-nominal

    batch = _batch_from(fit_sys, data, NF_C, 64)
    l_sub = float(fit_sys.loss(*batch, nf=NF_C))
    l_par = float(SSE_Interconnect.loss(fit_sys, *batch, nf=NF_C))
    pl = float(phy.param_loss())
    rel = abs(l_sub - (l_par + pl)) / max(abs(l_sub), 1e-30)
    ok = (pl > 0) and (rel <= TOL_C_REL)
    print(f'[C] loss integration: subclass={l_sub:.8e}  parent+param_loss='
          f'{l_par + pl:.8e}  param_loss={pl:.3e}  rel_diff={rel:.2e}  '
          f'{"PASS" if ok else "FAIL"}')
    return dict(name='C_loss_integration', loss_subclass=l_sub,
                loss_parent=l_par, param_loss=pl, rel_diff=rel,
                tol_rel=TOL_C_REL, status='PASS' if ok else 'FAIL')


class _StateReadoutEncoder(torch.nn.Module):
    """Exact, parameter-free encoder for check D.

    In check D the synthetic system's output IS the full state (output block
    C = I), so the exact state estimate is the last normalized measurement:
    x_hat(k) = y_norm(k) (na_right=1). This removes state estimation from the
    test entirely -- check D isolates whether fit() moves log_params to truth
    through the full trainer stack, which is the D-076 machinery under test.

    Why not the pipeline configuration: two recorded Gate-2 attempts showed
    that with position-only outputs and a trainable encoder, the ENCODER
    absorbs the loss (random init: 15 epochs spent learning it; even with a
    good init the co-trained encoder compensates short-window parameter error)
    -- see gate2 logs. That identifiability caveat is a Phase-4 run-design
    input, not a defect of the block/trainer, and is recorded in the todo.
    """

    def forward(self, upast, ypast):
        return ypast[:, -1, :]


def build_fit_sys_stateout(phy_block, train_data, y0, ystd):
    """Minimal fit system for check D: full-state output (C=I), exact
    readout encoder, no ANN. The only trainable parameters are log_params."""
    from model_augmentation.fit_systems.interconnect import (
        Interconnect, SSE_Interconnect_ParamLoss)
    from model_augmentation.fit_systems.blocks import Linear_Output_Block
    from model_augmentation.utils.utils import selection_matrix, expansion_matrix

    ic = Interconnect(6, 3, 6, debugging=False)
    out = Linear_Output_Block(C=np.eye(6, dtype=np.float32),
                              D=np.zeros((6, 3), dtype=np.float32))
    ic.add_block(phy_block)
    ic.add_block(out)
    ix = np.arange(6)
    ic.connect_signals("x", phy_block, "concat", selection_matrix(ix, 6))
    ic.connect_block_signals(phy_block, ["u"], [])
    ic.connect_signals(phy_block, "xp", "additive", expansion_matrix(ix, 6))
    ic.connect_signals("x", out, "concat", selection_matrix(ix, 6))
    ic.connect_block_signals(out, ["u"], ["y"])

    fs = SSE_Interconnect_ParamLoss(
        interconnect=ic, na=2, nb=2, na_right=1, nb_right=1,
        e_net_kwargs={"n_nodes_per_layer": 16, "n_hidden_layers": 2})
    fs.norm.u0 = np.zeros(3)
    fs.norm.ustd = U_STD2.copy()
    fs.norm.y0 = y0
    fs.norm.ystd = ystd
    fs.encoder = _StateReadoutEncoder()   # injected BEFORE init_model (pipeline order)
    fs.init_model(sys_data=train_data, auto_fit_norm=False)
    return fs


def check_D():
    import time as _time
    import deepSI

    # --- ground-truth data from the FIXED nominal block (no absorber) -----
    # The DATA here is the full physical state trajectory (y := x_phys, 6 ch):
    # with output C = I and the exact readout encoder, the training loss is
    # purely parameter mismatch. Generation stats are arbitrary (identity
    # round-trip); training blocks use stats MEASURED from the trajectories.
    u_tr = gen_multisine_u(seed=101)
    u_val = gen_multisine_u(seed=202)
    gen_blk = make_block_p2('fixed')
    _, x_tr = simulate_y(gen_blk, u_tr)
    _, x_val = simulate_y(gen_blk, u_val)
    train_data = deepSI.System_data(u=u_tr, y=x_tr, dt=TS)
    val_data = deepSI.System_data(u=u_val, y=x_val, dt=TS)
    std_x2 = (x_tr.std(axis=0) + 1e-12).reshape(6, 1)
    x_mean2 = x_tr.mean(axis=0).reshape(6, 1)
    y0, ystd = x_mean2.flatten(), std_x2.flatten()   # y IS the state

    def mk(kind, **kwargs):
        common = dict(Y_op=None, std_x=std_x2, std_u=U_STD2.reshape(3, 1),
                      x_mean=x_mean2, u_mean=U_MEAN, Ts=TS, up_sample=UP_SAMPLE)
        if kind == 'fixed':
            return Gantry_State_Block(**common)
        from model_augmentation.fit_systems.blocks import Parameterized_Gantry_State_Block
        return Parameterized_Gantry_State_Block(**common, **kwargs)

    # --- fit systems: parameterized (recovery) + fixed (timing reference) --
    # flag_loss_reg=False: pure-MSE recovery. Check C already validates the
    # regularization path; anchoring reg to the DETUNED init would actively
    # pin parameters there once the MSE landscape flattens (observed in
    # Gate-2 attempt 2).
    p_detuned = P_TRUE * DETUNE
    phy = mk('param', params_init=p_detuned, flag_loss_reg=False)
    fit_sys = build_fit_sys_stateout(phy, train_data, y0, ystd)
    fit_ref = build_fit_sys_stateout(mk('fixed'), train_data, y0, ystd)

    # --- runtime overhead: forward-only, like-for-like (the fixed reference
    # here has NO trainable parameters at all, so it has no backward pass) --
    def time_forward(fs, n=3):
        batch = _batch_from(fs, train_data, NF_D, BATCH_D)
        with torch.no_grad():
            t0 = _time.time()
            for _ in range(n):
                fs.loss(*batch, nf=NF_D)
        return (_time.time() - t0) / n

    def time_train_step(fs, n=3):
        batch = _batch_from(fs, train_data, NF_D, BATCH_D)
        t0 = _time.time()
        for _ in range(n):
            loss = fs.loss(*batch, nf=NF_D)
            loss.backward()
        return (_time.time() - t0) / n

    t_param = time_forward(fit_sys)
    t_fixed = time_forward(fit_ref)
    t_param_train = time_train_step(fit_sys)
    overhead = t_param / t_fixed - 1.0
    print(f'[D] per-batch forward: param={t_param:.2f}s fixed={t_fixed:.2f}s '
          f'overhead={overhead * 100:+.1f}%  (param fwd+bwd: {t_param_train:.2f}s)')

    # --- train ------------------------------------------------------------
    # Windowed validation matching nf: on this K=0 plant, sim-RMS validation
    # of windowed training hits the horizon-mismatch checkpoint trap
    # (problem log section 3) -- observed here as best=epoch 0 in the first
    # Gate 2 attempt. Windowed validation is Jan's own recommendation.
    fit_sys.fit(train_sys_data=train_data, val_sys_data=val_data,
                epochs=EPOCHS_D, batch_size=BATCH_D, auto_fit_norm=False,
                loss_kwargs={'nf': NF_D}, optimizer_kwargs={'lr': LR_D},
                validation_measure=f'{NF_D}-step-average-RMS')

    # NOTE: fit() restores the best checkpoint by replacing fit_sys state --
    # the pre-fit `phy` reference is stale; read params from the loaded hfn.
    from model_augmentation.fit_systems.blocks import Parameterized_Gantry_State_Block
    blk = [m for m in fit_sys.hfn.connected_blocks
           if hasattr(m, 'identifiable_combinations')][0]
    Lb = blk.Lb.item()
    combos = Parameterized_Gantry_State_Block._combos_from_raw
    true_c = combos(dict(zip(RAW_NAMES, P_TRUE)), Lb)
    det_c = combos(dict(zip(RAW_NAMES, p_detuned)), Lb)
    lrn_c = blk.identifiable_combinations()

    # Recovery is judged on the 10 IDENTIFIABLE combinations, not the raw
    # splits (kb1/kb2 etc. are flat directions the data cannot separate).
    # Well-conditioned subset is gated (<= 0.5, as in the 5-param gate);
    # mass combos are reported -- their identifiability from 1 s of data is
    # the open run-design question, so they are only guarded vs divergence.
    V1 = {'kb_sum', 'cg1', 'cg2', 'cy', 'cb_sum'}
    order = Parameterized_Gantry_State_Block._IDENT_ORDER
    ratios = {}
    print('[D] identifiable-combination recovery '
          '(true / detuned / learned / gap-ratio):')
    for n in order:
        t, dv, lv = true_c[n], det_c[n], lrn_c[n]
        g0 = abs(dv - t)
        r = abs(lv - t) / g0 if g0 > 1e-12 else float('nan')
        ratios[n] = r
        tag = 'gate<=0.5' if n in V1 else 'report'
        print(f'    {n:8s} {t:11.4f} {dv:11.4f} {lv:11.4f}   ratio={r:.3f}  [{tag}]')

    v1_ok = all(ratios[n] <= TOL_D_GAP for n in V1)
    finite_ok = all(np.isfinite(v) for v in ratios.values())
    nodiv_ok = all(ratios[n] <= TOL_D_MASS for n in ratios if np.isfinite(ratios[n]))
    ok = bool(v1_ok and finite_ok and nodiv_ok)
    v1_max = max(ratios[n] for n in V1)
    print(f'[D] v1-subset max ratio {v1_max:.3f} (gate <= {TOL_D_GAP}); '
          f'mass combos reported only  {"PASS" if ok else "FAIL"}')
    return dict(name='D_mini_recovery',
                ratios={k: (None if not np.isfinite(v) else float(v))
                        for k, v in ratios.items()},
                v1_max_ratio=float(v1_max),
                true=true_c, detuned=det_c, learned=lrn_c,
                mode='14 raw detuned; gate v1 subset, mass combos reported '
                     '(state-readout encoder C=I, flag_loss_reg=False)',
                epochs=EPOCHS_D, nf=NF_D, lr=LR_D, batch=BATCH_D,
                sec_fwd_param=t_param, sec_fwd_fixed=t_fixed,
                sec_fwdbwd_param=t_param_train,
                overhead_fwd_frac=float(overhead),
                tol_gap=TOL_D_GAP, tol_mass=TOL_D_MASS,
                status='PASS' if ok else 'FAIL')


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)

    captured_now = capture_reference()
    results = []

    results.append(check_A0())

    try:
        from model_augmentation.fit_systems.blocks import Parameterized_Gantry_State_Block  # noqa: F401
        has_param_block = True
    except ImportError:
        has_param_block = False
        print('[A1] SKIP -- Parameterized_Gantry_State_Block not implemented yet')
        print('[B]  SKIP -- Parameterized_Gantry_State_Block not implemented yet')

    if has_param_block:
        results.append(check_A1())
        results.append(check_B())
        results.append(check_matrix())

    with open(RESULT_PATH, 'w') as f:
        json.dump(dict(captured_now=captured_now, results=results), f, indent=2)
    print(f'Results written: {RESULT_PATH}')

    statuses = [r['status'] for r in results]
    print('GATE 1:', 'PASS' if all(s == 'PASS' for s in statuses) and has_param_block
          else ('INCOMPLETE (param block missing)' if not has_param_block
                else 'FAIL'))
    if any(s == 'FAIL' for s in statuses):
        sys.exit(1)

    # ---- Gate 2 (requires trainer subclass) ------------------------------
    try:
        from model_augmentation.fit_systems.interconnect import SSE_Interconnect_ParamLoss  # noqa: F401
        has_trainer = True
    except ImportError:
        has_trainer = False
        print('[C] SKIP -- SSE_Interconnect_ParamLoss not implemented yet')
        print('[D] SKIP -- SSE_Interconnect_ParamLoss not implemented yet')

    if has_param_block and has_trainer:
        results2 = [check_C(), check_D()]
        with open(RESULT2_PATH, 'w') as f:
            json.dump(dict(results=results2), f, indent=2)
        print(f'Results written: {RESULT2_PATH}')
        statuses2 = [r['status'] for r in results2]
        print('GATE 2:', 'PASS' if all(s == 'PASS' for s in statuses2) else 'FAIL')
        if any(s == 'FAIL' for s in statuses2):
            sys.exit(1)
