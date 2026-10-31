"""The known controller K_fb: K_1 per stage axis, tuned on the nominal baseline at the design Y and discretised by
Tustin at T_s, gathered in the ControllerBank of the closed loop. Code names: Cfb = K_fb, W = w_b,
Cnorm(s) = K_1,j(s) / kappa_j."""
__project_origin__ = "added"

import os

import numpy as np
import torch
from scipy.linalg import block_diag
from scipy.signal import cont2discrete, tf2ss

from model_augmentation.fit_systems.closed_loop import ControllerBank, ClosedLoopSimulator, closed_loop_step

from .data import servo_sigma

# torch.compile backend of the training rollout; the mode is cfg.compile_mode
COMPILE_BACKEND = 'inductor'

# THEORY: physical parameters of the controller design (gtd_config.m)
mb, mh, m1, m2, Jb, Jh = 22.8, 10.1, 10.2, 10.7, 1.0, 0.05
cg1, cg2, cy, cb1, cb2 = 14.5, 20.3, 10.0, 9.0, 9.0
kb1, kb2, Lb, d = 1987.5, 1987.5, 0.725, 0.1

FBW = 100.0             # THEORY: gtd_config.m, design bandwidth [Hz]
W = 2 * np.pi * FBW

# THEORY: stage-to-logical transform P and the damping and stiffness matrices (getss.m)
P = np.array([[1., 1., 0.], [Lb / 2, -Lb / 2, 0.], [0., 0., 1.]])
C_DAMP = np.array([[cg1 + cg2, (cg1 - cg2) * Lb / 2, 0.],
                   [(cg1 - cg2) * Lb / 2, cb1 + cb2 + (cg1 + cg2) * Lb ** 2 / 4, 0.],
                   [0., 0., cy]])
K_STIFF = np.array([[0., 0., 0.], [0., kb1 + kb2, 0.], [0., 0., 0.]])


def mass_matrix_at(Y_op):
    """Inertia matrix for the controller design.

    M(Y_op) = [[m_Sigma + m_h, (L_b/2) m_Delta - m_h Y, 0], [., J_eff + m_h (d^2 + Y^2), -m_h d], [0, -m_h d, m_h]]
    at the nominal parameters; the K_1 tuning uses the plant frozen at this position. THEORY: gtd_build_plant.m.

    Args:
        Y_op: frozen payload position Y, metres.
    Returns:
        (3, 3) M(Y_op), physical.
    """
    return np.array([
        [m1 + m2 + mb + mh, (m1 - m2) * Lb / 2 - mh * Y_op, 0.],
        [(m1 - m2) * Lb / 2 - mh * Y_op,
         Jb + Jh + (m1 + m2) * Lb ** 2 / 4 + mh * d ** 2 + mh * Y_op ** 2, -mh * d],
        [0., -mh * d, mh]])


def stage_frequency_response(Y_op, s):
    """Plant frequency response for the controller tuning.

    P^T G(s) P, G(s) = [I 0] (s I - A)^-1 B of xdot = F(x, u) frozen at Y_op: actuator force to actuator position.
    THEORY: getss.m.

    Args:
        Y_op: frozen payload position Y, metres.
        s: complex Laplace frequency, 1/s.
    Returns:
        (3, 3) complex response, m/N.
    """
    Minv = np.linalg.inv(mass_matrix_at(Y_op))
    # A = [0 I; -R K -R C], B = [0; R], R = M(Y_op)^-1.
    A = np.block([[np.zeros((3, 3)), np.eye(3)], [-Minv @ K_STIFF, -Minv @ C_DAMP]])
    B = np.vstack([np.zeros((3, 3)), Minv])
    Cm = np.hstack([np.eye(3), np.zeros((3, 3))])
    return P.T @ (Cm @ np.linalg.solve(s * np.eye(6) - A, B)) @ P


def normalized_controller_coefficients():
    """Shape of the controller K_1.

    Cnorm(s) = 10 w_b (s + w_b/6)(s + w_b/3) / (s (s + 3 w_b)(s + 10 w_b)), scaled per axis by kappa_j in the
    tuning. THEORY: ruleOfThumb.m.

    Returns:
        num (3,) and den (4,), coefficients in descending powers of s.
    """
    # Numerator 10 w_b (s + w_b/6)(s + w_b/3), denominator s (s + 3 w_b)(s + 10 w_b).
    num = 10 * W * np.polymul([1., W / 6], [1., W / 3])
    den = np.polymul([1., 0.], np.polymul([1., 3 * W], [1., 10 * W]))
    return num, den


def build_cfb_at(Y_op, ts):
    """Per-axis controller K_1.

    K_1,j(z) = kappa_j Cnorm(s) by Tustin at ts, kappa_j = 1 / |G_jj(j w_b) Cnorm(j w_b)|: unit open-loop gain at the
    bandwidth on the nominal baseline at Y_op.

    Args:
        Y_op: design position Y, metres.
        ts: sample time T_s, seconds.
    Returns:
        three (b, a) pairs, each (4,) in descending powers of z.
    """
    # 1. Plant and controller shape at w_b.
    sysw = stage_frequency_response(Y_op, 1j * W)
    num, den = normalized_controller_coefficients()
    cw = np.polyval(num, 1j * W) / np.polyval(den, 1j * W)
    out = []
    for j in range(3):
        # 2. Gain kappa_j.
        # THEORY: k_j = 1 / |G_jj(jW) Cnorm(jW)|, unit loop gain at the bandwidth (ruleOfThumb.m)
        kj = 1.0 / abs(sysw[j, j] * cw)
        # 3. Tustin at ts.
        b, a, _ = cont2discrete((kj * num, den), ts, method='bilinear')
        out.append((np.asarray(b).ravel(), np.asarray(a).ravel()))
    return out


def controller_ss(Y_op, ts, gain=1.0):
    """Controller K_fb as a state space.

    K_fb(z) = gain diag(K_1,1(z), K_1,2(z), K_1,3(z)) as one block-diagonal (A_fb, B_fb, C_fb, D_fb), from the
    position error r - y (metres) to the feedback force (newtons) on the actuator axes.

    Args:
        Y_op: design position Y, metres.
        ts: sample time T_s, seconds.
        gain: dimensionless output scale.
    Returns:
        A_fb (9, 9), B_fb (9, 3), C_fb (3, 9), D_fb (3, 3), physical.
    """
    As, Bs, Cs, Ds = zip(*(tf2ss(b, a) for b, a in build_cfb_at(Y_op, ts)))
    D = np.diag([Dj.ravel()[0] for Dj in Ds])
    return block_diag(*As), block_diag(*Bs), gain * block_diag(*Cs), gain * D


def _stem(name):
    """A record's file name without folder and .mat suffix: the key of _CONTROLLER_METADATA."""
    stem = os.path.basename(str(name))
    return stem[:-4] if stem.endswith('.mat') else stem


# stem -> (design Y, gain) of a stored controller the bank can build, or the description of one it cannot
_CONTROLLER_METADATA = {}


def register_stored_controller(name, ctrl_name, design_y, gain, fbw_hz):
    """Register a record's stored controller: (design Y, gain) if the bank can build it, else its description."""
    _CONTROLLER_METADATA[_stem(name)] = ((design_y, float(gain)) if abs(fbw_hz - FBW) < 1e-9
                                         else '%s (gain %g, %g Hz)' % (ctrl_name, gain, fbw_hz))


def y_op_for(name, use_gain=False):
    """Controller design position Y of a record, by file name with or without .mat."""
    stem = _stem(name)
    entry = _CONTROLLER_METADATA.get(stem)
    if isinstance(entry, tuple):
        gain_for(stem, use_gain)
        return entry[0]
    if entry is not None:
        raise KeyError('record %r uses controller %s, which the bank cannot build (it builds the %g Hz '
                       'ruleOfThumb design at a design Y); leave the record out of training and validation'
                       % (stem, entry, FBW))
    raise KeyError('no controller registered for record %r; load the record through gantry_dynamic.data, '
                   'which registers its stored controller' % stem)


def gain_for(name, use_gain=False):
    """Stored gain of a record's controller, 1.0 unless stored otherwise."""
    stem = _stem(name)
    entry = _CONTROLLER_METADATA.get(stem)
    g = entry[1] if isinstance(entry, tuple) else 1.0
    if abs(g - 1.0) > 1e-12 and not use_gain:
        raise KeyError('record %r was generated with %g x K1; use a caller that explicitly passes use_gain=True '
                       'to build_controller_bank, or restrict this run to unit-gain records' % (stem, g))
    return g


def build_controller_bank(record_names, ts, ystd, std_u, dtype=torch.float32, use_gain=False):
    """Controller bank of a data set.

    One K_fb per distinct design (normalised in ControllerBank) and the bank row of each record.

    Args:
        record_names: record file names.
        ts: sample time T_s, seconds.
        ystd: (3,) sigma_y, metres.
        std_u: (3, 1) or (3,) sigma_u, newtons.
        dtype: model dtype.
        use_gain: allow controllers with a stored gain other than 1.
    Returns:
        bank, rows (list of n_records ints), and the distinct design keys.
    """
    keys = [(y_op_for(n, use_gain), gain_for(n, use_gain)) for n in record_names]
    uniq = sorted(set(keys))
    rows = [uniq.index(k) for k in keys]
    mats = [controller_ss(Y_op, float(ts), g) for Y_op, g in uniq]
    A, B, C, D = (np.stack([np.asarray(m[k], float) for m in mats]) for k in range(4))
    bank = ControllerBank(A, B, C, D, ystd=ystd, std_u=std_u, dtype=dtype)
    return bank, rows, (uniq if use_gain else [y for y, _ in uniq])


def build_closed_loop(fs, norm, cfg, *, train_files, val_files, val_data, e_train, e_val, verbose=True):
    """Closed-loop simulator of training and validation.

    The bank of K_fb, W_r = diag(sigma_y / sigma_e,r) per training record and sigma_e per validation record, with
    the closed-loop step compiled when cfg.compile_mode is set.

    Args:
        fs: the augmented model.
        norm: normalisation.
        cfg: run configuration.
        train_files, val_files: record file names.
        val_data: unused, kept for callers.
        e_train, e_val: lists of (N_r, 3) servo errors y - r, metres.
        verbose: print the bank and compile settings.
    Returns:
        ClosedLoopSimulator.
    """
    # 1. Controller bank over the training and validation records.
    names = [_stem(f) for f in list(train_files) + list(val_files)]
    if len(set(names)) != len(names):
        raise ValueError('records %s appear more than once in train_files and val_files; give each record to '
                         'exactly one split' % [n for n in names if names.count(n) > 1])

    bank, rows, y_ops = build_controller_bank(
        names, cfg.ts_new, ystd=norm.ystd, std_u=norm.std_u, dtype=cfg.dtype_pt)
    n_tr = len(train_files)
    train_rows, val_rows = rows[:n_tr], rows[n_tr:]

    if verbose:
        print('[closed loop] one bank over %d records, %d distinct controllers, Y_op %s'
              % (len(names), bank.n_controllers, y_ops))
        print('[closed loop] stepped at ts = %g s (%d Hz), nc = %d states'
              % (cfg.ts_new, round(1 / cfg.ts_new), bank.nc))
    # 2. Compiled step.
    # compile the closed-loop step for gradient-enabled CUDA objectives; validation uses the eager rollout
    compiled = None
    if cfg.compile_mode is not None:
        compiled = torch.compile(closed_loop_step(fs.hfn, bank), backend=COMPILE_BACKEND, mode=cfg.compile_mode,
                                 fullgraph=True)
        if verbose:
            print('[closed loop] training rollout COMPILED: backend=%s mode=%s fullgraph=True'
                  % (COMPILE_BACKEND, cfg.compile_mode))
    # 3. Loss and validation scales.
    k0 = max(fs.na, fs.nb)
    val_sigma_e = servo_sigma(e_val, k0)
    # Loss weight: W_r = diag(sigma_y / sigma_e,r) per training record.
    train_weights = np.asarray(norm.ystd, np.float64).ravel() / servo_sigma(e_train, k0)
    return ClosedLoopSimulator(bank, train_rows, val_rows, compiled=compiled,
                               train_weights=train_weights, val_sigma_e=val_sigma_e)
