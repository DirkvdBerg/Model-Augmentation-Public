"""The known feedback controller Cfb, from the physical parameters to the ControllerBank of the closed loop."""
__project_origin__ = "added"

import os

import numpy as np
import torch
from scipy.linalg import block_diag
from scipy.signal import cont2discrete, tf2ss

from model_augmentation.fit_systems.closed_loop import ControllerBank, ClosedLoopSimulator

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
    """Mass matrix M(Y) at the frozen position Y_op. THEORY: gtd_build_plant.m."""
    return np.array([
        [m1 + m2 + mb + mh, (m1 - m2) * Lb / 2 - mh * Y_op, 0.],
        [(m1 - m2) * Lb / 2 - mh * Y_op,
         Jb + Jh + (m1 + m2) * Lb ** 2 / 4 + mh * d ** 2 + mh * Y_op ** 2, -mh * d],
        [0., -mh * d, mh]])


def stage_frequency_response(Y_op, s):
    """P^T G(s) P at s, with G the logical plant frozen at Y_op. THEORY: getss.m."""
    Minv = np.linalg.inv(mass_matrix_at(Y_op))
    A = np.block([[np.zeros((3, 3)), np.eye(3)], [-Minv @ K_STIFF, -Minv @ C_DAMP]])
    B = np.vstack([np.zeros((3, 3)), Minv])
    Cm = np.hstack([np.eye(3), np.zeros((3, 3))])
    return P.T @ (Cm @ np.linalg.solve(s * np.eye(6) - A, B)) @ P


def normalized_controller_coefficients():
    """Cnorm(s) = 10 W (s + W/6)(s + W/3) / (s (s + 3 W)(s + 10 W)) as (num, den). THEORY: ruleOfThumb.m."""
    num = 10 * W * np.polymul([1., W / 6], [1., W / 3])
    den = np.polymul([1., 0.], np.polymul([1., 3 * W], [1., 10 * W]))
    return num, den


def build_cfb_at(Y_op, ts):
    """Three discrete SISO controllers, one per stage channel, at sample time ts."""
    sysw = stage_frequency_response(Y_op, 1j * W)
    num, den = normalized_controller_coefficients()
    cw = np.polyval(num, 1j * W) / np.polyval(den, 1j * W)
    out = []
    for j in range(3):
        # THEORY: k_j = 1 / |G_jj(jW) Cnorm(jW)|, unit loop gain at the bandwidth (ruleOfThumb.m)
        kj = 1.0 / abs(sysw[j, j] * cw)
        b, a, _ = cont2discrete((kj * num, den), ts, method='bilinear')
        out.append((np.asarray(b).ravel(), np.asarray(a).ravel()))
    return out


def _tf_to_block_diag_ss(cfb):
    """Per-channel (b, a) to one block-diagonal (A, B, C, D) of the diagonal Cfb."""
    As, Bs, Cs, Ds = zip(*(tf2ss(b, a) for b, a in cfb))
    return block_diag(*As), block_diag(*Bs), block_diag(*Cs), np.diag([Dj.ravel()[0] for Dj in Ds])


def controller_ss(Y_op, ts, gain=1.0):
    """Cfb at sample time ts as one 3-input, 3-output state space (A, B, C, D), scaled by gain."""
    A, B, C, D = _tf_to_block_diag_ss(build_cfb_at(Y_op, ts))
    return A, B, gain * C, gain * D


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
    """Return the bank, record rows, and distinct positions or (position, gain) keys when use_gain is enabled."""
    keys = [(y_op_for(n, use_gain), gain_for(n, use_gain)) for n in record_names]
    uniq = sorted(set(keys))
    rows = [uniq.index(k) for k in keys]
    mats = [controller_ss(Y_op, float(ts), g) for Y_op, g in uniq]
    A, B, C, D = (np.stack([np.asarray(m[k], float) for m in mats]) for k in range(4))
    bank = ControllerBank(A, B, C, D, ystd=ystd, std_u=std_u, dtype=dtype)
    return bank, rows, (uniq if use_gain else [y for y, _ in uniq])


def build_closed_loop(fs, norm, cfg, *, train_files, val_files, val_data, e_train, e_val, verbose=True):
    """Build the simulator with controller rows in file order; val_data is retained for callers and unused."""
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
    # compile the pair for gradient-enabled CUDA objectives; validation uses the eager rollout
    compiled = None
    if cfg.compile_mode is not None:
        compiled = (
            torch.compile(fs.hfn, backend=COMPILE_BACKEND, mode=cfg.compile_mode,
                          fullgraph=True),
            torch.compile(fs.hfn.output_only, backend=COMPILE_BACKEND, mode=cfg.compile_mode,
                          fullgraph=True),
        )
        if verbose:
            print('[closed loop] training rollout COMPILED: backend=%s mode=%s fullgraph=True'
                  % (COMPILE_BACKEND, cfg.compile_mode))
    k0 = max(fs.na, fs.nb)
    val_sigma_e = servo_sigma(e_val, k0)
    # loss weight W_r = diag(sigma_y / sigma_e,r) per training record
    train_weights = np.asarray(norm.ystd, np.float64).ravel() / servo_sigma(e_train, k0)
    return ClosedLoopSimulator(bank, train_rows, val_rows, compiled=compiled,
                               train_weights=train_weights, val_sigma_e=val_sigma_e)
