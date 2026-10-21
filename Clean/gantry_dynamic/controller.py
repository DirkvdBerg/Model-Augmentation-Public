"""Cfb: the known feedback controller, from physical parameters to a ControllerBank."""
__project_origin__ = "added"

import os

import numpy as np
import torch
from scipy.linalg import block_diag
from scipy.signal import cont2discrete, tf2ss

from model_augmentation.fit_systems.closed_loop import ControllerBank, ClosedLoopSimulator

from .data import servo_sigma

# torch.compile backend for the training rollout (the only fusing backend; the mode is cfg.compile_mode)
COMPILE_BACKEND = 'inductor'

# THEORY: gtd_config.m:39-41, 61-68 physical parameters.
mb, mh, m1, m2, Jb, Jh = 22.8, 10.1, 10.2, 10.7, 1.0, 0.05
cg1, cg2, cy, cb1, cb2 = 14.5, 20.3, 10.0, 9.0, 9.0
kb1, kb2, Lb, d = 1987.5, 1987.5, 0.725, 0.1

FBW = 100.0             # THEORY: gtd_config.m, design bandwidth [Hz]
W = 2 * np.pi * FBW

# THEORY: getss.m:2-6, stage-to-logical transform and the frozen damping/stiffness.
P = np.array([[1., 1., 0.], [Lb / 2, -Lb / 2, 0.], [0., 0., 1.]])
C_DAMP = np.array([[cg1 + cg2, (cg1 - cg2) * Lb / 2, 0.],
                   [(cg1 - cg2) * Lb / 2, cb1 + cb2 + (cg1 + cg2) * Lb ** 2 / 4, 0.],
                   [0., 0., cy]])
K_STIFF = np.array([[0., 0., 0.], [0., kb1 + kb2, 0.], [0., 0., 0.]])


def M_op(Y_op):
    """Mass matrix at a frozen operating point. THEORY: gtd_build_plant.m:18-20."""
    return np.array([
        [m1 + m2 + mb + mh, (m1 - m2) * Lb / 2 - mh * Y_op, 0.],
        [(m1 - m2) * Lb / 2 - mh * Y_op,
         Jb + Jh + (m1 + m2) * Lb ** 2 / 4 + mh * d ** 2 + mh * Y_op ** 2, -mh * d],
        [0., -mh * d, mh]])


def sys_stage_frf(Y_op, s):
    """sys = P' getss(M_op, C_damp, K) P evaluated at s. THEORY: getss.m:2-6."""
    Minv = np.linalg.inv(M_op(Y_op))
    A = np.block([[np.zeros((3, 3)), np.eye(3)], [-Minv @ K_STIFF, -Minv @ C_DAMP]])
    B = np.vstack([np.zeros((3, 3)), Minv])
    Cm = np.hstack([np.eye(3), np.zeros((3, 3))])
    return P.T @ (Cm @ np.linalg.solve(s * np.eye(6) - A, B)) @ P


def cnorm_coeffs():
    """Cnorm(s) as (num, den) polynomials. THEORY: DERIVATION.md section 2."""
    num = 10 * W * np.polymul([1., W / 6], [1., W / 3])
    den = np.polymul([1., 0.], np.polymul([1., 3 * W], [1., 10 * W]))
    return num, den


def build_cfb_at(Y_op, ts):
    """Three discrete SISO controllers, one per stage channel, at sample time `ts`."""
    sysw = sys_stage_frf(Y_op, 1j * W)
    num, den = cnorm_coeffs()
    cw = np.polyval(num, 1j * W) / np.polyval(den, 1j * W)
    out = []
    for j in range(3):
        kj = 1.0 / abs(sysw[j, j] * cw)                                   # THEORY: ruleOfThumb.m:11
        b, a, _ = cont2discrete((kj * num, den), ts, method='bilinear')
        out.append((np.asarray(b).ravel(), np.asarray(a).ravel()))
    return out


def _tf_to_ss_batch(cfb):
    """Per-channel (b, a) -> block-diagonal (A, B, C, D) for the 3-channel diagonal Cfb."""
    As, Bs, Cs, Ds = zip(*(tf2ss(b, a) for b, a in cfb))
    return block_diag(*As), block_diag(*Bs), block_diag(*Cs), np.diag([Dj.ravel()[0] for Dj in Ds])


def controller_ss(Y_op, ts, gain=1.0):
    """Cfb at sample time `ts`, as one 3-in 3-out state space (A, B, C, D)."""
    A, B, C, D = _tf_to_ss_batch(build_cfb_at(Y_op, ts))
    return A, B, gain * C, gain * D


# THEORY: Matlab-scripts/Augmentation/data/gtd_build_records.m:36-67. Keys are the loader's
# file stems (gantry_dynamic/data.py TRAIN_FILES etc.) without '.mat'.


def _stem(name):
    stem = os.path.basename(str(name))
    return stem[:-4] if stem.endswith('.mat') else stem


# stem -> (design Y, gain) of a supported stored controller, or the text naming an unsupported one
_REGISTRY = {}


def register_stored_controller(name, ctrl_name, design_y, gain, fbw_hz):
    _REGISTRY[_stem(name)] = ((design_y, float(gain)) if abs(fbw_hz - FBW) < 1e-9
                              else '%s (gain %g, %g Hz)' % (ctrl_name, gain, fbw_hz))


def y_op_for(name, use_gain=False):
    """Controller design Y for a record file name, with or without the .mat suffix."""
    stem = _stem(name)
    entry = _REGISTRY.get(stem)
    if isinstance(entry, tuple):
        gain_for(stem, use_gain)
        return entry[0]
    if entry is not None:
        raise KeyError('record %r uses controller %s; the bank builds ruleOfThumb at gain 1 and '
                       '%g Hz from a design Y only, so this record cannot be trained or validated '
                       'on in closed loop (D-221)' % (stem, entry, FBW))
    raise KeyError('no Y_op known for record %r; add it to RECORD_Y_OP from '
                   'gtd_build_records.m, or load it through gantry_dynamic.data so its stored '
                   'controller is registered (D-221)' % stem)


def gain_for(name, use_gain=False):
    """The stored gain of a record's controller; 1.0 unless stored otherwise. D-227."""
    stem = _stem(name)
    entry = _REGISTRY.get(stem)
    g = entry[1] if isinstance(entry, tuple) else 1.0
    if abs(g - 1.0) > 1e-12 and not use_gain:
        raise KeyError('record %r was generated with %g x K1; set RunConfig.controller_gain to '
                       'score it with its own controller (D-227)' % (stem, g))
    return g


def build_controller_bank(record_names, ts, ystd, std_u, dtype=torch.float32, use_gain=False):
    """The framework's ControllerBank for a set of records, plus each record's row in it."""
    keys = [(y_op_for(n, use_gain), gain_for(n, use_gain)) for n in record_names]
    uniq = sorted(set(keys))
    rows = [uniq.index(k) for k in keys]
    mats = [controller_ss(Y_op, float(ts), g) for Y_op, g in uniq]
    A, B, C, D = (np.stack([np.asarray(m[k], float) for m in mats]) for k in range(4))
    bank = ControllerBank(A, B, C, D, ystd=ystd, std_u=std_u, dtype=dtype)
    return bank, rows, (uniq if use_gain else [y for y, _ in uniq])


def build_closed_loop(fs, norm, cfg, *, train_files, val_files, val_data, e_train, e_val, verbose=True):
    """The ClosedLoopSimulator for this pipeline. Assign it to `fs.simulator`. Validation pairs its
    records with the controller rows of `val_files`, in order; `val_data` stays for the callers."""
    names = [_stem(f) for f in list(train_files) + list(val_files)]
    if len(set(names)) != len(names):
        raise ValueError('the same record appears in both train_files and val_files, or twice in '
                         'one of them: %s. A record belongs to exactly one split, and a duplicate '
                         'here means the row lists below no longer line up with the data.'
                         % [n for n in names if names.count(n) > 1])

    bank, rows, y_ops = build_controller_bank(
        names, cfg.ts_new, ystd=norm.ystd, std_u=norm.std_u, dtype=cfg.dtype_pt)
    n_tr = len(train_files)
    train_rows, val_rows = rows[:n_tr], rows[n_tr:]

    if verbose:
        print('[closed loop] one bank over %d records, %d distinct controllers, Y_op %s'
              % (len(names), bank.n_controllers, y_ops))
        print('[closed loop] stepped at ts = %g s (%d Hz), nc = %d states'
              % (cfg.ts_new, round(1 / cfg.ts_new), bank.nc))
    # CHANGED (D-169): compilation is built HERE, where `fs` is already in hand, and handed to the
    # simulator rather than substituted for `fs.hfn`; see ClosedLoopSimulator.__init__ for why
    # that distinction is what keeps validation and the diagnostics eager.
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
    train_weights = np.asarray(norm.ystd, np.float64).ravel() / servo_sigma(e_train, k0)
    return ClosedLoopSimulator(bank, train_rows, val_rows, compiled=compiled,
                               train_weights=train_weights, val_sigma_e=val_sigma_e)
