"""Cfb: the known feedback controller, from physical parameters to a ControllerBank.

This is the gantry side of the closed-loop training path, and the whole of it. The framework
side (`model_augmentation/fit_systems/closed_loop.py`) owns the rollout, the bank, the window
index and the validation scoring, and knows nothing about a gantry. This file owns everything
that framework must never learn: the physical parameters, the design plant, the `ruleOfThumb`
design rule, and which record sits at which operating point.

`build_controller_bank` is the boundary. What crosses it is four stacked (A, B, C, D) matrices
and one integer row per record. `Y_op`, `ruleOfThumb` and record names stop here.

Read top to bottom and the file is the derivation: physical parameters, design plant, design
rule, discretisation, record map, bank. Nothing runs at import.

DERIVATION.md section 2 writes the controller in closed form:

    Cnorm(s) = 10w (s + w/6)(s + w/3) / [ s (s + 3w)(s + 10w) ],   w = 2*pi*f_bw
    kappa_j  = 1 / |sys_jj(i w) Cnorm(i w)|
    C_j(z)   = tustin( kappa_j Cnorm(s), ts )

THE SAMPLE RATE IS ALWAYS AN ARGUMENT. `Cfb` is discretised by Tustin, which sends z = inf to
the finite frequency s = 2/ts, so `Dc_jj = kappa_j * Cnorm(2/ts)` and the controller is a
DIFFERENT operator at every rate. Records are stored at 20 kHz (`TS`); training steps at
`cfg.ts_new`, 4 kHz by default. `Dc` at Y_op = 0 is diag [2.844e+06 2.914e+06 1.509e+06] N/m at
20 kHz and diag [8.055e+06 8.253e+06 4.275e+06] N/m at 4 kHz, a factor 2.832, which is exactly
Cnorm(8000) / Cnorm(40000). Both are correct at their own rate, so `TS` below is exported as the
record rate and NO function in this file defaults to it.

THE FORM, residual: u_plant[k] = u_data[k] + Cfb * (y_data[k] - y_model[k]). Because the
controller filters the output RESIDUAL and the model was not running before a window opened,
xc = 0 at a window start is a definition rather than an approximation. `Dc != 0` at every rate,
so the step order (y before u) is forced; `closed_loop_rollout` implements it.

PER-RECORD Cfb. `generate_trajectory_data.m:43` calls `gtd_build_plant(rec.Y_op, cfg)` inside
the per-record loop, so the controller is rebuilt per record and frozen within it. Nine distinct
controllers across the 22 records.

Provenance: this replaces the six-module import chain under
`scripts/gantry/closed-loop-controller/` (and the `core/` copy of it) that the entry point used
until 2026-08-28. Those files are kept for reference and are no longer imported by anything on
the training path.
"""
__project_origin__ = "added"

import os

import numpy as np
import torch
from scipy.signal import cont2discrete, tf2ss

from model_augmentation.fit_systems.closed_loop import ControllerBank

# ── 1. the plant the controller was designed against ────────────────────────────────
# THEORY: gtd_config.m:39-41, 61-68 physical parameters.
mb, mh, m1, m2, Jb, Jh = 22.8, 10.1, 10.2, 10.7, 1.0, 0.05
cg1, cg2, cy, cb1, cb2 = 14.5, 20.3, 10.0, 9.0, 9.0
kb1, kb2, Lb, d = 1987.5, 1987.5, 0.725, 0.1

FBW = 100.0             # THEORY: gtd_config.m, design bandwidth [Hz]
W = 2 * np.pi * FBW
TS = 1.0 / 20e3         # the RECORD rate. Never a default; see the header.

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


# ── 2. the design rule ──────────────────────────────────────────────────────────────

def cnorm_coeffs():
    """Cnorm(s) as (num, den) polynomials. THEORY: DERIVATION.md section 2."""
    num = 10 * W * np.polymul([1., W / 6], [1., W / 3])
    den = np.polymul([1., 0.], np.polymul([1., 3 * W], [1., 10 * W]))
    return num, den


def cnorm_at(s):
    num, den = cnorm_coeffs()
    return np.polyval(num, s) / np.polyval(den, s)


def build_cfb_at(Y_op, ts):
    """Three discrete SISO controllers, one per stage channel, at sample time `ts`.

    Returns ([(b, a), ...], kappa). `ts` is an argument and has no default: Cfb is a different
    operator at every rate, so a caller that has not thought about the rate must say so.
    """
    sysw = sys_stage_frf(Y_op, 1j * W)
    cw = cnorm_at(1j * W)
    num, den = cnorm_coeffs()
    out, gains = [], []
    for j in range(3):
        kj = 1.0 / abs(sysw[j, j] * cw)                                   # THEORY: ruleOfThumb.m:11
        b, a, _ = cont2discrete((kj * num, den), ts, method='bilinear')   # c2d(.., 'tustin')
        out.append((np.asarray(b).ravel(), np.asarray(a).ravel()))
        gains.append(kj)
    return out, np.array(gains)


# ── 3. transfer function to state space ─────────────────────────────────────────────

def _tf_to_ss_batch(cfb):
    """Per-channel (b, a) -> block-diagonal (A, B, C, D) for the 3-channel diagonal Cfb."""
    As, Bs, Cs, Ds = [], [], [], []
    for b, a in cfb:
        A, B, C, D = tf2ss(b, a)
        As.append(A); Bs.append(B); Cs.append(C); Ds.append(D)
    n = sum(A.shape[0] for A in As)
    A = np.zeros((n, n)); B = np.zeros((n, 3)); C = np.zeros((3, n)); D = np.zeros((3, 3))
    i = 0
    for j, (Aj, Bj, Cj, Dj) in enumerate(zip(As, Bs, Cs, Ds)):
        m = Aj.shape[0]
        A[i:i + m, i:i + m] = Aj
        B[i:i + m, j] = Bj.ravel()
        C[j, i:i + m] = Cj.ravel()
        D[j, j] = Dj.ravel()[0]
        i += m
    return A, B, C, D


def controller_ss(Y_op, ts):
    """Cfb at sample time `ts`, as one 3-in 3-out state space (A, B, C, D)."""
    cfb, _ = build_cfb_at(Y_op, ts)
    return _tf_to_ss_batch(cfb)


# ── 4. the boundary: which record sits at which operating point ─────────────────────
# THEORY: Matlab-scripts/Augmentation/data/gtd_build_records.m:36-67. Keys are the loader's
# file stems (gantry_dynamic/data.py TRAIN_FILES etc.) without '.mat'.
RECORD_Y_OP = {
    'T1_standstill_Ym30': -0.30, 'T2_standstill_Ym15': -0.15, 'T3_standstill_Y000': 0.00,
    'T4_standstill_Yp15': 0.15,  'T5_standstill_Yp30': 0.30,
    'T6_ysweep_slow': 0.00, 'T7_ysweep_fast': 0.00, 'T8_ysweep_xmix': 0.00,
    'T9_aprbs_30': 0.00, 'T10_aprbs_60': 0.00, 'T11_aprbs_100': 0.00, 'T12_aprbs_yaw': 0.00,
    'T13_lissajous': 0.00, 'T14_lissajous_yaw': 0.00,
    'V1_standstill_Yp10': 0.10, 'V2_aprbs_Ylow': -0.22, 'V3_ysweep_Yp10': 0.10,
    'V4_lissajous_Ym10': -0.10,
    'E1_resonance_sweep': 0.00, 'E2_multisine_Yp22': 0.22, 'E3_aprbs_above': 0.00,
    'E4_multisine_off': 0.00,
}


def _stem(name):
    stem = os.path.basename(str(name))
    return stem[:-4] if stem.endswith('.mat') else stem


def y_op_for(name):
    """Y_op for a record file name, with or without the .mat suffix."""
    stem = _stem(name)
    if stem not in RECORD_Y_OP:
        raise KeyError('no Y_op known for record %r; add it to RECORD_Y_OP from '
                       'gtd_build_records.m' % stem)
    return RECORD_Y_OP[stem]


def build_controller_bank(record_names, ts, ystd, std_u, dtype=torch.float32):
    """The framework's ControllerBank for a set of records, plus each record's row in it.

    ONE bank over whatever list it is given, indexed globally: the controller belongs to a
    trajectory, and train versus validation is a property of the split, not an axis of the
    controller. One (A, B, C, D) per DISTINCT operating point, not per record, because several
    records share one (T6-T14 are all at 0.00) and identical rows would widen the gather for
    nothing.

    Returns (bank, rows, y_ops_unique) with rows[i] the controller row of record_names[i].
    """
    y_ops = [y_op_for(n) for n in record_names]
    uniq = sorted(set(y_ops))
    rows = [uniq.index(v) for v in y_ops]
    mats = [controller_ss(Y_op, float(ts)) for Y_op in uniq]
    A, B, C, D = (np.stack([np.asarray(m[k], float) for m in mats]) for k in range(4))
    return ControllerBank(A, B, C, D, ystd=ystd, std_u=std_u, dtype=dtype), rows, uniq


# ── 5. the one call the entry point makes ───────────────────────────────────────────

def build_closed_loop(fs, norm, cfg, *, train_files, val_files, val_data, verbose=True):
    """The ClosedLoopSimulator for this pipeline. Assign it to `fs.simulator`.

    Not monkey patching: `simulator` is a declared class attribute on SSE_Interconnect with a
    documented no-op default (None = open loop), set on the instance after construction exactly
    as `orth_penalty` already is (D7.1/D7.8).

    The record arguments are KEYWORD-ONLY, deliberately. `train_files` and `val_files` must be in
    the order their System_data_lists were built, and `val_data` must be the SAME object passed
    to fit(); silently swapping two of them attaches the wrong controller to every record and
    produces a plausible loss. The two checks below are what catch that.

    fs         the built fit system (used only for its dtype; the simulator holds no model handles)
    norm       the pipeline's Norm, for ystd and std_u
    cfg        RunConfig, for ts_new: the controller is stepped at the TRAINING rate, not the
               record rate, and Dc is rate dependent. See the module header.
    """
    from model_augmentation.fit_systems.closed_loop import ClosedLoopSimulator

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

    sdl = val_data.sdl if hasattr(val_data, 'sdl') else [val_data]
    if len(sdl) != len(val_files):
        raise ValueError('val_data has %d records but %d val_files were given; the simulator '
                         'would register the wrong controller against each record. This is the '
                         'check that catches val_data and the file list having been built from '
                         'different splits.' % (len(sdl), len(val_files)))
    val_records = [(names[n_tr + i], sd, val_rows[i]) for i, sd in enumerate(sdl)]

    if verbose:
        print('[closed loop] one bank over %d records, %d distinct controllers, Y_op %s'
              % (len(names), bank.n_controllers, y_ops))
        print('[closed loop] stepped at ts = %g s (%d Hz), nc = %d states'
              % (cfg.ts_new, round(1 / cfg.ts_new), bank.nc))
        print('[closed loop] diag(Dc) physical [%s] N/m'
              % ' '.join('%.4e' % v for v in bank.physical_D()[0].diagonal()))
    return ClosedLoopSimulator(bank, train_rows, val_records)
