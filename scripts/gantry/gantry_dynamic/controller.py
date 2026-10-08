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
controllers across the 22 records. The thesis data instead use ONE controller, K1 designed at
Y = 0, stored in each record's meta (section 4b, D-221).

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


def controller_ss(Y_op, ts, gain=1.0):
    """Cfb at sample time `ts`, as one 3-in 3-out state space (A, B, C, D).

    `gain` scales the whole controller (D-227: K2-g+ = 1.2 K1); the controller is linear, so
    scaling its output matrices C and D is the scaled controller exactly. 1.0 = Cfb itself.
    """
    cfb, _ = build_cfb_at(Y_op, ts)
    A, B, C, D = _tf_to_ss_batch(cfb)
    return A, B, gain * C, gain * D


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
    # D-206: Telica-derived operational records, from gtd_build_records_telica.m.
    # Y_op is BOTH the controller linearisation point and the record's starting Y, and it
    # fixes which Y lattice the record dwells on (Y_op + k*0.080). Training occupies
    # residues Y_op mod 0.080 in {0.00, 0.02, 0.06}; validation and test take the held-out
    # residue 0.04, so their dwell positions are disjoint from training's (verified: 23
    # training dwell positions, 8 held-out, 0 shared). Do not "tidy" VP2 to +0.10 or EP1 to
    # 0.00: those sit on training lattices and would destroy the split.
    'TP1_telica_y000': 0.00, 'TP2_telica_y000_rev': 0.00,
    'TP3_telica_yp06': 0.06, 'TP4_telica_ym06': -0.06,
    'VP1_telica_ylow': -0.20, 'VP2_telica_yp12': 0.12,
    'EP1_telica_test': -0.04,
}


def _stem(name):
    stem = os.path.basename(str(name))
    return stem[:-4] if stem.endswith('.mat') else stem


# ── 4b. records that carry their own controller (D-221) ─────────────────────────────
# The thesis data (Thesis-writeup/Data) store meta.controller = (name, design_Y, gain, fbw_Hz,
# rule). Every training and validation record uses K1: ruleOfThumb on the section-1 rigid nominal
# plant at design_Y = 0, gain 1, FBW (DATA-DESIGN.md 5.11). The loader registers each record it
# reads. The bank builds a controller from a design Y and a gain, so a stored controller is accepted
# at FBW; its gain (E5: K2, gain 1.2) is used only when a bank is asked to (D-227,
# RunConfig.controller_gain), and a record with gain != 1 is refused otherwise, as before.
_STORED_DESIGN_Y = {}
_STORED_GAIN = {}
_UNSUPPORTED = {}


def register_stored_controller(name, ctrl_name, design_y, gain, fbw_hz):
    stem = _stem(name)
    if abs(fbw_hz - FBW) < 1e-9:
        _STORED_DESIGN_Y[stem] = design_y
        _STORED_GAIN[stem] = float(gain)
        _UNSUPPORTED.pop(stem, None)
    else:
        _UNSUPPORTED[stem] = '%s (gain %g, %g Hz)' % (ctrl_name, gain, fbw_hz)
        _STORED_DESIGN_Y.pop(stem, None)
        _STORED_GAIN.pop(stem, None)


def y_op_for(name, use_gain=False):
    """Controller design Y for a record file name, with or without the .mat suffix.

    Legacy records: their RECORD_Y_OP entry. Thesis records: the design_Y of their stored K1.
    A record stored with a gain other than 1 is refused unless `use_gain` (D-227), so no caller
    gets K1's design point for a K2 record without asking.
    """
    stem = _stem(name)
    if stem in RECORD_Y_OP:
        return RECORD_Y_OP[stem]
    if stem in _STORED_DESIGN_Y:
        gain_for(stem, use_gain)                  # raises for gain != 1 without use_gain
        return _STORED_DESIGN_Y[stem]
    if stem in _UNSUPPORTED:
        raise KeyError('record %r uses controller %s; the bank builds ruleOfThumb at gain 1 and '
                       '%g Hz from a design Y only, so this record cannot be trained or validated '
                       'on in closed loop (D-221)' % (stem, _UNSUPPORTED[stem], FBW))
    raise KeyError('no Y_op known for record %r; add it to RECORD_Y_OP from '
                   'gtd_build_records.m, or load it through gantry_dynamic.data so its stored '
                   'controller is registered (D-221)' % stem)


def gain_for(name, use_gain=False):
    """The stored gain of a record's controller; 1.0 unless stored otherwise. D-227.

    A gain other than 1 is returned only with `use_gain`; without it the record is refused, so a
    record generated with K2 is never silently scored through K1.
    """
    stem = _stem(name)
    g = _STORED_GAIN.get(stem, 1.0)
    if abs(g - 1.0) > 1e-12 and not use_gain:
        raise KeyError('record %r was generated with %g x K1; set RunConfig.controller_gain to '
                       'score it with its own controller (D-227)' % (stem, g))
    return g


def build_controller_bank(record_names, ts, ystd, std_u, dtype=torch.float32, use_gain=False):
    """The framework's ControllerBank for a set of records, plus each record's row in it.

    ONE bank over whatever list it is given, indexed globally: the controller belongs to a
    trajectory, and train versus validation is a property of the split, not an axis of the
    controller. One (A, B, C, D) per DISTINCT controller, not per record, because several
    records share one (T6-T14 are all at 0.00) and identical rows would widen the gather for
    nothing. A controller is its design Y and, with `use_gain` (D-227), its stored gain.

    Returns (bank, rows, uniq) with rows[i] the controller row of record_names[i]; uniq lists
    the distinct design Y values, or (design Y, gain) pairs when `use_gain` is set.
    """
    keys = [(y_op_for(n, use_gain), gain_for(n, use_gain)) for n in record_names]
    uniq = sorted(set(keys))
    rows = [uniq.index(k) for k in keys]
    mats = [controller_ss(Y_op, float(ts), g) for Y_op, g in uniq]
    A, B, C, D = (np.stack([np.asarray(m[k], float) for m in mats]) for k in range(4))
    bank = ControllerBank(A, B, C, D, ystd=ystd, std_u=std_u, dtype=dtype)
    return bank, rows, (uniq if use_gain else [y for y, _ in uniq])


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
    # CHANGED (D-169): cfg.checkpoint_chunk reaches the rollout here. The bank itself is left on
    # the CPU: closed_loop_rollout re-homes it to the data's device on first use, which is the
    # only thing that survives deepSI's per-validation cpu()/cuda() flip.
    #
    # CHANGED (D-169): compilation is built HERE, where `fs` is already in hand, and handed to the
    # simulator rather than substituted for `fs.hfn` -- see ClosedLoopSimulator.__init__ for why
    # that distinction is what keeps validation and the diagnostics eager.
    #
    # BOTH callables are compiled. torch.compile(module) returns an OptimizedModule that proxies
    # attribute access, so `.output_only` on it is still the uncompiled bound method; compiling
    # only the module would leave the rollout half compiled and roughly halve the speedup.
    compiled = None
    if cfg.compile_mode is not None:
        import torch                                                    # noqa: E402
        from .model import COMPILE_BACKEND                              # noqa: E402
        compiled = (
            torch.compile(fs.hfn, backend=COMPILE_BACKEND, mode=cfg.compile_mode,
                          fullgraph=True),
            torch.compile(fs.hfn.output_only, backend=COMPILE_BACKEND, mode=cfg.compile_mode,
                          fullgraph=True),
        )
        if verbose:
            print('[closed loop] training rollout COMPILED: backend=%s mode=%s fullgraph=True'
                  % (COMPILE_BACKEND, cfg.compile_mode))
            print('[closed loop] validation and diagnostics stay EAGER (they use fs.hfn directly)')
    return ClosedLoopSimulator(bank, train_rows, val_records,
                               checkpoint_chunk=cfg.checkpoint_chunk, compiled=compiled)
