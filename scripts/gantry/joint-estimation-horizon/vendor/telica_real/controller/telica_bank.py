"""The Telica controller as the framework's ControllerBank, residual form (G3, TR-008/TR-011).

    u_plant[k] = u_data[k] + diag(Kt s_F) K (y_data[k] - y_model[k])        [N]

K is the per-axis cascade of second-order sections: the IDENTIFIED logged path by default
(`telica_sos_identified.npz`, TR-011), or the zpk file (`source='zpk'`). An identified integer lag
d > 0 is realised as d extra delay states (the output becomes strictly proper). One controller for
every record: the Telica controller is not scheduled on Y, so the bank has ONE row and every
window maps to row 0. Defined at 20 kHz only (TR-008): any other step is refused.
"""
import os

import numpy as np
import torch

from model_augmentation.fit_systems.closed_loop import ControllerBank, ClosedLoopSimulator
from params import telica_params as tp
from controller import telica_ctrl as tc

ID_FILE = os.path.join(tc.HERE, 'telica_sos_identified.npz')
ID_FILE_5K = os.path.join(tc.HERE, 'telica_sos_identified_5k.npz')     # TR-022
SOURCE_TS = {'zpk': tp.TS, 'identified': tp.TS, 'identified_5k': 4 * tp.TS}


def source_for_ts(ts):
    """The identified controller that exists at step `ts` (20 kHz or 5 kHz), else ValueError."""
    for src in ('identified', 'identified_5k'):
        if abs(float(ts) - SOURCE_TS[src]) < 1e-12:
            return src
    raise ValueError(f'no Telica controller at ts = {ts} s; available: 20 kHz (TR-008) and 5 kHz '
                     f'(TR-022)')


def controller_sos(source='identified'):
    """({axis: sos}, {axis: lag}) for 'identified', 'identified_5k' or 'zpk'."""
    if source == 'zpk':
        return tc.sos_bank(), {a: 0 for a in tc.AXES}
    if source not in ('identified', 'identified_5k'):
        raise ValueError(source)
    z = np.load(ID_FILE if source == 'identified' else ID_FILE_5K)
    lags = dict(zip(tc.AXES, [int(v) for v in z['lags']]))
    return {a: z[a] for a in tc.AXES}, lags


def _with_delay(A, B, C, D, d):
    """SISO (A,B,C,D) followed by a d-sample delay, as one state space."""
    if d == 0:
        return A, B, C, D
    if d < 0:
        raise ValueError('a negative lag (current leading the error) is not causal; it cannot '
                         'be part of the loop and must be treated as a logging offset')
    n = A.shape[0]
    N = n + d
    A2 = np.zeros((N, N)); B2 = np.zeros(N); C2 = np.zeros(N)
    A2[:n, :n] = A
    A2[n, :n] = C                     # first delay state = controller output
    B2[:n] = B
    B2[n] = D
    for k in range(1, d):
        A2[n + k, n + k - 1] = 1.0
    C2[N - 1] = 1.0
    return A2, B2, C2, 0.0


def physical_ss(source='identified', reading=tp.FORCE_SCALE_READING):
    """3x3 block-diagonal (A, B, C, D): stage error [m] -> stage force [N]."""
    sos, lags = controller_sos(source)
    g = np.asarray(tp.a_to_n(reading))
    blocks = []
    for j, ax in enumerate(tc.AXES):
        A, B, C, D = tc.axis_ss(sos[ax])
        blocks.append(_with_delay(A, B, C, D, lags[ax]))
    n = sum(b[0].shape[0] for b in blocks)
    A = np.zeros((n, n)); B = np.zeros((n, 3)); C = np.zeros((3, n)); D = np.zeros((3, 3))
    i = 0
    for j, (Aj, Bj, Cj, Dj) in enumerate(blocks):
        m = Aj.shape[0]
        A[i:i + m, i:i + m] = Aj
        B[i:i + m, j] = Bj
        C[j, i:i + m] = g[j] * Cj
        D[j, j] = g[j] * Dj
        i += m
    return A, B, C, D


def build_telica_bank(ts, ystd, std_u, dtype=torch.float32, source='identified',
                      reading=tp.FORCE_SCALE_READING, compute_dtype=torch.float64):
    """ControllerBank with ONE row (the machine has one controller for all operating points)."""
    if abs(float(ts) - SOURCE_TS[source]) > 1e-12:
        raise ValueError(f'controller {source!r} exists at Ts = {SOURCE_TS[source]} s only '
                         f'(TR-008 / TR-022); got ts = {ts}.')
    A, B, C, D = physical_ss(source, reading)
    # TR-012: float64 controller arithmetic regardless of the rollout dtype.
    return ControllerBank(A[None], B[None], C[None], D[None], ystd=ystd, std_u=std_u,
                          dtype=dtype, compute_dtype=compute_dtype)


def build_closed_loop_telica(fs, norm, cfg, *, train_files, val_files, val_data,
                             source=None, verbose=True):
    """Drop-in for gantry_dynamic.controller.build_closed_loop on real data (G6)."""
    names = [str(f) for f in list(train_files) + list(val_files)]      # 'split/op/iter' keys
    source = source_for_ts(cfg.ts_new) if source is None else source
    if len(set(names)) != len(names):
        raise ValueError('a record appears twice across train_files/val_files: %s' % names)
    bank = build_telica_bank(cfg.ts_new, ystd=norm.ystd, std_u=norm.std_u, dtype=cfg.dtype_pt,
                             source=source)
    sdl = val_data.sdl if hasattr(val_data, 'sdl') else [val_data]
    if len(sdl) != len(val_files):
        raise ValueError('val_data has %d records but %d val_files' % (len(sdl), len(val_files)))
    train_rows = [0] * len(train_files)
    val_records = [(names[len(train_files) + i], sd, 0) for i, sd in enumerate(sdl)]
    if verbose:
        print('[closed loop] Telica controller (%s), one row, nc = %d, ts = %g s'
              % (source, bank.nc, cfg.ts_new))
        print('[closed loop] diag(Dc) physical [%s] N/m'
              % ' '.join('%.4e' % v for v in bank.physical_D()[0].diagonal()))
    compiled = None
    if cfg.compile_mode is not None:
        from gantry_dynamic.model import COMPILE_BACKEND
        compiled = (torch.compile(fs.hfn, backend=COMPILE_BACKEND, mode=cfg.compile_mode,
                                  fullgraph=True),
                    torch.compile(fs.hfn.output_only, backend=COMPILE_BACKEND,
                                  mode=cfg.compile_mode, fullgraph=True))
    return ClosedLoopSimulator(bank, train_rows, val_records,
                               checkpoint_chunk=cfg.checkpoint_chunk, compiled=compiled)
