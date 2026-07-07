"""
telica_loader.py
----------------
Load a Telica iter*.log file and return (u, q1, fs) in the same format as
precompute._load_trajectory() so the rest of the precompute/train pipeline
works without modification.

Contract (matches precompute._load_trajectory):
    u   : torch.Tensor  (1, T, 3)  stage force [N]  -- [X1, X2, Y]
    q1  : torch.Tensor  (T, 3)     actual stage position [m]  -- [X1, X2, Y]
    fs  : float                    sampling rate [Hz] = 20000.0

Beam head: BHL (left) -- the beam head covered by the FP model.

Signal units in the raw log file:
    M0, M2    : um (micrometres).  pos[m] = raw * 1e-6
    MF30, MF230: already in Amperes.  F[N] = I[A] * Kt[N/A]

Native logging rate: 20 kHz (= 1/SamplingTime, Telica.mat SamplingFrequency
    "The number of samples logged per second" = 20000). Confirmed empirically:
    the notch of the real LX1 controller (dFeedbackControllersTelica.mat,
    designed at 20 kHz) appears at the SAME normalized frequency in the
    iter0 log FRF, which is only possible if the log samples are consecutive
    20 kHz DSP samples. See D-073 and
    simulations/gantry_subnet/diagnostics/controller_fingerprint/.
    The earlier 10 kHz assumption (D-061, FsHz = 1/(2*TsSec)) stretched the
    time axis by 2x and is superseded. No resampling is performed.

Conversion source:
    Kt_X = 109 N/A, Kt_Y = 77.6 N/A  (Telica.mat Axes.X/Y.Motor.MotorForceConst)
"""

__project_origin__ = "added"

import os

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat

# -- Load motor force constants from Telica.mat -------------------------------

_HERE     = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.normpath(os.path.join(_HERE, '..', '..', '..'))
_TELICA_MAT = os.path.join(_ROOT, 'kamtin-data', 'Telica.mat')

def _load_telica_params():
    """Read MotorForceConst (per axis) and SamplingFrequency from Telica.mat."""
    mat = loadmat(_TELICA_MAT, squeeze_me=True, struct_as_record=False)
    mp  = mat['MachineParam']
    ax  = mp.Axes
    kt_x      = float(ax.X.ElectronicHardwareInfo.Motor.MotorForceConst.Value)
    kt_y      = float(ax.Y.ElectronicHardwareInfo.Motor.MotorForceConst.Value)
    # THEORY: fs_native = 1/SamplingTime = SamplingFrequency.Value = 20000 Hz
    #         (Telica.mat: "The number of samples logged per second").
    #         Empirical confirmation via controller-notch fingerprint: D-073.
    #         The old FsHz = 1/(2*TsSec) = 10 kHz (D-061) is superseded.
    ts_sec    = float(ax.X.SamplingTime.Value)
    fs_native = 1.0 / ts_sec
    return kt_x, kt_y, fs_native

_KT_X, _KT_Y, _NATIVE_FS = _load_telica_params()

# Per-axis force gain [A -> N]: [GTRX1, GTRX2, GTRY]
# THEORY: F[N] = MF30[A] * Kt[N/A]  (MF30 is logged in Amperes -- see module doc)
_A_TO_N = np.array([_KT_X, _KT_X, _KT_Y])

# THEORY: must match simulation pipeline (train_param_recovery.py: FS_NEW = 20000)
_FS_TARGET: float = 20_000.0

# HEURISTIC: M0 and M2 in um -> m (*1e-6), consistent with
#            runFDILCAllHostSwLog.m line 89.
_DPI_TO_M: float = 1e-6

# HEURISTIC: 50 ms of standstill kept before motion start, matching
#            runFDILCAllHostSwLog.m lines 100-110.
_PRE_MOTION_MS: float = 50.0

_BEAM_HEAD = 'BHL'
_AXES      = ('GTRX1', 'GTRX2', 'GTRY')   # column order -> [X1, X2, Y]


# -- Internal helpers ---------------------------------------------------------

def _clean_header(line: str) -> list:
    """
    Strip :controller.axis suffix and replace . with _ -- matches readDataLog()
    in kamtin-data/runFDILCAllHostSwLog.m.

    Example: 'BHL_GTRX1.M2:0.0' -> 'BHL_GTRX1_M2'
    """
    fields = line.strip().split('\t')
    cleaned = []
    for f in fields:
        f = f.strip()
        if ':' in f:
            f = f.split(':')[0]
        f = f.replace('.', '_')
        cleaned.append(f)
    while cleaned and cleaned[-1] == '':
        cleaned.pop()
    return cleaned


def _find_motion_start(m0_cols: np.ndarray) -> int:
    """
    Return the index of the first sample where any axis setpoint deviates from
    its initial value -- matches MATLAB line 100:
        find(abs(BHL_GTRX1_M0 - BHL_GTRX1_M0(1)) > 0, 1, "first")

    m0_cols is in raw um units (integers like -60000 for -60 mm).
    Threshold 0.5 um catches any real controller update (M0 steps in integer um)
    while ignoring floating-point noise.
    """
    initial = m0_cols[0]
    deviations = np.any(np.abs(m0_cols - initial) > 0.5, axis=1)
    idx = int(np.argmax(deviations))
    if not deviations[idx]:
        return 0
    return idx


# -- Public API ---------------------------------------------------------------

def load_telica_log(path: str, dtype=torch.float64):
    """
    Load one Telica iter*.log file.

    Parameters
    ----------
    path  : str            absolute or relative path to the .log file
    dtype : torch.dtype    output tensor dtype (default float64)

    Returns
    -------
    u   : (1, T, 3) tensor   stage force [N]           axes: [X1, X2, Y]
    q1  : (T, 3)   tensor    actual stage position [m]  axes: [X1, X2, Y]
    fs  : float              20000.0 Hz
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Telica log not found: {path}')

    # -- 1. Read header and data ----------------------------------------------
    with open(path, 'r') as fh:
        header_line = fh.readline()
    cols = _clean_header(header_line)

    # index_col=False: prevents pandas from treating the TimeStamp column as
    # the row index, which would shift all column assignments by one.
    raw = pd.read_csv(path, sep='\t', header=None, names=cols,
                      skiprows=1, engine='python', index_col=False)
    raw = raw.dropna(axis=1, how='all')

    # -- 2. Native sampling rate ----------------------------------------------
    # THEORY: fs_native = 1/SamplingTime = 20 kHz (Telica.mat; D-073).
    fs_native = _NATIVE_FS   # 20000 Hz

    # -- 3. Extract M0, M2, MF30 for the BHL beam head -----------------------
    bh = _BEAM_HEAD
    M0   = raw[[f'{bh}_{ax}_M0'   for ax in _AXES]].to_numpy(dtype=np.float64)
    M2   = raw[[f'{bh}_{ax}_M2'   for ax in _AXES]].to_numpy(dtype=np.float64)
    MF30 = raw[[f'{bh}_{ax}_MF30' for ax in _AXES]].to_numpy(dtype=np.float64)

    # -- 4. Trim pre-motion samples -------------------------------------------
    # Keep _PRE_MOTION_MS of standstill before the first setpoint change.
    # M0 is in raw um (integer values); threshold 0.5 catches any step.
    motion_idx  = _find_motion_start(M0)
    pre_samples = max(0, int(round(_PRE_MOTION_MS * fs_native / 1000.0)))
    trim_start  = max(0, motion_idx - pre_samples)

    M0    = M0[trim_start:]
    M2    = M2[trim_start:]
    MF30  = MF30[trim_start:]

    # -- 5. Unit conversions --------------------------------------------------
    # THEORY: pos[m] = (M0 - M2)[um] * 1e-6  (runFDILCAllHostSwLog.m line 89)
    q1_raw = (M0 - M2) * _DPI_TO_M    # [m]

    # THEORY: F[N] = MF30[A] * Kt[N/A]  (MF30 logged in Amperes, see module doc)
    u_raw = MF30 * _A_TO_N             # [N]

    # -- 6. No resampling: native rate IS the pipeline rate (D-073) -----------
    # The log samples are consecutive 20 kHz DSP samples; the old 2x upsample
    # (10 kHz assumption, D-061) stretched the time axis and is removed.
    if fs_native != _FS_TARGET:
        raise ValueError(
            f'Native rate {fs_native} Hz != pipeline rate {_FS_TARGET} Hz; '
            f'resampling was removed per D-073 -- revisit if this triggers.')

    # -- 7. Convert to tensors ------------------------------------------------
    u_tensor  = torch.tensor(u_raw,  dtype=dtype).unsqueeze(0)  # (1, T, 3)
    q1_tensor = torch.tensor(q1_raw, dtype=dtype)               # (T, 3)

    return u_tensor, q1_tensor, _FS_TARGET


def load_telica_log_cl(path: str, dtype=torch.float64):
    """
    Closed-loop evaluation variant of load_telica_log (D-074).

    Same file reading, trimming and unit conventions as load_telica_log, but
    returns the signals the closed-loop simulation needs. The training
    contract of load_telica_log is untouched.

    Returns dict with keys:
        r     : (T, 3) tensor  reference position [m]         M0 * 1e-6
        q1    : (T, 3) tensor  measured position [m]          (M0 - M2) * 1e-6
        u_ff  : (T, 3) tensor  ILC feedforward force [N]      (MF30 - MF230) * Kt
        i_fb  : (T, 3) tensor  logged feedback current [A]    MF230
        fs    : float          20000.0
    Note: tracking error in meters = r - q1 = M2 * 1e-6.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Telica log not found: {path}')

    with open(path, 'r') as fh:
        header_line = fh.readline()
    cols = _clean_header(header_line)
    raw = pd.read_csv(path, sep='\t', header=None, names=cols,
                      skiprows=1, engine='python', index_col=False)
    raw = raw.dropna(axis=1, how='all')

    bh = _BEAM_HEAD
    M0    = raw[[f'{bh}_{ax}_M0'    for ax in _AXES]].to_numpy(dtype=np.float64)
    M2    = raw[[f'{bh}_{ax}_M2'    for ax in _AXES]].to_numpy(dtype=np.float64)
    MF30  = raw[[f'{bh}_{ax}_MF30'  for ax in _AXES]].to_numpy(dtype=np.float64)
    MF230 = raw[[f'{bh}_{ax}_MF230' for ax in _AXES]].to_numpy(dtype=np.float64)

    motion_idx  = _find_motion_start(M0)
    pre_samples = max(0, int(round(_PRE_MOTION_MS * _NATIVE_FS / 1000.0)))
    trim_start  = max(0, motion_idx - pre_samples)
    M0, M2, MF30, MF230 = (M0[trim_start:], M2[trim_start:],
                           MF30[trim_start:], MF230[trim_start:])

    r_m    = M0 * _DPI_TO_M                    # [m]
    q1_m   = (M0 - M2) * _DPI_TO_M             # [m]
    u_ff_n = (MF30 - MF230) * _A_TO_N          # [N]

    return {
        'r':    torch.tensor(r_m,    dtype=dtype),
        'q1':   torch.tensor(q1_m,   dtype=dtype),
        'u_ff': torch.tensor(u_ff_n, dtype=dtype),
        'i_fb': torch.tensor(MF230,  dtype=dtype),
        'fs':   _FS_TARGET,
    }
