"""
telica_loader.py
----------------
Load a Telica iter*.log file and return (u, q1, fs) in the same format as
precompute._load_trajectory() so the rest of the precompute/train pipeline
works without modification.

Contract (matches precompute._load_trajectory):
    u   : torch.Tensor  (1, T, 3)  stage force [N]  — [X1, X2, Y]
    q1  : torch.Tensor  (T, 3)     actual stage position [m]  — [X1, X2, Y]
    fs  : float                    sampling rate [Hz] = 20000.0

Beam head: BHL (left) — the beam head covered by the FP model.

Conversion chain (all values from kamtin-data/Telica.mat):
    I [A]  = MF30 [ci] × AmplifierGain          (same for all axes: 0.0020751953125)
    F [N]  = I [A]  × MotorForceConst            (per axis: X=109 N/A, Y=77.6 N/A)
    pos[m] = (M0 - M2) [dpi] × 1e-6             (1 dpi = 1 µm, runFDILCAllHostSwLog.m line 89)
"""

__project_origin__ = "added"

import os

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat

# ── Load machine parameters from Telica.mat ───────────────────────────────────

_HERE     = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.normpath(os.path.join(_HERE, '..', '..', '..'))
_TELICA_MAT = os.path.join(_ROOT, 'kamtin-data', 'Telica.mat')

def _load_telica_params():
    """Read AmplifierGain, MotorForceConst (per axis), and SamplingFrequency
    directly from Telica.mat so no values are hardcoded here."""
    mat = loadmat(_TELICA_MAT, squeeze_me=True, struct_as_record=False)
    mp  = mat['MachineParam']
    ax  = mp.Axes
    amp_gain = float(ax.X.ElectronicHardwareInfo.Motor.AmplifierGain.Value)
    kt_x     = float(ax.X.ElectronicHardwareInfo.Motor.MotorForceConst.Value)
    kt_y     = float(ax.Y.ElectronicHardwareInfo.Motor.MotorForceConst.Value)
    fs_native = float(ax.X.SamplingFrequency.Value)
    return amp_gain, kt_x, kt_y, fs_native

_AMP_GAIN, _KT_X, _KT_Y, _NATIVE_FS = _load_telica_params()

# Per-axis gain vector [ci → N]: [GTRX1, GTRX2, GTRY]
# THEORY: F[N] = MF30[ci] × AmplifierGain[A/ci] × Kt[N/A]
#         (Telica.mat: Axes.X/Y.ElectronicHardwareInfo.Motor)
_CI_TO_N = np.array([
    _AMP_GAIN * _KT_X,   # GTRX1 (X motor 1)
    _AMP_GAIN * _KT_X,   # GTRX2 (X motor 2, same motor type)
    _AMP_GAIN * _KT_Y,   # GTRY  (Y motor, different Kt)
])

# THEORY: must match simulation pipeline which runs at 20 kHz
#         (train_param_recovery.py: FS_NEW = 20000)
_FS_TARGET: float = 20_000.0

# HEURISTIC: M0 and M2 treated as µm → m (×1e-6), consistent with active
#            MATLAB loading code (runFDILCAllHostSwLog.m line 89).
_DPI_TO_M: float = 1e-6

# HEURISTIC: 50 ms of standstill kept before motion start, matching the
#            MATLAB trimming logic (runFDILCAllHostSwLog.m lines 100-110)
_PRE_MOTION_MS: float = 50.0

_BEAM_HEAD = 'BHL'
_AXES      = ('GTRX1', 'GTRX2', 'GTRY')   # column order → [X1, X2, Y]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _clean_header(line: str) -> list:
    """
    Strip :controller.axis suffix and replace . with _ — matches readDataLog()
    in kamtin-data/runFDILCAllHostSwLog.m.

    Example: 'BHL_GTRX1.M2:0.0' → 'BHL_GTRX1_M2'
    """
    fields = line.strip().split('\t')
    cleaned = []
    for f in fields:
        f = f.strip()
        if ':' in f:
            f = f.split(':')[0]
        f = f.replace('.', '_')
        cleaned.append(f)
    # Drop trailing empty column that appears when the line ends with \t
    while cleaned and cleaned[-1] == '':
        cleaned.pop()
    return cleaned


def _find_motion_start(m0_cols: np.ndarray) -> int:
    """
    Return the index of the first sample where any axis setpoint deviates from
    its initial value — matches MATLAB line 100:
        find(abs(BHL_GTRX1_M0 - BHL_GTRX1_M0(1)) > 0, 1, "first")
    (runFDILCAllHostSwLog.m, applied after the ×1e-6 conversion).

    Threshold 1e-9 catches any real controller update while ignoring
    floating-point representation noise below the sub-nm level.
    """
    initial = m0_cols[0]
    deviations = np.any(np.abs(m0_cols - initial) > 1e-9, axis=1)
    idx = int(np.argmax(deviations))
    # argmax returns 0 when no True found — treat as no motion detected
    if not deviations[idx]:
        return 0
    return idx


# ── Public API ────────────────────────────────────────────────────────────────

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

    # ── 1. Read header and data ───────────────────────────────────────────────
    with open(path, 'r') as fh:
        header_line = fh.readline()
    cols = _clean_header(header_line)

    raw = pd.read_csv(path, sep='\t', header=None, names=cols,
                      skiprows=1, engine='python')
    raw = raw.dropna(axis=1, how='all')

    ts_ms = raw['TimeStamp'].to_numpy(dtype=np.float64)
    ts_ms = ts_ms - ts_ms[0]   # zero at first sample

    # ── 2. Native sampling rate ───────────────────────────────────────────────
    # THEORY: Telica.mat Axes.X.SamplingFrequency = 20000 Hz
    #         (SamplingTime = 5e-5 s = 50 µs → Fs = 20 kHz).
    #         Raw .log timestamps are host-side reception times (non-uniform);
    #         MATLAB discards them and replaces with 1/Fs synthetic spacing
    #         (runFDILCAllHostSwLog.m line 92). Same approach used here.
    fs_native = _NATIVE_FS   # 20000 Hz from Telica.mat

    # ── 3. Extract M0, M2, MF30 for the BHL beam head ────────────────────────
    bh = _BEAM_HEAD
    M0   = raw[[f'{bh}_{ax}_M0'   for ax in _AXES]].to_numpy(dtype=np.float64)
    M2   = raw[[f'{bh}_{ax}_M2'   for ax in _AXES]].to_numpy(dtype=np.float64)
    MF30 = raw[[f'{bh}_{ax}_MF30' for ax in _AXES]].to_numpy(dtype=np.float64)

    # ── 4. Trim pre-motion samples ────────────────────────────────────────────
    # Keep _PRE_MOTION_MS of standstill before the first setpoint change.
    motion_idx  = _find_motion_start(M0)
    pre_samples = max(0, int(round(_PRE_MOTION_MS * fs_native / 1000.0)))
    trim_start  = max(0, motion_idx - pre_samples)

    M0    = M0[trim_start:]
    M2    = M2[trim_start:]
    MF30  = MF30[trim_start:]

    # ── 5. Unit conversions ───────────────────────────────────────────────────
    # Actual stage position: M1 = M0 - M2  (M2 = tracking error = M0 - M1)
    # THEORY: pos[m] = (M0 - M2)[dpi] × 1e-6  (1 dpi = 1 µm, MATLAB line 89)
    q1_raw = (M0 - M2) * _DPI_TO_M    # [m]

    # THEORY: F[N] = MF30[ci] × AmplifierGain[A/ci] × Kt[N/A]  (Telica.mat)
    #         _CI_TO_N is a (3,) vector with per-axis gains [GTRX1, GTRX2, GTRY]
    u_raw = MF30 * _CI_TO_N            # [N]

    # ── 6. No resampling needed ───────────────────────────────────────────────
    # Native Fs = 20 kHz = target Fs — pass through directly.
    # (Previous version incorrectly assumed 10 kHz and upsampled 2×.)
    assert abs(fs_native - _FS_TARGET) < 1.0, (
        f'Native Fs ({fs_native} Hz) != target ({_FS_TARGET} Hz) — resampling needed'
    )

    # ── 7. Convert to tensors ─────────────────────────────────────────────────
    u_tensor  = torch.tensor(u_raw,  dtype=dtype).unsqueeze(0)  # (1, T, 3)
    q1_tensor = torch.tensor(q1_raw, dtype=dtype)               # (T, 3)

    return u_tensor, q1_tensor, _FS_TARGET
