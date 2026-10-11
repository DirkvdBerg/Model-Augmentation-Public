"""Shared paths and loaders for the thesis test-set evaluation (THESIS-RESULTS.md section 5).

Evaluation only: models are loaded from the 2026-10-11 server runs, scored on the test records with
the pipeline's own closed-loop scorer (D-227), and nothing in the pipeline is modified.
"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TW = HERE.parents[1]                                   # Thesis-writeup/
REPO = TW.parent
for p in (REPO, REPO / 'scripts' / 'gantry', REPO / 'scripts' / 'gantry' / 'thesis-results'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

RUNS_DIR = TW / 'Results' / '11-10-2026' / 'thesis_taf_lowpass_noise_linear_map'
BB_CKPT = TW / 'Results' / '11-10-2026' / 'Black-box' / 'checkpoints' / 'bb88757'
CACHE = TW / 'Results' / 'thesis-eval'
DATA_NOISY = TW / 'Data' / 'Coulomb-tanh-and-MSD-lowpass-noise'
DATA_NF = TW / 'Data' / 'Coulomb-tanh-and-MSD-noise-free'
FS = 4000.0
K0 = 29                                                # encoder lag; every score starts here
AXES = ('X1', 'X2', 'Y')
BAND = (230.0, 297.0)                                  # absorber band on Y (THESIS-RESULTS step 2)

# THESIS-RESULTS section 4 run table -> SLURM job id of the 2026-10-11 runs (logs thesis-results-2)
RUNS = {'R1': 88762, 'R2': 88755, 'R3': 88763, 'R4': 88764, 'R5': 88767, 'R6': 88768, 'R7': 88756}
BB_CODE = '5rVESQ'                                     # 88757 index 3, nx16_w08, selected on validation


def record_name(f):
    return os.path.basename(str(f).replace('\\', '/')).split('_')[0]


def band_rms(e, fs=FS, band=BAND):
    """RMS of e (N,) in [band) by Parseval on the one-sided rfft (as band_compare.band_rms)."""
    N = len(e)
    X = np.fft.rfft(e)
    f = np.fft.rfftfreq(N, 1 / fs)
    w = np.full(len(f), 2.0)
    w[0] = 1.0
    if N % 2 == 0:
        w[-1] = 1.0
    m = (f >= band[0]) & (f < band[1])
    return float(np.sqrt(np.sum(w[m] * np.abs(X[m]) ** 2) / N ** 2))
