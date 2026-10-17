"""Shared paths and loaders for the thesis test-set evaluation (THESIS-RESULTS.md section 5).

Evaluation only: models are loaded from the server runs (the 2026-10-11 runs, or the final runs with
THESIS_EVAL=final), scored on the training, validation and test records and the multisine twins
(D-269) with the pipeline's own closed-loop scorer (D-227), and nothing in the pipeline is modified.
The scored interval starts at each model's own k0 = max(na, nb) (29 for the 2026-10-11 runs, 30 for
the final ones, D-250).
"""
import csv
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

DATA_NOISY = TW / 'Data' / 'Coulomb-tanh-and-MSD-lowpass-noise'
DATA_NF = TW / 'Data' / 'Coulomb-tanh-and-MSD-noise-free'
# D-258: evaluation-only multisine twins of the test records (never trained or validated on)
TWINS_NOISY = TW / 'Data' / 'Multisine-test-twins' / 'Coulomb-tanh-and-MSD-lowpass-noise'
TWINS_NF = TW / 'Data' / 'Multisine-test-twins' / 'Coulomb-tanh-and-MSD-noise-free'
FS = 4000.0
AXES = ('X1', 'X2', 'Y')
BAND = (230.0, 297.0)                                  # absorber band on Y (THESIS-RESULTS step 2)
SPLITS = ('train', 'val', 'test', 'twins')             # D-269: every metric for every split

# Which campaign is evaluated: THESIS_EVAL=final selects the final runs; the default keeps the
# 2026-10-11 runs.
_FINAL = os.environ.get('THESIS_EVAL') == 'final'
# THESIS-RESULTS section 4 run table -> SLURM job id (2026-10-11: logs thesis-results-2)
RUNS_1011 = {'R1': 88762, 'R2': 88755, 'R3': 88763, 'R4': 88764, 'R5': 88767, 'R6': 88768, 'R7': 88756}
# The final runs, identified by what config.json records (arm, added states, start, PS2), so no job id
# has to be typed in. Values: (OBC, NX_ANN, detuned start, PS2).
FINAL_DESIGN = {'R1': (True, 2, True, True), 'R2': (True, 2, False, True), 'R3': (False, 2, False, False),
                'R4': (False, 2, True, True), 'R5': (False, 2, False, True), 'R6': (False, 0, False, False),
                'R7': (False, 8, False, True), 'R13': (True, 0, True, False), 'R14': (False, 0, True, False)}


def discover_runs(runs_dir):
    """R label -> SLURM job id of the full runs in `runs_dir` (launch tests and capped runs, which
    record TRUNCATE_S or N_ITS, are ignored). Two full runs of one design is an error to resolve."""
    import json
    found = {}
    for cfg_file in sorted(Path(runs_dir).glob('*/config.json')):
        c = json.load(open(cfg_file, encoding='utf-8'))
        if c.get('TRUNCATE_S') is not None or c.get('N_ITS') is not None or not str(c['run_id']).isdigit():
            continue
        key = (bool(c['OBC']), int(c['NX_ANN']), c.get('COMBO_INIT_DETUNE') is not None, bool(c.get('PS2')))
        for run, design in FINAL_DESIGN.items():
            if design == key:
                if run in found:
                    raise ValueError('two full runs of %s in %s: %s and %s' % (run, runs_dir, found[run], c['run_id']))
                found[run] = int(c['run_id'])
    return found


if _FINAL:
    # where the server writes them (config.save_dir); THESIS_RUNS_DIR if they were copied elsewhere
    RUNS_DIR = Path(os.environ.get('THESIS_RUNS_DIR',
                                   REPO / 'simulations' / 'gantry_subnet' / 'thesis_taf_lowpass_noise_data_fit'))
    RUNS = discover_runs(RUNS_DIR)
    # the selected R8 configuration's output folder (run_bb_search.sh OUT), chosen after the grid
    _bb = os.environ.get('THESIS_BB_DIR')
    BB_RESULT = Path(_bb) / 'result.json' if _bb else None
    BB_CKPT = Path(_bb) / 'localappdata' / 'deepSI' / 'checkpoints' if _bb else None
    if _bb:
        import json as _json
        # '<Class>_<code>_best.pth' (deepSI's System.name + '_best')
        BB_CODE = _json.load(open(BB_RESULT))['checkpoint'].rsplit('_', 2)[-2]
    else:
        BB_CODE = None
else:
    RUNS = RUNS_1011
    RUNS_DIR = TW / 'Results' / '11-10-2026' / 'thesis_taf_lowpass_noise_linear_map'
    # black box: checkpoint folder, deepSI code and result.json of the selected configuration
    BB_CKPT = TW / 'Results' / '11-10-2026' / 'Black-box' / 'checkpoints' / 'bb88757'
    BB_CODE = '5rVESQ'                                 # 88757 index 3, nx16_w08, selected on validation
    BB_RESULT = TW / 'Results' / '11-10-2026' / 'Black-box' / 'thesis_r8_grid' / '3_nx16_w08_n20000_88757' / 'result.json'
# One prediction cache per campaign, so the two can never be mixed (audit 2026-10-16); readers
# check the stored identity with check_identity.
CACHE = TW / 'Results' / 'thesis-eval' / ('final' if _FINAL else '')


def check_identity(name, d):
    """Refuse a cached prediction file that belongs to another model than this campaign's: the
    grey-box job id or the black-box deepSI code must match RUNS / BB_CODE."""
    run = name.split('_')[0]
    want = BB_CODE if run == 'BB' else RUNS.get(run)
    got = d.get('code') if run == 'BB' else d.get('job')
    if want is None or got is None or str(np.asarray(got)) != str(want):
        raise ValueError('cache %s holds %r, this campaign expects %r' % (name, got, want))


def twin_files():
    """The twin record files (relative to TWINS_NOISY), in manifest order."""
    with open(TWINS_NOISY / 'MANIFEST.csv', newline='') as fh:
        return ['%s/%s.mat' % (r['folder'], r['file']) for r in csv.DictReader(fh)]


def multisine(record):
    """True when the record carries the training multisine (MANIFEST `excitation`), so results can
    be reported separately with and without it (D-256)."""
    rows = {}
    for folder in (DATA_NOISY, TWINS_NOISY):
        with open(folder / 'MANIFEST.csv', newline='') as fh:
            rows.update({r['file'].split('_')[0]: r['excitation'] for r in csv.DictReader(fh)})
    return rows[record] == 'multisine'


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
