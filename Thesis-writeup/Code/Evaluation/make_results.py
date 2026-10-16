"""Saved evaluation results -> one CSV per thesis figure or table (Results/<stem>.csv).

Sources, so no figure ever needs a rollout or training again:
  - the runs' own results npz (cl_test_rms: per test record and axis, rows model, physics part,
    baseline start, baseline true; y_hat_test_cl: the full model prediction of I1)
  - Results/thesis-eval/preds_<run>.npz (eval_greybox.py: predictions from k0) and preds_BB.npz
    (eval_blackbox.py: black box, y, r and the noise-free twin y_nf from k0)
  - Results/res_obc_overlap.csv (eval_overlap.py: the two inner products and the learned mode)
Every error is RMS(y_hat - y) from the model's k0 on the noisy record, in metres, and beside it
NRMS_e = RMS / sigma_e, sigma_e the servo-error std over the same samples (D-256, D-262). The floor
is the noisy record minus its noise-free twin. E5 is not scored (user 2026-10-11). `res_errors`
lists every model on every split (train, val, test, twins; D-269) from the preds caches.

Usage: python Thesis-writeup/Code/Evaluation/make_results.py [stem ...]   (default: all)
"""
import csv
import json
import sys

import numpy as np
from scipy.io import loadmat
from scipy.signal import welch

from eval_common import (TW, CACHE, RUNS_DIR, RUNS, AXES, FS, DATA_NOISY, SPLITS, record_name, multisine,
                         check_identity)

RESULTS = TW / 'Results'
INTERP = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6']
ROW = {'model': 0, 'physics': 1, 'base_start': 2, 'base_true': 3}


def cache(name):
    d = dict(np.load(CACHE / ('preds_%s.npz' % name), allow_pickle=True))
    check_identity(name, d)                                 # never mix campaigns (audit 2026-10-16)
    return d


def saved(run, split='test'):
    jid = RUNS[run]
    s = np.load(RUNS_DIR / str(jid) / ('gantry_results_%d.npz' % jid), allow_pickle=True)
    recs = [record_name(f) for f in s['cl_%s_files' % split]]
    return s, recs


def run_rms(run, row, records, split='test'):
    """(len(records), 3) per-record per-axis RMS from the run's own report (val or test)."""
    s, recs = saved(run, split)
    return np.stack([s['cl_%s_rms' % split][ROW[row], recs.index(n)] for n in records])


def bb_rms(records, key='bb', split='test'):
    bb = cache('BB' if split == 'test' else 'BB_%s' % split)
    ix = {n: i for i, n in enumerate(bb['records'])}
    return np.stack([np.sqrt(np.mean((bb[key][ix[n]] - bb['y'][ix[n]]) ** 2, axis=0)) for n in records])


def floor_rms(records, split='test'):
    return bb_rms(records, key='y_nf', split=split)


def sigma_e(records, split='test'):
    """(len(records), 3) servo-error std per record and axis over the scored samples (D-262)."""
    bb = cache('BB' if split == 'test' else 'BB_%s' % split)
    ix = {n: i for i, n in enumerate(bb['records'])}
    return np.stack([bb['sigma_e'][ix[n]] for n in records])


def write(stem, header, rows):
    out = RESULTS / ('%s.csv' % stem)
    with open(out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print('wrote', out)


def f(v):
    return '%.6e' % v


# ------------------------------------------------------------------ Fig. 1: LPV vs LTI baseline
def lpv_lti():
    d = cache('R5')
    jx = {n: j for j, n in enumerate(d['records'])}
    recs = ['I2', 'E1', 'I3', 'E2']
    yop = {r['file'].split('_')[0]: float(r['Y_op']) for r in
           csv.DictReader(open(DATA_NOISY / 'MANIFEST.csv', newline=''))}
    rows = []
    for n in recs:
        y = d['y'][jx[n]]
        flo = floor_rms([n])[0]
        for model, key in (('LPV', 'base_true'), ('LTI', 'base_lti')):
            a = np.sqrt(np.mean((d[key][jx[n]] - y) ** 2, axis=0))
            rows.append([n, yop[n], model] + [f(v) for v in a] + [f(v) for v in flo])
    write('res_lpv_lti', ['record', 'Y_op', 'model'] + ['rms_%s' % a for a in AXES]
          + ['floor_%s' % a for a in AXES], rows)


# ------------------------------------------------------------------ Table I: augmentation structure
def structure(split='val'):
    """Mean over the records of a split of the per-record RMS and NRMS_e per axis. Appendix design
    choices are decided on VALIDATION (D-269); test and training rows are in res_errors."""
    items = [('Baseline, true parameters', 'R5', 'base_true'),
             ('Baseline, detuned start', 'R1', 'base_start'),
             ('Static parallel, $n_a=0$', 'R6', 'model'),
             ('Plain start, $n_a=2$', 'R3', 'model'),
             ('PS start, $n_a=2$', 'R5', 'model'),
             ('PS start, $n_a=2$, physics part only', 'R5', 'physics'),
             ('PS start, $n_a=8$', 'R7', 'model'),
             ('PS start, $n_a=2$, detuned start', 'R4', 'model'),
             ('Final model (OBC), detuned start', 'R1', 'model')]
    recs = INTERP if split == 'test' else [record_name(n) for n in saved('R5', split)[1]]
    sg = sigma_e(recs, split)
    both = lambda a: [f(v) for v in a.mean(0)] + [f(v) for v in (a / sg).mean(0)]   # noqa: E731
    rows = [[lbl, run, RUNS[run], row] + both(run_rms(run, row, recs, split)) for lbl, run, row in items]
    rows.append(['Black box', 'R8', '', 'bb'] + both(bb_rms(recs, split=split)))
    rows.append(['Noise floor', '', '', 'floor'] + both(floor_rms(recs, split)))
    write('res_structure', ['label', 'run', 'job', 'row'] + ['rms_%s' % a for a in AXES]
          + ['nrms_e_%s' % a for a in AXES], rows)


# ------------------------------------------------------------------ Fig. 2: error spectrum
def spectrum(split='val'):
    """ASD of the Y error on the multisine records of a split (Welch, averaged). Validation decides
    the PS vs plain choice (D-269); split='test' gives the earlier I1, I2 view."""
    sfx = '' if split == 'test' else '_%s' % split
    r3, r5, bb = cache('R3' + sfx), cache('R5' + sfx), cache('BB' + sfx)
    recs = ['I1', 'I2'] if split == 'test' else [n for n in r5['records'] if multisine(str(n))]
    ix = lambda d, n: list(d['records']).index(n)                     # noqa: E731
    cols = {}
    for tag, src, key in (('baseline', r5, 'base_true'), ('plain', r3, 'model'), ('ps', r5, 'model')):
        p = [welch(src[key][ix(src, n), :, 2] - src['y'][ix(src, n), :, 2], fs=FS, nperseg=4096)
             for n in recs]
        cols[tag] = np.sqrt(np.mean([q[1] for q in p], axis=0))
    p = [welch(bb['y_nf'][ix(bb, n), :, 2] - bb['y'][ix(bb, n), :, 2], fs=FS, nperseg=4096) for n in recs]
    fr = p[0][0]
    cols['floor'] = np.sqrt(np.mean([q[1] for q in p], axis=0))
    write('res_ps_spectrum', ['f_Hz'] + ['asd_%s' % k for k in cols],
          [['%.4f' % fr[k]] + [f(cols[c][k]) for c in cols] for k in range(len(fr))])


# ------------------------------------------------------------------ Table II: combinations
def combinations():
    """Ten combinations (true, start, learned) and test RMS of R4, R1, R5, R2. `params_init` is
    the START vector, so the true values come from a correct-start run (both must agree)."""
    npz = {r: saved(r)[0] for r in ('R4', 'R1', 'R5', 'R2')}
    true = npz['R5']['params_init']
    assert np.allclose(true, npz['R2']['params_init'], rtol=1e-12)
    rows = []
    for run in ('R4', 'R1', 'R5', 'R2'):
        s = npz[run]
        cfg = json.loads(str(s['config']))
        det = cfg.get('COMBO_INIT_DETUNE') or [1.0] * 10
        start, learned = s['params_init'], s['params_learned']
        assert np.allclose(start, true * np.asarray(det), rtol=1e-6), run
        test = run_rms(run, 'model', INTERP).mean(0)
        for k, n in enumerate(str(x) for x in s['param_names']):
            rows.append([run, RUNS[run], 'obc' if cfg['OBC'] else 'u',
                         'detuned' if cfg.get('COMBO_INIT_DETUNE') else 'correct',
                         n, '%.8e' % true[k], '%.8e' % start[k], '%.8e' % learned[k]] + [f(v) for v in test])
    write('res_obc_combinations', ['run', 'job', 'arm', 'start', 'combination', 'true', 'start_value',
                                   'learned'] + ['test_rms_%s' % a for a in AXES], rows)


# ------------------------------------------------------------------ Fig. 3: generalisation
def generalisation():
    recs = list(cache('BB')['records'])
    rows = []
    # baseline = the detuned start of the final model (user 2026-10-11): the model before augmentation
    for model, vals in (('baseline', run_rms('R1', 'base_start', recs)), ('final', run_rms('R1', 'model', recs)),
                        ('bb', bb_rms(recs)), ('floor', floor_rms(recs))):
        for n, a, sg in zip(recs, vals, sigma_e(recs)):
            rows.append([n, model] + [f(v) for v in a] + [f(np.mean(a))] + [f(v) for v in a / sg])
    write('res_generalisation', ['record', 'model'] + ['rms_%s' % a for a in AXES] + ['rms_mean']
          + ['nrms_e_%s' % a for a in AXES], rows)


# ------------------------------------------------------------------ Fig. 4: time trace
def time_trace():
    """I1, Y axis: measured, LPV baseline (R5 cache, true parameters), final model (R1's own saved
    prediction of I1), reference; zoom window chosen from the baseline and reference only:
    the move with the largest baseline error RMS, from its start to 50 ms after it stops."""
    s, _ = saved('R1')
    r5, bb = cache('R5'), cache('BB')
    k0 = int(s['cheat_n'])                                  # R1's scored interval starts here
    assert int(r5['k0']) == k0 == int(bb['k0']), 'the three sources must score the same samples'
    j, i = list(r5['records']).index('I1'), list(bb['records']).index('I1')
    y = bb['y'][i][:, 2]
    assert np.allclose(s['y_test_ref'][k0:, 2], y, atol=1e-12)
    final = s['y_hat_test_cl'][k0:, 2]
    base = r5['base_true'][j][:, 2]
    r = bb['r'][i][:, 2]
    t = np.arange(len(y)) / FS + k0 / FS
    # moves: samples where the reference moves (|dr| above 1e-9 m per sample), merged over gaps
    # shorter than 20 ms; HEURISTIC thresholds that only segment the record
    mv = np.abs(np.diff(r, prepend=r[0])) > 1e-9
    edges = np.flatnonzero(np.diff(mv.astype(int)))
    starts, stops = list(edges[::2] + 1), list(edges[1::2] + 1)
    segs = []
    for a, b in zip(starts, stops):
        if segs and a - segs[-1][1] < int(0.02 * FS):
            segs[-1] = (segs[-1][0], b)
        else:
            segs.append((a, b))
    eb = base - y
    a, b = max(segs, key=lambda ab: np.sqrt(np.mean(eb[ab[0]:min(ab[1] + int(0.05 * FS), len(y))] ** 2)))
    b = min(b + int(0.05 * FS), len(y))
    rows = [['%.6f' % t[k], f(r[k]), f(y[k]), f(base[k]), f(final[k]), int(a <= k < b)] for k in range(len(y))]
    write('res_time_trace', ['t_s', 'r_Y', 'y_Y', 'baseline_Y', 'final_Y', 'zoom'], rows)


# ------------------------------------------------------------------ every model on every split (D-269)
def errors():
    """Long table: per split, model, record and axis the RMS [m] and NRMS_e, with the multisine flag,
    from every preds cache that exists (eval_greybox.py --split=..., eval_blackbox.py)."""
    rows = []
    for split in SPLITS:
        sfx = '' if split == 'test' else '_%s' % split
        srcs = [(run, cache('%s%s' % (run, sfx))) for run in RUNS
                if (CACHE / ('preds_%s%s.npz' % (run, sfx))).exists()]
        if (CACHE / ('preds_BB%s.npz' % sfx)).exists():
            srcs.append(('R8', cache('BB%s' % sfx)))
        for run, d in srcs:
            for key in [k for k in ('model', 'physics', 'base_start', 'base_true', 'clamped', 'base_lti', 'bb', 'y_nf') if k in d]:
                for n, p, y, sg in zip(d['records'], d[key], d['y'], d['sigma_e']):
                    a = np.sqrt(np.mean((p - y) ** 2, axis=0))
                    rows.append([split, run, 'floor' if key == 'y_nf' else key, n, int(multisine(n))]
                                + [f(v) for v in a] + [f(v) for v in a / sg])
    write('res_errors', ['split', 'run', 'model', 'record', 'multisine'] + ['rms_%s' % a for a in AXES]
          + ['nrms_e_%s' % a for a in AXES], rows)


STEMS = {'res_errors': errors, 'res_lpv_lti': lpv_lti, 'res_structure': structure, 'res_ps_spectrum': spectrum,
         'res_obc_combinations': combinations, 'res_generalisation': generalisation,
         'res_time_trace': time_trace}

if __name__ == '__main__':
    for st in (sys.argv[1:] or STEMS):
        try:
            STEMS[st]()
        except (FileNotFoundError, KeyError) as e:
            print('skipped %s: %r' % (st, e))
