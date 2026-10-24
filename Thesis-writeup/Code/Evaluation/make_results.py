"""Saved evaluation results -> one CSV per thesis figure or table (Results/<stem>.csv).

Sources, so no figure ever needs a rollout or training again:
  - Results/thesis-eval/<campaign>/preds_<run>[_<split>].npz (eval_greybox.py: predictions from k0
    of the model, physics part, baseline start and true, clamped; the checkpoint label, the start
    and learned combinations, the config) and preds_BB[_<split>].npz (eval_blackbox.py: black box,
    y, r, sigma_e and the noise-free twin y_nf from k0). The run reports (gantry_results_<id>.npz)
    are not read: a run that has not finished has none, and its numbers carry the label
    "preliminary, update <n> of <N>" (eval_common.checkpoint_status) in a `status` column.
  - Results/res_obc_overlap.csv, res_obc_bias.csv, res_negation.csv (eval_overlap.py,
    eval_negation.py), read by the figure scripts directly.
Every error is RMS(y_hat - y) from the model's k0 on the noisy record, in metres, and beside it
NRMS_e = RMS / sigma_e, sigma_e the servo-error std over the same samples (D-256, D-262). The floor
is the noisy record minus its noise-free twin. E5 is not scored (user 2026-10-11). `res_errors`
lists every model on every split (train, val, test, twins; D-269) from the preds caches.

Usage: python Thesis-writeup/Code/Evaluation/make_results.py [stem ...]   (default: all)
"""
import csv
import functools
import json
import sys

import numpy as np
from scipy.signal import welch

from eval_common import TW, CACHE, RUNS, AXES, FS, DATA_NOISY, SPLITS, multisine, check_identity

RESULTS = TW / 'Results'
INTERP = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6']
EXTRAP = ['E1', 'E2', 'E3', 'E4', 'E6']    # D-283, the OBC accuracy rows only; E5 is not used (D-227)


@functools.lru_cache(maxsize=12)          # each file is read and decompressed once, not once per table row
def cache(name):
    d = dict(np.load(CACHE / ('preds_%s.npz' % name), allow_pickle=True))
    check_identity(name, d)                                 # never mix campaigns (audit 2026-10-16)
    return d


def sfx(split):
    return '' if split == 'test' else '_%s' % split


def gb(run, split='test'):
    """The grey-box prediction cache of `run` on `split` (eval_greybox.py)."""
    return cache(run + sfx(split))


def status(run):
    """'final', or 'preliminary, update <n> of <N>' for a run evaluated before it finished."""
    d = gb(run, 'val') if (CACHE / ('preds_%s_val.npz' % run)).exists() else gb(run)
    return str(d['status']) if 'status' in d else 'final'


def rms_rows(d, key, records):
    """(len(records), 3) per-record per-axis RMS(prediction - y) [m] of cache `d`, row `key`."""
    ix = {n: i for i, n in enumerate(d['records'])}
    return np.stack([np.sqrt(np.mean((d[key][ix[n]] - d['y'][ix[n]]) ** 2, axis=0)) for n in records])


def run_rms(run, row, records, split='test'):
    """(len(records), 3) per-record per-axis RMS of a grey-box row (model, physics, base_start,
    base_true, clamped) from its prediction cache; on val and test the cache reproduces the run
    report to 1e-6 (eval_greybox.py check), on train and the twins it is the only source."""
    return rms_rows(gb(run, split), row, records)


def bb_rms(records, key='bb', split='test'):
    return rms_rows(cache('BB' + sfx(split)), key, records)


def floor_rms(records, split='test'):
    return bb_rms(records, key='y_nf', split=split)


def sigma_e(records, split='test'):
    """(len(records), 3) servo-error std per record and axis over the scored samples (D-262)."""
    bb = cache('BB' + sfx(split))
    ix = {n: i for i, n in enumerate(bb['records'])}
    return np.stack([bb['sigma_e'][ix[n]] for n in records])


def split_records(split):
    """The scored records of a split (the black-box cache lists them; E5 is not scored)."""
    return [str(n) for n in cache('BB' + sfx(split))['records']]


def groups(split):
    """Record groups of a split, reported separately (D-256): with and without the training
    multisine. On test only I1 to I6 (the E records are the generalisation result)."""
    recs = split_records(split)
    if split == 'test':
        recs = [n for n in recs if n in INTERP]
    out = [('multisine', [n for n in recs if multisine(n)]), ('no multisine', [n for n in recs if not multisine(n)])]
    return [(g, r) for g, r in out if r]


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
        flo, sg = floor_rms([n])[0], sigma_e([n])[0]
        for model, key in (('LPV', 'base_true'), ('LTI', 'base_lti')):
            a = np.sqrt(np.mean((d[key][jx[n]] - y) ** 2, axis=0))
            rows.append([n, yop[n], model] + [f(v) for v in a] + [f(v) for v in flo] + [f(v) for v in a / sg]
                        + [f(v) for v in flo / sg])
    write('res_lpv_lti', ['record', 'Y_op', 'model'] + ['rms_%s' % a for a in AXES]
          + ['floor_%s' % a for a in AXES] + ['nrms_e_%s' % a for a in AXES]
          + ['floor_nrms_e_%s' % a for a in AXES], rows)


# ------------------------------------------------------------------ Table I: augmentation structure
STRUCTURE = [('Baseline, true parameters', 'R5', 'base_true'),
             ('Baseline, detuned start', 'R1', 'base_start'),
             ('Static parallel, $n_a=0$', 'R6', 'model'),
             # the detuned n_a = 0 pair (THESIS-RESULTS presentation 2026-10-16; final campaign only)
             ('Static parallel, $n_a=0$, detuned start', 'R14', 'model'),
             ('Static parallel (OBC), $n_a=0$, detuned start', 'R13', 'model'),
             ('Plain start, $n_a=2$', 'R3', 'model'),
             ('PS start, $n_a=2$', 'R5', 'model'),
             ('PS start, $n_a=2$, physics part only', 'R5', 'physics'),
             ('PS start, $n_a=8$', 'R7', 'model'),
             ('PS start, $n_a=2$, detuned start', 'R4', 'model'),
             ('Final model (OBC), detuned start', 'R1', 'model')]


def structure():
    """Per split and record group (with / without multisine, D-256; on test I1 to I6 only) the mean
    over the group's records of the per-record RMS [m] and NRMS_e per axis, every model. Appendix
    design choices are decided on VALIDATION (D-269); train, test and twins are reported beside it."""
    rows = []
    for split in SPLITS:
        for grp, recs in groups(split):
            sg = sigma_e(recs, split)
            both = lambda a: [f(v) for v in a.mean(0)] + [f(v) for v in (a / sg).mean(0)]   # noqa: E731
            head = [split, grp, ' '.join(recs)]
            for lbl, run, row in STRUCTURE:
                if run not in RUNS or not (CACHE / ('preds_%s%s.npz' % (run, sfx(split)))).exists():
                    print('  res_structure: no %s cache on %s, row "%s" left out' % (run, split, lbl))
                    continue
                rows.append(head + [lbl, run, RUNS[run], row, status(run)] + both(run_rms(run, row, recs, split)))
            rows.append(head + ['Black box', 'R8', '', 'bb', 'final'] + both(bb_rms(recs, split=split)))
            rows.append(head + ['Noise floor', '', '', 'floor', ''] + both(floor_rms(recs, split)))
    write('res_structure', ['split', 'group', 'records', 'label', 'run', 'job', 'row', 'status']
          + ['rms_%s' % a for a in AXES] + ['nrms_e_%s' % a for a in AXES], rows)


# ------------------------------------------------------------------ Fig. 2: error spectrum
def spectrum():
    """ASD of the Y error on the multisine records of every split (Welch, averaged), baseline (true
    parameters), plain (R3), PS (R5), floor. Validation decides the PS vs plain choice (D-269); on
    test the I records I1, I2."""
    rows = []
    for split in SPLITS:
        try:
            r3, r5, bb = gb('R3', split), gb('R5', split), cache('BB' + sfx(split))
        except FileNotFoundError as e:
            print('  res_ps_spectrum: %s skipped (%r)' % (split, e))
            continue
        recs = [n for n in r5['records'] if multisine(str(n)) and (split != 'test' or n in INTERP)]
        ix = lambda d, n: list(d['records']).index(n)                     # noqa: E731
        cols = {}
        for tag, src, key in (('baseline', r5, 'base_true'), ('plain', r3, 'model'), ('ps', r5, 'model')):
            p = [welch(src[key][ix(src, n), :, 2] - src['y'][ix(src, n), :, 2], fs=FS, nperseg=4096)
                 for n in recs]
            cols[tag] = np.sqrt(np.mean([q[1] for q in p], axis=0))
        p = [welch(bb['y_nf'][ix(bb, n), :, 2] - bb['y'][ix(bb, n), :, 2], fs=FS, nperseg=4096) for n in recs]
        fr = p[0][0]
        cols['floor'] = np.sqrt(np.mean([q[1] for q in p], axis=0))
        rows += [[split, ' '.join(str(n) for n in recs), '%.4f' % fr[k]] + [f(cols[c][k]) for c in cols]
                 for k in range(len(fr))]
    write('res_ps_spectrum', ['split', 'records', 'f_Hz'] + ['asd_%s' % k for k in
                                                           ('baseline', 'plain', 'ps', 'floor')], rows)


# ------------------------------------------------------------------ Table II: combinations
OBC_RUNS = ('R4', 'R1', 'R5', 'R2', 'R14', 'R13')


def combinations():
    """Ten combinations (true, start, learned) of R4, R1, R5, R2 and, in the final campaign, the
    detuned n_a = 0 pair R14 (U), R13 (OBC): parameter recovery without added states (appendix,
    Jasper point 5). From the prediction caches (eval_greybox.py stores the report's params_init,
    the START vector, and params_learned), so a running run is included with its label. The true
    values come from a correct-start run (both must agree)."""
    runs = [r for r in OBC_RUNS if r in RUNS]
    npz = {r: gb(r) for r in runs}
    true = npz['R5']['params_init']
    assert np.allclose(true, npz['R2']['params_init'], rtol=1e-12)
    rows = []
    for run in runs:
        s = npz[run]
        cfg = json.loads(str(s['config']))
        det = cfg.get('COMBO_INIT_DETUNE') or [1.0] * 10
        start, learned = s['params_init'], s['params_learned']
        assert np.allclose(start, true * np.asarray(det), rtol=1e-6), run
        for k, n in enumerate(str(x) for x in s['param_names']):
            rows.append([run, RUNS[run], 'obc' if cfg['OBC'] else 'u',
                         'detuned' if cfg.get('COMBO_INIT_DETUNE') else 'correct', status(run),
                         n, '%.8e' % true[k], '%.8e' % start[k], '%.8e' % learned[k]])
    write('res_obc_combinations', ['run', 'job', 'arm', 'start', 'status', 'combination', 'true', 'start_value',
                                   'learned'], rows)


def obc_accuracy():
    """Table C accuracy rows (D-268 stacks; 05_setup Table C): mean over the stack's records of the
    per-record RMS [m] and NRMS_e per axis, full model and physical part alone (Gyorok 2025
    Table 1), per run. Stacks as eval_overlap.py: training, validation, test I1 to I6 and extrap
    E1 to E4, E6 (D-283; both from the test predictions). Group 'all' (the main Table C) and the
    multisine split of D-256 (the appendix version, Dirk 2026-10-18)."""
    rows = []
    for stack, split, keep in (('train', 'train', None), ('val', 'val', None),
                               ('test', 'test', INTERP), ('extrap', 'test', EXTRAP)):
        recs = [n for n in split_records(split) if keep is None or n in keep]
        for grp, rs in [('all', recs)] + [(g, [n for n in recs if multisine(n) == (g == 'multisine')])
                                          for g in ('multisine', 'no multisine')]:
            if not rs:
                continue
            sg = sigma_e(rs, split)
            for run in [r for r in OBC_RUNS if r in RUNS]:
                for row in ('model', 'physics'):
                    a = run_rms(run, row, rs, split)
                    rows.append([run, RUNS[run], stack, grp, row, status(run), ' '.join(rs)]
                                + [f(v) for v in a.mean(0)] + [f(v) for v in (a / sg).mean(0)])
    write('res_obc_accuracy', ['run', 'job', 'stack', 'group', 'row', 'status', 'records'] + ['rms_%s' % a for a in AXES]
          + ['nrms_e_%s' % a for a in AXES], rows)


# ------------------------------------------------------------------ Fig. 3: generalisation
def generalisation():
    """Per split and record: baseline at the TRUE parameters (M_LPV, 05_setup) and at the DETUNED
    start (the final model before augmentation, 06_results); the two texts disagree, so both rows
    are written (handoff 2026-10-18 item 7.B.2). All from R1's cache (same encoder), final model R1,
    black box, floor. Test holds the I/E pairs; the twins are the multisine extension (D-258)."""
    rows = []
    for split in SPLITS:
        recs = split_records(split)
        sg = sigma_e(recs, split)
        for model, vals, st in (('baseline_true', run_rms('R1', 'base_true', recs, split), status('R1')),
                                ('baseline_detuned', run_rms('R1', 'base_start', recs, split), status('R1')),
                                ('final', run_rms('R1', 'model', recs, split), status('R1')),
                                ('bb', bb_rms(recs, split=split), 'final'),
                                ('floor', floor_rms(recs, split), '')):
            for n, a, s_ in zip(recs, vals, sg):
                rows.append([split, n, int(multisine(n)), model, st] + [f(v) for v in a] + [f(np.mean(a))]
                            + [f(v) for v in a / s_])
    write('res_generalisation', ['split', 'record', 'multisine', 'model', 'status'] + ['rms_%s' % a for a in AXES]
          + ['rms_mean'] + ['nrms_e_%s' % a for a in AXES], rows)


# ------------------------------------------------------------------ Fig. 4: time trace
def time_trace():
    """I1, Y axis: measured, LPV baseline (R5 cache, true parameters), final model (R1 cache),
    reference; zoom window chosen from the baseline and reference only:
    the move with the largest baseline error RMS, from its start to 50 ms after it stops."""
    r1, r5, bb = gb('R1'), gb('R5'), cache('BB')
    k0 = int(r1['k0'])                                      # R1's scored interval starts here
    assert int(r5['k0']) == k0 == int(bb['k0']), 'the three sources must score the same samples'
    j, i, m = (list(d['records']).index('I1') for d in (r5, bb, r1))
    y = bb['y'][i][:, 2]
    assert np.allclose(r1['y'][m][:, 2], y, atol=1e-12) and np.allclose(r5['y'][j][:, 2], y, atol=1e-12)
    final = r1['model'][m][:, 2]
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
    st = status('R1')                                       # once, not per sample
    rows = [['%.6f' % t[k], f(r[k]), f(y[k]), f(base[k]), f(final[k]), int(a <= k < b), st]
            for k in range(len(y))]
    write('res_time_trace', ['t_s', 'r_Y', 'y_Y', 'baseline_Y', 'final_Y', 'zoom', 'status_final'], rows)


# ------------------------------------------------------------------ every model on every split (D-269)
def errors():
    """Long table: per split, model, record and axis the RMS [m] and NRMS_e, with the multisine flag
    and the checkpoint label, from every preds cache that exists (eval_greybox.py --split=...,
    eval_blackbox.py)."""
    rows = []
    for split in SPLITS:
        srcs = [(run, gb(run, split)) for run in RUNS
                if (CACHE / ('preds_%s%s.npz' % (run, sfx(split)))).exists()]
        if (CACHE / ('preds_BB%s.npz' % sfx(split))).exists():
            srcs.append(('R8', cache('BB' + sfx(split))))
        for run, d in srcs:
            st = str(d['status']) if 'status' in d else 'final'
            for key in [k for k in ('model', 'physics', 'base_start', 'base_true', 'clamped', 'base_lti', 'bb', 'y_nf') if k in d]:
                for n, p, y, sg in zip(d['records'], d[key], d['y'], d['sigma_e']):
                    a = np.sqrt(np.mean((p - y) ** 2, axis=0))
                    rows.append([split, run, 'floor' if key == 'y_nf' else key, st, n, int(multisine(n))]
                                + [f(v) for v in a] + [f(v) for v in a / sg])
    write('res_errors', ['split', 'run', 'model', 'status', 'record', 'multisine'] + ['rms_%s' % a for a in AXES]
          + ['nrms_e_%s' % a for a in AXES], rows)


def checks():
    """Acceptance (handoff 2026-10-18 section 10): per grey-box run the evaluation's validation
    sim-NRMSe beside the score its checkpoint recorded (= the best validation of the log)."""
    rows = []
    for run in RUNS:
        if not (CACHE / ('preds_%s_val.npz' % run)).exists():
            continue
        d = gb(run, 'val')
        if 'val_score' not in d:
            continue
        mine, ref = float(d['val_score']), float(d['ref_score'])
        rows.append([run, RUNS[run], str(d['status']), '%.6f' % mine, '%.6f' % ref, '%.2e' % (abs(mine - ref) / ref),
                     int(abs(mine - ref) / ref < 1e-3)])
    write('res_checks', ['run', 'job', 'status', 'val_nrmse_eval', 'val_nrmse_checkpoint', 'rel_diff', 'ok'], rows)


STEMS = {'res_errors': errors, 'res_checks': checks, 'res_lpv_lti': lpv_lti, 'res_structure': structure,
         'res_ps_spectrum': spectrum, 'res_obc_combinations': combinations, 'res_obc_accuracy': obc_accuracy,
         'res_generalisation': generalisation, 'res_time_trace': time_trace}

if __name__ == '__main__':
    for st in (sys.argv[1:] or STEMS):
        try:
            STEMS[st]()
        except (FileNotFoundError, KeyError) as e:
            print('skipped %s: %r' % (st, e))
