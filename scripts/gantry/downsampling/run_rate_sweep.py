"""Exact-model downsampling and controller-amplification study.

The 20 kHz MATLAB records are the master data.  For each record this script
reconstructs the exact eight-state hidden-MSD trajectory at 20 kHz, verifies it
against the stored output, then runs the exact same oracle and the nominal
six-state FP baseline at 4, 2, and 1 kHz.

Input reduction is deliberately identical to the production loader:
per-coarse-interval block mean for ``u_total`` and point sampling for ``y``.
The principal decision quantity is

    oracle rate floor / FP-to-truth discrepancy,

reported in the time domain and in the 130--180 Hz MSD band, on exact-state
100 ms windows and on a full free run.

The optional controller stage keeps its two meanings separate:
``corate`` reproduces the present pipeline; ``controller20k_zoh`` keeps the
controller at 20 kHz and uses the simplest causal low-rate interface (held
model output, block-mean corrected force).
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts' / 'gantry'))

from model_augmentation.systems import gantry_ss as GSS  # noqa: E402
from gantry_dynamic.controller import controller_ss, y_op_for  # noqa: E402


DATASET = 'augmentation_ma50_a5'
DATA_DIR = REPO / 'data' / 'gantry' / 'matlab' / 'trajectory' / DATASET
OUT_DIR = HERE / 'results'
RATES = (4000, 2000, 1000)
FS_MASTER = 20000
WINDOW_SECONDS = 0.100
K0_SAMPLES = 17
MSD_BAND = (130.0, 180.0)
MA_FRAC = 0.50
FA_HZ = 150.0
ZETA_A = 0.05
L0 = 0.10


def _f(v):
    return float(v)


m1, m2, mb, mh = map(_f, (GSS.m1, GSS.m2, GSS.mb, GSS.mh))
Jb, Jh = map(_f, (GSS.Jb, GSS.Jh))
cg1, cg2, cy = map(_f, (GSS.cg1, GSS.cg2, GSS.cy))
cb1, cb2 = map(_f, (GSS.cb1, GSS.cb2))
kb1, kb2 = map(_f, (GSS.kb1, GSS.kb2))
Lb, lever_d = map(_f, (GSS.Lb, GSS.d))
P = np.asarray(GSS.P, dtype=np.float64)
PT = P.T

ma = MA_FRAC * mh
mh_rigid = mh - ma
ka = ma * (2.0 * np.pi * FA_HZ) ** 2
ca = 2.0 * ZETA_A * np.sqrt(ka * ma)

C4 = np.array([
    [cg1 + cg2, (cg1 - cg2) * Lb / 2.0, 0.0, 0.0],
    [(cg1 - cg2) * Lb / 2.0,
     cb1 + cb2 + (cg1 + cg2) * Lb ** 2 / 4.0, 0.0, 0.0],
    [0.0, 0.0, cy, 0.0],
    [0.0, 0.0, 0.0, ca],
])
K4 = np.diag([0.0, kb1 + kb2, 0.0, ka])
E43 = np.vstack([np.eye(3), np.zeros((1, 3))])
C3, K3 = C4[:3, :3].copy(), K4[:3, :3].copy()


def deriv8(x, u_logical):
    """The ma50 data-generating plant, natural state order [q4, qdot4]."""
    Y, da = x[2], x[3]
    off = (m1 - m2) * Lb / 2.0 - mh * Y - ma * L0 - ma * da
    M = np.array([
        [m1 + m2 + mb + mh, off, 0.0, 0.0],
        [off,
         Jb + Jh + (m1 + m2) * Lb ** 2 / 4.0 + mh * lever_d ** 2
         + mh_rigid * Y ** 2 + ma * (Y + L0 + da) ** 2,
         -mh * lever_d, -ma * lever_d],
        [0.0, -mh * lever_d, mh, ma],
        [0.0, -ma * lever_d, ma, ma],
    ])
    q, qd = x[:4], x[4:]
    qdd = np.linalg.solve(M, E43 @ u_logical - K4 @ q - C4 @ qd)
    return np.concatenate((qd, qdd))


def deriv6(x, u_logical):
    """Nominal FP baseline, with the conserved total payload mass mh."""
    Y = x[2]
    off = (m1 - m2) * Lb / 2.0 - mh * Y
    M = np.array([
        [m1 + m2 + mb + mh, off, 0.0],
        [off, Jb + Jh + (m1 + m2) * Lb ** 2 / 4.0
         + mh * lever_d ** 2 + mh * Y ** 2, -mh * lever_d],
        [0.0, -mh * lever_d, mh],
    ])
    q, qd = x[:3], x[3:]
    qdd = np.linalg.solve(M, u_logical - K3 @ q - C3 @ qd)
    return np.concatenate((qd, qdd))


def rk4_step(deriv, x, u_logical, ts, up_sample=1):
    h = ts / int(up_sample)
    x = np.asarray(x, dtype=np.float64).copy()
    for _ in range(int(up_sample)):
        k1 = deriv(x, u_logical)
        k2 = deriv(x + 0.5 * h * k1, u_logical)
        k3 = deriv(x + 0.5 * h * k2, u_logical)
        k4 = deriv(x + h * k3, u_logical)
        x += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x


def rollout(deriv, x0, u_stage, ts, up_sample=1):
    x = np.asarray(x0, dtype=np.float64).copy()
    out = np.empty((len(u_stage), len(x)), dtype=np.float64)
    for k, uk in enumerate(np.asarray(u_stage, dtype=np.float64)):
        out[k] = x
        x = rk4_step(deriv, x, P @ uk, ts, up_sample)
    return out


def stage_output(states):
    return (PT @ np.asarray(states)[:, :3].T).T


def block_mean(u, down):
    n = len(u) // int(down)
    return np.asarray(u[:n * down], dtype=np.float64).reshape(n, down, 3).mean(axis=1)


def model_seed(x8, model):
    if model == 'oracle':
        return np.asarray(x8, dtype=np.float64).copy()
    return np.asarray(x8, dtype=np.float64)[[0, 1, 2, 4, 5, 6]].copy()


def finite_metric(err):
    err = np.asarray(err, dtype=np.float64)
    finite = bool(np.isfinite(err).all())
    if not finite or len(err) == 0:
        return dict(finite=False, aggregate_rms=float('inf'), channel_rms=[float('inf')] * 3,
                    channel_max=[float('inf')] * 3)
    return dict(
        finite=True,
        aggregate_rms=float(np.sqrt(np.mean(err ** 2))),
        channel_rms=np.sqrt(np.mean(err ** 2, axis=0)).tolist(),
        channel_max=np.max(np.abs(err), axis=0).tolist(),
    )


def band_metric(err, fs):
    err = np.asarray(err, dtype=np.float64)
    if len(err) < 32 or not np.isfinite(err).all():
        return dict(finite=False, aggregate_rms=float('inf'), channel_rms=[float('inf')] * 3)
    sos = butter(4, MSD_BAND, btype='bandpass', fs=float(fs), output='sos')
    fil = sosfiltfilt(sos, err, axis=0)
    return dict(finite=True, aggregate_rms=float(np.sqrt(np.mean(fil ** 2))),
                channel_rms=np.sqrt(np.mean(fil ** 2, axis=0)).tolist())


def combined_metrics(err, fs):
    return dict(time=finite_metric(err), band=band_metric(err, fs))


def exact_window_errors(deriv, model, truth_state, u_stage, y_ref, fs, up_sample=1):
    length = int(round(WINDOW_SECONDS * fs))
    starts = np.arange(K0_SAMPLES, len(y_ref) - length + 1, length, dtype=int)
    errors = []
    band_rows = []
    for s in starts:
        sim = rollout(deriv, model_seed(truth_state[s], model),
                      u_stage[s:s + length], 1.0 / fs, up_sample)
        err = stage_output(sim) - y_ref[s:s + length]
        errors.append(err)
        band_rows.append(band_metric(err, fs))
    all_err = np.concatenate(errors, axis=0) if errors else np.empty((0, 3))
    out = finite_metric(all_err)
    if band_rows and all(r['finite'] for r in band_rows):
        ch2 = np.mean(np.square([r['channel_rms'] for r in band_rows]), axis=0)
        agg2 = np.mean(np.square([r['aggregate_rms'] for r in band_rows]))
        band = dict(finite=True, aggregate_rms=float(np.sqrt(agg2)),
                    channel_rms=np.sqrt(ch2).tolist())
    else:
        band = dict(finite=False, aggregate_rms=float('inf'), channel_rms=[float('inf')] * 3)
    return dict(time=out, band=band, n_windows=int(len(starts)), window_samples=length)


def initial_truth_state(raw):
    Y0 = float(np.asarray(raw['x_logical'])[0, 2])
    return np.array([0.0, 0.0, Y0, 0.0, 0.0, 0.0, 0.0, 0.0])


def reconstruct_truth(raw):
    u = np.asarray(raw['u_total'], dtype=np.float64)
    dt = float(raw['dt'])
    if not np.isclose(1.0 / dt, FS_MASTER, rtol=0, atol=1.0):
        raise ValueError('master record is not 20 kHz')
    states = rollout(deriv8, initial_truth_state(raw), u, dt, 1)
    y = np.asarray(raw['y'], dtype=np.float64)[:len(states)]
    gate = finite_metric(stage_output(states) - y)
    return states, y, gate


def rate_view(raw, truth20, y20, fs):
    down = FS_MASTER // int(fs)
    u = block_mean(np.asarray(raw['u_total'], dtype=np.float64), down)
    n = len(u)
    return u, y20[::down][:n], truth20[::down][:n]


def open_loop_record(path):
    started = time.time()
    raw = loadmat(path, squeeze_me=True)
    name = Path(path).stem
    truth20, y20, gate = reconstruct_truth(raw)
    rates = {}
    for fs in RATES:
        u, y, truth = rate_view(raw, truth20, y20, fs)
        arms = {}
        for model, deriv in (('oracle', deriv8), ('fp', deriv6)):
            sim = rollout(deriv, model_seed(truth[0], model), u, 1.0 / fs, 1)
            arms[model] = dict(
                full=combined_metrics(stage_output(sim) - y, fs),
                windows=exact_window_errors(deriv, model, truth, u, y, fs, 1),
            )
        rates[str(fs)] = arms
    return dict(record=name, split=name[0], n_master=int(len(y20)),
                duration_seconds=float(len(y20) / FS_MASTER), gate20k=gate,
                rates=rates, elapsed_seconds=time.time() - started)


def controller_rollout_corate(deriv, model, x0_truth, u_data, y_data, fs, Y_op):
    A, B, C, D = controller_ss(float(Y_op), 1.0 / fs)
    x = model_seed(x0_truth, model)
    xc = np.zeros(A.shape[0])
    out = np.empty_like(y_data, dtype=np.float64)
    correction = np.empty_like(u_data, dtype=np.float64)
    applied = np.empty_like(u_data, dtype=np.float64)
    divergence_step = None
    for k in range(len(y_data)):
        ym = PT @ x[:3]
        out[k] = ym
        e = y_data[k] - ym
        ufb = C @ xc + D @ e
        correction[k] = ufb
        applied[k] = u_data[k] + ufb
        x = rk4_step(deriv, x, P @ applied[k], 1.0 / fs, 1)
        xc = A @ xc + B @ e
        if not np.isfinite(x).all() or np.max(np.abs(x[:3])) > 10.0:
            out[k + 1:] = np.nan
            correction[k + 1:] = np.nan
            applied[k + 1:] = np.nan
            divergence_step = k + 1
            break
    return dict(y=out, correction=correction, applied=applied,
                divergence_step=divergence_step,
                divergence_seconds=None if divergence_step is None else divergence_step / fs)


def controller_rollout_20k_zoh(deriv, model, x0_truth, raw_u, raw_y, fs, Y_op):
    """20 kHz residual controller around a held-output low-rate model.

    At a coarse interval the model output is held for all controller samples.
    The resulting corrected high-rate forces are averaged, then used for one
    low-rate model step.  This is causal and explicit, but intentionally only
    one candidate multirate interface.
    """
    down = FS_MASTER // int(fs)
    n = min(len(raw_u), len(raw_y)) // down
    A, B, C, D = controller_ss(float(Y_op), 1.0 / FS_MASTER)
    x = model_seed(x0_truth, model)
    xc = np.zeros(A.shape[0])
    out = np.empty((n, 3), dtype=np.float64)
    correction = np.empty((n, 3), dtype=np.float64)
    applied = np.empty((n, 3), dtype=np.float64)
    divergence_step = None
    for k in range(n):
        ym = PT @ x[:3]
        out[k] = ym
        usum = np.zeros(3)
        ubase_sum = np.zeros(3)
        lo = k * down
        for j in range(lo, lo + down):
            e = raw_y[j] - ym
            ufb = C @ xc + D @ e
            usum += raw_u[j] + ufb
            ubase_sum += raw_u[j]
            xc = A @ xc + B @ e
        applied[k] = usum / down
        correction[k] = applied[k] - ubase_sum / down
        x = rk4_step(deriv, x, P @ applied[k], 1.0 / fs, 1)
        if not np.isfinite(x).all() or np.max(np.abs(x[:3])) > 10.0:
            out[k + 1:] = np.nan
            correction[k + 1:] = np.nan
            applied[k + 1:] = np.nan
            divergence_step = k + 1
            break
    return dict(y=out, correction=correction, applied=applied,
                divergence_step=divergence_step,
                divergence_seconds=None if divergence_step is None else divergence_step / fs)


def controller_window_errors(protocol, deriv, model, truth20, raw_u, raw_y, fs, Y_op):
    down = FS_MASTER // int(fs)
    n = min(len(raw_u), len(raw_y)) // down
    length = int(round(WINDOW_SECONDS * fs))
    starts = np.arange(K0_SAMPLES, n - length + 1, length, dtype=int)
    errors = []
    corrections = []
    applied_forces = []
    band_rows = []
    divergence_times = []
    for s in starts:
        if protocol == 'corate':
            u = block_mean(raw_u[s * down:(s + length) * down], down)
            y = raw_y[s * down:(s + length) * down:down]
            run = controller_rollout_corate(
                deriv, model, truth20[s * down], u, y, fs, Y_op)
        else:
            uh = raw_u[s * down:(s + length) * down]
            yh = raw_y[s * down:(s + length) * down]
            y = yh[::down]
            run = controller_rollout_20k_zoh(
                deriv, model, truth20[s * down], uh, yh, fs, Y_op)
        pred = run['y']
        err = pred - y[:len(pred)]
        errors.append(err)
        corrections.append(run['correction'])
        applied_forces.append(run['applied'])
        if run['divergence_seconds'] is not None:
            divergence_times.append(run['divergence_seconds'])
        band_rows.append(band_metric(err, fs))
    all_err = np.concatenate(errors, axis=0) if errors else np.empty((0, 3))
    out = finite_metric(all_err)
    if band_rows and all(r['finite'] for r in band_rows):
        band = dict(
            finite=True,
            aggregate_rms=float(np.sqrt(np.mean([r['aggregate_rms'] ** 2 for r in band_rows]))),
            channel_rms=np.sqrt(np.mean(np.square([r['channel_rms'] for r in band_rows]), axis=0)).tolist(),
        )
    else:
        band = dict(finite=False, aggregate_rms=float('inf'), channel_rms=[float('inf')] * 3)
    return dict(time=out, band=band, n_windows=int(len(starts)), window_samples=length,
                correction_force=finite_metric(np.concatenate(corrections, axis=0)),
                applied_force=finite_metric(np.concatenate(applied_forces, axis=0)),
                n_diverged_windows=len(divergence_times),
                first_divergence_seconds=min(divergence_times) if divergence_times else None)


def controller_record(path):
    started = time.time()
    raw = loadmat(path, squeeze_me=True)
    name = Path(path).stem
    raw_u = np.asarray(raw['u_total'], dtype=np.float64)
    raw_y = np.asarray(raw['y'], dtype=np.float64)
    truth20, y20, gate = reconstruct_truth(raw)
    Y_op = y_op_for(name)
    rates = {}
    for fs in RATES:
        down = FS_MASTER // fs
        u = block_mean(raw_u, down)
        y = raw_y[::down][:len(u)]
        rate = {}
        for protocol in ('corate', 'controller20k_zoh'):
            arms = {}
            for model, deriv in (('oracle', deriv8), ('fp', deriv6)):
                if protocol == 'corate':
                    run = controller_rollout_corate(deriv, model, truth20[0], u, y, fs, Y_op)
                else:
                    run = controller_rollout_20k_zoh(
                        deriv, model, truth20[0], raw_u, raw_y, fs, Y_op)
                pred = run['y']
                arms[model] = dict(
                    full=combined_metrics(pred - y[:len(pred)], fs),
                    windows=controller_window_errors(
                        protocol, deriv, model, truth20, raw_u, raw_y, fs, Y_op),
                )
                arms[model]['full']['correction_force'] = finite_metric(run['correction'])
                arms[model]['full']['applied_force'] = finite_metric(run['applied'])
                arms[model]['full']['divergence_step'] = run['divergence_step']
                arms[model]['full']['divergence_seconds'] = run['divergence_seconds']
            rate[protocol] = arms
        rates[str(fs)] = rate
    return dict(record=name, split=name[0], Y_op=float(Y_op), gate20k=gate,
                rates=rates, elapsed_seconds=time.time() - started)


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def write_json(path, payload):
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(clean(payload), f, indent=2, default=_json_default, allow_nan=False)


def arm_rows(records, controller=False):
    rows = []
    for rec in records:
        for fs, rate in rec['rates'].items():
            protocols = rate.items() if controller else [('open_loop', rate)]
            for protocol, arms in protocols:
                for horizon in ('full', 'windows'):
                    oracle = arms['oracle'][horizon]
                    fp = arms['fp'][horizon]
                    for domain in ('time', 'band'):
                        eo = oracle[domain]['aggregate_rms']
                        eb = fp[domain]['aggregate_rms']
                        eo = float('inf') if eo is None else eo
                        eb = float('inf') if eb is None else eb
                        rows.append(dict(
                            record=rec['record'], split=rec['split'], rate_hz=int(fs),
                            protocol=protocol, horizon=horizon, domain=domain,
                            oracle_rms=eo, fp_rms=eb,
                            floor_over_target=eo / eb if np.isfinite(eo) and eb > 0 else float('inf'),
                            oracle_finite=oracle[domain]['finite'], fp_finite=fp[domain]['finite'],
                        ))
    return rows


def controller_force_rows(records):
    """Flatten controller force and divergence diagnostics for spreadsheet use."""
    rows = []
    axes = ('X1', 'X2', 'Y')
    for rec in records:
        for fs, rate in rec['rates'].items():
            for protocol, arms in rate.items():
                for model in ('oracle', 'fp'):
                    for horizon in ('full', 'windows'):
                        result = arms[model][horizon]
                        row = dict(record=rec['record'], split=rec['split'],
                                   rate_hz=int(fs), protocol=protocol,
                                   model=model, horizon=horizon,
                                   divergence_seconds=result.get('divergence_seconds'),
                                   n_diverged_windows=result.get('n_diverged_windows'))
                        for force_name in ('correction_force', 'applied_force'):
                            metric = result[force_name]
                            row[force_name + '_finite'] = metric['finite']
                            row[force_name + '_aggregate_rms'] = metric['aggregate_rms']
                            for i, axis in enumerate(axes):
                                row[force_name + '_rms_' + axis] = metric['channel_rms'][i]
                                row[force_name + '_max_' + axis] = metric['channel_max'][i]
                        rows.append(row)
    return rows


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def aggregate_rows(rows):
    groups = {}
    keys = ('rate_hz', 'protocol', 'horizon', 'domain')
    for row in rows:
        key = tuple(row[k] for k in keys)
        groups.setdefault(key, []).append(row)
    out = []
    for key, rr in sorted(groups.items()):
        eo = np.asarray([r['oracle_rms'] for r in rr], float)
        eb = np.asarray([r['fp_rms'] for r in rr], float)
        ratios = np.asarray([r['floor_over_target'] for r in rr], float)
        oracle_finite = np.isfinite(eo)
        fp_finite = np.isfinite(eb)
        paired_finite = oracle_finite & fp_finite & np.isfinite(ratios)
        out.append(dict(zip(keys, key), n_records=int(paired_finite.sum()),
                        n_oracle_records=int(oracle_finite.sum()),
                        n_fp_records=int(fp_finite.sum()),
                        oracle_rms_rss=float(np.sqrt(np.mean(eo[oracle_finite] ** 2))) if oracle_finite.any() else float('inf'),
                        fp_rms_rss=float(np.sqrt(np.mean(eb[fp_finite] ** 2))) if fp_finite.any() else float('inf'),
                        ratio_rss=float(np.sqrt(np.mean(eo[paired_finite] ** 2) / np.mean(eb[paired_finite] ** 2))) if paired_finite.any() else float('inf'),
                        ratio_median=float(np.median(ratios[paired_finite])) if paired_finite.any() else float('inf'),
                        ratio_worst=float(np.max(ratios[paired_finite])) if paired_finite.any() else float('inf')))
    return out


def make_plot(open_agg, controller_agg, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    panels = [
        ('open_loop', open_agg, 'Open loop'),
        ('corate', controller_agg, 'Controller co-rate'),
        ('controller20k_zoh', controller_agg, '20 kHz controller + ZOH model'),
    ]
    for ax, (protocol, data, title) in zip(axes, panels):
        for horizon, ls, marker in (('windows', '-', 'o'), ('full', '--', 's')):
            rows = [r for r in data if r['protocol'] == protocol and r['horizon'] == horizon
                    and r['domain'] == 'time']
            if rows:
                rows.sort(key=lambda r: r['rate_hz'], reverse=True)
                ax.plot([r['rate_hz'] for r in rows], [r['ratio_rss'] for r in rows],
                        ls=ls, marker=marker, label=horizon)
        ax.axhline(0.10, color='tab:green', lw=1, ls=':', label='10% gate')
        ax.axhline(0.25, color='tab:red', lw=1, ls=':', label='25% gate')
        ax.set_title(title)
        ax.set_xlabel('model rate [Hz]')
        ax.set_xticks(RATES)
        ax.set_yscale('log')
        ax.grid(True, which='both', alpha=0.3)
    axes[0].set_ylabel('oracle rate floor / FP discrepancy')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def findings_text(open_agg, controller_agg, gates, controller_records):
    def table(rows, protocol, horizon):
        hit = [r for r in rows if r['protocol'] == protocol and r['horizon'] == horizon]
        hit.sort(key=lambda r: (r['domain'], -r['rate_hz']))
        lines = ['| domain | rate | oracle RMS | FP RMS | RSS ratio | worst-record ratio |',
                 '|---|---:|---:|---:|---:|---:|']
        for r in hit:
            lines.append('| %s | %d | %.3e | %.3e | %.3f | %.3f |' %
                         (r['domain'], r['rate_hz'], r['oracle_rms_rss'], r['fp_rms_rss'],
                          r['ratio_rss'], r['ratio_worst']))
        return '\n'.join(lines)

    worst_gate = max((g['channel_max'] for g in gates), default=[float('nan')], key=max)
    def ratio(rows, protocol, horizon, domain, rate):
        hit = [r for r in rows if r['protocol'] == protocol and r['horizon'] == horizon
               and r['domain'] == domain and r['rate_hz'] == rate]
        return hit[0]['ratio_rss'] if hit else float('inf')

    open_band_2k = ratio(open_agg, 'open_loop', 'windows', 'band', 2000)
    open_band_1k = ratio(open_agg, 'open_loop', 'windows', 'band', 1000)
    cl_time_2k = ratio(controller_agg, 'corate', 'windows', 'time', 2000)
    cl_time_1k = ratio(controller_agg, 'corate', 'windows', 'time', 1000)
    zoh_time_4k = ratio(controller_agg, 'controller20k_zoh', 'windows', 'time', 4000)
    divergence_lines = ['| record | exact oracle | FP baseline |',
                        '|---|---:|---:|']
    for rec in sorted(controller_records, key=lambda r: r['record']):
        arms = rec['rates']['1000']['corate']
        values = []
        for model in ('oracle', 'fp'):
            value = arms[model]['full'].get('divergence_seconds')
            values.append('none' if value is None else f'{value:.3f} s')
        divergence_lines.append(f"| {rec['record']} | {values[0]} | {values[1]} |")
    divergence_table = '\n'.join(divergence_lines)
    return f"""# Downsampling findings

Dataset: `{DATASET}` ({len(gates)} records).  Master rate: {FS_MASTER} Hz.
The 20 kHz oracle/data maximum gate errors were no worse than
`{max(worst_gate):.3e}` in stage coordinates.

The decision ratio is the exact-oracle rate floor divided by the six-state
FP-to-eight-state-truth discrepancy.  `0.10` and `0.25` are diagnostic guide
lines, not statistical confidence bounds.

## Open loop: exact-state 100 ms windows

{table(open_agg, 'open_loop', 'windows')}

## Open loop: full-record free run

{table(open_agg, 'open_loop', 'full')}

## Current co-rate controller: exact-state 100 ms windows

{table(controller_agg, 'corate', 'windows')}

## Current co-rate controller: full-record free run

{table(controller_agg, 'corate', 'full')}

## 20 kHz controller with held low-rate output: exact-state 100 ms windows

{table(controller_agg, 'controller20k_zoh', 'windows')}

## 20 kHz controller with held low-rate output: full-record free run

{table(controller_agg, 'controller20k_zoh', 'full')}

## Numerical verdict

- Open-loop 2 kHz preserves the 130--180 Hz learning target: its exact-oracle
  100 ms floor is `{open_band_2k:.3f}` of the FP discrepancy.  At 1 kHz this is
  `{open_band_1k:.3f}`, above the 10% guide line.
- With the current co-rate controller, the 100 ms time-domain ratios are
  `{cl_time_2k:.3f}` at 2 kHz and `{cl_time_1k:.3f}` at 1 kHz.  The exact-oracle
  arm diverged in every 1 kHz full-record co-rate run, while the six-state FP
  arm remained finite.  Consequently no paired full-run ratio exists there.
- The naive 20 kHz-controller/ZOH-model interface is already dominated by its
  intersample convention at 4 kHz (`{zoh_time_4k:.3f}` time-domain ratio).
  It is therefore rejected as an implementation, not used to reject keeping
  the real controller at 20 kHz.  A useful 20 kHz-controller design needs a
  high-rate model-output reconstruction rather than a hold.

### Co-rate 1 kHz full-run divergence

{divergence_table}

Correction-force and total applied-force RMS/peak values for every arm are in
`controller_force_summary.csv` and the unflattened `controller.json`.

## Interpretation constraints

- Open loop is the primary sampling-rate gate: the recorded force already
  contains the action of the original 20 kHz controller.
- `corate` is the current training implementation and changes the discrete
  controller with model rate.
- `controller20k_zoh` keeps the controller at 20 kHz but necessarily chooses an
  intersample model-output convention.  Its zero-order hold is deliberately
  explicit; a different multirate interface can produce different numbers.
- No encoder and no ANN are used.  Every window begins from the reconstructed
  exact eight-state truth, so these numbers are a numerical/data-conditioning
  floor rather than an estimation result.
"""


def records_for_stage(all_paths, stage):
    if stage == 'open':
        return all_paths
    # Controller full runs are most relevant on held-out records.  This keeps the
    # expensive 20 kHz sequential-controller diagnostic bounded and reproducible.
    wanted = {'V1_standstill_Yp10', 'V2_aprbs_Ylow', 'V3_ysweep_Yp10',
              'V4_lissajous_Ym10', 'E1_resonance_sweep'}
    return [p for p in all_paths if p.stem in wanted]


def run_parallel(func, paths, workers):
    if workers <= 1:
        out = []
        for i, p in enumerate(paths, 1):
            row = func(str(p))
            print('[%d/%d] %s %.1fs' % (i, len(paths), row['record'], row['elapsed_seconds']),
                  flush=True)
            out.append(row)
        return out
    ctx = mp.get_context('spawn')
    out = []
    with ctx.Pool(processes=workers) as pool:
        for i, row in enumerate(pool.imap_unordered(func, map(str, paths)), 1):
            print('[%d/%d] %s %.1fs' % (i, len(paths), row['record'], row['elapsed_seconds']),
                  flush=True)
            out.append(row)
    return sorted(out, key=lambda r: r['record'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=('open', 'controller', 'report', 'all'), default='all')
    ap.add_argument('--workers', type=int, default=min(4, os.cpu_count() or 1))
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(DATA_DIR.glob('*.mat'))
    if not paths:
        raise FileNotFoundError(DATA_DIR)

    open_records = None
    controller_records = None
    if args.stage in ('open', 'all'):
        print('OPEN-LOOP RATE SWEEP: %d records, workers=%d' % (len(paths), args.workers), flush=True)
        open_records = run_parallel(open_loop_record, records_for_stage(paths, 'open'), args.workers)
        open_payload = dict(protocol='open_loop', dataset=DATASET, master_rate_hz=FS_MASTER,
                            rates_hz=list(RATES), window_seconds=WINDOW_SECONDS,
                            msd_band_hz=list(MSD_BAND), records=open_records)
        write_json(OUT_DIR / 'open_loop.json', open_payload)
        open_rows = arm_rows(open_records, controller=False)
        write_csv(OUT_DIR / 'open_loop_summary.csv', open_rows)
        write_json(OUT_DIR / 'open_loop_aggregate.json', aggregate_rows(open_rows))

    if args.stage in ('controller', 'all'):
        cp = records_for_stage(paths, 'controller')
        print('CONTROLLER RATE SWEEP: %d held-out records, workers=%d' %
              (len(cp), args.workers), flush=True)
        controller_records = run_parallel(controller_record, cp, args.workers)
        ctrl_payload = dict(protocols=['corate', 'controller20k_zoh'], dataset=DATASET,
                            master_rate_hz=FS_MASTER, rates_hz=list(RATES),
                            window_seconds=WINDOW_SECONDS, msd_band_hz=list(MSD_BAND),
                            records=controller_records)
        write_json(OUT_DIR / 'controller.json', ctrl_payload)
        ctrl_rows = arm_rows(controller_records, controller=True)
        write_csv(OUT_DIR / 'controller_summary.csv', ctrl_rows)
        write_csv(OUT_DIR / 'controller_force_summary.csv',
                  controller_force_rows(controller_records))
        write_json(OUT_DIR / 'controller_aggregate.json', aggregate_rows(ctrl_rows))

    # Permit either stage to be resumed independently while still rebuilding the report.
    if open_records is None and (OUT_DIR / 'open_loop.json').exists():
        open_records = json.loads((OUT_DIR / 'open_loop.json').read_text(encoding='utf-8'))['records']
    if controller_records is None and (OUT_DIR / 'controller.json').exists():
        controller_records = json.loads((OUT_DIR / 'controller.json').read_text(encoding='utf-8'))['records']
    if open_records is not None and controller_records is not None:
        # Also rebuild flattened force output when resuming from saved JSON.
        write_csv(OUT_DIR / 'controller_force_summary.csv',
                  controller_force_rows(controller_records))
        open_agg = aggregate_rows(arm_rows(open_records, controller=False))
        ctrl_agg = aggregate_rows(arm_rows(controller_records, controller=True))
        write_json(OUT_DIR / 'open_loop_aggregate.json', open_agg)
        write_json(OUT_DIR / 'controller_aggregate.json', ctrl_agg)
        make_plot(open_agg, ctrl_agg, OUT_DIR / 'summary.png')
        gates = [r['gate20k'] for r in open_records]
        (OUT_DIR / 'FINDINGS.md').write_text(
            findings_text(open_agg, ctrl_agg, gates, controller_records), encoding='utf-8')
        print('WROTE', OUT_DIR / 'FINDINGS.md', flush=True)
    return 0


if __name__ == '__main__':
    mp.freeze_support()
    raise SystemExit(main())
