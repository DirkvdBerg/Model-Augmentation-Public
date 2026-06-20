"""Falsifiable plots for the memory-issue diagnosis of gantry_interconnect_dynamic.py.

Each figure is set up as a hypothesis test: a prediction made from one set of
measurements is compared against an INDEPENDENT measurement, and the deviation
is stated on the figure. If the diagnosis (predictable allocation, no leak)
were wrong, the plots would show it.

All numbers are MEASURED on the cluster (blade4, AMD EPYC 7742) with
scripts/gantry/verification/_memory_diagnostic.py:
  - Stage 1/2/3 tables: SLURM jobs 66233 / 66236 (2026-06-11)
  - Direct single-step measurement at the failing config: --measure-step 4000 1200 2 400 1
  - Failing training job: 66051 (sacct: ReqMem=16Gn, MaxRSS=12.3 GB, OOM at first batch)

Outputs (all in simulations/gantry_subnet/diagnostics/):
  memory_diagnostic_data.json   every measured number + derived predictions/errors
  mem1_bptt_scaling.png   Test 1: is step memory linear in batch*nf, and does the
                          fit from SMALL configs predict the failing config?
  mem2_no_leak.png        Test 2: does RSS or epoch time grow over epochs (leak)?
  mem3_data_expansion.png Test 3: does the zero-free-parameter windowing formula
                          predict the training-array size?
  mem4_memory_budget.png  Test 4: does the resulting budget reproduce the OOM of
                          job 66051, and does the proposed config fit?
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
OUT_DIR = os.path.join(_ROOT, 'simulations', 'gantry_subnet', 'diagnostics')
os.makedirs(OUT_DIR, exist_ok=True)

## ==============================================================================
## Measured data (server, job 66236 unless noted)
## ==============================================================================

# Stage 2, fresh-subprocess peak measurement: (batch, nf, up_sample) -> peak-base MB
STAGE2 = [
    (64,   100, 2,  53.0),
    (256,  100, 2,  78.0),
    (1000, 100, 2, 214.0),
    (256,   25, 2,  18.0),
    (256,  400, 2, 321.0),
    (256,  100, 1,  48.0),
]
# Independent direct measurement of the exact failing config (na=nb=400,
# full stride-1 data array held): MEASURE_RESULT base=2628.7 peak=13339.5
DIRECT_UNITS = 4000 * 1200
DIRECT_PEAK_MB = 13339.5 - 2628.7   # 10710.8 MB step peak above baseline
DIRECT_BASE_MB = 2628.7             # data array + model + python baseline

# sacct of failing job 66051
SACCT_MAXRSS_MB = 12308124 / 1024.0  # 12.0 GB (last sample before the kill)
JOB_LIMIT_MB = 16 * 1024.0           # ReqMem = 16Gn

# Stage 3 (job 66233): RSS [MB] and wall time [s] per epoch, tiny config
STAGE3_RSS = [929, 929, 929, 929, 929, 929, 929, 929, 929, 928, 929, 928, 929, 929, 929]
STAGE3_T   = [75.6, 76.2, 76.1, 76.1, 76.1, 76.2, 76.2, 76.2, 76.2, 76.2,
              76.2, 76.2, 76.2, 76.3, 76.3]

# Stage 1: na=nb=400, stride=1: nf -> measured array MB; plus stride-reduced points
STAGE1_NF = np.array([25, 100, 400, 1200])
STAGE1_MB = np.array([590, 687, 1055, 1875])
STAGE1_STRIDE = [(100, 50, 14), (400, 200, 5), (1200, 600, 3)]  # (nf, stride, MB)
RAW_MB = 1.5
VANILLA_DEEPSI_NF1200_MB = 1875  # Stage 4: vanilla deepSI SS_encoder, same data

# Geometry for the analytic formula (8 trajectories x 8000 samples, nu=ny=3, float32)
NA = NB = 400
NU = NY = 3
N_SAMPLES = [8000] * 8


def analytic_array_mb(nf, stride=1, na=NA, nb=NB):
    """Zero-free-parameter prediction: n_windows x bytes_per_window."""
    nf = int(nf)
    k0 = max(na, nb)
    n_win = sum(len(range(k0 + nf, N + 1, stride)) for N in N_SAMPLES)
    return n_win * (nb * NU + na * NY + nf * (NU + NY)) * 4 / 2**20


## ==============================================================================
## Derived predictions and errors (computed once, saved to JSON, shown on plots)
## ==============================================================================

# Test 1: fit slope through the SMALL up_sample=2 configs only, then predict the
# failing config (batch*nf 48x larger than the largest fit point).
units2 = np.array([b * f for b, f, u, m in STAGE2 if u == 2])
peaks2 = np.array([m for b, f, u, m in STAGE2 if u == 2])
slope = np.sum(units2 * peaks2) / np.sum(units2 ** 2)   # MB per (batch*nf) unit
pred_fail_mb = slope * DIRECT_UNITS
t1_dev = (DIRECT_PEAK_MB - pred_fail_mb) / pred_fail_mb * 100.0

# Test 2: linear fit of RSS and epoch time vs epoch (a leak shows as positive slope)
ep = np.arange(len(STAGE3_RSS))
rss_slope = np.polyfit(ep, STAGE3_RSS, 1)[0]   # MB / epoch
t_slope = np.polyfit(ep, STAGE3_T, 1)[0]       # s / epoch

# Test 3: analytic formula vs measured array sizes (stride=1 and stride=nf/2)
t3_pred_s1 = np.array([analytic_array_mb(f) for f in STAGE1_NF])
t3_err_s1 = (STAGE1_MB - t3_pred_s1) / t3_pred_s1 * 100.0
t3_pred_str = np.array([analytic_array_mb(f, s) for f, s, m in STAGE1_STRIDE])
t3_meas_str = np.array([m for f, s, m in STAGE1_STRIDE])
t3_err_str = (t3_meas_str - t3_pred_str) / t3_pred_str * 100.0

# Test 4: budget of the failing run vs sacct, and the proposed config
baseline_mb = 750.0                          # python + torch + model (base minus data)
data_fail = 1875.0                           # Stage 1, measured
tape_fail = DIRECT_PEAK_MB                   # measured directly
total_fail = baseline_mb + data_fail + tape_fail
t4_dev = (total_fail - SACCT_MAXRSS_MB) / SACCT_MAXRSS_MB * 100.0
data_prop = analytic_array_mb(400, stride=10)
tape_prop = slope * 256 * 400                # from the Test-1 fit
total_prop = baseline_mb + data_prop + tape_prop

json_payload = {
    'provenance': {
        'measured_with': 'scripts/gantry/verification/_memory_diagnostic.py',
        'node': 'blade4 (AMD EPYC 7742)', 'date': '2026-06-11',
        'slurm_jobs': {'stage_1_2': 66236, 'stage_3_leak': 66233,
                       'failing_training_run': 66051},
    },
    'stage2_step_peaks': [
        {'batch': b, 'nf': f, 'up_sample': u, 'peak_mb': m} for b, f, u, m in STAGE2],
    'direct_failing_config': {
        'batch': 4000, 'nf': 1200, 'up_sample': 2, 'na_nb': 400,
        'base_mb': DIRECT_BASE_MB, 'step_peak_mb': DIRECT_PEAK_MB},
    'sacct_job_66051': {'reqmem_mb': JOB_LIMIT_MB, 'maxrss_mb': SACCT_MAXRSS_MB,
                        'state': 'OUT_OF_MEMORY', 'elapsed': '00:02:37'},
    'stage3_leak_check': {'rss_mb_per_epoch': STAGE3_RSS, 'epoch_time_s': STAGE3_T},
    'stage1_array_sizes': {
        'nf': STAGE1_NF.tolist(), 'measured_mb_stride1': STAGE1_MB.tolist(),
        'stride_points_nf_stride_mb': STAGE1_STRIDE, 'raw_data_mb': RAW_MB,
        'vanilla_deepsi_nf1200_mb': VANILLA_DEEPSI_NF1200_MB},
    'geometry': {'na': NA, 'nb': NB, 'nu': NU, 'ny': NY,
                 'n_samples_per_traj': N_SAMPLES},
    'test1_linearity': {
        'fit_slope_mb_per_batch_nf': slope,
        'predicted_failing_step_peak_mb': pred_fail_mb,
        'measured_failing_step_peak_mb': DIRECT_PEAK_MB,
        'deviation_pct': t1_dev},
    'test2_leak': {'rss_slope_mb_per_epoch': rss_slope,
                   'epoch_time_slope_s_per_epoch': t_slope},
    'test3_windowing_formula': {
        'stride1_predicted_mb': t3_pred_s1.tolist(),
        'stride1_error_pct': t3_err_s1.tolist(),
        'stride_reduced_predicted_mb': t3_pred_str.tolist(),
        'stride_reduced_error_pct': t3_err_str.tolist()},
    'test4_budget': {
        'failing_total_mb': total_fail, 'sacct_maxrss_mb': SACCT_MAXRSS_MB,
        'deviation_pct_vs_maxrss': t4_dev,
        'proposed': {'batch': 256, 'nf': 400, 'stride': 10,
                     'data_mb': data_prop, 'tape_mb': tape_prop,
                     'total_mb': total_prop}},
}
with open(os.path.join(OUT_DIR, 'memory_diagnostic_data.json'), 'w') as fh:
    json.dump(json_payload, fh, indent=2)

## ==============================================================================
## Figure 1, Test 1: if step memory is a deterministic allocation, a line fitted
## through SMALL configs must predict the 48x larger failing config.
## ==============================================================================

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.loglog(units2, peaks2, 'o', color='C0', ms=8,
          label='Measured small configs, up_sample=2 (fit set)')
b1, f1, _, m1 = STAGE2[-1]
ax.loglog([b1 * f1], [m1], 's', color='C2', ms=8, label='Measured, up_sample=1')

xs = np.logspace(np.log10(units2.min() / 2), np.log10(DIRECT_UNITS * 2), 100)
ax.loglog(xs, slope * xs, '--', color='C0', lw=1,
          label=f'Fit through small configs: {slope*1000:.1f} kB per (batch x nf)')
ax.loglog([DIRECT_UNITS], [pred_fail_mb], 'x', color='C0', ms=12, mew=2,
          label=f'PREDICTION at batch=4000, nf=1200: {pred_fail_mb/1024:.1f} GB')
ax.loglog([DIRECT_UNITS], [DIRECT_PEAK_MB], '*', color='C3', ms=18,
          label=f'INDEPENDENT measurement there: {DIRECT_PEAK_MB/1024:.1f} GB')

ax.text(0.97, 0.05,
        f'Prediction vs measurement at the failing config: {t1_dev:+.0f}%\n'
        'A leak or unbounded growth would not stay on a line\n'
        'fitted to configs 48x to 750x smaller.',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
        bbox=dict(boxstyle='round', fc='white', ec='0.5'))
ax.set_xlabel('batch_size x nf  (windows x rollout steps per training step)')
ax.set_ylabel('Peak memory of ONE training step [MB]')
ax.set_title('Test 1: is step memory a predictable linear function of batch x nf?')
ax.grid(True, which='both', alpha=0.3)
ax.legend(fontsize=8, loc='upper left')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'mem1_bptt_scaling.png'), dpi=150)

## ==============================================================================
## Figure 2, Test 2: a leak grows with epochs; measure the growth rate.
## ==============================================================================

fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
axes[0].plot(ep, STAGE3_RSS, 'o-', color='C0')
axes[0].plot(ep, np.polyval(np.polyfit(ep, STAGE3_RSS, 1), ep), '--', color='k', lw=1,
             label=f'fitted growth: {rss_slope:+.2f} MB/epoch '
                   f'({rss_slope*100:+.0f} MB over 100 epochs)')
axes[0].set_ylabel('Process RSS [MB]')
axes[0].set_ylim(0, max(STAGE3_RSS) * 1.3)
axes[0].set_title('Test 2: does memory or epoch time grow over epochs? '
                  '(server, job 66233)')
axes[0].legend(fontsize=9, loc='lower center')
axes[0].grid(True, alpha=0.3)
axes[1].plot(ep, STAGE3_T, 's-', color='C1')
axes[1].plot(ep, np.polyval(np.polyfit(ep, STAGE3_T, 1), ep), '--', color='k', lw=1,
             label=f'fitted growth: {t_slope:+.3f} s/epoch')
axes[1].set_ylabel('Epoch wall time [s]')
axes[1].set_ylim(0, max(STAGE3_T) * 1.3)
axes[1].set_xlabel('Epoch')
axes[1].legend(fontsize=9, loc='lower center')
axes[1].grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'mem2_no_leak.png'), dpi=150)

## ==============================================================================
## Figure 3, Test 3: the windowing formula has zero free parameters; if window
## duplication explains the array, measurements must land on the curve.
## ==============================================================================

fig, ax = plt.subplots(figsize=(8, 5.5))
nfs = np.linspace(25, 1300, 200)
ax.semilogy(nfs, [analytic_array_mb(f) for f in nfs], '-', color='C0', lw=1,
            label='Prediction (no fit): windows x (na*ny + nb*nu + nf*(nu+ny)) x 4B')
ax.semilogy(STAGE1_NF, STAGE1_MB, 'o', color='C0', ms=8,
            label='Measured, stride=1 (na=nb=400)')
for x, m, e in zip(STAGE1_NF, STAGE1_MB, t3_err_s1):
    ax.annotate(f'{e:+.1f}%', xy=(x, m), xytext=(0, 8),
                textcoords='offset points', ha='center', fontsize=8)
ax.semilogy([1200], [VANILLA_DEEPSI_NF1200_MB], 'd', color='C4', ms=9,
            label='Vanilla deepSI black-box, same data (framework check)')
sx = [f for f, s, m in STAGE1_STRIDE]
ax.semilogy(sx, t3_pred_str, '^', mfc='none', mec='C2', ms=11, mew=1.5,
            label='Prediction with stride=nf/2')
ax.semilogy(sx, t3_meas_str, '^', color='C2', ms=7, label='Measured with stride=nf/2')
ax.axhline(RAW_MB, color='k', lw=1.2, linestyle=':')
ax.text(30, RAW_MB * 1.4, 'Raw measurement data: 1.5 MB (8 traj x 8000 samples)',
        fontsize=8)
ax.text(0.97, 0.18,
        'Formula vs measurement: max |error|\n'
        f'{np.max(np.abs(t3_err_s1)):.1f}% (stride=1), '
        f'{np.max(np.abs(t3_err_str)):.1f}% (stride=nf/2).\n'
        'Vanilla deepSI gives the same size:\n'
        'standard deepSI windowing, not our blocks.',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
        bbox=dict(boxstyle='round', fc='white', ec='0.5'))
ax.set_xlabel('nf (rollout horizon in samples)')
ax.set_ylabel('Training array size [MB]')
ax.set_title('Test 3: does stride-1 window duplication predict the 1.9 GB array?')
ax.grid(True, which='both', alpha=0.3)
ax.legend(fontsize=8, loc='center right')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'mem3_data_expansion.png'), dpi=150)

## ==============================================================================
## Figure 4, Test 4: does the measured budget reproduce the observed OOM, and
## does the proposed config fit the limit?
## ==============================================================================

fig, ax = plt.subplots(figsize=(8, 5.5))
labels = ['Failing run (job 66051)\nbatch=4000, nf=1200, stride=1',
          'Proposed\nbatch=256, nf=400, stride=10']
segs = {
    'Python + torch + model': (baseline_mb, baseline_mb),
    'Training data array': (data_fail, data_prop),
    'BPTT graph (one step peak)': (tape_fail, tape_prop),
}
bottoms = np.zeros(2)
for (name, vals), color in zip(segs.items(), ['0.7', 'C0', 'C3']):
    vals = np.array(vals)
    ax.bar(labels, vals, bottom=bottoms, color=color, label=name, width=0.5)
    bottoms += vals

ax.axhline(JOB_LIMIT_MB, color='k', lw=1.5)
ax.text(0.52, JOB_LIMIT_MB * 1.02, 'SLURM job limit (sacct ReqMem): 16 GB',
        fontsize=9)
ax.plot([0], [SACCT_MAXRSS_MB], '_', color='k', ms=40, mew=2)
ax.text(0.3, SACCT_MAXRSS_MB - 100,
        f'sacct MaxRSS of job 66051: {SACCT_MAXRSS_MB/1024:.1f} GB\n'
        '(last 30 s sample before the oom-kill;\n'
        f'budget total is {t4_dev:+.0f}% vs this sample)',
        fontsize=8, va='top', ha='left')
for i, total in enumerate(bottoms):
    ax.text(i, total + 350, f'{total/1024:.1f} GB', ha='center', fontsize=10,
            weight='bold')

ax.set_ylabel('Memory [MB]')
ax.set_ylim(0, JOB_LIMIT_MB * 1.15)
ax.set_title('Test 4: does the measured budget reproduce the OOM of job 66051?')
ax.legend(fontsize=8, loc='center right')
ax.grid(True, axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'mem4_memory_budget.png'), dpi=150)

## ==============================================================================

print(f'Outputs written to {OUT_DIR}')
print('Test 1 (linearity):  predicted step peak '
      f'{pred_fail_mb/1024:.1f} GB vs measured {DIRECT_PEAK_MB/1024:.1f} GB '
      f'({t1_dev:+.0f}%)')
print(f'Test 2 (leak):       RSS growth {rss_slope:+.2f} MB/epoch, '
      f'epoch time growth {t_slope:+.3f} s/epoch')
print('Test 3 (windowing):  max |error| stride=1 '
      f'{np.max(np.abs(t3_err_s1)):.1f}%, stride=nf/2 '
      f'{np.max(np.abs(t3_err_str)):.1f}%')
print(f'Test 4 (budget):     failing total {total_fail/1024:.1f} GB vs sacct '
      f'MaxRSS {SACCT_MAXRSS_MB/1024:.1f} GB ({t4_dev:+.0f}%); '
      f'proposed total {total_prop/1024:.2f} GB vs 16 GB limit')
