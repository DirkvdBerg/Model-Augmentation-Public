"""
analyze_param_recovery.py
-------------------------
Load a saved param-recovery .pt file and produce:
  1. Printed summary table  (RMSE + per-parameter recovery)
  2. Normalised-deviation bar chart  (detuned vs learned, % from true)

Run:
    conda run -n GraduationProject python -m lpv_lfr_baseline.scripts.analyze_param_recovery
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path to the result file — edit if needed
# ---------------------------------------------------------------------------
RESULT_FILE = os.path.join(
    os.path.dirname(__file__), '..', '..',
    'simulations', 'param_recovery',
    'lfr_param_recovery_T1-T6-T2-T3-T4-T5_e600_plw0.0.pt',
)

PARAM_NAMES = ['kb1', 'kb2', 'cg1', 'cg2', 'cy', 'cb1', 'cb2',
               'mh', 'm1', 'm2', 'mb', 'Jb', 'Jh']

PARAM_UNITS = {
    'kb1': 'N·m/rad', 'kb2': 'N·m/rad',
    'cg1': 'N·s/m',   'cg2': 'N·s/m',   'cy': 'N·s/m',
    'cb1': 'N·m·s/rad', 'cb2': 'N·m·s/rad',
    'mh': 'kg', 'm1': 'kg', 'm2': 'kg', 'mb': 'kg',
    'Jb': 'kg·m²', 'Jh': 'kg·m²',
}

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
d = torch.load(RESULT_FILE, map_location='cpu', weights_only=False)

params_true    = d['params_true'].numpy()     # (13,)
params_detuned = d['params_init'].numpy()     # (13,)  detuned starting point
log_params     = d['log_params'].numpy()      # (13,)
params_learned = params_detuned * np.exp(log_params)

rmse_baseline = float(d['RMSE_baseline'])     # metres
rmse_trained  = float(d['eval_rmse'])         # metres

epochs            = d['epochs']
lr                = d['lr']
param_loss_weight = d['param_loss_weight']
active_trajs      = ', '.join(d['active_traj_ids'])

# ---------------------------------------------------------------------------
# 1. Print summary
# ---------------------------------------------------------------------------
print('=' * 72)
print('  Parameter recovery — result summary')
print('=' * 72)
print(f'  File    : {os.path.basename(RESULT_FILE)}')
print(f'  Trajs   : {active_trajs}')
print(f'  Epochs  : {epochs}  |  lr={lr:.1e}  |  param_loss_weight={param_loss_weight}')
print()
print(f'  RMSE baseline (detuned)  : {rmse_baseline:.3e} m')
print(f'  RMSE trained             : {rmse_trained:.3e} m')
print(f'  Improvement              : {rmse_baseline / rmse_trained:.1f}x')
print()

# Per-parameter recovery table
detuned_delta = (params_detuned - params_true) / np.abs(params_true) * 100
learned_delta = (params_learned - params_true) / np.abs(params_true) * 100
# % of the initial error recovered toward true (100% = perfect, 0% = no movement)
denom = params_detuned - params_true
recovered_pct = np.where(
    np.abs(denom) > 1e-12,
    (params_detuned - params_learned) / denom * 100,
    np.nan,
)

hdr = f"  {'Param':<6}  {'True':>12}  {'Detuned Δ%':>11}  {'Learned Δ%':>11}  {'Recovered%':>11}  Unit"
print(hdr)
print('  ' + '-' * (len(hdr) - 2))
for i, name in enumerate(PARAM_NAMES):
    print(
        f"  {name:<6}  {params_true[i]:>12.4g}  "
        f"{detuned_delta[i]:>+10.2f}%  "
        f"{learned_delta[i]:>+10.2f}%  "
        f"{recovered_pct[i]:>+10.1f}%  "
        f"{PARAM_UNITS[name]}"
    )
print('=' * 72)

# ---------------------------------------------------------------------------
# 2. Bar chart — normalised deviation (% from true) for detuned vs learned
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))

x = np.arange(len(PARAM_NAMES))
w = 0.35

bars_det = ax.bar(x - w / 2, detuned_delta, w, label='Detuned (init)',
                  color='#aaaaaa', edgecolor='white', linewidth=0.5)

# Colour learned bars by recovery quality
colors = []
for pct in recovered_pct:
    if np.isnan(pct):
        colors.append('#888888')
    elif pct >= 80:
        colors.append('#2ca02c')   # green  — well recovered
    elif pct >= 40:
        colors.append('#ff7f0e')   # orange — partial
    else:
        colors.append('#d62728')   # red    — poor / diverged

bars_lrn = ax.bar(x + w / 2, learned_delta, w, label='Learned',
                  color=colors, edgecolor='white', linewidth=0.5)

ax.axhline(0, color='black', linewidth=0.8)
ax.axhline(+10, color='#aaaaaa', linewidth=0.6, linestyle='--')
ax.axhline(-10, color='#aaaaaa', linewidth=0.6, linestyle='--')

ax.set_xticks(x)
ax.set_xticklabels(PARAM_NAMES, fontsize=9)
ax.set_ylabel('Deviation from true value [%]')
ax.set_title(
    f'Parameter recovery  —  {epochs} epochs\n'
    f'RMSE: {rmse_baseline:.2e} m  →  {rmse_trained:.2e} m  '
    f'({rmse_baseline / rmse_trained:.1f}× improvement)'
)
ax.legend(loc='upper right', fontsize=9)
ax.set_xlim(-0.6, len(PARAM_NAMES) - 0.4)

# Annotate learned bars with recovered% (skip if nan)
for i, (bar, pct) in enumerate(zip(bars_lrn, recovered_pct)):
    if np.isnan(pct):
        continue
    y_pos = bar.get_height()
    va = 'bottom' if y_pos >= 0 else 'top'
    offset = 0.3 if y_pos >= 0 else -0.3
    ax.text(bar.get_x() + bar.get_width() / 2, y_pos + offset,
            f'{pct:.0f}%', ha='center', va=va, fontsize=7, color='#333333')

fig.tight_layout()
out_path = os.path.join(os.path.dirname(RESULT_FILE), 'param_recovery_deviation.pdf')
fig.savefig(out_path, dpi=150)
print(f'\n  Plot saved: {out_path}')
plt.show()
