"""
plot_telica_param_recovery.py
-----------------------------
Post-training analysis for a Telica param recovery run.

Produces:
  Figure 1 -- Train loss curve (mse_loss vs epoch, log-y scale)
  Figure 2 -- Trajectory comparison + residuals per channel
              Left column : measured vs trained overlay
              Right column: residual (sim - measured) with RMS annotated

Run as:
    conda run -n GraduationProject python scripts/gantry/real-data-verification/plot_telica_param_recovery.py
"""

__project_origin__ = "added"

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch

from telica_loader import load_telica_log
from lpv_lfr_baseline.blocks.lfr_param_block import ParameterizedLFRBlock
from lpv_lfr_baseline.scripts.precompute import _build_state_traj_logical
from lpv_lfr_baseline.scripts.train_param_recovery import _run_no_grad
from lpv_lfr_baseline.core.physics import P as _P, ts as _ts

# --- Config ------------------------------------------------------------------

PT_FILE = os.path.join(
    'C:\\Users\\20203253\\OneDrive - TU Eindhoven\\Graduation Project\\Simulation',
    'param_recovery_telica_xpos_-60_ypos-40',
    'lfr_param_recovery_telica_xpos_-60_ypos-40_ETEL_e1500_68254.pt'
)

OP_FOLDER  = 'xpos_-60_ypos-40'
LOG_FILE   = 'iterETEL.log'
_DATA_ROOT = os.path.join(
    _ROOT, 'kamtin-data', 'Data Telica', '06 40 mm XL 80 mm YL', 'train', OP_FOLDER
)
SAVE_DIR   = os.path.join(_ROOT, 'simulations', f'param_recovery_telica_{OP_FOLDER}')
AXES       = ['X1', 'X2', 'Y']

# --- Plot style --------------------------------------------------------------

plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.size':         9,
    'axes.titlesize':    9,
    'axes.labelsize':    9,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'legend.fontsize':   8,
    'lines.linewidth':   0.9,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.color':        '#dddddd',
    'grid.linewidth':    0.5,
    'figure.dpi':        150,
})

C_MEASURED = '#222222'   # near-black for measured signal
C_MODEL    = '#2166ac'   # blue for model output
C_RESID    = '#d6604d'   # red-orange for residual

# --- Helpers -----------------------------------------------------------------

def _simulate(log_params_tensor, log_path, dtype=torch.float64):
    """Simulate full trajectory with given log_params. Returns (y_sim, q1, fs)."""
    import lpv_lfr_baseline.blocks.lfr_param_block as _lfr_pb
    _lfr_pb._DETUNED_PARAMS = _lfr_pb._TRUE_PARAMS

    u, q1, fs = load_telica_log(log_path, dtype=dtype)
    x0        = _build_state_traj_logical(q1[:2], _P.to(dtype), float(_ts), dtype)[:1]
    ts_tensor = torch.tensor(float(_ts), dtype=dtype)
    block     = ParameterizedLFRBlock(RMSE_baseline=1.0).to(dtype=dtype)
    with torch.no_grad():
        block.log_params.copy_(log_params_tensor.to(dtype=dtype))
    result = _run_no_grad(block, x0, u, ts_tensor)
    y_sim  = result.Y[0].detach()
    T      = min(y_sim.shape[0], q1.shape[0])
    return y_sim[:T], q1[:T], fs


def _rmse_nrmse(y_sim, q1):
    diff     = y_sim - q1
    rmse_ch  = diff.pow(2).mean(dim=0).sqrt()
    sigma_ch = q1.std(dim=0).clamp(min=1e-9)
    nrmse_ch = rmse_ch / sigma_ch * 100.0
    rmse_tot = float(diff.pow(2).mean().sqrt())
    return rmse_ch, nrmse_ch, rmse_tot


def _print_table(label, rmse_ch, nrmse_ch, rmse_tot):
    print(f'\n  {label}')
    print(f'  {"Ch":<4}  {"RMSE [m]":>12}  {"NRMSE [%]":>10}  Verdict')
    print(f'  {"-"*4}  {"-"*12}  {"-"*10}  {"-"*12}')
    for i, ax in enumerate(AXES):
        n = nrmse_ch[i].item()
        v = 'GOOD (<15%)' if n < 15 else ('AMBIGUOUS' if n < 30 else 'POOR (>30%)')
        print(f'  {ax:<4}  {rmse_ch[i].item():>12.4e}  {n:>10.2f}  {v}')
    print(f'\n  Overall RMSE: {rmse_tot:.4e} m')


# --- Load .pt ----------------------------------------------------------------

print(f'Loading: {PT_FILE}')
pt = torch.load(PT_FILE, weights_only=False)

history    = pt['history']
log_params = pt['log_params']
dtype      = torch.float64
log_path   = os.path.join(_DATA_ROOT, LOG_FILE)
epochs_run = pt['epochs']
lr         = pt['lr']
run_id     = pt['run_id']

print(f'Run {run_id}: epochs={epochs_run}, lr={lr}, history={len(history)} entries')

os.makedirs(SAVE_DIR, exist_ok=True)

# --- Figure 1: Train loss curve ----------------------------------------------

epochs_list = [h['epoch']    for h in history]
loss_list   = [h['mse_loss'] for h in history]

fig1, ax1 = plt.subplots(figsize=(6, 3))
ax1.semilogy(epochs_list, loss_list, color=C_MODEL, lw=1.0)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Normalised MSE loss')
ax1.set_title('Training loss')
ax1.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
fig1.tight_layout()
fig1_path = os.path.join(SAVE_DIR, f'train_loss_{run_id}.png')
fig1.savefig(fig1_path, bbox_inches='tight')
plt.close(fig1)
print(f'Figure 1 saved: {fig1_path}')

# --- Simulate trained model --------------------------------------------------

print('Simulating trained model...')
y_trained, q1_ref, fs = _simulate(log_params, log_path, dtype=dtype)
rmse_ch, nrmse_ch, rmse_tot = _rmse_nrmse(y_trained, q1_ref)
_print_table('Trained model', rmse_ch, nrmse_ch, rmse_tot)

# --- Summary table -----------------------------------------------------------

print('\n' + '=' * 60)
print('Summary')
print('=' * 60)
print(f'\n  {"Ch":<4}  {"RMSE [m]":>12}  {"NRMSE [%]":>10}  Verdict')
print(f'  {"-"*4}  {"-"*12}  {"-"*10}  {"-"*12}')
for i, ax in enumerate(AXES):
    n = nrmse_ch[i].item()
    v = 'GOOD (<15%)' if n < 15 else ('AMBIGUOUS' if n < 30 else 'POOR (>30%)')
    print(f'  {ax:<4}  {rmse_ch[i].item():>12.4e}  {n:>10.2f}  {v}')
print(f'\n  Overall RMSE: {rmse_tot:.4e} m')

# --- Figure 2: Trajectory overlay + residuals --------------------------------

T   = q1_ref.shape[0]
t_s = (torch.arange(T).float() / fs).numpy()

fig2, axs = plt.subplots(
    3, 2,
    figsize=(10, 7),
    sharex=True,
    constrained_layout=True,
    gridspec_kw={'width_ratios': [2, 1]}
)

for i, ch in enumerate(AXES):
    q_m   = q1_ref[:, i].numpy()
    yf_m  = y_trained[:, i].numpy()
    res_m = (y_trained[:, i] - q1_ref[:, i]).numpy()
    rms_m = float(rmse_ch[i])

    # Left: overlay
    axl = axs[i, 0]
    axl.plot(t_s, q_m,  color=C_MEASURED, lw=0.8, label='Measured')
    axl.plot(t_s, yf_m, color=C_MODEL,    lw=0.8, label='Model', ls='--')
    axl.set_ylabel(f'{ch} [m]')
    axl.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    axl.yaxis.get_major_formatter().set_scientific(True)
    axl.yaxis.get_major_formatter().set_powerlimits((-1, 1))
    if i == 0:
        axl.legend(loc='upper right')
    axl.annotate(
        f'RMS = {rms_m:.2e} m',
        xy=(0.03, 0.05), xycoords='axes fraction',
        fontsize=8, color='#444444'
    )

    # Right: residual
    axr = axs[i, 1]
    axr.plot(t_s, res_m, color=C_RESID, lw=0.7)
    axr.axhline( rms_m, color=C_RESID, lw=0.8, ls=':', alpha=0.7)
    axr.axhline(-rms_m, color=C_RESID, lw=0.8, ls=':', alpha=0.7)
    axr.axhline(0,       color='#aaaaaa', lw=0.6)
    axr.set_ylabel(f'{ch} error [m]')
    axr.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    axr.yaxis.get_major_formatter().set_scientific(True)
    axr.yaxis.get_major_formatter().set_powerlimits((-1, 1))
    axr.annotate(
        f'RMS = {rms_m:.2e} m',
        xy=(0.05, 0.88), xycoords='axes fraction',
        fontsize=8, color=C_RESID
    )

axs[-1, 0].set_xlabel('Time [s]')
axs[-1, 1].set_xlabel('Time [s]')

fig2_path = os.path.join(SAVE_DIR, f'trajectory_comparison_{run_id}.png')
fig2.savefig(fig2_path, bbox_inches='tight')
plt.close(fig2)
print(f'Figure 2 saved: {fig2_path}')
