"""
plot_phase1_results.py
----------------------
Regenerate all Phase 1 plots from a saved phase1_results_<run_id>.npz file.

Plots produced:
  1. Validation loss convergence
  2. Output simulation — encoder-init vs zero-state vs reference (X1, X2, Y)
  3. State trajectories  — encoder-init vs zero-state vs reference (stage coords)
  4. Validation trajectory + forces — positions and input forces from .mat file

Usage:
    conda run -n GraduationProject python scripts/gantry/verification/plot_phase1_results.py [path/to/phase1_results_<run_id>.npz]

If no path is given, the most recently modified phase1_results_*.npz in
simulations/gantry_subnet/ is used automatically.
"""

import sys
import os
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from model_augmentation.systems.gantry_ss import P

# ── Locate npz ────────────────────────────────────────────────────────────────
_SIM_DIR  = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'simulations', 'gantry_subnet')
_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'gantry', 'matlab')

N_HOLD = 10000  # hold samples stripped from each end of MATLAB trajectory (0.5 s at 20 kHz)

_DEFAULT_NPZ = os.path.join(_SIM_DIR, 'phase1_results_64869.npz')

if len(sys.argv) > 1:
    npz_path = sys.argv[1]
else:
    npz_path = _DEFAULT_NPZ

print(f'Loading: {npz_path}')
d = np.load(npz_path)

# ── Unpack npz ────────────────────────────────────────────────────────────────
y_ref       = d['y_ref']        # (T, 3) stage positions [m]
y_hat_enc   = d['y_hat_enc']    # (T, 3)
y_zero      = d['y_zero']       # (T, 3)
t_val       = d['t_val']        # (T,)   time axis [s]
epoch_id    = d['epoch_id']     # (E,)
loss_val    = d['loss_val']     # (E,)
nrms_enc    = d['nrms_enc']     # (3,)
nrms_zero   = d['nrms_zero']    # (3,)
x_ref       = d['x_ref']        # (T, 6) logical coordinates
x_enc_phys  = d['x_enc_phys']   # (T, 6) logical, NaN in warmup
x_zero_phys = d['x_zero_phys']  # (T, 6) logical
cheat_n     = int(d['cheat_n'])
dt          = float(d['dt'])
NA          = int(d['NA'])
NB          = int(d['NB'])

cheat_t = cheat_n * dt   # encoder window end [s]

# ── Stage coordinate helpers ──────────────────────────────────────────────────
P_np = P.numpy()   # (3, 3)  q_stage = P.T @ q_logical  (P maps logical forces→stage; P.T maps logical positions→stage)

def to_stage(x_logical_arr):
    """Convert (T, 6) logical-frame array to (T, 6) stage-frame array."""
    q_stage  = (P_np.T @ x_logical_arr[:, 0:3].T).T
    dq_stage = (P_np.T @ x_logical_arr[:, 3:6].T).T
    return np.concatenate([q_stage, dq_stage], axis=1)

# NaN-fill encoder warmup for display (saved npz retains NaN)
x_enc_plot = x_enc_phys.copy()
x_enc_plot[:cheat_n] = x_enc_phys[cheat_n]

x_ref_stage  = to_stage(x_ref)
x_enc_stage  = to_stage(x_enc_plot)
x_zero_stage = to_stage(x_zero_phys)

# ── Output directory — save PNGs next to the npz ──────────────────────────────
out_dir = os.path.dirname(os.path.abspath(npz_path))
run_id  = os.path.basename(npz_path).replace('phase1_results_', '').replace('.npz', '')

# ── Plot 1: Validation loss convergence ───────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(7, 3.5))
ax1.semilogy(epoch_id, loss_val, label='Val loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Validation RMSE')
ax1.set_title('Validation loss convergence')
ax1.legend()
ax1.grid(True, which='both')
fig1.tight_layout()
fig1.savefig(os.path.join(out_dir, f'phase1_val_loss_{run_id}.png'), dpi=150)
print('Saved: phase1_val_loss')

# ── Plot 2: Output simulation ─────────────────────────────────────────────────
ch_labels = ['X1 [m]', 'X2 [m]', 'Y [m]']
fig2, axes2 = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
for ch, (ax, lab) in enumerate(zip(axes2, ch_labels)):
    ax.plot(t_val, y_ref[:, ch],     color='black', lw=0.8, label='Reference')
    ax.plot(t_val, y_hat_enc[:, ch], color='C0',    lw=0.9, label=f'Encoder-init (NRMS={nrms_enc[ch]:.3f})')
    ax.plot(t_val, y_zero[:, ch],    color='C1',    lw=0.9, linestyle='--', label=f'Zero-state   (NRMS={nrms_zero[ch]:.3f})')
    ax.set_ylabel(lab)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True)
axes2[-1].set_xlabel('Time [s]')
fig2.suptitle('Validation simulation - encoder-init vs zero-state')
fig2.tight_layout()
fig2.savefig(os.path.join(out_dir, f'phase1_simulation_{run_id}.png'), dpi=150)
print('Saved: phase1_simulation')

# ── Plot 3: State trajectories — stage coordinates ────────────────────────────
# Per-channel state NRMS over post-warmup window, per channel (not global).
_ref_s = x_ref_stage[cheat_n:]
_eps   = 1e-8
nrms_st_enc  = (np.sqrt(((x_enc_stage[cheat_n:]  - _ref_s)**2).mean(axis=0))
                / (_ref_s.std(axis=0) + _eps))   # (6,)
nrms_st_zero = (np.sqrt(((x_zero_stage[cheat_n:] - _ref_s)**2).mean(axis=0))
                / (_ref_s.std(axis=0) + _eps))   # (6,)

st_labels = ['X1 [m]', 'X2 [m]', 'Y [m]', 'dX1 [m/s]', 'dX2 [m/s]', 'dY [m/s]']
fig3, axes3 = plt.subplots(6, 1, figsize=(12, 12), sharex=True)
for ch, (ax, lab) in enumerate(zip(axes3, st_labels)):
    ax.plot(t_val, x_ref_stage[:, ch],  color='black', lw=0.8, label='Reference (MATLAB)')
    ax.plot(t_val, x_enc_stage[:, ch],  color='C0',    lw=0.9, label=f'Encoder-init (NRMS={nrms_st_enc[ch]:.3f})')
    ax.plot(t_val, x_zero_stage[:, ch], color='C1',    lw=0.9, linestyle='--', label=f'Zero-state   (NRMS={nrms_st_zero[ch]:.3f})')
    ax.set_ylabel(lab)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True)
axes3[-1].set_xlabel('Time [s]')
fig3.suptitle('Validation state trajectory - encoder-init vs zero-state\n'
              'Stage coordinates [X1, X2, Y | dX1, dX2, dY]')
fig3.tight_layout()
fig3.savefig(os.path.join(out_dir, f'phase1_states_{run_id}.png'), dpi=150)
print('Saved: phase1_states')

# ── Plot 4: Training trajectory + forces — shows what the encoder was trained on ──
# Shaded region marks the encoder input window (first cheat_n samples).
mat_train = loadmat(os.path.join(_DATA_DIR, 'gantry_lti_train.mat'), squeeze_me=True)
u_train = mat_train['u'][N_HOLD:-N_HOLD].astype(np.float32)  # (T, 3) stage forces [N]
y_train = mat_train['y'][N_HOLD:-N_HOLD].astype(np.float32)  # (T, 3) stage positions [m]
t_train = np.arange(len(y_train)) * float(mat_train['dt'])
cheat_t_train = cheat_n * float(mat_train['dt'])

pos_labels   = ['X1 [m]',  'X2 [m]',  'Y [m]' ]
force_labels = ['FX1 [N]', 'FX2 [N]', 'FY [N]']

fig4, axes4 = plt.subplots(6, 1, figsize=(12, 12), sharex=True)
for ch in range(3):
    axes4[ch].plot(t_train, y_train[:, ch], color='black', lw=0.8)
    enc_label = f'Encoder input ({cheat_n} samples, {cheat_t_train*1e3:.0f} ms)' if ch == 0 else '_nolegend_'
    axes4[ch].axvspan(t_train[0], cheat_t_train, alpha=0.10, color='steelblue', label=enc_label)
    axes4[ch].axvline(cheat_t_train, color='steelblue', linestyle='--', lw=0.8)
    axes4[ch].set_ylabel(pos_labels[ch])
    axes4[ch].legend(fontsize=7, loc='upper right')
    axes4[ch].grid(True)
for ch in range(3):
    axes4[3 + ch].plot(t_train, u_train[:, ch], color='C2', lw=0.8)
    axes4[3 + ch].set_ylabel(force_labels[ch])
    axes4[3 + ch].grid(True)
axes4[-1].set_xlabel('Time [s]')
fig4.suptitle('Training set - trajectory and input forces\n'
              'Shaded region: encoder input window used to estimate initial state')
fig4.tight_layout()
fig4.savefig(os.path.join(out_dir, f'phase1_trajectory_{run_id}.png'), dpi=150)
print('Saved: phase1_trajectory')

plt.show()
