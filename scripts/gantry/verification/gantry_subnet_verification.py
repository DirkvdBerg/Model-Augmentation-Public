"""
gantry_subnet_verification.py
------------------------------
Verify that the full SSE_Interconnect pipeline (encoder + Gantry_State_Block +
Linear_Output_Block) trains correctly on matched MATLAB data (Phase 1).

Two checks after training:
  1. Encoder-initialised sim-NRMS per channel (X1, X2, Y) — primary success metric
  2. Zero-state sim-NRMS per channel — encoder must beat this

Check 3 (x̂₀ vs x_logical) is not implemented: the encoder output is calibrated to
satisfy Cd @ x̂₀ ≈ (y - y0)/ystd, not x̂₀ ≈ x_logical/std_x. A direct state
comparison is not valid without knowing the model's internal coordinate convention.

Run from project root:
    conda run -n GraduationProject python scripts/gantry/verification/gantry_subnet_verification.py
"""

import sys
import os
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import torch
import matplotlib.pyplot as plt
import deepSI
from scipy.io import loadmat

from model_augmentation.fit_systems.interconnect import Interconnect, SSE_Interconnect
from model_augmentation.fit_systems.blocks import Gantry_State_Block, Linear_Output_Block
from model_augmentation.systems.gantry_ss import Cd, Dd, P

# ── Hyperparameters ───────────────────────────────────────────────────────────
NA     = 200   # encoder history [samples] — 10 ms at 20 kHz
NB     = 200
NF     = 200   # BPTT horizon [samples]   — 10 ms at 20 kHz
EPOCHS = 1
BATCH  = 256
SAVE   = True
N_HOLD = 10000 # hold samples at start/end of each MATLAB trajectory (0.5 s at 20 kHz, no motion)

run_id = os.environ.get('SLURM_JOB_ID') or datetime.now().strftime('%Y%m%d_%H%M%S')

NX, NU, NY = 6, 3, 3

# ── Load data ─────────────────────────────────────────────────────────────────
_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'gantry', 'matlab')

def load_mat(split):
    d = loadmat(os.path.join(_DATA_DIR, f'gantry_lti_{split}.mat'), squeeze_me=True)
    # Strip N_HOLD samples from start and end: those are static hold periods (no motion).
    # The encoder window (cheat_n = max(NA, NB)) must fall during motion, not during the hold,
    # otherwise the encoder sees a trivially static initial condition and is never tested properly.
    return deepSI.System_data(
        u  = d['u'][N_HOLD:-N_HOLD].astype(np.float32),         # (T, 3)  stage forces [N]
        y  = d['y'][N_HOLD:-N_HOLD].astype(np.float32),         # (T, 3)  stage positions [m]
        x  = d['x_logical'][N_HOLD:-N_HOLD].astype(np.float32), # (T, 6)  logical state — encoder verify only
        dt = float(d['dt']),
    )

train_data = load_mat('train')
val_data   = load_mat('val')
print(f'Train: T={len(train_data.u)}  Val: T={len(val_data.u)}  (hold trimmed: {N_HOLD} samples each side)')

# ── Normalisation stats for Gantry_State_Block ────────────────────────────────
# std_x / std_u normalise the physical ODE inputs inside the block.
# These are also used as the SSE_Interconnect's (u, y) normalisation below,
# with no mean subtraction (u0=y0=0), so that:
#   u_phys = u_norm * std_u = u   (block sees full physical force)
#   y_norm = y_phys / ystd        (consistent with Cd_norm below)
std_u = train_data.u.std(axis=0).reshape(NU, 1).astype(np.float32) + 1e-8  # (3, 1) [N]
std_x = train_data.x.std(axis=0).reshape(NX, 1).astype(np.float32) + 1e-8  # (6, 1) [m, m/s]
# 1e-8 guard: Y position is frozen at 0.3 m → std_x[2] ≈ 0 without it.

# Output normalisation (stage frame) — used for Cd_norm and fit_sys.norm.
# Computed from training data so fit_sys.norm can be set before training.
ystd = train_data.y.std(axis=0).astype(np.float32) + 1e-8   # (3,) [m]

# ── Normalised output matrix ──────────────────────────────────────────────────
# Physical output equation: y_phys = Cd @ x_phys = P.T @ q_logical
# In normalised space (x_norm = x_phys/std_x, y_norm = y_phys/ystd):
#   y_norm = Cd_norm @ x_norm   where Cd_norm[i,j] = Cd[i,j] * std_x[j] / ystd[i]
# This applies P.T BEFORE rescaling, so translation (large std_x[0]) and
# rotation (tiny std_x[1]) are mixed with their correct physical weights.
# Without this, the rotational mode is amplified by std_x[0]/std_x[1] ≈ 100-300x.
Cd_norm = Cd.numpy() * std_x.flatten()[None, :] / ystd[:, None]  # (3, 6)

# ── Build Interconnect ────────────────────────────────────────────────────────
interconnect = Interconnect(nx=NX, nu=NU, ny=NY)

gantry_block = Gantry_State_Block(Y_op=0.3, std_x=std_x, std_u=std_u)
output_block  = Linear_Output_Block(C=Cd_norm, D=Dd.numpy())

interconnect.add_block(gantry_block)
interconnect.add_block(output_block)

# Wiring — identical to verify_interconnect.py (already verified bit-exact)
interconnect.connect_signals("x", gantry_block)
interconnect.connect_block_signals(gantry_block, ["u"], [])
interconnect.connect_signals(gantry_block, "xp")

interconnect.connect_signals("x", output_block)
interconnect.connect_block_signals(output_block, ["u"], ["y"])

# ── Train ─────────────────────────────────────────────────────────────────────
fit_sys = SSE_Interconnect(
    na=NA, nb=NB,
    interconnect=interconnect,
    e_net_kwargs={'n_nodes_per_layer': 64, 'n_hidden_layers': 2},
)

# Set normalisation manually — no mean subtraction (u0=y0=0).
# With u0=0: u_norm = u/ustd → Gantry block sees u_phys = u_norm*std_u = u (correct).
# With y0=0: y_norm = y/ystd → consistent with Cd_norm which maps x_norm → y_phys/ystd.
# auto_fit_norm=True would overwrite these with mean-subtracted values, breaking both.
fit_sys.norm.u0   = np.zeros(NU, dtype=np.float32)
fit_sys.norm.ustd = std_u.flatten()
fit_sys.norm.y0   = np.zeros(NY, dtype=np.float32)
fit_sys.norm.ystd = ystd

fit_sys.fit(
    train_sys_data=train_data,
    val_sys_data=val_data,
    epochs=EPOCHS,
    batch_size=BATCH,
    auto_fit_norm=False,
    loss_kwargs={'nf': NF},
    validation_measure='sim-NRMS',
)

fit_sys.eval()

# ── Check 1: Encoder-initialised sim-NRMS ────────────────────────────────────
# apply_experiment: encoder(u[:NB], y[:NA]) → x̂₀, then rolls out the interconnect
# for the full trajectory. Returns fixed_System_data with .normed=False (physical y)
# and .cheat_n = max(NA, NB): the first cheat_n rows are copied verbatim from
# val_data.y (encoder warmup). NRMS must exclude these rows — error is zero there
# by construction, not because the model predicted correctly.
fit_sys.hfn.reset_saved_signals()
sim_result = fit_sys.apply_experiment(val_data)
cheat_n    = sim_result.cheat_n          # = max(NA, NB)
y_hat_enc  = sim_result.y               # (T, 3) physical [m], normed=False
y_ref      = val_data.y                 # (T, 3) physical [m]
x_ref      = val_data.x                 # (T, 6) physical [m, m/s] — x_logical from MATLAB

# State trajectory from encoder-initialised rollout.
# saved_output_signals concatenates ALL interconnect output signals:
#   [xp (NX), y (NY), block_outputs...] → shape (NX+NY+block_dims, T)
# First NX rows are xp — the normalised state (interconnect internal space).
# Denormalise with std_x to recover physical units [m, m/s].
x_enc_norm = np.array(fit_sys.hfn.saved_output_signals)              # (NX+NY+..., T-cheat_n)
x_enc_phys = np.full((len(val_data.y), NX), np.nan, dtype=np.float32)
x_enc_phys[cheat_n:] = (x_enc_norm[:NX, :] * std_x).T               # (T, NX) physical [m, m/s]; NaN for warmup

nrms_enc = (
    np.sqrt(((y_hat_enc[cheat_n:] - y_ref[cheat_n:]) ** 2).mean(axis=0))
    / y_ref[cheat_n:].std(axis=0)
)
print('\n=== Check 1: Encoder-initialised sim-NRMS ===')
for ch, label in enumerate(['X1', 'X2', 'Y ']):
    print(f'  {label}: {nrms_enc[ch]:.4f}')

# ── Check 2: Zero-state sim-NRMS ─────────────────────────────────────────────
# Simulate the same trajectory from x₀ = 0, with no encoder — baseline to beat.
# The interconnect operates on normalised (u, y); output is renormalised after.
val_norm = fit_sys.norm.transform(val_data)  # normalised u and y; x is stripped (becomes None)
T_val    = len(val_norm.u)

x_zero       = torch.zeros(1, NX)
y_zero_norm  = np.zeros((T_val, NY), dtype=np.float32)
x_zero_traj  = np.zeros((T_val, NX), dtype=np.float32)

with torch.no_grad():
    for t in range(T_val):
        x_zero_traj[t] = x_zero.squeeze().numpy()   # state AT t (before step)
        u_t = torch.tensor(val_norm.u[t], dtype=torch.float32).unsqueeze(0)  # (1, 3)
        y_t, x_zero = interconnect(x_zero, u_t)
        y_zero_norm[t] = y_t.squeeze().numpy()       # output AT t (from state at t)

y_zero      = y_zero_norm * fit_sys.norm.ystd + fit_sys.norm.y0  # denormalise → physical [m]
x_zero_phys = x_zero_traj * std_x.flatten()                       # (T, NX) physical [m, m/s]

nrms_zero = (
    np.sqrt(((y_zero[cheat_n:] - y_ref[cheat_n:]) ** 2).mean(axis=0))
    / y_ref[cheat_n:].std(axis=0)
)
print('\n=== Check 2: Zero-state vs encoder sim-NRMS ===')
for ch, label in enumerate(['X1', 'X2', 'Y ']):
    status = 'PASS' if nrms_enc[ch] < nrms_zero[ch] else 'FAIL'
    print(f'  {label}: enc={nrms_enc[ch]:.4f}  zero={nrms_zero[ch]:.4f}  {status}')

# ── Plots ─────────────────────────────────────────────────────────────────────
plot_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'simulations', 'gantry_subnet')
os.makedirs(plot_dir, exist_ok=True)

# Plot 1: Validation loss convergence (mirrors Fig. 4 in Hoekstra 2025)
fig1, ax1 = plt.subplots(figsize=(7, 3.5))
ax1.semilogy(fit_sys.epoch_id, fit_sys.Loss_val)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Validation RMSE')
ax1.set_title('SSE_Interconnect - validation loss convergence')
ax1.grid(True, which='both')
fig1.tight_layout()
fig1.savefig(os.path.join(plot_dir, 'phase1_val_loss.png'), dpi=150)

# Plot 2: Simulated vs reference per channel (mirrors Fig. 5 in Hoekstra 2025)
# Shows encoder-initialised sim, zero-state sim, and reference over the full
# (trimmed) validation trajectory. Shaded region marks the encoder input window
# [t=0, cheat_t]; dashed line marks rollout start.
t_val   = np.arange(len(y_ref)) * val_data.dt   # time axis [s]
cheat_t = cheat_n * val_data.dt                  # encoder window end / rollout start [s]
ch_labels = ['X1 [m]', 'X2 [m]', 'Y [m]']

fig2, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
for ch, (ax, lab) in enumerate(zip(axes, ch_labels)):
    ax.plot(t_val, y_ref[:, ch],     color='black', lw=0.8, label='Reference')
    ax.plot(t_val, y_hat_enc[:, ch], color='C0',    lw=0.9, label=f'Encoder-init (NRMS={nrms_enc[ch]:.3f})')
    ax.plot(t_val, y_zero[:, ch],    color='C1',    lw=0.9, linestyle='--', label=f'Zero-state   (NRMS={nrms_zero[ch]:.3f})')
    # Encoder input window: shaded band [t=0, cheat_t] + dashed boundary at rollout start
    enc_label = f'Encoder input ({cheat_n} samples, {cheat_t*1e3:.0f} ms)' if ch == 0 else '_nolegend_'
    ax.axvspan(t_val[0], cheat_t, alpha=0.10, color='steelblue', label=enc_label)
    ax.axvline(cheat_t, color='steelblue', linestyle='--', lw=0.8)
    ax.set_ylabel(lab)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True)
axes[-1].set_xlabel('Time [s]')
fig2.suptitle('Validation simulation - encoder-init vs zero-state')
fig2.tight_layout()
fig2.savefig(os.path.join(plot_dir, 'phase1_simulation.png'), dpi=150)

# Plot 3: State trajectory — all 6 channels (positions + velocities), stage coordinates.
# Same structure as Plot 2. Reference is x_logical from MATLAB, converted to stage frame.
# Encoder-init and zero-state denormalised from interconnect internal space, also converted.
# Stage positions: q_stage = P.T @ q_logical.  Stage velocities: dq_stage = P.T @ dq_logical.
# P maps logical forces → stage forces; P.T maps logical positions → stage positions.
# Note: velocities are finite-differenced in MATLAB on q_logical at 20 kHz.
P_np = P.numpy()   # (3, 3) transform: q_stage = P.T @ q_logical

def to_stage(x_logical_arr):
    """Convert (T, 6) logical-frame array to (T, 6) stage-frame array."""
    q_stage  = (P_np.T @ x_logical_arr[:, 0:3].T).T   # (T, 3)
    dq_stage = (P_np.T @ x_logical_arr[:, 3:6].T).T   # (T, 3)
    return np.concatenate([q_stage, dq_stage], axis=1)

x_ref_stage      = to_stage(x_ref)
x_zero_stage     = to_stage(x_zero_phys)

# NaN-fill encoder-init warmup so the line starts at cheat_n (avoids gap in plot).
# Saved npz still gets the original x_enc_phys with NaN for the warmup window.
x_enc_plot = x_enc_phys.copy()
x_enc_plot[:cheat_n] = x_enc_phys[cheat_n]          # repeat first valid value
x_enc_stage = to_stage(x_enc_plot)

st_labels = ['X1 [m]', 'X2 [m]', 'Y [m]', 'dX1 [m/s]', 'dX2 [m/s]', 'dY [m/s]']

fig3, axes3 = plt.subplots(6, 1, figsize=(12, 12), sharex=True)
for ch, (ax, lab) in enumerate(zip(axes3, st_labels)):
    ax.plot(t_val, x_ref_stage[:, ch],   color='black', lw=0.8, label='Reference (MATLAB)')
    ax.plot(t_val, x_enc_stage[:, ch],   color='C0',    lw=0.9, label='Encoder-init')
    ax.plot(t_val, x_zero_stage[:, ch],  color='C1',    lw=0.9, linestyle='--', label='Zero-state')
    enc_label = f'Encoder input ({cheat_n} samples, {cheat_t*1e3:.0f} ms)' if ch == 0 else '_nolegend_'
    ax.axvspan(t_val[0], cheat_t, alpha=0.10, color='steelblue', label=enc_label)
    ax.axvline(cheat_t, color='steelblue', linestyle='--', lw=0.8)
    ax.set_ylabel(lab)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True)
axes3[-1].set_xlabel('Time [s]')
fig3.suptitle('Validation state trajectory - encoder-init vs zero-state\n'
              'Stage coordinates [X1, X2, Y | dX1, dX2, dY]')
fig3.tight_layout()
fig3.savefig(os.path.join(plot_dir, 'phase1_states.png'), dpi=150)

plt.show()

# ── Save ──────────────────────────────────────────────────────────────────────
if SAVE:
    # Model checkpoint — can be reloaded with SSE_Interconnect.load_system()
    save_path = os.path.join(plot_dir, f'phase1_{run_id}')
    fit_sys.save_system(save_path)

    # All results in one file — full trajectories, loss history, metrics.
    # Load with: d = np.load('phase1_results_{run_id}.npz'); d['y_hat_enc'] etc.
    results_path = os.path.join(plot_dir, f'phase1_results_{run_id}.npz')
    np.savez(
        results_path,
        # Trajectories — full length (T,3), physical units [m]
        y_ref     = y_ref,
        y_hat_enc = y_hat_enc,
        y_zero    = y_zero,
        t_val     = t_val,        # time axis [s], shape (T,)
        # Convergence
        epoch_id  = np.array(fit_sys.epoch_id),
        loss_val  = np.array(fit_sys.Loss_val),
        # Per-channel NRMS (post-warmup window)
        nrms_enc  = nrms_enc,     # shape (3,): [X1, X2, Y]
        nrms_zero = nrms_zero,    # shape (3,): [X1, X2, Y]
        # State trajectories — full length (T,6), physical units [m, m/s], logical coordinates
        x_ref      = x_ref,        # x_logical from MATLAB
        x_enc_phys = x_enc_phys,   # encoder-init rollout states
        x_zero_phys= x_zero_phys,  # zero-state rollout states
        # Metadata needed to interpret the arrays
        cheat_n   = np.array(cheat_n),
        dt        = np.array(val_data.dt),
        NA        = np.array(NA),
        NB        = np.array(NB),
        NF        = np.array(NF),
    )
    print(f'\nSaved model  : {save_path}')
    print(f'Saved results: {results_path}')
