import os
import sys
import numpy as np
import torch
import deepSI
from scipy.io import loadmat
from datetime import datetime
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from model_augmentation.utils.utils import *
from model_augmentation.utils.torch_nets import zero_init_feed_forward_nn
from model_augmentation.fit_systems.interconnect import *
from model_augmentation.fit_systems.blocks import *
from model_augmentation.systems.gantry_ss import Cd, Dd

## ------------- Hyper params -----------------
# model structure parameters
NX_PHYS = 6   # physical states: q1, q2, q3, dq1, dq2, dq3
NX_ANN  = 2   # ANN latent states (implicit delta_a, vdelta_a)
nxd = NX_PHYS + NX_ANN   # = 8
nu  = 3
ny  = 3
Y_OP = None   # None = LPV self-scheduled; float = frozen operating point [m]
SEED = 42

# training parameters
na = 2*nxd + 1; nb = 2*nxd + 1   # = 17, follows Jan's formula
nf = 200; epochs = 5; batch_size = 256

# utility parameters
save_flag = True
USE_F64  = False
DTYPE_NP = np.float64    if USE_F64 else np.float32
DTYPE_PT = torch.float64 if USE_F64 else torch.float32

run_id = os.environ.get('SLURM_JOB_ID') or datetime.now().strftime('%Y%m%d_%H%M%S')

np.random.seed(SEED)
torch.manual_seed(SEED)

## ------------- Load data -----------------
TRAJ_DIR = os.path.join(os.path.dirname(__file__), '..', '..',
                        'Matlab-output', 'identification-trajectories-no-multisine')
print(f'Trajectory dir: {TRAJ_DIR}')

TRAIN_FILES = [
    'T1_Y_sweep_conservative.mat',
    'T2_X_sym_Y030.mat',
    'T3_X_sym_Y000.mat',
    'T4_X_antisym_Y020.mat',
    'T5_X_sym_Y_sweep.mat',
    'T6_Y_sweep_aggressive.mat',
    'T7_X_antisym_Y_sweep.mat',
    'T8_X_sym_anti_Y_sweep.mat',
]
VAL_FILE  = 'V1_X_sym_Y_mid_sweep.mat'
TEST_FILE = 'E1_X_sym_anti_Y_low_offset_sweep.mat'

def load_traj(filename):
    d = loadmat(os.path.join(TRAJ_DIR, filename), squeeze_me=True)
    return deepSI.System_data(
        u=d['u_total'].astype(DTYPE_NP),
        y=d['q1'].astype(DTYPE_NP),
        dt=1.0 / float(d['fs']),
    )

train_list = [load_traj(f) for f in TRAIN_FILES]
train_data = deepSI.System_data_list(train_list)
val_data   = load_traj(VAL_FILE)
test_data  = load_traj(TEST_FILE)

print(f'Loaded {len(train_list)} training trajectories, 1 val, 1 test')
for i, (f, t) in enumerate(zip(TRAIN_FILES, train_list)):
    print(f'  T{i+1}: {t.u.shape[0]} samples  ({f})')

## ------------- Normalisation -----------------
# Global normalization from concatenated training data.
# Velocities computed via numerical differentiation for std_x / x_mean.
u_all = np.concatenate([t.u for t in train_list])
y_all = np.concatenate([t.y for t in train_list])

fs = 1.0 / train_list[0].dt
x_logical_list = []
for t in train_list:
    vel = np.diff(t.y, axis=0) * fs          # (N-1, 3)
    vel = np.vstack([vel[:1], vel])           # (N, 3) — repeat first sample
    x_logical_list.append(np.hstack([t.y, vel]))  # (N, 6)
x_all = np.concatenate(x_logical_list)

x_mean = x_all.mean(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP)
std_x  = x_all.std(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP) + 1e-8
std_u  = u_all.std(axis=0).reshape(nu, 1).astype(DTYPE_NP) + 1e-8
u_mean = u_all.mean(axis=0).reshape(nu, 1).astype(DTYPE_NP)
ystd   = y_all.std(axis=0).astype(DTYPE_NP) + 1e-8
y0     = (Cd.numpy() @ x_mean.flatten()).astype(DTYPE_NP)

# Cd_norm[i,j] = Cd[i,j] * std_x[j] / ystd[i]
Cd_norm = Cd.numpy() * std_x.flatten()[None, :] / ystd[:, None]  # (3, 6)
Dd_np   = Dd.numpy()                                               # (3, 3)

## ------------- Define augmentation structure -----------------
PHY_IX = np.arange(NX_PHYS)   # [0,1,2,3,4,5]

interconnect = Interconnect(nxd, nu, ny, debugging=False)

physical_state_model_block  = Gantry_State_Block(Y_op=Y_OP, std_x=std_x, std_u=std_u, x_mean=x_mean, u_mean=u_mean).to(DTYPE_PT)
physical_output_model_block = Linear_Output_Block(C=Cd_norm, D=Dd_np)
interconnect.add_block(physical_state_model_block)
interconnect.add_block(physical_output_model_block)

# ----- (dynamic) parallel -------
ANN_state_block = Static_ANN_Block(nz=nxd+nu, nw=nxd, n_nodes_per_layer=64,
                                    net=zero_init_feed_forward_nn, activation=torch.nn.Tanh)
interconnect.add_block(ANN_state_block)

interconnect.connect_block_signals(ANN_state_block, ["x", "u"], ["xp"])

interconnect.connect_signals("x", physical_state_model_block, "concat", selection_matrix(PHY_IX, nxd))
interconnect.connect_block_signals(physical_state_model_block, ["u"], [])
interconnect.connect_signals(physical_state_model_block, "xp", "additive", expansion_matrix(PHY_IX, nxd))

interconnect.connect_signals("x", physical_output_model_block, "concat", selection_matrix(PHY_IX, nxd))
interconnect.connect_block_signals(physical_output_model_block, ["u"], ["y"])

## ------------- Train fit system -----------------
fit_sys = SSE_Interconnect(interconnect=interconnect, na=na, nb=nb,
                            e_net_kwargs={"n_nodes_per_layer": 64, "n_hidden_layers": 2})

# Manual normalisation: Gantry_State_Block is nonlinear, auto_fit_norm=True would break this.
# u0=mean(u): block has matching u_mean so it recovers physical forces correctly.
# y0=Cd@x_mean: mean output offset consistent with x_mean in the block.
fit_sys.norm.u0   = u_mean.flatten()
fit_sys.norm.ustd = std_u.flatten()
fit_sys.norm.y0   = y0
fit_sys.norm.ystd = ystd

fit_sys.fit(train_sys_data=train_data, val_sys_data=val_data, batch_size=batch_size,
            epochs=epochs, auto_fit_norm=False, loss_kwargs={'nf': nf},
            validation_measure="sim-RMS")

## ------------- Save fit system -----------------
save_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'simulations', 'gantry_subnet')
os.makedirs(save_dir, exist_ok=True)
if save_flag:
    model_file_path = os.path.join(save_dir, f'gantry_{run_id}')
    fit_sys.save_system(model_file_path)
    print(f'Saved model: {model_file_path}')

## =============================================================================
## Evaluation
## =============================================================================

# Capture full loss history before best-checkpoint restore truncates it.
fit_sys.checkpoint_load_system(name='_last')
epoch_id_full   = fit_sys.epoch_id.copy()
loss_val_full   = fit_sys.Loss_val.copy()
loss_train_full = fit_sys.Loss_train.copy()
fit_sys.checkpoint_load_system(name='_best')
fit_sys.eval()

# ── Encoder-initialised simulation ───────────────────────────────────────────
fit_sys.hfn.reset_saved_signals()
sim_result = fit_sys.apply_experiment(val_data)
cheat_n   = sim_result.cheat_n
y_hat_enc = sim_result.y       # (T, 3) physical [m]
y_ref     = val_data.y

x_enc_norm = np.array(fit_sys.hfn.saved_output_signals)           # (nxd+ny+..., T-cheat_n)
x_enc_phys = np.full((len(y_ref), NX_PHYS), np.nan, dtype=DTYPE_NP)
x_enc_phys[cheat_n:] = (x_enc_norm[:NX_PHYS, :] * std_x + x_mean).T
x_enc_ann  = np.full((len(y_ref), NX_ANN), np.nan, dtype=DTYPE_NP)
x_enc_ann[cheat_n:]  = x_enc_norm[NX_PHYS:nxd, :].T

nrms_enc = (np.sqrt(((y_hat_enc[cheat_n:] - y_ref[cheat_n:]) ** 2).mean(axis=0)) / ystd)
print('\n=== Encoder-initialised sim-NRMS ===')
for ch, lbl in enumerate(['X1', 'X2', 'Y ']):
    print(f'  {lbl}: {nrms_enc[ch]:.4f}')

ann_rms_enc = np.sqrt((x_enc_ann[cheat_n:] ** 2).mean(axis=0))
print('\n=== ANN latent state RMS ===')
for ch in range(NX_ANN):
    print(f'  x[{NX_PHYS+ch}]: enc={ann_rms_enc[ch]:.4e}')

# ── x_logical-initialised simulation (oracle baseline) ───────────────────────
# Only available when val_data has state information (x_logical).
if hasattr(val_data, 'x') and val_data.x is not None:
    val_norm = fit_sys.norm.transform(val_data)
    u_val_norm = torch.tensor(np.ascontiguousarray(val_norm.u), dtype=DTYPE_PT)

    x_xlog = torch.zeros(1, nxd)
    x_xlog[0, :NX_PHYS] = torch.tensor((val_data.x[0] - x_mean.flatten()) / std_x.flatten(), dtype=DTYPE_PT)

    y_xlog_list = []
    with torch.no_grad():
        for t in range(len(u_val_norm)):
            y_t, x_xlog = fit_sys.hfn(x_xlog, u_val_norm[t:t+1])
            y_xlog_list.append(y_t.squeeze().numpy())
    y_hat_xlog = np.array(y_xlog_list) * ystd + y0

    nrms_xlog = np.sqrt(((y_hat_xlog - y_ref) ** 2).mean(axis=0)) / ystd
    print('\n=== x_logical-initialised sim-NRMS ===')
    for ch, lbl in enumerate(['X1', 'X2', 'Y ']):
        print(f'  {lbl}: {nrms_xlog[ch]:.4f}')
    HAS_ORACLE = True
else:
    print('\n=== x_logical-initialised simulation skipped (no state data) ===')
    y_hat_xlog = None
    nrms_xlog  = None
    HAS_ORACLE = False

# ── Plots ─────────────────────────────────────────────────────────────────────
t_val   = np.arange(len(y_ref)) * val_data.dt
cheat_t = cheat_n * val_data.dt

# Plot 1: Loss convergence
fig1, ax1 = plt.subplots(figsize=(7, 3.5))
ax1.semilogy(epoch_id_full, loss_val_full,   color='C0', label='Val loss')
ax1.semilogy(epoch_id_full, loss_train_full, color='C1', linestyle='--', alpha=0.7, label='Train loss')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('sim-RMS')
ax1.set_title(f'Loss convergence - dynamic parallel (NX_ANN={NX_ANN})')
ax1.legend(); ax1.grid(True, which='both')
fig1.tight_layout()
fig1.savefig(os.path.join(save_dir, f'gantry_val_loss_{run_id}.png'), dpi=150)

# Plot 2: Validation simulation
ch_labels = ['X1 [m]', 'X2 [m]', 'Y [m]']
fig2, axes2 = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
for ch, (ax, lab) in enumerate(zip(axes2, ch_labels)):
    ax.plot(t_val, y_ref[:, ch], 'k', lw=0.8, label='Reference')
    ax.plot(t_val, y_hat_enc[:, ch], 'C0', lw=0.9, label=f'Encoder-init (NRMS={nrms_enc[ch]:.3f})')
    if HAS_ORACLE:
        ax.plot(t_val, y_hat_xlog[:, ch], 'C1', lw=0.9, linestyle='--', label=f'x_logical-init (NRMS={nrms_xlog[ch]:.3f})')
    enc_lbl = f'Encoder warmup ({cheat_n} samples)' if ch == 0 else '_nolegend_'
    ax.axvspan(t_val[0], cheat_t, alpha=0.10, color='steelblue', label=enc_lbl)
    ax.axvline(cheat_t, color='steelblue', linestyle='--', lw=0.8)
    ax.set_ylabel(lab); ax.legend(fontsize=7, loc='upper right'); ax.grid(True)
axes2[-1].set_xlabel('Time [s]')
fig2.suptitle(f'Validation simulation - dynamic parallel (NX_ANN={NX_ANN})')
fig2.tight_layout()
fig2.savefig(os.path.join(save_dir, f'gantry_simulation_{run_id}.png'), dpi=150)

# Plot 3: ANN latent state trajectories
fig3, axes3 = plt.subplots(NX_ANN, 1, figsize=(12, 4), sharex=True)
for ch, ax in enumerate(axes3):
    ax.plot(t_val, x_enc_ann[:, ch], 'C0', lw=0.8, label=f'Encoder-init (RMS={ann_rms_enc[ch]:.2e})')
    ax.axvspan(t_val[0], cheat_t, alpha=0.10, color='steelblue')
    ax.axvline(cheat_t, color='steelblue', linestyle='--', lw=0.8)
    ax.set_ylabel(f'x[{NX_PHYS+ch}]'); ax.legend(fontsize=7); ax.grid(True)
axes3[-1].set_xlabel('Time [s]')
fig3.suptitle('ANN latent states x[6:8] (dimensionless)')
fig3.tight_layout()
fig3.savefig(os.path.join(save_dir, f'gantry_ann_states_{run_id}.png'), dpi=150)

plt.show()

# ── Results npz ───────────────────────────────────────────────────────────────
if save_flag:
    save_dict = dict(
        y_ref=y_ref, y_hat_enc=y_hat_enc, t_val=t_val,
        epoch_id=epoch_id_full, loss_val=loss_val_full, loss_train=loss_train_full,
        nrms_enc=nrms_enc, x_enc_phys=x_enc_phys, x_enc_ann=x_enc_ann,
        cheat_n=np.array(cheat_n), dt=np.array(val_data.dt),
        na=np.array(na), nb=np.array(nb), nf=np.array(nf),
        NX_PHYS=np.array(NX_PHYS), NX_ANN=np.array(NX_ANN), nxd=np.array(nxd),
    )
    if HAS_ORACLE:
        save_dict['y_hat_xlog'] = y_hat_xlog
        save_dict['nrms_xlog'] = nrms_xlog
    np.savez(os.path.join(save_dir, f'gantry_results_{run_id}.npz'), **save_dict)
    print(f'Saved results: gantry_results_{run_id}.npz')
