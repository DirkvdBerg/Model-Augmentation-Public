"""
diagnose_convergence.py
-----------------------
Diagnostic script: why does the gantry SUBNET encoder fail to converge?

Runs a short training (DIAG_EPOCHS) then immediately runs six targeted tests:
  1. Train vs val NRMS table         - overfitting vs fundamental encoder failure
  2. Windowed rollout (full traj)    - where across the trajectory does the encoder fail?
  3. x0 per channel vs position      - is Y wrong? are velocities erratic?
  4. Cd_norm @ x0 consistency        - are initial conditions physically inconsistent?
  5. Per-timestep error growth       - bad x0 (spike at start) vs drift (gradual growth)?
  6. Gradient norm per layer group   - explosion or vanishing before reaching encoder?

The windowed rollout (Tests 2-4) slides through the FULL trajectory including
N_HOLD hold regions - the encoder warmup window is u[t-NB:t], y[t-NA:t], so
no data before t is discarded. This directly tests whether the encoder works
when the warmup falls in the static hold vs the dynamic motion region.

Run from project root:
    conda run -n GraduationProject python scripts/gantry/verification/diagnose_convergence.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import torch
import matplotlib.pyplot as plt
import deepSI
from scipy.io import loadmat

from model_augmentation.fit_systems.interconnect import Interconnect, SSE_Interconnect
from model_augmentation.fit_systems.blocks import Gantry_State_Block, Linear_Output_Block
from model_augmentation.systems.gantry_ss import Cd, Dd

# ── Configuration (must match gantry_subnet_verification.py exactly) ─────────
NA          = 200
NB          = 200
NF          = 200
BATCH       = 256
N_HOLD      = 10000
Y_OP        = 0.3
SEED        = 42
DIAG_EPOCHS = 10   # short training run — enough to diagnose, not to fully converge
NX, NU, NY  = 6, 3, 3

# Windowed rollout stride — NF//2 gives 50% overlap between windows.
WINDOW_STRIDE = NF // 2

np.random.seed(SEED)
torch.manual_seed(SEED)

run_id   = os.environ.get('SLURM_JOB_ID') or 'local'
_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'gantry', 'matlab')
plot_dir  = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'simulations', 'gantry_subnet', 'diagnostics')
os.makedirs(plot_dir, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
def _load_raw(split):
    return loadmat(os.path.join(_DATA_DIR, f'gantry_lti_{split}.mat'), squeeze_me=True)

def load_stripped(split):
    """Same trimming as gantry_subnet_verification.py — for training and Tests 1, 5, 6."""
    d = _load_raw(split)
    return deepSI.System_data(
        u  = d['u'][N_HOLD:-N_HOLD].astype(np.float32),
        y  = d['y'][N_HOLD:-N_HOLD].astype(np.float32),
        x  = d['x_logical'][N_HOLD:-N_HOLD].astype(np.float32),
        dt = float(d['dt']),
    )

def load_full(split):
    """Full trajectory, no trimming — for windowed rollout Tests 2-4."""
    d = _load_raw(split)
    return (
        d['u'].astype(np.float32),
        d['y'].astype(np.float32),
        float(d['dt']),
    )

train_data = load_stripped('train')
val_data   = load_stripped('val')

u_train_full, y_train_full, dt = load_full('train')
u_val_full,   y_val_full,   _  = load_full('val')

T_train_full = len(u_train_full)
T_val_full   = len(u_val_full)

print(f'Stripped  — Train: T={len(train_data.u)}  Val: T={len(val_data.u)}')
print(f'Full traj — Train: T={T_train_full}  Val: T={T_val_full}')
print(f'N_HOLD={N_HOLD}  NA=NB={NA}  NF={NF}  DIAG_EPOCHS={DIAG_EPOCHS}')

# ── Normalisation (identical to gantry_subnet_verification.py) ────────────────
std_u = train_data.u.std(axis=0).reshape(NU, 1).astype(np.float32) + 1e-8
std_x = train_data.x.std(axis=0).reshape(NX, 1).astype(np.float32) + 1e-8
std_x[2] = Y_OP
ystd  = train_data.y.std(axis=0).astype(np.float32) + 1e-8
ystd[2] = Y_OP
Cd_norm = Cd.numpy() * std_x.flatten()[None, :] / ystd[:, None]  # (3, 6)

# ── Build interconnect ────────────────────────────────────────────────────────
interconnect = Interconnect(nx=NX, nu=NU, ny=NY)
gantry_block = Gantry_State_Block(Y_op=Y_OP, std_x=std_x, std_u=std_u)
output_block  = Linear_Output_Block(C=Cd_norm, D=Dd.numpy())
interconnect.add_block(gantry_block)
interconnect.add_block(output_block)
interconnect.connect_signals("x", gantry_block)
interconnect.connect_block_signals(gantry_block, ["u"], [])
interconnect.connect_signals(gantry_block, "xp")
interconnect.connect_signals("x", output_block)
interconnect.connect_block_signals(output_block, ["u"], ["y"])

fit_sys = SSE_Interconnect(
    na=NA, nb=NB,
    interconnect=interconnect,
    e_net_kwargs={'n_nodes_per_layer': 64, 'n_hidden_layers': 2},
)
fit_sys.norm.u0   = np.zeros(NU, dtype=np.float32)
fit_sys.norm.ustd = std_u.flatten()
fit_sys.norm.y0   = np.zeros(NY, dtype=np.float32)
fit_sys.norm.ystd = ystd

# ── Short training run ────────────────────────────────────────────────────────
print(f'\nTraining for {DIAG_EPOCHS} epochs...')
fit_sys.fit(
    train_sys_data=train_data,
    val_sys_data=val_data,
    epochs=DIAG_EPOCHS,
    batch_size=BATCH,
    auto_fit_norm=False,
    loss_kwargs={'nf': NF},
    validation_measure='sim-RMS',
)
fit_sys.checkpoint_load_system(name='_best')
fit_sys.eval()
# checkpoint_load_system replaces fit_sys.__dict__ entirely, so fit_sys.hfn is now
# a freshly-deserialized object (best-epoch weights). Rebind the module-level variable
# so rollout_from_x0 and zero_state_rollout use the same weights as the encoder.
interconnect = fit_sys.hfn
print('Training done. Running diagnostics...')

# ── Helpers ───────────────────────────────────────────────────────────────────
def nrms(y_hat, y_ref):
    """Per-channel NRMS using training ystd as denominator."""
    return np.sqrt(((y_hat - y_ref) ** 2).mean(axis=0)) / ystd

def normalize_u(u):
    return u / std_u.flatten()

def normalize_y(y):
    return y / ystd

def denormalize_y(y_norm):
    return y_norm * ystd

def rollout_from_x0(x0_tensor, u_norm_seq):
    """
    Roll out the interconnect for len(u_norm_seq) steps from x0_tensor.
    x0_tensor: (1, NX)
    u_norm_seq: (T, NU) numpy, already normalised
    Returns y_hat_phys: (T, NY) numpy, physical units
    """
    x = x0_tensor
    y_out = []
    with torch.no_grad():
        for t in range(len(u_norm_seq)):
            u_t = torch.tensor(u_norm_seq[t], dtype=torch.float32).unsqueeze(0)
            y_t, x = interconnect(x, u_t)
            y_out.append(y_t.squeeze().numpy())
    return denormalize_y(np.array(y_out))

def encoder_x0(u_past_np, y_past_np):
    """
    Run encoder on past window (physical units — normalised inside).
    u_past_np: (NB, NU)  y_past_np: (NA, NY)
    Returns x0: (1, NX) tensor
    """
    u_norm_t = torch.tensor(np.ascontiguousarray(normalize_u(u_past_np)), dtype=torch.float32).unsqueeze(0)
    y_norm_t = torch.tensor(np.ascontiguousarray(normalize_y(y_past_np)), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return fit_sys.encoder(u_norm_t, y_norm_t)

# ── TEST 1: Train vs val NRMS table ──────────────────────────────────────────
print('\n' + '='*60)
print('TEST 1: Train vs val NRMS table')
print('='*60)

fit_sys.hfn.reset_saved_signals()
sim_val   = fit_sys.apply_experiment(val_data)
cheat_n   = sim_val.cheat_n
y_hat_val = sim_val.y
y_ref_val = val_data.y

fit_sys.hfn.reset_saved_signals()
sim_train   = fit_sys.apply_experiment(train_data)
y_hat_train = sim_train.y
y_ref_train = train_data.y

def zero_state_rollout(data):
    data_norm = fit_sys.norm.transform(data)
    x = torch.zeros(1, NX)
    x[0, :] = torch.tensor(data.x[0] / std_x.flatten(), dtype=torch.float32)  # actual state at trim point
    y_out = np.zeros((len(data_norm.u), NY), dtype=np.float32)
    with torch.no_grad():
        for t in range(len(data_norm.u)):
            u_t = torch.tensor(data_norm.u[t], dtype=torch.float32).unsqueeze(0)
            y_t, x = interconnect(x, u_t)
            y_out[t] = y_t.squeeze().numpy()
    return denormalize_y(y_out)

y_zero_val   = zero_state_rollout(val_data)
y_zero_train = zero_state_rollout(train_data)

nrms_enc_val    = nrms(y_hat_val[cheat_n:],    y_ref_val[cheat_n:])
nrms_zero_val   = nrms(y_zero_val[cheat_n:],   y_ref_val[cheat_n:])
nrms_enc_train  = nrms(y_hat_train[cheat_n:],  y_ref_train[cheat_n:])
nrms_zero_train = nrms(y_zero_train[cheat_n:], y_ref_train[cheat_n:])

print(f'\n{"":12s}  {"X1":>8s}  {"X2":>8s}  {"Y":>8s}')
print(f'{"enc (val)":12s}  {nrms_enc_val[0]:8.4f}  {nrms_enc_val[1]:8.4f}  {nrms_enc_val[2]:8.4f}')
print(f'{"zero (val)":12s}  {nrms_zero_val[0]:8.4f}  {nrms_zero_val[1]:8.4f}  {nrms_zero_val[2]:8.4f}')
print(f'{"enc (train)":12s}  {nrms_enc_train[0]:8.4f}  {nrms_enc_train[1]:8.4f}  {nrms_enc_train[2]:8.4f}')
print(f'{"zero (train)":12s}  {nrms_zero_train[0]:8.4f}  {nrms_zero_train[1]:8.4f}  {nrms_zero_train[2]:8.4f}')

if np.all(nrms_enc_train > nrms_zero_train):
    print('\n=> Encoder WORSE than zero-state on TRAINING data: encoder has not learned at all.')
elif np.all(nrms_enc_train < nrms_zero_train) and np.all(nrms_enc_val > nrms_zero_val):
    print('\n=> Encoder better on train but worse on val: OVERFITTING.')
else:
    print('\n=> Mixed result - inspect per-channel.')

# ── TESTS 2, 3, 4: Windowed rollout across full trajectory ───────────────────
print('\n' + '='*60)
print('TESTS 2/3/4: Windowed rollout across full trajectory')
print('='*60)

def windowed_rollout(u_full, y_full):
    """
    Slide a window of length NF through the FULL (unstripped) trajectory.
    At each position t the encoder sees u[t-NB:t], y[t-NA:t] — no data before
    t is discarded. The window includes hold regions.
    """
    T_full = len(u_full)
    u_norm_full = normalize_u(u_full)
    y_norm_full = normalize_y(y_full)

    t_starts    = []
    nrms_enc_w  = []
    nrms_zero_w = []
    x0_windows  = []
    cd_residual = []

    valid_start = max(NA, NB)
    valid_end   = T_full - NF

    for t in range(valid_start, valid_end, WINDOW_STRIDE):
        x0    = encoder_x0(u_full[t-NB:t], y_full[t-NA:t])
        y_hat = rollout_from_x0(x0, u_norm_full[t:t+NF])
        y_ref = y_full[t:t+NF]

        x0_zero = torch.zeros(1, NX)
        x0_zero[0, 2] = Y_OP / std_x[2].item()  # Y channel: normalised Y_OP, not 0
        y_zero  = rollout_from_x0(x0_zero, u_norm_full[t:t+NF])

        t_starts.append(t)
        nrms_enc_w.append(nrms(y_hat, y_ref))
        nrms_zero_w.append(nrms(y_zero, y_ref))
        x0_windows.append(x0.squeeze().numpy())

        # Test 4: Cd_norm @ x0 should approximate y_norm[t]
        cd_pred = Cd_norm @ x0.squeeze().numpy()
        cd_residual.append(np.linalg.norm(cd_pred - y_norm_full[t]))

    return (np.array(t_starts), np.array(nrms_enc_w), np.array(nrms_zero_w),
            np.array(x0_windows), np.array(cd_residual))

print('Running windowed rollout on full train trajectory...')
t_tr, nrms_enc_tr, nrms_zero_tr, x0_tr, cd_tr = windowed_rollout(u_train_full, y_train_full)
print('Running windowed rollout on full val trajectory...')
t_vl, nrms_enc_vl, nrms_zero_vl, x0_vl, cd_vl = windowed_rollout(u_val_full,   y_val_full)

t_tr_s = t_tr * dt
t_vl_s = t_vl * dt

ch_labels = ['X1', 'X2', 'Y']

# Figure 2: NRMS per channel + Cd residual vs window position
fig2, axes2 = plt.subplots(4, 2, figsize=(14, 10), sharex='col')
for col, (t_s, nrms_e, nrms_z, cd_res, T_full, label) in enumerate([
    (t_tr_s, nrms_enc_tr, nrms_zero_tr, cd_tr, T_train_full, 'Train'),
    (t_vl_s, nrms_enc_vl, nrms_zero_vl, cd_vl, T_val_full,   'Val'),
]):
    hold_e = N_HOLD * dt
    hold_s = (T_full - N_HOLD) * dt
    for ch in range(NY):
        ax = axes2[ch, col]
        ax.semilogy(t_s, nrms_e[:, ch], color='C0', lw=0.8, label='Encoder')
        ax.semilogy(t_s, nrms_z[:, ch], color='C1', lw=0.8, linestyle='--', label='Zero-state')
        ax.axvspan(t_s[0], hold_e, alpha=0.12, color='green',  label='Hold (pre-motion)')
        ax.axvspan(hold_s, t_s[-1], alpha=0.12, color='orange', label='Hold (post-motion)')
        ax.set_ylabel(f'NRMS {ch_labels[ch]}')
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(True, which='both')
        if ch == 0:
            ax.set_title(f'{label} trajectory')
    ax = axes2[3, col]
    ax.plot(t_s, cd_res, color='C2', lw=0.8)
    ax.axvspan(t_s[0], hold_e, alpha=0.12, color='green')
    ax.axvspan(hold_s, t_s[-1], alpha=0.12, color='orange')
    ax.set_ylabel('||Cd x0 - y_norm||')
    ax.set_xlabel('Window start time [s]')
    ax.grid(True)
fig2.suptitle(f'Tests 2+4: Windowed rollout NRMS and Cd_norm consistency ({DIAG_EPOCHS} epochs)\n'
              'Green=pre-motion hold, Orange=post-motion hold')
fig2.tight_layout()
fig2.savefig(os.path.join(plot_dir, f'diag_windowed_nrms_{run_id}.png'), dpi=150)
print(f'Saved: diag_windowed_nrms_{run_id}.png')

# Figure 3: x0 per channel vs window position
x0_labels = ['q_log0 [norm]', 'q_log1 [norm]', 'Y [norm]', 'qdot0 [norm]', 'qdot1 [norm]', 'Ydot [norm]']
fig3, axes3 = plt.subplots(NX, 2, figsize=(14, 10), sharex='col')
for col, (t_s, x0_w, T_full, label) in enumerate([
    (t_tr_s, x0_tr, T_train_full, 'Train'),
    (t_vl_s, x0_vl, T_val_full,   'Val'),
]):
    hold_e = N_HOLD * dt
    hold_s = (T_full - N_HOLD) * dt
    for ch in range(NX):
        ax = axes3[ch, col]
        ax.plot(t_s, x0_w[:, ch], color='C3', lw=0.8)
        ax.axvspan(t_s[0], hold_e, alpha=0.12, color='green')
        ax.axvspan(hold_s, t_s[-1], alpha=0.12, color='orange')
        ax.set_ylabel(x0_labels[ch])
        ax.grid(True)
        if ch == 0:
            ax.set_title(f'{label} — x0 per channel')
    axes3[-1, col].set_xlabel('Window start time [s]')
    axes3[2, col].axhline(Y_OP / float(std_x[2].item()), color='black', linestyle='--',
                          lw=0.8, label=f'Expected Y_op ({Y_OP/float(std_x[2].item()):.2f})')
    axes3[2, col].legend(fontsize=6)
fig3.suptitle(f'Test 3: Encoder x0 per channel ({DIAG_EPOCHS} epochs)\n'
              'Channel 2 (Y) should be near expected Y_op line; velocities should be smooth')
fig3.tight_layout()
fig3.savefig(os.path.join(plot_dir, f'diag_x0_channels_{run_id}.png'), dpi=150)
print(f'Saved: diag_x0_channels_{run_id}.png')

print('\nTest 3 — x0 statistics (train windows):')
print(f'  {"Channel":20s}  {"mean":>8s}  {"std":>8s}  {"min":>8s}  {"max":>8s}')
for ch, lab in enumerate(x0_labels):
    vals = x0_tr[:, ch]
    print(f'  {lab:20s}  {vals.mean():8.3f}  {vals.std():8.3f}  {vals.min():8.3f}  {vals.max():8.3f}')
print(f'\n  Expected: Y channel (index 2) mean ~ {Y_OP/float(std_x[2].item()):.3f}')

# ── TEST 5: Per-timestep error growth ─────────────────────────────────────────
print('\n' + '='*60)
print('TEST 5: Per-timestep error growth')
print('='*60)

t_val_ax   = np.arange(len(y_ref_val))   * dt
t_train_ax = np.arange(len(y_ref_train)) * dt

err_enc_val    = np.abs(y_hat_val   - y_ref_val)
err_zero_val   = np.abs(y_zero_val  - y_ref_val)
err_enc_train  = np.abs(y_hat_train - y_ref_train)
err_zero_train = np.abs(y_zero_train - y_ref_train)

fig5, axes5 = plt.subplots(NY, 2, figsize=(14, 7), sharex='col')
for ch, lab in enumerate(ch_labels):
    for col, (t_ax, err_e, err_z, label) in enumerate([
        (t_train_ax, err_enc_train, err_zero_train, 'Train'),
        (t_val_ax,   err_enc_val,   err_zero_val,   'Val'),
    ]):
        ax = axes5[ch, col]
        ax.plot(t_ax, err_e[:, ch], color='C0', lw=0.5, label='Encoder', alpha=0.8)
        ax.plot(t_ax, err_z[:, ch], color='C1', lw=0.5, label='Zero-state', alpha=0.8, linestyle='--')
        ax.axvline(cheat_n * dt, color='steelblue', lw=0.8, linestyle='--', label='Rollout start')
        ax.set_ylabel(f'|error| {lab} [m]')
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(True)
        if ch == 0:
            ax.set_title(f'{label}: per-timestep absolute error')
axes5[-1, 0].set_xlabel('Time [s]')
axes5[-1, 1].set_xlabel('Time [s]')
fig5.suptitle(f'Test 5: Per-timestep error growth ({DIAG_EPOCHS} epochs)\n'
              'Spike at rollout start = bad x0; gradual growth = ODE drift')
fig5.tight_layout()
fig5.savefig(os.path.join(plot_dir, f'diag_error_growth_{run_id}.png'), dpi=150)
print(f'Saved: diag_error_growth_{run_id}.png')

# ── TEST 6: Gradient norms ────────────────────────────────────────────────────
print('\n' + '='*60)
print('TEST 6: Gradient norms per layer group')
print('='*60)

fit_sys.train()

data_norm = fit_sys.norm.transform(train_data)
T_tr = len(data_norm.u)
rng  = np.random.default_rng(SEED)
idx  = rng.integers(max(NA, NB), T_tr - NF, size=BATCH)

uhist   = torch.tensor(np.stack([data_norm.u[i-NB:i] for i in idx]), dtype=torch.float32)
yhist   = torch.tensor(np.stack([data_norm.y[i-NA:i] for i in idx]), dtype=torch.float32)
ufuture = torch.tensor(np.stack([data_norm.u[i:i+NF] for i in idx]), dtype=torch.float32)
yfuture = torch.tensor(np.stack([data_norm.y[i:i+NF] for i in idx]), dtype=torch.float32)

all_params = list(fit_sys.encoder.parameters()) + list(fit_sys.hfn.parameters())
for p in all_params:
    if p.grad is not None:
        p.grad.zero_()

loss_val = fit_sys.loss(uhist, yhist, ufuture, yfuture)
loss_val.backward()

def grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5

enc_norm   = grad_norm(fit_sys.encoder.parameters())
phys_norm  = grad_norm(fit_sys.hfn.parameters())
total_norm = grad_norm(all_params)

print(f'\n  Training loss (one batch): {loss_val.item():.6f}')
print(f'\n  Gradient norms:')
print(f'    Encoder (all layers):   {enc_norm:.4f}')
print(f'    Physics block (hfn):    {phys_norm:.4f}')
print(f'    Total:                  {total_norm:.4f}')

print(f'\n  Encoder per-layer gradient norms:')
for name, p in fit_sys.encoder.named_parameters():
    if p.grad is not None:
        g = p.grad.data.norm(2).item()
        flag = '  *** EXPLODING' if g > 100 else ('  *** VANISHING' if g < 1e-6 else '')
        print(f'    {name:40s}  {g:.6f}{flag}')

if enc_norm > 100:
    print('\n=> GRADIENT EXPLOSION in encoder — BPTT chain too long.')
elif enc_norm < 1e-4:
    print('\n=> GRADIENT VANISHING in encoder — gradient dies before reaching encoder.')
else:
    print(f'\n=> Encoder gradient norm {enc_norm:.4f} — within normal range.')

fit_sys.eval()

# ── Summary ───────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('SUMMARY')
print('='*60)
print(f'  Test 1 — NRMS table printed above.')
print(f'  Test 2 — Windowed NRMS plot: diag_windowed_nrms_{run_id}.png')
print(f'           Look for: encoder NRMS below zero-state in green hold regions?')
print(f'  Test 3 — x0 channels plot:  diag_x0_channels_{run_id}.png')
print(f'           Look for: Y channel (index 2) not near expected line? Erratic velocities?')
print(f'  Test 4 — Cd consistency in bottom row of diag_windowed_nrms_{run_id}.png')
print(f'           Look for: residual large everywhere or only in motion region?')
print(f'  Test 5 — Per-timestep error: diag_error_growth_{run_id}.png')
print(f'           Look for: spike at rollout start (bad x0) vs gradual growth (ODE drift)?')
print(f'  Test 6 — Gradient norms printed above.')

plt.show()
