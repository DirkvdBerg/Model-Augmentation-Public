import os
import sys
import json
import numpy as np
import torch
import deepSI
from scipy.io import loadmat
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from model_augmentation.utils.utils import *
from model_augmentation.utils.torch_nets import zero_init_feed_forward_nn, HybridGantryEncoder
from model_augmentation.fit_systems.interconnect import *
from model_augmentation.fit_systems.blocks import *
from model_augmentation.fit_systems.pre_encoder import linear_encoder_init_aug
from model_augmentation.systems.gantry_ss import Cd, Dd, P
from model_augmentation.systems.gantry_linearization import gantry_linearize_and_discretize
from model_augmentation.utils.utils import normalize_linear_ss_matrices

## ═══════════════════════════════════════════════════════════════════════════════
## Configuration
## ═══════════════════════════════════════════════════════════════════════════════

# --- Data source: 'trajectories' or 'multisine' ---
MODE = 'multisine'

# --- Fixed model constants ---
NX_PHYS = 6   # physical states: q1, q2, q3, dq1, dq2, dq3
nu  = 3
ny  = 3
Y_OP = None   # None = LPV self-scheduled; float = frozen operating point [m]
# Encoder: 'linear_map' = Hoekstra 2026 reconstructability init (trainable);
#          'hybrid' = analytical physical + learned augmented (detached);
#          'default' = standard deepSI learned encoder
ENCODER_INIT = 'linear_map'
USE_HYBRID_ENCODER = (ENCODER_INIT == 'hybrid')
ANN_ACTIVATION = 'tanh'    # 'tanh' = nonlinear ANN (default); 'linear' = Identity activation (Jan's ECC setup)
SEED = 42

# --- Resampling ---
FS_ORIG = 20000
FS_NEW  = 4000          # None = no downsampling (use FS_ORIG); int = target sample rate [Hz]
if FS_NEW is None:
    FS_NEW = FS_ORIG
D       = FS_ORIG // FS_NEW
TS_NEW  = 1.0 / FS_NEW

# --- Dtype ---
USE_F64  = False
DTYPE_NP = np.float64    if USE_F64 else np.float32
DTYPE_PT = torch.float64 if USE_F64 else torch.float32

# --- Utility ---
save_flag = True
run_id = os.environ.get('SLURM_JOB_ID') or datetime.now().strftime('%Y%m%d_%H%M%S')

# --- Optuna hyperparameter search ---
USE_OPTUNA = False
N_OPTUNA_TRIALS = 40
OPTUNA_STUDY_NAME = "gantry_subnet_augmented"

# --- Time-based horizons (converted to samples via TS_NEW) ---
NF_SECONDS = 0.100   # [s] rollout horizon (5*tau_msd, tau=1/(zeta*wn)=20ms, 5tau=100ms)
# THEORY: na=nb=nxd*2+1 (Jan's standard; nxd=NX_PHYS+NX_ANN encoder history)

# --- Default hyperparameters (used when USE_OPTUNA=False) ---
DEFAULT_HP = dict(
    NX_ANN=2,
    n_nodes_per_layer=16,
    n_hidden_layers=2,
    up_sample=2,
    nf=max(1, int(NF_SECONDS / TS_NEW)),
    na_nb=0,          # set below via Jan's formula
    batch_size=256,
    lr=1e-4,
    epochs=10,
)
DEFAULT_HP['na_nb'] = (NX_PHYS + DEFAULT_HP['NX_ANN']) * 2 + 1   # THEORY: nxd*2+1 (Jan's standard)

# Diagnostic interval: run aug-state R2 snapshot every N epochs during training
DIAG_INTERVAL = 10

## ═══════════════════════════════════════════════════════════════════════════════
## Data loading (run once)
## ═══════════════════════════════════════════════════════════════════════════════

np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_SUBDIR = 'multisine' if MODE == 'multisine' else 'trajectories'
TRAJ_DIR = os.path.join(os.path.dirname(__file__), '..', '..',
                        'data', 'gantry', 'matlab', DATA_SUBDIR)
print(f'Data dir ({MODE}): {TRAJ_DIR}')

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

def _load_u(d):
    """Return plant input: 'u_total' for multisine data, 'u' for trajectory data."""
    if 'u_total' in d:
        return d['u_total']
    return d['u']

def load_traj(filename):
    d = loadmat(os.path.join(TRAJ_DIR, filename), squeeze_me=True)
    return deepSI.System_data(
        u=_load_u(d)[::D].astype(DTYPE_NP),
        y=d['y'][::D].astype(DTYPE_NP),
        dt=TS_NEW,
    )

def load_mat_aug(filename):
    """Load u, y, x_logical, delta_a, vdelta_a from augmented simulation mat file.

    Used for diagnostics only (not for training). Returns:
      u         (N, nu)   plant input [N]
      y         (N, ny)   plant output [m]
      x_logical (N, 6)    physical states from augmented sim [m, m/s]
      x_aug     (N, 2)    [delta_a, vdelta_a] -- vdelta_a via backward FD
    """
    d = loadmat(os.path.join(TRAJ_DIR, filename), squeeze_me=True)
    u         = _load_u(d)[::D].astype(DTYPE_NP)
    y         = d['y'][::D].astype(DTYPE_NP)
    x_logical = d['x_logical'][::D].astype(DTYPE_NP)    # (N, 6)
    delta_a   = d['delta_a'][::D].astype(DTYPE_NP)      # (N,)
    # HEURISTIC: backward FD for velocity -- O(Ts) accurate, consistent with x_logical convention
    vdelta_a      = np.zeros_like(delta_a)
    vdelta_a[1:]  = (delta_a[1:] - delta_a[:-1]) * FS_NEW
    vdelta_a[0]   = vdelta_a[1]
    x_aug = np.stack([delta_a, vdelta_a], axis=1)       # (N, 2)
    return u, y, x_logical, x_aug

train_list = [load_traj(f) for f in TRAIN_FILES]
train_data = deepSI.System_data_list(train_list)
val_data   = load_traj(VAL_FILE)
test_data  = load_traj(TEST_FILE)

# Load augmented ground-truth fields for validation trajectory (diagnostics only)
_, _, val_x_logical, val_x_aug = load_mat_aug(VAL_FILE)
print(f'Loaded val augmented GT: x_logical={val_x_logical.shape}  '
      f'delta_a std={val_x_aug[:,0].std():.3e} m  '
      f'vdelta_a std={val_x_aug[:,1].std():.3e} m/s')

print(f'Loaded {len(train_list)} training trajectories, 1 val, 1 test')
for i, (f, t) in enumerate(zip(TRAIN_FILES, train_list)):
    print(f'  T{i+1}: {t.u.shape[0]} samples  ({f})')

## ═══════════════════════════════════════════════════════════════════════════════
## Normalisation (run once - all NX_PHYS-dimensional, independent of NX_ANN)
## ═══════════════════════════════════════════════════════════════════════════════

u_all = np.concatenate([t.u for t in train_list])
y_all = np.concatenate([t.y for t in train_list])

fs = 1.0 / train_list[0].dt
P_inv_T = np.linalg.inv(P.numpy().T).astype(DTYPE_NP)  # stage -> logical
x_logical_list = []
for t in train_list:
    pos_logical = (P_inv_T @ t.y.T).T        # (N, 3) stage -> logical
    vel_logical = np.diff(pos_logical, axis=0) * fs  # (N-1, 3)
    vel_logical = np.vstack([vel_logical[:1], vel_logical])  # (N, 3)
    x_logical_list.append(np.hstack([pos_logical, vel_logical]))  # (N, 6)
x_all = np.concatenate(x_logical_list)

x_mean = x_all.mean(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP)
std_x  = x_all.std(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP) + 1e-8
std_u  = u_all.std(axis=0).reshape(nu, 1).astype(DTYPE_NP) + 1e-8
u_mean = u_all.mean(axis=0).reshape(nu, 1).astype(DTYPE_NP)
ystd   = y_all.std(axis=0).astype(DTYPE_NP) + 1e-8
y0     = y_all.mean(axis=0).astype(DTYPE_NP)

# Cd_norm[i,j] = Cd[i,j] * std_x[j] / ystd[i]
Cd_norm = Cd.numpy() * std_x.flatten()[None, :] / ystd[:, None]  # (3, 6)
Dd_np   = Dd.numpy()                                               # (3, 3)

PHY_IX = np.arange(NX_PHYS)   # [0,1,2,3,4,5]

_encoder_tag = ENCODER_INIT
save_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'simulations',
                        'gantry_subnet', f'{MODE}_{_encoder_tag}')
os.makedirs(save_dir, exist_ok=True)


## ═══════════════════════════════════════════════════════════════════════════════
## build_model / train_model
## ═══════════════════════════════════════════════════════════════════════════════

def _get_encoder_dims(hp):
    """Return (na, nb, na_right, nb_right) consistent with build_model logic."""
    if ENCODER_INIT == 'linear_map':
        na = 4 * NX_PHYS + 1
        nb = na
        na_right = 1
        nb_right = 1
    else:
        na = hp.get('na_nb', 2 * (NX_PHYS + hp['NX_ANN']) + 1)
        nb = na
        na_right = 0
        nb_right = 0
    return na, nb, na_right, nb_right


def build_model(hp):
    """Build interconnect + SSE_Interconnect from hp dict, return fit_sys (untrained)."""
    NX_ANN = hp['NX_ANN']
    nxd = NX_PHYS + NX_ANN

    # Encoder history: for linear_map, use Jan's 4*nx+1 rule; otherwise time-based
    # HEURISTIC: Jan's rule of thumb na=4*nx+1=25 for nx=6; na_right=1 needed so
    # reconstructability map can use y(k) to compute x(k).
    na, nb, na_right, nb_right = _get_encoder_dims(hp)

    ic = Interconnect(nxd, nu, ny, debugging=False)

    phy_block = Gantry_State_Block(
        Y_op=Y_OP, std_x=std_x, std_u=std_u,
        x_mean=x_mean, u_mean=u_mean, Ts=TS_NEW,
        up_sample=hp['up_sample'],
    ).to(DTYPE_PT)
    # CHANGED: output augmentation -- y = Cd@x_phys + C_aug@x_aug (D-065).
    # Gradient path: loss -> y -> C_aug -> x_aug -> ANN. Never passes through
    # A_phys integrators (|z|=1 at 4kHz, diag14). Without this, ANN->x_aug is
    # unobservable because y=Cd@x_phys ignores x_aug (diag11 T1: ANN grad=0).
    # Parameterized so C_aug trains jointly with the ANN.
    # HEURISTIC: C_aug[2,0]=1e-2 (Y <- delta_a): absorber primarily displaces
    # Y axis; 1e-2 keeps the init contribution sub-percent of normalized output
    # (ANN is zero-init, encoder x_aug is O(1) normalized). Verified by diag15.
    C_aug_init = np.zeros((ny, NX_ANN), dtype=DTYPE_NP)
    C_aug_init[2, 0] = 1e-2   # HEURISTIC: Y <- delta_a
    C_full     = np.hstack([Cd_norm, C_aug_init])         # (ny, nxd)
    out_block  = Parameterized_Linear_Output_Block(C=C_full, D=Dd_np, flag_loss_reg=False)
    ic.add_block(phy_block)
    ic.add_block(out_block)

    _act = torch.nn.Identity if ANN_ACTIVATION == 'linear' else torch.nn.Tanh
    AUG_IX = np.arange(NX_PHYS, nxd)   # augmented state indices [6, 7]
    ann_block = Static_ANN_Block(
        nz=nxd + nu, nw=NX_ANN,         # CHANGED: nw=NX_ANN (was nxd) — ANN outputs only to augmented states
        n_nodes_per_layer=hp['n_nodes_per_layer'],
        n_hidden_layers=hp['n_hidden_layers'],
        net=zero_init_feed_forward_nn,
        activation=_act,
    )
    ic.add_block(ann_block)

    # CHANGED: route ANN output only to augmented state rows [NX_PHYS:nxd] via expansion_matrix,
    # matching Jan's journal approach (msd_ndof_interconnect_fit.py line 203-208).
    # Previously connect_block_signals routed ANN to all nxd states, allowing the ANN to
    # corrupt physical state dynamics and destabilise the near-unit-circle gantry eigenvalues.
    ic.connect_block_signals(ann_block, ["x", "u"], [])
    ic.connect_signals(ann_block, "xp", "additive", expansion_matrix(AUG_IX, nxd))
    ic.connect_signals("x", phy_block, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(phy_block, ["u"], [])
    ic.connect_signals(phy_block, "xp", "additive", expansion_matrix(PHY_IX, nxd))
    ic.connect_signals("x", out_block, "concat", selection_matrix(np.arange(nxd), nxd))  # CHANGED: full x (was PHY_IX)
    ic.connect_block_signals(out_block, ["u"], ["y"])

    fit_sys = SSE_Interconnect(
        interconnect=ic, na=na, nb=nb,
        na_right=na_right, nb_right=nb_right,
        e_net_kwargs={
            "n_nodes_per_layer": hp['n_nodes_per_layer'],
            "n_hidden_layers": hp['n_hidden_layers'],
        },
    )

    # Manual normalisation: Gantry_State_Block is nonlinear, auto_fit_norm=True would break this.
    fit_sys.norm.u0   = u_mean.flatten()
    fit_sys.norm.ustd = std_u.flatten()
    fit_sys.norm.y0   = y0
    fit_sys.norm.ystd = ystd

    # --- Encoder injection (BEFORE init_model) ---
    if ENCODER_INIT == 'linear_map':
        # THEORY: Hoekstra 2026 Eq. 16-17 — initialize encoder from reconstructability map
        Ad, Bd, Cd_dt, Dd_dt = gantry_linearize_and_discretize(dt=TS_NEW)

        # Build System_data_with_x for normalization (needs state trajectories)
        baseline_npz_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'gantry',
            'baseline_simulations', f'{MODE}_LPV', 'baseline_states.npz')
        if os.path.exists(baseline_npz_path):
            bl = np.load(baseline_npz_path, allow_pickle=True)
            x_phys_all = np.concatenate(bl['x_train_phys'])
        else:
            # Fallback: use the finite-diff states from the data loading section
            x_phys_all = x_all
            print("WARNING: baseline_states.npz not found, using finite-diff states for normalization")

        sys_data_with_x = deepSI.System_data(u=u_all, y=y_all)
        sys_data_with_x.x = x_phys_all

        Ad_bar, Bd_bar, Cd_bar, Dd_bar = normalize_linear_ss_matrices(
            Ad, Bd, Cd_dt, Dd_dt, sys_data_with_x)

        # CHANGED: D-055 -- single encoder replaces linear_encoder_init + LinearInitEncoderWrapper.
        # na/nb passed without right extension: W^b is built for (na+1) timesteps, and
        # the SSE_Interconnect window with na_right=1 gives exactly (na+1) timesteps.
        # Convention fix (D-017) applied natively via u_mean/std_u/y0/ystd/x_mean/std_x.
        fit_sys.encoder = linear_encoder_init_aug(
            A=Ad_bar, B=Bd_bar, C=Cd_bar, D=Dd_bar,
            nx=NX_PHYS, nu=nu, ny=ny, na=na, nb=nb,
            nx_aug=NX_ANN,
            n_nodes_per_layer=hp['n_nodes_per_layer'],
            n_hidden_layers=hp['n_hidden_layers'],
            flag_linear_only=False,
            u_mean=u_mean, std_u=std_u,
            y0=y0, ystd=ystd,
            x_mean=x_mean, std_x=std_x,
        ).to(DTYPE_PT)

    elif ENCODER_INIT == 'hybrid':
        fit_sys.encoder = HybridGantryEncoder(
            nb=nb, nu=nu, na=na, ny=ny, nx=nxd,
            P_inv_T=P_inv_T, y0=y0, ystd=ystd,
            x_mean=x_mean.flatten(), std_x=std_x.flatten(),
            fs=fs, NX_PHYS=NX_PHYS,
            n_nodes_per_layer=hp['n_nodes_per_layer'],
            n_hidden_layers=hp['n_hidden_layers'],
        ).to(DTYPE_PT)
    # else: ENCODER_INIT == 'default' — let SSE_Interconnect create its own

    fit_sys.init_model(sys_data=train_data, auto_fit_norm=False)
    if ENCODER_INIT in ('hybrid', 'linear_map'):
        fit_sys.hfn.to(DTYPE_PT)
    else:
        for net in (fit_sys.encoder, fit_sys.hfn):
            net.to(DTYPE_PT)

    return fit_sys


def train_model(fit_sys, hp, epochs=None):
    """Train fit_sys for given epochs. Supports continuation (call multiple times)."""
    fit_sys.fit(
        train_sys_data=train_data, val_sys_data=val_data,
        batch_size=hp['batch_size'], epochs=epochs or hp['epochs'],
        auto_fit_norm=False,
        loss_kwargs={'nf': hp['nf']},
        optimizer_kwargs={'lr': hp['lr']},
        validation_measure="sim-RMS",
    )
    return fit_sys.bestfit


## ═══════════════════════════════════════════════════════════════════════════════
## evaluate_and_save
## ═══════════════════════════════════════════════════════════════════════════════

def compute_gradient_norms(fit_sys, hp):
    """Single forward+backward pass on training data, return gradient norms per parameter group."""
    fit_sys.train()
    data_train = fit_sys.make_training_data(fit_sys.norm.transform(train_data), nf=hp['nf'])
    batch_size = min(hp['batch_size'], len(data_train[0]))
    batch = [torch.tensor(d[:batch_size], dtype=DTYPE_PT) for d in data_train]

    fit_sys.optimizer.zero_grad()
    loss = fit_sys.loss(*batch, nf=hp['nf'])
    loss.backward()

    grad_norms = {}
    for attr_name, item in fit_sys.parameters_with_names.items():
        params = item['params']
        if isinstance(params, nn.Parameter):
            params = [params]
        else:
            params = list(params)
        for i, p in enumerate(params):
            pname = f'{attr_name}.{i}'
            if p.grad is not None:
                grad_norms[pname] = float(torch.norm(p.grad).item())
            else:
                grad_norms[pname] = 0.0

    # Aggregate by parameter group (encoder, hfn.blocks.0=physics, hfn.blocks.2=ANN)
    group_norms = {}
    for name, norm in grad_norms.items():
        if name.startswith('encoder'):
            group = 'encoder'
        elif name.startswith('hfn'):
            group = 'hfn'
        else:
            group = 'other'
        group_norms[group] = group_norms.get(group, 0.0) + norm ** 2
    group_norms = {k: float(np.sqrt(v)) for k, v in group_norms.items()}

    print('\n=== Gradient norms (single batch, post-training) ===')
    for group, norm in sorted(group_norms.items()):
        print(f'  {group:20s}: {norm:.4e}')

    return grad_norms, group_norms


def evaluate_and_save(fit_sys, hp, rid, diag_conv=None, baseline_nrms=None):
    """Load best checkpoint, simulate, compute NRMS, plot, save."""
    NX_ANN = hp['NX_ANN']
    nxd = NX_PHYS + NX_ANN
    na, nb, na_right, nb_right = _get_encoder_dims(hp)

    # Save model
    if save_flag:
        model_path = os.path.join(save_dir, f'gantry_{rid}')
        fit_sys.save_system(model_path)
        print(f'Saved model: {model_path}')

    # Capture full loss history before best-checkpoint restore truncates it.
    fit_sys.checkpoint_load_system(name='_last')
    epoch_id_full   = fit_sys.epoch_id.copy()
    loss_val_full   = fit_sys.Loss_val.copy()
    loss_train_full = fit_sys.Loss_train.copy()
    fit_sys.checkpoint_load_system(name='_best')
    fit_sys.eval()

    # ── Encoder-initialised simulation ──────────────────────────────────────
    fit_sys.hfn.reset_saved_signals()
    sim_result = fit_sys.apply_experiment(val_data)
    cheat_n   = sim_result.cheat_n
    y_hat_enc = sim_result.y       # (T, 3) physical [m]
    y_ref     = val_data.y

    x_enc_norm = np.array(fit_sys.hfn.saved_output_signals)
    x_enc_phys = np.full((len(y_ref), NX_PHYS), np.nan, dtype=DTYPE_NP)
    x_enc_phys[cheat_n:] = (x_enc_norm[:NX_PHYS, :] * std_x + x_mean).T
    x_enc_ann  = np.full((len(y_ref), NX_ANN), np.nan, dtype=DTYPE_NP)
    x_enc_ann[cheat_n:]  = x_enc_norm[NX_PHYS:nxd, :].T

    nrms_enc = np.sqrt(((y_hat_enc[cheat_n:] - y_ref[cheat_n:]) ** 2).mean(axis=0)) / ystd
    if baseline_nrms is not None:
        print('\n=== Sim-NRMS comparison: augmented model vs baseline FP ===')
        for ch, lbl in enumerate(['X1', 'X2', 'Y ']):
            improv = 100.0 * (baseline_nrms[ch] - nrms_enc[ch]) / (baseline_nrms[ch] + 1e-12)
            print(f'  {lbl}:  augmented={nrms_enc[ch]:.4f}  baseline_FP={baseline_nrms[ch]:.4f}'
                  f'  improvement={improv:+.1f}%')
    else:
        print('\n=== Encoder-initialised sim-NRMS ===')
        for ch, lbl in enumerate(['X1', 'X2', 'Y ']):
            print(f'  {lbl}: {nrms_enc[ch]:.4f}')

    ann_rms_enc = np.sqrt((x_enc_ann[cheat_n:] ** 2).mean(axis=0))
    print('\n=== ANN latent state RMS ===')
    for ch in range(NX_ANN):
        print(f'  x[{NX_PHYS+ch}]: enc={ann_rms_enc[ch]:.4e}')

    # ── x_logical-initialised simulation (oracle baseline) ──────────────────
    if hasattr(val_data, 'x') and val_data.x is not None:
        val_norm = fit_sys.norm.transform(val_data)
        u_val_norm = torch.tensor(np.ascontiguousarray(val_norm.u), dtype=DTYPE_PT)

        x_xlog = torch.zeros(1, nxd)
        x_xlog[0, :NX_PHYS] = torch.tensor(
            (val_data.x[0] - x_mean.flatten()) / std_x.flatten(), dtype=DTYPE_PT)

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

    # ── Plots ───────────────────────────────────────────────────────────────
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
    fig1.savefig(os.path.join(save_dir, f'gantry_val_loss_{rid}.png'), dpi=150)

    # Plot 2: Validation simulation
    ch_labels = ['X1 [m]', 'X2 [m]', 'Y [m]']
    fig2, axes2 = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for ch, (ax, lab) in enumerate(zip(axes2, ch_labels)):
        ax.plot(t_val, y_ref[:, ch], 'k', lw=0.8, label='Reference')
        ax.plot(t_val, y_hat_enc[:, ch], 'C0', lw=0.9,
                label=f'Encoder-init (NRMS={nrms_enc[ch]:.3f})')
        if HAS_ORACLE:
            ax.plot(t_val, y_hat_xlog[:, ch], 'C1', lw=0.9, linestyle='--',
                    label=f'x_logical-init (NRMS={nrms_xlog[ch]:.3f})')
        enc_lbl = f'Encoder warmup ({cheat_n} samples)' if ch == 0 else '_nolegend_'
        ax.axvspan(t_val[0], cheat_t, alpha=0.10, color='steelblue', label=enc_lbl)
        ax.axvline(cheat_t, color='steelblue', linestyle='--', lw=0.8)
        ax.set_ylabel(lab); ax.legend(fontsize=7, loc='upper right'); ax.grid(True)
    axes2[-1].set_xlabel('Time [s]')
    fig2.suptitle(f'Validation simulation - dynamic parallel (NX_ANN={NX_ANN})')
    fig2.tight_layout()
    fig2.savefig(os.path.join(save_dir, f'gantry_simulation_{rid}.png'), dpi=150)

    # Plot 3: ANN latent state trajectories
    if NX_ANN == 1:
        fig3, axes3 = plt.subplots(1, 1, figsize=(12, 3), sharex=True)
        axes3 = [axes3]
    else:
        fig3, axes3 = plt.subplots(NX_ANN, 1, figsize=(12, 4), sharex=True)
    for ch, ax in enumerate(axes3):
        ax.plot(t_val, x_enc_ann[:, ch], 'C0', lw=0.8,
                label=f'Encoder-init (RMS={ann_rms_enc[ch]:.2e})')
        ax.axvspan(t_val[0], cheat_t, alpha=0.10, color='steelblue')
        ax.axvline(cheat_t, color='steelblue', linestyle='--', lw=0.8)
        ax.set_ylabel(f'x[{NX_PHYS+ch}]'); ax.legend(fontsize=7); ax.grid(True)
    axes3[-1].set_xlabel('Time [s]')
    fig3.suptitle(f'ANN latent states x[{NX_PHYS}:{nxd}] (dimensionless)')
    fig3.tight_layout()
    fig3.savefig(os.path.join(save_dir, f'gantry_ann_states_{rid}.png'), dpi=150)

    # Plot 4: Training convergence (loss + R2_linmap) — only when diag_conv available
    if diag_conv is not None and len(diag_conv['epochs']) > 0:
        fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 4))

        # Left: loss curves with baseline FP NRMS as horizontal reference
        ax4a.semilogy(epoch_id_full, loss_val_full,   color='C0', label='Val loss')
        ax4a.semilogy(epoch_id_full, loss_train_full, color='C1', linestyle='--',
                      alpha=0.7, label='Train loss')
        if baseline_nrms is not None:
            ch_colors = ['C2', 'C3', 'C4']
            for ch, (lbl, col) in enumerate(zip(['X1', 'X2', 'Y '], ch_colors)):
                ax4a.axhline(baseline_nrms[ch], color=col, linestyle=':', lw=1.2,
                             label=f'Baseline FP {lbl}={baseline_nrms[ch]:.4f}')
        ax4a.set_xlabel('Epoch'); ax4a.set_ylabel('sim-RMS')
        ax4a.set_title('Loss + baseline reference')
        ax4a.legend(fontsize=7); ax4a.grid(True, which='both')

        # Right: R2_linmap for each augmented state channel over epochs
        aug_labels_short = ['delta_a', 'vdelta_a']
        conv_epochs = diag_conv['epochs']
        conv_r2lin  = diag_conv['r2_linmap']   # (n_chunks, NX_ANN)
        for ch in range(hp['NX_ANN']):
            lbl = aug_labels_short[ch] if ch < len(aug_labels_short) else f'x_ann[{ch}]'
            ax4b.plot(conv_epochs, conv_r2lin[:, ch], marker='o', ms=4, label=lbl)
        ax4b.axhline(0.0, color='k', lw=0.5, linestyle='--')
        ax4b.axhline(1.0, color='k', lw=0.5, linestyle=':')
        ax4b.set_xlabel('Epoch'); ax4b.set_ylabel('R2_linmap')
        ax4b.set_title('Aug state convergence (R2_linmap vs delta_a)')
        ax4b.legend(fontsize=8); ax4b.grid(True)
        ax4b.set_ylim([-0.1, 1.05])

        fig4.suptitle(f'Training convergence (NX_ANN={NX_ANN})')
        fig4.tight_layout()
        fig4.savefig(os.path.join(save_dir, f'gantry_convergence_{rid}.png'), dpi=150)

    plt.close('all')

    # ── Results npz ─────────────────────────────────────────────────────────
    if save_flag:
        save_dict = dict(
            # Predictions and targets
            y_ref=y_ref, y_hat_enc=y_hat_enc, t_val=t_val,
            u_val=val_data.u,
            # Loss curves
            epoch_id=epoch_id_full, loss_val=loss_val_full, loss_train=loss_train_full,
            # Per-channel metrics
            nrms_enc=nrms_enc, x_enc_phys=x_enc_phys, x_enc_ann=x_enc_ann,
            # Normalization constants (for reconstruction)
            std_x=std_x, x_mean=x_mean, std_u=std_u, u_mean=u_mean,
            ystd=ystd, y0=y0, Cd_norm=Cd_norm, Dd_np=Dd_np,
            P_matrix=P.numpy(),
            # Model dimensions
            cheat_n=np.array(cheat_n), dt=np.array(val_data.dt),
            na=np.array(na), nb=np.array(nb), nf=np.array(hp['nf']),
            NX_PHYS=np.array(NX_PHYS), NX_ANN=np.array(NX_ANN), nxd=np.array(nxd),
            # Config metadata
            hp=json.dumps(hp),
            config=json.dumps(dict(
                MODE=MODE, USE_HYBRID_ENCODER=USE_HYBRID_ENCODER,
                ANN_ACTIVATION=ANN_ACTIVATION, FS_NEW=FS_NEW, D=D,
                FS_ORIG=FS_ORIG, SEED=SEED,
            )),
        )
        if HAS_ORACLE:
            save_dict['y_hat_xlog'] = y_hat_xlog
            save_dict['nrms_xlog'] = nrms_xlog
        if diag_conv is not None:
            save_dict['diag_conv_epochs']    = diag_conv['epochs']
            save_dict['diag_conv_r2_raw']    = diag_conv['r2_raw']
            save_dict['diag_conv_r2_linmap'] = diag_conv['r2_linmap']
        if baseline_nrms is not None:
            save_dict['baseline_nrms'] = baseline_nrms
        np.savez(os.path.join(save_dir, f'gantry_results_{rid}.npz'), **save_dict)
        print(f'Saved results: gantry_results_{rid}.npz')


## ═══════════════════════════════════════════════════════════════════════════════
## state_recovery_diagnostic (D-053)
## ═══════════════════════════════════════════════════════════════════════════════

def state_recovery_diagnostic(fit_sys, hp, rid, max_windows=2000):
    """Linear-map state recovery test: basis rotation vs lost information.

    Compares encoder state estimates x_hat(k) on the validation set against
    physical states reconstructed from measurements:
      R2_raw      x_hat[:, :6] read directly as normalized physical states
      R2_linmap   best least-squares linear map x_true ~ x_hat @ W + b
      R2_raw_lag1 raw comparison against x_true(k-1) (detects one-sample lag)
    Interpretation:
      R2_linmap ~ 1, R2_raw low  -> information present, basis rotated
      R2_linmap low              -> information absent from encoder state
      R2_raw_lag1 > R2_raw       -> encoder aligned to k-1 (history off-by-one)
    """
    NX_ANN = hp['NX_ANN']
    nxd = NX_PHYS + NX_ANN
    na, nb, na_right, nb_right = _get_encoder_dims(hp)
    fit_sys.eval()

    # True physical states from augmented simulation mat file (more accurate than P_inv+FD)
    x_true = val_x_logical.astype(DTYPE_NP)                                  # (N,6) x_logical from mat
    x_true_norm = (x_true - x_mean.flatten()) / std_x.flatten()              # (N,6)

    # Encoder estimates. deepSI hist convention (na_right=0): ypast = y[k-na:k],
    # upast = u[k-nb:k]; the encoder output initializes the state at time k.
    val_norm = fit_sys.norm.transform(val_data)
    yn = np.ascontiguousarray(val_norm.y, dtype=DTYPE_NP)
    un = np.ascontiguousarray(val_norm.u, dtype=DTYPE_NP)
    N = len(yn)
    k0 = max(na, nb) + 1                                      # +1 so k-1 exists for the lag column
    stride = max(1, (N - k0) // max_windows)                  # HEURISTIC: cap window count to bound memory
    k_ix = np.arange(k0, N, stride)
    ypast = np.stack([yn[k - na : k + na_right] for k in k_ix])  # (Nk, na+na_right, ny)
    upast = np.stack([un[k - nb : k + nb_right] for k in k_ix])  # (Nk, nb+nb_right, nu)
    with torch.no_grad():
        x_hat = fit_sys.encoder(
            torch.tensor(upast, dtype=DTYPE_PT),
            torch.tensor(ypast, dtype=DTYPE_PT),
        ).numpy()                                             # (Nk, nxd)

    xt   = x_true_norm[k_ix]                                  # x_true(k)
    xt_l = x_true_norm[k_ix - 1]                              # x_true(k-1)

    def r2_per_channel(ref, est):
        # THEORY: coefficient of determination R^2 = 1 - SS_res/SS_tot (standard OLS regression)
        ss_res = ((ref - est) ** 2).sum(axis=0)
        ss_tot = ((ref - ref.mean(axis=0)) ** 2).sum(axis=0)
        return 1.0 - ss_res / ss_tot

    r2_raw = r2_per_channel(xt,   x_hat[:, :NX_PHYS])
    r2_lag = r2_per_channel(xt_l, x_hat[:, :NX_PHYS])

    # Best affine map x_true ~ [x_hat, 1] @ W — THEORY: ordinary least squares
    A = np.hstack([x_hat, np.ones((len(x_hat), 1), dtype=DTYPE_NP)])
    W, *_ = np.linalg.lstsq(A, xt, rcond=None)
    r2_lin = r2_per_channel(xt, A @ W)

    labels = ['q1 ', 'q2 ', 'q3 ', 'dq1', 'dq2', 'dq3']
    print('\n=== State recovery diagnostic (D-053) ===')
    print(f'  {len(k_ix)} windows (stride {stride}), na=nb={na}, '
          f'encoder={"hybrid" if USE_HYBRID_ENCODER else "default"}')
    print('  channel   R2_raw      R2_linmap   R2_raw_lag1')
    for ch in range(NX_PHYS):
        print(f'  {labels[ch]}     {r2_raw[ch]:+10.4f}  {r2_lin[ch]:+10.4f}  {r2_lag[ch]:+10.4f}')
    print('  R2_linmap ~ 1 & R2_raw low -> basis rotation;')
    print('  R2_linmap low              -> information absent from encoder state;')
    print('  R2_raw_lag1 > R2_raw       -> encoder aligned to k-1 (one-sample lag)')

    # --- Augmented state R2 vs delta_a / vdelta_a ---
    r2_aug_raw, r2_aug_lin = aug_state_r2(fit_sys, hp)
    aug_labels = ['delta_a  ', 'vdelta_a ']
    aug_notes  = ['(mat file)', '(FD estimate)']
    print('\n=== Augmented state R2 vs saved GT (delta_a/vdelta_a from mat file) ===')
    print(f'  {"state":<12s}  {"R2_raw":>10s}  {"R2_linmap":>10s}  note')
    for ch in range(hp['NX_ANN']):
        lbl  = aug_labels[ch] if ch < len(aug_labels) else f'x_ann[{ch}]'
        note = aug_notes[ch]  if ch < len(aug_notes)  else ''
        print(f'  {lbl}  {r2_aug_raw[ch]:+10.4f}  {r2_aug_lin[ch]:+10.4f}  {note}')
    print('  R2_linmap ~ 1 -> augmented state captured MSD dynamics')
    print('  R2_linmap ~ 0 -> augmented state did not learn delta_a')

    if save_flag:
        np.savez(os.path.join(save_dir, f'gantry_state_recovery_{rid}.npz'),
                 r2_raw=r2_raw, r2_lin=r2_lin, r2_lag=r2_lag,
                 W=W, k_ix=k_ix, x_hat=x_hat, x_true_norm=xt,
                 r2_aug_raw=r2_aug_raw, r2_aug_lin=r2_aug_lin)
        print(f'Saved state recovery diagnostic: gantry_state_recovery_{rid}.npz')


## ═══════════════════════════════════════════════════════════════════════════════
## compute_baseline_fp_nrms
## ═══════════════════════════════════════════════════════════════════════════════

def compute_baseline_fp_nrms(hp):
    """Simulate baseline FP model (no MSD, no ANN) on val data.

    Runs Gantry_State_Block alone (zero ANN contribution) starting from the
    true initial state (val_x_logical[0] from the augmented simulation mat file).
    Compares to val_data.y which contains the augmented simulation output
    (y WITH MSD effect). The NRMS gap measures how much the hidden MSD
    degrades the baseline-only prediction -- the trained augmented model must
    beat this to justify augmentation.

    Returns:
        nrms_baseline  (ny,)  per-channel NRMS
        y_hat_baseline (N, ny) simulated y in physical units [m]
    """
    phy_block = Gantry_State_Block(
        Y_op=Y_OP, std_x=std_x, std_u=std_u,
        x_mean=x_mean, u_mean=u_mean, Ts=TS_NEW,
        up_sample=hp['up_sample'],
    ).to(DTYPE_PT)
    phy_block.eval()

    # Start from true initial state (from augmented sim mat file)
    x0_phys   = val_x_logical[0].astype(DTYPE_NP)
    x_norm_np = (x0_phys - x_mean.flatten()) / std_x.flatten()  # (NX_PHYS,)

    u_val_norm = ((val_data.u - u_mean.flatten()) / std_u.flatten()).astype(DTYPE_NP)

    y_hat_list = []
    with torch.no_grad():
        for t in range(len(val_data.u)):
            u_norm_np = u_val_norm[t]

            # Output: y_norm = Cd_norm @ x_norm + Dd_np @ u_norm (Dd_np ~ 0)
            y_norm = Cd_norm @ x_norm_np + Dd_np @ u_norm_np   # (ny,)
            y_phys = y_norm * ystd + y0
            y_hat_list.append(y_phys)

            # State transition: x_{k+1} = phy_block(x_k, u_k)
            x_t = torch.tensor(x_norm_np, dtype=DTYPE_PT).view(1, NX_PHYS, 1)
            u_t = torch.tensor(u_norm_np, dtype=DTYPE_PT).view(1, nu, 1)
            z   = torch.cat([x_t, u_t], dim=1)     # (1, NX_PHYS+nu, 1)
            x_norm_next = phy_block(z)              # (1, NX_PHYS, 1) or (1, NX_PHYS)
            x_norm_np = x_norm_next.view(NX_PHYS).cpu().numpy()

    y_hat = np.array(y_hat_list, dtype=DTYPE_NP)   # (N, ny)
    y_ref  = val_data.y
    nrms   = np.sqrt(((y_hat - y_ref)**2).mean(axis=0)) / ystd

    print('\n=== Baseline FP model sim-NRMS (no MSD, reference to beat) ===')
    for ch, lbl in enumerate(['X1', 'X2', 'Y ']):
        print(f'  {lbl}: {nrms[ch]:.4f}')

    return nrms, y_hat


## ═══════════════════════════════════════════════════════════════════════════════
## aug_state_r2  (lightweight, reused during convergence tracking)
## ═══════════════════════════════════════════════════════════════════════════════

def aug_state_r2(fit_sys, hp):
    """Encoder augmented state R2 vs delta_a / vdelta_a from mat file.

    Runs the encoder on strided validation windows and computes:
      R2_raw    -- direct comparison (encoder output vs normalized GT)
      R2_linmap -- best affine map from ALL encoder outputs to GT channel
                   (catches arbitrary scale/offset, shows information content)

    Both quantities returned as arrays of shape (NX_ANN,).
    """
    NX_ANN = hp['NX_ANN']
    na, nb, na_right, nb_right = _get_encoder_dims(hp)
    fit_sys.eval()

    val_norm = fit_sys.norm.transform(val_data)
    yn = np.ascontiguousarray(val_norm.y, dtype=DTYPE_NP)
    un = np.ascontiguousarray(val_norm.u, dtype=DTYPE_NP)
    N  = len(yn)
    k0 = max(na, nb) + 1
    stride = max(1, (N - k0) // 2000)    # HEURISTIC: cap to ~2000 windows for speed
    k_ix = np.arange(k0, N, stride)

    ypast = np.stack([yn[k - na : k + na_right] for k in k_ix])
    upast = np.stack([un[k - nb : k + nb_right] for k in k_ix])
    with torch.no_grad():
        x_hat = fit_sys.encoder(
            torch.tensor(upast, dtype=DTYPE_PT),
            torch.tensor(ypast, dtype=DTYPE_PT),
        ).numpy()   # (Nk, NX_PHYS + NX_ANN)

    x_ann = x_hat[:, NX_PHYS:]                    # (Nk, NX_ANN)

    # Normalize GT so encoder (dimensionless) and GT are on comparable axes
    gt_raw  = val_x_aug[k_ix]                     # (Nk, NX_ANN) physical units
    gt_mean = gt_raw.mean(axis=0)
    gt_std  = gt_raw.std(axis=0) + 1e-8
    gt_norm = (gt_raw - gt_mean) / gt_std          # (Nk, NX_ANN) normalized

    def _r2(ref, est):
        ss_res = ((ref - est)**2).sum(axis=0)
        ss_tot = ((ref - ref.mean(axis=0))**2).sum(axis=0)
        return 1.0 - ss_res / (ss_tot + 1e-12)

    r2_raw = _r2(gt_norm, x_ann)

    # Best affine map: gt_norm ~ [x_ann_all_channels, 1] @ W (per GT channel)
    # THEORY: ordinary least squares -- uses all NX_ANN encoder channels jointly
    A_aug = np.hstack([x_ann, np.ones((len(x_ann), 1), dtype=DTYPE_NP)])
    W_aug, *_ = np.linalg.lstsq(A_aug, gt_norm, rcond=None)
    r2_lin = _r2(gt_norm, A_aug @ W_aug)

    return r2_raw, r2_lin


## ═══════════════════════════════════════════════════════════════════════════════
## train_model_with_diagnostics
## ═══════════════════════════════════════════════════════════════════════════════

def train_model_with_diagnostics(fit_sys, hp):
    """Train in DIAG_INTERVAL-epoch chunks; record aug-state R2 at each checkpoint.

    Calls train_model() repeatedly, runs aug_state_r2() after each chunk.
    Returns bestfit (float) and diag_conv dict with convergence arrays.
    """
    total_epochs = hp['epochs']
    n_chunks  = max(1, total_epochs // DIAG_INTERVAL)
    remainder = total_epochs % DIAG_INTERVAL

    diag_epochs  = []
    diag_r2_raw  = []
    diag_r2_lin  = []

    print(f'\nTraining {total_epochs} epochs in {n_chunks} chunks of {DIAG_INTERVAL} '
          f'(+{remainder} remainder), NX_ANN={hp["NX_ANN"]}')

    for chunk in range(n_chunks):
        epochs_this = DIAG_INTERVAL + (remainder if chunk == n_chunks - 1 else 0)
        bestfit = train_model(fit_sys, hp, epochs=epochs_this)

        fit_sys.eval()
        r2_raw, r2_lin = aug_state_r2(fit_sys, hp)
        epoch_now = int(fit_sys.epoch_id[-1]) if len(fit_sys.epoch_id) > 0 else (chunk + 1) * epochs_this
        diag_epochs.append(epoch_now)
        diag_r2_raw.append(r2_raw.copy())
        diag_r2_lin.append(r2_lin.copy())

        r2_str = '  '.join([f'{hp_name}={r2_lin[i]:+.4f}'
                            for i, hp_name in enumerate(['delta_a', 'vdelta_a'][:hp['NX_ANN']])])
        print(f'  [epoch {epoch_now:4d}]  bestfit={bestfit:.5f}  R2_linmap: {r2_str}')

    diag_conv = dict(
        epochs   = np.array(diag_epochs),
        r2_raw   = np.array(diag_r2_raw),     # (n_chunks, NX_ANN)
        r2_linmap= np.array(diag_r2_lin),     # (n_chunks, NX_ANN)
    )
    return bestfit, diag_conv


## ═══════════════════════════════════════════════════════════════════════════════
## Optuna objective
## ═══════════════════════════════════════════════════════════════════════════════

OPTUNA_EPOCHS      = 30   # total epochs per trial
OPTUNA_CHUNK_SIZE  = 10   # epochs per pruning checkpoint

def objective(trial):
    hp = dict(
        NX_ANN            = trial.suggest_categorical("NX_ANN", [2, 4]),
        n_nodes_per_layer = trial.suggest_categorical("n_nodes_per_layer", [64, 128, 256]),
        n_hidden_layers   = trial.suggest_int("n_hidden_layers", 1, 3),
        na_nb             = trial.suggest_int("na_nb", 10, 50, step=5),
        nf                = trial.suggest_int("nf", 50, 500, step=50),
        batch_size        = trial.suggest_categorical("batch_size", [1000, 2000, 4000]),
        lr                = trial.suggest_float("lr", 5e-5, 5e-3, log=True),
        epochs            = OPTUNA_EPOCHS,
    )

    print(f"\n{'='*70}")
    print(f"Trial {trial.number}")
    for k, v in hp.items():
        print(f"  {k}: {v}")
    print(f"{'='*70}")

    trial_seed = SEED + trial.number
    np.random.seed(trial_seed)
    torch.manual_seed(trial_seed)

    try:
        fit_sys = build_model(hp)

        n_chunks = OPTUNA_EPOCHS // OPTUNA_CHUNK_SIZE
        for chunk in range(n_chunks):
            train_model(fit_sys, hp, epochs=OPTUNA_CHUNK_SIZE)

            trial.report(fit_sys.bestfit, chunk)
            if trial.should_prune():
                print(f"  Trial {trial.number} PRUNED at chunk {chunk+1}/{n_chunks}")
                raise optuna.TrialPruned()

    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial {trial.number} FAILED: {e}")
        return float('inf')

    print(f"\nTrial {trial.number} finished: bestfit = {fit_sys.bestfit:.6f}")
    return fit_sys.bestfit


## ═══════════════════════════════════════════════════════════════════════════════
## Main
## ═══════════════════════════════════════════════════════════════════════════════

if USE_OPTUNA:
    import optuna
    from optuna.samplers import TPESampler

    def next_study_name(base_name, directory):
        """Return base_name if no DB exists, else base_name_v2, _v3, etc."""
        version = 1
        while True:
            name = base_name if version == 1 else f"{base_name}_v{version}"
            db = os.path.join(directory, f"optuna_{name}.db")
            if not os.path.exists(db):
                return name
            version += 1

    study_name = next_study_name(OPTUNA_STUDY_NAME, save_dir)
    db_path = os.path.join(save_dir, f"optuna_{study_name}.db")
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
        direction="minimize",
    )

    print(f"\nOptuna study '{study_name}' - {N_OPTUNA_TRIALS} trials")
    print(f"DB: {db_path}\n")

    study.optimize(objective, n_trials=N_OPTUNA_TRIALS)

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"OPTUNA STUDY COMPLETE - {len(study.trials)} trials")
    print(f"{'='*70}")
    print(f"Best trial:  #{study.best_trial.number}")
    print(f"Best value:  {study.best_value:.6f}")
    print(f"Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    print(f"\nAll trials (sorted by value):")
    for t in sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf')):
        status = "OK" if t.value is not None and t.value < float('inf') else "FAIL"
        val_str = f"{t.value:.6f}" if t.value is not None else "None"
        state = t.state.name
        print(f"  #{t.number:3d}  val={val_str}  [{state}]  {t.params}")

    # Save CSV and best params JSON (before retrain, so results survive SLURM timeout)
    df = study.trials_dataframe()
    csv_path = os.path.join(save_dir, f'optuna_trials_{run_id}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved trials CSV: {csv_path}")

    best_params_path = os.path.join(save_dir, f'optuna_best_params_{run_id}.json')
    with open(best_params_path, 'w') as f:
        json.dump(dict(
            best_trial=study.best_trial.number,
            best_value=study.best_value,
            best_params=study.best_params,
        ), f, indent=2)
    print(f"Saved best params: {best_params_path}")

    # ── Retrain best and do full evaluation ─────────────────────────────────
    best_hp = {**study.best_params, 'epochs': 200}
    print(f"\nRetraining best configuration for {best_hp['epochs']} epochs...")
    np.random.seed(SEED + study.best_trial.number)
    torch.manual_seed(SEED + study.best_trial.number)
    fit_sys = build_model(best_hp)
    train_model(fit_sys, best_hp)
    evaluate_and_save(fit_sys, best_hp, f"optuna_best_{run_id}")
    state_recovery_diagnostic(fit_sys, best_hp, f"optuna_best_{run_id}")

else:
    # ── Single run with default hyperparameters ─────────────────────────────
    print(f"\nConfiguration:")
    print(f"  MODE:               {MODE}")
    print(f"  USE_HYBRID_ENCODER: {USE_HYBRID_ENCODER}")
    print(f"  save_dir:           {save_dir}")
    print(f"  NF_SECONDS:         {NF_SECONDS}")
    print(f"  na_nb (samples):    {DEFAULT_HP['na_nb']}  ({DEFAULT_HP['na_nb']/FS_NEW*1000:.2f} ms)")
    print(f"  Sampling rate:      {FS_NEW} Hz (D={D})")
    print(f"  Dtype:              {'float64' if USE_F64 else 'float32'}")
    print(f"  USE_OPTUNA:         {USE_OPTUNA}")
    print(f"  ANN_ACTIVATION:     {ANN_ACTIVATION}")
    print(f"\nHyperparameters:")
    for k, v in DEFAULT_HP.items():
        print(f"  {k}: {v}")

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    fit_sys = build_model(DEFAULT_HP)

    print('\nComputing baseline FP NRMS (fixed reference, no MSD)...')
    baseline_nrms, _ = compute_baseline_fp_nrms(DEFAULT_HP)

    bestfit, diag_conv = train_model_with_diagnostics(fit_sys, DEFAULT_HP)
    print(f"\nTraining complete. Best validation sim-RMS: {bestfit:.6f}")

    evaluate_and_save(fit_sys, DEFAULT_HP, run_id, diag_conv=diag_conv, baseline_nrms=baseline_nrms)
    state_recovery_diagnostic(fit_sys, DEFAULT_HP, run_id)

    # Gradient norm snapshot (after evaluation, non-critical)
    try:
        grad_norms, group_norms = compute_gradient_norms(fit_sys, DEFAULT_HP)
        if save_flag:
            np.savez(os.path.join(save_dir, f'gantry_grad_norms_{run_id}.npz'),
                     grad_norms=json.dumps(grad_norms),
                     group_norms=json.dumps(group_norms))
    except Exception as e:
        print(f"Warning: gradient norm computation failed: {e}")
