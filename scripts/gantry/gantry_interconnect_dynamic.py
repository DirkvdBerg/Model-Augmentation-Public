import os
import sys
import json
import numpy as np
import torch
import deepSI
from scipy.io import loadmat
import time
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from model_augmentation.utils.utils import *
from model_augmentation.utils.torch_nets import zero_init_feed_forward_nn
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
# Encoder: 'linear_map' = Hoekstra 2026 reconstructability init (trainable);
#          'default' = standard deepSI learned encoder
ENCODER_INIT = 'linear_map'
ANN_ACTIVATION = 'linear'  # 'linear' = Identity activation (Jan's ECC setup, D-071); 'tanh' = nonlinear ANN
JOINT_ESTIMATION = True   # D-076: True = trainable damping/stiffness scalars in the physics block
PARAM_RMSE_BASELINE = 0.01 # HEURISTIC: measured initial sqrt-loss, jobs 68675/68676 (D-076 Lambda scale)
# D-076 run design: None = start at true values (run T: measures absorber-induced bias).
# List of 5 factors on [kb_sum, cg1, cg2, cy, cb_sum] = detuned start (run D: recovery test);
# [1.10, 1.10, 0.90, 1.10, 0.90] reproduces the lpv_lfr_baseline detuning signs (D-076).
# NOTE: param_loss anchors to the (possibly detuned) INIT values -- Jan's prior semantics.
PARAM_INIT_DETUNE = None
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

# --- Time-based horizons (converted to samples via TS_NEW) ---
NF_SECONDS = 0.100   # [s] rollout horizon (5*tau_msd, tau=1/(zeta*wn)=20ms, 5tau=100ms)
# THEORY: na=nb=nxd*2+1 (Jan's standard; nxd=NX_PHYS+NX_ANN encoder history)

# --- Default hyperparameters ---
DEFAULT_HP = dict(
    NX_ANN=2,
    n_nodes_per_layer=16,
    n_hidden_layers=2,
    up_sample=2,
    nf=max(1, int(NF_SECONDS / TS_NEW)),
    na_nb=0,          # set below via Jan's formula
    batch_size=256,
    lr=1e-4,
    epochs=5,
)
DEFAULT_HP['na_nb'] = (NX_PHYS + DEFAULT_HP['NX_ANN']) * 2 + 1   # THEORY: nxd*2+1 (Jan's standard)

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

# Test-set (E1) ground truth for the generalization baseline (D-071)
try:
    _, _, test_x_logical, test_x_aug = load_mat_aug(TEST_FILE)
except KeyError as e:
    print(f'WARNING: test file lacks augmented GT fields ({e}); skipping test baseline')
    test_x_logical = None

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

PHY_IX   = np.arange(NX_PHYS)        # [0,1,2,3,4,5]
STIFF_IX = np.array([1, 4, 6, 7])   # Theta pos/vel + absorber pos/vel (K>0 only)

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
        na = hp.get('na_nb', 2 * (NX_PHYS + hp['NX_ANN']) + 1)  # THEORY: nxd*2+1 (Jan's standard)
        nb = na
        na_right = 1   # reconstructability map uses y(k), so window is [k-na, k]
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

    # Encoder history: na=nb=nxd*2+1 (Jan's standard, nxd=NX_PHYS+NX_ANN).
    # For linear_map, na_right=1 so reconstructability map can use y(k) to compute x(k).
    na, nb, na_right, nb_right = _get_encoder_dims(hp)

    ic = Interconnect(nxd, nu, ny, debugging=False)

    if JOINT_ESTIMATION:
        # D-076: trainable [kb_sum, cg1, cg2, cy, cb_sum] via log-reparam
        _pi = None
        if PARAM_INIT_DETUNE is not None:
            from model_augmentation.systems.gantry_ss import kb1, kb2, cg1, cg2, cy, cb1, cb2
            _nominal = np.array([float(kb1 + kb2), float(cg1), float(cg2),
                                 float(cy), float(cb1 + cb2)])
            _pi = _nominal * np.asarray(PARAM_INIT_DETUNE, dtype=float)
        phy_block = Parameterized_Gantry_State_Block(
            Y_op=None, std_x=std_x, std_u=std_u,
            x_mean=x_mean, u_mean=u_mean, Ts=TS_NEW,
            up_sample=hp['up_sample'],
            RMSE_baseline=PARAM_RMSE_BASELINE,
            params_init=_pi,
        ).to(DTYPE_PT)
    else:
        phy_block = Gantry_State_Block(
            Y_op=None, std_x=std_x, std_u=std_u,
            x_mean=x_mean, u_mean=u_mean, Ts=TS_NEW,
            up_sample=hp['up_sample'],
        ).to(DTYPE_PT)
    # CHANGED: C_aug / Parameterized_Linear_Output_Block removed — diag_gradient_routing
    # confirms near-zero C_aug init (Frobenius ~1e-2) chokes ANN gradient by same factor.
    out_phys = Linear_Output_Block(C=Cd_norm, D=Dd_np)
    ic.add_block(phy_block)
    ic.add_block(out_phys)

    _act = torch.nn.Identity if ANN_ACTIVATION == 'linear' else torch.nn.Tanh
    ann_block = Static_ANN_Block(
        nz=nxd + nu, nw=len(STIFF_IX),  # D-068: stiff routing — 4 outputs at K>0 rows only
        n_nodes_per_layer=hp['n_nodes_per_layer'],
        n_hidden_layers=hp['n_hidden_layers'],
        net=zero_init_feed_forward_nn,
        activation=_act,
    )
    ic.add_block(ann_block)

    # D-068: stiff routing — corrections placed only at K>0 rows (Theta + absorber).
    # X and Y axis rows (0,3,2,5) excluded: K=0 integrators accumulate without restoring force.
    ic.connect_block_signals(ann_block, ["x", "u"], [])
    ic.connect_signals(ann_block, "xp", "additive", expansion_matrix(STIFF_IX, nxd))
    ic.connect_signals("x", phy_block, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(phy_block, ["u"], [])
    ic.connect_signals(phy_block, "xp", "additive", expansion_matrix(PHY_IX, nxd))
    # Physical: x_phys + u -> y (additive)
    ic.connect_signals("x", out_phys, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(out_phys, ["u"], ["y"])

    # D-076: ParamLoss subclass used unconditionally — exact no-op when no
    # block exposes param_loss (i.e. identical to SSE_Interconnect for
    # JOINT_ESTIMATION=False).
    fit_sys = SSE_Interconnect_ParamLoss(
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

    fit_sys.init_model(sys_data=train_data, auto_fit_norm=False)
    if ENCODER_INIT == 'linear_map':
        fit_sys.hfn.to(DTYPE_PT)
    else:
        for net in (fit_sys.encoder, fit_sys.hfn):
            net.to(DTYPE_PT)

    return fit_sys


def train_model(fit_sys, hp, epochs=None, nf=None, validation_measure=None):
    """Train fit_sys for given epochs. nf and validation_measure override hp defaults."""
    fit_sys.fit(
        train_sys_data=train_data, val_sys_data=val_data,
        batch_size=hp['batch_size'], epochs=epochs or hp['epochs'],
        auto_fit_norm=False,
        loss_kwargs={'nf': nf if nf is not None else hp['nf']},
        optimizer_kwargs={'lr': hp['lr']},
        validation_measure=validation_measure if validation_measure is not None else 'sim-RMS',
    )
    return fit_sys.bestfit


## ══════════════════════════════════════════════════════════════════════════════
## evaluate_and_save
## ═══════════════════════════════════════════════════════════════════════════════

def r2_per_channel(ref, est):
    """Per-channel coefficient of determination between reference and estimate."""
    # THEORY: R^2 = 1 - SS_res/SS_tot (standard OLS); 1e-12 guards constant channels
    ss_res = ((ref - est) ** 2).sum(axis=0)
    ss_tot = ((ref - ref.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


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


def evaluate_and_save(fit_sys, hp, rid, diag_conv=None, baseline_nrms=None,
                      baseline_test_nrms=None, baseline_encinit_nrms=None,
                      baseline_test_encinit_nrms=None):
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

    # ── Joint estimation report (D-076, present only when the param block is used)
    _pblocks = [m for m in fit_sys.hfn.connected_blocks if hasattr(m, 'physical_params')]
    params_init_np = None
    params_learned_np = None
    if _pblocks:
        _pb = _pblocks[0]
        params_init_np = _pb.params_init.detach().cpu().numpy().copy()
        params_learned_np = np.array([_pb.physical_params()[n] for n in _pb.PARAM_NAMES])
        print('\n=== Joint estimation: physical parameters (best checkpoint) ===')
        print(f"  {'param':8s} {'init':>12s} {'learned':>12s} {'delta':>9s}")
        for _i, _n in enumerate(_pb.PARAM_NAMES):
            _d = 100.0 * (params_learned_np[_i] - params_init_np[_i]) / params_init_np[_i]
            print(f'  {_n:8s} {params_init_np[_i]:12.4f} {params_learned_np[_i]:12.4f} {_d:+8.2f}%')

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
    rms_enc  = nrms_enc * ystd   # [m]
    if baseline_nrms is not None:
        rms_baseline = baseline_nrms * ystd   # [m]
        print('\n=== Sim-NRMS + RMS: augmented vs baseline FP ===')
        print(f"  {'':4s}  {'augmented':>22s}  {'baseline FP':>22s}  {'improve':>8s}")
        print(f"  {'':4s}  {'(NRMS':>11s} {'RMS [m])':>11s}  {'(NRMS':>11s} {'RMS [m])':>11s}")
        for ch, lbl in enumerate(['X1', 'X2', 'Y ']):
            improv = 100.0 * (baseline_nrms[ch] - nrms_enc[ch]) / (baseline_nrms[ch] + 1e-12)
            print(f"  {lbl}:  {nrms_enc[ch]:.4f}  {rms_enc[ch]:.3e} m"
                  f"    {baseline_nrms[ch]:.4f}  {rms_baseline[ch]:.3e} m"
                  f"    {improv:+.1f}%")
    else:
        print('\n=== Encoder-initialised sim-NRMS ===')
        for ch, lbl in enumerate(['X1', 'X2', 'Y ']):
            print(f'  {lbl}:  {nrms_enc[ch]:.4f}  {rms_enc[ch]:.3e} m')

    ann_rms_enc = np.sqrt((x_enc_ann[cheat_n:] ** 2).mean(axis=0))
    print('\n=== ANN latent state RMS ===')
    for ch in range(NX_ANN):
        print(f'  x[{NX_PHYS+ch}]: enc={ann_rms_enc[ch]:.4e}')

    # ── Rollout aug-state R2 vs GT absorber (closed-loop simulation) ─────────
    # Complements the encoder-based R2 (D-053/aug_state_r2): this tests whether
    # the MODEL's simulated x[6:7] trajectories track delta_a, i.e. whether the
    # augmented states became the absorber in rollout. Activation-agnostic.
    r2_roll_raw, r2_roll_lin = None, None
    try:
        x_ann_roll = x_enc_ann[cheat_n:]                    # (N-cheat_n, NX_ANN) simulated
        gt_roll    = val_x_aug[cheat_n:]                    # (N-cheat_n, NX_ANN) physical
        gt_norm = (gt_roll - gt_roll.mean(axis=0)) / (gt_roll.std(axis=0) + 1e-8)

        r2_roll_raw = r2_per_channel(gt_norm, x_ann_roll)
        # THEORY: ordinary least squares — best affine map from all rollout aug channels
        A_roll = np.hstack([x_ann_roll, np.ones((len(x_ann_roll), 1), dtype=DTYPE_NP)])
        W_roll, *_ = np.linalg.lstsq(A_roll, gt_norm, rcond=None)
        r2_roll_lin = r2_per_channel(gt_norm, A_roll @ W_roll)

        aug_names = ['delta_a ', 'vdelta_a']
        print('\n=== Rollout aug-state R2 vs GT (closed-loop simulation) ===')
        for ch in range(NX_ANN):
            lbl = aug_names[ch] if ch < len(aug_names) else f'x_ann[{ch}]'
            print(f'  {lbl}  R2_raw={r2_roll_raw[ch]:+.4f}  R2_linmap={r2_roll_lin[ch]:+.4f}')
        print('  R2_linmap ~ 1 -> simulated aug states carry the absorber dynamics;')
        print('  R2_linmap ~ 0 -> aug states unused in rollout (encoder R2 tests encoder only)')
    except Exception as e:
        print(f'Warning: rollout aug-state R2 failed: {e}')

    # ── Test-set simulation (E1, unseen excitation) — generalization (D-071) ─
    nrms_test  = None
    y_hat_test = None
    try:
        fit_sys.hfn.reset_saved_signals()
        test_result = fit_sys.apply_experiment(test_data)
        y_hat_test  = test_result.y
        tc = test_result.cheat_n
        nrms_test = np.sqrt(((y_hat_test[tc:] - test_data.y[tc:]) ** 2).mean(axis=0)) / ystd
        print('\n=== Test-set (E1) sim-NRMS + RMS — generalization to unseen excitation ===')
        if baseline_test_nrms is not None:
            print(f"  {'':4s}  {'augmented':>22s}  {'baseline FP':>22s}  {'improve':>8s}")
            print(f"  {'':4s}  {'(NRMS':>11s} {'RMS [m])':>11s}  {'(NRMS':>11s} {'RMS [m])':>11s}")
            for ch, lbl in enumerate(['X1', 'X2', 'Y ']):
                improv = 100.0 * (baseline_test_nrms[ch] - nrms_test[ch]) / (baseline_test_nrms[ch] + 1e-12)
                print(f'  {lbl}:  {nrms_test[ch]:.4f}  {nrms_test[ch]*ystd[ch]:.3e} m'
                      f'    {baseline_test_nrms[ch]:.4f}  {baseline_test_nrms[ch]*ystd[ch]:.3e} m'
                      f'    {improv:+.1f}%')
        else:
            for ch, lbl in enumerate(['X1', 'X2', 'Y ']):
                print(f'  {lbl}:  {nrms_test[ch]:.4f}  {nrms_test[ch]*ystd[ch]:.3e} m')
    except Exception as e:
        print(f'Warning: test-set (E1) evaluation failed: {e}')

    # ── x_logical-initialised simulation (model from TRUE x0, D-072) ────────
    # Symmetric counterpart to the true-x0 baseline: isolates model quality from
    # encoder quality. Seeded from val_x_logical (val_data.x is never set by load_traj).
    if val_x_logical is not None:
        val_norm = fit_sys.norm.transform(val_data)
        u_val_norm = torch.tensor(np.ascontiguousarray(val_norm.u), dtype=DTYPE_PT)

        x_xlog = torch.zeros(1, nxd)
        x_xlog[0, :NX_PHYS] = torch.tensor(
            (val_x_logical[0] - x_mean.flatten()) / std_x.flatten(), dtype=DTYPE_PT)

        y_xlog_list = []
        with torch.no_grad():
            for t in range(len(u_val_norm)):
                y_t, x_xlog = fit_sys.hfn(x_xlog, u_val_norm[t:t+1])
                y_xlog_list.append(y_t.squeeze().numpy())
        y_hat_xlog = np.array(y_xlog_list) * ystd + y0

        # Averaged from cheat_n like the encoder-init model metric (D-072 alignment)
        nrms_xlog = np.sqrt(((y_hat_xlog[cheat_n:] - y_ref[cheat_n:]) ** 2).mean(axis=0)) / ystd
        print('\n=== Model, true-x0 init: sim-NRMS + RMS ===')
        for ch, lbl in enumerate(['X1', 'X2', 'Y ']):
            print(f'  {lbl}: {nrms_xlog[ch]:.4f}  {nrms_xlog[ch] * ystd[ch]:.3e} m')
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
                label=f'Encoder-init (NRMS={nrms_enc[ch]:.3f}, RMS={rms_enc[ch]:.2e} m)')
        if HAS_ORACLE:
            ax.plot(t_val, y_hat_xlog[:, ch], 'C1', lw=0.9, linestyle='--',
                    label=f'x_logical-init (NRMS={nrms_xlog[ch]:.3f}, RMS={nrms_xlog[ch]*ystd[ch]:.2e} m)')
        enc_lbl = f'Encoder warmup ({cheat_n} samples)' if ch == 0 else '_nolegend_'
        ax.axvspan(t_val[0], cheat_t, alpha=0.10, color='steelblue', label=enc_lbl)
        ax.axvline(cheat_t, color='steelblue', linestyle='--', lw=0.8)
        ax.set_ylabel(lab); ax.legend(fontsize=7, loc='upper right'); ax.grid(True)
    axes2[-1].set_xlabel('Time [s]')
    fig2.suptitle(f'Validation simulation - dynamic parallel (NX_ANN={NX_ANN})')
    fig2.tight_layout()
    fig2.savefig(os.path.join(save_dir, f'gantry_simulation_{rid}.png'), dpi=150)

    # Plot 3: ANN latent state trajectories vs ground-truth absorber states
    aug_gt_labels  = ['delta_a [m]', 'vdelta_a [m/s]']
    aug_gt_names   = ['delta_a', 'vdelta_a']
    if NX_ANN == 1:
        fig3, axes3 = plt.subplots(1, 1, figsize=(12, 3), sharex=True)
        axes3 = [axes3]
    else:
        fig3, axes3 = plt.subplots(NX_ANN, 1, figsize=(12, 4), sharex=True)
    for ch, ax in enumerate(axes3):
        ax.plot(t_val, x_enc_ann[:, ch], 'C0', lw=0.8,
                label=f'x_ann[{NX_PHYS+ch}] (RMS={ann_rms_enc[ch]:.2e})')
        ax.axvspan(t_val[0], cheat_t, alpha=0.10, color='steelblue')
        ax.axvline(cheat_t, color='steelblue', linestyle='--', lw=0.8)
        ax.set_ylabel(f'x[{NX_PHYS+ch}] (dim-less)', color='C0')
        ax.tick_params(axis='y', labelcolor='C0')
        if ch < len(aug_gt_labels):
            ax2 = ax.twinx()
            ax2.plot(t_val, val_x_aug[:, ch], 'C1', lw=0.8, alpha=0.7,
                     label=f'GT {aug_gt_names[ch]}')
            ax2.set_ylabel(aug_gt_labels[ch], color='C1')
            ax2.tick_params(axis='y', labelcolor='C1')
            lines1, labs1 = ax.get_legend_handles_labels()
            lines2, labs2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labs1 + labs2, fontsize=7, loc='upper right')
        else:
            ax.legend(fontsize=7)
        ax.grid(True)
    axes3[-1].set_xlabel('Time [s]')
    fig3.suptitle(f'ANN latent states x[{NX_PHYS}:{nxd}] vs GT absorber (dimensionless vs physical)')
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
            # Aggregate sim-RMS [m]: same formula as the plotted validation loss,
            # so the reference line and the curve share units (NRMS lines were not comparable).
            rms_agg_baseline = float(np.sqrt(np.mean((baseline_nrms * ystd) ** 2)))
            ax4a.axhline(rms_agg_baseline, color='C2', linestyle=':', lw=1.2,
                         label=f'Baseline FP sim-RMS={rms_agg_baseline:.2e} m')
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
            # Per-channel metrics (rms_* = nrms_* x ystd, [m])
            nrms_enc=nrms_enc, rms_enc=rms_enc,
            x_enc_phys=x_enc_phys, x_enc_ann=x_enc_ann,
            val_x_aug=val_x_aug,
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
                MODE=MODE, ENCODER_INIT=ENCODER_INIT,
                ANN_ACTIVATION=ANN_ACTIVATION, FS_NEW=FS_NEW, D=D,
                FS_ORIG=FS_ORIG, SEED=SEED,
                JOINT_ESTIMATION=JOINT_ESTIMATION,
                PARAM_RMSE_BASELINE=PARAM_RMSE_BASELINE,
                PARAM_INIT_DETUNE=PARAM_INIT_DETUNE,
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
            save_dict['baseline_rms']  = baseline_nrms * ystd
        if nrms_test is not None:
            save_dict['nrms_test']  = nrms_test
            save_dict['rms_test']   = nrms_test * ystd
            save_dict['y_hat_test'] = y_hat_test
            save_dict['y_test_ref'] = test_data.y
        if r2_roll_lin is not None:
            save_dict['r2_rollout_raw']    = r2_roll_raw
            save_dict['r2_rollout_linmap'] = r2_roll_lin
        if baseline_test_nrms is not None:
            save_dict['baseline_test_nrms'] = baseline_test_nrms
            save_dict['baseline_test_rms']  = baseline_test_nrms * ystd
        if baseline_encinit_nrms is not None:
            save_dict['baseline_encinit_nrms'] = baseline_encinit_nrms
            save_dict['baseline_encinit_rms']  = baseline_encinit_nrms * ystd
        if baseline_test_encinit_nrms is not None:
            save_dict['baseline_test_encinit_nrms'] = baseline_test_encinit_nrms
            save_dict['baseline_test_encinit_rms']  = baseline_test_encinit_nrms * ystd
        if params_learned_np is not None:
            save_dict['params_init']    = params_init_np
            save_dict['params_learned'] = params_learned_np
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

    r2_raw = r2_per_channel(xt,   x_hat[:, :NX_PHYS])
    r2_lag = r2_per_channel(xt_l, x_hat[:, :NX_PHYS])

    # Best affine map x_true ~ [x_hat, 1] @ W — THEORY: ordinary least squares
    A = np.hstack([x_hat, np.ones((len(x_hat), 1), dtype=DTYPE_NP)])
    W, *_ = np.linalg.lstsq(A, xt, rcond=None)
    r2_lin = r2_per_channel(xt, A @ W)

    labels = ['q1 ', 'q2 ', 'q3 ', 'dq1', 'dq2', 'dq3']
    print('\n=== State recovery diagnostic (D-053) ===')
    print(f'  {len(k_ix)} windows (stride {stride}), na=nb={na}, '
          f'encoder={ENCODER_INIT}')
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

def compute_baseline_fp_nrms(hp, data=None, x0_phys=None, x0_norm=None,
                             start_ix=0, avg_from=0, label='val'):
    """Simulate baseline FP model (no MSD, no ANN) on val data (default) or given data.

    Initialization (D-072):
      x0_phys  physical initial state (default: true x0 from val_x_logical) — 'true x0'
      x0_norm  normalized initial state (overrides x0_phys) — for encoder-init baselines
      start_ix simulate from data sample start_ix onward (encoder init estimates x(k0))
      avg_from average the error only from this sample of the simulated window
               (aligns with the model metric, which excludes the encoder warm-up)

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
    if data is None:
        data = val_data

    phy_block = Gantry_State_Block(
        Y_op=None, std_x=std_x, std_u=std_u,
        x_mean=x_mean, u_mean=u_mean, Ts=TS_NEW,
        up_sample=hp['up_sample'],
    ).to(DTYPE_PT)
    phy_block.eval()

    # Initial state: normalized estimate (encoder-init) or physical true x0 (D-072)
    if x0_norm is not None:
        x_norm_np = np.asarray(x0_norm, dtype=DTYPE_NP).flatten()  # (NX_PHYS,)
    else:
        if x0_phys is None:
            x0_phys = val_x_logical[0]
        x0_phys   = np.asarray(x0_phys, dtype=DTYPE_NP)
        x_norm_np = (x0_phys - x_mean.flatten()) / std_x.flatten()  # (NX_PHYS,)

    u_val_norm = ((data.u[start_ix:] - u_mean.flatten()) / std_u.flatten()).astype(DTYPE_NP)

    y_hat_list = []
    with torch.no_grad():
        for t in range(len(u_val_norm)):
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

    y_hat = np.array(y_hat_list, dtype=DTYPE_NP)   # (N-start_ix, ny)
    y_ref  = data.y[start_ix:]
    nrms   = np.sqrt(((y_hat[avg_from:] - y_ref[avg_from:])**2).mean(axis=0)) / ystd

    rms = nrms * ystd   # [m]
    # THEORY: deepSI System_data.RMS — sqrt(mean sq. error over all samples AND channels)
    rms_agg = float(np.sqrt(np.mean(rms ** 2)))
    print(f'\n=== Baseline FP model ({label}, no MSD, reference to beat) ===')
    print(f"  {'':4s}  {'NRMS':>8s}  {'RMS':>11s}")
    for ch, lbl in enumerate(['X1', 'X2', 'Y ']):
        print(f'  {lbl}:  {nrms[ch]:8.4f}  {rms[ch]:.3e} m')
    print(f'  aggregate sim-RMS (same formula as validation loss): {rms_agg:.4e} m')

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

    r2_raw = r2_per_channel(gt_norm, x_ann)

    # Best affine map: gt_norm ~ [x_ann_all_channels, 1] @ W (per GT channel)
    # THEORY: ordinary least squares -- uses all NX_ANN encoder channels jointly
    A_aug = np.hstack([x_ann, np.ones((len(x_ann), 1), dtype=DTYPE_NP)])
    W_aug, *_ = np.linalg.lstsq(A_aug, gt_norm, rcond=None)
    r2_lin = r2_per_channel(gt_norm, A_aug @ W_aug)

    return r2_raw, r2_lin


## ═══════════════════════════════════════════════════════════════════════════════
## train_model_with_diagnostics
## ═══════════════════════════════════════════════════════════════════════════════

def train_model_with_diagnostics(fit_sys, hp, resume_ckpt=None, checkpoint_dir=None):
    """Train for hp['epochs'] epochs with sim-RMS validation; record aug-state R2 after.

    resume_ckpt : base path (no extension) of a prior checkpoint to resume from.
    checkpoint_dir : directory for checkpoint; None = no saving.
    """
    diag_epochs, diag_r2_raw, diag_r2_lin = [], [], []
    done_epochs = 0
    bestfit = float('inf')

    # --- Resume ---
    if resume_ckpt is not None:
        meta = np.load(resume_ckpt + '.npz', allow_pickle=True)
        # D-070: SSE_Interconnect has no state_dict; checkpoint holds component state_dicts
        ckpt = torch.load(resume_ckpt + '.pt', map_location='cpu')
        # D-076: JE checkpoints carry log_params; pre-JE checkpoints cannot resume a JE run
        if JOINT_ESTIMATION and not any('log_params' in k for k in ckpt['hfn']):
            raise RuntimeError(
                'RESUME_CHECKPOINT points at a pre-JE checkpoint (no log_params); '
                'JOINT_ESTIMATION runs must start from fresh checkpoints (D-076)')
        fit_sys.hfn.load_state_dict(ckpt['hfn'])
        fit_sys.encoder.load_state_dict(ckpt['encoder'])
        if 'optimizer' in ckpt:
            fit_sys.optimizer.load_state_dict(ckpt['optimizer'])
        done_epochs = int(meta['done_epochs'])
        bestfit     = float(meta['bestfit'])
        diag_epochs = list(meta['diag_epochs'])
        diag_r2_raw = [meta['diag_r2_raw'][i]  for i in range(len(meta['diag_r2_raw']))]
        diag_r2_lin = [meta['diag_r2_linmap'][i] for i in range(len(meta['diag_r2_linmap']))]
        print(f'Resumed from {resume_ckpt}  ({done_epochs} epochs done)')

    epochs_remaining = hp['epochs'] - done_epochs
    print(f'\nTraining: nf={hp["nf"]}  epochs={hp["epochs"]}  val=sim-RMS  NX_ANN={hp["NX_ANN"]}')
    if done_epochs > 0:
        print(f'  Resuming: {done_epochs} done, {epochs_remaining} remaining')

    fit_sys.bestfit = float('inf')
    t0 = time.time()
    bestfit = train_model(fit_sys, hp,
                          epochs=epochs_remaining,
                          nf=hp['nf'],
                          validation_measure='sim-RMS')
    elapsed = time.time() - t0
    done_epochs = hp['epochs']

    # --- Checkpoint weights first: diagnostics below must not be able to lose them (D-070)
    ckpt_base = None
    if checkpoint_dir is not None:
        ckpt_base = os.path.join(checkpoint_dir, f'gantry_ckpt_{run_id}')
        # SSE_Interconnect is a deepSI System (no state_dict); save its torch components
        torch.save({
            'hfn':       fit_sys.hfn.state_dict(),
            'encoder':   fit_sys.encoder.state_dict(),
            'optimizer': fit_sys.optimizer.state_dict(),
        }, ckpt_base + '.pt')
        print(f'  Checkpoint weights: {ckpt_base}.pt')

    fit_sys.eval()
    try:
        r2_raw, r2_lin = aug_state_r2(fit_sys, hp)
    except Exception as e:
        print(f'Warning: aug_state_r2 failed ({e}); recording NaN')
        r2_raw = np.full(hp['NX_ANN'], np.nan)
        r2_lin = np.full(hp['NX_ANN'], np.nan)
    diag_epochs.append(done_epochs)
    diag_r2_raw.append(r2_raw.copy())
    diag_r2_lin.append(r2_lin.copy())

    r2_str = '  '.join([f'{n}={r2_lin[i]:+.4f}'
                        for i, n in enumerate(['delta_a', 'vdelta_a'][:hp['NX_ANN']])])
    print(f'Training done | {done_epochs} ep | {elapsed:.0f}s | '
          f'bestfit={bestfit:.5f}  R2_linmap: {r2_str}')

    # --- Checkpoint metadata (weights already saved above) ---
    if ckpt_base is not None:
        np.savez(ckpt_base + '.npz',
                 done_epochs    = np.array(done_epochs),
                 bestfit        = np.array(bestfit),
                 elapsed        = np.array(elapsed),
                 diag_epochs    = np.array(diag_epochs),
                 diag_r2_raw    = np.array(diag_r2_raw),
                 diag_r2_linmap = np.array(diag_r2_lin),
                 hp             = np.array(json.dumps(hp)),
                 orig_run_id    = np.array(run_id),
        )
        print(f'  Checkpoint meta: {ckpt_base}.npz')

    return bestfit, dict(
        epochs    = np.array(diag_epochs),
        r2_raw    = np.array(diag_r2_raw),
        r2_linmap = np.array(diag_r2_lin),
    )


## ═══════════════════════════════════════════════════════════════════════════════
## Main
## ═══════════════════════════════════════════════════════════════════════════════

print(f"\nConfiguration:")
print(f"  MODE:           {MODE}")
print(f"  ENCODER_INIT:   {ENCODER_INIT}")
print(f"  ANN_ACTIVATION: {ANN_ACTIVATION}")
print(f"  save_dir:       {save_dir}")
print(f"  NF_SECONDS:     {NF_SECONDS}")
print(f"  na_nb (samples): {DEFAULT_HP['na_nb']}  ({DEFAULT_HP['na_nb']/FS_NEW*1000:.2f} ms)")
print(f"  Sampling rate:  {FS_NEW} Hz (D={D})")
print(f"  Dtype:          {'float64' if USE_F64 else 'float32'}")
print(f"\nDefault hyperparameters (may be overridden by checkpoint):")
for k, v in DEFAULT_HP.items():
    print(f"  {k}: {v}")

resume_ckpt = os.environ.get('RESUME_CHECKPOINT')  # e.g. /path/to/gantry_ckpt_68458_stage3
if resume_ckpt:
    _meta = np.load(resume_ckpt + '.npz', allow_pickle=True)
    hp = json.loads(str(_meta['hp']))
    print(f'\nResuming checkpoint: {resume_ckpt}')
    print(f'  Saved hp: {hp}')
else:
    hp = DEFAULT_HP

np.random.seed(SEED)
torch.manual_seed(SEED)
fit_sys = build_model(hp)

print('\nComputing baseline FP RMS/NRMS (fixed reference, no MSD)...')
_na, _nb, _na_right, _nb_right = _get_encoder_dims(hp)
K0 = max(_na, _nb)   # first sample with a full encoder window (~ model cheat_n)


def _encoder_init_state(data):
    """Untrained linear-map encoder estimate of x(K0) from the first I/O window (D-072).

    Runs BEFORE training, so fit_sys.encoder is still the pure reconstructability
    map built from the baseline linearization — a baseline-only quantity.
    """
    dn = fit_sys.norm.transform(data)
    yp = np.ascontiguousarray(dn.y, dtype=DTYPE_NP)[K0 - _na: K0 + _na_right][None]
    up = np.ascontiguousarray(dn.u, dtype=DTYPE_NP)[K0 - _nb: K0 + _nb_right][None]
    with torch.no_grad():
        x0 = fit_sys.encoder(torch.tensor(up, dtype=DTYPE_PT),
                             torch.tensor(yp, dtype=DTYPE_PT)).numpy()[0]
    return x0[:NX_PHYS]


# True-x0 (oracle) baselines — error averaged from K0 to match the model metric (D-072)
baseline_nrms, _ = compute_baseline_fp_nrms(hp, avg_from=K0, label='val, true x0')
if test_x_logical is not None:
    baseline_test_nrms, _ = compute_baseline_fp_nrms(
        hp, data=test_data, x0_phys=test_x_logical[0], avg_from=K0,
        label='test E1, true x0')
else:
    baseline_test_nrms = None

# Encoder-init baselines — same init information as the model, no oracle (D-072).
# Only meaningful for the linear_map encoder (untrained = reconstructability map).
if ENCODER_INIT == 'linear_map':
    baseline_encinit_nrms, _ = compute_baseline_fp_nrms(
        hp, x0_norm=_encoder_init_state(val_data), start_ix=K0,
        label='val, encoder-init (untrained linear map)')
    baseline_test_encinit_nrms, _ = compute_baseline_fp_nrms(
        hp, data=test_data, x0_norm=_encoder_init_state(test_data), start_ix=K0,
        label='test E1, encoder-init (untrained linear map)')
else:
    baseline_encinit_nrms = None
    baseline_test_encinit_nrms = None

ckpt_dir = save_dir if save_flag else None
bestfit, diag_conv = train_model_with_diagnostics(
    fit_sys, hp, resume_ckpt=resume_ckpt, checkpoint_dir=ckpt_dir)
print(f"\nTraining complete. Best validation sim-RMS: {bestfit:.6f}")

evaluate_and_save(fit_sys, hp, run_id, diag_conv=diag_conv, baseline_nrms=baseline_nrms,
                  baseline_test_nrms=baseline_test_nrms,
                  baseline_encinit_nrms=baseline_encinit_nrms,
                  baseline_test_encinit_nrms=baseline_test_encinit_nrms)
state_recovery_diagnostic(fit_sys, hp, run_id)

# Gradient norm snapshot (after evaluation, non-critical)
try:
    grad_norms, group_norms = compute_gradient_norms(fit_sys, hp)
    if save_flag:
        np.savez(os.path.join(save_dir, f'gantry_grad_norms_{run_id}.npz'),
                 grad_norms=json.dumps(grad_norms),
                 group_norms=json.dumps(group_norms))
except Exception as e:
    print(f"Warning: gradient norm computation failed: {e}")
