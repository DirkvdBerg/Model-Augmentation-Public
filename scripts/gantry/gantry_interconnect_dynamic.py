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
from model_augmentation.utils.torch_nets import zero_init_feed_forward_nn
from model_augmentation.fit_systems.interconnect import *
from model_augmentation.fit_systems.blocks import *
from model_augmentation.systems.gantry_ss import Cd, Dd, P

## ═══════════════════════════════════════════════════════════════════════════════
## Configuration
## ═══════════════════════════════════════════════════════════════════════════════

# --- Fixed model constants ---
NX_PHYS = 6   # physical states: q1, q2, q3, dq1, dq2, dq3
nu  = 3
ny  = 3
Y_OP = None   # None = LPV self-scheduled; float = frozen operating point [m]
SEED = 42

# --- Resampling ---
FS_ORIG = 20000
FS_NEW  = 1000          # 1 kHz - Nyquist safe for 150 Hz MSD resonance
D       = FS_ORIG // FS_NEW   # = 20
TS_NEW  = 1.0 / FS_NEW        # = 0.001 s

# --- Dtype ---
USE_F64  = False
DTYPE_NP = np.float64    if USE_F64 else np.float32
DTYPE_PT = torch.float64 if USE_F64 else torch.float32

# --- Utility ---
save_flag = True
run_id = os.environ.get('SLURM_JOB_ID') or datetime.now().strftime('%Y%m%d_%H%M%S')

# --- Optuna hyperparameter search ---
USE_OPTUNA = True
N_OPTUNA_TRIALS = 40
OPTUNA_STUDY_NAME = "gantry_subnet_augmented"

# --- Default hyperparameters (used when USE_OPTUNA=False) ---
DEFAULT_HP = dict(
    NX_ANN=4,
    n_nodes_per_layer=128,
    n_hidden_layers=3,
    nf=350,
    batch_size=4000,
    lr=2e-4,
    epochs=200,
)

## ═══════════════════════════════════════════════════════════════════════════════
## Data loading (run once)
## ═══════════════════════════════════════════════════════════════════════════════

np.random.seed(SEED)
torch.manual_seed(SEED)

TRAJ_DIR = os.path.join(os.path.dirname(__file__), '..', '..',
                        'data', 'gantry', 'matlab', 'trajectories')
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
        u=d['u'][::D].astype(DTYPE_NP),
        y=d['y'][::D].astype(DTYPE_NP),
        dt=TS_NEW,
    )

train_list = [load_traj(f) for f in TRAIN_FILES]
train_data = deepSI.System_data_list(train_list)
val_data   = load_traj(VAL_FILE)
test_data  = load_traj(TEST_FILE)

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
y0     = (Cd.numpy() @ x_mean.flatten()).astype(DTYPE_NP)

# Cd_norm[i,j] = Cd[i,j] * std_x[j] / ystd[i]
Cd_norm = Cd.numpy() * std_x.flatten()[None, :] / ystd[:, None]  # (3, 6)
Dd_np   = Dd.numpy()                                               # (3, 3)

PHY_IX = np.arange(NX_PHYS)   # [0,1,2,3,4,5]

save_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'simulations', 'gantry_subnet')
os.makedirs(save_dir, exist_ok=True)


## ═══════════════════════════════════════════════════════════════════════════════
## build_and_train
## ═══════════════════════════════════════════════════════════════════════════════

def build_and_train(hp):
    """Build interconnect from hp dict, train, return (fit_sys, bestfit)."""
    NX_ANN = hp['NX_ANN']
    nxd = NX_PHYS + NX_ANN
    na = 2 * nxd + 1
    nb = 2 * nxd + 1

    ic = Interconnect(nxd, nu, ny, debugging=False)

    phy_block = Gantry_State_Block(
        Y_op=Y_OP, std_x=std_x, std_u=std_u,
        x_mean=x_mean, u_mean=u_mean, Ts=TS_NEW,
    ).to(DTYPE_PT)
    out_block = Linear_Output_Block(C=Cd_norm, D=Dd_np)
    ic.add_block(phy_block)
    ic.add_block(out_block)

    ann_block = Static_ANN_Block(
        nz=nxd + nu, nw=nxd,
        n_nodes_per_layer=hp['n_nodes_per_layer'],
        n_hidden_layers=hp['n_hidden_layers'],
        net=zero_init_feed_forward_nn,
        activation=torch.nn.Tanh,
    )
    ic.add_block(ann_block)

    ic.connect_block_signals(ann_block, ["x", "u"], ["xp"])
    ic.connect_signals("x", phy_block, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(phy_block, ["u"], [])
    ic.connect_signals(phy_block, "xp", "additive", expansion_matrix(PHY_IX, nxd))
    ic.connect_signals("x", out_block, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(out_block, ["u"], ["y"])

    fit_sys = SSE_Interconnect(
        interconnect=ic, na=na, nb=nb,
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

    fit_sys.fit(
        train_sys_data=train_data, val_sys_data=val_data,
        batch_size=hp['batch_size'], epochs=hp['epochs'],
        auto_fit_norm=False,
        loss_kwargs={'nf': hp['nf']},
        optimizer_kwargs={'lr': hp['lr']},
        validation_measure="sim-RMS",
    )

    return fit_sys, fit_sys.bestfit


## ═══════════════════════════════════════════════════════════════════════════════
## evaluate_and_save
## ═══════════════════════════════════════════════════════════════════════════════

def evaluate_and_save(fit_sys, hp, rid):
    """Load best checkpoint, simulate, compute NRMS, plot, save."""
    NX_ANN = hp['NX_ANN']
    nxd = NX_PHYS + NX_ANN
    na = 2 * nxd + 1
    nb = 2 * nxd + 1

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

    plt.close('all')

    # ── Results npz ─────────────────────────────────────────────────────────
    if save_flag:
        save_dict = dict(
            y_ref=y_ref, y_hat_enc=y_hat_enc, t_val=t_val,
            epoch_id=epoch_id_full, loss_val=loss_val_full, loss_train=loss_train_full,
            nrms_enc=nrms_enc, x_enc_phys=x_enc_phys, x_enc_ann=x_enc_ann,
            cheat_n=np.array(cheat_n), dt=np.array(val_data.dt),
            na=np.array(na), nb=np.array(nb), nf=np.array(hp['nf']),
            NX_PHYS=np.array(NX_PHYS), NX_ANN=np.array(NX_ANN), nxd=np.array(nxd),
            hp=json.dumps(hp),
        )
        if HAS_ORACLE:
            save_dict['y_hat_xlog'] = y_hat_xlog
            save_dict['nrms_xlog'] = nrms_xlog
        np.savez(os.path.join(save_dir, f'gantry_results_{rid}.npz'), **save_dict)
        print(f'Saved results: gantry_results_{rid}.npz')


## ═══════════════════════════════════════════════════════════════════════════════
## Optuna objective
## ═══════════════════════════════════════════════════════════════════════════════

def objective(trial):
    hp = dict(
        NX_ANN            = trial.suggest_int("NX_ANN", 2, 6),
        n_nodes_per_layer = trial.suggest_categorical("n_nodes_per_layer", [64, 128, 256]),
        n_hidden_layers   = trial.suggest_int("n_hidden_layers", 1, 3),
        nf                = trial.suggest_int("nf", 150, 500, step=50),
        batch_size        = trial.suggest_categorical("batch_size", [1000, 2000, 4000]),
        lr                = trial.suggest_float("lr", 5e-5, 5e-3, log=True),
        epochs            = 40,
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
        _, bestfit = build_and_train(hp)
    except Exception as e:
        print(f"Trial {trial.number} FAILED: {e}")
        return float('inf')

    print(f"\nTrial {trial.number} finished: bestfit = {bestfit:.6f}")
    return bestfit


## ═══════════════════════════════════════════════════════════════════════════════
## Main
## ═══════════════════════════════════════════════════════════════════════════════

if USE_OPTUNA:
    import optuna
    from optuna.samplers import TPESampler

    db_path = os.path.join(save_dir, f"optuna_{OPTUNA_STUDY_NAME}.db")
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=OPTUNA_STUDY_NAME,
        storage=storage,
        sampler=TPESampler(seed=SEED),
        direction="minimize",
        load_if_exists=True,
    )

    print(f"\nOptuna study '{OPTUNA_STUDY_NAME}' - {N_OPTUNA_TRIALS} trials")
    print(f"DB: {db_path}")
    print(f"Completed trials so far: {len(study.trials)}\n")

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
        print(f"  #{t.number:3d}  val={t.value:.6f}  [{status}]  {t.params}")

    # Save CSV
    df = study.trials_dataframe()
    csv_path = os.path.join(save_dir, f'optuna_trials_{run_id}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved trials CSV: {csv_path}")

    # ── Retrain best and do full evaluation ─────────────────────────────────
    best_hp = {**study.best_params, 'epochs': 100}
    print(f"\nRetraining best configuration for full evaluation...")
    np.random.seed(SEED + study.best_trial.number)
    torch.manual_seed(SEED + study.best_trial.number)
    fit_sys, bestfit = build_and_train(best_hp)
    evaluate_and_save(fit_sys, best_hp, f"optuna_best_{run_id}")

else:
    # ── Single run with default hyperparameters ─────────────────────────────
    print(f"\nHyperparameters:")
    for k, v in DEFAULT_HP.items():
        print(f"  {k}: {v}")

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    fit_sys, bestfit = build_and_train(DEFAULT_HP)
    print(f"\nTraining complete. Best validation sim-RMS: {bestfit:.6f}")
    evaluate_and_save(fit_sys, DEFAULT_HP, run_id)
