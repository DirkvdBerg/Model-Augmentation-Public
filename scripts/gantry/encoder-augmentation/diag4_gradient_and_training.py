"""
diag4_gradient_and_training.py
-------------------------------
Two-level diagnostic for linear_encoder_init_aug.

Level A: Gradient connectivity (< 1 second, no data needed)
  Verify that after one forward+backward pass through a minimal dynamics
  model, W^a rows receive nonzero gradients. This confirms x_a is connected
  to the autograd graph and will be trained jointly with the dynamics.

Level B: Synthetic training test (< 2 minutes on CPU)
  True system: 2D linear oscillator [pos, vel], only position observed.
  Physical backbone: 1D pos-only model (deliberately incomplete).
  Compare nx_aug=0 vs nx_aug=1 over 60 training epochs.
  Expected: nx_aug=1 converges lower and x_a correlates with true velocity.

  Both levels mirror the interconnect from gantry_interconnect_dynamic.py:
    x_b(t+1) = A_phys @ x_b(t) + B_phys @ u(t) + delta[:nx_phys]
    x_a(t+1) = delta[nx_phys:]                                       (ANN-driven)
    y(t)     = C_phys @ x_b(t)
  where delta = ann([x_b, x_a, u]) with zero-init final layer.

Usage:
    conda run -n GraduationProject python \\
        scripts/gantry/encoder-augmentation/diag4_gradient_and_training.py
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from model_augmentation.fit_systems.pre_encoder import linear_encoder_init_aug
from model_augmentation.utils.torch_nets import zero_init_feed_forward_nn

OUT_DIR = os.path.join(PROJECT_ROOT, 'simulations', 'gantry_subnet', 'encoder')
os.makedirs(OUT_DIR, exist_ok=True)

TOL = 1e-10   # gradient norm threshold for "nonzero"

# =============================================================================
# Shared check helper
# =============================================================================

results = {}

def check(name, condition, detail=''):
    status = 'PASS' if condition else 'FAIL'
    results[name] = status
    marker = '  ' if condition else '!!'
    print(f'  [{status}] {marker} {name}' + (f'  ({detail})' if detail else ''))
    return condition


# =============================================================================
# Level A: Gradient connectivity test
# =============================================================================

def run_level_A():
    print('\n' + '='*70)
    print('LEVEL A: Gradient connectivity')
    print('='*70)

    # --- Synthetic stable LTI (same style as diag1) ---
    NX_PHYS = 2
    NX_AUG  = 2
    NU      = 1
    NY      = 1
    NA = NB = 4
    BATCH   = 32

    rng = np.random.default_rng(7)
    A_raw = rng.standard_normal((NX_PHYS, NX_PHYS))
    A_phys = A_raw / (np.max(np.abs(np.linalg.eigvals(A_raw))) + 0.1)
    B_phys = rng.standard_normal((NX_PHYS, NU))
    C_phys = rng.standard_normal((NY, NX_PHYS))
    D_phys = rng.standard_normal((NY, NU))

    # Encoder under test
    torch.manual_seed(0)
    encoder = linear_encoder_init_aug(
        A=A_phys, B=B_phys, C=C_phys, D=D_phys,
        nx=NX_PHYS, nu=NU, ny=NY, na=NA, nb=NB,
        nx_aug=NX_AUG,
        n_nodes_per_layer=16, n_hidden_layers=2,
        flag_linear_only=False,
    )

    # Minimal dynamics that uses ALL nxd states (NOT zero-init, so grad flows)
    # Mirrors: ann_block sees [x, u] -> delta in gantry_interconnect_dynamic.py
    nxd = NX_PHYS + NX_AUG
    F = nn.Linear(nxd, nxd, bias=False)   # state transition
    G = nn.Linear(NU,  nxd, bias=False)   # input contribution
    H = nn.Linear(nxd, NY,  bias=False)   # output (uses all states incl. x_a)

    # Zero out all gradients before the pass
    for p in list(encoder.parameters()) + [F.weight, G.weight, H.weight]:
        if p.grad is not None:
            p.grad.zero_()

    # Forward pass
    torch.manual_seed(42)
    uhist    = torch.randn(BATCH, (NB + 1) * NU)
    yhist    = torch.randn(BATCH, (NA + 1) * NY)
    u_now    = torch.randn(BATCH, NU)
    y_target = torch.randn(BATCH, NY)

    x0   = encoder(uhist, yhist)          # (BATCH, NX_PHYS+NX_AUG)
    xp   = F(x0) + G(u_now)              # one-step dynamics
    yhat = H(xp)                          # output depends on ALL states
    loss = nn.functional.mse_loss(yhat, y_target)
    loss.backward()

    # --- Checks ---
    print('\n--- Encoder parameter gradients ---')

    wa_y_n = encoder.Wa_psi_y.grad.norm().item()
    wa_u_n = encoder.Wa_psi_u.grad.norm().item()
    wb_y_n = encoder.Wb_psi_y.grad.norm().item()
    wb_u_n = encoder.Wb_psi_u.grad.norm().item()

    check('A1: Wa_psi_y.grad nonzero', wa_y_n > TOL,
          f'norm={wa_y_n:.3e}  (x_a from y-history in graph)')
    check('A2: Wa_psi_u.grad nonzero', wa_u_n > TOL,
          f'norm={wa_u_n:.3e}  (x_a from u-history in graph)')
    check('A3: Wb_psi_y.grad nonzero', wb_y_n > TOL,
          f'norm={wb_y_n:.3e}  (x_b from y-history in graph)')
    check('A4: Wb_psi_u.grad nonzero', wb_u_n > TOL,
          f'norm={wb_u_n:.3e}  (x_b from u-history in graph)')

    print('\n--- Encoder net gradients ---')
    # net[-1] is the zero-init final linear layer
    # net[0]  is the first linear layer
    # Zero-init final layer: grad of final_layer.weight is NONZERO (it's the
    # terminal matmul), but grad of first hidden layer is ZERO because
    # d(output)/d(hidden) = final_weight = 0 kills backprop through hidden layers.
    final_w_n = encoder.net[-1].weight.grad.norm().item()
    first_w_n = encoder.net[0].weight.grad.norm().item()

    check('A5: net final layer.weight.grad nonzero', final_w_n > TOL,
          f'norm={final_w_n:.3e}  (correction net terminal layer in graph)')
    check('A6: net first layer.weight.grad == 0  [expected]', first_w_n < TOL,
          f'norm={first_w_n:.3e}  (expected: zero-init final layer kills hidden grad at init)')

    print('\n--- Dynamics parameter gradients ---')
    f_n = F.weight.grad.norm().item()
    check('A7: F.weight.grad nonzero', f_n > TOL,
          f'norm={f_n:.3e}  (dynamics updated through full graph)')

    print(f'\n  Level A done.')


# =============================================================================
# Level B: Synthetic training test
# =============================================================================

# --- True 2D oscillator system (pos + vel, only pos observed) ---
# THEORY: standard linear discrete-time oscillator with one unobserved state.
# A 1D physical backbone cannot represent this; nx_aug=1 can learn the velocity.
A_TRUE = np.array([[0.99,  0.01],
                   [-0.10, 0.95]], dtype=np.float32)
B_TRUE = np.array([[0.00], [0.01]], dtype=np.float32)
C_TRUE = np.array([[1.0, 0.0]],    dtype=np.float32)

# --- Physical backbone (deliberately incomplete: 1D pos-only) ---
A_PHYS = np.array([[0.99]], dtype=np.float32)
B_PHYS = np.array([[0.00]], dtype=np.float32)
C_PHYS = np.array([[1.0]],  dtype=np.float32)
D_PHYS = np.array([[0.0]],  dtype=np.float32)

NX_PHYS_B = 1
NU_B = NY_B = 1
NA_B = NB_B = 4
NF_B = 15         # rollout horizon
N_NODES_B = 16
N_HIDDEN_B = 2

# HEURISTIC: small trajectory count + length keeps runtime under 2 minutes on CPU
N_TRAJ_TRAIN = 16
N_TRAJ_VAL   = 4
T_EACH        = 300
NOISE_STD_FRAC = 0.02   # fraction of y std


def simulate_true(u_traj, x0=None, noise_std=0.0):
    """Simulate the 2D true system. Returns y (T, ny) and x (T, nx_true)."""
    T = len(u_traj)
    x = np.zeros(2, dtype=np.float32) if x0 is None else x0.copy()
    ys, xs = [], []
    for t in range(T):
        ys.append((C_TRUE @ x).copy())
        xs.append(x.copy())
        x = A_TRUE @ x + (B_TRUE @ u_traj[t:t+1]).squeeze()
    y = np.array(ys, dtype=np.float32)       # (T, 1)
    x = np.array(xs, dtype=np.float32)       # (T, 2)
    if noise_std > 0:
        y += np.random.randn(*y.shape).astype(np.float32) * noise_std
    return y, x


def make_windows(u, y, na, nb, nf):
    """Sliding windows. u: (T,1), y: (T,1). Returns float32 arrays."""
    N = len(u)
    k0 = max(na, nb)
    uh, yh, uf, yf = [], [], [], []
    for k in range(k0, N - nf):
        # (nb+1) samples ending at k inclusive; most-recent last -- matches encoder
        uh.append(u[k - nb: k + 1].flatten())   # (nb+1)*nu
        yh.append(y[k - na: k + 1].flatten())   # (na+1)*ny
        uf.append(u[k + 1: k + 1 + nf])         # (nf, nu)
        yf.append(y[k + 1: k + 1 + nf])         # (nf, ny)
    return (np.array(uh, dtype=np.float32),
            np.array(yh, dtype=np.float32),
            np.array(uf, dtype=np.float32),
            np.array(yf, dtype=np.float32))


class ToyAugModel(nn.Module):
    """Minimal augmented model mirroring gantry_interconnect_dynamic.py.

    State update:
      x_b(t+1) = A_phys @ x_b + B_phys @ u + delta[:nx_phys]
      x_a(t+1) = delta[nx_phys:]
      y(t)     = C_phys @ x_b
    where delta = ann([x_b, x_a, u]), zero-init final layer.
    """

    def __init__(self, nx_phys, nx_aug, nu, ny):
        super().__init__()
        self.nx_phys = nx_phys
        self.nx_aug  = nx_aug
        nxd          = nx_phys + nx_aug

        self.A_phys = torch.tensor(A_PHYS, dtype=torch.float32)
        self.B_phys = torch.tensor(B_PHYS, dtype=torch.float32)
        self.C_phys = torch.tensor(C_PHYS, dtype=torch.float32)

        n_in_ann  = nx_phys + nx_aug + nu
        n_out_ann = nxd   # delta[:nx_phys] -> x_b correction, delta[nx_phys:] -> x_a
        self.ann = zero_init_feed_forward_nn(
            n_in=n_in_ann, n_out=n_out_ann,
            n_nodes_per_layer=N_NODES_B, n_hidden_layers=N_HIDDEN_B,
        )

    def step(self, x, u):
        """x: (batch, nxd), u: (batch, nu). Returns y: (batch, ny), xnext: (batch, nxd)."""
        x_b = x[:, :self.nx_phys]
        x_a = x[:, self.nx_phys:]

        A = self.A_phys.to(x.device)
        B = self.B_phys.to(x.device)
        C = self.C_phys.to(x.device)

        xb_phys  = x_b @ A.T + u @ B.T
        ann_in   = torch.cat([x_b, x_a, u], dim=1)
        delta    = self.ann(ann_in)                          # (batch, nxd)

        xb_next  = xb_phys + delta[:, :self.nx_phys]
        xa_next  = delta[:, self.nx_phys:]
        x_next   = torch.cat([xb_next, xa_next], dim=1)
        y        = x_b @ C.T
        return y, x_next

    def rollout(self, x0, u_future):
        """u_future: (batch, nf, nu). Returns y_pred: (batch, nf, ny)."""
        x = x0
        ys = []
        for t in range(u_future.shape[1]):
            y_t, x = self.step(x, u_future[:, t, :])
            ys.append(y_t)
        return torch.stack(ys, dim=1)    # (batch, nf, ny)


def train_one_run(nx_aug, seed, train_wins, val_wins, std_u, std_y, n_epochs=60):
    """Train encoder+model for one nx_aug value. Returns val_loss_curve and final model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    encoder = linear_encoder_init_aug(
        A=A_PHYS, B=B_PHYS, C=C_PHYS, D=D_PHYS,
        nx=NX_PHYS_B, nu=NU_B, ny=NY_B, na=NA_B, nb=NB_B,
        nx_aug=nx_aug,
        n_nodes_per_layer=N_NODES_B, n_hidden_layers=N_HIDDEN_B,
        flag_linear_only=False,
    )
    model = ToyAugModel(NX_PHYS_B, nx_aug, NU_B, NY_B)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(model.parameters()), lr=5e-3)

    # Unpack windows (already normalized)
    uh_tr, yh_tr, uf_tr, yf_tr = [torch.tensor(w) for w in train_wins]
    uh_va, yh_va, uf_va, yf_va = [torch.tensor(w) for w in val_wins]

    BATCH = 128
    N_TR  = uh_tr.shape[0]

    def compute_loss(uh, yh, uf, yf):
        x0     = encoder(uh, yh)
        y_pred = model.rollout(x0, uf)          # (batch, nf, ny)
        return nn.functional.mse_loss(y_pred, yf)

    val_losses  = []
    train_losses = []

    for epoch in range(n_epochs):
        encoder.train(); model.train()
        idx   = torch.randperm(N_TR)
        t_acc = 0.0
        n_batches = 0
        for start in range(0, N_TR - BATCH + 1, BATCH):
            bi = idx[start: start + BATCH]
            optimizer.zero_grad()
            loss = compute_loss(uh_tr[bi], yh_tr[bi], uf_tr[bi], yf_tr[bi])
            loss.backward()
            optimizer.step()
            t_acc += loss.item()
            n_batches += 1
        train_losses.append(t_acc / max(n_batches, 1))

        encoder.eval(); model.eval()
        with torch.no_grad():
            vloss = compute_loss(uh_va, yh_va, uf_va, yf_va).item()
        val_losses.append(vloss)

        if (epoch + 1) % 10 == 0:
            print(f'    epoch {epoch+1:3d}  train={train_losses[-1]:.4e}  val={val_losses[-1]:.4e}')

    return encoder, model, np.array(train_losses), np.array(val_losses)


def run_level_B():
    print('\n' + '='*70)
    print('LEVEL B: Synthetic training test')
    print('='*70)
    print(f'  True system: 2D oscillator  A_true={A_TRUE.tolist()}')
    print(f'  Physical backbone: 1D pos-only  A_phys=[[0.99]]')
    print(f'  na=nb={NA_B}  nf={NF_B}  n_epochs=60')

    # --- Generate data ---
    rng = np.random.default_rng(42)
    all_u  = []
    all_y  = []
    all_x  = []

    for _ in range(N_TRAJ_TRAIN + N_TRAJ_VAL):
        u_t = rng.standard_normal((T_EACH, 1)).astype(np.float32)
        y_t, x_t = simulate_true(u_t)
        all_u.append(u_t)
        all_y.append(y_t)
        all_x.append(x_t)

    # Normalization: pure-scaled (matches linear encoder convention)
    # HEURISTIC: divide by std only (no mean subtraction), consistent with diag2/diag3
    u_cat = np.concatenate(all_u[:N_TRAJ_TRAIN])
    y_cat = np.concatenate(all_y[:N_TRAJ_TRAIN])
    std_u = float(u_cat.std()) + 1e-8
    std_y = float(y_cat.std()) + 1e-8

    noise_std = NOISE_STD_FRAC * std_y
    # Add measurement noise now that we know std_y
    rng2 = np.random.default_rng(99)
    for i in range(len(all_y)):
        all_y[i] = all_y[i] + rng2.standard_normal(all_y[i].shape).astype(np.float32) * noise_std

    def norm_u(u): return u / std_u
    def norm_y(y): return y / std_y

    # Build windows
    def build_wins(indices):
        wuh, wyh, wuf, wyf = [], [], [], []
        for i in indices:
            u_n = norm_u(all_u[i])
            y_n = norm_y(all_y[i])
            uh, yh, uf, yf = make_windows(u_n, y_n, NA_B, NB_B, NF_B)
            wuh.append(uh); wyh.append(yh); wuf.append(uf); wyf.append(yf)
        return [np.concatenate(w) for w in [wuh, wyh, wuf, wyf]]

    train_idx = list(range(N_TRAJ_TRAIN))
    val_idx   = list(range(N_TRAJ_TRAIN, N_TRAJ_TRAIN + N_TRAJ_VAL))
    train_wins = build_wins(train_idx)
    val_wins   = build_wins(val_idx)
    print(f'\n  Training windows: {train_wins[0].shape[0]}  '
          f'Validation windows: {val_wins[0].shape[0]}')

    # --- Run nx_aug=0 (baseline) ---
    print('\n  Training nx_aug=0 (no augmentation)...')
    enc0, mdl0, tl0, vl0 = train_one_run(0, seed=42, train_wins=train_wins,
                                          val_wins=val_wins, std_u=std_u, std_y=std_y)

    # --- Run nx_aug=1 (augmented) ---
    print('\n  Training nx_aug=1 (one ANN state)...')
    enc1, mdl1, tl1, vl1 = train_one_run(1, seed=42, train_wins=train_wins,
                                          val_wins=val_wins, std_u=std_u, std_y=std_y)

    # --- Checks ---
    print('\n--- Checks ---')

    final_loss_0 = float(vl0[-1])
    final_loss_1 = float(vl1[-1])
    improvement  = (final_loss_0 - final_loss_1) / (final_loss_0 + 1e-12)

    check('B1: val loss aug < val loss baseline',
          final_loss_1 < final_loss_0,
          f'nx_aug=0: {final_loss_0:.4e}  nx_aug=1: {final_loss_1:.4e}')
    check('B2: relative improvement > 10%',
          improvement > 0.10,
          f'improvement={improvement*100:.1f}%')

    # Collect x_a over validation set with trained aug encoder
    uh_va, yh_va = torch.tensor(val_wins[0]), torch.tensor(val_wins[1])
    enc1.eval()
    with torch.no_grad():
        x_enc = enc1(uh_va, yh_va).numpy()   # (N_val_wins, nxd)
    x_a_val = x_enc[:, NX_PHYS_B:]           # (N_val_wins, 1)
    x_b_val = x_enc[:, :NX_PHYS_B]

    xa_rms = float(np.sqrt(np.mean(x_a_val ** 2)))
    xb_rms = float(np.sqrt(np.mean(x_b_val ** 2)))
    check('B3: x_a active after training',
          xa_rms > 0.05 * xb_rms,
          f'x_a RMS={xa_rms:.3e}  x_b RMS={xb_rms:.3e}  ratio={xa_rms/(xb_rms+1e-12):.3f}')

    # Correlation between x_a and true velocity at matching window indices
    # Window k starts at k0=max(na,nb)=4, so window index w corresponds to time k0+w
    k0 = max(NA_B, NB_B)
    # val trajectories
    dq_vals = []
    for i in val_idx:
        dq_vals.append(all_x[i][:, 1])  # velocity column of true state
    dq_all = np.concatenate(dq_vals)

    n_wins_per_traj = train_wins[0].shape[0] // N_TRAJ_TRAIN  # approx
    n_val_wins = val_wins[0].shape[0]
    # Collect true velocity at the encoder output time (index k for each window)
    dq_at_windows = []
    for i in val_idx:
        x_traj = all_x[i]
        u_n = norm_u(all_u[i])
        y_n = norm_y(all_y[i])
        _, _, _, _ = make_windows(u_n, y_n, NA_B, NB_B, NF_B)  # just to count
        for k in range(k0, T_EACH - NF_B):
            dq_at_windows.append(x_traj[k, 1])   # velocity at encoder time k

    dq_at_windows = np.array(dq_at_windows, dtype=np.float32)
    min_len = min(len(dq_at_windows), len(x_a_val))
    corr = float(np.corrcoef(x_a_val[:min_len, 0],
                              dq_at_windows[:min_len])[0, 1])
    check('B4: |corr(x_a, true dq)| > 0.5',
          abs(corr) > 0.5,
          f'corr={corr:.3f}  (|corr| > 0.5 means x_a learned velocity)')

    # x_b NRMS improvement
    enc0.eval()
    with torch.no_grad():
        xb0 = enc0(uh_va, yh_va).numpy()[:, :NX_PHYS_B]

    # True position at window times
    pos_at_windows = []
    for i in val_idx:
        for k in range(k0, T_EACH - NF_B):
            pos_at_windows.append(all_x[i][k, 0])
    pos_at_windows = np.array(pos_at_windows, dtype=np.float32)
    pos_norm = pos_at_windows / std_y    # position in normalized units

    xb1 = x_enc[:, :NX_PHYS_B]
    n = min(len(pos_norm), len(xb0), len(xb1))
    nrms0 = float(np.sqrt(np.mean((xb0[:n, 0] - pos_norm[:n]) ** 2)) /
                  (np.sqrt(np.mean(pos_norm[:n] ** 2)) + 1e-12))
    nrms1 = float(np.sqrt(np.mean((xb1[:n, 0] - pos_norm[:n]) ** 2)) /
                  (np.sqrt(np.mean(pos_norm[:n] ** 2)) + 1e-12))
    check('B5: x_b NRMS improves with augmentation',
          nrms1 < nrms0,
          f'NRMS nx_aug=0: {nrms0:.4f}  nx_aug=1: {nrms1:.4f}')

    print(f'\n  Level B done.')
    print(f'    Final val loss:  nx_aug=0 = {final_loss_0:.4e}  '
          f'nx_aug=1 = {final_loss_1:.4e}  ({improvement*100:.1f}% better)')
    print(f'    x_a correlation with true velocity: {corr:.3f}')

    # --- Plots ---
    _plot_loss_curves(tl0, vl0, tl1, vl1)
    _plot_xa_correlation(x_a_val[:min_len, 0], dq_at_windows[:min_len])

    return dict(
        final_loss_0=final_loss_0,
        final_loss_1=final_loss_1,
        improvement=improvement,
        xa_rms=xa_rms, xb_rms=xb_rms,
        corr=corr,
        nrms0=nrms0, nrms1=nrms1,
    )


def _plot_loss_curves(tl0, vl0, tl1, vl1):
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = np.arange(1, len(vl0) + 1)
    ax.semilogy(epochs, vl0, 'C0-',  lw=1.5, label='val  nx_aug=0 (baseline)')
    ax.semilogy(epochs, tl0, 'C0--', lw=0.8, alpha=0.5, label='train nx_aug=0')
    ax.semilogy(epochs, vl1, 'C1-',  lw=1.5, label='val  nx_aug=1 (augmented)')
    ax.semilogy(epochs, tl1, 'C1--', lw=0.8, alpha=0.5, label='train nx_aug=1')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE loss (normalized)')
    ax.set_title('Level B: val loss convergence -- nx_aug=1 should finish lower')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'diag4_loss_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


def _plot_xa_correlation(x_a_flat, dq_flat):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(dq_flat[::5], x_a_flat[::5], s=4, alpha=0.3, color='C1')
    ax.set_xlabel('True velocity dq (normalized)')
    ax.set_ylabel('Learned x_a (normalized)')
    ax.set_title('Level B: x_a vs true velocity -- high correlation = learned state')
    corr = float(np.corrcoef(x_a_flat, dq_flat)[0, 1])
    ax.text(0.05, 0.95, f'corr = {corr:.3f}', transform=ax.transAxes,
            fontsize=10, va='top',
            color='green' if abs(corr) > 0.5 else 'red')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'diag4_xa_correlation.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


# =============================================================================
# Main
# =============================================================================

def main():
    print('=' * 70)
    print('Diagnostic 4: Gradient connectivity + synthetic training test')
    print('=' * 70)

    run_level_A()
    b_metrics = run_level_B()

    # --- Final summary ---
    n_pass = sum(v == 'PASS' for v in results.values())
    n_fail = sum(v == 'FAIL' for v in results.values())

    print('\n' + '=' * 70)
    print(f'  {n_pass}/{len(results)} checks passed')

    if n_fail > 0:
        failed = [k for k, v in results.items() if v == 'FAIL']
        print(f'  FAILED: {failed}')
        if any(k.startswith('A') for k in failed):
            print('  Level A failure: x_a is NOT in the autograd graph.')
            print('  Check linear_encoder_init_aug forward() for detach() or in-place ops.')
        if any(k.startswith('B') for k in failed):
            print('  Level B failure: augmentation did not help on a system where it should.')
            print('  Check ToyAugModel.step() -- verify x_a feeds into ann_in.')
    else:
        print('  All checks passed:')
        print('    Level A: W^a rows are in the computation graph, gradients flow.')
        print('    Level B: nx_aug=1 outperforms nx_aug=0; x_a learned velocity.')

    # Save JSON summary
    json_path = os.path.join(OUT_DIR, 'diag4_results.json')
    with open(json_path, 'w') as f:
        json.dump(dict(
            checks=results,
            level_B=b_metrics,
            summary=dict(n_pass=n_pass, n_fail=n_fail, total=len(results)),
        ), f, indent=2)
    print(f'\n  Saved: {json_path}')

    if n_fail > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
