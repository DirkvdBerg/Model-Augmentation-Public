"""
compare_dtype.py
----------------
Compare float64 vs float32 simulation accuracy against MATLAB ground truth.

Run as: conda run -n GraduationProject python -m lpv_lfr_baseline.compare_dtype
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from scipy.io import loadmat

from lpv_lfr_baseline.core.physics import (
    M0, M1, M2, K, C, P, ts, build_poly_constants,
    mh as _mh, m1 as _m1, m2 as _m2, mb as _mb, Jb as _Jb, Jh as _Jh,
    Lb as _Lb, d as _d,
)
from lpv_lfr_baseline.core.lfr_matrices import build_G_matrix
from lpv_lfr_baseline.core.lfr_simulate import simulate

mat = loadmat(os.path.join(os.path.dirname(__file__), '..', 'Matlab-output', 'lpv_sim_varying_y.mat'))
q1    = torch.tensor(mat['q1'],   dtype=torch.float64)          # (N, 3) MATLAB reference
u_f64 = torch.tensor(mat['u_q1'], dtype=torch.float64).unsqueeze(0)  # (1, N, 3)
u_f32 = u_f64.float()
x0_f64 = torch.tensor([[0.0, 0.0, 0.3, 0.0, 0.0, 0.0]], dtype=torch.float64)
x0_f32 = torch.tensor([[0.0, 0.0, 0.3, 0.0, 0.0, 0.0]], dtype=torch.float32)

# x0_f32 = x0_f64.float()

G_f64    = build_G_matrix(M0, M1, M2, K, C)
alpha_f64, beta_f64, gamma_f64, N0_f64, N1_f64, N2_f64 = build_poly_constants(
    _m1, _m2, _mb, _mh, _Jb, _Jh, _Lb, _d
)

G_f32    = build_G_matrix(M0.float(), M1.float(), M2.float(), K.float(), C.float())
alpha_f32, beta_f32, gamma_f32, N0_f32, N1_f32, N2_f32 = build_poly_constants(
    _m1.float(), _m2.float(), _mb.float(), _mh.float(),
    _Jb.float(), _Jh.float(), _Lb.float(), _d.float(),
)

print("Running float64 simulation...")
with torch.no_grad():
    r64 = simulate(
        x0_f64, u_f64,
        G_f64, K, C, _mh, alpha_f64, beta_f64, gamma_f64, N0_f64, N1_f64, N2_f64,
        P, ts,
    )

print("Running float32 simulation...")
with torch.no_grad():
    r32 = simulate(
        x0_f32, u_f32,
        G_f32, K.float(), C.float(), _mh.float(),
        alpha_f32, beta_f32, gamma_f32, N0_f32, N1_f32, N2_f32,
        P.float(), ts.float(),
    )

err64 = (r64.Y[0] - q1).abs()
err32 = (r32.Y[0].double() - q1).abs()

ch = ['X1', 'X2', 'Y']
print()
print(f"{'Channel':<6}  {'float64 vs MATLAB':>20}  {'float32 vs MATLAB':>20}")
print("-" * 50)
for i, name in enumerate(ch):
    print(f"{name:<6}  {err64[:,i].max().item():>20.3e}  {err32[:,i].max().item():>20.3e}")

print()
print(f"Detuning signal on Y (reference): ~2.9e-02 m")
print(f"float32/float64 ratio on Y: {err32[:,2].max() / err64[:,2].max().clamp(min=1e-30):.1f}x worse")

import matplotlib.pyplot as plt

N = q1.shape[0]
t = torch.arange(N, dtype=torch.float64) * ts.item()
t_np = t.numpy()
q1_np   = q1.numpy()
y64_np  = r64.Y[0].numpy()
y32_np  = r32.Y[0].double().numpy()
err64_np = err64.numpy()
err32_np = err32.numpy()

fig, axes = plt.subplots(3, 2, figsize=(12, 9))
fig.suptitle("float64 vs float32 vs MATLAB ground truth")

for i, name in enumerate(ch):
    ax_traj, ax_err = axes[i, 0], axes[i, 1]

    ax_traj.plot(t_np, q1_np[:, i],  label='MATLAB',   color='k',      linewidth=1.0)
    ax_traj.plot(t_np, y64_np[:, i], label='float64',  color='tab:blue', linewidth=0.8, linestyle='--')
    ax_traj.plot(t_np, y32_np[:, i], label='float32',  color='tab:orange', linewidth=0.8, linestyle=':')
    ax_traj.set_ylabel(f"{name} (m)")
    ax_traj.legend(fontsize=7)
    if i == 0:
        ax_traj.set_title("Trajectory")
    if i == 2:
        ax_traj.set_xlabel("Time (s)")

    ax_err.semilogy(t_np, err64_np[:, i] + 1e-20, label='float64', color='tab:blue', linewidth=0.8)
    ax_err.semilogy(t_np, err32_np[:, i] + 1e-20, label='float32', color='tab:orange', linewidth=0.8)
    ax_err.set_ylabel(f"|err| {name} (m)")
    ax_err.legend(fontsize=7)
    if i == 0:
        ax_err.set_title("Absolute error vs MATLAB")
    if i == 2:
        ax_err.set_xlabel("Time (s)")

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'compare_dtype_outputs.png'), dpi=150)
print("Plot saved to compare_dtype_outputs.png")

# --- State difference: float64 vs float32 ---
# X shape: (batch, N+1, 6)  states in logical coordinates: [xdot1, xdot2, Ydot, x1, x2, Y]
state_names = ['ẋ1 (m/s)', 'ẋ2 (m/s)', 'Ẏ (m/s)', 'x1 (m)', 'x2 (m)', 'Y (m)']
t_state = torch.arange(N + 1, dtype=torch.float64) * ts.item()
t_state_np = t_state.numpy()
state_diff = (r64.X[0] - r32.X[0].double()).abs().numpy()   # (N+1, 6)

fig2, axes2 = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
fig2.suptitle("State difference: |float64 − float32| (logical coordinates)")
for i, name in enumerate(state_names):
    axes2[i].semilogy(t_state_np, state_diff[:, i] + 1e-30, linewidth=0.8)
    axes2[i].set_ylabel(f"|Δ| {name}")
axes2[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'compare_dtype_states.png'), dpi=150)
print("Plot saved to compare_dtype_states.png")

# --- M(Y) difference: float64 vs float32 ---
# M(Y) = M0 + M1*Y + M2*Y^2, re-evaluated at each step using the recorded Y state (index 2)
# X has N+1 entries; use all of them.
Y_f64 = r64.X[0, :, 2]                   # (N+1,) float64
Y_f32 = r32.X[0, :, 2]                   # (N+1,) float32

# Vectorised: (N+1, 3, 3)
M_f64 = (M0[None]
         + M1[None] * Y_f64[:, None, None]
         + M2[None] * Y_f64[:, None, None] ** 2)
M_f32 = (M0.float()[None]
         + M1.float()[None] * Y_f32[:, None, None]
         + M2.float()[None] * Y_f32[:, None, None] ** 2).double()

M_diff = (M_f64 - M_f32).abs().numpy()   # (N+1, 3, 3)

fig3, axes3 = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
fig3.suptitle("|M_f64(Y) − M_f32(Y)| for each element  (M(Y) = M0 + M1·Y + M2·Y²)")
for i in range(3):
    for j in range(3):
        axes3[i, j].semilogy(t_state_np, M_diff[:, i, j] + 1e-30, linewidth=0.8)
        axes3[i, j].set_title(f"M[{i},{j}]", fontsize=9)
        if j == 0:
            axes3[i, j].set_ylabel("|Δ| (kg)")
        if i == 2:
            axes3[i, j].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'compare_dtype_M.png'), dpi=150)
print("Plot saved to compare_dtype_M.png")
plt.show()
