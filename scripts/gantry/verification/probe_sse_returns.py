"""
probe_sse_returns.py
--------------------
Diagnostic script: run BEFORE finalising gantry_subnet_verification.py.

Probes exactly what Jan's framework returns for each uncertain call,
using synthetic data generated from Gantry_State_Block (no MATLAB files needed).
Fits for 1 epoch only — enough to initialise all internals, not to train properly.

Probes:
  A. fit_sys.apply_experiment(data)  — type, shape, cheat_n, length vs data.y
  B. fit_sys.encoder(u_past, y_past) — shape, dtype, value range, vs x_logical
  C. interconnect.forward(zeros, u)  — type, shape (normalised or physical space?)
  D. fit_sys.norm.transform(data).x  — does x survive the transform?

Run from project root:
    conda run -n GraduationProject python scripts/gantry/verification/probe_sse_returns.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import torch
import deepSI

from model_augmentation.fit_systems.interconnect import Interconnect, SSE_Interconnect
from model_augmentation.fit_systems.blocks import Gantry_State_Block, Linear_Output_Block
from model_augmentation.systems.gantry_ss import Cd, Dd

# ── Synthetic data — step Gantry_State_Block directly ────────────────────────
T    = 1000
Y_op = 0.3
std_x_ones = np.ones((6, 1), dtype=np.float32)
std_u_ones = np.ones((3, 1), dtype=np.float32)

block = Gantry_State_Block(Y_op=Y_op, std_x=std_x_ones, std_u=std_u_ones)
block.eval()

x0      = np.array([0.05, 0.01, 0.30, 0.02, -0.01, 0.05], dtype=np.float32)
u_const = np.array([10.0, -5.0, 3.0], dtype=np.float32)
Cd_np   = Cd.numpy()
Dd_np   = Dd.numpy()

x_arr = np.zeros((T, 6), dtype=np.float32)
u_arr = np.tile(u_const, (T, 1))
y_arr = np.zeros((T, 3), dtype=np.float32)
x_arr[0] = x0

with torch.no_grad():
    for t in range(T - 1):
        z_t = torch.cat([
            torch.from_numpy(x_arr[t]).reshape(1, 6, 1),
            torch.from_numpy(u_const).reshape(1, 3, 1),
        ], dim=1)
        x_arr[t + 1] = block(z_t).squeeze().numpy()
    for t in range(T):
        y_arr[t] = Cd_np @ x_arr[t] + Dd_np @ u_const

NX, NU, NY = 6, 3, 3
NA, NB     = 10, 10
NF         = 50

data = deepSI.System_data(u=u_arr, y=y_arr, x=x_arr, dt=1/20000)
print(f'Synthetic data: T={T}, u={data.u.shape}, y={data.y.shape}, x={data.x.shape}')

# ── Build minimal SSE_Interconnect ────────────────────────────────────────────
std_u = data.u.std(axis=0).reshape(NU, 1).astype(np.float32) + 1e-8
std_x = data.x.std(axis=0).reshape(NX, 1).astype(np.float32) + 1e-8

interconnect = Interconnect(nx=NX, nu=NU, ny=NY)
gantry_block = Gantry_State_Block(Y_op=Y_op, std_x=std_x, std_u=std_u)
output_block  = Linear_Output_Block(C=Cd, D=Dd)
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
    e_net_kwargs={'n_nodes_per_layer': 16, 'n_hidden_layers': 1},
)

print('\nFitting 1 epoch (init only)...')
fit_sys.fit(
    train_sys_data=data,
    val_sys_data=data,
    epochs=1,
    batch_size=64,
    auto_fit_norm=True,
    loss_kwargs={'nf': NF},
    validation_measure='sim-NRMS',
    verbose=0,
)
fit_sys.eval()
print('Done.\n')

SEP = '─' * 60

# ── Probe A: fit_sys.apply_experiment ────────────────────────────────────────
print(SEP)
print('PROBE A: fit_sys.apply_experiment(data)')
print(SEP)
sim_result = fit_sys.apply_experiment(data)

print(f'  type(sim_result)              : {type(sim_result)}')
print(f'  hasattr .y                    : {hasattr(sim_result, "y")}')
print(f'  hasattr .cheat_n              : {hasattr(sim_result, "cheat_n")}')
print(f'  sim_result.cheat_n            : {sim_result.cheat_n}')
print(f'  sim_result.y.shape            : {sim_result.y.shape}')
print(f'  data.y.shape                  : {data.y.shape}')
print(f'  length match                  : {sim_result.y.shape[0] == data.y.shape[0]}')
print(f'  sim_result.normed             : {sim_result.normed}')
print(f'  first cheat_n rows == data.y  : {np.allclose(sim_result.y[:sim_result.cheat_n], data.y[:sim_result.cheat_n], atol=1e-5)}')
print()
cheat = sim_result.cheat_n
nrms = np.sqrt(((sim_result.y[cheat:] - data.y[cheat:])**2).mean(axis=0)) / (data.y[cheat:].std(axis=0) + 1e-12)
print(f'  NRMS excl. warmup [X1,X2,Y]   : {nrms}  (untrained — expect ~1.0)')

# ── Probe B: fit_sys.encoder ──────────────────────────────────────────────────
print(f'\n{SEP}')
print('PROBE B: fit_sys.encoder(u_past, y_past)')
print(SEP)

data_norm = fit_sys.norm.transform(data)
u_past = torch.tensor(data_norm.u[:NB], dtype=torch.float32).unsqueeze(0)  # (1, NB, 3)
y_past = torch.tensor(data_norm.y[:NA], dtype=torch.float32).unsqueeze(0)  # (1, NA, 3)

with torch.no_grad():
    x_hat = fit_sys.encoder(u_past, y_past)

print(f'  type(x_hat)                   : {type(x_hat)}')
print(f'  x_hat.shape                   : {x_hat.shape}')
print(f'  x_hat.dtype                   : {x_hat.dtype}')
print(f'  x_hat (raw encoder output)    : {x_hat.squeeze().numpy()}')
print(f'  x_hat * std_x                 : {x_hat.squeeze().numpy() * std_x.flatten()}')
print(f'  x_logical[NA] (true physical) : {data.x[NA]}')
print(f'  diff (x_hat*std_x - x_true)   : {x_hat.squeeze().numpy() * std_x.flatten() - data.x[NA]}')
print()
print(f'  fit_sys.norm.y0               : {fit_sys.norm.y0}')
print(f'  fit_sys.norm.ystd             : {fit_sys.norm.ystd}')
print(f'  fit_sys.norm.u0               : {fit_sys.norm.u0}')
print(f'  fit_sys.norm.ustd             : {fit_sys.norm.ustd}')

# ── Probe C: interconnect.forward with x=0 ───────────────────────────────────
print(f'\n{SEP}')
print('PROBE C: interconnect.forward(x=zeros, u_norm[0])')
print(SEP)

x_zero = torch.zeros(1, NX)
u_t    = torch.tensor(data_norm.u[0], dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    y_t, x_next = interconnect(x_zero, u_t)

y_t_np = y_t.squeeze().numpy()
print(f'  type(y_t)                     : {type(y_t)}')
print(f'  y_t.shape                     : {y_t.shape}')
print(f'  y_t (raw interconnect output) : {y_t_np}')
print(f'  y_t denorm (* ystd + y0)      : {y_t_np * fit_sys.norm.ystd + fit_sys.norm.y0}')
print(f'  data.y[0]  (physical)         : {data.y[0]}')
print(f'  type(x_next)                  : {type(x_next)}')
print(f'  x_next.shape                  : {x_next.shape}')

# ── Probe D: does x survive norm.transform? ───────────────────────────────────
print(f'\n{SEP}')
print('PROBE D: fit_sys.norm.transform(data).x')
print(SEP)

data_norm2 = fit_sys.norm.transform(data)
print(f'  type(data_norm2)              : {type(data_norm2)}')
print(f'  data.x is None                : {data.x is None}')
print(f'  data_norm2.x is None          : {data_norm2.x is None}')
if data_norm2.x is not None:
    print(f'  data_norm2.x.shape            : {data_norm2.x.shape}')
    print(f'  data_norm2.x == data.x        : {np.allclose(data_norm2.x, data.x)}')

# ── Probe E: interconnect output space — physical or normalised? ──────────────
# Pass known non-zero x (x_arr[0]) and u_norm=0 to the interconnect.
# Compute the expected output two ways:
#   y_phys_expected = Cd @ x_phys + Dd @ u_phys_in_block
#                   = Cd @ (x_norm * std_x) + Dd @ (u_norm * std_u)
#                   = Cd @ x_arr[0] + Dd @ 0      (since std_x=ones, u_norm=0)
#   y_norm_expected = (y_phys_expected - fit_sys.norm.y0) / fit_sys.norm.ystd
#
# Whichever candidate matches y_t_raw tells us the interconnect's output space.
# This resolves whether Check 2's denorm line (* ystd + y0) is correct.
print(f'\n{SEP}')
print('PROBE E: output space — physical or normalised? (x=x_arr[0], u_norm=0)')
print(SEP)

from model_augmentation.systems.gantry_ss import Dd as Dd_tensor
Dd_np_e = Dd_tensor.numpy()

x_known_phys = x_arr[0]                                          # (6,) physical
x_known_norm = torch.tensor(
    (x_known_phys / std_x.flatten()).reshape(1, NX), dtype=torch.float32
)
u_zero_norm  = torch.zeros(1, NU)

with torch.no_grad():
    y_e, _ = interconnect(x_known_norm, u_zero_norm)

y_e_np = y_e.squeeze().numpy()

# u_phys_in_block = u_norm * std_u = 0 * std_u = 0 (regardless of std_u)
y_phys_expected = Cd_np @ x_known_phys + Dd_np_e @ np.zeros(NU)
y_norm_expected = (y_phys_expected - fit_sys.norm.y0) / fit_sys.norm.ystd

print(f'  x_known (physical) [m, m/s]   : {x_known_phys}')
print(f'  x_known_norm (/ std_x)         : {x_known_norm.squeeze().numpy()}')
print(f'  u_norm passed to interconnect  : zeros({NU})')
print()
print(f'  y_t_raw (interconnect output)  : {y_e_np}')
print(f'  y_phys_expected (Cd@x+Dd@0)   : {y_phys_expected}')
print(f'  y_norm_expected (above-y0)/ystd: {y_norm_expected}')
print()
print(f'  |y_t_raw - y_phys_expected|   : {np.abs(y_e_np - y_phys_expected)}')
print(f'  |y_t_raw - y_norm_expected|   : {np.abs(y_e_np - y_norm_expected)}')
print()
print(f'  std_u (block normalisation)    : {std_u.flatten()}')
print(f'  fit_sys.norm.ustd              : {fit_sys.norm.ustd}')
print(f'  std_x (block normalisation)    : {std_x.flatten()}')
print(f'  fit_sys.norm.ystd              : {fit_sys.norm.ystd}')

print(f'\n{SEP}')
print('All probes complete.')
print(SEP)
