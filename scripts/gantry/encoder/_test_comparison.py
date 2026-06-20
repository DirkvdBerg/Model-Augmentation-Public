"""Quick integration test for the state MSE vs output prediction comparison."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from scripts.gantry.encoder.encoder_io_validation import *
import copy as cp

# Test: load one file, create both window types
u, y, x_log = load_mat(TRAIN_FILES[0])
print(f'Loaded: u={u.shape}, y={y.shape}, x_log={x_log.shape}')

norm = compute_normalization([(u, y, x_log)])

# IO windows
uh, yh, uf, yf = create_io_windows(u, y, norm, HP['n_steps'])
print(f'IO windows: u_hist={uh.shape}, u_future={uf.shape}')

# State windows
suh, syh, sxt = create_state_windows(u, y, x_log, norm)
print(f'State windows: u_hist={suh.shape}, x_target={sxt.shape}')

# Build encoder + copy
enc = build_encoder(norm)
enc_smse = cp.deepcopy(enc)
print(f'Encoder params: {sum(p.numel() for p in enc.parameters())}')

# Quick forward pass test
import torch
u_t = torch.tensor(suh[:8], dtype=torch.float32)
y_t = torch.tensor(syh[:8], dtype=torch.float32)
x_hat = enc_smse(u_t, y_t)
print(f'Encoder output shape: {x_hat.shape}')

# State MSE loss test
x_tgt = torch.tensor(sxt[:8], dtype=torch.float32)
loss = torch.mean((x_hat - x_tgt) ** 2)
loss.backward()
print(f'State MSE loss: {loss.item():.4e} - gradient OK')

print('\nAll integration tests passed.')
