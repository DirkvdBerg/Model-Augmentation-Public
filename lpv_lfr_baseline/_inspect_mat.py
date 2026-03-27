from scipy.io import loadmat
import numpy as np
import os

base = os.path.join(os.path.dirname(__file__), '..', 'Matlab-output')

# Check Y range in lpv_matrices
mat = loadmat(os.path.join(base, 'lpv_matrices.mat'))
Y_values = mat['Y_values'].squeeze()
print(f"lpv_matrices.mat — Y_values: min={Y_values.min():.4f}, max={Y_values.max():.4f}, n={len(Y_values)}")
print(f"  first 5: {Y_values[:5]}")
print(f"  det_M range: min={mat['det_M'].min():.4f}, max={mat['det_M'].max():.4f}")

# Check simulation files
for fname in ['gantry_q3_lsim.mat', 'gantry_q_simscape.mat', 'gantry_input.mat', 'lpv_sim_varying_y.mat']:
    path = os.path.join(base, fname)
    print(f"\n=== {fname} ===")
    m = loadmat(path)
    for k, v in m.items():
        if k.startswith('_'):
            continue
        arr = np.array(v)
        print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
        if arr.size <= 6:
            print(f"    values={arr.squeeze()}")
