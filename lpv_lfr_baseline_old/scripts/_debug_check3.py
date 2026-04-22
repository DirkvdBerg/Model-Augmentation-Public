import sys, os

if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    import torch
    import numpy as np
    from scipy.io import loadmat

    mat_base = os.path.join(os.path.dirname(__file__), '..', 'Matlab-output')
    mat_input = loadmat(os.path.join(mat_base, 'gantry_input.mat'))
    mat_lsim  = loadmat(os.path.join(mat_base, 'gantry_q3_lsim.mat'))

    u = mat_input['u']
    r = mat_input['r']
    t = mat_input['t'].squeeze()
    q3 = mat_lsim['q3']

    print("=== gantry_input.mat ===")
    print(f"u shape: {u.shape}  (stage forces [F_X1, F_X2, F_Y])")
    print(f"r shape: {r.shape}  (stage reference [X1, X2, Y])")
    print(f"t: 0 to {t[-1]:.4f} s  ({len(t)} steps)")
    print()
    print("Reference r first/last 3 rows:")
    print(f"  r[0]  = {r[0]}")
    print(f"  r[1]  = {r[1]}")
    print(f"  r[-1] = {r[-1]}")
    print()
    print(f"r Y col (col 2): min={r[:,2].min():.4f}  max={r[:,2].max():.4f}  first={r[0,2]:.4f}")
    print(f"r X1 col (col 0): min={r[:,0].min():.4f}  max={r[:,0].max():.4f}  first={r[0,0]:.4f}")
    print()
    print("=== gantry_q3_lsim.mat ===")
    print(f"q3 shape: {q3.shape}  (stage output [X1, X2, Y])")
    print(f"q3[0]  = {q3[0]}  (initial output)")
    print(f"q3[1]  = {q3[1]}")
    print(f"q3[-1] = {q3[-1]}")
    print()
    print(f"q3 Y col: min={q3[:,2].min():.4f}  max={q3[:,2].max():.4f}  first={q3[0,2]:.4f}")
    print()
    print("Input u stats:")
    print(f"  u col 0 (F_X1): min={u[:,0].min():.2f}  max={u[:,0].max():.2f}")
    print(f"  u col 1 (F_X2): min={u[:,1].min():.2f}  max={u[:,1].max():.2f}")
    print(f"  u col 2 (F_Y):  min={u[:,2].min():.2f}  max={u[:,2].max():.2f}")
