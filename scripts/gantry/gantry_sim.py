"""
gantry_sim.py
-------------
Simulates the discrete-time gantry FP baseline model (frozen LTI) using
a pre-recorded input signal from MATLAB.

Imports the model from gantry_ss.py — do not redefine A, B, C, D here.

Simulation convention (deviation coordinates):
    MATLAB's lsim in main.m subtracts the Y operating point (Y=0.3m) from
    the input before simulation and adds it back to the output afterwards
    (main.m lines 231-233). We follow the same convention:
        - x_0 = 0  (all states start at zero deviation)
        - u is the pre-recorded controller output, already in deviation coords
        - After simulation: y[:, 2] += Y_op  (restore absolute Y position)

Reference: kamtin-fp-model/03 Simulink gantry/main.m, lines 231-233
"""

import numpy as np
import os
from scipy.io import loadmat

import sys
sys.path.insert(0, os.path.dirname(__file__))
from gantry_ss import gantry_discrete_ss


def simulate(u, Y_op=0.3, fs=16e3):
    """
    Simulate the gantry FP baseline model for a given input sequence.

    Parameters
    ----------
    u    : np.ndarray, shape (N, 3)
           Force input sequence in stage coordinates [F_X1, F_X2, F_Y].
           Must be in deviation coordinates (as produced by MATLAB lsim).
    Y_op : float, Y operating point offset added back to output channel 2.
           Default: 0.3 m  (matches main.m)
    fs   : float, sample frequency. Default: 16000 Hz.

    Returns
    -------
    y : np.ndarray, shape (N, 3)
        Simulated output in absolute stage coordinates [X1, X2, Y].
    """
    A, B, C, D = gantry_discrete_ss(Y=Y_op, fs=fs)

    N = u.shape[0]
    y = np.zeros((N, 3))
    x = np.zeros(6)   # x_0 = 0 (deviation coordinates, matches MATLAB lsim)

    for k in range(N):
        y[k] = C @ x
        x = A @ x + B @ u[k]

    # Restore absolute Y position (undo deviation coordinate convention)
    y[:, 2] += Y_op

    return y


if __name__ == '__main__':
    # Load input signal from MATLAB
    repo_root = os.path.join(os.path.dirname(__file__), '..', '..')
    input_path = os.path.join(repo_root, 'Matlab-output', 'gantry_input.mat')

    mat = loadmat(input_path, squeeze_me=True)
    u = mat['u']   # (N, 3) force inputs in deviation coordinates
    t = mat['t']   # (N,)  time vector

    print(f"Loaded input: {u.shape[0]} timesteps, {u.shape[1]} channels")
    print(f"Duration: {t[-1]:.3f} s  at  fs = {1/(t[1]-t[0]):.0f} Hz")

    # Run simulation
    y = simulate(u)

    print(f"Simulation complete. Output shape: {y.shape}")
    print(f"  X1 range: [{y[:,0].min():.4f}, {y[:,0].max():.4f}] m")
    print(f"  X2 range: [{y[:,1].min():.4f}, {y[:,1].max():.4f}] m")
    print(f"  Y  range: [{y[:,2].min():.4f}, {y[:,2].max():.4f}] m")

    # Save output to simulations/frozen_lti/
    out_dir = os.path.join(repo_root, 'simulations', 'frozen_lti')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'y.npz')
    np.savez(out_path, y=y, t=t)
    print(f"\nSaved simulation output to: {out_path}")
