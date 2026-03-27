"""
lfr_simulate.py
---------------
Standalone RK4 simulation loop for the dual-gantry LPV-LFR baseline.

Uses lfr_forward.py for the CT vector field evaluation at each RK4 sub-step.
Discretization: RK4 with fixed step ts = 1/fs. Consistent with D-018.

At each timestep k:
    - Y[k] = x[k][2]  (self-scheduled from state, logical coordinate index 2)
    - RK4 sub-steps each re-evaluate lfr_forward at intermediate states
    - z[k] and w[k] are recorded at the start of each step (from x[k], not sub-steps)
    - y[k] = Cy @ x[k]

Note on RK4 vs ZOH:
    Validation against gantry_lpv_torch.py (ZOH) will show small differences
    at 16kHz (ts = 62.5us). This is expected — RK4 and ZOH are different
    discretization methods. Differences should be bounded and small.
    Do not switch to ZOH here to "match" MATLAB exactly — RK4 is the chosen method (D-018).

Provides:
    simulate(x0, u_seq, fs=16e3) -> SimResult
        x0     : (6,)   initial state in logical coordinates
        u_seq  : (N, 3) input sequence in stage coordinates
        returns SimResult with fields:
            X  : (N+1, 6)  state trajectory (logical coordinates)
            Y  : (N, 3)    output trajectory (stage coordinates)
            Z  : (N, 6)    latent z trajectory
            W  : (N, 6)    latent w trajectory
"""
