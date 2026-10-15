"""Memory of the closed loop with K1 against the 0.1 s training window (Section V-B).

Justifies: "Training windows last 0.1 s. This exceeds the memory of the closed loop, in which a force
error has decayed after about 50 ms" (D-220, method A: model matrices; simulation-only design info).
A force error w of the model reaches the servo error through PS = (I + G K1)^-1 G (D-220). Per training
position, from the linearised simulated system (absorber on, friction off, continuous time):
  tau_max  = 1 / min |Re p| over the closed-loop poles (slowest time constant)
  t_decay  = last time at which the largest entry of the impulse response of PS exceeds 1 % of its peak
Input: meta.truth. Output: Results/setup_loop_memory.csv; macros SetupTauMax [ms], SetupDecay [ms],
SetupWindow [ms].
"""
import os

import numpy as np
from scipy.linalg import expm
from scipy.signal import tf2ss

import _setup_common as sc

TRAIN_Y = (-0.30, -0.15, 0.0, 0.15, 0.30)
WINDOW_S = 0.1                  # n_f T_s, the thesis window (D-220, entry THESIS_NF default)
# HEURISTIC: 1 % of the peak as "decayed" (D-220 / JH-013 use their own threshold; reported, not a gate)
DECAY_FRAC = 0.01


def plant_ss(T, Y):
    P = sc.P_matrix(T)
    C, K = sc.CK_logical(T)
    M = sc.M_truth(T, Y)
    Ce = np.zeros((4, 4)); Ce[:3, :3] = C; Ce[3, 3] = T['ca']
    Ke = np.zeros((4, 4)); Ke[:3, :3] = K; Ke[3, 3] = T['ka']
    Mi = np.linalg.inv(M)
    A = np.block([[np.zeros((4, 4)), np.eye(4)], [-Mi @ Ke, -Mi @ Ce]])
    B = np.vstack([np.zeros((4, 3)), Mi @ np.vstack([P, np.zeros((1, 3))])])
    Cy = np.hstack([P.T, np.zeros((3, 5))])
    return A, B, Cy


def controller_ss(T):
    num, den, kappa = sc.k1_continuous(T)
    blocks = [tf2ss(kappa[j] * num, den) for j in range(3)]
    n = sum(b[0].shape[0] for b in blocks)
    A = np.zeros((n, n)); B = np.zeros((n, 3)); C = np.zeros((3, n)); D = np.zeros((3, 3))
    i = 0
    for j, (Aj, Bj, Cj, Dj) in enumerate(blocks):
        m = Aj.shape[0]
        A[i:i + m, i:i + m] = Aj; B[i:i + m, j] = Bj[:, 0]; C[j, i:i + m] = Cj[0]; D[j, j] = Dj[0, 0]
        i += m
    return A, B, C, D


def main():
    ds = sc.dataset_arg()
    T = sc.truth_params(ds)
    Ac, Bc, Cc, Dc = controller_ss(T)
    rows, taus, decays = [], [], []
    dt, tend = 1e-4, 0.3
    for Y in TRAIN_Y:
        Ap, Bp, Cp = plant_ss(T, Y)
        # THEORY: negative feedback u = Cc xc + Dc e + w, e = -y; closed loop from w to y is PS
        A = np.block([[Ap - Bp @ Dc @ Cp, Bp @ Cc], [-Bc @ Cp, Ac]])
        B = np.vstack([Bp, np.zeros((Ac.shape[0], 3))])
        C = np.hstack([Cp, np.zeros((3, Ac.shape[0]))])
        p = np.linalg.eigvals(A)
        assert np.all(p.real < 0), 'closed loop unstable at Y=%g' % Y
        tau = 1.0 / np.min(np.abs(p.real))
        Phi = expm(A * dt)
        x = B.copy(); h = []
        for _ in range(int(round(tend / dt))):
            h.append(np.max(np.abs(C @ x))); x = Phi @ x
        h = np.array(h); t = dt * np.arange(len(h))
        td = t[np.nonzero(h > DECAY_FRAC * h.max())[0][-1]]
        taus.append(tau); decays.append(td)
        rows.append([Y, tau, td])
        print('Y %+.2f  tau_max %.1f ms  1%%-decay %.1f ms' % (Y, 1e3 * tau, 1e3 * td))
    sc.write_csv('loop_memory', ['Y', 'tau_max_s', 't_decay_1pct_s'], rows)
    sc.set_macros(os.path.splitext(os.path.basename(__file__))[0], {
        'SetupTauMax': ('%.0f' % (1e3 * max(taus)), 'slowest closed-loop time constant over training Y [ms]'),
        'SetupDecay': ('%.0f' % (1e3 * max(decays)), 'largest 1 % decay time of the PS impulse response [ms]'),
        'SetupWindow': ('%.0f' % (1e3 * WINDOW_S), 'training window [ms]'),
    })


if __name__ == '__main__':
    main()
