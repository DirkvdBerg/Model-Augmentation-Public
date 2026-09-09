"""Minimal testbed reproducing the gantry's structure at 2 states instead of 6.

WHAT THIS IS FOR, and the label it must always carry.

  Category: DEVELOPMENT AND FALSIFICATION TOOL. Not evidence.
  A method that fails here is dead. A method that works here still has to be shown on the
  gantry and then on Telica. Results from this folder may never be quoted as support for a
  claim about the gantry (RULES.md Rule 1 reporting requirement).

WHY IT IS SHAPED LIKE THIS. Four properties are reproduced deliberately; a smaller or tidier
example would test nothing.

  1. THE DISCREPANCY IS DYNAMIC. The truth is the baseline plus a hidden mass-spring-damper
     absorber, the same story as the gantry's hidden MSD. A static map cannot represent it, so
     the additional states are forced to matter. With a static nonlinearity instead, a static
     network would suffice and the entire question would evaporate.
  2. PARTIAL OBSERVATION. Output is position only, so the state must be reconstructed by an
     encoder. Without that, the encoder freedom `E dxi` of gaps.md Section 9 is untestable.
  3. EXACT RANK DEFICIENCY. The stiffness is split as `k = ka + kb`, two springs in parallel.
     Only the sum is identifiable, so there is an exact null direction `(0, 0, 1, -1)`. This is
     the same structure as the gantry's `kb1 + kb2`, and it forces the `kappa` reparameterisation
     to be exercised rather than assumed away.
  4. A WEAK DIRECTION. Light damping makes the `c` direction poorly excited, so the sensitivity
     spectrum is graded rather than flat. A well-conditioned toy would make every method look
     fine and prove nothing; the gantry spectrum spans 1.9e+01 to 2.9e-19.

MODEL

  truth      m1 xdd1 + c1 xd1 + (ka+kb) x1 + c2 (xd1-xd2) + k2 (x1-x2) = u
             m2 xdd2 + c2 (xd2-xd1) + k2 (x2-x1) = 0
  baseline   m xdd + c xd + (ka+kb) x = u                    theta = (m, c, ka, kb)
  output     y = x1                                          positions only
  augment    w = net([x_phys; x_aug; u]) in R^4; rows 0-1 correct x_phys, rows 2-3 become x_aug

Discretisation is exact zero-order hold via the matrix exponential, in torch, so the parameter
Jacobian comes from autograd with no integrator error mixed in.

Run: conda run -n GraduationProject python scripts/gantry/orthogonality/testbed/system.py
"""
__project_origin__ = "added"

import numpy as np
import torch

F64 = torch.float64

# ---- truth parameters. Absorber tuned above the main resonance. --------------------------
M1, C1, K1 = 1.0, 0.40, 100.0          # main:     wn = 10 rad/s = 1.59 Hz, zeta = 0.02
M2, C2, K2 = 0.005, 0.02, 2.0           # absorber: wn = 20 rad/s = 3.18 Hz, zeta_a = 0.10
# Mass ratio 0.5%, giving a discrepancy of ~10% of the output std. Chosen deliberately:
# at the first setting tried (5% mass) the absorber contributed 80% of the output, which
# is a different system rather than a perturbation, and the entire framework under test
# is FIRST ORDER. A discrepancy that large would strain the assumption being tested and
# would make the testbed unable to falsify anything about it.
KA_TRUE, KB_TRUE = 50.0, 50.0          # ka + kb = K1, only the sum is identifiable

THETA_NAMES = ('m', 'c', 'ka', 'kb')
THETA_TRUE = np.array([M1, C1, KA_TRUE, KB_TRUE])

# fs = 50 Hz is 15x the highest mode of interest (the absorber at 3.18 Hz). The first version
# used 200 Hz, a 60x oversampling that bought nothing and made every rollout 4x longer for the
# same physical time. Rollout cost scales with the number of STEPS, so oversampling was the
# single largest reason this testbed was not fast.
FS = 50.0
TS = 1.0 / FS
N_SAMPLES = 2000                       # 40 s
BAND_HZ = (0.10, 5.0)                  # covers the main mode AND the absorber at 3.18 Hz


# ============================================================ continuous-time state matrices
def truth_ct():
    """4-state truth: [x1, x2, xd1, xd2]. Absorber included."""
    M = np.diag([M1, M2])
    C = np.array([[C1 + C2, -C2], [-C2, C2]])
    K = np.array([[K1 + K2, -K2], [-K2, K2]])
    Minv = np.linalg.inv(M)
    A = np.block([[np.zeros((2, 2)), np.eye(2)],
                  [-Minv @ K, -Minv @ C]])
    B = np.vstack([np.zeros((2, 1)), Minv @ np.array([[1.0], [0.0]])])
    Cy = np.array([[1.0, 0.0, 0.0, 0.0]])          # y = x1
    return A, B, Cy


def baseline_ct(theta):
    """2-state baseline [x, xd] from theta = (m, c, ka, kb). torch, differentiable."""
    m, c, ka, kb = theta[0], theta[1], theta[2], theta[3]
    k = ka + kb                                     # only the SUM enters: exact null direction
    z = torch.zeros((), dtype=theta.dtype)
    o = torch.ones((), dtype=theta.dtype)
    A = torch.stack([torch.stack([z, o]),
                     torch.stack([-k / m, -c / m])])
    B = torch.stack([torch.stack([z]), torch.stack([o / m])])
    return A, B


def zoh(A, B, ts):
    """Exact ZOH discretisation via one matrix exponential. Differentiable in torch."""
    n, p = A.shape[0], B.shape[1]
    Mblk = torch.zeros((n + p, n + p), dtype=A.dtype)
    Mblk[:n, :n] = A * ts
    Mblk[:n, n:] = B * ts
    E = torch.linalg.matrix_exp(Mblk)
    return E[:n, :n], E[:n, n:]


def baseline_dt(theta, ts=TS):
    return zoh(*baseline_ct(theta), ts)


# ============================================================================ data generation
def multisine(n, fs, band, seed=0, rms=1.0):
    """Flat-amplitude random-phase multisine over `band`. Deterministic given the seed."""
    rng = np.random.default_rng(seed)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    U = np.zeros(len(f), dtype=complex)
    sel = (f >= band[0]) & (f <= band[1])
    U[sel] = np.exp(2j * np.pi * rng.random(sel.sum()))
    u = np.fft.irfft(U, n)
    return (u / u.std() * rms).astype(np.float64)


def simulate_truth(u, ts=TS):
    """Noiseless ZOH simulation of the 4-state truth. Returns (y, x_full)."""
    A, B, Cy = truth_ct()
    At, Bt = zoh(torch.tensor(A, dtype=F64), torch.tensor(B, dtype=F64), ts)
    At, Bt = At.numpy(), Bt.numpy()
    x = np.zeros(4)
    X = np.empty((len(u), 4))
    for k in range(len(u)):
        X[k] = x
        x = At @ x + Bt @ np.array([u[k]])
    return (X @ Cy.T).ravel(), X


def simulate_baseline(u, theta, ts=TS):
    """Noiseless baseline simulation at theta. Returns (y, x)."""
    th = torch.tensor(theta, dtype=F64) if not torch.is_tensor(theta) else theta
    Ad, Bd = baseline_dt(th, ts)
    Ad, Bd = Ad.detach().numpy(), Bd.detach().numpy()
    x = np.zeros(2)
    X = np.empty((len(u), 2))
    for k in range(len(u)):
        X[k] = x
        x = Ad @ x + Bd @ np.array([u[k]])
    return X[:, 0].copy(), X


def baseline_states(u, theta=None, ts=TS):
    """Baseline states along the WHOLE record, for encoder targets at arbitrary offsets.

    BUG FIXED 2026-09-07. Callers previously wrote `simulate_baseline(u[k0:k0+1], ...)[1][0]`
    to get the state at sample k0. That restarts the simulation from rest and runs one step,
    so it returns the ZERO initial state for every k0, not the state at k0. The encoder's
    least-squares targets were therefore all zero, and the "frozen encoder" experiment froze a
    zero encoder rather than a reconstructed initial state. Use this function instead.
    """
    return simulate_baseline(u, THETA_TRUE if theta is None else theta, ts)[1]


def make_dataset(seed=0, n=N_SAMPLES):
    """One record: input, truth output, truth states, and the baseline-at-truth output.

    `delta` is the discrepancy the augmentation must explain. It exists ONLY here, never on
    the gantry and never on Telica, which is exactly why this folder is a development tool.
    """
    u = multisine(n, FS, BAND_HZ, seed=seed)
    y, X = simulate_truth(u)
    y_base, _ = simulate_baseline(u, THETA_TRUE)
    return dict(u=u, y=y, x_truth=X, y_base=y_base, delta=y - y_base)


# ================================================================================= reporting
def _fmt(v):
    return '  '.join(f'{x:10.3e}' for x in v)


def main():
    torch.set_default_dtype(F64)
    print('=' * 78)
    print('TESTBED: 1-DOF baseline with a hidden absorber. Category: development tool.')
    print('=' * 78)

    d = make_dataset(seed=0)
    print(f'  fs = {FS:.0f} Hz, {len(d["u"])} samples ({len(d["u"])/FS:.0f} s), '
          f'band {BAND_HZ[0]}-{BAND_HZ[1]} Hz, noiseless')
    print(f'  main mode  {np.sqrt(K1/M1)/2/np.pi:.2f} Hz (zeta {C1/(2*np.sqrt(K1*M1)):.3f})')
    print(f'  absorber   {np.sqrt(K2/M2)/2/np.pi:.2f} Hz, mass ratio {M2/M1:.1%}')
    print(f'  y std      {d["y"].std():.4e}')
    print(f'  discrepancy std / y std = {d["delta"].std()/d["y"].std():.4f}'
          '   <- how much the hidden absorber matters')

    # Property 3: the exact null direction must annihilate the model.
    th = torch.tensor(THETA_TRUE, dtype=F64, requires_grad=True)
    Ad, Bd = baseline_dt(th)
    J = []
    for i in range(2):
        for j in range(2):
            g = torch.autograd.grad(Ad[i, j], th, retain_graph=True)[0]
            J.append(g)
    for i in range(2):
        g = torch.autograd.grad(Bd[i, 0], th, retain_graph=True)[0]
        J.append(g)
    J = torch.stack(J).numpy()
    v_null = np.array([0.0, 0.0, 1.0, -1.0])
    resid = np.linalg.norm(J @ v_null)
    print(f'\n  exact null direction (0,0,1,-1):  ||J v|| = {resid:.3e}   '
          f'{"OK" if resid < 1e-10 else "FAILED"}')
    s = np.linalg.svd(J, compute_uv=False)
    print(f'  one-step Jacobian singular values: {_fmt(s)}')
    print(f'  numerical rank (rtol 1e-12): {int((s > 1e-12 * s[0]).sum())} of 4')
    print('\n  Note: this is the ONE-STEP map. The graded/weak direction is a property of the')
    print('  windowed output sensitivity under this excitation, measured in geometry.py.')


if __name__ == '__main__':
    main()
