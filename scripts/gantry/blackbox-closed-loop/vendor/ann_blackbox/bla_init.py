"""Best Linear Approximation initialisation of the SUBNET, per Ramkannan et al. (2023).

  Ramkannan, R., Beintema, G. I., Toth, R., Schoukens, M., "Initialization Approach for Nonlinear
  State-Space Identification via the Subspace Encoder Approach", IFAC-PapersOnLine 56(2):5146-5151,
  2023, DOI 10.1016/j.ifacol.2023.10.010 (CC BY-NC-ND). Their conclusion, verbatim: "The state-space
  matrices of the linear approximate model are used as a linear bypass in the neural networks that
  represent the state and output equations. ... Hence the reconstructability map of the linear
  approximate model is used to initialize the encoder network."

STILL A BLACK BOX. The BLA is estimated from input-output data alone by N4SID. No truth model, no
physical parameters, no coordinate assumptions. The linear model is consumed at initialisation;
after that there is one network and nothing carries a physical claim, so there is nothing for the
nonlinear part to negate in the sense that matters for augmentation.

Their arms, named as in the paper, are reproduced here:
  RanDY + RanENC   deepSI's own random init (the control)
  LinDY + RanENC   --bla dyn      BLA into the state and output nets only
  LinDY + LinENC   --bla full     the above plus the encoder from the reconstructability map

THE RECONSTRUCTABILITY MAP, derived here rather than cited, because the paper states it for its own
notation. With D = 0 and n = na = nb past samples, stacking Y = [y[k-n] ... y[k-1]] gives

    Y = O_n x[k-n] + T U,      O_n = [C; CA; ...; CA^(n-1)],  T = block-Toeplitz of C A^i B
    x[k-n] = pinv(O_n) (Y - T U)
    x[k]   = A^n x[k-n] + Cc U,   Cc = [A^(n-1) B, ..., A B, B]

so  x[k] = W_y Y + W_u U  with  W_y = A^n pinv(O_n)  and  W_u = Cc - A^n pinv(O_n) T.
n = 17 > nx = 8, so O_n is tall and pinv is the least-squares reconstruction, which is what makes
the encoder a minimum-variance observer rather than a bare inverse.

TIME ALIGNMENT, verified against deepSI rather than assumed: `to_hist_future_data` returns
uhist = u[k-nb:k], yhist = y[k-na:k], and `SS_encoder_general_hf.loss` computes yhat, x = hfn(x, u)
with the first yhat compared against yfuture[0] = y[k]. So the encoder must return x[k] with
y[k] = C x[k], which is exactly the x[k] derived above.

NORMALISATION: the nets operate on deepSI's normalised u and y, so the BLA is fitted on
norm.transform(train_data), not on raw data. Fitting it on raw data would put the bypass weights
off by the normalisation scaling and is the obvious way to get this silently wrong.
"""
__project_origin__ = "added"

import os

import numpy as np
import torch
from torch import nn


SS_F_GRID = (40, 30, 50, 60, 20, 80)


def fit_bla(fit_sys, train_data, nx, verbose=True):
    """N4SID on the NORMALISED training data. Returns the deepSI SS_linear model.

    SS_f is SELECTED, not fixed, and this is load-bearing. Measured on T10 at 800 Hz, nx=8:
    SS_f=20 (deepSI's default) gives max|eig(A)| = 1.41 and SS_f=80 gives 1.34, both unstable, and
    deepSI's own post-fit state rescaling then raises 'x exploded'. SS_f=40 gives max|eig| =
    0.999995 with the two closest poles 4.6e-6 and 1.4e-5 from z=1, i.e. it captures the
    integrators. So the grid is swept and the arm chosen on the model's own fit cost Vn among
    those whose A is stable, which uses no knowledge of the truth.

    SS_A_stability is left OFF deliberately. Forcing it drags the poles to 0.90-0.975, i.e.
    |eig - 1| of 0.03 to 0.12, which is five orders worse than the 8.4e-8 the acceptance bar needs
    on a 9600-step free run. A "stable" BLA is the wrong model for a plant with free integrators.
    """
    from deepSI.fit_systems.ss_linear import SS_linear
    norm_data = fit_sys.norm.transform(train_data)
    best = None
    for f in SS_F_GRID:
        bla = SS_linear(nx=nx, feedthrough=False)
        try:
            bla._fit(norm_data, SS_f=f)
        except (AssertionError, np.linalg.LinAlgError, ValueError) as e:
            # AssertionError = deepSI's own 'x exploded' guard; LinAlgError = singular matrix in the
            # rescaling or the realisation. Both mean this SS_f produced an unusable model.
            if verbose: print(f'[BLA] SS_f={f:3d}  rejected: {type(e).__name__}')
            continue
        mx = float(np.max(np.abs(np.linalg.eigvals(np.asarray(bla.A)))))
        vn = float(bla.model.Vn) if np.isfinite(bla.model.Vn) else np.inf
        ok = mx < 1.0 and np.isfinite(vn)
        if verbose: print(f'[BLA] SS_f={f:3d}  max|eig|={mx:.6f}  Vn={vn:.3e}  {"ok" if ok else "rejected"}')
        if ok and (best is None or vn < best[0]):
            best = (vn, f, bla)
    assert best is not None, 'no SS_f in the grid produced a stable BLA'
    if verbose: print(f'[BLA] selected SS_f={best[1]} on lowest Vn={best[0]:.3e}')
    best[2].fitted = True
    return best[2]


def reconstructability_map(A, B, C, n):
    """W_u, W_y such that x[k] = W_u @ U + W_y @ Y, with U, Y the flattened past windows."""
    nx, nu = B.shape
    ny = C.shape[0]
    O = np.vstack([C @ np.linalg.matrix_power(A, i) for i in range(n)])          # (n*ny, nx)
    T = np.zeros((n * ny, n * nu))                                              # block lower triangular
    for i in range(n):
        for j in range(i):
            T[i*ny:(i+1)*ny, j*nu:(j+1)*nu] = C @ np.linalg.matrix_power(A, i - j - 1) @ B
    Cc = np.hstack([np.linalg.matrix_power(A, n - 1 - j) @ B for j in range(n)])  # (nx, n*nu)
    An_Opinv = np.linalg.matrix_power(A, n) @ np.linalg.pinv(O)
    return Cc - An_Opinv @ T, An_Opinv


def encoder_map_ridge(A, B, C, Un, Yn, na, n_ls=2000, lam_grid=(1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)):
    """Encoder weights by RIDGE REGRESSION of the BLA's own state trajectory on the past window.

    WHY NOT THE ANALYTIC MAP. `reconstructability_map` above is algebraically correct and was
    verified to reproduce the training record's output to 3.0e-05 normalised. It is nonetheless
    unusable here, because it contains pinv(O_n) and O_n is nearly rank-deficient in exactly the
    directions we care about: over a 17-sample window at 800 Hz an integrator mode changes by
    (1 - 1.755e-5)^17 = 0.9997, so those singular values are tiny and 1/sigma amplifies any model
    mismatch. Measured on V2, the analytic map returned a state component of -567 where the optimal
    was order 1, and the resulting epoch-0 sim-RMS was 5.41e-01, worse than random init's 1.61e-01.

    Ridge regression is the same object obtained by a well-conditioned route: it trades exactness in
    the weakly observed directions for bounded gain, which is the minimum-variance-observer
    behaviour Beintema et al. report when the lag exceeds the theoretical minimum (Automatica
    156:111210, Section 5.6). lambda is chosen on a held-out split of the TRAINING record; the
    validation record is never touched.
    """
    nx = A.shape[0]
    # A state trajectory consistent with the data: solve for x0 by least squares, then simulate.
    # x0 = 0 would be wrong here, since with |eig| ~ 1 the initial-condition error never decays.
    def sim(x0, U):
        x, out = np.asarray(x0, float).copy(), np.empty((len(U), nx))
        for k in range(len(U)):
            out[k] = x
            x = A @ x + B @ U[k]
        return out
    forced = sim(np.zeros(nx), Un)
    Phi0 = np.vstack([C @ np.linalg.matrix_power(A, k) for k in range(n_ls)])
    x0, *_ = np.linalg.lstsq(Phi0, (Yn[:n_ls] - (forced[:n_ls] @ C.T)).reshape(-1), rcond=None)
    X = sim(x0, Un)

    n = len(Un) - na
    idx = np.arange(na, na + n)
    Phi = np.hstack([Un[i:i + n] for i in range(na)] + [Yn[i:i + n] for i in range(na)])
    Phi = np.hstack([Phi, np.ones((n, 1))])                       # intercept
    Xt = X[idx]
    cut = int(0.8 * n)                                            # held-out split of the TRAIN record
    Ptr, Xtr, Pva, Xva = Phi[:cut], Xt[:cut], Phi[cut:], Xt[cut:]
    G, rhs = Ptr.T @ Ptr, Ptr.T @ Xtr
    I = np.eye(G.shape[0]); I[-1, -1] = 0.0                       # do not penalise the intercept
    best = None
    for lam in lam_grid:
        W = np.linalg.solve(G + lam * len(Ptr) * I, rhs)
        e = float(np.sqrt(np.mean((Xva - Pva @ W) ** 2)))
        if best is None or e < best[0]:
            best = (e, lam, W)
    e, lam, _ = best
    G, rhs = Phi.T @ Phi, Phi.T @ Xt                              # refit on the full record at that lam
    W = np.linalg.solve(G + lam * len(Phi) * I, rhs)
    return W[:-1].T, W[-1], lam, e


def _zero_nonlinear(net):
    """Zero the last layer of the nonlinear branch so the module starts EXACTLY at its bypass."""
    if net.net_non_lin is None:
        return
    last = [m for m in net.net_non_lin.modules() if isinstance(m, nn.Linear)][-1]
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)


def apply_bla_init(fit_sys, train_data, mode='full', zero_nonlinear=False, verbose=True):
    """Write the BLA into the linear bypasses. Call AFTER fit_sys.init_model, BEFORE fit.

    mode: 'dyn'  = LinDY + RanENC     'full' = LinDY + LinENC
    zero_nonlinear: if True the nonlinear branches start at zero, so epoch 0 IS the BLA exactly.
      The paper leaves them at their random init; this switch exists because a plant with poles at
      z = 1 is far more sensitive to a random perturbation than the Wiener-Hammerstein system they
      demonstrate on, and the difference is worth measuring rather than assuming.
    """
    nx = fit_sys.nx
    bla = fit_bla(fit_sys, train_data, nx)
    A, B, C = np.asarray(bla.A), np.asarray(bla.B), np.asarray(bla.C)
    if verbose:
        ev = np.linalg.eigvals(A)
        print(f'[BLA] nx={nx}  |eig| = {np.array2string(np.sort(np.abs(ev))[::-1], precision=6)}')
        print(f'[BLA] closest pole to z=1: {np.min(np.abs(ev - 1.0)):.3e}')

    with torch.no_grad():
        # --- state equation: net_lin over [x, u] -> x_next, so the weight is [A | B] -----------
        fn = fit_sys.hfn.fn.net
        W = np.hstack([A, B])
        assert fn.net_lin.weight.shape == W.shape, (fn.net_lin.weight.shape, W.shape)
        fn.net_lin.weight.copy_(torch.tensor(W, dtype=torch.float32))
        fn.net_lin.bias.zero_()
        if zero_nonlinear:
            _zero_nonlinear(fn)

        # --- output equation: net_lin over x -> y, no feedthrough, so the weight is C ---------
        hn = fit_sys.hfn.hn.net
        assert hn.net_lin.weight.shape == C.shape, (hn.net_lin.weight.shape, C.shape)
        hn.net_lin.weight.copy_(torch.tensor(C, dtype=torch.float32))
        hn.net_lin.bias.zero_()
        if zero_nonlinear:
            _zero_nonlinear(hn)

        # --- encoder: net_lin over [U_flat, Y_flat] -> x[k] ------------------------------------
        if mode == 'full':
            na, nb = fit_sys.na, fit_sys.nb
            assert na == nb, 'the derivation above assumes one past-window length'
            nd = fit_sys.norm.transform(train_data)
            Wenc, benc, lam, e = encoder_map_ridge(A, B, C, np.asarray(nd.u), np.asarray(nd.y), na)
            if verbose:
                print(f'[BLA] encoder ridge: lambda={lam:.1e}, held-out state RMS={e:.3e}, '
                      f'max|W|={np.max(np.abs(Wenc)):.3e}')
            en = fit_sys.encoder.net
            assert en.net_lin.weight.shape == Wenc.shape, (en.net_lin.weight.shape, Wenc.shape)
            en.net_lin.weight.copy_(torch.tensor(Wenc, dtype=torch.float32))
            en.net_lin.bias.copy_(torch.tensor(benc, dtype=torch.float32))
            if zero_nonlinear:
                _zero_nonlinear(en)
    if verbose:
        print(f'[BLA] applied mode={mode}, zero_nonlinear={zero_nonlinear}')
    return bla


def apply_frf_bla_init(fit_sys, train_data, ss_path=None, zero_nonlinear=True, verbose=True):
    """Write the FREQUENCY-DOMAIN BLA into the linear bypasses. Same contract as `apply_bla_init`.

    WHY A SECOND ESTIMATOR. `fit_bla` above runs N4SID on the training record. Measured against
    the frozen truth at 800 Hz, its poles miss BOTH resonances: the 157.9 Hz absorber by 9.19e-01
    in the z-plane and the 5.12 Hz pair by 4.02e-02 (`bla_decimation_test.py`). The cause is the
    estimator, not the data: N4SID minimises a time-domain prediction error dominated by the
    low-frequency output energy, and the absorber carries ~1e-8 of it (`frf_diagnostic.py`), so it
    is truncated away. A frequency-domain fit normalises per line, so a mode with negligible
    energy but a clear FRF signature survives.

    The model written here comes from `frf_init.py`: LPM FRF of a standstill multisine record ->
    vector fitting for the poles -> linear residue fit in s -> realisation -> warm-started B,C
    refinement -> one discretisation. Worst pole error over all eight: 7.92e-03, max|eig| exactly
    1.000000, FRF fit 0.83 % median.

    INPUT CLASS, stated because it is the one substantive difference from `fit_bla`. A BLA is
    defined for an input class, and this one is estimated from a STANDSTILL MULTISINE experiment
    rather than from the trajectory records the network trains on. That is a deliberate choice of
    identification experiment, and it is why it works: standstill records are exactly periodic
    (no leakage) and sit at one fixed Y, so an 8-pole model fits their FRF to 0.5 %, whereas NO
    model of ANY order fits the trajectory-record FRF better than 50 %, because those records
    sweep Y over +-0.29 m and the plant is genuinely time-varying during them.

    NORMALISATION. deepSI's nets see normalised u and y, so the physical-unit model is converted
    as A_n = A, B_n = B diag(ustd), C_n = diag(1/ystd) C, with zero bias because the FRF is
    estimated on mean-removed data and therefore describes deviations. The states are then
    rescaled to unit standard deviation on the training input, following Schoukens, ECC 2021
    (arXiv:2103.14516), which normalises the linear model so each state has unit std before
    writing it into the bypass.
    """
    import numpy as _np
    if ss_path is None:
        ss_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'results', 'frf_init', 'frf_init_ss.npz')
    d = _np.load(ss_path)
    A, B, C = _np.asarray(d['A'], float), _np.asarray(d['B'], float), _np.asarray(d['C'], float)
    nx, nu = B.shape
    ny = C.shape[0]
    assert nx == fit_sys.nx, f'model order {nx} != fit_sys.nx {fit_sys.nx}'

    def _vec(attr, n):
        return _np.broadcast_to(_np.asarray(getattr(fit_sys.norm, attr), float), (n,)).copy()
    Bn = B @ _np.diag(_vec('ustd', nu))
    Cn = _np.diag(1.0 / _vec('ystd', ny)) @ C

    nd = fit_sys.norm.transform(train_data)
    U = _np.asarray(nd.u, float)
    x = _np.zeros(nx); X = _np.empty((len(U), nx))
    for k in range(len(U)):
        X[k] = x
        x = A @ x + Bn @ U[k]
    sd = X.std(axis=0); sd[sd < 1e-12] = 1.0
    T, Ti = _np.diag(sd), _np.diag(1.0 / sd)
    A2, B2, C2 = Ti @ A @ T, Ti @ Bn, Cn @ T

    if verbose:
        ev = _np.linalg.eigvals(A2)
        print(f'[BLA-frf] loaded {os.path.basename(ss_path)}, nx={nx}')
        print(f'[BLA-frf] max|eig| = {_np.max(_np.abs(ev)):.6f}, '
              f'closest pole to z=1: {_np.min(_np.abs(ev - 1.0)):.3e}')
        print(f'[BLA-frf] |eig| = {_np.array2string(_np.sort(_np.abs(ev))[::-1], precision=6)}')

    with torch.no_grad():
        fn = fit_sys.hfn.fn.net
        W = _np.hstack([A2, B2])
        assert fn.net_lin.weight.shape == W.shape, (fn.net_lin.weight.shape, W.shape)
        fn.net_lin.weight.copy_(torch.tensor(W, dtype=torch.float32))
        fn.net_lin.bias.zero_()
        if zero_nonlinear:
            _zero_nonlinear(fn)
        hn = fit_sys.hfn.hn.net
        assert hn.net_lin.weight.shape == C2.shape, (hn.net_lin.weight.shape, C2.shape)
        hn.net_lin.weight.copy_(torch.tensor(C2, dtype=torch.float32))
        hn.net_lin.bias.zero_()
        if zero_nonlinear:
            _zero_nonlinear(hn)
    if verbose:
        print(f'[BLA-frf] applied, zero_nonlinear={zero_nonlinear}')
    return A2, B2, C2
