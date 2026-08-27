"""Step 3: place the augmented poles by IDENTIFYING the residual the baseline leaves.

    (rho, theta) = identify_poles(u, r, ts, n_pairs)

WHY THIS EXISTS. The training objective FREEZES the augmented poles: with the true mode planted,
`dL/d(nu_log) < 0` on 7 of 8 disjoint batches and monotone over 150 steps (C6); no non-negative
residual weighting can flip that sign (T3); and both trained arms moved their poles under `0.15 Hz`
over 520 updates. So the block is a fixed basis whose SPAN decides the outcome, and whatever is
installed at initialisation is what is kept. Placement is therefore not a hyperparameter, it is the
experiment, and it has to come from the data.

    # THEORY: Schoukens, ECC 2021 (schoukens2021ssnn_init), Sec. IV.3. Verified quote (MATCH OK,
    # 2026-08-23): "The linear state-space matrices are directly used to initialize the A, B, C, D
    # matrices in eq. (5)." We take only A (the poles); see the departure note below.

WHICH RESIDUAL. `r = y - y_baseline`, formed in the CLOSED loop the machine actually runs, which is
what `cl_residual_spectrum.residual_for` produces from logged `u`, logged `y` and the baseline
alone. NOT the open-loop `rho = y - P0 u`: measured `1.224e-04` against `2.187e-06`, a factor 56,
and that factor is almost entirely low-frequency motion mismatch the loop suppresses by `47x` to
`2800x` while AMPLIFYING the absorber band by `1.81x` (`loop_sensitivity.py`). Identifying the
open-loop residual points the fit at the part of the spectrum the objective never sees.

This module takes `u` and `r` as ARRAYS and never forms them. That keeps it pure and testable, and
it is what makes the same code run on Telica, where the residual comes from a different source and
the excitation is a non-periodic point-to-point move.

WHY PARAMETRIC AND NOT A BLA. A BLA needs periodic excitation with realisations
(`pintelon2020_bla-feedback-process-noise`: "specially designed periodic excitation signals called
random phase multisines and periodic noise"). Telica has a jerk-limited move with `Y` sweeping
inside every record, so the plant is not even LTI over it. A parametric fit needs none of that.
Measured comparison on `T3_standstill_Y000` against the plant's `158.1139 Hz`, `zeta 0.05`:
parametric ARX `-0.079 %` / `+4.9 %`, peak picking `+0.056 %` / `+16.6 %`, FRF ratio `-0.82 %` /
`-13.3 %`. Peak picking's damping bias is the input spectrum's shape leaking into a half-power
width; damping sets `rho`, which is the memory ingredient, so it matters.

DEPARTURE FROM SCHOUKENS, one-directional. He initialises `A, B, C, D` from the linear model. We
take only the POLES and keep `B_u` random per `hoekstra2026lfrfp` p10, because D-158 refuses the
fitted input map as `NONCAUSAL_IDENTIFICATION_COORDINATES` and the `[u, x_b]` regressor split is
unidentifiable in open loop. We use LESS of the identification than he does, never more.

REFUSALS. This module refuses rather than substituting a default, because a default would silently
decide the run: too few stable complex pairs, an unstable fit, or an order sweep in which nothing
beats the zero predictor out of sample.
"""
__project_origin__ = "added"

import math
from typing import Optional, Sequence, Tuple

import numpy as np


class PoleIdentificationRefusal(Exception):
    """Raised instead of returning a default set of poles."""


# ── the fit ──────────────────────────────────────────────────────────────────────────────────

def fit_shared_denominator(u: np.ndarray, r: np.ndarray, na: int, nb: int):
    """One scalar `A(q)` shared by every channel, with a per-channel-per-input numerator.

        r_c(k) + sum_{p=1..na} a_p r_c(k-p) = sum_j sum_{l=1..nb} b_{cjl} u_j(k-l)

    A SHARED denominator is what makes one pole set exist at all. Fitting an independent `A_j(q)`
    per channel gives one pole set per channel, and an augmented block has exactly one.

    Column scaling before the solve: `u` is in newtons and `r` in metres, so the two blocks differ
    by roughly eight decades and an unscaled normal-equation solve is rank deficient in float64.
    The scaling is exact and undone afterwards; it changes conditioning, not the estimate.
    """
    u = np.asarray(u, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    n, nch = len(r), r.shape[1]
    nin = u.shape[1]
    m = max(na, nb)
    if n <= m + 10:
        raise PoleIdentificationRefusal('record of %d samples is too short for na=%d' % (n, na))
    rows, rhs = [], []
    for c in range(nch):
        Ra = -np.stack([r[m - p:n - p, c] for p in range(1, na + 1)], axis=1)
        Rb = np.zeros((n - m, nch * nin * nb))
        for j in range(nin):
            for l in range(1, nb + 1):
                Rb[:, c * nin * nb + j * nb + (l - 1)] = u[m - l:n - l, j]
        rows.append(np.hstack([Ra, Rb]))
        rhs.append(r[m:n, c])
    Phi, Y = np.vstack(rows), np.concatenate(rhs)
    s = np.linalg.norm(Phi, axis=0)
    s[s == 0] = 1.0
    theta, *_ = np.linalg.lstsq(Phi / s, Y, rcond=None)
    theta = theta / s
    return theta[:na], theta[na:].reshape(nch, nin, nb)


def _free_run_vaf(u, r, a, B, na, nb):
    """Out-of-sample FREE-RUN VAF of a fitted `(a, B)`: the order criterion D-159 specifies.

    Free run means the model is driven by `u` alone and its own past OUTPUT, never by the measured
    `r`. That is the difference that matters here. One-step prediction is handed the true `r(k-p)`
    at every step, so a broad, near-memoryless pole that merely tracks the previous sample scores
    well; over a free run the same pole contributes nothing and is penalised. Measured consequence
    on `T3_standstill_Y000`: one-step MSE selects `na = 28`, where the absorber ranks SECOND behind
    a `141 Hz, zeta 0.465` catch-all, which would make an `n_pairs = 1` arm install the wrong pole.

    VAF = 100 * (1 - var(r - r_hat) / var(r)), per channel, then averaged. Higher is better; a
    model that cannot beat the mean scores at or below 0.
    """
    n, nch = len(r), r.shape[1]
    nin = u.shape[1]
    m = max(na, nb)
    rhat = np.zeros((n, nch))
    for c in range(nch):
        # exogenous part first: it does not depend on the simulated output
        ex = np.zeros(n)
        for j in range(nin):
            for l in range(1, nb + 1):
                ex[m:] += B[c, j, l - 1] * u[m - l:n - l, j]
        for k in range(m, n):
            acc = ex[k]
            for p in range(1, na + 1):
                acc -= a[p - 1] * rhat[k - p, c]        # own past output, NOT the measured r
            rhat[k, c] = acc
    e, y = r[m:] - rhat[m:], r[m:]
    if not np.all(np.isfinite(rhat)):
        return -np.inf
    vaf = [100.0 * (1.0 - np.var(e[:, c]) / max(np.var(y[:, c]), 1e-300)) for c in range(nch)]
    return float(np.mean(vaf))


# ── modal selection ──────────────────────────────────────────────────────────────────────────

def _modal_pairs(a: np.ndarray, B: np.ndarray, nb: int):
    """Stable complex-conjugate pairs of `A(q)`, ranked by modal energy. Largest first.

    # THEORY: modal truncation (Antoulas 2005). D-159 replaced balanced singular perturbation with
    # modal selection because BSP returns a Schur complement whose poles are NOT a subset of the
    # unreduced spectrum: measured, it retained the 157.99 Hz mode at NO tested order, collapsing
    # it to 5.04 Hz. Modal selection keeps the identified pair by construction.

    Energy of pole `lam` is `|residue|^2 / (1 - |lam|^2)`, i.e. the stationary contribution of that
    mode, with the residue read off the partial-fraction expansion `B(lam) / A'(lam)` summed in
    quadrature over channels and inputs.
    """
    poly = np.concatenate([[1.0], np.asarray(a, dtype=np.float64)])
    lam = np.roots(poly)
    dpoly = np.polyder(poly)
    out = []
    seen = np.zeros(len(lam), dtype=bool)
    for i, li in enumerate(lam):
        if seen[i] or abs(li.imag) < 1e-12:
            continue                       # real pole: no rotation-pair realisation (D-159 refusal)
        if not (0.0 < abs(li) < 1.0):
            continue                       # unstable or degenerate
        # mark the conjugate so a pair is counted once
        j = int(np.argmin(np.abs(lam - np.conj(li))))
        seen[i] = seen[j] = True
        num = 0.0
        for c in range(B.shape[0]):
            for k in range(B.shape[1]):
                num += abs(np.polyval(B[c, k, ::-1], li)) ** 2
        denom = abs(np.polyval(dpoly, li)) ** 2
        if denom == 0.0:
            continue
        energy = (num / denom) / (1.0 - abs(li) ** 2)
        out.append((float(energy), complex(li)))
    out.sort(key=lambda t: -t[0])
    return out


# ── the entry point ──────────────────────────────────────────────────────────────────────────

def identify_poles(u, r, ts: float, n_pairs: int,
                   na_range: Sequence[int] = tuple(range(12, 29, 2)),
                   nb: Optional[int] = None, split: float = 0.5
                   ) -> Tuple[np.ndarray, np.ndarray, dict]:
    """`(rho, theta, info)` for `n_pairs` augmented pole pairs, at the model rate.

    Order is chosen out of sample by FREE-RUN VAF, which is the criterion D-159 and `fit_reduce.py`
    specify: fit on the first `split` of the record, free-run the model on the rest, keep the
    highest VAF. Ties go to the smaller order. Oracle-free; nothing here sees the plant.
    """
    u = np.asarray(u, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    if u.ndim != 2 or r.ndim != 2 or len(u) != len(r):
        raise ValueError('u and r must both be (N, k) with equal N')
    cut = int(len(r) * split)
    best = None
    tried = []
    for na in sorted(na_range):
        nb_ = na if nb is None else nb
        try:
            a, B = fit_shared_denominator(u[:cut], r[:cut], na, nb_)
        except PoleIdentificationRefusal:
            continue
        if np.max(np.abs(np.roots(np.concatenate([[1.0], a])))) >= 1.0:
            tried.append({'na': na, 'vaf': None, 'reason': 'unstable fit'})
            continue
        vaf = _free_run_vaf(u[cut:], r[cut:], a, B, na, nb_)
        tried.append({'na': na, 'vaf': vaf})
        if best is None or vaf > best[0]:        # strict >, so ties keep the SMALLER order
            best = (vaf, na, a, B, nb_)
    if best is None:
        raise PoleIdentificationRefusal(
            'no stable model at any order in %s; refusing rather than defaulting' % (na_range,))
    vaf, na, a, B, nb_ = best
    # a model that cannot beat the zero predictor has identified nothing. This is the refusal
    # fit_reduce.py already hits on its noisy condition (out-of-sample VAF -0.0136, D-159).
    if not (vaf > 0.0):
        raise PoleIdentificationRefusal(
            'best out-of-sample free-run VAF %.4f does not beat the zero predictor; the residual '
            'is not a linear object on this data' % vaf)

    pairs = _modal_pairs(a, B, nb_)
    if len(pairs) < n_pairs:
        raise PoleIdentificationRefusal(
            'need %d stable complex pairs, the fit at na=%d has %d; refusing rather than '
            'replicating a pole (replication adds parameters, not span)' % (n_pairs, na, len(pairs)))
    keep = pairs[:n_pairs]
    rho = np.array([abs(l) for _, l in keep], dtype=np.float64)
    theta = np.array([abs(np.angle(l)) for _, l in keep], dtype=np.float64)
    info = {
        'na': int(na), 'nb': int(nb_), 'free_run_vaf': vaf, 'orders_tried': tried,
        'modes': [{'rho': float(abs(l)),
                   'f_d_hz': float(abs(np.angle(l)) / (2 * math.pi * ts)),
                   'zeta': float(-np.log(l).real / abs(np.log(l))),
                   'modal_energy': float(e)} for e, l in keep],
        'n_pairs_available': len(pairs),
    }
    return rho, theta, info
