"""ground_truth.py -- Phase B: the ground-truth and biased-reference DC-direction gradients.

This builds the BIAS reference for the ARTBP verification (README Phase B). No ARTBP, no random
truncation, no reweighting here -- those are the VARIANCE axis (Phase C/D). Phase B answers only:
how biased is a short-window gradient relative to the long-horizon truth, in the DC direction?

The DC direction (established convention, v3b dLoss/d(bias) + v11 landscape): a CONSTANT c added to
the dY state row (index 5) after each rollout step, `x = x + c * e_dY`. The object measured is the
per-step-averaged windowed-loss gradient along that direction, at c=0:

    grad(H) = d/dc [ (1/H) sum_{k=1..H} mean_{B,3ch}( (yhat_k - y_real_k)^2 ) ] |_{c=0}

evaluated by autograd on a single scalar leaf c. Both references are the SAME function grad(H):
  fixed_grad(nf) = grad(nf)   # paper Eq. 9: full BPTT within an independent window of length nf (BIASED)
  true_grad(T)   = grad(T)    # paper Eq. 7-8: full BPTT over the largest feasible horizon T (REFERENCE)
Same window starts and same encoder init for every H, so the ONLY variable is horizon truncation ->
the gap grad(nf) - grad(T) is purely the truncation bias.

Scope (honest): T is the largest FEASIBLE full-BPTT horizon, not the ~48000-step deployment horizon.
grad(T) is ground truth for horizon T (the horizon ARTBP will be unbiased for), and the plateau check
grad(T_CHECK) - grad(T) reports whether T is long enough to be trusted as truth.

Sanity anchor (harness-correctness gate): fixed_grad(400) must reproduce v3b's dLoss/d(bias) dY
~ +2.4e-5 (positive sign, order 1e-5). This is measured at the UNTRAINED zero-ANN baseline vs v3b's
trained-epoch average, so the gate is same-sign + same-order-of-magnitude, not an exact match.

Convention: data -> ./data/b_bias_gap.npz, figure -> ./figures/b_bias_gap.png.
"""
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE   = os.path.dirname(os.path.abspath(__file__))
GANTRY = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(GANTRY, 'drift-demo'))

import demo_common as dm
from demo_common import CFG
from gantry_dynamic.data import TRAIN_FILES

# ── one config surface ─────────────────────────────────────────────────────────
SEED       = int(os.environ.get('GT_SEED', str(CFG.seed)))
B          = int(os.environ.get('GT_B', '64'))            # windows per gradient batch
N_BATCHES  = int(os.environ.get('GT_NBATCH', '8'))        # independent batches -> mean +/- SE over batches
STRIDE     = int(os.environ.get('GT_STRIDE', '20'))       # window-start spacing
H_SWEEP    = [50, 100, 200, 400, 800, 1600, 3200]         # horizons for the biased-window sweep
T_PRIMARY  = int(os.environ.get('GT_TPRIMARY', '3200'))   # primary ground-truth horizon (in H_SWEEP)
T_CHECK    = int(os.environ.get('GT_TCHECK', '6400'))     # plateau-check horizon (>= max H_SWEEP)
PROBE_HS   = [800, 1600, 3200, 6400]                      # memory/compute probe horizons
V3B_ANCHOR = 2.4e-5                                        # v3b dLoss/d(bias) dY (positive); cross-check
IY = 5                                                    # dY state index (state order [X,Th,Y,dX,dTh,dY,delta_a,vdelta_a])

figDir = os.path.join(HERE, 'figures'); os.makedirs(figDir, exist_ok=True)
datDir = os.path.join(HERE, 'data');    os.makedirs(datDir, exist_ok=True)

HMAX = max(max(H_SWEEP), T_CHECK)                         # rollouts cached to this length


def build_window_bank(fit_sys, norm, na, nb, K0):
    """Sample N_BATCHES*B window starts, gather full-length (HMAX) rollout data + encoder init once.

    Returns a list of batches; each batch is (U_full, Y_full, x0) with U_full/Y_full shape
    (B, HMAX, 3) normalized and x0 the DETACHED encoder init (B, 8). Same data reused for every H.
    """
    um = norm.u_mean.flatten()[None, :]; us = norm.std_u.flatten()[None, :]
    y0 = np.asarray(norm.y0).flatten()[None, :]; ys = np.asarray(norm.ystd).flatten()[None, :]

    recs = []
    starts = []
    for ri, f in enumerate(TRAIN_FILES):
        u, y, _, _ = dm.load_T(f, CFG)
        un = ((u - um) / us).astype(np.float32); yn = ((y - y0) / ys).astype(np.float32)
        recs.append((un, yn))
        N = len(un)
        for p in range(max(K0, na, nb), N - HMAX, STRIDE):   # room for HMAX + encoder history
            starts.append((ri, p))
    starts = np.array(starts)
    rng = np.random.default_rng(SEED)
    rng.shuffle(starts)
    need = N_BATCHES * B
    assert len(starts) >= need, f'only {len(starts)} windows, need {need}'
    starts = starts[:need]
    print(f'[bank] {len(starts)} windows ({N_BATCHES} batches x {B}), HMAX={HMAX}, stride={STRIDE}')

    batches = []
    for bi in range(N_BATCHES):
        bs = starts[bi * B:(bi + 1) * B]
        U = np.empty((B, HMAX, 3), np.float32); Y = np.empty((B, HMAX, 3), np.float32)
        UH = np.empty((B, nb + 1, 3), np.float32); YH = np.empty((B, na + 1, 3), np.float32)
        for i, (ri, p) in enumerate(bs):
            un, yn = recs[ri]
            U[i] = un[p:p + HMAX]; Y[i] = yn[p:p + HMAX]
            UH[i] = un[p - nb:p + 1]; YH[i] = yn[p - na:p + 1]
        U = torch.from_numpy(U); Y = torch.from_numpy(Y)
        UH = torch.from_numpy(np.ascontiguousarray(UH)); YH = torch.from_numpy(np.ascontiguousarray(YH))
        with torch.no_grad():
            x0 = fit_sys.encoder(UH.contiguous(), YH.contiguous()).detach()   # (B,8) encoder init
        batches.append((U, Y, x0))
    return batches


def batch_grad(fit_sys, batch, H, eY):
    """DC-direction gradient of the H-step mean-MSE windowed loss on one batch, at c=0.

    Loss reduction is IDENTICAL to v12_artbp.py fixed mode: mean over the H steps of the per-step
    batch/channel MSE. c is injected AFTER each hfn step (v11 convention), so it drives steps 1..H-1.
    """
    U, Y, x0 = batch
    c = torch.zeros(1, requires_grad=True)
    x = x0.clone()
    acc = 0.0
    for t in range(H):
        yhat, x = fit_sys.hfn(x, U[:, t, :])
        acc = acc + torch.mean((Y[:, t, :] - yhat) ** 2)
        x = x + c * eY                                       # THEORY: v11 DC probe, constant on dY row
    loss = acc / H                                           # THEORY: paper Eq. 2 total loss, mean-normalized
    loss.backward()
    return float(c.grad.item()), float(loss.item())


def grad_over_batches(fit_sys, batches, H, eY):
    """Return (grads array over batches, mean-loss). grad mean +/- SE = mean/std over N_BATCHES."""
    gs, ls = [], []
    for batch in batches:
        g, l = batch_grad(fit_sys, batch, H, eY)
        gs.append(g); ls.append(l)
    return np.array(gs), float(np.mean(ls))


def main():
    print(f'ground_truth (Phase B) | seed={SEED} | B={B} x {N_BATCHES} batches | '
          f'H_SWEEP={H_SWEEP} | T_PRIMARY={T_PRIMARY} T_CHECK={T_CHECK}')
    fit_sys, norm, K0, na, nb, na_right, nb_right = dm.build_pipeline(cfg=CFG, verbose=True)

    # Freeze all model params: only the scalar c carries grad, so backward computes just the c-path
    # (leaner graph -> larger feasible T). Phase B never trains.
    nfz = 0
    for p in list(fit_sys.hfn.parameters()) + list(fit_sys.encoder.parameters()):
        p.requires_grad_(False); nfz += 1
    print(f'[freeze] {nfz} param tensors frozen (no training in Phase B)')

    eY = torch.zeros(1, 8); eY[0, IY] = 1.0
    batches = build_window_bank(fit_sys, norm, na, nb, K0)

    # ── memory/compute probe: time one batch at increasing H, pick/confirm T ───────
    print('\n[probe] one-batch full-BPTT timing (grad in the DC direction):')
    for H in PROBE_HS:
        t0 = time.time()
        g, l = batch_grad(fit_sys, batches[0], H, eY)
        dt = time.time() - t0
        print(f'  H={H:5d} | {dt:6.2f} s/batch | grad={g:+.3e} loss={l:.3e}')

    # ── the sweep: grad(H) over all batches for every H (biased windows + T_PRIMARY) ─
    print('\n[sweep] grad(H) mean +/- SE over batches:')
    sweep = {}
    for H in H_SWEEP:
        gs, l = grad_over_batches(fit_sys, batches, H, eY)
        sweep[H] = gs
        se = gs.std(ddof=1) / np.sqrt(len(gs))
        tag = '  <- T_PRIMARY (ground truth)' if H == T_PRIMARY else ''
        print(f'  grad({H:5d}) = {gs.mean():+.4e} +/- {se:.2e}{tag}')

    # plateau check: grad at the longer T_CHECK horizon
    gs_check, l_check = grad_over_batches(fit_sys, batches, T_CHECK, eY)
    se_check = gs_check.std(ddof=1) / np.sqrt(len(gs_check))
    g_T = sweep[T_PRIMARY].mean(); se_T = sweep[T_PRIMARY].std(ddof=1) / np.sqrt(N_BATCHES)
    plateau = gs_check.mean() - g_T
    print(f'  grad({T_CHECK:5d}) = {gs_check.mean():+.4e} +/- {se_check:.2e}  (plateau check)')
    print(f'\n[plateau] grad(T_CHECK) - grad(T_PRIMARY) = {plateau:+.3e} '
          f'(vs nf=400 bias {sweep[400].mean() - g_T:+.3e}); '
          f'{"T ok" if abs(plateau) < 0.25 * abs(sweep[400].mean() - g_T) else "T short: bias gap is a LOWER bound"}')

    # 1/H asymptote (secondary, clearly labeled): fit grad(H) = g_inf + a/H on H >= 400
    Hs_fit = np.array([H for H in H_SWEEP if H >= 400] + [T_CHECK], float)
    gs_fit = np.array([sweep[H].mean() for H in H_SWEEP if H >= 400] + [gs_check.mean()])
    A = np.vstack([np.ones_like(Hs_fit), 1.0 / Hs_fit]).T
    (g_inf, a_coef), *_ = np.linalg.lstsq(A, gs_fit, rcond=None)
    print(f'[extrap] 1/H fit -> g_inf = {g_inf:+.3e}, a = {a_coef:+.3e}  (secondary, not the truth)')

    # ── sanity anchor: fixed_grad(400) vs v3b +2.4e-5 ─────────────────────────────
    g400 = sweep[400].mean()
    ok_sign = g400 > 0
    ok_order = 3e-6 < abs(g400) < 1e-4
    print(f'\n[anchor] fixed_grad(400) = {g400:+.4e}  vs v3b dLoss/d(bias) dY ~ +2.4e-5')
    print(f'[anchor] sign {"OK" if ok_sign else "FAIL"} (expect +), '
          f'order {"OK" if ok_order else "FAIL"} (expect ~1e-5) -> '
          f'{"HARNESS OK" if (ok_sign and ok_order) else "HARNESS CHECK -- inspect loss reduction (mean-MSE vs nf-RMS)"}')

    # ── save ──────────────────────────────────────────────────────────────────────
    np.savez(os.path.join(datDir, 'b_bias_gap.npz'),
             H_sweep=np.array(H_SWEEP), grads=np.array([sweep[H] for H in H_SWEEP]),
             T_primary=T_PRIMARY, T_check=T_CHECK, grads_check=gs_check,
             g_inf=g_inf, a_coef=a_coef, v3b_anchor=V3B_ANCHOR, seed=SEED, B=B, n_batches=N_BATCHES)

    # ── figure: 2 panels, falsifiable (no conclusion in title) ────────────────────
    Hs = np.array(H_SWEEP, float)
    means = np.array([sweep[H].mean() for H in H_SWEEP])
    ses   = np.array([sweep[H].std(ddof=1) / np.sqrt(N_BATCHES) for H in H_SWEEP])

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))

    # Panel A: grad(H) vs H, signed, log-x. Prediction: descends toward grad(T) as H grows.
    ax[0].errorbar(Hs, means, yerr=2 * ses, fmt='o-', lw=1.4, ms=4, capsize=3, color='tab:blue',
                   label='grad(H) = fixed_grad(H)  (+/- 2 SE)')
    ax[0].errorbar([T_CHECK], [gs_check.mean()], yerr=[2 * se_check], fmt='s', ms=7, capsize=3,
                   color='tab:orange', label=f'grad({T_CHECK}) plateau check')
    ax[0].axhspan(g_T - 2 * se_T, g_T + 2 * se_T, color='tab:green', alpha=0.18)
    ax[0].axhline(g_T, color='tab:green', lw=1.4, label=f'true_grad(T={T_PRIMARY})  +/- 2 SE')
    ax[0].axhline(g_inf, color='0.4', ls=':', lw=1.2, label=f'1/H asymptote g_inf={g_inf:+.1e} (secondary)')
    href = 400.0
    ax[0].plot(Hs, g_inf + (means[np.argmin(np.abs(Hs - href))] - g_inf) * (href / Hs),
               '--', color='tab:red', lw=1.0, label='1/H reference (anchored at nf=400)')
    ax[0].plot([400], [V3B_ANCHOR], 'P', ms=11, color='k', label=f'v3b anchor +{V3B_ANCHOR:.1e}')
    ax[0].axhline(0, color='k', lw=0.6)
    ax[0].set_xscale('log'); ax[0].set_xlabel('rollout horizon H  [steps, 4 kHz]')
    ax[0].set_ylabel('DC-direction gradient  d(mean-MSE)/dc')
    ax[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax[0].grid(True, which='both', alpha=0.3); ax[0].legend(fontsize=7.5, loc='best')
    ax[0].set_title('A  DC-direction gradient vs horizon (nf=400 is the training window)', fontsize=9)

    # Panel B: bias gap |grad(H) - grad(T_CHECK)| vs H, log-log, with a slope=-1 (1/H) guide.
    ref = gs_check.mean()
    gap = np.abs(means - ref)
    m = Hs < T_CHECK
    ax[1].loglog(Hs[m], gap[m], 'o-', lw=1.4, ms=4, color='tab:purple',
                 label='|grad(H) - grad(T_check)|')
    guide = gap[m][0] * (Hs[m][0] / Hs[m])                      # slope -1 anchored at first point
    ax[1].loglog(Hs[m], guide, '--', color='0.4', lw=1.1, label='1/H guide (slope -1)')
    ax[1].set_xlabel('rollout horizon H  [steps]'); ax[1].set_ylabel('|bias gap|')
    ax[1].grid(True, which='both', alpha=0.3); ax[1].legend(fontsize=8)
    ax[1].set_title('B  Bias gap vs horizon: does it fall like 1/H?', fontsize=9)

    fig.suptitle('Phase B  Truncation bias of the DC-direction gradient (biased window vs long-horizon truth)',
                 fontsize=11)
    fig.text(0.005, 0.005, f'b_bias_gap.npz | seed={SEED} | {N_BATCHES}x{B} windows | untrained baseline ANN | '
             f'fixed_grad(400)={g400:+.2e} | 2026-07-22', fontsize=6, color='0.45', va='bottom')
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    p = os.path.join(figDir, 'b_bias_gap.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f'\nsaved {p}\ndone | data -> {os.path.join(datDir, "b_bias_gap.npz")}')


if __name__ == '__main__':
    main()
