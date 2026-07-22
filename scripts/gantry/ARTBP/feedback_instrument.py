"""feedback_instrument.py -- Phase B0-feedback: no-training test of whether ARTBP could suppress the
SECOND drift component (the state-dependent Y destabilization, v5/v6/D-117), in the kappa(H) framework.

B0 perturbed the loss along a CONSTANT c on the Y-route and found the loss weakly pins it. Here the
perturbation is a FEEDBACK GAIN g on a Y-state (not a constant): the ANN's Y-route output gets an
extra term g*phi(x) recomputed each step, where
  phi = dY (index 5)  -> ANTI-DAMPING (force +g*dY opposes the tiny physical damping; g>0 flips net
                         damping negative -> EXPONENTIAL growth). Primary; the classic z=1 destabilizer.
  phi = Y  (index 2)  -> negative-stiffness / positive position feedback. Secondary.

Injection is FAITHFUL (through ann.forward, so it routes through the model like B0's route_const, NOT
a raw state-row hack): the patched forward reads the current physical state via a holder set each step.

The feedback landscape is ONE-SIDED (g<0 = extra damping, harmless; g>0 = anti-damping, blows up), so
we do NOT force a symmetric parabola. Decisive readouts vs horizon H:
  - DL_plus(H)  = [L_H(+g0) - L_H(0)] / L_H(0)   (relative loss penalty of a small DESTABILIZING gain)
  - DL_minus(H) = the stabilizing-side contrast
  - div_frac(H,g) = fraction of windows whose rollout diverged (non-finite) = the instability itself
  - a near-zero local kappa_g(H) for continuity with the B0 framework.
Prediction (mechanism): DL_plus(H) ~ 0 for H << the instability e-folding time (the window is too short
to see the blow-up, so training gets NO restoring signal), then turns on sharply once H reaches it.
The TURN-ON HORIZON decides the estimator (ARTBP) vs structural (D-117) route: if it is within a
feasible ARTBP cap, ARTBP can fight the instability; if it needs an infeasible horizon, D-117 is needed.

CALIBRATION FIRST (verify-knob-moves-the-target): a coarse log-g sweep at one long H prints where the
loss responds / diverges, so the fine g-grid and the sweep are not wasted on a dead or all-diverged range.

Convention: data -> ./data/b0fb_feedback.npz, figures -> ./figures/b0fb_*.png. No em-dashes.
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
from model_augmentation.fit_systems.blocks import Static_ANN_Block

# ── one config surface ─────────────────────────────────────────────────────────
SEED      = int(os.environ.get('FB_SEED', str(CFG.seed)))
B         = int(os.environ.get('FB_B', '32'))
N_BATCHES = int(os.environ.get('FB_NBATCH', '2'))          # 2 x 32 = 64 windows (long H is expensive)
STRIDE    = int(os.environ.get('FB_STRIDE', '40'))
PHI       = os.environ.get('FB_PHI', 'dY')                 # 'dY' anti-damping (primary) | 'Y' position
H_LAND    = [400, 1600, 3200, 6400, 12800]                 # into the ~seconds instability regime (3.2 s)
H_CAL     = int(os.environ.get('FB_HCAL', '6400'))         # calibration horizon
G_CAL     = np.array([-1e-2, -1e-3, -1e-4, -1e-5, 0.0, 1e-5, 1e-4, 1e-3, 1e-2])   # coarse, both signs
GMAX      = float(os.environ.get('FB_GMAX', '1e-3'))       # fine-grid half-range (adjust after calib)
NG        = 13
GGRID     = np.linspace(-GMAX, GMAX, NG)
NBOOT     = 200
PHI_IDX   = {'dY': 5, 'Y': 2}[PHI]
IY        = 5                                               # dY state = the ANN Y-route target
V12_HMAX  = 1600                                            # current ARTBP cap (0.4 s) for the feasibility line

figDir = os.path.join(HERE, 'figures'); os.makedirs(figDir, exist_ok=True)
datDir = os.path.join(HERE, 'data');    os.makedirs(datDir, exist_ok=True)
HMAX = max(max(H_LAND), H_CAL)


def main():
    print(f'feedback_instrument (Phase B0-feedback) | seed={SEED} | PHI={PHI} (idx {PHI_IDX}) | '
          f'{N_BATCHES}x{B}={N_BATCHES*B} windows | H_LAND={H_LAND} | GMAX={GMAX:.0e}')
    fit_sys, norm, K0, na, nb, na_right, nb_right = dm.build_pipeline(cfg=CFG, verbose=True)
    for p in list(fit_sys.hfn.parameters()) + list(fit_sys.encoder.parameters()):
        p.requires_grad_(False)

    ann = next(m for m in fit_sys.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
    nw = ann.nw
    route_col = list(int(i) for i in np.asarray(CFG.ann_route_ix).ravel()).index(IY)
    print(f'[ann] nw={nw}, dY route column={route_col}, feedback state phi={PHI} (idx {PHI_IDX})')

    # Faithful feedback injection through ann.forward: extra term g*phi(x) on the Y-route column.
    # cur_x holds the state ENTERING each hfn step (what the ANN/feedback reads); gbox holds g.
    e_col = torch.zeros(1, nw, 1); e_col[0, route_col, 0] = 1.0
    cur_x = {'x': None}; gbox = {'g': 0.0}
    orig_forward = ann.forward
    def forward_patched(z):
        out = orig_forward(z)
        if gbox['g'] != 0.0 and cur_x['x'] is not None:
            phi = cur_x['x'][:, PHI_IDX]                    # (B,) normalized state fed back
            out = out + (gbox['g'] * phi).view(-1, 1, 1) * e_col
        return out
    ann.forward = forward_patched

    # window bank (real data, encoder init)
    um = norm.u_mean.flatten()[None, :]; us = norm.std_u.flatten()[None, :]
    y0 = np.asarray(norm.y0).flatten()[None, :]; ys = np.asarray(norm.ystd).flatten()[None, :]
    recs, starts = [], []
    for ri, f in enumerate(TRAIN_FILES):
        u, y, _, _ = dm.load_T(f, CFG)
        un = ((u - um) / us).astype(np.float32); yn = ((y - y0) / ys).astype(np.float32)
        recs.append((un, yn)); N = len(un)
        for p in range(max(K0, na, nb), N - HMAX, STRIDE):
            starts.append((ri, p))
    starts = np.array(starts)
    rng = np.random.default_rng(SEED); rng.shuffle(starts)
    starts = starts[:N_BATCHES * B]
    print(f'[bank] {len(starts)} windows, HMAX={HMAX}')
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
            x0 = fit_sys.encoder(UH.contiguous(), YH.contiguous()).detach()
        batches.append(dict(U=U, Y=Y, x0=x0))

    def perwin_loss(x0, U, Y, H, g):
        """Per-window mean-MSE vs real target for feedback gain g. Non-finite (diverged) -> nan."""
        gbox['g'] = float(g)
        with torch.no_grad():
            x = x0.clone(); se = torch.zeros(x.shape[0])
            for t in range(H):
                cur_x['x'] = x
                yhat, x = fit_sys.hfn(x, U[:, t, :])
                se = se + torch.mean((Y[:, t, :] - yhat) ** 2, dim=1)
            gbox['g'] = 0.0; cur_x['x'] = None
            return (se / H).cpu().numpy()                    # (B,)

    def loss_grid(H, g_arr):
        """(Nwin, len(g_arr)) per-window losses; nan where diverged."""
        L = np.full((N_BATCHES * B, len(g_arr)), np.nan, np.float32)
        for bi, bat in enumerate(batches):
            sl = slice(bi * B, (bi + 1) * B)
            for gi, g in enumerate(g_arr):
                L[sl, gi] = perwin_loss(bat['x0'], bat['U'], bat['Y'], H, g)
        L[~np.isfinite(L)] = np.nan
        return L

    # ── CALIBRATION (printed first) ────────────────────────────────────────────────
    print(f'\n[calib] H={H_CAL}: does the feedback gain move / diverge the loss? '
          f'(L(g)/L(0), div-frac)')
    Lc = loss_grid(H_CAL, G_CAL)
    L0 = np.nanmean(Lc[:, np.argmin(np.abs(G_CAL))])
    for gi, g in enumerate(G_CAL):
        col = Lc[:, gi]; div = np.mean(~np.isfinite(col))
        ratio = (np.nanmean(col) / L0) if np.isfinite(np.nanmean(col)) else np.inf
        print(f'   g={g:+.0e}: L/L0={ratio:9.3e} | div-frac={div*100:5.1f}%')
    print(f'[calib] if the destabilizing side (g>0) neither rises nor diverges at H={H_CAL}, widen GMAX;'
          f' if it diverges at every g>0, shrink GMAX. Current fine GMAX={GMAX:.0e}.')

    # ── the kappa_g(H) / DL_plus(H) sweep on the fine grid ─────────────────────────
    j0 = int(np.argmin(np.abs(GGRID)))
    res = {}
    for H in H_LAND:
        t0 = time.time()
        L = loss_grid(H, GGRID)
        Lm = np.nanmean(L, axis=0)                           # (NG,)
        div = np.mean(~np.isfinite(L), axis=0)               # per-g divergence fraction
        L0h = Lm[j0]
        # near-zero local parabola (finite region around 0) -> kappa_g, slope
        near = np.abs(GGRID) <= 0.5 * GMAX
        kappa_g = slope0 = np.nan
        if np.sum(np.isfinite(Lm[near])) >= 3:
            a, b, _ = np.polyfit(GGRID[near], np.nan_to_num(Lm[near], nan=np.nanmax(Lm[near])), 2)
            kappa_g = 2 * a; slope0 = b
        # one-sided responses at the first grid point either side of 0
        gp = GGRID[j0 + 1]; gm = GGRID[j0 - 1]
        DLp = (Lm[j0 + 1] - L0h) / L0h if np.isfinite(Lm[j0 + 1]) else np.inf
        DLm = (Lm[j0 - 1] - L0h) / L0h if np.isfinite(Lm[j0 - 1]) else np.inf
        res[H] = dict(Lm=Lm, div=div, kappa_g=kappa_g, slope0=slope0, DLp=DLp, DLm=DLm,
                      gp=gp, gm=gm, div_pos=np.mean(div[GGRID > 0]))
        print(f'[H={H:6d}] DL+({gp:+.1e})={DLp:+.3e}  DL-({gm:+.1e})={DLm:+.3e} | '
              f'kappa_g={kappa_g:.2e} | div-frac(g>0 mean)={res[H]["div_pos"]*100:.0f}% | {time.time()-t0:.0f}s')

    ann.forward = orig_forward

    # ── verdict: at what H does the destabilizing side turn on, vs the feasible cap ─
    turn_on = None
    for H in H_LAND:
        if (np.isfinite(res[H]['DLp']) and res[H]['DLp'] > 0.5) or res[H]['div_pos'] > 0.1:
            turn_on = H; break
    print('\n==== VERDICT (Phase B0-feedback) ====')
    if turn_on is None:
        print(f'DL+ never turned on up to H={H_LAND[-1]} at GMAX={GMAX:.0e} -> widen the grid or the '
              f'gain is below scale; INCONCLUSIVE (recalibrate).')
    else:
        feasible = turn_on <= V12_HMAX
        print(f'Destabilizing side turns on at H~{turn_on} ({turn_on/4000:.2f} s). Current ARTBP cap '
              f'H_max={V12_HMAX} ({V12_HMAX/4000:.2f} s) -> {"WITHIN reach (estimator/ARTBP route viable)" if feasible else "BEYOND reach: ARTBP would need a much larger cap -> favors the D-117 STRUCTURAL (passivity) route"}.')

    np.savez(os.path.join(datDir, 'b0fb_feedback.npz'),
             ggrid=GGRID, g_cal=G_CAL, H_land=np.array(H_LAND), phi=PHI,
             Lm=np.array([res[H]['Lm'] for H in H_LAND]),
             div=np.array([res[H]['div'] for H in H_LAND]),
             DLp=np.array([res[H]['DLp'] for H in H_LAND]),
             DLm=np.array([res[H]['DLm'] for H in H_LAND]),
             kappa_g=np.array([res[H]['kappa_g'] for H in H_LAND]),
             Lcal=Lc, turn_on=(turn_on if turn_on else -1), v12_hmax=V12_HMAX)

    # ── figures ─────────────────────────────────────────────────────────────────────
    gx = GGRID * 1e3
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for H in H_LAND:
        Lm = res[H]['Lm']
        ax[0].plot(gx, Lm / np.nanmin(Lm), 'o-', lw=1.2, ms=3, label=f'H={H}')
    ax[0].axvline(0, color='k', lw=0.6); ax[0].set_yscale('log')
    ax[0].set_xlabel(f'feedback gain g on {PHI} route  [1e-3]'); ax[0].set_ylabel('loss / its min')
    ax[0].grid(True, which='both', alpha=0.3); ax[0].legend(fontsize=8)
    ax[0].set_title('A  Feedback landscape L(g): a one-sided wall forming on g>0?', fontsize=9)

    Hs = np.array(H_LAND, float)
    DLp = np.array([res[H]['DLp'] for H in H_LAND]); DLm = np.array([res[H]['DLm'] for H in H_LAND])
    ax[1].loglog(Hs, np.clip(DLp, 1e-6, None), 'o-', label='DL+ (destabilizing g>0)')
    ax[1].loglog(Hs, np.clip(np.abs(DLm), 1e-6, None), 's--', color='0.5', label='|DL-| (stabilizing g<0)')
    ax[1].axvline(V12_HMAX, color='tab:red', ls=':', label=f'ARTBP cap {V12_HMAX} ({V12_HMAX/4000:.1f}s)')
    ax[1].set_xlabel('horizon H  [steps, 4 kHz]'); ax[1].set_ylabel('relative loss response')
    ax[1].grid(True, which='both', alpha=0.3); ax[1].legend(fontsize=8)
    ax[1].set_title('B  Does the destabilizing response turn on, and at what horizon?', fontsize=9)

    dpos = np.array([res[H]['div_pos'] for H in H_LAND])
    ax[2].semilogx(Hs, dpos * 100, 'o-', color='tab:purple')
    ax[2].axvline(V12_HMAX, color='tab:red', ls=':')
    ax[2].set_xlabel('horizon H  [steps]'); ax[2].set_ylabel('diverged windows (g>0)  [%]')
    ax[2].grid(True, which='both', alpha=0.3)
    ax[2].set_title('C  Runaway fraction on the destabilizing side vs horizon', fontsize=9)

    fig.suptitle(f'Phase B0-feedback  ARTBP-vs-Y-destabilization: kappa(H) along a {PHI} feedback gain',
                 fontsize=11)
    fig.text(0.005, 0.005, f'b0fb_feedback.npz | seed={SEED} | phi={PHI} | {N_BATCHES*B} windows | 2026-07-22',
             fontsize=6, color='0.45', va='bottom')
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(os.path.join(figDir, 'b0fb_feedback.png'), dpi=150); plt.close(fig)
    print(f'\nsaved {os.path.join(figDir, "b0fb_feedback.png")}\ndone | data -> '
          f'{os.path.join(datDir, "b0fb_feedback.npz")}')


if __name__ == '__main__':
    main()
