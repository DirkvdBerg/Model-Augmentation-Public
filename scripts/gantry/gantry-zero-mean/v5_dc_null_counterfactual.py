"""v5_dc_null_counterfactual.py -- would a soft DC penalty (option B) fix the sim drift, at no fit cost?

Option B = a soft penalty that drives the ANN's K=0-row output mean (the DC) to zero. Its ENTIRE
premise is: the DC direction is loss-flat, so removing it costs ~no windowed loss while it kills the
free-run drift. This diagnostic tests that premise DIRECTLY on the trained (drifted) checkpoint, with
NO retraining: for a scale alpha in [0,1] we subtract alpha * DC from the ANN output on the K=0 rows
(via the exact freerun shadow v4 uses) and measure both quantities that matter:

  L(alpha) = windowed (nf-horizon) free-run loss  -- the FIT COST of removing the DC
  D(alpha) = long free-run position drift on X, Y -- the DRIFT the DC causes

Read:
  * L(1) ~ L(0) (flat) AND D(alpha)->baseline as alpha->1  => option B fixes the drift at ~no fit cost.
  * L(alpha) rises as alpha->1                              => the DC is load-bearing; B trades fit for
                                                              drift (report the tradeoff curve).

Why this is a RIGOROUS upper bound, not just suggestive: retraining WITH a soft penalty can only reach
a loss <= this static counterfactual's L(alpha=1) at DC~0 (re-optimization only helps). So L(1) is an
UPPER BOUND on option B's fit cost. A flat L => B is guaranteed near-free. The same script on real
data reveals whether the real DC is load-bearing (L rises = genuine friction/physics, do NOT suppress)
or free (L flat = nuisance, B removes it).

DC target = the K=0 correction rows {X(0), Y(2), dX(3), dY(5)} (matches option B's pin target); the
drift-causing part is the velocity rows dX/dY (a constant velocity offset integrates to a position
ramp). Theta (sprung) is NOT pinned. Reuses demo_common + the v4 freerun shadow; rolls a FIXED
checkpoint (no training). Convention: figures -> ./figures/v5_*.png, data -> ./data/v5_*.npz.
"""
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE   = os.path.dirname(os.path.abspath(__file__))
GANTRY = os.path.dirname(HERE)
REPO   = os.path.dirname(os.path.dirname(GANTRY))
sys.path.insert(0, os.path.join(GANTRY, 'drift-demo'))

import deepSI
import demo_common as dm
from demo_common import CFG
from model_augmentation.fit_systems.blocks import Static_ANN_Block

# ─── knobs ───────────────────────────────────────────────────────────────────
ALPHAS      = [float(a) for a in os.environ.get('V5_ALPHAS', '0,0.25,0.5,0.75,1.0').split(',')]
LOSS_RECS   = ['T1_standstill_Ym30.mat', 'T3_standstill_Y000.mat', 'T5_standstill_Yp30.mat',
               'T6_ysweep_slow.mat', 'T13_lissajous.mat']   # mix of standstill + moving for the fit
DRIFT_RECS  = ['T1_standstill_Ym30.mat', 'T3_standstill_Y000.mat', 'T5_standstill_Yp30.mat']  # drift shows at standstill
NF_LOSS     = int(os.environ.get('V5_NF', '400'))     # windowed-loss horizon = training nf
N_WIN_LOSS  = int(os.environ.get('V5_NWIN', '8'))     # non-overlapping nf-windows per record for L(alpha)
N_DRIFT     = int(os.environ.get('V5_NDRIFT', '8000'))# long free-run length for D(alpha) (~2 s at 4 kHz)

CKPT = os.path.join(REPO, 'simulations', 'gantry_subnet', 'augmentation_linear_map',
                    '71167', 'gantry_drift_71167_last.pth')
STATE8 = ['X', 'Theta', 'Y', 'dX', 'dTheta', 'dY', 'delta_a', 'vdelta_a']
K0_ROWS = [0, 2, 3, 5]        # X, Y positions + dX, dY velocities = option B pin target
CH = ['X', 'Theta', 'Y']      # logical output channels
K0_CH = [0, 2]                # X, Y logical channels (the drifting ones)

figDir = os.path.join(HERE, 'figures'); os.makedirs(figDir, exist_ok=True)
datDir = os.path.join(HERE, 'data');    os.makedirs(datDir, exist_ok=True)


def freerun(fit_sys, ann, u, y, n, ts, subtract=None):
    """v4 shadow free-run: apply_experiment with the ANN output optionally DC-subtracted."""
    orig = ann.forward

    def shadow(z):
        w = orig(z)
        return w if subtract is None else w - subtract.view(1, ann.nw, 1)

    ann.forward = shadow
    try:
        r = fit_sys.apply_experiment(deepSI.System_data(u=u[:n], y=y[:n], dt=ts))
    finally:
        ann.forward = orig
    return np.asarray(r.y, dtype=np.float64)


def main():
    print('v5_dc_null_counterfactual | does nulling the K=0 DC remove drift at no fit cost?')
    if not os.path.exists(CKPT):
        sys.exit(f'checkpoint not found: {CKPT}')

    fit_sys, norm, K0, na, nb, na_right, nb_right = dm.build_pipeline()
    ts = CFG.ts_new
    fit_sys.__dict__ = torch.load(CKPT, map_location='cpu', weights_only=False)
    fit_sys.hfn.eval()
    ann = next(m for m in fit_sys.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
    assert ann.nw == 8, f'expected full routing nw=8, got {ann.nw}'
    ystd = np.asarray(norm.ystd, dtype=np.float64).flatten()   # (3,) stage-output std for loss weighting

    # DC vector = model mean output over a full free-run, MASKED to the K=0 rows (option B target).
    u0, y0, _, _ = dm.load_T(LOSS_RECS[0])
    orig = ann.forward
    caught = []
    ann.forward = lambda z: (caught.append(orig(z).detach().view(-1, ann.nw).cpu().numpy()) or orig(z))
    _ = fit_sys.apply_experiment(deepSI.System_data(u=u0, y=y0, dt=ts))
    ann.forward = orig
    w0 = np.concatenate(caught, axis=0)
    if float(np.abs(w0).max()) == 0.0:
        sys.exit(f'ANN output identically zero: {CKPT} is the _best/epoch-0 trap, not drifted _last.')
    mean_w = w0.mean(axis=0)
    dc_vec = np.zeros(ann.nw); dc_vec[K0_ROWS] = mean_w[K0_ROWS]
    dc_t = torch.tensor(dc_vec, dtype=next(ann.parameters()).dtype)
    print('  full DC (masked to K=0 rows): ' +
          '  '.join(f'{STATE8[r]}={dc_vec[r]:+.3e}' for r in K0_ROWS))

    # ── L(alpha): windowed nf-loss (normalized stage-pos MSE), averaged over windows ──
    # ── D(alpha): long free-run drift = tail RMS of |position error| on X, Y (logical) ──
    WLEN = NF_LOSS + K0            # freerun strips K0 warm-up -> ~NF_LOSS rolled steps
    tail = lambda e: np.sqrt(np.mean(e[-max(1, len(e) // 4):] ** 2))   # last-25% RMS

    # precollect loss windows and drift segments once (same across alpha)
    loss_wins = []
    for rec in LOSS_RECS:
        u, y, _, _ = dm.load_T(rec)
        for s in list(range(0, len(u) - WLEN, WLEN))[:N_WIN_LOSS]:
            loss_wins.append((u[s:s + WLEN], y[s:s + WLEN]))
    print(f'  L(alpha): {len(loss_wins)} windows of ~{NF_LOSS} steps | '
          f'D(alpha): {len(DRIFT_RECS)} records x {N_DRIFT} steps')

    results = {}
    for a in ALPHAS:
        sub = None if a == 0.0 else (a * dc_t)
        # windowed loss
        mses = []
        for us, ys in loss_wins:
            yf = freerun(fit_sys, ann, us, ys, WLEN, ts, subtract=sub)
            M = len(yf); truth = ys[WLEN - M:WLEN]
            mses.append(np.mean(((yf - truth) / ystd[None, :]) ** 2))
        L = float(np.mean(mses))
        # long-horizon drift (logical X, Y tail RMS)
        drifts = []
        for rec in DRIFT_RECS:
            u, y, _, _ = dm.load_T(rec)
            yf = freerun(fit_sys, ann, u, y, N_DRIFT, ts, subtract=sub)
            M = len(yf); truth = y[N_DRIFT - M:N_DRIFT]
            elog = np.abs(dm.stage_pos_to_logical(yf - truth))   # (M,3) logical position error
            drifts.append([tail(elog[:, c]) for c in range(3)])
        D = np.mean(drifts, axis=0)   # (3,) per logical channel
        results[a] = dict(L=L, D=D)
        print(f'  alpha={a:.2f} | L={L:.4e} (L/L0={L/results[ALPHAS[0]]["L"]:.3f}) | '
              f'drift X={D[0]:.3e} Y={D[2]:.3e} Theta={D[1]:.3e}')

    # ── ANN-OFF baseline (zero ANN) for reference on both axes ──
    zero_t = dc_t * 0.0

    def freerun_off(u, y, n):
        o = ann.forward
        ann.forward = lambda z: o(z) * 0.0
        try:
            r = fit_sys.apply_experiment(deepSI.System_data(u=u[:n], y=y[:n], dt=ts))
        finally:
            ann.forward = o
        return np.asarray(r.y, dtype=np.float64)

    off_mses = []
    for us, ys in loss_wins:
        yf = freerun_off(us, ys, WLEN); M = len(yf)
        off_mses.append(np.mean(((yf - ys[WLEN - M:WLEN]) / ystd[None, :]) ** 2))
    L_off = float(np.mean(off_mses))
    off_dr = []
    for rec in DRIFT_RECS:
        u, y, _, _ = dm.load_T(rec)
        yf = freerun_off(u, y, N_DRIFT); M = len(yf)
        elog = np.abs(dm.stage_pos_to_logical(yf - y[N_DRIFT - M:N_DRIFT]))
        off_dr.append([tail(elog[:, c]) for c in range(3)])
    D_off = np.mean(off_dr, axis=0)
    print(f'  ANN-OFF (baseline) | L_off={L_off:.4e} | drift X={D_off[0]:.3e} Y={D_off[2]:.3e}')

    # ── verdict ──
    L0 = results[ALPHAS[0]]['L']; L1 = results[ALPHAS[-1]]['L']
    dX0, dX1 = results[ALPHAS[0]]['D'][0], results[ALPHAS[-1]]['D'][0]
    dY0, dY1 = results[ALPHAS[0]]['D'][2], results[ALPHAS[-1]]['D'][2]
    print('\n  === VERDICT ===')
    print(f'  fit cost of nulling DC:  L(1)/L(0) = {L1/L0:.3f}  (flat ~1.0 => option B is ~free; '
          f'>1 => DC load-bearing)')
    print(f'  drift removed X: {dX0:.3e} -> {dX1:.3e}  ({dX1/dX0:.2f}x; baseline {D_off[0]:.3e})')
    print(f'  drift removed Y: {dY0:.3e} -> {dY1:.3e}  ({dY1/dY0:.2f}x; baseline {D_off[2]:.3e})')
    verdict = ('B PREDICTED TO FIX AT ~NO FIT COST' if (L1 / L0 < 1.05 and dY1 < 0.5 * dY0)
               else 'DC PARTLY LOAD-BEARING or drift not DC-only -- see tradeoff curve')
    print(f'  => {verdict}')

    # ── figure: L/L0 and drift vs alpha ──
    al = np.array(ALPHAS)
    fh, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].plot(al, [results[a]['L'] / L0 for a in ALPHAS], 'o-', color=[0.8, 0.1, 0.1])
    ax[0].axhline(1.0, color='k', lw=0.6, ls='--')
    ax[0].axhline(L_off / L0, color=[0.2, 0.6, 0.2], lw=1.0, ls=':', label=f'ANN-off = {L_off/L0:.2f}')
    ax[0].set_xlabel('alpha (fraction of DC removed)'); ax[0].set_ylabel('L(alpha) / L(0)')
    ax[0].set_title('FIT COST of removing the DC (flat ~1 => B is free)'); ax[0].grid(True, alpha=0.4)
    ax[0].legend(fontsize=8)
    for c, nm, col in [(0, 'X', [0.1, 0.2, 0.8]), (2, 'Y', [0.8, 0.4, 0.1])]:
        ax[1].plot(al, [results[a]['D'][c] for a in ALPHAS], 'o-', color=col, label=f'{nm} drift')
        ax[1].axhline(D_off[c], color=col, lw=1.0, ls=':', alpha=0.7)
    ax[1].set_xlabel('alpha (fraction of DC removed)'); ax[1].set_ylabel('tail-RMS position error [m]')
    ax[1].set_title('DRIFT vs DC removed (dotted = ANN-off baseline)'); ax[1].grid(True, alpha=0.4)
    ax[1].legend(fontsize=8)
    fh.suptitle(f'v5 DC-null counterfactual: {verdict}')
    fh.tight_layout(); fh.savefig(os.path.join(figDir, 'v5_dc_null_counterfactual.png'), dpi=150)
    plt.close(fh)

    np.savez(os.path.join(datDir, 'v5_dc_null_counterfactual.npz'),
             alphas=al, L=np.array([results[a]['L'] for a in ALPHAS]),
             D=np.array([results[a]['D'] for a in ALPHAS]), L_off=L_off, D_off=D_off,
             dc_vec=dc_vec, verdict=verdict)
    print(f'\n  figure -> {figDir}\\v5_dc_null_counterfactual.png | data -> {datDir}\\v5_dc_null_counterfactual.npz')


if __name__ == '__main__':
    main()
