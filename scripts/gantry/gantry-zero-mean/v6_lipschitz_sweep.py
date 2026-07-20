"""v6_lipschitz_sweep.py -- does a by-construction Lipschitz cap on the aug ANN kill the v5 divergence?

D-118 / D-117 test of the stability-preserving route. For each Lipschitz cap L (incl. 'off' = control,
the current free ANN), train the full X+Theta+Y augmentation the same way (lr, nf, epochs matched) and
measure whether the trained ANN still DESTABILIZES the long free-run (v5: on the K=0/z=1 axis the ANN
makes drift ~50x the physics baseline). The cap L is the static-ANN analog of the Gyorok contraction
rate (a magnitude/Jacobian bound). We read three things per L:
  - long-horizon free-run DRIFT on X and Y (tail-RMS position error over ~2 s, standstill records),
    for the FULL ANN and with the ANN OFF (baseline) -- does the cap pull full-ANN drift toward baseline?
  - windowed nf-RMS FIT -- does the cap cost accuracy? (the contraction-vs-fit trade-off)
Read: an L that brings full-ANN drift down to ~baseline WITHOUT wrecking the fit => stability-by-
construction works. Drift only dies when the fit degrades / integrator leaks => contraction too blunt
for the z=1 mode -> passivity route (D-117).

Knobs (env): V6_LS='off,1.0' (cap list), V6_EPOCHS=4, V6_LR=1e-7, V6_NDRIFT=8000, V6_SEED=0.
Convention: gantry-zero-mean/; data -> ./data/v6_*.npz, figures -> ./figures/v6_*.png. DOES train.
"""
import os
import sys
from dataclasses import replace

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE   = os.path.dirname(os.path.abspath(__file__))
GANTRY = os.path.dirname(HERE)
ROOT   = os.path.dirname(os.path.dirname(GANTRY))
for p in (ROOT, GANTRY, os.path.join(GANTRY, 'drift-demo')):
    if p not in sys.path:
        sys.path.insert(0, p)

import deepSI
from gantry_dynamic.config import RunConfig
from gantry_dynamic.data import load_datasets, compute_normalization
from gantry_dynamic.model import build_model, train_model
from model_augmentation.fit_systems.blocks import Static_ANN_Block
import demo_common as dm   # load_T, stage_pos_to_logical

# ─── knobs ───────────────────────────────────────────────────────────────────
LS_LIST  = os.environ.get('V6_LS', 'off,1.0').split(',')          # 'off' = control (free ANN)
EPOCHS   = int(os.environ.get('V6_EPOCHS', '4'))
LR       = float(os.environ.get('V6_LR', '1e-7'))
N_DRIFT  = int(os.environ.get('V6_NDRIFT', '8000'))
SEED     = int(os.environ.get('V6_SEED', '0'))
DRIFT_RECS = ['T1_standstill_Ym30.mat', 'T3_standstill_Y000.mat', 'T5_standstill_Yp30.mat']
NF_FIT   = 400

STATE8 = ['X', 'Theta', 'Y', 'dX', 'dTheta', 'dY', 'delta_a', 'vdelta_a']
figDir = os.path.join(HERE, 'figures'); os.makedirs(figDir, exist_ok=True)
datDir = os.path.join(HERE, 'data');    os.makedirs(datDir, exist_ok=True)


def base_config(seed):
    """Mirror run 71167 (the v5 drift checkpoint): full X+Theta+Y routing, nominal theta, stride=10."""
    return RunConfig(
        mode='augmentation', encoder_init='linear_map', ann_activation='tanh',
        joint_estimation=False, param_rmse_baseline=0.01,
        orth_beta=0.0, orth_observe=False,
        param_init_detune=None, snr=None, seed=seed,
        fs_orig=20000, fs_new=4000, stride=10, use_f64=False,
        save_flag=False, nf_probe_print=True,
        nx_ann=2, ann_route_ix=(0, 1, 2, 3, 4, 5, 6, 7),
        n_nodes_per_layer=16, n_hidden_layers=2, up_sample=1,
        batch_size=256, lr=LR, epochs=EPOCHS, nf_seconds=0.100,
    )


def freerun(fit_sys, ann, u, y, n, ts, zero=False):
    orig = ann.forward
    ann.forward = (lambda z: orig(z) * 0.0) if zero else orig
    try:
        r = fit_sys.apply_experiment(deepSI.System_data(u=u[:n], y=y[:n], dt=ts))
    finally:
        ann.forward = orig
    return np.asarray(r.y, dtype=np.float64)


def eval_drift_and_fit(fit_sys, ann, ts, ystd):
    """Long-horizon drift (tail-RMS |pos err| on logical X,Y) full vs ANN-off, + windowed nf-RMS fit."""
    tail = lambda e: float(np.sqrt(np.mean(e[-max(1, len(e) // 4):] ** 2)))
    dr_full, dr_off, fit_win = [], [], []
    for rec in DRIFT_RECS:
        u, y, _, _ = dm.load_T(rec)
        yf = freerun(fit_sys, ann, u, y, N_DRIFT, ts)
        yo = freerun(fit_sys, ann, u, y, N_DRIFT, ts, zero=True)
        M = min(len(yf), len(yo)); truth = y[N_DRIFT - M:N_DRIFT]
        ef = np.abs(dm.stage_pos_to_logical(yf[-M:] - truth))
        eo = np.abs(dm.stage_pos_to_logical(yo[-M:] - truth))
        dr_full.append([tail(ef[:, c]) for c in range(3)])
        dr_off.append([tail(eo[:, c]) for c in range(3)])
        # windowed nf-RMS fit (full ANN) over up to 6 non-overlapping nf windows
        for s in range(0, min(len(u) - NF_FIT, 6 * NF_FIT), NF_FIT):
            yw = freerun(fit_sys, ann, u[s:s + NF_FIT], y[s:s + NF_FIT], NF_FIT, ts)
            mw = len(yw)
            fit_win.append(np.mean(((yw - y[s + NF_FIT - mw:s + NF_FIT]) / ystd[None, :]) ** 2))
    return (np.mean(dr_full, axis=0), np.mean(dr_off, axis=0), float(np.sqrt(np.mean(fit_win))))


def run_one(L, data, norm):
    if L == 'off':
        os.environ.pop('ANN_LIPSCHITZ', None)
    else:
        os.environ['ANN_LIPSCHITZ'] = str(L)
    cfg = base_config(SEED); hp = cfg.hp
    np.random.seed(SEED); torch.manual_seed(SEED)
    fit_sys = build_model(hp, cfg, data, norm)
    ann = next(m for m in fit_sys.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
    ts = cfg.ts_new; ystd = np.asarray(norm.ystd, dtype=np.float64).flatten()
    print(f'\n===== L={L} | epochs={EPOCHS} lr={LR:.0e} seed={SEED} =====', flush=True)
    # No-op deepSI checkpoint save (v3 pattern), for TWO reasons:
    #  (1) the spectral-norm parametrization cannot be torch.save'd (only via state_dict) -> would crash;
    #  (2) sim-RMS selects _best = epoch-0 (zero ANN), but the DRIFT lives in _LAST (D-114) -> we must
    #      evaluate the trained/drifted model, not the reverted _best. No save -> deepSI keeps _last.
    orig_ckpt = fit_sys.checkpoint_save_system
    fit_sys.checkpoint_save_system = lambda *a, **k: None
    try:
        train_model(fit_sys, hp, cfg, data, epochs=EPOCHS, nf=hp['nf'], validation_measure='sim-RMS')
    finally:
        fit_sys.checkpoint_save_system = orig_ckpt
    fit_sys.hfn.eval()
    dr_full, dr_off, fit_win = eval_drift_and_fit(fit_sys, ann, ts, ystd)
    print(f'  L={L}: drift FULL  X={dr_full[0]:.3e} Y={dr_full[2]:.3e}', flush=True)
    print(f'  L={L}: drift OFF   X={dr_off[0]:.3e} Y={dr_off[2]:.3e}  (baseline)', flush=True)
    print(f'  L={L}: full/off ratio X={dr_full[0]/dr_off[0]:.2f} Y={dr_full[2]/dr_off[2]:.2f} | fit nf-RMS={fit_win:.3e}', flush=True)
    return dict(L=str(L), drift_full=dr_full, drift_off=dr_off, fit=fit_win)


def main():
    print(f'v6_lipschitz_sweep | L={LS_LIST} epochs={EPOCHS} lr={LR:.0e} seed={SEED}', flush=True)
    cfg0 = base_config(SEED)
    np.random.seed(cfg0.seed); torch.manual_seed(cfg0.seed)
    data = load_datasets(cfg0)
    norm = compute_normalization(cfg0, data)
    res = [run_one(L.strip(), data, norm) for L in LS_LIST]

    print('\n=== SUMMARY (drift = tail-RMS |pos err| over 2 s; ratio = full/ANN-off) ===', flush=True)
    print(f'{"L":>6} | {"Ydrift full":>11} {"Ydrift off":>11} {"Y ratio":>8} | {"Xdrift full":>11} {"X ratio":>8} | {"fit nfRMS":>10}', flush=True)
    for r in res:
        df, do = r['drift_full'], r['drift_off']
        print(f'{r["L"]:>6} | {df[2]:11.3e} {do[2]:11.3e} {df[2]/do[2]:8.2f} | {df[0]:11.3e} {df[0]/do[0]:8.2f} | {r["fit"]:10.3e}', flush=True)

    # figure: Y-drift ratio (full/off) and fit vs L
    labels = [r['L'] for r in res]
    yratio = [r['drift_full'][2] / r['drift_off'][2] for r in res]
    fits = [r['fit'] for r in res]
    fh, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].bar(labels, yratio, color=[0.8, 0.3, 0.1]); ax[0].axhline(1.0, color='k', ls='--', lw=0.8)
    ax[0].set_ylabel('Y drift  full / ANN-off'); ax[0].set_title('Divergence vs Lipschitz cap (1.0 = baseline)')
    ax[0].set_xlabel('L (off = free ANN control)'); ax[0].grid(True, axis='y', alpha=0.4)
    ax[1].bar(labels, fits, color=[0.1, 0.3, 0.8]); ax[1].set_ylabel('windowed nf-RMS (fit)')
    ax[1].set_title('Fit cost vs Lipschitz cap'); ax[1].set_xlabel('L'); ax[1].grid(True, axis='y', alpha=0.4)
    fh.suptitle('v6: by-construction Lipschitz cap on the aug ANN -- drift vs fit')
    fh.tight_layout(); fh.savefig(os.path.join(figDir, 'v6_lipschitz_sweep.png'), dpi=150)
    plt.close(fh)
    np.savez(os.path.join(datDir, 'v6_lipschitz_sweep.npz'),
             labels=np.array(labels), drift_full=np.array([r['drift_full'] for r in res]),
             drift_off=np.array([r['drift_off'] for r in res]), fit=np.array(fits),
             epochs=EPOCHS, lr=LR, seed=SEED)
    print(f'\ndone | figure -> {figDir}\\v6_lipschitz_sweep.png | data -> {datDir}\\v6_lipschitz_sweep.npz', flush=True)


if __name__ == '__main__':
    main()
