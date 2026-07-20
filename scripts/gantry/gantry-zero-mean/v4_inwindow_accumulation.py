"""v4_inwindow_accumulation.py -- in-window per-step error accumulation (Theme D / V4).

Question (supervisor 2026-07-17, "kijk naar de eerste 100 stappen ... de error accumuleert
over de nf range ... explodeert het na de eerste 100 stappen"): for the FIXED drifted model,
HOW does the free-run prediction error grow as a function of rollout STEP INDEX within one
window, where does it take off, does it ramp or explode, and how does the learned DC change it?

This is a DIFFERENT axis from v3: v3 walked training/optimizer steps (the DC's birth in
weight-space); this walks rollout TIME-steps within one window for a fixed model (the anatomy of
one loss evaluation). G-A is closed (physics zero-mean, v1f); v3 showed the DC is systematic; this
shows why the windowed loss rewards it and how it accumulates.

Method (reuses the verified drift-demo machinery; no physics re-derived):
  - build the pipeline encoder + load the drifted checkpoint gantry_drift_71167_last (D-114; NOT
    _best, which is the epoch-0 zero-ANN trap -- guarded below);
  - the demo3 `freerun` shadow (copied from drift-visual/generate_data.py:396) gives three passes
    per window by monkey-patching ann.forward: FULL, DC-MUTED (subtract the model's measured mean
    output), ANN-OFF (zero the output = baseline);
  - roll each NON-OVERLAPPING window (length HORIZON_L) from its own encoder-init via
    `fit_sys.apply_experiment` (it strips the K0 encoder warm-up), average per-step error over
    windows and records so it is the TYPICAL in-window profile, not one lucky window.

What it computes (the approved core; the rigorous FTLE / init-vs-model pieces are the follow-on):
  1. per-step |error|(k) per logical channel (X, Y = K=0; Theta = sprung), LINEAR and LOG y;
  2. growth-law fit per channel: e ~ a*k (linear position error <=> constant VELOCITY offset),
     ~ a*k^2 (<=> constant force/accel), or log-linear (<=> instability). # THEORY: kinematics of
     an integrator (constant velocity -> ramp, constant accel -> parabola);
  3. K=0-vs-Theta envelope ratio (late/early RMS) -- the no-fading-memory test;
  4. DC-on vs DC-muted vs ANN-off overlaid (the reward-mechanism panel);
  5. reference subtraction: full-minus-off = the ANN-induced accumulation, full-minus-muted = the
     DC's own contribution (does removing just the DC flatten the ramp).

Convention: lives in scripts/gantry/gantry-zero-mean/; figures -> ./figures/v4_*.png, data ->
./data/v4_*.npz. Rolls a FIXED checkpoint (no training). Run by the user.
"""
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE   = os.path.dirname(os.path.abspath(__file__))     # .../scripts/gantry/gantry-zero-mean
GANTRY = os.path.dirname(HERE)                          # .../scripts/gantry
REPO   = os.path.dirname(os.path.dirname(GANTRY))       # repo root
sys.path.insert(0, os.path.join(GANTRY, 'drift-demo'))  # demo_common (sets up the rest of the path)

import deepSI
import demo_common as dm
from demo_common import CFG
from model_augmentation.fit_systems.blocks import Static_ANN_Block

# ─────────────────────────────────────────────────────────────────────────────
# Config knobs (one surface)
# ─────────────────────────────────────────────────────────────────────────────
RECORDS   = ['T1_standstill_Ym30.mat', 'T3_standstill_Y000.mat', 'T5_standstill_Yp30.mat']
HORIZON_L = 800    # HEURISTIC: window length in samples (2x the nf=400 training window; sees
                   # within-window growth and just past the boundary). ~0.2 s at 4 kHz.
MAX_SEG_PER_REC = 20   # HEURISTIC: non-overlapping windows averaged per record (enough to average)
TAIL_FRAC = 0.25   # HEURISTIC: fraction used for the late/early envelope ratio
DEC       = 4      # HEURISTIC: plot decimation only

CKPT = os.path.join(REPO, 'simulations', 'gantry_subnet', 'augmentation_linear_map',
                    '71167', 'gantry_drift_71167_last.pth')
CH        = ['X', 'Theta', 'Y']       # logical output channels (positions)
K0_CH     = [0, 2]                     # X, Y = K=0 (no spring); Theta (index 1) is sprung
STATE8    = ['X', 'Theta', 'Y', 'dX', 'dTheta', 'dY', 'delta_a', 'vdelta_a']

figDir = os.path.join(HERE, 'figures')
datDir = os.path.join(HERE, 'data')
os.makedirs(figDir, exist_ok=True)
os.makedirs(datDir, exist_ok=True)


def freerun(fit_sys, ann, u, y, n, ts, subtract=None, zero=False):
    """demo3 shadow free-run (copied from drift-visual/generate_data.py:396): free-run with the
    ANN output captured / DC-subtracted / zeroed. Returns (yhat_stage (M,3), w_records (?,8))."""
    records = []
    orig = ann.forward

    def shadow(z):
        w = orig(z)
        records.append(w.detach().view(-1, ann.nw).cpu().numpy())
        if zero:
            return w * 0.0
        if subtract is not None:
            return w - subtract.view(1, ann.nw, 1)
        return w

    ann.forward = shadow
    try:
        sd = deepSI.System_data(u=u[:n], y=y[:n], dt=ts)
        r = fit_sys.apply_experiment(sd)
    finally:
        ann.forward = orig
    return np.asarray(r.y, dtype=np.float64), np.concatenate(records, axis=0)


def _r2(y, yhat):
    ss = np.sum((y - np.mean(y)) ** 2)
    return float(1 - np.sum((y - yhat) ** 2) / ss) if ss > 0 else np.nan


def fit_growth(k, e):
    """Fit e(k) ~ a*k, a*k^2, and log-linear; return the best form + params. # THEORY: on a K=0
    integrator a constant velocity offset gives position error ~ k, a constant force ~ k^2, a
    dynamical instability ~ exp(k)."""
    k = k.astype(float)
    m = k >= 1
    kk, ee = k[m], e[m]
    a1 = np.sum(kk * ee) / np.sum(kk * kk);      r1 = _r2(ee, a1 * kk)
    a2 = np.sum(kk ** 2 * ee) / np.sum(kk ** 4); r2 = _r2(ee, a2 * kk ** 2)
    pos = ee > 1e-30
    if pos.sum() > 5:
        b, c = np.polyfit(kk[pos], np.log(ee[pos]), 1)
        rexp = _r2(np.log(ee[pos]), b * kk[pos] + c)
        dbl = float(np.log(2) / b) if b > 0 else np.inf
    else:
        b = c = rexp = np.nan; dbl = np.inf
    forms = {'linear': r1, 'quad': r2, 'exp': rexp}
    best = max(forms, key=lambda f: forms[f] if np.isfinite(forms[f]) else -np.inf)
    return dict(a_lin=float(a1), r2_lin=r1, a_quad=float(a2), r2_quad=r2,
                b_exp=float(b), r2_exp=rexp, doubling=dbl, best=best)


def envelope_ratio(e):
    q = max(1, int(len(e) * TAIL_FRAC))
    early = np.sqrt(np.mean(e[:q] ** 2))
    late = np.sqrt(np.mean(e[-q:] ** 2))
    return float(late / early) if early > 0 else np.inf


def main():
    print('v4_inwindow_accumulation | per-rollout-step error growth of the drifted model')
    if not os.path.exists(CKPT):
        sys.exit(f'checkpoint not found: {CKPT}')

    fit_sys, norm, K0, na, nb, na_right, nb_right = dm.build_pipeline()
    ts = CFG.ts_new
    fit_sys.__dict__ = torch.load(CKPT, map_location='cpu', weights_only=False)
    fit_sys.hfn.eval()
    ann = next(m for m in fit_sys.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
    assert ann.nw == 8, f'expected full X+Theta+Y routing (nw=8), got nw={ann.nw}'

    # global DC = the model's mean per-row output over a full free-run on the first record.
    u0, y0, _, _ = dm.load_T(RECORDS[0])
    _, w0 = freerun(fit_sys, ann, u0, y0, len(u0), ts)
    if float(np.abs(w0).max()) == 0.0:
        sys.exit(f'ANN output identically zero: {CKPT} is the _best/epoch-0 model, not the '
                 'drifted _last (D-114). Refusing to run.')
    mean_w = w0.mean(axis=0)
    rms_w = np.sqrt((w0 ** 2).mean(axis=0))
    mean_w_t = torch.tensor(mean_w, dtype=next(ann.parameters()).dtype)
    print('  ANN per-row |mean|/rms: ' + '  '.join(
        f'{nm}={abs(m)/(r+1e-30):.2f}' for nm, m, r in zip(STATE8, mean_w, rms_w)))
    print(f'  DC on K=0 rows: dX={mean_w[3]:+.3e} dY={mean_w[5]:+.3e} (normalized ANN output units)')

    # ── segment loop: 3 passes per non-overlapping window, error in logical coords ──
    Ef, Em, Eo, Dfm, Dfo = [], [], [], [], []
    nseg = 0
    for rec in RECORDS:
        u, y, _, _ = dm.load_T(rec)
        starts = list(range(0, len(u) - HORIZON_L, HORIZON_L))[:MAX_SEG_PER_REC]
        print(f'  {rec}: {len(starts)} windows of {HORIZON_L} samples')
        for s in starts:
            us, ys = u[s:s + HORIZON_L], y[s:s + HORIZON_L]
            yf, _ = freerun(fit_sys, ann, us, ys, HORIZON_L, ts)
            ym, _ = freerun(fit_sys, ann, us, ys, HORIZON_L, ts, subtract=mean_w_t)
            yo, _ = freerun(fit_sys, ann, us, ys, HORIZON_L, ts, zero=True)
            M = min(len(yf), len(ym), len(yo))
            truth = ys[HORIZON_L - M:HORIZON_L]
            Ef.append(np.abs(dm.stage_pos_to_logical(yf[-M:] - truth)))
            Em.append(np.abs(dm.stage_pos_to_logical(ym[-M:] - truth)))
            Eo.append(np.abs(dm.stage_pos_to_logical(yo[-M:] - truth)))
            Dfm.append(np.abs(dm.stage_pos_to_logical(yf[-M:] - ym[-M:])))   # DC's own contribution
            Dfo.append(np.abs(dm.stage_pos_to_logical(yf[-M:] - yo[-M:])))   # all-ANN contribution
            nseg += 1

    Mmin = min(a.shape[0] for a in Ef)
    st = lambda L: np.stack([a[:Mmin] for a in L])            # (nseg, Mmin, 3)
    Ef, Em, Eo, Dfm, Dfo = st(Ef), st(Em), st(Eo), st(Dfm), st(Dfo)
    ef, em, eo = Ef.mean(0), Em.mean(0), Eo.mean(0)
    dfm, dfo = Dfm.mean(0), Dfo.mean(0)
    k = np.arange(Mmin)
    print(f'  averaged over {nseg} windows | per-window rollout length M={Mmin} steps '
          f'({Mmin*ts*1e3:.0f} ms; nf=400 boundary at step {400-K0})')

    # ── per-channel growth-law + envelope ──
    fits, envs = {}, {}
    print('\n  channel | full best-fit (R2) | full slope a_lin | envelope late/early: full/muted/off')
    for c, name in enumerate(CH):
        f_full = fit_growth(k, ef[:, c])
        fits[name] = f_full
        envs[name] = dict(full=envelope_ratio(ef[:, c]), muted=envelope_ratio(em[:, c]),
                          off=envelope_ratio(eo[:, c]))
        tag = 'K=0' if c in K0_CH else 'sprung'
        print(f'  {name:5s} [{tag:6s}] | {f_full["best"]:6s} '
              f'(R2 lin={f_full["r2_lin"]:.3f} quad={f_full["r2_quad"]:.3f} exp={f_full["r2_exp"]:.3f}) '
              f'| a_lin={f_full["a_lin"]:.3e} | {envs[name]["full"]:.2f} / '
              f'{envs[name]["muted"]:.2f} / {envs[name]["off"]:.2f}')

    # ── figures ──
    ip = np.arange(0, Mmin, DEC)
    nf_boundary = 400 - K0

    # Fig 1: per-step error, 3 passes, linear (top) + log (bottom) per channel
    fh, ax = plt.subplots(2, 3, figsize=(15, 8))
    for c, name in enumerate(CH):
        for row, ylabel in [(0, 'linear'), (1, 'log')]:
            a = ax[row, c]
            a.plot(k[ip], eo[ip, c], color=[0.2, 0.6, 0.2], lw=0.9, label='ANN off (baseline)')
            a.plot(k[ip], em[ip, c], color=[0.2, 0.3, 0.8], lw=0.9, label='DC muted')
            a.plot(k[ip], ef[ip, c], color=[0.8, 0.1, 0.1], lw=1.1, label='full ANN')
            a.axvline(nf_boundary, color='0.5', ls=':', lw=1.0)
            a.grid(True, alpha=0.4)
            if row == 1:
                a.set_yscale('log'); a.set_xlabel('rollout step k')
            tag = 'K=0' if c in K0_CH else 'sprung'
            a.set_title(f'{name} [{tag}] |error| ({ylabel} y)')
            if c == 0:
                a.set_ylabel(f'|error| [{"m" if name != "Theta" else "rad"}] ({ylabel})')
            if c == 2 and row == 0:
                a.legend(fontsize=7, loc='upper left')
    fh.suptitle('v4: in-window per-step |error| vs rollout step (dotted = nf=400 window boundary). '
                'Straight on linear = ramp; straight on log = exponential.')
    fh.tight_layout()
    fh.savefig(os.path.join(figDir, 'v4_perstep_error.png'), dpi=150)
    plt.close(fh)

    # Fig 2: growth-law fit of the full pass per channel + envelope ratios
    fh, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for c, name in enumerate(CH):
        f = fits[name]
        a = ax[c]
        a.plot(k[ip], ef[ip, c], 'k.', ms=2, label='full |error|')
        a.plot(k, f['a_lin'] * k, color=[0.8, 0.1, 0.1], lw=1.0, label=f'lin fit R2={f["r2_lin"]:.3f}')
        a.plot(k, f['a_quad'] * k ** 2, color=[0.2, 0.3, 0.8], lw=1.0, label=f'quad fit R2={f["r2_quad"]:.3f}')
        a.axvline(nf_boundary, color='0.5', ls=':', lw=1.0)
        a.grid(True, alpha=0.4); a.set_xlabel('rollout step k')
        tag = 'K=0' if c in K0_CH else 'sprung'
        a.set_title(f'{name} [{tag}]: best={f["best"]} | env full/off={envs[name]["full"]:.1f}/{envs[name]["off"]:.1f}')
        if c == 0:
            a.set_ylabel('|error|')
        a.legend(fontsize=7)
    fh.suptitle('v4: growth-law fit (full pass). linear-wins = constant velocity offset; quad = force; '
                'envelope ratio > 1 = no fading memory')
    fh.tight_layout()
    fh.savefig(os.path.join(figDir, 'v4_growth_law.png'), dpi=150)
    plt.close(fh)

    # Fig 3: reference subtraction -- the DC's own contribution and the all-ANN contribution
    fh, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for c, name in enumerate(CH):
        a = ax[c]
        a.plot(k[ip], dfo[ip, c], color=[0.6, 0.3, 0.1], lw=1.0, label='full - off (all ANN)')
        a.plot(k[ip], dfm[ip, c], color=[0.2, 0.3, 0.8], lw=1.0, label='full - muted (the DC)')
        a.axvline(nf_boundary, color='0.5', ls=':', lw=1.0)
        a.grid(True, alpha=0.4); a.set_xlabel('rollout step k')
        tag = 'K=0' if c in K0_CH else 'sprung'
        a.set_title(f'{name} [{tag}]: ANN-induced trajectory deviation')
        if c == 0:
            a.set_ylabel('|deviation|')
        if c == 2:
            a.legend(fontsize=7)
    fh.suptitle('v4: reference subtraction. full-minus-off = the ANN-induced accumulation; '
                'full-minus-muted isolates the DC (a ramp = the drift-causing constant)')
    fh.tight_layout()
    fh.savefig(os.path.join(figDir, 'v4_reference_subtraction.png'), dpi=150)
    plt.close(fh)

    np.savez(os.path.join(datDir, 'v4_results.npz'),
             k=k, ts=ts, nf_boundary=nf_boundary, ef=ef, em=em, eo=eo, dfm=dfm, dfo=dfo,
             mean_w=mean_w, rms_w=rms_w, channels=np.array(CH), records=np.array(RECORDS),
             horizon_L=HORIZON_L, nseg=nseg,
             fits=np.array([str(fits)], dtype=object), envs=np.array([str(envs)], dtype=object))
    print(f'\nfigures -> {figDir}\\v4_*.png | data -> {datDir}\\v4_results.npz')


if __name__ == '__main__':
    main()
