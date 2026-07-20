"""
d8_dc_visibility_horizon.py -- D-109: at what evaluation horizon nf does the
drift-driving slow ANN force become VISIBLE to the windowed RMS loss?

The de-confound run showed: in-window nf-RMS improves while the 12 s free-run
sim-RMS worsens. The open question deciding the next design step (moderate-nf
training vs Layer-2 projection) is whether a LONGER window would even carry a
gradient against the slow force error. This probe answers that with FORWARD
simulation only (no BPTT, so the nf=4000 training memory wall does not apply):

  1. Windowed pass at nf=400 (encoder re-init per window, exactly the training
     loss geometry, d7 S1a pattern), capturing the ANN output per step (d6
     shadow pattern) -> per-routed-row time-mean mean_w on the TRAINING
     distribution (D-109: near-truth windows, not the drifted free-run).
  2. For each horizon nf in HORIZONS: same non-overlapping windowed evaluation,
     model as-is vs model with the fixed mean_w subtracted from the ANN output.
  3. Visibility curve: Delta-RMS(nf) = RMS(full) - RMS(debiased), with the
     across-window standard error as the significance yardstick.

Expected signatures (falsifiable either way):
  * loss-blind at 0.1 s : Delta-RMS(400) << window RMS (drift ramp RMS-invisible);
  * visibility onset    : the nf where Delta-RMS rises above ~2x standard error
    is where a windowed loss would START to fight the drift-driving component;
  * if Delta-RMS is significant already at nf=400, the loss is NOT blind and the
    horizon-mismatch story must be revisited.

Run:
  conda run -n GraduationProject python scripts/gantry/diagnostics-drift/d8_dc_visibility_horizon.py
Env: CKPT      (default simulations/gantry_subnet/diagnostics/checkpoints/gantry_drift_last.pth)
     HORIZONS  (default "400,1000,2000,4000", comma-separated nf in samples)
     N_WINDOWS (cap windows per horizon; default all non-overlapping)
Outputs -> simulations/gantry_subnet/diagnostics/ (npz + png)
"""
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
GANTRY = os.path.join(REPO, 'scripts', 'gantry')
sys.path.insert(0, GANTRY)
sys.path.insert(0, os.path.dirname(__file__))

import deepSI
import drift_common as dc
from gantry_dynamic.data import load_datasets, compute_normalization
from gantry_dynamic.model import build_model, get_encoder_dims
from model_augmentation.fit_systems.blocks import Static_ANN_Block
from gantry_interconnect_dynamic import CFG as cfg

CKPT = os.environ.get(
    'CKPT', os.path.join(dc.OUT_DIR, 'checkpoints', 'gantry_drift_last.pth'))
HORIZONS  = [int(h) for h in os.environ.get('HORIZONS', '400,1000,2000,4000').split(',')]
N_WINDOWS = os.environ.get('N_WINDOWS', None)

# -- Build pipeline + load the trained drifted checkpoint (d7 pattern) --------
np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
data = load_datasets(cfg)
norm = compute_normalization(cfg, data)
np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
fit_sys = build_model(cfg.hp, cfg, data, norm)
if CKPT and os.path.exists(CKPT):
    fit_sys.__dict__ = torch.load(CKPT, map_location='cpu', weights_only=False)
    print(f'Loaded CKPT {os.path.basename(CKPT)}; bestfit={getattr(fit_sys, "bestfit", float("nan")):.3e}')
else:
    print(f'WARNING: CKPT not found ({CKPT}); untrained ANN -- machinery check only.')
    CKPT = None
fit_sys.hfn.eval()

na, nb, _, _ = get_encoder_dims(cfg.hp, cfg)
warm = max(na, nb)
ts   = cfg.ts_new
route_ix = np.asarray(cfg.ann_route_ix)
names8   = np.array(['X', 'Theta', 'Y', 'dX', 'dTheta', 'dY', 'delta_a', 'vdelta_a'])
k0_rows  = {0, 2, 3, 5}          # K=0 rows (X/Y position + velocity)
std_x    = norm.std_x.flatten()
absorber_rms = data.val_x_aug[:, 0].std()

v1 = data.val_data
u  = np.asarray(v1.u, dtype=np.float64)
y  = np.asarray(v1.y, dtype=np.float64)
Ntot = len(u)
print(f'V1: {Ntot} samples @ {1/ts:.0f} Hz ({Ntot*ts:.1f}s)  warm={warm}  '
      f'horizons={HORIZONS}  absorber RMS={absorber_rms:.3e} m')

# -- Shadow the ANN forward: record outputs + optional fixed-vector subtraction
ann = next(m for m in fit_sys.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
_orig_forward = ann.forward
_records  = []
_subtract = {'vec': None}


def _shadow_forward(z):
    w = _orig_forward(z)
    _records.append(w.detach().view(-1, ann.nw).cpu().numpy())
    if _subtract['vec'] is not None:
        w = w - _subtract['vec'].view(1, ann.nw, 1)
    return w


ann.forward = _shadow_forward


def _window_starts(h):
    starts = list(range(warm, Ntot - h, h))          # non-overlapping
    if N_WINDOWS is not None:
        step = max(1, len(starts) // int(N_WINDOWS))
        starts = starts[::step][:int(N_WINDOWS)]
    return starts


def _windowed_pass(h, capture=False):
    """Encoder re-init per window, free-run h samples (d7 S1a geometry).

    Returns per-window RMS arrays: per-channel (nwin,3) and pooled (nwin,).
    If capture, also returns the concatenated ANN outputs of the pass.
    """
    starts = _window_starts(h)
    rms_ch, rms_pool = [], []
    if capture:
        _records.clear()
    for s in starts:
        sl = slice(s - warm, s + h)
        sd = deepSI.System_data(u=u[sl], y=y[sl], dt=ts)
        yhat = np.asarray(fit_sys.apply_experiment(sd).y, dtype=np.float64)[-h:]
        err  = yhat - y[s:s + h]
        rms_ch.append(np.sqrt((err ** 2).mean(axis=0)))
        rms_pool.append(np.sqrt((err ** 2).mean()))
    out = (np.asarray(rms_ch), np.asarray(rms_pool), len(starts))
    if capture:
        ann_out = np.concatenate(_records, axis=0) if _records else np.zeros((0, ann.nw))
        return out + (ann_out,)
    return out


# -- Step 1: nf=400 full-ANN pass, capture -> mean_w on the training distribution
h0 = HORIZONS[0]
print(f'\nStep 1: full-ANN windowed pass at nf={h0} (capture ANN output) ...')
t0 = time.time()
rms_ch_full0, rms_pool_full0, nwin0, ann_out = _windowed_pass(h0, capture=True)
print(f'  {nwin0} windows, {ann_out.shape[0]} ANN calls  ({time.time()-t0:.0f}s)')

mean_w = ann_out.mean(axis=0)
rms_w  = np.sqrt((ann_out ** 2).mean(axis=0))
print('\n=== ANN output per routed row, measured over near-truth windows (D-109) ===')
print(f"  {'row':10s} {'mean [norm]':>13s} {'rms [norm]':>12s} {'|mean|/rms':>11s} {'mean [phys]':>13s}")
for j, r in enumerate(route_ix):
    phys = f'{mean_w[j] * std_x[r]:.3e}' if r < 6 else '   (aug)'
    flag = '  <-- K=0' if r in k0_rows else ''
    print(f'  {names8[r]:10s} {mean_w[j]:>13.3e} {rms_w[j]:>12.3e} '
          f'{abs(mean_w[j]) / (rms_w[j] + 1e-30):>11.3f} {phys:>13s}{flag}')

# -- Step 2: full vs debiased windowed RMS per horizon -------------------------
results = {}   # h -> dict(full/debias per-channel + pooled stats)
for h in HORIZONS:
    if h == h0:
        rms_ch_f, rms_pool_f, nwin = rms_ch_full0, rms_pool_full0, nwin0
    else:
        print(f'\nfull-ANN windowed pass at nf={h} ...')
        t0 = time.time()
        rms_ch_f, rms_pool_f, nwin = _windowed_pass(h)
        print(f'  {nwin} windows ({time.time()-t0:.0f}s)')

    print(f'debiased windowed pass at nf={h} (ANN minus fixed mean_w) ...')
    t0 = time.time()
    _subtract['vec'] = torch.tensor(mean_w, dtype=next(ann.parameters()).dtype)
    rms_ch_d, rms_pool_d, _ = _windowed_pass(h)
    _subtract['vec'] = None
    print(f'  done ({time.time()-t0:.0f}s)')

    results[h] = dict(
        nwin=nwin,
        full_ch=rms_ch_f.mean(axis=0),   deb_ch=rms_ch_d.mean(axis=0),
        full_pool=rms_pool_f.mean(),     deb_pool=rms_pool_d.mean(),
        # standard error of the across-window mean RMS
        se_pool=rms_pool_f.std(ddof=1) / np.sqrt(nwin) if nwin > 1 else np.nan,
        full_pool_windows=rms_pool_f,    deb_pool_windows=rms_pool_d,
        full_ch_windows=rms_ch_f,        deb_ch_windows=rms_ch_d,
    )

# -- Report --------------------------------------------------------------------
print('\n=== Visibility curve: windowed RMS, full ANN vs ANN-minus-mean ===')
print(f"  {'nf':>6s} {'sec':>6s} {'nwin':>5s} {'RMS full':>11s} {'RMS debias':>11s} "
      f"{'Delta':>11s} {'SE':>10s} {'Delta/SE':>9s}")
for h in HORIZONS:
    r = results[h]
    delta = r['full_pool'] - r['deb_pool']
    ratio = delta / r['se_pool'] if np.isfinite(r['se_pool']) and r['se_pool'] > 0 else np.nan
    print(f"  {h:>6d} {h*ts:>6.2f} {r['nwin']:>5d} {r['full_pool']:>11.3e} "
          f"{r['deb_pool']:>11.3e} {delta:>11.3e} {r['se_pool']:>10.2e} {ratio:>9.2f}")
print('\n  per-channel Delta (full - debias) [m]:')
print(f"  {'nf':>6s} {'X1':>11s} {'X2':>11s} {'Y':>11s}")
for h in HORIZONS:
    d = results[h]['full_ch'] - results[h]['deb_ch']
    print(f'  {h:>6d} {d[0]:>11.3e} {d[1]:>11.3e} {d[2]:>11.3e}')
print(f'\n  reference: absorber RMS = {absorber_rms:.3e} m; '
      f'training val nf-RMS was ~3e-5 m at nf=400.')
# HEURISTIC: 2x standard error as the visibility significance yardstick (plotted,
# not hard-coded into a verdict; the reader judges from the error bars).
print('  reading: Delta/SE >~ 2 at horizon h  => a windowed loss at h SEES the '
      'slow-force component;\n           Delta/SE << 2 up to 4000 => brute-force '
      'nf cannot see it on this data (Layer-2 route).')

# -- Plot ------------------------------------------------------------------------
hs = np.array(HORIZONS, dtype=float) * ts
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fp = np.array([results[h]['full_pool'] for h in HORIZONS])
dp = np.array([results[h]['deb_pool'] for h in HORIZONS])
se = np.array([results[h]['se_pool'] for h in HORIZONS])
axes[0].errorbar(hs, fp, yerr=se, fmt='o-', label='full ANN', capsize=3)
axes[0].errorbar(hs, dp, yerr=se, fmt='s-', label='ANN minus mean', capsize=3)
axes[0].axhline(absorber_rms, color='0.5', ls=':', label='absorber RMS')
axes[0].set_xscale('log'); axes[0].set_yscale('log')
axes[0].set_xlabel('window length [s]'); axes[0].set_ylabel('windowed RMS [m]')
axes[0].set_title('Does the windowed loss distinguish the model\nfrom its mean-debiased twin?')
axes[0].grid(True, which='both'); axes[0].legend(fontsize=8)
axes[1].errorbar(hs, fp - dp, yerr=2 * se, fmt='o-', color='C3', capsize=3,
                 label='Delta RMS (full - debias), 2SE bars')
axes[1].axhline(0, color='k', lw=0.6)
axes[1].set_xscale('log')
axes[1].set_xlabel('window length [s]'); axes[1].set_ylabel('Delta windowed RMS [m]')
axes[1].set_title('Visibility of the slow-force component vs horizon')
axes[1].grid(True, which='both'); axes[1].legend(fontsize=8)
tag = os.path.basename(CKPT).split('.')[0] if CKPT else 'untrained'
fig.suptitle(f'd8 DC-visibility probe (D-109) on {tag}')
fig.tight_layout()
stem = os.path.join(dc.OUT_DIR, f'd8_dc_visibility_{tag}')
fig.savefig(stem + '.png', dpi=150)
np.savez(stem + '.npz',
         horizons=np.array(HORIZONS), ts=ts, mean_w=mean_w, rms_w=rms_w,
         route_ix=route_ix, absorber_rms=absorber_rms, ckpt=str(CKPT),
         **{f'{k}_{h}': v for h in HORIZONS for k, v in results[h].items()
            if isinstance(v, (int, float, np.ndarray))})
print(f'\nSaved: {stem}.png')
print(f'Saved: {stem}.npz')
