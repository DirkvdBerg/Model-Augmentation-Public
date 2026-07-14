"""
demo6_objective_split.py -- F3 (plan doc §12): the objective-vs-deployment split + the
"Roland" cross-nf figure. NO simulation: reads the training histories embedded in the
copied _last checkpoints (each fit_sys.__dict__ carries Loss_val (sim-RMS per epoch,
cumulative across warm rungs) and Loss_train_nf/Loss_val_nf (per-rung/trial, the probe
resets them at every install, training.py _NfProbe.__init__)).

CLAIM (caption-claim): minimizing the training window worsens the free-run at every window
length, warm and cold, so no amount of window training fixes the drift.

E/O/E/D: EXPECT window and free-run to improve together (faithful proxy; supervisors suggested
longer windows). OBSERVE the split (window flat/down, sim-RMS up 24x at nf=400) and no trained
model below ANN-off at any nf. EXPLAIN: lr/epochs/unlearnable/objective. DISCRIMINATE: the split
itself rules out lr/epochs as driver; d8's sign rules out longer-nf; F8 rules out "nothing
learnable" => the objective cannot price the free-run cost of the DC.

Figures -> simulations/gantry_subnet/diagnostics/drift-demo/:
  f3a_split.png      warm(70903 rung0, nf400) + cold(71013 trial3, nf800): train nf-RMS (top)
                     vs val sim-RMS (bottom, log, ANN-off line). Stacked panels, NO twin axes.
  f3b_cross_nf.png   start/best/end sim-RMS per nf; warm circles CONNECTED, cold squares
                     independent; ANN-off line. Caveat: 8-epoch budgets = lower bounds.
  f3_split.npz

Run: conda run -n GraduationProject python scripts/gantry/drift-demo/demo6_objective_split.py
"""
__project_origin__ = "added"

import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import demo_common as dm            # noqa: F401  (imports pipeline classes so torch.load unpickles)
from demo_common import REPO

ALM = os.path.join(REPO, 'simulations', 'gantry_subnet', 'augmentation_linear_map')
ANN_OFF = 8.014968e-05              # epoch-0 (zero-init ANN) sim-RMS on the cropped val (both runs' logs)
EVAL_SET = 'cropped V1 validation (8000 samples), free-run sim-RMS [m]'

COLD = {nf: os.path.join(ALM, 'trial_ckpts_71013', f'trial{i}_nf{nf}_lr1e-07_last.pth')
        for i, nf in [(0, 2400), (1, 3200), (2, 1600), (3, 800)]}
WARM = {nf: os.path.join(ALM, 'curriculum_70903', f'rung{r}_nf{nf}_last.pth')
        for r, nf in [(0, 400), (1, 800), (2, 1600), (3, 2000)]}


def hist(path):
    """Histories from a saved fit_sys.__dict__ (no model rebuild needed)."""
    d = torch.load(path, map_location='cpu', weights_only=False)
    return {
        'Loss_val':      np.asarray(d.get('Loss_val', []), dtype=float),      # sim-RMS per epoch (cumulative over warm rungs)
        'Loss_train_nf': np.asarray(d.get('Loss_train_nf', []), dtype=float), # THIS segment only (probe resets per install)
        'Loss_val_nf':   np.asarray(d.get('Loss_val_nf', []), dtype=float),
        'bestfit':       float(d.get('bestfit', np.nan)),
    }


print('Loading checkpoint histories ...')
cold = {nf: hist(p) for nf, p in COLD.items() if os.path.exists(p)}
warm = {nf: hist(p) for nf, p in WARM.items() if os.path.exists(p)}
for nf, h in sorted(cold.items()):
    print(f'  cold nf={nf:4d}: Loss_val n={len(h["Loss_val"])}  train_nf n={len(h["Loss_train_nf"])}')
for nf, h in sorted(warm.items()):
    print(f'  warm nf={nf:4d}: Loss_val n={len(h["Loss_val"])} (cumulative)  train_nf n={len(h["Loss_train_nf"])}')

# Warm per-rung sim-RMS segments: consecutive cumulative lengths.
warm_seg = {}
prev = 0
for r, nf in [(0, 400), (1, 800), (2, 1600), (3, 2000)]:
    if nf not in warm:
        continue
    lv = warm[nf]['Loss_val']
    warm_seg[nf] = lv[prev:]
    prev = len(lv)

# ── Figure A: the split (stacked, shared x, per run) ─────────────────────────
cases = []
if 400 in warm:
    cases.append(('WARM 70903 rung 0 (nf=400, lr=1e-7, 8 ep)', warm[400]['Loss_train_nf'], warm_seg[400]))
if 800 in cold:
    cases.append(('COLD 71013 trial 3 (nf=800, lr=1e-7, 8 ep)', cold[800]['Loss_train_nf'], cold[800]['Loss_val']))

figA, axs = plt.subplots(2, len(cases), figsize=(6.5 * len(cases), 7), sharex='col')
axs = np.atleast_2d(axs.T).T if len(cases) > 1 else axs.reshape(2, 1)
for c, (title, tr, sv) in enumerate(cases):
    ep_t = np.arange(1, len(tr) + 1)
    ep_s = np.arange(1, len(sv) + 1)
    axT, axB = axs[0, c], axs[1, c]
    axT.plot(ep_t, tr, 'C0-o', ms=3, label='train nf-RMS (the OBJECTIVE, windowed) [m]')
    axT.set_ylabel('train nf-RMS [m]'); dm.sci_axes(axT); axT.grid(True)
    axT.set_title(title, fontsize=10)
    axT.legend(fontsize=7)
    d_obj = (tr[-1] - tr[0]) / tr[0] * 100 if len(tr) else np.nan
    axB.semilogy(ep_s, sv, 'C3-o', ms=3, label='val sim-RMS (the DEPLOYMENT metric, free-run)')
    axB.axhline(ANN_OFF, color='k', ls='--', lw=1.0, label=f'ANN off (epoch 0) = {ANN_OFF:.1e}')
    axB.set_ylabel(EVAL_SET); axB.set_xlabel('epoch'); axB.grid(True, which='both')
    axB.legend(fontsize=7, loc='lower right')
    d_dep = sv[-1] / ANN_OFF if len(sv) else np.nan
    axB.set_title(f'objective {d_obj:+.0f}%  vs  deployment x{d_dep:.0f} vs ANN-off', fontsize=9)
figA.suptitle('Does minimizing the training window improve the free-run?  (the SPLIT)')
dm.add_provenance(figA, 'histories embedded in _last checkpoints | 70903 warm curriculum, '
                        '71013 cold sweep | full X+Theta+Y routing, lr=1e-7')
figA.tight_layout()
pA = os.path.join(dm.OUT_DIR, 'f3a_split.png')
figA.savefig(pA, dpi=150); print(f'Saved: {pA}')

# ── Figure B: cross-nf ("increase the window", Roland) ───────────────────────
figB, ax = plt.subplots(figsize=(9, 6))
# warm: connected in ladder order (one continuous model)
w_nfs = sorted(warm_seg.keys())
for stat, marker, alpha in (('start', 'o', 0.35), ('best', 'o', 0.7), ('end', 'o', 1.0)):
    vals = [dict(start=warm_seg[n][0], best=np.nanmin(warm_seg[n]), end=warm_seg[n][-1])[stat]
            for n in w_nfs]
    ax.plot(w_nfs, vals, marker=marker, color='C1', alpha=alpha, lw=1.2 if stat == 'end' else 0.0,
            ls='-' if stat == 'end' else 'none', ms=8,
            label=f'WARM 70903 {stat} (one model, ladder order)')
# cold: independent squares
c_nfs = sorted(cold.keys())
for stat, alpha in (('start', 0.35), ('best', 0.7), ('end', 1.0)):
    vals = [dict(start=cold[n]['Loss_val'][0], best=np.nanmin(cold[n]['Loss_val']),
                 end=cold[n]['Loss_val'][-1])[stat] for n in c_nfs]
    ax.plot(c_nfs, vals, 's', color='C0', alpha=alpha, ms=8,
            label=f'COLD 71013 {stat} (fresh init per nf)')
ax.axhline(ANN_OFF, color='k', ls='--', lw=1.2, label=f'ANN off (epoch 0) = {ANN_OFF:.1e}')
ax.set_xscale('log', base=2); ax.set_yscale('log')
ax.set_xticks(sorted(set(w_nfs + c_nfs)))
ax.set_xticklabels([str(n) for n in sorted(set(w_nfs + c_nfs))])
ax.set_xlabel('training window length nf [samples]  (x ts = horizon [s])')
ax.set_ylabel(EVAL_SET)
ax.grid(True, which='both')
ax.legend(fontsize=7, loc='upper right')
ax.set_title('"Increase the window": does ANY nf produce a net free-run improvement?\n'
             '(8-epoch budgets: levels are LOWER BOUNDS; claim is "never below ANN-off", '
             'not "converged optima")', fontsize=10)
dm.add_provenance(figB, 'warm = recovery-from-drift dynamics; cold = drift-acquisition dynamics; '
                        'agreement from opposite directions | lr=1e-7 fixed, routing (0..7)')
figB.tight_layout()
pB = os.path.join(dm.OUT_DIR, 'f3b_cross_nf.png')
figB.savefig(pB, dpi=150); print(f'Saved: {pB}')

# ── Numeric table + npz ───────────────────────────────────────────────────────
print(f'\n=== cross-nf summary ({EVAL_SET}); ANN-off = {ANN_OFF:.3e} ===')
print(f"  {'run':6s} {'nf':>5s} {'start':>11s} {'best':>11s} {'end':>11s} {'end/ANN-off':>12s}")
for n in w_nfs:
    s = warm_seg[n]
    print(f'  warm  {n:5d} {s[0]:>11.3e} {np.nanmin(s):>11.3e} {s[-1]:>11.3e} {s[-1]/ANN_OFF:>12.1f}')
for n in c_nfs:
    s = cold[n]['Loss_val']
    print(f'  cold  {n:5d} {s[0]:>11.3e} {np.nanmin(s):>11.3e} {s[-1]:>11.3e} {s[-1]/ANN_OFF:>12.1f}')

np.savez(os.path.join(dm.OUT_DIR, 'f3_split.npz'),
         ann_off=ANN_OFF,
         **{f'warm{n}_simrms': warm_seg[n] for n in w_nfs},
         **{f'warm{n}_trainnf': warm[n]['Loss_train_nf'] for n in w_nfs},
         **{f'cold{n}_simrms': cold[n]['Loss_val'] for n in c_nfs},
         **{f'cold{n}_trainnf': cold[n]['Loss_train_nf'] for n in c_nfs})
print(f"Saved: {os.path.join(dm.OUT_DIR, 'f3_split.npz')}")
