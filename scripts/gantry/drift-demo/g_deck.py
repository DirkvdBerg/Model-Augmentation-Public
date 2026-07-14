"""
g_deck.py -- builds the redesigned supervisor figure deck G0-G7 (plan doc §14.3).

All figures are rebuilt from the SAVED results of demo1/demo3/demo6/f4/f5 (figures/*.npz)
plus light data loads -- no new simulation. Conventions (§14.1): one hypothesis per figure
(title = question); red solid = cause present, blue dashed = cause removed, grey = "no ANN",
black dashed = independent prediction, dotted band = absorber scale; <= 3 curves/panel;
plain language (no DC/K=0/nf jargon on figures); physical units; endpoint annotations.

Run: conda run -n GraduationProject python scripts/gantry/drift-demo/g_deck.py   (~1 min)
Outputs -> scripts/gantry/drift-demo/figures/g*.png
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
import demo_common as dm
from demo_common import CFG, REPO
from model_augmentation.systems.gantry_ss import mh as _mh, cy as _cy

FIG = dm.OUT_DIR
MH, CY = float(_mh), float(_cy)          # Y-row mass [kg], Y viscous damping [N s/m]
ts = CFG.ts_new

# ── Load saved results ────────────────────────────────────────────────────────
d1 = np.load(os.path.join(FIG, 'f1_encoder_ic.npz'))
d2 = np.load(os.path.join(FIG, 'f2_dc.npz'), allow_pickle=True)
d3 = np.load(os.path.join(FIG, 'f3_split.npz'))
d5 = np.load(os.path.join(FIG, 'f5_horizon.npz'))
absorber = float(d1['absorber_rms'])

# Pipeline (for std_x unit conversion) + T1 data (for the G1 absolute view)
fit_sys, norm, K0, na, nb, na_right, nb_right = dm.build_pipeline(verbose=False)
std_dY = float(norm.std_x.flatten()[5])
u_w, y_w, xl_w, da_w = dm.load_T('T1_standstill_Ym30.mat', need_absorber=True)

# THEORY: the ANN adds dv = w*std_dY [m/s] to the Y-velocity each step ts -> equivalent
# continuous force F = m_h * dv/ts; terminal velocity of that force v_ss = F / c_y.
w2force_mN = lambda w: MH * (w * std_dY / ts) * 1e3
v_ss_pred = MH * (float(d2['mean_w'][5]) * std_dY / ts) / CY

C = dict(red='#c62828', blue='#1565c0', grey='0.45', pred='k')


def foot(fig, text):
    fig.text(0.005, 0.003, text, fontsize=6, color='0.45', va='bottom')


def endlab(ax, x, y, text, color, dy=0):
    ax.annotate(text, xy=(x, y), xytext=(4, dy), textcoords='offset points',
                fontsize=8, color=color, va='center', fontweight='bold')


# ══ G0 -- opening summary ═════════════════════════════════════════════════════
e_full, e_off = d2['e_full'][:, 2], d2['e_off'][:, 2]
t2 = d2['t']
fig, (aL, aR) = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
aL.plot(t2, e_off * 100, color=C['grey'], lw=1.2)
aR.plot(t2, e_full * 100, color=C['red'], lw=1.2)
for a in (aL, aR):
    a.axhspan(-absorber * 100, absorber * 100, color='0.75', alpha=0.5)
    a.set_xlabel('time [s]'); a.grid(True)
aL.set_ylabel('Y position error [cm]')
aL.set_title('without the ANN')
aR.set_title('with the TRAINED ANN')
aL.annotate('bounded: 0.02 cm (encoder-start offset)', xy=(0.5, 0.82),
            xycoords='axes fraction', ha='center', fontsize=10, color=C['grey'],
            fontweight='bold')
aL.annotate('grey band at 0 = the absorber signal we want to learn (0.002 cm)',
            xy=(0.5, 0.72), xycoords='axes fraction', ha='center', fontsize=7, color='0.35')
aR.annotate('unbounded: -2.9 cm\n(constant learned force x no stiffness)',
            xy=(0.55, 0.35), xycoords='axes fraction', ha='center', fontsize=10,
            color=C['red'], fontweight='bold')
fig.suptitle('What goes wrong when we augment the gantry model?  (12 s free-run, same scale)')
foot(fig, 'V1 standstill | trained checkpoint gantry_drift_last vs ANN output zeroed | free-run vs truth')
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'g0_summary.png'), dpi=150); plt.close(fig)

# ══ G1 -- context: trajectory + model-vs-measurement ═════════════════════════
sl = slice(K0, K0 + len(d1['t']))
y_true_sim = y_w[sl] + d1['stage_R']            # model(true start) = truth + R
t1 = d1['t']
fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.5))
a1.plot(t1, y_w[sl][:, 2], color='0.2', lw=1.0)
a1.set_ylabel('measured Y [m]'); a1.grid(True)
a1.set_title('(a) the trajectory: head parked at Y = -0.30 m; the excitation only makes a '
             'tiny ripple')
axi = a1.inset_axes([0.55, 0.15, 0.42, 0.6])
seg = slice(0, int(0.05 / ts))
axi.plot(t1[seg], (y_w[sl][seg, 2] + 0.3) * 1e6, color='0.2', lw=0.7)
axi.set_title('zoom: ripple around -0.30 m [um]', fontsize=7)
axi.tick_params(labelsize=6); axi.grid(True)
a2.plot(t1, y_w[sl][:, 2], color='0.2', lw=1.6, label='measured')
a2.plot(t1, y_true_sim[:, 2], color=C['blue'], lw=0.8, ls='--', label='model (true start)')
dev = np.abs(d1['stage_R'][:, 2]).max()
a2.set_ylabel('Y [m]'); a2.set_xlabel('time [s]'); a2.grid(True)
a2.legend(fontsize=8, loc='upper right')
a2.set_title(f'(b) model vs measurement, overlaid at trajectory scale: indistinguishable '
             f'(max deviation {dev*1e3:.2f} mm = {dev/0.3*100:.2f}% of the 0.3 m position)')
fig.suptitle('What does the system do, and does the model track it?')
foot(fig, 'T1 standstill (Y=-0.30 m) | model = baseline physics simulated from the TRUE initial state')
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'g1_context.png'), dpi=150); plt.close(fig)

# ══ G2 -- the encoder start ══════════════════════════════════════════════════
E, R, dIC = d1['log_E'], d1['log_R'], d1['log_encIC']
pred_X = float(d1['pred_X'])
fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
for ch, (ax, lab, unit) in enumerate(zip(axes[:3], ['X', 'yaw', 'Y'], ['m', 'rad', 'm'])):
    ax.plot(t1, E[:, ch], color=C['red'], lw=0.9)
    ax.plot(t1, R[:, ch], color=C['blue'], lw=0.9, ls='--')
    ax.axhspan(-absorber, absorber, color='0.75', alpha=0.4)
    ax.set_ylabel(f'{lab} error [{unit}]'); ax.grid(True)
    dm.sci_axes(ax)
axes[0].annotate('encoder start', xy=(0.75, 0.80), xycoords='axes fraction',
                 fontsize=9, color=C['red'], fontweight='bold')
axes[0].annotate('true start', xy=(0.75, 0.12), xycoords='axes fraction',
                 fontsize=9, color=C['blue'], fontweight='bold')
axes[0].set_title('same model, same input -- ONLY the initial state differs')
axes[1].annotate('yaw dies out: it has a spring', xy=(2.5, 0), fontsize=8, color='0.35')
axes[2].annotate('X and Y park: no stiffness, nothing pulls them back',
                 xy=(6, R[-1, 2] * 0.4), fontsize=8, color='0.35')
ax = axes[3]
ax.plot(t1, dIC[:, 0], color=C['red'], lw=1.1)
ax.axhline(pred_X, color=C['pred'], ls='--', lw=1.2)
ax.annotate(f'measured: settles at {dIC[-1,0]*1e3:+.2f} mm', xy=(0.72, 0.80),
            xycoords='axes fraction', fontsize=9, color=C['red'], fontweight='bold')
ax.annotate(f'predicted: tau x start-velocity error = {pred_X*1e3:+.2f} mm',
            xy=(0.35, 0.60), xycoords='axes fraction', fontsize=9, color=C['pred'],
            fontweight='bold')
ax.axhspan(-absorber, absorber, color='0.75', alpha=0.4)
ax.set_ylabel('X error caused by\nthe encoder start [m]'); ax.set_xlabel('time [s]')
ax.grid(True); dm.sci_axes(ax)
ax.set_title('difference of the two runs above vs an independent prediction', fontsize=10)
fig.suptitle('Does the free-run error come from the encoder\'s estimated initial state?')
foot(fig, 'T1 | untrained baseline | grey band = absorber signal | true-start residual: the '
          'absorber the ANN must learn (model-vs-data match: G1; numerics floor 2e-6..6e-6, backup)')
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'g2_encoder_start.png'), dpi=150); plt.close(fig)

# ══ G3 -- constant-force universality (single dY panel, physical units) ══════
names = [str(n) for n in d2['bar_names']]
means_mN = w2force_mN(d2['bar_means'][:, 5])
rms_mN = w2force_mN(d2['bar_rms'][:, 5])
short = [n.replace('cold_trial', 'fresh nf').replace('warm_rung', 'continued nf')
          .replace('_lr1e-07_last', '').replace('_last', '').replace('gantry_drift', 'dissected run')
          .replace('0_nf', ' ').replace('1_nf', ' ').replace('2_nf', ' ').replace('3_nf', ' ')
         for n in names]
fig, ax = plt.subplots(figsize=(11, 5))
xpos = np.arange(len(names))
ax.bar(xpos, means_mN, yerr=np.abs(rms_mN), capsize=3, color=C['blue'], alpha=0.85)
ax.axhline(0, color='k', lw=0.9)
ax.set_xticks(xpos); ax.set_xticklabels(short, rotation=25, ha='right', fontsize=8)
ax.set_ylabel('learned constant force on Y [mN]\n(mean over a 2 s free-run; whisker = rms)')
ax.grid(True, axis='y')
ax.annotate('all 9 independently trained models push the SAME direction;\n'
            'longer training window -> smaller force, but never zero',
            xy=(0.45, 0.12), xycoords='axes fraction', fontsize=9, color='0.25')
ax.set_title('Is the constant force a one-run accident?')
foot(fig, 'V1 | force = m_head x (learned velocity increment)/ts | context: the excitation is 45 N '
          '| other state rows: 10-100x smaller, mixed sign (backup)')
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'g3_universality.png'), dpi=150); plt.close(fig)

# ══ G4 -- the split (train/val window fit + free-run) ════════════════════════
ALM = os.path.join(REPO, 'simulations', 'gantry_subnet', 'augmentation_linear_map')
CASES = [('continued training (window 400 = 0.1 s)',
          os.path.join(ALM, 'curriculum_70903', 'rung0_nf400_last.pth'), slice(0, 9)),
         ('fresh start (window 800 = 0.2 s)',
          os.path.join(ALM, 'trial_ckpts_71013', 'trial3_nf800_lr1e-07_last.pth'), slice(None))]
fig, axs = plt.subplots(2, 2, figsize=(12.5, 7), sharex='col')
for c, (title, path, seg) in enumerate(CASES):
    h = torch.load(path, map_location='cpu', weights_only=False)
    tr = np.asarray(h['Loss_train_nf'], dtype=float)
    vl = np.asarray(h['Loss_val_nf'], dtype=float)
    sv = np.asarray(h['Loss_val'], dtype=float)[seg]
    ep_t, ep_s = np.arange(len(tr)), np.arange(len(sv))
    aT, aB = axs[0, c], axs[1, c]
    aT.plot(ep_t, tr * 1e3, color=C['blue'], lw=1.3, marker='o', ms=3)
    aT.plot(ep_t, vl * 1e3, color=C['blue'], lw=1.3, marker='s', ms=3, ls='--')
    aT.annotate('train window fit', xy=(0.45, 0.86), xycoords='axes fraction',
                fontsize=9, color=C['blue'], fontweight='bold')
    aT.annotate('val window fit', xy=(0.45, 0.10), xycoords='axes fraction',
                fontsize=9, color=C['blue'], fontweight='bold')
    aT.set_ylabel('window fit [mm]' if c == 0 else ''); aT.grid(True)
    aT.set_title(f'{title}\nthe OBJECTIVE: healthy, train ~ val (no overfitting)', fontsize=9)
    aB.semilogy(ep_s, sv, color=C['red'], lw=1.3, marker='o', ms=3)
    aB.axhline(float(d3['ann_off']), color=C['grey'], ls='--', lw=1.2)
    aB.annotate('free-run error', xy=(0.45, 0.85), xycoords='axes fraction',
                fontsize=9, color=C['red'], fontweight='bold')
    aB.annotate('no ANN', xy=(0.05, 0.06), xycoords='axes fraction',
                fontsize=9, color=C['grey'], fontweight='bold')
    aB.set_ylabel('free-run error [m]' if c == 0 else ''); aB.set_xlabel('epoch')
    aB.grid(True, which='both')
    aB.set_title(f'the DEPLOYMENT metric: x{sv[-1]/float(d3["ann_off"]):.0f} worse than no ANN',
                 fontsize=9)
fig.suptitle('Does improving the training objective improve the deployed model?')
foot(fig, 'histories from the saved checkpoints | window fit: same window as training, encoder '
          're-init per window | free-run: cropped V1 validation | window fits comparable only within a column')
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'g4_split.png'), dpi=150); plt.close(fig)

# ══ G5 -- "increase the window": ratio bars ══════════════════════════════════
ann_off = float(d3['ann_off'])
warm = {n: d3[f'warm{n}_simrms'][-1] / ann_off for n in (400, 800, 1600, 2000)}
cold = {n: d3[f'cold{n}_simrms'][-1] / ann_off for n in (800, 1600, 2400, 3200)}
fig, ax = plt.subplots(figsize=(11, 5.5))
nfs = sorted(set(warm) | set(cold))
xw = 0.38
for i, n in enumerate(nfs):
    if n in warm:
        b = ax.bar(i - xw / 2, warm[n], width=xw, color='#ef6c00', alpha=0.9)
        ax.annotate(f'{warm[n]:.1f}x', xy=(i - xw / 2, warm[n]), ha='center',
                    va='bottom', fontsize=8)
    if n in cold:
        ax.bar(i + xw / 2, cold[n], width=xw, color=C['blue'], alpha=0.9)
        ax.annotate(f'{cold[n]:.1f}x', xy=(i + xw / 2, cold[n]), ha='center',
                    va='bottom', fontsize=8)
ax.axhline(1.0, color='k', lw=1.4)
ax.annotate('break-even: as good as NO ANN', xy=(0.02, 1.04), xycoords=('axes fraction', 'data'),
            fontsize=9, fontweight='bold')
ax.set_yscale('log')
ax.set_ylim(bottom=1.0)   # bars grow FROM break-even: height = how much worse than no ANN
ax.set_xticks(range(len(nfs)))
ax.set_xticklabels([f'{n}\n({n*ts:.2f} s)' for n in nfs])
ax.set_xlabel('training window length [samples] (seconds)')
ax.set_ylabel('free-run error after training / no ANN\n(1.0 = break-even, log scale)')
ax.grid(True, axis='y', which='both')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#ef6c00', label='continued training (one model, 70903)'),
                   Patch(color=C['blue'], label='fresh start per window (71013)')],
          fontsize=8, loc='upper right')
ax.set_title('Does a longer training window make the augmented model better than NO model?\n'
             '(helps monotonically, saturates at 1.3x, never crosses break-even; '
             '8-epoch budgets: bars are lower bounds)')
foot(fig, 'end-of-training free-run error on the cropped V1 validation | measured cost 6-47 s/batch '
          'for window 400-3200 | start/best trajectories: backup')
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'g5_window_ratio.png'), dpi=150); plt.close(fig)

# ══ G6 -- the counterfactual (small multiples + insets) ══════════════════════
e_deb = d2['e_deb'][:, 2]
v_err = d2['v_err']
w_dY = None  # panel (c) uses saved mean/rms only if full trace absent
fig = plt.figure(figsize=(12.5, 9))
gs = fig.add_gridspec(3, 3, height_ratios=[1.3, 1, 1])
titles = ['trained ANN', 'constant force removed', 'no ANN']
data = [e_full, e_deb, e_off]
cols = [C['red'], C['blue'], C['grey']]
ends = ['-2.9 cm', '0.02 cm', '0.02 cm']
for i in range(3):
    ax = fig.add_subplot(gs[0, i])
    ax.plot(t2, data[i] * 100, color=cols[i], lw=1.0)
    ax.set_ylim(-3.05, 0.35)
    ax.set_title(titles[i], fontsize=10, color=cols[i], fontweight='bold')
    ax.grid(True)
    ax.annotate(ends[i], xy=(0.97, 0.06 if i == 0 else 0.86), xycoords='axes fraction',
                ha='right', fontsize=9, color=cols[i], fontweight='bold')
    if i == 0:
        ax.set_ylabel('Y error [cm]')
    else:
        axi = ax.inset_axes([0.25, 0.12, 0.7, 0.5])
        axi.plot(t2, data[i] * 1e4, color=cols[i], lw=0.5)
        axi.set_title('zoom [x1e-4 m]: bounded', fontsize=7)
        axi.tick_params(labelsize=6); axi.grid(True)
    ax.set_xlabel('time [s]')
axv = fig.add_subplot(gs[1, :])
axv.plot(t2, v_err * 1e3, color=C['red'], lw=0.5)
axv.axhline(v_ss_pred * 1e3, color=C['pred'], ls='--', lw=1.4)
axv.annotate(f'black dash: predicted terminal velocity from the measured force = '
             f'{v_ss_pred*1e3:.1f} mm/s', xy=(0.98, 0.06), xycoords='axes fraction',
             ha='right', fontsize=9, color=C['pred'], fontweight='bold')
axv.set_ylabel('Y error velocity [mm/s]'); axv.grid(True)
axv.set_title('the VELOCITY stays bounded at the predicted level -> not an energy problem, '
              'a marginal integrator under a constant push', fontsize=9)
axf = fig.add_subplot(gs[2, :])
mean_mN = w2force_mN(float(d2['mean_w'][5]))
# The dY force trace is not in the npz; capture a short free-run (the force is stationary,
# d15: -0.7% over 12 s), reusing demo3's shadow machinery.
from demo3_shared import capture_force_trace  # noqa: E402  (small helper, below)
t_f, w_dY = capture_force_trace(fit_sys, seconds=2.0)
axf.plot(t_f, w2force_mN(w_dY), color='#6a1b9a', lw=0.5)
axf.axhline(0, color='k', lw=0.8)
axf.axhline(mean_mN, color=C['pred'], ls='--', lw=1.4)
endlab(axf, t_f[-1], mean_mN, f' mean = {mean_mN:.1f} mN', C['pred'], dy=8)
axf.annotate('never crosses zero (|mean|/rms = 0.997); vs the 45 N excitation this is a '
             '~0.06% parasitic constant:\ninvisible in the training window, fatal in free-run',
             xy=(0.03, 0.72), xycoords='axes fraction', fontsize=9, color='0.25')
axf.set_ylabel('equivalent force on Y [mN]'); axf.set_xlabel('time [s]')
axf.grid(True)
axf.set_title('the cause, measured: a constant parasitic force (first 2 s shown; '
              'stationary over 12 s)', fontsize=9)
fig.suptitle('Is the learned constant force the drift?  (three free-runs, one change each: '
             'removing ONE constant collapses the drift 133x)')
foot(fig, 'V1 | gantry_drift_last | force = m_head x (learned velocity increment)/ts; '
          'terminal velocity = force / damping (independent prediction)')
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'g6_counterfactual.png'), dpi=150); plt.close(fig)

# ══ G7 -- horizon (single curve) ═════════════════════════════════════════════
T, c_drift, floor, T_star = d5['T'], d5['c_drift'], float(d5['floor']), float(d5['T_star'])
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.loglog(T, c_drift, color=C['red'], lw=1.6)
ax.axhline(floor, color='0.4', ls=':', lw=1.2)
ax.annotate('the absorber signal (detectability floor)', xy=(T[2], floor * 1.25),
            fontsize=8, color='0.35')
for x, lab, dy in ((0.1, 'training window:\n0.11x floor -> BLIND', 4e-6),
                   (T_star, 'T*: drift crosses\nthe floor', 6e-5),
                   (0.5, 'memory wall\n(longest trainable window)', 8e-4)):
    ax.axvline(x, color='k', ls='--', lw=0.9)
    ax.annotate(lab, xy=(x * 1.07, dy), fontsize=8)
ax.set_xlabel('evaluation horizon [s]')
ax.set_ylabel('drift contribution to the Y error, RMS over [0, T] [m]')
ax.grid(True, which='both')
ax.set_title('At which horizon can training SEE the drift?\n'
             '(blind below T*; visible but tolerated/preferred by the windowed loss above it)')
foot(fig, 'gantry_drift_last | V1 | drift contribution = full-ANN trace minus constant-force-'
          'removed trace | in-window preference: d8/d12')
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'g7_horizon.png'), dpi=150); plt.close(fig)

print('G-deck saved to', FIG)
for f in sorted(os.listdir(FIG)):
    if f.startswith('g') and f.endswith('.png'):
        print('  ', f)
