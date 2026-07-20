"""make_meeting_figures.py -- six clean, one-message-per-figure meeting figures for Jan.

Re-plots from SAVED data (no re-training, no re-simulation). Each figure answers ONE of Jan's meeting
points with a single clear message and large fonts. Diagnostic figures (v3x0sgd multiseed, v4 2x3 grid,
etc.) stay as backup; these are the presentation cut.

  fig1  Theme A   the correction the ANN SHOULD make is zero-mean; the one it learns is not
  fig2  Theme D   within the window the K=0 error is a LINEAR ramp (no explosion, no fading memory)
  fig3  Theme C   the DC is SYSTEMATIC: same sign across 3 unfixed seeds
  fig4  Theme C   Adam is the cause: SGD builds ~2000x less DC at the same loss  (NEW figure)
  fig5  reframe   the dominant drift is the ANN DESTABILIZING, not a DC
  fig6  direction stability-preserving prototype: control diverges; L=1 cap was non-binding

Sources: v4_results.npz, v3b_perstep_seed{0,1,2}.npz, v3x0ctrl_encoder_seed0.npz (Adam),
v3x0sgd_encoder_seed0.npz (SGD), v5_dc_null_counterfactual.npz, v6_lipschitz_sweep.npz.
Output: figures/meeting/meeting_fig{1..6}_*.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14,
                     'legend.fontsize': 12, 'figure.dpi': 150})
C_ANN, C_OFF, C_REQ = '#c0392b', '#27ae60', '#2c3e50'
C_ADAM, C_SGD = '#c0392b', '#2980b9'

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, 'data')
OUT = os.path.join(HERE, 'figures', 'meeting'); os.makedirs(OUT, exist_ok=True)
L8 = ['X', 'Theta', 'Y', 'dX', 'dTheta', 'dY', 'delta_a', 'vdelta_a']
load = lambda f: np.load(os.path.join(D, f), allow_pickle=True)


def fig0_system_zero_mean():
    """The SYSTEM is zero-mean: adding the hidden MSD shifts no signal's mean (with vs without, 12 records)."""
    o = load('v1b_dmean.npz')
    dmean, se = o['dmean'], o['se']                    # (records, channels)
    sig = np.abs(dmean) / (2 * se + 1e-30)             # significance; 1.0 = 95% confidence interval
    sig_max = sig.max(axis=0)                          # worst record per channel (conservative)
    sel = [3, 4, 5, 6, 7, 8, 9, 10, 11]               # logical X,Theta,Y ; dX,dTheta,dY ; F_X1,F_X2,F_Y
    names = ['X', 'Theta', 'Y', 'dX', 'dTheta', 'dY', 'F_X1', 'F_X2', 'F_Y']
    vals = [sig_max[i] for i in sel]
    x = np.arange(len(sel))
    fh, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.bar(x, vals, color=C_OFF, width=0.62)
    ax.axhline(1.0, color=C_ANN, ls='--', lw=1.6, label='95% significance threshold')
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel('mean-shift significance  ( |Δmean| / 95% CI )'); ax.set_ylim(0, 1.2)
    ax.set_xlabel('signal channel  (positions, velocities, forces)')
    ax.set_title('The system is zero-mean: adding the hidden mass shifts no signal\'s mean')
    ax.legend(loc='upper right'); ax.grid(True, axis='y', alpha=0.3)
    da_m = float(np.abs(o['da_mean']).max()); da_s = float(o['da_std'].mean())
    fh.text(0.5, -0.02, f'With vs without MSD, 12 records, open-loop 20 kHz. Every bar << 1: no '
            f'statistically significant mean shift on any signal.\nThe absorber itself is zero-mean '
            f'(|mean| {da_m:.0e} m vs std {da_s:.0e} m).', ha='center', fontsize=9.5, style='italic')
    fh.tight_layout(); fh.savefig(os.path.join(OUT, 'meeting_fig0_system_zero_mean.png'), bbox_inches='tight')
    plt.close(fh)


def fig1_zero_mean():
    """Fraction-DC (|mean|/rms, unit-free) of the LEARNED correction vs the REQUIRED correction, K=0 rows."""
    o = load('v4_results.npz'); mw, rw = o['mean_w'], o['rms_w']
    rows = [0, 2, 3, 5]; names = ['X', 'Y', 'dX', 'dY']
    ann = [abs(mw[r]) / (rw[r] + 1e-30) for r in rows]
    req = [0.0, 0.0, 0.0, 0.0]              # f04b: required |mean|/rms = 0.000 on every row
    x = np.arange(len(rows)); w = 0.38
    fh, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.bar(x - w/2, ann, w, color=C_ANN, label='learned ANN correction')
    ax.bar(x + w/2, req, w, color=C_REQ, label='required correction (truth - baseline)')
    for xi, v in zip(x - w/2, ann):
        ax.text(xi, v - 0.05, f'{v:.2f}', ha='center', va='top', fontsize=12, color='white', fontweight='bold')
    for xi in x + w/2:
        ax.text(xi, 0.03, '0.00', ha='center', va='bottom', fontsize=11, color=C_REQ)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel('fraction constant  ( |mean| / rms )'); ax.set_ylim(0, 1.28)
    ax.set_xlabel('K=0 correction row')
    ax.set_title('Theme A: the required correction is zero-mean; the learned one is not', pad=10)
    ax.legend(loc='upper center', ncol=2, framealpha=0.95); ax.grid(True, axis='y', alpha=0.3)
    fh.text(0.5, -0.01, 'Physics carries no meaningful DC: truth-minus-baseline DC <= 1e-7 (v1f), '
            '~100-1000x below the ANN\'s parked DC.', ha='center', fontsize=10, style='italic')
    fh.tight_layout(); fh.savefig(os.path.join(OUT, 'meeting_fig1_zero_mean.png'), bbox_inches='tight')
    plt.close(fh)


def fig2_linear_ramp():
    """Within-window error vs step: Y (K=0) ramps linearly, Theta (sprung) parks. No explosion."""
    o = load('v4_results.npz'); k = o['k']; ef, eo = o['ef'], o['eo']
    fits = eval(str(o['fits'][0])); aY, r2Y = fits['Y']['a_lin'], fits['Y']['r2_lin']
    nf_b = int(o['nf_boundary'])
    fh, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(k, ef[:, 2] * 1e3, color=C_ANN, lw=2.0, label='Y error (K=0, full ANN)')
    ax.plot(k, aY * k * 1e3, color='k', lw=1.2, ls='--', label=f'linear fit  (R²={r2Y:.3f})')
    ax.plot(k, ef[:, 1] * 1e3, color=C_OFF, lw=2.0, label='Theta error (sprung)')
    ax.axvline(nf_b, color='0.5', ls=':', lw=1.2)
    ax.text(nf_b - 8, ax.get_ylim()[1] * 0.9, 'nf=400 window', rotation=90, va='top', ha='right', fontsize=10, color='0.4')
    ax.set_xlabel('rollout step within the window'); ax.set_ylabel('|error|  [mm]')
    ax.set_title('Theme D: K=0 error is a LINEAR ramp — no explosion, no fading memory')
    ax.legend(loc='upper left'); ax.grid(True, alpha=0.3)
    fh.tight_layout(); fh.savefig(os.path.join(OUT, 'meeting_fig2_linear_ramp.png'), bbox_inches='tight')
    plt.close(fh)


def fig3_systematic_seeds():
    """dY DC vs update step, 3 unfixed seeds overlaid: same sign = systematic (not wander)."""
    fh, ax = plt.subplots(figsize=(8.6, 5.0))
    cols = ['#c0392b', '#e67e22', '#8e44ad']
    for s, c in zip([0, 1, 2], cols):
        m = load(f'v3b_perstep_seed{s}.npz')['mean']
        ax.plot(np.arange(m.shape[0]), m[:, 5] * 1e6, color=c, lw=1.6, label=f'seed {s}')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('parameter update step (epoch 1)'); ax.set_ylabel('ANN dY output mean (DC)  [x1e-6]')
    ax.set_title('Theme C: the DC is SYSTEMATIC — same sign across independent seeds')
    ax.legend(loc='lower left'); ax.grid(True, alpha=0.3)
    ax.text(150, -0.7, 'born by ~step 13,\nall 3 seeds go negative (same sign)', fontsize=11,
            ha='left', va='center', color='0.3')
    fh.tight_layout(); fh.savefig(os.path.join(OUT, 'meeting_fig3_systematic_seeds.png'), bbox_inches='tight')
    plt.close(fh)


def fig4_adam_vs_sgd():
    """THE new figure: Adam vs SGD dY DC on the SAME axes -> ~2000x gap, same loss."""
    ad = load('v3x0ctrl_encoder_seed0.npz')['mean'][:, 5]
    sg = load('v3x0sgd_encoder_seed0.npz')['mean'][:, 5]
    ka, ks = np.arange(len(ad)), np.arange(len(sg))
    fh, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(ka, ad * 1e6, color=C_ADAM, lw=2.2, label='Adam')
    ax.plot(ks, sg * 1e6, color=C_SGD, lw=2.2, label='SGD')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('parameter update step'); ax.set_ylabel('ANN dY output mean (DC)  [x1e-6]')
    ax.set_title('Theme C: Adam is the cause — SGD builds ~2000x less DC at the SAME loss', pad=12)
    ax.set_ylim(-4.4, 0.7)
    ax.legend(loc='lower left')
    ax.annotate(f'Adam: {ad[-20:].mean():.2e}', xy=(len(ad)-1, ad[-1]*1e6), xytext=(150, -2.6),
                color=C_ADAM, fontsize=11, arrowprops=dict(arrowstyle='->', color=C_ADAM))
    ax.annotate(f'SGD: {sg[-20:].mean():.2e}  (flat at 0)', xy=(200, 0), xytext=(95, 0.42),
                color=C_SGD, fontsize=11, arrowprops=dict(arrowstyle='->', color=C_SGD))
    ax.grid(True, alpha=0.3)
    fh.tight_layout(); fh.savefig(os.path.join(OUT, 'meeting_fig4_adam_vs_sgd.png'), bbox_inches='tight')
    plt.close(fh)


def fig4b_loss_curves():
    """Adam vs SGD training loss -- they overlap => the DC (fig4) is NOT a fit-quality difference."""
    adL = np.asarray(load('v3x0ctrl_encoder_seed0.npz')['loss'])
    sgL = np.asarray(load('v3x0sgd_encoder_seed0.npz')['loss'])
    rw = 15
    roll = lambda x: np.convolve(x, np.ones(rw) / rw, mode='valid')
    ka, ks = np.arange(len(adL)), np.arange(len(sgL))
    fh, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(ka, adL, color=C_ADAM, lw=0.6, alpha=0.30)                 # raw (noisy minibatch loss)
    ax.plot(ks, sgL, color=C_SGD, lw=0.6, alpha=0.30)
    ax.plot(ka[rw - 1:], roll(adL), color=C_ADAM, lw=2.6, label=f'Adam  (end {adL[-20:].mean():.2e})')
    ax.plot(ks[rw - 1:], roll(sgL), color=C_SGD, lw=2.6, label=f'SGD   (end {sgL[-20:].mean():.2e})')
    ax.set_yscale('log')
    ax.set_xlabel('parameter update step'); ax.set_ylabel('training loss  (windowed sim-MSE)')
    ax.set_title('Same loss: Adam and SGD train to the same error\n(so the DC in fig 4 is not a fit-quality difference)')
    ax.legend(loc='upper right'); ax.grid(True, alpha=0.3, which='both')
    fh.text(0.5, -0.02, 'Faint = raw (noisy minibatch) loss; bold = 15-step running mean. The curves '
            'overlap -> SGD is not stalled; it fits as well as Adam, without the DC.', ha='center',
            fontsize=9.5, style='italic')
    fh.tight_layout(); fh.savefig(os.path.join(OUT, 'meeting_fig4b_loss_curves.png'), bbox_inches='tight')
    plt.close(fh)


def fig5_reframe():
    """3-bar (ANN-off / full / DC-nulled) long-horizon drift for X and Y: Y drift is dynamic, not DC."""
    o = load('v5_dc_null_counterfactual.npz')
    D_off = o['D_off']; D = o['D']  # D[alpha][channel]; alpha=0 full, alpha=-1 DC-nulled
    # values in mm
    Xvals = [D_off[0], D[0][0], D[-1][0]]
    Yvals = [D_off[2], D[0][2], D[-1][2]]
    labels = ['ANN off\n(baseline)', 'full ANN', 'DC nulled']
    cols = [C_OFF, C_ANN, '#e67e22']
    fh, ax = plt.subplots(1, 2, figsize=(11, 5.0))
    for a, vals, name in zip(ax, [Xvals, Yvals], ['X (minor axis)', 'Y (dominant axis)']):
        a.bar(labels, np.array(vals) * 1e3, color=cols)
        for i, v in enumerate(vals):
            a.text(i, v * 1e3, f'{v*1e3:.2f}', ha='center', va='bottom', fontsize=11)
        a.set_ylabel('drift  [mm]'); a.set_title(name); a.grid(True, axis='y', alpha=0.3)
    ax[0].annotate('DC nulled -> back to baseline', xy=(2, Xvals[2]*1e3), fontsize=10, ha='center',
                   xytext=(1.5, max(Xvals)*1e3*0.7), color='0.3')
    ax[1].annotate('DC nulled -> WORSE\n(drift is the dynamic output)', xy=(2, Yvals[2]*1e3),
                   fontsize=10, ha='center', xytext=(1.0, Yvals[2]*1e3*0.8), color='0.3')
    fh.suptitle('Reframe: the dominant (Y) drift is the ANN destabilizing the free-run, not a DC', fontsize=15)
    fh.tight_layout(); fh.savefig(os.path.join(OUT, 'meeting_fig5_reframe.png'), bbox_inches='tight')
    plt.close(fh)


def fig6_direction():
    """v6: control diverges (Y ratio 114x); L=1 cap non-binding (ratio unchanged, fit identical)."""
    o = load('v6_lipschitz_sweep.npz')
    lab = [str(x) for x in o['labels']]
    df, do, fit = o['drift_full'], o['drift_off'], o['fit']
    ratio = [df[i][2] / do[i][2] for i in range(len(lab))]
    names = ['off\n(free ANN)', 'L=1.0\n(cap)']
    fh, ax = plt.subplots(1, 2, figsize=(11, 5.0))
    ax[0].bar(names, ratio, color=[C_ANN, C_SGD])
    for i, v in enumerate(ratio):
        ax[0].text(i, v, f'{v:.0f}x', ha='center', va='bottom', fontsize=13)
    ax[0].axhline(1.0, color='k', ls='--', lw=1.0); ax[0].set_ylabel('Y drift  full / ANN-off')
    ax[0].set_title('Divergence (1 = baseline)'); ax[0].grid(True, axis='y', alpha=0.3)
    ax[1].bar(names, fit * 1e4, color=[C_ANN, C_SGD])
    for i, v in enumerate(fit):
        ax[1].text(i, v*1e4, f'{v:.2e}', ha='center', va='bottom', fontsize=11)
    ax[1].set_ylabel('windowed fit nf-RMS  [x1e-4]'); ax[1].set_title('Fit (identical -> cap non-binding)')
    ax[1].grid(True, axis='y', alpha=0.3)
    fh.suptitle('Direction: stability-preserving prototype -- control diverges; L=1 cap did not bind (need tighter)', fontsize=14)
    fh.tight_layout(); fh.savefig(os.path.join(OUT, 'meeting_fig6_direction.png'), bbox_inches='tight')
    plt.close(fh)


def main():
    fig0_system_zero_mean(); fig1_zero_mean(); fig2_linear_ramp(); fig3_systematic_seeds()
    fig4_adam_vs_sgd(); fig4b_loss_curves(); fig5_reframe(); fig6_direction()
    print('done ->', OUT)
    for f in sorted(os.listdir(OUT)):
        print('  ', f)


if __name__ == '__main__':
    main()
