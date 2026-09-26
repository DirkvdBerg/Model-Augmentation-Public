"""J7: the multisine band from the measured FRF of the Coulomb + MSD truth (EXCITATION-VALIDATION.md s15, 15b).

Inputs: outputs/j2_bla.npz, cases K1_Y-0.30 ... K1_Y+0.30 (B6 to B10: K1, production level, M = 10).
Per Y: the truth FRF (BLA) as the 3 x 3 in stage coordinates, its std, the analytic nominal and 10 %
detuned baseline with K1 at that Y, and the differences E = PS_truth - PS_model.
Average over Y (user, 2026-09-26): mean of |E|^2 and of sigma^2 over the five points.
T1: resolved where sqrt(mean |E|^2) > 2.45 sqrt(mean sigma^2) (95 % bound for a complex FRF, L8 slide 47).
Crossover: the first |L_jj| = 1 crossing (bandwidth), highest over the three channels, the five Y points and both loops
(nominal baseline and measured truth) with K1; the training multisine lies at or above it (user).
T2: weight w(f) = sum over the 9 entries of the resolved mean |E|^2 (L9 slide 13); the band is the
narrowest contiguous range above the crossover holding 90 % of w there (HEURISTIC, user); 80 and 95 % beside.
Writes outputs/j7_band.json, figures/fig11_differences_avg.png, figures/fig12_band_weight.png.
"""
__project_origin__ = "added"

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

import common as C                # noqa: E402

plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7, 'legend.fontsize': 6.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e4e3df', 'grid.linewidth': 0.5})
YS = [-0.30, -0.15, 0.0, 0.15, 0.30]
CH = ['X1', 'X2', 'Y']
Z95 = 2.45                                  # THEORY: 95 % bound for a complex-valued FRF (L8 slide 47)
SHARES = (0.80, 0.90, 0.95)                 # 0.90 decided (HEURISTIC, user); 0.80 and 0.95 as sensitivity

j2 = np.load(os.path.join(C.OUT, 'j2_bla.npz'))
f = j2['fl_K1_Y+0.00'][:, 1]                # middle line of each zippered triple
nf = len(f)
K1 = C.ctrl_frf(0.0, f)                     # K1: designed once at Y = 0 (DATA-DESIGN.md 5.11)
E = {'nominal': [], 'detuned': []}
S2 = []
Lcross = []


def last_crossing(fr, mag):
    """Loop crossover = the FIRST frequency where |L| falls through 1 (the controller bandwidth).
    A later re-crossing near a lightly damped resonance (the absorber lifts |L| above 1 again around
    213 Hz on the truth's Y loop) is not the bandwidth; the first run of this script took the last
    crossing and got 264 Hz on Y, which is that resonance."""
    idx = np.where((mag[:-1] >= 1.0) & (mag[1:] < 1.0))[0]
    return float(fr[idx[0] + 1]) if len(idx) else float('nan')


fg = np.logspace(0, 3, 3000)                # fine grid for the model crossover
Kg = C.ctrl_frf(0.0, fg)
for y in YS:
    tag = 'K1_Y%+.2f' % y
    assert np.allclose(j2['fl_' + tag][:, 1], f), tag
    PSt = j2['PS_' + tag] @ C.P                                   # logical inputs -> stage inputs
    S2.append(np.real(j2['vt_' + tag]) @ (C.P ** 2))              # variance of the BLA estimate, stage
    for k, th in (('nominal', C.THETA0), ('detuned', C.THETA0 * C.DETUNE_PROD)):
        _, PSm = C.loop(C.frf_zoh(*C.base_mck(th, y), f), K1)
        E[k].append(PSt - PSm)
    # crossovers: nominal baseline (fine grid) and measured truth, G_truth = PS (I - K PS)^-1
    Gb = C.frf_zoh(*C.base_mck(C.THETA0, y), fg)
    Gt = PSt @ np.linalg.inv(np.eye(3)[None] - K1 @ PSt)
    for j in range(3):
        Lcross.append(dict(Y=y, ch=CH[j], baseline=last_crossing(fg, np.abs(Gb[:, j, j] * Kg[:, j, j])),
                           truth=last_crossing(f, np.abs(Gt[:, j, j] * K1[:, j, j]))))
S2m = np.mean(S2, axis=0)
fc = float(np.nanmax([[c['baseline'], c['truth']] for c in Lcross]))
print('crossover with K1 (first |L_jj| = 1, the bandwidth), per Y and channel [Hz], baseline / truth:')
for c in Lcross:
    print('  Y %+.2f %-2s  %6.1f / %6.1f' % (c['Y'], c['ch'], c['baseline'], c['truth']))
print('-> band lower bound f_c = %.1f Hz' % fc)

res = dict(f_c=fc, crossovers=Lcross, comparisons={})
above = f >= fc
for k in ('nominal', 'detuned'):
    Em2 = np.mean(np.abs(np.array(E[k])) ** 2, axis=0)             # mean over Y of |E|^2, (nf, 3, 3)
    resolved = np.sqrt(Em2) > Z95 * np.sqrt(S2m)                  # T1
    w_all = np.sum(Em2 * resolved, axis=(1, 2))                   # T2 weight per frequency
    share_below = float(w_all[~above].sum() / w_all.sum())
    w = np.where(above, w_all, 0.0)
    cw = np.concatenate([[0.0], np.cumsum(w)])
    tot = cw[-1]
    bands = {}
    for s in SHARES:
        best = (np.inf, 0, 0)
        i0s = np.where(above)[0]
        for a in i0s:                                             # narrowest window holding share s
            b = np.searchsorted(cw, cw[a] + s * tot)
            if b <= nf and f[min(b, nf) - 1] - f[a] < best[0]:
                best = (f[min(b, nf) - 1] - f[a], a, min(b, nf) - 1)
        bands['%d%%' % round(100 * s)] = [float(f[best[1]]), float(f[best[2]])]
    # plant features inside the 90 % band? (dip of Y <- F_Y; notch of the cross entries)
    m = (f > 120) & (f < 400)
    dip = float(f[m][np.argmin(np.mean([np.abs((j2['PS_K1_Y%+.2f' % y] @ C.P)[m, 2, 2]) for y in YS], axis=0))])
    notch = [float(f[m][np.argmin(np.abs((j2['PS_K1_Y%+.2f' % y] @ C.P)[m, 0, 2]))]) for y in YS]
    b90 = bands['90%']
    res['comparisons'][k] = dict(bands=bands, share_of_resolved_weight_below_fc=share_below,
                                 resolved_fraction_per_entry={'%s<-%s' % (CH[o], CH[i]): float(resolved[:, o, i].mean())
                                                              for o in range(3) for i in range(3)},
                                 dip_Hz=dip, notch_Y_to_X1_Hz=notch,
                                 features_inside_90=bool(b90[0] <= dip <= b90[1] and all(b90[0] <= n <= b90[1] for n in notch)))
    print('\n%s baseline vs Coulomb + MSD:' % k)
    print('  resolved weight below f_c (the references\' job): %.1f %% of the total' % (100 * share_below))
    for s, b in bands.items():
        print('  band holding %s of the resolved weight above f_c: %.1f to %.1f Hz' % (s, b[0], b[1]))
    print('  plant features: dip %.1f Hz (Y<-F_Y, mean over Y); notch X1<-F_Y per Y %s Hz; inside the 90 %% band: %s'
          % (dip, [round(n, 1) for n in notch], res['comparisons'][k]['features_inside_90']))
    res['comparisons'][k]['Em2'] = None
    if k == 'nominal':
        Em2_nom, res_nom, w_nom = Em2, resolved, w_all
    else:
        Em2_det, res_det, w_det = Em2, resolved, w_all

# line density at the training multisine's resolution (1/12 Hz, 12 s records) against the feature widths
print('\nline density: 1/12 Hz spacing gives >= 4 lines per 3 dB width for any width >= 0.33 Hz (Geerardyn et al. 2013)')
with open(os.path.join(C.OUT, 'j7_band.json'), 'w') as fh:
    json.dump(res, fh, indent=1)

# == figure 11: averaged differences, 3 x 3, resolved parts, crossover and 90 % bands
fig, axs = plt.subplots(3, 3, figsize=(7.0, 6.6), sharex=True, sharey=True)
b_nom, b_det = res['comparisons']['nominal']['bands']['90%'], res['comparisons']['detuned']['bands']['90%']
for o in range(3):
    for i in range(3):
        ax = axs[o, i]
        ax.axvspan(*b_nom, color='#2a78d6', alpha=0.10, lw=0)
        ax.axvspan(*b_det, color='#eb6834', alpha=0.10, lw=0)
        ax.loglog(f, np.sqrt(Em2_nom[:, o, i]), color='#2a78d6', lw=0.9, label='nominal vs truth, rms over Y')
        ax.loglog(f, np.sqrt(Em2_det[:, o, i]), color='#eb6834', lw=0.9, ls='--', label='detuned vs truth, rms over Y')
        ax.loglog(f, Z95 * np.sqrt(S2m[:, o, i]), color='#8a8983', lw=0.7, label='2.45 sigma (T1 threshold)')
        ax.axvline(fc, color='black', lw=0.7, ls='-.')
        ax.set_xlim(1, 1000); ax.set_ylim(1e-12, 1e-6)
        if o == 0:
            ax.set_title('input F_%s' % CH[i])
        if i == 0:
            ax.set_ylabel('|difference| [m/N]\noutput %s' % CH[o])
        if o == 2:
            ax.set_xlabel('frequency [Hz]')
h, l = axs[0, 0].get_legend_handles_labels()
h += [plt.Rectangle((0, 0), 1, 1, color='#2a78d6', alpha=0.25), plt.Rectangle((0, 0), 1, 1, color='#eb6834', alpha=0.25)]
l += ['90 %% band, nominal: %.0f to %.0f Hz' % tuple(b_nom), '90 %% band, detuned: %.0f to %.0f Hz' % tuple(b_det)]
fig.legend(h, l, loc='lower center', ncol=3, frameon=False)
fig.suptitle('FRF differences to the Coulomb + MSD truth, K1, averaged over Y = -0.30 to +0.30 m (rms); dash-dot: crossover %.0f Hz' % fc,
             fontsize=7.5)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(os.path.join(C.HERE, 'figures', 'fig11_differences_avg.png'), dpi=200)
plt.close(fig)

# == figure 12: cumulative resolved weight over frequency, with the crossover and the bands
fig, ax = plt.subplots(figsize=(7.0, 2.8))
for w, col, lab in ((w_nom, '#2a78d6', 'nominal vs truth'), (w_det, '#eb6834', 'detuned vs truth')):
    ax.semilogx(f, np.cumsum(w) / w.sum(), color=col, lw=1.1, label=lab)
ax.axvline(fc, color='black', lw=0.7, ls='-.', label='crossover %.0f Hz' % fc)
ax.axvspan(*b_nom, color='#2a78d6', alpha=0.10, lw=0)
ax.set_xlim(1, 1000); ax.set_ylim(0, 1.02)
ax.set_xlabel('frequency [Hz]'); ax.set_ylabel('cumulative resolved |E|^2 weight')
ax.legend(frameon=False, loc='upper left')
fig.tight_layout()
fig.savefig(os.path.join(C.HERE, 'figures', 'fig12_band_weight.png'), dpi=200)
print('\nwrote outputs/j7_band.json, figures/fig11_differences_avg.png, figures/fig12_band_weight.png')
