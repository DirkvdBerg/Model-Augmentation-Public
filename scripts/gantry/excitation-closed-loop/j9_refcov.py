"""J9: do the planned training references cover what lies below the crossover? (EXCITATION-VALIDATION.md s17)

Inputs: outputs/j9_refs.mat (matlab/ref_planned.m: the 18 training references of DATA-DESIGN.md 7.1, 4 kHz),
outputs/j2_bla.npz (K1 truth FRFs B6 to B10), outputs/j7_band.json (crossover f_c).
T3: sigma_data^2 = sigma^2 ESD_meas / ESD_ref per logical column, then to stage entries; pass where a resolved
difference below f_c satisfies |E| > 2.45 sigma_data. Friction: velocity coverage of the references.
Writes outputs/j9_refcov.json and figures/fig13_t3_references.png.
"""
__project_origin__ = "added"

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
from scipy.io import loadmat      # noqa: E402

import common as C                # noqa: E402

plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7, 'legend.fontsize': 6.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e4e3df', 'grid.linewidth': 0.5})
YS = [-0.30, -0.15, 0.0, 0.15, 0.30]
CH = ['X1', 'X2', 'Y']
Z95 = 2.45                                  # THEORY: 95 % bound for a complex-valued FRF (L8 slide 47)
A_MEAS = np.array([240.0, 87.0, 180.0])     # measurement multisine rms per logical channel (bla_truth.m, H)
T_MEAS = 10 * 2 * 2.0                       # M realisations x P measured periods x 2 s (B6 to B10)
B_MEAS = 999.0                              # measurement band 1 to 1000 Hz
CELL = 1.5                                  # FRF line spacing per column (zippered 0.5 Hz grid)
BANDS = [(1.0, 20.0), (20.0, 50.0), (50.0, None)]
V_BRK = 1e-3                                # the truth's Karnopp stick band (D-209)
A_CONST = 0.05                              # HEURISTIC: inertial force <= ~1.6 N on 32 kg vs 11.6 to 18.4 N Coulomb (s17)
T_MIN = 0.020                               # HEURISTIC: minimum duration of a constant-speed segment or a dwell gap
LEVEL_SEP = 0.1                             # HEURISTIC: levels distinct by 0.1 m/s
SPEED_BINS = [V_BRK, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
T_ACTIVE = (0.5, 10.5)                      # active window [s] (gtd_config.m: 0.5 s hold, 10 s active)

fc = json.load(open(os.path.join(C.OUT, 'j7_band.json')))['f_c']
j2 = np.load(os.path.join(C.OUT, 'j2_bla.npz'))
f_all = j2['fl_K1_Y+0.00'][:, 1]
sel = f_all < fc
f = f_all[sel]
K1 = C.ctrl_frf(0.0, f)

# == differences and measurement variance, averaged over the five Y points (as j7_band.py)
E2 = {'nominal': [], 'detuned': []}
VT, S2 = [], []
for y in YS:
    tag = 'K1_Y%+.2f' % y
    PSt = j2['PS_' + tag][sel] @ C.P
    VT.append(np.real(j2['vt_' + tag][sel]))                     # logical columns
    S2.append(np.real(j2['vt_' + tag][sel]) @ (C.P ** 2))
    for k, th in (('nominal', C.THETA0), ('detuned', C.THETA0 * C.DETUNE_PROD)):
        _, PSm = C.loop(C.frf_zoh(*C.base_mck(th, y), f), K1)
        E2[k].append(np.abs(PSt - PSm) ** 2)
VTm, S2m = np.mean(VT, axis=0), np.mean(S2, axis=0)
Em2 = {k: np.mean(v, axis=0) for k, v in E2.items()}

# == reference energy spectral density per logical channel, summed over the training records
d = loadmat(os.path.join(C.OUT, 'j9_refs.mat'), simplify_cells=True)
R, dt, ids = np.asarray(d['R'], float), float(d['dt']), [str(s) for s in d['ids']]
n, N4 = R.shape[0], R.shape[1]
fk = np.fft.rfftfreq(N4, dt)
mk = (fk > 0) & (fk < fc + CELL)
fkm = fk[mk]
Kk = C.ctrl_frf(0.0, fkm)
zk = np.exp(1j * 2 * np.pi * fkm * dt)
esd = np.zeros((n, len(fkm), 3))
for i in range(n):
    # THEORY: records start and end at rest, so the difference sequence has finite support and
    # X(w) = D(w) / (e^{jw} - 1) exactly for w > 0 (no leakage, no window); continuous FT ~ dt X
    Dk = np.fft.rfft(np.diff(R[i], axis=0), n=N4, axis=0)[mk]
    Xr = dt * Dk / (zk - 1.0)[:, None]
    W = np.einsum('kij,kj->ki', Kk, Xr)                          # w = K1 r at the plant input, stage [N s]
    U = W @ C.P.T                                                # logical channels f = P w
    esd[i] = 2.0 * np.abs(U) ** 2                                # THEORY: one-sided energy spectral density [N^2 s / Hz]
ESD_ref_k = esd.sum(0)
ESD_ref = np.array([ESD_ref_k[(fkm >= fl - CELL / 2) & (fkm < fl + CELL / 2)].mean(0) for fl in f])   # per FRF line cell
ESD_meas = A_MEAS ** 2 * T_MEAS / B_MEAS                        # THEORY: one-sided PSD A^2/B times the measured duration

# == T3
with np.errstate(divide='ignore'):
    ratio = ESD_meas[None, :] / ESD_ref                          # (nf, 3 logical)
VTd = VTm * ratio[:, None, :]                                    # THEORY: variance ~ 1 / input energy (L7; L11 slide 37)
S2d = VTd @ (C.P ** 2)
res = dict(f_c=fc, gate_note='sweeps and Lissajous exceed the 7.1 peak velocities (generator fade); see log',
           comparisons={})
for k in ('nominal', 'detuned'):
    resolved = np.sqrt(Em2[k]) > Z95 * np.sqrt(S2m)              # T1
    passed = resolved & (np.sqrt(Em2[k]) > Z95 * np.sqrt(S2d))   # T3
    w = Em2[k]
    tot = float((w * resolved).sum())
    out = dict(share_pass=float((w * passed).sum() / tot) if tot > 0 else float('nan'), bands={}, entries={})
    for lo, hi in BANDS:
        hi_ = fc if hi is None else hi
        b = ((f >= lo) & (f < hi_))[:, None, None]
        tb = float((w * resolved * b).sum())
        out['bands']['%g-%g' % (lo, hi_)] = dict(resolved_share_of_below_fc=tb / tot if tot > 0 else float('nan'),
                                                 pass_share=float((w * passed * b).sum() / tb) if tb > 0 else float('nan'))
    for o in range(3):
        for i in range(3):
            te = float((w[:, o, i] * resolved[:, o, i]).sum())
            out['entries']['%s<-F_%s' % (CH[o], CH[i])] = float((w[:, o, i] * passed[:, o, i]).sum() / te) if te > 0 else float('nan')
    res['comparisons'][k] = out
    print('\n%s baseline vs Coulomb + MSD, below f_c = %.1f Hz:' % (k, fc))
    print('  share of the resolved weight that passes T3: %.1f %%' % (100 * out['share_pass']))
    for bk, bv in out['bands'].items():
        print('  %-10s Hz: %5.1f %% of the resolved weight below f_c, of which %5.1f %% passes'
              % (bk, 100 * bv['resolved_share_of_below_fc'], 100 * bv['pass_share']))
    print('  per entry (pass share of its resolved weight): ' +
          ', '.join('%s %.0f %%' % (e, 100 * v) if np.isfinite(v) else '%s -' % e for e, v in out['entries'].items()))

# which records deliver the reference energy, per logical channel and band
print('\nreference energy per logical channel [sym, anti, Y], share by record, per band:')
cls = {'TR-S': 'standstill', 'TR-Y': 'Y sweeps', 'TR-P': 'S-curve moves', 'TR-L': 'Lissajous', 'TR-T': 'ILC shapes'}
share = {}
for lo, hi in BANDS:
    hi_ = fc if hi is None else hi
    b = (fkm >= lo) & (fkm < hi_)
    totc = esd[:, b].sum((0, 1))
    row = {}
    for pre, name in cls.items():
        ii = [j for j, s in enumerate(ids) if s.startswith(pre)]
        row[name] = (esd[ii][:, b].sum((0, 1)) / totc).tolist()
    share['%g-%g' % (lo, hi_)] = row
    print('  %g-%.0f Hz: ' % (lo, hi_) + '; '.join('%s %s' % (nm, np.array2string(np.array(v) * 100, precision=0)) for nm, v in row.items()))
res['energy_share_by_class'] = share
print('median ESD_ref / ESD_meas per band [sym, anti, Y]:')
for lo, hi in BANDS:
    hi_ = fc if hi is None else hi
    b = (f >= lo) & (f < hi_)
    print('  %g-%.0f Hz: %s' % (lo, hi_, np.array2string(np.median(1.0 / ratio[b], axis=0), precision=3)))

# == friction coverage from the reference velocities, stage axes X1, X2, Y, over the training set
t = np.arange(N4) * dt
act = (t >= T_ACTIVE[0]) & (t < T_ACTIVE[1])
nmin = int(round(T_MIN / dt))


def runs(mask):
    """start, stop index pairs of True runs"""
    m = np.concatenate([[False], mask, [False]]).astype(int)
    e = np.flatnonzero(np.diff(m))
    return list(zip(e[::2], e[1::2]))


fric = {}
for ax, name in enumerate(CH):
    lv = {+1: [], -1: []}
    rev_dwell = rev_direct = 0
    t_pos = t_neg = t_dwell = 0.0
    n_dwell = 0
    hist = np.zeros((2, len(SPEED_BINS) - 1))
    for i in range(n):
        v = np.gradient(R[i, :, ax], dt)
        a = np.gradient(v, dt)
        v, a = v[act], a[act]
        t_pos += dt * np.sum(v > V_BRK); t_neg += dt * np.sum(v < -V_BRK)
        still = np.abs(v) <= V_BRK
        t_dwell += dt * still.sum()
        n_dwell += sum(1 for s, e in runs(still) if e - s >= nmin)
        for s, e in runs((np.abs(a) < A_CONST) & (np.abs(v) > V_BRK)):
            if e - s >= nmin:
                lv[int(np.sign(v[s:e].mean()))].append(float(np.abs(v[s:e]).mean()))
        sg = np.sign(v) * (np.abs(v) > V_BRK)
        nz = np.flatnonzero(sg)
        for p, q in zip(nz[:-1], nz[1:]):
            if sg[p] != sg[q]:
                if q - p >= nmin:
                    rev_dwell += 1
                else:
                    rev_direct += 1
        hist[0] += dt * np.histogram(v[v > 0], SPEED_BINS)[0]
        hist[1] += dt * np.histogram(-v[v < 0], SPEED_BINS)[0]
    levels = {}
    for sgn, vals in lv.items():
        vals = sorted(vals)
        grp = []
        for x in vals:
            if not grp or x - grp[-1][0] >= LEVEL_SEP:
                grp.append([x])
            else:
                grp[-1].append(x)
        levels['+' if sgn > 0 else '-'] = [round(float(np.mean(g)), 3) for g in grp]
    ok = dict(both_directions=bool(t_pos > 0 and t_neg > 0),
              three_levels=bool(len(levels['+']) >= 3 and len(levels['-']) >= 3),
              reversals_both_kinds=bool(rev_dwell > 0 and rev_direct > 0), dwell=bool(n_dwell > 0))
    fric[name] = dict(t_pos_s=t_pos, t_neg_s=t_neg, t_dwell_s=t_dwell, n_dwell=n_dwell, levels=levels,
                      reversals_through_dwell=rev_dwell, reversals_direct=rev_direct,
                      speed_hist_s={'+': hist[0].tolist(), '-': hist[1].tolist(), 'edges': SPEED_BINS}, criteria=ok)
    print('\nfriction coverage %s: moving + %.1f s, - %.1f s, dwell %.1f s (%d dwells >= 20 ms); reversals %d through dwell, %d direct'
          % (name, t_pos, t_neg, t_dwell, n_dwell, rev_dwell, rev_direct))
    print('  constant-speed levels + %s' % levels['+'])
    print('  constant-speed levels - %s' % levels['-'])
    print('  time per speed bin %s m/s: + %s s, - %s s' % (SPEED_BINS, np.array2string(hist[0], precision=1),
                                                          np.array2string(hist[1], precision=1)))
    print('  criteria: %s -> %s' % (ok, 'PASS' if all(ok.values()) else 'FAIL'))
res['friction'] = fric
with open(os.path.join(C.OUT, 'j9_refcov.json'), 'w') as fh:
    json.dump(res, fh, indent=1)

# == figure 13: per stage entry, the averaged differences against 2.45 sigma (measurement) and 2.45 sigma_data
fig, axs = plt.subplots(3, 3, figsize=(7.0, 6.6), sharex=True, sharey=True)
for o in range(3):
    for i in range(3):
        ax = axs[o, i]
        ax.loglog(f, np.sqrt(Em2['nominal'][:, o, i]), color='#2a78d6', lw=0.9, label='nominal vs truth, rms over Y')
        ax.loglog(f, np.sqrt(Em2['detuned'][:, o, i]), color='#eb6834', lw=0.9, ls='--', label='detuned vs truth, rms over Y')
        ax.loglog(f, Z95 * np.sqrt(S2m[:, o, i]), color='#8a8983', lw=0.7, label='2.45 sigma, FRF measurement (T1)')
        ax.loglog(f, Z95 * np.sqrt(S2d[:, o, i]), color='black', lw=0.9, label='2.45 sigma_data, planned references (T3)')
        ax.axvspan(20, 50, color='#8a8983', alpha=0.08, lw=0)
        ax.set_xlim(1, fc); ax.set_ylim(1e-12, 1e-5)
        if o == 0:
            ax.set_title('input F_%s' % CH[i])
        if i == 0:
            ax.set_ylabel('|difference| [m/N]\noutput %s' % CH[o])
        if o == 2:
            ax.set_xlabel('frequency [Hz]')
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=2, frameon=False)
fig.suptitle('T3: FRF differences below the crossover (%.0f Hz) against the planned references\' resolution; '
             'shaded 20 to 50 Hz: linear prediction approximate (J5)' % fc, fontsize=7.5)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(os.path.join(C.HERE, 'figures', 'fig13_t3_references.png'), dpi=200)
print('\nwrote outputs/j9_refcov.json, figures/fig13_t3_references.png')
