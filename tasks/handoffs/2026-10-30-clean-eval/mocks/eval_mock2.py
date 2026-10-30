"""Full layout mock of evaluate(run_dir): every figure and the terminal output.
REAL: NRMS_e tables and bars (R1 final CSVs), physical parameters (R1), record I1 servo error, reference and
noise floor (data files), I1 Y predictions (res_time_trace.csv). SYNTHETIC: I1 X1/X2 predictions, loss curves."""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(REPO, 'Clean'))
from gantry_dynamic.config import RunConfig, PRESETS            # noqa: E402
from gantry_dynamic.data import _load_record, _record_at_fs_new  # noqa: E402
from scipy.io import loadmat                                     # noqa: E402

CSV = os.path.join(REPO, 'Thesis-writeup', 'Results', '24-10-2026', 'CSV')
AX = ('X1', 'X2', 'Y')
FS = 4000.0
K0 = 30
WIN = int(0.05 * FS)
C_AUG, C_BASE, C_TRUE, C_SYS = '#1f77b4', '#ff7f0e', '#9e9e9e', '0.25'
MODELS = [('model', 'augmented', C_AUG), ('physics', 'baseline (trained params)', C_BASE),
          ('base_true', 'baseline (true params)', C_TRUE)]
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3})
RUN = '89792'
ED = os.path.join(OUT, RUN, 'evaluation')
for s in ('train', 'val', 'test'):
    os.makedirs(os.path.join(ED, s), exist_ok=True)

term = []
p = lambda s='': term.append(s)
rows = list(csv.DictReader(open(os.path.join(CSV, 'res_errors.csv'))))
def get(split, model):
    return {r['record']: [float(r['nrms_e_' + a]) for a in AX] for r in rows
            if r['split'] == split and r['model'] == model and (r['run'] == 'R1' or model == 'floor')}

p('=' * 100)
p('[evaluate] run %s  (simulations/gantry_clean/thesis_.../%s)' % (RUN, RUN))
p('[evaluate] checkpoint gantry_ckpt_%s.pt (after L-BFGS polish, accepted)  device cpu  float64' % RUN)
p('[evaluate] models: augmented | baseline, trained params (learned part removed) | baseline, true params')
p('[evaluate] NRMS_e = RMS(e_hat - e) / sigma_e per record and axis, from k0 = %d, free-run closed loop' % K0)
p('=' * 100)

fmt = lambda v: ' '.join('%6.4f' % u for u in v)
for split in ('train', 'val', 'test'):
    data = {m: get(split, m) for m, _, _ in MODELS}
    floor = get(split, 'floor')
    recs = list(data['model'])
    p()
    p('%s  (%d records)   NRMS_e per record and axis' % (split.upper(), len(recs)))
    p('%-7s | %-20s | %-20s | %-20s | %-20s' % ('record', 'augmented', 'baseline (trained)', 'baseline (true)',
                                               'noise floor'))
    p('%-7s | %-20s | %-20s | %-20s | %-20s' % ('', *(['    X1     X2      Y'] * 4)))
    p('-' * 97)
    for r in recs + ['mean']:
        vals = [np.mean([d[q] for q in recs], 0) if r == 'mean' else d[r]
                for d in (data['model'], data['physics'], data['base_true'], floor)]
        if r == 'mean':
            p('-' * 97)
        p('%-7s | %s | %s | %s | %s' % (r, *(fmt(v) for v in vals)))
    mean_aug = np.mean([data['model'][q] for q in recs])
    p('mean over records and axes: augmented %.4f | baseline (trained) %.4f | baseline (true) %.4f' %
      (mean_aug, np.mean([data['physics'][q] for q in recs]), np.mean([data['base_true'][q] for q in recs])))
    if split == 'val':
        p('check: val %.6f = best validation score of training %.6f (rel. diff <1e-3 required): OK'
          % (mean_aug, mean_aug))

    # bar chart
    x = np.r_[np.arange(len(recs)), len(recs) + 0.6]
    fig, axs = plt.subplots(3, 1, figsize=(max(7, 0.5 * len(recs) + 2.5), 7), sharex=True)
    w = 0.27
    for i, a in enumerate(AX):
        for j, (m, lab, c) in enumerate(MODELS):
            v = [data[m][q][i] for q in recs]
            v.append(np.mean(v))
            axs[i].bar(x + (j - 1) * w, v, w, color=c, label=lab)
        axs[i].set_ylabel('NRMS$_e$ %s' % a)
        axs[i].axvline(len(recs) - 0.2, color='0.5', lw=0.8)
    axs[-1].set_xticks(x, recs + ['mean'], rotation=45, ha='right')
    fig.legend(*axs[0].get_legend_handles_labels(), ncol=3, loc='upper center', bbox_to_anchor=(0.5, 0.95))
    fig.suptitle('run %s, %s: NRMS$_e$ per record and axis' % (RUN, split))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(ED, split, 'nrms_per_axis.png'), dpi=110)
    plt.close(fig)

# physical parameters (R1, real)
p()
p('PHYSICAL PARAMETERS  (identifiable combinations; true = data-generating value)')
p('%-9s | %12s | %12s | %12s | %9s' % ('', 'true', 'start', 'learned', 'rel. err'))
p('-' * 66)
for r in csv.DictReader(open(os.path.join(CSV, 'res_obc_combinations.csv'))):
    if r['run'] == 'R1':
        t, s, l = float(r['true']), float(r['start_value']), float(r['learned'])
        p('%-9s | %12.5g | %12.5g | %12.5g | %+8.1f%%' % (r['combination'], t, s, l, 100 * (l - t) / t))

# record I1: real data
cfg = RunConfig(**PRESETS['R1'], noise='lowpass', data_root=os.path.join(REPO, 'Thesis-writeup', 'Data'))
f = 'Test/I1_r1.mat'
sd, e = _load_record(f, cfg)
_, e_nf = _load_record(f, cfg, folder=os.path.join(REPO, 'Thesis-writeup', 'Data', 'Coulomb-tanh-and-MSD-noise-free'))
d = loadmat(os.path.join(REPO, 'Thesis-writeup', 'Data', 'Coulomb-tanh-and-MSD-lowpass-noise', f), squeeze_me=True)
_, _, r = _record_at_fs_new(d, cfg, ('r_sim',))
n = len(e)
t = np.arange(n) / FS
tt = list(csv.DictReader(open(os.path.join(CSV, 'res_time_trace.csv'))))
ty = np.array([float(q['t_s']) for q in tt])
yy, ry = np.array([float(q['y_Y']) for q in tt]), np.array([float(q['r_Y']) for q in tt])
by, my = np.array([float(q['baseline_Y']) for q in tt]), np.array([float(q['final_Y']) for q in tt])
k = np.searchsorted(t, ty[0] - 1e-9)
m = min(len(ty), n - k)
print('I1 Y loader vs csv: max |e - (y - r)| = %.3e m' % np.max(np.abs(e[k:k + m, 2] - (yy[:m] - ry[:m]))))
sl = slice(k, k + m)
t, e, e_nf, r = t[sl], e[sl], e_nf[sl], r[sl]
rng = np.random.default_rng(1)
smooth = lambda s, L: np.convolve(s, np.ones(L) / L, 'same')
ehat = {'aug': np.empty_like(e), 'base': np.empty_like(e)}
tab = {'aug': get('test', 'model')['I1'], 'base': get('test', 'physics')['I1']}
sig = e.std(0)
for i in range(2):                       # X1, X2: synthetic residual at the table NRMS_e
    for key in ehat:
        res = smooth(rng.standard_normal(len(e)), 60) * (1 + 0.8 * np.abs(np.gradient(r[:, 2])) / np.abs(np.gradient(r[:, 2])).max())
        res *= tab[key][i] * sig[i] / np.sqrt(np.mean(res ** 2))
        ehat[key][:, i] = e[:, i] + res
ehat['aug'][:, 2] = (my - ry)[:m]
ehat['base'][:, 2] = (by - ry)[:m]
mov = lambda s: np.sqrt(np.convolve(s ** 2, np.ones(WIN) / WIN, mode='same'))

fig, axs = plt.subplots(4, 1, figsize=(10, 8.2), sharex=True, gridspec_kw=dict(height_ratios=[1, 1, 1, 0.7]))
for i, a in enumerate(AX):
    nr = {key: np.sqrt(np.mean((ehat[key][:, i] - e[:, i]) ** 2)) / sig[i] for key in ehat}
    nf = np.sqrt(np.mean((e[:, i] - e_nf[:, i]) ** 2)) / sig[i]
    axs[i].semilogy(t, 100 * mov(ehat['base'][:, i] - e[:, i]) / sig[i], color=C_BASE, lw=0.9, label='baseline (trained)')
    axs[i].semilogy(t, 100 * mov(ehat['aug'][:, i] - e[:, i]) / sig[i], color=C_AUG, lw=0.9, label='augmented')
    axs[i].semilogy(t, 100 * mov(e[:, i] - e_nf[:, i]) / sig[i], color=C_SYS, lw=0.7, ls=':', label='noise floor')
    axs[i].set_ylabel('moving NRMS$_{e,%s}$ (%%)' % a)
    axs[i].text(1.005, 0.5, 'record NRMS$_e$\nbaseline %.3f\naugmented %.3f\nfloor %.4f' % (nr['base'], nr['aug'], nf),
                transform=axs[i].transAxes, va='center', fontsize=7.5)
    if i < 2:
        axs[i].text(0.005, 0.04, 'SYNTHETIC prediction (mock)', transform=axs[i].transAxes, color='r', fontsize=8)
axs[3].plot(t, r[:, 2], color='k', lw=1, label='r$_Y$')
axs[3].plot(t, 0.5 * (r[:, 0] + r[:, 1]), color='0.55', lw=1, label='r$_X$ = (r$_{X1}$ + r$_{X2}$)/2')
axs[3].set_ylabel('reference (m)')
axs[3].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=7.5, frameon=False)
axs[3].set_xlabel('t (s)')
axs[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
fig.suptitle('run %s, test record I1: moving NRMS$_e$ (50 ms window), from k0' % RUN)
fig.align_ylabels(axs)
fig.tight_layout(rect=(0, 0, 0.88, 1))
fig.savefig(os.path.join(ED, 'test', 'I1.png'), dpi=110)
plt.close(fig)

# loss curves (synthetic)
upd = np.arange(0, 20001, 250)
tr = 0.02 * np.exp(-upd / 4000) + 0.004 + 0.0005 * rng.standard_normal(upd.size) ** 2
va = 0.6 * np.exp(-upd / 5000) + 0.0775 + 0.003 * rng.standard_normal(upd.size)
b = np.argmin(va)
fig, axs = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
axs[0].semilogy(upd, tr, color=C_AUG)
axs[0].set_ylabel('training loss\n(windowed MSE, n$_f$ steps)')
axs[1].semilogy(upd, va, color='#2ca02c', label='validation, Adam')
axs[1].scatter(upd[b], va[b], s=50, color='r', zorder=5, label='selected checkpoint')
axs[1].scatter(upd[-1] + 400, va[b] * 0.98, s=70, marker='*', color='purple', zorder=5, label='after L-BFGS polish')
axs[1].set_ylabel('validation NRMS$_e$\n(free run, mean of records, axes)')
axs[1].set_xlabel('Adam update')
axs[1].legend()
axs[0].set_title('run %s: loss curves   (SYNTHETIC mock values)' % RUN)
fig.tight_layout()
fig.savefig(os.path.join(ED, 'loss.png'), dpi=110)
plt.close(fig)

p()
p('saved %s/evaluation/' % RUN)
p('  scores.csv          %d rows (split, record, model, rms and NRMS_e per axis, sigma_e)' % 0)
p('  predictions.npz     per record: t, r, e, e_noise_free, e_hat augmented, e_hat baseline (trained/true), k0')
p('  loss.npz, loss.png  training loss and validation score per validation; selected update; polish rows')
p('  params.csv          true, start and learned physical combinations')
p('  evaluation.json     checkpoint, run id, git commit, validation check')
p('  train/ val/ test/   nrms_per_axis.png + one moving-NRMS_e figure per record (18 + 6 + 11)')
p('[evaluate] done in <..> s')
txt = '\n'.join(term)
n_rows = sum(len(get(s, 'model')) for s in ('train', 'val', 'test')) * 3
txt = txt.replace('%d rows' % 0, '%d rows' % n_rows)
open(os.path.join(OUT, 'terminal.txt'), 'w').write(txt + '\n')
print(txt)
