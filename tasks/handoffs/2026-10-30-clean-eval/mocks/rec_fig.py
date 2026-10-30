"""Per-record figure candidates for I1, stage and generalised frames.
REAL: e, r, noise floor (data files), Y predictions (res_time_trace.csv). SYNTHETIC: X1/X2 predictions."""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(REPO, 'Clean'))
from gantry_dynamic.config import RunConfig, PRESETS            # noqa: E402
from gantry_dynamic.data import _load_record, _record_at_fs_new  # noqa: E402
from model_augmentation.systems.gantry_ss import Lb             # noqa: E402
from scipy.io import loadmat                                     # noqa: E402

FS = 4000.0
C_AUG, C_BASE, C_SYS = '#1f77b4', '#ff7f0e', '0.15'
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3})
DATA = os.path.join(REPO, 'Thesis-writeup', 'Data')
CSV = os.path.join(REPO, 'Thesis-writeup', 'Results', '24-10-2026', 'CSV')

cfg = RunConfig(**PRESETS['R1'], noise='lowpass', data_root=DATA)
f = 'Test/I1_r1.mat'
_, e = _load_record(f, cfg)
_, e_nf = _load_record(f, cfg, folder=os.path.join(DATA, 'Coulomb-tanh-and-MSD-noise-free'))
_, _, r = _record_at_fs_new(loadmat(os.path.join(DATA, 'Coulomb-tanh-and-MSD-lowpass-noise', f), squeeze_me=True),
                            cfg, ('r_sim',))
tt = list(csv.DictReader(open(os.path.join(CSV, 'res_time_trace.csv'))))
col = lambda k: np.array([float(q[k]) for q in tt])
ty, yy, ry, by, my = col('t_s'), col('y_Y'), col('r_Y'), col('baseline_Y'), col('final_Y')
t = np.arange(len(e)) / FS
k = np.searchsorted(t, ty[0] - 1e-9)
m = min(len(ty), len(e) - k)
sl = slice(k, k + m)
t, e, e_nf, r = t[sl] - t[k], e[sl], e_nf[sl], r[sl]

# synthetic X1/X2 predictions: residual tied to the reference acceleration (errors during moves) plus coloured noise
rng = np.random.default_rng(3)
acc = np.gradient(np.gradient(r, axis=0), axis=0) * FS ** 2
sm = lambda s, L: np.convolve(s, np.ones(L) / L, 'same')
tab = {'aug': (0.0711, 0.0737), 'base': (0.1375, 0.1535)}
sig = e.std(0)
ehat = {key: e.copy() for key in tab}
for key, (n1, n2) in tab.items():
    for i, nr in ((0, n1), (1, n2)):
        res = sm(acc[:, i], 80) * (1.5 if key == 'base' else 0.6) + sm(rng.standard_normal(len(e)), 40) * np.std(sm(acc[:, i], 80))
        res *= nr * sig[i] / np.sqrt(np.mean(res ** 2))
        ehat[key][:, i] += res
ehat['aug'][:, 2] = (my - ry)[:m]
ehat['base'][:, 2] = (by - ry)[:m]

T = np.array([[0.5, 0.5, 0], [1 / Lb, -1 / Lb, 0], [0, 0, 1]])    # q = P^-T y: X, Theta, Y
FRAMES = {'stage': (('X1', 'X2', 'Y'), np.eye(3), ('m', 'm', 'm')),
          'generalised': (('X', r'$\Theta$', 'Y'), T, ('m', 'rad', 'm'))}


def nrms(a, b, s):
    return np.sqrt(np.mean((a - b) ** 2)) / s


for frame, (names, M, units) in FRAMES.items():
    E, Enf, R = e @ M.T, e_nf @ M.T, r @ M.T
    H = {key: v @ M.T for key, v in ehat.items()}
    S = E.std(0)
    # design A: normalised residual (e_hat - e) / sigma_e, linear, floor as band
    fig, axs = plt.subplots(4, 1, figsize=(10, 7.8), sharex=True, gridspec_kw=dict(height_ratios=[1, 1, 1, 0.6]))
    for i, nm in enumerate(names):
        fl = np.std(E[:, i] - Enf[:, i]) / S[i]
        axs[i].plot(t, (H['base'][:, i] - E[:, i]) / S[i], color=C_BASE, lw=0.6,
                    label='baseline (trained)')
        axs[i].plot(t, (H['aug'][:, i] - E[:, i]) / S[i], color=C_AUG, lw=0.6, label='augmented')
        axs[i].set_ylabel(r'$(\hat e - e)/\sigma_{e}$  %s' % nm)
        axs[i].text(1.008, 0.5, 'NRMS$_e$\nbaseline %.3f\naugmented %.3f\nfloor %.4f'
                    % (nrms(H['base'][:, i], E[:, i], S[i]), nrms(H['aug'][:, i], E[:, i], S[i]), fl),
                    transform=axs[i].transAxes, va='center', fontsize=7.5)
    for i, nm in enumerate(names):
        if names[i] != 'Y' or frame == 'stage':
            pass
    axs[3].plot(t, R[:, 2], color='k', lw=1, label='Y')
    axs[3].plot(t, R[:, 0], color='0.55', lw=1, label=names[0])
    axs[3].set_ylabel('reference (m)')
    axs[3].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=7.5)
    axs[3].set_xlabel('t (s)')
    axs[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.suptitle('I1 (test, aprbs multisine), %s frame: prediction error over the record' % frame)
    fig.text(0.01, 0.005, 'mock: X1/X2 predictions synthetic, Y real (R1)', color='r', fontsize=7)
    fig.align_ylabels(axs)
    fig.tight_layout(rect=(0, 0.01, 0.88, 1))
    fig.savefig(os.path.join(OUT, 'A_I1_%s.png' % frame), dpi=110)
    plt.close(fig)

    # design B: thesis style servo error / sigma_e: system thick light, models thin on top
    fig, axs = plt.subplots(4, 1, figsize=(10, 7.8), sharex=True, gridspec_kw=dict(height_ratios=[1, 1, 1, 0.6]))
    for i, nm in enumerate(names):
        axs[i].plot(t, E[:, i] / S[i], color='0.75', lw=2.2, label='system')
        axs[i].plot(t, H['base'][:, i] / S[i], color=C_BASE, lw=0.6, label='baseline (trained)')
        axs[i].plot(t, H['aug'][:, i] / S[i], color=C_AUG, lw=0.6, label='augmented')
        axs[i].set_ylabel(r'$e/\sigma_{e}$  %s' % nm)
        axs[i].text(1.008, 0.5, 'NRMS$_e$\nbaseline %.3f\naugmented %.3f'
                    % (nrms(H['base'][:, i], E[:, i], S[i]), nrms(H['aug'][:, i], E[:, i], S[i])),
                    transform=axs[i].transAxes, va='center', fontsize=7.5)
    axs[3].plot(t, R[:, 2], color='k', lw=1, label='Y')
    axs[3].plot(t, R[:, 0], color='0.55', lw=1, label=names[0])
    axs[3].set_ylabel('reference (m)')
    axs[3].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=7.5)
    axs[3].set_xlabel('t (s)')
    axs[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.suptitle('I1 (test, aprbs multisine), %s frame: servo error' % frame)
    fig.text(0.01, 0.005, 'mock: X1/X2 predictions synthetic, Y real (R1)', color='r', fontsize=7)
    fig.align_ylabels(axs)
    fig.tight_layout(rect=(0, 0.01, 0.88, 1))
    fig.savefig(os.path.join(OUT, 'B_I1_%s.png' % frame), dpi=110)
    plt.close(fig)
print(sorted(os.listdir(OUT)))
