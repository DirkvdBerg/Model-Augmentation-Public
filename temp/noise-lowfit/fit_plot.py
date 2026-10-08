"""Low-order encoder-noise model (Jasper 2026-10-05): 2nd-order band-pass + white floor per axis,
fitted so the simulated standstill error S_sim v follows the ENVELOPE of the measured error; gain set by rms.
Compared with plain white noise (rms-matched). Linear loop at Y=0 (truth at rest + tanh slope, K1)."""
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import least_squares
from scipy.signal import welch, bilinear, lfilter, dlsim

REPO = r"C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation"
m = loadmat(REPO + "/Thesis-writeup/Code/Data/noise/noise_target.mat", squeeze_me=True)
fw, Pw = m['f'], np.real(m['P']); dfw = fw[1] - fw[0]
pe = np.stack([Pw[:, j, j] for j in range(3)], 1)            # measured error PSD, m^2/Hz
rms_meas = np.sqrt(pe[1:].sum(0) * dfw)
s = loadmat('S_Y0.mat', squeeze_me=True)
fs_, S2 = s['f'], s['S2']; ts = float(s['ts']); fs = 1 / ts
def S2at(f):  # |S_jk|^2 interpolated on log f
    return np.stack([[np.interp(np.log(f), np.log(fs_), S2[:, r, c]) for c in range(3)] for r in range(3)])  # 3x3xn

def bp2(f, f0, Q):  # unit-peak 2nd-order band-pass, |.|^2
    x = f / f0
    return (x / Q) ** 2 / ((1 - x ** 2) ** 2 + (x / Q) ** 2)

def phi_v(f, p):  # p = [log a, log f0, log Q, log b]; v PSD m^2/Hz
    a, f0, Q, b = np.exp(p)
    return a ** 2 * bp2(f, f0, Q) + b ** 2

ff = fw[1:]; Sf = S2at(ff)
fit_band = ff >= 50
params, models = [], {}
for j in range(3):
    def res(p):
        pred = Sf[j, j] * phi_v(ff, p)
        return (np.log(pred) - np.log(pe[1:, j]))[fit_band]
    p0 = np.log([np.sqrt(pe[1:, j].max()), 300, 1.0, np.sqrt(pe[-50:, j].mean())])
    p = least_squares(res, p0, bounds=(np.log([1e-13, 20, 0.2, 1e-13]), np.log([1e-7, 5000, 10, 1e-7]))).x
    # rms calibration of the simulated error (cross terms: same model per axis, small)
    pred = Sf[j, j] * phi_v(ff, p)
    k = rms_meas[j] / np.sqrt(pred.sum() * dfw)
    p[0] += np.log(k); p[3] += np.log(k)
    bw = rms_meas[j] / np.sqrt((Sf[j, j]).sum() * dfw)   # white: one-sided PSD level bw^2
    params.append((p, bw))
    a, f0, Q, b = np.exp(p)
    print(f"axis {j}: band-pass f0 {f0:.0f} Hz Q {Q:.2f} peak {a*1e9*1e3:.0f} pm/rtHz, floor {b*1e12:.0f} pm/rtHz, rms gain fix {k:.3f}; white {bw*1e12:.0f} pm/rtHz")

# time-domain realisation: v per axis, e = -S v (discrete loop), 12 s at 20 kHz
N = 240000; rng = np.random.default_rng(20261005)
V = {'lo': np.zeros((N, 3)), 'white': np.zeros((N, 3))}
for j in range(3):
    p, bw = params[j]; a, f0, Q, b = np.exp(p); w0 = 2 * np.pi * f0
    # continuous BP with unit peak: (w0/Q) s / (s^2 + (w0/Q) s + w0^2); bilinear with prewarp at f0
    bz, az = bilinear([w0 / Q, 0], [1, w0 / Q, w0 ** 2], fs=fs)
    sd = np.sqrt(fs / 2)                                   # unit one-sided PSD white noise
    w1, w2, w3 = rng.standard_normal((3, N)) * sd
    V['lo'][:, j] = a * lfilter(bz, az, w1) + b * w2
    V['white'][:, j] = bw * w3
E = {}
for k, v in V.items():
    _, y, _ = dlsim((s['A'], s['B'], s['C'], s['D'], ts), v)
    E[k] = -y[N // 6:]                                     # drop 2 s transient
kw = dict(fs=fs, window='hann', nperseg=2048, noverlap=1024, axis=0)
f, PL = welch(E['lo'], **kw); _, PW = welch(E['white'], **kw); _, PV = welch(V['lo'], **kw)

fig, axes = plt.subplots(3, 1, figsize=(8, 9.5), sharex=True)
for j, (ax, nm) in enumerate(zip(axes, ['X1', 'X2', 'Y'])):
    ax.loglog(fw[1:], np.sqrt(pe[1:, j]), color='k', lw=1.4, label=f'measured error (Telica standstill), rms {rms_meas[j]*1e9:.1f} nm')
    ax.loglog(f[1:], np.sqrt(PL[1:, j]), color='#2a78d6', lw=1.1, label=f'simulated error, low-order noise, rms {np.std(E["lo"][:, j])*1e9:.1f} nm')
    ax.loglog(f[1:], np.sqrt(PW[1:, j]), color='#e08a1e', lw=1.0, label=f'simulated error, white noise, rms {np.std(E["white"][:, j])*1e9:.1f} nm')
    ax.loglog(f[1:], np.sqrt(PV[1:, j]), color='#2a78d6', lw=0.9, ls='--', alpha=0.7, label='injected encoder noise v (low-order model)')
    ax.axhline(params[j][1], color='#e08a1e', lw=0.9, ls='--', alpha=0.7, label='injected encoder noise v (white)')
    ax.set_ylabel(f'{nm}  [m/$\sqrt{{Hz}}$]'); ax.grid(True, which='major', color='#d9d9d9', lw=0.5)
    ax.set_xlim(9, 1e4)
    if j == 0: ax.legend(fontsize=7.5, loc='lower left')
axes[-1].set_xlabel('frequency [Hz]')
fig.suptitle('Standstill servo error at Y = 0: measured vs simulated with a low-order or white encoder noise', fontsize=10)
fig.tight_layout(); fig.savefig('noise_lowfit.png', dpi=150)
for j in range(3):
    for k in ('lo', 'white'):
        e = E[k][:, j]; fe, Pe = welch(e, **kw)
        print(j, k, 'frac>200 Hz %.2f' % (Pe[fe > 200].sum() / Pe[1:].sum()))
print('in-band rms (<1.6 kHz, training band after anti-alias), nm')
for j in range(3):
    mb = (fw > 0) & (fw < 1600); r = [np.sqrt(pe[mb, j].sum() * dfw) * 1e9]
    for P_ in (PL, PW):
        fb = (f > 0) & (f < 1600); r.append(np.sqrt(P_[fb, j].sum() * (f[1] - f[0])) * 1e9)
    print(j, 'measured %.1f  low-order %.1f  white %.1f' % tuple(r))
