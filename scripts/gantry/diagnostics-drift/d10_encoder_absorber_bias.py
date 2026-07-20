"""
d10_encoder_absorber_bias.py -- step 1a after d9: WHY does the untrained linear
encoder have a systematic dY init error, and does a longer window (na) fix it?

Hypothesis under test (from d9): the encoder init map W^b = A^n pinv(O_n) is built
from the BASELINE linearization, which has NO absorber mode. The data comes from
the 8-state truth (baseline + 150 Hz absorber). The window is na+1 = 18 samples
@ 4 kHz = 4.5 ms < one absorber period (6.67 ms), so the absorber's contribution
to the measured Y window is mis-read by the map as baseline state content ->
per-window dY error set by the absorber phase at the window start (large variance,
and a nonzero mean when absorber phase correlates with the excitation).

Four closed-form/forward-only parts (no training; encoder calls are milliseconds):
  P1 Frequency response of the dY row of the ACTUAL pipeline encoder to a unit
     sinusoid on the Y output channel, vs the ideal differentiator response
     |H|=omega, for na in {17, 27, 40, 53} (27 samples ~ one absorber period).
     Offsets cancelled by calling enc(window) - enc(zero window).
  P2 Data check: Welch PSD of measured y_Y and u on V1 -- is the 150 Hz line
     present in y and absent in u (so it is absorber-caused, not excitation)?
  P3 Per-window decomposition on V1: bandpass y in [120,180] Hz -> the map's
     response to that component (enc(y) - enc(y - y_bp), linearity) minus the
     TRUE bandpassed dY = predicted absorber-induced init error. Scatter against
     the measured total dY init error (119 window starts): slope ~ 1 and high R^2
     confirm the mechanism INCLUDING the mean.
  P4 Fix design: rebuild the linear-only encoder at na in {17, 27, 40, 53} (same
     Ad_bar..Dd_bar as build_model) and re-measure the dY init error over the
     same window starts -> predicted bias/variance collapse vs na.

Run:
  conda run -n GraduationProject python scripts/gantry/diagnostics-drift/d10_encoder_absorber_bias.py
Env: NA_LIST (default "17,27,40,53"), NF (window stride for starts, default 400).
Outputs -> simulations/gantry_subnet/diagnostics/ (npz + png)
"""
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal as sps

REPO   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
GANTRY = os.path.join(REPO, 'scripts', 'gantry')
sys.path.insert(0, GANTRY)
sys.path.insert(0, os.path.dirname(__file__))

import drift_common as dc
from gantry_dynamic.data import load_datasets, compute_normalization
from gantry_dynamic.model import build_model, get_encoder_dims
from gantry_dynamic.config import REPO_ROOT
from model_augmentation.fit_systems.pre_encoder import linear_encoder_init_aug
from model_augmentation.systems.gantry_linearization import gantry_linearize_and_discretize
from model_augmentation.utils.utils import normalize_linear_ss_matrices
from gantry_interconnect_dynamic import CFG as cfg

NA_LIST = [int(x) for x in os.environ.get('NA_LIST', '17,27,40,53').split(',')]
NF      = int(os.environ.get('NF', '400'))
FA      = dc.fa                      # absorber frequency [Hz] (drift_common: 150.0)
BAND    = (120.0, 180.0)             # bandpass around the absorber line

# -- Pipeline (UNTRAINED: the init map is the object under analysis) -----------
np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
data = load_datasets(cfg)
norm = compute_normalization(cfg, data)
np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
fit_sys = build_model(cfg.hp, cfg, data, norm)
fit_sys.hfn.eval()
enc0 = fit_sys.encoder
assert float(enc0.net[-1].weight.abs().max()) == 0.0, "encoder net not zero-init; not the pure map"

na, nb, na_right, nb_right = get_encoder_dims(cfg.hp, cfg)
ts   = cfg.ts_new
fs   = 1.0 / ts
std_x = norm.std_x.flatten()
ystd  = np.asarray(norm.ystd).flatten()
y0v   = np.asarray(norm.y0).flatten()
ustd  = np.asarray(norm.ustd if hasattr(norm, 'ustd') else fit_sys.norm.ustd).flatten()
u0v   = np.asarray(norm.u0 if hasattr(norm, 'u0') else fit_sys.norm.u0).flatten()

v1 = data.val_data
u_raw = np.asarray(v1.u, dtype=np.float64)
y_raw = np.asarray(v1.y, dtype=np.float64)
x_true = data.val_x_logical.astype(np.float64)
Ntot = len(u_raw)
print(f'V1: {Ntot} samples @ {fs:.0f} Hz  pipeline na={na} (window {(na+1)*ts*1e3:.2f} ms; '
      f'absorber period {1e3/FA:.2f} ms)  NA_LIST={NA_LIST}')

# normalized (pipeline-convention) copies for encoder input
y_n = (y_raw - y0v) / ystd
u_n = (u_raw - u0v) / ustd


def _enc_call(enc, u_win_n, y_win_n):
    """One encoder forward on normalized windows (n+1, ch); returns physical x (6,)."""
    u_c = np.ascontiguousarray(u_win_n, dtype=cfg.dtype_np)
    y_c = np.ascontiguousarray(y_win_n, dtype=cfg.dtype_np)
    with torch.no_grad():
        x = enc(torch.tensor(u_c[None], dtype=cfg.dtype_pt),
                torch.tensor(y_c[None], dtype=cfg.dtype_pt)).numpy()[0]
    return x[:6] * std_x + norm.x_mean.flatten()


def _build_linear_encoder(na_i):
    """Rebuild the linear-only init map at window na_i, exactly as build_model does."""
    Ad, Bd, Cd_dt, Dd_dt = gantry_linearize_and_discretize(dt=ts)
    # mirror build_model's normalization path (baseline_states.npz absent -> same
    # finite-diff fallback happens inside compute_normalization; reuse its output)
    import deepSI
    sd = deepSI.System_data(u=np.asarray(data.train_data.u), y=np.asarray(data.train_data.y),
                            dt=ts)
    sd_n = fit_sys.norm.transform(sd)
    sd_n.x = data.train_x_phys_norm if hasattr(data, 'train_x_phys_norm') else None
    if sd_n.x is None:
        # normalize_linear_ss_matrices needs state std; reconstruct the same
        # normalized matrices from the pipeline encoder instead of re-deriving:
        # scale rows/cols of (Ad,Bd,Cd,Dd) with the pipeline's std_x/ustd/ystd.
        Sx  = np.diag(std_x); Sxi = np.diag(1.0 / std_x)
        Su  = np.diag(ustd);  Syi = np.diag(1.0 / ystd)
        Ad_bar = Sxi @ Ad @ Sx
        Bd_bar = Sxi @ Bd @ Su
        Cd_bar = Syi @ Cd_dt @ Sx
        Dd_bar = Syi @ Dd_dt @ Su
    e = linear_encoder_init_aug(
        A=Ad_bar, B=Bd_bar, C=Cd_bar, D=Dd_bar,
        nx=6, nu=3, ny=3, na=na_i, nb=na_i, nx_aug=cfg.hp['NX_ANN'],
        flag_linear_only=True,
        u_mean=norm.u_mean, std_u=norm.std_u, y0=norm.y0, ystd=norm.ystd,
        x_mean=norm.x_mean, std_x=norm.std_x,
    ).to(cfg.dtype_pt)
    e.eval()
    return e


# sanity: rebuilt na=17 map must match the pipeline encoder on a real window
enc_chk = _build_linear_encoder(na)
s_chk = 4000
x_pipe = _enc_call(enc0, u_n[s_chk - nb: s_chk + nb_right], y_n[s_chk - na: s_chk + na_right])
x_reb  = _enc_call(enc_chk, u_n[s_chk - nb: s_chk + nb_right], y_n[s_chk - na: s_chk + na_right])
rel = np.abs(x_pipe - x_reb) / (np.abs(x_pipe) + 1e-12)
print(f'P0 sanity: rebuilt na={na} vs pipeline encoder, max rel diff = {rel.max():.2e} '
      f'({"OK" if rel.max() < 1e-4 else "MISMATCH -- do not trust P1/P4"})')

encoders = {na: enc0}
for na_i in NA_LIST:
    if na_i != na:
        encoders[na_i] = _build_linear_encoder(na_i)

# -- P1: frequency response of the dY row to a unit Y-channel sinusoid ---------
print('\nP1: dY-row frequency response (unit sinusoid on Y channel, u=0) ...')
freqs = np.linspace(10, 1200, 120)
H = {na_i: np.empty(len(freqs), dtype=complex) for na_i in encoders}
for na_i, e in encoders.items():
    L = na_i + 1
    tk = (np.arange(L) - (L - 1)) * ts          # window ends at t=0 (current sample)
    uz = np.zeros((L, 3))
    y_base = np.zeros((L, 3))
    x_off = _enc_call(e, uz, y_base)            # offset (mean-convention) reference
    for i, f in enumerate(freqs):
        w = 2 * np.pi * f
        re_win = np.zeros((L, 3)); re_win[:, 2] = np.cos(w * tk) / ystd[2]
        im_win = np.zeros((L, 3)); im_win[:, 2] = np.sin(w * tk) / ystd[2]
        xr = _enc_call(e, uz, re_win) - x_off
        xi = _enc_call(e, uz, im_win) - x_off
        H[na_i][i] = xr[5] + 1j * xi[5]         # dY response to e^{jwt} [ (m/s) / m ]
ideal = 2 * np.pi * freqs
i150 = np.argmin(np.abs(freqs - FA))
print(f"  {'na':>4s} {'win[ms]':>8s} {'|H(150)| [1/s]':>15s} {'ideal w=942':>12s} {'excess x':>9s}")
for na_i in sorted(encoders):
    h = abs(H[na_i][i150])
    print(f'  {na_i:>4d} {(na_i+1)*ts*1e3:>8.2f} {h:>15.1f} {ideal[i150]:>12.1f} {h/ideal[i150]:>9.2f}')

# -- P2: is the 150 Hz line absorber-caused (in y, not in u)? -------------------
print('\nP2: Welch PSD around the absorber line ...')
fw, Pyy = sps.welch(y_raw[:, 2], fs=fs, nperseg=8192)
band = (fw >= BAND[0]) & (fw <= BAND[1])
ref  = (fw >= 30) & (fw <= 90)
py_band, py_ref = Pyy[band].mean(), Pyy[ref].mean()
print(f'  y_Y : mean PSD {BAND[0]:.0f}-{BAND[1]:.0f} Hz = {py_band:.3e}, 30-90 Hz = {py_ref:.3e} '
      f'(ratio {py_band/py_ref:.2f})')
pu_bands = []
for ch in range(3):
    fu, Puu = sps.welch(u_raw[:, ch], fs=fs, nperseg=8192)
    pu_bands.append(Puu[band].mean() / (Puu[ref].mean() + 1e-30))
print(f'  u ch ratios (band/ref): {pu_bands[0]:.3f} {pu_bands[1]:.3f} {pu_bands[2]:.3f} '
      f'(<<1 means no excitation content at the absorber line)')

# -- P3: per-window decomposition -- does the absorber component explain the error?
print('\nP3: per-window absorber-component prediction vs measured dY init error ...')
sos = sps.butter(4, BAND, btype='bandpass', fs=fs, output='sos')
y_bp   = sps.sosfiltfilt(sos, y_raw, axis=0)
dY_bp  = sps.sosfiltfilt(sos, x_true[:, 5])
y_nb   = (y_raw - y_bp - y0v) / ystd            # normalized y with absorber band removed

warm = max(na, nb)
starts = list(range(warm, Ntot - NF, NF))
meas = np.empty(len(starts)); pred = np.empty(len(starts))
for i, s in enumerate(starts):
    uw = u_n[s - nb: s + nb_right]
    x_full = _enc_call(enc0, uw, y_n[s - na: s + na_right])
    x_nobp = _enc_call(enc0, uw, y_nb[s - na: s + na_right])
    meas[i] = x_full[5] - x_true[s, 5]                        # measured total error
    pred[i] = (x_full[5] - x_nobp[5]) - dY_bp[s]              # absorber-induced part
A1 = np.vstack([pred, np.ones_like(pred)]).T
(slope, icpt), *_ = np.linalg.lstsq(A1, meas, rcond=None)
r2 = 1 - ((meas - A1 @ [slope, icpt]) ** 2).sum() / ((meas - meas.mean()) ** 2).sum()
print(f'  windows={len(starts)}  slope={slope:.3f}  intercept={icpt:+.3e}  R^2={r2:.3f}')
print(f'  mean measured = {meas.mean():+.3e} m/s   mean predicted (absorber) = {pred.mean():+.3e} m/s')
print(f'  std  measured = {meas.std():.3e}        std  predicted            = {pred.std():.3e}')

# -- P4: na-sweep -- does a longer window collapse the dY init error? -----------
print('\nP4: dY init error vs na (same window starts, full measured y) ...')
p4 = {}
print(f"  {'na':>4s} {'win[ms]':>8s} {'nwin':>5s} {'mean dY err':>13s} {'SE':>10s} {'perwin std':>11s}")
for na_i in sorted(encoders):
    e = encoders[na_i]
    st = [s for s in starts if s >= na_i]
    errs = np.empty(len(st))
    for i, s in enumerate(st):
        errs[i] = _enc_call(e, u_n[s - na_i: s + nb_right], y_n[s - na_i: s + na_right])[5] \
                  - x_true[s, 5]
    p4[na_i] = errs
    se = errs.std(ddof=1) / np.sqrt(len(st))
    print(f'  {na_i:>4d} {(na_i+1)*ts*1e3:>8.2f} {len(st):>5d} {errs.mean():>+13.3e} '
          f'{se:>10.2e} {errs.std():>11.3e}')

# -- Plot -----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
ax = axes[0, 0]
for na_i in sorted(encoders):
    ax.loglog(freqs, np.abs(H[na_i]), label=f'na={na_i} ({(na_i+1)*ts*1e3:.1f} ms)')
ax.loglog(freqs, ideal, 'k--', label='ideal |H|=omega')
ax.axvline(FA, color='C3', ls=':', label=f'absorber {FA:.0f} Hz')
ax.set_xlabel('f [Hz]'); ax.set_ylabel('|dY response| [(m/s)/m]')
ax.set_title('P1: dY-row response to a Y-channel sinusoid'); ax.grid(True, which='both')
ax.legend(fontsize=7)
ax = axes[0, 1]
ax.semilogy(fw, Pyy, lw=0.8)
ax.axvspan(*BAND, color='C3', alpha=0.15, label=f'band {BAND[0]:.0f}-{BAND[1]:.0f} Hz')
ax.set_xlim(0, 500); ax.set_xlabel('f [Hz]'); ax.set_ylabel('PSD y_Y [m^2/Hz]')
ax.set_title('P2: measured Y-position spectrum (V1)'); ax.grid(True); ax.legend(fontsize=8)
ax = axes[1, 0]
ax.plot(pred, meas, '.', ms=4, alpha=0.7)
lim = np.array([min(pred.min(), meas.min()), max(pred.max(), meas.max())])
ax.plot(lim, lim, 'k--', lw=0.8, label='y = x')
ax.plot(lim, slope * lim + icpt, 'C3-', lw=1.0, label=f'fit: slope {slope:.2f}, R2 {r2:.2f}')
ax.set_xlabel('predicted absorber-induced dY error [m/s]')
ax.set_ylabel('measured dY init error [m/s]')
ax.set_title('P3: does the absorber component explain the error?'); ax.grid(True)
ax.legend(fontsize=8)
ax = axes[1, 1]
nas = sorted(p4)
ax.errorbar(nas, [p4[n].mean() for n in nas],
            yerr=[2 * p4[n].std(ddof=1) / np.sqrt(len(p4[n])) for n in nas],
            fmt='o-', capsize=3, label='mean dY init error (2 SE)')
ax2 = ax.twinx()
ax2.plot(nas, [p4[n].std() for n in nas], 'C3s--', label='per-window std')
ax.axhline(0, color='k', lw=0.6)
ax.axvline(1e3 / FA / (ts * 1e3) - 1, color='0.5', ls=':', label='window = 1 absorber period')
ax.set_xlabel('na (window = na+1 samples)'); ax.set_ylabel('mean dY init error [m/s]')
ax2.set_ylabel('per-window std [m/s]', color='C3')
ax.set_title('P4: fix design -- dY init error vs na'); ax.grid(True)
ax.legend(fontsize=8, loc='upper right')
fig.suptitle('d10: is the encoder dY init error the baseline-map mis-reading the absorber band? (untrained map)')
fig.tight_layout()
stem = os.path.join(dc.OUT_DIR, 'd10_encoder_absorber_bias')
fig.savefig(stem + '.png', dpi=150)
np.savez(stem + '.npz', freqs=freqs, ideal=ideal, fa=FA, band=np.array(BAND),
         fw=fw, Pyy=Pyy, u_band_ratios=np.array(pu_bands),
         pred=pred, meas=meas, slope=slope, intercept=icpt, r2=r2,
         starts=np.array(starts), na_pipeline=na, ts=ts,
         **{f'H_{k}': v for k, v in H.items()},
         **{f'p4err_{k}': v for k, v in p4.items()})
print(f'\nSaved: {stem}.png')
print(f'Saved: {stem}.npz')
