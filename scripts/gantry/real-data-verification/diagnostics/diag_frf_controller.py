"""
diag_frf_controller.py
----------------------
Extract the controller that was ACTUALLY active during the FRF campaign in
Telica.mat and compare it against Filter1*Filter2. This is excitation-based
(multisine/swept FRF measurement) and therefore free of any closed-loop
correlation concern.

Identity used:
    S = (I + G K)^-1  (output sensitivity)
    =>  K_eff(f) = G(f)^-1 (S(f)^-1 - I)
    THEORY: standard closed-loop relations, Skogestad & Postlethwaite (2005),
    Multivariable Feedback Control, Ch. 2. G is frfPlant [cnt/dac], so K_eff
    is in [dac/cnt], directly comparable to Filter1*Filter2 (no unit
    conversions involved).

If |K_eff / (F1*F2)| is flat over frequency, the machine applies an extra
constant per-channel gain (e.g. a HIGS in gain mode) and its value can be
read off. If the ratio is 1, the documented filters are the whole controller
and the mismatch must come from log interpretation instead.

ASSUMPTION: FRF channel order is (X, Y, Z), matching the Axes struct order.

Run:
    conda run -n GraduationProject python scripts/gantry/real-data-verification/diag_frf_controller.py
"""

__project_origin__ = "added"

import os
import json
import numpy as np
from scipy.io import loadmat
from scipy.signal import freqz

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..', '..'))
_TELICA   = os.path.join(_ROOT, 'kamtin-data', 'Telica.mat')
_SAVE_DIR = os.path.join(_ROOT, 'simulations', 'gantry_subnet',
                         'diagnostics', 'controller_frf')

FS_CTRL   = 20_000.0        # Hz, DSP rate (Telica.mat SamplingTime = 5e-5 s)
BAND      = (10.0, 2000.0)  # HEURISTIC: report median ratio where servo FRFs are clean
_CH       = {'X': 0, 'Y': 1}   # FRF channel index per axis (see ASSUMPTION above)


def load_telica():
    mat = loadmat(_TELICA, squeeze_me=True, struct_as_record=False)
    mp = mat['MachineParam']
    frf = mp.Modules.XYZ.Plant.Local.FRFMeasurement

    def cells(field):
        arr = getattr(frf, field)
        return [np.asarray(arr[i]) for i in range(len(arr))]

    out = {
        'G':  cells('frfPlant'),               # 9 x (3,3,700) [cnt/dac]
        'S':  cells('frfSensitivity'),         # 9 x (3,3,700)
        'PS': cells('frfProcessSensitivity'),  # 9 x (3,3,700)
        'f':  np.asarray(frf.freqsPlant, float).ravel(),
        'names': [str(n) for n in frf.datalog_names],
    }

    filts = {}
    for axname in ('X', 'Y'):
        ctr = getattr(mp.Axes, axname).Controllers
        pair = []
        for fname in ('Filter1', 'Filter2'):
            v = getattr(ctr, fname).Values
            b = np.trim_zeros(np.array([getattr(v, f'b{i}') for i in range(7)], float), 'b')
            a = np.trim_zeros(np.concatenate(([1.0],
                 [getattr(v, f'a{i}') for i in range(1, 7)])).astype(float), 'b')
            pair.append((b, a))
        filts[axname] = pair
    out['filters'] = filts
    return out


def controller_response(filters, axname, f_hz):
    """Filter1*Filter2 response in [dac/cnt] at physical frequencies (20 kHz design)."""
    w = 2 * np.pi * f_hz / FS_CTRL
    (b1, a1), (b2, a2) = filters[axname]
    _, h1 = freqz(b1, a1, worN=w)
    _, h2 = freqz(b2, a2, worN=w)
    return h1 * h2


def main():
    os.makedirs(_SAVE_DIR, exist_ok=True)
    d = load_telica()
    f = d['f']
    nf = len(f)
    npos = len(d['G'])

    print('=' * 70)
    print('DIAG FRF CONTROLLER: K_eff = G^-1 (S^-1 - I)  vs  Filter1*Filter2')
    print('=' * 70)
    print(f'{npos} positions, {nf} frequencies [{f.min():.0f}, {f.max():.0f}] Hz')
    print('positions:', ', '.join(d['names']))
    print('dtypes: G', d['G'][0].dtype, ' S', d['S'][0].dtype)

    # --- self-consistency: is PS = S*G or G*S? (matrix product per frequency) ---
    Gc = np.moveaxis(d['G'][4], -1, 0)   # center position, (700,3,3)
    Sc = np.moveaxis(d['S'][4], -1, 0)
    PSc = np.moveaxis(d['PS'][4], -1, 0)
    e_sg = np.median(np.abs(Sc @ Gc - PSc)) / np.median(np.abs(PSc))
    e_gs = np.median(np.abs(Gc @ Sc - PSc)) / np.median(np.abs(PSc))
    print(f'\nProcess-sensitivity ordering check (center pos): '
          f'|S*G - PS| rel = {e_sg:.3f},  |G*S - PS| rel = {e_gs:.3f}')

    # --- K_eff per position ---
    I3 = np.eye(3)
    K_eff = []   # npos x (700,3,3)
    for p in range(npos):
        Gp = np.moveaxis(d['G'][p], -1, 0)
        Sp = np.moveaxis(d['S'][p], -1, 0)
        Kp = np.linalg.inv(Gp) @ (np.linalg.inv(Sp) - I3)
        K_eff.append(Kp)

    band = (f >= BAND[0]) & (f <= BAND[1])
    summary = {}
    for axname, ch in _CH.items():
        H_ctrl = controller_response(d['filters'], axname, f)
        fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

        ratios_band = []
        for p in range(npos):
            Kii = K_eff[p][:, ch, ch]
            ratio = Kii / H_ctrl
            ratios_band.append(np.median(np.abs(ratio[band])))
            axs[0].loglog(f, np.abs(Kii), color='0.6', lw=0.7,
                          label='K_eff (9 positions)' if p == 0 else None)
            axs[1].loglog(f, np.abs(ratio), color='0.6', lw=0.7)
            axs[2].semilogx(f, np.angle(ratio, deg=True), color='0.6', lw=0.7)

        med_ratio = float(np.median(ratios_band))
        iqr = float(np.percentile(ratios_band, 75) - np.percentile(ratios_band, 25))

        # frequency-flatness of the median position's ratio inside the band
        r_c = np.abs(K_eff[4][:, ch, ch] / H_ctrl)
        flat_db = float(20 * np.std(np.log10(r_c[band])))

        axs[0].loglog(f, np.abs(H_ctrl), 'C0', lw=1.5, label='Filter1*Filter2 (Telica.mat)')
        axs[0].set_ylabel('|K|  [dac/cnt]')
        axs[0].legend(fontsize=8)
        axs[0].set_title(f'{axname}: does the FRF-measured controller equal the '
                         f'documented filters times a constant?')
        axs[1].axhline(1.0, color='C0', ls=':', lw=1, label='ratio = 1 (filters complete)')
        axs[1].axhline(med_ratio, color='C3', ls='--', lw=1,
                       label=f'median ratio = {med_ratio:.4f}')
        axs[1].set_ylabel('|K_eff / (F1*F2)|')
        axs[1].legend(fontsize=8)
        axs[2].set_ylabel('phase(K_eff / (F1*F2))  [deg]')
        axs[2].set_xlabel('frequency  [Hz]')
        axs[2].set_ylim(-200, 200)
        for a in axs:
            a.grid(True, which='both', alpha=0.3)
            a.axvspan(*BAND, color='C2', alpha=0.06)
        fig.tight_layout()
        fp = os.path.join(_SAVE_DIR, f'K_eff_vs_filters_{axname}.png')
        fig.savefig(fp, dpi=150)
        plt.close(fig)

        print(f'\n{axname}: median |K_eff/(F1*F2)| in [{BAND[0]:.0f},{BAND[1]:.0f}] Hz '
              f'= {med_ratio:.4f}  (1/ratio = {1/med_ratio:.3f})')
        print(f'    spread over 9 positions (IQR) = {iqr:.4f}')
        print(f'    flatness over frequency (center pos) = {flat_db:.2f} dB std')
        print(f'    figure -> {os.path.relpath(fp, _ROOT)}')
        summary[axname] = dict(median_ratio=med_ratio, inv_ratio=1 / med_ratio,
                               iqr_positions=iqr, flatness_dB=flat_db,
                               ratios_per_position=[float(r) for r in ratios_band])

    summary['ps_ordering'] = dict(rel_err_SG=float(e_sg), rel_err_GS=float(e_gs))
    with open(os.path.join(_SAVE_DIR, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    print('\nsummary  ->', os.path.relpath(os.path.join(_SAVE_DIR, 'summary.json'), _ROOT))


if __name__ == '__main__':
    main()
