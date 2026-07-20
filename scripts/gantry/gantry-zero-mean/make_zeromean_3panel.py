"""make_zeromean_3panel.py -- the "system is zero-mean" figure as WITH / WITHOUT / DIFFERENCE.

Recomputes mean(with-MSD) and mean(without-MSD) per channel from the raw standstill trajectory files
(the saved v1b npz only stored the difference). Positions have their known SETPOINT subtracted (clean
operating point on standstill records; Y from the record name, 0 for X/Theta). Metric per channel =
the fraction of the signal that is a constant = |mean| / std (dimensionless, physically meaningful,
sample-size robust). Three panels sharing the channel axis:
    A  WITH MSD        -- each signal ~ zero-mean
    B  WITHOUT MSD     -- each signal ~ zero-mean
    C  DIFFERENCE      -- |mean(with) - mean(without)| / std : the MSD adds no mean
Log y so the tiny fractions are visible; reference line at 1.0 = a pure constant (100% DC).
Reuses v1b's loader/paths. Output: figures/meeting/meeting_fig0_zeromean_3panel.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat

plt.rcParams.update({'font.size': 13, 'axes.titlesize': 14, 'axes.labelsize': 12})
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DIR_W = os.path.join(REPO, 'data', 'gantry', 'matlab', 'trajectory', 'augmentation')
DIR_N = os.path.join(DIR_W, 'baseline')
OUT = os.path.join(HERE, 'figures', 'meeting'); os.makedirs(OUT, exist_ok=True)

# 12 raw channels (v1b order); we show the 9 logical + force channels.
SEL = [3, 4, 5, 6, 7, 8, 9, 10, 11]
NAMES = ['X', 'Theta', 'Y', 'dX', 'dTheta', 'dY', 'F_X1', 'F_X2', 'F_Y']
YSET = {'Ym30': -0.30, 'Ym15': -0.15, 'Y000': 0.0, 'Yp15': 0.15, 'Yp30': 0.30}


def load_record(directory, f):
    d = loadmat(os.path.join(directory, f), squeeze_me=True)
    y = np.asarray(d['y'], float); xl = np.asarray(d['x_logical'], float); u = np.asarray(d['u_total'], float)
    n = min(len(y), len(xl), len(u))
    return np.hstack([y[:n], xl[:n, :3], xl[:n, 3:6], u[:n]])   # (n,12)


def main():
    in_n = {f for f in os.listdir(DIR_N) if f.endswith('.mat')}
    recs = sorted(f for f in os.listdir(DIR_W) if f.endswith('.mat') and f in in_n and 'standstill' in f)
    print('standstill records:', recs)
    poolW = {c: [] for c in range(12)}; poolN = {c: [] for c in range(12)}
    for f in recs:
        key = next((k for k in YSET if k in f), None)
        op = np.zeros(12); op[2] = op[5] = YSET.get(key, 0.0)   # subtract Y setpoint (stage Y idx2, logical Y idx5)
        sw = load_record(DIR_W, f) - op; sn = load_record(DIR_N, f) - op
        for c in range(12):
            poolW[c].append(sw[:, c]); poolN[c].append(sn[:, c])

    fracW, fracN, fracD = [], [], []
    for c in SEL:
        w = np.concatenate(poolW[c]); n = np.concatenate(poolN[c])
        sw = w.std() + 1e-30
        fracW.append(abs(w.mean()) / sw)
        fracN.append(abs(n.mean()) / n.std() + 1e-30)
        fracD.append(abs(w.mean() - n.mean()) / sw)     # difference relative to the signal's own scale

    x = np.arange(len(SEL))
    fh, ax = plt.subplots(1, 3, figsize=(15, 5.0), sharey=True)
    data = [(fracW, 'A:  WITH MSD (truth)', '#c0392b'),
            (fracN, 'B:  WITHOUT MSD (baseline)', '#27ae60'),
            (fracD, 'C:  DIFFERENCE  |with - without|', '#2c3e50')]
    for a, (vals, title, col) in zip(ax, data):
        a.bar(x, vals, color=col, width=0.62)
        a.axhline(1.0, color='k', ls='--', lw=1.2)
        a.text(len(SEL) - 0.5, 1.15, 'pure constant (100%)', ha='right', va='bottom', fontsize=9, color='0.3')
        a.axhline(0.01, color='0.6', ls=':', lw=1.0)
        a.text(0, 0.0115, '1%', fontsize=8, color='0.4')
        a.set_yscale('log'); a.set_ylim(1e-6, 3.0)
        a.set_xticks(x); a.set_xticklabels(NAMES, rotation=45, ha='right')
        a.set_title(title, color=col); a.grid(True, axis='y', alpha=0.3, which='both')
        for xi, v in zip(x, vals):
            a.text(xi, v * 1.3, f'{v:.0e}', ha='center', va='bottom', fontsize=8, rotation=90)
    ax[0].set_ylabel('fraction of the signal that is constant\n( |mean| / std )')
    fh.suptitle('The system is zero-mean: every signal is <1% constant with AND without the MSD, '
                'and the MSD adds no mean (panel C)', fontsize=14)
    fh.text(0.5, 0.005, 'Standstill records, native 20 kHz, positions minus their setpoint. '
            'All bars far below 1% -> no signal carries a meaningful constant; panel C -> adding the '
            'hidden mass shifts no mean.', ha='center', fontsize=9.5, style='italic')
    fh.tight_layout(rect=(0, 0.03, 1, 0.96))
    p = os.path.join(OUT, 'meeting_fig0_zeromean_3panel.png')
    fh.savefig(p, dpi=150); plt.close(fh)
    print('saved', p)
    print('WITH :', dict(zip(NAMES, [f'{v:.1e}' for v in fracW])))
    print('WO   :', dict(zip(NAMES, [f'{v:.1e}' for v in fracN])))
    print('DIFF :', dict(zip(NAMES, [f'{v:.1e}' for v in fracD])))


if __name__ == '__main__':
    main()
