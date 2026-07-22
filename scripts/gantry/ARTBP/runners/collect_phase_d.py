"""collect_phase_d.py -- read the Phase D grid npz files (train_<mode>_seed<seed>.npz), compute the
per-mode statistics + the pre-registered verdict, and save the comparison figure. Run AFTER the
cluster job array finishes:  python scripts/gantry/ARTBP/runners/collect_phase_d.py

Pre-registered criteria (D-120 reframe):
 1. control valid: fixed DC sign-locked ~ -4.5e-6; geom collapses to the +/-3e-7-class band.
 2. poly collapses: poly4/poly6 endpoint DC in the collapsed band, sign scattered across seeds.
 3. variance: report Var(poly)/Var(geom); expect << 1 (both poly-tails). Which alpha wins?
 4. fit preserved: poly/geom heldout nf-RMS not worse than the fixed control.
"""
import os
import sys
import glob

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # scripts/gantry/ARTBP
datDir = os.path.join(HERE, 'data'); figDir = os.path.join(HERE, 'figures')
MODES  = ['fixed', 'geom', 'poly4', 'poly6']


def load_grid():
    runs = {m: [] for m in MODES}
    for f in sorted(glob.glob(os.path.join(datDir, 'train_*_seed*.npz'))):
        d = np.load(f, allow_pickle=True)
        m = str(d['mode'])
        if m in runs:
            runs[m].append(dict(seed=int(d['seed']), dc=float(d['dc_endpoint']),
                                var=float(d['dcgrad_var']), nfrms=float(d['heldout_nfrms']),
                                dc_traj=d['dc'], wt=float(d['walltime'])))
    return runs


def main():
    runs = load_grid()
    present = {m: len(v) for m, v in runs.items()}
    print(f'[collect] runs found: {present}')
    if sum(present.values()) == 0:
        print('No train_*_seed*.npz in', datDir, '-- run the grid first.'); sys.exit(1)

    stats = {}
    print('\n==== PER-MODE (mean +/- sd over seeds) ====')
    for m in MODES:
        r = runs[m]
        if not r:
            print(f'  {m:6s}: (no runs)'); continue
        dc = np.array([x['dc'] for x in r]); var = np.array([x['var'] for x in r])
        nf = np.array([x['nfrms'] for x in r])
        stats[m] = dict(dc=dc, var=var, nf=nf,
                        signs=np.sign(dc), frac_neg=float(np.mean(dc < 0)))
        print(f'  {m:6s} (n={len(r)}): DC {dc.mean():+.3e} (sd {dc.std():.2e}, '
              f'frac<0 {stats[m]["frac_neg"]:.2f}) | var {var.mean():.3e} (sd {var.std():.2e}) | '
              f'nf-RMS {nf.mean():.4e}')

    # verdict
    print('\n==== PRE-REGISTERED VERDICT ====')
    if 'fixed' in stats:
        f = stats['fixed']
        c1 = (f['dc'].mean() < -2e-6) and (f['frac_neg'] > 0.99)
        print(f'1 control: fixed DC {f["dc"].mean():+.2e}, sign-locked neg {f["frac_neg"]:.2f} -> '
              f'{"PASS" if c1 else "CHECK"}')
    band = 1e-6
    for m in ('geom', 'poly4', 'poly6'):
        if m in stats:
            s = stats[m]; coll = np.mean(np.abs(s['dc'])) < band
            print(f'2 collapse {m}: |DC| {np.mean(np.abs(s["dc"])):.2e} < {band:.0e} -> '
                  f'{"PASS" if coll else "CHECK"} (sign frac<0 {s["frac_neg"]:.2f})')
    if 'geom' in stats:
        vg = stats['geom']['var'].mean()
        for m in ('poly4', 'poly6'):
            if m in stats:
                ratio = stats[m]['var'].mean() / vg
                print(f'3 variance {m}: Var({m})/Var(geom) = {ratio:.3f} -> '
                      f'{"PASS (poly reduces variance)" if ratio < 1 else "FAIL"}')
        if 'poly4' in stats and 'poly6' in stats:
            r46 = stats['poly6']['var'].mean() / stats['poly4']['var'].mean()
            print(f'  alpha winner: Var(poly6)/Var(poly4) = {r46:.3f} -> '
                  f'{"poly6 (a=6) lower" if r46 < 1 else "poly4 (a=4) lower"} '
                  f'(sd poly4 {stats["poly4"]["var"].std():.2e}, poly6 {stats["poly6"]["var"].std():.2e})')
    if 'fixed' in stats:
        base = stats['fixed']['nf'].mean()
        for m in ('geom', 'poly4', 'poly6'):
            if m in stats:
                dd = (stats[m]['nf'].mean() - base) / base
                print(f'4 fit {m}: nf-RMS {stats[m]["nf"].mean():.4e} ({dd*100:+.1f}% vs fixed) -> '
                      f'{"PASS" if dd < 0.1 else "CHECK"}')

    # figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    xs = np.arange(len(MODES))
    for i, m in enumerate(MODES):
        if m in stats:
            ax[0].scatter([i] * len(stats[m]['dc']), stats[m]['dc'] * 1e6, s=30)
    ax[0].axhline(0, color='k', lw=0.6); ax[0].axhspan(-0.3, 0.3, color='tab:green', alpha=0.12)
    ax[0].set_xticks(xs); ax[0].set_xticklabels(MODES)
    ax[0].set_ylabel('endpoint DC dY  [1e-6]'); ax[0].grid(True, alpha=0.3)
    ax[0].set_title('A  DC collapse per mode (band = clean noise)', fontsize=9)
    vv = [stats[m]['var'].mean() if m in stats else np.nan for m in MODES]
    ax[1].bar(xs, vv, color=['0.5', 'tab:red', 'tab:blue', 'tab:green'])
    ax[1].set_yscale('log'); ax[1].set_xticks(xs); ax[1].set_xticklabels(MODES)
    ax[1].set_ylabel('dcgrad variance (2nd half)'); ax[1].grid(True, which='both', alpha=0.3)
    ax[1].set_title('B  Estimator variance (lower is better)', fontsize=9)
    nn = [stats[m]['nf'].mean() * 1e3 if m in stats else np.nan for m in MODES]
    ax[2].bar(xs, nn, color=['0.5', 'tab:red', 'tab:blue', 'tab:green'])
    ax[2].set_xticks(xs); ax[2].set_xticklabels(MODES)
    ax[2].set_ylabel('heldout nf-RMS  [1e-3]'); ax[2].grid(True, alpha=0.3)
    ax[2].set_ylim(min(v for v in nn if np.isfinite(v)) * 0.98, max(v for v in nn if np.isfinite(v)) * 1.02)
    ax[2].set_title('C  Fit preserved?', fontsize=9)
    fig.suptitle('Phase D  ARTBP variance comparison (fixed / geom / poly4 / poly6)', fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    p = os.path.join(figDir, 'phase_d_comparison.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f'\nsaved {p}')


if __name__ == '__main__':
    main()
