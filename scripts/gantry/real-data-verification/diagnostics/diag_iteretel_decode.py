"""
diag_iteretel_decode.py
-----------------------
Decode the raw log schema. Telica.mat DatalogListVarMapping lists 25 ETEL
channels (X_CMD_POS, X_ENC_POS, X_DAC, X_HIGS_INPUT, X_HIGS_OUTPUT,
X_FF_OUTPUT, X_FB_OUTPUT, X_TOTAL_OUTPUT, ...), and iter0.log was found to
have 25 columns of which only 13 (TimeStamp + M0/M2/MF230/MF30 x 3 axes) have
been identified. This script:

  Stage 1 (always): dump the FULL header and per-column stats of
                    iterETEL.log and iter0.log.
  Stage 2 (conditional on what exists):
    - HIGS input/output scatter: a HIGS in gain mode gives output = k_h*input
      on a straight line; the slope is the candidate missing gain.
    - Raw-unit chain check: error counts -> Filter1 -> Filter2 vs DAC/FB
      columns, zero unit conversions involved.

Run:
    conda run -n GraduationProject python scripts/gantry/real-data-verification/diag_iteretel_decode.py
"""

__project_origin__ = "added"

import os
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import lfilter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..', '..'))

OP_FOLDER  = 'xpos_-60_ypos-40'
_DATA_ROOT = os.path.join(_ROOT, 'kamtin-data', 'Data Telica',
                          '06 40 mm XL 80 mm YL', 'train', OP_FOLDER)
_TELICA    = os.path.join(_ROOT, 'kamtin-data', 'Telica.mat')
_SAVE_DIR  = os.path.join(_ROOT, 'simulations', 'gantry_subnet',
                          'diagnostics', 'iteretel_decode')

_FILES = ['iterETEL.log', 'iter0.log']


def read_header_and_data(path):
    with open(path, 'r') as fh:
        header = fh.readline()
    raw_names = [c.strip() for c in header.rstrip('\n').split('\t')]
    clean, seen = [], {}
    for c in raw_names:
        base = c.split(':')[0].replace('.', '_').strip()
        if base == '':
            base = '_empty'
        n = seen.get(base, 0)
        seen[base] = n + 1
        clean.append(base if n == 0 else f'{base}__{n}')
    df = pd.read_csv(path, sep='\t', header=None, names=clean, skiprows=1,
                     engine='python', index_col=False)
    return raw_names, df


def dump_schema(fname):
    path = os.path.join(_DATA_ROOT, fname)
    if not os.path.exists(path):
        print(f'\n### {fname}: NOT FOUND')
        return None
    raw_names, df = read_header_and_data(path)
    print(f'\n### {fname}: {df.shape[0]} rows x {len(raw_names)} header fields')
    print(f'{"idx":>4} {"header (raw)":42s} {"min":>14} {"max":>14} '
          f'{"std":>12} {"n_uniq":>7}')
    for i, rn in enumerate(raw_names):
        col = df.columns[i] if i < df.shape[1] else None
        if col is None:
            print(f'{i:4d} {rn:42s}  (no data column)')
            continue
        v = pd.to_numeric(df[col], errors='coerce').to_numpy(float)
        ok = np.isfinite(v)
        if ok.sum() == 0:
            print(f'{i:4d} {rn:42s}  (all NaN / non-numeric)')
            continue
        vv = v[ok]
        print(f'{i:4d} {rn:42s} {vv.min():14.5g} {vv.max():14.5g} '
              f'{vv.std():12.5g} {len(np.unique(vv[:2000])):7d}')
    return df


def load_filters():
    mat = loadmat(_TELICA, squeeze_me=True, struct_as_record=False)
    mp = mat['MachineParam']
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
    return filts


def find_cols(df, substr):
    return [c for c in df.columns if substr.lower() in c.lower()]


def higs_scatter(df, fname):
    """If HIGS input/output pairs exist, fit slope and plot scatter."""
    results = {}
    for ax in ('X', 'Y'):
        cin  = find_cols(df, f'{ax}_HIGS_INPUT')
        cout = find_cols(df, f'{ax}_HIGS_OUTPUT')
        if not cin or not cout:
            continue
        u = pd.to_numeric(df[cin[0]],  errors='coerce').to_numpy(float)
        y = pd.to_numeric(df[cout[0]], errors='coerce').to_numpy(float)
        ok = np.isfinite(u) & np.isfinite(y)
        u, y = u[ok], y[ok]
        if u.std() == 0:
            print(f'  {ax}: HIGS input is constant, skip')
            continue
        slope = float(np.dot(u, y) / np.dot(u, u))
        corr  = float(np.corrcoef(u, y)[0, 1])
        # fraction of samples on the gain line within 5% (HEURISTIC band)
        on_line = np.abs(y - slope * u) <= 0.05 * (np.abs(slope * u) + 1e-12)
        frac = float(on_line.mean())
        print(f'  {ax}: HIGS out = {slope:.4f} * in   corr={corr:.4f}   '
              f'fraction within 5% of line = {frac:.2f}')
        results[ax] = dict(slope=slope, corr=corr, frac_on_line=frac)

        fig, ax1 = plt.subplots(figsize=(6, 6))
        ax1.plot(u, y, '.', ms=1, alpha=0.3)
        uu = np.linspace(u.min(), u.max(), 10)
        ax1.plot(uu, slope * uu, 'C3--', lw=1.2, label=f'slope {slope:.4f}')
        ax1.set_xlabel(f'{ax}_HIGS_INPUT')
        ax1.set_ylabel(f'{ax}_HIGS_OUTPUT')
        ax1.set_title(f'{ax} HIGS: pure gain would collapse onto the line')
        ax1.legend()
        ax1.grid(alpha=0.3)
        fp = os.path.join(_SAVE_DIR, f'higs_scatter_{ax}_{fname.replace(".log","")}.png')
        fig.savefig(fp, dpi=150)
        plt.close(fig)
        print(f'    figure -> {os.path.relpath(fp, _ROOT)}')
    return results


def raw_chain_check(df, filts, fname):
    """If CMD/ENC (or HIGS output) and DAC/FB columns exist, verify
    Filter1->Filter2 in raw units (counts in, DAC out): zero conversions."""
    results = {}
    for ax in ('X', 'Y'):
        (b1, a1), (b2, a2) = filts[ax]
        # candidate error inputs, most direct first
        cands_in = []
        c_cmd, c_enc = find_cols(df, f'{ax}_CMD_POS'), find_cols(df, f'{ax}_ENC_POS')
        if c_cmd and c_enc:
            cmd = pd.to_numeric(df[c_cmd[0]], errors='coerce').to_numpy(float)
            enc = pd.to_numeric(df[c_enc[0]], errors='coerce').to_numpy(float)
            cands_in.append(('CMD-ENC [cnt]', cmd - enc))
        c_hin = find_cols(df, f'{ax}_HIGS_OUTPUT')
        if c_hin:
            cands_in.append((f'{ax}_HIGS_OUTPUT',
                             pd.to_numeric(df[c_hin[0]], errors='coerce').to_numpy(float)))
        # candidate outputs
        cands_out = []
        for tag in (f'{ax}_FB_OUTPUT', f'{ax}_DAC', f'{ax}_TOTAL_OUTPUT'):
            c = find_cols(df, tag)
            if c:
                cands_out.append((tag, pd.to_numeric(df[c[0]], errors='coerce').to_numpy(float)))
        if not cands_in or not cands_out:
            continue
        print(f'  {ax}: raw chain check')
        for in_name, e in cands_in:
            ok = np.isfinite(e)
            if ok.sum() < 100 or np.nanstd(e) == 0:
                continue
            e0 = np.where(ok, e, 0.0)
            rec = lfilter(b2, a2, lfilter(b1, a1, e0))
            for out_name, yout in cands_out:
                ok2 = ok & np.isfinite(yout)
                r, y = rec[ok2], yout[ok2]
                if y.std() == 0:
                    continue
                corr = float(np.corrcoef(r, y)[0, 1])
                scale = float(np.dot(r, y) / np.dot(r, r))
                print(f'    F2(F1({in_name})) vs {out_name}: '
                      f'corr={corr:7.4f}  LS_scale={scale:.6g}')
                results[f'{ax}:{in_name}->{out_name}'] = dict(corr=corr, scale=scale)
    return results


def main():
    os.makedirs(_SAVE_DIR, exist_ok=True)
    print('=' * 70)
    print('DIAG ITERETEL DECODE: full schema dump + conditional chain checks')
    print('=' * 70)

    filts = load_filters()
    all_results = {}
    for fname in _FILES:
        df = dump_schema(fname)
        if df is None:
            continue
        res = {}
        print(f'\n--- {fname}: HIGS check ---')
        res['higs'] = higs_scatter(df, fname)
        if not res['higs']:
            print('  no HIGS columns found')
        print(f'--- {fname}: raw-unit chain check ---')
        res['chain'] = raw_chain_check(df, filts, fname)
        if not res['chain']:
            print('  no raw CMD/ENC/DAC/FB columns found')
        all_results[fname] = res

    with open(os.path.join(_SAVE_DIR, 'summary.json'), 'w') as fh:
        json.dump(all_results, fh, indent=2)
    print('\nsummary  ->', os.path.relpath(os.path.join(_SAVE_DIR, 'summary.json'), _ROOT))


if __name__ == '__main__':
    main()
