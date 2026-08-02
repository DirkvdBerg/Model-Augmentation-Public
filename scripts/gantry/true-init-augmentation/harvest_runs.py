"""Recover the training-arm series from the run logs.

The three arms were killed externally at roughly half their epoch budget, before
`true_init_train.py` reached its end-of-run JSON dump. Everything needed is still
in the streamed stdout (one line per epoch) and in the per-epoch `_last`
checkpoints, so the runs are recoverable rather than lost. This parses the logs
into the JSON the run would have written, and reports the series statistics that
decide the pre-registered predictions P1 and P2.

Usage:
  python -u scripts/gantry/true-init-augmentation/harvest_runs.py <log> [<log> ...]
"""
__project_origin__ = "added"

import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
OUT = os.path.join(REPO, 'simulations', 'gantry_subnet', 'diagnostics')

EP = re.compile(
    r'^\s*ep\s+(\d+)\s+train sqrtMSE ([\deE.+-]+)\s+val nf-RMS ([\deE.+-]+) m\s+'
    r'\(([+-][\d.]+) %\)\s+DC_Y ([\deE.+-]+) \(([+-][\d.]+) %\)\s+'
    r'\|g\| ([\deE.+-]+)\s+\|W_fin\| ([\deE.+-]+)\s+\|w\| ([\deE.+-]+)\s+(\d+) s',
    re.M)
BASE = re.compile(r'\[ANN off\] val nf-RMS ([\deE.+-]+) m')
BASEDC = re.compile(r"\[ANN off\] val per-window DC scatter \[m\] \['([\deE.+-]+)', "
                    r"'([\deE.+-]+)', '([\deE.+-]+)'\]")
LRRE = re.compile(r'lr ([\deE.+-]+)\s')


def parse(path):
    txt = open(path, encoding='utf-8', errors='replace').read()
    base = float(BASE.search(txt).group(1))
    bdc = [float(v) for v in BASEDC.search(txt).groups()]
    lr = float(LRRE.search(txt).group(1))
    rows = []
    for m in EP.finditer(txt):
        rows.append(dict(epoch=int(m.group(1)), train_sqrt=float(m.group(2)),
                         val=float(m.group(3)), val_pct=float(m.group(4)),
                         dc_y=float(m.group(5)), dc_pct=float(m.group(6)),
                         grad=float(m.group(7)), w_fin=float(m.group(8)),
                         ann_out=float(m.group(9)), sec=int(m.group(10))))
    return dict(lr=lr, ann_off=base, ann_off_dc=bdc, history=rows)


def main(paths):
    res = {}
    print('Training arms, recovered from the run logs (all three killed at ~half budget)\n')
    print(f'  {"lr":>8}{"epochs":>8}{"updates":>9}{"ANN off":>13}{"best":>13}'
          f'{"best %":>9}{"final %":>9}{"worst %":>9}{"DC_Y best %":>13}{"|w| final":>12}')
    for p in paths:
        d = parse(p)
        h = d['history']
        v = np.array([r['val'] for r in h])
        dc = np.array([r['dc_pct'] for r in h])
        best_i = int(v.argmin())
        d.update(n_epochs=len(h), n_updates=26 * len(h),
                 best=float(v.min()), best_epoch=int(h[best_i]['epoch']),
                 best_pct=float(h[best_i]['val_pct']),
                 final_pct=float(h[-1]['val_pct']),
                 worst_pct=float(v.max() / d['ann_off'] * 100 - 100),
                 dc_best_pct=float(dc.min()), dc_final_pct=float(dc[-1]),
                 frac_above=float((v > d['ann_off']).mean()),
                 w_fin_final=float(h[-1]['w_fin']),
                 ann_out_final=float(h[-1]['ann_out']))
        res[f"lr{d['lr']:g}"] = d
        print(f"  {d['lr']:>8.0e}{len(h):>8}{26*len(h):>9}{d['ann_off']:>13.4e}"
              f"{d['best']:>13.4e}{d['best_pct']:>9.2f}{d['final_pct']:>9.2f}"
              f"{d['worst_pct']:>9.2f}{dc.min():>13.2f}{h[-1]['ann_out']:>12.2e}")

    print('\n  P1  "val nf-RMS improves"      -> best % below zero on any arm')
    print('  P2  "the Y per-window DC does NOT improve materially"'
          '  -> DC_Y best % near zero')
    print(f'\n  fraction of validation points ABOVE the ANN-off value:')
    for k, d in res.items():
        print(f'    {k:<10}{100*d["frac_above"]:.0f} %  '
              f'({d["n_epochs"]} points, {d["n_updates"]} updates)')
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, 'true_init_train_harvest.json')
    with open(p, 'w') as f:
        json.dump(res, f, indent=2)
    print(f'\n  wrote {p}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
