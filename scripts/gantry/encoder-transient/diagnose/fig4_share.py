"""What share of the loss was transient in meeting 07-09's Figure 4? (read-only recomputation)

Reads the SAME cached curves the figure was drawn from (meeting-07-09-2026/code/figure_data_<run>.npz,
`window_curve` = per-step RMS over 256 training windows, metres, (nf, 3)) and computes
share = (MS[0:400] - MS[100:400]) / MS[0:400] per channel, pooled in metres, and pooled in the
objective's own weighting (each channel divided by ystd^2, the training normalisation).
"""
__project_origin__ = "added"

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import et_paths                                                  # noqa: E402
import enc_common as ec                                          # noqa: E402

FIG = os.path.join(et_paths.REPO, 'scripts', 'gantry', 'meeting', 'meeting-07-09-2026', 'code')
K = 100


def share(ms_t, w):
    """ms_t (nf, 3) per-step mean square; w (3,) channel weights."""
    m = ms_t @ w
    return float((m.mean() - m[K:].mean()) / m.mean()), float(np.sqrt(m.mean())), float(np.sqrt(m[K:].mean()))


def main():
    cfg, data, norm, fit_sys, hp, dims = ec.build()
    ystd = np.asarray(norm.ystd, float).ravel()
    print(f'  ystd (training normalisation) X1/X2/Y: {ystd}')
    out = {}
    for run in ('81262', '81265'):
        d = np.load(os.path.join(FIG, f'figure_data_{run}.npz'))
        c = np.asarray(d['window_curve'], float)                 # (nf, 3) per-step RMS [m]
        ms = c ** 2
        r = dict(nf=int(c.shape[0]), grow_fig=float(d['window_grow']))
        for i, ch in enumerate(('X1', 'X2', 'Y')):
            w = np.zeros(3); w[i] = 1.0
            r[f'share_{ch}'], r[f'rms_{ch}'], r[f'rmsK_{ch}'] = share(ms, w)
        r['share_pooled_m'], r['rms_pooled_m'], _ = share(ms, np.ones(3))
        r['share_objective'], _, _ = share(ms, 1.0 / ystd ** 2)
        r['objective_weight_of_Y'] = float((ms.mean(0) / ystd ** 2)[2] / (ms.mean(0) / ystd ** 2).sum())
        out[run] = r
        print(f'\n  run {run} (nf {r["nf"]}, figure grow {r["grow_fig"]:.3f})')
        for ch in ('X1', 'X2', 'Y'):
            print(f'    {ch:<3} transient share {r["share_" + ch]:6.1%}   RMS window {r["rms_" + ch]:.3e}'
                  f'  RMS after step 100 {r["rmsK_" + ch]:.3e} m')
        print(f'    pooled in metres: share {r["share_pooled_m"]:.1%}')
        print(f'    pooled in the OBJECTIVE weighting (1/ystd^2): share {r["share_objective"]:.1%}  '
              f'(Y carries {r["objective_weight_of_Y"]:.1%} of the objective)')
    od = os.path.join(et_paths.OUT, 'ckpt_modes')
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, 'fig4_share.json'), 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
