"""Transient share in the OBJECTIVE's weighting (each channel / ystd^2) for the checkpoint runs.

R029b-R033 pooled the error in metres, where Y (ystd 0.19 m) dominates and X1/X2 (ystd 0.032 m) barely
count, while the training loss divides each channel by ystd. This re-weights their saved per-channel
window RMS (`chan0` over [0, 400), `chanK` over [100, 400)): share = sum_c (MS0_c - MSK_c)/ystd_c^2
/ sum_c MS0_c/ystd_c^2. Read-only.
"""
__project_origin__ = "added"

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import et_paths                                                  # noqa: E402
import enc_common as ec                                          # noqa: E402

G5 = os.path.join(et_paths.OUT, 'g5')
CM = os.path.join(et_paths.OUT, 'ckpt_modes')


def shares(m, ystd):
    c0 = np.asarray(m['chan0']) ** 2
    cK = np.asarray(m['chanK']) ** 2
    w = 1.0 / ystd ** 2
    per = (c0 - cK) / c0
    return dict(objective=float(((c0 - cK) * w).sum() / (c0 * w).sum()),
                metres=float((c0 - cK).sum() / c0.sum()), per_channel=per.tolist())


def main():
    cfg, data, norm, fit_sys, hp, dims = ec.build()
    ystd = np.asarray(norm.ystd, float).ravel()
    src = [('81757 val (no burn-in)', os.path.join(G5, 'r2_ckpt_81757', 'r2_ckpt.json'), 'closed_loop'),
           ('81758 val (burn-in 100)', os.path.join(G5, 'r2_ckpt_81758', 'r2_ckpt.json'), 'closed_loop'),
           ('84032 val (no projection)', os.path.join(G5, 'r2_ckpt_84032', 'r2_ckpt.json'), 'closed_loop'),
           ('84032 TRAIN T1-T14', os.path.join(CM, '84032_train_transient.json'), 'pooled')]
    out = {}
    for lab, path, key in src:
        d = json.load(open(path))[key]
        out[lab] = {}
        print(f'\n  {lab}')
        for arm, m in d.items():
            s = shares(m, ystd)
            out[lab][arm] = s
            print(f'    x0 = {arm:<16} share in metres {s["metres"]:6.1%}   in OBJECTIVE weighting '
                  f'{s["objective"]:6.1%}   per channel X1/X2/Y ' + ' '.join(f'{x:6.1%}' for x in s['per_channel']))
    with open(os.path.join(CM, 'objective_share.json'), 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
