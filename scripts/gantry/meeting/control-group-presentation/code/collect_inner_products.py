"""Per-parameter inner products of the learned correction with the ten parameter directions.

    python -u collect_inner_products.py      # writes inner_products.json next to this file

Reuses `_arm_fields` from `meeting-21-09-2026/code/fig_orthogonality.py` unchanged (imported,
not copied), so the fields, reference sets and expansion points are exactly those behind
`orthogonality_data.json`. Only the ten parameter directions J are used: no affine offset column,
no affine run. For each arm (84032 no projection, 84033 projection) and each split (the 14
training records, the 4 held-out records), and for the network output before removal (`raw`)
and the correction that reaches the model (`applied`):

    cos_j = <J_j, F> / (||J_j|| ||F||),   j = 1..10      normalised inner product per parameter
    rho   = ||J J^+ F|| / ||F||                          must reproduce orthogonality_data.json
"""
import json
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
M21 = HERE.parents[1] / 'meeting-21-09-2026' / 'code'
sys.path.insert(0, str(M21))

from fig_orthogonality import _arm_fields, _rho   # noqa: E402

NAMES = ['kb_sum', 'cg1', 'cg2', 'cy', 'cb_sum', 'mh', 'm_total', 'm_diff', 'J_eff', 'd']


def cosines(J, F):
    num = J.T @ F
    return (num / (torch.linalg.vector_norm(J, dim=0) * torch.linalg.vector_norm(F))).tolist()


def main():
    out, ref_holder = {}, {}
    for arm in ('noproj', 'tangent'):
        out[arm] = {}
        for split in ('train', 'val'):
            F_raw, F_app, J, _gamma = _arm_fields(arm, ref_holder, split)
            rec = {}
            for kind, F in (('raw', F_raw), ('applied', F_app)):
                rec[kind] = dict(cos=dict(zip(NAMES, cosines(J, F))), rho=_rho(J, F))
                print('%-8s %-5s %-7s rho %.3e  max|cos| %.3e' % (
                    arm, split, kind, rec[kind]['rho'],
                    max(abs(v) for v in rec[kind]['cos'].values())), flush=True)
            rec['n_samples'] = int(J.shape[0] // 6)
            out[arm][split] = rec
            del F_raw, F_app, J, _gamma
    p = HERE / 'inner_products.json'
    p.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print('wrote %s' % p)


if __name__ == '__main__':
    main()
