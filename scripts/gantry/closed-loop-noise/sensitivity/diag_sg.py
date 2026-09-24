"""Diagnostic for G2 attempt 2 (no gate role): |SG_emp| against the model readings, and where in
the record the feedforward differences live (motion vs post-motion settling).

    python sensitivity/diag_sg.py <g2 run folder name>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cn_env                                                           # noqa: E402

import numpy as np                                                      # noqa: E402

from measure import spec                                                # noqa: E402
from sensitivity import loop                                            # noqa: E402
from sensitivity.g2_sensitivity import model_S, A2N, PRE                # noqa: E402
from real_data_verification.telica_loader import _A_TO_N                # noqa: E402
cn_env.check_no_leak()

RUN = sys.argv[1] if len(sys.argv) > 1 else 'g2_sensitivity'
OUT = cn_env.out_dir('g2_diag_sg')
AX = cn_env.AX


def main():
    z = np.load(os.path.join(cn_env.OUTPUTS, RUN, 'g2.npz'), allow_pickle=True)
    fe, SG, coh, P_list = z['fe'], z['SG_emp'], z['coh'], list(z['positions'])
    Sm, SGm, _ = model_S(P_list, fe, readings=('sliding', 'tanh'))
    # where do feedforward differences live? energy of d i_ff per phase, a few positions
    recs = spec.all_records()
    per_op = {}
    for (s, op, it, p) in recs[:40]:
        try:
            d = spec.load(p)
        except Exception:
            continue
        j0 = d['j0']
        per_op.setdefault(op, []).append((d['u_ff'] / _A_TO_N)[j0 - PRE:j0 - PRE + 6978])
        mv = np.where(np.abs(np.diff(d['r'], axis=0)).sum(1) > 0)[0]
        if op not in ('_len',):
            per_op.setdefault('_len', []).append(int(mv[-1] - mv[0]))
    L = per_op.pop('_len')
    print(f'[diag] motion duration (first to last setpoint change): median {np.median(L)} samples '
          f'({np.median(L) / spec.FS * 1e3:.0f} ms)')
    for op, v in per_op.items():
        U = np.stack(v); dU = U - U.mean(0)
        e_mot = (dU[:, :PRE + int(np.median(L)) + 200] ** 2).sum()
        e_all = (dU ** 2).sum()
        print(f'[diag] {op}: {len(v)} logs, share of feedforward-difference energy in '
              f'[j0 - {PRE}, end of motion + 10 ms): {e_mot / e_all:.3f}')
    for i in range(3):
        m = coh[:, i] >= 0.9
        for rd in ('sliding', 'tanh'):
            SGA = (SGm[rd] * A2N[None, None, None, :]).mean(0)
            r = np.abs(SG[m, i, i]) / np.abs(SGA[m, i, i])
            ph = np.degrees(np.angle(SG[m, i, i] / SGA[m, i, i]))
            for lo, hi in ((5, 30), (30, 80), (80, 150), (150, 250)):
                b = (fe[m] >= lo) & (fe[m] < hi)
                if b.any():
                    print(f'[diag] SG_{AX[i]}{AX[i]} emp/model({rd:7s}) {lo:3d}-{hi:3d} Hz: |ratio| '
                          f'median {np.median(r[b]):.3f} [{r[b].min():.3f}, {r[b].max():.3f}], '
                          f'phase median {np.median(ph[b]):+.1f} deg')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for i in range(3):
        a = ax[i]
        m = coh[:, i] >= 0.9
        a.loglog(fe[m], np.abs(SG[m, i, i]), '.', color=spec.COLORS[i], ms=3, label='empirical (coh >= 0.9)')
        for rd, ls in (('sliding', '-'), ('tanh', 'dashed')):
            SGA = (SGm[rd] * A2N[None, None, None, :]).mean(0)
            a.loglog(fe[1:], np.abs(SGA[1:, i, i]), color=spec.INK, ls=ls, lw=1, label=f'model {rd}')
        a.set(xlim=(3, 1000), xlabel='frequency [Hz]', ylabel='|SG| [m/A]',
              title=f'Process sensitivity {AX[i]} <- i_ff {AX[i]}')
        a.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(OUT, 'sg_emp_vs_model.png'), dpi=110)


if __name__ == '__main__':
    main()
