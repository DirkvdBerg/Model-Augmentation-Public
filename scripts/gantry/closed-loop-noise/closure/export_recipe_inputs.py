"""Export the Telica loop and the noise models for the MATLAB recipe (G6, matlab/recipe_sensitivity.m).

    python closure/export_recipe_inputs.py
Writes outputs/recipe_inputs/telica_loop.mat: per position (15) the discrete plant (A, B, C) for
the tanh and sliding readings, the identified controller (A, B, C, D; error m -> force N), the
position weights of the G1 pool, the VAR shaping filters (tanh = production source; stuck = a fit
of the MEASURED error spectrum, used to show what injecting Phi_e would do), and the measured rms.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cn_env                                                           # noqa: E402

import numpy as np                                                      # noqa: E402
from scipy.io import savemat                                            # noqa: E402

from sensitivity import loop                                            # noqa: E402
cn_env.check_no_leak()

OUT = cn_env.out_dir('recipe_inputs')


def op_of(n):
    return n.split('/')[1]


def main():
    g1 = np.load(os.path.join(cn_env.OUTPUTS, 'g1_spectra', 'g1.npz'), allow_pickle=True)
    names = list(g1['names']); keep = g1['keep']; Kr = g1['Kr']; xy = g1['xy']
    ops = sorted(set(op_of(n) for n in names))
    pos = np.stack([xy[[i for i, n in enumerate(names) if op_of(n) == op][0]] for op in ops])
    w = np.array([Kr[[i for i, n in enumerate(names) if op_of(n) == op and keep[i]]].sum() for op in ops], float)
    w /= w.sum()
    d = dict(positions=pos, w=w, ops=np.array(ops, dtype=object), Ts=loop.TS)
    for rd in ('tanh', 'sliding'):
        Ap, Bp, Cp = zip(*[loop.plant_ss(y, rd) for y in pos])
        d[f'Ap_{rd}'] = np.stack(Ap); d[f'Bp_{rd}'] = np.stack(Bp); d[f'Cp_{rd}'] = np.stack(Cp)
    d['Ac'], d['Bc'], d['Cc'], d['Dc'] = loop.ctrl_ss()
    for rd in ('tanh', 'stuck'):
        z = np.load(os.path.join(cn_env.OUTPUTS, 'g3_source', f'hv_{rd}.npz'))
        d[f'A_{rd}'] = z['A']; d[f'Sig_{rd}'] = z['Sig']
    g1j = json.load(open(os.path.join(cn_env.OUTPUTS, 'g1_spectra', 'g1.json')))
    d['rms_meas_nm'] = np.array(g1j['rms_psd_nm'])
    savemat(os.path.join(OUT, 'telica_loop.mat'), d, do_compression=True)
    print(f'[export] {len(ops)} positions, weights {np.round(w, 3)}; wrote telica_loop.mat')


if __name__ == '__main__':
    main()
