"""G0: the vendored D-197 R0 harness reproduces V1 (ET-001).

Runs the vendored `r0_encoder_init.analyse` on V1 at stride 1 (every window, as D-197) and checks
Y and dY RMS error against the exact truth within 1 % of 2.18e-05 m and 2.79e-02 m/s.
"""
__project_origin__ = "added"

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import et_paths                                                  # noqa: E402

import numpy as np                                               # noqa: E402
import enc_common as ec                                          # noqa: E402
import r0_encoder_init as r0                                     # noqa: E402

REF = {'Y': 2.18e-05, 'dY': 2.79e-02}      # problem log 19.2, V1 column
TOL = 0.01                                 # ET-001


def main():
    et_paths.check()
    cfg, data, norm, fit_sys, hp, dims = ec.build()
    print(f'  mode {cfg.mode}  na=nb={dims[0]}  nx_ann={cfg.nx_ann}  encoder {cfg.encoder_init}  '
          f'closed_loop {cfg.closed_loop}')
    res = r0.analyse(cfg, norm, fit_sys, dims, ['V1_standstill_Yp10'], stride=1)
    sc = res['V1_standstill_Yp10']['enc_vs_exact']
    ec.print_scores('V1 encoder vs EXACT (vendored)', sc)
    out, ok = {}, True
    for ch, ref in REF.items():
        i = ec.CHANNELS.index(ch)
        got = float(sc['rms_phys'][i])
        rel = abs(got - ref) / ref
        ok &= rel <= TOL
        out[ch] = dict(got=got, ref=ref, rel_dev=rel)
        print(f'  G0 {ch:<3} got {got:.4e}  D-197 {ref:.2e}  rel dev {rel:.3%}  '
              f'{"OK" if rel <= TOL else "FAIL"}')
    out['all_channels_rms'] = dict(zip(ec.CHANNELS, map(float, sc['rms_phys'])))
    out['bias'] = dict(zip(ec.CHANNELS, map(float, sc['bias_phys'])))
    out['pass'] = bool(ok)
    os.makedirs(os.path.join(et_paths.OUT, 'g0'), exist_ok=True)
    with open(os.path.join(et_paths.OUT, 'g0', 'g0_reproduce.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'G0 {"PASS" if ok else "FAIL"}')


if __name__ == '__main__':
    main()
