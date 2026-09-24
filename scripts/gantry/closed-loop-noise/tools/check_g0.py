"""G0 checks (CN-002, CN-003): vendored loader reproduces the 09-22 standstill rms; LITERATURE.md
holds the method register; imports resolve inside closed-loop-noise.

    python tools/check_g0.py
Prints summaries only (data access policy).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cn_env                                                           # noqa: E402

import numpy as np                                                      # noqa: E402

import data_index                                                       # noqa: E402
from real_data_verification.telica_loader import load_telica_log_cl     # noqa: E402
from baseline import cl_sim                                             # noqa: E402,F401
from controller import telica_bank                                      # noqa: E402,F401
cn_env.check_no_leak()

QUOTED = np.array([8.69, 10.63, 5.94])          # nm, 09-22 handoff section 4
OUT = cn_env.out_dir('g0_check')


def first_change(r):
    """First index j0 with r[j0 + 1] != r[j0] on any axis (09-22 probe, CN-003)."""
    mv = np.abs(np.diff(r, axis=0)).sum(1) > 0
    return int(np.argmax(mv))


def main():
    res = {}
    p = data_index.path('train', 'xpos_-60_ypos-40', 'iter0')
    d = load_telica_log_cl(p, pre_motion_ms=None)
    e = d['e'].numpy()
    k = d['motion_idx']
    j0 = first_change(d['r'].numpy())         # CN-003: the 09-22 probe's definition
    w = e[:j0 - 200]
    std = w.std(0) * 1e9
    print(f'[g0] record train/xpos_-60_ypos-40/iter0: T {len(e)}, motion_idx {k}, j0 {j0}, '
          f'window {len(w)} samples = {len(w) / cn_env.FS * 1e3:.1f} ms')
    print('[g0] standstill std [nm] %s (quoted %s), |diff| %s'
          % (np.round(std, 4), QUOTED, np.round(np.abs(std - QUOTED), 4)))
    e_r = (d['r'] - d['q1']).numpy()[:j0 - 200]
    print('[g0] r - q1 vs M2*1e-6 on the window: max |diff| %.3e m' % np.abs(e_r - w).max())
    ok_rms = bool(np.all(np.abs(std - QUOTED) < 0.005))
    d50 = load_telica_log_cl(p)
    ok_default = d50['motion_idx'] == 1000 and len(d50['r']) == len(e) - (k - 1000)
    print(f'[g0] default call: motion_idx {d50["motion_idx"]} (expect 1000), '
          f'T {len(d50["r"])} (expect {len(e) - (k - 1000)})')
    dc = load_telica_log_cl(p, pre_motion_ms=None, engine='c')
    same = all(np.array_equal(dc[x].numpy(), d[x].numpy()) for x in ('r', 'q1', 'e', 'i_fb', 'u_ff'))
    print(f'[g0] c parser bit-identical to python parser: {same}')
    res['loader'] = dict(std_nm=std.tolist(), quoted=QUOTED.tolist(), pass_rms=ok_rms,
                         default_trim_ok=bool(ok_default), c_engine_identical=bool(same))

    lit = open(os.path.join(cn_env.CN, 'LITERATURE.md'), encoding='utf-8').read().lower()
    names = {'S-only bound': ['s-only', 'only bound', '|s|^2', '|s|²'],
             'indirect LPM': ['indirect'], 'ILC residual bound': ['ilc'],
             'direct LPM': ['direct'], 'robust/fast BLA': ['bla'],
             'Oomen-Bosgra': ['oomen'], 'CLOE': ['cloe'], 'ALS': ['als'],
             'three-cornered hat': ['three-cornered', 'cornered hat'],
             'Ljung-Forssell shaping': ['forssell']}
    found = {n: any(k_ in lit for k_ in keys) for n, keys in names.items()}
    for n, f in found.items():
        print(f'[g0] LITERATURE.md mentions {n:24s}: {f}')
    res['literature'] = found
    res['pass'] = ok_rms and ok_default and all(found.values())
    print(f'[g0] loader PASS {ok_rms and ok_default}; literature PASS {all(found.values())}')
    json.dump(res, open(os.path.join(OUT, 'g0_check.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
