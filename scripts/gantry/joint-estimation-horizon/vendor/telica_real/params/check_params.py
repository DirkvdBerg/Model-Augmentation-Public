"""G1 check (TR-005): every parameter has a row; vendored copies read params, not literals."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tr_env                                                     # noqa: E402

import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402

from params import telica_params as tp                            # noqa: E402
from model_augmentation.systems import gantry_ss as gss           # noqa: E402
from gantry_dynamic import controller as gctl                     # noqa: E402
from real_data_verification import telica_loader as tl            # noqa: E402
tr_env.check_no_leak()

ok_all = True


def check(label, ok, detail=''):
    global ok_all
    ok_all &= bool(ok)
    print(f'{"PASS" if ok else "FAIL"}  {label}  {detail}')


# (1) rows
need = list(tp.RAW14_NAMES) + ['Lb', 'cc1', 'cc2', 'ccy', 'V_BRK', 'V0_TANH',
                               'Kt_X', 'Kt_Y', 's_F_X', 's_F_Y', 'Ts', 'm_X']
missing = [n for n in need if n not in tp.PARAMS]
check('(1) every used parameter has a row', not missing, f'missing={missing} n_rows={len(tp.ROWS)}')

# (2) vendored gantry_ss and controller constants
names15 = list(tp.RAW14_NAMES) + ['Lb']
bad_gss = [n for n in names15
           if float(getattr(gss, n)) != float(torch.tensor(tp.PARAMS[n].value, dtype=torch.float32))]
check('(2a) gantry_ss constants == params (float32)', not bad_gss, f'mismatch={bad_gss}')
bad_ctl = [n for n in names15 if float(getattr(gctl, n)) != tp.PARAMS[n].value]
check('(2b) gantry_dynamic.controller constants == params', not bad_ctl, f'mismatch={bad_ctl}')
check('(2c) gantry_ss sample time == params', float(gss.ts) == float(torch.tensor(tp.TS)),
      f'{float(gss.ts)}')

# (3) loader
a2n = np.asarray(tp.a_to_n())
check('(3a) loader _A_TO_N == params.a_to_n()', np.array_equal(tl._A_TO_N, a2n),
      f'{tl._A_TO_N} vs {a2n}')
check('(3b) Telica 1.mat Kt == datasheet Kt', (tl._KT_X, tl._KT_X, tl._KT_Y) == tp.KT,
      f'{(tl._KT_X, tl._KT_Y)}')

# (4) no literal of the old sets survives as an assignment
old = ('22.8', '10.1', '10.2', '10.7', '14.5', '20.3', '1987.5', '0.725')
pat = re.compile(r'^\s*[A-Za-z_][\w, ]*=\s*.*\b(%s)\b' % '|'.join(re.escape(o) for o in old))
hits = []
for f in (gss.__file__, gctl.__file__):
    for i, line in enumerate(open(f, encoding='utf-8'), 1):
        code = line.split('#')[0]
        if pat.search(code):
            hits.append(f'{os.path.basename(f)}:{i}: {line.strip()}')
check('(4) no old literal assignments in vendored gantry_ss / controller', not hits, str(hits))

print()
print('combos10:', {k: round(v, 4) for k, v in tp.combos10().items()})
print('G1 CHECK', 'PASS' if ok_all else 'FAIL')
