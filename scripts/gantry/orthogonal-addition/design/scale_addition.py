"""Scale a stateless addition file by a factor (every oa_* matrix; absorber masses too).
Usage: scale_addition.py <in.mat> <factor> <out.mat>"""
__project_origin__ = "added"

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import addition_io as aio                              # noqa: E402

src, a, dst = sys.argv[1], float(sys.argv[2]), sys.argv[3]
oa = aio.load_mat(src)
for k in ('oa_Ma0', 'oa_Ma1', 'oa_Ma2', 'oa_Ca0', 'oa_Ka0', 'oa_Ka1', 'oa_Ka2', 'oa_m'):
    oa[k] = a * oa[k]
aio.save_mat(dst, oa, '%s scaled by %.6g' % (os.path.basename(src), a))
print('wrote %s (x%.6g)' % (dst, a))
