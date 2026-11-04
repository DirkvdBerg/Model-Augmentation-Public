"""D-295: the thesis black-box evaluation (Thesis-writeup/Code/Evaluation/eval_blackbox.py) of R8, written into OUT.

    THESIS_BB_DIR=<R8 run folder> python -u bb_mainline.py OUT

eval_blackbox writes its prediction files to eval_common.CACHE (under Thesis-writeup/Results); here CACHE points at
OUT before eval_blackbox is imported, so nothing is written into the results tree. Compare OUT with the clean side's
bb_check.py output: compare_bb_preds.py MAINLINE_OUT CLEAN_OUT.
"""
import os
import sys
from pathlib import Path

import hcommon as H

if len(sys.argv) != 2:
    sys.exit(__doc__)
out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=False)
H.select_side('mainline')
os.environ.setdefault('THESIS_EVAL', 'final')
sys.path.insert(0, os.path.join(H.REPO, 'Thesis-writeup', 'Code', 'Evaluation'))
import eval_common  # noqa: E402

eval_common.CACHE = out
import eval_blackbox  # noqa: E402

eval_blackbox.CACHE = out
eval_blackbox.main()
written = sorted(p.name for p in out.glob('preds_BB*.npz'))
print('BB_MAINLINE DONE %s: %s' % (out, written), flush=True)
sys.exit(0 if written else 1)
