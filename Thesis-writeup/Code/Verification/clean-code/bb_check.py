"""D-295: the thesis black box (R8) through Clean/blackbox's own evaluation.

    python -u bb_check.py BB_RUN_DIR OUT

clean-up-verification/convert_bb.py writes the thesis run into OUT/run with deepSI's net classes (Clean/ has no
jan_model); then `python -m blackbox.eval_bb OUT/run OUT` runs from the Clean/ root of hcommon (HARNESS_CLEAN_ROOT).
Compare OUT with compare_csv.py bb.
"""
import os
import subprocess
import sys

import hcommon as H

if len(sys.argv) != 3:
    sys.exit(__doc__)
src, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
run = os.path.join(out, 'run')
subprocess.run([sys.executable, '-u', os.path.join(H.REPO, 'clean-up-verification', 'convert_bb.py'), src, run],
               check=True)
env = dict(os.environ, PYTHONPATH=H.CLEAN)
subprocess.run([sys.executable, '-u', '-m', 'blackbox.eval_bb', run, out], cwd=H.CLEAN, env=env, check=True)
print('BB_CHECK DONE %s -> %s (Clean root %s)' % (src, out, H.CLEAN), flush=True)
