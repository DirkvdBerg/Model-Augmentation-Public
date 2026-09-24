"""Import bootstrap for every closed-loop-noise script (CN-003).

Usage at the top of a script in a closed-loop-noise subfolder:

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import cn_env                      # noqa: E402

Puts the vendored telica-real copy (`vendor/telica_real`, and its own `vendor/`) first on sys.path
through that copy's `tr_env`, so `params`, `friction`, `controller`, `baseline`, `data_index` and
the loader resolve to the copies here, never to `telica-real/` or the repo originals.
`check_no_leak()` enforces it. `out_dir(run)` is `closed-loop-noise/outputs/<run>`.
"""
import os
import sys

CN = os.path.dirname(os.path.abspath(__file__))
TRV = os.path.join(CN, 'vendor', 'telica_real')
if TRV not in sys.path:
    sys.path.insert(0, TRV)
import tr_env                                   # noqa: E402  (inserts TRV and TRV/vendor first)

if CN in sys.path:
    sys.path.remove(CN)
sys.path.append(CN)                             # our own packages, after the vendored ones

REPO = tr_env.REPO
OUTPUTS = tr_env.OUTPUTS
FS = 20000.0
AX = ('X1', 'X2', 'Y')


def check_no_leak():
    tr_env.check_no_leak()


def out_dir(run):
    return tr_env.out_dir(run)
