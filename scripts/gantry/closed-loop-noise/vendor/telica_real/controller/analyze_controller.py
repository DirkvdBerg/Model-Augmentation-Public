"""G3 step 1: what the Telica zpk controller is (poles, integrators, gain vs frequency)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tr_env                                                          # noqa: E402

import numpy as np                                                     # noqa: E402
from scipy.signal import freqz                                         # noqa: E402

from real_data_verification.telica_controller import _load_ba          # noqa: E402
from params import telica_params as tp                                 # noqa: E402
tr_env.check_no_leak()

ba, Ts = _load_ba()
print(f'Ts = {Ts} s (params TS {tp.TS})')
for name in ('LX1', 'LX2', 'LY'):
    b, a = ba[name]
    b = b / a[0]; a = a / a[0]
    p, z = np.roots(a), np.roots(b)
    n_int = int(np.sum(np.abs(p - 1.0) < 1e-6))
    print(f'\n{name}: order num {len(b)-1} den {len(a)-1}; b0 = {b[0]:.4e} (feedthrough D)')
    print(f'  poles at z=1 (integrators): {n_int}; |p| max {np.abs(p).max():.9f}')
    for pk in sorted(p, key=lambda c: -abs(c)):
        f = abs(np.angle(pk)) / (2 * np.pi * Ts)
        print(f'   pole |z|={abs(pk):.8f}  f={f:9.2f} Hz')
    for zk in sorted(z, key=lambda c: -abs(c)):
        f = abs(np.angle(zk)) / (2 * np.pi * Ts)
        print(f'   zero |z|={abs(zk):.8f}  f={f:9.2f} Hz')
    fr = np.array([0.1, 1, 5, 10, 30, 50, 100, 150, 200, 300, 500, 586, 957, 1000, 2000, 2393,
                   3057, 5000, 9999])
    _, H = freqz(b, a, worN=fr, fs=1 / Ts)
    print('  |C| [A/m] and phase [deg]:')
    for f, h in zip(fr, H):
        print(f'   {f:8.1f} Hz  |C| = {abs(h):.4e}  phase = {np.degrees(np.angle(h)):8.2f}')
