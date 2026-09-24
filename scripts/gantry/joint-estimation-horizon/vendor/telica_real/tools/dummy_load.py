"""G0 dummy load: hold ~300 MB and keep one core busy for DURATION seconds (TR-002).

Run only through tools/watchdog.ps1. Prints its PID so the kill test can check it is gone.
"""
import os
import sys
import time

import numpy as np

DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0

print(f'[dummy] pid={os.getpid()} duration={DURATION:.0f}s', flush=True)
buf = np.ones((300 * 1024 * 1024) // 8)     # ~300 MB resident
t0 = time.time()
k = 0
while time.time() - t0 < DURATION:
    buf[::4096] += 1.0                       # touch pages, keep a core busy
    k += 1
    if k % 2000 == 0:
        print(f'[dummy] t={time.time() - t0:5.1f}s sum={buf[::1 << 20].sum():.3e}', flush=True)
print(f'[dummy] done after {time.time() - t0:.1f}s', flush=True)
