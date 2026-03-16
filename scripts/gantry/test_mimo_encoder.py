"""
test_mimo_encoder.py
--------------------
Verifies that the encoder network inside SSE_Interconnect correctly handles
MIMO (Multiple-Input Multiple-Output) systems after fixing a SISO-only bug.

Background
----------
The augmentation framework uses an encoder network to map a window of past
inputs u[k-nb:k] and past outputs y[k-na:k] into an initial hidden state x[0].
Its input size must be:

    input_size = nb * nu + na * ny

where nb, na are the history window lengths and nu, ny are the number of
input and output channels.

The bug (model_augmentation/fit_systems/interconnect.py, line 361)
-------------------------------------------------------------------
The original code hardcoded self.ny = tuple(), which forces:

    np.prod(tuple()) = 1   (empty product equals 1)

so the encoder input was always nb*nu + na*1, regardless of the actual ny.
For a SISO system (ny=1) this is harmless. For the gantry (ny=3) it
silently dropped output channels 2 and 3 from the encoder history:

    buggy input size  = nb*nu + na*1  = 10*3 + 10*1 = 40
    correct input size = nb*nu + na*ny = 10*3 + 10*3 = 60

The fix
-------
Uncomment the original line so ny is set from the constructor argument:

    self.ny = tuple() if ny is None else ((ny,) if isinstance(ny, int) else ny)

This matches the MIMO-capable version used in the wafer-stage variant of the
framework (model-aug-wafer, interconnect.py line 527).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from model_augmentation.fit_systems.interconnect import modified_encoder_net

# History window lengths (arbitrary — only the ratio matters for this test)
nb = 10   # past input samples fed to encoder
na = 10   # past output samples fed to encoder

print("=" * 55)
print("Encoder input size verification")
print("=" * 55)

# --- Test 1: SISO (nu=1, ny=1) -------------------------------------------
# The fix must not change behavior for SISO systems.
# Both old and new code give np.prod((1,)) = 1, so input size is unchanged.
nu_siso, ny_siso, nx_siso = 1, 1, 2
siso = modified_encoder_net(nb=nb, nu=nu_siso, na=na, ny=ny_siso, nx=nx_siso)
siso_in   = siso.net.net_lin.weight.shape[1]
expected  = nb * nu_siso + na * ny_siso          # = 10 + 10 = 20
result    = "PASS" if siso_in == expected else "FAIL"
print(f"\nTest 1 — SISO (nu={nu_siso}, ny={ny_siso})")
print(f"  Expected encoder input size : {expected}")
print(f"  Actual encoder input size   : {siso_in}")
print(f"  Result                      : {result}")

# --- Test 2: MIMO (nu=3, ny=3) — gantry dimensions -----------------------
# After the fix, output history now contributes na*ny=30 instead of na*1=10.
nu_mimo, ny_mimo, nx_mimo = 3, 3, 6
mimo = modified_encoder_net(nb=nb, nu=nu_mimo, na=na, ny=ny_mimo, nx=nx_mimo)
mimo_in      = mimo.net.net_lin.weight.shape[1]
expected     = nb * nu_mimo + na * ny_mimo        # = 30 + 30 = 60
buggy_would  = nb * nu_mimo + na * 1             # = 30 + 10 = 40
result       = "PASS" if mimo_in == expected else "FAIL"
print(f"\nTest 2 — MIMO (nu={nu_mimo}, ny={ny_mimo})  [gantry dimensions]")
print(f"  Expected encoder input size : {expected}  (nb*nu + na*ny = {nb}*{nu_mimo} + {na}*{ny_mimo})")
print(f"  Buggy encoder input size    : {buggy_would}  (nb*nu + na*1  = {nb}*{nu_mimo} + {na}*1)")
print(f"  Actual encoder input size   : {mimo_in}")
print(f"  Result                      : {result}")

print("\n" + "=" * 55)
all_pass = (siso_in == nb * nu_siso + na * ny_siso) and (mimo_in == nb * nu_mimo + na * ny_mimo)
print("Overall:", "ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED")
print("=" * 55)
