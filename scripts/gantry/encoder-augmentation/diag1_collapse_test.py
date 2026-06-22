"""
diag1_collapse_test.py
----------------------
Verify that linear_encoder_init_aug(nx_aug=0) is numerically identical
to Jan's linear_encoder_init with the same parameters.

Tests:
  1. Wb_psi_y identical (deterministic from A/B/C/D, no randomness)
  2. Wb_psi_u identical
  3. Wa_psi_y shape (0, ...) and Wa_psi_u shape (0, ...) when nx_aug=0
  4. Forward pass (flag_linear_only=True): outputs identical on same input
  5. Forward pass (flag_linear_only=False): outputs identical after copying
     net weights from Jan's encoder into aug encoder

All checks must pass to machine precision (atol=1e-6). No data required.

Usage:
    conda run -n GraduationProject python \\
        scripts/gantry/encoder-augmentation/diag1_collapse_test.py
"""

import sys
import numpy as np
import torch

from model_augmentation.fit_systems.pre_encoder import (
    linear_encoder_init,
    linear_encoder_init_aug,
)

# --------------------------------------------------------------------------
# Config — small synthetic system, collapse property is purely algebraic
# --------------------------------------------------------------------------
NX   = 6   # physical states (matches gantry)
NU   = 1
NY   = 3
NA   = 4   # keep small so matrix powers are fast
NB   = 4
N_NODES = 16
N_HIDDEN = 2
BATCH = 32
TOL = 1e-6

# --------------------------------------------------------------------------
# Synthetic system matrices (random stable LTI)
# --------------------------------------------------------------------------
rng = np.random.default_rng(0)
A_raw = rng.standard_normal((NX, NX))
# make stable: scale eigenvalues inside unit circle
eigvals = np.linalg.eigvals(A_raw)
A = A_raw / (np.max(np.abs(eigvals)) + 0.1)
B = rng.standard_normal((NX, NU))
C = rng.standard_normal((NY, NX))
D = rng.standard_normal((NY, NU))

# --------------------------------------------------------------------------
# Construct both encoders
# --------------------------------------------------------------------------
enc_jan = linear_encoder_init(
    A=A, B=B, C=C, D=D,
    nx=NX, nu=NU, ny=NY, na=NA, nb=NB,
    n_nodes_per_layer=N_NODES,
    n_hidden_layers=N_HIDDEN,
    flag_linear_only=False,
)

enc_aug = linear_encoder_init_aug(
    A=A, B=B, C=C, D=D,
    nx=NX, nu=NU, ny=NY, na=NA, nb=NB,
    nx_aug=0,
    n_nodes_per_layer=N_NODES,
    n_hidden_layers=N_HIDDEN,
    flag_linear_only=False,
)

# --------------------------------------------------------------------------
# Helper
# --------------------------------------------------------------------------
results = {}

def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results[name] = status
    marker = "  " if condition else "!!"
    print(f"  [{status}] {marker} {name}" + (f"  ({detail})" if detail else ""))
    return condition

# --------------------------------------------------------------------------
# Test 1 & 2: W^b weights identical
# --------------------------------------------------------------------------
print("\n--- W^b weight identity ---")
diff_y = (enc_jan.Wb_psi_y.detach() - enc_aug.Wb_psi_y.detach()).abs().max().item()
diff_u = (enc_jan.Wb_psi_u.detach() - enc_aug.Wb_psi_u.detach()).abs().max().item()
check("Wb_psi_y identical", diff_y < TOL, f"max_diff={diff_y:.2e}")
check("Wb_psi_u identical", diff_u < TOL, f"max_diff={diff_u:.2e}")

# --------------------------------------------------------------------------
# Test 3: W^a shapes are (0, ...) when nx_aug=0
# --------------------------------------------------------------------------
print("\n--- W^a zero-row shapes ---")
check("Wa_psi_y shape (0, (NA+1)*NY)",
      enc_aug.Wa_psi_y.shape == (0, (NA + 1) * NY),
      str(enc_aug.Wa_psi_y.shape))
check("Wa_psi_u shape (0, (NB+1)*NU)",
      enc_aug.Wa_psi_u.shape == (0, (NB + 1) * NU),
      str(enc_aug.Wa_psi_u.shape))

# --------------------------------------------------------------------------
# Test 4: Forward pass with flag_linear_only=True
# --------------------------------------------------------------------------
print("\n--- Forward pass: flag_linear_only=True ---")
enc_jan_lin = linear_encoder_init(
    A=A, B=B, C=C, D=D,
    nx=NX, nu=NU, ny=NY, na=NA, nb=NB,
    flag_linear_only=True,
)
enc_aug_lin = linear_encoder_init_aug(
    A=A, B=B, C=C, D=D,
    nx=NX, nu=NU, ny=NY, na=NA, nb=NB,
    nx_aug=0,
    flag_linear_only=True,
)

torch.manual_seed(42)
u_test = torch.randn(BATCH, (NB + 1) * NU)
y_test = torch.randn(BATCH, (NA + 1) * NY)

with torch.no_grad():
    out_jan = enc_jan_lin(u_test, y_test)
    out_aug = enc_aug_lin(u_test, y_test)

diff_fwd = (out_jan - out_aug).abs().max().item()
check("linear-only forward identical", diff_fwd < TOL, f"max_diff={diff_fwd:.2e}")
check("output shape (batch, NX)", out_aug.shape == (BATCH, NX), str(out_aug.shape))

# --------------------------------------------------------------------------
# Test 5: Forward pass with flag_linear_only=False (copy net weights)
# --------------------------------------------------------------------------
print("\n--- Forward pass: flag_linear_only=False (shared net weights) ---")
enc_aug.net.load_state_dict(enc_jan.net.state_dict())

with torch.no_grad():
    out_jan_nl = enc_jan(u_test, y_test)
    out_aug_nl = enc_aug(u_test, y_test)

diff_nl = (out_jan_nl - out_aug_nl).abs().max().item()
check("nonlinear forward identical (after net copy)", diff_nl < TOL, f"max_diff={diff_nl:.2e}")
check("output shape (batch, NX)", out_aug_nl.shape == (BATCH, NX), str(out_aug_nl.shape))

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
n_pass = sum(v == "PASS" for v in results.values())
n_fail = sum(v == "FAIL" for v in results.values())
print(f"\n{'='*50}")
print(f"  {n_pass}/{len(results)} checks passed")
if n_fail > 0:
    print(f"  {n_fail} FAILED -- linear_encoder_init_aug does NOT collapse to")
    print(f"  linear_encoder_init when nx_aug=0. Fix before proceeding.")
    sys.exit(1)
else:
    print("  Collapse property confirmed: nx_aug=0 is numerically identical")
    print("  to Jan's linear_encoder_init.")
