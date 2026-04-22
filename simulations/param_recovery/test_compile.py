"""Quick diagnostics: torch.compile and checkpoint BPTT."""
import torch
import sys
sys.path.insert(0, '.')

from lpv_lfr_baseline.core.lfr_forward import lfr_forward
from lpv_lfr_baseline.core.lfr_simulate import simulate
from lpv_lfr_baseline.core.physics import (
    M1, M2, K, C, P, ts, build_poly_constants,
    mh as _mh, m1 as _m1, m2 as _m2, mb as _mb,
    Jb as _Jb, Jh as _Jh, Lb as _Lb, d as _d,
)
from lpv_lfr_baseline.core.lfr_matrices import build_G_matrix

dtype = torch.float64
alpha, beta, gamma, N0, N1, N2 = build_poly_constants(
    _m1, _m2, _mb, _mh, _Jb, _Jh, _Lb, _d
)
d0 = _mh * (alpha * gamma - beta ** 2)
G = build_G_matrix(N0, d0, M1, M2, K, C)

batch = 8
N = 200
x0 = torch.zeros(batch, 6, dtype=dtype)
u  = torch.randn(batch, N, 3, dtype=dtype) * 5.0

# ── Test 1: torch.compile ────────────────────────────────────────────────────
print("=" * 60)
print("Test 1: torch.compile on lfr_forward")
print("=" * 60)
try:
    compiled = torch.compile(lfr_forward, mode='reduce-overhead')
    # warm-up call
    x_t = x0[:, :, None].squeeze(-1) if False else x0[:1]
    x_t = torch.zeros(1, 6, dtype=dtype)
    u_t = torch.zeros(1, 3, dtype=dtype)
    Y_t = torch.zeros(1, dtype=dtype)
    _ = compiled(x_t, u_t, Y_t, G, K, C, _mh, alpha, beta, gamma, N0, N1, N2)
    print("  torch.compile: OK (warm-up succeeded)")
except Exception as e:
    print(f"  torch.compile FAILED: {type(e).__name__}: {e}")

# ── Test 2: checkpoint BPTT ──────────────────────────────────────────────────
print()
print("=" * 60)
print("Test 2: bptt_mode='checkpoint' simulate")
print("=" * 60)
try:
    x0_g = x0.clone().requires_grad_(True)
    res = simulate(
        x0_g, u, G, K, C, _mh, alpha, beta, gamma, N0, N1, N2, P, ts,
        bptt_mode='checkpoint',
    )
    loss = res.Y.pow(2).mean()
    loss.backward()
    print(f"  checkpoint simulate: OK, x0.grad norm = {x0_g.grad.norm().item():.4e}")
except Exception as e:
    print(f"  checkpoint simulate FAILED: {type(e).__name__}: {e}")

# ── Test 3: truncated BPTT ───────────────────────────────────────────────────
print()
print("=" * 60)
print("Test 3: bptt_mode='truncated' simulate")
print("=" * 60)
try:
    x0_g = x0.clone().requires_grad_(True)
    res = simulate(
        x0_g, u, G, K, C, _mh, alpha, beta, gamma, N0, N1, N2, P, ts,
        bptt_mode='truncated', segment_len=20,
    )
    loss = res.Y.pow(2).mean()
    loss.backward()
    print(f"  truncated simulate: OK, x0.grad norm = {x0_g.grad.norm().item():.4e}")
except Exception as e:
    print(f"  truncated simulate FAILED: {type(e).__name__}: {e}")

# ── Test 4: torch.jit.script ─────────────────────────────────────────────────
print()
print("=" * 60)
print("Test 4: torch.jit.script on lfr_forward")
print("=" * 60)
try:
    scripted = torch.jit.script(lfr_forward)
    print("  torch.jit.script: OK")
except Exception as e:
    print(f"  torch.jit.script FAILED: {type(e).__name__}: {e}")

print()
print("Done.")
