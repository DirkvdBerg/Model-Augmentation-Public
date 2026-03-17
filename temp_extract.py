"""
temp_extract.py
---------------
Exhaustive validation that PyTorch can implement the exact ZOH (Tóth) discretization
method for the gantry LPV model.

Answers the question: "Can we replace scipy cont2discrete with torch.linalg.matrix_exp
inside the autograd graph, and trust the results?"

Six tests, each with a clear pass criterion:

  Test 1 — matrix_exp numerical correctness vs scipy expm
  Test 2 — augmented matrix trick gives same A_d, B_d as scipy cont2discrete
  Test 3 — A_c singularity: naive formula fails, augmented trick succeeds
  Test 4 — backward() reaches Y (differentiability confirmed)
  Test 5 — gradient value correct (finite-difference verification)
  Test 6 — correctness across Y values (not specific to Y=0.3)

Run from repo root:
  conda run -n GraduationProject python temp_extract.py
"""

import torch
import numpy as np
from scipy.linalg import expm
from scipy.signal import cont2discrete

torch.set_default_dtype(torch.float64)

PASS = "PASS"
FAIL = "FAIL"
SEP  = "=" * 60


def build_gantry_Ac_Bc(Y_val: float):
    """Build continuous-time A_c, B_c for gantry at scalar Y [m]. Returns numpy arrays."""
    mb, mh, m1, m2  = 22.8, 10.1, 10.2, 10.7
    Jb, Jh          = 1.0, 0.05
    cg1, cg2, cy    = 14.5, 20.3, 10.0
    cb1, cb2        = 9.0, 9.0
    kb1, kb2        = 1987.5, 1987.5
    Lb, d           = 0.725, 0.1

    M = np.array([
        [m1+m2+mb+mh,                (m1-m2)*Lb/2 - mh*Y_val,                             0],
        [(m1-m2)*Lb/2 - mh*Y_val,   Jb+Jh+(m1+m2)*Lb**2/4 + mh*d**2 + mh*Y_val**2, -mh*d],
        [0,                          -mh*d,                                               mh],
    ])
    C_damp = np.array([
        [cg1+cg2,           (cg1-cg2)*Lb/2,                   0],
        [(cg1-cg2)*Lb/2,    cb1+cb2+(cg1+cg2)*Lb**2/4,        0],
        [0,                  0,                                cy],
    ])
    K = np.array([[0,0,0],[0,kb1+kb2,0],[0,0,0]], dtype=float)
    P = np.array([[1, 1, 0],[Lb/2, -Lb/2, 0],[0, 0, 1]], dtype=float)

    Mi  = np.linalg.inv(M)
    A_c = np.block([[np.zeros((3,3)), np.eye(3)], [-Mi@K, -Mi@C_damp]])
    B_c = np.block([[np.zeros((3,3))], [Mi]]) @ P   # stage coordinates
    return A_c, B_c, np.linalg.det(M)


# ======================================================================
print(SEP)
print("Test 1 — matrix_exp numerical correctness vs scipy expm")
print(SEP)

A_c_np, _, _ = build_gantry_Ac_Bc(Y_val=0.3)
ts = 1.0 / 16e3

expm_scipy = expm(A_c_np * ts)
expm_torch = torch.linalg.matrix_exp(torch.tensor(A_c_np) * ts).numpy()

err1 = np.max(np.abs(expm_torch - expm_scipy))
tol1 = 1e-14
status1 = PASS if err1 < tol1 else FAIL
print(f"  max|expm_torch - expm_scipy| = {err1:.2e}  (tol {tol1:.0e})  ->  {status1}")


# ======================================================================
print()
print(SEP)
print("Test 2 — augmented matrix trick: A_d, B_d match scipy cont2discrete")
print(SEP)

A_c_np, B_c_np, _ = build_gantry_Ac_Bc(Y_val=0.3)
n, m = 6, 3

# scipy reference
A_d_ref, B_d_ref, _, _, _ = cont2discrete((A_c_np, B_c_np, np.eye(n), np.zeros((n,m))),
                                           dt=ts, method='zoh')

# torch augmented matrix exponential
M_aug = torch.zeros(n+m, n+m)
M_aug[:n, :n] = torch.tensor(A_c_np)
M_aug[:n, n:] = torch.tensor(B_c_np)
EM = torch.linalg.matrix_exp(M_aug * ts)
A_d_torch = EM[:n, :n].numpy()
B_d_torch = EM[:n, n:].numpy()

err_A = np.max(np.abs(A_d_torch - A_d_ref))
err_B = np.max(np.abs(B_d_torch - B_d_ref))
tol2  = 1e-14
status2 = PASS if err_A < tol2 and err_B < tol2 else FAIL
print(f"  max|A_d_torch - A_d_scipy| = {err_A:.2e}  (tol {tol2:.0e})  ->  {PASS if err_A < tol2 else FAIL}")
print(f"  max|B_d_torch - B_d_scipy| = {err_B:.2e}  (tol {tol2:.0e})  ->  {PASS if err_B < tol2 else FAIL}")
print(f"  Overall  ->  {status2}")


# ======================================================================
print()
print(SEP)
print("Test 3 — singular A_c: naive B_d formula fails, augmented trick succeeds")
print(SEP)

A_c_np, B_c_np, _ = build_gantry_Ac_Bc(Y_val=0.3)

# Confirm A_c is singular
rank = np.linalg.matrix_rank(A_c_np)
det  = np.linalg.det(A_c_np)
print(f"  rank(A_c) = {rank} (full rank would be {n})")
print(f"  det(A_c)  = {det:.2e}")
singular = rank < n
print(f"  A_c is singular: {singular}  ->  {PASS if singular else FAIL}")

# Naive formula attempt: B_d = A_c^{-1} (A_d - I) B_c
A_d_ref = expm(A_c_np * ts)
try:
    A_c_inv = np.linalg.inv(A_c_np)
    B_d_naive = A_c_inv @ (A_d_ref - np.eye(n)) @ B_c_np
    print(f"  Naive formula: did not raise (singular matrix not caught by numpy inv)")
    # Check if result is numerically garbage
    B_d_ref = cont2discrete((A_c_np, B_c_np, np.eye(n), np.zeros((n,m))), dt=ts, method='zoh')[1]
    naive_err = np.max(np.abs(B_d_naive - B_d_ref))
    print(f"  Naive B_d error vs scipy: {naive_err:.2e}  {'(garbage — singular inv)' if naive_err > 1e-6 else ''}")
    naive_fails = naive_err > 1e-6
except np.linalg.LinAlgError:
    print(f"  Naive formula: raised LinAlgError (singular matrix)  ->  confirmed fails")
    naive_fails = True

# Augmented trick
M_aug = torch.zeros(n+m, n+m)
M_aug[:n, :n] = torch.tensor(A_c_np)
M_aug[:n, n:] = torch.tensor(B_c_np)
EM = torch.linalg.matrix_exp(M_aug * ts)
B_d_aug = EM[:n, n:].numpy()
B_d_ref = cont2discrete((A_c_np, B_c_np, np.eye(n), np.zeros((n,m))), dt=ts, method='zoh')[1]
aug_err  = np.max(np.abs(B_d_aug - B_d_ref))
print(f"  Augmented trick B_d error vs scipy: {aug_err:.2e}  ->  {PASS if aug_err < 1e-14 else FAIL}")
status3 = PASS if naive_fails and aug_err < 1e-14 else FAIL
print(f"  Overall  ->  {status3}")


# ======================================================================
print()
print(SEP)
print("Test 4 — backward() reaches Y through full augmented matrix_exp pipeline")
print(SEP)

Y_t = torch.tensor(0.3, requires_grad=True)
mh  = torch.tensor(10.1)
Lb  = torch.tensor(0.725)

# Minimal M(Y) that captures Y-dependence (just the off-diagonal and diagonal terms)
M_01 = (torch.tensor(10.2) - torch.tensor(10.7)) * Lb / 2 - mh * Y_t
M_11 = torch.tensor(1.0) + torch.tensor(0.05) + (torch.tensor(10.2) + torch.tensor(10.7)) * Lb**2/4 \
       + mh * torch.tensor(0.1)**2 + mh * Y_t**2

# Import the full implementation to test end-to-end
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts', 'gantry'))
from gantry_lpv_torch import gantry_lpv_matrices_torch

Y_grad = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
A_d, B_d, _, _ = gantry_lpv_matrices_torch(Y_grad)
loss = A_d.sum() + B_d.sum()
loss.backward()

grad_exists = Y_grad.grad is not None
status4 = PASS if grad_exists else FAIL
print(f"  backward() succeeded: {grad_exists}  ->  {status4}")
if grad_exists:
    print(f"  dL/dY = {Y_grad.grad.item():.6e}")


# ======================================================================
print()
print(SEP)
print("Test 5 — gradient value correct: autograd vs central finite differences")
print(SEP)

eps    = 1e-5
Y_nom  = 0.3

def scalar_loss(Y_val):
    Y_t = torch.tensor(Y_val, dtype=torch.float64)
    A_d, B_d, _, _ = gantry_lpv_matrices_torch(Y_t)
    return (A_d.sum() + B_d.sum()).item()

# Central finite difference
fd_grad = (scalar_loss(Y_nom + eps) - scalar_loss(Y_nom - eps)) / (2 * eps)

# Autograd gradient
Y_ag = torch.tensor(Y_nom, dtype=torch.float64, requires_grad=True)
A_d, B_d, _, _ = gantry_lpv_matrices_torch(Y_ag)
(A_d.sum() + B_d.sum()).backward()
ag_grad = Y_ag.grad.item()

rel_err = abs(ag_grad - fd_grad) / (abs(fd_grad) + 1e-30)
tol5    = 1e-5   # finite differences have O(eps^2) error, expect match to ~5 digits
status5 = PASS if rel_err < tol5 else FAIL
print(f"  Autograd  dL/dY = {ag_grad:.8e}")
print(f"  Finite diff dL/dY = {fd_grad:.8e}")
print(f"  Relative error   = {rel_err:.2e}  (tol {tol5:.0e})  ->  {status5}")


# ======================================================================
print()
print(SEP)
print("Test 6 — correctness across Y values (Y = 0.05 to 0.75 m, 10 points)")
print(SEP)

Y_sweep = np.linspace(0.05, 0.75, 10)
max_err_A = max_err_B = 0.0
all_grad  = True
tol6 = 1e-14

for Y_val in Y_sweep:
    A_c_np, B_c_np, det_M = build_gantry_Ac_Bc(Y_val)

    # scipy reference
    A_ref, B_ref = cont2discrete(
        (A_c_np, B_c_np, np.eye(6), np.zeros((6,3))), dt=ts, method='zoh'
    )[:2]

    # torch
    Y_t  = torch.tensor(Y_val, dtype=torch.float64, requires_grad=True)
    A_d, B_d, _, _ = gantry_lpv_matrices_torch(Y_t)
    (A_d.sum() + B_d.sum()).backward()

    max_err_A = max(max_err_A, np.max(np.abs(A_d.detach().numpy() - A_ref)))
    max_err_B = max(max_err_B, np.max(np.abs(B_d.detach().numpy() - B_ref)))
    if Y_t.grad is None:
        all_grad = False

status6 = PASS if max_err_A < tol6 and max_err_B < tol6 and all_grad else FAIL
print(f"  max|A_d error| across sweep = {max_err_A:.2e}  (tol {tol6:.0e})  ->  {PASS if max_err_A < tol6 else FAIL}")
print(f"  max|B_d error| across sweep = {max_err_B:.2e}  (tol {tol6:.0e})  ->  {PASS if max_err_B < tol6 else FAIL}")
print(f"  backward() succeeded at all Y: {all_grad}  ->  {PASS if all_grad else FAIL}")
print(f"  Overall  ->  {status6}")


# ======================================================================
print()
print(SEP)
all_status = [status1, status2, status3, status4, status5, status6]
all_pass   = all(s == PASS for s in all_status)
print("SUMMARY")
print(SEP)
labels = [
    "Test 1 — matrix_exp vs scipy expm",
    "Test 2 — augmented trick vs cont2discrete",
    "Test 3 — singular A_c handled correctly",
    "Test 4 — backward() reaches Y",
    "Test 5 — gradient value (finite difference)",
    "Test 6 — correctness across Y sweep",
]
for label, status in zip(labels, all_status):
    print(f"  {status}  {label}")
print()
print(f"  {'ALL PASS — torch ZOH method validated' if all_pass else 'SOME FAILED — review output above'}")
print(SEP)
