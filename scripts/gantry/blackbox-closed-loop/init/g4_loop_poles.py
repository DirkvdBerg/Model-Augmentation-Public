"""Mechanism for BB-013: the FRF-init black box's LINEAR closed loop, poles and sensitivity.

At init the nonlinear branches' last layers are zero, so f and h are their linear bypasses
(x+ = A x + B u + b, y = C x + c) in normalised coordinates. The bank holds Cfb folded into the
same coordinates, so the model loop is a plain discrete feedback interconnection. Reported per
distinct controller: max|z| of the loop, and the spread of max|z| over 200 random-sign
perturbations of [A B] and C of size delta (the parameters the step size moves).
Compared against the grey box's FP plant in the same coordinates is unnecessary: g2_loop_check
already gives the physical FP loop at 4 kHz (max|z| 0.977).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'adapter'))
import bbcl                                                                # noqa: E402

cfg, CFG, entry = bbcl.load_cfg(local=True)
import numpy as np                                                         # noqa: E402
import torch                                                               # noqa: E402
from gantry_dynamic.data import load_datasets, compute_normalization      # noqa: E402
import bb_model                                                            # noqa: E402

np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
data = load_datasets(cfg)
norm = compute_normalization(cfg, data)
fs = bb_model.build_blackbox(cfg, data, norm, arm='frf', attach=True)
bank = fs.simulator.bank
fn, hn = fs.hfn.fn.net, fs.hfn.hn.net
W = fn.net_lin.weight.detach().double().numpy()
A, B = W[:, :bb_model.NX], W[:, bb_model.NX:]
C = hn.net_lin.weight.detach().double().numpy()
nu = bank.nu
Ms, Me = bank.M_state.double().numpy(), bank.M_error.double().numpy()


def loop(A, B, C, k):
    Ck, Ak = Ms[k, :nu], Ms[k, nu:]
    Dk, Bk = Me[k, :nu], Me[k, nu:]
    # model x+ = A x + B (u + Ck xc + Dk e), e = -C x (residual form, data terms are exogenous)
    return np.block([[A - B @ Dk @ C, B @ Ck], [-Bk @ C, Ak]])


res = dict(open_loop_max_abs_eig=float(np.max(np.abs(np.linalg.eigvals(A)))), per_controller=[])
rng = np.random.default_rng(cfg.seed)
for k in range(bank.n_controllers):
    r0 = float(np.max(np.abs(np.linalg.eigvals(loop(A, B, C, k)))))
    spread = {}
    for d in (1e-4, 1e-5, 1e-6, 1e-7):
        rs = []
        for _ in range(200):
            dA = d * rng.choice([-1.0, 1.0], size=A.shape)
            dB = d * rng.choice([-1.0, 1.0], size=B.shape)
            dC = d * rng.choice([-1.0, 1.0], size=C.shape)
            rs.append(np.max(np.abs(np.linalg.eigvals(loop(A + dA, B + dB, C + dC, k)))))
        spread[str(d)] = dict(median=float(np.median(rs)), max=float(np.max(rs)),
                              frac_unstable=float(np.mean(np.array(rs) >= 1.0)))
    res['per_controller'].append(dict(row=k, max_abs_z=r0, perturbed=spread))
    print('[poles] controller %d: loop max|z| %.6f | ' % (k, r0) + '  '.join(
        'd=%s: med %.4f max %.4f unstable %.0f%%' % (d, v['median'], v['max'], 100 * v['frac_unstable'])
        for d, v in spread.items()))
print('[poles] open-loop max|eig(A)| %.6f; |D_k| scale: %s' % (
    res['open_loop_max_abs_eig'],
    np.array2string(np.abs(Me[:, :nu]).max(axis=(1, 2)), precision=3)))
os.makedirs(os.path.join(bbcl.OUT, 'g4'), exist_ok=True)
json.dump(res, open(os.path.join(bbcl.OUT, 'g4', 'loop_poles.json'), 'w'), indent=1)
