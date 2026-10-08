"""Do the added states 'steal' baseline poles? Checks on a probe checkpoint (u_probe / f_eeoe format,
zero-init RowSplitNet model), at the encoder states of 72 fixed eval windows. Diagnostic only.
  1. Full one-step Jacobian eigenvalues with the learned block ON vs OFF (Static_ANN_Block out_gate 0):
     do the physical (rigid / theta) poles move, and does an added pole sit on a baseline pole?
  2. R^2 of the encoder's x_a regressed (least squares, with intercept) on the encoder's physical states:
     a high R^2 = the added states carry baseline information.
  3. Physical combinations: relative change from the freshly built (detuned) start.

Usage: python g2_steal.py CKPT_FILE   (file in outputs/, e.g. f_eeoe.pt)
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import n_phase1 as n1                                                          # noqa: E402
import u_probe as up                                                           # noqa: E402


def jac(fs, x, u, ann, off):
    gate = ann.out_gate.clone()
    if off:
        ann.out_gate = torch.zeros_like(gate)
    try:
        f = lambda xx: fs.hfn(xx, u)[1].reshape(x.shape[0], -1).sum(0)          # noqa: E731
        J = torch.autograd.functional.jacobian(f, x.clone()).permute(1, 0, 2).detach().numpy()
    finally:
        ann.out_gate = gate
    return J


def main(fname):
    os.environ['U_INIT'] = 'zero'
    cfg, data, norm, fs, ann, others = up.setup()
    phy = next(b for b in fs.hfn.connected_blocks if hasattr(b, 'free_params'))
    with torch.no_grad():
        p0 = phy.recover_combinations().clone()          # the ten combinations at the detuned start
    st = torch.load(os.path.join(HERE, 'outputs', fname), weights_only=False)
    fs.hfn.load_state_dict(st['hfn']); fs.encoder.load_state_dict(st['enc'])
    recs, na_r, nb_r = n1.windows(cfg, data, fs)
    ev = n1.batch(recs, fs, na_r, nb_r, 72, np.random.default_rng(123), cfg.dtype_pt)
    with torch.no_grad():
        x0 = fs.encoder(ev[0], ev[1])
    u0 = ev[2][:, 0]
    J_on, J_off = jac(fs, x0, u0, ann, False), jac(fs, x0, u0, ann, True)
    E_on = np.array([np.sort(np.abs(np.linalg.eigvals(j))) for j in J_on])
    E_off = np.array([np.sort(np.abs(np.linalg.eigvals(j[:6, :6]))) for j in J_off])
    Ea = np.array([np.linalg.eigvals(j[6:, 6:]) for j in J_on])
    print('\n######## steal check [%s, %d updates]' % (fname, st['n_upd']), flush=True)
    print('  physical poles, learned block OFF (6x6), median sorted |eig|: %s' % np.round(np.median(E_off, 0), 5))
    print('  full model, learned block ON (8x8),   median sorted |eig|: %s' % np.round(np.median(E_on, 0), 5))
    phys_on = np.array([np.sort(np.abs(np.linalg.eigvals(j[:6, :6]))) for j in J_on])
    print('  physical 6x6 block with learned block ON, median sorted |eig|: %s' % np.round(np.median(phys_on, 0), 5))
    print('  max |shift| of physical-block poles (ON vs OFF, per point, median): %.2e' % np.median(np.abs(phys_on - E_off).max(1)))
    # symmetry-drift observables (Tian, Chen and Ganguli, ICML 2021): a complex pair needs a negative
    # discriminant (a11 - a22)^2 + 4 a12 a21; the antisymmetric fraction measures the rotation part
    Aa = J_on[:, 6:, 6:]
    disc = (Aa[:, 0, 0] - Aa[:, 1, 1]) ** 2 + 4 * Aa[:, 0, 1] * Aa[:, 1, 0]
    anti = np.array([np.linalg.norm((a - a.T) / 2) / max(np.linalg.norm(a), 1e-300) for a in Aa])
    print('  A_aa discriminant median %.3e (min %.3e, %.0f %% negative); antisymmetric fraction median %.3f' % (
        np.median(disc), disc.min(), 100 * np.mean(disc < 0), np.median(anti)))
    print('  A_aa median:\n%s' % np.round(np.median(Aa, 0), 4))
    dom = Ea[np.arange(len(Ea)), np.abs(Ea).argmax(1)]
    near = np.array([np.min(np.abs(np.abs(d) - e)) for d, e in zip(dom, E_off)])
    print('  added dominant pole |z| median %.4f; distance to the nearest baseline |pole| median %.4f' % (
        np.median(np.abs(dom)), np.median(near)))
    X = x0.numpy()
    A = np.c_[X[:, :6], np.ones(len(X))]
    for j in (6, 7):
        coef, *_ = np.linalg.lstsq(A, X[:, j], rcond=None)
        r = X[:, j] - A @ coef
        print('  x_a[%d] on physical states: R^2 %.3f' % (j - 6, 1 - r.var() / X[:, j].var()))
    with torch.no_grad():
        rel = (phy.recover_combinations() - p0) / p0.abs()
    print('  physical combinations, change from the start (%%): %s' % np.round(100 * rel.numpy(), 3))


if __name__ == '__main__':
    main(sys.argv[1])
