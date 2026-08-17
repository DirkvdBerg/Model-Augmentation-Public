"""P3 and the C loss: the output sensitivity So as an explicit, applyable filter.

C weights the open-loop residual by So before taking the norm:

    current   L = || y_model - y_data ||
    option C  L = || So(q) [ y_model - y_data ] ||

Justification, controller-in-derivation.tex section 6.3: the recorded u_total already equals
Si w, so the open-loop residual carries that factor and left-multiplying by So turns it into the
closed-loop output error. Exact in SISO, approximate in MIMO.

State-space construction. With the plant Gop strictly proper (Dg = 0, baseline write-up eq. 10)
and the controller biproper (Dc != 0 from Tustin):

    xg+ = Ag xg + Bg (Cc xc + Dc e)      e = v - Cg xg
    xc+ = Ac xc + Bc e
    out = e = v - Cg xg

gives So = (I + Gop Cfb)^-1 as

    A_so = [[Ag - Bg Dc Cg,  Bg Cc],      B_so = [[Bg Dc],      C_so = [-Cg, 0]   D_so = I
            [   -Bc Cg,        Ac]]               [  Bc  ]]

15 states: 6 plant plus 9 controller. So has a ZERO at z = 1 (because Cfb has a pole there), so
its DC gain is zero and its own transient decays. That is what makes it safe to apply per window
with zero initial state, unlike the plant integrator.

The DC zero is also the substantive property: weighting by So makes the loss blind to a constant
output offset by construction, which is the formal version of "the controller would pull it to
zero".
"""
__project_origin__ = "added"

import os
import numpy as np
from scipy.signal import cont2discrete

from verify_controller import M_op, P, C_DAMP, K_STIFF, TS


def plant_ss(Y_op, ts=TS):
    """Gop: stage force -> stage position, ZOH discretised. gtd_build_plant.m:22, 30."""
    M = M_op(Y_op)
    Minv = np.linalg.inv(M)
    A = np.block([[np.zeros((3, 3)), np.eye(3)], [-Minv @ K_STIFF, -Minv @ C_DAMP]])
    B = np.vstack([np.zeros((3, 3)), Minv])
    Cm = np.hstack([np.eye(3), np.zeros((3, 3))])
    Ad, Bd, Cd, Dd, _ = cont2discrete((A, B @ P, P.T @ Cm, np.zeros((3, 3))), ts, method='zoh')
    return Ad, Bd, Cd, Dd


def so_ss(Y_op, ctrl, ts=TS):
    """So = (I + Gop Cfb)^-1 as (A, B, C, D). ctrl = (Ac, Bc, Cc, Dc)."""
    Ag, Bg, Cg, Dg = plant_ss(Y_op, ts)
    assert np.allclose(Dg, 0.0), 'plant feedthrough must be zero, else there is an algebraic loop'
    Ac, Bc, Cc, Dc = ctrl
    n, m = Ag.shape[0], Ac.shape[0]
    A = np.block([[Ag - Bg @ Dc @ Cg, Bg @ Cc],
                  [-Bc @ Cg, Ac]])
    B = np.vstack([Bg @ Dc, Bc])
    C = np.hstack([-Cg, np.zeros((3, m))])
    D = np.eye(3)
    return A, B, C, D


def apply_ss(sys, v):
    """y_k = C x_k + D v_k, x_k+1 = A x_k + B v_k, from rest."""
    A, B, C, D = sys
    x = np.zeros(A.shape[0])
    out = np.empty((len(v), C.shape[0]))
    for k in range(len(v)):
        out[k] = C @ x + D @ v[k]
        x = A @ x + B @ v[k]
    return out


def so_frf(sys, f, ts=TS):
    A, B, C, D = sys
    z = np.exp(1j * 2 * np.pi * f * ts)
    return C @ np.linalg.solve(z * np.eye(A.shape[0]) - A, B) + D


if __name__ == '__main__':
    import closed_loop as CL
    print('P3  So as a filter, checked against the sensitivity computed frequency by frequency\n')
    for name in CL.available_records():
        Ac, Bc, Cc, Dc, Y_op = CL.load_controller(name)
        S = so_ss(Y_op, (Ac, Bc, Cc, Dc))
        print('%s  Y_op %.2f, So has %d states' % (name, Y_op, S[0].shape[0]))

        # cross-check the state-space So against the direct inverse at a few frequencies
        Ag, Bg, Cg, Dg = plant_ss(Y_op)
        worst = 0.0
        for f in (1.0, 10.0, 50.0, 100.0, 150.0, 180.0, 500.0):
            z = np.exp(1j * 2 * np.pi * f * TS)
            G = Cg @ np.linalg.solve(z * np.eye(6) - Ag, Bg) + Dg
            Cz = Cc @ np.linalg.solve(z * np.eye(Ac.shape[0]) - Ac, Bc) + Dc
            direct = np.linalg.inv(np.eye(3) + G @ Cz)
            viass = so_frf(S, f)
            worst = max(worst, np.abs(direct - viass).max() / np.abs(direct).max())
        print('   state-space vs direct inverse, worst relative %.3e   %s'
              % (worst, 'PASS' if worst < 1e-10 else 'FAIL'))

        sv = [np.linalg.svd(so_frf(S, f), compute_uv=False)[0]
              for f in (1.0, 10.0, 50.0, 100.0, 150.0, 180.0, 500.0)]
        print('   sigma_max(So) [%s]' % ' '.join('%.4f' % v for v in sv))
        dc = np.linalg.svd(so_frf(S, 1e-6), compute_uv=False)[0]
        print('   sigma_max(So) at DC = %.3e   (the zero at z = 1; this is why C is offset-blind)\n'
              % dc)
