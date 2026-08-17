"""B step 2 and C step 6, the parts that need no training: is there a gradient to follow?

B's oracle learnability test asks whether the closed-loop loss can see the thing the ANN is
supposed to learn. Here the quantity to be learned is the absorber, so the test is run by
DETUNING it: replace the truth's absorber frequency f_a = 150 Hz by f_a' and sweep. At f_a' = f_a
the model is the truth and the loss must sit at the numerical floor; away from it the loss must
rise monotonically and smoothly. If it does not, no optimiser will find the absorber in this
configuration and B is dead before any training starts.

Three losses are evaluated on the same sweep, which is what makes this also the evaluation half
of C:

  L_ol      || y_model - y_data ||                      open loop, what training uses today
  L_so      || So (y_model - y_data) ||                 open loop, weighted (option C)
  L_cl      || y_model - y_data ||, model in the loop   closed loop (option B)

Reading:
  a floor at the correct f_a and monotone growth away from it  -> S-B1 satisfiable
  L_so and L_cl tracking each other                            -> C is a cheap proxy for B
  L_so or L_cl markedly sharper than L_ol near the minimum     -> the loop weighting helps
"""
__project_origin__ = "added"

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import closed_loop as CL
import plant as PL
import so_filter as SOF

# Detuning sweep. THEORY: f_a = 150 Hz is the truth's absorber frequency (gtd_config.m:50).
FA_TRUE = 150.0
FA_GRID = np.array([120., 130., 140., 145., 148., 150., 152., 155., 160., 170., 185.])
T_SKIP = 0.5
CH = ['X1', 'X2', 'Y']
RECORD = 'V1_standstill_Yp10'      # standstill, so the absorber is the dominant discrepancy


def deriv8_detuned(fa):
    """PL.deriv8 with the absorber stiffness and damping rebuilt at a different f_a.

    Rebuilds exactly as plant.py does: KA = MA (2 pi fa)^2 and CA = 2 zeta sqrt(KA MA).
    Only the absorber changes; the rest of the plant is untouched.
    """
    MA, MHR, L0, ZETA = PL.MA, PL.MHR, PL.L0, PL.ZETA_A
    KA = MA * (2 * np.pi * fa) ** 2
    CA = 2 * ZETA * np.sqrt(KA * MA)
    C4 = PL._C4.copy(); C4[3, 3] = CA
    K4 = PL._K4.copy(); K4[3, 3] = KA
    E43 = PL._E43

    def f(x, u):
        q, qd = x[:4], x[4:]
        M = PL.M8(x[2], x[3], L0, False)
        qdd = np.linalg.solve(M, E43 @ u - K4 @ q - C4 @ qd)
        return np.concatenate([qd, qdd])
    return f


def main():
    rec = CL.load_record(RECORD)
    Ac, Bc, Cc, Dc, Y_op = CL.load_controller(RECORD)
    ctrl = (Ac, Bc, Cc, Dc)
    So = SOF.so_ss(Y_op, ctrl)
    t = np.arange(len(rec['r'])) * CL.TS
    m = t >= T_SKIP
    x0 = CL.x0_for('truth', Y_op)

    print('%s  Y_op %.2f, absorber detuning sweep about f_a = %g Hz' % (RECORD, Y_op, FA_TRUE))
    print('%8s %14s %14s %14s' % ('fa [Hz]', 'L_ol', 'L_so (C)', 'L_cl (B)'))

    rows = []
    for fa in FA_GRID:
        f8 = deriv8_detuned(fa)

        # open loop, driven by the recorded force
        y_ol = np.empty((len(rec['u_total']), 3))
        x = x0.copy()
        Pt = PL.P_np.T
        for k in range(len(rec['u_total'])):
            y_ol[k] = Pt @ x[:3]
            ul = PL.P_np @ rec['u_total'][k]
            k1 = f8(x, ul); k2 = f8(x + .5 * CL.TS * k1, ul)
            k3 = f8(x + .5 * CL.TS * k2, ul); k4 = f8(x + CL.TS * k3, ul)
            x = x + (CL.TS / 6.) * (k1 + 2 * k2 + 2 * k3 + k4)
        e_ol = y_ol - rec['y']

        # closed loop, driven by the reference
        y_cl, _, _ = CL.simulate(f8, x0, rec['r'], rec['f_ms'], ctrl)
        e_cl = y_cl - rec['y']

        # the C weighting applied to the open-loop residual
        e_so = SOF.apply_ss(So, e_ol)

        L_ol = np.sqrt(np.mean(e_ol[m] ** 2))
        L_so = np.sqrt(np.mean(e_so[m] ** 2))
        L_cl = np.sqrt(np.mean(e_cl[m] ** 2))
        rows.append((fa, L_ol, L_so, L_cl))
        print('%8.1f %14.6e %14.6e %14.6e' % (fa, L_ol, L_so, L_cl))

    R = np.array(rows)
    fa, Lol, Lso, Lcl = R[:, 0], R[:, 1], R[:, 2], R[:, 3]
    i0 = int(np.argmin(np.abs(fa - FA_TRUE)))

    print('\nS-B1 checks')
    for lab, L in (('L_ol', Lol), ('L_so', Lso), ('L_cl', Lcl)):
        at_min = int(np.argmin(L))
        floor_ok = at_min == i0
        left_mono = bool(np.all(np.diff(L[:i0 + 1]) < 0))
        right_mono = bool(np.all(np.diff(L[i0:]) > 0))
        contrast = L[-1] / L[i0] if L[i0] > 0 else np.inf
        print('  %-5s minimum at %6.1f Hz %s   monotone left %s right %s   contrast %.2e'
              % (lab, fa[at_min], 'PASS' if floor_ok else 'FAIL',
                 'PASS' if left_mono else 'FAIL', 'PASS' if right_mono else 'FAIL', contrast))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    for lab, L, c in (('L_ol open loop', Lol, '#333333'),
                      ('L_so weighted (C)', Lso, '#0072B2'),
                      ('L_cl closed loop (B)', Lcl, '#D55E00')):
        ax.semilogy(fa, L, 'o-', color=c, lw=1.2, ms=4, label=lab)
    ax.axvline(FA_TRUE, color='#999999', lw=1.0, ls=':')
    ax.set_xlabel('absorber frequency in the model [Hz]')
    ax.set_ylabel('loss')
    ax.set_title('Absolute loss. A floor at 150 Hz means the loss\ncan see the absorber.',
                 fontsize=10)
    ax.grid(alpha=0.25, lw=0.5, which='both')
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    for lab, L, c in (('L_ol', Lol, '#333333'), ('L_so', Lso, '#0072B2'),
                      ('L_cl', Lcl, '#D55E00')):
        ax.semilogy(fa, L / L[i0], 'o-', color=c, lw=1.2, ms=4, label=lab)
    ax.axvline(FA_TRUE, color='#999999', lw=1.0, ls=':')
    ax.set_xlabel('absorber frequency in the model [Hz]')
    ax.set_ylabel('loss / loss at 150 Hz')
    ax.set_title('Normalised. Steeper is a stronger gradient\nfor the optimiser to follow.',
                 fontsize=10)
    ax.grid(alpha=0.25, lw=0.5, which='both')
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle('Learnability of the absorber under the three losses, %s' % RECORD, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(CL.HERE, 'figures')
    os.makedirs(out, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out, 'bc_learnability.%s' % ext), dpi=160, bbox_inches='tight')
    print('\nwrote %s' % os.path.join(out, 'bc_learnability.png'))
    np.save(os.path.join(CL.HERE, 'bc_learnability.npy'), R)


if __name__ == '__main__':
    main()
