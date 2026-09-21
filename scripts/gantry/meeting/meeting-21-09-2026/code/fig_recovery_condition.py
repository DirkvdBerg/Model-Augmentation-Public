"""The geometric picture behind Maarten's objection: uniqueness is not recovery.

    python fig_recovery_condition.py

WHAT IT SHOWS. `fig_uniqueness.py` already reports the NUMBERS (rho, and the Eq. 21 per-parameter
bias). This file draws the one-picture reason those numbers matter, because the objection is
geometric and a table does not carry it.

  (a) Condition 4 HOLDS. The unmodelled dynamics are orthogonal to the baseline's parameter
      sensitivities, so nothing of `Delta` lies along a direction the parameters could have
      produced, the projection leaves them alone, and Theorem 7 gives exact recovery.
  (b) Condition 4 FAILS, which is our dataset. `Delta` has a component inside
      `range(Phi_vbar)`. The projection FORBIDS the learning component from representing that
      component, the data still has to be fitted, so the PARAMETERS absorb it and move away
      from their true values by exactly `Phi^+ Delta`.

THE POINT THE PICTURE HAS TO MAKE, and it is the one that is easy to get backwards: the split is
unique either way. Uniqueness is a property of the problem we pose. Correctness is a property of
the plant we pose it about, and it has to be engineered into the plant.

THE ANGLE IS DRAWN TO SCALE. `rho_Phi = ||Phi Phi^+ Delta|| / ||Delta||` is the cosine of the
angle between `Delta` and the subspace, so panel (b) is drawn at the MEASURED value and looks
almost perpendicular. That is honest and it is half the message: a 21 percent geometric overlap
produces a 105 percent error on `m_diff`, because `Phi` is ill conditioned and `Phi^+` amplifies
whatever lands in its weakly excited directions. The caption says so rather than letting the
picture imply the error is small.

NUMBERS come from `figures/uniqueness_data.json`, written by `fig_uniqueness.py`, so this figure
cannot drift from the one next to it in the deck. Nothing here recomputes anything.
"""
__project_origin__ = "added"

import json

import numpy as np

from fig_common import FIGDIR, INK, MUTED, style

# Okabe-Ito, the same assignment the settling figures use, so the deck reads as one set.
C_SPAN = '#777777'      # the subspace the parameters can reach
C_DELTA = '#7A3E9D'     # the true unmodelled dynamics (the data colour of this deck)
C_BIAS = '#D55E00'      # the component the PARAMETERS are forced to absorb
C_LEARN = '#009E73'     # the component the learning component legitimately keeps

VARIANT = 'data'        # the one-step discrepancy built from the recorded next state

# The deck writes the ten combinations as in `meeting-18-09-2026/slides/obc-slide-panels.tex`,
# where `m_diff` is `m_{\Delta}`. Keeping the slide notation here means the caption and the
# equations on the neighbouring panels use the same symbol for the same thing.
NAME_TEX = {'kb_sum': r'k_{b,\mathrm{sum}}', 'cg1': r'c_{g1}', 'cg2': r'c_{g2}',
            'cy': r'c_{y}', 'cb_sum': r'c_{b,\mathrm{sum}}', 'mh': r'm_{h}',
            'm_total': r'm_{\mathrm{tot}}', 'm_diff': r'm_{\Delta}',
            'J_eff': r'J_{\mathrm{eff}}', 'd': r'd'}


def _right_angle(ax, x, y, ux, uy, size=0.075):
    """A square corner mark at (x, y) opening along the two unit directions given."""
    vx, vy = -uy, ux
    ax.plot([x + size * ux, x + size * (ux + vx), x + size * vx],
            [y + size * uy, y + size * (uy + vy), y + size * vy],
            color=MUTED, lw=0.9, solid_joinstyle='miter', zorder=3)


def _panel(ax, rho, title, subtitle):
    """One geometry panel. `rho` is the cosine of the angle between Delta and the subspace.

    LABELLING RULE for this figure: every arrow carries its SYMBOL plus at most one word, and
    all prose lives in the caption. The words were what made the first draft busy, and the one
    label that genuinely earns its space is the identity on the in-span component, which is the
    whole argument in a single line.
    """
    L = 1.30
    ax.axhspan(-0.05, 0.05, xmin=0.03, xmax=0.97, color=C_SPAN, alpha=0.13, lw=0)
    ax.plot([-L, L], [0, 0], color=C_SPAN, lw=1.6, solid_capstyle='round', zorder=2)
    ax.text(L, 0.09, r'$\mathrm{range}\,(\Phi_{\bar v})$', color=C_SPAN,
            ha='right', va='bottom', fontsize=9)

    r = 1.02
    dx, dy = r * rho, r * np.sqrt(max(1.0 - rho ** 2, 0.0))

    ax.annotate('', xy=(dx, dy), xytext=(0, 0),
                arrowprops=dict(arrowstyle='-|>', color=C_DELTA, lw=2.1,
                                shrinkA=0, shrinkB=0, mutation_scale=13), zorder=5)
    # The ANN component. In panel (a) it IS Delta, so the two are labelled as one identity
    # rather than stacked on top of each other.
    ax.annotate('', xy=(dx, dy), xytext=(dx, 0),
                arrowprops=dict(arrowstyle='-|>', color=C_LEARN, lw=1.6,
                                shrinkA=0, shrinkB=0, mutation_scale=11), zorder=4)

    if rho > 1e-6:
        ax.annotate('', xy=(dx, 0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='-|>', color=C_BIAS, lw=2.6,
                                    shrinkA=0, shrinkB=0, mutation_scale=12), zorder=6)
        _right_angle(ax, dx, 0.0, 0.0, 1.0)
        ax.text(-0.02, -0.10, r'$\Phi\Phi^{+}\Delta=\Phi\,(\hat v-v^{*})$',
                color=C_BIAS, fontsize=9.5, ha='left', va='top')
        ax.text(-0.02, -0.255, 'forbidden to the ANN\n$\\Rightarrow$ parameters absorb it',
                color=C_BIAS, fontsize=8, ha='left', va='top', linespacing=1.3)
        ax.text(dx - 0.09, dy * 0.60, r'$\Delta$', color=C_DELTA, fontsize=12,
                ha='right', va='center')
        ax.text(dx + 0.07, dy * 0.58, r'$(I-\Phi\Phi^{+})\Delta$', color=C_LEARN,
                fontsize=9.5, ha='left', va='center')
        ax.text(dx + 0.07, dy * 0.58 - 0.135, 'ANN', color=C_LEARN, fontsize=8,
                ha='left', va='center')
        # The angle, stated rather than eyeballed: at true scale this reads as almost
        # perpendicular, and a viewer should see the number instead of guessing it.
        from matplotlib.patches import Arc
        ang = float(np.degrees(np.arctan2(dy, dx)))
        ax.add_patch(Arc((0, 0), 0.40, 0.40, theta1=0.0, theta2=ang,
                         color=MUTED, lw=0.9, zorder=3))
        ax.text(-0.045, 0.235, r'$%.0f^\circ$' % ang, color=MUTED, fontsize=8,
                ha='right', va='center')
    else:
        # Delta and the ANN component are the SAME arrow here, so they are labelled on opposite
        # sides of it. That also defines both symbols once, on the left panel, and lets panel
        # (b) carry the bare symbols.
        _right_angle(ax, 0.0, 0.0, 1.0, 0.0)
        ax.text(-0.075, dy * 0.60, r'$\Delta$', color=C_DELTA, fontsize=12,
                ha='right', va='center')
        ax.text(-0.075, dy * 0.60 - 0.145, 'true unmodelled\ndynamics', color=C_DELTA,
                fontsize=8, ha='right', va='top', linespacing=1.3)
        ax.text(0.085, dy * 0.58, r'$(I-\Phi\Phi^{+})\Delta$', color=C_LEARN,
                fontsize=9.5, ha='left', va='center')
        ax.text(0.085, dy * 0.58 - 0.135, 'ANN', color=C_LEARN, fontsize=8,
                ha='left', va='center')
        ax.text(-0.02, -0.10, r'$\Phi\Phi^{+}\Delta=0$', color=C_BIAS, fontsize=9.5,
                ha='left', va='top')
        ax.text(-0.02, -0.255, 'nothing for the\nparameters to absorb', color=C_BIAS,
                fontsize=8, ha='left', va='top', linespacing=1.3)

    ax.set_title(title, fontsize=9.5, color=INK, pad=13)
    ax.text(0.5, 1.005, subtitle, transform=ax.transAxes, ha='center', va='bottom',
            fontsize=8, color=MUTED)
    ax.set_xlim(-0.72, L + 0.26)
    ax.set_ylim(-0.52, 1.20)
    ax.set_aspect('equal')
    ax.axis('off')


def main():
    plt = style()
    d = json.load(open(FIGDIR / 'uniqueness_data.json'))
    rho = float(d['rho_table']['rho'][VARIANT]['J'])
    bias = np.array(d['eq21_bias_pct'][VARIANT])
    names = d['combo_names']
    j = int(np.argmax(np.abs(bias)))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    # (a) is the TARGET, not a hypothetical: it is the configuration a validation dataset has to
    # be built to satisfy. Saying so costs a word and turns the figure into a plan.
    _panel(axes[0], 0.0,
           r'(a)  target:  $\Phi_{\bar v}^{\mathsf{T}}\Delta = 0$',
           r'$\hat v = v^{*}$    exact recovery  (Thm. 7)')
    _panel(axes[1], rho,
           r'(b)  this dataset:  $\rho_{\Phi} = %.2f$' % rho,
           r'$\hat v - v^{*} = \Phi_{\bar v}^{+}\Delta \neq 0$    unique, but biased')

    fig.text(0.5, 0.045,
             r'Unique in both. Correct only in (a).   Here $\rho_{\Phi}=%.2f$ already biases '
             r'$%s$ by $%.0f\%%$ of nominal.'
             % (rho, NAME_TEX.get(names[j], names[j]), abs(bias[j])),
             ha='center', va='top', fontsize=8.6, color=INK)
    fig.text(0.5, -0.022,
             r'Schematic: $\Delta\in\R^{Nn_x}$, $\mathrm{range}(\Phi_{\bar v})$ is '
             r'$10$-dimensional.'.replace(r'\R', r'\mathbb{R}'),
             ha='center', va='top', fontsize=7, color=MUTED)

    fig.tight_layout(w_pad=2.2)
    for ext in ('pdf', 'png'):
        fig.savefig(FIGDIR / ('recovery_condition.%s' % ext), bbox_inches='tight')
    print('wrote %s' % (FIGDIR / 'recovery_condition.pdf'))
    print('  rho_Phi = %.4f (%s variant), worst parameter %s at %.1f %% of nominal'
          % (rho, VARIANT, names[j], bias[j]))


if __name__ == '__main__':
    main()
