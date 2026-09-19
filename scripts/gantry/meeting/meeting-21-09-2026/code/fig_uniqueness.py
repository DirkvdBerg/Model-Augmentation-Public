"""Uniqueness, separation and the recovery CONDITION. The figures Maarten's objection asks for.

    python fig_uniqueness.py

WHY THESE EXIST. `meeting-18-09-2026/figures/OBC/obc_fig1_fit_vs_parameters` plots distance to the
true physical parameters and calls it parameter recovery. Gyorok et al. (2026, arXiv:2511.01321)
do not claim that unconditionally, and the paper is explicit about why.

  * Recovery of the true baseline parameters is Theorem 7, and it holds only under CONDITION 4,
        sum_i phi^T(x_i) delta(x_i) = 0,
    i.e. the baseline regressor must be orthogonal to the UNMODELLED dynamics on the data. That
    is an experiment-design property, not something the parametrization provides.
  * Sect. 4.2, when Condition 4 fails: "the true baseline parameters can not be recovered, but
    the acquired theta_b value is unique and the estimation error is given by (21)", with
        ||theta_b* - theta_b_hat|| = ||(Phi^T Phi)^-1 Phi^T Delta||.

So the deliverable is UNIQUENESS plus a computable error, and recovery only as a conditional
extra. Two further reasons it cannot be claimed here: Assumption 1 needs rank(Phi) = n_theta and
our J has rank 10 of its columns, and Gyorok has measured states with no encoder, so Theorem 7
does not transfer to a SUBNET-style rollout at all.

WHAT IS DRAWN. Nothing in `meeting-18-09-2026/` is touched; these are new files.

  fig_condition4     rho(J; delta) for the three discrepancy definitions. Condition 4 is
                     rho = 0. Whatever it reads is the honest statement of how far this dataset
                     is from a recovery guarantee. Paired with the Eq. 21 per-parameter
                     predicted bias, which is the quantity that replaces "parameter error".
  fig_separation     rho(J; F_ANN), the fraction of each arm's LEARNED component lying inside
                     the baseline's column span. This is the tractable analogue of the paper's
                     Fig. 4 (zero covariance): their ANN has 16 neurons and ours has 10871
                     parameters, so the full asymptotic covariance matrix is not a figure, but
                     the cross-block vanishing is exactly this overlap going to zero.

EVERYTHING NUMERICAL IS EXISTING, QUALIFIED CODE. `gantry_dynamic/obc_diagnostics.py` already
implements the 8-state truth one-step discrepancy and the rho table as preflight tools for
D-192; `model_augmentation/fit_systems/obc.py::OBCBasis` already implements `r_overlap` and
`coefficient`, and `coefficient` IS Eq. 21's `(Phi^T Phi)^-1 Phi^T`. This script builds the basis
at the NOMINAL (true) parameters, calls them, and plots.
"""
__project_origin__ = "added"

import json
import sys

import numpy as np
import torch

from fig_common import ARMS, ARM_ORDER, FIGDIR, INK, MUTED, sci_axis, style
from build import build_arm

MA_FRAC, ZETA_A = 0.50, 0.03      # the dataset's plant: augmentation_ma50_..._z03


def _basis_at_nominal(fit_sys, cfg, norm):
    """Reference set, float64 baseline block at the NOMINAL parameters, and the basis J.

    At NOMINAL, not at the trained or detuned values, because Gyorok's Delta is defined against
    the physically true theta_b* (Eq. 14-15): Y = Phi theta_b* + Delta + E. Evaluating the
    baseline step at anything else folds parameter error into Delta and the discrepancy stops
    being the unmodelled dynamics.
    """
    from gantry_dynamic.obc_gantry import (COMBO_NAMES, build_reference_set, make_basis_builder,
                                           make_float64_block, make_reference_field)
    from model_augmentation.fit_systems.blocks import (Reduced_Gantry_State_Block,
                                                       Parameterized_Gantry_State_Block)
    from model_augmentation.fit_systems.obc import OBCBasis, evaluate_step_rows
    from model_augmentation.systems import gantry_ss as gss

    phy = next(b for b in fit_sys.hfn.connected_blocks
               if isinstance(b, (Reduced_Gantry_State_Block, Parameterized_Gantry_State_Block)))
    blk64 = make_float64_block(phy)
    ref = build_reference_set(cfg, norm, cfg.nx_ann, verbose=True)

    # The expansion point must be a FREE coordinate, not a combination vector. `free_params` is
    # a ten-vector that is ZERO at the block's `combo_init`, and the combinations are recovered
    # as `combo_init * exp(free)` for nine of them and `combo_init * (1 + free)` for `m_diff`
    # (blocks.py:1358-1362). Passing raw combinations here overflows `exp` and fills J with
    # non-finite values, which is exactly how this was caught.
    #
    # `combo_init` is the DETUNED point for these runs (COMBO_INIT_DETUNE), so the free
    # coordinate of the NOMINAL parameters is the inverse detune, not zero.
    nominal_raw = torch.stack([getattr(gss, n) for n in
                               Parameterized_Gantry_State_Block.PARAM_NAMES]).double()
    nominal = Reduced_Gantry_State_Block.combos_of(nominal_raw, gss.Lb).double()
    combo_init = blk64.combo_init.double()
    mdiff = blk64._m_diff_mask
    ratio = nominal / combo_init
    vbar = torch.where(mdiff, ratio - 1.0, torch.log(ratio)).double()
    back = torch.where(mdiff, combo_init * (1.0 + vbar), combo_init * torch.exp(vbar))
    err = float((back - nominal).abs().max() / nominal.abs().max())
    if not np.isfinite(err) or err > 1e-12:
        raise RuntimeError('free-coordinate inversion is wrong: max rel error %.3e' % err)
    print('[basis] expansion point = NOMINAL parameters, reached as free coordinate '
          '(max |free| = %.4f, inversion error %.1e)' % (float(vbar.abs().max()), err))

    row_ix = list(range(cfg.nx_phys))
    builder = make_basis_builder(blk64, ref, row_ix, chunk=cfg.obc_ref_chunk,
                                 device=torch.device('cpu'), space='tangent')
    J, secs = builder(vbar)
    if not torch.isfinite(J).all():
        raise RuntimeError('J has %d non-finite entries; the expansion point is wrong'
                           % int((~torch.isfinite(J)).sum()))
    basis = OBCBasis.from_matrix(J, vbar)
    print('[basis] J %s built in %.1f s at NOMINAL parameters; rank %d of %d columns'
          % (tuple(J.shape), secs, basis.n_cols if hasattr(basis, 'n_cols') else -1, J.shape[1]))

    base_next = evaluate_step_rows(lambda v, z: blk64.transition_from_free(v, z),
                                   vbar, ref.Z_step, row_ix,
                                   chunk=cfg.obc_ref_chunk, device=torch.device('cpu'))
    base_next_n = base_next.reshape(ref.n, len(row_ix))
    return ref, blk64, vbar, J, basis, base_next_n, row_ix, list(COMBO_NAMES)


def main():
    plt = style()
    FIGDIR.mkdir(parents=True, exist_ok=True)

    # The basis and the discrepancy depend ONLY on the physical block and the reference set, so
    # one arm is enough to build them. `noproj` is used because it is the arm whose replay has
    # been verified against its own bestfit (rel 1.3e-05).
    fit_sys, data, norm, cfg = build_arm('noproj', verbose=False)
    ref, blk64, vbar, J, basis, base_next_n, row_ix, names = _basis_at_nominal(fit_sys, cfg, norm)

    from gantry_dynamic.obc_diagnostics import discrepancy_fields, rho_table
    from gantry_dynamic.obc_gantry import make_reference_field
    from model_augmentation.fit_systems.blocks import Static_ANN_Block

    deltas = discrepancy_fields(ref, base_next_n, norm, cfg.up_sample, MA_FRAC, ZETA_A, row_ix)
    Gamma = base_next_n.reshape(-1)          # the affine column, for the [J|Gamma] variant
    tab = rho_table(J, Gamma, deltas)

    print('\n== CONDITION 4: rho(J; delta) -- recovery needs 0 ==')
    for k, v in tab['rho'].items():
        print('  %-9s rho(J) = %.4f   rho(J|Gamma) = %.4f   ||delta|| = %.4e'
              % (k, v['J'], v['J|Gamma'], v['delta_norm']))
    print('  rank(J) = %d, rank([J|Gamma]) = %d' % (tab['rank_J'], tab['rank_JG']))

    # Eq. 21: the parameter bias this discrepancy PREDICTS, per identifiable combination.
    bias = {k: basis.coefficient(d.to(torch.float64)).cpu().numpy() for k, d in deltas.items()}
    # Eq. 21 returns a bias in the FREE coordinate, which is log-scaled for nine of the ten
    # combinations, so a free-coordinate step IS very nearly a relative change. Reporting it
    # as a percentage of nominal is therefore the natural reading, and exact for m_diff.
    rel = {k: 100.0 * b for k, b in bias.items()}
    print('\n== Eq. 21: predicted parameter bias from the unmodelled term, %% of nominal ==')
    for i, n in enumerate(names):
        print('  %-9s recorded %+8.3f %%   zero %+8.3f %%   data %+8.3f %%'
              % (n, rel['recorded'][i], rel['zero'][i], rel['data'][i]))

    # Each arm's learned component on the SAME reference set, and its overlap with the baseline.
    field = make_reference_field(ref, list(range(cfg.nx_phys)))
    overlap = {}
    for arm in ARM_ORDER:
        if arm == 'baseline':
            continue
        fs, _d, _n, _c = build_arm(arm, verbose=False)
        ann = next(b for b in fs.hfn.connected_blocks if isinstance(b, Static_ANN_Block))
        with torch.no_grad():
            F = field(ann).detach().double().cpu()
        overlap[arm] = dict(rho=basis.r_overlap(F), rperp=basis.r_perp(F),
                            norm=float(torch.linalg.vector_norm(F)))
        print('[arm] %-9s rho(J; F_ANN) = %.6f   r_perp = %.3e   ||F_ANN|| = %.4e'
              % (arm, overlap[arm]['rho'], overlap[arm]['rperp'], overlap[arm]['norm']))
        del fs

    _fig_condition4(plt, tab, rel, names)
    _fig_separation(plt, overlap)

    out = FIGDIR / 'uniqueness_data.json'
    out.write_text(json.dumps(
        {'rho_table': tab, 'eq21_bias_pct': {k: v.tolist() for k, v in rel.items()},
         'combo_names': names, 'overlap': overlap,
         'ma_frac': MA_FRAC, 'zeta_a': ZETA_A}, indent=1), encoding='utf-8')
    print('\nwrote %s' % out)
    return 0


def _fig_condition4(plt, tab, rel, names):
    """Left: how badly Condition 4 fails. Right: the bias Eq. 21 predicts because it fails."""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.6, 3.4),
                                   gridspec_kw={'width_ratios': [1, 2.2]})

    keys = ['recorded', 'zero', 'data']
    lbl = {'recorded': 'truth at recorded $x_a$', 'zero': 'truth at $x_a=0$',
           'data': 'recorded next state'}
    x = np.arange(len(keys))
    axl.bar(x - 0.2, [tab['rho'][k]['J'] for k in keys], 0.4, color='#0072B2', label=r'$J$')
    axl.bar(x + 0.2, [tab['rho'][k]['J|Gamma'] for k in keys], 0.4, color='#009E73',
            label=r'$[J\,|\,\Gamma]$')
    axl.axhline(0.0, color=INK, lw=1.0)
    axl.set_xticks(x); axl.set_xticklabels([lbl[k] for k in keys], rotation=20, ha='right')
    axl.set_ylabel(r'$\rho(B;\delta)=\|BB^{+}\delta\|/\|\delta\|$')
    axl.set_title('Condition 4 needs $\\rho=0$', loc='left')
    axl.legend(frameon=False, fontsize=7)

    xs = np.arange(len(names))
    axr.bar(xs - 0.25, rel['recorded'], 0.25, color='#0072B2', label='truth at recorded $x_a$')
    axr.bar(xs,        rel['zero'],     0.25, color='#009E73', label='truth at $x_a=0$')
    axr.bar(xs + 0.25, rel['data'],     0.25, color='#CC79A7', label='recorded next state')
    axr.axhline(0.0, color=INK, lw=1.0)
    axr.set_xticks(xs); axr.set_xticklabels(names, rotation=35, ha='right')
    axr.set_ylabel('predicted bias [% of nominal]')
    axr.set_title('Eq. 21: the parameter error the unmodelled term forces', loc='left')
    axr.legend(frameon=False, fontsize=7, ncol=3)

    fig.suptitle('Recovery is CONDITIONAL: orthogonality gives uniqueness, Condition 4 gives '
                 'recovery', fontsize=9.5, y=1.04)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(FIGDIR / ('uniqueness_condition4.%s' % ext))
    print('  wrote %s' % (FIGDIR / 'uniqueness_condition4.{png,pdf}'))
    plt.close(fig)


def _fig_separation(plt, overlap):
    """rho(J; F_ANN): how much of the learned component lies in the baseline's span."""
    arms = [a for a in ARM_ORDER if a in overlap]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    vals = [overlap[a]['rho'] for a in arms]
    ax.bar(np.arange(len(arms)), vals, 0.55, color=[ARMS[a]['c'] for a in arms])
    for i, v in enumerate(vals):
        ax.annotate('%.2e' % v, (i, v), textcoords='offset points', xytext=(0, 3),
                    ha='center', fontsize=7, color=MUTED)
    ax.set_xticks(np.arange(len(arms)))
    ax.set_xticklabels([ARMS[a]['label'] for a in arms], rotation=15, ha='right')
    ax.set_yscale('log')
    ax.set_ylabel(r'$\rho(J;F_{\mathrm{ANN}})$')
    ax.set_title('Overlap of the learned component with the baseline span\n'
                 '(the paper\'s zero-covariance claim, measured)', loc='left', fontsize=8.5)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(FIGDIR / ('uniqueness_separation.%s' % ext))
    print('  wrote %s' % (FIGDIR / 'uniqueness_separation.{png,pdf}'))
    plt.close(fig)


if __name__ == '__main__':
    sys.exit(main())
