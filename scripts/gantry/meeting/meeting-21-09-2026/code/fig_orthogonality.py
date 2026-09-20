"""Is the learned component orthogonal to the baseline? Measured, per arm, as actually applied.

    python fig_orthogonality.py

THE CLAIM THIS FIGURE IS FOR: "the ANN output is orthogonal to the baseline, and without the
projection it is not."

WHAT "ORTHOGONAL TO THE BASELINE" HAS TO MEAN. Two different objects, and the arms separate on
the difference between them:

  J       (10 col)  the baseline's PARAMETER DIRECTIONS, d f_base/d theta on the reference set.
                    Overlap here means the ANN is doing something a change of physical parameters
                    could have done instead. This is the negation question, and it is what the
                    TANGENT projection removes.
  Gamma   (1 col)   the baseline's OWN OUTPUT, f_base(x, u): literally the trained model with the
                    ANN silenced. rho against a single column is |cos| between the two fields.
  [J|Gamma] (11)    both. In a nonlinear-in-parameters baseline Gamma is NOT inside span(J),
                    which is exactly why the AFFINE variant appends it as an eleventh column.

So the expected reading, and the reason to run two OBC variants at all:

  no projection   high against all three
  OBC tangent     near zero against J, NOT against Gamma
  OBC affine      near zero against all three

WHAT IS MEASURED IS THE APPLIED FIELD, NOT THE RAW ANN. An earlier version of this measurement
(in `fig_uniqueness.py`) reported rho(J; F_ANN) of 0.86 / 0.92 / 0.94 and those OBC numbers were
meaningless. In this implementation the projection is not a penalty and it is not baked into the
ANN weights: it is a correction SUBTRACTED inside the state transition,
`interconnect.py:168-170`, `xp = xp - obc_correction(x, u)`. So the component that actually
reaches the model is

    F_applied = F_ANN - ( J @ theta[:10]  +  theta[10] * Gamma  if affine )

with `theta = obc_correction.theta_frozen` and the expansion point `obc_correction.vbar`, both
stored as buffers in the checkpoint. The no-projection arm carries no correction, so for it
F_applied IS F_ANN.

EACH ARM IS MEASURED AT ITS OWN EXPANSION POINT, because that is where its own correction was
computed. The question being answered is "is THIS model's learned component orthogonal to ITS
OWN baseline", which is the claim the thesis makes. A common basis would answer a different
question and would flatter whichever arm happened to sit closest to it.
"""
__project_origin__ = "added"

import json
import sys

import numpy as np
import torch

from fig_common import ARMS, ARM_ORDER, FIGDIR, INK, MUTED, style
from build import build_arm

SPANS = ['J', 'Gamma', 'J|Gamma']
SPAN_LABEL = {'J': r'$J$  (parameter directions)',
              'Gamma': r'$\Gamma$  (baseline output)',
              'J|Gamma': r'$[J\,|\,\Gamma]$  (both)'}
SPAN_COLOR = {'J': '#0072B2', 'Gamma': '#009E73', 'J|Gamma': '#CC79A7'}


def _rho(B, F, eps=1e-300):
    """||B B^+ F|| / ||F||: the in-span fraction of F. B is (n, k), F is (n,). float64."""
    U, S, _ = torch.linalg.svd(B, full_matrices=False)
    tol = S.max() * max(B.shape) * torch.finfo(torch.float64).eps
    r = int((S > tol).sum())
    proj = U[:, :r] @ (U[:, :r].T @ F)
    return float(torch.linalg.vector_norm(proj) / (torch.linalg.vector_norm(F) + eps))


def _arm_fields(arm, ref_holder):
    """(F_applied, J, Gamma) for one arm, all float64, at THAT arm's expansion point."""
    from gantry_dynamic.obc_gantry import (build_reference_set, make_basis_builder,
                                           make_float64_block, make_reference_field)
    from model_augmentation.fit_systems.blocks import (Reduced_Gantry_State_Block,
                                                       Parameterized_Gantry_State_Block,
                                                       Static_ANN_Block)
    from model_augmentation.fit_systems.obc import evaluate_step_rows

    fit_sys, _data, norm, cfg = build_arm(arm, verbose=False)
    phy = next(b for b in fit_sys.hfn.connected_blocks
               if isinstance(b, (Reduced_Gantry_State_Block, Parameterized_Gantry_State_Block)))
    ann = next(b for b in fit_sys.hfn.connected_blocks if isinstance(b, Static_ANN_Block))
    corr = getattr(fit_sys.hfn, 'obc_correction', None)

    if ref_holder.get('ref') is None:
        # One reference set for every arm: it depends only on cfg and norm, which are identical
        # across the three runs (same MODE, same seed, same normalisation).
        ref_holder['ref'] = build_reference_set(cfg, norm, cfg.nx_ann, verbose=True)
    ref = ref_holder['ref']

    blk64 = make_float64_block(phy)
    if corr is not None:
        vbar = corr.vbar.detach().double().cpu()
        theta = corr.theta_frozen.detach().double().cpu()
        space = corr.space
    else:
        vbar = phy.free_params.detach().double().cpu()
        theta, space = None, None

    row_ix = list(range(cfg.nx_phys))
    J, _s = make_basis_builder(blk64, ref, row_ix, chunk=cfg.obc_ref_chunk,
                               device=torch.device('cpu'), space='tangent')(vbar)
    J = J.double()
    Gamma = evaluate_step_rows(lambda v, z: blk64.transition_from_free(v, z),
                               vbar, ref.Z_step, row_ix,
                               chunk=cfg.obc_ref_chunk, device=torch.device('cpu')).double()

    with torch.no_grad():
        F = make_reference_field(ref, row_ix)(ann).detach().double().cpu()

    if theta is None:
        F_applied, note = F, 'no correction attached'
    elif space == 'tangent':
        F_applied = F - J @ theta
        note = 'F_ANN - J theta'
    else:
        F_applied = F - (J @ theta[:J.shape[1]] + theta[-1] * Gamma)
        note = 'F_ANN - (J theta + alpha Gamma)'
    print('[arm] %-9s space=%-7s applied = %s   ||F_ANN|| = %.4e  ||F_applied|| = %.4e'
          % (arm, space or '-', note, float(torch.linalg.vector_norm(F)),
             float(torch.linalg.vector_norm(F_applied))))
    del fit_sys
    return F, F_applied, J, Gamma


def main():
    plt = style()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    arms = [a for a in ARM_ORDER if a != 'baseline']
    ref_holder, out = {}, {}

    for arm in arms:
        F_raw, F_app, J, Gamma = _arm_fields(arm, ref_holder)
        JG = torch.cat([J, Gamma[:, None]], dim=1)
        rec = {}
        for span, B in (('J', J), ('Gamma', Gamma[:, None]), ('J|Gamma', JG)):
            rec[span] = dict(applied=_rho(B, F_app), raw=_rho(B, F_raw))
        rec['norm_raw'] = float(torch.linalg.vector_norm(F_raw))
        rec['norm_applied'] = float(torch.linalg.vector_norm(F_app))
        out[arm] = rec
        print('       rho(applied):  J %.3e   Gamma %.3e   [J|Gamma] %.3e'
              % tuple(rec[s]['applied'] for s in SPANS))
        print('       rho(raw ANN):  J %.3e   Gamma %.3e   [J|Gamma] %.3e'
              % tuple(rec[s]['raw'] for s in SPANS))
        del J, Gamma, JG, F_raw, F_app

    _draw(plt, out, arms)
    p = FIGDIR / 'orthogonality_data.json'
    p.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print('\nwrote %s' % p)
    return 0


def _draw(plt, out, arms):
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    x = np.arange(len(arms))
    w = 0.26
    floor = 1e-17
    for i, span in enumerate(SPANS):
        v = [max(out[a][span]['applied'], floor) for a in arms]
        ax.bar(x + (i - 1) * w, v, w, color=SPAN_COLOR[span], label=SPAN_LABEL[span])
        for xi, vi in zip(x + (i - 1) * w, v):
            ax.annotate('%.1e' % vi, (xi, vi), textcoords='offset points', xytext=(0, 2),
                        ha='center', fontsize=6, rotation=90, color=MUTED)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([ARMS[a]['label'] for a in arms])
    ax.set_ylabel(r'$\rho(B;\,F_{\mathrm{applied}})=\|BB^{+}F\|/\|F\|$')
    ax.axhline(1.0, color=INK, lw=0.8, ls=':')
    ax.legend(frameon=False, fontsize=7, loc='lower left')
    ax.set_title('Overlap of the LEARNED component with the baseline, as applied in the loop\n'
                 'each arm at its own expansion point; 1.0 = entirely explainable by the '
                 'baseline, 0 = orthogonal', loc='left', fontsize=8.5)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(FIGDIR / ('orthogonality_applied.%s' % ext))
    print('  wrote %s' % (FIGDIR / 'orthogonality_applied.{png,pdf}'))
    plt.close(fig)


if __name__ == '__main__':
    sys.exit(main())
