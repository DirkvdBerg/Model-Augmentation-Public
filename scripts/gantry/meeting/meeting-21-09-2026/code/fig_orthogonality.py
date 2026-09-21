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
# TRAIN is where the projection is constructed; VALIDATION is held out from it. Reported side by
# side, never averaged together.
SPLITS = ['train', 'val']
# Notation follows the 2026-09-18 deck so the slide and the figure cannot disagree: Phi_vbar is
# the stacked baseline Jacobian, f_base its own output, [Phi | f_base] the extended space. (That
# deck writes the affine column as Gamma_vbar = col_k[f_base - Phi_vbar vbar]; it differs from
# f_base by a column already inside range(Phi_vbar), so the SPAN and hence the projection are
# identical, and only the single-column cosine differs.)
SPAN_LABEL = {'J': r'$\Phi_{\bar v}$  (parameter directions)',
              'Gamma': r'$f_{\mathrm{base}}$  (baseline output)',
              'J|Gamma': r'$[\Phi_{\bar v}\,|\,f_{\mathrm{base}}]$  (both)'}
SPAN_COLOR = {'J': '#0072B2', 'Gamma': '#009E73', 'J|Gamma': '#CC79A7'}


def _rho(B, F, eps=1e-300):
    """||B B^+ F|| / ||F||: the in-span fraction of F. B is (n, k), F is (n,). float64."""
    U, S, _ = torch.linalg.svd(B, full_matrices=False)
    tol = S.max() * max(B.shape) * torch.finfo(torch.float64).eps
    r = int((S > tol).sum())
    proj = U[:, :r] @ (U[:, :r].T @ F)
    return float(torch.linalg.vector_norm(proj) / (torch.linalg.vector_norm(F) + eps))


def _arm_fields(arm, ref_holder, split):
    """(F_raw, F_applied, J, Gamma) for one arm on one SPLIT, at that arm's expansion point.

    TRAIN is where the projection was constructed, so near-zero there confirms the
    implementation and nothing more. VALIDATION records the projection never saw, so near-zero
    there is a generalisation result and non-zero is the honest limitation. The two are reported
    side by side and never merged: merging averages away the only comparison worth making.

    `theta_frozen` is NOT refitted on validation. It is the coefficient the training set
    produced, and it is what the model actually carries when it runs on unseen data, so applying
    it at validation points is the faithful thing to measure.
    """
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

    if ref_holder.get(split) is None:
        # One reference set per split, shared by every arm: it depends only on cfg and norm,
        # which are identical across the three runs (same MODE, same seed, same normalisation).
        from build import PINNED_TRAIN, PINNED_VAL
        files = PINNED_TRAIN if split == 'train' else PINNED_VAL
        ref_holder[split] = build_reference_set(cfg, norm, cfg.nx_ann, files=files, verbose=True)
    ref = ref_holder[split]

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
    print('[arm] %-9s %-5s space=%-7s applied = %s   ||F_ANN|| = %.4e  ||F_applied|| = %.4e'
          % (arm, split, space or '-', note, float(torch.linalg.vector_norm(F)),
             float(torch.linalg.vector_norm(F_applied))))
    del fit_sys
    return F, F_applied, J, Gamma


def main():
    plt = style()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    arms = [a for a in ARM_ORDER if a != 'baseline']
    ref_holder, out = {}, {}

    for arm in arms:
        out[arm] = {}
        for split in SPLITS:
            F_raw, F_app, J, Gamma = _arm_fields(arm, ref_holder, split)
            JG = torch.cat([J, Gamma[:, None]], dim=1)
            rec = {}
            for span, B in (('J', J), ('Gamma', Gamma[:, None]), ('J|Gamma', JG)):
                rec[span] = dict(applied=_rho(B, F_app), raw=_rho(B, F_raw))
            rec['norm_raw'] = float(torch.linalg.vector_norm(F_raw))
            rec['norm_applied'] = float(torch.linalg.vector_norm(F_app))
            rec['n_samples'] = int(J.shape[0] // 6)
            out[arm][split] = rec
            print('       %-5s rho(applied):  J %.3e   Gamma %.3e   [J|Gamma] %.3e'
                  % ((split,) + tuple(rec[s]['applied'] for s in SPANS)))
            del J, Gamma, JG, F_raw, F_app

    _draw(plt, out, arms)
    p = FIGDIR / 'orthogonality_data.json'
    p.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print('\nwrote %s' % p)
    return 0


def _draw(plt, out, arms):
    """Two panels, train and validation, on a SHARED log axis so they compare by eye."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    x = np.arange(len(arms))
    w = 0.26
    floor = 1e-17
    title = {'train': 'train  (where the projection is constructed)',
             'val': 'validation  (held out from it)'}
    for ax, split in zip(axes, SPLITS):
        for i, span in enumerate(SPANS):
            v = [max(out[a][split][span]['applied'], floor) for a in arms]
            ax.bar(x + (i - 1) * w, v, w, color=SPAN_COLOR[span], label=SPAN_LABEL[span])
            for xi, vi in zip(x + (i - 1) * w, v):
                ax.annotate('%.0e' % vi, (xi, vi), textcoords='offset points', xytext=(0, 2),
                            ha='center', fontsize=5.5, rotation=90, color=MUTED)
        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels([ARMS[a]['label'] for a in arms], fontsize=7)
        ax.axhline(1.0, color=INK, lw=0.8, ls=':')
        n = out[arms[0]][split]['n_samples']
        ax.set_title('%s\n%d one-step samples' % (title[split], n),
                     loc='left', fontsize=8, color=MUTED)
    axes[0].set_ylabel(r'$\rho = \|AA^{+}\widetilde F\| / \|\widetilde F\|$')
    axes[0].legend(frameon=False, fontsize=7, loc='lower left')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(FIGDIR / ('orthogonality_applied.%s' % ext))
    print('  wrote %s' % (FIGDIR / 'orthogonality_applied.{png,pdf}'))
    plt.close(fig)


if __name__ == '__main__':
    sys.exit(main())
