"""Stage 4: does the library projection remove the baseline-representable component exactly?

__project_origin__ = "added"

The construction, in `model_augmentation/fit_systems/obc_projection.py`: `OBCBasis` holds the
frozen `[J | c]` and its column-scaled pseudo-inverse; `OBC_Projected_ANN_Block` wraps a
`Static_ANN_Block`, refits the coefficient from the current ANN output on every forward pass and
subtracts `[J(z) | c(z)] a` pointwise via one directional derivative.

  T1  the orthogonality residual on the construction evaluations, at FULL rank   <- CRITERION
  T2  the same at deliberately REDUCED rank, which the derivation says costs only the uniqueness
      of the coefficient, not the exactness of the projection
  T3  the identity `g~ = (I - P) g` against an independently formed projector
  T4  the additional-state rows: the basis is exactly zero there, so `g_a` passes through
      UNCHANGED at fixed arguments, with no special case in the code
  T5  the pointwise caveat: the per-sample product does NOT vanish while the stacked sum does.
      No claim of ours may be worded pointwise (the source's Eq. (40) is wrong to display it so)
  T6  `rho` before and after, on the reference set and OFF it, which is the honest report: the
      construction is exact on the frozen set and only partially effective away from it
  T7  the coefficient and the correction are identical whether computed through the wrapper or
      by hand from the stacked algebra, so the wrapper is not quietly doing something else, and
      the flag-off path returns the wrapped block's tensor ITSELF. The wrapper inside a full
      `Interconnect` is stage 6's business, not this one's.

Criterion (handoff Sect. 10): `norm(F' g~) / (norm(F) norm(g~))` at or below `1e-12` on the
construction evaluations, at full rank AND at deliberately reduced rank. The investigation's
prototype achieved `1.96e-14`.
"""

__project_origin__ = "added"

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C                                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# THEORY: the criterion is the handoff's, Sect. 10, scale free so the offset column cannot
# dominate it the way a raw inner product does.
TOL = 1e-12
M_REF = 256
ROUTE_IX = tuple(range(C.N_P + C.N_A))      # production routing, config.py ann_route_ix


def make_ann(seed=0, n_nodes=16, n_layers=2):
    """A `Static_ANN_Block` at the pipeline's shape, given NONZERO weights.

    The pipeline initialises the last layer to zero so the augmentation starts as a no-op. A zero
    write would make every orthogonality residual trivially zero, so the tests re-initialise the
    network: a projection that removes nothing is not evidence of anything.
    """
    from model_augmentation.fit_systems.blocks import Static_ANN_Block
    torch.manual_seed(seed)
    ann = Static_ANN_Block(nz=C.N_P + C.N_A + C.N_U, nw=len(ROUTE_IX),
                           n_nodes_per_layer=n_nodes, n_hidden_layers=n_layers).to(C.F64)
    with torch.no_grad():
        for p in ann.parameters():
            p.copy_(torch.randn_like(p) * 0.3)
    return ann


def ref_points(M, seed=0, latent_scale=1.0):
    """`(M, nz, 1)` ANN inputs: physical state, NONZERO additional states, input."""
    x, u = C.design_points(M, seed=seed)
    g = torch.Generator().manual_seed(seed + 500)
    xa = torch.rand(M, C.N_A, generator=g, dtype=C.F64) * 2 * latent_scale - latent_scale
    return torch.cat([x, xa, u], dim=1).unsqueeze(-1)


def main():
    torch.manual_seed(0)
    sys.stdout = C.Tee(os.path.join(HERE, 'results.log'))
    from model_augmentation.fit_systems import obc_projection as obc

    red = C.make_reduced_block()
    ann = make_ann()
    blk = obc.OBC_Projected_ANN_Block(ann, red, ROUTE_IX, C.N_P, C.N_A, enabled=True)

    Z_ref = ref_points(M_REF, seed=0)
    basis = blk.rebuild_basis(Z_ref, expected_rank=11)

    print('=' * 78)
    print('STAGE 4 -- the projection: does it remove the baseline component exactly?')
    print('=' * 78)
    print(f'  M = {M_REF} construction evaluations, routing {ROUTE_IX}, '
          f'n_p = {C.N_P}, n_a = {C.N_A}, float64')
    print(f'  basis rank {basis.rank}/11 at rcond {basis.rcond:.0e}, '
          f'condition number {basis.spectrum()["condition_number"]:.4e}')
    print(f'  criterion: scale-free residual at or below {TOL:.0e}')

    G = blk._stack_physical_write(ann(Z_ref)).detach()
    print(f'  stacked physical write: shape {tuple(G.shape)}, norm '
          f'{torch.linalg.vector_norm(G).item():.6e}')

    # ------------------------------------------------------------------ T1 the criterion
    print('\n--- T1  orthogonality residual on the construction evaluations, FULL rank ---')
    r1 = basis.residual_ratio(G)
    a = basis.coefficient(G)
    Gt = G - basis.F @ a
    print(f'  coefficient a: {a.detach().numpy()}')
    print(f'  norm(F\' G~)                     = {torch.linalg.vector_norm(basis.F.T @ Gt):.6e}')
    print(f'  norm(F) norm(G~)                = '
          f'{(torch.linalg.matrix_norm(basis.F) * torch.linalg.vector_norm(Gt)).item():.6e}')
    print(f'  SCALE-FREE RESIDUAL             = {r1:.4e}   '
          f'{"MET" if r1 <= TOL else "NOT MET"} (<= {TOL:.0e})')
    print(f'  investigation prototype achieved  1.96e-14')

    # ------------------------------------------------------------------ T2 reduced rank
    print('\n--- T2  the same at deliberately REDUCED rank ---')
    print('  "Reduced rank" has to mean a GENUINELY rank-deficient basis, one with a zero')
    print('  singular value, which is what the any-rank proposition of the derivation is about.')
    print('  Rank deficiency then costs only the uniqueness of the coefficient. Truncating a')
    print('  NONZERO singular value is a different operation, measured separately below.')
    print('\n  (a) GENUINE deficiency, a duplicated column (span unchanged, one sigma = 0):')
    print(f'    {"basis":<34}  {"rank":>5}  {"residual":>12}  {"norm(G~)/norm(G)":>18}')
    for tag, Jx in (('full [J | c]', basis.F[:, :-1]),
                    ('column 3 duplicated', torch.cat([basis.F[:, :-1], basis.F[:, 3:4]], 1)),
                    ('columns 3 and 7 duplicated',
                     torch.cat([basis.F[:, :-1], basis.F[:, 3:4], basis.F[:, 7:8]], 1)),
                    ('a column zeroed',
                     torch.cat([basis.F[:, :3], torch.zeros_like(basis.F[:, 3:4]),
                                basis.F[:, 4:-1]], 1))):
        b = obc.OBCBasis(Jx, basis.F[:, -1:], v_bar=basis.v_bar, log=lambda *_: None)
        Gt_ = G - b.F @ b.coefficient(G)
        print(f'    {tag:<34}  {b.rank:>5}  {b.residual_ratio(G):>12.4e}  '
              f'{(torch.linalg.vector_norm(Gt_) / torch.linalg.vector_norm(G)).item():>18.4f}')
    b_def = obc.OBCBasis(torch.cat([basis.F[:, :-1], basis.F[:, 3:4]], 1), basis.F[:, -1:],
                         v_bar=basis.v_bar, log=lambda *_: None)
    r2 = b_def.residual_ratio(G)
    print(f'  genuine rank deficiency, residual {r2:.4e}   '
          f'{"MET" if r2 <= TOL else "NOT MET"} (<= {TOL:.0e})')

    print('\n  (b) ARTIFICIAL truncation of NONZERO singular values, for the record:')
    print(f'    {"rcond":>10}  {"rank":>5}  {"residual":>12}  {"norm(G~)/norm(G)":>18}')
    for rc in (1e-10, 1e-2, 1e-1, 0.3, 0.6):
        b = obc.OBCBasis(basis.F[:, :-1], basis.F[:, -1:], v_bar=basis.v_bar, rcond=rc,
                         log=lambda *_: None)
        Gt_ = G - b.F @ b.coefficient(G)
        print(f'    {rc:>10.0e}  {b.rank:>5}  {b.residual_ratio(G):>12.4e}  '
              f'{(torch.linalg.vector_norm(Gt_) / torch.linalg.vector_norm(G)).item():>18.4f}')
    print('  Truncating a nonzero sigma_i leaves exactly that direction in the residual, so')
    print('  exactness is LOST, not preserved. The any-rank guarantee covers genuine deficiency')
    print('  only. That is an argument for the column scaling of stage 2, which is what keeps a')
    print('  sigma_0-relative cutoff from discarding genuine directions in the first place.')

    # ------------------------------------------------------------------ T3 the projector identity
    print('\n--- T3  the identity g~ = (I - P) g against an independent projector ---')
    Q = torch.linalg.qr(basis.F)[0]
    Gt_proj = G - Q @ (Q.T @ G)
    d3 = (Gt - Gt_proj).abs().max().item() / Gt.abs().max().item()
    print(f'  QR projector vs pseudo-inverse subtraction: max rel difference {d3:.4e}')

    # ------------------------------------------------------------------ T4 additional-state rows
    print('\n--- T4  the additional-state write passes through UNCHANGED ---')
    Zt = ref_points(64, seed=3)
    with torch.no_grad():
        w_raw = ann(Zt)
        w_proj = blk(Zt)
    aug_cols = [j for j, r in enumerate(ROUTE_IX) if r >= C.N_P]
    phy_cols = [j for j, r in enumerate(ROUTE_IX) if r < C.N_P]
    d_aug = (w_raw[:, aug_cols] - w_proj[:, aug_cols]).abs().max().item()
    d_phy = (w_raw[:, phy_cols] - w_proj[:, phy_cols]).abs().max().item()
    print(f'  ANN columns writing ADDITIONAL-state rows {aug_cols}: '
          f'max abs change = {d_aug:.3e}   (exactly 0.0 expected)')
    print(f'  ANN columns writing PHYSICAL rows {phy_cols}: '
          f'max abs change = {d_phy:.3e}   (nonzero, or the projection did nothing)')
    assert d_aug == 0.0
    assert d_phy > 0.0
    print('  No special case produces this. The basis has no additional-state rows, so the')
    print('  coefficient never sees g_a and the subtraction has nothing to write there.')

    # ------------------------------------------------------------------ T5 pointwise caveat
    print('\n--- T5  the stacked sum vanishes, the PER-SAMPLE product does not ---')
    Fs = basis.F.reshape(M_REF, C.N_P, 11)
    Gts = Gt.reshape(M_REF, C.N_P)
    per = torch.stack([torch.linalg.vector_norm(Fs[i].T @ Gts[i]) for i in range(M_REF)])
    print(f'  per-sample norm(F_k\' g~_k): median {per.median().item():.4e}, '
          f'min {per.min().item():.4e}, max {per.max().item():.4e}')
    print(f'  stacked sum norm(F\' G~)   : {torch.linalg.vector_norm(basis.F.T @ Gt).item():.4e}')
    print(f'  ratio median per-sample to stacked sum: '
          f'{(per.median() / torch.linalg.vector_norm(basis.F.T @ Gt)).item():.2e}')
    print('  Only the SUM vanishes. No claim may be worded pointwise; the source displays the')
    print('  block structure per data point in its Eq. (40) and that display is wrong.')

    # ------------------------------------------------------------------ T6 rho on and off the set
    print('\n--- T6  rho before and after, ON and OFF the construction evaluations ---')
    # The residual column is the FROZEN-coefficient residual: the coefficient fitted on the
    # construction evaluations, applied on the query set. Refitting on the query set would make
    # every row trivially zero and would say nothing about generalisation.
    print(f'    {"point set":<34}  {"rho before":>11}  {"rho after":>10}  '
          f'{"frozen residual":>16}')
    for tag, Zq in (('the construction evaluations', Z_ref),
                    ('a foreign designed set, seed 11', ref_points(M_REF, seed=11)),
                    ('a foreign set, 4x latent scale', ref_points(M_REF, seed=11, latent_scale=4.)),
                    ('a narrower Y range, frozen-ish', None)):
        if Zq is None:
            xq, uq = C.design_points(M_REF, seed=21, y_range=(-0.02, 0.02))
            gg = torch.Generator().manual_seed(999)
            xa = torch.rand(M_REF, C.N_A, generator=gg, dtype=C.F64) * 2 - 1
            Zq = torch.cat([xq, xa, uq], dim=1).unsqueeze(-1)
        with torch.no_grad():
            Gq = blk._stack_physical_write(ann(Zq))
            Gq_t = blk._stack_physical_write(blk(Zq))
            Jq, cq, _ = obc.stacked_basis(red, blk.phys_input(Zq), v_bar=basis.v_bar)
            bq = obc.OBCBasis(Jq, cq, v_bar=basis.v_bar, log=lambda *_: None)
        num = torch.linalg.vector_norm(bq.F.T @ Gq_t)
        den = (torch.linalg.matrix_norm(bq.F) * torch.linalg.vector_norm(Gq_t)).clamp(min=1e-300)
        print(f'    {tag:<34}  {bq.rho(Gq):>11.4f}  {bq.rho(Gq_t):>10.4f}  '
              f'{float(num / den):>16.4e}')
    print('  The first row is the guarantee: exact on the frozen set. The others are the honest')
    print('  report: away from it the construction removes part, not all. rho is reported, never')
    print('  a raw inner product, which is dominated by the offset direction.')

    # ------------------------------------------------------------------ T8 wrapper vs hand algebra
    print('\n--- T7  the wrapper against the stacked algebra, computed by hand ---')
    Zq = ref_points(32, seed=5)
    with torch.no_grad():
        w_wrapper = blk(Zq)
        # by hand
        a_hand = basis.coefficient(blk._stack_physical_write(ann(Z_ref)))
        Jq, cq, _ = obc.stacked_basis(red, blk.phys_input(Zq), v_bar=basis.v_bar)
        delta_hand = (torch.cat([Jq, cq], dim=1) @ a_hand).reshape(-1, C.N_P)
        w_hand = ann(Zq).clone()
        for j, r in enumerate(ROUTE_IX):
            if r < C.N_P:
                w_hand[:, j, 0] = w_hand[:, j, 0] - delta_hand[:, r]
    d8 = (w_wrapper - w_hand).abs().max().item()
    scale8 = w_hand.abs().max().item()
    print(f'  coefficient a, wrapper vs hand: max abs difference '
          f'{(a_hand - blk.coefficient().detach()).abs().max().item():.3e}')
    print(f'  output, wrapper vs hand: max abs {d8:.3e}, relative {d8 / scale8:.3e}')
    print(f'  (the wrapper uses ONE directional jvp, the hand path the full [J | c] matvec;')
    print(f'   agreeing to round-off is the evidence those are the same object)')
    assert d8 / scale8 < 1e-12

    # flag off must return the wrapped tensor ITSELF
    blk.enabled = False
    with torch.no_grad():
        w_off = blk(Zq)
        w_ann = ann(Zq)
    print(f'  flag OFF: max abs (wrapper - ANN) = {(w_off - w_ann).abs().max().item():.1e}, '
          f'exactly equal = {bool(torch.equal(w_off, w_ann))}')
    assert torch.equal(w_off, w_ann)
    blk.enabled = True

    print('\n' + '=' * 78)
    print(f'STAGE 4 VERDICT: scale-free residual {r1:.4e} at full rank against a criterion of '
          f'{TOL:.0e} -- {"MET" if r1 <= TOL else "NOT MET"}')
    print('=' * 78)
    sys.stdout.flush()


if __name__ == '__main__':
    main()
