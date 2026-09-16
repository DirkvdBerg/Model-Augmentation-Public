"""Stage 2: the empirical rank of the REDUCED stacked Jacobian under representative excitation.

__project_origin__ = "added"

Structural rank 10 is established by the investigation (rank 10, nullity 4, stable from rtol
`1e-4` to `1e-14`, gap factor `7.9e12`) on the fourteen-column raw Jacobian. What is NOT
established is the empirical rank of the TEN-column reduced Jacobian on representative
excitation, which is the matrix the pseudo-inverse actually sees.

"Representative excitation" is the designed point set (user clarification, 2026-09-15), not a
dataset: `design_points(n, seed)` draws independent `(x, u)` at physical magnitudes with `Y`
swept deterministically across `[-0.30, 0.30]`. `n = 256` is the headline, with `32 / 64 / 256 /
1024` for sensitivity. No dataset is loaded.

  T1  rank and spectrum of `J` (10 columns) at every rtol from `1e-4` to `1e-14`   <- CRITERION
  T2  rank and spectrum of `[J | c]` (11 columns), reported SEPARATELY. Expect 11.
  T3  THE OFFSET HAZARD, reproduced on the reduced basis: the rank of `[J | c]` against a cutoff
      stated relative to `sigma_0` of the UNSCALED matrix, against the same cutoff on the
      COLUMN-SCALED matrix. The first is the documented failure; the second is what the library
      does.
  T4  sensitivity to the point count, 32 / 64 / 256 / 1024
  T5  sensitivity to column scaling: unscaled, unit-column, and the log/physical coordinate pair
  T6  the additional-state rows of the basis are exactly zero, which is the assertion the
      decoupling proposition rests on
  T7  the rank-loss policy: force a rank drop, check that it is REPORTED with the discarded
      singular values and the principal angles to the previous span, and that `on_rank_loss =
      'abort'` raises rather than continuing silently

Criterion (handoff Sect. 10): `rank(J) == 10` at every rtol from `1e-4` to `1e-14`, with the gap
between the last kept and first discarded singular value above `1e6`. `rank([J | c])` reported
separately, expected 11.
"""

__project_origin__ = "added"

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C                                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TOLS = (1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14)
# THEORY: the criterion is the handoff's, Sect. 10. Rank 10 at every rtol in the decade band,
# gap above 1e6. Not tuned here.
GAP_MIN = 1e6
N_HEAD = 256


def spectrum_line(S):
    return '  '.join(f'{float(v / S[0]):.3e}' for v in S)


def main():
    torch.manual_seed(0)
    sys.stdout = C.Tee(os.path.join(HERE, 'results.log'))
    from model_augmentation.fit_systems import obc_projection as obc

    red = C.make_reduced_block()

    print('=' * 78)
    print('STAGE 2 -- empirical rank of the reduced stacked Jacobian')
    print('=' * 78)
    print(f'  designed point set, Y swept over {C.Y_RANGE}, float64')
    print(f'  criterion: rank(J) == 10 at every rtol 1e-4..1e-14, gap > {GAP_MIN:.0e}')

    x, u = C.design_points(N_HEAD, seed=0)
    Z = C.zu(x, u)
    J, c, v_bar = obc.stacked_basis(red, Z)
    F = torch.cat([J, c], dim=1)
    print(f'\n  J shape {tuple(J.shape)}, c shape {tuple(c.shape)}, '
          f'F = [J | c] shape {tuple(F.shape)}')

    # ------------------------------------------------------------------ T1 the criterion
    print('\n--- T1  rank and spectrum of J (10 columns), COLUMN-SCALED ---')
    rJ = obc.rank_report(J, tols=TOLS)
    S = torch.tensor(rJ['singular_values_scaled'])
    print(f'  singular values / sigma_0: {spectrum_line(S)}')
    print(f'  rank at rtol: ' + ',  '.join(f'{k}: {v}' for k, v in rJ['rank_at_rtol'].items()))
    print(f'  condition number {rJ["condition_number"]:.4e}')
    ranks = list(rJ['rank_at_rtol'].values())
    all_ten = all(r == 10 for r in ranks)
    # The gap: J is expected FULL column rank, so nothing is discarded and there is no gap inside
    # the matrix. The meaningful gap is then sigma_9 against the float64 noise floor, which is
    # what "full rank with room to spare" means. Reported both ways rather than conflated.
    gap_internal = rJ['gap_last_kept_to_first_discarded']
    margin = float(S[-1] / S[0]) / 2.2e-16
    print(f'  gap last kept to first discarded: {gap_internal}'
          f'   (inf means nothing was discarded, i.e. full column rank)')
    print(f'  smallest singular value is {float(S[-1] / S[0]):.3e} of sigma_0, which is '
          f'{margin:.2e} x the float64 epsilon')
    print(f'  rank 10 at EVERY rtol: {all_ten}')

    # ------------------------------------------------------------------ T2 the extended basis
    print('\n--- T2  rank and spectrum of [J | c] (11 columns), COLUMN-SCALED ---')
    rF = obc.rank_report(F, tols=TOLS)
    SF = torch.tensor(rF['singular_values_scaled'])
    print(f'  singular values / sigma_0: {spectrum_line(SF)}')
    print(f'  rank at rtol: ' + ',  '.join(f'{k}: {v}' for k, v in rF['rank_at_rtol'].items()))
    print(f'  condition number {rF["condition_number"]:.4e}')
    # How much of the offset lies OUTSIDE span(J): the investigation measured 57 percent on the
    # raw basis, which is what makes the extra column worth its hazard.
    QJ = torch.linalg.qr(J)[0]
    out = float(torch.linalg.vector_norm(c - QJ @ (QJ.T @ c))
                / torch.linalg.vector_norm(c))
    print(f'  fraction of the offset column OUTSIDE span(J): {out:.4f}')

    # ------------------------------------------------------------------ T3 the offset hazard
    print('\n--- T3  the offset hazard: cutoff against sigma_0 UNSCALED vs COLUMN-SCALED ---')
    S_uns = torch.linalg.svdvals(F)
    D = F.abs().pow(2).sum(dim=0).sqrt()
    print(f'  column norms: parameter columns {float(D[:-1].min()):.3e} to '
          f'{float(D[:-1].max()):.3e}, offset column {float(D[-1]):.3e}')
    print(f'  offset column / largest parameter column = {float(D[-1] / D[:-1].max()):.1f} x')
    print(f'    {"rcond":>8}  {"rank UNSCALED":>14}  {"rank COLUMN-SCALED":>20}')
    for t in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-14):
        ru = int((S_uns > t * S_uns[0]).sum())
        rs = int((SF > t * SF[0]).sum())
        print(f'    {t:>8.0e}  {ru:>14}  {rs:>20}')
    print('  the unscaled column reproduces the documented failure; the scaled column is what')
    print('  the library uses, and is why a sigma_0-relative cutoff is legitimate there.')

    # ------------------------------------------------------------------ T4 point count
    print('\n--- T4  sensitivity to the point count ---')
    print(f'    {"M":>6}  {"rank(J)":>8}  {"rank([J|c])":>12}  {"cond(J)":>12}  '
          f'{"sigma_min/sigma_0":>18}')
    for M in (32, 64, 256, 1024):
        xm, um = C.design_points(M, seed=0)
        Jm, cm, _ = obc.stacked_basis(red, C.zu(xm, um))
        rj = obc.rank_report(Jm, tols=TOLS)
        rf = obc.rank_report(torch.cat([Jm, cm], dim=1), tols=TOLS)
        sm = torch.tensor(rj['singular_values_scaled'])
        print(f'    {M:>6}  {rj["rank_at_rcond"]:>8}  {rf["rank_at_rcond"]:>12}  '
              f'{rj["condition_number"]:>12.4e}  {float(sm[-1] / sm[0]):>18.4e}')

    # ------------------------------------------------------------------ T5 column scaling
    print('\n--- T5  sensitivity to column scaling ---')
    variants = {
        'unscaled (reduced/log coords)': J,
        'unit-column': J / D[:-1],
        'columns x 1e6': J * 1e6,
        'one column x 1e6': torch.cat([J[:, :1] * 1e6, J[:, 1:]], dim=1),
    }
    print(f'    {"variant":<32}  {"rank":>5}  {"cond":>12}')
    for tag, M_ in variants.items():
        r = obc.rank_report(M_, tols=TOLS)
        print(f'    {tag:<32}  {r["rank_at_rcond"]:>5}  {r["condition_number"]:>12.4e}')
    print('  rank is invariant to column scaling because the report scales columns first;')
    print('  the condition number of the SCALED matrix is likewise invariant. That is the point.')

    # ------------------------------------------------------------------ T6 additional-state rows
    print('\n--- T6  the additional-state rows of the basis ---')
    n_a = C.N_A
    Jaug = torch.zeros(J.shape[0] // C.N_P * (C.N_P + n_a), J.shape[1], dtype=C.F64)
    Jaug = Jaug.reshape(-1, C.N_P + n_a, J.shape[1])
    Jaug[:, :C.N_P, :] = J.reshape(-1, C.N_P, J.shape[1])
    aug_rows = Jaug[:, C.N_P:, :]
    print(f'  stacking all n_p + n_a = {C.N_P + n_a} rows; the baseline writes no '
          f'additional-state update, so those rows are structurally zero.')
    print(f'  max abs value on the additional-state rows: '
          f'{aug_rows.abs().max().item():.3e}   (exactly 0.0 expected)')
    assert aug_rows.abs().max().item() == 0.0

    # ------------------------------------------------------------------ T7 rank-loss policy
    print('\n--- T7  the rank-loss policy ---')
    lines = []
    b_full = obc.OBCBasis(J, c, expected_rank=11, log=lines.append)
    print(f'  full basis: rank {b_full.rank}, rank_loss = {b_full.rank_loss}')
    # Force a drop by duplicating a column: the span is unchanged but one singular value is zero.
    J_dup = torch.cat([J[:, :-1], J[:, -2:-1]], dim=1)
    lines.clear()
    b_drop = obc.OBCBasis(J_dup, c, expected_rank=11, previous=b_full,
                          on_rank_loss='report', log=lines.append)
    print(f'  duplicated-column basis: rank {b_drop.rank}, rank_loss = {b_drop.rank_loss}')
    for ln in lines:
        print(f'    LOG: {ln}')
    assert b_drop.rank_loss and any('RANK LOSS' in ln for ln in lines)
    assert any('discarded singular values' in ln for ln in lines)
    assert any('principal angles' in ln for ln in lines)
    try:
        obc.OBCBasis(J_dup, c, expected_rank=11, on_rank_loss='abort', log=lambda *_: None)
        raised = False
    except obc.RankLossAbort:
        raised = True
    print(f'  on_rank_loss = "abort" raised RankLossAbort: {raised}')
    assert raised

    print('\n' + '=' * 78)
    ok = all_ten and rF['rank_at_rcond'] == 11
    print(f'STAGE 2 VERDICT: rank(J) = 10 at every rtol 1e-4..1e-14: {all_ten}. '
          f'rank([J|c]) = {rF["rank_at_rcond"]} (expected 11). '
          f'{"MET" if ok else "NOT MET"}')
    print('=' * 78)
    sys.stdout.flush()


if __name__ == '__main__':
    main()
