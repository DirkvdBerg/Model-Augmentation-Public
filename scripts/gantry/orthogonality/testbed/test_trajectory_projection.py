"""V0/V3/V4 for the shared component `model_augmentation/fit_systems/trajectory_orth_projection.py`.

Category: verification of the GENERIC mathematical component. Not a claim about any system.

The point of this file is that the gantry adapter and the synthetic testbed import the SAME
module, so these checks apply to both. They are deliberately arranged so that a wrong
implementation FAILS rather than merely looks plausible.

  V3  geometry     orthonormality, Q_E perpendicular to Q_S, encoder profiling actually
                   removing a direction, and the case where profiling removes ALL physical
                   information, which must raise rather than silently produce a basis.
  V4  chunking     the exact two-pass form must reproduce the single-stack value AND its
                   gradient. The test that has teeth is a CONSTRUCTED CROSS-CHUNK
                   CANCELLATION case: chunks whose projected residuals nearly cancel, so
                   ||sum_b P_b d_b||^2 is small while sum_b ||P_b d_b||^2 is large. A uniform
                   random case passes under BOTH formulations and proves nothing.
  V0  diagnostics  shrinkage must leave `ell` invariant while both amplitudes fall;
                   selective removal of the in-span part must lower `ell`. A metric that
                   cannot tell these apart cannot support the method's claim.

Run: conda run -n GraduationProject python scripts/gantry/orthogonality/testbed/test_trajectory_projection.py
"""
__project_origin__ = "added"

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

from model_augmentation.fit_systems.trajectory_orth_projection import (  # noqa: E402
    TrajectoryProjectionGeometry, ProjectedResidualAccumulator, penalty_full_stack,
    compressed_nuisance_basis)

F64 = torch.float64
PASS, FAIL = [], []


def check(name, ok, detail=''):
    (PASS if ok else FAIL).append(name)
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f'   {detail}' if detail else ''))


def hdr(s):
    print('\n' + '=' * 78 + f'\n{s}\n' + '=' * 78)


# ============================================================================ V3
def v3_geometry():
    hdr('V3  geometry')
    rng = np.random.default_rng(0)
    N, r, ne = 60, 3, 4
    S = rng.standard_normal((N, r))
    E = rng.standard_normal((N, ne))
    g = TrajectoryProjectionGeometry(S, E)
    g.validate()
    check('Q_S orthonormal and perpendicular to Q_E', True, g.summary().splitlines()[3].strip())
    check('rank(Q_E) = n_xi for generic E', g.rank_E == ne, f'{g.rank_E} of {ne}')
    check('rank(Q_S) = r when E does not span S', g.rank_S == r, f'{g.rank_S} of {r}')

    # Encoder that reproduces one physical direction exactly: that direction must be removed.
    E2 = np.hstack([rng.standard_normal((N, 2)), S[:, [0]]])
    g2 = TrajectoryProjectionGeometry(S, E2)
    check('a physical direction reproducible by the encoder is PROFILED OUT',
          g2.rank_S == r - 1, f'rank(S) {g2.rank_S_before_profiling} -> {g2.rank_S}')

    # Encoder spanning every physical direction: must RAISE, not return an empty basis.
    try:
        TrajectoryProjectionGeometry(S, np.hstack([S, rng.standard_normal((N, 1))]))
        check('profiling away ALL physical information raises', False, 'no exception')
    except ValueError as e:
        check('profiling away ALL physical information raises', True, str(e)[:60] + '...')

    # A diagonal weight must change the geometry; identity must not be silently assumed.
    w = np.exp(rng.standard_normal(N))
    gw = TrajectoryProjectionGeometry(S, E, L=w)
    ang = np.linalg.svd(gw.Q_S.numpy().T @ g.Q_S.numpy(), compute_uv=False).min()
    check('a diagonal loss weight changes the basis', ang < 0.999, f'min cos {ang:.4f}')


# ============================================================================ V4
def _split(n, k):
    e = np.linspace(0, n, k + 1).astype(int)
    return [slice(e[i], e[i + 1]) for i in range(k)]


def v4_chunking():
    hdr('V4  exact two-pass chunking')
    rng = np.random.default_rng(1)
    N, r, beta = 96, 3, 0.7
    S = rng.standard_normal((N, r))
    g = TrajectoryProjectionGeometry(S, None)          # identity L, so rows chunk cleanly

    # `d` is a differentiable function of a parameter, so gradients can be compared.
    q = torch.tensor(rng.standard_normal(5), dtype=F64, requires_grad=True)
    M = torch.tensor(rng.standard_normal((N, 5)), dtype=F64)

    def d_of(q, rows=None):
        v = M @ q + torch.tanh(M @ q)
        return v if rows is None else v[rows]

    # ---- reference: one stack -------------------------------------------------
    q.grad = None
    V_full = penalty_full_stack(g, d_of(q), beta)
    V_full.backward()
    g_full = q.grad.clone()

    for k in (2, 3, 4, 8):
        rows = _split(N, k)
        acc = ProjectedResidualAccumulator(g, beta)
        with torch.no_grad():
            for sl in rows:
                acc.accumulate_value(d_of(q, sl), rows=sl)
        acc.freeze()
        q.grad = None
        for sl in rows:
            acc.backward_chunk(d_of(q, sl), rows=sl)
        ev = abs(acc.value - float(V_full)) / max(abs(float(V_full)), 1e-300)
        eg = float((q.grad - g_full).norm() / max(g_full.norm(), torch.tensor(1e-300)))
        check(f'{k} chunks reproduce the single-stack value and gradient',
              ev < 1e-12 and eg < 1e-10, f'rel value {ev:.2e}, rel grad {eg:.2e}')

    # ---- the test with teeth: cross-chunk cancellation -------------------------
    # Construct chunk contributions whose projected residuals cancel to a prescribed small
    # remainder. Chunk 1 has residual r1; chunk 2 is solved to have residual -r1 + eps.
    # Then the CORRECT penalty is beta||eps||^2 while the naive per-chunk sum is about
    # 2 beta||r1||^2, so the two formulations differ by a controlled, large factor. An
    # earlier version cancelled only the FIRST basis component, leaving the other two
    # uncancelled, which capped the discrepancy at 93x and made the threshold arbitrary.
    rows = _split(N, 2)
    r1 = torch.tensor(rng.standard_normal(g.rank_S), dtype=F64)
    eps = torch.tensor(1e-3 * rng.standard_normal(g.rank_S), dtype=F64)
    d_c = torch.zeros(N, dtype=F64)
    for sl, target in ((rows[0], r1), (rows[1], -r1 + eps)):
        B = g.Q_S[sl]                                   # (n_b, rank), n_b > rank
        d_c[sl] = B @ torch.linalg.solve(B.T @ B, target)
    resid = g.Q_S.T @ d_c
    assert torch.allclose(resid, eps, atol=1e-10), 'construction failed'
    correct = float(penalty_full_stack(g, d_c, beta))
    naive = sum(float(beta * (g.Q_S[sl].T @ d_c[sl]).pow(2).sum()) for sl in rows)
    acc = ProjectedResidualAccumulator(g, beta)
    with torch.no_grad():
        for sl in rows:
            acc.accumulate_value(d_c[sl], rows=sl)
    acc.freeze()
    ratio = naive / max(correct, 1e-300)
    check('two-pass is exact on a cross-chunk cancellation case',
          abs(acc.value - correct) / max(correct, 1e-300) < 1e-10,
          f'two-pass {acc.value:.3e} vs stack {correct:.3e}')
    check('the naive per-chunk penalty is DIFFERENT on that case (test has teeth)',
          ratio > 1e4, f'naive/correct = {ratio:.3e}   (correct {correct:.2e}, '
                       f'naive {naive:.2e})')


# ============================================================================ V0
def v0_diagnostics():
    hdr('V0  diagnostics separate shrinkage from rotation')
    rng = np.random.default_rng(2)
    N, r = 40, 2
    S = rng.standard_normal((N, r))
    g = TrajectoryProjectionGeometry(S, None)
    d = torch.tensor(rng.standard_normal(N), dtype=F64)
    m0 = g.contribution_metrics(d)

    m_shrunk = g.contribution_metrics(0.1 * d)
    check('uniform shrinking leaves ell UNCHANGED',
          abs(m_shrunk['ell'] - m0['ell']) < 1e-12,
          f'ell {m0["ell"]:.6f} -> {m_shrunk["ell"]:.6f}')
    check('uniform shrinking lowers BOTH amplitudes',
          m_shrunk['a_par'] < m0['a_par'] and m_shrunk['a_perp'] < m0['a_perp'],
          f'a_par {m0["a_par"]:.3e} -> {m_shrunk["a_par"]:.3e}')

    QS = g.Q_S
    d_rot = d - 0.9 * (QS @ (QS.T @ d))       # remove 90% of the in-span part only
    m_rot = g.contribution_metrics(d_rot)
    check('selective in-span removal LOWERS ell',
          m_rot['ell'] < 0.2 * m0['ell'], f'ell {m0["ell"]:.4f} -> {m_rot["ell"]:.4f}')
    check('selective removal leaves the perpendicular amplitude intact',
          abs(m_rot['a_perp'] - m0['a_perp']) < 1e-10,
          f'a_perp {m0["a_perp"]:.6e} -> {m_rot["a_perp"]:.6e}')
    check('ell is undefined (nan) for a negligible contribution',
          np.isnan(g.contribution_metrics(0.0 * d)['ell']))


# ============================================================================ V3b
def v3b_compressed_nuisance():
    """THE COMPRESSED NUISANCE CONSTRUCTION, calling the PRODUCTION FUNCTION.

    ADDED 2026-09-08 after review found `compressed_nuisance_basis` computing `U_w^T J_w`
    instead of `(U_w^T Gamma_w) J_w`. The omission is dimensionally invalid whenever
    `T n_y != n_x`, so it raises rather than silently returning a wrong basis -- and it survived
    a suite that reported the construction verified.

    THE REASON IT SURVIVED IS THE POINT OF THIS TEST. The small mathematical reference case
    implemented the same construction a second time, correctly, inline, and compared THAT copy
    against an explicitly assembled `E`. Two independent implementations agreeing is exactly
    what the dual-engine rule asks for at the level of MATHEMATICS, and it establishes nothing
    whatever about the function the gantry actually calls. An independent reference must be
    compared against the production implementation on the same inputs, not merely against
    itself.

    Deliberately small and dependency-free, so it runs in a second and can be the first thing a
    change to that function has to survive.
    """
    hdr('V3b  compressed nuisance basis (production function)')
    rng = np.random.default_rng(20260908)

    for label, (rows, nx, n_xi, deficient) in {
        'square-ish, full rank': ((7, 7, 7), 4, 9, False),
        'tall windows, T*ny >> nx': ((30, 30, 30, 30), 6, 50, False),
        'wide encoder, n_xi >> rows': ((5, 5), 3, 40, False),
        'rank-deficient Gamma_w': ((12, 12, 12), 5, 20, True),
    }.items():
        gammas, jacs = [], []
        for m in rows:
            G = rng.standard_normal((m, nx))
            if deficient:
                G[:, -1] = G[:, 0]            # exactly one dependent column
            gammas.append(G)
            jacs.append(rng.standard_normal((nx, n_xi)))
        N = int(sum(rows))

        # explicit E, assembled block-diagonally, as the definition says
        E = np.vstack([gammas[w] @ jacs[w] for w in range(len(rows))])
        U, sv, _ = np.linalg.svd(E, full_matrices=False)
        rE = int(np.sum(sv > 1e-12 * sv[0]))
        Q_exp = U[:, :rE]

        Q, info = compressed_nuisance_basis(gammas, jacs, rtol=1e-12, N=N)

        orth = float(np.abs(Q.T @ Q - np.eye(Q.shape[1])).max()) if Q.shape[1] else 0.0
        # ||P - P'||_2 in the factors; never form the N-by-N projectors
        if Q.shape[1] and rE:
            dist = max(float(np.linalg.norm(Q - Q_exp @ (Q_exp.T @ Q), 2)),
                       float(np.linalg.norm(Q_exp - Q @ (Q.T @ Q_exp), 2)))
        else:
            dist = float(Q.shape[1] != rE)
        exp_rank = min(rE, nx * len(rows))
        check(f'compressed basis spans range(E) [{label}]',
              Q.shape[1] == rE and dist < 1e-9 and orth < 1e-10,
              f'rank {Q.shape[1]} vs explicit {rE} (expected <= {exp_rank}), '
              f'projector distance {dist:.2e}, orthonormality {orth:.2e}')
        if deficient:
            check(f'per-window rank deficiency is detected [{label}]',
                  all(r == nx - 1 for r in info["ranks_w"]),
                  f'ranks_w {info["ranks_w"]} against nx={nx}')

    # Z must be r_G-by-n_xi, which is the whole point: it must NOT carry the trajectory rows.
    gammas = [rng.standard_normal((200, 6)) for _ in range(4)]
    jacs = [rng.standard_normal((6, 500)) for _ in range(4)]
    Q, info = compressed_nuisance_basis(gammas, jacs, rtol=1e-12, N=800)
    check('Z is r_G-by-n_xi and never carries the scored rows',
          info['Z_shape'] == [info['r_G'], 500] and info['r_G'] <= 6 * 4,
          f'Z {info["Z_shape"]}, r_G {info["r_G"]}, against an explicit E of [800, 500]')

    # a zero propagation block contributes nothing and must not crash
    gammas[1] = np.zeros((200, 6))
    Q0, info0 = compressed_nuisance_basis(gammas, jacs, rtol=1e-12, N=800)
    check('a zero Gamma_w block contributes rank 0 rather than raising',
          info0['ranks_w'][1] == 0 and np.allclose(Q0[200:400], 0.0),
          f'ranks_w {info0["ranks_w"]}')

# ============================================================================ V3c
def v3c_geometry_round_trip():
    """save_state / from_state must restore EVERY field, not just the ones projection uses.

    ADDED 2026-09-08 after review. The first reload path built the object from a `zeros((N, 1))`
    placeholder `S` and patched in the bases and ranks afterwards. It projected correctly and
    described itself wrongly: `zero_rank` stayed True, `scale_LS` stayed 0, `rank_atol` was the
    constructor default rather than the saved value, and `sv_E` was empty. Each of those feeds a
    decision somewhere, so "the multiplication works" is not the property to test.
    """
    hdr('V3b  geometry serialisation round trip')
    rng = np.random.default_rng(7)
    N, r, n_xi = 60, 5, 9
    S = rng.standard_normal((N, r))
    S[:, -1] = S[:, 0]                       # a deliberate rank deficiency to carry through
    E = rng.standard_normal((N, n_xi))
    g = TrajectoryProjectionGeometry(S, E=E, rank_rtol=1e-12, strict=False,
                                     metadata={'tag': 'round-trip'})
    st = g.save_state()
    h = TrajectoryProjectionGeometry.from_state(st, dtype=torch.float64)

    fields = ('N', 'n_phys', 'rank_rtol', 'rank_atol', 'rank_E', 'rank_S', 'scale_LS',
              'rank_S_before_profiling', 'zero_rank')
    bad = [f for f in fields if getattr(g, f) != getattr(h, f)]
    check('every scalar field survives the round trip', not bad,
          'differing: ' + str(bad) if bad else ', '.join(fields))
    check('the bases survive exactly',
          torch.equal(g.Q_S, h.Q_S) and torch.equal(g.Q_E, h.Q_E))
    check('the spectra survive',
          np.allclose(g.sv_S, h.sv_S) and np.allclose(g.sv_E, h.sv_E),
          f'sv_E length {len(h.sv_E)} (the placeholder path left this empty)')
    d = torch.as_tensor(rng.standard_normal(N), dtype=torch.float64)
    check('the penalty value is identical after a round trip',
          float(g.penalty_mse(d, 0.3)) == float(h.penalty_mse(d, 0.3)),
          f'{float(h.penalty_mse(d, 0.3)):.12e}')
    m1, m2 = g.contribution_metrics(d), h.contribution_metrics(d)
    check('the diagnostics agree after a round trip',
          all(abs(m1[k] - m2[k]) < 1e-15 for k in ('ell', 'a_par', 'a_perp')))

    # dtype/device alignment: a float64 geometry against a float32 model is the concrete
    # mismatch the gantry build path can produce.
    h32 = TrajectoryProjectionGeometry.from_state(st, dtype=torch.float32)
    check('from_state honours a float32 request', h32.Q_S.dtype == torch.float32)
    g.to_model(dtype=torch.float32)
    check('to_model moves the buffers in place', g.Q_S.dtype == torch.float32
          and g.Q_E.dtype == torch.float32)

    # a corrupted record must be refused, not loaded
    bad_st = dict(st)
    bad_st['rank_S'] = int(st['rank_S']) + 1
    try:
        TrajectoryProjectionGeometry.from_state(bad_st)
        check('a bases/ranks disagreement is refused', False, 'it loaded')
    except AssertionError:
        check('a bases/ranks disagreement is refused', True)

if __name__ == '__main__':
    torch.set_default_dtype(F64)
    v3_geometry()
    v3b_compressed_nuisance()
    v3c_geometry_round_trip()
    v4_chunking()
    v0_diagnostics()
    print('\n' + '=' * 78)
    print(f'{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        print('FAILED: ' + ', '.join(FAIL))
    sys.exit(1 if FAIL else 0)
