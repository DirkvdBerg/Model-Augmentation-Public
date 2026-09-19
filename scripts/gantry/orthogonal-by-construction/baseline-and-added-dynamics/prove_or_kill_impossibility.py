"""Can an added DYNAMIC subsystem be orthogonal to the baseline for EVERY excitation?

The question, stated exactly. The baseline at frozen `Y` is `M q̈ + C q̇ + K q = u_log`, with the
ten identifiable parameters entering only through `M`, `C`, `K`. An added linear subsystem, after
its own states are eliminated, contributes a reaction force on the three baseline coordinates

    F_a(w)  =  -G(w) Q(w),        G(w) in C^{3x3}   the addition's dynamic stiffness.

`G` constant and real is a memoryless perturbation of `(M, C, K)`. `G(w)` frequency dependent is a
real added degree of freedom, an absorber being the case of interest.

On an exact-bin multisine, Parseval makes the orthogonality condition per frequency line:

    <J_j, Delta>  =  SUM_w  Re[ Q(w)^H S_j(w)^H W G(w) Q(w) ],
    S_j(w) = dK_j - w^2 dM_j + i w dC_j,      W = M^-T D M^-1,   D = diag(1/std_x[3:6]^2).

Requiring this for EVERY excitation means requiring it for every `Q(w)` in C^3 at every line.
Since `Re[Q^H A Q] = Q^H Herm(A) Q` and a Hermitian form vanishes on all of C^3 only if the
matrix is zero, the requirement collapses to

    Herm( S_j(w)^H W G(w) ) = 0    for all ten j and every w.                       (*)

The `w` factors divide out of (*), so it is FREQUENCY INDEPENDENT, and the three families read:

    mass      Herm( (dM_j) W G ) = 0          i.e. (dM_j) W G is SKEW-Hermitian
    stiffness Herm( (dK)   W G ) = 0          i.e. (dK)   W G is SKEW-Hermitian
    damping   SkewHerm( (dC_j) W G ) = 0      i.e. (dC_j) W G is Hermitian

Ten matrix conditions, nine real equations each, on the eighteen real degrees of freedom of a
complex `G`. Counting says only `G = 0` survives, but the gantry's `dM`, `dC`, `dK` are full of
structural zeros, so counting is not an answer. This script builds the 90 x 18 real system and
computes its null space, at several `Y`, for four nested cases:

    case 1  G complex, unconstrained          (a general linear dynamic addition)
    case 2  G real, unconstrained             (memoryless, possibly non-symmetric)
    case 3  G real symmetric                  (a physical M_a, C_a or K_a perturbation)
    case 4  the DECLARED-EXCITATION contrast  (ten SCALAR conditions, not ten matrix ones)

Case 4 is the control: it must be solvable, and its solution space must be large. If it were not,
the whole framing would be wrong rather than the claim being strong.

Run:
  PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output \
      -n GraduationProject python -u prove_or_kill_impossibility.py
"""
__project_origin__ = "added"

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
for _p in (REPO, os.path.join(REPO, 'scripts', 'gantry')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_augmentation.systems import gantry_ss as gss        # noqa: E402
from gantry_dynamic.config import RunConfig                    # noqa: E402
from gantry_dynamic.data import compute_normalization, load_datasets   # noqa: E402

LINE = '=' * 90
COMBO_INIT_DETUNE = [1.1, 1.1, 0.9, 1.1, 0.9, 0.9, 1.1, 1.1, 0.9, 1.1]

mh = float(gss.mh); d = float(gss.d); Lb = float(gss.Lb)
m1 = float(gss.m1); m2 = float(gss.m2); mb = float(gss.mb)
Jb = float(gss.Jb); Jh = float(gss.Jh)
m_total = m1 + m2 + mb
m_diff = m1 - m2
J_eff = Jb + Jh + (m1 + m2) * Lb ** 2 / 4


def banner(t):
    print('\n' + LINE); print(t); print(LINE, flush=True)


def E(i, j):
    A = np.zeros((3, 3)); A[i, j] = 1.0; return A


def sym(i, j):
    return E(i, j) + E(j, i) if i != j else E(i, i)


def M_of(Y):
    """The baseline 3x3 mass matrix at scheduling value Y (absorber absent)."""
    off = m_diff * Lb / 2 - mh * Y
    return np.array([[m_total + mh, off, 0.0],
                     [off, J_eff + mh * d ** 2 + mh * Y ** 2, -mh * d],
                     [0.0, -mh * d, mh]], dtype=np.float64)


def dM_list(Y):
    """d M / d combo, in the order mh, m_total, m_diff, J_eff, d."""
    dmh = np.array([[1.0, -Y, 0.0],
                    [-Y, d ** 2 + Y ** 2, -d],
                    [0.0, -d, 1.0]])
    dmt = E(0, 0)
    dmd = (Lb / 2) * sym(0, 1)
    dJe = E(1, 1)
    dd = np.array([[0.0, 0.0, 0.0],
                   [0.0, 2 * mh * d, -mh],
                   [0.0, -mh, 0.0]])
    return [('mh', dmh), ('m_total', dmt), ('m_diff', dmd), ('J_eff', dJe), ('d', dd)]


def dC_list():
    """d C / d combo, in the order cg1, cg2, cy, cb_sum."""
    dcg1 = np.array([[1.0, Lb / 2, 0.0], [Lb / 2, Lb ** 2 / 4, 0.0], [0.0, 0.0, 0.0]])
    dcg2 = np.array([[1.0, -Lb / 2, 0.0], [-Lb / 2, Lb ** 2 / 4, 0.0], [0.0, 0.0, 0.0]])
    return [('cg1', dcg1), ('cg2', dcg2), ('cy', E(2, 2)), ('cb_sum', E(1, 1))]


def dK_list():
    return [('kb_sum', E(1, 1))]


# ------------------------------------------------------------------ the linear system
def rows_for(A, want_hermitian):
    """Real equations expressing Herm(A G) = 0 (want_hermitian False) or SkewHerm(A G) = 0.

    `G = Gr + i Gi`. For a real `A`:
        A G      = A Gr + i A Gi
        (A G)^H  = Gr^T A^T - i Gi^T A^T
    Herm part zero   <=>  A Gr + Gr^T A^T = 0  AND  A Gi - Gi^T A^T = 0
    SkewHerm zero    <=>  A Gr - Gr^T A^T = 0  AND  A Gi + Gi^T A^T = 0
    Each is 9 real equations in the 18 unknowns (Gr, Gi) stacked column-major.
    """
    # X = A G = A Gr + i A Gi,   X^H = G^H A^T = Gr^T A^T - i Gi^T A^T
    #   X + X^H = [A Gr + Gr^T A^T] + i [A Gi - Gi^T A^T]     (Herm part zero)
    #   X - X^H = [A Gr - Gr^T A^T] + i [A Gi + Gi^T A^T]     (SkewHerm part zero)
    # so the REAL block takes s = +1 for the Hermitian case and -1 for the other, and the
    # IMAGINARY block takes -s. An inverted sign here silently SWAPS the two conditions between
    # the parameter families and changes the answer, so `rows_for` carries a self-test below.
    eqs = []
    s = 1.0 if want_hermitian else -1.0          # sign on the transpose for the REAL block
    for a in range(3):
        for b in range(3):
            # real block
            row = np.zeros(18)
            for k in range(3):
                row[k * 3 + b] += A[a, k]                    # (A Gr)[a,b]
                row[k * 3 + a] += s * A[b, k]                # +/- (Gr^T A^T)[a,b]
            eqs.append(row)
            # imaginary block, opposite sign
            row = np.zeros(18)
            for k in range(3):
                row[9 + k * 3 + b] += A[a, k]
                row[9 + k * 3 + a] += -s * A[b, k]
            eqs.append(row)
    return np.array(eqs)


def selftest_rows_for(n_trials=200, seed=3):
    """`rows_for` against a direct complex evaluation. This is not optional.

    The first version of this file had the sign inverted, which swapped the Hermitian and
    skew-Hermitian conditions between the parameter families and produced a confident, wrong
    verdict. The bug was invisible in the answer and obvious to this test.
    """
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_trials):
        A = rng.standard_normal((3, 3))
        Gr, Gi = rng.standard_normal((3, 3)), rng.standard_normal((3, 3))
        g = np.concatenate([Gr.reshape(-1), Gi.reshape(-1)])
        G = Gr + 1j * Gi
        for herm in (True, False):
            X = A @ G
            target = (X + X.conj().T) if herm else (X - X.conj().T)
            got = (rows_for(A, want_hermitian=herm) @ g).reshape(3, 3, 2)
            ref = np.stack([target.real, target.imag], axis=-1)
            worst = max(worst, float(np.abs(got - ref).max()))
    return worst


def build_system(Y, W):
    """The stacked real system for (*) at scheduling value Y."""
    blocks, labels = [], []
    for name, dm in dM_list(Y):
        blocks.append(rows_for(dm @ W, want_hermitian=True)); labels.append('mass ' + name)
    for name, dk in dK_list():
        blocks.append(rows_for(dk @ W, want_hermitian=True)); labels.append('stiff ' + name)
    for name, dc in dC_list():
        blocks.append(rows_for(dc @ W, want_hermitian=False)); labels.append('damp ' + name)
    return np.vstack(blocks), labels


def nullspace_dim(A, tol_rel=1e-10):
    s = np.linalg.svd(A, compute_uv=False)
    if len(s) == 0 or s[0] == 0:
        return A.shape[1], s
    return int((s < s[0] * tol_rel).sum()) + (A.shape[1] - len(s)), s


def main():
    cfg = RunConfig(
        mode='augmentation_ma50_b140-230_a6_z03', encoder_init='linear_map',
        joint_estimation=True, physics_parameterization='reduced',
        combo_init_detune=COMBO_INIT_DETUNE, param_init_detune=None,
        obc=True, obc_space='tangent', nx_ann=8, up_sample=1, stride=10,
        fs_new=4000, snr=None, save_flag=False)
    banner('SELF-TEST of the linear map (a sign error here manufactures a false verdict)')
    worst = selftest_rows_for()
    print('  rows_for vs direct complex evaluation, worst abs error over 400 cases: %.3e' % worst)
    assert worst < 1e-12, 'rows_for is wrong; the verdict below would be meaningless'
    print('  PASS')

    banner('SETUP')
    data = load_datasets(cfg)
    norm = compute_normalization(cfg, data)
    std_x = np.asarray(norm.std_x, np.float64).reshape(6)
    D = np.diag(1.0 / std_x[3:6] ** 2)
    print('  D = diag(1/std_x[3:6]^2) = %s' % np.array2string(np.diag(D), precision=4))
    print('  parameters: mh %.4g  d %.4g  Lb %.4g  m_total %.4g  m_diff %.4g  J_eff %.6g'
          % (mh, d, Lb, m_total, m_diff, J_eff))

    banner('STRUCTURAL PRELIMINARY: what the parameters CANNOT reach in (M, C, K)')
    for tag, lst in (('dM', dM_list(0.15)), ('dC', dC_list()), ('dK', dK_list())):
        V = np.array([m[np.triu_indices(3)] for _n, m in lst])     # 6 upper-tri entries
        r = np.linalg.matrix_rank(V, tol=1e-10)
        print('  %s spans rank %d of 6 in the symmetric 3x3 space' % (tag, r))
    print('  so 6-1 + 6-4 + 6-5 = 1 + 2 + 5 = 8 structural directions are unreachable.')
    print('  Unreachable: mass X-Y coupling; damping X-Y and Theta-Y; all of K but the yaw entry.')
    print('  NOTE: unreachable is NOT the same as orthogonal. That is what (*) decides.')

    banner('THE TEST: null space of the ten matrix conditions (*)')
    print('  A nonzero null space IS the construction. A trivial one PROVES the impossibility.\n')
    print('  %8s %10s %10s %10s %10s %14s' % ('Y [m]', 'case1 cplx', 'case2 real', 'case3 sym',
                                              'rank', 'smallest sv'))
    verdict_nontrivial = False
    for Y in (-0.30, -0.15, 0.0, 0.15, 0.30):
        W = np.linalg.inv(M_of(Y)).T @ D @ np.linalg.inv(M_of(Y))
        A, _lab = build_system(Y, W)
        n1, s1 = nullspace_dim(A)
        # case 2: G real  ->  drop the imaginary unknowns (columns 9..17)
        n2, _ = nullspace_dim(A[:, :9])
        # case 3: G real symmetric -> 6 dof, map them into the 9 real entries
        Bsym = np.zeros((9, 6))
        for c, (i, j) in enumerate(zip(*np.triu_indices(3))):
            Msym = sym(i, j)
            Bsym[:, c] = Msym.reshape(-1)
        n3, _ = nullspace_dim(A[:, :9] @ Bsym)
        rank = np.linalg.matrix_rank(A, tol=1e-10)
        print('  %8.2f %10d %10d %10d %10d %14.4e'
              % (Y, n1, n2, n3, rank, s1[min(len(s1) - 1, 17)]))
        verdict_nontrivial |= (n1 > 0)

    banner('CONTROL (case 4): the DECLARED-EXCITATION problem, for contrast')
    print('  If orthogonality is required only for ONE given excitation rather than for every')
    print('  excitation, the ten MATRIX conditions collapse to ten SCALAR conditions, because')
    print('  each becomes a single number summed over the record. Ten scalars on eighteen real')
    print('  unknowns leaves an 8-dimensional solution space at least. This is the control: it')
    print('  must be large, or the framing itself is wrong.')
    Y = 0.15
    W = np.linalg.inv(M_of(Y)).T @ D @ np.linalg.inv(M_of(Y))
    rng = np.random.default_rng(0)
    Qs = rng.standard_normal((400, 3)) + 1j * rng.standard_normal((400, 3))   # a fake spectrum
    rows = []
    for _n, dm in dM_list(Y):
        rows.append(('mass', dm @ W, True))
    for _n, dk in dK_list():
        rows.append(('stiff', dk @ W, True))
    for _n, dc in dC_list():
        rows.append(('damp', dc @ W, False))
    Asc = np.zeros((10, 18))
    for r, (_t, A_, herm) in enumerate(rows):
        for u in range(18):
            g = np.zeros(18); g[u] = 1.0
            G = g[:9].reshape(3, 3) + 1j * g[9:].reshape(3, 3)
            X = A_ @ G if herm else (-1j) * (A_ @ G)
            Asc[r, u] = float(np.real(np.einsum('ki,ij,kj->', Qs.conj(), X, Qs)))
    n4, _ = nullspace_dim(Asc)
    print('\n  declared-excitation null space dimension: %d of 18' % n4)

    banner('VERDICT')
    if verdict_nontrivial:
        print('  A NONZERO G satisfies all ten matrix conditions. The impossibility claim is')
        print('  KILLED, and the null-space basis above IS the admissible addition class.')
    else:
        print('  The only G satisfying all ten conditions at every scheduling point is G = 0.')
        print('  IMPOSSIBILITY PROVEN, for this baseline and this parameterisation:')
        print()
        print('    No linear added dynamics, and no memoryless (M_a, C_a, K_a) perturbation,')
        print('    is orthogonal to the baseline parameter sensitivities for EVERY excitation.')
        print()
        print('  Consequences, and they are the useful part:')
        print('   1. Orthogonality by construction, independent of the experiment, is NOT')
        print('      available. Any orthogonal addition must come with a declared excitation.')
        print('   2. The contrast with case 4 says where the freedom actually lives: an')
        print('      8-dimensional solution space once the excitation is fixed, against nothing')
        print('      when it is not. Addition design and excitation design are ONE problem.')
        print('   3. This is very likely why Gyorok et al. Sect. 5.2 concede their construction')
        print('      breaks once the regressor carries past outputs and say a different data')
        print('      acquisition would be required. The obstruction is structural, not an')
        print('      artefact of their example.')


if __name__ == '__main__':
    main()
