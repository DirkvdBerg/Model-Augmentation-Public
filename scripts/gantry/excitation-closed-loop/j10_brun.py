"""J10: T4 for the detuned case, Brun collinearity of the ten combinations on the planned data (EXCITATION-VALIDATION.md s18).

Data: the 18 training references of DATA-DESIGN.md 7.1 at 20 kHz (matlab/ref_planned.m with out20; path as the
first argument), the chosen multisine (D-219: 106 to 297 Hz on a 0.5 Hz grid, A_prod per logical channel) on every
record except the ILC shapes, controller K1, noiseless. Sensitivities as J4 (j4_noiseless.py): baseline closed loop
(vendored replica) at a base point and with each combination +10 %, S_i = e_i - e_base on the three stage channels,
unweighted; Gram matrix summed over records (Parseval). Base points: theta_0 (primary) and the D-191 detuned start.
Writes outputs/j10_brun.json.
"""
__project_origin__ = "added"

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.io import loadmat

import common as C
import replica as R

T0 = time.perf_counter()
BAND, DF = (106.0, 297.0), 0.5              # D-219
A_PROD = np.array([240.0, 87.0, 180.0])     # logical rms [N, N m, N] (DATA-DESIGN.md 0)
GAMMA_MAX = 5.0                             # T4 (s15b): lower end of Brun's critical range 5 to 20 (their experience)
INFO_BANDS = [(0.0, 106.0), (106.0, 297.0), (297.0, 10000.0)]
NAMES = C.COMBO_NAMES

d = loadmat(sys.argv[1], simplify_cells=True)
R20, ids = np.asarray(d['R20'], np.float32), [str(s) for s in d['ids']]
n_rec, N = R20.shape[0], R20.shape[1]
assert N == R.N20
FK = np.fft.rfftfreq(N, C.TS)[1:N // 2]


def multisine(seed):
    """Random-phase multisine per logical channel on the D-219 lines, rms A_prod, to stage force.
    HEURISTIC: no crest-factor selection (gtd_make_multisine keeps the best of 30 draws; same spectrum)."""
    rng = np.random.default_rng(seed)
    k = np.round(np.arange(BAND[0], BAND[1] + DF / 2, DF) * N * C.TS).astype(int)   # DFT bins of the 12 s record
    X = np.zeros((N // 2 + 1, 3), complex)
    X[k] = np.exp(1j * 2 * np.pi * rng.random((len(k), 3)))
    x = np.fft.irfft(X, n=N, axis=0)
    x = x / np.sqrt(np.mean(x ** 2, axis=0)) * A_PROD
    return (x @ C.PINV.T).astype(np.float32)                    # logical -> stage (gtd_logical_to_stage)


BASES = {'theta0': C.THETA0, 'detuned': C.THETA0 * C.DETUNE_PROD}
THETAS = {}
for b, th0 in BASES.items():
    THETAS[(b, 'base')] = th0
    for i, nm in enumerate(NAMES):
        th = th0.copy(); th[i] *= 1.10
        THETAS[(b, nm)] = th


def gram(sims, b):
    """per-line Gram of the ten sensitivities (THEORY: Gauss-Newton normal matrix of an unweighted
    output-error loss, Parseval, as J4), summed over lines and split by band for the diagonal"""
    S = np.stack([np.fft.rfft(sims[(b, 'base')] - sims[(b, nm)], axis=0)[1:N // 2] for nm in NAMES], axis=2)
    dens = 2.0 * np.real(np.einsum('kim,kin->kmn', np.conj(S), S)) / N
    diag_b = np.array([[dens[(FK >= lo) & (FK < hi), i, i].sum() for lo, hi in INFO_BANDS] for i in range(10)])
    return dens.sum(0), diag_b


G = {}
for i, rid in enumerate(ids):
    t1 = time.perf_counter()
    r = R20[i]
    has_ms = not rid.startswith('TR-T')
    moving = bool(np.ptp(r, axis=0).max() > 0)
    f = multisine(7000 + i) if has_ms else np.zeros_like(r)
    jobs = [('all', key) for key in THETAS]
    if has_ms and moving:
        jobs += [('ref', key) for key in THETAS if key[0] == 'theta0']
    with ThreadPoolExecutor(4) as ex:
        ys = list(ex.map(lambda jk: R.simulate(r, f if jk[0] == 'all' else np.zeros_like(r), 0.0, THETAS[jk[1]])[0], jobs))
    sims = {}
    for (v, key), y in zip(jobs, ys):
        sims.setdefault(v, {})[key] = y
    for v, s in sims.items():
        for b in BASES:
            if (b, 'base') in s:
                G[(rid, v, b)] = gram(s, b)
    if has_ms and not moving:
        pass                                                    # standstill: reference alone gives nothing
    elif not has_ms:
        G[(rid, 'ref', 'theta0')] = G[(rid, 'all', 'theta0')]   # ILC shapes: the reference is all there is
    print('[J10] %-12s ms %-5s moving %-5s %d sims  %.1f s' % (rid, has_ms, moving, len(jobs), time.perf_counter() - t1), flush=True)
    del sims, ys


def sep(Gm):
    dg = np.sqrt(np.diag(Gm))
    if np.any(dg <= 0):
        return None
    Gn = Gm / np.outer(dg, dg)
    lam = np.linalg.eigvalsh(Gn)
    ri = 1.0 / np.sqrt(np.diag(np.linalg.inv(Gn)))              # HEURISTIC read-out: independent share (as fig8_detuned.py)
    off = np.abs(Gn - np.eye(10)); a, b = np.unravel_index(np.argmax(off), off.shape)
    gamma = 1.0 / np.sqrt(lam[0])                               # THEORY: Brun, Reichert, Kuensch 2001 Eq. 13
    vmin = np.linalg.eigh(Gn)[1][:, 0]
    return dict(gamma=float(gamma), r_i={n: float(x) for n, x in zip(NAMES, ri)},
                worst_pair=[NAMES[a], NAMES[b], float(Gn[a, b])],
                weakest_direction={n: round(float(x), 3) for n, x in zip(NAMES, vmin)})


standstill = [r for r in ids if r.startswith('TR-S')]
sets = {'planned data, at theta_0 (T4)': [(r, 'all', 'theta0') for r in ids],
        'planned data, at the detuned start': [(r, 'all', 'detuned') for r in ids],
        'references alone (no multisine), at theta_0': [(r, 'ref', 'theta0') for r in ids if (r, 'ref', 'theta0') in G],
        'multisine alone (standstill records), at theta_0': [(r, 'all', 'theta0') for r in standstill]}
out = dict(gamma_max=GAMMA_MAX, sets={})
for s, keys in sets.items():
    Gs = sum(G[k][0] for k in keys)
    res = sep(Gs)
    out['sets'][s] = res
    if res is None:
        print('\n%s: no information on some combination' % s); continue
    print('\n%s: gamma %.2f -> %s' % (s, res['gamma'], 'PASS' if res['gamma'] < GAMMA_MAX else 'FAIL'))
    print('  independent share r_i: ' + ', '.join('%s %.2f' % (n, x) for n, x in res['r_i'].items()))
    print('  most collinear pair: %s / %s, normalised Gram %+.3f' % tuple(res['worst_pair']))
    print('  weakest direction (eigenvector of the smallest eigenvalue): %s' % res['weakest_direction'])
Db = sum(G[(r, 'all', 'theta0')][1] for r in ids)
Db = Db / Db.sum(1, keepdims=True)
out['info_share_by_band_theta0'] = {n: {'%g-%g' % bb: float(Db[i, j]) for j, bb in enumerate(INFO_BANDS)} for i, n in enumerate(NAMES)}
print('\ninformation share per combination (diagonal Gram), planned data at theta_0, bands %s Hz:' % INFO_BANDS)
for i, n in enumerate(NAMES):
    print('  %-8s %s' % (n, ' '.join('%5.2f' % x for x in Db[i])))
with open(os.path.join(C.OUT, 'j10_brun.json'), 'w') as fh:
    json.dump(out, fh, indent=1)
print('\n[J10] wrote outputs/j10_brun.json, %.0f s' % (time.perf_counter() - T0), flush=True)
