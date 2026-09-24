"""G2 screen, route C: 4 lossless tuned absorbers + the physical mount (OA-007).

Addition:  f_a = -SUM_i m_i b_i(Y) d_i'' - (k_x, k_y, k_xp, gamma terms),
           d_i'' + w_i^2 d_i = -b_i(Y)^T qdd,  b_i(Y) = [cos a_i, l_i - Y cos a_i, sin a_i].
Unknowns:  geometry g = (f_i, a_i, l_i), i = 1..4  (nonlinear),  v = (m_1..m_4, mount) (linear).
Condition: A(g) v = 0 with A = J^T [Delta_abs_1..4, Delta_mount] (production discrete Jacobian).
For a geometry the best v is the generalised eigenvector of (Q, Gram) (min rho*); the geometry
minimises that rho* plus a penalty on negative masses. Multi-start Powell, strided reference set.
Env: OA_MODE, OBC_REF_STRIDE (search stride), OA_BAND ('in' | 'out'), OA_STARTS, OA_SEED,
     OA_MAXFEV, OA_TAG. Writes outputs/screen_c_<tag>.json and design/route_c_member_<tag>.npz.
"""
__project_origin__ = "added"

import json
import os
import sys
import time

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from oa_core import Problem, COMBO_NAMES, OA           # noqa: E402
import columns as colib                                # noqa: E402

N_ABS = 4
DELTA_MIN_STRIDE1 = 8.49          # OA-004
BANDS = {'in': [(140.0, 230.0)], 'out': [(20.0, 130.0), (240.0, 600.0)]}


class RouteC:
    def __init__(self, pb, band):
        self.pb = pb
        self.band = BANDS[band]
        mount = colib.physical_mount(pb)
        self.mount_names = list(mount)
        self.Dm = np.stack([pb.delta_of(f) for f in mount.values()], axis=1)   # (n6, 4)
        self.Am = np.stack([pb.stats(self.Dm[:, c])[0] for c in range(4)], axis=1)
        self.Um = np.stack([pb.stats(self.Dm[:, c])[1] for c in range(4)], axis=1)
        self.nfev = 0

    def unpack(self, g):
        """g -> [(f_hz, alpha, lever)]; frequency through a squashed map into its family."""
        geo = []
        for i in range(N_ABS):
            s, a, l = g[3 * i], g[3 * i + 1], g[3 * i + 2]
            lo, hi = self.band[i % len(self.band)]
            f = lo * (hi / lo) ** (0.5 * (1.0 + np.tanh(s)))          # log-uniform in [lo, hi]
            geo.append((f, a, 0.4 * np.tanh(l)))                     # |lever| < 0.4 m
        return geo

    def columns(self, geo):
        D = []
        for f, a, l in geo:
            b0, b1 = colib.absorber_b(a, l)
            D.append(self.pb.delta_of(self.pb.absorber_force(f, b0, b1)))
        Da = np.stack(D, axis=1)                                     # (n6, 4)
        Dall = np.concatenate([Da, self.Dm], axis=1)                 # (n6, 8)
        Aa = np.stack([self.pb.stats(Da[:, c])[0] for c in range(N_ABS)], axis=1)
        Ua = np.stack([self.pb.stats(Da[:, c])[1] for c in range(N_ABS)], axis=1)
        A = np.concatenate([Aa, self.Am], axis=1)
        U = np.concatenate([Ua, self.Um], axis=1)
        Gram = Dall.T @ Dall
        return A, U, Gram, Dall

    def best_v(self, U, Gram):
        d = np.sqrt(np.diag(Gram))
        Q = (U / d).T @ (U / d)
        G = Gram / np.outer(d, d)
        w, V = eigh(Q, G)
        v = V[:, 0] / d
        if v[:N_ABS].sum() < 0:
            v = -v
        return float(np.sqrt(max(w[0], 0.0))), v

    def objective(self, g):
        self.nfev += 1
        geo = self.unpack(g)
        A, U, Gram, _ = self.columns(geo)
        rho, v = self.best_v(U, Gram)
        m = v[:N_ABS]
        neg = np.minimum(m, 0.0).sum() / max(np.abs(m).sum(), 1e-300)
        return rho + 10.0 * neg * neg


def main():
    t0 = time.perf_counter()
    band = os.environ.get('OA_BAND', 'out')
    n_starts = int(os.environ.get('OA_STARTS', '4'))
    maxfev = int(os.environ.get('OA_MAXFEV', '1500'))
    rng = np.random.default_rng(int(os.environ.get('OA_SEED', '0')))
    pb = Problem()
    tag = os.environ.get('OA_TAG', 'c_%s_%s_s%d' % (band, pb.mode, pb.stride))
    rc = RouteC(pb, band)
    print('[C] band family %s %s, %d starts, maxfev %d' % (band, rc.band, n_starts, maxfev), flush=True)

    best = None
    for st in range(n_starts):
        g0 = np.concatenate([[rng.normal(0, 1), rng.uniform(-np.pi, np.pi), rng.normal(0, 0.7)]
                             for _ in range(N_ABS)])
        t1 = time.perf_counter(); rc.nfev = 0
        res = minimize(rc.objective, g0, method='Powell',
                       options=dict(maxfev=maxfev, xtol=1e-6, ftol=1e-12))
        print('  start %d: objective %.4e after %d evaluations, %.0f s'
              % (st, res.fun, rc.nfev, time.perf_counter() - t1), flush=True)
        if best is None or res.fun < best[0]:
            best = (res.fun, res.x)

    geo = rc.unpack(best[1])
    A, U, Gram, Dall = rc.columns(geo)
    rho, v = rc.best_v(U, Gram)
    D = Dall @ v
    dn = float(np.linalg.norm(D))
    dmin = DELTA_MIN_STRIDE1 / np.sqrt(pb.stride)
    s = 1.5 * dmin / dn
    rho_chk = pb.rho(D)
    pct, own = pb.bias_pct(s * D)
    print('\n[C] best: rho* %.4e (check %.4e), masses %s kg, mount %s'
          % (rho, rho_chk, np.array2string(s * v[:N_ABS], precision=4),
             dict(zip(rc.mount_names, np.round(s * v[N_ABS:], 6)))))
    for i, (f, a, l) in enumerate(geo):
        print('    absorber %d: f %.2f Hz  alpha %+.1f deg  lever %+.4f m  mass %.4f kg'
              % (i, f, np.degrees(a), l, s * v[i]))
    print('    ||Delta|| scaled %.4e (minimum %.4e at this stride)' % (s * dn, dmin))
    print('    bias J^+ Delta (combo %%): %s  m_diff own %+.3e %%'
          % ('  '.join('%s %+.2e' % (n_, p) for n_, p in zip(COMBO_NAMES, pct)), own))
    mh = 10.1
    print('    total absorber mass %.3f kg = %.1f %% of the payload'
          % (s * v[:N_ABS].sum(), 100 * s * v[:N_ABS].sum() / mh))
    out = dict(mode=pb.mode, stride=pb.stride, band=band, rho=rho, rho_check=rho_chk,
               geometry=[[float(f), float(a), float(l)] for f, a, l in geo],
               masses=(s * v[:N_ABS]).tolist(), mount=dict(zip(rc.mount_names, (s * v[N_ABS:]).tolist())),
               delta_norm=s * dn, delta_min=dmin, bias_pct=pct.tolist(), m_diff_own=own,
               masses_positive=bool(np.all(v[:N_ABS] > 0)), seconds=time.perf_counter() - t0)
    with open(os.path.join(OA, 'outputs', 'screen_c_%s.json' % tag), 'w') as fh:
        json.dump(out, fh, indent=2)
    np.savez(os.path.join(HERE, 'route_c_member_%s.npz' % tag), geometry=np.array(out['geometry']),
             masses=np.array(out['masses']), mount=np.array(s * v[N_ABS:]),
             mount_names=np.array(rc.mount_names))
    print('[out] outputs/screen_c_%s.json  (%.0f s)' % (tag, time.perf_counter() - t0))


if __name__ == '__main__':
    main()
