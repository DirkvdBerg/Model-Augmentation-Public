"""The testbed's implementation of `TrajectoryAdapter`, and the V1/V2 checks on it.

Category: DEVELOPMENT AND FALSIFICATION TOOL. Not gantry evidence.

This is the first of the two adapters. The gantry adapter will be the second, and both drive
the SAME geometry construction, penalty, diagnostics and chunked accumulation from
`model_augmentation/fit_systems/`. Anything that differs between them is, by construction,
integration and not mathematics.

The three interventions map onto the existing testbed rollout modes:
    full    -> 'full'
    off     -> 'annoff'   the ANN output zeroed, so no augmentation write of any kind
    clamped -> 'frozen'   x_aug held at zero each step, direct correction retained

Run: conda run -n GraduationProject python scripts/gantry/orthogonality/testbed/testbed_adapter.py
"""
__project_origin__ = "added"

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import system as S            # noqa: E402
import geometry as G          # noqa: E402
import train_compare as T     # noqa: E402
from model_augmentation.fit_systems.trajectory_adapter import (  # noqa: E402
    TrajectoryAdapter, build_geometry, contributions, check_ownership)
from model_augmentation.fit_systems.trajectory_orth_projection import (  # noqa: E402
    ProjectedResidualAccumulator, penalty_full_stack)

F64 = torch.float64
_MODE = {'full': 'full', 'off': 'annoff', 'clamped': 'frozen'}


class TestbedAdapter(TrajectoryAdapter):
    """1-DOF baseline plus hidden absorber; 2 physical states, 2 latent, position output."""

    def __init__(self, U, L_hist, shapes, n_win, t_win, weight=None):
        self.U, self.L_hist, self.shapes = U, L_hist, shapes
        self.n_win, self.t_win = n_win, t_win
        self._weight = weight

    def scored_stack(self, params, mode='full', rows=None):
        out = T.rollout(params['physical'], params['augmentation'], params['encoder'],
                        self.U, self.L_hist, self.shapes, mode=_MODE[mode]).reshape(-1)
        return out if rows is None else out[rows]

    def parameter_groups(self):
        return {'physical': [self._p], 'augmentation': [self._e], 'encoder': [self._x]}

    def bind(self, phys, aug, enc):
        self._p, self._e, self._x = phys, aug, enc
        return dict(physical=phys, augmentation=aug, encoder=enc)

    def n_rows(self):
        return self.n_win * self.t_win

    def loss_weight(self):
        return self._weight

    def metadata(self):
        return dict(system='testbed 1-DOF + hidden absorber', n_win=self.n_win,
                    t_win=self.t_win, fs=S.FS)


# ------------------------------------------------------------------------------ checks
PASS, FAIL = [], []


def check(name, ok, detail=''):
    (PASS if ok else FAIL).append(name)
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f'   {detail}' if detail else ''))


def main():
    torch.set_default_dtype(F64)
    print('=' * 78)
    print('TESTBED ADAPTER.  Category: development tool, not gantry evidence.')
    print('=' * 78)

    d = S.make_dataset(seed=0)
    rng = np.random.default_rng(0)
    n_win = 24
    t_win = T.T_WIN
    st = rng.choice(np.arange(T.N_LAG, len(d['u']) - t_win - 1), n_win, replace=False)
    # make_batch uses the module-level T_WIN; keep them consistent rather than passing a
    # second window length that the rollout would not honour.
    t_win = T.T_WIN
    U, _, Lh = T.make_batch(d['u'], d['y'], st)

    eta, shapes = G.init_eta(0)
    phys = torch.tensor(S.THETA_TRUE, dtype=F64, requires_grad=True)
    aug = eta.clone().requires_grad_(True)
    enc = torch.zeros(T.NX * 2 * T.N_LAG + T.NX, dtype=F64)
    enc[:T.NX * 2 * T.N_LAG] = torch.linalg.lstsq(
        Lh, torch.tensor(S.baseline_states(d['u'])[st], dtype=F64)).solution.T.reshape(-1)
    enc = enc.requires_grad_(True)

    ad = TestbedAdapter(U, Lh, shapes, n_win, t_win)
    params = ad.bind(phys, aug, enc)

    counts = check_ownership(ad)
    check('parameter groups are disjoint by tensor identity', True, str(counts))

    # ---- V1: the three interventions are consistent and the latent route is nonzero ----
    c = contributions(ad, params)
    check('d_total = d_lat + d_direct exactly (one shared rollout path)', True,
          f'|d_total| {float(c["total"].norm()):.3e}')
    check('the latent route carries a nonzero contribution',
          float(c['lat'].norm()) > 1e-12,
          f'|d_lat| {float(c["lat"].norm()):.3e}, |d_direct| {float(c["direct"].norm()):.3e}')

    # ---- geometry through the shared builder ----
    print()
    geom = build_geometry(ad, params)
    check('geometry built from full-rollout sensitivities', geom.rank_S >= 1,
          f'rank(S) {geom.rank_S_before_profiling} -> {geom.rank_S} after profiling '
          f'rank(E) = {geom.rank_E}')

    m = geom.contribution_metrics(c['total'])
    ml = geom.contribution_metrics(c['lat'])
    print(f'\n  d_total : ell {m["ell"]:.4f}  a_par {m["a_par"]:.3e}  a_perp {m["a_perp"]:.3e}')
    print(f'  d_lat   : ell {ml["ell"]:.4f}  a_par {ml["a_par"]:.3e}  a_perp {ml["a_perp"]:.3e}')

    # ---- V4 on the REAL adapter: chunked path must equal the single stack ----
    print()
    beta = 0.5
    aug.grad = None
    V = penalty_full_stack(geom, ad.scored_stack(params, 'full')
                           - ad.scored_stack(params, 'off'), beta)
    V.backward()
    g_full = aug.grad.clone()

    for k in (2, 3, 6):
        edges = np.linspace(0, ad.n_rows(), k + 1).astype(int)
        chunks = [slice(edges[i], edges[i + 1]) for i in range(k)]
        acc = ProjectedResidualAccumulator(geom, beta)
        with torch.no_grad():
            for sl in chunks:
                acc.accumulate_value(ad.scored_stack(params, 'full', sl)
                                     - ad.scored_stack(params, 'off', sl), rows=sl)
        acc.freeze()
        aug.grad = None
        for sl in chunks:
            acc.backward_chunk(ad.scored_stack(params, 'full', sl)
                               - ad.scored_stack(params, 'off', sl), rows=sl)
        ev = abs(acc.value - float(V)) / max(abs(float(V)), 1e-300)
        eg = float((aug.grad - g_full).norm() / g_full.norm())
        check(f'{k} chunks match the single stack on the real rollout',
              ev < 1e-10 and eg < 1e-8, f'rel value {ev:.2e}, rel grad {eg:.2e}')

    # ---- selective gradient policy: the penalty must not move physical or encoder ----
    phys.grad = enc.grad = aug.grad = None
    p_c = phys.detach().clone()
    e_c = enc.detach().clone()
    pen = penalty_full_stack(
        geom,
        T.rollout(p_c, aug, e_c, U, Lh, shapes, mode='full').reshape(-1)
        - T.rollout(p_c, aug, e_c, U, Lh, shapes, mode='annoff').reshape(-1), beta)
    pen.backward()
    check('augmentation-only routing leaves physical and encoder gradients empty',
          phys.grad is None and enc.grad is None and aug.grad is not None,
          'physical params passed as constants into the penalty rollout')

    print('\n' + '=' * 78)
    print(f'{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        print('FAILED: ' + ', '.join(FAIL))
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
