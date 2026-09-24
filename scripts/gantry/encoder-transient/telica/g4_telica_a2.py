"""G4 attempt 2 (ET-010): the Telica map built from the model the Telica encoder FEEDS.

The Telica pipeline's physical block is GantryFrictionBlock with the RECOVERED attempt-2 parameters
and tanh Coulomb friction; attempt 1 built the map from the datasheet constants without friction.
Here: map = model_map on that block (cc = 0, Jacobians at rest at each Y); friction = the block's
own input correction u_eff = u - cc tanh(v_stage / v0), v_stage from the measured window positions.
Reference, bounds, scoring and PASS rule: unchanged from attempt 1 (g4_telica.py, ET-006/ET-009).
Output: outputs/telica/g4/g4_telica_a2.json and g2_telica_a2_encoder.pt
"""
__project_origin__ = "added"

import copy
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g4_telica as g4                                           # noqa: E402  (vendor_telica first)
import core                                                      # noqa: E402
import model_map as mm                                           # noqa: E402
import tr_env                                                    # noqa: E402
import rate                                                      # noqa: E402
from friction.karnopp import GantryFrictionBlock, rebuild_float64   # noqa: E402
from pipeline.telica_model import baseline_params               # noqa: E402
from pipeline.real_data import frozen_norm                       # noqa: E402
from params import telica_params as tp                           # noqa: E402

DEGS = list(range(0, 11))


class BlockHfn(torch.nn.Module):
    """The pipeline's physical block alone, with the Interconnect's (x, u) -> (y, x_next) API."""

    def __init__(self, blk, Cn):
        super().__init__()
        self.blk = blk
        self.register_buffer('Cn', torch.as_tensor(np.asarray(Cn, float)))

    def output_only(self, x, u=None):
        return x @ self.Cn.T

    def forward(self, x, u):
        xn = self.blk(torch.cat([x, u], 1).unsqueeze(-1)).squeeze(-1)
        return self.output_only(x), xn


def recovered_block(norm, dt):
    """As telica_model.build_model_real's physical block (recovered raw14, float64), cc = 0."""
    raw, cc = baseline_params('_a2')
    phy = GantryFrictionBlock(cc=(0.0, 0.0, 0.0), mode='tanh', Y_op=None, std_x=norm.std_x,
                              std_u=norm.std_u, x_mean=norm.x_mean, u_mean=norm.u_mean,
                              Ts=dt, up_sample=1)
    phy = rebuild_float64(phy, p=raw)
    f64 = lambda a, shp: torch.as_tensor(np.asarray(a, np.float64)).reshape(shp)   # noqa: E731
    phy.std_x = f64(norm.std_x, (6, 1)); phy.std_x_1d = f64(norm.std_x, (6,))
    phy.std_u = f64(norm.std_u, (3, 1)); phy.x_mean = f64(norm.x_mean, (6, 1))
    phy.u_mean = f64(norm.u_mean, (3, 1))
    phy.early_exit = False
    return phy.to(torch.float64).eval(), np.asarray(cc, float)


class FrictionCompEncoder(torch.nn.Module):
    """u_eff = u - cc tanh(v_stage / v0) on the window, v_stage = torch.gradient of the MEASURED
    window positions (HEURISTIC estimator, ET-010), then the inner encoder. Deployable."""

    def __init__(self, inner, cc, v0, dt, norm):
        super().__init__()
        self.inner = inner
        t = lambda a: torch.as_tensor(np.asarray(a, float).ravel())   # noqa: E731
        self.register_buffer('cc', t(cc)); self.register_buffer('y0', t(norm.y0))
        self.register_buffer('ystd', t(norm.ystd)); self.register_buffer('um', t(norm.u_mean))
        self.register_buffer('su', t(norm.std_u))
        self.v0, self.dt = float(v0), float(dt)

    def forward(self, uhist, yhist):
        y = yhist * self.ystd + self.y0
        v = torch.gradient(y, spacing=self.dt, dim=1)[0]
        u = uhist * self.su + self.um - self.cc * torch.tanh(v / self.v0)
        return self.inner((u - self.um) / self.su, yhist)


def fric_np(yw, uw, cc, v0, dt):
    return uw - cc * np.tanh(np.gradient(yw, dt, axis=1) / v0)


def main():
    t0 = time.time()
    tr_env.check_no_leak()
    norm = frozen_norm(g4._Cfg)
    for k in ('x_mean', 'std_x', 'std_u', 'u_mean'):
        setattr(norm, k, np.asarray(getattr(norm, k), float))
    dt = float(rate.TS5)
    sx = norm.std_x.ravel(); su = norm.std_u.ravel(); sy = np.asarray(norm.ystd, float)
    recs = g4.load_all()
    posb, refs = [], []
    for s, op, ypos, d, d5 in recs:
        pb, bo = g4.bounds(d, g4.FC_GRID)
        posb.append(pb); refs.append(bo)
    dist = g4.pooled_distortion(refs, g4.FC_GRID)
    for bo in refs:
        for fc in g4.FC_GRID:
            bo[fc]['bound'] = np.sqrt(bo[fc]['noise'] ** 2 + dist[fc] ** 2)
    pooled_fc = {fc: np.sqrt(np.mean([bo[fc]['bound'] ** 2 for bo in refs], 0)) for fc in g4.FC_GRID}
    fc_best = [min(g4.FC_GRID, key=lambda fc: pooled_fc[fc][j]) for j in range(3)]
    print(f'  reference as attempt 1: fc {fc_best}  [{time.time() - t0:.0f} s]')

    # ==== the model the encoder feeds ====
    blk, cc = recovered_block(norm, dt)
    hfn = BlockHfn(blk, norm.Cd_norm)
    grid, off = mm.model_grid(hfn, norm, 6, g4.NA)
    v0 = float(tp.V0_TANH)
    print(f'  recovered-block Jacobian grid built: max affine offset at rest {off:.1e}; cc {cc.round(1)} N, '
          f'v0 {v0}  [{time.time() - t0:.0f} s]')
    p0 = g4.build_p0(norm, dt)
    W0 = grid[2]
    My = (g4.NA + 1) * 3
    parent_R = copy.deepcopy(p0)
    with torch.no_grad():
        parent_R.Wb_psi_y.copy_(torch.as_tensor(W0[:, :My])); parent_R.Wb_psi_u.copy_(torch.as_tensor(W0[:, My:]))
    # degree rule (truth-free, ET-006) on friction-compensated train windows
    tw = [g4.windows5(r[4], g4.NA) for r in recs if r[0] == 'train']
    twf = [(yw, fric_np(yw, uw, cc, v0, dt)) for yw, uw, _, _ in tw]
    x_r0 = [core.numpy_sched(np.zeros((0,)), W0, yw, uf, sx, su, sy) for yw, uf in twf]
    x_gr = [core.grid_apply(grid, yw, uf, yw[:, -1, 2], sx, su, sy) for yw, uf in twf]
    gain = np.sqrt(sum(((a - b) ** 2).sum(0) for a, b in zip(x_gr, x_r0)))
    deg_sel, deg_tab = None, {}
    for dg in DEGS:
        C, _, rr = core.fit_schedule(grid, dg)
        xp = [core.numpy_sched(C, W0, yw, uf, sx, su, sy) for yw, uf in twf]
        ratio = np.sqrt(sum(((a - b) ** 2).sum(0) for a, b in zip(xp, x_gr))) / np.maximum(gain, 1e-300)
        deg_tab[dg] = dict(ratio=ratio.tolist(), resid=rr)
        if deg_sel is None and np.all(ratio <= 0.1):
            deg_sel = dg
    rule_met = deg_sel is not None
    deg_sel = deg_sel if rule_met else min(DEGS, key=lambda dg: max(deg_tab[dg]['ratio']))
    C_R, _, _ = core.fit_schedule(grid, deg_sel)
    print(f'  recovered-map degree {deg_sel} (rule met {rule_met}); ratios at that degree '
          + ' '.join(f'{v:.3f}' for v in deg_tab[deg_sel]['ratio']))
    g2a1 = torch.load(os.path.join(g4.OUT, 'g2_telica_encoder.pt'), weights_only=False)
    encs = {
        'P0': p0,
        'G2a1': core.ScheduledReconEncoder(copy.deepcopy(p0), g2a1['C'], g4.NA, 3, 3, norm.y0, norm.ystd).double(),
        'R0': parent_R,
        'RS': core.ScheduledReconEncoder(copy.deepcopy(parent_R), C_R, g4.NA, 3, 3, norm.y0, norm.ystd).double(),
        'RF0': FrictionCompEncoder(copy.deepcopy(parent_R), cc, v0, dt, norm).double(),
        'RFS': FrictionCompEncoder(core.ScheduledReconEncoder(copy.deepcopy(parent_R), C_R, g4.NA, 3, 3,
                                                              norm.y0, norm.ystd), cc, v0, dt, norm).double(),
    }
    for e in encs.values():
        e.eval()
    # numpy/module consistency on one record (the deployable torch path vs the numpy map)
    yw, uw, _, _ = tw[0]
    xm_ = g4.enc_phys(encs['RFS'], yw, uw, norm)
    xn_ = core.numpy_sched(C_R, W0, yw, fric_np(yw, uw, cc, v0, dt), sx, su, sy)
    cons = float(np.abs(xm_ - xn_).max() / np.abs(xn_).max())
    print(f'  RFS module vs numpy map, max rel diff {cons:.1e}')

    names = list(encs)
    agg = {e: [np.zeros(6), 0] for e in names}
    bsq = np.zeros(6)
    by_y = {}
    nf = {e: [] for e in names}
    per = {}
    for (s, op, ypos, d, d5), bo, pb in zip(recs, refs, posb):
        yw, uw, k5, k20 = g4.windows5(d5, g4.NA)
        qref = (np.asarray(d['q1'], float) @ g4.PT_INV.T)[k20]
        vref = np.stack([bo[fc_best[j]]['v'][k20, j] for j in range(3)], 1)
        xref = np.concatenate([qref, vref], 1)
        bnd = np.concatenate([pb, [bo[fc_best[j]]['bound'][j] for j in range(3)]])
        bsq += bnd ** 2 * len(k5)
        ysw, usw = g4.standstill_windows5(d5, g4.NA)
        per[f'{s}/{op}'] = {}
        for e in names:
            ms = ((g4.enc_phys(encs[e], yw, uw, norm) - xref) ** 2).mean(0)
            agg[e][0] += ms * len(k5); agg[e][1] += len(k5)
            by_y.setdefault((ypos, e), [np.zeros(6), 0])
            by_y[(ypos, e)][0] += ms * len(k5); by_y[(ypos, e)][1] += len(k5)
            nf[e].append(g4.enc_phys(encs[e], ysw, usw, norm)[:, 3:].std(0))
            per[f'{s}/{op}'][e] = np.sqrt(ms).tolist()
    n = agg['P0'][1]
    pooled = {e: np.sqrt(agg[e][0] / n) for e in names}
    bpool = np.sqrt(bsq / n)
    resolvable = bpool < pooled['P0']
    ylev = sorted({r[2] for r in recs})
    per_y = {str(y): {e: np.sqrt(by_y[(y, e)][0] / by_y[(y, e)][1]).tolist() for e in names} for y in ylev}
    worst_y = max(float(np.max(np.asarray(per_y[str(y)]['RFS'])[resolvable]
                                / np.asarray(per_y[str(y)]['P0'])[resolvable])) for y in ylev)
    lower = bool(np.all(pooled['RFS'][resolvable] < pooled['P0'][resolvable]))
    PASS = bool(resolvable.any() and lower and worst_y <= 1.01)
    print(f'\n  pooled RMS vs reference, 15 records (bound {" ".join(f"{v:.2e}" for v in bpool)})')
    print(f'  {"enc":<6}' + ''.join(f'{c:>11}' for c in g4.CH))
    for e in names:
        print(f'  {e:<6}' + ''.join(f'{v:>11.3e}' for v in pooled[e]))
    print('  ratio to P0:')
    for e in names[1:]:
        print(f'  {e:<6}' + ''.join(f'{v:>11.4f}' for v in pooled[e] / pooled['P0']))
    print('\n  per ypos, RFS / P0:')
    for y in ylev:
        a, b = np.asarray(per_y[str(y)]['RFS']), np.asarray(per_y[str(y)]['P0'])
        print(f'    ypos {y:>6.0f}  ' + ' '.join(f'{c} {v:.4f}' for c, v in zip(g4.CH, a / b)))
    nfp = {e: np.sqrt((np.array(v) ** 2).mean(0)).tolist() for e, v in nf.items()}
    print('  encoder velocity noise floor on standstill windows: '
          + '  '.join(f'{e} ' + ' '.join(f'{v:.2e}' for v in nfp[e]) for e in ('P0', 'RFS')))
    print(f'\n  G4 attempt 2 verdict: resolvable {resolvable.tolist()}; RFS lower on all resolvable: {lower}; '
          f'worst per-ypos ratio {worst_y:.4f} (<= 1.01) -> {"PASS" if PASS else "FAIL"}')
    res = dict(fc=fc_best, deg=deg_sel, deg_rule_met=rule_met, deg_table={str(k): v for k, v in deg_tab.items()},
               affine_offset=off, cc=cc.tolist(), v0=v0, module_vs_numpy=cons,
               pooled={e: v.tolist() for e, v in pooled.items()}, bound_pooled=bpool.tolist(),
               resolvable=resolvable.tolist(), per_ypos=per_y, per_record=per, noise_floor=nfp,
               worst_ypos_ratio=worst_y, PASS=PASS)
    with open(os.path.join(g4.OUT, 'g4_telica_a2.json'), 'w') as f:
        json.dump(res, f, indent=1, default=float)
    torch.save(dict(W0=W0, C=C_R, deg=deg_sel, cc=cc, v0=v0, dt=dt,
                    rfs_state=encs['RFS'].state_dict()), os.path.join(g4.OUT, 'g2_telica_a2_encoder.pt'))
    print(f'  saved outputs/telica/g4/g4_telica_a2.json  [{time.time() - t0:.0f} s]')


if __name__ == '__main__':
    main()
