"""MIGRATION STEP 3 gate: `Interconnect.output_only` against the full forward.

`output_only` evaluates only the output signal's dependency cone, so a closed-loop rollout can
read `y = h(x)` before forming `u` and step the model ONCE per timestep. It replaces
`cl_plant.identify_output_map`, which reverse-engineered the same map with `nx + 1` probe forward
passes and then assumed it stayed affine and frozen. Those assumptions are now unnecessary, and
`identify_output_map` survives only as one of the checks below.

Four checks, each failing differently:

  1  CONE          the resolved cone is a strict subset of the full computation order, and the
                   blocks it excludes are named, so "it is cheaper" is visible rather than claimed.
  2  EXACTNESS     output_only(x, u) equals the full forward's y to MACHINE PRECISION on random
                   states, and on the same states with a large random u. Same graph, same
                   arithmetic, no reordering, so anything above 0 here is a real disagreement.
  3  NO FEEDTHROUGH  y(x, u=0) == y(x, u=1e3*randn) through the REAL interconnect. The wiring
                   passes u into the output block, so D_d = 0 is a property of the block's
                   coefficients and not of the graph; it has to be measured. If this ever fails,
                   the closed-loop step order is invalid and there is a genuine algebraic loop.
  4  AFFINE MAP    output_only agrees with the (C, b) that identify_output_map extracts, which is
                   what the previous implementation used. This is the bridge between the two
                   implementations and it is why the old code can be deleted rather than kept.

Usage:
  PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python -u cl_test_output_only.py
"""
__project_origin__ = "added"

import dataclasses
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
GANTRY = os.path.join(REPO, 'scripts', 'gantry')
for p in (REPO, GANTRY, HERE, os.path.join(GANTRY, 'drift-demo'),
          os.path.join(GANTRY, 'msd-offset')):
    if p not in sys.path:
        sys.path.insert(0, p)

import demo_common as dm                                                  # noqa: E402
from demo_common import CFG                                               # noqa: E402
import cl_plant as PLANT                                                  # noqa: E402

torch.set_num_threads(int(os.environ.get('CL_THREADS', 1)))
TOL_EXACT = 0.0          # same graph, same order: bit-identical or it is wrong
TOL_FEEDTHROUGH = 1e-30  # y must not move at all when u changes by 1e3
TOL_AFFINE = 1e-6        # the identified map is float32 arithmetic, not the same graph

print('=' * 96)
print('MIGRATION STEP 3: output_only against the full forward')
print('=' * 96)
cfg = dataclasses.replace(CFG, seed=0, ann_route_ix=tuple(range(8)), lr=1e-7)
fs, norm, K0, na, nb, na_r, nb_r = dm.build_pipeline(cfg=cfg, verbose=False)
nx, nu, ny = cfg.nx_phys + cfg.nx_ann, cfg.nu, cfg.ny
hfn = fs.hfn

# one forward so the graph is initialised and the cone is resolved
with torch.no_grad():
    hfn(torch.zeros(1, nx, dtype=cfg.dtype_pt), torch.zeros(1, nu, dtype=cfg.dtype_pt))

ok = True

# ---- 1. the cone ------------------------------------------------------------------------------
names = [type(m).__name__ for m in hfn.connected_blocks]
cone = hfn.output_cone
full = tuple(hfn.order_output_signal_computation)
skipped = [k for k in full if k not in cone]
print('\n1  CONE')
print('   full computation order  %s' % (full,))
print('   output cone             %s' % (cone,))
print('   blocks evaluated        %s'
      % [names[k - 2] for k in cone if k >= 2])
print('   blocks SKIPPED          %s'
      % [names[k - 2] for k in skipped if k >= 2])
c1 = len(cone) < len(full)
ok &= c1
print('   cone is a strict subset: %s' % ('PASS' if c1 else 'FAIL (nothing is saved)'))

# ---- 2. exactness -----------------------------------------------------------------------------
g = torch.Generator().manual_seed(0)
X = torch.randn(64, nx, generator=g, dtype=cfg.dtype_pt)
U = torch.randn(64, nu, generator=g, dtype=cfg.dtype_pt) * 1e2
print('\n2  EXACTNESS vs the full forward, 64 random states')
worst = 0.0
for tag, uu in (('u = 0', torch.zeros(64, nu, dtype=cfg.dtype_pt)), ('u = 1e2 randn', U)):
    with torch.no_grad():
        y_full = hfn(X, uu)[0]
        y_cone = hfn.output_only(X, uu)
    d = float((y_full - y_cone).abs().max())
    worst = max(worst, d)
    print('   %-14s max |dy| %.3e   bitwise equal %s'
          % (tag, d, bool(torch.equal(y_full, y_cone))))
c2 = worst <= TOL_EXACT
ok &= c2
print('   worst %.3e   tol %.1e   %s' % (worst, TOL_EXACT, 'PASS' if c2 else 'FAIL'))

# ---- 3. no feedthrough ------------------------------------------------------------------------
print('\n3  NO FEEDTHROUGH through the real interconnect (D_d = 0 is a COEFFICIENT property)')
d_ft_full = PLANT.check_no_feedthrough(hfn, nx, nu, dtype=cfg.dtype_pt)
with torch.no_grad():
    ua = torch.zeros(64, nu, dtype=cfg.dtype_pt)
    ub = torch.randn(64, nu, generator=g, dtype=cfg.dtype_pt) * 1e3
    d_ft_cone = float((hfn.output_only(X, ua) - hfn.output_only(X, ub)).abs().max())
print('   full forward  max |y(u=0) - y(u=1e3 randn)| %.3e' % d_ft_full)
print('   output_only   max |y(u=0) - y(u=1e3 randn)| %.3e' % d_ft_cone)
c3 = max(d_ft_full, d_ft_cone) <= TOL_FEEDTHROUGH
ok &= c3
print('   %s' % ('PASS' if c3 else 'FAIL: the closed-loop step order is invalid, y depends on u'))

# ---- 4. against the identified affine map -----------------------------------------------------
print('\n4  AGAINST identify_output_map, the map the previous implementation used')
C_out, b_out = PLANT.identify_output_map(hfn, nx, nu, dtype=cfg.dtype_pt)
with torch.no_grad():
    y_map = X @ C_out.T + b_out
    y_cone = hfn.output_only(X)
d4 = float((y_map - y_cone).abs().max())
den = float(y_cone.abs().max())
c4 = (d4 / max(den, 1e-30)) <= TOL_AFFINE
ok &= c4
print('   max |dy| %.3e   relative %.3e   tol %.1e   %s'
      % (d4, d4 / max(den, 1e-30), TOL_AFFINE, 'PASS' if c4 else 'FAIL'))

print('\n' + '=' * 96)
print('OVERALL: %s' % ('PASS' if ok else 'FAIL'))
sys.exit(0 if ok else 1)
