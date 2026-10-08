"""D-232 follow-up: velocities from noisy positions in the Aug-OBC reference set (and in the norm).

The OBC reference set (obc_gantry.build_reference_set) takes q = P^-T y and qdot by a fourth-order central
difference; the normalisation (data.compute_normalization) takes its velocity scale from a first difference.
On the noisy thesis mode y is now the D-232 decimation (zero-phase Butterworth 8 at 1.6 kHz on y - r).
# THEORY: Janot, Gautier, Brunot 2019 (EMPS), Sect. 3.1: filter the position forward and backward with a
#   Butterworth, cutoff w_fq >= 5 w_dyn (w_dyn the natural frequency of the closed loop's highest mode), then
#   central differences ("bandpass filtering"). Here w_dyn = the absorber pole, 212 Hz: 5 w_dyn = 1.06 kHz.

Variants on the SAME tuples (18 training records, the noisy loader's samples, every STRIDE-th stencil
sample), all with the loader's u and one norm:
  truth        x_logical of the noisy run (exact state)
  clean        the stencil on y_true (the same run's true position, point-sampled): the method, no noise
  filtered     the stencil on the loader's y (production after D-232)
  point        the stencil on point-sampled noisy y (production before D-232)
  lowcut       the stencil on r + BW8 at 1.06 kHz (y - r): the candidate if 'filtered' fails
For each: velocity error against truth (rms, relative to the true velocity rms, per logical axis), the stacked
one-step sensitivity J of the six protected rows to the ten combinations at the true combinations (the matrix
the OBC basis spans), its principal angles to 'clean' and 'truth', and ||J - J_clean|| / ||J_clean||.
Norm check: std_x from the noisy loader against the noise-free loader (compute_normalization's formula).
Criteria and adoption rule: docs/decisions.md, D-232 OBC velocity (stated before the run).

Run: conda run -n GraduationProject python obc_velocity.py
"""
__project_origin__ = "added"

import os
import sys
import time
from dataclasses import replace

import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
for _p in (REPO, os.path.join(REPO, 'scripts', 'gantry')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from model_augmentation.fit_systems.blocks import (                                # noqa: E402
    Parameterized_Gantry_State_Block, Reduced_Gantry_State_Block)
from model_augmentation.fit_systems.obc import build_stacked_sensitivity           # noqa: E402
from model_augmentation.systems import gantry_ss as gss                             # noqa: E402
from gantry_dynamic.config import RunConfig, record_trim                            # noqa: E402
from gantry_dynamic import data as G                                                # noqa: E402
from gantry_dynamic.obc_gantry import _load_record_float64, fourth_order_central_difference  # noqa: E402

STRIDE = 5                     # HEURISTIC: memory; fewer rows average less noise, so conservative
ROWS = (0, 1, 2, 3, 4, 5)      # the six protected physical rows (route_ix = range(6 + n_a))
F_POLE = 212.0                 # THEORY: fa sqrt(1 + ma / mh_rigid) = 150 sqrt(2) Hz (tdg_config.m, T_AF)
LOWCUT = butter(8, 5 * F_POLE, fs=20000.0, output='sos')
SIN_MAX = 0.01                 # decisive criterion (docs/decisions.md)
NORM_TOL = 0.01

cfg = replace(RunConfig(), mode='thesis_taf_noisy', joint_estimation=True, physics_parameterization='reduced',
              param_init_detune=None, obc=True, obc_space='tangent', nx_ann=2, use_f64=True)   # as _THESIS_FIXED
cfg_free = replace(cfg, mode='thesis_taf_noisefree')
files = G.record_files(cfg)[0]
K = record_trim(cfg)
PinvT = np.linalg.inv(gss.P.detach().cpu().numpy().astype(np.float64).T)
t0 = time.perf_counter()


def norm_stats(c):
    """compute_normalization's x statistics (float32 loader data, first-difference velocity)."""
    xs = []
    for f in files:
        sd = G.load_traj(f, c)
        pos = (np.linalg.inv(gss.P.numpy().T).astype(c.dtype_np) @ sd.y.T).T
        vel = np.diff(pos, axis=0) * (1.0 / sd.dt)
        xs.append(np.hstack([pos, np.vstack([vel[:1], vel])]))
    x = np.concatenate(xs)
    return x.mean(0), x.std(0) + 1e-8


x_mean, std_x = norm_stats(cfg)
_, std_x_free = norm_stats(cfg_free)
u_all = np.concatenate([G.load_traj(f, cfg).u for f in files])
u_mean, std_u = u_all.mean(0), u_all.std(0) + 1e-8
del u_all
print('[norm] std_x noisy %s' % np.array2string(std_x, precision=5))
print('[norm] std_x noisy / noise-free - 1: %s  (tolerance %.2f)'
      % (np.array2string(std_x / std_x_free - 1, precision=6), NORM_TOL), flush=True)

VARIANTS = ('truth', 'clean', 'filtered', 'point', 'lowcut')
Z = {v: [] for v in VARIANTS}
verr = {v: [] for v in VARIANTS}
vtrue = []
for f in files:
    u, y_filt, x_log, _ = _load_record_float64(f, cfg)
    d = loadmat(os.path.join(G.traj_dir(cfg), f), squeeze_me=True)
    N0 = d['u_total'].shape[0] // cfg.d
    cut = lambda a: a[::cfg.d][:N0][K:N0 - K]                                      # noqa: E731
    r = d['r_sim']
    ys = {'filtered': y_filt, 'clean': cut(d['y_true']), 'point': cut(d['y']),
          'lowcut': cut(r + sosfiltfilt(LOWCUT, d['y'] - r, axis=0))}
    assert all(len(a) == len(u) for a in ys.values()) and len(x_log) == len(u)
    k = np.arange(2, len(u) - 2)[::STRIDE]
    xs = {'truth': x_log[k]}
    for v, yv in ys.items():
        q = yv @ PinvT.T
        xs[v] = np.hstack([q[k], fourth_order_central_difference(q, cfg.ts_new)[k - 2]])
    vtrue.append(x_log[k, 3:])
    un = (u[k] - u_mean) / std_u
    for v in VARIANTS:
        verr[v].append(xs[v][:, 3:] - x_log[k, 3:])
        Z[v].append(np.hstack([(xs[v] - x_mean) / std_x, un]))

vt_rms = np.sqrt(np.mean(np.concatenate(vtrue) ** 2, axis=0))
print('\n[velocity] rms error against the true velocity, relative to its rms (X, Theta, Y logical):')
for v in VARIANTS:
    e = np.sqrt(np.mean(np.concatenate(verr[v]) ** 2, axis=0))
    print('  %-9s %s   abs %s' % (v, np.array2string(e / vt_rms, precision=5),
                                   np.array2string(e, precision=3)))
print('  true velocity rms %s' % np.array2string(vt_rms, precision=4), flush=True)

c_true = Reduced_Gantry_State_Block.combos_of(
    torch.stack([getattr(gss, n) for n in Parameterized_Gantry_State_Block.PARAM_NAMES]).to(torch.float64),
    float(gss.Lb))
raw = torch.stack([getattr(gss, n) for n in Parameterized_Gantry_State_Block.PARAM_NAMES]).to(torch.float64)
col = lambda a: np.asarray(a, np.float64).reshape(-1, 1)          # (n, 1) as model.py passes Norm  # noqa: E731
block = Reduced_Gantry_State_Block(
    params_init=raw, combo_init=c_true.clone(), Y_op=None, std_x=col(std_x), std_u=col(std_u),
    x_mean=col(x_mean), u_mean=col(u_mean), Ts=cfg.ts_new, up_sample=cfg.hp['up_sample'],
    RMSE_baseline=cfg.param_rmse_baseline, flag_loss_reg=False).to(torch.float64)
vbar = block.free_from_combinations(c_true)
step = lambda v, z: block.transition_from_free(v, z)                                  # noqa: E731

J = {}
for v in VARIANTS:
    Zv = torch.from_numpy(np.concatenate(Z[v])).to(torch.float64).unsqueeze(-1)
    Z[v] = None
    J[v] = build_stacked_sensitivity(step, vbar, Zv, ROWS, chunk=cfg.obc_ref_chunk,
                                     device=torch.device('cpu')).to(torch.float64)
    del Zv
    print('[J] %-9s %d x %d built (%.0f s)' % (v, J[v].shape[0], J[v].shape[1], time.perf_counter() - t0),
          flush=True)
Q = {ref: torch.linalg.qr(J[ref], mode='reduced')[0] for ref in ('clean', 'truth')}   # memory: 2 kept


def sin_max(a, b):
    Qa = Q[a] if a in Q else torch.linalg.qr(J[a], mode='reduced')[0]
    c = torch.linalg.svdvals(Qa.T @ Q[b]).clamp(max=1.0)
    return float(torch.sqrt(1.0 - c.min() ** 2)), torch.rad2deg(torch.arccos(c)).numpy()


print('\n[basis] principal angles (deg) and the largest one as sin; relative J change')
res = {}
for v in VARIANTS:
    for ref in ('clean', 'truth'):
        if v == ref:
            continue
        s, ang = sin_max(v, ref)
        rel = float(torch.linalg.norm(J[v] - J[ref]) / torch.linalg.norm(J[ref]))
        res[(v, ref)] = s
        print('  %-9s vs %-6s sin(max) %.2e  ||dJ||/||J|| %.2e  angles %s'
              % (v, ref, s, rel, np.array2string(np.sort(ang)[::-1], precision=4)))
sv = {v: torch.linalg.svdvals(J[v]) for v in VARIANTS}
print('  cond(J): ' + '  '.join('%s %.3g' % (v, float(sv[v][0] / sv[v][-1])) for v in VARIANTS))

print('\n== verdict (decisive: sin of the largest angle to clean <= %.2f) ==' % SIN_MAX)
for v in ('filtered', 'lowcut', 'point'):
    print('  %-9s %.2e -> %s' % (v, res[(v, 'clean')], 'PASS' if res[(v, 'clean')] <= SIN_MAX else 'FAIL'))
print('  norm: max |std_x noisy / noise-free - 1| = %.2e -> %s'
      % (float(np.max(np.abs(std_x / std_x_free - 1))),
         'PASS' if np.max(np.abs(std_x / std_x_free - 1)) <= NORM_TOL else 'FAIL'))
print('done in %.0f s' % (time.perf_counter() - t0))
