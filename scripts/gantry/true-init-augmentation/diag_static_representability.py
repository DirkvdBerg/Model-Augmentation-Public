"""Is the correction the ANN must supply a FUNCTION of what the ANN can see?

WHY THIS EXISTS. The handoff's section 8 asks the training arm to discriminate
between three causes of failure: persistence (rows 6-7 are rebuilt from scratch
every step, so the model cannot carry a second-order oscillator), coordinate
pinning (nothing ties rows 6-7 to anything), and capacity (the ANN simply cannot
represent the correction). A training run answers "did it learn", not "why not".
This measures the why directly, and without training.

THE OBJECT. Give the baseline the TRUE current physical state and one input
sample. The one-step state it produces differs from the truth's next state by

    Delta(k) = x6_truth(k+1) - Phi_base(x6_truth(k), u(k))          [normalised]

which is exactly what the ANN's `xp` contribution would have to be, at that step,
for the model to be exact. That is the ideal target, constructed from the truth,
with no encoder, no rollout and no optimizer in the way.

THE ANN'S INPUT is `z = [x_norm (8 rows), u_norm (3)]`. Rows 6-7 of `x_norm` are
identically zero: the ANN's zero-initialised final layer means the propagated
`x_aug` is exactly `0.000000e+00` at every step (`verify_ms_gradient.py` gate
G6), and nothing else writes those rows (`model.py:132` / `:135`). So the ANN is a
STATIC map of nine informative numbers.

FOUR TESTS, cheapest first, each answering something the previous cannot:

  L   linear least squares  Delta ~ [z, 1]                  R^2
  A   linear least squares  Delta ~ [z, da, vda, 1]         R^2   (does the
      absorber state close the gap? if L is ~0 and A is ~1, the missing
      information IS the absorber state and it is not in z)
  N   nearest-neighbour consistency. For the pairs closest in z, compare their
      Delta. Any continuous static function f(z) must give nearly equal outputs
      for nearly equal inputs, so a large Delta spread over near-duplicate z
      falsifies EVERY static f, not just linear ones. Reported against a CONTROL
      target that is a static function of z by construction, so the test cannot
      pass for the wrong reason.
  M   direct supervised fit of the pipeline's own ANN (2x16 tanh, same widths)
      from z to Delta, no dynamics, no BPTT, no encoder, thousands of epochs.
      This is the most generous capacity test the architecture can be given.

Run:
  PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output \\
      -n GraduationProject python -u \\
      scripts/gantry/true-init-augmentation/diag_static_representability.py
"""
__project_origin__ = "added"

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry'))

from scipy.spatial import cKDTree                                    # noqa: E402
from model_augmentation.utils.torch_nets import zero_init_feed_forward_nn  # noqa: E402

from gantry_dynamic.config import RunConfig                          # noqa: E402
from gantry_dynamic.data import load_datasets, compute_normalization  # noqa: E402
from plant_cog import make_block                                     # noqa: E402
from data_exact import exact_truth                                   # noqa: E402

OUT = os.path.join(REPO, 'simulations', 'gantry_subnet', 'diagnostics')
RECORDS = ('V1_standstill_Yp10', 'T9_aprbs_30')
STATES = ['X', 'Theta', 'Y', 'dX', 'dTheta', 'dY']
CFG = RunConfig(mode='augmentation', fs_new=4000, nx_ann=2, up_sample=1, use_f64=False)


def r2(target, feats):
    """Least-squares R^2 of target (N, k) on feats (N, p), per column and overall."""
    A = np.hstack([feats, np.ones((len(feats), 1))])
    coef, *_ = np.linalg.lstsq(A, target, rcond=None)
    res = target - A @ coef
    ss_res = (res ** 2).sum(axis=0)
    ss_tot = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
    per = 1.0 - ss_res / np.maximum(ss_tot, 1e-300)
    tot = 1.0 - ss_res.sum() / max(ss_tot.sum(), 1e-300)
    return per, float(tot)


def nn_consistency(z, targets, frac=0.05, sub=20000, seed=0):
    """Spread of `targets` over the pairs that are closest in z.

    Returns, per named target, rms(t_k - t_k') / (sqrt(2) * std(t)) taken over the
    `frac` closest pairs. A value near 0 means "a static function of z could
    produce this"; near 1 means the target is, at that resolution in z, as
    unrelated to z as two random samples are.
    """
    rng = np.random.default_rng(seed)
    n = len(z)
    ix = rng.choice(n, size=min(sub, n), replace=False)
    zz = z[ix]
    tree = cKDTree(zz)
    d, j = tree.query(zz, k=2)          # k=1 is the point itself
    d, j = d[:, 1], j[:, 1]
    keep = np.argsort(d)[:max(10, int(frac * len(zz)))]
    out = {'median_pair_dist': float(np.median(d[keep])),
           'median_all_dist': float(np.median(d))}
    for name, t in targets.items():
        tt = t[ix]
        diff = tt[keep] - tt[j[keep]]
        out[name] = float(np.sqrt((diff ** 2).mean())
                          / max(np.sqrt(2) * tt.std(axis=0).mean(), 1e-300))
    return out


def fit_mlp(z, target, epochs=3000, seed=0, lr=1e-3, n_nodes=16, n_layers=2):
    """Direct supervised fit with the pipeline's own ANN architecture."""
    torch.manual_seed(seed)
    net = zero_init_feed_forward_nn(n_in=z.shape[1], n_out=target.shape[1],
                                    n_nodes_per_layer=n_nodes,
                                    n_hidden_layers=n_layers,
                                    activation=torch.nn.Tanh).double()
    Z = torch.as_tensor(z, dtype=torch.float64)
    T = torch.as_tensor(target, dtype=torch.float64)
    ntr = int(0.8 * len(Z))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    var_tr = float(((T[:ntr] - T[:ntr].mean(0)) ** 2).mean())
    var_va = float(((T[ntr:] - T[ntr:].mean(0)) ** 2).mean())
    best = 1.0
    for ep in range(epochs):
        opt.zero_grad()
        L = torch.nn.functional.mse_loss(net(Z[:ntr]), T[:ntr])
        L.backward()
        opt.step()
        if (ep + 1) % 250 == 0:
            with torch.no_grad():
                Lv = float(torch.nn.functional.mse_loss(net(Z[ntr:]), T[ntr:]))
            best = min(best, Lv / var_va)
    with torch.no_grad():
        r2_tr = 1.0 - float(torch.nn.functional.mse_loss(net(Z[:ntr]), T[:ntr])) / var_tr
        r2_va = 1.0 - float(torch.nn.functional.mse_loss(net(Z[ntr:]), T[ntr:])) / var_va
    return dict(r2_train=r2_tr, r2_val=r2_va, r2_val_best=1.0 - best)


def build_targets(name, norm, cfg):
    """Delta(k), the ANN's ideal per-step contribution, plus the ANN's inputs."""
    tr = exact_truth(name)
    rec, x6, x8 = tr['rec'], tr['x6'], tr['x8']
    blk = make_block(Y_op=None, cog=True, ts=rec['ts'],
                     up_sample=cfg.up_sample, dtype=torch.float64)
    N = len(x6) - 1
    with torch.no_grad():
        xt = torch.as_tensor(x6[:N], dtype=torch.float64).reshape(N, 6, 1)
        ut = torch.as_tensor(rec['u'][:N], dtype=torch.float64).reshape(N, 3, 1)
        nxt = blk.nonlinear_function(torch.cat([xt, ut], dim=1))[:, :, 0].numpy()
    xs = norm.std_x.flatten()
    xm = norm.x_mean.flatten()
    delta = (x6[1:N + 1] - nxt) / xs                     # NORMALISED, as the ANN outputs
    # control: a quantity that IS a static function of z by construction
    control = (nxt - x6[:N]) / xs                        # the baseline's own increment
    z = np.zeros((N, 6 + cfg.nx_ann + 3))
    z[:, :6] = (x6[:N] - xm) / xs
    z[:, 6:6 + cfg.nx_ann] = 0.0                          # rows 6-7 are identically zero (G6)
    z[:, 6 + cfg.nx_ann:] = (rec['u'][:N] - norm.u_mean.flatten()) / norm.std_u.flatten()
    aug = x8[:N][:, [3, 7]]                               # [delta_a, vdelta_a]
    return dict(z=z, delta=delta, control=control, aug=aug, tr=tr, N=N)


def main():
    print('Is the ANN\'s ideal per-step correction a FUNCTION of the ANN\'s inputs?\n')
    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)
    data = load_datasets(CFG)
    norm = compute_normalization(CFG, data)
    res = {}
    for name in RECORDS:
        print(f'\n=== {name} ===')
        D = build_targets(name, norm, CFG)
        z, delta, control, aug = D['z'], D['delta'], D['control'], D['aug']
        print(f'  {D["N"]} samples, z has {z.shape[1]} columns '
              f'(6 physical + {CFG.nx_ann} augmented, identically zero + 3 input)')
        print(f'  |Delta| rms per row (normalised state units): '
              f'{["%.3e" % v for v in delta.std(axis=0)]}')

        perL, totL = r2(delta, z)
        perA, totA = r2(delta, np.hstack([z, aug]))
        perC, totC = r2(control, z)
        print(f'\n  L  R^2 of Delta on z            overall {totL:.6f}')
        print(f'     per state ' + '  '.join(f'{STATES[c]} {perL[c]:.4f}' for c in range(6)))
        print(f'  A  R^2 of Delta on [z, da, vda] overall {totA:.6f}')
        print(f'     per state ' + '  '.join(f'{STATES[c]} {perA[c]:.4f}' for c in range(6)))
        print(f'  C  R^2 of the CONTROL on z      overall {totC:.6f}   '
              f'(a static function of z by construction -> must be ~1)')

        nnres = nn_consistency(z, dict(delta=delta, control=control))
        print(f'\n  N  nearest-neighbour consistency over the 5 % closest pairs in z')
        print(f'     median pair distance {nnres["median_pair_dist"]:.3e} '
              f'(median over all pairs {nnres["median_all_dist"]:.3e})')
        print(f'     Delta   {nnres["delta"]:.4f}   '
              f'(0 = a static f(z) could produce it, 1 = as unrelated as random pairs)')
        print(f'     control {nnres["control"]:.4f}   (must be near 0)')

        t0 = time.time()
        mlp = fit_mlp(z, delta)
        mlpc = fit_mlp(z, control)
        print(f'\n  M  direct supervised fit, pipeline ANN architecture '
              f'(2x16 tanh, 3000 Adam steps, 80/20 split, {time.time()-t0:.0f} s)')
        print(f'     Delta    R^2 train {mlp["r2_train"]:.4f}  val {mlp["r2_val"]:.4f}  '
              f'best val {mlp["r2_val_best"]:.4f}')
        print(f'     control  R^2 train {mlpc["r2_train"]:.4f}  val {mlpc["r2_val"]:.4f}  '
              f'(must be high: the architecture is not the limit)')

        res[name] = dict(
            N=int(D['N']), delta_rms=delta.std(axis=0).tolist(),
            R2_delta_on_z=dict(overall=totL, per_state=perL.tolist()),
            R2_delta_on_z_aug=dict(overall=totA, per_state=perA.tolist()),
            R2_control_on_z=dict(overall=totC, per_state=perC.tolist()),
            nn=nnres, mlp_delta=mlp, mlp_control=mlpc)

    print('\n=== HOW TO READ IT ===')
    print('  L ~ 0 and A ~ 1  ->  the missing information is exactly [delta_a, vdelta_a],')
    print('                       and it is NOT in the ANN\'s input. PERSISTENCE.')
    print('  N(Delta) ~ 1 with N(control) ~ 0  ->  NO static f(z) can produce Delta,')
    print('                       linear or otherwise. This is architecture-independent.')
    print('  M low for Delta but high for control  ->  the ANN is not under-sized;')
    print('                       the target is not a function of its inputs. NOT CAPACITY.')
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, 'true_init_static_representability.json')
    with open(p, 'w') as f:
        json.dump(res, f, indent=2, default=float)
    print(f'\n  wrote {p}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
