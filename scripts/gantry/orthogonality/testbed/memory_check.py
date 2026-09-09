"""Does memory earn its place? Static versus dynamic augmentation, plus K and G.

Category: DEVELOPMENT AND FALSIFICATION TOOL (see system.py). Not gantry evidence.

WHY THIS RUNS BEFORE ANY MORE REGULARISER WORK. The four-condition sweep showed the
projection penalty retains ~20x more augmentation than a plain size penalty at the same
parameter error. That only matters if the retained augmentation is worth something, and in
that sweep the augmentation improved prediction NOWHERE: 1.02e-3 baseline-alone, 1.09e-3
free, 1.03e-3 constrained. So "useful dynamic correction is preserved" currently rests on
nothing. This script tests the prerequisite directly.

THE COMPARISON. Physical parameters FIXED AT TRUTH, no penalty, so nothing competes with the
augmentation and the only question is predictive value:

    none     no augmentation at all
    static   trained with x_aug held at zero every step, so the network is a static map
             of (x, 0, u). Trained that way, not merely evaluated that way.
    dynamic  the full recurrent augmentation

Evaluated on HELD-OUT data at rollout lengths LONGER than training. A static correction
cannot accumulate, so any advantage from internal state should grow with horizon. If it does
not appear there, it does not exist.

WHY K AND G ARE MEASURED (D-180). The additional states have no declared A/B matrices; the
matrices are Jacobian blocks of the shared MLP:
    K = d w_phys / d x_aug    latent -> physical, the block that decides the contribution
    G = d w_aug  / d x_aug    latent dynamics, the latent "A"
    L = d w_phys / d x_phys   direct correction, for scale
Without these, a null result is uninterpretable. With them it splits into two diagnoses that
need different fixes:
    ||K|| tiny relative to ||L||   -> the latent path is barely wired in. Architecture or
                                      initialisation problem, not a plant problem.
    rho(G) ~ 0                     -> the latent states decay in one step and carry no
                                      memory, whatever their magnitude.
    both healthy but no gain       -> the PLANT has nothing for memory to capture, and the
                                      testbed itself needs changing.

Run: conda run -n GraduationProject python scripts/gantry/orthogonality/testbed/memory_check.py
"""
__project_origin__ = "added"

import os
import sys
import json
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import system as S            # noqa: E402
import geometry as G          # noqa: E402
import train_compare as T     # noqa: E402

F64 = torch.float64
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'memory_check.json')

NX, NA, NU, N_LAG = T.NX, T.NA, T.NU, T.N_LAG
N_WIN = 48
T_TRAIN = 32                      # same as the sweep
T_EVAL = (32, 64, 128, 256)       # held out, and LONGER than training
N_ITER = 600
LR = 2e-2
MODES = ('none', 'static', 'dynamic')
_MODE_TO_ROLL = {'none': 'annoff', 'static': 'frozen', 'dynamic': 'full'}


def batch(u, y, starts, T_win):
    U = torch.tensor(np.stack([u[k:k + T_win] for k in starts]), dtype=F64)
    Y = torch.tensor(np.stack([y[k:k + T_win] for k in starts]), dtype=F64)
    L = torch.tensor(np.stack([np.concatenate([u[k - N_LAG:k], y[k - N_LAG:k]])
                               for k in starts]), dtype=F64)
    return U, Y, L


def jac_blocks(eta, xi, shapes, x_ref, xa_ref, u_ref):
    """K, G, L and the latent spectral radius at a reference point (D-180)."""
    W = G.unpack(eta, shapes)

    def net(x, xa, u):
        return T.ann_b(torch.cat([x, xa, u], dim=1), W)[0]

    x = x_ref.clone().requires_grad_(True)
    xa = xa_ref.clone().requires_grad_(True)
    w = net(x, xa, u_ref)
    Kb = torch.zeros(NX, NA, dtype=F64)
    Lb = torch.zeros(NX, NX, dtype=F64)
    Gb = torch.zeros(NA, NA, dtype=F64)
    for r in range(NX + NA):
        gx, ga = torch.autograd.grad(w[r], [x, xa], retain_graph=(r < NX + NA - 1))
        if r < NX:
            Lb[r], Kb[r] = gx[0], ga[0]
        else:
            Gb[r - NX] = ga[0]
    Gn = Gb.detach().numpy()
    rho = float(np.max(np.abs(np.linalg.eigvals(Gn)))) if NA else 0.0
    return (Kb.detach().numpy(), Gn, Lb.detach().numpy(), rho)


def train_one(mode, data, shapes, seed=0):
    """theta FIXED AT TRUTH, no penalty. Only the network and encoder are trained."""
    u, y, starts = data
    U, Y, L = batch(u, y, starts, T_TRAIN)
    theta = torch.tensor(S.THETA_TRUE, dtype=F64)          # fixed, not estimated
    eta0, _ = G.init_eta(seed)
    eta = eta0.clone().requires_grad_(True)
    xi = torch.zeros(NX * 2 * N_LAG + NX, dtype=F64)
    xi[:NX * 2 * N_LAG] = torch.linalg.lstsq(
        L, torch.tensor(S.baseline_states(u)[starts], dtype=F64)).solution.T.reshape(-1)
    xi = xi.requires_grad_(True)

    roll_mode = _MODE_TO_ROLL[mode]
    params = [xi] if mode == 'none' else [eta, xi]
    opt = torch.optim.Adam(params, lr=LR)
    for _ in range(N_ITER):
        opt.zero_grad()
        yh = T.rollout(theta, eta, xi, U, L, shapes, mode=roll_mode)
        ((yh - Y) ** 2).mean().backward()
        opt.step()
    return theta, eta.detach(), xi.detach(), roll_mode


def main():
    torch.set_default_dtype(F64)
    t0 = time.time()
    print('=' * 78)
    print('MEMORY CHECK.  Does the latent path improve prediction at all?')
    print('Category: development tool, not gantry evidence.')
    print('=' * 78)

    d_tr, d_va = S.make_dataset(seed=0), S.make_dataset(seed=7)
    rng = np.random.default_rng(0)
    st_tr = rng.choice(np.arange(N_LAG, len(d_tr['u']) - T_TRAIN - 1), N_WIN, replace=False)
    _, shapes = G.init_eta(0)
    print(f'  theta FIXED at truth, no penalty. Train T = {T_TRAIN}, {N_ITER} iters, '
          f'{N_WIN} windows.')
    print(f'  Held-out evaluation horizons: {T_EVAL} (seed-7 record)\n')

    # ONE validation manifest per horizon, drawn BEFORE the mode loop and shared by every
    # condition. BUG FIXED 2026-09-07 (found in review): these starts were drawn inside the
    # mode loop, so none / static / dynamic were each evaluated on DIFFERENT windows and the
    # horizon table was not comparing like with like.
    val_manifest = {Te: rng.choice(np.arange(N_LAG, len(d_va['u']) - Te - 1),
                                   N_WIN, replace=False) for Te in T_EVAL}
    res = {'category': 'development tool; testbed only', 'T_train': T_TRAIN,
           'T_eval': list(T_EVAL), 'runs': {},
           'shared_val_manifest': {int(k): [int(v) for v in val_manifest[k]]
                                   for k in val_manifest}}
    rmse = {}
    for mode in MODES:
        theta, eta, xi, roll_mode = train_one(mode, (d_tr['u'], d_tr['y'], st_tr), shapes)
        row = []
        for Te in T_EVAL:
            Uv, Yv, Lv = batch(d_va['u'], d_va['y'], val_manifest[Te], Te)
            with torch.no_grad():
                yh = T.rollout(theta, eta, xi, Uv, Lv, shapes, mode=roll_mode)
                row.append(float(((yh - Yv) ** 2).mean().sqrt()))
        rmse[mode] = row
        res['runs'][mode] = {'rmse': row}
        print(f'  {mode:8s} held-out RMSE  ' +
              '  '.join(f'T={t}: {v:.3e}' for t, v in zip(T_EVAL, row)), flush=True)

        if mode == 'dynamic':
            U0, _, L0 = batch(d_tr['u'], d_tr['y'], st_tr, T_TRAIN)
            with torch.no_grad():
                Ad, Bd = S.baseline_dt(theta)
                Wenc = xi[:NX * 2 * N_LAG].reshape(NX, 2 * N_LAG)
                x = (L0 @ Wenc.T + xi[NX * 2 * N_LAG:])[:1]
                xa = torch.zeros(1, NA, dtype=F64)
                for k in range(T_TRAIN // 2):        # settle onto the realised trajectory
                    w = T.ann_b(torch.cat([x, xa, U0[:1, k:k + 1]], dim=1),
                                G.unpack(eta, shapes))
                    x = x @ Ad.T + U0[:1, k:k + 1] @ Bd.T + w[:, :NX]
                    xa = w[:, NX:]
            Kb, Gb, Lb, rho = jac_blocks(eta, xi, shapes, x, xa, U0[:1, T_TRAIN // 2:
                                                                    T_TRAIN // 2 + 1])
            nK, nL = np.linalg.norm(Kb, 2), np.linalg.norm(Lb, 2)
            res['jacobians'] = dict(norm_K=float(nK), norm_L=float(nL),
                                    ratio_K_over_L=float(nK / nL) if nL > 0 else None,
                                    rho_G=float(rho),
                                    norm_xa=float(np.linalg.norm(xa.numpy())))
            print(f'\n  JACOBIAN BLOCKS at the realised trajectory (D-180):')
            print(f'    ||K|| = {nK:.3e}   latent -> physical')
            print(f'    ||L|| = {nL:.3e}   direct correction')
            print(f'    ||K||/||L|| = {nK/nL:.3e}   <- is the latent path wired in at all')
            print(f'    rho(G) = {rho:.4f}          <- latent memory; ~0 means one-step decay')
            print(f'    ||x_aug|| = {np.linalg.norm(xa.numpy()):.3e}   (magnitude only, '
                  'NOT a contribution measure)')

    print('\n' + '-' * 78)
    print('VERDICT')
    for Te, i in zip(T_EVAL, range(len(T_EVAL))):
        g = rmse['static'][i] / rmse['dynamic'][i]
        print(f'  T = {Te:4d}:  static/dynamic = {g:6.3f}   '
              + ('dynamic better' if g > 1.02 else
                 'static better' if g < 0.98 else 'no difference'))
    print('\n  If the ratio does not exceed 1 and grow with horizon, memory does not earn its')
    print('  place on this plant. Read the Jacobian blocks to tell WHY:')
    print('    ||K||/||L|| tiny -> latent path barely wired in (architecture/init)')
    print('    rho(G) ~ 0       -> the LOCAL latent block decays fast. NOTE: this does not')
    print('                        establish absence of memory in the COUPLED system, since a')
    print('                        correction also propagates through the physical dynamics.')
    print('    both healthy     -> the PLANT has nothing for memory to capture; change it')
    print(f'\n  wall clock {time.time() - t0:.1f} s')

    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=2)
    print(f'  wrote {OUT}')


if __name__ == '__main__':
    main()
