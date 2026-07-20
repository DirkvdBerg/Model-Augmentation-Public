"""
diag_window_lr.py
-----------------
Real-data diagnostic for the two knobs behind the failed run 68734 (D-075):
segment length (window) and learning rate. All measurements are made on the
real Telica logs; no synthetic/model-generated data (lessons.md rule).

Part 1 -- window sweep (forward-only, 1-D loss slices):
    For each window length W, multiply one parameter direction at a time by
    a set of factors and evaluate the exact training loss (sigma-normalized
    MSE, teacher-forced window start) on motion-anchored windows of B
    motion-rich trajectories.
        flat slice   -> direction invisible at that horizon (explains the
                        random walk of run 68734 at W=650)
        curved slice -> identifiable; the minimum shows where the REAL-data
                        loss pulls that direction (not necessarily nominal:
                        that offset is the friction-compensation effect)
    Degenerate pairs (kb1+kb2, cb1+cb2, Jb+Jh) are sliced jointly.
    Also: gradient norm at nominal per W.

Part 2 -- learning-rate probe (short descent runs at one chosen W):
    N Adam steps from nominal init per candidate lr, on the same windows.
    Tracked per step: training loss, max multiplicative parameter change.
    After each probe: full-trajectory open-loop RMSE on one validation
    trajectory. Verdict: largest lr with smooth loss descent and bounded
    parameter motion.

Run (server recommended for the full grid):
    conda run -n GraduationProject python scripts/gantry/real-data-verification/diag_window_lr.py
Options:
    --part {1,2,both}   default both
    --fast              reduced grids (W {650,2600}, mults {0.5,2.0})
    --smoke             minimal correctness run (minutes)
    --w 2600            window for part 2
    --lrs 3e-2,1e-2,3e-3,1e-3
    --steps 25          Adam steps per lr probe
"""

__project_origin__ = "added"

import os
import sys
import json
import time
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import run_telica_param_recovery as rt          # applies all pipeline patches
from telica_loader import load_telica_log
from lpv_lfr_baseline.core.lfr_simulate import simulate

_SAVE_DIR = os.path.join(_ROOT, 'simulations', 'gantry_subnet',
                         'diagnostics', 'window_lr')

# Motion-rich subset: iter0 (feedback-dominated) at 6 OPs spanning the grid
_DIAG_TRAJ_IDS = ('T1a', 'T2a', 'T5a', 'T7a', 'T9a', 'T10a')
_VAL_CHECK_ID  = 'V2b'   # full-trajectory outside check in part 2

# Parameter directions: joint indices for the non-identifiable splits
# _PARAM_NAMES = [kb1 kb2 cg1 cg2 cy cb1 cb2 mh m1 m2 mb Jb Jh d]
_DIRECTIONS = (
    ('kb_sum', (0, 1)), ('cg1', (2,)), ('cg2', (3,)), ('cy', (4,)),
    ('cb_sum', (5, 6)), ('mh', (7,)), ('m1', (8,)), ('m2', (9,)),
    ('mb', (10,)), ('J_sum', (11, 12)), ('d', (13,)),
)

W_GRID_FULL  = (650, 1300, 2600, 5200)
MULTS_FULL   = (0.5, 0.7, 1.4, 2.0)
LR_GRID_FULL = (3e-2, 1e-2, 3e-3, 1e-3)

FLAT_THRESHOLD_PCT = 5.0   # HEURISTIC: slice range below this = "flat"
PRE_MOTION_SAMPLES = 200   # HEURISTIC: window starts 10 ms before motion
MOTION_EPS_M       = 1e-6  # HEURISTIC: 1 um deviation marks motion start


def load_diag_data(w_max, dtype):
    """Load the diagnostic subset; one motion-anchored window start per
    trajectory, fixed across all W (horizon grows from the same start)."""
    spec_by_id = {s['id']: s for s in rt.TRAJ_SPECS + rt.VAL_SPECS}
    P = rt._P.to(dtype)
    data = []
    for tid in _DIAG_TRAJ_IDS:
        path = os.path.join(rt._DATASET_ROOT, spec_by_id[tid]['file'])
        u, q1, fs = load_telica_log(path, dtype=dtype)
        st = rt._build_state_traj_logical(q1, P, float(rt._ts), dtype)
        dev = (q1 - q1[0]).abs().max(dim=1).values
        motion = torch.nonzero(dev > MOTION_EPS_M)
        m0 = int(motion[0]) if len(motion) else 0
        start = max(0, min(m0 - PRE_MOTION_SAMPLES, q1.shape[0] - w_max))
        if q1.shape[0] - start < w_max:
            print(f'  WARNING {tid}: only {q1.shape[0]-start} samples from '
                  f'start, largest window will be truncated')
        data.append({'id': tid, 'u': u[0], 'q1': q1, 'x0': st[start],
                     'start': start, 'T': q1.shape[0],
                     'sigma': q1.std(dim=0).clamp(min=1e-9)})
        print(f'  {tid}: T={q1.shape[0]}  motion@{m0}  window start={start}')
    return data


def make_batch(data, W):
    x0    = torch.stack([d['x0'] for d in data])                              # (B,6)
    u_w   = torch.stack([d['u'][d['start']:d['start'] + W] for d in data])    # (B,W,3)
    q1_w  = torch.stack([d['q1'][d['start']:d['start'] + W] for d in data])   # (B,W,3)
    sigma = torch.stack([d['sigma'] for d in data])                           # (B,3)
    return x0, u_w, q1_w, sigma


def window_loss(block, batch, ts_tensor):
    """Exact training-loss form: sigma-normalized MSE on teacher-forced windows."""
    x0, u_w, q1_w, sigma = batch
    G, K, C, mh, alpha, beta, gamma, N0, N1, N2 = rt.tr._build_sim_params(block)
    res = simulate(x0, u_w, G, K, C, mh, alpha, beta, gamma, N0, N1, N2,
                   block._P, ts_tensor, bptt_mode='full', return_latents=False)
    err = (res.Y - q1_w) / sigma.unsqueeze(1)
    return err.pow(2).mean()


def fresh_block(dtype):
    return rt._lfr_pb.ParameterizedLFRBlock(RMSE_baseline=1.0).to(dtype=dtype)


def part1(data, w_grid, mults, dtype, ts_tensor):
    print('\n' + '=' * 74)
    print('PART 1: per-direction loss slices on real data')
    print('=' * 74)
    results = {}
    block = fresh_block(dtype)
    base = block.log_params.detach().clone()

    for W in w_grid:
        t0 = time.time()
        batch = make_batch(data, W)
        with torch.no_grad():
            block.log_params.data = base.clone()
            loss_nom = float(window_loss(block, batch, ts_tensor))
        # gradient norm at nominal
        block.log_params.data = base.clone()
        block.log_params.grad = None
        loss_g = window_loss(block, batch, ts_tensor)
        loss_g.backward()
        gnorm = float(block.log_params.grad.norm())

        slices = {}
        with torch.no_grad():
            for dname, idxs in _DIRECTIONS:
                row = {}
                for m in mults:
                    lp = base.clone()
                    for i in idxs:
                        lp[i] = float(np.log(m))
                    block.log_params.data = lp
                    row[m] = float(window_loss(block, batch, ts_tensor))
                slices[dname] = row
        block.log_params.data = base.clone()

        results[W] = {'loss_nom': loss_nom, 'grad_norm': gnorm, 'slices': slices}
        print(f'\nW = {W} ({W / 20:.1f} ms)   loss_nom = {loss_nom:.4e}   '
              f'grad_norm = {gnorm:.3e}   [{time.time()-t0:.0f} s]')
        print(f'  {"direction":<8} {"range%":>8}  {"argmin":>6}  verdict   '
              f'(loss ratio at each multiplier)')
        for dname, _ in _DIRECTIONS:
            row = results[W]['slices'][dname]
            ratios = {m: row[m] / loss_nom for m in mults}
            all_l = list(row.values()) + [loss_nom]
            rng = 100.0 * (max(all_l) - min(all_l)) / loss_nom
            amin = min(list(ratios.items()) + [(1.0, 1.0)], key=lambda kv: kv[1])[0]
            verdict = 'FLAT' if rng < FLAT_THRESHOLD_PCT else 'sensitive'
            rs = '  '.join(f'x{m}:{ratios[m]:.3f}' for m in mults)
            print(f'  {dname:<8} {rng:>7.1f}%  {amin:>6}  {verdict:<9} {rs}')

    # figure: one panel per direction, one line per W
    ncol = 4
    nrow = int(np.ceil(len(_DIRECTIONS) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow),
                            sharex=True)
    axs = np.atleast_2d(axs)
    for k, (dname, _) in enumerate(_DIRECTIONS):
        ax = axs[k // ncol, k % ncol]
        for W in w_grid:
            r = results[W]
            xs = [min(mults)] + [1.0] + [m for m in mults if m > 1]
            xs = sorted(set(list(mults) + [1.0]))
            ys = [r['slices'][dname].get(m, r['loss_nom']) / r['loss_nom']
                  for m in xs]
            ax.plot(xs, ys, 'o-', ms=3, lw=1, label=f'W={W}')
        ax.axhline(1.0, color='k', lw=0.5, ls=':')
        ax.axvline(1.0, color='k', lw=0.5, ls=':')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(dname, fontsize=9)
        ax.grid(alpha=0.3, which='both')
        if k == 0:
            ax.legend(fontsize=7)
    for k in range(len(_DIRECTIONS), nrow * ncol):
        axs[k // ncol, k % ncol].axis('off')
    fig.suptitle('Real-data loss slices: loss(direction x mult)/loss(nominal) '
                 'per window length\nflat line = invisible at that horizon; '
                 'minimum away from 1.0 = where the data pulls the parameter')
    fig.supxlabel('parameter multiplier')
    fig.supylabel('loss / loss_nominal')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fp = os.path.join(_SAVE_DIR, 'loss_slices.png')
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f'\n  figure -> {os.path.relpath(fp, _ROOT)}')
    return results


def part2(data, W, lrs, n_steps, dtype, ts_tensor):
    print('\n' + '=' * 74)
    print(f'PART 2: learning-rate probe at W = {W} ({n_steps} Adam steps each)')
    print('=' * 74)
    batch = make_batch(data, W)

    # outside check: one validation trajectory, full open-loop RMSE
    spec_by_id = {s['id']: s for s in rt.VAL_SPECS}
    vpath = os.path.join(rt._DATASET_ROOT, spec_by_id[_VAL_CHECK_ID]['file'])
    vu, vq1, _ = load_telica_log(vpath, dtype=dtype)
    vx0 = rt._build_state_traj_logical(vq1[:2], rt._P.to(dtype),
                                       float(rt._ts), dtype)[:1]

    def val_rmse(block):
        res = rt._run_no_grad(block, vx0, vu, ts_tensor)
        T = min(res.Y.shape[1], vq1.shape[0])
        return float((res.Y[0, :T] - vq1[:T]).pow(2).mean().sqrt())

    results = {}
    for lr in lrs:
        block = fresh_block(dtype)
        rmse0 = val_rmse(block)
        opt = torch.optim.Adam([block.log_params], lr=lr)
        hist_loss, hist_dev = [], []
        t0 = time.time()
        for k in range(n_steps):
            opt.zero_grad()
            loss = window_loss(block, batch, ts_tensor)
            loss.backward()
            opt.step()
            hist_loss.append(float(loss))
            hist_dev.append(float(block.log_params.detach().abs().max()))
        rmse1 = val_rmse(block)

        drops = sum(1 for a, b in zip(hist_loss, hist_loss[1:]) if b <= a * 1.001)
        monotone_frac = float(drops / max(1, len(hist_loss) - 1))
        max_change_pct = float(100.0 * (np.exp(hist_dev[-1]) - 1.0))
        ok = bool(hist_loss[-1] < hist_loss[0] and monotone_frac >= 0.8
                  and max_change_pct < 50.0)   # HEURISTIC verdict bounds
        results[lr] = dict(loss=hist_loss, dev=hist_dev,
                           monotone_frac=monotone_frac,
                           max_change_pct=max_change_pct,
                           val_rmse_before=rmse0, val_rmse_after=rmse1, ok=ok)
        print(f'  lr={lr:g}: loss {hist_loss[0]:.4e} -> {hist_loss[-1]:.4e}  '
              f'monotone {monotone_frac:.0%}  max param change {max_change_pct:.1f}%  '
              f'val RMSE {rmse0:.3e} -> {rmse1:.3e}  '
              f'{"OK" if ok else "reject"}  [{time.time()-t0:.0f} s]')

    good = [lr for lr in lrs if results[lr]['ok']]
    print(f'\n  verdict: largest acceptable lr = '
          f'{max(good) if good else "NONE (all rejected)"}')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for lr in lrs:
        r = results[lr]
        ax1.semilogy(r['loss'], 'o-', ms=3, lw=1, label=f'lr={lr:g}')
        ax2.plot(100.0 * (np.exp(np.array(r['dev'])) - 1.0), 'o-', ms=3, lw=1,
                 label=f'lr={lr:g}')
    ax1.set_xlabel('Adam step')
    ax1.set_ylabel('training window loss')
    ax1.set_title(f'Does the loss descend smoothly? (W={W})')
    ax2.set_xlabel('Adam step')
    ax2.set_ylabel('max parameter change [%]')
    ax2.set_title('Do parameters move by percents, not orders of magnitude?')
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fp = os.path.join(_SAVE_DIR, f'lr_probe_W{W}.png')
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f'  figure -> {os.path.relpath(fp, _ROOT)}')
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=['1', '2', 'both'], default='both')
    ap.add_argument('--fast', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--w', type=int, default=2600)
    ap.add_argument('--lrs', type=str, default='')
    ap.add_argument('--steps', type=int, default=25)
    args = ap.parse_args()

    dtype = rt.tr.DTYPE
    ts_tensor = torch.tensor(float(rt._ts), dtype=dtype)
    os.makedirs(_SAVE_DIR, exist_ok=True)

    if args.smoke:
        w_grid, mults = (650,), (2.0,)
        lrs, steps = (1e-2,), 2
        global _DIRECTIONS
        _DIRECTIONS = _DIRECTIONS[:2]
    elif args.fast:
        w_grid, mults = (650, 2600), (0.5, 2.0)
        lrs, steps = LR_GRID_FULL, args.steps
    else:
        w_grid, mults = W_GRID_FULL, MULTS_FULL
        lrs, steps = LR_GRID_FULL, args.steps
    if args.lrs:
        lrs = tuple(float(s) for s in args.lrs.split(','))

    print('=' * 74)
    print('DIAG WINDOW / LEARNING RATE  (real data only)')
    print('=' * 74)
    print(f'trajectories: {", ".join(_DIAG_TRAJ_IDS)}   W grid: {w_grid}   '
          f'mults: {mults}   lrs: {lrs}')

    w_max = max(max(w_grid), args.w)
    data = load_diag_data(w_max, dtype)

    out = {'w_grid': list(w_grid), 'mults': list(mults),
           'traj_ids': list(_DIAG_TRAJ_IDS)}
    if args.part in ('1', 'both'):
        r1 = part1(data, w_grid, mults, dtype, ts_tensor)
        out['part1'] = {str(W): {'loss_nom': r['loss_nom'],
                                 'grad_norm': r['grad_norm'],
                                 'slices': {d: {str(m): v for m, v in row.items()}
                                            for d, row in r['slices'].items()}}
                        for W, r in r1.items()}
    if args.part in ('2', 'both'):
        r2 = part2(data, args.w, lrs, steps, dtype, ts_tensor)
        out['part2'] = {str(lr): {k: v for k, v in r.items()}
                        for lr, r in r2.items()}

    with open(os.path.join(_SAVE_DIR, 'summary.json'), 'w') as fh:
        json.dump(out, fh, indent=2)
    print('\nsummary -> ' + os.path.relpath(
        os.path.join(_SAVE_DIR, 'summary.json'), _ROOT))


if __name__ == '__main__':
    main()
