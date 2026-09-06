"""Test 1 (rank of Psi) and Test 2 (leakage ratio) on a trained checkpoint.

These are the two measurements that decide whether the derivation in `affine_leakage.py` is a
contribution or a footnote. The derivation proves the pointwise penalty leaves the state path
unconstrained; it says nothing about whether a TRAINED network actually uses that freedom.

TWO AMBIENT SPACES, and keeping them apart is the point of this script.

  PER-WINDOW, in R^nx.  Psi_w = sum_j (prod_{m>j} A_m) Phi_j is 6 x 14, so span(Psi_w) is a
      subspace of R^6. If its rank is 6 the projection is the IDENTITY, `Q^T dx = dx`, and the
      per-window orthogonality condition is vacuous. This is the space `affine_leakage.py`
      Part D used, and measuring the rank is what tells us whether that framing was wrong.

  STACKED, over all windows.  This is the true analogue of the deployed penalty. Gyorok's Q is
      an orthonormal basis of the column span of the stacked regressor, ambient dimension
      n_points * n_rows, with span(Q) of dimension at most n_theta + 1. Here the ambient space
      is n_windows * nx and span(Q_Psi) is at most 14, so the projection is a genuine
      restriction and admits a nonzero solution. Any deployable penalty must be built here.

WHAT IS MEASURED, per window, from x_aug = 0 and the recorded inputs (open loop, so the
quantity is well defined; the runs' training loss was closed loop, which does not change what
the augmentation contributes to the state):

    dx        = x_T(full)  - x_T(ANN output forced to zero)      total ANN contribution
    dx_state  = x_T(full)  - x_T(x_aug forced to zero each step) the part via the state path
    r_state   = ||dx_state|| / ||dx||                            is the state path used at all
    r_leak    = ||Q^T dx_state|| / ||dx||                        THE HEADLINE
    r_total   = ||Q^T dx|| / ||dx||                              for scale

Rule 1: every quantity is a Jacobian of the baseline, a network output, or measured data. No
truth model. All of it computes on Telica.

Run: conda run -n GraduationProject python scripts/gantry/orthogonality/code/leakage_measure.py
"""
import os
import sys
import json

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

from gantry_dynamic.config import RunConfig
from gantry_dynamic.data import load_datasets, compute_normalization
from gantry_dynamic.orth_penalty import theta_bar_for, _x_logical_from_data
from model_augmentation.fit_systems.blocks import Parameterized_Gantry_State_Block

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'leakage_measure.json')
CKPT = os.path.join(HERE, '..', '..', 'meeting', 'meeting-07-09-2026', 'server',
                    'checkpoints', 'SSE_Interconnect_Composed_p2LPDA_best.pth')  # run 81262

F64 = torch.float64
NX_PHYS, NU = 6, 3
T_WIN = int(os.environ.get('T_WIN', 100))   # steps per window (training nf was 400)
N_WIN = int(os.environ.get('N_WIN', 48))    # windows sampled across the validation records
RANK_TOL = 1e-12


def cfg_81262():
    """Run 81262 as logged: nx_ann=8, linear_map encoder, up_sample=1, 4 kHz, orth OFF."""
    return RunConfig(mode='augmentation_ma50_b140-230_a6_z03', encoder_init='linear_map',
                     nx_ann=8, ann_route_ix=tuple(range(14)), n_nodes_per_layer=24,
                     n_hidden_layers=3, up_sample=1, fs_new=4000, stride=10,
                     na_nb_override=29, nf_override=400, joint_estimation=False,
                     save_flag=False, device='cpu')


def jacobians(blk, x6, u3, theta_t):
    """(A, Phi) at one point: A = d x_next / d x (6x6), Phi = d x_next / d theta (6x14)."""
    x = x6.clone().requires_grad_(True)
    z = torch.cat([x.view(1, NX_PHYS, 1), u3.view(1, NU, 1)], dim=1)
    xn = blk.nonlinear_function(z).view(NX_PHYS)
    A = torch.zeros(NX_PHYS, NX_PHYS, dtype=F64)
    Phi = torch.zeros(NX_PHYS, 14, dtype=F64)
    for r in range(NX_PHYS):
        gx, gp = torch.autograd.grad(xn[r], [x, blk.log_params],
                                     retain_graph=(r < NX_PHYS - 1))
        A[r] = gx
        Phi[r] = gp / theta_t          # d/dlog theta -> d/dtheta
    return A.detach(), Phi.detach()


def main():
    cfg = cfg_81262()
    print('Loading data and normalisation for run 81262 ...')
    data = load_datasets(cfg)
    norm = compute_normalization(cfg, data)
    theta_bar = theta_bar_for(cfg)
    theta_t = torch.tensor(theta_bar, dtype=F64)

    blk = Parameterized_Gantry_State_Block(
        Y_op=None, std_x=norm.std_x, std_u=norm.std_u,
        x_mean=norm.x_mean, u_mean=norm.u_mean, Ts=cfg.ts_new, up_sample=cfg.up_sample,
        params_init=theta_t).to(F64)
    blk.eval()

    print(f'Loading checkpoint {os.path.basename(CKPT)} ...')
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    hfn = ck['hfn'].to(torch.float32)
    hfn.eval()
    nxd = ck['nx']
    nx_ann = nxd - NX_PHYS
    print(f'  nx = {nxd} (nx_ann = {nx_ann})   bestfit = {ck["bestfit"]:.4e}')
    ann = [b for b in hfn.connected_blocks if type(b).__name__ == 'Static_ANN_Block'][0]

    # ---- validation records, normalised, plus data-derived physical states for x0 ----
    recs = []
    for sd in data.val_list:
        xl = _x_logical_from_data(sd)
        xn_ = (xl - norm.x_mean.flatten()) / norm.std_x.flatten()
        un_ = (sd.u - norm.u_mean.flatten()) / norm.std_u.flatten()
        recs.append((xn_.astype(np.float64), un_.astype(np.float64)))
    print(f'  {len(recs)} validation records, {recs[0][0].shape[0]} samples each')

    rng = np.random.default_rng(0)
    starts = [(rng.integers(len(recs)), rng.integers(1, recs[0][0].shape[0] - T_WIN - 1))
              for _ in range(N_WIN)]

    # ---- rollout helper: three variants sharing one input sequence ----
    def roll(x0, useq, mode):
        """mode: 'full' | 'frozen' (x_aug := 0 each step) | 'annoff' (ANN output := 0)."""
        x = torch.zeros(1, nxd, dtype=torch.float32)
        x[0, :NX_PHYS] = torch.tensor(x0, dtype=torch.float32)
        orig = ann.forward
        if mode == 'annoff':
            ann.forward = lambda z: torch.zeros(z.shape[0], ann.nw, 1, dtype=z.dtype)
        try:
            for k in range(useq.shape[0]):
                u = torch.tensor(useq[k], dtype=torch.float32).view(1, NU)
                _, x = hfn(x, u)
                if mode == 'frozen':
                    x = x.clone()
                    x[:, NX_PHYS:] = 0.0
        finally:
            ann.forward = orig
        return x[0, :NX_PHYS].detach().numpy().astype(np.float64)

    Psi_list, dx_list, dxs_list, rank_w = [], [], [], []
    print(f'\nRolling out {N_WIN} windows of {T_WIN} steps (3 variants each) '
          f'and accumulating Psi ...')
    for wi, (ri, k0) in enumerate(starts):
        xs, us = recs[ri]
        x0 = xs[k0]
        useq = us[k0:k0 + T_WIN]

        # Psi_w = sum_j (prod_{m=j+1..T-1} A_m) Phi_j, along the realised trajectory.
        Psi = np.zeros((NX_PHYS, 14))
        Prod = np.eye(NX_PHYS)
        for j in range(T_WIN - 1, -1, -1):
            A, Phi = jacobians(blk, torch.tensor(xs[k0 + j], dtype=F64),
                               torch.tensor(us[k0 + j], dtype=F64), theta_t)
            Psi += Prod @ Phi.numpy()
            Prod = Prod @ A.numpy()
        Psi_list.append(Psi)
        rank_w.append(np.linalg.matrix_rank(Psi, tol=RANK_TOL * np.linalg.norm(Psi, 2)))

        xf = roll(x0, useq, 'full')
        xz = roll(x0, useq, 'frozen')
        xo = roll(x0, useq, 'annoff')
        dx_list.append(xf - xo)
        dxs_list.append(xf - xz)
        if (wi + 1) % 12 == 0:
            print(f'  window {wi+1}/{N_WIN}', flush=True)

    Psi_w = np.array(Psi_list)                 # (N_WIN, 6, 14)
    dx = np.array(dx_list)                     # (N_WIN, 6)
    dxs = np.array(dxs_list)

    res = {'run': 81262, 'ckpt': os.path.basename(CKPT), 'T_win': T_WIN, 'n_win': N_WIN,
           'nx_ann': nx_ann}

    # ================================================================ TEST 1
    print('\n' + '=' * 78)
    print('TEST 1  rank of Psi, in both ambient spaces')
    print('=' * 78)
    rw = np.array(rank_w)
    print(f'  PER-WINDOW  Psi_w is 6x14.  rank: min {rw.min()}  median {int(np.median(rw))}'
          f'  max {rw.max()}   (n_x = 6)')
    if rw.min() == NX_PHYS:
        print('  -> span(Psi_w) = R^6 in every window. A per-window projection is the')
        print('     IDENTITY, so the per-window condition of affine_leakage Part D is')
        print('     VACUOUS, not merely restrictive. That framing was in the wrong space.')
    res['rank_per_window'] = dict(min=int(rw.min()), median=int(np.median(rw)),
                                  max=int(rw.max()), nx=NX_PHYS)

    Psi_stack = Psi_w.reshape(N_WIN * NX_PHYS, 14)
    Qs, Ss, _ = np.linalg.svd(Psi_stack, full_matrices=False)
    n_keep = int(np.sum(Ss > RANK_TOL * Ss[0]))
    Q = Qs[:, :n_keep]
    print(f'  STACKED     Psi is {Psi_stack.shape[0]}x14.  numerical rank {n_keep}'
          f'   (ambient {Psi_stack.shape[0]})')
    print(f'              sigma ratio kept: {Ss[n_keep-1]/Ss[0]:.2e}, '
          f'spectrum spans {Ss[0]:.3e} .. {Ss[-1]:.3e}')
    print(f'  -> span(Q) is {n_keep} of {Psi_stack.shape[0]} dimensions, a genuine')
    print('     restriction. This is the space a deployable penalty must live in.')
    res['rank_stacked'] = dict(rank=int(n_keep), ambient=int(Psi_stack.shape[0]),
                               sigma_ratio=float(Ss[n_keep-1]/Ss[0]))

    # ================================================================ TEST 2
    print('\n' + '=' * 78)
    print('TEST 2  leakage ratio on the trained checkpoint')
    print('=' * 78)
    dxf = dx.reshape(-1)
    dxsf = dxs.reshape(-1)
    n_dx = np.linalg.norm(dxf)
    r_state = np.linalg.norm(dxsf) / n_dx
    r_leak = np.linalg.norm(Q.T @ dxsf) / n_dx
    r_total = np.linalg.norm(Q.T @ dxf) / n_dx
    print(f'  ||dx||                    = {n_dx:.4e}   (normalised state units)')
    print(f'  r_state = ||dx_state||/||dx||       = {r_state:9.4f}'
          '   <- fraction of the ANN effect flowing through x_aug')
    print(f'  r_leak  = ||Q^T dx_state||/||dx||   = {r_leak:9.4f}'
          '   <- HEADLINE: unconstrained in-span part')
    print(f'  r_total = ||Q^T dx||/||dx||         = {r_total:9.4f}'
          '   <- total in-span fraction, for scale')
    if r_state > 0:
        print(f'  in-span fraction OF the state path  = {r_leak/r_state:9.4f}')
    res['ratios'] = dict(norm_dx=float(n_dx), r_state=float(r_state),
                         r_leak=float(r_leak), r_total=float(r_total))

    # per-window spread, so a single aggregate cannot hide a heavy tail
    per = []
    for w in range(N_WIN):
        d, ds = dx[w], dxs[w]
        nd = np.linalg.norm(d)
        if nd > 0:
            per.append(np.linalg.norm(ds) / nd)
    per = np.array(per)
    print(f'\n  per-window r_state: median {np.median(per):.4f}  '
          f'p90 {np.percentile(per,90):.4f}  max {per.max():.4f}')
    res['r_state_per_window'] = dict(median=float(np.median(per)),
                                     p90=float(np.percentile(per, 90)),
                                     max=float(per.max()))

    # Per-window r_leak, and how concentrated the norm-weighted aggregate is. A single
    # ratio over the stack is dominated by the largest windows; if a handful of windows
    # carry most of ||dx||^2 the aggregate is not describing typical behaviour.
    rl = []
    for w in range(N_WIN):
        nd = np.linalg.norm(dx[w])
        if nd > 0:
            qs = Q[w * NX_PHYS:(w + 1) * NX_PHYS, :]
            rl.append(np.linalg.norm(qs.T @ dxs[w]) / nd)
    rl = np.array(rl)
    e = (dx ** 2).sum(axis=1)
    share = np.sort(e)[::-1].cumsum() / e.sum()
    print(f'  per-window r_leak : median {np.median(rl):.4f}  p90 {np.percentile(rl,90):.4f}'
          f'  max {rl.max():.4f}')
    print(f'  concentration     : top 1 window = {share[0]*100:.1f}% of ||dx||^2, '
          f'top 5 = {share[4]*100:.1f}%, top 10 = {share[9]*100:.1f}%')
    res['r_leak_per_window'] = dict(median=float(np.median(rl)),
                                    p90=float(np.percentile(rl, 90)), max=float(rl.max()))
    res['concentration'] = dict(top1=float(share[0]), top5=float(share[4]),
                                top10=float(share[9]))

    # ---- NULL BASELINE. span(Q) is n_keep of (N_WIN*nx) dimensions, so a vector with no
    # relationship to the parameter subspace still projects with ratio ~ sqrt(n_keep/ambient).
    # Quoting r_leak without this is meaningless. Two nulls: the analytic chance level, and a
    # randomisation that preserves the per-window norms (so the heavy concentration in a few
    # windows cannot by itself manufacture alignment).
    amb = N_WIN * NX_PHYS
    chance = np.sqrt(n_keep / amb)
    wnorm = np.linalg.norm(dxs, axis=1)
    rngn = np.random.default_rng(1)
    null = []
    for _ in range(500):
        v = rngn.standard_normal((N_WIN, NX_PHYS))
        v *= (wnorm / np.maximum(np.linalg.norm(v, axis=1), 1e-300))[:, None]
        vf = v.reshape(-1)
        null.append(np.linalg.norm(Q.T @ vf) / n_dx)
    null = np.array(null)
    print('')
    print(f'  NULL   analytic chance level sqrt({n_keep}/{amb})        = {chance:9.4f}')
    print(f'         randomisation, per-window norms preserved  = {null.mean():9.4f}'
          f'  (p95 {np.percentile(null,95):.4f})')
    print(f'         observed r_leak / null mean                = '
          f'{r_leak/null.mean():9.2f}x')
    if r_leak > np.percentile(null, 99):
        print('         -> the alignment is NOT chance. The trained network places its')
        print('            state-path contribution inside the parameter subspace far more')
        print('            than an unrelated signal of the same shape would.')
    else:
        print('         -> NOT distinguishable from chance. The mechanism is available to')
        print('            the network but there is no evidence it is being used.')
    res['null'] = dict(chance=float(chance), rand_mean=float(null.mean()),
                       rand_p95=float(np.percentile(null, 95)),
                       rand_p99=float(np.percentile(null, 99)),
                       ratio_over_null=float(r_leak / null.mean()))

    print('\n  CAVEATS, binding on how these numbers may be quoted:')
    print('   * run 81262 has orth=OFF and joint=False, so this is the UNREGULARISED')
    print('     network. It measures how much leakage training produces when nothing')
    print('     opposes it, which is the right baseline but is not a test of the penalty.')
    print('   * open-loop rollout from data-derived x0 with x_aug = 0; the training loss')
    print('     was closed loop. Category: data-computable (Rule 1).')

    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=2)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
