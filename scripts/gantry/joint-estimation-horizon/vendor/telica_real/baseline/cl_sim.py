"""Closed-loop replay of the baseline with the Telica controller, batched over records (G4).

Two forms, stacked in one batch:
  DIRECT    u = u_ff + K(r - y_model)          the machine's loop, from the reference
  RESIDUAL  u = u_data + K(y_data - y_model)   the training form (closed_loop.py)
K is the identified controller (G3) as a physical state space, error [m] -> force [N].
float64, 20 kHz, the friction block's own RK4 step (friction/karnopp.py).
"""
import numpy as np
import torch

from params import telica_params as tp
from friction.karnopp import GantryFrictionBlock, rebuild_float64, _unit_norm
from controller.telica_bank import physical_ss

D = torch.float64
P = np.array([[1.0, 1.0, 0.0], [tp.Lb / 2, -tp.Lb / 2, 0.0], [0.0, 0.0, 1.0]])


def make_block(raw=None, cc=tp.CC, mode='karnopp', ts=tp.TS):
    """Friction block at raw params `raw` (dict of the 14 + optional Lb), float64, LPV branch."""
    blk = GantryFrictionBlock(cc=cc, mode=mode, Y_op=None, Ts=ts, up_sample=1, **_unit_norm())
    blk = rebuild_float64(blk, p=raw)
    blk.cc_init = torch.as_tensor(cc, dtype=D).reshape(1, 3).clone()
    return blk


def pack(recs, pre=2000):
    """Records -> padded arrays from `pre` samples before motion to the end.

    recs: list of loader dicts (with 'u', 'u_fb'). Returns dict of (B, T, 3) float64 arrays + mask.
    """
    segs = []
    for d in recs:
        ks = max(0, d['motion_idx'] - pre)
        segs.append(dict(r=d['r'][ks:], y=d['q1'][ks:], u=d['u'][ks:],
                         uff=(d['u'] - d['u_fb'])[ks:], e=d['e'][ks:], k_motion=d['motion_idx'] - ks))
    T = max(len(s['r']) for s in segs)
    B = len(segs)
    out = {k: np.zeros((B, T, 3)) for k in ('r', 'y', 'u', 'uff', 'e')}
    mask = np.zeros((B, T), bool)
    for i, s in enumerate(segs):
        n = len(s['r'])
        for k in out:
            out[k][i, :n] = s[k]
            out[k][i, n:] = s[k][-1]              # hold the last value in the padding
        mask[i, :n] = True
    out['mask'] = mask
    out['k_motion'] = np.array([s['k_motion'] for s in segs])
    out['n'] = np.array([len(s['r']) for s in segs])
    return out


def simulate(block, data, source='identified', log_every=4000, log=print):
    """Run DIRECT and RESIDUAL forms for every record in `data` (from pack).

    Returns (y_direct, y_residual, u_direct, u_residual), each (B, T, 3) float64 numpy arrays.
    """
    A, Bc, C, Dc = (torch.as_tensor(m, dtype=D) for m in physical_ss(source))
    Bn = data['r'].shape[0]
    T = data['r'].shape[1]
    t = lambda a: torch.as_tensor(a, dtype=D)                          # noqa: E731
    r, y, u, uff = t(data['r']), t(data['y']), t(data['u']), t(data['uff'])
    Pt = t(P)
    # stacked batch: first Bn = direct, next Bn = residual
    q0 = torch.linalg.solve(Pt.T, y[:, 0, :].T).T                      # stage -> logical
    x = torch.cat([q0, torch.zeros_like(q0)], 1).repeat(2, 1).unsqueeze(-1)   # (2B, 6, 1)
    xc = torch.zeros(2 * Bn, A.shape[0], dtype=D)
    is_dir = torch.cat([torch.ones(Bn, 1, dtype=D), torch.zeros(Bn, 1, dtype=D)])
    yout = torch.empty(2 * Bn, T, 3, dtype=D)
    uout = torch.empty(2 * Bn, T, 3, dtype=D)
    import time
    t0 = time.time()
    with torch.no_grad():
        for k in range(T):
            ym = (x[:, :3, 0] @ Pt)                                   # stage positions
            yout[:, k] = ym
            ref = torch.cat([r[:, k], y[:, k]])                        # direct: r ; residual: y_data
            e = ref - ym
            ufb = xc @ C.T + e @ Dc.T
            u_k = is_dir * torch.cat([uff[:, k], uff[:, k]]) + (1 - is_dir) * torch.cat(
                [u[:, k], u[:, k]]) + ufb
            uout[:, k] = u_k
            xc = xc @ A.T + e @ Bc.T
            x = block.nonlinear_function(torch.cat([x, u_k.unsqueeze(-1)], 1))
            if not torch.isfinite(x).all():
                log(f'[cl_sim] non-finite state at step {k}; stopping this batch')
                yout[:, k:] = float('nan')
                break
            if log_every and k % log_every == 0 and k:
                log(f'[cl_sim] step {k}/{T} ({time.time() - t0:.0f} s)')
    Y = yout.numpy(); U = uout.numpy()
    return Y[:Bn], Y[Bn:], U[:Bn], U[Bn:]
