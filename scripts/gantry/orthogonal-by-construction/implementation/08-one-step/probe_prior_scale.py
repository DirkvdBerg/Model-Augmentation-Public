"""Does the D-076 parameter prior pin the physical parameters at their detuned start? (D-193)

Turning on `joint_estimation` also turns on `Reduced_Gantry_State_Block.param_loss`, a
Lambda-weighted L2 anchored to `combo_init`, which in the D-192 arms is the deliberately
ten-percent-detuned starting point. Its weight `Lambda = param_rmse_baseline / |combo_init|` was
calibrated (D-076) against a different objective. If the prior's gradient dominates the data
term's, both arms sit at their detuned start and the projection comparison measures nothing.

This decides `param_prior` with a number. It walks the RECOVERY PATH in the block's own free
coordinates, from the detuned start (`s = 0`, where `free_params = 0`) to the true combinations
(`s = 1`), and at each point reports, per combination:

    g_data    d(closed-loop window MSE) / d(free_params)     the prior switched OFF
    g_prior   d(param_loss)             / d(free_params)     the prior alone
    ratio     |g_prior| / |g_data|                            > 1 means the prior wins

The two terms are additive in the objective, so measuring them separately is exact, not an
approximation. Both arms share `g_prior` exactly and differ only in `g_data`; this probe runs the
NO-PROJECTION arm, which is the reference the prior would pin.

FAST BY DESIGN: no OBC attach (no reference set, no basis), no compilation, a small window batch.
A few minutes on a GPU node, dominated by loading the 22 records.
"""
__project_origin__ = "added"

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rig import RESULTS_DIR, Tee, obc_config, build_system, make_batch, cuda_sync  # noqa: E402

BATCH = 128                                  # windows; the gradient is a mean, so this is enough
STRIDE = 500                                 # window decimation, for a quick training-data build
S_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)    # fraction of the way from detuned to true


def free_coordinate_for(block, target_combo):
    """The free coordinate at which `combinations_from_free` equals `target_combo`.

    Nine combinations are in log and `m_diff` is in a relative linear coordinate, so the two are
    inverted differently. `free = 0` is `combo_init`, the detuned start, by construction.
    """
    ci = block.combo_init.detach().double().cpu()
    tc = target_combo.double().cpu()
    f = torch.log(tc / ci)
    f[block.M_DIFF_IX] = tc[block.M_DIFF_IX] / ci[block.M_DIFF_IX] - 1.0
    return f


def main():
    sys.stdout = Tee(os.path.join(RESULTS_DIR, 'probe_prior_scale.log'))
    use_cuda = torch.cuda.is_available()
    cfg = obc_config(obc=False, device='cuda' if use_cuda else 'cpu', compile_mode=None,
                     stride=STRIDE)
    print('device %s | nf %d | batch %d | stride %d | param_rmse_baseline %g'
          % ('cuda' if use_cuda else 'cpu', cfg.nf, BATCH, STRIDE, cfg.param_rmse_baseline))
    t0 = time.perf_counter()
    rig = build_system(cfg, attach=False)        # no OBC: this question is arm-independent
    fs = rig.fs
    # `rig.phy` reaches the block through the OBC adapter, which is absent here.
    phy = next(b for b in fs.hfn.connected_blocks if hasattr(b, 'combo_init'))
    if use_cuda:
        fs.cuda()
    dev = next(phy.parameters()).device
    batch, kw = make_batch(rig, B=BATCH, seed=0, device=dev)
    print('build + data: %.1f s; windows in batch %d' % (time.perf_counter() - t0, batch[0].shape[0]))

    # The true combinations, and the free coordinate that reaches them.
    from model_augmentation.systems import gantry_ss as gss
    nominal_raw = torch.stack([getattr(gss, n) for n in phy.PARAM_NAMES]).double()
    v_true = free_coordinate_for(phy, phy.combos_of(nominal_raw, float(gss.Lb))).to(dev)
    names = phy.COMBO_NAMES
    print('detune in free coordinates (s = 1 is recovery): '
          + ' '.join('%s=%+.4f' % (n, float(v)) for n, v in zip(names, v_true)))
    print('Lambda = param_rmse_baseline / |combo_init|: '
          + ' '.join('%s=%.3e' % (n, float(v)) for n, v in zip(names, phy.Lambda_combo)))

    rows = []
    for s in S_GRID:
        with torch.no_grad():
            phy.free_params.copy_((s * v_true).to(phy.free_params.dtype))
        # --- data term alone: the prior contributes 0.0 and no gradient ---------------
        phy.flag_loss_reg = False
        fs.optimizer.zero_grad()
        L = fs.loss(*batch[:4], **kw)
        L.backward()
        g_data = phy.free_params.grad.detach().double().cpu().clone()
        mse = float(L)
        # --- prior alone: no rollout, so this is cheap -------------------------------
        phy.flag_loss_reg = True
        fs.optimizer.zero_grad()
        P = phy.param_loss()
        g_prior = (torch.autograd.grad(P, phy.free_params)[0].detach().double().cpu()
                   if torch.is_tensor(P) and P.requires_grad
                   else torch.zeros_like(g_data))
        fs.optimizer.zero_grad()
        cuda_sync()
        ratio = float(g_prior.norm() / (g_data.norm() + 1e-300))
        rows.append({'s': s, 'mse': mse, 'param_loss': float(P),
                     'g_data': g_data.tolist(), 'g_prior': g_prior.tolist(),
                     'norm_ratio': ratio})
        print('\ns = %.2f   window MSE %.4e   param_loss %.4e   '
              '||g_prior|| / ||g_data|| = %.3e' % (s, mse, float(P), ratio))
        print('  %-9s %12s %12s %10s' % ('combo', 'g_data', 'g_prior', 'ratio'))
        for i, n in enumerate(names):
            gd, gp = float(g_data[i]), float(g_prior[i])
            print('  %-9s %+12.3e %+12.3e %10.2e' % (n, gd, gp, abs(gp) / (abs(gd) + 1e-300)))

    # ---- verdict -----------------------------------------------------------------------
    print('\n==== verdict ====')
    mid = [r for r in rows if r['s'] == 0.5][0]
    dominated = [names[i] for i in range(len(names))
                 if abs(mid['g_prior'][i]) > abs(mid['g_data'][i])]
    print('At the midpoint of the recovery path (s = 0.50) the prior gradient exceeds the data '
          'gradient on %d of %d combinations: %s' % (len(dominated), len(names), dominated or 'none'))
    print('||g_prior|| / ||g_data|| along the path: '
          + ' '.join('s=%.2f:%.2e' % (r['s'], r['norm_ratio']) for r in rows))
    print('\nREAD THIS AS: a combination whose ratio exceeds 1 is held at its DETUNED value by the')
    print('prior, in BOTH arms, for a reason unrelated to the projection. If that list is not')
    print('empty, set param_prior=False in the _SHARED block of gantry_interconnect_dynamic.py')
    print('(or lower param_rmse_baseline until it is) before launching the pair. The choice is')
    print('recorded as PARAM_PRIOR in config.json either way (D-193).')

    out = {'device': str(dev), 'nf': cfg.nf, 'batch': BATCH, 'stride': STRIDE,
           'param_rmse_baseline': cfg.param_rmse_baseline, 'combos': list(names),
           'v_true_free': v_true.double().cpu().tolist(),
           'lambda_combo': [float(v) for v in phy.Lambda_combo],
           'rows': rows, 'dominated_at_half': dominated}
    with open(os.path.join(RESULTS_DIR, 'probe_prior_scale.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print('\nwritten', os.path.join(RESULTS_DIR, 'probe_prior_scale.json'))


if __name__ == '__main__':
    main()
