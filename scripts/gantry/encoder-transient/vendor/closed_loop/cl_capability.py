"""CAN THIS ANN REPRESENT THE MISSING PHYSICS AT ALL, AND DOES THE OBJECTIVE PREFER IT?

THE QUESTION, and why it outranks every optimiser question
----------------------------------------------------------
Training plateaus with the model 21x above the per-window oracle floor (D-147). Two readings
survive: the optimiser cannot find the solution, or the solution is not in this function class.
They lead to opposite work, and no amount of epochs, learning rates or optimiser fixes
distinguishes them. This does, because the missing physics is KNOWN here: the data comes from
`plant.deriv8`, the FP model plus the true absorber, and the baseline is `plant.deriv6`.

THE TARGET IS EXACTLY COMPUTABLE, and it is a STATIC function of (x, u)
----------------------------------------------------------------------
The ANN is added to the model's next state:  x[k+1] = phi_base(x[k], u[k]) + W_a f(x[k], u[k]).
So the correction it must supply is

    target(k) = phi_true(x[k], u[k]) - phi_base(x[k], u[k])                       (normalised)

with `phi_true` an RK4 step of `deriv8` and `phi_base` the model's OWN `Gantry_State_Block` at the
same Ts and up_sample, so the discretisation is shared and does not appear in the target. Both
sides are functions of the current state and input alone, which is precisely the ANN's input
(`nz = nxd + nu`). Representability is therefore a plain regression question, and this script
answers it by regression rather than by training a rollout.

THE LATENT GAUGE, stated because it is a real caveat
----------------------------------------------------
Rows 6-7 of the model state are the ANN's own latents; nothing fixes their scale or meaning, and
`compute_normalization` only produces `std_x` for the six PHYSICAL rows. This script FIXES that
gauge to the true absorber coordinates, normalised by their own training-set std. That is legitimate
for an existence question ("does a solution exist in this class") and NOT for "training will find
this one": training is free to choose any other gauge.

WHAT THIS IS NOT
----------------
Not an initialisation for a deliverable model. The target is built from `x_aug`, i.e. oracle
information, and a thesis claim that the method learns the physics cannot rest on a model that was
handed the physics. It is a DIAGNOSTIC CEILING and is reported as one.

THREE OUTCOMES, decided before running
--------------------------------------
  regression fits, planted model scores near the floor      -> the class is fine and the objective
                                                               ranks the true solution correctly;
                                                               the failure is search. Optimiser work.
  regression fits, planted model scores far from the floor  -> one-step correctness does not survive
                                                               the rollout, or the objective does not
                                                               prefer the true solution. That would
                                                               outrank everything else in the file.
  regression does not fit                                   -> capacity, routing or input set. No
                                                               optimiser change can help.

Usage: python -u cl_capability.py
"""
__project_origin__ = "added"

import dataclasses
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
GANTRY = os.path.join(REPO, 'scripts', 'gantry')
for p in (REPO, GANTRY, HERE, os.path.join(GANTRY, 'drift-demo'),
          os.path.join(GANTRY, 'msd-offset')):
    if p not in sys.path:
        sys.path.insert(0, p)

import deepSI                                                              # noqa: E402
import demo_common as dm                                                   # noqa: E402
from demo_common import CFG                                                # noqa: E402
from gantry_dynamic.data import load_traj, load_mat_aug, TRAIN_FILES, VAL_FILES   # noqa: E402
from cl_pipeline import build_closed_loop                                  # noqa: E402
from model_augmentation.fit_systems.blocks import Static_ANN_Block, Gantry_State_Block  # noqa: E402
from model_augmentation.fit_systems.closed_loop import (                   # noqa: E402
    closed_loop_rollout, closed_loop_free_run_rms, closed_loop_window_rms, window_starts,
    make_window_tensors)
import plant as PL                                                         # noqa: E402

# model rows  [X, Th, Y, dX, dTh, dY, da, vda]
# deriv8 rows [X, Th, Y, da, dX, dTh, dY, vda]
M2D = np.array([0, 1, 2, 6, 3, 4, 5, 7])      # model order -> deriv8 order
D2M = np.array([0, 1, 2, 4, 5, 6, 3, 7])      # deriv8 order -> model order
SUB = int(os.environ.get('CL_CAP_SUB', 5))    # sample stride for the regression set
N_REC = int(os.environ.get('CL_CAP_NREC', 14))
STEPS = int(os.environ.get('CL_CAP_STEPS', 4000))
OUT = os.path.join(HERE, 'runs', 'cl_capability.json')
t0 = time.time()


def true_step(x8_model, u_stage, ts, up):
    """One RK4 step of `plant.deriv8`, at the MODEL's rate and up_sample. Physical, model order.

    Same integrator and sub-stepping as `Gantry_State_Block.nonlinear_function`, so the target
    below is the correction the model's own discretisation needs, not a discretisation error.
    """
    x = np.asarray(x8_model, dtype=np.float64)[:, M2D]        # -> deriv8 order
    ul = (PL.P_np @ np.asarray(u_stage, dtype=np.float64).T).T
    h = ts / up
    for _ in range(up):
        k1 = np.array([PL.deriv8(xi, ui) for xi, ui in zip(x, ul)])
        k2 = np.array([PL.deriv8(xi, ui) for xi, ui in zip(x + 0.5 * h * k1, ul)])
        k3 = np.array([PL.deriv8(xi, ui) for xi, ui in zip(x + 0.5 * h * k2, ul)])
        k4 = np.array([PL.deriv8(xi, ui) for xi, ui in zip(x + h * k3, ul)])
        x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x[:, D2M]                                          # -> model order


def main():
    cfg = dataclasses.replace(CFG, seed=0, ann_route_ix=tuple(range(8)), lr=1e-7)
    ts, up = cfg.ts_new, cfg.hp['up_sample']
    print('=' * 100)
    print('CAPABILITY: can the ANN represent the absorber correction, and does the loss prefer it?')
    print('=' * 100)
    fs, norm, K0, na, nb, na_r, nb_r = dm.build_pipeline(cfg=cfg, verbose=True)
    tr_files = list(TRAIN_FILES)[:N_REC]
    tr = [load_traj(f, cfg) for f in tr_files]
    vl_files = list(VAL_FILES)
    vl = [load_traj(f, cfg) for f in vl_files]
    val_data = deepSI.System_data_list(vl)
    fs.simulator = build_closed_loop(fs, norm, cfg, train_files=tr_files, val_files=vl_files,
                                     val_data=val_data, verbose=True)
    bank = fs.simulator.bank
    phy = next(m for m in fs.hfn.connected_blocks if isinstance(m, Gantry_State_Block))
    ann0 = next(m for m in fs.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
    x_mean = np.asarray(norm.x_mean).reshape(1, 6)
    std_x = np.asarray(norm.std_x).reshape(1, 6)
    u_mean = np.asarray(norm.u_mean).reshape(1, 3)
    std_u = np.asarray(norm.std_u).reshape(1, 3)

    # ---- the regression set -------------------------------------------------------------------
    print('\nbuilding the target on %d records, every %dth sample' % (len(tr_files), SUB),
          flush=True)
    aug_all = [load_mat_aug(f, cfg)[3] for f in tr_files]
    std_aug = np.concatenate(aug_all).std(axis=0).reshape(1, 2)     # the latent gauge, see docstring
    print('latent gauge (training-set std of the TRUE absorber states): '
          'delta_a %.4e m, vdelta_a %.4e m/s' % (std_aug[0, 0], std_aug[0, 1]))

    Z, W = [], []
    for f, sd, x_aug in zip(tr_files, tr, aug_all):
        _, _, x_log, _ = load_mat_aug(f, cfg)
        n = len(sd.u) - 1
        ix = np.arange(0, n, SUB)
        x8 = np.concatenate([np.asarray(x_log, dtype=np.float64)[ix],
                             np.asarray(x_aug, dtype=np.float64)[ix]], axis=1)   # model order
        u = np.asarray(sd.u, dtype=np.float64)[ix]
        xn = np.concatenate([(x8[:, :6] - x_mean) / std_x, x8[:, 6:] / std_aug], axis=1)
        un = (u - u_mean) / std_u
        # phi_base: the model's own baseline step on the six physical rows. Rows 6-7 have NO
        # dynamics without the ANN, so their baseline next state is exactly zero.
        with torch.no_grad():
            z = torch.as_tensor(np.concatenate([xn[:, :6], un], axis=1)[:, :, None],
                                dtype=torch.float32)
            xp_base6 = phy.nonlinear_function(z)[:, :, 0].numpy().astype(np.float64)
        xp_base = np.concatenate([xp_base6, np.zeros((len(ix), 2))], axis=1)
        xp_true = true_step(x8, u, ts, up)
        xp_true_n = np.concatenate([(xp_true[:, :6] - x_mean) / std_x,
                                    xp_true[:, 6:] / std_aug], axis=1)
        Z.append(np.concatenate([xn, un], axis=1))
        W.append(xp_true_n - xp_base)
    Z = np.concatenate(Z).astype(np.float32)
    W = np.concatenate(W).astype(np.float32)
    print('regression set: %d samples, nz = %d, nw = %d   [%.0fs]'
          % (len(Z), Z.shape[1], W.shape[1], time.time() - t0))
    rows = ['X', 'Th', 'Y', 'dX', 'dTh', 'dY', 'da', 'vda']
    print('\ntarget magnitude per row (normalised state units), i.e. what the ANN must output:')
    for i, r in enumerate(rows):
        print('  %-4s std %.4e   max %.4e' % (r, W[:, i].std(), np.abs(W[:, i]).max()))

    # ---- regression ---------------------------------------------------------------------------
    from model_augmentation.utils.torch_nets import feed_forward_nn
    Zt = torch.as_tensor(Z)[:, :, None]
    Wt = torch.as_tensor(W)[:, :, None]
    scale = Wt.std(dim=0, keepdim=True).clamp_min(1e-20)   # per-row natural magnitude
    n = len(Zt)

    def fit(row_scaled, label):
        """Regress the SAME architecture onto the target. row_scaled=True divides the target by
        its per-row std, so every output is O(1) and the scale is folded back into the last layer
        afterwards. That is the SAME function class either way: the arms differ only in the
        optimisation geometry, which is exactly the hypothesis under test."""
        torch.manual_seed(0)
        net = Static_ANN_Block(nz=Z.shape[1], nw=W.shape[1],
                               n_nodes_per_layer=cfg.hp['n_nodes_per_layer'],
                               n_hidden_layers=cfg.hp['n_hidden_layers'],
                               net=feed_forward_nn,
                               activation=torch.nn.Tanh).to(cfg.dtype_pt)
        tgt = Wt / scale if row_scaled else Wt
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
        print('\n[%s] %d params, %d steps'
              % (label, sum(p.numel() for p in net.parameters()), STEPS), flush=True)
        for it in range(STEPS):
            b = torch.randint(0, n, (1024,))
            opt.zero_grad(set_to_none=True)
            # Both arms are scored by the SAME scaled criterion, so the printed loss is comparable.
            loss = (((net(Zt[b]) - tgt[b]) / (1.0 if row_scaled else scale)) ** 2).mean()
            loss.backward()
            opt.step()
            sch.step()
            if (it + 1) % max(1, STEPS // 6) == 0:
                print('  step %6d   scaled MSE %.4e   [%.0fs]'
                      % (it + 1, float(loss), time.time() - t0), flush=True)
        if row_scaled:
            # Fold the per-row scale into the final Linear, making the net directly plantable.
            with torch.no_grad():
                last = [m for m in net.net.net if isinstance(m, torch.nn.Linear)][-1]
                last.weight.mul_(scale[0, :, 0][:, None])
                last.bias.mul_(scale[0, :, 0])
        with torch.no_grad():
            pred = torch.cat([net(Zt[i:i + 8192]) for i in range(0, n, 8192)])
        res = (pred - Wt)[:, :, 0].numpy()
        out = {}
        print('  fit quality, 1 - R^2 per row (0 = perfect, >1 = worse than the mean):')
        for i, r in enumerate(rows):
            v = W[:, i].var()
            out[r] = float(res[:, i].var() / v) if v > 0 else float('nan')
            print('    %-4s 1-R^2 %.4e   residual std %.4e   target std %.4e'
                  % (r, out[r], res[:, i].std(), W[:, i].std()))
        return net, out

    SKIP_A = bool(int(os.environ.get('CL_CAP_SKIP_A', 0)))
    r2_a = {r: float('nan') for r in rows}
    ann_a = None
    if not SKIP_A:
        ann_a, r2_a = fit(False, 'ARM A: as parameterised today')
    ann_b, r2_b = fit(True, 'ARM B: per-row output scaling, folded back in')
    better = sum(1 for r in rows if r2_b[r] < r2_a[r])   # nan compares False
    print('\nARM B beats ARM A on %d of %d rows' % (better, len(rows)))
    mean_a = float(np.nanmean([r2_a[r] for r in rows]))
    mean_b = float(np.nanmean([r2_b[r] for r in rows]))
    ann, r2 = (ann_b, r2_b) if (SKIP_A or mean_b < mean_a) else (ann_a, r2_a)
    print('planting the better arm: %s' % ('B' if r2 is r2_b else 'A'))

    # ---- plant it and score it ----------------------------------------------------------------
    print('\nplanting the regressed ANN into the model and scoring it', flush=True)
    ann0.load_state_dict(ann.state_dict())
    torch.save(ann.state_dict(), os.path.join(HERE, 'runs', 'cl_capability_planted_ann.pt'))
    res_out = dict(regression_1_minus_r2=r2, arm_a=r2_a, arm_b=r2_b, latent_gauge=std_aug.ravel().tolist(),
                   n_samples=int(n), steps=STEPS)
    # TWO initialisations, because the encoder cannot supply what it was never built for: its
    # augmented rows are the W^a block, and in this planted test the latents have a FIXED meaning
    # (the true absorber states in the gauge above) that the encoder has no reason to reproduce.
    # Scoring only from the encoder would report an initialisation failure as a representation
    # failure. The true-x0 arm removes the encoder from the question entirely.
    k0 = max(fs.na, fs.nb)
    rv = lambda a: np.asarray(a).ravel()                                    # noqa: E731
    per, agg = [], []
    for (name, sd, row), fname in zip(fs.simulator.val_records, vl_files):
        _, a = closed_loop_free_run_rms(fs, sd, bank, row)
        _, aw, _ = closed_loop_window_rms(fs, sd, bank, row, cfg.nf)
        # true-x0 free run: same rollout, x0 read from the truth in the model's own gauge
        _, _, x_log_v, x_aug_v = load_mat_aug(fname, cfg)
        x0 = np.concatenate([(np.asarray(x_log_v[k0], dtype=np.float64)[None] - x_mean) / std_x,
                             np.asarray(x_aug_v[k0], dtype=np.float64)[None] / std_aug], axis=1)
        un = ((sd.u - rv(fs.norm.u0)) / rv(fs.norm.ustd)).astype(np.float32)
        yn = ((sd.y - rv(fs.norm.y0)) / rv(fs.norm.ystd)).astype(np.float32)
        with torch.no_grad():
            y_pred, _, _ = closed_loop_rollout(
                fs.hfn, fs.hfn.output_only,
                torch.as_tensor(np.ascontiguousarray(un[None, k0:])),
                torch.as_tensor(np.ascontiguousarray(yn[None, k0:])),
                torch.as_tensor(x0, dtype=torch.float32), bank,
                torch.tensor([int(row)], dtype=torch.long))
        e = y_pred[0].numpy() * rv(fs.norm.ystd) + rv(fs.norm.y0) - np.asarray(sd.y)[k0:]
        a_true = float(np.sqrt(np.mean(np.mean(e ** 2, axis=0))))
        per.append((name, a, aw, a_true))
        agg.append((a, aw, a_true))
        print('  %-24s free run %.6e m (encoder-init)   %.6e m (TRUE x0)   window %.6e m'
              % (name, a, a_true, aw), flush=True)
    free = float(np.sqrt(np.mean([a ** 2 for a, _, _ in agg])))
    win = float(np.sqrt(np.mean([w ** 2 for _, w, _ in agg])))
    free_true = float(np.sqrt(np.mean([t ** 2 for _, _, t in agg])))
    res_out['planted'] = dict(free_run=free, free_run_true_x0=free_true, window=win,
                              per_record={n_: [a, w, t] for n_, a, w, t in per})
    print('\n' + '=' * 100)
    print('PLANTED MODEL (encoder-init, the same scoring path as training)')
    print('=' * 100)
    print('free run %.6e m   against untrained 2.1866e-06, trained 1.3934e-06, ORACLE 2.81e-08'
          % free)
    print('free run %.6e m from the TRUE x0, i.e. the encoder removed from the question' % free_true)
    print('window   %.6e m   against untrained 2.2210e-06, trained 1.5174e-06, FLOOR 7.02e-08'
          % win)
    print('\nverdict: %s' % (
        'the class CAN express it and the objective ranks it correctly -> the failure is SEARCH'
        if min(free, free_true) < 3 * 2.81e-08 else
        'one-step correctness does NOT survive the rollout, or the objective does not prefer the '
        'true solution -> this outranks the optimiser questions'))
    # ---- WHERE DOES THE WINDOW PENALTY COME FROM? three latent initialisations ------------
    # The planted model is 3.3x better than the trained one on the FREE RUN and only ~20% better
    # on the training WINDOW, which is the quantity the loss minimises. The free run pays a bad
    # initial state once in 48000 samples; the window pays it at every one of 1666 window starts.
    # These three arms differ ONLY in rows 6-7 of x0, so whatever separates them is the latent
    # initialisation and nothing else.
    #   1 W^a as it runs today: a random linear map of the history, kaiming_uniform_(a=0),
    #     U(+/-0.333) on 54 inputs, so the latent starts at an arbitrary O(1) value.
    #   2 W^a = 0: the convention the repo's OTHER two encoders use (HybridGantryEncoder and
    #     LinearInitEncoderWrapper both take their augmented states from a zero-init ANN).
    #   3 the TRUE absorber state, in the gauge the planted ANN was fitted in: the ceiling a
    #     perfect latent observer would reach.
    print('\n' + '=' * 100)
    print('LATENT INITIALISATION: three arms, identical except rows 6-7 of x0')
    print('=' * 100)
    enc = fs.encoder
    wa_y0 = enc.Wa_psi_y.detach().clone()
    wa_u0 = enc.Wa_psi_u.detach().clone()
    arms = {}
    for arm in ('W^a random (today)', 'W^a = 0', 'TRUE latent x0'):
        with torch.no_grad():
            enc.Wa_psi_y.copy_(torch.zeros_like(wa_y0) if arm != 'W^a random (today)' else wa_y0)
            enc.Wa_psi_u.copy_(torch.zeros_like(wa_u0) if arm != 'W^a random (today)' else wa_u0)
        sse, npts = 0.0, 0
        for (name, sd, ctrl_row), fname in zip(fs.simulator.val_records, vl_files):
            _, _, xl_v, xa_v = load_mat_aug(fname, cfg)
            st = window_starts(len(sd.u), fs.na, fs.nb, cfg.nf, na_r, nb_r, cfg.nf)
            uh, yh, uf, yf = make_window_tensors(fs, sd, cfg.nf, st)
            with torch.no_grad():
                x0 = fs.encoder(uh, yh)
                if arm == 'TRUE latent x0':
                    lat = np.asarray(xa_v, dtype=np.float64)[st] / std_aug
                    x0 = torch.cat([x0[:, :6], torch.as_tensor(lat, dtype=x0.dtype)], dim=1)
                y_pred, _, _ = closed_loop_rollout(
                    fs.hfn, fs.hfn.output_only, uf, yf, x0, bank,
                    torch.full((len(st),), int(ctrl_row), dtype=torch.long))
                e = (y_pred - yf).numpy() * rv(fs.norm.ystd)
            sse += float(np.sum(e ** 2))
            npts += e.shape[0] * e.shape[1] * e.shape[2]
        arms[arm] = float(np.sqrt(sse / npts))
        print('  %-22s window RMS %.6e m' % (arm, arms[arm]), flush=True)
    with torch.no_grad():
        enc.Wa_psi_y.copy_(wa_y0)
        enc.Wa_psi_u.copy_(wa_u0)
    res_out['latent_arms'] = arms
    base_arm = arms['W^a random (today)']
    print('\nagainst: trained model window 1.5174e-06, per-window FLOOR 7.02e-08, '
          'and this same planted model on the FREE run %.4e' % free)
    print('the latent initialisation is worth %.2fx of window error (today / true)'
          % (base_arm / max(arms['TRUE latent x0'], 1e-30)))
    print('zeroing W^a alone recovers %.1f %% of that gap'
          % (100.0 * (base_arm - arms['W^a = 0'])
             / max(base_arm - arms['TRUE latent x0'], 1e-30)))

    json.dump(res_out, open(OUT, 'w'), indent=2)
    print('\nwrote %s   [%.0fs]' % (OUT, time.time() - t0))


if __name__ == '__main__':
    main()
