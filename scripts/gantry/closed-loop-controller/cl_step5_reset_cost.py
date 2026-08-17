"""STEP 5: what does `xc = 0` at every training-window start actually cost?

This is the ONLY approximation in the closed-loop training path. Everything else in this build is
either exactly true or exactly false and is covered by a gate. So it gets measured, not argued.

THE APPROXIMATION
-----------------
In the residual form the controller filters `y_data - y_model`. Before a training window opens the
model was not running, so that residual does not exist and `xc = 0` is a definition rather than an
estimate. Kessels' Remark 5.4 reconstruction does not apply: it is for the lumped-`r` form where
the controller filters `y_model` and `xc` at a window start is a large physical unknown.

What `xc = 0` DOES cost is memory. `Cnorm` has a pole at `s = 0`, so the discrete controller has a
pole at `z = 1` and never forgets DC. Run continuously, `xc` converges to whatever constant force
nulls the DC part of the model error. Reset per window throws that away, so each training window
sees a loop with less integral action than the validation free run has. That is a train/validation
asymmetry of the same species as the one that invalidated variant B, though far milder, and its
size is an empirical question.

TWO MEASUREMENTS
----------------
M1, pure isolation. A naive continuous-against-windowed comparison is confounded: `x` is
re-anchored by the encoder at each boundary AND `xc` resets, so the difference mixes two effects.
M1 removes the first by taking the residual sequence `e[k]` from ONE continuous rollout and then
driving the controller through that SAME fixed `e` twice, with `xc` carried and with `xc` reset
every `nf`. The difference is purely the reset.

M2, training geometry. The decision-relevant number. Both arms are windowed and encoder-initialised
exactly as training will be; only `xc0` differs, zero against the value the continuous run had at
that window start. This includes the feedback of `xc` into the state trajectory, which M1 excludes
by construction.

THE YARDSTICK
-------------
Ratio to `rms(u_fb)` is the obvious one, but the one that decides is the ratio to the MODEL ERROR
the loss is actually fitting. If the reset perturbs the window trajectory by far less than the
model error the loss is trying to reduce, it cannot change what the optimiser learns.

PRE-REGISTERED PREDICTION (D-090), written before running
---------------------------------------------------------
`nf = 400` at 4 kHz is 100 ms and the loop bandwidth is 100 Hz, so a window spans about 10 loop
periods. Prediction: most of the integral force is rebuilt within the first 10-20 % of a window,
and the cost lands below the 1 % trigger. If it lands above, the follow-up question is whether the
difference is DC-dominated, since that is the case where a warm-up lead-in buys a lot cheaply.

The ANN is forced to zero, so the model IS the baseline and the residual is the genuine MSD
mismatch, i.e. the error regime the ANN will be working in.

Usage
-----
  python -u cl_step5_reset_cost.py
"""
__project_origin__ = "added"

import dataclasses
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

import demo_common as dm                                                  # noqa: E402
from demo_common import CFG                                               # noqa: E402
from gantry_dynamic.data import load_traj                                 # noqa: E402

import cl_plant as PLANT                                                  # noqa: E402
import cl_validation as CV                                                # noqa: E402
from cl_controller import ControllerBank, rollout, y_op_for               # noqa: E402

CH = ['X1', 'X2', 'Y']
RECORDS = ['T1_standstill_Ym30', 'T10_aprbs_60', 'V1_standstill_Yp10', 'V2_aprbs_Ylow']
N_WINDOWS = 40          # window starts sampled per record for M2
TRIGGER = 0.01          # HEURISTIC (this session): the 1 % of rms(u_fb) trigger for warm-up
t0 = time.time()


def line(label, v, fmt='%.4e'):
    return '    %-38s [' % label + ' '.join(fmt % x for x in v) + ']'


def ctrl_replay(A, B, C, D, e_phys, reset_every=None):
    """Drive the controller through a FIXED physical residual sequence. Returns u_fb [N], xc traj."""
    N = len(e_phys)
    nc = A.shape[0]
    xc = np.zeros(nc)
    u = np.empty((N, C.shape[0]))
    xs = np.empty((N, nc))
    for k in range(N):
        if reset_every is not None and k % reset_every == 0:
            xc = np.zeros(nc)
        xs[k] = xc
        u[k] = C @ xc + D @ e_phys[k]
        xc = A @ xc + B @ e_phys[k]
    return u, xs


print('=' * 100)
print('STEP 5: cost of xc = 0 at every training-window start')
print('=' * 100)

cfg = dataclasses.replace(CFG, seed=0)
nf, k0 = cfg.nf, None
print('rate %d Hz, nf = %d (%.3f s), loop bandwidth 100 Hz -> %.1f loop periods per window'
      % (cfg.fs_new_hz, nf, nf * cfg.ts_new, nf * cfg.ts_new * 100))
print('\nbuilding the interconnect ...', flush=True)
fs, norm, K0, na, nb, na_r, nb_r = dm.build_pipeline(cfg=cfg, verbose=True)
k0 = K0
nx = cfg.nx_phys + cfg.nx_ann
C_out, b_out = PLANT.identify_output_map(fs.hfn, nx, cfg.nu, dtype=cfg.dtype_pt)
step_fn, out_fn = PLANT.make_fns(fs, C_out, b_out)
restore = PLANT.zero_the_ann(fs)
print('ANN forced to zero: the residual is the genuine MSD mismatch   [%.0fs]'
      % (time.time() - t0), flush=True)

bank = ControllerBank(RECORDS, cfg.ts_new, dtype=cfg.dtype_pt, ystd=norm.ystd, std_u=norm.std_u)
ystd = np.asarray(fs.norm.ystd).ravel()
summary = []

for i, name in enumerate(RECORDS):
    sd = load_traj(name + '.mat', cfg)
    un = ((sd.u - fs.norm.u0) / fs.norm.ustd).astype(cfg.dtype_np)
    yn = ((sd.y - fs.norm.y0) / fs.norm.ystd).astype(cfg.dtype_np)
    ctrl = bank.gather(torch.tensor([i], dtype=torch.long))
    row = bank.rec_to_ctrl[i].item()
    A = bank.A[row].numpy().astype(float); B = bank.B[row].numpy().astype(float)
    Cc = bank.C[row].numpy().astype(float); Dc = bank.D[row].numpy().astype(float)

    print('\n' + '=' * 100)
    print('%s   Y_op %+.2f' % (name, y_op_for(name)), flush=True)

    # ---- continuous reference rollout ------------------------------------
    x0 = CV.encoder_x0(fs.encoder, un, yn, k0, na, nb, na_r, nb_r, cfg.dtype_pt)
    u_t = torch.as_tensor(np.ascontiguousarray(un[None, k0:]), dtype=cfg.dtype_pt)
    y_t = torch.as_tensor(np.ascontiguousarray(yn[None, k0:]), dtype=cfg.dtype_pt)
    with torch.no_grad():
        y_ref, _, _ = rollout(step_fn, out_fn, u_t, y_t, x0, bank, ctrl)
    y_ref = y_ref[0].numpy()
    e_norm = yn[k0:] - y_ref                      # the residual the controller filters
    e_phys = e_norm * ystd                        # [m]
    model_err = np.sqrt(np.mean((y_ref - yn[k0:]) ** 2, axis=0)) * ystd     # [m], the loss target

    # ---- M1: pure controller-state isolation -----------------------------
    u_cont, xs_cont = ctrl_replay(A, B, Cc, Dc, e_phys, reset_every=None)
    u_res, _ = ctrl_replay(A, B, Cc, Dc, e_phys, reset_every=nf)
    d = u_res - u_cont
    rms_u = np.sqrt(np.mean(u_cont ** 2, axis=0))
    rms_d = np.sqrt(np.mean(d ** 2, axis=0))
    print('  M1  pure isolation (same e, xc carried against xc reset every nf)')
    print(line('rms u_fb continuous [N]', rms_u))
    print(line('rms difference [N]', rms_d))
    print(line('ratio to rms(u_fb)', rms_d / rms_u, '%10.5f'))

    # is the difference DC-dominated? mean per window against its rms
    nwin = (len(d) // nf) * nf
    dw = d[:nwin].reshape(-1, nf, 3)
    dc_frac = np.abs(dw.mean(axis=1)).mean(axis=0) / (np.sqrt((dw ** 2).mean(axis=1)).mean(axis=0)
                                                      + 1e-30)
    print(line('DC share of the difference', dc_frac, '%10.5f'))

    # recovery: samples after a reset until |d| falls under 10 % of its own window peak
    thr = np.abs(dw).max(axis=1, keepdims=True) * 0.10
    below = np.abs(dw) < thr
    rec = np.array([[np.argmax(below[w, :, j]) if below[w, :, j].any() else nf
                     for j in range(3)] for w in range(dw.shape[0])])
    print(line('median recovery [samples of %d]' % nf, np.median(rec, axis=0), '%10.1f'))
    print(line('  the same, as %% of the window', 100 * np.median(rec, axis=0) / nf, '%10.2f'))

    # ---- M2: training geometry -------------------------------------------
    starts = np.linspace(k0, len(un) - nf - 2, N_WINDOWS).astype(int)
    xc_ref = torch.as_tensor(xs_cont[starts - k0], dtype=cfg.dtype_pt)
    ux = torch.as_tensor(np.stack([un[s:s + nf] for s in starts]), dtype=cfg.dtype_pt)
    yx = torch.as_tensor(np.stack([yn[s:s + nf] for s in starts]), dtype=cfg.dtype_pt)
    x0w = torch.cat([CV.encoder_x0(fs.encoder, un, yn, int(s), na, nb, na_r, nb_r, cfg.dtype_pt)
                     for s in starts], dim=0)
    ctrl_w = bank.gather(torch.full((len(starts),), i, dtype=torch.long))
    with torch.no_grad():
        y_zero, _, _ = rollout(step_fn, out_fn, ux, yx, x0w, bank, ctrl_w)
        y_seed, _, _ = rollout(step_fn, out_fn, ux, yx, x0w, bank, ctrl_w, xc0=xc_ref)
    dz = (y_zero - y_seed).numpy() * ystd                       # [m]
    err_zero = (y_zero.numpy() - yx.numpy()) * ystd             # what the loss sees
    rms_dz = np.sqrt(np.mean(dz ** 2, axis=(0, 1)))
    rms_ez = np.sqrt(np.mean(err_zero ** 2, axis=(0, 1)))
    print('  M2  training geometry (%d windows, encoder-init both arms, only xc0 differs)'
          % len(starts))
    print(line('rms window model error [m]', rms_ez))
    print(line('rms effect of the reset [m]', rms_dz))
    print(line('ratio to the model error', rms_dz / rms_ez, '%10.5f'))
    worst = float(np.max(rms_dz / rms_ez))
    print('    worst channel ratio %.5f   trigger %.3f   -> %s'
          % (worst, TRIGGER, 'WARM-UP INDICATED' if worst > TRIGGER else 'xc = 0 is fine'))
    summary.append((name, rms_d / rms_u, rms_dz / rms_ez, np.median(rec, axis=0), dc_frac))

restore()

print('\n' + '=' * 100)
print('SUMMARY')
print('=' * 100)
print('%-22s %-26s %-26s %-18s' % ('record', 'M1 diff / rms(u_fb)', 'M2 diff / model error',
                                   'recovery [samples]'))
for name, m1, m2, rec, dcf in summary:
    print('%-22s %s   %s   %s'
          % (name, ' '.join('%7.5f' % v for v in m1), ' '.join('%7.5f' % v for v in m2),
             ' '.join('%5.0f' % v for v in rec)))
worst_all = max(float(np.max(m2)) for _, _, m2, _, _ in summary)
print('\nworst M2 ratio over all records and channels: %.5f   trigger %.3f' % (worst_all, TRIGGER))
print('VERDICT: %s' % ('WARM-UP INDICATED' if worst_all > TRIGGER
                       else 'xc = 0 per window is adequate; no warm-up needed'))
print('[%.0fs]' % (time.time() - t0))
