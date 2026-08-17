"""STEP 2 GATES for the closed-loop path. Read-only apart from its own log output.

Six gates, in increasing order of what they can catch. Every one of them can fail independently,
and the order matters: the last gate is meaningless if the earlier ones have not passed.

  G1  no feedthrough        y must not depend on u. `model.py:140` wires u into out_phys, so the
                            wiring permits D != 0 even though gantry_ss.Dd is exactly zero. The
                            entire closed-loop step order rests on this.
  G2  affine output map     x @ C.T + b reproduces hfn's y on states other than the basis used
                            to identify it, so the rollout can cost ONE hfn call per step.
  G3  output frozen         no trainable parameter on the output path, or capturing (C, b) as
                            constants would silently cut the gradient.
  G4  UNITS                 a known physical residual in, the expected physical force out.
                            THIS IS THE GATE G6 CANNOT PROVIDE: with the ANN off the model
                            reproduces the record, so e = 0 and u_fb = 0 identically, and any
                            scale error on Cfb is multiplied by zero. G6 passes regardless.
  G5  open-loop replay      reproduces D-139's measured open-loop numbers. Establishes that this
                            script's model, x0 and normalisation match the ones D-139 used,
                            BEFORE the loop is switched on. Without this a G6 failure is
                            ambiguous between the loop and the setup.
  G6  closed-loop replay    the new shared rollout reproduces D-139's measured CLOSED-loop
                            numbers. This is the regression test that pins the loop.

Why the target is D-139's numbers and not a numerics floor: the records were generated at 20 kHz
and this path runs at 4 kHz (D-141), so the model cannot reproduce `y` to the 3.9e-08 m floor and
was never expected to. D-139 already measured what the 4 kHz baseline replay actually achieves,
open loop and closed loop, on V1 and V2. Reproducing those is a sharper test than a bound.

Usage
-----
  python -u cl_gate_replay.py
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
import closed_loop as CL                                                  # noqa: E402

import cl_plant as PLANT                                                  # noqa: E402
from cl_controller import ControllerBank, rollout, open_loop_rollout, check_units  # noqa: E402

CH = ['X1', 'X2', 'Y']
RECORDS = [('V1_standstill_Yp10', 0.10), ('V2_aprbs_Ylow', -0.22)]

# D-139, measured: baseline replay rms error [m] per channel at 4 kHz, float32, from sample 0.
D139 = {
    'V1_standstill_Yp10': {'open':   np.array([2.992e-08, 2.995e-08, 2.260e-06]),
                           'closed': np.array([4.149e-07, 3.768e-07, 3.742e-06])},
    'V2_aprbs_Ylow':      {'open':   np.array([2.154e-04, 2.416e-04, 4.597e-05]),
                           'closed': np.array([3.356e-07, 4.094e-07, 3.769e-06])},
}
# HEURISTIC: D-139's figures are quoted to 4 significant digits and were produced by a different
# script with its own x0 and rollout, so an exact match is not the claim. A factor of 2 catches a
# structurally wrong loop (the failure modes here are orders of magnitude, not percentages) while
# tolerating implementation differences at that level.
TOL_FACTOR = 2.0

results = {}
t0 = time.time()


def rms(a):
    return np.sqrt(np.mean(np.asarray(a) ** 2, axis=0))


def line(label, v, fmt='%.4e'):
    return '    %-34s [' % label + ' '.join(fmt % x for x in v) + ']'


print('=' * 92)
print('STEP 2 GATES: closed-loop path')
print('=' * 92)

cfg = dataclasses.replace(CFG, seed=0)
print('rate %d Hz, ts %.6e s, dtype %s' % (cfg.fs_new_hz, cfg.ts_new, cfg.dtype_np.__name__))
print('\nbuilding the interconnect (loads the train set, takes minutes) ...', flush=True)
fs, norm, K0, na, nb, na_r, nb_r = dm.build_pipeline(cfg=cfg, verbose=True)
nx = cfg.nx_phys + cfg.nx_ann
print('built  nx = %d (%d phys + %d ann), K0 = %d   [%.0fs]'
      % (nx, cfg.nx_phys, cfg.nx_ann, K0, time.time() - t0), flush=True)

restore_ann = PLANT.zero_the_ann(fs)
print('ANN forced to exactly zero: the augmented model IS the baseline')

# ---------------------------------------------------------------- G1
print('\n' + '-' * 92)
print('G1  NO FEEDTHROUGH   (model.py:140 wires u into out_phys, so this is not free)')
ft = PLANT.check_no_feedthrough(fs.hfn, nx, cfg.nu, dtype=cfg.dtype_pt)
print('    max |y(x, u=0) - y(x, u=1e3*randn)|  =  %.3e' % ft)
g1 = ft == 0.0
print('    %s   (must be EXACTLY zero; anything else is an algebraic loop)'
      % ('PASS' if g1 else 'FAIL'))
results['G1'] = g1

# ---------------------------------------------------------------- G2
print('\n' + '-' * 92)
print('G2  AFFINE OUTPUT MAP   (lets the rollout cost ONE hfn call per step)')
C, b = PLANT.identify_output_map(fs.hfn, nx, cfg.nu, dtype=cfg.dtype_pt)
abs_err, rel_err = PLANT.check_affine(fs.hfn, C, b, nx, cfg.nu, dtype=cfg.dtype_pt)
print('    C %s   b %s' % (tuple(C.shape), tuple(b.shape)))
print('    max abs err on random states  %.3e   relative  %.3e' % (abs_err, rel_err))
g2 = rel_err < 1e-6
print('    %s' % ('PASS' if g2 else 'FAIL'))
results['G2'] = g2

# ---------------------------------------------------------------- G3
print('\n' + '-' * 92)
print('G3  OUTPUT PATH FROZEN   ((C, b) are captured as constants)')
out_n, counts = PLANT.check_output_frozen(fs)
print('    trainable params on the output block: %d' % out_n)
print('    per-block trainable counts: %s' % counts)
g3 = out_n == 0
print('    %s   (nonzero would mean freezing C, b cuts the gradient)'
      % ('PASS' if g3 else 'FAIL'))
results['G3'] = g3

# ---------------------------------------------------------------- G4
print('\n' + '-' * 92)
print('G4  UNITS   (G6 CANNOT catch this: with the ANN off, e = 0 so u_fb = 0 identically)')
bank = ControllerBank([r for r, _ in RECORDS], cfg.ts_new, dtype=cfg.dtype_pt,
                      ystd=norm.ystd, std_u=norm.std_u)
print('    n_FB = %d, distinct Y_op %s' % (bank.nc, bank.y_ops_unique))
print('    norm.ystd  [%.4e %.4e %.4e] m' % tuple(np.asarray(norm.ystd).ravel()))
print('    norm.std_u [%.4e %.4e %.4e] N' % tuple(np.asarray(norm.std_u).ravel()))
u_norm, expect, rel = check_units(bank, rec_ix=0, e_phys=1e-4)
print(line('e = 1e-4 m -> u_fb normalised', u_norm))
print(line('expected physical Dc @ e [N]', expect))
print(line('round-trip relative error', rel, '%.3e'))
g4 = float(np.max(rel)) < 1e-6
print('    %s' % ('PASS' if g4 else 'FAIL'))
results['G4'] = g4

# ---------------------------------------------------------------- G5 and G6
step_fn, out_fn = PLANT.make_fns(fs, C, b)
print('\n' + '-' * 92)
print('G5/G6  REPLAY against D-139 (rms error [m] from sample 0, per channel)')

for rec_ix, (name, Y_op) in enumerate(RECORDS):
    sd = load_traj(name + '.mat', cfg)
    un = PLANT.normalise_u(sd.u, norm, cfg.dtype_np)
    y_raw = sd.y.astype(np.float64)
    N = len(un)

    x0_phys = CL.x0_for('baseline', Y_op)
    x0n = np.zeros(nx, dtype=cfg.dtype_np)
    x0n[:cfg.nx_phys] = ((x0_phys - np.asarray(norm.x_mean).ravel())
                         / np.asarray(norm.std_x).ravel()).astype(cfg.dtype_np)
    x0 = torch.from_numpy(x0n[None, :])
    u_t = torch.from_numpy(np.ascontiguousarray(un[None, :, :]))
    y_t = torch.from_numpy(np.ascontiguousarray(
        PLANT.normalise_y(y_raw, norm, cfg.dtype_np)[None, :, :]))
    ix = torch.tensor([rec_ix], dtype=torch.long)

    print('\n  %s   Y_op %+.2f   N = %d (%.2f s)' % (name, Y_op, N, N * cfg.ts_new), flush=True)
    with torch.no_grad():
        y_ol, _ = open_loop_rollout(step_fn, out_fn, u_t, x0)
        y_cl, _, xc_end = rollout(step_fn, out_fn, u_t, y_t, x0, bank, bank.gather(ix))

    e_ol = rms(PLANT.denormalise_y(y_ol[0].numpy(), norm) - y_raw)
    e_cl = rms(PLANT.denormalise_y(y_cl[0].numpy(), norm) - y_raw)
    ref_ol, ref_cl = D139[name]['open'], D139[name]['closed']
    print(line('open   loop rms [m]', e_ol))
    print(line('  D-139 reference', ref_ol))
    print(line('  ratio to reference', e_ol / ref_ol, '%9.4f'))
    print(line('closed loop rms [m]', e_cl))
    print(line('  D-139 reference', ref_cl))
    print(line('  ratio to reference', e_cl / ref_cl, '%9.4f'))
    print(line('closed / open', e_cl / e_ol, '%9.4f'))
    print('    xc final norm %.4e  (0 would mean the controller never moved)'
          % float(xc_end.norm()))

    ok_ol = np.all(e_ol / ref_ol < TOL_FACTOR) and np.all(e_ol / ref_ol > 1 / TOL_FACTOR)
    ok_cl = np.all(e_cl / ref_cl < TOL_FACTOR) and np.all(e_cl / ref_cl > 1 / TOL_FACTOR)
    print('    G5 open %s     G6 closed %s   (tol factor %.1f both ways)'
          % ('PASS' if ok_ol else 'FAIL', 'PASS' if ok_cl else 'FAIL', TOL_FACTOR))
    results['G5 ' + name] = ok_ol
    results['G6 ' + name] = ok_cl

restore_ann()

print('\n' + '=' * 92)
for k in sorted(results):
    print('%-28s %s' % (k, 'PASS' if results[k] else 'FAIL'))
allok = all(results.values())
print('=' * 92)
print('STEP 2 %s   [%.0fs]' % ('PASSED' if allok else 'HAS FAILURES', time.time() - t0))
sys.exit(0 if allok else 1)
