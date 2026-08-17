"""Step-1 smoke test for cl_controller: shapes, per-record gather, units, loop sanity.

No model, no deepSI. A trivial linear plant stands in for the gantry so the rollout wiring can be
checked on its own. Run: python cl_smoke.py
"""
__project_origin__ = "added"

import numpy as np
import torch

from cl_controller import ControllerBank, RECORD_Y_OP, rollout, open_loop_rollout, check_units

TS = 1.0 / 4000.0          # D-141: the closed-loop path runs at 4 kHz
NAMES = ['T1_standstill_Ym30.mat', 'T3_standstill_Y000.mat', 'T5_standstill_Yp30.mat',
         'T10_aprbs_60.mat', 'V1_standstill_Yp10.mat', 'V2_aprbs_Ylow.mat']
YSTD = np.array([1.2e-3, 1.3e-3, 9.0e-2])          # stand-ins, real ones come from norm
STDU = np.array([4.5e+1, 4.5e+1, 3.6e+1])

print('=' * 78)
print('STEP 1 SMOKE: cl_controller')
print('=' * 78)

bank = ControllerBank(NAMES, TS, dtype=torch.float64, ystd=YSTD, std_u=STDU)
print('records            %s' % bank.record_names)
print('distinct Y_op      %s' % bank.y_ops_unique)
print('rec_ix -> ctrl row %s' % bank.rec_to_ctrl.tolist())
print('A %s  B %s  C %s  D %s   nc = %d'
      % (tuple(bank.A.shape), tuple(bank.B.shape), tuple(bank.C.shape), tuple(bank.D.shape),
         bank.nc))
assert bank.nc == 9, 'n_FB must be 9 (D-140)'
assert bank.A.shape[0] == len(set(RECORD_Y_OP[n] for n in bank.record_names))
# T3 and T10 are both Y_op = 0.00 and must share a controller row
assert bank.rec_to_ctrl[1] == bank.rec_to_ctrl[3], 'records at equal Y_op must share a row'
print('OK  distinct-Y_op sharing: T3 and T10 both map to row %d' % bank.rec_to_ctrl[1])

# --- the gather actually differentiates records -------------------------------
ix = torch.tensor([0, 1, 2], dtype=torch.long)      # Y_op -0.30, 0.00, +0.30
A, B, C, D = bank.gather(ix)
print('\ngathered shapes    A %s  D %s' % (tuple(A.shape), tuple(D.shape)))
dd = torch.stack([torch.diagonal(D[i]) for i in range(3)])
print('D diag per record  (rows = Y_op -0.30, 0.00, +0.30)')
for i, Y in enumerate([-0.30, 0.00, 0.30]):
    print('   Y_op %+5.2f   [%.4e %.4e %.4e]' % (Y, *dd[i].numpy()))
assert not torch.allclose(dd[0], dd[2]), 'X1/X2 gains must differ across Y_op (D-140)'
print('OK  gather returns genuinely different controllers per record')

# --- units gate ---------------------------------------------------------------
print('\nUNITS GATE (the zero-ANN replay gate cannot catch this)')
u_norm, expect, rel = check_units(bank, rec_ix=1, e_phys=1e-4)
print('  e_phys 1e-4 m -> u_fb normalised [%.6e %.6e %.6e]' % tuple(u_norm))
print('  expected physical  Dc @ e       [%.6e %.6e %.6e] N' % tuple(expect))
print('  round-trip rel err              [%.3e %.3e %.3e]' % tuple(rel))
assert np.max(rel) < 1e-12, 'physical -> normalised -> physical round trip must be exact'
print('OK  units round-trip exact')

# --- rollout wiring on a trivial plant ----------------------------------------
# Double integrator per channel in normalised coordinates: the K = 0 case, so an open loop
# drifts and a closed loop must not. x = [pos(3), vel(3)].
torch.manual_seed(0)
nb, nf, ny, nu = 4, 200, 3, 3
G = 0.02


def step_fn(x, u):
    pos, vel = x[:, :3], x[:, 3:]
    return torch.cat([pos + TS * vel, vel + TS * G * u / TS], dim=1)


def out_fn(x):
    return x[:, :3]


u_data = torch.zeros(nb, nf, nu, dtype=torch.float64)
y_data = torch.zeros(nb, nf, ny, dtype=torch.float64)
x0 = torch.zeros(nb, 6, dtype=torch.float64)
x0[:, 3:] = 1e-3                                    # a wrong initial velocity, the D-139 artefact
ix = torch.tensor([0, 1, 2, 3], dtype=torch.long)

y_ol, _ = open_loop_rollout(step_fn, out_fn, u_data, x0)
y_cl, xf, xcf = rollout(step_fn, out_fn, u_data, y_data, x0, bank, bank.gather(ix))
print('\nROLLOUT on a double integrator, x0 velocity error 1e-3, target y = 0')
print('  y_pred shape        %s   xc_final shape %s' % (tuple(y_cl.shape), tuple(xcf.shape)))
assert y_cl.shape == (nb, nf, ny) and xcf.shape == (nb, bank.nc)
print('  open   loop |y| end [%.4e %.4e %.4e]' % tuple(y_ol[0, -1].abs().numpy()))
print('  closed loop |y| end [%.4e %.4e %.4e]' % tuple(y_cl[0, -1].abs().numpy()))
ratio = (y_cl[0, -1].abs() / y_ol[0, -1].abs().clamp_min(1e-30)).numpy()
print('  closed / open       [%.4e %.4e %.4e]' % tuple(ratio))
assert np.all(ratio < 1.0), 'the loop must suppress the drift it is there to suppress'
print('OK  the loop is closed and acts in the right direction')

# --- xc = 0 is the window-start default, and xc0 is honoured -------------------
y_a, _, xc_a = rollout(step_fn, out_fn, u_data[:, :50], y_data[:, :50], x0, bank, bank.gather(ix))
y_b, _, _ = rollout(step_fn, out_fn, u_data[:, :50], y_data[:, :50], x0, bank, bank.gather(ix),
                    xc0=xc_a)
print('\n  xc0 default is zero, and a supplied xc0 changes the result: %s'
      % (not torch.allclose(y_a, y_b)))
assert not torch.allclose(y_a, y_b)

# --- gradients flow through the controller ------------------------------------
x0g = x0.clone().requires_grad_(True)
y_g, _, _ = rollout(step_fn, out_fn, u_data[:, :20], y_data[:, :20], x0g, bank, bank.gather(ix))
y_g.pow(2).mean().backward()
print('  gradient reaches x0 through the loop: %s  (norm %.4e)'
      % (x0g.grad is not None and torch.isfinite(x0g.grad).all().item(), x0g.grad.norm()))
assert x0g.grad is not None and torch.isfinite(x0g.grad).all()

print('\n' + '=' * 78)
print('STEP 1 SMOKE PASSED')
print('=' * 78)
