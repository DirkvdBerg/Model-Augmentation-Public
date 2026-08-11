"""Is the trained model limited by its ENCODER or by its DYNAMICS? No training, minutes to run.

The 1-hour BLA-dyn run reached 9.72e-02 m on V2. The BLA it started from reaches 9.89e-02 m from a
zero initial state and 2.54e-02 m from a least-squares-optimal one. So there is a 3.8x gap that is
purely about the initial state, and this script decides whether the trained model is sitting in it.

  A  free run from the ENCODER's state           the reported sim-RMS
  B  free run from an OPTIMISED initial state    x0 refined by gradient descent, model frozen
  C  eigenvalues of df/dx along the trajectory   did the poles drift away from the BLA's?

Reading: B near 2.5e-02 means the encoder is the binding constraint. B near A means the initial
state is not the issue and the dynamics themselves moved. C says whether the near-unit poles the
BLA supplied survived training, which is the quantity the whole free-run metric hangs on.
"""
__project_origin__ = "added"

import os
import sys
import json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import load
import deepSI

MODEL = os.path.join(HERE, 'results', 'ann_blackbox_fs800_nf3700_s0_bladynz')
VAL, FS = 'V2_aprbs_Ylow', 800.0
N_OPT = 60                                        # Adam steps on x0 only


def rollout(fit_sys, x0, Un, grad=False):
    """Free run in NORMALISED coordinates. Returns normalised yhat."""
    x = x0 if grad else x0.detach()
    out = []
    for k in range(len(Un)):
        y, x = fit_sys.hfn(x, Un[k:k + 1])
        out.append(y)
    return torch.cat(out, dim=0)


fit_sys = deepSI.load_system(MODEL)
va = load(VAL, FS)
nd = fit_sys.norm.transform(va)
Un = torch.tensor(np.asarray(nd.u), dtype=torch.float32)
Yn = torch.tensor(np.asarray(nd.y), dtype=torch.float32)
ystd = torch.tensor(fit_sys.norm.ystd, dtype=torch.float32)
y0n = torch.tensor(fit_sys.norm.y0, dtype=torch.float32)


def rms_m(yhat_n, off=0):
    """Denormalise and score in metres against the recorded output, as sim-RMS does."""
    y = yhat_n.detach() * ystd + y0n
    ref = torch.tensor(va.y[off:off + len(y)], dtype=torch.float32)
    return float(torch.sqrt(torch.mean((y - ref) ** 2)))


# --- A: the encoder's own initial state ---------------------------------------------------
na, nb = fit_sys.na, fit_sys.nb
k0 = max(na, nb)
uh = Un[k0 - nb:k0].reshape(1, nb, -1)
yh = Yn[k0 - na:k0].reshape(1, na, -1)
with torch.no_grad():
    x_enc = fit_sys.encoder(uh, yh)
y_enc = rollout(fit_sys, x_enc, Un[k0:])
rms_enc = rms_m(y_enc, off=k0)
print(f'A  encoder initial state        {rms_enc:.4e} m')

# --- B: initial state optimised, dynamics frozen -------------------------------------------
x0 = x_enc.clone().requires_grad_(True)
for mod in (fit_sys.hfn, fit_sys.encoder):      # fit_sys.parameters is a LIST attribute, not a method
    for p in mod.parameters():
        p.requires_grad_(False)
opt = torch.optim.Adam([x0], lr=1e-2)
for it in range(N_OPT):
    opt.zero_grad()
    loss = torch.mean((rollout(fit_sys, x0, Un[k0:], grad=True) - Yn[k0:]) ** 2)
    loss.backward()
    opt.step()
    if it % 15 == 0 or it == N_OPT - 1:
        print(f'   x0 opt it {it:3d}  normalised MSE {loss.item():.5e}')
with torch.no_grad():
    rms_opt = rms_m(rollout(fit_sys, x0, Un[k0:]), off=k0)
print(f'B  optimised initial state      {rms_opt:.4e} m      ratio A/B = {rms_enc/rms_opt:.2f}')

# --- C: eigenvalues of df/dx along the trajectory ------------------------------------------
with torch.no_grad():
    x = x_enc.clone()
    states = []
    for k in range(0, 4000):
        _, x = fit_sys.hfn(x, Un[k0 + k:k0 + k + 1])
        if k % 800 == 0:
            states.append(x.clone())
rows = []
for i, xs in enumerate(states):
    J = torch.autograd.functional.jacobian(
        lambda z: fit_sys.hfn.fn(z, Un[k0 + i * 800:k0 + i * 800 + 1]), xs)
    J = J.reshape(fit_sys.nx, fit_sys.nx).numpy()
    ev = np.linalg.eigvals(J)
    rows.append(dict(k=i * 800, max_abs=float(np.max(np.abs(ev))),
                     closest_to_1=float(np.min(np.abs(ev - 1.0))),
                     n_above_099=int(np.sum(np.abs(ev) > 0.99))))
    print(f'C  k={i*800:5d}  max|eig|={rows[-1]["max_abs"]:.6f}  '
          f'|eig-1|min={rows[-1]["closest_to_1"]:.3e}  n(|eig|>0.99)={rows[-1]["n_above_099"]}')

print(f'\n   BLA at init had |eig-1| = 1.755e-05 and the bar needs 8.4e-08')
json.dump(dict(model=os.path.basename(MODEL), val=VAL, fs=FS,
               rms_encoder_init=rms_enc, rms_optimised_init=rms_opt, jacobian=rows),
          open(os.path.join(HERE, 'results', 'encoder_vs_dynamics.json'), 'w'), indent=1)
