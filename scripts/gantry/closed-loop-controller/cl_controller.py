"""The controller as a subsystem, and the one closed-loop rollout both paths call.

D-140 settled the placement: `Cfb` is a SEPARATE subsystem stepped alongside the model, not a
block whose 9 states join the interconnect state vector. D-141 settled the implementation: 4 kHz,
`scripts/gantry/` first, per-record controller carried by an explicit index.

Nothing here imports the fit system, the gantry model, or deepSI. It takes a state-step callable
and returns trajectories, so it can be lifted into `model_augmentation/fit_systems/` later as a
move rather than a rewrite (D-141).

THE FORM
--------
Residual form, verified equivalent and cheaper than the lumped-`r` form:

    u_plant[k] = u_data[k] + Cfb * (y_data[k] - y_model[k])

Only `u_total` and `y` are needed, both of which the loader already returns; `r_sim` and `f_sim`
are not used by the training path at all. Because the controller filters the output RESIDUAL,
and the model was not running before a window opened, `xc = 0` at a window start is a definition
rather than an approximation, and Kessels' Remark 5.4 reconstruction is not needed. (In the
lumped-`r` form the controller filters `y_model`, `xc` is then a large unknown, and Remark 5.4 is
exactly the machinery required. That is the world Kessels is in, and it is why that remark exists
for him and not for us.)

STEP ORDER, and why it is forced
--------------------------------
The plant has no feedthrough (`D_d = 0`), so `y_model[k]` depends on the state only and is
computable before the input. The controller IS biproper: Tustin gives `Dc != 0`, measured at
`diag [2.844e+06 2.914e+06 1.509e+06]` N/m at `Y_op = 0`. So the order below is the only one that
closes the loop without an algebraic loop, and it is the same order `gtd_run_simulation.m` and
`closed_loop.py` use:

    y_model[k] = h(x[k])
    e[k]       = y_data[k] - y_model[k]                     PHYSICAL [m]
    u_fb[k]    = Cc xc[k] + Dc e[k]                         PHYSICAL [N]
    u_cl[k]    = u_data[k] + u_fb[k]                        NORMALISED, see UNITS
    x[k+1]     = step(x[k], u_cl[k])
    xc[k+1]    = Ac xc[k] + Bc e[k]

`ClosedLoopLossMixin` calls the model twice per step to work around this ordering. Here the model
is called once, which halves the FP-plus-ANN forward cost per step.

UNITS
-----
The model works in normalised coordinates; `Cfb` is physical, m -> N. Kessels does the same thing
in (5.13c)-(5.13d): the tracking error and the controller are physical and the scaling `S_u` is
applied to the controller output at the point it enters the model. So the residual is
denormalised, filtered, and the resulting force renormalised as an INCREMENT (the input mean
cancels for an increment, which is why `u_mean` does not appear).

WARNING, the units cannot be checked by the zero-ANN replay gate. With the ANN forced to zero the
model reproduces the record, so `e = 0` and `u_fb = 0` identically and ANY scale factor on `Cfb`
is multiplied by zero. The gate passes regardless. Units need a separate gate with a perturbed
initial state so that `e != 0`; see `check_units` below.

PER-RECORD Cfb
--------------
`generate_trajectory_data.m:43` calls `gtd_build_plant(rec.Y_op, cfg)` inside the per-record loop,
so the controller is rebuilt for every record and frozen within it. Nine distinct controllers
across the 22 records; `kappa` moves 1.5x on X1/X2 across the `Y_op` range. Verified in D-140:
rebuilding at each record's own `Y_op` reproduced the stored `u_fb` on every record, which a
single fixed controller could not have done.
"""
__project_origin__ = "added"

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from loss_variants import controller_ss                      # noqa: E402

# Y_op per record, Matlab-scripts/Augmentation/data/gtd_build_records.m:36-67.
# Keys are the loader's file stems (gantry_dynamic/data.py TRAIN_FILES etc.) without '.mat'.
RECORD_Y_OP = {
    'T1_standstill_Ym30': -0.30, 'T2_standstill_Ym15': -0.15, 'T3_standstill_Y000': 0.00,
    'T4_standstill_Yp15': 0.15,  'T5_standstill_Yp30': 0.30,
    'T6_ysweep_slow': 0.00, 'T7_ysweep_fast': 0.00, 'T8_ysweep_xmix': 0.00,
    'T9_aprbs_30': 0.00, 'T10_aprbs_60': 0.00, 'T11_aprbs_100': 0.00, 'T12_aprbs_yaw': 0.00,
    'T13_lissajous': 0.00, 'T14_lissajous_yaw': 0.00,
    'V1_standstill_Yp10': 0.10, 'V2_aprbs_Ylow': -0.22, 'V3_ysweep_Yp10': 0.10,
    'V4_lissajous_Ym10': -0.10,
    'E1_resonance_sweep': 0.00, 'E2_multisine_Yp22': 0.22, 'E3_aprbs_above': 0.00,
    'E4_multisine_off': 0.00,
}


def y_op_for(filename):
    """Y_op for a record file name, with or without the .mat suffix."""
    stem = os.path.basename(str(filename))
    if stem.endswith('.mat'):
        stem = stem[:-4]
    if stem not in RECORD_Y_OP:
        raise KeyError('no Y_op known for record %r; add it to RECORD_Y_OP from '
                       'gtd_build_records.m' % stem)
    return RECORD_Y_OP[stem]


class ControllerBank(torch.nn.Module):
    """The distinct `Cfb` instances for a set of records, gathered per batch element.

    Holds one stacked (A, B, C, D) per DISTINCT Y_op, not one per record, because several records
    share an operating point (T6-T14 are all at 0.00). Records map onto that small set.

    Buffers, not parameters: the controller is known exactly (D-140) and is never trained. They
    are registered as buffers so `.to(device)`, `.float()`/`.double()` and `state_dict()` carry
    them with the model.
    """

    def __init__(self, record_names, ts, dtype=torch.float32, ystd=None, std_u=None):
        """record_names: ordered list of record file names; index i is that record's rec_ix.

        ystd  (ny,)  normalised -> physical output scale [m], the pipeline's own norm.ystd
        std_u (nu,)  physical -> normalised input scale [N], the pipeline's own norm.std_u
        """
        super().__init__()
        self.record_names = [os.path.basename(str(n)).replace('.mat', '') for n in record_names]
        self.ts = float(ts)

        y_ops = [y_op_for(n) for n in self.record_names]
        uniq = sorted(set(y_ops))
        self.y_ops_unique = uniq
        # rec_ix -> row in the stacked controller tensors
        self.register_buffer('rec_to_ctrl',
                             torch.tensor([uniq.index(v) for v in y_ops], dtype=torch.long))

        As, Bs, Cs, Ds = [], [], [], []
        for Y_op in uniq:
            A, B, C, D = controller_ss(Y_op, self.ts)
            As.append(A); Bs.append(B); Cs.append(C); Ds.append(D)
        T = lambda M: torch.tensor(np.asarray(M, float), dtype=dtype)   # noqa: E731
        self.register_buffer('A', torch.stack([T(M) for M in As]))      # (K, nc, nc)
        self.register_buffer('B', torch.stack([T(M) for M in Bs]))      # (K, nc, ny)
        self.register_buffer('C', torch.stack([T(M) for M in Cs]))      # (K, nu, nc)
        self.register_buffer('D', torch.stack([T(M) for M in Ds]))      # (K, nu, ny)
        self.nc = self.A.shape[-1]

        if ystd is None or std_u is None:
            raise ValueError('ystd and std_u are required: the residual is denormalised before '
                             'the controller and the force renormalised after it, and getting '
                             'this wrong silently rescales the loop (see UNITS in the docstring)')
        self.register_buffer('ystd', T(np.asarray(ystd).ravel()))
        self.register_buffer('stdu', T(np.asarray(std_u).ravel()))

    def gather(self, rec_ix):
        """(A, B, C, D) for each element of a batch of record indices."""
        row = self.rec_to_ctrl[rec_ix]
        return self.A[row], self.B[row], self.C[row], self.D[row]

    def zero_state(self, batch, dtype=None, device=None):
        return torch.zeros(batch, self.nc,
                           dtype=self.A.dtype if dtype is None else dtype,
                           device=self.A.device if device is None else device)

    def step(self, xc, y_err_norm, ctrl):
        """One controller step, batched.

        xc          (batch, nc)      controller state
        y_err_norm  (batch, ny)      y_data - y_model, NORMALISED
        ctrl        gathered (A, B, C, D), each (batch, ...)

        Returns (u_fb_norm, xc_next), the feedback force as a NORMALISED input increment.
        """
        A, B, C, D = ctrl
        e = y_err_norm * self.ystd                                   # -> physical [m]
        u_fb = torch.einsum('bij,bj->bi', C, xc) + torch.einsum('bij,bj->bi', D, e)   # [N]
        xc_next = torch.einsum('bij,bj->bi', A, xc) + torch.einsum('bij,bj->bi', B, e)
        return u_fb / self.stdu, xc_next                             # -> normalised increment


def rollout(step_fn, out_fn, u_data, y_data, x0, bank, ctrl, xc0=None):
    """The single closed-loop rollout. Training windows and validation free runs both call this.

    step_fn(x, u) -> x_next      one model step, NORMALISED coordinates
    out_fn(x)     -> y           model output from state only (requires D_d = 0), NORMALISED
    u_data        (batch, nf, nu) recorded plant input, NORMALISED
    y_data        (batch, nf, ny) recorded output, NORMALISED
    x0            (batch, nx)     initial model state, e.g. from the encoder
    bank          ControllerBank
    ctrl          gathered (A, B, C, D) for this batch
    xc0           (batch, nc) or None for zero (the window-start default, see THE FORM)

    Returns (y_pred, x_final, xc_final) with y_pred (batch, nf, ny), NORMALISED.
    """
    nf = u_data.shape[1]
    x = x0
    xc = bank.zero_state(u_data.shape[0], dtype=u_data.dtype, device=u_data.device) \
        if xc0 is None else xc0
    ys = []
    for t in range(nf):
        y_model = out_fn(x)                                  # D_d = 0: state only
        ys.append(y_model)
        u_fb, xc = bank.step(xc, y_data[:, t] - y_model, ctrl)
        x = step_fn(x, u_data[:, t] + u_fb)
    return torch.stack(ys, dim=1), x, xc


def open_loop_rollout(step_fn, out_fn, u_data, x0):
    """The same rollout with the loop OPEN, for the side-by-side the gates need."""
    nf = u_data.shape[1]
    x = x0
    ys = []
    for t in range(nf):
        ys.append(out_fn(x))
        x = step_fn(x, u_data[:, t])
    return torch.stack(ys, dim=1), x


def check_units(bank, rec_ix=0, e_phys=1e-4):
    """Units gate. The zero-ANN replay gate CANNOT catch a scale error (see UNITS above).

    Drives the controller with a known PHYSICAL residual for one step from rest, so the output is
    exactly `Dc @ e_phys`, and checks the value that comes back after renormalisation.
    Returns (u_fb_norm, u_fb_phys_expected, rel_err) per channel.
    """
    ix = torch.tensor([rec_ix], dtype=torch.long)
    ctrl = bank.gather(ix)
    ny = bank.ystd.shape[0]
    e = torch.full((1, ny), float(e_phys), dtype=bank.A.dtype)
    xc = bank.zero_state(1)
    u_norm, _ = bank.step(xc, e / bank.ystd, ctrl)            # feed a physical residual
    expect = (ctrl[3][0] @ e[0])                              # Dc @ e, from rest
    got = u_norm[0] * bank.stdu
    rel = (got - expect).abs() / expect.abs().clamp_min(1e-30)
    return u_norm[0].detach().numpy(), expect.detach().numpy(), rel.detach().numpy()
