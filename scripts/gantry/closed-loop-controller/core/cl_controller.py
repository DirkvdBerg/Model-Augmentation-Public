"""Which record got which controller, and the design rule that built it. Gantry side only.

MIGRATION step 7: the rollout, the `ControllerBank` and the units gate have MOVED to
`model_augmentation/fit_systems/closed_loop.py`, which was always the plan (D-141: implement in
`scripts/gantry/` first, lift as a move rather than a rewrite). What stays here is the part the
framework must never learn: the map from a record to its operating point, and the `ruleOfThumb`
design rule that turns an operating point into a controller. `build_controller_bank` is the
boundary between the two.

D-140 settled the placement: `Cfb` is a SEPARATE subsystem stepped alongside the model, not a
block whose 9 states join the interconnect state vector. D-141 settled the implementation: 4 kHz,
per-record controller carried by an explicit index.

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
`diag [8.055e+06 8.253e+06 4.275e+06]` N/m at `Y_op = 0` AND `ts = 1/4000` s, the rate this
module's `ControllerBank` is built at. So the order below is the only one that closes the loop
without an algebraic loop, and it is the same order `gtd_run_simulation.m` and `closed_loop.py`
use:

    y_model[k] = h(x[k])
    e[k]       = y_data[k] - y_model[k]                     PHYSICAL [m]
    u_fb[k]    = Cc xc[k] + Dc e[k]                         PHYSICAL [N]
    u_cl[k]    = u_data[k] + u_fb[k]                        NORMALISED, see UNITS
    x[k+1]     = step(x[k], u_cl[k])
    xc[k+1]    = Ac xc[k] + Bc e[k]

The framework's rollout calls the model ONCE per step, using `Interconnect.output_only`
for `y = h(x)`. Its predecessor called it twice to work around the ordering, which doubled
the FP-plus-ANN forward cost of every step.

`Dc` IS RATE DEPENDENT, so always quote it with its `ts`. Tustin sends `z = inf` to the FINITE
frequency `s = 2/ts`, so `Dc_jj = kappa_j * Cnorm(2/ts)`, and `Cnorm` is rolling off above
`10*w_b = 6283` rad/s:

    ts = 5e-5  (20 kHz, the RECORD rate)     2/ts = 40000    Cnorm = 0.1307
    ts = 1/4000 (4 kHz, the TRAINING rate)   2/ts =  8000    Cnorm = 0.3701

a factor 2.832, which is exactly `8.055e6 / 2.844e6`. D-140's `diag [2.844e+06 2.914e+06
1.509e+06]` in `docs/decisions.md` and the problem log is the 20 kHz value and is correct THERE:
that work verified `Cfb` against the stored records, which are 20 kHz. The training loop steps
at `cfg.ts_new` (D-141) and gets the 4 kHz value. The ordering argument is unaffected either
way, since `Dc != 0` at every rate; only the magnitude differs. Verified against MATLAB at BOTH
rates by `test_controller_exact.py` (L1 and L5).

UNITS. The framework's `ControllerBank` folds the normalisation into B, C and D once at
construction and its `check_units` unfolds them to verify, so nothing here converts per step.

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


def build_controller_bank(record_names, ts, ystd, std_u, dtype=torch.float32):
    """The framework's ControllerBank for a set of records, plus each record's row in it.

    THIS is the boundary the framework never crosses. Everything gantry-specific stays on this
    side: which record sits at which `Y_op` (RECORD_Y_OP), the `ruleOfThumb` design rule, the
    frozen design plant and the Tustin discretisation (`controller_ss`). What crosses is four
    stacked matrices and an integer per record. `model_augmentation/` never learns what `Y_op` is.

    One bank over WHATEVER list it is given, indexed globally. Build it over all records once and
    slice the row list per split: the controller belongs to a trajectory, and train versus
    validation is a property of the split, not an axis of the controller. Two banks existed only
    because the old index was a position in a per-list array, so 0 meant T1 in one context and V1
    in the other.

    Returns (bank, rows, y_ops_unique) with rows[i] the controller row of record_names[i].
    """
    from model_augmentation.fit_systems.closed_loop import ControllerBank as _Bank
    names = [os.path.basename(str(n)).replace('.mat', '') for n in record_names]
    y_ops = [y_op_for(n) for n in names]
    uniq = sorted(set(y_ops))
    rows = [uniq.index(v) for v in y_ops]
    # One (A, B, C, D) per DISTINCT operating point, not per record: several records share one
    # (T6-T14 are all at 0.00), and rebuilding the same controller per record would put identical
    # rows in the stack and make the gather wider for nothing.
    mats = [controller_ss(Y_op, float(ts)) for Y_op in uniq]
    A, B, C, D = (np.stack([np.asarray(m[k], float) for m in mats]) for k in range(4))
    return _Bank(A, B, C, D, ystd=ystd, std_u=std_u, dtype=dtype), rows, uniq
