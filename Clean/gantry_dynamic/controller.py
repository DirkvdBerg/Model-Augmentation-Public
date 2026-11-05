"""The feedback controller K_fb of the closed loop: K_1, one controller per actuator axis (X1, X2, Y), tuned on the
nominal baseline at a design Y and discretised with Tustin's method. The controller bank holds one K_fb per design
in the records; the thesis data use one, K_1."""
__project_origin__ = "added"

import numpy as np
import torch
from scipy.linalg import block_diag
from scipy.signal import cont2discrete, tf2ss

from model_augmentation.fit_systems.closed_loop import ControllerBank, ClosedLoopSimulator, closed_loop_step
from model_augmentation.systems import gantry_ss

from .data import servo_sigma

BANDWIDTH_HZ = 100.0                         # [Hz] bandwidth the controller is tuned to
BANDWIDTH_RAD_S = 2 * np.pi * BANDWIDTH_HZ   # [rad/s]

# the controller is designed on the nominal baseline: nominal parameters theta^0, P, stiffness K and damping C of
# gantry_ss
P = gantry_ss.P.numpy()
K_STIFF, C_DAMP = (m.numpy() for m in gantry_ss.stiffness_and_damping(
    gantry_ss.kb1, gantry_ss.kb2, gantry_ss.cg1, gantry_ss.cg2, gantry_ss.cy, gantry_ss.cb1, gantry_ss.cb2,
    gantry_ss.Lb))


def mass_matrix_at_fixed_y(Y_op):
    """Mass matrix M(Y_op) of the nominal baseline in generalised coordinates (X, Theta, Y), with the payload held
    at position Y_op along the cross-arm.

    Args:
        Y_op: payload position Y along the cross-arm [m].
    Returns:
        (3, 3) M(Y_op).
    """
    return gantry_ss.inertia_matrix(Y_op, gantry_ss.m1, gantry_ss.m2, gantry_ss.mb, gantry_ss.mh, gantry_ss.Jb,
                                    gantry_ss.Jh, gantry_ss.Lb, gantry_ss.d).numpy()


def actuator_response_at_fixed_y(Y_op, s):
    """Frequency response of the nominal baseline with the payload held at position Y_op along the cross-arm, from
    the actuator forces (F_X1, F_X2, F_Y) to the actuator positions (X1, X2, Y).

    Args:
        Y_op: payload position Y along the cross-arm [m].
        s: Laplace variable [rad/s].
    Returns:
        (3, 3) complex response [m/N].
    """
    Minv = np.linalg.inv(mass_matrix_at_fixed_y(Y_op))
    # state space of M qddot + C qdot + K q = u in (q, qdot); P on the input and P^T on the output give the response
    # between the actuator axes
    A =np.block([[np.zeros((3, 3)), np.eye(3)], [-Minv @ K_STIFF, -Minv @ C_DAMP]])
    B = np.vstack([np.zeros((3, 3)), Minv])
    Cm = np.hstack([np.eye(3), np.zeros((3, 3))])
    return P.T @ (Cm @ np.linalg.solve(s * np.eye(6) - A, B)) @ P


def controller_shape():
    """Each axis's controller before its gain: integral action (s + w_b/6)/s, a lead filter (s + w_b/3)/(s + 3 w_b) and a
    low-pass 10 w_b/(s + 10 w_b), with w_b the bandwidth in rad/s.

    Returns:
        num (3,) and den (4,), coefficients in descending powers of s.
    """
    w_b = BANDWIDTH_RAD_S
    num = 10 * w_b * np.polymul([1., w_b / 6], [1., w_b / 3])
    den = np.polymul([1., 0.], np.polymul([1., 3 * w_b], [1., 10 * w_b]))
    return num, den


def axis_controllers(Y_op, ts):
    """The three per-axis controllers of K_1 for the design payload position Y_op: the shape times a gain per axis that
    sets the open-loop magnitude of that axis to 1 at the bandwidth, discretised with Tustin's method at ts.

    Args:
        Y_op: design payload position Y along the cross-arm [m].
        ts: sample time T_s [s].
    Returns:
        three (b, a) pairs: the numerator and denominator coefficients of each axis's K_1(z).
    """
    # 1. Plant and controller shape at the bandwidth.
    plant_at_bandwidth = actuator_response_at_fixed_y(Y_op, 1j * BANDWIDTH_RAD_S)
    num, den = controller_shape()
    shape_at_bandwidth = np.polyval(num, 1j * BANDWIDTH_RAD_S) / np.polyval(den, 1j * BANDWIDTH_RAD_S)
    axis_coefficients = []
    for j in range(3):
        # 2. Gain that sets this axis's open-loop magnitude to 1 at the bandwidth, tuned on the diagonal entry of the
        # plant only, as the data generator does
        axis_gain = 1.0 / abs(plant_at_bandwidth[j, j] * shape_at_bandwidth)
        # 3. Tustin (scipy's 'bilinear') at ts, the rule the data generator used for K_1; ZOH not compared
        b, a, _ = cont2discrete((axis_gain * num, den), ts, method='bilinear')
        axis_coefficients.append((np.asarray(b).ravel(), np.asarray(a).ravel()))
    return axis_coefficients


def controller_ss(Y_op, ts, gain=1.0):
    """K_fb as one state space: the three per-axis controllers block diagonal, so the position error of each axis
    drives only the force of that axis, times gain. Input the position error r - y [m], output the feedback force [N]
    per actuator axis.

    Args:
        Y_op: design payload position Y along the cross-arm [m].
        ts: sample time T_s [s].
        gain: factor on the controller output; 1 for K_1, other values for a record generated with a scaled K_1.
    Returns:
        A_fb (9, 9), B_fb (9, 3), C_fb (3, 9), D_fb (3, 3), in physical units.
    """
    As, Bs, Cs, Ds = zip(*(tf2ss(b, a) for b, a in axis_controllers(Y_op, ts)))
    return block_diag(*As), block_diag(*Bs), gain * block_diag(*Cs), gain * block_diag(*Ds)


def design_y_and_gain(name, spec, use_gain=False):
    """Design Y and gain of a record's stored controller. Raises a ValueError for a bandwidth other than BANDWIDTH_HZ
    or, without use_gain, a gain other than 1."""
    controller_name, design_y, gain, bandwidth_hz = spec
    # small tolerances, so stored values that differ only by rounding still match
    if abs(bandwidth_hz - BANDWIDTH_HZ) >= 1e-9:
        raise ValueError('record %r uses controller %s at %g Hz; the bank only builds the %g Hz design'
                         % (name, controller_name, bandwidth_hz, BANDWIDTH_HZ))
    if abs(gain - 1.0) > 1e-12 and not use_gain:
        raise ValueError('record %r uses %g x K_1; a gain other than 1 is only built with use_gain=True'
                         % (name, gain))
    return design_y, gain


def build_controller_bank(record_names, controllers, ts, ystd, std_u, dtype, use_gain=False):
    """ControllerBank with the stored controller of each record. Records with the same controller share one entry;
    rows gives each record's entry.

    Args:
        record_names: record file names.
        controllers: record file name -> stored controller spec (Datasets.controllers).
        ts: sample time T_s [s].
        ystd: (3,) standard deviation of the measured positions [m].
        std_u: (3, 1) or (3,) standard deviation of the actuator forces [N].
        dtype: model dtype.
        use_gain: allow controllers with a stored gain other than 1.
    Returns:
        bank, rows (index of each record's K_fb), and the distinct (design Y, gain) pairs.
    """
    designs = [design_y_and_gain(name, controllers[name], use_gain) for name in record_names]
    distinct_designs = sorted(set(designs))
    rows = [distinct_designs.index(design) for design in designs]
    state_spaces = [controller_ss(Y_op, float(ts), gain) for Y_op, gain in distinct_designs]
    A, B, C, D = (np.stack([np.asarray(state_space[k], float) for state_space in state_spaces]) for k in range(4))
    bank = ControllerBank(A, B, C, D, ystd=ystd, std_u=std_u, dtype=dtype)
    return bank, rows, distinct_designs


def build_closed_loop(model, norm, cfg, *, train_files, val_files, controllers, e_train, e_val, verbose=True):
    """Closed-loop simulator of training and validation.

    It holds the bank of K_fb, the loss weight per training record and channel (sigma_y / sigma_e) and sigma_e of
    the validation records.

    Args:
        model: the augmented model.
        norm: means and standard deviations of the training records (Norm).
        cfg: run configuration.
        train_files, val_files: record file names.
        controllers: record file name -> stored controller spec (Datasets.controllers).
        e_train, e_val: lists of (N_r, 3) servo errors y - r [m].
        verbose: print the bank and compile settings.
    Returns:
        ClosedLoopSimulator.
    """
    # 1. Controller bank over the training and validation records.
    bank, rows, distinct_designs = build_controller_bank(
        list(train_files) + list(val_files), controllers, cfg.ts_new, ystd=norm.ystd, std_u=norm.std_u,
        dtype=cfg.dtype_pt)
    n_train = len(train_files)
    train_rows, val_rows = rows[:n_train], rows[n_train:]

    if verbose:
        print('[closed loop] one bank over %d records, %d distinct controllers, (design Y, gain) %s'
              % (len(rows), bank.M_state.shape[0], distinct_designs))
        print('[closed loop] stepped at ts = %g s (%d Hz), %d controller states'
              % (cfg.ts_new, round(1 / cfg.ts_new), bank.nc))
    # 2. Compiled closed-loop step (residual form, closed_loop.py).
    # used only for training rollouts on CUDA; validation and evaluation simulate whole records uncompiled, without
    # gradients
    compiled = None
    if cfg.compile_mode is not None:
        compiled = torch.compile(closed_loop_step(model.hfn, bank), mode=cfg.compile_mode, fullgraph=True)
        if verbose:
            print('[closed loop] training rollout COMPILED: mode=%s fullgraph=True' % cfg.compile_mode)
    # 3. Loss and validation scales.
    # first scored sample: the encoder uses the first max(na, nb) samples
    k0 = max(model.na, model.nb)
    val_sigma_e = servo_sigma(e_val, k0)
    # loss weight per training record and channel: sigma_y / sigma_e, so the loss is relative to each record's servo
    # error
    train_weights = np.asarray(norm.ystd, np.float64).ravel() / servo_sigma(e_train, k0)
    return ClosedLoopSimulator(bank, train_rows, val_rows, compiled=compiled,
                               train_weights=train_weights, val_sigma_e=val_sigma_e)
