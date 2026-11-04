"""Closed-loop simulation of the augmented model with the known controller K_fb, in residual form (Kessels): the
controller bank, one closed-loop step, the rollout, and the simulator that training and validation call."""
__project_origin__ = "added"

from contextlib import contextmanager

import numpy as np
import torch

# name of the validation score in the logs; defined in ClosedLoopSimulator.validation_error
SCORE_NAME = 'sim-NRMSe'


class ControllerBank(torch.nn.Module):
    """The known feedback controllers K_fb of the training and validation records, rewritten to act on normalised
    signals. Records with the same controller design share one bank entry. The controller parameters are not trained."""

    def __init__(self, A, B, C, D, ystd, std_u, dtype=torch.float32):
        """Store each K_fb in the normalised coordinates of the model.

        With reference r, measured positions y and controller state x^fb, the known controller maps position
        error r - y [m] to feedback force u^fb [N]:
            x^fb_{k+1} = A_fb x^fb_k + B_fb (r_k - y_k),
            u^fb_k = C_fb x^fb_k + D_fb (r_k - y_k).
        The gantry controller has nc = 9 states, three per actuator axis.

        Args:
            A, B, C, D: A_fb (n_d, nc, nc), B_fb (n_d, nc, 3), C_fb (n_d, 3, nc), D_fb (n_d, 3, 3), stacked over the
                n_d distinct designs.
            ystd: sigma_y, standard deviation of the measured positions [m].
            std_u: sigma_u, standard deviation of the actuator forces [N].
            dtype: model dtype.
        """
        super().__init__()
        T = lambda M: torch.as_tensor(np.asarray(M, dtype=float), dtype=dtype)
        A, B, C, D = T(A), T(B), T(C), T(D)
        ystd_t, stdu_t = T(np.asarray(ystd).ravel()), T(np.asarray(std_u).ravel())
        self.nc, self.nu = A.shape[-1], C.shape[-2]

        # The model uses y^n = T_y (y - mu_y) and u^n = T_u (u_act - mu_u), with
        # T_i = diag(sigma_i)^-1, while the controller state stays in physical units. A normalised error goes back to
        # metres with T_y^-1 and a force to normalised units with T_u:
        #   B_fb^n = B_fb T_y^-1,  C_fb^n = T_u C_fb,  D_fb^n = T_u D_fb T_y^-1,  A_fb unchanged.
        # The means cancel in output and force differences; the recorded u^n already includes subtraction of mu_u.
        Bn = B * ystd_t[None, None, :]
        Cn = C / stdu_t[None, :, None]
        Dn = D / stdu_t[None, :, None] * ystd_t[None, None, :]
        # stacked so that one controller step is one product, see step
        self.register_buffer('M_state', torch.cat([Cn, A], dim=1))
        self.register_buffer('M_error', torch.cat([Dn, Bn], dim=1))

    def step(self, xc, e_norm, ctrl):
        """One step of the known controller in residual form.

        x^c = x_hat^fb - x^fb is the model's controller state minus the system's; e^n = y^n - y_hat^n is the
        normalised output residual. Both controllers act on r minus their own output. Subtracting their equations
        cancels the common reference r and gives
            u_fb^n = C_fb^n x^c_k + D_fb^n e^n_k,
            x^c_{k+1} = A_fb x^c_k + B_fb^n e^n_k.
        u_fb^n is the normalised difference between the two feedback forces, added to the recorded force.

        Args:
            xc: (B, nc) x^c_k, in physical controller units.
            e_norm: (B, 3) e^n_k.
            ctrl: (M_state, M_error) of each sequence, gathered by closed_loop_rollout.
        Returns:
            u_fb^n (B, 3) and x^c_{k+1} (B, nc).
        """
        M_state, M_error = ctrl
        # (u_fb^n, x^c_{k+1}) = [C_fb^n; A_fb] x^c_k + [D_fb^n; B_fb^n] e^n_k
        out = torch.baddbmm(torch.bmm(M_error, e_norm.unsqueeze(-1)),
                            M_state, xc.unsqueeze(-1)).squeeze(-1)
        return out[:, :self.nu], out[:, self.nu:]


def closed_loop_step(hfn, bank):
    """One closed-loop step of the augmented model, built as a function of hfn and bank so it can be compiled.

    x_tilde^n is the physical part of the model state and C^n its output map. The output has no input
    feedthrough, so compute it before the controller correction u_fb^n from ControllerBank.step:
        y_hat^n_k = C^n x_tilde^n_k,   u_hat^n_k = u^n_k + u_fb^n_k.
    Then advance hfn with u_hat^n_k. This order avoids an algebraic loop through the controller feedthrough D_fb
    (Kessels Remark 5.1).

    Args:
        hfn: the interconnect, one step of the augmented model.
        bank: the ControllerBank.
    Returns:
        step(x, xc, u_t, y_t, ctrl, obc_pass, pass_mats) -> (y_hat^n_k, x_hat^n_{k+1}, x^c_{k+1}), with x the model
        state (B, 6 + n_a), xc (B, nc), u_t and y_t the recorded u^n_k and y^n_k (B, 3), and the rest as in
        closed_loop_rollout.
    """
    def step(x, xc, u_t, y_t, ctrl, obc_pass, pass_mats):
        """One closed-loop step; see closed_loop_step."""
        y_model = hfn.output_only(x)
        u_fb, xc = bank.step(xc, y_t - y_model, ctrl)
        # hfn also returns y_hat^n_k, already computed above
        _, x = hfn(x, u_t + u_fb, obc_pass, pass_mats)
        return y_model, x, xc
    return step


def closed_loop_rollout(step, u_data, y_data, x0, bank, ctrl_ix, *, pass_mats, obc_pass=None):
    """Simulate the augmented model in closed loop with each sequence's K_fb, from x0 over the length of u_data.

    The model input is u_hat^n_k = u^n_k + u_fb^n_k (ControllerBank.step), so only the recorded u^n and y^n are
    needed, not the common reference r or feedforward u^ff. This reproduces their explicit model loop when the
    recorded force was generated by the same discrete K_fb. Here the records were generated at 20 kHz, while the
    controller is re-discretised at the model rate and recorded forces are averaged. The resulting force mismatch
    acts as an extra input term; exact equivalence is not assumed.

    x0 is the encoder estimate x_hat^n_{tau|tau} in training and validation, and the baseline start state in the
    encoder start (model.py).

    Args:
        step: closed_loop_step(hfn, bank), compiled or eager.
        u_data, y_data: (B, N, 3) recorded u^n, y^n from the first simulated sample on.
        x0: (B, 6 + n_a) initial model state, n_a added states.
        bank: the ControllerBank.
        ctrl_ix: (B,) design index of each sequence.
        pass_mats: baseline matrices at the current parameters (objective_matrices).
        obc_pass: OBC correction matrices; None on the U arm, where the step has no correction.
    Returns:
        y_hat^n (B, N, 3), the final x_hat^n (B, 6 + n_a) and the final x^c (B, nc).
    """
    # the bank is not a submodule of the model, so moving the model does not move it
    if bank.M_state.device != u_data.device:
        bank.to(u_data.device)
    # each sequence's controller, gathered once for all steps
    ctrl = (bank.M_state[ctrl_ix], bank.M_error[ctrl_ix])

    # Since x^c is the controller-state difference, x^c_tau = 0 starts both controllers in the same state.
    # The encoder therefore only estimates the model states.
    xc = torch.zeros(u_data.shape[0], bank.nc, dtype=u_data.dtype, device=u_data.device)

    x, ys = x0, []
    for u_t, y_t in zip(u_data.unbind(1), y_data.unbind(1)):
        y_model, x, xc = step(x, xc, u_t, y_t, ctrl, obc_pass, pass_mats)
        ys.append(y_model)
    return torch.stack(ys, dim=1), x, xc


def window_counts(fit_sys, sdl, nf, stride):
    """Number of training windows per record, as deepSI's to_hist_future_data builds them: window starts
    tau = k0, k0 + stride, ... with tau + n_f <= N, where k0 = max(na, nb) leaves room for the encoder history."""
    first = max(fit_sys.na, fit_sys.nb) + nf
    return [len(range(first, len(sd.u) + 1, stride)) for sd in sdl]


class ClosedLoopSimulator:
    """Closed-loop simulator of training and validation, attached as fit_sys.simulator: the training loss calls it
    through fit_sys.simulate, make_training_data calls augment_training_data, and cal_validation_error calls
    validation_error."""

    def __init__(self, bank, train_ctrl_rows, val_rows, compiled, train_weights, val_sigma_e):
        """Store controllers and per-record scales: sigma_y is the output standard deviation; sigma_e is the
        standard deviation of servo error y - r after the initial encoder history.

        Args:
            bank: ControllerBank over the training and validation records.
            train_ctrl_rows, val_rows: design index of each training and validation record, in record order.
            compiled: torch.compile of closed_loop_step for the training rollouts on CUDA, or None.
            train_weights: (R_train, 3) diagonal of W_r = diag(sigma_y / sigma_e,r) of each training record.
            val_sigma_e: (R_val, 3) sigma_e of each validation record [m].
        """
        self.bank = bank
        # per-window arrays appended by augment_training_data; fit passes them to loss under these names
        self.extra_array_names = ('ctrl_ix', 'loss_w')
        self.val_sigma_e = np.asarray(val_sigma_e, np.float64)
        self._compiled = compiled
        self.train_ctrl_rows = [int(r) for r in train_ctrl_rows]
        self.record_weights = [np.asarray(w) for w in train_weights]
        self.val_rows = [int(r) for r in val_rows]

    def __getstate__(self):
        """Leave the compiled step out when deepSI pickles the model; a restored simulator runs eager."""
        state = self.__dict__.copy()
        state['_compiled'] = None
        return state

    @contextmanager
    def eager(self):
        """Use the eager step inside the block, then restore the compiled one; the polish falls back to it when the
        compiled loss raises or does not reproduce."""
        saved, self._compiled = self._compiled, None
        try:
            yield
        finally:
            self._compiled = saved

    def __call__(self, fit_sys, x, ufuture, yfuture, ctrl_ix, *, pass_mats, obc_pass=None, **kw):
        """Simulate a batch of windows for V (AugmentedModel.loss) or the PS residual target.

        Args:
            fit_sys: the augmented model.
            x: (B, 6 + n_a) encoder estimates x_hat^n_{tau|tau}.
            ufuture, yfuture: (B, n_f, 3) recorded u^n, y^n of each window.
            ctrl_ix: (B,) design index of each window.
            pass_mats, obc_pass: as in closed_loop_rollout.
            kw: nf and stride, which loss passes on; unused here.
        Returns:
            y_hat^n_{k|tau} (B, n_f, 3) and the final x_hat^n (B, 6 + n_a).
        """
        # the compiled step serves only CUDA rollouts that record gradients, i.e. training losses; others run eager
        use_compiled = (self._compiled is not None and ufuture.is_cuda and torch.is_grad_enabled())
        if use_compiled:
            # the interconnect's one-time setup (computation order, connection matrices on the model's device) cannot
            # be traced; run it eagerly before the first compiled call, which a polish-only job reaches first
            if not fit_sys.hfn.initialized_forward_function:
                fit_sys.hfn.init_forward()
                fit_sys.hfn.initialized_forward_function = True
            step = self._compiled
            # marks a new call, so the CUDA graphs may reuse the previous call's buffers
            torch.compiler.cudagraph_mark_step_begin()
        else:
            step = closed_loop_step(fit_sys.hfn, self.bank)

        y_pred, x_final, _ = closed_loop_rollout(step, ufuture, yfuture, x, self.bank, ctrl_ix.long(),
                                                 obc_pass=obc_pass, pass_mats=pass_mats)
        if use_compiled:
            # copy out of the CUDA graph buffers, which the next call overwrites
            y_pred, x_final = y_pred.clone(), x_final.clone()
        return y_pred, x_final

    def augment_training_data(self, data, sys_data, fit_sys, **kw):
        """Append to deepSI's training windows the design index and the W_r diagonal of each window's record.

        deepSI concatenates windows in record order; the metadata must follow that order to select the right
        controller and error scale. Check counts and each record's first and last future-input window.
        In AugmentedModel.loss, W_r = diag(sigma_y / sigma_e,r) converts normalised output error to
        (y - y_hat) / sigma_e,r, relative to each record's servo-error standard deviation.

        Returns:
            data + [ctrl_ix (N_w,), loss_w (N_w, 3)], N_w windows in total.
        """
        # deepSI's own defaults, for a direct call; fit and the polish pass nf and stride
        nf = kw.get('nf', 25)
        stride = kw.get('stride', 1)
        sdl = sys_data.sdl
        k0 = max(fit_sys.na, fit_sys.nb)
        counts = window_counts(fit_sys, sdl, nf, stride)
        ctrl_ix = np.concatenate([np.full(c, int(r), dtype=np.int64)
                                  for c, r in zip(counts, self.train_ctrl_rows)])
        if len(ctrl_ix) != len(data[0]):
            raise RuntimeError(
                'ctrl_ix misalignment: %d windows derived, %d in the training data; make the window count here '
                'follow the windowing of WindowedSystemData' % (len(ctrl_ix), len(data[0])))
        self.verify_by_content(data, sdl, counts, k0, nf, stride)
        loss_w = np.concatenate([np.tile(w, (c, 1)) for w, c in zip(self.record_weights, counts)])
        return list(data) + [ctrl_ix, loss_w]

    @staticmethod
    def verify_by_content(data, sdl, counts, k0, nf, stride):
        """Compare the input u^n of each record's first and last window with the record itself.

        Only these boundary inputs are checked; outputs, histories and intermediate windows are not checked."""
        ufuture = data[2]
        off = 0
        for sd, c in zip(sdl, counts):
            for j in (0, c - 1):
                start = k0 + j * stride
                got = np.asarray(ufuture[off + j])
                want = np.asarray(sd.u[start:start + nf], dtype=got.dtype)
                if not np.array_equal(got, want):
                    raise RuntimeError(
                        'window %d does not match its record at offset %d; check the window values and offsets '
                        'before trusting the controller-row assignment' % (off + j, start))
            off += c

    def validation_error(self, fit_sys, val_sys_data):
        """Validation score sim-NRMSe, averaged over R records and axes j = X1, X2, Y.

        For record r, RMS_{r,j} is the whole-record closed-loop output error [m] and sigma_{e,r,j} is the standard
        deviation of servo error y_j - reference_j [m] over the same scored samples:
            sim-NRMSe = 1 / (3 R) sum_r sum_j RMS_{r,j} / sigma_{e,r,j}.
        Using sigma_e instead of the output standard deviation avoids normalising by the much larger reference
        motion. Predicting the reference plus the mean servo error gives a score of 1.

        Adam keeps the lowest-score checkpoint. Polish acceptance requires finite scores and final training loss,
        with a strict score reduction.
        """
        sdl = val_sys_data.sdl
        per_channel, _ = closed_loop_free_run_rms_batch(fit_sys, sdl, self.bank, self.val_rows)
        # kept for the per-record validation history in training.py
        self.last_per_channel = np.asarray(per_channel)          # (R, 3) RMS_{r,j} [m]
        return float(np.mean(self.last_per_channel / self.val_sigma_e))


def closed_loop_free_run_rms_batch(fit_sys, sdl, bank, ctrl_rows, return_pred=False):
    """Closed-loop free run of whole records, for validation and evaluation.

    Each record starts from one encoder estimate x_hat^n_{k0} = psi(chi_{k0}) and controller-state difference
    x^c = 0. Neither state is reset during the record; the residual feedback remains active.
    The first k0 = max(fit_sys.na, fit_sys.nb) samples form the encoder history. For a record of N samples,
    score samples k0 .. N - 1 per axis j:

        RMS_j = ( 1 / (N - k0) sum_{k=k0}^{N-1} (y_hat_{j,k} - y_{j,k})^2 )^(1/2)   [m].

    Runs without gradients and with the eager step; records of equal length run as one batch.

    Args:
        fit_sys: the augmented model.
        sdl: the records, unnormalised.
        bank, ctrl_rows: the controller bank and the design index of each record.
        return_pred: also return the predictions.
    Returns:
        per_channel: per record (3,) RMS_j [m]; aggregate: per record (mean_j RMS_j^2)^(1/2) [m]; with return_pred
        also preds: per record (N - k0, 3) y_hat [m].
    """
    norm = fit_sys.norm
    na, nb = fit_sys.na, fit_sys.nb
    k0 = max(na, nb)
    parameter = next(fit_sys.hfn.parameters())
    dtype = parameter.dtype
    device = parameter.device
    np_dtype = np.float64 if dtype == torch.float64 else np.float32
    rv = lambda a: np.asarray(a).ravel()
    T = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)

    if len(sdl) != len(ctrl_rows):
        raise RuntimeError(
            'closed_loop_free_run_rms_batch got %d records and %d controller rows; pass one controller row per '
            'record, in record order' % (len(sdl), len(ctrl_rows)))

    # u^n = T_u (u - mu_u), y^n = T_y (y - mu_y)
    un = [((sd.u - rv(norm.u0)) / rv(norm.ustd)).astype(np_dtype) for sd in sdl]
    yn = [((sd.y - rv(norm.y0)) / rv(norm.ystd)).astype(np_dtype) for sd in sdl]

    # records of equal length share one batch
    groups = {}
    for i, u in enumerate(un):
        groups.setdefault(len(u) - k0, []).append(i)

    per_channel = [None] * len(sdl)
    aggregate = [None] * len(sdl)
    preds = [None] * len(sdl)                          # filled with return_pred only
    with torch.no_grad():
        # baseline and OBC matrices at the current parameters, shared by every group and step
        step_kw = fit_sys.objective_matrices()
    for ixs in groups.values():
        # chi_{k0}: the nb inputs and na outputs before k0
        uh = T(np.stack([un[i][k0 - nb:k0] for i in ixs]))
        yh = T(np.stack([yn[i][k0 - na:k0] for i in ixs]))
        with torch.no_grad():
            x0 = fit_sys.encoder(uh, yh)
            u_t = T(np.stack([un[i][k0:] for i in ixs]))
            y_t = T(np.stack([yn[i][k0:] for i in ixs]))
            y_pred, _, _ = closed_loop_rollout(
                closed_loop_step(fit_sys.hfn, bank), u_t, y_t, x0, bank,
                torch.tensor([int(ctrl_rows[i]) for i in ixs], dtype=torch.long, device=device), **step_kw)
        y_pred = y_pred.cpu().numpy()
        for j, i in enumerate(ixs):
            # back to metres: y_hat = T_y^-1 y_hat^n + mu_y
            y_phys = y_pred[j] * rv(norm.ystd) + rv(norm.y0)
            e = y_phys - np.asarray(sdl[i].y)[k0:]
            mse_per_channel = np.mean(e ** 2, axis=0)
            per_channel[i] = np.sqrt(mse_per_channel)
            aggregate[i] = float(np.sqrt(np.mean(mse_per_channel)))
            if return_pred:
                preds[i] = y_phys
    if return_pred:
        return per_channel, aggregate, preds
    return per_channel, aggregate
