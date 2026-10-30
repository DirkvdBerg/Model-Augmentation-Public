"""Closed-loop model simulation with the known controller K_fb in residual form (Kessels): the controller bank, the
rollout and the simulator of training and validation.

Conventions: batch rows are samples (batched products bmm); superscript n = normalised; xc = x^c = x_hat^fb - x^fb,
kept in physical controller units; e^n = y^n - y_hat^n. C^n, D^n here are the controller seen from the normalised
model (B^n = B diag(sigma_y), C^n = diag(1/sigma_u) C, D^n = diag(1/sigma_u) D diag(sigma_y)), not the model's
output map C^n.
"""
__project_origin__ = "added"

from contextlib import contextmanager

import numpy as np
import torch


class ControllerBank(torch.nn.Module):
    """The distinct controllers K_fb of a data set seen from the normalised model, gathered per window by design
    index."""

    def __init__(self, A, B, C, D, ystd, std_u, dtype=torch.float32):
        """Controller bank, one per distinct design.

        Built once, so the rollout uses normalised position errors and force corrections while keeping x^c in the
        controller's own state coordinates.

        Args:
            A: (K, nc, nc) controller state matrices.
            B: (K, nc, ny) maps the physical position error (m) to the next controller state.
            C: (K, nu, nc) maps the controller state to physical force (N).
            D: (K, nu, ny) feedthrough, N/m.
            ystd: sigma_y, (ny,) or (ny, 1), metres.
            std_u: sigma_u, (nu,) or (nu, 1), newtons.
            dtype: model dtype.
        """
        super().__init__()
        T = lambda M: torch.as_tensor(np.asarray(M, dtype=float), dtype=dtype)
        A, B, C, D = T(A), T(B), T(C), T(D)
        ystd_t, stdu_t = T(np.asarray(ystd).ravel()), T(np.asarray(std_u).ravel())
        self.nc, self.nu = A.shape[-1], C.shape[-2]

        # THEORY: eps = T_y^-1 e^n and u_fb^n = T_u u_fb give B^n = B diag(sigma_y), C^n = diag(1/sigma_u) C,
        # D^n = diag(1/sigma_u) D diag(sigma_y).
        Bn = B * ystd_t[None, None, :]
        Cn = C / stdu_t[None, :, None]
        Dn = D / stdu_t[None, :, None] * ystd_t[None, None, :]
        # one step is one product: (u_fb^n, xc_next) = [C^n; A] xc + [D^n; B^n] e^n
        self.register_buffer('M_state', torch.cat([Cn, A], dim=1))
        self.register_buffer('M_error', torch.cat([Dn, Bn], dim=1))

    @property
    def n_controllers(self):
        """Number of distinct controllers in the bank."""
        return self.M_state.shape[0]

    def step(self, xc, e_norm, ctrl):
        """Controller step in residual form.

        u_fb^n = C^n x^c + D^n e^n, x^c_{k+1} = A x^c + B^n e^n, with e^n = y^n - y_hat^n.

        Args:
            xc: (B, nc) x^c_k.
            e_norm: (B, ny) e^n_k.
            ctrl: (M_state, M_error) rows of the bank for this batch.
        Returns:
            u_fb^n (B, nu) and x^c_{k+1} (B, nc).
        """
        # THEORY: the controller is a separate state equation beside the model, not part of the model state
        M_state, M_error = ctrl
        # Both rows in one product: (u_fb^n, x^c_{k+1}) = [C^n; A] x^c + [D^n; B^n] e^n.
        out = torch.baddbmm(torch.bmm(M_error, e_norm.unsqueeze(-1)),
                            M_state, xc.unsqueeze(-1)).squeeze(-1)
        return out[:, :self.nu], out[:, self.nu:]


def closed_loop_step(hfn, bank):
    """Closed-loop model step.

    y_hat^n = C^n x_tilde^n, u_hat^n = u^n + T_u (C_fb x^c + D_fb eps), x^c update, then x_hat^n_{k+1} from the
    augmented model with u_hat^n; the training rollout compiles the returned step as one graph.

    Args:
        hfn: the interconnect (augmented model).
        bank: the ControllerBank.
    Returns:
        step(x, xc, u_t, y_t, ctrl, obc_pass, pass_mats) -> (y_hat^n_k, x_hat^n_{k+1}, x^c_{k+1}), with x (B, nx),
        xc (B, nc) in controller units, u_t (B, nu) and y_t (B, ny) normalised, ctrl = (M_state (B, nu + nc, nc),
        M_error (B, nu + nc, ny)).
    """
    def step(x, xc, u_t, y_t, ctrl, obc_pass, pass_mats):
        """One step: y_hat^n = C^n x_tilde^n, u_hat^n = u^n + u_fb^n, then x_hat^n_{k+1}."""
        # THEORY: the controller sees the model output; output before input, so D_fb closes no algebraic loop
        # (Kessels Remark 5.1)
        # Output: y_hat^n = C^n x_tilde^n.
        y_model = hfn.output_only(x)
        # Feedback on e^n = y^n - y_hat^n, and x^c_{k+1}.
        u_fb, xc = bank.step(xc, y_t - y_model, ctrl)
        # State update with the model input u_hat^n = u^n + u_fb^n.
        _, x = hfn(x, u_t + u_fb, obc_pass, pass_mats)
        return y_model, x, xc
    return step


def closed_loop_rollout(step, u_data, y_data, x0, bank, ctrl_ix, obc_pass=None, pass_mats=None):
    """Closed-loop simulation of a window.

    y_hat^n_{k|tau} for k = tau .. tau + n_f - 1 from x_hat^n_{tau|tau} = x0 and x^c_tau = 0, driven by the recorded
    u^n and y^n; used by training, validation and the encoder start.

    Args:
        step: a closed_loop_step (compiled or eager).
        u_data, y_data: (B, n_f, 3) recorded u^n and y^n, normalised.
        x0: (B, 6 + n_xbar) x_hat^n_{tau|tau}.
        bank: the ControllerBank.
        ctrl_ix: (B,) controller row of each window.
        obc_pass, pass_mats: matrices of this objective.
    Returns:
        y_hat^n (B, n_f, 3); x final (B, 6 + n_xbar), x_tilde^n normalised and xbar; x^c final (B, nc), controller
        units.
    """
    if bank.M_state.device != u_data.device:
        bank.to(u_data.device)
    ctrl = (bank.M_state[ctrl_ix], bank.M_error[ctrl_ix])

    # THEORY: x^c = x_hat^fb - x^fb, the two controller loops subtracted (Kessels Eqs. (5.12), (5.13)); x^c_tau = 0
    # sets the model's controller state to the system's at the window start
    xc = torch.zeros(u_data.shape[0], bank.nc, dtype=u_data.dtype, device=u_data.device)

    x, ys = x0, []
    for u_t, y_t in zip(u_data.unbind(1), y_data.unbind(1)):
        y_model, x, xc = step(x, xc, u_t, y_t, ctrl, obc_pass, pass_mats)
        ys.append(y_model)
    return torch.stack(ys, dim=1), x, xc


def window_counts(fit_sys, sdl, nf, stride):
    """Number of training windows per record, as WindowedSystemData builds them: ends k0 + n_f .. N, at the stride."""
    first = max(fit_sys.na, fit_sys.nb) + nf
    return [len(range(first, len(sd.u) + 1, stride)) for sd in sdl]


class ClosedLoopSimulator:
    """The closed loop of training and validation; assigned to fit_sys.simulator."""

    def __init__(self, bank, train_ctrl_rows, val_rows, compiled, train_weights, val_sigma_e):
        """Store the bank, the controller row and W_r per training record, sigma_e per validation record, the step."""
        self.bank = bank
        # loss() receives these per-window arrays by name
        self.extra_array_names = ('ctrl_ix', 'loss_w')
        # sigma_e per validation record and axis (n_val, ny), the denominator of sim-NRMSe
        self.val_sigma_e = np.asarray(val_sigma_e, np.float64)
        # the closed_loop_step compiled by torch.compile, or None for eager
        self._compiled = compiled
        self.train_ctrl_rows = [int(r) for r in train_ctrl_rows]
        # W_r per training record, one (ny,) row each, appended to every window as loss_w
        self.record_weights = [np.asarray(w) for w in train_weights]
        self.val_rows = [int(r) for r in val_rows]

    def __getstate__(self):
        """The state without the compiled step, a cache that cannot be pickled."""
        state = self.__dict__.copy()
        state['_compiled'] = None
        return state

    @contextmanager
    def eager(self):
        """Run through the uncompiled step inside the block, then restore."""
        saved, self._compiled = self._compiled, None
        try:
            yield
        finally:
            self._compiled = saved

    def __call__(self, fit_sys, x, ufuture, yfuture, ctrl_ix, obc_pass=None, pass_mats=None, **kw):
        """Closed-loop rollout of a batch of windows (the simulator of V).

        (y_hat^n_{k|tau}, x final) from x_hat^n_{tau|tau} = x; gradient-enabled CUDA rollouts use the compiled step.

        Args:
            fit_sys: the augmented model.
            x: (B, 6 + n_xbar) encoder states.
            ufuture, yfuture: (B, n_f, 3) recorded u^n, y^n.
            ctrl_ix: (B,) controller rows; obc_pass, pass_mats: matrices of this objective.
        Returns:
            y_hat^n (B, n_f, 3) and the final state.
        """
        # use the compiled step for gradient-enabled CUDA rollouts, including the polish
        use_compiled = (self._compiled is not None and ufuture.is_cuda and torch.is_grad_enabled())
        if use_compiled:
            step = self._compiled
            torch.compiler.cudagraph_mark_step_begin()
        else:
            step = closed_loop_step(fit_sys.hfn, self.bank)

        y_pred, x_final, _ = closed_loop_rollout(step, ufuture, yfuture, x, self.bank, ctrl_ix.long(),
                                                 obc_pass=obc_pass, pass_mats=pass_mats)
        if use_compiled:
            # copies out of the CUDA graph's buffers, which the next step overwrites
            y_pred, x_final = y_pred.clone(), x_final.clone()
        return y_pred, x_final

    def augment_training_data(self, data, sys_data, fit_sys, **kw):
        """Check window alignment and append each window's controller row and W_r = diag(sigma_y / sigma_e,r)."""
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
        # same record order and counts as ctrl_ix, so the checks above cover it
        loss_w = np.concatenate([np.tile(w, (c, 1)) for w, c in zip(self.record_weights, counts)])
        return list(data) + [ctrl_ix, loss_w]

    @staticmethod
    def verify_by_content(data, sdl, counts, k0, nf, stride):
        """Each record's first and last window against its raw data at the derived offset."""
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
        """Validation error for checkpoint selection.

        Mean over records and axes j of NRMS_e,j = RMS_j / sigma_e,j, each record simulated in closed loop from one
        encoder estimate; Adam keeps the parameters with the lowest value.

        Args:
            fit_sys: the augmented model.
            val_sys_data: validation records.
        Returns:
            The validation error (dimensionless).
        """
        sdl = val_sys_data.sdl
        per_channel, per_record = closed_loop_free_run_rms_batch(fit_sys, sdl, self.bank, self.val_rows)
        self.last_per_channel = np.asarray(per_channel)          # (n_val, ny), metres
        # THEORY: deepSI's multi-record sim-RMS, the mean of the per-record RMS weighted by the full record
        # length (System_data_list.RMS)
        # sim_RMS = sum_r N_r RMS_r / sum_r N_r.
        self.last_sim_rms = float(np.average(np.asarray(per_record),
                                             weights=[len(sd.u) for sd in sdl]))
        return float(np.mean(self.last_per_channel / self.val_sigma_e))


def closed_loop_free_run_rms_batch(fit_sys, sdl, bank, ctrl_rows, return_pred=False):
    """Closed-loop free run of full records.

    RMS_j = (mean over k = k0 .. N - 1 of (y_hat_j,k - y_j,k)^2)^(1/2), from x_hat^n_{k0} = psi(chi_{k0}) and
    k0 = n + 1; records of equal length are simulated as one batch.

    Args:
        fit_sys: the augmented model; sdl: records (physical); bank, ctrl_rows: bank and one row per record.
        return_pred: also return the physical predictions.
    Returns:
        per_channel: length-R list of (ny,) RMS arrays in metres; aggregate: length-R list of scalar record RMS
        values in metres, both in record order; with return_pred, preds: length-R list of (N_r - k0, ny) physical
        position predictions in metres.
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

    # 1. Normalise, u^n = (u - mu_u) / sigma_u, y^n = (y - mu_y) / sigma_y, and group records by length.
    un = [((sd.u - rv(norm.u0)) / rv(norm.ustd)).astype(np_dtype) for sd in sdl]
    yn = [((sd.y - rv(norm.y0)) / rv(norm.ystd)).astype(np_dtype) for sd in sdl]

    groups = {}
    for i, u in enumerate(un):
        groups.setdefault(len(u) - k0, []).append(i)

    per_channel = [None] * len(sdl)
    aggregate = [None] * len(sdl)
    preds = [None] * len(sdl)                          # filled with return_pred only
    with torch.no_grad():
        # the matrices at the current parameters, once for every group and step
        step_kw = fit_sys.objective_matrices()
    # 2. Free run per group, then RMS per record.
    for ixs in groups.values():
        uh = T(np.stack([un[i][k0 - nb:k0] for i in ixs]))
        yh = T(np.stack([yn[i][k0 - na:k0] for i in ixs]))
        with torch.no_grad():
            # x_hat^n_{k0} = psi(chi_{k0}); the scored samples are k0 .. N - 1
            x0 = fit_sys.encoder(uh, yh)
            u_t = T(np.stack([un[i][k0:] for i in ixs]))
            y_t = T(np.stack([yn[i][k0:] for i in ixs]))
            y_pred, _, _ = closed_loop_rollout(
                closed_loop_step(fit_sys.hfn, bank), u_t, y_t, x0, bank,
                torch.tensor([int(ctrl_rows[i]) for i in ixs], dtype=torch.long, device=device), **step_kw)
        y_pred = y_pred.cpu().numpy()
        for j, i in enumerate(ixs):
            # y_hat = y_hat^n sigma_y + mu_y; RMS_j = (mean_k e_j,k^2)^(1/2), record RMS = (mean_j RMS_j^2)^(1/2).
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
