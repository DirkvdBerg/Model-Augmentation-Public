"""Closed-loop model simulation with a known controller, following Kessels: the controller bank, the rollout and
the simulator of training and validation."""
__project_origin__ = "added"

from contextlib import contextmanager

import numpy as np
import torch


class ControllerBank(torch.nn.Module):
    """The distinct controllers of a data set in normalised coordinates, gathered per window.

    # THEORY: the controller seen from the normalised model: B^n = B diag(sigma_y), C^n = diag(1/sigma_u) C,
    # D^n = diag(1/sigma_u) D diag(sigma_y), applied once here.
    """

    def __init__(self, A, B, C, D, ystd, std_u, dtype=torch.float32):
        """A (K,nc,nc), B (K,nc,ny), C (K,nu,nc), D (K,nu,ny), all physical; ystd (ny,), std_u (nu,)."""
        super().__init__()
        T = lambda M: torch.as_tensor(np.asarray(M, dtype=float), dtype=dtype)
        A, B, C, D = T(A), T(B), T(C), T(D)
        ystd_t, stdu_t = T(np.asarray(ystd).ravel()), T(np.asarray(std_u).ravel())
        self.nc, self.nu = A.shape[-1], C.shape[-2]

        Bn = B * ystd_t[None, None, :]
        Cn = C / stdu_t[None, :, None]
        Dn = D / stdu_t[None, :, None] * ystd_t[None, None, :]
        # one step is one product: (u_fb^n, xc_next) = [C^n; A] xc + [D^n; B^n] e^n
        self.register_buffer('M_state', torch.cat([Cn, A], dim=1))
        self.register_buffer('M_error', torch.cat([Dn, Bn], dim=1))

    @property
    def n_controllers(self):
        return self.M_state.shape[0]

    def step(self, xc, e_norm, ctrl):
        """One batched controller step: u_fb^n = C^n xc + D^n e^n, xc_next = A xc + B^n e^n."""
        # THEORY: the controller is a separate state equation beside the model, not part of the model state
        M_state, M_error = ctrl
        out = torch.baddbmm(torch.bmm(M_error, e_norm.unsqueeze(-1)),
                            M_state, xc.unsqueeze(-1)).squeeze(-1)
        return out[:, :self.nu], out[:, self.nu:]


def closed_loop_rollout(hfn, output_only, u_data, y_data, x0, bank, ctrl_ix,
                        obc_pass=None, pass_mats=None):
    """The closed-loop rollout of training, validation and selection; returns (y_hat^n, x final, xc final)."""
    if bank.M_state.device != u_data.device:
        bank.to(u_data.device)
    ctrl = (bank.M_state[ctrl_ix], bank.M_error[ctrl_ix])

    # HEURISTIC: reset the controller-state difference xc at each window start, dropping the effect of earlier
    # model errors; the recorded controller output remains in u_data
    xc = torch.zeros(u_data.shape[0], bank.nc, dtype=u_data.dtype, device=u_data.device)

    x, ys = x0, []
    for u_t, y_t in zip(u_data.unbind(1), y_data.unbind(1)):
        # THEORY: the controller sees the model output; in this residual form e^n = y^n - y_hat^n and the model
        # input is u_hat^n = u^n + u_fb^n
        y_model = output_only(x)
        ys.append(y_model)
        u_fb, xc = bank.step(xc, y_t - y_model, ctrl)
        _, x = hfn(x, u_t + u_fb, obc_pass, pass_mats)
    return torch.stack(ys, dim=1), x, xc


class ClosedLoopSimulator:
    """The closed loop of training and validation; assigned to fit_sys.simulator."""

    def __init__(self, bank, train_ctrl_rows, val_rows, compiled, train_weights, val_sigma_e):
        self.bank = bank
        # loss() receives these per-window arrays by name
        self.extra_array_names = ('ctrl_ix', 'loss_w')
        # sigma_e per validation record and axis (n_val, ny), the denominator of sim-NRMSe
        self.val_sigma_e = np.asarray(val_sigma_e, np.float64)
        # (hfn, output_only) compiled by torch.compile, or None for eager
        self._compiled = compiled
        self.train_ctrl_rows = [int(r) for r in train_ctrl_rows]
        # W_r per training record, one (ny,) row each, appended to every window as loss_w
        self.record_weights = [np.asarray(w) for w in train_weights]
        self.val_rows = [int(r) for r in val_rows]

    def __getstate__(self):
        """The state without the compiled pair, a cache that cannot be pickled."""
        state = self.__dict__.copy()
        state['_compiled'] = None
        return state

    @contextmanager
    def eager(self):
        """Run through the uncompiled pair inside the block, then restore."""
        saved, self._compiled = self._compiled, None
        try:
            yield
        finally:
            self._compiled = saved

    def __call__(self, fit_sys, x, ufuture, yfuture, ctrl_ix, obc_pass=None, pass_mats=None, **kw):
        """(y_hat^n, x final) of the closed-loop rollout of a batch of windows from x."""
        # use the compiled pair for gradient-enabled CUDA rollouts, including the polish
        use_compiled = (self._compiled is not None and ufuture.is_cuda and torch.is_grad_enabled())
        if use_compiled:
            hfn, out_fn = self._compiled
            torch.compiler.cudagraph_mark_step_begin()
        else:
            hfn, out_fn = fit_sys.hfn, fit_sys.hfn.output_only

        y_pred, x_final, _ = closed_loop_rollout(hfn, out_fn,
                                                 ufuture, yfuture, x, self.bank, ctrl_ix.long(),
                                                 obc_pass=obc_pass, pass_mats=pass_mats)
        if use_compiled:
            # copies out of the CUDA graph's buffers, which the next step overwrites
            y_pred, x_final = y_pred.clone(), x_final.clone()
        return y_pred, x_final

    def augment_training_data(self, data, sys_data, fit_sys, **kw):
        """Check window alignment and append each window's controller row and loss weights."""
        nf = kw.get('nf', 25)
        stride = kw.get('stride', 1)
        sdl = sys_data.sdl
        k0 = max(fit_sys.na, fit_sys.nb)
        counts = [len(range(k0 + nf, len(sd.u) + 1, stride)) for sd in sdl]
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
        """Dimensionless sim-NRMSe: mean over records and axes of the closed-loop RMS divided by sigma_e."""
        sdl = val_sys_data.sdl
        per_channel, per_record = closed_loop_free_run_rms_batch(fit_sys, sdl, self.bank, self.val_rows)
        self.last_per_channel = np.asarray(per_channel)          # (n_val, ny), metres
        # THEORY: deepSI's multi-record sim-RMS, the mean of the per-record RMS weighted by the full record
        # length (System_data_list.RMS)
        self.last_sim_rms = float(np.average(np.asarray(per_record),
                                             weights=[len(sd.u) for sd in sdl]))
        return float(np.mean(self.last_per_channel / self.val_sigma_e))


def closed_loop_free_run_rms_batch(fit_sys, sdl, bank, ctrl_rows, return_pred=False):
    """Return closed-loop RMS per record and axis and per record, batching equal lengths; optionally physical y_hat."""
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

    un = [((sd.u - rv(norm.u0)) / rv(norm.ustd)).astype(np_dtype) for sd in sdl]
    yn = [((sd.y - rv(norm.y0)) / rv(norm.ystd)).astype(np_dtype) for sd in sdl]

    groups = {}
    for i, u in enumerate(un):
        groups.setdefault(len(u) - k0, []).append(i)

    per_channel = [None] * len(sdl)
    aggregate = [None] * len(sdl)
    preds = [None] * len(sdl)                          # filled with return_pred only
    for ixs in groups.values():
        uh = T(np.stack([un[i][k0 - nb:k0] for i in ixs]))
        yh = T(np.stack([yn[i][k0 - na:k0] for i in ixs]))
        with torch.no_grad():
            # x_hat^n at k0 from the samples before it; the scored samples are k0 .. N-1
            x0 = fit_sys.encoder(uh, yh)
            u_t = T(np.stack([un[i][k0:] for i in ixs]))
            y_t = T(np.stack([yn[i][k0:] for i in ixs]))
            y_pred, _, _ = closed_loop_rollout(
                fit_sys.hfn, fit_sys.hfn.output_only, u_t, y_t, x0, bank,
                torch.tensor([int(ctrl_rows[i]) for i in ixs], dtype=torch.long, device=device))
        y_pred = y_pred.cpu().numpy()
        for j, i in enumerate(ixs):
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
