"""Closing a known LTI feedback controller around the model during a loss rollout."""
__project_origin__ = "added"

from contextlib import contextmanager

import numpy as np
import torch


class ControllerBank(torch.nn.Module):
    """The distinct controllers for a dataset, held in the MODEL's coordinates, gathered per batch.

    # THEORY: Kessels (2025) Eq. (5.13c), p157: u_hat = S_u(u_FB + u_FF), scaling at the control
    # interface. Applied once here rather than restated at every timestep.
    """

    def __init__(self, A, B, C, D, ystd, std_u, dtype=torch.float32):
        """A (K,nc,nc), B (K,nc,ny), C (K,nu,nc), D (K,nu,ny), all PHYSICAL. ystd (ny,), std_u (nu,)."""
        super().__init__()
        T = lambda M: torch.as_tensor(np.asarray(M, dtype=float), dtype=dtype)
        A, B, C, D = T(A), T(B), T(C), T(D)
        ystd_t, stdu_t = T(np.asarray(ystd).ravel()), T(np.asarray(std_u).ravel())
        self.nc, self.nu, self.ny = A.shape[-1], C.shape[-2], B.shape[-1]

        Bn = B * ystd_t[None, None, :]
        Cn = C / stdu_t[None, :, None]
        Dn = D / stdu_t[None, :, None] * ystd_t[None, None, :]
        self.register_buffer('M_state', torch.cat([Cn, A], dim=1))
        self.register_buffer('M_error', torch.cat([Dn, Bn], dim=1))
        self.register_buffer('ystd', ystd_t)
        self.register_buffer('stdu', stdu_t)

    @property
    def n_controllers(self):
        return self.M_state.shape[0]

    def gather(self, ctrl_ix):
        """The stacked matrices for each element of a batch of controller-row indices."""
        return self.M_state[ctrl_ix], self.M_error[ctrl_ix]

    def zero_state(self, batch, dtype=None, device=None):
        return torch.zeros(batch, self.nc,
                           dtype=self.M_state.dtype if dtype is None else dtype,
                           device=self.M_state.device if device is None else device)

    def step(self, xc, e_norm, ctrl):
        """One controller step, batched, entirely in normalised coordinates.

        # THEORY: Kessels (2025) Eq. (5.13d), p157: the FB controller is a separate state
        # equation constrained alongside the model, not part of the model state vector.
        """
        M_state, M_error = ctrl
        out = torch.baddbmm(torch.bmm(M_error, e_norm.unsqueeze(-1)),
                            M_state, xc.unsqueeze(-1)).squeeze(-1)
        return out[:, :self.nu], out[:, self.nu:]


def _rollout_segment(hfn, output_only, step, ctrl, u_seg, y_seg, x, xc, obc_pass=None,
                     pass_mats=None):
    """@added (D-169). The timestep loop, over ONE contiguous segment of the horizon."""
    ys = []
    for u_t, y_t in zip(u_seg.unbind(1), y_seg.unbind(1)):
        # THEORY: Kessels (2025) Eq. (5.13d), p157: the controller error is formed against the
        # MODEL output, e_hat = r_bar - y_hat. Here in residual form, so it is y_data - y_model.
        y_model = output_only(x)
        ys.append(y_model)
        u_fb, xc = step(xc, y_t - y_model, ctrl)
        if obc_pass is None and pass_mats is None:
            _, x = hfn(x, u_t + u_fb)
        else:
            _, x = hfn(x, u_t + u_fb, obc_pass, pass_mats)
    return torch.stack(ys, dim=1), x, xc


def closed_loop_rollout(hfn, output_only, u_data, y_data, x0, bank, ctrl_ix, xc0=None,
                        obc_pass=None, pass_mats=None):
    """THE closed-loop rollout. Training, validation and checkpoint selection all call this one."""
    # CHANGED (D-169): re-home the bank to the data's device.
    if bank.M_state.device != u_data.device:
        bank.to(u_data.device)
    step = bank.step
    ctrl = bank.gather(ctrl_ix)

    # HEURISTIC: xc = 0 at every window start. NOT Kessels' Remark 5.4 (p157), which
    # reconstructs xc from (y_bar, r_bar, controller) for the lumped-r form where the
    # controller filters y_hat against the reference. In the residual form the controller
    # filters (y_data - y_model), which does not exist before the window opens, so this is
    # the definition of an initial condition rather than an estimate of an unknown. It is
    # also the unique value for which the correction vanishes when the model is exact.
    # Cost: lost integral memory against the validation free run, measured, see
    # scripts/gantry/closed-loop-controller/cl_step5_reset_cost.py.
    xc = bank.zero_state(u_data.shape[0], dtype=u_data.dtype,
                         device=u_data.device) if xc0 is None else xc0

    return _rollout_segment(hfn, output_only, step, ctrl, u_data, y_data, x0, xc,
                            obc_pass, pass_mats)


def window_controller_index(sys_data, record_ctrl_rows, na, nb, nf, na_right, nb_right, stride):
    """The controller row per training window, aligned to how deepSI concatenates records."""
    sdl = sys_data.sdl if hasattr(sys_data, 'sdl') else [sys_data]
    k0, k0_right = max(na, nb), max(nf, na_right, nb_right)
    counts = [len(range(k0 + k0_right, len(sd.u) + 1, stride)) for sd in sdl]
    ctrl_ix = np.concatenate([np.full(c, int(r), dtype=np.int64)
                              for c, r in zip(counts, record_ctrl_rows)])
    return ctrl_ix, counts


class WindowControllerIndex:
    """Which controller row each TRAINING WINDOW belongs to, and the checks that it is right."""

    def __init__(self, record_rows, record_weights=None):
        self.record_rows = [int(r) for r in record_rows]
        # CHANGED (D-256): optional fixed loss weight per record, one (ny,) row each, appended per
        # window as `loss_w` beside `ctrl_ix` from the same counts. None = no weight array.
        self.record_weights = None if record_weights is None else [np.asarray(w) for w in record_weights]
        self.last_counts = None

    def build(self, data, sys_data, fit_sys, **kw):
        """The extra arrays to append to deepSI's four. Returns a list, checked before returning."""
        nf = kw.get('nf', 25)
        stride = kw.get('stride', 1)
        na_r = getattr(fit_sys, 'na_right', 0)
        nb_r = getattr(fit_sys, 'nb_right', 0)
        ctrl_ix, counts = window_controller_index(
            sys_data, self.record_rows, fit_sys.na, fit_sys.nb, nf, na_r, nb_r, stride)
        sdl = sys_data.sdl if hasattr(sys_data, 'sdl') else [sys_data]

        if len(ctrl_ix) != len(data[0]):
            raise RuntimeError(
                'ctrl_ix misalignment: %d windows derived, %d in the training data'
                % (len(ctrl_ix), len(data[0])))
        self.verify_by_content(data, sdl, counts, fit_sys, nf, stride, na_r, nb_r)
        self.last_counts = list(counts)
        # CHANGED (D-256): same record order and counts as ctrl_ix, so the checks above cover it.
        loss_w = np.concatenate([np.tile(w, (c, 1)) for w, c in zip(self.record_weights, counts)])
        return [ctrl_ix, loss_w]

    @staticmethod
    def verify_by_content(data, sdl, counts, fit_sys, nf, stride, na_r, nb_r):
        """Each record's FIRST and LAST window against its raw data at the derived offset."""
        k0 = max(fit_sys.na, fit_sys.nb)
        ufuture = data[2]
        off = 0
        for sd, c in zip(sdl, counts):
            for j in (0, c - 1):
                start = k0 + max(nf, na_r, nb_r) - nf + j * stride
                got = np.asarray(ufuture[off + j])
                want = np.asarray(sd.u[start:start + nf], dtype=got.dtype)
                if not np.array_equal(got, want):
                    raise RuntimeError(
                        'window %d does not match its raw record data at the expected offset '
                        '%d. The per-window controller assignment is built on that offset, so it '
                        'is wrong.' % (off + j, start))
            off += c


class ClosedLoopSimulator:
    """The controller closed around the model. Assigned to `fit_sys.simulator`."""

    def __init__(self, bank, train_ctrl_rows, val_records=(),
                 compiled=None, train_weights=None, val_sigma_e=None):
        self.bank = bank
        # CHANGED (D-256, D-262): `train_weights` (one (ny,) row per training record) makes the loss
        # weighted, so the instance names a second array; `val_sigma_e` ((n_val, ny), servo-error
        # std per validation record and axis) enables the 'sim-NRMSe' selection measure.
        self.extra_array_names = ('ctrl_ix', 'loss_w')
        self.val_sigma_e = None if val_sigma_e is None else np.asarray(val_sigma_e, np.float64)
        self.last_per_channel = None
        self.last_sim_rms = None
        # CHANGED (D-169): (hfn, output_only) compiled with torch.compile, or None for eager.
        self._compiled = compiled
        self.train_ctrl_rows = [int(r) for r in train_ctrl_rows]
        self.indexer = WindowControllerIndex(self.train_ctrl_rows, train_weights)
        self.val_records = [(str(n), sd, int(r)) for n, sd, r in val_records]

    # CHANGED (D-169, job 80706): compiled artefacts are a CACHE, not state, and must not be
    # serialised.
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_compiled'] = None
        return state

    @contextmanager
    def eager(self):
        """Run through the UNCOMPILED pair for the duration, then restore."""
        saved, self._compiled = self._compiled, None
        try:
            yield
        finally:
            self._compiled = saved

    def __call__(self, fit_sys, x, ufuture, yfuture, ctrl_ix=None, obc_pass=None,
                 pass_mats=None, **kw):
        # CHANGED (D-194/D-195): `obc_pass` is the once-per-objective tangent pair; it is threaded
        # to the rollout unchanged and reaches the model step as a tensor-tuple argument.
        # CHANGED (D-169): route to the compiled pair only when it is VALID to do so.
        use_c = (self._compiled is not None and ufuture.is_cuda and torch.is_grad_enabled())
        if use_c:
            hfn, out_fn = self._compiled
            torch.compiler.cudagraph_mark_step_begin()
        else:
            hfn, out_fn = fit_sys.hfn, fit_sys.hfn.output_only

        y_pred, x_final, _ = closed_loop_rollout(hfn, out_fn,
                                                 ufuture, yfuture, x, self.bank, ctrl_ix.long(),
                                                 obc_pass=obc_pass, pass_mats=pass_mats)
        if use_c:
            y_pred, x_final = y_pred.clone(), x_final.clone()
        return y_pred, x_final

    def augment_training_data(self, data, sys_data, fit_sys, **kw):
        """Append whatever the rollout needs per window. Delegated, see WindowControllerIndex."""
        arrays = self.indexer.build(data, sys_data, fit_sys, **kw)
        return list(data) + arrays

    def validation_error(self, fit_sys, val_sys_data, validation_measure='sim-RMS'):
        """Closed-loop free run per record, scored in metres. Same rollout as __call__."""
        # CHANGED (D-256, D-262): 'sim-NRMSe' is the mean over records and axes of the per-channel
        # RMS divided by the servo-error std of that record and axis, from the same rollout.
        sdl = val_sys_data.sdl if hasattr(val_sys_data, 'sdl') else [val_sys_data]
        # CHANGED (D-169): ONE rollout over all records instead of one per record.
        per_channel, per_record = closed_loop_free_run_rms_batch(
            fit_sys, sdl, self.bank, [row for (_, _, row) in self.val_records])
        self.last_per_record = per_record
        self.last_per_channel = np.asarray(per_channel)          # CHANGED (D-267): (n_val, ny), m
        # THEORY: deepSI aggregates a multi-record sim-RMS in System_data_list.RMS as
        # weighted_mean([sd.RMS(sdo) for ...]), i.e. the ARITHMETIC mean of the per-record RMS
        # values weighted by sd.N_samples (system_data.py:705-713). The quadratic mean used here
        # before was justified by System_data.RMS being sqrt over samples AND channels, which is
        # true WITHIN one record but is not how deepSI combines records; it made the closed-loop
        # selection scalar sit above the open-loop one on the same rollout. The weight is the FULL
        # record length, matching N_samples of the simulated record, which includes the cheat_n
        # prefix and so is not the post-k0 length scored in closed_loop_free_run_rms.
        self.last_sim_rms = float(np.average(np.asarray(per_record),
                                             weights=[len(sd.u) for sd in sdl]))
        return float(np.mean(self.last_per_channel / self.val_sigma_e))


def closed_loop_free_run_rms_batch(fit_sys, sdl, bank, ctrl_rows, k0=None, return_pred=False):
    """@added (D-169). N records scored in ONE rollout. Returns (per_channel_list, aggregate_list)."""
    norm = fit_sys.norm
    na, nb = fit_sys.na, fit_sys.nb
    na_r = getattr(fit_sys, 'na_right', 0)
    nb_r = getattr(fit_sys, 'nb_right', 0)
    k0 = max(na, nb) if k0 is None else k0
    _p = next(fit_sys.hfn.parameters())
    dtype = _p.dtype
    # CHANGED (D-169): every tensor built here follows the MODEL's device.
    device = _p.device
    np_dtype = np.float64 if dtype == torch.float64 else np.float32
    rv = lambda a: np.asarray(a).ravel()
    T = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)

    if len(sdl) != len(ctrl_rows):
        raise RuntimeError(
            'closed_loop_free_run_rms_batch got %d records and %d controller rows. They are '
            'positional: a length mismatch means some record would be scored through another '
            "record's controller, which produces a plausible number and no error."
            % (len(sdl), len(ctrl_rows)))

    un = [((sd.u - rv(norm.u0)) / rv(norm.ustd)).astype(np_dtype) for sd in sdl]
    yn = [((sd.y - rv(norm.y0)) / rv(norm.ystd)).astype(np_dtype) for sd in sdl]

    groups = {}
    for i, u in enumerate(un):
        groups.setdefault(len(u) - k0, []).append(i)

    per_channel = [None] * len(sdl)
    aggregate = [None] * len(sdl)
    preds = [None] * len(sdl)                          # CHANGED (D-227): filled only on request
    for ixs in groups.values():
        uh = T(np.stack([un[i][k0 - nb:k0 + nb_r] for i in ixs]))
        yh = T(np.stack([yn[i][k0 - na:k0 + na_r] for i in ixs]))
        with torch.no_grad():
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
