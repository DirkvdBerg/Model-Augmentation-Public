"""Closing a known LTI feedback controller around the model during a loss rollout.

The formulation is Kessels', not ours, and is attributed at the lines that implement it:

    B.M. Kessels, PhD thesis, TU/e, 2025, Chapter 5 "Extension and augmentation-based model
    structure updating". `literature/augmentation/kessels2025_ai-control.pdf`.
    PAGE OFFSET: PDF page = thesis page + 26.

The one place this deviates from him is the controller initial condition, and it says so as a
contrast rather than a citation; see `closed_loop_rollout`.

WHAT IS IN HERE AND WHAT IS NOT
-------------------------------
This module knows how to step a controller alongside a model. It does NOT know what a gantry is,
what `ruleOfThumb` is, what `Y_op` means, or which records exist. It receives stacked (A, B, C, D)
matrices and an integer row index per window, and that is all. Everything that decides WHICH
controller a trajectory got stays in `scripts/gantry/` (plan 3.9).

THE FORM
--------
Residual form, verified equivalent to driving the loop from the reference and cheaper:

    u_plant[k] = u_data[k] + Cfb * (y_data[k] - y_model[k])

Only `u_total` and `y` are needed, both of which the loader already returns; `r_sim` and `f_sim`
are not used by the training path at all. Verified exact for the NONLINEAR augmented model with
the ANN active: 2.62e-14 m over 100 closed-loop steps in float64, and 1.79e-03 m for the same
comparison in float32, i.e. the gap scales with machine epsilon rather than with the model
(`scripts/gantry/closed-loop-controller/cl_direct_vs_residual.py`).

The condition this rests on is that BOTH loops apply the same operator. They do, because the
machine froze `Cfb` at each record's nominal operating point, which is exogenous. A controller
scheduled on the model's own state would break the subtraction while still being linear.

STEP ORDER, and why it is forced
--------------------------------
The plant has no feedthrough (`D_d = 0`), so `y_model[k]` depends on the state alone and is
computable before the input. The controller IS biproper: Tustin gives `Dc != 0`. So this order is
the only one that closes the loop without an algebraic loop:

    y_model[k] = h(x[k])
    e[k]       = y_data[k] - y_model[k]
    u_fb[k]    = Cc xc[k] + Dc e[k]
    u_cl[k]    = u_data[k] + u_fb[k]
    x[k+1]     = step(x[k], u_cl[k])
    xc[k+1]    = Ac xc[k] + Bc e[k]

`h(x)` comes from `fit_sys.hfn.output_only`, which evaluates the output signal's dependency cone, so
the model is stepped ONCE per timestep. The predecessor implementation called it twice to work
around the ordering and doubled the forward cost of every step; that one-call property is
load-bearing.

UNITS
-----
The model works in normalised coordinates and the controller is physical, m -> N. Rather than
denormalise, filter and renormalise on every timestep, the scalings are folded into the stored
matrices ONCE in `ControllerBank.__init__`, so the bank holds the controller in the model's own
coordinates. See there for why this does not weaken the units gate.
"""
__project_origin__ = "added"

import numpy as np
import torch

# `DiscreteController` appears in the design sketch as "one biproper LTI controller as a batched
# state-space step". It is not written, deliberately: it would be the K = 1 special case of
# `ControllerBank` and would mean two implementations of the same step, which is exactly what the
# design rule about one rollout implementation exists to prevent. A single controller is a bank
# with one row.


class ControllerBank(torch.nn.Module):
    """The distinct controllers for a dataset, held in the MODEL's coordinates, gathered per batch.

    Takes stacked physical matrices, one row per DISTINCT controller (not one per record: several
    records usually share an operating point). Which row a window belongs to arrives as `ctrl_ix`,
    resolved at data-build time by the code that knows the answer.

    Buffers, not parameters: the controller is known exactly and is never trained. Registering
    them as buffers means `.to(device)`, `.float()`/`.double()` and `state_dict()` carry them
    with the model.

    NORMALISATION IS FOLDED IN, ONCE, HERE
    --------------------------------------
    The controller is linear, so the per-timestep sandwich

        e_phys = e_norm * ystd ;  u_phys = C xc + D e_phys ;  u_norm = u_phys / stdu

    is identical to storing

        B' = B diag(ystd) ,  C' = diag(1/stdu) C ,  D' = diag(1/stdu) D diag(ystd)

    and stepping entirely in normalised coordinates. That removes one multiply, one divide and two
    module attribute lookups from every timestep, and it is the cleaner object: one controller in
    one coordinate system instead of a physical controller wrapped in a conversion.

    # THEORY: Kessels (2025) Eq. (5.13c), p157 -- u_hat = S_u(u_FB + u_FF), scaling at the control
    # interface. Applied once here rather than restated at every timestep.

    It does not weaken the units gate, which cannot be replaced by the zero-ANN replay gate: with
    the ANN forced to zero the residual is identically zero and ANY scale error on the controller
    is multiplied by zero, so the replay gate passes regardless. `physical_D()` UNFOLDS the stored
    matrix so the gate checks the physical N/m value, which means it now tests the folding as well.
    Do not keep a second set of physical buffers for the gate: two representations of one object
    is how they drift apart.

    THE MATRICES ARE STACKED
    ------------------------
    `u_fb = C' xc + D' e` and `xc' = A xc + B' e` are four small matmuls, i.e. four dispatches per
    timestep. Stacking `[C'; A]` and `[D'; B']` once here makes a step two matmuls and one add,
    and `torch.baddbmm` fuses the add into the second, so a step is two dispatches. On a batch of
    32 windows a `(32,8)@(8,8)` matmul runs at about 60 MFLOP/s, i.e. the arithmetic is free and
    the dispatch count is the whole cost, which is why this is worth doing and why exploiting the
    block-diagonal structure of the controller (27 nonzeros of 81) is not: that cuts FLOPs and
    leaves the dispatch count alone.
    """

    def __init__(self, A, B, C, D, ystd, std_u, dtype=torch.float32):
        """A (K,nc,nc), B (K,nc,ny), C (K,nu,nc), D (K,nu,ny), all PHYSICAL. ystd (ny,), std_u (nu,).

        ystd  normalised -> physical output scale [m]
        std_u physical -> normalised input scale [N]
        """
        super().__init__()
        T = lambda M: torch.as_tensor(np.asarray(M, dtype=float), dtype=dtype)   # noqa: E731
        A, B, C, D = T(A), T(B), T(C), T(D)
        ystd_t, stdu_t = T(np.asarray(ystd).ravel()), T(np.asarray(std_u).ravel())
        if A.ndim != 3 or A.shape[-1] != A.shape[-2]:
            raise ValueError('A must be (K, nc, nc), got %s' % (tuple(A.shape),))
        self.nc, self.nu, self.ny = A.shape[-1], C.shape[-2], B.shape[-1]
        if ystd_t.numel() != self.ny or stdu_t.numel() != self.nu:
            raise ValueError('ystd must have ny=%d entries and std_u nu=%d, got %d and %d'
                             % (self.ny, self.nu, ystd_t.numel(), stdu_t.numel()))

        Bn = B * ystd_t[None, None, :]                       # B  diag(ystd)
        Cn = C / stdu_t[None, :, None]                       # diag(1/stdu) C
        Dn = D / stdu_t[None, :, None] * ystd_t[None, None, :]   # diag(1/stdu) D diag(ystd)
        # Named slices, assembled once. The state-space structure has to stay readable here
        # because it is not readable at the call site any more.
        self.register_buffer('M_state', torch.cat([Cn, A], dim=1))    # (K, nu+nc, nc)
        self.register_buffer('M_error', torch.cat([Dn, Bn], dim=1))   # (K, nu+nc, ny)
        self.register_buffer('ystd', ystd_t)
        self.register_buffer('stdu', stdu_t)

    @property
    def n_controllers(self):
        return self.M_state.shape[0]

    def physical_D(self):
        """The stored feedthrough, unfolded back to physical N/m. For the units gate only."""
        Dn = self.M_error[:, :self.nu, :]
        return self.stdu[None, :, None] * Dn / self.ystd[None, None, :]

    def gather(self, ctrl_ix):
        """The stacked matrices for each element of a batch of controller-row indices."""
        return self.M_state[ctrl_ix], self.M_error[ctrl_ix]

    def zero_state(self, batch, dtype=None, device=None):
        return torch.zeros(batch, self.nc,
                           dtype=self.M_state.dtype if dtype is None else dtype,
                           device=self.M_state.device if device is None else device)

    def step(self, xc, e_norm, ctrl):
        """One controller step, batched, entirely in normalised coordinates.

        xc      (batch, nc)      controller state
        e_norm  (batch, ny)      y_data - y_model, NORMALISED
        ctrl    (M_state, M_error) as returned by gather(), each (batch, ...)

        Returns (u_fb_norm, xc_next).

        # THEORY: Kessels (2025) Eq. (5.13d), p157 -- the FB controller is a separate state
        # equation constrained alongside the model, not part of the model state vector.
        """
        M_state, M_error = ctrl
        # baddbmm(input, b1, b2) = input + b1 @ b2 in ONE dispatch, so the two matmuls and the add
        # cost two operations rather than five.
        out = torch.baddbmm(torch.bmm(M_error, e_norm.unsqueeze(-1)),
                            M_state, xc.unsqueeze(-1)).squeeze(-1)
        return out[:, :self.nu], out[:, self.nu:]


    def check_units(self, ctrl_ix=0, e_phys=1e-4):
        """Self-check: does the folded representation round-trip to the physical controller?

        A METHOD rather than a free gate function, and that is the point. It asks a question about
        THIS object's own invariant (the normalisation folded into B, C and D in `__init__` is
        recoverable), so it belongs to the object the way a `validate()` does, not to whichever
        script happens to call it. It was briefly moved out to a test file on the grounds that gate
        code does not belong in a production module; that made a live script import from a test,
        which is worse, so it came back as a method.

        Drives the controller with a known PHYSICAL residual for one step from rest, so the output
        is exactly `Dc @ e_phys`, and checks what comes back after the folded renormalisation.

        THIS CANNOT BE REPLACED BY THE ZERO-ANN REPLAY GATE. With the ANN forced to zero the model
        reproduces the record, so the residual is identically zero and ANY scale error on Cfb is
        multiplied by zero: that gate passes regardless of the units being right.

        Returns (u_fb_norm, expected physical N, relative error) per channel.
        """
        ix = torch.tensor([ctrl_ix], dtype=torch.long)
        ctrl = self.gather(ix)
        e = torch.full((1, self.ny), float(e_phys), dtype=self.M_state.dtype)
        u_norm, _ = self.step(self.zero_state(1), e / self.ystd, ctrl)
        expect = self.physical_D()[ctrl_ix] @ e[0]
        got = u_norm[0] * self.stdu
        rel = (got - expect).abs() / expect.abs().clamp_min(1e-30)
        return u_norm[0].detach().numpy(), expect.detach().numpy(), rel.detach().numpy()


def closed_loop_rollout(hfn, output_only, u_data, y_data, x0, bank, ctrl_ix, xc0=None):
    """THE closed-loop rollout. Training, validation and checkpoint selection all call this one.

    hfn          the model step, (x, u) -> (y, x_next), NORMALISED
    output_only  y from the state alone, x -> y; requires D_d = 0, NORMALISED
    u_data       (batch, nf, nu) recorded plant input, NORMALISED
    y_data       (batch, nf, ny) recorded output, NORMALISED
    x0           (batch, nx) initial model state, e.g. from the encoder
    bank         ControllerBank
    ctrl_ix      (batch,) long, the controller row per batch element
    xc0          (batch, nc) or None for zero; see below

    Returns (y_pred, x_final, xc_final), y_pred (batch, nf, ny), NORMALISED.
    """
    # Bound once: these are attribute lookups inside an nf-iteration loop otherwise, and
    # nn.Module.__getattr__ walks _parameters, _buffers and _modules on every hit.
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

    x = x0
    ys = []
    # unbind(1) is ONE dispatch for all nf views; indexing per timestep is nf separate selects.
    for u_t, y_t in zip(u_data.unbind(1), y_data.unbind(1)):
        # THEORY: Kessels (2025) Eq. (5.13d), p157 -- the controller error is formed against the
        # MODEL output, e_hat = r_bar - y_hat. Here in residual form, so it is y_data - y_model.
        y_model = output_only(x)                       # D_d = 0: state only
        ys.append(y_model)
        u_fb, xc = step(xc, y_t - y_model, ctrl)
        _, x = hfn(x, u_t + u_fb)
    return torch.stack(ys, dim=1), x, xc


def window_controller_index(sys_data, record_ctrl_rows, na, nb, nf, na_right, nb_right, stride):
    """The controller row per training window, aligned to how deepSI concatenates records.

    `System_data_list.to_hist_future_data` maps each record through `to_hist_future_data` and
    concatenates the results in list order, so record identity is recoverable from per-record
    window counts. That derivation is the fragile part and it has been wrong once already (an
    off-by-one from the right-hand encoder extension, which would have attached the wrong
    controller to most of the training set and would have looked like a training problem rather
    than a bookkeeping one). Nothing crashes when it happens: the loss still decreases and the
    model is fitted inside the wrong loop.

    The count is NOT guessed. deepSI has two branches and they agree:

      stride == 1  sliding_window_view over u[npast:], npast = max(na, nb), giving
                   `len(u) - npast - nf + 1` windows. This branch IGNORES na_right and nb_right.
      stride != 1  loops `for k in range(k0 + k0_right, len(u) + 1, stride)` with k0 = max(na, nb)
                   and k0_right = max(nf, na_right, nb_right).

    The second reproduces the first at stride 1 whenever nf >= na_right, nb_right, which holds for
    every configuration this pipeline uses, so one formula covers both. It is written as an actual
    `range` so it cannot drift from deepSI's own loop.

    Returns (ctrl_ix, counts). The caller asserts both against the real call; see
    ClosedLoopSimulator.augment_training_data.
    """
    sdl = sys_data.sdl if hasattr(sys_data, 'sdl') else [sys_data]
    if len(record_ctrl_rows) != len(sdl):
        raise RuntimeError(
            'the simulator was given %d per-record controller rows but the training data has %d '
            'records. Those two lists must be the same object in the same order, or every window '
            'after the first mismatch is trained inside the wrong loop.'
            % (len(record_ctrl_rows), len(sdl)))
    k0, k0_right = max(na, nb), max(nf, na_right, nb_right)
    counts = [len(range(k0 + k0_right, len(sd.u) + 1, stride)) for sd in sdl]
    ctrl_ix = np.concatenate([np.full(c, int(r), dtype=np.int64)
                              for c, r in zip(counts, record_ctrl_rows)])
    return ctrl_ix, counts



class WindowControllerIndex:
    """Which controller row each TRAINING WINDOW belongs to, and the checks that it is right.

    Separated from `ClosedLoopSimulator` on purpose. The simulator's job is how the model is
    DRIVEN: it holds the rollout and the validation score, which share one implementation and
    therefore belong together. This is a different job, deepSI window bookkeeping, and it is the
    part that knows about `to_hist_future_data`, strides, `na_right`/`nb_right` and concatenation
    order. Putting it on the simulator made one object answer to two unrelated changes: a new
    driving strategy and a new deepSI data convention.

    It is also the fragile part. The count derivation has been wrong once already, an off-by-one
    from the right-hand encoder extension that would have attached the wrong controller to most of
    the training set. Nothing crashes when that happens: the loss still decreases and the model is
    fitted inside the wrong loop.
    """

    def __init__(self, record_rows):
        self.record_rows = [int(r) for r in record_rows]
        self.last_counts = None

    def build(self, data, sys_data, fit_sys, **kw):
        """The extra arrays to append to deepSI's four. Returns a list, checked before returning.

        deepSI's `My_Simple_DataLoader` slices every array in the list by the same shuffled ids,
        so an index appended here arrives in `loss` correctly shuffled and batched alongside its
        own window.

        Two checks, each failing differently: the total against what arrived, and each record's
        first and last window against its raw data at the derived offset. The second is the one
        that can see an offset error; see below for why the per-record count rebuild that used to
        sit between them is gone.
        """
        nf = kw.get('nf', 25)
        stride = kw.get('stride', 1)
        na_r = getattr(fit_sys, 'na_right', 0)
        nb_r = getattr(fit_sys, 'nb_right', 0)
        ctrl_ix, counts = window_controller_index(
            sys_data, self.record_rows, fit_sys.na, fit_sys.nb, nf, na_r, nb_r, stride)
        sdl = sys_data.sdl if hasattr(sys_data, 'sdl') else [sys_data]

        # 1. THE TOTAL, against what actually arrived. This also covers a change in concatenation.
        #
        # There used to be a per-record count check here too, calling to_hist_future_data once per
        # record purely to COUNT windows, after deepSI had already built them: the entire training
        # set constructed twice at setup, about 700 MB of window copies, to produce fourteen
        # integers. It is gone, and only because check 2 below was shown to stand without it:
        # deliberately corrupting the derived offset by ONE sample makes the content check raise
        # ("content check fired at offset 18"). A count check cannot see an offset error at all,
        # so the cheap check was never the strong one; it was the expensive one.
        if len(ctrl_ix) != len(data[0]):
            raise RuntimeError(
                'ctrl_ix misalignment: %d windows derived, %d in the training data'
                % (len(ctrl_ix), len(data[0])))
        # 2. By CONTENT, not by a re-derivation of deepSI's conventions: the first and last window
        # of each record must be that record's raw data at the derived offset. This is the strong
        # one. It catches a stride change, an off-by-one, or a reordering inside System_data_list,
        # none of which any count check can see, and it costs two array comparisons per record
        # against a full rebuild.
        self.verify_by_content(data, sdl, counts, fit_sys, nf, stride, na_r, nb_r)
        self.last_counts = list(counts)
        return [ctrl_ix]

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
    """The controller closed around the model. Assigned to `fit_sys.simulator`.

    Knows nothing about the loss, the penalties, or which fit-system class it is attached to, so
    `param_loss` and the orthogonality penalty are inherited rather than copied and cannot be
    dropped by accident: they are never mentioned here.

    HOLDS NO MODEL HANDLES. `fit_sys.hfn` and `fit_sys.hfn.output_only` are resolved at CALL
    deepSI's `checkpoint_load_system` does `self.__dict__ = torch.load(file)`, so anything that
    captured `hfn` at attach time points at stale modules after a `_best` reload. Holding only the
    parameter-free bank removes that whole class of bug, and because this is an ordinary
    importable class it is pickled and restored like any other attribute instead of being silently
    dropped the way a patched bound method is.

    Training and validation are two methods on ONE object sharing ONE `closed_loop_rollout`, so
    "training, validation and selection cannot disagree about what the loop is" is enforced by
    construction rather than by discipline.

    Parameters
    ----------
    bank : ControllerBank
        One bank over every record. The controller belongs to a trajectory; train versus
        validation is a property of the split, not an axis of the controller.
    train_ctrl_rows : sequence of int
        One controller row per TRAINING record, in the order the training `System_data_list` was
        built. Resolved by the pipeline, which knows which trajectory got which controller.
    val_records : sequence of (label, sys_data, ctrl_row)
        The ordered validation records. `label` is an opaque string used only in error messages;
        this module never interprets it.
    """

    # The arrays this simulator asks make_training_data to append, in the order it appends them.
    # fit() zips these names onto the extra slices of each batch and passes them to loss() BY NAME,
    # so nothing downstream carries a positional convention it cannot check. If this tuple and
    # augment_training_data disagree, fit() raises rather than letting an array arrive unnamed.
    extra_array_names = ('ctrl_ix',)

    def __init__(self, bank, train_ctrl_rows, val_records=()):
        self.bank = bank
        self.train_ctrl_rows = [int(r) for r in train_ctrl_rows]
        # The deepSI window bookkeeping is a collaborator, not a responsibility of this class.
        # See WindowControllerIndex for why.
        self.indexer = WindowControllerIndex(self.train_ctrl_rows)
        self.val_records = [(str(n), sd, int(r)) for n, sd, r in val_records]
        self._val_checked = False
        self.last_window_counts = None

    # ---- training -----------------------------------------------------------------------
    def __call__(self, fit_sys, x, ufuture, yfuture, ctrl_ix=None, **kw):
        if ctrl_ix is None:
            raise RuntimeError(
                'the closed loop was driven without ctrl_ix. make_training_data must supply it as '
                'the fifth array; without it there is no way to know which controller a window '
                'belongs to, and guessing one would train every window inside the wrong loop.')
        # The seam's contract is (y_pred, x_final). The controller state is dropped here on
        # purpose: carrying it across a boundary is only meaningful for multiple shooting, where
        # whether xc resets at a segment start is an unanswered modelling question, and that
        # combination is refused rather than guessed (see SSE_Interconnect_MultipleShooting).
        y_pred, x_final, _ = closed_loop_rollout(fit_sys.hfn, fit_sys.hfn.output_only,
                                                 ufuture, yfuture, x, self.bank, ctrl_ix.long())
        return y_pred, x_final

    def augment_training_data(self, data, sys_data, fit_sys, **kw):
        """Append whatever the rollout needs per window. Delegated, see WindowControllerIndex."""
        arrays = self.indexer.build(data, sys_data, fit_sys, **kw)
        self.last_window_counts = self.indexer.last_counts
        return list(data) + arrays

    # ---- validation and selection -------------------------------------------------------
    def _check_val_records(self, val_sys_data):
        """Count, position and CONTENT, once, cached. Identity is asserted, never trusted."""
        sdl = val_sys_data.sdl if hasattr(val_sys_data, 'sdl') else [val_sys_data]
        if len(sdl) != len(self.val_records):
            raise RuntimeError(
                'the simulator was registered with %d validation records but %d arrived. Either '
                'the split changed or the wrong list was passed to fit(); scoring would run every '
                'record through some other record\'s controller.'
                % (len(self.val_records), len(sdl)))
        for i, (sd, (label, ref, _)) in enumerate(zip(sdl, self.val_records)):
            n = min(len(sd.u), len(ref.u))
            if not (np.array_equal(np.asarray(sd.u[:n]), np.asarray(ref.u[:n]))
                    and np.array_equal(np.asarray(sd.y[:n]), np.asarray(ref.y[:n]))):
                raise RuntimeError(
                    'validation record %d does not match the one registered as %r. If deepSI '
                    'reordered, or the validation list was rebuilt in a different order, every '
                    'record would be scored through the wrong controller and the selection would '
                    'still look plausible.' % (i, label))
        self._val_checked = True

    def validation_error(self, fit_sys, val_sys_data, validation_measure='sim-RMS'):
        """Closed-loop free run per record, scored in metres. Same rollout as __call__.

        This is deepSI's single selection hook, so replacing it moves SELECTION and not just
        reporting. It exists as a seam rather than as an `apply_experiment` because that interface
        drives the model with u in and y out and cannot carry `y_data`, which the residual form
        needs.

        Only 'sim-RMS' is honoured, and anything else RAISES. The earlier guard here tested
        `startswith('sim')`, which let deepSI's own DEFAULT of 'sim-NRMS' through and then returned
        an RMS in metres: exactly the silent substitution the guard's message claims to prevent,
        and exactly the failure class this whole migration exists to remove. deepSI's other
        measures ('sim-NRMS', 'sim-NRMS_sys_norm', 'X-step-...') normalise or window differently,
        so honouring them means implementing them, not relabelling this one.
        """
        if validation_measure != 'sim-RMS':
            raise ValueError(
                'the closed-loop validator computes a full free run scored in metres, i.e. '
                'sim-RMS, and cannot honour validation_measure=%r. Returning an RMS under another '
                'name would silently change what selection means, which is how a run gets '
                'optimised against one objective and selected on another. Pass '
                "validation_measure='sim-RMS' to fit()." % validation_measure)
        if not self._val_checked:
            self._check_val_records(val_sys_data)
        sdl = val_sys_data.sdl if hasattr(val_sys_data, 'sdl') else [val_sys_data]
        per_record = [closed_loop_free_run_rms(fit_sys, sd, self.bank, row)[1]
                      for sd, (_, _, row) in zip(sdl, self.val_records)]
        self.last_per_record = per_record
        # THEORY: deepSI System_data.RMS is sqrt(mean squared error over all samples AND channels),
        # so the aggregate over records is their quadratic mean, not their arithmetic mean.
        return float(np.sqrt(np.mean(np.asarray(per_record) ** 2)))


def closed_loop_free_run_rms(fit_sys, sys_data, bank, ctrl_row, k0=None):
    """One record: encoder-init, closed-loop free run, error in metres. (per_channel, aggregate).

    The scoring path, in ONE place. `validation_error` is a loop over this plus a quadratic mean,
    and any diagnostic that wants the same number calls this rather than rebuilding the
    normalise -> encoder-window -> rollout -> denormalise -> rms chain. That chain existed twice
    after the rollout itself was unified, which is the same defect one level up: two
    implementations that agree today and drift the first time one of them is touched.

    k0 defaults to max(na, nb), the first sample at which the encoder window is complete.
    """
    norm = fit_sys.norm
    na, nb = fit_sys.na, fit_sys.nb
    na_r = getattr(fit_sys, 'na_right', 0)
    nb_r = getattr(fit_sys, 'nb_right', 0)
    k0 = max(na, nb) if k0 is None else k0
    dtype = next(fit_sys.hfn.parameters()).dtype
    np_dtype = np.float64 if dtype == torch.float64 else np.float32
    rv = lambda a: np.asarray(a).ravel()                                    # noqa: E731

    un = ((sys_data.u - rv(norm.u0)) / rv(norm.ustd)).astype(np_dtype)
    yn = ((sys_data.y - rv(norm.y0)) / rv(norm.ystd)).astype(np_dtype)
    uh = torch.as_tensor(np.ascontiguousarray(un[None, k0 - nb:k0 + nb_r]), dtype=dtype)
    yh = torch.as_tensor(np.ascontiguousarray(yn[None, k0 - na:k0 + na_r]), dtype=dtype)
    with torch.no_grad():
        x0 = fit_sys.encoder(uh, yh)
        u_t = torch.as_tensor(np.ascontiguousarray(un[None, k0:]), dtype=dtype)
        y_t = torch.as_tensor(np.ascontiguousarray(yn[None, k0:]), dtype=dtype)
        y_pred, _, _ = closed_loop_rollout(fit_sys.hfn, fit_sys.hfn.output_only,
                                           u_t, y_t, x0, bank,
                                           torch.tensor([int(ctrl_row)], dtype=torch.long))
    y_phys = y_pred[0].cpu().numpy() * rv(norm.ystd) + rv(norm.y0)
    e = y_phys - np.asarray(sys_data.y)[k0:]
    # The aggregate is formed from the per-channel MSE, not from the per-channel RMS, so that it
    # is `sqrt(mean(mean(e**2)))` exactly as before this was extracted. Taking sqrt per channel and
    # squaring it back would be the same number in exact arithmetic and a different one in
    # floating point, which would cost bit-identity against the recorded selection scalar for no
    # reason at all.
    mse_per_channel = np.mean(e ** 2, axis=0)
    return np.sqrt(mse_per_channel), float(np.sqrt(np.mean(mse_per_channel)))
