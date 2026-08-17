"""Closed-loop free run, the sim-RMS that selects checkpoints, and the two baselines.

WHAT THIS FIXES
---------------
Variant B optimised a closed-loop objective and was then scored and selected on an OPEN-loop free
run, so the model was never asked to be good at what it was measured on. That run is invalid
rather than a negative result. The fix is not "remember to close the loop in validation too": it
is that training, validation and checkpoint selection all call the SAME rollout
(`cl_controller.rollout`), so they cannot disagree by construction.

`fit_sys.cal_validation_error(val_sys_data, validation_measure=...)` is deepSI's single selection
hook (`fit_system.py:199`); its return value is what `fit()` minimises over epochs. Replacing it
is therefore sufficient to move SELECTION, not just reporting. Note it receives the
UN-normalised data and normalises internally via `apply_experiment`, so this module normalises
explicitly with `fs.norm`.

THE BASELINES MUST MOVE TOO
---------------------------
If the model is scored closed loop and the baselines are not, the comparison is meaningless and
the variant A/B mismatch reappears in a new dress. Both D-072 baselines are re-derived here on the
same rollout:

  true-x0 baseline        FP block alone, seeded with the true state, closed loop.
  encoder-init baseline   FP block alone, seeded by the encoder from the first measured I/O
                          window, closed loop.

FROZEN ENCODER, and why (extends D-072)
---------------------------------------
D-072 deliberately uses the UNTRAINED encoder for the encoder-init baseline: before training it is
purely baseline-derived, so "baseline plus linear init" is well defined, whereas the trained
encoder is co-trained with the augmented dynamics and would not be. That argument binds harder
once the metric is computed every validation step: if the baseline used the LIVE encoder it would
drift as training proceeds, and the reference the model is judged against would move underneath
it. `snapshot_encoder` takes a deepcopy before training and the baseline uses that forever.

SELF-CONSISTENCY GATE
---------------------
At initialisation the ANN output is exactly zero, so the augmented model IS the baseline. The
closed-loop model score and the closed-loop encoder-init baseline score must therefore be
EQUAL at init, to numerical precision. Any difference is a bug in one of the two paths, and it is
checked rather than assumed (`cl_gate_validation.py`).
"""
__project_origin__ = "added"

import copy

import numpy as np
import torch

from cl_controller import rollout, open_loop_rollout


def encoder_window(un, yn, k0, na, nb, na_right=0, nb_right=0, dtype=torch.float32):
    """(uhist, yhist) for the encoder at sample k0, matching deepSI's to_hist_future_data slicing.

    uhist = u[k0-nb : k0+nb_right], yhist = y[k0-na : k0+na_right]  (system_data.py:276-300)
    """
    uh = un[k0 - nb:k0 + nb_right]
    yh = yn[k0 - na:k0 + na_right]
    return (torch.as_tensor(np.ascontiguousarray(uh[None]), dtype=dtype),
            torch.as_tensor(np.ascontiguousarray(yh[None]), dtype=dtype))


def encoder_x0(encoder, un, yn, k0, na, nb, na_right=0, nb_right=0, dtype=torch.float32):
    """Encoder-estimated normalised state at sample k0."""
    uh, yh = encoder_window(un, yn, k0, na, nb, na_right, nb_right, dtype)
    with torch.no_grad():
        return encoder(uh, yh)


def snapshot_encoder(obj):
    """Frozen deepcopy of the encoder, for the encoder-init baseline (see FROZEN ENCODER).

    Accepts either a fit system (uses `.encoder`) or an encoder module directly, so that
    re-snapshotting an existing snapshot is not an error.
    """
    enc = copy.deepcopy(obj.encoder if hasattr(obj, 'encoder') else obj)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def free_run(step_fn, out_fn, un, yn, x0, bank=None, ctrl=None, k0=0, closed=True):
    """One free run over a record from sample k0. Returns normalised y_pred, (N-k0, ny).

    closed=True closes Cfb around the model via the shared rollout; closed=False is the
    open-loop run, kept so the two can be reported side by side on identical footing.
    """
    u_t = torch.as_tensor(np.ascontiguousarray(un[None, k0:]), dtype=x0.dtype)
    with torch.no_grad():
        if not closed:
            y, _ = open_loop_rollout(step_fn, out_fn, u_t, x0)
        else:
            y_t = torch.as_tensor(np.ascontiguousarray(yn[None, k0:]), dtype=x0.dtype)
            y, _, _ = rollout(step_fn, out_fn, u_t, y_t, x0, bank, ctrl)
    return y[0].cpu().numpy()


def rms_phys(y_pred_norm, y_ref_phys, ystd, y0, avg_from=0):
    """Per-channel rms error [m] and the aggregate deepSI uses for selection.

    # THEORY: deepSI System_data.RMS is sqrt(mean squared error over all samples AND channels),
    # so the aggregate is the quadratic mean of the per-channel rms, not their arithmetic mean.
    """
    y_pred = np.asarray(y_pred_norm) * np.asarray(ystd) + np.asarray(y0)
    e = y_pred[avg_from:] - np.asarray(y_ref_phys)[avg_from:]
    per_ch = np.sqrt(np.mean(e ** 2, axis=0))
    return per_ch, float(np.sqrt(np.mean(per_ch ** 2)))


class ClosedLoopValidator:
    """Drop-in replacement for `fit_sys.cal_validation_error`: selection on the CLOSED-loop free run.

    Install with `install(...)`; the returned original can be restored. The scalar returned is the
    quadratic mean over records of the closed-loop aggregate rms [m], so `fit()` selects on it.

    INCOMPATIBLE with `training._install_nf_val_probe` as a selector: that probe wraps
    `cal_validation_error` and returns the ORIGINAL (open-loop) value, so whichever is installed
    last decides selection. If both are wanted, this one must be outermost and must call the probe
    for its side effects while returning its own value; `probe` does exactly that.
    """

    def __init__(self, fs, bank, step_fn, out_fn, record_names, k0, dims, dtype=torch.float32,
                 probe=None, verbose=True):
        self.fs, self.bank = fs, bank
        self.step_fn, self.out_fn = step_fn, out_fn
        self.record_names = list(record_names)
        self.k0 = k0
        self.na, self.nb, self.na_right, self.nb_right = dims
        self.dtype = dtype
        self.probe = probe
        self.verbose = verbose
        self.history = []

    def _records(self, val_sys_data):
        return val_sys_data.sdl if hasattr(val_sys_data, 'sdl') else [val_sys_data]

    def __call__(self, val_sys_data, validation_measure='sim-RMS'):
        if self.probe is not None:                      # side effects only, value discarded
            try:
                self.probe(val_sys_data, validation_measure=validation_measure)
            except Exception as e:
                print('    [cl-val] probe failed (non-fatal): %s' % e)
        norm = self.fs.norm
        per_record = []
        for i, sd in enumerate(self._records(val_sys_data)):
            un = ((sd.u - norm.u0) / norm.ustd).astype(np.float32 if self.dtype == torch.float32
                                                       else np.float64)
            yn = ((sd.y - norm.y0) / norm.ystd).astype(un.dtype)
            x0 = encoder_x0(self.fs.encoder, un, yn, self.k0, self.na, self.nb,
                            self.na_right, self.nb_right, self.dtype)
            ctrl = self.bank.gather(torch.tensor([i], dtype=torch.long))
            y_pred = free_run(self.step_fn, self.out_fn, un, yn, x0, self.bank, ctrl,
                              k0=self.k0, closed=True)
            _, agg = rms_phys(y_pred, sd.y[self.k0:], norm.ystd, norm.y0)
            per_record.append(agg)
        score = float(np.sqrt(np.mean(np.asarray(per_record) ** 2)))
        self.history.append((score, per_record))
        if self.verbose:
            print('    [cl-val] closed-loop sim-RMS %.6e m   per record [%s]'
                  % (score, ' '.join('%.3e' % v for v in per_record)))
        return score


def install(fs, validator):
    """Replace `fs.cal_validation_error`; returns the original for restoration."""
    orig = fs.cal_validation_error
    fs.cal_validation_error = validator
    return orig
