"""Encoder-window helpers for the diagnostics in this folder.

WHAT THIS FIXED, and where it lives now
---------------------------------------
Variant B optimised a closed-loop objective and was then scored and selected on an OPEN-loop free
run, so the model was never asked to be good at what it was measured on. That run is invalid
rather than a negative result. The fix is not "remember to close the loop in validation too": it
is that training, validation and checkpoint selection all call the SAME rollout, so they cannot
disagree by construction.

MIGRATION step 7: that is now enforced by the framework rather than by this module.
`SSE_Interconnect.cal_validation_error` is a declared seam that delegates to
`ClosedLoopSimulator.validation_error`, and both it and the training rollout call
`model_augmentation.fit_systems.closed_loop.closed_loop_rollout`, of which exactly one definition
exists. `ClosedLoopValidator` and its `install()` are gone with the monkey patch they needed.

What remains here is the encoder-window slicing the diagnostics use directly, which is deepSI
window arithmetic rather than anything closed-loop specific. The free run and its physical-rms
reduction are gone; see the note at the bottom of the file.

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


# MIGRATION step 7 (continued): `free_run` and `rms_phys` are DELETED, not moved. They
# re-implemented the normalise -> encoder-init -> closed-loop rollout -> denormalise -> physical
# rms chain that the framework's validation path also runs. Unifying the ROLLOUT and leaving two
# copies of the SCORING is the same defect one level up: two implementations that agree today and
# drift the first time one of them is touched. The one implementation is now
#
#     model_augmentation.fit_systems.closed_loop.closed_loop_free_run_rms(fit_sys, sd, bank, row)
#         -> (per_channel_rms, aggregate)   both in metres
#
# and `ClosedLoopSimulator.validation_error` is a loop over it plus a quadratic mean over records.
# Callers wanting an OPEN-loop free run for a side-by-side comparison should use
# `fit_sys.simulate(x0, u)` with `fit_sys.simulator = None`, which is the production open-loop
# path rather than a private twin of it.
