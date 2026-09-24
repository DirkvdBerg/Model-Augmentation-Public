"""This folder's closed-loop scorer: the harness the black box is measured with outside fit().

`closed_loop_score` is a thin caller of the framework's `closed_loop_free_run_rms_batch` with a
bank built from record NAMES (vendored `build_controller_bank`), aggregated exactly as
`ClosedLoopSimulator.validation_error` does. G1(c) proves it reproduces the grey box's own
selection number; G2 and G3 then use it.

`closed_loop_traces` returns the free-run trajectories for the stability rule (BB-007). It repeats
the scoring chain step for step and ASSERTS that its RMS equals the harness's, so it cannot become
a second, drifting implementation of the number.
"""
import numpy as np
import torch

from model_augmentation.fit_systems.closed_loop import (closed_loop_free_run_rms_batch,
                                                        closed_loop_rollout)

# HEURISTIC: traces run batch-of-one, the harness batches records of equal length; float32 BLAS
# reassociation then differs, and an OPEN-loop free run through integrators amplifies it: measured
# 4.3e-05 relative on T10 at 800 Hz (runs g2_n4sid, g2_frf, attempt 1). The trace only feeds the
# stability flag; every reported RMS is the harness's.
TRACE_TOL = 1e-3
STAB_FACTOR = 10.0     # HEURISTIC (BB-007): bounded misfit versus divergence on a free integrator


def bank_for(names, ts, ystd, std_u, dtype=torch.float32):
    from gantry_dynamic.controller import build_controller_bank
    return build_controller_bank(list(names), ts, ystd=ystd, std_u=std_u, dtype=dtype)


def closed_loop_score(fit_sys, names, sdl, ts, dtype=torch.float32, bank=None, rows=None):
    """(score, per_record, per_channel): N-sample weighted mean of per-record free-run RMS [m]."""
    if bank is None:
        bank, rows, _ = bank_for(names, ts, fit_sys.norm.ystd, fit_sys.norm.ustd, dtype)
    per_channel, per_record = closed_loop_free_run_rms_batch(fit_sys, list(sdl), bank, rows)
    # THEORY: deepSI System_data_list.RMS weighting, as ClosedLoopSimulator.validation_error.
    score = float(np.average(np.asarray(per_record), weights=[len(sd.u) for sd in sdl]))
    return score, per_record, per_channel, bank, rows


def closed_loop_traces(fit_sys, sdl, bank, rows, per_record_ref=None):
    """Per record: (y_model_phys, y_data_phys) over the scored samples, and a stability verdict."""
    norm = fit_sys.norm
    na, nb = fit_sys.na, fit_sys.nb
    na_r, nb_r = getattr(fit_sys, 'na_right', 0), getattr(fit_sys, 'nb_right', 0)
    k0 = max(na, nb)
    p = next(fit_sys.hfn.parameters())
    rv = lambda a: np.asarray(a).ravel()                                     # noqa: E731
    T = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=p.dtype)   # noqa: E731
    npdt = np.float64 if p.dtype == torch.float64 else np.float32
    out = []
    for i, sd in enumerate(sdl):
        un = ((sd.u - rv(norm.u0)) / rv(norm.ustd)).astype(npdt)
        yn = ((sd.y - rv(norm.y0)) / rv(norm.ystd)).astype(npdt)
        with torch.no_grad():
            x0 = fit_sys.encoder(T(un[None, k0 - nb:k0 + nb_r]), T(yn[None, k0 - na:k0 + na_r]))
            yp, _, _ = closed_loop_rollout(fit_sys.hfn, fit_sys.hfn.output_only,
                                           T(un[None, k0:]), T(yn[None, k0:]), x0, bank,
                                           torch.tensor([int(rows[i])], dtype=torch.long))
        y_model = yp[0].numpy() * rv(norm.ystd) + rv(norm.y0)
        y_data = np.asarray(sd.y)[k0:]
        e = y_model - y_data
        rms = float(np.sqrt(np.mean(np.mean(e ** 2, axis=0))))
        if per_record_ref is not None:
            ref = float(per_record_ref[i])
            same = (np.isfinite(ref) == np.isfinite(rms)) and \
                (not np.isfinite(ref) or abs(rms - ref) <= TRACE_TOL * max(abs(ref), 1e-30))
            if not same:
                raise RuntimeError('trace RMS %.6e disagrees with the harness %.6e on record %d'
                                   % (rms, ref, i))
        finite = bool(np.all(np.isfinite(y_model)))
        emax = float(np.nanmax(np.abs(e))) if finite else float('inf')
        ymax = float(np.max(np.abs(y_data)))
        stable = finite and emax <= STAB_FACTOR * ymax
        out.append(dict(rms=rms, finite=finite, max_abs_err=emax, max_abs_y=ymax, stable=stable,
                        y_model=y_model, y_data=y_data))
    return out
