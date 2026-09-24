"""MIGRATION STEP 6: attach the closed loop to a built pipeline. One assignment, no patching.

This is the whole gantry-side wiring, and the point of the migration is that it is this short:

    fs.simulator = build_closed_loop(fs, norm, cfg, train_files, val_files)

following the `orth_penalty` precedent exactly (D7.1/D7.8): a declared class attribute with a
no-op default, set on the instance after construction. What it replaces was
`cl_fitsys.attach(fs, bank, step_fn, out_fn)` creating a fit-system class at runtime with
`type()`, binding it into module globals so `pickle` could find it, plus a separate
`ClosedLoopValidator` monkey-patched onto `cal_validation_error`, plus two captured model handles
that went stale whenever `checkpoint_load_system` replaced `fit_sys.__dict__`.

ONE BANK OVER ALL RECORDS, indexed globally. The controller belongs to a trajectory; train versus
validation is a property of the split and not an axis of the controller. The two banks that used
to exist were an artefact of indexing by position within a per-split list, so index 0 meant T1 in
one context and V1 in the other.

THE BOUNDARY. `build_controller_bank` (cl_controller.py) turns record names into stacked matrices
and an integer row per record. Everything gantry-specific stops there: `model_augmentation/` sees
four tensors and some integers, never `Y_op`, `ruleOfThumb`, or a record name it interprets.
Validation record labels are passed through only so a failed identity assertion can say WHICH
record it was.

Usage:
  from cl_pipeline import build_closed_loop
  fs.simulator = build_closed_loop(fs, norm, cfg, TRAIN_FILES, VAL_FILES, val_data)
"""
__project_origin__ = "added"

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from model_augmentation.fit_systems.closed_loop import ClosedLoopSimulator   # noqa: E402
from cl_controller import build_controller_bank                              # noqa: E402


def build_closed_loop(fs, norm, cfg, *, train_files, val_files, val_data, verbose=True):
    """The ClosedLoopSimulator for this pipeline. Assign it to `fs.simulator`.

    The record arguments are KEYWORD-ONLY, deliberately. `train_files` and `val_files` must be in
    the order their System_data_lists were built, and `val_data` must be the SAME object passed to
    fit(); silently swapping two of them attaches the wrong controller to every record, produces a
    plausible loss, and is the exact failure mode plan 3.5 exists to prevent. Positional order is
    not a thing to get right by remembering, and a call site that reads
    `build_closed_loop(fs, norm, cfg, val_files=..., train_files=...)` is now correct rather than
    silently inverted.

    fs         the built fit system (used only for its dtype; the simulator holds no model handles)
    norm       the pipeline's Norm, for ystd and std_u
    cfg        RunConfig, for ts_new: the controller is stepped at the TRAINING rate, not the
               record rate. Dc is rate dependent (Dc_jj = kappa_j * Cnorm(2/ts)), so a controller
               built at 20 kHz and stepped at 4 kHz would be the wrong operator.
    train_files, val_files   record file names, in the order their System_data_lists were built
    val_data   the validation System_data_list actually passed to fit(), registered so the
               simulator can assert by CONTENT that it is scoring what it thinks it is scoring
    """
    names = [os.path.basename(f).replace('.mat', '') for f in list(train_files) + list(val_files)]
    if len(set(names)) != len(names):
        raise ValueError('the same record appears in both train_files and val_files, or twice in '
                         'one of them: %s. A record belongs to exactly one split, and a duplicate '
                         'here means the row lists below no longer line up with the data.'
                         % [n for n in names if names.count(n) > 1])
    bank, rows, y_ops = build_controller_bank(
        names, cfg.ts_new, ystd=norm.ystd, std_u=norm.std_u, dtype=cfg.dtype_pt)
    n_tr = len(train_files)
    train_rows, val_rows = rows[:n_tr], rows[n_tr:]

    sdl = val_data.sdl if hasattr(val_data, 'sdl') else [val_data]
    if len(sdl) != len(val_files):
        raise ValueError('val_data has %d records but %d val_files were given; the simulator '
                         'would register the wrong controller against each record. This is the '
                         'check that catches val_data and the file list having been built from '
                         'different splits.' % (len(sdl), len(val_files)))
    val_records = [(names[n_tr + i], sd, val_rows[i]) for i, sd in enumerate(sdl)]

    if verbose:
        print('[closed loop] one bank over %d records, %d distinct controllers, Y_op %s'
              % (len(names), bank.n_controllers, y_ops))
        print('[closed loop] stepped at ts = %g s (%d Hz), nc = %d states'
              % (cfg.ts_new, round(1 / cfg.ts_new), bank.nc))
        print('[closed loop] diag(Dc) physical [%s] N/m'
              % ' '.join('%.4e' % v for v in bank.physical_D()[0].diagonal()))
    return ClosedLoopSimulator(bank, train_rows, val_records)
