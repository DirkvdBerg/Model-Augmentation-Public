"""MIGRATION STEP 8: the closed-loop training run, on the framework implementation.

Replaces `cl_step6_run.py`, which is kept unmodified as the record of what run 76573 actually did
and no longer imports (it wants `cl_fitsys.attach` and `ClosedLoopValidator`, both deleted in
step 7). The configuration below is the same one, so the result is comparable:

    ann_route_ix = 0..7, nf = 400, stride = 10, lr = 1e-7, batch 256, 12 epochs, full-length
    validation, concurrent_val = False

WHAT CHANGED, and it is only the wiring
---------------------------------------
    before                                          after
    cl_fitsys.attach(fs, bank, step_fn, out_fn)     fs.simulator = build_closed_loop(...)
      creates a fit-system class at runtime           an ordinary class, imported by name
      binds it into module globals for pickle         nothing to bind
    CV.install(fs, validator)                       the cal_validation_error SEAM
      patches cal_validation_error                    delegation, no patch
    _install_nf_val_probe patches it too            a validation_probes entry
      whichever was last decided selection            probes cannot decide selection
    two banks, per-split indices                    one bank, global rows
    ModelStep / AffineOutput capture hfn            resolved from fit_sys at call time

The equivalence of the two was measured before the old one was deleted, on a fixed batch:
loss rel 1.4e-05, gradient 1 - cos 5.3e-09, trajectory 2.8e-07, selection scalar within 1e-10 m
(`cl_test_closed_loop.py`).

WHY THE EXISTING CHECKPOINTS DO NOT LOAD, and why no shim was written
--------------------------------------------------------------------
`FitSys_ClosedLoop_Go1qTA_{best,last}.pth` pickle a class that `attach()` created at runtime, and
that class no longer exists. A compatibility shim would mean keeping the runtime-class machinery
alive purely to read old files, i.e. coding around the structure being removed. Every number
already extracted from them survives in `server-results/step6_result_76573.json` and in this
folder's scripts. Plan section 8.

CONCURRENT VALIDATION IS ON BY DEFAULT (D-146). `cl_sanity.py` recorded that `concurrent_val`
MUST be False for the OLD path: the concurrent branch pickles the fit system into a subprocess,
the monkeypatched `cal_validation_error` did not survive that, and the child validated with
deepSI's default, i.e. the OPEN-loop measure, so selection silently optimised one objective and
chose on another. That failure mode is structurally gone, `simulator` being a declared attribute
holding an importable class, and it was MEASURED rather than argued: the concurrent run returned
the untrained closed-loop scalar to rel 7.13e-07.

Two guards keep it honest, because the failure it replaces was silent:
  * the inline pre-fit validation is compared against UNTRAINED_SEL below;
  * `Loss_val[0]`, which comes back THROUGH the concurrent path, is compared against the same
    number after fit(). That second one is the one that watches the subprocess.
Caveat carried from D-146: the verification ran on Windows, whose start method is spawn, while
the cluster forks. CL_CONCURRENT=0 disables it.

Usage:
  CL_EPOCHS=12 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python -u cl_train.py
  CL_SMOKE=1 ...   a few iterations on truncated data, to prove the path runs end to end
"""
__project_origin__ = "added"

import dataclasses
import io
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
GANTRY = os.path.join(REPO, 'scripts', 'gantry')
for p in (REPO, GANTRY, HERE, os.path.join(GANTRY, 'drift-demo'),
          os.path.join(GANTRY, 'msd-offset')):
    if p not in sys.path:
        sys.path.insert(0, p)

import deepSI                                                             # noqa: E402
import demo_common as dm                                                  # noqa: E402
from demo_common import CFG                                               # noqa: E402
from gantry_dynamic.data import load_traj, TRAIN_FILES, VAL_FILES         # noqa: E402
from cl_pipeline import build_closed_loop                                 # noqa: E402

SMOKE = bool(int(os.environ.get('CL_SMOKE', 0)))
EPOCHS = int(os.environ.get('CL_EPOCHS', 12))
N_ITS = int(os.environ.get('CL_ITS', 0)) or None
# DEFAULT 1e-7, NOT 1e-3. Routing to the K = 0 rows (X/Y: 0,2,3,5) needs a much smaller lr per
# D-101/D-102, and ann_route_ix = 0..7 includes exactly those rows. The first launch of the
# predecessor used 1e-3 and hit a NaN in the training loss at iteration 81.
LR = float(os.environ.get('CL_LR', 1e-7))
STRIDE = int(os.environ.get('CL_STRIDE', 10))
BATCH = int(os.environ.get('CL_BATCH', 256))
CONCURRENT = bool(int(os.environ.get('CL_CONCURRENT', 1)))   # D-146; CL_CONCURRENT=0 to disable
_ipv = os.environ.get('CL_ITS_PER_VAL', 'epoch')
ITS_PER_VAL = _ipv if _ipv == 'epoch' else int(_ipv)
TAG = os.environ.get('CL_TAG') or os.environ.get('SLURM_JOB_ID') or 'local'
OUT = os.path.join(HERE, 'runs', 'cl_train_%s.json' % TAG)
# Created NOW, not at the end: the JSON is written after fit() returns, so a missing directory
# would raise only after the full run and lose hours of training.
os.makedirs(os.path.dirname(OUT), exist_ok=True)
# The untrained closed-loop selection scalar, recorded in references/step1_reference.json. Used
# as a sanity check on the first validation: if selection has silently reverted to the open-loop
# measure, this is what catches it.
UNTRAINED_SEL = 2.186602663362536e-06
t0 = time.time()


def main():
    """The run. Inside a function, under a __main__ guard, because concurrent_val forks.

    MEASURED (plan open item 3): with `concurrent_val=True` and the body at module level, the
    validation subprocess re-imports this module and re-executes EVERYTHING at import time. On
    Windows, whose multiprocessing start method is spawn, that raises

        RuntimeError: An attempt has been made to start a new process before the current process
        has finished its bootstrapping phase ... use the "freeze_support()" line

    and on Linux, where the default is fork, it would silently rebuild the pipeline in the child.
    The fix is structural, not a flag.

    What that run DID establish, which is the thing plan open item 3 actually asks: the child
    printed the untrained closed-loop scalar 2.1866011034e-06 m, matching the recorded value
    exactly. So the simulator survives pickling into the subprocess and the child scores the
    CLOSED loop. The old failure mode, a monkeypatched cal_validation_error not surviving the
    boundary and selection silently reverting to the OPEN-loop measure, is gone by construction:
    `simulator` is a declared attribute holding an importable class.
    """
    # SMOKE rebinds these below, which makes them LOCAL for the whole function and shadows the
    # module-level constants. Without this declaration the non-smoke path raises
    # UnboundLocalError on the first read, and the smoke path does NOT, because it assigns before
    # it reads: the cheap test masks the bug in the expensive one. Caught by running the real
    # configuration, not by the smoke run.
    global EPOCHS, N_ITS, ITS_PER_VAL, BATCH

    print('=' * 96)
    print('CLOSED-LOOP TRAINING on the framework implementation%s' % ('  [SMOKE]' if SMOKE else ''))
    print('=' * 96)
    cfg = dataclasses.replace(CFG, seed=0, ann_route_ix=tuple(range(8)), lr=LR)
    if SMOKE:
        EPOCHS, N_ITS, ITS_PER_VAL, BATCH = 1, 3, 2, 8
    fs, norm, K0, na, nb, na_r, nb_r = dm.build_pipeline(cfg=cfg, verbose=True)
    print('ann_route_ix %s  nf %d  stride %d  lr %.0e  epochs %s  n_its %s  batch %d'
          % (str(cfg.ann_route_ix), cfg.nf, STRIDE, LR, EPOCHS, N_ITS, BATCH))

    train_files, val_files = list(TRAIN_FILES), list(VAL_FILES)
    train_list = [load_traj(f, cfg) for f in train_files]
    val_list = [load_traj(f, cfg) for f in val_files]
    if SMOKE:
        # Truncate rather than subsample: the window arithmetic and the controller assignment are what
        # the smoke test is proving, and both depend on record LENGTH, not on record content.
        def _cut(sd, n=4000):
            return deepSI.System_data(u=np.array(sd.u[:n]), y=np.array(sd.y[:n]), dt=sd.dt)
        train_list = [_cut(sd) for sd in train_list[:3]]
        val_list = [_cut(sd) for sd in val_list[:2]]
        train_files, val_files = train_files[:3], val_files[:2]
    train_data = deepSI.System_data_list(train_list)
    val_data = deepSI.System_data_list(val_list)

    # ---- THE WIRING. One assignment. -------------------------------------------------------------
    fs.simulator = build_closed_loop(fs, norm, cfg, train_files=train_files, val_files=val_files,
                                    val_data=val_data)

    got = [g['lr'] for g in fs.optimizer.param_groups]
    assert all(abs(v - LR) < 1e-15 for v in got), (
        'optimizer lr %s does not match cfg.lr %g; build_model did not pass it through, and setting '
        'it here instead would repeat the D-101 mistake' % (got, LR))
    torch.save(fs.__dict__, io.BytesIO())      # pre-flight: the simulator must pickle
    print('pre-flight: picklable with the simulator attached   [%.0fs]' % (time.time() - t0),
          flush=True)

    base = fs.cal_validation_error(val_data, validation_measure='sim-RMS')
    print('\nUNTRAINED closed-loop sim-RMS: %.10e m' % base)
    if not SMOKE:
        drift = abs(base - UNTRAINED_SEL) / UNTRAINED_SEL
        print('   against the recorded untrained scalar %.10e m: rel %.3e  %s'
              % (UNTRAINED_SEL, drift,
                 'as expected' if drift < 1e-3 else
                 'UNEXPECTED: selection is not measuring what it measured before'))
    print('-' * 96, flush=True)

    fs.fit(train_data, val_data, epochs=EPOCHS, n_its=N_ITS, batch_size=BATCH,
           loss_kwargs=dict(nf=cfg.nf, stride=STRIDE), auto_fit_norm=False,
           validation_measure='sim-RMS', its_per_val=ITS_PER_VAL,
           concurrent_val=CONCURRENT, verbose=2)
    print('-' * 96, flush=True)

    # fit() ends with checkpoint_load_system('_best'), which REPLACES fs.__dict__. The simulator holds
    # no model handles, so it survives that intact; this is the trap that made the old implementation
    # rebuild its whole eval stack here.
    assert fs.simulator is not None, 'the simulator did not survive the best-checkpoint reload'
    # D-146 guard on the CONCURRENT path specifically. Loss_val[0] is deepSI's initial validation,
    # which under concurrent_val is computed in the SUBPROCESS. If the child had scored the open
    # loop, as the old monkeypatched path silently did, this is where it shows: an open-loop free
    # run on these records is orders away from the closed-loop value, not a few ulp.
    if CONCURRENT and len(vl_check := np.asarray(fs.Loss_val, dtype=float)) and not SMOKE:
        d_conc = abs(float(vl_check[0]) - UNTRAINED_SEL) / UNTRAINED_SEL
        print('concurrent-path check: Loss_val[0] = %.10e m against %.10e m, rel %.3e  %s'
              % (vl_check[0], UNTRAINED_SEL, d_conc,
                 'closed loop confirmed' if d_conc < 1e-3 else
                 'FAILED: the subprocess is not scoring the closed loop'))
    final = fs.cal_validation_error(val_data, validation_measure='sim-RMS')
    per_record = list(fs.simulator.last_per_record)
    imp = 100.0 * (base - final) / base
    vl = np.asarray(fs.Loss_val, dtype=float)

    print('\n' + '=' * 96)
    print('RESULT')
    print('=' * 96)
    print('untrained %.10e m   trained %.10e m   %+.4f %%' % (base, final, imp))
    print('val series (%d points): %s' % (len(vl), ' '.join('%.4e' % v for v in vl)))
    for nm, v in zip(val_files, per_record):
        print('  %-24s %.6e m' % (nm, v))
    if len(vl) > 1:
        print('best at validation %d of %d   %s'
              % (int(np.nanargmin(vl)), len(vl),
                 'SELECTION RETURNED ITERATION 0' if int(np.nanargmin(vl)) == 0
                 else 'selection picked a trained checkpoint'))
    json.dump(dict(base=base, final=final, imp=imp, epochs=EPOCHS, n_its=N_ITS, lr=LR,
                   stride=STRIDE, batch=BATCH, nf=cfg.nf, smoke=SMOKE,
                   concurrent_val=CONCURRENT, val=vl.tolist(),
                   per_record=dict(names=val_files, final=list(map(float, per_record))),
                   seconds=time.time() - t0),
              open(OUT, 'w'), indent=2)
    print('\nwrote %s   [%.0fs]' % (OUT, time.time() - t0))


if __name__ == '__main__':
    main()
