"""Launch test of Clean/: a short run on the CPU (2 s records, 4 Adam updates, a 2-iteration polish and, on a PS2
row, the short PS2 opening), then evaluate() on its run folder. It replaces the former --smoke flag of the entry
script (D-284 amendment 9, E2): the short settings enter from here, through RunConfig fields and a patched PS2Spec.

    python -u clean-up-verification/smoke.py [PRESET [NOISE]]      default: the settings below

The run folder is printed last (SMOKE_RUN_DIR <path>), for val_check.py.
"""
import functools
import os
import sys
from datetime import datetime

# Settings
PRESET = 'R6'          # a preset of Clean/gantry_dynamic/config.py PRESETS
NOISE = 'lowpass'      # 'lowpass' or 'noisefree'
# the short run, as RunConfig fields
SHORT_RUN = dict(device='cpu', compile_mode=None, truncate_s=2.0, n_its=4, its_per_val=4, lbfgs_inner_iter=1,
                 lbfgs_max_iter=2, lbfgs_windows=256)
# the short PS2 opening, as PS2Spec fields (the former ps2_smoke settings of training.py)
SHORT_PS2 = dict(eval_every=2, s1_min=0, s1_cap=4, screen_at=-1, screen_max=float('inf'), n_cal=32, n_snap=256,
                 n_snap_cal=64, s2_batch=64, s2_check=50, s2_min=50, s2_cap=100)


def use_short_ps2():
    """Make gantry_dynamic.training build the short PS2 opening (also used by the harness, hcommon.short_ps2)."""
    import gantry_dynamic.training as training
    training.PS2Spec = functools.partial(training.PS2Spec, **SHORT_PS2)


def main(preset=PRESET, noise=NOISE):
    """Train the short run, evaluate it, and return its run folder."""
    import gantry_interconnect_dynamic as entry
    from gantry_dynamic.config import RunConfig, PRESETS, save_dir
    use_short_ps2()
    cfg = RunConfig(**PRESETS[preset], noise=noise, **SHORT_RUN)
    run_id = 'smoke-%s-%s-%s' % (preset, noise, datetime.now().strftime('%Y%m%d_%H%M%S'))
    entry.main(cfg, run_id=run_id, evaluate_run=True)
    run_dir = os.path.join(save_dir(cfg), run_id)
    print('SMOKE_RUN_DIR %s' % run_dir, flush=True)
    return run_dir


if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Clean'))
    main(*sys.argv[1:3])
