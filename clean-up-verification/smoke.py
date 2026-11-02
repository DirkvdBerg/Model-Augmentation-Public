"""Launch test of Clean/: a short run on the CPU (2 s records, 4 Adam updates, a 2-iteration polish and, with the PS
opening, its short version), then evaluate() on its run folder. It replaces the former --smoke flag of the entry
script (D-284 amendment 9, E2): the short settings enter from here, as RunConfig fields over the entry script's
SETTINGS (D-292).

    python -u clean-up-verification/smoke.py [ARM START N_A PS2 [NOISE]]      default: the settings below

The run folder is printed last (SMOKE_RUN_DIR <path>), for val_check.py.
"""
import os
import sys
from datetime import datetime

# Settings
AXES = dict(arm='u', start='correct', n_a=0, ps2=False)   # the model axes (the thesis model M_U at n_a = 0)
NOISE = 'lowpass'      # 'lowpass' or 'noisefree'
# the short run, as RunConfig fields
SHORT_RUN = dict(device='cpu', compile_mode=None, truncate_s=2.0, n_its=4, its_per_val=4, lbfgs_inner_iter=1,
                 lbfgs_max_iter=2, lbfgs_windows=256)
# the short PS opening, as ps2_opening fields (the former ps2_smoke settings of training.py)
SHORT_PS2 = dict(eval_every=2, s1_min=0, s1_cap=4, screen_at=-1, screen_max=float('inf'), n_cal=32, n_snap=256,
                 n_snap_cal=64, s2_batch=64, s2_check=50, s2_min=50, s2_cap=100)


def short_ps2_opening(settings):
    """The ps2_opening of `settings` (the entry script's SETTINGS) with the short values (also used by the harness)."""
    return dict(settings['ps2_opening'], **SHORT_PS2)


def main(axes=AXES, noise=NOISE, **fields):
    """Train the short run, evaluate it, and return its run folder; fields override RunConfig fields (out_root)."""
    import gantry_interconnect_dynamic as entry
    from gantry_dynamic.config import RunConfig, save_dir
    cfg = RunConfig(**dict(entry.SETTINGS, **axes, noise=noise, ps2_opening=short_ps2_opening(entry.SETTINGS),
                           **dict(SHORT_RUN, **fields)))
    run_id = 'smoke-%s-%s-%d-%s-%s-%s' % (cfg.arm, cfg.start, cfg.n_a, 'ps' if cfg.ps2 else 'plain', noise,
                                          datetime.now().strftime('%Y%m%d_%H%M%S'))
    entry.main(cfg, run_id=run_id, evaluate_run=True)
    run_dir = os.path.join(save_dir(cfg), run_id)
    print('SMOKE_RUN_DIR %s' % run_dir, flush=True)
    return run_dir


if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Clean'))
    argv = sys.argv[1:]
    if argv:
        if len(argv) not in (4, 5):
            sys.exit(__doc__)
        a = dict(arm=argv[0], start=argv[1], n_a=int(argv[2]), ps2=argv[3].lower() in ('1', 'true', 'ps'))
        main(a, *argv[4:5])
    else:
        main()
