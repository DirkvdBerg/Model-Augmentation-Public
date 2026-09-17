"""Prove what `OBC_ARM` actually does to the run configuration (D-192, D-193).

This file does NOT define the arms. The arms are defined in exactly one place, the `OBC_ARM`
override block at the bottom of `scripts/gantry/gantry_interconnect_dynamic.py`, and a second
definition here would drift from it. Instead this imports the entry file under each value of
`OBC_ARM` and prints the field-by-field difference, which is the check `run_dataset_ab.sh`
performed by hand before the dataset pair was launched: the arms are verified by IMPORTING the
entry file, not by reading it.

It asserts the three properties the pair depends on and exits non-zero if any fails:

  1. a submission WITHOUT `OBC_ARM` is unchanged, so every existing runner keeps its behaviour;
  2. the control and tangent arms differ only in `obc`, while the affine arm differs from the
     tangent arm only in `obc_space`;
  3. a misspelled arm is refused rather than silently ignored.

Run it on the server before submitting, and again after any edit to `CFG`:

    python scripts/gantry/orthogonal-by-construction/implementation/08-one-step/experiment_configs.py

THE SOFT-PENALTY ARM. The 2025 soft penalty (`orth=True`) is deliberately NOT reachable through
`OBC_ARM`. `CFG` records that the Gyorok 2025 regularisation has its own entry point at
`orthogonality/Gyorok/train_gyorok.py`, and whether that path and this one build the same basis
and point set has not been checked. Launching `orth=True` from here would risk comparing against
something that is not the published method. Settle that question before adding a third arm.
"""
__project_origin__ = "added"

import importlib
import os
import sys
from dataclasses import fields

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
import gantry_interconnect_dynamic as G                          # noqa: E402
from gantry_dynamic.config import config_json_dict               # noqa: E402

ARMS = ('noproj', 'obc', 'obc_affine')
# What each arm is FOR, in one line. Documentation only; the values come from the entry file.
PURPOSE = {
    'noproj': 'joint estimation of the ten combinations, no projection (the control)',
    'obc':    'the same, with the one-step orthogonal-by-construction subtraction (D-192)',
    'obc_affine': 'the same OBC lifecycle with the frozen transition offset as column eleven',
}


def effective(arm):
    """The RunConfig the entry file resolves to under this `OBC_ARM`. `None` = unset."""
    if arm is None:
        os.environ.pop('OBC_ARM', None)
    else:
        os.environ['OBC_ARM'] = arm
    importlib.reload(G)          # `main()` is guarded, so importing runs no training
    return G.CFG


def diff(a, b):
    return {f.name: (getattr(a, f.name), getattr(b, f.name))
            for f in fields(a) if getattr(a, f.name) != getattr(b, f.name)}


def main():
    ok = True
    base = effective(None)
    cfgs = {arm: effective(arm) for arm in ARMS}
    effective(None)              # leave the environment as we found it

    # The IMPORTED file, not the intended one. A stray second copy of the entry file elsewhere on
    # sys.path is otherwise invisible here: the checks below would report a config that no
    # submission will ever run, which is what happened on job 83749.
    print('entry file  : %s' % os.path.abspath(G.__file__))
    print('dataset %s | seed %d | nf %d | batch %d | lr %g | epochs %d | its_per_val %s'
          % (base.mode, base.seed, base.nf, base.batch_size, base.lr, base.epochs,
             base.its_per_val))
    print('physics     : joint_estimation=%s parameterization=%s param_prior=%s (strength %g)'
          % (cfgs['noproj'].joint_estimation, cfgs['noproj'].physics_parameterization,
             cfgs['noproj'].param_prior, cfgs['noproj'].param_rmse_baseline))
    print('detuning    : %s' % (base.combo_init_detune,))

    print('\n1. a submission without OBC_ARM is unchanged')
    d = diff(base, cfgs['noproj'])
    print('   unset -> noproj differs in: %s' % (d or '{}'))
    if base.obc or base.joint_estimation:
        print('   FAIL: the unset default already carries the experiment (obc=%s, joint=%s)'
              % (base.obc, base.joint_estimation)); ok = False
    else:
        print('   PASS: unset default is obc=False, joint_estimation=False')

    print('\n2. each adjacent comparison changes exactly its intended field')
    d = diff(cfgs['noproj'], cfgs['obc'])
    print('   noproj -> obc differs in: %s' % d)
    if set(d) != {'obc'}:
        print('   FAIL: expected exactly {\'obc\'}'); ok = False
    else:
        print('   PASS')
    d = diff(cfgs['obc'], cfgs['obc_affine'])
    print('   obc -> obc_affine differs in: %s' % d)
    if set(d) != {'obc_space'}:
        print('   FAIL: expected exactly {\'obc_space\'}'); ok = False
    else:
        print('   PASS')

    print('\n3. a misspelled arm is refused')
    try:
        effective('joint_obc')       # the old name from the first draft of this file
        print('   FAIL: an unknown arm was accepted'); ok = False
    except ValueError as e:
        print('   PASS: %s' % str(e).split('\n')[0][:100])
    finally:
        effective(None)

    print('\nrecorded in config.json per arm:')
    for arm in ARMS:
        j = config_json_dict(cfgs[arm])
        print('   OBC_ARM=%-10s OBC=%-5s SPACE=%-7s ORTH=%-5s JOINT_ESTIMATION=%-5s '
              'PARAM_PRIOR=%-5s  %s'
              % (arm, j['OBC'], j['OBC_SPACE'], j['ORTH'], j['JOINT_ESTIMATION'],
                 j['PARAM_PRIOR'], PURPOSE[arm]))

    print('\n%s' % ('ALL CHECKS PASSED; neither arm was launched.' if ok else 'CHECKS FAILED.'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
