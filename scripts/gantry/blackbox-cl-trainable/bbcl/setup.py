"""Shared setup: the grey box's CFG (read from its entry file, never retyped), overridden only in
the fields BB2-006 / BB2-009 name, plus data and normalisation through the pipeline's own loaders.
Read-only use of the pipeline; nothing outside this folder is written."""
__project_origin__ = "added"

import os
import sys
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                   # blackbox-cl-trainable
REPO = os.path.abspath(os.path.join(ROOT, '..', '..', '..'))
for p in (os.path.join(REPO, 'scripts', 'gantry'), REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

VERSIONS = {'noisy': 'thesis_taf_noisy', 'noisefree': 'thesis_taf_noisefree',
            'lowpass': 'thesis_taf_lowpass_noise'}          # D-239 data set (pipeline mode)


def base_cfg():
    for k in ('THESIS_ARM', 'OBC_ARM', 'DATASET_AB_MODE', 'DATASET_AB_SEED', 'BURN_IN_AB',
              'LBFGS_AB_FREEZE'):
        assert k not in os.environ, '%s set: the entry file would change CFG' % k
    import gantry_interconnect_dynamic as entry
    return entry.CFG


def cfg_for(version, seed, local=True, **over):
    """BB2-006 shared settings; `local` adds the BB2-009 CPU deviations (stride, batch)."""
    c = replace(base_cfg(), mode=VERSIONS[version], seed=seed, use_f64=True, na_nb_override=29,
                nf_seconds=0.1, burn_in=0, closed_loop=True, joint_estimation=False,
                orth=False, obc=False, lr=1e-3, adam_eps=1e-8, stride=10, batch_size=512,
                lbfgs=False)
    if local:
        c = replace(c, device='cpu', compile_mode=None, stride=100, batch_size=64)
    return replace(c, **over)


def load(cfg):
    from gantry_dynamic.data import load_datasets, compute_normalization
    data = load_datasets(cfg)
    norm = compute_normalization(cfg, data)
    return data, norm
