"""The geometry cache key must change when the reference changes. Including `M_error`.

WHY. `traj_orth._cache_key` originally keyed the checkpoint by BASENAME, so two different
checkpoints sharing a filename reused each other's geometry. That was fixed with content hashes,
and review then found the fix incomplete: the controller hash listed `M_state, M_in, M_out,
M_ff`, three of which do not exist on `ControllerBank` (its buffers are `M_state`, `M_error`,
`ystd`, `stdu`), and an `isinstance(..., Tensor)` filter skipped the missing names in silence.
The hash therefore covered `M_state` alone, and `M_error` -- the entire error-feedback path --
could change the closed-loop predictor without changing the key.

That is the failure this file exists to prevent, and it is the reason the test perturbs
`M_error` specifically rather than "some buffer".

Run: conda run -n GraduationProject python scripts/gantry/orthogonality/testbed/test_cache_provenance.py
"""
__project_origin__ = "added"

import os
import sys
import tempfile

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..', '..', '..'))
for _p in (_REPO, os.path.join(_REPO, 'scripts', 'gantry')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gantry_dynamic import traj_orth as T                              # noqa: E402
from gantry_dynamic.config import RunConfig                            # noqa: E402

PASS, FAIL = [], []


def check(name, ok, detail=''):
    (PASS if ok else FAIL).append(name)
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f'   {detail}' if detail else ''))


class FakeBank(torch.nn.Module):
    """The buffer names the real ControllerBank actually registers."""

    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer('M_state', torch.randn(2, 4, 3, generator=g))
        self.register_buffer('M_error', torch.randn(2, 4, 3, generator=g))
        self.register_buffer('ystd', torch.randn(3, generator=g))
        self.register_buffer('stdu', torch.randn(3, generator=g))


class FakeSim:
    def __init__(self, bank, rows=(0, 1)):
        self.bank = bank
        self.train_ctrl_rows = list(rows)


class FakeNorm:
    def __init__(self, seed=0):
        r = np.random.default_rng(seed)
        for k, shape in (('x_mean', (6, 1)), ('std_x', (6, 1)), ('std_u', (3, 1)),
                         ('u_mean', (3, 1)), ('ystd', (3,)), ('y0', (3,)),
                         ('Cd_norm', (3, 6)), ('Dd_np', (3, 3))):
            setattr(self, k, r.standard_normal(shape))


class FakeEncoder(torch.nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w = torch.nn.Parameter(torch.randn(4, 4, generator=g))


class FakeFitSys:
    def __init__(self, sim, encoder):
        self.simulator = sim
        self.encoder = encoder


class FakeEnv:
    def __init__(self, seed=0, rows=(0, 1)):
        self.norm = FakeNorm(seed)
        self.fit_sys = FakeFitSys(FakeSim(FakeBank(seed), rows), FakeEncoder(seed))


def key_for(env, ckpt, cfg, windows=('T1@0',)):
    return T._cache_key(cfg, ckpt, list(windows), env=env, encoder=env.fit_sys.encoder)[0]


def main():
    cfg = RunConfig(joint_estimation=True)
    tmp = tempfile.mkdtemp()
    ck = os.path.join(tmp, 'ckpt_a')
    with open(ck + '.pt', 'wb') as f:
        f.write(b'checkpoint-A-contents')

    env = FakeEnv()
    base = key_for(env, ck, cfg)

    check('the same reference gives the same key', key_for(FakeEnv(), ck, cfg) == base)

    # ---- THE REGRESSION THIS FILE IS FOR --------------------------------------------
    env2 = FakeEnv()
    with torch.no_grad():
        env2.fit_sys.simulator.bank.M_error.add_(1e-3)
    check('perturbing M_error CHANGES the key', key_for(env2, ck, cfg) != base,
          'the old hash covered M_state only, so this was invisible')

    env3 = FakeEnv()
    with torch.no_grad():
        env3.fit_sys.simulator.bank.M_state.add_(1e-3)
    check('perturbing M_state changes the key', key_for(env3, ck, cfg) != base)

    env3b = FakeEnv()
    with torch.no_grad():
        env3b.fit_sys.simulator.bank.ystd.add_(1e-3)
    check('perturbing ystd changes the key (state_dict covers every buffer)',
          key_for(env3b, ck, cfg) != base)

    # ---- the controller ROW ASSIGNMENT is part of the reference ---------------------
    check('permuting the per-record controller rows changes the key',
          key_for(FakeEnv(rows=(1, 0)), ck, cfg) != base)

    # ---- a renamed/removed buffer must FAIL, not shrink the hash --------------------
    env4 = FakeEnv()
    del env4.fit_sys.simulator.bank._buffers['M_error']
    raised = ''
    try:
        key_for(env4, ck, cfg)
    except RuntimeError as e:
        raised = str(e)
    check('a missing required buffer raises instead of hashing a subset',
          'M_error' in raised, raised[:80])

    # ---- checkpoint CONTENT, not name ------------------------------------------------
    ck2 = os.path.join(tempfile.mkdtemp(), 'ckpt_a')       # same basename, different bytes
    with open(ck2 + '.pt', 'wb') as f:
        f.write(b'checkpoint-B-contents')
    check('a different checkpoint with the SAME basename gives a different key',
          key_for(FakeEnv(), ck2, cfg) != base,
          'this is the original defect: the key used os.path.basename')

    # ---- normalisation and encoder ---------------------------------------------------
    env5 = FakeEnv()
    env5.norm.std_x = env5.norm.std_x + 1e-6
    check('perturbing the normalisation changes the key', key_for(env5, ck, cfg) != base)

    env6 = FakeEnv()
    with torch.no_grad():
        env6.fit_sys.encoder.w.add_(1e-6)
    check('perturbing the encoder state changes the key',
          key_for(env6, ck, cfg) != base)

    # ---- and the load path must REFUSE a mismatched payload --------------------------
    from model_augmentation.fit_systems.trajectory_orth_projection import (
        TrajectoryProjectionGeometry)
    rng = np.random.default_rng(0)
    geom = TrajectoryProjectionGeometry(rng.standard_normal((40, 3)), rank_rtol=1e-12,
                                        strict=False)
    path = os.path.join(tmp, 'geom.npz')
    _, payload = T._cache_key(cfg, ck, ['T1@0'], env=env, encoder=env.fit_sys.encoder)
    T.save_geometry(path, geom, payload, {'windows': ['T1@0']})
    ok = T.load_geometry(path, payload, dtype=torch.float64)
    check('a matching payload loads', ok.rank_S == geom.rank_S)
    _, other = T._cache_key(cfg, ck, ['T1@0'], env=env2, encoder=env2.fit_sys.encoder)
    raised = ''
    try:
        T.load_geometry(path, other, dtype=torch.float64)
    except RuntimeError as e:
        raised = str(e)
    check('a mismatched payload is REFUSED at load, not silently reused',
          'different reference' in raised, raised[:70])

    print('\n' + '=' * 78)
    print(f'{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        print('FAILED: ' + ', '.join(FAIL))
    return 1 if FAIL else 0


if __name__ == '__main__':
    print('=' * 78 + '\ngeometry cache provenance\n' + '=' * 78)
    sys.exit(main())
