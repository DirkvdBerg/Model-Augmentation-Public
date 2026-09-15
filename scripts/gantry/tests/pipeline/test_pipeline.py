"""Gates for the L-BFGS polish phase (D-171).

Three tiers, deliberately:

* Tier 1, PURE, always runs, milliseconds: the acceptance decision, the meter comparison, the
  window-count arithmetic, the config validation. No dataset, no checkpoint, no rollout. This is
  where the acceptance rule gets real coverage, including the joint-estimation and orth
  combinations the local fixture cannot produce.
* Tier 2, synthetic I/O: the checkpoint-format normalisation and the architecture guard.
* Tier 3, fixture-gated, skipped when the fixture is absent: the phase against a real
  closed-loop system restored from job 81262. Only this tier can prove the closure is
  deterministic, the rollback bit-exact, and the chunked gradient exact, because all three are
  properties of the real rollout.

Fixture (present in the repo):
  scripts/gantry/meeting/meeting-07-09-2026/server/checkpoints/
      SSE_Interconnect_Composed_p2LPDA_best.pth      (job 81262, nx_ann=8, 24x3)
  data/gantry/matlab/trajectory/augmentation_ma50_b140-230_a6_z03/

Still NOT covered, stated rather than implied: `orth=True` and `joint_estimation=True` on a REAL
system. The fixture has both off, so `generic_meters` returns {} there. The acceptance logic over
those meters is covered in Tier 1, and the meter COMPUTATION needs a joint run to confirm.
"""

import json
import os
import pathlib
import sys
import tempfile
import unittest
from dataclasses import replace

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts' / 'gantry'))

from model_augmentation.fit_systems.lbfgs_polish import (  # noqa: E402
    FixedBatch, PolishSpec, _eager, _loss_value, _named_params, _quiet_probes, decide,
    lbfgs_polish,
    meter_regression, window_count,
)

FIXTURE = (ROOT / 'scripts' / 'gantry' / 'meeting' / 'meeting-07-09-2026' / 'server'
           / 'checkpoints' / 'SSE_Interconnect_Composed_p2LPDA_best.pth')
DATA = ROOT / 'data' / 'gantry' / 'matlab' / 'trajectory' / 'augmentation_ma50_b140-230_a6_z03'
HAVE_FIXTURE = FIXTURE.exists() and DATA.exists()


# =================================================================================================
# Tier 1: pure
# =================================================================================================

class TestMeterRegression(unittest.TestCase):
    """Lower is better for every meter; only a relative worsening beyond tol is a regression."""

    def test_improvement_is_not_a_regression(self):
        self.assertEqual(meter_regression({'orth_frac': 0.5}, {'orth_frac': 0.4}, 1e-3), [])

    def test_worsening_beyond_tol_is_reported(self):
        bad = meter_regression({'orth_frac': 0.5}, {'orth_frac': 0.6}, 1e-3)
        self.assertEqual(len(bad), 1)
        self.assertIn('orth_frac', bad[0])

    def test_worsening_within_tol_is_tolerated(self):
        self.assertEqual(meter_regression({'param_loss': 1.0}, {'param_loss': 1.0005}, 1e-3), [])

    def test_nan_is_skipped_not_treated_as_regression(self):
        # orth_frac reads NaN while the learned block is still ~0, a legitimate state.
        self.assertEqual(meter_regression({'orth_frac': float('nan')},
                                          {'orth_frac': 0.9}, 1e-3), [])
        self.assertEqual(meter_regression({'orth_frac': 0.1},
                                          {'orth_frac': float('nan')}, 1e-3), [])

    def test_meter_absent_after_is_skipped(self):
        self.assertEqual(meter_regression({'combo_err': 0.1}, {}, 1e-3), [])

    def test_near_zero_before_uses_absolute_scale(self):
        self.assertEqual(len(meter_regression({'V_orth': 0.0}, {'V_orth': 1.0}, 1e-3)), 1)

    def test_every_regressed_meter_is_named(self):
        bad = meter_regression({'a': 1.0, 'b': 1.0, 'c': 1.0}, {'a': 2.0, 'b': 1.0, 'c': 3.0},
                               1e-3)
        self.assertEqual(len(bad), 2)


class TestDecide(unittest.TestCase):
    """The acceptance rule, as a table. This is what protects the thesis contribution.

    Extracted from the phase precisely so these cases cost microseconds instead of a rollout;
    the joint-estimation and orth combinations below cannot be produced by the local fixture.
    """

    OK = dict(loss_finite=True, meter_rel_tol=1e-3)

    def test_improvement_with_flat_meters_is_accepted(self):
        acc, why = decide(1e-5, 9e-6, {'orth_frac': 0.2}, {'orth_frac': 0.2}, **self.OK)
        self.assertTrue(acc, why)

    def test_worse_validation_is_rejected_even_with_better_meters(self):
        acc, why = decide(1e-5, 1.1e-5, {'orth_frac': 0.5}, {'orth_frac': 0.1}, **self.OK)
        self.assertFalse(acc)
        self.assertIn('did not improve', why)

    def test_equal_validation_is_rejected(self):
        """Strictly better, or roll back. An equal fit is not worth a weight change."""
        acc, _ = decide(1e-5, 1e-5, {}, {}, **self.OK)
        self.assertFalse(acc)

    def test_negation_is_rejected_even_though_the_fit_improved(self):
        """THE case this rule exists for: better fit, more negation, drifting parameters."""
        acc, why = decide(1e-5, 8e-6,
                          {'orth_frac': 0.10, 'combo_err': 0.05, 'param_loss': 1e-8},
                          {'orth_frac': 0.40, 'combo_err': 0.31, 'param_loss': 9e-8},
                          **self.OK)
        self.assertFalse(acc)
        self.assertIn('orth_frac', why)
        self.assertIn('combo_err', why)

    def test_non_finite_loss_is_rejected_first(self):
        acc, why = decide(1e-5, 1e-9, {}, {}, loss_finite=False, meter_rel_tol=1e-3)
        self.assertFalse(acc)
        self.assertIn('non-finite', why)

    def test_no_validation_refuses_by_default(self):
        """A caller who forgets val_sys_data must NOT get a silent unconditional accept."""
        acc, why = decide(float('nan'), float('nan'), {}, {}, **self.OK)
        self.assertFalse(acc)
        self.assertIn('accept_without_validation', why)

    def test_no_validation_accepts_only_when_explicitly_allowed(self):
        acc, why = decide(float('nan'), float('nan'), {}, {},
                          accept_without_validation=True, **self.OK)
        self.assertTrue(acc)
        self.assertIn('WITHOUT', why)

    def test_no_validation_still_honours_the_meters(self):
        acc, why = decide(float('nan'), float('nan'), {'orth_frac': 0.1}, {'orth_frac': 0.9},
                          accept_without_validation=True, **self.OK)
        self.assertFalse(acc)
        self.assertIn('orth_frac', why)

    def test_meter_tolerance_is_honoured(self):
        below = decide(1e-5, 9e-6, {'param_loss': 1.0}, {'param_loss': 1.0005}, **self.OK)[0]
        above = decide(1e-5, 9e-6, {'param_loss': 1.0}, {'param_loss': 1.5}, **self.OK)[0]
        self.assertTrue(below)
        self.assertFalse(above)


class TestFailureHandling(unittest.TestCase):
    """A compile failure or an OOM must degrade the phase, never kill the run.

    Every server failure this phase has had (81461, 81462) was the compiled path raising where
    eager would not. The phase runs AFTER Adam, so a crash here throws away hours of training
    that succeeded.
    """

    def test_is_oom_recognises_both_spellings(self):
        from model_augmentation.fit_systems.lbfgs_polish import _is_oom
        self.assertTrue(_is_oom(RuntimeError('CUDA out of memory. Tried to allocate 2 GiB')))
        self.assertTrue(_is_oom(RuntimeError('CUDA OUT OF MEMORY')))
        self.assertFalse(_is_oom(RuntimeError('Dynamic control flow is not supported')))
        self.assertFalse(_is_oom(ValueError('nope')))

    def test_probe_closure_catches_a_raising_closure(self):
        from model_augmentation.fit_systems.lbfgs_polish import _probe_closure

        def boom():
            raise RuntimeError('Dynamic control flow is not supported')

        ok, first, err = _probe_closure(boom, 'compiled', verbose=False)
        self.assertFalse(ok)
        self.assertTrue(np.isnan(first))
        self.assertIsInstance(err, RuntimeError)

    def test_probe_closure_reports_non_determinism_without_an_error(self):
        from model_augmentation.fit_systems.lbfgs_polish import _probe_closure
        vals = iter([1.0, 1.0000001])
        ok, first, err = _probe_closure(lambda: next(vals), 'compiled', verbose=False)
        self.assertFalse(ok)
        self.assertIsNone(err)                      # ran fine, just not reproducible


class TestOptimizerMoments(unittest.TestCase):
    """D-173. A resume must keep the MOMENTS and the RUN's hyperparameters, not the checkpoint's.

    Pure torch: two optimizers over identical parameters, no dataset, no GPU, no fixture. This is
    the whole behaviour, so it can be proven here rather than on the cluster.
    """

    @staticmethod
    def _make(lr, eps):
        p = torch.nn.Parameter(torch.zeros(4))
        return p, torch.optim.Adam([p], lr=lr, eps=eps)

    def _trained_state(self):
        """An Adam that has taken real steps, so its moments are non-zero."""
        p, opt = self._make(lr=1e-3, eps=1e-8)
        for i in range(5):
            opt.zero_grad()
            p.grad = torch.full_like(p, 0.1 * (i + 1))
            opt.step()
        return opt.state_dict()

    def test_moments_are_restored(self):
        from gantry_dynamic.training import load_optimizer_moments
        saved = self._trained_state()
        p, fresh = self._make(lr=5e-5, eps=1e-16)
        self.assertEqual(len(fresh.state_dict()['state']), 0)     # nothing before
        load_optimizer_moments(fresh, saved)
        got = fresh.state_dict()['state'][0]
        want = saved['state'][0]
        self.assertEqual(int(got['step']), int(want['step']))
        self.assertTrue(torch.allclose(got['exp_avg'], want['exp_avg']))
        self.assertTrue(torch.allclose(got['exp_avg_sq'], want['exp_avg_sq']))

    def test_run_hyperparameters_win(self):
        """The point of the whole exercise: cfg stays the authority on lr, eps and betas."""
        from gantry_dynamic.training import load_optimizer_moments
        saved = self._trained_state()                              # lr 1e-3, eps 1e-8
        p, fresh = self._make(lr=5e-5, eps=1e-16)
        load_optimizer_moments(fresh, saved)
        g = fresh.param_groups[0]
        self.assertEqual(g['lr'], 5e-5)
        self.assertEqual(g['eps'], 1e-16)

    def test_plain_load_state_dict_would_have_taken_the_checkpoints(self):
        """Pins the torch behaviour this exists to avoid, so the reason cannot rot."""
        saved = self._trained_state()                              # lr 1e-3, eps 1e-8
        p, fresh = self._make(lr=5e-5, eps=1e-16)
        fresh.load_state_dict(saved)
        self.assertEqual(fresh.param_groups[0]['lr'], 1e-3)        # the trap, demonstrated
        self.assertEqual(fresh.param_groups[0]['eps'], 1e-8)

    def test_the_optimizer_still_steps_after_a_restore(self):
        from gantry_dynamic.training import load_optimizer_moments
        saved = self._trained_state()
        p, fresh = self._make(lr=5e-5, eps=1e-16)
        load_optimizer_moments(fresh, saved)
        before = p.detach().clone()
        fresh.zero_grad()
        p.grad = torch.full_like(p, 0.3)
        fresh.step()
        self.assertFalse(torch.equal(p.detach(), before))

    def test_group_count_mismatch_is_named(self):
        from gantry_dynamic.training import load_optimizer_moments
        saved = self._trained_state()
        a, b = torch.nn.Parameter(torch.zeros(4)), torch.nn.Parameter(torch.zeros(4))
        two = torch.optim.Adam([{'params': [a]}, {'params': [b]}], lr=1e-4)
        with self.assertRaises(RuntimeError) as e:
            load_optimizer_moments(two, saved)
        self.assertIn('parameter groups do not match', str(e.exception))

    @unittest.skipUnless(FIXTURE.exists(), 'whole-system fixture not present')
    def test_whole_system_checkpoint_yields_an_optimizer_state(self):
        """The mistake D-173 fixes: this used to return None and claim none existed."""
        from gantry_dynamic.training import resolve_checkpoint
        _hfn, _enc, opt_sd, _meta, fmt = resolve_checkpoint(str(FIXTURE))
        self.assertEqual(fmt, 'whole')
        self.assertIsNotNone(opt_sd, 'the pickled system carries a live Adam object')
        self.assertIn('state', opt_sd)
        self.assertIn('param_groups', opt_sd)


class _FakeRecord:
    def __init__(self, n):
        self.u = np.zeros(n)


class _FakeList:
    def __init__(self, n, k):
        self.sdl = [_FakeRecord(n) for _ in range(k)]


class _FakeSys:
    na = nb = 29
    na_right = 1
    nb_right = 0


class TestWindowCount(unittest.TestCase):
    """Pins deepSI's window arithmetic, which the stride scaling depends on."""

    def test_matches_the_measured_pipeline_number(self):
        """The real pipeline reported 66612 windows before this function existed."""
        self.assertEqual(window_count(_FakeSys(), _FakeList(48000, 14), nf=400, stride=10), 66612)

    def test_single_record_is_not_special_cased(self):
        one = window_count(_FakeSys(), _FakeRecord(48000), nf=400, stride=10)
        self.assertEqual(one * 14, 66612)

    def test_count_falls_as_stride_rises(self):
        a = window_count(_FakeSys(), _FakeList(48000, 14), nf=400, stride=10)
        b = window_count(_FakeSys(), _FakeList(48000, 14), nf=400, stride=160)
        self.assertLess(b * 10, a)

    def test_record_shorter_than_the_window_yields_none(self):
        self.assertEqual(window_count(_FakeSys(), _FakeRecord(100), nf=400, stride=10), 0)


class TestConfigValidation(unittest.TestCase):
    """States the phase cannot run in are refused at config construction, not mid-run."""

    def setUp(self):
        from gantry_dynamic.config import RunConfig
        self.cfg = RunConfig()

    def test_defaults_are_a_no_op(self):
        self.assertFalse(self.cfg.lbfgs)
        self.assertEqual(self.cfg.start_phase, 'adam')
        self.assertEqual(self.cfg.burn_in, 0)          # D-178: the pre-D-178 objective

    def test_burn_in_must_leave_something_to_score(self):
        replace(self.cfg, burn_in=self.cfg.nf - 1)     # legal, one sample scored
        for bad in (-1, self.cfg.nf, self.cfg.nf + 1):
            with self.assertRaises(ValueError, msg='burn_in=%r was accepted' % bad):
                replace(self.cfg, burn_in=bad)

    def test_burn_in_reaches_config_json(self):
        from gantry_dynamic.config import config_json_dict
        self.assertEqual(config_json_dict(replace(self.cfg, burn_in=100))['BURN_IN'], 100)

    def test_bad_start_phase(self):
        with self.assertRaises(ValueError):
            replace(self.cfg, start_phase='quasi-newton')

    def test_lbfgs_start_without_the_phase_enabled(self):
        with self.assertRaises(ValueError):
            replace(self.cfg, start_phase='lbfgs', lbfgs=False)

    def test_cap_below_inner_iter(self):
        with self.assertRaises(ValueError):
            replace(self.cfg, lbfgs=True, lbfgs_max_iter=5, lbfgs_inner_iter=20)

    def test_negative_window_count(self):
        with self.assertRaises(ValueError):
            replace(self.cfg, lbfgs=True, lbfgs_windows=-1)

    def test_full_batch_without_chunking_is_refused(self):
        """lbfgs_windows=0 used to be advertised and unimplementable; now it needs a chunk."""
        with self.assertRaises(ValueError) as e:
            replace(self.cfg, lbfgs=True, lbfgs_windows=0, lbfgs_chunk=None)
        self.assertIn('lbfgs_chunk', str(e.exception))

    def test_full_batch_with_chunking_is_legal(self):
        c = replace(self.cfg, lbfgs=True, lbfgs_windows=0, lbfgs_chunk=512)
        self.assertEqual(c.lbfgs_windows, 0)

    def test_bad_chunk(self):
        with self.assertRaises(ValueError):
            replace(self.cfg, lbfgs=True, lbfgs_chunk=0)

    def test_every_knob_reaches_config_json(self):
        from gantry_dynamic.config import config_json_dict
        d = config_json_dict(replace(self.cfg, lbfgs=True))
        for k in ('LBFGS', 'LBFGS_MAX_ITER', 'LBFGS_INNER_ITER', 'LBFGS_HISTORY',
                  'LBFGS_WINDOWS', 'LBFGS_CHUNK', 'LBFGS_TOL_GRAD', 'LBFGS_TOL_CHANGE',
                  'LBFGS_FREEZE', 'LBFGS_TIME_BUDGET_S', 'LBFGS_METER_REL_TOL', 'START_PHASE'):
            self.assertIn(k, d)


# =================================================================================================
# Tier 2: checkpoint I/O
# =================================================================================================

class TestResolveCheckpoint(unittest.TestCase):
    """Both formats normalise to the same four things, and a mismatch is named."""

    def test_pair_format(self):
        from gantry_dynamic.training import resolve_checkpoint
        with tempfile.TemporaryDirectory() as td:
            base = os.path.join(td, 'gantry_ckpt_test')
            torch.save({'hfn': {'a': torch.zeros(2)}, 'encoder': {'b': torch.ones(3)},
                        'optimizer': {'state': {}}}, base + '.pt')
            np.savez(base + '.npz', done_epochs=np.array(7), bestfit=np.array(1.5),
                     hp=np.array(json.dumps({'nf': 400})))
            hfn, enc, opt, meta, fmt = resolve_checkpoint(base)
            self.assertEqual(fmt, 'pair')
            self.assertIn('a', hfn)
            self.assertIsNotNone(opt)
            self.assertEqual(int(meta['done_epochs']), 7)
            # np.load returns a LAZY NpzFile holding the file open, which is also what the
            # production loader hands back. On Windows that blocks the TemporaryDirectory
            # cleanup, so the test closes it; callers keeping the handle are unaffected.
            meta.close()

    def test_optimizer_state_may_be_omitted_and_still_loads(self):
        """An accepted polish drops the Adam state; both routes must cope with its absence.

        Adam's moments belong to the pre-polish weights, so keeping them next to polished
        parameters would silently mis-start a later `start_phase='adam'` resume.
        """
        from gantry_dynamic.training import resolve_checkpoint
        with tempfile.TemporaryDirectory() as td:
            base = os.path.join(td, 'gantry_ckpt_polished')
            torch.save({'hfn': {'a': torch.zeros(2)}, 'encoder': {'b': torch.ones(3)}},
                       base + '.pt')                       # no 'optimizer' key at all
            np.savez(base + '.npz', done_epochs=np.array(3), bestfit=np.array(1.0),
                     hp=np.array(json.dumps({'nf': 400})),
                     optimizer_state_dropped=np.array(True))
            hfn, enc, opt, meta, fmt = resolve_checkpoint(base)
            self.assertEqual(fmt, 'pair')
            self.assertIsNone(opt)                          # absent, not an error
            self.assertTrue(bool(meta['optimizer_state_dropped']))
            meta.close()

    def test_resuming_adam_without_optimizer_state_says_so(self):
        """A missing Adam state must be ANNOUNCED, not silently replaced by fresh moments."""
        import contextlib
        import io as _io
        from gantry_dynamic.training import load_checkpoint

        class _Stub:
            class _M(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.w = torch.nn.Parameter(torch.zeros(2))
            def __init__(self):
                self.hfn = self._M()
                self.encoder = self._M()

        sys_ = _Stub()
        with tempfile.TemporaryDirectory() as td:
            base = os.path.join(td, 'gantry_ckpt_polished')
            torch.save({'hfn': sys_.hfn.state_dict(), 'encoder': sys_.encoder.state_dict()},
                       base + '.pt')
            np.savez(base + '.npz', done_epochs=np.array(3), bestfit=np.array(1.0),
                     hp=np.array(json.dumps({'nf': 400})),
                     optimizer_state_dropped=np.array(True))
            buf = _io.StringIO()
            with contextlib.redirect_stdout(buf):
                meta = load_checkpoint(sys_, base, joint_estimation=False, start_phase='adam')
            meta.close()
        out = buf.getvalue()
        self.assertIn('NO Adam optimizer state', out)
        self.assertIn('optimizer_state_dropped=True', out)
        self.assertIn('_pre_lbfgs', out)          # tells the reader where the real state is

    def test_save_checkpoint_weights_can_omit_the_optimizer(self):
        from gantry_dynamic.training import save_checkpoint_weights

        class _Stub:
            class _M:
                @staticmethod
                def state_dict():
                    return {'w': torch.zeros(1)}
            hfn = encoder = _M()

            class _Opt:
                @staticmethod
                def state_dict():
                    return {'state': {}}
            optimizer = _Opt()

        with tempfile.TemporaryDirectory() as td:
            with_opt = os.path.join(td, 'with')
            without = os.path.join(td, 'without')
            save_checkpoint_weights(_Stub(), with_opt, include_optimizer=True)
            save_checkpoint_weights(_Stub(), without, include_optimizer=False)
            self.assertIn('optimizer', torch.load(with_opt + '.pt'))
            self.assertNotIn('optimizer', torch.load(without + '.pt'))

    def test_missing_checkpoint_names_both_routes(self):
        from gantry_dynamic.training import resolve_checkpoint
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError) as e:
                resolve_checkpoint(os.path.join(td, 'nothing_here'))
            self.assertIn('.pth', str(e.exception))

    def test_architecture_guard_names_the_offender(self):
        from gantry_dynamic.training import _assert_same_architecture
        target = {'w': torch.zeros(8, 4), 'b': torch.zeros(8)}
        ckpt = {'w': torch.zeros(2, 4), 'b': torch.zeros(2)}
        with self.assertRaises(RuntimeError) as e:
            _assert_same_architecture(target, ckpt, 'hfn')
        self.assertIn('shape mismatch', str(e.exception))
        self.assertIn('nx_ann', str(e.exception))          # points at the usual cause

    def test_architecture_guard_passes_on_a_match(self):
        from gantry_dynamic.training import _assert_same_architecture
        _assert_same_architecture({'w': torch.zeros(8, 4)}, {'w': torch.zeros(8, 4)}, 'hfn')

    @unittest.skipUnless(FIXTURE.exists(), 'whole-system fixture not present')
    def test_whole_system_format(self):
        from gantry_dynamic.training import resolve_checkpoint
        hfn, enc, opt, meta, fmt = resolve_checkpoint(str(FIXTURE))
        self.assertEqual(fmt, 'whole')
        # D-173: this used to assert None. The pickled system carries a live Adam object, and
        # its state dict is what makes an Adam resume from this format a real continuation.
        self.assertIsNotNone(opt)
        self.assertTrue(all(torch.is_tensor(v) for v in hfn.values()))
        self.assertTrue(np.isfinite(float(meta['bestfit'])))

    @unittest.skipUnless(FIXTURE.exists(), 'whole-system fixture not present')
    def test_every_natural_spelling_of_the_path_resolves(self):
        """Job 81500 died here: `..._p2LPDA_best` became `..._p2LPDA_best_best.pth`.

        Three spellings are all natural depending on how the path was copied off disk, and the
        earlier code accepted only one of them. The test that existed passed the full `.pth`
        path, so the branch the RUNNER used was never exercised.
        """
        from gantry_dynamic.training import resolve_checkpoint
        full = str(FIXTURE)                                  # .../..._p2LPDA_best.pth
        no_ext = full[:-len('.pth')]                         # .../..._p2LPDA_best
        deepsi_name = no_ext[:-len('_best')]                 # .../..._p2LPDA
        first = None
        for spelling in (full, no_ext, deepsi_name):
            hfn, _enc, _opt, _meta, fmt = resolve_checkpoint(spelling)
            self.assertEqual(fmt, 'whole', spelling)
            keys = sorted(hfn)
            if first is None:
                first = keys
            self.assertEqual(keys, first, 'spelling %r resolved to a different file' % spelling)

    @unittest.skipUnless(FIXTURE.exists(), 'whole-system fixture not present')
    def test_the_runners_own_checkpoint_path_resolves(self):
        """Read the LITERAL path out of the SLURM runner and resolve it.

        This closes the class of failure that produced job 81500. The path spelling was testable
        locally all along; the test simply used a different spelling than the runner did, so the
        two drifted and only the cluster noticed. Anything the runner hardcodes should be checked
        against the code that consumes it, not paraphrased into a test.

        The runner's path is absolute on the cluster, so it is re-rooted here by splitting on the
        repository directory name.
        """
        import re
        from gantry_dynamic.training import resolve_checkpoint
        runner = ROOT / 'scripts' / 'gantry' / 'lbfgs' / 'runners' / 'lbfgs_polish_smoke.sh'
        if not runner.exists():
            self.skipTest('runner not present')
        m = re.search(r'^export RESUME_CHECKPOINT=(\S+)', runner.read_text(), re.M)
        self.assertIsNotNone(m, 'the runner no longer exports RESUME_CHECKPOINT')
        cluster_path = m.group(1)
        marker = '/scripts/gantry/'
        self.assertIn(marker, cluster_path, 'cannot re-root %r' % cluster_path)
        local = ROOT / 'scripts' / 'gantry' / cluster_path.split(marker, 1)[1]
        _hfn, _enc, _opt, _meta, fmt = resolve_checkpoint(str(local))
        self.assertEqual(fmt, 'whole')

    def test_missing_checkpoint_error_lists_what_it_tried(self):
        from gantry_dynamic.training import resolve_checkpoint
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError) as e:
                resolve_checkpoint(os.path.join(td, 'nope_best'))
            msg = str(e.exception)
            self.assertIn('nope_best.pth', msg)              # every candidate is named,
            self.assertIn('nope_best_best.pth', msg)         # so the mismatch is diagnosable


# =================================================================================================
# Tier 3: the real phase
# =================================================================================================

def _build_fixture_system():
    """The job-81262 closed-loop system on the CPU, with its trained weights loaded.

    The ARCHITECTURE comes from the checkpoint, not from `CFG`. Job 81461 failed because the
    cluster's entry file carried a 16x2 nx_ann=2 configuration while the fixture is 24x3
    nx_ann=8, and a test that silently depends on which config is checked out is not a test.
    Everything else (dataset, nf, stride, closed loop) still comes from `CFG`, because those are
    properties of the experiment rather than of the weights.
    """
    import gantry_interconnect_dynamic as G
    from gantry_dynamic.data import load_datasets, compute_normalization, VAL_FILES, TRAIN_FILES
    from gantry_dynamic.model import build_model
    from gantry_dynamic.controller import build_closed_loop
    from gantry_dynamic.training import config_for_checkpoint, resolve_checkpoint

    hfn_sd, enc_sd, _opt, _meta, _fmt = resolve_checkpoint(str(FIXTURE))
    # Pin every switch this fixture depends on rather than inheriting it. CFG is edited
    # between runs (it currently carries smoke-test values), and a test that reads whatever
    # happens to be there is testing the file, not the phase.
    cfg = replace(G.CFG, device='cpu', compile_mode=None, save_flag=False,
                  lbfgs=True, start_phase='adam')
    cfg = config_for_checkpoint(cfg, hfn_sd)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    data = load_datasets(cfg)
    norm = compute_normalization(cfg, data)
    hp = cfg.hp
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    fit_sys = build_model(hp, cfg, data, norm)
    fit_sys.simulator = build_closed_loop(fit_sys, norm, cfg, train_files=TRAIN_FILES,
                                          val_files=VAL_FILES, val_data=data.val_ckpt_data)
    try:
        fit_sys.hfn.load_state_dict(hfn_sd)
        fit_sys.encoder.load_state_dict(enc_sd)
    except RuntimeError as e:
        # Skip rather than ERROR: a fixture that does not fit this checkout is a missing
        # fixture, not a broken phase. The message names what differs.
        raise unittest.SkipTest('fixture does not match this configuration: %s' % e)
    return fit_sys, data, {'nf': hp['nf'], 'stride': cfg.stride}


@unittest.skipUnless(HAVE_FIXTURE, 'job-81262 checkpoint or its dataset not present')
class TestPolishAgainstFixture(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fit_sys, cls.data, cls.lk = _build_fixture_system()
        cls.snap = [p.detach().clone() for _, p in _named_params(cls.fit_sys)]

    def setUp(self):
        # Every test starts from the fixture weights, so ordering cannot matter.
        with torch.no_grad():
            for (_, p), s in zip(_named_params(self.fit_sys), self.snap):
                p.data.copy_(s)

    # ---- cost ---------------------------------------------------------------------------------

    def test_batch_build_scales_with_the_REQUEST_not_the_dataset(self):
        """The regression test for the defect that passed 27 behavioural gates.

        Asking for 64 windows must not materialise all 66,612. The build cost is exactly the
        window count at the stride the batch chose, so that is what is asserted: deterministic,
        and it targets the defect directly rather than sampling a memory counter.
        """
        small = FixedBatch(self.fit_sys, self.data.train_data, self.lk, 64, 42, verbose=False)
        built = window_count(self.fit_sys, self.data.train_data, self.lk['nf'],
                             small.effective_stride)
        self.assertEqual(small.n_used, 64)
        self.assertLess(built, 4 * 64, 'built %d windows for a request of 64' % built)
        self.assertLess(built, 0.02 * small.n_total)
        self.assertGreater(small.effective_stride, self.lk['stride'])

    def test_a_bigger_request_builds_more(self):
        a = FixedBatch(self.fit_sys, self.data.train_data, self.lk, 64, 42, verbose=False)
        b = FixedBatch(self.fit_sys, self.data.train_data, self.lk, 512, 42, verbose=False)
        self.assertLess(b.effective_stride, a.effective_stride)
        self.assertLess(a.bytes_on_device, b.bytes_on_device)

    # ---- determinism and reproducibility -------------------------------------------------------

    def test_closure_is_bit_deterministic(self):
        """The precondition strong Wolfe needs: same batch, same weights, same loss, exactly."""
        with _eager(self.fit_sys), _quiet_probes(self.fit_sys):
            batch = FixedBatch(self.fit_sys, self.data.train_data, self.lk, 32, 42, verbose=False)
            uh, yh, uf, yf, kw, w = next(iter(batch))
            l1 = float(self.fit_sys.loss(uh, yh, uf, yf, **kw))
            l2 = float(self.fit_sys.loss(uh, yh, uf, yf, **kw))
        self.assertEqual(l1, l2)

    def test_batch_is_reproducible_from_the_seed(self):
        with _quiet_probes(self.fit_sys):
            b1 = FixedBatch(self.fit_sys, self.data.train_data, self.lk, 32, 7, verbose=False)
            b2 = FixedBatch(self.fit_sys, self.data.train_data, self.lk, 32, 7, verbose=False)
        for c1, c2 in zip(iter(b1), iter(b2)):
            for t1, t2 in zip(c1[:4], c2[:4]):
                self.assertTrue(torch.equal(t1, t2))
            # ctrl_ix must stay an integer index (Dtype_DataLoader._conv rule).
            for k, v in c1[4].items():
                if torch.is_tensor(v):
                    self.assertFalse(v.is_floating_point(), '%s became floating point' % k)

    # ---- chunked accumulation ------------------------------------------------------------------

    def test_chunked_gradient_equals_the_single_chunk_gradient(self):
        """The claim that makes full batch reachable: accumulation is EXACT, not approximate."""
        def loss_and_grad(chunk):
            with _eager(self.fit_sys), _quiet_probes(self.fit_sys):
                batch = FixedBatch(self.fit_sys, self.data.train_data, self.lk, 64, 42,
                                   chunk=chunk, verbose=False)
                params = [p for _, p in _named_params(self.fit_sys) if p.requires_grad]
                for p in params:
                    p.grad = None
                total = 0.0
                for uh, yh, uf, yf, kw, w in batch:
                    l = self.fit_sys.loss(uh, yh, uf, yf, **kw) * w
                    l.backward()
                    total += float(l)
                g = torch.cat([(p.grad if p.grad is not None
                                else torch.zeros_like(p)).reshape(-1) for p in params])
                return total, g

        l_one, g_one = loss_and_grad(None)
        l_four, g_four = loss_and_grad(16)          # 64 windows in 4 chunks
        self.assertLessEqual(abs(l_four - l_one) / abs(l_one), 1e-5)
        rel = float(torch.linalg.vector_norm(g_four - g_one)
                    / torch.linalg.vector_norm(g_one))
        self.assertLess(rel, 1e-4, 'chunked gradient differs by %.2e relative' % rel)

    def test_chunking_does_not_change_the_window_set(self):
        with _quiet_probes(self.fit_sys):
            a = FixedBatch(self.fit_sys, self.data.train_data, self.lk, 64, 42, verbose=False)
            b = FixedBatch(self.fit_sys, self.data.train_data, self.lk, 64, 42, chunk=16,
                           verbose=False)
        self.assertEqual(a.n_used, b.n_used)
        self.assertEqual(b.n_chunks, 4)
        self.assertTrue(torch.equal(torch.cat([c[0] for c in a]),
                                    torch.cat([c[0] for c in b])))

    # ---- burn-in (D-178) ------------------------------------------------------------------------

    def test_burn_in_zero_is_the_old_objective_bit_exactly(self):
        """The default must not perturb a single run that predates the knob."""
        uh, yh, uf, yf, kw, _w = next(iter(FixedBatch(
            self.fit_sys, self.data.train_data, self.lk, 16, 42, verbose=False)))
        self.fit_sys.burn_in = 0
        with torch.no_grad():
            a = float(self.fit_sys.loss(uh, yh, uf, yf, **kw))
            b = float(self.fit_sys.loss(uh, yh, uf, yf, **kw))
        self.assertEqual(a, b, 'the loss is not even reproducible; the rest of this means nothing')
        self.assertTrue(np.isfinite(a))

    def test_burn_in_scores_exactly_the_slice(self):
        """`burn_in=K` must equal the MSE over [K, nf), computed independently of the loss."""
        import torch.nn.functional as F
        uh, yh, uf, yf, kw, _w = next(iter(FixedBatch(
            self.fit_sys, self.data.train_data, self.lk, 16, 42, verbose=False)))
        K = 100
        try:
            with torch.no_grad():
                x = self.fit_sys.encoder(uh, yh)
                y_pred, _ = self.fit_sys.simulate(x, uf, yf, **kw)
                want = float(F.mse_loss(yf[:, K:], y_pred[:, K:]))
                self.fit_sys.burn_in = K
                got = float(self.fit_sys.loss(uh, yh, uf, yf, **kw))
        finally:
            self.fit_sys.burn_in = 0
        # This fixture has orth OFF and joint estimation OFF, so loss() is the MSE alone.
        self.assertAlmostEqual(got, want, delta=abs(want) * 1e-9,
                               msg='burn_in did not score exactly [K, nf)')

    def test_burn_in_changes_the_loss(self):
        """A burn-in that changed nothing would be a silent no-op, which is the failure to fear."""
        uh, yh, uf, yf, kw, _w = next(iter(FixedBatch(
            self.fit_sys, self.data.train_data, self.lk, 16, 42, verbose=False)))
        try:
            with torch.no_grad():
                self.fit_sys.burn_in = 0
                full = float(self.fit_sys.loss(uh, yh, uf, yf, **kw))
                self.fit_sys.burn_in = 100
                scored = float(self.fit_sys.loss(uh, yh, uf, yf, **kw))
        finally:
            self.fit_sys.burn_in = 0
        self.assertNotEqual(full, scored)
        # grow < 1 on this pipeline, i.e. the window error DECAYS, so dropping the transient must
        # LOWER the score. That is the whole premise of D-178 and it is checked, not assumed.
        self.assertLess(scored, full, 'the window is not transient-dominated on this fixture')

    def test_burn_in_past_the_window_raises(self):
        uh, yh, uf, yf, kw, _w = next(iter(FixedBatch(
            self.fit_sys, self.data.train_data, self.lk, 16, 42, verbose=False)))
        try:
            self.fit_sys.burn_in = uf.shape[1] + 1
            with self.assertRaises(ValueError):
                with torch.no_grad():
                    self.fit_sys.loss(uh, yh, uf, yf, **kw)
        finally:
            self.fit_sys.burn_in = 0

    # ---- the phase ------------------------------------------------------------------------------

    def test_smoke(self):
        out = lbfgs_polish(self.fit_sys, self.data.train_data, loss_kwargs=self.lk,
                           spec=PolishSpec(n_windows=32, max_iter=2, inner_iter=1),
                           val_sys_data=None, accept_without_validation=True, verbose=False)
        self.assertTrue(out['ran'])
        self.assertTrue(out['deterministic'])
        self.assertTrue(np.isfinite(out['loss_first']))
        self.assertEqual(out['n_windows_used'], 32)
        self.assertGreater(out['effective_stride'], self.lk['stride'])

    def test_loss_does_not_increase(self):
        out = lbfgs_polish(self.fit_sys, self.data.train_data, loss_kwargs=self.lk,
                           spec=PolishSpec(n_windows=64, max_iter=6, inner_iter=3),
                           val_sys_data=None, accept_without_validation=True, verbose=False)
        self.assertLessEqual(out['loss_last'], out['loss_first'])

    def test_rollback_on_regressed_meter_is_bit_exact(self):
        before = [p.detach().clone() for _, p in _named_params(self.fit_sys)]
        calls = [0]

        def worsening(_fs):
            calls[0] += 1
            return {'injected': float(calls[0])}       # 1.0 before, 2.0 after

        out = lbfgs_polish(self.fit_sys, self.data.train_data, loss_kwargs=self.lk,
                           spec=PolishSpec(n_windows=32, max_iter=2, inner_iter=1),
                           val_sys_data=None, accept_without_validation=True,
                           meters_fn=worsening, verbose=False)
        self.assertFalse(out['accepted'])
        self.assertIn('injected', out['reason'])
        for (_, p), b in zip(_named_params(self.fit_sys), before):
            self.assertTrue(torch.equal(p.detach(), b))

    def test_no_validation_rolls_back_by_default(self):
        """Without a validation measure and without the explicit opt-in, nothing is kept."""
        before = [p.detach().clone() for _, p in _named_params(self.fit_sys)]
        out = lbfgs_polish(self.fit_sys, self.data.train_data, loss_kwargs=self.lk,
                           spec=PolishSpec(n_windows=32, max_iter=2, inner_iter=1),
                           val_sys_data=None, verbose=False)
        self.assertFalse(out['accepted'])
        for (_, p), b in zip(_named_params(self.fit_sys), before):
            self.assertTrue(torch.equal(p.detach(), b))

    def test_freeze_encoder_leaves_it_untouched(self):
        enc_before = [p.detach().clone() for p in self.fit_sys.encoder.parameters()]
        n_all = sum(p.numel() for _, p in _named_params(self.fit_sys))
        for p in self.fit_sys.encoder.parameters():
            p.grad = None
        out = lbfgs_polish(self.fit_sys, self.data.train_data, loss_kwargs=self.lk,
                           spec=PolishSpec(n_windows=32, max_iter=2, inner_iter=1,
                                           freeze=('encoder',)),
                           val_sys_data=None, accept_without_validation=True, verbose=False)
        self.assertLess(out['params_polished'], n_all)
        for p, b in zip(self.fit_sys.encoder.parameters(), enc_before):
            self.assertTrue(torch.equal(p.detach(), b))
        # D-175: unchanged weights were never the whole claim. Before `_frozen`, backward still
        # computed the encoder's gradients and nothing ever cleared them, so freezing saved no
        # compute and left a stale `.grad` behind. No gradient is what "frozen" has to mean.
        for p in self.fit_sys.encoder.parameters():
            self.assertIsNone(p.grad, 'a frozen parameter received a gradient')
            self.assertTrue(p.requires_grad, 'requires_grad was not restored after the phase')

    def test_iteration_count_is_measured_not_inferred(self):
        """D-174. The count torch reports, bounded by the evaluations it actually paid for.

        The old code inferred `(k + 1) * inner_iter`, which reported 500 iterations for 138
        closures in job 81655. Every L-BFGS iteration costs at least one closure evaluation, so
        that ordering is the invariant the inference violated.
        """
        out = lbfgs_polish(self.fit_sys, self.data.train_data, loss_kwargs=self.lk,
                           spec=PolishSpec(n_windows=32, max_iter=6, inner_iter=3),
                           val_sys_data=None, accept_without_validation=True, verbose=False)
        self.assertGreater(out['n_iter'], 0)
        self.assertLessEqual(out['n_iter'], out['max_iter'])
        self.assertLessEqual(out['n_iter'], out['n_closure'])
        self.assertLessEqual(out['func_evals'], out['n_closure'])

    def test_loss_last_is_the_loss_at_the_polished_point(self):
        """D-174. `step()` returns the START of its block; `loss_last` must not be that value.

        Checked through the rollback, which is bit-exact: after a rejected phase the weights are
        the ones `loss_first` was measured at, so re-evaluating the objective must reproduce
        `loss_first` exactly and must NOT reproduce `loss_last`, which belongs to the point the
        phase reached before being rejected.
        """
        before = [p.detach().clone() for _, p in _named_params(self.fit_sys)]
        out = lbfgs_polish(self.fit_sys, self.data.train_data, loss_kwargs=self.lk,
                           spec=PolishSpec(n_windows=32, max_iter=6, inner_iter=3),
                           val_sys_data=None, verbose=False)      # no val -> always rolled back
        self.assertFalse(out['accepted'])
        for (_, p), b in zip(_named_params(self.fit_sys), before):
            self.assertTrue(torch.equal(p.detach(), b))
        batch = FixedBatch(self.fit_sys, self.data.train_data, self.lk, 32, 42, verbose=False)
        self.assertAlmostEqual(_loss_value(self.fit_sys, batch), out['loss_first'], places=12)
        self.assertLess(out['loss_last'], out['loss_first'])

    def test_an_oom_mid_phase_restores_the_weights_and_does_not_raise(self):
        """The guarantee: a crash inside the phase cannot cost the run its trained weights."""
        before = [p.detach().clone() for _, p in _named_params(self.fit_sys)]
        real_loss = self.fit_sys.loss
        calls = [0]

        def failing_loss(*a, **kw):
            calls[0] += 1
            if calls[0] > 3:          # survive the warm-up and the determinism pair, then fail
                raise RuntimeError('CUDA out of memory. Tried to allocate 20.00 GiB')
            return real_loss(*a, **kw)

        self.fit_sys.loss = failing_loss
        try:
            out = lbfgs_polish(self.fit_sys, self.data.train_data, loss_kwargs=self.lk,
                               spec=PolishSpec(n_windows=32, max_iter=2, inner_iter=1),
                               val_sys_data=None, accept_without_validation=True, verbose=False)
        finally:
            del self.fit_sys.loss     # drop the instance attribute, restore the bound method
        self.assertFalse(out['accepted'])
        self.assertIn('out of memory', out['reason'].lower())
        self.assertIn('lbfgs_windows', out['reason'])       # tells the user what to change
        for (_, p), b in zip(_named_params(self.fit_sys), before):
            self.assertTrue(torch.equal(p.detach(), b))

    def test_a_raising_rollout_aborts_cleanly_rather_than_propagating(self):
        before = [p.detach().clone() for _, p in _named_params(self.fit_sys)]
        real_loss = self.fit_sys.loss
        calls = [0]

        def failing_loss(*a, **kw):
            calls[0] += 1
            if calls[0] > 1:          # warm-up succeeds, every measured evaluation fails
                raise RuntimeError('Dynamic control flow is not supported')
            return real_loss(*a, **kw)

        self.fit_sys.loss = failing_loss
        try:
            out = lbfgs_polish(self.fit_sys, self.data.train_data, loss_kwargs=self.lk,
                               spec=PolishSpec(n_windows=32, max_iter=2, inner_iter=1),
                               val_sys_data=None, accept_without_validation=True, verbose=False)
        finally:
            del self.fit_sys.loss
        self.assertFalse(out['accepted'])
        self.assertFalse(out['deterministic'])
        for (_, p), b in zip(_named_params(self.fit_sys), before):
            self.assertTrue(torch.equal(p.detach(), b))

    def test_unknown_freeze_name_raises(self):
        with self.assertRaises(ValueError) as e:
            lbfgs_polish(self.fit_sys, self.data.train_data, loss_kwargs=self.lk,
                         spec=PolishSpec(n_windows=8, max_iter=1, inner_iter=1,
                                         freeze=('encodr',)),
                         val_sys_data=None, verbose=False)
        self.assertIn('encodr', str(e.exception))

    def test_run_id_becomes_the_deepsi_checkpoint_name(self):
        """deepSI's own `_best.pth` / `_last.pth` should carry the job id, not a random code.

        `name` is a property over `unique_code` (`deepSI/systems/system.py:79-80`), so setting
        the code is the whole change; the directory is untouched.
        """
        original = self.fit_sys.unique_code
        try:
            self.fit_sys.unique_code = '81463'
            self.assertTrue(self.fit_sys.name.endswith('_81463'), self.fit_sys.name)
            self.assertIn('SSE_Interconnect', self.fit_sys.name)
        finally:
            self.fit_sys.unique_code = original

    def test_post_run_evaluation_does_not_touch_the_weights(self):
        """D-172. `capture_loss_history` used to reload `_best.pth` into the system.

        Two failures came out of that. On a polish-only job the file did not exist and the run
        crashed after finishing (job 81503). On a normal job it DID exist, so after an accepted
        polish it silently put the unpolished weights back and every diagnostic below reported
        the wrong model. `fit()` now keeps its own history, so this function reads it and
        touches nothing.
        """
        from gantry_dynamic.evaluation import capture_loss_history
        import gantry_interconnect_dynamic as G
        cfg = replace(G.CFG, device='cpu', compile_mode=None, save_flag=False,
                      lbfgs=True, start_phase='adam')
        before = [p.detach().clone() for _, p in _named_params(self.fit_sys)]
        ep, lv, lt = capture_loss_history(self.fit_sys, cfg, save_dir=None, rid='x')
        for (_, p), b in zip(_named_params(self.fit_sys), before):
            self.assertTrue(torch.equal(p.detach(), b), 'capture_loss_history moved a weight')
        # And it returns what the system holds, rather than reconstructing it from disk.
        self.assertEqual(len(np.asarray(lv)), len(np.asarray(self.fit_sys.Loss_val)))
        self.assertEqual(len(np.asarray(ep)), len(np.asarray(self.fit_sys.epoch_id)))
        self.assertEqual(len(np.asarray(lt)), len(np.asarray(self.fit_sys.Loss_train)))

    def test_named_params_has_no_duplicates(self):
        named = _named_params(self.fit_sys)
        self.assertEqual(len({id(p) for _, p in named}), len(named))

    def test_lbfgs_false_is_an_exact_no_op(self):
        """The invariant the whole design rests on: lbfgs=False changes nothing."""
        import gantry_interconnect_dynamic as G
        from gantry_dynamic.training import run_lbfgs_polish
        # `start_phase='adam'` explicitly, not inherited: whatever CFG happens to hold, the
        # no-op being tested is "phase disabled on a normal run", and lbfgs=False with
        # start_phase='lbfgs' is a state RunConfig rightly refuses.
        cfg_off = replace(G.CFG, device='cpu', compile_mode=None, save_flag=False,
                          lbfgs=False, start_phase='adam')
        before = [p.detach().clone() for _, p in _named_params(self.fit_sys)]
        self.assertIsNone(run_lbfgs_polish(self.fit_sys, cfg_off.hp, cfg_off, self.data))
        for (_, p), b in zip(_named_params(self.fit_sys), before):
            self.assertTrue(torch.equal(p.detach(), b))

    def test_compiled_flag_is_restored(self):
        """`_eager` must put the simulator back exactly as it found it, and only when active."""
        sim = self.fit_sys.simulator
        sentinel = object()
        sim._compiled = sentinel
        try:
            with _eager(self.fit_sys, active=True):
                self.assertIsNone(sim._compiled)
            self.assertIs(sim._compiled, sentinel)
            with _eager(self.fit_sys, active=False):
                self.assertIs(sim._compiled, sentinel)      # left alone when not forced
            self.assertIs(sim._compiled, sentinel)
        finally:
            sim._compiled = None


if __name__ == '__main__':
    unittest.main(verbosity=2)
