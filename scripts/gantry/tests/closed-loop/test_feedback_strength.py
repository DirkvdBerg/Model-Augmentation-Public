"""Unit gates for residual-feedback strength.

These tests are deliberately synthetic: they exercise the one framework rollout directly, without
loading or regenerating gantry datasets. Integration tests in the experiment runner cover the real
controller bank and model.
"""

import pathlib
import sys
import unittest

import numpy as np
import torch


ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from model_augmentation.fit_systems.closed_loop import (  # noqa: E402
    ControllerBank,
    closed_loop_rollout,
)


class _Step(torch.nn.Module):
    """Scalar state model: y=x and x_next=x+u."""

    @staticmethod
    def output_only(x):
        return x

    def forward(self, x, u):
        return self.output_only(x), x + u


def _bank(dtype=torch.float64):
    return ControllerBank(
        A=np.array([[[0.5]]]),
        B=np.array([[[0.2]]]),
        C=np.array([[[0.3]]]),
        D=np.array([[[0.4]]]),
        ystd=np.ones(1), std_u=np.ones(1), dtype=dtype,
    )


def _case(nf=4, requires_grad=False):
    dtype = torch.float64
    hfn = _Step()
    u = torch.tensor([[[0.2], [-0.1], [0.4], [0.3]]], dtype=dtype)[:, :nf]
    y = torch.tensor([[[0.7], [0.1], [-0.2], [0.5]]], dtype=dtype,
                     requires_grad=requires_grad)[:, :nf]
    x0 = torch.tensor([[0.25]], dtype=dtype)
    ix = torch.zeros(1, dtype=torch.long)
    return hfn, u, y, x0, ix


def _open_loop(hfn, u, x0):
    x = x0
    ys = []
    for u_t in u.unbind(1):
        y_t, x = hfn(x, u_t)
        ys.append(y_t)
    return torch.stack(ys, dim=1), x


class FeedbackStrengthTests(unittest.TestCase):

    def test_default_is_bit_identical_to_explicit_one(self):
        hfn, u, y, x0, ix = _case()
        got_default = closed_loop_rollout(hfn, hfn.output_only, u, y, x0, _bank(), ix)
        got_one = closed_loop_rollout(
            hfn, hfn.output_only, u, y, x0, _bank(), ix, feedback_strength=1.0)
        for a, b in zip(got_default, got_one):
            self.assertTrue(torch.equal(a, b))

    def test_zero_is_exact_recorded_input_replay(self):
        hfn, u, y, x0, ix = _case()
        y_zero, x_zero, _ = closed_loop_rollout(
            hfn, hfn.output_only, u, y, x0, _bank(), ix, feedback_strength=0.0)
        y_open, x_open = _open_loop(hfn, u, x0)
        self.assertTrue(torch.equal(y_zero, y_open))
        self.assertTrue(torch.equal(x_zero, x_open))

    def test_zero_removes_output_target_gradient_path(self):
        hfn, u, y, x0, ix = _case(requires_grad=True)
        y_zero, _, _ = closed_loop_rollout(
            hfn, hfn.output_only, u, y, x0, _bank(), ix, feedback_strength=0.0)
        # This synthetic model has no trainable parameters, so removing the only possible target
        # path makes the returned prediction entirely non-differentiable with respect to y_data.
        self.assertFalse(y_zero.requires_grad)

    def test_single_step_input_correction_is_linear_in_strength(self):
        hfn, u, y, x0, ix = _case(nf=1)
        finals = []
        for alpha in (0.0, 0.5, 1.0):
            _, x_final, _ = closed_loop_rollout(
                hfn, hfn.output_only, u, y, x0, _bank(), ix,
                feedback_strength=alpha)
            finals.append(x_final)
        self.assertTrue(torch.allclose(
            finals[1] - finals[0], 0.5 * (finals[2] - finals[0]),
            atol=1e-15, rtol=1e-15))

    def test_invalid_strength_is_rejected(self):
        hfn, u, y, x0, ix = _case(nf=1)
        for value in (-0.1, 1.1, np.nan, np.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'feedback_strength'):
                    closed_loop_rollout(
                        hfn, hfn.output_only, u, y, x0, _bank(), ix,
                        feedback_strength=value)


if __name__ == '__main__':
    unittest.main()
