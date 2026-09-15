"""Numerical gates for the combined continuous-time RK4 augmentation."""

import os
import sys
import unittest

import numpy as np
import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model_augmentation.fit_systems.blocks import Gantry_State_Block, Static_ANN_Block
from model_augmentation.fit_systems.continuous_augmentation import (
    CombinedContinuousGantryBlock, rk4_step,
)
from model_augmentation.fit_systems.interconnect import Interconnect
from scripts.gantry.gantry_dynamic.multirate_data import matched_decimate


class _ConstantNet(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer('values', torch.as_tensor(values, dtype=torch.float64))
        self.calls = 0

    def forward(self, z):
        self.calls += 1
        return self.values.expand(z.shape[0], -1)


class ContinuousAugmentationTests(unittest.TestCase):
    def test_zero_field_matches_physical_rk4_and_holds_augmented_state(self):
        dtype = torch.float64
        physical = Gantry_State_Block(
            Y_op=None, Ts=1 / 2000, up_sample=2,
            std_x=np.ones((6, 1)), std_u=np.ones((3, 1)),
            x_mean=np.zeros((6, 1)), u_mean=np.zeros((3, 1))).to(dtype)
        ann = Static_ANN_Block(nz=11, nw=8, n_nodes_per_layer=4,
                               n_hidden_layers=1).to(dtype)
        combined = CombinedContinuousGantryBlock(physical, ann, range(8), 8, 3).to(dtype)
        state = torch.tensor([[0.01, -0.02, 0.03, 0.1, -0.1, 0.05, 0.7, -0.4]],
                             dtype=dtype).unsqueeze(-1)
        u = torch.tensor([[0.2, -0.3, 0.1]], dtype=dtype).unsqueeze(-1)
        expected_phys = physical(torch.cat((state[:, :6], u), dim=1))
        actual = combined(torch.cat((state, u), dim=1))
        torch.testing.assert_close(actual[:, :6], expected_phys, rtol=0, atol=2e-15)
        torch.testing.assert_close(actual[:, 6:], state[:, 6:], rtol=0, atol=0)

    def test_ann_runs_at_every_substage_and_integrates_augmented_derivative(self):
        dtype = torch.float64
        physical = Gantry_State_Block(Ts=0.01, up_sample=3).to(dtype)
        ann = Static_ANN_Block(nz=11, nw=2, n_nodes_per_layer=2,
                               n_hidden_layers=1).to(dtype)
        counter = _ConstantNet([2.0, -3.0])
        ann.net = counter
        combined = CombinedContinuousGantryBlock(physical, ann, (6, 7), 8, 3).to(dtype)
        state = torch.zeros((1, 8, 1), dtype=dtype)
        u = torch.zeros((1, 3, 1), dtype=dtype)
        actual = combined(torch.cat((state, u), dim=1))
        self.assertEqual(counter.calls, 4 * physical.up_sample)
        torch.testing.assert_close(actual[0, 6:, 0],
                                   torch.tensor([0.02, -0.03], dtype=dtype),
                                   rtol=0, atol=2e-15)

    def test_registered_disconnected_ann_runs_only_inside_combined_block(self):
        dtype = torch.float64
        physical = Gantry_State_Block(Ts=0.01, up_sample=1).to(dtype)
        ann = Static_ANN_Block(nz=11, nw=2, n_nodes_per_layer=2,
                               n_hidden_layers=1).to(dtype)
        counter = _ConstantNet([0.0, 0.0])
        ann.net = counter
        combined = CombinedContinuousGantryBlock(physical, ann, (6, 7), 8, 3).to(dtype)
        ic = Interconnect(8, 3, 1)
        ic.add_block(combined)
        ic.add_block(ann)  # checkpoint/diagnostic registration, no graph connection
        ic.connect_block_signals(combined, ['x', 'u'], ['xp'])
        ic.connect_block_signals(ann, ['x', 'u'], [])
        ic.connect_signals('x', 'y', 'additive', torch.zeros((1, 8)))
        x = torch.zeros((1, 8), dtype=dtype)
        u = torch.zeros((1, 3), dtype=dtype)
        _, xp = ic(x, u)
        self.assertEqual(counter.calls, 4)
        self.assertEqual(tuple(xp.shape), (1, 8))

    def test_input_and_output_receive_the_same_antialias_operator(self):
        rng = np.random.default_rng(4)
        signal = rng.normal(size=(301, 2))
        u, y = matched_decimate(signal, signal.copy(), down=5)
        np.testing.assert_array_equal(u, y)

    def test_combined_is_fourth_order_and_post_step_euler_is_first_order(self):
        def f(x):
            return -x + 0.3 * x * x

        def g(x):
            return 0.5 * torch.sin(x)

        def integrate(h, combined):
            x = torch.tensor(0.7, dtype=torch.float64)
            for _ in range(round(1.0 / h)):
                if combined:
                    x = rk4_step(lambda q: f(q) + g(q), x, h)
                else:
                    x = rk4_step(f, x, h) + h * g(x)
            return float(x)

        ref = integrate(2.0 ** -18, True)
        hs = (1 / 20, 1 / 40, 1 / 80, 1 / 160)
        err_combined = np.array([abs(integrate(h, True) - ref) for h in hs])
        err_post = np.array([abs(integrate(h, False) - ref) for h in hs])
        order_combined = np.log2(err_combined[:-1] / err_combined[1:]).mean()
        order_post = np.log2(err_post[:-1] / err_post[1:]).mean()
        self.assertGreater(order_combined, 3.8)
        self.assertLess(abs(order_post - 1.0), 0.12)


if __name__ == '__main__':
    unittest.main()
