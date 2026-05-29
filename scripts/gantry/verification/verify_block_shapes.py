"""
verify_block_shapes.py
----------------------
Check that Gantry_State_Block produces the correct output shape and
contains no NaN or Inf values, for both frozen-Y and LPV modes.

Run from project root:
    conda run -n GraduationProject python scripts/gantry/verification/verify_block_shapes.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch
import numpy as np
from model_augmentation.fit_systems.blocks import Gantry_State_Block

def check(name, condition):
    status = 'PASS' if condition else 'FAIL'
    print(f'  {name:<50s}  {status}')
    return condition

def run():
    print('=' * 60)
    print('verify_block_shapes.py')
    print('=' * 60)

    batch = 8
    nx, nu = 6, 3

    # Small nonzero std so normalisation is non-trivial
    std_x = np.ones((6, 1)) * 0.01
    std_u = np.ones((3, 1)) * 10.0

    results = []

    for label, Y_op in [('frozen Y_op=0.3', 0.3), ('LPV Y_op=None', None)]:
        print(f'\n  [{label}]')
        block = Gantry_State_Block(Y_op=Y_op, std_x=std_x, std_u=std_u)
        block.eval()

        # z = [x_norm; u_norm] — small random inputs
        torch.manual_seed(0)
        z = torch.randn(batch, nx + nu, 1) * 0.1

        with torch.no_grad():
            out = block(z)

        results.append(check('output shape == (batch, 6, 1)',  tuple(out.shape) == (batch, nx, 1)))
        results.append(check('no NaN in output',               not out.isnan().any().item()))
        results.append(check('no Inf in output',               not out.isinf().any().item()))

    print()
    overall = all(results)
    print(f'Overall: {"ALL PASS" if overall else "SOME FAILED"}')
    return overall

if __name__ == '__main__':
    ok = run()
    sys.exit(0 if ok else 1)
