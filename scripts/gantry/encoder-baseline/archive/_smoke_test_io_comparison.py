"""Smoke test for encoder_io_comparison.py — validates all steps without full training."""
import sys, os
import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from scripts.gantry.encoder.encoder_io_comparison import (
    load_mat, compute_normalization, compute_velocities_from_positions,
    compute_analytical_baseline, create_io_windows,
    build_encoder, build_interconnect, rollout, HP, NX_PHYS, nu, ny,
    na, nb, na_right, nb_right, TRAJ_DIR, TRAIN_FILES, VAL_FILE,
    DTYPE_NP, DTYPE_PT, FS_NEW,
)

def main():
    print('=== Smoke test: encoder_io_comparison.py ===\n')
    nf = HP['nf']

    # 1. Data files
    print('1. Checking data files...')
    for f in TRAIN_FILES + [VAL_FILE]:
        p = os.path.join(TRAJ_DIR, f)
        assert os.path.exists(p), f'MISSING: {p}'
    print('   All data files found.\n')

    # 2. Load one training file + validation
    print('2. Loading data...')
    train_data = [load_mat(TRAIN_FILES[0])]  # just one for speed
    val_u, val_y, val_x_logical, val_delta_a = load_mat(VAL_FILE)
    print(f'   Train: u={train_data[0][0].shape}, y={train_data[0][1].shape}')
    print(f'   Val:   u={val_u.shape}, y={val_y.shape}, delta_a={val_delta_a is not None}\n')

    # 3. Normalization
    print('3. Computing normalization...')
    norm = compute_normalization(train_data)
    print(f'   std_x shape: {norm["std_x"].shape}')
    print(f'   Cd_norm shape: {norm["Cd_norm"].shape}\n')

    # 4. Velocity computation
    print('4. Velocity verification...')
    x_python = compute_velocities_from_positions(val_y, norm['P_inv_T'])
    assert x_python.shape == (len(val_y), 6), f'Expected (N,6), got {x_python.shape}'
    print(f'   x_python shape: {x_python.shape} OK\n')

    # 5. Create I/O windows
    print('5. Creating I/O windows...')
    val_windows = create_io_windows(val_u, val_y, norm, nf)
    u_hist, y_hist, u_future, y_future = val_windows
    print(f'   u_hist:    {u_hist.shape}')
    print(f'   y_hist:    {y_hist.shape}')
    print(f'   u_future:  {u_future.shape}')
    print(f'   y_future:  {y_future.shape}')
    M = u_hist.shape[0]
    na_total = na + na_right
    nb_total = nb + nb_right
    assert u_hist.shape == (M, nb_total, nu), f'u_hist shape mismatch'
    assert y_hist.shape == (M, na_total, ny), f'y_hist shape mismatch'
    assert u_future.shape == (M, nf, nu), f'u_future shape mismatch'
    assert y_future.shape == (M, nf, ny), f'y_future shape mismatch'
    print('   Shapes OK\n')

    # 6. Build + forward pass for both NX_ANN values
    for nx_ann in [0, 2]:
        nxd = NX_PHYS + nx_ann
        print(f'6. NX_ANN={nx_ann}: building encoder + interconnect...')

        encoder = build_encoder(norm, nx_ann)
        interconnect = build_interconnect(norm, nx_ann)

        n_enc = sum(p.numel() for p in encoder.parameters())
        n_ic = sum(p.numel() for p in interconnect.parameters())
        print(f'   Encoder params: {n_enc}')
        print(f'   Interconnect params: {n_ic}')

        # Small batch forward pass
        B = 4
        ub = torch.tensor(u_hist[:B], dtype=DTYPE_PT)
        yb = torch.tensor(y_hist[:B], dtype=DTYPE_PT)
        ufb = torch.tensor(u_future[:B], dtype=DTYPE_PT)
        yfb = torch.tensor(y_future[:B], dtype=DTYPE_PT)

        print(f'   Forward pass (batch={B})...')
        encoder.train()
        interconnect.train()
        y_hat, x0 = rollout(encoder, interconnect, ub, yb, ufb)
        print(f'   x0 shape: {x0.shape} (expected ({B}, {nxd}))')
        print(f'   y_hat shape: {y_hat.shape} (expected ({B}, {nf}, {ny}))')
        assert x0.shape == (B, nxd), f'x0 shape mismatch: {x0.shape}'
        assert y_hat.shape == (B, nf, ny), f'y_hat shape mismatch: {y_hat.shape}'

        # Backward pass
        print(f'   Backward pass...')
        loss = nn.MSELoss()(y_hat, yfb)
        loss.backward()
        print(f'   Loss: {loss.item():.4e}')

        # Check gradients exist
        enc_grads = sum(1 for p in encoder.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        ic_grads = sum(1 for p in interconnect.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        enc_total = sum(1 for p in encoder.parameters())
        ic_total = sum(1 for p in interconnect.parameters())
        print(f'   Encoder: {enc_grads}/{enc_total} params have nonzero grad')
        print(f'   Interconnect: {ic_grads}/{ic_total} params have nonzero grad')

        if nx_ann == 2:
            # Check augmented state ANN in encoder gets gradient
            ann_grads = sum(1 for p in encoder.ann.parameters()
                           if p.grad is not None and p.grad.abs().sum() > 0)
            ann_total = sum(1 for _ in encoder.ann.parameters())
            print(f'   Encoder ANN (aug states): {ann_grads}/{ann_total} params have nonzero grad')
            if ann_grads == 0:
                print('   Expected at step 0: zero-init ANN blocks break gradient chain.')
                print('   Verifying gradient flows after a few optimizer steps...')

                # Do 5 optimizer steps, then check again
                all_params = list(encoder.parameters()) + list(interconnect.parameters())
                optimizer = torch.optim.Adam(all_params, lr=1e-3)
                for step in range(5):
                    optimizer.zero_grad()
                    y_hat2, _ = rollout(encoder, interconnect, ub, yb, ufb)
                    loss2 = nn.MSELoss()(y_hat2, yfb)
                    loss2.backward()
                    optimizer.step()

                # Check again after 5 steps
                optimizer.zero_grad()
                y_hat3, _ = rollout(encoder, interconnect, ub, yb, ufb)
                loss3 = nn.MSELoss()(y_hat3, yfb)
                loss3.backward()

                ann_grads_after = sum(1 for p in encoder.ann.parameters()
                                     if p.grad is not None and p.grad.abs().sum() > 0)
                ic_grads_after = sum(1 for p in interconnect.parameters()
                                    if p.grad is not None and p.grad.abs().sum() > 0)
                print(f'   After 5 steps:')
                print(f'     Encoder ANN: {ann_grads_after}/{ann_total} params have nonzero grad')
                print(f'     Interconnect: {ic_grads_after}/{ic_total} params have nonzero grad')
                if ann_grads_after > 0:
                    print('     OK: Augmented states receive gradient after warmup.')
                else:
                    print('     FAIL: Augmented states STILL have no gradient after 5 steps!')
                    sys.exit(1)
            else:
                print('   OK: Augmented states receive gradient through I/O loss.')

        # Check no NaN
        assert not torch.isnan(y_hat).any(), 'NaN in y_hat!'
        assert not torch.isnan(loss), 'NaN in loss!'
        print(f'   No NaN detected.')
        print()

    # 7. Analytical baseline
    print('7. Analytical baseline...')
    x_ana = compute_analytical_baseline(val_y, norm)
    print(f'   Shape: {x_ana.shape} (expected ({M}, 6))')
    assert x_ana.shape == (M, 6), f'Shape mismatch: {x_ana.shape}'
    print('   OK\n')

    print('=== ALL SMOKE TESTS PASSED ===')


if __name__ == '__main__':
    main()
