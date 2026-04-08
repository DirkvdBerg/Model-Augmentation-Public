"""
lfr_fit_system.py
-----------------
Thin subclass of SSE_Interconnect that extends the training loss to include
param_loss() from any block that exposes it (D-032).

Why a subclass and not an edit to Jan's code:
    model_augmentation/ is read-only (CLAUDE.md). SSE_Interconnect.loss()
    calls param_loss() only on hard-coded isinstance checks for Jan's own
    block types. ParameterizedLFRBlock is not one of those types.
    Overriding loss() here with a generic hasattr check is the minimal,
    non-invasive extension -- Jan's code stays untouched.

What changes vs SSE_Interconnect.loss():
    The isinstance block-type sweep is replaced with:

        loss_theta = 0
        for m in self.hfn.connected_blocks:
            if hasattr(m, 'param_loss'):
                loss_theta = loss_theta + m.param_loss()

    This covers ParameterizedLFRBlock and, as a side effect, also covers
    Jan's Parameterized_Linear_State_Block and Parameterized_Linear_Output_Block
    (both have param_loss()). The only Jan block NOT covered is
    Parameterized_MSD_State_Block, which computes its regularization inline
    rather than via param_loss() -- this is irrelevant for the gantry pipeline.

    All other loss logic (encoder, simulation MSE loop, normalization) is
    inherited unchanged from SSE_Interconnect.

Usage
-----
    from lpv_lfr_baseline.lfr_fit_system import LFRFitSystem

    fit_sys = LFRFitSystem(interconnect=ic, na=na, nb=nb)
    fit_sys.init_model(sys_data=train_data, device=device, auto_fit_norm=True)
    fit_sys.fit(train_sys_data=train_data, val_sys_data=val_data, ...)
"""

import torch
import torch.nn as nn

from model_augmentation.fit_systems.interconnect import SSE_Interconnect


class LFRFitSystem(SSE_Interconnect):
    """
    SSE_Interconnect subclass with generic param_loss() support.

    Drop-in replacement for SSE_Interconnect. Only loss() is overridden.
    All other methods (init_model, fit, save_system, apply_experiment, ...)
    are inherited from SSE_Interconnect unchanged.
    """

    def loss(self, uhist, yhist, ufuture, yfuture, **Loss_kwargs):
        """
        Simulation MSE + param_loss() from all blocks that expose it.

        Identical to SSE_Interconnect.loss() except the isinstance sweep
        is replaced by a generic hasattr(m, 'param_loss') check.
        """
        # --- Encoder: initialise state from history ---
        x = self.encoder(uhist, yhist)

        # --- Simulation MSE over the future horizon ---
        errors = []
        for y, u in zip(torch.transpose(yfuture, 0, 1),
                        torch.transpose(ufuture, 0, 1)):
            yhat, x = self.hfn(x, u)
            errors.append(nn.functional.mse_loss(y, yhat))
        loss_MSE = torch.mean(torch.stack(errors))

        # --- Parameter regularization from all blocks that expose it ---
        loss_theta = 0
        for m in self.hfn.connected_blocks:
            if hasattr(m, 'param_loss'):
                loss_theta = loss_theta + m.param_loss()

        return loss_MSE + loss_theta


# ----------------------------------------------------------------------
# Smoke test
# (run as: conda run -n GraduationProject python -m lpv_lfr_baseline.lfr_fit_system)
# ----------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    import os
    import numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from model_augmentation.utils.utils import selection_matrix
    from model_augmentation.fit_systems.interconnect import Interconnect
    from lpv_lfr_baseline.lfr_param_block import ParameterizedLFRBlock
    from lpv_lfr_baseline.physics import P

    # ------------------------------------------------------------------
    # Build minimal interconnect with ParameterizedLFRBlock
    # (same wiring as build_baseline_interconnect in test_jan_compat.py)
    # ------------------------------------------------------------------
    ic = Interconnect(nx=6, nu=3, ny=3, debugging=False)
    lfr_block = ParameterizedLFRBlock(RMSE_baseline=1.0)
    ic.add_block(lfr_block)

    ic.connect_signals('x', lfr_block)
    ic.connect_signals('u', lfr_block)

    S_xp = selection_matrix(np.arange(6), 18)
    ic.connect_signals(lfr_block, 'xp', connection_matrix=S_xp)

    S_y = (P.T @ selection_matrix(np.arange(3), 18).double()).float()
    ic.connect_signals(lfr_block, 'y', connection_matrix=S_y)

    # ------------------------------------------------------------------
    # Check 1 -- LFRFitSystem is a subclass of SSE_Interconnect
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Check 1: LFRFitSystem is subclass of SSE_Interconnect")
    print("=" * 60)

    fit_sys = LFRFitSystem(interconnect=ic, na=2, nb=2)

    is_subclass = isinstance(fit_sys, SSE_Interconnect)
    loss_overridden = type(fit_sys).loss is not SSE_Interconnect.loss
    print(f"  isinstance(fit_sys, SSE_Interconnect) : {is_subclass}")
    print(f"  loss() is overridden                  : {loss_overridden}")
    status_1 = is_subclass and loss_overridden
    print(f"\nCheck 1: {'PASS' if status_1 else 'FAIL'}")

    # ------------------------------------------------------------------
    # Check 2 -- ParameterizedLFRBlock found in connected_blocks
    #            and exposes param_loss()
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Check 2: ParameterizedLFRBlock found with param_loss()")
    print("=" * 60)

    blocks_with_param_loss = [
        m for m in fit_sys.hfn.connected_blocks
        if hasattr(m, 'param_loss')
    ]
    found_ok   = len(blocks_with_param_loss) == 1
    is_lfr     = isinstance(blocks_with_param_loss[0], ParameterizedLFRBlock) if found_ok else False

    print(f"  Blocks with param_loss() : {len(blocks_with_param_loss)}  (expected 1)")
    print(f"  Block is ParameterizedLFRBlock : {is_lfr}")

    # Verify param_loss() returns a finite scalar
    loss_val   = blocks_with_param_loss[0].param_loss() if found_ok else None
    is_scalar  = loss_val is not None and loss_val.shape == torch.Size([])
    is_finite  = loss_val is not None and torch.isfinite(loss_val).item()
    print(f"  param_loss() is scalar   : {is_scalar}  value={loss_val.item():.4e}" if is_scalar else
          f"  param_loss() is scalar   : False")
    print(f"  param_loss() is finite   : {is_finite}")

    status_2 = found_ok and is_lfr and is_scalar and is_finite
    print(f"\nCheck 2: {'PASS' if status_2 else 'FAIL'}")

    # ------------------------------------------------------------------
    # Check 3 -- param_loss() is included in loss_theta:
    #            perturbing log_params changes the total loss
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Check 3: Perturbing log_params changes loss_theta in loss()")
    print("=" * 60)

    # We test the param_loss path directly (full loss() requires init_model).
    # Sum param_loss from all blocks -- same logic as loss() override.
    def _collect_theta_loss(interconnect):
        loss_theta = 0
        for m in interconnect.connected_blocks:
            if hasattr(m, 'param_loss'):
                loss_theta = loss_theta + m.param_loss()
        return loss_theta

    loss_at_init = _collect_theta_loss(ic).item()

    # Perturb log_params slightly
    with torch.no_grad():
        lfr_block.log_params[0] += 0.05

    loss_perturbed = _collect_theta_loss(ic).item()

    # Restore
    with torch.no_grad():
        lfr_block.log_params[0] -= 0.05

    at_init_small  = loss_at_init   < 1e-20
    perturbed_larger = loss_perturbed > loss_at_init

    print(f"  loss_theta at init     : {loss_at_init:.4e}  (expect ~0)")
    print(f"  loss_theta perturbed   : {loss_perturbed:.6f}  (expect > 0)")
    print(f"  At init ~0             : {at_init_small}")
    print(f"  Perturbed > init       : {perturbed_larger}")
    status_3 = at_init_small and perturbed_larger
    print(f"\nCheck 3: {'PASS' if status_3 else 'FAIL'}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    all_pass = all([status_1, status_2, status_3])
    for label, ok in [
        ("Check 1 (subclass + loss overridden)",        status_1),
        ("Check 2 (block found, param_loss finite)",    status_2),
        ("Check 3 (param_loss changes with params)",    status_3),
    ]:
        print(f"  {label:<45} {'PASS' if ok else 'FAIL'}")
    print()
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 60)
