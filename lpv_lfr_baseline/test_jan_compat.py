"""
test_jan_compat.py
------------------
Integration tests for LFRBaselineBlock inside Jan's Interconnect framework.

These tests actually import model_augmentation/ and wire the LFR block into
a real Interconnect — verifying end-to-end compatibility before any training.

Interconnect wiring (baseline only, no augmentation):
    Signals:
        0: x  (nx=6, logical state)
        1: u  (nu=3, stage forces)
        2: LFRBaselineBlock (nz=9 in, nw=18 out)

    Connections:
        x ──────────────────┐
                            ├──> LFRBaselineBlock.z_in  (9,)
        u ──────────────────┘

        LFRBaselineBlock.w_out[:6]   ──> xp   (x_next, logical)
        LFRBaselineBlock.w_out[:3]   ──> y    (position proxy, 3 channels)

    Block output slot map (nw=18):
        w_out[:, 0:6,  :]  = x_next  (logical state)
        w_out[:, 6:12, :]  = z_lfr   (LFR latent — augmentation connects here)
        w_out[:, 12:,  :]  = w_lfr   (LFR latent — augmentation connects here)

Augmented Interconnect wiring (checks C–D):
    Signals:
        0: x  (nx=6)
        1: u  (nu=3)
        2: LFRBaselineBlock   (nz=9,  nw=18)
        3: Static_ANN_Block   (nz=6,  nw=6)   — augmentation

    Additional connections (on top of baseline):
        LFRBaselineBlock.w_out[6:12]  ──> Static_ANN_Block.z_in   (z_lfr → aug)
        Static_ANN_Block.w_out        ──> xp                       (additive correction)

Checks
------
    Check 1 — isinstance and nz/nw:
        LFRBaselineBlock inherits from Jan's Block with correct dimensions.

    Check 2 — detect_algebraic_loop:
        Interconnect.init_forward() must not raise — Y is not an external signal.

    Check 3 — Interconnect forward pass shapes:
        y shape (batch, ny), xp shape (batch, nx).

    Check 4 — Gradient flows through Interconnect to block input:
        xp.sum().backward() must give x.grad != None.

    Check 5 — z_lfr slot accessible via selection matrix:
        Simulate what adding an augmentation block would do: extract
        w_out[:, 6:12, :] via a selection matrix and verify grad_fn survives.
        This is the structural check that augmentation routing will work.

    Check A — xp content matches rk4_step:
        Values in xp after ic.forward() equal x_next from a direct rk4_step
        call at the same inputs (to float32 tolerance).  Shape-only checks
        (Check 3) miss wrong slot selection.

    Check B — z_lfr slot contains correct latent signal:
        w_out[:, 6:12, 0] matches z from lfr_forward at float32 tolerance.
        Catches z_lfr / w_lfr slot swap — both are 6-dim, silent error.

    Check C — Augmentation weight gradient end-to-end:
        Wire Static_ANN_Block(nz=6, nw=6) to z_lfr slot via Interconnect.
        xp.sum().backward() must give non-None grad to aug block parameters.

    Check D — Additive correction changes xp:
        Zero-init aug produces xp == baseline.  Non-zero aug weight produces
        xp != baseline.  Confirms routing is live, not silently discarded.

    Check E — Multi-step BPTT (3 steps):
        x0 → xp1 → xp2 → xp3; gradient must flow back to x0.
        Single-step BPTT (Check 4) misses truncation at step boundaries.

    Check F — Trainable physical parameters: gradient via G matrix entries:
        Part 1: lfr_forward with M0 as nn.Parameter (static G) gives
                M0.grad != None via the linalg.solve path.
        Part 2: lfr_forward with G rebuilt from M0 (dynamic G) gives a
                strictly larger M0 grad norm — the Ax/Bw/Bu paths also
                contribute.  Proves lfr_block.py must call
                build_G_matrix() inside forward() for full identifiability.

Run as:
    conda run -n GraduationProject python -m lpv_lfr_baseline.test_jan_compat
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch

from model_augmentation.fit_systems.blocks import Block, Static_ANN_Block
from model_augmentation.fit_systems.interconnect import Interconnect
from model_augmentation.utils.utils import detect_algebraic_loop, selection_matrix

from lpv_lfr_baseline.lfr_block import LFRBaselineBlock
from lpv_lfr_baseline.lfr_forward import lfr_forward
from lpv_lfr_baseline.lfr_simulate import rk4_step
from lpv_lfr_baseline.lfr_matrices import G as G_module, build_G_matrix
from lpv_lfr_baseline.physics import M0, M1, M2, K, C, P, ts


def build_baseline_interconnect(debugging=False):
    """
    Wire a minimal Interconnect containing only the LFR baseline block.

    nx=6 (logical state), nu=3 (stage forces), ny=3 (position proxy).
    """
    ic = Interconnect(nx=6, nu=3, ny=3, debugging=debugging)

    lfr_block = LFRBaselineBlock()
    ic.add_block(lfr_block)

    # Input wiring: z_in = cat([x (6), u (3)])
    ic.connect_signals('x', lfr_block)   # all 6 state dims → first 6 of z_in
    ic.connect_signals('u', lfr_block)   # all 3 input dims → last 3 of z_in

    # Output → xp: select x_next (first 6 of nw=18) → global state xp
    S_xp = selection_matrix(np.arange(6), 18)    # (6, 18)
    ic.connect_signals(lfr_block, 'xp', connection_matrix=S_xp)

    # Output → y: select first 3 of x_next as position proxy
    S_y = selection_matrix(np.arange(3), 18)     # (3, 18)
    ic.connect_signals(lfr_block, 'y', connection_matrix=S_y)

    return ic, lfr_block


def build_augmented_interconnect(debugging=False):
    """
    Wire LFR baseline block + Static_ANN_Block augmentation.

    Augmentation receives z_lfr (w_out[6:12]) and adds its output to xp.

    Block indices:
        2: LFRBaselineBlock   (nz=9, nw=18)
        3: Static_ANN_Block   (nz=6, nw=6)
    """
    ic = Interconnect(nx=6, nu=3, ny=3, debugging=debugging)

    lfr_block = LFRBaselineBlock()
    aug_block = Static_ANN_Block(nz=6, nw=6)
    ic.add_block(lfr_block)
    ic.add_block(aug_block)

    # LFR wiring (same as baseline)
    ic.connect_signals('x', lfr_block)
    ic.connect_signals('u', lfr_block)
    S_xp = selection_matrix(np.arange(6),    18).float()   # (6, 18)
    S_y  = selection_matrix(np.arange(3),    18).float()   # (3, 18)
    ic.connect_signals(lfr_block, 'xp', connection_matrix=S_xp)
    ic.connect_signals(lfr_block, 'y',  connection_matrix=S_y)

    # z_lfr (indices 6:12 of LFR w_out) → aug block input
    S_z_to_aug = selection_matrix(np.arange(6, 12), 18).float()   # (6, 18)
    ic.connect_signals(lfr_block, aug_block, connection_matrix=S_z_to_aug)

    # aug output → xp (additive; eye(6) connection matrix)
    S_aug_to_xp = selection_matrix(np.arange(6), 6).float()       # (6, 6) = I6
    ic.connect_signals(aug_block, 'xp', connection_matrix=S_aug_to_xp)

    return ic, lfr_block, aug_block


if __name__ == '__main__':

    results = {}

    # ------------------------------------------------------------------
    # Check 1 — isinstance and nz / nw
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Check 1: LFRBaselineBlock isinstance and nz/nw")
    print("=" * 60)

    lfr_block = LFRBaselineBlock()

    is_block   = isinstance(lfr_block, Block)
    nz_ok      = lfr_block.nz == 9
    nw_ok      = lfr_block.nw == 18

    print(f"  isinstance(block, Jan Block) : {is_block}")
    print(f"  nz == 9                      : {nz_ok}  (got {lfr_block.nz})")
    print(f"  nw == 18                     : {nw_ok}  (got {lfr_block.nw})")
    status = is_block and nz_ok and nw_ok
    results['Check 1 (isinstance, nz/nw)'] = status
    print(f"\nCheck 1: {'PASS' if status else 'FAIL'}")

    # ------------------------------------------------------------------
    # Check 2 — detect_algebraic_loop does not trigger
    #
    # Interconnect.init_forward() asserts not detect_algebraic_loop(...).
    # Triggering the first forward pass exercises init_forward().
    # If Y were routed as an external signal, a loop would exist here.
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Check 2: detect_algebraic_loop does not trigger")
    print("=" * 60)

    ic, _ = build_baseline_interconnect()

    x0 = torch.zeros(1, 6)
    u0 = torch.zeros(1, 3)

    loop_raised = False
    try:
        with torch.no_grad():
            ic.forward(x0.float(), u0.float())   # triggers init_forward on first call
    except AssertionError:
        loop_raised = True
    except Exception as e:
        print(f"  Unexpected error during init_forward: {e}")
        loop_raised = True

    status = not loop_raised
    results['Check 2 (no algebraic loop)'] = status
    print(f"  init_forward completed without algebraic loop : {status}")
    print(f"\nCheck 2: {'PASS' if status else 'FAIL'}")

    # ------------------------------------------------------------------
    # Check 3 — Interconnect forward pass: output shapes correct
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Check 3: Interconnect forward pass — output shapes")
    print("=" * 60)

    ic, _ = build_baseline_interconnect()

    batch = 4
    x_in = torch.zeros(batch, 6).float()
    x_in[:, 2] = 0.3   # Y=0.3 m initial position — non-trivial scheduling
    u_in = torch.zeros(batch, 3).float()

    with torch.no_grad():
        y_out, xp_out = ic.forward(x_in, u_in)

    y_shape_ok  = tuple(y_out.shape)  == (batch, 3)
    xp_shape_ok = tuple(xp_out.shape) == (batch, 6)
    print(f"  y  shape (batch=4, ny=3) : {y_shape_ok}   got {tuple(y_out.shape)}")
    print(f"  xp shape (batch=4, nx=6) : {xp_shape_ok}  got {tuple(xp_out.shape)}")
    status = y_shape_ok and xp_shape_ok
    results['Check 3 (forward shapes)'] = status
    print(f"\nCheck 3: {'PASS' if status else 'FAIL'}")

    # ------------------------------------------------------------------
    # Check 4 — Gradient flows through Interconnect to block input
    #
    # xp.sum().backward() must give x.grad != None.
    # This is the core requirement for BPTT during training.
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Check 4: Gradient flows through Interconnect (BPTT path)")
    print("=" * 60)

    ic, _ = build_baseline_interconnect()

    x_grad = torch.zeros(1, 6, requires_grad=True).float()
    x_grad.data[:, 2] = 0.3
    u_in = torch.zeros(1, 3).float()

    y_out, xp_out = ic.forward(x_grad, u_in)
    xp_out.sum().backward()

    grad_ok = x_grad.grad is not None
    print(f"  x.grad is not None after xp.backward() : {grad_ok}")
    if grad_ok:
        print(f"  x.grad norm : {x_grad.grad.norm().item():.6e}")
    results['Check 4 (gradient BPTT)'] = grad_ok
    print(f"\nCheck 4: {'PASS' if grad_ok else 'FAIL'}")

    # ------------------------------------------------------------------
    # Check 5 — z_lfr slot accessible via selection matrix
    #
    # Augmentation blocks connect to z_lfr = w_out[:, 6:12, :].
    # Verify a selection matrix can extract this slice while preserving
    # grad_fn — a None grad_fn here would silently break augmentation training.
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Check 5: z_lfr slot accessible via selection matrix (aug routing)")
    print("=" * 60)

    x_in = torch.zeros(1, 6, requires_grad=True).float()
    x_in.data[:, 2] = 0.3
    u_in = torch.zeros(1, 3).float()

    # Call block directly (Interconnect routes via matmul; we test slot contract here)
    z_in = torch.cat([x_in, u_in], dim=1).unsqueeze(-1)   # (1, 9, 1)
    w_out = lfr_block.forward(z_in)                        # (1, 18, 1)

    # Simulate what connect_signals does: selection_matrix @ w_out
    S_z = selection_matrix(np.arange(6, 12), 18).float()   # (6, 18)
    z_lfr_extracted = torch.matmul(S_z, w_out)             # (1, 6, 1) — augmentation input

    has_grad_fn = z_lfr_extracted.grad_fn is not None
    print(f"  z_lfr extracted via S_z @ w_out : shape {tuple(z_lfr_extracted.shape)}")
    print(f"  grad_fn preserved               : {has_grad_fn}")
    results['Check 5 (z_lfr aug routing)'] = has_grad_fn
    print(f"\nCheck 5: {'PASS' if has_grad_fn else 'FAIL'}")

    # ------------------------------------------------------------------
    # Check A — xp content matches rk4_step
    #
    # Check 3 only verified shapes.  Here we compare the actual values in
    # xp against a direct rk4_step call at the same inputs.
    # Catches: wrong slot selection in S_xp, missing P-transform.
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Check A: xp content matches rk4_step (value correctness)")
    print("=" * 60)

    ic, _ = build_baseline_interconnect()

    x_test_a = torch.zeros(1, 6).float()
    x_test_a[:, 2] = 0.3   # non-trivial Y for scheduling
    u_test_a = torch.tensor([[10.0, -5.0, 3.0]]).float()

    with torch.no_grad():
        _, xp_ic = ic.forward(x_test_a, u_test_a)   # (1, 6)

    # Direct rk4_step reference (float64)
    u_logical_a = u_test_a.double() @ P.T            # (1, 3)
    with torch.no_grad():
        x_next_ref, _, _, _ = rk4_step(
            x_test_a.double(), u_logical_a,
            G_module, M0, M1, M2, K, C, ts,
        )

    err_a = (xp_ic.double() - x_next_ref).abs().max().item()
    tol_a = 1e-5   # float32 rounding from lfr_block dtype cast
    status = err_a < tol_a
    print(f"  max|xp_ic - x_next_ref| : {err_a:.2e}   (tol {tol_a:.0e})")
    results['Check A (xp value correctness)'] = status
    print(f"\nCheck A: {'PASS' if status else 'FAIL'}")

    # ------------------------------------------------------------------
    # Check B — z_lfr and w_lfr slot content correct
    #
    # w_out[:, 6:12, 0] must match z from lfr_forward; likewise w_lfr.
    # Catches silent slot swap (z and w are both 6-dim — easy mix-up).
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Check B: z_lfr / w_lfr slot content matches lfr_forward")
    print("=" * 60)

    x_test_b = torch.tensor([[0.05, 0.01, 0.30, 0.02, -0.01, 0.05]]).float()
    u_test_b = torch.tensor([[10.0, -5.0, 3.0]]).float()
    z_in_b   = torch.cat([x_test_b, u_test_b], dim=1).unsqueeze(-1)   # (1, 9, 1)

    with torch.no_grad():
        w_out_b = lfr_block.forward(z_in_b)   # (1, 18, 1)

    # Reference from lfr_forward (float64)
    u_logical_b = u_test_b.double() @ P.T                            # (1, 3)
    Y_b         = x_test_b[:, 2].double()                            # (1,)
    with torch.no_grad():
        _, z_ref, w_ref, _ = lfr_forward(
            x_test_b.double(), u_logical_b, Y_b,
            G_module, M0, M1, M2, K, C,
        )

    z_lfr_slot = w_out_b[:, 6:12, 0].double()    # (1, 6)
    w_lfr_slot = w_out_b[:, 12:,  0].double()    # (1, 6)

    tol_b = 1e-6   # float32 rounding
    err_z = (z_lfr_slot - z_ref).abs().max().item()
    err_w = (w_lfr_slot - w_ref).abs().max().item()
    z_ok  = err_z < tol_b
    w_ok  = err_w < tol_b
    print(f"  max|z_lfr_slot - z_ref| : {err_z:.2e}   (tol {tol_b:.0e})   {'OK' if z_ok else 'FAIL'}")
    print(f"  max|w_lfr_slot - w_ref| : {err_w:.2e}   (tol {tol_b:.0e})   {'OK' if w_ok else 'FAIL'}")
    status = z_ok and w_ok
    results['Check B (z_lfr/w_lfr slot content)'] = status
    print(f"\nCheck B: {'PASS' if status else 'FAIL'}")

    # ------------------------------------------------------------------
    # Check C — Augmentation weight gradient end-to-end
    #
    # Wire Static_ANN_Block(nz=6, nw=6) to receive z_lfr via Interconnect,
    # add its output additively to xp.  After xp.sum().backward(), at
    # least one aug parameter must have a non-None gradient.
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Check C: Augmentation weight gradient end-to-end")
    print("=" * 60)

    ic_aug, _, aug_block = build_augmented_interconnect()

    x_in_c = torch.zeros(1, 6).float()
    x_in_c[:, 2] = 0.3
    u_in_c = torch.tensor([[10.0, -5.0, 3.0]]).float()

    _, xp_c = ic_aug.forward(x_in_c, u_in_c)
    xp_c.sum().backward()

    aug_grads = [p.grad for p in aug_block.parameters() if p.grad is not None]
    aug_grad_ok = len(aug_grads) > 0
    if aug_grad_ok:
        total_norm = sum(g.norm().item() ** 2 for g in aug_grads) ** 0.5
        print(f"  aug parameters with grad  : {len(aug_grads)}")
        print(f"  combined grad norm        : {total_norm:.6e}")
    else:
        print(f"  NO aug parameters received a gradient")
    results['Check C (aug weight gradient)'] = aug_grad_ok
    print(f"\nCheck C: {'PASS' if aug_grad_ok else 'FAIL'}")

    # ------------------------------------------------------------------
    # Check D — Additive correction changes xp
    #
    # Zero-init aug → xp equals baseline (augmentation does nothing).
    # Non-zero final-layer weight → xp differs from baseline.
    # Confirms the aug→xp routing is live, not silently discarded.
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Check D: Additive correction changes xp (routing live)")
    print("=" * 60)

    x_in_d = torch.zeros(1, 6).float()
    x_in_d[:, 2] = 0.3
    u_in_d = torch.tensor([[10.0, -5.0, 3.0]]).float()

    # Baseline xp (no augmentation)
    ic_base_d, _ = build_baseline_interconnect()
    with torch.no_grad():
        _, xp_base_d = ic_base_d.forward(x_in_d, u_in_d)

    # Augmented — zero-init weights → should equal baseline
    ic_aug_d, _, aug_block_d = build_augmented_interconnect()
    with torch.no_grad():
        _, xp_aug_zero = ic_aug_d.forward(x_in_d, u_in_d)

    zero_effect_ok = (xp_aug_zero - xp_base_d).abs().max().item() < 1e-6

    # Perturb the final linear layer of the ANN → non-zero aug output
    final_layer = aug_block_d.net.net[-1]   # last element of inner nn.Sequential
    final_layer.weight.data.fill_(0.1)

    # Re-build to reset Interconnect init (connection matrices are re-init on first forward)
    ic_aug_d2, _, aug_block_d2 = build_augmented_interconnect()
    final_layer2 = aug_block_d2.net.net[-1]
    final_layer2.weight.data.fill_(0.1)
    with torch.no_grad():
        _, xp_aug_nonzero = ic_aug_d2.forward(x_in_d, u_in_d)

    nonzero_effect_ok = (xp_aug_nonzero - xp_base_d).abs().max().item() > 1e-6

    print(f"  Zero-aug  |xp_aug - xp_base| : {(xp_aug_zero - xp_base_d).abs().max().item():.2e}   {'OK (no effect)' if zero_effect_ok else 'FAIL'}")
    print(f"  Non-zero  |xp_aug - xp_base| : {(xp_aug_nonzero - xp_base_d).abs().max().item():.2e}   {'OK (effect present)' if nonzero_effect_ok else 'FAIL'}")
    status = zero_effect_ok and nonzero_effect_ok
    results['Check D (additive correction live)'] = status
    print(f"\nCheck D: {'PASS' if status else 'FAIL'}")

    # ------------------------------------------------------------------
    # Check E — Multi-step BPTT (3 steps)
    #
    # x0 → xp1 → xp2 → xp3.  Gradient must reach x0 through all steps.
    # Single-step BPTT (Check 4) misses truncation at step boundaries.
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Check E: Multi-step BPTT — gradient flows through 3 steps")
    print("=" * 60)

    ic_e, _ = build_baseline_interconnect()

    x0_e = torch.zeros(1, 6, requires_grad=True).float()
    x0_e.data[:, 2] = 0.3
    u0_e = torch.zeros(1, 3).float()

    _, xp1_e = ic_e.forward(x0_e,  u0_e)
    _, xp2_e = ic_e.forward(xp1_e, u0_e)
    _, xp3_e = ic_e.forward(xp2_e, u0_e)
    xp3_e.sum().backward()

    grad_ok_e    = x0_e.grad is not None and x0_e.grad.norm().item() > 0
    graph_ok_xp1 = xp1_e.grad_fn is not None
    graph_ok_xp2 = xp2_e.grad_fn is not None

    print(f"  x0.grad is not None after 3-step backward : {x0_e.grad is not None}")
    if x0_e.grad is not None:
        print(f"  x0.grad norm                              : {x0_e.grad.norm().item():.6e}")
    print(f"  xp1 has grad_fn (graph not truncated)     : {graph_ok_xp1}")
    print(f"  xp2 has grad_fn (graph not truncated)     : {graph_ok_xp2}")
    status = grad_ok_e and graph_ok_xp1 and graph_ok_xp2
    results['Check E (multi-step BPTT)'] = status
    print(f"\nCheck E: {'PASS' if status else 'FAIL'}")

    # ------------------------------------------------------------------
    # Check F — Trainable physical parameters: gradient path through G
    #
    # Part 1 (static G): M0 as nn.Parameter + module-level G.
    #   Gradient flows via linalg.solve (M_Y → v → z/w → xdot).
    #   M0.grad must be non-None.
    #
    # Part 2 (dynamic G): G rebuilt from M0 inside forward context.
    #   Both solve-path AND G-entry paths contribute.
    #   M0.grad norm must be strictly larger than Part 1.
    #   This is the required design for lfr_block.py when params are
    #   trainable: call build_G_matrix() inside forward(), not at init.
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Check F: Trainable physical parameters — gradient via G matrix")
    print("=" * 60)

    x_f_test  = torch.tensor([[0.05, 0.01, 0.30, 0.02, -0.01, 0.05]], dtype=torch.float64)
    u_f_stage = torch.tensor([[10.0, -5.0, 3.0]], dtype=torch.float64)
    u_f_log   = u_f_stage @ P.T        # (1, 3)  logical coords
    Y_f       = x_f_test[:, 2]         # (1,)

    # --- Part 1: static G (module-level, precomputed from fixed M0 at import) ---
    M0_param1 = torch.nn.Parameter(M0.clone())
    xdot1, _, _, _ = lfr_forward(x_f_test, u_f_log, Y_f, G_module, M0_param1, M1, M2, K, C)
    xdot1.sum().backward()

    part1_ok = M0_param1.grad is not None
    norm1    = M0_param1.grad.norm().item() if part1_ok else 0.0
    print(f"  Part 1 (static G)  M0.grad is not None : {part1_ok}  norm = {norm1:.6e}")

    # --- Part 2: dynamic G (built from M0_param inside forward context) ---
    M0_param2 = torch.nn.Parameter(M0.clone())
    G_dynamic = build_G_matrix(M0_param2, M1, M2, K, C)
    xdot2, _, _, _ = lfr_forward(x_f_test, u_f_log, Y_f, G_dynamic, M0_param2, M1, M2, K, C)
    xdot2.sum().backward()

    part2_ok   = M0_param2.grad is not None
    norm2      = M0_param2.grad.norm().item() if part2_ok else 0.0
    larger_ok  = norm2 > norm1
    print(f"  Part 2 (dynamic G) M0.grad is not None : {part2_ok}  norm = {norm2:.6e}")
    print(f"  Dynamic G norm > static G norm (more paths active): {larger_ok}")
    print(f"  (Ratio: {norm2/norm1:.2f}x)" if norm1 > 0 else "")
    print(f"  Design note: lfr_block.py must call build_G_matrix() inside forward()")
    print(f"               for M0/M1/M2/K/C to be fully identifiable from data.")
    status = part1_ok and part2_ok and larger_ok
    results['Check F (trainable param grad)'] = status
    print(f"\nCheck F: {'PASS' if status else 'FAIL'}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    all_pass = all(results.values())
    for name, passed in results.items():
        print(f"  {name:45s} {'PASS' if passed else 'FAIL'}")
    print()
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 60)
