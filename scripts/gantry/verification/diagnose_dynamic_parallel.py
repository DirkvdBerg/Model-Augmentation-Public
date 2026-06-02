"""
diagnose_dynamic_parallel.py
-----------------------------
15 tests validating the dynamic parallel augmentation structure before
implementing gantry_interconnect_dynamic.py.

Physical motivation: NX_ANN=2 comes from adding a hidden MSD on the Y-beam
(delta_a, vdelta_a from additional_state_lagrangian.m). The ANN learns the
nonlinear coupling force without explicit parametrisation.

Groups:
  Group 1 (1-3):   selection / expansion matrix utilities
  Group 2 (4-8):   forward-pass shapes and signal routing
  Group 3 (9-10):  zero-init ANN behaviour (silent at init = physics-only)
  Group 4 (11):    encoder output is 8-D (NX_TOTAL)
  Group 5 (12-13): gradient flow and BPTT smoke test (no NaN, no crash)
  Group 6 (14-15): ANN latent-state dynamics (states evolve; perturbation propagates)

Run from project root:
    conda run -n GraduationProject python scripts/gantry/verification/diagnose_dynamic_parallel.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import numpy as np
import torch
import deepSI

from model_augmentation.fit_systems.interconnect import Interconnect, SSE_Interconnect
from model_augmentation.fit_systems.blocks import (
    Gantry_State_Block,
    Linear_Output_Block,
    Static_ANN_Block,
)
from model_augmentation.systems.gantry_ss import Cd, Dd
from model_augmentation.utils.utils import selection_matrix, expansion_matrix
from model_augmentation.utils.torch_nets import zero_init_feed_forward_nn

# ── Shared constants ───────────────────────────────────────────────────────────
NX_PHYS  = 6          # physical states: q1, q2, q3, dq1, dq2, dq3
NX_ANN   = 2          # ANN latent states: delta_a (disp), vdelta_a (vel)  [additional_state_lagrangian.m]
NX_TOTAL = NX_PHYS + NX_ANN   # = 8
NU       = 3          # inputs:  FX1, FX2, FY
NY       = 3          # outputs: X1, X2, Y
PHY_IX   = np.arange(NX_PHYS)   # [0,1,2,3,4,5]
Y_OP     = 0.3        # frozen Y operating point [m]

# Identity normalisation (all stds = 1) — keeps tests independent of real data.
STD_X = np.ones((NX_PHYS, 1), dtype=np.float32)
STD_U = np.ones((NU, 1), dtype=np.float32)

Cd_np = Cd.numpy()    # (3, 6)
Dd_np = Dd.numpy()    # (3, 3)

torch.manual_seed(0)
np.random.seed(0)


# ══════════════════════════════════════════════════════════════════════════════
#  Builder helpers
# ══════════════════════════════════════════════════════════════════════════════

def build_augmented_interconnect(random_ann: bool = False):
    """
    Build the 8-state dynamic parallel interconnect.

    Wiring (adapted from Jan's ECC-2025 msd_ndof_interconnect_dynamic.py):
      ANN  : x(8) + u(3) → xp(8)  [additive, zero-init]
      Gantry: x[0:6](sel) + u → xp[0:6](exp, additive)
      Output: x[0:6](sel) + u → y
    """
    gantry_block = Gantry_State_Block(Y_op=Y_OP, std_x=STD_X, std_u=STD_U)
    output_block = Linear_Output_Block(C=Cd_np, D=Dd_np)
    ann_block    = Static_ANN_Block(
        nz=NX_TOTAL + NU,   # 11
        nw=NX_TOTAL,        # 8
        n_nodes_per_layer=8,
        n_hidden_layers=2,
        net=zero_init_feed_forward_nn,
        activation=torch.nn.Tanh,
    )

    if random_ann:
        # Override zero-init to test state dynamics (Tests 14-15).
        # Restore is caller's responsibility if needed.
        for p in ann_block.parameters():
            torch.nn.init.normal_(p, std=0.01)

    ic = Interconnect(nx=NX_TOTAL, nu=NU, ny=NY)
    ic.add_block(gantry_block)
    ic.add_block(output_block)
    ic.add_block(ann_block)

    # ANN: full x and u → full xp (additive by default for "xp")
    ic.connect_block_signals(ann_block, ["x", "u"], ["xp"])

    # Gantry: x[0:6] (via selection) and u → xp[0:6] (via expansion, additive)
    ic.connect_signals("x",          gantry_block, "concat",  selection_matrix(PHY_IX, NX_TOTAL))
    ic.connect_block_signals(gantry_block, ["u"], [])
    ic.connect_signals(gantry_block, "xp",         "additive", expansion_matrix(PHY_IX, NX_TOTAL))

    # Output: x[0:6] (via selection) and u → y
    ic.connect_signals("x",          output_block, "concat",  selection_matrix(PHY_IX, NX_TOTAL))
    ic.connect_block_signals(output_block, ["u"], ["y"])

    return ic, gantry_block, output_block, ann_block


def build_phase1_interconnect():
    """Build the 6-state Phase 1 interconnect (no ANN) — identical to gantry_subnet_verification.py."""
    gantry_block = Gantry_State_Block(Y_op=Y_OP, std_x=STD_X, std_u=STD_U)
    output_block = Linear_Output_Block(C=Cd_np, D=Dd_np)

    ic = Interconnect(nx=NX_PHYS, nu=NU, ny=NY)
    ic.add_block(gantry_block)
    ic.add_block(output_block)

    ic.connect_signals("x",  gantry_block)
    ic.connect_block_signals(gantry_block, ["u"], [])
    ic.connect_signals(gantry_block, "xp")

    ic.connect_signals("x",  output_block)
    ic.connect_block_signals(output_block, ["u"], ["y"])

    return ic, gantry_block, output_block


# ══════════════════════════════════════════════════════════════════════════════
#  Test runner
# ══════════════════════════════════════════════════════════════════════════════

_results: dict = {}


def _test(name: str):
    """Decorator — catches any exception and records PASS/FAIL."""
    def decorator(fn):
        def wrapper():
            try:
                fn()
                _results[name] = "PASS"
                print(f"  PASS  {name}")
            except Exception as exc:
                _results[name] = f"FAIL: {exc}"
                print(f"  FAIL  {name}:  {exc}")
        return wrapper
    return decorator


# ══════════════════════════════════════════════════════════════════════════════
#  Group 1 — selection / expansion matrix utilities (Tests 1-3)
# ══════════════════════════════════════════════════════════════════════════════

@_test("T01: selection_matrix shape and values")
def test_01():
    sel = selection_matrix(PHY_IX, NX_TOTAL)   # (6, 8)
    assert sel.shape == (NX_PHYS, NX_TOTAL), f"shape={sel.shape}"
    for i in range(NX_PHYS):
        assert sel[i, i].item() == 1.0,   f"sel[{i},{i}] != 1"
        assert sel[i, NX_PHYS:].sum().item() == 0.0, f"sel row {i} has nonzero ANN cols"
    for j in range(NX_ANN):
        assert sel[:, NX_PHYS + j].sum().item() == 0.0, f"ANN col {j} nonzero"


@_test("T02: expansion_matrix shape and values")
def test_02():
    exp = expansion_matrix(PHY_IX, NX_TOTAL)   # (8, 6)
    assert exp.shape == (NX_TOTAL, NX_PHYS), f"shape={exp.shape}"
    for i in range(NX_PHYS):
        assert exp[i, i].item() == 1.0, f"exp[{i},{i}] != 1"
    # Rows corresponding to ANN states must be zero
    assert exp[NX_PHYS:].abs().sum().item() == 0.0, "ANN rows of expansion_matrix nonzero"


@_test("T03: expansion @ selection preserves phys states, zeros ANN states")
def test_03():
    sel = selection_matrix(PHY_IX, NX_TOTAL)   # (6, 8)
    exp = expansion_matrix(PHY_IX, NX_TOTAL)   # (8, 6)
    # Operate on column vectors — matches interconnect internal convention
    x8 = torch.randn(NX_TOTAL, 1)
    recovered = exp @ (sel @ x8)               # (8, 1)
    assert torch.allclose(recovered[:NX_PHYS], x8[:NX_PHYS]), "Physical states not preserved"
    assert torch.allclose(recovered[NX_PHYS:], torch.zeros(NX_ANN, 1)), "ANN states not zeroed"


# ══════════════════════════════════════════════════════════════════════════════
#  Group 2 — forward-pass shapes and signal routing (Tests 4-8)
# ══════════════════════════════════════════════════════════════════════════════

@_test("T04: interconnect forward pass output shapes")
def test_04():
    ic, _, _, _ = build_augmented_interconnect()
    x = torch.zeros(1, NX_TOTAL)
    u = torch.zeros(1, NU)
    with torch.no_grad():
        y, xp = ic(x, u)
    assert y.shape  == (1, NY),       f"y shape={y.shape}"
    assert xp.shape == (1, NX_TOTAL), f"xp shape={xp.shape}"


@_test("T05: gantry block receives (batch, 9, 1) input — x[0:6] selected + u")
def test_05():
    ic, gantry_block, _, _ = build_augmented_interconnect()
    captured = {}
    orig_fwd = gantry_block.forward

    def capturing_fwd(z):
        captured["shape"] = tuple(z.shape)
        return orig_fwd(z)

    gantry_block.forward = capturing_fwd
    x = torch.randn(1, NX_TOTAL)
    u = torch.randn(1, NU)
    with torch.no_grad():
        ic(x, u)
    gantry_block.forward = orig_fwd   # restore

    expected = (1, NX_PHYS + NU, 1)  # (1, 9, 1)
    assert captured.get("shape") == expected, (
        f"gantry received {captured.get('shape')}, expected {expected}"
    )


@_test("T06: ANN block receives (batch, 11, 1) input — full x(8) + u(3)")
def test_06():
    ic, _, _, ann_block = build_augmented_interconnect()
    captured = {}
    orig_fwd = ann_block.forward

    def capturing_fwd(z):
        captured["shape"] = tuple(z.shape)
        return orig_fwd(z)

    ann_block.forward = capturing_fwd
    x = torch.randn(1, NX_TOTAL)
    u = torch.randn(1, NU)
    with torch.no_grad():
        ic(x, u)
    ann_block.forward = orig_fwd

    expected = (1, NX_TOTAL + NU, 1)  # (1, 11, 1)
    assert captured.get("shape") == expected, (
        f"ANN received {captured.get('shape')}, expected {expected}"
    )


@_test("T07: xp[0:6] is physics-driven (nonzero); xp[6:8] = 0 with zero-init ANN")
def test_07():
    ic, _, _, _ = build_augmented_interconnect()
    # Use a physically meaningful state (Y≈Y_OP so block doesn't produce zeros)
    x = torch.zeros(1, NX_TOTAL)
    x[0, 2] = Y_OP   # q3 = Y ≈ operating point (in normalised units with STD_X=1)
    u = torch.ones(1, NU) * 10.0   # nonzero force
    with torch.no_grad():
        _, xp = ic(x, u)
    phys_part = xp[0, :NX_PHYS]
    ann_part  = xp[0, NX_PHYS:]
    assert phys_part.abs().max().item() > 1e-6, (
        f"xp[0:6] unexpectedly near zero (max={phys_part.abs().max().item():.2e})"
    )
    assert ann_part.abs().max().item() < 1e-7, (
        f"xp[6:8] nonzero with zero-init ANN: {ann_part.tolist()}"
    )


@_test("T08: output block ignores ANN states x[6:8] (output unchanged when perturbed)")
def test_08():
    ic, _, _, _ = build_augmented_interconnect()
    x_base = torch.randn(1, NX_TOTAL)
    u      = torch.randn(1, NU)

    x_a = x_base.clone()
    x_b = x_base.clone()
    x_b[0, NX_PHYS:] = 999.0   # large perturbation in ANN state slots

    with torch.no_grad():
        y_a, _ = ic(x_a, u)
        y_b, _ = ic(x_b, u)

    assert torch.allclose(y_a, y_b, atol=1e-5), (
        f"Output changed when x[6:8] perturbed: max_diff={( y_a - y_b).abs().max().item():.2e}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Group 3 — zero-init ANN behaviour (Tests 9-10)
# ══════════════════════════════════════════════════════════════════════════════

@_test("T09: zero-init ANN outputs all zeros for any (non-trivial) input")
def test_09():
    ann_block = Static_ANN_Block(
        nz=NX_TOTAL + NU,
        nw=NX_TOTAL,
        n_nodes_per_layer=8,
        n_hidden_layers=2,
        net=zero_init_feed_forward_nn,
        activation=torch.nn.Tanh,
    )
    z = torch.randn(4, NX_TOTAL + NU, 1) * 100.0   # large, non-trivial input
    with torch.no_grad():
        w = ann_block(z)
    assert w.abs().max().item() < 1e-8, (
        f"zero-init ANN produced nonzero output: max={w.abs().max().item():.2e}"
    )


@_test("T10: augmented (zero ANN) gives same y and xp[0:6] as Phase 1")
def test_10():
    ic_aug,   _, _, _ = build_augmented_interconnect()
    ic_phase1, _, _   = build_phase1_interconnect()

    x6  = torch.randn(1, NX_PHYS)
    x8  = torch.zeros(1, NX_TOTAL)
    x8[:, :NX_PHYS] = x6          # same physical state, ANN states = 0
    u   = torch.randn(1, NU)

    with torch.no_grad():
        y_aug,  xp_aug  = ic_aug(x8, u)
        y_p1,   xp_p1   = ic_phase1(x6, u)

    assert torch.allclose(y_aug, y_p1, atol=1e-5), (
        f"y differs: max_diff={(y_aug - y_p1).abs().max().item():.2e}"
    )
    assert torch.allclose(xp_aug[:, :NX_PHYS], xp_p1, atol=1e-5), (
        f"xp[0:6] differs: max_diff={(xp_aug[:, :NX_PHYS] - xp_p1).abs().max().item():.2e}"
    )
    assert torch.allclose(xp_aug[:, NX_PHYS:], torch.zeros(1, NX_ANN), atol=1e-7), (
        f"xp[6:8] nonzero with zero-init ANN: {xp_aug[:, NX_PHYS:].tolist()}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Group 4 — encoder dimensionality (Test 11)
# ══════════════════════════════════════════════════════════════════════════════

@_test("T11: encoder output is NX_TOTAL=8 dimensional")
def test_11():
    ic, _, _, _ = build_augmented_interconnect()
    fit_sys = SSE_Interconnect(
        na=5, nb=5,
        interconnect=ic,
        e_net_kwargs={"n_nodes_per_layer": 8},
    )
    fit_sys.init_nets(nu=NU, ny=NY)

    assert fit_sys.encoder is not None, "Encoder was not initialised"
    assert fit_sys.nx == NX_TOTAL, f"fit_sys.nx={fit_sys.nx}, expected {NX_TOTAL}"

    upast = torch.zeros(1, 5, NU)
    ypast = torch.zeros(1, 5, NY)
    with torch.no_grad():
        x0 = fit_sys.encoder(upast, ypast)   # (1, NX_TOTAL)

    assert x0.shape == (1, NX_TOTAL), f"encoder output shape={x0.shape}"


# ══════════════════════════════════════════════════════════════════════════════
#  Group 5 — gradient flow and BPTT smoke test (Tests 12-13)
# ══════════════════════════════════════════════════════════════════════════════

@_test("T12: gradients flow through ANN parameters after one forward pass")
def test_12():
    ic, _, _, ann_block = build_augmented_interconnect()
    x = torch.randn(1, NX_TOTAL)
    u = torch.randn(1, NU)

    y, xp = ic(x, u)
    loss  = y.sum() + xp.sum()
    loss.backward()

    ann_params = list(ann_block.parameters())
    assert len(ann_params) > 0, "ANN has no parameters"
    grads_ok = [p.grad is not None and p.grad.abs().max().item() > 0.0 for p in ann_params]
    # At least one ANN parameter must receive a nonzero gradient
    # Note: final layer is zero-init so its weight/bias grad may come from chain rule
    assert any(p.grad is not None for p in ann_params), "No ANN parameter has a gradient"


@_test("T13: BPTT smoke test — 1 epoch, small synthetic data, no NaN in loss")
def test_13():
    T_syn = 300
    rng   = np.random.default_rng(42)
    u_syn = rng.standard_normal((T_syn, NU)).astype(np.float32) * 50.0
    y_syn = rng.standard_normal((T_syn, NY)).astype(np.float32) * 0.05
    train_data = deepSI.System_data(u=u_syn, y=y_syn)

    ic, _, _, _ = build_augmented_interconnect()
    fit_sys = SSE_Interconnect(
        na=5, nb=5,
        interconnect=ic,
        e_net_kwargs={"n_nodes_per_layer": 8},
    )
    fit_sys.fit(
        train_sys_data=train_data,
        val_sys_data=train_data,
        epochs=1,
        batch_size=16,
        auto_fit_norm=True,
        loss_kwargs={"nf": 20},
        validation_measure="sim-RMS",
    )
    last_val = fit_sys.Loss_val[-1]
    assert not np.isnan(last_val), f"Validation loss is NaN after 1 epoch"
    assert not np.isinf(last_val), f"Validation loss is Inf after 1 epoch"


# ══════════════════════════════════════════════════════════════════════════════
#  Group 6 — ANN latent-state dynamics (Tests 14-15)
# ══════════════════════════════════════════════════════════════════════════════

@_test("T14: ANN latent states [6:8] evolve (non-zero) over 10 steps with random-init ANN")
def test_14():
    ic, _, _, _ = build_augmented_interconnect(random_ann=True)

    x = torch.zeros(1, NX_TOTAL)
    u = torch.ones(1, NU) * 10.0   # constant force excitation

    ann_states_over_time = []
    with torch.no_grad():
        for _ in range(10):
            _, xp = ic(x, u)
            x = xp   # roll forward
            ann_states_over_time.append(xp[0, NX_PHYS:].tolist())

    # After 10 steps the ANN latent states should be nonzero
    final_ann = x[0, NX_PHYS:].abs().max().item()
    assert final_ann > 1e-8, (
        f"ANN states stayed near zero over 10 steps (max |x[6:8]|={final_ann:.2e}). "
        "Check random_ann init or ANN wiring."
    )


@_test("T15: ANN state perturbation x[6:8]=0.1 propagates to output after 2 steps")
def test_15():
    ic, _, _, _ = build_augmented_interconnect(random_ann=True)

    x_base = torch.zeros(1, NX_TOTAL)
    x_base[0, 2] = Y_OP   # physical state near operating point
    u = torch.ones(1, NU) * 10.0

    # Trajectory A: x[6:8] = 0 (default)
    x_a = x_base.clone()
    # Trajectory B: x[6:8] = 0.1 (perturbed ANN states)
    x_b = x_base.clone()
    x_b[0, NX_PHYS:] = 0.1

    with torch.no_grad():
        # Step 1: produces different xp[0:6] because ANN sees different x[6:8]
        _, xp_a = ic(x_a, u)
        _, xp_b = ic(x_b, u)
        # Step 2: different x[0:6] → different output y
        y_a, _ = ic(xp_a, u)
        y_b, _ = ic(xp_b, u)

    diff = (y_a - y_b).abs().max().item()
    assert diff > 1e-9, (
        f"Output unchanged after 2 steps despite x[6:8] perturbation (max|Δy|={diff:.2e}). "
        "ANN states not feeding back into physics."
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Run all tests
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("diagnose_dynamic_parallel.py — 15-test suite")
    print("NX_TOTAL=8  (6 physical + 2 ANN latent [delta_a, vdelta_a])")
    print("=" * 65)

    print("\nGroup 1 — selection / expansion matrix utilities")
    test_01()
    test_02()
    test_03()

    print("\nGroup 2 — forward-pass shapes and signal routing")
    test_04()
    test_05()
    test_06()
    test_07()
    test_08()

    print("\nGroup 3 — zero-init ANN behaviour")
    test_09()
    test_10()

    print("\nGroup 4 — encoder dimensionality")
    test_11()

    print("\nGroup 5 — gradient flow and BPTT smoke test")
    test_12()
    test_13()

    print("\nGroup 6 — ANN latent-state dynamics")
    test_14()
    test_15()

    # ── Summary ────────────────────────────────────────────────────────────────
    n_pass = sum(1 for v in _results.values() if v == "PASS")
    n_fail = len(_results) - n_pass
    print("\n" + "=" * 65)
    print(f"Results: {n_pass}/{len(_results)} PASS   {n_fail} FAIL")
    print("=" * 65)
    if n_fail:
        print("\nFailed tests:")
        for name, status in _results.items():
            if status != "PASS":
                print(f"  {name}: {status}")
    else:
        print("All tests passed. Safe to implement gantry_interconnect_dynamic.py.")
