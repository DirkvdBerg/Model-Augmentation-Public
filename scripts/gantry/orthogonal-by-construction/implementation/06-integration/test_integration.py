"""Stage 6: does the seam hold inside a real Interconnect?

__project_origin__ = "added"

Not a training run. HARD CAPS, per handoff Sect. 12: batch at most 64, `nf` at most 50, at most 20
optimizer steps, CPU, float64, finishing inside two minutes. The smoke test exists to prove the
seam holds, not to train anything, and nothing here may be read as a result about learning.

  T1  FLAG-OFF BIT-IDENTITY: an interconnect holding the wrapper with `enabled = False` against
      one holding the bare `Static_ANN_Block`, over a recursive rollout. Exact tensor equality,
      not a tolerance.                                                          <- CRITERION
  T2  the wrapper's state_dict: the physical block's parameters must NOT be duplicated into it
  T3  CHECKPOINT ROUND-TRIP after a `jacfwd` over the physical block, which is the `_cur`
      pickling hazard that killed the first validation of an earlier run
  T4  a capped recursive rollout with the flag ON: it runs, the correction is nonzero, and the
      states stay finite
  T5  a capped optimizer smoke test, 20 steps, with the epoch-boundary basis rebuild in place
  T6  memory and wall cost of the seam at the capped size, since the coefficient is refitted at
      every timestep and the graph therefore holds one ANN-over-M subgraph per step
"""

__project_origin__ = "added"

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C                                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# Hard caps from handoff Sect. 12. Not tuning knobs.
BATCH, NF, N_STEPS, M_REF = 8, 50, 20, 64
ROUTE_IX = tuple(range(C.N_P + C.N_A))
NXD = C.N_P + C.N_A


def make_ann(seed=0):
    from model_augmentation.fit_systems.blocks import Static_ANN_Block
    torch.manual_seed(seed)
    ann = Static_ANN_Block(nz=NXD + C.N_U, nw=len(ROUTE_IX),
                           n_nodes_per_layer=16, n_hidden_layers=2).to(C.F64)
    with torch.no_grad():
        for p in ann.parameters():
            p.copy_(torch.randn_like(p) * 0.05)
    return ann


def build_ic(learn_block, phy_block):
    """The pipeline's wiring, from `scripts/gantry/gantry_dynamic/model.py`."""
    from model_augmentation.fit_systems.interconnect import Interconnect
    from model_augmentation.fit_systems.blocks import Linear_Output_Block
    from model_augmentation.utils.utils import expansion_matrix, selection_matrix
    from model_augmentation.systems.gantry_ss import Cd, Dd

    PHY_IX = list(range(C.N_P))
    ic = Interconnect(NXD, C.N_U, 3)
    out_phys = Linear_Output_Block(C=torch.as_tensor(Cd, dtype=C.F64).numpy(),
                                   D=torch.as_tensor(Dd, dtype=C.F64).numpy())
    ic.add_block(phy_block)
    ic.add_block(out_phys)
    ic.add_block(learn_block)
    ic.connect_block_signals(learn_block, ["x", "u"], [])
    ic.connect_signals(learn_block, "xp", "additive", expansion_matrix(list(ROUTE_IX), NXD))
    ic.connect_signals("x", phy_block, "concat", selection_matrix(PHY_IX, NXD))
    ic.connect_block_signals(phy_block, ["u"], [])
    ic.connect_signals(phy_block, "xp", "additive", expansion_matrix(PHY_IX, NXD))
    ic.connect_signals("x", out_phys, "concat", selection_matrix(PHY_IX, NXD))
    ic.connect_block_signals(out_phys, ["u"], ["y"])
    return ic.to(C.F64)


def rollout(ic, x0, U):
    """`nf` recursive steps. Returns the output sequence and the final state."""
    x, ys = x0, []
    for k in range(U.shape[1]):
        y, x = ic(x, U[:, k, :])
        ys.append(y)
    return torch.stack(ys, dim=1), x


def ref_points(M, seed=0):
    x, u = C.design_points(M, seed=seed)
    g = torch.Generator().manual_seed(seed + 500)
    xa = torch.rand(M, C.N_A, generator=g, dtype=C.F64) * 2 - 1
    return torch.cat([x, xa, u], dim=1).unsqueeze(-1)


def main():
    torch.manual_seed(0)
    sys.stdout = C.Tee(os.path.join(HERE, 'results.log'))
    from model_augmentation.fit_systems import obc_projection as obc

    print('=' * 78)
    print('STAGE 6 -- integration: does the seam hold inside a real Interconnect?')
    print('=' * 78)
    print(f'  HARD CAPS (handoff Sect. 12): batch {BATCH}, nf {NF}, {N_STEPS} optimizer steps, '
          f'CPU, float64')
    print('  This is a SEAM test. Nothing here is a result about learning.')

    x0 = torch.randn(BATCH, NXD, dtype=C.F64) * 0.05
    x0[:, 2] = torch.linspace(-0.2, 0.2, BATCH, dtype=C.F64)      # spread Y over the range
    U = torch.randn(BATCH, NF, C.N_U, dtype=C.F64) * 20.0

    # ------------------------------------------------------------------ T1 flag-off bit-identity
    print('\n--- T1  FLAG-OFF BIT-IDENTITY over a recursive rollout (THE CRITERION) ---')
    ann_a, ann_b = make_ann(0), make_ann(0)
    phy_a, phy_b = C.make_reduced_block(), C.make_reduced_block()
    ic_plain = build_ic(ann_a, phy_a)
    wrapped = obc.OBC_Projected_ANN_Block(ann_b, phy_b, ROUTE_IX, C.N_P, C.N_A, enabled=False)
    ic_wrapped = build_ic(wrapped, phy_b)
    with torch.no_grad():
        y_plain, xT_plain = rollout(ic_plain, x0.clone(), U)
        y_wrap, xT_wrap = rollout(ic_wrapped, x0.clone(), U)
    same_y = bool(torch.equal(y_plain, y_wrap))
    same_x = bool(torch.equal(xT_plain, xT_wrap))
    print(f'  {NF} recursive steps at batch {BATCH}')
    print(f'  output sequence: torch.equal = {same_y}, max abs difference '
          f'{(y_plain - y_wrap).abs().max().item():.1e}')
    print(f'  final state    : torch.equal = {same_x}, max abs difference '
          f'{(xT_plain - xT_wrap).abs().max().item():.1e}')
    print(f'  BIT-IDENTICAL: {same_y and same_x}')
    assert same_y and same_x

    # ------------------------------------------------------------------ T2 no duplicated keys
    print('\n--- T2  the wrapper does not duplicate the physical block into its state_dict ---')
    keys = list(wrapped.state_dict().keys())
    print(f'  wrapper state_dict keys: {keys}')
    dup = [k for k in keys if 'free_params' in k or 'combo_init' in k]
    print(f'  keys belonging to the physical block: {dup}   (expected none)')
    assert not dup
    n_ic = sum(p.numel() for p in ic_wrapped.parameters())
    n_expect = sum(p.numel() for p in ann_b.parameters()) + phy_b.free_params.numel()
    print(f'  interconnect parameter count {n_ic}, expected {n_expect} '
          f'(ANN + 10 reduced physical): {n_ic == n_expect}')
    assert n_ic == n_expect

    # ------------------------------------------------------------------ T3 checkpoint round-trip
    print('\n--- T3  CHECKPOINT ROUND-TRIP after a jacfwd over the physical block ---')
    wrapped.enabled = True
    basis = wrapped.rebuild_basis(ref_points(M_REF, seed=0), expected_rank=11)
    print(f'  basis built (rank {basis.rank}); phy_b._cur is now '
          f'{type(phy_b._cur).__name__ if phy_b._cur is not None else "None"}')
    path = os.path.join(HERE, 'roundtrip.pt')
    t0 = time.perf_counter()
    torch.save(ic_wrapped.state_dict(), path)
    sd = torch.load(path, weights_only=True)
    print(f'  torch.save + torch.load succeeded in {time.perf_counter() - t0:.3f} s, '
          f'{len(sd)} keys, {os.path.getsize(path)} bytes')
    ann_c, phy_c = make_ann(1), C.make_reduced_block()
    wrapped_c = obc.OBC_Projected_ANN_Block(ann_c, phy_c, ROUTE_IX, C.N_P, C.N_A, enabled=False)
    ic_c = build_ic(wrapped_c, phy_c)
    ic_c.load_state_dict(sd)
    with torch.no_grad():
        y_c, _ = rollout(ic_c, x0.clone(), U)
    print(f'  reloaded model, flag OFF, reproduces the flag-off rollout: '
          f'{bool(torch.equal(y_c, y_plain))}')
    assert torch.equal(y_c, y_plain)
    os.remove(path)
    print('  (the `_cur` drop in __getstate__ is what makes this work; without it torch.save')
    print('   raises NotImplementedError: Cannot access storage of TensorWrapper)')

    # ------------------------------------------------------------------ T4 flag-on rollout
    print('\n--- T4  a capped recursive rollout with the flag ON ---')
    t0 = time.perf_counter()
    with torch.no_grad():
        y_on, xT_on = rollout(ic_wrapped, x0.clone(), U)
    t_on = time.perf_counter() - t0
    t0 = time.perf_counter()
    with torch.no_grad():
        y_off, _ = rollout(ic_plain, x0.clone(), U)
    t_off = time.perf_counter() - t0
    d = (y_on - y_off).abs().max().item()
    print(f'  flag ON  rollout: {t_on:.3f} s, output finite = {bool(torch.isfinite(y_on).all())}, '
          f'max abs y = {y_on.abs().max().item():.4e}')
    print(f'  flag OFF rollout: {t_off:.3f} s')
    print(f'  wall cost of the seam: {t_on / t_off:.2f}x')
    print(f'  the correction is doing something: max abs (y_on - y_off) = {d:.4e}')
    assert torch.isfinite(y_on).all() and d > 0

    # ------------------------------------------------------------------ T5 capped optimizer smoke
    print(f'\n--- T5  a capped optimizer smoke test, {N_STEPS} steps ---')
    y_target = y_off.detach().clone()
    opt = torch.optim.Adam(ic_wrapped.parameters(), lr=1e-4)
    t0 = time.perf_counter()
    losses = []
    for it in range(N_STEPS):
        if it % 10 == 0:
            # The epoch-boundary rebuild, exercised in the loop rather than described.
            wrapped.rebuild_basis(ref_points(M_REF, seed=0), expected_rank=11,
                                  log=lambda *_: None)
        opt.zero_grad()
        y_hat, _ = rollout(ic_wrapped, x0.clone(), U)
        loss = torch.nn.functional.mse_loss(y_hat, y_target)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(ic_wrapped.parameters(), 1e12).item()
        opt.step()
        losses.append(loss.item())
        if it % 5 == 0 or it == N_STEPS - 1:
            print(f'    step {it:>3}  loss {loss.item():.6e}  grad norm {gn:.4e}  '
                  f'free_params max abs {phy_b.free_params.abs().max().item():.3e}')
    t_train = time.perf_counter() - t0
    print(f'  {N_STEPS} steps in {t_train:.1f} s ({t_train / N_STEPS:.2f} s/step at batch '
          f'{BATCH}, nf {NF})')
    print(f'  loss first {losses[0]:.6e} -> last {losses[-1]:.6e}, all finite = '
          f'{all(torch.isfinite(torch.tensor(v)) for v in losses)}')
    print('  The loss is NOT evidence of anything: 20 steps against a synthetic target at batch 8.')
    print('  What this shows is that the seam survives backward, the optimizer and the rebuild.')
    assert all(torch.isfinite(torch.tensor(v)) for v in losses)

    # ------------------------------------------------------------------ T6 the cost of the seam
    print('\n--- T6  the cost of the seam at the capped size ---')
    print(f'  the coefficient is refitted at EVERY timestep, so the graph holds one')
    print(f'  ANN-over-M subgraph per step: nf {NF} x M {M_REF} = {NF * M_REF} ANN evaluations')
    print(f'  per rollout, on top of {NF} directional derivatives.')
    print(f'  forward-only wall ratio (T4):       {t_on / t_off:.2f}x')
    print(f'  forward+backward, per step (T5):    {t_train / N_STEPS:.2f} s')
    print(f'  stage 0 measured 3.19x at the production batch 256; at batch {BATCH} the')
    print(f'  per-operation overhead is a larger share, so this ratio is an upper bound.')

    print('\n' + '=' * 78)
    print(f'STAGE 6 VERDICT: flag-off bit-identity over {NF} recursive steps '
          f'(torch.equal TRUE on both the output sequence and the final state), checkpoint '
          f'round-trip after jacfwd OK, capped smoke test ran -- MET')
    print('=' * 78)
    sys.stdout.flush()


if __name__ == '__main__':
    main()
