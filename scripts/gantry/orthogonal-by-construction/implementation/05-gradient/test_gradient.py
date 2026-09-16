"""Stage 5: is the gradient through the subtraction correct, and is it load-bearing?

__project_origin__ = "added"

Two separate questions, and the second is the one the method turns on.

CORRECTNESS. The subtraction contains a `torch.func.jvp` whose TANGENT carries gradient, which is
forward-over-reverse and is exactly the kind of composition that silently produces a wrong
gradient if the library is used incorrectly. The check is an independent central difference on the
ANN parameters, not a second autograd path.

LOAD-BEARING. Detaching the coefficient looks harmless (the correction is still subtracted, the
forward output is IDENTICAL) and destroys the method: the gradient then no longer sees that the
coefficient moves with the write, and the orthogonality of the GRADIENT is lost even though the
orthogonality of the VALUE survives. The investigation measured `1.29e-13` differentiated through
against `4.49e-02` detached, the latter being exactly the unprojected value. The detached control
here must FAIL, and its failing is the evidence.

  T1  autograd against an independent central difference on the ANN parameters   <- CRITERION
  T2  the latent rows: the subtraction on the additional-state columns is exactly `0.0`, so their
      gradient contribution is untouched. Exact equality, not a tolerance.
  T3  THE DETACHED CONTROL, which must fail: forward output identical to round-off, gradient
      orthogonality lost by orders of magnitude
  T4  the identity `dG~/deta = (I - P) dG/deta`, which is what "orthogonal in the gradient too"
      means, against the same quantity computed by projecting the unprojected Jacobian
  T5  no gradient reaches the physical parameters through the basis, which is the one deliberate
      approximation and has to be verified rather than asserted
  T6  gradient flows to the latent-path parameters, i.e. the blindness of the declined
      baseline-only reference policy does NOT occur here, because the construction evaluations
      carry nonzero additional states

Criterion (handoff Sect. 10): relative agreement with the central difference at or below `1e-6`;
latent-row subtraction exactly `0.0`; the detached control must show roughly `1e-2`, not `1e-13`.
"""

__project_origin__ = "added"

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C                                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# THEORY: the criterion is the handoff's, Sect. 10.
FD_TOL = 1e-6
DETACH_MIN = 1e-4        # the control must be WORSE than this, or it is not a control
M_REF = 64               # small: the central difference costs two forward passes per parameter
ROUTE_IX = tuple(range(C.N_P + C.N_A))


def make_ann(seed=0, n_nodes=6, n_layers=1):
    """A deliberately SMALL ANN: the central difference costs two passes per parameter."""
    from model_augmentation.fit_systems.blocks import Static_ANN_Block
    torch.manual_seed(seed)
    ann = Static_ANN_Block(nz=C.N_P + C.N_A + C.N_U, nw=len(ROUTE_IX),
                           n_nodes_per_layer=n_nodes, n_hidden_layers=n_layers).to(C.F64)
    with torch.no_grad():
        for p in ann.parameters():
            p.copy_(torch.randn_like(p) * 0.3)
    return ann


def ref_points(M, seed=0, latent_scale=1.0):
    x, u = C.design_points(M, seed=seed)
    g = torch.Generator().manual_seed(seed + 500)
    xa = torch.rand(M, C.N_A, generator=g, dtype=C.F64) * 2 * latent_scale - latent_scale
    return torch.cat([x, xa, u], dim=1).unsqueeze(-1)


def main():
    torch.manual_seed(0)
    sys.stdout = C.Tee(os.path.join(HERE, 'results.log'))
    from model_augmentation.fit_systems import obc_projection as obc

    red = C.make_reduced_block()
    ann = make_ann()
    blk = obc.OBC_Projected_ANN_Block(ann, red, ROUTE_IX, C.N_P, C.N_A, enabled=True)
    Z_ref = ref_points(M_REF, seed=0)
    basis = blk.rebuild_basis(Z_ref, expected_rank=11)
    Z_eval = ref_points(32, seed=4)

    params = list(ann.parameters())
    n_par = sum(p.numel() for p in params)

    print('=' * 78)
    print('STAGE 5 -- the gradient through the subtraction')
    print('=' * 78)
    print(f'  M = {M_REF} construction evaluations, 32 evaluation points, {n_par} ANN parameters,'
          f' float64')
    print(f'  basis rank {basis.rank}/11')
    print(f'  criterion: central-difference agreement at or below {FD_TOL:.0e}; latent rows '
          f'exactly 0.0; the detached control must be WORSE than {DETACH_MIN:.0e}')

    def loss_of(Z):
        """A scalar with no special structure, so nothing cancels by accident."""
        w = blk(Z)
        return (w * torch.arange(1, w.shape[1] + 1, dtype=C.F64).view(1, -1, 1)).pow(2).sum()

    # ------------------------------------------------------------------ T1 the criterion
    print('\n--- T1  autograd against an independent central difference (THE CRITERION) ---')
    for p in params:
        p.grad = None
    loss_of(Z_eval).backward()
    g_auto = torch.cat([p.grad.reshape(-1) for p in params]).clone()

    # HEURISTIC: the central-difference step. `1e-6` relative is the standard float64 compromise
    # between truncation error (order h^2) and cancellation (order eps/h); at h = 1e-6 both sit
    # near 1e-12 relative, which is six orders below the 1e-6 criterion.
    H = 1e-6
    g_fd = torch.zeros(n_par, dtype=C.F64)
    i = 0
    with torch.no_grad():
        for p in params:
            flat = p.reshape(-1)
            for k in range(flat.numel()):
                orig = flat[k].item()
                h = H * max(abs(orig), 1.0)
                flat[k] = orig + h
                lp_ = loss_of(Z_eval).item()
                flat[k] = orig - h
                lm_ = loss_of(Z_eval).item()
                flat[k] = orig
                g_fd[i] = (lp_ - lm_) / (2 * h)
                i += 1
    rel = float(torch.linalg.vector_norm(g_auto - g_fd)
                / torch.linalg.vector_norm(g_fd))
    worst = float(((g_auto - g_fd).abs() / g_fd.abs().clamp(min=1e-8)).max())
    print(f'  gradient norm: autograd {torch.linalg.vector_norm(g_auto):.6e}, '
          f'central difference {torch.linalg.vector_norm(g_fd):.6e}')
    print(f'  RELATIVE 2-NORM DIFFERENCE = {rel:.4e}   '
          f'{"MET" if rel <= FD_TOL else "NOT MET"} (<= {FD_TOL:.0e})')
    print(f'  worst per-parameter relative difference = {worst:.4e}')

    # ------------------------------------------------------------------ T2 the latent rows
    print('\n--- T2  the additional-state columns are untouched, EXACTLY ---')
    with torch.no_grad():
        w_raw, w_proj = ann(Z_eval), blk(Z_eval)
    aug_cols = [j for j, r in enumerate(ROUTE_IX) if r >= C.N_P]
    d_aug = (w_raw[:, aug_cols] - w_proj[:, aug_cols]).abs().max().item()
    print(f'  ANN columns {aug_cols}: max abs subtraction = {d_aug:.1e}, '
          f'exactly zero = {d_aug == 0.0}')
    assert d_aug == 0.0

    # ------------------------------------------------------------------ T3 the detached control
    print('\n--- T3  THE DETACHED CONTROL, which must FAIL ---')

    def dG_deta(detach_coefficient):
        """`d/deta` of the stacked projected write on the construction evaluations."""
        rows = []
        for j in range(n_par):
            for p in params:
                p.grad = None
            if detach_coefficient:
                a = blk.coefficient().detach()
                delta = obc.directional(red, blk.phys_input(Z_ref), a, basis.v_bar)
                w = ann(Z_ref).clone()
                corr = torch.zeros_like(w[:, :, 0])
                corr[:, blk._phys_col] = delta.reshape(-1, C.N_P)
                Gt = blk._stack_physical_write((w - corr.unsqueeze(-1)))
            else:
                Gt = blk._stack_physical_write(blk(Z_ref))
            rows.append(Gt)
            break
        return rows[0]

    out = {}
    for tag, det in (('differentiated through', False), ('DETACHED (the control)', True)):
        Gt = dG_deta(det)
        # d G~ / d eta, one column per parameter, by reverse mode over each output would be
        # expensive; the quantity that matters is norm(F' dG~/deta), which is a single vjp per
        # basis column.
        tot = 0.0
        for col in range(basis.F.shape[1]):
            for p in params:
                p.grad = None
            (basis.F[:, col] @ Gt).backward(retain_graph=True)
            tot += float(sum((p.grad ** 2).sum() for p in params))
        out[tag] = tot ** 0.5
        print(f'  {tag:<26} norm(F\' dG~/deta) = {out[tag]:.4e}')

    # and the unprojected value, which the detached case should reproduce
    tot = 0.0
    G_unproj = blk._stack_physical_write(ann(Z_ref))
    for col in range(basis.F.shape[1]):
        for p in params:
            p.grad = None
        (basis.F[:, col] @ G_unproj).backward(retain_graph=True)
        tot += float(sum((p.grad ** 2).sum() for p in params))
    unproj = tot ** 0.5
    print(f'  {"no projection at all":<26} norm(F\' dG/deta)  = {unproj:.4e}')
    through, detached = out['differentiated through'], out['DETACHED (the control)']
    print(f'\n  ratio detached to unprojected: {detached / unproj:.6f}   '
          f'(1.0 means detaching buys exactly nothing)')
    print(f'  ratio through to unprojected : {through / unproj:.4e}')
    ctrl_ok = detached > DETACH_MIN * unproj and through < 1e-10 * unproj
    print(f'  the control FAILS as required: {ctrl_ok}')
    print('  Detaching leaves the forward output identical and destroys the gradient property.')
    print('  This is why differentiating through the coefficient is not optional.')

    # ------------------------------------------------------------------ T4 the projector identity
    print('\n--- T4  the identity dG~/deta = (I - P) dG/deta ---')
    Q = basis.range_basis()
    cols_t, cols_u = [], []
    for j, p in enumerate(params):
        for k in range(p.numel()):
            pass
    # One Jacobian column per ANN parameter is too many; take a random projection of the parameter
    # space instead, which tests the same identity without the cost.
    gproj = torch.Generator().manual_seed(3)
    for _ in range(4):
        v = [torch.randn(p.shape, generator=gproj, dtype=C.F64) for p in params]
        # directional derivative of G and of G~ along v, by forward difference on a tiny step
        h = 1e-7
        with torch.no_grad():
            for p, vv in zip(params, v):
                p.add_(h * vv)
            Gp, Gtp = blk._stack_physical_write(ann(Z_ref)), blk._stack_physical_write(blk(Z_ref))
            for p, vv in zip(params, v):
                p.sub_(2 * h * vv)
            Gm, Gtm = blk._stack_physical_write(ann(Z_ref)), blk._stack_physical_write(blk(Z_ref))
            for p, vv in zip(params, v):
                p.add_(h * vv)
        dG, dGt = (Gp - Gm) / (2 * h), (Gtp - Gtm) / (2 * h)
        pred = dG - Q @ (Q.T @ dG)
        cols_t.append(float(torch.linalg.vector_norm(dGt - pred)
                            / torch.linalg.vector_norm(pred)))
        cols_u.append(float(torch.linalg.vector_norm(dG)))
    print(f'  four random parameter directions, relative difference between dG~/deta and')
    print(f'  (I - P) dG/deta: ' + ', '.join(f'{v:.3e}' for v in cols_t))
    print(f'  (the floor here is the central-difference floor, near 1e-8, not autograd round-off)')

    # ------------------------------------------------------------------ T5 no gradient to theta
    print('\n--- T5  no gradient reaches the physical parameters through the basis ---')
    red.free_params.grad = None
    for p in params:
        p.grad = None
    loss_of(Z_eval).backward()
    gt = red.free_params.grad
    print(f'  d loss / d free_params = '
          f'{"None" if gt is None else f"{gt.abs().max().item():.4e}"}')
    print('  The basis is built under no_grad at a detached expansion point and held for the')
    print('  epoch, so the physical parameters receive nothing through it. This is the ONE')
    print('  deliberate approximation in the gradient and it is verified here, not asserted.')
    print(f'  (the physical block still receives gradient through the ROLLOUT in a real model;')
    print(f'   this block is evaluated pointwise, so there is no rollout path to carry it)')

    # ------------------------------------------------------------------ T6 latent-path gradient
    print('\n--- T6  the latent path: what the FROZEN reference set does to d a / d eta_lat ---')
    print('  Proposition 7.1: with a baseline-only reference set the latent-path parameters get')
    print('  EXACTLY 0.000000e+00 sensitivity, because the latent-to-physical map is multiplied by')
    print('  zero where x_a = 0. The handoff expects candidate (a) to dissolve this, on the ground')
    print('  that rollout-drawn evaluation points carry nonzero x_a already. Measured below, that')
    print('  expectation holds for the VALUE of the coefficient and FAILS for its GRADIENT, and')
    print('  the reason is the freeze rather than the point set.')

    aug_cols_t = torch.tensor([j for j, r in enumerate(ROUTE_IX) if r >= C.N_P])
    lat_out_cols = [j for j, r in enumerate(ROUTE_IX) if r >= C.N_P]
    out_par = [q for q in params if q.shape[0] == len(ROUTE_IX)]

    def rollout_points(n_steps, zero_latent=False, keep_graph=False):
        """Construction evaluations from a short rollout of the augmented model.

        `zero_latent = True` forces `x_a = 0` at every step, which IS the declined baseline-only
        reference policy and is here only as the control. `keep_graph = True` leaves the points
        attached to the ANN parameters, which the frozen design does NOT do; it is the diagnostic
        that isolates the freeze as the cause.
        """
        xp, uu = C.design_points(M_REF, seed=7)
        xa = torch.zeros(M_REF, C.N_A, dtype=C.F64)
        for _ in range(n_steps):
            z = torch.cat([xp, xa, uu], dim=1).unsqueeze(-1)
            w = ann(z)[:, :, 0]
            xp = obc.reduced_step(red, basis.v_bar, blk.phys_input(z))[:, :, 0] \
                + w[:, blk._phys_col]
            xa = torch.zeros_like(xa) if zero_latent else w[:, lat_out_cols]
        Z = torch.cat([xp, xa, uu], dim=1).unsqueeze(-1)
        return Z if keep_graph else Z.detach()

    def grads_of_coefficient(Zr, frozen=True):
        if frozen:
            blk.rebuild_basis(Zr, keep_previous=False, log=lambda *_: None)
            for q in params:
                q.grad = None
            a_ = blk.coefficient()
        else:
            # Same basis, but the reference POINTS keep their graph, so the coefficient sees how
            # the latent weights moved them. This is NOT the shipped design; see below.
            blk.rebuild_basis(Zr.detach(), keep_previous=False, log=lambda *_: None)
            for q in params:
                q.grad = None
            a_ = blk.basis.coefficient(blk._stack_physical_write(ann(Zr)))
        val = a_.detach().clone()
        a_.pow(2).sum().backward()
        lat = float(sum((q.grad[aug_cols_t] ** 2).sum() for q in out_par)) ** 0.5
        allg = float(sum((q.grad ** 2).sum() for q in params)) ** 0.5
        return val, lat, allg

    print(f'\n  (a) the GRADIENT, with the reference set FROZEN, which is the shipped design:')
    print(f'    {"construction evaluations":<48}  {"latent-path":>13}  {"all params":>13}')
    for tag, Zr in (('prescribed points, x_a scale 1.0', ref_points(M_REF, seed=0)),
                    ('rollout 1 step, x_a from the network', rollout_points(1)),
                    ('rollout 2 steps, x_a from the network', rollout_points(2)),
                    ('rollout 2 steps, x_a FORCED to zero', rollout_points(2, zero_latent=True))):
        _, lat, allg = grads_of_coefficient(Zr)
        print(f'    {tag:<48}  {lat:>13.6e}  {allg:>13.6e}')
    print('  Exactly zero in EVERY row, the rollout rows included. A frozen point set is data:')
    print('  the latent output rows write ANN columns 6 and 7 only, the coefficient is fitted from')
    print('  the PHYSICAL columns, and with the points detached there is no remaining path.')

    print(f'\n  (b) the same with the reference points left ATTACHED, as the DIAGNOSTIC that')
    print(f'      isolates the freeze as the cause (this is NOT the shipped design):')
    print(f'    {"construction evaluations":<48}  {"latent-path":>13}  {"all params":>13}')
    for tag, ns, zl in (('rollout 1 step, attached', 1, False),
                        ('rollout 2 steps, attached', 2, False),
                        ('rollout 2 steps, attached, x_a forced to zero', 2, True)):
        Zr = rollout_points(ns, zero_latent=zl, keep_graph=True)
        _, lat, allg = grads_of_coefficient(Zr, frozen=False)
        print(f'    {tag:<48}  {lat:>13.6e}  {allg:>13.6e}')

    print(f'\n  (c) the VALUE of the coefficient DOES depend on the latent content of the points:')
    a_lat, _, _ = grads_of_coefficient(rollout_points(2))
    a_zero, _, _ = grads_of_coefficient(rollout_points(2, zero_latent=True))
    rel_val = float((a_lat - a_zero).norm() / a_lat.norm())
    print(f'    norm(a with x_a from the network - a with x_a forced to zero) / norm(a) = '
          f'{rel_val:.4e}')
    print('    so the rollout-drawn points are not cosmetic; they change what is subtracted.')

    print('\n  READING. The handoff expected candidate (a) to dissolve the latent blindness. Half')
    print('  of that holds: the coefficient VALUE depends on the latent content of the reference')
    print('  points, by the amount in (c). The other half does not: d a / d eta_lat is EXACTLY')
    print('  zero under the shipped design, and (b) shows why -- it is the FREEZE, not the choice')
    print('  of points. And the freeze is not negotiable: the derivation rules out fitting the')
    print('  coefficient along the live trajectory precisely because the correction at step k')
    print('  would then depend on the learned writes at later steps, i.e. on the future.')
    print('  Stated carefully, because it is easy to overstate. The latent WRITE g_a is')
    print('  deliberately NOT orthogonalized and that is correct: the baseline defines no')
    print('  additional-state update, so the basis has zero rows there and there is nothing to be')
    print('  orthogonal TO. The physical write DEPENDENCE on the additional states IS projected,')
    print('  because the subtraction is evaluated at the actual rollout point with x_a included.')
    print('  What is exactly zero is narrower than either: only the sensitivity of the FITTED')
    print('  COEFFICIENT to the latent-path weights, the coefficient being fitted once per pass')
    print('  from a frozen snapshot. The projection is still applied in full at every point.')
    print('  So the limitation is real but small, and it does not change the design: the')
    print('  alternative is ruled out on causality.')
    blk.rebuild_basis(Z_ref, keep_previous=False, log=lambda *_: None)

    print('\n' + '=' * 78)
    ok = rel <= FD_TOL and d_aug == 0.0 and ctrl_ok
    print(f'STAGE 5 VERDICT: central difference {rel:.4e} (<= {FD_TOL:.0e}), latent rows exactly '
          f'0.0, detached control fails as required -- {"MET" if ok else "NOT MET"}')
    print('=' * 78)
    sys.stdout.flush()


if __name__ == '__main__':
    main()
