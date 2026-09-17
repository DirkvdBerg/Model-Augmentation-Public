"""Focused qualification of the eleven-column affine OBC arm (D-200).

Uses the production gantry model with a decimated reference set.  The full-data qualification
remains a separate expensive run; this test targets algebra, gradients, and rollout wiring.
"""
__project_origin__ = "added"

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rig import obc_config, build_system, perturb_output_layer, make_batch  # noqa: E402
from model_augmentation.fit_systems.obc import OBCBasis  # noqa: E402


def main():
    # Scaling-aware r_J must remain accurate for almost-equal matrices; the trace identity loses
    # digits by cancellation in precisely this regime.
    torch.manual_seed(4)
    A = torch.randn(73, 11, dtype=torch.float64); A[:, -1] *= 1e5
    B_small = A + 1e-3 * torch.randn_like(A)
    v_small = torch.zeros(10, dtype=torch.float64)
    ba = OBCBasis.from_matrix(A, v_small, scale_columns=True)
    bb = OBCBasis.from_matrix(B_small, v_small, scale_columns=True)
    rj = ba.relative_change(bb)
    rj_explicit = float((A - B_small).norm() / B_small.norm())
    assert abs(rj - rj_explicit) < 1e-15

    cfg = obc_config(device='cpu', compile_mode=None, nf_override=5, stride=3000,
                     obc_space='affine',
                     static_pass_buffers=bool(os.environ.get('STATIC_PASS')))
    rig = build_system(cfg, ref_stride=500)
    life, corr, ann = rig.life, rig.corr, rig.ann
    perturb_output_layer(ann)
    life.refresh(corr.expansion_point(), epoch=0)

    assert life.basis.n_cols == 11 and life.basis.rank == 11
    assert life.basis.report.extra['column_scaled'] is True
    theta = life.coefficient(ann)
    assert theta.shape == (11,)

    # The scaled SVD must still expose products in the original [J | fbar] coordinates.
    B, _ = life.basis_builder(corr.vbar.to(torch.float64))
    rel_apply = ((life.basis.apply(theta) - B @ theta.to(torch.float64)).norm()
                 / ((B @ theta.to(torch.float64)).norm() + 1e-300))
    F = life.reference_field(ann)
    r_perp = life.basis.r_perp(life.basis.project_out(F))
    assert rel_apply < 1e-12 and r_perp < 1e-12

    # Every coefficient, including alpha, must remain differentiable in the ANN field.
    grad_norms = []
    params = tuple(ann.parameters())
    for j in range(11):
        grads = torch.autograd.grad(theta[j], params, retain_graph=True, allow_unused=True)
        grad_norms.append(float(sum((g * g).sum() for g in grads if g is not None).sqrt()))
    assert all(g > 0 and torch.isfinite(torch.tensor(g)) for g in grad_norms)

    # Optimized hand sensitivity and transformed JVP reference implement the same correction.
    x = rig.ref.Z_field[:7, :rig.fs.hfn.nx]
    u = rig.ref.Z_field[:7, rig.fs.hfn.nx:]
    optimized = corr.correction_from_theta(x, u, theta, use_jvp=False)
    reference = corr.correction_from_theta(x, u, theta, use_jvp=True)
    rel_runtime = float((optimized - reference).norm() / (reference.norm() + 1e-30))
    assert rel_runtime < 5e-6

    # The affine offset reuses the primal returned beside the tangent: one RK4 call, not two.
    obc_pass = corr.tangent_pair(theta)
    calls = {'n': 0}
    original = rig.phy._rk4_with_tangent
    def counted(*args, **kwargs):
        calls['n'] += 1
        return original(*args, **kwargs)
    rig.phy._rk4_with_tangent = counted
    try:
        corr(x, u, obc_pass)
    finally:
        rig.phy._rk4_with_tangent = original
    assert calls['n'] == 1

    # Exercise the production objective/backward and prediction buffer shape.
    batch, kw = make_batch(rig, B=2)
    loss = rig.fs.loss(*batch[:4], **kw)
    loss.backward()
    fresh = life.prediction_coefficient(ann)
    corr.set_frozen(fresh, life.basis.vbar)
    assert corr.theta_frozen.shape == (11,) and torch.equal(corr.theta_frozen, fresh)
    prediction_optimized = corr(x, u)
    corr.use_jvp = True
    try:
        prediction_reference = corr(x, u)
    finally:
        corr.use_jvp = False
    assert torch.allclose(prediction_optimized, prediction_reference, rtol=5e-6, atol=1e-9)

    print('AFFINE OBC PASSED: rank=11, r_perp=%.3e, apply_rel=%.3e, '
          'runtime_rel=%.3e, min_coeff_grad=%.3e, rk4_calls=%d, loss=%.3e'
          % (r_perp, rel_apply, rel_runtime, min(grad_norms), calls['n'], float(loss)))


if __name__ == '__main__':
    main()
