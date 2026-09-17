"""The ten checks for lightweight OBC validation monitoring (D-201).

Monitoring must describe the FRESH post-update model that validation scores, must cost no second
reference-set ANN pass, must refresh no basis, and must not print soft-penalty meters on runs
that have no soft penalty.

  V1  the diagnostics' coefficient is recomputed AFTER an optimizer update
  V2  no basis refresh happens just to produce diagnostics
  V3  validation synchronisation plus diagnostics make exactly ONE reference-field ANN pass
  V4  rho_overlap agrees with `OBCBasis.r_overlap(F)`
  V5  r_perp agrees with the explicit stored-factor construction
  V6  nothing cached carries an autograd graph (or a tensor at all)
  V7  tangent mode reports ten coefficients and no alpha
  V8  affine mode reports eleven coefficients and alpha = theta[-1]
  V9  non-OBC and soft-penalty logging stay valid, and no line claims "ANN~0" unmeasured
  V10 `Probe_combo_err` history and checkpoint selection are unchanged

CPU, production model, decimated reference set: the checks are about the monitoring seam, not
the dataset.
"""
__project_origin__ = "added"

import io
import contextlib
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rig import obc_config, build_system, perturb_output_layer, make_batch   # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', '..'))                          # scripts/gantry
from gantry_dynamic.training import _NfProbe                                # noqa: E402

RESULTS = {}


def verdict(name, ok, detail=''):
    RESULTS[name] = bool(ok)
    print('  -> %s: %s  %s' % ('MET' if ok else 'NOT MET', name, detail))
    assert ok, name


def sync_and_capture(fs):
    """Run the production validation synchronisation and return its cached diagnostics."""
    fs._sync_prediction_state_for_validation(refresh_basis=False)
    return fs.obc_validation_diagnostics


def counted_field(life):
    """Wrap `reference_field` with a call counter; returns (restore, counter dict)."""
    original = life.reference_field
    n = {'calls': 0}

    def wrapped(component):
        n['calls'] += 1
        return original(component)

    life.reference_field = wrapped
    return (lambda: setattr(life, 'reference_field', original)), n


def build_probe(rig):
    """A real `_NfProbe`, printing, attached to the rig's system."""
    return _NfProbe(rig.fs, rig.hp['nf'], rig.cfg, rig.data.val_ckpt_data, do_print=True)


def probe_text(probe):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        probe._joint_probe()
    return buf.getvalue()


class _FakePenalty:
    """The soft-penalty object's read interface, as `_joint_probe` uses it."""

    def __init__(self, ann, nx, nu, dtype):
        self.beta = 1e-3
        self.Z_pts = torch.zeros(4, nx + nu, 1, dtype=dtype).normal_()
        self.route_cols = [0, 1]
        self.Q = torch.linalg.qr(torch.randn(8, 2, dtype=dtype))[0]


def run_tangent():
    print('\n=== tangent arm ===')
    cfg = obc_config(device='cpu', compile_mode=None, nf_override=5, stride=3000)
    rig = build_system(cfg, ref_stride=500)
    fs, life, corr, ann = rig.fs, rig.life, rig.corr, rig.ann
    perturb_output_layer(ann)
    life.refresh(corr.expansion_point(), epoch=0)
    fs.epoch_counter = 0.0

    batch, kw = make_batch(rig, B=2)

    # ---- V1 fresh coefficient after an optimizer update -----------------------------
    print('\nV1 diagnostics use a coefficient recomputed after the optimizer update')
    fs.optimizer.zero_grad()
    fs.loss(*batch[:4], **kw).backward()
    theta_objective = life.last_theta.clone()              # theta_aux(eta_k), pre-step
    frozen_objective = corr.theta_frozen.detach().clone()
    fs.optimizer.step()
    fs.optimizer.zero_grad()
    d = sync_and_capture(fs)
    theta_diag = torch.tensor(d['theta'], dtype=torch.float64)
    with torch.no_grad():
        theta_truth = life.basis.coefficient(life.reference_field(ann).detach())
    moved = float((theta_diag - theta_objective).norm() / theta_objective.norm())
    agrees = float((theta_diag - theta_truth).norm())
    print('  |theta_diag - theta_objective| / |theta_objective| = %.3e' % moved)
    print('  |theta_diag - fresh solve from the updated ANN|      = %.3e' % agrees)
    verdict('V1', moved > 1e-9 and agrees == 0.0
            and not torch.equal(corr.theta_frozen, frozen_objective))

    # ---- V2 no basis refresh for diagnostics ---------------------------------------
    print('\nV2 no basis refresh happens for the diagnostics')
    n_refresh, vbar_before = life.n_refresh, life.basis.vbar.clone()
    basis_id = id(life.basis)
    sync_and_capture(fs)
    print('  n_refresh %d -> %d, basis object identical: %s, vbar unchanged: %s'
          % (n_refresh, life.n_refresh, id(life.basis) == basis_id,
             bool(torch.equal(vbar_before, life.basis.vbar))))
    verdict('V2', life.n_refresh == n_refresh and id(life.basis) == basis_id
            and torch.equal(vbar_before, life.basis.vbar))

    # ---- V3 exactly one reference-field pass ----------------------------------------
    print('\nV3 validation synchronisation plus diagnostics make one reference-field pass')
    restore, n = counted_field(life)
    try:
        sync_and_capture(fs)
    finally:
        restore()
    print('  reference_field calls per validation synchronisation = %d' % n['calls'])
    verdict('V3', n['calls'] == 1)

    # ---- V4/V5 the two residual meters against their explicit definitions ------------
    print('\nV4/V5 rho_overlap and r_perp against the explicit stored-factor constructions')
    d = sync_and_capture(fs)
    with torch.no_grad():
        F = life.reference_field(ann).detach()
        rho_ref = life.basis.r_overlap(F)
        F_tilde = F.to(torch.float64) - life.basis.apply(life.basis.coefficient(F))
        r_perp_ref = life.basis.r_perp(F_tilde)
    print('  rho_overlap %.12e vs r_overlap %.12e' % (d['rho_overlap'], rho_ref))
    print('  r_perp      %.12e vs explicit  %.12e' % (d['r_perp'], r_perp_ref))
    verdict('V4', abs(d['rho_overlap'] - rho_ref) <= 1e-12 * max(1.0, rho_ref))
    verdict('V5', abs(d['r_perp'] - r_perp_ref) <= 1e-12 * max(1.0, r_perp_ref))

    # ---- V6 nothing cached carries a graph ------------------------------------------
    print('\nV6 no cached value is a tensor, let alone one with an autograd graph')
    bad = []
    for k, v in d.items():
        vals = v if isinstance(v, list) else [v]
        for x in vals:
            if isinstance(x, torch.Tensor):
                bad.append((k, 'tensor, requires_grad=%s' % x.requires_grad))
            elif not isinstance(x, (float, int, str, bool, type(None))):
                bad.append((k, type(x).__name__))
    print('  cached keys: %s' % ', '.join(sorted(d)))
    verdict('V6', not bad, str(bad))
    same = life.last_validation_diagnostics
    verdict('V6-lifecycle', same is not None and same['r_perp'] == d['r_perp'])

    # ---- V7 tangent shape ------------------------------------------------------------
    print('\nV7 tangent mode reports ten coefficients and no alpha')
    print('  space=%s, len(theta)=%d, alpha=%r, n_cols=%d'
          % (d['space'], len(d['theta']), d['alpha'], d['n_cols']))
    verdict('V7', d['space'] == 'tangent' and len(d['theta']) == 10
            and d['alpha'] is None and d['n_cols'] == 10)

    # ---- V9 printing separation ------------------------------------------------------
    print('\nV9 printing separation: OBC labels, soft-penalty labels, and neither')
    probe = build_probe(rig)
    sync_and_capture(fs)
    obc_text = probe_text(probe)
    print(obc_text, end='')
    ok_obc = ('[obc]' in obc_text and 'orth-frac' not in obc_text
              and 'ANN~0' not in obc_text and '[combos]' in obc_text
              and obc_text.count('=') >= 10)

    saved = fs.obc_validation_diagnostics
    fs.obc_validation_diagnostics = None                     # a plain joint run, no OBC
    plain_text = probe_text(probe)
    print(plain_text, end='')
    ok_plain = ('[obc]' not in plain_text and 'orth-frac' not in plain_text
                and 'ANN~0' not in plain_text and 'combo-err' in plain_text)

    fs.orth_penalty = _FakePenalty(rig.ann, fs.hfn.nx, rig.cfg.nu, rig.cfg.dtype_pt)
    try:
        soft_text = probe_text(probe)
    finally:
        del fs.orth_penalty
    print(soft_text, end='')
    ok_soft = ('orth-frac' in soft_text and 'V_orth' in soft_text
               and '[obc]' not in soft_text)
    fs.obc_validation_diagnostics = saved
    verdict('V9', ok_obc and ok_plain and ok_soft,
            'obc=%s plain=%s soft=%s' % (ok_obc, ok_plain, ok_soft))

    # ---- V9b a state mismatch is reported, never silently mixed ----------------------
    print('\nV9b a diagnostics/validation state mismatch is reported rather than printed')
    fs.obc_validation_diagnostics = dict(saved, batch_counter=saved['batch_counter'] + 7)
    stale_text = probe_text(probe)
    print(stale_text, end='')
    fs.obc_validation_diagnostics = saved
    verdict('V9b', 'STALE' in stale_text and 'overlap rho' not in stale_text)

    # ---- V10 history and selection ---------------------------------------------------
    print('\nV10 Probe_combo_err history and checkpoint selection unchanged')
    n_hist = len(fs.Probe_combo_err)
    best_before = fs.bestfit
    probe._joint_probe()
    combos = rig.phy.identifiable_combinations()
    rel = [(combos[k] - probe._combo_nom[k]) / s for k, s in probe._combo_scale.items()]
    expect = float(torch.tensor(rel, dtype=torch.float64).pow(2).mean().sqrt())
    print('  history %d -> %d, last=%.6e, recomputed=%.6e, bestfit unchanged: %s'
          % (n_hist, len(fs.Probe_combo_err), fs.Probe_combo_err[-1], expect,
             fs.bestfit == best_before))
    verdict('V10', len(fs.Probe_combo_err) == n_hist + 1
            and abs(fs.Probe_combo_err[-1] - expect) <= 1e-12 * max(1.0, expect)
            and fs.bestfit == best_before
            and len(fs.Probe_orth_frac) == n_hist + 1
            and len(fs.Probe_param_loss) == n_hist + 1)


def run_affine():
    print('\n=== affine arm ===')
    cfg = obc_config(device='cpu', compile_mode=None, nf_override=5, stride=3000,
                     obc_space='affine')
    rig = build_system(cfg, ref_stride=500)
    fs, life, corr, ann = rig.fs, rig.life, rig.corr, rig.ann
    perturb_output_layer(ann)
    life.refresh(corr.expansion_point(), epoch=0)
    fs.epoch_counter = 0.0

    print('\nV8 affine mode reports eleven coefficients and alpha')
    d = sync_and_capture(fs)
    alpha_ok = d['alpha'] == d['theta'][-1]
    frozen_ok = abs(float(corr.theta_frozen[-1]) - d['alpha']) <= 1e-6 * max(1.0, abs(d['alpha']))
    print('  space=%s, len(theta)=%d, n_cols=%d, alpha=%+.6e (= theta[-1]: %s)'
          % (d['space'], len(d['theta']), d['n_cols'], d['alpha'], alpha_ok))
    verdict('V8', d['space'] == 'affine' and len(d['theta']) == 11 and d['n_cols'] == 11
            and alpha_ok and frozen_ok)

    # The two residual meters must also hold under column equilibration, where the stored
    # factors are of the SCALED matrix and every public product converts back.
    with torch.no_grad():
        F = life.reference_field(ann).detach()
        rho_ref = life.basis.r_overlap(F)
        F_tilde = F.to(torch.float64) - life.basis.apply(life.basis.coefficient(F))
        r_perp_ref = life.basis.r_perp(F_tilde)
    print('  scaled-basis rho_overlap %.12e vs %.12e; r_perp %.3e vs %.3e'
          % (d['rho_overlap'], rho_ref, d['r_perp'], r_perp_ref))
    verdict('V8-meters', abs(d['rho_overlap'] - rho_ref) <= 1e-12 * max(1.0, rho_ref)
            and abs(d['r_perp'] - r_perp_ref) <= 1e-12 * max(1.0, r_perp_ref))

    probe = build_probe(rig)
    text = probe_text(probe)
    print(text, end='')
    verdict('V8-print', 'alpha' in text and 'space affine' in text and 'ANN~0' not in text)


def main():
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 2))
    run_tangent()
    run_affine()
    print('\nALL VALIDATION-MONITORING CHECKS MET: %d/%d'
          % (sum(RESULTS.values()), len(RESULTS)))


if __name__ == '__main__':
    main()
