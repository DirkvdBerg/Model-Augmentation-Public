"""Dataset qualification, rho table, orthogonality floor, runtime and memory at the FULL reference
set (every valid training sample) for the one-step OBC path (D-192, handoff sects. 9.4, 9.5, 9.8).

Bounded preflight on the ACTIVE dataset only (`augmentation_ma50_b140-230_a6_z03`). Nothing is
generated and no other dataset directory is compared. If the set is rank deficient or severely ill
conditioned the script says so and names the directions and records; it does not stop the
generic implementation.

Runs the production attach (`cfg.obc=True` through `build_model`), so the reference set is the
one training would use. The float64 basis build runs on the CUDA device when there is one, the
SVDs on the CPU.
"""
__project_origin__ = "added"

import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rig import RESULTS_DIR, Tee, obc_config, perturb_output_layer, cuda_sync   # noqa: E402
from rig import load_datasets, compute_normalization, build_model, build_closed_loop  # noqa: E402
from rig import TRAIN_FILES, VAL_FILES, make_batch, Rig                        # noqa: E402
from gantry_dynamic.obc_gantry import COMBO_NAMES, make_basis_builder            # noqa: E402
from gantry_dynamic.obc_diagnostics import (qualify_basis, check_truth_parameters,   # noqa: E402
                                            discrepancy_fields, rho_table, principal_angles)
from model_augmentation.fit_systems.obc import OBCBasis, evaluate_step_rows       # noqa: E402

# D-188: the active dataset was generated at MA_FRAC=0.50, zeta_a=0.03. Confirmed against the
# recorded one-step transitions below before it is used for delta.
TRUTH_HYPOTHESES = [('D-188 ma0.50 z0.03', 0.50, 0.03), ('ma0.50 z0.05', 0.50, 0.05),
                    ('oracle.py ma0.10 z0.05', 0.10, 0.05), ('ma0.10 z0.03', 0.10, 0.03)]


def free_coordinate_for(block, target_combo):
    """The free coordinate at which `combinations_from_free` equals `target_combo`."""
    ci = block.combo_init.detach().double().cpu(); tc = target_combo.double().cpu()
    f = torch.log(tc / ci)
    f[block.M_DIFF_IX] = tc[block.M_DIFF_IX] / ci[block.M_DIFF_IX] - 1.0
    return f


def main():
    sys.stdout = Tee(os.path.join(RESULTS_DIR, 'run_qualification.log'))
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    use_cuda = torch.cuda.is_available()
    dev = torch.device('cuda' if use_cuda else 'cpu')
    cfg = obc_config(obc=True, device='cuda' if use_cuda else 'cpu', compile_mode=None,
                     nf_override=20, stride=1000)
    out = {'dataset': cfg.mode, 'device': str(dev), 'torch': torch.__version__}
    print('dataset %s | device %s | production attach with every valid training sample' % (cfg.mode, dev))

    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    data = load_datasets(cfg); norm = compute_normalization(cfg, data); hp = cfg.hp
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    t0 = time.perf_counter()
    fs = build_model(hp, cfg, data, norm)             # attaches OBC (cfg.obc=True)
    print('build_model (incl. reference set): %.1f s' % (time.perf_counter() - t0))
    fs.simulator = build_closed_loop(fs, norm, cfg, train_files=TRAIN_FILES, val_files=VAL_FILES,
                                     val_data=data.val_ckpt_data, verbose=False)
    life, corr, ref = fs.obc, fs.hfn.obc_correction, fs.obc_reference
    rig = Rig(cfg, data, norm, hp, fs, life, corr, ref)
    phy = corr._phy[0]
    if use_cuda:
        fs.cuda()
    out['reference'] = {'n_tuples': ref.n, 'n_per_record': ref.n_per_record,
                        'rows': ref.n * len(corr.row_ix), 'row_ix': list(corr.row_ix),
                        'hot_bytes': ref.memory_bytes(), 'all_bytes': ref.memory_bytes(False)}
    print('reference set: %d tuples, %d rows, hot tensors %.1f MB (all %.1f MB)'
          % (ref.n, out['reference']['rows'], ref.memory_bytes() / 1e6, ref.memory_bytes(False) / 1e6))

    # ---- the two parameter points -----------------------------------------------------
    from model_augmentation.systems import gantry_ss as gss
    names = phy.PARAM_NAMES
    nominal_raw = torch.stack([getattr(gss, n) for n in names]).double()
    nominal_combo = phy.combos_of(nominal_raw, float(gss.Lb))
    v_detuned = torch.zeros(10, dtype=torch.float64)                  # the block's combo_init
    v_nominal = free_coordinate_for(phy, nominal_combo)
    with torch.no_grad():
        chk = phy.combinations_from_free(v_nominal.to(phy.combo_init)).double().cpu()
    print('nominal point check: max rel |combos(v_nominal) - nominal| = %.2e'
          % float(((chk - nominal_combo) / nominal_combo).abs().max()))
    points = {'nominal': v_nominal, 'detuned_10pct': v_detuned}
    builder = life.basis_builder
    quals, bases, Js = {}, {}, {}
    for pname, v in points.items():
        print('\n==== basis at %s: v = %s' % (pname, ' '.join('%+.4f' % x for x in v)))
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(); m0 = torch.cuda.memory_allocated()
        J, secs = builder(v.to(dev))       # build on the device, like a training refresh
        if use_cuda:
            cuda_sync(); peak = torch.cuda.max_memory_allocated() - m0
            print('  build %.1f s on %s, peak device memory during build %.1f MB' % (secs, dev, peak / 1e6))
        else:
            print('  build %.1f s' % secs)
        Jc = J.cpu()
        q = qualify_basis(Jc, ref, corr.row_ix, rtol=cfg.obc_rank_rtol, names=COMBO_NAMES)
        q['build_seconds'] = secs
        quals[pname] = q
        bases[pname] = OBCBasis.from_matrix(Jc, v, rtol=cfg.obc_rank_rtol, build_seconds=secs)
        Js[pname] = Jc
        del J
    out['qualification'] = quals
    rank_ok = all(q['rank'] == 10 for q in quals.values())
    print('\nrank ten at both points: %s' % rank_ok)
    # basis movement between the two points
    out['r_J_nominal_vs_detuned'] = bases['detuned_10pct'].relative_change(bases['nominal'])
    print('||J_detuned - J_nominal||_F / ||J_nominal||_F = %.4e' % out['r_J_nominal_vs_detuned'])

    # ---- truth-parameter check and the rho table (nominal point) -----------------------
    print('\n==== truth plant parameter check (one-step prediction of the recorded next state)')
    hyp = check_truth_parameters(ref, hp['up_sample'], TRUTH_HYPOTHESES)
    for h in hyp:
        print('  %-24s rel RMS %.3e  per row %s' % (h['name'], h['rel_rms'], ' '.join('%.2e' % v for v in h['rel_rms_per_row'])))
    best = min(hyp, key=lambda h: h['rel_rms'])
    out['truth_check'] = hyp; out['truth_used'] = best['name']
    print('  using %s (smallest error) for delta' % best['name'])
    rho = {}
    for pname in ('nominal', 'detuned_10pct'):
        v = points[pname]
        # float64 baseline transition at v on every reference tuple, all six physical rows
        blk64 = builder.blk64
        base64 = evaluate_step_rows(lambda vv, z: blk64.transition_from_free(vv, z), v.to(dev),
                                    ref.Z_step, list(range(6)), chunk=cfg.obc_ref_chunk, device=dev)
        base_n = base64.reshape(ref.n, 6).cpu()
        deltas = discrepancy_fields(ref, base_n, norm, hp['up_sample'], best['ma_frac'], best['zeta_a'], corr.row_ix)
        J = Js[pname]
        Gamma = base_n[:, list(corr.row_ix)].reshape(-1) - J @ v          # Gamma_k = f_base - J_k vbar
        r = rho_table(J, Gamma, deltas)
        rho[pname] = r
        print('\n==== rho(B; delta) at %s: rank(J)=%d rank([J|Gamma])=%d, Gamma outside span(J): %.3f'
              % (pname, r['rank_J'], r['rank_JG'], r['gamma_outside_J']))
        print('  %-10s %12s %12s %12s' % ('delta', 'J', '[J|Gamma]', '||delta||'))
        for key in ('recorded', 'zero', 'data'):
            e = r['rho'][key]
            print('  %-10s %12.4f %12.4f %12.3e' % (key, e['J'], e['J|Gamma'], e['delta_norm']))
    out['rho'] = rho

    # ---- decimation comparison (diagnostic only; primary uses every sample) -------------
    # Column spaces of a full and a decimated stack live in different row spaces, so principal
    # angles between them are not defined. What IS comparable: the singular values (scaled by
    # sqrt(stride)) and the COEFFICIENT the decimated set assigns to the same field, i.e. whether
    # the decimated pseudo-inverse solve lands on the same theta.
    print('\n==== reference decimation (diagnostic only): scaled singular values and coefficient agreement')
    perturb_output_layer(rig.ann, 1e-3)
    Jf = Js['detuned_10pct']; nr = len(corr.row_ix)
    v_d = points['detuned_10pct']
    with torch.no_grad():
        F_ann = life.reference_field(rig.ann).detach().cpu().double()
    g = torch.Generator().manual_seed(0)
    fields = {'ann': F_ann}
    for k in range(3):
        R = torch.randn(F_ann.shape, generator=g, dtype=torch.float64)
        fields['random%d' % k] = R * (F_ann.norm() / R.norm())
    th_full = {k: bases['detuned_10pct'].coefficient(F) for k, F in fields.items()}
    dec = {}
    for stride in (10, 100, 1000):
        keep = torch.arange(ref.n)[::stride]
        rows = (keep[:, None] * nr + torch.arange(nr)[None, :]).reshape(-1)
        Jd = Jf[rows]
        bd = OBCBasis.from_matrix(Jd, v_d)
        Sd = torch.tensor(bd.report.singular_values) * math.sqrt(stride)
        Sf = torch.tensor(quals['detuned_10pct']['singular_values'])
        th_err = {k: float((bd.coefficient(F[rows]) - th_full[k]).norm() / th_full[k].norm())
                  for k, F in fields.items()}
        dec[str(stride)] = {'n': int(len(keep)), 'rank': bd.rank, 'cond': bd.report.cond,
                            'sv_scaled': Sd.tolist(), 'sv_rel_err': ((Sd - Sf).abs() / Sf).tolist(),
                            'theta_rel_err': th_err}
        print('  stride %4d (N=%6d): rank %d cond %.3e, max rel scaled-sv error %.3e, '
              'theta rel error ann %.3e random %s'
              % (stride, len(keep), bd.rank, bd.report.cond, max(dec[str(stride)]['sv_rel_err']),
                 th_err['ann'], ' '.join('%.3e' % th_err['random%d' % k] for k in range(3))))
    out['decimation'] = dec
    del Js

    # ---- orthogonality floor and runtime at the full set, on the device ------------------
    print('\n==== full-set lifecycle on %s: refresh, coefficient, orthogonality, runtime, memory' % dev)
    perturb_output_layer(rig.ann, 1e-3)
    if use_cuda:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); m_base = torch.cuda.memory_allocated()
    t0 = time.perf_counter(); life.refresh(phy.free_params, epoch=0); t_ref = time.perf_counter() - t0
    if use_cuda:
        cuda_sync(); m_basis = torch.cuda.memory_allocated() - m_base
    print('  refresh (build + svd) %.1f s; basis factors %.1f MB (%s)' % (t_ref, life.basis.memory_bytes() / 1e6, life.basis.U.dtype))
    # coefficient with the full set, forward and backward, ten repetitions
    ts_f, ts_fb = [], []
    for i in range(10):
        cuda_sync(); t0 = time.perf_counter()
        th = life.coefficient(rig.ann); cuda_sync(); t1 = time.perf_counter()
        if use_cuda and i == 0:
            m_graph = torch.cuda.memory_allocated() - m_base
        th.sum().backward(); cuda_sync(); t2 = time.perf_counter()
        ts_f.append(t1 - t0); ts_fb.append(t2 - t0)
    fs.optimizer.zero_grad()
    print('  field + solve: median %.4f s forward, %.4f s with backward (N=%d)' % (sorted(ts_f)[5], sorted(ts_fb)[5], ref.n))
    # orthogonality: float64 construction and the float32 rollout path; floor from random fields
    d = life.stacked_diagnostics(rig.ann)
    from test_behavior import jvp_path_field, orthogonality_floor
    theta = life.last_theta.to(cfg.dtype_pt)
    r_roll = life.basis.r_perp(jvp_path_field(rig, theta, chunk=cfg.obc_ref_chunk))
    floor = orthogonality_floor(rig, n_draws=5)
    print('  r_perp float64 construction %.3e | rollout (float32 JVP) path %.3e | random-field floor max %.3e (5 draws) | r_overlap %.4f'
          % (d['r_perp'], r_roll, max(floor), d['r_overlap']))
    out['orthogonality_full'] = {'r_perp_f64': d['r_perp'], 'r_perp_rollout': r_roll, 'floor': floor,
                                 'r_overlap': d['r_overlap'], 'kappa': life.basis.report.cond}
    # per-step cost at batch 512 with the frozen coefficient (eager)
    batch, kw = make_batch(rig, B=512, seed=0, device=dev)
    x0 = torch.randn(512, fs.hfn.nx, 1, device=dev, dtype=cfg.dtype_pt) * 0.1
    u0 = torch.randn(512, 3, 1, device=dev, dtype=cfg.dtype_pt) * 0.1
    def bench(theta, n=30, backward=False):
        xs = x0.clone().requires_grad_(backward)
        for _ in range(3):
            _, xp = fs.hfn(xs, u0, theta)
            if backward: xp.sum().backward()
        cuda_sync(); t0 = time.perf_counter()
        for _ in range(n):
            _, xp = fs.hfn(xs, u0, theta)
            if backward: xp.sum().backward()
        cuda_sync(); return (time.perf_counter() - t0) / n
    saved = fs.hfn.obc_correction; fs.hfn.obc_correction = None
    off_f, off_fb = bench(None), bench(None, backward=True)
    fs.hfn.obc_correction = saved
    on_f, on_fb = bench(theta), bench(theta, backward=True)
    # total update time: one full objective (nf=20 here) + backward, on vs off
    def update(on):
        fs.hfn.obc_correction = saved if on else None
        fs.obc = life if on else None
        cuda_sync(); t0 = time.perf_counter()
        fs.optimizer.zero_grad(); L = fs.loss(*batch[:4], **kw); L.backward(); cuda_sync()
        return time.perf_counter() - t0
    upd_off = sorted(update(False) for _ in range(3))[1]
    upd_on = sorted(update(True) for _ in range(3))[1]
    fs.hfn.obc_correction = saved; fs.obc = life
    rt = {'step_fwd_off_ms': off_f * 1e3, 'step_fwd_on_ms': on_f * 1e3, 'step_fwd_bwd_off_ms': off_fb * 1e3,
          'step_fwd_bwd_on_ms': on_fb * 1e3, 'ratio_fwd': on_f / off_f, 'ratio_fwd_bwd': on_fb / off_fb,
          'update_off_s': upd_off, 'update_on_s': upd_on, 'update_ratio': upd_on / upd_off,
          'field_solve_fwd_s': sorted(ts_f)[5], 'field_solve_fwd_bwd_s': sorted(ts_fb)[5], 'refresh_s': t_ref,
          'nf_for_update': hp['nf'], 'batch': 512}
    out['runtime'] = rt
    print('  per step (batch 512, eager %s): fwd off %.3f ms on %.3f ms (%.2fx); fwd+bwd off %.3f ms on %.3f ms (%.2fx)'
          % (dev, off_f * 1e3, on_f * 1e3, on_f / off_f, off_fb * 1e3, on_fb * 1e3, on_fb / off_fb))
    print('  one update (nf=%d, batch 512, incl. field+solve): off %.3f s, on %.3f s (%.2fx)' % (hp['nf'], upd_off, upd_on, upd_on / upd_off))
    mem = {'reference_hot_bytes': ref.memory_bytes(), 'basis_bytes': life.basis.memory_bytes()}
    if use_cuda:
        mem.update({'device_basis_delta_bytes': int(m_basis), 'device_retained_graph_delta_bytes': int(m_graph - m_basis),
                    'device_peak_bytes': int(torch.cuda.max_memory_allocated())})
        print('  device memory: basis %.1f MB, retained ANN graph after the field pass %.1f MB, peak %.1f MB'
              % (m_basis / 1e6, (m_graph - m_basis) / 1e6, torch.cuda.max_memory_allocated() / 1e6))
    out['memory'] = mem

    out['dataset_qualified_rank10'] = rank_ok
    with open(os.path.join(RESULTS_DIR, 'run_qualification.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print('\nwritten', os.path.join(RESULTS_DIR, 'run_qualification.json'))


if __name__ == '__main__':
    main()
