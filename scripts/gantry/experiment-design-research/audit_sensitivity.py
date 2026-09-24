"""Per-record-family audit of parameter informativity and of Gyorok's Condition 4 (read-only).

For each group of records it builds, with the PRODUCTION machinery of the OBC path
(`build_reference_set`, `build_stacked_sensitivity`, `evaluate_step_rows`; the same imports as
`orthogonal-by-construction/baseline-and-added-dynamics/measure_delta_orthogonality.py`):

  J        stacked one-step sensitivity of the six protected rows to the ten identifiable
           combinations, at theta*, on that group's reference tuples
  rank, cond(J), sigma_min / sqrt(n_rows)                        (Gyorok 2026 Assumption 1, Eq. 7)
  collinearity index gamma = 1 / sqrt(lambda_min(S~^T S~)), columns normalised to unit length
           # THEORY: Brun, Reichert, Kuensch 2001, Water Resour. Res., Eq. (12) normalisation and
           # Eq. (13) index; critical range 5 to 20 "according to our experience" (p. 6).
  cos(cg1, cg2) and the most collinear column pair (uncentred cosine, Brun's convention)
  rho* = ||P Delta*|| / ||Delta*||, and the bias J^+ Delta* in training's percent convention
           # THEORY: Gyorok et al. 2026, Condition 4 Eq. (17) and error Eq. (21).

Groups: each standstill record alone (local, frozen-Y identifiability), each family, all T,
all training records (T + TP), and E4 (motion with the multisine OFF: what the reference motion
alone carries). Coverage: for every validation and test tuple, the nearest-neighbour distance to
the training tuples in the normalised block-input space z = [x; u], divided by the median
within-training nearest-neighbour distance (HEURISTIC coverage ratio; the space-filling
literature measures coverage by fill distance, this is its per-point version).

Stride: OBC_REF_STRIDE (default 7). baseline-and-added-dynamics.md caveat 4: rho* is
stride-robust, J^+ Delta* is not at stride >= 97. Writes outputs/audit_sensitivity.json.
"""
__project_origin__ = "added"

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
MEAS = os.path.join(REPO, 'scripts', 'gantry', 'orthogonal-by-construction',
                    'baseline-and-added-dynamics')
for _p in (REPO, os.path.join(REPO, 'scripts', 'gantry'), MEAS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_augmentation.fit_systems.blocks import Reduced_Gantry_State_Block      # noqa: E402
from model_augmentation.fit_systems.obc import (                                  # noqa: E402
    OBCBasis, build_stacked_sensitivity, evaluate_step_rows)
from model_augmentation.systems.gantry_ss import P as P_pt                         # noqa: E402
from gantry_dynamic.config import RunConfig                                        # noqa: E402
from gantry_dynamic.data import TRAIN_FILES                                         # noqa: E402
from gantry_dynamic.obc_gantry import (                                            # noqa: E402
    build_reference_set, fourth_order_central_difference, _load_record_float64)
from measure_delta_orthogonality import (                                          # noqa: E402
    COMBO_INIT_DETUNE, ROWS, combo_scale, free_of_combinations, percentages,
    stack_rows, truth_combinations)

MODE = 'augmentation_ma50_z03_b140-230_a6_telica'
COMBO = Reduced_Gantry_State_Block.COMBO_NAMES
ST = ['T1_standstill_Ym30', 'T2_standstill_Ym15', 'T3_standstill_Y000', 'T4_standstill_Yp15',
      'T5_standstill_Yp30']
FAM = {
    'standstill': ST,
    'sweep': ['T6_ysweep_slow', 'T7_ysweep_fast', 'T8_ysweep_xmix'],
    'aprbs': ['T9_aprbs_30', 'T10_aprbs_60', 'T11_aprbs_100', 'T12_aprbs_yaw'],
    'lissajous': ['T13_lissajous', 'T14_lissajous_yaw'],
    'telica_profile': ['TP1_telica_y000', 'TP2_telica_y000_rev', 'TP3_telica_yp06',
                       'TP4_telica_ym06'],
}
GROUPS = {n: [n] for n in ST}
GROUPS.update(FAM)
GROUPS['all_T'] = [n for f in ('standstill', 'sweep', 'aprbs', 'lissajous') for n in FAM[f]]
GROUPS['all_train'] = GROUPS['all_T'] + FAM['telica_profile']
GROUPS['E4_motion_only'] = ['E4_multisine_off']
VAL = ['V1_standstill_Yp10', 'V2_aprbs_Ylow', 'V3_ysweep_Yp10', 'V4_lissajous_Ym10',
       'VP1_telica_ylow', 'VP2_telica_yp12']
TEST = ['E1_resonance_sweep', 'E2_multisine_Yp22', 'E3_aprbs_above', 'E4_multisine_off',
        'EP1_telica_test']


def next_states_stencil(cfg, norm, files, stride):
    """`[q[k+1]; D4 q[k+1]]` on the reference tuples of `files`, normalised (measure script's
    'stencil' variant, generalised to a file list)."""
    x_mean = np.asarray(norm.x_mean, np.float64).reshape(1, 6)
    std_x = np.asarray(norm.std_x, np.float64).reshape(1, 6)
    PinvT = np.linalg.inv(P_pt.detach().cpu().numpy().astype(np.float64).T)
    nxt, valid = [], []
    for f in files:
        u, y, _xl, _ = _load_record_float64(f, cfg)
        q = y @ PinvT.T
        qd = fourth_order_central_difference(q, cfg.ts_new)
        k = np.arange(2, len(u) - 2)
        if stride > 1:
            k = k[::stride]
        idx = k + 1 - 2
        nxt.append(np.hstack([q[np.clip(k + 1, 0, len(q) - 1)], qd[np.clip(idx, 0, len(qd) - 1)]]))
        valid.append((idx < len(qd)) & (k + 1 < len(q)))
    return (np.concatenate(nxt) - x_mean) / std_x, np.concatenate(valid)


def light_norm(cfg):
    """The four statistics `build_reference_set` and the block read from `Norm`, computed with
    the formulas of `data.py::compute_normalization` (lines 214-251) record by record, so the
    29-record `load_datasets` (about 1 GB resident, which tripped the watchdog's RAM guard with
    about 3.5 GB free on the host) is not needed. float32 cast as the pipeline's train_list."""
    PinvT = np.linalg.inv(P_pt.detach().cpu().numpy().T).astype(np.float32)
    xs, us = [], []
    for f in TRAIN_FILES:
        u, y, _xl, _ = _load_record_float64(f, cfg)
        u, y = u.astype(np.float32), y.astype(np.float32)
        pos = (PinvT @ y.T).T
        vel = np.diff(pos, axis=0) / np.float32(cfg.ts_new)
        vel = np.vstack([vel[:1], vel])
        xs.append(np.hstack([pos, vel]))
        us.append(u)
    x_all, u_all = np.concatenate(xs), np.concatenate(us)
    from types import SimpleNamespace
    return SimpleNamespace(
        x_mean=x_all.mean(0).reshape(6, 1).astype(np.float32),
        std_x=x_all.std(0).reshape(6, 1).astype(np.float32) + 1e-8,
        std_u=u_all.std(0).reshape(3, 1).astype(np.float32) + 1e-8,
        u_mean=u_all.mean(0).reshape(3, 1).astype(np.float32))


def collinearity(J):
    S = J / np.linalg.norm(J, axis=0, keepdims=True)
    G = S.T @ S
    lam = np.linalg.eigvalsh(G)
    gamma = 1.0 / np.sqrt(max(lam[0], 1e-300))
    C = np.abs(G - np.eye(G.shape[0]))
    i, j = np.unravel_index(np.argmax(C), C.shape)
    return gamma, G, (COMBO[i], COMBO[j], float(G[i, j]))


def main():
    stride = int(os.environ.get('OBC_REF_STRIDE', '7'))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = RunConfig(mode=MODE, encoder_init='linear_map', joint_estimation=True,
                    physics_parameterization='reduced', combo_init_detune=COMBO_INIT_DETUNE,
                    param_init_detune=None, obc=True, obc_space='tangent', nx_ann=8, up_sample=1,
                    stride=10, fs_new=4000, snr=None, save_flag=False)
    norm = light_norm(cfg)
    print('[norm] std_x', norm.std_x.ravel(), 'std_u', norm.std_u.ravel(), flush=True)
    true_combo, raw = truth_combinations()
    block = Reduced_Gantry_State_Block(
        params_init=raw, combo_init=true_combo * torch.tensor(COMBO_INIT_DETUNE, dtype=torch.float64),
        Y_op=None, std_x=norm.std_x, std_u=norm.std_u, x_mean=norm.x_mean, u_mean=norm.u_mean,
        Ts=cfg.ts_new, up_sample=cfg.up_sample, RMSE_baseline=cfg.param_rmse_baseline,
        flag_loss_reg=False).to(torch.float64)
    vbar = free_of_combinations(block, true_combo)
    scale = combo_scale()
    block.to(dev)
    step = lambda v, z: block.transition_from_free(v, z)          # noqa: E731
    print('[setup] mode=%s stride=%d device=%s ts=%.3g' % (MODE, stride, dev, cfg.ts_new), flush=True)

    out_path = os.path.join(HERE, 'outputs', 'audit_sensitivity.json')
    prev = json.load(open(out_path)) if os.path.exists(out_path) else {}
    res = prev.get('groups', {})
    sel = [g for g in os.environ.get('AUDIT_GROUPS', ','.join(GROUPS)).split(',') if g]
    for g, files in ((g, GROUPS[g]) for g in sel if g in GROUPS):
        t0 = time.perf_counter()
        ref = build_reference_set(cfg, norm, nx_ann=cfg.nx_ann, files=files, stride=stride,
                                  verbose=False)
        J = build_stacked_sensitivity(step, vbar.to(dev), ref.Z_step, ROWS,
                                      chunk=cfg.obc_ref_chunk, device=dev).cpu()
        f_base = evaluate_step_rows(step, vbar.to(dev), ref.Z_step, ROWS,
                                    chunk=cfg.obc_ref_chunk, device=dev).cpu()
        xn, valid = next_states_stencil(cfg, norm, files, stride)
        assert xn.shape[0] == ref.n
        D = stack_rows(xn) - f_base
        mask = torch.from_numpy(np.repeat(valid, len(ROWS)))
        D = torch.where(mask, D, torch.zeros_like(D))
        Jn = J.numpy()
        sv = np.linalg.svd(Jn, compute_uv=False)
        rank = int((sv > sv[0] * 1e-12).sum())
        gamma, G, worst = collinearity(Jn)
        i1, i2 = COMBO.index('cg1'), COMBO.index('cg2')
        b = OBCBasis.from_matrix(J, vbar.cpu())
        rho = b.r_overlap(D)
        dth = b.coefficient(D).numpy()
        # Linearised bias: nine log coordinates and relative-linear m_diff, so 100 * dtheta is the
        # first-order percent error of each combination. `percentages` exponentiates, which is
        # meaningless once |dtheta| >> 1 (ill-conditioned single records).
        lin = 100.0 * dth
        pct = percentages(block.to('cpu'), vbar + b.coefficient(D), true_combo, scale)
        block.to(dev)
        # direction of the weakest singular vector
        _, _, Vh = np.linalg.svd(Jn / np.linalg.norm(Jn, axis=0), full_matrices=False)
        weak = COMBO[int(np.argmax(np.abs(Vh[-1])))]
        r = dict(files=files, n_tuples=int(ref.n), rank=rank, cond=float(sv[0] / sv[-1]),
                 sigma=sv.tolist(), sigma_min_per_row=float(sv[-1] / np.sqrt(Jn.shape[0])),
                 gamma=float(gamma), cos_cg1_cg2=float(G[i1, i2]), worst_pair=worst,
                 weakest_direction=weak, rho=float(rho),
                 bias_pct=dict(zip(COMBO, [float(v) for v in pct])),
                 bias_rms=float(np.sqrt(np.mean(pct ** 2))),
                 bias_lin_pct=dict(zip(COMBO, [float(v) for v in lin])),
                 bias_lin_rms=float(np.sqrt(np.mean(lin ** 2))))
        res[g] = r
        big = COMBO[int(np.argmax(np.abs(lin)))]
        print('%-18s n=%6d rank=%2d cond=%9.3e gamma=%8.2f cos(cg1,cg2)=%+.3f worst=%s/%s %+.3f '
              'weak=%-7s rho*=%.4f linbiasRMS=%9.2f%% largest=%s %+.1f%%  (%.1fs)'
              % (g, ref.n, rank, r['cond'], gamma, r['cos_cg1_cg2'], worst[0], worst[1], worst[2],
                 weak, rho, r['bias_lin_rms'], big, lin[COMBO.index(big)],
                 time.perf_counter() - t0), flush=True)
        del J, b
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
        with open(out_path, 'w') as fh:           # incremental: a kill loses one group at most
            json.dump(dict(prev, mode=MODE, stride=stride, combo=list(COMBO), groups=res), fh,
                      indent=1)
    if os.environ.get('AUDIT_COVERAGE', '1') != '1':
        return
    cstride = int(os.environ.get('AUDIT_COV_STRIDE', str(stride)))

    # coverage of val/test by the training tuples in z = [x; u] (normalised)
    from scipy.spatial import cKDTree
    Z = lambda fl: build_reference_set(cfg, norm, nx_ann=cfg.nx_ann, files=fl, stride=cstride,  # noqa: E731
                                       verbose=False).Z_step.squeeze(-1).double().cpu().numpy()
    Ztr = Z(GROUPS['all_train'])
    tree = cKDTree(Ztr)
    d_in, _ = tree.query(Ztr, k=2)
    h = float(np.median(d_in[:, 1]))
    cov = {}
    for f in VAL + TEST:
        dq, _ = tree.query(Z([f]), k=1)
        cov[f] = dict(median_ratio=float(np.median(dq) / h), p95_ratio=float(np.percentile(dq, 95) / h),
                      frac_over_10=float((dq > 10 * h).mean()))
        print('coverage %-20s NN/h median %.2f  p95 %.2f  frac>10h %.3f'
              % (f, cov[f]['median_ratio'], cov[f]['p95_ratio'], cov[f]['frac_over_10']), flush=True)
    with open(out_path, 'w') as fh:
        json.dump(dict(mode=MODE, stride=stride, combo=list(COMBO), groups=res,
                       coverage=dict(h_train_median_nn=h, per_record=cov, stride=cstride)), fh,
                  indent=1)
    print('[out] outputs/audit_sensitivity.json')


if __name__ == '__main__':
    main()
