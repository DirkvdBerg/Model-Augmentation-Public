"""How non-orthogonal are the TRUE added dynamics, and what parameter bias does that predict?

The numerical companion to `derive_condition.py`. That script established, symbolically,

    dtheta = theta_hat - theta* = J^+ Delta*,   and   theta_hat = theta*  iff  J^T Delta* = 0,

with `J` the stacked baseline parameter sensitivity on the OBC reference set and `Delta*` the
true missing dynamics on the same set. This script measures both sides on the gantry data the
three-arm comparison actually ran on (`augmentation_ma50_b140-230_a6_z03`):

    rho*       = ||P Delta*|| / ||Delta*||          how non-orthogonal the truth is, in [0, 1]
    J^+ Delta*                                      the predicted parameter bias, per parameter
    RMS(pct)                                        the predicted `combo-err`, comparable to the
                                                    observed 7.69 percent final / 6.98 minimum

Everything is imported from the production path: `build_reference_set` for the tuples,
`build_stacked_sensitivity` and `evaluate_step_rows` for `J` and the baseline step, `OBCBasis`
for the projector. No second projector, no second reference set.

UNITS (D-202). `Reduced_Gantry_State_Block.transition_from_free` consumes the block's NORMALISED
input `z` and returns the NORMALISED next physical state, so `J` has normalised state rows.
`Delta*` is therefore formed in normalised coordinates,
`Delta* = (x_next_phys - x_mean) / std_x - step_base(theta*, z)`, which puts both sides of the
inner product in the frame OBC actually orthogonalises in.

NEXT-STATE STENCIL (D-203). Two variants of the truth's next state are reported:
  'recorded'  `x_logical[k+1]` straight from the record. Its velocity rows are a MATLAB
              `gradient()` of single-precision positions (D-197), a different stencil from the
              fourth-order central difference `build_reference_set` uses for `x_phys`, so the
              velocity rows of `Delta*` inherit the DIFFERENCE of two stencils.
  'stencil'   `[q[k+1]; D4 q[k+1]]`, the same fourth-order stencil on the same positions.
              This is the primary number; it removes the mismatch and is still data-derived.

NULL TEST. The floor of the whole construction is measured by replaying the BASELINE model over
short windows of the recorded input, throwing its exact velocities away, reconstructing them with
the same fourth-order stencil, and computing the same one-step residual. A trajectory that is
exactly a baseline trajectory must give a residual at the stencil floor; anything else is a units,
alignment or off-by-one error in this script.

Run (background, live output, per the project convention):
  PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output -n GraduationProject \
      python -u measure_delta_orthogonality.py
"""
__project_origin__ = "added"

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# CHANGED (vendor, OA-001): this copy lives in scripts/gantry/orthogonal-addition/measure/ and
# imports the VENDORED measurement path; the production scripts/gantry is NOT put on sys.path,
# so the vendored `gantry_dynamic` cannot be shadowed by the production package.
OA = os.path.abspath(os.path.join(HERE, '..'))
REPO = os.path.abspath(os.path.join(OA, '..', '..', '..'))
for _p in (REPO, OA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_augmentation.fit_systems.blocks import (            # noqa: E402
    Parameterized_Gantry_State_Block, Reduced_Gantry_State_Block)
from vendor.obc import (                                       # noqa: E402
    OBCBasis, build_stacked_sensitivity, evaluate_step_rows)
from model_augmentation.systems import gantry_ss as gss        # noqa: E402
from model_augmentation.systems.gantry_ss import P as P_pt     # noqa: E402

from vendor.gantry_dynamic.config import RunConfig             # noqa: E402
from vendor.gantry_dynamic.data import (                       # noqa: E402
    TRAIN_FILES, compute_normalization, load_datasets)
from vendor.gantry_dynamic.obc_gantry import (                 # noqa: E402
    build_reference_set, fourth_order_central_difference, _load_record_float64)

# CHANGED (OA-001): the dataset is a parameter. Default: the dataset rho* = 0.2046 was measured on.
ORIGINAL_MODE = 'augmentation_ma50_b140-230_a6_z03'
MODE = os.environ.get('OA_MODE', ORIGINAL_MODE)
OUT_DIR = os.path.join(OA, 'outputs')
# CHANGED (OA-019): lean mode for design scans: only the basis at theta* (the detuned basis costs
# another 650 MB and pushed a scan measurement under the watchdog's RAM floor, R-027s215).
LEAN = os.environ.get('OA_LEAN') == '1'

COMBO_NAMES = Reduced_Gantry_State_Block.COMBO_NAMES
ROWS = (0, 1, 2, 3, 4, 5)                 # the protected rows of both OBC arms
ROW_LABELS = ['X', 'Theta', 'Y', 'dX', 'dTheta', 'dY']
# Run 84033's detuning, in the order of COMBO_NAMES. Transcribed from the effective-config
# snapshot at the head of `server/useable/obc_arm84033.out`.
COMBO_INIT_DETUNE = [1.1, 1.1, 0.9, 1.1, 0.9, 0.9, 1.1, 1.1, 0.9, 1.1]
OBC_DATA = os.path.join(REPO, 'scripts', 'gantry', 'meeting', 'meeting-18-09-2026',
                        'figures', 'OBC', 'obc_data.json')

LINE = '=' * 90


def banner(title):
    print('\n' + LINE)
    print(title)
    print(LINE, flush=True)


# ------------------------------------------------------------------ truth and its coordinates
def truth_combinations() -> torch.Tensor:
    """The ten identifiable combinations at the TRUE parameters, float64.

    Single source: `gantry_ss`, the same module `training.py` reads for its `combo-err` meter,
    so this vector is the one the reported percentages are measured against by construction.
    """
    names = Parameterized_Gantry_State_Block.PARAM_NAMES
    raw = torch.stack([getattr(gss, n) for n in names]).to(torch.float64)
    return Reduced_Gantry_State_Block.combos_of(raw, float(gss.Lb)), raw


def combo_scale() -> np.ndarray:
    """The per-combination error scale the training meter uses (`training.py:418-424`).

    |true| for nine of the ten; for the signed `m_diff = m1 - m2` the mean actuator mass, so a
    ten percent detune reads about half a percent rather than 110 percent.
    """
    names = Parameterized_Gantry_State_Block.PARAM_NAMES
    raw = {n: float(getattr(gss, n)) for n in names}
    true_combo, _ = truth_combinations()
    scale = np.abs(true_combo.numpy()).copy()
    scale[Reduced_Gantry_State_Block.M_DIFF_IX] = 0.5 * (raw['m1'] + raw['m2'])
    return scale


def free_of_combinations(block: Reduced_Gantry_State_Block, combo: torch.Tensor) -> torch.Tensor:
    """Invert `combinations_from_free`: nine log coordinates, `m_diff` relative-linear."""
    init = block.combo_init.to(torch.float64)
    free = torch.log(combo / init)
    i = Reduced_Gantry_State_Block.M_DIFF_IX
    free[i] = combo[i] / init[i] - 1.0
    return free


def percentages(block, free, true_combo, scale):
    """The ten per-parameter errors in percent at a free coordinate, training's own convention."""
    dev = block.combo_init.device
    combo = block.combinations_from_free(
        free.detach().to(device=dev, dtype=torch.float64)).detach().cpu().numpy()
    return 100.0 * (combo - true_combo.numpy()) / scale


# ------------------------------------------------------------------ setup
def build_everything(stride: int, device):
    cfg = RunConfig(
        mode=MODE, encoder_init='linear_map',
        joint_estimation=True, physics_parameterization='reduced',
        combo_init_detune=COMBO_INIT_DETUNE, param_init_detune=None,
        obc=True, obc_space='tangent', nx_ann=8, up_sample=1, stride=10,
        fs_new=4000, snr=None, save_flag=False)
    print('[setup] mode=%s  ts_new=%.6g s  nx_phys=%d  nx_ann=%d'
          % (cfg.mode, cfg.ts_new, cfg.nx_phys, cfg.nx_ann), flush=True)

    data = load_datasets(cfg)
    norm = compute_normalization(cfg, data)

    true_combo, raw_nominal = truth_combinations()
    combo_init = true_combo * torch.tensor(COMBO_INIT_DETUNE, dtype=torch.float64)
    block = Reduced_Gantry_State_Block(
        params_init=raw_nominal, combo_init=combo_init,
        Y_op=None, std_x=norm.std_x, std_u=norm.std_u,
        x_mean=norm.x_mean, u_mean=norm.u_mean, Ts=cfg.ts_new,
        up_sample=cfg.up_sample, RMSE_baseline=cfg.param_rmse_baseline,
        flag_loss_reg=False).to(torch.float64)

    # The two expansion points, in this block's free coordinates.
    vbar_init = torch.zeros(10, dtype=torch.float64)          # the detuned start (run's epoch 0)
    vbar_true = free_of_combinations(block, true_combo)
    check = block.combinations_from_free(vbar_true).detach()
    err = float((check - true_combo).abs().max())
    print('[setup] vbar_true reproduces the true combinations to %.3e (absolute)' % err)
    assert err < 1e-10, 'the free coordinate of the truth is wrong'

    ref = build_reference_set(cfg, norm, nx_ann=cfg.nx_ann, stride=stride, verbose=True)
    return cfg, norm, block, ref, vbar_init, vbar_true, true_combo


# ------------------------------------------------------------------ Delta*
def next_states(cfg, norm, ref, stride):
    """The truth's next state on the reference tuples, both variants, in NORMALISED coordinates.

    Returns `(x_next_recorded, x_next_stencil, valid)` each `(N, 6)` / `(N,)`, aligned with
    `ref.Z_step`. `valid` is False on the last tuple of every record, where the fourth-order
    stencil at `k+1` does not exist.
    """
    x_mean = np.asarray(norm.x_mean, dtype=np.float64).reshape(1, 6)
    std_x = np.asarray(norm.std_x, dtype=np.float64).reshape(1, 6)
    P64 = P_pt.detach().cpu().numpy().astype(np.float64)
    PinvT = np.linalg.inv(P64.T)

    nxt_rec, nxt_sten, valid = [], [], []
    for f in TRAIN_FILES:
        u, y, x_log, _ = _load_record_float64(f, cfg)
        q = y @ PinvT.T
        qd = fourth_order_central_difference(q, cfg.ts_new)      # aligned with q[2:-2]
        k = np.arange(2, len(u) - 2)
        if stride > 1:
            k = k[::stride]
        # 'recorded': exactly what `GantryReferenceSet.x_phys_next` holds.
        nxt_rec.append(x_log[k + 1])
        # 'stencil': q at k+1 with the SAME fourth-order velocity, i.e. qd[(k+1) - 2].
        idx = k + 1 - 2
        ok = idx < len(qd)
        idx_safe = np.clip(idx, 0, len(qd) - 1)
        nxt_sten.append(np.hstack([q[np.clip(k + 1, 0, len(q) - 1)], qd[idx_safe]]))
        valid.append(ok & (k + 1 < len(q)))
    nxt_rec = np.concatenate(nxt_rec)
    nxt_sten = np.concatenate(nxt_sten)
    valid = np.concatenate(valid)
    assert nxt_rec.shape[0] == ref.n, 'next-state stacking lost alignment with the reference set'
    return ((nxt_rec - x_mean) / std_x, (nxt_sten - x_mean) / std_x, valid)


def stack_rows(x_next_norm, rows=ROWS):
    """Sample-major stack of the selected rows, matching `evaluate_step_rows`' ordering."""
    return torch.from_numpy(np.ascontiguousarray(x_next_norm[:, list(rows)])).reshape(-1)


# ------------------------------------------------------------------ null test
def null_test(cfg, norm, block, vbar_true, n_windows=56, nf=400, seed=0):
    """The floor of the construction: a BASELINE trajectory pushed through the same pipeline.

    Roll the baseline block forward `nf` steps from a recorded state, discard its exact
    velocities, rebuild them with the fourth-order stencil from its own positions, and form the
    same one-step residual. Because the rolled trajectory satisfies `x(k+1) = step(x(k), u(k))`
    exactly, everything that comes out is velocity-reconstruction error, not model error.

    Windows rather than whole records: the baseline driven open loop by a force recorded in
    CLOSED loop around the true plant drifts out of the scheduling range over twelve seconds,
    and a floor measured outside the operating range would not be the floor that matters. At
    `nf = 400` (0.1 s, the training window) it stays inside.

    Returns `{quantised: (residual, next_state)}` for float64 and for float32-quantised
    positions, from ONE rollout, so the two differ only in the quantisation.
    """
    rng = np.random.default_rng(seed)
    x_mean = np.asarray(norm.x_mean, dtype=np.float64).reshape(1, 6)
    std_x = np.asarray(norm.std_x, dtype=np.float64).reshape(1, 6)
    u_mean = np.asarray(norm.u_mean, dtype=np.float64).reshape(1, 3)
    std_u = np.asarray(norm.std_u, dtype=np.float64).reshape(1, 3)
    P64 = P_pt.detach().cpu().numpy().astype(np.float64)
    PinvT = np.linalg.inv(P64.T)

    t0 = time.perf_counter()
    x0n, uwin = [], []
    for w in range(n_windows):
        f = TRAIN_FILES[w % len(TRAIN_FILES)]
        u, y, _x_log, _ = _load_record_float64(f, cfg)
        q = y @ PinvT.T
        qd = fourth_order_central_difference(q, cfg.ts_new)
        m0 = int(rng.integers(2, len(u) - nf - 4))
        x0n.append((np.hstack([q[m0], qd[m0 - 2]]).reshape(1, 6) - x_mean) / std_x)
        uwin.append((u[m0:m0 + nf] - u_mean) / std_u)
    W = n_windows
    xk = torch.from_numpy(np.concatenate(x0n)).reshape(W, 6, 1)
    un = torch.from_numpy(np.stack(uwin, axis=1)).reshape(nf, W, 3, 1)   # (nf, W, 3, 1)

    # Roll every window at once. `transition_from_free` is the same map `J` differentiates.
    xs = torch.empty(nf + 1, W, 6, dtype=torch.float64)
    xs[0] = xk[:, :, 0]
    with torch.no_grad():
        for i in range(nf):
            xk = block.transition_from_free(vbar_true, torch.cat([xk, un[i]], dim=1))
            xs[i + 1] = xk[:, :, 0]
    Xb = xs.numpy()                                        # (nf+1, W, 6) normalised
    print('[null] %d windows x %d steps rolled in %.1f s'
          % (W, nf, time.perf_counter() - t0), flush=True)

    q_f64 = Xb[:, :, :3] * std_x[0, :3] + x_mean[0, :3]    # (nf+1, W, 3) physical positions
    m = np.arange(2, nf + 1 - 3)                           # the stencil must exist at m AND m+1
    out = {}
    for quant in (False, True):
        # The records store positions as MATLAB `single`; matching that is what makes this a
        # floor for the REAL Delta*, not for an idealised float64 one.
        qb = q_f64.astype(np.float32).astype(np.float64) if quant else q_f64
        qdb = fourth_order_central_difference(qb, cfg.ts_new)      # aligned with qb[2:-2]
        x_t = np.concatenate([qb[m], qdb[m - 2]], axis=-1).reshape(-1, 6)
        x_t1 = np.concatenate([qb[m + 1], qdb[m - 1]], axis=-1).reshape(-1, 6)
        xn_t = torch.from_numpy((x_t - x_mean) / std_x).reshape(-1, 6, 1)
        xn_t1 = (x_t1 - x_mean) / std_x
        uu = un[m].reshape(-1, 3, 1)
        with torch.no_grad():
            pred = block.transition_from_free(vbar_true, torch.cat([xn_t, uu], dim=1))
        res = (xn_t1 - pred[:, :, 0].numpy())[:, list(ROWS)]
        out[quant] = (res, xn_t1[:, list(ROWS)], torch.cat([xn_t, uu], dim=1))
    print('[null] %d tuples per variant' % out[True][0].shape[0], flush=True)
    return out


# ------------------------------------------------------------------ reporting
def row_norms(vec_flat, n_rows=len(ROWS)):
    """Per-protected-row RMS of a sample-major stacked field."""
    a = np.asarray(vec_flat).reshape(-1, n_rows)
    return {ROW_LABELS[i]: float(np.sqrt((a[:, i] ** 2).mean())) for i in range(n_rows)}


def report_bias(tag, basis, Delta, block, vbar_expansion, true_combo, scale, obs=None):
    dtheta = basis.coefficient(Delta)                    # J^+ Delta*
    rho = basis.r_overlap(Delta)
    free_hat = vbar_expansion.to(torch.float64) + dtheta.cpu()
    pct = percentages(block, free_hat, true_combo, scale)
    rms = float(np.sqrt((pct ** 2).mean()))
    # CHANGED (OA-001): m_diff on its OWN scale (|m1 - m2| = 0.5 kg), handoff sect. 9 G4; the
    # combo meter divides it by the 10.45 kg mean actuator mass and so understates it 21x.
    i_md = Reduced_Gantry_State_Block.M_DIFF_IX
    m_diff_own = float(pct[i_md] * scale[i_md] / abs(float(true_combo[i_md])))
    print('\n--- %s' % tag)
    print('    rho* = ||P Delta||/||Delta|| = %.6f      ||Delta|| = %.6e' % (
        rho, float(torch.linalg.vector_norm(Delta.to(torch.float64)))))
    print('    predicted combo-err (RMS over the ten) = %.3f %%' % rms)
    print('    m_diff on its own scale (|m1-m2|) = %+.3f %%' % m_diff_own)
    print('    %-9s %12s %12s' % ('param', 'predicted %', 'observed %' if obs else ''))
    for i, n in enumerate(COMBO_NAMES):
        o = ('%12.2f' % obs[n]) if obs else ''
        print('    %-9s %12.2f %s' % (n, pct[i], o))
    # WHERE the bias comes from, direction by direction. `dtheta = sum_i (u_i^T Delta / s_i) v_i`,
    # so a bias carried by the small singular values is a CONDITIONING effect: a little of Delta*
    # along a weakly excited parameter direction buys a large parameter displacement.
    with torch.no_grad():
        D64 = Delta.to(torch.float64)
        proj = (basis.U.T @ D64)                       # u_i^T Delta
        coef = proj / basis.S                          # the contribution's size in v_i
    share = (coef ** 2) / (coef ** 2).sum()
    print('    singular-direction breakdown (i, sigma_i, |u_i^T Delta|, |c_i|, share of'
          ' ||dtheta||^2, dominant column):')
    for i in range(basis.rank):
        j = int(torch.argmax(basis.Vh[i].abs()))
        print('      %2d  sigma %10.4e   |u^T D| %10.4e   |c| %10.4e   %5.1f %%   %s'
              % (i, float(basis.S[i]), abs(float(proj[i])), abs(float(coef[i])),
                 100.0 * float(share[i]), COMBO_NAMES[j]))
    return dict(tag=tag, rho=rho, pct=pct.tolist(), rms=rms, m_diff_own_pct=m_diff_own,
                delta_norm=float(torch.linalg.vector_norm(Delta.to(torch.float64))),
                jt_delta=basis.jt(Delta.to(torch.float64)).cpu().numpy().tolist(),
                dtheta=dtheta.cpu().numpy().tolist(),
                sigma=[float(v) for v in basis.S],
                ut_delta=[float(v) for v in proj],
                direction_share=[float(v) for v in share])


def main():
    stride = int(os.environ.get('OBC_REF_STRIDE', '1'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    banner('SETUP  (reference stride %d, device %s)' % (stride, device))
    cfg, norm, block, ref, vbar_init, vbar_true, true_combo = build_everything(stride, device)
    scale = combo_scale()

    # ------------------------------------------------------------------ calibration
    banner('CALIBRATION  the detuned start must reproduce obc_data.json iteration 0')
    obc = json.load(open(OBC_DATA))
    arm = next(a for a in obc['arms'] if a['key'] == 'obc')
    start = arm['records'][0]['params']
    pct0 = percentages(block, vbar_init, true_combo, scale)
    worst = 0.0
    for i, n in enumerate(COMBO_NAMES):
        worst = max(worst, abs(pct0[i] - start[n]))
        print('    %-9s  this script %+8.3f %%   obc_data.json %+8.3f %%' % (n, pct0[i], start[n]))
    print('    largest disagreement %.4f percentage points' % worst)
    assert worst < 0.01, 'the percentage convention does not match the training meter'
    print('    RMS %.4f %% against the logged combo_start %.4f %%'
          % (float(np.sqrt((pct0 ** 2).mean())), arm['combo_start']))

    # ------------------------------------------------------------------ null test
    banner('NULL TEST  a baseline trajectory through the same construction')
    null = null_test(cfg, norm, block, vbar_true)
    for quant in (False, True):
        res, sig, _z = null[quant]
        print('    quantise positions to float32: %s' % quant)
        print('      ||residual|| / ||next state|| = %.3e'
              % (np.linalg.norm(res) / np.linalg.norm(sig)))
        print('      per-row RMS residual: %s' % {k: '%.3e' % v
                                                  for k, v in row_norms(res.reshape(-1)).items()})
    null_rms = float(np.sqrt((null[True][0] ** 2).mean()))
    null_rms_f64 = float(np.sqrt((null[False][0] ** 2).mean()))
    print('    floor RMS: %.4e (float32 positions, the realistic one), %.4e (float64)'
          % (null_rms, null_rms_f64))

    # What parameter bias would the FLOOR ALONE predict? The same J^+ applied to the null
    # residual on the null tuples. If this is comparable to the prediction from Delta*, the
    # prediction is an artefact of the velocity reconstruction and nothing can be concluded.
    res_null, _sig, z_null = null[True]
    block.to(device)
    J_null = build_stacked_sensitivity(lambda v, z: block.transition_from_free(v, z),
                                       vbar_true.to(device), z_null, ROWS,
                                       chunk=cfg.obc_ref_chunk, device=device)
    b_null = OBCBasis.from_matrix(J_null.cpu(), vbar_true.cpu())
    d_null = torch.from_numpy(np.ascontiguousarray(res_null)).reshape(-1)
    pct_null = percentages(block, vbar_true.cpu() + b_null.coefficient(d_null),
                           true_combo, scale)
    floor_rms_pct = float(np.sqrt((pct_null ** 2).mean()))
    print('    FLOOR BIAS: rho_null = %.4f, J_null^+ Delta_null predicts %.3f %% combo-err'
          % (b_null.r_overlap(d_null), floor_rms_pct))
    print('      ' + '  '.join('%s %+.2f' % (n, v) for n, v in zip(COMBO_NAMES, pct_null)))
    print('      This is the parameter bias the numerical floor alone would produce. The')
    print('      prediction from Delta* is only meaningful well above it.')
    del J_null
    block.to('cpu')
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ Delta*
    banner('DELTA*  the true added dynamics on the reference set')
    x_next_rec, x_next_sten, valid = next_states(cfg, norm, ref, stride)
    print('[delta] %d of %d tuples carry a valid fourth-order stencil at k+1' % (valid.sum(), ref.n))

    block.to(device)          # the heavy parts run where the Jacobian is built
    f_base = evaluate_step_rows(lambda v, z: block.transition_from_free(v, z),
                                vbar_true.to(device), ref.Z_step, ROWS,
                                chunk=cfg.obc_ref_chunk, device=device).cpu()
    deltas = {}
    for tag, xn in (('recorded', x_next_rec), ('stencil', x_next_sten)):
        d = stack_rows(xn) - f_base
        mask = torch.from_numpy(np.repeat(valid, len(ROWS)))
        d = torch.where(mask, d, torch.zeros_like(d))     # drop the invalid last-of-record tuples
        deltas[tag] = d
        ref_norm = float(torch.linalg.vector_norm(stack_rows(xn)))
        print('    %-9s ||Delta|| = %.6e  (%.3e of ||x_next||)   per-row RMS %s'
              % (tag, float(torch.linalg.vector_norm(d)),
                 float(torch.linalg.vector_norm(d)) / ref_norm,
                 {k: '%.3e' % v for k, v in row_norms(d.numpy()).items()}))
    print('    null-test RMS (float32-quantised) = %.3e   for comparison' % null_rms)
    for tag, d in deltas.items():
        print('    signal-to-floor for %-9s: %.1f x' % (
            tag, float(np.sqrt((d.numpy() ** 2).mean())) / null_rms))

    # ------------------------------------------------------------------ bases
    banner('BASES  J at the true parameters and at the detuned start')
    bases = {}
    for tag, vbar in (((('theta*', vbar_true),) if LEAN else
                       (('theta*', vbar_true), ('theta_0 (detuned)', vbar_init)))):
        t0 = time.perf_counter()
        J = build_stacked_sensitivity(lambda v, z: block.transition_from_free(v, z),
                                      vbar.to(device), ref.Z_step, ROWS,
                                      chunk=cfg.obc_ref_chunk, device=device)
        b = OBCBasis.from_matrix(J.cpu(), vbar.cpu())
        bases[tag] = b
        print('\n[basis @ %s]  %.1f s' % (tag, time.perf_counter() - t0))
        for line in b.report.lines(COMBO_NAMES):
            print('    ' + line)
        del J
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ the answer
    banner('RESULT  rho* and the predicted parameter bias')
    obs_final = arm['records'][-1]['params']
    out = []
    for btag, b in bases.items():
        for dtag, d in deltas.items():
            out.append(report_bias('J @ %s, Delta* (%s)' % (btag, dtag), b, d, block,
                                   vbar_true, true_combo, scale,
                                   obs=obs_final if (dtag == 'stencil' and cfg.mode == ORIGINAL_MODE) else None))

    # The oblique prediction of R4: constraint basis at theta_0, parameter direction at theta*.
    banner('R4  oblique prediction with the basis frozen at the detuned start')
    J_true, J_0 = bases['theta*'], bases.get('theta_0 (detuned)', bases['theta*'])
    for dtag, d in deltas.items():
        # (J_0^T J)^-1 J_0^T Delta, through the stored factors; no dense J is materialised.
        M = np.empty((10, 10))
        for j in range(10):
            e = torch.zeros(10, dtype=torch.float64)
            e[j] = 1.0
            M[:, j] = J_0.jt(J_true.apply(e)).numpy()
        rhs = J_0.jt(d).numpy()
        dtheta = torch.from_numpy(np.linalg.solve(M, rhs))
        pct = percentages(block, vbar_true + dtheta, true_combo, scale)
        rms = float(np.sqrt((pct ** 2).mean()))
        print('    Delta* (%s): predicted combo-err %.3f %%   cond(J_0^T J) %.3e'
              % (dtag, rms, np.linalg.cond(M)))
        print('      ' + '  '.join('%s %+.2f' % (n, p) for n, p in zip(COMBO_NAMES, pct)))
        out.append(dict(tag='oblique, Delta* (%s)' % dtag, rho=None, pct=pct.tolist(), rms=rms,
                        dtheta=dtheta.numpy().tolist()))

    # ------------------------------------------------------------------ verdict
    # CHANGED (OA-001): the comparison with the observed obc arm (run 84033) only means something
    # on the dataset that arm trained on; on any other dataset it is skipped.
    extra = {}
    if cfg.mode == ORIGINAL_MODE:
        banner('AGAINST THE OBSERVED obc ARM')
        print('    observed: start %.2f %%, final %.2f %%, minimum %.2f %% (at it %d)'
              % (arm['combo_start'], arm['combo_final'], arm['combo_min'], arm['combo_min_it']))
        for r in out:
            verdict = ('EXPLAINS' if arm['combo_min'] <= r['rms'] <= arm['combo_final']
                       else 'RULES OUT' if r['rms'] < 2.0 else 'INCONCLUSIVE')
            print('    %-40s %7.3f %%   %s' % (r['tag'], r['rms'], verdict))
        extra['observed'] = dict(start=arm['combo_start'], final=arm['combo_final'],
                                 minimum=arm['combo_min'], minimum_it=arm['combo_min_it'],
                                 final_params=obs_final)

    payload = dict(
        mode=cfg.mode, stride=stride, n_reference=ref.n, rows=list(ROWS),
        combo_names=COMBO_NAMES, detune=COMBO_INIT_DETUNE,
        true_combo=true_combo.numpy().tolist(), combo_scale=scale.tolist(),
        null_rms=null_rms, null_rms_float64=null_rms_f64,
        floor_bias_pct=pct_null.tolist(), floor_bias_rms=floor_rms_pct,
        floor_rho=b_null.r_overlap(d_null), delta_norm={k: float(torch.linalg.vector_norm(v))
                                       for k, v in deltas.items()},
        delta_row_rms={k: row_norms(v.numpy()) for k, v in deltas.items()},
        basis={k: dict(rank=b.rank, cond=b.report.cond,
                       singular_values=b.report.singular_values)
               for k, b in bases.items()},
        predictions=out, **extra)
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = os.environ.get('OA_TAG', '%s_s%d' % (cfg.mode, stride))
    path = os.path.join(OUT_DIR, 'measure_%s.json' % tag)
    with open(path, 'w') as fh:
        json.dump(payload, fh, indent=2)
    print('[out] wrote %s' % path)
    if os.environ.get('OA_SAVE_DELTA', '0') == '1':
        np.save(os.path.join(OUT_DIR, 'delta_stencil_%s.npy' % tag), deltas['stencil'].numpy())
        print('[out] wrote delta_stencil_%s.npy' % tag)


if __name__ == '__main__':
    main()
