"""G5 (b): R2 against a TRAINED checkpoint, the arm D-197 said only the server can run (ET-008).

For the trained model (frictionless dataset, production config), three initial states:
    exact      the exact physical state (truth.py), latent rows from the production encoder
    P0         the checkpoint's own trained encoder
    G2model    the model-Jacobian map of the TRAINED model (encoder/model_map.py, deg 6, all rows)
and two measurements: open-loop free run (V1-V4, 20 starts, H 400 / 4000, as D-197 R2) and the
closed-loop nf = 400 window metrics (RMS, transient share, grow) that the training loss sees.

    CKPT=<base or _best.pth> python -u runners/r2_ckpt.py     # server
    R2_SMOKE=1 python -u runners/r2_ckpt.py                   # no checkpoint: untrained model, dry
Note: vendor/transient/r2_freerun.py unpacks 4 values from resolve_checkpoint, which now returns 5;
this runner does not reuse that path.
"""
__project_origin__ = "added"

import copy
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
import et_paths                                                  # noqa: E402
sys.path.insert(0, os.path.join(et_paths.ET, 'encoder'))
sys.path.insert(0, os.path.join(et_paths.ET, 'diagnose'))
import cl_common as cc                                           # noqa: E402
import model_map as mm                                           # noqa: E402
import truth                                                     # noqa: E402
import recon                                                     # noqa: E402
from g1_transient import setup, prod_encoder                     # noqa: E402
from g3_r2 import rollout_ol, enc_out                            # noqa: E402

SMOKE = os.environ.get('R2_SMOKE') == '1'
HORIZONS = [400] if SMOKE else [400, 4000]
N_STARTS = 4 if SMOKE else 20
RECORDS = cc.VAL[:1] if SMOKE else cc.VAL


def ckpt_overrides(path):
    """Model-structure fields from the checkpoint's own config.json (CKPT_CONFIG), so the model is
    built as it was trained. Refuses runs whose rollout needs more than the model step."""
    if not path:
        return None
    c = json.load(open(path))
    if c.get('MODE') != et_paths.MODE:
        raise SystemExit(f'checkpoint dataset {c.get("MODE")!r} is not {et_paths.MODE!r}')
    if c.get('OBC') or c.get('ORTH'):
        raise SystemExit('OBC / orth runs need their projection inside the rollout; not supported here')
    keys = {'JOINT_ESTIMATION': 'joint_estimation', 'PHYSICS_PARAMETERIZATION': 'physics_parameterization',
            'COMBO_INIT_DETUNE': 'combo_init_detune', 'PARAM_INIT_DETUNE': 'param_init_detune',
            'PARAM_PRIOR': 'param_prior', 'ANN_ACTIVATION': 'ann_activation', 'NX_ANN': 'nx_ann',
            'N_NODES_PER_LAYER': 'n_nodes_per_layer', 'N_HIDDEN_LAYERS': 'n_hidden_layers',
            'UP_SAMPLE': 'up_sample', 'ENCODER_INIT': 'encoder_init'}
    ov = {v: c[k] for k, v in keys.items() if k in c}
    if 'ANN_ROUTE_IX' in c:
        ov['ann_route_ix'] = tuple(c['ANN_ROUTE_IX'])
    if 'NA_NB' in c:
        ov['na_nb_override'] = int(c['NA_NB'])
    print(f'[r2_ckpt] structure from {path}: {ov}')
    return ov


def main():
    t0 = time.time()
    cfg, norm, fit_sys, hfn, enc_prod, bank, rowmap, planted, sa, dims = setup(
        ckpt_overrides(os.environ.get('CKPT_CONFIG')))
    na = dims[0]
    ck = os.environ.get('CKPT')
    if ck:
        from gantry_dynamic.training import resolve_checkpoint
        hfn_sd, enc_sd = resolve_checkpoint(ck)[:2]
        hfn.load_state_dict(hfn_sd if isinstance(hfn_sd, dict) else hfn_sd.state_dict())
        enc_prod.load_state_dict(enc_sd if isinstance(enc_sd, dict) else enc_sd.state_dict())
        hfn.to(cc.DT); enc_prod.to(cc.DT)
        print(f'[r2_ckpt] loaded {ck}')
    elif not SMOKE:
        raise SystemExit('set CKPT=<checkpoint base> (or R2_SMOKE=1 for the dry path)')
    nx = cfg.nx_phys + cfg.nx_ann
    enc_g2, resid, off = mm.consistent_encoder(copy.deepcopy(enc_prod), hfn, norm, nx, na, 6,
                                               norm.y0, norm.ystd)
    enc_g2.eval()
    print(f'[r2_ckpt] model-Jacobian map of the {"trained" if ck else "untrained"} model: '
          f'resid {resid["max_rel_to_W"]:.1e}, max affine offset {off:.1e}')
    from gantry_dynamic.data import load_traj
    ystd = np.asarray(norm.ystd).ravel()
    res = dict(ckpt=ck, smoke=SMOKE, resid=resid, offset=off, open_loop={}, closed_loop={})
    sq = {}
    cl = {a: [] for a in ('exact', 'P0', 'G2model')}
    for nm in RECORDS:
        sd = load_traj(nm + '.mat', cfg)
        u = np.ascontiguousarray(sd.u, dtype=float); y = np.ascontiguousarray(sd.y, dtype=float)
        x8 = truth.exact_truth(nm, cfg.mode)['x8'][:, recon.T2M]
        for H in HORIZONS:
            ks = np.linspace(na + 1, len(y) - H, N_STARTS).astype(int)
            yw = np.stack([y[k - na:k + 1] for k in ks]); uw = np.stack([u[k - na:k + 1] for k in ks])
            un = torch.as_tensor(np.ascontiguousarray(np.stack([
                (u[k:k + H] - np.asarray(norm.u_mean).ravel()) / np.asarray(norm.std_u).ravel()
                for k in ks])), dtype=cc.DT)
            yref = np.stack([y[k:k + H] for k in ks])
            xe = prod_encoder(enc_prod, dict(yw=yw, uw=uw), norm)
            xg = enc_out(enc_g2, yw, uw, norm)
            xx = xe.copy(); xx[:, :6] = cc.x0_norm(x8[ks, :6], norm)
            for a, x0 in (('exact', xx), ('P0', xe), ('G2model', xg)):
                yh = rollout_ol(hfn, x0, un) * ystd + np.asarray(norm.y0).ravel()
                sq.setdefault((a, H), [np.zeros(3), 0])
                sq[(a, H)][0] += ((yh - yref) ** 2).sum(axis=(0, 1)); sq[(a, H)][1] += len(ks) * H
        w = cc.windows_cl(nm, cfg, norm, na, 400)
        xe = prod_encoder(enc_prod, w, norm)
        xg = enc_out(enc_g2, w['yw'], w['uw'], norm)
        xx = xe.copy(); xx[:, :6] = cc.x0_norm(w['x8'][:, :6], norm)
        for a, x0 in (('exact', xx), ('P0', xe), ('G2model', xg)):
            cl[a].append(cc.rollout_cl(hfn, x0, w, bank, rowmap[nm], norm.ystd))
        print(f'[r2_ckpt] {nm} done [{time.time() - t0:.0f} s]')
    for (a, H), (s, n) in sq.items():
        res['open_loop'][f'{a}|{H}'] = np.sqrt(s / n).tolist()
    for a in cl:
        res['closed_loop'][a] = cc.metrics(cl[a], 100)
    for H in HORIZONS:
        print(f'  open loop H={H}: ' + '  '.join(
            f'{a} Y {res["open_loop"][f"{a}|{H}"][2]:.3e}' for a in ('exact', 'P0', 'G2model')))
    for a in cl:
        r = res['closed_loop'][a]
        print(f'  closed loop nf=400 {a:<8} RMS {r["rms0"]:.4e}  share {r["share"]:.1%}  grow {r["grow"]:.2f}')
    sdir = os.path.join(et_paths.OUT, 'g5', 'r2_ckpt_smoke' if SMOKE else
                        f'r2_ckpt_{os.environ.get("R2_TAG") or os.environ.get("SLURM_JOB_ID", "run")}')
    os.makedirs(sdir, exist_ok=True)
    with open(os.path.join(sdir, 'r2_ckpt.json'), 'w') as f:
        json.dump(res, f, indent=1, default=float)
    print(f'[r2_ckpt] done, 0 optimizer updates, {time.time() - t0:.0f} s -> {sdir}')


if __name__ == '__main__':
    main()
