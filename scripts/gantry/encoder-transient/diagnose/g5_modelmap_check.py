"""G5 pre-check: does the training-time builder (model Jacobians, encoder/model_map.py) reproduce G2?

No training. (1) Planted model: map from PlantedHfn's own Jacobians (RK4 step, not ZOH) -> closed
loop V1-V4 nf = 400: must sit at the floor like G2. (2) Untrained pipeline model: map from its own
Jacobians -> physical rows vs the G2 baseline-source map, latent rows must be ~0 (unobservable), and
the closed-loop window RMS. (3) State error of (1) against the exact truth. Output: outputs/g5/modelmap_check.json
"""
__project_origin__ = "added"

import copy
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import et_paths                                                  # noqa: E402
sys.path.insert(0, os.path.join(et_paths.ET, 'encoder'))
import cl_common as cc                                           # noqa: E402
import sched_encoder as se                                       # noqa: E402
import model_map as mm                                           # noqa: E402
from g1_transient import setup                                   # noqa: E402
from g2_fix import module_phys                                   # noqa: E402
from g1_state import windows as state_windows                    # noqa: E402

NF, K, DEG = 400, 100, 6
OUT = os.path.join(et_paths.OUT, 'g5')


def main():
    t0 = time.time()
    cfg, norm, fit_sys, hfn, enc_prod, bank, rowmap, planted, sa, dims = setup()
    na, dt, hp = dims[0], cfg.ts_new, cfg.hp
    sx6 = np.asarray(norm.x_all, float).std(0)
    su = np.asarray(norm.u_all, float).std(0)
    sy = np.asarray(norm.y_all, float).std(0)
    torch.manual_seed(cfg.seed)
    par_pl = se.make_parent('planted', enc_prod, norm, dt, na, sx6, su, sy, sa, cc.DT, hp)
    enc_pl, res_pl, off_pl = mm.consistent_encoder(par_pl, planted, norm, 8, na, DEG, norm.y0, norm.ystd)
    enc_pl.eval()
    print(f'  planted: model-Jacobian map built, fit resid {res_pl["max_rel_to_W"]:.2e}, max affine offset '
          f'{off_pl:.2e}  [{time.time() - t0:.0f} s]')
    par_u = copy.deepcopy(enc_prod).to(cc.DT)
    enc_u, res_u, off_u = mm.consistent_encoder(par_u, hfn, norm, 14, na, DEG, norm.y0, norm.ystd)
    enc_u.eval()
    print(f'  untrained: model-Jacobian map built, fit resid {res_u["max_rel_to_W"]:.2e}, max affine offset '
          f'{off_u:.2e}  [{time.time() - t0:.0f} s]')
    lat = float(max(enc_u.parent.Wa_psi_y.abs().max(), enc_u.parent.Wa_psi_u.abs().max(),
                    enc_u.Cy[:, 6:].abs().max(), enc_u.Cu[:, 6:].abs().max()))
    print(f'  untrained: max |latent-row coefficient| = {lat:.2e} (unobservable latents -> 0 expected)')

    errs = {a: [] for a in ('U|G2jac', 'PL|G2jac')}
    for nm in cc.VAL:
        w = cc.windows_cl(nm, cfg, norm, na, NF)
        _, xu = module_phys(enc_u, w['yw'], w['uw'], norm, None)
        _, xp = module_phys(enc_pl, w['yw'], w['uw'], norm, sa)
        errs['U|G2jac'].append(cc.rollout_cl(hfn, xu, w, bank, rowmap[nm], norm.ystd))
        errs['PL|G2jac'].append(cc.rollout_cl(planted, xp, w, bank, rowmap[nm], norm.ystd))
    res = {a: cc.metrics(errs[a], K) for a in errs}
    res['disc_K0'] = res['U|G2jac']['rms0'] / res['PL|G2jac']['rms0']
    for a in errs:
        r = res[a]
        print(f'  {a:<10} RMS[0:400] {r["rms0"]:.4e}  RMS[100:] {r["rmsK"]:.4e}  share {r["share"]:.1%}  '
              f'grow {r["grow"]:.2f}')
    print(f'  discrimination K=0: {res["disc_K0"]:.1f}x')

    # physical rows of the untrained Jacobian map vs the G2 baseline-source (ZOH) map, V1 windows
    g2 = torch.load(os.path.join(et_paths.OUT, 'g2', 'g2_attempt1_encoders.pt'), weights_only=False)
    yv, uv, xv, _, _ = state_windows('V1_standstill_Yp10', cfg, na)
    xj, _ = module_phys(enc_u, yv, uv, norm, None)
    xz = se.numpy_sched(g2['Cb'], g2['W0b'], yv, uv, sx6, su, sy)
    xpj, _ = module_phys(enc_pl, yv, uv, norm, sa)
    e_zoh = np.sqrt(((xz - xv[:, :6]) ** 2).mean(0))
    res['untrained_jac_vs_zoh_rel'] = (np.sqrt(((xj - xz) ** 2).mean(0)) / e_zoh).tolist()
    res['planted_jac_state_rms_V1'] = np.sqrt(((xpj - xv) ** 2).mean(0)).tolist()
    res['max_latent_coeff_untrained'] = lat
    res['offsets'] = dict(planted=off_pl, untrained=off_u)
    print('  untrained Jacobian map vs G2 baseline ZOH map (V1), RMS diff / ZOH error: '
          + ' '.join(f'{v:.2e}' for v in res['untrained_jac_vs_zoh_rel']))
    print('  planted Jacobian map, V1 state RMS vs exact: '
          + ' '.join(f'{v:.2e}' for v in res['planted_jac_state_rms_V1']))
    os.makedirs(OUT, exist_ok=True)
    torch.save(dict(enc_pl=enc_pl.state_dict(), enc_u=enc_u.state_dict()), os.path.join(OUT, 'modelmap_encoders.pt'))
    with open(os.path.join(OUT, 'modelmap_check.json'), 'w') as f:
        json.dump(res, f, indent=1, default=float)
    print(f'  saved outputs/g5/modelmap_check.json  [{time.time() - t0:.0f} s]')


if __name__ == '__main__':
    main()
