"""ET-008 (c) / ET-012: swap the G4 encoder into the telica-real pipeline. Imported by the vendored
telica_augment.py when ET_G2_ENCODER=<path> is set (with ET_NA_NB matching the saved window).

The saved encoder is FrictionCompEncoder(ScheduledReconEncoder(parent, C)) from g4_telica_a3.py
(attempt 3, the one that passed G4). The same structure is built around the pipeline's own encoder
and the saved state is loaded into it: W^b (recovered-block map), the Y schedule C_k and the
friction constants come from G4; W^a and the correction net are overwritten by the G4 build's seeded
draws (same class and shapes, kaiming / zero-final-layer). The optimizer is rebuilt over the new
parameter set.
"""
__project_origin__ = "added"

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', 'encoder'))
sys.path.append(HERE)
import core                                                      # noqa: E402


def swap(fit_sys, cfg, norm, hp, path):
    from g4_telica_a2 import FrictionCompEncoder
    g = torch.load(path, weights_only=False)
    na = int(hp.get('na_nb'))
    if 'na' in g and int(g['na']) != na:
        raise RuntimeError(f'saved encoder window na={g["na"]} but the pipeline runs na={na}; '
                           f'set ET_NA_NB={g["na"]}')
    st = g['rfs_state']
    C = np.zeros(tuple(st['inner.Cy'].shape[:2]) + ((na + 1) * 6,))
    inner = core.ScheduledReconEncoder(fit_sys.encoder, C, na, 3, 3, norm.y0, norm.ystd)
    from params import telica_params as tp
    enc = FrictionCompEncoder(inner, st['cc'].numpy(), float(tp.V0_TANH), cfg.ts_new, norm)
    enc.load_state_dict(st, strict=True)
    enc = enc.to(cfg.dtype_pt)
    frz = (0, 0)
    if os.environ.get('ENC_TRAIN_LINEAR') != '1':                  # ET-013
        import model_map
        frz = model_map.freeze_linear(enc)
    fit_sys.encoder = enc
    groups = [dict(item) for _, item in fit_sys.parameters_with_names.items()]
    fit_sys.optimizer = fit_sys.init_optimizer(groups, lr=hp['lr'], eps=cfg.adam_eps)
    print(f'[et_hook] G4 encoder swapped in: na {na}, schedule deg {int(C.shape[0])}, friction cc '
          f'{np.round(st["cc"].numpy(), 1).tolist()} N, v0 {enc.v0}; linear map frozen {frz[0]} params, trainable {frz[1]}; '
          f'optimizer rebuilt', flush=True)
    return fit_sys
