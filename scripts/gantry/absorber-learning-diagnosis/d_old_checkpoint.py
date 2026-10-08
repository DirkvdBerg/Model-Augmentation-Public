"""Phase D: the old checkpoint 84032 through the CURRENT pipeline (pipeline-change row) and its
captured fraction: model vs the same weights with the learned output zeroed, per val record, Y in
the absorber band. Evaluation only, no optimizer step.

Usage: python d_old_checkpoint.py [pair|best]   (pair = the polished final, gantry_ckpt_84032)
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', '..')))

from gantry_dynamic.config import RunConfig                                   # noqa: E402
from gantry_dynamic.data import load_datasets, compute_normalization          # noqa: E402
from gantry_dynamic.model import build_model                                   # noqa: E402
from gantry_dynamic.controller import build_closed_loop, build_controller_bank  # noqa: E402
from gantry_dynamic.training import load_checkpoint                            # noqa: E402
from gantry_dynamic.closed_loop_report import _without_learned, _true_free     # noqa: E402
from model_augmentation.fit_systems.closed_loop import closed_loop_free_run_rms_batch  # noqa: E402
from b_error_budget import band_rms, BANDS, ABS_BAND                           # noqa: E402

UD = os.path.abspath(os.path.join(HERE, '..', 'meeting', 'meeting-18-09-2026', 'server', 'useable'))
CKPT = {'pair': os.path.join(UD, '84032', 'gantry_ckpt_84032'),
        'pre': os.path.join(UD, '84032', 'gantry_ckpt_84032_pre_lbfgs'),
        'best': os.path.join(UD, 'SSE_Interconnect_Composed_84032_best.pth')}

CFG = RunConfig(   # 84032/config.json, eager on the CPU
    mode='augmentation_ma50_b140-230_a6_z03', encoder_init='linear_map', ann_activation='tanh',
    joint_estimation=True, physics_parameterization='reduced', param_init_detune=None,
    combo_init_detune=[1.1, 1.1, 0.9, 1.1, 0.9, 0.9, 1.1, 1.1, 0.9, 1.1], param_prior=False,
    orth=False, obc=False, closed_loop=True, seed=42, stride=10, nx_ann=8,
    ann_route_ix=tuple(range(14)), n_nodes_per_layer=24, n_hidden_layers=3, up_sample=1,
    batch_size=512, lr=1e-5, adam_eps=1e-16, nf_seconds=0.1, use_f64=False, device='cpu',
    compile_mode=None, lbfgs=False, save_flag=False)


def load_v1(fs, base):
    """Load a pre-D-229 checkpoint: same combinations, other m_diff coordinate. The v1 map (commit
    e5dd057, blocks.py) is m_diff = init (1 + free), the nine others init exp(free); the combinations
    are formed with it and re-expressed in the current coordinate, so the model is physically the
    same system. Every other tensor loads as is."""
    from gantry_dynamic.training import resolve_checkpoint
    hfn_sd, enc_sd, _, _, _ = resolve_checkpoint(base)
    phy = next(b for b in fs.hfn.connected_blocks if hasattr(b, 'free_params'))
    key = next(k for k in hfn_sd if k.endswith('free_params'))
    init_key = key.replace('free_params', 'combo_init')
    free1, init1 = hfn_sd[key].double(), hfn_sd[init_key].double()
    i = phy.M_DIFF_IX
    c = init1 * torch.exp(free1)
    c[i] = init1[i] * (1.0 + free1[i])
    sd = fs.hfn.state_dict()
    missing = [k for k in hfn_sd if k not in sd]
    extra = [k for k in sd if k not in hfn_sd]
    print('  [v1 load] keys only in ckpt %s | only in model %s' % (missing, extra))
    assert torch.allclose(init1.to(phy.combo_init.dtype), phy.combo_init), 'combo_init differs'
    for k, v in hfn_sd.items():
        if k in sd and k != key:
            sd[k] = v
    sd[key] = phy.free_from_combinations(c.to(phy.combo_init.dtype)).to(sd[key].dtype)
    fs.hfn.load_state_dict(sd)
    fs.encoder.load_state_dict(enc_sd)
    got = phy.combinations_from_free(phy.free_params).double()
    print('  [v1 load] combinations %s | max rel diff %.2e' % (
        ' '.join('%.4g' % v for v in c.tolist()), float(((got - c) / c).abs().max())))


def main(which):
    cfg = CFG
    # the old run's record lists (its log: 14 train, 4 val); the legacy constants gained the D-206
    # Telica records afterwards, which this folder does not hold
    from gantry_dynamic import data as gd
    gd.TRAIN_FILES, gd.VAL_FILES, gd.TEST_FILES = gd.TRAIN_FILES[:14], gd.VAL_FILES[:4], gd.TEST_FILES[:4]
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    data = load_datasets(cfg)
    norm = compute_normalization(cfg, data)
    hp = cfg.hp
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    fs = build_model(hp, cfg, data, norm)
    fs.simulator = build_closed_loop(fs, norm, cfg, train_files=data.train_files,
                                     val_files=data.val_files, val_data=data.val_ckpt_data, verbose=False)
    load_v1(fs, CKPT[which])
    files = list(data.val_files[:4]); sdl = list(data.val_list[:4])
    bank, rows, _ = build_controller_bank(files, cfg.ts_new, ystd=norm.ystd, std_u=norm.std_u, dtype=cfg.dtype_pt)
    phy = next(b for b in fs.hfn.connected_blocks if hasattr(b, 'free_params'))
    systems = {'model': fs, 'silenced': _without_learned(fs),
               'baseline_true': _without_learned(fs, _true_free(phy)),
               'baseline_start': _without_learned(fs, torch.zeros_like(phy.free_params))}
    k0 = max(fs.na, fs.nb)
    out = {}
    for name, s in systems.items():
        with torch.no_grad():
            pc, agg, preds = closed_loop_free_run_rms_batch(s, sdl, bank, rows, return_pred=True)
        e = [np.asarray(sd.y)[k0:] - np.asarray(p) for sd, p in zip(sdl, preds)]
        out[name] = dict(pc=np.asarray(pc), agg=np.asarray(agg), band=np.stack([band_rms(x, cfg.fs_new_hz) for x in e]))
        print('%-15s val sim-RMS %.4e m | Y per record %s | Y 140-230 Hz %s' % (
            name, np.mean(agg), '  '.join('%.2e' % v for v in np.asarray(pc)[:, 2]),
            '  '.join('%.2e' % v for v in out[name]['band'][:, ABS_BAND, 2])), flush=True)
    np.savez(os.path.join(HERE, 'outputs', 'd_old_%s.npz' % which),
             **{'%s_%s' % (n, k): v for n, d in out.items() for k, v in d.items()}, files=np.array(files))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'pair')
