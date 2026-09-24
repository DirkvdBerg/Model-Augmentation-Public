"""G5 (a): the simulation training arms, production pipeline with the encoder swapped (ET-008).

    ARM=burnin    production encoder, burn_in = 100 (the current burn-in arm)
    ARM=g2static  G2 baseline-source Y-scheduled map (deg 2, outputs/g2), W^a = 0, burn_in = 0
    ARM=g2refresh model-Jacobian map (encoder/model_map.py, deg 6, all rows incl. W^a), rebuilt from
                  the CURRENT model every REFRESH_EVERY updates (default: one epoch), burn_in = 0

Everything else is the vendored production entry (`gantry_interconnect_dynamic.main`): the
frictionless dataset (ET-001), closed loop, nf, nx_ann, seed, optimizer, validation, checkpoints.
The encoder swap happens after build_model, and the optimizer is rebuilt over the new parameter set.

Modes:
    DRY=1     build, swap, one closed-loop forward on V1 windows, 0 updates (smoke for any arm)
    SMOKE=1   CPU, eager, n_its = 1, tiny batch and nf (1 optimizer update)
Server: runners/sim_arm.sh. NOT submitted by this session.
"""
__project_origin__ = "added"

import copy
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
import et_paths                                                  # noqa: E402
sys.path.insert(0, os.path.join(et_paths.ET, 'encoder'))
sys.path.insert(0, os.path.join(et_paths.ET, 'diagnose'))
import core                                                      # noqa: E402
import model_map as mm                                           # noqa: E402

ARM = os.environ.get('ARM', 'g2refresh')
DRY = os.environ.get('DRY') == '1'
SMOKE = os.environ.get('SMOKE') == '1'
G2_FILE = os.path.join(et_paths.OUT, 'g2', 'g2_attempt1_encoders.pt')
DEG_REFRESH = 6                    # ET-008: the planted-source degree of the G2 rule


def make_cfg():
    import gantry_interconnect_dynamic as entry
    cfg = replace(entry.CFG, mode=et_paths.MODE, burn_in=(100 if ARM == 'burnin' else 0))
    if SMOKE or DRY:
        cfg = replace(cfg, device='cpu', compile_mode=None, checkpoint_chunk=0, batch_size=8,
                      n_its=1, epochs=1, stride=400, save_flag=False,
                      nf_override=max(int(os.environ.get('SMOKE_NF', '20')), cfg.burn_in + 20))
    return cfg


def swap_encoder(fit_sys, cfg, norm, hp, dims):
    """Replace fit_sys.encoder per ARM and rebuild the optimizer over the new parameter set."""
    na = dims[0]
    if ARM == 'burnin':
        return None
    parent = copy.deepcopy(fit_sys.encoder)
    if ARM == 'g2static':
        g2 = torch.load(G2_FILE, weights_only=False)
        enc = core.ScheduledReconEncoder(parent, g2['Cb'], na, 3, 3, norm.y0, norm.ystd).to(cfg.dtype_pt)
        with torch.no_grad():                                    # W^a = 0: consistent with the zero ANN
            enc.parent.Wa_psi_y.zero_(); enc.parent.Wa_psi_u.zero_()
        info = dict(deg=int(g2['Cb'].shape[0]))
    else:
        nx = cfg.nx_phys + cfg.nx_ann
        enc, res, off = mm.consistent_encoder(parent, fit_sys.hfn, norm, nx, na, DEG_REFRESH,
                                              norm.y0, norm.ystd, dtype=cfg.dtype_pt)
        info = dict(deg=DEG_REFRESH, fit_resid=res['max_rel_to_W'], offset=off)
    if os.environ.get('ENC_TRAIN_LINEAR') != '1':                  # ET-013
        info['frozen'], info['trainable'] = mm.freeze_linear(enc)
    fit_sys.encoder = enc
    groups = [dict(item) for _, item in fit_sys.parameters_with_names.items()]
    fit_sys.optimizer = fit_sys.init_optimizer(groups, lr=hp['lr'], eps=cfg.adam_eps)
    print(f'[sim_arm] encoder swapped: {ARM} {info}; optimizer rebuilt over '
          f'{sum(p.numel() for p in fit_sys.encoder.parameters())} encoder parameters', flush=True)
    return info


def install_refresh(fit_sys, cfg, norm, dims, every):
    """Replace fit_sys.loss by the picklable model_map.RefreshingLoss (a closure breaks deepSI's
    checkpoint pickling, run R019)."""
    rl = mm.RefreshingLoss(fit_sys, norm, cfg.nx_phys + cfg.nx_ann, dims[0], every)
    fit_sys.loss = rl
    return rl


def redirect_deepsi_workdir(sdir):
    """deepSI writes per-validation checkpoints to %LOCALAPPDATA%/deepSI; keep them in this run's
    folder (same fix as telica-real's telica_augment.redirect_deepsi_workdir)."""
    import deepSI.fit_systems.fit_system as _fsm
    base = os.path.join(sdir, 'deepsi_work')
    dirs = dict(base=base, data_sets=os.path.join(base, 'data_sets'),
                checkpoints=os.path.join(base, 'checkpoints'))
    for dd in dirs.values():
        os.makedirs(dd, exist_ok=True)
    _fsm.get_work_dirs = lambda: dirs


def dry_forward(fit_sys, cfg, norm, dims):
    """One closed-loop forward over V1 (production rollout), no gradient: window RMS and share."""
    import cl_common as cc
    from gantry_dynamic.controller import build_controller_bank
    bank, rows, _ = build_controller_bank(['V1_standstill_Yp10'], cfg.ts_new, ystd=norm.ystd,
                                          std_u=norm.std_u, dtype=cc.DT)
    hfn = copy.deepcopy(fit_sys.hfn).to(cc.DT)
    enc = copy.deepcopy(fit_sys.encoder).to(cc.DT).eval()
    w = cc.windows_cl('V1_standstill_Yp10', cfg, norm, dims[0], 400)
    un = (w['uw'] - np.asarray(norm.u_mean).ravel()) / np.asarray(norm.std_u).ravel()
    yn = (w['yw'] - np.asarray(norm.y0).ravel()) / np.asarray(norm.ystd).ravel()
    with torch.no_grad():
        x0 = enc(torch.as_tensor(np.ascontiguousarray(un)), torch.as_tensor(np.ascontiguousarray(yn))).numpy()
    m = cc.metrics([cc.rollout_cl(hfn, x0, w, bank, rows[0], norm.ystd)], 100)
    print(f'[dry] V1 closed loop nf=400: RMS {m["rms0"]:.4e}  share {m["share"]:.1%}  grow {m["grow"]:.2f}')
    return m


def main():
    t0 = time.time()
    et_paths.check()
    from gantry_dynamic.config import save_dir
    from gantry_dynamic.data import load_datasets, compute_normalization, VAL_FILES
    from gantry_dynamic.model import build_model, get_encoder_dims
    from gantry_dynamic.training import train_model_with_diagnostics
    cfg = make_cfg()
    hp = cfg.hp
    run_id = os.environ.get('SLURM_JOB_ID') or time.strftime('%Y%m%d_%H%M%S')
    sdir = os.path.join(et_paths.OUT, 'g5', f'sim_{ARM}_{"dry" if DRY else "smoke" if SMOKE else run_id}')
    os.makedirs(sdir, exist_ok=True)
    redirect_deepsi_workdir(sdir)
    print(f'[sim_arm] ARM={ARM} DRY={DRY} SMOKE={SMOKE} mode={cfg.mode} burn_in={cfg.burn_in} '
          f'nf={cfg.nf} batch={cfg.batch_size} n_its={cfg.n_its} device={cfg.device}', flush=True)
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    data = load_datasets(cfg)
    norm = compute_normalization(cfg, data)
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    fit_sys = build_model(hp, cfg, data, norm)
    if cfg.closed_loop:
        from gantry_dynamic.controller import build_closed_loop
        from gantry_dynamic.data import TRAIN_FILES
        fit_sys.simulator = build_closed_loop(fit_sys, norm, cfg, train_files=TRAIN_FILES,
                                              val_files=VAL_FILES, val_data=data.val_ckpt_data)
    dims = get_encoder_dims(hp, cfg)
    info = swap_encoder(fit_sys, cfg, norm, hp, dims)
    out = dict(arm=ARM, dry=DRY, smoke=SMOKE, cfg={k: str(v) for k, v in cfg.__dict__.items()}, info=info)
    if DRY:
        out['dry'] = dry_forward(fit_sys, cfg, norm, dims)
        out['n_updates'] = 0
    else:
        state = None
        if ARM == 'g2refresh':
            every = int(os.environ.get('REFRESH_EVERY', '0')) or None
            if every is None:
                n_win = sum(len(sd.u) for sd in data.train_list) // max(1, cfg.stride)
                every = max(1, n_win // hp['batch_size'])            # ~ one epoch of updates
            state = install_refresh(fit_sys, cfg, norm, dims, 1 if SMOKE else every)
        bestfit, _ = train_model_with_diagnostics(fit_sys, hp, cfg, data, norm, resume_ckpt=None,
                                                  checkpoint_dir=(sdir if cfg.save_flag else None),
                                                  run_id=run_id)
        out['bestfit_val_sim_rms'] = float(bestfit)
        out['n_updates'] = int(getattr(fit_sys, 'batch_counter', -1))
        if SMOKE and state is not None:                          # the post-update refresh, no update
            state.refresh(int(getattr(fit_sys, 'batch_counter', 0)))
        out['refresh_log'] = state.log if state is not None else None
    out['wall_s'] = time.time() - t0
    with open(os.path.join(sdir, 'sim_arm.json'), 'w') as f:
        json.dump(out, f, indent=1, default=str)
    print(f'[sim_arm] done: n_updates {out["n_updates"]}, {out["wall_s"]:.0f} s -> {sdir}', flush=True)


if __name__ == '__main__':
    main()
