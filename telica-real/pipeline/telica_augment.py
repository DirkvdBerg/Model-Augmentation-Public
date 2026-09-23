"""Augmentation training on REAL Telica data (G6): the vendored pipeline, adapted.

    python telica-real/pipeline/telica_augment.py                 full run (server)
    TELICA_SMOKE=1 python telica-real/pipeline/telica_augment.py  local smoke test (<= 2 updates)
    python telica-real/pipeline/telica_augment.py --build-norm    write pipeline/norm_frozen.json

What differs from scripts/gantry/gantry_interconnect_dynamic.py, item by item (handoff sec. 4):
  1 controller   the IDENTIFIED Telica controller (G3), controller/telica_bank.py, one row
  2 parameters   params/ (G1) + the G4 attempt-2 baseline (recovered_params_a2.json), tanh friction
  3 normalise    frozen from the train split with low-passed velocities (pipeline/norm_frozen.json)
  4 decimation   20 kHz (TR-008) or 5 kHz (TR-022, anti-aliased), from ann_settings fs_train
  5 GT-only      no load_mat_aug / true-x0 baselines / oracle / aug-state plots; output-only eval
  6 files        split by operating point from data_index; loader records, not .mat simulations
  7 ANN          nx_ann, routing, horizon from learnability/ann_settings.json (G5), not from the
                 simulated 212 Hz absorber
float64 throughout (TR-013). Checkpoints, logs and metrics go to telica-real/outputs/<run>/.
"""
import json
import os
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tr_env                                                          # noqa: E402

import numpy as np                                                     # noqa: E402
import torch                                                           # noqa: E402

from gantry_dynamic.config import RunConfig                            # noqa: E402
from pipeline.real_data import (load_datasets_real, frozen_norm,       # noqa: E402
                                build_frozen_norm, NORM_FILE)
from pipeline.telica_model import build_model_real                     # noqa: E402
from pipeline.memory import rss_mb, peak_rss_mb                        # noqa: E402
from controller.telica_bank import build_closed_loop_telica            # noqa: E402

TR = tr_env.TR
ANN_FILE = os.path.join(TR, 'learnability', 'ann_settings.json')
SMOKE = os.environ.get('TELICA_SMOKE') == '1'


def ann_settings():
    if not os.path.isfile(ANN_FILE):
        raise FileNotFoundError(f'{ANN_FILE} missing: G5 derives the ANN settings from the residual')
    return json.load(open(ANN_FILE))


def _env_overrides(A):
    """Server-arm overrides (TR-020): NX_ANN, NF_SECONDS, BATCH, CHUNK, EPOCHS, LR, STRIDE.
    Anything set here is printed and stored in the run's augment_metrics.json."""
    m = {'NX_ANN': ('nx_ann', int), 'NF_SECONDS': ('nf_seconds', float), 'BATCH': ('batch_size', int),
         'CHUNK': ('checkpoint_chunk', int), 'EPOCHS': ('epochs', int), 'LR': ('lr', float),
         'STRIDE': ('stride', int), 'ITS_PER_VAL': ('its_per_val', int),
         'FS_TRAIN': ('fs_train', int)}
    used = {}
    for env, (key, typ) in m.items():
        if os.environ.get(env):
            A[key] = typ(os.environ[env]); used[key] = A[key]
    if 'nx_ann' in used:                       # augmented rows follow nx_ann
        A['ann_route_ix'] = [r for r in A['ann_route_ix'] if r < 6] + list(range(6, 6 + A['nx_ann']))
    if used:
        print(f'[augment] ENV OVERRIDES {used}')
    return A


def make_cfg():
    A = _env_overrides(ann_settings())
    cfg = RunConfig(
        mode='telica_real', encoder_init='linear_map', ann_activation='tanh',
        joint_estimation=False, orth=False, obc=False, param_prior=False, snr=None, seed=42,
        closed_loop=True, fs_orig=20000,
        fs_new=(None if int(A.get('fs_train', 20000)) == 20000 else int(A['fs_train'])),
        stride=int(A.get('stride', 20)), burn_in=int(A.get('burn_in', 0)),
        use_f64=True,                                  # TR-013
        device=os.environ.get('DEVICE', 'cuda'),
        compile_mode=(None if os.environ.get('COMPILE_MODE', 'reduce-overhead') == 'none'
                      else os.environ.get('COMPILE_MODE', 'reduce-overhead')),
        checkpoint_chunk=int(A.get('checkpoint_chunk', 200)),
        nx_ann=int(A['nx_ann']), ann_route_ix=tuple(A['ann_route_ix']),
        n_nodes_per_layer=int(A.get('n_nodes_per_layer', 24)),
        n_hidden_layers=int(A.get('n_hidden_layers', 3)), up_sample=1,
        batch_size=int(A.get('batch_size', 256)), lr=float(A.get('lr', 1e-5)), adam_eps=1e-16,
        epochs=int(A.get('epochs', 100)), its_per_val=A.get('its_per_val', None), n_its=None,
        nf_seconds=float(A['nf_seconds']), lbfgs=False, start_phase='adam',
        param_init_detune=None,
    )
    if SMOKE:
        # <= 2 optimizer updates, batch 8, one short window, CPU, eager (handoff G6)
        cfg = replace(cfg, device='cpu', compile_mode=None, checkpoint_chunk=0, batch_size=8,
                      n_its=2, epochs=1, nf_seconds=float(os.environ.get('SMOKE_NF_S', '0.005')),
                      stride=int(os.environ.get('SMOKE_STRIDE', '400')), its_per_val=None,
                      burn_in=0)
    return cfg


def _ann(fit_sys):
    from model_augmentation.fit_systems.blocks import Static_ANN_Block
    return next(m for m in fit_sys.hfn.connected_blocks if isinstance(m, Static_ANN_Block))


def ann_snapshot(fit_sys):
    return {k: v.detach().clone() for k, v in _ann(fit_sys).state_dict().items()}


def ann_changed(fit_sys, snap):
    now = _ann(fit_sys).state_dict()
    return float(max((now[k].double() - snap[k].double()).abs().max() for k in snap))


def output_only_eval(fit_sys, sdl, names, bank, label, out):
    """Closed-loop free run per record, scored on the OUTPUT only (no latent-state judgement)."""
    from model_augmentation.fit_systems.closed_loop import closed_loop_free_run_rms_batch
    from model_augmentation.fit_systems.blocks import Static_ANN_Block
    per, agg = closed_loop_free_run_rms_batch(fit_sys, sdl, bank, [0] * len(sdl))
    ann = next(m for m in fit_sys.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
    saved = {k: v.clone() for k, v in ann.state_dict().items()}
    with torch.no_grad():
        for p in ann.parameters():
            p.zero_()
    per0, agg0 = closed_loop_free_run_rms_batch(fit_sys, sdl, bank, [0] * len(sdl))
    ann.load_state_dict(saved)
    rows = [dict(record=n, rms_m=np.asarray(a).tolist(), rms_baseline_m=np.asarray(b).tolist(),
                 ratio=(np.asarray(a) / np.asarray(b)).tolist()) for n, a, b in zip(names, per, per0)]
    med = np.median([r['ratio'] for r in rows], axis=0)
    print(f'[eval] {label}: median per-axis RMS ratio augmented / baseline-alone '
          f'{np.round(med, 4).tolist()} over {len(rows)} records', flush=True)
    out[label] = dict(records=rows, median_ratio=med.tolist())


def redirect_deepsi_workdir(sdir):
    """deepSI writes its per-validation checkpoints to %LOCALAPPDATA%/deepSI (~/.deepSI on Linux).
    Point it at this run's own folder instead, so nothing is written outside telica-real/outputs
    and the server run keeps its checkpoints next to its logs. fit_system binds get_work_dirs by
    name at import, so that module attribute is the one to replace."""
    import deepSI.fit_systems.fit_system as _fsm
    base = os.path.join(sdir, 'deepsi_work')
    dirs = dict(base=base, data_sets=os.path.join(base, 'data_sets'),
                checkpoints=os.path.join(base, 'checkpoints'))
    for dd in dirs.values():
        os.makedirs(dd, exist_ok=True)
    _fsm.get_work_dirs = lambda: dirs


def main():
    if '--build-norm' in sys.argv:
        build_frozen_norm()
        return
    cfg = make_cfg()
    run = os.environ.get('SLURM_JOB_ID') or time.strftime('%Y%m%d_%H%M%S')
    sdir = tr_env.out_dir(('g6_smoke_' if SMOKE else 'augment_') + run)
    redirect_deepsi_workdir(sdir)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    t0 = time.time()
    print(f'[augment] smoke={SMOKE} nf={cfg.nf} ({cfg.nf_seconds * 1e3:.1f} ms) batch={cfg.batch_size} '
          f'nx_ann={cfg.nx_ann} route={cfg.ann_route_ix} f64={cfg.use_f64} device={cfg.device}')
    kw = dict(max_records=2, max_len=int(os.environ.get('SMOKE_LEN', '6000'))) if SMOKE else {}
    data = load_datasets_real(cfg, **kw)
    if not os.path.isfile(NORM_FILE):
        raise FileNotFoundError('run --build-norm first (frozen normalisation, TR-016)')
    norm = frozen_norm(cfg)
    hp = cfg.hp
    print(f'[augment] data + norm loaded ({time.time() - t0:.0f} s), rss {rss_mb():.0f} MB')
    fit_sys = build_model_real(hp, cfg, data, norm, baseline_tag='_a2')
    fit_sys.simulator = build_closed_loop_telica(fit_sys, norm, cfg, train_files=data.train_names,
                                                 val_files=data.val_names,
                                                 val_data=data.val_ckpt_data)
    print(f'[augment] model built, rss {rss_mb():.0f} MB')
    out = dict(cfg={k: str(v) for k, v in cfg.__dict__.items()}, run=run)
    if SMOKE and os.environ.get('MEM_PROBE') == '1':
        from pipeline.memory import mem_probe
        out['mem_probe'] = mem_probe(fit_sys, cfg, data)
    from gantry_dynamic.training import train_model_with_diagnostics
    ann0 = ann_snapshot(fit_sys)
    bestfit, _ = train_model_with_diagnostics(fit_sys, hp, cfg, data, norm, resume_ckpt=None,
                                              checkpoint_dir=sdir, run_id=run)
    out['bestfit_val_sim_rms_m'] = float(bestfit)
    out['n_updates'] = int(getattr(fit_sys, 'batch_counter', -1))
    out['ann_changed'] = ann_changed(fit_sys, ann0)
    print(f'[augment] training done ({time.time() - t0:.0f} s), bestfit {bestfit:.4e} m, '
          f'rss {rss_mb():.0f} MB, peak {peak_rss_mb():.0f} MB', flush=True)
    bank = fit_sys.simulator.bank
    output_only_eval(fit_sys, data.val_list, data.val_names, bank, 'validation', out)
    output_only_eval(fit_sys, data.test_list, data.test_names, bank, 'test', out)
    out['peak_rss_mb'] = peak_rss_mb()
    out['wall_s'] = time.time() - t0
    json.dump(out, open(os.path.join(sdir, 'augment_metrics.json'), 'w'), indent=1)
    print(f'[augment] done: peak rss {out["peak_rss_mb"]:.0f} MB, {out["wall_s"]:.0f} s -> {sdir}')


if __name__ == '__main__':
    main()
