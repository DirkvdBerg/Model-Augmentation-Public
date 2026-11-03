"""Convert a run folder written by the mainline code (the thesis runs) or by an earlier Clean/ into a run folder that
Clean/ reads (D-284 amendment 10, R1 and R3): Clean/ owns its config.json schema and checkpoint state, and
verification loads the thesis folders through this converter only.

    python -u clean-up-verification/convert_run.py SRC_RUN_DIR DST_RUN_DIR

- config.json: a thesis folder's axes come from its mainline keys (OBC, COMBO_INIT_DETUNE, NX_ANN, PS2, MODE), and
  every other mainline key must equal the value Clean/ computes with (THESIS_SETTINGS below, the check
  RunConfig.from_config_json made before amendment 10); a folder with other settings is refused. DST gets the
  config.json Clean/ writes (config_json_dict), plus run_id, data_dir and RESUMES when the
  source has them, and CONVERTED_FROM (the source folder). A Clean folder must pass RunConfig.from_config_json.
- checkpoint pairs gantry_ckpt_*.pt / .npz: the state_dict keys of DROPPED_STATE_KEYS leave hfn (R3: unread state
  Clean/ no longer registers); the .npz keeps the keys Clean/'s pair holds, as plain arrays (R19: Clean/ reads it
  with allow_pickle=False).
- val_hist.npz and loss.npz are copied when present. A thesis deepSI dump (*.pth) pickles mainline classes, so it is
  converted in two steps: the harness's extract_dump.py (mainline root) writes its state as plain tensors, and
  convert_dump() here (Clean/ root) writes a dump that Clean/'s resolve_checkpoint reads (D-295).
DST must not exist; SRC is only read.
"""
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# R3: hfn state_dict keys of the thesis checkpoints that Clean/ does not register (last name component; they belong
# to the baseline block connected_blocks.0): the 13 nominal matrices at theta^0 and the parameter-prior scale
DROPPED_STATE_KEYS = ('mh', 'alpha', 'beta', 'gamma_', 'N0', 'N1', 'N2', 'Ax', 'Bw', 'Bu', 'A_combined', 'K_mat',
                      'C_mat', 'Lambda_combo')
# hfn state_dict keys of the OBC correction (2026-11-03): mainline's row selection E, which Clean/ replaces by zero
# padding, and the two buffers Clean/ renamed (mainline name -> Clean/ name); the harness's hcommon holds the same
DROPPED_OBC_KEYS = ('obc_correction.E',)
RENAMED_STATE_KEYS = {'obc_correction.vbar': 'obc_correction.beta_lin',
                      'obc_correction.theta_frozen': 'obc_correction.theta_aux'}
# the keys of Clean/'s checkpoint pair .npz (training.save_checkpoint_pair)
PAIR_NPZ_KEYS = ('done_epochs', 'bestfit', 'lbfgs_outcome', 'resumed_at_epoch', 'ps2_summary')
# RunConfig fields a thesis folder records as its own run environment (the former config._ENV_KEYS)
_ENV_FIELDS = ('device', 'compile_mode', 'truncate_s', 'epochs', 'n_its', 'its_per_val', 'lbfgs_max_iter',
               'lbfgs_windows', 'baseline_form')
# config.json keys of the run environment and other per-run values, not compared
_NOT_COMPARED = ('DEVICE', 'COMPILE_MODE', 'TRUNCATE_S', 'EPOCHS', 'N_ITS', 'ITS_PER_VAL', 'LBFGS_MAX_ITER',
                 'LBFGS_INNER_ITER', 'LBFGS_WINDOWS', 'BASELINE_FORM', 'GIT', 'NF_PROBE_PRINT', 'START_PHASE')


def _y_decimation(cfg):
    """The mainline's Y_DECIMATION text; the thesis config.json records the anti-alias filter in it."""
    if cfg.noise == 'noisy':
        return 'r + zero-phase Butterworth %d at %.0f Hz on y - r, then [::%d]' % (cfg.aa_order, cfg.aa_fc_hz,
                                                                                   cfg.decimation)
    return 'point sampling [::%d]' % cfg.decimation


def thesis_settings(cfg):
    """The mainline config.json keys of a thesis run with the axes of cfg, as Clean/ computes with them (the keys and
    values RunConfig.config_json_dict wrote before amendment 10)."""
    from model_augmentation.fit_systems import lbfgs_polish as LP
    return json.loads(json.dumps(dict(
        MODE={'noisy': 'thesis_taf_lowpass_noise', 'noisefree': 'thesis_taf_noisefree'}[cfg.noise],
        ENCODER_INIT='data_lsq', ANN_ACTIVATION='tanh', ANN_ARCH='mlp', G_INIT='live',
        LIVE_G_WEIGHT_INIT='kaiming_uniform_a_sqrt5', WA_ENCODER_INIT='xavier_uniform_gain1',
        FS_NEW=cfg.fs_new, D=cfg.decimation, FS_ORIG=cfg.fs_orig, SEED=cfg.seed, SNR=None, JOINT_ESTIMATION=True,
        PHYSICS_PARAMETERIZATION='reduced', PARAM_RMSE_BASELINE=0.01, PARAM_INIT_DETUNE=None,
        COMBO_INIT_DETUNE=cfg.combo_init_detune, ORTH_BETA=4.66e-4, ORTH_POINT_STRIDE=100, ORTH_RANK_TOL=1e-12,
        CLOSED_LOOP=True, ADAM_EPS=cfg.adam_eps, ORTH=False, ANN_ROUTE_IX=list(range(cfg.nx_phys + cfg.n_a)),
        STRIDE=cfg.stride, NF_SECONDS=cfg.nf_seconds, USE_F64=True, SAVE_FLAG=True, NX_ANN=cfg.n_a,
        N_NODES_PER_LAYER=cfg.n_nodes_per_layer, N_HIDDEN_LAYERS=cfg.n_hidden_layers, UP_SAMPLE=cfg.up_sample,
        BATCH_SIZE=cfg.batch_size, LR=cfg.lr, NF=cfg.nf, NA_NB=cfg.na_nb, CHECKPOINT_CHUNK=0, LBFGS=True,
        LBFGS_HISTORY=LP.HISTORY_SIZE, LBFGS_CHUNK=None, LBFGS_TOL_GRAD=LP.TOL_GRAD,
        LBFGS_TOL_CHANGE=LP.TOL_CHANGE, LBFGS_FREEZE=[], LBFGS_TIME_BUDGET_S=None, LBFGS_METER_REL_TOL=0.001,
        PARAM_PRIOR=False, STATIC_PASS_BUFFERS=False, OBC=cfg.obc, OBC_SPACE='affine', OBC_REFRESH='epoch',
        OBC_REF_CHUNK=16384, OBC_RANK_RTOL=None, CONTROLLER_GAIN=False, PS2=cfg.ps2,
        Y_DECIMATION=_y_decimation(cfg), RECORD_TRIM=cfg.record_trim, LOSS_WEIGHTING='servo',
        VALIDATION_MEASURE='sim-NRMSe', LBFGS_METERS_GATE=False, OBC_REF_SUCCESSOR=True)))


def config_of(meta):
    """RunConfig of a source config.json: Clean/'s reader for a Clean folder, else the thesis axes checked against
    THESIS_SETTINGS. Settings a folder does not record (an earlier Clean/ or the thesis code held them fixed) take
    the thesis values of the entry script's SETTINGS (D-292)."""
    from gantry_interconnect_dynamic import SETTINGS
    from gantry_dynamic.config import RunConfig, config_json_dict
    # a Clean folder records ARM; a thesis folder records OBC instead
    if 'ARM' in meta:
        thesis = config_json_dict(RunConfig(**SETTINGS))
        return RunConfig.from_config_json({**{k: v for k, v in thesis.items() if k not in meta}, **meta})
    axes = dict(arm='obc' if meta['OBC'] else 'u', start='correct' if meta['COMBO_INIT_DETUNE'] is None else 'detuned',
                n_a=int(meta['NX_ANN']), ps2=bool(meta.get('PS2', False)),
                noise={'thesis_taf_lowpass_noise': 'noisy', 'thesis_taf_noisefree': 'noisefree'}[meta['MODE']])
    # the run environment of the thesis folder; every other setting is checked against THESIS_SETTINGS below
    env = {f: meta[f.upper()] for f in _ENV_FIELDS if f.upper() in meta}
    cfg = RunConfig(**dict(SETTINGS, **axes, **env))
    want = thesis_settings(cfg)
    bad = {k: (v, meta.get(k, '<missing>')) for k, v in want.items()
           if k not in _NOT_COMPARED and meta.get(k, '<missing>') != v}
    if bad:
        raise ValueError('the thesis config.json has settings Clean/ does not compute with (key: (Clean, file)): %s'
                         % bad)
    return cfg


def _plain(v):
    """A .npz value as a plain array: an object array is unwrapped to its item, then stored as a str."""
    a = np.asarray(v)
    if a.dtype == object:
        item = a.item()
        return np.array(item if isinstance(item, str) else json.dumps(item))
    return a


def convert_hfn_state(sd):
    """A thesis hfn state_dict as Clean/ registers it: without the DROPPED_STATE_KEYS and DROPPED_OBC_KEYS, the OBC
    buffers under their Clean/ names; returns (state, dropped keys)."""
    dropped = [k for k in sd if (k.startswith('connected_blocks.0.') and k.rsplit('.', 1)[1] in DROPPED_STATE_KEYS)
               or k in DROPPED_OBC_KEYS]
    return {RENAMED_STATE_KEYS.get(k, k): v for k, v in sd.items() if k not in dropped}, dropped


def convert_pair(src_base, dst_base):
    """One checkpoint pair <base>.pt / <base>.npz: hfn through convert_hfn_state, the .npz keeps PAIR_NPZ_KEYS as
    plain arrays; returns the dropped hfn keys."""
    import torch
    ckpt = torch.load(src_base + '.pt', map_location='cpu')
    ckpt['hfn'], dropped = convert_hfn_state(ckpt['hfn'])
    torch.save(ckpt, dst_base + '.pt')
    with np.load(src_base + '.npz', allow_pickle=True) as z:
        meta = {k: _plain(z[k]) for k in PAIR_NPZ_KEYS if k in z.files}
    np.savez(dst_base + '.npz', **meta)
    return dropped


class SavedState:
    """A saved state_dict behind the one method Clean/'s resolve_checkpoint calls on a dump's hfn, encoder and
    optimizer entries."""

    def __init__(self, sd):
        self.sd = sd

    def state_dict(self):
        return self.sd


def convert_dump(plain_pt, dst_pth):
    """A thesis deepSI dump, extracted by extract_dump.py into plain_pt, as a dump at dst_pth that Clean/ resumes from:
    hfn through convert_hfn_state; encoder, Adam state and progress (epochs_before, epoch_counter, bestfit,
    ps2_opening) unchanged. Reading dst_pth needs this module importable (the harness puts it on sys.path)."""
    import torch
    if os.path.exists(dst_pth):
        raise FileExistsError(dst_pth)
    d = torch.load(plain_pt, map_location='cpu', weights_only=False)
    hfn, dropped = convert_hfn_state(d['hfn'])
    out = {k: d[k] for k in ('epochs_before', 'epoch_counter', 'bestfit', 'ps2_opening') if k in d}
    out.update(hfn=SavedState(hfn), encoder=SavedState(d['encoder']), optimizer=SavedState(d['optimizer']))
    os.makedirs(os.path.dirname(os.path.abspath(dst_pth)), exist_ok=True)
    torch.save(out, dst_pth)
    return dropped


def convert_run(src, dst):
    """Convert the run folder src into dst (created here); returns a report dict."""
    src, dst = os.path.abspath(src), os.path.abspath(dst)
    if os.path.exists(dst):
        raise FileExistsError('%s exists; the converter never writes into an existing folder' % dst)
    from gantry_dynamic.config import config_json_dict
    with open(os.path.join(src, 'config.json'), encoding='utf-8') as fh:
        meta = json.load(fh)
    cfg = config_of(meta)
    os.makedirs(dst)
    out = config_json_dict(cfg)
    out.update({k: meta[k] for k in ('run_id', 'data_dir', 'RESUMES') if k in meta})
    out['CONVERTED_FROM'] = src
    with open(os.path.join(dst, 'config.json'), 'w') as fh:
        json.dump(out, fh, indent=2)
    report = dict(src=src, dst=dst, source_code='clean' if 'ARM' in meta else 'mainline', pairs={}, copied=[])
    for f in sorted(os.listdir(src)):
        if f.startswith('gantry_ckpt_') and f.endswith('.pt') and os.path.exists(os.path.join(src, f[:-3] + '.npz')):
            report['pairs'][f[:-3]] = convert_pair(os.path.join(src, f[:-3]), os.path.join(dst, f[:-3]))
        elif f in ('val_hist.npz', 'loss.npz'):
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
            report['copied'].append(f)
    return report


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    # as Clean/gantry_interconnect_dynamic.py runs: Clean/ first on sys.path (an importer of this module, the
    # harness, has put its code root there already)
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'Clean'))
    rep = convert_run(sys.argv[1], sys.argv[2])
    print(json.dumps(rep, indent=1))
    print('CONVERT_RUN DONE %s -> %s' % (rep['src'], rep['dst']))
