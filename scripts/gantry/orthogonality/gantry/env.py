"""Build the production gantry predictor from a saved checkpoint, and select fixed windows.

Specification Sect. 1.2 step 1 and Sect. 13 step 1: "select the actual checkpoint/configuration
and verify dimensions, routes, scoring, controller and anchor flags; save the manifest before
any run." Nothing here interprets geometry; it only assembles the environment and records what
it assembled.

TWO CONVERSIONS HAPPEN HERE, and both are OUTPUT-PRESERVING BY CONSTRUCTION AND CHECKED, never
assumed:

  1. PHYSICAL BLOCK.  The selected checkpoints were trained with `JOINT_ESTIMATION=false`, i.e.
     with `Gantry_State_Block`, whose physical parameters are buffers and carry no derivative.
     The geometry needs `S = D_lambda p`, so the block is rebuilt as
     `Parameterized_Gantry_State_Block` with `log_params = 0`, at which point
     `_recover_params()` returns `params_init`, the nominal `gantry_ss` values. The rebuilt
     block must reproduce the original block's predictions; `check_physical_conversion` asserts
     it rather than trusting the argument.

  2. ENCODER.  `linear_encoder_init_aug` shares ONE nonlinear correction net between the
     physical and the latent initial-state rows, so `xi_p` and `eta_a` are not disjoint by
     tensor identity. `encoder_split.split_encoder` performs the specification's default
     migration (Sect. 4.2). It is applied to a DISPOSABLE COPY here; parity is checked.

DTYPE. The checkpoints are float32. Geometry is built in float64 on a widened copy, which is
exact for the parameter VALUES (float32 -> float64 loses nothing) but recomputes the block's
own constants at float64, so the widened model is NOT bit-identical to the production one. Both
are constructed and the parity gap is measured rather than assumed away.
"""
__project_origin__ = "added"

import json
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
_SCRIPTS_GANTRY = os.path.join(REPO, 'scripts', 'gantry')
if _SCRIPTS_GANTRY not in sys.path:
    sys.path.insert(0, _SCRIPTS_GANTRY)

from dataclasses import replace                                          # noqa: E402

from gantry_dynamic.config import RunConfig                              # noqa: E402
from gantry_dynamic.data import (load_datasets, compute_normalization,   # noqa: E402
                                 TRAIN_FILES, VAL_FILES)
from gantry_dynamic.model import build_model, get_encoder_dims           # noqa: E402
from gantry_dynamic.training import resolve_checkpoint                   # noqa: E402
from model_augmentation.fit_systems.blocks import Static_ANN_Block       # noqa: E402


# --------------------------------------------------------------------------- configuration
_CFG_MAP = {
    'mode': 'MODE', 'encoder_init': 'ENCODER_INIT', 'ann_activation': 'ANN_ACTIVATION',
    'fs_new': 'FS_NEW', 'fs_orig': 'FS_ORIG', 'seed': 'SEED', 'snr': 'SNR',
    'closed_loop': 'CLOSED_LOOP', 'stride': 'STRIDE', 'burn_in': 'BURN_IN',
    'nf_seconds': 'NF_SECONDS', 'nx_ann': 'NX_ANN', 'n_nodes_per_layer': 'N_NODES_PER_LAYER',
    'n_hidden_layers': 'N_HIDDEN_LAYERS', 'up_sample': 'UP_SAMPLE',
    'param_rmse_baseline': 'PARAM_RMSE_BASELINE',
}


def config_from_run(run_dir, use_f64=False, joint_estimation=True, device='cpu',
                    compile_mode=None):
    """A RunConfig reproducing the recorded run, with the deliberate deviations named.

    DEVIATIONS, all recorded in the manifest:
      device=<argument>                 CHANGED 2026-09-08: this was hard-coded to 'cpu',
                                        which made the reviewer's point that a GPU retry is not
                                        plug-and-play literally true. It is now the caller's
                                        choice and is recorded in the manifest.
      compile_mode=<argument>           CHANGED 2026-09-08: was hard-coded to None, for the same
                                        reason `device` was, and with the same consequence. The
                                        DEFAULT IS STILL None, so every verification stage runs
                                        eager unless a caller asks otherwise: a compiled rollout
                                        hides the per-operation timings this suite records, and
                                        a compiled parity check is a separate, explicit stage
                                        (`run_gpu_ladder.py`) rather than a silent default.
                                        Compilation is only wired for CUDA, and RunConfig's own
                                        guard rejects the combination on any other device.
      joint_estimation=True             makes the 14 physical log-coordinates differentiable.
                                        The run itself had them fixed; see the module header.
      param_init_detune=None            the reference is the nominal parameter vector.
    """
    with open(os.path.join(run_dir, 'config.json')) as f:
        rec = json.load(f)
    kw = {}
    for field, key in _CFG_MAP.items():
        if key in rec:
            kw[field] = rec[key]
    if 'ANN_ROUTE_IX' in rec:
        kw['ann_route_ix'] = tuple(rec['ANN_ROUTE_IX'])
    if 'NA_NB' in rec:
        kw['na_nb_override'] = rec['NA_NB']
    if 'NF' in rec:
        kw['nf_override'] = rec['NF']
    kw.update(device=device, compile_mode=compile_mode, use_f64=use_f64, save_flag=False,
              joint_estimation=joint_estimation, param_init_detune=None,
              orth=False, lbfgs=False, nf_probe_print=False)
    return RunConfig(**kw), rec


# --------------------------------------------------------------------------- data cache
def load_data_cached(cfg, cache_dir=None):
    """Datasets and normalisation, cached by (mode, fs_new, snr, dtype) in the scratchpad.

    The normalisation statistics are computed over EVERY training record, so the full set has to
    be loaded at least once; the cache exists so that repeated verification runs do not pay it
    again. The cache stores u/y arrays only, never a model.
    """
    data = load_datasets(cfg)
    norm = compute_normalization(cfg, data)
    return data, norm


# --------------------------------------------------------------------------- environment
class GantryEnv:
    """Everything a verification stage needs, assembled once and described in a manifest."""

    def __init__(self, fit_sys, cfg, data, norm, run_dir, ckpt_base, rec_cfg, dtype):
        self.fit_sys, self.cfg, self.data, self.norm = fit_sys, cfg, data, norm
        self.run_dir, self.ckpt_base, self.rec_cfg, self.dtype = run_dir, ckpt_base, rec_cfg, dtype
        self.na, self.nb, self.na_right, self.nb_right = get_encoder_dims(cfg.hp, cfg)
        self.device = torch.device(cfg.device)

    # -- structural description, spec Sect. 9.1 item 1 --------------------------------
    @property
    def ann_block(self):
        return next(m for m in self.fit_sys.hfn.connected_blocks
                    if isinstance(m, Static_ANN_Block))

    @property
    def phys_block(self):
        from model_augmentation.fit_systems.blocks import Gantry_State_Block
        return next(m for m in self.fit_sys.hfn.connected_blocks
                    if isinstance(m, Gantry_State_Block))

    def manifest(self):
        cfg = self.cfg
        route_ix = np.asarray(cfg.ann_route_ix)
        nxd = cfg.nx_phys + cfg.nx_ann
        ann = self.ann_block
        bank = getattr(self.fit_sys.simulator, 'bank', None)
        m = dict(
            run_dir=self.run_dir, checkpoint=self.ckpt_base,
            recorded_config={k: self.rec_cfg.get(v) for k, v in _CFG_MAP.items()},
            dtype=str(self.dtype), device=str(cfg.device), compile_mode=cfg.compile_mode,
            nx_phys=cfg.nx_phys, nx_ann=cfg.nx_ann, nxd=nxd, nu=cfg.nu, ny=cfg.ny,
            nf=cfg.nf, burn_in=cfg.burn_in, T_scored=cfg.nf - cfg.burn_in,
            na=self.na, nb=self.nb, na_right=self.na_right, nb_right=self.nb_right,
            stride=cfg.stride, up_sample=cfg.up_sample, ts_new=cfg.ts_new,
            ann_route_ix=[int(v) for v in route_ix],
            ann_n_outputs=int(ann.nw), ann_n_inputs=int(ann.nz),
            # ROUTE_COLS ARE ANN-OUTPUT INDICES, NOT STATE ROWS (spec Sect. 4.3).
            route_cols_physical=[int(j) for j in range(len(route_ix)) if route_ix[j] < cfg.nx_phys],
            route_cols_latent=[int(j) for j in range(len(route_ix)) if route_ix[j] >= cfg.nx_phys],
            joint_estimation=cfg.joint_estimation,
            physical_block=type(self.phys_block).__name__,
            param_anchor_flag_loss_reg=getattr(self.phys_block, 'flag_loss_reg', None),
            closed_loop=cfg.closed_loop,
            controller_n_states=(int(bank.nc) if bank is not None else None),
            controller_n_rows=(int(bank.n_controllers) if bank is not None else None),
            orth_pointwise_attached=getattr(self.fit_sys, 'orth_penalty', None) is not None,
            checkpoint_chunk=cfg.checkpoint_chunk,
            loss='torch.nn.functional.mse_loss over the scored slice (plain MSE, N = n_W*T*ny)',
        )
        return m


def load_env(run_dir, ckpt_name=None, use_f64=True, joint_estimation=True, verbose=True,
             data_norm=None, device='cpu', compile_mode=None):
    """Build the production fit system at a checkpoint. Returns a GantryEnv.

    `joint_estimation=True` swaps in `Parameterized_Gantry_State_Block`; the checkpoint has no
    `log_params`, so that key is EXPECTED to be missing and every other key must match exactly.
    Anything else is a hard failure, not a warning.
    """
    cfg, rec = config_from_run(run_dir, use_f64=use_f64, joint_estimation=joint_estimation,
                               device=device, compile_mode=compile_mode)
    if ckpt_name is None:
        cands = [f for f in os.listdir(run_dir)
                 if f.startswith('gantry_ckpt_') and f.endswith('.pt')
                 and 'pre_lbfgs' not in f]
        if not cands:
            raise FileNotFoundError(f'no gantry_ckpt_*.pt in {run_dir}')
        ckpt_name = sorted(cands)[0]
    ckpt_base = os.path.join(run_dir, ckpt_name[:-3] if ckpt_name.endswith('.pt') else ckpt_name)

    if data_norm is None:
        data, norm = load_data_cached(cfg)
    else:
        data, norm = data_norm

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    fit_sys = build_model(cfg.hp, cfg, data, norm)

    if cfg.closed_loop:
        from gantry_dynamic.controller import build_closed_loop
        fit_sys.simulator = build_closed_loop(
            fit_sys, norm, cfg, train_files=TRAIN_FILES, val_files=VAL_FILES,
            val_data=data.val_ckpt_data, verbose=verbose)

    hfn_sd, enc_sd, _optim_sd, _meta, fmt = resolve_checkpoint(ckpt_base)
    target = fit_sys.hfn.state_dict()
    missing = [k for k in target if k not in hfn_sd]
    extra = [k for k in hfn_sd if k not in target]
    allowed_missing = {'connected_blocks.0.log_params', 'connected_blocks.0.params_init',
                       'connected_blocks.0.Lambda', 'connected_blocks.0.Lb'}
    bad_missing = [k for k in missing if k.rsplit('.', 1)[-1]
                   not in ('log_params', 'params_init', 'Lambda', 'Lb')]
    if bad_missing or extra:
        raise RuntimeError(
            'checkpoint/hfn key mismatch beyond the physical-block conversion. missing=%s '
            'extra=%s' % (bad_missing[:8], extra[:8]))
    # widen to the model dtype: exact for float32 -> float64
    want = next(fit_sys.hfn.parameters()).dtype
    hfn_sd = {k: (v.to(want) if torch.is_tensor(v) and v.is_floating_point() else v)
              for k, v in hfn_sd.items()}
    enc_sd = {k: (v.to(want) if torch.is_tensor(v) and v.is_floating_point() else v)
              for k, v in enc_sd.items()}
    fit_sys.hfn.load_state_dict(hfn_sd, strict=False)
    fit_sys.encoder.load_state_dict(enc_sd, strict=True)
    fit_sys.hfn.eval()
    fit_sys.encoder.eval()
    # BUILD ON CPU, THEN MOVE. `build_model` constructs on the CPU deliberately (D-169: the
    # encoder-init capture and the diagnostics that run between build and fit are CPU code), and
    # `train_model` moves around `fit()` only. The same pairing is kept here rather than a second
    # convention. The controller bank is NOT moved: `closed_loop_rollout` re-homes it to the
    # data's device on first use, which is the only thing that survives deepSI's per-validation
    # device flip.
    #
    # `.cuda()`, NOT `.to()`. deepSI's `System_torch` defines `cuda()`, `cpu()` and `to_device()`
    # (fit_system.py:534-538) and NO `to()`, so `fit_sys.to(device)` raised AttributeError the
    # first time this ran on a GPU (job 82430). The comment above already said the production
    # pairing was being kept; the code did not. `model.py:321` is that production site.
    if device != 'cpu':
        assert str(device).startswith('cuda'), (
            'load_env moves the model with the deepSI cuda()/cpu() pair, which is what '
            'production uses (model.py:321). '
            'device=%r is neither cpu nor cuda, and RunConfig refuses a '
            'device index anyway.' % (device,))
        fit_sys.cuda()
    if verbose:
        print(f'[env] checkpoint {os.path.basename(ckpt_base)} format={fmt} dtype->{want} '
              f'({len(missing)} conversion keys absent from the checkpoint: '
              f'{sorted(set(k.rsplit(".",1)[-1] for k in missing))})')
    return GantryEnv(fit_sys, cfg, data, norm, run_dir, ckpt_base, rec, want)


# ------------------------------------------------------------------ production window set
def production_windows(env, records, per_record, margin_s=1.0, seed=20260908, verbose=True):
    """Fixed regularisation windows, built by the PRODUCTION window constructor.

    `SSE_Interconnect.make_training_data` is the only thing that knows deepSI's window index
    convention, the right-hand encoder extension and the controller-row bookkeeping, so it is
    what builds the windows here. To avoid materialising all ~66600 training windows, it is
    called on a System_data_list of CONTIGUOUS SLICES of the selected records: slicing changes
    WHICH windows exist, never the content of a window that lies wholly inside the slice, and
    `check_window_content_parity` verifies exactly that against the untouched record.

    Returns a dict with the stacked arrays and the provenance of every window.
    """
    import deepSI
    cfg = env.cfg
    rng = np.random.default_rng(seed)
    k0 = max(env.na, env.nb)
    k0r = max(cfg.nf, env.na_right, env.nb_right)
    span = k0 + k0r + cfg.stride * (per_record - 1) + 1

    slices, prov = [], []
    for r in records:
        sd = env.data.train_list[r]
        n = len(sd.u)
        lo = int(rng.integers(int(margin_s * cfg.fs_new_hz),
                              max(int(margin_s * cfg.fs_new_hz) + 1, n - span - 1)))
        sl = slice(lo, lo + span)
        slices.append(deepSI.System_data(u=sd.u[sl].copy(), y=sd.y[sl].copy(), dt=sd.dt))
        prov.append(dict(record_index=int(r), record=TRAIN_FILES[r], slice_start=int(lo),
                         slice_stop=int(lo + span)))
    sdl = deepSI.System_data_list(slices)

    # The production call, including the closed-loop seam's ctrl_ix augmentation.
    #
    # `norm.transform` FIRST, exactly as fit() does (interconnect.py:857): deepSI's
    # `make_training_data` asserts `sys_data.normed`, and skipping the transform would build
    # windows in SI units and feed them to a model whose every buffer is normalised.
    #
    # The indexer's per-record controller rows are narrowed to the SELECTED records for the
    # duration of the call. It requires one row per record in list order and would otherwise
    # refuse the 4-record list, which is the correct behaviour: that check exists because a
    # mismatch there trains every window inside the wrong loop and still looks plausible. Its
    # own `verify_by_content` then compares each selected record's first and last window against
    # the raw slice at the derived offset, which is exactly the window-content parity this
    # slicing needs and is why it is not re-implemented here.
    idx = getattr(env.fit_sys.simulator, 'indexer', None)
    saved_rows = None if idx is None else list(idx.record_rows)
    try:
        if idx is not None:
            idx.record_rows = [saved_rows[r] for r in records]
        arrays = env.fit_sys.make_training_data(env.fit_sys.norm.transform(sdl),
                                                nf=cfg.nf, stride=cfg.stride)
    finally:
        if idx is not None:
            idx.record_rows = saved_rows
    names = ['uhist', 'yhist', 'ufuture', 'yfuture'] + (['ctrl_ix'] if len(arrays) > 4 else [])
    out = {n: np.asarray(a) for n, a in zip(names, arrays)}
    counts = getattr(env.fit_sys.simulator, 'last_window_counts', None)
    if verbose:
        print(f'[windows] {out["ufuture"].shape[0]} windows from {len(records)} records '
              f'(per-record counts {counts}), nf={cfg.nf}, stride={cfg.stride}')
    out['provenance'] = prov
    out['counts'] = counts
    out['n_windows'] = int(out['ufuture'].shape[0])
    return out


def to_tensors(win, dtype, device='cpu'):
    """Window arrays as tensors on the target device.

    The controller index stays an integer tensor on the same device: `closed_loop_rollout` calls
    `.long()` on it and indexes the bank with it, and a CPU index against CUDA bank tensors
    raises rather than silently working.
    """
    t = {}
    for k in ('uhist', 'yhist', 'ufuture', 'yfuture'):
        t[k] = torch.as_tensor(win[k], dtype=dtype, device=device)
    if 'ctrl_ix' in win:
        t['ctrl_ix'] = torch.as_tensor(win['ctrl_ix'], dtype=torch.long, device=device)
    return t


def select(win_t, ix):
    """A row-subset of a window batch, keeping every array aligned."""
    out = {k: v[ix] for k, v in win_t.items()}
    return out
